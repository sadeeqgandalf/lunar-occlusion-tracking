import { Mat, wrapAngle } from '../core/linalg.js';
import { Filter, resid, wrapState } from './base.js';

/**
 * Particle filter (Probabilistic Robotics Table 4.3): sample from the motion model, weight by p(z|x),
 * resample with the low-variance sampler (Table 4.4), with the Augmented-MCL recovery of Table 8.3 (random
 * particles injected when the fast/slow likelihood averages diverge). Standard-practice deviations: resample
 * only when N_eff < N/2, a flat outlier floor in the likelihood, and a small roughening jitter.
 */
export class ParticleFilter extends Filter {
  static id = 'pf';
  static label = 'Particle Filter';
  static color = '#ff6bd6';
  static blurb = 'Non-parametric belief (Thrun Table 4.3) with Augmented-MCL recovery (Table 8.3). Consistent and robust, at roughly 200× the CPU of an EKF.';
  static paramSpec = {
    qScale: { label: 'Process noise ×', min: 0.2, max: 60, step: 0.1, value: 4 },
    rScale: { label: 'Measurement σ ×', min: 0.3, max: 8, step: 0.1, value: 2 },
  };

  constructor(model, sensors, params, rng, N = 1500) {
    super(model, sensors, params, rng);
    this.N = N;
    this.nx = model.nx;
    this.p = new Float64Array(N * this.nx);
    this.w = new Float64Array(N).fill(1 / N);
    this._cache = null;
  }

  init(x0, P0) {
    const L = P0.chol(), N = this.N, n = this.nx;
    for (let i = 0; i < N; i++) {
      const z = Array.from({ length: n }, () => this.rng.randn());
      const x = Array.from(x0);
      for (let r = 0; r < n; r++) for (let c = 0; c <= r; c++) x[r] += L.get(r, c) * z[c];
      wrapState(this.model, x);
      for (let r = 0; r < n; r++) this.p[i * n + r] = x[r];
    }
    this.w.fill(1 / N);
    this.wSlow = 0; this.wFast = 0;
    this.sd0 = Array.from({ length: n }, (_, r) => Math.sqrt(P0.get(r, r)));
    this._cache = null;
  }

  /** sample_motion_model (Table 5.3): draw w ~ N(0, qScale W), push each particle through g(u, w). */
  predict(u, dt) {
    const M = this.model, n = this.nx, rng = this.rng;
    const Lw = M.W(u, dt, this.sensors).scale(this.params.qScale).chol();
    const x = new Array(n);
    for (let i = 0; i < this.N; i++) {
      const o = i * n;
      for (let r = 0; r < n; r++) x[r] = this.p[o + r];
      const z = Array.from({ length: M.nw }, () => rng.randn()), w = new Array(M.nw).fill(0);
      for (let r = 0; r < M.nw; r++) for (let c = 0; c <= r; c++) w[r] += Lw.get(r, c) * z[c];
      const y = wrapState(M, M.f(x, u, w, dt, this.sensors));
      for (let r = 0; r < n; r++) this.p[o + r] = y[r];
    }
    this._cache = null;
  }

  update(m) {
    const M = this.model, spec = M.meas[m.kind];
    if (!spec) return { accepted: false };
    const N = this.N, n = this.nx, eps = 0.02, Ri = spec.R(m, this.params.rScale).inv(), k = spec.dim, z = spec.z(m);
    const x = new Array(n);
    let sum = 0;
    for (let i = 0; i < N; i++) {
      for (let r = 0; r < n; r++) x[r] = this.p[i * n + r];
      const e = resid(spec.angleDims, z, spec.h(x, m));
      let d2 = 0;
      for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) d2 += e[a] * Ri.get(a, b) * e[b];
      this.w[i] *= (1 - eps) * Math.exp(-0.5 * d2) + eps;
      sum += this.w[i];
    }
    // Augmented MCL (Table 8.3): `sum` is the mean measurement likelihood since weights were normalised to 1.
    this.wSlow = this.wSlow ? this.wSlow + 0.05 * (sum - this.wSlow) : sum;
    this.wFast = this.wFast ? this.wFast + 0.5 * (sum - this.wFast) : sum;
    this.pInject = Math.min(0.3, Math.max(0, 1 - this.wFast / (this.wSlow || 1)));
    if (!(sum > 1e-300)) this.w.fill(1 / N);
    else for (let i = 0; i < N; i++) this.w[i] /= sum;
    let n2 = 0;
    for (let i = 0; i < N; i++) n2 += this.w[i] * this.w[i];
    if (1 / n2 < N / 2 || this.pInject > 0.05) this._resample();
    this._cache = null;
    return { accepted: true, nis: NaN };
  }

  /** Low-variance sampler (Table 4.4). */
  _resample() {
    const N = this.N, n = this.nx, np = new Float64Array(N * n), jit = this._jitter();
    // random particles ~ N(mean, (3 sd)^2), sd capped at 30x the prior sd so repeated injections cannot blow the cloud up
    const { m, P } = this._moments(), sd = Array.from({ length: n }, (_, r) => Math.min(3 * Math.sqrt(Math.max(P.get(r, r), 0)), 30 * this.sd0[r]));
    let u = this.rng.next() / N, c = this.w[0], j = 0;
    for (let i = 0; i < N; i++) {
      if (this.rng.next() < this.pInject) {
        const z = Array.from({ length: n }, () => this.rng.randn()), x = m.slice();
        for (let r = 0; r < n; r++) x[r] += sd[r] * z[r];
        wrapState(this.model, x);
        for (let r = 0; r < n; r++) np[i * n + r] = x[r];
        continue;
      }
      const t = u + i / N;
      while (t > c && j < N - 1) { j++; c += this.w[j]; }
      for (let r = 0; r < n; r++) np[i * n + r] = this.p[j * n + r] + jit[r] * this.rng.randn();
    }
    this.p = np;
    this.w.fill(1 / N);
    this._cache = null;
  }

  /** Roughening scale per state dim: K * spread * N^(-1/n) (Gordon et al.), K small. */
  _jitter() {
    const { P } = this._moments(), n = this.nx, K = 0.1;
    return Array.from({ length: n }, (_, r) => K * Math.sqrt(Math.max(P.get(r, r), 0)) * Math.pow(this.N, -1 / n));
  }

  _moments() {
    if (this._cache) return this._cache;
    const M = this.model, N = this.N, n = this.nx, m = new Array(n).fill(0);
    for (let i = 0; i < N; i++) for (let r = 0; r < n; r++) m[r] += this.w[i] * this.p[i * n + r];
    for (const r of M.angleStates) {
      let s = 0, c = 0;
      for (let i = 0; i < N; i++) { s += this.w[i] * Math.sin(this.p[i * n + r]); c += this.w[i] * Math.cos(this.p[i * n + r]); }
      m[r] = Math.atan2(s, c);
    }
    const P = Mat.zeros(n, n), x = new Array(n);
    for (let i = 0; i < N; i++) {
      for (let r = 0; r < n; r++) x[r] = this.p[i * n + r];
      const d = resid(M.angleStates, x, m);
      for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) P.d[a * n + b] += this.w[i] * d[a] * d[b];
    }
    return (this._cache = { m, P });
  }

  getState() { return this._moments().m.slice(); }
  getCov() { return this._moments().P; }
  getParticles() { return this.p; }
}
