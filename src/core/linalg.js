// Minimal dense linear algebra for small state vectors (n <= ~10).
export const TAU = Math.PI * 2;
export const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
export function wrapAngle(a) {
  a = (a + Math.PI) % TAU;
  if (a < 0) a += TAU;
  return a - Math.PI;
}

export class Mat {
  constructor(r, c, d) { this.r = r; this.c = c; this.d = d || new Float64Array(r * c); }
  static zeros(r, c) { return new Mat(r, c); }
  static eye(n) { const m = new Mat(n, n); for (let i = 0; i < n; i++) m.d[i * n + i] = 1; return m; }
  static diag(v) { const n = v.length, m = new Mat(n, n); for (let i = 0; i < n; i++) m.d[i * n + i] = v[i]; return m; }
  static from(rows) {
    const r = rows.length, c = rows[0].length, m = new Mat(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) m.d[i * c + j] = rows[i][j];
    return m;
  }
  static coerce(x) { return x instanceof Mat ? x : Mat.from(x); }
  get(i, j) { return this.d[i * this.c + j]; }
  set(i, j, v) { this.d[i * this.c + j] = v; return this; }
  clone() { return new Mat(this.r, this.c, Float64Array.from(this.d)); }
  add(b) { const o = this.clone(); for (let i = 0; i < o.d.length; i++) o.d[i] += b.d[i]; return o; }
  sub(b) { const o = this.clone(); for (let i = 0; i < o.d.length; i++) o.d[i] -= b.d[i]; return o; }
  scale(s) { const o = this.clone(); for (let i = 0; i < o.d.length; i++) o.d[i] *= s; return o; }
  mul(b) {
    const { r, c } = this, n = b.c, o = new Mat(r, n);
    for (let i = 0; i < r; i++)
      for (let k = 0; k < c; k++) {
        const a = this.d[i * c + k];
        if (a === 0) continue;
        for (let j = 0; j < n; j++) o.d[i * n + j] += a * b.d[k * n + j];
      }
    return o;
  }
  T() {
    const o = new Mat(this.c, this.r);
    for (let i = 0; i < this.r; i++) for (let j = 0; j < this.c; j++) o.d[j * this.r + i] = this.d[i * this.c + j];
    return o;
  }
  /** Force symmetry (guards covariance against round-off). */
  sym() {
    const n = this.r, o = this.clone();
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const v = 0.5 * (this.d[i * n + j] + this.d[j * n + i]);
      o.d[i * n + j] = v; o.d[j * n + i] = v;
    }
    return o;
  }
  sub3() { // top-left 3x3 block
    const o = new Mat(3, 3);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) o.d[i * 3 + j] = this.d[i * this.c + j];
    return o;
  }
  /** Gauss-Jordan inverse with partial pivoting. */
  inv() {
    const n = this.r;
    if (n !== this.c) throw new Error('inv: not square');
    const a = this.clone().d, b = Mat.eye(n).d;
    for (let col = 0; col < n; col++) {
      let piv = col, best = Math.abs(a[col * n + col]);
      for (let r = col + 1; r < n; r++) { const v = Math.abs(a[r * n + col]); if (v > best) { best = v; piv = r; } }
      if (best < 1e-14) throw new Error('inv: singular matrix');
      if (piv !== col) for (let k = 0; k < n; k++) {
        [a[col * n + k], a[piv * n + k]] = [a[piv * n + k], a[col * n + k]];
        [b[col * n + k], b[piv * n + k]] = [b[piv * n + k], b[col * n + k]];
      }
      const d = a[col * n + col];
      for (let k = 0; k < n; k++) { a[col * n + k] /= d; b[col * n + k] /= d; }
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = a[r * n + col];
        if (f === 0) continue;
        for (let k = 0; k < n; k++) { a[r * n + k] -= f * a[col * n + k]; b[r * n + k] -= f * b[col * n + k]; }
      }
    }
    return new Mat(n, n, b);
  }
  /** Lower Cholesky factor; adds growing diagonal jitter if P has lost positive-definiteness. */
  chol() {
    const n = this.r;
    let jitter = 0;
    for (let attempt = 0; attempt < 8; attempt++) {
      const L = new Mat(n, n);
      let ok = true;
      for (let i = 0; i < n && ok; i++)
        for (let j = 0; j <= i; j++) {
          let s = this.d[i * n + j] + (i === j ? jitter : 0);
          for (let k = 0; k < j; k++) s -= L.d[i * n + k] * L.d[j * n + k];
          if (i === j) { if (s <= 1e-14) { ok = false; break; } L.d[i * n + i] = Math.sqrt(s); }
          else L.d[i * n + j] = s / L.d[j * n + j];
        }
      if (ok) return L;
      jitter = jitter ? jitter * 10 : 1e-10;
    }
    throw new Error('chol: matrix not positive definite');
  }
  rows() { const o = []; for (let i = 0; i < this.r; i++) o.push(Array.from(this.d.subarray(i * this.c, (i + 1) * this.c))); return o; }
}
