// The five learned policies, running the exact float16 weights exported by policy/train_eval.py.
// A port of policy/model.py, policy/diffusion.py, policy/flow.py, policy/cvae.py and policy/tokenizer.py (checked by test/policy.test.js).
import { H, fromUnit, toUnit } from './sim.js';

// ---------- weights ----------

const f16 = (h) => {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  return e === 0 ? s * 2 ** -14 * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + m / 1024);
};

/** Split one checkpoint's float16 blob into {bc, ar, dp, fp, cv} maps of name -> {data: Float32Array, shape}. */
export function unpack(manifest, buffer) {
  const raw = new Uint16Array(buffer), out = {};
  let at = 0;
  for (const [model, tensors] of Object.entries(manifest.tensors)) {
    out[model] = {};
    for (const [name, shape] of tensors) {
      const n = shape.reduce((a, b) => a * b, 1), data = new Float32Array(n);
      for (let i = 0; i < n; i++) data[i] = f16(raw[at + i]);
      out[model][name] = { data, shape }; at += n;
    }
  }
  if (at !== raw.length) throw new Error(`weights: read ${at} of ${raw.length} values`);
  return out;
}

// ---------- small dense ops (one vector at a time) ----------

/** y = W x + b with PyTorch's (out, in) weight layout. */
function linear(x, W, b) {
  const [n, m] = W.shape, y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = b ? b.data[i] : 0;
    for (let j = 0, r = i * m; j < m; j++) s += W.data[r + j] * x[j];
    y[i] = s;
  }
  return y;
}
const map = (x, f) => x.map(f);
const silu = (x) => map(x, (v) => v / (1 + Math.exp(-v)));
const gelu = (x) => map(x, (v) => 0.5 * v * (1 + Math.tanh(0.7978845608 * (v + 0.044715 * v ** 3))));
/** LayerNorm; without g and b it is the plain normalisation (elementwise_affine=False). */
function layerNorm(x, g, b) {
  const n = x.length, mu = x.reduce((a, v) => a + v, 0) / n;
  const sd = Math.sqrt(x.reduce((a, v) => a + (v - mu) ** 2, 0) / n + 1e-5);
  return map(x, (v, i) => ((v - mu) / sd) * (g ? g.data[i] : 1) + (b ? b.data[i] : 0));
}
const add = (a, b) => map(a, (v, i) => v + b[i]);
const normalize = (P, o) => Float32Array.from(o, (v, i) => (v - P['norm.mean'].data[i]) / P['norm.std'].data[i]);
export function softmax(z, temperature = 1) {
  const m = Math.max(...z), e = z.map((v) => Math.exp((v - m) / temperature)), s = e.reduce((a, v) => a + v, 0);
  return e.map((v) => v / s);
}
function sampleIndex(p, rng) {
  let u = rng.uniform();
  for (let i = 0; i < p.length; i++) if ((u -= p[i]) < 0) return i;
  return p.length - 1;
}

// ---------- tokenizer (DCT, FAST without BPE) ----------

export const K_FREQ = 4, VOCAB = 65, BOUND = 2.5;
export const DCT = Array.from({ length: H }, (_, k) =>
  Array.from({ length: H }, (_, t) => Math.cos((Math.PI * (t + 0.5) * k) / H) * Math.sqrt((k ? 2 : 1) / H)));

export function encode(chunk) {
  const u = chunk.map(toUnit), tokens = [];
  for (let k = 0; k < K_FREQ; k++)
    for (let d = 0; d < 2; d++) {
      const c = u.reduce((a, row, t) => a + DCT[k][t] * row[d], 0);
      tokens.push(Math.min(VOCAB - 1, Math.max(0, Math.round(((c + BOUND) / (2 * BOUND)) * (VOCAB - 1)))));
    }
  return tokens;
}

/** Coefficient value a token stands for. */
export const tokenValue = (tok) => (tok / (VOCAB - 1)) * 2 * BOUND - BOUND;

export function decode(tokens) {
  return Array.from({ length: H }, (_, t) =>
    fromUnit([0, 1].map((d) => {
      let s = 0;
      for (let k = 0; k < K_FREQ; k++) s += DCT[k][t] * tokenValue(tokens[2 * k + d]);
      return s;
    })));
}

// ---------- BC-MSE ----------

export function bcAct(P, o) {
  const h1 = silu(linear(normalize(P, o), P['net.0.weight'], P['net.0.bias']));
  const h2 = silu(linear(h1, P['net.2.weight'], P['net.2.bias']));
  const y = linear(h2, P['net.4.weight'], P['net.4.bias']);
  return Array.from({ length: H }, (_, t) => fromUnit([y[2 * t], y[2 * t + 1]]));
}

// ---------- token transformer ----------

function attention(P, pre, xs, causal = true) {
  const W = P[`${pre}.in_proj_weight`], B = P[`${pre}.in_proj_bias`], d = xs[0].length, heads = 4, dh = d / heads;
  const qkv = xs.map((x) => linear(x, W, B));
  return xs.map((_, i) => {
    const out = new Float32Array(d);
    for (let h = 0; h < heads; h++) {
      const o = h * dh, score = [];
      const last = causal ? i : xs.length - 1;   // causal: position i attends to 0..i
      for (let j = 0; j <= last; j++) {
        let s = 0;
        for (let c = 0; c < dh; c++) s += qkv[i][o + c] * qkv[j][d + o + c];
        score.push(s / Math.sqrt(dh));
      }
      const p = softmax(score);
      for (let j = 0; j <= last; j++) for (let c = 0; c < dh; c++) out[o + c] += p[j] * qkv[j][2 * d + o + c];
    }
    return linear(out, P[`${pre}.out_proj.weight`], P[`${pre}.out_proj.bias`]);
  });
}

/** Logits for every position of [o, prev...]: row i predicts token i. */
export function arLogits(P, o, prev) {
  const d = P['pos'].shape[1];
  let xs = [linear(silu(linear(normalize(P, o), P['obs_in.0.weight'], P['obs_in.0.bias'])), P['obs_in.2.weight'], P['obs_in.2.bias'])];
  for (const t of prev) xs.push(P['tok.weight'].data.slice(t * d, (t + 1) * d));
  xs = xs.map((x, i) => add(x, P['pos'].data.subarray(i * d, (i + 1) * d)));
  for (let b = 0; P[`blocks.${b}.ln1.weight`]; b++) {
    const p = `blocks.${b}`;
    const a = attention(P, `${p}.attn`, xs.map((x) => layerNorm(x, P[`${p}.ln1.weight`], P[`${p}.ln1.bias`])));
    xs = xs.map((x, i) => add(x, a[i]));
    xs = xs.map((x) => add(x, linear(gelu(linear(layerNorm(x, P[`${p}.ln2.weight`], P[`${p}.ln2.bias`]), P[`${p}.mlp.0.weight`], P[`${p}.mlp.0.bias`])), P[`${p}.mlp.2.weight`], P[`${p}.mlp.2.bias`])));
  }
  return xs.map((x) => Array.from(linear(layerNorm(x, P['ln.weight'], P['ln.bias']), P['head.weight'])));
}

/** Sample 2K tokens one at a time. Returns tokens and the probabilities each one was drawn from. */
export function arSample(P, o, temperature, rng) {
  const tokens = [], probs = [];
  for (let i = 0; i < 2 * K_FREQ; i++) {
    const p = softmax(arLogits(P, o, tokens)[i], temperature);
    probs.push(p); tokens.push(sampleIndex(p, rng));
  }
  return { tokens, probs, chunk: decode(tokens) };
}

// ---------- diffusion ----------

export function cosineSchedule(K) {
  const f = (t) => Math.cos((((t / K) + 0.008) / 1.008) * (Math.PI / 2)) ** 2;
  const betas = Array.from({ length: K }, (_, k) => Math.min(1 - f(k + 1) / f(k), 0.999));
  let a = 1;
  return { betas, abar: betas.map((b) => (a *= 1 - b)) };
}

function stepEmbedding(k, dim = 64) {
  const half = dim / 2, e = new Float32Array(dim);
  for (let i = 0; i < half; i++) { const a = k * Math.exp((-Math.log(10000) * i) / half); e[i] = Math.sin(a); e[half + i] = Math.cos(a); }
  return e;
}

/** Predicted noise in a noisy unit chunk x (flat, length 2H) at step k. */
export function dpEps(P, x, k, o) {
  const obsEmb = linear(silu(linear(normalize(P, o), P['obs_in.0.weight'], P['obs_in.0.bias'])), P['obs_in.2.weight'], P['obs_in.2.bias']);
  const c = Float32Array.from([...obsEmb, ...stepEmbedding(k)]);
  let h = linear(x, P['inp.weight'], P['inp.bias']);
  for (let b = 0; P[`blocks.${b}.film.weight`]; b++) {
    const p = `blocks.${b}`, film = linear(c, P[`${p}.film.weight`], P[`${p}.film.bias`]), n = h.length;
    const z = silu(map(layerNorm(h, P[`${p}.norm.weight`], P[`${p}.norm.bias`]), (v, i) => v * (1 + film[i]) + film[n + i]));
    h = add(h, linear(silu(linear(z, P[`${p}.fc1.weight`], P[`${p}.fc1.bias`])), P[`${p}.fc2.weight`], P[`${p}.fc2.bias`]));
  }
  return linear(silu(layerNorm(h, P['out.0.weight'], P['out.0.bias'])), P['out.2.weight'], P['out.2.bias']);
}

const toChunk = (x) => Array.from({ length: H }, (_, t) => fromUnit([x[2 * t], x[2 * t + 1]]));

/** DDPM sampling from pure noise; keep = steps whose noisy chunk is returned too (step 0 = the final plan). */
export function dpSample(P, o, schedule, rng, keep = []) {
  const { betas, abar } = schedule, n = 2 * H;
  let x = Float32Array.from({ length: n }, () => rng.gauss());
  const snaps = {};
  for (let k = betas.length - 1; k >= 0; k--) {
    if (k && keep.includes(k)) snaps[k] = toChunk(x);
    const a = abar[k], aPrev = k ? abar[k - 1] : 1, beta = betas[k], eps = dpEps(P, x, k, o);
    const x0 = map(x, (v, i) => Math.min(1, Math.max(-1, (v - Math.sqrt(1 - a) * eps[i]) / Math.sqrt(a))));
    const sd = Math.sqrt((beta * (1 - aPrev)) / (1 - a));
    x = map(x, (v, i) => (Math.sqrt(aPrev) * beta * x0[i] + Math.sqrt(1 - beta) * (1 - aPrev) * v) / (1 - a) + (k ? sd * rng.gauss() : 0));
  }
  const chunk = toChunk(x);
  if (keep.includes(0)) snaps[0] = chunk;
  return { chunk, snaps };
}

// ---------- flow matching ----------

/** Velocity of a unit chunk x (flat, length 2H) at time t in [0, 1]: one token per action, adaLN-Zero blocks. */
export function fpVelocity(P, x, t, o) {
  const d = P['pos'].shape[1], lin = (v, name) => linear(v, P[`${name}.weight`], P[`${name}.bias`]);
  const c = add(lin(silu(lin(normalize(P, o), 'obs_in.0')), 'obs_in.2'), lin(silu(lin(stepEmbedding(1000 * t, d), 'time.1')), 'time.3'));
  let xs = Array.from({ length: H }, (_, i) => add(lin(x.subarray(2 * i, 2 * i + 2), 'inp'), P['pos'].data.subarray(i * d, (i + 1) * d)));
  for (let b = 0; P[`blocks.${b}.ada.weight`]; b++) {
    const p = `blocks.${b}`, ada = lin(silu(c), `${p}.ada`), part = (k) => ada.subarray(k * d, (k + 1) * d);
    const [s1, b1, g1, s2, b2, g2] = [0, 1, 2, 3, 4, 5].map(part);
    const mod = (v, s, sh) => map(layerNorm(v), (u, i) => u * (1 + s[i]) + sh[i]);
    const a = attention(P, `${p}.attn`, xs.map((v) => mod(v, s1, b1)), false);
    xs = xs.map((v, i) => map(v, (u, j) => u + g1[j] * a[i][j]));
    xs = xs.map((v) => { const m = lin(gelu(lin(mod(v, s2, b2), `${p}.mlp.0`)), `${p}.mlp.2`); return map(v, (u, j) => u + g2[j] * m[j]); });
  }
  return Float32Array.from(xs.flatMap((v) => [...lin(layerNorm(v), 'out')]));
}

/** Euler integration from noise (t = 0) to a plan (t = 1); keep = step indices whose chunk is returned too. */
export function fpSample(P, o, steps, rng, keep = []) {
  let x = Float32Array.from({ length: 2 * H }, () => rng.gauss());
  const snaps = {};
  for (let i = 0; i < steps; i++) {
    if (keep.includes(i)) snaps[i] = toChunk(x);
    const v = fpVelocity(P, x, i / steps, o);
    x = map(x, (u, j) => u + v[j] / steps);
  }
  const chunk = toChunk(x);
  if (keep.includes(steps)) snaps[steps] = chunk;
  return { chunk, snaps };
}

// ---------- CVAE ----------

/** Decoder only (the encoder is used in training): chunk from the observation and a latent code z. */
export function cvDecode(P, o, z) {
  const lin = (v, i) => linear(v, P[`decoder.${i}.weight`], P[`decoder.${i}.bias`]);
  return toChunk(lin(silu(lin(silu(lin(Float32Array.from([...normalize(P, o), ...z]), 0)), 2)), 4));
}

/** A fresh code z ~ N(0, I) for every chunk. */
export function cvAct(P, o, rng) {
  const zDim = P['decoder.0.weight'].shape[1] - o.length;
  return cvDecode(P, o, Array.from({ length: zDim }, () => rng.gauss()));
}
