// Seeded PRNG (mulberry32) + Box-Muller gaussian. Determinism = reproducible benchmarks.
export class RNG {
  constructor(seed = 1) { this.s = seed >>> 0; this.spare = null; }
  next() {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  uniform(a = 0, b = 1) { return a + (b - a) * this.next(); }
  int(n) { return Math.floor(this.next() * n); }
  randn() {
    if (this.spare !== null) { const s = this.spare; this.spare = null; return s; }
    let u = 0;
    while (u === 0) u = this.next();
    const v = this.next();
    const m = Math.sqrt(-2 * Math.log(u));
    this.spare = m * Math.sin(2 * Math.PI * v);
    return m * Math.cos(2 * Math.PI * v);
  }
}
