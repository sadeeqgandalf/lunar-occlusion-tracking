// Kuhn-Munkres (Hungarian) minimum-cost assignment, O(n^3), rectangular via padding.
// Returns assign[row] = col or -1. Pairs whose cost >= `forbid` are never returned.
export function hungarian(cost, forbid = 1e9) {
  const nr = cost.length, nc = nr ? cost[0].length : 0, n = Math.max(nr, nc);
  if (!n) return [];
  const BIG = forbid;
  const a = Array.from({ length: n + 1 }, (_, i) => Array.from({ length: n + 1 }, (_, j) => (i && j && i <= nr && j <= nc ? Math.min(cost[i - 1][j - 1], BIG) : i && j ? BIG : 0)));
  const u = new Array(n + 1).fill(0), v = new Array(n + 1).fill(0), p = new Array(n + 1).fill(0), way = new Array(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i; let j0 = 0;
    const minv = new Array(n + 1).fill(Infinity), used = new Array(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0]; let delta = Infinity, j1 = 0;
      for (let j = 1; j <= n; j++) if (!used[j]) {
        const cur = a[i0][j] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j++) if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const assign = new Array(nr).fill(-1);
  for (let j = 1; j <= n; j++) {
    const i = p[j];
    if (i >= 1 && i <= nr && j <= nc && cost[i - 1][j - 1] < forbid) assign[i - 1] = j - 1;
  }
  return assign;
}
