"""Kuhn-Munkres assignment, line-for-line port of src/mot/hungarian.js (ties resolve identically)."""
import math


def hungarian(cost, forbid=1e9):
    nr = len(cost)
    nc = len(cost[0]) if nr else 0
    n = max(nr, nc)
    if not nr or not nc:
        return [-1] * nr
    a = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            a[i][j] = min(cost[i - 1][j - 1], forbid) if i <= nr and j <= nc else forbid
    u, v, p, way = [0.0] * (n + 1), [0.0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [math.inf] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0][j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0:
                break
    out = [-1] * nr
    for j in range(1, n + 1):
        i = p[j]
        if 1 <= i <= nr and j <= nc and cost[i - 1][j - 1] < forbid:
            out[i - 1] = j - 1
    return out
