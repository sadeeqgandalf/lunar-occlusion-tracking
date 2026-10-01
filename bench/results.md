# Benchmark results

Mean over seeds 1,2,3,4,5; oracle autopilot, identical sensor streams for every filter.

| platform | scenario | filter | RMSE [m] | ANEES (ideal = dof) | NEES in 95% band | µs/step |
|---|---|---|---:|---:|---:|---:|
| rover | nominal | dr | 17.0 | 1.38 (dof 3) | 83% | 2.4 |
| rover | nominal | ekf | 0.27 | 1.14 (dof 3) | 89% | 2.8 |
| rover | nominal | ukf | 0.26 | 1.12 (dof 3) | 89% | 15.3 |
| rover | nominal | pf | 0.26 | 0.54 (dof 3) | 64% | 552.6 |
| rover | dust | dr | 63.7 | 15.8 (dof 3) | 43% | 2.4 |
| rover | dust | ekf | 3.97 | 14.7 (dof 3) | 74% | 2.5 |
| rover | dust | ukf | 4.90 | 8.04 (dof 3) | 75% | 13.8 |
| rover | dust | pf | 1.58 | 1.37 (dof 3) | 65% | 544.9 |
| rover | sand | dr | 27.0 | 1.78 (dof 3) | 99% | 2.3 |
| rover | sand | ekf | 2.75 | 1.95 (dof 3) | 89% | 2.5 |
| rover | sand | ukf | 2.72 | 1.93 (dof 3) | 88% | 13.7 |
| rover | sand | pf | 3.23 | 1.54 (dof 3) | 72% | 543.5 |
| jet | corridor | dr | 1336 | 1.60 (dof 4) | 95% | 4.2 |
| jet | corridor | ekf | 3.85 | 1.50 (dof 4) | 88% | 5.9 |
| jet | corridor | ukf | 3.85 | 1.50 (dof 4) | 88% | 25.8 |
| jet | corridor | pf | 3.83 | 0.53 (dof 4) | 40% | 704.8 |
| jet | jammed | dr | 4350 | 25.8 (dof 4) | 29% | 4.5 |
| jet | jammed | ekf | 9.06 | 2.00 (dof 4) | 89% | 5.9 |
| jet | jammed | ukf | 9.06 | 2.00 (dof 4) | 89% | 25.7 |
| jet | jammed | pf | 9.28 | 0.66 (dof 4) | 46% | 696.7 |
| spacecraft | approach | dr | 40.3 | 2.04 (dof 4) | 80% | 3.0 |
| spacecraft | approach | ekf | 0.23 | 1.31 (dof 4) | 80% | 3.3 |
| spacecraft | approach | ukf | 0.23 | 1.31 (dof 4) | 80% | 20.2 |
| spacecraft | approach | pf | 0.32 | 1.63 (dof 4) | 43% | 780.1 |
| spacecraft | blind | dr | 59.2 | 2.52 (dof 4) | 100% | 3.1 |
| spacecraft | blind | ekf | 0.57 | 1.67 (dof 4) | 86% | 3.2 |
| spacecraft | blind | ukf | 0.57 | 1.66 (dof 4) | 86% | 20.2 |
| spacecraft | blind | pf | 0.82 | 2.26 (dof 4) | 58% | 787.1 |
