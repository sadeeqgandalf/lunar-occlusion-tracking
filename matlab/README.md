# MATLAB: RPO docking with an EKF

`docking_sim.m` is a MATLAB port of the web lab's spacecraft scenario (RPO 1 · Docking Approach): same
Clohessy-Wiltshire dynamics, sensors, noise levels, EKF tuning and guidance, so results can be compared side by side.

```matlab
cd matlab
docking_sim                         % watch it live (4x real time)
docking_sim(Speed=16)               % faster
docking_sim(Video="docking.mp4")    % record the animation
out = docking_sim(Animate=false);   % run only; out has truth, estimate, covariance, NEES, event log
docking_sim(QScale=1, RScale=1)     % untuned filter: compare consistency (ANEES)
```

Needs base MATLAB only (tested on R2026b). The seed-7 run docks after about 205 s, with every gate passed and a
position RMSE of 0.23 m.
