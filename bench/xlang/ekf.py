"""Reference Python/numpy implementation of the rover EKF: same equations as ekf.cpp / src/filters/ekf.js."""
import sys, math, time
import numpy as np

SV, SVR, SW, SB, SS, QS, RS, GATE = 0.03, 0.03, 0.01, 0.00005, 0.02, 4.0, 1.5, 9.21
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi

class EKF:
    def init(self, x0):
        self.x = np.array(x0, float); self.P = np.diag([0.25, 0.25, 0.0025, 4e-6, 9e-4]); self.streak = {0: 0, 1: 0}
    def predict(self, vmeas, wm, dt):
        x = self.x; th = x[2]; w = wm - x[3]; v = vmeas * (1 - x[4])
        if abs(w) < 1e-3:
            m = th + w * dt / 2; s, c = math.sin(m), math.cos(m)
            dxdth, dydth, dxdw, dydw, dxdv, dydv = -v*dt*s, v*dt*c, -v*dt*dt*s/2, v*dt*dt*c/2, dt*c, dt*s
            nx, ny = x[0] + v*dt*c, x[1] + v*dt*s
        else:
            s0, c0, s1, c1, r = math.sin(th), math.cos(th), math.sin(th+w*dt), math.cos(th+w*dt), v / w
            dxdth, dydth = r*(c1-c0), r*(s1-s0)
            dxdw = -v/(w*w)*(s1-s0) + r*dt*c1; dydw = -v/(w*w)*(c0-c1) + r*dt*s1
            dxdv, dydv = (s1-s0)/w, (c0-c1)/w
            nx, ny = x[0] + r*(s1-s0), x[1] + r*(c0-c1)
        G = np.array([[1,0,dxdth,-dxdw,-vmeas*dxdv],[0,1,dydth,-dydw,-vmeas*dydv],[0,0,1,-dt,0],[0,0,0,1,0],[0,0,0,0,1]])
        V = np.array([[dxdv*(1-x[4]),dxdw,0,0],[dydv*(1-x[4]),dydw,0,0],[0,dt,0,0],[0,0,1,0],[0,0,0,1]])
        W = np.diag([SV**2 + (SVR*vmeas)**2, SW**2, SB**2*dt, SS**2*dt])
        P = G @ self.P @ G.T + QS * (V @ W @ V.T)
        self.P = 0.5 * (P + P.T); self.x = np.array([nx, ny, wrap(th + w*dt), x[3], x[4]])
    def correct(self, kind, y, H, R):
        S = H @ self.P @ H.T + R; Si = np.linalg.inv(S); nis = y @ Si @ y
        if not nis <= GATE:
            self.streak[kind] += 1
            if self.streak[kind] >= 10: self.P *= 1.5
            return
        self.streak[kind] = 0
        K = self.P @ H.T @ Si; self.x = self.x + K @ y; self.x[2] = wrap(self.x[2])
        IKH = np.eye(5) - K @ H; P = IKH @ self.P @ IKH.T + K @ R @ K.T; self.P = 0.5 * (P + P.T)
    def landmark(self, lx, ly, rng, brg, sR, sB):
        x = self.x; dx, dy = lx - x[0], ly - x[1]; q = dx*dx + dy*dy; r = math.sqrt(q)
        H = np.array([[-dx/r, -dy/r, 0, 0, 0], [dy/q, -dx/q, -1, 0, 0]])
        self.correct(0, np.array([rng - r, wrap(brg - wrap(math.atan2(dy, dx) - x[2]))]), H, np.diag([(sR*RS)**2, (sB*RS)**2]))
    def fix(self, fx, fy, s):
        H = np.array([[1,0,0,0,0],[0,1,0,0,0]], float)
        self.correct(1, np.array([fx - self.x[0], fy - self.x[1]]), H, np.diag([(s*RS)**2]*2))

rec = [(l[0], [float(v) for v in l[1:]]) for l in (ln.split() for ln in open(sys.argv[1]))]
reps = int(sys.argv[2]); x0 = rec[0][1][:5]; f = EKF(); steps = 0; t0 = time.perf_counter()
for _ in range(reps):
    f.init(x0)
    for tag, v in rec[1:]:
        if tag == 'U': f.predict(v[0], v[1], 0.05); steps += 1
        elif tag == 'L': f.landmark(*v)
        elif tag == 'F': f.fix(*v)
us = (time.perf_counter() - t0) * 1e6 / steps
print(" ".join(f"{v:.9f}" for v in f.x), f"{us:.4f}")
