// Reference C++17 implementation of the rover EKF (Probabilistic Robotics Table 3.3), dependency-free.
// Same equations, gating and covariance-inflation recovery as src/filters/ekf.js; replays a recorded trace.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

template <int R, int C> struct Mat { array<double, R * C> d{}; double& operator()(int i, int j) { return d[i * C + j]; } double operator()(int i, int j) const { return d[i * C + j]; } };
template <int R, int K, int C> Mat<R, C> mul(const Mat<R, K>& a, const Mat<K, C>& b) { Mat<R, C> o; for (int i = 0; i < R; i++) for (int k = 0; k < K; k++) { double v = a(i, k); for (int j = 0; j < C; j++) o(i, j) += v * b(k, j); } return o; }
template <int R, int C> Mat<C, R> tr(const Mat<R, C>& a) { Mat<C, R> o; for (int i = 0; i < R; i++) for (int j = 0; j < C; j++) o(j, i) = a(i, j); return o; }
template <int R, int C> Mat<R, C> add(const Mat<R, C>& a, const Mat<R, C>& b) { Mat<R, C> o; for (int i = 0; i < R * C; i++) o.d[i] = a.d[i] + b.d[i]; return o; }
template <int R, int C> Mat<R, C> scl(const Mat<R, C>& a, double s) { Mat<R, C> o; for (int i = 0; i < R * C; i++) o.d[i] = a.d[i] * s; return o; }
static void sym(Mat<5, 5>& P) { for (int i = 0; i < 5; i++) for (int j = i + 1; j < 5; j++) { double v = 0.5 * (P(i, j) + P(j, i)); P(i, j) = P(j, i) = v; } }
static double wrap(double a) { a = fmod(a + M_PI, 2 * M_PI); if (a < 0) a += 2 * M_PI; return a - M_PI; }

struct Cfg { double sigmaV = 0.03, sigmaVRel = 0.03, sigmaW = 0.01, sigmaBias = 0.00005, sigmaSlip = 0.02, qScale = 4, rScale = 1.5, gate = 9.21; };

struct EKF {
  Cfg c; array<double, 5> x{}; Mat<5, 5> P; int streak[2] = {0, 0};
  void init(array<double, 5> x0) { x = x0; P = {}; P(0, 0) = 0.25; P(1, 1) = 0.25; P(2, 2) = 0.0025; P(3, 3) = 4e-6; P(4, 4) = 9e-4; }
  void predict(double vmeas, double wm, double dt) {
    double v = vmeas * (1 - x[4]); // ground speed = odometry * (1 - slip)
    double th = x[2], w = wm - x[3], vm = v, dxdth, dydth, dxdw, dydw, dxdv, dydv, nx, ny;
    if (fabs(w) < 1e-3) {
      double m = th + w * dt / 2, s = sin(m), cc = cos(m);
      dxdth = -v * dt * s; dydth = v * dt * cc; dxdw = -v * dt * dt * s / 2; dydw = v * dt * dt * cc / 2; dxdv = dt * cc; dydv = dt * s;
      nx = x[0] + v * dt * cc; ny = x[1] + v * dt * s;
    } else {
      double s0 = sin(th), c0 = cos(th), s1 = sin(th + w * dt), c1 = cos(th + w * dt), r = v / w;
      dxdth = r * (c1 - c0); dydth = r * (s1 - s0);
      dxdw = -v / (w * w) * (s1 - s0) + r * dt * c1; dydw = -v / (w * w) * (c0 - c1) + r * dt * s1;
      dxdv = (s1 - s0) / w; dydv = (c0 - c1) / w;
      nx = x[0] + r * (s1 - s0); ny = x[1] + r * (c0 - c1);
    }
    Mat<5, 5> G; for (int i = 0; i < 5; i++) G(i, i) = 1; G(0, 2) = dxdth; G(0, 3) = -dxdw; G(0, 4) = -vmeas * dxdv; G(1, 2) = dydth; G(1, 3) = -dydw; G(1, 4) = -vmeas * dydv; G(2, 3) = -dt;
    Mat<5, 4> V; V(0, 0) = dxdv * (1 - x[4]); V(0, 1) = dxdw; V(1, 0) = dydv * (1 - x[4]); V(1, 1) = dydw; V(2, 1) = dt; V(3, 2) = 1; V(4, 3) = 1;
    Mat<4, 4> W; W(0, 0) = c.sigmaV * c.sigmaV + pow(c.sigmaVRel * vmeas, 2); W(1, 1) = c.sigmaW * c.sigmaW; W(2, 2) = c.sigmaBias * c.sigmaBias * dt; W(3, 3) = c.sigmaSlip * c.sigmaSlip * dt;
    Mat<5, 5> Q = scl(mul(mul(V, W), tr(V)), c.qScale);
    P = add(mul(mul(G, P), tr(G)), Q); sym(P);
    x = {nx, ny, wrap(th + w * dt), x[3], x[4]};
  }
  void correct(int kind, const array<double, 2>& y, const Mat<2, 5>& H, const array<double, 2>& Rd) {
    Mat<2, 2> S = add(mul(mul(H, P), tr(H)), Mat<2, 2>{{Rd[0], 0, 0, Rd[1]}});
    double det = S(0, 0) * S(1, 1) - S(0, 1) * S(1, 0);
    Mat<2, 2> Si{{S(1, 1) / det, -S(0, 1) / det, -S(1, 0) / det, S(0, 0) / det}};
    double nis = y[0] * (Si(0, 0) * y[0] + Si(0, 1) * y[1]) + y[1] * (Si(1, 0) * y[0] + Si(1, 1) * y[1]);
    if (!(nis <= c.gate)) { if (++streak[kind] >= 10) P = scl(P, 1.5); return; }
    streak[kind] = 0;
    Mat<5, 2> K = mul(mul(P, tr(H)), Si);
    for (int i = 0; i < 5; i++) x[i] += K(i, 0) * y[0] + K(i, 1) * y[1];
    x[2] = wrap(x[2]);
    Mat<5, 5> IKH; for (int i = 0; i < 5; i++) IKH(i, i) = 1; Mat<5, 5> KH = mul(K, H); for (int i = 0; i < 25; i++) IKH.d[i] -= KH.d[i];
    Mat<2, 2> R{{Rd[0], 0, 0, Rd[1]}};
    P = add(mul(mul(IKH, P), tr(IKH)), mul(mul(K, R), tr(K))); sym(P);
  }
  void landmark(double lx, double ly, double range, double bearing, double sR, double sB) {
    double dx = lx - x[0], dy = ly - x[1], q = dx * dx + dy * dy, r = sqrt(q);
    Mat<2, 5> H; H(0, 0) = -dx / r; H(0, 1) = -dy / r; H(1, 0) = dy / q; H(1, 1) = -dx / q; H(1, 2) = -1;
    correct(0, {range - r, wrap(bearing - wrap(atan2(dy, dx) - x[2]))}, H, {pow(sR * c.rScale, 2), pow(sB * c.rScale, 2)});
  }
  void fix(double fx, double fy, double s) {
    Mat<2, 5> H; H(0, 0) = 1; H(1, 1) = 1;
    correct(1, {fx - x[0], fy - x[1]}, H, {pow(s * c.rScale, 2), pow(s * c.rScale, 2)});
  }
};

int main(int argc, char** argv) {
  ifstream in(argv[1]); int reps = atoi(argv[2]); string line; vector<vector<double>> rec; vector<char> tag;
  while (getline(in, line)) { istringstream ss(line); char t; ss >> t; vector<double> v; double d; while (ss >> d) v.push_back(d); tag.push_back(t); rec.push_back(v); }
  array<double, 5> x0{}; for (int i = 0; i < 5; i++) x0[i] = rec[0][i];
  EKF f; auto t0 = chrono::steady_clock::now(); long steps = 0;
  for (int r = 0; r < reps; r++) {
    f.init(x0); f.streak[0] = f.streak[1] = 0;
    for (size_t i = 1; i < rec.size(); i++) {
      const auto& v = rec[i];
      if (tag[i] == 'U') { f.predict(v[0], v[1], 0.05); steps++; }
      else if (tag[i] == 'L') f.landmark(v[0], v[1], v[2], v[3], v[4], v[5]);
      else if (tag[i] == 'F') f.fix(v[0], v[1], v[2]);
    }
  }
  double us = chrono::duration<double, micro>(chrono::steady_clock::now() - t0).count();
  printf("%.9f %.9f %.9f %.9f %.9f %.4f\n", f.x[0], f.x[1], f.x[2], f.x[3], f.x[4], us / steps);
}
