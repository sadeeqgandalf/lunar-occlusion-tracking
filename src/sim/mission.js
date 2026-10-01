// Platform-agnostic mission engine: truth + sensors (via a Platform), N estimators in parallel on the same
// measurement stream, command uplink latency, hazards/objectives/scoring, metrics and history. DOM-free.
import { RNG } from '../core/rng.js';
import { wrapAngle } from '../core/linalg.js';
import { getFilterClass } from '../filters/index.js';
import { resid } from '../filters/base.js';
import { getPlatform } from '../platforms/index.js';
import { nees, neesBounds, sigmaMax } from './metrics.js';

const HIST_MAX = 1500;
const fmt = (v) => (v >= 1000 ? (v / 1000).toFixed(1) + 'k' : v >= 10 ? v.toFixed(0) : v.toFixed(1));

export class Mission {
  constructor(opts = {}) {
    const o = { platform: 'rover', scenario: 'nominal', seed: 7, filters: ['dr', 'ekf', 'ukf', 'pf'], primary: 'ekf', oracle: false, ...opts };
    this.o = o;
    this.platform = getPlatform(o.platform);
    this.model = this.platform.model;
    this.dt = 0.05;
    this.rng = new RNG(o.seed + 1000);
    this.world = this.platform.build(o.scenario, o.seed);
    this.sensors = { ...this.platform.sensors, ...(this.world.sensors || {}) };

    this.t = 0;
    this.status = 'running'; // running | complete | failed
    this.failReason = '';
    this.hull = 100;
    this.resource = 100;
    this.score = 0;
    this.events = [];
    this.faults = new Map();
    this.queue = [];
    this.uplinkDelay = 0;
    this.manual = this.platform.idleCmd(this);
    this.manualActive = false;
    this.waypoints = [];
    this.safing = true;
    this.safeHold = false;
    this.immobUntil = 0;
    this.dwell = null;
    this.inHazard = new Set();
    this.lastMeas = [];
    this.nextSense = 0;
    this.targets = this.world.targets.map((t) => ({ ...t, done: false }));

    // truth is drawn from the same prior the filters are given -> NEES is meaningful from t=0
    const nominal = this.platform.nominalStart(this.world), P0 = this.model.P0(), L = P0.chol(), n = this.model.nx;
    const z = Array.from({ length: n }, () => this.rng.randn());
    this.truth = nominal.slice();
    for (let r = 0; r < n; r++) for (let c = 0; c <= r; c++) this.truth[r] += L.get(r, c) * z[c];
    this.ps = this.platform.initPlatformState(this);
    this.nominal = nominal;

    this.fs = new Map();
    this.primary = o.primary;
    for (const id of o.filters) this.addFilter(id, nominal, P0);
    if (!this.fs.has(this.primary)) this.primary = [...this.fs.keys()][0];

    this.hist = [];
    this.truthTrail = [];
    this.nextRec = 0;
    this.log(`${this.platform.label} · ${this.world.cfg.label} online`, 'info');
  }

  // ---------------------------------------------------------------- filters
  addFilter(id, x0 = null, P0 = null) {
    const Cls = getFilterClass(id);
    if (!Cls || this.fs.has(id)) return;
    if (!x0) { // hot-add: seed from the primary estimate so the newcomer is comparable
      const p = this.fs.get(this.primary)?.filter;
      x0 = p ? p.getState() : this.nominal; P0 = p ? p.getCov() : this.model.P0();
    }
    let h = 0; for (const c of id) h = (h * 31 + c.charCodeAt(0)) >>> 0;
    const f = new Cls(this.model, this.sensors, {}, new RNG(this.o.seed * 31 + h));
    f.init(x0, P0);
    this.fs.set(id, { id, Cls, filter: f, trail: [], m: { n: 0, sumSq: 0, neesN: 0, neesIn: 0, neesSum: 0, accepted: 0, rejected: 0, us: 0, steps: 0, last: {}, lastAccT: 0, recent: [] } });
  }
  removeFilter(id) {
    if (this.fs.size <= 1) return;
    this.fs.delete(id);
    if (this.primary === id) this.primary = [...this.fs.keys()][0];
  }
  setPrimary(id) { if (this.fs.has(id)) this.primary = id; }

  estimate(id = this.primary) {
    const f = this.fs.get(id)?.filter;
    return f ? { x: f.getState(), P: f.getCov() } : null;
  }

  // ---------------------------------------------------------------- commands & faults
  log(text, level = 'info') { this.events.push({ t: this.t, text, level }); if (this.events.length > 300) this.events.shift(); }

  issue(cmd) {
    if (this.uplinkDelay <= 0) this._apply(cmd);
    else { this.queue.push({ at: this.t + this.uplinkDelay, cmd }); this.queue.sort((a, b) => a.at - b.at); }
  }
  _apply(c) {
    switch (c.type) {
      case 'drive': this.manual = c.cmd; this.manualActive = this.platform.isActive(c.cmd); break;
      case 'goto': this.waypoints.push({ x: c.x, y: c.y, kind: c.kind || 'goto', targetId: c.targetId }); this.safeHold = false; break;
      case 'clear': this.waypoints = []; this.dwell = null; break;
      case 'stop': this.waypoints = []; this.manual = this.platform.idleCmd(this); this.manualActive = false; this.dwell = null; break;
      case 'plan': this.waypoints = this.world.route.map((r) => ({ ...r })); this.safeHold = false; this.log(`Nominal plan uplinked (${this.waypoints.length} legs)`, 'info'); break;
      case 'undo': this.waypoints.pop(); break;
    }
  }
  faultActive(type) { return (this.faults.get(type) || 0) > this.t; }
  injectFault(type) {
    const d = this.platform.applyFault(this, type);
    if (d > 0) this.faults.set(type, this.t + d);
    this.log(`FAULT INJECTED: ${this.platform.faultLabel(type)}`, 'fault');
  }

  // ---------------------------------------------------------------- step
  step() {
    if (this.status !== 'running') return;
    const dt = this.dt, P = this.platform;
    this.t += dt;
    while (this.queue.length && this.queue[0].at <= this.t) this._apply(this.queue.shift().cmd);
    for (const s of this.world.schedule) if (!s.done && this.t >= s.t) { s.done = true; this.injectFault(s.type); }

    const cmd = this._control();
    const u = P.step(this, cmd, dt);
    this._hazards();
    if (this.resource <= 0 && P.resource.fatal) return this._fail(`${P.resource.name} exhausted`);

    for (const f of this.fs.values()) { const t0 = performance.now(); f.filter.predict(u, dt); f.m.us += (performance.now() - t0) * 1000; f.m.steps++; }
    if (this.t >= this.nextSense) { this.nextSense = this.t + this.sensors.sensePeriod; this._sense(); }
    if (this.t >= this.nextRec) { this.nextRec = this.t + 0.2; this._record(); }
    if (this.world.goal && this.targets.length && this.targets.every((t) => t.done)) this._complete();
  }

  _sense() {
    const meas = this.platform.sense(this, this.rng);
    this.lastMeas = meas.filter((m) => m.lx !== undefined).map((m) => ({ lx: m.lx, ly: m.ly, t: this.t }));
    for (const m of meas) for (const f of this.fs.values()) {
      const t0 = performance.now();
      const r = f.filter.update(m);
      f.m.us += (performance.now() - t0) * 1000;
      if (r.accepted) { f.m.accepted++; if (m.kind !== 'airspeed') f.m.lastAccT = this.t; } else if (r.nis !== undefined) f.m.rejected++;
      if (m.kind !== 'airspeed' && (r.accepted || r.nis !== undefined)) { // exteroceptive outcomes only; keep a 10 s window
        f.m.recent.push({ t: this.t, ok: r.accepted });
        while (f.m.recent.length && f.m.recent[0].t < this.t - 10) f.m.recent.shift();
      }
    }
  }

  _ctrlPose() {
    const x = this.o.oracle ? this.truth : this.fs.get(this.primary).filter.getState();
    return this.platform.pose(x);
  }

  _control() {
    const P = this.platform, idle = P.idleCmd(this);
    const hold = () => (P.holdCmd ? P.holdCmd(this, this._ctrlPose()) : idle);
    if (this.t < this.immobUntil) return idle;
    if (this.dwell) {
      if (this.t < this.dwell.until) return hold();
      this._resolve(this.dwell.target); this.dwell = null;
    }
    if (this.manualActive) return this.manual;
    const wp = this.waypoints[0];
    if (!wp) return hold();
    const est = this._ctrlPose();

    if (this.safing && !this.o.oracle) {
      const sig = sigmaMax(this.model, this.fs.get(this.primary).filter.getCov()), margin = P.safeMargin(this.world);
      for (const h of this.world.hazards) {
        const clear = Math.hypot(est.x - h.x, est.y - h.y) - h.r;
        const ahead = Math.cos(Math.atan2(h.y - est.y, h.x - est.x) - est.th);
        if (clear < 3 * sig + margin && ahead > 0.3) {
          if (!this.safeHold) this.log(`SAFE-HOLD: 3σ pose bubble (${(3 * sig).toFixed(1)} ${P.unit}) overlaps hazard ahead`, 'warn');
          this.safeHold = true;
          return P.safeCmd(this, est, h);
        }
      }
      this.safeHold = false;
    }
    const r = P.autopilot(this, est, wp);
    if (r.arrived) this._arrive(wp);
    return r.cmd;
  }

  _arrive(wp) {
    this.waypoints.shift();
    if (wp.kind !== 'objective') return;
    const target = this.targets.find((t) => t.id === wp.targetId);
    if (!target || target.done) return;
    const d = this.platform.objective.dwell;
    if (d > 0) { this.dwell = { until: this.t + d, target }; this.log(`${this.platform.objective.verb} #${target.id}…`, 'info'); }
    else this._resolve(target);
  }

  _resolve(target) {
    const r = this.platform.checkObjective(this, target), O = this.platform.objective;
    this.resource = Math.max(0, this.resource - O.cost);
    if (r.ok) { target.done = true; this.score += O.points; this.log(`${O.noun} #${target.id} SUCCESS · ${r.text}`, 'good'); }
    else this.log(`${O.noun} #${target.id} MISSED · ${r.text}`, 'warn');
  }

  _hazards() {
    const P = this.platform, p = P.pose(this.truth);
    for (const [i, h] of this.world.hazards.entries()) {
      const inside = Math.hypot(p.x - h.x, p.y - h.y) < h.r;
      if (inside && !this.inHazard.has(i)) {
        this.inHazard.add(i);
        this.hull = Math.max(0, this.hull - P.hazard.damage);
        this.immobUntil = this.t + P.hazard.immobilize;
        P.onHazard(this, h);
        this.log(`${P.hazard.label}! hull ${Math.round(this.hull)}%`, 'bad');
        if (this.hull <= 0) this._fail(`${P.hazard.label}: vehicle lost`);
      } else if (!inside) this.inHazard.delete(i);
    }
  }

  _fail(reason) { this.status = 'failed'; this.failReason = reason; this.log(`MISSION FAILED · ${reason}`, 'bad'); }
  _complete() {
    this.status = 'complete';
    this.score += Math.round(this.hull * 2 + this.resource);
    this.log(`MISSION COMPLETE · score ${this.score}`, 'good');
  }

  // ---------------------------------------------------------------- metrics
  _record() {
    const M = this.model, truthPose = this.platform.pose(this.truth);
    const entry = { t: this.t, truth: truthPose, f: {} };
    for (const f of this.fs.values()) {
      const x = f.filter.getState(), P = f.filter.getCov(), e = resid(M.angleStates, this.truth, x);
      const pos = Math.hypot(e[M.posIdx[0]], e[M.posIdx[1]]);
      const head = M.headingIdx >= 0 ? e[M.headingIdx] : NaN;
      const sig = sigmaMax(M, P), ne = nees(M, this.truth, x, P);
      const m = f.m;
      m.n++; m.sumSq += pos * pos;
      if (Number.isFinite(ne)) {
        const [lo, hi] = neesBounds(M.neesIdx.length);
        m.neesN++; m.neesSum += ne; if (ne >= lo && ne <= hi) m.neesIn++;
      }
      m.last = { pos, head, sig, nees: ne };
      entry.f[f.id] = { x: x[M.posIdx[0]], y: x[M.posIdx[1]], th: M.headingIdx >= 0 ? x[M.headingIdx] : 0, pos, sig, nees: ne };
      f.trail.push([x[M.posIdx[0]], x[M.posIdx[1]]]);
      if (f.trail.length > 900) f.trail.shift();
    }
    this.hist.push(entry);
    if (this.hist.length > HIST_MAX) this.hist.shift();
    this.truthTrail.push([truthPose.x, truthPose.y]);
    if (this.truthTrail.length > 900) this.truthTrail.shift();
  }

  summary(id) {
    const m = this.fs.get(id)?.m;
    if (!m || !m.n) return null;
    const dof = this.model.neesIdx.length, anees = m.neesN ? m.neesSum / m.neesN : NaN;
    return {
      rmse: Math.sqrt(m.sumSq / m.n), anees, dof, neesFrac: m.neesN ? m.neesIn / m.neesN : NaN,
      rejected: m.rejected, accepted: m.accepted, usPerStep: m.steps ? m.us / m.steps : 0, last: m.last,
    };
  }

  /**
   * Navigation health from onboard quantities only (an operator never sees ground truth):
   *  - 3-sigma position bubble vs the accuracy the next objective needs,
   *  - fraction of recent outside measurements the filter rejected (a filter that disagrees with the world),
   *  - time since the last accepted outside measurement (how long it has been flying blind).
   * Real systems build exactly this kind of integrity monitor, with a decision attached.
   */
  health(id = this.primary) {
    const f = this.fs.get(id);
    if (!f) return null;
    const sig3 = 3 * sigmaMax(this.model, f.filter.getCov()), tol = this.platform.objective.tol;
    const rec = f.m.recent, rej = rec.length ? rec.filter((r) => !r.ok).length / rec.length : 0;
    const age = this.t - f.m.lastAccT, reasons = [];
    let level = 'good';
    const bump = (l, why) => { reasons.push(why); if (l === 'bad' || level === 'good') level = l === 'bad' ? 'bad' : level === 'good' ? l : level; };
    if (sig3 > tol) bump('bad', `3σ position bubble ${fmt(sig3)} ${this.platform.unit} exceeds the ${fmt(tol)} ${this.platform.unit} the objective needs`);
    // first 8 s: the filter is settling from its launch prior, not degrading
    else if (sig3 > 0.5 * tol && this.t >= 8) bump('warn', `3σ bubble ${fmt(sig3)} ${this.platform.unit} is over half the objective tolerance`);
    if (rec.length >= 4 && rej > 0.5) bump('bad', `${Math.round(rej * 100)}% of recent measurements rejected: the filter and the world disagree`);
    else if (rec.length >= 4 && rej > 0.2) bump('warn', `${Math.round(rej * 100)}% of recent measurements rejected`);
    if (this.t > 5 && age > 40) bump('bad', `no accepted outside measurement for ${age.toFixed(0)} s: flying on inertial/odometry only`);
    else if (this.t > 5 && age > 15) bump('warn', `${age.toFixed(0)} s since the last accepted outside measurement`);
    return { level, reasons, sig3, tol, rej, age, anyData: rec.length > 0 };
  }

  activeFaults() { return [...this.faults.entries()].filter(([, u]) => u > this.t).map(([type, u]) => ({ type, left: u - this.t })); }
}
