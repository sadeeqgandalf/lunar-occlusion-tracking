// The Coach: turns raw mission state into plain-language situation awareness + a suggested next action.
// It reads only what a real operator could know (health, faults, filter internals) until the debrief.

const FAULT_NOTES = {
  gyroStep: ['Gyro bias jump', 'The gyro now reads a constant turn rate that is not real (thermal shock or a radiation upset). Heading drifts a little more every second.', 'Dead reckoning keeps drifting. Filters with a bias state re-learn it once outside measurements arrive.'],
  slip: ['Wheel slip', 'The wheels spin on loose ground: odometry says you travelled farther than you did.', 'DR error climbs. The EKF/UKF absorb it into their slip state, so their 3σ bubble widens first, then recovers.'],
  blackout: ['Beacon blackout', 'Dust blocks the beacon sensor, so there are no outside measurements.', 'Every filter is now dead-reckoning and the 3σ bubble grows. When Health turns red, stop and wait; do not drive on.'],
  outliers: ['Outlier burst', 'Some readings are wildly wrong but still claim normal precision (multipath, ghosts, spoofing).', 'Gated filters reject them (Rej column rises). The particle filter\'s heavy tails help. A filter with no gate gets dragged off.'],
  orbiterLoss: ['Missed orbiter pass', 'The overhead relay pass that gives an absolute position fix did not happen.', 'Fixes stop. Beacons are the only correction left, so keep to areas where beacons are visible.'],
  gnssJam: ['GNSS jammed', 'Satellite navigation is denied. Only radar beacons and inertial navigation remain.', 'DR drifts by kilometres. Filters with beacon updates hold, but only while a beacon is in range.'],
  wind: ['Jet-stream gust', 'Ground track no longer follows heading × airspeed. Inertial + airspeed navigation cannot see wind directly.', 'The filters\' wind states absorb it over about 20 s; DR simply drifts. Watch the 3σ bubble swell, then settle.'],
  imuBias: ['Accelerometer bias', 'The accelerometer reads thrust plus a constant error, so velocity error grows with time.', 'Filters with a bias state recover through lidar and fix updates; DR does not.'],
  sunBlind: ['Lidar sun-blind', 'The sun dazzles the lidar. No relative range or bearing to the target.', 'You are left with sparse position fixes. Do not close in while Health is red.'],
  fixLoss: ['Fix lost', 'The ground-tracked position fix is unavailable.', 'Only the lidar corrects position. Fine near the target, weak far from it.'],
};

const rank = (m) => [...m.fs.keys()].map((id) => ({ id, s: m.summary(id) })).filter((r) => r.s).sort((a, b) => a.s.rmse - b.s.rmse);
const lbl = (m, id) => m.fs.get(id).Cls.label;

export function coach(m) {
  const P = m.platform, h = m.health(), u = P.unit;
  const idle = !m.waypoints.length && !m.manualActive && !m.dwell;

  if (m.status !== 'running') {
    const r = rank(m), best = r[0], mine = r.find((x) => x.id === m.primary);
    const hits = Math.round((100 - m.hull) / P.hazard.damage);
    let body = m.status === 'complete' ? `All objectives done. Score ${m.score}.` : `${m.failReason}.`;
    if (hits) body += ` The vehicle hit a hazard ${hits}× because its position belief was off.`;
    if (best && mine) body += ` Your primary (${lbl(m, m.primary)}) averaged ${fmt(mine.s.rmse)} ${u} error; best was ${lbl(m, best.id)} at ${fmt(best.s.rmse)} ${u}.`;
    return { tone: m.status === 'complete' ? 'good' : 'bad', title: m.status === 'complete' ? 'Debrief · mission complete' : 'Debrief · mission failed', body, cta: [['Fly it again', 'reset'], ['What do these numbers mean?', 'learn']] };
  }
  if (m.t < 2 && idle) return { tone: 'info', title: 'Vehicle parked. Nothing moves until you command it', body: `Click the map to add a waypoint, press <kbd>P</kbd> to uplink the nominal plan, or drive with <kbd>WASD</kbd>. The autopilot steers using the PRIMARY filter's belief, not the truth.`, cta: [['▶ Run the nominal plan', 'start']] };
  if (m.safeHold) return { tone: 'warn', title: 'SAFE-HOLD · the autopilot refused to drive', body: `The 3σ pose bubble overlaps a hazard ahead, so it cannot rule out hitting it. Real vehicles do this. Options: wait for a beacon/fix to shrink the bubble, retarget around the hazard, or untick safing (you accept the risk).`, cta: [['Clear route', 'stop']] };

  const faults = m.activeFaults();
  if (faults.length) {
    const [name, what, watch] = FAULT_NOTES[faults[0].type] || [faults[0].type, '', ''];
    return { tone: 'warn', title: `Active fault · ${name} (${faults[0].left.toFixed(0)} s)`, body: `${what}<br><b>Watch:</b> ${watch}` };
  }
  if (h?.level === 'bad') return { tone: 'bad', title: 'Do not trust this navigation solution', body: `${h.reasons.map((r) => `• ${r}`).join('<br>')}<br><b>Real-life move:</b> stop the vehicle, let it re-localise (beacon/fix), then continue. Driving on a solution like this is how vehicles get lost.`, cta: [['Stop vehicle', 'stop']] };
  if (h?.level === 'warn') return { tone: 'warn', title: 'Navigation degrading', body: `${h.reasons.map((r) => `• ${r}`).join('<br>')}<br>Keep clear of hazards until this recovers.` };
  if (m.primary === 'dr' && m.t > 8) return { tone: 'warn', title: 'Primary is dead reckoning', body: 'It never corrects itself, so its error only grows. The autopilot is following a belief that gets worse every second. Make the EKF primary and compare.', cta: [['Make EKF primary', 'ekf']] };
  if (idle && m.t >= 2) return { tone: 'info', title: 'Waypoint queue empty', body: 'The vehicle is holding. Add waypoints, or press <kbd>P</kbd> to uplink the nominal plan again.', cta: [['Upload plan', 'start']] };
  return { tone: 'good', title: 'Nominal · what to watch', body: `Primary error should stay inside its own 3σ bound (second chart). ANEES should sit near ${m.model.neesIdx.length}. Try a fault from the left panel to see what each filter does when the world stops matching its model.` };
}

const fmt = (v) => (v >= 1000 ? (v / 1000).toFixed(1) + 'k' : v >= 10 ? v.toFixed(0) : v.toFixed(2));

export function healthHtml(m) {
  const h = m.health();
  if (!h) return '';
  const label = { good: 'NAV NOMINAL', warn: 'NAV DEGRADED', bad: 'DO NOT TRUST' }[h.level];
  const fmtv = (v) => (v >= 1000 ? (v / 1000).toFixed(1) + 'k' : v.toFixed(v < 10 ? 1 : 0));
  return `<div class="hpill ${h.level}">${label}</div>
    <div class="hrow"><span>3σ bubble</span><b>${fmtv(h.sig3)} ${m.platform.unit}</b><span>needs ≤ ${fmtv(h.tol)}</span></div>
    <div class="hrow"><span>Last outside fix</span><b>${h.age < 1 && m.t > 5 ? 'now' : h.age.toFixed(0) + ' s ago'}</b><span>${h.anyData ? Math.round(h.rej * 100) + '% rejected' : 'none yet'}</span></div>`;
}

export const BRIEF = `
<h2>You are the navigation operator</h2>
<p class="lede">The vehicle is out of sight. You never see where it <i>really</i> is. You see only what its estimators <i>believe</i>, and how sure they are. This console teaches you to judge whether to trust that belief.</p>
<div class="three">
  <div><b>1 · The map</b><p>The <span class="k white">white outline</span> is ground truth. It exists only in training. Colored shapes are each filter's belief, and the ring around each is its own <b>95% uncertainty</b>. A good filter's ring contains the white outline.</p></div>
  <div><b>2 · Navigation Health</b><p>Top-left. It uses <b>only onboard information</b>: how big the uncertainty bubble is versus what the task needs, how many measurements are being rejected, and how long since an outside fix. This is what you would have in real life.</p></div>
  <div><b>3 · The scoreboard</b><p><b>Err vs 3σ</b>: error should stay inside the bound. <b>ANEES</b> should sit near the degrees of freedom: red means the filter is overconfident, the dangerous direction.</p></div>
</div>
<p class="lede"><b>Your loop:</b> command → watch Health → inject a fault → decide: <i>continue, slow down, stop, or re-localise.</i> The Coach (bottom of the map) narrates what is happening and why.</p>
<div class="btns"><button id="briefTour" class="accent big">▶ Guided tour (2 min)</button><button id="briefGo" class="big">Start the mission</button><button id="briefSkip" class="big">Explore on my own</button></div>
<p class="fine">Press <b>? Learn</b> any time for guided experiments. Press <kbd>T</kbd> to hide ground truth and fly blind, like real operations.</p>`;
