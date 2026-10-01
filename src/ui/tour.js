// Guided tour: spotlights one control at a time, explains it in plain language, and (optionally) does the
// action for you so you see the consequence. Starts from a fresh, paused rover mission so it is repeatable.

const STEPS = (api) => [
  { at: '#mapwrap', title: '1 · The map: belief vs truth',
    text: `The <b>white outline</b> is where the vehicle <i>really</i> is. In real life you never see it. The <b>coloured shapes</b> are what each estimator <i>believes</i>, and the ring around each is how sure it is (95%). An honest estimator's ring contains the white outline.` },
  { at: '#health', title: '2 · Navigation Health',
    text: `The one verdict you would have in real life. It never looks at ground truth. It checks: is the uncertainty bubble small enough for the task? Are measurements being rejected? How long since an outside fix? <b>Green = carry on, amber = careful, red = stop.</b>` },
  { at: '#btnPlan', title: '3 · Give it a job',
    text: `The vehicle is parked until you command it. <b>Upload plan</b> sends the nominal route (or click the map to add your own waypoints). The autopilot steers using the <b>primary</b> filter's belief, not the truth.`,
    action: { label: '▶ Upload plan and run at 4×', run: () => api.start() } },
  { at: '#score-table', title: '4 · The scoreboard',
    text: `Give it ~20 s. <b>DR</b> (dead reckoning) only integrates wheel and gyro data and never corrects itself, so its error keeps growing. <b>EKF / UKF / PF</b> also use beacons and fixes, so they stay close. <b>Err</b> should stay under <b>3σ</b>, the filter's own claim about its error. <b>ANEES</b> near the degrees of freedom means it is honest.` },
  { at: '#faults', title: '5 · Break something',
    text: `Real sensors fail. Inject a <b>beacon blackout</b>: the outside measurements vanish and every filter must coast on inertial data alone. Watch the rings grow.`,
    action: { label: '⚡ Inject beacon blackout', run: () => api.fault('blackout') } },
  { at: '#coach', title: '6 · The Coach',
    text: `It tells you what is happening and why, in plain words, and what to do about it. Watch it together with <b>Navigation Health</b>: as the bubble grows, Health goes amber, then red. <b>Red means a real operator stops the vehicle.</b>` },
  { at: '#stopTop', title: '7 · STOP',
    text: `<b>■ STOP</b> halts the vehicle: it clears waypoints and manual drive (also <kbd>Esc</kbd>). A rover stops; a jet cannot, so it enters a holding orbit; a spacecraft station-keeps. <b>Pause</b> is different: it freezes the <i>simulation clock</i>, the vehicle isn't doing anything about it.`,
    action: { label: '■ Stop the vehicle now', run: () => api.stop() } },
  { at: '#optTruth', title: '8 · Fly blind',
    text: `Untick this (or press <kbd>T</kbd>) to hide ground truth. Now you only have beliefs and their uncertainty, exactly like real operations. Try judging the situation from Health and the rings alone.`,
    action: { label: '👁 Hide ground truth', run: () => api.truth(false) } },
  { at: '#filters', title: '9 · Who is steering?',
    text: `The <b>PRIMARY</b> filter drives the autopilot. Make <b>Dead Reckoning</b> primary and re-run the plan: the vehicle faithfully follows a belief that gets worse every second. That is how a bad navigation solution loses a real vehicle.`,
    action: { label: '🧭 Make DR primary and replan', run: () => api.dr() } },
  { at: '#labToggle', title: '10 · Write your own estimator',
    text: `The <b>Filter Lab</b> lets you write an estimator in JavaScript. It joins the race on the same sensor stream and is scored the same way: low error, honest uncertainty. The same code layout applies to the jet and spacecraft.` },
  { at: null, title: "You've seen the loop",
    text: `<b>Command → watch Health → something breaks → decide: continue, slow, stop, or re-localise.</b> Press <b>? Learn</b> for guided experiments and a plain-English glossary, or read <code>docs/GUIDE.md</code>. Press <b>Reset</b> to start fresh.` },
];

export function runTour(api, makeSteps = STEPS) {
  api.prepare(); // fresh, paused starting state
  const steps = makeSteps(api);
  const root = document.createElement('div'); root.id = 'tour';
  root.innerHTML = '<div class="tour-hole"></div><div class="tour-card"></div>';
  document.body.appendChild(root);
  const hole = root.querySelector('.tour-hole'), card = root.querySelector('.tour-card');
  let i = 0;

  const place = () => {
    const s = steps[i], el = s.at && document.querySelector(s.at);
    const VW = innerWidth, VH = innerHeight, CW = Math.min(360, VW - 24);
    let rect = null;
    if (el) { el.scrollIntoView?.({ block: 'nearest' }); const r = el.getBoundingClientRect(); rect = { l: r.left, t: r.top, w: r.width, h: r.height }; }
    if (rect) { hole.style.cssText = `display:block;left:${rect.l - 6}px;top:${rect.t - 6}px;width:${rect.w + 12}px;height:${rect.h + 12}px`; } else hole.style.cssText = 'display:none';
    card.style.width = CW + 'px';
    let x, y; const ch = card.offsetHeight || 220;
    if (!rect) { x = (VW - CW) / 2; y = (VH - ch) / 2; }
    else if (rect.w > VW * 0.4) { x = rect.l + 24; y = rect.t > VH * 0.5 ? rect.t - ch - 16 : rect.t + 52; } // wide target: above it if docked low (Coach), else inside it (map)
    else if (rect.l + rect.w + CW + 20 < VW) { x = rect.l + rect.w + 16; y = rect.t; }          // room on the right
    else if (rect.l - CW - 20 > 0) { x = rect.l - CW - 16; y = rect.t; }                        // room on the left
    else { x = Math.min(VW - CW - 12, Math.max(12, rect.l)); y = rect.t + rect.h + 14; }        // below
    card.style.left = Math.max(12, Math.min(VW - CW - 12, x)) + 'px';
    card.style.top = Math.max(12, Math.min(VH - ch - 12, y)) + 'px';
  };
  const render = () => {
    const s = steps[i];
    card.innerHTML = `<div class="tour-n">${i + 1} / ${steps.length}</div><b>${s.title}</b><p>${s.text}</p>
      ${s.action ? `<button class="accent" data-t="act">${s.action.label}</button>` : ''}
      <div class="tour-nav"><button data-t="skip">Skip tour</button><span></span>${i ? '<button data-t="back">← Back</button>' : ''}<button class="accent" data-t="next">${i === steps.length - 1 ? 'Finish' : 'Next →'}</button></div>`;
    place();
  };
  const end = () => { root.remove(); removeEventListener('resize', place); removeEventListener('keydown', key, true); };
  const key = (e) => { if (e.key === 'Escape') { e.stopPropagation(); end(); } else if (e.key === 'ArrowRight') { e.stopPropagation(); card.querySelector('[data-t=next]')?.click(); } };
  card.onclick = (e) => {
    const t = e.target.dataset.t; if (!t) return;
    if (t === 'act') { steps[i].action.run(); e.target.textContent = '✔ Done'; e.target.disabled = true; }
    else if (t === 'next') { if (i === steps.length - 1) end(); else { i++; render(); } }
    else if (t === 'back') { i--; render(); }
    else if (t === 'skip') end();
  };
  addEventListener('resize', place); addEventListener('keydown', key, true);
  render();
}
