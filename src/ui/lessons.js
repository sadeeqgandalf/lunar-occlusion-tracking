export const LESSONS = `
<h2>MISSION BRIEFING</h2>
<p>You are flying a vehicle you can't see directly. Everything you know comes from noisy sensors, and every estimator on the right is trying to answer one question: <i>where am I, and how sure am I?</i> The white outline is ground truth (untick it for blind ops). Coloured shapes are beliefs, and each ellipse is that filter's own 95% uncertainty.</p>

<h4>1 · Watch dead reckoning fail</h4>
<p>Make <b>Dead Reckoning</b> the primary, press <kbd>P</kbd> to uplink the plan and set speed to 8×. Its ellipse swells because it knows it is drifting, then the rover follows its wrong belief into a hazard. Now switch the primary to <b>EKF</b> and reset.</p>

<h4>2 · Is a filter honest? (consistency)</h4>
<p>Accuracy isn't enough. A filter must know how wrong it is. The scoreboard's <b>ANEES</b> is the average normalised estimation error squared, which should be close to the degrees of freedom. Red means overconfident (dangerous), amber means conservative. Drag <b>Process noise ×</b> down and watch the EKF turn red. That is the classic mistake of trusting your model too much.</p>

<h4>3 · Gate lock-out</h4>
<p>Pick <b>Sol 140 · Sand Trap</b>. Wheel slip makes odometry lie while the filter still believes it. Beacon measurements start to look like outliers and get rejected (<b>Rej</b> column). Lower the <b>NIS gate</b> to make it worse; the built-in recovery inflates covariance after a streak of rejections.</p>

<h4>4 · What the gate buys you</h4>
<p>Pick <b>Sandbox</b>, press <kbd>P</kbd>, then inject <b>Beacon outlier burst</b> a few times. The EKF and UKF reject the ghost readings (the <b>Rej</b> column climbs) and stay accurate and honest. Now drag the EKF's <b>NIS gate</b> to its maximum and repeat. Measured over four seeds, RMSE rose from 0.12–0.79 m to 0.75–1.17 m and ANEES jumped from about 1 to 38–86: without the gate the filter swallows bad data and becomes badly overconfident. The particle filter has no gate; its heavy-tailed likelihood copes about as well as the gated filters here.</p>

<h4>5 · Latency</h4>
<p>Slide <b>Uplink latency</b> to 5 s and drive with <kbd>WASD</kbd>. Commands arrive late. This is why real rovers run on-board autonomy plus waypoints, which is exactly what the autopilot does using its own estimate.</p>

<h4>6 · Other vehicles</h4>
<p>Switch <b>Platform</b> to the <b>Fighter Jet</b> (GNSS-denied navigation with beacons, INS drift and unmodelled wind) or the <b>Spacecraft</b> (Clohessy-Wiltshire relative motion, lidar docking). The same EKF/UKF/PF classes drive all three. Only the vehicle model changes.</p>

<h4>7 · Write your own filter</h4>
<p>Open the <b>Filter Lab</b> at the bottom, edit the starter complementary filter and compile. It joins the race live, using the same sensor stream and the same scoreboard.</p>

<h4>Plain-English glossary</h4>
<dl class="gloss">
<dt>State estimation</dt><dd>Working out where a vehicle is (and how fast, which way, how its sensors are drifting) from noisy measurements. Nothing is measured perfectly, so the answer is a best guess <i>plus a statement of how unsure it is</i>.</dd>
<dt>Dead reckoning</dt><dd>Adding up wheel/gyro/accelerometer readings from a known start. It never checks itself, so small errors accumulate forever.</dd>
<dt>Filter (EKF, UKF, particle)</dt><dd>A recipe that repeatedly <b>predicts</b> where the vehicle went, then <b>corrects</b> that guess with outside measurements (beacons, GNSS, lidar), weighing each by how trustworthy it is.</dd>
<dt>Uncertainty ring / covariance</dt><dd>The filter's own statement of how wrong it might be. The ring is its 95% region. Bigger ring = less sure.</dd>
<dt>3σ</dt><dd>"Three standard deviations": a conservative bound on the error. Real error should almost always sit inside it.</dd>
<dt>Innovation / gate / rejection</dt><dd>The gap between what a measurement says and what the filter expected. If the gap is implausibly large the measurement is <b>rejected</b> (the gate). This protects against ghosts and outliers.</dd>
<dt>Consistency, NEES, ANEES</dt><dd>Does the filter's claimed uncertainty match its real error? ANEES near the degrees of freedom means yes. Much higher = <b>overconfident</b> (dangerous). Much lower = needlessly cautious.</dd>
<dt>Bias / slip / wind</dt><dd>Hidden quantities that corrupt the dead-reckoned path (gyro offset, wheel slip, crosswind). Good filters estimate them as extra states instead of ignoring them.</dd>
<dt>Navigation Health</dt><dd>An integrity check built only from what the vehicle itself knows: uncertainty size vs the task's needs, rejection rate, time since an outside fix.</dd>
<dt>Safing / SAFE-HOLD</dt><dd>The autopilot refuses to continue when the uncertainty bubble overlaps a hazard ahead, because it can no longer promise to avoid it.</dd>
</dl>
<h4>Theory</h4>
<p>Implemented from <i>Probabilistic Robotics</i> (Thrun, Burgard, Fox): velocity motion model (Ch. 5, eq. 5.9), EKF (Table 3.3), UKF (Table 3.4, augmented), particle filter with low-variance resampling (Tables 4.3 / 4.4) and Augmented MCL recovery (Table 8.3). See <code>docs/THEORY.md</code>.</p>
<p><b>Keys:</b> <kbd>Space</kbd> pause · <kbd>R</kbd> reset · <kbd>P</kbd> plan · <kbd>Esc</kbd> stop · <kbd>T</kbd> truth · <kbd>F</kbd> follow · <kbd>WASD</kbd> drive · right-click undo leg</p>`;
