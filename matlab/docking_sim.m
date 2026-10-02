function out = docking_sim(opts)
%DOCKING_SIM  RPO docking approach to Gateway, flown on an EKF estimate. MATLAB port of the web lab.
%
%   docking_sim                       % run and watch it live (4x real time)
%   docking_sim(Speed=16)             % faster playback
%   docking_sim(Animate=false)        % just run, then show the result plots
%   docking_sim(Video="docking.mp4")  % also record the animation to a video file
%   out = docking_sim(Seed=3);        % different noise; out holds the full history
%
% Same model and numbers as the browser lab (src/models/spacecraft.js, src/platforms/spacecraft.js,
% src/filters/ekf.js), so the two can be compared directly:
%   * Truth: in-plane Clohessy-Wiltshire (Hill) relative motion, exact state transition over each step,
%     thrust as zero-order-hold acceleration. Mean motion n = 0.01 rad/s (LEO is ~1.1e-3; time-compressed
%     so the orbital coupling shows up within minutes, as in the lab).
%   * Sensors: accelerometer (with bias) every step; lidar range + bearing to the target every 0.5 s;
%     a ground-tracked position fix for 2 s out of every 20 s.
%   * Estimator: 6-state EKF [x_R y_T vx vy b_ax b_ay] (Thrun, Probabilistic Robotics, Table 3.3),
%     Joseph-form update, chi-square innovation gate at 99%. Default tuning as the lab: Q x4, sigma_R x1.5.
%   * Guidance: velocity-shaped approach on the ESTIMATE with CW feed-forward; holds at the 150 m and 60 m
%     gates, then docks. Gate and dock checks use the TRUE state, as a real inspection would.
%
% Axes follow the RPO convention: along-track (V-bar) horizontal, radial (R-bar) vertical, Gateway at 0.

arguments
    opts.Seed (1,1) double = 7
    opts.Animate (1,1) logical = true
    opts.Speed (1,1) double {mustBePositive} = 4     % simulated seconds per real second while watching
    opts.Video (1,1) string = ""                      % e.g. "docking.mp4"
    opts.QScale (1,1) double = 4
    opts.RScale (1,1) double = 1.5
    opts.Gate (1,1) double = 9.21                     % chi2inv(0.99, 2)
end

%% ---------------------------------------------------------------- parameters (identical to the lab)
n = 0.01; dt = 0.05; sensePeriod = 0.5; aMax = 0.2;
sigAcc = 0.002; sigBiasWalk = 0.0005;
lidarRange = 320; sigR = 0.3; sigB = 0.004;
sigFix = 3; fixPeriod = 20; fixWindow = 2;
holdTime = 3; tMax = 900;

% route: two transit legs, then the 150 m gate, the 60 m gate and the docking port
route = [18 -290; 10 -220; 0 -150; 0 -60; 0 -6];
kind  = ["start" "goto" "gate" "gate" "dock"];
names = ["Start" "Transit" "Gate 1 (150 m)" "Gate 2 (60 m)" "Dock"];

rng(opts.Seed);
P0 = diag([4 4 0.01 0.01 9e-6 9e-6]);
xNom = [route(1,:)'; 0; 0; 0; 0];
x  = xNom + chol(P0, 'lower') * randn(6,1);   % truth drawn from the prior the filter is given (NEES meaningful)
xh = xNom; P = P0;
bias = x(5:6);

[Phi, Bm] = cw(n, dt);
G = eye(6); G(1:4,1:4) = Phi; G(1:4,5:6) = -Bm;
V = zeros(6,4); V(1:4,1:2) = Bm; V(5,3) = 1; V(6,4) = 1;
W = diag([sigAcc^2 sigAcc^2 sigBiasWalk^2*dt sigBiasWalk^2*dt]) * opts.QScale;
Q = V * W * V';

%% ---------------------------------------------------------------- history
N = ceil(tMax / dt) + 1;
H.t = nan(N,1); H.x = nan(N,6); H.xh = nan(N,6); H.P = nan(N,6,6); H.nees = nan(N,1);
H.lidar = false(N,1); H.fix = false(N,1); H.rej = false(N,1);
evlog = strings(0,1);

%% ---------------------------------------------------------------- figure
fig = []; vw = []; frameSize = [];
if opts.Animate || opts.Video ~= ""
    [fig, A] = setupFigure(route, names, kind);
    if opts.Video ~= ""
        vw = VideoWriter(opts.Video, 'MPEG-4'); vw.FrameRate = 30; vw.Quality = 90; open(vw);
    end
end

%% ---------------------------------------------------------------- run
k = 2; holdLeft = NaN; t = 0; i = 0; docked = false; nextSense = 0; nextDraw = 0; wall0 = tic;
while t < tMax
    i = i + 1;
    % ---- guidance on the estimate
    wp = route(k,:)';
    if isnan(holdLeft)
        [cmd, dist, arrived] = autopilot(xh, wp, kind(k), n, aMax);
        if arrived && kind(k) ~= "goto"
            holdLeft = holdTime;
            evlog(end+1) = sprintf('T+%5.1f s  arrived at %s, station-keeping %d s', t, names(k), holdTime); %#ok<AGROW>
        elseif arrived
            k = k + 1;
        end
    else
        cmd = holdCmd(xh, n, aMax);
        holdLeft = holdLeft - dt;
        if holdLeft <= 0
            [ok, txt] = checkTruth(x, wp, kind(k));
            evlog(end+1) = sprintf('T+%5.1f s  %s: %s (%s)', t, names(k), passfail(ok), txt); %#ok<AGROW>
            if kind(k) == "dock", docked = true; end
            holdLeft = NaN; k = k + 1;
        end
    end

    % ---- truth: thrust is exactly the command; the accelerometer reads thrust + bias + noise
    imu = sigAcc * randn(2,1);
    u = cmd + bias + imu;
    x(1:4) = Phi * x(1:4) + Bm * cmd;
    t = t + dt;

    % ---- EKF predict with the accelerometer reading
    xh(1:4) = Phi * xh(1:4) + Bm * (u - xh(5:6));
    P = G * P * G' + Q;

    % ---- EKF update
    lid = false; fx = false; rej = false;
    if t >= nextSense - 1e-9
        nextSense = t + sensePeriod;
        r = hypot(x(1), x(2));
        if r < lidarRange && r > 1
            z = [r + sigR*randn; wrapPi(atan2(-x(2), -x(1)) + sigB*randn)];
            q = xh(1)^2 + xh(2)^2; rh = sqrt(q);
            zh = [rh; atan2(-xh(2), -xh(1))];
            Hm = [xh(1)/rh xh(2)/rh 0 0 0 0; -xh(2)/q xh(1)/q 0 0 0 0];
            R = diag([(sigR*opts.RScale)^2 (sigB*opts.RScale)^2]);
            y = z - zh; y(2) = wrapPi(y(2));
            [xh, P, acc] = kfUpdate(xh, P, y, Hm, R, opts.Gate);
            lid = acc; rej = ~acc;
        end
        if mod(t, fixPeriod) < fixWindow
            z = x(1:2) + sigFix*randn(2,1);
            Hm = [eye(2) zeros(2,4)];
            [xh, P, acc] = kfUpdate(xh, P, z - xh(1:2), Hm, (sigFix*opts.RScale)^2*eye(2), opts.Gate);
            fx = acc; rej = rej || ~acc;
        end
    end

    % ---- record
    e = x(1:4) - xh(1:4);
    H.t(i) = t; H.x(i,:) = x; H.xh(i,:) = xh; H.P(i,:,:) = P; H.nees(i) = e' / P(1:4,1:4) * e;
    H.lidar(i) = lid; H.fix(i) = fx; H.rej(i) = rej;

    % ---- draw
    if ~isempty(fig) && t >= nextDraw
        nextDraw = t + 0.5;
        if ~isvalid(fig), fig = []; else
            recent = max(1, i-9):i;                % measurements in the last 0.5 s, so the indicators do not flicker
            drawFrame(A, H, i, route, k, kind, names, holdLeft, cmd, any(H.lidar(recent)), any(H.fix(recent)));
            if ~isempty(vw)
                fr = getframe(fig);
                if isempty(frameSize), frameSize = size(fr.cdata, [1 2]); end
                if ~isequal(size(fr.cdata, [1 2]), frameSize), fr.cdata = imresize(fr.cdata, frameSize); end
                writeVideo(vw, fr);
            end
            if opts.Animate
                lag = t / opts.Speed - toc(wall0);
                if lag > 0, pause(lag); else, drawnow limitrate; end
            end
        end
    end
    if docked || k > numel(kind), break; end
end
if ~isempty(vw), close(vw); end

%% ---------------------------------------------------------------- results
H = trimHistory(H, i);
err = vecnorm(H.x(:,1:2) - H.xh(:,1:2), 2, 2);
sig3 = 3 * sqrt(H.P(:,1,1) + H.P(:,2,2));
out = struct('t', H.t, 'truth', H.x, 'estimate', H.xh, 'P', H.P, 'nees', H.nees, ...
    'docked', docked, 'log', evlog, 'rmse', sqrt(mean(err.^2)), 'anees', mean(H.nees, 'omitnan'), ...
    'inside3sigma', mean(err <= sig3));

fprintf('\n%s\n', strjoin(evlog, newline));
fprintf(['\nDocked: %s after %.1f s  |  position RMSE %.2f m  |  ANEES %.2f (4 = honest, lower = cautious, higher = overconfident)  |' ...
    '  error inside 3-sigma %.0f%% of the time\n'], passfail(docked), H.t(end), out.rmse, out.anees, 100*out.inside3sigma);

if opts.Animate || opts.Video ~= ""
    resultsFigure(H, err, sig3);
end
end

%% =================================================================== dynamics, guidance, filter
function [Phi, Bm] = cw(n, dt)
% exact Clohessy-Wiltshire transition for [x_R y_T vx vy] over dt, plus zero-order-hold acceleration input
s = sin(n*dt); c = cos(n*dt);
Phi = [4-3*c        0  s/n          2/n*(1-c)
       6*(s-n*dt)   1  2/n*(c-1)    (4*s-3*n*dt)/n
       3*n*s        0  c            2*s
       6*n*(c-1)    0  -2*s         4*c-3];
Bm = [0.5*dt^2 0; 0 0.5*dt^2; dt 0; 0 dt];
end

function f = ff(n, x)
% feed-forward that cancels the CW coupling terms
f = [-(3*n^2*x(1) + 2*n*x(4)); 2*n*x(3)];
end

function [cmd, dist, arrived] = autopilot(xh, wp, kind, n, aMax)
d = wp - xh(1:2); dist = norm(d); speed = norm(xh(3:4));
vmax = min(2.5, 0.08*dist);                       % speed tapers with range for a soft arrival
kk = 0; if dist > 1e-6, kk = vmax/dist; end
cmd = max(-aMax, min(aMax, 0.35*(d*kk - xh(3:4)) + ff(n, xh)));
tol = 5; if kind ~= "goto", tol = 1.2; end
arrived = dist < tol && speed < 0.6;
end

function cmd = holdCmd(xh, n, aMax)
cmd = max(-aMax, min(aMax, -0.5*xh(3:4) + ff(n, xh)));
end

function [ok, txt] = checkTruth(x, wp, kind)
d = norm(x(1:2) - wp); v = norm(x(3:4));
maxD = 4; maxV = 0.6; if kind == "dock", maxD = 2; maxV = 0.4; end
ok = d <= maxD && v <= maxV;
txt = sprintf('%.1f m off at %.2f m/s, limit %g m / %g m/s', d, v, maxD, maxV);
end

function [x, P, accepted] = kfUpdate(x, P, y, Hm, R, gate)
S = Hm*P*Hm' + R; nis = y' / S * y;
accepted = nis <= gate;                            % chi-square gate rejects outliers
if ~accepted, return; end
K = P*Hm' / S; x = x + K*y;
IKH = eye(numel(x)) - K*Hm;
P = IKH*P*IKH' + K*R*K'; P = (P + P')/2;           % Joseph form keeps P symmetric positive definite
end

function a = wrapPi(a)
a = mod(a + pi, 2*pi) - pi;
end

function s = passfail(ok)
if ok, s = "PASS"; else, s = "FAIL"; end
end

function H = trimHistory(H, i)
f = fieldnames(H);
for j = 1:numel(f)
    v = H.(f{j});
    if ndims(v) == 3, H.(f{j}) = v(1:i,:,:); else, H.(f{j}) = v(1:i,:); end
end
end

%% =================================================================== drawing
function [fig, A] = setupFigure(route, names, kind)
fig = figure('Name', 'RPO docking · EKF', 'Color', [0.04 0.05 0.08], 'Position', [80 80 1280 720]);
A.ax = axes(fig, 'Position', [0.05 0.1 0.62 0.82], 'Color', [0 0 0], 'XColor', [0.6 0.65 0.75], ...
    'YColor', [0.6 0.65 0.75], 'GridColor', [0.3 0.35 0.45], 'FontSize', 10);
hold(A.ax, 'on'); grid(A.ax, 'on'); axis(A.ax, 'equal');
xlabel(A.ax, 'along-track y_T, V-bar [m]  (Gateway at 0)'); ylabel(A.ax, 'radial x_R, R-bar [m]');
title(A.ax, 'Approach seen in the target''s orbit frame', 'Color', [0.9 0.92 0.96]);
xlim(A.ax, [-320 30]); ylim(A.ax, [-70 70]);
% stars
rs = RandStream('mt19937ar', 'Seed', 1);
scatter(A.ax, -320 + 350*rand(rs,200,1), -70 + 140*rand(rs,200,1), 2, [0.7 0.7 0.8], 'filled', 'HandleVisibility', 'off');
% Gateway: truss, modules and solar wings
patch(A.ax, [-4 4 4 -4], [-1.2 -1.2 1.2 1.2], [0.75 0.78 0.82], 'EdgeColor', 'none');
patch(A.ax, [-2.5 2.5 2.5 -2.5], [1.2 1.2 14 14], [0.15 0.25 0.55], 'EdgeColor', [0.5 0.6 0.9]);
patch(A.ax, [-2.5 2.5 2.5 -2.5], [-1.2 -1.2 -14 -14], [0.15 0.25 0.55], 'EdgeColor', [0.5 0.6 0.9]);
text(A.ax, 6, 16, 'Gateway', 'Color', [0.55 0.8 1], 'FontWeight', 'bold');
% route and gates
plot(A.ax, route(:,2), route(:,1), ':', 'Color', [0.5 0.5 0.55]);
for j = 2:numel(kind)
    if kind(j) == "goto", continue; end
    plot(A.ax, route(j,2), route(j,1), 'o', 'MarkerSize', 12, 'LineWidth', 1.5, 'Color', [1 0.82 0.4]);
    text(A.ax, route(j,2), route(j,1) - 7, names(j), 'Color', [1 0.82 0.4], 'HorizontalAlignment', 'center');
end
A.trailT = plot(A.ax, nan, nan, '-', 'Color', [1 1 1 0.8], 'LineWidth', 1.2);
A.trailE = plot(A.ax, nan, nan, '-', 'Color', [0.3 0.79 0.94], 'LineWidth', 1);
A.ell = plot(A.ax, nan, nan, '-', 'Color', [0.3 0.79 0.94], 'LineWidth', 1.4);
A.ray = plot(A.ax, nan, nan, '-', 'Color', [1 0.71 0.33 0.7]);
A.fix = plot(A.ax, nan, nan, 's', 'MarkerSize', 9, 'Color', [0.74 0.55 1], 'LineWidth', 1.5);
A.ship = patch(A.ax, nan, nan, [0.95 0.95 0.97], 'EdgeColor', [0.4 0.4 0.45]);
A.est = plot(A.ax, nan, nan, '+', 'MarkerSize', 10, 'LineWidth', 1.6, 'Color', [0.3 0.79 0.94]);
A.thr = quiver(A.ax, nan, nan, nan, nan, 0, 'Color', [1 0.45 0.3], 'LineWidth', 1.6, 'MaxHeadSize', 0.8);
legend(A.ax, [A.trailT A.trailE A.ell A.ray A.fix], {'truth', 'EKF estimate', '3\sigma uncertainty', ...
    'lidar ray', 'position fix'}, 'TextColor', [0.85 0.88 0.95], 'Color', [0.08 0.09 0.13], 'Location', 'southwest');
% side panel
A.hud = annotation(fig, 'textbox', [0.69 0.55 0.29 0.37], 'Color', [0.88 0.9 0.95], 'FontName', 'Menlo', ...
    'FontSize', 11, 'EdgeColor', [0.25 0.3 0.4], 'BackgroundColor', [0.07 0.08 0.12], 'Interpreter', 'none');
A.e = axes(fig, 'Position', [0.71 0.1 0.27 0.36], 'Color', [0.05 0.06 0.09], 'XColor', [0.6 0.65 0.75], ...
    'YColor', [0.6 0.65 0.75], 'FontSize', 9); hold(A.e, 'on'); grid(A.e, 'on');
title(A.e, 'position error vs 3\sigma bound', 'Color', [0.88 0.9 0.95]); xlabel(A.e, 'time [s]'); ylabel(A.e, 'm');
A.eErr = plot(A.e, nan, nan, 'Color', [0.3 0.79 0.94], 'LineWidth', 1.3);
A.eSig = plot(A.e, nan, nan, '--', 'Color', [1 0.82 0.4]);
end

function drawFrame(A, H, i, route, k, kind, names, holdLeft, cmd, lid, fx)
x = H.x(i,:); xh = H.xh(i,:); P = squeeze(H.P(i,:,:)); t = H.t(i);
set(A.trailT, 'XData', H.x(1:i,2), 'YData', H.x(1:i,1));
set(A.trailE, 'XData', H.xh(1:i,2), 'YData', H.xh(1:i,1));
% 3-sigma ellipse of the position estimate (plotted as [y_T, x_R])
[Ev, D] = eig(P([2 1],[2 1])); th = linspace(0, 2*pi, 60);
c = Ev * sqrt(max(D, 0)) * [cos(th); sin(th)] * sqrt(11.83);  % chi2inv(0.9973, 2)
set(A.ell, 'XData', xh(2) + c(1,:), 'YData', xh(1) + c(2,:));
set(A.est, 'XData', xh(2), 'YData', xh(1));
% chaser: a small capsule shape pointing at Gateway
span = max(60, min(330, 1.3*abs(x(2)) + 40));       % view width follows the range to Gateway
hdg = atan2(-x(1), -x(2)); s = span/90; shape = [1.4 0; -0.8 0.9; -0.8 -0.9]' * s;
Rm = [cos(hdg) -sin(hdg); sin(hdg) cos(hdg)]; p = Rm * shape;
set(A.ship, 'XData', x(2) + p(1,:), 'YData', x(1) + p(2,:));
set(A.thr, 'XData', x(2), 'YData', x(1), 'UData', -cmd(2)*60, 'VData', -cmd(1)*60);
if lid && hypot(x(1), x(2)) < 320, set(A.ray, 'XData', [x(2) 0], 'YData', [x(1) 0]); else, set(A.ray, 'XData', nan, 'YData', nan); end
if fx, set(A.fix, 'XData', x(2), 'YData', x(1)); else, set(A.fix, 'XData', nan, 'YData', nan); end
% keep the chaser in view as it closes in
xlim(A.ax, [-span 0.15*span]); ylim(A.ax, 0.32*span*[-1 1]);
% HUD
kk = min(k, numel(kind));
phase = sprintf('heading to %s', names(kk)); if ~isnan(holdLeft), phase = sprintf('holding at %s  %.1f s', names(kk), holdLeft); end
err = norm(x(1:2) - xh(1:2)); sig = 3*sqrt(P(1,1) + P(2,2));
set(A.hud, 'String', {sprintf('T+%6.1f s', t), '', phase, '', ...
    sprintf('range to Gateway  %7.1f m', hypot(x(1), x(2))), sprintf('closing speed     %7.2f m/s', norm(x(3:4))), ...
    sprintf('thrust            %7.3f m/s^2', norm(cmd)), '', ...
    sprintf('EKF error         %7.2f m', err), sprintf('EKF 3-sigma       %7.2f m', sig), ...
    sprintf('accel bias est.   %5.1f %5.1f mm/s^2', 1e3*xh(5), 1e3*xh(6)), '', ...
    sprintf('lidar  %s     fix  %s', onoff(lid), onoff(fx))});
e = vecnorm(H.x(1:i,1:2) - H.xh(1:i,1:2), 2, 2); sg = 3*sqrt(H.P(1:i,1,1) + H.P(1:i,2,2));
set(A.eErr, 'XData', H.t(1:i), 'YData', e); set(A.eSig, 'XData', H.t(1:i), 'YData', sg);
drawnow limitrate;
end

function s = onoff(b)
if b, s = 'ON '; else, s = ' - '; end
end

function resultsFigure(H, err, sig3)
f = figure('Name', 'RPO docking · filter check', 'Color', 'w', 'Position', [120 120 1000 620]);
tl = tiledlayout(f, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, 'EKF consistency over the approach');
nexttile(tl); plot(H.t, err, H.t, sig3, '--'); grid on; legend('position error', '3\sigma bound');
xlabel('time [s]'); ylabel('m'); title('Is the error inside the uncertainty the filter reports?');
nexttile(tl); plot(H.t, H.nees, '.', 'MarkerSize', 4); yline(13.28, '--r', '99% bound');  % chi2inv(0.99, 4)
yline(4, ':k', 'expected 4'); grid on; xlabel('time [s]'); ylabel('NEES'); title('NEES (4 dof): near 4 = honest');
nexttile(tl); plot(H.t, 1e3*H.x(:,5:6), '-', H.t, 1e3*H.xh(:,5:6), '--'); grid on;
legend('true b_x', 'true b_y', 'est b_x', 'est b_y'); xlabel('time [s]'); ylabel('mm/s^2'); title('Accelerometer bias: estimated online');
nexttile(tl); plot(H.t, hypot(H.x(:,1), H.x(:,2)), H.t, vecnorm(H.x(:,3:4), 2, 2)*100); grid on;
legend('range [m]', 'speed [cm/s]'); xlabel('time [s]'); title('Range and speed: holds at the gates, soft dock');
end
