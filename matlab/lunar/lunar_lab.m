function lunar_lab(opts)
%LUNAR_LAB  Lunar Occlusion Tracking Lab in MATLAB: a 3-D Artemis EVA scene, the rover camera's own view, and
%   four trackers compared live on the same detections.
%
%   lunar_lab                                   % boulder field, seed 8, live
%   lunar_lab(Scenario="polar")                 % south pole: long cast shadows hide people too
%   lunar_lab(Video="lunar.mp4", Seconds=90)    % record 90 s of simulation to a video instead of a window
%
%   What you see
%     P1..P5 (suit-stripe colour)  who each astronaut really is
%     #11, #12 ...                 the ID the selected tracker gave them (ring + tag; mask colour in the camera view)
%     #11? with a shaded area      hidden: the tracker's best guess of where they are
%     purple wedges                where the camera's view is blocked by tall rocks
%     orange rings                 this frame's detections (the only thing the trackers get)
%   Drag the 3-D view to orbit, scroll to zoom. Space = pause.
arguments
    opts.Scenario (1,1) string {mustBeMember(opts.Scenario, ["boulders" "crossing" "polar" "open"])} = "boulders"
    opts.Seed (1,1) double = 8
    opts.Speed (1,1) double = 2
    opts.Toolbox (1,1) logical = true          % add MathWorks trackerGNN (told what is visible) as a 4th tracker
    opts.Video (1,1) string = ""
    opts.Seconds (1,1) double = inf
end

%% ------------------------------------------------------------------ palette and shared state
BG = [0.04 0.05 0.08]; FG = [0.86 0.89 0.95]; DIM = [0.56 0.61 0.7];
TRK = [0.29 0.87 0.50; 0.30 0.79 0.94; 1.00 0.42 0.42; 0.74 0.56 1.00];  % card colours per tracker
PAL = hex(["#F3C300" "#A1CAF1" "#F38400" "#E68FAC" "#00C27C" "#C2B280" "#FF5C6C" "#8DB600" "#4FA3E0" "#F99379" ...
           "#D97BA4" "#DCD300" "#B98AD4" "#E25822" "#2EC4B6" "#B794F6"]);
STRIPE = hex(["#e5322d" "#f2c12e" "#2f6fe0" "#2fb34a" "#f07a1a" "#8e4fd6" "#1fb5c9" "#e64fa3" "#9ccc2b" "#8a5a32"]);
PLAIN = struct('negInfo', 'waits, and searches only the blind spot', 'aware', 'knows the blind spots, so it waits', ...
    'naive', 'forgets anyone it cannot see', 'gnn', 'MathWorks toolbox tracker, told what is visible');
SHORT = ["NI" "Aware" "Naive" "GNN"];

S = struct('scenario', opts.Scenario, 'seed', opts.Seed, 'speed', opts.Speed, 'paused', false, 'sel', 1, ...
    'truth', true, 'panel', true, 'rebuild', true);
sess = []; lbl = {}; idHist = {}; prevMatch = {}; prevHidden = []; feed = strings(0, 1); pending = struct('id', {}, 't', {}, 'dur', {}, 'res', {});
seenEv = []; marks = struct('x', {}, 'y', {}, 't', {}, 'kept', {}); swSeries = {};
G = struct(); vw = []; recording = opts.Video ~= "";

%% ------------------------------------------------------------------ figure, controls, axes
fig = figure('Name', 'Lunar Occlusion Tracking Lab', 'NumberTitle', 'off', 'Color', BG, 'Position', [30 30 1600 900], ...
    'MenuBar', 'none', 'ToolBar', 'figure', 'KeyPressFcn', @onKey, 'Visible', onoff(~recording));
uictl('text', [0.006 0.958 0.2 0.032], 'LUNAR OCCLUSION TRACKING LAB', 'FontWeight', 'bold', 'FontSize', 13, 'ForegroundColor', [1 0.82 0.4], 'HorizontalAlignment', 'left');
hPlay = uictl('pushbutton', [0.21 0.958 0.07 0.034], '⏸  Pause', 'Callback', @(~, ~) togglePause(), 'FontWeight', 'bold');
SPEEDS = [1 2 4 8 16];
uictl('popupmenu', [0.285 0.958 0.055 0.034], {'1×', '2×', '4×', '8×', '16×'}, 'Value', max(1, find(SPEEDS == S.speed, 1)), ...
    'Callback', @(h, ~) setfield2('speed', SPEEDS(h.Value)));
scen = LunarWorld.scenarios(); keys = string(fieldnames(scen))';
scenLabels = cell(1, numel(keys)); for q = 1:numel(keys), scenLabels{q} = char(scen.(keys(q)).label); end
uictl('popupmenu', [0.345 0.958 0.17 0.034], scenLabels, 'Value', find(keys == S.scenario), ...
    'Callback', @(h, ~) setScenario(keys(h.Value)));
uictl('text', [0.52 0.958 0.03 0.03], 'Seed', 'HorizontalAlignment', 'right');
uictl('edit', [0.553 0.958 0.04 0.034], num2str(S.seed), 'Callback', @(h, ~) setSeed(str2double(h.String)));
uictl('text', [0.6 0.958 0.04 0.03], 'Show', 'HorizontalAlignment', 'right');
hSel = uictl('popupmenu', [0.643 0.958 0.15 0.034], {'Aware + neg. info'}, 'Callback', @(h, ~) setfield2('sel', h.Value));
uictl('checkbox', [0.8 0.958 0.075 0.034], 'Real people', 'Value', 1, 'Callback', @(h, ~) setfield2('truth', h.Value == 1));
uictl('pushbutton', [0.878 0.958 0.055 0.034], 'Reset', 'Callback', @(~, ~) setfield2('rebuild', true));
hPanelBtn = uictl('pushbutton', [0.937 0.958 0.058 0.034], 'Panel ▸', 'Callback', @(~, ~) togglePanel());

% each 3-D view lives in its own panel: a zoomed perspective axes draws beyond its own rectangle, a uipanel clips it
pan3 = uipanel(fig, 'Units', 'normalized', 'Position', [0.004 0.335 0.67 0.615], 'BorderType', 'none', 'BackgroundColor', [0 0 0]);
panc = uipanel(fig, 'Units', 'normalized', 'Position', [0.004 0.008 0.67 0.32], 'BorderType', 'none', 'BackgroundColor', [0 0 0]);
ax3 = axes(pan3, 'Position', [0 0 1 1]);
axc = axes(panc, 'Position', [0 0 1 1]);
axp = axes(fig, 'Position', [0.68 0.17 0.316 0.78], 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1]);
axs = axes(fig, 'Position', [0.712 0.03 0.276 0.115], 'Color', [0.06 0.07 0.1], 'XColor', DIM, 'YColor', DIM, 'FontSize', 8, 'Box', 'off');
hold(axp, 'on'); hold(axs, 'on'); grid(axs, 'on'); axs.GridColor = [0.3 0.33 0.4];
title(axs, 'ID switches so far (lower is better)', 'Color', FG, 'FontSize', 9, 'FontWeight', 'normal');
annotation(pan3, 'textbox', [0.006 0.875 0.64 0.11], 'String', ...
    {'\bfP1–P5\rm real astronaut (suit stripe)    \bf#11\rm tracker''s ID    \bf#11?\rm hidden, best guess', ...
     '\color[rgb]{0.74 0.56 1}purple\color[rgb]{0.86 0.89 0.95} = camera blind zone    \color[rgb]{1 0.7 0.33}orange ring\color[rgb]{0.86 0.89 0.95} = detection'}, ...
    'Color', FG, 'FontSize', 10, 'EdgeColor', 'none', 'BackgroundColor', BG, 'FaceAlpha', 0.75, 'Interpreter', 'tex');
annotation(panc, 'textbox', [0.006 0.87 0.3 0.11], 'String', 'ROVER CAMERA · 2.2 m mast', 'Color', [1 0.82 0.4], ...
    'FontSize', 10, 'FontWeight', 'bold', 'EdgeColor', 'none', 'BackgroundColor', BG, 'FaceAlpha', 0.7);
P = struct();   % panel text handles, built in buildPanel

if recording
    vw = VideoWriter(opts.Video, 'MPEG-4'); vw.FrameRate = 20; vw.Quality = 92; open(vw);
end

%% ------------------------------------------------------------------ main loop
wall = tic; acc = 0; frameSize = [];
while isvalid(fig)
    if S.rebuild, newSession(); end
    if ~S.paused
        if recording, acc = acc + S.speed/20; else, dtw = min(toc(wall), 0.25); acc = acc + dtw*S.speed; end
        wall = tic; n = 0;
        while acc >= 0.1 && n < 40
            simStep(); acc = acc - 0.1; n = n + 1;
        end
    else
        wall = tic;
    end
    render();
    if recording
        fr = getframe(fig);
        if isempty(frameSize), frameSize = size(fr.cdata, [1 2]); end
        if ~isequal(size(fr.cdata, [1 2]), frameSize), fr.cdata = imresize(fr.cdata, frameSize); end
        writeVideo(vw, fr);
        if sess.world.t >= opts.Seconds, break; end
    else
        drawnow limitrate; pause(0.005);
        if sess.world.t >= opts.Seconds, break; end
    end
end
if ~isempty(vw), close(vw); fprintf('Video written: %s\n', opts.Video); end
if recording && isvalid(fig), close(fig); end

%% ================================================================== simulation side
    function newSession()
        tb = "none"; if opts.Toolbox, tb = "aware"; end
        sess = LunarSession(S.scenario, S.seed, Toolbox=tb);
        nR = numel(sess.runs); nP = numel(sess.world.targets);
        lbl = repmat({containers.Map('KeyType', 'double', 'ValueType', 'double')}, 1, nR);
        for k = 1:nR, lbl{k} = containers.Map('KeyType', 'double', 'ValueType', 'double'); end
        idHist = repmat({repmat({[]}, 1, nP)}, 1, nR); prevMatch = repmat({nan(1, nP)}, 1, nR);
        prevHidden = false(1, nP); feed = strings(0, 1); pending = pending([]); seenEv = zeros(1, nR);
        marks = marks([]); swSeries = repmat({zeros(0, 2)}, 1, nR);
        names = arrayfun(@(r) char(r.tracker.name), sess.runs, 'UniformOutput', false);
        hSel.String = names; S.sel = min(S.sel, nR); hSel.Value = S.sel;
        pushFeed(sprintf('%s · seed %d', sess.world.cfg.label, S.seed), DIM);
        buildScene(); buildPanel(); S.rebuild = false;
    end

    function simStep()
        sess.step(); W = sess.world;
        for k = 1:numel(sess.runs)                       % display IDs #11, #12 ... in confirmation order
            for t = sess.runs(k).tracker.tracks
                if t.confirmed && ~lbl{k}.isKey(t.id), lbl{k}(t.id) = 11 + lbl{k}.Count; end
            end
            lm = sess.runs(k).metrics.lastMatch;          % ID history per person; announce switches of the shown tracker
            for g = 1:numel(lm)
                kid = lm(g); if isnan(kid), continue; end
                n = dispId(k, kid); h = idHist{k}{g};
                if isempty(h) || h(end) ~= n, idHist{k}{g}(end+1) = n; end
                b = prevMatch{k}(g);
                if k == S.sel && ~isnan(b) && b ~= kid
                    pushFeed(sprintf('%s  %s → %s  ID switch', pname(g), idtex(dispId(k, b)), idtex(n)), [1 0.6 0.64]);
                end
                prevMatch{k}(g) = kid;
            end
        end
        for g = 1:numel(W.targets)
            v = W.targets(g).vis; hid = v.inFov && v.pd < 0.15;
            if hid && ~prevHidden(g)
                if v.shadow > 0.3, why = 'in shadow'; else, why = 'behind rock'; end
                pushFeed(sprintf('%s  %s', pname(g), why), [1 0.82 0.4]);
            end
            prevHidden(g) = hid;
        end
        for k = 1:numel(sess.runs)                       % reappearances: collect every tracker's verdict
            ev = sess.runs(k).metrics.evts;
            for e = seenEv(k)+1:numel(ev)
                q = find([pending.id] == ev(e).target, 1);
                if isempty(q), pending(end+1) = struct('id', ev(e).target, 't', W.t, 'dur', ev(e).dur, 'res', nan(1, numel(sess.runs))); q = numel(pending); end %#ok<AGROW>
                pending(q).res(k) = ev(e).kept;
                if k == S.sel
                    g = ev(e).target; marks(end+1) = struct('x', W.targets(g).x, 'y', W.targets(g).y, 't', W.t, 'kept', ev(e).kept); %#ok<AGROW>
                end
            end
            seenEv(k) = numel(ev);
        end
        done = false(1, numel(pending));
        for q = 1:numel(pending)
            p = pending(q);
            if all(~isnan(p.res)) || W.t - p.t > 2.5
                parts = strings(1, numel(p.res));
                for k = 1:numel(p.res)
                    if isnan(p.res(k)), m = '…'; elseif p.res(k), m = '✔'; else, m = '✘'; end
                    parts(k) = sprintf('\\color[rgb]{%.2f %.2f %.2f}%s%s', TRK(k, :), SHORT(k), m);
                end
                if any(p.res == 1), c = [0.45 0.9 0.6]; else, c = [1 0.6 0.64]; end
                pushFeed(sprintf('%s  back (%.0f s)  %s', pname(p.id), p.dur, strjoin(parts, '  ')), c);
                done(q) = true;
            end
        end
        pending(done) = [];
        if mod(W.frame, 10) == 0
            for k = 1:numel(sess.runs), swSeries{k}(end+1, :) = [W.t sess.runs(k).metrics.idsw]; end
        end
    end

    function n = dispId(k, kid)
        if lbl{k}.isKey(kid), n = lbl{k}(kid); else, n = 11 + lbl{k}.Count; lbl{k}(kid) = n; end
    end
    function c = idColor(n), c = PAL(mod(n - 11, size(PAL, 1)) + 1, :); end
    function s = idtex(n), s = sprintf('\\color[rgb]{%.2f %.2f %.2f}\\bf#%d\\rm', idColor(n), n); end
    function s = pname(g), s = sprintf('\\color[rgb]{%.2f %.2f %.2f}■\\color[rgb]{%.2f %.2f %.2f} P%d', STRIPE(g, :), FG, g); end
    function pushFeed(txt, c)
        feed = [sprintf('\\color[rgb]{%.2f %.2f %.2f}%5.0fs  \\color[rgb]{%.2f %.2f %.2f}%s', DIM, sessT(), c, txt); feed];
        if numel(feed) > 11, feed = feed(1:11); end
    end
    function t = sessT(), if isempty(sess), t = 0; else, t = sess.world.t; end, end

%% ================================================================== scene
    function buildScene()
        W = sess.world; G = struct();
        sunAz = W.sun.az; sunEl = W.sun.el; if ~W.cfg.shadows, sunEl = deg2rad(32); end
        sd = [cos(sunEl)*cos(sunAz) cos(sunEl)*sin(sunAz) sin(sunEl)];
        for a = [ax3 axc]
            cla(a); delete(findobj(a, 'Type', 'light')); hold(a, 'on');
            set(a, 'Color', [0 0 0], 'XColor', 'none', 'YColor', 'none', 'ZColor', 'none', 'DataAspectRatio', [1 1 1], ...
                'Clipping', 'on', 'XLim', [-470 470], 'YLim', [-450 480], 'ZLim', [-5 470], 'Projection', 'perspective', ...
                'AmbientLightColor', [0.42 0.44 0.5]);
            ground(a); stars(a); earth(a); rover(a);
            for b = W.boulders, boulder(a, b); end
            light(a, 'Style', 'infinite', 'Position', sd, 'Color', [1 0.97 0.9]);
            light(a, 'Style', 'infinite', 'Position', [-0.3 -1 0.5], 'Color', [0.16 0.18 0.24]);   % earthshine fill
            for b = W.boulders, castShadow(a, b, sd, W.cfg.shadows); end
        end
        blindZones(ax3); fovLines(ax3);
        % people: one model in each view (the camera copy gets the tracker's mask colour)
        G.trail = gobjects(1, numel(W.targets)); G.ptxt = G.trail;
        for g = 1:numel(W.targets)
            G.p3(g) = person(ax3, STRIPE(g, :), sd); G.pc(g) = person(axc, STRIPE(g, :), []);
            G.trail(g) = plot3(ax3, nan, nan, nan, ':', 'Color', [1 1 1 0.35], 'LineWidth', 1);
            G.ptxt(g) = text(ax3, 0, 0, 0, '', 'FontSize', 9, 'FontWeight', 'bold', 'Margin', 1, 'Color', [0.05 0.06 0.08]);
        end
        G.ring = gobjects(0); G.area = gobjects(0); G.tag = gobjects(0); G.ping = gobjects(0); G.box = gobjects(0); G.btag = gobjects(0); G.mark = gobjects(0);
        % views
        ax3.CameraPosition = [-7 -27.5 18]; ax3.CameraTarget = [0 17 0]; ax3.CameraUpVector = [0 0 1]; ax3.CameraViewAngle = 34;
        axc.CameraPosition = [W.cam.x W.cam.y W.cam.h]; axc.CameraTarget = [W.cam.x W.cam.y + 40 W.cam.h - 2.2];
        axc.CameraUpVector = [0 0 1]; fitCamera();
        % MATLAB needs each perspective camera OUTSIDE its axes box (inside, the box background covers the window):
        % the box starts just in front of the lens for the rover camera and just in front of the orbit camera
        ax3.YLim = [-20 480]; axs.Toolbar.Visible = 'off'; axp.Toolbar.Visible = 'off'; axc.YLim = [W.cam.y + 0.3 480];
        cla(axs); hold(axs, 'on');
        for k = 1:numel(sess.runs), G.sw(k) = plot(axs, nan, nan, 'Color', TRK(k, :), 'LineWidth', 1.6); end
    end

    function fitCamera()
        % match the rover camera's 100 deg horizontal field of view to the panel's aspect ratio
        px = getpixelposition(axc); asp = px(3)/max(px(4), 1);
        axc.CameraViewAngle = 2*atand(tand(50)/asp);
    end

    function ground(a)
        [X, Y] = meshgrid(-60:1:60, -20:1:70);
        Z = 0.035*(sin(0.31*X + 1.3) + sin(0.27*Y + 0.4) + 0.6*sin(0.71*X + 0.53*Y) + 0.4*sin(1.3*X - 0.9*Y + 2));
        cr = [-38 58 8 0.9; 22 63 10 1.1; 47 22 6 0.6; -50 12 6 0.6; 30 -14 7 0.7; -24 -12 5 0.5; 52 52 9 1];
        alb = 0.5 + 0.03*sin(0.23*X + 0.3).*sin(0.19*Y) + 0.02*sin(0.07*X - 0.11*Y + 1);
        for q = 1:size(cr, 1)
            d = hypot(X - cr(q, 1), Y - cr(q, 2)) / cr(q, 3);
            Z = Z - cr(q, 4)*max(0, 1 - d.^2) + 0.35*cr(q, 4)*exp(-((d - 1)/0.25).^2);
            alb = alb - 0.06*max(0, 1 - d.^2);
        end
        C = cat(3, alb*1.0, alb*0.975, alb*0.94);
        surf(a, X, Y, Z, C, 'EdgeColor', 'none', 'FaceColor', 'interp', 'FaceLighting', 'gouraud', 'AmbientStrength', 0.32, ...
            'DiffuseStrength', 0.85, 'SpecularStrength', 0.02);
    end

    function stars(a)
        rs = RandStream('mt19937ar', 'Seed', 5); n = 900;
        az = 2*pi*rand(rs, n, 1); el = asin(0.05 + 0.95*rand(rs, n, 1)); R = 450;
        scatter3(a, R*cos(el).*cos(az), 20 + R*cos(el).*sin(az), R*sin(el), 1 + 5*rand(rs, n, 1).^3, ...
            [0.85 0.88 1].*(0.5 + 0.5*rand(rs, n, 1)), 'filled');
    end

    function earth(a)
        [X, Y, Z] = sphere(48); c = [-55 300 62]; r = 11;
        land = (sin(3*X + 1).*cos(2*Y) + sin(5*Z + 2*X)) > 0.75; cloud = (sin(9*X + 4*Z) .* sin(7*Y - 3*Z)) > 0.55;
        C = cat(3, 0.12 + 0.2*land + 0.75*cloud, 0.3 + 0.25*land + 0.65*cloud, 0.62 - 0.35*land + 0.35*cloud);
        surf(a, c(1) + r*X, c(2) + r*Y, c(3) + r*Z, min(C, 1), 'EdgeColor', 'none', 'FaceLighting', 'gouraud', ...
            'AmbientStrength', 0.35);
    end

    function rover(a)
        c = sess.world.cam; x0 = c.x; y0 = c.y - 0.6;
        box(a, [x0 y0 0.75], [1.5 2.0 0.45], [0.82 0.83 0.85]);                         % chassis
        box(a, [x0 y0 1.02], [1.7 2.1 0.06], [0.12 0.18 0.36]);                         % solar deck
        for sx = [-0.9 0.9], for sy = [-0.75 0 0.75], wheel(a, [x0 + sx, y0 + sy, 0.32]); end, end
        [X, Y, Z] = cylinder(0.06, 12); surf(a, X + x0, Y + c.y - 0.15, 1.0 + Z*(c.h - 1.05), 'FaceColor', [0.7 0.7 0.72], 'EdgeColor', 'none');
        box(a, [x0 c.y c.h], [0.55 0.28 0.24], [0.22 0.23 0.26]);                       % camera head (stereo pair)
        for sx = [-0.17 0.17]
            [X, Y, Z] = cylinder(0.055, 16);
            surf(a, X + x0 + sx, c.y + 0.12 + 0.06*Z, c.h + Y, 'FaceColor', [0.85 0.65 0.15], 'EdgeColor', 'none');
        end
    end

    function wheel(a, p)
        [Y, Z, X] = cylinder(0.32, 20);                                                 % axle along x
        surf(a, p(1) + 0.24*(X - 0.5), p(2) + Y, p(3) + Z, 'FaceColor', [0.25 0.25 0.27], 'EdgeColor', 'none', 'FaceLighting', 'gouraud');
    end

    function boulder(a, b)
        [X, Y, Z] = sphere(14);
        k = 7.3*b.x + 3.1*b.y;                                                          % deterministic per rock
        rn = 1 + 0.12*sin(2.1*X + 1.3*Y + 0.9*Z + k) + 0.07*sin(3.7*Y - 2.9*Z + 1.7*k) + 0.05*sin(5.3*X + 4.1*Z - k);
        Zs = Z .* rn * b.h; Zs(Zs < -0.3) = -0.3;
        Zs(Z > 0.8) = Zs(Z > 0.8) * 0.93;                                               % slightly flattened top
        sh = 0.3 + 0.03*sin(k) + 0.04*sin(4*X + 3*Y + k);
        C = cat(3, sh*1.05, sh, sh*0.94);
        surf(a, b.x + X.*rn*b.r, b.y + Y.*rn*b.r, Zs, C, 'EdgeColor', 'none', 'FaceLighting', 'flat', ...
            'AmbientStrength', 0.3, 'DiffuseStrength', 0.9, 'SpecularStrength', 0.04);   % flat facets read as rock
    end

    function castShadow(a, b, sd, polar)
        [X, Y, Z] = sphere(20); k = Z(:) >= 0;
        px = b.x + b.r*X(k); py = b.y + b.r*Y(k); pz = b.h*Z(k);
        gx = px - sd(1)/sd(3)*pz; gy = py - sd(2)/sd(3)*pz;                          % project along the sun ray
        h = convhull(gx, gy);
        patch(a, gx(h), gy(h), 0.05*ones(size(h)), [0 0 0], 'EdgeColor', 'none', 'FaceAlpha', 0.45 + 0.25*polar, 'FaceLighting', 'none');
    end

    function blindZones(a)
        c = sess.world.cam; first = true;
        for b = sess.world.boulders
            if b.h < c.h, continue; end                                                 % low rocks: heads still visible
            d = hypot(b.x - c.x, b.y - c.y); half = asin(min(1, b.r/d)); cb = atan2(b.y - c.y, b.x - c.x);
            th = linspace(cb - half, cb + half, 16); tl = d*cos(half);
            x = [c.x + tl*cos(cb - half), c.x + c.range*cos(th), c.x + tl*cos(cb + half)];
            y = [c.y + tl*sin(cb - half), c.y + c.range*sin(th), c.y + tl*sin(cb + half)];
            patch(a, x, y, 0.07*ones(size(x)), [0.55 0.35 0.95], 'FaceAlpha', 0.14, 'EdgeColor', [0.7 0.5 1], 'EdgeAlpha', 0.35, ...
                'LineStyle', '--', 'FaceLighting', 'none');
            if first, text(a, c.x + 0.8*c.range*cos(cb), c.y + 0.8*c.range*sin(cb), 0.3, 'blind zone', 'Color', [0.8 0.66 1], ...
                    'FontSize', 9, 'FontAngle', 'italic', 'HorizontalAlignment', 'center'); first = false; end
        end
    end

    function fovLines(a)
        c = sess.world.cam; th = c.th + linspace(-c.fov/2, c.fov/2, 60);
        plot3(a, [c.x, c.x + c.range*cos(th), c.x], [c.y, c.y + c.range*sin(th), c.y], 0.06*ones(1, 62), ...
            '-', 'Color', [0.3 0.79 0.94 0.45], 'LineWidth', 1.2);
    end

    function m = person(a, stripe, sd)
        m.tf = hgtransform(a); white = [0.95 0.95 0.96]; m.suit = gobjects(0);
        c = @(r, z0, z1, y, col) cylinderAt(m.tf, r, z0, z1, [0 y], col);
        m.suit(end+1) = c(0.09, 0.1, 0.85, 0.11, white); m.suit(end+1) = c(0.09, 0.1, 0.85, -0.11, white);   % legs
        c(0.1, 0, 0.12, 0.11, [0.35 0.35 0.37]); c(0.1, 0, 0.12, -0.11, [0.35 0.35 0.37]);                   % boots
        m.suit(end+1) = c(0.24, 0.8, 1.48, 0, white);                                                         % torso
        m.suit(end+1) = c(0.075, 0.95, 1.45, 0.31, white); m.suit(end+1) = c(0.075, 0.95, 1.45, -0.31, white); % arms
        c(0.247, 1.02, 1.11, 0, stripe);                                                                      % waist stripe
        c(0.079, 1.28, 1.36, 0.31, stripe); c(0.079, 1.28, 1.36, -0.31, stripe);                             % arm stripes
        c(0.094, 0.5, 0.57, 0.11, stripe); c(0.094, 0.5, 0.57, -0.11, stripe);                               % leg stripes
        [X, Y, Z] = sphere(16);
        m.suit(end+1) = surf(0.27*X, 0.33*Y, 1.47 + 0.11*Z, 'Parent', m.tf, 'FaceColor', white, 'EdgeColor', 'none', 'FaceLighting', 'gouraud');
        box(m.tf, [-0.33 0 1.18], [0.2 0.46 0.6], [0.82 0.83 0.85]);                                           % life-support pack
        m.suit(end+1) = surf(0.19*X, 0.19*Y, 1.7 + 0.19*Z, 'Parent', m.tf, 'FaceColor', white, 'EdgeColor', 'none', 'FaceLighting', 'gouraud');
        V = X; V(X < 0.25) = NaN;                                                                             % gold visor
        surf(0.197*V, 0.197*Y, 1.7 + 0.197*Z, 'Parent', m.tf, 'FaceColor', [0.85 0.64 0.12], 'EdgeColor', 'none', 'FaceLighting', 'gouraud', 'SpecularStrength', 0.9);
        H = Z; H(Z < 0.82) = NaN;                                                                             % helmet stripe
        surf(0.2*X, 0.2*Y, 1.7 + 0.2*H, 'Parent', m.tf, 'FaceColor', stripe, 'EdgeColor', 'none', 'FaceLighting', 'gouraud');
        set(findobj(m.tf, 'Type', 'surface'), 'AmbientStrength', 0.45);
        m.sh = [];
        if ~isempty(sd)                                                                                       % long sun shadow on the ground
            L = 1.8*cos(asin(sd(3)))/sd(3); ang = atan2(-sd(2), -sd(1)); w = 0.22;
            u = linspace(0, 1, 12); xs = [u*L, fliplr(u*L)]; ys = [w*(1 - u.^2), -w*fliplr(1 - u.^2)];
            m.shTf = hgtransform(a);
            patch(xs*cos(ang) - ys*sin(ang), xs*sin(ang) + ys*cos(ang), 0.06*ones(size(xs)), [0 0 0], 'Parent', m.shTf, ...
                'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceLighting', 'none');
        else
            m.shTf = [];
        end
    end

%% ================================================================== per-frame drawing
    function render()
        W = sess.world; k = S.sel; T = sess.runs(k).tracker; lm = sess.runs(k).metrics.lastMatch;
        rep = T.reported(); repIds = [rep.id];
        for g = 1:numel(W.targets)
            t = W.targets(g); M = makehgtform('translate', [t.x t.y 0], 'zrotate', t.h);
            G.p3(g).tf.Matrix = M; G.pc(g).tf.Matrix = M;
            set([G.p3(g).tf; findobj(G.p3(g).tf)], 'Visible', onoff(S.truth));
            if ~isempty(G.p3(g).shTf), G.p3(g).shTf.Matrix = makehgtform('translate', [t.x t.y 0]); set(findobj(G.p3(g).shTf), 'Visible', onoff(S.truth)); end
            set(G.trail(g), 'XData', t.trail(:, 1), 'YData', t.trail(:, 2), 'ZData', 0.08*ones(size(t.trail, 1), 1), 'Visible', onoff(S.truth));
            hid = t.vis.inFov && t.vis.pd < 0.15;
            if hid, if t.vis.shadow > 0.3, s = sprintf('P%d in shadow', g); else, s = sprintf('P%d behind rock', g); end, bg = [0.85 0.87 0.9]; else, s = sprintf('P%d', g); bg = STRIPE(g, :); end
            set(G.ptxt(g), 'Position', [t.x + 0.6, t.y, 0.25], 'String', s, 'BackgroundColor', bg, 'Visible', onoff(S.truth));
            % camera view: tint the visible pixels of tracked people in their tracker-ID colour (a modal instance mask)
            kid = lm(g); col = [0.95 0.95 0.96];
            if ~isnan(kid) && any(repIds == kid) && ~hid && t.vis.inFov, col = 0.45*[0.95 0.95 0.96] + 0.55*idColor(dispId(k, kid)); end
            set(G.pc(g).suit, 'FaceColor', col);
        end
        % tracker beliefs in the 3-D view, boxes and tags in the camera view
        nR = 0; nA = 0; nB = 0; nTag = 0;
        for t = T.confirmedTracks()
            n = dispId(k, t.id); c = idColor(n); x = t.x(1); y = t.x(2); lost = t.lastSeen > 3;
            if lost
                nA = nA + 1; G.area = pool(G.area, nA, @() patch(ax3, nan, nan, nan, 'k', 'FaceAlpha', 0.2, 'EdgeColor', 'k', 'LineStyle', '--', 'LineWidth', 1.6, 'FaceLighting', 'none'));
                [ex, ey] = ellipse(t.P(1:2, 1:2), x, y);
                set(G.area(nA), 'XData', ex, 'YData', ey, 'ZData', 0.09*ones(size(ex)), 'FaceColor', c, 'EdgeColor', c, 'Visible', 'on');
                nTag = nTag + 1; tagAt(ax3, nTag, [x y 1.2], sprintf('#%d?', n), c, true);
            else
                nR = nR + 1; G.ring = pool(G.ring, nR, @() plot3(ax3, nan, nan, nan, 'LineWidth', 3.2));
                th = linspace(0, 2*pi, 40); set(G.ring(nR), 'XData', x + 0.75*cos(th), 'YData', y + 0.75*sin(th), 'ZData', 0.1*ones(1, 40), 'Color', c, 'Visible', 'on');
                nTag = nTag + 1; tagAt(ax3, nTag, [x y 2.55], sprintf('#%d', n), c, false);
            end
            % camera: box around the person (projected 3-D extent), dashed when only a guess
            dx = x - W.cam.x; dy = y - W.cam.y;
            if abs(wrapang(atan2(dy, dx) - W.cam.th)) < W.cam.fov/2 && hypot(dx, dy) > 1.5
                nB = nB + 1; G.box = pool(G.box, nB, @() plot3(axc, nan, nan, nan, 'LineWidth', 2));
                nn = [-dy dx]/hypot(dx, dy)*0.5; zz = [0 0 2.05 2.05 0];
                set(G.box(nB), 'XData', x + nn(1)*[-1 1 1 -1 -1], 'YData', y + nn(2)*[-1 1 1 -1 -1], 'ZData', zz, 'Color', c, ...
                    'LineStyle', ifelse(lost, '--', '-'), 'Visible', 'on');
                G.btag = pool(G.btag, nB, @() text(axc, 0, 0, 0, '', 'FontWeight', 'bold', 'FontSize', 10, 'Margin', 1, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom'));
                if lost, set(G.btag(nB), 'String', sprintf('#%d?', n), 'BackgroundColor', [0.04 0.05 0.08], 'Color', c, 'EdgeColor', c, 'LineStyle', '--');
                else, set(G.btag(nB), 'String', sprintf('#%d', n), 'BackgroundColor', c, 'Color', [0.05 0.06 0.08], 'EdgeColor', 'none'); end
                set(G.btag(nB), 'Position', [x y 2.15], 'Visible', 'on');
            end
        end
        hideFrom(G.ring, nR + 1); hideFrom(G.area, nA + 1); hideFrom(G.tag, nTag + 1); hideFrom(G.box, nB + 1); hideFrom(G.btag, nB + 1);
        % detections this frame
        d = sess.dets; th = linspace(0, 2*pi, 24);
        for j = 1:size(d, 1)
            G.ping = pool(G.ping, j, @() plot3(ax3, nan, nan, nan, 'Color', [1 0.7 0.33], 'LineWidth', 1.8));
            px = W.cam.x + d(j, 1)*cos(d(j, 2)); py = W.cam.y + d(j, 1)*sin(d(j, 2));
            set(G.ping(j), 'XData', px + 0.4*cos(th), 'YData', py + 0.4*sin(th), 'ZData', 0.11*ones(1, 24), 'Visible', 'on');
        end
        hideFrom(G.ping, size(d, 1) + 1);
        % same-ID / new-ID verdicts, fading after 3 s
        marks = marks(W.t - [marks.t] < 3);
        for q = 1:numel(marks)
            G.mark = pool(G.mark, q, @() text(ax3, 0, 0, 0, '', 'FontWeight', 'bold', 'FontSize', 11, 'Margin', 2, 'HorizontalAlignment', 'center'));
            if marks(q).kept, set(G.mark(q), 'String', '✔ same ID', 'Color', [0.45 0.95 0.6], 'BackgroundColor', [0.08 0.33 0.18]);
            else, set(G.mark(q), 'String', '✘ new ID', 'Color', [1 0.6 0.6], 'BackgroundColor', [0.3 0.07 0.1]); end
            set(G.mark(q), 'Position', [marks(q).x marks(q).y 3.4], 'Visible', 'on');
        end
        hideFrom(G.mark, numel(marks) + 1);
        if S.panel, updatePanel(); end
    end

    function tagAt(a, i, pos, s, c, dashed)
        G.tag = pool(G.tag, i, @() text(a, 0, 0, 0, '', 'FontWeight', 'bold', 'FontSize', 11, 'Margin', 1.5, 'HorizontalAlignment', 'center'));
        if dashed, set(G.tag(i), 'BackgroundColor', [0.04 0.05 0.08], 'Color', c, 'EdgeColor', c, 'LineStyle', '--');
        else, set(G.tag(i), 'BackgroundColor', c, 'Color', [0.05 0.06 0.08], 'EdgeColor', 'none'); end
        set(G.tag(i), 'Position', pos, 'String', s, 'Visible', 'on');
    end

%% ================================================================== side panel
    function buildPanel()
        cla(axp); P = struct(); y = 0.985;
        P.h1 = text(axp, 0, y, 'TRACKERS · same camera data, different handling of a miss', 'Color', DIM, 'FontSize', 9, 'FontWeight', 'bold');
        for k = 1:numel(sess.runs)
            y = y - 0.035; P.name(k) = text(axp, 0.0, y, '', 'Interpreter', 'tex', 'FontSize', 11, 'Color', FG);
            y = y - 0.03;  P.desc(k) = text(axp, 0.03, y, '', 'Color', DIM, 'FontSize', 9);
            y = y - 0.032; P.num(k) = text(axp, 0.03, y, '', 'Interpreter', 'tex', 'FontSize', 10, 'Color', FG);
            y = y - 0.012;
        end
        y = y - 0.03; P.h2 = text(axp, 0, y, '', 'Color', DIM, 'FontSize', 9, 'FontWeight', 'bold', 'Interpreter', 'tex');
        for g = 1:numel(sess.world.targets), y = y - 0.03; P.row(g) = text(axp, 0.0, y, '', 'Interpreter', 'tex', 'FontSize', 10, 'Color', FG); end
        y = y - 0.045; text(axp, 0, y, 'WHAT HAPPENED', 'Color', DIM, 'FontSize', 9, 'FontWeight', 'bold');
        for q = 1:11, y = y - 0.027; P.feed(q) = text(axp, 0.0, y, '', 'Interpreter', 'tex', 'FontSize', 9, 'Color', FG, 'FontName', 'Menlo'); end
    end

    function updatePanel()
        W = sess.world; key = ["negInfo" "aware" "naive" "gnn"];
        for k = 1:numel(sess.runs)
            m = sess.runs(k).metrics; ev = m.evts; kept = nnz([ev.kept]);
            mark = ''; if k == S.sel, mark = '◉ '; end
            set(P.name(k), 'String', sprintf('\\color[rgb]{%.2f %.2f %.2f}\\bf%s%s', TRK(k, :), mark, sess.runs(k).tracker.name));
            set(P.desc(k), 'String', PLAIN.(key(k)));
            set(P.num(k), 'String', sprintf('came back with same ID  \\bf%d / %d\\rm      ID switches  \\bf%d', kept, numel(ev), m.idsw));
        end
        k = S.sel; T = sess.runs(k).tracker; lm = sess.runs(k).metrics.lastMatch; rep = T.reported(); live = [rep.id];
        set(P.h2, 'String', sprintf('PEOPLE · IDs given by \\color[rgb]{%.2f %.2f %.2f}%s', TRK(k, :), T.name));
        for g = 1:numel(W.targets)
            v = W.targets(g).vis; hid = v.inFov && v.pd < 0.15; kid = lm(g);
            if hid, now = '\color[rgb]{1 0.82 0.4}hidden';
            elseif ~isnan(kid) && any(live == kid), now = idtex(dispId(k, kid));
            else, now = '\color[rgb]{0.56 0.61 0.7}–'; end
            h = idHist{k}{g}; shown = arrayfun(@idtex, h(max(1, end-4):end), 'UniformOutput', false);
            if numel(h) > 5, shown = [{'…'} shown]; end %#ok<AGROW>
            sw = ''; if numel(h) > 1, sw = sprintf('  \\color[rgb]{1 0.6 0.64}%d✘', numel(h) - 1); end
            set(P.row(g), 'String', sprintf('%s   now %s   \\color[rgb]{0.56 0.61 0.7}IDs: %s%s', pname(g), now, strjoin(shown, ' '), sw));
        end
        for q = 1:11, if q <= numel(feed), set(P.feed(q), 'String', feed(q)); else, set(P.feed(q), 'String', ''); end, end
        for k = 1:numel(sess.runs)
            s = swSeries{k}; if isempty(s), continue; end
            set(G.sw(k), 'XData', s(:, 1), 'YData', s(:, 2), 'LineWidth', 1.2 + 1.2*(k == S.sel));
        end
        hPlay.String = ifelse(S.paused, '▶  Play', '⏸  Pause');
    end

%% ================================================================== controls
    function togglePause(), S.paused = ~S.paused; end
    function setfield2(f, v), S.(f) = v; end
    function setScenario(s), S.scenario = s; S.rebuild = true; end
    function setSeed(v), if isfinite(v) && v >= 1, S.seed = round(v); S.rebuild = true; end, end
    function togglePanel()
        S.panel = ~S.panel; w = ifelse(S.panel, 0.67, 0.992);
        pan3.Position(3) = w; panc.Position(3) = w; axp.Visible = 'off';
        set(findobj(axp, 'Type', 'text'), 'Visible', onoff(S.panel)); axs.Visible = onoff(S.panel); set(axs.Children, 'Visible', onoff(S.panel));
        hPanelBtn.String = ifelse(S.panel, 'Panel ▸', '◂ Panel'); fitCamera();
    end
    function onKey(~, e)
        if strcmp(e.Key, 'space'), togglePause(); end
    end
    function h = uictl(style, pos, str, varargin)
        h = uicontrol(fig, 'Style', style, 'Units', 'normalized', 'Position', pos, 'String', str, 'FontSize', 11, ...
            'BackgroundColor', [0.1 0.12 0.17], 'ForegroundColor', FG, varargin{:});
    end
end

%% ==================================================================== helpers (no shared state)
function C = hex(h)
C = zeros(numel(h), 3);
for i = 1:numel(h), s = char(h(i)); C(i, :) = [hex2dec(s(2:3)) hex2dec(s(4:5)) hex2dec(s(6:7))]/255; end
end
function s = onoff(b), if b, s = 'on'; else, s = 'off'; end, end
function v = ifelse(c, a, b), if c, v = a; else, v = b; end, end
function H = pool(H, i, make)
if numel(H) < i || ~isgraphics(H(i)), H(i) = make(); end
end
function hideFrom(H, i)
if i <= numel(H), set(H(i:end), 'Visible', 'off'); end
end
function [x, y] = ellipse(P, cx, cy)
% 95% region of the position estimate (chi-square, 2 dof)
[V, D] = eig((P + P')/2); th = linspace(0, 2*pi, 48);
e = V*sqrt(max(D, 0))*[cos(th); sin(th)]*sqrt(5.991);
e = e .* max(1, 0.8./max(vecnorm(e), eps));        % never smaller than the person
x = cx + e(1, :); y = cy + e(2, :);
end
function h = cylinderAt(parent, r, z0, z1, xy, col)
[X, Y, Z] = cylinder(r, 18);
h = surf(X + xy(1), Y + xy(2), z0 + Z*(z1 - z0), 'Parent', parent, 'FaceColor', col, 'EdgeColor', 'none', 'FaceLighting', 'gouraud');
end
function h = box(parent, c, s, col)
[x, y, z] = ndgrid([-0.5 0.5]); v = [x(:) y(:) z(:)].*s + c;
f = [1 2 4 3; 5 6 8 7; 1 2 6 5; 3 4 8 7; 1 3 7 5; 2 4 8 6];
h = patch('Parent', parent, 'Vertices', v, 'Faces', f, 'FaceColor', col, 'EdgeColor', 'none', 'FaceLighting', 'flat');
end
