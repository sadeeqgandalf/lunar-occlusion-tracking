classdef LunarWorld < handle
    %LUNARWORLD  Artemis EVA scene: a rover mast camera watches astronauts walking among boulders.
    %   Port of src/mot/world.js. Same seed => same boulders, walks and detections as the web lab.
    %   The trackers receive ONLY detections [range bearing]; gt(i) is the true person id (0 = false alarm),
    %   kept separately for scoring.
    properties
        cfg; id; seed; cam; sun; boulders; targets
        t = 0; dt = 0.1; frame = 0
    end
    properties (Access = private)
        rng; srng
    end
    methods (Static)
        function S = scenarios()
            S.boulders = struct('label', "Artemis EVA · Boulder field", 'targets', 5, 'boulders', 9, 'pd', 0.95, 'clutter', 0.8, 'shadows', false, 'lowFrac', 0.35);
            S.crossing = struct('label', "Worksite · Crossing paths", 'targets', 8, 'boulders', 7, 'pd', 0.92, 'clutter', 1.2, 'shadows', false, 'lowFrac', 0.35);
            S.polar    = struct('label', "South pole · Long shadows", 'targets', 5, 'boulders', 9, 'pd', 0.95, 'clutter', 0.8, 'shadows', true, 'lowFrac', 0.35);
            S.open     = struct('label', "Open plain · Sparse cover", 'targets', 5, 'boulders', 1, 'pd', 0.95, 'clutter', 0.8, 'shadows', false, 'lowFrac', 0.35);
        end
        function c = camera()
            c = struct('x', 0, 'y', -4, 'h', 2.2, 'th', pi/2, 'fov', 100*pi/180, 'range', 48, ...
                'sigB', 0.004, 'sigR0', 0.08, 'sigRk', 0.0012, 'targetH', 1.8, 'targetR', 0.4);
        end
        function s = sigRange(cam, r)
            s = cam.sigR0 + cam.sigRk * r.^2;      % stereo depth noise grows with range squared
        end
    end
    methods
        function o = LunarWorld(scenario, seed, overrides)
            if nargin < 1, scenario = "boulders"; end
            if nargin < 2, seed = 7; end
            S = LunarWorld.scenarios(); o.cfg = S.(char(scenario));
            if nargin >= 3, f = fieldnames(overrides); for k = 1:numel(f), o.cfg.(f{k}) = overrides.(f{k}); end, end
            o.id = string(scenario); o.seed = seed; o.cam = LunarWorld.camera();
            o.rng = Mulberry32(seed*9973 + 17);   % world + truth motion
            o.srng = Mulberry32(seed*7717 + 5);   % sensor noise (separate stream)
            o.sun = struct('az', atan2(-25, -60), 'el', atan2(9, hypot(60, 25)));
            o.placeBoulders(); o.spawnTargets();
        end

        function placeBoulders(o)
            r = o.rng; c = o.cam; B = struct('x', {}, 'y', {}, 'r', {}, 'h', {});
            k = 0;
            while numel(B) < o.cfg.boulders && k < 2000
                k = k + 1;
                rg = r.uniform(9, 36); off = r.uniform(-0.42, 0.42) * c.fov; br = r.uniform(1.4, 3.2);
                if r.next() < o.cfg.lowFrac, h = r.uniform(0.7, 1.4); else, h = r.uniform(2.4, 3.6); end
                b = struct('x', c.x + rg*cos(c.th + off), 'y', c.y + rg*sin(c.th + off), 'r', br, 'h', h);
                if any(arrayfun(@(q) hypot(q.x - b.x, q.y - b.y) < q.r + b.r + 3.5, B)), continue; end
                B(end+1) = b; %#ok<AGROW>
            end
            o.boulders = B;
        end

        function spawnTargets(o)
            r = o.rng; c = o.cam; T = struct('id', {}, 'x', {}, 'y', {}, 'h', {}, 'v', {}, 'goal', {}, 'trail', {}, 'vis', {});
            for i = 1:o.cfg.targets
                for k = 1:500
                    rr = r.uniform(12, c.range - 8); oo = r.uniform(-0.38, 0.38) * c.fov;
                    p = [c.x + rr*cos(c.th + oo), c.y + rr*sin(c.th + oo)];
                    if ~o.nearRock(p, 1.2), break; end
                end
                h = r.uniform(-pi, pi); v = r.uniform(0.6, 1.3);
                T(i) = struct('id', i, 'x', p(1), 'y', p(2), 'h', h, 'v', v, 'goal', [], 'trail', zeros(0, 2), 'vis', o.view(p(1), p(2)));
            end
            o.targets = T;
        end

        function tf = nearRock(o, p, margin)
            tf = false;
            for b = o.boulders, if hypot(b.x - p(1), b.y - p(2)) < b.r + margin, tf = true; return; end, end
        end

        function g = goal(o)
            r = o.rng; c = o.cam;
            for k = 1:200
                rr = r.uniform(11, c.range - 6); oo = r.uniform(-0.4, 0.4) * c.fov;
                g = [c.x + rr*cos(c.th + oo), c.y + rr*sin(c.th + oo)];
                if ~o.nearRock(g, 1.5), return; end
            end
            g = [c.x, c.y + 25];
        end

        function step(o)
            %STEP  Goal-directed walking between worksites, small heading noise, steering around boulders.
            dt = o.dt; r = o.rng;
            for i = 1:numel(o.targets)
                t = o.targets(i);
                if isempty(t.goal) || hypot(t.goal(1) - t.x, t.goal(2) - t.y) < 1.5, t.goal = o.goal(); end
                a = atan2(t.goal(2) - t.y, t.goal(1) - t.x); sx = cos(a); sy = sin(a);
                for b = o.boulders                       % walk around rocks, not through them
                    dx = t.x - b.x; dy = t.y - b.y; d = hypot(dx, dy) - b.r;
                    if d < 2.0, w = 1.6*(2.0 - max(d, 0.05))/2.0; sx = sx + dx/(d + b.r)*w; sy = sy + dy/(d + b.r)*w; end
                end
                t.h = t.h + min(1, max(-1, wrapang(atan2(sy, sx) - t.h))) * 1.2*dt + 0.08*sqrt(dt)*r.randn();
                t.x = t.x + t.v*cos(t.h)*dt; t.y = t.y + t.v*sin(t.h)*dt;
                for b = o.boulders                       % never inside a rock
                    dx = t.x - b.x; dy = t.y - b.y; d = hypot(dx, dy);
                    if d < b.r + 0.4, t.x = b.x + dx/d*(b.r + 0.4); t.y = b.y + dy/d*(b.r + 0.4); end
                end
                t.vis = o.view(t.x, t.y);
                if mod(o.frame, 3) == 0, t.trail(end+1, :) = [t.x t.y]; if size(t.trail, 1) > 120, t.trail(1, :) = []; end, end
                o.targets(i) = t;
            end
            o.t = o.t + dt; o.frame = o.frame + 1;
        end

        function v = view(o, x, y)
            %VIEW  What the camera can see of people at (x, y): geometry, cast-shadow darkness, detection probability.
            v = occ_visibility(o.cam, o.boulders, x, y, o.cam.targetR);
            if o.cfg.shadows, v.shadow = occ_shadow(o.boulders, x, y, o.sun); else, v.shadow = zeros(size(v.range)); end
            v.pd = occ_pd(v.inFov, v.visFrac, o.cfg.pd, v.shadow);
        end

        function [dets, gt] = sense(o)
            %SENSE  One camera frame. dets: K x 2 [range bearing]; gt: true person id per detection (0 = false alarm).
            c = o.cam; s = o.srng; dets = zeros(0, 2); gt = zeros(0, 1);
            for i = 1:numel(o.targets)
                v = o.targets(i).vis;
                if s.next() >= v.pd, continue; end
                sr = LunarWorld.sigRange(c, v.range);
                sb = c.sigB * (1 + (v.visFrac < 0.9));     % partially hidden: the centroid jitters more
                rg = v.range + sr*s.randn(); br = v.visCenter + sb*s.randn();
                dets(end+1, :) = [rg br]; gt(end+1, 1) = o.targets(i).id; %#ok<AGROW>
            end
            nFalse = LunarWorld.poisson(o.cfg.clutter, s);
            for k = 1:nFalse
                rg = s.uniform(4, c.range); br = c.th + s.uniform(-c.fov/2, c.fov/2);
                dets(end+1, :) = [rg br]; gt(end+1, 1) = 0; %#ok<AGROW>
            end
        end

        function h = isHidden(~, v)
            h = v.inFov & v.pd < 0.15;                  % in the field of view but effectively undetectable
        end
    end
    methods (Static)
        function k = poisson(lam, s)
            k = 0; p = exp(-lam); acc = p; u = s.next();
            while u > acc && k < 50, k = k + 1; p = p*lam/k; acc = acc + p; end
        end
    end
end
