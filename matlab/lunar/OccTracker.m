classdef OccTracker < handle
    %OCCTRACKER  Multi-object tracker: per-track constant-velocity EKF in range/bearing, chi-square gating,
    %   Hungarian assignment with an explicit "missed" option, and a Bayesian existence probability r
    %   (Bernoulli filter, Ristic et al. 2013 [R2]; IPDA, Musicki et al. 1994 [R1]). Port of src/mot/tracker.js.
    %
    %   The three variants differ ONLY in the detection probability a track expects:
    %     naive            P_D = 0.95 everywhere: a miss is evidence the person left
    %     occlusion-aware  P_D x visibility of the predicted position (3-D line of sight + shadows)
    %     + neg. info      also moves the estimate into the blind zone: p(x|miss) ~ p(x)(1 - P_D(x)) [R3, Koch 2007]
    properties
        name; aware; negInfo; cam; boulders; sun; p
        tracks; nextId = 1; dt = 0.1
    end
    properties (Constant)
        DEFAULTS = struct('sigA', 0.9, 'sigACoast', 0.9, 'gate', 11.83, 'rBirth', 0.25, 'pSurvive', 0.998, ...
            'rConfirm', 0.75, 'rDelete', 0.08, 'maxSigma', 7, 'pd', 0.95, 'clutter', 0.8)
    end
    methods (Static)
        function Z = samples()
            % 64 fixed antithetic samples of a 4-D standard normal, identical to the JS ones
            persistent Zc
            if isempty(Zc)
                r = Mulberry32(424242); Zc = zeros(64, 4);
                for i = 1:32
                    v = [r.randn() r.randn() r.randn() r.randn()];
                    Zc(2*i-1, :) = v; Zc(2*i, :) = -v;
                end
            end
            Z = Zc;
        end
        function L = cholj(P)
            % lower Cholesky with the same growing diagonal jitter as Mat.chol in the JS
            jit = 0;
            for attempt = 1:8
                [L, flag] = chol(P + jit*eye(size(P)), 'lower');
                if flag == 0, return; end
                if jit == 0, jit = 1e-10; else, jit = jit*10; end
            end
            error('cholj:notPD', 'covariance not positive definite');
        end
        function T = emptyTracks()
            T = struct('id', {}, 'x', {}, 'P', {}, 'r', {}, 'hits', {}, 'age', {}, 'lastSeen', {}, ...
                'confirmed', {}, 'hidden', {}, 'pdExp', {}, 'ptsX', {}, 'ptsPd', {});
        end
    end
    methods
        function o = OccTracker(cam, boulders, variant, params)
            % variant: "naive" | "aware" | "negInfo"
            if nargin < 4, params = struct(); end
            o.cam = cam; o.boulders = boulders; o.sun = [];
            o.aware = variant ~= "naive"; o.negInfo = variant == "negInfo";
            names = struct('naive', "Naive", 'aware', "Occlusion-aware", 'negInfo', "Aware + neg. info");
            o.name = names.(char(variant));
            o.p = OccTracker.DEFAULTS;
            f = fieldnames(params);
            for k = 1:numel(f)
                if strcmp(f{k}, 'sun'), o.sun = params.sun; else, o.p.(f{k}) = params.(f{k}); end
            end
            o.tracks = OccTracker.emptyTracks();
        end

        function [pd, X, pds] = expectedPd(o, t)
            % P_D x visibility, averaged over 64 samples of the predicted Gaussian
            if ~o.aware, pd = o.p.pd; X = []; pds = []; return; end
            L = OccTracker.cholj(t.P);
            X = t.x' + OccTracker.samples() * L';
            v = occ_visibility(o.cam, o.boulders, X(:, 1), X(:, 2), o.cam.targetR);
            if isempty(o.sun), sh = 0; else, sh = occ_shadow(o.boulders, X(:, 1), X(:, 2), o.sun); end
            pds = occ_pd(v.inFov, v.visFrac, o.p.pd, sh);
            pd = mean(pds);
        end

        function step(o, dets)
            %STEP  Process one frame of detections, dets = K x 2 [range bearing] (no identities).
            p = o.p; c = o.cam; dt = o.dt;
            F = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];
            a = dt^4/4; b = dt^3/2; cc = dt^2;
            Q0 = [a 0 b 0; 0 a 0 b; b 0 cc 0; 0 b 0 cc];
            nT = numel(o.tracks);
            for i = 1:nT                                   % predict
                t = o.tracks(i);
                if t.lastSeen > 0, q = p.sigACoast^2; else, q = p.sigA^2; end
                t.x = F*t.x; t.P = F*t.P*F' + Q0*q; t.P = (t.P + t.P')/2;
                t.r = t.r * p.pSurvive; t.age = t.age + 1;
                [t.pdExp, t.ptsX, t.ptsPd] = o.expectedPd(t); t.hidden = t.pdExp < 0.3;
                o.tracks(i) = t;
            end
            kappa = p.clutter / (c.fov*(c.range - 4));    % clutter density per (m x rad)
            D = size(dets, 1); FORBID = 1e9; PG = 0.9973;
            cost = FORBID*ones(nT, D + nT); g = zeros(nT, D); pre = cell(nT, 1);
            for i = 1:nT
                t = o.tracks(i);
                dx = t.x(1) - c.x; dy = t.x(2) - c.y; qq = dx^2 + dy^2; rr = sqrt(qq);
                H = [dx/rr dy/rr 0 0; -dy/qq dx/qq 0 0];
                R = diag([LunarWorld.sigRange(c, rr)^2, c.sigB^2*2]);
                S = H*t.P*H' + R; Si = inv(S); zh = [rr; atan2(dy, dx)];
                pre{i} = struct('H', H, 'R', R, 'Si', Si, 'zh', zh);
                pdi = max(t.pdExp, 0.02); dS = S(1,1)*S(2,2) - S(1,2)*S(2,1);
                for j = 1:D
                    y = [dets(j,1) - zh(1); wrapang(dets(j,2) - zh(2))];
                    nis = y(1)*(Si(1,1)*y(1) + Si(1,2)*y(2)) + y(2)*(Si(2,1)*y(1) + Si(2,2)*y(2));
                    if nis > p.gate, continue; end
                    g(i,j) = exp(-0.5*nis)/(2*pi*sqrt(dS));
                    cost(i,j) = -log(pdi*g(i,j)/kappa);   % take detection j: likelihood ratio vs clutter
                end
                cost(i, D + i) = -log(1 - pdi*PG);        % or declare it missed
            end
            asg = hungarian(cost, FORBID);
            used = false(D, 1);
            for i = 1:nT
                t = o.tracks(i); j = asg(i);
                if j >= 1 && j <= D                       % EKF update
                    used(j) = true; m = pre{i};
                    y = [dets(j,1) - m.zh(1); wrapang(dets(j,2) - m.zh(2))];
                    K = t.P*m.H'*m.Si; t.x = t.x + K*y;
                    IKH = eye(4) - K*m.H; t.P = IKH*t.P*IKH' + K*m.R*K'; t.P = (t.P + t.P')/2;
                    pdH = max(t.pdExp, 0.02);
                    t.r = t.r*(pdH*g(i,j) + (1 - pdH)*kappa) / (t.r*pdH*g(i,j) + (1 - t.r*pdH)*kappa);
                    t.hits = t.hits + 1; t.lastSeen = 0;
                    if ~t.confirmed && t.r >= p.rConfirm && t.hits >= 3, t.confirmed = true; end
                else                                      % missed: Bayes existence update with the EXPECTED P_D
                    pd = t.pdExp;
                    t.r = t.r*(1 - pd)/(1 - t.r*pd); t.lastSeen = t.lastSeen + 1;
                    if o.negInfo, t = OccTracker.missUpdate(t); end
                end
                o.tracks(i) = t;
            end
            gated = any(cost(:, 1:D) < FORBID, 1);
            if nT
                keep = arrayfun(@(t) t.r >= p.rDelete && sqrt(max(t.P(1,1), t.P(2,2))) <= p.maxSigma, o.tracks);
                o.tracks = o.tracks(keep);
            end
            for j = 1:D                                   % births only outside every gate (no duplicates)
                if ~used(j) && ~gated(j), o.birth(dets(j, :)); end
            end
        end

        function birth(o, d)
            c = o.cam; cb = cos(d(2)); sb = sin(d(2)); r = d(1);
            J = [cb -r*sb; sb r*cb]; Rm = diag([LunarWorld.sigRange(c, r)^2, c.sigB^2*2]);
            P = diag([0 0 1 1]); P(1:2, 1:2) = J*Rm*J' + 0.05*eye(2);
            t = struct('id', o.nextId, 'x', [c.x + r*cb; c.y + r*sb; 0; 0], 'P', P, 'r', o.p.rBirth, 'hits', 1, ...
                'age', 0, 'lastSeen', 0, 'confirmed', false, 'hidden', false, 'pdExp', 1, 'ptsX', [], 'ptsPd', []);
            o.tracks(end+1) = t; o.nextId = o.nextId + 1;
        end

        function T = reported(o)
            % what the system would draw as boxes: confirmed and seen within 0.3 s (same rule for every variant)
            T = o.tracks([o.tracks.confirmed] & [o.tracks.lastSeen] <= 3);
        end
        function T = confirmedTracks(o)
            T = o.tracks([o.tracks.confirmed]);
        end
    end
    methods (Static)
        function t = missUpdate(t)
            % negative information: moment-match p(x)(1 - P_D(x)) over this frame's samples
            if isempty(t.ptsX), return; end
            w = 1 - t.ptsPd; W = sum(w); N = numel(w);
            if W < 1e-6 || W > N*(1 - 1e-6), return; end   % all hypotheses equally (in)visible: nothing learned
            m = (w' * t.ptsX) / W; Xc = t.ptsX - m;
            P = (Xc .* (w/W))' * Xc;
            P = P + diag([0.02 0.02 0.01 0.01]);            % floor: never overconfident after collapsing samples
            t.x = m'; t.P = (P + P')/2;
        end
    end
end
