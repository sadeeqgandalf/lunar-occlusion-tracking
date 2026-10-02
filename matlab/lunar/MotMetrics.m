classdef MotMetrics < handle
    %MOTMETRICS  Tracking scores against ground truth. Port of src/mot/metrics.js.
    %   * Reported tracks = confirmed and seen within 0.3 s (identical rule for every tracker).
    %   * CLEAR-MOT (Bernardin & Stiefelhagen 2008) [E1]: MOTA, MOTP, ID switches, keeping previous matches.
    %   * IDF1 (Ristani et al. 2016) [E2]. GOSPA (Rahmathullah et al. 2017) [E3], p=2, c=2 m, alpha=2.
    %   * Occlusion events: tracked before hiding for >= 1 s; success = same track ID within 2 s of reappearing.
    properties
        f = 0; gtCount = 0; tp = 0; fp = 0; fn = 0; idsw = 0; distSum = 0; gospaSum = 0; predCount = 0
        lastMatch                      % lastMatch(personId) = track id, NaN = never matched
        pairCounts = zeros(0, 0)       % pairCounts(personId, trackId): frames within 2 m (IDF1)
        occ                            % per-person occlusion state machine
        evts = struct('t', {}, 'target', {}, 'dur', {}, 'kept', {})
        hiddenFrames = 0; hiddenAlive = 0; hiddenCovered = 0; hiddenErrSum = 0
        switchLog = zeros(0, 4)        % [time personId oldTrackId newTrackId]
    end
    properties (Constant)
        TAU = 2.0; GOSPA_C = 2.0; MIN_OCC = 1.0; REACQ = 2.0; CHI2_95_2 = 5.991
    end
    methods
        function o = MotMetrics(nPeople)
            o.lastMatch = nan(1, nPeople);
            o.occ = repmat(struct('state', "idle", 'since', NaN, 'before', NaN, 'dur', NaN, 'lvm', NaN), 1, nPeople);
        end

        function update(o, world, tracker)
            T = world.t; F = world.frame; G0 = world.targets;
            V = [G0.vis]; inFov = [V.inFov]; hid = [V.inFov] & [V.pd] < 0.15;
            vis = G0(inFov & ~hid);
            rep = tracker.reported();
            o.f = o.f + 1; o.gtCount = o.gtCount + numel(vis); o.predCount = o.predCount + numel(rep);
            if isempty(rep), rx = zeros(0, 2); rid = zeros(1, 0); else, rx = [arrayfun(@(k) k.x(1), rep)' arrayfun(@(k) k.x(2), rep)']; rid = [rep.id]; end
            gx = [reshape([vis.x], [], 1) reshape([vis.y], [], 1)]; gid = reshape([vis.id], 1, []);   % shapes hold when nobody is visible
            Dm = hypot(gx(:,1) - rx(:,1)', gx(:,2) - rx(:,2)');                 % person x track distances
            if isempty(Dm), Dm = zeros(numel(vis), numel(rep)); end

            % --- CLEAR-MOT: keep previous correspondences that are still valid, Hungarian for the rest
            matchK = zeros(1, numel(vis)); usedK = false(1, numel(rep));
            for a = 1:numel(vis)
                kid = o.lastMatch(gid(a));
                if isnan(kid), continue; end
                kk = find(rid == kid, 1);
                if ~isempty(kk) && ~usedK(kk) && Dm(a, kk) <= o.TAU, matchK(a) = kk; usedK(kk) = true; end
            end
            Gi = find(matchK == 0); Ki = find(~usedK);
            C = Dm(Gi, Ki); C(C > o.TAU) = 1e9;
            asg = hungarian(C, 1e9);
            for q = 1:numel(Gi), if asg(q) > 0, matchK(Gi(q)) = Ki(asg(q)); usedK(Ki(asg(q))) = true; end, end
            for a = find(matchK > 0)
                k = rid(matchK(a)); prev = o.lastMatch(gid(a));
                if ~isnan(prev) && prev ~= k, o.idsw = o.idsw + 1; o.switchLog(end+1, :) = [T gid(a) prev k]; end
                o.lastMatch(gid(a)) = k; o.tp = o.tp + 1; o.distSum = o.distSum + Dm(a, matchK(a));
            end
            nM = nnz(matchK);
            o.fn = o.fn + numel(vis) - nM; o.fp = o.fp + numel(rep) - nM;

            % --- IDF1 co-occurrence
            [ia, ka] = find(Dm <= o.TAU);
            for q = 1:numel(ia)
                gi = gid(ia(q)); ki = rid(ka(q));
                if size(o.pairCounts, 1) < gi || size(o.pairCounts, 2) < ki, o.pairCounts(gi, ki) = 0; end
                o.pairCounts(gi, ki) = o.pairCounts(gi, ki) + 1;
            end

            % --- GOSPA
            C = Dm.^2; C(Dm >= o.GOSPA_C) = 1e9;
            ga = hungarian(C, 1e9); gs = 0; nA = 0;
            for q = 1:numel(ga), if ga(q) > 0, gs = gs + Dm(q, ga(q))^2; nA = nA + 1; end, end
            gs = gs + o.GOSPA_C^2/2 * (numel(vis) - nA + numel(rep) - nA);
            o.gospaSum = o.gospaSum + sqrt(gs);

            % --- occlusion events
            for gI = 1:numel(G0)
                g = G0(gI); id = g.id; hiddenNow = hid(gI); visibleNow = inFov(gI) && ~hiddenNow;
                s = o.occ(id);
                if s.state == "idle" && hiddenNow && ~isnan(o.lastMatch(id)) && s.lvm >= F - 5
                    s = MotMetrics.st("hidden", T, o.lastMatch(id), NaN);
                elseif s.state == "hidden"
                    if ~inFov(gI)
                        s = MotMetrics.st("idle", NaN, NaN, NaN);         % walked out of view: not an occlusion
                    elseif visibleNow
                        if T - s.since >= o.MIN_OCC, s = MotMetrics.st("reacq", T, s.before, T - s.since);
                        else, s = MotMetrics.st("idle", NaN, NaN, NaN); end
                    else                                                   % still hidden: is the coasting track honest?
                        o.hiddenFrames = o.hiddenFrames + 1;
                        k = tracker.tracks([tracker.tracks.id] == s.before);
                        if ~isempty(k)
                            o.hiddenAlive = o.hiddenAlive + 1;
                            e = [g.x - k.x(1); g.y - k.x(2)]; Pp = k.P(1:2, 1:2);
                            o.hiddenErrSum = o.hiddenErrSum + norm(e);
                            if det(Pp) > 0 && e'/Pp*e <= o.CHI2_95_2, o.hiddenCovered = o.hiddenCovered + 1; end
                        end
                    end
                elseif s.state == "reacq"
                    a = find(gid == id, 1); m = 0;
                    if ~isempty(a) && matchK(a) > 0, m = rid(matchK(a)); end
                    if m > 0
                        o.evts(end+1) = struct('t', T, 'target', id, 'dur', s.dur, 'kept', m == s.before);
                        s = MotMetrics.st("idle", NaN, NaN, NaN);
                    elseif T - s.since > o.REACQ || ~inFov(gI)
                        o.evts(end+1) = struct('t', T, 'target', id, 'dur', s.dur, 'kept', false);
                        s = MotMetrics.st("idle", NaN, NaN, NaN);
                    end
                end
                a = find(gid == id, 1);
                if visibleNow && ~isempty(a) && matchK(a) > 0, s.lvm = F; end
                o.occ(id) = s;
            end
        end

        function S = summary(o)
            pc = o.pairCounts; idtp = 0;
            if ~isempty(pc) && any(pc(:))
                gI = find(any(pc, 2)); kI = find(any(pc, 1));
                maxC = max(1, max(pc(:)));
                a = hungarian(maxC - pc(gI, kI), 1e12);
                for q = 1:numel(a), if a(q) > 0, idtp = idtp + pc(gI(q), kI(a(q))); end, end
            end
            ev = o.evts; kept = [ev.kept]; dur = [ev.dur];
            bins = [1 2; 2 5; 5 10; 10 inf]; byDur = struct('range', {}, 'n', {}, 'kept', {});
            for q = 1:4
                in = dur >= bins(q,1) & dur < bins(q,2);
                byDur(q) = struct('range', bins(q,:), 'n', nnz(in), 'kept', nnz(kept & in));
            end
            S = struct('frames', o.f, 'mota', 1 - (o.fn + o.fp + o.idsw)/max(o.gtCount, eps), ...
                'motp', o.distSum/max(o.tp, 1), 'idf1', 2*idtp/max(o.gtCount + o.predCount, eps), ...
                'idsw', o.idsw, 'fp', o.fp, 'fn', o.fn, 'gospa', o.gospaSum/max(1, o.f), ...
                'occEvents', numel(ev), 'occKept', nnz(kept), 'occByDuration', byDur, ...
                'hiddenAliveFrac', o.hiddenAlive/max(o.hiddenFrames, 1), 'hiddenCoverage', o.hiddenCovered/max(o.hiddenAlive, 1));
        end
    end
    methods (Static)
        function s = st(state, since, before, dur)
            s = struct('state', state, 'since', since, 'before', before, 'dur', dur, 'lvm', NaN);
        end
    end
end
