classdef LunarSession < handle
    %LUNARSESSION  One experiment: a world, its detections, and several trackers scored on the SAME detections.
    %   s = LunarSession("boulders", 7);  s.run(600);  s.table()
    %   Trackers: our three variants (port of src/mot/session.js), optionally MathWorks' trackerGNN as a fourth.
    properties
        world; runs; dets = zeros(0, 2); gt = zeros(0, 1)
    end
    methods
        function o = LunarSession(scenario, seed, opts)
            arguments
                scenario (1,1) string = "boulders"
                seed (1,1) double = 7
                opts.Toolbox (1,1) string {mustBeMember(opts.Toolbox, ["none" "aware" "both"])} = "none"
                opts.Overrides struct = struct()
            end
            o.world = LunarWorld(scenario, seed, opts.Overrides);
            W = o.world; prm = struct('pd', W.cfg.pd, 'clutter', W.cfg.clutter);
            if W.cfg.shadows, prm.sun = W.sun; end
            vars = ["negInfo" "aware" "naive"];
            o.runs = struct('tracker', {}, 'metrics', {});
            for v = vars
                o.runs(end+1) = struct('tracker', OccTracker(W.cam, W.boulders, v, prm), 'metrics', MotMetrics(numel(W.targets)));
            end
            % independent cross-check: MathWorks trackerGNN, told each track's visibility ("aware"), and/or plain
            if opts.Toolbox ~= "none"
                o.runs(end+1) = struct('tracker', ToolboxGNN(W, prm, true), 'metrics', MotMetrics(numel(W.targets)));
            end
            if opts.Toolbox == "both"
                o.runs(end+1) = struct('tracker', ToolboxGNN(W, prm, false), 'metrics', MotMetrics(numel(W.targets)));
            end
        end
        function step(o)
            o.world.step();
            [o.dets, o.gt] = o.world.sense();
            for k = 1:numel(o.runs)
                o.runs(k).tracker.step(o.dets);           % trackers never see identities
                o.runs(k).metrics.update(o.world, o.runs(k).tracker);
            end
        end
        function run(o, seconds)
            for i = 1:round(seconds/o.world.dt), o.step(); end
        end
        function T = table(o)
            names = strings(numel(o.runs), 1); kept = names; idsw = zeros(numel(o.runs), 1); idf1 = idsw; mota = idsw; gospa = idsw;
            for k = 1:numel(o.runs)
                s = o.runs(k).metrics.summary(); b = s.occByDuration(1:2);
                names(k) = o.runs(k).tracker.name;
                kept(k) = sprintf('%d / %d', sum([b.kept]), sum([b.n]));
                idsw(k) = s.idsw; idf1(k) = s.idf1; mota(k) = s.mota; gospa(k) = s.gospa;
            end
            T = table(names, kept, idsw, idf1, mota, gospa, 'VariableNames', ...
                {'Tracker', 'ID kept 1-5 s', 'ID switches', 'IDF1', 'MOTA', 'GOSPA'});
        end
    end
end
