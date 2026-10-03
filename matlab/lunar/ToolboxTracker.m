classdef ToolboxTracker < handle
    %TOOLBOXTRACKER  A MathWorks Sensor Fusion and Tracking Toolbox tracker run on our detections and scored by our
    %   MotMetrics, for an independent comparison. Kinds:
    %     "gnn"    trackerGNN    global nearest neighbour, score-based track logic
    %     "jipda"  trackerJPDA   joint probabilistic data association with TrackLogic='Integrated' (JIPDA): every
    %                            track carries an existence probability, the toolbox cousin of our Bernoulli existence
    %     "tomht"  trackerTOMHT  track-oriented multiple hypothesis tracking: keeps alternative assignments alive
    %   Aware=true: each frame the tracker is told every track's (or hypothesis branch's) visibility-based detection
    %   probability, the toolbox's own way to say "a miss behind a rock is expected" (detectable track/branch IDs).
    properties
        name; kind; aware; tk; cam; boulders; sun; pdMax
        tracks = struct('id', {}, 'x', {}, 'P', {}, 'confirmed', {}, 'lastSeen', {}, 'pdExp', {})
        t = 0; dt = 0.1; maxSigma = 7
    end
    properties (Access = private)
        lastSeenMap
    end
    methods
        function o = ToolboxTracker(world, prm, kind, aware)
            o.kind = string(kind); o.aware = aware; o.cam = world.cam; o.boulders = world.boulders; o.pdMax = prm.pd;
            if isfield(prm, 'sun'), o.sun = prm.sun; else, o.sun = []; end
            label = struct('gnn', "trackerGNN", 'jipda', "trackerJPDA (JIPDA)", 'tomht', "trackerTOMHT");
            o.name = label.(char(o.kind)) + ifelse(aware, " + visibility", "");
            c = o.cam; area = c.fov/2*(c.range^2 - 4^2); far = prm.clutter/area;     % false alarms per m^2
            init = @ToolboxTracker.initFilter;
            switch o.kind
                case "gnn"
                    o.tk = trackerGNN('FilterInitializationFcn', init, 'TrackLogic', 'Score', 'DetectionProbability', prm.pd, ...
                        'FalseAlarmRate', far, 'Volume', 1, 'Beta', 1e-3, 'ConfirmationThreshold', 18, 'DeletionThreshold', -7, ...
                        'AssignmentThreshold', [20 Inf], 'HasDetectableTrackIDsInput', aware, 'MaxNumTracks', 200);
                case "jipda"
                    o.tk = trackerJPDA('FilterInitializationFcn', init, 'TrackLogic', 'Integrated', 'DetectionProbability', prm.pd, ...
                        'ClutterDensity', far, 'NewTargetDensity', 1e-3, 'DeathRate', 0.002, ...
                        'ConfirmationThreshold', 0.75, 'DeletionThreshold', 0.08, 'AssignmentThreshold', [20 Inf], ...
                        'HasDetectableTrackIDsInput', aware, 'MaxNumTracks', 200);
                case "tomht"
                    o.tk = trackerTOMHT('FilterInitializationFcn', init, 'DetectionProbability', prm.pd, 'FalseAlarmRate', far, ...
                        'Volume', 1, 'Beta', 1e-3, 'ConfirmationThreshold', 18, 'DeletionThreshold', -7, ...
                        'AssignmentThreshold', [9 21 30 Inf], 'MaxNumHypotheses', 5, 'MaxNumTrackBranches', 3, ...
                        'HasDetectableBranchIDsInput', aware, 'MaxNumTracks', 200);
            end
            o.lastSeenMap = containers.Map('KeyType', 'double', 'ValueType', 'double');
        end

        function step(o, dets)
            o.t = o.t + o.dt; c = o.cam;
            D = cell(size(dets, 1), 1);
            for j = 1:size(dets, 1)                                     % range/bearing -> position with its covariance
                r = dets(j, 1); b = dets(j, 2);
                J = [cos(b) -r*sin(b); sin(b) r*cos(b)];
                C = J*diag([LunarWorld.sigRange(c, r)^2, c.sigB^2*2])*J';
                D{j} = objectDetection(o.t, [c.x + r*cos(b); c.y + r*sin(b)], 'MeasurementNoise', (C + C')/2);
            end
            if isempty(D) && ~isLocked(o.tk), return; end              % the toolbox needs a detection on its first update
            if o.aware
                pred = [];
                if isLocked(o.tk)
                    if o.kind == "tomht", pred = predictTracksToTime(o.tk, 'branch', 'all', o.t);   % every hypothesis branch
                    else, pred = predictTracksToTime(o.tk, 'all', o.t); end
                end
                ids = zeros(numel(pred), 2);
                for k = 1:numel(pred)
                    s = pred(k).State;
                    if o.kind == "tomht", ids(k, 1) = pred(k).BranchID; else, ids(k, 1) = pred(k).TrackID; end
                    v = occ_visibility(c, o.boulders, s(1), s(3), c.targetR);
                    sh = 0; if ~isempty(o.sun), sh = occ_shadow(o.boulders, s(1), s(3), o.sun); end
                    ids(k, 2) = max(occ_pd(v.inFov, v.visFrac, o.pdMax, sh), 1e-3);
                end
                [~, ~, all] = o.tk(D, o.t, ids);
            else
                [~, ~, all] = o.tk(D, o.t);
            end
            T = struct('id', {}, 'x', {}, 'P', {}, 'confirmed', {}, 'lastSeen', {}, 'pdExp', {});
            for k = 1:numel(all)
                a = all(k); id = double(a.TrackID); idx = [1 3 2 4];       % toolbox [x vx y vy] -> ours [x y vx vy]
                P = a.StateCovariance(idx, idx);
                if sqrt(max(P(1,1), P(2,2))) > o.maxSigma
                    % same uncertainty limit as ours. TOMHT keeps a branch history that external deletes would break,
                    % so for TOMHT the track is only left out of scoring and its own score logic deletes it
                    if o.kind ~= "tomht", try, deleteTrack(o.tk, a.TrackID); catch, end, end
                    continue
                end
                ls = 0; if o.lastSeenMap.isKey(id), ls = o.lastSeenMap(id); end
                if a.IsCoasted, ls = ls + 1; else, ls = 0; end
                o.lastSeenMap(id) = ls;
                T(end+1) = struct('id', id, 'x', a.State(idx), 'P', P, 'confirmed', a.IsConfirmed, 'lastSeen', ls, 'pdExp', NaN); %#ok<AGROW>
            end
            o.tracks = T;
        end
        function T = reported(o), T = o.tracks([o.tracks.confirmed] & [o.tracks.lastSeen] <= 3); end
        function T = confirmedTracks(o), T = o.tracks([o.tracks.confirmed]); end
    end
    methods (Static)
        function f = initFilter(det)
            z = det.Measurement; R = det.MeasurementNoise;
            P = diag([R(1,1)+0.05 1 R(2,2)+0.05 1]); P(1,3) = R(1,2); P(3,1) = R(1,2);
            f = trackingEKF(@constvel, @(s) s([1 3], :), [z(1); 0; z(2); 0], 'StateTransitionJacobianFcn', @constveljac, ...
                'MeasurementJacobianFcn', @(s) [1 0 0 0; 0 0 1 0], 'StateCovariance', P, 'MeasurementNoise', R, ...
                'ProcessNoise', 0.9^2*eye(2), 'HasAdditiveProcessNoise', false);
        end
    end
end
function v = ifelse(c, a, b), if c, v = a; else, v = b; end, end
