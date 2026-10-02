classdef ToolboxGNN < handle
    %TOOLBOXGNN  MathWorks Sensor Fusion and Tracking Toolbox trackerGNN, wrapped to run on the same detections
    %   and be scored by the same MotMetrics as our trackers. An independent implementation for cross-checking.
    %
    %   Same ingredients as ours where the toolbox allows: constant-velocity EKF (constvel), score-based track
    %   logic (log-likelihood ratio, the sequential test behind our existence probability), Pd = 0.95, the same
    %   false-alarm density. With Aware=true it is told, every frame, each track's visibility-based detection
    %   probability (HasDetectableTrackIDsInput): the toolbox's own way of saying "a miss behind a rock is expected".
    properties
        name; aware; tk; cam; boulders; sun; pdMax
        tracks = struct('id', {}, 'x', {}, 'P', {}, 'confirmed', {}, 'lastSeen', {}, 'pdExp', {})
        t = 0; dt = 0.1; maxSigma = 7
    end
    properties (Access = private)
        lastSeenMap
    end
    methods
        function o = ToolboxGNN(world, prm, aware)
            if nargin < 3, aware = true; end
            o.aware = aware; o.cam = world.cam; o.boulders = world.boulders; o.pdMax = prm.pd;
            if isfield(prm, 'sun'), o.sun = prm.sun; else, o.sun = []; end
            if aware, o.name = "trackerGNN + visibility"; else, o.name = "trackerGNN (toolbox)"; end
            c = o.cam; area = c.fov/2*(c.range^2 - 4^2);          % field-of-view sector, m^2
            o.tk = trackerGNN('FilterInitializationFcn', @ToolboxGNN.initFilter, 'TrackLogic', 'Score', ...
                'DetectionProbability', prm.pd, 'FalseAlarmRate', prm.clutter/area, 'Volume', 1, 'Beta', 1e-3, ...
                'ConfirmationThreshold', 18, 'DeletionThreshold', -7, 'AssignmentThreshold', [20 Inf], ...
                'HasDetectableTrackIDsInput', aware, 'MaxNumTracks', 200);
            o.lastSeenMap = containers.Map('KeyType', 'double', 'ValueType', 'double');
        end

        function step(o, dets)
            o.t = o.t + o.dt; c = o.cam;
            D = cell(size(dets, 1), 1);
            for j = 1:size(dets, 1)                            % range/bearing -> position with its covariance
                r = dets(j, 1); b = dets(j, 2);
                J = [cos(b) -r*sin(b); sin(b) r*cos(b)];
                C = J*diag([LunarWorld.sigRange(c, r)^2, c.sigB^2*2])*J';
                D{j} = objectDetection(o.t, [c.x + r*cos(b); c.y + r*sin(b)], 'MeasurementNoise', (C + C')/2);
            end
            if isempty(D) && ~isLocked(o.tk), return; end   % the toolbox needs a detection on its very first update
            if o.aware
                pred = [];
                if isLocked(o.tk), pred = predictTracksToTime(o.tk, 'all', o.t); end   % no tracks before the first update
                ids = zeros(numel(pred), 2);
                for k = 1:numel(pred)
                    s = pred(k).State; ids(k, 1) = pred(k).TrackID;
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
                a = all(k); id = double(a.TrackID); idx = [1 3 2 4];   % toolbox state [x vx y vy] -> ours [x y vx vy]
                P = a.StateCovariance(idx, idx);
                if sqrt(max(P(1,1), P(2,2))) > o.maxSigma, deleteTrack(o.tk, a.TrackID); continue; end  % same rule as ours
                ls = 0; if o.lastSeenMap.isKey(id), ls = o.lastSeenMap(id); end
                if a.IsCoasted, ls = ls + 1; else, ls = 0; end
                o.lastSeenMap(id) = ls;
                T(end+1) = struct('id', id, 'x', a.State(idx), 'P', P, 'confirmed', a.IsConfirmed, 'lastSeen', ls, 'pdExp', NaN); %#ok<AGROW>
            end
            o.tracks = T;
        end

        function T = reported(o)
            T = o.tracks([o.tracks.confirmed] & [o.tracks.lastSeen] <= 3);
        end
        function T = confirmedTracks(o)
            T = o.tracks([o.tracks.confirmed]);
        end
    end
    methods (Static)
        function f = initFilter(det)
            % constant-velocity EKF in 2-D, initialised at the detection with 1 m/s velocity uncertainty
            z = det.Measurement; R = det.MeasurementNoise;
            P = diag([R(1,1)+0.05 1 R(2,2)+0.05 1]); P(1,3) = R(1,2); P(3,1) = R(1,2);
            f = trackingEKF(@constvel, @(s) s([1 3], :), [z(1); 0; z(2); 0], 'StateTransitionJacobianFcn', @constveljac, ...
                'MeasurementJacobianFcn', @(s) [1 0 0 0; 0 0 1 0], 'StateCovariance', P, 'MeasurementNoise', R, ...
                'ProcessNoise', 0.9^2*eye(2), 'HasAdditiveProcessNoise', false);
        end
    end
end
