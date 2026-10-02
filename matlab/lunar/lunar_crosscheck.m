function ok = lunar_crosscheck(scenario, seed, seconds)
%LUNAR_CROSSCHECK  Run the same scenario and seed in MATLAB and in the web lab's JavaScript, compare the numbers.
%   lunar_crosscheck                         % boulder field, seed 7, 120 s
%   lunar_crosscheck("polar", 3, 300)
%   Needs Node.js on the PATH (it runs js_reference.mjs). Boulders, walks and detections come from the same
%   seeded generator, so they should agree to rounding; tracker scores should then agree exactly.
arguments
    scenario (1,1) string = "boulders"
    seed (1,1) double = 7
    seconds (1,1) double = 120
end
here = fileparts(mfilename('fullpath'));
node = "node"; if ~isempty(getenv('NODE')), node = string(getenv('NODE')); end
for cand = ["/opt/homebrew/bin/node" "/usr/local/bin/node"], if isfile(cand), node = cand; end, end
[st, out] = system(sprintf('cd "%s" && "%s" js_reference.mjs %s %d %g', here, node, scenario, seed, seconds));
if st ~= 0, error('lunar_crosscheck:node', 'Node.js run failed:\n%s', out); end
js = jsondecode(strtrim(out));

s = LunarSession(scenario, seed); nDets = 0;
for i = 1:round(seconds/s.world.dt), s.step(); nDets = nDets + size(s.dets, 1); end

B = s.world.boulders; bm = [[B.x]' [B.y]' [B.r]' [B.h]'];
tm = [[s.world.targets.x]' [s.world.targets.y]'];
fprintf('\n%s · seed %d · %g s\n', scenario, seed, seconds);
fprintf('  boulders      max difference %.1e m\n', max(abs(bm - js.boulders), [], 'all'));
fprintf('  people (end)  max difference %.1e m\n', max(abs(tm - js.truth), [], 'all'));
fprintf('  detections    MATLAB %d   JS %d\n\n', nDets, js.nDets);
fprintf('  %-20s %14s %14s %16s %16s\n', 'tracker', 'ID switches', 'IDs kept', 'IDF1', 'MOTA');
ok = max(abs(tm - js.truth), [], 'all') < 1e-6 && nDets == js.nDets;
for k = 1:numel(s.runs)
    m = s.runs(k).metrics.summary(); j = js.runs(k);
    same = m.idsw == j.idsw && m.occKept == j.occKept && m.occEvents == j.occEvents && abs(m.idf1 - j.idf1) < 1e-9;
    ok = ok && same;
    fprintf('  %-20s %6d | %-6d %6s | %-6s %7.4f | %-7.4f %7.4f | %-7.4f %s\n', s.runs(k).tracker.name, m.idsw, j.idsw, ...
        sprintf('%d/%d', m.occKept, m.occEvents), sprintf('%d/%d', j.occKept, j.occEvents), m.idf1, j.idf1, m.mota, j.mota, ...
        string(ifelse(same, 'match', 'DIFFERENT')));
end
fprintf('\n  (each column: MATLAB | JavaScript)\n  Result: %s\n', string(ifelse(ok, 'MATLAB reproduces the web lab exactly', 'differences found')));
end
function v = ifelse(c, a, b), if c, v = a; else, v = b; end, end
