function assign = hungarian(cost, forbid)
%HUNGARIAN  Kuhn-Munkres minimum-cost assignment, rectangular via padding. Line-for-line port of
%   src/mot/hungarian.js so ties resolve identically. assign(i) = column for row i, or 0 if unassigned.
%   Pairs whose cost >= forbid are never returned.
if nargin < 2, forbid = 1e9; end
[nr, nc] = size(cost); n = max(nr, nc); assign = zeros(nr, 1);
if nr == 0 || nc == 0, return; end   % JS: no rows -> no columns -> nothing to assign
a = forbid * ones(n + 1); a(1, :) = 0; a(:, 1) = 0;
if nr && nc, a(2:nr+1, 2:nc+1) = min(cost, forbid); end
u = zeros(1, n + 1); v = zeros(1, n + 1); p = zeros(1, n + 1); way = zeros(1, n + 1);
for i = 1:n
    p(1) = i; j0 = 0;
    minv = inf(1, n + 1); used = false(1, n + 1);
    while true
        used(j0 + 1) = true;
        i0 = p(j0 + 1); delta = inf; j1 = 0;
        for j = 1:n
            if ~used(j + 1)
                cur = a(i0 + 1, j + 1) - u(i0 + 1) - v(j + 1);
                if cur < minv(j + 1), minv(j + 1) = cur; way(j + 1) = j0; end
                if minv(j + 1) < delta, delta = minv(j + 1); j1 = j; end
            end
        end
        for j = 0:n
            if used(j + 1), u(p(j + 1) + 1) = u(p(j + 1) + 1) + delta; v(j + 1) = v(j + 1) - delta;
            else, minv(j + 1) = minv(j + 1) - delta; end
        end
        j0 = j1;
        if p(j0 + 1) == 0, break; end
    end
    while true
        j1 = way(j0 + 1); p(j0 + 1) = p(j1 + 1); j0 = j1;
        if j0 == 0, break; end
    end
end
for j = 1:n
    i = p(j + 1);
    if i >= 1 && i <= nr && j <= nc && cost(i, j) < forbid, assign(i) = j; end
end
end
