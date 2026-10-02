function v = occ_visibility(cam, B, X, Y, rt)
%OCC_VISIBILITY  3-D line of sight from the mast camera to upright people at (X, Y) (column vectors).
%   16 sight lines per person (4 heights x 4 lateral offsets) are tested against half-buried ellipsoid boulders.
%   A low rock hides legs but not heads, exactly as a real camera sees it. Vectorised over people and rays.
%   Returns struct with fields inFov, range, bearing, visFrac, visCenter (each N x 1). Port of src/mot/occlusion.js.
if nargin < 5, rt = 0.4; end
RAY_H = [0.3 0.8 1.3 1.7]; RAY_L = [-0.28 -0.09 0.09 0.28];
X = X(:); Y = Y(:);
dx = X - cam.x; dy = Y - cam.y; range = hypot(dx, dy); bearing = atan2(dy, dx);
inFov = range <= cam.range & abs(wrapang(bearing - cam.th)) <= cam.fov/2 & range > 0.5;
sc = rt / 0.4;
L = repelem(RAY_L, numel(RAY_H)) * sc;           % 1 x 16, same order as the JS loops (lateral outer, height inner)
Hz = repmat(RAY_H, 1, numel(RAY_L));
nx = -sin(bearing); ny = cos(bearing);
PX = X + L .* nx; PY = Y + L .* ny;              % N x 16 ray end points
hit = false(size(PX));
for k = 1:numel(B)
    b = B(k); h = b.h; if ~isfinite(h), h = 1e6; end
    ox = (cam.x - b.x)/b.r; oy = (cam.y - b.y)/b.r; oz = cam.h/h;
    ddx = (PX - cam.x)/b.r; ddy = (PY - cam.y)/b.r; ddz = (Hz - cam.h)/h;
    a = ddx.^2 + ddy.^2 + ddz.^2; bq = 2*(ox*ddx + oy*ddy + oz*ddz); c = ox^2 + oy^2 + oz^2 - 1;
    disc = bq.^2 - 4*a.*c; s = sqrt(max(disc, 0));
    t1 = (-bq - s)./(2*a); t2 = (-bq + s)./(2*a);
    hit = hit | (disc > 0 & t1 < 1 & t2 > 0);
end
vis = sum(~hit, 2); visFrac = vis / numel(L);
latSum = sum((~hit) .* L, 2);
visCenter = bearing;
part = vis > 0 & vis < numel(L);                 % fully visible: centre = bearing exactly (as the JS short-cut)
visCenter(part) = wrapang(bearing(part) + atan2(latSum(part) ./ vis(part), range(part)));
v = struct('inFov', inFov, 'range', range, 'bearing', bearing, 'visFrac', visFrac, 'visCenter', visCenter);
end
