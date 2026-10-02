function f = occ_shadow(B, X, Y, sun)
%OCC_SHADOW  Fraction of each body (3 heights) lying in a boulder's cast shadow under a low sun (az, el in rad).
%   Port of shadowFraction in src/mot/occlusion.js, vectorised over people.
X = X(:); Y = Y(:); f = zeros(size(X));
if isempty(sun), return; end
SH = [0.4 0.9 1.4]; LEN = 80;
cx = cos(sun.el)*cos(sun.az); cy = cos(sun.el)*sin(sun.az); cz = sin(sun.el);
dark = false(numel(X), numel(SH));
for k = 1:numel(B)
    b = B(k); h = b.h; if ~isfinite(h), h = 1e6; end
    bx = b.x - X; by = b.y - Y; along = bx*cos(sun.az) + by*sin(sun.az);
    near = along > -b.r & along < LEN & abs(-bx*sin(sun.az) + by*cos(sun.az)) < b.r + 0.5;
    if ~any(near), continue; end
    for j = 1:numel(SH)
        ox = (X - b.x)/b.r; oy = (Y - b.y)/b.r; oz = SH(j)/h;
        ddx = cx*LEN/b.r; ddy = cy*LEN/b.r; ddz = cz*LEN/h;
        a = ddx^2 + ddy^2 + ddz^2; bq = 2*(ox*ddx + oy*ddy + oz*ddz); c = ox.^2 + oy.^2 + oz^2 - 1;
        disc = bq.^2 - 4*a*c; s = sqrt(max(disc, 0));
        t1 = (-bq - s)/(2*a); t2 = (-bq + s)/(2*a);
        dark(:, j) = dark(:, j) | (near & disc > 0 & t1 < 1 & t2 > 0);
    end
end
f = sum(dark, 2) / numel(SH);
end
