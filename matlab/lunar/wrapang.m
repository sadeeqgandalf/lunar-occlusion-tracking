function a = wrapang(a)
%WRAPANG  Wrap angles to [-pi, pi), same arithmetic as wrapAngle in src/core/linalg.js.
a = rem(a + pi, 2*pi);
a(a < 0) = a(a < 0) + 2*pi;
a = a - pi;
end
