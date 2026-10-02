function pd = occ_pd(inFov, visFrac, pdMax, shadow)
%OCC_PD  Detection probability: needs >= 25% of the body in view, full rate from 70%; a fully shadowed person
%   is detected 80% less often. Port of detectionProb in src/mot/occlusion.js (vectorised).
if nargin < 4, shadow = 0; end
pd = pdMax * min(1, (visFrac - 0.25)/0.45) .* (1 - 0.8*shadow);
pd(~inFov | visFrac < 0.25) = 0;
end
