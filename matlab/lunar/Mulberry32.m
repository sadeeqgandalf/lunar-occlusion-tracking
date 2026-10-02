classdef Mulberry32 < handle
    %MULBERRY32  Seeded PRNG + Box-Muller gaussian, bit-identical to src/core/rng.js.
    %   Same seed -> same boulders, walkers and detections as the web lab, so results can be cross-checked.
    properties (Access = private)
        s
        spare = []
    end
    methods
        function o = Mulberry32(seed)
            o.s = mod(double(seed), 2^32);
        end
        function u = next(o)
            o.s = mod(o.s + 1831565813, 2^32);                       % + 0x6D2B79F5
            t = o.s;
            t = Mulberry32.imul(bitxor(t, floor(t / 2^15)), bitor(t, 1));
            t = bitxor(t, mod(t + Mulberry32.imul(bitxor(t, floor(t / 2^7)), bitor(t, 61)), 2^32));
            u = bitxor(t, floor(t / 2^14)) / 4294967296;
        end
        function v = uniform(o, a, b)
            v = a + (b - a) * o.next();
        end
        function z = randn(o)
            if ~isempty(o.spare), z = o.spare; o.spare = []; return; end
            u = 0;
            while u == 0, u = o.next(); end
            v = o.next();
            m = sqrt(-2 * log(u));
            o.spare = m * sin(2 * pi * v);
            z = m * cos(2 * pi * v);
        end
    end
    methods (Static)
        function r = imul(a, b)
            % low 32 bits of a*b for 0 <= a, b < 2^32, exact in doubles (each partial product < 2^49)
            alo = mod(a, 65536); ahi = floor(a / 65536);
            r = mod(alo * b + mod(ahi * b, 65536) * 65536, 2^32);
        end
    end
end
