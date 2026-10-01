# Cross-language EKF benchmark

One recorded sensor trace (rover, scenario `dust`, seed 7, oracle autopilot): 3000 predict steps + 270 measurements.
All three implementations replay the **same file** (`.build/trace.txt`), re-initialising the filter for each repetition, so the final state is directly comparable.

| impl | final-state max abs diff vs JS | µs per predict step (incl. amortised updates) | repetitions timed |
|---|---|---:|---:|
| JS (shipped `src/filters/ekf.js`) | – | 2.43 | 100 |
| C++17 (`ekf.cpp`) | 4.02e-10 | 0.131 | 100 |
| Python + numpy (`ekf.py`) | 4.02e-10 | 8.95 | 20 |

Machine: Apple M5, 10 cores, darwin 25.6.0
Toolchain: node v25.6.1; Apple clang version 21.0.0 (clang-2100.3.34.2); flags `-O2 -std=c++17`; Python 3.14.3, numpy 2.4.4

Caveats: single trace, single machine, single run (no variance reported); C++ uses fixed-size stack matrices, Python uses numpy on 5x5 arrays where call overhead dominates (a plain-Python or compiled-numpy comparison would differ); JS time includes the generic Filter/Mat abstraction of the shipped code. Python is timed over fewer repetitions for run time only. Timings are not pinned to a core and vary a few percent between runs.
