# References

Every method in this repository traces to the literature below. Entries marked ✓ were checked against the publisher, an indexing service or the authors' open-access copy (venue, year and pages) on 2026-10-01. Links are given where a stable public copy exists.

## Tracking, data association and track existence

| # | Reference | Used for (code) |
|---|---|---|
| R1 ✓ | D. Musicki, R. Evans, S. Stankovic. **Integrated probabilistic data association.** *IEEE Transactions on Automatic Control* 39(6):1237–1241, 1994. | Track existence probability, recursively updated by detections and misses (`src/mot/tracker.js`). |
| R2 ✓ | B. Ristic, B.-T. Vo, B.-N. Vo, A. Farina. **A tutorial on Bernoulli filters: theory, implementation and applications.** *IEEE Transactions on Signal Processing* 61(13):3406–3430, 2013. doi:10.1109/TSP.2013.2257765 | The exact existence update with clutter density: detected `r ← r[P_D g + (1−P_D)κ] / [r P_D g + (1−r P_D)κ]`, missed `r ← r(1−P_D)/(1−r P_D)` (`tracker.js`). |
| R3 ✓ | W. Koch. **On exploiting 'negative' sensor evidence for target tracking and sensor data fusion.** *Information Fusion* 8(1):28–39, 2007. doi:10.1016/j.inffus.2005.09.002 | "Expected but missing" measurements as information: state-dependent detection probability and the negative-information update `p(x | miss) ∝ p(x)(1 − P_D(x))` (`tracker.js`, `_missUpdate`). |
| R4 ✓ | M. Yang, Y. Liu, L. Wen, Z. You, S. Z. Li. **A probabilistic framework for multitarget tracking with mutual occlusions.** *CVPR* 2014. [open access](https://openaccess.thecvf.com/content_cvpr_2014/papers/Yang_A_Probabilistic_Framework_2014_CVPR_paper.pdf) | Motivation for modelling occlusion inside the likelihood: an occluded target's observation likelihood vanishes even when its location is known. Our aware trackers condition `P_D` on predicted visibility. |
| R5 | Y. Bar-Shalom, X. R. Li, T. Kirubarajan. **Estimation with Applications to Tracking and Navigation.** Wiley, 2001. | Constant-velocity (white-noise acceleration) model, χ² gating, NEES/NIS consistency. |
| R6 | S. Blackman, R. Popoli. **Design and Analysis of Modern Tracking Systems.** Artech House, 1999. | Likelihood-ratio track scoring; assignment with an explicit "missed" option `−log(1 − P_D P_G)`. |
| R7 | R. Mahler. **Statistical Multisource-Multitarget Information Fusion.** Artech House, 2007. | Random-finite-set view of state-dependent detection probability and clutter (background for R2). |
| R8 | H. W. Kuhn. **The Hungarian method for the assignment problem.** *Naval Research Logistics Quarterly* 2(1–2):83–97, 1955. J. Munkres. **Algorithms for the assignment and transportation problems.** *J. SIAM* 5(1):32–38, 1957. | Optimal one-to-one assignment (`src/mot/hungarian.js`). |
| R9 | A. Bewley, Z. Ge, L. Ott, F. Ramos, B. Upcroft. **Simple online and realtime tracking (SORT).** *ICIP* 2016. | The "Kalman filter + Hungarian assignment" baseline family our tracker belongs to. |
| R10 ✓ | J. Cao, J. Pang, X. Weng, R. Khirodkar, K. Kitani. **Observation-Centric SORT: rethinking SORT for robust multi-object tracking.** *CVPR* 2023, 9686–9696. [open access](https://openaccess.thecvf.com/content/CVPR2023/html/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.html) | Related approach to the same problem: error accumulates during occlusion under linear motion. Planned comparison (roadmap). |

## Evaluation

| # | Reference | Used for (code) |
|---|---|---|
| E1 | K. Bernardin, R. Stiefelhagen. **Evaluating multiple object tracking performance: the CLEAR MOT metrics.** *EURASIP Journal on Image and Video Processing* 2008. | MOTA, MOTP, ID switches, keeping previous correspondences when still valid (`src/mot/metrics.js`). |
| E2 ✓ | E. Ristani, F. Solera, R. Zou, R. Cucchiara, C. Tomasi. **Performance measures and a data set for multi-target, multi-camera tracking.** *ECCV Workshops* 2016, LNCS 9914:17–35. | IDF1: global identity mapping (`metrics.js`). |
| E3 ✓ | A. S. Rahmathullah, Á. F. García-Fernández, L. Svensson. **Generalized optimal sub-pattern assignment metric (GOSPA).** *FUSION* 2017. doi:10.23919/ICIF.2017.8009645, [arXiv:1601.05585](https://arxiv.org/abs/1601.05585) | GOSPA with p = 2, c = 2 m, α = 2 (`metrics.js`). |
| E4 | J. Luiten et al. **HOTA: a higher order metric for evaluating multi-object tracking.** *IJCV* 129:548–578, 2021. | Not yet computed here; the metric to use on real data via TrackEval (roadmap). |
| E5 | E. B. Wilson. **Probable inference, the law of succession, and statistical inference.** *JASA* 22(158):209–212, 1927. | Wilson 95% intervals on "ID kept" proportions (`experiments/occlusion_ablation.mjs`). |
| E6 | A. Milan, L. Leal-Taixé, I. Reid, S. Roth, K. Schindler. **MOT16: a benchmark for multi-object tracking.** arXiv:1603.00831, 2016. | Real occlusion-labelled data for future validation; per-object visibility ratio like ours. |

## Geometry, sensing and the lunar environment

| # | Reference | Used for (code) |
|---|---|---|
| G1 | R. Hartley, A. Zisserman. **Multiple View Geometry in Computer Vision**, 2nd ed. Cambridge University Press, 2004. | Pinhole projection (rendered camera views). Stereo depth `z = fB/d` gives `δz ≈ z²δd/(fB)`, which is why our range noise grows with the square of range (`src/mot/world.js`, `sigRange`). |
| G2 ✓ | E. Mazarico, G. A. Neumann, D. E. Smith, M. T. Zuber, M. H. Torrence. **Illumination conditions of the lunar polar regions using LOLA topography.** *Icarus* 211(2):1066–1081, 2011. | Polar lighting. The Moon's spin axis is tilted only ~1.5° to the ecliptic, so the sun stays within a few degrees of the horizon at the poles and casts very long shadows (the "South pole · Long shadows" scenario; we use ~8° for readability). |
| G3 ✓ | R. P. Mueller, R. E. Cox, T. Ebert, J. D. Smith, J. M. Schuler, A. J. Nick. **Regolith Advanced Surface Systems Operations Robot (RASSOR).** *IEEE Aerospace Conference* 2013. | The NASA lunar robot model carrying our camera mast (visual only). |
| G4 | S. Thrun, W. Burgard, D. Fox. **Probabilistic Robotics.** MIT Press, 2005. | EKF/UKF/particle filters of the navigation lab (see `docs/THEORY.md`). |
| G6 ✓ | Smithsonian National Air and Space Museum. **Cernan's spacesuit, Apollo 17** (object record). [link](https://airandspace.si.edu/exhibition/outside-spacecraft/image-details/5263.html) | Real identity markings: red commander stripes were added to EVA suits from Apollo 13 so TV cameras could tell the moonwalkers apart. This is the basis of our suit-stripe colour code for *true* identity (`src/ui/idcolor.js`). |
| G5 | W. H. Clohessy, R. S. Wiltshire. **Terminal guidance system for satellite rendezvous.** *Journal of the Aerospace Sciences* 27(9):653–658, 1960. | Relative-motion model of the docking scenario (`src/models/spacecraft.js`). |

## PyTorch3D lab: rendering, re-identification, PHD density (`pytorch3d/`)

Not yet individually verified against the publisher (no ✓); details as commonly cited.

| # | Reference | Used for (code) |
|---|---|---|
| P1 | N. Ravi, J. Reizenstein, D. Novotny, T. Gordon, W.-Y. Lo, J. Johnson, G. Gkioxari. **Accelerating 3D deep learning with PyTorch3D.** arXiv:2007.08501, 2020. | Mesh rasterisation, depth and per-pixel face ids for the camera views (`lunar3d/render.py`). |
| P2 | X. Ma, V. Hegde, L. Yolyan. **3D Deep Learning with Python.** Packt, 2022 (code: the user's fork of PacktPublishing/3D-Deep-Learning-with-Python). | Rendering pipeline pattern, ch. 2 (`render.py`). |
| P3 | N. Wojke, A. Bewley, D. Paulus. **Simple online and realtime tracking with a deep association metric (DeepSORT).** ICIP 2017. | Appearance-assisted association and re-identification of lost tracks (`tracker.py`, `reid`). Our signature is a colour histogram, not a learned embedding. |
| P4 | R. Mahler. **Multitarget Bayes filtering via first-order multitarget moments.** IEEE Trans. Aerospace and Electronic Systems 39(4), 2003. | The PHD (intensity) filter (`phd.py`). |
| P5 | B.-N. Vo, S. Singh, A. Doucet. **Sequential Monte Carlo methods for multitarget filtering with random finite sets.** IEEE Trans. Aerospace and Electronic Systems 41(4), 2005. | Particle implementation of the PHD filter (`phd.py`). |
| P6 | B. Ristic, D. Clark, B.-N. Vo, B.-T. Vo. **Adaptive target birth intensity for PHD and CPHD filters.** IEEE Trans. Aerospace and Electronic Systems 48(2), 2012. | Measurement-driven birth, so no phantom mass accumulates behind rocks (`phd.py`). |

## How this project relates to the literature (honest positioning)

* **What is standard:** CV-EKF tracks, χ² gating, Hungarian assignment (R5, R8, R9); Bernoulli/IPDA existence (R1, R2); CLEAR-MOT, IDF1, GOSPA scoring (E1–E3).
* **What this project combines:** a state-dependent `P_D` computed from 3-D line of sight to a known boulder map *and* from cast shadows under a polar sun (R3, R4, G2). It is used in three places at once: the existence update, the assignment cost (so hidden tracks are not hijacked) and the negative-information moment update. The ablation isolates the effect of each step.
* **What is not claimed:** novelty over the cited methods, or performance on real imagery. The detector is a sensor model, not a neural network, and walkers are simulated. Validation on MOT17/KITTI with HOTA (E4, E6) is the next scientific step.
