# Credits

## Dataset

**Sen1Floods11** (v1.1), hand-labelled subset (`S2Hand`, `LabelHand`), read anonymously from the public
Google Cloud Storage bucket `gs://sen1floods11`.

> Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: A georeferenced dataset to
> train and test deep learning flood algorithms for Sentinel-1.* Proceedings of the IEEE/CVF Conference on
> Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 210-211.
> Repository: https://github.com/cloudtostreet/Sen1Floods11

License: the Sen1Floods11 repository states that the dataset is released under the Creative Commons
Attribution 4.0 International license (CC BY 4.0). Sentinel-2 imagery inside the chips is Copernicus data
(contains modified Copernicus Sentinel data). Check the repository for the current terms before
redistributing the chips; FloodWing only downloads a small subset at run time and does not redistribute it.

Sen1Floods11 contains **no survivor labels**. The survivor-likelihood target in FloodWing is a hand-defined
proxy computed from the flood mask and Sentinel-2 indices (see `src/floodwing/data.py::proxy_target`).

When the bucket cannot be reached, FloodWing generates synthetic chips procedurally
(`src/floodwing/data.py::generate_synthetic_chip`); no third-party data is involved in that path.

## Software

| Package | Use | License |
| --- | --- | --- |
| Qiskit, Qiskit Aer | circuits, statevector simulation, noise models | Apache-2.0 |
| qiskit-algorithms | `QAOA`, optimizers (COBYLA, SPSA, Nelder-Mead) | Apache-2.0 |
| qiskit-optimization | `QuadraticProgram`, `QuadraticProgramToQubo`, `MinimumEigenOptimizer` | Apache-2.0 |
| PyTorch | heatmap CNN | BSD-3-Clause |
| NumPy, SciPy, pandas, scikit-learn | numerics, statistics (Wilcoxon), tables | BSD-3-Clause |
| matplotlib, seaborn, Pillow | figures, GIF | PSF-style / BSD / HPND |
| rasterio | GeoTIFF reading | BSD-3-Clause |
| gcsfs | anonymous GCS access | BSD-3-Clause |
| pylatexenc | Qiskit matplotlib circuit drawer dependency | MIT |

## Methods and ideas (implemented from the papers, no code copied)

* QAOA: Farhi, Goldstone, Gutmann (2014), *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028.
* Linear-ramp (Trotterised annealing) parameter initialisation: Sack and Serbyn (2021), *Quantum annealing
  initialization of the quantum approximate optimization algorithm*, Quantum 5, 491.
* XY / ring mixers and Dicke initial states for Hamming-weight constraints: Hadfield et al. (2019), *From the
  Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz*, Algorithms 12(2), 34;
  Wang, Rubin, Dominy, Rieffel (2020), *XY mixers: analytical and numerical results for QAOA*, PRA 101, 012320.
* Warm-starting variational parameters across related instances follows the general parameter-transfer idea in
  the QAOA literature (e.g. Brandao et al. 2018, arXiv:1812.04170).

## Reused code

None. All code in `src/`, `tests/` and `tools/` was written for this project. Qiskit API usage follows the
official documentation of the pinned versions.
