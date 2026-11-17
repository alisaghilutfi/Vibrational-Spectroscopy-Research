# Paper 4 — Hyperparameter Optimisation of SpecNet CNN

[![DOI](https://img.shields.io/badge/DOI-10.1364%2FOPTCON.527778-blue)](https://doi.org/10.1364/OPTCON.527778) [![Optics Continuum](https://img.shields.io/badge/Optics_Continuum-2024-brightgreen)](https://opg.optica.org/optcon/fulltext.cfm?uri=optcon-3-8-1461)

> Applies Bayesian hyperparameter search (Optuna) to SpecNet, resolving two systematic
> failure modes of the baseline model: inability to recover peaks near spectral edges, and
> poor peak intensity matching across the spectrum. Achieves ~96% accuracy with MSE ~60x
> lower than the unoptimised baseline.

**Authors:** Junjuri R., Saghi A., Lensu L., Vartiainen E.M.

---

## Key finding

Hyperparameter-optimised SpecNet reaches ~96% accuracy with MSE ~60x lower than the
baseline. Edge artefacts and intensity mismatch — the two principal failure modes of the
original SpecNet — are both resolved by the optimised architecture.

---

## Contents

| File | Description |
|---|---|
| `Hyperoptimized_SpecNet.ipynb` | Full optimisation pipeline: Optuna search, best-model training, evaluation |
| `On the relevance of hyperparameter optimization of convolutional neural networks for retrieving spectral signals.pdf` | Published paper PDF |

**Test data** (`x_test_300_merge_spectra3.npy`, `y_test_300_merge_spectra3.npy`) lives in
`../paper1_NRB_effect_on_CNN/` and is shared across the paper series.

---

## Data

Synthetic test spectra: `../paper1_NRB_effect_on_CNN/`.
Experimental CARS spectra are available on request —
contact [Erik M. Vartiainen](https://research.lut.fi/converis/portal/detail/Person/56843) (LUT University).

---

## Requirements

```bash
pip install -r ../requirements.txt   # TensorFlow 2.7, NumPy, Matplotlib, Optuna
```

Tested in Spyder (Python 3). Use `tensorflow_env_higher` (TF 2.11 + NumPy 1.23)
if TF 2.7 raises a NumPy compatibility error.

---

## Citation

```bibtex
@article{junjuri2024relevance,
  title   = {On the relevance of hyperparameter optimization of convolutional neural networks
             for retrieving spectral signals},
  author  = {Junjuri, Rajendhar and Saghi, Ali and Lensu, Lasse and Vartiainen, Erik M.},
  journal = {Optics Continuum},
  year    = {2024},
  doi     = {10.1364/OPTCON.527778}
}
```
