# Paper 1 — Effect of NRB Type on CNN Generalisation

[![DOI](https://img.shields.io/badge/DOI-10.1039%2Fd2ra03983d-blue)](https://doi.org/10.1039/d2ra03983d) [![RSC Advances](https://img.shields.io/badge/RSC_Advances-2022-brightgreen)](https://pubs.rsc.org/en/content/articlehtml/2022/ra/d2ra03983d)

> Investigates how the choice of synthetic NRB shape during training data generation
> affects a CNN's ability to generalise to experimental CARS spectra. Establishes that
> upstream data engineering decisions critically determine downstream model performance.

**Authors:** Junjuri R., Saghi A., Lensu L., Vartiainen E.M.

---

## Key finding

Polynomial NRB training yields the best generalisation to experimental spectra.
Three NRB shapes were compared — product of two sigmoids, one sigmoid, and
fourth-order polynomial — across training and experimental evaluation sets.

---

## Contents

| File | Description |
|---|---|
| `RSS_Advances_CNN_to_train_with_different_NRBs.py` | Train SpecNet CNN with any of the three NRB types |
| `RSS_Advances_CNN_prediction_on_test_data.py` | Run predictions on the 300-spectrum test set |
| `Specnet_weights.h5` | Trained weights — two-sigmoid NRB |
| `One_sigmoid_NRB_model_weights.h5` | Trained weights — one-sigmoid NRB |
| `Polynomial_NRB_model_weights.h5` | Trained weights — polynomial NRB (best) |
| `x_test_300_merge_spectra3.npy` | 300 synthetic test CARS spectra (input) |
| `y_test_300_merge_spectra3.npy` | 300 synthetic test Raman signals (ground truth) |

**Test set structure:** 100 spectra per NRB type (two-sigmoid: 1–100, one-sigmoid: 101–200, polynomial: 201–300).

---

## Data

Synthetic test spectra are included in this folder.
Experimental CARS spectra are available on request —
contact [Erik M. Vartiainen](https://research.lut.fi/converis/portal/detail/Person/56843) (LUT University).

---

## Model architecture

SpecNet CNN — adapted from [Valensic et al.](https://github.com/Valensicv/SpecNet).
Input: `(640, 1)` CARS spectrum. Output: `(640,)` Raman signal.
Five Conv1D layers → two Dense layers → Flatten → Dropout → Dense output.

---

## Requirements

```bash
pip install -r ../requirements.txt   # TensorFlow 2.7, NumPy, Matplotlib
```

Tested in Spyder (Python 3). Use `tensorflow_env_higher` (TF 2.11 + NumPy 1.23)
if TF 2.7 raises a NumPy compatibility error.

---

## Citation

```bibtex
@article{junjuri2022effect,
  title   = {Effect of non-resonant background on the extraction of Raman signals
             from CARS spectra using deep neural networks},
  author  = {Junjuri, Rajendhar and Saghi, Ali and Lensu, Lasse and Vartiainen, Erik M.},
  journal = {RSC Advances},
  year    = {2022},
  doi     = {10.1039/d2ra03983d}
}
```
