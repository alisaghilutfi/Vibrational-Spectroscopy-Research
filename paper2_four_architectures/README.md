# Paper 2 — Four Deep Learning Architectures Compared

[![DOI](https://img.shields.io/badge/DOI-10.1039%2FD3CP01618H-blue)](https://doi.org/10.1039/D3CP01618H) [![PCCP](https://img.shields.io/badge/Phys._Chem._Chem._Phys.-2023-brightgreen)](https://pubs.rsc.org/en/Content/ArticleLanding/2023/CP/D3CP01618H)

> First application of Bi-LSTM to CARS spectroscopy. Benchmarks four architectures —
> CNN (SpecNet), LSTM, VECTOR (deep convolutional autoencoder), and Bi-LSTM — on
> both synthetic and experimental spectra. Bi-LSTM outperforms all others on every metric.

**Authors:** Junjuri R., Saghi A., Lensu L., Vartiainen E.M.

---

## Key finding

Bi-LSTM outperforms CNN, LSTM, and VECTOR on all metrics — synthetic and experimental.
All models were trained on data synthesised with three NRB types (polynomial, one-sigmoid,
two-sigmoid product).

---

## Contents

| File | Description |
|---|---|
| `bi-LSTM_train.py` | Train the Bi-LSTM model |
| `LSTM_train.py` | Train the LSTM model |
| `Bi_LSTM_model_weights.h5` | Trained Bi-LSTM weights |
| `LSTM_model_weights.h5` | Trained LSTM weights |

**CNN weights** are in `../paper1_NRB_effect_on_CNN/Polynomial_NRB_model_weights.h5` (polynomial NRB, best variant).
**VECTOR weights** are available on request — see Data section below.

**Test data** (`x_test_300_merge_spectra3.npy`, `y_test_300_merge_spectra3.npy`) lives in
`../paper1_NRB_effect_on_CNN/` and is shared across Papers 1 and 2.

---

## Data

Synthetic test spectra: `../paper1_NRB_effect_on_CNN/`.
Experimental CARS spectra and VECTOR model weights are available on request —
contact [Erik M. Vartiainen](https://research.lut.fi/converis/portal/detail/Person/56843) (LUT University).

---

## Model architectures

**Bi-LSTM:** 3 × Bidirectional LSTM (30 units, `tanh`, `return_sequences=True`) →
TimeDistributed Dense(1, `tanh`). Input: `(640, 1)`. Output: `(640, 1)`.

**LSTM:** 1 × LSTM (30 units, `tanh`, `return_sequences=True`) → Dense(1).
Input: `(640, 1)`. Output: `(640, 1)`.

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
@article{junjuri2023evaluating,
  title   = {Evaluating different deep learning models for efficient extraction
             of Raman signals from CARS spectra},
  author  = {Junjuri, Rajendhar and Saghi, Ali and Lensu, Lasse and Vartiainen, Erik M.},
  journal = {Physical Chemistry Chemical Physics},
  year    = {2023},
  doi     = {10.1039/D3CP01618H}
}
```
