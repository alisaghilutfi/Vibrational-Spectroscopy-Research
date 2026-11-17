# Paper 3 — Fine-tuning a Pre-trained CNN (Finetuned SpecNet)

[![DOI](https://img.shields.io/badge/DOI-10.1364%2FOPTCON.472573-blue)](https://doi.org/10.1364/OPTCON.472573) [![Optics Continuum](https://img.shields.io/badge/Optics_Continuum-2022-brightgreen)](https://opg.optica.org/optcon/fulltext.cfm?uri=optcon-1-11-2360)

> Applies transfer learning to adapt the pre-trained SpecNet CNN to semi-synthetic CARS
> spectra. Achieves 86% prediction accuracy on semi-synthetic data and outperforms the
> original SpecNet on four experimental CARS spectra. Also characterises model sensitivity
> to noise — performance degrades non-linearly with increasing noise level.

**Authors:** Saghi A., Junjuri R., Lensu L., Vartiainen E.M.

---

## Key finding

86% accuracy on semi-synthetic data; fine-tuned SpecNet outperforms the original
pre-trained SpecNet on all four experimental CARS spectra. Noise sensitivity is
non-linear — small noise increases cause disproportionate performance drops.

---

## Contents

| File | Description |
|---|---|
| `Hyperoptimized_SpecNet.ipynb` | Fine-tuning notebook: data preparation, training, evaluation |

**Pre-trained SpecNet weights** (starting point for fine-tuning) are in
`../paper1_NRB_effect_on_CNN/Polynomial_NRB_model_weights.h5`.

---

## Data

Semi-synthetic CARS spectra used for fine-tuning are described in the paper.
Experimental CARS spectra are available on request —
contact [Erik M. Vartiainen](https://research.lut.fi/converis/portal/detail/Person/56843) (LUT University).

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
@article{saghi2022semi,
  title   = {Semi-synthetic data generation to fine-tune a convolutional neural network
             for retrieving Raman signals from CARS spectra},
  author  = {Saghi, Ali and Junjuri, Rajendhar and Lensu, Lasse and Vartiainen, Erik M.},
  journal = {Optics Continuum},
  year    = {2022},
  doi     = {10.1364/OPTCON.472573}
}
```
