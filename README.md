# Vibrational Spectroscopy Research: A Deep Learning Approach

This repository documents PhD research conducted at LUT University (Computational Physics) on applying deep learning to automate spectroscopic signal processing — specifically, extracting the pure Raman signal from Coherent anti-Stokes Raman Scattering (CARS) spectra.

The core challenge: CARS spectra contain a strong non-resonant background (NRB) that distorts the underlying Raman signal. Traditional retrieval methods require manually tuned parameters and domain expertise. This research replaces that manual process with trained neural networks — CNN, Bi-LSTM, LSTM, and autoencoder architectures — that generalise across experimental conditions.

**Key results across the project series:**
- Fine-tuned CNN achieving **86% prediction accuracy** on semi-synthetic data, with superior performance over the baseline model on experimental spectra
- Hyperparameter-optimised SpecNet reaching **~96% accuracy**, with mean-squared error **~60× lower** than the unoptimised baseline
- Bi-LSTM outperforming CNN, LSTM, and VECTOR autoencoder on both synthetic and experimental spectra — the first application of Bi-LSTM to CARS spectroscopy
- Demonstrated end-to-end ownership: data synthesis, model architecture, training, optimisation, and peer-reviewed publication

**Transferable skills demonstrated:** end-to-end deep learning pipeline design, Bayesian hyperparameter optimisation (Optuna), time-series modelling, scientific data engineering, model comparison and benchmarking, Python/TensorFlow/Keras.

---

## Projects

### Project 1 — Fine-tuning a Pre-trained CNN for Raman Signal Retrieval
`Folder: Finetuned_SpecNet`

Applied transfer learning to adapt a pre-trained CNN (SpecNet) to two sets of semi-synthetic CARS spectra. The fine-tuned model achieved **86% prediction accuracy** on semi-synthetic data and outperformed the original model on experimental spectra. Demonstrates the ability to leverage existing architectures and adapt them efficiently to new datasets — a pattern directly applicable to modern ML engineering workflows.

---

### Project 2 — Hyperparameter Optimisation of SpecNet
`Folder: Hyperparameter_Optimization_of_CNN`

The baseline SpecNet model struggled to accurately identify all spectral peaks and their intensities. Through systematic hyperparameter optimisation, a refined architecture was developed that significantly improved peak detection and intensity matching on both synthetic and experimental datasets. The optimised model reached **~96% accuracy** with mean-squared error approximately **60× lower** than the unoptimised baseline. Demonstrates proficiency in structured model tuning — the same discipline applied in production ML systems.

---

### Project 3 — Four Deep Learning Architectures Compared
`Folder: Four_DL_models_comparison`

Introduced **bidirectional LSTM (Bi-LSTM)** to CARS spectroscopy for the first time and benchmarked it against three architectures: LSTM, CNN, and a very deep convolutional autoencoder (VECTOR). All models were trained on data synthesised with three NRB types. The Bi-LSTM consistently outperformed the other architectures in Raman signal extraction from both synthetic and experimental spectra — establishing it as the strongest model in this domain. Demonstrates the ability to design and execute rigorous multi-model comparisons.

---

### Project 4 — Impact of Non-Resonant Background on CNN Performance
`Folder: NRBs_on_extraction_of_Raman_signals_from_CARS_spectra`

Investigated how the choice of synthetic non-resonant background (NRB) during training data generation affects a CNN's ability to generalise to experimental spectra. Three NRB types were tested: product of two sigmoids, single sigmoid, and fourth-order polynomial. The polynomial-trained model demonstrated superior line-shape extraction and closest agreement with ground truth. Highlights how data engineering decisions upstream of model training critically determine downstream performance — a principle that applies equally to feature engineering in any data pipeline.

---

## Publications

All peer-reviewed publications from this research programme are indexed on Google Scholar:

**[View full publication list →](https://scholar.google.com/citations?user=GcWhnFcAAAAJ&hl=en)**

6 publications in optics, vibrational spectroscopy, and computational methods.

---

## Technical Stack

| Category | Tools & frameworks |
|---|---|
| Deep learning | TensorFlow, Keras |
| Architectures | CNN, Bi-LSTM, LSTM, convolutional autoencoder (VECTOR) |
| Optimisation | Bayesian hyperparameter search (Optuna) |
| Data | Synthetic spectral data generation, experimental CARS spectra |
| Language | Python |

---

## Connect

- **Google Scholar:** [Ali Saghi](https://scholar.google.com/citations?user=GcWhnFcAAAAJ&hl=en)
- **LinkedIn:** [ali-saghi-fabric](https://www.linkedin.com/in/ali-saghi-fabric/)
- **Blog:** [lotusoftware.hashnode.dev](https://lotusoftware.hashnode.dev) — current work in Microsoft Fabric, Power BI, and agentic analytics engineering
- **Email:** ali.saghi.2015@gmail.com
