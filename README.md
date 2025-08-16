
# Vibrational Spectroscopy Research: A Deep Learning Approach
This repository showcases a series of deep learning projects conducted during my PhD in Computational Physics, focused on advancing the analysis of vibrational spectroscopy data, specifically from Coherent anti-Stokes Raman scattering (CARS) microscopy. The work is motivated by the challenge of extracting the pure Raman signal from the CARS spectrum, which is often distorted by a non-resonant background (NRB).

The projects demonstrate my expertise in developing, optimizing, and applying deep learning models—including CNNs and RNNs—to solve complex signal processing and data analysis problems. The methodologies explored offer automated, data-driven solutions that perform comparably to or better than traditional numerical methods, a key step toward automating complex scientific data pipelines.

### Project 1: Fine-tuning a Pre-trained CNN for Raman Signal Retrieval
This project focused on transfer learning in a computational physics context. I fine-tuned a pre-trained convolutional neural network (CNN) on two sets of semi-synthetic spectra to improve its ability to retrieve the Raman signal from CARS spectra. The fine-tuned model achieved an 86% prediction accuracy on semi-synthetic data and demonstrated superior performance on experimental spectra compared to the original model. This study highlights my ability to leverage existing models and adapt them for new challenges.

### Project 2: Hyperparameter Optimization for SpecNet
This project addresses a critical challenge in model performance: hyperparameter optimization. The goal was to overcome limitations of a known CNN model, SpecNet, which struggled to accurately identify all spectral peaks and their intensities. By optimizing the model's hyperparameters, I developed a refined version that significantly improved peak detection and intensity matching on both synthetic and experimental datasets. This demonstrates my proficiency in fine-tuning models to solve specific, real-world data issues.

### Project 3: Exploring Bi-LSTM for Spectral Signal Extraction
In this study, we introduced bidirectional long short-term memory (Bi-LSTM), a powerful recurrent neural network, to the field of CARS spectroscopy for the first time. We compared the Bi-LSTM model's performance against three other deep learning architectures: LSTM, CNN, and a very deep convolutional autoencoder (VECTOR). All models were trained on data synthesized with three different types of NRBs. The Bi-LSTM model consistently outperformed the other models in extracting the Raman signal from both synthetic and experimental spectra, showcasing my ability to innovate and apply novel deep learning architectures to complex time-series data.

### Project 4: Impact of Non-Resonant Backgrounds on CNN Performance
This study investigated how the choice of non-resonant background (NRB) during synthetic data generation affects a CNN model's ability to retrieve the Raman signal. We trained separate CNN models using three distinct NRB types: product of two sigmoids, a single sigmoid, and a fourth-order polynomial function. The results showed that the model trained with the Polynomial NRB demonstrated a superior ability to extract the imaginary part of the signal, and the extracted line shapes were in excellent agreement with the ground truth. This project highlights my data engineering skills, specifically my understanding of how data preprocessing and synthesis can critically impact model performance.

