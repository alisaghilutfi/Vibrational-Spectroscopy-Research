# Hyperparameter Optimization of CNN for Retrieving Raman from CARS
 Coherent anti-Stokes Raman scattering (CARS) is a third-order nonlinear optical
 process used in spectroscopy to analyze molecular structures. One significant drawback of
 this approach is its non-resonant background contribution, which distorts spectral line shapes
 and consequently diminishes the accuracy of chemical information. A state-of-the-art solution
 for automatically extracting the Raman signals from CARS spectra is the convolutional neural
 network (CNN) model. In this research, we studied the use of hyperparameter optimization
 of SpecNet, a CNN model proposed in the literature, to improve the extraction of the Raman
 signal from CARS spectra. The original SpecNet has two major problems: first, the model is
 incapable of recovering spectral peaks near the edges of the spectral range, and second, it cannot
 match the intensity of the peaks throughout the spectrum. In this work, these two problems were
 successfully mitigated by optimization of the hyperparameters of the SpecNet model.
