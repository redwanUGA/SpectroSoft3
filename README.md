# Spectral Processing Tool

The Spectral Processing Tool is a sophisticated Python-based application designed for the in-depth analysis and processing of spectroscopy data. Leveraging a Tkinter GUI, this tool offers an integrated environment for loading, preprocessing, analyzing, and modeling spectral data.

## Key Features

### Data Loading
- **File Compatibility**: Supports `.txt`, `.csv`, and `.xlsx` files, allowing users to load data from various sources.
- **Interactive GUI**: Users can select files or directories through a user-friendly graphical interface, which simplifies the process of loading and organizing spectral datasets.

### Preprocessing
- **Smoothing and Denoising**: Implements Savitzky-Golay filters, wavelet denoising, and FIR filters to smooth spectra and reduce noise.
- **Normalization**: Features normalization by area or peak to standardize spectra intensities.
- **Baseline Removal**: Includes methods like Gaussian-Lorentzian fitting (GLF), Modified Polynomial Fitting (ModPoly), and adaptive iteratively reweighted penalized least squares (airPLS).
- **Despiking and Cropping**: Offers tools to remove spikes and crop spectra to focus on regions of interest.
- **Interpolation**: Allows resampling of spectra to standardize data points across datasets.

### Statistical Analysis
- **Spectral Operations**: Calculate average, standard deviation, correlation, and other statistical measures across multiple spectra.
- **Visualization**: Built-in plotting capabilities to visualize spectra and statistical results directly within the application.

### Machine Learning
- **Model Integration**: Includes SVM and LDA for classifying spectra based on their features.
- **Training and Testing**: Facilitates training with cross-validation and testing, providing metrics like accuracy, precision, recall, and F1-scores.

## Installation

Before running the application, install the necessary Python libraries using:

```bash
pip install numpy pandas scipy matplotlib scikit-learn seaborn pywt dask
