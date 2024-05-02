# Spectral Processing Tool

The Spectral Processing Tool is a sophisticated Python-based application designed for the in-depth analysis and processing of spectroscopy data. Leveraging a Tkinter GUI, this tool offers an integrated environment for loading, preprocessing, analyzing, and modeling spectral data.


## Features

### Data Loading
- **Support for Multiple Formats**: The tool can load data from various file formats including plain text (.txt), comma-separated values (.csv), and Excel spreadsheets (.xlsx), accommodating a wide range of data sources.
- **Graphical User Interface**: Built on Tkinter, the tool provides a graphical interface for easy navigation and operation, allowing users to interactively select and load files or entire directories of data.

### Preprocessing
- **Smoothing Techniques**: Offers several methods to smooth noisy data, including Savitzky-Golay filters, wavelet transformations for denoising, and Finite Impulse Response (FIR) filters.
- **Normalization Options**: Users can normalize spectra based on the total area under the curve or the height of the peak intensity, standardizing data for better comparison and analysis.
- **Baseline Removal**: Implements multiple baseline correction methods such as Gaussian-Lorentzian fitting, Modified Polynomial Fitting, and adaptive iteratively reweighted penalized least squares (airPLS) to correct for baseline drifts in spectral data.
- **Despiking and Cropping**: Provides functionality to identify and remove spikes from data and to crop spectra to focus analyses on specific regions of interest.
- **Interpolation**: Facilitates the resampling of spectra to standardize the number of data points across different measurements.

### Statistical Analysis
- **Comprehensive Metrics**: Compute statistical metrics such as mean, standard deviation, and correlation across multiple spectra.
- **Visualization Tools**: Integrated plotting tools allow for immediate visualization of spectra and statistical outcomes, facilitating a quick assessment of data characteristics and analysis results.

### Machine Learning
- **Classification Algorithms**: Includes support for Support Vector Machines (SVM) and Linear Discriminant Analysis (LDA) to classify spectral data based on their features.
- **Model Training and Evaluation**: Supports training models with cross-validation to ensure robustness and generalizability and provides evaluation metrics such as accuracy, precision, recall, and F1-scores to assess model performance.

## Code Structure

### Main Application (`spectral_processing_tool.py`)
- **SpectralApp**: Root class for the application, initializing the main window and tabs.
- **LoadDataTab, PreprocessingTab, StatisticsTab, MachineLearningTab**: Each tab provides a specific set of functionalities and user interface elements for different aspects of spectral processing.

### Data Handling (`database.py`)
- **SERSSpectra**: A class representing a single spectrum with methods for preprocessing.
- **SERSSpectraBatch**: Manages a batch of spectra and performs batch operations.
- **SERSSpectraDataset**: Organizes multiple batches and facilitates data loading and preprocessing for machine learning.

### Utilities (`utils.py`)
- **airPLS**: Function for baseline removal using adaptive iteratively reweighted penalized least squares.
- **mixed_gauss_lorentz**: Implements a combined Gaussian-Lorentzian function for fitting spectral baselines.
- **align_spectra, calculate_first_derivative**: Functions for aligning spectra and calculating derivatives, which are essential for certain preprocessing and analysis tasks.

