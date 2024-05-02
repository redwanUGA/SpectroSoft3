from scipy.signal import savgol_filter, firwin, lfilter, find_peaks
import scipy.interpolate as interp1d
import pywt
from scipy.optimize import curve_fit
from utils import airPLS, mixed_gauss_lorentz
import numpy as np
import copy
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


from utils import align_spectra, calculate_first_derivative

class SERSSpectra:
    def __init__(self, raman_shifts=None, raman_intensities=None,label=None):
        self.raman_shifts = np.array(raman_shifts) if raman_shifts is not None else None
        self.raman_intensities = np.array(raman_intensities) if raman_intensities is not None else None
        self.label = label

    def apply_savgol_filter(self, window_length=11, polyorder=2):
        """Apply Savitzky-Golay filter to smooth the spectrum."""
        self.raman_intensities = savgol_filter(self.raman_intensities, window_length, polyorder)
        return self.raman_shifts, self.raman_intensities

    def denoise_by_wavelet(self, wavelet='db4', level=None):
        """Denoise using discrete wavelet transform with correct data length handling."""
        if level is None:
            level = pywt.dwt_max_level(len(self.raman_intensities), wavelet)

        coeffs = pywt.wavedec(self.raman_intensities, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(self.raman_intensities)))

        coeffs[1:] = [pywt.threshold(i, threshold, mode='soft') for i in coeffs[1:]]
        reconstructed = pywt.waverec(coeffs, wavelet)

        # Ensure the reconstructed array matches the original length
        if len(reconstructed) != len(self.raman_intensities):
            reconstructed = np.resize(reconstructed, len(self.raman_intensities))

        self.raman_intensities = reconstructed
        return self.raman_shifts, self.raman_intensities

    def normalize_by_area(self):
        """Normalize the spectrum by the area under the curve, ensuring the area is treated as absolute."""
        area = np.trapz(self.raman_intensities, self.raman_shifts)
        self.raman_intensities /= abs(area)  # Ensure the area is always positive
        return self.raman_shifts, self.raman_intensities

    def normalize_by_peak(self):
        """Normalize the spectrum by the maximum peak intensity."""
        peak_intensity = np.max(self.raman_intensities)
        self.raman_intensities /= peak_intensity
        return self.raman_shifts, self.raman_intensities

    def despiking(self, threshold=1.0):
        """Remove spikes from the spectrum based on a threshold multiplier."""
        diff = np.diff(self.raman_intensities)
        std = np.std(diff)
        peaks, _ = find_peaks(np.abs(diff), height=std * threshold)
        for peak in peaks:
            if peak + 1 < len(self.raman_intensities):
                self.raman_intensities[peak] = (self.raman_intensities[peak - 1] + self.raman_intensities[peak + 1]) / 2
        return self.raman_shifts, self.raman_intensities

    def crop(self, start, stop):
        """Crop the spectrum to a specified range of Raman shifts."""
        indices = (self.raman_shifts >= start) & (self.raman_shifts <= stop)
        self.raman_shifts = self.raman_shifts[indices]
        self.raman_intensities = self.raman_intensities[indices]
        return self.raman_shifts, self.raman_intensities

    def interpolate(self, start, stop, step):
        """Interpolate the spectrum using numpy.interp and append the interpolated data to the original arrays, then sort by Raman shifts."""
        new_shifts = np.arange(start, stop, step)
        if len(new_shifts) == 0:
            return self.raman_shifts, self.raman_intensities

        # Using numpy.interp for linear interpolation
        new_intensities = np.interp(new_shifts, self.raman_shifts, self.raman_intensities)

        # Append the new data to the original arrays
        self.raman_shifts = np.concatenate((self.raman_shifts, new_shifts))
        self.raman_intensities = np.concatenate((self.raman_intensities, new_intensities))

        # Sort arrays by Raman shifts
        sorted_indices = np.argsort(self.raman_shifts)
        self.raman_shifts = self.raman_shifts[sorted_indices]
        self.raman_intensities = self.raman_intensities[sorted_indices]

        return self.raman_shifts, self.raman_intensities

    def remove_baseline_airPLS(self, lambda_=100, porder=1, itermax=15):
        """Remove baseline using airPLS algorithm."""
        baseline = airPLS(self.raman_intensities, lambda_=lambda_, porder=porder, itermax=itermax)
        self.raman_intensities -= baseline
        return self.raman_shifts, self.raman_intensities

    def baseline_removal_by_GLF(self, params):
        """Remove baseline using Gaussian-Lorentzian fitting."""
        # Fitting to a Gaussian-Lorentzian mix model
        popt, _ = curve_fit(mixed_gauss_lorentz, self.raman_shifts, self.raman_intensities, p0=params)
        baseline = mixed_gauss_lorentz(self.raman_shifts, *popt)
        self.raman_intensities -= baseline
        return self.raman_shifts, self.raman_intensities

    def baseline_removal_by_ModPoly(self, polynomial_order):
        """
        Remove baseline using a polynomial fitting of specified order.

        Args:
        polynomial_order (int): The order of the polynomial used to fit the baseline.
        """
        # Fit a polynomial to the data
        p = np.polyfit(self.raman_shifts, self.raman_intensities, polynomial_order)

        # Evaluate the polynomial at the original x points
        baseline = np.polyval(p, self.raman_shifts)

        # Subtract the baseline from the original intensities
        self.raman_intensities -= baseline

        return self.raman_shifts, self.raman_intensities

    def denoise_by_fir_filter(self, numtaps=51, window_name='hamming'):
        """
        Apply an FIR filter to smooth the spectrum using a windowed sinc function.

        Parameters:
        numtaps (int): The number of taps in the filter, representing the filter order + 1.
                       Must be odd. Higher numtaps increase the smoothness and the delay.
        window_name (str): The name of the window to use. Common windows include 'hamming', 'hann', 'blackman', etc.

        Returns:
        self: For chaining methods
        """
        if numtaps % 2 == 0:  # Ensure numtaps is odd
            numtaps += 1

        # Generate filter coefficients using the specified window
        fir_coeff = firwin(numtaps, cutoff=0.1, window=window_name)
        self.raman_intensities = lfilter(fir_coeff, 1.0, self.raman_intensities)
        return self


    def clone(self):
        """Create a deep copy of this spectra object."""
        return copy.deepcopy(self)

class SERSSpectraBatch:
    def __init__(self, spectra_list, reference_spectra=None):
        self.spectra_list = spectra_list
        # Check if the list is not empty before assigning a default reference
        if spectra_list:
            self.reference_spectra = reference_spectra if reference_spectra is not None else spectra_list[0]
        else:
            self.reference_spectra = None

    def spectral_average(self):
        aligned_spectra = align_spectra(self.spectra_list)
        avg_intensity = np.mean(aligned_spectra, axis=0)
        return SERSSpectra(raman_shifts=self.reference_spectra.raman_shifts, raman_intensities=avg_intensity)

    def spectral_SD(self):
        aligned_spectra = align_spectra(self.spectra_list)
        sd_intensity = np.std(aligned_spectra, axis=0)
        return SERSSpectra(raman_shifts=self.reference_spectra.raman_shifts, raman_intensities=sd_intensity)

    def spectral_correlation(self):
        """Compute the correlation matrix across spectra for each pair of Raman shifts."""
        if not self.spectra_list:
            return np.array([])

        # Align all spectra to the same Raman shifts if necessary
        # This assumes all spectra have the same Raman shifts, if not, align them first
        intensities = np.array(
            [spectra.raman_intensities for spectra in self.spectra_list if spectra.raman_intensities is not None])

        # Check if intensities array is not empty and is 2D
        if intensities.ndim != 2:
            return np.array([])

        # Calculate the correlation matrix of Raman intensities across all spectra
        correlation_matrix = np.corrcoef(intensities)
        return correlation_matrix


    def FDAD(self):
        ref_derivative = calculate_first_derivative(self.reference_spectra.raman_shifts, self.reference_spectra.raman_intensities)
        abs_diffs = []
        for spectrum in self.spectra_list:
            spectrum_derivative = calculate_first_derivative(spectrum.raman_shifts, spectrum.raman_intensities)
            abs_diff = np.abs(spectrum_derivative - ref_derivative)
            abs_diffs.append(abs_diff)
        avg_abs_diff = np.mean(abs_diffs, axis=0)
        return SERSSpectra(raman_shifts=self.reference_spectra.raman_shifts, raman_intensities=avg_abs_diff)

    def spectral_MAE(self):
        min_mae = np.inf
        closest_spectrum = None
        for spectrum in self.spectra_list:
            abs_diff = np.abs(spectrum.raman_intensities - self.reference_spectra.raman_intensities[:len(spectrum.raman_intensities)])
            mae = np.mean(abs_diff)
            if mae < min_mae:
                min_mae = mae
                closest_spectrum = spectrum
        return closest_spectrum if closest_spectrum else SERSSpectra()

    def spectral_MSE(self):
        min_mse = np.inf
        most_similar_spectrum = None
        for spectrum in self.spectra_list:
            mse = np.mean((spectrum.raman_intensities - self.reference_spectra.raman_intensities[:len(spectrum.raman_intensities)])**2)
            if mse < min_mse:
                min_mse = mse
                most_similar_spectrum = spectrum
        return most_similar_spectrum if most_similar_spectrum else SERSSpectra()

    def inner_product(self):
        max_inner_product = -np.inf
        most_similar_spectrum = None
        for spectrum in self.spectra_list:
            dot_product = np.dot(spectrum.raman_intensities, self.reference_spectra.raman_intensities[:len(spectrum.raman_intensities)])
            if dot_product > max_inner_product:
                max_inner_product = dot_product
                most_similar_spectrum = spectrum
        return most_similar_spectrum if most_similar_spectrum else SERSSpectra()

    def clone(self):
        """Clone the entire batch of spectra."""
        cloned_spectra_list = [spectra.clone() for spectra in self.spectra_list]
        return SERSSpectraBatch(cloned_spectra_list, self.reference_spectra.clone() if self.reference_spectra else None)

    def get_batch_statistics(self):
        # Check if there are spectra to process
        if not self.spectra_list:
            return "No spectra available."

        # Initialize lists to hold all raman shifts
        all_raman_shifts = []

        # Gather all raman shifts
        for spectra in self.spectra_list:
            if spectra.raman_shifts is not None:
                all_raman_shifts.append(spectra.raman_shifts)

        # Convert list to a single numpy array for easier manipulation
        all_raman_shifts = np.concatenate(all_raman_shifts)

        # Calculate statistics
        min_raman_shift = np.min(all_raman_shifts)
        max_raman_shift = np.max(all_raman_shifts)

        # Prepare and return a summary of statistics
        stats_summary = {
            'Minimum Raman Shift': min_raman_shift,
            'Maximum Raman Shift': max_raman_shift
        }

        return stats_summary


class SERSSpectraDataset:
    def __init__(self):
        self.dataset = {}  # Dictionary to hold batches keyed by subfolder/class label

    def add_batch(self, label, batch):
        """Add a batch of spectra under a specific label/class."""
        if label in self.dataset:
            raise ValueError(f"Batch with label {label} already exists.")
        self.dataset[label] = batch

    def get_batch(self, label):
        """Retrieve a batch by its label."""
        return self.dataset.get(label)

    def get_labels(self):
        """Retrieve all labels/classes in the dataset."""
        return list(self.dataset.keys())

    def __str__(self):
        return '\n'.join([f"Label: {label}, Spectra Count: {len(batch.spectra_list)}" for label, batch in self.dataset.items()])

    def load_data_from_folder(self, folder_path):
        """
        Load datasets from folders where each subfolder corresponds to a class label.
        Assumes subfolder names are class labels.
        """
        for subdir, dirs, files in os.walk(folder_path):
            label = os.path.basename(subdir)
            spectra_list = []
            for file in files:
                if file.endswith(('.txt', '.csv', '.xlsx')):
                    file_path = os.path.join(subdir, file)
                    spectra = self.load_spectra(file_path)
                    if spectra:
                        spectra.label = label
                        spectra_list.append(spectra)
            if spectra_list:
                self.spectra_batches[label] = spectra_list

    def load_spectra(self, file_path):
        """
        Load a single spectra file and return a SERSSpectra object.
        Supports TXT, CSV, and XLSX formats.
        """
        if file_path.endswith('.txt'):
            data = np.loadtxt(file_path, delimiter='\t')
        elif file_path.endswith('.csv'):
            data = np.loadtxt(file_path, delimiter=',')
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path, header=None).values
        else:
            return None

        raman_shifts, raman_intensities = data[:, 0], data[:, 1]
        return SERSSpectra(raman_shifts, raman_intensities)

    def get_features_and_labels(self):
        """
        Extract features and labels from the loaded data, suitable for machine learning models.
        Returns features as numpy arrays and labels as encoded integers.
        """
        features = []
        labels = []
        for label, spectra_batch in self.spectra_batches.items():
            for spectra in spectra_batch:
                features.append(spectra.raman_intensities)
                labels.append(label)

        # Encode labels if they are not numeric
        labels = self.label_encoder.fit_transform(labels)
        return np.array(features), np.array(labels)

    def get_label_mapping(self):
        """
        Returns the mapping of encoded labels to original labels.
        """
        return dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))

    def prepare_features_and_labels(self):
        """Prepare feature matrix X and target vector y from the dataset, standardizing features."""
        features = []
        labels = []
        for label, batch in self.dataset.items():
            for spectra in batch.spectra_list:
                features.append(spectra.raman_intensities)  # Using intensities as features
                labels.append(label)  # Using subfolder names as labels

        # Standardize features
        scaler = StandardScaler()
        features = scaler.fit_transform(np.array(features))

        return features, np.array(labels)

    def export_classification_results(self, predictions, actual, accuracy, folder_path):
        """Export classification results and the confusion matrix to an Excel file."""
        if not folder_path:
            return  # User cancelled the dialog or provided an invalid path

        with pd.ExcelWriter(f"{folder_path}/classification_results.xlsx") as writer:
            # Export classification metrics
            data = {
                'Predictions': predictions,
                'Actual': actual,
                'Accuracy': [accuracy] * len(actual),
                'F1 Score': [f1_score(actual, predictions, average='macro')] * len(actual),
                'Precision': [precision_score(actual, predictions, average='macro')] * len(actual),
                'Recall': [recall_score(actual, predictions, average='macro')] * len(actual)
            }
            df_metrics = pd.DataFrame(data)
            df_metrics.to_excel(writer, sheet_name='Metrics', index=False)

            # Export the confusion matrix
            conf_matrix = confusion_matrix(actual, predictions)
            df_conf_matrix = pd.DataFrame(conf_matrix, index=[f"Actual {label}" for label in set(actual)],
                                          columns=[f"Predicted {label}" for label in set(actual)])
            df_conf_matrix.to_excel(writer, sheet_name='Confusion Matrix', index=True)