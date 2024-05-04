import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import savgol_filter, firwin, lfilter
import pywt
from scipy.optimize import curve_fit
from database import SERSSpectra, SERSSpectraBatch, SERSSpectraDataset  # Assuming this class is defined as needed
from utils import airPLS, mixed_gauss_lorentz  # Assuming these functions are available
import os
from functools import partial

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"  # MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # For VecLib
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # For NumExpr


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import dask.dataframe as dd


# Main application window
class SpectralApp(ttk.Window):
    def __init__(self):
        super().__init__(themename='flatly')

        style = ttk.Style()
        style.configure('TNotebook.Tab', font=('Helvetica', 12, 'bold'), padding=[20, 10], background='#f0f0f0')
        style.map('TNotebook.Tab', background=[('selected', '#007bff'), ('active', '#0056b3')],
                  foreground=[('selected', '#ffffff'), ('active', '#ffffff')])

        style.configure('TCombobox', font=('Helvetica', 12), padding=10)
        style.map('TCombobox', fieldbackground=[('readonly', '#ffffff')],
                  selectbackground=[('readonly', '#ffffff')],
                  selectforeground=[('readonly', 'black')],
                  background=[('readonly', '#007bff')],
                  foreground=[('readonly', '#ffffff')],
                  arrowcolor=[('readonly', '#ffffff')])

        # Initialize datasets to None or as appropriate default values
        self.training_dataset = None
        self.testing_dataset = None

        self.init_ui()

    def init_ui(self):
        self.title('Spectral Processing Tool')
        self.geometry('1200x800')  # Adjusted for better visibility

        # Initialize Tabs
        self.notebook = ttk.Notebook(self)
        self.load_data_tab = LoadDataTab(self.notebook, self)  # Pass 'self' here
        self.preprocessing_tab = PreprocessingTab(self.notebook, self)  # Pass 'self' here
        self.statistics_tab = StatisticsTab(self.notebook, self)  # Pass 'self' here
        self.machine_learning_tab = MachineLearningTab(self.notebook, self)  # Pass 'self' here

        # Adding tabs to the notebook
        self.notebook.add(self.load_data_tab, text='Load Data')
        self.notebook.add(self.preprocessing_tab, text='Preprocessing')
        self.notebook.add(self.statistics_tab, text='Statistics')
        self.notebook.add(self.machine_learning_tab, text='Machine Learning')

        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)


class LoadDataTab(ttk.Frame):
    def __init__(self, notebook, app):
        super().__init__(notebook)
        self.app = app

        self.load_button = ttk.Button(self, text='Load Data', bootstyle=PRIMARY, command=self.load_data)
        self.load_button.pack(pady=20)

        # Dropdown menu for dataset type selection
        self.dataset_type_var = tk.StringVar()
        self.dataset_type_combo = ttk.Combobox(self, textvariable=self.dataset_type_var,
                                               values=['Load Training Data Folder', 'Load Testing Data Folder'],
                                               state="readonly")
        self.dataset_type_combo.pack(pady=10)

        # Load Dataset button
        self.load_dataset_button = ttk.Button(self, text='Load DataSet', bootstyle=SUCCESS, command=self.load_dataset)
        self.load_dataset_button.pack()


    def load_data(self):
        files = filedialog.askopenfilenames(title="Select one or more files to load",
                                            filetypes=[("All supported types", "*.txt *.csv *.xlsx"),
                                                           ("Text files", "*.txt"),
                                                           ("CSV files", "*.csv"),
                                                           ("Excel files", "*.xlsx")])
        if files:
            self.process_files(files)
        else:
            messagebox.showinfo("Info", "No files were selected.")

    def process_files(self, files):
        spectra_list = []
        for file_path in files:
            try:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"The file {file_path} does not exist.")

                # Determine the file type and set appropriate read options
                if file_path.endswith('.csv'):
                    delimiter = ','
                elif file_path.endswith('.txt'):
                    delimiter = '\t'
                else:
                    raise ValueError("Unsupported file format.")

                # Use Dask to read the file, skipping the first 10 rows
                ddf = dd.read_csv(file_path, assume_missing=True, skiprows=10, delimiter=delimiter, header=None)

                # Convert all columns to numeric, replacing errors with NaN
                ddf = ddf.map_partitions(lambda df: df.apply(pd.to_numeric, errors='coerce'))

                # Compute results to convert from Dask DataFrame to Pandas DataFrame
                data = ddf.compute()

                # Drop rows with any NaN values
                data.dropna(inplace=True)

                # Ensure the data has at least two columns (Raman shifts and at least one intensity column)
                if data.shape[1] < 2:
                    raise ValueError("File must have at least two columns for Raman shifts and intensities.")

                raman_shifts = data.iloc[:, 0]
                # Create a SERSSpectra object for each spectra column
                for column in range(1, data.shape[1]):
                    raman_intensities = data.iloc[:, column]
                    spectra_list.append(SERSSpectra(raman_shifts, raman_intensities))

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

        if spectra_list:
            self.app.batch = SERSSpectraBatch(spectra_list)
            if hasattr(self.app, 'preprocessing_tab'):
                self.app.preprocessing_tab.load_spectra(spectra_list)
            messagebox.showinfo("Success", "Data loaded successfully and processed into spectra.")
        else:
            messagebox.showinfo("Info", "No valid spectra data was loaded.")

    def load_dataset(self):
        folder_path = filedialog.askdirectory(title="Select Dataset Folder")
        if not folder_path:
            return  # User cancelled the operation

        try:
            subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
            if len(subfolders) < 2:
                raise ValueError("The folder must contain at least two subfolders with data.")

            dataset = SERSSpectraDataset()

            for subfolder in subfolders:
                files = [f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]
                if not files:
                    raise ValueError(f"No data files found in the subfolder {subfolder}.")

                spectra_list = []
                for file in files:
                    file_path = os.path.join(subfolder, file)
                    data = pd.read_csv(file_path, delimiter=('\t' if file.endswith('.txt') else ','))
                    if data.shape[1] != 2:
                        raise ValueError("Files must have exactly two columns.")

                    spectra = SERSSpectra(data.iloc[:, 0].values, data.iloc[:, 1].values)
                    spectra_list.append(spectra)

                batch = SERSSpectraBatch(spectra_list)
                dataset.add_batch(os.path.basename(subfolder), batch)

            # Set and update dataset appropriately
            if self.dataset_type_var.get() == 'Load Training Data Folder':
                self.app.training_dataset = dataset
                self.app.machine_learning_tab.update_dataset_summary(self.app.machine_learning_tab.training_frame,
                                                                        'training')
            elif self.dataset_type_var.get() == 'Load Testing Data Folder':
                self.app.testing_dataset = dataset
                self.app.machine_learning_tab.update_dataset_summary(self.app.machine_learning_tab.testing_frame,
                                                                        'testing')

            messagebox.showinfo("Success", "Dataset successfully loaded.")

        except Exception as e:
            messagebox.showerror("Error", str(e))


class PreprocessingTab(ttk.Frame):
    def __init__(self, notebook, app):
        super().__init__(notebook)
        self.app = app
        self.spectra_list = []  # This will hold a list of SERSSpectra objects
        self.wavelet_names = pywt.wavelist(kind='discrete')  # Fetching discrete wavelet names

        # Define methods and sub-methods with parameters before calling init_gui
        self.method_options = {
            "Normalization": {
                "By Area": [],
                "By Peak": []
            },
            "Smoothing": {
                "SG Filter": [("Window Length", "11"), ("Polynomial Order", "2")],
                "Wavelet Filter": [("Wavelet Name", self.wavelet_names), ("Decomposition Level", "4")],
                "FIR Filter": [("Filter Order", "51"), ("Window Type", ["hamming", "hann", "blackman"])]
            },
            "Baseline Removal": {
                "GLF": [("Start 1", "100"), ("Stop 1", "200"), ("Start 2", "300"), ("Stop 2", "400")],
                "ModPoly": [("Polynomial Degree", "2")],
                "airPLS": [("Lambda", "100"), ("Porder", "1"), ("Itermax", "15")]
            },
            "Despiking": {
                "Simple Threshold": [("Threshold", "0.5")]
            },
            "Interpolation": {
                "Linear": [("Start", "100"), ("Stop", "1800"), ("Step", "1")],
                "Cubic": [("Start", "100"), ("Stop", "1800"), ("Step", "1")]
            },
            "Cropping": {
                "Manual Crop": [("Start Raman Shift", "100"), ("End Raman Shift", "1800")]
            }
        }

        self.undo_stack = []  # Stack to hold previous states
        self.preprocess_log = []
        self.init_gui()

    def init_gui(self):
        self.method_var = tk.StringVar()
        self.method_menu = ttk.Combobox(self, textvariable=self.method_var, values=list(self.method_options.keys()))
        self.method_menu.bind('<<ComboboxSelected>>', self.update_submethod_menu)
        self.method_menu.pack()

        self.submethod_var = tk.StringVar()
        self.submethod_menu = ttk.Combobox(self, textvariable=self.submethod_var)
        self.submethod_menu.bind('<<ComboboxSelected>>', self.update_parameters_inputs)
        self.submethod_menu.pack()

        self.parameters_frame = ttk.Frame(self)
        self.parameters_frame.pack()

        self.process_button = ttk.Button(self, text="Process Data", bootstyle='success', command=self.process_data)
        self.process_button.pack()

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.apply_to_batches_button = ttk.Button(self, text="Apply to Batches", bootstyle='info', command=self.apply_to_all_batches)
        self.apply_to_batches_button.pack(pady=10)

        self.log_table = ttk.Treeview(self, columns=('details',), show='headings')
        self.log_table.heading('details', text='Preprocessing Steps Applied')
        self.log_table.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.undo_button = ttk.Button(self, text="Undo Last Change", bootstyle='warning', command=self.undo_last_change)
        self.undo_button.pack()

        self.save_data_button = ttk.Button(self, text="Save Processed Data", bootstyle='primary', command=self.save_processed_data)
        self.save_data_button.pack(pady=10)

    def log_preprocessing_step(self, function_signature, parameters=None):
        if parameters is None:
            parameters = "{}"  # Use empty dictionary as placeholder
        log_entry = f"{function_signature} - {parameters}"
        self.preprocess_log.append(log_entry)
        self.log_table.insert('', 'end', values=(log_entry,))

    def update_submethod_menu(self, event=None):
        method = self.method_var.get()
        submethods = self.method_options[method]
        self.submethod_menu['values'] = list(submethods.keys())
        self.submethod_menu.current(0)
        self.update_parameters_inputs()

    def update_parameters_inputs(self, event=None):
        for widget in self.parameters_frame.winfo_children():
            widget.destroy()

        submethod = self.submethod_var.get()
        parameters = self.method_options[self.method_var.get()][submethod]
        for param, default in parameters:
            if isinstance(default, list):
                self.create_parameter_dropdown(param, default)
            else:
                self.create_parameter_input(param, default)

    def create_parameter_dropdown(self, label, options):
        frame = ttk.Frame(self.parameters_frame)
        frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(frame, text=label).pack(side=tk.LEFT)
        combo = ttk.Combobox(frame, values=options)
        combo.pack(side=tk.RIGHT, expand=True, fill='x')
        combo.set(options[0])

    def create_parameter_input(self, label, default_value):
        frame = ttk.Frame(self.parameters_frame)
        frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(frame, text=label).pack(side=tk.LEFT)
        entry = ttk.Entry(frame)
        entry.insert(0, default_value)
        entry.pack(side=tk.RIGHT, expand=True, fill='x')

    def process_data(self):
        """Save current state before processing and apply changes."""
        # Save the current state before making changes
        if self.app.batch.spectra_list:
            self.undo_stack.append(self.app.batch.clone())

        method = self.method_var.get()
        submethod = self.submethod_var.get()
        params = {child.winfo_children()[0].cget("text"): child.winfo_children()[1].get() for child in self.parameters_frame.winfo_children()}

        log_message = f"{method} - {submethod} with parameters {params}"
        try:
            # Apply preprocessing steps to each spectrum in the batch
            for spectra in self.app.batch.spectra_list:
                # Call the appropriate methods based on the selected options
                self.apply_preprocessing_method(spectra, method, submethod, params)

            # Log the operation once all spectra have been processed
            self.log_preprocessing_step(log_message)
        except Exception as e:
            messagebox.showerror("Error", str(e))

        self.update_plot()
        self.app.statistics_tab.update_file_list(self.app.batch.spectra_list)  # Optionally refresh the statistics tab if needed

    def apply_preprocessing_method(self, spectra, method, submethod, params):
        """Applies selected preprocessing methods to the given spectra."""
        if method == "Smoothing":
            if submethod == "SG Filter":
                spectra.apply_savgol_filter(int(params["Window Length"]), int(params["Polynomial Order"]))
            elif submethod == "Wavelet Filter":
                spectra.denoise_by_wavelet(params["Wavelet Name"], int(params["Decomposition Level"]))
            elif submethod == "FIR Filter":
                spectra.denoise_by_fir_filter(int(params["Filter Order"]), params["Window Type"])
        elif method == "Normalization":
            if submethod == "By Area":
                spectra.normalize_by_area()
            elif submethod == "By Peak":
                spectra.normalize_by_peak()
        elif method == "Baseline Removal":
            if submethod == "GLF":
                spectra.baseline_removal_by_GLF((float(params["Start 1"]), float(params["Stop 1"]), float(params["Start 2"]), float(params["Stop 2"])))
            elif submethod == "ModPoly":
                spectra.baseline_removal_by_ModPoly(int(params["Polynomial Degree"]))
            elif submethod == "airPLS":
                spectra.remove_baseline_airPLS(float(params["Lambda"]), int(params["Porder"]), int(params["Itermax"]))
        elif method == "Despiking":
            spectra.despiking(float(params["Threshold"]))
        elif method == "Interpolation":
            spectra.interpolate(float(params["Start"]), float(params["Stop"]), float(params["Step"]))
        elif method == "Cropping":
            spectra.crop(float(params["Start Raman Shift"]), float(params["End Raman Shift"]))

    def undo_last_change(self):
        """Undo the last change if possible, update the plot and remove last log entry."""
        if self.undo_stack:
            last_state = self.undo_stack.pop()  # Get the last saved state
            self.app.batch = last_state  # Restore the batch to its previous state

            self.update_plot()  # Update the plot to reflect the restored state

            if len(self.log_table.get_children()) > 0:
                self.log_table.delete(self.log_table.get_children()[-1])  # Remove the last entry from the log table

            messagebox.showinfo("Undo", "Last change has been undone.")
            self.app.statistics_tab.update_file_list(self.app.batch.spectra_list)  # Update any dependent views
        else:
            messagebox.showinfo("Undo", "No more changes to undo.")

    def load_spectra(self, spectra_list):
        """Load spectra into the preprocessing tab and update the plot, clearing previous history."""
        self.app.batch = SERSSpectraBatch(spectra_list)  # Assign spectra to batch

        # Clear previous preprocessing history and undo stack
        self.preprocess_log.clear()  # Clear the preprocessing log
        self.undo_stack.clear()  # Clear the undo stack
        self.log_table.delete(*self.log_table.get_children())  # Clear the log display in the GUI

        self.update_plot()  # Update plot with new data


    def update_plot(self):
        """Update the plot with the spectra data."""
        self.ax.clear()
        if self.app.batch and self.app.batch.spectra_list:
            for spectra in self.app.batch.spectra_list:
                min_length = min(len(spectra.raman_shifts), len(spectra.raman_intensities))
                self.ax.plot(spectra.raman_shifts[:min_length], spectra.raman_intensities[:min_length], label='Spectra')
            self.ax.legend()
            self.ax.set_xlabel('Raman Shift')
            self.ax.set_ylabel('Intensity')
            self.ax.set_title('Spectra')
            self.canvas.draw()

    def plot_spectra(self, processed=False):
        """Plot either the initial or processed spectra, ensuring matching array lengths."""
        self.ax.clear()
        for spectra in self.app.batch.spectra_list:
            min_length = min(len(spectra.raman_shifts), len(spectra.raman_intensities))
            self.ax.plot(spectra.raman_shifts[:min_length], spectra.raman_intensities[:min_length],
                         label='Processed' if processed else 'Initial')
        self.ax.legend()
        self.ax.set_title('Processed Data' if processed else 'Initial Spectra')
        self.canvas.draw()

    def save_processed_data(self):
        if not self.app.batch.spectra_list:
            messagebox.showerror("Save Error", "There is no processed data to save.")
            return

        folder_path = filedialog.askdirectory(title="Select Directory to Save Processed Data")
        if not folder_path:
            messagebox.showerror("Save Error", "No directory was selected.")
            return

        if os.listdir(folder_path):
            messagebox.showerror("Save Error", "The chosen directory is not empty. Please select an empty directory.")
            return

        file_type = getattr(self.app.load_data_tab, 'file_type', 'CSV')  # Use a default file type if not set

        try:
            self.save_data_in_format(folder_path, file_type)
            messagebox.showinfo("Save Success", f"All processed data has been successfully saved in {folder_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving the data: {str(e)}")

    def save_data_in_format(self, folder_path, file_type):
        """Saves the data in the specified format."""
        try:
            for i, spectra in enumerate(self.app.batch.spectra_list):
                file_path = os.path.join(folder_path, f"spectra_{i + 1}.{file_type.lower()}")

                if file_type == 'TXT':
                    np.savetxt(file_path, np.column_stack((spectra.raman_shifts, spectra.raman_intensities)),
                               delimiter='\t')
                elif file_type == 'CSV':
                    np.savetxt(file_path, np.column_stack((spectra.raman_shifts, spectra.raman_intensities)),
                               delimiter=',', fmt='%s')
                elif file_type == 'XLSX':
                    df = pd.DataFrame({
                        'Raman Shifts': spectra.raman_shifts,
                        'Intensities': spectra.raman_intensities
                    })
                    df.to_excel(file_path, index=False)
        except Exception as e:
            raise Exception(f"Failed to save data: {e}")

    def apply_to_all_batches(self):
        """Apply the recorded preprocessing steps to all batches in the machine learning data."""
        training_dataset = self.app.machine_learning_tab.master.training_dataset
        testing_dataset = self.app.machine_learning_tab.master.testing_dataset

        if training_dataset:
            for batch in training_dataset.dataset.values():
                self.apply_log_to_batch(batch)

        if testing_dataset:
            for batch in testing_dataset.dataset.values():
                self.apply_log_to_batch(batch)

        messagebox.showinfo("Preprocessing Applied", "All preprocessing steps have been applied to all batches.")

    def apply_log_to_batch(self, batch):
        """Apply the logged preprocessing steps to a single batch."""
        for entry in self.preprocess_log:
            for spectra in batch.spectra_list:
                self.execute_preprocessing_step(spectra, entry)

    def execute_preprocessing_step(self, spectra, step):
        parts = step.split(' - ', 2)
        if len(parts) < 3:
            messagebox.showerror("Error", "Log format error, cannot unpack step: " + step)
            return

        method, submethod, param_str = parts
        try:
            params = eval(param_str) if param_str.strip() else {}  # Safely handle empty parameter strings
        except SyntaxError as e:
            messagebox.showerror("Error", "Failed to parse parameters: " + str(e))
            return

        # Apply methods based on parsed log
        if method == "Normalization":
            if submethod == "By Area":
                spectra.normalize_by_area()  # Assuming no parameters needed
            elif submethod == "By Peak":
                spectra.normalize_by_peak()  # Assuming no parameters needed
        elif method == "Smoothing":
            # Assume parameters are needed and handle them
            if submethod == "SG Filter":
                spectra.apply_savgol_filter(int(params.get("Window Length", 11)),
                                            int(params.get("Polynomial Order", 2)))
            elif submethod == "Wavelet Filter":
                spectra.denoise_by_wavelet(params.get("Wavelet Name", "db4"), int(params.get("Decomposition Level", 4)))
            elif submethod == "FIR Filter":
                spectra.denoise_by_fir_filter(int(params.get("Filter Order", 51)), params.get("Window Type", "hamming"))
        elif method == "Normalization":
            if submethod == "By Area":
                spectra.normalize_by_area()
            elif submethod == "By Peak":
                spectra.normalize_by_peak()
        elif method == "Baseline Removal":
            if submethod == "GLF":
                # Implementation needed for GLF when ready
                pass
            elif submethod == "ModPoly":
                spectra.baseline_removal_by_ModPoly(int(params["Polynomial Degree"]))
            elif submethod == "airPLS":
                spectra.remove_baseline_airPLS(float(params["Lambda"]), int(params["Porder"]), int(params["Itermax"]))
        elif method == "Despiking":
            spectra.despiking(float(params["Threshold"]))
        elif method == "Interpolation":
            spectra.interpolate(float(params["Start"]), float(params["Stop"]), float(params["Step"]))
        elif method == "Cropping":
            spectra.crop(float(params["Start Raman Shift"]), float(params["End Raman Shift"]))

class StatisticsTab(ttk.Frame):
    def __init__(self, notebook, app):
        super().__init__(notebook)
        self.app = app
        self.init_ui()

    def init_ui(self):
        self.reference_frame = ttk.LabelFrame(self, text="Select Reference Spectrum")
        self.reference_frame.pack(fill='x', padx=5, pady=5)

        self.selected_reference_index = tk.StringVar()
        self.reference_dropdown = ttk.Combobox(self.reference_frame, textvariable=self.selected_reference_index)
        self.reference_dropdown.pack(padx=5, pady=5)

        self.confirm_ref_button = ttk.Button(self.reference_frame, text="Confirm Reference", bootstyle='info', command=self.confirm_reference)
        self.confirm_ref_button.pack(pady=5)

        self.operation_var = tk.StringVar()
        self.operations = ["Spectral Average", "Spectral SD", "Spectral Correlation", "FDAD", "Spectral MAE", "Spectral MSE", "Inner Product"]
        self.operation_menu = ttk.Combobox(self, textvariable=self.operation_var, values=self.operations)
        self.operation_menu.pack(pady=5)

        self.execute_button = ttk.Button(self, text="Execute", bootstyle='success', command=self.execute_operation)
        self.execute_button.pack(pady=5)

        self.export_button = ttk.Button(self, text="Export All Results", bootstyle='primary', command=self.export_all_results)
        self.export_button.pack(pady=10)

        self.ref_figure, self.ref_ax = plt.subplots()
        self.ref_canvas = FigureCanvasTkAgg(self.ref_figure, self.reference_frame)
        self.ref_canvas_widget = self.ref_canvas.get_tk_widget()
        self.ref_canvas_widget.pack(fill=tk.BOTH, expand=True)

    def update_file_list(self, spectra_list):
        # Update dropdown with the list of file names
        self.reference_dropdown['values'] = [f"File {idx + 1}" for idx, _ in enumerate(spectra_list)]
        self.reference_dropdown.current(0)  # Default to the first file as reference
        self.confirm_reference()  # Auto-confirm the first file as reference upon loading

    def confirm_reference(self):
        # Get the index of the selected file
        idx = int(self.selected_reference_index.get().split(' ')[-1]) - 1
        if self.app.batch and 0 <= idx < len(self.app.batch.spectra_list):
            self.app.batch.reference_spectra = self.app.batch.spectra_list[idx]
            self.plot_reference_spectrum()

    def plot_reference_spectrum(self):
        self.ref_ax.clear()
        reference_spectrum = self.app.batch.reference_spectra
        if reference_spectrum:
            self.ref_ax.plot(reference_spectrum.raman_shifts, reference_spectrum.raman_intensities, label='Reference Spectrum')
            self.ref_ax.legend()
            self.ref_ax.set_title('Reference Spectrum')
            self.ref_canvas.draw()

    def execute_operation(self):
        """Execute the selected operation and handle displaying results appropriately."""
        operation = self.operation_var.get()
        batch = self.app.batch
        result = None

        try:
            if operation == "Spectral Correlation":
                result = batch.spectral_correlation()
                self.display_correlation_matrix(result)  # Display correlation matrix in a new window
            elif operation == "Spectral Average":
                result = batch.spectral_average()
                self.plot_operation_result(result, "Spectral Average")
            elif operation == "Spectral SD":
                result = batch.spectral_SD()
                self.plot_operation_result(result, "Spectral Standard Deviation")
            elif operation == "FDAD":
                result = batch.FDAD()
                self.plot_operation_result(result, "Functional Data Analysis Derivatives")
            elif operation == "Spectral MAE":
                result = batch.spectral_MAE()
                self.plot_operation_result(result, "Spectral Mean Absolute Error")
            elif operation == "Spectral MSE":
                result = batch.spectral_MSE()
                self.plot_operation_result(result, "Spectral Mean Squared Error")
            elif operation == "Inner Product":
                result = batch.inner_product()
                self.plot_operation_result(result, "Inner Product of Spectra")
            else:
                messagebox.showerror("Error", f"Operation {operation} is not available.")
                return

            if result is not None and not isinstance(result, np.ndarray):
                self.plot_operation_result(result, operation)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_correlation_matrix(self, matrix):
        """Display the correlation matrix in a new Toplevel window."""
        if matrix is None or not matrix.size:
            messagebox.showerror("Error", "No data available to display.")
            return

        correlation_window = tk.Toplevel(self)
        correlation_window.title("Spectral Correlation Matrix")
        correlation_window.geometry("600x400")  # Adjust size as needed

        # Use a Text widget to display the matrix in a simple way
        text_widget = tk.Text(correlation_window, wrap="none")
        text_widget.pack(expand=True, fill='both')

        # Convert the matrix to a string format for display
        for row in matrix:
            line = "\t".join(f"{val:.3f}" for val in row) + "\n"
            text_widget.insert("end", line)

        text_widget.config(state="disabled")  # Make the text widget read-only
        # Add scrollbars if necessary
        scroll_y = ttk.Scrollbar(correlation_window, orient='vertical', command=text_widget.yview)
        scroll_y.pack(side='right', fill='y')
        text_widget.config(yscrollcommand=scroll_y.set)

        scroll_x = ttk.Scrollbar(correlation_window, orient='horizontal', command=text_widget.xview)
        scroll_x.pack(side='bottom', fill='x')
        text_widget.config(xscrollcommand=scroll_x.set)

    def plot_operation_result(self, spectra, operation_name):
        """Plot the results of operations that yield a spectrum."""
        self.results_ax.clear()
        if spectra.raman_shifts is not None and spectra.raman_intensities is not None:
            self.results_ax.plot(spectra.raman_shifts, spectra.raman_intensities, label=operation_name)
            self.results_ax.legend()
            self.results_ax.set_title(operation_name)
        self.results_canvas.draw()


    def display_table(self, data):
        # Clear any previous widgets in the results canvas
        for widget in self.results_canvas_widget.winfo_children():
            widget.destroy()

        # Assuming 'data' is a 2D list or numpy array
        for row_index, row in enumerate(data):
            for column_index, cell in enumerate(row):
                label = tk.Label(self.results_canvas_widget, text=f"{cell:.2f}", borderwidth=1, relief="solid")
                label.grid(row=row_index, column=column_index, padx=1, pady=1)
        # Update the layout
        self.results_canvas_widget.update_idletasks()

    def export_all_results(self):
        """Export all spectral results to files in the selected format."""
        folder_path = filedialog.askdirectory(title="Select Directory to Export Results")
        if not folder_path:
            return  # User cancelled the dialog

        # Check if the directory is empty
        if os.listdir(folder_path):
            messagebox.showerror("Export Error",
                                 "The selected directory is not empty. Please select an empty directory.")
            return  # End the operation if the directory is not empty

        file_type = self.app.load_data_tab.file_type  # Assuming this attribute exists and is set during import

        # Prepare data to export
        data_dict = {
            "Reference": self.app.batch.reference_spectra.raman_intensities,
            "Spectral_Average": self.app.batch.spectral_average().raman_intensities,
            "Spectral_SD": self.app.batch.spectral_SD().raman_intensities,
            # "Spectral_Correlation": self.app.batch.spectral_correlation(),
            "FDAD": self.app.batch.FDAD().raman_intensities,
            "Spectral_MAE": self.app.batch.spectral_MAE().raman_intensities,
            "Spectral_MSE": self.app.batch.spectral_MSE().raman_intensities,
            "Inner_Product": self.app.batch.inner_product().raman_intensities
        }

        if file_type in ['TXT', 'CSV']:
            self.export_text_or_csv(data_dict, folder_path, file_type)
        elif file_type == 'XLSX':
            self.export_excel(data_dict, folder_path)

    def export_text_or_csv(self, data_dict, folder_path, file_type):
        delimiter = '\t' if file_type == 'TXT' else ','
        raman_shifts = self.app.batch.reference_spectra.raman_shifts

        for key, data in data_dict.items():
            if isinstance(data, np.ndarray) and key != "Spectral_Correlation":  # Skip correlation matrix for this type
                file_path = os.path.join(folder_path, f"{key}.{file_type.lower()}")
                combined_data = np.column_stack((raman_shifts, data))  # Combine Raman shifts with data
                np.savetxt(file_path, combined_data, delimiter=delimiter, fmt='%0.4f')
            elif isinstance(data, np.matrix):
                file_path = os.path.join(folder_path, f"{key}.{file_type.lower()}")
                np.savetxt(file_path, data, delimiter=delimiter, fmt='%0.4f')

    def export_excel(self, data_dict, folder_path):
        with pd.ExcelWriter(os.path.join(folder_path, 'results.xlsx')) as writer:
            raman_shifts = self.app.batch.reference_spectra.raman_shifts
            # Create a DataFrame for each type of data except the correlation matrix
            for key, data in data_dict.items():
                if key != "Spectral_Correlation":  # Exclude correlation matrix from Excel export
                    df = pd.DataFrame({
                        'Raman Shifts': raman_shifts,
                        key: data
                    })
                    df.to_excel(writer, sheet_name=key, index=False)


class MachineLearningTab(ttk.Frame):
    def __init__(self, notebook, app):
        super().__init__(notebook)
        self.app = app
        self.setup_model_controls()
        self.setup_dataset_summary()

    def setup_model_controls(self):
        settings_container = ttk.Frame(self)
        settings_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        settings_container.columnconfigure(1, weight=1)

        model_var = tk.StringVar()
        model_menu = ttk.Combobox(settings_container, textvariable=model_var, values=['SVM', 'LDA'], state="readonly", width=15)
        model_menu.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        model_menu.bind('<<ComboboxSelected>>', self.update_model_settings)

        settings_frame = ttk.Frame(settings_container)
        settings_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        self.build_model_button = ttk.Button(settings_container, text="Build Model", bootstyle='success', command=self.build_model)
        self.build_model_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        self.test_model_button = ttk.Button(settings_container, text="Test Model", bootstyle='danger', command=self.test_model)
        self.test_model_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

    def setup_dataset_summary(self):
        self.training_frame = ttk.LabelFrame(self, text="Training Data Summary")
        self.training_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.testing_frame = ttk.LabelFrame(self, text="Testing Data Summary")
        self.testing_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        self.setup_dataset_controls(self.training_frame, 'training')
        self.setup_dataset_controls(self.testing_frame, 'testing')

    def setup_dataset_controls(self, parent, dataset_type):
        batch_var = tk.StringVar()
        batch_dropdown = ttk.Combobox(parent, textvariable=batch_var)
        batch_dropdown.pack(padx=5, pady=5)
        load_button = ttk.Button(parent, text="Load for Preprocessing", command=lambda: self.load_batch_for_processing(batch_var.get(), dataset_type))
        load_button.pack(padx=5, pady=5)
        self.update_dataset_summary(parent, dataset_type)

    def update_dataset_summary(self, parent, dataset_type):
        # Clear all existing children in the parent frame before adding new content
        for widget in parent.winfo_children():
            widget.destroy()

        # Adding a Treeview for displaying dataset summary
        tree = ttk.Treeview(parent, columns=(
        'Folder Name', 'Number of Files', 'Min Raman Shift', 'Max Raman Shift', 'Min Features'), show='headings')
        tree.heading('Folder Name', text='Folder Name')
        tree.heading('Number of Files', text='Number of Files')
        tree.heading('Min Raman Shift', text='Min Raman Shift')
        tree.heading('Max Raman Shift', text='Max Raman Shift')
        tree.heading('Min Features', text='Min Features')
        tree.pack(fill='both', expand=True)

        # Retrieve the appropriate dataset based on the type (training or testing)
        dataset = self.app.training_dataset if dataset_type == 'training' else self.app.testing_dataset
        if dataset and hasattr(dataset, 'dataset'):
            for label, batch in dataset.dataset.items():
                min_raman_shift = min([min(s.raman_shifts) for s in batch.spectra_list])
                max_raman_shift = max([max(s.raman_shifts) for s in batch.spectra_list])
                min_features = min([len(s.raman_shifts) for s in batch.spectra_list])
                tree.insert('', 'end', values=(
                label, len(batch.spectra_list), f"{min_raman_shift:.2f}", f"{max_raman_shift:.2f}", min_features))

        # Add the Combobox and the button below the tree view
        batch_var = tk.StringVar()
        batch_dropdown = ttk.Combobox(parent, textvariable=batch_var,
                                      values=list(dataset.dataset.keys()) if dataset and hasattr(dataset,
                                                                                                 'dataset') else [])
        batch_dropdown.pack(padx=5, pady=5)
        load_button = ttk.Button(parent, text="Load for Preprocessing",
                                 command=lambda: self.load_batch_for_processing(batch_var.get(), dataset_type))
        load_button.pack(padx=5, pady=5)

    def load_batch_for_processing(self, label, dataset_type):
        if dataset_type == 'training':
            batch = self.app.training_dataset.get_batch(label)
        elif dataset_type == 'testing':
            batch = self.app.testing_dataset.get_batch(label)
        if batch:
            self.app.preprocessing_tab.load_spectra(batch.spectra_list)
            messagebox.showinfo("Batch Loaded", f"Batch from {label} has been loaded for preprocessing.")
        else:
            messagebox.showerror("Load Error", "Failed to load the selected batch.")

    def setup_model_controls(self):
        self.settings_container = ttk.Frame(self)
        self.settings_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.settings_container.columnconfigure(1, weight=1)

        self.model_var = tk.StringVar()
        self.model_menu = ttk.Combobox(self.settings_container, textvariable=self.model_var,
                                       values=['SVM', 'LDA'], state="readonly", width=15)
        self.model_menu.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.model_menu.bind('<<ComboboxSelected>>', self.update_model_settings)

        self.settings_frame = ttk.Frame(self.settings_container)
        self.settings_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        self.setup_cv_controls()

        # Build Model Button
        self.build_model_button = ttk.Button(self.settings_container, text="Build Model", command=self.build_model)
        self.build_model_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Test Model Button
        self.test_model_button = ttk.Button(self.settings_container, text="Test Model", command=self.test_model)
        self.test_model_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

    def setup_cv_controls(self):
        ttk.Label(self.settings_container, text="Number of CV Folds:").grid(row=2, column=0, padx=5, pady=5)
        self.cv_folds_var = tk.IntVar(value=5)
        self.cv_folds_slider = ttk.Scale(self.settings_container, from_=1, to=10, orient="horizontal",
                                         variable=self.cv_folds_var, command=self.update_cv_folds_label)
        self.cv_folds_slider.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.cv_folds_label = ttk.Label(self.settings_container, text="5")
        self.cv_folds_label.grid(row=2, column=2, padx=5, pady=5)

    def update_model_settings(self, event=None):
        for widget in self.settings_frame.winfo_children():
            widget.destroy()

        model = self.model_var.get()
        if model == 'SVM':
            self.create_svm_settings()
        elif model == 'LDA':
            self.create_lda_settings()

    def create_svm_settings(self):
        ttk.Label(self.settings_frame, text="Kernel:").grid(row=0, column=0, padx=5, pady=5)
        self.svm_kernel_var = tk.StringVar(value="linear")
        ttk.Combobox(self.settings_frame, textvariable=self.svm_kernel_var,
                     values=["linear", "poly", "rbf", "sigmoid"], state="readonly", width=10).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    def create_lda_settings(self):
        ttk.Label(self.settings_frame, text="Solver:").grid(row=0, column=0, padx=5, pady=5)
        self.lda_solver_var = tk.StringVar(value="svd")
        ttk.Combobox(self.settings_frame, textvariable=self.lda_solver_var,
                     values=["svd", "lsqr", "eigen"], state="readonly", width=10).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    def update_cv_folds_label(self, value):
        self.cv_folds_label.config(text=f"{int(float(value))}")


    def get_selected_model(self):
        """Retrieve the model based on the selection in the GUI."""
        model_type = self.model_var.get()
        if model_type == 'SVM':
            # Retrieve kernel type from the combobox (ensure this variable is correctly managed in the GUI)
            kernel = self.svm_kernel_var.get()
            return SVC(kernel=kernel)
        elif model_type == 'LDA':
            # Retrieve solver type from the combobox
            solver = self.lda_solver_var.get()
            return LinearDiscriminantAnalysis(solver=solver)
        else:
            raise ValueError("Selected model type is not supported.")


    def prepare_data(self, dataset):
        """Prepare feature matrix X and target vector y from the dataset."""
        features = []
        labels = []
        for label, batch in dataset.dataset.items():
            for spectra in batch.spectra_list:
                features.append(spectra.raman_intensities)  # Using intensities as features
                labels.append(label)  # Using subfolder names as labels

        return StandardScaler().fit_transform(np.array(features)), np.array(labels)  # Standardize features

    def build_model(self):
        """Build the model based on selected parameters and perform cross-validation."""
        X, y = self.app.training_dataset.prepare_features_and_labels()
        model = self.get_selected_model()  # Assuming this method fetches the model based on the combo box selection

        # Cross-validation and metrics calculation
        cv_results = cross_val_score(model, X, y, cv=self.cv_folds_var.get(), scoring='accuracy')
        model.fit(X, y)  # Train the model on the full training set
        self.model = model  # Save the trained model

        # Displaying results
        training_accuracy = np.mean(cv_results)
        self.display_model_results(model, X, y, training_accuracy, "Training Results")

    def test_model(self):
        """Test the already trained model on test data and show metrics."""
        if self.model is None:
            messagebox.showerror("Model Testing", "No model has been built yet.")
            return

        X_test, y_test = self.app.testing_dataset.prepare_features_and_labels()
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Displaying metrics
        self.display_model_results(self.model, X_test, y_test, accuracy, "Testing Results")

    def display_model_results(self, model, X, y, accuracy, title):
        """Show detailed model results in a new window."""
        window = tk.Toplevel(self)
        window.title(title)
        window.geometry("800x600")

        conf_matrix = confusion_matrix(y, model.predict(X))
        self.display_confusion_matrix(conf_matrix, accuracy, window)

        # Classification report
        report_text = ttk.Label(window, text=classification_report(y, model.predict(X)))
        report_text.pack(pady=20)

        # Export results button
        export_button = ttk.Button(window, text="Export Results",
                                   command=lambda: self.app.training_dataset.export_classification_results(
                                       model.predict(X), y, accuracy, filedialog.askdirectory()))
        export_button.pack(pady=20)

    def display_confusion_matrix(self, conf_matrix, accuracy, window):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", ax=ax)
        ax.set_title(f"Confusion Matrix\nAccuracy: {accuracy:.2f}")
        canvas = FigureCanvasTkAgg(fig, window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        return canvas



if __name__ == "__main__":
    app = SpectralApp()
    app.mainloop()
