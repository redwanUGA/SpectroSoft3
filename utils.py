import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
import torch


def WhittakerSmooth(x, w, lambda_, differences=1):
    '''
    Penalized least squares algorithm for background fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties

    output
        the fitted background vector
    '''
    X = np.matrix(x)
    m = X.size
    E = eye(m, format='csc')
    for i in range(differences):
        E = E[1:] - E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * E.T * E))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting

    output
        the fitted background vector
    '''
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if (i == itermax): print('WARING max iteration reached!')
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    return z


def mixed_gauss_lorentz(x, A, v_g, sigma_g, L, v_l, sigma_l, I_0):
    '''
    A mixture of Gaussian and Lorentzian function for GLF fitting.

    input
        x: input wavenumber
        A: amplitude of the Gaussian function
        v_g: center of the Gaussian peak
        sigma_g: standard deviation of the Gaussian function
        L: area of the Lorentzian function
        v_l: center of the Lorentzian peak
        sigma_l: width of the Lorentzian peak
        I_0: “ground” level of the SERS spectrum at wavenumber x

    output
        the value of the gaussian-lorentzian function at wavenumber x
    '''
    gaussian = A * np.exp(-(x - v_g) ** 2 / (2 * sigma_g ** 2))
    lorentzian = (2 * L * sigma_l) / (4 * np.pi * ((x - v_l) ** 2) + sigma_l ** 2)
    return gaussian + lorentzian + I_0


def align_spectra(spectra_list):
    """
    Aligns the lengths of spectra in a list by trimming them to the shortest length.

    Parameters:
    - spectra_list: A list of SERSSpectra objects.

    Returns:
    - A list of aligned raman_intensity arrays.
    """
    if not spectra_list:
        return np.array([])

    min_length = min(len(spectra.raman_intensities) for spectra in spectra_list)
    return [spectra.raman_intensities[:min_length] for spectra in spectra_list]


def calculate_first_derivative(raman_shifts, intensities):
    # Calculate the first derivative using a simple finite difference method
    derivative_intensities = np.diff(intensities) / np.diff(raman_shifts)
    # To maintain the array length, append the last derivative at the end
    derivative_intensities = np.append(derivative_intensities, derivative_intensities[-1])
    return derivative_intensities


def predict_with_CNN(model, X_test):
    # Convert test data to PyTorch tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # Add channel dimension

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to the correct device
    model.to(device)

    # Ensure model is in evaluation mode
    model.eval()

    # Move test data to the correct device
    X_test_tensor = X_test_tensor.to(device)

    # No gradient calculation needed
    with torch.no_grad():
        # Forward pass to get the logits
        logits = model(X_test_tensor)

        # Convert logits to probabilities and then to class predictions
        # Assuming you're dealing with a classification problem
        preds = torch.argmax(logits, dim=1)

    # Move predictions back to CPU if needed and convert to numpy array
    preds = preds.cpu().numpy()

    return preds

