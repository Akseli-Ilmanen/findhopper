import numpy as np
from scipy.signal import butter, filtfilt

def filter(data, lowcut=None, highcut=None, fs=None, order=5, btype='band'):
    """
    Apply a Butterworth filter to the input data.
    
    Parameters
    ----------
    data : np.ndarray
        Input signal array. Can be 1D (single channel) or 2D (multi-channel, shape: [samples, channels]).
    lowcut : float, optional
        Lower cutoff frequency for the filter (Hz). Required for 'band' and 'low' types.
    highcut : float, optional
        Upper cutoff frequency for the filter (Hz). Required for 'band' and 'high' types.
    fs : float, optional
        Sampling frequency of the input data (Hz).
    order : int, optional
        Order of the Butterworth filter. Default is 5.
    btype : {'band', 'low', 'high'}, optional
        Type of filter to apply: 'band' for bandpass, 'low' for lowpass, 'high' for highpass. Default is 'band'.
    
    Returns
    -------
    filtered_data : np.ndarray
        Filtered signal array with the same shape as the input data.
    """
    nyquist = 0.5 * fs
    if btype == 'band':
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype=btype)
    elif btype == 'low':
        low = lowcut / nyquist
        b, a = butter(order, low, btype=btype)
    elif btype == 'high':
        high = highcut / nyquist
        b, a = butter(order, high, btype=btype)

    # Apply filter to each channel separately
    if data.ndim == 1:
        return filtfilt(b, a, data)
    else:
        return np.stack([filtfilt(b, a, data[:, ch]) for ch in range(data.shape[1])], axis=1)
    


def find_peak_boundaries(norm_data, norm_peaks, boundary_threshold):
    """
    For each peak index in `norm_peaks`, the function searches left and right from the peak position
    in `norm_data` until the signal drops below or equals the `boundary_threshold`.
    
    Parameters
    ----------
        norm_data (array-like): The normalized and resampled signal channel as a 1D array.
        norm_peaks (array-like): Indices of detected peaks within the signal channel.
        boundary_threshold (float): The threshold value used to determine the boundaries of each peak region.

    Returns:
    ----------
        list of tuple: A list of (start_index, end_index) tuples, e.g. [(0, 100), (250, 350)]
    """
    regions = []
    used = set()
    for idx, peak in enumerate(norm_peaks):
        # Search left
        left = peak
        while left > 0 and norm_data[left] > boundary_threshold:
            left -= 1
        # Search right
        right = peak
        while right < len(norm_data) - 1 and norm_data[right] > boundary_threshold:
            right += 1
        # Only add if this region is not already covered
        region = (int(left), int(right))
        if region not in used:
            regions.append(region)
            used.add(region)
    return regions
