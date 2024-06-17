import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy 
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.misc import derivative
from scipy.integrate import quad
import statistics as sts



def calculate_sts_sections(df, col,col2, height, func,w, move, mode):
    # Find the peak indices for the specified column
    indices = find_peaks_indices(height, df, col2, mode)
    
    # Calculate the mean for each section and assign it back to the DataFrame
    for i in range( len(indices) - 1):  # Exclude the first and last index
        start_idx, end_idx = indices[i] + move, indices[i + 1] + move
        # Ensure end_idx does not exceed the length of the DataFrame
        end_idx = min(end_idx, len(df) - 1)
        
        section_mean = func(df.loc[start_idx:end_idx, col],w)
        df.loc[start_idx:end_idx, col] = section_mean


def calculate_section_sts(df, col,col2, height,move,mode,func):
    # Find the peak indices for the specified column
    indices = find_peaks_indices(height, df, col2,mode)
    
    # Initialize a list to store the means of each section
    section_means = []
    
    # Calculate the mean for each section and store it in the list
    for i in range(len(indices) - 1):
        start_idx, end_idx = indices[i]+move, indices[i + 1]+move
        section_mean = func(df.loc[start_idx:end_idx, col])
        section_means.append(section_mean)
            # Return the list of means
    return section_means




def calculate_func_sections(df, col, height, func, move, mode):
    # Find the peak indices for the specified column
    indices = find_peaks_indices(height, df, col, mode)
    
    # Calculate the mean for each section and assign it back to the DataFrame
    for i in range( len(indices) - 1):  # Exclude the first and last index
        start_idx, end_idx = indices[i] + move, indices[i + 1] + move
        # Ensure end_idx does not exceed the length of the DataFrame
        end_idx = min(end_idx, len(df) - 1)
        
        section_mean = func(df.loc[start_idx:end_idx, col])
        df.loc[start_idx:end_idx, col] = section_mean


def calculate_section_func(df, col, height,move,mode,func,column):
    # Find the peak indices for the specified column
    indices = find_peaks_indices(height, df, column,mode)
    
    # Initialize a list to store the means of each section
    section_means = []
    
    # Calculate the mean for each section and store it in the list
    for i in range(1,len(indices) - 2):
        start_idx, end_idx = indices[i]+move, indices[i + 1]+move
        section_mean = func(df.loc[start_idx:end_idx, col])
        section_means.append(section_mean)
            # Return the list of means
    return section_means


def find_peaks_indices( height,df,column,mode=""):
    indices, _ = find_peaks(df[f"{column}_{mode}"], height=height)
    indices = np.insert(indices, 0, 0)  # Insert the first index at the beginning
    indices = np.append(indices, len(df) - 1)  # Append the last index at the end
    return indices


# Function to normalize data using min-max scaling
def normalize_section(section):
    return (section - section.min()) / (section.max() - section.min())

def nothing(series):
    return series

def mean_normalization(series):
    return (series - series.mean()) / series.std()


def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())


def robust_scaler(series):
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    return (series - series.median()) / iqr


def similarity_scaling(series):
    col_mean = series.mean()
    col_std = series.std()
    return (series - col_mean) / col_std



def normalize(indices,data,Normfunc,w):
    # Splitting the data into sections and normalizing each one
    data_copy = data.copy()
    normalized_sections = []
    for i in range(len(indices) - 1):
        start_idx, end_idx = indices[i], indices[i+1]
        section = data_copy.iloc[start_idx:end_idx]
        # Check if the section is not all zeros
        if not section.eq(0).all():
            normalized_section = Normfunc(section,w)
            normalized_sections.append(normalized_section)
        else:
            normalized_sections.append(section)
        
    # for idx in indices:
    #     data_copy.iloc[max(0, idx-1):min(len(data_copy), idx+2)] =  data_copy.iloc[max(0, idx-1):min(len(data_copy), idx+2)].std()
    # Concatenating the normalized sections back into one series
    normalized_data = pd.concat(normalized_sections).reset_index(drop=True)
    return normalized_data
    
def labeling(indices, data):
    # Make sure 'Label' column exists
    if 'Label' not in data.columns:
        data['Label'] = ''
    
    # Iterate through pairs of indices to label each section with 'g1', 'g2', ..., 'gn'
    for i in range(len(indices) - 1):
        start_index = indices[i]
        end_index = indices[i + 1] - 1  # Subtract 1 to include the end_index in the current section
        label = f'g{i + 1}'  # Create label string
        data.loc[start_index:end_index, 'Label'] = label  # Assign label to the section



# Means
def pmean(series):
    return series.mean()
def wmean(series,w):
    return series.rolling(window=w).mean()
# Median
def pmedian(series):
    return sts.median(series)

# Variance
def pvar(series):
    return series.var()
def wvar(series,w):
    return series.rolling(window=w).var()

# STD
def pstd(series):
    return series.std()
def wstd(series,w):
    return series.rolling(window=w).std()


# Skew
def pskew(series):
    return series.skew()
def wskew(series,w):
    return series.rolling(window=w).skew()


# Kurt
def pkurt(series):
    return series.kurt()
def wkurt(series,w):
    return series.rolling(window=w).kurt()


def mean(data):
    """Calculate the mean of the data."""
    return np.mean(data)

def std(data):
    """Calculate the standard deviation of the data."""
    return np.std(data)

def rang(data):
    """Calculate the range of the data."""
    return np.max(data) - np.min(data)

def median(data):
    """Calculate the median of the data."""
    return np.median(data)

def iqr(data):
    """Calculate the interquartile range of the data."""
    return np.percentile(data, 75) - np.percentile(data, 25)

def skewness(data):
    """Calculate the skewness of the data."""
    return scipy.stats.skew(data)

def kurtosis(data):
    """Calculate the kurtosis of the data."""
    return scipy.stats.kurtosis(data)

def peak_amplitude(data):
    """Find the peak amplitude in the data."""
    return np.max(data)

def num_peaks(data):
    """Count the number of peaks in the data."""
    peaks, _ = scipy.signal.find_peaks(data)
    return len(peaks)

def zero_crossings(data):
    """Count the number of zero crossings in the data."""
    zero_crossings = np.where(np.diff(np.sign(data)))[0]
    return len(zero_crossings)

def slope(data):
    """Calculate the slope of the data."""
    x = np.arange(len(data))
    slope, intercept = np.polyfit(x, data, 1)
    return slope

def dominant_frequency(data):
    """Find the dominant frequency in the data."""
    fft = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), d=1/42)  # Assuming a sample rate of 42 Hz
    dominant_freq = freqs[np.argmax(np.abs(fft))]
    return dominant_freq

def frequency_bandwidth(data):
    """Calculate the frequency bandwidth of the data."""
    # Calculate Power Spectral Density
    psd = np.abs(np.fft.fft(data)) ** 2
    freqs = np.fft.fftfreq(len(data), d=1/42)

    # Find significant peaks in PSD
    peaks, _ = scipy.signal.find_peaks(psd)

    # Calculate bandwidth
    bandwidth = freqs[peaks[-1]] - freqs[peaks[0]]
    return bandwidth

def power_spectral_density(data):
    """Calculate the power spectral density of the data."""
    psd = np.abs(np.fft.fft(data)) ** 2
    freqs = np.fft.fftfreq(len(data), d=1/42)
    return freqs, psd

def finger_closure_duration(data):
    """Calculate the duration of finger closure."""
    # Define a threshold to determine finger closure (e.g., based on sensor values)
    threshold = 0.5 # Example threshold

    # Find start and end indices of closure periods
    closure_starts = np.where(np.diff(data > threshold) == 1)[0] + 1
    closure_ends = np.where(np.diff(data > threshold) == -1)[0] + 1

    # Calculate closure durations
    closure_durations = closure_ends - closure_starts
    return closure_durations

def finger_closure_rate(data):
    """Calculate the rate of finger closure."""
    # Calculate closure durations
    closure_durations = finger_closure_duration(data)
    # Calculate closure rates (using appropriate time units)
    closure_rates = np.diff(data) / closure_durations
    return closure_rates

def finger_opening_rate(data):
    """Calculate the rate of finger opening."""
    # Calculate opening durations (similar to closure duration)
    opening_durations = ...  
    # Calculate opening rates
    opening_rates = ...
    return opening_rates

def finger_closure_sequence(data):
    """Determine the sequence of finger closure."""
    # Define a threshold to determine finger closure
    threshold = 0.5 

    # Find closure and opening times for each finger
    closure_times = []
    opening_times = []
    for finger_data in data:
        closure_starts = np.where(np.diff(finger_data > threshold) == 1)[0] + 1
        closure_ends = np.where(np.diff(finger_data > threshold) == -1)[0] + 1
        closure_times.append(closure_starts)
        opening_times.append(closure_ends)

    # Order fingers by closure time
    finger_order = np.argsort(closure_times, axis=0)
    return finger_order

def first_derivative(data):
    """
    Calculates the first derivative (velocity) of the sensor data.

    Args:
        data (np.ndarray): Sensor data for a single finger.

    Returns:
        np.ndarray: First derivative of the sensor data.
    """
    return np.diff(data)


def second_derivative(data):
    """
    Calculates the second derivative (acceleration) of the sensor data.

    Args:
        data (np.ndarray): Sensor data for a single finger.

    Returns:
        np.ndarray: Second derivative of the sensor data.
    """
    return np.diff(data, n=2)

def n_derivative(data,xdata,order=1):
    """
    Calculates the second derivative (acceleration) of the sensor data.

    Args:
        data (np.ndarray): Sensor data for a single finger.

    Returns:
        np.ndarray: Second derivative of the sensor data.
    """
    return derivative(data,xdata,n=order)
