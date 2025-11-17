# compat.py
# Used to make RULModels compatible between different datasets.

from scipy.fft import rfft
import numpy as np
import pandas as pd

def convert_femto_to_cmapss(df: pd.DataFrame)->pd.DataFrame:
    """
        Transforms the original FEMTO dataset into a CMAPSS-like alternative version where
        each segment is flattened into several statistical time/frequency-related features.
    
        Parameters
        ----------            
        df : pd.DataFrame 
            The raw FEMTO dataframe.

        Returns
        -------
        df : pd.DataFrame 
            The new CMAPSS-like dataframe.
    """
    old_names = ["hori_acc","vert_acc"]
    f_idx = list(np.repeat(["f" + str(i) for i in range(16)],2))
    names = [old_names[i%2]+"_"+f_idx[i] for i in range(32)]
    
    X, y, u, c = get_ts_and_labels(df, vars_to_keep=old_names)
    X = extract_time_freq_features(X) # shape : (segments, 2*8*features)

    new_df = pd.DataFrame(data=X, columns=names)
    new_df["RUL"] = y
    new_df["unit"] = u
    new_df["condition"] = c

    return new_df

def extract_time_features(x: np.array)->np.array:
    """
        Computes the following time-domain features per segment for each unit :
        Mean, std, RMS, kurtosis, max and min of absolute values, skewness and peak-to-peak.

        Parameters
        ----------            
        x : np.array 
            The feature dataset array of shape (segments, time, features).

        Returns
        -------
        x_new : np.array
            The new feature dataset array of shape (segments, 8, features).
    """
    # Mean, Std
    mu = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    
    # RMS
    rms = np.mean(np.square(x), axis=1, keepdims=True)**0.5

    # Kurtosis
    kurtosis = np.mean( ((x - mu)/std)**4, axis=1, keepdims=True)

    # Max, min of abs
    maxi = np.max(np.abs(x), axis=1, keepdims=True)
    mini = np.min(np.abs(x), axis=1, keepdims=True)

    # Skewness
    skew = np.mean( ((x - mu)/std)**3, axis=1, keepdims=True)

    # Peak-to-peak
    p2p = np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True)

    x_new = np.hstack([mu, std, rms, kurtosis, maxi, mini, skew, p2p])

    return x_new

def extract_freq_features(x: np.array)->np.array:
    """
        Computes the following frequency-domain features per segment for each unit :
        Mean, std, RMS, kurtosis, max and min of absolute values, skewness and peak-to-peak.

        Parameters
        ----------            
        x : np.array 
            The feature dataset array of shape (segments, time, features).

        Returns
        -------
        x_new : np.array
            The new feature dataset array of shape (segments, 8, features).
    """
    y = np.abs(rfft(x, axis=1)) # amplitude of frequencies
    return extract_time_features(y)

def extract_time_freq_features(x: np.array)->np.array:
    """
        Computes the following time and frequency-domain features per segment for each unit :
        Mean, std, RMS, kurtosis, max and min of absolute values, skewness and peak-to-peak.

        Parameters
        ----------            
        x : np.array 
            The feature dataset array of shape (segments, time, features).

        Returns
        -------
        x_new : np.array
            The new feature dataset array of shape (segments, 16, features).
    """
    x1 = extract_time_features(x)
    x2 = extract_freq_features(x)
    x = np.hstack([x1,x2])
    x_new = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
    return x_new

def get_ts_and_labels(df: pd.DataFrame,
                      vars_to_keep: list,
                      )->tuple:
    """
        Returns the time series and its labels.

        Parameters
        ----------
        df : df.DataFrame
            The dataset. Must have "unit" and "RUL" as columns at least, and these indices must be sorted.

        vars_to_keep : list of str
            The list of variables to keep.

        Returns
        -------
        X : np.array
            The array of segments of each unit, of shape (segments, time, features).

        labels : np.array
            The associated RUL values array of shape (segments).

        units : np.array
            The unit indices array of shape (segments).

        conditions : np.array
            The operating condition group indices array of shape (segments).
    """
    groups = df.groupby(['unit','RUL'], sort=False)
    data_series = groups[vars_to_keep].apply(np.array)
    labels = groups['RUL'].head(1).values
    units = groups['unit'].head(1).values
    conditions = groups['condition'].head(1).values
    X = np.array(data_series.to_list())
    return X, labels, units, conditions