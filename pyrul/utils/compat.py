# compat.py used to make RULModels compatible between different datasets.
from scipy.fft import rfft
import numpy as np
import pandas as pd

def convert_cmapss_to_femto(df: pd.DataFrame)->pd.DataFrame:
    old_names = ["hori_acc","vert_acc"]
    f_idx = list(np.repeat(["f" + str(i) for i in range(16)],2))
    names = [old_names[i%2]+"_"+f_idx[i] for i in range(32)]
    
    X, y, u, c = get_ts_and_labels(df, vars_to_keep=old_names)
    X = extract_time_freq_features(X) # shape : (segments, 8*features)

    new_df = pd.DataFrame(data=X, columns=names)
    new_df["RUL"] = y
    new_df["unit"] = u
    new_df["condition"] = c

    return new_df

def extract_time_features(x: np.array
                          )->np.array:
    """
        Computes the following time-domain features per sub segment for each unit :
        RMS, Kurtosis, Max, Std, Variance, Skewness, Peak-to-Peak and Min.

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

def extract_freq_features(x: np.array
                          )->np.array:
    """
        Computes the following frequency-domain features per sub segment for each unit :
        RMS, Kurtosis, Max, Std, Variance, Skewness, Peak-to-Peak and Min.

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
    x1 = extract_time_features(x)
    x2 = extract_freq_features(x)
    x = np.hstack([x1,x2])
    x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
    return x

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
            The list of time series of each unit.

        y : np.array
            The associated RUL values.

        units : np.array
            The unit indices.

        c : np.array
            The operating condition groups indices.
    """
    groups = df.groupby(['unit','RUL'], sort=False)
    data_series = groups[vars_to_keep].apply(np.array)
    labels = groups['RUL'].head(1).values
    units = groups['unit'].head(1).values
    conditions = groups['condition'].head(1).values
    X = np.array(data_series.to_list())
    y = np.expand_dims(labels,-1)
    u = np.expand_dims(units,-1)
    c = np.expand_dims(conditions,-1)
    return X, y, units, c