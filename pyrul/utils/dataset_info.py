import numpy as np
import pandas as pd

def get_segment_len(ds_name: str):
    """
        Returns the length of a segment depending on the chosen dataset.
    """
    if ds_name == "cmapss_fd001":
        return 1
    elif ds_name == "femto":
        return 2560
    else:
        raise ValueError("Dataset name'",ds_name,"'not found.")
        
def get_min_ts_size(ds_name: str):
    """
        Returns the minimum length of a trajectory depending on the chosen dataset.
    """
    if ds_name == "cmapss_fd001":
        return 31
    elif ds_name == "femto":
        return 229
    else:
        raise ValueError("Dataset name'",ds_name,"'not found.")

def get_max_ts_size(ds_name: str):
    """
        Returns the minimum length of a trajectory depending on the chosen dataset.
    """
    if ds_name == "cmapss_fd001":
        return 1000
    elif ds_name == "femto":
        return 3000
    else:
        raise ValueError("Dataset name'",ds_name,"'not found.")

def get_default_folds(ds_name: str):
    """
        Returns the number of folds depending on the chosen dataset.
    """
    if ds_name == "cmapss_fd001":
        return 10
    elif ds_name == "femto":
        return 6
    else:
        raise ValueError("Dataset name'",ds_name,"'not found.")

def get_defaults_vars(ds_name: str, ds_name_ori: str):
    """
        Returns the list of the default variables of the dataset.

        Parameters
        ----------
        ds_name : str
            The name of the dataset.

        ds_name_ori : str
            The name of the original dataset the associated model was designed for.

        Returns
        ----------
        names : list
            List of feature names.
    """
    if ds_name == ds_name_ori:
        if ds_name == "femto":
            names = ["hori_acc", "vert_acc"]
            
        elif ds_name == "cmapss_fd001":
            names = ['T24','T30','T50','P30','Nf','Nc','Ps30','phi','NRf','NRe','BPR','htBleed','W31','W32']
    
        else:
            raise ValueError("Dataset name'",ds_name,"'not found.")
    else:
        if ds_name == "femto" and ds_name_ori == "cmapss_fd001":
            old_names = ["hori_acc","vert_acc"]
            f_idx = list(np.repeat(["f" + str(i) for i in range(16)],2))
            names = [old_names[i%2]+"_"+f_idx[i] for i in range(32)]
            
        elif ds_name == "cmapss_fd001" and ds_name_ori == "femto":
            names = ['T24','T30','T50','P30','Nf','Nc','Ps30','phi','NRf','NRe','BPR','htBleed','W31','W32']
            
        else:
            raise ValueError("Dataset name'",ds_name,"'not found.")
            
    return names

def random_truncation(df):
    MIN_TS_SIZE = 31
    SEGMENT_LEN = 2560
    
    # Converting validation data into incomplete-run-to-failure
    if self._dataset_name == "cmapss_fd001":
        df = df.groupby('unit').apply(lambda x: x.iloc[:np.random.randint(MIN_TS_SIZE, len(x)+1)]).reset_index(drop=True)
        
    elif self._dataset_name == "femto":
        df = df.groupby('unit').apply(lambda x: x.iloc[:SEGMENT_LEN*np.random.randint(MIN_TS_SIZE, (len(x)+1)//SEGMENT_LEN+1)]).reset_index(drop=True)

    return df