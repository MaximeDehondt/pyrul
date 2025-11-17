# dataset_info.py 
# Used by RULModels to get default information about datasets.

import numpy as np
import pandas as pd

def get_segment_len(ds_name: str)->int:
    """
        Returns the length of a segment depending on the chosen dataset.
    """
    if ds_name == "cmapss_fd001":
        return 1
    elif ds_name == "femto":
        return 2560
    else:
        raise ValueError("Dataset name '"+ds_name+"' not found.")
        
def get_min_ts_size(ds_name: str)->int:
    """
        Returns the minimum length of a trajectory depending on the chosen dataset.
    """
    if ds_name == "cmapss_fd001":
        return 31
    elif ds_name == "femto":
        return 229
    else:
        raise ValueError("Dataset name '"+ds_name+"' not found.")

def get_max_ts_size(ds_name: str)->int:
    """
        Returns the maximum length of a trajectory depending on the chosen dataset.
    """
    if ds_name == "cmapss_fd001":
        return 1000
    elif ds_name == "femto":
        return 3000
    else:
        raise ValueError("Dataset name '"+ds_name+"' not found.")

def get_default_folds(ds_name: str)->int:
    """
        Returns the number of folds for k-fold cross validation, depending on the chosen dataset.
    """
    if ds_name == "cmapss_fd001":
        return 10
    elif ds_name == "femto":
        return 6
    else:
        raise ValueError("Dataset name '"+ds_name+"' not found.")

def get_defaults_vars(ds_name: str, ds_name_ori: str)->list:
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
    if ds_name == ds_name_ori: # standard case
        if ds_name == "femto":
            names = ["hori_acc", "vert_acc"]
            
        elif ds_name == "cmapss_fd001":
            names = ['T24','T30','T50','P30','Nf','Nc','Ps30','phi','NRf','NRe','BPR','htBleed','W31','W32']
    
        else:
            raise ValueError("Dataset name '"+ds_name+"' not found.")
            
    else: # cross-compatibility case
        if ds_name == "femto" and ds_name_ori == "cmapss_fd001":
            old_names = ["hori_acc","vert_acc"]
            f_idx = list(np.repeat(["f" + str(i) for i in range(16)],2))
            names = [old_names[i%2]+"_"+f_idx[i] for i in range(32)]
            
        elif ds_name == "cmapss_fd001" and ds_name_ori == "femto":
            names = ['T24','T30','T50','P30','Nf','Nc','Ps30','phi','NRf','NRe','BPR','htBleed','W31','W32']
            
        else:
            raise ValueError("Datasets '"+ds_name+"' and '"+ds_name_ori+"' not handled.")
            
    return names