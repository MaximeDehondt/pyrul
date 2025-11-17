# various.py
# Various utility functions.

from tensorflow import is_tensor
import numpy as np

def tensor_to_numpy(x)->np.array:
    """
        Converts the object to numpy if it is a tensor.
    """
    if is_tensor(x):
        return x.numpy()
    else:
        return x