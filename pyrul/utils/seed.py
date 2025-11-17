# seed.py
# To handle reproducibility.

import numpy as np
import random
import os
import tensorflow as tf

# Fixing seed (no deterministic behavior to avoid increasing model training time)
def set_seed(seed: int)->None:
    """
        Sets the os, python, rd and tensorflow seeds. The keras backend session is also cleared.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    tf.keras.backend.clear_session()

def get_seed(idx: int)->int:
    """
        Returns the seed depending on the given trial index (from 0 to 9). 
        If index==0, returns the seed used for hyperparameter tuning.
    """
    if idx>=0 and idx<=9:
        seeds = [7841329, 102457, 9407, 33718, 86180, 718777, 4541, 678714, 5207, 21346]
        return seeds[idx]
    else:
        raise ValueError("Seed index is invalid. Allowed values range from 0 to 9. Received value ="+str(idx)+".")