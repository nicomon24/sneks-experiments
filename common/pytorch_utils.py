"""
    Misc PyTorch utils
"""

import numpy as np

def NCHW_from_NHWC(x):
    x = np.array(x)
    if len(x.shape) == 4:
        return np.transpose(x, (0, 3, 1, 2))
    elif len(x.shape) == 3:
        return np.transpose(x, (2, 0, 1))
    else:
        raise Exception("Unrecognized shape.")

def NHWC_from_NCHW(x):
    x = np.array(x)
    if len(x.shape) == 4:
        return np.transpose(x, (0, 2, 3, 1))
    elif len(x.shape) == 3:
        return np.transpose(x, (1, 2, 0))
    else:
        raise Exception("Unrecognized shape.")
