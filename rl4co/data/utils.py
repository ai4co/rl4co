import os

import numpy as np

from tensordict.tensordict import TensorDict


def load_npz_to_tensordict(filename):
    """Load a npz file directly into a TensorDict
    We assume that the npz file contains a dictionary of numpy arrays
    This is at least an order of magnitude faster than pickle
    """
    x = np.load(filename)
    x_dict = dict(x)
    batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
    return TensorDict(x_dict, batch_size=batch_size)


def check_extension(filename, extension=".npz"):
    """Check that filename has extension, otherwise add it"""
    if os.path.splitext(filename)[1] != extension:
        return filename + extension
    return filename
