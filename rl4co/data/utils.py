import os

import numpy as np
import vrplib

from tensordict.tensordict import TensorDict

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(CURR_DIR))


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


def load_solomon_instance(name, path=None):
    """Load a solomon instance from a file"""
    if not path:
        path = "data/solomon/instances/"
        path = os.path.join(ROOT_PATH, path)
    if not os.path.isfile(f"{path}{name}.txt"):
        vrplib.download_instance(name=name, path=path)
    return vrplib.read_instance(path=f"{path}{name}.txt", instance_format="solomon")


def load_solomon_solution(name, path=None):
    if not path:
        path = "data/solomon/solutions/"
        path = os.path.join(ROOT_PATH, path)
    if not os.path.isfile(f"{path}{name}.sol"):
        vrplib.download_solution(name=name, path=path)
    return vrplib.read_solution(path=f"{path}{name}.sol")
