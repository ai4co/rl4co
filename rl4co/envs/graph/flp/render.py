import matplotlib.pyplot as plt
import numpy as np
import torch

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td, actions=None, ax=None):
    # TODO: better rendering, e.g., visualization 
    print(td)
    print(actions)