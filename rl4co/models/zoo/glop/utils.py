import numpy as np
import torch

try:
    from . import insertion
except ImportError:
    insertion = None


def tsp_eval_insertion(tsp_insts: torch.Tensor) -> torch.Tensor:
    # TODO: add instructions for downloading insertion support from GLOP
    assert insertion is not None
    tsp_insts_np = tsp_insts.numpy()
    results = insertion.tsp_random_insertion_parallel(tsp_insts_np)
    actions = torch.from_numpy(results)
    return actions


def shpp_eval_insertion(shpp_insts: torch.Tensor) -> torch.Tensor:
    # TODO: add instructions for downloading insertion support from GLOP
    assert insertion is not None
    shpp_insts_np = shpp_insts.numpy()
    results = insertion.shpp_random_insertion_parallel(shpp_insts_np)
    actions = torch.from_numpy(results.astype(np.int64))
    return actions


def eval_lkh(coordinates: torch.Tensor) -> torch.Tensor:
    # TODO
    raise NotImplementedError()
