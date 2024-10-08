import numpy as np
import torch

try:
    from .insertion import random_insertion
except ImportError:
    random_insertion = None


def eval_insertion(tsp_insts: torch.Tensor) -> torch.Tensor:
    # TODO: add instructions for downloading insertion support from GLOP
    assert random_insertion is not None
    tsp_insts_np = tsp_insts.numpy()
    results = [random_insertion(instance) for instance in tsp_insts_np]
    actions = torch.from_numpy(np.stack([x[0] for x in results]))
    return actions


def eval_lkh(coordinates: torch.Tensor) -> torch.Tensor:
    # TODO
    raise NotImplementedError()
