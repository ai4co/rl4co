import numpy as np

try:
    from .insertion import random_insertion
except ImportError:
    random_insertion = None


def eval_insertion(tsp_insts: np.ndarray) -> np.ndarray:
    # TODO: add instructions for downloading insertion support from GLOP
    assert random_insertion is not None
    results = [random_insertion(instance) for instance in tsp_insts]
    actions = np.array([x[0] for x in results])
    return actions


def eval_lkh(coordinates: np.ndarray) -> np.ndarray:
    # TODO
    raise NotImplementedError()
