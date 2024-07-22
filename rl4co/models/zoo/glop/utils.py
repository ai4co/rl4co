import numpy as np

try:
    from GLOP.utils.insertion import random_insertion
except ImportError:
    random_insertion = None

def eval_insertion(tsp_insts):
    assert random_insertion is not None
    results = [random_insertion(instance) for instance in tsp_insts]
    actions = np.array([x[0] for x in results])
    costs = np.array([x[1] for x in results])
    return actions, costs
