from typing import Union, Callable

import torch

from torch.distributions import Uniform
from tensordict.tensordict import TensorDict

from rl4co.utils.pylogger import get_pylogger
from rl4co.envs.common.utils import get_sampler, Generator

log = get_pylogger(__name__)


class PDPGenerator(Generator):
    """Data generator for the Pickup and Delivery Problem (PDP).
    Args:
        num_loc: number of locations (customers) in the PDP, without the depot. (e.g. 10 means 10 locs + 1 depot)
            - 1 depot
            - `num_loc` / 2 pickup locations
            - `num_loc` / 2 delivery locations
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        loc_distribution: distribution for the location coordinates
        depot_distribution: distribution for the depot location
    
    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc + 1, 2]: locations of each customer and the depot
            depot [batch_size, 2]: location of the depot
    """
    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[
            int, float, str, type, Callable
        ] = Uniform,
        depot_distribution: Union[
            int, float, str, type, Callable
        ] = Uniform,
        **kwargs
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc

        # Number of locations must be even
        if num_loc % 2 != 0:
            log.warn("Number of locations must be even. Adding 1 to the number of locations.")
            self.num_loc += 1

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler("loc", loc_distribution, min_loc, max_loc, **kwargs)

        # Depot distribution
        if kwargs.get("depot_sampler", None) is not None:
            self.depot_sampler = kwargs["depot_sampler"]
        else:
            self.depot_sampler = get_sampler("depot", depot_distribution, min_loc, max_loc, **kwargs)

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))

        # Sample depot
        depot = self.depot_sampler.sample((*batch_size, 2))

        return TensorDict(
            {
                "locs": torch.cat((depot[:, None, :], locs), dim=1),
                "depot": depot,
            },
            batch_size=batch_size,
        )
