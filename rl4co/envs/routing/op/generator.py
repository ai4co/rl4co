from typing import Union, Callable

import torch

from torch.distributions import Uniform
from tensordict.tensordict import TensorDict

from rl4co.utils.pylogger import get_pylogger
from rl4co.envs.common.utils import get_sampler, Generator

log = get_pylogger(__name__)

# From Kool et al. 2019
MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0}


class OPGenerator(Generator):
    """Data generator for the Orienteering Problem (OP).
    Args:
        num_loc: number of locations (cities) in the OP, without the depot. (e.g. 10 means 10 locs + 1 depot)
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        loc_distribution: distribution for the location coordinates
        min_prize: minimum value for the prize of each city
        max_prize: maximum value for the prize of each city
        prize_distribution: distribution for the prize of each city
        max_length: maximum length of the path

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc + 1, 2]: locations of each city and the depot
            depot [batch_size, 2]: location of the depot
            prize [batch_size, num_loc + 1]: prize of each city and the depot, 
                while the prize of the depot is 0
            max_length [batch_size, 1]: maximum length of the path for each city
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
        min_prize: float = 1.0,
        max_prize: float = 1.0,
        prize_distribution: Union[
            int, float, type, Callable
        ] = Uniform,
        max_length: float = None,
        **kwargs
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_prize = min_prize
        self.max_prize = max_prize
        self.max_length = max_length

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

        # Prize distribution
        if kwargs.get("prize_sampler", None) is not None:
            self.prize_sampler = kwargs["prize_sampler"]
        elif prize_distribution == 'dist': # If prize_distribution is 'dist', then the prize is the distance from the depot
            self.prize_sampler = None
        else:
            self.prize_sampler = get_sampler("prize", prize_distribution, min_prize, max_prize, **kwargs)

        # Max length
        if max_length is not None:
            self.max_length = max_length
        else:
            self.max_length = MAX_LENGTHS.get(num_loc, None)
        if self.max_length is None:
            cloest_num_loc = min(MAX_LENGTHS.keys(), key=lambda x: abs(x - num_loc))
            self.max_length = MAX_LENGTHS[cloest_num_loc]
            log.warning(
                f"The max length for {num_loc} locations is not defined. Using the closest max length: {self.max_length}\
                    with {cloest_num_loc} locations."
            )

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))

        # Sample depot
        depot = self.depot_sampler.sample((*batch_size, 2))

        # Sample prizes
        if self.prize_sampler is None:
            prize = torch.norm(locs - depot[:, None, :], dim=-1)
        else:
            prize = self.prize_sampler.sample((*batch_size, self.num_loc))
        prize = torch.cat((torch.zeros_like(prize[:, :1]), prize), dim=-1) # Add depot prize with 0 as a placeholder

        # Init the max length
        max_length = torch.full((*batch_size, 1), self.max_length, dtype=torch.float32)
        max_length = max_length - torch.norm(torch.cat((depot[:, None, :], locs), dim=1) - depot[:, None, :], dim=-1) - 1e-5

        return TensorDict(
            {
                "locs": torch.cat((depot[:, None, :], locs), dim=1),
                "depot": depot,
                "prize": prize,
                "max_length": max_length,
            },
            batch_size=batch_size,
        )
