from typing import Union, Callable

import torch

from torch.distributions import Uniform
from tensordict.tensordict import TensorDict

from rl4co.utils.pylogger import get_pylogger
from rl4co.envs.common.utils import get_sampler, Generator

log = get_pylogger(__name__)


class SVRPGenerator(Generator):
    """Data generator for the Skill Vehicle Routing Problem (SVRP).
    Args:
        num_loc: number of locations (cities) in the TSP
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        loc_distribution: distribution for the location coordinates
        min_skill: minimum value for the technic skill
        max_skill: maximum value for the technic skill
        skill_distribution: distribution for the technic skill
        tech_costs: list of the technic costs
    
    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each city
            depot [batch_size, 2]: location of the depot
            techs [batch_size, num_loc+1]: technic requirements of each city and the depot
            skills [batch_size, num_loc+1]: skills of the vehicles
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
        min_skill: float = 1.0,
        max_skill: float = 10.0,
        skill_distribution: Union[
            int, float, type, Callable
        ] = Uniform,
        tech_costs: list = [1, 2, 3],
        **kwargs
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_skill = min_skill
        self.max_skill = max_skill
        self.num_tech = len(tech_costs)
        self.tech_costs = torch.tensor(tech_costs)

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

        # Skill distribution
        if kwargs.get("skill_sampler", None) is not None:
            self.skill_sampler = kwargs["skill_sampler"]
        else:
            self.skill_sampler = get_sampler("skill", skill_distribution, min_skill, max_skill, **kwargs)

        # Uniform distribution for the scaling for skill
        self.uniform_sampler = get_sampler("", "uniform", 0.0, 1.0, **kwargs)

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))

        # Sample depot
        depot = self.depot_sampler.sample((*batch_size, 2))
        
        # Sample technic requirements
        techs, _ = torch.sort(self.skill_sampler.sample((*batch_size, self.num_tech)), dim=-1)
        techs[:, 0] = 0.0 # Technic of the depot is 0 as a placeholder

        # Sample skills, make sure there exist at least one vehicle with the highest tech level skill
        skills = self.uniform_sampler.sample((*batch_size, self.num_loc+1))
        skills[:, 0] = 0.0 # Skill of the depot is 0 as a placeholder
        skills = torch.max(techs, dim=1, keepdim=True).values * skills

        return TensorDict(
            {
                "locs": torch.cat((depot[:, None, :], locs), dim=1),
                "depot": depot,
                "techs": techs,
                "skills": skills,
            },
            batch_size=batch_size,
        )
