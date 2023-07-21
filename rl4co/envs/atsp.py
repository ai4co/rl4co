from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ATSPEnv(RL4COEnvBase):
    """
    Asymmetric Traveling Salesman Problem environment
    At each step, the agent chooses a city to visit. The reward is the -infinite unless the agent visits all the cities.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.
    Unlike the TSP, the distance matrix is asymmetric, i.e., the distance from A to B is not necessarily the same as the distance from B to A.

    Args:
        num_loc: number of locations (cities) in the TSP
        td_params: parameters of the environment
        seed: seed for the environment
        device: device to use.  Generally, no need to set as tensors are updated on the fly
    """

    name = "atsp"

    def __init__(
        self,
        num_loc: int = 10,
        min_dist: float = 0,
        max_dist: float = 1,
        tmat_class: bool = True,
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.tmat_class = tmat_class
        self._make_spec(td_params)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"]
        first_node = current_node if batch_to_scalar(td["i"]) == 0 else td["first_node"]

        # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )

        # We are done there are no unvisited locations
        done = torch.count_nonzero(available, dim=-1) <= 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to -inf here
        reward = torch.ones_like(done) * float("-inf")

        # The output must be written in a ``"next"`` entry
        return TensorDict(
            {
                "next": {
                    "cost_matrix": td["cost_matrix"],
                    "first_node": first_node,
                    "current_node": current_node,
                    "i": td["i"] + 1,
                    "action_mask": available,
                    "reward": reward,
                    "done": done,
                }
            },
            td.shape,
        )

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize distance matrix
        cost_matrix = (
            td["cost_matrix"] if td is not None else None
        )  # dm = distance matrix
        if batch_size is None:
            batch_size = (
                self.batch_size if cost_matrix is None else cost_matrix.shape[:-2]
            )
        self.device = device = (
            cost_matrix.device if cost_matrix is not None else self.device
        )
        if cost_matrix is None:
            cost_matrix = self.generate_data(batch_size=batch_size).to(device)[
                "cost_matrix"
            ]

        # Other variables
        current_node = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        available = torch.ones(
            (*batch_size, self.num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "cost_matrix": cost_matrix,
                "first_node": current_node,
                "current_node": current_node,
                "i": i,
                "action_mask": available,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, td_params: TensorDict = None):
        self.observation_spec = CompositeSpec(
            cost_matrix=BoundedTensorSpec(
                minimum=self.min_dist,
                maximum=self.max_dist,
                shape=(self.num_loc, self.num_loc),
                dtype=torch.float32,
            ),
            first_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.num_loc),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.input_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            minimum=0,
            maximum=self.num_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    def get_reward(self, td, actions) -> TensorDict:
        distance_matrix = td["cost_matrix"]
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Get indexes of tour edges
        nodes_src = actions
        nodes_tgt = torch.roll(actions, 1, dims=1)
        batch_idx = torch.arange(
            distance_matrix.shape[0], device=distance_matrix.device
        ).unsqueeze(1)
        # return negative tour length
        return -distance_matrix[batch_idx, nodes_src, nodes_tgt].sum(-1)

    def generate_data(self, batch_size) -> TensorDict:
        # Generate distance matrices inspired by the reference MatNet (Kwon et al., 2021)
        # We satifsy the triangle inequality (TMAT class) in a batch
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        dms = (
            torch.rand((*batch_size, self.num_loc, self.num_loc), generator=self.rng)
            * (self.max_dist - self.min_dist)
            + self.min_dist
        )
        dms[..., torch.arange(self.num_loc), torch.arange(self.num_loc)] = 0
        log.info("Using TMAT class (triangle inequality): {}".format(self.tmat_class))
        if self.tmat_class:
            while True:
                old_dms = dms.clone()
                dms, _ = (
                    dms[..., :, None, :] + dms[..., None, :, :].transpose(-2, -1)
                ).min(dim=-1)
                if (dms == old_dms).all():
                    break
        return TensorDict({"cost_matrix": dms}, batch_size=batch_size)

    @staticmethod
    def render(td, actions=None, ax=None):
        try:
            import networkx as nx
        except ImportError:
            log.warn(
                "Networkx is not installed. Please install it with `pip install networkx`"
            )
            return

        td = td.detach().cpu()
        if actions is None:
            actions = td.get("action", None)

        # if batch_size greater than 0 , we need to select the first batch element
        if td.batch_size != torch.Size([]):
            td = td[0]
            actions = actions[0]

        src_nodes = actions
        tgt_nodes = torch.roll(actions, 1, dims=0)

        # Plot with networkx
        G = nx.DiGraph(td["cost_matrix"].numpy())
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="skyblue",
            node_size=800,
            edge_color="white",
        )

        # draw edges src_nodes -> tgt_nodes
        edgelist = [
            (src_nodes[i].item(), tgt_nodes[i].item()) for i in range(len(src_nodes))
        ]
        nx.draw_networkx_edges(
            G, pos, edgelist=edgelist, width=2, alpha=1, edge_color="black"
        )
