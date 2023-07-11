from typing import Optional

import numpy as np
import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.dpp import DPPEnv
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MDPPEnv(DPPEnv):
    """Multiple decap placement problem (mDPP) environment
    This is a modified version of the DPP environment where we allow multiple probing ports
    The reward can be calculated as:
        - minmax: min of the max of the decap scores
        - meansum: mean of the sum of the decap scores
    The minmax is more challenging as it requires to find the best decap location for the worst case

    Args:
        num_probes_min: minimum number of probes
        num_probes_max: maximum number of probes
        reward_type: reward type, either minmax or meansum
        td_params: TensorDict parameters
    """

    name = "mdpp"

    def __init__(
        self,
        *,
        num_probes_min: int = 2,
        num_probes_max: int = 5,
        reward_type: str = "minmax",
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_probes_min = num_probes_min
        self.num_probes_max = num_probes_max
        assert reward_type in [
            "minmax",
            "meansum",
        ], "reward_type must be minmax or meansum"
        self.reward_type = reward_type
        self._make_spec(td_params)

    def _step(self, td: TensorDict) -> TensorDict:
        # Step function is the same as DPPEnv, only masking changes
        return super()._step(td)

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Reset function is the same as DPPEnv, only masking changes due to probes
        td_reset = super()._reset(td, batch_size=batch_size)

        # Action mask is 0 if both action_mask (e.g. keepout) and probe are 0
        action_mask = torch.logical_and(td_reset["action_mask"], ~td_reset["probe"])
        # Keepout regions are the inverse of action_mask
        td_reset.set_("keepout", ~td_reset["action_mask"])
        td_reset.set_("action_mask", action_mask)
        return td_reset

    def _make_spec(self, td_params):
        """Make the observation and action specs from the parameters"""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(self.size**2, 2),
                dtype=torch.float32,
            ),
            probe=UnboundedDiscreteTensorSpec(
                shape=(self.size**2),
                dtype=torch.bool,
            ),  # probe is a boolean of multiple locations (1=probe, 0=not probe)
            keepout=UnboundedDiscreteTensorSpec(
                shape=(self.size**2),
                dtype=torch.bool,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.size**2),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.input_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            minimum=0,
            maximum=self.size**2,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    def get_reward(self, td, actions):
        """We call the reward function with the final sequence of actions to get the reward
        Calling per-step would be very time consuming due to decap simulation
        """
        # We do the operation in a batch
        if len(td.batch_size) == 0:
            td = td.unsqueeze(0)
            actions = actions.unsqueeze(0)

        # Reward calculation is expensive since we need to run decap simulation (not vectorizable)
        reward = torch.stack(
            [
                self._single_env_reward(td_single, action)
                for td_single, action in zip(td, actions)
            ]
        )
        return reward

    def _single_env_reward(self, td, actions):
        """Get reward for single environment. We"""

        list_probe = torch.nonzero(td["probe"]).squeeze()
        scores = torch.zeros_like(list_probe, dtype=torch.float32)
        for i, probe in enumerate(list_probe):
            # Get the decap scores for the probe location
            scores[i] = self._decap_simulator(probe, actions)
        # If minmax, return min of max decap scores else mean
        return scores.min() if self.reward_type == "minmax" else scores.mean()

    def generate_data(self, batch_size):
        """
        Generate initial observations for the environment with locations, probe, and action mask
        Action_mask eliminates the keepout regions and the probe location, and is updated to eliminate placed decaps
        """

        m = n = self.size
        # if int, convert to list and make it a batch for easier generation
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        batched = len(batch_size) > 0
        bs = [1] if not batched else batch_size

        # Create a list of locs on a grid
        locs = torch.meshgrid(torch.arange(m), torch.arange(n))
        locs = torch.stack(locs, dim=-1).reshape(-1, 2)
        # normalize the locations by the number of rows and columns
        locs = locs / torch.tensor([m, n], dtype=torch.float)
        locs = locs[None].expand(*bs, -1, -1)

        # Create available mask
        available = torch.ones((*bs, m * n), dtype=torch.bool)

        # Sample probe location from m*n
        probe = torch.randint(m * n, size=(*bs, 1))
        available.scatter_(1, probe, False)

        # Sample probe locatins
        num_probe = torch.randint(
            self.num_probes_min,
            self.num_probes_max,
            size=(*bs, 1),
        )
        probe = [torch.randperm(m * n)[:p] for p in num_probe]
        probes = torch.zeros((*bs, m * n), dtype=torch.bool)
        for i, (a, p) in enumerate(zip(available, probe)):
            available[i] = a.scatter(0, p, False)
            probes[i] = probes[i].scatter(0, p, True)

        # Sample keepout locations from m*n except probe
        num_keepout = torch.randint(
            self.num_keepout_min,
            self.num_keepout_max,
            size=(*bs, 1),
        )
        keepouts = [torch.randperm(m * n)[:k] for k in num_keepout]
        for i, (a, k) in enumerate(zip(available, keepouts)):
            available[i] = a.scatter(0, k, False)

        return TensorDict(
            {
                "locs": locs if batched else locs.squeeze(0),
                "probe": probes if batched else probes.squeeze(0),
                "action_mask": available if batched else available.squeeze(0),
            },
            batch_size=batch_size,
        )

    def render(self, td, actions=None, ax=None, legend=True, settings=None):
        """Plot a grid of squares representing the environment.
        The keepout regions are the action_mask - decaps - probe
        """

        import matplotlib.pyplot as plt

        from matplotlib.lines import Line2D
        from matplotlib.patches import Annulus, Rectangle, RegularPolygon

        if settings is None:
            settings = {
                "available": {"color": "white", "label": "available"},
                "keepout": {"color": "grey", "label": "keepout"},
                "probe": {"color": "tab:red", "label": "probe"},
                "decap": {"color": "tab:blue", "label": "decap"},
            }

        def draw_capacitor(ax, x, y, color="black"):
            # Backgrund rectangle: same as color but with alpha=0.5
            ax.add_patch(Rectangle((x, y), 1, 1, color=color, alpha=0.5))

            # Create the plates of the capacitor
            plate_width, plate_height = (
                0.3,
                0.1,
            )  # Width and height switched to make vertical
            plate_gap = 0.2
            plate1 = Rectangle(
                (x + 0.5 - plate_width / 2, y + 0.5 - plate_height - plate_gap / 2),
                plate_width,
                plate_height,
                color=color,
            )
            plate2 = Rectangle(
                (x + 0.5 - plate_width / 2, y + 0.5 + plate_gap / 2),
                plate_width,
                plate_height,
                color=color,
            )

            # Add the plates to the axes
            ax.add_patch(plate1)
            ax.add_patch(plate2)

            # Add connection lines (wires)
            line_length = 0.2
            line1 = Line2D(
                [x + 0.5, x + 0.5],
                [
                    y + 0.5 - plate_height - plate_gap / 2 - line_length,
                    y + 0.5 - plate_height - plate_gap / 2,
                ],
                color=color,
            )
            line2 = Line2D(
                [x + 0.5, x + 0.5],
                [
                    y + 0.5 + plate_height + plate_gap / 2,
                    y + 0.5 + plate_height + plate_gap / 2 + line_length,
                ],
                color=color,
            )

            # Add the lines to the axes
            ax.add_line(line1)
            ax.add_line(line2)

        def draw_probe(ax, x, y, color="black"):
            # Backgrund rectangle: same as color but with alpha=0.5
            ax.add_patch(Rectangle((x, y), 1, 1, color=color, alpha=0.5))
            ax.add_patch(Annulus((x + 0.5, y + 0.5), (0.2, 0.2), 0.1, color=color))

        def draw_keepout(ax, x, y, color="black"):
            # Backgrund rectangle: same as color but with alpha=0.5
            ax.add_patch(Rectangle((x, y), 1, 1, color=color, alpha=0.5))
            ax.add_patch(
                RegularPolygon(
                    (x + 0.5, y + 0.5), numVertices=6, radius=0.45, color=color
                )
            )

        size = self.size
        td = td.detach().cpu()
        # if batch_size greater than 0 , we need to select the first batch element
        if td.batch_size != torch.Size([]):
            td = td[0]

        if actions is None:
            actions = td.get("action", None)

        # Transform actions from idx to one-hot
        decaps = torch.zeros(size**2)
        decaps.scatter_(0, actions, 1)
        decaps = decaps.reshape(size, size)

        keepout = ~td["action_mask"].reshape(size, size)
        probes = td["probe"].reshape(size, size)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6))

        grid = np.meshgrid(np.arange(0, size), np.arange(0, size))
        grid = np.stack(grid, axis=-1)

        xdim, ydim = grid.shape[0], grid.shape[1]
        # ax.imshow(np.zeros((xdim, ydim)), cmap="gray")

        ax.set_xlim(0, xdim)
        ax.set_ylim(0, ydim)

        for i in range(xdim):
            for j in range(ydim):
                x, y = grid[i, j, 0], grid[i, j, 1]

                if decaps[i, j] == 1:
                    draw_capacitor(ax, x, y, color=settings["decap"]["color"])
                elif probes[i, j] == 1:
                    draw_probe(ax, x, y, color=settings["probe"]["color"])
                elif keepout[i, j] == 1:
                    draw_keepout(ax, x, y, color=settings["keepout"]["color"])

        ax.grid(
            which="major", axis="both", linestyle="-", color="k", linewidth=1, alpha=0.5
        )
        # set 10 ticks
        ax.set_xticks(np.arange(0, xdim, 1))
        ax.set_yticks(np.arange(0, ydim, 1))

        # Invert y axis
        ax.invert_yaxis()

        # # Add legend
        if legend:
            colors = [settings[k]["color"] for k in settings.keys()]
            labels = [settings[k]["label"] for k in settings.keys()]
            handles = [
                plt.Rectangle(
                    (0, 0), 1, 1, color=c, edgecolor="k", linestyle="-", linewidth=1
                )
                for c in colors
            ]
            ax.legend(
                handles,
                [label for label in labels],
                ncol=len(colors),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.1),
            )
