import torch

from tensordict.tensordict import TensorDict

from rl4co.envs.common.utils import Generator
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def remove_repeat(x: torch.Tensor) -> torch.Tensor:
    """
    Remove the repeated elements in each row (i.e., the last dimension) of the input tensor x,
    and change the repeated elements to 0

    Ref: https://stackoverflow.com/questions/62300404

    Args:
        x: input tensor
    """

    # sorting the rows so that duplicate values appear together
    # e.g., first row: [1, 2, 3, 3, 3, 4, 4]
    y, indices = x.sort(dim=-1)

    # subtracting, so duplicate values will become 0
    # e.g., first row: [1, 2, 3, 0, 0, 4, 0]
    y[..., 1:] *= ((y[..., 1:] - y[..., :-1]) != 0).long()

    # retrieving the original indices of elements
    indices = indices.sort(dim=-1)[1]

    # re-organizing the rows following original order
    # e.g., first row: [1, 2, 3, 4, 0, 0, 0]
    return torch.gather(y, -1, indices)


class MCPGenerator(Generator):
    """Data generator for the Maximum Coverage Problem (MCP).

    Args:
        num_items: number of items in the MCP
        num_sets: number of sets in the MCP
        min_weight: minimum value for the item weights
        max_weight: maximum value for the item weights
        min_size: minimum size for the sets
        max_size: maximum size for the sets
        n_sets_to_choose: number of sets to choose in the MCP

    Returns:
        A TensorDict with the following keys:
            membership [batch_size, num_sets, max_size]: membership of items in sets
            weights [batch_size, num_items]: weights of the items
            n_sets_to_choose [batch_size, 1]: number of sets to choose in the MCP
    """

    def __init__(
        self,
        num_items: int = 200,
        num_sets: int = 100,
        min_weight: int = 1,
        max_weight: int = 10,
        min_size: int = 5,
        max_size: int = 15,
        n_sets_to_choose: int = 10,
        **kwargs,
    ):
        self.num_items = num_items
        self.num_sets = num_sets
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_size = min_size
        self.max_size = max_size
        self.n_sets_to_choose = n_sets_to_choose

    def _generate(self, batch_size) -> TensorDict:
        try:
            batch_size = batch_size[0]
        except TypeError:
            batch_size = batch_size

        weights_tensor = torch.randint(
            self.min_weight, self.max_weight + 1, (batch_size, self.num_items)
        ).float()
        # Create membership tensor
        membership_tensor_max_size = torch.randint(
            1, self.num_items + 1, (batch_size, self.num_sets, self.max_size)
        )
        # Cutoffs for each set
        cutoffs = torch.randint(
            self.min_size, self.max_size + 1, (batch_size, self.num_sets)
        )

        cutoffs_masks = torch.arange(self.max_size).view(1, 1, -1) < cutoffs.unsqueeze(-1)
        # Take the masked elements, 0 means the item is invalid
        membership_tensor = (
            membership_tensor_max_size * cutoffs_masks
        )  # (batch_size, num_sets, max_size)

        # Remove repeated items in each set
        membership_tensor = remove_repeat(membership_tensor)

        return TensorDict(
            {
                "membership": membership_tensor.float(),  # (batch_size, num_sets, max_size)
                "weights": weights_tensor.float(),  # (batch_size, num_items)
                "n_sets_to_choose": torch.ones(batch_size, 1)
                * self.n_sets_to_choose,  # (batch_size, 1)
            },
            batch_size=batch_size,
        )
