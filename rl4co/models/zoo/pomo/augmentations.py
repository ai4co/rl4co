import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict

from rl4co.utils.ops import batchify


def augment_xy_data_by_8_fold(xy):
    """
    Augmentation for POMO for grid-based data (x, y)
    This is a Dihedral group of order 8 (rotations and reflections)
    https://en.wikipedia.org/wiki/Examples_of_groups#dihedral_group_of_order_8
    """
    # [batch, graph, 2]
    x, y = xy.split(1, dim=2)
    # augmnetations [batch, graph, 2]
    z0 = torch.cat((x, y), dim=2)
    z1 = torch.cat((1 - x, y), dim=2)
    z2 = torch.cat((x, 1 - y), dim=2)
    z3 = torch.cat((1 - x, 1 - y), dim=2)
    z4 = torch.cat((y, x), dim=2)
    z5 = torch.cat((1 - y, x), dim=2)
    z6 = torch.cat((y, 1 - x), dim=2)
    z7 = torch.cat((1 - y, 1 - x), dim=2)
    # [batch*8, graph, 2]
    aug_xy = torch.cat((z0, z1, z2, z3, z4, z5, z6, z7), dim=0)
    return aug_xy


def env_aug_feats(env_name: str):
    return ("locs", "depot") if env_name == "op" else ("locs",)


class StateAugmentation(nn.Module):
    def __init__(self, env_name, num_augment: int = 8):
        """Augment state by 8 fold for POMO"""
        super(StateAugmentation, self).__init__()
        self.num_augment = num_augment
        assert num_augment == 8, "Only 8 fold augmentation is supported for POMO"
        self.augmentation = augment_xy_data_by_8_fold
        self.feats = env_aug_feats(env_name)

    def forward(self, td: TensorDict, **unused_kwargs) -> TensorDict:
        td_aug = batchify(td, self.num_augment)
        for feat in self.feats:
            aug_feat = self.augmentation(td[feat])
            td_aug[feat] = aug_feat
        return td_aug
