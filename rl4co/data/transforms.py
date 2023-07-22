import math

import torch

from tensordict.tensordict import TensorDict
from torch import Tensor

from rl4co.utils.ops import batchify


def dihedral_8_augmentation(xy: Tensor) -> Tensor:
    """
    Augmentation (x8) for grid-based data (x, y) as done in POMO.
    This is a Dihedral group of order 8 (rotations and reflections)
    https://en.wikipedia.org/wiki/Examples_of_groups#dihedral_group_of_order_8

    Args:
        xy: [batch, graph, 2] tensor of x and y coordinates
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


def symmetric_transform(x: Tensor, y: Tensor, phi: Tensor, offset: float = 0.5):
    """SR group transform with rotation and reflection
    Like the one in SymNCO, but a vectorized version

    Args:
        x: [batch, graph, 1] tensor of x coordinates
        y: [batch, graph, 1] tensor of y coordinates
        phi: [batch, 1] tensor of random rotation angles
        offset: offset for x and y coordinates
    """
    x, y = x - offset, y - offset
    # random rotation
    x_prime = torch.cos(phi) * x - torch.sin(phi) * y
    y_prime = torch.sin(phi) * x + torch.cos(phi) * y
    # make random reflection if phi > 2*pi (i.e. 50% of the time)
    mask = phi > 2 * math.pi
    # vectorized random reflection: swap axes x and y if mask
    xy = torch.cat((x_prime, y_prime), dim=-1)
    xy = torch.where(mask, xy.flip(-1), xy)
    return xy + offset


def symmetric_augmentation(xy: Tensor, num_augment: int = 8):
    """Augment xy data by `num_augment` times via symmetric rotation transform and concatenate to original data

    Args:
        xy: [batch, graph, 2] tensor of x and y coordinates
        num_augment: number of augmentations
    """
    # create random rotation angles (4*pi for reflection, 2*pi for rotation)
    phi = torch.rand(xy.shape[0], device=xy.device) * 4 * math.pi
    # set phi to 0 for first , i.e. no augmnetation as in original paper
    phi[: xy.shape[0] // num_augment] = 0.0
    x, y = xy[..., [0]], xy[..., [1]]
    return symmetric_transform(x, y, phi[:, None, None])


def env_aug_feats(env_name: str = None):
    """What features to augment for a given environment
    Usually, locs already includes depot, so we don't need to augment depot
    """
    return ("locs",)


def min_max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())


class StateAugmentation(object):
    """Augment state by N times via symmetric rotation/reflection transform

    Args:
        env_name: environment name
        num_augment: number of augmentations
        use_dihedral_8: whether to use dihedral_8_augmentation.  If True, then num_augment must be 8
        normalize: whether to normalize the augmented data
    """

    def __init__(
        self,
        env_name: str = None,
        num_augment: int = 8,
        use_dihedral_8: bool = False,
        normalize: bool = False,
    ):
        assert not (
            use_dihedral_8 and num_augment != 8
        ), "If use_dihedral_8 is True, then num_augment must be 8"
        if use_dihedral_8:
            self.augmentation = dihedral_8_augmentation
        else:
            self.augmentation = symmetric_augmentation

        self.feats = env_aug_feats(env_name)
        self.num_augment = num_augment
        self.normalize = normalize

    def __call__(self, td: TensorDict) -> TensorDict:
        td_aug = batchify(td, self.num_augment)
        for feat in self.feats:
            aug_feat = self.augmentation(td_aug[feat], self.num_augment)
            td_aug[feat] = aug_feat
            if self.normalize:
                td_aug[feat] = min_max_normalize(td_aug[feat])

        return td_aug
