import math

import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict

from rl4co.utils.ops import batchify


def rotation_reflection_transform(x, y, phi, offset=0.5):
    """SR group transform with rotation and reflection (~2x faster than original)"""
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


def augment_xy_data_by_n_fold(xy, num_augment: int = 8):
    """Augment xy data by N times via symmetric rotation transform and concatenate to original data"""
    # create random rotation angles (4*pi for reflection, 2*pi for rotation)
    phi = torch.rand(xy.shape[0], device=xy.device) * 4 * math.pi
    # set phi to 0 for first , i.e. no augmnetation as in original paper
    phi[: xy.shape[0] // num_augment] = 0.0
    x, y = xy[..., [0]], xy[..., [1]]
    return rotation_reflection_transform(x, y, phi[:, None, None])


def env_aug_feats(env_name: str):
    return ("locs", "depot") if env_name == "op" else ("locs",)


def min_max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())


class StateAugmentation(nn.Module):
    """Augment state by N times via symmetric rotation/reflection transform"""

    def __init__(self, env_name: str, num_augment: int = 8, normalize: bool = False):
        super(StateAugmentation, self).__init__()
        self.augmentation = augment_xy_data_by_n_fold
        self.feats = env_aug_feats(env_name)
        self.num_augment = num_augment
        self.normalize = normalize

    def forward(
        self, td: TensorDict, num_augment: int = None, normalize: bool = False
    ) -> TensorDict:
        num_augment = num_augment if num_augment is not None else self.num_augment
        normalize = normalize if normalize is not None else False

        td_aug = batchify(td, num_augment)
        for feat in self.feats:
            aug_feat = self.augmentation(td_aug[feat], num_augment)
            td_aug[feat] = aug_feat
            if normalize:
                td_aug[feat] = min_max_normalize(td_aug[feat])

        return td_aug
