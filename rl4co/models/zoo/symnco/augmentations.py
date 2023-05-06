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
    return ("observation", "depot") if env_name == "op" else ("observation",)


class StateAugmentation(nn.Module):
    def __init__(self, env_name: str):
        """Augment state by N times via symmetric rotation/reflection transform"""
        super(StateAugmentation, self).__init__()
        self.augmentation = augment_xy_data_by_n_fold
        self.feats = env_aug_feats(env_name)

    def forward(self, td: TensorDict, num_augment: int = 8) -> TensorDict:
        td_aug = batchify(td, num_augment)
        for feat in self.feats:
            aug_feat = self.augmentation(td_aug[feat], num_augment)
            td_aug[feat] = aug_feat
        return td_aug
