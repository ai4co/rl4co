import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class CNN(nn.Module):
    """Simple CNN network with 1D output dimensions (e.g. actions)
    Adapted from:
    https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/rl/common/networks.py
    """

    def __init__(self, input_shape, output_shape, hidden_size=128, **kwargs):
        """
        Args:
            input_shape: input dimensions (channels, height, width)
            output_shape: output dimensions (e.g. actions)
            hidden_size: size of hidden layers
        """
        super().__init__()
        # If the input is larger than 48x48, use larger conv layers for better parameter efficiency
        if (input_shape[1] or input_shape[2]) > 48:
            self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )
        conv_out_size = self._get_conv_out(input_shape)
        self.head = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_shape),
        )
        self.input_shape = input_shape

    def _get_conv_out(self, shape) -> int:
        """Calculates the output size of the last conv layer.
        Args:
            shape: input dimensions
        Returns:
            size of the conv output
        """
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))

    def forward(self, input_x) -> Tensor:
        """Forward pass through network.
        Args:
            x: input to network
        Returns:
            output of network
        """
        input_x = input_x.view((-1, *self.input_shape)).float()
        conv_out = self.conv(input_x).view(input_x.size()[0], -1)
        return self.head(conv_out)
