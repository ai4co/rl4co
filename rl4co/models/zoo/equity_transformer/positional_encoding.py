import torch

from torch import nn


class PositionalEncoding(nn.Module):
    """Compute sinusoid encoding.
    Reference: https://arxiv.org/abs/2306.02689

    Warning:
        This implementation is under development and subject to change.

    Args:
        d_model: Dimension of model.
        max_len: Max sequence length.
    """

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        # Initialize encoding matrix
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # no need to compute gradient

        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        _2i = torch.arange(0, d_model, step=2).float()

        # Compute the positional encodings
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, seq_len):
        # Return encoding matrix for the current sequence length
        return self.encoding[:seq_len, :]
