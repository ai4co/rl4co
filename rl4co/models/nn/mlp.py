from typing import List

import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_neurons: List[int] = [64, 32],
        hidden_act: str = "ReLU",
        out_act: str = "Identity",
        input_norm: str = "None",
        output_norm: str = "None",
    ):
        super(MLP, self).__init__()

        assert input_norm in ["Batch", "Layer", "None"]
        assert output_norm in ["Batch", "Layer", "None"]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self.lins = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            self.lins.append(nn.Linear(in_dim, out_dim))

        self.input_norm = self._get_norm_layer(input_norm, input_dim)
        self.output_norm = self._get_norm_layer(output_norm, output_dim)

    def forward(self, xs):
        xs = self.input_norm(xs)
        for i, lin in enumerate(self.lins[:-1]):
            xs = lin(xs)
            xs = self.hidden_act(xs)
        xs = self.lins[-1](xs)
        xs = self.out_act(xs)
        xs = self.output_norm(xs)
        return xs

    @staticmethod
    def _get_norm_layer(norm_method, dim):
        if norm_method == "Batch":
            in_norm = nn.BatchNorm1d(dim)
        elif norm_method == "Layer":
            in_norm = nn.LayerNorm(dim)
        elif norm_method == "None":
            in_norm = nn.Identity()  # kinda placeholder
        else:
            raise RuntimeError(
                "Not implemented normalization layer type {}".format(norm_method)
            )
        return in_norm

    def _get_act(self, is_last):
        return self.out_act if is_last else self.hidden_act
