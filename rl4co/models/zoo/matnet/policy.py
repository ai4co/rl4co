from math import factorial
from typing import List, Union

import torch
import torch.nn as nn

from tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.scheduling.ffsp import FFSPEnv
from rl4co.models.zoo.common.autoregressive import AutoregressivePolicy
from rl4co.models.zoo.matnet.decoder import (
    MatNetDecoder,
    MatNetFFSPDecoder,
    MultiStageFFSPDecoder,
)
from rl4co.models.zoo.matnet.encoder import MatNetEncoder
from rl4co.utils.ops import batchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MatNetPolicy(AutoregressivePolicy):
    """MatNet Policy from Kwon et al., 2021.
    Reference: https://arxiv.org/abs/2106.11113

    Warning:
        This implementation is under development and subject to change.

    Args:
        env_name: Name of the environment used to initialize embeddings
        embedding_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        **kwargs: keyword arguments passed to the `AutoregressivePolicy`

    Default paarameters are adopted from the original implementation.
    """

    def __init__(
        self,
        env_name: Union[str, RL4COEnvBase],
        embedding_dim: int = 256,
        num_encoder_layers: int = 5,
        num_heads: int = 16,
        normalization: str = "instance",
        init_embedding_kwargs: dict = {"mode": "RandomOneHot"},
        use_graph_context: bool = False,
        bias: bool = False,
        **kwargs,
    ):
        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name

        if env_name not in ["atsp", "ffsp"]:
            log.error(f"env_name {env_name} is not originally implemented in MatNet")

        if env_name == "ffsp":
            decoder = MatNetFFSPDecoder(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                use_graph_context=use_graph_context,
                out_bias=True,
            )

        else:
            decoder = MatNetDecoder(
                env_name=env_name,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                use_graph_context=use_graph_context,
            )

        super(MatNetPolicy, self).__init__(
            env_name=env_name,
            encoder=MatNetEncoder(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                normalization=normalization,
                init_embedding_kwargs=init_embedding_kwargs,
                bias=bias,
            ),
            decoder=decoder,
            embedding_dim=embedding_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            **kwargs,
        )


class MultiStageFFSPPolicy(nn.Module):
    """Policy for solving the FFSP using a seperate encoder and decoder for each
    stage. This requires the 'while not td["done"].all()'-loop to be on policy level
    (instead of decoder level)."""

    def __init__(
        self,
        stage_cnt: int,
        embedding_dim: int = 256,
        num_heads: int = 16,
        num_encoder_layers: int = 3,
        use_graph_context: bool = False,
        normalization: str = "instance",
        feed_forward_hidden: int = 512,
        bias: bool = False,
        train_decode_type: str = "sampling",
        val_decode_type: str = "sampling",  # authors report better results for sampling
        test_decode_type: str = "sampling",
    ):
        super().__init__()
        self.stage_cnt = stage_cnt

        self.encoders: List[MatNetEncoder] = nn.ModuleList(
            [
                MatNetEncoder(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    num_layers=num_encoder_layers,
                    normalization=normalization,
                    feed_forward_hidden=feed_forward_hidden,
                    bias=bias,
                    init_embedding_kwargs={"mode": "RandomOneHot"},
                )
                for _ in range(self.stage_cnt)
            ]
        )
        self.decoders: List[MultiStageFFSPDecoder] = nn.ModuleList(
            [
                MultiStageFFSPDecoder(embedding_dim, num_heads, use_graph_context)
                for _ in range(self.stage_cnt)
            ]
        )

        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def pre_forward(self, td: TensorDict, env: FFSPEnv, num_starts: int):
        for stage_idx in range(self.stage_cnt):
            td["cost_matrix"] = td["run_time"][:, :, :, stage_idx]
            encoder = self.encoders[stage_idx]
            embeddings, _ = encoder(td)
            decoder = self.decoders[stage_idx]
            decoder._precompute_cache(embeddings, td)

        if num_starts > 1:
            # repeat num_start times
            td = batchify(td, num_starts)
            # update machine idx and action mask
            td = env.pre_step(td)

        return td

    def forward(
        self,
        td: TensorDict,
        env: FFSPEnv,
        phase="train",
        num_starts=1,
        return_actions: bool = False,
        **decoder_kwargs,
    ):
        assert not env.flatten_stages, "Multistage model only supports unflattened env"
        assert num_starts <= factorial(env.num_machine)

        # Get decode type depending on phase
        decode_type = getattr(self, f"{phase}_decode_type")
        device = td.device

        td = self.pre_forward(td, env, num_starts)

        # NOTE: this must come after pre_forward due to batchify op
        batch_size = td.size(0)
        prob_list = torch.zeros(size=(batch_size, 0), device=device)
        action_list = []

        while not td["done"].all():
            action_stack = torch.empty(
                size=(batch_size, self.stage_cnt), dtype=torch.long, device=device
            )
            prob_stack = torch.empty(size=(batch_size, self.stage_cnt), device=device)

            for stage_idx in range(self.stage_cnt):
                decoder = self.decoders[stage_idx]
                action, prob = decoder(td, decode_type, num_starts, **decoder_kwargs)
                action_stack[:, stage_idx] = action
                prob_stack[:, stage_idx] = prob

            gathering_index = td["stage_idx"][:, None]
            # shape: (batch, 1)
            action = action_stack.gather(dim=1, index=gathering_index).squeeze(dim=1)
            prob = prob_stack.gather(dim=1, index=gathering_index).squeeze(dim=1)
            # shape: (batch)
            action_list.append(action)
            # transition
            td.set("action", action)
            td = env.step(td)["next"]

            prob_list = torch.cat((prob_list, prob[:, None]), dim=1)

        out = {
            "reward": td["reward"],
            "log_likelihood": prob_list.log().sum(1),
        }

        if return_actions:
            out["actions"] = torch.stack(action_list, 1)

        return out
