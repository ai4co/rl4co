from typing import Optional, Type, Union

import torch.nn as nn

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.common.constructive.nonautoregressive.policy import (
    NonAutoregressivePolicy,
)
from rl4co.models.zoo.deepaco.antsystem import AntSystem
from rl4co.utils.utils import merge_with_defaults


class DeepACOPolicy(NonAutoregressivePolicy):
    """Implememts DeepACO policy based on :class:`NonAutoregressivePolicy`.

    Args:
        # TODO
        env_name: Name of the environment used to initialize embeddings
        encoder: Encoder module. Can be passed by sub-classes
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        edge_embedding: Model to use for the edge embedding. If None, use the default embedding for the environment
        heatmap_generator: Model to use for converting the edge embeddings to the heuristic information.
            If None, use the default MLP defined in :class:`~rl4co.models.common.nonautoregressive.decoder.EdgeHeatmapGenerator`.
        embed_dim: Dimension of the embeddings
        num_encoder_layers: Number of layers in the encoder
        num_decoder_layers: Number of layers in the decoder
        **decoder_kwargs: Additional arguments to be passed to the DeepACO decoder.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        env_name: Union[str, RL4COEnvBase] = "tsp",
        aco_class: Optional[Type[AntSystem]] = None,
        aco_kwargs: dict = {},
        n_ants: Optional[Union[int, dict]] = None,
        n_iterations: Optional[Union[int, dict]] = None,
        **encoder_kwargs,
    ):
        super(DeepACOPolicy, self).__init__(
            encoder=encoder,
            env_name=env_name,
            train_decode_type="multistart_sampling",
            val_decode_type="multistart_sampling",
            test_decode_type="multistart_sampling",
            **encoder_kwargs,
        )

        self.aco_class = AntSystem if aco_class is None else aco_class
        self.aco_kwargs = aco_kwargs
        self.n_ants = merge_with_defaults(n_ants, train=20, val=20, test=20)
        self.n_iterations = merge_with_defaults(n_iterations, train=1, val=20, test=100)

    def forward(
        self,
        td_initial: TensorDict,
        env: Union[str, RL4COEnvBase, None] = None,
        calc_reward: bool = True,
        phase: str = "train",
        actions=None,
        return_actions: bool = False,
        **kwargs,
    ):
        if phase == "train":
            #  we just use the constructive policy
            return super().forward(
                td_initial,
                env,
                phase=phase,
                decode_type="multistart_sampling",
                calc_reward=calc_reward,
                num_starts=self.n_ants[phase],
                actions=actions,
                return_actions=return_actions,
                **kwargs,
            )

        # Instantiate environment if needed
        if env is None or isinstance(env, str):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        heatmap_logits, _ = self.encoder(td_initial)

        aco = self.aco_class(heatmap_logits, n_ants=self.n_ants[phase], **self.aco_kwargs)
        td, actions, reward = aco.run(td_initial, env, self.n_iterations[phase])

        if calc_reward:
            td.set("reward", reward)

        out = {"reward": td["reward"]}

        if return_actions:
            out["actions"] = actions

        return out
