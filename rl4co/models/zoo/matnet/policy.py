import torch.nn as nn

from rl4co.models.zoo.common.autoregressive import AutoregressivePolicy
from rl4co.models.zoo.matnet.decoder import MatNetFFSPDecoder
from rl4co.models.zoo.matnet.encoder import MatNetEncoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MultiStageMatNetPolicy(nn.Module):
    ...


#     """implements a MatNetPolicy for each stage of the FFSP problem instance as
#     described by Kwon et al., 2021
#     WARNING: This does not work. We need to implement the model selection logic within the autoregressive loop of the decoder.
#     This will be tricky
#     """

#     def __init__(
#             self,
#             env_name,
#             stage_cnt,
#             train_decode_type: str = "sampling",
#             val_decode_type: str = "greedy",
#             test_decode_type: str = "greedy",
#             **model_params):

#         if env_name != "ffsp":
#             log.error(f"env_name {env_name} is not implemented for multi-stage MatNet ")

#         super().__init__()
#         self.stage_cnt = stage_cnt

#         self.stage_models = nn.ModuleList(
#             [MatNetPolicy(env_name, **model_params) for _ in range(self.stage_cnt)]
#         )

#         self.train_decode_type = train_decode_type
#         self.val_decode_type = val_decode_type
#         self.test_decode_type = test_decode_type

#     def forward(
#         self,
#         td: TensorDict,
#         env=None,
#         return_actions: bool = False,
#         return_entropy: bool = False,
#         return_init_embeds: bool = False,
#         **policy_kwargs,
#     ) -> dict:
#         # TODO: actions might have different shapes for different stage
#         assert not return_actions, "returning actions currently not supported for multi-stage MatNet"
#         assert not return_entropy, "returning actions currently not supported for multi-stage MatNet"
#         assert not return_init_embeds, "returning actions currently not supported for multi-stage MatNet"

#         num_starts = policy_kwargs.get("num_starts", 1)
#         bs = td.size(0)

#         ret_td = TensorDict({
#             "reward": torch.empty((bs*num_starts,)),
#             "log_likelihood": torch.empty((bs*num_starts,), dtype=torch.bfloat16),
#         }, batch_size=bs*num_starts)


#         for stage_idx in range(self.stage_cnt):

#             batch_idx = td["stage_idx"] == stage_idx
#             batch_pomo_idx = batch_idx.repeat(num_starts)
#             if not batch_idx.any():
#                 continue
#             stage_model = self.stage_models[stage_idx]
#             stage_out = stage_model(td[batch_idx], env, **policy_kwargs)
#             ret_td.update_at_(stage_out, idx=batch_pomo_idx)


#         return ret_td.to_dict()

#     def evaluate_action(
#         self,
#         td: TensorDict,
#         action: Tensor,
#         env = None,
#     ) -> Tuple[Tensor, Tensor]:
#         raise NotImplementedError("Multi-stage MatNet only supports REINFORCE and variants at the moment")


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
        env_name: str,
        embedding_dim: int = 256,
        num_encoder_layers: int = 5,
        num_heads: int = 16,
        normalization: str = "instance",
        init_embedding_kwargs: dict = {"mode": "RandomOneHot"},
        use_graph_context: bool = False,
        **kwargs,
    ):
        if env_name not in ["atsp", "ffsp"]:
            log.error(f"env_name {env_name} is not originally implemented in MatNet")

        super(MatNetPolicy, self).__init__(
            env_name=env_name,
            encoder=MatNetEncoder(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                normalization=normalization,
                init_embedding_kwargs=init_embedding_kwargs,
            ),
            decoder=MatNetFFSPDecoder(
                env_name=env_name,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                use_graph_context=use_graph_context,
            ),
            embedding_dim=embedding_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            **kwargs,
        )
