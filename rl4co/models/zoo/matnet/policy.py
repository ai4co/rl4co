import torch.nn as nn

from rl4co.models.zoo.common.autoregressive import AutoregressivePolicy
from rl4co.models.zoo.matnet.decoder import MatNetFFSPDecoder
from rl4co.models.zoo.matnet.encoder import MatNetEncoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MultiStageMatNetPolicy(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        stage_cnt = self.model_params["stage_cnt"]
        self.stage_models = nn.ModuleList(
            [MatNetPolicy(**model_params) for _ in range(stage_cnt)]
        )

    def pre_forward(self, reset_state):
        stage_cnt = self.model_params["stage_cnt"]
        for stage_idx in range(stage_cnt):
            problems = reset_state.problems_list[stage_idx]
            model = self.stage_models[stage_idx]
            model.pre_forward(problems)

    def forward(
        self,
        td,
        env=None,
        phase: str = "train",
        return_actions: bool = False,
        return_entropy: bool = False,
        return_init_embeds: bool = False,
        **decoder_kwargs,
    ) -> dict:
        bs, num_jobs, num_ma, _ = td["run_time"].shape
        stage_idx = td["stage_idx"]

        td["cost_matrix"] = td["run_time"].gather(
            3, stage_idx[:, None, None, None].expand(bs, num_jobs, num_ma, 1)
        )

        model = self.stage_models[stage_idx]
        td_out = model(td, env)

        return td_out


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
