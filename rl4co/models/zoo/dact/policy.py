import torch
import torch.nn as nn

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.common.improvement.base import ImprovementPolicy
from rl4co.models.zoo.dact.decoder import DACTDecoder
from rl4co.models.zoo.dact.encoder import DACTEncoder
from rl4co.utils.decoding import DecodingStrategy, get_decoding_strategy
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class DACTPolicy(ImprovementPolicy):
    """
    DACT Policy based on Ma et al. (2021)
    This model first encodes the input graph and current solution using a DACT encoder (:class:`DACTEncoder`)
    and then decodes the 2-opt action (:class:`DACTDecoder`)

    Args:
        embed_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        feedforward_hidden: Dimension of the hidden layer in the feedforward network
        env_name: Name of the environment used to initialize embeddings
        pos_type: Name of the used positional encoding method (CPE or APE)
        init_embedding: Module to use for the initialization of the embeddings
        pos_embedding: Module to use for the initialization of the positional embeddings
        temperature: Temperature for the softmax
        tanh_clipping: Tanh clipping value (see Bello et al., 2016)
        train_decode_type: Type of decoding to use during training
        val_decode_type: Type of decoding to use during validation
        test_decode_type: Type of decoding to use during testing
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_encoder_layers: int = 3,
        num_heads: int = 4,
        normalization: str = "layer",
        feedforward_hidden: int = 64,
        env_name: str = "tsp_kopt",
        pos_type: str = "CPE",
        init_embedding: nn.Module = None,
        pos_embedding: nn.Module = None,
        temperature: float = 1.0,
        tanh_clipping: float = 6.0,
        train_decode_type: str = "sampling",
        val_decode_type: str = "sampling",
        test_decode_type: str = "sampling",
    ):
        super(DACTPolicy, self).__init__()

        self.env_name = env_name

        # Encoder and decoder
        self.encoder = DACTEncoder(
            embed_dim=embed_dim,
            init_embedding=init_embedding,
            pos_embedding=pos_embedding,
            env_name=env_name,
            pos_type=pos_type,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
        )

        self.decoder = DACTDecoder(embed_dim=embed_dim, num_heads=num_heads)

        # Decoding strategies
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(
        self,
        td: TensorDict,
        env: str | RL4COEnvBase = None,
        phase: str = "train",
        return_actions: bool = True,
        return_embeds: bool = False,
        only_return_embed: bool = False,
        actions=None,
        **decoding_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            return_actions: Whether to return the actions
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # Encoder: get encoder output and initial embeddings from initial state
        NFE, PFE = self.encoder(td)
        h_featrues = torch.cat((NFE, PFE), -1)

        if only_return_embed:
            return {"embeds": h_featrues.detach()}

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)
        assert env.two_opt_mode, "DACT only support 2-opt"

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=True,
            improvement_method_mode=True,
            **decoding_kwargs,
        )

        # Perform the decoding
        batch_size, seq_length = td["rec_current"].size()
        logits = self.decoder(td, NFE, PFE).view(batch_size, -1)

        # Get mask
        mask = env.get_mask(td)
        if "action" in td.keys():
            mask[torch.arange(batch_size), td["action"][:, 0], td["action"][:, 1]] = False
            mask[torch.arange(batch_size), td["action"][:, 1], td["action"][:, 0]] = False
        mask = mask.view(batch_size, -1)

        # Get action and log-likelihood
        logprob, action_sampled = decode_strategy.step(
            logits,
            mask,
            action=(
                actions[:, 0] * seq_length + actions[:, 1]
                if actions is not None
                else None
            ),
        )
        action_sampled = action_sampled.unsqueeze(-1)
        if phase == "train":
            log_likelihood = logprob.gather(1, action_sampled)
        else:
            log_likelihood = torch.zeros(batch_size, device=td.device)

        ## return
        DACT_action = torch.cat(
            (
                action_sampled // seq_length,
                action_sampled % seq_length,
            ),
            -1,
        )

        outdict = {"log_likelihood": log_likelihood, "cost_bsf": td["cost_bsf"]}
        td.set("action", DACT_action)

        if return_embeds:
            outdict["embeds"] = h_featrues.detach()

        if return_actions:
            outdict["actions"] = DACT_action

        return outdict
