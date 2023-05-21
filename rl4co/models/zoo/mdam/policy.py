import sys; sys.path.append('.')
import torch
import torch.nn as nn

from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.envs import EnvBase

from rl4co.envs import TSPEnv
from rl4co.data.dataset import TensorDictCollate
from rl4co.models.nn.env_embedding import env_init_embedding
from rl4co.models.zoo.mdam.encoder import GraphAttentionEncoder
from rl4co.models.zoo.mdam.decoder import Decoder



class AttentionModelPolicy(nn.Module):
    def __init__(
        self,
        env: EnvBase,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        embedding_dim: int = 128,
        num_encode_layers: int = 3,
        num_heads: int = 8,
        num_paths: int = 5,
        eg_step_gap: int = 200,
        normalization: str = "batch",
        mask_inner: bool = True,
        mask_logits: bool = True,
        tanh_clipping: float = 10.0,
        shrink_size=None,
        force_flash_attn: bool = False,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw
    ):
        """
        Args:
            env: environment to solve
            encoder: encoder module
            decoder: decoder module
            embedding_dim: embedding dimension/hidden dimension
            num_encode_layers: number of layers in encoder
            num_heads: number of heads in multi-head attention
            num_paths: number of paths to sample (specific feature for MDAM)
            eg_step_gap: number of steps between each path sampling (specific feature for MDAM)
            normalization: normalization type
            mask_inner: whether to mask the inner product in attention
            mask_logits: whether to mask the logits in attention
            tanh_clipping: tanh clipping value
            shrink_size: shrink size for the decoder
            force_flash_attn: whether to force use flash attention
            train_decode_type: decode type for training
            val_decode_type: decode type for validation
            test_decode_type: decode type for testing
        """
        super(AttentionModelPolicy, self).__init__()
        if len(unused_kw) > 0: print(f"Unused kwargs: {unused_kw}")

        self.env = env
        self.init_embedding = env_init_embedding(
            self.env.name, {"embedding_dim": embedding_dim}
        )

        self.encoder = (
            GraphAttentionEncoder(
                num_heads=num_heads,
                embed_dim=embedding_dim,
                num_layers=num_encode_layers,
                normalization=normalization,
                force_flash_attn=force_flash_attn,
            )
            if encoder is None
            else encoder
        )

        self.decoder = (
            Decoder(
                env=env,
                embedding_dim=embedding_dim, 
                num_heads=num_heads, 
                num_paths=num_paths,
                mask_inner=mask_inner,
                mask_logits=mask_logits,
                eg_step_gap=eg_step_gap,
                tanh_clipping=tanh_clipping,
                force_flash_attn=force_flash_attn,
                shrink_size=shrink_size,
                train_decode_type=train_decode_type,
                val_decode_type=val_decode_type,
                test_decode_type=test_decode_type,
            )
            if decoder is None
            else decoder
        )

    def forward(
        self,
        td: TensorDict,
        phase: str = "train",
        return_actions: bool = False,
        **decoder_kwargs,
    ) -> TensorDict:
        embedding = self.init_embedding(td)
        encoded_inputs, _, attn, V, h_old = self.encoder(embedding)

        # Get decode type depending on phase
        if decoder_kwargs.get("decode_type", None) is None:
            decoder_kwargs["decode_type"] = getattr(self, f"{phase}_decode_type")

        reward, log_likelihood, kl_divergence, actions = self.decoder(td, encoded_inputs, attn, V, h_old, **decoder_kwargs)
        out = {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "kl_divergence": kl_divergence,
            "actions": actions if return_actions else None,
        }
        return out


# SECTION: for model testing
from rl4co.models.rl.reinforce.base import REINFORCE
from rl4co.models.rl.reinforce.baselines import RolloutBaseline, WarmupBaseline, ExponentialBaseline
class AttentionModel(REINFORCE):
    def __init__(self, env, policy=None, baseline=None):
        """
        Attention Model for neural combinatorial optimization based on REINFORCE
        Based on Wouter Kool et al. (2018) https://arxiv.org/abs/1803.08475
        Refactored from reference implementation: https://github.com/wouterkool/attention-learn-to-route

        Args:
            env: TorchRL Environment
            policy: Policy
            baseline: REINFORCE Baseline
        """
        super(AttentionModel, self).__init__(env, policy, baseline)
        self.env = env
        self.policy = AttentionModelPolicy(env) if policy is None else policy
        self.baseline = (
            WarmupBaseline(RolloutBaseline()) if baseline is None else baseline
        )

if __name__ == "__main__":
    # SECTION: load the environment with test data
    env = TSPEnv()

    dataset = env.dataset(batch_size=[10000])

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=TensorDictCollate(),
        drop_last=True,
    )

    td = next(iter(dataloader)).to("cuda")
    td = env.reset(td)

    # SECTION: test the policy
    policy = AttentionModelPolicy(
        env,
    ).to("cuda")
    out = policy(td, decode_type="sampling", return_actions=False)
    # print(out)

    # SECTION: test full model environment
    baseline = WarmupBaseline(RolloutBaseline())
    model = AttentionModel(
        env,
        policy,
        baseline=baseline,
    ).to("cuda")

    td = next(iter(dataloader)).to("cuda")
    td = env.reset(td)

    out = model(td, decode_type="sampling")
    # print(out)

    # SECTION: test the model convergence with pytorch lightning
    from rl4co.tasks.rl4co import RL4COLitModule
    from omegaconf import DictConfig

    config = DictConfig(
        {
            "data": {
                "train_size": 100000, # with 1 epochs, this is 1k samples
                "val_size": 10000, 
                "train_batch_size": 512,
                "val_batch_size": 1024,
            },
            "optimizer": {
                "_target_": "torch.optim.Adam",
                "lr": 1e-4,
                "weight_decay": 1e-5,
            },
            "metrics": {
                "train": ["loss", "reward"],
                "val": ["reward"],
                "test": ["reward"],
                "log_on_step": True,
            },
            
        }
    )

    lit_module = RL4COLitModule(config, env, model)

    # Set debugging level as info to see all message printouts
    import logging
    logging.basicConfig(level=logging.INFO)

    # Trick to make calculations faster
    torch.set_float32_matmul_precision("medium")

    # Trainer
    import lightning as L
    trainer = L.Trainer(
        max_epochs=3, # 10
        accelerator="gpu",
        logger=None, # can replace with WandbLogger, TensorBoardLogger, etc.
        precision="16-mixed", # Lightning will handle casting to float16
        log_every_n_steps=1,   
        gradient_clip_val=1.0, # clip gradients to avoid exploding gradients!
        reload_dataloaders_every_n_epochs=1, # necessary for sampling new data
    )

    # Fit the model
    trainer.fit(lit_module)