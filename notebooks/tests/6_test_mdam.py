import torch
from torch.utils.data import DataLoader

from rl4co.envs import TSPEnv
from rl4co.data.dataset import TensorDictCollate
from rl4co.models.rl.reinforce.base import REINFORCE
from rl4co.models.rl.reinforce.baselines import RolloutBaseline, WarmupBaseline, ExponentialBaseline
from rl4co.models import MDAMPolicy


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
        self.policy = MDAMPolicy(env) if policy is None else policy
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
    policy = MDAMPolicy(
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