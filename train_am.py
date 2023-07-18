from math import e

from lightning.pytorch.loggers import WandbLogger

from rl4co.envs import TSPEnv
from rl4co.models.zoo.am import AttentionModel
from rl4co.utils.trainer import RL4COTrainer

FREEZE_DECODER = True

if __name__ == "__main__":
    env = TSPEnv(num_loc=50)

    model = AttentionModel(env, baseline="warmup")
    # freeze decoder
    if FREEZE_DECODER:
        model.policy.decoder.requires_grad_(False)
        wandb_name = "am-frozen_decoder"
    else:
        wandb_name = "am"

    trainer = RL4COTrainer(
        max_epochs=100,
        accelerator="gpu",
        devices=[0],
        logger=WandbLogger(project="frozen_dcoder", name=wandb_name),
    )
    trainer.fit(model)
