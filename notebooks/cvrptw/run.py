import argparse
import torch

from rl4co.envs import CVRPTWEnv
from rl4co.models.nn.utils import rollout, random_policy
from rl4co.models.zoo.am import AttentionModel
from rl4co.utils.trainer import RL4COTrainer

device_str = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if (torch.backends.mps.is_available() and torch.backends.mps.is_built())
    else "cpu"
)
device = torch.device(device_str)

env_cvrptw = CVRPTWEnv(
    num_loc=30,
    min_loc=0,
    max_loc=150,
    min_demand=1,
    max_demand=10,
    vehicle_capacity=1,
    capacity=10,
    min_time=0,
    max_time=480,
    scale=True,
    device=device_str,
)

env = env_cvrptw

# batch size
batch_size = 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", help="Number of epochs to train for", type=int, default=3
    )
    args = parser.parse_args()

    ### --- random policy --- ###
    reward, td, actions = rollout(
        env=env,
        td=env.reset(batch_size=[batch_size]).to(device),
        policy=random_policy,
        max_steps=1000,
    )
    assert reward.shape == (batch_size,)

    env.get_reward(td, actions)
    CVRPTWEnv.check_solution_validity(td, actions)

    # env.render(td, actions)

    ### --- AM --- ###
    # Model: default is AM with REINFORCE and greedy rollout baseline
    model = AttentionModel(
        env,
        baseline="rollout",
        train_data_size=100_000,
        val_data_size=10_000,
    )

    # Greedy rollouts over untrained model
    td_init = env.reset(batch_size=[3]).to(device)
    model = model.to(device)
    out = model(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)

    ### --- Logging --- ###
    from datetime import date, datetime
    import wandb
    from lightning.pytorch.loggers import WandbLogger

    date_time_str = datetime.now().strftime("%Y/%m/%d_%H:%M:%S")

    wandb.login()
    logger = WandbLogger(
        project="routefinder",
        name=f"cvrptw-am_{date_time_str}",
    )

    ### --- Training --- ###
    # The RL4CO trainer is a wrapper around PyTorch Lightning's `Trainer` class which adds some functionality and more efficient defaults
    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
    )

    # fit model
    trainer.fit(model)

    ### --- Testing --- ###
    trainer.test(model)

    ### --- Saving --- ###
    trainer.save_checkpoint(f"data/cvrptw_{date_time_str}.ckpt")
