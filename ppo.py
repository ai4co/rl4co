import torch

from rl4co.envs import CVRPEnv, TSPEnv
from rl4co.models.zoo.ppo.model import PPOModel
from rl4co.utils.trainer import RL4COTrainer

if __name__ == "__main__":
    # RL4CO env based on TorchRL
    env = CVRPEnv(num_loc=20)

    # Model: default is AM with REINFORCE and greedy rollout baseline
    model = PPOModel(
        env,
        train_data_size=100,
        val_data_size=10,
        mini_batch_size=1.0,
        optimizer_kwargs={"lr": 0.0001},
    )

    # Greedy rollouts over untrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td_init = env.reset(batch_size=[3]).to(device)
    model = model.to(device)
    out = model(td_init, phase="test", decode_type="greedy", return_actions=True)

    # Plotting
    print(f"Tour lengths: {[f'{-r.item():.2f}' for r in out['reward']]}")
    for td, actions in zip(td_init, out["actions"].cpu()):
        env.render(td, actions)

    trainer = RL4COTrainer(max_epochs=3, accelerator="gpu", precision="32", logger=None)

    trainer.fit(model)
