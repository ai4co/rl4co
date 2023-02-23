from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from omegaconf import DictConfig
import gymnasium as gym
from gymnasium import Env
import hydra

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.strategies import DataParallelStrategy
from torch import Tensor, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


from src.models.rl.common.experience import Experience, ExperienceSourceDataset
from src.models.rl.common.memory import MultiStepBuffer
from src.models.rl.common.agents import ValueAgent
from src.models.rl.common.losses import dqn_loss
from src.utils import utils

logger = utils.get_logger(__name__)


class DQNBase(LightningModule):
    """Basic DQN Model.
    PyTorch Lightning implementation of `DQN <https://arxiv.org/abs/1312.5602>`_
    Paper authors: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves,
    Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller.
    Model implemented by:
        - `Donal Byrne <https://github.com/djbyrne>`
    Example:
        >>> from pl_bolts.models.rl.dqn_model import DQN
        ...
        >>> model = DQN("PongNoFrameskip-v4")
    Train::
        trainer = Trainer()
        trainer.fit(model)
    Note:
        This example is based on:
        https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/02_dqn_pong.py
    Note:
        Currently only supports CPU and single GPU training with `accelerator=dp`
    Reference:
    Adapted from https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/rl/dqn_model.py
    """

    def __init__(
        self,
        env: str or DictConfig = "CartPole-v0",
        model: str or DictConfig = "mlp",
        eps_start: float = 1.0,
        eps_end: float = 0.02,
        eps_last_frame: int = 150000,
        sync_rate: int = 1000,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        replay_size: int = 100000,
        warm_start_size: int = 10000,
        avg_reward_len: int = 100,
        min_episode_reward: int = -21,
        seed: int = 42,
        batches_per_epoch: int = 1000,
        n_steps: int = 1,
        observation_space_dim: int = 1,
        **kwargs,
    ):
        """
        Args:
            env: gym environment tag
            eps_start: starting value of epsilon for the epsilon-greedy exploration
            eps_end: final value of epsilon for the epsilon-greedy exploration
            eps_last_frame: the final frame in for the decrease of epsilon. At this frame espilon = eps_end
            sync_rate: the number of iterations between syncing up the target network with the train network
            gamma: discount factor
            learning_rate: learning rate
            batch_size: size of minibatch pulled from the DataLoader
            replay_size: total capacity of the replay buffer
            warm_start_size: how many random steps through the environment to be carried out at the start of
                training to fill the buffer with a starting point
            avg_reward_len: how many episodes to take into account when calculating the avg reward
            min_episode_reward: the minimum score that can be achieved in an episode. Used for filling the avg buffer
                before training begins
            seed: seed value for all RNG used
            batches_per_epoch: number of batches per epoch
            n_steps: size of n step look ahead
            observation_space_dim: dimension of the observation space
        """
        super().__init__()

        # Environment
        self.exp = None
        self.env = self.make_environment(env, seed)
        self.test_env = self.make_environment(env)

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.observation_space_dim = observation_space_dim

        # Model Attributes
        self.buffer = None
        self.dataset = None
        self.instantiate_models(model)

        self.agent = ValueAgent(
            self.net,
            self.n_actions,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_frames=eps_last_frame,
        )

        # Hyperparameters
        self.sync_rate = sync_rate
        self.gamma = gamma
        self.lr = learning_rate
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.warm_start_size = warm_start_size
        self.batches_per_epoch = batches_per_epoch
        self.n_steps = n_steps

        self.save_hyperparameters()

        # Metrics
        self.total_episode_steps = [0]
        self.total_rewards = [0]
        self.done_episodes = 0
        self.total_steps = 0

        # Average Rewards
        self.avg_reward_len = avg_reward_len

        for _ in range(avg_reward_len):
            self.total_rewards.append(
                torch.tensor(min_episode_reward, device=self.device)
            )

        self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))

        self.state = self.env.reset()

    def run_n_episodes(
        self, env, n_epsiodes: int = 1, epsilon: float = 1.0
    ) -> List[int]:
        """Carries out N episodes of the environment with the current agent.
        Args:
            env: environment to use, either train environment or test environment
            n_epsiodes: number of episodes to run
            epsilon: epsilon value for DQN agent
        """
        total_rewards = []

        for _ in range(n_epsiodes):
            episode_state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                self.agent.epsilon = epsilon
                action = self.agent(episode_state, self.device, self.get_env_mask())
                next_state, reward, done, _ = env.step(action[0])
                episode_state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

        return total_rewards

    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience."""
        if warm_start > 0:
            self.state = self.env.reset()

            for _ in range(warm_start):
                self.agent.epsilon = 1.0
                action = self.agent(self.state, self.device, self.get_env_mask())
                next_state, reward, done, _ = self.env.step(action[0])
                exp = Experience(
                    state=self.state,
                    action=action[0],
                    reward=reward,
                    done=done,
                    new_state=next_state,
                )
                self.buffer.append(exp)
                self.state = next_state

                if done:
                    self.state = self.env.reset()

    def instantiate_models(self, model: DictConfig or str) -> None:
        """Instantiates the models."""
        obs_size = (
            self.env.observation_space.n * self.observation_space_dim
        )  # hacky way to get the observation space size
        n_actions = self.env.action_space.n
        if model == "mlp":
            model = DictConfig({"_target_": "src.models.rl.common.networks.DQN_mlp"})
        logger.info(f"Instantiating  <{model._target_}>")
        self.net = hydra.utils.instantiate(
            model, obs_size=obs_size, n_actions=n_actions
        )
        self.target_net = hydra.utils.instantiate(
            model, obs_size=obs_size, n_actions=n_actions
        )

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.
        Args:
            x: environment state
        Returns:
            q values
        """
        output = self.net(x)
        return output

    def train_batch(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.
        Returns:
            yields a Experience tuple containing the state, action, reward, done and next_state.
        """
        episode_reward = 0
        episode_steps = 0

        while True:
            self.total_steps += 1
            action = self.agent(self.state, self.device, self.get_env_mask())

            next_state, r, is_done, _ = self.env.step(action[0])

            episode_reward += r
            episode_steps += 1

            exp = Experience(
                state=self.state,
                action=action[0],
                reward=r,
                done=is_done,
                new_state=next_state,
            )

            self.agent.update_epsilon(self.global_step)
            self.buffer.append(exp)
            self.state = next_state

            if is_done:
                self.done_episodes += 1
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(
                    np.mean(self.total_rewards[-self.avg_reward_len :])
                )
                self.state = self.env.reset()
                episode_steps = 0
                episode_reward = 0

            states, actions, rewards, dones, new_states = self.buffer.sample(
                self.batch_size
            )

            for idx, _ in enumerate(dones):
                yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[
                    idx
                ]

            # Simulates epochs
            if self.total_steps % self.batches_per_epoch == 0:
                break

    def training_step(self, batch: Tuple[Tensor, Tensor], _) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.
        Args:
            batch: current mini batch of replay data
            _: batch number, not used
        Returns:
            Training loss and log metrics
        """

        # calculates training loss
        loss = dqn_loss(batch, self.net, self.target_net, self.gamma)

        if self._use_dp(self.trainer):
            loss = loss.unsqueeze(0)

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict(
            {
                "train/total_reward": self.total_rewards[-1],
                "train/avg_reward": self.avg_rewards,
                "train/loss": loss,
                "train/episodes": self.done_episodes,
                "train/episode_steps": self.total_episode_steps[-1],
                "train/epsilon": self.agent.epsilon,
            }
        )

        return {"loss": loss, "avg_reward": self.avg_rewards}

    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Evaluate the agent for 10 episodes."""
        test_reward = self.run_n_episodes(self.test_env, 1, 0)
        avg_reward = sum(test_reward) / len(test_reward)
        return {"test/reward": avg_reward}

    def test_epoch_end(self, outputs) -> Dict[str, Tensor]:
        """Log the avg of the test results."""
        rewards = [x["test/reward"] for x in outputs]
        avg_reward = sum(rewards) / len(rewards)
        self.log("test/avg_reward", avg_reward)
        return {"test/avg_reward": avg_reward}

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        self.buffer = MultiStepBuffer(self.replay_size, self.n_steps)
        self.populate(self.warm_start_size)
        self.dataset = ExperienceSourceDataset(self.train_batch)
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size)

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get test loader."""
        return self._dataloader()

    def get_env_mask(self, env: Optional[Env] = None) -> Tensor:
        """Get the mask for the current environment."""
        return self.env.get_mask() if hasattr(self.env, "get_mask") else None

    @staticmethod
    def make_environment(env: str or DictConfig, seed: Optional[int] = None) -> Env:
        """Initialise gym  environment.
        Args:
            env_name: environment name or tag
            seed: value to seed the environment RNG for reproducibility
        Returns:
            gym environment
        """
        # Looks like declaring the environment here is way faster for some reason
        if isinstance(env, str):
            env = gym.make(env)
        elif isinstance(env, DictConfig):
            logger.info(f"Instantiating  <{env._target_}>")
            env = hydra.utils.instantiate(env)
        else:
            env = env()
        if seed:
            if hasattr(env, "seed"):
                env.seed(seed)
            else:
                logger.warning(
                    "Environment does not have seed method, skipping seeding. May need to modify this code"
                )
        return env

    @staticmethod
    def _use_dp(trainer: Trainer) -> bool:
        return isinstance(trainer.strategy, DataParallelStrategy)


class DQN(DQNBase):
    """DQN with custom testing"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        pass

    def test_epoch_end(self, outputs):
        rewards = self.custom_test()
        self.log("test/reward_mean", rewards.mean())
        self.log("test/reward_std", rewards.std())
        return {"test/reward_mean": rewards.mean(), "test/reward_std": rewards.std()}

    def custom_test(self):
        """Test the agent on the environment."""
        self.agent.net.eval()
        with torch.no_grad():
            rewards = []
            test_envs = len(self.env.initial_layouts)
            for i in range(test_envs):
                state = self.env.reset(i)
                done = False
                reward = 0
                while not done:
                    mask = self.env.get_mask()
                    action = self.agent(state, self.device, mask)
                    _, cur_rew, done, mask = self.env.step(action)
                    reward += cur_rew
                rewards.append(reward)  # only the last reward is used
            return torch.tensor(rewards)
