from collections import OrderedDict
from typing import Tuple

import torch
from torch import Tensor

from src.models.rl.dqn import DQN
from src.models.rl.common.losses import double_dqn_loss


class DoubleDQNBase(DQN):
    """Double Deep Q-network (DDQN) PyTorch Lightning implementation of `Double DQN`_.
    Paper authors: Hado van Hasselt, Arthur Guez, David Silver
    Model implemented by:
        - `Donal Byrne <https://github.com/djbyrne>`
    Example:
        >>> from pl_bolts.models.rl.double_dqn_model import DoubleDQN
        ...
        >>> model = DoubleDQN("PongNoFrameskip-v4")
    Train::
        trainer = Trainer()
        trainer.fit(model)
    Note:
        This example is based on
        https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter08/03_dqn_double.py
    Note:
        Currently only supports CPU and single GPU training with `accelerator=dp`
    .. _`Double DQN`: https://arxiv.org/pdf/1509.06461.pdf
    Reference:
    Adapted from https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/rl/double_dqn_model.py
    """

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
        loss = double_dqn_loss(batch, self.net, self.target_net, self.gamma)

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
                "train/epsilon": self.agent.epsilon,
                # "episodes": self.total_episode_steps,
            }
        )

        return {"loss": loss, "avg_reward": self.avg_rewards}


class DoubleDQN(DoubleDQNBase):
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
