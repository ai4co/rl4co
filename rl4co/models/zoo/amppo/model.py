from rl4co.models.rl.ppo.model import PPO
from rl4co.models.rl.reinforce.critic import CriticNetwork
from rl4co.models.zoo.amppo.policy import PPOAttentionModelPolicy


class AttentionModel(PPO):
    def __init__(self, env, policy=None, critic=None, **policy_kwargs):
        policy = (
            PPOAttentionModelPolicy(env=env, **policy_kwargs)
            if policy is None
            else policy
        )
        critic = CriticNetwork(env=env) if critic is None else critic
        super(AttentionModel, self).__init__(
            env=env,
            policy=policy,
            critic=critic,
            **policy_kwargs,
        )
