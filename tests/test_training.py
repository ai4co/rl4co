import pytest

from rl4co.envs import PDPEnv, TSPEnv, ATSPEnv
from rl4co.models import (
    ActiveSearch,
    AttentionModel,
    AutoregressivePolicy,
    EASEmb,
    EASLay,
    HeterogeneousAttentionModel,
    MatNet,
    PPOModel,
    SymNCO,
)
from rl4co.utils import RL4COTrainer


# Test out simple training loop and test with multiple baselines
@pytest.mark.parametrize("baseline", ["rollout", "exponential", "critic", "mean", "no"])
def test_reinforce(baseline):
    env = TSPEnv(num_loc=20)

    model = AttentionModel(
        env, baseline=baseline, train_data_size=10, val_data_size=10, test_data_size=10
    )

    trainer = RL4COTrainer(max_epochs=1, devices=1)
    trainer.fit(model)
    trainer.test(model)


def test_ppo():
    env = TSPEnv(num_loc=20)
    model = PPOModel(env, train_data_size=10, val_data_size=10, test_data_size=10)
    trainer = RL4COTrainer(max_epochs=1, gradient_clip_val=None, devices=1)
    trainer.fit(model)
    trainer.test(model)


def test_symnco():
    env = TSPEnv(num_loc=20)
    model = SymNCO(
        env,
        train_data_size=10,
        val_data_size=10,
        test_data_size=10,
        num_augment=2,
        num_starts=20,
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1)
    trainer.fit(model)
    trainer.test(model)


def test_ham():
    env = PDPEnv(num_loc=20)
    model = HeterogeneousAttentionModel(
        env, train_data_size=10, val_data_size=10, test_data_size=10
    )
    trainer = RL4COTrainer(max_epochs=1, devices=1)
    trainer.fit(model)
    trainer.test(model)


def test_matnet():
    env = ATSPEnv(num_loc=20)
    model = MatNet(
        env, 
        baseline="shared", 
        train_data_size=10, 
        val_data_size=10, 
        test_data_size=10,
    ) 
    trainer = RL4COTrainer(max_epochs=1, devices=1)
    trainer.fit(model)
    trainer.test(model)


@pytest.mark.parametrize("SearchMethod", [ActiveSearch, EASEmb, EASLay])
def test_search_methods(SearchMethod):
    env = TSPEnv(num_loc=20)
    batch_size = 2 if SearchMethod not in [ActiveSearch] else 1
    dataset = env.dataset(2)
    policy = AutoregressivePolicy(env)
    model = SearchMethod(env, policy, dataset, max_iters=2, batch_size=batch_size)
    trainer = RL4COTrainer(max_epochs=1, devices=1)
    trainer.fit(model)
    trainer.test(model)
