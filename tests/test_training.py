import pytest

from rl4co.envs import TSPEnv
from rl4co.models import AttentionModel
from rl4co.utils import RL4COTrainer


# Test out simple training loop and test with multiple baselines
@pytest.mark.parametrize("baseline", ["rollout", "exponential", "critic", "no"])
def test_trainer(baseline):
    env = TSPEnv(num_loc=20)

    model = AttentionModel(
        env, baseline=baseline, train_data_size=10, val_data_size=10, test_data_size=10
    )

    trainer = RL4COTrainer(max_epochs=1)
    trainer.fit(model)
    trainer.test(model)
