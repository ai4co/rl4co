import pytest

from rl4co.envs import TSPEnv


@pytest.mark.parametrize("size", [10, 50])
def test_tsp(size):
    env = TSPEnv(num_loc=size)
    a = env.reset()
    assert a["observation"].shape == (size, 2)
