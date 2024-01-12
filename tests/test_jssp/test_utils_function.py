import pytest
import torch

from rl4co.envs.scheduling.jssp import end_time_lb, last_nonzero_indices


@pytest.fixture
def durations():
    return torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)


@pytest.fixture
def ending_times():
    ending_times = torch.zeros((2, 2), dtype=torch.float32)
    ending_times[0, 0] = 1
    ending_times[1, 0] = 3
    ending_times[1, 1] = 5
    return ending_times


def test_last_non_zero(ending_times):
    x, y = last_nonzero_indices(ending_times)
    assert x.tolist() == [0, 1]
    assert y.tolist() == [0, 1]


def test_end_time_lb(ending_times, durations):
    returned = end_time_lb(ending_times, durations)
    expected = torch.tensor([[1, 3], [3, 5]], dtype=torch.float32)
    assert torch.allclose(returned, expected)


def test_get_action_nbghs():
    pass
