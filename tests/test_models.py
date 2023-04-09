import pytest

from ncobench.models.components.am.policy import AttentionModelPolicy
from ncobench.utils.test_utils import generate_env_data


@pytest.mark.parametrize("size", [10, 50])
def test_am(size):
    env, x = generate_env_data('tsp', size)

    model = AttentionModelPolicy(
        env,
        embedding_dim=64,
        hidden_dim=64,
        n_encode_layers=2,
    )

    out = model(x, decode_type="sampling")
    assert out["reward"].shape == (2,)