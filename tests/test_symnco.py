import pytest
import torch

from torch.nn.functional import cosine_similarity

from rl4co.models.zoo.symnco.losses import invariance_loss


@pytest.mark.parametrize("num_augment", [2, 4])
def test_invariance_loss_prefers_aligned_embeddings(num_augment):
    aligned = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]).repeat(num_augment, 1, 1)
    opposite = aligned.clone()
    opposite[1:] *= -1

    assert invariance_loss(aligned, num_augment) < invariance_loss(opposite, num_augment)


@pytest.mark.parametrize("num_augment", [2, 4])
def test_invariance_loss_gradient_increases_similarity(num_augment):
    embeddings = torch.tensor(
        [[[1.0, 0.0]]] + [[[0.0, 1.0]]] * (num_augment - 1), requires_grad=True
    )
    loss = invariance_loss(embeddings, num_augment)
    (gradient,) = torch.autograd.grad(loss, embeddings)

    with torch.no_grad():
        updated = embeddings - 0.1 * gradient
        before = cosine_similarity(embeddings[0], embeddings[1:], dim=-1)
        after = cosine_similarity(updated[0], updated[1:], dim=-1)

    assert torch.all(after > before)
