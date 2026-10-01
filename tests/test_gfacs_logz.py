import pytest
import torch
from torch import nn
from tensordict import TensorDict

from rl4co.models.zoo.gfacs.encoder import GFACSEncoder, _pool_edge_logz


@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("counts", [(4, 4), (3, 5), (3, 4), (5,)])
def test_pool_graph_means(counts, heads):
    values = torch.arange(len(counts) * heads, dtype=torch.float32).reshape(-1, heads) + 10
    membership = torch.arange(len(counts)).repeat_interleave(torch.tensor(counts))
    predictions = values[membership].clone().requires_grad_()
    pooled = _pool_edge_logz(predictions, membership, len(counts))
    torch.testing.assert_close(pooled, values)
    pooled[0].sum().backward()
    expected = torch.zeros_like(predictions)
    expected[membership == 0] = 1 / counts[0]
    torch.testing.assert_close(predictions.grad, expected)


def test_pool_graph_permutation_and_isolation():
    predictions = torch.tensor([[1., 2.], [3., 4.], [10., 20.], [30., 40.], [50., 60.]])
    membership = torch.tensor([0, 0, 1, 1, 1])
    expected = torch.tensor([[2., 3.], [30., 40.]])
    torch.testing.assert_close(_pool_edge_logz(predictions, membership, 2), expected)
    order = torch.tensor([2, 3, 4, 0, 1])
    torch.testing.assert_close(_pool_edge_logz(predictions[order], 1 - membership[order], 2), expected.flip(0))
    changed = predictions.clone()
    changed[membership == 1] += 1000
    torch.testing.assert_close(_pool_edge_logz(changed, membership, 2)[0], expected[0])


@pytest.mark.parametrize("heads", [1, 2])
def test_encoder_forward_pools_using_edge_membership(heads):
    pyg = pytest.importorskip("torch_geometric.data")
    graph = pyg.Batch.from_data_list([
        pyg.Data(x=torch.zeros(2, heads), edge_index=torch.tensor([[0, 1, 0], [1, 0, 1]]),
                 edge_attr=torch.full((3, heads), 10.)),
        pyg.Data(x=torch.zeros(2, heads), edge_index=torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]]),
                 edge_attr=torch.full((4, heads), 30.)),
    ])

    class Edges(nn.Module):
        def forward(self, td, nodes):
            return graph

    class Network(nn.Module):
        def forward(self, nodes, edges, attributes):
            return nodes, attributes

    class Heatmap(nn.Module):
        def forward(self, graph):
            return graph.edge_attr

    encoder = GFACSEncoder(embed_dim=heads, init_embedding=nn.Identity(),
                          edge_embedding=Edges(), graph_network=Network(),
                          heatmap_generator=Heatmap(), z_out_dim=heads)
    encoder.Z_net = nn.Linear(heads, heads, bias=False)
    with torch.no_grad():
        encoder.Z_net.weight.copy_(torch.eye(heads))
    _, _, logz = encoder(TensorDict({}, batch_size=[2]))
    torch.testing.assert_close(logz, torch.tensor([[10.] * heads, [30.] * heads]))
    logz[0].sum().backward()
    torch.testing.assert_close(encoder.Z_net.weight.grad, torch.full((heads, heads), 10.))
