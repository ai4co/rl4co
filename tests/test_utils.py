import pytest
import torch

from tensordict import TensorDict
from torch.nn.functional import scaled_dot_product_attention

from rl4co.models.nn.attention import scaled_dot_product_attention_simple
from rl4co.utils.decoding import process_logits
from rl4co.utils.ops import batchify, unbatchify


@pytest.mark.parametrize(
    "a",
    [
        torch.randn(10, 20, 2),
        TensorDict({"a": torch.randn(10, 20, 2), "b": torch.randn(10, 20, 2)}, batch_size=10),
    ],
)
@pytest.mark.parametrize("shape", [(2,), (2, 2), (2, 2, 2)])
def test_batchify(a, shape):
    # batchify: [b, ...] -> [b * prod(shape), ...]
    # unbatchify: [b * prod(shape), ...] -> [b, shape[0], shape[1], ...]
    a_batch = batchify(a, shape)
    a_unbatch = unbatchify(a_batch, shape)
    if isinstance(a, TensorDict):
        a, a_unbatch = a["a"], a_unbatch["a"]
    index = (slice(None),) + (0,) * len(shape)  # (slice(None), 0, 0, ..., 0)
    assert torch.allclose(a, a_unbatch[index])


@pytest.mark.parametrize("top_p", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("top_k", [0, 5, 10])
def test_top_k_top_p_sampling(top_p, top_k):
    logits = torch.randn(8, 10)
    mask = torch.ones(8, 10).bool()
    logprobs = process_logits(logits, mask, top_p=top_p, top_k=top_k)
    assert len(logprobs) == logits.size(0)


def test_scaled_dot_product_attention():
    bs, ns, ds = 2, 3, 4
    q = torch.rand(bs, ns, ds)
    k = torch.rand(bs, ns, ds)
    v = torch.rand(bs, ns, ds)
    attn_mask = torch.rand(bs, ns, ns) > 0.5
    attn_mask[:, 0, :] = True  # at least one row element is True
    attn_mask[:, :, 0] = True  # at least one column element is True
    attn_torch = scaled_dot_product_attention(q, k, v, attn_mask)
    attn_rl4co = scaled_dot_product_attention_simple(q, k, v, attn_mask)
    assert torch.allclose(attn_torch, attn_rl4co)


from rl4co.utils.pe import (  # noqa: E402
    AbsolutePE,
    ALiBiBias,
    CrossRoutePE,
    CycleFormerPE,
    DACTCyclicPE,
    HierarchicalPE,
    InRoutePE,
    LaplacianPE,
    RandomWalkSE,
    RelativePE,
    RotaryPE,
    ShortestPathBias,
    SinusoidalPE,
    build_route_graph,
    get_positional_encoding,
)


def _closed_route_coords(batch: int, num_loc: int) -> torch.Tensor:
    """Random route-ordered coords with the route closed (v_1 == v_L == depot)."""
    coords = torch.rand(batch, num_loc, 2)
    coords[:, -1, :] = coords[:, 0, :]
    return coords


@pytest.mark.parametrize("embed_dim", [16, 32])
@pytest.mark.parametrize("num_loc", [7, 20])
def test_pe_per_node_shapes(embed_dim, num_loc):
    bs = 3
    positions = torch.arange(num_loc).unsqueeze(0).expand(bs, num_loc).contiguous()
    coords = _closed_route_coords(bs, num_loc)
    depot = coords[:, 0, :]
    assert AbsolutePE(embed_dim)(positions).shape == (bs, num_loc, embed_dim)
    assert SinusoidalPE(embed_dim)(positions).shape == (bs, num_loc, embed_dim)
    assert DACTCyclicPE(embed_dim, max_len=64)(positions).shape == (bs, num_loc, embed_dim)
    assert CycleFormerPE(embed_dim)(positions).shape == (bs, num_loc, embed_dim)
    assert InRoutePE(embed_dim, direction_aware=True)(coords).shape == (bs, num_loc, embed_dim)
    assert InRoutePE(embed_dim, direction_aware=False)(coords).shape == (bs, num_loc, embed_dim)
    assert CrossRoutePE(embed_dim)(coords, depot).shape == (bs, num_loc, embed_dim)
    assert HierarchicalPE(embed_dim)(coords).shape == (bs, num_loc, 2 * embed_dim)


@pytest.mark.parametrize("embed_dim", [16, 32])
def test_pe_bias_shapes(embed_dim):
    seq_len = 9
    assert RelativePE(window=4)(seq_len).shape == (seq_len, seq_len)
    assert ALiBiBias(num_heads=8)(seq_len).shape == (8, seq_len, seq_len)
    routes = torch.tensor([[0, 1, 2, 3, 0]])
    assert ShortestPathBias(max_spd=8)(routes=routes, num_nodes=4).shape == (1, 4, 4)


@pytest.mark.parametrize("embed_dim", [16, 32])
def test_rope_relative_offset_property(embed_dim):
    seq_len = 6
    rope = RotaryPE(embed_dim)
    positions = torch.arange(seq_len).unsqueeze(0)
    const = torch.ones(1, seq_len, embed_dim)
    q_rot, k_rot = rope.rotate_queries_keys(const.clone(), const.clone(), positions)
    logits = (q_rot @ k_rot.transpose(-1, -2))[0]
    # logit(i, j) depends only on i - j: constant along each diagonal
    for off in range(seq_len):
        diag = torch.diagonal(logits, offset=off)
        assert torch.allclose(diag, diag[0].expand_as(diag), atol=1e-4)


@pytest.mark.parametrize("num_heads", [4, 8])
def test_alibi_bias_values(num_heads):
    seq_len = 7
    module = ALiBiBias(num_heads)
    bias = module(seq_len)
    idx = torch.arange(seq_len)
    expected = -module.slopes[:, None, None] * (idx[:, None] - idx[None, :]).abs().float()
    assert torch.allclose(bias, expected, atol=1e-6)


@pytest.mark.parametrize("embed_dim", [16, 32])
@pytest.mark.parametrize("num_loc", [7, 20])
def test_ipe_circularity_and_reversal(embed_dim, num_loc):
    coords = _closed_route_coords(2, num_loc)
    # direction-invariant variant: closed-route endpoints share the encoding (D2 topological)
    ipe_inv = InRoutePE(embed_dim, direction_aware=False)
    enc = ipe_inv(coords)
    # tolerance accounts for float32 error in cos() of the largest harmonic (arg up to 2*pi*D)
    tol = 1e-4
    assert torch.allclose(enc[:, 0], enc[:, -1], atol=tol)
    # ... and is invariant to reversing the route order
    enc_rev = ipe_inv(coords.flip(1)).flip(1)
    assert torch.allclose(enc, enc_rev, atol=tol)
    assert enc.shape[-1] == embed_dim
    # direction-aware variant: endpoints still equal, but the encoding is NOT reversal-invariant
    ipe_dir = InRoutePE(embed_dim, direction_aware=True)
    enc_d = ipe_dir(coords)
    assert torch.allclose(enc_d[:, 0], enc_d[:, -1], atol=tol)
    enc_d_rev = ipe_dir(coords.flip(1)).flip(1)
    assert not torch.allclose(enc_d, enc_d_rev, atol=1e-3)
    # cosine half (odd channels) unchanged, sine half (even channels) negated under reversal
    assert torch.allclose(enc_d[..., 1::2], enc_d_rev[..., 1::2], atol=tol)
    assert torch.allclose(enc_d[..., 0::2], -enc_d_rev[..., 0::2], atol=tol)
    assert enc_d.shape[-1] == embed_dim


@pytest.mark.parametrize("embed_dim", [16, 32])
def test_xpe_angle_only_dependence(embed_dim):
    xpe = CrossRoutePE(embed_dim, k=4)
    depot = torch.zeros(1, 2)
    # two coordinate sets sharing the same depot-relative angles but different radii
    base = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]]])
    scaled = base * 3.7
    assert torch.allclose(xpe(base, depot), xpe(scaled, depot), atol=1e-5)
    # zero-padding beyond 2K columns
    enc = xpe(base, depot)
    assert enc.shape[-1] == embed_dim
    assert torch.allclose(enc[..., 2 * 4 :], torch.zeros_like(enc[..., 2 * 4 :]))
    # 2K > D is rejected
    with pytest.raises(ValueError):
        CrossRoutePE(embed_dim, k=embed_dim)


@pytest.mark.parametrize("embed_dim", [16, 32])
@pytest.mark.parametrize("num_loc", [7, 20])
def test_cycleformer_wraparound(embed_dim, num_loc):
    cf = CycleFormerPE(embed_dim)
    p1 = torch.arange(num_loc).unsqueeze(0)
    p2 = p1 + num_loc
    assert torch.allclose(cf(p1, seq_len=num_loc), cf(p2, seq_len=num_loc), atol=1e-6)


@pytest.mark.parametrize("embed_dim", [16, 32])
def test_laplacian_pe_orthonormality_and_padding(embed_dim):
    routes = torch.tensor([[0, 1, 2, 3, 0]])
    lap = LaplacianPE(embed_dim, k=8).eval()  # deterministic in eval (no sign flips)
    enc = lap(routes=routes, num_nodes=4)[0]  # [N=4, D]
    k_eff = min(8, 4 - 1)
    gram = enc[:, :k_eff].T @ enc[:, :k_eff]
    assert torch.allclose(gram, torch.eye(k_eff), atol=1e-4)
    assert torch.allclose(enc[:, k_eff:], torch.zeros_like(enc[:, k_eff:]), atol=1e-6)


@pytest.mark.parametrize("embed_dim", [16, 32])
def test_rwse_range(embed_dim):
    routes = torch.tensor([[0, 1, 2, 3, 0]])
    enc = RandomWalkSE(embed_dim, k=8)(routes=routes, num_nodes=4)
    assert enc.min().item() >= -1e-6 and enc.max().item() <= 1.0 + 1e-6


def test_build_route_graph_symmetry():
    routes = torch.tensor([[0, 1, 2, 3, 0]])
    adj = build_route_graph(routes, num_nodes=4)
    assert adj.shape == (1, 4, 4)
    assert torch.allclose(adj, adj.transpose(-1, -2))
    assert torch.allclose(torch.diagonal(adj, dim1=-2, dim2=-1), torch.zeros(1, 4))
    # SPD on the same graph: symmetric and zero diagonal distance index
    spd = ShortestPathBias(max_spd=8)(adj=adj)[0]
    assert torch.allclose(spd, spd.T)
    diag_bias = torch.diagonal(spd)
    assert torch.allclose(diag_bias, diag_bias[0].expand_as(diag_bias), atol=1e-6)


def test_pe_factory_dispatch():
    names = [
        "APE",
        "SIN",
        "RoPE",
        "RPE",
        "ALiBi",
        "LapPE",
        "RWSE",
        "SPD",
        "DACT",
        "CycleFormer",
        "IPE",
        "XPE",
        "Hierarchical",
    ]
    kwargs = {
        "APE": {"embed_dim": 16},
        "SIN": {"embed_dim": 16},
        "RoPE": {"embed_dim": 16},
        "RPE": {"window": 4},
        "ALiBi": {"num_heads": 8},
        "LapPE": {"embed_dim": 16},
        "RWSE": {"embed_dim": 16},
        "SPD": {"max_spd": 8},
        "DACT": {"embed_dim": 16},
        "CycleFormer": {"embed_dim": 16},
        "IPE": {"embed_dim": 16},
        "XPE": {"embed_dim": 16},
        "Hierarchical": {"embed_dim": 16},
    }
    for name in names:
        module = get_positional_encoding(name, **kwargs[name])
        assert isinstance(module, torch.nn.Module)
    assert isinstance(get_positional_encoding("Hierarchical", embed_dim=16), HierarchicalPE)
    with pytest.raises(ValueError):
        get_positional_encoding("does-not-exist")
