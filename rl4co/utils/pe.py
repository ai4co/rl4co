from __future__ import annotations

import math

import torch
import torch.nn as nn

from torch import Tensor

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

__all__ = [
    # index-based / NLP-style
    "AbsolutePE",
    "SinusoidalPE",
    "RotaryPE",
    "RelativePE",
    "ALiBiBias",
    # routing-specific cyclic
    "DACTCyclicPE",
    "CycleFormerPE",
    # graph-transformer
    "build_route_graph",
    "LaplacianPE",
    "RandomWalkSE",
    "ShortestPathBias",
    # proposed method (paper §4)
    "InRoutePE",
    "CrossRoutePE",
    "HierarchicalPE",
    # factory
    "get_positional_encoding",
]


# ======================================================================================
# Helpers
# ======================================================================================
def _geometric_frequencies(num_freq: int, dim: int, base: float = 10000.0) -> Tensor:
    """Geometric frequency schedule ``base^{-2k/dim}`` for ``k = 0 .. num_freq - 1``.

    This matches the schedule used by the original Transformer sinusoidal PE
    (:cite:`vaswani2017attention`) and by RoPE.

    Args:
        num_freq: Number of frequency bands to produce.
        dim: Embedding dimension used in the exponent ``2k / dim``.
        base: The wavelength base ``λ`` (default ``10000``).

    Returns:
        A 1-D tensor of shape ``[num_freq]``.
    """
    k = torch.arange(num_freq, dtype=torch.float32)
    return base ** (-2.0 * k / dim)


def _binary_reflected_gray(n: int, num_bits: int) -> list[int]:
    """Return the ``num_bits``-bit binary-reflected Gray code of integer ``n`` as a list.

    Consecutive integers ``n`` and ``n + 1`` differ in exactly one bit.  Note this is *not*
    bit-wise circular when the cycle length is not a power of two (the wrap from the last
    code back to ``0`` may flip more than one bit).

    Args:
        n: The non-negative integer to encode.
        num_bits: Number of output bits (most-significant bit first).

    Returns:
        A list of ``num_bits`` values in ``{0, 1}``.
    """
    gray = n ^ (n >> 1)
    return [(gray >> b) & 1 for b in range(num_bits - 1, -1, -1)]


# ======================================================================================
# Index-based / NLP-style PEs
# ======================================================================================
class AbsolutePE(nn.Module):
    """Absolute PE (APE): a learnable lookup table indexed by within-route position.

    Implements ``PE^APE(v_i) = E_i`` with ``E ∈ R^{(max_len + 1) × D}`` trainable
    (index ``0`` reserved for the depot).

    Args:
        embed_dim: Embedding dimension ``D``.
        max_len: Maximum within-route position the table can index (``L_max``).
    """

    def __init__(self, embed_dim: int, max_len: int = 1000) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.embedding = nn.Embedding(max_len + 1, embed_dim)

    def forward(self, positions: Tensor) -> Tensor:
        """Look up the per-position embeddings.

        Args:
            positions: Long tensor of shape ``[..., L]`` with within-route indices.

        Returns:
            Tensor of shape ``[..., L, D]``.
        """
        return self.embedding(positions.long().clamp(max=self.max_len))


class SinusoidalPE(nn.Module):
    """Sinusoidal PE (SIN): fixed sin/cos of the integer index (:cite:`vaswani2017attention`).

    ``PE^SIN_{2k}(v_i) = sin(i / λ^{2k/D})`` and ``PE^SIN_{2k+1}(v_i) = cos(i / λ^{2k/D})``
    with ``λ = 10000`` and ``k = 0 .. D/2 - 1``.

    Args:
        embed_dim: Embedding dimension ``D`` (assumed even; an odd ``D`` drops the last
            channel of the cosine half).
        base: Wavelength base ``λ`` (default ``10000``).
    """

    def __init__(self, embed_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("freqs", _geometric_frequencies(embed_dim // 2, embed_dim, base))

    def forward(self, positions: Tensor) -> Tensor:
        """Compute the sinusoidal encoding for the given positions.

        Args:
            positions: Tensor of shape ``[..., L]`` (int or float) with within-route indices.

        Returns:
            Tensor of shape ``[..., L, D]``.
        """
        angles = positions.unsqueeze(-1).float() * self.freqs  # [..., L, D/2]
        out = torch.zeros(
            *angles.shape[:-1], self.embed_dim, device=angles.device, dtype=angles.dtype
        )
        out[..., 0::2] = torch.sin(angles)
        out[..., 1::2] = torch.cos(angles[..., : out[..., 1::2].shape[-1]])
        return out


class RotaryPE(nn.Module):
    """Rotary PE (RoPE): query/key rotation rather than an additive embedding.

    For each channel pair ``(2k, 2k+1)`` the query/key is rotated by ``θ_i = i · λ^{-2k/D}``
    (:cite:`su2024roformer`).  The resulting attention logit ``q_i^T k_j`` depends only on the
    index difference ``i - j``.

    Args:
        embed_dim: Per-head dimension ``D`` (assumed even).
        base: Wavelength base ``λ`` (default ``10000``).
    """

    def __init__(self, embed_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, "RotaryPE requires an even embed_dim"
        self.embed_dim = embed_dim
        self.register_buffer("freqs", _geometric_frequencies(embed_dim // 2, embed_dim, base))

    def _rotary_components(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        # positions: [..., L] -> cos/sin of shape [..., L, D] (each freq duplicated for the pair)
        angles = positions.unsqueeze(-1).float() * self.freqs  # [..., L, D/2]
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # [..., L, D]
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)  # [..., L, D]
        return cos, sin

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        # (..., [x0, x1, x2, x3, ...]) -> (..., [-x1, x0, -x3, x2, ...])
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        """Apply the rotary transform to a single tensor.

        Args:
            x: Tensor of shape ``[..., L, D]``.
            positions: Tensor of shape ``[..., L]`` with the index of each element.

        Returns:
            Rotated tensor of the same shape as ``x``.
        """
        cos, sin = self._rotary_components(positions)
        return x * cos + self._rotate_half(x) * sin

    def rotate_queries_keys(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate both queries and keys by their positions.

        Args:
            q: Query tensor ``[..., L, D]``.
            k: Key tensor ``[..., L, D]``.
            positions: Index tensor ``[..., L]`` shared by ``q`` and ``k``.

        Returns:
            Tuple ``(q_rot, k_rot)`` with the same shapes as the inputs.
        """
        return self.rotate(q, positions), self.rotate(k, positions)

    def forward(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Alias for :meth:`rotate_queries_keys`."""
        return self.rotate_queries_keys(q, k, positions)


class RelativePE(nn.Module):
    """Relative PE (RPE): a learnable bias indexed by signed index offset (:cite:`shaw2018self`).

    ``logit(i, j) += b_{clip(i - j, -W, W)}`` with ``{b_Δ}_{Δ=-W}^{W}`` shared across heads.

    Args:
        window: Clipping window ``W``; offsets are clipped to ``[-W, W]``.
    """

    def __init__(self, window: int = 16) -> None:
        super().__init__()
        self.window = window
        self.bias = nn.Embedding(2 * window + 1, 1)

    def forward(self, seq_len: int, device: torch.device | None = None) -> Tensor:
        """Build the ``[L, L]`` bias matrix added to attention logits.

        Args:
            seq_len: Sequence length ``L``.
            device: Optional device for the output.

        Returns:
            Tensor of shape ``[L, L]`` where entry ``(i, j)`` is ``b_{clip(i-j, -W, W)}``.
        """
        idx = torch.arange(seq_len, device=device)
        offset = (idx[:, None] - idx[None, :]).clamp(-self.window, self.window) + self.window
        return self.bias(offset).squeeze(-1)


class ALiBiBias(nn.Module):
    """ALiBi: a fixed linear penalty subtracted from attention logits (:cite:`press2021train`).

    ``logit(i, j) -= m_h · |i - j|`` with head-specific slopes ``m_h`` following the
    geometric schedule of the original paper.  The returned tensor is shaped
    ``[num_heads, L, L]`` (broadcasts over the batch dimension).

    Args:
        num_heads: Number of attention heads ``n_h``.
    """

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.register_buffer("slopes", self._get_slopes(num_heads))

    @staticmethod
    def _get_slopes(num_heads: int) -> Tensor:
        # Original ALiBi schedule: for n_h a power of two, m_h = 2^{-8h/n_h}, h = 1..n_h.
        def power_of_two_slopes(n: int) -> list[float]:
            start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
            return [start ** (i + 1) for i in range(n)]

        if math.log2(num_heads).is_integer():
            return torch.tensor(power_of_two_slopes(num_heads), dtype=torch.float32)
        # Non-power-of-two fallback (as in the original implementation): take the slopes of
        # the closest lower power of two, then interleave slopes of the next power of two.
        closest = 2 ** math.floor(math.log2(num_heads))
        slopes = power_of_two_slopes(closest)
        extra = power_of_two_slopes(2 * closest)[0::2][: num_heads - closest]
        return torch.tensor(slopes + extra, dtype=torch.float32)

    def forward(self, seq_len: int, device: torch.device | None = None) -> Tensor:
        """Build the ``[num_heads, L, L]`` ALiBi bias.

        Args:
            seq_len: Sequence length ``L``.
            device: Optional device for the output.

        Returns:
            Tensor of shape ``[num_heads, L, L]`` equal to ``-m_h · |i - j|``.
        """
        idx = torch.arange(seq_len, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().float()  # [L, L]
        slopes = self.slopes.to(device=device)
        return -slopes[:, None, None] * dist[None, :, :]


# ======================================================================================
# Routing-specific cyclic PEs
# ======================================================================================
class DACTCyclicPE(nn.Module):
    """DACT cyclic PE: a Gray-code lookup over the cyclic index, projected to ``D``.

    ``PE^DACT(v_i) = Linear(Gray(i mod L))`` (:cite:`ma2021learning`).  The Gray code uses
    ``ceil(log2(L))`` bits so consecutive cyclic positions differ by exactly one bit; this is
    *not* bit-wise circular when ``L`` is not a power of two (documented limitation).

    Args:
        embed_dim: Output embedding dimension ``D``.
        max_len: Maximum route length ``L`` supported (sets the number of Gray-code bits).
    """

    def __init__(self, embed_dim: int, max_len: int = 1000) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.num_bits = max(1, math.ceil(math.log2(max(2, max_len))))
        codes = torch.tensor(
            [_binary_reflected_gray(n, self.num_bits) for n in range(max_len)], dtype=torch.float32
        )
        self.register_buffer("gray_codes", codes)  # [max_len, num_bits]
        self.proj = nn.Linear(self.num_bits, embed_dim)

    def forward(self, positions: Tensor, seq_len: int | None = None) -> Tensor:
        """Encode the within-route positions.

        Args:
            positions: Long tensor ``[..., L]`` of within-route indices.
            seq_len: Route length ``L`` used for the cyclic ``i mod L`` reduction.  Defaults
                to ``positions.shape[-1]``.

        Returns:
            Tensor of shape ``[..., L, D]``.
        """
        if seq_len is None:
            seq_len = positions.shape[-1]
        idx = positions.long() % max(1, seq_len)
        codes = self.gray_codes[idx.clamp(max=self.max_len - 1)]  # [..., L, num_bits]
        return self.proj(codes)


class CycleFormerPE(nn.Module):
    """CycleFormer circular PE: a sinusoidal map of the index wrapped around the tour length.

    ``PE^Cyc_{2k}(v_i) = sin(2π (i mod L) / L · ω_k)`` and the cosine counterpart, with the
    same geometric frequency schedule as :class:`SinusoidalPE` (:cite:`yook2024cycleformer`).
    Reducing the index modulo ``L`` makes the encoding identical for ``i`` and ``i + L``
    regardless of the frequency schedule.

    Args:
        embed_dim: Embedding dimension ``D`` (assumed even).
        base: Wavelength base ``λ`` for the geometric schedule (default ``10000``).
    """

    def __init__(self, embed_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("freqs", _geometric_frequencies(embed_dim // 2, embed_dim, base))

    def forward(self, positions: Tensor, seq_len: int | None = None) -> Tensor:
        """Compute the circular sinusoidal encoding.

        Args:
            positions: Tensor ``[..., L]`` of within-route indices.
            seq_len: Tour length ``L`` for the ``2π i / L`` wrap.  Defaults to
                ``positions.shape[-1]``.

        Returns:
            Tensor of shape ``[..., L, D]``.
        """
        if seq_len is None:
            seq_len = positions.shape[-1]
        idx = positions.float() % float(max(1, seq_len))
        phase = 2.0 * math.pi * idx / float(max(1, seq_len))  # [..., L]
        angles = phase.unsqueeze(-1) * self.freqs  # [..., L, D/2]
        out = torch.zeros(
            *angles.shape[:-1], self.embed_dim, device=angles.device, dtype=angles.dtype
        )
        out[..., 0::2] = torch.sin(angles)
        out[..., 1::2] = torch.cos(angles[..., : out[..., 1::2].shape[-1]])
        return out


# ======================================================================================
# Route-graph helper + graph-transformer PEs
# ======================================================================================
def build_route_graph(
    routes: Tensor, num_nodes: int | None = None, pad_value: int | None = None
) -> Tensor:
    """Build a symmetric adjacency matrix from padded route permutations.

    Edges connect consecutive nodes on each route; the depot (index ``0``) connects to the
    route endpoints through the usual closed representation ``[0, c_1, ..., c_k, 0, ...]``.
    Self-loops (e.g. from zero-padding ``[..., c_k, 0, 0, 0]``) are dropped, and any arc
    touching ``pad_value`` (when given) is ignored.

    Args:
        routes: Long tensor of shape ``[..., R, Lr]`` (multiple routes per instance) or
            ``[..., Lr]`` (a single route per instance).
        num_nodes: Number of nodes ``N`` (including the depot).  Defaults to
            ``routes.max() + 1``.
        pad_value: Optional padding value to ignore (e.g. ``-1``).  If ``None``, only
            self-loops are removed.

    Returns:
        Symmetric adjacency tensor of shape ``[..., N, N]`` with ``{0, 1}`` entries and a
        zero diagonal.
    """
    routes = routes.long()
    # Normalize to shape [..., R, Lr] (a route axis is inserted for [..., Lr] inputs).
    if routes.dim() == 1:
        routes = routes.view(1, 1, -1)
    elif routes.dim() == 2:
        # Ambiguous [R, Lr] vs [B, Lr]; treat as [B, Lr] -> [B, 1, Lr]. A bare [R, Lr] still
        # yields the right graph since edges from all R rows are unioned anyway.
        routes = routes.unsqueeze(-2)
    # else: already [..., R, Lr]

    if num_nodes is None:
        num_nodes = int(routes.max().item()) + 1 if routes.numel() > 0 else 1
        if pad_value is not None and num_nodes - 1 == pad_value:
            num_nodes = int(routes[routes != pad_value].max().item()) + 1

    *batch, n_routes, route_len = routes.shape
    adj = torch.zeros(*batch, num_nodes, num_nodes, device=routes.device)
    if route_len < 2:
        return adj

    src = routes[..., :-1].reshape(*batch, -1)  # [..., R*(Lr-1)]
    dst = routes[..., 1:].reshape(*batch, -1)
    valid = src != dst
    if pad_value is not None:
        valid = valid & (src != pad_value) & (dst != pad_value)
    valid = valid & (src >= 0) & (dst >= 0) & (src < num_nodes) & (dst < num_nodes)

    src = src.clamp(0, num_nodes - 1)
    dst = dst.clamp(0, num_nodes - 1)
    w = valid.float()
    # Scatter into the adjacency for each batch element.
    flat_adj = adj.reshape(-1, num_nodes, num_nodes)
    flat_src = src.reshape(-1, src.shape[-1])
    flat_dst = dst.reshape(-1, dst.shape[-1])
    flat_w = w.reshape(-1, w.shape[-1])
    for b in range(flat_adj.shape[0]):
        flat_adj[b].index_put_((flat_src[b], flat_dst[b]), flat_w[b], accumulate=True)
        flat_adj[b].index_put_((flat_dst[b], flat_src[b]), flat_w[b], accumulate=True)
    adj = flat_adj.reshape(*batch, num_nodes, num_nodes)
    adj = (adj > 0).float()
    # zero the diagonal
    eye = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)
    adj = adj * (1.0 - eye)
    return adj


def _normalize_routes_or_adj(
    routes: Tensor | None, adj: Tensor | None, num_nodes: int | None, pad_value: int | None
) -> Tensor:
    if adj is None:
        if routes is None:
            raise ValueError("Provide either `routes` or `adj`.")
        adj = build_route_graph(routes, num_nodes=num_nodes, pad_value=pad_value)
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)
    return adj.float()


class LaplacianPE(nn.Module):
    """Laplacian PE (Lap. PE): the ``K`` smallest non-trivial Laplacian eigenvectors.

    For each instance the route graph is built (depot plus route arcs), the unnormalized
    Laplacian ``L = diag(deg) - A`` is formed, and the eigenvectors of the ``K`` smallest
    non-trivial eigenvalues are taken (:cite:`dwivedi2020generalization`).  Sign ambiguity is
    resolved by random flips during training (deterministic, no flips, in eval).  The result
    is zero-padded to ``D`` columns.

    Note: only ``K_eff = min(K, N - 1)`` columns can be genuinely non-trivial for a graph on
    ``N`` nodes; the remaining ``D - K_eff`` columns are zeros.

    Args:
        embed_dim: Output embedding dimension ``D``.
        k: Number of non-trivial eigenvectors to use (default ``8``).
    """

    def __init__(self, embed_dim: int, k: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.k = k

    def forward(
        self,
        routes: Tensor | None = None,
        adj: Tensor | None = None,
        num_nodes: int | None = None,
        pad_value: int | None = None,
    ) -> Tensor:
        """Compute the Laplacian eigenvector PE.

        Args:
            routes: Padded route permutations ``[..., R, Lr]`` or ``[..., Lr]``.
            adj: Alternatively, a pre-built symmetric adjacency ``[..., N, N]``.
            num_nodes: Number of nodes (passed to :func:`build_route_graph`).
            pad_value: Padding value to ignore (passed to :func:`build_route_graph`).

        Returns:
            Tensor of shape ``[..., N, D]``.
        """
        adj = _normalize_routes_or_adj(routes, adj, num_nodes, pad_value)
        n = adj.shape[-1]
        deg = adj.sum(-1)  # [..., N]
        lap = torch.diag_embed(deg) - adj  # [..., N, N]
        # eigh returns ascending eigenvalues
        _, eigvecs = torch.linalg.eigh(lap)  # eigvecs: [..., N, N]
        k_eff = min(self.k, max(0, n - 1))
        out = torch.zeros(*adj.shape[:-1], self.embed_dim, device=adj.device, dtype=eigvecs.dtype)
        if k_eff == 0:
            return out
        # drop the first (trivial) eigenvector, take the next k_eff
        vecs = eigvecs[..., 1 : 1 + k_eff]  # [..., N, k_eff]
        if self.training:
            signs = (
                torch.randint(0, 2, vecs.shape[:-2] + (1, k_eff), device=vecs.device).float() * 2
                - 1
            )
            vecs = vecs * signs
        out[..., :k_eff] = vecs
        return out


class RandomWalkSE(nn.Module):
    """Random-Walk Structural Encoding (RWSE): diagonals of random-walk matrix powers.

    ``PE^RWSE(v) = [(R^k)_{vv}]_{k=1}^{K}`` with ``R = D^{-1} A`` on the route graph
    (:cite:`dwivedi2021graph`), zero-padded to ``D``.  Isolated nodes (degree ``0``) get a
    zero row of ``R`` so their landing probabilities are ``0``.

    Args:
        embed_dim: Output embedding dimension ``D``.
        k: Number of random-walk steps ``K`` (default ``8``).
    """

    def __init__(self, embed_dim: int, k: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.k = k

    def forward(
        self,
        routes: Tensor | None = None,
        adj: Tensor | None = None,
        num_nodes: int | None = None,
        pad_value: int | None = None,
    ) -> Tensor:
        """Compute the random-walk structural encoding.

        Args:
            routes: Padded route permutations ``[..., R, Lr]`` or ``[..., Lr]``.
            adj: Alternatively, a pre-built symmetric adjacency ``[..., N, N]``.
            num_nodes: Number of nodes (passed to :func:`build_route_graph`).
            pad_value: Padding value to ignore (passed to :func:`build_route_graph`).

        Returns:
            Tensor of shape ``[..., N, D]`` with entries in ``[0, 1]``.
        """
        adj = _normalize_routes_or_adj(routes, adj, num_nodes, pad_value)
        n = adj.shape[-1]
        deg = adj.sum(-1, keepdim=True).clamp_min(1e-12)  # [..., N, 1]
        rw = adj / deg  # row-stochastic where degree > 0
        out = torch.zeros(*adj.shape[:-1], self.embed_dim, device=adj.device, dtype=adj.dtype)
        kmax = min(self.k, self.embed_dim)
        if kmax == 0 or n == 0:
            return out
        power = torch.eye(n, device=adj.device, dtype=adj.dtype).expand_as(adj).clone()
        for j in range(kmax):
            power = power @ rw
            out[..., j] = torch.diagonal(power, dim1=-2, dim2=-1)
        return out.clamp(0.0, 1.0)


class ShortestPathBias(nn.Module):
    """Shortest-path-distance attention bias (SPD): a learnable scalar per integer distance.

    ``logit(i, j) += b_{spd(v_i, v_j)}`` with a learnable scalar bias per integer
    shortest-path distance on the route graph (:cite:`ying2021do`), capped at ``max_spd``
    (unreachable pairs are clamped to ``max_spd``).

    Args:
        max_spd: Maximum shortest-path distance the bias table indexes (default ``16``).
    """

    def __init__(self, max_spd: int = 16) -> None:
        super().__init__()
        self.max_spd = max_spd
        self.bias = nn.Embedding(max_spd + 1, 1)

    @staticmethod
    def _all_pairs_shortest_paths(adj: Tensor) -> Tensor:
        # adj: [..., N, N] with {0, 1}. Floyd-Warshall with a large constant for unreachable.
        n = adj.shape[-1]
        big = float(n + 1)
        dist = torch.where(adj > 0, torch.ones_like(adj), torch.full_like(adj, big))
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).bool()
        dist = dist.masked_fill(eye, 0.0)
        for k in range(n):
            dist = torch.minimum(dist, dist[..., :, k : k + 1] + dist[..., k : k + 1, :])
        return dist  # [..., N, N], entries in {0, 1, ..., n-1} or `big` if unreachable

    def forward(
        self,
        routes: Tensor | None = None,
        adj: Tensor | None = None,
        num_nodes: int | None = None,
        pad_value: int | None = None,
    ) -> Tensor:
        """Compute the ``[..., N, N]`` SPD bias added to attention logits.

        Args:
            routes: Padded route permutations ``[..., R, Lr]`` or ``[..., Lr]``.
            adj: Alternatively, a pre-built symmetric adjacency ``[..., N, N]``.
            num_nodes: Number of nodes (passed to :func:`build_route_graph`).
            pad_value: Padding value to ignore (passed to :func:`build_route_graph`).

        Returns:
            Tensor of shape ``[..., N, N]``.  It is symmetric and the ``spd = 0`` (diagonal)
            entries all share the single learnable bias ``b_0``.
        """
        adj = _normalize_routes_or_adj(routes, adj, num_nodes, pad_value)
        dist = self._all_pairs_shortest_paths(adj)
        idx = dist.clamp(0, self.max_spd).round().long()
        return self.bias(idx).squeeze(-1)


# ======================================================================================
# Proposed method (paper §4): IPE, XPE, Hierarchical
# ======================================================================================
class InRoutePE(nn.Module):
    """In-route PE (IPE): distance-indexed multi-frequency sinusoid (paper §4.1).

    For a closed route ``r = (v_1, ..., v_L)`` with ``v_1 = v_L = depot``, define the
    cumulative travel distance ``d_i = Σ_{j=2}^{i} ||x_{v_j} - x_{v_{j-1}}||`` and rescale it
    to one period ``d̂_i = 2π d_i / d_L ∈ [0, 2π]``.  Then

    * ``direction_aware=True``  → ``[sin(ω_k d̂_i), cos(ω_k d̂_i)]_{k=0}^{D/2-1}`` (distinguishes
      a route from its reversal: reversal maps ``d_i → d_L - d_i``, flipping the sine half).
    * ``direction_aware=False`` → ``[cos(ω_k d̂_i)]_{k=0}^{D-1}`` (invariant to route reversal
      by construction, since ``cos(2π - x) = cos(x)`` with integer frequencies).

    Implementation note: the paper writes ``ω_k = λ^{-2k/D}`` (geometric), but the stated
    properties ``IPE(v_1) = IPE(v_L)`` and the reversal (in)variance only hold for *integer*
    frequencies.  This module therefore uses integer harmonics ``ω_k = k + 1``.

    Args:
        embed_dim: Output embedding dimension ``D``.
        direction_aware: Whether to use the sin/cos variant (``True``) or the cosine-only
            reversal-invariant variant (``False``).
        eps: Small constant guarding the ``d_L`` division for degenerate (zero-length) routes.
    """

    def __init__(self, embed_dim: int, direction_aware: bool = False, eps: float = 1e-9) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.direction_aware = direction_aware
        self.eps = eps
        if direction_aware:
            num_freq = embed_dim // 2
        else:
            num_freq = embed_dim
        self.register_buffer("freqs", torch.arange(1, num_freq + 1, dtype=torch.float32))

    def forward(self, coords: Tensor) -> Tensor:
        """Compute the in-route PE for route-ordered coordinates.

        Args:
            coords: Route-ordered coordinates of shape ``[..., L, 2]``.  The route is assumed
                closed (``coords[..., 0, :] == coords[..., -1, :] == depot``).

        Returns:
            Tensor of shape ``[..., L, D]``.
        """
        seg = torch.linalg.norm(coords[..., 1:, :] - coords[..., :-1, :], dim=-1)  # [..., L-1]
        cum = torch.cumsum(seg, dim=-1)  # [..., L-1]
        zero = torch.zeros(*cum.shape[:-1], 1, device=cum.device, dtype=cum.dtype)
        d = torch.cat([zero, cum], dim=-1)  # [..., L], d_1 = 0
        d_total = d[..., -1:].clamp_min(self.eps)  # [..., 1]
        d_hat = 2.0 * math.pi * d / d_total  # [..., L]
        angles = d_hat.unsqueeze(-1) * self.freqs  # [..., L, num_freq]
        if self.direction_aware:
            out = torch.zeros(
                *angles.shape[:-1], self.embed_dim, device=angles.device, dtype=angles.dtype
            )
            out[..., 0::2] = torch.sin(angles)
            out[..., 1::2] = torch.cos(angles[..., : out[..., 1::2].shape[-1]])
            return out
        return torch.cos(angles)  # [..., L, D]


class CrossRoutePE(nn.Module):
    """Cross-route PE (XPE): depot-anchored polar-angle sinusoid (paper §4.2).

    For node ``v`` with coordinate ``x_v`` and depot ``x_0``, define
    ``θ_v = atan2(y_v - y_0, x_v - x_0) ∈ [-π, π)`` and encode

    ``XPE(v) = [sin(ω'_k θ_v), cos(ω'_k θ_v)]_{k=0}^{K-1}``,  ``ω'_k = 2^k``,

    zero-padded from ``2K`` to ``D`` columns (requires ``2K ≤ D``).  A customer coincident
    with the depot yields ``atan2(0, 0) = 0`` (acceptable).

    Args:
        embed_dim: Output embedding dimension ``D``.
        k: Number of frequency bands ``K`` (default ``4``; must satisfy ``2K ≤ D``).
    """

    def __init__(self, embed_dim: int, k: int = 4) -> None:
        super().__init__()
        if 2 * k > embed_dim:
            raise ValueError(
                f"CrossRoutePE requires 2*k <= embed_dim, got k={k}, embed_dim={embed_dim}"
            )
        self.embed_dim = embed_dim
        self.k = k
        self.register_buffer("freqs", 2.0 ** torch.arange(k, dtype=torch.float32))

    def forward(self, coords: Tensor, depot: Tensor) -> Tensor:
        """Compute the cross-route PE.

        Args:
            coords: Node coordinates of shape ``[..., 2]`` or ``[..., L, 2]``.
            depot: Depot coordinate of shape ``[..., 2]`` (broadcast against ``coords``).

        Returns:
            Tensor of shape ``coords.shape[:-1] + (D,)``.
        """
        depot_b = depot
        while depot_b.dim() < coords.dim():
            depot_b = depot_b.unsqueeze(-2)
        rel = coords - depot_b  # [..., 2]
        theta = torch.atan2(rel[..., 1], rel[..., 0])  # [...]
        angles = theta.unsqueeze(-1) * self.freqs  # [..., K]
        out = torch.zeros(*theta.shape, self.embed_dim, device=coords.device, dtype=coords.dtype)
        out[..., 0 : 2 * self.k : 2] = torch.sin(angles)
        out[..., 1 : 2 * self.k : 2] = torch.cos(angles)
        return out


class HierarchicalPE(nn.Module):
    """Hierarchical PE: per-node concatenation ``[IPE(v) ‖ XPE(v)]`` (paper §4.3, pre-projection).

    This produces the positional part of the §4.3 fusion ``FF([x_v ‖ IPE(v) ‖ XPE(v)])``;
    the raw-coordinate concatenation and the shared feed-forward projection itself are out of
    scope for this module.  The output width is ``2 * embed_dim`` (``IPE`` width ``embed_dim``
    plus ``XPE`` width ``embed_dim``).

    Args:
        embed_dim: Per-component embedding dimension ``D``; the output width is ``2D``.
        direction_aware: Forwarded to :class:`InRoutePE`.
        xpe_k: Forwarded to :class:`CrossRoutePE` as ``k`` (default ``4``).
    """

    def __init__(self, embed_dim: int, direction_aware: bool = False, xpe_k: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = 2 * embed_dim
        self.ipe = InRoutePE(embed_dim, direction_aware=direction_aware)
        self.xpe = CrossRoutePE(embed_dim, k=xpe_k)

    def forward(self, coords: Tensor, depot: Tensor | None = None) -> Tensor:
        """Compute the concatenated hierarchical PE for a route.

        Args:
            coords: Route-ordered coordinates ``[..., L, 2]`` (closed route; index ``0`` is
                the depot).
            depot: Optional depot coordinate ``[..., 2]``.  Defaults to ``coords[..., 0, :]``.

        Returns:
            Tensor of shape ``[..., L, 2D]``.
        """
        if depot is None:
            depot = coords[..., 0, :]
        ipe = self.ipe(coords)  # [..., L, D]
        xpe = self.xpe(coords, depot)  # [..., L, D]
        return torch.cat([ipe, xpe], dim=-1)


# ======================================================================================
# Factory
# ======================================================================================
_PE_REGISTRY: dict[str, type[nn.Module]] = {
    "APE": AbsolutePE,
    "SIN": SinusoidalPE,
    "RoPE": RotaryPE,
    "RPE": RelativePE,
    "ALiBi": ALiBiBias,
    "LapPE": LaplacianPE,
    "RWSE": RandomWalkSE,
    "SPD": ShortestPathBias,
    "DACT": DACTCyclicPE,
    "CycleFormer": CycleFormerPE,
    "IPE": InRoutePE,
    "XPE": CrossRoutePE,
    "Hierarchical": HierarchicalPE,
}


def get_positional_encoding(name: str, **kwargs) -> nn.Module:
    """Instantiate a positional-encoding module by name.

    Args:
        name: One of ``"APE" | "SIN" | "RoPE" | "RPE" | "ALiBi" | "LapPE" | "RWSE" | "SPD" |
            "DACT" | "CycleFormer" | "IPE" | "XPE" | "Hierarchical"``.
        **kwargs: Forwarded to the selected class constructor.

    Returns:
        The instantiated :class:`torch.nn.Module`.

    Raises:
        ValueError: If ``name`` is not a registered positional encoding.
    """
    if name not in _PE_REGISTRY:
        raise ValueError(
            f"Unknown positional encoding '{name}'. Available positional encodings: "
            f"{list(_PE_REGISTRY.keys())}"
        )
    return _PE_REGISTRY[name](**kwargs)
