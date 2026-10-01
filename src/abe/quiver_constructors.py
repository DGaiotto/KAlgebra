"""Tight `BPSKAlgebra` constructors for graph-shape quivers of `U(N)`
factors.

Concrete constructors:

  * `LinearUQuiver(N_list, M_list)` -- linear `A`-shape chain.
  * `GraphUQuiver(N_list, M_list, edges)` -- general undirected graph
    of bifundamental edges; the workhorse the others delegate to.
  * `CircularUQuiver(N_list, M_list)` -- affine `Â`-shape cycle (k ≥ 2;
    k = 2 is a doubled bifundamental between two nodes).
  * `DShapeUQuiver(N_list, M_list)` -- D-Dynkin shape (chain with one
    extra branch off the second-to-last node).
  * `E6UQuiver` / `E7UQuiver` / `E8UQuiver` -- E-Dynkin shapes.

All shapes are *graphs of unitary gauge factors*, distinct from
ADE-type gauge groups (which would be a single SO / E_n at one node;
those are `pure_ade.PureADE.D_N` / `.E_6` etc.).

Construction recipe (per graph topology):

  * Per-factor structure is delegated to `pure_ade.UN_Nf` (which
    bakes in the `(2N − 1)`-fund-Wilson-line support and the
    closed-form pure-gauge spec).
  * Each bifundamental edge contributes a `pure_ade._bifund_pair_order_AN`
    parity-chain-ordered block of `(2N_i − 1)(2N_j − 1)` charges, with
    a fresh flavour slot per edge.
  * Block-diagonal pairing matrix from per-factor `B`'s (bifund
    flavour slots are central).
  * Cone witness assembled in closed form from per-factor witnesses
    + bifund-direction adjustments.
  * Spec audited at construction by
    `BPSQuiver.verify_spectrum_generator`; failure raises rather
    than silently producing a malformed algebra.  For linear and
    affine (cyclic / D / E) shapes covered by published recipes this
    succeeds; future shapes may require tuning the spec ordering and
    will fail loudly when they do.

Asymptotic freedom (`strict=True`, default): at every gauge node `i`,

    Σ_{j ~ i} N_j  +  M_i  ≤  2 N_i,

where the sum runs over edges incident to `i` (counted with
multiplicity, so a doubled bifundamental contributes `2 N_j`).

No `frozen=` anywhere -- the new `BPSKAlgebra` doesn't carry that
concept.
"""

from __future__ import annotations

from typing import Sequence

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bps_kalgebra import BPSKAlgebra
import pure_ade as _pa
import bps_quiver_tools as _bps


Vec = tuple[int, ...]


# ---------------------------------------------------------------------------
# LinearUQuiver
# ---------------------------------------------------------------------------


class LinearUQuiver:
    """Linear `A`-shape quiver `U(N_1) − ⋯ − U(N_k)` with fundamentals
    `M_i` at each gauge node and one bifundamental between every
    consecutive pair.

    Constructor builds and validates a `BPSKAlgebra` (`self.algebra`)
    whose spec is the parity-chain-ordered concatenation of
    bifundamental blocks followed by per-factor pure-gauge specs.

    Asymp-freedom gate (`strict=True`, default): for every gauge node
    `i ∈ {1, …, k}`, with `N_0 = N_{k+1} = 0`,

        N_{i−1} + M_i + N_{i+1}  ≤  2 N_i.

    Public attributes (all set after `__init__`):

      * `algebra: BPSKAlgebra` — the constructed algebra.
      * `B`, `nodes`, `spec`, `cone_witness` — defining data.
      * `factors: list[pure_ade.UN_Nf]` — per-node theories.
      * `gauge_node_indices: list[int]` — indices of pure-gauge nodes
        in `nodes` (one block of `2 N_i` per factor).
      * `fund_node_indices: dict[int, list[int]]` — per-factor list of
        fundamental-matter node indices.
      * `bifund_node_indices: list[int]` — indices of bifundamental
        matter nodes.
      * `matter_node_indices: list[int]` — union of fund + bifund;
        candidate set for an RG flow that integrates out matter.
    """

    def __init__(
        self,
        N_list: Sequence[int],
        M_list: Sequence[int] | None = None,
        *,
        strict: bool = True,
        antisym_counts: Sequence[int] | None = None,
        antisym_sides: Sequence[Sequence[str]] | None = None,
        fund_sides: Sequence[Sequence[str]] | None = None,
        # New per-block bps_aut kwargs (BPS-quiver Z_2 automorphism flags).
        # Each fund / antisym / bifund-leg matter factor independently
        # has a bool indicating whether σ is applied to that factor's
        # charges, yielding a distinct BPS-quiver presentation of the
        # same physical theory.
        antisym_bps_aut: Sequence[Sequence[bool]] | None = None,
        fund_bps_aut: Sequence[Sequence[bool]] | None = None,
        bifund_bps_aut: Sequence[tuple[bool, bool]] | None = None,
    ):
        # ---- validation -------------------------------------------------
        N_list = [int(x) for x in N_list]
        if M_list is None:
            M_list = [0] * len(N_list)
        else:
            M_list = [int(x) for x in M_list]
        if len(N_list) != len(M_list):
            raise ValueError(
                f"len(N_list) = {len(N_list)} != len(M_list) = {len(M_list)}"
            )
        k = len(N_list)
        if k < 1:
            raise ValueError("LinearUQuiver: need at least one gauge factor")
        for i, N in enumerate(N_list):
            if N < 1:
                raise ValueError(
                    f"LinearUQuiver: N_list[{i}] = {N}; each N_i ≥ 1 required"
                )
        for i, M in enumerate(M_list):
            if M < 0:
                raise ValueError(
                    f"LinearUQuiver: M_list[{i}] = {M}; each M_i ≥ 0 required"
                )
        if antisym_counts is None:
            antisym_counts_norm = [0] * k
        else:
            antisym_counts_norm = [int(x) for x in antisym_counts]
            if len(antisym_counts_norm) != k:
                raise ValueError(
                    f"LinearUQuiver: len(antisym_counts) = "
                    f"{len(antisym_counts_norm)} != k = {k}"
                )
            for i, c in enumerate(antisym_counts_norm):
                if c < 0:
                    raise ValueError(
                        f"LinearUQuiver: antisym_counts[{i}] = {c}; each ≥ 0"
                    )
                if c > 0 and N_list[i] < 2:
                    raise ValueError(
                        f"LinearUQuiver: antisym^2 of U({N_list[i]}) at "
                        f"node {i} is trivial / undefined (need N_i ≥ 2)"
                    )
        if strict:
            N_ext = [0] + list(N_list) + [0]
            for i in range(k):
                antisym_contrib = (N_list[i] - 2) * antisym_counts_norm[i]
                lhs = N_ext[i] + M_list[i] + N_ext[i + 2] + antisym_contrib
                rhs = 2 * N_list[i]
                if lhs > rhs:
                    raise ValueError(
                        f"asymptotic freedom violated at node {i + 1} "
                        f"(U({N_list[i]})): N_left + M + N_right + "
                        f"(N-2)*n_a = {N_ext[i]} + {M_list[i]} + "
                        f"{N_ext[i+2]} + {antisym_contrib} = {lhs} > "
                        f"2 N_i = {rhs}.  Pass `strict=False` to bypass."
                    )

        self.k = k
        self.N_list: tuple[int, ...] = tuple(N_list)
        self.M_list: tuple[int, ...] = tuple(M_list)
        self.antisym_counts: tuple[int, ...] = tuple(antisym_counts_norm)

        # ---- per-factor antisym_sides / fund_sides ----------------------
        if antisym_sides is None:
            antisym_sides_norm = [["left"] * c for c in antisym_counts_norm]
        else:
            antisym_sides_norm = [list(s) for s in antisym_sides]
            if len(antisym_sides_norm) != k:
                raise ValueError("antisym_sides length != k")
            for i, s in enumerate(antisym_sides_norm):
                if len(s) != antisym_counts_norm[i]:
                    raise ValueError(
                        f"antisym_sides[{i}] length != antisym_counts[{i}]"
                    )
        if fund_sides is None:
            fund_sides_norm = [["left"] * M for M in M_list]
        else:
            fund_sides_norm = [list(s) for s in fund_sides]
            if len(fund_sides_norm) != k:
                raise ValueError("fund_sides length != k")
            for i, s in enumerate(fund_sides_norm):
                if len(s) != M_list[i]:
                    raise ValueError(
                        f"fund_sides[{i}] length != M_list[{i}]"
                    )
        # ---- per-factor antisym_bps_aut / fund_bps_aut flags ----------
        n_bifund_local = k - 1
        if antisym_bps_aut is None:
            antisym_bps_aut_norm = [[False] * c for c in antisym_counts_norm]
        else:
            antisym_bps_aut_norm = [list(s) for s in antisym_bps_aut]
            if len(antisym_bps_aut_norm) != k:
                raise ValueError(
                    f"LinearUQuiver: len(antisym_bps_aut) = "
                    f"{len(antisym_bps_aut_norm)} != k = {k}"
                )
            for i, s in enumerate(antisym_bps_aut_norm):
                if len(s) != antisym_counts_norm[i]:
                    raise ValueError(
                        f"antisym_bps_aut[{i}] length {len(s)} != "
                        f"antisym_counts[{i}] = {antisym_counts_norm[i]}"
                    )
                for x in s:
                    if not isinstance(x, bool):
                        raise ValueError(
                            f"antisym_bps_aut[{i}] entries must be bool "
                            f"(got {x!r})"
                        )
        if fund_bps_aut is None:
            fund_bps_aut_norm = [[False] * M for M in M_list]
        else:
            fund_bps_aut_norm = [list(s) for s in fund_bps_aut]
            if len(fund_bps_aut_norm) != k:
                raise ValueError(
                    f"LinearUQuiver: len(fund_bps_aut) = "
                    f"{len(fund_bps_aut_norm)} != k = {k}"
                )
            for i, s in enumerate(fund_bps_aut_norm):
                if len(s) != M_list[i]:
                    raise ValueError(
                        f"fund_bps_aut[{i}] length {len(s)} != "
                        f"M_list[{i}] = {M_list[i]}"
                    )
                for x in s:
                    if not isinstance(x, bool):
                        raise ValueError(
                            f"fund_bps_aut[{i}] entries must be bool "
                            f"(got {x!r})"
                        )
        if bifund_bps_aut is None:
            bifund_bps_aut_norm = [(False, False)] * n_bifund_local
        else:
            bifund_bps_aut_norm = [tuple(e) for e in bifund_bps_aut]
            if len(bifund_bps_aut_norm) != n_bifund_local:
                raise ValueError(
                    f"LinearUQuiver: len(bifund_bps_aut) = "
                    f"{len(bifund_bps_aut_norm)} != n_bifund = {n_bifund_local}"
                )
            for ei, t in enumerate(bifund_bps_aut_norm):
                if len(t) != 2 or not all(isinstance(x, bool) for x in t):
                    raise ValueError(
                        f"bifund_bps_aut[{ei}] must be a (bool, bool) tuple "
                        f"(got {t!r})"
                    )

        self.antisym_sides = antisym_sides_norm
        self.fund_sides = fund_sides_norm
        self.antisym_bps_aut = antisym_bps_aut_norm
        self.fund_bps_aut = fund_bps_aut_norm
        self.bifund_bps_aut = bifund_bps_aut_norm

        # ---- per-factor theories : UN_antisym (handles both kinds of
        #      matter; reduces to UN_Nf when n_a = 0).  AF already
        #      validated globally above.  bps_aut flags flow through
        #      to UN_antisym's per-block bps_aut_applied kwargs.
        def _factor(fi, N, M):
            kwargs = dict(strict=False)
            n_a = antisym_counts_norm[fi]
            # Antisym sides + bps_aut.
            if any(antisym_bps_aut_norm[fi]):
                kwargs["antisym_positions"] = [
                    "before" if s == "left" else "after"
                    for s in antisym_sides_norm[fi]
                ]
                kwargs["antisym_bps_aut_applied"] = list(antisym_bps_aut_norm[fi])
            elif n_a > 0:
                kwargs["antisym_sides"] = antisym_sides_norm[fi]
            # Fund sides + bps_aut.
            if any(fund_bps_aut_norm[fi]):
                kwargs["fund_positions"] = [
                    "before" if s == "left" else "after"
                    for s in fund_sides_norm[fi]
                ]
                kwargs["fund_bps_aut_applied"] = list(fund_bps_aut_norm[fi])
            elif M > 0:
                kwargs["fund_sides"] = fund_sides_norm[fi]
            return _pa.UN_antisym(N, n_a=n_a, Nf=M, **kwargs)
        factors = [
            _factor(fi, N, M) for fi, (N, M) in enumerate(zip(N_list, M_list))
        ]
        self.factors = factors
        r_full = [len(f.B) for f in factors]            # 2 N_i + M_i
        factor_start = [0]
        for r in r_full:
            factor_start.append(factor_start[-1] + r)
        gauge_dim = factor_start[-1]                     # ambient offset where bifunds start
        n_bifund = k - 1
        total = gauge_dim + n_bifund

        # ---- block-diagonal exchange matrix ----------------------------
        B = [[0] * total for _ in range(total)]
        for fi, f in enumerate(factors):
            s = factor_start[fi]
            for i in range(r_full[fi]):
                for j in range(r_full[fi]):
                    B[s + i][s + j] = int(f.B[i][j])
        # Bifund slots are central (rows / cols zero) -- they live in ker(B).

        def _pad(v: Sequence[int], start: int) -> Vec:
            out = [0] * total
            for i, x in enumerate(v):
                out[start + i] = int(x)
            return tuple(out)

        nodes_per_factor = [
            [_pad(g, factor_start[fi]) for g in factors[fi].nodes]
            for fi in range(k)
        ]
        spec_per_factor = [
            [_pad(g, factor_start[fi]) for g in factors[fi].spec]
            for fi in range(k)
        ]

        # ---- bifundamental blocks (parity-chain ordering) --------------
        bifund_specs: list[list[Vec]] = []
        bifund_nodes: list[Vec] = []
        for edge in range(n_bifund):
            Ni, Nj = N_list[edge], N_list[edge + 1]
            alphas_i = _pa._fundamental_wilson_support_UN(Ni)
            alphas_j = _pa._fundamental_wilson_support_UN(Nj)
            n_i, n_j = 2 * Ni - 1, 2 * Nj - 1
            pair_order = _pa._bifund_pair_order_AN(n_i, n_j)
            si = factor_start[edge]
            sj = factor_start[edge + 1]
            mu_slot = gauge_dim + edge

            def _bifund(a, b, *,
                        si=si, sj=sj, mu_slot=mu_slot,
                        alphas_i=alphas_i, alphas_j=alphas_j) -> Vec:
                out = [0] * total
                for kk, x in enumerate(alphas_i[a]):
                    out[si + kk] = int(x)
                for kk, x in enumerate(alphas_j[b]):
                    out[sj + kk] = int(x)
                out[mu_slot] = 1
                return tuple(out)

            block = [_bifund(a, b) for (a, b) in pair_order]
            bifund_node = _bifund(0, 0)
            # Per-leg bps_aut application.
            ba_i, ba_j = bifund_bps_aut_norm[edge]
            if ba_i:
                block = [
                    _pa.apply_bps_aut_to_charge_UN(c, Ni, gauge_offset=si)
                    for c in block
                ]
                bifund_node = _pa.apply_bps_aut_to_charge_UN(
                    bifund_node, Ni, gauge_offset=si,
                )
            if ba_j:
                block = [
                    _pa.apply_bps_aut_to_charge_UN(c, Nj, gauge_offset=sj)
                    for c in block
                ]
                bifund_node = _pa.apply_bps_aut_to_charge_UN(
                    bifund_node, Nj, gauge_offset=sj,
                )
            bifund_specs.append(block)
            bifund_nodes.append(bifund_node)
        self.bifund_specs = bifund_specs
        self.bifund_nodes = bifund_nodes

        # ---- assemble nodes / spec -------------------------------------
        nodes_UV: list[Vec] = []
        for fi in range(k):
            nodes_UV.extend(nodes_per_factor[fi])
        nodes_UV.extend(bifund_nodes)

        spec_UV: list[Vec] = []
        for block in bifund_specs:
            spec_UV.extend(block)
        for fi in range(k):
            spec_UV.extend(spec_per_factor[fi])

        # ---- node-index registry --------------------------------------
        # UN_antisym per-factor node order:
        #     pure_nodes (2(N-1)) , antisym_nodes (n_a) , fund_nodes (Nf)
        # Bifund matter nodes come after all factors.
        gauge_idx: list[int] = []
        antisym_idx: dict[int, list[int]] = {}
        fund_idx: dict[int, list[int]] = {}
        node_offset = 0
        for fi, (Ni, Mi) in enumerate(zip(N_list, M_list)):
            n_ai = antisym_counts_norm[fi]
            n_pure_nodes = len(factors[fi].nodes) - n_ai - Mi
            for i in range(n_pure_nodes):
                gauge_idx.append(node_offset + i)
            for i in range(n_ai):
                antisym_idx.setdefault(fi, []).append(
                    node_offset + n_pure_nodes + i
                )
            for i in range(Mi):
                fund_idx.setdefault(fi, []).append(
                    node_offset + n_pure_nodes + n_ai + i
                )
            node_offset += n_pure_nodes + n_ai + Mi
        bifund_idx = list(range(node_offset, node_offset + n_bifund))
        matter_idx = (
            [j for fi in range(k) for j in antisym_idx.get(fi, [])]
            + [j for fi in range(k) for j in fund_idx.get(fi, [])]
            + bifund_idx
        )

        self.gauge_node_indices = gauge_idx
        self.antisym_node_indices = antisym_idx
        self.fund_node_indices = fund_idx
        self.bifund_node_indices = bifund_idx
        self.matter_node_indices = matter_idx

        # ---- closed-form cone witness ---------------------------------
        f_lin: list[int] = []
        for f in factors:
            f_lin.extend(int(x) for x in f.cone_witness)
        f_lin.extend([0] * n_bifund)
        for edge in range(n_bifund):
            bn = bifund_nodes[edge]
            val = sum(fi_ * gi for fi_, gi in zip(f_lin, bn))
            f_lin[gauge_dim + edge] = max(1 - val, 0)
        self.cone_witness = tuple(f_lin)

        self.B = B
        self.nodes = nodes_UV
        self.spec = spec_UV

        # ---- audit: verify spec is a green negating sequence -----------
        Q = _bps.BPSQuiver.from_pairing(
            [list(g) for g in nodes_UV],
            [list(row) for row in B],
        )
        ok, info = Q.verify_spectrum_generator(spec_UV)
        if not ok:
            raise RuntimeError(
                f"LinearUQuiver({N_list}, {M_list}): assembled spec is "
                f"not a valid negating sequence on the BPS quiver. "
                f"This is an internal bug, not a user error.  Info: {info}"
            )

        # ---- final BPSKAlgebra ----------------------------------------
        self.algebra = BPSKAlgebra(
            pairing=B,
            node_charges=nodes_UV,
            spec=spec_UV,
            cone_witness=self.cone_witness,
            verify="off",   # already verified the green-spec condition
        )

    # ----- public API surface ------------------------------------------

    def verify(self) -> bool:
        """Re-verify the spec on demand.  Returns False if the BPS-quiver
        spectrum-generator check fails (shouldn't happen post-construction)."""
        Q = _bps.BPSQuiver.from_pairing(
            [list(g) for g in self.nodes],
            [list(row) for row in self.B],
        )
        ok, _ = Q.verify_spectrum_generator(self.spec)
        return ok

    def __repr__(self) -> str:
        m_part = (
            f", M={list(self.M_list)}"
            if any(m > 0 for m in self.M_list) else ""
        )
        return f"LinearUQuiver(N={list(self.N_list)}{m_part})"


# ---------------------------------------------------------------------------
# GraphUQuiver -- general undirected graph of bifundamental edges
# ---------------------------------------------------------------------------


def _validate_graph_inputs(
    N_list: Sequence[int],
    M_list: Sequence[int] | None,
    edges: Sequence[tuple[int, int]],
    *,
    strict: bool,
    label: str,
    antisym_counts: Sequence[int] | None = None,
):
    """Shared validation for `GraphUQuiver` and its specialisations.

    Returns the normalised `(N_list, M_list, edges_canonical,
    antisym_counts)`.  Each edge is canonicalised to
    `(min(i, j), max(i, j))` so multi-edges show up as duplicates of
    the same canonical pair.

    Asymp-freedom gate: at every node `i`,

        Σ_{e ∼ i} N_{other(e)}  +  M_i  +  (N_i − 2) · n_a[i]   ≤   2 N_i

    where `n_a[i]` is the number of antisymmetric-tensor hypers at node
    `i` (each contributing `(N_i − 2)` to the matter count via
    `T(Λ²(fund)) / T(fund) = N_i − 2`).
    """
    N_list = [int(x) for x in N_list]
    if M_list is None:
        M_list = [0] * len(N_list)
    else:
        M_list = [int(x) for x in M_list]
    if len(N_list) != len(M_list):
        raise ValueError(
            f"{label}: len(N_list) = {len(N_list)} != "
            f"len(M_list) = {len(M_list)}"
        )
    k = len(N_list)
    if k < 1:
        raise ValueError(f"{label}: need at least one gauge factor")
    for i, N in enumerate(N_list):
        if N < 1:
            raise ValueError(
                f"{label}: N_list[{i}] = {N}; each N_i ≥ 1 required"
            )
    for i, M in enumerate(M_list):
        if M < 0:
            raise ValueError(
                f"{label}: M_list[{i}] = {M}; each M_i ≥ 0 required"
            )
    if antisym_counts is None:
        antisym_counts_norm = [0] * k
    else:
        antisym_counts_norm = [int(x) for x in antisym_counts]
        if len(antisym_counts_norm) != k:
            raise ValueError(
                f"{label}: len(antisym_counts) = "
                f"{len(antisym_counts_norm)} != k = {k}"
            )
        for i, c in enumerate(antisym_counts_norm):
            if c < 0:
                raise ValueError(
                    f"{label}: antisym_counts[{i}] = {c}; "
                    f"each entry must be ≥ 0"
                )
            if c > 0 and N_list[i] < 2:
                raise ValueError(
                    f"{label}: antisym^2 of U({N_list[i]}) at node {i} "
                    f"is trivial / undefined (need N_i ≥ 2)"
                )
    edges_canon: list[tuple[int, int]] = []
    for ei, (i, j) in enumerate(edges):
        i, j = int(i), int(j)
        if i == j:
            raise ValueError(
                f"{label}: edges[{ei}] = ({i}, {j}) is a self-loop; "
                f"self-loops are not supported"
            )
        if not (0 <= i < k) or not (0 <= j < k):
            raise ValueError(
                f"{label}: edges[{ei}] = ({i}, {j}) out of range "
                f"[0, {k})"
            )
        if i > j:
            i, j = j, i
        edges_canon.append((i, j))
    if strict:
        for node in range(k):
            incident_N = 0
            for (i, j) in edges_canon:
                if i == node:
                    incident_N += N_list[j]
                elif j == node:
                    incident_N += N_list[i]
            antisym_contrib = (N_list[node] - 2) * antisym_counts_norm[node]
            lhs = incident_N + M_list[node] + antisym_contrib
            rhs = 2 * N_list[node]
            if lhs > rhs:
                raise ValueError(
                    f"{label}: asymptotic freedom violated at node "
                    f"{node} (U({N_list[node]})): incident-N + M + "
                    f"(N-2)*n_a = {incident_N} + {M_list[node]} + "
                    f"{antisym_contrib} = {lhs} > 2 N = {rhs}.  "
                    f"Pass strict=False to bypass."
                )
    return N_list, M_list, edges_canon, antisym_counts_norm


class GraphUQuiver:
    """Unitary-quiver gauge theory on an arbitrary undirected graph of
    bifundamental edges.

    `LinearUQuiver`, `CircularUQuiver`, `DShapeUQuiver`, and the
    `E6/E7/E8UQuiver` constructors are convenience wrappers that
    populate a specific edge set and delegate here.

    Parameters
    ----------
    N_list
        Gauge ranks `[N_1, ..., N_k]`.
    M_list
        Per-node fundamental counts (default: zero).
    edges
        Iterable of `(i, j)` pairs (0-indexed into `N_list`).
        Self-loops are rejected.  Multi-edges are allowed (= multiple
        bifundamentals between the same pair) and contribute multiple
        flavour slots and bifund blocks.
    strict
        Asymp-freedom enforcement at each node, summing incident
        `N_j`'s with multiplicity.

    Public attributes (all set after `__init__`):
      * `algebra: BPSKAlgebra`,
      * `B`, `nodes`, `spec`, `cone_witness`,
      * `factors: list[pure_ade.UN_Nf]` per gauge node,
      * `edges`: canonicalised list of `(i, j)` with `i ≤ j`,
      * `gauge_node_indices`, `fund_node_indices`,
        `bifund_node_indices`, `matter_node_indices` -- node-list
        positions for the four roles.
    """

    def __init__(
        self,
        N_list: Sequence[int],
        M_list: Sequence[int] | None = None,
        edges: Sequence[tuple[int, int]] = (),
        *,
        strict: bool = True,
        fund_sides: Sequence[Sequence[str]] | None = None,
        bifund_placements: Sequence[str] | None = None,
        antisym_counts: Sequence[int] | None = None,
        antisym_sides: Sequence[Sequence[str]] | None = None,
        # Per-block BPS-quiver Z_2 automorphism flags.
        antisym_bps_aut: Sequence[Sequence[bool]] | None = None,
        fund_bps_aut: Sequence[Sequence[bool]] | None = None,
        bifund_bps_aut: Sequence[tuple[bool, bool]] | None = None,
    ):
        N_list, M_list, edges, antisym_counts_norm = _validate_graph_inputs(
            N_list, M_list, edges,
            strict=strict, label=type(self).__name__,
            antisym_counts=antisym_counts,
        )
        self.k = len(N_list)
        self.N_list: tuple[int, ...] = tuple(N_list)
        self.M_list: tuple[int, ...] = tuple(M_list)
        self.edges: tuple[tuple[int, int], ...] = tuple(edges)
        self.antisym_counts: tuple[int, ...] = tuple(antisym_counts_norm)

        # ---- per-factor fund placements + per-edge bifund placements ----
        if fund_sides is None:
            fund_sides_norm: list[list[str]] = [["left"] * M for M in M_list]
        else:
            fund_sides_norm = [list(s) for s in fund_sides]
            if len(fund_sides_norm) != self.k:
                raise ValueError(
                    f"fund_sides length {len(fund_sides_norm)} != k = {self.k}"
                )
            for i, s in enumerate(fund_sides_norm):
                if len(s) != M_list[i]:
                    raise ValueError(
                        f"fund_sides[{i}] length {len(s)} != M_list[{i}] = "
                        f"{M_list[i]}"
                    )
        n_bifund = len(edges)
        if bifund_placements is None:
            bifund_placements_norm = ["left"] * n_bifund
        else:
            bifund_placements_norm = list(bifund_placements)
            if len(bifund_placements_norm) != n_bifund:
                raise ValueError(
                    f"bifund_placements length {len(bifund_placements_norm)} "
                    f"!= |edges| = {n_bifund}"
                )
            for p in bifund_placements_norm:
                if p not in ("left", "between", "right"):
                    raise ValueError(
                        f"bifund_placements entries must be 'left', "
                        f"'between', or 'right' (got {p!r})"
                    )
        # ---- antisym_sides : per-factor list of 'left'/'right', one
        #      entry per antisym^2 hyper at that factor -----------------
        if antisym_sides is None:
            antisym_sides_norm: list[list[str]] = [
                ["left"] * c for c in antisym_counts_norm
            ]
        else:
            antisym_sides_norm = [list(s) for s in antisym_sides]
            if len(antisym_sides_norm) != self.k:
                raise ValueError(
                    f"antisym_sides length {len(antisym_sides_norm)} "
                    f"!= k = {self.k}"
                )
            for i, s in enumerate(antisym_sides_norm):
                if len(s) != antisym_counts_norm[i]:
                    raise ValueError(
                        f"antisym_sides[{i}] length {len(s)} != "
                        f"antisym_counts[{i}] = {antisym_counts_norm[i]}"
                    )
                for x in s:
                    if x not in ("left", "right"):
                        raise ValueError(
                            f"antisym_sides[{i}] entries must be 'left' "
                            f"or 'right'  (got {x!r})"
                        )

        # ---- normalize per-block bps_aut flags ------------------------
        if antisym_bps_aut is None:
            antisym_bps_aut_norm = [[False] * c for c in antisym_counts_norm]
        else:
            antisym_bps_aut_norm = [list(s) for s in antisym_bps_aut]
            if len(antisym_bps_aut_norm) != self.k:
                raise ValueError(
                    f"antisym_bps_aut length {len(antisym_bps_aut_norm)} "
                    f"!= k = {self.k}"
                )
            for i, s in enumerate(antisym_bps_aut_norm):
                if len(s) != antisym_counts_norm[i]:
                    raise ValueError(
                        f"antisym_bps_aut[{i}] length {len(s)} != "
                        f"antisym_counts[{i}] = {antisym_counts_norm[i]}"
                    )
                for x in s:
                    if not isinstance(x, bool):
                        raise ValueError(
                            f"antisym_bps_aut[{i}] entries must be bool "
                            f"(got {x!r})"
                        )
        if fund_bps_aut is None:
            fund_bps_aut_norm = [[False] * M for M in M_list]
        else:
            fund_bps_aut_norm = [list(s) for s in fund_bps_aut]
            if len(fund_bps_aut_norm) != self.k:
                raise ValueError(
                    f"fund_bps_aut length {len(fund_bps_aut_norm)} "
                    f"!= k = {self.k}"
                )
            for i, s in enumerate(fund_bps_aut_norm):
                if len(s) != M_list[i]:
                    raise ValueError(
                        f"fund_bps_aut[{i}] length {len(s)} != "
                        f"M_list[{i}] = {M_list[i]}"
                    )
                for x in s:
                    if not isinstance(x, bool):
                        raise ValueError(
                            f"fund_bps_aut[{i}] entries must be bool "
                            f"(got {x!r})"
                        )
        if bifund_bps_aut is None:
            bifund_bps_aut_norm = [(False, False)] * n_bifund
        else:
            bifund_bps_aut_norm = [tuple(e) for e in bifund_bps_aut]
            if len(bifund_bps_aut_norm) != n_bifund:
                raise ValueError(
                    f"bifund_bps_aut length {len(bifund_bps_aut_norm)} "
                    f"!= |edges| = {n_bifund}"
                )
            for ei, t in enumerate(bifund_bps_aut_norm):
                if len(t) != 2 or not all(isinstance(x, bool) for x in t):
                    raise ValueError(
                        f"bifund_bps_aut[{ei}] must be a (bool, bool) tuple "
                        f"(got {t!r})"
                    )

        self.fund_sides = fund_sides_norm
        self.bifund_placements = bifund_placements_norm
        self.antisym_sides = antisym_sides_norm
        self.antisym_bps_aut = antisym_bps_aut_norm
        self.fund_bps_aut = fund_bps_aut_norm
        self.bifund_bps_aut = bifund_bps_aut_norm

        # ---- per-factor structure : UN_antisym handles per-factor
        #      antisym^2 + fund matter uniformly  (n_a = 0 reduces to
        #      UN_Nf behaviour).  bps_aut flags flow through to its
        #      per-block bps_aut_applied kwargs. ---------------------------
        def _factor(fi, N, M):
            kwargs = dict(strict=False)
            n_a = antisym_counts_norm[fi]
            if any(antisym_bps_aut_norm[fi]):
                kwargs["antisym_positions"] = [
                    "before" if s == "left" else "after"
                    for s in antisym_sides_norm[fi]
                ]
                kwargs["antisym_bps_aut_applied"] = list(antisym_bps_aut_norm[fi])
            elif n_a > 0:
                kwargs["antisym_sides"] = antisym_sides_norm[fi]
            if any(fund_bps_aut_norm[fi]):
                kwargs["fund_positions"] = [
                    "before" if s == "left" else "after"
                    for s in fund_sides_norm[fi]
                ]
                kwargs["fund_bps_aut_applied"] = list(fund_bps_aut_norm[fi])
            elif M > 0:
                kwargs["fund_sides"] = fund_sides_norm[fi]
            return _pa.UN_antisym(N, n_a=n_a, Nf=M, **kwargs)
        factors = [
            _factor(fi, N, M) for fi, (N, M) in enumerate(zip(N_list, M_list))
        ]
        self.factors = factors
        r_full = [len(f.B) for f in factors]
        factor_start = [0]
        for r in r_full:
            factor_start.append(factor_start[-1] + r)
        gauge_dim = factor_start[-1]
        n_bifund = len(edges)
        total = gauge_dim + n_bifund

        # ---- block-diagonal pairing ------------------------------------
        B = [[0] * total for _ in range(total)]
        for fi, f in enumerate(factors):
            s = factor_start[fi]
            for i in range(r_full[fi]):
                for j in range(r_full[fi]):
                    B[s + i][s + j] = int(f.B[i][j])

        def _pad(v: Sequence[int], start: int) -> Vec:
            out = [0] * total
            for i, x in enumerate(v):
                out[start + i] = int(x)
            return tuple(out)

        nodes_per_factor = [
            [_pad(g, factor_start[fi]) for g in factors[fi].nodes]
            for fi in range(self.k)
        ]
        spec_per_factor = [
            [_pad(g, factor_start[fi]) for g in factors[fi].spec]
            for fi in range(self.k)
        ]

        # ---- bifund blocks per edge ------------------------------------
        bifund_specs: list[list[Vec]] = []
        bifund_nodes: list[Vec] = []
        r_pure = [2 * N for N in N_list]
        for ei, (i, j) in enumerate(edges):
            placement = bifund_placements_norm[ei]
            side_i = "left" if placement == "left" else "right"
            side_j = "left" if placement in ("left", "between") else "right"
            Ni, Nj = N_list[i], N_list[j]
            if side_i == "left":
                alphas_i = _pa._fundamental_wilson_support_UN(Ni)
            else:
                alphas_i = [tuple(-x for x in a)
                            for a in _pa._antifund_wilson_support_UN(Ni)]
            if side_j == "left":
                alphas_j = _pa._fundamental_wilson_support_UN(Nj)
            else:
                alphas_j = [tuple(-x for x in a)
                            for a in _pa._antifund_wilson_support_UN(Nj)]
            n_i, n_j = 2 * Ni - 1, 2 * Nj - 1
            pair_order = _pa._bifund_pair_order_AN(n_i, n_j)
            si = factor_start[i]
            sj = factor_start[j]
            mu_slot = gauge_dim + ei

            def _bifund(a, b, *,
                        si=si, sj=sj, mu_slot=mu_slot,
                        alphas_i=alphas_i, alphas_j=alphas_j) -> Vec:
                out = [0] * total
                for kk, x in enumerate(alphas_i[a]):
                    out[si + kk] = int(x)
                for kk, x in enumerate(alphas_j[b]):
                    out[sj + kk] = int(x)
                out[mu_slot] = 1
                return tuple(out)

            block = [_bifund(a, b) for (a, b) in pair_order]
            # bifund starter node = (lowest_i, lowest_j, +1), independent
            # of placement (same convention as UN_Nf and UN_bifund).
            starter = [0] * total
            starter[si + r_pure[i] - 1] = 1
            starter[sj + r_pure[j] - 1] = 1
            starter[mu_slot] = 1
            bifund_node = tuple(starter)
            # Per-leg bps_aut application: σ on slot-i or slot-j of
            # the bifund block + node, independently.
            ba_i, ba_j = bifund_bps_aut_norm[ei]
            if ba_i:
                block = [
                    _pa.apply_bps_aut_to_charge_UN(c, Ni, gauge_offset=si)
                    for c in block
                ]
                bifund_node = _pa.apply_bps_aut_to_charge_UN(
                    bifund_node, Ni, gauge_offset=si,
                )
            if ba_j:
                block = [
                    _pa.apply_bps_aut_to_charge_UN(c, Nj, gauge_offset=sj)
                    for c in block
                ]
                bifund_node = _pa.apply_bps_aut_to_charge_UN(
                    bifund_node, Nj, gauge_offset=sj,
                )
            bifund_specs.append(block)
            bifund_nodes.append(bifund_node)
        self.bifund_specs = bifund_specs
        self.bifund_nodes = bifund_nodes

        # Route each bifund block to a "gap" position in the linear
        # factor ordering.  gap_g sits between factor g-1 and factor g
        # (with gap_0 = before F_0, gap_k = after F_{k-1}).  For an
        # edge (i, j) with i < j:
        #     "left"    -> gap i        (before F_i)
        #     "between" -> gap (i + 1)  (right after F_i; commutes with
        #                                F_{i+1}, ..., F_{j-1} which
        #                                don't touch slot j)
        #     "right"   -> gap (j + 1)  (after  F_j)
        # Multiple bifund blocks routed to the same gap are emitted in
        # edge-index order.
        gaps: list[list[list[Vec]]] = [[] for _ in range(self.k + 1)]
        for ei, (i, j) in enumerate(edges):
            placement = bifund_placements_norm[ei]
            if placement == "left":
                gap_idx = i
            elif placement == "between":
                gap_idx = i + 1
            else:  # right
                gap_idx = j + 1
            gaps[gap_idx].append(bifund_specs[ei])

        # ---- assemble nodes / spec -------------------------------------
        nodes_UV: list[Vec] = []
        for fi in range(self.k):
            nodes_UV.extend(nodes_per_factor[fi])
        nodes_UV.extend(bifund_nodes)

        spec_UV: list[Vec] = []
        for fi in range(self.k):
            for block in gaps[fi]:
                spec_UV.extend(block)
            spec_UV.extend(spec_per_factor[fi])
        for block in gaps[self.k]:
            spec_UV.extend(block)

        # ---- registry --------------------------------------------------
        # UN_antisym per-factor node order:  pure_nodes (2(N-1)) ,
        # antisym_nodes (n_a) , fund_nodes (Nf) .  Bifund matter nodes
        # come after all factors.
        gauge_idx: list[int] = []
        antisym_idx: dict[int, list[int]] = {}
        fund_idx: dict[int, list[int]] = {}
        node_offset = 0
        for fi, (Ni, Mi) in enumerate(zip(N_list, M_list)):
            n_ai = antisym_counts_norm[fi]
            n_pure_nodes = len(factors[fi].nodes) - n_ai - Mi
            for i in range(n_pure_nodes):
                gauge_idx.append(node_offset + i)
            for i in range(n_ai):
                antisym_idx.setdefault(fi, []).append(
                    node_offset + n_pure_nodes + i
                )
            for i in range(Mi):
                fund_idx.setdefault(fi, []).append(
                    node_offset + n_pure_nodes + n_ai + i
                )
            node_offset += n_pure_nodes + n_ai + Mi
        bifund_idx = list(range(node_offset, node_offset + n_bifund))
        matter_idx = (
            [j for fi in range(self.k) for j in antisym_idx.get(fi, [])]
            + [j for fi in range(self.k) for j in fund_idx.get(fi, [])]
            + bifund_idx
        )

        self.gauge_node_indices = gauge_idx
        self.antisym_node_indices = antisym_idx
        self.fund_node_indices = fund_idx
        self.bifund_node_indices = bifund_idx
        self.matter_node_indices = matter_idx

        # ---- closed-form cone witness ---------------------------------
        f_lin: list[int] = []
        for f in factors:
            f_lin.extend(int(x) for x in f.cone_witness)
        f_lin.extend([0] * n_bifund)
        for ei in range(n_bifund):
            bn = bifund_nodes[ei]
            val = sum(fi_ * gi for fi_, gi in zip(f_lin, bn))
            f_lin[gauge_dim + ei] = max(1 - val, 0)
        self.cone_witness = tuple(f_lin)

        self.B = B
        self.nodes = nodes_UV
        self.spec = spec_UV

        # ---- audit ----------------------------------------------------
        Q = _bps.BPSQuiver.from_pairing(
            [list(g) for g in nodes_UV],
            [list(row) for row in B],
        )
        ok, info = Q.verify_spectrum_generator(spec_UV)
        if not ok:
            raise RuntimeError(
                f"{type(self).__name__}({N_list}, {M_list}, {edges}): "
                f"assembled spec is not a valid negating sequence on "
                f"the BPS quiver -- this graph topology may need a "
                f"different spec ordering than the parity-chain "
                f"recipe used here.  Info: {info}"
            )

        # ---- final BPSKAlgebra ----------------------------------------
        self.algebra = BPSKAlgebra(
            pairing=B,
            node_charges=nodes_UV,
            spec=spec_UV,
            cone_witness=self.cone_witness,
            verify="off",
        )

    def verify(self) -> bool:
        Q = _bps.BPSQuiver.from_pairing(
            [list(g) for g in self.nodes],
            [list(row) for row in self.B],
        )
        ok, _ = Q.verify_spectrum_generator(self.spec)
        return ok

    def __repr__(self) -> str:
        m_part = (
            f", M={list(self.M_list)}"
            if any(m > 0 for m in self.M_list) else ""
        )
        return (
            f"{type(self).__name__}(N={list(self.N_list)}{m_part}, "
            f"edges={list(self.edges)})"
        )


# ---------------------------------------------------------------------------
# CircularUQuiver -- affine Â-shape cycle
# ---------------------------------------------------------------------------


class CircularUQuiver(GraphUQuiver):
    """Affine `Â_{k-1}`-shape cyclic quiver on `k ≥ 2` U(N) factors.

    Edges: `[(0, 1), (1, 2), …, (k − 1, 0)]` for `k ≥ 3`.  For `k = 2`
    the cycle has *two* parallel bifundamentals between the two
    factors (the affine `Â_1` / Kronecker shape), edges
    `[(0, 1), (0, 1)]`.
    """

    def __init__(
        self,
        N_list: Sequence[int],
        M_list: Sequence[int] | None = None,
        *,
        strict: bool = True,
        **kwargs,
    ):
        k = len(N_list)
        if k < 2:
            raise ValueError(
                f"CircularUQuiver: need k ≥ 2 nodes, got {k}"
            )
        if k == 2:
            edges = [(0, 1), (0, 1)]
        else:
            edges = [(i, (i + 1) % k) for i in range(k)]
        super().__init__(N_list, M_list, edges, strict=strict, **kwargs)


# ---------------------------------------------------------------------------
# DShapeUQuiver -- D-Dynkin shape (chain with one fork)
# ---------------------------------------------------------------------------


class DShapeUQuiver(GraphUQuiver):
    """D-Dynkin shape on `n ≥ 4` nodes.

    Topology: a chain `0 − 1 − 2 − ⋯ − (n − 2)` with one extra branch
    node `n − 1` attached to node `n − 3`.  Concretely:

        edges = [(0,1), (1,2), …, (n−3, n−2), (n−3, n−1)].

    The fork is at node `n − 3`.
    """

    def __init__(
        self,
        N_list: Sequence[int],
        M_list: Sequence[int] | None = None,
        *,
        strict: bool = True,
        **kwargs,
    ):
        n = len(N_list)
        if n < 4:
            raise ValueError(
                f"DShapeUQuiver: need n ≥ 4 nodes, got {n}"
            )
        edges = [(i, i + 1) for i in range(n - 2)]
        edges.append((n - 3, n - 1))
        super().__init__(N_list, M_list, edges, strict=strict, **kwargs)


# ---------------------------------------------------------------------------
# E_n shapes -- T-graph branching at node 2 of the long arm
# ---------------------------------------------------------------------------


class _ENUQuiver(GraphUQuiver):
    """Internal base for E_6 / E_7 / E_8."""

    _expected_n: int
    _edges_template: tuple[tuple[int, int], ...]

    def __init__(
        self,
        N_list: Sequence[int],
        M_list: Sequence[int] | None = None,
        *,
        strict: bool = True,
        **kwargs,
    ):
        if len(N_list) != self._expected_n:
            raise ValueError(
                f"{type(self).__name__}: requires exactly "
                f"{self._expected_n} nodes, got {len(N_list)}"
            )
        super().__init__(
            N_list, M_list, list(self._edges_template),
            strict=strict, **kwargs,
        )


class E6UQuiver(_ENUQuiver):
    """E_6-Dynkin shape on 6 nodes: chain `0 − 1 − 2 − 3 − 4` with one
    branch `(2, 5)` off the middle node `2`."""
    _expected_n = 6
    _edges_template = (
        (0, 1), (1, 2), (2, 3), (3, 4), (2, 5),
    )


class E7UQuiver(_ENUQuiver):
    """E_7-Dynkin shape on 7 nodes: chain `0 − 1 − 2 − 3 − 4 − 5` with
    one branch `(2, 6)` off node `2`."""
    _expected_n = 7
    _edges_template = (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6),
    )


class E8UQuiver(_ENUQuiver):
    """E_8-Dynkin shape on 8 nodes: chain `0 − 1 − 2 − 3 − 4 − 5 − 6`
    with one branch `(2, 7)` off node `2`."""
    _expected_n = 8
    _edges_template = (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (2, 7),
    )


# ---------------------------------------------------------------------------
# Affine ADE variants
# ---------------------------------------------------------------------------
#
# Naming convention:
#   * Â_{n}   = CircularUQuiver  (already present; n nodes, cycle).
#   * D̂_{n}  = AffineDUQuiver   (n + 1 nodes with forks at BOTH ends).
#   * Ê_6/Ê_7/Ê_8 = AffineE6/E7/E8UQuiver  (one extra node attached to
#                  the affine root of the corresponding finite Dynkin).
#
# Affine D_n (D̂_{n}) has n + 1 nodes:
#   - For n = 4:  D̂_4  is the "star" with a central node connected
#     to 4 leaves  (5 nodes total).
#   - For n >= 5: chain  ``2 - 3 - ... - (n - 2)``  with two leaves
#     branching from each end:  ``{0, 1}``  off node  2  and
#     ``{n-1, n}``  off node  n - 2 .  (n + 1 nodes total.)
#
# Affine E_n shapes follow the standard extended Dynkin with one
# extra node attached at the canonical affine position:
#   - Ê_6:  extra node attached to node 5 (the trivalent "branch" node)
#           of the finite E_6 ; total 7 nodes.
#   - Ê_7:  extra node attached to node 0 (the long-arm tip) of E_7 ;
#           total 8 nodes.
#   - Ê_8:  extra node attached to node 6 (the long-arm tip) of E_8 ;
#           total 9 nodes.


def _affine_d_edges(n: int) -> list[tuple[int, int]]:
    """Edge list for D̂_{n}  on  n + 1  nodes (n >= 4).

    Layout for  n = 4 :   star -- node 2 central, leaves {0, 1, 3, 4}.
    Layout for  n >= 5 :  spine  ``2 - 3 - ... - (n - 2)`` , forks
    {0, 1}  at node 2 and  {n - 1, n}  at node  n - 2 .
    """
    if n < 4:
        raise ValueError(f"D̂_n requires n >= 4, got {n}")
    if n == 4:
        return [(0, 2), (1, 2), (2, 3), (2, 4)]
    edges: list[tuple[int, int]] = [(0, 2), (1, 2)]
    for i in range(2, n - 2):
        edges.append((i, i + 1))
    edges.append((n - 2, n - 1))
    edges.append((n - 2, n))
    return edges


class AffineDUQuiver(GraphUQuiver):
    """Affine  ``D̂_{n}`` -shape unitary quiver on  ``n + 1``  nodes
    (with  ``n >= 4`` ).

    For  ``n = 4`` :  central node 2 with leaves  {0, 1, 3, 4} .
    For  ``n >= 5`` :  spine  ``2 - 3 - ... - (n - 2)``  with extra
    leaves  {0, 1}  attached at node 2 and  {n - 1, n}  at node n - 2 .
    """

    def __init__(
        self,
        N_list: Sequence[int],
        M_list: Sequence[int] | None = None,
        *,
        strict: bool = True,
        **kwargs,
    ):
        n = len(N_list) - 1
        if n < 4:
            raise ValueError(
                f"AffineDUQuiver: need n >= 4 (n + 1 nodes), got "
                f"{len(N_list)} nodes"
            )
        super().__init__(
            N_list, M_list, _affine_d_edges(n), strict=strict, **kwargs,
        )


class _AffineENUQuiver(GraphUQuiver):
    """Internal base for canonical Ê_6 / Ê_7 / Ê_8 wrappers."""

    _expected_n: int
    _edges_template: tuple[tuple[int, int], ...]

    def __init__(
        self,
        N_list: Sequence[int],
        M_list: Sequence[int] | None = None,
        *,
        strict: bool = True,
        **kwargs,
    ):
        if len(N_list) != self._expected_n:
            raise ValueError(
                f"{type(self).__name__}: requires exactly "
                f"{self._expected_n} nodes, got {len(N_list)}"
            )
        super().__init__(
            N_list, M_list, list(self._edges_template),
            strict=strict, **kwargs,
        )


class AffineE6UQuiver(_AffineENUQuiver):
    """Affine Ê_6-shape unitary quiver on 7 nodes.

    Edges = E_6 edges + one extra node 6 attached to the trivalent
    branch endpoint of finite E_6 (node 5):
      ``(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)`` .
    """
    _expected_n = 7
    _edges_template = (
        (0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6),
    )


class AffineE7UQuiver(_AffineENUQuiver):
    """Affine Ê_7-shape unitary quiver on 8 nodes.

    Edges = E_7 edges + one extra node 7 attached to node 0 (the
    long-arm tip):  the E_7 edges  ``(0, 1), (1, 2), (2, 3), (3, 4),
    (4, 5), (2, 6)``  plus  ``(0, 7)`` .
    """
    _expected_n = 8
    _edges_template = (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (0, 7),
    )


class AffineE8UQuiver(_AffineENUQuiver):
    """Affine Ê_8-shape unitary quiver on 9 nodes.

    Edges = E_8 edges + one extra node 8 attached to node 6 (the
    long-arm tip):  the E_8 edges  ``(0, 1), (1, 2), (2, 3), (3, 4),
    (4, 5), (5, 6), (2, 7)``  plus  ``(6, 8)`` .
    """
    _expected_n = 9
    _edges_template = (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (2, 7), (6, 8),
    )


# ---------------------------------------------------------------------------
# PermutedUQuiver -- arbitrary permutation of gauge / matter blocks
# ---------------------------------------------------------------------------


# Block tag forms accepted in `spec_order`:
#   ("pure",   i)         -- pure-gauge spec block for factor i
#   ("bifund", e)         -- bifund block for edge e (index into `edges`)
#   ("fund",   i, j)      -- j-th fundamental on factor i
Block = tuple


def _expected_block_set(N_list, M_list, edges, antisym_counts=None):
    """Multiset of expected block tags for the given topology.

    Block tag forms:
      ("pure",    fi)            -- pure-gauge spec for factor fi
      ("fund",    fi, j)         -- j-th fundamental at factor fi
      ("antisym", fi, k)         -- k-th antisym^2 hyper at factor fi
      ("bifund",  ei)            -- bifund block for edge index ei
    """
    if antisym_counts is None:
        antisym_counts = [0] * len(N_list)
    blocks: set[Block] = set()
    for fi in range(len(N_list)):
        blocks.add(("pure", fi))
        for j in range(M_list[fi]):
            blocks.add(("fund", fi, j))
        for k in range(antisym_counts[fi]):
            blocks.add(("antisym", fi, k))
    for ei in range(len(edges)):
        blocks.add(("bifund", ei))
    return blocks


class DegenerateGaugeFactorError(ValueError):
    """A quiver configuration is rejected for a **spec-order-independent**
    reason (a degenerate U(1) gauge factor carrying matter or non-terminal
    edges).  Distinguishable so that permutation enumerators
    (`dictionaries_weak.seed_atoms` / `build_weak`) can skip the whole
    configuration instead of retrying factorially many spec-order
    permutations that all raise the same error — the failure mode that
    silently turned the dictionary self-tests into hour-long hangs."""


class PermutedUQuiver:
    """Unitary quiver with the spec generator assembled in a
    user-specified permutation of the elementary gauge / matter blocks.

    Parameters
    ----------
    N_list, M_list, edges
        Same as :class:`GraphUQuiver` -- gauge ranks, per-node fund
        counts, undirected bifundamental edges (i, j) with i ≤ j and
        no self-loops.
    spec_order
        Ordered list of block tags specifying the assembly order of
        the spec generator.  Must contain each block exactly once.
        Block tag forms:

          * ``("pure",   i)``     -- pure-gauge spec for factor i.
          * ``("bifund", e)``     -- bifund block for edge index e.
          * ``("fund",   i, j)``  -- j-th fundamental of factor i
                                     (j = 0, ..., M_list[i] - 1).

        Total number of blocks =  k + |edges| + sum(M_list) .
    strict
        Asymp-freedom enforcement (default True), exactly as in
        :class:`GraphUQuiver` .

    For each non-pure block, slot-i alphas are chosen automatically
    from the block's position relative to ``("pure", i)`` in
    ``spec_order`` :

        position(block) <  position(pure_i)   ->  fund(N_i)
        position(block) >  position(pure_i)   ->  -antifund(N_i)

    For a bifund block on edge (i, j), this rule is applied
    independently to slot i and slot j.  The same parity-chain
    pair-order :func:`pure_ade._bifund_pair_order_AN`  is used in all
    cases (it depends only on the support sizes  ``2N_i - 1`` ).

    For a fund block on factor i, only slot-i applies.

    Verified construction-time via
    :meth:`bps_quiver_tools.BPSQuiver.verify_spectrum_generator` :
    if the permutation produces an invalid spec (wrong negating
    sequence on the BPS quiver), the constructor raises rather than
    returning a malformed algebra.

    Attributes
    ----------
    spec_order
        Normalised tuple of block tags in assembly order.
    block_alphas
        Dict ``block_tag -> tuple of slot-alpha lists`` recording the
        side choice the constructor made per non-pure block (useful
        for debugging or reconstructing why a particular permutation
        succeeded).
    .B, .nodes, .spec, .algebra, .cone_witness, .factors,
    .bifund_specs, .bifund_nodes
        Standard surface (mirrors :class:`GraphUQuiver` ).
    """

    def __init__(
        self,
        N_list: Sequence[int],
        M_list: Sequence[int] | None = None,
        edges: Sequence[tuple[int, int]] = (),
        *,
        spec_order: Sequence[Block],
        strict: bool = True,
        build_algebra: bool = True,
        verify_at_construction: bool = True,
        antisym_counts: Sequence[int] | None = None,
        # Per-block BPS-quiver Z_2 automorphism flags.  Maps block tag
        # to bool (for "fund"/"antisym" blocks) or (bool, bool) tuple
        # (for "bifund" blocks, one flag per leg).  Missing entries
        # default to False (= no σ applied to that block).
        bps_aut: "Mapping[Block, Union[bool, tuple[bool, bool]]] | None" = None,
    ):
        N_list, M_list, edges, antisym_counts_norm = _validate_graph_inputs(
            N_list, M_list, edges,
            strict=strict, label=type(self).__name__,
            antisym_counts=antisym_counts,
        )
        self.antisym_counts: tuple[int, ...] = tuple(antisym_counts_norm)
        # Normalize bps_aut.  We resolve to a dict keyed by the canonical
        # block tag form, so any spec_order permutation accesses it
        # consistently.
        bps_aut_norm: dict = {}
        if bps_aut is not None:
            for k_blk, v_blk in bps_aut.items():
                bps_aut_norm[tuple(k_blk)] = v_blk

        def _bps_aut_for(blk, *, want_tuple=False):
            v = bps_aut_norm.get(tuple(blk))
            if v is None:
                return (False, False) if want_tuple else False
            if want_tuple:
                if not (isinstance(v, tuple) and len(v) == 2
                        and all(isinstance(x, bool) for x in v)):
                    raise ValueError(
                        f"bps_aut[{blk}] must be a (bool, bool) tuple "
                        f"for bifund blocks (got {v!r})"
                    )
                return v
            else:
                if not isinstance(v, bool):
                    raise ValueError(
                        f"bps_aut[{blk}] must be a bool for "
                        f"non-bifund blocks (got {v!r})"
                    )
                return v
        self.bps_aut = dict(bps_aut_norm)
        # U(1) gauge factors are forbidden *when degenerate*: the
        # pure-gauge spec block for U(1) is empty (it contributes
        # 2*(N-1) = 0 nodes), and matter or a second edge attached to a
        # U(1) factor disconnects the resulting BPS quiver.  A
        # *terminal, matterless* U(1) (single edge, M_i = 0) is fine —
        # that is exactly the affine-ADE leaf configuration the
        # `PermutedAffine*UQuiver` wrappers are built with (D̂₄ =
        # [1,1,2,1,1], …), matching the unpermuted `Affine*UQuiver`
        # acceptance.  Reject only the genuinely degenerate cases.
        for i, N_i in enumerate(N_list):
            if N_i < 2:
                degree = sum(1 for (a, b) in edges if i in (a, b))
                if M_list[i] > 0 or degree >= 2:
                    raise DegenerateGaugeFactorError(
                        f"{type(self).__name__}: U({N_i}) gauge factor at "
                        f"position {i} carries "
                        f"{'matter' if M_list[i] > 0 else 'non-terminal edges'}"
                        f"; a U(1) factor is allowed only as a terminal, "
                        f"matterless leaf (its pure spec block is empty, and "
                        f"matter / a second edge disconnects the BPS quiver)."
                    )
        self.k = len(N_list)
        self.N_list: tuple[int, ...] = tuple(N_list)
        self.M_list: tuple[int, ...] = tuple(M_list)
        self.edges: tuple[tuple[int, int], ...] = tuple(edges)

        # ---- validate spec_order ---------------------------------------
        spec_order_norm: list[Block] = [tuple(b) for b in spec_order]
        expected = _expected_block_set(
            N_list, M_list, edges, antisym_counts=antisym_counts_norm,
        )
        seen: set[Block] = set()
        for blk in spec_order_norm:
            if blk in seen:
                raise ValueError(
                    f"{type(self).__name__}: duplicate block in spec_order: "
                    f"{blk}"
                )
            if blk not in expected:
                raise ValueError(
                    f"{type(self).__name__}: spec_order entry {blk} is not "
                    f"a valid block tag for this topology  (expected one of "
                    f"{sorted(expected)})"
                )
            seen.add(blk)
        missing = expected - seen
        if missing:
            raise ValueError(
                f"{type(self).__name__}: spec_order missing blocks: "
                f"{sorted(missing)}"
            )
        self.spec_order: tuple[Block, ...] = tuple(spec_order_norm)
        position = {blk: idx for idx, blk in enumerate(spec_order_norm)}

        # ---- factor structure (pure-only; matter built per-block below) -
        # Per-factor lattice block layout :
        #     [ 2N pure-gauge coords  |  n_a antisym^2 mus  |  Nf fund mus ]
        # Bifund mus are appended after all factors.
        factors_pure = [_pa.UN_Nf(N, 0) for N in N_list]
        self.factors = factors_pure
        r_pure = [2 * N for N in N_list]
        r_full = [
            2 * N + antisym_counts_norm[fi] + M_list[fi]
            for fi, N in enumerate(N_list)
        ]
        factor_start = [0]
        for r in r_full:
            factor_start.append(factor_start[-1] + r)
        gauge_dim = factor_start[-1]
        n_bifund = len(edges)
        total = gauge_dim + n_bifund
        # Per-factor offset within its block to the antisym^2 mu slots (one
        # mu per antisym hyper) and the fund mu slots (one mu per fund).
        antisym_mu_offset = r_pure  # start of antisym mus inside factor block
        fund_mu_offset = [
            r_pure[fi] + antisym_counts_norm[fi] for fi in range(len(N_list))
        ]

        # ---- block-diagonal pairing (pure-gauge blocks only; flavour zero) -
        B = [[0] * total for _ in range(total)]
        for fi, f in enumerate(factors_pure):
            s = factor_start[fi]
            for i in range(r_pure[fi]):
                for j in range(r_pure[fi]):
                    B[s + i][s + j] = int(f.B[i][j])

        # ---- per-block alpha-side selector -----------------------------
        def alphas_for(slot_factor: int, my_pos: int) -> list[tuple[int, ...]]:
            if my_pos < position[("pure", slot_factor)]:
                return _pa._fundamental_wilson_support_UN(N_list[slot_factor])
            return [tuple(-x for x in a)
                    for a in _pa._antifund_wilson_support_UN(N_list[slot_factor])]

        # ---- assemble spec block-by-block ------------------------------
        spec_UV: list[tuple[int, ...]] = []
        bifund_specs: list[list[tuple[int, ...]]] = [None] * n_bifund   # type: ignore
        bifund_nodes: list[tuple[int, ...]] = [None] * n_bifund         # type: ignore
        block_alphas: dict[Block, tuple] = {}

        # Padded pure-gauge spec blocks (one per factor).
        pure_spec_padded: list[list[tuple[int, ...]]] = []
        for fi in range(self.k):
            s = factor_start[fi]
            block = []
            for g in factors_pure[fi].spec:
                v = [0] * total
                for kk, x in enumerate(g):
                    v[s + kk] = int(x)
                block.append(tuple(v))
            pure_spec_padded.append(block)

        for blk_idx, blk in enumerate(spec_order_norm):
            kind = blk[0]
            if kind == "pure":
                fi = blk[1]
                spec_UV.extend(pure_spec_padded[fi])
            elif kind == "bifund":
                ei = blk[1]
                i_, j_ = edges[ei]
                a_i = alphas_for(i_, blk_idx)
                a_j = alphas_for(j_, blk_idx)
                n_i = 2 * N_list[i_] - 1
                n_j = 2 * N_list[j_] - 1
                pair_order = _pa._bifund_pair_order_AN(n_i, n_j)
                si = factor_start[i_]
                sj = factor_start[j_]
                mu_slot = gauge_dim + ei

                def _bifund(a, b, *,
                            si=si, sj=sj, mu_slot=mu_slot,
                            a_i=a_i, a_j=a_j) -> tuple[int, ...]:
                    out = [0] * total
                    for kk, x in enumerate(a_i[a]):
                        out[si + kk] = int(x)
                    for kk, x in enumerate(a_j[b]):
                        out[sj + kk] = int(x)
                    out[mu_slot] = 1
                    return tuple(out)

                block = [_bifund(a, b) for (a, b) in pair_order]
                # bifund starter node (independent of placement)
                starter = [0] * total
                starter[si + r_pure[i_] - 1] = 1
                starter[sj + r_pure[j_] - 1] = 1
                starter[mu_slot] = 1
                bifund_node_local = tuple(starter)
                # Per-leg bps_aut on the bifund block.
                ba_i, ba_j = _bps_aut_for(blk, want_tuple=True)
                if ba_i:
                    block = [
                        _pa.apply_bps_aut_to_charge_UN(
                            c, N_list[i_], gauge_offset=si)
                        for c in block
                    ]
                    bifund_node_local = _pa.apply_bps_aut_to_charge_UN(
                        bifund_node_local, N_list[i_], gauge_offset=si,
                    )
                if ba_j:
                    block = [
                        _pa.apply_bps_aut_to_charge_UN(
                            c, N_list[j_], gauge_offset=sj)
                        for c in block
                    ]
                    bifund_node_local = _pa.apply_bps_aut_to_charge_UN(
                        bifund_node_local, N_list[j_], gauge_offset=sj,
                    )
                spec_UV.extend(block)
                bifund_specs[ei] = block
                bifund_nodes[ei] = bifund_node_local
                block_alphas[blk] = (a_i, a_j)
            elif kind == "fund":
                fi, j = blk[1], blk[2]
                a = alphas_for(fi, blk_idx)
                si = factor_start[fi]
                mu_slot = factor_start[fi] + fund_mu_offset[fi] + j
                block = []
                for x_ in a:
                    v = [0] * total
                    for kk, xx in enumerate(x_):
                        v[si + kk] = int(xx)
                    v[mu_slot] = 1
                    block.append(tuple(v))
                # Per-block bps_aut on the fund block.
                if _bps_aut_for(blk):
                    block = [
                        _pa.apply_bps_aut_to_charge_UN(
                            c, N_list[fi], gauge_offset=si)
                        for c in block
                    ]
                spec_UV.extend(block)
                block_alphas[blk] = (a,)
            elif kind == "antisym":
                fi, kk_anti = blk[1], blk[2]
                # Build the antisym^2 matter spec for factor fi using the
                # closed-form  P_N  pair set + (a+b, -a) ASC order with a
                # guided DFS to resolve the small N >= 5 swap.
                Ni = N_list[fi]
                # alphas chosen by position relative to ("pure", fi).
                if blk_idx < position[("pure", fi)]:
                    supp = _pa._fundamental_wilson_support_UN(Ni)
                    sign = +1
                else:
                    supp = _pa._antifund_wilson_support_UN(Ni)
                    sign = -1
                pairs = _pa._antisym2_pair_set_UN(Ni)
                si = factor_start[fi]
                mu_slot = factor_start[fi] + antisym_mu_offset[fi] + kk_anti
                # Build per-block (factor + this-mu) sub-quiver for DFS.
                r_local = r_pure[fi] + 1
                B_local = [[0] * r_local for _ in range(r_local)]
                for ri in range(r_pure[fi]):
                    for rj in range(r_pure[fi]):
                        B_local[ri][rj] = int(factors_pure[fi].B[ri][rj])
                # Per-block raw charges (length r_pure + 1, last slot = mu)
                pair_of_local = {}
                matter_unordered_local = []
                for (a, b) in pairs:
                    g_gauge = tuple(
                        sign * (supp[a][kg] + supp[b][kg])
                        for kg in range(r_pure[fi])
                    )
                    local_ch = tuple(list(g_gauge) + [1])
                    matter_unordered_local.append(local_ch)
                    pair_of_local[local_ch] = (a, b)
                # matter_node = lowest weight of antisym^2 + mu (local)
                lw_local = [0] * r_pure[fi]
                lw_local[2 * (Ni - 1) - 1] = 1
                lw_local[2 * Ni - 1] = 1
                matter_node_local = tuple(lw_local + [1])
                nodes_local = (
                    [tuple(list(g) + [0]) for g in factors_pure[fi].nodes]
                    + [matter_node_local]
                )
                pure_spec_local = [
                    tuple(list(g) + [0]) for g in factors_pure[fi].spec
                ]
                # 'side' for the DFS is left when antisym block is before
                # ("pure", fi) in spec_order, else right.
                local_side = ("left"
                              if blk_idx < position[("pure", fi)]
                              else "right")
                ordered_local = _pa._antisym2_dfs_order(
                    Ni, B_local, nodes_local, pure_spec_local,
                    matter_unordered_local, pair_of_local, local_side,
                )
                # Lift ordered_local back into the full UV lattice.
                def _lift(v_local, *, si=si, mu_slot=mu_slot,
                          r_pure_fi=r_pure[fi]):
                    out = [0] * total
                    for k_, x in enumerate(v_local[:r_pure_fi]):
                        out[si + k_] = int(x)
                    if v_local[r_pure_fi] != 0:
                        out[mu_slot] = int(v_local[r_pure_fi])
                    return tuple(out)
                lifted = [_lift(v) for v in ordered_local]
                # Per-block bps_aut on the antisym^2 block.
                if _bps_aut_for(blk):
                    lifted = [
                        _pa.apply_bps_aut_to_charge_UN(
                            c, N_list[fi], gauge_offset=si)
                        for c in lifted
                    ]
                spec_UV.extend(lifted)
                block_alphas[blk] = (supp, sign, pairs)
            else:
                raise ValueError(
                    f"{type(self).__name__}: unrecognised spec_order kind: "
                    f"{kind!r}  in block {blk}"
                )

        self.bifund_specs = bifund_specs
        self.bifund_nodes = bifund_nodes
        self.block_alphas = block_alphas

        # ---- assemble nodes (gauge + antisym + fund + bifund) ----------
        nodes_UV: list[tuple[int, ...]] = []
        antisym_node_indices: dict[int, list[int]] = {}
        fund_node_indices: dict[int, list[int]] = {}
        gauge_node_indices: list[int] = []
        for fi in range(self.k):
            s = factor_start[fi]
            for g in factors_pure[fi].nodes:
                gauge_node_indices.append(len(nodes_UV))
                v = [0] * total
                for kk, x in enumerate(g):
                    v[s + kk] = int(x)
                nodes_UV.append(tuple(v))
            # antisym^2 matter nodes (one per antisym hyper at factor fi)
            for k_anti in range(antisym_counts_norm[fi]):
                antisym_node_indices.setdefault(fi, []).append(len(nodes_UV))
                v = [0] * total
                # lowest weight of antisym^2 = eps_{N-1}^* + eps_N^*
                v[s + 2 * (N_list[fi] - 1) - 1] = 1
                v[s + 2 * N_list[fi] - 1] = 1
                v[s + antisym_mu_offset[fi] + k_anti] = 1
                node_tuple = tuple(v)
                if _bps_aut_for(("antisym", fi, k_anti)):
                    node_tuple = _pa.apply_bps_aut_to_charge_UN(
                        node_tuple, N_list[fi], gauge_offset=s,
                    )
                nodes_UV.append(node_tuple)
            # fund matter nodes
            for j in range(M_list[fi]):
                fund_node_indices.setdefault(fi, []).append(len(nodes_UV))
                v = [0] * total
                v[s + r_pure[fi] - 1] = 1
                v[s + fund_mu_offset[fi] + j] = 1
                node_tuple = tuple(v)
                if _bps_aut_for(("fund", fi, j)):
                    node_tuple = _pa.apply_bps_aut_to_charge_UN(
                        node_tuple, N_list[fi], gauge_offset=s,
                    )
                nodes_UV.append(node_tuple)
        bifund_first = len(nodes_UV)
        nodes_UV.extend(bifund_nodes)
        self.gauge_node_indices = gauge_node_indices
        self.antisym_node_indices = antisym_node_indices
        self.fund_node_indices = fund_node_indices
        self.bifund_node_indices = list(
            range(bifund_first, bifund_first + n_bifund)
        )
        self.matter_node_indices = (
            [j for fi in range(self.k) for j in antisym_node_indices.get(fi, [])]
            + [j for fi in range(self.k) for j in fund_node_indices.get(fi, [])]
            + self.bifund_node_indices
        )

        # ---- closed-form cone witness ----------------------------------
        f_lin: list[int] = [0] * total
        for fi, f in enumerate(factors_pure):
            s = factor_start[fi]
            for kk, x in enumerate(f.cone_witness):
                f_lin[s + kk] = int(x)
        # antisym^2 mus: pick c so that f . (lowest_antisym + mu) >= 1.
        for fi in range(self.k):
            s = factor_start[fi]
            # lowest weight of antisym^2 contributes
            #   f_lin[s + 2(N-1)-1] + f_lin[s + 2N-1]
            for k_anti in range(antisym_counts_norm[fi]):
                low_dot = (
                    f_lin[s + 2 * (N_list[fi] - 1) - 1]
                    + f_lin[s + 2 * N_list[fi] - 1]
                )
                f_lin[s + antisym_mu_offset[fi] + k_anti] = max(
                    1 - low_dot, 0
                )
        # fund flavour mus
        for fi in range(self.k):
            s = factor_start[fi]
            for j in range(M_list[fi]):
                lowest_dot = f_lin[s + r_pure[fi] - 1]
                f_lin[s + fund_mu_offset[fi] + j] = max(1 - lowest_dot, 0)
        # bifund flavour mus
        for ei in range(n_bifund):
            bn = bifund_nodes[ei]
            val = sum(fi_ * gi for fi_, gi in zip(f_lin, bn))
            f_lin[gauge_dim + ei] = max(1 - val, 0)
        self.cone_witness = tuple(f_lin)

        self.B = B
        self.nodes = nodes_UV
        self.spec = spec_UV

        # ---- audit at construction time --------------------------------
        # `verify_at_construction=False` skips this `BPSQuiver.verify_
        # spectrum_generator` call.  The PermutedUQuiver assembly recipe
        # is deterministic and well-tested; the verify call exists as a
        # historical safety net for novel permutations and is the bulk
        # of the per-permutation construction cost after the
        # `BPSKAlgebra.__init__` step is also skipped.  Bulk seed
        # enumerators (`dictionaries_weak.seed_atoms.permuted_seed_entries_at_n_fast`)
        # pass `False` here for the speed-up; the dictionary's own
        # audit (`audit_specs`) is the right place to catch any
        # algorithmic regression.
        if verify_at_construction:
            Q = _bps.BPSQuiver.from_pairing(
                [list(g) for g in nodes_UV],
                [list(row) for row in B],
            )
            ok, info = Q.verify_spectrum_generator(spec_UV)
            if not ok:
                raise RuntimeError(
                    f"{type(self).__name__}({N_list}, {M_list}, {edges}, "
                    f"spec_order=...): assembled spec is not a valid "
                    f"negating sequence on the BPS quiver.  Info: "
                    f"missing_step={info.get('missing_step')}, "
                    f"steps={info.get('steps')}"
                )

        # ---- final BPSKAlgebra ----------------------------------------
        # ``build_algebra=False`` skips this step for callers that only
        # need the raw ``(B, nodes, spec)`` data and want to avoid the
        # ~70 % of construction wall-clock that ``BPSKAlgebra.__init__``
        # consumes.  ``self.algebra`` is set to ``None`` in that case;
        # the ``.B`` / ``.nodes`` / ``.spec`` / ``.cone_witness``
        # attributes remain populated as usual.  Used by
        # ``dictionaries_weak.seed_atoms.permuted_seed_entries_at_n_fast``
        # for bulk seed enumeration in the trusted-closure builder.
        if build_algebra:
            self.algebra = BPSKAlgebra(
                pairing=B,
                node_charges=nodes_UV,
                spec=spec_UV,
                cone_witness=self.cone_witness,
                verify="off",
            )
        else:
            self.algebra = None

    def verify(self) -> bool:
        Q = _bps.BPSQuiver.from_pairing(
            [list(g) for g in self.nodes],
            [list(row) for row in self.B],
        )
        ok, _ = Q.verify_spectrum_generator(self.spec)
        return ok

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(N={list(self.N_list)}, "
            f"M={list(self.M_list)}, edges={list(self.edges)}, "
            f"spec_order=<{len(self.spec_order)} blocks>)"
        )


# ---------------------------------------------------------------------------
# ADE-shape wrappers around PermutedUQuiver
# ---------------------------------------------------------------------------


class PermutedLinearUQuiver(PermutedUQuiver):
    """Linear A-shape unitary quiver  ``U(N_1) - U(N_2) - ... - U(N_k)``
    with arbitrary user-specified spec ordering.

    Edges:  ``[(0, 1), (1, 2), ..., (k - 2, k - 1)]`` .
    """

    def __init__(
        self,
        N_list: Sequence[int],
        M_list: Sequence[int] | None = None,
        *,
        spec_order: Sequence[Block],
        strict: bool = True,
        build_algebra: bool = True,
        verify_at_construction: bool = True,
        **kwargs,
    ):
        if len(N_list) < 1:
            raise ValueError(
                f"PermutedLinearUQuiver: need k >= 1, got {len(N_list)}"
            )
        edges = [(i, i + 1) for i in range(len(N_list) - 1)]
        super().__init__(
            N_list, M_list, edges,
            spec_order=spec_order, strict=strict,
            build_algebra=build_algebra,
            verify_at_construction=verify_at_construction,
            **kwargs,
        )


class PermutedCircularUQuiver(PermutedUQuiver):
    """Affine  ``Â_{k-1}`` -shape cyclic unitary quiver with arbitrary
    user-specified spec ordering.

    Edges (k >= 3):  ``[(0, 1), (1, 2), ..., (k - 1, 0)]`` .
    For k = 2 the cycle has two parallel bifundamentals
    (the affine  ``Â_1``  / Kronecker shape):  ``[(0, 1), (0, 1)]`` .
    """

    def __init__(
        self,
        N_list: Sequence[int],
        M_list: Sequence[int] | None = None,
        *,
        spec_order: Sequence[Block],
        strict: bool = True,
        build_algebra: bool = True,
        verify_at_construction: bool = True,
        **kwargs,
    ):
        k = len(N_list)
        if k < 2:
            raise ValueError(
                f"PermutedCircularUQuiver: need k >= 2, got {k}"
            )
        if k == 2:
            edges = [(0, 1), (0, 1)]
        else:
            edges = [(i, (i + 1) % k) for i in range(k)]
            # Canonicalise i <= j (the wrap-around edge (k-1, 0)).
            edges = [(min(a, b), max(a, b)) for (a, b) in edges]
        super().__init__(
            N_list, M_list, edges,
            spec_order=spec_order, strict=strict,
            build_algebra=build_algebra,
            verify_at_construction=verify_at_construction,
            **kwargs,
        )


class PermutedDShapeUQuiver(PermutedUQuiver):
    """D-Dynkin shape on  ``n >= 4``  nodes with arbitrary user-specified
    spec ordering.

    Topology: a chain  ``0 - 1 - 2 - ... - (n - 2)``  with one extra
    branch node  ``n - 1``  attached to node  ``n - 3`` :

        edges = [(0, 1), (1, 2), ..., (n - 3, n - 2), (n - 3, n - 1)] .
    """

    def __init__(
        self,
        N_list: Sequence[int],
        M_list: Sequence[int] | None = None,
        *,
        spec_order: Sequence[Block],
        strict: bool = True,
        build_algebra: bool = True,
        verify_at_construction: bool = True,
        **kwargs,
    ):
        n = len(N_list)
        if n < 4:
            raise ValueError(
                f"PermutedDShapeUQuiver: need n >= 4 nodes, got {n}"
            )
        edges = [(i, i + 1) for i in range(n - 2)]
        edges.append((n - 3, n - 1))
        super().__init__(
            N_list, M_list, edges,
            spec_order=spec_order, strict=strict,
            build_algebra=build_algebra,
            verify_at_construction=verify_at_construction,
            **kwargs,
        )


class _PermutedENUQuiver(PermutedUQuiver):
    """Internal base for the permuted E_6 / E_7 / E_8 wrappers."""

    _expected_n: int
    _edges_template: tuple[tuple[int, int], ...]

    def __init__(
        self,
        N_list: Sequence[int],
        M_list: Sequence[int] | None = None,
        *,
        spec_order: Sequence[Block],
        strict: bool = True,
        build_algebra: bool = True,
        verify_at_construction: bool = True,
        **kwargs,
    ):
        if len(N_list) != self._expected_n:
            raise ValueError(
                f"{type(self).__name__}: requires exactly "
                f"{self._expected_n} nodes, got {len(N_list)}"
            )
        super().__init__(
            N_list, M_list, list(self._edges_template),
            spec_order=spec_order, strict=strict,
            build_algebra=build_algebra,
            verify_at_construction=verify_at_construction,
            **kwargs,
        )


class PermutedE6UQuiver(_PermutedENUQuiver):
    """E_6-Dynkin shape (6 nodes; chain `0-1-2-3-4` with branch (2, 5))
    with arbitrary user-specified spec ordering."""
    _expected_n = 6
    _edges_template = (
        (0, 1), (1, 2), (2, 3), (3, 4), (2, 5),
    )


class PermutedE7UQuiver(_PermutedENUQuiver):
    """E_7-Dynkin shape (7 nodes; chain `0-1-2-3-4-5` with branch (2, 6))
    with arbitrary user-specified spec ordering."""
    _expected_n = 7
    _edges_template = (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6),
    )


class PermutedE8UQuiver(_PermutedENUQuiver):
    """E_8-Dynkin shape (8 nodes; chain `0-1-2-3-4-5-6` with branch (2, 7))
    with arbitrary user-specified spec ordering."""
    _expected_n = 8
    _edges_template = (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (2, 7),
    )


# ---- Permuted affine ADE wrappers ---------------------------------------


class PermutedAffineDUQuiver(PermutedUQuiver):
    """Affine  ``D̂_{n}`` -shape unitary quiver  (n + 1 nodes, n >= 4)
    with arbitrary user-specified spec ordering.  Same edge layout as
    :class:`AffineDUQuiver` ."""

    def __init__(
        self,
        N_list: Sequence[int],
        M_list: Sequence[int] | None = None,
        *,
        spec_order: Sequence[Block],
        strict: bool = True,
        build_algebra: bool = True,
        verify_at_construction: bool = True,
        **kwargs,
    ):
        n = len(N_list) - 1
        if n < 4:
            raise ValueError(
                f"PermutedAffineDUQuiver: need n >= 4 (n + 1 nodes), got "
                f"{len(N_list)} nodes"
            )
        super().__init__(
            N_list, M_list, _affine_d_edges(n),
            spec_order=spec_order, strict=strict,
            build_algebra=build_algebra,
            verify_at_construction=verify_at_construction,
            **kwargs,
        )


class _PermutedAffineENUQuiver(PermutedUQuiver):
    """Internal base for the permuted Ê_6 / Ê_7 / Ê_8 wrappers."""

    _expected_n: int
    _edges_template: tuple[tuple[int, int], ...]

    def __init__(
        self,
        N_list: Sequence[int],
        M_list: Sequence[int] | None = None,
        *,
        spec_order: Sequence[Block],
        strict: bool = True,
        build_algebra: bool = True,
        verify_at_construction: bool = True,
        **kwargs,
    ):
        if len(N_list) != self._expected_n:
            raise ValueError(
                f"{type(self).__name__}: requires exactly "
                f"{self._expected_n} nodes, got {len(N_list)}"
            )
        super().__init__(
            N_list, M_list, list(self._edges_template),
            spec_order=spec_order, strict=strict,
            build_algebra=build_algebra,
            verify_at_construction=verify_at_construction,
            **kwargs,
        )


class PermutedAffineE6UQuiver(_PermutedAffineENUQuiver):
    """Affine Ê_6-shape (7 nodes) with arbitrary user-specified
    spec ordering."""
    _expected_n = 7
    _edges_template = (
        (0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6),
    )


class PermutedAffineE7UQuiver(_PermutedAffineENUQuiver):
    """Affine Ê_7-shape (8 nodes) with arbitrary user-specified
    spec ordering."""
    _expected_n = 8
    _edges_template = (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (0, 7),
    )


class PermutedAffineE8UQuiver(_PermutedAffineENUQuiver):
    """Affine Ê_8-shape (9 nodes) with arbitrary user-specified
    spec ordering."""
    _expected_n = 9
    _edges_template = (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (2, 7), (6, 8),
    )


def canonical_spec_order(N_list, M_list, edges, antisym_counts=None):
    """Build the canonical (default :class:`GraphUQuiver` ) spec_order.

    Each bifund edge  e = (i, j)  is routed to "gap i" (= before factor
    i in the linear ordering), interleaved per-factor with the per-mu
    antisym^2 + fund blocks (all "left" by default), then the pure-gauge
    spec:

        gap_0 + F_0 + gap_1 + F_1 + ... + gap_{k-1} + F_{k-1} + gap_k

    where  ``gap_g``  contains, in edge order, every bifund edge with
    ``min(i, j) == g`` , and

        F_i = [antisym_{i, 0}, ..., antisym_{i, n_a_i - 1},
               fund_{i, 0}, ..., fund_{i, M_i - 1},
               pure_i]

    A :class:`PermutedUQuiver`  built with
    ``spec_order=canonical_spec_order(...)``  matches the canonical
    :class:`GraphUQuiver`  spec.  Useful as a starting point for
    hand-permuting one or two blocks.
    """
    if antisym_counts is None:
        antisym_counts = [0] * len(N_list)
    k = len(N_list)
    gaps: list[list[Block]] = [[] for _ in range(k + 1)]
    for ei, (i, j) in enumerate(edges):
        gaps[min(i, j)].append(("bifund", ei))
    out: list[Block] = []
    for fi in range(k):
        out.extend(gaps[fi])
        for kk in range(antisym_counts[fi]):
            out.append(("antisym", fi, kk))
        for j in range(M_list[fi]):
            out.append(("fund", fi, j))
        out.append(("pure", fi))
    out.extend(gaps[k])
    return out


# ---------------------------------------------------------------------------
# Seed builders -- enumerate spec permutations for a given topology
# ---------------------------------------------------------------------------


def enumerate_seed_specs(
    N_list: Sequence[int],
    M_list: Sequence[int] | None = None,
    edges: Sequence[tuple[int, int]] = (),
    *,
    antisym_counts: Sequence[int] | None = None,
    strict: bool = True,
    max_specs: int = 64,
    permute_matter_only: bool = True,
):
    """Enumerate spec permutations for a given topology and return the
    subset that produce a valid negating sequence on the BPS quiver.

    Each yielded entry is a tuple  (spec_order, PermutedUQuiver) .

    The "all permutations" space is  (total blocks)!  which blows up
    fast.  With  ``permute_matter_only=True``  (default) the pure-gauge
    block positions are FIXED (in the canonical interleaved layout)
    and only the matter blocks  (bifund / fund / antisym)  routed to
    each "gap" between consecutive pure blocks are permuted.  This
    typically yields several valid orderings in the local-moves orbit
    of the canonical spec, each one a candidate seed.

    With  ``permute_matter_only=False``  every matter + pure block is
    permuted; this is exponentially more expensive and rarely useful
    beyond  k <= 3 .

    Parameters
    ----------
    N_list, M_list, edges, antisym_counts, strict
        Same as :class:`GraphUQuiver` .
    max_specs
        Cap on the number of valid specs yielded (default 64).  The
        full enumeration may have many more.

    Yields  (spec_order, algebra)  pairs.
    """
    import itertools

    if M_list is None:
        M_list = [0] * len(N_list)
    if antisym_counts is None:
        antisym_counts = [0] * len(N_list)

    k = len(N_list)
    blocks_per_gap: list[list[Block]] = [[] for _ in range(k + 1)]
    # Initial canonical assignment of matter blocks to "gaps" between
    # the (fixed) pure-gauge spec blocks.
    for ei, (i, j) in enumerate(edges):
        blocks_per_gap[min(i, j)].append(("bifund", ei))
    for fi in range(k):
        for kk in range(antisym_counts[fi]):
            blocks_per_gap[fi].append(("antisym", fi, kk))
        for j in range(M_list[fi]):
            blocks_per_gap[fi].append(("fund", fi, j))

    found = 0
    if permute_matter_only:
        # Cartesian product over per-gap permutations.
        gap_perms = [list(itertools.permutations(g)) for g in blocks_per_gap]
        for combo in itertools.product(*gap_perms):
            spec_order: list[Block] = []
            for fi in range(k):
                spec_order.extend(combo[fi])
                spec_order.append(("pure", fi))
            spec_order.extend(combo[k])
            try:
                m = PermutedUQuiver(
                    N_list, M_list, edges,
                    spec_order=spec_order, strict=strict,
                    antisym_counts=antisym_counts,
                    build_algebra=False,
                    verify_at_construction=True,
                )
            except (ValueError, RuntimeError):
                continue
            yield spec_order, m
            found += 1
            if found >= max_specs:
                return
    else:
        all_blocks = []
        for fi in range(k):
            all_blocks.extend(blocks_per_gap[fi])
        all_blocks.extend(blocks_per_gap[k])
        for fi in range(k):
            all_blocks.append(("pure", fi))
        for perm in itertools.permutations(all_blocks):
            try:
                m = PermutedUQuiver(
                    N_list, M_list, edges,
                    spec_order=list(perm), strict=strict,
                    antisym_counts=antisym_counts,
                    build_algebra=False,
                    verify_at_construction=True,
                )
            except (ValueError, RuntimeError):
                continue
            yield list(perm), m
            found += 1
            if found >= max_specs:
                return
