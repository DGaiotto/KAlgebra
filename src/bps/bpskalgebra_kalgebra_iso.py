"""
bpskalgebra_kalgebra_iso.py
===========================

Construct a `KAlgebraIso` between two `BPSKAlgebra`s related by BPS
quiver mutation.

Tropical mutation matches labels
--------------------------------
Two `BPSKAlgebra`s presenting the same `A_𝖖[T]` differently are
related by:

  * chart-graph mutations *within* each algebra (each preserves
    `S` up to conjugation by `E_q(X_g)`; labels transform by the
    tropical mutation rule `μ_g(γ) = γ + max(⟨γ, g⟩, 0) · g` at
    every forward step, and by `μ_g⁻¹` at every inverse step), plus
  * a residual unimodular relabel `A` (preserving the pairing) when
    the end-charts of the two walks match only up to a basis change.

This is exactly the data captured by
:class:`bpskalgebra_iso.WeakIso`: ``chain_1`` walks the source's
chart graph, ``chain_2`` walks the target's, and the inner
:class:`bpskalgebra_iso.Iso` records the residual `A` (plus a
local-move chain bridging the two specs at the meet point).

The induced K-algebra iso at the label level is the composition

    γ_1 ─chain_1 forward μ─▶ γ_C1
         ─A inner (linear)─▶ γ_C2
         ─chain_2 backward μ⁻¹─▶ γ_2

with the inverse map symmetric.  In the strong-`Iso` special case
(``chain_1 = chain_2 = []``), only the `A`-relabel applies — the
familiar `γ ↦ A · γ` map.

The user is encouraged to verify the constructed iso on a small
label window via :meth:`kalgebra_iso.KAlgebraIso.verify_all`; the
construction is *necessarily* correct if the witness is correct, but
local sanity checks catch bookkeeping errors cheaply.

API
---
* :func:`kalgebra_iso_from_witness(alg_1, alg_2, iso)`
    Promote a strong `Iso` witness into a `KAlgebraIso`.  The label
    map is the linear relabel ``γ ↦ A · γ``.
* :func:`kalgebra_iso_from_weak_witness(alg_1, alg_2, weak)`
    Promote a `WeakIso` witness into a `KAlgebraIso`.  The label map
    is the chain_1 → A_inner → chain_2-reversed composition.
* :func:`find_kalgebra_iso(alg_1, alg_2, ...)`
    High-level entry point: try the strong path first, fall back to
    the weak path on chart-graph depth limits.
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction
from typing import Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly
from bpskalgebra_iso import (
    Iso,
    WeakIso,
    find_isomorphism,
    find_weak_isomorphism,
    _solve_AM_eq_N,
)
from chart_graph import _mu_g, _mu_inv_g
from snf_kernel import int_det as _int_det


Vec = tuple[int, ...]


# ---------------------------------------------------------------------------
# Linear-algebra helpers (unimodular inverse + matrix-vector apply)
# ---------------------------------------------------------------------------


def _invert_unimodular(A: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Exact integer inverse of a unimodular integer matrix.

    `A` is assumed `det(A) ∈ {-1, +1}`; the inverse is then integer.
    """
    n = len(A)
    det = _int_det([list(row) for row in A])
    if det not in (-1, 1):
        raise ValueError(
            f"_invert_unimodular: matrix is not unimodular (det={det})"
        )
    identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    inv_q = _solve_AM_eq_N([list(row) for row in A], identity)
    if inv_q is None:
        raise ValueError(
            "_invert_unimodular: singular (unreachable for unimodular A)"
        )
    out: list[tuple[int, ...]] = []
    for row in inv_q:
        cast_row: list[int] = []
        for x in row:
            if isinstance(x, Fraction):
                if x.denominator != 1:
                    raise ValueError(
                        "_invert_unimodular: non-integer entry in inverse "
                        "of supposedly-unimodular matrix"
                    )
                cast_row.append(int(x.numerator))
            else:
                cast_row.append(int(x))
        out.append(tuple(cast_row))
    return tuple(out)


def _apply_matrix(A: Sequence[Sequence[int]], v: Sequence[int]) -> Vec:
    """`A · v` for square `A` and column vector `v`, returned as a tuple."""
    rank = len(A)
    return tuple(
        sum(A[i][j] * v[j] for j in range(rank)) for i in range(rank)
    )


# ---------------------------------------------------------------------------
# Chart-graph chain walking: collect per-step (node-charge, direction)
# data so labels can be replayed under tropical mutation.
# ---------------------------------------------------------------------------


def _collect_chain_steps(
    alg, chain: Sequence, *, max_local_moves: int = 64,
) -> list[tuple[Vec, str]]:
    """Walk `chain` on `alg`'s chart graph and collect each step's
    source-chart node-charge.  Returns a list of `(charge, direction)`
    tuples ready for tropical-mutation replay.

    Each `chain` entry is `(node_index, direction)`.  At each step
    `i`, the *source* chart's `nodes[node_index]` is the lattice
    point used for `_mu_g` (forward) or `_mu_inv_g` (inverse).
    """
    graph = alg._chart_graph
    if graph is None:
        if chain:
            raise ValueError(
                "_collect_chain_steps: algebra has no chart graph but "
                "a non-empty chain was supplied"
            )
        return []
    cur_id = graph.root_id
    steps: list[tuple[Vec, str]] = []
    for (k, direction) in chain:
        chart = graph.chart(cur_id)
        if not (0 <= k < len(chart.nodes)):
            raise IndexError(
                f"_collect_chain_steps: step has node_index={k} but "
                f"chart at id {cur_id} has {len(chart.nodes)} nodes"
            )
        if direction not in ("fwd", "inv"):
            raise ValueError(
                f"_collect_chain_steps: direction={direction!r} "
                f"must be 'fwd' or 'inv'"
            )
        steps.append((tuple(chart.nodes[k]), direction))
        dst_id = graph.mutate(
            cur_id, k, direction=direction,
            max_local_moves=max_local_moves,
        )
        if dst_id is None:
            raise ValueError(
                f"_collect_chain_steps: chart-graph mutation at step "
                f"(k={k}, direction={direction!r}) is inadmissible from "
                f"chart_id={cur_id}"
            )
        cur_id = dst_id
    return steps


def _apply_chain_forward(
    gamma: Vec,
    steps: list[tuple[Vec, str]],
    pairing: Sequence[Sequence[int]],
) -> Vec:
    """Replay `steps` forward on a label.  Each `(charge, direction)`
    step transforms `γ` by `_mu_g(γ, charge, B)` (forward) or
    `_mu_inv_g(γ, -charge, B)` (inverse).

    The exchange matrix `B` is the algebra's lattice pairing, which
    is *invariant* across the chart graph (chart-graph mutation
    rebuilds nodes/spec but keeps the underlying lattice).
    """
    pairing_l = [list(row) for row in pairing]
    g = tuple(gamma)
    for (charge, direction) in steps:
        if direction == "fwd":
            g = _mu_g(g, charge, pairing_l)
        else:  # 'inv'
            neg = tuple(-c for c in charge)
            g = _mu_inv_g(g, neg, pairing_l)
    return g


def _apply_chain_backward(
    gamma: Vec,
    steps: list[tuple[Vec, str]],
    pairing: Sequence[Sequence[int]],
) -> Vec:
    """Replay `steps` *backward* on a label.  Inverse of
    :func:`_apply_chain_forward` — each step undoes the corresponding
    forward step in reverse order.
    """
    pairing_l = [list(row) for row in pairing]
    g = tuple(gamma)
    for (charge, direction) in reversed(steps):
        if direction == "fwd":
            # Undo `_mu_g(·, charge)` with `_mu_inv_g(·, charge)`.
            g = _mu_inv_g(g, charge, pairing_l)
        else:
            # Undo `_mu_inv_g(·, -charge)` with `_mu_g(·, -charge)`.
            neg = tuple(-c for c in charge)
            g = _mu_g(g, neg, pairing_l)
    return g


# ---------------------------------------------------------------------------
# Witness → KAlgebraIso
# ---------------------------------------------------------------------------


def kalgebra_iso_from_witness(
    alg_1, alg_2, iso: Iso, *, name: Optional[str] = None,
) -> KAlgebraIso:
    """Promote a strong `Iso` witness into a `KAlgebraIso`.

    The `Iso` witness `(A, local_move_chain)` certifies that the
    unimodular relabel `A : Γ_1 → Γ_2` plus local moves on
    `A · alg_1.spec` reaches `alg_2.spec`.  Local moves preserve `S`,
    so the label-level iso is just the linear relabel:

        forward : γ_1 ↦ Element({ A · γ_1 : 1 })
        inverse : γ_2 ↦ Element({ A⁻¹ · γ_2 : 1 })

    This is the trivial-chain special case of
    :func:`kalgebra_iso_from_weak_witness`.
    """
    rank = alg_1.lattice.rank
    if alg_2.lattice.rank != rank:
        raise ValueError(
            f"kalgebra_iso_from_witness: lattice ranks differ "
            f"({rank} vs {alg_2.lattice.rank})"
        )
    A = tuple(tuple(row) for row in iso.A)
    A_inv = _invert_unimodular(A)
    one = LaurentPoly.one()

    def forward(gamma_1):
        return Element({_apply_matrix(A, gamma_1): one})

    def inverse(gamma_2):
        return Element({_apply_matrix(A_inv, gamma_2): one})

    if name is None:
        chain_len = len(iso.local_move_chain)
        name = (
            f"BPSKAlgebra ≅ BPSKAlgebra "
            f"(A={A}, local moves: {chain_len})"
        )
    return KAlgebraIso(
        alg_1, alg_2,
        forward_label_map=forward,
        inverse_label_map=inverse,
        name=name,
    )


def kalgebra_iso_from_weak_witness(
    alg_1, alg_2, weak: WeakIso, *,
    name: Optional[str] = None,
    max_local_moves: int = 64,
) -> KAlgebraIso:
    """Promote a `WeakIso` witness into a `KAlgebraIso`.

    The composition (forward direction):

      1. Replay `weak.chain_1` on `alg_1`'s chart graph: each
         `(k, direction)` step transforms the label by tropical
         mutation `_mu_g` (forward) or `_mu_inv_g` (inverse) at the
         source chart's `nodes[k]`.  This brings `γ_1` to its label
         `γ_C1` at the end chart `C_1`.
      2. Apply the inner Iso's linear `A`: `γ_C2 = A · γ_C1`.
      3. Replay `weak.chain_2` *backward* on `alg_2`'s chart graph:
         brings `γ_C2` to its label `γ_2` at the root of `alg_2`.

    The inverse direction reverses each phase symmetrically.
    """
    rank = alg_1.lattice.rank
    if alg_2.lattice.rank != rank:
        raise ValueError(
            f"kalgebra_iso_from_weak_witness: lattice ranks differ "
            f"({rank} vs {alg_2.lattice.rank})"
        )
    A = tuple(tuple(row) for row in weak.inner.A)
    A_inv = _invert_unimodular(A)
    one = LaurentPoly.one()
    pairing_1 = [list(row) for row in alg_1.lattice.pairing]
    pairing_2 = [list(row) for row in alg_2.lattice.pairing]

    steps_1 = _collect_chain_steps(
        alg_1, weak.chain_1, max_local_moves=max_local_moves,
    )
    steps_2 = _collect_chain_steps(
        alg_2, weak.chain_2, max_local_moves=max_local_moves,
    )

    def forward(gamma_1):
        g_c1 = _apply_chain_forward(gamma_1, steps_1, pairing_1)
        g_c2 = _apply_matrix(A, g_c1)
        g_2 = _apply_chain_backward(g_c2, steps_2, pairing_2)
        return Element({g_2: one})

    def inverse(gamma_2):
        g_c2 = _apply_chain_forward(gamma_2, steps_2, pairing_2)
        g_c1 = _apply_matrix(A_inv, g_c2)
        g_1 = _apply_chain_backward(g_c1, steps_1, pairing_1)
        return Element({g_1: one})

    if name is None:
        c1_str = ", ".join(f"({k},'{d}')" for k, d in weak.chain_1)
        c2_str = ", ".join(f"({k},'{d}')" for k, d in weak.chain_2)
        name = (
            f"BPSKAlgebra ≅ BPSKAlgebra (weak: "
            f"chain_1=[{c1_str}], chain_2=[{c2_str}], "
            f"inner A={A})"
        )
    return KAlgebraIso(
        alg_1, alg_2,
        forward_label_map=forward,
        inverse_label_map=inverse,
        name=name,
    )


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def find_kalgebra_iso(
    alg_1, alg_2,
    *,
    max_states: int = 4096,
    max_extra_length: int = 4,
    bidirectional: bool = True,
    weak: bool = True,
    max_depth_1: int = 1,
    max_depth_2: int = 1,
) -> Optional[KAlgebraIso]:
    """Search for an iso witness between `alg_1` and `alg_2` and
    return it as a `KAlgebraIso`.

    Tries `find_isomorphism` first (strong path); if that fails and
    `weak=True`, falls back to `find_weak_isomorphism` with the given
    chart-graph depths.

    Returns `None` if neither finder produces a witness within the
    given budgets.  `None` is *not* a proof of non-iso — only the
    absence of a certificate.
    """
    iso = find_isomorphism(
        alg_1, alg_2,
        max_states=max_states,
        max_extra_length=max_extra_length,
        bidirectional=bidirectional,
    )
    if iso is not None:
        return kalgebra_iso_from_witness(alg_1, alg_2, iso)
    if not weak:
        return None
    wk = find_weak_isomorphism(
        alg_1, alg_2,
        max_depth_1=max_depth_1,
        max_depth_2=max_depth_2,
        max_states=max_states,
        max_extra_length=max_extra_length,
        bidirectional=bidirectional,
    )
    if wk is None:
        return None
    return kalgebra_iso_from_weak_witness(alg_1, alg_2, wk)


# ---------------------------------------------------------------------------
# Verification-sample helper
# ---------------------------------------------------------------------------


def default_iso_samples(alg) -> list:
    """A small curated label window for `KAlgebraIso.verify_*`.

    Includes the identity, each node charge, each `ρ(node)` image,
    and each pairwise sum of node charges — exercises generators,
    σ-twists, and one composite layer.
    """
    one = LaurentPoly.one()
    labels: list[Vec] = [alg.identity()]
    for g in alg.node_charges:
        gt = tuple(g)
        if gt not in labels:
            labels.append(gt)
    for g in list(labels):
        if g == alg.identity():
            continue
        rg = alg.rho(g)
        if rg not in labels:
            labels.append(rg)
    nodes = [tuple(g) for g in alg.node_charges]
    for i, gi in enumerate(nodes):
        for j, gj in enumerate(nodes):
            if i == j:
                continue
            s = tuple(a + b for a, b in zip(gi, gj))
            if s not in labels:
                labels.append(s)
    return [Element({lbl: one}) for lbl in labels]


if __name__ == "__main__":
    from bps_kalgebra import BPSKAlgebra

    print("Demo 1: self-iso via strong witness (chain=[]):")
    A = BPSKAlgebra(
        pairing=[[0, 1], [-1, 0]],
        node_charges=[(1, 0), (0, 1)],
        verify="off",
    )
    iso = find_kalgebra_iso(A, A)
    print(f"  {iso}")
    src = default_iso_samples(A)
    pairs = [(a, b) for a in src for b in src]
    results = iso.verify_all(
        source_samples=src,
        target_samples=[iso.map(s) for s in src],
        source_pairs=pairs,
        target_pairs=[(iso.map(a), iso.map(b)) for a, b in pairs],
        trace_K=6,
    )
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")

    print("\nDemo 2: pentagon ≅ FZ-mutated pentagon (different B):")
    A1 = BPSKAlgebra(
        pairing=[[0, 1], [-1, 0]],
        node_charges=[(1, 0), (0, 1)],
        spec=[(1, 0), (0, 1)],
        verify="off",
    )
    A2 = BPSKAlgebra(
        pairing=[[0, -1], [1, 0]],
        node_charges=[(1, 0), (0, 1)],
        spec=[(0, 1), (1, 0)],
        verify="off",
    )
    iso_mut = find_kalgebra_iso(A1, A2)
    print(f"  {iso_mut}")
    src = default_iso_samples(A1)
    pairs = [(a, b) for a in src for b in src]
    results = iso_mut.verify_all(
        source_samples=src,
        target_samples=[iso_mut.map(s) for s in src],
        source_pairs=pairs,
        target_pairs=[(iso_mut.map(a), iso_mut.map(b)) for a, b in pairs],
        trace_K=4,
    )
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")
