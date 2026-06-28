"""BPS chart mutations as `KAlgebraObject` populations.

Each chart of a `BPSKAlgebra`'s chart graph is a *presentation* of the
same abstract `A_𝖖[T]`: mutation (cluster necklacing — `S = E(X_g)·S₂
↦ S₂·E(X_{−g})`, i.e. `S₂·ρ_IR⁻¹(S₁)` with `S₁` the head factor)
produces the adjacent-chamber `BPSKAlgebra`, and the canonical-basis
labels (lower tropical charges) transport by the piecewise-linear
`μ_g` (chart_graph._mu_g).  That label map *is* a `KAlgebraIso`
witness — certified here by the full battery (unit, round-trip,
multiplicativity, ρ-equivariance, trace-equivariance) — so a chain of
mutations populates a `KAlgebraObject` whose realizations are the
chamber presentations.

The monodromy theorem-by-experiment (pentagon, `tests/
test_bps_chart_object.py`): the pure rotation walk (always necklace at
the spec head) returns the chart data to the root after consuming each
particle and antiparticle once, and the composed witness loop is
**ρ²** — the same ρ² that twists the trace cyclicity axiom.  The loop
composite is therefore a *non-trivial automorphism*: the closing edge
must not be curated as a trivial identification, and
`KAlgebraObject.verify_coherence` correctly rejects an object closed
that way (isos between two presentations form an `Aut`-torsor;
coherence certifies the curation).

Scope ("when the spec cooperates"): `mutate_bpskalgebra` requires the
chart graph to produce the necklaced chart within the local-move
budget; otherwise it raises.  The RG-level generalization — mutating
an arbitrary `RGKAlgebra` along a factorization `S_RG = S₁·S₂` — is
Plan 26.
"""
from __future__ import annotations

from bps_kalgebra import BPSKAlgebra
from chart_graph import ChartGraph, _mu_g, _mu_inv_g
from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from kalgebra_object import KAlgebraObject
from laurent_poly import LaurentPoly


__all__ = [
    "mutate_bpskalgebra",
    "mutation_path_iso",
    "bps_chart_object",
    "rotation_loop_charges",
    "chart_monodromy_iso",
    "bps_frame_change",
]

_ONE = LaurentPoly.one()


def _int_inverse(T):
    """Exact inverse of a unimodular integer matrix."""
    from fractions import Fraction
    n = len(T)
    A = [[Fraction(T[i][j]) for j in range(n)] + [Fraction(int(i == j))
         for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = next(r for r in range(col, n) if A[r][col] != 0)
        A[col], A[piv] = A[piv], A[col]
        inv = 1 / A[col][col]
        A[col] = [x * inv for x in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [x - f * y for x, y in zip(A[r], A[col])]
    out = [[A[i][n + j] for j in range(n)] for i in range(n)]
    if any(x.denominator != 1 for row in out for x in row):
        raise ValueError("bps_frame_change: T is not unimodular")
    return [[int(x) for x in row] for row in out]


def _apply(T, v):
    return tuple(sum(T[i][j] * v[j] for j in range(len(v)))
                 for i in range(len(T)))


def bps_frame_change(A: BPSKAlgebra, T):
    """Re-coordinatize a BPS flow by a unimodular `T : Γ → Γ'`
    (`γ' = Tγ`) and return `(A', witness)` where the witness is an
    `RGKAlgebraIso`: the SAME flow in parallel coordinates — UV labels
    and the IR torus both relabel by `T`, the pairing becomes
    `B' = (Tᵀ)⁻¹·B·T⁻¹`, and `S'_RG` is `S_RG` transported (so
    `verify_s_rg_match` holds — unlike a chart mutation, which changes
    the flow)."""
    from rgkalgebra_object import RGKAlgebraIso

    Tinv = _int_inverse(T)
    B = [list(r) for r in A.lattice.pairing]
    n = len(B)
    # B' = (T^T)^{-1} B T^{-1} = (Tinv)^T B Tinv
    Bp = [[sum(Tinv[a][i] * B[a][b] * Tinv[b][j]
               for a in range(n) for b in range(n))
           for j in range(n)] for i in range(n)]
    A2 = BPSKAlgebra(
        pairing=Bp,
        node_charges=[_apply(T, v) for v in A.node_charges],
        spec=[_apply(T, g) for g in A.spec],
    )

    def fw(label):
        return Element({_apply(T, label): _ONE})

    def iv(label):
        return Element({_apply(Tinv, label): _ONE})

    aux_iso = KAlgebraIso(A.auxiliary(), A2.auxiliary(), fw, iv,
                          name="aux-frame")
    return A2, RGKAlgebraIso(A, A2, fw, iv, aux_iso,
                             name=f"frame[{T}]")


def _graph_of(A: BPSKAlgebra) -> ChartGraph:
    pairing = [list(r) for r in A.lattice.pairing]
    return ChartGraph(nodes=list(A.node_charges), spec=list(A.spec),
                      pairing=pairing)


def _mu_iso(src: BPSKAlgebra, dst: BPSKAlgebra, charge, pairing,
            name=None) -> KAlgebraIso:
    """The μ_g canonical-label witness across one forward necklace
    edge with consumed charge `charge`."""
    def fw(label):
        return Element({_mu_g(label, charge, pairing): _ONE})

    def iv(label):
        return Element({_mu_inv_g(label, charge, pairing): _ONE})

    return KAlgebraIso(src, dst, fw, iv, name=name)


def mutate_bpskalgebra(
    A: BPSKAlgebra, node_index: int, *, max_local_moves: int = 0,
) -> tuple[BPSKAlgebra, KAlgebraIso]:
    """Necklace `A` forward at `node_index` and return
    `(A_mutated, witness)` with the μ_g label-map `KAlgebraIso`.

    `max_local_moves=0` demands the strict "spec cooperates" case (the
    node charge already heads the spec); a positive budget allows
    pentagon-collapse / commute rearrangements first.  Raises
    `ValueError` when the chart graph cannot produce the mutation
    within budget."""
    g = _graph_of(A)
    dst_id = g.mutate(g.root_id, node_index, direction="fwd",
                      max_local_moves=max_local_moves)
    if dst_id is None:
        raise ValueError(
            f"mutate_bpskalgebra: spec does not cooperate at node "
            f"{node_index} within {max_local_moves} local moves")
    edge = g._edges[(g.root_id, dst_id)]
    chart = g.chart(dst_id)
    pairing = [list(r) for r in A.lattice.pairing]
    A2 = BPSKAlgebra(pairing=pairing, node_charges=chart.nodes,
                     spec=chart.spec)
    iso = _mu_iso(A, A2, edge.charge, pairing,
                  name=f"necklace[{edge.charge}]")
    return A2, iso


def rotation_loop_charges(A: BPSKAlgebra, max_steps: int = 64
                          ) -> list[tuple]:
    """Consumed charges of the pure rotation walk (necklace at the
    spec head each step) until the chart data returns to the root.
    Raises if it does not close within `max_steps`."""
    g = _graph_of(A)
    root_state = (tuple(map(tuple, g.root().nodes)),
                  tuple(map(tuple, g.root().spec)))
    cur = g.root_id
    charges: list[tuple] = []
    for _ in range(max_steps):
        ch = g.chart(cur)
        head = tuple(ch.spec[0])
        k = next(i for i, n in enumerate(ch.nodes) if tuple(n) == head)
        dst = g.mutate(cur, k, direction="fwd", max_local_moves=0)
        if dst is None:
            raise ValueError("rotation walk: spec stopped cooperating")
        charges.append(g._edges[(cur, dst)].charge)
        cur = dst
        ch2 = g.chart(cur)
        if (tuple(map(tuple, ch2.nodes)),
                tuple(map(tuple, ch2.spec))) == root_state:
            return charges
    raise ValueError(
        f"rotation walk did not close within {max_steps} steps")


def mutation_path_iso(
    A: BPSKAlgebra, node_sequence, *, max_local_moves: int = 0,
) -> "tuple[BPSKAlgebra, KAlgebraIso]":
    """Compose the per-edge μ_g canonical-label witnesses along a mutation
    **path** into a single `KAlgebraIso A → A_end`.

    `node_sequence` lists the steps to take, each at the *current* chart
    (successive entries refer to successive charts, exactly as a cluster mutation
    path).  A step is either a node index `k` (forward necklace) or a pair
    `(k, direction)` with `direction in {"fwd", "inv"}` — a general mutation path
    uses both directions.  Returns `(A_end, iso)` where `A_end` is the BPSKAlgebra
    of the end chart and `iso` is the composed witness (each edge transports by
    `μ_g` for a `fwd` step, `μ_g⁻¹` for an `inv` step).  For a *closing* path the
    end chart equals the start and `iso` is an automorphism witness —
    `chart_monodromy_iso` is the special case (spec head, all `fwd`, = ρ²).

    `max_local_moves` is forwarded to `ChartGraph.mutate` (0 = strict "spec
    cooperates"; positive allows pentagon-collapse / commute rearrangements to
    bring the target node to the head/tail first — these are canonical-basis
    preserving, so the per-edge transport stays a single `μ_g`).  Raises
    `ValueError` if a step does not cooperate."""
    g = _graph_of(A)
    pairing = [list(r) for r in A.lattice.pairing]
    cur = g.root_id
    edges: list = []   # (direction, consumed charge) per step
    for step, s in enumerate(node_sequence):
        k, direction = (s, "fwd") if isinstance(s, int) else (s[0], s[1])
        dst = g.mutate(cur, k, direction=direction,
                       max_local_moves=max_local_moves)
        if dst is None:
            raise ValueError(
                f"mutation_path_iso: step {step} ({k}, {direction}) does not "
                f"cooperate within {max_local_moves} local moves")
        edges.append((direction, g._edges[(cur, dst)].charge))
        cur = dst
    chart = g.chart(cur)
    A_end = BPSKAlgebra(pairing=pairing, node_charges=chart.nodes,
                        spec=chart.spec)

    # Lower-tropical label transport per edge.  Forward necklace at head `g`:
    # μ_g(·, g).  Inverse necklace records the tail `h = edge.charge` but
    # transports by μ_g⁻¹(·, −h) (see chart_graph._necklace_inverse: new_l =
    # _mu_inv_g(l, −h)) -- the sign on the inverse edge is essential.
    def _neg(c):
        return tuple(-x for x in c)

    def fw(label):
        for direction, c in edges:
            label = (_mu_g(label, c, pairing) if direction == "fwd"
                     else _mu_inv_g(label, _neg(c), pairing))
        return Element({label: _ONE})

    def iv(label):
        for direction, c in reversed(edges):
            label = (_mu_inv_g(label, c, pairing) if direction == "fwd"
                     else _mu_g(label, _neg(c), pairing))
        return Element({label: _ONE})

    iso = KAlgebraIso(A, A_end, fw, iv,
                      name=f"mutation-path{list(node_sequence)}")
    return A_end, iso


def chart_monodromy_iso(A: BPSKAlgebra, charges=None) -> KAlgebraIso:
    """The composed witness of the full rotation loop, as an
    automorphism witness `A → A`.  Conjecturally (verified on the
    pentagon) this is ρ²."""
    if charges is None:
        charges = rotation_loop_charges(A)
    pairing = [list(r) for r in A.lattice.pairing]

    def fw(label):
        for c in charges:
            label = _mu_g(label, c, pairing)
        return Element({label: _ONE})

    def iv(label):
        for c in reversed(charges):
            label = _mu_inv_g(label, c, pairing)
        return Element({label: _ONE})

    return KAlgebraIso(A, A, fw, iv, name="rotation-monodromy")


def bps_chart_object(A: BPSKAlgebra, n_charts: int,
                     name: str = "bps-charts") -> KAlgebraObject:
    """A `KAlgebraObject` populated by the first `n_charts` chambers of
    the pure rotation walk (keys ``'chart-0'``, ``'chart-1'``, …),
    witnesses = the per-edge μ_g maps.  The walk's closing edge is
    deliberately NOT added (the loop composite is the non-trivial
    monodromy ρ², not an identification — see module docstring)."""
    obj = KAlgebraObject(name)
    obj.add_realization("chart-0", A, {"chart", "trace-exact", "rg"})
    g = _graph_of(A)
    pairing = [list(r) for r in A.lattice.pairing]
    cur = g.root_id
    prev_alg = A
    for step in range(1, n_charts):
        ch = g.chart(cur)
        head = tuple(ch.spec[0])
        k = next(i for i, n in enumerate(ch.nodes) if tuple(n) == head)
        dst = g.mutate(cur, k, direction="fwd", max_local_moves=0)
        if dst is None:
            raise ValueError(
                f"bps_chart_object: spec stopped cooperating at step "
                f"{step}")
        edge = g._edges[(cur, dst)]
        chart = g.chart(dst)
        alg = BPSKAlgebra(pairing=pairing, node_charges=chart.nodes,
                          spec=chart.spec)
        obj.add_realization(f"chart-{step}", alg,
                            {"chart", "trace-exact", "rg"})
        obj.add_iso(f"chart-{step-1}", f"chart-{step}",
                    _mu_iso(prev_alg, alg, edge.charge, pairing,
                            name=f"{name}[{step-1}→{step}]"))
        cur = dst
        prev_alg = alg
    return obj
