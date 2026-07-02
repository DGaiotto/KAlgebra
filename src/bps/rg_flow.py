"""Single-node and subquiver RG flow on a `BPSKAlgebra`.

Given a `BPSKAlgebra` `A` with BPS-quiver data
`(pairing B, node_charges = [γ_1, ..., γ_n], spec = [g_1, ..., g_N], F-cache)`
and a node index `j`, the **single-node RG flow** produces a new
`BPSKAlgebra` `A_new` for the theory obtained by deleting the `j`-th
BPS-quiver node from the root chart.  Conceptually this is the
analogue of "freezing out" / integrating out the matter direction
labelled by `γ_j`.

The recipe:

  * Γ is unchanged; pairing `B` is unchanged.
  * `node_charges_new = [γ_i : i ≠ j]`.
  * `spec_new = [g : g ∈ spec, n_j(g) == 0]`, where
    `g = Σ_i n_i(g) · γ_i` is the unique non-negative integer
    decomposition of `g` in the **old** node-charge basis.  Spec
    entries containing `γ_j` get dropped.
  * For each cached `F_a` written as
    `F_a = Σ_δ c_δ(q) X_δ`,  the new theory's `F^new_a` keeps only
    those summands whose offset `δ - γ_a` decomposes (in the old
    node-charge basis) with `n_j == 0`:

        F^new_a  =  Σ_{δ : n_j(δ - γ_a) == 0}  c_δ(q) X_δ.

Tropical labels themselves are **not** required to be non-negative
combinations of node charges, but the offsets `δ - γ_a` always are
-- this is the doubly-tropical interval property of the canonical
basis.

The "RG flow map" itself (an algebra morphism
`A → A_new` whose values can be read off by grouping the dropped
pieces of each `F_a` by the power of `γ_j`) and the "RG generator"
(the `(stuff)` factor with `S_old = (stuff) · S_new`) are not
produced by this plain node-drop function; they are supplied by the
flow classes below (`SubquiverRG` / `SingleNodeRG`), which derive the
whole `RGKAlgebra` API from the flow data.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence, TYPE_CHECKING

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from rgkalgebra import RGKAlgebra, ComposedRG
from zplus_ring import RLaurent

if TYPE_CHECKING:
    from bps_kalgebra import BPSKAlgebra
    from habiro import HabiroElement


Vec = tuple[int, ...]


# ---------------------------------------------------------------------------
# Decomposition primitive
# ---------------------------------------------------------------------------


def _standard_basis_positions(
    node_charges: Sequence[Sequence[int]],
) -> list[int] | None:
    """If every entry in `node_charges` is a distinct standard-basis
    vector `e_p` (i.e. exactly one coordinate equals 1, the rest 0,
    and the positions are all distinct), return the list of positions
    `[p_0, p_1, ...]`; else return `None`.

    Used by `decompose_nonneg_in_node_basis` to short-circuit the
    rational Gauss-Jordan in the common case (the dictionary builder
    always uses standard-basis node charges).
    """
    seen = set()
    positions = []
    for g in node_charges:
        nz_pos = -1
        for k, x in enumerate(g):
            if x == 0:
                continue
            if nz_pos != -1 or x != 1:
                return None
            nz_pos = k
        if nz_pos == -1 or nz_pos in seen:
            return None
        seen.add(nz_pos)
        positions.append(nz_pos)
    return positions


def decompose_nonneg_in_node_basis_bulk(
    node_charges: Sequence[Sequence[int]],
    targets: Sequence[Sequence[int]],
) -> list | None:
    """Bulk version of `decompose_nonneg_in_node_basis`: solves
    `M · n_i = target_i` for *all* `target_i` in one Gauss-Jordan pass.

    Returns `[decomp_or_None for target in targets]`.  When `M` (the
    matrix whose columns are `node_charges`) is rank-deficient, the
    return is the full `None`, mirroring the single-target function's
    "decline rather than guess" semantics.

    Per-call cost: one `O(n³)` Gauss-Jordan + `len(targets)` constant-
    time decomp lookups, vs the per-target function's `len(targets) ·
    O(n³)`.  This is the fix for `_restandardize` in the dictionary
    builder, which previously decomposed each spec entry independently
    against the same `M`.

    Standard-basis fast path (same as the single-target version)
    short-circuits the whole thing when `node_charges` is a permuted
    standard basis.
    """
    k = len(node_charges)
    targets = [list(t) for t in targets]
    if not targets:
        return []
    r = len(targets[0])
    if k == 0:
        return [tuple() if all(t == 0 for t in tg) else None
                for tg in targets]

    # Standard-basis fast path: same as the single-target version,
    # applied to every target.
    positions = _standard_basis_positions(node_charges)
    if positions is not None:
        pos_set = set(positions)
        out = []
        for tg in targets:
            ok = all(tg[i] == 0 for i in range(r) if i not in pos_set)
            if not ok:
                out.append(None)
                continue
            row: list[int] = []
            valid = True
            for p in positions:
                v = tg[p]
                if v < 0:
                    valid = False
                    break
                row.append(int(v))
            out.append(tuple(row) if valid else None)
        return out

    # General path: one Gauss-Jordan over [M | T_1 | T_2 | ... | T_m].
    m = len(targets)
    aug: list[list[Fraction]] = []
    for i in range(r):
        row = [Fraction(node_charges[c][i]) for c in range(k)]
        for tg in targets:
            row.append(Fraction(tg[i]))
        aug.append(row)

    pivot_for_col: dict[int, int] = {}
    next_pivot_row = 0
    for c in range(k):
        piv = None
        for i in range(next_pivot_row, r):
            if aug[i][c] != 0:
                piv = i
                break
        if piv is None:
            return None  # rank-deficient: decline.
        aug[next_pivot_row], aug[piv] = aug[piv], aug[next_pivot_row]
        pv = aug[next_pivot_row][c]
        aug[next_pivot_row] = [x / pv for x in aug[next_pivot_row]]
        for i in range(r):
            if i != next_pivot_row and aug[i][c] != 0:
                f = aug[i][c]
                aug[i] = [
                    aug[i][col] - f * aug[next_pivot_row][col]
                    for col in range(k + m)
                ]
        pivot_for_col[c] = next_pivot_row
        next_pivot_row += 1

    # Per-target consistency + non-neg-integer check.
    out_results: list = []
    for ti in range(m):
        col_idx = k + ti
        # Consistency: rows beyond the pivots must vanish in this target column.
        ok = True
        for i in range(next_pivot_row, r):
            if aug[i][col_idx] != 0:
                ok = False
                break
        if not ok:
            out_results.append(None)
            continue
        decomp: list[int] = []
        for c in range(k):
            v = aug[pivot_for_col[c]][col_idx]
            if v.denominator != 1 or v.numerator < 0:
                decomp = None
                break
            decomp.append(v.numerator)
        out_results.append(tuple(decomp) if decomp is not None else None)
    return out_results


def decompose_nonneg_in_node_basis(
    node_charges: Sequence[Sequence[int]],
    target: Sequence[int],
) -> tuple[int, ...] | None:
    """Express `target` as `Σ_i n_i · γ_i` with non-negative integer `n_i`.

    Solves the linear system `M · n = target` where `M` is the matrix
    whose columns are `node_charges`, by Gauss-Jordan elimination over
    the rationals; succeeds iff the system is consistent, the solution
    is unique, integer-valued, and non-negative.

    Returns
    -------
    tuple[int, ...] or None
        The coefficient tuple `(n_1, ..., n_k)` if such a decomposition
        exists uniquely; `None` otherwise.

    Notes
    -----
    Uniqueness holds exactly when the columns of `M` are linearly
    independent over Q.  When they are not (`k > rank(M)`), this
    function conservatively returns `None` rather than picking one of
    the infinitely many solutions; the caller can decide whether to
    fall back to a more elaborate enumeration.  For the BPS-quiver
    cases the single-node RG flow is designed for (`node_charges`
    spanning a sublattice that contains all spec / F-summand offsets
    as positive combinations), independence holds.

    Fast path: when `node_charges` is a list of distinct standard-
    basis vectors (the common case from the dictionary builder), the
    matrix `M` is a permuted identity and the decomposition reduces
    to picking out the corresponding coordinates of `target` -- no
    Fraction arithmetic, no Gauss-Jordan.  Falls through to the
    general rational-solve path when nodes aren't standard-basis.
    """
    k = len(node_charges)
    target = list(target)
    r = len(target)
    if k == 0:
        return tuple() if all(t == 0 for t in target) else None

    # Standard-basis fast path.
    positions = _standard_basis_positions(node_charges)
    if positions is not None:
        # M @ n = target  =>  n[j] = target[positions[j]] for j in 0..k-1.
        # Components of target outside the selected positions must be 0
        # for consistency.
        pos_set = set(positions)
        for i in range(r):
            if i in pos_set:
                continue
            if target[i] != 0:
                return None
        result: list[int] = []
        for p in positions:
            v = target[p]
            if v < 0:
                return None
            result.append(int(v))
        return tuple(result)

    # Augmented matrix: r rows × (k + 1) columns (node-charge columns | target).
    aug: list[list[Fraction]] = []
    for i in range(r):
        row = [Fraction(node_charges[c][i]) for c in range(k)]
        row.append(Fraction(target[i]))
        aug.append(row)

    pivot_for_col: dict[int, int] = {}
    next_pivot_row = 0
    for c in range(k):
        # Find a pivot in column c at or below `next_pivot_row`.
        piv = None
        for i in range(next_pivot_row, r):
            if aug[i][c] != 0:
                piv = i
                break
        if piv is None:
            # Column c is in the span of earlier columns -- columns are
            # linearly dependent, so the decomposition (if it exists) is
            # not unique.  Decline.
            return None
        aug[next_pivot_row], aug[piv] = aug[piv], aug[next_pivot_row]
        pv = aug[next_pivot_row][c]
        aug[next_pivot_row] = [x / pv for x in aug[next_pivot_row]]
        for i in range(r):
            if i != next_pivot_row and aug[i][c] != 0:
                f = aug[i][c]
                aug[i] = [
                    aug[i][col] - f * aug[next_pivot_row][col]
                    for col in range(k + 1)
                ]
        pivot_for_col[c] = next_pivot_row
        next_pivot_row += 1

    # Consistency: all rows beyond the pivots must vanish in the last col.
    for i in range(next_pivot_row, r):
        if aug[i][k] != 0:
            return None

    result: list[int] = []
    for c in range(k):
        v = aug[pivot_for_col[c]][k]
        if v.denominator != 1 or v.numerator < 0:
            return None
        result.append(v.numerator)
    return tuple(result)


# ---------------------------------------------------------------------------
# Single-node RG flow
# ---------------------------------------------------------------------------


def single_node_rg_flow(
    A: "BPSKAlgebra",
    node_index: int,
) -> "BPSKAlgebra":
    """The `BPSKAlgebra` obtained from `A` by deleting the
    `node_index`-th BPS-quiver node.

    See module docstring for the recipe.

    Parameters
    ----------
    A
        Source algebra.  Must be in spec mode (recipe mode has no
        spec to filter).
    node_index
        Which node of `A.node_charges` to remove (0-based).

    Returns
    -------
    BPSKAlgebra
        New algebra with one fewer BPS-quiver node, same Γ, same `B`.
        F-cache is pre-populated with the filtered restrictions of
        `A`'s cached F's; the rest are computed lazily on demand.

    Raises
    ------
    NotImplementedError
        If `A` is in recipe mode (no spec to filter).
    ValueError
        If `node_index` is out of range; if `A`'s `node_charges` are
        linearly dependent (decomposition not unique); if some spec
        charge or F-summand offset has no non-negative integer
        decomposition in the old node-charge basis (recipe doesn't
        apply to this theory + node).
    """
    # Local import to avoid a top-level cycle (BPSKAlgebra exposes
    # `rg_flow` as a method that delegates here).
    from bps_kalgebra import BPSKAlgebra

    if A._chart is None:
        raise NotImplementedError(
            "single_node_rg_flow requires spec mode (recipe mode has "
            "no spec to filter)"
        )

    n = len(A.node_charges)
    if not 0 <= node_index < n:
        raise ValueError(
            f"node_index {node_index} out of range for {n} nodes"
        )

    j = node_index
    old_nodes: list[Vec] = [tuple(g) for g in A.node_charges]
    new_nodes: list[Vec] = old_nodes[:j] + old_nodes[j + 1 :]

    # ---- Filter spec ------------------------------------------------------
    new_spec: list[Vec] = []
    for g in A.spec:
        decomp = decompose_nonneg_in_node_basis(old_nodes, g)
        if decomp is None:
            raise ValueError(
                f"RG flow recipe inapplicable: spec charge {tuple(g)} "
                f"has no non-negative integer decomposition in the "
                f"old node-charge basis {old_nodes}.  This theory or "
                f"chamber is outside the recipe's domain."
            )
        if decomp[j] == 0:
            new_spec.append(tuple(g))

    # ---- Build the new BPSKAlgebra ---------------------------------------
    pairing = [list(row) for row in A.lattice.pairing]
    new_A = BPSKAlgebra(
        pairing=pairing,
        node_charges=new_nodes,
        spec=new_spec,
        cone_gens=new_nodes,
        verify="off",
    )

    # ---- Pre-populate F-cache by filtering the old one -------------------
    #
    # For each cached F_a in A, drop summands whose offset `δ - γ_a`
    # decomposes with n_j > 0.  The offset 0 (the leading X_{γ_a}
    # summand) always survives, so the filtered F is non-empty by
    # construction.
    for gamma_a, F_dict in A._F_cache.items():
        gamma_a = tuple(gamma_a)
        filtered: dict[Vec, object] = {}
        for delta, coeff in F_dict.items():
            offset = tuple(d - g for d, g in zip(delta, gamma_a))
            decomp = decompose_nonneg_in_node_basis(old_nodes, offset)
            if decomp is None:
                raise ValueError(
                    f"RG flow recipe inapplicable: F-summand offset "
                    f"{offset} (in F_{gamma_a}) has no non-negative "
                    f"integer decomposition in the old node-charge "
                    f"basis {old_nodes}."
                )
            if decomp[j] == 0:
                filtered[tuple(delta)] = coeff
        new_A._F_cache[gamma_a] = filtered

    return new_A


# ---------------------------------------------------------------------------
# Subquiver RG flow: remove multiple nodes at once
# ---------------------------------------------------------------------------


def subquiver_rg_flow_raw(
    pairing: Sequence[Sequence[int]],
    node_charges: Sequence[Sequence[int]],
    spec: Sequence[Sequence[int]],
    node_indices: Sequence[int],
    *,
    F_cache: dict | None = None,
) -> tuple[list[list[int]], list[Vec], list[Vec], dict | None]:
    """Plain-data variant of `subquiver_rg_flow` operating directly on
    `(pairing, node_charges, spec[, F_cache])` tuples.

    Same recipe and same validation as the wrapper: drop the listed
    nodes, filter `spec` to entries whose old-basis decomposition is
    zero on the dropped indices, optionally filter the F-cache.

    Returns `(pairing_unchanged, new_node_charges, new_spec,
    new_F_cache_or_None)`.

    Avoids constructing a `BPSKAlgebra` (and the `BPSQuiver` it builds
    internally) on either input or output -- the dictionary builder's
    `_drop_node_entry` previously paid both wrappers per attempt, and
    profile showed `BPSKAlgebra.__init__` accounting for ~70% of build
    time.  This raw API skips both.
    """
    n = len(node_charges)
    indices = list(node_indices)
    if any(not 0 <= i < n for i in indices):
        raise ValueError(
            f"node_indices contain out-of-range entry: {indices}, "
            f"valid range [0, {n})"
        )
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"node_indices must be distinct: got {indices}"
        )
    drop_set = set(indices)

    old_nodes: list[Vec] = [tuple(g) for g in node_charges]
    new_nodes: list[Vec] = [
        old_nodes[i] for i in range(n) if i not in drop_set
    ]

    # Bulk-decompose the entire spec (and any F-cache offsets) against
    # the old node basis in a single Gauss-Jordan pass -- amortising
    # the per-target rational solve, the same trick as in
    # `_restandardize`.
    spec_t: list[tuple] = [tuple(g) for g in spec]
    decomps = decompose_nonneg_in_node_basis_bulk(old_nodes, spec_t)
    if decomps is None:
        raise ValueError(
            "subquiver RG flow recipe inapplicable: node basis is "
            "rank-deficient over the spec."
        )
    new_spec: list[Vec] = []
    for g, d in zip(spec_t, decomps):
        if d is None:
            raise ValueError(
                f"subquiver RG flow recipe inapplicable: spec charge "
                f"{g} has no non-negative integer decomposition in the "
                f"old node-charge basis {old_nodes}."
            )
        if all(d[i] == 0 for i in drop_set):
            new_spec.append(g)

    new_F_cache: dict | None = None
    if F_cache:
        new_F_cache = {}
        for gamma_a, F_dict in F_cache.items():
            gamma_a_t = tuple(gamma_a)
            offsets = [
                tuple(d - g for d, g in zip(delta, gamma_a_t))
                for delta in F_dict
            ]
            offset_decomps = decompose_nonneg_in_node_basis_bulk(
                old_nodes, offsets,
            )
            if offset_decomps is None:
                raise ValueError(
                    "subquiver RG flow recipe inapplicable: F-cache "
                    "offsets indecomposable in old node-charge basis."
                )
            filtered: dict = {}
            for (delta, coeff), d in zip(F_dict.items(), offset_decomps):
                if d is None:
                    raise ValueError(
                        f"subquiver RG flow recipe inapplicable: "
                        f"F-summand offset {tuple(de - g for de, g in zip(delta, gamma_a_t))} "
                        f"(in F_{gamma_a_t}) has no non-negative integer "
                        f"decomposition in the old node-charge basis "
                        f"{old_nodes}."
                    )
                if all(d[i] == 0 for i in drop_set):
                    filtered[tuple(delta)] = coeff
            new_F_cache[gamma_a_t] = filtered

    pairing_out = [list(row) for row in pairing]
    return pairing_out, new_nodes, new_spec, new_F_cache


def subquiver_rg_flow(
    A: "BPSKAlgebra",
    node_indices: Sequence[int],
) -> "BPSKAlgebra":
    """Drop a *set* of BPS-quiver nodes at once and return the
    resulting `BPSKAlgebra`.

    Mathematically equivalent to composing single-node RG flows in
    any order (order is immaterial per the recipe), but built directly
    in one pass: same Γ, same B, node_charges loses the listed
    entries, spec is filtered to keep only entries whose decomposition
    in the old node-charge basis has `n_i = 0` for every `i` in
    `node_indices`, and cached F's are filtered analogously on
    `δ - γ_a`.

    Wrapper around `subquiver_rg_flow_raw` for callers operating at
    the `BPSKAlgebra` level.  Hot-path callers that work in
    `(pairing, nodes, spec)` form directly (notably the dictionary
    builder's `_drop_node_entry`) should call the raw variant to
    skip the `BPSKAlgebra` construction overhead -- see
    `subquiver_rg_flow_raw` for details.

    Parameters
    ----------
    A
        Source algebra in spec mode.
    node_indices
        Iterable of distinct indices into `A.node_charges`.  An empty
        iterable returns an algebra equivalent to `A` (no flow).

    Returns
    -------
    BPSKAlgebra
        New algebra with the listed nodes removed.

    Raises
    ------
    NotImplementedError
        If `A` is in recipe mode.
    ValueError
        On out-of-range or duplicate indices, or if some spec charge
        or F-summand offset has no non-negative integer decomposition
        in the old node-charge basis.
    """
    from bps_kalgebra import BPSKAlgebra

    if A._chart is None:
        raise NotImplementedError(
            "subquiver_rg_flow requires spec mode (recipe mode has no "
            "spec to filter)"
        )
    pairing_out, new_nodes, new_spec, new_F_cache = subquiver_rg_flow_raw(
        A.lattice.pairing, A.node_charges, A.spec, node_indices,
        F_cache=A._F_cache,
    )
    new_A = BPSKAlgebra(
        pairing=pairing_out,
        node_charges=new_nodes,
        spec=new_spec,
        cone_gens=new_nodes if new_nodes else None,
        verify="off",
    )
    if new_F_cache is not None:
        for gamma_a, filtered in new_F_cache.items():
            new_A._F_cache[gamma_a] = filtered
    return new_A


# ---------------------------------------------------------------------------
# Cone-minimal helper (used by the QT → IR-F-basis decomposition)
# ---------------------------------------------------------------------------


def _cone_minimal(
    charges: Sequence[Vec],
    cone_gens: Sequence[Sequence[int]],
) -> Vec | None:
    """Return a charge in `charges` that is minimal w.r.t. the partial
    order induced by the non-negative integer span of `cone_gens`.

    `δ ≤ δ'`  iff  `δ' - δ` decomposes non-negatively in `cone_gens`.
    "Minimal" = no other element of `charges` is strictly less than it.

    Returns `None` only on the empty input.
    """
    charges = list(charges)
    if not charges:
        return None
    cone_gens = [list(g) for g in cone_gens]
    for delta in charges:
        is_min = True
        for delta_prime in charges:
            if delta == delta_prime:
                continue
            diff = tuple(d - dp for d, dp in zip(delta, delta_prime))
            decomp = decompose_nonneg_in_node_basis(cone_gens, diff)
            if decomp is not None and any(x > 0 for x in decomp):
                is_min = False
                break
        if is_min:
            return tuple(delta)
    # No strict minimum (rare; means a cycle in the partial order, which
    # shouldn't happen for a pointed cone).  Return any element.
    return tuple(charges[0])


# ---------------------------------------------------------------------------
# Decompose a QT element into IR canonical basis
# ---------------------------------------------------------------------------


def _decompose_qt_in_ir_basis(
    qt: dict,
    ir_algebra: "BPSKAlgebra",
):
    """Greedy peel-off decomposition of a QT element into the IR
    canonical basis `{F_b}`.

    `qt` is a `dict[Vec, Coeff]` with `Coeff` supporting `+`, `-`,
    `is_zero()`, and multiplication by `LaurentPoly` on the right
    (which is the coefficient type of `ir_algebra.F(b)`).  Both
    `LaurentPoly` and `HabiroElement` qualify, so the same routine
    serves `apply` (LaurentPoly inputs) and `rg_generator` (Habiro
    inputs).

    Algorithm: while `qt` is non-zero, find a cone-minimal charge `δ`
    in its support; the leading coefficient there is the
    `F_δ`-component of the decomposition (since `F_δ` starts with
    `X_δ` with coefficient 1 in `LaurentPoly`).  Subtract
    `leading_coeff · F_δ` from `qt` and continue.

    Returns `dict[Vec, Coeff]` mapping IR labels to coefficients.
    """
    qt = {k: v for k, v in qt.items() if not v.is_zero()}
    cone_gens = ir_algebra.node_charges
    result: dict[Vec, object] = {}
    while qt:
        delta = _cone_minimal(list(qt.keys()), cone_gens)
        leading = qt[delta]
        F_delta = ir_algebra.F(delta)  # dict[Vec, LaurentPoly]
        # Accumulate.
        result[delta] = (
            leading if delta not in result else result[delta] + leading
        )
        # Subtract  leading · F_delta  from qt.
        for d, c_lp in F_delta.items():
            term = leading * c_lp
            if d in qt:
                qt[d] = qt[d] - term
            else:
                qt[d] = -term
            if qt[d].is_zero():
                del qt[d]
    return result


# ---------------------------------------------------------------------------
# RGKAlgebra concrete subclasses: SubquiverRG (general) and SingleNodeRG
# (thin |S|=1 wrapper).
# ---------------------------------------------------------------------------


def _laurentpoly_to_rlaurent(lp, R):
    """Convert a `quantum_torus.LaurentPoly` to an `RLaurent` over `R`.

    Each integer q-coefficient is lifted to `R.basis_element(one)`
    (handled automatically by `RLaurent`'s constructor when given an
    `int`).
    """
    return RLaurent(R, dict(lp._coeffs))


# ---------------------------------------------------------------------------
# SubquiverRG / SingleNodeRG: directional node-drop flows from a UV instance
# ---------------------------------------------------------------------------

from directional_subquiver_rg import (   # noqa: E402  (placed by its classes)
    DirectionalSubquiverRG,
    uv_f_oracle,
)


class SubquiverRG(DirectionalSubquiverRG):
    """`RGKAlgebra` for the deletion of a subset of BPS-quiver nodes at
    once, constructed from an existing UV `BPSKAlgebra`.

    A thin constructor over
    `DirectionalSubquiverRG`: the whole KAlgebra API is *derived* from
    IR + RG data (apex-peel multiply, tRG-mirror ρ, transported trace),
    rather than delegating `multiply`/`rho`/`trace` to the UV instance
    (which would make any cross-check against the UV circular).
    The UV instance is retained only as

      * `starting_algebra()` — composition endpoints (`ComposedRG`
        validates `first.auxiliary() is second.starting_algebra()`),
      * the default **F-oracle** — `RG(a)` = the UV `F(a)` peeled into
        the IR canonical basis,
        cross-checked against the generic exact solve whenever the
        ρ⁻¹ mirror runs.

    `rg_generator(cutoff)` uses the window semantics
    (**total dropped-node multiplicity** `Σ_{i∈S} n_i(δ) ≤ cutoff`),
    with values given by the closed-form ket-peel (validated
    term-by-term against direct UV-state extraction).  `grading()` is **total** (off-cone
    labels score negative charges instead of raising — the generic
    machinery needs to score ρ-mirror images; off-cone-ness is
    `grading().in_cone`).  Any spec ordering is accepted, as before
    (`require_stuff_first=False` — the version-(c) ket factorization)."""

    def __init__(
        self,
        A_uv: "BPSKAlgebra",
        node_indices: Sequence[int],
    ):
        from bps_kalgebra import BPSKAlgebra
        if not isinstance(A_uv, BPSKAlgebra):
            raise TypeError("SubquiverRG: UV algebra must be a BPSKAlgebra")
        if A_uv._chart is None:
            raise NotImplementedError(
                "SubquiverRG: UV algebra must be in spec mode "
                "(recipe mode not supported)"
            )
        n = len(A_uv.node_charges)
        idxs = list(node_indices)
        if any(not 0 <= i < n for i in idxs):
            raise ValueError(
                f"node_indices contain out-of-range entry: {idxs}, "
                f"valid range [0, {n})"
            )
        if len(set(idxs)) != len(idxs):
            raise ValueError(
                f"node_indices must be distinct: got {idxs}"
            )
        super().__init__(
            [list(row) for row in A_uv.lattice.pairing],
            [tuple(g) for g in A_uv.node_charges],
            [tuple(g) for g in A_uv.spec],
            sorted(set(idxs)),
            require_stuff_first=False,
        )
        self._uv_ref = A_uv
        self._f_oracle = uv_f_oracle(A_uv, self.auxiliary())

    # ----- sharp closed-form ρ (contract-sanctioned override) -------------
    # The UV chart's half-monodromy σ IS this flow's ρ_UV (the directional
    # tRG-derived ρ re-derives the same map and is certified equal on the
    # pentagon/A₄ batteries; it remains the default on the pure
    # `DirectionalSubquiverRG`).  The compat classes delegate to the UV's
    # closed form — "subclasses with a combinatorial σ may override".

    def rho(self, a):
        return self._uv_ref.rho(a)

    def rho_inverse(self, a):
        return self._uv_ref.rho_inverse(a)

    # ----- UV/IR introspection surface ------------------------------------

    @property
    def uv_algebra(self):
        return self._uv_ref

    @property
    def ir_algebra(self):
        return self.auxiliary()

    @property
    def node_indices(self) -> tuple[int, ...]:
        return self.drop_indices

    def tropical(self, uv_label):
        return tuple(uv_label)


class SingleNodeRG(SubquiverRG):
    """`RGKAlgebra` for the deletion of a single BPS-quiver node — the
    `|S| = 1` case of `SubquiverRG` (kept for the naming + the `int`
    argument); see `SubquiverRG`.

    Exposes `_j` and `_gamma_j` for callers that address the dropped
    node directly."""

    def __init__(self, A_uv: "BPSKAlgebra", node_index: int):
        super().__init__(A_uv, [node_index])
        self._gamma_j = tuple(A_uv.node_charges[node_index])

    @property
    def _j(self) -> int:
        return self.drop_indices[0]


# ---------------------------------------------------------------------------
# Subquiver convenience: factor a BPS UV->IR flow through a subquiver's
# sub-flow (the driving identity: total spec = (outside)(subquiver spec)).
# ---------------------------------------------------------------------------


def factor_through_subquiver(A: "BPSKAlgebra", subquiver_node_indices):
    """Recover the UV->MS flow of a BPS theory by factoring its UV->IR flow
    through the sub-flow generated by a **subquiver**.

    Requires the total spec to factor as

        spec  =  (factors with >=1 outside-subquiver charge) . (subquiver spec)

    i.e. the spec entries supported purely on subquiver-node charges form a
    contiguous **right tail** (the user's driving class).  Then `MS->IR` is
    the subquiver theory flowing to the shared quantum torus,
    `mid = BPSKAlgebra(A.pairing, subquiver nodes, subquiver spec)`, and the
    returned `A.factor_through(mid)` is the `UV->MS` flow: its `RG` is the
    embedding `A^UV` into `A^MS` (the subquiver theory), `rg_generator` is
    `S^UV_MS`, and `grading` is the quotient `Gamma^UV_MS = Gamma / Gamma^MS_IR`.

    Raises `ValueError` if the spec is not of this form.
    """
    from bps_kalgebra import BPSKAlgebra
    if A._chart is None:
        raise NotImplementedError(
            "factor_through_subquiver requires spec mode (recipe mode has no "
            "spec to factor)"
        )
    n = len(A.node_charges)
    sub = sorted(set(int(i) for i in subquiver_node_indices))
    if any(not 0 <= i < n for i in sub):
        raise ValueError(f"subquiver indices out of range [0,{n}): {sub}")
    if not sub:
        raise ValueError("empty subquiver")
    outside = [i for i in range(n) if i not in set(sub)]
    old_nodes = [tuple(g) for g in A.node_charges]

    # Classify each spec entry: subquiver-only iff zero coeff on every
    # outside node in the node-charge basis.
    is_sub: list[bool] = []
    for g in A.spec:
        d = decompose_nonneg_in_node_basis(old_nodes, tuple(g))
        if d is None:
            raise ValueError(
                f"factor_through_subquiver: spec entry {tuple(g)} has no "
                f"non-negative decomposition in the node basis {old_nodes}."
            )
        is_sub.append(all(d[i] == 0 for i in outside))

    # Compatibility: the subquiver-only entries are a contiguous right tail.
    first_sub = next((k for k, s in enumerate(is_sub) if s), len(is_sub))
    if not all(is_sub[k] for k in range(first_sub, len(is_sub))):
        raise ValueError(
            "factor_through_subquiver: spec does not factor as "
            "(outside)(subquiver spec) -- the subquiver-only entries are not "
            "a contiguous right tail (S^UV_IR not 'of the correct form')."
        )
    subquiver_spec = [tuple(A.spec[k]) for k in range(first_sub, len(A.spec))]
    sub_nodes = [old_nodes[i] for i in sub]

    mid = BPSKAlgebra(
        pairing=[list(r) for r in A.lattice.pairing],
        node_charges=sub_nodes,
        spec=subquiver_spec,
        cone_gens=sub_nodes,
        verify="off",
    )

    # The *direct* UV→MS flow: drop the outside nodes in one shot.  Its IR is
    # the subquiver theory (same pairing, same subquiver node_charges, same
    # subquiver spec as `mid`), and its `rg_generator` computes `S^UV_MS` —
    # the outside factors — enumerated by **dropped-node multiplicity** with
    # exact HabiroElement coefficients.  That is the right truncation measure for the
    # quotient grading `Γ^UV_MS` (outside bound states `γ_a+γ_b` are kept even
    # when their q-order exceeds the cutoff), so we hand it to `factor_through`
    # as the authoritative UV→MS flow (`uv_ms_flow`) rather than re-deriving
    # `S^UV_MS` by the general pullback (which would truncate by q-order and
    # drop those bound states).
    direct = SubquiverRG(A, outside)

    # gamma_inclusion = the Gamma-coordinate each subquiver node occupies
    # (standard-basis nodes; the deg=id grading's Gamma^MS_IR coords).  For
    # non-standard-basis nodes no general SNF inclusion is implemented; pass
    # None so the core RG/rg_generator still work (grading() then defers).
    gamma_inclusion: list[int] | None = []
    for g in sub_nodes:
        nz = [k for k, x in enumerate(g) if x != 0]
        if len(nz) == 1 and g[nz[0]] == 1:
            gamma_inclusion.append(nz[0])
        else:
            gamma_inclusion = None
            break

    return A.factor_through(
        mid, gamma_inclusion=gamma_inclusion, uv_ms_flow=direct,
    )
