"""The quantum trace map Tr^lq : S(S^2_{0,n}) -> Y_Delta.

Composition:
        rho_Delta              (X) phi
    S(S) ---------> (X)_T S^s(T) -------->  (X)_T Y(T)  -------> Y_Delta
                                                 (project)

For a simple closed curve with tropical charge `a`, this module
implements the explicit state-sum form of the composition:

    Tr^lq(gamma_a) = sum_{s : crossed_edges -> {+/-}}
                       projection( otimes_T  product_{arc in T}  phi(arc, s) ).

Annihilation shortcut: any state with phi(arc, eps, eps') = 0 (the
"(-, +)" pattern) contributes nothing and is skipped.

Scope (asserted via curve_realization): single simple closed curve,
single-arc-per-triangle, n >= 3.

Self-contained: depends only on
    half_laurent
    triangulation
    triangle_algebra
    y_delta
    phi_map
    curve_realization.
"""

from __future__ import annotations

from typing import Sequence

from curve_realization import realize_curve
from half_laurent import HalfLaurent, ONE as HL_ONE
from phi_map import phi
from triangle_algebra import TriangleAlgebra
from triangulation import Triangulation
from y_delta import YDelta, project_triangle_monomial


def quantum_trace(
    triangulation: Triangulation,
    charge: Sequence[int],
) -> YDelta:
    """Compute Tr^lq(gamma_a) in Y_Delta as a HalfLaurent-coefficient sum
    of canonical-ordered monomials.

    The state-sum:
        * iterate s in {+1, -1}^k where k = #crossed-edges;
        * for each triangle, evaluate phi on its arcs, multiply in
          elevation order (trivially length 1 in our single-arc scope);
        * tensor across triangles;
        * project to Y_Delta via lift-parity checking;
        * accumulate.

    Returns a YDelta element.  Use `.coefficient(...)` for ordered-form
    coefficients or `.weyl_coefficient(...)` to compare against the
    Weyl-normalized monomials.
    """
    arcs_per_triangle = realize_curve(triangulation, charge)

    # Identify the global state index space: each crossed edge has |a_e|
    # intersection points (= 1 in our scope), and each receives one
    # +/- state.  Different arcs in different triangles share the same
    # state at a given edge.
    edges_with_state: list[int] = []
    edge_to_state_idx: dict[int, int] = {}
    for tlist in arcs_per_triangle:
        for arc in tlist:
            for e in (arc.edge_at_eps, arc.edge_at_eps_prime):
                if e not in edge_to_state_idx:
                    edge_to_state_idx[e] = len(edges_with_state)
                    edges_with_state.append(e)

    n_states = len(edges_with_state)

    result = YDelta.zero(triangulation)

    for state_int in range(1 << n_states):
        # Decode state: the i-th bit is +1 if set, -1 if unset.
        state_at_edge = {
            edges_with_state[i]: (1 if (state_int >> i) & 1 else -1)
            for i in range(n_states)
        }

        # Per-triangle contribution.  Each triangle contributes a single
        # ordered monomial in Y(T) with HalfLaurent coefficient.
        triangle_exps: list[tuple[int, int, int]] = []
        global_coef: HalfLaurent = HL_ONE
        annihilated = False

        for tlist in arcs_per_triangle:
            triangle_factor = TriangleAlgebra.one()
            for arc in tlist:
                eps = state_at_edge[arc.edge_at_eps]
                eps_prime = state_at_edge[arc.edge_at_eps_prime]
                f = phi(arc.arc_type, eps, eps_prime)
                if f.is_zero():
                    annihilated = True
                    break
                triangle_factor = triangle_factor * f
            if annihilated:
                break

            # In the single-arc-per-triangle scope, triangle_factor is
            # always a single ordered monomial (or the identity for
            # arc-free triangles).
            terms = triangle_factor.items()
            if len(terms) != 1:
                raise NotImplementedError(
                    "multi-arc triangles produce polynomials in Y(T); "
                    "out of scope for this build (Section 9.4 caveat)"
                )
            (exp_triple, c_local), = terms
            triangle_exps.append(exp_triple)
            global_coef = global_coef * c_local

        if annihilated:
            continue

        proj = project_triangle_monomial(triangulation, triangle_exps, global_coef)
        if proj is None:
            # lift-parity mismatch -> contribution is zero
            continue
        charge_vec, ordered_coef = proj

        result = result + YDelta.monomial(triangulation, charge_vec, ordered_coef)

    return result


def n_states(triangulation: Triangulation, charge: Sequence[int]) -> int:
    """Number of states the state-sum iterates over (= 2^k, k = #crossed-edges).

    Useful for sanity-checking annihilation counts.
    """
    return 1 << sum(1 for x in charge if x != 0)
