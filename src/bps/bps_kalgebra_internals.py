"""Parameterized internals for `BPSKAlgebra`: F-solving and Schur-index
computation parameterized by a single callable `s_coefficient(γ) -> HabiroElement`.

This is the seam that lets `BPSKAlgebra` accept either a finite spec
(in which case `s_coefficient` is derived by expanding the product
`E_q(X_{γ_1}) ⋯ E_q(X_{γ_N})` and reading off `[S|0⟩]_γ`) or a
user-provided recipe `s_coefficient_fn(γ) -> HabiroElement` directly
(for theories like SU(2) N=2* with no finite-`E_q` factorization but
a closed form on the lattice).

The math is a transcription of `f_solver.solve_F` and
`nahm_data.{schur_index_nahm, c_gamma_habiro}` with their
`(spec_t, kmat)` arguments replaced by the more general
`s_coefficient_fn`. Identical results in spec mode; supports recipe
mode as a free byproduct.
"""

from __future__ import annotations

from itertools import product as _iproduct
from typing import Callable, Sequence

from laurent_poly import LaurentPoly
from lattice import Lattice, LatticeTorus, Vec
from lattice import cone_contains, make_cone_predicate
from habiro import HabiroElement
from q_number_poly import QNumberPoly
from qpoch import PowerSeries, qpoch_infty
from nahm_local import (
    s_gamma_habiro,
    s_table_in_cone,
    fs_dict_from_s_table,
)


# ---------------------------------------------------------------------------
# F-solver
# ---------------------------------------------------------------------------


def _shifted_s(s_el: HabiroElement, twist: int) -> HabiroElement:
    """Return ``q^twist * s_el`` as a HabiroElement (no simplify needed).

    Multiplication by a monomial cannot introduce a new ``(1-q^{2k})``
    factor in the numerator, so the canonical-form invariant is preserved.
    """
    if twist == 0:
        return s_el
    return HabiroElement(
        s_el.numerator * LaurentPoly({twist: 1}), dict(s_el.denom),
    )


def _propagate_lp(
    lattice: Lattice,
    delta: Vec,
    f_lp_coeffs: dict[int, int],
    forward_targets: list[tuple[Vec, Vec]],
    s_lookup: Callable[[Vec], HabiroElement | None],
    contribs_by_eta: dict[Vec, list[HabiroElement]],
    *,
    skip_positive_q: bool = False,
) -> None:
    """Push ``f_lp · q^{<delta,eta-delta>} · [S]_{eta-delta}`` into the
    deferred-sum buffers for every ``eta`` reachable from ``delta``.

    Used by both the initial-guess seeding pass and the per-delta
    peel correction pass.  Monomial fast path for the common
    pentagon/SU(2)/SU(3) case where ``[S]_·.numerator`` is a single
    monomial (folds all shifts and the sign into one dict comprehension).

    Skip-positive-q optimisation.  When ``skip_positive_q=True``,
    contribs whose predicted Laurent-expansion leading exponent
    ``k_min(f_lp) + twist + k_min([S]_·)`` is ``> 0`` are *not*
    built --- such contribs can only feed the ``q^{>0}`` part of
    ``FS_resid[eta]``, which the peel never touches.  Sound only if
    the caller does not consume the full ``FS_resid`` (e.g. recipe-
    mode ``solve_F_via_s_coefficient`` returns only the F dict;
    spec-mode ``solve_F_with_fs_table`` consumes ``FS_resid`` as a
    Schur byproduct and must pass ``skip_positive_q=False``).
    """
    if not f_lp_coeffs:
        return
    f_lead = min(f_lp_coeffs) if skip_positive_q else None
    for eta, diff in forward_targets:
        s_el = s_lookup(diff)
        if s_el is None or s_el.is_zero():
            continue
        twist = lattice.bracket(delta, diff)
        s_num = s_el.numerator._coeffs
        if skip_positive_q and s_num:
            # Predict leading q-exp of this contribution; skip if > 0.
            predicted = f_lead + twist + min(s_num)
            if predicted > 0:
                continue
        if len(s_num) == 1:
            s_exp, s_coef = next(iter(s_num.items()))
            total_shift = twist + s_exp
            if s_coef == 1:
                if total_shift == 0:
                    new_num = LaurentPoly(dict(f_lp_coeffs))
                else:
                    new_num = LaurentPoly(
                        {e + total_shift: c for e, c in f_lp_coeffs.items()}
                    )
            elif s_coef == -1:
                new_num = LaurentPoly(
                    {e + total_shift: -c for e, c in f_lp_coeffs.items()}
                )
            else:
                new_num = LaurentPoly(
                    {e + total_shift: c * s_coef for e, c in f_lp_coeffs.items()}
                )
        else:
            f_lp = LaurentPoly(f_lp_coeffs)
            shifted = (
                f_lp * LaurentPoly({twist: 1}) if twist else f_lp
            )
            new_num = shifted * s_el.numerator
        contribs_by_eta[eta].append(
            HabiroElement(new_num, dict(s_el.denom))
        )


def _solve_F_by_peeling(
    lattice: Lattice,
    cone_t: list[Vec],
    gamma_t: Vec,
    ordered: list[Vec],
    s_lookup: Callable[[Vec], HabiroElement | None],
    *,
    initial_guess: dict[Vec, QNumberPoly] | None = None,
    skip_positive_q: bool = False,
) -> tuple[dict[Vec, QNumberPoly], dict[Vec, HabiroElement], int]:
    """Solve ``F·S|0> = X_γ|0> + O(q)``, returning ``f_δ`` as ``QNumberPoly``.

    Algorithm.  Maintain a deferred-sum buffer ``contribs_by_eta[η]``
    of ``HabiroElement`` contributions to ``[F·S|0>]_η``.  Seed it
    from ``F_qn`` (either the trivial ``{γ_t: 1}`` or an
    ``initial_guess``).  Then walk ``ordered`` (BFS support); at each
    ``δ`` materialise ``FS_resid[δ]``, expand the non-positive part,
    and peel ``[n]_q`` corrections.  Propagate any correction
    forward.

    Initial-guess hook.  ``initial_guess`` is an optional
    ``{δ: f_δ}`` candidate.  When supplied, the peel at each ``δ``
    produces only the *correction* to add to the guess's ``f_δ``;
    a guess equal to the true ``F_γ`` produces zero corrections
    (the peel loop exits immediately at every ``δ``).  Useful as
    the entry point for product-of-known-F's warm starts or ML
    F-prediction (the predictor outputs a candidate ``F_γ`` and
    this routine certifies / corrects it).

    Returns ``(F_qn, FS_resid, n_corrections)`` where
    ``n_corrections`` is the total number of ``[n]_q`` peels
    performed (0 iff the guess was exact for every ``δ``).
    """
    n_charges = len(ordered)
    in_cone = make_cone_predicate(cone_t)
    support_set = set(ordered)

    # Precompute forward[delta] = [(eta, eta-delta) for eta reachable
    # from delta via the cone partial order].  Used by the seeding pass
    # and the per-delta correction pass; the inner loop has zero
    # cone-membership checks.
    forward: dict[Vec, list[tuple[Vec, Vec]]] = {}
    for i, d in enumerate(ordered):
        targets: list[tuple[Vec, Vec]] = []
        for j in range(i + 1, n_charges):
            e = ordered[j]
            diff = tuple(ek - dk for ek, dk in zip(e, d))
            if in_cone(diff):
                targets.append((e, diff))
        forward[d] = targets

    # Seed F_qn from the guess (if any) or the canonical trivial start.
    F_qn: dict[Vec, QNumberPoly] = {}
    if initial_guess:
        for delta_g, qn_g in initial_guess.items():
            if qn_g is None or qn_g.is_zero():
                continue
            F_qn[delta_g] = qn_g
    if gamma_t not in F_qn or F_qn[gamma_t].is_zero():
        F_qn[gamma_t] = QNumberPoly.one()

    # Seed contribs_by_eta by propagating every known f_{δ_seed}.  For
    # the trivial start this is just f_{γ_t}=1.  With a guess we
    # propagate every guess coefficient whose δ sits inside the BFS
    # support (guess entries outside the support are silently ignored
    # --- the canonical basis would not assign coefficients there).
    contribs_by_eta: dict[Vec, list[HabiroElement]] = {eta: [] for eta in ordered}
    for delta_seed, qn_seed in F_qn.items():
        if delta_seed not in support_set:
            continue
        _propagate_lp(
            lattice, delta_seed, qn_seed.to_laurent()._coeffs,
            forward[delta_seed], s_lookup, contribs_by_eta,
            skip_positive_q=skip_positive_q,
        )
        # Self-contribution at η = δ_seed: [S]_0 = 1, twist = 0, so the
        # contribution is just qn_seed as a polynomial.
        contribs_by_eta[delta_seed].append(
            HabiroElement(qn_seed.to_laurent(), {})
        )

    FS_resid: dict[Vec, HabiroElement] = {}
    n_corrections = 0

    for delta in ordered:
        contribs = contribs_by_eta[delta]

        # Cheap leading-q precheck: each contrib ``n_i/D_i`` in
        # ``contribs_by_eta[delta]`` has Laurent-expansion leading
        # exponent equal to ``k_min(n_i)`` (since every ``D_i`` has
        # constant term 1).  If every contrib individually starts at
        # ``q^{>0}``, their sum also starts at ``q^{>0}``: no peel
        # corrections possible at delta, so we can skip the
        # ``expand(0)`` + peel-loop work.  Cancellation can only
        # raise the leading exponent further, never lower it, so
        # the precheck is sound (no false skips).  Pays off at
        # "dead" deltas in the tropical support (where F's actual
        # support is empty but the BFS interval still includes
        # them) and at every delta when an exact guess has already
        # cancelled all non-positive contributions.
        all_positive = True
        for h in contribs:
            n_coeffs = h.numerator._coeffs
            if not n_coeffs:
                continue
            if min(n_coeffs) <= 0:
                all_positive = False
                break

        FS_now = (
            HabiroElement.sum(contribs, simplify=False) if contribs
            else HabiroElement.zero()
        )
        FS_resid[delta] = FS_now

        if all_positive or delta == gamma_t:
            # Precheck guarantees positive q-order ⇒ skip the
            # expand(0)+peel work.  (At gamma_t we never peel by
            # definition: f_{gamma_t} is fixed to 1 by the canonical
            # basis; any non-positive residual there would indicate
            # a contradictory guess, which we silently ignore.)
            continue

        # Peel: expand the non-positive part of FS_now, then peel [n]_q
        # corrections.  For the guess to be exact at delta, FS_now must
        # already have positive q-order (no negative leading terms).
        f_correction: dict[int, int] = {}
        if not FS_now.is_zero():
            nonpos = {
                e: c for e, c in FS_now.expand(0)._coeffs.items() if e <= 0
            }
        else:
            nonpos = {}
        while nonpos:
            k = min(nonpos)
            c = nonpos.pop(k)
            if c == 0:
                continue
            n = 1 - k
            f_correction[n] = f_correction.get(n, 0) - c
            if f_correction[n] == 0:
                del f_correction[n]
            n_corrections += 1
            for i in range(1, n):
                exp = k + 2 * i
                if exp > 0:
                    break
                nonpos[exp] = nonpos.get(exp, 0) - c
                if nonpos[exp] == 0:
                    del nonpos[exp]

        if not f_correction:
            continue

        # Apply correction to F_qn[delta] (additive on the [n]_q basis).
        cur = dict(F_qn.get(delta, QNumberPoly())._coeffs)
        for n, c in f_correction.items():
            cur[n] = cur.get(n, 0) + c
            if cur[n] == 0:
                del cur[n]
        if cur:
            F_qn[delta] = QNumberPoly(cur)
        else:
            F_qn.pop(delta, None)

        # Expand the correction [n]_q-dict directly to a LaurentPoly
        # coeff dict, bypassing the QNumberPoly construction +
        # to_laurent round-trip (saves ~10% on deep F-solves where
        # f_correction has many [n]_q's and gets converted twice).
        corr_lp_coeffs: dict[int, int] = {}
        for nq, cn in f_correction.items():
            e = -(nq - 1)
            while e <= nq - 1:
                corr_lp_coeffs[e] = corr_lp_coeffs.get(e, 0) + cn
                e += 2
        # Propagate only the CORRECTION to forward eta (the guess
        # contribution was propagated up front in the seeding pass).
        _propagate_lp(
            lattice, delta, corr_lp_coeffs,
            forward[delta], s_lookup, contribs_by_eta,
            skip_positive_q=skip_positive_q,
        )
        # Self-contribution at delta is irrelevant downstream (delta is
        # done), but for consistency of FS_resid as the Schur byproduct
        # we update it.
        FS_resid[delta] = FS_now + HabiroElement(
            LaurentPoly(corr_lp_coeffs), {},
        )

    return F_qn, FS_resid, n_corrections


def _support_bfs_order(
    lower: Vec, upper: Vec, cone_gens: list[Vec],
    max_degree: int | None = None,
    degree_fn: "Callable[[Vec], int] | None" = None,
) -> list[Vec]:
    """Doubly-tropical interval [lower, upper] in BFS cone-order from lower.

    When ``max_degree`` and ``degree_fn`` are supplied, the interval is
    additionally intersected with the **cone-degree simplex**
    ``{δ : degree_fn(δ − lower) ≤ max_degree}`` — i.e. only charges within
    total cone-degree ``max_degree`` of ``lower`` are enumerated.  The box
    ``[lower, upper]`` grows as ``∏(per-generator extent)``; the simplex it
    contains grows as ``max_degree^g / g!`` — much smaller in higher rank, and
    *exactly* the cone-truncated support the spec-free F-solver wants (matter
    dressing costs cone-degree, so a flavour-unbounded box over-explores).  Spec
    mode passes neither and keeps the tight σ⁻¹ box unchanged.
    """
    in_cone = make_cone_predicate(cone_gens)
    if not in_cone(tuple(u - l for u, l in zip(upper, lower))):
        return [lower] if lower == upper else []
    capped = max_degree is not None and degree_fn is not None
    visited: dict[Vec, int] = {lower: 0}
    frontier = [lower]
    order = [lower]
    layer = 0
    while frontier:
        new_frontier: list[Vec] = []
        layer += 1
        for delta in frontier:
            for g in cone_gens:
                nxt = tuple(d + gi for d, gi in zip(delta, g))
                if nxt in visited:
                    continue
                diff = tuple(u - n for u, n in zip(upper, nxt))
                if not in_cone(diff):
                    continue
                if capped and degree_fn(tuple(n - l for n, l in zip(nxt, lower))) > max_degree:
                    continue
                visited[nxt] = layer
                new_frontier.append(nxt)
                order.append(nxt)
        frontier = new_frontier
    return order


def _make_palindromic(neg_part: dict[int, int]) -> dict[int, int]:
    """Given non-positive part of a palindromic poly, produce the full poly.

    For each q^{-a} term with a > 0, adds the mirror q^{+a} term.
    """
    result = dict(neg_part)
    for exp, coeff in neg_part.items():
        if exp < 0:
            result[-exp] = result.get(-exp, 0) + coeff
    return {e: c for e, c in result.items() if c != 0}


def solve_F_via_s_coefficient(
    lattice: Lattice,
    cone_gens: list[Vec],
    gamma: Vec,
    s_coefficient_fn: Callable[[Vec], HabiroElement],
    sigma_inverse_fn: Callable[[Vec], Vec],
    max_degree: int | None = None,
    degree_fn: "Callable[[Vec], int] | None" = None,
) -> dict[Vec, QNumberPoly]:
    """Canonical-basis element `F_γ` as `dict[Vec, QNumberPoly]`.

    Parameterized version of the F-solver: the spec-dependent
    `s_gamma_habiro(γ, spec_t, kmat)` and
    `sigma_inverse(lattice, spec_t, γ)` calls are replaced by user-
    supplied callables, so this works for both spec-defined and
    recipe-defined `S`.

    Each ``f_delta`` is palindromic by construction and returned as
    a :class:`QNumberPoly` -- the natural ring of integral
    combinations of ``[n]_q`` quantum integers in which the F-solver
    naturally operates.

    Uses the peeling algorithm in :func:`_solve_F_by_peeling`: rather
    than recompute the full ``[F·S|0>]_delta`` HabiroElement at every
    ``delta`` in the support, it maintains a running residual and
    peels ``[n]_q`` contributions one at a time, exploiting
    :meth:`HabiroElement.times_q_number` to bypass the polynomial
    expansion of ``[n]_q`` whenever ``n`` already appears in the
    Habiro denominator.
    """
    gamma_t: Vec = lattice.check(gamma)
    cone_t = [lattice.check(g) for g in cone_gens]

    sinv = sigma_inverse_fn(gamma_t)
    upper: Vec = tuple(-x for x in sinv)
    ordered = _support_bfs_order(gamma_t, upper, cone_t,
                                 max_degree=max_degree, degree_fn=degree_fn)

    F_qn, _FS_resid, _n_corr = _solve_F_by_peeling(
        lattice, cone_t, gamma_t, ordered, s_coefficient_fn,
        skip_positive_q=True,
    )
    return {d: qn for d, qn in F_qn.items() if not qn.is_zero()}


# ---------------------------------------------------------------------------
# Table-driven F-solver: a single n-driven Nahm-tuple enumeration
# (`s_table_in_cone`) feeds every `[S|0>]_μ` lookup the F-solver needs,
# and the same table assembles `[F·S|0>]_δ` for every δ in the support
# as a free byproduct.
# ---------------------------------------------------------------------------


def solve_F_with_fs_table(
    lattice: Lattice,
    cone_gens: list[Vec],
    gamma: Vec,
    spec_t: list[Vec],
    kmat: list[list[int]],
    sigma_inverse_fn: Callable[[Vec], Vec],
) -> tuple[dict[Vec, QNumberPoly], dict[Vec, HabiroElement], dict[Vec, HabiroElement]]:
    """Canonical-basis element `F_γ` together with `[F·S|0⟩]_·` and `[S|0⟩]_·`.

    Spec-mode F-solver.  Precomputes a single n-driven Nahm-tuple table
    covering the F-support cone in one pass, then runs the peeling
    F-solver (:func:`_solve_F_by_peeling`) against table-lookup
    ``[S|0>]_μ`` values.

    Returns a triple:

    - ``F_dict``  : ``{δ : f_δ}`` as :class:`QNumberPoly`'s.
    - ``FS_dict`` : ``{δ : [F·S|0⟩]_δ}`` for δ in the F-support,
                    falling out of the same peeling pass.  Schur-index
                    callers extend this to a wider η-region via
                    :func:`fs_dict_from_s_table`.
    - ``s_tbl``   : ``{μ : [S|0⟩]_μ}`` covering ``[0, upper − γ]``.
                    Returned so callers avoid recomputing it.
    """
    gamma_t: Vec = lattice.check(gamma)
    cone_t = [lattice.check(g) for g in cone_gens]

    sinv = sigma_inverse_fn(gamma_t)
    upper: Vec = tuple(-x for x in sinv)
    ordered = _support_bfs_order(gamma_t, upper, cone_t)

    # `[S|0>]_μ` is only ever queried at μ = δ − δ' for δ, δ' in
    # `[γ, upper] ∩ (γ + cone)`.  With μ̄ := δ − γ and μ̄' := δ' − γ
    # (both in `[0, upper − γ] ∩ cone`), μ = μ̄ − μ̄'.  Nonzero
    # `[S|0>]_μ` requires μ ∈ cone, so μ ∈ [0, upper − γ] ∩ cone --
    # exactly what `s_table_in_cone(..., upper = upper − γ)` covers.
    span_upper = tuple(u - g for u, g in zip(upper, gamma_t))
    s_tbl = s_table_in_cone(spec_t, kmat, cone_t, span_upper)

    F_qn, FS_resid, _n_corr = _solve_F_by_peeling(
        lattice, cone_t, gamma_t, ordered, s_tbl.get,
    )
    F_dict = {d: qn for d, qn in F_qn.items() if not qn.is_zero()}
    FS_dict = {eta: fs for eta, fs in FS_resid.items() if not fs.is_zero()}
    return F_dict, FS_dict, s_tbl


def solve_F_with_initial_guess(
    lattice: Lattice,
    cone_gens: list[Vec],
    gamma: Vec,
    s_coefficient_fn: Callable[[Vec], HabiroElement],
    sigma_inverse_fn: Callable[[Vec], Vec],
    initial_guess: dict[Vec, QNumberPoly],
) -> tuple[dict[Vec, QNumberPoly], int]:
    """Solve ``F_γ`` starting from a candidate ``initial_guess``.

    Hook for warm-started F-solves: an ML F-predictor, a
    product-of-known-F's reconstruction, or any other source can
    propose a candidate ``F̃_γ`` (as a ``{δ: f_δ}`` dict), and this
    routine certifies / corrects it by running the peeling
    F-solver from the candidate state instead of from
    ``f_{γ_t} = 1, f_{δ>γ_t} = 0``.

    A correct guess yields zero peels and returns the same dict;
    a slightly-wrong guess produces only the corrections needed to
    restore ``F·S = X_γ + O(q)``.  Returns
    ``(F_dict, n_corrections)`` with ``n_corrections == 0`` iff the
    guess was exact at every ``δ``.

    Cost note.  An exact guess saves only ~15-20% wall time: the
    peel-decision and correction-propagation drop out, but the
    per-δ ``HabiroElement.sum`` (common-denominator unification of
    ``F_candidate · S``) still has to run in order to verify the
    guess.  For a real shortcut, the caller needs an external
    certificate (e.g. zero ``σ``-defect for a product reconstruction)
    and a no-verify mode; see ``scripts/bench_product_guess.py``.
    """
    gamma_t: Vec = lattice.check(gamma)
    cone_t = [lattice.check(g) for g in cone_gens]

    sinv = sigma_inverse_fn(gamma_t)
    upper: Vec = tuple(-x for x in sinv)
    ordered = _support_bfs_order(gamma_t, upper, cone_t)

    F_qn, _FS_resid, n_corr = _solve_F_by_peeling(
        lattice, cone_t, gamma_t, ordered, s_coefficient_fn,
        initial_guess=initial_guess, skip_positive_q=True,
    )
    return ({d: qn for d, qn in F_qn.items() if not qn.is_zero()}, n_corr)


# ---------------------------------------------------------------------------
# Modified F-solver: S_1 · F' · S_2 |0⟩ = X_γ |0⟩ + O(q)
# ---------------------------------------------------------------------------
#
# F' = forward(F_γ, S_1) is the canonical-basis element of A_𝖖 viewed
# at position |S_1| in the spec.  Caller supplies the F'-support
# explicitly (typically the doubly-tropical interval `[l_cur, u_cur]`
# at chart-end, computed by `bps_kalgebra._try_solve_F_via_b`).
#
# The defining bracketed condition decomposes as a triple/big Nahm sum:
#
#     [S_1 · F' · S_2]_η  =  Σ_δ f'(δ) · [S_1 · X_δ · S_2]_η
#
# where the inner factor is itself a Nahm sum over Nahm-indices for both
# S_1 and S_2 separately:
#
#     [S_1 · X_δ · S_2]_η  =  Σ_{α + β = η − δ}
#                              [S_1]_α · [S_2]_β
#                              · q^{⟨α, δ⟩ + ⟨α, β⟩ + ⟨δ, β⟩}
#
# with [S_k]_⋅ = `s_gamma_habiro(⋅, S_k_t, kmat_S_k)` (the standard
# Nahm-sum coefficient for the spec S_k as a finite product of E_q's).
#
# Tradeoff (per user): we lose the cached `s_gamma_habiro(γ, full_spec)`
# (which currently caches per-theory across many F's), since the
# bracketed sum uses S_1 and S_2 separately. We gain when multiple F's
# share the same (S_1, S_2) split — the inner s1_xdelta_s2 results
# cache across them.
# ---------------------------------------------------------------------------


def s1_xdelta_s2_coeff(
    delta: Vec,
    eta: Vec,
    lattice: Lattice,
    S1_t: list[Vec],
    S2_t: list[Vec],
    kmat_S1: list[list[int]],
    kmat_S2: list[list[int]],
    cone_gens: list[Vec],
    cone_cutoff: int,
) -> HabiroElement:
    """`[S_1 · X_δ · S_2]_η` as an exact HabiroElement.

    Triple sum over (α, β) with α + β = η − δ, weighted by
    `[S_1]_α · [S_2]_β · q^{⟨α, δ⟩ + ⟨α, β⟩ + ⟨δ, β⟩}`. Each `[S_k]_·`
    is a Nahm-sum HabiroElement; the multiplication keeps everything
    exact in the localized Habiro ring.
    """
    from collections import deque
    target = tuple(e - d for e, d in zip(eta, delta))
    rank = lattice.rank

    # Enumerate α in the positive cone, bounded by L1-norm.
    # β = target - α; only valid if β is also in the positive cone.
    visited: set = set()
    queue: deque = deque([tuple(0 for _ in range(rank))])
    visited.add(queue[0])
    contribs = []
    while queue:
        alpha = queue.popleft()
        if sum(abs(x) for x in alpha) > cone_cutoff:
            continue
        beta = tuple(t - a for t, a in zip(target, alpha))
        if cone_contains(beta, cone_gens):
            s1_alpha = s_gamma_habiro(alpha, S1_t, kmat_S1)
            if not s1_alpha.is_zero():
                s2_beta = s_gamma_habiro(beta, S2_t, kmat_S2)
                if not s2_beta.is_zero():
                    twist = (lattice.bracket(alpha, delta)
                             + lattice.bracket(alpha, beta)
                             + lattice.bracket(delta, beta))
                    scale = LaurentPoly({twist: 1})
                    # Combine into a single HabiroElement.
                    prod_h = s1_alpha * s2_beta
                    if twist != 0:
                        prod_h = HabiroElement(
                            prod_h.numerator * scale, dict(prod_h.denom),
                        )
                    contribs.append(prod_h)
        # Extend BFS in the positive cone.
        for g in cone_gens:
            nxt = tuple(a + gi for a, gi in zip(alpha, g))
            if nxt in visited:
                continue
            if sum(abs(x) for x in nxt) > cone_cutoff:
                continue
            visited.add(nxt)
            queue.append(nxt)
    if not contribs:
        return HabiroElement.zero()
    return HabiroElement.sum(contribs)


def solve_F_modified(
    lattice: Lattice,
    S1: list[Vec],
    S2: list[Vec],
    cone_gens: list[Vec],
    gamma: Vec,
    F_prime_support: list[Vec],
    cone_cutoff: int = 12,
) -> dict[Vec, QNumberPoly]:
    """Solve `S_1 · F' · S_2 |0⟩ = X_γ + O(q)` for F' with predicted support.

    Implementation mirrors `solve_F_via_s_coefficient`, but the
    coefficient `[S_1 F' S_2]_η` is computed via the triple/big Nahm
    sum (see `s1_xdelta_s2_coeff`), and F' is built incrementally over
    the predicted support order (BFS from γ).

    The trivial split (S_1 empty) reduces to standard solve_F on
    spec=S_2.
    """
    if not S1:
        # No bracketing -- just use standard solver via s_gamma_habiro on S_2.
        N2 = len(S2)
        kmat_S2 = [[lattice.bracket(S2[a], S2[b]) for b in range(N2)]
                   for a in range(N2)]

        def s_S2(g):
            return s_gamma_habiro(tuple(g), S2, kmat_S2)
        from spec_sigma import sigma_inverse as _sinv
        return solve_F_via_s_coefficient(
            lattice, cone_gens, gamma, s_S2,
            lambda g: tuple(_sinv(lattice, S2, tuple(g))),
        )

    N1, N2 = len(S1), len(S2)
    kmat_S1 = [[lattice.bracket(S1[a], S1[b]) for b in range(N1)]
               for a in range(N1)]
    kmat_S2 = [[lattice.bracket(S2[a], S2[b]) for b in range(N2)]
               for a in range(N2)]

    # Topological sort of F'-support along the ORIGINAL cone: a term
    # f'(δ) only contributes corrections to charges δ + (cone), so
    # processing support in any linear extension of the cone-poset
    # ensures every dependency is already known when we reach δ.
    #
    # Cheap linear extension: sort by ⟨w, δ - γ⟩ for any w with
    # ⟨w, c⟩ > 0 on every cone generator c. The sum of cone generators
    # is a generic such w when the cone is pointed.
    rank = len(gamma)
    w = [0] * rank
    for c in cone_gens:
        for k in range(rank):
            w[k] += c[k]

    def _depth(delta):
        return sum(w[k] * (delta[k] - gamma[k]) for k in range(rank))

    sorted_support = sorted(F_prime_support, key=_depth)

    # Build F' incrementally
    f_prime: dict[Vec, dict[int, int]] = {gamma: {0: 1}}
    for delta in sorted_support:
        if delta == gamma:
            continue
        # Compute non-positive q-powers of [S_1 F' S_2]_delta from current F'
        contribs = []
        for delta_prime, c_dict in f_prime.items():
            if not c_dict:
                continue
            sx = s1_xdelta_s2_coeff(
                delta_prime, delta, lattice,
                S1, S2, kmat_S1, kmat_S2, cone_gens, cone_cutoff,
            )
            if sx.is_zero():
                continue
            scale = LaurentPoly(c_dict)
            contribs.append(HabiroElement(scale * sx.numerator, dict(sx.denom)))
        if not contribs:
            continue
        total = HabiroElement.sum(contribs)
        if total.is_zero():
            continue
        k_min = total.k_min()
        if k_min is None or k_min > 0:
            continue
        np = total.expand(0)
        non_pos = {e: c for e, c in np._coeffs.items() if e <= 0 and c != 0}
        if not non_pos:
            continue
        f_prime[delta] = _make_palindromic({s: -v for s, v in non_pos.items()})

    return {
        delta: QNumberPoly.from_palindromic_laurent(LaurentPoly(c_dict))
        for delta, c_dict in f_prime.items()
        if c_dict
    }


# ---------------------------------------------------------------------------
# Schur index
# ---------------------------------------------------------------------------


def _f_coeff_to_laurent(coeff):
    """Coerce an F-coefficient (LaurentPoly or QNumberPoly) to LaurentPoly."""
    if isinstance(coeff, QNumberPoly):
        return coeff.to_laurent()
    return coeff


def c_gamma_via_s(
    gamma: Vec,
    F: dict[Vec, LaurentPoly | QNumberPoly] | None,
    s_coefficient_fn: Callable[[Vec], HabiroElement],
    lattice: Lattice,
) -> HabiroElement:
    """`[F·S|0⟩]_γ` as a HabiroElement, parameterized by `s_coefficient_fn`.

    If F is None, treats it as the identity (i.e., returns `[S|0⟩]_γ`).
    Accepts either :class:`LaurentPoly` or :class:`QNumberPoly`
    coefficients in ``F``; the latter (palindromic native form) is
    converted on the fly.
    """
    if F is None:
        return s_coefficient_fn(gamma)
    contribs = []
    for delta, coeff in F.items():
        lp = _f_coeff_to_laurent(coeff)
        if lp.is_zero():
            continue
        eta = tuple(g - d for g, d in zip(gamma, delta))
        s_el = s_coefficient_fn(eta)
        if s_el.is_zero():
            continue
        twist = lattice.bracket(delta, eta)
        scale = lp * LaurentPoly({twist: 1})
        contribs.append(HabiroElement(scale * s_el.numerator, dict(s_el.denom)))
    return HabiroElement.sum(contribs)


def _enumerate_output_charges(
    Fs: list[dict[Vec, LaurentPoly] | None],
    cone_gens: list[Vec],
    rank: int,
    cone_cutoff: int,
    cone_witness: Sequence[int] | None = None,
    K_joint: int | None = None,
) -> set[Vec]:
    """Charges η in the support of `[F·S|0⟩]_η` for at least one F.

    Uses cone-driven BFS (start at every F-delta, walk by cone generators
    while keeping the filter under the cutoff).

    Filter shape:
      * `cone_witness=None` (legacy): filter by L1 norm
        ``sum(abs(γ)) <= cone_cutoff``.  Correct shape only for
        axis-aligned cones (where ``⟨f, γ⟩ = |γ|_1`` for cone-positive γ
        with ``f = (1,…,1)``).  For oblique cones this both over-includes
        (charges outside the cone whose c-data is zero) and may
        under-include (charges shallow in the cone but large in L1) -- a
        known accuracy bug in the legacy default.
      * `cone_witness=f`: filter by the cone-witness L-shell predicate
        ``⟨f, γ⟩ <= cone_cutoff``.  This is the principled cone-shape
        filter: ``f`` is a strict cone-pointedness witness
        (``⟨f, g⟩ >= 1`` for every cone generator ``g``), so the
        predicate is cone-monotone, the BFS terminates, and the η-set
        is naturally cone-shaped.  Same shape as the L-shell used in
        ``fs_dict_for_eta_set`` (the inner Nahm walk) and
        ``_warm_fs_cache_for_schur``.  Recommended.

    `K_joint` (audit A10 fix; needs `cone_witness`): additionally prune
    by the PAIRED-CONTRIBUTION bound.  The Schur path pairs
    ``c_a(γ)·c_b(γ)`` and truncates at ``q^{K_joint}``; each side obeys
    the cone-witness Nahm bound

        lead c_x(γ)  ≥  min_{δ∈F_x} ( lead(F_δ) + ⟨f, γ−δ⟩ )

    (vacuum side: ``≥ ⟨f, γ⟩``) — the ``lead(F_δ)`` term matters:
    flavoured dressings have F-coefficients with negative leading
    q-powers (a3 gen 0: −1), and omitting it loses exactly the
    ρ²-cyclicity tail pairs.  Measured with zero violations on 984
    stratified e6 samples + the a3/zoo exact suites.  Any γ with the
    bounds summing above ``K_joint`` has ``lead(c_a·c_b) > K_joint``
    and contributes nothing to the truncated output: dropping it is
    exact, not an approximation.  The joint bound is cone-monotone
    (each min strictly increases along cone moves), so it prunes the
    BFS safely.  Without it, the η-shell for a trace (vacuum × deep-F)
    is budgeted by the symmetric worst-case formula and explodes
    polynomially in the lattice rank (e6: 134 596 charges at the
    default cutoff where dozens contribute; e8 front seeds: OOM at
    16 GB — finding A10).
    """
    from collections import deque
    zero = tuple(0 for _ in range(rank))
    all_deltas: set[Vec] = set()
    per_F_deltas: list[list[Vec]] = []
    for F in Fs:
        if F is None:
            all_deltas.add(zero)
            per_F_deltas.append([zero])
        else:
            for delta in F:
                all_deltas.add(delta)
            per_F_deltas.append(list(F))
    if cone_witness is not None:
        f = tuple(int(x) for x in cone_witness)
        def _val(g: Vec) -> int:
            return sum(fi * gi for fi, gi in zip(f, g))
    else:
        def _val(g: Vec) -> int:
            return sum(abs(x) for x in g)

    if K_joint is not None and cone_witness is not None:
        # precompute, per F and per δ, ⟨f, δ⟩ − lead(F_δ): then
        # bound_x(γ) = ⟨f, γ⟩ − max_δ (…)  =  min_δ (lead F_δ + ⟨f, γ−δ⟩)
        def _lead(poly) -> int:
            if poly is None or isinstance(poly, int):
                return 0
            lp = poly.to_laurent() if hasattr(poly, "to_laurent") else poly
            cs = lp._coeffs if hasattr(lp, "_coeffs") else lp.coeffs
            nz = [e for e, c in cs.items() if c]
            return min(nz) if nz else 0
        per_F_maxdv: list[int] = []
        for F, ds in zip(Fs, per_F_deltas):
            if F is None:
                per_F_maxdv.append(0)
            else:
                per_F_maxdv.append(
                    max(_val(d) - _lead(F[d]) for d in ds))
        def _joint_ok(gamma: Vec) -> bool:
            gv = _val(gamma)
            tot = 0
            for mdv in per_F_maxdv:
                tot += gv - mdv
                if tot > K_joint:
                    return False
            return True
    else:
        def _joint_ok(gamma: Vec) -> bool:
            return True

    output_charges: set[Vec] = set()
    for delta in all_deltas:
        visited: set[Vec] = {zero}
        queue: deque = deque([zero])
        while queue:
            eta = queue.popleft()
            gamma = tuple(d + e for d, e in zip(delta, eta))
            if _val(gamma) <= cone_cutoff and _joint_ok(gamma):
                output_charges.add(gamma)
            for gi in cone_gens:
                nxt = tuple(e + g for e, g in zip(eta, gi))
                if nxt in visited:
                    continue
                nxt_gamma = tuple(d + e for d, e in zip(delta, nxt))
                if _val(nxt_gamma) > cone_cutoff or not _joint_ok(nxt_gamma):
                    continue
                visited.add(nxt)
                queue.append(nxt)
    return output_charges


def _habiro_to_ps(h: HabiroElement, K: int) -> PowerSeries:
    """Expand a HabiroElement to a truncated Laurent series (≤ q^K).

    NOTE: must keep negative q-powers — `c_a(η)` for deep-support F's
    routinely has them, and they cancel exactly against the
    `(q²;q²)_∞^r` prefactor in the final overlap. Filtering them out
    here would silently miss huge cancellations and produce wrong
    Schur indices.
    """
    if h.is_zero():
        return PowerSeries.zero(K)
    lp = h.expand(K)
    return PowerSeries({e: c for e, c in lp._coeffs.items() if e <= K}, K)


# ---------------------------------------------------------------------------
# Quantum-torus product on dict-of-LaurentPoly elements
# ---------------------------------------------------------------------------


def qt_multiply(
    A,
    B,
    lattice: Lattice,
) -> dict[Vec, LaurentPoly]:
    """The quantum-torus product `A · B` in `Q_Γ`, as `dict[Vec, LaurentPoly]`.

    `X_{γ_1} · X_{γ_2} = q^{⟨γ_1, γ_2⟩} X_{γ_1 + γ_2}`; for full elements
    we sum monomial-by-monomial and absorb the twist `q^{⟨γ_1, γ_2⟩}` into
    the LaurentPoly coefficient at the combined charge.

    Accepts dict[Vec, LaurentPoly] or dict[Vec, QNumberPoly] for either
    operand; the palindromic q-number representation is converted to
    LaurentPoly on entry (the QT twist destroys palindromy in general,
    so the output is always LaurentPoly).
    """
    def _as_lp_coeffs(coeff):
        if isinstance(coeff, QNumberPoly):
            return coeff.to_laurent()._coeffs
        return coeff._coeffs

    result: dict[Vec, dict[int, int]] = {}
    for g1, c1 in A.items():
        c1_coeffs = _as_lp_coeffs(c1)
        for g2, c2 in B.items():
            twist = lattice.bracket(g1, g2)
            ng = tuple(a + b for a, b in zip(g1, g2))
            bucket = result.setdefault(ng, {})
            c2_coeffs = _as_lp_coeffs(c2)
            for e1, v1 in c1_coeffs.items():
                for e2, v2 in c2_coeffs.items():
                    e = e1 + e2 + twist
                    bucket[e] = bucket.get(e, 0) + v1 * v2
    out: dict[Vec, LaurentPoly] = {}
    for g, c in result.items():
        lp = LaurentPoly(c)
        if not lp.is_zero():
            out[g] = lp
    return out


# ---------------------------------------------------------------------------
# Pointed-cone witness search and cone-order minimum
# ---------------------------------------------------------------------------


def compute_strict_cone_witness(
    rank: int,
    gens: Sequence[Sequence[int]],
    *,
    candidate: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """A strict integer functional `f` with `⟨f, g⟩ ≥ 1` for every
    generator `g` of the positive cone.

    Used by `find_lowest` to cheaply rank candidate cone-minima:
    strict positivity of `f` on the cone makes `L(c) := ⟨f, c⟩`
    strictly monotone in the cone partial order, so the inner loop
    of `find_lowest` can early-break once `L(other) ≥ L(cand)`.

    A witness exists iff the cone is **pointed** (no non-trivial
    non-negative combination of `gens` sums to zero); existence
    proves pointedness.  We try a few cheap candidates first, then
    fall back to a bounded box search.

    Parameters
    ----------
    rank
        Ambient lattice rank.
    gens
        Cone generators.  Empty `gens` returns the zero vector.
    candidate
        Optional certificate from the caller.  Verified in
        `O(rank · |gens|)`; if it satisfies the witness condition
        we return it directly without searching.

    Raises
    ------
    ValueError
        If the cone is not pointed (typical diagnoses: a generator
        equals zero, or two generators are antipodal) or no witness
        is found in the search box.
    """
    gens = [tuple(int(x) for x in g) for g in gens]
    if not gens:
        return tuple([0] * rank)

    for g in gens:
        if all(x == 0 for x in g):
            raise ValueError(
                "Positive cone is not pointed: a generator equals zero."
            )

    def _witnesses(f):
        return all(sum(fi * gi for fi, gi in zip(f, g)) >= 1 for g in gens)

    # Caller-supplied certificate (fast path).
    if candidate is not None:
        w = tuple(int(x) for x in candidate)
        if len(w) != rank:
            raise ValueError(
                f"candidate witness has length {len(w)}, expected rank={rank}"
            )
        if _witnesses(w):
            return w
        # Fall through to search: candidate failed, try harder.

    # Project out coordinates that are zero in every generator.  Padding
    # dimensions (e.g. U(1) slots in a PureADE) carry no information for
    # the cone check but inflate the box search exponentially.
    nonzero_coords = [k for k in range(rank)
                      if any(g[k] != 0 for g in gens)]
    if len(nonzero_coords) < rank:
        gens_proj = [tuple(g[k] for k in nonzero_coords) for g in gens]
        w_proj = compute_strict_cone_witness(len(nonzero_coords), gens_proj)
        w_full = [0] * rank
        for j, k in enumerate(nonzero_coords):
            w_full[k] = w_proj[j]
        return tuple(w_full)

    # Cheap candidates first.
    s = tuple(sum(g[i] for g in gens) for i in range(rank))
    if any(x != 0 for x in s) and _witnesses(s):
        return s
    for g in gens:
        if _witnesses(g):
            return g
    sg = tuple((1 if x > 0 else -1 if x < 0 else 0) for x in s)
    if any(x != 0 for x in sg) and _witnesses(sg):
        return sg

    # Box search (fast path for the common small-coordinate witnesses).
    M = max(max(abs(x) for x in g) for g in gens)
    bound = max(3, 2 * M)
    for f in _iproduct(range(-bound, bound + 1), repeat=rank):
        if _witnesses(f):
            return tuple(f)

    # Exact LP fallback: the strict witness can lie outside the box (a
    # near-antipodal generator pair forces a large coordinate — e.g. a
    # mutated flavoured BPS chamber whose witness needs a −4 entry while the
    # box stops at ±3).  `sigma_iso._lp_feasible_strict` is a two-phase
    # rational simplex that finds an integer witness iff the cone is pointed,
    # with no box (sound positive: ⟨f, g⟩ ≥ 1 since g, f are integral and
    # ⟨f, g⟩ > 0; sound negative by LP duality).  Reused, not reimplemented.
    from sigma_iso import _lp_feasible_strict
    feasible, w = _lp_feasible_strict(gens, rank)
    if feasible and w is not None and _witnesses(w):
        return tuple(int(x) for x in w)

    # Diagnostic: surface antipodal generator pairs.
    for i, gi in enumerate(gens):
        for j in range(i + 1, len(gens)):
            gj = gens[j]
            if all(a + b == 0 for a, b in zip(gi, gj)):
                raise ValueError(
                    f"Positive cone is not pointed: generators {gi} and "
                    f"{gj} are antipodal."
                )
    raise ValueError(
        f"Positive cone is not pointed: the exact LP strict-witness solve "
        f"(sigma_iso._lp_feasible_strict) found no f with ⟨f, g⟩ > 0 for "
        f"every generator (box search to [-{bound},{bound}]^{rank} also empty)."
    )


def find_lowest(
    charges: Sequence[Vec],
    cone_gens: Sequence[Vec],
    witness: Sequence[int],
) -> Vec:
    """Pick a charge minimal in the cone partial order on `charges`.

    `c'` dominates `c` iff `c - c'` is in the cone spanned by
    `cone_gens` (with non-negative integer coefficients).  A minimum
    is any charge with no proper dominator in `charges`.

    Implementation: rank by `L(c) = ⟨witness, c⟩`.  Strict positivity
    of `witness` on the cone makes `L` strictly monotone in the
    cone order, so any dominator of `c` has strictly smaller `L`,
    and the inner loop can early-break once `L(other) ≥ L(cand)`.

    The caller is responsible for supplying a strict witness — see
    `compute_strict_cone_witness`.  If no minimum is found (which
    should not happen on a pointed cone with a strict witness), we
    fall back to `min(charges)` for safety.
    """
    cone_t = [tuple(int(x) for x in g) for g in cone_gens]
    f = tuple(int(x) for x in witness)
    in_cone = make_cone_predicate(cone_t)

    def L_of(c):
        return sum(fi * ci for fi, ci in zip(f, c))

    ranked = sorted(charges, key=lambda c: (L_of(c), c))
    L = {c: L_of(c) for c in ranked}
    for cand in ranked:
        Lc = L[cand]
        is_lowest = True
        for other in ranked:
            if other is cand:
                continue
            if L[other] >= Lc:
                break
            diff = tuple(c - o for c, o in zip(cand, other))
            if in_cone(diff):
                is_lowest = False
                break
        if is_lowest:
            return cand
    return min(charges)
