"""Local Nahm-sum primitives for ``[S|0>]_γ`` and ``[F·S|0>]_γ``.

This module provides the per-γ Nahm-coefficient entry points
(``s_gamma_habiro``, ``c_gamma_habiro``) plus **n-driven** bulk builders
that compute ``[S|0>]_γ`` for an entire γ-region in a single pass.  It is
deliberately self-contained (LaurentPoly / lattice / the localised-ring
arithmetic only), so both the spine-free vacuum path (``vacuum_nahm``)
and the BPS realisation layer (``BPSKAlgebra``, the F-solver,
Schur-index) can consume it.

Why n-driven.  The per-γ path inverts ``Σ nᵢ γᵢ = γ`` via
Gaussian elimination over ``Fraction`` plus a backtracking
enumeration of free variables (see ``_solve_nahm_indices``).  That
is the exact lookup, but it pays the inversion cost once per γ.  An
n-driven walk visits every Nahm tuple **once**, looks up its γ in
O(1), and fills every γ in the requested region simultaneously --
equivalent to multiplying out

    S = E_𝖖(X_{γ_1}) · ⋯ · E_𝖖(X_{γ_N})

term by term.  The rejection predicate ``accept_gamma`` prunes
the recursion as soon as the partial γ leaves the target region;
because the cone is pointed and every spec charge is cone-positive,
once the partial γ overshoots ``upper`` it stays out under further
cone-positive additions, so pruning is sound.

The pointwise API is kept uniform so spec-mode and recipe-mode code
paths in ``BPSKAlgebra`` stay interchangeable.  In spec mode the bulk
path pre-fills the per-γ cache, so subsequent ``s_gamma_habiro`` calls
are O(1) lookups.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Sequence

from laurent_poly import LaurentPoly
from lattice import Lattice, Vec
from lattice import cone_contains
from habiro import HabiroElement


# ---------------------------------------------------------------------------
# Caches (module-local; clear via ``clear_nahm_cache``).
# ---------------------------------------------------------------------------

# Nahm-tuple enumeration is a function of (γ, spec_t) only; pairings
# enter only at the shift step.
_solve_cache: dict[tuple[Vec, tuple], list[tuple[int, ...]]] = {}
# Final ``[S|0>]_γ`` value depends on (γ, spec_t, kmat).
_sket_habiro_cache: dict[tuple[Vec, tuple, tuple], HabiroElement] = {}


def clear_nahm_cache() -> None:
    """Clear all module-local Nahm caches."""
    _solve_cache.clear()
    _sket_habiro_cache.clear()


# ---------------------------------------------------------------------------
# Per-γ Nahm-index solver (cold path).
# ---------------------------------------------------------------------------

def _solve_nahm_indices(
    gamma: Vec,
    spec_t: Sequence[Vec],
    max_n: int = 50,
) -> list[tuple[int, ...]]:
    """All ``(n_1, …, n_N)`` with ``n_a ≥ 0`` and ``Σ n_a γ_a = γ``.

    Gaussian elimination over Q for a particular rational solution
    plus the kernel; enumerate free variables over non-negative
    integers, checking integrality and non-negativity of the pivot
    variables.
    """
    N = len(spec_t)
    r = len(gamma)

    M = [[Fraction(spec_t[j][i]) for j in range(N)] + [Fraction(gamma[i])]
         for i in range(r)]

    pivots: list[int] = []
    row = 0
    for col in range(N):
        piv = None
        for rr in range(row, r):
            if M[rr][col] != 0:
                piv = rr
                break
        if piv is None:
            continue
        pivots.append(col)
        M[row], M[piv] = M[piv], M[row]
        for rr in range(r):
            if rr != row and M[rr][col] != 0:
                f = M[rr][col] / M[row][col]
                for cc in range(N + 1):
                    M[rr][cc] -= f * M[row][cc]
        row += 1

    for rr in range(row, r):
        if M[rr][N] != 0:
            return []

    base = [Fraction(0)] * N
    for idx, col in enumerate(pivots):
        base[col] = M[idx][N] / M[idx][col]

    free = [j for j in range(N) if j not in pivots]

    if not free:
        if all(v.denominator == 1 and v >= 0 for v in base):
            return [tuple(int(v) for v in base)]
        return []

    pivot_effect: dict[int, dict[int, Fraction]] = {}
    for fi in free:
        pe: dict[int, Fraction] = {}
        for idx, col in enumerate(pivots):
            c = -M[idx][fi] / M[idx][col]
            if c != 0:
                pe[col] = c
        pivot_effect[fi] = pe

    results: list[tuple[int, ...]] = []

    def _enumerate(fi_pos: int, ns: list[Fraction]) -> None:
        if fi_pos == len(free):
            if all(v.denominator == 1 and v >= 0 for v in ns):
                results.append(tuple(int(v) for v in ns))
            return

        fi = free[fi_pos]
        saved = ns[:]
        for t in range(max_n + 1):
            ns[fi] = Fraction(t)
            for col, c in pivot_effect[fi].items():
                ns[col] = saved[col] + Fraction(t) * c
            can_continue = True
            for col in pivots:
                if ns[col] < 0:
                    c_fi = pivot_effect[fi].get(col, Fraction(0))
                    if c_fi <= 0:
                        fixable_later = any(
                            pivot_effect[lf].get(col, Fraction(0)) > 0
                            for lf in free[fi_pos + 1:]
                        )
                        if not fixable_later:
                            can_continue = False
                            break
            if not can_continue:
                break
            _enumerate(fi_pos + 1, ns)
        for i in range(N):
            ns[i] = saved[i]

    _enumerate(0, list(base))
    return results


def _nahm_shift(ns: Sequence[int], kmat: Sequence[Sequence[int]]) -> int:
    """``Σ n_a + Σ_{a<b} n_a n_b k_{ab}``."""
    N = len(ns)
    s = sum(ns)
    for i in range(N):
        ni = ns[i]
        if ni == 0:
            continue
        row = kmat[i]
        for j in range(i + 1, N):
            s += ni * ns[j] * row[j]
    return s


# ---------------------------------------------------------------------------
# Pointwise ``[S|0>]_γ`` (cold path).  Used in recipe mode and as the
# fallback when a γ outside any precomputed region is requested.
# ---------------------------------------------------------------------------

def s_gamma_habiro(
    gamma: Vec,
    spec_t: Sequence[Vec],
    kmat: Sequence[Sequence[int]],
) -> HabiroElement:
    """``[S|0>]_γ`` as an exact ``HabiroElement``.

    Cached per ``(γ, spec_t, kmat)``.  Pairings enter only via
    ``kmat``; two theories with the same ``spec_t`` but different
    pairings produce different shifts, so ``kmat`` is part of the key.
    """
    spec_key = tuple(tuple(g) for g in spec_t)
    kmat_key = tuple(tuple(row) for row in kmat)
    key = (gamma, spec_key, kmat_key)
    cached = _sket_habiro_cache.get(key)
    if cached is not None:
        return cached

    idx_key = (gamma, spec_key)
    if idx_key in _solve_cache:
        all_ns = _solve_cache[idx_key]
    else:
        all_ns = _solve_nahm_indices(gamma, spec_t)
        _solve_cache[idx_key] = all_ns

    if not all_ns:
        result = HabiroElement.zero()
    else:
        terms = []
        for ns in all_ns:
            sign = 1 if sum(ns) % 2 == 0 else -1
            shift = _nahm_shift(ns, kmat)
            terms.append(HabiroElement.nahm_term(sign, shift, list(ns)))
        result = HabiroElement.sum(terms)
    _sket_habiro_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# n-driven bulk: enumerate every Nahm tuple ``(n_1, …, n_N)`` once,
# bucket by ``γ = Σ n_i γ_i``.  This is what direct E_q multiplication
# would do, term by term.
# ---------------------------------------------------------------------------

def enumerate_nahm_buckets(
    spec_t: Sequence[Vec],
    accept_gamma: Callable[[Vec], bool],
    *,
    rank: int | None = None,
) -> dict[Vec, list[tuple[int, ...]]]:
    """All Nahm tuples whose γ is accepted, bucketed by γ.

    Contract on the predicate.  ``accept_gamma(γ)`` must be
    **cone-monotone**: there is a pointed cone ``C`` containing every
    spec charge such that

        γ rejected  ∧  c ∈ C  ⟹  γ + c rejected.

    The standard instance is ``accept_gamma(γ) = (upper − γ ∈ C)``
    for some pointed cone ``C`` containing the spec charges; the
    region's complement is then closed under cone-positive
    additions because ``v ∉ C ∧ c ∈ C ⟹ v − c ∉ C`` (else
    ``v = (v − c) + c ∈ C``).

    The walk prunes at the first rejection on the partial γ.
    Termination is guaranteed by the contract: any infinite chain
    of cone-positive additions eventually exits the region.

    ``rank`` is required when ``spec_t`` is empty.
    """
    spec_t = [tuple(g) for g in spec_t]
    N = len(spec_t)
    if N == 0:
        if rank is None:
            raise ValueError("rank must be supplied when spec_t is empty")
        zero = tuple(0 for _ in range(rank))
        return {zero: [()]} if accept_gamma(zero) else {}

    if rank is None:
        rank = len(spec_t[0])
    zero = tuple(0 for _ in range(rank))
    if not accept_gamma(zero):
        return {}

    buckets: dict[Vec, list[tuple[int, ...]]] = {}
    ns = [0] * N

    def recurse(i: int, gamma_partial: Vec) -> None:
        if i == N:
            buckets.setdefault(gamma_partial, []).append(tuple(ns))
            return
        gi = spec_t[i]
        gamma_n = gamma_partial
        n = 0
        while True:
            ns[i] = n
            recurse(i + 1, gamma_n)
            n += 1
            gamma_n = tuple(c + gk for c, gk in zip(gamma_n, gi))
            if not accept_gamma(gamma_n):
                break
        ns[i] = 0

    recurse(0, zero)
    return buckets


def gammas_to_q_order(
    spec_t: Sequence[Vec],
    kmat: Sequence[Sequence[int]],
    K: int,
    *,
    rank: int | None = None,
    max_total_n: int | None = None,
    stabilize: bool = False,
) -> set[Vec]:
    """Set of γ's such that some Nahm tuple `n = (n_1, …, n_N)` with
    `Σ n_a γ_a = γ` has `shift(n) ≤ K`, where
    `shift(n) = Σ n_a + Σ_{a<b} n_a n_b kmat[a][b]`.

    Used by ``BPSKAlgebra.rg_generator(K)`` to determine the γ-set
    whose ``s_γ`` data should be included in the truncated spectrum
    generator.

    Implementation
    --------------
    DFS over Nahm tuples with the search space bounded by
    ``Σ n_a ≤ max_total_n``. Each terminal tuple's `shift(n)` is
    computed; if `≤ K`, the corresponding γ is added to the result.

    The default ``max_total_n`` is chosen based on the off-diagonal
    range of ``kmat``:

    * If all off-diagonal ``kmat[a][b]`` (a < b) are non-negative,
      `shift(n) ≥ Σ n_a` so `max_total_n = K` is the tight sound
      bound. This covers pentagon, SU(2), pure ADE in the SC
      interleaved basis.
    * If any off-diagonal is negative (e.g., hexagon, certain SU(N)
      Nf chambers, …), the negative quadratic part can bring shift
      below `Σ n_a`. The default falls back to
      ``2 * K * (1 + |min_neg|) + 2``, which is empirically
      sufficient for the mixed-sign theories included here; it is
      *not* mathematically sound in the worst case (a sufficiently
      negative single off-diagonal can produce shift-≤-K tuples
      with arbitrarily large `Σ n_a`). When in doubt, pass
      ``max_total_n`` explicitly, or set ``stabilize=True``.

    ``stabilize=True`` (used by the ``Tr(1)`` path
    ``vacuum_nahm.vacuum_trace_rps``) closes the mixed-sign hole: it
    auto-widens the bound until the shift-≤-K γ-set is stable over two
    consecutive windows (sound because over-inclusion is harmless and the
    region is finite for a genuine BPS pairing), and **raises** rather than
    silently truncating if it never stabilises. A no-op for non-negative
    pairings (the closed-form bound is already tight and sound) and when an
    explicit ``max_total_n`` is given.

    Over-inclusion is harmless for ``rg_generator`` consumption:
    a γ in the set whose minimum-shift tuple has shift > K
    contributes only at q-orders > K and is invisible to
    ``s_γ.expand(K)``.

    Parameters
    ----------
    spec_t
        Spec charges as a list of Vec.
    kmat
        Pairwise pairing matrix (the same `kmat` used by
        ``s_gamma_habiro`` and ``_nahm_shift``).
    K
        q-order cutoff.
    rank
        Required when ``spec_t`` is empty.
    max_total_n
        Upper bound on `Σ n_a` for the DFS. Defaults to the
        per-spec bound described above.
    """
    spec_t = [tuple(g) for g in spec_t]
    kmat = [list(row) for row in kmat]
    N = len(spec_t)
    if N == 0:
        if rank is None:
            raise ValueError("rank must be supplied when spec_t is empty")
        zero = tuple(0 for _ in range(rank))
        return {zero} if K >= 0 else set()
    if rank is None:
        rank = len(spec_t[0])
    min_off_diag = min(
        (kmat[i][j] for i in range(N) for j in range(i + 1, N)),
        default=0,
    )
    user_bound = max_total_n is not None
    if max_total_n is None:
        if min_off_diag >= 0:
            max_total_n = max(K, 0)
        else:
            max_total_n = max(2 * K * (1 + abs(min_off_diag)) + 2, 0)

    if K < 0:
        return set()

    zero = tuple(0 for _ in range(rank))

    def _enum(mtn: int) -> set[Vec]:
        """γ's with some Nahm tuple of shift ≤ K and Σ n_a ≤ ``mtn``."""
        if mtn < 0:
            return set()
        found: set[Vec] = {zero}    # zero tuple has shift = 0, always included
        ns = [0] * N

        def recurse(i: int, total_so_far: int, gamma_partial: Vec) -> None:
            if i == N:
                if _nahm_shift(ns, kmat) <= K:
                    found.add(gamma_partial)
                return
            gi = spec_t[i]
            upper = mtn - total_so_far
            new_gamma = gamma_partial
            for n in range(upper + 1):
                ns[i] = n
                recurse(i + 1, total_so_far + n, new_gamma)
                new_gamma = tuple(c + gk for c, gk in zip(new_gamma, gi))
            ns[i] = 0

        recurse(0, 0, zero)
        return found

    # Sound case: an explicit caller bound, or a non-negative pairing
    # (`shift(n) ≥ Σ n_a`, so `max_total_n = K` is the tight sound bound), or
    # the caller did not ask for the stabilisation guard.
    if user_bound or not stabilize or min_off_diag >= 0:
        return _enum(max_total_n)

    # stabilize=True on a MIXED-SIGN pairing: the closed-form bound is not
    # provably sound (a negative quadratic term can push a shift-≤-K tuple past
    # Σ n_a = max_total_n and be silently dropped).  Auto-widen until the
    # shift-≤-K γ-set STOPS growing.  Over-inclusion is harmless (extra γ
    # contribute only at q-orders > K), and for a genuine BPS pairing the
    # shift-≤-K region is finite, so successive no-growth windows certify
    # completeness.  Require TWO consecutive stable windows; raise if it never
    # stabilises (so Tr(1) is either exact or an explicit error -- never
    # silently truncated).
    step = max(K, 1)
    bound = max_total_n
    prev = _enum(bound)
    stable = 0
    for _ in range(64):
        bound += step
        cur = _enum(bound)
        if cur == prev:
            stable += 1
            if stable >= 2:
                return cur
        else:
            stable = 0
        prev = cur
    raise RuntimeError(
        f"gammas_to_q_order: Nahm γ-set did not stabilise for a mixed-sign "
        f"pairing (min off-diagonal {min_off_diag}) up to Σn_a={bound} at "
        f"q-order K={K}; Tr(1) cannot be certified exact -- pass an explicit "
        f"sound max_total_n."
    )


def s_table(
    spec_t: Sequence[Vec],
    kmat: Sequence[Sequence[int]],
    accept_gamma: Callable[[Vec], bool],
    *,
    populate_cache: bool = True,
) -> dict[Vec, HabiroElement]:
    """``{γ : [S|0>]_γ}`` for every γ in the accepted region.

    Single n-driven enumeration of Nahm tuples; each tuple becomes one
    ``HabiroElement.nahm_term``; same-γ terms are summed via
    ``HabiroElement.sum`` (one ``simplify`` per γ).

    ``populate_cache=True`` (default) also fills the module-level per-γ
    cache, so subsequent ``s_gamma_habiro(γ, spec_t, kmat)`` calls are
    O(1) lookups for any γ in the accepted region.  γ's outside the
    region are recorded as zero only if ``accept_gamma(γ)`` returns
    True and the bucket is empty -- otherwise they fall through to the
    per-γ solver.
    """
    spec_t = [tuple(g) for g in spec_t]
    kmat = [list(row) for row in kmat]
    buckets = enumerate_nahm_buckets(spec_t, accept_gamma)

    out: dict[Vec, HabiroElement] = {}
    for gamma, tuples in buckets.items():
        terms = []
        for ns in tuples:
            sign = 1 if sum(ns) % 2 == 0 else -1
            shift = _nahm_shift(ns, kmat)
            terms.append(HabiroElement.nahm_term(sign, shift, list(ns)))
        out[gamma] = HabiroElement.sum(terms)

    if populate_cache:
        spec_key = tuple(tuple(g) for g in spec_t)
        kmat_key = tuple(tuple(row) for row in kmat)
        for gamma, val in out.items():
            _sket_habiro_cache[(gamma, spec_key, kmat_key)] = val

    return out


def s_table_in_cone(
    spec_t: Sequence[Vec],
    kmat: Sequence[Sequence[int]],
    cone_gens: Sequence[Vec],
    upper: Vec,
    *,
    populate_cache: bool = True,
) -> dict[Vec, HabiroElement]:
    """``[S|0>]_γ`` for every γ with ``upper − γ ∈ cone(cone_gens)``.

    Convenience wrapper around ``s_table`` with the standard cone
    region predicate.  This is the natural region for the F-solver:
    γ ranges over ``δ − δ'`` for δ, δ' in the doubly-tropical support
    of ``F_{γ_F}``, which fits inside ``[0, upper − γ_F]``.

    Caller is responsible for supplying a ``cone_gens`` whose pointed
    cone contains every spec charge; otherwise the walk is not
    guaranteed to terminate.
    """
    from lattice import make_cone_predicate
    upper = tuple(upper)
    cone_gens_t = [tuple(g) for g in cone_gens]
    in_cone = make_cone_predicate(cone_gens_t)

    def accept(gamma: Vec) -> bool:
        return in_cone(tuple(u - g for u, g in zip(upper, gamma)))

    return s_table(spec_t, kmat, accept, populate_cache=populate_cache)


# ---------------------------------------------------------------------------
# ``[F·S|0>]_γ`` from a precomputed s_table.
# ---------------------------------------------------------------------------

def fs_dict_from_s_table(
    F_dict,
    s_tbl: dict[Vec, HabiroElement],
    lattice: Lattice,
    accept_gamma: Callable[[Vec], bool] | None = None,
) -> dict[Vec, HabiroElement]:
    """Assemble ``{γ : [F·S|0>]_γ}`` from F and a precomputed s_table.

        [F·S|0>]_γ = Σ_{δ ∈ F} f_δ · q^{⟨δ, γ−δ⟩} · [S|0>]_{γ−δ}

    γ ranges over ``support(F) + support(s_tbl)``; ``accept_gamma``
    can further restrict the output to a γ-region of interest.

    F coefficients may be :class:`LaurentPoly` or :class:`QNumberPoly`;
    palindromic native form is converted to LaurentPoly on entry.
    """
    contribs_by_gamma: dict[Vec, list[HabiroElement]] = {}

    for delta, coeff in F_dict.items():
        f_lp = coeff.to_laurent() if hasattr(coeff, "to_laurent") else coeff
        if f_lp.is_zero():
            continue
        for mu, s_el in s_tbl.items():
            if s_el.is_zero():
                continue
            gamma = tuple(d + m for d, m in zip(delta, mu))
            if accept_gamma is not None and not accept_gamma(gamma):
                continue
            twist = lattice.bracket(delta, mu)
            scale = f_lp * LaurentPoly({twist: 1})
            term = HabiroElement(scale * s_el.numerator, dict(s_el.denom))
            contribs_by_gamma.setdefault(gamma, []).append(term)

    return {
        gamma: HabiroElement.sum(contribs)
        for gamma, contribs in contribs_by_gamma.items()
    }


def fs_dict_for_eta_set(
    spec_t: Sequence[Vec],
    kmat: Sequence[Sequence[int]],
    F_dict,
    eta_set: Sequence[Vec],
    lattice: Lattice,
    cone_witness: Sequence[int],
    *,
    populate_cache: bool = True,
) -> tuple[dict[Vec, HabiroElement], dict[Vec, HabiroElement]]:
    """Compute ``[F·S|0>]_η`` for every η in ``eta_set`` in one n-walk.

    F coefficients may be :class:`LaurentPoly` or :class:`QNumberPoly`
    (palindromic native form); see :func:`fs_dict_from_s_table`.

    Driver for the Schur-index path.  The η-region of interest is
    enumerated by the caller (``_enumerate_output_charges`` in the
    BPSKAlgebra Schur path) and supplied here directly.

    Steps:

    1. Determine the smallest cone-witness L-shell containing every
       ``μ = η − δ`` with η ∈ ``eta_set`` and δ in F's support:
       ``L_max = max ⟨f, η − δ⟩`` for the strict cone-pointedness
       witness ``f = cone_witness``.  ``μ`` is what ``[S|0>]_μ`` is
       queried at; only μ ∈ cone contributes (else ``[S|0>]_μ = 0``),
       and for cone-positive γ the predicate ``⟨f, γ⟩ ≤ L_max`` is
       cone-monotone (every spec charge has ``⟨f, ·⟩ ≥ 1``), so the
       walk terminates and covers every needed μ.
    2. Run a single n-driven Nahm-tuple walk over that L-shell,
       building the ``s_tbl``.
    3. Assemble ``[F·S|0>]_η`` for every η ∈ ``eta_set`` by summing
       ``f_δ · q^{⟨δ, η−δ⟩} · s_tbl[η−δ]``.

    Returns ``(s_tbl, fs_dict)``.  Empty F is treated as the
    identity: ``fs_dict = {η : [S|0>]_η}`` for η ∈ ``eta_set``.

    The earlier implementation initialised an integer ``upper`` to
    ``[0]*rank`` and only ratcheted up componentwise, which is wrong
    for cones whose generators have negative coordinates (e.g.
    ``cone_gens = [(1,0), (0,-1)]``): every required μ with negative
    second coordinate fell outside the box, the walk returned a
    truncated table, and ``[F·S|0>]_η`` came out zero where it
    shouldn't.  The witness-based L-shell is cone-direction-agnostic and always
    contains the needed μ.
    """
    if not eta_set:
        return {}, {}
    eta_list = list(eta_set)
    rank = len(eta_list[0])
    f = tuple(int(x) for x in cone_witness)

    # Compute L_max = max ⟨f, μ⟩ over μ = η − δ, where ⟨f, ·⟩ is the
    # strict cone-witness functional (⟨f, g⟩ ≥ 1 for every cone gen).
    deltas = list(F_dict.keys()) if F_dict else [tuple(0 for _ in range(rank))]
    L_max = 0
    for eta in eta_list:
        for delta in deltas:
            lv = sum(fk * (eta[k] - delta[k]) for k, fk in enumerate(f))
            if lv > L_max:
                L_max = lv

    # Predicate is cone-monotone in spec mode (each spec charge has
    # ⟨f, g⟩ ≥ 1, so ⟨f, γ + g⟩ > ⟨f, γ⟩); ``enumerate_nahm_buckets``
    # only walks γ-partials in the cone-positive cone of ``spec_t``.
    def accept(gamma: Vec) -> bool:
        return sum(fk * gk for fk, gk in zip(f, gamma)) <= L_max

    s_tbl = s_table(spec_t, kmat, accept, populate_cache=populate_cache)

    if not F_dict:
        eta_filter = set(eta_list)
        fs_dict = {
            eta: s_tbl[eta]
            for eta in s_tbl
            if eta in eta_filter
        }
        return s_tbl, fs_dict

    eta_filter = set(eta_list)
    fs_dict = fs_dict_from_s_table(
        F_dict, s_tbl, lattice,
        accept_gamma=eta_filter.__contains__,
    )
    return s_tbl, fs_dict


def c_gamma_habiro(
    gamma: Vec,
    F: dict[Vec, LaurentPoly] | None,
    spec_t: Sequence[Vec],
    kmat: Sequence[Sequence[int]],
    lattice: Lattice,
) -> HabiroElement:
    """``[F·S|0>]_γ`` as a HabiroElement (pointwise).

    Single-γ entry point.
    Uses the cached per-γ ``s_gamma_habiro``; the bulk path
    ``fs_dict_from_s_table`` is preferred when many γ's are needed.
    """
    if F is None:
        return s_gamma_habiro(gamma, spec_t, kmat)

    contribs = []
    for delta, coeff in F.items():
        f_coeff = coeff.to_laurent() if hasattr(coeff, "to_laurent") else coeff
        if f_coeff.is_zero():
            continue
        eta = tuple(g - d for g, d in zip(gamma, delta))
        s_eta = s_gamma_habiro(eta, spec_t, kmat)
        if s_eta.is_zero():
            continue
        twist = lattice.bracket(delta, eta)
        scale = f_coeff * LaurentPoly({twist: 1})
        contribs.append(HabiroElement(scale * s_eta.numerator, dict(s_eta.denom)))
    return HabiroElement.sum(contribs)
