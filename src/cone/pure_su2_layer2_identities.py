"""Pure SU(2) Layer 2 trace identities.

`tr_h0_bridge` (m=1 anchor) and the m=2 anchor bridges (`tr_h0sq_bridge`,
`tr_h1sq_bridge`) live here.  Both derive their values from ρ²-twisted
cyclicity identities on the H-tower.

Closed-form trace identities derived from ρ²-twisted cyclicity on the
H-tower.  These provide algebraic recursions that the SU(2) Schur trace
must satisfy.  Complementary to `pure_su2_h_abelianized.PureSU2HAbeKAlg.
trace` (which computes the Schur measure directly via the abelianized
chart).

Identities:

1. Wilson recursion (Layer 1, single-Wilson traces).
   Define G(n) := Tr(W_{2n}) − Tr(W_{2n+2}).  Then for n ≥ 2:

       G(n) = q^{2n} · G(n−2)

   Equivalently, G(2k) = q^{2k(k+1)} · G(0) and G(2k+1) = q^{2k(k+2)} ·
   G(1).  Verified against the SU(2) Schur closed form

       F(v) = (q² v²; q²)_∞² · (q² v^{-2}; q²)_∞² · (q²; q²)_∞²

   where Tr(W_n) = [coef of χ_n(v) in F(v)], by direct q-expansion.

2. Tr(H_0) bridge formula (Layer 2, magnetic anchor).
   Derived from the cyclicity identity

       Tr(H_{−a} · H_a − q^{2a} · H_0²) =
           Tr(ρ²(H_a) · H_{−a} − q^{2a} · ρ²(H_0) · H_0)

   for a = 1, 2 (eliminating Tr(H_0²) between the two), yielding Tr(H_0)
   as a Q(q)-rational combination of Tr(W_0), Tr(W_2), Tr(W_4):

       Tr(H_0) = −q^{−1} · Tr(W_0)
                 + (2q² − 1) · Tr(W_2) / [2q³(1 − q²)]
                 + Tr(W_4) / [2q³(1 − q²)]

   Verified at q ∈ {1/2, 1/3, 1/5, 1/7, 2/5, 3/7} against a reference
   pure-SU(2) realisation (not included in this repository), residuals
   < 10^{−9} (Schur-truncation noise at q-cap 30).
"""
from __future__ import annotations
from fractions import Fraction

from laurent_poly import LaurentPoly
from pure_su2_h_gap_k import h_mul_h


def G(W_lps: dict[int, LaurentPoly], n: int) -> LaurentPoly:
    """G(n) := Tr(W_{2n}) − Tr(W_{2n+2})."""
    return _lp_sub(W_lps[2 * n], W_lps[2 * n + 2])


def wilson_recursion_check(W_lps: dict[int, LaurentPoly],
                            n_range=(2, 3, 4),
                            q_max: int = 24) -> bool:
    """Verify G(n) = q^{2n} · G(n−2) at each n in `n_range`, against a
    dict {2k: Tr(W_{2k}) as LaurentPoly in q}.  Returns True if the
    recursion holds exactly to q^q_max."""
    for n in n_range:
        diff = _lp_sub(G(W_lps, n), _lp_shift(G(W_lps, n - 2), 2 * n))
        diff = _lp_truncate(diff, q_max)
        if not diff.is_zero():
            return False
    return True


def tr_h0_bridge(tr_w0: LaurentPoly, tr_w2: LaurentPoly,
                  tr_w4: LaurentPoly, q_max: int = 30) -> LaurentPoly:
    """Compute Tr(H_0) as a Q(q)-rational combination of Tr(W_0),
    Tr(W_2), Tr(W_4):

        Tr(H_0) = −q^{−1} · Tr(W_0)
                  + [(2q² − 1) · Tr(W_2) + Tr(W_4)] / [2q³(1 − q²)]

    The denominator (1 − q²) is expanded as the formal q-Laurent series
    1 + q² + q⁴ + …, truncated at q^q_max.
    """
    # Term 1: −q^{−1} · Tr(W_0)
    term1 = _lp_shift(_lp_neg(tr_w0), -1)
    # Bracket: (2q² − 1) · Tr(W_2) + Tr(W_4)
    #        = 2·q² · Tr(W_2) − Tr(W_2) + Tr(W_4)
    bracket = _lp_add(
        _lp_add(_lp_scale_int(_lp_shift(tr_w2, 2), 2), _lp_neg(tr_w2)),
        tr_w4,
    )
    # bracket / [2 q³ (1 − q²)] = (1/2) · q^{−3} · bracket · Σ_{k≥0} q^{2k}
    geom_max = max(q_max + 4, 16)
    geom = LaurentPoly({2 * k: 1 for k in range(geom_max // 2 + 1)})
    bracket_div = _lp_truncate(bracket * geom, q_max + 6)
    bracket_div = _lp_shift(bracket_div, -3)
    bracket_div = _lp_half(bracket_div)
    return _lp_truncate(_lp_add(term1, bracket_div), q_max)


# ---------- m=2 anchor bridges ----------
#
# Cyclicity Tr(H_a · H_b) = Tr(H_{b-8} · H_a) yields, at (a, b) =
# (-1, 1) and (0, 2):
#
#   (a, b) = (-1, 1):  LHS = h_mul_h(-1, 1) = q · L_{(1, 0)} + q² · L_{(2, 0)}
#                      RHS = h_mul_h(-7, -1)
#                          = (2 + q²) · L_{(0, 0)} + q^{-2} · L_{(0, 2)}
#                            + (3q + q³ + q⁵) · [L_{(1, even)} traces fold to Tr(H_0)]
#                            + q⁶ · L_{(2, -8)}  [e = -8 ≡ 0 mod 4: Tr(H_0²)]
#
#                      ⇒ (q² − q⁶) · Tr(H_0²)
#                          = (2 + q²) Tr(W_0) + q^{-2} Tr(W_2)
#                            + (2q + q³ + q⁵) Tr(H_0)
#                        [Tr(L_{(1, 0)}) on LHS shifts the q · Tr(H_0)
#                         coefficient by −q]
#
#   (a, b) = ( 0, 2):  LHS = h_mul_h( 0, 2) = q² · L_{(2, 2)}
#                      RHS = h_mul_h(-6, 0)
#                          = (2 + q²) L_{(0, 0)} + q^{-2} L_{(0, 2)}
#                            + (2q + 2q³) [Tr(H_0) folded]
#                            + q⁶ · L_{(2, -6)}  [e = -6 ≡ 2 mod 4: Tr(H_1²)]
#
#                      ⇒ (q² − q⁶) · Tr(H_1²)
#                          = (2 + q²) Tr(W_0) + q^{-2} Tr(W_2)
#                            + (2q + 2q³) Tr(H_0)
#
# Both bridges share the denominator q² · (1 − q⁴); the bracket numerators
# differ only in the Tr(H_0) coefficient.

_TWO_PLUS_Q2 = LaurentPoly({0: 2, 2: 1})              # 2 + q²
_Q_MINUS_2   = LaurentPoly({-2: 1})                   # q^{-2}


def _bracket_div_q2_one_minus_q4(num: LaurentPoly, q_max: int) -> LaurentPoly:
    """Multiply `num` by `q^{-2} · (1 + q⁴ + q⁸ + …)` truncated to `q^{q_max}`.

    Implements division by `q² · (1 − q⁴)` against the formal Laurent
    power series in q (no rational-function objects; the geometric
    series expansion is the only truncation step, deferred until here).
    """
    geom_max = max(q_max + 4, 16)
    geom = LaurentPoly({4 * k: 1 for k in range(geom_max // 4 + 1)})
    divided = _lp_truncate(num * geom, q_max + 4)
    return _lp_truncate(_lp_shift(divided, -2), q_max)


def tr_h0sq_bridge(tr_w0: LaurentPoly, tr_w2: LaurentPoly,
                    tr_H0: LaurentPoly, q_max: int = 30) -> LaurentPoly:
    """Compute `Tr(L_{(2, 0)}) = Tr(H_0²)` as a `Q(q)`-rational combination
    of `Tr(W_0)`, `Tr(W_2)`, `Tr(H_0)`:

        Tr(H_0²) = [(2 + q²)·Tr(W_0) + q⁻²·Tr(W_2)
                    + (2q + q³ + q⁵)·Tr(H_0)] / [q²·(1 − q⁴)]

    Derived from ρ²-cyclicity `Tr(H_{−1}·H_1) = Tr(H_{−7}·H_{−1})`
    with the canonical `h_mul_h` expansion of both sides.
    """
    h0_coef = LaurentPoly({1: 2, 3: 1, 5: 1})         # 2q + q³ + q⁵
    num = _lp_add(
        _lp_add(_TWO_PLUS_Q2 * tr_w0, _Q_MINUS_2 * tr_w2),
        h0_coef * tr_H0,
    )
    return _bracket_div_q2_one_minus_q4(num, q_max)


def tr_h1sq_bridge(tr_w0: LaurentPoly, tr_w2: LaurentPoly,
                    tr_H0: LaurentPoly, q_max: int = 30) -> LaurentPoly:
    """Compute `Tr(L_{(2, 2)}) = Tr(H_1²)` (the second m=2 H-shift anchor)
    as a `Q(q)`-rational combination of `Tr(W_0)`, `Tr(W_2)`, `Tr(H_0)`:

        Tr(H_1²) = [(2 + q²)·Tr(W_0) + q⁻²·Tr(W_2)
                    + (2q + 2q³)·Tr(H_0)] / [q²·(1 − q⁴)]

    Derived from ρ²-cyclicity `Tr(H_0·H_2) = Tr(H_{−6}·H_0)`.
    """
    h0_coef = LaurentPoly({1: 2, 3: 2})               # 2q + 2q³
    num = _lp_add(
        _lp_add(_TWO_PLUS_Q2 * tr_w0, _Q_MINUS_2 * tr_w2),
        h0_coef * tr_H0,
    )
    return _bracket_div_q2_one_minus_q4(num, q_max)


# ---------- m=3 anchor bridges via 3-letter cyclicity --------------
#
# H-shift on m=3 sends e -> e + 6, so three even-e anchors:
#   Tr(L_{(3, 0)}) = Tr(H_0^3),
#   Tr(L_{(3, 2)}),
#   Tr(L_{(3, 4)}).
#
# Three independent 3-letter cyclicity equations
#   Tr(H_a H_b H_c) = Tr(H_{c-8} H_a H_b)
# at triples (a, b, c) in {(-1, 0, 1), (-1, 0, 3), (-1, 0, 5)} give
# a 3x3 linear system in the three anchors with coefficient matrix
#
#   M = [[ q^4,  0,    -q^14],
#        [-q^10, q^8,   0  ],
#        [ 0,   -q^6,   q^12]]
#
# and det M = q^24 (1 - q^6).  The right-hand side is a linear
# combination of the elementary traces {Tr(W_e), Tr(H_0), Tr(H_0^2),
# Tr(H_1^2)} with Laurent-polynomial-in-q coefficients (derived
# computationally from multiply_native + trace_of_seed).


_M3_MATRIX = (                                       # 3x3, Laurent-poly entries
    (LaurentPoly({4: 1}),  LaurentPoly.zero(),     LaurentPoly({14: -1})),
    (LaurentPoly({10: -1}), LaurentPoly({8: 1}),    LaurentPoly.zero()),
    (LaurentPoly.zero(),    LaurentPoly({6: -1}),   LaurentPoly({12: 1})),
)
_M3_TRIPLES = ((-1, 0, 1), (-1, 0, 3), (-1, 0, 5))


def _m3_rhs(traces_lp, q_max):
    """Compute the 3-vector of right-hand-side constants for the m=3
    cyclicity system, given elementary traces."""
    from pure_su2_h_multiply import multiply_native, _native_to_psu2

    def _trace_of_seed(m, e):
        if e % 2 == 1:
            return LaurentPoly.zero()
        if m == 0:
            return traces_lp[f'TrW_{abs(e)}']
        if m == 1:
            return traces_lp['TrH0']
        if m == 2:
            return traces_lp['TrH0sq'] if e % 4 == 0 else traces_lp['TrH1sq']
        raise ValueError(f"_m3_rhs: m={m} unsupported (m=3 is unknown)")

    def _three_letter(a, b, c):
        bc = multiply_native(((b, 1),), ((c, 1),)).terms
        out = {}
        for lbl, coef in bc.items():
            prod = multiply_native(((a, 1),), lbl).terms
            for k, v in prod.items():
                s = _native_to_psu2(k)
                out[s] = out.get(s, LaurentPoly.zero()) + coef * v
        return out

    def _eq_const(a, b, c):
        lhs = _three_letter(a, b, c)
        rhs = _three_letter(c - 8, a, b)
        # Move LHS to other side: const = Tr(RHS_non_m3) - Tr(LHS_non_m3).
        # (m=3 terms cancel: matrix is built separately.)
        const = LaurentPoly.zero()
        for seed, coef in rhs.items():
            m, e = seed
            if m == 3:
                continue
            const = const + coef * _trace_of_seed(m, e)
        for seed, coef in lhs.items():
            m, e = seed
            if m == 3:
                continue
            const = const - coef * _trace_of_seed(m, e)
        return _lp_truncate(const, q_max)

    return tuple(_eq_const(a, b, c) for (a, b, c) in _M3_TRIPLES)


def _det3(M):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


def _lp_series_inverse(D: LaurentPoly, q_max: int) -> LaurentPoly:
    """1/D as a truncated formal Laurent power series in q.

    Requires `D` to have a unique lowest-order term (`q^{k0} · c0` with
    `c0 != 0`); the result is `q^{-k0} · (1/c0) · sum_{n>=0} (-p)^n`
    where `p = (D - q^{k0} c0) / (q^{k0} c0)`.  Truncated to `q^{q_max}`.
    """
    coeffs = D._coeffs
    if not coeffs:
        raise ZeroDivisionError("_lp_series_inverse: zero")
    k0 = min(coeffs.keys())
    c0 = coeffs[k0]
    p = LaurentPoly({(e - k0): Fraction(c, c0) for e, c in coeffs.items() if e != k0})
    inner_qmax = q_max + k0
    inv = LaurentPoly({0: Fraction(1)})
    term = LaurentPoly({0: Fraction(1)})
    sign = 1
    while True:
        term = _lp_truncate(term * p, inner_qmax)
        if term.is_zero():
            break
        sign = -sign
        inv = _lp_add(inv, LaurentPoly({e: sign * c for e, c in term._coeffs.items()}))
    inv_scaled = {e - k0: Fraction(c, c0) for e, c in inv._coeffs.items()}
    return _lp_truncate(LaurentPoly(inv_scaled), q_max)


def solve_m3_anchors(traces_lp: dict, q_max: int = 30):
    """Compute the three m=3 H-shift anchors:

        (Tr(L_{(3, 0)}), Tr(L_{(3, 2)}), Tr(L_{(3, 4)}))

    `traces_lp` must contain:
      * `'TrH0'`, `'TrH0sq'`, `'TrH1sq'` (m<=2 anchors);
      * `'TrW_e'` for every `e` in 0, 2, ..., up to whatever the multiply
        expansion produces (defensively go up to ~16).

    All inputs should be `LaurentPoly` over Z (or Fraction).  Solves via
    Cramer + truncated power-series inversion of `det M = q^24 (1 - q^6)`.
    """
    M = [list(row) for row in _M3_MATRIX]
    b = _m3_rhs(traces_lp, q_max)
    det = _det3(M)
    inv_det = _lp_series_inverse(det, q_max)
    sols = []
    for i in range(3):
        Mi = [row[:] for row in M]
        for r in range(3):
            Mi[r][i] = b[r]
        num = _lp_truncate(_det3(Mi), q_max)
        sols.append(_lp_truncate(num * inv_det, q_max))
    return tuple(sols)


# ---------- Generic m-anchor cyclicity solver (all m >= 2) ----------
#
# m-anchor H-shift orbits: at level m, the canonical anchors are
# Tr(L_{(m, 2j)}) for j = 0, 1, ..., m-1 (H-shift sends e -> e + 2m;
# odd e vanish by Z2).
#
# Generic cyclicity system: take m m-tuples of the form
#   (0, 0, ..., 0, c)        with c in {0, 2, 4, ..., 2(m-1)}
# (prefix m-1 zeros, varying last letter).  The cyclicity identity
#   Tr(H_0^{m-1} * H_c) = Tr(H_{c-8} * H_0^{m-1})
# gives one linear equation per c in the m unknown anchors with
# coefficients pulled from `multiply_native` expansions of both sides
# (and the lower-m anchor values, recursively).


def _det_n(M):
    """Cofactor expansion (square matrix of `LaurentPoly` entries)."""
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    if n == 3:
        return _det3(M)
    out = LaurentPoly.zero()
    for j in range(n):
        minor = [row[:j] + row[j + 1:] for row in M[1:]]
        sign = 1 if j % 2 == 0 else -1
        out = _lp_add(out, _lp_scale_int(M[0][j] * _det_n(minor), sign))
    return out


def _m_letter_product(letters):
    """Compute the literal product `H_{l_1} · H_{l_2} · ... · H_{l_n}` as
    a dict {pSU2 seed (m, e): LP coef}.

    Uses `multiply_native` from right to left (cleanest associativity for
    the literal-word reducer).
    """
    from pure_su2_h_multiply import multiply_native, _native_to_psu2

    if len(letters) == 1:
        return {(1, letters[0]): LaurentPoly({0: 1})}
    inner_terms = _m_letter_product(letters[1:])
    out: dict = {}
    for seed_inner, coef in inner_terms.items():
        # Convert seed_inner back to native to feed multiply_native.
        from pure_su2_h_multiply import _psu2_to_native
        m_in, e_in = seed_inner
        native_in = _psu2_to_native(m_in, e_in)
        prod = multiply_native(((letters[0], 1),), native_in).terms
        for k, v in prod.items():
            s = _native_to_psu2(k)
            out[s] = out.get(s, LaurentPoly.zero()) + coef * v
    return out


_PREFIX_CANDIDATES_BASE = (
    # Each generator builds an `m-1` length prefix from the integer m.
    lambda m: list(range(-1, m - 2)),                  # (-1, 0, ..., m-3)
    lambda m: [-1] + list(range(0, m - 3)) + [m - 1],  # (-1, 0, ..., m-4, m-1)
    lambda m: [1] + [0] * (m - 2),                     # (1, 0, ..., 0)
    lambda m: [-2] + list(range(0, m - 2)),            # (-2, 0, 1, ..., m-3)
    lambda m: list(range(0, m - 1)),                   # (0, 1, ..., m-2)
    lambda m: [-1, 0] + list(range(2, m)),             # (-1, 0, 2, 3, ..., m-1)
    lambda m: [-3] + list(range(0, m - 2)),            # (-3, 0, 1, ..., m-3)
)


def _cyclicity_expansion(prefix, c):
    """Return (lhs_terms, rhs_terms) — pSU2 seed dicts for the LHS and
    RHS of the m-letter cyclicity at letters `prefix + [c]`."""
    return (
        _m_letter_product(prefix + [c]),
        _m_letter_product([c - 8] + prefix),
    )


def _build_M_for_prefix(m: int, prefix: list):
    """Construct the m x m coefficient matrix `M[i][j]` (Laurent poly
    entries) given a fixed prefix and varying last letter `c` indexed by
    anchor `e_anc = i`.

    Independent of lower-m elementary traces: the m=m component of the
    m-letter products is encoded fully by `multiply_native`.
    """
    anchors_e = list(range(0, 2 * m, 2))
    prefix_sum = sum(prefix)
    expansions = []                                  # cache for re-use in b
    M = [[LaurentPoly.zero()] * m for _ in range(m)]
    for i, e_anc in enumerate(anchors_e):
        c = (e_anc - prefix_sum) % (2 * m)
        lhs, rhs = _cyclicity_expansion(prefix, c)
        expansions.append((c, lhs, rhs))
        row = [LaurentPoly.zero()] * m
        for terms, sign in ((lhs, 1), (rhs, -1)):
            for seed, coef in terms.items():
                m_, e_ = seed
                if e_ % 2 == 1:
                    continue
                if m_ == m:
                    j = (e_ % (2 * m)) // 2
                    row[j] = _lp_add(row[j], _lp_scale_int(coef, sign))
        M[i] = row
    return M, expansions


def find_nonsingular_prefix(m: int):
    """Search the candidate prefix list, return `(prefix, M, expansions)`
    for the first prefix whose matrix is non-singular.  `expansions` is
    the cached list of `(c, lhs_terms, rhs_terms)` per anchor row (so the
    caller can build the b vector without re-running `multiply_native`).
    """
    for gen in _PREFIX_CANDIDATES_BASE:
        prefix = gen(m)
        M, expansions = _build_M_for_prefix(m, prefix)
        if not _det_n(M).is_zero():
            return prefix, M, expansions
    raise RuntimeError(
        f"find_nonsingular_prefix(m={m}): all candidate prefixes give "
        f"singular matrix.  Need to extend the candidate set."
    )


def solve_anchors_via_cyclicity(m: int, traces_lp: dict, q_max: int):
    """Generic m-anchor cyclicity solver for m >= 2.

    `traces_lp` must contain `'TrW_e'` (Wilson Schur traces) and
    `'TrL{m'}_{e_anchor}'` for every `m' < m` and every even
    `e_anchor in [0, 2m')`.  The `TrW_e` precision should exceed
    `q_max` by the matrix's leading-order shift (the function adds an
    internal buffer determined from `det M`).

    Returns dict `{e_anchor: LP}` for `e_anchor in {0, 2, ..., 2m-2}`.
    """
    anchors_e = list(range(0, 2 * m, 2))

    def _trace_of_seed(m_, e_):
        if e_ % 2 == 1:
            return LaurentPoly.zero()
        if m_ == 0:
            return traces_lp[f'TrW_{abs(e_)}']
        if m_ < m:
            e_anchor = e_ % (2 * m_)
            return traces_lp[f'TrL{m_}_{e_anchor}']
        raise ValueError(
            f"solve_anchors: unexpected m_={m_} >= target m={m}"
        )

    # Pick a non-singular prefix and grab the cached LHS/RHS expansions.
    prefix, M, expansions = find_nonsingular_prefix(m)

    # Build the RHS vector b from the lower-m traces (cyclicity equation
    # rearranges as `M · X = -const = b`).
    b = [LaurentPoly.zero()] * m
    for i, (c, lhs, rhs) in enumerate(expansions):
        const = LaurentPoly.zero()
        for terms, sign in ((lhs, 1), (rhs, -1)):
            for seed, coef in terms.items():
                m_, e_ = seed
                if e_ % 2 == 1:
                    continue
                if m_ < m:
                    t = _trace_of_seed(m_, e_)
                    const = _lp_add(const, _lp_scale_int(coef * t, sign))
        b[i] = LaurentPoly({e: -c_ for e, c_ in const._coeffs.items()})

    det = _det_n(M)
    inv_det = _lp_series_inverse(det, q_max)
    sols: dict = {}
    for j, e_anc in enumerate(anchors_e):
        Mj = [row[:] for row in M]
        for r in range(m):
            Mj[r][j] = b[r]
        num = _det_n(Mj)
        sols[e_anc] = _lp_truncate(num * inv_det, q_max)
    return sols


# ---------- LaurentPoly helpers ----------
def _lp_sub(a: LaurentPoly, b: LaurentPoly) -> LaurentPoly:
    out = dict(a._coeffs)
    for e, c in b._coeffs.items():
        out[e] = out.get(e, 0) - c
    return LaurentPoly({e: c for e, c in out.items() if c != 0})


def _lp_add(a: LaurentPoly, b: LaurentPoly) -> LaurentPoly:
    out = dict(a._coeffs)
    for e, c in b._coeffs.items():
        out[e] = out.get(e, 0) + c
    return LaurentPoly({e: c for e, c in out.items() if c != 0})


def _lp_shift(a: LaurentPoly, n: int) -> LaurentPoly:
    return LaurentPoly({e + n: c for e, c in a._coeffs.items()})


def _lp_neg(a: LaurentPoly) -> LaurentPoly:
    return LaurentPoly({e: -c for e, c in a._coeffs.items()})


def _lp_scale_int(a: LaurentPoly, n: int) -> LaurentPoly:
    return LaurentPoly({e: n * c for e, c in a._coeffs.items()})


def _lp_half(a: LaurentPoly) -> LaurentPoly:
    return LaurentPoly({e: Fraction(c) / 2 for e, c in a._coeffs.items()})


def _lp_truncate(a: LaurentPoly, q_max: int) -> LaurentPoly:
    return LaurentPoly({e: c for e, c in a._coeffs.items() if e <= q_max})
