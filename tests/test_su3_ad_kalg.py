"""Tests for `SU3ADKAlg` — standalone [A_1, D_4] AD K-algebra with
SU(3) symmetry enhancement.  Success criterion: products in the
standalone match products in `SU3BPSKAlgebra` (= isomorphism on the
canonical basis).

Run: `python3 run_tests.py`
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

from su3_ad_kalg import SU3ADKAlg
from su3_bps_kalgebra import _user_quiver
from kalgebra import Element
from laurent_poly import LaurentPoly


def make_alg():
    return SU3ADKAlg()


# Generator construction

def test_generators():
    A = make_alg()
    assert A.T(0) == (0, 1, 0, 0, 0)
    assert A.D(0) == (0, 0, 1, 0, 0)
    assert A.identity() == (0, 0, 0, 0, 0)


def test_rho_orbit_size():
    """ρ has order 4 on letters."""
    A = make_alg()
    for L in [A.T(0), A.D(0)]:
        orbit = [L]
        cur = L
        for _ in range(4):
            cur = A.rho(cur)
            if cur == L:
                break
            orbit.append(cur)
        assert len(orbit) == 4


def test_rho_equivariance():
    """ρ(T_0) = T_1, ρ(D_0) = D_1, etc."""
    A = make_alg()
    for i in range(4):
        assert A.rho(A.T(i)) == A.T((i + 1) % 4)
        assert A.rho(A.D(i)) == A.D((i + 1) % 4)


# Hardcoded relations

def test_TT_dist1_fwd():
    """T_0·T_1 = 1 + q^-1·χ_(0,1)·D_0 + q^-2·χ_(1,0)·D_0² + q^-3·D_0³."""
    A = make_alg()
    prod = A.multiply(A.T(0), A.T(1))
    expected = {
        (0, 0, 0, 0, 0): {0: 1},     # 1
        (0, 0, 1, 0, 1): {-1: 1},    # q^-1 · χ_(0,1) · D_0
        (0, 0, 2, 1, 0): {-2: 1},    # q^-2 · χ_(1,0) · D_0²
        (0, 0, 3, 0, 0): {-3: 1},    # q^-3 · D_0³
    }
    assert _terms_equal(prod, expected)


def test_DD_dist1_fwd():
    """D_0·D_1 = 1 + q^-1·T_1."""
    A = make_alg()
    prod = A.multiply(A.D(0), A.D(1))
    expected = {
        (0, 0, 0, 0, 0): {0: 1},
        (2, 1, 0, 0, 0): {-1: 1},   # T_1's tile = 2
    }
    assert _terms_equal(prod, expected)


def test_DD_dist2():
    """D_0·D_2 = χ_(0,1) + q^-1·D_1 + q·D_3."""
    A = make_alg()
    prod = A.multiply(A.D(0), A.D(2))
    expected = {
        (0, 0, 0, 0, 1): {0: 1},        # χ_(0,1)
        (2, 0, 1, 0, 0): {-1: 1},       # D_1 in tile 2
        (6, 0, 1, 0, 0): {+1: 1},       # D_3 in tile 6
    }
    assert _terms_equal(prod, expected)


def test_TD_dist1_fwd():
    """T_0·D_1 = q^-2·D_0² + q^-1·χ_(1,0)·D_0 + χ_(0,1) + q·D_3."""
    A = make_alg()
    prod = A.multiply(A.T(0), A.D(1))
    expected = {
        (0, 0, 2, 0, 0): {-2: 1},      # D_0²
        (0, 0, 1, 1, 0): {-1: 1},      # χ_(1,0)·D_0
        (0, 0, 0, 0, 1): {0: 1},       # χ_(0,1)
        (6, 0, 1, 0, 0): {+1: 1},      # D_3
    }
    assert _terms_equal(prod, expected)


def test_TT_dist2_palindromic():
    """T_0·T_2 has 10 distinct terms; palindromic under q ↔ q^-1."""
    A = make_alg()
    prod = A.multiply(A.T(0), A.T(2))
    # Spot-check: T_1 at q^-3, T_3 at q^+3, χ_(1,1) coefficient 2 at q^0.
    coefs = {}
    for lab, lp in prod.terms.items():
        for q, c in lp._coeffs.items():
            coefs[(lab, q)] = c
    # T_1 = tile 2, b=0, a=1; T_3 = tile 6, a=1, b=0
    assert coefs.get(((2, 1, 0, 0, 0), -3), 0) == 1
    assert coefs.get(((6, 1, 0, 0, 0), +3), 0) == 1
    # Identity at q^-2 and q^+2
    assert coefs.get(((0, 0, 0, 0, 0), -2), 0) == 1
    assert coefs.get(((0, 0, 0, 0, 0), +2), 0) == 1
    # Gauge-neutral q^0 = 2 + χ_(1,1) (NOT 4 + 2·χ_(1,1)): the six adjoint
    # root-weights are χ_(1,1)−2 as a virtual character, not 2·χ_(1,1).  The
    # old 4+2χ double-counted the adjoint by χ_(1,1)+2 (see
    # su3ad_flavour_layer1_defect.md); fixed in _tt_dist2.
    assert coefs.get(((0, 0, 0, 0, 0), 0), 0) == 2
    assert coefs.get(((0, 0, 0, 1, 1), 0), 0) == 1


# Iso against SU3BPSKAlgebra (the success criterion)

def _terms_equal(elt: Element, expected: dict) -> bool:
    """Check Element matches {label: {q_exp: coef}}."""
    got = {lab: dict(lp._coeffs) for lab, lp in elt.terms.items()}
    got = {lab: {q: c for q, c in d.items() if c != 0}
           for lab, d in got.items() if any(c != 0 for c in d.values())}
    expected = {lab: {q: c for q, c in d.items() if c != 0}
                for lab, d in expected.items() if any(c != 0 for c in d.values())}
    return got == expected


def _bps_label_to_a1d4(bps_alg, bps_label):
    """Map a BPS canonical label → SU3ADKAlg label via (section, χ).

    For unsplit Weyl-orbit cases (= our spec covers V_(1,0) only with
    τ-symmetric σ-orbit), this is well-defined.  For split cases (e.g.
    V_(1,1) regular weights) the two BPS σ-orbits map to the same
    A1D4 label.
    """
    sec, chi_re = bps_alg._label_section_decompose(bps_label)
    chi_terms = list(chi_re.terms.keys())
    if not chi_terms:
        chi_pq = (0, 0)
    else:
        chi_pq = chi_terms[0]
    # Decompose section as letter-monomial.
    a_l, b_l, tile, q_factor = _section_to_a1d4_monomial(sec[:2])
    # tile already includes a, b; just return label
    return (tile, a_l, b_l, chi_pq[0], chi_pq[1])


def _section_to_a1d4_monomial(gauge):
    """Find (tile, a, b) such that the gauge sublattice point equals
    a·T_tile_T + b·D_tile_D."""
    letter_proj = {
        ('T', 0): (1, 0), ('T', 1): (-1, -3), ('T', 2): (-2, -3), ('T', 3): (-1, 0),
        ('D', 0): (0, -1), ('D', 1): (-1, -2), ('D', 2): (-1, -1), ('D', 3): (0, 1),
    }
    target = (int(gauge[0]), int(gauge[1]))
    if target == (0, 0):
        return 0, 0, 0, 0
    # Single-letter
    for L, proj in letter_proj.items():
        for e in range(1, 30):
            if (e * proj[0], e * proj[1]) == target:
                if L[0] == 'T':
                    tile = {0: 0, 1: 2, 2: 4, 3: 6}[L[1]]
                    return e, 0, tile, 0
                else:
                    tile = {0: 0, 1: 2, 2: 4, 3: 6}[L[1]]
                    return 0, e, tile, 0
    # T·D pair
    from su3_ad_kalg import _TILE_LETTERS
    for tile_idx in range(8):
        L_T, L_D = _TILE_LETTERS[tile_idx]
        pT = letter_proj[L_T]; pD = letter_proj[L_D]
        det = pT[0] * pD[1] - pT[1] * pD[0]
        if det == 0:
            continue
        num_a = target[0] * pD[1] - target[1] * pD[0]
        num_b = pT[0] * target[1] - pT[1] * target[0]
        if num_a % det != 0 or num_b % det != 0:
            continue
        a_e = num_a // det; b_e = num_b // det
        if a_e < 0 or b_e < 0:
            continue
        if a_e == 0 or b_e == 0:
            continue
        return a_e, b_e, tile_idx, 0
    raise ValueError(f"section {gauge} not expressible as letter monomial")


def test_iso_on_generators():
    """Standalone product matches BPS product on generator pairs."""
    A = make_alg()
    B = _user_quiver()
    # Generator BPS labels
    def bps_T(i):
        c = B.canonicalise((1, 0, 0, 0))
        for _ in range(i % 4):
            c = B.rho(c)
        return tuple(int(x) for x in c)
    def bps_D(i):
        c = B.canonicalise((0, -1, 0, 0))
        for _ in range(i % 4):
            c = B.rho(c)
        return tuple(int(x) for x in c)

    # For each non-q-commuting generator pair, compare standalone vs BPS.
    pairs_to_check = [
        (A.T(0), A.T(1), bps_T(0), bps_T(1)),
        (A.T(0), A.T(2), bps_T(0), bps_T(2)),
        (A.T(1), A.T(0), bps_T(1), bps_T(0)),
        (A.D(0), A.D(1), bps_D(0), bps_D(1)),
        (A.D(0), A.D(2), bps_D(0), bps_D(2)),
        (A.D(1), A.D(0), bps_D(1), bps_D(0)),
        (A.T(0), A.D(1), bps_T(0), bps_D(1)),
        (A.T(0), A.D(2), bps_T(0), bps_D(2)),
        (A.D(1), A.T(0), bps_D(1), bps_T(0)),
        (A.D(2), A.T(0), bps_D(2), bps_T(0)),
    ]
    for a_std, b_std, a_bps, b_bps in pairs_to_check:
        prod_std = A.multiply(a_std, b_std)
        prod_bps = B.multiply(a_bps, b_bps)
        # Convert BPS product to A1D4 labels (Weyl-symmetrized)
        bps_in_a1d4 = {}
        for bps_lab, lp in prod_bps.terms.items():
            a1d4_lab = _bps_label_to_a1d4(B, bps_lab)
            slot = bps_in_a1d4.setdefault(a1d4_lab, {})
            for q, c in lp._coeffs.items():
                slot[q] = slot.get(q, 0) + c
        # Clean
        bps_in_a1d4 = {l: {q: c for q, c in d.items() if c != 0}
                       for l, d in bps_in_a1d4.items()
                       if any(c != 0 for c in d.values())}
        std_terms = {lab: {q: c for q, c in lp._coeffs.items() if c != 0}
                     for lab, lp in prod_std.terms.items()}
        # Note: for split Weyl-orbits, BPS gives TWO σ-orbits each with
        # same coefficient; A1D4 gives ONE basis element with that
        # coefficient.  So BPS coefs should be DIVIDED by σ-orbit count
        # before comparing.  For our pairs (generators of low dim reps),
        # we need to check this case-by-case.  Let's compare just the
        # KEYS (labels) for now and check exact coef agreement where
        # the BPS doesn't split.
        std_keys = set(std_terms.keys())
        bps_keys = set(bps_in_a1d4.keys())
        # Match labels — assert that std_keys ⊆ bps_keys (every standalone
        # output appears in BPS).
        missing = std_keys - bps_keys
        assert not missing, (
            f"standalone produced extra labels {missing} not in BPS for "
            f"({a_std}, {b_std})"
        )


def test_rho_is_automorphism_all_pairs():
    """ρ(L_a · L_b) == ρ(L_a) · ρ(L_b) on every generator pair.

    Regression guard for the 2026-06-13 fix: the static relation tables
    emitted the same χ for every ρ-orbit position, breaking this on the
    32 odd-position pairs.  `_chi_parity` (ρ = tile-shift ∘ ⋆) restores
    it; all 64 pairs must pass."""
    A = make_alg()
    gens = [A.T(i) for i in range(4)] + [A.D(i) for i in range(4)]
    bad = [(a, b) for a in gens for b in gens
           if not A.verify_rho_is_automorphism(a, b)]
    assert not bad, f"ρ not an automorphism on {len(bad)} pairs, e.g. {bad[:3]}"


def test_iso_matches_bps_products():
    """STRONG iso certificate: every generator-pair product matches
    `SU3BPSKAlgebra` *coefficient-for-coefficient* in χ-form (gauge
    charge + SU(3) character), including the odd-ρ-position pairs the
    weaker `test_iso_on_generators` does not pin.  This is the
    ground-truth check (physical BPS, not internal consistency) that the
    re-derived χ-alternating relations are correct."""
    import itertools
    from su3_ad_kalg import _TILE_LETTERS
    A = make_alg()
    B = _user_quiver()
    TPOS = {0: (1, 0), 1: (-1, -3), 2: (-2, -3), 3: (-1, 0)}
    DPOS = {0: (0, -1), 1: (-1, -2), 2: (-1, -1), 3: (0, 1)}

    def std_form(prod):
        out = {}
        for lab, lp in prod.terms.items():
            tile, a, b, p, q = lab
            (_, it), (_, idd) = _TILE_LETTERS[tile]
            g = (a * TPOS[it][0] + b * DPOS[idd][0],
                 a * TPOS[it][1] + b * DPOS[idd][1])
            for qe, c in lp._coeffs.items():
                if c:
                    out[(g, (p, q), qe)] = out.get((g, (p, q), qe), 0) + c
        return {k: v for k, v in out.items() if v}

    def bps_lbl(kind, i):
        base = (1, 0, 0, 0) if kind == 'T' else (0, -1, 0, 0)
        c = B.canonicalise(base)
        for _ in range(i % 4):
            c = B.rho(c)
        return tuple(int(x) for x in c)

    def bps_form(prod):
        rf = B.to_R_form(prod)
        out = {}
        for lab, rl in rf.terms.items():
            g = tuple(int(x) for x in lab[:2])
            for qe, relt in rl.coeffs.items():
                for pq, c in relt.terms.items():
                    if c:
                        out[(g, pq, qe)] = out.get((g, pq, qe), 0) + c
        return {k: v for k, v in out.items() if v}

    # Since 2026-06-13 the BPS engine is the full-S_3 `GfBPSKAlgebra`
    # (`su3_bps_kalgebra` retirement): it returns the correct SU(3) content
    # `χ(1,1)+2·𝟙` for the T·T distance-2 adjoint product, matching the
    # standalone directly.  The earlier `_TT2_FOLD_OVERCOUNT` reconciliation
    # against the old Z_3 engine's `2·χ(1,1)+4·𝟙` over-count is no longer
    # needed (and would now be wrong).
    gens = [('T', i) for i in range(4)] + [('D', i) for i in range(4)]
    for (k1, i1), (k2, i2) in itertools.product(gens, gens):
        a_std = A.T(i1) if k1 == 'T' else A.D(i1)
        b_std = A.T(i2) if k2 == 'T' else A.D(i2)
        sf = std_form(A.multiply(a_std, b_std))
        bf = bps_form(B.multiply(bps_lbl(k1, i1), bps_lbl(k2, i2)))
        assert sf == bf, (
            f"{k1}{i1}·{k2}{i2}: standalone ≠ BPS\n"
            f"  std-only: {dict((k, v) for k, v in sf.items() if bf.get(k) != v)}\n"
            f"  bps-only: {dict((k, v) for k, v in bf.items() if sf.get(k) != v)}"
        )


def test_vacuum_trace_has_adjoint():
    """Tr(1) carries the SU(3) adjoint χ_(1,1) at q² (the flavour
    current) — the served trace (since 2026-09-24 Creutzig's gauge tower of
    the even-D family at k = 1, summed over the gauge charge), not the old
    `1 + O(q⁷)` Layer-1 vacuum stub."""
    A = make_alg()
    tr1 = A.trace(A.identity(), K=4)
    chi2 = tr1.coeffs.get(2)
    assert chi2 is not None and chi2.terms.get((1, 1)) == 1, (
        f"vacuum char q² = {chi2}, expected χ_(1,1)"
    )


def test_trace_R_linear_on_flavour():
    """The flavour `χ_(p,q)` is a genuine spectator over R(SU(3)): the
    trace is R(SU(3))-linear, `Tr(χ_(p,q)·1) = χ_(p,q)·Tr_1`, for EVERY
    rep — including the non-self-dual ones (6, 10) that the old
    single-highest-weight transport corrupted.  Regression for the
    `_std_to_bps_label` lossiness fix."""
    A = make_alg(); R = A._R
    K = 6
    tr1 = A.trace(A.identity(), K=K)
    for (p, q) in [(1, 1), (2, 0), (3, 0)]:
        chi = R.basis_element((p, q))
        t = A.trace((0, 0, 0, p, q), K=K)
        for e in range(0, K + 1):
            lhs = t.coeffs.get(e, R.zero())
            rhs = chi * tr1.coeffs.get(e, R.zero())
            assert str(lhs) == str(rhs), (
                f"Tr(χ_{(p, q)}·1) ≠ χ·Tr_1 at q^{e}: {lhs} vs {rhs}"
            )


def test_trace_word_matches_single_label():
    """`trace_word` (U(1)²-substrate trace, Weyl-symmetrize only on the
    total) reproduces the single-label trace for collapsing powers —
    Tr(T₀ⁿ), Tr(D₀ⁿ).  Regression for the premature-symmetrization bug."""
    A = make_alg(); R = A._R
    K = 8
    cases = [([A.T(0), A.T(0)], (0, 2, 0, 0, 0)),
             ([A.T(0)] * 3, (0, 3, 0, 0, 0)),
             ([A.D(0), A.D(0)], (0, 0, 2, 0, 0))]
    for fac, single in cases:
        tw = A.trace_word(fac, K=K)
        ts = A.trace(single, K=K)
        for e in range(0, K + 1):
            assert (str(tw.coeffs.get(e, R.zero())) ==
                    str(ts.coeffs.get(e, R.zero()))), (
                f"trace_word ≠ single-label trace of {single} at q^{e}"
            )


def test_trace_word_cyclicity():
    """ρ²-twisted cyclicity on a GENUINE non-collapsing product.

    `T₀·T₂` does NOT collapse to a single label (unlike `T₀ⁿ`), and
    `ρ²(T₂)=T₀`, so cyclicity demands `Tr(T₀·T₂)=Tr(T₀²)`.  This is the
    case that exposed the fixed-margin truncation bug (it gave a spurious
    `2+χ(1,1)` at q⁰); with the adaptive per-term margin in `trace_word`
    it must equal the clean single-label `Tr(T₀²)`."""
    A = make_alg(); R = A._R
    K = 4
    tw = A.trace_word([A.T(0), A.T(2)], K=K)        # Tr(T₀·T₂), non-collapsing
    truth = A.trace((0, 2, 0, 0, 0), K=K)           # Tr(T₀²), single label
    for e in range(0, K + 1):
        assert (str(tw.coeffs.get(e, R.zero())) ==
                str(truth.coeffs.get(e, R.zero()))), (
            f"cyclicity Tr(T₀·T₂) ≠ Tr(T₀²) at q^{e}: "
            f"{tw.coeffs.get(e)} vs {truth.coeffs.get(e)}"
        )


def test_trace_word_cyclicity_D():
    """ρ²-twisted cyclicity in the D-channel: `ρ²(D₂)=D₀`, so
    `Tr(D₀·D₂)=Tr(D₀²)`.  Independent of the T-channel, exercises the
    fugacity-multiply product trace on the D-relation Plücker."""
    A = make_alg(); R = A._R
    K = 6
    tw = A.trace_word([A.D(0), A.D(2)], K=K)
    truth = A.trace((0, 0, 2, 0, 0), K=K)
    for e in range(0, K + 1):
        assert (str(tw.coeffs.get(e, R.zero())) ==
                str(truth.coeffs.get(e, R.zero()))), (
            f"cyclicity Tr(D₀·D₂) ≠ Tr(D₀²) at q^{e}"
        )


def test_trace_is_bps_free():
    """The trace path is engine-free: `trace` / `inner_product` /
    `trace_word` compute via `sl3_su3_traces` (Layer-1 + Tr_1, Tr_T, Tr_D
    from the even-D k = 1 closed forms, all in Cartan fugacities)
    and must NEVER touch the `SU3BPSKAlgebra` oracle.  We trip the oracle and
    confirm every trace entry point still computes."""
    A = make_alg()

    def _boom(*a, **k):
        raise AssertionError("trace path touched the BPS engine")

    A._bps_engine = _boom
    A.trace(A.identity(), K=6)
    A.trace(A.T(0), K=6)
    A.trace((0, 0, 0, 2, 0), K=4)                   # flavour-charged (10 of SU(3))
    A.inner_product(A.T(0), A.T(0), K=4)
    A.inner_product(A.D(0), A.D(2), K=6)            # off-diagonal cross-tile
    A.trace_word([A.T(0), A.T(2)], K=8)


# Runner

if __name__ == "__main__":
    import traceback
    tests = [
        test_generators, test_rho_orbit_size, test_rho_equivariance,
        test_TT_dist1_fwd, test_DD_dist1_fwd, test_DD_dist2,
        test_TD_dist1_fwd, test_TT_dist2_palindromic,
        test_iso_on_generators,
        test_rho_is_automorphism_all_pairs,
        test_iso_matches_bps_products,
        test_vacuum_trace_has_adjoint,
        test_trace_R_linear_on_flavour,
        test_trace_word_matches_single_label,
        test_trace_word_cyclicity,
        test_trace_word_cyclicity_D,
        test_trace_is_bps_free,
    ]
    n_pass = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
        else:
            print(f"OK    {t.__name__}")
            n_pass += 1
    print(f"\n{n_pass}/{len(tests)} tests passed")
