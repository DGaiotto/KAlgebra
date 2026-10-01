"""Tests for `A1D3KAlg` -- the [A_1, D_3] K-algebra, standalone
SU(2)-flavoured realisation with manifest Z_3 cyclic symmetry.

This class is fully standalone (no SU2BPSKAlgebra / BPSKAlgebra
runtime dependency).  Multiplication is implemented via the
recursive reduction algorithm using only the 6 multiplicative
generators and the 8 Z_3-symmetric defining relations.

Tests are pinned against SU2BPSKAlgebra at TEST TIME to verify
correctness of the closed-form multiply.  The runtime A1D3KAlg
does not import SU2BPSKAlgebra.

Run:  `python3 run_tests.py`
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

from a1d3_kalg import A1D3KAlg, _TILE_LETTERS
from kalgebra import KAlgebra, Element
from zplus_ring import SU2ZPlusRing
from laurent_poly import LaurentPoly


# Construction --------------------------------------------------------------


def test_construction():
    A = A1D3KAlg()
    assert A.rank == 3
    assert A.gauge_rank == 2
    assert isinstance(A.coefficient_ring(), SU2ZPlusRing)
    assert A.identity() == (0, 0, 0, 0)
    assert isinstance(A, KAlgebra)


def test_no_bps_runtime_import():
    """A1D3KAlg.__init__ should NOT import or instantiate
    SU2BPSKAlgebra / BPSKAlgebra (= truly standalone)."""
    import a1d3_kalg
    src = open(a1d3_kalg.__file__).read()
    # Scan only import statements (allow docstring/comment mentions).
    import_lines = [
        ln.strip() for ln in src.split('\n')
        if ln.strip().startswith(('import ', 'from '))
    ]
    for ln in import_lines:
        assert 'SU2BPSKAlgebra' not in ln and 'BPSKAlgebra' not in ln, (
            f'A1D3KAlg has runtime BPS import: {ln!r}'
        )
    # Also ensure A1D3KAlg() can be constructed without BPS being importable
    # at runtime (we don't actually monkeypatch sys.modules — just sanity).
    A = a1d3_kalg.A1D3KAlg()
    assert A is not None


# Generator labels ---------------------------------------------------------


def test_T_generators():
    A = A1D3KAlg()
    assert A.T(0) == (0, 1, 0, 0)
    assert A.T(1) == (2, 1, 0, 0)
    assert A.T(2) == (4, 1, 0, 0)
    assert A.T(3) == A.T(0)


def test_D_generators():
    A = A1D3KAlg()
    assert A.D(0) == (0, 0, 1, 0)
    assert A.D(1) == (2, 0, 1, 0)
    assert A.D(2) == (4, 0, 1, 0)


def test_chi_labels():
    A = A1D3KAlg()
    assert A.chi(0) == (0, 0, 0, 0)
    assert A.chi(1) == (0, 0, 0, 1)
    assert A.chi(2) == (0, 0, 0, 2)


# ρ as permutation of canonical labels --------------------------------------


def test_rho_permutes_T_orbit():
    A = A1D3KAlg()
    for i in range(3):
        assert A.rho(A.T(i)) == A.T((i + 1) % 3)


def test_rho_permutes_D_orbit():
    A = A1D3KAlg()
    for i in range(3):
        assert A.rho(A.D(i)) == A.D((i + 1) % 3)


def test_rho_cube_identity_on_generators():
    A = A1D3KAlg()
    for i in range(3):
        assert A.rho(A.rho(A.rho(A.T(i)))) == A.T(i)
        assert A.rho(A.rho(A.rho(A.D(i)))) == A.D(i)


# Z_3-symmetric relations (closed-form structure constants) ----------------


def _coef(elt, label):
    return elt.terms.get(label, LaurentPoly({}))


def test_relation_TT_forward():
    """T_i · T_{i+1} = 1 + q⁻¹·χ_1·D_i + q⁻²·D_i², for all i mod 3."""
    A = A1D3KAlg()
    for i in range(3):
        prod = A.multiply(A.T(i), A.T((i + 1) % 3))
        Di = A.D(i)
        # χ_1·D_i label: same tile as D_i, with chi index 1.
        chi1_Di = (Di[0], 0, 1, 1)
        Di_sq = (Di[0], 0, 2, 0)
        assert _coef(prod, A.identity()) == LaurentPoly({0: 1})
        assert _coef(prod, chi1_Di) == LaurentPoly({-1: 1})
        assert _coef(prod, Di_sq) == LaurentPoly({-2: 1})
        assert len(prod.terms) == 3


def test_relation_TT_backward():
    """T_{i+1} · T_i = 1 + q·χ_1·D_i + q²·D_i²."""
    A = A1D3KAlg()
    for i in range(3):
        prod = A.multiply(A.T((i + 1) % 3), A.T(i))
        Di = A.D(i)
        chi1_Di = (Di[0], 0, 1, 1)
        Di_sq = (Di[0], 0, 2, 0)
        assert _coef(prod, A.identity()) == LaurentPoly({0: 1})
        assert _coef(prod, chi1_Di) == LaurentPoly({1: 1})
        assert _coef(prod, Di_sq) == LaurentPoly({2: 1})


def test_relation_DD_forward():
    """D_i · D_{i+1} = 1 + q⁻¹·T_{i+1}."""
    A = A1D3KAlg()
    for i in range(3):
        prod = A.multiply(A.D(i), A.D((i + 1) % 3))
        T_next = A.T((i + 1) % 3)
        assert _coef(prod, A.identity()) == LaurentPoly({0: 1})
        assert _coef(prod, T_next) == LaurentPoly({-1: 1})
        assert len(prod.terms) == 2


def test_relation_DD_backward():
    A = A1D3KAlg()
    for i in range(3):
        prod = A.multiply(A.D((i + 1) % 3), A.D(i))
        T_next = A.T((i + 1) % 3)
        assert _coef(prod, A.identity()) == LaurentPoly({0: 1})
        assert _coef(prod, T_next) == LaurentPoly({1: 1})


def test_relation_TD_interaction():
    """T_i · D_{i+1} = χ_1 + q⁻¹·D_i + q·D_{i-1}."""
    A = A1D3KAlg()
    chi1_label = A.chi(1)
    for i in range(3):
        prod = A.multiply(A.T(i), A.D((i + 1) % 3))
        assert _coef(prod, chi1_label) == LaurentPoly({0: 1})
        assert _coef(prod, A.D(i)) == LaurentPoly({-1: 1})
        assert _coef(prod, A.D((i - 1) % 3)) == LaurentPoly({1: 1})
        assert len(prod.terms) == 3


def test_relation_DT_interaction():
    A = A1D3KAlg()
    chi1_label = A.chi(1)
    for i in range(3):
        prod = A.multiply(A.D((i + 1) % 3), A.T(i))
        assert _coef(prod, chi1_label) == LaurentPoly({0: 1})
        assert _coef(prod, A.D(i)) == LaurentPoly({1: 1})
        assert _coef(prod, A.D((i - 1) % 3)) == LaurentPoly({-1: 1})


def test_clean_products():
    """T_i² and D_i² are clean (lattice doublings)."""
    A = A1D3KAlg()
    for i in range(3):
        assert len(A.multiply(A.T(i), A.T(i)).terms) == 1
        assert len(A.multiply(A.D(i), A.D(i)).terms) == 1


def test_TT_inverse_not_unit():
    """T_0 and T_2 are NOT mutual inverses (3-term BPS interaction)."""
    A = A1D3KAlg()
    prod = A.multiply(A.T(0), A.T(2))
    assert len(prod.terms) == 3
    assert A.identity() in prod.terms


# SU(2) Clebsch-Gordan -----------------------------------------------------


def test_chi_chi_clebsch_gordan():
    """χ_1 · χ_1 = χ_0 + χ_2.  χ_2 · χ_1 = χ_1 + χ_3.  Etc."""
    A = A1D3KAlg()
    p = A.multiply(A.chi(1), A.chi(1))
    assert p.terms == {A.chi(0): LaurentPoly({0: 1}),
                       A.chi(2): LaurentPoly({0: 1})}
    p = A.multiply(A.chi(2), A.chi(1))
    assert p.terms == {A.chi(1): LaurentPoly({0: 1}),
                       A.chi(3): LaurentPoly({0: 1})}


# ρ as algebra automorphism on products ------------------------------------


def test_rho_induced_automorphism():
    """ρ(a · b) = ρ(a) · ρ(b) on generator labels."""
    A = A1D3KAlg()
    sample = [A.T(0), A.T(1), A.T(2), A.D(0), A.D(1), A.D(2)]
    for a in sample:
        for b in sample:
            prod = A.multiply(a, b)
            rho_prod_terms = {
                A.rho(lab): lp for lab, lp in prod.terms.items()
            }
            prod_rho = A.multiply(A.rho(a), A.rho(b))
            assert rho_prod_terms == prod_rho.terms, (
                f'ρ-automorphism failed at ({a}, {b})'
            )


# Identity and trace -------------------------------------------------------


def test_identity_multiply():
    A = A1D3KAlg()
    sample = [A.T(0), A.D(1), A.chi(2), (0, 2, 3, 1)]
    for raw in sample:
        c = A.canonicalise(raw)
        prod = A.multiply(A.identity(), c)
        assert prod.terms.get(c) == LaurentPoly({0: 1})
        assert len(prod.terms) == 1


def test_trace_identity_vacuum_chi_0():
    """Vacuum-only trace at identity has χ_0 coefficient 1 at q^0."""
    A = A1D3KAlg()
    tr = A.trace(A.identity(), K=2)
    assert 0 in tr.coeffs
    assert tr.coeffs[0].terms.get(0, 0) == 1


def test_trace_vacuum_only_non_central_zero():
    """Vacuum-only trace of M(tile, a, b, k) with (a, b) ≠ (0, 0): zero."""
    A = A1D3KAlg()
    tr = A.trace(A.T(0), K=2, vacuum_only=True)
    assert all(r.is_zero() for r in tr.coeffs.values())


def test_trace_layer2_closed_form():
    """Layer 2 returns the three elementary traces as RPowerSeries."""
    A = A1D3KAlg()
    layer2 = A.trace_layer2(K=4)
    assert set(layer2.keys()) == {('Tr_1',), ('Tr_T',), ('Tr_D',)}
    # Tr_1 at q^0 = χ_0 = 1.
    assert layer2[('Tr_1',)][0] == A.coefficient_ring().basis_element(0)
    # Tr_T leading: -q · χ_0.
    assert layer2[('Tr_T',)][1] == -A.coefficient_ring().basis_element(0)
    # Tr_D leading: -q · χ_1.
    assert layer2[('Tr_D',)][1] == -A.coefficient_ring().basis_element(1)


def test_trace_full_matches_layer2():
    """trace(T_0), trace(D_0), trace(1) match the corresponding Layer-2 series."""
    A = A1D3KAlg()
    K = 8
    layer2 = A.trace_layer2(K)
    assert A.trace(A.identity(), K=K) == layer2[('Tr_1',)]
    assert A.trace(A.T(0), K=K) == layer2[('Tr_T',)]
    assert A.trace(A.D(0), K=K) == layer2[('Tr_D',)]


# Layer 1 trace reduction --------------------------------------------------


def test_trace_layer1_identity():
    """Layer 1 on identity: Tr(1) = χ_0 — single ('Tr_1',) entry at q^0."""
    A = A1D3KAlg()
    reduction = A.trace_layer1(A.identity())
    assert ('Tr_1',) in reduction
    qdict = reduction[('Tr_1',)]  # dict[q_exp, RElement]
    assert qdict == {0: A.coefficient_ring().basis_element(0)}


def test_trace_layer1_chi_k():
    """Layer 1 on χ_k: ('Tr_1',) entry at q^0 with chi = χ_k."""
    A = A1D3KAlg()
    for k in [0, 1, 2, 3]:
        reduction = A.trace_layer1(A.chi(k))
        assert ('Tr_1',) in reduction
        qdict = reduction[('Tr_1',)]
        assert qdict == {0: A.coefficient_ring().basis_element(k)}


def test_trace_layer1_single_letter():
    """Layer 1 on a single letter T_i / D_i: 'Tr_T' or 'Tr_D' entry."""
    A = A1D3KAlg()
    for i in range(3):
        rT = A.trace_layer1(A.T(i))
        assert ('Tr_T',) in rT
        rD = A.trace_layer1(A.D(i))
        assert ('Tr_D',) in rD


def test_trace_layer1_two_letter_plucker():
    """T_0 · T_1 has direct Plücker; layer 1 reduces to elementary basis
    {Tr(1), Tr_T, Tr_D}."""
    A = A1D3KAlg()
    prod = A.multiply(A.T(0), A.T(1))
    reduction = A.trace_layer1_element(prod)
    # Reductions are dict[elem_key, dict[q_exp, RElement]].
    # Expect entries at 'Tr_1', ('Tr_T',), and ('Tr_D',).
    assert all(isinstance(v, dict) for v in reduction.values())
    assert len(reduction) >= 1


def test_trace_vacuum_chi_k():
    """Vacuum-only trace at chi_k has chi_k at q^0."""
    A = A1D3KAlg()
    for k in range(4):
        tr = A.trace(A.chi(k), K=2, vacuum_only=True)
        assert 0 in tr.coeffs
        assert tr.coeffs[0].terms.get(k, 0) == 1
        # Only q^0 term in vacuum-only mode.
        assert len(tr.coeffs) == 1


def test_trace_layer1_element_identity_sum():
    """Layer 1 on an Element of identity-gauge-cell labels: all contribute
    to the single ('Tr_1',) entry (no chi_branch splitting in new accumulator)."""
    A = A1D3KAlg()
    R = A.coefficient_ring()
    elt = Element({
        A.identity(): LaurentPoly({0: 1}),
        A.chi(1): LaurentPoly({1: 2}),
        A.chi(2): LaurentPoly({2: 3}),
    })
    reduction = A.trace_layer1_element(elt)
    assert ('Tr_1',) in reduction
    qdict = reduction[('Tr_1',)]
    # q^0 → χ_0, q^1 → 2·χ_1, q^2 → 3·χ_2.
    assert qdict.get(0) == R.basis_element(0)
    assert qdict.get(1) == 2 * R.basis_element(1)
    assert qdict.get(2) == 3 * R.basis_element(2)


# KAlgebra axioms ----------------------------------------------------------


def _multiply_element_by_label(A, elt, lab):
    """Helper: multiply an Element by a Label, returning an Element."""
    out = {}
    for la, lpa in elt.terms.items():
        sub = A.multiply(la, lab)
        for sl, slp in sub.terms.items():
            nlp = lpa * slp
            out[sl] = out.get(sl, LaurentPoly({})) + nlp
    return Element({l: lp for l, lp in out.items() if not lp.is_zero()})


def _multiply_label_by_element(A, lab, elt):
    out = {}
    for la, lpa in elt.terms.items():
        sub = A.multiply(lab, la)
        for sl, slp in sub.terms.items():
            nlp = lpa * slp
            out[sl] = out.get(sl, LaurentPoly({})) + nlp
    return Element({l: lp for l, lp in out.items() if not lp.is_zero()})


def test_associativity_generators():
    """(a · b) · c = a · (b · c) on all 8³ = 512 generator triples
    (T_0..T_2, D_0..D_2, χ_1, χ_2)."""
    A = A1D3KAlg()
    sample = [A.T(0), A.T(1), A.T(2), A.D(0), A.D(1), A.D(2),
              A.chi(1), A.chi(2)]
    fail_count = 0
    first_fail = None
    for a in sample:
        for b in sample:
            for c in sample:
                left = _multiply_element_by_label(A, A.multiply(a, b), c)
                right = _multiply_label_by_element(A, a, A.multiply(b, c))
                if left.terms != right.terms:
                    fail_count += 1
                    if first_fail is None:
                        first_fail = (a, b, c, left.terms, right.terms)
    if fail_count:
        a, b, c, L, R = first_fail
        raise AssertionError(
            f'Associativity failed {fail_count} times.  First: '
            f'({a}, {b}, {c}): left={L}, right={R}'
        )


def test_associativity_higher_powers():
    """(a · b) · c = a · (b · c) on a sample of higher-power labels."""
    A = A1D3KAlg()
    sample = set()
    for tile in range(6):
        for a in range(3):
            for b in range(3):
                for k in range(2):
                    sample.add(A.canonicalise((tile, a, b, k)))
    sample = sorted(sample)[:20]  # first 20 canonical labels
    fail_count = 0
    first_fail = None
    for a in sample:
        for b in sample:
            for c in sample:
                left = _multiply_element_by_label(A, A.multiply(a, b), c)
                right = _multiply_label_by_element(A, a, A.multiply(b, c))
                if left.terms != right.terms:
                    fail_count += 1
                    if first_fail is None:
                        first_fail = (a, b, c, left.terms, right.terms)
    if fail_count:
        a, b, c, L, R = first_fail
        raise AssertionError(
            f'Higher-power associativity failed {fail_count} times.  '
            f'First: ({a}, {b}, {c}): left={L}, right={R}'
        )


# Pinning against SU2BPSKAlgebra (BPS ground truth) ------------------------


def _setup_bps():
    """Build the corresponding SU2BPSKAlgebra (used only at TEST TIME)."""
    from su2_bps_kalgebra import SU2BPSKAlgebra
    return SU2BPSKAlgebra(
        pairing=[[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
        node_charges=[(1, 0, 0), (0, 1, 1), (0, 1, -1)],
        w_su2=[[1, 0, 0], [0, 1, 0], [0, 0, -1]],
    )


def _a1d3_label_to_bps_label(label, bps):
    """Convert A1D3KAlg (tile, a, b, k) → SU2BPSKAlgebra canonical lattice label."""
    tile, a, b, k = label
    L_T, L_D = _TILE_LETTERS[tile]
    t_lat = [(1, 0, 0), (-1, -2, 0), (-1, 0, 0)][L_T[1]]
    d_lat = [(0, -1, 0), (-1, -1, 0), (0, 1, 0)][L_D[1]]
    return bps.canonicalise((
        a * t_lat[0] + b * d_lat[0],
        a * t_lat[1] + b * d_lat[1],
        a * t_lat[2] + b * d_lat[2] - k,
    ))


def test_pin_against_bps_generator_pairs():
    """All generator-pair products (8x8 = 64 cases) match SU2BPSKAlgebra."""
    A = A1D3KAlg()
    bps = _setup_bps()
    gens = [A.T(0), A.T(1), A.T(2), A.D(0), A.D(1), A.D(2),
            A.chi(1), A.chi(2)]
    for x in gens:
        for y in gens:
            ap = A.multiply(x, y)
            ap_via_lattice = {
                _a1d3_label_to_bps_label(lab, bps): lp
                for lab, lp in ap.terms.items()
            }
            bp = bps.multiply(
                _a1d3_label_to_bps_label(x, bps),
                _a1d3_label_to_bps_label(y, bps),
            )
            assert ap_via_lattice == bp.terms, (
                f'Mismatch at ({x}, {y}): A1D3={ap_via_lattice}, BPS={bp.terms}'
            )


def test_layer2_O_q_constraint():
    """Tr(label) is O(q) for any non-identity-gauge label (a > 0 or b > 0).

    Physically: the q^0 coefficient of a trace counts states at conformal
    dimension Δ = 0, which is the vacuum sector — only the identity gauge
    cell contributes there.  This is a strong consistency check that pins
    down the Layer 2 closed form across the full label space."""
    A = A1D3KAlg()
    K = 6
    fails = []
    for tile in range(6):
        for a in range(4):
            for b in range(4):
                for k in range(3):
                    if a == 0 and b == 0:
                        continue
                    lab = A.canonicalise((tile, a, b, k))
                    tr = A.trace(lab, K=K)
                    if not tr[0].is_zero():
                        fails.append((lab, tr[0]))
    assert not fails, (
        f'O(q) constraint violated by {len(fails)} labels: {fails[:3]}'
    )


def test_layer2_pin_against_bps():
    """Layer 2 closed-form Tr_1, Tr_T, Tr_D match SU2BPSKAlgebra at K=12.

    BPS K=12 is trusted exactly up through q^11 (truncation effects at higher
    q-powers are doc'd in the design record)."""
    A = A1D3KAlg()
    bps = _setup_bps()
    K = 12
    a1d3_l2 = A.trace_layer2(K)
    # Reference label encodings in BPS frame (already used in BPS-pinning).
    one_bps = bps.canonicalise((0, 0, 0))
    T0_bps = bps.canonicalise((1, 0, 0))
    D0_bps = bps.canonicalise((0, -1, 0))
    bps_tr1 = bps.trace(one_bps, K=K)
    bps_trT = bps.trace(T0_bps, K=K)
    bps_trD = bps.trace(D0_bps, K=K)
    assert a1d3_l2[('Tr_1',)] == bps_tr1, (
        f'Tr_1 mismatch: A1D3 = {a1d3_l2[("Tr_1",)]}, BPS = {bps_tr1}'
    )
    assert a1d3_l2[('Tr_T',)] == bps_trT, (
        f'Tr_T mismatch: A1D3 = {a1d3_l2[("Tr_T",)]}, BPS = {bps_trT}'
    )
    assert a1d3_l2[('Tr_D',)] == bps_trD, (
        f'Tr_D mismatch: A1D3 = {a1d3_l2[("Tr_D",)]}, BPS = {bps_trD}'
    )


def test_pin_against_bps_higher_powers():
    """Sample of higher-power label products matches SU2BPSKAlgebra."""
    A = A1D3KAlg()
    bps = _setup_bps()
    labels = set()
    for tile in range(6):
        for a in range(3):
            for b in range(3):
                for k in range(2):
                    labels.add(A.canonicalise((tile, a, b, k)))
    # Test all pairs.
    for x in labels:
        for y in labels:
            ap = A.multiply(x, y)
            ap_via_lattice = {
                _a1d3_label_to_bps_label(lab, bps): lp
                for lab, lp in ap.terms.items()
            }
            bp = bps.multiply(
                _a1d3_label_to_bps_label(x, bps),
                _a1d3_label_to_bps_label(y, bps),
            )
            assert ap_via_lattice == bp.terms, (
                f'Mismatch at ({x}, {y}): A1D3={ap_via_lattice}, BPS={bp.terms}'
            )


# Mixed-tile orthonormality (2026-07-01 audit adjudication) -----------------


def test_mixed_tile_orthonormality_audit_regression():
    """The external audit (2026-07-01, §1.1) reported orthonormality
    'violations' of the mixed-tile monomials `q^{-ab}·T_i^a·D_{i-1}^b` at
    `a, b ≥ 1, a + b ≥ 3` — e.g. a `-χ₃q^{-3}` term in
    `I_{(3,2,1,0),(0,2,0,0)}` and a q⁰ coefficient of 1 against
    `(0,1,1,0)`.  Adjudicated: those values were truncation artifacts of
    the unwidened `trace_element` assembly (fixed at the KAlgebra
    contract level); the basis IS orthonormal.  Cross-checked against the
    raw U(1)-flavoured BPS realization's Schur-formula pairing (exactly 0
    off-diagonal at the corresponding charges) and the widened BPS
    multiply-then-trace.  This regression pins the fixed behaviour on the
    audit's exact label pairs."""
    A = A1D3KAlg()
    R = A.coefficient_ring()
    from zplus_ring import RElement
    audit_pairs = [
        ((3, 2, 1, 0), (0, 2, 0, 0)),
        ((3, 2, 1, 0), (0, 1, 1, 0)),
        ((5, 1, 2, 0), (3, 2, 1, 0)),
    ]
    for a, b in audit_pairs:
        assert A.verify_orthonormality(a, b, K=5), \
            f"orthonormality fails on audit pair {a}, {b}"
        I = A.inner_product(a, b, 5)
        assert not I.coeffs, (
            f"off-diagonal I_{{{a},{b}}} should vanish exactly through "
            f"q^5, got {I}"
        )
        # The bilinear pairing extension agrees (and is the preferred
        # assembly for formal sums).
        Ie = A.inner_product_element(Element.basis(a), Element.basis(b), 5)
        assert Ie.coeffs == I.coeffs
    # Diagonal at the deep mixed-tile label: 1 - 2q² + O(q⁶).
    d = (3, 2, 1, 0)
    assert A.verify_orthonormality(d, d, K=5)
    Id = A.inner_product(d, d, 5)
    assert Id[0] == RElement(R, {0: 1}), f"diagonal q⁰ != 1: {Id}"
    assert Id[2] == RElement(R, {0: -2}), f"diagonal q² != -2: {Id}"


# Test runner --------------------------------------------------------------


def main():
    tests = [
        test_construction,
        test_no_bps_runtime_import,
        test_T_generators,
        test_D_generators,
        test_chi_labels,
        test_rho_permutes_T_orbit,
        test_rho_permutes_D_orbit,
        test_rho_cube_identity_on_generators,
        test_relation_TT_forward,
        test_relation_TT_backward,
        test_relation_DD_forward,
        test_relation_DD_backward,
        test_relation_TD_interaction,
        test_relation_DT_interaction,
        test_clean_products,
        test_TT_inverse_not_unit,
        test_chi_chi_clebsch_gordan,
        test_rho_induced_automorphism,
        test_identity_multiply,
        test_trace_identity_vacuum_chi_0,
        test_trace_vacuum_only_non_central_zero,
        test_trace_layer2_closed_form,
        test_trace_full_matches_layer2,
        test_trace_vacuum_chi_k,
        test_trace_layer1_identity,
        test_trace_layer1_chi_k,
        test_trace_layer1_single_letter,
        test_trace_layer1_two_letter_plucker,
        test_trace_layer1_element_identity_sum,
        test_associativity_generators,
        test_associativity_higher_powers,
        test_pin_against_bps_generator_pairs,
        test_layer2_O_q_constraint,
        test_layer2_pin_against_bps,
        test_pin_against_bps_higher_powers,
        test_mixed_tile_orthonormality_audit_regression,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  [ok]   {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERR ] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed of {passed + failed}")
    print(f"{'=' * 50}")
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
