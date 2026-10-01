"""`GBPSKAlgebra` — the BPS chart with non-Abelian flavour `G`.

Name ruled by the author, 2026-09-14 (*"I think a GBPSKAlgebra class would be
fine"*).

The acceptance test is `src/cone/a1dn_kalg.py::A1DnKAlg`(3)`, the
`[A_1, D_3]` algebra on the curves of the triangle with one interior puncture:
`FlavouredQuiver([[0,1],[-1,0]], [1,2])` unfolds to exactly the `D_3` pairing
(`_build_d_n_pairing(3)`), so the general machine must reproduce it.  Through a
test-local table of the six curves' gauge charges (the design record,
the charge-to-curve map at `v = 3`) it reproduces the **products** on the full
population — 𝖖-commuting and crossing pairs alike — the **traces**, and **ρ**.
Until 2026-09-23 the class at that name was the `D_3`-chamber quantum torus,
whose ρ (the antipode) disagreed with the BPS half-monodromy (the audit, fixed by the rebuild; the torus is the archived tree).

Labels are `(γ, r)`: a reduced gauge charge and an irrep, the element being the
sum of unfolded canonicals over `r`'s weight diagram.  That this — rather than
`(γ, w)` — is the right index set was *measured*: an `A1DnKAlg` basis element
squares to a Clebsch–Gordan sum, so it carries a whole rep, not one weight.

Run:  `python3 run_tests.py`
"""
import sys

sys.path.insert(0, ".")

from a1dn_kalg import A1DnKAlg
from bps_kalgebra import BPSKAlgebra
from flavoured_factor_spectrum import FlavouredQuiver
from gbps_kalgebra import GBPSKAlgebra

# The flavoured quiver whose unfolding IS `A1DnKAlg(3)`'s pairing:
# reduced node 0 is the trivalent node (unflavoured), reduced node 1 carries the
# `SU(2)` under which the two leaves rotate.
ANCHOR = FlavouredQuiver([[0, 1], [-1, 0]], [1, 2])

LABELS = [(0, 1, 0), (0, 1, 1), (0, 2, 0), (1, 0, 0),
          (1, 1, 0), (1, 1, 1), (0, 2, 1), (1, 2, 0)]

# `A1DnKAlg(3)`'s six curves `(x, ℓ)` and their reduced gauge charges on the
# anchor (test-local, the design record).  The loop `(0, 3)` is the trivalent node,
# the short curve `(0, 2)` the leaf pair; ρ — `x ↦ x + 1` on curves, the
# half-monodromy on charges — carries each to the next.  Each cone (a pair of
# non-crossing curves) is a unimodular basis of the charge lattice, so the map
# `(curves, κ) ↦ (Σ m·charge, χ_κ)` is a bijection of canonical labels.
CURVE_CHARGE = {(0, 3): (1, 0), (1, 3): (-1, -2), (2, 3): (-1, 0),
                (0, 2): (0, 1), (1, 2): (0, -1), (2, 2): (-1, -1)}


def _to_g(gamma):
    """An `A1DnKAlg` label `γ` (with `γ₁ ≥ γ₂`) ↦ a `GBPSKAlgebra` label."""
    g0, g1, g2 = gamma
    assert g1 >= g2, gamma
    return ((g0, g1 + g2), ((), (g1 - g2,) if g1 - g2 else ()))


def _from_g(label):
    (c0, c1), rep = label
    n = rep[1][0] if rep[1] else 0
    return (c0, (c1 + n) // 2, (c1 - n) // 2)


def _curve_to_g(label, table=CURVE_CHARGE):
    """An `A1DnKAlg(3)` label `(curves, κ)` ↦ a `GBPSKAlgebra` label."""
    curves, kappa = label
    d0 = sum(m * table[c][0] for c, m in curves)
    d1 = sum(m * table[c][1] for c, m in curves)
    return ((d0, d1), ((), (kappa,) if kappa else ()))


def _g_to_curve(K, label):
    """The inverse, by the unique cone containing the charge."""
    (d0, d1), rep = label
    kappa = rep[1][0] if rep[1] else 0
    hits = set()
    for cone in K.cone_data().cones():
        c1, c2 = sorted(cone)
        (a0, a1), (b0, b1) = CURVE_CHARGE[c1], CURVE_CHARGE[c2]
        det = a0 * b1 - a1 * b0
        assert abs(det) == 1, (cone, det)
        m1, m2 = (d0 * b1 - d1 * b0) * det, (a0 * d1 - a1 * d0) * det
        if m1 >= 0 and m2 >= 0:
            hits.add(K.canonicalise((((c1, m1), (c2, m2)), kappa)))
    assert len(hits) == 1, (label, hits)
    return hits.pop()


def _mapped_product(K, a, b, table=CURVE_CHARGE):
    out = {}
    for lab, poly in K.multiply(a, b).terms.items():
        g = _curve_to_g(lab, table)
        out[g] = out[g] + poly if g in out else poly
    return {g: p for g, p in out.items() if not p.is_zero()}


def _population(K):
    """The six curves at κ ∈ {0, 1} and every degree-2 monomial at κ = 0."""
    cs = sorted(CURVE_CHARGE)
    atoms = [K.curve(x, ell, kappa=kap) for (x, ell) in cs for kap in (0, 1)]
    deg2 = [(((c, 2),), 0) for c in cs]
    deg2 += [K.canonicalise((((c, 1), (d, 1)), 0))
             for i, c in enumerate(cs) for d in cs[i + 1:]
             if K.cone_data().q_commute(c, d)]
    return atoms + deg2


def test_the_unfolded_anchor_is_the_a1dn_quiver():
    """`A2` with `SU(2)` at one node unfolds to `A1DnKAlg(3)`'s exchange matrix.

    Which is why that module is the acceptance test at all: the two describe
    the same chart, one by hand for a single `SU(2)` and one by the general
    machine.
    """
    from a1dn_kalg import _build_d_n_pairing
    B, _charges, labels = ANCHOR.unfold()
    assert B == _build_d_n_pairing(3), B
    assert labels == [(0, (0,)), (1, (-1,)), (1, (1,))], labels


def test_the_labels_are_charge_and_irrep_not_charge_and_weight():
    """A basis element carries a whole REP, measured on the anchor.

    `A1DnKAlg(3)`'s leaf doublet (the curve `(0, 2)` at `κ = 1`) squares to
    the Clebsch–Gordan sum of `(0, 2)²` at `κ = 0` and `κ = 2`, while the
    unfolded chart's `L_{(0,1,0)}²` is the single term `L_{(0,2,0)}`.  So the
    flavoured basis element is not one weight — it is `χ_r · L_section`, and
    the index set is `(charge, irrep)`.
    """
    K = A1DnKAlg(3)
    unf = BPSKAlgebra(pairing=ANCHOR.unfold()[0],
                      node_charges=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    doublet = K.curve(0, 2, kappa=1)      # the leaf doublet: charge (0,1), χ₁
    assert _curve_to_g(doublet) == ((0, 1), ((), (1,)))
    assert set(K.multiply(doublet, doublet).terms) == {
        ((((0, 2), 2),), 0), ((((0, 2), 2),), 2)}
    assert len(unf.multiply((0, 1, 0), (0, 1, 0)).terms) == 1

    G = GBPSKAlgebra(ANCHOR)
    # and the flavoured element really is the sum over the weight diagram
    assert G.expand(((0, 1), ((), (1,)))) == {(0, 1, -1): 1, (0, 1, 1): 1}


def test_the_weight_lattice_is_a_factor_so_n_ality_is_relaxed():
    """The flavour weight lattice is a FACTOR of the charge lattice.

    The author's ruling, 2026-09-14: *"in order to place your algebra in a flavoured
    KAlgebra context, you need to have the weight lattice of the non-Abelian
    flavour as a factor of the overall charge lattice.  Then when looking for
    F's you can relax the gauge-N-ality connection and satisfy the axioms"*.

    An earlier frame built the chart on the UNFOLDED quiver, whose node charges
    span only a finite-index sublattice of this one — for `SU(2)` they generate
    the root `2ω`, never the fundamental weight `ω`.  That index manufactured a
    gauge-N-ality condition coupling the irrep to the charge.  Here `ω` is a
    lattice point, so every irrep lives over every gauge charge, and the label
    the old frame refused outright is an ordinary canonical.
    """
    G = GBPSKAlgebra(ANCHOR, max_irrep=4)
    odd, triv = ((0, 1), ((), (1,))), ((0, 1), ((), ()))
    # the trivial irrep over an ODD gauge charge — refused by the old N-ality
    assert G.expand(triv) == {(0, 1, 0): 1}
    assert G.verify_orthonormality(triv, triv, K=4)
    # and both irreps live over the same charge, so the filter is gone
    reps = G.admissible_irreps((0, 1))
    assert ((), ()) in reps and ((), (1,)) in reps and ((), (2,)) in reps
    assert G.admissible_irreps((0, 1)) == G.admissible_irreps((0, 2))
    # ω itself is in the lattice: the chart's flavour direction IS the weight
    assert G._chart_flavour_weights() == [(1,)], G._chart_flavour_weights()
    del odd


def test_the_chart_is_the_one_gf_bps_kalgebra_already_takes():
    """`Γ_gauge ⊕ P` reproduces `GfBPSKAlgebra`'s own input, mechanically.

    `src/bps/gf_bps_kalgebra.py::GfBPSKAlgebra` is the established
    single-simple-factor G-flavoured BPS realisation, and its `SU(2)` and
    `SU(3)` test instances are hand-written matrices.  `gauge_weight_chart()`
    derives exactly those from the flavoured quiver — which is the check that
    this construction sits in the established frame rather than beside it.
    """
    M, charges, _ = ANCHOR.gauge_weight_chart()
    assert M == [[0, 1, 0], [-1, 0, 0], [0, 0, 0]], M
    assert sorted(charges) == sorted([(1, 0, 0), (0, 1, 1), (0, 1, -1)]), charges

    from flavoured_factor_spectrum import FlavouredQuiver as FQ
    M3, c3, _ = FQ([[0, 1], [-1, 0]], [1, 3]).gauge_weight_chart()
    assert M3 == [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], M3
    assert sorted(c3) == sorted([(1, 0, 0, 0), (0, 1, 1, 0),
                                 (0, 1, 0, 1), (0, 1, -1, -1)]), c3


def test_products_reproduce_the_hand_written_a1dn_instance():
    """The acceptance test: `GBPSKAlgebra` == `A1DnKAlg(3)` on products.

    Exact, structure constant for structure constant, with the irreps fusing
    by Clebsch–Gordan as they must: the eight charge labels of the earlier
    test (through `_to_g` and the inverse curve map), and the full population
    of the six curves at `κ ∈ {0, 1}` with every degree-2 monomial — 576
    ordered pairs, crossing (non-𝖖-commuting) pairs included.  Negative
    control: the charge table with the `ℓ = 2` orbit shifted by one position
    must fail.
    """
    G = GBPSKAlgebra(ANCHOR)
    K = A1DnKAlg(3)
    # the eight earlier labels
    curve_of = {g: _g_to_curve(K, _to_g(g)) for g in LABELS}
    assert curve_of[(0, 1, 0)] == K.curve(0, 2, kappa=1)
    assert curve_of[(1, 0, 0)] == K.curve(0, 3)
    for a in LABELS:
        for b in LABELS:
            want = dict(G.multiply(_to_g(a), _to_g(b)).terms)
            got = _mapped_product(K, curve_of[a], curve_of[b])
            assert got == want, (a, b, got, want)
    # the full population
    pop = _population(K)
    assert len(pop) == 24, len(pop)
    crossing = 0
    for a in pop:
        for b in pop:
            want = dict(G.multiply(_curve_to_g(a), _curve_to_g(b)).terms)
            got = _mapped_product(K, a, b)
            assert got == want, (a, b, got, want)
            crossing += len(want) > 1 or len(got) > 1
    assert crossing > 0
    # negative control: shift the ℓ = 2 orbit
    shifted = dict(CURVE_CHARGE)
    shifted.update({(x, 2): CURVE_CHARGE[((x + 1) % 3, 2)] for x in range(3)})
    bad = sum(
        _mapped_product(K, a, b, shifted)
        != dict(G.multiply(_curve_to_g(a, shifted),
                           _curve_to_g(b, shifted)).terms)
        for a in pop for b in pop)
    assert bad > 0, "a shifted charge table must fail the product test"


def test_the_chamber_torus_is_not_the_algebra():
    """Negative control for the product test: the `D_3`-chamber quantum torus
    (`SU2QuantumTorusKAlg` on the reduced chart — the class that stood at the
    name `A1DnKAlg` until 2026-09-23) agrees with `A1DnKAlg(3)` on every
    𝖖-commuting pair of single curves and on NO crossing pair, so the
    comparison above can tell the two apart."""
    from su2_quantum_torus_kalgebra import SU2QuantumTorusKAlg
    K = A1DnKAlg(3)
    S = SU2QuantumTorusKAlg([[0, 1], [-1, 0]])

    def to_s(label):
        (d0, d1), rep = _curve_to_g(label)
        return (d0, d1, rep[1][0] if rep[1] else 0)
    atoms = [K.curve(x, ell, kappa=kap) for (x, ell) in sorted(CURVE_CHARGE)
             for kap in (0, 1)]
    agree = {True: 0, False: 0}
    total = {True: 0, False: 0}
    for a in atoms:
        for b in atoms:
            qc = K.cone_data().q_commute(a[0][0][0], b[0][0][0])
            got = {}
            for lab, poly in K.multiply(a, b).terms.items():
                s = to_s(lab)
                got[s] = got[s] + poly if s in got else poly
            total[qc] += 1
            agree[qc] += (got == dict(S.multiply(to_s(a), to_s(b)).terms))
    assert agree[True] == total[True] == 72, (agree, total)
    assert agree[False] == 0 and total[False] == 72, (agree, total)


def test_traces_reproduce_the_a1dn_instance():
    """`GBPSKAlgebra`'s trace (the BPS chart's, un-branched to `R(SU(2))`)
    equals `A1DnKAlg(3)`'s — an oracle for the rebuilt class's trace
    independent of its `a1dodd_layer2` seeds: the identity, `χ₁` and the six
    curves at `κ ∈ {0, 1}` through `𝖖¹²`, the degree-2 monomials of the
    population through `𝖖⁶`.  Negative control: the charge table with the
    `ℓ = 2` and `ℓ = 3` assignments swapped must fail on every single curve.
    (The product test's shifted-orbit table is no control here: the trace of
    a single curve does not depend on `x`, since ρ² generates every rotation
    at odd `n`.)"""
    G = GBPSKAlgebra(ANCHOR)
    K = A1DnKAlg(3)

    def as_dict(series, int_labels, KQ):
        out = {}
        for q, r in series.coeffs.items():
            if q > KQ or r.is_zero():
                continue
            out[q] = {(b if int_labels else (b[0] if b else 0)): c
                      for b, c in r.terms.items() if c}
        return out

    g_cache = {}

    def g_trace(label, KQ):
        if (label, KQ) not in g_cache:
            g_cache[(label, KQ)] = as_dict(G.trace(label, KQ), False, KQ)
        return g_cache[(label, KQ)]

    curves = [K.curve(x, ell, kappa=kap) for (x, ell) in sorted(CURVE_CHARGE)
              for kap in (0, 1)]
    for a in [K.identity(), K.chi(kappa=1)] + curves:
        assert g_trace(_curve_to_g(a), 12) == as_dict(K.trace(a, K=12), True, 12), a
    deg2 = [a for a in _population(K) if sum(m for _c, m in a[0]) == 2]
    assert len(deg2) == 12, len(deg2)
    for a in deg2:
        assert g_trace(_curve_to_g(a), 6) == as_dict(K.trace(a, K=6), True, 6), a
    # negative control: swap the ℓ = 2 and ℓ = 3 charge assignments
    swapped = {(x, 2): CURVE_CHARGE[(x, 3)] for x in range(3)}
    swapped.update({(x, 3): CURVE_CHARGE[(x, 2)] for x in range(3)})
    still_equal = sum(g_trace(_curve_to_g(a, swapped), 12)
                      == as_dict(K.trace(a, K=12), True, 12) for a in curves)
    assert still_equal == 0, still_equal


def test_the_coefficient_ring_is_the_non_abelian_flavour_ring():
    """`R(SU(2))`, natively — not the `U(1)^f` the unfolded chart carries.

    This is the point of the class: the non-Abelian flavour is the
    presentation, rather than something recovered afterwards by un-branching a
    Cartan presentation (`flavour_enhancement`).
    """
    from zplus_ring import AbelianZPlusRing, SUNZPlusRing

    G = GBPSKAlgebra(ANCHOR)
    assert isinstance(G.coefficient_ring(), SUNZPlusRing)
    assert G.coefficient_ring().N == 2
    unf = BPSKAlgebra(pairing=ANCHOR.unfold()[0],
                      node_charges=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    assert isinstance(unf.coefficient_ring(), AbelianZPlusRing)


def test_the_kalgebra_axioms_hold():
    """The contract's verifiers on the flavoured surface."""
    G = GBPSKAlgebra(ANCHOR)
    labels = [((0, 0), ((), ())), ((0, 1), ((), (1,))), ((1, 0), ((), ())),
              ((0, 2), ((), (2,))), ((1, 1), ((), (1,)))]
    assert G.verify_identity_in_basis()
    assert G.verify_rho_fixes_identity()
    for a in labels:
        assert G.verify_rho_inverse(a), a
    for a in labels:
        for b in labels:
            assert G.verify_bar_involution(a, b), (a, b)
            assert G.verify_rho_is_automorphism(a, b), (a, b)


def test_rho_is_the_half_monodromy_and_a1dn_agrees():
    """`ρ` is pinned by the TRACE axiom, not by being an automorphism.

    The pre-2026-09-23 `A1DnKAlg` (the `D_3`-chamber torus, now
    the archived tree) had `ρ: γ ↦ −γ`, the antipode — a
    genuine automorphism, but not the canonical `ρ`, which is fixed by
    `I_{a,b} = Tr(ρ(a)·b) = δ + O(\\fq)` (the audit).  The rebuilt
    class's `ρ` (the rotation `x ↦ x + 1` of the curves) agrees with the BPS
    half-monodromy on every label below — the six earlier charge labels and
    the whole population.  The antipode stays below as the negative control:
    on the unfolded chart orthonormality picks `σ` (leading coefficient `1`),
    and negation fails it (`𝖖²`, `0`, `𝖖⁴`).
    """
    G = GBPSKAlgebra(ANCHOR)
    K = A1DnKAlg(3)
    unf = BPSKAlgebra(pairing=ANCHOR.unfold()[0],
                      node_charges=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])

    agree = []
    for gamma in [(0, 1, 0), (0, 1, 1), (0, 2, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]:
        c = _g_to_curve(K, _to_g(gamma))
        if G.rho(_to_g(gamma)) == _curve_to_g(K.rho(c)):
            agree.append(gamma)
    assert agree == [(0, 1, 0), (0, 1, 1), (0, 2, 0), (1, 0, 0), (1, 1, 0),
                     (1, 1, 1)], agree
    for a in _population(K):
        assert G.rho(_curve_to_g(a)) == _curve_to_g(K.rho(a)), a

    # Orthonormality on the unfolded chart decides, and decides for `σ`.
    for lab in [(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)]:
        assert unf.verify_orthonormality(lab, lab, K=4), lab
    # while the antipode fails it away from the agreeing label: the `\fq^0`
    # coefficient of `Tr(L_{-a}·L_a)` is not 1, so negation is not the `ρ` the
    # orthonormality axiom pins.
    def q0_coefficient(series):
        term = series.coeffs.get(0)
        if term is None:
            return 0
        # the identity (trivial-character) component of the `\fq^0` term
        return term.terms.get(series.ring.one_basis(), 0)

    checked = 0
    for lab in [(1, 0, 0), (1, 1, 0), (1, 1, 1)]:
        neg = tuple(-x for x in lab)
        got = q0_coefficient(unf.trace_element(unf.multiply(neg, lab), 4))
        assert got != 1, (lab, got)
        checked += 1
    assert checked == 3
    # and the control: with the chart's own `σ` it IS 1
    for lab in [(1, 0, 0), (1, 1, 0), (1, 1, 1)]:
        got = q0_coefficient(
            unf.trace_element(unf.multiply(unf.rho(lab), lab), 4))
        assert got == 1, (lab, got)


def test_the_trace_is_valued_in_the_flavour_ring_not_the_chart_cartan():
    """`Tr(L_a) ∈ R(G)((\fq))`, as the contract requires.

    The chart's own traces are valued in its Cartan ring `Z[U(1)^f]`; returning
    them verbatim satisfies no axiom and makes every trace-derived consumer
    raise `ValueError: RPowerSeries: R-coefficient ring mismatch` — including
    `verify_orthonormality`, the contract's central axiom.  Asserted here on the
    RING and on the consumers, NOT by restating the summation, which is the
    error the previous version of this test made: it re-derived the
    implementation and so could not see the wrong ring.
    """
    from zplus_ring import SUNZPlusRing

    G = GBPSKAlgebra(ANCHOR)
    for label in [((0, 0), ((), ())), ((0, 1), ((), (1,))),
                  ((0, 2), ((), (2,))), ((1, 1), ((), (1,)))]:
        tr = G.trace(label, 4)
        assert tr.ring == G.coefficient_ring(), (label, tr.ring)
        assert isinstance(tr.ring, SUNZPlusRing)
        assert G.verify_orthonormality(label, label, K=4), label
    # the contract's default K is honoured (a bare `trace(a)` must work)
    assert G.trace(((0, 1), ((), (1,)))).ring == G.coefficient_ring()


def test_the_inner_product_is_the_character_product_not_a_dimension():
    """`I_{a,b}[\fq⁰] = δ_{γ,γ'}·χ_{r*}·χ_{r'}` — character-valued.

    The discriminator: a doublet's self-pairing is `χ_triv + χ_adj`, NOT the
    number `dim = 2`.  Reading the flavour off as a dimension is exactly what an
    un-lifted (augmented) trace would give, so this separates the two.  The
    contract's `verify_orthonormality` then passes because it tests the `χ₀`
    component, which is `δ` by Schur orthogonality.
    """
    G = GBPSKAlgebra(ANCHOR, max_irrep=4)
    R = G.coefficient_ring()
    F = ANCHOR.flavour
    labels = [((0, 1), ((), (1,))), ((0, 1), ((), ())), ((0, 2), ((), (2,))),
              ((1, 0), ((), ())), ((1, 1), ((), (1,)))]
    checked = 0
    for a in labels:
        for b in labels:
            c0 = G.inner_product(a, b, 2).coeffs.get(0)
            got = {k: v for k, v in (c0.terms.items() if c0 is not None else [])
                   if v}
            if a[0] == b[0]:
                want = {k: v for k, v in R.multiply_basis(
                    R.star_basis(F.irrep_to_ring_key(a[1])),
                    F.irrep_to_ring_key(b[1])).items() if v}
            else:
                want = {}
            assert got == want, (a, b, got, want)
            checked += 1
    assert checked == 25
    # the discriminator, stated outright
    d = ((0, 1), ((), (1,)))
    c0 = G.inner_product(d, d, 2).coeffs.get(0)
    assert dict(c0.terms) == {(): 1, (2,): 1}, dict(c0.terms)


def test_the_flavour_lift_coordinate_meets_the_contract():
    """`r_label_decompose` hands back a `coefficient_ring()` BASIS KEY.

    An `Irrep` (one partition slot per flavour factor, trivial factors included)
    is a different object from a ring key, and the ring does not reject it —
    `verify_section_is_single_irrep` passes on one because it only counts terms,
    while `R.dim` raises and `to_R_form` emits ring-invalid data silently.  So
    the assertion is on `dim` and on the round trip, not on term counts.
    """
    from kalgebra import Element

    G = GBPSKAlgebra(ANCHOR, max_irrep=4)
    R = G.coefficient_ring()
    F = ANCHOR.flavour
    labels = [((0, 0), ((), ())), ((0, 1), ((), (1,))), ((0, 1), ((), ())),
              ((0, 2), ((), (2,))), ((1, 1), ((), (1,)))]
    for lab in labels:
        section, key = G.r_label_decompose(lab)
        assert section == (lab[0], F.trivial_irrep()), (lab, section)
        assert R.dim(key) == F.irrep_dim(lab[1]), (lab, key)
        assert G.r_label_compose(section, key) == lab, lab
        assert G.verify_section_is_single_irrep(lab), lab
        assert G.verify_embed_section_roundtrip(lab), lab
        assert G.from_R_form(G.to_R_form(Element.basis(lab))) == \
            Element.basis(lab), lab
    # `embed_R` is TOTAL here: every irrep sits over the zero gauge charge
    assert G.embed_R(R.basis_element((1,))) == \
        Element.basis(((0, 0), ((), (1,))))
    # and `forget()` weights by the dimension, which needs the key to be real
    adj = ((0, 2), ((), (2,)))
    assert G.forget().multiply(adj, adj).terms == \
        {((0, 4), ((), ())): __import__("laurent_poly").LaurentPoly({0: 9})}


def test_rhos_flavour_half_is_rep_duality():
    """`ρ(γ, r) = (…, r*)` — the flavour half of `ρ` IS the rep dual.

    `kalgebra.md` states that `⋆` is `ρ` restricted to the central flavour
    subalgebra, so this is a derivation from the axioms, not a coincidence.
    Measured here on non-self-dual groups, where it has content: at `SU(3)` the
    fundamental must go to the anti-fundamental.

    NOT turned into a delegation.  `_transport` computes this the long way, via
    the weight diagram of the image — about 36x slower than asking the
    coefficient ring for `star_basis` — and an audit proposed replacing it.  The
    reason to keep the long route is that this class's job, since the product
    extension landed in `GfBPSKAlgebra`, is to be the
    INDEPENDENT construction that cross-certifies it
    (`tests/test_gf_bps_kalgebra.py::test_agrees_with_the_independent_gbps_construction`).
    Deriving `ρ` from the coefficient ring would make it share exactly the step
    it exists to check.  Pinned as a test so the identity is recorded without
    the two routes collapsing into one.
    """
    import itertools

    checked = 0
    for factors in ([1, 2], [1, 3], [1, 4], [2, 2], [2, 3]):
        fq = FlavouredQuiver([[0, 1], [-1, 0]], factors)
        G = GBPSKAlgebra(fq, max_irrep=4)
        F, R = fq.flavour, G.coefficient_ring()
        labels = [(c, r) for c in itertools.product(range(-2, 3), repeat=fq.rank)
                  for r in G.admissible_irreps(c)][:40]
        for lab in labels:
            want = F.ring_key_to_irrep(R.star_basis(F.irrep_to_ring_key(lab[1])))
            assert G.rho(lab)[1] == want, (factors, lab, G.rho(lab)[1], want)
            checked += 1
    assert checked > 150, checked

    # content check: at SU(3) the fundamental is NOT self-dual
    fq = FlavouredQuiver([[0, 1], [-1, 0]], [1, 3])
    G = GBPSKAlgebra(fq, max_irrep=4)
    F = fq.flavour
    fund = F.fundamental(1)
    assert G.rho(((0, 1), fund))[1] != fund, "SU(3) fundamental must not be self-dual"


def test_a_degenerate_reduced_pairing_is_refused_not_silently_halved():
    """A residual abelian flavour must not be discarded into the trivial character.

    If the REDUCED pairing is degenerate, the chart on `Γ_gauge ⊕ P` carries an
    abelian flavour from `ker(B)` on top of the weight lattice, and the residual
    direction projects to flavour weight `0` — so un-branching would conflate
    that `U(1)` charge with the trivial character and return a trace whose
    grading is silently gone.  `verify_orthonormality` does NOT catch it: it
    tests the `χ₀` component only, and returned `True` throughout while the
    trace was wrong.  Refused until the ring-growth rule
    `TensorZPlusRing([AbelianZPlusRing(rk ker B), ⊗_a R(SU(N_a))])` and the
    matching section split are built.

    Not an edge case: every antisymmetric matrix of ODD rank is degenerate.
    """
    from flavoured_factor_spectrum import FlavourSpace

    # positive control: the anchor's reduced pairing is NON-degenerate and builds
    assert GBPSKAlgebra(ANCHOR).coefficient_ring() is not None

    cases = [
        FlavouredQuiver([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], FlavourSpace([1, 1, 2]),
                        [((), (), ()), ((), (), ()), ((), (), (1,))]),
        FlavouredQuiver([[0, 1, 0], [-1, 0, 1], [0, -1, 0]], [1, 1, 2]),
    ]
    for fq in cases:
        try:
            GBPSKAlgebra(fq)
        except NotImplementedError as exc:
            assert "ker B" in str(exc), str(exc)
            continue
        raise AssertionError(f"expected a degenerate reduced pairing to be refused: {fq}")


def test_a_shared_flavour_factor_is_refused():
    """Out of scope, and said so rather than silently mishandled."""
    from flavoured_factor_spectrum import FlavourSpace

    shared = FlavouredQuiver([[0, 1], [-1, 0]], FlavourSpace([2]),
                             [((1,),), ((1,),)])
    try:
        GBPSKAlgebra(shared)
    except NotImplementedError:
        return
    raise AssertionError("expected a shared flavour factor to be refused")


TESTS = [
    test_the_unfolded_anchor_is_the_a1dn_quiver,
    test_the_labels_are_charge_and_irrep_not_charge_and_weight,
    test_the_weight_lattice_is_a_factor_so_n_ality_is_relaxed,
    test_the_chart_is_the_one_gf_bps_kalgebra_already_takes,
    test_products_reproduce_the_hand_written_a1dn_instance,
    test_traces_reproduce_the_a1dn_instance,
    test_the_chamber_torus_is_not_the_algebra,
    test_the_coefficient_ring_is_the_non_abelian_flavour_ring,
    test_the_kalgebra_axioms_hold,
    test_rho_is_the_half_monodromy_and_a1dn_agrees,
    test_the_trace_is_valued_in_the_flavour_ring_not_the_chart_cartan,
    test_the_inner_product_is_the_character_product_not_a_dimension,
    test_the_flavour_lift_coordinate_meets_the_contract,
    test_rhos_flavour_half_is_rep_duality,
    test_a_degenerate_reduced_pairing_is_refused_not_silently_halved,
    test_a_shared_flavour_factor_is_refused,
]


if __name__ == "__main__":
    for fn in TESTS:
        fn()
        print(f"  PASS: {fn.__name__}")
    print(f"All {len(TESTS)} GBPSKAlgebra tests passed.")
