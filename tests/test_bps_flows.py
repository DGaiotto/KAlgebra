"""Self-test for the BPSKAlgebra (Step 4) — the BPS-quiver realization engine.

Unlike Steps 1–3 (deliberately spine-free), Step 4 *is* the BPS spine: the
single-chart `BPSKAlgebra` realization of `A_𝖖[T]` over a BPS quiver + spectrum
generator, with the F-finder, chart graph, spec shortening, node-deletion RG
flows, and isomorphism witnesses. Every operation accepts arbitrary inputs and
every trace is improvable to any q-order; the spine-free guarantee of the
earlier layers does not apply here — this layer is the spine.

What this exercises:

  * `test_pentagon_spec`     — the pentagon `A_𝖖([A₁,A₂])` from its BPS quiver
                               (`pairing=[[0,1],[-1,0]]`, nodes `(1,0),(0,1)`):
                               known multiply / inner_product values, the axiom
                               battery (`verify_canonical_basis`), orthonormality,
                               and trace truncation-stability with
                               **no RuntimeWarning**.
  * `test_pentagon_spec_free`— the same algebra built **spec-free** (`build_S=True`,
                               `S` from the quiver alone): multiply / trace /
                               inner_product reproduce the spec-mode values over a
                               label grid.
  * `test_pentagon_iso`      — a `KAlgebraIso` BPS-pentagon ↔ the Step-1
                               `PentagonSampleKAlgebra` (`verify_all`): the
                               cross-check that the two presentations are the same
                               abstract algebra (multiplicative + trace-equivariant).
  * `test_hexagon_flavoured` — a flavoured theory (the hexagon, `ker B = (1,1,1)`,
                               one U(1) flavour): `coefficient_ring`, `to_R_form`,
                               and the flavoured trace over `R((q))`.
  * `test_directional_nodedrop` — a node-deletion RG flow (`DirectionalSingleNodeRG`)
                               certified against an independent UV `BPSKAlgebra`.

The `S`-building layer, whose checks are deliberately weighted towards
**independent** cross-checks — an engine's agreement with itself says nothing
about the conjecture underneath it:

  * `test_bps_factor_spectrum`      — `bps_factor_spectrum` (`S` from its leading data as a
                               product of palindromic BPS factors, one per
                               `(γ, s)` pair) against the
                               Nahm-sum expansion of a known chamber spec and
                               against the chart's own `[S|0⟩]_γ`;
                               order-independence of `S` asserted *non-vacuously*;
                               the central charge selecting pure SU(2)'s
                               weak-coupling chamber (dyon tower + the W boson,
                               its one spin-1/2 state); and coverage of Markov
                               (= N=2*), where the peel gate trips.
  * `test_peel_engine_retired` — every door into the peel recursion refuses, and
                               refuses with the *retirement* error; an unknown
                               engine does not; both opt-ins reach the intact
                               engine, which still agrees with the factor build; and
                               the context manager does not leak the opt-in.
  * `test_fs_builder`        — `F_γ` and `S` grown together out of
                               `F_γ·S = X_γ + O(𝖖)`, reproducing the F-solver over
                               a charge grid by a route that never saw the support
                               window — plus the degenerate `Ω ≡ 0` solution,
                               asserted because the relation *alone* pins nothing.
  * `test_factor_order_search` — a cored quiver's BPS factors collapsing to
                               spin-0 ones that rebuild `S` independently, and the acyclic
                               short-circuit landing on the node charges.

Run with every `src/<layer>/` directory on the path (the BPS layer imports the
Step-1 core and the Step-3 engine — nothing is duplicated):

    python3 run_tests.py
"""
import os
import sys
import warnings

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
for _root, _dirs, _ in os.walk(_SRC):
    _dirs[:] = [_d for _d in _dirs if _d != "__pycache__"]
    if _root not in sys.path:
        sys.path.insert(0, _root)

from laurent_poly import LaurentPoly
from habiro import HabiroElement
from q_number_poly import QNumberPoly
from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from samples import PentagonSampleKAlgebra
from bps_kalgebra import BPSKAlgebra
from directional_subquiver_rg import (
    DirectionalSingleNodeRG,
    certify_directional_vs_bps,
)
from bps_atlas import BPSAtlas
from bps_factor_spectrum import (
    PLACEMENT_ORDERS,
    BPSFactorSpectrum,
    build_spectrum_generator_from_factors,
    spin_decompose,
)
from recursive_spectrum import (
    RetiredEngineError,
    Theory,
    build_spectrum_generator,
    enable_retired_peel_engine,
)
from fs_builder import FSBuilder
from factor_order_search import find_simple_factorisation

_ONE = LaurentPoly.one()

PENTA_PAIRING = [[0, 1], [-1, 0]]
PENTA_NODES = [(1, 0), (0, 1)]
PENTA_STUFF_FIRST = [(0, 1), (1, 1), (1, 0)]

# The five pentagon BPS chord-charges, in ρ-orbit order (the generator
# correspondence to the Step-1 PentagonSampleKAlgebra labels 0..4).
CHORD_CHARGE = {0: (1, 0), 1: (0, -1), 2: (-1, -1), 3: (-1, 0), 4: (0, 1)}


def _ser(rps, K):
    return {e: str(r) for e, r in rps.coeffs.items()
            if e <= K and str(r) not in ("0", "")}


def _nm(el):
    return {k: str(v) for k, v in el.terms.items() if not v.is_zero()}


# ---------------------------------------------------------------------------
# Pentagon BPS-pentagon ↔ PentagonSampleKAlgebra iso (constructive forward).
# ---------------------------------------------------------------------------
#
# The Step-1 sample labels a canonical element `L_i^a · L_{i+1}^b` by `(i, a, b)`;
# the BPS realization labels it by its charge.  The forward image of `(i, a, b)`
# is the single BPS charge supporting the canonical-order product of the
# generator images (the q-cocycle phase from the BPS multiply is absorbed — a
# canonical-basis iso sends `L_a` to a single `L_{f(a)}` with coefficient 1).

def _support_charge(B, i, a, b):
    img = Element({(0, 0): _ONE})
    gi = Element({CHORD_CHARGE[i % 5]: _ONE})
    gi1 = Element({CHORD_CHARGE[(i + 1) % 5]: _ONE})
    for _ in range(a):
        img = B.multiply_elements(img, gi)
    for _ in range(b):
        img = B.multiply_elements(img, gi1)
    supp = [k for k, v in img.terms.items() if not v.is_zero()]
    assert len(supp) == 1, ("non-monomial forward image", (i, a, b), supp)
    return supp[0]


def _bps_to_pent_label(charge):
    """BPS charge `(m, n)` → pentagon canonical label `(i, a, b)`."""
    m, n = charge
    if m == 0 and n == 0:
        return (0, 0, 0)
    for i in range(5):
        gi = CHORD_CHARGE[i]
        gi1 = CHORD_CHARGE[(i + 1) % 5]
        det = gi[0] * gi1[1] - gi[1] * gi1[0]
        if det == 0:
            continue
        a_num = m * gi1[1] - n * gi1[0]
        b_num = -m * gi[1] + n * gi[0]
        if a_num % det or b_num % det:
            continue
        a = a_num // det
        b = b_num // det
        if a < 0 or b < 0:
            continue
        if a == 0 and b == 0:
            return (0, 0, 0)
        if b == 0:
            return (i, a, 0)
        if a == 0:
            return ((i + 1) % 5, b, 0)
        return (i, a, b)
    raise ValueError(f"BPS charge {charge} lies in no pentagon cone")


def _pentagon_iso(P, B):
    def fwd(label):
        i, a, b = label
        return Element({_support_charge(B, i, a, b): _ONE})

    def inv(charge):
        return Element({_bps_to_pent_label(tuple(charge)): _ONE})

    return KAlgebraIso(P, B, fwd, inv, name="PentagonSample ≅ BPS(A₂-quiver)")


# Canonical pentagon labels (avoid the redundant `(i, 0, b)` forms, which
# canonicalize to a different representative of the same element).
_PENT_LABELS = (
    [(0, 0, 0)]
    + [(i, 1, 0) for i in range(5)]
    + [(i, a, 0) for i in range(5) for a in (2, 3)]
    + [(i, a, b) for i in range(5) for a in (1, 2) for b in (1, 2)]
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pentagon_spec():
    """Pentagon from its BPS quiver (spec mode): known values, axioms,
    orthonormality, and trace truncation-stability with no warnings."""
    B = BPSKAlgebra(pairing=PENTA_PAIRING, node_charges=PENTA_NODES)

    # Known structure constant + Schur index.
    assert _nm(B.multiply((1, 0), (0, 1))) == {(1, 1): "q"}, _nm(B.multiply((1, 0), (0, 1)))
    assert _ser(B.inner_product((1, 0), (1, 0), K=6), 6) == \
        {0: "1", 2: "-1", 4: "1", 6: "1"}, _ser(B.inner_product((1, 0), (1, 0), 6), 6)

    # Axiom battery (the four canonical-basis axioms).
    bat = B.verify_canonical_basis(K=6)
    for ax in ("unital", "multiplicative", "bar_invariant", "orthonormality"):
        assert bat[ax], ("axiom", ax, bat)

    # Full battery on a label grid: orthonormality, the F·S discovery
    # relation, bar involution, ρ²-twisted cyclicity, and the two faces of
    # the pairing (the sharp Schur formula vs multiply-then-trace) — the
    # last three previously never ran on any BPS instance in this gate.
    sm = [(0, 0), (1, 0), (0, 1), (1, 1), (2, -1)]
    for a in sm:
        if a != (0, 0):
            assert B.verify_F_S_leading(a), ("F·S leading", a)
        for b in sm:
            assert B.verify_orthonormality(a, b, 6), ("ortho", a, b)
            assert B.verify_bar_involution(a, b), ("bar", a, b)
            assert B.verify_rho_twisted_trace(a, b, 6), ("rho2-cyc", a, b)
            assert B.verify_inner_product_consistent(a, b, 6), ("ip-faces", a, b)

    # Trace truncation-stability: trace(·,6) ≡ trace(·,10) through q^6, no warnings.
    tl = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, -1), (1, -1)]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for l in tl:
            t6 = _ser(B.trace(l, 6), 6)
            assert t6 == _ser(B.trace(l, 10), 6), ("unstable", l)
        assert len(w) == 0, [str(x.message) for x in w]
    print("  PASS: test_pentagon_spec")


def test_pentagon_spec_free():
    """Demo of the spec-FREE constructors: the recursive spectrum-generator
    engine builds `S` directly from the quiver — no spec supplied.

    (i) `build_S=True` reproduces the spec-mode multiply / trace / inner_product
        on the generator sector (spec extraction recovers the finite spec from
        the recursive S, so this certifies the extraction path).
    (ii) `spec_free_sigma="principled"` with `build_S=True, extract_spec=False`
        — the **genuinely spec-free** engine — installs the axiom-derived σ
        (`σ⁻¹(a) = −upper(F_a)`, `σ(a) = −upper(F̃_a)`; the principled spec-free
        half-monodromy) and reproduces the spec-mode ρ / ρ⁻¹ / multiply.
        (Previously this part passed `spec_free_sigma="principled"` without
        `build_S=True`, which the constructor silently ignored — the check
        compared spec mode against itself.  The constructor now refuses the
        inert combination, asserted in (iii).)

    (A demo, not a full grid: the recursive engine is fast on the generator
    sector but not yet across higher charges.)"""
    B = BPSKAlgebra(pairing=PENTA_PAIRING, node_charges=PENTA_NODES)
    gens = [(1, 0), (0, 1)]

    # (i) spec-free S.
    Bf = BPSKAlgebra(pairing=PENTA_PAIRING, node_charges=PENTA_NODES,
                     build_S=True, build_S_cutoff=8)
    for a in gens:
        for b in gens:
            assert _nm(Bf.multiply(a, b)) == _nm(B.multiply(a, b)), ("mult", a, b)
    for l in [(0, 0), (1, 0), (0, 1)]:
        assert _ser(Bf.trace(l, 6), 6) == _ser(B.trace(l, 6), 6), ("trace", l)
    assert _ser(Bf.inner_product((1, 0), (1, 0), 6), 6) == \
        _ser(B.inner_product((1, 0), (1, 0), 6), 6)

    # (ii) principled spec-free σ — the real path (no spec extraction).
    Bp = BPSKAlgebra(pairing=PENTA_PAIRING, node_charges=PENTA_NODES,
                     build_S=True, extract_spec=False,
                     spec_free_sigma="principled", build_S_cutoff=8)
    assert Bp._spec_free, "principled-σ instance must be on the spec-free path"
    labs = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for a in labs:
        assert Bp.rho(a) == B.rho(a), ("principled rho", a)
        assert Bp.rho_inverse(a) == B.rho_inverse(a), ("principled rho_inv", a)
        for b in labs:
            assert _nm(Bp.multiply(a, b)) == _nm(B.multiply(a, b)), ("principled mult", a, b)

    # (iii) the inert combination is refused loudly.
    for kwargs in (dict(spec_free_sigma="principled"),
                   dict(spec_free_sigma="principled", build_S=True,
                        build_S_cutoff=8)):   # extraction recovers the spec
        try:
            BPSKAlgebra(pairing=PENTA_PAIRING, node_charges=PENTA_NODES,
                        **kwargs)
            raise AssertionError(
                f"inert spec_free_sigma combination accepted: {kwargs}")
        except ValueError:
            pass
    print("  PASS: test_pentagon_spec_free (incl. genuine principled-σ path)")


def test_pentagon_iso():
    """KAlgebraIso BPS-pentagon ↔ Step-1 PentagonSampleKAlgebra (verify_all):
    the two presentations are the same abstract algebra."""
    P = PentagonSampleKAlgebra()
    B = BPSKAlgebra(pairing=PENTA_PAIRING, node_charges=PENTA_NODES)
    iso = _pentagon_iso(P, B)
    # Round-trip both label maps on the canonical labels.
    for l in _PENT_LABELS:
        assert _bps_to_pent_label(_support_charge(B, *l)) == l, ("round-trip", l)
    src = [Element({l: _ONE}) for l in _PENT_LABELS]
    pairs = [(Element({a: _ONE}), Element({b: _ONE}))
             for a in _PENT_LABELS[:10] for b in _PENT_LABELS[:10]]
    res = iso.verify_all(src, [iso.map(s) for s in src],
                         pairs, [(iso.map(x), iso.map(y)) for x, y in pairs],
                         trace_K=6)
    for chk in ("unit", "round_trip", "multiplicative",
                "rho_equivariant", "trace_equivariant"):
        assert res[chk], (chk, res)
    print("  PASS: test_pentagon_iso")


def test_hexagon_flavoured():
    """A flavoured BPS theory: the hexagon (`ker B = (1,1,1)`, one U(1)
    flavour). Coefficient ring, the R-form view, and the flavoured trace."""
    H = BPSKAlgebra(
        pairing=[[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
        node_charges=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    )
    R = H.coefficient_ring()
    assert R.__class__.__name__ == "AbelianZPlusRing" and R.rank == 1, R

    # Z-form multiply carries the flavour shift in the labels; to_R_form folds
    # it onto a μ-monomial R-coefficient.
    m = H.multiply((1, 0, 0), (0, 1, 0))
    rform = _nm(H.to_R_form(m))
    assert rform == {(0, 0, 0): "[(1,)]", (1, 1, 0): "q"}, rform

    # Flavoured trace: vacuum coefficient is 1, q^2 carries the μ^{±1} content.
    tr = _ser(H.trace((0, 0, 0), K=4), 4)
    assert tr[0] == "1", tr
    assert tr[2] == "[(-1,)] + 1 + [(1,)]", tr
    print("  PASS: test_hexagon_flavoured")


def test_directional_nodedrop():
    """Bonus: a node-deletion RG flow (`DirectionalSingleNodeRG`) certified
    against an independently built UV `BPSKAlgebra` (closed-form S_RG ≡
    extraction, derived multiply / ρ / trace, and the identity-label iso)."""
    D = DirectionalSingleNodeRG(
        PENTA_PAIRING, PENTA_NODES, PENTA_STUFF_FIRST, gamma_drop=(0, 1),
    )
    UV = BPSKAlgebra(pairing=PENTA_PAIRING, node_charges=PENTA_NODES,
                     spec=PENTA_STUFF_FIRST, verify="off")
    rep = certify_directional_vs_bps(D, UV, trace_K=5)
    failures = {k: v for k, v in rep.items() if k != "iso" and v is not True}
    assert not failures, failures
    print("  PASS: test_directional_nodedrop")


def test_atlas():
    """The `BPSAtlas` layer: an ensemble of `BPSKAlgebra` charts with
    automated, certified `KAlgebraIso` transition maps across mutation chains.

    Each step is a quiver/cluster mutation, and the atlas certifies it preserves the
    *whole* `K_𝖖` structure (multiply, ρ, **and the Schur index**), with the
    rotation monodromy closing to `ρ²`.  `certificate()` runs, over the pentagon
    rotation chamber chain: the per-edge full battery, `multiply` and
    **Schur-index** chart-invariance (`I_a` identical in every chamber), and the
    `ρ² = monodromy` closure — the axiomatics-vs-cluster demonstration."""
    At = BPSAtlas(BPSKAlgebra(pairing=PENTA_PAIRING, node_charges=PENTA_NODES))

    # the bare minimum: an automated, full-battery-certified iso across a mutation
    key, iso = At.mutate_head(())
    se = [Element({l: _ONE}) for l in [(0, 0), (1, 0), (0, 1), (1, 1)]]
    te = [iso.map(e) for e in se]
    battery = iso.verify_all(
        se, te, [(se[1], se[2])], [(te[1], te[2])], trace_K=6)
    assert all(battery.values()), ("mutation iso battery", battery)

    # the certificate (per-edge battery + chart-invariance + ρ² monodromy)
    cert = At.certificate(trace_K=6)
    assert cert["all_ok"], cert
    assert cert["multiply_chart_invariant"] and cert["trace_chart_invariant"]
    assert cert["monodromy"]["is_rho2"]

    # intrinsic memoization: the atlas trace == the chart trace, cached
    A = At.root
    assert At.trace((1, 0), 6) == A.trace((1, 0), 6)

    # loop-aware atlas: the pentagon's chart graph carries a loop whose
    # automorphism is ρ², discovered from the loop and certified.
    auts = At.discover_automorphisms(max_depth=6)
    assert len(auts) == 1, ("loop automorphisms", len(auts))
    aut_img = tuple(next(iter(auts[0].map(Element({l: _ONE})).terms))
                    for l in [(0, 0), (1, 0), (0, 1), (1, 1)])
    assert aut_img == tuple(A.rho(A.rho(l))
                            for l in [(0, 0), (1, 0), (0, 1), (1, 1)]), aut_img

    # the one-call catalogue record
    s = At.summary(trace_K=6)
    assert s["all_ok"] and s["monodromy_is_rho2"] and s["n_loop_automorphisms"] == 1
    print(f"  PASS: test_atlas (rotation period {cert['period']}, "
          f"Schur index chart-invariant, monodromy = ρ², "
          f"{s['n_loop_automorphisms']} loop automorphism = ρ²)")


def test_atlas_examples():
    """The Argyres–Douglas example gallery (`bps_atlas_examples`) + the
    **mutation-complete** folded atlas.  Every finite-type example completes
    (rotation cycle) and mutation-completes (folded by chart-iso); the pentagon
    is a single chart with two outgoing mutation self-loops."""
    import bps_atlas_examples as g
    # mutation_complete: the pentagon folds to one chart, two self-loops
    mc = BPSAtlas(BPSKAlgebra(pairing=PENTA_PAIRING,
                              node_charges=PENTA_NODES)).mutation_complete()
    assert mc["n_charts"] == 1 and mc["self_loops"] == 2 and mc["closed"], mc
    # the gallery completes the pentagon (rotation cycle) and folds [A1,A3]
    _, comp = g.complete_atlas("pentagon")
    assert comp["n_charts"] == 4 and comp["closed"], comp
    _, m3 = g.mutation_complete_atlas("a3")
    assert m3["n_charts"] == 4 and m3["closed"] and m3["rank_regular"], m3
    print(f"  PASS: test_atlas_examples (pentagon→1 chart/2 self-loops, "
          f"[A1,A3]→4 charts, gallery complete+mutation-complete)")


def test_gauge_atlas_examples():
    """The gauge-theory gallery (`bps_atlas_gauge_examples`): SU(2)/A₁ class-S
    theories are mutation-finite (the fold closes — and SU(2)-gauged [A₁,Dₙ] gives
    the Catalan numbers), while SU(3) is mutation-infinite (the fold runs away and
    the chambers are mostly wild — no reasonable S)."""
    import bps_atlas_gauge_examples as ge
    recs = {r["theory"]: r for r in ge.su2_mutation_finite_counts()}
    # every SU(2) example closes (mutation-finite), and the Dₙ chain is Catalan
    assert all(r["closed"] and r["rank_regular"] for r in recs.values())
    assert recs["pure_SU2"]["n_charts"] == 1 and recs["SU2_cubed"]["n_charts"] == 138
    assert all(recs[f"SU2gA1D{n}"]["n_charts"] == ge.CATALAN_A1DN[n]
               for n in (3, 4, 5, 6)), "SU(2)-gauged [A1,Dn] = Catalan(n)"
    # SU(3): mutation_complete does NOT close (infinite orbit), arrows blow up,
    # and wild chambers (no finite spec ⇒ no reasonable S) appear by depth 2
    run = ge.su3_mutation_runaway(500)
    assert run["n_charts"] == 500 and not run["closed"], run
    growth = ge.su3_arrow_growth(8)
    assert growth[0] == 2 and growth[-1] > 1000 and growth == sorted(growth), growth
    cen = ge.su3_chamber_census(max_depth=2)
    assert cen["n_wild"] > 0 and cen["n_finite"] > 0, cen
    # the recursive direct-S finder itself is cutoff-dependent in a wild chamber
    # (drifts 2 → 4), so it finds no reasonable S — the same verdict as the census
    drift = ge.su3_recursive_S_drift((2, 4))
    assert drift[0][1] != drift[1][1], "recursive S should drift (no convergent S)"
    # restricting to charts where S-finding works folds the infinite wild orbit to
    # a finite 2-chart atlas (the tame core), with 2 wild walls
    fa = ge.su3_finite_spec_atlas()
    assert fa["n_charts"] == 2 and fa["walls"] == 2 and fa["closed"], fa
    print(f"  PASS: test_gauge_atlas_examples (SU(2) mutation-finite incl. "
          f"Catalan; SU(3) infinite, {cen['n_wild']}/{cen['n_chambers']} wild, "
          f"recursive-S drifts; finite-spec atlas → {fa['n_charts']} charts)")


# ---------------------------------------------------------------------------
# the S- and F·S-building engines (the Step-4 increment)
# ---------------------------------------------------------------------------

RAY_H0 = HabiroElement.zero()
KRONECKER2 = [[0, 2], [-2, 0]]
B3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
# Markov (= N=2*): the three-cycle with every bracket 2.  It admits no finite
# `E_𝖖`-product, which is exactly why it is here.
MARKOV = [[0, 2, -2], [-2, 0, 2], [2, -2, 0]]
CYCLE111 = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
# Weak coupling for pure SU(2): the central charge that selects the chamber with
# the dyon tower and the W boson, rather than the two-state strong-coupling one.
WEAK_SU2 = [complex(-1, 1), complex(1, 1)]


def _same_in_cone(A, B, theory):
    """Equality of two `S`'s on the positive cone only.

    Both builds are truncated along the CONE, not in `𝖖`, and they need not agree
    on charges outside it — outside is where each one's own truncation debris
    lives.  Comparing there would fail for reasons that say nothing about `S`.
    """
    return all(A.get(g, RAY_H0) == B.get(g, RAY_H0)
               for g in set(A) | set(B) if theory.in_cone(g))


def _n_factors(content):
    """How many BPS factors `E^{(s)}_𝖖(X_γ)^{Ω(γ,s)}` a content carries.

    NOT the number of charges: a charge whose `Ω` spans several spins carries one
    factor per spin, and the two counts genuinely differ (the pentagon's free
    order gives 14 charges but 19 factors).  The factor is the unit the order
    places, so it is the unit worth counting.
    """
    return sum(len(spin_decompose(om)) for om in content.values())


def test_bps_factor_spectrum():
    """`bps_factor_spectrum`: `S` from its LEADING DATA, no spec and no `F`-solve.

    Four independent checks, in increasing order of what they would catch:

    (i)   against `Theory.S_from_spec` — ground truth.  That routine expands the
          ordered product `∏ E_𝖖(X_{γ_i})` of a KNOWN chamber spec as a Nahm sum
          and shares no code path with this recursion, so agreement is
          evidence rather than self-consistency.
    (ii)  order-independence.  The recursion needs a total order on the PAIRS
          `(γ, s)` — one position per BPS factor `E^{(s)}_𝖖(X_γ)^{Ω(γ,s)}` — and
          `S` does not depend on the choice, though the CONTENT `Ω` does.  Both
          halves are asserted: if every order produced the same number of BPS
          factors the first half would be testing nothing.
    (iii) the central charge selects the chamber.  Supplying weak-coupling
          phases for pure SU(2) reproduces the physical spectrum: the dyon tower
          plus the W boson, which is the one state of nonzero spin.  Same `S` as
          the two-factor strong-coupling factorisation.  This is the one place a
          central charge is in play, and so the one place charges organise
          into rays at all.
    (iv)  against the chart's own `[S|0⟩]_γ`, through
          `BPSKAlgebra.verify_spectrum_generator_from_factors`.

    Then the reason the engine exists: COVERAGE.  On Markov (= N=2*) the peel
    recursion's monomial-charge gate trips and it honest-fails; this construction
    has no gate to trip and builds.
    """
    theory = Theory("pentagon", PENTA_PAIRING, PENTA_NODES, CONE=8)

    # (i) ground truth
    S = build_spectrum_generator_from_factors(PENTA_PAIRING, PENTA_NODES, 8)
    assert _same_in_cone(S, theory.S_from_spec([(1, 0), (0, 1)]), theory)

    # (ii) order-independence, non-vacuously
    built = {}
    for order in PLACEMENT_ORDERS:
        try:
            b = BPSFactorSpectrum(PENTA_PAIRING, PENTA_NODES, 8, order=order)
        except ValueError:
            continue          # `strip` honest-fails where the strip leaves a core
        b.run()
        built[order] = b
    assert len(built) >= 3, list(built)
    ref = built[list(built)[0]].spectrum_generator()
    for order, b in built.items():
        assert _same_in_cone(ref, b.spectrum_generator(), theory), order
    # Count the PLACED UNIT — the BPS factors `E^{(s)}_𝖖(X_γ)^{Ω(γ,s)}`, one per
    # `(γ, s)` pair — not the charges carrying one.  The two differ (the free
    # order here gives 14 charges but 19 factors), and it is the factor count
    # that the order actually moves.
    counts = {o: _n_factors(b.multiplicities()) for o, b in built.items()}
    assert len(set(counts.values())) > 1, (
        f"every order gave {counts} BPS factors — order-independence is vacuous "
        f"here")

    # (iii) the central charge chooses the chamber
    strong = BPSFactorSpectrum(KRONECKER2, PENTA_NODES, 6, order="phase")
    strong.run()
    weak = BPSFactorSpectrum(KRONECKER2, PENTA_NODES, 6, order="phase",
                       phases=WEAK_SU2)
    weak.run()
    assert len(strong.omega) == 2, strong.omega
    su2 = Theory("su2", KRONECKER2, PENTA_NODES, CONE=6)
    assert _same_in_cone(strong.spectrum_generator(),
                         weak.spectrum_generator(), su2)
    spins = {g: spin_decompose(o) for g, o in weak.multiplicities().items()}
    assert spins[(1, 1)] == {1: 1}, spins            # the W boson, spin 1/2
    assert all(spins[g] == {0: 1} for g in spins if g != (1, 1)), spins

    # (iv) against the chart's own S
    A = BPSKAlgebra(pairing=PENTA_PAIRING, node_charges=PENTA_NODES)
    assert A.verify_spectrum_generator_from_factors(6)

    # coverage: the peel gate trips, this construction builds
    with enable_retired_peel_engine():
        try:
            build_spectrum_generator(MARKOV, B3, 5)
            raise AssertionError("the peel engine should gate out on Markov")
        except RetiredEngineError:                          # pragma: no cover
            raise AssertionError(
                "got the RETIREMENT error, so the gate was never reached")
        except ValueError as exc:
            assert "monomial-charge gate" in str(exc), str(exc)
    markov = BPSFactorSpectrum(MARKOV, B3, 5)
    markov.run()
    assert markov.omega, "the construction produced no factors on Markov"

    print(f"  PASS: test_bps_factor_spectrum (ground truth; {len(built)} orders give "
          f"one S with BPS-factor counts {sorted(set(counts.values()))}; weak-coupling "
          f"SU(2) = dyon tower + W boson; Markov builds where the peel gate trips)")


def test_peel_engine_retired():
    """The peel `S`-recursion is RETIRED — and kept, switchable, on purpose.

    Retired means every way *in* refuses.  The doors are enumerated rather than
    sampled: a retirement that closes three of four entrances is not a
    retirement, since a caller arriving through the fourth would reach the engine
    silently.

    It is gated rather than deleted because it is the only INDEPENDENT
    construction of `S` in the release — it solves `F_γ·S_sub = X_γ + O(𝖖)` and
    reattaches `E_𝖖(F_γ)`, sharing no mechanism with the factor recursion — so the
    cross-check between the two has to stay runnable.  Evidence you cannot run is
    not evidence.
    """
    import recursive_spectrum as rs

    doors = (
        ("build_spectrum_generator",
         lambda: rs.build_spectrum_generator(PENTA_PAIRING, PENTA_NODES, 4)),
        ("_build_S_by_engine",
         lambda: rs._build_S_by_engine(PENTA_PAIRING, PENTA_NODES, 4,
                                       engine="peel")),
        ("build_spectrum_generator_auto",
         lambda: rs.build_spectrum_generator_auto(PENTA_PAIRING, PENTA_NODES,
                                                  engine="peel")),
        ("extract_spec_from_quiver",
         lambda: rs.extract_spec_from_quiver(PENTA_PAIRING, PENTA_NODES,
                                             cutoff=4, engine="peel")),
        ("BPSKAlgebra",
         lambda: BPSKAlgebra(pairing=PENTA_PAIRING, node_charges=PENTA_NODES,
                             build_S=True, build_S_cutoff=4,
                             build_S_engine="peel")),
    )
    for label, call in doors:
        try:
            call()
        except RetiredEngineError:
            continue
        raise AssertionError(f"{label} still reaches the retired peel engine")
    assert rs.ACTIVE_SPEC_FREE_ENGINES == ("factors",)

    # An unknown engine and a retired one must not report the same thing: the
    # retired one exists and is reachable, and the message has to say so.
    try:
        BPSKAlgebra(pairing=PENTA_PAIRING, node_charges=PENTA_NODES,
                    build_S=True, build_S_cutoff=4, build_S_engine="bogus")
    except RetiredEngineError:                              # pragma: no cover
        raise AssertionError("an unknown engine reported itself as retired")
    except ValueError:
        pass

    # Intact behind both opt-ins, and still agreeing with the factor build.
    by_kwarg = rs.build_spectrum_generator(PENTA_PAIRING, PENTA_NODES, 6,
                                           allow_retired=True)
    with enable_retired_peel_engine():
        by_context = rs._build_S_by_engine(PENTA_PAIRING, PENTA_NODES, 6,
                                           engine="peel")
    assert by_kwarg == by_context
    theory = Theory("pentagon", PENTA_PAIRING, PENTA_NODES, CONE=6)
    assert _same_in_cone(
        by_kwarg,
        build_spectrum_generator_from_factors(PENTA_PAIRING, PENTA_NODES, 6),
        theory)

    # The opt-in does not leak: the context manager restores the retirement.
    assert rs.PEEL_RETIRED is True
    try:
        rs.build_spectrum_generator(PENTA_PAIRING, PENTA_NODES, 4)
    except RetiredEngineError:
        pass
    else:
        raise AssertionError("the context manager leaked the opt-in")

    print(f"  PASS: test_peel_engine_retired ({len(doors)} doors refuse; both "
          f"opt-ins reach the intact engine and it still agrees)")


def test_fs_builder():
    """`F_γ` and `S` grown TOGETHER out of `F_γ·S = X_γ + O(𝖖)`.

    Where the shipped F-solver takes `S` as given and enumerates the
    doubly-tropical support window `[γ₋, γ⁺]`, this route has neither: at each
    cone degree it places the BPS factors the leading data forces and reads off
    the palindromic `F`-coefficients the relation forces.  So the interval is an
    OUTPUT here, which is what makes agreement with the solver evidence rather
    than a restatement.

    Four checks:

    (i)   `F` agrees with `BPSKAlgebra.F` over a grid of pentagon charges,
          positive and negative;
    (ii)  the `S` it produces on the way is the standalone `bps_factor_spectrum` build;
    (iii) the two moves are ONE alphabet — `[n]_𝖖 = χ_{(n−1)/2}`, the F-solver's
          peel basis and the `S`-side multiplicity basis being the same
          `Z`-basis of the palindromic Laurent polynomials.  This is why the two
          recursions interleave at all, so it is pinned rather than asserted;
    (iv)  the honest caveat: the relation ALONE pins nothing.  `Ω ≡ 0` gives
          `S = 1` and `F = X_γ`, which satisfies `F·S = X_γ + O(𝖖)` exactly.
          What makes the build deterministic is `S`'s leading data.
    """
    A = BPSKAlgebra(pairing=PENTA_PAIRING, node_charges=PENTA_NODES)
    charges = [(1, 0), (0, 1), (1, 1), (-1, 0), (-1, 1), (0, -1), (2, 1)]

    # (i)
    for g in charges:
        need = FSBuilder(PENTA_PAIRING, PENTA_NODES, g, 2).cutoff_needed(A)
        assert need >= 0, (g, "the shipped F charge is not γ + cone")
        b = FSBuilder(PENTA_PAIRING, PENTA_NODES, g, max(need, 1))
        b.run()
        bad = b.verify_against_solve_F(A)
        assert not bad, (g, bad[:3])

    # (ii)
    b = FSBuilder(PENTA_PAIRING, PENTA_NODES, (1, 0), 6)
    b.run()
    theory = Theory("pentagon", PENTA_PAIRING, PENTA_NODES, CONE=6)
    assert _same_in_cone(
        b.spectrum_generator(),
        build_spectrum_generator_from_factors(PENTA_PAIRING, PENTA_NODES, 6),
        theory)

    # (iii)
    for n in range(1, 8):
        chi = {e: c for e, c in QNumberPoly({n: 1}).to_laurent()._coeffs.items()
               if c}
        assert spin_decompose(chi) == {n - 1: 1}, (n, chi)

    # (iv)
    z = FSBuilder(PENTA_PAIRING, PENTA_NODES, (-1, 0), 5, order="lex",
                  omega_policy=lambda k, forced, resid: {})
    z.run()
    assert z.factors.omega == {}, z.factors.omega
    assert z.F() == {(-1, 0): LaurentPoly({0: 1})}, z.F()

    print(f"  PASS: test_fs_builder ({len(charges)} charges == the F-solver, "
          f"support an output; S byproduct == the standalone build; one alphabet; "
          f"Ω ≡ 0 shows the relation alone pins nothing)")


def test_factor_order_search():
    """Searching the ORDER for a simple factorisation of `S`.

    `S` does not depend on the order the BPS factors are placed in; its
    FACTORISATION does, and some orders factor it far more simply than others.
    This is a BFS over **BPS-factor** insertion points, looking for an order
    with spin-0 factors only that stops populating inside the cone — which is
    what a consumer needs.

    ⚠ `is_spec` is bounded, not a certificate: stopping is a screen, the result
    is confirmed by rebuilding `S` and the candidate product `confirm_extra`
    degrees further, and a factorisation can agree everywhere the check looks
    and disagree one degree past it.  So the assertion below is read relative to
    `Result.confirmed_to`, and the independent Nahm-sum rebuild is what carries
    the weight here.

    (i)  on a quiver whose source/sink strip leaves a core, the default falls
         back to random placement: a legitimate total order, but a poor
         factorisation.  The search finds a spin-0 one, and it is genuinely
         better — both halves asserted, since "fewer factors" alone could be a
         lucky draw while "spin 0" is the property that matters.
    (ii) every reported factorisation is REBUILT through `Theory.S_from_spec`,
         the independent Nahm-sum route, so a wrong ORDER is caught and not just
         a wrong multiset.
    (iii) on an acyclic quiver it short-circuits to the strip order, which
         attains the provable floor: at cone degree 1 the product is empty, so
         every node carries a factor in every order.
    """
    # (i) + (ii) a cored quiver
    default = BPSFactorSpectrum(CYCLE111, B3, 5)
    assert default.order == "random", default.order
    default.run()
    content = default.multiplicities()
    found = find_simple_factorisation(CYCLE111, B3, 5)
    assert found.is_spec and found.content.max_spin_doubled == 0, found.spec
    # spin-0 throughout, so the spec length IS the factor count on this side.
    assert len(found.spec) < _n_factors(content), (len(found.spec),
                                                   _n_factors(content))

    theory = Theory("3-cycle(1,1,1)", CYCLE111, B3, CONE=5)
    lex = BPSFactorSpectrum(CYCLE111, B3, 5, order="lex")
    lex.run()
    assert _same_in_cone(theory.S_from_spec(list(found.spec)),
                         lex.spectrum_generator(), theory)

    # (iii) acyclic: the node charges themselves, in the strip order
    pent = find_simple_factorisation(PENTA_PAIRING, PENTA_NODES, 6)
    assert pent.is_spec and list(pent.spec) == list(PENTA_NODES), pent.spec

    print(f"  PASS: test_factor_order_search (3-cycle: {_n_factors(content)} BPS factors "
          f"→ {len(found.spec)} spin-0 factors, rebuilt independently; "
          f"pentagon short-circuits to its node charges)")

def main():
    test_pentagon_spec()
    test_pentagon_spec_free()
    test_pentagon_iso()
    test_hexagon_flavoured()
    test_directional_nodedrop()
    test_atlas()
    test_atlas_examples()
    test_gauge_atlas_examples()
    test_bps_factor_spectrum()
    test_peel_engine_retired()
    test_fs_builder()
    test_factor_order_search()
    print("\nALL BPSKAlgebra (Step 4) self-tests passed.")


if __name__ == "__main__":
    main()
