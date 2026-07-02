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
                               the recursive spectrum-generator engine): multiply /
                               trace / inner_product reproduce the spec-mode values
                               over a label grid.
  * `test_pentagon_iso`      — a `KAlgebraIso` BPS-pentagon ↔ the Step-1
                               `PentagonSampleKAlgebra` (`verify_all`): the
                               cross-check that the two presentations are the same
                               abstract algebra (multiplicative + trace-equivariant).
  * `test_hexagon_flavoured` — a flavoured theory (the hexagon, `ker B = (1,1,1)`,
                               one U(1) flavour): `coefficient_ring`, `to_R_form`,
                               and the flavoured trace over `R((q))`.
  * `test_directional_nodedrop` — a node-deletion RG flow (`DirectionalSingleNodeRG`)
                               certified against an independent UV `BPSKAlgebra`.

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
from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from samples import PentagonSampleKAlgebra
from bps_kalgebra import BPSKAlgebra
from directional_subquiver_rg import (
    DirectionalSingleNodeRG,
    certify_directional_vs_bps,
)
from bps_atlas import BPSAtlas

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

    # Orthonormality on a label grid.
    sm = [(0, 0), (1, 0), (0, 1), (1, 1), (2, -1)]
    for a in sm:
        for b in sm:
            assert B.verify_orthonormality(a, b, 6), ("ortho", a, b)

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
        on the generator sector.
    (ii) `spec_free_sigma="principled"` installs the axiom-derived σ
        (`σ⁻¹(a) = −upper(F_a)`, `σ(a) = −upper(F̃_a)`; the principled spec-free
        half-monodromy) and reproduces the spec-mode ρ / ρ⁻¹ / multiply.

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

    # (ii) principled spec-free σ.
    Bp = BPSKAlgebra(pairing=PENTA_PAIRING, node_charges=PENTA_NODES,
                     spec_free_sigma="principled", build_S_cutoff=8)
    labs = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for a in labs:
        assert Bp.rho(a) == B.rho(a), ("principled rho", a)
        assert Bp.rho_inverse(a) == B.rho_inverse(a), ("principled rho_inv", a)
        for b in labs:
            assert _nm(Bp.multiply(a, b)) == _nm(B.multiply(a, b)), ("principled mult", a, b)
    print("  PASS: test_pentagon_spec_free")


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


def main():
    test_pentagon_spec()
    test_pentagon_spec_free()
    test_pentagon_iso()
    test_hexagon_flavoured()
    test_directional_nodedrop()
    test_atlas()
    test_atlas_examples()
    test_gauge_atlas_examples()
    print("\nALL BPSKAlgebra (Step 4) self-tests passed.")


if __name__ == "__main__":
    main()
