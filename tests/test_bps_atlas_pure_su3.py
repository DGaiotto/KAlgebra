"""`BPSAtlas` for **pure SU(3)** super-Yang–Mills (the design record catalogue).

Pins `src/abe/pure_su3_atlas.py` and the atlas surface for the corner the
gauge catalogue deferred.  Certifies:

  * the strong-coupling chart builds (rank 4, unflavoured, |spec|=6) and passes
    the contract axioms + low-order orthonormality;
  * the canonical rotation orbit closes with **period 12** and rotation
    **monodromy = ρ²**, with ρ² a *non-trivial* drift (ρ is infinite-order — the
    Witten effect, user-confirmed 2026-06-28);
  * one per-edge `KAlgebraIso` structural battery + a multiply chart-invariance
    sample (the deep-chamber full sweep is slow — that is the script);
  * chart-iso recognition folds the rotation orbit (`fold_ratio > 1`);
  * **the wild-chamber boundary** — the atlas *declines* the wild mutation
    directions (raises, never hangs), a wild chamber has **no finite spec**, and
    the spec-free direct-`S` build is **non-convergent** there; the depth-2
    chamber census has both finite and wild chambers.

Run:  `python3 run_tests.py`
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing
from bps_atlas import BPSAtlas
from pure_su3_atlas import (
    pure_su3_chart, pure_su3_atlas, pure_su3_chamber_census,
    mutate_quiver, chamber_has_finite_spec, pure_su3_spec_free_chart,
    WILD_CHAMBER_PATHS,
)

PASS = []
FAIL = []
ONE = LaurentPoly.one()


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


def _struct_battery(iso, labels):
    se = [Element({tuple(l): ONE}) for l in labels]
    te = [iso.map(e) for e in se]
    pairs = [(se[i], se[j]) for i in range(len(se)) for j in range(len(se))]
    return {
        "unit": iso.verify_unit(),
        "round_trip": iso.verify_round_trip(se, te),
        "rho": iso.verify_rho_equivariant(se, te),
        "multiplicative": iso.verify_multiplicative(
            pairs, [(iso.map(a), iso.map(b)) for a, b in pairs]),
    }


# ---------------------------------------------------------------------------
def test_chart_builds():
    A = pure_su3_chart()
    check("pure SU(3) chart has rank 4",
          len(A.lattice.pairing) == 4)
    check("pure SU(3) is unflavoured (TrivialZPlusRing)",
          isinstance(A.coefficient_ring(), TrivialZPlusRing))
    check("strong-coupling spec has length 6", len(A.spec) == 6)
    check("identity is in the basis", A.verify_identity_in_basis())


def test_axioms():
    A = pure_su3_chart()
    nodes = [tuple(g) for g in A.node_charges]
    check("ρ is an automorphism on node pairs",
          all(A.verify_rho_is_automorphism(a, b) for a in nodes for b in nodes))
    check("bar involution on node pairs",
          all(A.verify_bar_involution(a, b) for a in nodes for b in nodes))
    check("ρ⁻¹ inverts ρ on nodes",
          all(A.verify_rho_inverse(a) for a in nodes))
    check("ρ fixes the identity", A.verify_rho_fixes_identity())


def test_orthonormality_low_order():
    A = pure_su3_chart()
    nodes = [tuple(g) for g in A.node_charges]
    R = A.coefficient_ring()
    one, zero = R.one(), R.zero()
    # diagonal: I_{a,a}[q^0] == 1  (fast — collinear)
    diag = all(A.inner_product(a, a, K=4)[0] == one for a in nodes)
    check("orthonormality diagonal I_{a,a}[q^0] == 1", diag)
    # one off-diagonal: I_{a,b}[q^0] == 0  (the slow direction — sample one pair)
    off = A.inner_product(nodes[0], nodes[1], K=3)[0] == zero
    check("orthonormality off-diagonal I_{a,b}[q^0] == 0 (sampled)", off)


def test_rotation_period_and_monodromy():
    at = pure_su3_atlas()
    labels = [tuple(g) for g in at.root.node_charges]
    keys = at.rotation_chambers(max_steps=24)
    check("canonical rotation closes with period 12", len(keys) - 1 == 12)

    mono = at.monodromy()
    is_rho2 = all(
        next(iter(mono.map(Element({l: ONE})).terms)) == at.rho(at.rho(l))
        for l in labels)
    check("rotation monodromy == ρ² on node labels", is_rho2)
    # ρ² is a non-trivial drift  =>  ρ is infinite-order (Witten effect)
    check("ρ² is non-trivial (ρ infinite-order, AF gauge theory)",
          any(at.rho(at.rho(l)) != l for l in labels))


def test_edge_battery_and_chart_invariance():
    at = pure_su3_atlas()
    labels = [tuple(g) for g in at.root.node_charges]
    keys = at.rotation_chambers(max_steps=24)
    batt = _struct_battery(at.iso(keys[0], keys[1]), labels)
    check("per-edge structural KAlgebraIso battery (edge 0)", all(batt.values()))
    # multiply chart-invariance on the first interior chamber (one pair)
    a, b = labels[0], labels[2]
    ck = keys[1]; ch = at.chart(ck)
    a_c = next(iter(at.transport(a, (), ck).terms))
    b_c = next(iter(at.transport(b, (), ck).terms))
    inv = at.iso(ck, ()).map(ch.multiply(a_c, b_c)) == at.root.multiply(a, b)
    check("multiply chart-invariant (sampled chamber)", inv)


def test_folded_graph():
    at = pure_su3_atlas()
    fg = at.folded_graph(max_steps=12)
    check("chart-iso recognition folds the rotation orbit (fold_ratio > 1)",
          fg["fold_ratio"] > 1.0 and fg["n_classes"] >= 1)


def test_wild_chamber_atlas_declines():
    # The spec-based chart graph declines the wild mutation directions at
    # max_local_moves=0 — it RAISES (never hangs), keeping the atlas in the
    # finite chamber.  Node-0-fwd cooperates (spec head); node-1-fwd is wild.
    at = pure_su3_atlas()
    declined = False
    try:
        at.mutate((), 1, "fwd")
    except ValueError as e:
        declined = "does not cooperate" in str(e)
    check("atlas declines a wild mutation direction (raises, no hang)", declined)
    # a cooperating direction still works
    ok = True
    try:
        at.mutate((), 0, "fwd")
    except Exception:
        ok = False
    check("a spec-cooperating direction still mutates", ok)


def test_wild_chamber_no_finite_spec():
    Qw = mutate_quiver(WILD_CHAMBER_PATHS[0])
    ns = chamber_has_finite_spec(Qw, spec_depth=20)
    check(f"wild chamber {WILD_CHAMBER_PATHS[0]} has NO finite spec (depth 20)",
          ns is None)


def test_wild_chamber_spec_free_non_convergent():
    Qw = mutate_quiver(WILD_CHAMBER_PATHS[0])
    wnodes = [tuple(c) for c in Qw.charges]
    m2 = pure_su3_spec_free_chart(wnodes, 2).multiply(wnodes[0], wnodes[1])
    m4 = pure_su3_spec_free_chart(wnodes, 4).multiply(wnodes[0], wnodes[1])
    check("spec-free direct-S is cutoff-dependent in a wild chamber "
          "(non-convergent; cutoff 2 ≠ 4)", m2 != m4)


def test_chamber_census():
    cen = pure_su3_chamber_census(max_depth=2, spec_depth=12)
    check("depth-2 census finds both finite-spec and wild chambers",
          cen["n_finite"] > 0 and cen["n_wild"] > 0)
    check("wild chambers appear by depth 2 (the author's warning)",
          any(len(p) == 2 for p in cen["wild_paths"]))


def test_intrinsic_memoization():
    at = pure_su3_atlas()
    a = tuple(at.root.node_charges[0])
    b = tuple(at.root.node_charges[1])
    at.clear_intrinsic_caches()
    first = at.multiply(a, b)
    cached = at.multiply(a, b)
    check("intrinsic multiply memoization consistent (cached == fresh)",
          first == cached)


def main():
    tests = [
        test_chart_builds,
        test_axioms,
        test_orthonormality_low_order,
        test_rotation_period_and_monodromy,
        test_edge_battery_and_chart_invariance,
        test_folded_graph,
        test_wild_chamber_atlas_declines,
        test_wild_chamber_no_finite_spec,
        test_wild_chamber_spec_free_non_convergent,
        test_chamber_census,
        test_intrinsic_memoization,
    ]
    for t in tests:
        t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILURES:", FAIL)
        sys.exit(1)
    print("All pure-SU(3) BPSAtlas tests passed.")


if __name__ == "__main__":
    main()
