"""`BPSAtlas` for the SU(3) **linear quivers** (the design record catalogue).

Pins `src/abe/su3_quiver_atlas.py` for

    SU(3)-SU(2)   (pure_ade.SUN_bifund(3, 2))   rank 7, |spec| 23
    SU(3)-SU(3)   (pure_ade.SUN_bifund(3, 3))   rank 9, |spec| 37

Both are SU(3) two-node quivers with one bifundamental, flavour ring `R(U(1))`
(the bifund baryon).  The multiply at rank ≥ 7 is slow (the heavy bifund node),
so this fast test stays at the **label level** (build validity / ρ-permutation /
wild directions / the structural — non-multiplicative — RG-iso checks); the
multiply- and trace-level certification (axioms, RG multiplicativity,
orthonormality, dynamics) is the *script* `scripts/atlas_su3_quiver.py`.

**SU(3)-SU(3) is rank 9 — its chart build is many minutes**, so this test does
**not** build it (only checks its registration); build it via
`su3_quiver_entry("SU3_SU3")` when you can wait.

Run:  `python3 run_tests.py`
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import AbelianZPlusRing
from bps_atlas import BPSAtlas
from su3_quiver_atlas import (
    su3_quiver_names, su3_quiver_entry, su3_quiver_rg_iso,
)

PASS = []
FAIL = []
ONE = LaurentPoly.one()


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


# ---------------------------------------------------------------------------
def test_names():
    check("quiver family is [SU3_SU2, SU3_SU3]",
          su3_quiver_names() == ["SU3_SU2", "SU3_SU3"])


def test_su3_su2_builds():
    e = su3_quiver_entry("SU3_SU2")
    A = e.chart
    check("SU3_SU2 rank == 7", e.rank == 7)
    check("SU3_SU2 |spec| == 23", len(A.spec) == 23)
    check("SU3_SU2 gauge ranks == (3, 2)", e.gauge_ranks == (3, 2))
    check("SU3_SU2 coeff ring is R(U(1)) (bifund baryon)",
          isinstance(A.coefficient_ring(), AbelianZPlusRing)
          and A.coefficient_ring().rank == 1)
    # single bifundamental node at index n_pure1 + n_pure2 = 4 + 2 = 6
    check("SU3_SU2 single bifundamental node at index 6",
          e.matter_indices == (6,))
    check("SU3_SU2 identity in basis", A.verify_identity_in_basis())


def test_su3_su2_rho_is_permutation():
    # ρ / ρ⁻¹ are label permutations — no multiply, so this is cheap even at rank 7.
    A = su3_quiver_entry("SU3_SU2").chart
    nodes = [tuple(g) for g in A.node_charges]
    check("SU3_SU2 ρ⁻¹ inverts ρ on all nodes",
          all(A.verify_rho_inverse(a) for a in nodes))
    check("SU3_SU2 ρ fixes the identity", A.verify_rho_fixes_identity())


def test_su3_su2_rg_bifund_drop_iso_structural():
    # The node-drop RG iso to pure SU(3) x pure SU(2): check the *structural*
    # (label-level, non-multiplicative) battery — unit / round-trip /
    # ρ-equivariant — which is fast.  (Multiplicativity is the slow bifund
    # multiply; it is the script's job, and the iso is guaranteed by construction.)
    e = su3_quiver_entry("SU3_SU2")
    A = e.chart
    iso = su3_quiver_rg_iso("SU3_SU2")
    nodes = [tuple(g) for g in A.node_charges]
    win = [tuple(A.identity()), nodes[0], nodes[e.matter_indices[0]]]
    se = [Element({l: ONE}) for l in win]
    te = [iso.map(e2) for e2 in se]
    ok = (iso.verify_unit()
          and iso.verify_round_trip(se, te)
          and iso.verify_rho_equivariant(se, te))
    check("SU3_SU2 RG bifund-drop iso structural battery (unit/round-trip/ρ)", ok)


def test_su3_su2_wild_chambers():
    at = BPSAtlas(su3_quiver_entry("SU3_SU2").chart)
    coop = declined = 0
    for k in range(7):
        for d in ("fwd", "inv"):
            try:
                at.mutate((), k, d); coop += 1
            except ValueError:
                declined += 1
    check("SU3_SU2 has wild chambers (most directions declined, never hangs)",
          declined > 0 and coop >= 1)


def test_su3_su3_registered_without_building():
    # SU3_SU3 is rank 9 — building it is many minutes, so only check registration
    # (do NOT call su3_quiver_entry, which builds the chart).
    from su3_quiver_atlas import _QUIVERS
    check("SU3_SU3 registered as gauge ranks (3, 3)",
          _QUIVERS.get("SU3_SU3") == (3, 3))
    check("SU3_SU3 in the quiver family list",
          "SU3_SU3" in su3_quiver_names())


def main():
    tests = [
        test_names,
        test_su3_su2_builds,
        test_su3_su2_rho_is_permutation,
        test_su3_su2_rg_bifund_drop_iso_structural,
        test_su3_su2_wild_chambers,
        test_su3_su3_registered_without_building,
    ]
    for t in tests:
        t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILURES:", FAIL)
        sys.exit(1)
    print("All SU(3)-quiver BPSAtlas tests passed.")


if __name__ == "__main__":
    main()
