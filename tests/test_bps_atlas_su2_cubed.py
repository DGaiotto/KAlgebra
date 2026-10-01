"""`BPSAtlas` for the **SU(2)³ linear quiver** (SU(2)-SU(2)-SU(2), the design record).

Pins `src/bps/su2_cubed_atlas.py` — the 3-node extension of the SU(2)-SU(2)
entry, whose BPS quiver is assembled directly as its Dirac matrix (three
Kronecker-2 gauge pairs + two bifundamental nodes) with a frozen length-16
negating sequence.  Rank 8, coeff `R(U(1)²)` (one baryon per bifundamental).

The multiply at rank 8 with the heavy bifundamental nodes is slow (minutes), so
this fast test stays at the **label level** (build validity / ρ-permutation / the
structural — non-multiplicative — node-drop RG iso / wild directions); the
multiply- and trace-level certification (axioms, orthonormality, dynamics) is the
*script* `scripts/atlas_su2_cubed.py`.

Run:  `python3 run_tests.py`
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import AbelianZPlusRing
from bps_atlas import BPSAtlas
from su2_cubed_atlas import (
    su2_cubed_chart, su2_cubed_rg_iso, BIFUND_INDICES, SU2_CUBED_NEG_SEQ,
)

PASS = []
FAIL = []
ONE = LaurentPoly.one()


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


# ---------------------------------------------------------------------------
def test_chart_builds():
    A = su2_cubed_chart()
    check("SU(2)^3 rank == 8", len(A.lattice.pairing) == 8)
    check("SU(2)^3 |spec| == 16", len(A.spec) == 16)
    check("SU(2)^3 coeff ring is R(U(1)^2) (two bifund baryons)",
          isinstance(A.coefficient_ring(), AbelianZPlusRing)
          and A.coefficient_ring().rank == 2)
    check("SU(2)^3 two bifundamental nodes at indices (6, 7)",
          BIFUND_INDICES == (6, 7))
    check("SU(2)^3 frozen negating sequence has length 16",
          len(SU2_CUBED_NEG_SEQ) == 16)
    check("SU(2)^3 identity in basis", A.verify_identity_in_basis())


def test_rho_is_permutation():
    # ρ / ρ⁻¹ are label permutations (no multiply) — cheap even at rank 8.
    A = su2_cubed_chart()
    nodes = [tuple(g) for g in A.node_charges]
    check("SU(2)^3 ρ⁻¹ inverts ρ on all nodes",
          all(A.verify_rho_inverse(a) for a in nodes))
    check("SU(2)^3 ρ fixes the identity", A.verify_rho_fixes_identity())


def test_rg_bifund_drop_iso_structural():
    # node-drop both bifunds → pure SU(2)³.  Structural (label-level) battery —
    # unit / round-trip / ρ-equivariant — is fast; multiplicativity (heavy bifund
    # multiply) is the script's job and the iso is guaranteed by construction.
    A = su2_cubed_chart()
    iso = su2_cubed_rg_iso()
    nodes = [tuple(g) for g in A.node_charges]
    win = [tuple(A.identity()), nodes[0], nodes[BIFUND_INDICES[0]], nodes[BIFUND_INDICES[1]]]
    se = [Element({l: ONE}) for l in win]
    te = [iso.map(e) for e in se]
    ok = (iso.verify_unit()
          and iso.verify_round_trip(se, te)
          and iso.verify_rho_equivariant(se, te))
    check("SU(2)^3 RG bifund-drop iso → pure SU(2)³ (structural battery)", ok)


def test_wild_chambers():
    at = BPSAtlas(su2_cubed_chart())
    coop = declined = 0
    for k in range(8):
        for d in ("fwd", "inv"):
            try:
                at.mutate((), k, d); coop += 1
            except ValueError:
                declined += 1
    check("SU(2)^3 has wild chambers (declined > cooperating, never hangs)",
          declined > coop and coop >= 1)


def main():
    tests = [
        test_chart_builds,
        test_rho_is_permutation,
        test_rg_bifund_drop_iso_structural,
        test_wild_chambers,
    ]
    for t in tests:
        t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILURES:", FAIL)
        sys.exit(1)
    print("All SU(2)^3 BPSAtlas tests passed.")


if __name__ == "__main__":
    main()
