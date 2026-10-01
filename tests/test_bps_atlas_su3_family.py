"""`BPSAtlas` for the **SU(3)+N_f family** (the design record catalogue).

Pins `src/abe/su3_family_atlas.py` and the atlas surface for SU(3)+N_f,
built from the (matter)(pure-SU(3)) recipe `pure_ade.SUN_Nf(3, N_f)`.  The
default family is `N_f = 1..MAX_NF` (= 4, the feasibility wall — rank-9
construction is many minutes).

Fast pins (the trace-level Schur-index certification is slow at rank ≥ 6 and is
the *script*, `scripts/atlas_su3_family.py`):

  * the family builds with the expected `rank = 4+N_f`, `|spec| = 6+5·N_f`,
    coefficient ring `R(U(1)^{N_f})` (N_f=1,2,3);
  * **SU(3)+N_f=1** rotates with **period 22** and rotation **monodromy = ρ²**,
    ρ² a non-trivial drift (ρ infinite-order — asymptotically free);
  * the **node-drop RG iso** (identity-on-labels, guaranteed) passes the light
    structural battery (N_f=1 and N_f=2);
  * wild chambers are present (most mutation directions are declined by the spec
    graph — the wild-chamber boundary inherited from pure SU(3));
  * orthonormality holds on a sampled node for N_f=1.

Run:  `python3 run_tests.py`
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import AbelianZPlusRing
from bps_atlas import BPSAtlas
from su3_family_atlas import (
    MAX_NF, su3_family_names, su3_family_entry, su3_family_rg_iso,
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
def test_default_family_is_nf_1_to_4():
    check("default family is SU3_Nf1..SU3_Nf4 (MAX_NF=4)",
          su3_family_names() == [f"SU3_Nf{n}" for n in range(1, 5)] and MAX_NF == 4)


def test_family_builds_with_expected_structure():
    ok = True
    for nf in (1, 2, 3):
        e = su3_family_entry(f"SU3_Nf{nf}")
        A = e.chart
        ok = ok and (A is e.chart)
        ok = ok and (e.rank == 4 + nf)
        ok = ok and (len(A.spec) == 6 + 5 * nf)
        ok = ok and isinstance(A.coefficient_ring(), AbelianZPlusRing)
        ok = ok and (A.coefficient_ring().rank == nf)
        ok = ok and (e.matter_indices == tuple(range(4, 4 + nf)))
    check("SU3_Nf1/2/3 build: rank=4+N_f, |spec|=6+5N_f, coeff R(U(1)^N_f)", ok)


def test_nf1_rotates_period22_monodromy_rho2():
    at = BPSAtlas(su3_family_entry("SU3_Nf1").chart)
    labels = [tuple(g) for g in at.root.node_charges][:3]
    keys = at.rotation_chambers(max_steps=30)
    check("SU3_Nf1 rotates with period 22", len(keys) - 1 == 22)
    mono = at.monodromy()
    is_rho2 = all(
        next(iter(mono.map(Element({l: ONE})).terms)) == at.rho(at.rho(l))
        for l in labels)
    check("SU3_Nf1 rotation monodromy == ρ² on node labels", is_rho2)
    check("SU3_Nf1 ρ² non-trivial (ρ infinite-order, AF)",
          any(at.rho(at.rho(l)) != l for l in labels))


def test_nf1_rg_nodedrop_iso():
    e = su3_family_entry("SU3_Nf1")
    A = e.chart
    iso = su3_family_rg_iso("SU3_Nf1")
    matter = tuple(A.node_charges[e.matter_indices[0]])
    win = [tuple(A.identity()), matter, tuple(A.node_charges[0])]
    check("SU3_Nf1 node-drop RG iso (light battery)",
          all(_struct_battery(iso, win).values()))


def test_nf2_rg_nodedrop_iso():
    e = su3_family_entry("SU3_Nf2")
    A = e.chart
    iso = su3_family_rg_iso("SU3_Nf2")
    matter = tuple(A.node_charges[e.matter_indices[0]])
    win = [tuple(A.identity()), matter, tuple(A.node_charges[0])]
    check("SU3_Nf2 node-drop RG iso (light battery)",
          all(_struct_battery(iso, win).values()))


def test_nf1_wild_chambers_present():
    at = BPSAtlas(su3_family_entry("SU3_Nf1").chart)
    coop = declined = 0
    for k in range(5):
        for d in ("fwd", "inv"):
            try:
                at.mutate((), k, d); coop += 1
            except ValueError:
                declined += 1
    check("SU3_Nf1 has wild chambers (most directions declined, spec graph "
          "never hangs)", declined > 0 and coop >= 1)


def test_nf1_orthonormality_sampled():
    A = su3_family_entry("SU3_Nf1").chart
    one = A.coefficient_ring().one()
    node0 = tuple(A.node_charges[0])
    check("SU3_Nf1 orthonormality I_aa[q^0]==1 (sampled node)",
          A.inner_product(node0, node0, K=2)[0] == one)


def main():
    tests = [
        test_default_family_is_nf_1_to_4,
        test_family_builds_with_expected_structure,
        test_nf1_rotates_period22_monodromy_rho2,
        test_nf1_rg_nodedrop_iso,
        test_nf2_rg_nodedrop_iso,
        test_nf1_wild_chambers_present,
        test_nf1_orthonormality_sampled,
    ]
    for t in tests:
        t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILURES:", FAIL)
        sys.exit(1)
    print("All SU(3)+N_f BPSAtlas tests passed.")


if __name__ == "__main__":
    main()
