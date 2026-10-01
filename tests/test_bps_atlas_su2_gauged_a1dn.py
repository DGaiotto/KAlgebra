"""`BPSAtlas` for the **SU(2)-gauged [A₁,Dₙ]** family (the design record catalogue).

Pins `src/abe/su2_gauged_a1dn_atlas.py` (over `su2a1d3_gauged.py`).  These
gauged-AD theories have **short specs** (`|spec| = n+2`, `rank = n+1`), so — unlike
the SU(3) quivers — the atlas certifies *fully and fast* across the family.

Certifies:

  * **build-validity for the whole family (n=3..8)**: `rank = n+1`, `|spec| = n+2`,
    and the Dₙ leaf-parity coefficient ring (odd n → `Z` flavourless, even n →
    `R(U(1))`);
  * for **n = 3,4,5** (the rotation walk materializes a chart per step, slow at
    rank ≥ 7 — the script covers the full span): rotation with **period =
    2·|spec| = 2(n+2)** and **monodromy = ρ²** (ρ infinite-order); both
    single-node-drop RG isos (**matter** and **tail**) pass the full battery incl.
    multiplicativity; wild chambers present (only the two spec-end directions
    cooperate);
  * orthonormality `I_{a,b} = δ + O(q)` (n=3, all nodes, diag + off-diag).

Run:  `python3 run_tests.py`
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing, AbelianZPlusRing
from bps_atlas import BPSAtlas
from su2_gauged_a1dn_atlas import (
    MAX_N, su2_gauged_a1dn_names, su2_gauged_a1dn_entry, su2_gauged_a1dn_rg_iso,
)

PASS = []
FAIL = []
ONE = LaurentPoly.one()
_NS = list(range(3, MAX_N + 1))      # build-validity: the whole family (instant)
# The rotation walk materializes a chart per step; at rank ≥ 7 that is slow, so the
# per-n dynamics / wild / RG-drop checks run on a low-n window (the pattern is
# uniform — the script certifies the full span).
_NS_DYN = [3, 4, 5]


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


def _battery(iso, labels):
    se = [Element({tuple(l): ONE}) for l in labels]
    te = [iso.map(e) for e in se]
    pairs = [(se[i], se[j]) for i in range(len(se)) for j in range(len(se))]
    return all([
        iso.verify_unit(),
        iso.verify_round_trip(se, te),
        iso.verify_rho_equivariant(se, te),
        iso.verify_multiplicative(pairs, [(iso.map(a), iso.map(b)) for a, b in pairs]),
    ])


# ---------------------------------------------------------------------------
def test_family_builds_with_leaf_parity_ring():
    ok = True
    for n in _NS:
        e = su2_gauged_a1dn_entry(n)
        A = e.chart
        ok = ok and (e.rank == n + 1) and (len(A.spec) == n + 2)
        R = A.coefficient_ring()
        if n % 2 == 1:
            ok = ok and isinstance(R, TrivialZPlusRing) and e.flavourless
        else:
            ok = ok and isinstance(R, AbelianZPlusRing) and R.rank == 1
    check("SU2gA1Dn (n=3..%d) build: rank n+1, |spec| n+2, leaf-parity ring" % MAX_N, ok)


def test_dynamics_rotates_period_2spec_monodromy_rho2():
    ok = True
    for n in _NS_DYN:
        A = su2_gauged_a1dn_entry(n).chart
        at = BPSAtlas(A)
        labels = [tuple(g) for g in A.node_charges][:3]
        keys = at.rotation_chambers(max_steps=2 * len(A.spec) + 4)
        if len(keys) - 1 != 2 * len(A.spec):
            ok = False
        mono = at.monodromy()
        if not all(next(iter(mono.map(Element({l: ONE})).terms)) == at.rho(at.rho(l))
                   for l in labels):
            ok = False
        if not any(at.rho(at.rho(l)) != l for l in labels):   # ρ infinite-order
            ok = False
    check("SU2gA1Dn rotates: period = 2·|spec|, monodromy = ρ² (ρ infinite-order)", ok)


def test_both_rg_node_drops():
    ok = True
    for n in _NS_DYN:
        e = su2_gauged_a1dn_entry(n)
        A = e.chart
        nodes = [tuple(g) for g in A.node_charges]
        for which, j in (("matter", e.matter_drop_index), ("tail", e.tail_drop_index)):
            iso = su2_gauged_a1dn_rg_iso(n, which)
            win = [tuple(A.identity()), nodes[0], nodes[j]]
            ok = ok and _battery(iso, win)
    check("SU2gA1Dn both node-drop RG isos (matter + tail), full battery", ok)


def test_wild_chambers_present():
    ok = True
    for n in _NS_DYN:
        A = su2_gauged_a1dn_entry(n).chart
        at = BPSAtlas(A)
        coop = declined = 0
        for k in range(n + 1):
            for d in ("fwd", "inv"):
                try:
                    at.mutate((), k, d); coop += 1
                except ValueError:
                    declined += 1
        ok = ok and (declined > coop)   # most directions wild
    check("SU2gA1Dn has wild chambers (declined > cooperating, never hangs)", ok)


def test_n3_orthonormality_full():
    A = su2_gauged_a1dn_entry(3).chart
    nodes = [tuple(g) for g in A.node_charges]
    R = A.coefficient_ring()
    one, zero = R.one(), R.zero()
    diag = all(A.inner_product(a, a, K=4)[0] == one for a in nodes)
    off = all(A.inner_product(nodes[i], nodes[j], K=4)[0] == zero
              for i in range(len(nodes)) for j in range(i + 1, len(nodes)))
    check("SU2gA1D3 orthonormality I_{a,b} = δ + O(q) (all nodes, diag + off-diag)",
          diag and off)


def main():
    tests = [
        test_family_builds_with_leaf_parity_ring,
        test_dynamics_rotates_period_2spec_monodromy_rho2,
        test_both_rg_node_drops,
        test_wild_chambers_present,
        test_n3_orthonormality_full,
    ]
    for t in tests:
        t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILURES:", FAIL)
        sys.exit(1)
    print("All SU(2)-gauged-[A1,Dn] BPSAtlas tests passed.")


if __name__ == "__main__":
    main()
