"""Tests for `tame_novel_atlas` — the candidate-novel tame BPS quivers
(S/spec-finder landscape programme) as `BPSKAlgebra` charts +
`BPSAtlas` objects.

Fast subset only (the full study sweep lives in
`scripts/atlas_tame_novel.py`): chart construction + spec replay for every
entry, the structural contract battery on the small-rank members, pinned
orthonormality q⁰ values, the pinned vacuum-index fingerprints, and the
closed-orbit atlas folds (r4-0121 = ONE chart with 4 mutation self-loops;
r5-tame-06 closes at 5 classes).

Run: `python3 run_tests.py`
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from tame_novel_atlas import (TAME_NAMES, tame_atlas, tame_chart, tame_entry,
                              tame_names)
from snf_kernel import integer_kernel_and_section


PASS = []
FAIL = []


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(("  PASS: " if ok else "  FAIL: ") + name)


def std_basis(n):
    return [tuple(1 if j == i else 0 for j in range(n)) for i in range(n)]


def test_all_entries_build():
    """Every recorded entry constructs; the negating sequence replays into a
    spec of the recorded length; the coefficient ring rank equals the corank."""
    built = 0
    for nm in tame_names():
        try:
            e = tame_entry(nm)
        except ValueError:
            print(f"    ({nm}: pending spec frame — skipped)")
            continue
        A = tame_chart(nm)
        ker, _ = integer_kernel_and_section(e.pairing)
        ring = A.coefficient_ring()
        rank_ok = (len(ker) == e.corank)
        ring_rank = getattr(ring, "rank", 0 if e.corank == 0 else None)
        ring_ok = (e.corank == 0) or (ring_rank == e.corank)
        spec_ok = len(A.spec) == len(e.negseq)
        check(f"build[{nm}]", rank_ok and ring_ok and spec_ok)
        built += 1
    check("at_least_ten_entries_build", built >= 10)


def test_structural_battery_small_ranks():
    """Full structural verifier battery on the fast small-rank members.
    (r5_tame00 is excluded: its node-pair multiplies run >10 min — the
    driver sweep covers it under timeboxes.)"""
    for nm in ("r4_0121", "r5_tame06"):
        A = tame_chart(nm)
        n = tame_entry(nm).rank
        std = std_basis(n)
        ok = (A.verify_identity_in_basis()
              and A.verify_rho_fixes_identity()
              and all(A.verify_rho_inverse(a) for a in std)
              and all(A.verify_bar_involution(a, b)
                      for a in std for b in std)
              and all(A.verify_rho_is_automorphism(a, b)
                      for a in std for b in std))
        check(f"structural[{nm}]", ok)


def test_orthonormality_q0_r4():
    """orthonormality at q⁰ for the tower root: I_{a,b}[0] = δ_{a,b} on nodes."""
    A = tame_chart("r4_0121")
    std = std_basis(4)
    one = A.coefficient_ring().one()
    zero = A.coefficient_ring().zero()
    diag = all(A.inner_product(a, a, K=2)[0] == one for a in std)
    off = all(A.inner_product(std[i], std[j], K=2)[0] == zero
              for i in range(4) for j in range(i + 1, 4))
    check("orthonormality_q0[r4_0121]", diag and off)


def test_vacuum_index_fingerprints():
    """Pin the exact low-order vacuum Schur indices (identification data).

    r4-0121's unflavoured vacuum index has a NEGATIVE q⁴ coefficient
    (1 − 2q⁴ + …) — unusual against the AD/gauge examples and part of its
    unidentified-fingerprint evidence sheet."""
    A = tame_chart("r4_0121")
    v = A.trace((0, 0, 0, 0), K=8)
    zero = A.coefficient_ring().zero()
    one = A.coefficient_ring().one()
    check("vacuum[r4_0121]", v[0] == one and v[2] == zero
          and v[4] == -(one + one) and v[6] == zero)


def test_atlas_r4_0121_folds_to_a_point():
    """r4-0121 is mutation-invariant: the mutation-complete atlas is ONE
    chart with a self-loop for every node mutation (cf. pentagon: 1 chart,
    2 self-loops)."""
    At = tame_atlas("r4_0121")
    mc = At.mutation_complete(max_charts=64)
    check("atlas_fold[r4_0121]",
          mc.get("closed") is True and mc.get("n_charts") == 1
          and mc.get("self_loops") == 4)


def test_atlas_r5_tame06_closes():
    """r5-tame-06's mutation-complete atlas CLOSES at 6 charts (10 mutation
    self-loops).  Note: the experiments' canonical-key orbit is 5 — the key
    quotients by the global sign flip B → −B (the opposite quiver), which
    the atlas's chart-iso fold (node-perm + Γ-automorphism) does not."""
    At = tame_atlas("r5_tame06")
    mc = At.mutation_complete(max_charts=256)
    check("atlas_fold[r5_tame06]",
          mc.get("closed") is True and mc.get("n_charts") == 6
          and mc.get("self_loops") == 10)


def test_subquiver_rg_flows():
    """RGKAlgebra presentations over recognized sub-quivers (the drop map):
    the flow builds, the RG axioms hold, and the RG-derived multiply equals
    the BPS multiply on the light entries."""
    from tame_novel_atlas import tame_flow, tame_flow_targets
    targets = tame_flow_targets()
    check("flow_targets_recorded", len(targets) == 7)
    for nm, pair_idx in (("r4_0121", (0, 1)), ("r5_tame06", (0, 3))):
        R = tame_flow(nm)
        A = tame_chart(nm)
        n = tame_entry(nm).rank
        std = std_basis(n)
        a, b = std[pair_idx[0]], std[pair_idx[1]]
        ok = (R.verify_rg_unital()
              and R.verify_rg_multiplicative(a, b)
              and R.verify_rg_bar_invariant(a)
              and R.multiply(a, b) == A.multiply(a, b))
        check(f"subquiver_rg[{nm}]", ok)


def main():
    test_all_entries_build()
    test_structural_battery_small_ranks()
    test_orthonormality_q0_r4()
    test_vacuum_index_fingerprints()
    test_atlas_r4_0121_folds_to_a_point()
    test_atlas_r5_tame06_closes()
    test_subquiver_rg_flows()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED, {len(PASS)} passed: {FAIL}")
        return 1
    print(f"All {len(PASS)} tame-novel atlas tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
