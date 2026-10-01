"""Tests for `a1dodd_layer2` — the general-k [A₁, D_{2k+3}] Layer-2 trace.

Verifies the general-k closed-form reproduces the per-k hand modules
`a1d5_layer2` (k=1, fully) and `a1d7_layer2` (k=2, the pinned seeds), and that
the seed-index <-> (level a, parity p) bijection is consistent.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "implementations"))

import a1dodd_layer2 as L
import a1d5_layer2 as L5
import a1d7_layer2 as L7

K = 16


def test_k1_reproduces_a1d5_fully():
    assert L.vacuum_trace(1, K) == L5.vacuum_trace(K)
    for idx in range(4):
        assert L.seed_trace(1, idx, K) == L5.seed_trace(idx, K), f"seed{idx}"


def test_k2_reproduces_a1d7_full():
    assert L.vacuum_trace(2, K) == L7.vacuum_trace(K)
    # ALL 6 seeds, including idx0 = (a=3, p=0) = the diameter (now pinned)
    for idx in range(6):
        assert L.seed_trace(2, idx, K) == L7.seed_trace(idx, K), f"seed{idx}"


def test_diameter_seed0_pinned():
    # k=2 idx0 = (a=3, p=0) = the χ₁ fork diameter: now closed-form, leading −𝖖³χ₀
    s0 = L.seed_trace(2, 0, K)
    assert min(s0) == 3 and s0[3] == {0: -1}, s0
    assert s0 == L7.seed_trace(0, K)


def test_idx_ap_bijection():
    for k in (1, 2, 3):
        seen = set()
        for idx in range(2 * k + 2):
            a, p = L.idx_to_ap(k, idx)
            assert 1 <= a <= k + 1 and p in (0, 1)
            assert L.ap_to_idx(k, a, p) == idx
            seen.add((a, p))
        assert len(seen) == 2 * (k + 1)


def test_leading_terms_match_scaffold():
    # leading Schur term of seed (a, p) is (-1)^a q^a chi_p (the scaffold result)
    for k in (1, 2):
        for a in range(1, k + 2):
            for p in (0, 1):
                tr = L.seed_trace_ap(k, a, p, K)
                lead = min(tr)
                assert lead == a, f"k={k} (a={a},p={p}) leading q^{lead} != q^{a}"
                assert tr[lead] == {p: (-1) ** a}, f"k={k} (a={a},p={p}) {tr[lead]}"


def test_vacuum_pe_matches_sigma():
    """The Pan-Yang PE closed form (vacuum_trace_pe) == the sigma_j/Kac-Wakimoto
    vacuum_trace, for k=1..4 (the latter is general-k too)."""
    for k in (1, 2, 3, 4):
        pe = L.vacuum_trace_pe(k, 2 * K)         # K paper-orders
        sg = L.vacuum_trace(k, 2 * K)
        # compare on shared q-bold powers <= 2K
        for qb in range(2 * K + 1):
            assert pe.get(qb, {}) == sg.get(qb, {}), (k, qb, pe.get(qb), sg.get(qb))


def test_vacuum_pe_general_k():
    """vacuum_trace_pe is a trivial closed form for ALL k (incl. k>=3, where the
    cone multiply is open).  Leading flavored structure: q^0 chi0, q^2 chi2."""
    for k in (3, 5, 8):
        pe = L.vacuum_trace_pe(k, 8)
        assert pe[0] == {0: 1}            # vacuum
        assert pe[2] == {2: 1}            # SU(2) adjoint flavour current at q_paper^1


def test_memo_equals_direct():
    """The depth-extending memo inside `vacuum_trace` / `seed_trace` returns
    exactly the direct evaluation, at depths asked in an order that extends
    and truncates the memo (21 series at k = 1..3, five depths each: 105
    comparisons).  Control: a memo keyed WITHOUT k (serving k = 1's vacuum
    for k = 2) is caught by the same comparison, so an equality here is not
    vacuous."""
    L._MEMO.clear()
    series = [("vac", k, None) for k in (1, 2, 3)]
    series += [("seed", k, idx) for k in (1, 2, 3) for idx in range(2 * k + 2)]
    n = 0
    for K in (6, 20, 11, 28, 2):
        for kind, k, idx in series:
            if kind == "vac":
                got, want = L.vacuum_trace(k, K), L._vacuum_trace_direct(k, K)
            else:
                got, want = L.seed_trace(k, idx, K), L._seed_trace_direct(k, idx, K)
            assert got == want, (kind, k, idx, K)
            n += 1
    assert n == 105
    for k in (1, 2, 3):                        # seed_trace_ap reads the same memo
        for a in range(1, k + 2):
            for p in (0, 1):
                assert L.seed_trace_ap(k, a, p, 17) == L._seed_trace_direct(
                    k, L.ap_to_idx(k, a, p), 17)
    # returned dicts are copies: mutating one does not reach the memo
    t = L.vacuum_trace(2, 10)
    t[0][0] = 99
    assert L.vacuum_trace(2, 10) == L._vacuum_trace_direct(2, 10)
    # negative control: a k-blind key collides and the comparison sees it
    wrong = {}
    def _wrong_vac(k, K):
        if "vac" not in wrong:
            wrong["vac"] = L._vacuum_trace_direct(k, K)
        return {q: d for q, d in wrong["vac"].items() if q <= K}
    _wrong_vac(1, 20)
    assert _wrong_vac(2, 20) != L._vacuum_trace_direct(2, 20)


if __name__ == "__main__":
    test_k1_reproduces_a1d5_fully()
    test_k2_reproduces_a1d7_full()
    test_diameter_seed0_pinned()
    test_idx_ap_bijection()
    test_leading_terms_match_scaffold()
    test_vacuum_pe_matches_sigma()
    test_vacuum_pe_general_k()
    test_memo_equals_direct()
    print("a1dodd_layer2: all tests pass")
