"""Tests for `a1d7_layer2` — explicit sl(2)₋₁₂/₇ Layer-2 characters of [A₁,D₇].

The v=7 sibling of `test_a1d5_layer2`.  Cross-checked against the reliable
su2 bootstrap (`su2_reliable`, which pins all 6 a1d7 seeds
memory-light) and exercised as an arbitrary-order closed form.

Run: `python3 run_tests.py`
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "implementations"))

import a1d7_layer2 as L
from su2_reliable import reliable_seeds, vacuum_char

K = 14


def test_vacuum_matches_reliable():
    ve = L.vacuum_trace(K)
    vb = vacuum_char(7, K)
    for q in range(0, K + 1):
        assert ve.get(q, {}) == vb.get(q, {}), (f"vacuum q^{q}", ve.get(q), vb.get(q))


def test_seeds_match_reliable():
    """The closed-form sl(2)₋₁₂/₇ characters reproduce the reliable bootstrap
    seeds for all six elementary traces of [A₁,D₇]."""
    rs = reliable_seeds("a1d7", K, amax=8)
    for idx in range(6):
        e = L.seed_trace(idx, K)
        b = rs.get(idx, {})
        for q in range(1, K + 1):
            assert e.get(q, {}) == b.get(q, {}), (
                f"seed{idx} q^{q}: closed={e.get(q)} reliable={b.get(q)}")


def test_leading_characters():
    assert L.vacuum_trace(4)[0] == {0: 1}
    assert L.vacuum_trace(4)[2] == {2: 1}                 # χ₂ flavour current
    assert L.seed_trace(2, 4)[1] == {0: -1}               # odd-q int seed lead
    assert L.seed_trace(3, 4)[1] == {1: -1}               # odd-q half-int seed lead


def test_arbitrarily_improvable():
    KK = 36
    for idx in range(6):
        t = L.seed_trace(idx, KK)
        assert max(t) >= KK - 3, (f"seed{idx} did not reach q^{KK}", max(t))
    assert max(L.vacuum_trace(KK)) >= KK - 2


if __name__ == "__main__":
    import traceback
    tests = [test_vacuum_matches_reliable, test_seeds_match_reliable,
             test_leading_characters, test_arbitrarily_improvable]
    n = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
        else:
            print(f"OK    {t.__name__}")
            n += 1
    print(f"\n{n}/{len(tests)} tests passed")
