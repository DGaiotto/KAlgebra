"""Tests for `su2_reliable` — the improved su2 bootstrap.

Tr(1) from the closed-form Kac–Wakimoto vacuum character + pure-power seed
pinning (small pool, cache-cleared) ⇒ reliable, memory-light, BPS-free
elementary traces.  Cross-checked against the closed-form `a1d5_layer2`
characters and against `FiniteA1D5KAlgebra.trace` in its reliable window.

Run: `python3 run_tests.py`
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "implementations"))

from su2_reliable import vacuum_char, reliable_seeds
import a1d5_layer2 as L

K = 22


def test_vacuum_char_matches_closed_form():
    vc = vacuum_char(5, K)
    vl = L.vacuum_trace(K)
    for q in range(0, K + 1):
        assert vc.get(q, {}) == vl.get(q, {}), (f"vacuum q^{q}", vc.get(q), vl.get(q))


def test_reliable_seeds_match_closed_form():
    """The reliable bootstrap reproduces the explicit sl(2)₋₈/₅ characters
    (and so is correct well past the frozen table's ~q¹⁹ breakdown)."""
    rs = reliable_seeds("a1d5", K)
    for idx in (0, 1, 2, 3):
        e = L.seed_trace(idx, K)
        b = rs.get(idx, {})
        for q in range(1, K + 1):
            assert e.get(q, {}) == b.get(q, {}), (
                f"seed{idx} q^{q}: closed={e.get(q)} reliable={b.get(q)}")


def test_fixes_frozen_tail_bug():
    """At q¹⁹ the frozen a1d5 table was wrong (under-resolved); it was removed
    on 2026-09-23, and the class now serves the closed form.  The
    reliable bootstrap and the class's served trace both agree with the closed
    form there."""
    from finite_a1d5_kalg import FiniteA1D5KAlgebra
    A = FiniteA1D5KAlgebra()
    rs = reliable_seeds("a1d5", 20)
    closed = L.seed_trace(1, 20)
    served = A.trace(((1, 1),), K=26)
    sv19 = {(n[0] if isinstance(n, tuple) else n): c
            for n, c in served.coeffs.get(19).terms.items() if c}
    rel19 = rs[1].get(19, {})
    cl19 = closed.get(19, {})
    assert rel19 == cl19, "reliable != closed at q^19"
    assert sv19 == cl19, "the served a1d5 trace != closed at q^19"


def test_a1d3_vacuum():
    """sl(2)₋₄/₃ vacuum (v=3): χ₀, then the χ₁₁-analog at q² (= χ_2 here)."""
    vc = vacuum_char(3, 6)
    assert vc[0] == {0: 1}
    assert vc.get(2, {}).get(2) == 1


if __name__ == "__main__":
    import traceback
    tests = [test_vacuum_char_matches_closed_form,
             test_reliable_seeds_match_closed_form,
             test_fixes_frozen_tail_bug, test_a1d3_vacuum]
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
