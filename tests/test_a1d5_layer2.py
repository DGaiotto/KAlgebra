"""Tests for `a1d5_layer2` — explicit sl(2)₋₈/₅ Layer-2 characters of [A₁,D₅].

The four elementary traces + vacuum as closed-form affine sl(2)₋₈/₅
admissible-character combinations (the v=5 analogue of `a1d3_kalg`'s su(2)₋₄/₃
κ-recipe).  Validated against `FiniteA1D5KAlgebra.trace` inside the latter's
reliable window, and exercised as an arbitrary-order closed form.

Run: `python3 run_tests.py`
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "implementations"))

import a1d5_layer2 as L
from finite_a1d5_kalg import FiniteA1D5KAlgebra

# NOTE: the *frozen* a1d5 elem-trace table is only reliable up to ~q¹⁸ (its
# upper half is under-resolved — the very limitation `a1d5_layer2` + the
# improved bootstrap oracle correct, see the design notes).  So the
# cross-check window is q ≤ K_CMP, comfortably inside it; the closed form is
# exact far beyond (validated to q²⁴ against the reliable oracle in dev).
K_CMP = 16


def _boot_char(A, lab, K):
    tr = A.trace(lab, K=K)
    return {q: {(n[0] if isinstance(n, tuple) else n): c
                for n, c in r.terms.items() if c}
            for q, r in tr.coeffs.items() if not r.is_zero()}


def test_vacuum_matches_trace():
    A = FiniteA1D5KAlgebra()
    e = L.vacuum_trace(K_CMP)
    b = _boot_char(A, A.identity(), K_CMP + 6)
    for q in range(0, K_CMP + 1):
        assert e.get(q, {}) == b.get(q, {}), (f"vacuum q^{q}", e.get(q), b.get(q))


def test_seeds_match_trace():
    A = FiniteA1D5KAlgebra()
    for idx in (0, 1, 2, 3):
        e = L.seed_trace(idx, K_CMP)
        b = _boot_char(A, ((idx, 1),), K_CMP + 6)
        for q in range(0, K_CMP + 1):
            assert e.get(q, {}) == b.get(q, {}), (
                f"seed{idx} q^{q}: explicit={e.get(q)} boot={b.get(q)}")


def test_leading_characters():
    """The documented leading content (sl(2)₋₈/₅ vacuum current + the
    Tr_T/Tr_D-analog seed leads)."""
    assert L.vacuum_trace(4)[0] == {0: 1}
    assert L.vacuum_trace(4)[2] == {2: 1}                 # χ₂ flavour current
    assert L.seed_trace(1, 4)[1] == {0: -1}               # seed1 ~ −χ₀ 𝖖
    assert L.seed_trace(2, 4)[1] == {1: -1}               # seed2 ~ −χ₁ 𝖖
    assert L.seed_trace(3, 4)[2] == {1: 1}                # seed3 ~ +χ₁ 𝖖²


def test_arbitrarily_improvable():
    """Closed form ⇒ no fixed-K cap: evaluate far past the frozen window,
    spine-free (pure theta arithmetic)."""
    K = 40
    for idx in (0, 1, 2, 3):
        t = L.seed_trace(idx, K)
        assert max(t) >= K - 2, (f"seed{idx} did not reach q^{K}", max(t))
    v = L.vacuum_trace(K)
    assert max(v) >= K - 2


if __name__ == "__main__":
    import traceback
    tests = [test_vacuum_matches_trace, test_seeds_match_trace,
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
