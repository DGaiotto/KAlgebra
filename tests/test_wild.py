"""Self-test for the **"wild" formal RGKAlgebra examples** (Step 3) — the
capstone showing the framework is decoupled from physics: `S_RG = E_𝖖(L_a)` on
*any* monomial cone ray of *any* (product of) `ConeKAlgebra`(s) gives a
well-formed, truncation-safe `RGKAlgebra` (closed multiply, orthonormal canonical
basis, convergent trace) — even when no 4d N=2 theory realises it.

  * `WildMonopoleRGKAlgebra` — `S_RG = E_𝖖(L_{1,0})` on the 't Hooft monopole ray
    of pure SU(2) (`aux = PureSU2KAlg`, Step 2).  Integrating out a *monopole*
    hyper is not a standard RG flow, yet `Tr(1) = 1 + 3q² + 9q⁴ + …` is clean and
    truncation-stable.
  * `WildA1D3SquaredRGKAlgebra` — two `[A₁,D₃]` coupled by `E_𝖖(μ·L·L')` over
    `A1DoddConeKAlg(0) ⊗ A1DoddConeKAlg(0) ⊗ QT_μ` (Step 2 ⊗ Step 1) — a
    fictional μ-dressed coupling; a genuine **pair-of-cones** example.

Both pure exact-FS (no override), spine-free.  Run (from the repository root):

    python3 run_tests.py
"""
import os
import sys
import warnings

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
for _root, _dirs, _ in os.walk(_SRC):
    _dirs[:] = [_d for _d in _dirs if _d != "__pycache__"]
    if _root not in sys.path:
        sys.path.insert(0, _root)

from wild_rgkalgebras import WildMonopoleRGKAlgebra, WildA1D3SquaredRGKAlgebra

# Spine-freeness: the module list is shared and filesystem-derived
# (tests/_spine.py) — the previous hand-copied tuple named modules that no
# longer exist and omitted real spine modules.
from _spine import SPINE as _SPINE


def _ser(rps, K):
    return {e: str(r) for e, r in sorted(rps.coeffs.items())
            if e <= K and str(r) not in ("0", "")}


def test_wild_monopole():
    """`S_RG = E_𝖖(L_{1,0})` (monopole ray) over pure SU(2): exact-FS, the formal
    `Tr(1) = 1 + 3q² + 9q⁴ + …` truncation-stable, leading orthonormality."""
    A = WildMonopoleRGKAlgebra()
    assert A._fs_exact_available()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v4 = _ser(A.trace(A.identity(), 4), 4)
        nw = len(w)
    assert nw == 0 and v4 == {0: "1", 2: "3", 4: "9"}, ("wild-monopole vacuum", v4)
    assert v4 == _ser(A.trace(A.identity(), 6), 4), "not truncation-stable"
    assert str(A.inner_product(A.identity(), A.identity(), 4).coeffs.get(0)) == "1"
    print("  PASS: test_wild_monopole (Tr(1) = 1 + 3q² + 9q⁴ over pure SU(2))")


def test_wild_a1d3_squared():
    """Two `[A₁,D₃]` coupled by `E_𝖖(μ L L')` — a pair-of-cones formal flow:
    exact-FS, truncation-stable `Tr(1)`, leading orthonormality."""
    A = WildA1D3SquaredRGKAlgebra()
    assert A._fs_exact_available()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v4 = _ser(A.trace(A.identity(), 4), 4)
        nw = len(w)
    assert nw == 0, ("warns", nw)
    assert v4.get(0) == "1", ("vacuum q⁰", v4)
    assert v4 == _ser(A.trace(A.identity(), 6), 4), "not truncation-stable"
    assert str(A.inner_product(A.identity(), A.identity(), 4).coeffs.get(0)) == "1"
    print("  PASS: test_wild_a1d3_squared (pair-of-cones formal flow)")


def test_spine_free():
    hit = sorted(m for m in sys.modules
                 if any(m == s or m.startswith(s + ".") for s in _SPINE))
    assert not hit, ("spine modules leaked into the wild release", hit)
    print("  PASS: test_spine_free (no spine modules imported)")


def main():
    test_wild_monopole()
    test_wild_a1d3_squared()
    test_spine_free()
    print("\nAll wild formal RGKAlgebra (Step 3) tests passed.")


if __name__ == "__main__":
    main()
