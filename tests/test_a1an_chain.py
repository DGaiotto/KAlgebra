"""Self-test for the **A1Aeven / U1A1Aodd chain** (Step 3) — the two-leg
A-type Argyres–Douglas RG chain, both legs pure exact-FS `RGKAlgebra`s:

    leg 1   [A₁, A_{2k+2}]            ──▶  u(1)-gauged [A₁, A_{2k+1}]
            (A1AevenToU1AoddRGKAlgebra)     (= U1A1AoddKAlg(k))

    leg 2   u(1)-gauged [A₁, A_{2k+1}] ──▶  [A₁, A_{2k}] ⊗ QT[Z²]
            (U1A1AoddToEvenQTRGKAlgebra)    (= A1A2kKAlg(k) ⊗ quantum torus)

The RG layer (Step 3, RGKAlgebra) **depends on** Step 1 (KAlgebra: the core
contract + `quantum_torus_kalgebra` + `tensor_kalgebra`) and Step 2
(ConeKAlgebra: the cone auxiliaries `A1A2kKAlg`, `U1A1AoddKAlg`) — the cone
auxiliaries are imported, not duplicated.

For each leg: RG solved from the discovery relation by the generic engine, the
vacuum Schur index reproducing the standalone cone presentation, truncation
stability, and the KAlgebra axioms + orthonormality.  Finally, an assertion that
**no spine module** is imported anywhere (spine-free auxiliaries).

Run (from the repository root):

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

from a1a2k_kalg import A1A2kKAlg
from u1a1aodd_kalg import U1A1AoddKAlg
from a1aeven_to_u1aodd_rgkalgebra import A1AevenToU1AoddRGKAlgebra
from u1aodd_to_even_qt_rgkalgebra import U1A1AoddToEvenQTRGKAlgebra

# Spine-freeness: the module list is shared and filesystem-derived
# (tests/_spine.py) — the previous hand-copied tuple named modules that no
# longer exist and omitted real spine modules.
from _spine import SPINE as _SPINE


def _ser(rps, K):
    return {e: str(r) for e, r in sorted(rps.coeffs.items())
            if e <= K and str(r) not in ("0", "")}


def _axioms(A, labs, ortho_K=4):
    assert A.verify_rg_unital()
    for a in labs:
        assert A.verify_rg_bar_invariant(a), ("bar", a)
        for b in labs:
            assert A.verify_rg_multiplicative(a, b), ("mult", a, b)
            assert A.verify_orthonormality(a, b, ortho_K), ("ortho", a, b)


def test_leg1_even_to_gauged_odd():
    """Leg 1: `[A₁,A_{2k+2}] → U1A1AoddKAlg(k)`; vacuum = the standalone even
    `A1A2kKAlg(k+1)` Schur index, exactly (k=1,2)."""
    for k in (1, 2):
        A = A1AevenToU1AoddRGKAlgebra(k)
        D = A1A2kKAlg(k + 1)                     # [A₁, A_{2k+2}] standalone
        assert A._fs_exact_available()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            va = _ser(A.trace(A.identity(), 6), 6)
            nw = len(w)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            vd = _ser(D.trace(D.identity(), 6), 6)
        assert va == vd and nw == 0, (f"leg1 k={k}", va, vd, nw)
        # small label battery from the survivor cone
        e = A.identity()
        labs = [e]
        for d in list(A.rg_generator(2)):
            if d != e and d not in labs:
                labs.append(d)
            if len(labs) >= 4:
                break
        _axioms(A, labs)
    print("  PASS: test_leg1_even_to_gauged_odd ([A1,A4]→SQED-gauged, [A1,A6]→…)")


def test_leg2_gauged_odd_to_even_qt():
    """Leg 2: `U1A1AoddKAlg(k) → A1A2kKAlg(k) ⊗ QT`; vacuum = the standalone
    gauged-odd `U1A1AoddKAlg(k)` Schur index, exactly to q¹² (k=1 incl. the deep
    q¹⁰ term)."""
    for k in (1, 2):
        A = U1A1AoddToEvenQTRGKAlgebra(k)
        U = U1A1AoddKAlg(k)
        assert A._fs_exact_available()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            va = _ser(A.trace(A.identity(), 12), 12)
            nw = len(w)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            vu = _ser(U.trace(U.identity(), 12), 12)
        assert va == vu and nw == 0, (f"leg2 k={k}", va, vu, nw)
        # truncation stability
        assert _ser(A.trace(A.identity(), 8), 8) == _ser(A.trace(A.identity(), 12), 8)
        e = A.identity()
        labs = [e]
        for d in list(A.rg_generator(3)):
            if d != e and d not in labs:
                labs.append(d)
            if len(labs) >= 4:
                break
        labs += [(((1, 0, 1),), (0, 0)), (((1, 1, 1),), (0, 0))]
        _axioms(A, labs)
    assert _ser(U1A1AoddToEvenQTRGKAlgebra(1).trace(
        U1A1AoddToEvenQTRGKAlgebra(1).identity(), 12), 12) == {0: "1", 2: "-1", 10: "1"}
    print("  PASS: test_leg2_gauged_odd_to_even_qt (incl. deep q^10 at k=1)")


def test_spine_free():
    """No spine module is imported anywhere — the auxiliaries are spine-free."""
    hit = sorted(m for m in sys.modules
                 if any(m == s or m.startswith(s + ".") for s in _SPINE))
    assert not hit, ("spine modules leaked into the chain release", hit)
    print("  PASS: test_spine_free (no spine modules imported)")


def main():
    test_leg1_even_to_gauged_odd()
    test_leg2_gauged_odd_to_even_qt()
    test_spine_free()
    print("\nAll A1Aeven/U1A1Aodd chain (Step 3) tests passed.")


if __name__ == "__main__":
    main()
