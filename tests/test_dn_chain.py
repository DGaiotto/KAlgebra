"""Self-test for the **D-type gauged chain** (Step 3) — the D-type analogue
of the A1Aeven/U1A1Aodd chain, the gauged rungs of the A1Dodd–U1A1Deven ladder:

    A1D3Sqed2:      [A₁, D₃]            →  SQED₂           (odd-rung base; single
                    (A1D3Sqed2RGKAlgebra)                  SU(2)-singlet monopole drop)

    U1A1DevenSqed:  u(1)-gauged [A₁, D_{2k+2}] → A1Dodd(k-1) ⊗ QT[Z²]
                    (U1A1DevenSqedRGKAlgebra)              (even-rung; single
                                                           gauge-dressed doublet drop)

Both are pure exact-FS `RGKAlgebra`s with spine-free auxiliaries:
  * A1D3Sqed2's aux is `SQEDNfSampleKAlgebra(2)` (Step 1) — the SU(2) is SQED₂'s
    *intrinsic* flavour, so `S_RG = E_𝖖(u₊)` drops a single SU(2)-**singlet**
    monopole (contrast the SQED₁⊗SU(2) flow, which drops a doublet);
  * U1A1DevenSqed's aux is `A1DoddConeKAlg(k-1) ⊗ QT[Z²]` (Step 2 ⊗ Step 1) — the
    SU(2) is intrinsic to the A1Dodd survivor, so `S_RG = E_𝖖(X_{(0,1)}·L)` drops a
    single gauge-dressed doublet (no `add_flavour` spectator).

Depends on Step 1 (KAlgebra) and Step 2 (ConeKAlgebra: `a1dodd_kalg`, `a1d3_kalg`).
For each: exact-FS, truncation-stable vacuum (A1D3Sqed2 cross-checked against the
standalone `A1D3KAlg`), the KAlgebra axioms, and a no-spine-imported assertion.

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

from a1d3_sqed2_rgkalgebra import A1D3Sqed2RGKAlgebra
from u1a1deven_sqed_rgkalgebra import U1A1DevenSqedRGKAlgebra
from a1d3_kalg import A1D3KAlg

_SPINE = ("bps_kalgebra", "rg_flow", "lattice_torus", "nahm_data", "chart_graph",
          "bps_quiver_tools", "f_solver", "directional_subquiver_rg")


def _ser(rps, K):
    return {e: str(r) for e, r in sorted(rps.coeffs.items())
            if e <= K and str(r) not in ("0", "")}


def _su2_to_sun(d):
    """`[k]` (SU2ZPlusRing) -> `[(k,)]` (SUNZPlusRing(2)) so the two R(SU(2))
    coefficient rings compare term-for-term."""
    return {e: s.replace("[", "[(").replace("]", ",)]") for e, s in d.items()}


def _axioms(A, labs, ortho_K=4):
    assert A.verify_rg_unital()
    for a in labs:
        assert A.verify_rg_bar_invariant(a), ("bar", a)
        for b in labs:
            assert A.verify_rg_multiplicative(a, b), ("mult", a, b)
            assert A.verify_orthonormality(a, b, ortho_K), ("ortho", a, b)


def test_a1d3_sqed2_odd_rung():
    """A1D3Sqed2 ([A₁,D₃] → SQED₂): exact-FS; vacuum = the standalone `A1D3KAlg`
    Schur index exactly (q² = the SU(2) adjoint χ₂), truncation-stable."""
    A = A1D3Sqed2RGKAlgebra()
    D = A1D3KAlg()
    assert A._fs_exact_available()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        va = _ser(A.trace(A.identity(), 6), 6)
        nw = len(w)
    assert nw == 0, ("warns", nw)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        vd = _ser(D.trace(D.identity(), 6), 6)
    assert va == _su2_to_sun(vd), ("vacuum vs A1D3KAlg", va, vd)
    assert va == _ser(A.trace(A.identity(), 8), 6), "not truncation-stable"
    labs = [A.identity(), (0, 1, ()), (0, 1, (2,)), (1, 0, ())]
    _axioms(A, labs)
    print("  PASS: test_a1d3_sqed2_odd_rung ([A1,D3] vacuum = A1D3KAlg, q² = χ₂)")


def test_u1a1deven_sqed_even_rung():
    """U1A1DevenSqed (u(1)-gauged [A₁,D₄] over A1Dodd(0) ⊗ QT): exact-FS; vacuum
    `1 − q² + …` with q² = −1 + χ₂ (the SU(2) current minus the gauged-U(1)
    subtraction), truncation-stable."""
    A = U1A1DevenSqedRGKAlgebra(1)
    assert A._fs_exact_available()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v6 = _ser(A.trace(A.identity(), 6), 6)
        nw = len(w)
    assert nw == 0, ("warns", nw)
    assert str(A.trace(A.identity(), 2).coeffs.get(2)) == "-1 + [2]", \
        ("q² should be −1 + χ₂", _ser(A.trace(A.identity(), 2), 2))
    assert v6 == _ser(A.trace(A.identity(), 8), 6), "not truncation-stable"
    e = A.identity()
    labs = [e]
    for d in list(A.rg_generator(2)):
        if d != e and d not in labs:
            labs.append(d)
        if len(labs) >= 4:
            break
    _axioms(A, labs)
    print("  PASS: test_u1a1deven_sqed_even_rung (gauged [A1,D4], q² = −1 + χ₂)")


def test_spine_free():
    hit = sorted(m for m in sys.modules
                 if any(m == s or m.startswith(s + ".") for s in _SPINE))
    assert not hit, ("spine modules leaked into the D-chain release", hit)
    print("  PASS: test_spine_free (no spine modules imported)")


def main():
    test_a1d3_sqed2_odd_rung()
    test_u1a1deven_sqed_even_rung()
    test_spine_free()
    print("\nAll D-type gauged chain (Step 3) tests passed.")


if __name__ == "__main__":
    main()
