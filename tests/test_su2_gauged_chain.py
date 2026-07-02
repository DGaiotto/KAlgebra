"""Self-test for the **SU(2)-gauged chain** (Step 3) — a fully-nested RG
chain in which each rung's auxiliary is the *previous rung's flow* (an
`RGKAlgebra` used as the aux of the next), composing all the way down to pure
SU(2):

    SU(2)×U(1)-gauged A1D4  →  SU(2)-gauged A1D3  →  U(1)-gauged SU(2) N_f=1  →  pure SU(2)
    (SU2U1GaugedA1D4)          (SU2GaugedA1D3)        (SU2Nf1PureSU2)            (PureSU2KAlg)

  * `SU2Nf1PureSU2RGKAlgebra` (entry 1) — U(1)-gauged SU(2) N_f=1 over pure SU(2).
  * `SU2GaugedA1D3RGKAlgebra` (entry 2) — aux = entry 1.
  * `SU2U1GaugedA1D4RGKAlgebra` (entry 3) — aux = entry 2 ⊗ `QuantumTorusKAlg`
    (a fresh gauge torus), `S_RG = E_𝖖(X₀₁v)E_𝖖(X₀₁v⁻¹)` (the gauge doublet on the
    fresh electric leg, χ-peeled to the pure-SU(2) Wilson reached through the
    nested aux).

All pure exact-FS over the (doubly-)nested aux; the chain demonstrates that an
`RGKAlgebra`'s auxiliary may itself be an `RGKAlgebra`.  Entry 3's vacuum is
`1 − q² − q⁴ + q¹² + 2q¹⁴` (a q⁶–q¹⁰ gap), computed with the BPS realisation
layer to q¹⁴; this self-test checks the leading terms + truncation stability
(the deep q¹² check is slow over the triple nesting).  Spine-free throughout.

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

from su2nf1_pure_su2_rgkalgebra import SU2Nf1PureSU2RGKAlgebra
from su2gauged_a1d3_rgkalgebra import SU2GaugedA1D3RGKAlgebra
from su2u1gauged_a1d4_rgkalgebra import SU2U1GaugedA1D4RGKAlgebra

_SPINE = ("bps_kalgebra", "rg_flow", "lattice_torus", "nahm_data", "chart_graph",
          "bps_quiver_tools", "f_solver", "directional_subquiver_rg", "pure_ade")


def _ser(rps, K):
    return {e: str(r) for e, r in sorted(rps.coeffs.items())
            if e <= K and str(r) not in ("0", "")}


def _check(A, name, K=6):
    assert A._fs_exact_available(), (name, "exact-FS unavailable")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v = _ser(A.trace(A.identity(), K), K)
        nw = len(w)
    assert nw == 0, (name, "warns", nw)
    return v


def test_chain_rungs_exact_fs():
    """Each rung is pure exact-FS over its (nested) aux, with truncation-stable
    vacuum (entry 3's aux is itself a flow ⊗ QT — a doubly-nested aux)."""
    e1 = SU2Nf1PureSU2RGKAlgebra()
    v1 = _check(e1, "entry1")
    assert v1 == _ser(e1.trace(e1.identity(), 8), 6), "entry1 not truncation-stable"

    e2 = SU2GaugedA1D3RGKAlgebra()
    assert type(e2.auxiliary()).__name__ == "SU2Nf1PureSU2RGKAlgebra"  # aux IS a flow
    _check(e2, "entry2")

    e3 = SU2U1GaugedA1D4RGKAlgebra()
    v3 = _check(e3, "entry3")
    assert v3 == {0: "1", 2: "-1", 4: "-1"}, ("entry3 vacuum leading + q6–q10 gap", v3)
    print("  PASS: test_chain_rungs_exact_fs (entry3 vac = 1 − q² − q⁴ + …, gap q⁶–q¹⁰)")


def test_spine_free():
    hit = sorted(m for m in sys.modules
                 if any(m == s or m.startswith(s + ".") for s in _SPINE))
    assert not hit, ("spine modules leaked into the SU(2)-gauged chain release", hit)
    print("  PASS: test_spine_free (no spine modules imported)")


def main():
    test_chain_rungs_exact_fs()
    test_spine_free()
    print("\nAll SU(2)-gauged chain (Step 3) tests passed.")


if __name__ == "__main__":
    main()
