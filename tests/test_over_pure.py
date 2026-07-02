"""Self-test for the **over-pure gauge-theory** corner (Step 3) — a
Lagrangian gauge theory built as an RG flow over a *pure-gauge* core, the first
non-AD class here:

    SU(2)×SU(2) + bifundamental   →   SU(2) × SU(2)   (the bifundamental
    (SU2xSU2BifundOverPure)            integrated out)

The bifundamental matter block is the spectrum generator
`S_RG = ∏_{ε₁,ε₂ ∈ {±}} E_𝔮(μ · v₁^{ε₁} v₂^{ε₂})`, expanded over its weights and
peeled to SU(2)₁×SU(2)₂ characters `χ_{w₁}(v₁)χ_{w₂}(v₂) → F⁽¹⁾_{w₁}F⁽²⁾_{w₂}`
(Wilson lines of each pure SU(2)).  The bifundamental's **baryonic U(1)** is
carried as an `add_flavour(U(1))` coefficient, so the bilinear exact-FS trace is
the **μ-refined** Schur index in `R(U(1))`.

Auxiliary = `(pure SU(2) ⊗ pure SU(2)).add_flavour(U(1))`, drawing the pure-SU(2)
cone K-algebra and its analytic Schur trace from Step 2 (`pure_su2_h_cone_data`,
`pure_su2_h_trace_analytic`).  Pure exact-FS (RG solved, no trace override),
fully spine-free.

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

from su2su2_bifund_over_pure import SU2xSU2BifundOverPure, su2su2_bifund_matter_spectrum
from su2_nf_over_pure import SU2NfOverPure
from su2_linear_quiver_over_pure import (
    SU2LinearQuiverOverPure, su2_linear_quiver_matter_spectrum,
)
from habiro import HabiroElement

_SPINE = ("bps_kalgebra", "rg_flow", "lattice_torus", "nahm_data", "chart_graph",
          "bps_quiver_tools", "f_solver", "directional_subquiver_rg", "pure_ade")


def _ser(rps, K):
    return {e: str(r) for e, r in sorted(rps.coeffs.items())
            if e <= K and str(r) not in ("0", "")}


def test_bifund_spectrum():
    """The bifundamental block `Ψ`: vacuum singlet, the `(2,2)` at level 1."""
    S = su2su2_bifund_matter_spectrum(2)
    assert S[(0, 0, 0, 0, 0)] == HabiroElement.one()
    assert S[(0, 1, 0, 1, 1)] == su2su2_bifund_matter_spectrum(1)[(0, 1, 0, 1, 1)]
    print("  PASS: test_bifund_spectrum (vacuum + (2,2) at level 1)")


def test_mu_refined_trace():
    """Pure exact-FS; μ-refined vacuum index `R(U(1))`-valued and
    truncation-stable; q² carries the baryonic `μ^{±2}` currents; the base
    pure-SU(2)×SU(2) index (μ-neutral part) is recovered."""
    A = SU2xSU2BifundOverPure()
    assert A._fs_exact_available()
    for over in ("trace", "inner_product", "rg_s_graded"):
        assert over not in SU2xSU2BifundOverPure.__dict__, f"unexpected override {over}"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v4 = _ser(A.trace(A.identity(), 4), 4)
        nw = len(w)
    assert nw == 0, ("warns", nw)
    q2 = dict(A.trace(A.identity(), 4).coeffs[2].terms)
    assert q2.get((-2,)) and q2.get((2,)), ("q² should carry μ^{±2}", q2)
    assert v4 == _ser(A.trace(A.identity(), 6), 4), "vacuum not truncation-stable"
    print("  PASS: test_mu_refined_trace", v4)


def test_axioms():
    """KAlgebra axioms (bar / RG-multiplicative / orthonormality)."""
    A = SU2xSU2BifundOverPure()
    labs = [A.identity(),
            ((0, 1, 0, 0), (0,)), ((0, 0, 0, 1), (0,)), ((0, 1, 0, 1), (1,))]
    assert A.verify_rg_unital()
    for a in labs:
        assert A.verify_rg_bar_invariant(a), ("bar", a)
        for b in labs:
            assert A.verify_rg_multiplicative(a, b), ("mult", a, b)
            assert A.verify_orthonormality(a, b, 4), ("ortho", a, b)
    print("  PASS: test_axioms")


def test_su2_nf():
    """SU(2)+N_f over pure SU(2): exact-FS, μ-refined trace.  N_f=1: SO(2)=U(1)
    current (q²=1).  N_f=2: the q² currents are the **SO(4) adjoint** —
    `2 + Σ μ_1^{±1}μ_2^{±1}` = 6 (the SU(2)+N_f=2 flavour symmetry)."""
    A1 = SU2NfOverPure(1)
    assert A1._fs_exact_available()
    for over in ("trace", "inner_product", "rg_s_graded"):
        assert over not in SU2NfOverPure.__dict__, f"unexpected override {over}"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v1 = _ser(A1.trace(A1.identity(), 4), 4)
        nw = len(w)
    assert nw == 0 and v1.get(0) == "1" and v1.get(2) == "1", ("N_f=1 vacuum", v1)
    assert v1 == _ser(A1.trace(A1.identity(), 6), 4), "N_f=1 not truncation-stable"
    # N_f=2: the SO(4) flavour adjoint at q² (kept light — only K=2)
    A2 = SU2NfOverPure(2)
    q2 = dict(A2.trace(A2.identity(), 2).coeffs[2].terms)
    charged = {k: v for k, v in q2.items() if k != (0, 0)}
    assert q2.get((0, 0)) == 2 and len(charged) == 4 and all(v == 1 for v in charged.values()), \
        ("N_f=2 q² should be the SO(4) adjoint (2 + 4 charged = 6)", q2)
    print("  PASS: test_su2_nf (N_f=1 U(1) current; N_f=2 SO(4) adjoint at q²)")


def test_linear_quiver():
    """The SU(2)ⁿ linear quiver: n=2 (no flavour) reproduces the bifundamental
    exactly (spectrum + μ-refined vacuum); n=3 / end-flavoured cases construct
    and are exact-FS.  (n≥3 *traces* are correct but slow — the generic exact-FS
    over the n-fold tensor — so only the n=2 trace is exercised here.)"""
    # n=2, no flavour == the SU(2)×SU(2) bifundamental
    assert su2_linear_quiver_matter_spectrum(2, 3) == su2su2_bifund_matter_spectrum(3)
    Q2 = SU2LinearQuiverOverPure(2)
    B = SU2xSU2BifundOverPure()
    assert Q2._fs_exact_available()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vq = _ser(Q2.trace(Q2.identity(), 2), 2)
        nw = len(w)
    assert nw == 0 and vq == _ser(B.trace(B.identity(), 2), 2), \
        ("n=2 quiver vacuum == bifund", vq)
    # n=3 (two links) and an end-flavoured n=2 construct + are exact-FS;
    # the n=3 spectrum carries the two independent link baryon levels.
    Q3 = SU2LinearQuiverOverPure(3)
    assert Q3._fs_exact_available() and Q3._L == 2
    s3 = su2_linear_quiver_matter_spectrum(3, 1)
    assert (0, 1, 0, 1, 0, 0, 1, 0) in s3 and (0, 0, 0, 1, 0, 1, 0, 1) in s3  # the two links
    Qf = SU2LinearQuiverOverPure(2, Nf1=1)
    assert Qf._fs_exact_available() and Qf._L == 2
    print("  PASS: test_linear_quiver (n=2 == bifund; n=3 + end-flavour exact-FS)")


def test_spine_free():
    hit = sorted(m for m in sys.modules
                 if any(m == s or m.startswith(s + ".") for s in _SPINE))
    assert not hit, ("spine modules leaked into the over-pure release", hit)
    print("  PASS: test_spine_free (no spine modules imported)")


def main():
    test_bifund_spectrum()
    test_mu_refined_trace()
    test_axioms()
    test_su2_nf()
    test_linear_quiver()
    test_spine_free()
    print("\nAll over-pure gauge-theory (SU(2)+N_f, bifund, SU(2)ⁿ quiver) Step 3 tests passed.")


if __name__ == "__main__":
    main()
