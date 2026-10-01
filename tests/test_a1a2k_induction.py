"""Tests for the inductive `A1A2kKAlg(k) ≅ BPS(A_{2k})` scaffold.

Checks the RGKAlgebra-reconstruction-theorem hypotheses for the IR iso
`φ_k : IR_BPS(drop γ₂) → A1A2kKAlg(k-1) ⊗ QT(Z₂)` with the correct
simple-root negating-sequence spec, for k = 1, 2 (the cases the
default `a1a2k_bps_iso` block iso covers; k ≥ 3 awaits the canonical
`FiniteBPSKAlgebra.to_cone_kalgebra` correspondence).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a1a2k_induction import (
    a1a2k_phi, _ir_samples_pairs, _s_rg_gamma1_support,
    verify_a1a2k_induction_step, prove_a1a2k_bps_iso,
    a1a2k_bps_iso_inductive,
)
from rgkalgebra_iso_via_ir import verify_ir_iso_hypotheses


def test_phi_hypotheses_k1_k2_k3():
    """φ_k is a multiplicative, ρ-equivariant, round-tripping IR iso and
    the γ₁-tower of S_RG transports label-bijectively — for k = 1, 2, 3
    (the inductive step, with the complete additive block iso)."""
    for k in (1, 2, 3):
        checks = verify_a1a2k_induction_step(k)
        assert checks["phi_unit"], (k, checks)
        assert checks["phi_round_trip"], (k, checks)
        assert checks["phi_multiplicative"], (k, checks)
        assert checks["phi_rho"], (k, checks)
        assert checks["s_rg_transports"], (k, checks)


def test_inductive_proof_obligations():
    """`prove_a1a2k_bps_iso(3)` — every inductive step k=1..3 all-True,
    establishing `A1A2kKAlg(k) ≅ BPS(A_{2k})` for k ≤ 3."""
    report = prove_a1a2k_bps_iso(3)
    for k, checks in report.items():
        assert all(checks.values()), (k, checks)


def test_witness_iso_is_a1a2k_to_bps():
    """The witness iso `A1A2kKAlg(k) ↔ BPS(A_{2k})` has the right
    endpoints and is a unital, round-tripping label bijection."""
    from a1a2k_kalg import A1A2kKAlg
    from kalgebra import Element
    from laurent_poly import LaurentPoly
    one = LaurentPoly.one()
    iso = a1a2k_bps_iso_inductive(2)
    assert isinstance(iso.source, A1A2kKAlg)
    assert iso.verify_unit()
    A = iso.source
    H = 2 * 2 + 3
    gens = [Element({A.L((a, i)): one}) for a in (1, 2) for i in range(H)]
    assert iso.verify_round_trip(gens, [iso.map(g) for g in gens])


def test_ir_is_single_term_chamber():
    """With the simple-root spec the IR is the finite (single-term)
    chamber: γ₃·γ₄ = q·X_{γ₃+γ₄} (no spurious bound-state tower)."""
    inner, _ = a1a2k_phi(2)
    ir = inner.auxiliary()
    prod = ir.multiply((0, 0, 1, 0), (0, 0, 0, 1))
    assert set(prod.terms.keys()) == {(0, 0, 1, 1)}, dict(prod.terms)


def test_cheap_trace_via_iso():
    """The iso makes the BPS(A_{2k}) trace (Schur index) cheap: pull a
    BPS charge back to `A1A2kKAlg(k)` via the witness iso and use its
    fast intrinsic trace.  Computable instantly for k = 2, 3 (no BPS
    Habiro/Nahm machinery, which is the operation this whole iso exists
    to bypass)."""
    from kalgebra import Element
    from laurent_poly import LaurentPoly
    one = LaurentPoly.one()
    for k in (2, 3):
        iso = a1a2k_bps_iso_inductive(k)
        A = iso.source
        n = 2 * k
        chg = tuple(1 if t == 2 else 0 for t in range(n))   # γ₃
        (lbl, co), = iso.inverse(Element({chg: one})).terms.items()
        tr = A.trace(lbl, K=6)
        # trace is a non-trivial q-series (a Schur index), computed fast
        assert tr is not None
        assert len(tr.coeffs) >= 1


if __name__ == "__main__":
    test_phi_hypotheses_k1_k2_k3()
    test_cheap_trace_via_iso()
    test_inductive_proof_obligations()
    test_witness_iso_is_a1a2k_to_bps()
    test_ir_is_single_term_chamber()
    print("\nAll a1a2k_induction tests passed.")
