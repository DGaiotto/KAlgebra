"""Tests for `rgkalgebra_iso_via_ir` — the formalisation of
'RGKAlgebra is determined up to iso by (IR, RG, S_RG)'.

Exercised on the heptagon→hexagon single-node RG (terminal drop, so
fast): `inner = SingleNodeRGKAlgebra(A_4, drop γ_4)` with
`φ : IR_BPS → U1HexagonKAlg`.  We check the theorem's hypotheses on
IR-side samples and that the UV iso witness is unital and round-trips.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heptagon_hexagon_rgkalg import HeptagonHexagonRGKAlg
from rgkalgebra_iso_via_ir import rgkalgebra_iso_via_ir, verify_ir_iso_hypotheses
from kalgebra import Element
from laurent_poly import LaurentPoly

_ONE = LaurentPoly.one()


def _setup():
    HHR = HeptagonHexagonRGKAlg()
    inner = HHR._inner
    phi = HHR._iso
    ir = inner.auxiliary()
    nodes = list(ir.node_charges)
    ir_samples = [Element({ir.identity(): _ONE})] + \
        [Element({n: _ONE}) for n in nodes]
    # small pair set (keeps the U1Hexagon-side multiply cheap)
    ir_pairs = [(Element({nodes[0]: _ONE}), Element({n: _ONE}))
                for n in nodes]
    # S_RG = E_q(X_{γ_4}); support is the dropped-node tower {n·γ_4}.
    s_rg_support = [(0, 0, 0, n) for n in range(0, 4)]
    return HHR, inner, phi, ir_samples, ir_pairs, s_rg_support


def test_ir_iso_hypotheses():
    """φ is a multiplicative, unital, round-tripping KAlgebraIso of the
    IR, and S_RG transports label-bijectively across it."""
    _, inner, phi, ir_samples, ir_pairs, s_rg_support = _setup()
    checks = verify_ir_iso_hypotheses(
        inner, phi, ir_samples, ir_pairs,
        s_rg_support=s_rg_support, rho=False,
    )
    assert checks["phi_unit"], checks
    assert checks["phi_round_trip"], checks
    assert checks["phi_multiplicative"], checks
    assert checks["s_rg_transports"], checks


def test_uv_iso_witness():
    """The induced UV iso `inner ≅ IsoComposedRGKAlgebra(inner, φ)` is
    unital and round-trips on UV labels (identity-label map)."""
    HHR, inner, phi, *_ = _setup()
    uviso = rgkalgebra_iso_via_ir(inner, phi, transported=HHR)
    assert uviso.verify_unit()
    uv = [inner.identity()] + list(inner._uv_ext.node_charges)
    samp = [Element({l: _ONE}) for l in uv]
    assert uviso.verify_round_trip(samp, [uviso.map(s) for s in samp])


if __name__ == "__main__":
    test_ir_iso_hypotheses()
    test_uv_iso_witness()
    print("\nAll rgkalgebra_iso_via_ir tests passed.")
