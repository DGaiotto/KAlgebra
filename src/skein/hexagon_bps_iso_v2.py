"""
hexagon_bps_iso_v2.py
=====================

Alternative `KAlgebraIso` for `HexagonKAlg`: target is `BPSKAlgebra` on the
**linear A_3 BPS quiver** O → O → O (no frozen flavour node), per user
suggestion that the ungauging "can be done in parallel in the iso for
U(1)Hex" (verified in `u1_hexagon_bps_iso.py`).

The construction mirrors `u1_hexagon_bps_iso.py` exactly, with two parallel
ungauging steps:
  - U(1)Hex side: restrict cone monomials to mag-zero ⇒ HexagonKAlg.
  - BPS side: drop the frozen flavour node F (3D charge lattice instead
    of 4D).  The mu-flavour appears via B_UNGAUGED's rank-1 kernel.

Charge correspondence: project the U(1)Hex 4D charge to the first 3
coordinates (the linear-A_3 dynamical-node directions).  The 4th coordinate
becomes μ-flavour, which is in the B-kernel of the ungauged quiver too.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly
from bps_kalgebra import BPSKAlgebra
from hexagon_kalg import HexagonKAlg
from u1_hexagon_kalg import U1HexagonKAlg
from u1_hexagon_bps_iso import _cone_monomial_for_charge


B_UNGAUGED = [[0, 1, 0], [-1, 0, 1], [0, -1, 0]]
NODE_CHARGES_UNGAUGED = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def hexagon_bps_iso_v2() -> KAlgebraIso:
    """HexagonKAlg ↔ BPSKAlgebra(linear-A_3) via parallel ungauging of
    `u1_hexagon_bps_iso.u1_hexagon_bps_iso`.

    Forward: Hex label (factors, μ_charge) → compute U(1)Hex charge (4D),
    project first 3 coordinates → BPS linear-A_3 charge.

    Inverse: BPS charge (3D) → lift to U(1)Hex 4D charge by appending 0
    in the 4th slot, then `_cone_monomial_for_charge` to a Hex cone monomial.
    """
    H = HexagonKAlg()
    U = U1HexagonKAlg()
    Bp = BPSKAlgebra(pairing=B_UNGAUGED, node_charges=NODE_CHARGES_UNGAUGED)
    one = LaurentPoly.one()

    def forward(hex_label):
        # Hex label = (factors, μ_charge), valid as a U(1)Hex label too
        u1_charge_4d = U._charge_of_label(hex_label)
        # Project to first 3 coords (linear-A_3 dynamical-node directions).
        bps_charge_3d = u1_charge_4d[:3]
        return Element({bps_charge_3d: one})

    def inverse(bps_charge):
        # Lift 3D → 4D by appending 0 in the 4th slot.
        u1_charge_4d = tuple(bps_charge) + (0,)
        decomp = _cone_monomial_for_charge(u1_charge_4d)
        if decomp is None:
            raise ValueError(
                f"hexagon_bps_iso_v2.inverse: charge {bps_charge} doesn't lift."
            )
        # Verify the resulting Hex label is mag-zero (else not a valid Hex basis)
        try:
            H._assert_mag_zero(decomp, "iso inverse")
        except ValueError:
            return Element({})  # not in Hex's image
        return Element({decomp: one})

    return KAlgebraIso(
        H, Bp,
        forward_label_map=forward,
        inverse_label_map=inverse,
        name="HexagonKAlg ≅ BPSKAlgebra(linear-A_3)",
    )


if __name__ == "__main__":
    iso = hexagon_bps_iso_v2()
    print(iso)
    H, Bp = iso.source, iso.target

    print("\nForward map on Hex mult-generators:")
    samples = [("id", H.identity()), ("μ", ((), 1))]
    for j in range(3): samples.append((f"L_long({j})", H.L_long(j)))
    for i in range(3): samples.append((f"L_diam({i})", H.L_diam(i)))
    for name, lbl in samples:
        u1_chg = U1HexagonKAlg()._charge_of_label(lbl)
        bps_chg = u1_chg[:3]
        print(f"  {name:<10} = {str(lbl):<30}  →  L_{bps_chg}   (4D: {u1_chg})")

    print(f"\nverify_unit: {iso.verify_unit()}")

    one = LaurentPoly.one()
    src_samples = [Element({lbl: one}) for _, lbl in samples]
    tgt_samples = [iso.map(s) for s in src_samples]
    rt = iso.verify_round_trip(src_samples, tgt_samples)
    print(f"verify_round_trip on {len(samples)} generators: {rt}")

    # verify_multiplicative on generator pairs (only mag-zero combos!)
    gens = H.mult_generators()
    src_pairs = [(Element({a: one}), Element({b: one})) for a in gens for b in gens]
    tgt_pairs = [(iso.map(a), iso.map(b)) for a, b in src_pairs]
    mu = iso.verify_multiplicative(src_pairs, tgt_pairs)
    print(f"verify_multiplicative on 36 Hex generator pairs: {mu}")
