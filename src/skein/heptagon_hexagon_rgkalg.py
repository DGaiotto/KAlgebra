"""`HeptagonHexagonRGKAlg`: the `Heptagon → Hexagon` RG, built via the
systematic factory.

  * UV: `HeptagonKAlg` = `A_𝖖([A_1, A_4])` ≅ `BPSKAlgebra(A_4-quiver)`.
  * IR: `U1HexagonKAlg` ≅ `BPSKAlgebra(O→O→O→F)` (u(1)-gauged hexagon =
    k=1 member of the U1A1Aodd family).
  * RG: drop the last mutable node `γ_4 = (0,0,0,1)` of the A_4 quiver.

Constructed by:

    SingleNodeRGKAlgebra(A_4-quiver, drop γ_4)
        |  auxiliary() = BPSKAlgebra(O→O→O→F)
        |
        + inverted u1_hexagon_bps_iso  (BPS → U1Hexagon)
        |
        v
    IsoComposedRGKAlgebra
        auxiliary() = U1HexagonKAlg
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra_iso import KAlgebraIso
from u1_hexagon_bps_iso import u1_hexagon_bps_iso
from single_node_rgkalgebra import SingleNodeRGKAlgebra
from iso_composed_rgkalgebra import IsoComposedRGKAlgebra


# A_4 antisymmetric pairing on Z^4 (= B_GAUGED).
A4_PAIRING = [
    [ 0,  1,  0,  0],
    [-1,  0,  1,  0],
    [ 0, -1,  0,  1],
    [ 0,  0, -1,  0],
]
A4_NODE_CHARGES = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
GAMMA_DROP = (0, 0, 0, 1)  # γ_4 — the node we delete to flow to U1Hexagon

# A_4 positive roots, arranged STRICTLY as [stuff_containing_γ_4, ir_spec].
# Positive roots of A_4 (10 total):
#   length-1: γ_1, γ_2, γ_3, γ_4
#   length-2: γ_1+γ_2, γ_2+γ_3, γ_3+γ_4
#   length-3: γ_1+γ_2+γ_3, γ_2+γ_3+γ_4
#   length-4: γ_1+γ_2+γ_3+γ_4
# Containing γ_4: γ_4, γ_3+γ_4, γ_2+γ_3+γ_4, γ_1+γ_2+γ_3+γ_4  (4 entries)
# Not containing γ_4: the other 6.
A4_SPEC_STUFF_FIRST = [
    # stuff (containing γ_4) — descending "support starting from γ_4"
    (0, 0, 0, 1),  # γ_4
    (0, 0, 1, 1),  # γ_3+γ_4
    (0, 1, 1, 1),  # γ_2+γ_3+γ_4
    (1, 1, 1, 1),  # γ_1+γ_2+γ_3+γ_4
    # ir_spec (not containing γ_4) — A_3 positive-root chamber
    (0, 0, 1, 0),  # γ_3
    (0, 1, 1, 0),  # γ_2+γ_3
    (1, 1, 1, 0),  # γ_1+γ_2+γ_3
    (0, 1, 0, 0),  # γ_2
    (1, 1, 0, 0),  # γ_1+γ_2
    (1, 0, 0, 0),  # γ_1
]


class HeptagonHexagonRGKAlg(IsoComposedRGKAlgebra):
    """The `Heptagon → Hexagon` RGKAlgebra, IR = `U1HexagonKAlg`.

    Constructed by `IsoComposedRGKAlgebra(SingleNodeRGKAlgebra(A_4-quiver,
    drop γ_4), inverted u1_hexagon_bps_iso)`.

    UV labels are BPS-charge tuples `(a, b, c, d)` ∈ Z^4 (the lattice
    form of HeptagonKAlg via the standard A_4 quiver iso).
    """

    def __init__(self) -> None:
        inner = SingleNodeRGKAlgebra(
            pairing=A4_PAIRING,
            node_charges=A4_NODE_CHARGES,
            spec=A4_SPEC_STUFF_FIRST,
            gamma_drop=GAMMA_DROP,
        )
        ub = u1_hexagon_bps_iso()
        # ub runs U1HexagonKAlg → BPSKAlgebra(O→O→O→F); we need the
        # other direction with source = inner.auxiliary() (identity).
        inverted_iso = KAlgebraIso(
            source=inner.auxiliary(),
            target=ub.source,  # U1HexagonKAlg
            forward_label_map=ub._inverse,
            inverse_label_map=ub._forward,
            name="BPSKAlgebra(O→O→O→F) → U1HexagonKAlg",
        )
        super().__init__(inner, inverted_iso)
