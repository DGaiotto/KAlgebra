"""
u1_hexagon_bps_iso.py
=====================

`KAlgebraIso` between `U1HexagonKAlg` and `BPSKAlgebra` on the gauged-hexagon
quiver:  `O_1 → O_2 → O_3 → F`  (3 dynamical nodes + 1 frozen flavour-node
attached at the end of the linear A_3 chain, giving the u(1) gauging).

The BPS algebra is the one originally used to derive `FULL_PLUCKER_TABLE` —
the "crutch" that `U1HexagonKAlg` reimplements in closed form.  Verifying
the iso closes the loop: the BPS-free U1HexagonKAlg is provably equivalent
to its BPS realisation.

Mult-gen correspondence (11 generators):
  L_{1, i}  for i ∈ Z/6  →  letter charge γ(L_{1, i})  ∈ Z⁴
  L_{2, i}  for i ∈ Z/3  →  letter charge γ(L_{2, i})  ∈ Z⁴
  E         (= (3, 0))   →  (1, 0, 1, 0)   (torus direction)
  E⁻¹       (= (3, 1))   →  (-1, 0, -1, 0)

The full forward label map is derived by `KAlgebraIso.from_cone_mult_gen_map`
— canonical-order multiplication on the BPS side reconstructs compound
cone-monomials automatically, with cocycle phases absorbed by BPS.multiply.

Inverse map (BPS charge → U(1)Hex canonical cone-monomial label) still
requires the cone-decomposition step (`_cone_monomial_for_charge`).
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
from u1_hexagon_kalg import U1HexagonKAlg, B_GAUGED
from u1_hexagon_kalg import charge as _charge_of_letter
from u1_hexagon_cone_data import E_GEN, E_INV

# 14 cones (same as cone_algebra.CONES)
_CONES = [
    ((1,1),(1,3),(1,5)), ((1,1),(1,3),(2,0)), ((1,3),(1,5),(2,2)),
    ((1,0),(1,3),(2,2)), ((1,0),(1,3),(2,0)), ((1,1),(1,5),(2,1)),
    ((1,1),(1,4),(2,1)), ((1,1),(1,4),(2,0)), ((1,2),(1,5),(2,1)),
    ((1,2),(1,4),(2,1)), ((1,2),(1,5),(2,2)), ((1,0),(1,2),(2,2)),
    ((1,0),(1,2),(1,4)), ((1,0),(1,4),(2,0)),
]


def _cone_monomial_for_charge(gamma):
    """Inverse: BPS charge γ ∈ Z⁴ → U(1)Hex canonical-form `(factors, e_E)`.
    Picks the lex-min cone decomposition (smallest |factors| + |e_E|, then
    smallest cone-index, then smallest exponents)."""
    best = None
    for cone_idx, cone in enumerate(_CONES):
        g1, g2, g3 = (_charge_of_letter(cone[0]),
                      _charge_of_letter(cone[1]),
                      _charge_of_letter(cone[2]))
        for e1 in range(0, 25):
            for e2 in range(0, 25):
                for e3 in range(0, 25):
                    rem = tuple(gamma[k] - e1*g1[k] - e2*g2[k] - e3*g3[k]
                                for k in range(4))
                    if rem[1] != 0 or rem[3] != 0:
                        continue
                    if rem[0] != rem[2]:
                        continue
                    e_E = rem[0]
                    sz = e1 + e2 + e3 + abs(e_E)
                    cand = (sz, cone_idx, (e1, e2, e3), e_E)
                    if best is None or cand < best:
                        best = cand
    if best is None:
        return None
    _, cone_idx, (e1, e2, e3), e_E = best
    cone = _CONES[cone_idx]
    factors = []
    for k, e in enumerate([e1, e2, e3]):
        if e > 0:
            factors.append((cone[k][0], cone[k][1], e))
    return tuple(sorted(factors)), e_E


def u1_hexagon_bps_iso() -> KAlgebraIso:
    """Construct the U1HexagonKAlg ↔ BPSKAlgebra(gauged-hexagon) iso via
    the cone-data-driven factory."""
    U = U1HexagonKAlg()
    Bp = BPSKAlgebra(
        pairing=B_GAUGED,
        node_charges=[(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)],
    )
    one = LaurentPoly.one()

    # 11-entry mult-gen forward map.  Letter charges come from
    # `_charge_of_letter`; torus directions (E_GEN, E_INV) carry charge
    # (1, 0, 1, 0) and (-1, 0, -1, 0) respectively.
    mult_gen_forward = {}
    for i in range(6):
        mult_gen_forward[(1, i)] = Element({_charge_of_letter((1, i)): one})
    for i in range(3):
        mult_gen_forward[(2, i)] = Element({_charge_of_letter((2, i)): one})
    mult_gen_forward[E_GEN] = Element({(1, 0, 1, 0): one})
    mult_gen_forward[E_INV] = Element({(-1, 0, -1, 0): one})

    def inverse(bps_charge):
        decomp = _cone_monomial_for_charge(tuple(bps_charge))
        if decomp is None:
            raise ValueError(
                f"u1_hexagon_bps_iso.inverse: charge {bps_charge} doesn't "
                f"admit a cone-monomial decomposition."
            )
        return Element({decomp: one})

    return KAlgebraIso.from_cone_mult_gen_map(
        source=U, target=Bp,
        mult_gen_forward=mult_gen_forward,
        target_label_to_source=inverse,
        name="U1HexagonKAlg ≅ BPSKAlgebra(gauged-hexagon)",
    )


if __name__ == "__main__":
    iso = u1_hexagon_bps_iso()
    print(iso)
    U, Bp = iso.source, iso.target

    print("\nForward map on U(1)Hex generators:")
    samples = [("id", ((), 0)),
               ("E",  ((), 1)),
               ("E⁻¹",((), -1))]
    for i in range(6):
        samples.append((f"L_{{1,{i}}}", U.L((1, i))))
    for i in range(3):
        samples.append((f"L_{{2,{i}}}", U.L((2, i))))
    for name, lbl in samples:
        chg = U._charge_of_label(lbl)
        print(f"  {name:<8} = {str(lbl):<22}  →  L_{chg}")

    print(f"\nverify_unit:  {iso.verify_unit()}")

    one = LaurentPoly.one()
    src_samples = [Element({lbl: one}) for _, lbl in samples]
    tgt_samples = [iso.map(s) for s in src_samples]
    rt = iso.verify_round_trip(src_samples, tgt_samples)
    print(f"verify_round_trip on {len(samples)} generators: {rt}")

    src_pairs = [(Element({U.L(la): one}), Element({U.L(lb): one}))
                 for la in [(1, 0), (1, 3), (2, 0), (2, 1)]
                 for lb in [(1, 0), (1, 1), (2, 0), (2, 2)]]
    tgt_pairs = [(iso.map(a), iso.map(b)) for a, b in src_pairs]
    mu = iso.verify_multiplicative(src_pairs, tgt_pairs)
    print(f"verify_multiplicative on {len(src_pairs)} generator pairs: {mu}")
