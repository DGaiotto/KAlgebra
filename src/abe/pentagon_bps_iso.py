"""
pentagon_bps_iso.py
===================

A concrete `KAlgebraIso` between

    PentagonKAlg  (intrinsic chord-pentagon algebra A_𝖖([A_1, A_2]))

and

    BPSKAlgebra(pairing=[[0,1],[-1,0]],
                node_charges=[(1,0),(0,1)])
    (the BPS-quiver realisation for the A_2 = O→O quiver).

There is a 5-element family of such isomorphisms — they are permuted
by the Z_5 ρ-symmetry — and this module picks one canonical
representative.

Generator correspondence (= the only data this file declares):
    L_0 ↔ ( 1,  0)
    L_2 ↔ (-1, -1)
    L_4 ↔ ( 0,  1)
    L_1 ↔ ( 0, -1)
    L_3 ↔ (-1,  0)

The full forward label map is then derived universally by
`KAlgebraIso.from_cone_mult_gen_map` — it lifts the mult-gen
correspondence through `PentagonConeData.to_cone_label` +
canonical-order multiplication on the BPS side, recovering the
cocycle-phased compound-label image automatically.

The inverse direction (BPS charge → pentagon `(i, a, b)`) still
requires the cone-decomposition step — given as
`_bps_to_pent_label` below.
"""
from __future__ import annotations

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from kalgebra_samples import PentagonKAlg
from bps_kalgebra import BPSKAlgebra
from laurent_poly import LaurentPoly


# ---------------------------------------------------------------------------
# Generator correspondence
# ---------------------------------------------------------------------------

CHORD_CHARGE: dict[int, tuple[int, int]] = {
    0: (1, 0),
    2: (-1, -1),
    4: (0, 1),
    1: (0, -1),
    3: (-1, 0),
}


# ---------------------------------------------------------------------------
# Forward helper (kept for callers that want the raw charge; the iso's
# `.map` route does the full Element-level lift through the factory).
# ---------------------------------------------------------------------------

def _pent_to_bps_charge(i: int, a: int, b: int) -> tuple[int, int]:
    """Pentagon label `(i, a, b)` → BPS charge `a·γ_i + b·γ_{i+1}`."""
    gi = CHORD_CHARGE[i % 5]
    gi1 = CHORD_CHARGE[(i + 1) % 5]
    return (a * gi[0] + b * gi1[0],
            a * gi[1] + b * gi1[1])


# ---------------------------------------------------------------------------
# Inverse direction: BPS charge → pentagon canonical label
# ---------------------------------------------------------------------------
#
# Pentagon's canonical basis label is `(i, a, b)` for `L_i^a · L_{i+1}^b`
# with `a, b ≥ 0`.  The inverse map decomposes a BPS charge into one
# of the 5 cones spanned by consecutive `(γ_i, γ_{i+1})` pairs.

def _bps_to_pent_label(charge: tuple[int, int]) -> tuple[int, int, int]:
    """BPS charge `(m, n)` → pentagon canonical label `(i, a, b)`."""
    m, n = charge
    if m == 0 and n == 0:
        return (0, 0, 0)
    for i in range(5):
        gi = CHORD_CHARGE[i]
        gi1 = CHORD_CHARGE[(i + 1) % 5]
        det = gi[0] * gi1[1] - gi[1] * gi1[0]
        if det == 0:
            continue
        a_num = m * gi1[1] - n * gi1[0]
        b_num = -m * gi[1] + n * gi[0]
        if a_num % det != 0 or b_num % det != 0:
            continue
        a = a_num // det
        b = b_num // det
        if a < 0 or b < 0:
            continue
        if a == 0 and b == 0:
            return (0, 0, 0)
        if b == 0:
            return (i, a, 0)
        if a == 0:
            return ((i + 1) % 5, b, 0)
        return (i, a, b)
    raise ValueError(
        f"_bps_to_pent_label: BPS charge {charge} doesn't lie in any of "
        f"the 5 cones spanned by (γ_i, γ_{{i+1}}) pairs."
    )


# ---------------------------------------------------------------------------
# Iso construction (via the factory)
# ---------------------------------------------------------------------------

def pentagon_bps_iso() -> KAlgebraIso:
    """Return the canonical `PentagonKAlg ↔ BPSKAlgebra(A_2-quiver)` iso,
    built via `KAlgebraIso.from_cone_mult_gen_map`."""
    P = PentagonKAlg()
    B = BPSKAlgebra(pairing=[[0, 1], [-1, 0]],
                    node_charges=[(1, 0), (0, 1)])
    one = LaurentPoly.one()

    # Mult-gen forward: each pentagon mult-gen (0..4) maps to its BPS
    # charge as a single-term Element.
    mult_gen_forward = {
        i: Element({CHORD_CHARGE[i]: one}) for i in range(5)
    }

    def inverse(bps_charge):
        return Element({_bps_to_pent_label(tuple(bps_charge)): one})

    return KAlgebraIso.from_cone_mult_gen_map(
        source=P, target=B,
        mult_gen_forward=mult_gen_forward,
        target_label_to_source=inverse,
        name="PentagonKAlg ≅ BPSKAlgebra(A_2-quiver)",
    )


if __name__ == "__main__":
    iso = pentagon_bps_iso()
    print(iso)

    from kalgebra_iso import canonical_iso_samples
    P, B = iso.source, iso.target
    one = LaurentPoly.one()

    samp = canonical_iso_samples(P, include_pairs=True)
    pent_samples = samp["samples"] + [
        Element({(0, 2, 0): one}),
        Element({(0, 1, 1): one}),
        Element({(0, 2, 1): one}),
        Element({(0, 1, 2): one}),
    ]
    bps_samples = [Element({CHORD_CHARGE[i]: one}) for i in range(5)] + [
        Element({(2, 0): one}),
        Element({(1, -1): one}),
        Element({(2, -1): one}),
        Element({(1, -2): one}),
    ]
    pent_pairs = samp["pairs"]
    bps_pairs = [(iso.map(a), iso.map(b)) for a, b in pent_pairs]

    results = iso.verify_all(
        source_samples=pent_samples,
        target_samples=bps_samples,
        source_pairs=pent_pairs,
        target_pairs=bps_pairs,
    )
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")
