"""
u1a1aodd_k2_chord_charges.py
============================

Chord-generator BPS charges for `U1A1Aodd_k2` (u(1)-gauged [A_1, A_5]).

The octagon (H = 2k+4 = 8) has chord types at cyclic distance 2, 3, 4.
**Primitive** chord generators for the U1OctaKAlg multiplicative
structure (verified by exhaustive BPS-multiply analysis):

  L_1 (short,  distance 2): 8 generators, ρ-orbit of (0,1,0,1,0,1)
  L_2 (medium, distance 3): 8 generators, ρ-orbit of (1,0,0,0,0,0)
  E (central flavour): (1, 0, 1, 0, 1, 0)

Total: 16 primitive chord generators + E^± = 18 mult generators.

**L_3 (diameters, distance 4) are NOT primitive — they are composite.**
The bigger-than-Plücker behaviour (4-term and 12-term BPS multiplies)
seen with L_3 in the generator set indicated that L_3 is a product of
simpler chords.  Explicit factorisation:

    L_3(0) = E    · L_2(3) · L_2(7)
    L_3(1) = E⁻¹ · L_2(0) · L_2(4)
    L_3(2) = E    · L_2(1) · L_2(5)
    L_3(3) = E⁻¹ · L_2(2) · L_2(6)

Geometrically: an octagon diameter (i, i+4) is the product of two
opposite medium chords L_2(j) and L_2(j+4) that both pass through the
center where the diameter would lie.

With L_3 demoted to composite, all 256 = 16² primitive chord-pair
BPS products are clean 1-term (q-commute) or 2-term (Plücker), and
their full closed-form table lives in `u1_octagon_mult_table.py`.

Chord positions on the octagon (cyclic Z_8):
  L_1(i) ↔ chord (i, i+2)
  L_2(i) ↔ chord (i, i+3)
  L_3(i) ↔ diameter (i, i+4) [composite via L_2·L_2·E]
"""
from __future__ import annotations


# ---- Primitive chord charges -----------------------------------------------

L_1_CHARGES: dict[int, tuple[int, ...]] = {
    0: (0,  1,  0,  1,  0,  1),
    1: (0, -1,  0, -1,  0, -1),
    2: (-1, 0, -1,  0, -1,  1),
    3: (1,  0,  1,  0,  1, -1),
    4: (-1, 0, -1,  0, -2,  1),
    5: (1,  0,  1, -1,  2, -1),
    6: (-1, 0, -2,  1, -2,  1),
    7: (1, -1,  2, -1,  2, -1),
}

L_2_CHARGES: dict[int, tuple[int, ...]] = {
    0: (1,  0,  0,  0,  0,  0),
    1: (-1, -1, -1, -1, -1, 0),
    2: (0,  0,  0,  0,  1,  0),
    3: (0,  0,  0,  0, -1,  0),
    4: (0,  0,  0, -1,  0,  0),
    5: (0,  0, -1,  0,  0,  0),
    6: (0, -1,  0,  0,  0,  0),
    7: (-1, 0,  0,  0,  0,  0),
}

E_CHARGE: tuple[int, ...] = (1, 0, 1, 0, 1, 0)


# ---- TRUE primitive L_3 diameter charges (8 generators, mod-E orbit) -------
#
# DISCOVERED LATE-SESSION: L_3 IS a primitive chord family — 8 generators
# with clean 1- or 2-term BPS multiplies (mirroring L_1 structure).  The
# "composite L_3" charges below (L_3_COMPOSITE_*) are a SEPARATE set of
# canonical-basis elements at different lattice charges — not the true
# diameter chord generators.
#
# The true L_3 family is the ρ-orbit of (0, 0, -1, 0, -1, 1), closing
# mod E at period 8 with -2E cocycle (analogous to L_1's structure).
#
# Surfaced by the user via "Plücker by definition should return linear
# combo of cone monomials only" — when L_1(0)·L_2(1) Plücker produced
# a term at charge (0, 0, -1, 0, -1, 1) that didn't decompose into
# q-commuting L_1+L_2+E products, that term IS a primitive L_3.

L_3_CHARGES: dict[int, tuple[int, ...]] = {
    0: (0,  0, -1,  0, -1,  1),
    1: (0, -1,  0, -1,  1, -1),
    2: (-1, 0, -1,  1, -1,  1),
    3: (1,  0,  1, -1,  1, -1),
    4: (-1, 0, -2,  0, -2,  1),
    5: (1, -1,  1, -1,  2, -1),
    6: (-2, 0, -2,  1, -2,  1),
    7: (2,  0,  2, -1,  2, -1),
}


# ---- Legacy "composite L_3" (= E^±·L_2·L_2) — kept for back-reference -----
# These are DIFFERENT canonical-basis elements at different charges.
# Not the true L_3 chord generators; rather products of two opposite L_2's
# with an E factor.

L_3_COMPOSITE_CHARGES: dict[int, tuple[int, ...]] = {
    0: (0,  0,  1,  0,  0,  0),
    1: (0,  0, -1, -1, -1,  0),
    2: (0, -1, -1, -1,  0,  0),
    3: (-1, -1, -1, 0,  0,  0),
}

L_3_COMPOSITE_FACTORISATION: dict[int, tuple[int, tuple[int, int]]] = {
    0: (+1, (3, 7)),
    1: (-1, (0, 4)),
    2: (+1, (1, 5)),
    3: (-1, (2, 6)),
}


# ---- Family metadata --------------------------------------------------------

# Primitive families: L_1, L_2, L_3 (TRUE diameter, NOT the composite).
PRIMITIVE_FAMILY_PERIOD = {1: 8, 2: 8, 3: 8}

# All families — same as primitive now (no separate composite family).
FAMILY_PERIOD = {1: 8, 2: 8, 3: 8}


def chord_charge(a: int, i: int) -> tuple[int, ...]:
    """BPS charge of the chord generator L_a(i).

    a=1: short (primitive).  a=2: medium (primitive).  a=3: diameter
    (COMPOSITE — see `L_3_FACTORISATION` for the L_2·L_2·E^± form).
    """
    if a == 1:
        return L_1_CHARGES[i % 8]
    if a == 2:
        return L_2_CHARGES[i % 8]
    if a == 3:
        return L_3_CHARGES[i % 8]
    raise ValueError(f"chord family {a} not in {{1, 2, 3}}")


def l3_composite_as_l2_product(i: int) -> tuple[int, tuple[tuple[int, int], tuple[int, int]]]:
    """For the LEGACY 'composite L_3' charges (= L_3_COMPOSITE_CHARGES),
    return the (e_E, ((2, j_1), (2, j_2))) expression as L_2 · L_2 · E^{e_E}.
    These are NOT the true L_3 chord generators — see L_3_CHARGES."""
    i = i % 4
    e_E, (j1, j2) = L_3_COMPOSITE_FACTORISATION[i]
    L_3_chg = L_3_COMPOSITE_CHARGES[i]
    composite_chg = tuple(
        L_2_CHARGES[j1][k] + L_2_CHARGES[j2][k] + e_E * E_CHARGE[k]
        for k in range(6)
    )
    assert composite_chg == L_3_chg
    return (e_E, ((2, j1), (2, j2)))


if __name__ == "__main__":
    print("U1A1Aodd_k2 chord generators (TRUE primitive set, 24 + E^±):")
    print(f"  E = {E_CHARGE}")
    print()
    for a in (1, 2, 3):
        print(f"  L_{a} (period {PRIMITIVE_FAMILY_PERIOD[a]}):")
        for i in range(PRIMITIVE_FAMILY_PERIOD[a]):
            print(f"    L_({a}, {i}) = {chord_charge(a, i)}")
    print()
    print("Legacy 'composite L_3' charges (= L_2·L_2·E^±, DIFFERENT from true L_3):")
    for i in range(4):
        e_E, factors = l3_composite_as_l2_product(i)
        sign = "+" if e_E > 0 else "-"
        print(f"  L_3_composite({i}) = E^{sign}1 · L_{factors[0]} · L_{factors[1]}")
        print(f"                charge = {L_3_COMPOSITE_CHARGES[i]}")
