"""
u1a1aodd_k3_chord_charges.py
============================

Chord-generator BPS charges for U1DecagonKAlg (k=3, u(1)-gauged [A_1, A_7]).

Primitive chord generators (verified via the bigger-than-Plücker
diagnostic):
  L_1 (short,  distance 2, magnetic):     10 generators
  L_2 (medium, distance 3, non-magnetic): 10 generators
  L_3 (with F, magnetic):                 10 generators
  L_4 (diameter, magnetic):                5 generators
  E (central flavour):                    1 (with E^-1)

Total: 35 chord + E^± = 37 mult generators.

Universal seed pattern (for k ≥ 2), as ρ-orbits in the gauged-A_8 BPS quiver:
  L_1[0] = (0, 1, 0, 1, …, 0, 1)         [alternating 01]
  L_2[0] = (1, 0, 0, 0, …, 0, 0)         [single 1 at position 0]
  L_3[0] = (0, 0, -1, 0, -1, …, -1, 1)   [zeros at odd, -1 at even (except 0), +1 at F]
  L_4[0] = (1, 0, 0, 0, 0, 0, 1, 0)      [diameter]
  E      = (1, 0, 1, 0, …, 1, 0)         [alternating 10]

The charges below are **frozen literals** (the ρ-orbits of those seeds),
so importing this module pulls in NO BPS — the standalone decagon is
BPS-free at runtime, matching the octagon (k=2) / dodecagon (k=4).  To
regenerate them from the gauged-A_8 BPS quiver, call `_regenerate_via_bps()`.
"""
from __future__ import annotations


L_1_CHARGES = {
    0: (0, 1, 0, 1, 0, 1, 0, 1),
    1: (0, -1, 0, -1, 0, -1, 0, -1),
    2: (-1, 0, -1, 0, -1, 0, -1, 1),
    3: (1, 0, 1, 0, 1, 0, 1, -1),
    4: (-1, 0, -1, 0, -1, 0, -2, 1),
    5: (1, 0, 1, 0, 1, -1, 2, -1),
    6: (-1, 0, -1, 0, -2, 1, -2, 1),
    7: (1, 0, 1, -1, 2, -1, 2, -1),
    8: (-1, 0, -2, 1, -2, 1, -2, 1),
    9: (1, -1, 2, -1, 2, -1, 2, -1),
}
L_2_CHARGES = {
    0: (1, 0, 0, 0, 0, 0, 0, 0),
    1: (-1, -1, -1, -1, -1, -1, -1, 0),
    2: (0, 0, 0, 0, 0, 0, 1, 0),
    3: (0, 0, 0, 0, 0, 0, -1, 0),
    4: (0, 0, 0, 0, 0, -1, 0, 0),
    5: (0, 0, 0, 0, -1, 0, 0, 0),
    6: (0, 0, 0, -1, 0, 0, 0, 0),
    7: (0, 0, -1, 0, 0, 0, 0, 0),
    8: (0, -1, 0, 0, 0, 0, 0, 0),
    9: (-1, 0, 0, 0, 0, 0, 0, 0),
}
L_3_CHARGES = {
    0: (0, 0, -1, 0, -1, 0, -1, 1),
    1: (0, -1, 0, -1, 0, -1, 1, -1),
    2: (-1, 0, -1, 0, -1, 1, -1, 1),
    3: (1, 0, 1, 0, 1, -1, 1, -1),
    4: (-1, 0, -1, 0, -2, 0, -2, 1),
    5: (1, 0, 1, -1, 1, -1, 2, -1),
    6: (-1, 0, -2, 0, -2, 1, -2, 1),
    7: (1, -1, 1, -1, 2, -1, 2, -1),
    8: (-2, 0, -2, 1, -2, 1, -2, 1),
    9: (2, 0, 2, -1, 2, -1, 2, -1),
}
L_4_CHARGES = {
    0: (1, 0, 0, 0, 0, 0, 1, 0),
    1: (-1, -1, -1, -1, -1, 0, -1, 0),
    2: (0, 0, 0, 0, 1, 0, 1, 0),
    3: (0, 0, 0, 0, -1, 0, -1, 0),
    4: (0, 0, 0, -1, 0, -1, 0, 0),
}

E_CHARGE = (1, 0, 1, 0, 1, 0, 1, 0)

PRIMITIVE_FAMILY_PERIOD = {1: 10, 2: 10, 3: 10, 4: 5}
FAMILY_PERIOD = {1: 10, 2: 10, 3: 10, 4: 5}


def chord_charge(a: int, i: int) -> tuple:
    """BPS charge of chord generator L_a(i).  a ∈ {1, 2, 3, 4}."""
    if a == 1:
        return L_1_CHARGES[i % 10]
    if a == 2:
        return L_2_CHARGES[i % 10]
    if a == 3:
        return L_3_CHARGES[i % 10]
    if a == 4:
        return L_4_CHARGES[i % 5]
    raise ValueError(f"chord family {a} not in {{1, 2, 3, 4}}")


def _regenerate_via_bps():
    """Recompute the frozen charge dicts as ρ-orbits of the universal seeds in
    the gauged-A_8 BPS quiver (offline reproducibility; imports BPS lazily).
    Returns {a: {i: charge}}; asserts they match the frozen literals."""
    from bps_kalgebra import BPSKAlgebra
    n = 8  # 2k+2 nodes = 7 dyn + 1 flavour
    B = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        B[i][i + 1] = 1
        B[i + 1][i] = -1
    bps = BPSKAlgebra(pairing=B,
                      node_charges=[tuple(1 if k == i else 0 for k in range(n))
                                    for i in range(7)],
                      verify="off")

    def orbit(seed, length):
        cur, out = seed, [seed]
        for _ in range(length - 1):
            cur = tuple(bps.rho(cur))
            out.append(cur)
        return {i: c for i, c in enumerate(out)}

    regen = {
        1: orbit((0, 1, 0, 1, 0, 1, 0, 1), 10),
        2: orbit((1, 0, 0, 0, 0, 0, 0, 0), 10),
        3: orbit((0, 0, -1, 0, -1, 0, -1, 1), 10),
        4: orbit((1, 0, 0, 0, 0, 0, 1, 0), 5),
    }
    frozen = {1: L_1_CHARGES, 2: L_2_CHARGES, 3: L_3_CHARGES, 4: L_4_CHARGES}
    assert regen == frozen, "frozen decagon charges differ from BPS ρ-orbits!"
    return regen


# Backward-compat: `_BPS` was a module-level BPSKAlgebra in earlier revisions
# (used only by tests / `U1DecagonKAlg.verify_against_bps`).  Build it lazily
# on attribute access so importing this module stays BPS-free.
def __getattr__(name):
    if name == "_BPS":
        from bps_kalgebra import BPSKAlgebra
        n = 8
        B = [[0] * n for _ in range(n)]
        for i in range(n - 1):
            B[i][i + 1] = 1
            B[i + 1][i] = -1
        return BPSKAlgebra(pairing=B,
                           node_charges=[tuple(1 if k == i else 0 for k in range(n))
                                         for i in range(7)],
                           verify="off")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    print("U1DecagonKAlg chord generators (frozen; 35 mult-gens + E^±):")
    print(f"  E = {E_CHARGE}")
    _regenerate_via_bps()
    print("  frozen charges == gauged-A_8 BPS ρ-orbits  [OK]")
