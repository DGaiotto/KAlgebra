"""
a1a2k_bps_iso.py
================

`KAlgebraIso` family between `A1A2kKAlg(k)` and `BPSKAlgebra` on the
linear A_{2k} = O→O→…→O quiver (2k nodes).

For each k ≥ 1:
  source = A1A2kKAlg(k)   (intrinsic, no BPS runtime — H = 2k + 3 chord
                            positions, k chord-length-orbits;
                            total k·(2k+3) named generators)
  target = BPSKAlgebra(pairing = A_{2k} quiver, node_charges = identity in Z^{2k})

The mult-gen correspondence is computed via `natural_orbit_seeds(k)` +
BPS ρ-iteration; for k = 2 there is an empirical +3 shift on orbit 2.

The full forward label map is derived universally by
`KAlgebraIso.from_cone_mult_gen_map` from the k·(2k+3)-entry mult-gen
correspondence dict.  The inverse direction is a brute-force charge
search.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from a1a2k_kalg import A1A2kKAlg
from bps_kalgebra import BPSKAlgebra
from laurent_poly import LaurentPoly


def _a4_pairing(n: int):
    """A_n linear-quiver pairing."""
    B = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        B[i][i+1] = 1
        B[i+1][i] = -1
    return B


def _standard_basis(n: int):
    return [tuple(1 if k == i else 0 for k in range(n)) for i in range(n)]


def _compute_chord_charges(k: int, orbit_shifts: dict[int, int]):
    """Build the (a, i) → BPS-charge dict for A1A2kKAlg(k).

    Uses the closed-form, **BPS-free** lattice ρ (`a2k_rho` = `sigma_forward`
    on the linear-quiver spec) — no `BPSKAlgebra` is constructed."""
    from A1A2k_naming_audit import natural_orbit_seeds, a2k_rho
    seeds = natural_orbit_seeds(k)
    H = 2 * k + 3
    out: dict[tuple[int, int], tuple[int, ...]] = {}
    for a, seed in seeds.items():
        natural = {0: tuple(seed)}
        cur = tuple(seed)
        for i in range(1, H):
            cur = tuple(a2k_rho(k, cur))
            natural[i] = cur
        shift = orbit_shifts.get(a, 0)
        for j in range(H):
            out[(a, j)] = natural[(j - shift) % H]
    return out


def a1a2k_bps_iso(k: int, orbit_shifts: dict[int, int] | None = None) -> KAlgebraIso:
    """Build the A1A2kKAlg(k) ↔ BPSKAlgebra(A_{2k}-quiver) iso via the
    cone-data-driven factory.

    `orbit_shifts`: optional {a: shift_in_Z/(2k+3)} per orbit a ∈ {1, ..., k}.
    Default shift is 0 (= natural-seed alignment).
    """
    if orbit_shifts is None:
        orbit_shifts = {a: 0 for a in range(1, k + 1)}
    H = 2 * k + 3
    n_nodes = 2 * k

    A = A1A2kKAlg(k)
    Bp = BPSKAlgebra(
        pairing=_a4_pairing(n_nodes),
        node_charges=_standard_basis(n_nodes),
        verify="off",
    )
    one = LaurentPoly.one()

    chord_charges = _compute_chord_charges(k, orbit_shifts)

    # k·(2k+3)-entry mult-gen forward map.
    mult_gen_forward = {
        (a, i): Element({chord_charges[(a, i)]: one})
        for (a, i) in chord_charges
    }

    def charge_to_label(charge):
        target = tuple(charge)
        if all(c == 0 for c in target):
            return ()
        for (a, i), c in chord_charges.items():
            if c == target:
                return ((a, i, 1),)
        for (ka, ia), ca in chord_charges.items():
            for (kb, ib), cb in chord_charges.items():
                if (ka, ia) >= (kb, ib): continue
                if tuple(ca[j] + cb[j] for j in range(n_nodes)) == target:
                    return tuple(sorted([(ka, ia, 1), (kb, ib, 1)]))
        for (a, i), c in chord_charges.items():
            for e in range(2, 6):
                if tuple(e * c[j] for j in range(n_nodes)) == target:
                    return ((a, i, e),)
        raise ValueError(
            f"a1a2k_bps_iso(k={k}).inverse: charge {charge} not in brute-force range."
        )

    def inverse(chg):
        return Element({charge_to_label(tuple(chg)): one})

    return KAlgebraIso.from_cone_mult_gen_map(
        source=A, target=Bp,
        mult_gen_forward=mult_gen_forward,
        target_label_to_source=inverse,
        name=f"A1A2kKAlg(k={k}) ≅ BPSKAlgebra(A_{2*k}-quiver)",
    )


if __name__ == "__main__":
    print("=" * 60)
    print("k = 1 (Pentagon-equivalent):")
    iso1 = a1a2k_bps_iso(1)
    print(iso1)
    print(f"  verify_unit:  {iso1.verify_unit()}")
    one = LaurentPoly.one()
    A1 = iso1.source
    gens1 = [Element({A1.L((1, i)): one}) for i in range(5)]
    print(f"  verify_round_trip on 5 generators: "
          f"{iso1.verify_round_trip(gens1, [iso1.map(g) for g in gens1])}")
    pairs1 = [(a, b) for a in gens1 for b in gens1]
    print(f"  verify_multiplicative on 25 generator pairs: "
          f"{iso1.verify_multiplicative(pairs1, [(iso1.map(a), iso1.map(b)) for a,b in pairs1])}")

    print()
    print("=" * 60)
    print("k = 2 (A1A2kKAlg(k=2), no shifts):")
    iso2 = a1a2k_bps_iso(2, orbit_shifts={1: 0, 2: 0})
    print(iso2)
    print(f"  verify_unit:  {iso2.verify_unit()}")
    A2 = iso2.source
    gens2 = [Element({A2.L((k, i)): one}) for k in [1, 2] for i in range(7)]
    print(f"  verify_round_trip on 14 generators: "
          f"{iso2.verify_round_trip(gens2, [iso2.map(g) for g in gens2])}")
    pairs2 = [(a, b) for a in gens2 for b in gens2]
    print(f"  verify_multiplicative on 196 generator pairs: "
          f"{iso2.verify_multiplicative(pairs2, [(iso2.map(a), iso2.map(b)) for a,b in pairs2])}")
