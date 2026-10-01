"""Heptagon K-algebra wrapper around BPSKAlgebra.

The heptagon is the K-theoretic Coulomb-branch algebra of the
`(A_1, A_4)` Argyres-Douglas theory.  Its BPS quiver is the linear
A_4 chain

    o > o > o > o

with simple-root node charges `e_1, e_2, e_3, e_4`.  The DT
transformation `ρ` has order 7, and there are two distinguished
`ρ`-orbits of "canonical" basis charges, both of length 7, indexed
here as

    L((1, i)) := ρⁱ (1, 0, 1, 0)        i = 0, ..., 6
    L((2, i)) := ρⁱ (1, 0, 0, 0)        i = 0, ..., 6.

The orbit index `k ∈ {1, 2}` matches the author's notation; orbit 1 is
the (1,0,1,0) orbit and orbit 2 is the simple-root (1,0,0,0) orbit.

Geometrically the named L's correspond to chords on a 7-vertex cyclic
polygon: orbit 1 is the **length-2** chord `{i, i+2}` (short diagonal),
and orbit 2 is the **length-3** chord `{i+4, i}` (long diagonal — the
orbit-2 index is offset by 3 from the chord's canonical starting
vertex, equivalently `L((2, j))` is the long diagonal `(j - 3, j) mod 7`).
The chord endpoints are *unordered*: `L_chord(i, j) == L_chord(j, i)`.

This naming is the unique (up to a global ρ rotation) assignment for
which the Ptolemy/Plücker rule on the 7-gon matches the BPSKAlgebra
product structure with zero mismatches: two letters have a multi-term
(Plücker) product exactly when their chords cross, and every other pair
a single-term product.  Whether a non-crossing pair commutes or
q-commutes is not decided by sharing a vertex — of the 84 ordered
vertex-sharing pairs 42 q-commute and 42 commute, of the 28 disjoint
pairs 14 commute and 14 q-commute (the audit; the
rule restated to its crossing clause by the author's ruling of
2026-09-26, the design record).  See `heptagon_kalg_naming_audit.py` for the
reproducer.

The wrapper inherits multiplication, ρ, trace, inner-product from
`BPSKAlgebra`; it adds:

* `L((k, i))`             — tropical charge of the i-th element of orbit `k`.
* `L_label(γ)`            — inverse `γ → (k, i)`, or `None`.
* `L_chord(i, j)`         — tropical charge of the chord with endpoints
                            `{i, j}` (unordered).
* `L_chord_label(γ)`      — inverse `γ → (min, max)` endpoint pair, or `None`.

This is a *naming* layer only — a Pentagon-style direct rewrite
(`KAlgebra` subclass with an inlined three-letter reducer and a
closed-form two-layer trace) is left for a follow-up.
"""

from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bps_kalgebra import BPSKAlgebra

H = 7  # ρ-order (= the heptagon's cyclic symmetry order)


_HEPTAGON_PAIRING: list[list[int]] = [
    [0, 1, 0, 0],
    [-1, 0, 1, 0],
    [0, -1, 0, 1],
    [0, 0, -1, 0],
]
_HEPTAGON_NODES: list[tuple[int, ...]] = [
    (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
]
# Orbits indexed 1, 2 (matching user's notation).
# k=1: seed (1,0,1,0);   k=2: seed (1,0,0,0).
_HEPTAGON_ORBIT_SEEDS: dict[int, tuple[int, ...]] = {
    1: (1, 0, 1, 0),
    2: (1, 0, 0, 0),
}
_HEPTAGON_RHO_ORDER = H


class HeptagonKAlg(BPSKAlgebra):
    """`K_𝖖([A_1, A_4])` (the "heptagon") as a `BPSKAlgebra` with
    symbolic `L((k, i))` names for the two ρ-orbits of canonical basis
    charges:  `L((1, *))` through `(1, 0, 1, 0)` and `L((2, *))` through
    `(1, 0, 0, 0)`."""

    def __init__(self):
        super().__init__(
            pairing=_HEPTAGON_PAIRING,
            node_charges=_HEPTAGON_NODES,
        )
        self._L_orbits: dict[int, tuple[tuple[int, ...], ...]] = {
            k: self._compute_orbit(seed)
            for k, seed in _HEPTAGON_ORBIT_SEEDS.items()
        }
        self._L_lookup: dict[tuple[int, ...], tuple[int, int]] = {}
        for k, orbit in self._L_orbits.items():
            for i, gamma in enumerate(orbit):
                self._L_lookup.setdefault(gamma, (k, i))

    def _compute_orbit(self, seed) -> tuple[tuple[int, ...], ...]:
        orbit: list[tuple[int, ...]] = [tuple(seed)]
        cur = tuple(seed)
        for _ in range(_HEPTAGON_RHO_ORDER - 1):
            cur = self.rho(cur)
            orbit.append(cur)
        closed = self.rho(cur)
        if closed != tuple(seed):
            raise AssertionError(
                f"Heptagon: ρ-orbit of {seed} did not close at order "
                f"{_HEPTAGON_RHO_ORDER}; ρ^{_HEPTAGON_RHO_ORDER}(seed) = {closed}"
            )
        return tuple(orbit)

    @property
    def rho_order(self) -> int:
        return _HEPTAGON_RHO_ORDER

    def L(self, label: tuple[int, int]) -> tuple[int, ...]:
        """Tropical charge tuple of `L((k, i))`.  `k ∈ {1, 2}`; `i` is
        taken mod `rho_order`."""
        k, i = label
        return self._L_orbits[k][i % _HEPTAGON_RHO_ORDER]

    def L_label(self, gamma) -> tuple[int, int] | None:
        """Inverse lookup: returns `(k, i)` if `γ` is one of the named
        L-generators, else `None`."""
        return self._L_lookup.get(tuple(gamma))

    def L_orbits(self) -> dict[int, tuple[tuple[int, ...], ...]]:
        """The two precomputed ρ-orbits, keyed by orbit index `k`."""
        return self._L_orbits

    def L_chord(self, i: int, j: int) -> tuple[int, ...]:
        """The tropical charge of the L-generator with cyclic endpoints
        `{i, j} ⊂ Z/7`.  Endpoints are unordered: `L_chord(i, j) ==
        L_chord(j, i)`.  The cyclic distance `d = min(j-i, i-j) mod 7`
        ∈ {1, 2, 3} selects the chord class:

          * d = 1 — *edges* of the polygon, normalised to the algebra
                    identity `1`; **not** a named L (returns identity
                    charge `(0, 0, 0, 0)`).
          * d = 2 — short diagonal `(i, i+2)`, orbit 1
                    (seed `(1, 0, 1, 0)`):
                    `L_chord(i, i+2) == L((1, i))`
          * d = 3 — long  diagonal `(i, i+3)`, orbit 2
                    (seed `(1, 0, 0, 0)`):
                    `L_chord(i, i+3) == L((2, (i + 3) mod 7))`
                    (orbit-2 index is **shifted by 3** relative to the
                    chord's canonical starting vertex).

        Under this assignment the BPSKAlgebra product structure matches
        the Ptolemy/Plücker rule on the 7-gon: crossing chords give a
        multi-term (Plücker) product and every other pair a single term
        (whether that pair commutes or q-commutes is not decided by
        sharing a vertex; the audit)."""
        i_m, j_m = i % H, j % H
        if i_m == j_m:
            raise ValueError(f"L_chord: endpoints must be distinct, got {i}, {j}")
        forward = (j_m - i_m) % H
        backward = (i_m - j_m) % H
        if forward <= backward:
            d, start = forward, i_m
        else:
            d, start = backward, j_m
        if d == 1:
            # Edge -- normalised to the algebra identity.
            return (0,) * 4
        if d == 2:
            return self.L((1, start))
        if d == 3:
            return self.L((2, (start + 3) % H))
        raise ValueError(
            f"L_chord: unexpected cyclic distance {d} for endpoints {i}, {j}"
        )

    def L_chord_label(self, gamma) -> tuple[int, int] | None:
        """Inverse: returns `(i, j)` with `i < j` (the canonical
        unordered-pair representative) if `γ` is one of the named L's,
        else `None`.  The identity charge is not a named chord (it
        represents any of the 7 edges; no unique label)."""
        name = self.L_label(gamma)
        if name is None:
            return None
        k, idx = name
        if k == 1:
            # orbit 1 ↔ short diagonal (idx, idx + 2)
            a, b = idx % H, (idx + 2) % H
        else:
            # orbit 2 ↔ long diagonal (idx - 3, idx) ≡ ((idx + 4) % 7, idx)
            a, b = (idx + 4) % H, idx % H
        return (min(a, b), max(a, b))

