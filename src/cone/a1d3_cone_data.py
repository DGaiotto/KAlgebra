"""
a1d3_cone_data.py
=================

`A1D3ConeData(FiniteConeData)` — cone-data wiring for the [A_1, D_3]
K-algebra over `SU2ZPlusRing` (= R(SU(2))).  Smoke test for the
R-widening of `ConeData` (PR landing 225d5ac): exercises the path
where `cross_product` returns `RLaurent[R]` daughters with non-trivial
R-content (`χ_1` from the T·T and T·D Plücker relations).

Mult-gens:
  ('T', 0), ('T', 1), ('T', 2), ('D', 0), ('D', 1), ('D', 2)   (6 total)

Cones (6, Bron-Kerbosch on the q-commute graph):
  tile 0:  {T_0, D_0}  (twist -1)
  tile 1:  {T_0, D_2}  (twist +1)
  tile 2:  {T_1, D_1}  (twist -1)
  tile 3:  {T_1, D_0}  (twist +1)
  tile 4:  {T_2, D_2}  (twist -1)
  tile 5:  {T_2, D_1}  (twist +1)

Native label: `(tile, a, b)` (3-tuple) — `a`, `b` are the exponents of
the tile's T- and D-letters respectively.  χ-content lives entirely in
the R-coefficient of an `Element` over this cone-data (NOT in the
label), per the cone-data convention.

Plücker relations (from a1d3_kalg._interaction):
  T_i · T_{i+1}  =  1  +  q⁻¹·χ_1·D_i  +  q⁻²·D_i²
  T_{i+1} · T_i  =  1  +  q  ·χ_1·D_i  +  q² ·D_i²
  D_i · D_{i+1}  =  1  +  q⁻¹·T_{i+1}
  D_{i+1} · D_i  =  1  +  q  ·T_{i+1}
  T_i · D_{i+1}  =  χ_1  +  q⁻¹·D_i  +  q·D_{i-1}
  D_{i+1} · T_i  =  χ_1  +  q  ·D_i  +  q⁻¹·D_{i-1}
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cone_data import FiniteConeData, Cone
from zplus_ring import SU2ZPlusRing, RLaurent
from a1d3_kalg import (
    _q_commute_twist, _interaction,
    _TILE_LETTERS, _TILE_TWIST, _tile_for_pair,
    _tile_for_T_letter, _tile_for_D_letter,
)


class A1D3ConeData(FiniteConeData):
    """Cone-data for [A_1, D_3] K-algebra over R = R(SU(2))."""

    _MULT_GENS = tuple((k, i) for k in ('T', 'D') for i in range(3))

    def __init__(self):
        self._R = SU2ZPlusRing()
        self._cones = None

    # ---- R-widening hook ------------------------------------------------

    def coefficient_ring(self):
        return self._R

    # ---- ConeData primitives --------------------------------------------

    def mult_gens(self):
        return self._MULT_GENS

    def cones(self):
        if self._cones is None:
            V = list(self._MULT_GENS)
            neighbours = {
                v: frozenset(u for u in V if u != v and self.q_commute(v, u))
                for v in V
            }
            cliques = []

            def bk(R, P, X):
                if not P and not X:
                    cliques.append(R); return
                pivot = max(P | X, key=lambda u: len(P & neighbours[u]))
                for v in list(P - neighbours[pivot]):
                    bk(R | {v}, P & neighbours[v], X & neighbours[v])
                    P = P - {v}; X = X | {v}

            bk(frozenset(), frozenset(V), frozenset())
            self._cones = tuple(cliques)
        return self._cones

    def q_commute(self, g, h):
        if g == h: return True
        return _q_commute_twist(g, h) is not None

    def cocycle(self, g, h):
        if g == h: return 0
        t = _q_commute_twist(g, h)
        if t is None:
            raise ValueError(f"cocycle: {g}, {h} not q-commuting")
        return t

    def cross_product(self, g, h):
        """Return literal-product expansion as list of (RLaurent, word)."""
        terms = _interaction(g, h)
        if terms is None:
            raise ValueError(f"cross_product: {g}, {h} are q-commuting")
        R = self._R
        out = []
        for q_delta, letters, chi_k in terms:
            # literal word from letters dict (multiplicity-expanded)
            word: list = []
            for L in sorted(letters.keys()):
                word.extend([L] * letters[L])
            r = R.basis_element(chi_k) if chi_k > 0 else R.one()
            coef = RLaurent(R, {q_delta: r})
            out.append((coef, tuple(word)))
        return out

    # ---- label bijection ------------------------------------------------

    def to_cone_label(self, native_label):
        tile, a, b = native_label
        L_T, L_D = _TILE_LETTERS[tile]
        if a == 0 and b == 0:
            return (frozenset(), {})
        if a > 0 and b == 0:
            return (frozenset({L_T}), {L_T: a})
        if a == 0 and b > 0:
            return (frozenset({L_D}), {L_D: b})
        return (frozenset({L_T, L_D}), {L_T: a, L_D: b})

    def from_cone_label(self, gens, powers):
        if not gens:
            return (0, 0, 0)
        t_letters = [g for g in gens if g[0] == 'T']
        d_letters = [g for g in gens if g[0] == 'D']
        if len(t_letters) > 1 or len(d_letters) > 1:
            raise ValueError(f"non-tile gens {gens}")
        if t_letters and not d_letters:
            L_T = t_letters[0]
            return (_tile_for_T_letter(L_T), powers[L_T], 0)
        if d_letters and not t_letters:
            L_D = d_letters[0]
            return (_tile_for_D_letter(L_D), 0, powers[L_D])
        L_T, L_D = t_letters[0], d_letters[0]
        tile = _tile_for_pair(L_T, L_D)
        if tile is None:
            raise ValueError(f"({L_T}, {L_D}) not a q-commuting tile pair")
        return (tile, powers[L_T], powers[L_D])

    # ---- monomial-cone iteration ----------------------------------------

    def iter_cones(self):
        for cone_gens in self.cones():
            yield Cone(self, cone_gens)


if __name__ == "__main__":
    D = A1D3ConeData()
    print(f"A1D3ConeData: {len(D.mult_gens())} mult-gens")
    print(f"  coefficient_ring = {D.coefficient_ring()}")
    cones = D.cones()
    print(f"  {len(cones)} cones:")
    for c in sorted(cones, key=lambda x: tuple(sorted(x))):
        print(f"    {sorted(c)}")
    print()
    print("Sample cross_product T_0 · T_1:")
    for coef, word in D.cross_product(('T', 0), ('T', 1)):
        print(f"  {coef} · {word}")
    print()
    print("Sample cross_product T_0 · D_1:")
    for coef, word in D.cross_product(('T', 0), ('D', 1)):
        print(f"  {coef} · {word}")
