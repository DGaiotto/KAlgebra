"""
su3_ad_cone_data.py
===================

`SU3ADConeData(FiniteConeData)` — cone-data wiring for SU3ADKAlg
([A_1, D_4] AD theory with SU(3) symmetry enhancement, Z_4-symmetric).

Parallel to `a1d3_cone_data.py`, with 8 mult-gens (T_0..T_3, D_0..D_3)
and `SU3ZPlusRing` coefficients (vs A1D3's 6 mult-gens with
`SU2ZPlusRing`).
"""
from __future__ import annotations
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cone_data import FiniteConeData, Cone
from zplus_ring import SU3ZPlusRing, RLaurent
from su3_ad_kalg import (
    _q_commute_twist, _interaction,
    _TILE_LETTERS, _tile_for_pair,
    _tile_for_T_letter, _tile_for_D_letter,
)


class SU3ADConeData(FiniteConeData):
    """Cone-data for SU3ADKAlg over R = R(SU(3))."""

    _MULT_GENS = tuple((k, i) for k in ('T', 'D') for i in range(4))

    def __init__(self):
        self._R = SU3ZPlusRing()
        self._cones = None

    def coefficient_ring(self):
        return self._R

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
        for q_delta, letters, pq, coef in terms:
            word: list = []
            for L in sorted(letters.keys()):
                word.extend([L] * letters[L])
            # SU(3) basis element χ_{(p, q)} (= R.one() at (0, 0)).
            r_elt = R.basis_element(pq) * coef if pq != (0, 0) \
                else (R.one() * coef if coef != 1 else R.one())
            r_coef = RLaurent(R, {q_delta: r_elt})
            out.append((r_coef, tuple(word)))
        return out

    def to_cone_label(self, native_label):
        """Native 5-tuple (tile, a, b, 0, 0) → (gens, powers).  Caller
        must strip (p, q) before calling — this is the cone-data layer."""
        tile, a, b, p, q = native_label
        if p != 0 or q != 0:
            raise ValueError(
                f"SU3ADConeData.to_cone_label: native label must have "
                f"χ=(0,0) (got {native_label}); χ-content is folded at "
                f"the SU3ADKAlg.multiply boundary."
            )
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
            return (0, 0, 0, 0, 0)
        t_letters = [g for g in gens if g[0] == 'T']
        d_letters = [g for g in gens if g[0] == 'D']
        if len(t_letters) > 1 or len(d_letters) > 1:
            raise ValueError(f"non-tile gens {gens}")
        if t_letters and not d_letters:
            L_T = t_letters[0]
            return (_tile_for_T_letter(L_T), powers[L_T], 0, 0, 0)
        if d_letters and not t_letters:
            L_D = d_letters[0]
            return (_tile_for_D_letter(L_D), 0, powers[L_D], 0, 0)
        L_T, L_D = t_letters[0], d_letters[0]
        tile = _tile_for_pair(L_T, L_D)
        if tile is None:
            raise ValueError(f"({L_T}, {L_D}) not a q-commuting tile pair")
        return (tile, powers[L_T], powers[L_D], 0, 0)

    def iter_cones(self):
        for cone_gens in self.cones():
            yield Cone(self, cone_gens)


if __name__ == "__main__":
    D = SU3ADConeData()
    print(f"SU3ADConeData: {len(D.mult_gens())} mult-gens, "
          f"R = {D.coefficient_ring()}")
    cones = D.cones()
    print(f"  {len(cones)} cones (size {len(cones[0]) if cones else 0})")
    for c in sorted(cones, key=lambda x: tuple(sorted(x))):
        print(f"    {sorted(c)}")
    print()
    print("Sample cross_product T_0 · T_1 (= 1 + q^{-1}·χ_(0,1)·D_0 + q^{-2}·χ_(1,0)·D_0² + q^{-3}·D_0³):")
    for coef, word in D.cross_product(('T', 0), ('T', 1)):
        print(f"  {coef} · {word}")
