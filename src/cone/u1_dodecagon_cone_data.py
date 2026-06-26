"""
u1_dodecagon_cone_data.py
=========================

`U1DodecagonConeData(FiniteConeData)` — standalone cone-data wiring for
`U1DodecagonKAlg` (k=4, u(1)-gauged [A_1, A_9]).

Primitives: L_1..L_4 (12 each) + L_5 (6) + E^± = 54 chord + E^± = 56 mult-gens.
Cone data is driven entirely by the frozen `u1_dodecagon_mult_table.MULT_TABLE`
and `u1a1aodd_k4_chord_charges.MU_LETTER_QPOWER` — NO BPS / canonical import at
runtime (mirrors `u1_octagon_cone_data` / `u1_decagon_cone_data`).
"""
from __future__ import annotations

from cone_data import FiniteConeData, CrossProductTerm, Cone
from laurent_poly import LaurentPoly
from u1_dodecagon_mult_table import MULT_TABLE
from u1a1aodd_k4_chord_charges import MU_LETTER_QPOWER


# E_GEN / E_INV — encoded as (6, 0) / (6, 1) so default sorted() works and
# doesn't collide with L_1..L_5 (family indices 1..5).
E_GEN = (6, 0)
E_INV = (6, 1)


def _all_chord_gens():
    return ([(a, i) for a in (1, 2, 3, 4) for i in range(12)]
            + [(5, i) for i in range(6)])


def _all_mult_gens():
    return _all_chord_gens() + [E_GEN, E_INV]


def _is_L_letter(g):
    return g[0] in (1, 2, 3, 4, 5)


def _native_label_to_word(native_label):
    factors, e_E = native_label
    word = []
    for (a, i, exp) in sorted(factors):
        word.extend([(a, i)] * exp)
    if e_E > 0:
        word.extend([E_GEN] * e_E)
    elif e_E < 0:
        word.extend([E_INV] * (-e_E))
    return tuple(word)


class U1DodecagonConeData(FiniteConeData):
    """Cone-data for the standalone U1DodecagonKAlg (k=4)."""

    def __init__(self):
        self._mult_gens = tuple(_all_mult_gens())
        # (a, i) chord generators — read by the orthonormality-bootstrap driver.
        self._chords = _all_chord_gens()
        self._cones = None

    def mult_gens(self):
        return self._mult_gens

    def cones(self):
        if self._cones is None:
            V = list(self._mult_gens)
            neighbours = {
                v: frozenset(u for u in V if u != v and self.q_commute(v, u))
                for v in V
            }
            cliques = []

            def bk(R, P, X):
                if not P and not X:
                    cliques.append(R)
                    return
                pivot = max(P | X, key=lambda u: len(P & neighbours[u]))
                for v in list(P - neighbours[pivot]):
                    bk(R | {v}, P & neighbours[v], X & neighbours[v])
                    P = P - {v}
                    X = X | {v}

            bk(frozenset(), frozenset(V), frozenset())
            self._cones = tuple(cliques)
        return self._cones

    def q_commute(self, g, h):
        if g == h:
            return True
        if {g, h} == {E_GEN, E_INV}:
            return True
        if _is_L_letter(g) and _is_L_letter(h):
            fwd = MULT_TABLE.get((g, h))
            return fwd is not None and len(fwd) == 1
        return True  # L-letter ↔ E always q-commute

    def cocycle(self, g, h):
        if g == h:
            return 0
        if {g, h} == {E_GEN, E_INV}:
            return 0
        if _is_L_letter(g) and _is_L_letter(h):
            fwd = MULT_TABLE.get((g, h))
            bwd = MULT_TABLE.get((h, g))
            if fwd is None or len(fwd) != 1 or bwd is None or len(bwd) != 1:
                raise ValueError(f"cocycle: L{g}, L{h} not q-commuting")
            c_full = fwd[0][0] - bwd[0][0]
            assert c_full % 2 == 0
            return c_full // 2
        if g == E_GEN and _is_L_letter(h):
            return -MU_LETTER_QPOWER[h]
        if h == E_GEN and _is_L_letter(g):
            return MU_LETTER_QPOWER[g]
        if g == E_INV and _is_L_letter(h):
            return MU_LETTER_QPOWER[h]
        if h == E_INV and _is_L_letter(g):
            return -MU_LETTER_QPOWER[g]
        raise ValueError(f"cocycle undefined for ({g}, {h})")

    def cross_product(self, g, h):
        if {g, h} == {E_GEN, E_INV}:
            return [(LaurentPoly.one(), ())]
        if _is_L_letter(g) and _is_L_letter(h):
            terms = MULT_TABLE.get((g, h))
            if terms is None:
                raise ValueError(f"MULT_TABLE missing ({g}, {h})")
            out = []
            for (q_pow, e_E, factors, q_coef) in terms:
                word = list(factors)
                if e_E > 0:
                    word.extend([E_GEN] * e_E)
                elif e_E < 0:
                    word.extend([E_INV] * (-e_E))
                gens_set = frozenset(word)
                powers = {gx: word.count(gx) for gx in gens_set}
                phase = self.cone_label_phase(gens_set, powers)
                coef = q_coef * LaurentPoly({q_pow + phase: 1})
                out.append((coef, tuple(word)))
            return out
        raise ValueError(f"cross_product undefined for q-commuting ({g}, {h})")

    def to_cone_label(self, native_label):
        """Return the label's own letter-set as its cone + per-letter powers.

        Mirrors the canonical `U1A1AoddConeData.to_cone_label`: the letters of
        a cone monomial pairwise q-commute, so they ARE a (sub)cone; returning
        them directly is O(letters) and avoids scanning the 16796 maximal
        cones (extra zero-power gens of a maximal cone do not affect
        `cone_label_phase`).  This is the hot path for trace/multiply."""
        factors, e_E = native_label
        gens = set()
        powers = {}
        for (a, i, ex) in factors:
            if ex <= 0:
                continue
            g = (a, i)
            gens.add(g)
            powers[g] = powers.get(g, 0) + ex
        if e_E > 0:
            gens.add(E_GEN)
            powers[E_GEN] = e_E
        elif e_E < 0:
            gens.add(E_INV)
            powers[E_INV] = -e_E
        return frozenset(gens), powers

    def from_cone_label(self, gens, powers):
        factors_dict = {}
        e_E = 0
        for g, p in powers.items():
            if p == 0:
                continue
            if g == E_GEN:
                e_E += p
            elif g == E_INV:
                e_E -= p
            else:
                a, i = g
                factors_dict[(a, i)] = factors_dict.get((a, i), 0) + p
        factors = tuple(sorted((a, i, p) for (a, i), p in factors_dict.items()))
        return (factors, e_E)

    def _torus_inverse_letter(self, g):
        if g == E_GEN:
            return E_INV
        if g == E_INV:
            return E_GEN
        return None

    def iter_cones(self):
        for cone_gens in self.cones():
            torus_in_cone = frozenset({E_GEN, E_INV}) & cone_gens
            yield Cone(self, cone_gens, torus_gens=torus_in_cone)


if __name__ == "__main__":
    D = U1DodecagonConeData()
    print(f"U1DodecagonConeData: {len(D.mult_gens())} mult-gens")
    cones = D.cones()
    print(f"  {len(cones)} maximal cones (Bron-Kerbosch)")
    from collections import Counter
    print(f"  Cone sizes: {dict(sorted(Counter(len(c) for c in cones).items()))}")
