"""
u1_octagon_cone_data.py
=======================

`U1OctaConeData(FiniteConeData)` — cone-data wiring for U1OctaKAlg.

Once this class is built, `derived_multiply` and
`simplify_trace_via_cone_data` give compound-label multiply + trace
for free (analogous to A1A2kKAlg, U1HexagonKAlg).

Primitive mult-gens: 16 chord generators (L_1 × 8 + L_2 × 8) + E + E⁻¹
= **18 multiplicative generators**.

Cones: Maximal q-commuting subsets, enumerated by Bron-Kerbosch on the
q-commute graph (derived from `MULT_TABLE`).
"""
from __future__ import annotations

from typing import Sequence
from cone_data import FiniteConeData, CrossProductTerm, Cone
from laurent_poly import LaurentPoly
from u1_octagon_mult_table import MULT_TABLE


# Mult-gen labels: (a, i) for chord generators; ('E', +1) / ('E', -1) for E±
# E_GEN / E_INV — encoded as integer tuples (a, i) so default sorted() works.
# Use family index 4 (beyond L_1, L_2, L_3 which now use 1, 2, 3).
E_GEN = (4, 0)
E_INV = (4, 1)


def _all_chord_gens():
    return [(a, i) for a in (1, 2, 3) for i in range(8)]


def _all_mult_gens():
    return _all_chord_gens() + [E_GEN, E_INV]


def _is_L_letter(g):
    return g[0] in (1, 2, 3)  # primitive chord families; (4, 0/1) is E±


# MU_LETTER_QPOWER: cocycle c(L_l, E) such that L_l · E = q^{2c} · E · L_l
# Pattern (extracted from BPS once): L_1 alternates ±1 by parity of i; L_2 all 0.
MU_LETTER_QPOWER = {}
for i in range(8):
    MU_LETTER_QPOWER[(1, i)] = -1 if i % 2 == 0 else +1
    MU_LETTER_QPOWER[(2, i)] = 0
    # L_3 magnetic cocycle with E — extract from BPS, store explicitly later
    # For now, derive same as L_1 (also magnetic-charged)
    MU_LETTER_QPOWER[(3, i)] = -1 if i % 2 == 0 else +1  # TODO verify


def _native_label_to_word(native_label):
    """Native label = (factors, e_E) where factors = tuple of (a, i, exp).
    Return list of mult-gen letters in canonical order."""
    factors, e_E = native_label
    word = []
    for (a, i, exp) in sorted(factors):
        word.extend([(a, i)] * exp)
    if e_E > 0:
        word.extend([E_GEN] * e_E)
    elif e_E < 0:
        word.extend([E_INV] * (-e_E))
    return tuple(word)


class U1OctaConeData(FiniteConeData):
    """Cone-data for the U1OctaKAlg (k=2 closed-form K-algebra)."""

    def __init__(self):
        self._mult_gens = tuple(_all_mult_gens())
        # (a, i) chord generators — read by the orthonormality-bootstrap
        # driver (`u1aodd_trace_bootstrap.solve_intermediate`).
        self._chords = _all_chord_gens()
        self._cones = None  # lazy

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
        # E and E⁻¹ now q-commute with cocycle 0 (QTCone / Laurent-cone
        # convention): they live in the SAME cone as a torus direction.
        # The cancellation E · E⁻¹ = 1 is handled by the universal
        # `_torus_inverse_letter` collapse in
        # `ConeData._word_to_gens_powers`, NOT via cross_product.
        if {g, h} == {E_GEN, E_INV}:
            return True
        if _is_L_letter(g) and _is_L_letter(h):
            fwd = MULT_TABLE.get((g, h))
            return fwd is not None and len(fwd) == 1
        # L-letter ↔ E: always q-commute via MU_LETTER_QPOWER
        return True

    def cocycle(self, g, h):
        if g == h:
            return 0
        # E and E⁻¹ commute (cocycle 0) — torus direction.
        if {g, h} == {E_GEN, E_INV}:
            return 0
        if _is_L_letter(g) and _is_L_letter(h):
            fwd = MULT_TABLE.get((g, h))
            bwd = MULT_TABLE.get((h, g))
            if fwd is None or len(fwd) != 1 or bwd is None or len(bwd) != 1:
                raise ValueError(f"cocycle: L{g}, L{h} not q-commuting")
            c_full = fwd[0][0] - bwd[0][0]
            assert c_full % 2 == 0, f"cocycle non-integer at ({g}, {h})"
            return c_full // 2
        # L-letter ↔ E:
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
        """`L_g · L_h` as a list of (LaurentPoly, word) pairs in mult-gens
        (literal-product convention).

        MULT_TABLE stores entries as canonical-basis coefficients:
            entry = (q_pow, e_E, factors, q_coef)
            term = q_coef · q^{q_pow} · L_{native(factors, e_E)}
        Convert to literal-product coef by adding `cone_label_phase` to q_pow:
            literal_coef = q_coef · q^{q_pow + phase}.

        Note: (E_GEN, E_INV) no longer reaches here — they q-commute
        with cocycle 0, and their cancellation E · E⁻¹ = 1 is handled
        by `ConeData._word_to_gens_powers` via `_torus_inverse_letter`.
        """
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
                # Compute cone_label_phase for the daughter cone monomial
                # (gens = unique letters in word, powers = counts).
                gens_set = frozenset(word)
                powers = {gx: word.count(gx) for gx in gens_set}
                phase = self.cone_label_phase(gens_set, powers)
                coef = q_coef * LaurentPoly({q_pow + phase: 1})
                out.append((coef, tuple(word)))
            return out
        raise ValueError(f"cross_product undefined for q-commuting ({g}, {h})")

    def to_cone_label(self, native_label):
        """Native (factors, e_E) → (cone_gens_frozenset, powers_dict).
        Picks a cone containing the mult-gens in this label's word."""
        word = _native_label_to_word(native_label)
        if not word:
            # Identity element — pick any cone containing E (or empty cone)
            cones = self.cones()
            for c in cones:
                if E_GEN in c:
                    return (c, {g: 0 for g in c})
            # No cone contains E (shouldn't happen) — return any cone
            return (cones[0], {g: 0 for g in cones[0]})
        gens_in_word = set(word)
        # Find a cone containing all letters in the word
        for cone in self.cones():
            if gens_in_word <= set(cone):
                powers = {g: word.count(g) for g in cone}
                return (cone, powers)
        raise ValueError(f"no cone contains all letters of {native_label}: {gens_in_word}")

    def from_cone_label(self, gens, powers):
        """(cone_gens, powers) → native label (factors, e_E)."""
        factors_dict = {}
        e_E = 0
        for g, p in powers.items():
            if p == 0: continue
            if g == E_GEN:
                e_E += p
            elif g == E_INV:
                e_E -= p
            else:
                a, i = g
                key = (a, i)
                factors_dict[key] = factors_dict.get(key, 0) + p
        factors = tuple(sorted((a, i, p) for (a, i), p in factors_dict.items()))
        return (factors, e_E)

    # -- torus inverse letter (Laurent E direction; QTCone semantics) -----

    def _torus_inverse_letter(self, g):
        """E ↔ E⁻¹ pairing.  Triggers the universal cancellation
        collapse in `ConeData._word_to_gens_powers` (replacing the
        previous cross_product Plücker pair handling)."""
        if g == E_GEN:
            return E_INV
        if g == E_INV:
            return E_GEN
        return None

    # -- QTCone iter ------------------------------------------------------

    def iter_cones(self):
        """Yield QTCone instances (one per maximal q-commuting clique).
        Each cone has `torus_gens = {E, E⁻¹} ∩ cone_gens` — Laurent
        torus direction where present."""
        for cone_gens in self.cones():
            torus_in_cone = frozenset({E_GEN, E_INV}) & cone_gens
            yield Cone(self, cone_gens, torus_gens=torus_in_cone)


if __name__ == "__main__":
    D = U1OctaConeData()
    print(f"U1OctaConeData: {len(D.mult_gens())} mult-gens")
    print(f"  L_1: 8, L_2: 8, E±: 2")
    print(f"  Enumerating cones (Bron-Kerbosch on q-commute graph)...")
    cones = D.cones()
    print(f"  {len(cones)} maximal cones")
    from collections import Counter
    sizes = Counter(len(c) for c in cones)
    print(f"  Cone sizes: {dict(sorted(sizes.items()))}")

    # Test cross_product on a known Plücker
    print()
    print("Sample cross_product L_1(0) · L_1(1):")
    for coef, word in D.cross_product((1, 0), (1, 1)):
        print(f"  {coef} · {word}")
