"""
a1d5_cone_data.py
=================

`A1D5ConeData(FiniteConeData)` — cone-data wiring for the [A_1, D_5]
K-algebra over `SU2ZPlusRing` (R(SU(2))).

  * 20 atomic mult-gens: 4 ρ-orbits {T, D, V, W} × 5 indices.
  * 70 maximal cones (size 4 = gauge_rank), enumerated by
    Bron-Kerbosch on the q-commute graph.
  * `cross_product` daughters built from `_PLUCKER_BASE_I0` with
    cluster-monomial decomposition (`a1d5_decomposer.decompose`) for
    the 52 distinct non-atomic Plücker output g-vectors.

Native label convention: 5-tuple g-vector with last coord = 0 (the
χ-content is stripped at the A1D5KAlg.multiply boundary and threaded
as an R-coefficient).  Cone-data operates purely on the monomial part.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cone_data import FiniteConeData, Cone
from zplus_ring import SU2ZPlusRing, RLaurent
from a1d5_kalg import (
    _PLUCKER_BASE_I0, _LATTICE_TO_MULTGEN,
    _T_ORBIT, _D_ORBIT, _V_ORBIT, _W_ORBIT,
    _rho_apply, _rho_n_apply,
)
from a1d5_decomposer import (
    ATOMIC_GENS, gvec, _q_commute, decompose,
)


class A1D5ConeData(FiniteConeData):
    """Cone-data for [A_1, D_5] K-algebra over R = R(SU(2))."""

    def __init__(self):
        self._R = SU2ZPlusRing()
        self._mult_gens = tuple(ATOMIC_GENS)
        self._cones = None
        # Pre-compute cluster-monomial decomposition cache for fast lookup.
        self._decompose_cache: dict[tuple, tuple] = {}

    def coefficient_ring(self):
        return self._R

    def mult_gens(self):
        return self._mult_gens

    def cones(self):
        if self._cones is None:
            V = list(self._mult_gens)
            neighbours = {
                v: frozenset(u for u in V if u != v and _q_commute(v, u))
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
        return _q_commute(g, h)

    def cocycle(self, g, h):
        """Integer c with L_g L_h = q^{2c} L_h L_g.

        From _PLUCKER_BASE_I0: for q-commuting (g, h), entry has
        single term `(1, q_exp, label)` meaning L_g L_h = q^{q_exp}
        L_{g+h}.  Reversed: L_h L_g = q^{-q_exp} L_{g+h}.  So
        L_g L_h = q^{2*q_exp} L_h L_g, giving c = q_exp.
        """
        if g == h:
            return 0
        (k_g, i_g), (k_h, i_h) = g, h
        j = (i_h - i_g) % 5
        entry = _PLUCKER_BASE_I0.get((k_g, k_h, j))
        if entry is None or len(entry) != 1:
            raise ValueError(f"cocycle: ({g}, {h}) not q-commuting")
        _coef, q_exp, _lab = entry[0]
        return q_exp

    def _decompose_mono(self, mono_5tuple):
        """Cached cluster decomposition of a 5-tuple g-vector
        (last coord 0) into a sorted tuple of atomic mult-gens."""
        if mono_5tuple in self._decompose_cache:
            return self._decompose_cache[mono_5tuple]
        d = decompose(mono_5tuple)
        if d is None:
            raise ValueError(f"_decompose_mono: cannot decompose {mono_5tuple}")
        self._decompose_cache[mono_5tuple] = d
        return d

    def cross_product(self, g, h):
        """Literal-product expansion as list of (RLaurent, word).

        For each Plücker term `(coef, q_exp, lab)` in the
        `_PLUCKER_BASE_I0[(k_g, k_h, j)]` entry (after applying ρ^{i_g}
        to `lab`), build:
          * χ_k = -lab_canon[-1] ≥ 0 (after canonicalise flip)
          * mono = lab_canon[:-1] + (0,)
          * word = `_decompose_mono(mono)`
          * RLaurent coef = `q^{q_exp} · coef · χ_k`
        """
        (k_g, i_g), (k_h, i_h) = g, h
        j = (i_h - i_g) % 5
        entry = _PLUCKER_BASE_I0.get((k_g, k_h, j))
        if entry is None:
            raise ValueError(f"cross_product: no entry for ({g}, {h})")
        if len(entry) == 1:
            raise ValueError(f"cross_product: ({g}, {h}) q-commute (single term)")
        R = self._R
        out = []
        for coef, q_exp, lab in entry:
            # Apply ρ^{i_g} to the i=0-row output label.
            lab_shifted = _rho_n_apply(lab, i_g)
            # Canonicalise: flip last coord if > 0 (SU(2) Weyl).
            if lab_shifted[-1] > 0:
                lab_canon = lab_shifted[:-1] + (-lab_shifted[-1],)
            else:
                lab_canon = lab_shifted
            chi_k = -lab_canon[-1]  # >= 0
            mono = lab_canon[:-1] + (0,)
            word = self._decompose_mono(mono)
            # Cone_label_phase of the daughter monomial — converts the
            # stored canonical-basis q-power to literal-product q-power
            # (which is what cross_product returns by contract).
            if word:
                from collections import Counter
                ctr = Counter(word)
                gens_set = frozenset(ctr.keys())
                powers = dict(ctr)
                phase = self.cone_label_phase(gens_set, powers)
            else:
                phase = 0
            # R-coefficient: coef · χ_k
            r_elt = R.basis_element(chi_k) * coef if chi_k > 0 \
                else (R.one() * coef if coef != 1 else R.one())
            r_coef = RLaurent(R, {q_exp + phase: r_elt})
            out.append((r_coef, word))
        return out

    def to_cone_label(self, native_label):
        """Native 5-tuple (last coord = 0, χ-stripped) → (gens, powers).

        Decomposes the lattice charge into atomic mult-gens via the
        cluster-monomial decomposer, then collects exponents.
        """
        if native_label[-1] != 0:
            raise ValueError(
                f"A1D5ConeData.to_cone_label: native label must have χ=0 "
                f"(got {native_label}); χ-content is handled at the "
                f"A1D5KAlg.multiply boundary, not in cone-data labels."
            )
        word = self._decompose_mono(native_label)
        if not word:
            return (frozenset(), {})
        from collections import Counter
        c = Counter(word)
        gens = frozenset(c.keys())
        powers = dict(c)
        return (gens, powers)

    def from_cone_label(self, gens, powers):
        """(gens, powers) → 5-tuple by summing g-vectors."""
        if not gens:
            return (0, 0, 0, 0, 0)
        total = [0] * 5
        for g, p in powers.items():
            if p == 0: continue
            gv = gvec(g)
            for k in range(5):
                total[k] += p * gv[k]
        return tuple(total)

    def iter_cones(self):
        for cone_gens in self.cones():
            yield Cone(self, cone_gens)


if __name__ == "__main__":
    D = A1D5ConeData()
    print(f"A1D5ConeData: {len(D.mult_gens())} mult-gens, "
          f"R = {D.coefficient_ring()}")
    cones = D.cones()
    print(f"  {len(cones)} cones (size {len(cones[0])})")
    print()
    print("Sample cross_product T_0 · T_1:")
    for coef, word in D.cross_product(('T', 0), ('T', 1)):
        print(f"  {coef} · {word}")
    print()
    print("Sample cross_product T_0 · V_0:")
    for coef, word in D.cross_product(('T', 0), ('V', 0)):
        print(f"  {coef} · {word}")
    print()
    print("Sample to_cone_label / from_cone_label round-trip:")
    test_lab = (1, 1, 1, 1, 0)
    cl = D.to_cone_label(test_lab)
    print(f"  {test_lab} -> {cl}")
    print(f"  back -> {D.from_cone_label(*cl)}")
