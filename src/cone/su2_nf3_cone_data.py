"""`SU2Nf3ConeData` — H-tower cone data for `A_𝖖[SU(2)+N_f=3]`, the SU(4)
lift of `su2_nf2_cone_data` (rank-2 H-pair cones, Wilson cone).

Gauge skeleton identical to pure SU(2) / N_f=1,2 (rank-2 pair cones
`C_a = {H_a, H_{a+1}}`, q-commute iff |b-a| ≤ 1); the flavour widening is the
**SU(4)** character ring.  ρ shifts the H index by `4 − N_f = −1` and acts on
the flavour character by complex conjugation (star, 4↔4̄).

Native label: `(h_factors, (m1, m2, m3))`.  `multiply` is supplied by the
standalone via `su2_nf3_h_multiply.multiply_native`; this data object furnishes
the cone skeleton + the gauge cross-product table for single generators.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from typing import Sequence

from cone_data import ConeData, CrossProductTerm
from su2_nf3_h_gap_k import R_SU4
from pure_su2_h_wilson import is_wilson_label, WILSON_FUND


def cone_index_for(n_lo: int, n_hi: int):
    """Rank-2 pair cone C_a = {H_a, H_{a+1}}: contains both iff n_hi-n_lo<=1."""
    return n_lo if (n_hi - n_lo) <= 1 else None


class SU2Nf3ConeData(ConeData):
    """SU(2)+N_f=3 cone data: pure-SU(2) H-tower skeleton, SU(4) flavour
    widening.  Mult-gens `H_n=(n,)`, `w_1`.  Cones `C_a={H_a,H_{a+1}}`."""

    def __init__(self) -> None:
        self._R = R_SU4
        self._gens: dict = {}
        self._cones: dict = {}
        self._wilson_cone = frozenset({WILSON_FUND})

    def coefficient_ring(self):
        return self._R

    def _gen(self, n: int) -> tuple:
        if n not in self._gens:
            self._gens[n] = (n,)
        return self._gens[n]

    def _cone(self, a: int) -> frozenset:
        if a not in self._cones:
            self._cones[a] = frozenset({self._gen(a), self._gen(a + 1)})
        return self._cones[a]

    def iter_cones(self):
        yield self._wilson_cone
        yield self._cone(0)
        k = 1
        while True:
            yield self._cone(k)
            yield self._cone(-k)
            k += 1

    # -- gauge skeleton (flavour-blind) --------------------------------

    def q_commute(self, g, h) -> bool:
        if g == h:
            return True
        gw, hw = is_wilson_label(g), is_wilson_label(h)
        if gw and hw:
            return True
        if gw or hw:
            return False
        a, b = g[0], h[0]
        lo, hi = (a, b) if a < b else (b, a)
        return cone_index_for(lo, hi) is not None

    def cocycle(self, g, h) -> int:
        if g == h:
            return 0
        if is_wilson_label(g) and is_wilson_label(h):
            return 0
        if not self.q_commute(g, h):
            raise ValueError(f"cocycle: ({g}, {h}) not q-commuting")
        return h[0] - g[0]

    def cross_product(self, g, h) -> Sequence[CrossProductTerm]:
        """Gauge ray product of two single mult-generators, via the clean-seed
        generator OPE."""
        from su2_nf3_h_multiply import _gauge_mul, _native_gauge

        def _seed(x):
            if is_wilson_label(x):
                return (0, 1)
            return (1, x[0])

        def _to_gens(seed):
            hf = _native_gauge(seed)
            if not hf:
                return ()
            first = hf[0]
            if isinstance(first[0], tuple) and first[0][0] == 'W':
                return (WILSON_FUND,) * first[0][1]
            chain = []
            for (n, exp) in hf:
                chain += [self._gen(n)] * exp
            return tuple(chain)

        out = []
        for seed, coef in _gauge_mul(_seed(g), _seed(h)).items():
            if not coef.is_zero():
                out.append((coef, _to_gens(seed)))
        return out

    def canonical_cone_order(self, gens):
        return tuple(sorted(
            gens,
            key=lambda g: (1 if is_wilson_label(g) else 0,
                           g if not is_wilson_label(g) else (0,))))

    # -- cone-label bijection (gauge part; flavour rides separately) ----

    def to_cone_label(self, native_label):
        h_factors = native_label[0]
        if not h_factors:
            cone = self._cone(0)
            return cone, {g: 0 for g in cone}
        first = h_factors[0]
        if isinstance(first[0], tuple) and first[0][0] == 'W':
            return self._wilson_cone, {WILSON_FUND: 1}
        ns = sorted({n for (n, exp) in h_factors if exp > 0})
        a = cone_index_for(ns[0], ns[-1])
        if a is None:
            raise ValueError(f"to_cone_label: {ns} not in one cone")
        cone = self._cone(a)
        powers = {g: 0 for g in cone}
        for (n, exp) in h_factors:
            powers[(n,)] = powers.get((n,), 0) + exp
        return cone, powers

    def from_cone_label(self, gens, powers):
        factors = sorted(((g[0] if not is_wilson_label(g) else g), p)
                         for g, p in powers.items() if p > 0)
        return (tuple(factors), (0, 0, 0))


__all__ = ["SU2Nf3ConeData", "cone_index_for"]
