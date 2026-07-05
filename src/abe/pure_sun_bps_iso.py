"""`pure_sun_bps_iso` — the **all-N** `KAlgebraIso` between the pure-SU(N)
`AbeKAlgebra` realisation (`PureSUNKAlgebra`) and the **BPS** chart
(`pure_ade_kalgebra([("A", N−1)])`), on the anchor + rigidify architecture
(user directions 2026-07-02: "tropical maps are not linear … mult and rho
rigidify the lot", "all N if possible").

`SUNTropicalMap(N)` — BPS coordinates interleaved over the N−1 Kronecker
pairs `(n_1, e_1, …, n_{N−1}, e_{N−1})`:

  * **Wilson anchor**: `χ_λ ↦ (0, −p_1, 0, −p_2, …)` with `(p_i)` the Dynkin
    labels of λ (consecutive differences of the descending sort) —
    antidominant electric.  ρ conjugates the complex reps (fund ↔ antifund
    σ-2-cycles); the real reps (χ_adj, the middle fundamental of even N) are
    σ-fixed.
  * **Monopole anchor**: anti-dominant trace-zero `m ↦ (n_1, 0, n_2, 0, …)`
    with `(n_i)` the coroot coordinates (partial sums of the descending
    sort) — dominant magnetic; the adjoint `θ^∨` is `(1, 0, …, 1, 0)`.
  * Everything else **derived**: σ-transport first (ρ-images of
    anchored/cached labels — the Witten-drift cases), else exact
    structure-constant matching on `W_{λ↓}·L_{m,0}`; cached; honest-fail on
    coefficient collision.

Certified: N=3 (`tests/test_pure_su3_bps_iso.py`, via the delegating
`pure_su3_bps_iso`) and N=4 (`tests/test_pure_su4_bps_iso.py`) — anchors,
σ-structure, multiplicativity, ρ-equivariance, trace-equivariance on the
triality-neutral basket.  Feasibility: the A_{N−1} chart has `|spec| =
(N−1)·N` with rank `2(N−1)` F-solves — N=4 runs in background-job time;
N≥5 is constructible on demand (the map itself is closed-form on anchors
and lazy on derivations)."""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.dirname(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly
from pure_ade_lattice import pure_ade_kalgebra
from pure_sun_kalgebra import PureSUNKAlgebra


__all__ = ["SUNTropicalMap", "pure_sun_bps_iso"]

_ONE = LaurentPoly.one()


class SUNTropicalMap:
    """Anchored + product/ρ-rigidified `(m, λ) ↔ γ` map for pure SU(N)."""

    def __init__(self, abe, bps, N: int):
        self._A, self._B = abe, bps
        self._N = int(N)
        self._g: dict = {}
        self._inv: dict = {}

    def _learn(self, label, g):
        self._g[label] = g
        self._inv[g] = label

    def _interleave(self, ns, es):
        g = []
        for i in range(self._N - 1):
            g.extend((ns[i], es[i]))
        return tuple(g)

    def _dynkin(self, lam):
        s = sorted(lam, reverse=True)
        return tuple(s[i] - s[i + 1] for i in range(self._N - 1))

    def _coroot(self, m):
        s = sorted(m, reverse=True)
        out, acc = [], 0
        for i in range(self._N - 1):
            acc += s[i]
            out.append(acc)
        return tuple(out)

    def gamma(self, label):
        N = self._N
        m, lam = tuple(label[0]), tuple(label[1])
        hit = self._g.get((m, lam))
        if hit is not None:
            return hit
        zero = (0,) * N
        if m == zero:                          # Wilson anchor (Dynkin)
            p = self._dynkin(lam)
            g = self._interleave((0,) * (N - 1), tuple(-x for x in p))
            self._learn((m, lam), g)
            return g
        if lam == zero:                        # bare monopole anchor (coroot)
            n = self._coroot(m)
            g = self._interleave(n, (0,) * (N - 1))
            self._learn((m, lam), g)
            return g

        # σ-transport first (ρ-images of anchored/cached labels)
        def _cheap(s):
            return s in self._g or s[0] == zero or s[1] == zero
        for step, b_step in ((self._A.rho_inverse, self._B.rho),
                             (self._A.rho, self._B.rho_inverse)):
            try:
                src = step((m, lam))
            except NotImplementedError:
                continue
            src = (tuple(src[0]), tuple(src[1]))
            if _cheap(src):
                g = b_step(self.gamma(src))
                self._learn((m, lam), g)
                return g

        # derived: match W_{λ↓}·L_{m,0} against the BPS anchor product
        a = (zero, tuple(sorted(lam, reverse=True)))
        b = (m, zero)
        ma = dict(self._A.multiply(a, b).terms)
        gb = dict(self._B.multiply(self.gamma(a), self.gamma(b)).terms)
        by_c = {}
        for g, c in gb.items():
            by_c.setdefault(str(c), []).append(g)
        for lab, c in sorted(ma.items()):
            lab = (tuple(lab[0]), tuple(lab[1]))
            bucket = by_c.get(str(c), [])
            known = self._g.get(lab)
            if known is not None:
                if known in bucket:
                    bucket.remove(known)
                continue
            if len(bucket) != 1:
                raise NotImplementedError(
                    f"SUNTropicalMap: coefficient collision deriving {lab} "
                    f"from {a}·{b} (bucket {bucket})")
            self._learn(lab, bucket.pop(0))
        hit = self._g.get((m, lam))
        if hit is None:
            raise NotImplementedError(
                f"SUNTropicalMap: {label} absent from its anchor product")
        return hit

    def label(self, g):
        g = tuple(g)
        hit = self._inv.get(g)
        if hit is not None:
            return hit
        raise NotImplementedError(
            f"SUNTropicalMap: no label derived yet for {g} "
            f"(derive forward first — the inverse is cache-backed)")


def pure_sun_bps_iso(N: int, abe: PureSUNKAlgebra | None = None,
                     **abe_kwargs) -> KAlgebraIso:
    """`PureSUNKAlgebra(N) ≅ BPS pure SU(N)` — all-N constructor."""
    A = abe if abe is not None else PureSUNKAlgebra(N, **abe_kwargs)
    B = pure_ade_kalgebra([("A", N - 1)])
    T = SUNTropicalMap(A, B, N)

    def forward(label):
        return Element({T.gamma(label): _ONE})

    def inverse(g):
        return Element({T.label(g): _ONE})

    iso = KAlgebraIso(A, B, forward, inverse,
                      name=f"PureSUNKAlgebra({N}) ≅ BPS pure SU({N}) "
                           f"[anchored]")
    iso.tropical_map = T
    return iso
