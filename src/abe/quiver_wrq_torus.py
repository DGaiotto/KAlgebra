"""`quiver_wrq_torus` — `QuiverWRQTorus`: linear U(N_i) chains with per-node
fundamentals on the **group-general WRQ substrate**.

The multi-node generalization of `matter_wrq_torus.MatterWRQTorus`, over the
**product root datum** (`root_datum.product_datum` — block-embedded roots, no
cross-block roots, per-block atom phases and ρ signs):

  * μ-slots: links first (n−1), then flavours grouped by node — the
    `quiver_urq_torus._slots` layout;
  * rungs of `Z(m⃗)`: links on negative pair-differences `d_{jl} = m_j − m_l`
    (`j` in node `e`, `l` in node `e+1`, `vw = e_j − e_l` — a weight, not a
    root: no new poles), fundamentals on negative `m_j`;
  * multiply = the WRQ gauge cocycle over the product datum (which
    block-factorizes by construction) × the finite matter/link cocycle `W`
    (triangular division);
  * ρ/ρ⁻¹ = the product-datum gauge twist ± the image/source atom's matter
    factor (Z-top monomial, level star);
  * trace = link/flavour Nahm cells inserted into the datum-general
    `trace_residual` over the product datum (measure, `(𝖖²;𝖖²)^{2D}`
    prefactor and `|W| = ∏ N_a!` all come out of the datum).

Certified against the certified `QuiverURQTorus` at U(2)×U(1) and U(2)×U(2)
(the suite in the source repository): transport `vr_to_tr` is an isomorphism on
multiply, ρ/ρ⁻¹, W1, and the μ-refined pairing traces.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laurent_poly import LaurentPoly
from root_datum import u_n, product_datum
from weyl_torus_ring import TorusLaurent, TorusRational
from wrq_torus import CC, _rho_Gtilde, trace_residual, WRQTorus


__all__ = ["QuiverWRQTorus", "quiver_datum"]


def quiver_datum(ranks):
    """The product `u_n` datum of a linear chain."""
    return product_datum([u_n(r) for r in ranks])


def _offsets(ranks):
    off = [0]
    for r in ranks:
        off.append(off[-1] + r)
    return off


def _n_slots(ranks, Nf):
    return (len(ranks) - 1) + sum(Nf)


def _flavour_slot(ranks, Nf, a, i):
    return (len(ranks) - 1) + sum(Nf[:a]) + i


def _rungs(atom, ranks, Nf):
    """`(slot, vw, shift)` rungs of `Z(m⃗)` at `atom` — links on negative
    pair-differences, fundamentals on negative `m_j` (bar-centered)."""
    n = len(ranks)
    off = _offsets(ranks)
    M = off[-1]
    out = []
    for e in range(n - 1):
        for j in range(off[e], off[e + 1]):
            for l in range(off[e + 1], off[e + 2]):
                d = atom[j] - atom[l]
                if d < 0:
                    dep = -d
                    vw = tuple((1 if t == j else 0) - (1 if t == l else 0)
                               for t in range(M))
                    for s in range(dep):
                        out.append((e, vw, 2 * s - dep + 1))
    for a in range(n):
        for i in range(Nf[a]):
            slot = _flavour_slot(ranks, Nf, a, i)
            for j in range(off[a], off[a + 1]):
                if atom[j] < 0:
                    dep = -atom[j]
                    vw = tuple(1 if t == j else 0 for t in range(M))
                    for s in range(dep):
                        out.append((slot, vw, 2 * s - dep + 1))
    return out


def _rung_levels(datum, atom, ranks, Nf) -> dict:
    """`Z(m⃗)` per μ-level `{k⃗: TorusRational}` — finite bar-centered rung
    expansion."""
    L = _n_slots(ranks, Nf)
    d = datum.dim
    levels = {(0,) * L: {(0,) * d: LaurentPoly({0: 1})}}
    for (slot, vw, sh) in _rungs(atom, ranks, Nf):
        out: dict = {}
        for kv, row in levels.items():
            for ve, c in row.items():
                out.setdefault(kv, {}).setdefault(ve, LaurentPoly.zero())
                out[kv][ve] = out[kv][ve] + c
                kv2 = tuple(x + (1 if t == slot else 0)
                            for t, x in enumerate(kv))
                ve2 = tuple(x + w for x, w in zip(ve, vw))
                out.setdefault(kv2, {}).setdefault(ve2, LaurentPoly.zero())
                out[kv2][ve2] = out[kv2][ve2] + c * LaurentPoly({sh: 1})
        levels = out
    return {kv: TorusRational.from_laurent(TorusLaurent(
        datum, {ve: c for ve, c in row.items() if not c.is_zero()}))
        for kv, row in levels.items()}


_W_CACHE: dict = {}


def _quiver_w_cocycle(datum, m, mp, ranks, Nf) -> dict:
    """`W_{m⃗,m⃗′} = T_{−m⃗′}(Z_{m⃗})·T_{m⃗}(Z_{m⃗′})/Z_{m⃗+m⃗′}` per μ-level —
    finite net numerator by triangular division."""
    key = (tuple(m), tuple(mp), tuple(ranks), tuple(Nf))
    hit = _W_CACHE.get(key)
    if hit is not None:
        return hit
    t = tuple(x + y for x, y in zip(m, mp))
    Zm = _rung_levels(datum, m, ranks, Nf)
    Zmp = _rung_levels(datum, mp, ranks, Nf)
    Zt = _rung_levels(datum, t, ranks, Nf)
    neg_mp = tuple(-x for x in mp)
    num: dict = {}
    for k1, z1 in Zm.items():
        a = z1.q_shift(neg_mp)
        for k2, z2 in Zmp.items():
            b = z2.q_shift(tuple(m))
            k = tuple(x + y for x, y in zip(k1, k2))
            term = (a * b).simplify()
            num[k] = term if k not in num else (num[k] + term).simplify()
    budget = max((sum(k) for k in num), default=0) + len(
        _rungs(t, ranks, Nf)) + 1
    seen = set(num)
    for k in list(num):
        for dz in Zt:
            seen.add(tuple(x + y for x, y in zip(k, dz)))
    W: dict = {}
    zero = TorusRational.zero(datum)
    for k in sorted(seen, key=lambda k: (sum(k), k)):
        acc = num.get(k, zero)
        for kp, w in W.items():
            dr = tuple(x - y for x, y in zip(k, kp))
            if any(x < 0 for x in dr) or not any(dr):
                continue
            z = Zt.get(dr)
            if z is None:
                continue
            acc = (acc + (w * z * TorusRational.from_scalar(
                datum, LaurentPoly({0: -1})))).simplify()
        if not acc.is_zero():
            if sum(k) > budget:
                raise NotImplementedError(
                    f"quiver W[{m},{mp}]: tail past the rung budget at {k}")
            W[k] = acc
    _W_CACHE[key] = W
    return W


class QuiverWRQTorus:
    """A linear-chain U(N_i)+fundamentals element on the WRQ substrate:
    residuals `{atom m⃗: {μ-level k⃗: TorusRational}}` over the product datum."""

    __slots__ = ("ranks", "Nf", "datum", "_f", "_d", "_L")

    def __init__(self, ranks, Nf, f: dict, datum=None):
        self.ranks = tuple(int(r) for r in ranks)
        self.Nf = tuple(int(x) for x in Nf)
        self.datum = datum if datum is not None else quiver_datum(self.ranks)
        self._d = self.datum.dim
        self._L = _n_slots(self.ranks, self.Nf)
        out: dict = {}
        for m, row in f.items():
            dst = {}
            for k, fr in row.items():
                fr = fr.simplify()
                if not fr.is_zero():
                    dst[tuple(k)] = fr
            if dst:
                out[tuple(m)] = dst
        self._f = out

    # ----- reads -----
    def residuals(self) -> dict:
        return {m: dict(row) for m, row in self._f.items()}

    def support(self):
        return sorted(self._f)

    def is_zero(self) -> bool:
        return not self._f

    def _like(self, f):
        return QuiverWRQTorus(self.ranks, self.Nf, f, datum=self.datum)

    # ----- per-μ⃗-level family (the decompose substrate) -----
    def to_family(self) -> dict:
        """Per-link/flavour-level `WRQTorus` slices `{k⃗: WRQTorus}` (over the
        product datum) — the joint gauge content at each level."""
        out: dict = {}
        for m, row in self._f.items():
            for k, fr in row.items():
                out.setdefault(k, {})[m] = fr
        return {k: WRQTorus(self.datum, f) for k, f in out.items()}

    @classmethod
    def from_family(cls, ranks, Nf, family, datum=None) -> "QuiverWRQTorus":
        """Reassemble from `{k⃗: WRQTorus}` (inverse of `to_family`)."""
        f: dict = {}
        for k, U in family.items():
            for m, fr in U.residuals().items():
                f.setdefault(tuple(m), {})[tuple(k)] = fr
        return cls(ranks, Nf, f, datum=datum)

    def _scaled(self, C, k0):
        """`C(𝖖)·(μ⃗-shift by k0)` — scale every residual by the q-Laurent `C`
        and shift every level `k ↦ k + k0` (the attribution subtract step)."""
        sc = TorusRational.from_scalar(self.datum, C)
        f: dict = {}
        for m, row in self._f.items():
            dst = f.setdefault(m, {})
            for k, fr in row.items():
                kk = tuple(a + b for a, b in zip(k, k0))
                dst[kk] = (fr * sc).simplify()
        return self._like(f)

    # ----- ring ops -----
    def __add__(self, other: "QuiverWRQTorus") -> "QuiverWRQTorus":
        out = {m: dict(row) for m, row in self._f.items()}
        for m, row in other._f.items():
            dst = out.setdefault(m, {})
            for k, fr in row.items():
                dst[k] = (dst[k] + fr).simplify() if k in dst else fr
        return self._like(out)

    def __mul__(self, other: "QuiverWRQTorus") -> "QuiverWRQTorus":
        out: dict = {}
        for m, row1 in self._f.items():
            for mp, row2 in other._f.items():
                t = tuple(x + y for x, y in zip(m, mp))
                W = _quiver_w_cocycle(self.datum, m, mp, self.ranks, self.Nf)
                Rt = CC(self.datum, m, mp)
                neg_mp = tuple(-x for x in mp)
                dst = out.setdefault(t, {})
                for k1, f1 in row1.items():
                    a = f1.q_shift(neg_mp)
                    for k2, f2 in row2.items():
                        b = f2.q_shift(m)
                        base = (a * b * Rt).simplify()
                        for r, w in W.items():
                            K = tuple(x + y + z for x, y, z in zip(k1, k2, r))
                            term = (base * w).simplify()
                            dst[K] = term if K not in dst else (
                                dst[K] + term).simplify()
        return self._like(out)

    def bar(self) -> "QuiverWRQTorus":
        return self._like({m: {k: fr.bar() for k, fr in row.items()}
                           for m, row in self._f.items()})

    def well_formed_w1(self) -> bool:
        return self.bar() == self

    def well_formed(self):
        """W1+W2 canonical certificate on the WRQ quiver substrate: whole-element
        bar-invariance (W1), and the base-level slice a single leading Weyl orbit
        of multiplicity 1 reading the joint product-datum label (W2).  Returns
        `((m⃗, e⃗), k0)` — the flat joint label + base level — when well-formed,
        else `False`.  Mirrors `MatterWRQTorus.well_formed` on the product datum:
        the link/matter dressing is level shifts over the base joint canonical,
        so the acceptance reduces to whole-element W1 + the base slice's pure
        `WRQTorus.well_formed`."""
        if self.bar() != self:                       # W1 on all levels
            return False
        fam = self.to_family()
        if not fam:
            return False
        k0 = min(fam, key=lambda k: (sum(k), k))
        base = fam[k0].well_formed()                  # pure WRQTorus W1+W2 read
        if base is False:
            return False
        return (base, k0)

    def __eq__(self, other):
        if not isinstance(other, QuiverWRQTorus):
            return NotImplemented
        zero = TorusRational.zero(self.datum)
        for m in set(self._f) | set(other._f):
            r1, r2 = self._f.get(m, {}), other._f.get(m, {})
            for k in set(r1) | set(r2):
                if not (r1.get(k, zero) - r2.get(k, zero)).simplify().is_zero():
                    return False
        return True

    # ----- ρ -----
    def _rungs_ED(self, atom):
        E = [0] * self._d
        D = [0] * self._L
        for (slot, vw, _sh) in _rungs(atom, self.ranks, self.Nf):
            D[slot] += 1
            for t in range(self._d):
                E[t] += vw[t]
        return E, D

    def rho(self) -> "QuiverWRQTorus":
        out: dict = {}
        for m, row in self._f.items():
            a = tuple(-x for x in m)
            gt = _rho_Gtilde(self.datum, m, inverse=False)
            E, D = self._rungs_ED(a)
            mono = None
            if any(E):
                mono = TorusRational.from_laurent(TorusLaurent(
                    self.datum, {tuple(-x for x in E): LaurentPoly({0: 1})}))
            for k, fr in row.items():
                g = (fr.vinv() * gt).simplify()
                if mono is not None:
                    g = (g * mono).simplify()
                k2 = tuple(-x - dd for x, dd in zip(k, D))
                dst = out.setdefault(a, {})
                dst[k2] = g if k2 not in dst else (dst[k2] + g).simplify()
        return self._like(out)

    def rho_inverse(self) -> "QuiverWRQTorus":
        out: dict = {}
        for a, row in self._f.items():
            m = tuple(-x for x in a)
            gtinv = _rho_Gtilde(self.datum, a, inverse=True)
            E, D = self._rungs_ED(a)
            mono = None
            if any(E):
                mono = TorusRational.from_laurent(TorusLaurent(
                    self.datum, {tuple(E): LaurentPoly({0: 1})}))
            for k2, fr in row.items():
                g = fr if mono is None else (fr * mono).simplify()
                k = tuple(-x - dd for x, dd in zip(k2, D))
                g = (g.vinv() * gtinv).simplify()
                dst = out.setdefault(m, {})
                dst[k] = g if k not in dst else (dst[k] + g).simplify()
        return self._like(out)

    # ----- μ-refined trace -----
    def trace(self, K: int = 8, W: int = 4) -> dict:
        """`{μ⃗-level: LaurentPoly}` — link/flavour Nahm cells inserted into
        the product-datum Schur residue of the magnetic-0 residual."""
        from habiro import HabiroElement
        d, L = self._d, self._L
        ranks, Nf = self.ranks, self.Nf
        n = len(ranks)
        off = _offsets(ranks)
        row0 = self._f.get((0,) * d)
        if not row0:
            return {}
        Kq = K + 10

        def a_n(nn):
            return HabiroElement.nahm_term((-1) ** nn, nn, [nn]).expand(Kq + W)

        cells = []
        for e in range(n - 1):
            for j in range(off[e], off[e + 1]):
                for l in range(off[e + 1], off[e + 2]):
                    vw = tuple((1 if t == j else 0) - (1 if t == l else 0)
                               for t in range(d))
                    cells.append((e, vw))
        for a in range(n):
            for i in range(Nf[a]):
                slot = _flavour_slot(ranks, Nf, a, i)
                for j in range(off[a], off[a + 1]):
                    vw = tuple(1 if t == j else 0 for t in range(d))
                    cells.append((slot, vw))
        terms = {((0,) * L, (0,) * d): LaurentPoly({0: 1})}
        for (slot, vw) in cells:
            for sign in (+1, -1):
                new: dict = {}
                for nn in range(0, W + 1):
                    c = a_n(nn)
                    for (lv, ve), z in terms.items():
                        lv2 = tuple(x + (sign * nn if t == slot else 0)
                                    for t, x in enumerate(lv))
                        if sum(abs(x) for x in lv2) > W:
                            continue
                        ve2 = tuple(x + sign * nn * w
                                    for x, w in zip(ve, vw))
                        new[(lv2, ve2)] = new.get(
                            (lv2, ve2), LaurentPoly.zero()) + z * c
                terms = new
        Mfac: dict = {}
        for (lv, ve), z in terms.items():
            if not z.is_zero():
                Mfac.setdefault(lv, {})[ve] = z
        Mfac = {lv: TorusRational.from_laurent(TorusLaurent(self.datum, rr))
                for lv, rr in Mfac.items()}
        out: dict = {}
        for k, f0 in row0.items():
            for nlv, mf in Mfac.items():
                lp = trace_residual(self.datum, (f0 * mf).simplify(), K)
                if lp.is_zero():
                    continue
                mu = tuple(x + y for x, y in zip(k, nlv))
                out[mu] = out.get(mu, LaurentPoly.zero()) + lp
        return {mu: LaurentPoly({e: c for e, c in lp._coeffs.items()
                                 if 0 <= e <= K})
                for mu, lp in out.items()
                if any(0 <= e <= K and c for e, c in lp._coeffs.items())}

    def __repr__(self):
        if not self._f:
            return "QuiverWRQTorus(0)"
        return ("QuiverWRQTorus{" + ", ".join(
            f"{m}:{sorted(row)}" for m, row in sorted(self._f.items())) + "}")
