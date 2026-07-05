"""`matter_wrq_torus` — `MatterWRQTorus`: the U(N)+N_f enriched torus on the
**group-general WRQ substrate** (D10 Stage 2).

The flavoured extension of `WRQTorus` per the validated design
(`tests/test_flavoured_wrq.py`): the gauge foundation is unchanged (the WRQ
cocycle IS the flavoured gauge backbone), and matter enters **additively** —

  * a **μ-level grading** on residuals: `f = {atom m: {k ∈ Z^{N_f}:
    TorusRational}}` (one slot per fundamental flavour, mirroring the
    certified `QuiverURQTorus` level buckets at one node);
  * the **matter rung dressing** `Z(m)` — per flavour `i` and colour `j` with
    `m_j < 0`, the bar-centered rung ladder `∏_s (1 + 𝖖^{2s−d+1} μ_i v_j)`,
    `d = −m_j` — polynomial per level (NO new poles: denominators stay gauge
    roots);
  * the **matter cocycle** `W_{m,m'} = T_{−m'}(Z_m)·T_m(Z_{m'})/Z_{m+m'}` per
    μ-level — a finite net numerator, computed once by triangular division
    (the `quiver_urq_torus._quiver_cocycle` transcription);
  * ρ = the pure-gauge WRQ twist (`_rho_Gtilde`) followed by the matter
    factor of the image atom: division by the q-free Z-top monomial
    `v^{E(−m)}` and the **level star** `k_i ↦ −k_i − D_i(−m)` (`D` = the
    slot's rung count);
  * the **μ-refined trace**: the flavour matter factors
    `∏_{i,j} E(μ_i v_j)·E(μ_i⁻¹/v_j)` (Nahm window `W`) inserted into the
    datum-general Schur residue `wrq_torus.trace_residual` (which carries the
    D10 k=0-pole folding + completeness pad).

Public language: atoms `U_m`, monomials `v^e`, μ-levels, and the cocycles
`R̃·W` — no `u`'s, no dressing operators (user ruling 2026-07-02).

Certification (`tests/test_matter_wrq_torus.py`): transported
`UNNfKAlgebra(2,1)` / `(2,2)` charts — multiply, ρ, and the μ-refined trace
commute with the `VRational → TorusRational` transport, cross-engine against
the certified `QuiverURQTorus`.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laurent_poly import LaurentPoly
from root_datum import RootDatum, u_n
from weyl_torus_ring import TorusLaurent, TorusRational
from wrq_torus import cocycle_Rtilde, _rho_Gtilde, trace_residual, WRQTorus


__all__ = ["MatterWRQTorus", "vr_to_tr", "tr_to_vr"]


# ---------------------------------------------------------------------------
# transport: VRational (e-basis engine) → TorusRational (WRQ ring)
# ---------------------------------------------------------------------------
def vr_to_tr(datum: RootDatum, vr) -> TorusRational:
    """Convert a `VRational` to a `TorusRational` over the same `u_n` datum:
    `(v_i − 𝖖^M v_j) = v_i·(1 − 𝖖^M v^{e_j−e_i})`, so each denominator key
    `(i, j, M)` becomes the root factor `(e_j−e_i, M)` with a `v_i^{−1}`
    numerator monomial per multiplicity."""
    vr = vr.simplify()
    if getattr(vr, "_sq", None):
        raise NotImplementedError("vr_to_tr: sq denominators not handled")
    d = datum.dim
    comp = [0] * d                       # accumulated v_i^{-mult} correction
    den: dict = {}
    for (i, j, M), mult in vr.den.items():
        alpha = tuple((1 if t == j else 0) - (1 if t == i else 0)
                      for t in range(d))
        den[(alpha, M)] = den.get((alpha, M), 0) + mult
        comp[i] -= mult
    num = TorusLaurent(datum, {
        tuple(ve[t] + comp[t] for t in range(d)): LaurentPoly(dict(lp._coeffs))
        for ve, lp in vr.num._terms.items()})
    return TorusRational(datum, num, den)


def tr_to_vr(datum: RootDatum, tr: TorusRational):
    """The inverse of `vr_to_tr`: a `TorusRational` over a `u_n` datum back to a
    `VRational`.  Each denominator root factor `(α, M)` must be a `u_n` root
    `α = e_j − e_i` (single +1 / single −1), recovered as the key `(i, j, M)`
    with the matching `v_i^{−mult}` numerator correction undone.  Honest-fails
    on a non-`u_n`-root denominator (the transport is only defined on the
    unitary-datum image)."""
    from abelianized_torus import VLaurent, VRational
    tr = tr.simplify()
    d = datum.dim
    comp = [0] * d                       # accumulated v_i^{-mult} correction
    den: dict = {}
    for (alpha, M), mult in tr._den.items():
        pos = [t for t in range(d) if alpha[t] == 1]
        neg = [t for t in range(d) if alpha[t] == -1]
        if len(pos) != 1 or len(neg) != 1 or sum(abs(x) for x in alpha) != 2:
            raise NotImplementedError(
                f"tr_to_vr: denominator root {alpha} is not a u_n root e_j−e_i")
        j, i = pos[0], neg[0]
        den[(i, j, M)] = den.get((i, j, M), 0) + mult
        comp[i] -= mult
    num = VLaurent({
        tuple(ve[t] - comp[t] for t in range(d)): LaurentPoly(dict(lp._coeffs))
        for ve, lp in tr._num._t.items()}, n=d)
    return VRational(num, den, n=d)


# ---------------------------------------------------------------------------
# matter rungs / Z-levels / matter cocycle (1-node flavour transcription)
# ---------------------------------------------------------------------------
def _flavour_rungs(atom, N: int, Nf: int):
    """Rungs of `Z(m)` at `atom`: `(slot i, vw = e_j, shift)` for each flavour
    `i` and colour `j` with `atom_j < 0`, shifts bar-centered."""
    out = []
    for i in range(Nf):
        for j in range(N):
            if atom[j] < 0:
                dep = -atom[j]
                vw = tuple(1 if t == j else 0 for t in range(N))
                for s in range(dep):
                    out.append((i, vw, 2 * s - dep + 1))
    return out


def _rung_levels(datum, atom, Nf: int) -> dict:
    """`Z(m)` per μ-level `{k: TorusRational}` — the finite bar-centered rung
    expansion (each rung contributes `1 + 𝖖^{shift} μ_slot v^{vw}`)."""
    d = datum.dim
    levels = {(0,) * Nf: {(0,) * d: LaurentPoly({0: 1})}}
    for (slot, vw, sh) in _flavour_rungs(atom, d, Nf):
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


def _matter_cocycle(datum, m, mp, Nf: int) -> dict:
    """`W_{m,m'} = T_{−m'}(Z_m)·T_m(Z_{m'})/Z_{m+m'}` per μ-level — finite
    net numerator by triangular division (raises past the rung budget)."""
    key = (datum.name, tuple(m), tuple(mp), Nf)
    hit = _W_CACHE.get(key)
    if hit is not None:
        return hit
    t = tuple(x + y for x, y in zip(m, mp))
    Zm = _rung_levels(datum, m, Nf)
    Zmp = _rung_levels(datum, mp, Nf)
    Zt = _rung_levels(datum, t, Nf)
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
        _flavour_rungs(t, datum.dim, Nf)) + 1
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
                    f"matter cocycle W[{m},{mp}]: tail past the rung budget "
                    f"at level {k}")
            W[k] = acc
    _W_CACHE[key] = W
    return W


# ---------------------------------------------------------------------------
# the element
# ---------------------------------------------------------------------------
class MatterWRQTorus:
    """A U(N)+N_f enriched-torus element on the WRQ substrate: residuals
    `{atom m: {μ-level k ∈ Z^{N_f}: TorusRational}}`."""

    __slots__ = ("datum", "Nf", "_f", "_d")

    def __init__(self, datum: RootDatum, Nf: int, f: dict):
        self.datum = datum
        self.Nf = int(Nf)
        self._d = datum.dim
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

    # ----- per-μ-level family (the decompose substrate) -----
    def to_family(self) -> dict:
        """Per-μ-level `WRQTorus` slices `{k⃗: WRQTorus}` (the gauge content at
        each matter level) — the read `PureUNWRQ.decompose` consumes."""
        out: dict = {}
        for m, row in self._f.items():
            for k, fr in row.items():
                out.setdefault(k, {})[m] = fr
        return {k: WRQTorus(self.datum, f) for k, f in out.items()}

    @classmethod
    def from_family(cls, datum, Nf, family) -> "MatterWRQTorus":
        """Reassemble from `{k⃗: WRQTorus}` (inverse of `to_family`)."""
        f: dict = {}
        for k, U in family.items():
            for m, fr in U.residuals().items():
                f.setdefault(tuple(m), {})[tuple(k)] = fr
        return cls(datum, Nf, f)

    def _scaled(self, C, k0):
        """`C(𝖖)·(μ-shift by k0)` — scale every residual by the q-Laurent `C`
        and shift every μ-level `k ↦ k + k0` (the attribution's subtract step)."""
        sc = TorusRational.from_scalar(self.datum, C)
        f: dict = {}
        for m, row in self._f.items():
            dst = f.setdefault(m, {})
            for k, fr in row.items():
                kk = tuple(a + b for a, b in zip(k, k0))
                dst[kk] = (fr * sc).simplify()
        return MatterWRQTorus(self.datum, self.Nf, f)

    # ----- ring ops -----
    def __add__(self, other: "MatterWRQTorus") -> "MatterWRQTorus":
        out = {m: dict(row) for m, row in self._f.items()}
        for m, row in other._f.items():
            dst = out.setdefault(m, {})
            for k, fr in row.items():
                dst[k] = (dst[k] + fr).simplify() if k in dst else fr
        return MatterWRQTorus(self.datum, self.Nf, out)

    def __mul__(self, other: "MatterWRQTorus") -> "MatterWRQTorus":
        """`U_m U_{m'} = R̃_{m,m'}·W_{m,m'}·U_{m+m'}` — the WRQ gauge cocycle
        times the finite matter cocycle, μ-levels convolved."""
        out: dict = {}
        for m, row1 in self._f.items():
            neg_m = tuple(-x for x in m)
            for mp, row2 in other._f.items():
                t = tuple(x + y for x, y in zip(m, mp))
                W = _matter_cocycle(self.datum, m, mp, self.Nf)
                Rt = cocycle_Rtilde(self.datum, m, mp)
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
        return MatterWRQTorus(self.datum, self.Nf, out)

    def bar(self) -> "MatterWRQTorus":
        return MatterWRQTorus(self.datum, self.Nf, {
            m: {k: fr.bar() for k, fr in row.items()}
            for m, row in self._f.items()})

    def well_formed_w1(self) -> bool:
        return self.bar() == self

    def well_formed(self):
        """W1+W2 canonical certificate on the WRQ matter substrate: whole-element
        bar-invariance (W1), and the base-μ-level slice a single leading Weyl
        orbit of multiplicity 1 reading the lower-Kapustin pure label (W2).
        Returns `((m, e), k0)` — the pure label + base μ-level — when
        well-formed, else `False`.  Mirrors `MatterURQTorus.well_formed`
        (`(joint[0], k0)`): the matter dressing is χ_w-level shifts over the base
        pure canonical, so the acceptance reduces to whole-element W1 + the base
        slice's pure `WRQTorus.well_formed`.  (The WRQ read returns the public
        lower-Kapustin label directly — no joint-w0 frame reversal.)"""
        if self.bar() != self:                       # W1 on all μ-levels
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
        if not isinstance(other, MatterWRQTorus):
            return NotImplemented
        zero = TorusRational.zero(self.datum)
        keys = set(self._f) | set(other._f)
        for m in keys:
            r1, r2 = self._f.get(m, {}), other._f.get(m, {})
            for k in set(r1) | set(r2):
                if not (r1.get(k, zero) - r2.get(k, zero)).simplify().is_zero():
                    return False
        return True

    # ----- ρ -----
    def _rungs_ED(self, atom):
        """`(E, D)` of `Z(atom)`: the q-free Z-top v-exponent and the
        per-slot rung counts."""
        E = [0] * self._d
        D = [0] * self.Nf
        for (slot, vw, _sh) in _flavour_rungs(atom, self._d, self.Nf):
            D[slot] += 1
            for t in range(self._d):
                E[t] += vw[t]
        return E, D

    def rho(self) -> "MatterWRQTorus":
        """ρ — the gauge √measure twist (`G̃_m`), then the matter factor of
        the **image atom**: division by the q-free Z-top monomial `v^{E(−m)}`
        and the level star `k_i ↦ −k_i − D_i(−m)`."""
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
        return MatterWRQTorus(self.datum, self.Nf, out)

    def rho_inverse(self) -> "MatterWRQTorus":
        """ρ⁻¹ — undo the matter factor at the **source atom** first
        (multiply by `v^{E(a)}`, un-star the levels), then the inverse gauge
        twist.  Certified by ρ∘ρ⁻¹ = id and the engine transport."""
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
        return MatterWRQTorus(self.datum, self.Nf, out)

    # ----- μ-refined trace -----
    def trace(self, K: int = 8, W: int = 4) -> dict:
        """`{μ-level: LaurentPoly}` — the flavour matter factors
        `∏_{i,j} E(μ_i v_j)E(μ_i⁻¹ v_j⁻¹)` (Nahm window `W`) inserted into the
        datum-general Schur residue of the magnetic-0 residual."""
        from habiro import HabiroElement
        d, Nf = self._d, self.Nf
        row0 = self._f.get((0,) * d)
        if not row0:
            return {}
        Kq = K + 10

        def a_n(nn):
            return HabiroElement.nahm_term((-1) ** nn, nn, [nn]).expand(Kq + W)

        cells = [(i, tuple(1 if t == j else 0 for t in range(d)))
                 for i in range(Nf) for j in range(d)]
        terms = {((0,) * Nf, (0,) * d): LaurentPoly({0: 1})}
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
            return "MatterWRQTorus(0)"
        return ("MatterWRQTorus{" + ", ".join(
            f"{m}:{sorted(row)}" for m, row in sorted(self._f.items())) + "}")
