"""`wrq_torus` — WRQTorus (Weyl Rational Quantum Torus), the group-general
enriched-torus substrate, Layer 2 of the `AbeKAlgebra` redesign.

This generalizes `urq_torus.URQTorus` (U(N)-specific) to any `RootDatum`, built
on the Layer-1 ring `weyl_torus_ring.WeylTorusRing`.  Elements are residual
vectors `(f_m)` over the cocharacter lattice (`m ∈ X_*(T)`, dual to the weight
lattice), `x = Σ_m f_m(𝖖^m v)·U_m`, with the atoms `U_m` carrying a single
magnetic charge `m` and the dressing `ψ_m`.

Representation policy (user ruling, this session)
-------------------------------------------------
The **residual-vector form is the core representation**, and it needs no `u`'s:
the whole algebra (multiply via the cocycle, ρ, trace, inner, the Levi layer) is
expressed over the datum's roots, with NO difference-operator / `u^a` form.  The
chart/`DOp` form (`abelianized_torus`) is **demoted to an optional, future-facing
export** — useful for *acting on 3d indices*, but its current shape is not assumed
final — and is **kept, not erased**.  Canonical generators build residual-native
here (`wilson`, and `leading_orbit` + bubbling), so DOp is not load-bearing.

This module provides the dressing layer, the algebra (multiply/ρ/trace/inner),
and the Levi/recognize layer:

  * `dressing_psi(datum, m)` — the dressing `ψ_m`, datum-general:

        ψ_m = M(m) · ∏_{α∈Φ⁺} ∏_{l=0}^{⟨α,m⟩−1} 1/(1 − 𝖖^{2l} v^α),

    with the numerator monomial `M(m)` of weight `Σ_{α>0}⟨α,m⟩·α`, q-power
    `Σ_{α>0}⟨α,m⟩(⟨α,m⟩−1) − S`, and sign `(−1)^{S + Σ_{α>0}⟨α,m⟩}`.  The only
    coordinate-specific piece is the scalar atom-normalization phase `S`
    (`RootDatum.atom_phase` — the choice of cocycle representative; U(N) keeps
    the historical `Σ_j j·m_j`).  Because the engine evaluates `S` at the
    DOMINANT rep and Weyl-transports, `S` is effectively **Weyl-invariant,
    hence NOT linear in m**, and it does NOT cancel in mixed-chamber cocycles:
    its non-linear content must be the bar-honest `−½Σ_{α>0}⟨α,m_dom⟩` (up to
    linear terms, which do cancel) or `bar` fails antimultiplicativity on
    products — the D10 su_2 finding (2026-07-01); see `RootDatum.atom_phase`.

    NOTE (public language — user ruling 2026-07-02): the tier speaks **atoms
    `U_m`, weight monomials `v^e`, residuals, and the cocycle
    `U_a U_b = R_{a,b}(v)·U_{a+b}`** only.  The `ψ` of `dressing_psi` /
    `dressing_psi_inv` below is a PRIVATE derivational device for producing
    `R_{a,b}` and the ρ-twist `G_m` — no relation `U = ψ·u` (and no `u`'s at
    all) appears on any public surface; if/when 3d boundary/interface
    questions need a difference-operator export, that is a separate, future
    surface (`abelianized_torus`, demoted by D9).
  * `dressing_psi_inv(datum, m)` — `1/ψ_m` (a Laurent times a monomial).
  * `cocycle_R(datum, a, b)` — `R_{a,b} = ψ_a · S_a(ψ_b) · ψ_{a+b}^{-1}` with the
    full normal-ordering shift `S_a : v ↦ 𝖖^{2a} v`; the cocycle of
    `U_a U_b = R_{a,b}(v)·U_{a+b}`.

Dominant `m` (`⟨α,m⟩ ≥ 0 ∀α>0`) uses the closed form; non-dominant `m` is the
Weyl transport `ψ_{w·m_dom} = w·ψ_{m_dom}`.  The multiply / ρ / trace / inner /
recognize layers build on this (subsequent increments).
"""
from __future__ import annotations

import itertools

from laurent_poly import LaurentPoly
from root_datum import RootDatum
from weyl_torus_ring import TorusLaurent, TorusRational, _divide_by_factor


def _phase_S(datum: RootDatum, m) -> int:
    """The atom-normalization phase `S` — the datum's own convention
    (`RootDatum.atom_phase`; U(N) keeps the historical `Σ_j j·m_j`, everything
    else defaults to the bar-honest root-system form `−½Σ_{α>0}⟨α,m⟩`).
    Evaluated by the engine at DOMINANT `m` and Weyl-transported, so only its
    bar-relevant (non-linear-in-m) content matters — see `RootDatum.atom_phase`."""
    return datum.atom_phase(m)


def _pairing_root_cochar(datum: RootDatum, alpha, m) -> int:
    """`⟨α, m⟩` — root α paired against the magnetic cocharacter m."""
    return datum.shift_pairing(m, alpha)


def _weyl_transport(datum, m, build_dominant):
    """`build_dominant(m_dom)` for a dominant cocharacter m; else
    `w·build_dominant(m_dom)` with `w·m_dom = m`.

    `m` is a COCHARACTER, so dominance / the dominant rep / the carrying element
    are taken on the cocharacter side (`*_cochar`); the resulting ψ is a function
    of the v-WEIGHTS, so it transports by the weight action `weyl_act`.  (For U(N)
    the two sides coincide — self-dual.)"""
    m = tuple(m)
    if datum.is_dominant_cochar(m):
        return build_dominant(m)
    m_dom = datum.dominant_cochar_rep(m)
    base = build_dominant(m_dom)
    for w in datum.weyl_elements():
        if datum.act_cochar(w, m_dom) == m:
            return base.weyl_act(w)
    raise RuntimeError(f"no Weyl element carrying cocharacter {m_dom} to {m}")


def _psi_monomial_data(datum, m):
    """`(weight, qpow, sign_is_neg)` of the numerator monomial M(m) for dominant m."""
    d = datum.dim
    wt = [0] * d
    qpow = 0
    sgn = 0
    for a in datum.positive_roots():
        am = _pairing_root_cochar(datum, a, m)        # ⟨α, m⟩ ≥ 0
        for i in range(d):
            wt[i] += am * a[i]
        qpow += am * (am - 1)
        sgn += am
    S = _phase_S(datum, m)
    qpow -= S
    sgn += S
    return tuple(wt), qpow, (sgn % 2 == 1)


def _psi_dominant(datum, m):
    wt, qpow, neg = _psi_monomial_data(datum, m)
    coeff = LaurentPoly({qpow: -1 if neg else 1})
    rat = TorusRational.from_laurent(TorusLaurent.monomial(datum, wt, coeff))
    for a in datum.positive_roots():
        am = _pairing_root_cochar(datum, a, m)
        for l in range(am):
            rat = rat * TorusRational.factor_inv(datum, a, 2 * l)
    return rat


def _psi_inv_dominant(datum, m):
    wt, qpow, neg = _psi_monomial_data(datum, m)
    # 1/ψ = M^{-1} · ∏_{α>0} ∏_l (1 − 𝖖^{2l} v^α)   (a polynomial: empty denominator)
    inv_mono = TorusLaurent.monomial(
        datum, tuple(-x for x in wt), LaurentPoly({-qpow: -1 if neg else 1}))
    den = TorusLaurent.one(datum)
    for a in datum.positive_roots():
        am = _pairing_root_cochar(datum, a, m)
        for l in range(am):
            den = den * TorusLaurent(datum, {
                (0,) * datum.dim: LaurentPoly({0: 1}),
                tuple(a): LaurentPoly({2 * l: -1}),
            })
    return TorusRational.from_laurent(inv_mono * den)


def dressing_psi(datum, m):
    """The dressing `ψ_m` as a `TorusRational`."""
    return _weyl_transport(datum, m, lambda md: _psi_dominant(datum, md))


def dressing_psi_inv(datum, m):
    """`1/ψ_m` as a `TorusRational`."""
    return _weyl_transport(datum, m, lambda md: _psi_inv_dominant(datum, md))


def cocycle_R(datum, a, b):
    """`R_{a,b} = ψ_a · S_a(ψ_b) · ψ_{a+b}^{-1}`, `S_a : v ↦ 𝖖^{2a} v`."""
    a, b = tuple(a), tuple(b)
    tot = tuple(a[i] + b[i] for i in range(datum.dim))
    shift = tuple(2 * x for x in a)
    return (dressing_psi(datum, a)
            * dressing_psi(datum, b).q_shift(shift)
            * dressing_psi_inv(datum, tot)).simplify()


def cocycle_Rtilde(datum, a, b):
    """`R̃_{a,b} = T_{−(a+b)}(R_{a,b})` — the cocycle in the f-representation (the
    half-shift `v ↦ 𝖖^{−(a+b)} v` passing from the bare `c_m` to the public `f_m`)."""
    tot = tuple(a[i] + b[i] for i in range(datum.dim))
    return cocycle_R(datum, a, b).q_shift(tuple(-x for x in tot))


# ===========================================================================
# ρ — the √-measure G-cocycle (datum-general), T2c
# ===========================================================================
def _rho_block(datum, k):
    """`(sign, qpow, wexp)` of the per-magnetic ρ block, datum-general.

    Reduces the U(N) `pure_un_kalgebra._rho_block_data` to root-sums:
      `wexp = −Σ_{α>0}⟨α,k⟩·α`,
      `qpow = −Σ_{α>0}⟨α,k⟩(⟨α,k⟩−1) + 2·S(k)`   (the `q^{2Σt k_t}` J-grading),
      `sign = (−1)^{Σ_t (d−1−2t) k_t}`.
    Validated `== _rho_block_data` for U(N)."""
    d = datum.dim
    wexp = [0] * d
    qf = 0
    for a in datum.positive_roots():
        ak = datum.shift_pairing(k, a)            # ⟨α, k⟩
        for i in range(d):
            wexp[i] += ak * a[i]
        qf += ak * (ak - 1)
    wexp = tuple(-x for x in wexp)
    qpow = -qf + 2 * _phase_S(datum, k)
    sign_exp = datum.rho_sign_exp(k)
    return (-1 if sign_exp % 2 else 1), qpow, wexp


def _rho_G(datum, m, inverse=False):
    """The closed-form twist factor `G_m` (rational in v): the √-measure
    conjugation block, `G_m = v̄(ψ_m) · mono_m · ψ_{−m}^{-1}` with
    `mono_m = sign · 𝖖^{±qpow} · v^{−wexp}`.  `inverse=True` reads the block at
    `−m` and negates the q-power (the ρ⁻¹ variant)."""
    if inverse:
        sign, qpow, wexp = _rho_block(datum, tuple(-x for x in m))
        qpow = -qpow
    else:
        sign, qpow, wexp = _rho_block(datum, m)
    mono = TorusRational.from_laurent(TorusLaurent.monomial(
        datum, tuple(-x for x in wexp), LaurentPoly({qpow: sign})))
    return (dressing_psi(datum, m).vinv()
            * mono
            * dressing_psi_inv(datum, tuple(-x for x in m))).simplify()


def _rho_Gtilde(datum, m, inverse=False):
    """`G̃_m = T_m(G_m)` — the twist factor in the f-representation (half-shift)."""
    return _rho_G(datum, m, inverse).q_shift(tuple(m))


# ===========================================================================
# WRQTorus — the residual-vector element (multiply layer, T2b)
# ===========================================================================
class WRQTorus:
    """A Weyl Rational Quantum Torus element: the residual vector `(f_m)` over the
    cocharacter lattice, `x = Σ_m f_m(𝖖^m v)·U_m`, with `f_m` a `TorusRational`
    (half-shift / f-representation convention, as in `urq_torus.URQTorus`).

    Multiply is the cocycle convolution
    `f^C_{m+m'} += T_{−m'}(f_m)·T_m(g_{m'})·R̃_{m,m'}`.  This is the group-general
    counterpart of `URQTorus.__mul__`, on the Layer-1 ring."""

    __slots__ = ("datum", "_d", "_f")

    def __init__(self, datum, f_residuals: dict):
        self.datum = datum
        self._d = datum.dim
        f = {}
        for m, fr in f_residuals.items():
            fr = fr.simplify()
            if not fr.is_zero():
                f[tuple(m)] = fr
        self._f = f

    @classmethod
    def zero(cls, datum):
        return cls(datum, {})

    @classmethod
    def from_f(cls, datum, f):
        return cls(datum, f)

    def residual(self, m):
        return self._f.get(tuple(m), TorusRational.zero(self.datum))

    def residuals(self):
        return dict(self._f)

    def support(self):
        return sorted(self._f)

    def is_zero(self):
        return not self._f

    def __add__(self, other):
        out = dict(self._f)
        for m, fr in other._f.items():
            out[m] = (out[m] + fr).simplify() if m in out else fr
        return WRQTorus(self.datum, out)

    def __mul__(self, other):
        d = self._d
        out = {}
        for m, fm in self._f.items():
            neg_m = tuple(-x for x in m)
            for mp, gmp in other._f.items():
                tot = tuple(m[i] + mp[i] for i in range(d))
                term = (fm.q_shift(tuple(-x for x in mp))
                        * gmp.q_shift(m)
                        * cocycle_Rtilde(self.datum, m, mp)).simplify()
                out[tot] = term if tot not in out else (out[tot] + term).simplify()
        return WRQTorus(self.datum, out)

    def leading(self):
        """`(m_lead, f_{m_lead})` — the most cochar-dominant magnetic in the
        support and its residual (the bubbling-free leading Levi character)."""
        if not self._f:
            return None
        m_lead = max(self._f, key=lambda mm: (_cochar_height(self.datum, mm), mm))
        return (m_lead, self._f[m_lead])

    def recognize_q_extreme(self, at_infinity: bool = False):
        """The leading term of the 𝖖-power-series expansion of the residuals around
        𝖖=0 (`at_infinity=False`) or 𝖖=∞ (`True`): the bubbling-free leading Weyl
        orbit, decomposed into `{(m, e): int}`.  The W2 engine for `well_formed`.
        (𝖖=∞ is the `bar` image of 𝖖=0.)"""
        if at_infinity:
            return self.bar().recognize_q_extreme(False)
        datum, d = self.datum, self._d
        vals = {}
        for m, f in self._f.items():
            f = f.simplify()
            if f._num.is_zero():
                continue
            vnum = min(min(lp._coeffs) for lp in f._num._t.values())
            vden = sum(mult * (-k) for (a, k), mult in f._den.items() if k < 0)
            vals[m] = (vnum + vden, f, vnum)
        if not vals:
            return {}
        qext = min(v[0] for v in vals.values())
        sl_f = {}
        for m, (val, f, vnum) in vals.items():
            if val != qext:
                continue
            sign, shift, den_lead = 1, [0] * d, {}
            for (a, k), mult in f._den.items():
                for _ in range(mult):
                    if k > 0:                      # leading factor: 1 (𝖖^0, v^0)
                        pass
                    elif k == 0:                   # v-pole survives at 𝖖^0
                        den_lead[(a, 0)] = den_lead.get((a, 0), 0) + 1
                    else:                          # leading: −v^{−α} (𝖖^{−k})
                        for i in range(d):
                            shift[i] -= a[i]
                        sign = -sign
            acc = {}
            for ve, lp in f._num._t.items():
                if vnum in lp._coeffs:
                    nve = tuple(ve[i] + shift[i] for i in range(d))
                    acc[nve] = LaurentPoly({0: sign * lp._coeffs[vnum]})
            sl_f[m] = TorusRational(datum, TorusLaurent(datum, acc), den_lead)
        out = {}
        for label, c in WRQTorus(datum, sl_f).recognize_leading().items():
            vv = list(c._coeffs.values())
            out[label] = int(vv[0]) if len(vv) == 1 else int(sum(vv))
        return out

    def well_formed_w1(self):
        """W1 alone — bar-invariance (every residual q-palindromic).  The chart-bar
        verifier the `AbeKAlgebra` contract reads (`verify_chart_bar`)."""
        return self.bar() == self

    def well_formed(self):
        """The W1+W2 canonical certificate (trace-free): bar-invariance (W1) and a
        single leading Weyl orbit of multiplicity 1 (W2).  Returns the label `(m,e)`
        when well-formed, else `False`.  The Kazhdan–Lusztig acceptance for a
        residual-native build output (no Schur trace)."""
        if self.bar() != self:                     # (W1)
            return False
        r0 = self.recognize_q_extreme(False)        # (W2)
        if len(r0) != 1:
            return False
        (label, c), = r0.items()
        return label if c == 1 else False

    def orthonormality(self, other, K: int = 2):
        """`Tr(ρ(self)·other) = δ + O(𝖖)` — the q-series whose 𝖖⁰ is the Kronecker δ."""
        return self.inner(other, K)

    def recognize_leading(self):
        """Read the leading Weyl orbit: at the most cochar-dominant magnetic in the
        support (whose residual is bubbling-free), decompose the Levi character into
        canonical labels.  Returns `{(m, e): coeff}` — the inverse of
        `leading_orbit`, the seed-reader for bubbling-finding."""
        if not self._f:
            return {}
        datum = self.datum
        m_lead = max(self._f, key=lambda mm: (_cochar_height(datum, mm), mm))
        f = self._f[m_lead].simplify()
        if f._den:
            raise RuntimeError("recognize_leading: leading residual not polynomial "
                               "(not bubbling-free)")
        return {(m_lead, e): c for e, c in _levi_decompose(datum, m_lead, f._num)}

    def trace(self, K: int = 8, w_cutoff: bool = True):
        """The ρ²-twisted Schur trace — the Schur-measure residue of the
        magnetic-0 residual `f_0` (datum-general; see `trace_residual`)."""
        return trace_residual(self.datum, self.residual((0,) * self._d), K,
                              w_cutoff=w_cutoff)

    def inner(self, other, K: int = 8, w_cutoff: bool = True):
        """`⟨a,b⟩ = Tr(ρ(a)·b)` — the trace pairing (orthonormal on the canonical
        basis to `O(𝖖)`)."""
        return (self.rho() * other).trace(K, w_cutoff=w_cutoff)

    def _twist(self, inverse):
        out = {}
        for m, f in self._f.items():
            k = tuple(-x for x in m)
            cur = (f.vinv() * _rho_Gtilde(self.datum, m, inverse)).simplify()
            out[k] = (out[k] + cur).simplify() if k in out else cur
        return WRQTorus(self.datum, out)

    def rho(self):
        """`ρ(x)` — the √-measure conjugation: magnetic antipode `m → −m`,
        `v → 1/v`, and the G-cocycle `G̃_m`.  Datum-general (no baked _RHO)."""
        return self._twist(inverse=False)

    def rho_inverse(self):
        return self._twist(inverse=True)

    def bar(self):
        return WRQTorus(self.datum, {m: f.bar() for m, f in self._f.items()})

    def __eq__(self, other):
        if not isinstance(other, WRQTorus):
            return NotImplemented
        z = TorusRational.zero(self.datum)
        for m in set(self._f) | set(other._f):
            if not (self._f.get(m, z) - other._f.get(m, z)).simplify().is_zero():
                return False
        return True

    def __repr__(self):
        if not self._f:
            return "WRQTorus(0)"
        return "WRQTorus{" + ", ".join(f"{m}: {f}" for m, f in sorted(self._f.items())) + "}"


# ===========================================================================
# decompose / multiply-in-basis — structure constants (T2f)
# ===========================================================================
def _scale(w, C):
    """`C(𝖖)·w` for a scalar LaurentPoly `C` and a WRQTorus `w`."""
    s = TorusRational.from_scalar(w.datum, C)
    return WRQTorus(w.datum, {m: f * s for m, f in w.residuals().items()})


def is_minuscule(datum, m) -> bool:
    """Whether the cocharacter `m` is minuscule-type: `⟨α, m⟩ ∈ {0, ±1}` for
    every root α — exactly when `leading_orbit(m, e)` is bubbling-free, i.e.
    already the canonical.  (U(N): the fundamental coweights `φ_k + p·𝟙`,
    entries in `{p, p+1}`; su_2 has NO minuscule cocharacters — `⟨α,m⟩=2m`.)"""
    return all(abs(datum.shift_pairing(m, a)) <= 1 for a in datum.positive_roots())


def build_canonical(datum, m, e):
    """Dispatch to the residual-native canonical builder: bare (`e=0`) → the cone
    monomial; dressed minuscule (`m` minuscule, `e≠0`) → the dressed minuscule.
    Dressed NON-minuscule labels honest-fail here (their leading orbit is NOT
    the canonical — bubbling is missing); realisations reach them through the
    generic `peel_to_canonical` + `kl_bar_correct` (D10 wall (b))."""
    if all(x == 0 for x in e):
        return build_cone_monomial(datum, m)
    if not is_minuscule(datum, m):
        L = build_dressed_cone_monomial(datum, m, e)
        if L is not None:
            return L                          # a dressed cone monomial (in-span)
        raise NotImplementedError(
            f"build_canonical: dressed non-minuscule {m} — not a cone monomial "
            f"(asymmetric / Wilson-dressed); use the generic peel")
    return minuscule(datum, m, e)


def decompose(datum, x, build=build_canonical):
    """Decompose a bar-invariant WRQTorus element into the canonical basis:
    `x = Σ_c C_c(𝖖)·L_c`, returned as `{(m,e): C_c}`.  Top-down peel — at each step
    `recognize_leading` reads the canonicals at the most cochar-dominant magnetic
    (with their full 𝖖-coefficients), then subtracts `C_c·L_c`.  The structure-
    constant engine for multiply-in-basis (the `AbeKAlgebra` level-ascending read,
    no solve)."""
    out = {}
    guard = 0
    while not x.is_zero():
        guard += 1
        if guard > 2000:
            raise RuntimeError("decompose: no termination (off-scope element?)")
        leading = x.recognize_leading()
        if not leading:
            break
        for (m, e), C in leading.items():
            out[(m, e)] = out.get((m, e), LaurentPoly.zero()) + C
            x = x + _scale(build(datum, m, e), -C)
    return {lab: C for lab, C in out.items() if not C.is_zero()}


class PureUNWRQ:
    """The native pure-U(N) build/decompose on WRQTorus over a `u_n` datum —
    the shared engine (extracted from the retired per-tier WRQ classes so the
    keystone `PureUNKAlgebra` and the matter/quiver decompose all use it).

    `chart(m, e)` builds `L_{(m,e)}` (cone monomial → dressed cone monomial →
    family-4b dressed-monopole·cone peel → the two generic seeds), cached with a
    cycle guard; `decompose(x)` reads a WRQTorus element in that basis."""

    def __init__(self, datum, max_peel: int = 400):
        self.datum = datum
        self.N = datum.dim
        self._max_peel = max_peel
        self._cache: dict = {}
        self._building: set = set()

    def _fundamental_factor(self, m):
        for k in range(1, self.N):
            if m[k - 1] - m[k] >= 1:
                return tuple(1 if i < k else 0 for i in range(self.N))
        return None

    def _seed_charged(self, m, e):
        phi = self._fundamental_factor(m)
        if phi is None:
            raise NotImplementedError(f"no fundamental factor in {m}")
        m2 = tuple(m[i] - phi[i] for i in range(self.N))
        Wm2, Phim2 = _levi_data(self.datum, m2)
        e2 = _levi_dom_rep(self.datum, m2, e, Wm2, Phim2)
        return minuscule(self.datum, phi, (0,) * self.N) * self.chart(m2, e2)

    def _seed_wilson_cone(self, m, e):
        return wilson(self.datum, tuple(sorted(e, reverse=True))) * \
            build_cone_monomial(self.datum, m)

    def _build(self, m, e):
        if is_minuscule(self.datum, m) or all(x == 0 for x in e):
            return build_canonical(self.datum, m, e)
        cone = build_dressed_cone_monomial(self.datum, m, e)
        if cone is not None:
            return cone
        key = (m, e)
        if key in self._building:
            raise NotImplementedError(f"PureUNWRQ: cyclic dependency at L_{key}")
        self._building.add(key)
        try:
            mono = build_monopole_cone_peel(
                self.datum, m, e, lambda mt, et: self.chart(mt, et), self._max_peel)
            if mono is not None:
                return mono
            for seed_fn in (self._seed_charged, self._seed_wilson_cone):
                try:
                    L = peel_to_canonical(self.datum, (m, e), seed_fn(m, e),
                                          lambda mt, et: self.chart(mt, et),
                                          self._max_peel)
                    L = kl_bar_correct(self.datum, L, (m, e),
                                       lambda mt, et: self.chart(mt, et),
                                       self._max_peel)
                    if L.well_formed() == (tuple(m), tuple(e)):
                        return L
                except (NotImplementedError, RuntimeError):
                    pass
        finally:
            self._building.discard(key)
        raise NotImplementedError(f"PureUNWRQ: no seed certifies L_{key}")

    def chart(self, m, e):
        key = (tuple(m), tuple(e))
        if key not in self._cache:
            self._cache[key] = self._build(key[0], key[1])
        return self._cache[key]

    def decompose(self, x) -> dict:
        return decompose(self.datum, x, build=lambda dat, m, e: self.chart(m, e))


def multiply_in_basis(datum, a, b, build=build_canonical):
    """Structure constants `L_a·L_b = Σ_c C^c_{ab}(𝖖) L_c`: build both canonicals,
    multiply on WRQTorus, decompose.  Labels `a=(m,e)`, `b=(m',e')`."""
    P = build(datum, *a) * build(datum, *b)
    return decompose(datum, P, build)


# ===========================================================================
# Generic native build — product-and-peel + KL bar-correction (D10 wall (b))
#
# The datum-general transcription of the `PureSU2KAlgebra` (#765) native build
# (itself the SU-torus transcription of the U(N) keystone `cf_build_full`):
# dress a WELL-FOUNDED seed product (e.g. `H·L_{m−1,e}` — magnetically charged,
# coupling the target only to strictly-lower/other canonicals) into the
# canonical `L_target`, using only datum-general operations.  A realisation
# supplies `lower_chart(m, e)` — the already-built canonical for a recognized
# non-target label (its own cache/recursion; it must honest-fail on cycles).
# Never an in-presentation solve (constructive-build rule): the peel subtracts known
# canonicals read off by `recognize_leading` (bubbling-free leadings only), and
# the KL bar-correction subtracts the strictly-negative-q part of the
# bar-defect `L − bar(L)` (anti-palindromic, in the span of lower canonicals) —
# the unique bar-invariant completion with the same bar-fixed leading.
# Acceptance stays `well_formed` equality with the intended label (the caller's
# certification duty).
# ===========================================================================
_NEG_ONE = LaurentPoly({0: -1})


def _bare_lead_coeff(x, m, e):
    """The q-coefficient of `x` at (atom `m`, weight `e`), read off the
    simplified residual's numerator (zero if the atom is absent)."""
    f = x.residual(tuple(m)).simplify()
    return f._num._t.get(tuple(e), LaurentPoly.zero())


def _unit_lp_inverse(lp, what: str) -> LaurentPoly:
    """Inverse of a unit q-monomial `±q^k` (honest-fail otherwise)."""
    coeffs = lp._coeffs
    if len(coeffs) != 1:
        raise NotImplementedError(f"{what}: coefficient {lp} not a monomial")
    (p, z), = coeffs.items()
    if z not in (1, -1):
        raise NotImplementedError(f"{what}: coefficient {lp} not ±q^k")
    return LaurentPoly({-p: z})


def peel_to_canonical(datum, target, seed, lower_chart, max_peel: int = 400):
    """Strip the target's bar-fixed leading orbit from `seed`, subtract every
    full lower canonical `recognize_leading` identifies (re-recognizing after
    each subtraction), and normalize the leading coefficient to 1.  Returns the
    single-ordering peel result — generally still bar-NON-invariant; follow
    with `kl_bar_correct`.  A pole-carrying leading in the remainder is the
    target's own bubbling tail (it stays in `L`)."""
    m, e = tuple(target[0]), tuple(target[1])
    fac = _bare_lead_coeff(seed, m, e)
    if fac.is_zero():
        raise NotImplementedError(
            f"peel_to_canonical: seed has no bare leading at {(m, e)}")
    P = seed + _scale(leading_orbit(datum, m, e), fac * _NEG_ONE)
    L = seed
    for _ in range(max_peel):
        if P.is_zero():
            break
        try:
            rec = P.recognize_leading()
        except RuntimeError:
            break               # pole-carrying leading → only the target's tail
        progressed = False
        for (mt, et), C in sorted(rec.items()):
            if (tuple(mt), tuple(et)) == (m, e):
                continue        # the target's own orbit — stays in L
            Lc = lower_chart(mt, et)
            c0 = Lc.recognize_leading().get((tuple(mt), tuple(et)))
            if c0 is None:
                raise NotImplementedError(
                    f"peel_to_canonical: lower chart at {(mt, et)} does not "
                    f"lead at its own label")
            Ct = C * _unit_lp_inverse(c0, "peel_to_canonical")
            P = P + _scale(Lc, Ct * _NEG_ONE)
            L = L + _scale(Lc, Ct * _NEG_ONE)
            progressed = True
            break               # re-recognize after each subtraction
        if not progressed:
            break
    lead = _bare_lead_coeff(L, m, e)
    return _scale(L, _unit_lp_inverse(lead, "peel_to_canonical: leading"))


def kl_bar_correct(datum, L, target, lower_chart, max_iter: int = 400):
    """The KL bar-correction: while `L` is bar-non-invariant, read the leading
    of the anti-palindromic defect `D = L − bar(L)`, and subtract the
    strictly-negative-q part of its coefficient times the corresponding lower
    canonical.  Honest-fails (returns `L` unchanged for `well_formed` to
    reject) if the defect leads on the target's own orbit or is unreadable."""
    m, e = tuple(target[0]), tuple(target[1])
    for _ in range(max_iter):
        if L.bar() == L:
            return L
        D = L + _scale(L.bar(), _NEG_ONE)
        if D.is_zero():
            return L
        try:
            rec = D.recognize_leading()
        except RuntimeError:
            return L
        (mt, et), C = sorted(rec.items(), reverse=True)[0]
        if (tuple(mt), tuple(et)) == (m, e):
            return L
        Lc = lower_chart(mt, et)
        c0 = Lc.recognize_leading().get((tuple(mt), tuple(et)))
        if c0 is None:
            return L
        dc = C * _unit_lp_inverse(c0, "kl_bar_correct")
        pc = LaurentPoly({p: z for p, z in dc._coeffs.items() if p < 0})
        if pc.is_zero():
            return L
        L = L + _scale(Lc, pc * _NEG_ONE)
    return L


# ===========================================================================
# Levi layer — leading Weyl orbits of L_{m,e} and their inverse read (T2e)
# ===========================================================================
def _levi_data(datum, m):
    """`(W_m, Φ_m⁺)` for the Levi `L_m` = centralizer of the cocharacter m:
    `W_m = Stab_W(m)` (cochar stabilizer) and `Φ_m⁺ = {α∈Φ⁺ : ⟨α,m⟩=0}`."""
    m = tuple(m)
    Wm = [w for w in datum.weyl_elements() if datum.act_cochar(w, m) == m]
    Phim = [a for a in datum.positive_roots() if datum.shift_pairing(m, a) == 0]
    return Wm, Phim


def levi_character(datum, m, e):
    """The Levi-irrep character `χ_e[L_m]` (dominant weight e) as a TorusLaurent,
    via the Weyl character formula over `W_m`:

        χ_e = (Σ_{w∈W_m} sgn(w) v^{w·e + (w·ρ_m − ρ_m)}) / ∏_{α∈Φ_m⁺}(1 − v^{−α}),

    with `w·ρ_m − ρ_m = ½(Σ_{α∈Φ_m⁺} w·α − Σ α)` an integer vector (no half-integer
    ρ_m).  Regular m ⇒ Φ_m empty ⇒ χ_e = v^e.  For type-A Levi blocks this equals
    the product-of-Schur `pure_un_closed_form.levi_character`."""
    Wm, Phim = _levi_data(datum, m)
    d = datum.dim
    e = tuple(e)
    sum_pos = [sum(a[i] for a in Phim) for i in range(d)]      # Σ_{Φ_m⁺} α
    num = {}
    for w in Wm:
        we = datum.act(w, e)
        sw = [0] * d
        for a in Phim:
            wa = datum.act(w, a)
            for i in range(d):
                sw[i] += wa[i]
        wt = tuple(we[i] + (sw[i] - sum_pos[i]) // 2 for i in range(d))
        num[wt] = num.get(wt, 0) + datum.sign(w)
    terms = {wt: LaurentPoly({0: c}) for wt, c in num.items() if c}
    for a in Phim:                                             # ÷ ∏(1 − v^{−α})
        terms = _divide_by_factor(terms, tuple(-x for x in a), 0, d)
        if terms is None:
            raise RuntimeError("levi_character: Weyl-denominator division failed")
    return TorusLaurent(datum, terms)


def wilson(datum, e):
    """The Wilson-line generator `L_W(e)` = the full-group character `χ_e` at
    magnetic 0, built RESIDUAL-NATIVE (no DOp): `leading_orbit(datum, 0, e)`,
    whose only residual is `χ_e(v)` at `m=0` (q-free ⇒ bar-invariant ⇒ canonical).
    A generator constructed with no `u`'s at all."""
    return leading_orbit(datum, (0,) * datum.dim, e)


def det(datum, p):
    """The central-monopole generator `det^p` = the canonical at the central
    magnetic `p·𝟙` (all entries equal), built residual-native: `leading_orbit` at
    `(p,…,p)` with trivial dressing (q-free ⇒ canonical, no bubbling)."""
    return leading_orbit(datum, (p,) * datum.dim, (0,) * datum.dim)


def minuscule(datum, m, e):
    """The dressed-minuscule generator `L_{m,e}` for a minuscule cocharacter `m`:
    `leading_orbit(datum, m, e)` (for minuscule `m` the canonical IS its leading
    orbit — no bubbling; verified vs the closed-form engine)."""
    return leading_orbit(datum, tuple(m), tuple(e))


def _un_fundamental_coweights(N, m_dom):
    """U(N) decomposition of a dominant cocharacter into fundamental coweights:
    `m = Σ_k c_k φ_k + p·𝟙`, `φ_k = (1^k, 0^{N-k})`, `c_k = m_k − m_{k+1}`,
    `p = m_N`.  Returns `(list_of_φ_k_with_multiplicity, p)`."""
    m = list(m_dom)
    p = m[-1]
    factors = []
    for k in range(1, N):
        for _ in range(m[k - 1] - m[k]):
            factors.append(tuple(1 if i < k else 0 for i in range(N)))
    return factors, p


def build_cone_monomial(datum, m_dom):
    """The bare canonical `L_{m,0}` (U(N) cone monomial) built RESIDUAL-NATIVE as a
    product of minuscule generators × `det^p` — `∏_k minuscule(φ_k)^{c_k} · det^p` —
    multiplied via `WRQTorus.multiply`.  The bubbling emerges from the cocycle
    product; the result is `well_formed`-certified.  No DOp.  (The φ_k decomposition
    is the type-A recipe; the generic build = multiply leading-orbit generators.)"""
    N = datum.dim
    factors, p = _un_fundamental_coweights(N, tuple(m_dom))
    result = None
    for phi in factors:
        gen = minuscule(datum, phi, (0,) * N)
        result = gen if result is None else (result * gen)
    if p != 0:
        dg = det(datum, p)
        result = dg if result is None else (result * dg)
    return result if result is not None else leading_orbit(datum, (0,) * N, (0,) * N)


# ===========================================================================
# Dressed cone monomials — the type-A `_cone_build` recipe, residual-native.
#
# A DRESSED cone monomial `L_{(m,e)}` (non-minuscule `m`, `e ≠ 0`) is a
# q-commuting product of QTCone RAYS — per fundamental-coweight direction `k` a
# `low`/`mut` dressed minuscule of `φ_k = (1^k, 0^{N-k})` — times `det^p`, slid
# to the target dressing by the two free moves and bar-centered.  This is the
# residual-native transcription of `pure_un_kalgebra._cone_build_impl` into the
# dominant WRQ frame (the keystone works anti-dominant).  The `φ_n` dressing
# slide `φ_n(L_{(φ_k,d)}) = L_{(φ_k, d+n·φ_k)}` (exact, q-free) is FOLDED into
# the ray dressings (so no residual-level `φ_n` port is needed); the `w2^l`
# slide is the central Wilson `(∏v)^l = wilson(l·𝟙)`.  Every candidate is a
# product of canonical generators, so acceptance (`well_formed() == (m, e)`) is
# IN-SPAN by construction — never a solve.  `None` off the cone-monomial scope
# (asymmetric / Wilson-dressed labels fall through to the generic peel).
# ===========================================================================
def _qexps(X):
    """All 𝖖-exponents appearing in `X`'s (simplified) residual numerators."""
    out = []
    for res in X.residuals().values():
        for lp in res.simplify().num.terms.values():
            out.extend(lp._coeffs.keys())
    return out


def _bar_center(X):
    """The bar-invariant `𝖖^{-k}·X` (a genuine product `X = 𝖖^k·L` is a single
    canonical up to this scalar), or `None`.  The window is read off the
    numerator 𝖖-extent (fast) and widened once on a miss."""
    if X.is_zero():
        return None
    ex = _qexps(X)
    if not ex:
        return None
    k0 = (min(ex) + max(ex)) // 2
    for lo, hi in ((k0 - 6, k0 + 6), (k0 - 40, k0 + 40)):
        for k in range(lo, hi + 1):
            Y = _scale(X, LaurentPoly({-k: 1}))
            if not Y.is_zero() and Y.bar() == Y:
                return Y
    return None


def _cone_slide_nl(diff, m, N):
    """Solve `diff = n·m + l·𝟙` in integers `(n, l)` (closed form), or `None`."""
    if len(set(m)) == 1:                          # central m: n is free ⇒ take 0
        return (0, diff[0]) if len(set(diff)) == 1 else None
    i = next(t for t in range(N) if m[t] != m[0])
    dm = m[i] - m[0]
    if (diff[i] - diff[0]) % dm != 0:
        return None
    n = (diff[i] - diff[0]) // dm
    l = diff[0] - n * m[0]
    if tuple(n * m[t] + l for t in range(N)) != tuple(diff):
        return None
    return n, l


def _cone_box_product(datum, m, c, p, bs, n):
    """The box product `∏_k ray_k · det^p` with the `φ_n` slide folded into every
    ray (and into `det^p` via a central Wilson)."""
    N = datum.dim
    prod = None
    for k in range(1, N):
        ck = c[k - 1]; bk = bs[k - 1]
        pk = tuple(1 if i < k else 0 for i in range(N))     # φ_k
        dlow = tuple((N - k) if i < k else 0 for i in range(N))
        dmut = tuple(dlow[i] - pk[i] for i in range(N))
        for _ in range(ck - bk):                            # low rays
            d = tuple(dlow[i] + n * pk[i] for i in range(N))
            g = minuscule(datum, pk, d)
            prod = g if prod is None else prod * g
        for _ in range(bk):                                 # mut rays
            d = tuple(dmut[i] + n * pk[i] for i in range(N))
            g = minuscule(datum, pk, d)
            prod = g if prod is None else prod * g
    if p != 0:
        dg = det(datum, p)
        if n != 0:                                          # φ_n(det^p): +n·p·𝟙
            dg = dg * wilson(datum, (n * p,) * N)
        prod = dg if prod is None else prod * dg
    return prod


def build_dressed_cone_monomial(datum, m, e):
    """The dressed cone monomial `L_{(m,e)}` (dominant `m`) as a `WRQTorus`
    element, or `None` if `(m,e)` is not a cone monomial.  In-span product build;
    the acceptance is `well_formed() == (m,e)` (never a solve)."""
    N = datum.dim
    m = tuple(m); e = tuple(e)
    c = [m[k] - m[k + 1] for k in range(N - 1)]
    p = m[N - 1]
    if any(x < 0 for x in c):
        return None
    for bs in itertools.product(*[range(ck + 1) for ck in c]):
        base = _cone_box_product(datum, m, c, p, bs, 0)      # n=0 box product
        Lbase = _bar_center(base) if base is not None else None
        if Lbase is None:
            continue
        try:
            wb = Lbase.well_formed()
        except (NotImplementedError, RuntimeError):
            continue
        if wb[0] != m:
            continue
        nl = _cone_slide_nl(tuple(e[i] - wb[1][i] for i in range(N)), m, N)
        if nl is None:
            continue
        n, l = nl
        cand = _cone_box_product(datum, m, c, p, bs, n)      # rebuild with φ_n
        if cand is None:
            continue
        if l != 0:
            cand = cand * wilson(datum, (l,) * N)
        L = _bar_center(cand)
        if L is None:
            continue
        try:
            if L.well_formed() == (m, e):
                return L
        except (NotImplementedError, RuntimeError):
            continue
    return None


def _levi_dominant_dressing(phik, f):
    """Whether `f` is Levi-dominant for `φ_k = (1^k, 0^{N-k})` (each U(k), U(N-k)
    block non-increasing) — the condition for `minuscule(φ_k, f)` to build."""
    k = sum(phik)
    return (list(f[:k]) == sorted(f[:k], reverse=True) and
            list(f[k:]) == sorted(f[k:], reverse=True))


def build_monopole_cone_peel(datum, m, e, lower_chart, max_peel=400):
    """A `_cone_build`-MISS dressed non-minuscule `L_{(m,e)}` via the keystone's
    family-4b **dressed-monopole·cone** peel: `minuscule(φ_k, f)·cone(m', e')`
    with `m' = m − φ_k` (strictly-lower magnetic).  The WRQ leading map is
    additive — `leading(minuscule(φ_k,f)·cone(m',e')) = f + e'` — so `f = e − e'`
    is determined (no search); candidates are tried CHEAPEST-`f`-first, which
    reaches the well-founded case (small `f` ⇒ strictly-lower co-summands, a
    convergent single-ordering peel) before the ill-founded same-magnetic ones.
    In-span product build; `None` off scope.  `lower_chart(mt,et)` is the caller's
    (cycle-guarded) canonical builder."""
    N = datum.dim
    m = tuple(m); e = tuple(e)
    lo = min(0, min(e)); hi = max(0, max(e))
    cands = []
    for k in range(1, N):
        if m[k - 1] - m[k] <= 0:                    # need a φ_k factor (c_k > 0)
            continue
        phik = tuple(1 if i < k else 0 for i in range(N))
        mp = tuple(sorted((m[i] - phik[i] for i in range(N)), reverse=True))
        for ep in itertools.product(range(lo - 1, hi + 2), repeat=N):
            f = tuple(e[i] - ep[i] for i in range(N))
            if not _levi_dominant_dressing(phik, f):
                continue
            cands.append((sum(abs(x) for x in f) + sum(abs(x) for x in ep),
                          k, phik, mp, ep, f))
    cands.sort(key=lambda t: t[0])
    tried = 0
    for _, k, phik, mp, ep, f in cands:
        try:
            base = build_canonical(datum, mp, ep)   # cone dressing at m'
        except (NotImplementedError, RuntimeError):
            continue
        if base is None:
            continue
        try:
            seed = minuscule(datum, phik, f) * base
            if (m, e) not in seed.recognize_leading():
                continue                            # seed must lead at the target
        except (NotImplementedError, RuntimeError):
            continue
        tried += 1
        if tried > _MONOPOLE_CONE_MAXTRY:
            break
        try:
            L = peel_to_canonical(datum, (m, e), seed, lower_chart, max_peel)
            L = kl_bar_correct(datum, L, (m, e), lower_chart, max_peel)
            if L.well_formed() == (m, e):
                return L
        except (NotImplementedError, RuntimeError):
            continue
    return None


_MONOPOLE_CONE_MAXTRY = 24


def leading_orbit(datum, m, e):
    """The leading Weyl orbit of the canonical `L_{m,e}`:
    `Σ_{w∈W/W_m} χ_e(w·v)·U_{w·m}` — a WRQTorus element (the bar-non-invariant
    seed bubbling-finding dresses into the canonical).  `m` cochar-dominant, `e`
    a Levi-dominant weight; the residual at magnetic `w·m` is `w·χ_e`."""
    chi = levi_character(datum, m, e)
    f = {}
    for w in datum.weyl_elements():
        mw = datum.act_cochar(w, m)
        if mw in f:                       # dedup W/W_m (χ_e is W_m-invariant)
            continue
        f[mw] = TorusRational.from_laurent(chi.weyl_act(w))
    return WRQTorus(datum, f)


def _levi_dom_rep(datum, m, wt, Wm, Phim):
    """The Levi-dominant representative of weight `wt` in its `W_m` orbit."""
    for w in Wm:
        x = datum.act(w, wt)
        if all(sum(a[i] * x[i] for i in range(datum.dim)) >= 0 for a in Phim):
            return x
    return tuple(wt)


def _levi_height(datum, wt, Phim):
    return sum(sum(a[i] * wt[i] for i in range(datum.dim)) for a in Phim)


def _levi_decompose(datum, m, P):
    """Decompose a `W_m`-symmetric TorusLaurent `P` into Levi characters: returns
    `[(e, coeff)]` (inverse of `levi_character`), via a top-Levi-weight peel."""
    Wm, Phim = _levi_data(datum, m)
    work = {wt: dict(lp._coeffs) for wt, lp in P._t.items() if not lp.is_zero()}
    out = []
    guard = 0
    while any(any(v for v in lp.values()) for lp in work.values()):
        guard += 1
        if guard > 5000:
            raise RuntimeError("_levi_decompose: no termination")
        best = None
        for wt, lp in work.items():
            if not any(lp.values()):
                continue
            r = _levi_dom_rep(datum, m, wt, Wm, Phim)
            key = (_levi_height(datum, r, Phim), r)
            if best is None or key > best[0]:
                best = (key, r)
        e = best[1]
        c = dict(work.get(e, {}))
        if not any(c.values()):
            raise RuntimeError("_levi_decompose: dominant rep absent (not W_m-symmetric?)")
        out.append((e, LaurentPoly(c)))
        chi = levi_character(datum, m, e)
        for wt, lp in chi._t.items():
            for eq, cc in lp._coeffs.items():
                for ce, cv in c.items():
                    cur = work.setdefault(wt, {})
                    cur[eq + ce] = cur.get(eq + ce, 0) - cc * cv
        work = {wt: {q: v for q, v in lp.items() if v} for wt, lp in work.items()}
        work = {wt: lp for wt, lp in work.items() if lp}
    return out


def _cochar_height(datum, m):
    return sum(datum.shift_pairing(m, a) for a in datum.positive_roots())


# ===========================================================================
# trace — the Schur-measure residue (datum-general), T2d
# ===========================================================================
def _trunc_lp(lp: LaurentPoly, K: int) -> LaurentPoly:
    return LaurentPoly({e: c for e, c in lp._coeffs.items() if e <= K})


def _trunc_tl(tl: TorusLaurent, K: int) -> TorusLaurent:
    return TorusLaurent(tl.datum, {e: _trunc_lp(lp, K) for e, lp in tl._t.items()})


def _qpoch_q2(K: int) -> LaurentPoly:
    """`(𝖖²;𝖖²)_∞ = ∏_{l≥1}(1 − 𝖖^{2l})` to 𝖖^K."""
    r = LaurentPoly({0: 1})
    for l in range(1, K // 2 + 1):
        r = _trunc_lp(r * LaurentPoly({0: 1, 2 * l: -1}), K)
    return r


def _inv_qpoch_n(n: int, K: int) -> LaurentPoly:
    """`1/(𝖖²;𝖖²)_n` to 𝖖^K."""
    r = LaurentPoly({0: 1})
    for l in range(1, n + 1):
        geom = LaurentPoly({2 * l * t: 1 for t in range(K // (2 * l) + 1)})
        r = _trunc_lp(r * geom, K)
    return r


def _schur_measure(datum, K: int) -> TorusLaurent:
    """The Schur/Weyl measure `∏_{α∈Φ} (v^α;𝖖²)_∞ (𝖖²v^α;𝖖²)_∞` to 𝖖^K, via the
    Euler expansion `(𝖖^k v^α;𝖖²)_∞ = Σ_n (−1)^n 𝖖^{n²+(k−1)n} v^{nα}/(𝖖²;𝖖²)_n`."""
    res = TorusLaurent.one(datum)
    for a in datum.roots():
        for kk in (0, 2):
            terms = {}
            n = 0
            while n * n + (kk - 1) * n <= K:
                qpow = n * n + (kk - 1) * n
                coeff = _inv_qpoch_n(n, K - qpow)
                sgn = -1 if n % 2 else 1
                lp = LaurentPoly({qpow + e: sgn * c for e, c in coeff._coeffs.items()})
                wt = tuple(n * x for x in a)
                terms[wt] = (terms.get(wt, LaurentPoly.zero()) + lp)
                n += 1
            res = _trunc_tl(res * TorusLaurent(datum, terms), K)
    return res


def _factor_inv_series(datum, a, k, K: int) -> TorusLaurent:
    """`1/(1 − 𝖖^k v^α)` as a 𝖖-series to 𝖖^K (k ≠ 0): Σ_{n≥0}𝖖^{kn}v^{nα} for
    k>0, and `−Σ_{n≥1}𝖖^{|k|n}v^{−nα}` for k<0."""
    terms = {}
    if k > 0:
        n = 0
        while k * n <= K:
            terms[tuple(n * x for x in a)] = LaurentPoly({k * n: 1})
            n += 1
    else:
        ak = -k
        n = 1
        while ak * n <= K:
            terms[tuple(-n * x for x in a)] = LaurentPoly({ak * n: -1})
            n += 1
    return TorusLaurent(datum, terms)


def _v0_coeff(s: TorusLaurent, measure: TorusLaurent, K: int) -> LaurentPoly:
    """`[v^0](s·measure) = Σ_w s[w]·measure[−w]`, truncated to 𝖖^K (only v^0 needed)."""
    out = LaurentPoly.zero()
    M = measure._t
    for w, sc in s._t.items():
        mc = M.get(tuple(-x for x in w))
        if mc is not None:
            out = _trunc_lp(out + sc * mc, K)
    return out


def trace_residual(datum, f0: TorusRational, K: int = 8,
                   w_cutoff: bool = True) -> LaurentPoly:
    """`Tr = (𝖖²;𝖖²)_∞^{2·dim}/|W| · [v^0]( measure · f_0 )`, the Schur-measure
    residue of the magnetic-0 residual `f_0`.  k=0 denominator factors `(1−v^α)`
    cancel against the measure — first tried as exact numerator division (the
    U(N) fast path), else folded into the measure itself, which carries an
    order-2 zero per root pair at `v^α = 1` (`(v^α;𝖖²)_∞ = (1−v^α)·(𝖖²v^α;𝖖²)_∞`
    and `(1−v^{−α}) = −v^{−α}(1−v^α)`) — the datum-general prescription (D10
    wall (a); values unchanged where the fast path succeeds, since both compute
    the same `[v^0](measure·f_0)`).  k≠0 factors are 𝖖-expanded.  The
    `|W|`-divisibility cutoff marks the reliable 𝖖-order — NOTE it makes the
    result non-linear in `f0` (pieces of a decomposition may cut at different
    orders).  `w_cutoff=False` skips the division entirely and returns the raw
    UNDIVIDED series `[v^0](measure·f_0)·pref` (= `|W|·Tr`), which is exact and
    linear in `f0` — diagnostics/tests only."""
    f0 = f0.simplify()
    if f0.is_zero():
        return LaurentPoly.zero()
    num = f0._num
    nz = []
    k0 = []          # k=0 factors not dividing the numerator: cancel measure-side
    for (a, k), mult in f0._den.items():
        for _ in range(mult):
            if k == 0:
                q = _divide_by_factor(num._t, a, 0, datum.dim)
                if q is None:
                    k0.append(a)
                else:
                    num = TorusLaurent(datum, q)
            else:
                nz.append((a, k))
    # Internal margin: `num` may carry negative 𝖖-powers (ρ-twist factors), and
    # `[v^0]` at order t pairs them against measure/expansion content at
    # t + |negative depth| — without the pad the top `pad` orders of the result
    # are silently incomplete (and the |W| marker then mis-fires both ways).
    val = min((min(lp._coeffs) for lp in num._t.values() if lp._coeffs),
              default=0)
    pad = max(0, -val)
    Kint = K + pad
    s = num
    for a, k in nz:
        s = _trunc_tl(s * _factor_inv_series(datum, a, k, Kint), Kint)
    measure = _schur_measure(datum, Kint)
    for a in k0:
        qd = _divide_by_factor(measure._t, a, 0, datum.dim)
        if qd is None:
            raise RuntimeError(
                f"trace: (1−v^{a}) pole beyond the measure's zero (order > 2)")
        measure = TorusLaurent(datum, qd)
    v0 = _v0_coeff(s, measure, Kint)
    if v0.is_zero():
        return LaurentPoly.zero()
    pref = LaurentPoly({0: 1})
    poch = _qpoch_q2(Kint)
    for _ in range(2 * datum.dim):
        pref = _trunc_lp(pref * poch, Kint)
    qs = _trunc_lp(v0 * pref, K)
    if not w_cutoff:                     # raw mode: |W|·Tr, exact and linear
        return qs
    W = len(datum.weyl)
    out = {}
    for e in sorted(qs._coeffs):
        qd, r = divmod(qs._coeffs[e], W)
        if r != 0:                       # incomplete Weyl average ⇒ cutoff reached
            break
        if qd:
            out[e] = qd
    return LaurentPoly(out)
