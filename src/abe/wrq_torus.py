"""`wrq_torus` — WRQTorus (Weyl Rational Quantum Torus), the group-general
enriched-torus substrate, Layer 2 of the `AbeKAlgebra` redesign.

This generalizes `urq_torus.URQTorus` (U(N)-specific) to any `RootDatum`, built
on the Layer-1 ring `weyl_torus_ring.WeylTorusRing`.  Elements are residual
vectors `(f_m)` over the cocharacter lattice (`m ∈ X_*(T)`, dual to the weight
lattice), `x = Σ_m f_m(𝖖^m v)·U_m`, with the atoms `U_m` carrying a single
magnetic charge `m` and the dressing `ψ_m`.

Representation policy
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
    products — the su_2 finding (2026-07-01); see `RootDatum.atom_phase`.

    NOTE (public language): the tier speaks **atoms
    `U_m`, weight monomials `v^e`, residuals, and the cocycle
    `U_a U_b = R_{a,b}(v)·U_{a+b}`** only.  The `ψ` of `dressing_psi` /
    `dressing_psi_inv` below is a PRIVATE derivational device for producing
    `R_{a,b}` and the ρ-twist `G_m` — no relation `U = ψ·u` (and no `u`'s at
    all) appears on any public surface; if/when 3d boundary/interface
    questions need a difference-operator export, that is a separate, future
    surface (`abelianized_torus`, demoted by D9).
  * `dressing_psi_inv(datum, m)` — `1/ψ_m` (a Laurent times a monomial).
  * **`CC(datum, a, b)` — the cocycle `CC_{a,b}`, THE PRIMITIVE**.  It is
    what the product law is written in, it is defined by a closed form with no
    `ψ` in it, and it — not `R` — satisfies the bar axiom at torus level,
    `bar(CC_{a,b}) = CC_{b,a}` (measured; `R` fails it on most charge pairs).

    A product over the positive roots, supported exactly on those where `⟨a,α⟩`
    and `⟨b,α⟩` have **opposite signs**.  With `A = ⟨a,α⟩`, `B = ⟨b,α⟩`,
    `z = v^α`, and the **Clebsch–Gordan range** `d = ||A|−|B||`, `D = |A|+|B|`:

        𝖖^{|A||B|} · z^{min(|A|,|B|)}
        ────────────────────────────────────────────────────────
        (1 − 𝖖^d z)·∏_{k=d+2,…,D−2}(1 − 𝖖^k z)²·(1 − 𝖖^D z)

    — the CLOSED range, ends once and interior twice; `A < 0 < B` is the bar
    image.  **`CC[N]`, gauge and matter together, is
    `matter_wrq_torus.CC_N`**; the matter-only piece is the ratio
    `CC[N]/CC[0]`, the exact analogue on the same range with its NUMERATOR
    factors spanning the STRICT INTERIOR, once each
    (`matter_wrq_torus.matter_factor_exponents`).

  * `cocycle_R(datum, a, b)` — **derived**, `R = T_{a+b}(CC)`; the cocycle of
    `U_a U_b = R_{a,b}(v)·U_{a+b}`, with the bare argument because the atom is
    `ψ_m(v)·u^m` (dressing on the left — measured, not assumed).
  * `cocycle_R_via_psi(datum, a, b)` — the `ψ`-built route,
    `R_{a,b} = ψ_a · T_{2a}(ψ_b) · ψ_{a+b}^{-1}` with the full normal-ordering
    shift `T_p : v^λ ↦ 𝖖^{⟨p,λ⟩} v^λ` (`q_shift`), here at `p = 2a`.  Kept as the
    **trivialization** (`δψ = R`) and no longer the definition.

Dominant `m` (`⟨α,m⟩ ≥ 0 ∀α>0`) uses the closed form; non-dominant `m` is the
Weyl transport `ψ_{w·m_dom} = w·ψ_{m_dom}`.  The multiply / ρ / trace / inner /
recognize layers build on this (subsequent increments).
"""
from __future__ import annotations

import itertools
from fractions import Fraction

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


def _root_pairing_count(datum, a, m) -> int:
    """`⟨α, m⟩` as a Python **int**, for use as a loop bound in the ψ-dressing.

    Why this exists: a cocharacter of a **non-simply-connected** global form lives
    in the coweight lattice `P^∨ ⊋ Q^∨`, so its coordinates are `Fraction`s in the
    coroot basis — and then `⟨α, m⟩` comes back as a `Fraction` even when its value
    is a plain integer, which `range()` rejects on *type* alone.  Since `P^∨` is by
    definition the dual of the root lattice, `⟨α, m⟩ ∈ Z` for every root `α`
    whenever `m ∈ P^∨`, so the coercion below is always legitimate there and this
    is the only thing that stood between the ψ-dressing and the exotic forms.

    Honest-fails when the value is genuinely non-integral: that means `m ∉ P^∨`,
    the ψ-dressing product `∏_{l<⟨α,m⟩}` has no meaning, and no coercion can
    manufacture one.  (Distinct from the *other* boundary — an `m ∈ P^∨` with odd
    `⟨Σ⁺, m⟩`, where the dressing is fine but the *standalone monomial* `M(m)`
    wants `𝖖^{⟨ρ,m⟩}`.  That one is **not an obstruction**: the algebra needs
    the phase only through the integral coboundary `δS̃`, and `cocycle_R` restores
    the honest phase, so those charges build.  Either way, not a failure of this
    coercion.)"""
    v = _pairing_root_cochar(datum, a, m)
    iv = int(v)
    if iv != v:
        raise NotImplementedError(
            f"{datum.name}: ⟨α, m⟩ = {v} is not an integer at α={tuple(a)}, "
            f"m={tuple(m)} — the cocharacter is not in the coweight lattice "
            f"P^∨, so the ψ-dressing product ∏_{{l<⟨α,m⟩}} is undefined.  A "
            f"genuine global form has m ∈ P^∨; check the coweight normalization.")
    return iv


def _psi_monomial_data(datum, m):
    """`(weight, qpow, sign_is_neg)` of the numerator monomial M(m) for dominant m.

    ⚠ **The sign is `(−1)^{S + Σ_{α>0}⟨α,m⟩}`, and at odd `⟨Σ⁺,m⟩` with the honest
    `S = −⟨ρ,m⟩` that exponent is HALF-INTEGRAL — so `(−1)^{sgn}` is a FOURTH ROOT
    OF UNITY, not a sign.**  `sgn % 2 == 1` is `False` for every half-integer
    (`Fraction(1,2) % 2 == 1/2`), so this function then *silently* returns `+1`
    where the honest value is `i^{2·sgn}`.  Measured at SO(3)=PSU(2)
    (`⟨Σ⁺,m⟩ = m[0]`), honest `S = −m/2`:

        m=(1,)  Σ⟨α,m⟩=1  S=−1/2  sgn=1/2   -> code +1, honest i^1
        m=(3,)  Σ⟨α,m⟩=3  S=−3/2  sgn=3/2   -> code +1, honest i^3
        m=(2,)  Σ⟨α,m⟩=2  S=−1    sgn=1     -> code −1, honest −1   (agree)

    Consequences worth knowing before you measure anything here:

      * shipped code never reaches the half-integral branch — the `ε`-corrected
        `RootDatum.atom_phase` keeps `S` integral — so this is a **trap for
        probes** (the same species as `atom_phase_doubled`), not a live bug;
      * it is why "simulate `𝖖^{1/2}`" never worked at odd height: the honest atom
        normalisation there wants a square root of **`−𝖖`**, and simulating a
        half-power supplies the `𝖖` half while this line silently drops the `i`;
      * the phase is **not** a free convention here.  `inner` is BILINEAR, so
        rescaling an element by `c` moves `I` by `c²`; measured, `S: 0 → −1` at
        `m=(1,)` moves `I` by `𝖖²`, and at even `m=(2,)` **only** `S = −⟨ρ,m⟩`
        gives `𝖖⁰ = 1` (`S = 0` and `S = −⟨Σ⁺,m⟩` both fail).  Orthonormality
        PINS `S = −⟨ρ,m⟩`;
      * hence at odd `⟨Σ⁺,m⟩` the pinned value is half-integral — which is why
        THIS monomial cannot be built there.

    ⚠ **THE CONCLUSION THAT USED TO FOLLOW — "no integral `S` restores
    orthonormality, so this presentation does not carry those lines at all;
    absorbing `−𝖖^{−1}` would need the forbidden `i·𝖖^{−1/2}`" — IS RETRACTED
    (2026-07-29).**  It treated `M(m)` as if the tier needed it.  The tier
    needs only the *cocycle*, and there `S` appears solely through the coboundary
    `δS̃`, which is an INTEGER even when `S̃` is not (`π = ⟨Σ⁺,·⟩ mod 2` is
    Weyl-invariant and additive, so the halves cancel).  So this function keeps
    the **reduced** integral phase and `cocycle_R` restores the honest one
    exactly, via `(−𝖖)^{δ(S_honest − S_used)}`.  There is no scalar to absorb, no
    `𝖖^{1/2}` and no `i` anywhere; the odd sectors build and are orthonormal —
    certified at SO(3) against the `PureSO3KAlgebra` BPS oracle, and measured at
    SO(5)/SO(7).  Everything above about *this monomial* stays true: it is the
    square root of the measure, it genuinely wants `(−𝖖)^{⟨ρ,m⟩}`, and the
    half-integral branch remains a trap for probes that call it directly."""
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
        am = _root_pairing_count(datum, a, m)
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
        am = _root_pairing_count(datum, a, m)
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


_PHASE_GAP_CACHE: dict = {}


def _phase_gap(datum, m):
    """`S_honest(m̃) − S_used(m̃)` at the dominant rep `m̃`, as a `Fraction`.

    `S_honest = −⟨ρ, m⟩ = −½⟨Σ⁺, m⟩` is the phase the axioms want (bar
    antimultiplicativity pins its Weyl-invariant content — the finding);
    `S_used = RootDatum.atom_phase` is what the datum actually hands the ψ
    dressing, which must be an integer because `LaurentPoly` has integer
    exponents.  The gap is therefore:

      * `−π(m)/2` on the DEFAULT phase (`S_used = −(⟨Σ⁺,m⟩ − ε)/2` with
        `ε(m_dom) = +1` at odd height), i.e. a half-integer exactly at odd
        `⟨Σ⁺, m⟩`;
      * the CENTRAL shift on `u_n` / `product_datum`, which carry their own
        certified convention — `½Σ_j m_j` at `U(N)`, Weyl-invariant *and*
        additive, hence invisible in the cocycle (the centre is the only
        surviving freedom).  `U(2)` at `m = (1,0)` has odd `⟨Σ⁺,m⟩` yet needs no
        correction, which is exactly why the gap must be computed rather than
        read off the parity."""
    # NOTE: the per-datum sub-cache holds a STRONG reference to its datum.  Keying
    # on `id(datum)` alone is a live bug — CPython reuses addresses after GC, so a
    # short-lived datum's entries get served to an unrelated later one (observed:
    # a spurious non-integral gap of 3/2 at `su_n(3)` during the gate run).
    ent = _PHASE_GAP_CACHE.get(id(datum))
    if ent is None or ent[0] is not datum:
        ent = (datum, {})
        _PHASE_GAP_CACHE[id(datum)] = ent
    sub = ent[1]
    key = tuple(m)
    hit = sub.get(key)
    if hit is None:
        md = datum.dominant_cochar_rep(key)
        hit = (-Fraction(datum._root_height(md)) / 2
               - Fraction(datum.atom_phase(md)))
        sub[key] = hit
    return hit


def _phase_coboundary(datum, a, b) -> int:
    """`δ(S_honest − S_used)(a, b)` — an INTEGER, always.

    `S̃ = S ∘ dominant_cochar_rep` may be a
    half-integer, but its coboundary cannot be: the parity character
    `π = ⟨Σ⁺, ·⟩ mod 2` is Weyl-invariant *and* additive (it kills every simple
    coroot, `⟨Σ⁺, α_i^∨⟩ = 2`), so the halves cancel in
    `S̃(a+b) − S̃(a) − S̃(b)`.  Measured over a box at SO(3)/SO(5)/SO(7)/Spin(5)/
    Sp(2)/G₂/SU(3): 0 non-integral coboundaries, including at the 12 (SO(5)) and
    62 (SO(7)) points where `S̃` itself is half-integral."""
    tot = tuple(a[i] + b[i] for i in range(datum.dim))
    v = _phase_gap(datum, tot) - _phase_gap(datum, a) - _phase_gap(datum, b)
    if v.denominator != 1:
        raise AssertionError(
            f"{datum.name}: δ(S_honest − S_used) is non-integral at "
            f"a={tuple(a)}, b={tuple(b)} ({v}) — the datum's atom phase differs "
            f"from −⟨ρ,m⟩ by something that is not a half-integer character.")
    return int(v)


def cocycle_R_via_psi(datum, a, b):
    """`R_{a,b} = (−𝖖)^{δ(S_honest−S_used)(a,b)} · ψ_a · T_{2a}(ψ_b) · ψ_{a+b}^{-1}`
    — the TRIVIALIZATION route, no longer the definition.
    `T_p : v^λ ↦ 𝖖^{⟨p,λ⟩} v^λ` (`q_shift`), so `T_{2a} : v ↦ 𝖖^{2a} v`.

    (This shift was written `S_a` until 2026-08-25.  Retired: `S` already names
    the atom phase `S(m)` — which appears in this very formula as
    `S_honest`/`S_used` — as well as the spectrum generator `S` and the flow's
    `S_RG`, so an `S_`-subscripted shift is a collision waiting to happen.  `T_p` was already the spelling used for the same
    operator in the matter cocycle `W_{m,m'} = T_{−m'}(Z_m)·T_m(Z_{m'})/Z_{m+m'}`,
    so this removes an alias rather than introducing a symbol.)

    **The prefactor is the honest atom phase, restored** (2026-07-29), and
    it is what lets the tier carry 't Hooft lines at odd `⟨Σ⁺, m⟩` — SO(3),
    SO(5), Sp(4)/Z₂, SO(7), PSU(4).  On the default phase the exponent evaluates
    to `π(a)·π(b)` (so it fires only when BOTH charges have odd `⟨Σ⁺,·⟩`); on a
    datum with its own convention — `u_n`, `product_datum` — it is `0`, because
    that convention differs from `−⟨ρ,m⟩` by a *central* linear functional, which
    is Weyl-invariant and additive and so cancels in the coboundary.  `U(2)` at
    `m = (1,0)` is the case that discriminates: odd `⟨Σ⁺,m⟩`, yet no correction.

    The atom phase is `S(m) = −⟨ρ, m⟩ = −½⟨Σ⁺, m⟩`, an integer on the coroot
    lattice (`⟨Σ⁺, α_i^∨⟩ = 2`) and a half-integer exactly at odd `⟨Σ⁺, m⟩`.
    `_psi_monomial_data` cannot hold the half — it materialises the standalone
    monomial `M(m) ∝ (−𝖖)^{⟨ρ,m⟩}`, i.e. the SQUARE ROOT of the measure, which
    is a function on `G̃`'s torus and not on `G̃/H`'s.  But the algebra never
    needs `S` as a number: it needs only the coboundary
    `δS̃(a,b) = S̃(a+b) − S̃(a) − S̃(b)` (`S̃ = S ∘ dominant_cochar_rep`), and that
    is an INTEGER even where `S̃` is not, because `π` is additive.  Concretely,
    on the default phase `RootDatum.atom_phase` returns the *reduced* integral
    `S_red = −(⟨Σ⁺,m⟩ − ε)/2` with `ε(m_dom) = +1` at odd height, so

        S_honest − S_red = −π(m)/2
        δ(S_honest − S_red)(a, b) = ½(π(a) + π(b) − π(a+b)) = π(a)·π(b)
        R_honest = (−𝖖)^{π(a)π(b)} · R_red                    (the law)

    — vanishing unless BOTH charges have odd `⟨Σ⁺,·⟩`, so the entire certified
    even sector is bit-identical (measured).  `_phase_coboundary` computes the
    general `δ(S_honest − S_used)` rather than this special case, so that data
    carrying their own (central-shifted) convention are correctly left alone.

    Why `dressing_psi` is deliberately NOT changed: on the support of a single
    canonical `π` is constant (the bubbled cells differ from the leading one by
    coroots, which `π` kills), so the honest `ψ` differs from the reduced one by
    an overall scalar `(−𝖖)^{π/2}` — invisible to `star_bubbling`'s (★), which is
    a homogeneous residue-cancellation condition.  Keeping `ψ` reduced therefore
    keeps every difference-operator surface integral and unchanged.

    Battery: a probe in the source repository."""
    a, b = tuple(a), tuple(b)
    tot = tuple(a[i] + b[i] for i in range(datum.dim))
    shift = tuple(2 * x for x in a)
    red = (dressing_psi(datum, a)
           * dressing_psi(datum, b).q_shift(shift)
           * dressing_psi_inv(datum, tot)).simplify()
    e = _phase_coboundary(datum, a, b)
    if not e:
        return red
    phase = TorusRational.from_laurent(TorusLaurent.monomial(
        datum, (0,) * datum.dim, LaurentPoly({e: (-1) ** (e % 2)})))
    return (red * phase).simplify()


# ---------------------------------------------------------------------------
# The cocycle as a PRIMITIVE — a product over the charged directions
# ---------------------------------------------------------------------------
#
# The author's ruling, 2026-08-24: *"The key is the cocycle in `U_m` products.  Ideally
# it would be a primitive object instead of being built from pieces like `ψ`"*,
# and then *"`R` should be defined by the closed form … `ψ` and `Z` should have
# the property that they trivialize it in an appropriate sense"*, whose *"main
# role … is to allow the formulation of the star axiom"*.
#
# So `R_{a,b}` is DEFINED below, directly, and `ψ` is demoted to a
# trivialization of it (`cocycle_R_via_psi`, checked by
# `verify_psi_trivializes_cocycle`).  Nothing in the definition mentions `ψ`,
# the atom phase `S`, the Weyl transport, or the phase coboundary — the
# closed form absorbs all of them, including at odd `⟨Σ⁺,m⟩` (measured at
# SO(5)/SO(7), where the phase prefactor used to be needed).
#
# The support rule is the author's own (2026-08-24): *"the (gauge) cocycle
# receives contributions from roots for which `m` and `m'` inner products have
# opposite signs"* — measured exhaustively, and it holds for the matter cocycle
# too (`matter_wrq_torus`), with numerators there where the gauge factor has
# denominators.


def cocycle_range(A, B):
    """`(d, D)` — the **Clebsch–Gordan range** the cocycle factors live on:
    `d = ||A| − |B||`, `D = |A| + |B|`, stepping by 2.

    Both charged sectors sit on this one range, which is what makes them
    manifestly analogous:

    * the **vector multiplet** takes the CLOSED range as *denominators*, ends
      once and interior twice;
    * **matter** takes the STRICT INTERIOR as *numerators*, once each
      (`matter_wrq_torus.matter_factor_exponents`).

    Measured for every `|A|, |B| ≤ 5`."""
    return abs(abs(A) - abs(B)), abs(A) + abs(B)


def CC_root_factor(datum, alpha, A, B):
    """The one-root factor of `CC_{a,b}` at `α`, with `A = ⟨a,α⟩`, `B = ⟨b,α⟩`.

    **This is the primitive**.  `CC` is what the product law is written in; `R`
    only ever appears as `T_{a+b}(CC)`, so `R` is derived below rather than the
    other way round.

    Trivial unless `A` and `B` have **opposite signs**.  For `A > 0 > B`, with
    `(d, D) = cocycle_range(A, B)` and `z = v^α`:

        𝖖^{|A||B|} · z^{min(|A|,|B|)}
        ─────────────────────────────────────────────────────────
        (1 − 𝖖^d z) · ∏_{k = d+2, d+4, …, D−2} (1 − 𝖖^k z)² · (1 − 𝖖^D z)

    — the closed Clebsch–Gordan range, ends once and interior twice.  For
    `A < 0 < B` it is `bar` of the charge-conjugate, `bar(r̃(−A,−B))`; conjugating
    both charges is the bar involution, *not* the swap."""
    A, B = int(A), int(B)
    if A * B >= 0:
        return TorusRational.one(datum)
    if A < 0 < B:
        return CC_root_factor(datum, alpha, -A, -B).bar()
    d, D = cocycle_range(A, B)
    alpha = tuple(alpha)
    c = min(abs(A), abs(B))
    out = TorusRational.from_laurent(TorusLaurent.monomial(
        datum, tuple(int(c * x) for x in alpha),
        LaurentPoly({int(abs(A) * abs(B)): 1})))
    out = out * TorusRational.factor_inv(datum, alpha, d)
    for k in range(d + 2, D, 2):
        f = TorusRational.factor_inv(datum, alpha, k)
        out = out * f * f
    return out * TorusRational.factor_inv(datum, alpha, D)


def CC(datum, a, b):
    """`CC_{a,b}` — **the primitive** cocycle, as the `f`-representation uses it.

    `CC` is the pure-gauge case; `CC[N]` (`matter_wrq_torus.CC_N`) combines gauge
    and matter, so `CC = CC[0]`.

    A product of `CC_root_factor` over the positive roots, so it is
    manifestly supported on the roots where `⟨a,α⟩` and `⟨b,α⟩` have opposite
    signs, and manifestly `1` when `a`, `b` and `a+b` share a closed Weyl
    chamber.  The product law of `Σ_m f_m(𝖖^m v)·U_m` is written directly in it:

        [fg]_n(v) = Σ_a f_a(𝖖^{a−n} v)·g_{n−a}(𝖖^{a} v)·R̃_{a,n−a}(v)"""
    out = TorusRational.one(datum)
    for al in datum.positive_roots():
        al = tuple(al)
        out = out * CC_root_factor(
            datum, al, datum.shift_pairing(tuple(a), al),
            datum.shift_pairing(tuple(b), al))
    return out.simplify()


def cocycle_R(datum, a, b):
    """`R_{a,b}` of `U_a U_b = R_{a,b}(v)·U_{a+b}` — **derived** from `CC`.

    `R = T_{a+b}(CC)`, the inverse of the half-shift that takes the bare atom
    frame to the `f`-representation.  The atom ordering is what fixes that the
    cocycle here carries the *bare* argument: with `U_m = ψ_m(v)·u^m` (dressing
    on the left, measured — see the suite in the source repository),

        U_a U_b = ψ_a(v)·ψ_b(𝖖^{2a}v)·u^{a+b} = R_{a,b}(v)·U_{a+b}

    Certified equal to the `ψ`-built route (`cocycle_R_via_psi`); see
    `verify_psi_trivializes_cocycle`."""
    tot = tuple(a[i] + b[i] for i in range(datum.dim))
    return CC(datum, a, b).q_shift(tot).simplify()


def psi_floor_defect_exponent(datum, a, b) -> int:
    """The exponent `e` with `R_{a,b} = (−𝖖)^e · δψ_floored(a,b)`.

    Two `ψ`s have to be kept apart:

    * the **honest** `ψ` trivializes `R` on the nose, but carries a *fractional*
      power of `(−𝖖)` at odd `⟨Σ⁺,m⟩` — it does not live in `Z[𝖖^{±1}]` and
      cannot be represented here at all;
    * the **floored** `ψ` — *"`ψ` with a fractional power of `(−𝖖)` stripped
      off"* — is what `dressing_psi` returns.  It stays integral and
      therefore **does not trivialize `R`**.

    This is the whole of that failure, and it is a `v`-free scalar `(−𝖖)^e`,
    which is what makes the floor safe where `ψ` is still used: *"which does not
    affect the star axiom because it is an overall factor"*, and (★) is a
    homogeneous residue-cancellation condition, so an overall scalar is
    invisible to it.  `e` is an integer, and that is why the floored
    `ψ` keeps every surface in integral powers of `𝖖`."""
    return _phase_coboundary(datum, a, b)


def verify_floored_psi_defect_is_a_v_free_scalar(datum, a, b) -> bool:
    """The floored `ψ` fails to trivialize `R` only by an overall `𝖖`-scalar.

    Checked by rebuilding `R` from `δψ_floored` times `(−𝖖)^e` and requiring
    equality: if the defect had any `v`-dependence, no such scalar would exist
    and this would fail."""
    a, b = tuple(a), tuple(b)
    tot = tuple(a[i] + b[i] for i in range(datum.dim))
    delta_psi = (dressing_psi(datum, a)
                 * dressing_psi(datum, b).q_shift(tuple(2 * x for x in a))
                 * dressing_psi_inv(datum, tot)).simplify()
    e = psi_floor_defect_exponent(datum, a, b)
    scal = TorusRational.from_laurent(TorusLaurent.monomial(
        datum, (0,) * datum.dim, LaurentPoly({e: (-1) ** (e % 2)})))
    return (delta_psi * scal).simplify() == cocycle_R(datum, a, b)


def verify_psi_trivializes_cocycle(datum, a, b) -> bool:
    """`δψ · (phase) = R` — the `ψ` route reproduces the primitive cocycle.

    ⚠ Read the name carefully: what trivializes `R` on the nose is the **honest**
    `ψ`, which is not representable here (fractional `(−𝖖)` powers).  What this
    checks is the floored `ψ` *together with* the integral phase coboundary that
    `cocycle_R_via_psi` restores — the two combined are the honest `ψ`.  The
    bare floored `ψ` does **not** trivialize `R`; that defect is
    `psi_floor_defect`, and it is a pure scalar.

    Since `R` is now defined independently of `ψ`, this equality is **emergent
    evidence** rather than a tautology — which is the point of the demotion."""
    return cocycle_R_via_psi(datum, a, b).simplify() == cocycle_R(datum, a, b)




# ===========================================================================
# ρ — the √-measure G-cocycle (datum-general), T2c
# ===========================================================================
def _rho_block(datum, k):
    """`(sign, qpow, wexp)` of the per-magnetic ρ block, datum-general.

    Reduces the U(N) `pure_un_kalgebra._rho_block_data` (of the keystone retired
    2026-09-19) to root-sums:
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
    # the DOUBLED phase, which is integral on all of `P^∨` with no parity
    # correction — see `RootDatum.atom_phase_doubled`.  Using `2*_phase_S` here
    # shifted ρ by `𝖖^{±1}` at odd `⟨Σ⁺,k⟩` and broke `I_{a,a} = 1 + O(𝖖)`, since ρ
    # is *defined* so the seed contributes exactly 1.
    qpow = -qf + datum.atom_phase_doubled(tuple(k))
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


def sector_weight(datum, m):
    """`w_m` — the per-sector measure of the Schur pairing, as its own object.

    The pairing is a sum over magnetic charge of contour integrals,

        I_{a,b}  =  Σ_m ∮ w_m(v) · f^a_m(arg) · f^b_m(arg),

    and this returns the `w_m` of that expression (the full integrand's remaining
    factor is the common Schur measure applied by `trace_residual`, which also
    carries the `(𝖖²;𝖖²)^{2·dim}` prefactor and the `1/|W|` Haar normalization).
    The two slots' "arg" is `T_{−m}∘v̄` on the left and `T_{−m}` on the right;
    measure and evaluation points travel together, which is why the convention is
    stated here rather than left to the call site.

    Explicitly `w_m = T_{−m}(G̃_m) · R̃_{−m,m}` — the ρ twist factor at the sector,
    half-shifted, times the `U_{−m}U_m` cocycle.  Both factors are this module's
    own (`_rho_Gtilde`, `CC`), so `w_m` is a *derived* object and not a second
    convention: `WRQTorus.pairing_residual` assembles `f_0` from it and agrees
    exactly with the `ρ`-and-convolve route.

    Weyl-covariant in `m` and independent of the states, so it is cacheable per
    `(datum, m)` and is the natural object on which to state a symmetry of the
    pairing (e.g. behaviour under `v → 1/v`) without computing any index."""
    m = tuple(m)
    neg = tuple(-x for x in m)
    return (_rho_Gtilde(datum, m, False).q_shift(neg)
            * CC(datum, neg, m)).simplify()


def sector_measure(datum, m):
    """`B_m` — the per-sector measure in the variable `u = 𝖖^{−m} v`, the frame
    in which `I_{b,a} = ⋆(I_{a,b})` (axiom 5, ρ-equivariance of the trace) is
    manifest.

    Writing the sector integrand of `pairing_residual` in `u = 𝖖^{−m} v` moves
    the contour from `|v| = 1` to `|u| = 1`, i.e. across the annulus in which the
    residuals' `(1 − 𝖖^k u^α)` poles can sit — this is exactly the *"contour
    shift and non-trivial pole cancellations"* by which the
    sector form and `Tr(ρ(a)·b)` coincide; the shipped computation stays in the
    `v`-frame (`pairing_residual` → `trace_residual`), and the `u`-frame is the
    presentation.  The equality of the two constant terms is measured, not
    argued here: per sector 128/128 and summed 36/36 at pure SU(2), `K=10`, even
    `⟨m,α⟩` (review pass, 2026-09-18); the odd-`⟨m,α⟩` case is UNMEASURED in
    the `u`-frame (its shifted measure is an infinite ratio there).  In that
    frame the Schur pairing reads

        I_{a,b} = (𝖖²;𝖖²)_∞^{2·dim}/|W| · Σ_m [u⁰]( M_m(u) · f^a_m(1/u) · f^b_m(u) ),

        M_m(u) = T_{+m}μ(u) · B_m(u),      μ = ∏_{α∈Φ} (u^α;𝖖²)_∞ (𝖖²u^α;𝖖²)_∞,

    with `B_m = T_{+m}(w_m) = G̃_m · T_{+m}(R̃_{−m,m})` an exact rational function
    (this object) and `T_{+m}μ` the charge-shifted Schur measure.  The two
    residuals enter at `1/u` and `u`, so exchanging `a ↔ b` is `u → 1/u`, and
    `I_{b,a} = I_{a,b}` (with `⋆` once flavour fugacities, which live on the same
    torus, are inverted too) follows from **inversion invariance of `M_m`**:

        M_m(1/u) = M_m(u)   ⟺   v̄(B_m) = Q_m · B_m,   Q_m := T_{+m}μ / T_{−m}μ,

    `Q_m` a FINITE product (the two shifts differ by `2⟨m,α⟩`; see
    `sector_measure_inversion_ratio`).  That identity is exact — no states, no
    truncation — and is the certificate `verify_sector_measure_inversion_symmetric`
    checks: measured 16/16 at su_2, su_3, sp_4, g_2, odd `⟨m,α⟩` included.

    Why this frame and not a convention: the two faces
    `Tr(ρ(a)·b)` and `Tr(b·ρ⁻¹(a))` assemble at `T_{∓m}`, so the shift relating
    them is `2m` and the pairing sits at exactly half of it — the midpoint, where
    neither slot is privileged.

    `𝖖 → 0` (the author's seed principle, measured 12/12 at su_2/su_3/sp_4): the
    lowest-order term of `M_m` is `𝖖⁰` times the **m-Levi Vandermonde**
    `∏_{⟨m,α⟩=0}(1−u^α)` exactly (no monomial, sign `+`), so at leading order the
    pairing is Schur orthogonality of the seed Levi characters against the Levi
    Weyl measure and `I_{a,b} = δ_{a,b} + O(𝖖)` is automatic.

    Relation to the `v`-frame factor: `sector_weight(datum, m) = T_{−m}(B_m)`.

    **Closed form, all `G`** (`sector_measure_closed_form`, 2026-09-19):
    `B_m = u^{Σ_{α>0}⟨m,α⟩α}·∏_{α>0} b_{|⟨m,α⟩|}(u^α)` with
    `b_t(z) = z^t/[(1−𝖖^t z)(1−𝖖^{−t}z)∏_{j=1}^{t−1}(1−𝖖^{t−2j}z)²]` — derived
    there (the two dressing ladders of `G_m` cancel identically, the shifted
    cocycle factorises over the roots), which turns the inversion certificate
    below into a theorem.  This function stays the DERIVED object the closed
    form is certified against (`verify_sector_measure_closed_form`)."""
    return sector_weight(datum, m).q_shift(tuple(m)).simplify()


def sector_measure_inversion_ratio(datum, m):
    """`Q_m = T_{+m}μ / T_{−m}μ`, the finite product

        ∏_{α∈Φ, s=⟨m,α⟩>0} 1 / [ ∏_{j<s}(1−𝖖^{−s+2j}u^α) · ∏_{j<s}(1−𝖖^{2−s+2j}u^α) ]

    (the `s<0` roots contribute the reciprocal factors, i.e. numerators) — each
    root of `μ` carries two Pochhammers, `(u^α;𝖖²)_∞` and `(𝖖²u^α;𝖖²)_∞`, hence
    the two products.  Exact `TorusRational`; see `sector_measure`.

    It collapses to the monomial `u^{−2Σ_{α>0}⟨m,α⟩α}`
    (`sector_measure_inversion_ratio_closed_form`, derived there); this
    function keeps the finite product as the DEFINITION the collapse is
    certified against."""
    m = tuple(m)
    d = datum.dim
    one = TorusLaurent(datum, {(0,) * d: LaurentPoly({0: 1})})
    num = one
    den = {}
    for a in datum.roots():
        s = datum.shift_pairing(m, a)
        a = tuple(a)
        if s > 0:
            for j in range(s):
                for base in (-s + 2 * j, 2 - s + 2 * j):
                    den[(a, base)] = den.get((a, base), 0) + 1
        elif s < 0:
            t = -s
            for j in range(t):
                for base in (-t + 2 * j, 2 - t + 2 * j):
                    num = num * (one + TorusLaurent(
                        datum, {a: LaurentPoly({base: -1})}))
    return TorusRational(datum, num, den)


def verify_sector_measure_inversion_symmetric(datum, m) -> bool:
    """The manifest-symmetry certificate: `v̄(B_m) == Q_m · B_m` exactly, i.e. the
    half-shift-centred measure `M_m = T_{+m}μ·B_m` is invariant under `u → 1/u`.
    No states, no truncation, no index computed.

    Since 2026-09-19 a THEOREM, given the closed form: `B_m = u^{C}∏ b_t(u^α)`
    with each `b_t` inversion-invariant and `Q_m = u^{−2C}`
    (`sector_measure_closed_form`, `sector_measure_inversion_ratio_closed_form`).
    Kept as the exact check on the derived objects."""
    B = sector_measure(datum, m)
    lhs = B.vinv().simplify()
    rhs = (sector_measure_inversion_ratio(datum, m) * B).simplify()
    return (lhs - rhs).simplify().is_zero()


# ===========================================================================
# The per-sector measure in CLOSED FORM — derived for every RootDatum
# (2026-09-19; user: "the closed form of the sector measure beyond su_2 …
# for all G and N is needed")
# ===========================================================================
#
# `sector_measure` above is DERIVED from the tier's two definitions (the
# √-measure twist `G̃_m` and the cocycle `CC`).  Everything below is the same
# object written out, with the derivation in the docstrings; the closed form
# is asserted equal to the derived object by `verify_sector_measure_closed_form`
# (pinned in `tests/test_sector_measure_closed_form.py`), and it makes the
# inversion certificate `v̄(B_m) = Q_m·B_m` a two-line theorem instead of a
# computation.


def sector_root_factor(datum, alpha, t):
    """`b_t(z)` — the one-root factor of the closed-form sector measure at the
    positive root `α`, with `t = |⟨m,α⟩|` and `z = u^α`:

        b_0(z) = 1,
        b_t(z) = z^t / [ (1 − 𝖖^t z)(1 − 𝖖^{−t} z) · ∏_{j=1}^{t−1} (1 − 𝖖^{t−2j} z)² ].

    **Derivation.**  It is `T_{+m}` of `CC_root_factor(α, −s, s)`, `s = ⟨m,α⟩`,
    the cocycle factor of `CC_{−m,m}` at that root.  For `s = t > 0` that
    factor is `bar` of the `(t, −t)` one, whose range is `(d, D) = (0, 2t)`:

        bar( 𝖖^{t²} z^t / [(1−z) ∏_{k=2,4,…,2t−2}(1−𝖖^k z)² (1−𝖖^{2t} z)] )
          = 𝖖^{−t²} z^t / [(1−z) ∏_k (1−𝖖^{−k} z)² (1−𝖖^{−2t} z)],

    and the half shift `T_{+m}: z ↦ 𝖖^t z` multiplies the numerator by `𝖖^{t²}`
    and moves every denominator exponent up by `t`, which is the display.  For
    `s = −t < 0` the `(t, −t)` factor itself under `z ↦ 𝖖^{−t} z` gives the same
    display, so only `|s|` enters.

    **Inversion invariance, exactly:** `b_t(1/z) = b_t(z)`.  The denominator's
    𝖖-exponents `{t, −t} ∪ {±(t−2j)}` form a multiset symmetric under negation,
    and `(1 − 𝖖^k/z) = −𝖖^k z^{−1}(1 − 𝖖^{−k} z)` over it contributes
    `(−1)^{2t}𝖖^0 z^{−2t}` — against `z^{−t}` from the numerator, i.e.
    `z^{−t}/(z^{−2t}·den(z)) = z^t/den(z)`."""
    t = int(t)
    if t < 0:
        raise ValueError(f"sector_root_factor: t = {t} must be |⟨m,α⟩| ≥ 0")
    if t == 0:
        return TorusRational.one(datum)
    alpha = tuple(alpha)
    out = TorusRational.from_laurent(TorusLaurent.monomial(
        datum, tuple(t * x for x in alpha), LaurentPoly({0: 1})))
    out = out * TorusRational.factor_inv(datum, alpha, t)
    out = out * TorusRational.factor_inv(datum, alpha, -t)
    for j in range(1, t):
        f = TorusRational.factor_inv(datum, alpha, t - 2 * j)
        out = out * f * f
    return out


def sector_measure_prefactor(datum, m):
    """`(sign, qpow)` of the scalar `sign·𝖖^{qpow}` in front of the closed-form
    sector measure — the only place the datum's phase conventions enter.

    With `m̃` the dominant representative of `m` and `m̃'` that of `−m̃`
    (`= −w₀m̃`), `S_ψ = _phase_S` (the ψ-dressing's phase, `RootDatum.atom_phase`,
    parity-corrected) and `2S_ρ = RootDatum.atom_phase_doubled` (the ρ block's,
    honest and integral):

        qpow  =  −S_ψ(m̃) + S_ψ(m̃') + 2S_ρ(m̃) + ⟨Σ⁺, m̃⟩,
        sign  =  (−1)^{ S_ψ(m̃) + S_ψ(m̃') + 2S_ρ(m̃) + ⟨Σ⁺, m̃⟩ }.

    **Where it comes from** (`G_m = v̄(ψ_m)·mono_m·ψ_{−m}⁻¹`, `m̃` dominant,
    `s_α = ⟨α,m̃⟩`): `v̄(ψ_m̃)` carries `M(m̃) = ε 𝖖^{Σs_α(s_α−1) − S_ψ(m̃)} v^{Σ s_α α}`
    conjugated; `mono_m̃` carries `𝖖^{−Σ s_α(s_α−1) + 2S_ρ(m̃)} v^{Σ s_α α}` with
    sign `(−1)^{rho_sign_exp}`, `rho_sign_exp = 2S_ρ + ⟨Σ⁺,m̃⟩`; and `ψ_{−m̃}⁻¹`,
    transported by `w₀` from its dominant representative `m̃'`, carries
    `M(m̃')⁻¹ ↦ ε' 𝖖^{−Σ s_α(s_α−1) + S_ψ(m̃')} v^{Σ s_α α}` (the `w₀` image of
    `v^{−Σ s'_α α}` is `v^{+Σ s_α α}`, since `−w₀` permutes the positive roots).
    The `Σ s_α(s_α−1)` terms cancel, the half shift `T_{+m̃}` of the total
    weight `Σ s_α α` adds `Σ s_α² = Σ s_α(s_α−1) + ⟨Σ⁺,m̃⟩`, and the display
    follows; the sign collects `ε·ε'·(−1)^{rho_sign_exp}` with
    `ε = (−1)^{S_ψ + Σ s_α}` (`_psi_monomial_data`).

    **It is `1·𝖖⁰` for every shipped convention**, which is why the closed form
    carries no visible scalar:

    * default datum — `S_ψ = −½⟨Σ⁺,·⟩ + ½ε` with `ε ∈ {0, ±1}` the parity
      correction, `2S_ρ = −⟨Σ⁺,·⟩`: `qpow = ½(ε(m̃') − ε(m̃)) = 0` because at a
      DOMINANT cocharacter the correction is `+1` at odd height (sign of the
      first non-zero simple-root pairing) and `0` at even, for `m̃` and `m̃'`
      alike; the sign exponent is `−⟨Σ⁺,m̃⟩ + ε(m̃) ≡ 0 (mod 2)`.  This is the
      odd-height case (SO(3), SO(5), SO(7), PSU(4) ω₁/ω₃): the two phases
      differ there, and they cancel — no `𝖖^{1/2}`, no `i`.
    * `u_n` — `S_ψ = S_ρ = Σ_j j·m_j`: with `−w₀` the reversal, `S(m̃) + S(m̃')
      = Σ_j (2j − N + 1) m_j = −⟨Σ⁺, m̃⟩`, so both exponents vanish identically.
    * `product_datum` — additive over the factors.

    A datum with some other `atom_phase` override gets the honest value here,
    so the closed form stays exact rather than assuming the cancellation."""
    m = tuple(m)
    md = tuple(datum.dominant_cochar_rep(m))
    mdp = tuple(datum.dominant_cochar_rep(tuple(-x for x in md)))
    s_psi = _phase_S(datum, md)
    s_psi_p = _phase_S(datum, mdp)
    two_s_rho = datum.atom_phase_doubled(md)
    height = datum._root_height(md)
    for name, val in (("S_ψ(m̃)", s_psi), ("S_ψ(m̃')", s_psi_p),
                      ("2S_ρ(m̃)", two_s_rho), ("⟨Σ⁺,m̃⟩", height)):
        if int(val) != val:
            raise NotImplementedError(
                f"sector_measure_prefactor: {name} = {val} is not an integer "
                f"at m = {m} on {datum.name}")
    s_psi, s_psi_p, two_s_rho, height = (int(s_psi), int(s_psi_p),
                                         int(two_s_rho), int(height))
    qpow = -s_psi + s_psi_p + two_s_rho + height
    sign_exp = s_psi + s_psi_p + two_s_rho + height
    return (-1 if sign_exp % 2 else 1), qpow


def sector_measure_closed_form(datum, m):
    """`B_m` in closed form — the per-sector measure of the Schur pairing in the
    variable `u = 𝖖^{−m}v` (see `sector_measure`), for EVERY `RootDatum`:

        B_m(u)  =  sign·𝖖^{qpow} · u^{Σ_{α>0} ⟨m,α⟩ α} · ∏_{α>0} b_{|⟨m,α⟩|}(u^α),

    with `b_t` = `sector_root_factor` and `(sign, qpow)` = `sector_measure_prefactor`
    (`= (1, 0)` on every shipped convention).  Derived, not fitted:

    1. **The two dressing ladders cancel identically.**  At dominant `m`,
       `v̄(ψ_m) = v̄(M(m))·∏_{α>0}∏_{l<s_α}(1−𝖖^{2l}v^{−α})⁻¹` and `ψ_{−m}⁻¹`,
       carried by `w₀` from its dominant representative, is
       `w₀(M⁻¹)·∏_{β>0}∏_{l<s_β}(1−𝖖^{2l}v^{−β})` — the SAME ladder, because
       `−w₀` permutes `Φ⁺` and `⟨α, −w₀m⟩ = ⟨−w₀α, m⟩`.  So `G_m` is a monomial,
       `G̃_m = T_{+m}G_m = sign·𝖖^{qpow}·v^{Σ_{α>0}⟨α,m⟩α}` (the scalar is
       `sector_measure_prefactor`'s).  This is the whole content of the
       "√-measure twist" at the level of the sector measure.
    2. **The half-shifted cocycle factorises over the roots** into
       `sector_root_factor`, `T_{+m}(CC_{−m,m}) = ∏_{α>0} b_{|⟨m,α⟩|}(u^α)`, with
       no monomial and no scalar left over.
    3. **Weyl covariance** carries 1–2 from dominant `m` to the orbit: the
       weight `Σ_{α>0}⟨α,m⟩α` is covariant (a flipped root `β<0` contributes
       `⟨β,m⟩β = ⟨−β,m⟩(−β)`), the block's 𝖖-power `−Σ_{α>0}⟨α,m⟩(⟨α,m⟩−1) +
       2S_ρ(m)` is Weyl-INVARIANT because both terms shift by
       `−2Σ_{α>0, ⟨α,m⟩<0}|⟨α,m⟩|` under a Weyl move and the shifts cancel, its
       sign exponent is invariant mod 2 for the same reason, and `ψ` is
       transported by `_weyl_transport`.  Hence `B_{wm} = w·B_m` and the display
       — which depends on `m` only through the pairings `⟨m,α⟩` — is the
       formula at every `m`.

    Two consequences that used to be measured are now theorems:

    * `b_t(1/z) = b_t(z)` and `u^{C} ↦ u^{−C}` for `C = Σ_{α>0}⟨m,α⟩α`, so
      `v̄(B_m) = u^{−2C}·B_m` — and `Q_m = T_{+m}μ/T_{−m}μ` IS `u^{−2C}`
      (`sector_measure_inversion_ratio_closed_form`), which is the certificate
      `verify_sector_measure_inversion_symmetric` in one line.
    * at `𝖖 → 0` only the roots with `⟨m,α⟩ = 0` survive in `T_{+m}μ·B_m` as
      the m-Levi Vandermonde (the `b_t`, `t > 0`, are cancelled by the shifted
      Pochhammers they sit under), the seed principle of `sector_measure`;
    * **the full `u`-frame measure `M_m = T_{+m}μ·B_m` is ENTIRE.**  Per root
      pair, `T_{+m}μ = (𝖖^t z;𝖖²)_∞(𝖖^{2+t}z;𝖖²)_∞(𝖖^{−t}/z;𝖖²)_∞(𝖖^{2−t}/z;𝖖²)_∞`
      with `t = ⟨m,α⟩`, `z = u^α`, vanishes to order exactly 2 at every
      `z = 𝖖^{k}` with `k ≡ t (mod 2)` (the first two factors cover `k ≤ −t−2`
      twice and `k = −t` once; the last two, via `(1−𝖖^{k}/z) ∝ (1−𝖖^{−k}z)`,
      cover `k ≥ 2−t` twice and `k = −t` once), while `b_t`'s denominators sit
      at `k ∈ {t, −t} ∪ {±(t−2j)}`, all `≡ t (mod 2)`, with multiplicity ≤ 2.
      So the sector measure crosses no pole under the contour move
      `u → 1/u`: whatever residues that move meets come from the residuals
      `f^a_m, f^b_m` alone (pinned exactly on `B_m`'s denominator data in
      `tests/test_sector_measure_closed_form.py`).

    Certified equal to the derived `sector_measure` by
    `verify_sector_measure_closed_form` — 367/367 cocharacters at u_2, u_3,
    su_2, su_3, sp_4, so_3, so_5, Spin(5), g_2, U(1)², U(2)×U(1), odd height and
    non-dominant `m` included (`tests/test_sector_measure_closed_form.py`;
    415/415 on the wider development grid).
    the design record's `aux_space._sector_weight_parts` organises its weight by the same
    per-root datum `|⟨m,α⟩|` in a different frame (the shifted Vandermonde
    sits in the vacuum there); no term-by-term identity is claimed here.

    **The same object in the author's `K_𝖖-algebras` draft (eq. Iexplicit):**
    there the pairing measure is written with ABSOLUTE-VALUE shifts and no
    rational factor, `∏_α (𝖖^{|⟨m,α⟩|}v^α;𝖖²)_∞(𝖖^{2+|⟨m,α⟩|}v^α;𝖖²)_∞`.  That is
    `T_{+m}μ·B_m` exactly: on the side of each root pair where the signed shift
    is `−t` the finite ratio of Pochhammers is `u^{−tα}/b_t(u^α)`, which is what
    `B_m` supplies (pinned per root pair, `tests/test_sector_measure_closed_form.py`
    leg 6).  So `B_m` is the conversion from the signed-shift Schur measure the
    trace machinery carries to the `|·|`-shifted one, in which the `u → 1/u`
    symmetry is visible with no certificate at all.  Matched after the fact:
    the derivation above is from the code's definitions."""
    m = tuple(m)
    d = datum.dim
    sign, qpow = sector_measure_prefactor(datum, m)
    wt = [0] * d
    for a in datum.positive_roots():
        s = datum.shift_pairing(m, a)
        for i in range(d):
            wt[i] += s * a[i]
    out = TorusRational.from_laurent(TorusLaurent.monomial(
        datum, tuple(int(x) for x in wt), LaurentPoly({int(qpow): sign})))
    for a in datum.positive_roots():
        t = abs(int(datum.shift_pairing(m, a)))
        if t:
            out = out * sector_root_factor(datum, a, t)
    return out.simplify()


def sector_measure_inversion_ratio_closed_form(datum, m):
    """`Q_m = T_{+m}μ / T_{−m}μ` collapsed: the monomial `u^{−2Σ_{α>0}⟨m,α⟩α}`.

    Per root pair `±α` with `s = ⟨m,α⟩ > 0` the finite product of
    `sector_measure_inversion_ratio` is `∏_j (1−𝖖^{k_j}u^{−α}) / ∏_j (1−𝖖^{k_j}u^{α})`
    over the multiset `{−s+2j} ∪ {2−s+2j}`, `j < s`, which is symmetric under
    `k ↦ −k`; `(1−𝖖^k u^{−α}) = −𝖖^k u^{−α}(1−𝖖^{−k}u^α)` then cancels the
    denominator and leaves `(−1)^{2s}𝖖^{Σk}u^{−2sα} = u^{−2sα}`.  Certified
    against the finite product in `tests/test_sector_measure_closed_form.py`."""
    m = tuple(m)
    d = datum.dim
    wt = [0] * d
    for a in datum.positive_roots():
        s = datum.shift_pairing(m, a)
        for i in range(d):
            wt[i] -= 2 * s * a[i]
    return TorusRational.from_laurent(TorusLaurent.monomial(
        datum, tuple(int(x) for x in wt), LaurentPoly({0: 1})))


def verify_sector_measure_closed_form(datum, m) -> bool:
    """The closed form equals the derived object: `sector_measure_closed_form ==
    sector_measure`, exactly (no truncation, no states)."""
    lhs = sector_measure_closed_form(datum, m)
    rhs = sector_measure(datum, m)
    return (lhs - rhs).simplify().is_zero()


# ===========================================================================
# ρ on LABELS — the explicit closed form (promoted to primary, the author's ruling
# 2026-08-23; the chart→twist→decompose route is the demoted verification
# route, `AbeKAlgebra.verify_rho_via_twist`)
# ===========================================================================
def rho_level_star(slots, m, inverse=False):
    """The per-slot rung counts `D_i` of the label-level flavour star
    `k_i ↦ −k_i − D_i` (`matter_wrq_torus.rho`'s level star, read at the
    star atom): for ρ, `D_i = Σ_{w∈wt(N_i)} max(0, +⟨w,m⟩)`; for ρ⁻¹ the
    `−⟨w,m⟩` half.  Weyl-invariant in `m` (the slot weight sets are
    Weyl-stable), so any representative of the magnetic orbit may be passed."""
    s = -1 if inverse else 1
    out = []
    for wts in slots:
        tot = 0
        for w in wts:
            c = s * sum(x * y for x, y in zip(m, w))
            if c > 0:
                tot += c
        out.append(tot)
    return tuple(out)


def rho_label(datum, m, e, slots=(), inverse=False):
    """`ρ^{±1}` on a canonical label `(m, e)` — the explicit label-level
    closed form (derived 2026-08-23 from the tier's definitions; promoted to
    the PRIMARY ρ the same day, with the torus-twist route
    demoted to the verifier `AbeKAlgebra.verify_rho_via_twist`).

    On Weyl orbits of pairs `[(m, e)]` — no chamber assumed on the input;
    both shift sums are Weyl-covariant so the map is well-defined mod W:

        ρ^{±1}[(m, e)] = [( −m,  −e  +  Σ_{α ∈ Φ:  ±⟨α,m⟩ > 0} |⟨α,m⟩|·α
                                      −  Σ_{w ∈ wt(N): ±⟨w,m⟩ > 0} |⟨w,m⟩|·w )]

    (ρ uses the roots / matter weights positive on `m`, ρ⁻¹ the negative
    half; vector multiplet with `+`, hypermultiplet with `−` — the
    weight-valued analogue of the scalar monopole-dimension combination.)
    Flavour levels star as `rho_level_star` above.  Two corollaries pinned in
    `tests/test_abe_rho_label.py`: at `N = Adj` (`wt(Adj) = Φ ∪ {0}`) the two
    sums cancel identically and `ρ[(m,e)] = [(−m,−e)]`, the antipode
    (`ρ² = id` on the gauge charges); at a minuscule `ω_k` of U(N) the gauge
    sum is `pure_un_kalgebra.witten_shift`, so this generalises the (retired)
    U(N) keystone's `rho_label` maps to any datum.

    Derivation, in two steps.  (1) At an anti-dominant source atom the twist
    factor `G_k` (`_rho_G`) is a PURE MONOMIAL: substituting `β = −w₀α`
    matches the dressing ladder of `ψ_k(1/v)` with the ladder of
    `ψ_{−k}^{-1}` factor by factor, so they cancel exactly; the leftover
    monomial's v-weight is the gauge sum above, and its q-power/sign are
    pinned by ρ's own normalization (the doubled atom phase — "ρ is defined
    so the seed contributes exactly 1"), so they never reach the label.
    (2) The W1+W2 acceptance reads a label off the leading Weyl orbit at the
    most cochar-dominant atom; the magnetic antipode sends the image's
    dominant atom back to the source's ANTI-dominant atom, where (1)
    applies.  With matter, `matter_wrq_torus.rho` divides by the image
    atom's zero-mode-ladder top `v^{E}` and stars the levels (image atom for
    ρ, source atom for ρ⁻¹), giving the matter sum and `rho_level_star`.

    Input: any joint Weyl representative `(m, e)` with `e` the electric
    extremal weight aligned to `m`'s chamber (the stored label of the native
    dominant frame qualifies).  Output: the image in the native dominant
    frame (`m` cochar-dominant, `e` Levi-dominant).  `slots` = one weight
    set per hypermultiplet slot (`matter_wrq_torus.slot_weights`).
    Fraction-safe (non-simply-connected forms carry Fraction coweights)."""
    m, e = tuple(m), tuple(e)
    d = datum.dim
    # joint anti-dominant representative (m₋, e₋); e₋ = the Levi-lowest
    # transport of the electric extremal weight
    m_minus = None
    cands = []
    for w in datum.weyl_elements():
        mm = tuple(datum.act_cochar(w, m))
        if datum.is_dominant_cochar(tuple(-x for x in mm)):
            if m_minus is None:
                m_minus = mm
            if mm == m_minus:
                cands.append(tuple(datum.act(w, e)))
    levi_coroots = [cor for a, cor in zip(datum.simple_roots,
                                          datum.simple_coroots)
                    if sum(a[i] * m_minus[i] for i in range(d)) == 0]
    e_minus = None
    for ee in cands:
        if all(sum(ee[i] * cor[i] for i in range(d)) <= 0
               for cor in levi_coroots):
            e_minus = ee
            break
    if e_minus is None:                            # pragma: no cover
        raise ValueError(f"rho_label: no Levi-lowest transport for {(m, e)}")
    n = tuple(-x for x in m_minus)
    Eg = [0] * d
    for a in datum.positive_roots():
        an = datum.shift_pairing(n, a)             # ⟨α, n⟩ ≥ 0 (n dominant)
        for i in range(d):
            Eg[i] += an * a[i]
    atom = m_minus if inverse else n               # the matter star atom
    En = [0] * d
    for wts in slots:
        for w in wts:
            c = sum(x * y for x, y in zip(atom, w))
            if c < 0:
                for i in range(d):
                    En[i] += (-c) * w[i]
    sgn = -1 if inverse else 1
    return (n, tuple(-e_minus[i] - sgn * Eg[i] - En[i] for i in range(d)))


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
                        * CC(self.datum, m, mp)).simplify()
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
        basis to `O(𝖖)`).  Routes through `ρ(a)` and the full cocycle convolution;
        `inner_by_sector` is the independent per-sector-measure route."""
        return (self.rho() * other).trace(K, w_cutoff=w_cutoff)

    def pairing_residual(self, other):
        """The magnetic-0 residual of `ρ(self)·other`, assembled **directly** over
        the SHARED magnetic support — no `ρ(self)` object, no non-zero sectors of
        the convolution — with the two residuals entering SYMMETRICALLY:

            f_0  =  Σ_m  w_m · v̄( T_{+m} f^a_m ) · T_{−m}( f^b_m )

        and the per-sector measure `w_m = sector_weight(datum, m)`.  The two slots
        sit at OPPOSITE half-shifts `±m`, one inverted — the `x_m(1/v)·y_m(v)`
        shape — so exchanging them is `v → 1/v` and nothing else.

        **Why the half-shift is the right frame, not a convention**.  The same assembly for the other face, `Tr(b·ρ⁻¹(a))`, is
        `Σ_m w'_m · v̄(T_{−m} f^a_m)·T_{+m}(f^b_m)` with
        `w'_m = T_{+m}(G̃⁻_m)·R̃_{m,−m}`: the two faces sit at `T_{∓m}`, so the
        shift relating them is `2m` and this expression uses exactly HALF of it.
        Writing the pairing at the midpoint is what makes neither slot privileged.
        Measured: face1 == face2 == the shipped `b·ρ⁻¹(a)` route, 16/16 at pure
        SU(2) (`K=10`) and 9/9 at pure SU(3) (`K=8`).

        Derived from this module's own two definitions, exact and not an
        approximation: `_twist` gives `ρ(a)_{−m} = v̄(f^a_m)·G̃_m`, `__mul__`
        contributes to total charge `0` exactly through the pairs `(−m, m)`, and
        `T_{−m}∘v̄ = v̄∘T_{+m}` moves the inversion across the shift (measured
        16/16, 15/15 on raw residuals).

        Cost: linear in `|supp(a) ∩ supp(b)|` where `inner` is quadratic in
        `|supp(a)|·|supp(b)|`.  Measured identical to the `inner` route (raw, i.e.
        `w_cutoff=False`) at pure SU(2) 36/36 (`K=10`, supports to 9, 2.2× faster)
        and pure SU(3) 16/16 (`K=8`, supports to 7, 1.3×) —
        `tests/test_wrq_sector_pairing.py`."""
        datum = self.datum
        acc = None
        for m, fa in self._f.items():
            fb = other.residual(m)
            if fb.is_zero():
                continue
            neg = tuple(-x for x in m)
            term = (sector_weight(datum, m)
                    * fa.q_shift(tuple(m)).vinv()      # v̄(T_{+m} f^a_m)
                    * fb.q_shift(neg))                 # T_{−m} f^b_m
            acc = term if acc is None else (acc + term)
        return TorusRational.zero(datum) if acc is None else acc.simplify()

    def inner_by_sector(self, other, K: int = 8, w_cutoff: bool = True):
        """`⟨a,b⟩ = Tr(ρ(a)·b)` via the per-sector measure (`pairing_residual`).

        A genuinely **separate expression** from `inner`, not a refactor of it: it
        is the sum-over-sectors contour form, and its agreement with `Tr(ρ(a)·b)`
        is a theorem (the total-charge-0 part of the convolution), certified rather
        than assumed — see `tests/test_wrq_sector_pairing.py`.

        The per-sector pieces are summed **before** the residue is taken, which is
        required: `trace_residual` is non-linear in its argument when
        `w_cutoff=True` (the `|W|`-divisibility cutoff can fire at different orders
        on different pieces), so a per-sector `trace_residual` then sum would be
        wrong.  It also reads a `0` the same way `trace_residual` does — see that
        function's two warnings before believing a vanishing pairing."""
        return trace_residual(self.datum, self.pairing_residual(other), K,
                              w_cutoff=w_cutoff)

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
    generic `peel_to_canonical` + `kl_bar_correct`."""
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
    keystone `PureUNKAlgebra` and the matter/quiver decompose all used it).
    **No live consumer since 2026-09-19**: every class that used it was retired
    to the source repository's archive that day (`PureGAbeKAlgebra(u_n(N))` builds pure U(N) now).

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
# Generic native build — product-and-peel + KL bar-correction
#
# The datum-general transcription of the `PureSU2KAlgebra` native build
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
    the product-of-Schur `pure_un_closed_form.levi_character` (retired
    2026-09-19)."""
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
    `p = m_N`.  Returns `(list_of_φ_k_with_multiplicity, p)`.

    **Honest-fails on non-integral coordinate differences.**  The multiplicities
    `c_k` count *generators*, so they must be `int`; at a non-simply-connected form
    a cocharacter lives in `P^∨` and its coordinates are `Fraction`s, which
    `range()` rejects on type alone.  Where they are integral in value the
    coercion below rescues them; where they are genuinely fractional this type-A
    recipe has no meaning and the caller (`build_cone_monomial`, i.e. the `cone`
    route) must decline so the dispatcher falls through to a route that applies.
    Measured at `SO(7)` `m = (1, 2, 3/2)`, where the bare `range()` raised
    `TypeError` — which `_guarded_dispatch` does NOT treat as "route not
    applicable", so it escaped and killed `chart()` on a label the licensed (★)
    solve builds.  Same coercion discipline as `_root_pairing_count`."""
    m = list(m_dom)
    p = m[-1]
    factors = []
    for k in range(1, N):
        c = m[k - 1] - m[k]
        ic = int(c)
        if ic != c:
            raise NotImplementedError(
                f"_un_fundamental_coweights: c_{k} = {c} is not an integer at "
                f"m={tuple(m_dom)} — the U(N) fundamental-coweight decomposition "
                f"m = Σ c_k φ_k + p·𝟙 counts generators, so a fractional "
                f"difference means this type-A recipe does not apply here.")
        for _ in range(ic):
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
# residual-native transcription of `pure_un_kalgebra._cone_build_impl` (retired) into the
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
    # A cocharacter of a non-simply-connected form lives in the coweight lattice
    # `P^∨ ⊋ Q^∨`, so its coordinates are `Fraction`s.  The box product below
    # indexes INTEGER boxes (`range(ck + 1)`), so a genuinely fractional `c`/`p`
    # means this route does not apply — return `None` and let the route
    # dispatcher fall through, rather than raising `TypeError: 'Fraction' object
    # cannot be interpreted as an integer` out of `range()`.  Measured at
    # `SO(5)` `m = (1, 1)`, `e = (1, 0)`, where the coordinates are integral in
    # VALUE but `Fraction` in TYPE, so the coercion below rescues it outright.
    # Same coercion discipline as `_root_pairing_count`.
    def _exact_int(x):
        iv = int(x)
        return iv if iv == x else None
    c_i = [_exact_int(x) for x in c]
    p_i = _exact_int(p)
    if p_i is None or any(x is None for x in c_i):
        return None
    c, p = c_i, p_i
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


_DOM_FUNCTIONAL_CACHE: dict = {}


def _dominance_functional(datum):
    """A linear functional `f` on the weight lattice with `f(α_i) = 1` for every
    simple root — hence `f(α) > 0` for every positive root, and `f` is strictly
    increasing along the dominance order.

    Solves `C·t = (1,…,1)` over `Q`, `C` the matrix whose rows are the simple
    roots in the repo's weight coordinates; then `f(λ) = ⟨λ, t⟩`.

    **Why this replaced the previous root-dot test (2026-07-27).**  The old
    `_levi_dom_rep` decided dominance by `⟨α, x⟩ ≥ 0` with a plain dot product
    of the *root* coordinate vector against the weight, while the repo's own
    `RootDatum.is_dominant` uses the *coroot* pairing.  In these coordinates
    (`simple_coroots` = the standard basis, i.e. weights in the
    fundamental-weight basis) those are different functionals whenever
    root ≠ coroot — **non-simply-laced only**, which is why it went unnoticed.
    At `G₂` the effect was total: no weight of the **7** was recognised as
    dominant, `_levi_dom_rep` fell through to returning its input, two weights
    tied for the peel maximum, a wrong character was subtracted, and
    `_levi_decompose` never terminated — so `PureGAbeKAlgebra(g_2()).multiply`
    raised on *Wilson lines*.  The old `_levi_height` had the same defect:
    `Σ_{α>0}⟨α,·⟩` is not monotone along the dominance order at `G₂`
    (`Σ_{α>0}α = (2,2)` and `⟨(2,2), α₂⟩ = −2 < 0`).

    Measured on applying this: `G₂` repaired (`7⊗7 = 1+7+14+27`, dim 49) and
    `Spin(5)` / `Sp(4)` / `SU(3)` / `U(2)` outputs bit-identical."""
    key = datum.name
    got = _DOM_FUNCTIONAL_CACHE.get(key)
    if got is not None:
        return got
    from fractions import Fraction
    d = datum.dim
    rows = [[Fraction(x) for x in a] + [Fraction(1)] for a in datum.simple_roots]
    piv_cols, r = [], 0
    for c in range(d):
        piv = next((i for i in range(r, len(rows)) if rows[i][c] != 0), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        pv = rows[r][c]
        rows[r] = [x / pv for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c] != 0:
                f = rows[i][c]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        piv_cols.append(c)
        r += 1
        if r == len(rows):
            break
    for i in range(r, len(rows)):
        if all(x == 0 for x in rows[i][:d]) and rows[i][d] != 0:
            raise ValueError(
                f"_dominance_functional: no functional positive on the simple "
                f"roots of {datum.name}")
    t = [Fraction(0)] * d
    for i, c in enumerate(piv_cols):
        t[c] = rows[i][d]
    for a in datum.simple_roots:                    # verify, do not trust
        if sum(Fraction(a[i]) * t[i] for i in range(d)) != 1:
            raise ValueError(
                f"_dominance_functional: verification failed on {datum.name}")
    t = tuple(t)
    _DOM_FUNCTIONAL_CACHE[key] = t
    return t


def _levi_dom_rep(datum, m, wt, Wm, Phim):
    """The Levi-dominant representative of weight `wt` in its `W_m` orbit.

    Taken as the orbit element maximizing the dominance functional `f`: since
    `f` is strictly positive on `Φ_m⁺ ⊆ Φ⁺`, the orbit maximum **is** the
    Levi-dominant element (unique, with a deterministic tuple tie-break)."""
    t = _dominance_functional(datum)
    best, bestk = None, None
    for w in Wm:
        x = datum.act(w, wt)
        k = (sum(t[i] * x[i] for i in range(datum.dim)), x)
        if bestk is None or k > bestk:
            best, bestk = x, k
    return tuple(wt) if best is None else best


def _levi_height(datum, wt, Phim):
    """`f(wt)` — strictly increasing along the dominance order at ANY datum."""
    t = _dominance_functional(datum)
    return sum(t[i] * wt[i] for i in range(datum.dim))


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
    and `(1−v^{−α}) = −v^{−α}(1−v^α)`) — the datum-general prescription (values unchanged where the fast path succeeds, since both compute
    the same `[v^0](measure·f_0)`).  k≠0 factors are 𝖖-expanded.  The
    `|W|`-divisibility cutoff marks the reliable 𝖖-order — NOTE it makes the
    result non-linear in `f0` (pieces of a decomposition may cut at different
    orders).  `w_cutoff=False` skips the division entirely and returns the raw
    UNDIVIDED series `[v^0](measure·f_0)·pref` (= `|W|·Tr`), which is exact and
    linear in `f0` — diagnostics/tests only.

    ⚠ **A ZERO HERE CAN MEAN "CUT", NOT "ORTHOGONAL"** — read this before believing
    a vanishing pairing.  If the *leading* coefficient is not divisible by `|W|`
    the cutoff fires at order 0 and the whole series is returned as `0`, with no
    signal that anything was dropped.  Measured (2026-07-29, `su_2`): the
    odd-`⟨Σ⁺,m⟩` cocharacter `m = (½,)` has raw `|W|·Tr = 𝖖⁻² + 2𝖖⁴ − 2𝖖⁸` —
    manifestly non-zero — and `w_cutoff=True` reports `0`, while the even control
    `m = (1,)` gives raw `2 − 2𝖖² + 4𝖖⁶` → `1 − 𝖖² + 2𝖖⁶` as it should.  That `0`
    was once reported as "the leading orbit is trace-null at odd height", which was
    wrong: the element is non-zero and so is its pairing.

    So when a pairing comes out `0` and that is *surprising*, re-run with
    `w_cutoff=False` before drawing any conclusion.  Scanning the requested `K` is
    NOT enough — this cutoff is internal and `K`-invariant (checked 2…12), which is
    exactly how it imitates a real phenomenon.  Same shape as the matter Nahm-window
    bug (`langlands_iso`'s module docstring): an internal window producing a
    structured false signal.

    ⚠ **AND THERE IS A SECOND, DISTINCT ZERO MECHANISM**: a `K`-WINDOW zero,
    where `w_cutoff=True` *and* `w_cutoff=False` both give `0` because the series
    simply starts beyond the requested order.  Measured: at SO(3) with `S = −2m`,
    `I(x·y, x)` is `0/0` at `K = 8` but `−𝖖⁹ + 3𝖖¹³` at `K = 14`; at SU(2) with
    `S = 0`, `0/0` at `K = 8` but `+2𝖖¹⁰ − 𝖖¹² − 2𝖖¹⁴` at `K = 14`; and with
    `S = −2m`, `0` at both `K = 8` and `14`, only appearing at `K = 20`
    (`+2𝖖¹⁸ − 𝖖²⁰`).  So `w_cutoff=False` alone does **not** clear a surprising
    zero — **also raise `K`**.  The two mechanisms are independent: the `|W|` cutoff
    is `K`-invariant, the window zero is `w_cutoff`-invariant, and a probe that
    checks only one of them will be fooled by the other."""
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
