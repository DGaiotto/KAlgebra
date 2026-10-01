"""`matter_wrq_torus` — `MatterWRQTorus`: the `(G, N)` enriched torus on the
**group-general WRQ substrate**.

The flavoured extension of `WRQTorus` per the validated design
(`tests/test_flavoured_wrq.py`): the gauge foundation is unchanged (the WRQ
cocycle IS the flavoured gauge backbone), and matter enters **additively** —

  * a **μ-level grading** on residuals: `f = {atom m: {k ∈ Z^{#slots}:
    TorusRational}}` (one slot per irreducible matter summand, mirroring the
    certified `QuiverURQTorus` level buckets at one node);
  * the **matter rung dressing** `Z(m)` — per slot `i` and each weight `w` of
    that slot's representation with `c = ⟨m, w⟩ < 0`, the bar-centered rung
    ladder `∏_{s<|c|} (1 + 𝖖^{2s+c+1} μ_i v^w)` — polynomial per level (NO new
    poles: denominators stay gauge roots);
  * the **matter cocycle** `W_{m,m'} = T_{−m'}(Z_m)·T_m(Z_{m'})/Z_{m+m'}` per
    μ-level — a finite net numerator, computed once by triangular division
    (the `quiver_urq_torus._quiver_cocycle` transcription);
  * ρ = the pure-gauge WRQ twist (`_rho_Gtilde`) followed by the matter
    factor of the image atom: division by the q-free Z-top monomial
    `v^{E(−m)}` and the **level star** `k_i ↦ −k_i − D_i(−m)` (`D` = the
    slot's rung count);
  * the **μ-refined trace**: the matter factors `∏_{i,w} E(μ_i v^w)·E(μ_i⁻¹
    v^{−w})` (Nahm window `W`) inserted into the datum-general Schur residue
    `wrq_torus.trace_residual` (which carries the k=0-pole folding +
    completeness pad).

**Matter is a representation, not a count** (widened 2026-07-28 so the
`(G, N)` tier can name its theory; user: *"fix the torusshape so
`GMatterAbeKAlgebra` can name itself"*).  The `Nf` argument may be an `int` —
that many copies of the **defining** representation, the original U(N)+N_f
reading reproduced bit-for-bit — or a per-slot sequence of dominant highest
weights (`slot_weights` normalizes either).  The formula above is the general
one; at `w = e_j` (U(N) fundamentals) `c = m_j` and it collapses to the
colour-indexed ladder it was, which is why the widening is default-inert.  It
is the same rung law as `matter_star_bubbling.matter_monomials`, which is what
lets `GMatterAbeKAlgebra` present its charts here.

Public language: atoms `U_m`, monomials `v^e`, μ-levels, and the cocycles
`R̃·W` — no `u`'s, no dressing operators.

Certification (the suite in the source repository, made before the type-A layer was
retired on 2026-09-19): transported `UNNfKAlgebra(2,1)` / `(2,2)` charts — multiply, ρ, and the μ-refined trace
commute with the `VRational → TorusRational` transport, cross-engine against
the certified `QuiverURQTorus`; plus the widening's own leg (the `int` and
expanded-weight declarations agree cell for cell, and `Z` matches
`matter_star_bubbling.Z_levels` at a non-type-A datum).
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
from wrq_torus import CC, _rho_Gtilde, trace_residual, WRQTorus


__all__ = ["MatterWRQTorus", "vr_to_tr", "tr_to_vr", "rep_weights",
           "defining_weight", "slot_weights"]


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
# what the matter IS: per-slot weight sets
# ---------------------------------------------------------------------------
_REP_CACHE: dict = {}
_DEF_CACHE: dict = {}


def rep_weights(datum: RootDatum, lam) -> tuple:
    """The weights of the irreducible `G`-representation of highest weight
    `lam`, **repeated by multiplicity** — the terms of the Weyl character
    `wrq_torus.levi_character(datum, 0, lam)` (whose `W_m = W`, `Φ_m⁺ = Φ⁺`, so
    it is the full character of `G`).  Sorted, so the rung order is
    deterministic.  Memoized: `TorusShape` construction and `repr` both ask for
    this, and a Weyl character is not free.

    `rep_weights(u_n(N), (1,0,…,0)) == (e_1, …, e_N)`, which is what makes the
    general rung ladder below reduce to the U(N)+N_f one."""
    from wrq_torus import levi_character
    lam = tuple(lam)
    hit = _REP_CACHE.get((datum.name, lam))
    if hit is not None:
        return hit
    chi = levi_character(datum, (0,) * datum.dim, lam)
    out: list = []
    for wt, lp in chi.terms.items():
        mult = lp._coeffs.get(0, 0)
        if mult < 0:
            raise ValueError(
                f"rep_weights: negative multiplicity {mult} at {wt} — {lam} is "
                f"presumably not a dominant weight of {datum.name}")
        out.extend([tuple(wt)] * int(mult))
    if not out:
        raise ValueError(
            f"rep_weights: {lam} has empty character in {datum.name}")
    got = tuple(sorted(out))
    _REP_CACHE[(datum.name, lam)] = got
    return got


def defining_weight(datum: RootDatum) -> tuple:
    """The highest weight of the node's **defining** representation — what a
    bare `int` matter multiplicity is shorthand for.  `U(N)`/`SU(N)`: the first
    fundamental weight; in general the first fundamental weight with a
    non-empty character.  Memoized (see `rep_weights`)."""
    hit = _DEF_CACHE.get(datum.name)
    if hit is not None:
        return hit
    d = datum.dim
    for i in range(d):
        w = tuple(1 if j == i else 0 for j in range(d))
        try:
            if rep_weights(datum, w):
                _DEF_CACHE[datum.name] = w
                return w
        except Exception:
            continue
    raise ValueError(
        f"defining_weight: no fundamental weight of {datum.name} has a "
        f"non-empty character — give the matter weights explicitly")


def slot_weights(datum: RootDatum, matter) -> tuple:
    """Normalize a matter declaration to **one weight set per μ-slot**.

    Accepts, per the tier's widened matter language:

      * an `int` `Nf` — `Nf` copies of the defining representation (the legacy
        U(N)+N_f reading, reproduced exactly);
      * a sequence of dominant **highest weights** — one hypermultiplet per
        entry, the slot's weight set being that irrep's weights;
      * a sequence of already-expanded weight sets (tuples of weights), passed
        through.

    One μ-slot per irreducible summand `N_i` is the standing flavour convention."""
    if isinstance(matter, int):
        w = defining_weight(datum)
        return (rep_weights(datum, w),) * int(matter)
    d = datum.dim
    out = []
    for entry in tuple(matter):
        entry = tuple(entry)
        if entry and all(isinstance(x, int) for x in entry):
            if len(entry) != d:
                raise ValueError(
                    f"slot_weights: highest weight {entry} has length "
                    f"{len(entry)}, expected {d} for {datum.name}")
            out.append(rep_weights(datum, entry))
        else:                                   # an expanded weight set
            ws = tuple(tuple(w) for w in entry)
            for w in ws:
                if len(w) != d:
                    raise ValueError(
                        f"slot_weights: weight {w} has length {len(w)}, "
                        f"expected {d} for {datum.name}")
            out.append(tuple(sorted(ws)))
    return tuple(out)


# ---------------------------------------------------------------------------
# matter rungs / Z-levels / matter cocycle (1-node flavour transcription)
# ---------------------------------------------------------------------------
def _flavour_rungs(atom, slots):
    """Rungs of `Z(m)` at `atom`: `(slot i, vw = w, shift)` for each μ-slot `i`
    and each weight `w` of that slot's representation with `c = ⟨atom, w⟩ < 0`,
    shifts bar-centered — `2s + c + 1` for `s = 0 … |c|−1`.

    At `slots = ((e_1,…,e_N),)·Nf` (the U(N)+N_f case) `c = atom_j`, so this is
    verbatim the previous colour-indexed ladder; the general form is
    `matter_star_bubbling.matter_monomials`, which is the same formula.

    `c` is coerced to `int` for the same reason as
    `matter_star_bubbling.matter_pairing`: at a **non-simply-connected** global form
    the atom has `Fraction` coordinates, so `⟨atom, w⟩` arrives as a `Fraction` even
    when its value is integral, and `range()` rejects that on type alone.  For a
    legal matter representation of the form every weight lies in the form's electric
    lattice, so the value *is* integral and the coercion is sound; a genuinely
    non-integral `c` means the matter is not a representation of this global form
    (`GNAbeKAlgebra` rejects that up front) and it honest-fails."""
    out = []
    for i, wts in enumerate(slots):
        for w in wts:
            c = sum(x * y for x, y in zip(atom, w))
            ic = int(c)
            if ic != c:
                raise NotImplementedError(
                    f"⟨atom, w⟩ = {c} is not an integer at atom {tuple(atom)}, "
                    f"matter weight {tuple(w)} — the matter is not a "
                    f"representation of this global form (a centre-charged "
                    f"weight against a fractional coweight), so the zero-mode "
                    f"ladder ∏_{{s<|c|}} has no meaning.")
            c = ic
            if c >= 0:
                continue
            for s in range(-c):
                out.append((i, tuple(w), 2 * s + c + 1))
    return out


def _rung_levels(datum, atom, slots) -> dict:
    """`Z(m)` per μ-level `{k: TorusRational}` — the finite bar-centered rung
    expansion (each rung contributes `1 + 𝖖^{shift} μ_slot v^{vw}`)."""
    d = datum.dim
    levels = {(0,) * len(slots): {(0,) * d: LaurentPoly({0: 1})}}
    for (slot, vw, sh) in _flavour_rungs(atom, slots):
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


def _matter_factor_via_Z(datum, m, mp, slots) -> dict:
    """`W_{m,m'} = T_{−m'}(Z_m)·T_m(Z_{m'})/Z_{m+m'}` per μ-level — finite
    net numerator by triangular division (raises past the rung budget).

    The **trivialization** route, no longer the definition:
    `Z` trivializes `W` exactly as `ψ` trivializes the gauge cocycle `R`, and
    `_matter_factor` below defines it directly.  Kept, and checked against the
    definition by `verify_Z_trivializes_matter_factor`."""
    key = (datum.name, tuple(m), tuple(mp), slots)
    hit = _W_CACHE.get(key)
    if hit is not None:
        return hit
    t = tuple(x + y for x, y in zip(m, mp))
    Zm = _rung_levels(datum, m, slots)
    Zmp = _rung_levels(datum, mp, slots)
    Zt = _rung_levels(datum, t, slots)
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
        _flavour_rungs(t, slots)) + 1
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
# ---------------------------------------------------------------------------
# The matter cocycle as a PRIMITIVE — a product over the matter weights
# ---------------------------------------------------------------------------
#
# The matter half of the author's ruling (*"same for the matter
# cocycle, which we should not forget"*; *"including matter of course"*).  `W`
# is defined directly below; `Z` is demoted to a trivialization of it, whose
# main role is the formulation of the star axiom.
#
# The support rule is the same as the gauge cocycle's — a matter weight `w`
# contributes exactly when `⟨m,w⟩` and `⟨m',w⟩` have opposite signs — but the
# statistics are opposite: matter contributes NUMERATOR factors
# `(1 + 𝖖^k μ v^w)` where the vector multiplet contributes denominators
# `1/(1 − 𝖖^k v^α)`.


def matter_factor_exponents(A, B):
    """The `𝖖`-exponents of the surviving factors at one matter weight, with
    `A = ⟨m,w⟩` and `B = ⟨m',w⟩`.

    **The matter counterpart of `wrq_torus.CC`, and already in the
    same (`f`-representation) frame** — `MatterWRQTorus.__mul__` applies `R̃` for
    the gauge part and this factor *unshifted*, because its own definition
    carries the shifts `T_{−m'}`, `T_m`.  So there is no separate `W̃`.

    Empty unless `A` and `B` have opposite signs.  Otherwise the two `Z`-ladders
    cancel down to their overlap, `c = min(|A|, |B|)` factors with exponents

        sgn(A) · (|A| + |B| − 1 − 2j),      j = 0 … c−1

    — manifestly odd under conjugating both charges.

    **Stated on the Clebsch–Gordan range, the parallel with the vector
    multiplet is exact** (`wrq_torus.cocycle_range`): with `d = ||A|−|B||` and
    `D = |A|+|B|`, the gauge factor spans the **closed** range `d … D` in steps
    of 2 as *denominators* (ends once, interior twice), and this one spans its
    **strict interior** `d+1 … D−1` as *numerators*, once each.  Measured for
    every `|A|, |B| ≤ 5`."""
    A, B = int(A), int(B)
    if A * B >= 0:
        return []
    c = min(abs(A), abs(B))
    s = 1 if A > 0 else -1
    return [s * (abs(A) + abs(B) - 1 - 2 * j) for j in range(c)]


def _matter_factor(datum, m, mp, slots) -> dict:
    """`W_{m,m'}` per μ-level — **the primitive**, a product over matter weights.

    Certified equal to the `Z`-built route; see
    `verify_Z_trivializes_matter_factor`."""
    key = (datum.name, tuple(m), tuple(mp), slots, "primitive")
    hit = _W_CACHE.get(key)
    if hit is not None:
        return hit
    n_slots = len(slots)
    levels = {(0,) * n_slots: TorusRational.one(datum)}
    for i, wts in enumerate(slots):
        for w in wts:
            w = tuple(w)
            A = datum.shift_pairing(tuple(m), w)
            B = datum.shift_pairing(tuple(mp), w)
            for e in matter_factor_exponents(A, B):
                mono = TorusRational.from_laurent(TorusLaurent.monomial(
                    datum, tuple(int(x) for x in w),
                    LaurentPoly({int(e): 1})))
                out: dict = {}
                for k, val in levels.items():
                    out[k] = (out[k] + val).simplify() if k in out else val
                    k2 = tuple(x + (1 if t == i else 0)
                               for t, x in enumerate(k))
                    term = (val * mono).simplify()
                    out[k2] = ((out[k2] + term).simplify()
                               if k2 in out else term)
                levels = {k: v for k, v in out.items() if not v.is_zero()}
    _W_CACHE[key] = levels
    return levels


def CC_N(datum, m, mp, slots) -> dict:
    """`CC[N]_{m,m'}` — the cocycle of the `(G, N)` theory, per μ-level.

    Name and split ruled by the author, 2026-08-24: **`CC` for pure gauge,
    `CC[N]` combining gauge and matter**, so `CC = CC[0]` is a genuine
    specialisation rather than a notational coincidence, matching the repo's
    `(G, N)` / `(G, 0)` convention.  This is exactly what the product law of
    `Σ_m f_m(𝖖^m v)·U_m` multiplies by, so it is the object worth naming; the
    matter-only piece is the ratio `CC[N]/CC[0]` and keeps no name of its own
    (`_matter_factor`).

    ⚠ **`N` must be a representation whose weights are allowed by the global
    form `G`**.  Otherwise `CC[N]` is not a cocycle *of that
    form* at all: at `PSU(N)` the fundamental is not admissible while the
    adjoint is.  This module sees only a `RootDatum` and the weight lists, so it
    cannot check that itself — `verify_matter_weights_admitted` does, given the
    form's `LineLattice`, and `GNAbeKAlgebra` filters on `lines.elec_admits`
    when it builds the slots."""
    gauge = CC(datum, m, mp)
    out = {}
    for k, w in _matter_factor(datum, m, mp, slots).items():
        term = (gauge * w).simplify()
        if not term.is_zero():
            out[k] = term
    return out


def verify_matter_weights_admitted(lines, slots) -> bool:
    """Every weight of every matter slot is an electric charge of the form.

    The precondition of `CC[N]`.  `lines` is a
    `global_form.LineLattice`; the test is its `m = 0` fibre, i.e. which
    representations are representations *of this form*."""
    return all(lines.elec_admits(tuple(w)) for wts in slots for w in wts)


def verify_Z_trivializes_matter_factor(datum, m, mp, slots) -> bool:
    """`δZ = W` — that `Z` **trivializes** the matter cocycle.

    The matter counterpart of `wrq_torus.verify_psi_trivializes_cocycle`.  `W`
    is defined independently, so this equality is emergent evidence."""
    a = {k: v.simplify()
         for k, v in _matter_factor(datum, m, mp, slots).items()
         if not v.simplify().is_zero()}
    b = {k: v.simplify()
         for k, v in _matter_factor_via_Z(datum, m, mp, slots).items()
         if not v.simplify().is_zero()}
    return set(a) == set(b) and all(a[k] == b[k] for k in a)



def _rungs_ED_of(datum, atom, slots):
    """`(E, D)` of `Z(atom)`: the q-free Z-top v-exponent and the per-slot rung
    counts (module-level twin of `MatterWRQTorus._rungs_ED`)."""
    E = [0] * datum.dim
    D = [0] * len(slots)
    for (slot, vw, _sh) in _flavour_rungs(atom, slots):
        D[slot] += 1
        for t in range(datum.dim):
            E[t] += vw[t]
    return E, D


def matter_sector_weight(datum, slots, m) -> tuple:
    """The per-sector weight of the `(G, N)` Schur pairing, `{r⃗: w_{m,r⃗}}` plus
    the level-star offset `D(−m)` — the matter analogue of
    `wrq_torus.sector_weight`.

        w_{m,r⃗} = T_{−m}( G̃_m · v^{−E(−m)} ) · CC[N]_{−m,m}[r⃗]

    Derived from this module's two definitions exactly as the pure one was from
    `wrq_torus`'s: `rho` sends the row `(m, k⃗)` to the atom `−m` with residual
    `v̄(f)·G̃_m·v^{−E(−m)}` at level `−k⃗ − D(−m)` (the level star), and `__mul__`
    reaches total charge `0` exactly through the pairs `(−m, m)`, with the
    cocycle `CC[N]_{−m,m} = Σ_r⃗ w_r⃗ μ^{r⃗}` — gauge and matter together — and
    `T_{−m}∘v̄ = v̄∘T_{+m}`.  So the matter measure is the pure measure `w_m`
    times the matter factor `CC[N]/CC[0]` of the cocycle and the q-free Z-top
    monomial, distributed over the μ-level shifts `r⃗`.

    Closed form, all `(G, N)`: `matter_sector_measure_closed_form` (the
    `u`-frame object `B^N_m = B_m·∏_{i,w}Ξ_{⟨m,w⟩}(μ_i u^w)`, derived there);
    this stays the DERIVED object it is certified against."""
    neg = tuple(-x for x in m)
    gt = _rho_Gtilde(datum, m, inverse=False)
    E, D = _rungs_ED_of(datum, neg, slots)
    base = gt
    if any(E):
        base = (base * TorusRational.from_laurent(TorusLaurent(
            datum, {tuple(-x for x in E): LaurentPoly({0: 1})}))).simplify()
    base = base.q_shift(neg).simplify()                     # T_{−m}(G̃_m v^{−E(−m)})
    out = {}
    for r, w in CC_N(datum, neg, m, slots).items():
        term = (base * w).simplify()
        if not term.is_zero():
            out[tuple(r)] = term
    return out, tuple(D)


# ---------------------------------------------------------------------------
# The matter sector measure in CLOSED FORM — derived for every (G, N)
# (2026-09-19; user: "the closed form of the sector measure … for all G and N
# is needed")
# ---------------------------------------------------------------------------


def matter_sector_measure(datum, slots, m) -> dict:
    """`B^N_m` — the `(G, N)` per-sector measure in the variable `u = 𝖖^{−m}v`, as
    a μ-Laurent polynomial `{k⃗: TorusRational}`: the DERIVED object,

        B^N_m  =  Σ_r⃗ T_{+m}( w_{m,r⃗} ) · μ^{ r⃗ − D(−m) },

    i.e. `matter_sector_weight` half-shifted, with ρ's level star `−k⃗ − D(−m)`
    on the first slot folded into the measure's level (that is what
    `pairing_residual` does with it).  In this frame the pairing reads

        I_{a,b} = pref · Σ_m [u⁰]( T_{+m}(μ_G μ_N)(u,μ) · B^N_m(u,μ)
                                    · (v̄⋆F^a_m)(u,μ) · F^b_m(u,μ) ),

    `F_m = Σ_k f_{m,k} μ^k` the μ-resolved residual at `m`, `μ_N` the
    hypermultiplet Schur factor `∏_{i,w} 1/[(−𝖖μ_i v^w;𝖖²)_∞(−𝖖μ_i⁻¹v^{−w};𝖖²)_∞]`
    (the trace's Nahm sum `Σ_n (−𝖖x)^n/(𝖖²;𝖖²)_n`), and `v̄⋆` the inversion
    `(u, μ) ↦ (1/u, 1/μ)`: `T_{+m}∘v̄∘T_{+m} = v̄` and `T_{+m}` of the `v`-frame
    weight is this object.  Exchanging `a ↔ b` is `v̄⋆` on the integrand, so
    axiom 5 (ρ-equivariance of the trace) is manifest per sector iff the full
    measure `T_{+m}(μ_Gμ_N)·B^N_m` is `v̄⋆`-invariant — the certificate
    `verify_matter_sector_measure_inversion_symmetric`, exact.  The passage from
    the `v`-frame constant term (where `inner_by_sector` computes) to this one
    is the contour shift discussed at `wrq_torus.sector_measure`; it is the
    same identity in the `v`-frame under `ι = v̄∘T_{2m}` (`T_{−m}` of the
    `u`-frame statement), and the constant term's invariance under `ι` is
    what remains measured rather than derived (end to end, 36/36 at SU(2)+1,
    25/25 at U(2)+1 and U(2)+2 — `tests/test_matter_sector_pairing.py`)."""
    wdict, D = matter_sector_weight(datum, slots, m)
    m = tuple(m)
    out = {}
    for r, w in wdict.items():
        k = tuple(x - y for x, y in zip(r, D))
        out[k] = w.q_shift(m).simplify()
    return out


def matter_weight_factor(datum, w, c, slot, n_slots) -> dict:
    """`Ξ_c(x)`, `x = μ_slot u^w`, `c = ⟨m,w⟩` — the one-weight factor of the
    closed-form matter measure, as `{k⃗: TorusRational}` (level `k⃗` = the power
    of `μ_slot`):

        Ξ_c(x)  =  x^{−max(c,0)} · ∏_{j=0}^{|c|−1} (1 + 𝖖^{|c|−1−2j} x),   Ξ_0 = 1.

    **Derivation.**  `CC[N]_{−m,m}` carries, per weight with `c ≠ 0`, the
    `|c|` numerator factors `(1 + 𝖖^e μ v^w)` with `e = −sgn(c)(2|c|−1−2j)`
    (`matter_factor_exponents(−c, c)`); the half shift `T_{+m}: v^w ↦ 𝖖^c v^w`
    makes the exponents `sgn(c)(2j+1−|c|)`, the symmetric set
    `{1−|c|, 3−|c|, …, |c|−1}` in either order.  The q-free Z-top `v^{−E(−m)}`
    and the level star `μ^{−D(−m)}` of `matter_sector_weight` are
    `∏_{c>0}(μ u^w)^{−c}` — the rungs of `Z(−m)` are exactly the weights with
    `⟨−m,w⟩ < 0`, `c` copies each — which is the `x^{−max(c,0)}`.

    **Inversion:** `Ξ_c(1/x) = x^{c}·Ξ_c(x)` for EITHER sign of `c`.  Over the
    symmetric exponent set `∏(1+𝖖^k/x) = x^{−|c|}∏(1+𝖖^k x)`; for `c > 0`
    the prefactor `x^{−c} ↦ x^{c}` combines to `x^{c}`, and for `c < 0` there
    is no prefactor and `x^{−|c|} = x^{c}`."""
    c = int(c)
    d = datum.dim
    w = tuple(int(x) for x in w)
    zero = (0,) * n_slots
    levels = {zero: TorusRational.one(datum)}
    if c == 0:
        return levels
    for j in range(abs(c)):
        mono = TorusRational.from_laurent(TorusLaurent.monomial(
            datum, w, LaurentPoly({abs(c) - 1 - 2 * j: 1})))
        out: dict = {}
        for k, val in levels.items():
            out[k] = (out[k] + val).simplify() if k in out else val
            k2 = tuple(x + (1 if t == slot else 0) for t, x in enumerate(k))
            term = (val * mono).simplify()
            out[k2] = (out[k2] + term).simplify() if k2 in out else term
        levels = {k: v for k, v in out.items() if not v.is_zero()}
    if c > 0:
        mono = TorusRational.from_laurent(TorusLaurent.monomial(
            datum, tuple(-c * x for x in w), LaurentPoly({0: 1})))
        levels = {tuple(x - (c if t == slot else 0) for t, x in enumerate(k)):
                  (v * mono).simplify() for k, v in levels.items()}
    return levels


def matter_sector_measure_closed_form(datum, slots, m) -> dict:
    """`B^N_m` in closed form, for EVERY `(G, N)`:

        B^N_m(u, μ)  =  B_m(u) · ∏_{i} ∏_{w ∈ wt(N_i)} Ξ_{⟨m,w⟩}( μ_i u^w ),

    `B_m` the pure closed form (`wrq_torus.sector_measure_closed_form`) and
    `Ξ_c` = `matter_weight_factor`.  Derived from `matter_sector_weight`'s
    definition `w_{m,r⃗} = T_{−m}(G̃_m v^{−E(−m)})·CC[N]_{−m,m}[r⃗]` with the
    level star: `T_{+m}` puts `G̃_m·T_{+m}(CC)` = `B_m` in front, and the matter
    factor of the cocycle, the Z-top and the star assemble weight by weight
    into `Ξ` (see there).  Returned as `{k⃗: TorusRational}` on the same level
    convention as `matter_sector_measure`.

    Consequences, now theorems rather than measurements:

    * with `x_{i,w} = μ_i u^w` and `c = ⟨m,w⟩`, `Ξ_c(1/x) = x^c Ξ_c(x)`, so
      `v̄⋆(B^N_m) = Q_m · ∏_{i,w} (μ_i u^w)^{⟨m,w⟩} · B^N_m`, and the
      hypermultiplet measure gives EXACTLY that monomial as its shift ratio
      (`matter_sector_measure_inversion_ratio`): the full `u`-frame measure is
      inversion-invariant and axiom 5 (ρ-equivariance of the trace) is manifest
      per sector on the matter tier, `⋆` included;
    * the `c < 0` window is the μ-dressed double window of the design record's per-cell
      `Ξ` in `aux_space` ("trivial for `c ≥ 0`" there is the `x^{−c}`-free
      half here, up to that frame's Z-top monomial).

    **The same object in the author's `K_𝖖-algebras` draft (eq. Iexplicit):** the
    hyper measure there carries `|⟨m,w⟩|` shifts on BOTH members of the pair,
    `1/(−𝖖^{1+|c|}μ_i v^w;𝖖²)_∞(−𝖖^{1+|c|}μ_i^{−1}v^{−w};𝖖²)_∞`, and no `Ξ`; the
    ratio to the signed-shift `T_{+m}μ_N` on the `−|c|` side is the finite
    product `Ξ_c(μ_i u^w)` exactly (pinned per weight,
    `tests/test_sector_measure_closed_form.py` leg 6), so `T_{+m}μ_N·∏Ξ_c` IS
    the draft's hyper measure and the μ-window is its `|·|` shift.

    Certified equal to the derived `matter_sector_measure` by
    `verify_matter_sector_measure_closed_form` — 203/203 cocharacters over the
    thirteen `g_matter_roster` presets, U(2)+1 to G₂+7 and two-slot Spin(5)
    spinors (`tests/test_sector_measure_closed_form.py`; 267/267 on the wider
    development grid)."""
    from wrq_torus import sector_measure_closed_form
    m = tuple(m)
    n = len(slots)
    levels = {(0,) * n: sector_measure_closed_form(datum, m)}
    for i, wts in enumerate(slots):
        for w in wts:
            c = int(datum.shift_pairing(m, tuple(w)))
            if c == 0:
                continue
            fac = matter_weight_factor(datum, w, c, i, n)
            out: dict = {}
            for k1, v1 in levels.items():
                for k2, v2 in fac.items():
                    k = tuple(x + y for x, y in zip(k1, k2))
                    term = (v1 * v2).simplify()
                    out[k] = (out[k] + term).simplify() if k in out else term
            levels = {k: v for k, v in out.items() if not v.is_zero()}
    return levels


def matter_sector_measure_inversion_ratio(datum, slots, m) -> tuple:
    """`Q^N_m = T_{+m}(μ_Gμ_N) / T_{−m}(μ_Gμ_N)` — `(level shift, TorusRational)`:
    the gauge ratio `Q_m` (`wrq_torus.sector_measure_inversion_ratio`, the finite
    product) times the matter ratio, which is the MONOMIAL

        ∏_{i} ∏_{w ∈ wt(N_i)} (μ_i u^w)^{⟨m,w⟩}

    — returned as the per-slot level shift `Σ_{w∈wt(N_i)} ⟨m,w⟩` and the
    `u`-monomial `u^{Σ_{i,w}⟨m,w⟩ w}` multiplying `Q_m`.

    **Derivation.**  The trace inserts, per weight, the Nahm sum
    `Σ_n (−𝖖x)^n/(𝖖²;𝖖²)_n = 1/(−𝖖x;𝖖²)_∞` at `x = μ_i v^w` and at `1/x`
    (`MatterWRQTorus._mu_trace_rows`, `HabiroElement.nahm_term((−1)^n, n, [n])`),
    the hypermultiplet Schur factor.  With `c = ⟨m,w⟩ > 0`,

        [(−𝖖^{1−c}x)_∞/(−𝖖^{1+c}x)_∞] · [(−𝖖^{1+c}/x)_∞/(−𝖖^{1−c}/x)_∞]
          = ∏_{j<c}(1+𝖖^{1−c+2j}x) / ∏_{j<c}(1+𝖖^{1−c+2j}/x) = x^{c},

    the exponent multiset being symmetric (`(1+𝖖^k/x) = 𝖖^k x^{−1}(1+𝖖^{−k}x)`);
    `c < 0` gives the reciprocal structure and the same `x^{c}`.  It is the
    sign convention `(−𝖖x)` — the trace's, matching the cocycle's `(1+𝖖^e μv^w)`
    rungs — that makes the ratio sign-free."""
    from wrq_torus import sector_measure_inversion_ratio
    m = tuple(m)
    d = datum.dim
    lev = [0] * len(slots)
    wt = [0] * d
    for i, wts in enumerate(slots):
        for w in wts:
            c = int(datum.shift_pairing(m, tuple(w)))
            lev[i] += c
            for t in range(d):
                wt[t] += c * int(w[t])
    mono = TorusRational.from_laurent(TorusLaurent.monomial(
        datum, tuple(wt), LaurentPoly({0: 1})))
    return tuple(lev), (sector_measure_inversion_ratio(datum, m) * mono)


def verify_matter_sector_measure_closed_form(datum, slots, m) -> bool:
    """`matter_sector_measure_closed_form == matter_sector_measure`, level by
    level, exactly."""
    a = matter_sector_measure(datum, slots, m)
    b = matter_sector_measure_closed_form(datum, slots, m)
    return set(a) == set(b) and all(
        (a[k] - b[k]).simplify().is_zero() for k in a)


def verify_matter_sector_measure_inversion_symmetric(datum, slots, m) -> bool:
    """The manifest-symmetry certificate of the matter tier:
    `v̄⋆(B^N_m) == Q^N_m · B^N_m` exactly, `v̄⋆ : (u, μ) ↦ (1/u, 1/μ)` — i.e. the
    `u`-frame measure `T_{+m}(μ_Gμ_N)·B^N_m` is invariant under the exchange of
    the two slots, `⋆` included.  No states, no truncation."""
    B = matter_sector_measure(datum, slots, m)
    lev, Q = matter_sector_measure_inversion_ratio(datum, slots, m)
    lhs = {tuple(-x for x in k): v.vinv().simplify() for k, v in B.items()}
    rhs = {tuple(x + y for x, y in zip(k, lev)): (Q * v).simplify()
           for k, v in B.items()}
    return set(lhs) == set(rhs) and all(
        (lhs[k] - rhs[k]).simplify().is_zero() for k in lhs)


class MatterWRQTorus:
    """A `(G, N)` enriched-torus element on the WRQ substrate: residuals
    `{atom m: {μ-level k ∈ Z^{#slots}: TorusRational}}`.

    `Nf` declares **what the matter is** (`slot_weights` normalizes it): an
    `int` for the U(N)+N_f reading — that many copies of the defining
    representation, the original meaning, bit-for-bit — or a per-slot sequence
    of dominant highest weights for a general `N = ⊕N_i`, one μ-slot per
    summand.  Everything matter-side (rungs, the cocycle `W`, ρ's level star,
    the μ-refined trace window) is indexed by the weights of those
    representations; at `w = e_j` it is the colour-indexed ladder it was."""

    __slots__ = ("datum", "Nf", "slots", "_f", "_d")

    def __init__(self, datum: RootDatum, Nf, f: dict):
        self.datum = datum
        self.slots = slot_weights(datum, Nf)
        self.Nf = len(self.slots)
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
        return MatterWRQTorus(self.datum, self.slots, f)

    # ----- ring ops -----
    def __add__(self, other: "MatterWRQTorus") -> "MatterWRQTorus":
        out = {m: dict(row) for m, row in self._f.items()}
        for m, row in other._f.items():
            dst = out.setdefault(m, {})
            for k, fr in row.items():
                dst[k] = (dst[k] + fr).simplify() if k in dst else fr
        return MatterWRQTorus(self.datum, self.slots, out)

    def __mul__(self, other: "MatterWRQTorus") -> "MatterWRQTorus":
        """`U_m U_{m'} = CC[N]_{m,m'}·U_{m+m'}`, μ-levels convolved.

        `CC[N]` is gauge and matter together (`CC_N`); the pure-gauge cocycle is
        its `N = 0` specialisation `CC`."""
        out: dict = {}
        for m, row1 in self._f.items():
            neg_m = tuple(-x for x in m)
            for mp, row2 in other._f.items():
                t = tuple(x + y for x, y in zip(m, mp))
                # `CC[N]` -- gauge and matter together, which is exactly what
                # the product law multiplies by.
                W = CC_N(self.datum, m, mp, self.slots)
                neg_mp = tuple(-x for x in mp)
                dst = out.setdefault(t, {})
                for k1, f1 in row1.items():
                    a = f1.q_shift(neg_mp)
                    for k2, f2 in row2.items():
                        b = f2.q_shift(m)
                        base = (a * b).simplify()
                        for r, w in W.items():
                            K = tuple(x + y + z for x, y, z in zip(k1, k2, r))
                            term = (base * w).simplify()
                            dst[K] = term if K not in dst else (
                                dst[K] + term).simplify()
        return MatterWRQTorus(self.datum, self.slots, out)

    def bar(self) -> "MatterWRQTorus":
        return MatterWRQTorus(self.datum, self.slots, {
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
        for (slot, vw, _sh) in _flavour_rungs(atom, self.slots):
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
        return MatterWRQTorus(self.datum, self.slots, out)

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
        return MatterWRQTorus(self.datum, self.slots, out)

    # ----- the Schur pairing as a sum over magnetic sectors -----
    def pairing_residual(self, other: "MatterWRQTorus") -> dict:
        """The magnetic-0 rows `{μ-level K⃗: TorusRational}` of `ρ(self)·other`,
        assembled DIRECTLY over the shared magnetic support with the two
        residuals entering symmetrically (the matter twin of
        `WRQTorus.pairing_residual`):

            f_{0,K⃗} = Σ_m Σ_{k⃗₁,k⃗₂,r⃗: −k⃗₁−D(−m)+k⃗₂+r⃗ = K⃗}
                        w_{m,r⃗} · v̄( T_{+m} f^a_{m,k⃗₁} ) · T_{−m}( f^b_{m,k⃗₂} )

        with `w_{m,r⃗}, D(−m) = matter_sector_weight(datum, slots, m)`.  Exchanging
        the two slots is `v → 1/v` together with the level star — which is why
        `I_{b,a} = ⋆(I_{a,b})` (axiom 5, ρ-equivariance of the trace) is manifest
        in this form, `⋆` being the flavour duality `μ^{k⃗} ↦ μ^{−k⃗}` on the
        coefficients.  No `ρ(self)` object and no non-zero sectors of the
        convolution are formed."""
        datum = self.datum
        acc: dict = {}
        for m, row1 in self._f.items():
            row2 = other._f.get(m)
            if not row2:
                continue
            neg = tuple(-x for x in m)
            wdict, D = matter_sector_weight(datum, self.slots, m)
            for k1, f1 in row1.items():
                a = f1.q_shift(tuple(m)).vinv()                  # v̄(T_{+m} f^a)
                kstar = tuple(-x - dd for x, dd in zip(k1, D))   # the level star
                for k2, f2 in row2.items():
                    base = (a * f2.q_shift(neg)).simplify()      # · T_{−m} f^b
                    for r, w in wdict.items():
                        Kv = tuple(x + y + z for x, y, z in zip(kstar, k2, r))
                        term = (base * w).simplify()
                        acc[Kv] = term if Kv not in acc else (acc[Kv] + term).simplify()
        return {k: v for k, v in acc.items() if not v.is_zero()}

    def inner_by_sector(self, other: "MatterWRQTorus", K: int = 8, W: int = 4) -> dict:
        """`⟨a,b⟩ = Tr(ρ(a)·b)` as `{μ-level: LaurentPoly}` via the per-sector
        measure (`pairing_residual`) and the μ-refined Schur residue — a
        separate expression from `(self.rho() * other).trace(K, W)`, agreeing
        with it as a theorem (the total-charge-0 part of the convolution),
        certified in `tests/test_matter_sector_pairing.py`.  Same Nahm window
        `W` as `trace`; the sector pieces are summed before the residue."""
        return self._mu_trace_rows(self.pairing_residual(other), K, W)

    # ----- μ-refined trace -----
    def trace(self, K: int = 8, W: int = 4) -> dict:
        """`{μ-level: LaurentPoly}` — the flavour matter factors
        `∏_{i,j} E(μ_i v_j)E(μ_i⁻¹ v_j⁻¹)` (Nahm window `W`) inserted into the
        datum-general Schur residue of the magnetic-0 residual."""
        row0 = self._f.get((0,) * self._d)
        if not row0:
            return {}
        return self._mu_trace_rows(row0, K, W)

    def _mu_trace_rows(self, row0: dict, K: int, W: int) -> dict:
        """The μ-refined Schur residue of magnetic-0 rows `{k⃗: TorusRational}`
        (shared by `trace` and `inner_by_sector`).

        **The Nahm window is raised to `K + pad` when `W` is smaller** — `pad`
        the depth of the rows' 𝖖-expansion below `𝖖⁰`.  Every other factor of
        the residue starts at 𝖖⁰ or later: the Nahm term of level `n` is
        `O(𝖖ⁿ)`, the Schur measure has no negative power.  So a Nahm term the
        window leaves out (a level `n > W`, or a combination the `Σ|level| ≤ W`
        pruning drops, whose levels sum past `W`) reaches only orders
        `≥ W + 1 − pad`, and `W ≥ K + pad` makes the result exact through `𝖖^K`.
        The expansion is the one `trace_residual` makes (`_factor_inv_series`):
        `1/(1 − 𝖖^k v^α)` starts at `𝖖⁰` for `k > 0` and at `𝖖^{|k|}` for
        `k < 0` (`−𝖖^{|k|}v^{−α}/(1 − 𝖖^{|k|}v^{−α})`), so a row `N/D` starts at
        `val(N) + Σ_{k<0} |k|·mult` — a property of the function, unchanged by a
        common factor of `N` and `D`.  The callers' `W = K + 2` is the bound at
        `pad ≤ 2`; ρ-images reach deeper.  Measured (2026-09-23, the audit):
        at SU(2)+1 the pairing of `L_{(1),(−2)}` with `L_{(1),(2)}` has `pad = 4`
        (its numerator alone starts at `𝖖⁻¹⁰`), and at `W = K + 2 = 6` it carried
        a spurious `(μ⁶ + μ⁻⁶)𝖖⁴` — breaking axiom 5's `I_{b,a} = ⋆I_{a,b}` at
        `𝖖⁴` — while `W ≥ 7` agrees with the printed sum over `m` (which has no
        Nahm window) through `𝖖⁴`.  On the 81 pairs of that window's lines the
        window needed never exceeded `K + pad` (numerator depths up to 20, `pad`
        at most 4)."""
        from habiro import HabiroElement
        d, Nf = self._d, self.Nf
        if not row0:
            return {}
        import math
        pad = 0
        for f0 in row0.values():
            low = min((min(lp._coeffs) for lp in f0._num._t.values()
                       if lp._coeffs), default=0)
            low += sum(-k * mult for (_a, k), mult in f0._den.items() if k < 0)
            # exponents may be Fractions (the SO(3)-type sectors carry the
            # materialised half phase their coboundary removes): the ceiling keeps
            # W an integer and the bound W ≥ K + depth
            pad = max(pad, math.ceil(-low))
        W = max(W, K + pad)
        Kq = K + 10

        def a_n(nn):
            return HabiroElement.nahm_term((-1) ** nn, nn, [nn]).expand(Kq + W)

        cells = [(i, tuple(w)) for i, wts in enumerate(self.slots)
                 for w in wts]
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
