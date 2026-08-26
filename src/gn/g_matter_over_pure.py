"""`g_matter_over_pure` — `G` gauge theory with `T^*N` matter as an `RGKAlgebra`
wrapping **pure `G`** at a general `RootDatum`.

The group-general counterpart of `un_nf_over_pure_rgflow.UNNfOverPure` (U(N)+N_f)
and `quiver_over_pure.QuiverOverPure` (unitary quivers), built by the same
standard recipe and with the same division of labour:

  1. take **pure `G`** = `pure_g_abe_kalgebra.PureGAbeKAlgebra(datum)` — the
     `AbeKAlgebra` on the group-general `WRQTorus` (ruling D5/D6);
  2. promote it to the flavoured K-algebra the standard way —
     `add_flavour(AbelianZPlusRing(M))`, one `U(1)` per hypermultiplet slot;
  3. wrap it in an `RGKAlgebra` whose only extra datum is the matter spectrum
     generator

         S_RG  =  Ψ  =  ∏_{i=1}^{M} ∏_{w ∈ wt(N_i)} E_𝖖( μ_i · v^w ) ,

     the product over the **weights of the matter representation** (with
     multiplicity), expanded on the pure-`G` Wilson-line characters `χ_e`.

The generic `RGKAlgebra` machinery then derives the whole algebra — `RG`,
`multiply`, `trace`, `inner_product`, `ρ` — from the auxiliary's own
multiplication and trace.  Nothing matter-specific is put in by hand.

Why this is the same object as `UNNfOverPure` at type A
------------------------------------------------------
At `datum = u_n(N)` with `N_i` the defining representation, `wt(N_i) = {e_j}`
and `v^{e_j} = v_j`, so `Ψ = ∏_{i,j} E_𝖖(μ_i v_j)` — the U(N)+N_f generator
verbatim.  That equality is the certification anchor
(the suite in the source repository), not a design aspiration.

The two type-A-specific steps of the U(N) flow both collapse onto **one**
group-general object, the Weyl character `wrq_torus.levi_character(datum, 0, ·)`:

| step | U(N)+N_f | general `G` |
|---|---|---|
| what the matter weights are | the colour indices `v_j` | the terms of `χ_{λ_N}` |
| expanding `[Ψ]_k` on Wilson lines | inverse Kostka (monomial → Schur) | dominance peel against `χ_e` |

The Weyl character is used in **both** directions: forward to enumerate the
matter weights, backward (as the peel target) to read the level components on
the canonical Wilson basis.  `levi_character` at magnetic `0` has `W_m = W` and
`Φ_m⁺ = Φ⁺`, so it is the full irreducible character of `G` — verified to give
the `G₂` fundamental **7** and adjoint **14**, the `Spin(5)` vector **5** and
spinor **4**, and `Sp(4)`'s fundamental **4**.

The abelianized readout (what this flow exists to produce)
---------------------------------------------------------
The point of the flow is not the flow: it is the **data** `RG(a)`, read as an
`AbeKAlgebra` element on the pure-`G` `WRQTorus` — i.e. as an abelianized
𝖖-difference operator — which is the input for directly constructing the
`AbeKAlgebra` of `(G, N)` (the matter-side analogue of what
the design notes pinned for U(N)+N_f).  `rg_chart(a)` is that
readout: the per-μ-level `WRQTorus` images of `RG(a)`.  Feed a level to
`star_bubbling.conventional` to see it as `D = Σ d_a u^a`.

The matter dressing is DERIVED, and its general-`G` form
-------------------------------------------------------
the design notes (2026-07-27) derives the U(N)+N_f matter dressing
`Z` from a property of `S_RG` **weaker** than the discovery relation — per
cell, `RG(a)_m · U_m · S_RG` has **no matter denominators** — which fixes `Z`
as the polar ladder of `S_RG(𝖖^{2m}v)/S_RG(v)`.  That derivation is written per
colour index `j` with `m_j`; per **weight** it reads `c := ⟨m, w⟩`, so at a
general datum it predicts

    Z(m)  =  ∏_i ∏_{w ∈ wt(N_i) : ⟨m,w⟩ < 0} ∏_{s=0}^{|⟨m,w⟩|−1}
             ( 1 + μ_i · 𝖖^{2s − |⟨m,w⟩| + 1} · v^w ) ,

reducing to the U(N) formula at `w = e_j`, `⟨m,e_j⟩ = m_j`.  This is the same
`c = ⟨m,w⟩` sign split as the independently derived per-cell matter window `Ξ`
of the vacuum pairing (the design notes §4m).  `matter_dressing` below
is that prediction, exposed for the Step-2 comparison against the measured
`rg_chart` — it is a **prediction to test**, not an input to the flow.

Scope / honesty
---------------
* The **flavour ring is the Cartan** `AbelianZPlusRing(M)` = `R(U(1)^M)`,
  imitating `UNNfOverPure` exactly; the D5 enhancement to the faithful
  non-abelian flavour group is the downstream recognize-after layer
  (`sun_flavour_enhancement` in type A).  For a general `(G, N)` the faithful
  flavour group also depends on whether `N` is complex, real or pseudo-real
  (`U(M)` vs `SO(2M)` vs `Sp(M)`); that is a physics call, deliberately not
  guessed here.  **First ruled instance (user, 2026-07-30, ruling D33):** for
  the **adjoint** — a real rep — at one hyper the flavour symmetry is `SU(2)`
  with the hyper a **doublet**, not the `U(1)` the tier currently carries.  That
  settles the `n = 1` real case; the general `Sp(2n)` reading is extrapolation,
  not a ruling, and the conservative `∏_i U(n_i)` branch still stands.
* Everything the auxiliary honest-fails on, this flow honest-fails on — but odd
  `⟨Σ⁺, m⟩` cocharacters are **no longer among them**: the "theorem" this list
  used to cite is retracted by ruling D31, and they build on the `AbeKAlgebra`
  tier like any other charge.  Pure-`G` charts outside the global form's line
  lattice are still refused.
* Wilson×Wilson fusion routes through the auxiliary's own `multiply` (the
  contract surface), so the Littlewood–Richardson content is the pure-`G`
  algebra's, never a reimplementation.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from itertools import product

from grading import Grading
from habiro import HabiroElement
from rgkalgebra import RGKAlgebra
from root_datum import RootDatum
from wrq_torus import levi_character
from zplus_ring import AbelianZPlusRing

from pure_g_abe_kalgebra import PureGAbeKAlgebra


__all__ = [
    "GMatterOverPure",
    "matter_weights",
    "matter_dressing",
    "fuse_characters",
    "single_hyper_character_expansion",
    "E_q_coefficient",
]


# ---------------------------------------------------------------------------
# E_𝖖 coefficients and the matter weights
# ---------------------------------------------------------------------------


def E_q_coefficient(n: int) -> HabiroElement:
    """`a_n = [E_𝖖(x)]_n = (−1)^n 𝖖^n / (𝖖²;𝖖²)_n`, exact (zero for `n < 0`).

    The same coefficient the U(N)+N_f flow uses; kept local so this module
    states its own convention, and pinned against `UNNfOverPure` end-to-end by
    the certification test rather than by sharing a private symbol."""
    if n < 0:
        return HabiroElement.zero()
    return HabiroElement.nahm_term((-1) ** n, n, [n])


def matter_weights(datum: RootDatum, lam) -> list[tuple]:
    """The weights of the irreducible `G`-representation of highest weight
    `lam`, **repeated by multiplicity** — the terms of the Weyl character
    `levi_character(datum, 0, lam)` (whose `W_m = W`, `Φ_m⁺ = Φ⁺`, so it is the
    full character of `G`).

    `matter_weights(u_n(N), (1,0,…,0))` is `[e_1, …, e_N]`, which is what makes
    `Ψ` reduce to `∏_{i,j} E_𝖖(μ_i v_j)`."""
    lam = tuple(lam)
    chi = levi_character(datum, (0,) * datum.dim, lam)
    out: list[tuple] = []
    for wt, lp in chi.terms.items():
        mult = lp._coeffs.get(0, 0)
        if mult < 0:
            raise ValueError(
                f"matter_weights: negative multiplicity {mult} at {wt} — "
                f"{lam} is presumably not a dominant weight of {datum.name}")
        out.extend([tuple(wt)] * int(mult))
    if not out:
        raise ValueError(
            f"matter_weights: {lam} has empty character in {datum.name} "
            "(not a dominant weight in this coordinate convention?)")
    return out


def matter_dressing(datum: RootDatum, matter, m):
    """The **predicted** general-`G` matter zero-mode dressing `Z(m)` — the
    per-weight reading of the derivation in the design notes.

    Returned symbolically as a list of factors `(i, w, s, exponent)` meaning
    `(1 + μ_i · 𝖖^{exponent} · v^w)`, one per flavour slot `i`, per matter
    weight `w` with `c = ⟨m,w⟩ < 0`, per zero-mode level `s = 0 … |c|−1`, with
    the bar-centered ladder `exponent = 2s − |c| + 1`.

    This is a **prediction to be tested** against the measured `rg_chart`, not
    an input to the flow.  At `datum = u_n(N)` and `N_i` the defining rep it is
    the U(N)+N_f closed form verbatim (`⟨m, e_j⟩ = m_j`)."""
    m = tuple(m)
    factors = []
    for i, lam in enumerate(_normalize_matter(datum, matter)):
        for w in matter_weights(datum, lam):
            c = sum(x * y for x, y in zip(m, w))
            if c >= 0:
                continue
            for s in range(-c):
                factors.append((i, tuple(w), s, 2 * s + c + 1))
    return factors


def _normalize_matter(datum: RootDatum, matter):
    """`matter` → a tuple of highest weights, one per hypermultiplet slot."""
    if matter is None:
        return ()
    matter = tuple(matter)
    if matter and all(isinstance(x, int) for x in matter):
        # a single weight given bare, e.g. (1, 0)
        matter = (matter,)
    out = []
    for entry in matter:
        if (len(entry) == 2 and not isinstance(entry[0], int)
                and isinstance(entry[1], int)):
            lam, count = entry                      # (weight, multiplicity)
            out.extend([tuple(lam)] * int(count))
        else:
            out.append(tuple(entry))
    for lam in out:
        if len(lam) != datum.dim:
            raise ValueError(
                f"matter weight {lam} has length {len(lam)}, expected "
                f"{datum.dim} for {datum.name}")
    return tuple(out)


# ---------------------------------------------------------------------------
# Ψ on the Wilson-character basis: weight-space expansion, then dominance peel
# ---------------------------------------------------------------------------


_FUNCTIONAL_CACHE: dict = {}


def _dominance_functional(datum: RootDatum):
    """A linear functional `f` on the weight lattice with `f(α_i) = 1` for every
    simple root — so `f` is **strictly increasing along the dominance order**,
    which is exactly what the character peel needs to terminate.

    Obtained by solving `C·t = (1,…,1)` over `Q`, `C` the matrix whose rows are
    the simple roots in the repo's weight coordinates; then `f(λ) = ⟨λ, t⟩` and
    `f(α_i) = (C t)_i = 1`.

    **Why not `Σ_{α>0} ⟨α, ·⟩`.**  That is the shape `wrq_torus._levi_height`
    uses, and it is *not* monotone outside the simply-laced case: at `G₂`,
    `Σ_{α>0} α = (2,2)` and `⟨(2,2), α₂⟩ = −2 < 0`, so a peel ordered by it can
    select a non-highest weight.  It happens to give the right answer on small
    inputs, which is precisely why the failure mode is dangerous.  This
    functional is correct by construction at any datum.

    Honest-fails (`ValueError`) if the system is inconsistent — no such `f`
    exists, and the peel would have no valid order."""
    key = datum.name
    got = _FUNCTIONAL_CACHE.get(key)
    if got is not None:
        return got
    from fractions import Fraction
    rows = [[Fraction(x) for x in a] + [Fraction(1)]
            for a in datum.simple_roots]
    d = datum.dim
    piv_cols = []
    r = 0
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
                f"roots of {datum.name} (inconsistent system)")
    t = [Fraction(0)] * d
    for i, c in enumerate(piv_cols):
        t[c] = rows[i][d]
    # verify rather than trust the elimination
    for a in datum.simple_roots:
        if sum(Fraction(a[i]) * t[i] for i in range(d)) != 1:
            raise ValueError(
                f"_dominance_functional: solution failed verification on "
                f"{datum.name}")
    t = tuple(t)
    _FUNCTIONAL_CACHE[key] = t
    return t


def _height(datum: RootDatum, wt):
    """`f(wt)` for the dominance functional above — strictly increasing along
    the dominance order at ANY datum (simply-laced or not)."""
    t = _dominance_functional(datum)
    return sum(t[i] * wt[i] for i in range(datum.dim))


def _char_weights(datum: RootDatum, e, cache: dict) -> dict:
    """`χ_e` as `{weight: int multiplicity}` (q-free), memoized."""
    e = tuple(e)
    got = cache.get(e)
    if got is None:
        chi = levi_character(datum, (0,) * datum.dim, e)
        got = {tuple(wt): lp._coeffs.get(0, 0) for wt, lp in chi.terms.items()}
        got = {wt: c for wt, c in got.items() if c}
        cache[e] = got
    return got


def _peel_to_characters(datum: RootDatum, poly: dict, cache: dict,
                        zero=None, is_zero=None) -> dict:
    """Decompose a Weyl-invariant weight-space polynomial `{weight: coeff}` into
    irreducible characters, returning `{dominant weight e: coeff}`.

    The group-general replacement for the U(N) flow's inverse-Kostka step, and
    a **read**, not a solve: at each step the dominance-maximal weight present
    must be the highest weight of a character in the expansion (its `χ` has
    multiplicity 1 there), so the coefficient is *read off* and `c·χ_e` is
    subtracted.  Coefficients stay exact (`HabiroElement`) throughout — the peel
    is Z-linear in them, so no q-expansion ever happens here.

    Honest-fails if the input is not Weyl-invariant (the dominance-maximal
    weight would carry no coefficient), which is the signature of a wrong
    matter weight multiset.

    Coefficient-generic: `zero` / `is_zero` default to `HabiroElement`, and are
    overridden to plain `int` for the character-ring fusion below."""
    if zero is None:
        zero = HabiroElement.zero()
    if is_zero is None:
        def is_zero(x):
            return x.is_zero()
    work = {tuple(wt): c for wt, c in poly.items() if not is_zero(c)}
    out: dict = {}
    guard = 0
    while work:
        guard += 1
        if guard > 20000:
            raise RuntimeError("_peel_to_characters: no termination")
        best = None
        for wt in work:
            dom = datum.dominant_rep(wt)
            key = (_height(datum, dom), dom)
            if best is None or key > best:
                best = key
        e = best[1]
        c = work.get(e)
        if c is None or is_zero(c):
            raise RuntimeError(
                f"_peel_to_characters: dominance-maximal weight {e} carries no "
                "coefficient — the input is not Weyl-invariant")
        out[e] = out.get(e, zero) + c
        for wt, mult in _char_weights(datum, e, cache).items():
            work[wt] = work.get(wt, zero) + c * (-int(mult))
        work = {wt: v for wt, v in work.items() if not is_zero(v)}
    return out


def fuse_characters(datum: RootDatum, e1, e2, cache: dict | None = None) -> dict:
    """`χ_{e1} · χ_{e2} = Σ_e N^e_{e1,e2} χ_e` — the tensor-product decomposition,
    as `{dominant weight e: multiplicity}`.

    Computed as the weight-multiset product followed by the same dominance peel
    (both are `levi_character` reads).  This is the group-general
    Littlewood–Richardson content: at `G₂` it gives
    `7 ⊗ 7 = 1 + 7 + 14 + 27` (total 49).

    Used instead of routing the fusion through `PureGAbeKAlgebra.multiply`
    because that path is blocked at non-simply-laced data by a **pre-existing
    spine bug**: `wrq_torus._levi_dom_rep` tests dominance by dotting a weight
    with the *root* coordinate vector, while `RootDatum.is_dominant` uses the
    *coroot* pairing.  In the repo's basis (`simple_coroots` = the standard
    basis) those differ exactly when root ≠ coroot, so at `G₂` no weight of the
    **7** is recognised as dominant, `_levi_dom_rep` silently returns its input,
    and the peel fails to terminate — `PureGAbeKAlgebra(g_2()).multiply(1, χ_7)`
    raises.  Certified equal to the auxiliary's `multiply` wherever that path
    works (type A), by the suite in the source repository."""
    cache = {} if cache is None else cache
    w1 = _char_weights(datum, e1, cache)
    w2 = _char_weights(datum, e2, cache)
    prod: dict = {}
    for a, ca in w1.items():
        for b, cb in w2.items():
            k = tuple(x + y for x, y in zip(a, b))
            prod[k] = prod.get(k, 0) + ca * cb
    return _peel_to_characters(datum, prod, cache,
                               zero=0, is_zero=lambda x: x == 0)


def single_hyper_character_expansion(datum: RootDatum, lam, k: int,
                                     cache: dict | None = None,
                                     conjugate: bool = False) -> dict:
    """One hypermultiplet in the representation `lam`, at flavour level `k`:
    `[∏_{w ∈ wt(lam)} E_𝖖(μ v^w)]_{μ^k}` as `{dominant weight e: HabiroElement}`
    over the Wilson-line characters `χ_e`.

    With `conjugate=True` the other admissible orientation of `S` is used (user,
    2026-08-02: *"`∏_{w ∈ wt(lam)} E_𝖖(μ^{-1} v^{-w})` also works as an `S`, with
    appropriate positive cone."*), i.e. `[∏_w E_𝖖(μ⁻¹v^{−w})]_{μ^{−k}}`.

    **That is exactly the DUAL representation**, which is the whole content of the
    `N` vs `N*` care the user flagged: negating every weight of `lam` gives the
    weight multiset of `lam*`, so

        expansion(lam, k, conjugate=True)  ==  expansion(lam*, k)

    and the flavour level is re-indexed by `|k|` rather than the cone being moved —
    the two descriptions differ only in bookkeeping.  For a **self-conjugate** `lam`
    the flag is therefore a no-op (SU(2) fundamental, any adjoint), which is why a
    comparison run only at SU(2)+1 cannot see the orientation at all; SU(3)+1 is the
    cheapest place it bites.  `E_𝖖` itself is untouched — the conjugation is in the
    argument, never `E_{𝖖⁻¹}` (standing user ruling).

    The group-general analogue of
    `un_nf_over_pure_rgflow.single_hyper_wilson_expansion`."""
    cache = {} if cache is None else cache
    weights = matter_weights(datum, lam)
    if conjugate:
        weights = [tuple(-x for x in w) for w in weights]
    zero = (0,) * datum.dim
    levels = {0: {zero: HabiroElement.one()}}
    for w in weights:
        nxt: dict = {}
        for lvl, row in levels.items():
            for n in range(0, k - lvl + 1):
                an = E_q_coefficient(n)
                if an.is_zero():
                    continue
                shift = tuple(n * x for x in w)
                dest = nxt.setdefault(lvl + n, {})
                for wt, c in row.items():
                    key = tuple(a + b for a, b in zip(wt, shift))
                    dest[key] = dest.get(key, HabiroElement.zero()) + c * an
        levels = {lvl: {wt: c for wt, c in row.items() if not c.is_zero()}
                  for lvl, row in nxt.items()}
        levels = {lvl: row for lvl, row in levels.items() if row}
    return _peel_to_characters(datum, levels.get(k, {}), cache)


# ---------------------------------------------------------------------------
# The RGKAlgebra: pure G ⊗ U(1)^M flavour + S_RG = Ψ.  Nothing else.
# ---------------------------------------------------------------------------


class GMatterOverPure(RGKAlgebra):
    """`G` gauge theory with `T^*N` matter, built by promoting pure `G` to the
    flavoured K-algebra (`add_flavour`) and supplying `S_RG = Ψ`.

    `GMatterOverPure(su_2(), (1,), nf=2)` is SU(2) with two fundamental hypers;
    `GMatterOverPure(u_n(2), (1,0), nf=1)` is U(2)+N_f=1 and reproduces
    `UNNf1OverPure(2)`; `GMatterOverPure(g_2(), (1,0))` is `G₂` with one
    hypermultiplet in the **7**.

    Parameters
    ----------
    datum : RootDatum
        The gauge group.
    matter : sequence
        The matter content: a bare dominant weight (one hyper), a sequence of
        dominant weights (one per hypermultiplet slot), or a sequence of
        `(weight, multiplicity)` pairs.
    nf : int, optional
        Shorthand multiplicity applied when `matter` is a single bare weight.
    """

    def __init__(self, datum: RootDatum, matter, nf: int | None = None,
                 allow_solve: bool = True, strict_guard: bool = False,
                 conjugate_S: bool = False) -> None:
        """`conjugate_S` selects the other admissible orientation of `S`
        (`∏_w E_𝖖(μ⁻¹v^{−w})` rather than `∏_w E_𝖖(μ v^w)`) — see
        `single_hyper_character_expansion`, where it is shown to be the **dual**
        matter representation and hence a no-op at self-conjugate `lam`.

        Defaulted to the existing orientation, so no current caller changes; it
        exists so a consumer that fixes an orientation on its own side (the
        boundary's matter letter is the conjugated one) can be compared against
        this flow **at a non-self-conjugate rep**, where the two genuinely differ.
        Without it such a comparison is only ever run where the flag cannot
        matter, and agreement there says nothing about the orientation."""
        self.datum = datum
        if nf is not None:
            matter = ((tuple(matter), int(nf)),)
        self._matter = _normalize_matter(datum, matter)
        if not self._matter:
            raise ValueError(
                "GMatterOverPure: no matter — use PureGAbeKAlgebra directly")
        self.conjugate_S = bool(conjugate_S)
        self._M = len(self._matter)
        self._pure = PureGAbeKAlgebra(datum, allow_solve=allow_solve,
                                      strict_guard=strict_guard)
        self._aux = self._pure.add_flavour(AbelianZPlusRing(rank=self._M))
        self._chi_cache: dict = {}
        self._level_cache: dict = {}
        self._content_cache: dict = {}

    def __repr__(self) -> str:
        reps = ",".join(str(lam) for lam in self._matter)
        return f"GMatterOverPure({self.datum.name}; {reps})"

    @property
    def M(self) -> int:
        """The number of hypermultiplet slots (= flavour rank)."""
        return self._M

    @property
    def matter(self) -> tuple:
        """The matter highest weights, one per hypermultiplet slot."""
        return self._matter

    def pure(self) -> PureGAbeKAlgebra:
        """The pure-`G` IR container — the `AbeKAlgebra` this flow lands in."""
        return self._pure

    def wilson_label(self, e) -> tuple:
        """The pure-`G` Wilson-line label `(0, e)` for the character `χ_e`."""
        return ((0,) * self.datum.dim, tuple(e))

    # ----- KAlgebra primitives: straight from the flavoured auxiliary -----

    def coefficient_ring(self):
        return self._aux.coefficient_ring()

    def identity(self):
        return self._aux.identity()

    def _label_section_decompose(self, label):
        return self._aux._label_section_decompose(label)

    # ----- RGKAlgebra contract: auxiliary, grading, S_RG = Ψ --------------

    def auxiliary(self):
        return self._aux

    def grading(self):
        """`Γ_RG = Z^M` flavour levels (height = total flavour number; one cone
        generator per hypermultiplet slot) — identical to the U(N)+N_f flow."""
        M = self._M
        cone = tuple(tuple(1 if j == i else 0 for j in range(M))
                     for i in range(M))
        return Grading(rank=M, deg=lambda lab: tuple(lab[1]),
                       height=(1,) * M, cone_gens=cone)

    def _slot_level(self, i: int, k: int) -> dict:
        """`[Ψ_i]_{μ_i^k}` on Wilson characters — cached per `(slot, level)`."""
        key = (i, int(k))
        got = self._level_cache.get(key)
        if got is None:
            got = single_hyper_character_expansion(
                self.datum, self._matter[i], int(k), self._chi_cache,
                conjugate=self.conjugate_S)
            self._level_cache[key] = got
        return got

    def _matter_wilson_content(self, k_vec) -> dict:
        """Wilson content of `[Ψ]_{k_vec} = ∏_i [Ψ_i]_{k_i}` — the fusion of the
        per-slot expansions in the character ring `R(G)` (`fuse_characters`).
        Returns `{pure Wilson label: HabiroElement}`.

        The fusion is tensor-product decomposition of `G`-characters, which is
        what `levi_character` + the dominance peel compute directly; going
        through `PureGAbeKAlgebra.multiply` instead is blocked at non-simply-laced
        data by the spine bug documented on `fuse_characters`.  The two agree
        wherever the auxiliary path works — pinned at type A by the suite."""
        k_vec = tuple(int(x) for x in k_vec)
        got = self._content_cache.get(k_vec)
        if got is not None:
            return got
        zero_wt = (0,) * self.datum.dim
        content = {zero_wt: HabiroElement.one()}
        for i, ki in enumerate(k_vec):
            if ki < 0:
                content = {}
                break
            exp = self._slot_level(i, ki)
            nxt: dict = {}
            for g, cg in content.items():
                for e, ce in exp.items():
                    for out_e, mult in fuse_characters(
                            self.datum, g, e, self._chi_cache).items():
                        term = cg * ce * int(mult)
                        nxt[out_e] = (nxt.get(out_e, HabiroElement.zero())
                                      + term)
            content = {e: c for e, c in nxt.items() if not c.is_zero()}
        out = {self.wilson_label(e): c for e, c in content.items()}
        self._content_cache[k_vec] = out
        return out

    def _multi_levels(self, cutoff: int):
        for kv in product(range(max(cutoff, 0) + 1), repeat=self._M):
            if sum(kv) <= cutoff:
                yield kv

    def _s_rg_component(self, p):
        """`[Ψ]_p` — the exact matter component at flavour multilevel `p`; `{}`
        off the cone."""
        p = tuple(int(x) for x in p)
        if any(x < 0 for x in p):
            return {}
        return {(g, p): c for g, c in self._matter_wilson_content(p).items()}

    def rg_generator(self, cutoff: int) -> dict:
        """`Ψ` windowed to total flavour number ≤ `cutoff`, keyed by auxiliary
        labels `((m, e), k_vec)`."""
        out: dict = {}
        for k_vec in self._multi_levels(cutoff):
            for g, c in self._matter_wilson_content(k_vec).items():
                out[(g, k_vec)] = c
        return out

    # ------------------------------------------------------------------
    # The abelianized readout — why this flow exists
    # ------------------------------------------------------------------

    def rg_chart(self, a, Kq: int = 24) -> dict:
        """`RG(a)` as an **abelianized** object: `{flavour level k_vec:
        WRQTorus}`, each value the pure-`G` chart image of that μ-level of
        `RG(a)`.

        This is the data the flow exists to produce — `RG(a)` as an
        `AbeKAlgebra` difference operator on the pure-`G` torus, which is the
        input for directly constructing the `AbeKAlgebra` of `(G, N)`.  Pass a
        level to `star_bubbling.conventional` to see it as `D = Σ_a d_a u^a`.

        `Kq` is the q-order at which the exact Habiro coefficients of `RG(a)`
        are expanded (they are exact up to that point; the pure-`G` charts
        themselves are exact)."""
        from weyl_torus_ring import TorusRational
        from wrq_torus import WRQTorus

        levels: dict = {}
        for (lab, k_vec), c in self.RG(a).terms.items():
            k_vec = tuple(k_vec)
            lp = c.expand(Kq) if isinstance(c, HabiroElement) else c
            if lp.is_zero():
                continue
            img = self._pure.chart(lab)
            cv = TorusRational.from_scalar(self.datum, lp)
            scaled = WRQTorus(self.datum,
                              {m: (f * cv).simplify()
                               for m, f in img.residuals().items()})
            cur = levels.get(k_vec)
            levels[k_vec] = scaled if cur is None else cur + scaled
        return levels

    def predicted_dressing(self, m):
        """The predicted general-`G` matter dressing `Z(m)` at cocharacter `m`
        (see `matter_dressing`) — for comparison against `rg_chart`."""
        return matter_dressing(self.datum, self._matter, m)
