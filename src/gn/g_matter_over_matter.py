"""`g_matter_over_matter` — **removing matter** as an `RGKAlgebra`: the flow

    (G, N_keep ⊕ N_drop)   ⟶   (G, N_keep)

at any `RootDatum`, for any matter representations.  Physically: give a large
real mass to the `N_drop` hypermultiplets and flow to the IR; they decouple and
the `N_keep` ones survive.

Why this is the missing rung
----------------------------
The `(G, N)` ladder built in the design record reaches the
matter theory two ways, and until now both landed on the **pure** theory:

    GMatterOverPure         (G, N) → (G, 0)   — removes ALL the matter
    GNAbeKAlgebra           (G, N) natively on the AbeKAlgebra tier

So the only matter-removal flow in the repo removed *all* of it at once.  This
module is the **partial** flow, and it is what makes matter removal an operation
you can iterate: `SU(2)+3 → SU(2)+2 → SU(2)+1 → SU(2)`, each rung a certified
`RGKAlgebra`, each rung's UV algebra presented over the previous rung's IR.
That is Goal 1.2 (new algebras by RG flow) and Goal 1.4 (the induced algebra
maps `A_𝖖^{UV} ↪ A_𝖖^{IR}`) on the `(G, N)` family.

`GMatterOverPure` is the `N_keep = ∅` case of this class, and that is asserted,
not asserted-in-a-docstring: the suite in the source repository compares
`S_RG` and `RG(a)` against it term by term.

The construction — the standard recipe, one level up
---------------------------------------------------
Identical in shape to `GMatterOverPure`; only the auxiliary changes:

  1. take the **IR algebra** — `GNAbeKAlgebra(datum, N_keep)` when there is
     surviving matter, `PureGAbeKAlgebra(datum)` when there is not.  Both are
     `AbeKAlgebra`s on the group-general WRQTorus, and both are *complete*
     KAlgebras, which is what lets either sit in the auxiliary slot;
  2. promote it the standard way — `add_flavour(AbelianZPlusRing(M_drop))`, one
     `U(1)` per **dropped** hypermultiplet slot (these are the μ's the flow
     grades by; the surviving slots' flavour stays inside the IR algebra's own
     coefficient ring, `∏_i R(U(n_i))`, exactly where it belongs);
  3. supply the matter spectrum generator of the **dropped** slots only

         S_RG  =  Ψ_drop  =  ∏_{i ∈ drop} ∏_{w ∈ wt(N_i)} E_𝖖( μ_i · v^w ) .

The generic `RGKAlgebra` machinery derives everything else.

The one fact that makes step 3 legitimate
-----------------------------------------
`Ψ` expands on the **Wilson lines** of the auxiliary, so the construction needs
the IR algebra's Wilson sector to be the character ring `R(G)` — with the *same*
`χ_e` as pure `G`, undisturbed by the surviving matter.  It is:

  * measured — the `(G, N_keep)` chart of `L_{(0,e)}` is *literally* the pure-`G`
    chart of `L_{(0,e)}` (single μ-level `0⃗`, identical residuals), and
    `L_{(0,e)}·L_{(0,e')}` in `(G, N_keep)` is plain Littlewood–Richardson
    (`χ₁² = χ₀ + χ₂` at SU(2)+1, no matter correction);
  * and structural — it is the content of (★) as the user states it: the
    canonical elements are `𝖖`-difference operators **acting on `G` characters /
    symmetric Laurent polynomials**, i.e. preserving the Neumann module
    `Λ = R(G)`, and `L_{(0,e)}|N] = χ_e` exactly.  The Wilson sector *is* `R(G)`
    whatever the matter is.

So `matter_weights` / `single_hyper_character_expansion` / `fuse_characters`
transfer verbatim from `g_matter_over_pure` — they are functions of the
`RootDatum`, not of the flow — and this module imports rather than reimplements
them.  Wilson×Wilson fusion is checked against the auxiliary's own `multiply`
(the contract surface) in the battery.

Labels
------
Auxiliary labels are `(IR label, k⃗)` with `k⃗ ∈ Z^{M_drop}`, and UV labels are
the same tuples (`apex` is the default identity — UV labels *are* IR apex
labels), exactly as in `GMatterOverPure`.  The IR label is whatever the IR
algebra uses: `(m, e)` for pure `G`, `((m, e), w)` for `(G, N_keep)`.  In both
cases `ir.fold((0,…,0), e)` is the Wilson label, which is why this module needs
no per-IR-kind branching on the label side.

Scope / honesty
---------------
* The dropped slots' flavour ring is the Cartan `AbelianZPlusRing(M_drop)` =
  `R(U(1)^{M_drop})`, imitating `GMatterOverPure` exactly.  The surviving slots
  keep the IR algebra's `∏_i R(U(n_i))` (ruling TM7).  The full non-abelian
  enhancement on the *dropped* directions is the downstream recognize-after
  layer, and for real / pseudo-real `N` it is the open physics call recorded in
  the design record.
* Everything the IR algebra honest-fails on, this flow honest-fails on — but odd
  `⟨Σ⁺, m⟩` cocharacters are **no longer among them**: the "theorem, not a gap"
  this list used to cite is retracted by ruling D31, and those charges are atoms
  of the abelianized tier like any other.  What does still raise is a
  `(G, N_keep)` chart whose bubbled cells are out of the constructor's reach,
  which raises rather than guessing.
* Cost is inherited from the IR algebra's `chart`, which for `(G, N_keep)` is
  the (★)-guarded constructor — so a rung with surviving matter is materially
  more expensive than the corresponding `GMatterOverPure` rung.  This is a cost
  limit, not a correctness one (the design record: bubbled-cell count, not rank or `|W|`).

Run `PYTHONPATH=$(ls -d src/* | paste -sd:) python3 src/gn/g_matter_over_matter.py` for a smoke
tour; certification is the suite in the source repository.
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
from zplus_ring import AbelianZPlusRing

from g_matter_over_pure import (
    _normalize_matter,
    fuse_characters,
    matter_dressing,
    single_hyper_character_expansion,
)


__all__ = ["GMatterOverMatter", "matter_removal_tower"]


class GMatterOverMatter(RGKAlgebra):
    """`(G, N_keep ⊕ N_drop) → (G, N_keep)` — decouple the `N_drop`
    hypermultiplets.

    ``GMatterOverMatter(su_2(), keep=((1,), 1), drop=((1,), 1))`` is
    SU(2)+2 flowing to SU(2)+1; ``GMatterOverMatter(sp_n(2), keep=(),
    drop=((1,0), 2))`` is Sp(4)+2 flowing to pure Sp(4) — the
    `GMatterOverPure` case, kept reachable so the two can be compared.

    Parameters
    ----------
    datum : RootDatum
        The gauge group (unchanged by the flow — only matter decouples).
    keep : sequence
        The **surviving** matter: a bare dominant weight, a sequence of dominant
        weights (one per hypermultiplet slot), or a sequence of
        `(weight, multiplicity)` pairs.  Empty means the IR is pure `G`.
    drop : sequence
        The **decoupled** matter, in the same declaration format.  Must be
        non-empty — an empty flow is the identity, not an RG flow.
    """

    def __init__(self, datum: RootDatum, keep=(), drop=(),
                 allow_solve: bool = True, strict_guard: bool = False,
                 pad: int = 1) -> None:
        self.datum = datum
        self._keep = _normalize_matter(datum, keep)
        self._drop = _normalize_matter(datum, drop)
        if not self._drop:
            raise ValueError(
                "GMatterOverMatter: `drop` is empty — there is no matter to "
                "remove, so this is not an RG flow.  Use the IR algebra "
                "directly (GNAbeKAlgebra / PureGAbeKAlgebra).")
        self._M = len(self._drop)
        if self._keep:
            from gn_abe_kalgebra import GNAbeKAlgebra
            self._ir = GNAbeKAlgebra(datum, self._keep,
                                          allow_solve=allow_solve,
                                          strict_guard=strict_guard, pad=pad)
        else:
            from pure_g_abe_kalgebra import PureGAbeKAlgebra
            self._ir = PureGAbeKAlgebra(datum, allow_solve=allow_solve,
                                        strict_guard=strict_guard)
        self._aux = self._ir.add_flavour(AbelianZPlusRing(rank=self._M))
        self._chi_cache: dict = {}
        self._level_cache: dict = {}
        self._content_cache: dict = {}

    def __repr__(self) -> str:
        keep = ",".join(str(l) for l in self._keep) or "—"
        drop = ",".join(str(l) for l in self._drop)
        return f"GMatterOverMatter({self.datum.name}; keep {keep}; drop {drop})"

    @classmethod
    def from_uv(cls, datum: RootDatum, matter, drop, nf: int | None = None,
                **kw) -> "GMatterOverMatter":
        """Build the flow from the **UV theory** and which slots decouple —
        usually the way you want to think about it.

        `matter` (with the optional `nf` shorthand) is the UV matter content in
        `GNAbeKAlgebra`'s declaration format; `drop` is a slot index or an
        iterable of slot indices into the normalized slot list.  The surviving
        slots keep their declared order.

        ``GMatterOverMatter.from_uv(su_2(), (1,), nf=3, drop=2)`` is
        SU(2)+3 → SU(2)+2."""
        if nf is not None:
            matter = ((tuple(matter), int(nf)),)
        slots = _normalize_matter(datum, matter)
        idx = {int(drop)} if isinstance(drop, int) else {int(i) for i in drop}
        bad = [i for i in sorted(idx) if not 0 <= i < len(slots)]
        if bad:
            raise ValueError(
                f"GMatterOverMatter.from_uv: slot index/indices {bad} out of "
                f"range for {len(slots)} hypermultiplet slot(s)")
        return cls(datum,
                   keep=tuple(s for i, s in enumerate(slots) if i not in idx),
                   drop=tuple(slots[i] for i in sorted(idx)), **kw)

    # ----- what theory this is -------------------------------------------

    @property
    def M(self) -> int:
        """The number of **dropped** hypermultiplet slots (= the flow's flavour
        grading rank).  The surviving slots are not counted here — their flavour
        lives in the IR algebra's own coefficient ring."""
        return self._M

    @property
    def keep(self) -> tuple:
        """The surviving matter highest weights, one per slot."""
        return self._keep

    @property
    def drop(self) -> tuple:
        """The decoupled matter highest weights, one per slot."""
        return self._drop

    @property
    def uv_matter(self) -> tuple:
        """The UV theory's full matter content, `keep + drop` — what
        `GNAbeKAlgebra(datum, uv_matter)` would present natively."""
        return self._keep + self._drop

    def ir(self):
        """The IR algebra this flow lands in — `GNAbeKAlgebra(G, N_keep)`,
        or `PureGAbeKAlgebra(G)` when nothing survives.  It is an `AbeKAlgebra`
        either way, and it is the decompose engine, not a flow container."""
        return self._ir

    def uv_native(self):
        """The **same abstract algebra** this flow presents, built natively on
        the `AbeKAlgebra` tier instead (`GNAbeKAlgebra(G, keep ⊕ drop)`).

        The Goal-1.3 comparison, and the sharpest available check on the flow —
        but note what it is: the native algebra **embeds** in this flow's, it does
        not generally equal it.  The flavour conventions differ (this flow grades
        every integrated-out slot by its own `U(1)`, `R(U(1)^{M})`; the native
        class packages identical slots into `∏_i R(U(n_i))` — TM7), and the map
        between them is the Cartan expansion `χ_λ ↦ Σ_{w ∈ wt(λ)} μ^w`
        (`zplus_ring.un_to_cartan_hom`): injective and multiplicative, with image
        the flavour-*symmetric* part.  It is an isomorphism exactly when all matter
        irreps are distinct — `g_matter_roster`'s `flow_iso()` builds and certifies
        it in that case and returns `None` otherwise."""
        from gn_abe_kalgebra import GNAbeKAlgebra
        return GNAbeKAlgebra(self.datum, self.uv_matter)

    def wilson_label(self, e):
        """The IR Wilson-line label for the character `χ_e`.

        `ir.fold` accepts `(m, e)` at both IR kinds — `(m, e)` for pure `G`,
        `((m, e), χ₀)` for `(G, N_keep)` — so this needs no branching, and it is
        the only place the flow touches the IR label convention."""
        return self._ir.fold((0,) * self.datum.dim, tuple(e))

    # ----- KAlgebra primitives: straight from the flavoured auxiliary -----

    def coefficient_ring(self):
        return self._aux.coefficient_ring()

    def identity(self):
        return self._aux.identity()

    def _label_section_decompose(self, label):
        return self._aux._label_section_decompose(label)

    # ----- RGKAlgebra contract: auxiliary, grading, S_RG = Ψ_drop ---------

    def auxiliary(self):
        return self._aux

    def grading(self):
        """`Γ_RG = Z^{M_drop}` — one cone generator per **dropped**
        hypermultiplet slot, height = total dropped-flavour number.  Same shape
        as `GMatterOverPure`'s grading; only the rank differs, because only the
        decoupled slots are being integrated out."""
        M = self._M
        cone = tuple(tuple(1 if j == i else 0 for j in range(M))
                     for i in range(M))
        return Grading(rank=M, deg=lambda lab: tuple(lab[1]),
                       height=(1,) * M, cone_gens=cone)

    def _slot_level(self, i: int, k: int) -> dict:
        """`[Ψ_i]_{μ_i^k}` on Wilson characters, for the `i`-th **dropped**
        slot — cached per `(slot, level)`."""
        key = (i, int(k))
        got = self._level_cache.get(key)
        if got is None:
            got = single_hyper_character_expansion(
                self.datum, self._drop[i], int(k), self._chi_cache)
            self._level_cache[key] = got
        return got

    def _matter_wilson_content(self, k_vec) -> dict:
        """Wilson content of `[Ψ_drop]_{k_vec} = ∏_i [Ψ_i]_{k_i}` — the fusion of
        the per-slot expansions in the character ring `R(G)`.  Returns
        `{IR Wilson label: HabiroElement}`.

        The fusion is tensor-product decomposition of `G`-characters
        (`fuse_characters`), which is `levi_character` + the dominance peel.  It
        agrees with the auxiliary's own `multiply` on Wilson lines wherever that
        path works — pinned in the battery — and is used directly because the
        auxiliary route is blocked at non-simply-laced data by the spine bug
        documented on `fuse_characters`."""
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
        """`[Ψ_drop]_p` — the exact matter component at dropped-flavour
        multilevel `p`; `{}` off the cone."""
        p = tuple(int(x) for x in p)
        if any(x < 0 for x in p):
            return {}
        return {(g, p): c for g, c in self._matter_wilson_content(p).items()}

    def rg_generator(self, cutoff: int) -> dict:
        """`Ψ_drop` windowed to total dropped-flavour number ≤ `cutoff`, keyed by
        auxiliary labels `(IR label, k_vec)`."""
        out: dict = {}
        for k_vec in self._multi_levels(cutoff):
            for g, c in self._matter_wilson_content(k_vec).items():
                out[(g, k_vec)] = c
        return out

    # ------------------------------------------------------------------
    # The abelianized readout — RG(a) as an element of the IR torus
    # ------------------------------------------------------------------

    def rg_chart(self, a, Kq: int = 24) -> dict:
        """`RG(a)` as an **abelianized** object: `{dropped-flavour level k_vec:
        IR torus element}`, each value the IR chart image of that μ-level.

        The IR chart type follows the IR algebra: a `WRQTorus` when the IR is
        pure `G`, a `MatterWRQTorus` when matter survives.  Pass a level to
        `star_bubbling.conventional` to see it as `D = Σ_a d_a u^a` — the form in
        which "the element acts on `R(G)`" is a statement you can check.

        `Kq` is the q-order at which the exact Habiro coefficients of `RG(a)` are
        expanded; the IR charts themselves are exact."""
        levels: dict = {}
        for (lab, k_vec), c in self.RG(a).terms.items():
            k_vec = tuple(k_vec)
            lp = c.expand(Kq) if isinstance(c, HabiroElement) else c
            if lp.is_zero():
                continue
            scaled = self._scale_ir(self._ir.chart(lab), lp)
            cur = levels.get(k_vec)
            levels[k_vec] = scaled if cur is None else cur + scaled
        return levels

    def _scale_ir(self, img, lp):
        """Multiply an IR chart image by the scalar q-Laurent `lp`.

        `MatterWRQTorus` exposes `_scaled`; the pure `WRQTorus` has no scalar
        multiply, so scale its residuals through `TorusRational` (the same thing
        `GMatterOverPure.rg_chart` does inline)."""
        if hasattr(img, "_scaled"):
            return img._scaled(lp, (0,) * len(next(iter(img.to_family()))))
        from weyl_torus_ring import TorusRational
        from wrq_torus import WRQTorus
        cv = TorusRational.from_scalar(self.datum, lp)
        return WRQTorus(self.datum,
                        {m: (f * cv).simplify()
                         for m, f in img.residuals().items()})

    def predicted_dressing(self, m):
        """The predicted matter dressing `Z(m)` of the **dropped** slots at
        cocharacter `m` (see `g_matter_over_pure.matter_dressing`) — for
        comparison against `rg_chart`.  A prediction to test, never an input."""
        return matter_dressing(self.datum, self._drop, m)


def matter_removal_tower(datum: RootDatum, matter, nf: int | None = None,
                         allow_solve: bool = True):
    """The full one-slot-at-a-time removal tower for `(G, N)`:

        [ (G,N) → (G,N minus last slot),  … ,  (G, one slot) → (G, 0) ]

    a list of `GMatterOverMatter` flows, outermost (most matter) first, whose
    last element is the `GMatterOverPure` case.  Composing the tower walks the
    whole `(G, N)` family down to pure gauge one hypermultiplet at a time, which
    is the iterated form of Goal 1.2 / 1.4 on this family.

    Slots are dropped from the end of the declared list, so the surviving matter
    at each rung is a prefix of `matter` — deliberate, so the rungs' IR theories
    are the obvious ones (`SU(2)+3 → +2 → +1 → pure`).

    `matter` takes `GNAbeKAlgebra`'s declaration format, with the same `nf`
    shorthand: ``matter_removal_tower(su_2(), (1,), nf=3)``."""
    if nf is not None:
        matter = ((tuple(matter), int(nf)),)
    slots = _normalize_matter(datum, matter)
    if not slots:
        raise ValueError("matter_removal_tower: no matter to remove")
    return [GMatterOverMatter(datum, keep=slots[:i], drop=(slots[i],),
                              allow_solve=allow_solve)
            for i in range(len(slots) - 1, -1, -1)]


if __name__ == "__main__":
    import root_datum as rd

    # Deliberately cheap: rank-1 rungs only.  The full sweep — the
    # GMatterOverPure agreement leg, the Wilson-fusion leg, the RG battery and
    # the tower — is the suite in the source repository.
    print("=" * 68)
    print("SU(2)+2 → SU(2)+1  (partial: one slot survives)")
    print("=" * 68)
    F = GMatterOverMatter.from_uv(rd.su_2(), (1,), nf=2, drop=1)
    print(f"  {F!r}")
    print(f"  IR         = {F.ir()!r}")
    print(f"  aux        = {F.auxiliary()}")
    print(f"  UV matter  = {F.uv_matter}   (native: {F.uv_native()!r})")
    print(f"  grading    = rank {F.grading().rank}")
    srg = F.rg_generator(1)
    print(f"  S_RG(≤1)   = {len(srg)} terms")
    for lab, c in sorted(srg.items(), key=lambda kv: (sum(kv[0][1]), str(kv[0]))):
        print(f"      {lab}  ->  {c}")

    print()
    print("=" * 68)
    print("SU(2)+1 → SU(2)  (the GMatterOverPure case, keep = ∅)")
    print("=" * 68)
    P = GMatterOverMatter(rd.su_2(), keep=(), drop=((1,),))
    print(f"  {P!r}")
    print(f"  IR         = {P.ir()!r}")
    srg = P.rg_generator(1)
    print(f"  S_RG(≤1)   = {len(srg)} terms")
    for lab, c in sorted(srg.items(), key=lambda kv: (sum(kv[0][1]), str(kv[0]))):
        print(f"      {lab}  ->  {c}")

    print()
    print("=" * 68)
    print("the removal tower for SU(2)+3")
    print("=" * 68)
    for rung in matter_removal_tower(rd.su_2(), (1,), nf=3):
        print(f"  {rung!r}")
