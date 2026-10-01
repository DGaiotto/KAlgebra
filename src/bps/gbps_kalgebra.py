"""`GBPSKAlgebra` — a BPS-chart `KAlgebra` with non-Abelian flavour `G`.

Name ruled by the author, 2026-09-14 (*"I think a GBPSKAlgebra class would be
fine"*), after the assessment that a `∏_a SU(N_a)`-flavoured BPS chart is well
posed (a probe in the source repository: `σ`, `F` and the
structure constants are all equivariant for the node-permutation action
realising `W(G)`, 115/115).

⚠ **The letter `G` carries two meanings across tiers.**  In the abelianized
tier the `G` prefix is the **gauge** group — `PureGAbeKAlgebra` at any
`RootDatum`, `GNAbeKAlgebra` for `(G, N)`.  Here it is the **flavour** group,
following the author's own `\\cE_G` throughout this construction.  Flagged rather
than silently mixed; the ruling is the author's.

The labels
----------

A canonical label is a pair **`(γ, r)`** — a reduced gauge charge and an irrep
of `G` — and the element is

    L_{(γ,r)}  =  Σ_{w ∈ r}  L_{(γ, w)}

summed over the weight diagram of `r`, the `L_{(γ,w)}` being the canonical basis
of the chart on **`Γ_gauge ⊕ P`** — the BPS quiver whose charge lattice has the
flavour weight lattice `P = ⊕_a P(SU(N_a))` as a FACTOR, its node charges being
`e_a ⊕ w` for the weights `w` of node `a`'s rep
(`FlavouredQuiver.gauge_weight_chart`).  This is the repo's flavour-lift
coordinate `L_label = χ_w·L_section` made concrete: the flavour character `χ_r`
is central, and multiplying the section by it produces the labelled canonical.

That this is the right index set was **measured, not assumed**, against the
hand-written one-`SU(2)` instance then at `src/cone/a1dn_kalg.py::A1DnKAlg`
(the `D_3`-chamber torus, now the archived tree): there
`L_{(0,1,0)}² = L_{(0,1,1)} + L_{(0,2,0)}` — two terms, Clebsch–Gordan — where the
unfolded chart gives the single term `L_{(0,2,0)}`.  So an `A1DnKAlg` basis
element is not one weight but a whole rep, and the flavoured basis is indexed by
`(charge, irrep)`.  `A1DnKAlg(3)` is consequently this module's acceptance test;
since 2026-09-23 it is the rebuilt class on the curves of the triangle with one
interior puncture, and products, traces and `ρ` all agree with it through a
test-local table of the curves' gauge charges (`tests/test_gbps_kalgebra.py`).

**There is no N-ality condition**.  An earlier version of this module built the chart on the **unfolded**
quiver instead, whose node charges span only a finite-index sublattice of
`Γ_gauge ⊕ P` — for `SU(2)` they generate the root `2ω` and never the fundamental
weight `ω`.  That index is what produced the N-ality condition
`γ_a ≡ Σ(weight coordinates) (mod N_a)`, and with it three defects that all
dissolve in the correct frame: no flavour-trivial section at a charge of non-zero
N-ality (so `forget()` mis-weighted), a flavour ring that embedded only in
N-ality `0` (so `embed_R` could not be written), and a factor of `N` between the
chart's `ker(B)` coordinates and `FlavourSpace` weights.  Here every `(γ, w)` is a
charge, so every irrep lives over every gauge charge, `L_{(γ,r)} = χ_r·L_{(γ,triv)}`
at every `γ`, and `embed_R` is total.

What is native and what is transported
--------------------------------------

Native, and the reason the class exists: the **coefficient ring** is
`⊗_a R(SU(N_a))` rather than the `U(1)^f` the unfolded chart carries, the labels
are `(γ, r)`, and multiplication **fuses irreps** — so the non-Abelian flavour is
the presentation rather than something recovered afterwards by un-branching
(`flavour_enhancement`, the inverse of `base_change(restriction_hom)`).

Transported, and said plainly rather than dressed up: `multiply`, `rho` and
`trace` are computed by expanding onto the unfolded chart, using **its**
operations, and collecting the result back into irreps.  That is legitimate here
for a measured reason — the unfolded chart's `σ`, `F` and structure constants are
equivariant, so they descend — and it is the same pattern `SkeinKAlgebra` uses
(multiply native, `ρ`/trace transported from the certified BPS twin).  The
independent flavoured engines cross-check the parts they cover: `S`
(`flavoured_factor_spectrum`), `F` (`flavoured_f_solver`) and the spec
(`flavoured_spec`).

Prior art, and what is new
--------------------------

`src/bps/gf_bps_kalgebra.py::GfBPSKAlgebra` is the established
G-flavoured BPS realisation, and it is the same construction: the Weyl-invariant
subalgebra of a `BPSKAlgebra`, canonical labels the dominant `W`-orbit
representatives, `L_γ` the Weyl character in the F-basis, `multiply` by lifting
to F-elements and peeling the dominant character out, `ρ` the BPS half-monodromy
then canonicalise, and the trace un-branched into the rep ring.  Its `SU(2)` and
`SU(3)` instances are hand-written matrices, and `gauge_weight_chart()` derives
exactly those from the flavoured quiver — asserted in
`tests/test_gbps_kalgebra.py`.  Its stated scope is **a single simple factor**
(rings `SU2`/`SO3`/`SU3`).

What this module adds is the **product** flavour group `∏_a SU(N_a)` at any
`N_a`, read off a `FlavouredQuiver` rather than supplied as matrices, together
with the flavoured `S` / `F` / spec engines it cross-checks against
(`flavoured_factor_spectrum`, `flavoured_f_solver`, `flavoured_spec`).  It is
NOT a first implementation of G-flavoured BPS charts, and should not be read as
one.  (`flavour_enhancement` and `skein_sphere/su2n_flavour` are a different
thing again — deprecated trace-level recognition diagnostics whose `multiply` is
a pass-through and which deliberately do not reorganise the basis.)

Scope
-----

Each node carries an irrep of its **own** simple factor (the author's *"groups of
`N_a` nodes ⇒ `SU(N_a)`"*); a shared factor is refused, as it is by
`FlavouredQuiver.canonical_key`, because the node colours stop being a complete
invariant there.
"""

from __future__ import annotations

import itertools
from typing import Sequence

from kalgebra import Element, KAlgebra
from laurent_poly import LaurentPoly
from zplus_ring import RElement, RPowerSeries

from flavoured_factor_spectrum import FlavouredQuiver, Irrep, Vec, Weight

Label = tuple[Vec, Irrep]


class GBPSKAlgebra(KAlgebra):
    """The BPS chart of a flavoured quiver, presented over `⊗_a R(SU(N_a))`.

    Parameters
    ----------
    quiver
        A `FlavouredQuiver` — the reduced exchange matrix plus the node reps.
    max_irrep
        Bound on the irreps enumerated at a charge (highest-weight entries).
        Labels beyond it are still legal; this only bounds `basis_iter` and the
        re-collection of a product.
    """

    def __init__(self, quiver: FlavouredQuiver, *, max_irrep: int = 6):
        if quiver.shares_a_flavour_factor():
            raise NotImplementedError(
                "GBPSKAlgebra: a flavour factor shared between nodes is out of "
                "scope — see FlavouredQuiver.canonical_key")
        from snf_kernel import integer_kernel_and_section
        ker, _sec = integer_kernel_and_section(
            [list(row) for row in quiver.pairing])
        if ker and not quiver.flavour.is_trivial():
            raise NotImplementedError(
                "GBPSKAlgebra: the REDUCED pairing is degenerate "
                f"(rk ker B = {len(ker)}), so the chart on `Γ_gauge ⊕ P` "
                "carries an abelian flavour from ker(B) ON TOP OF the weight "
                "lattice, and `coefficient_ring()` would report only the "
                "non-Abelian part.  The honest ring is "
                "`TensorZPlusRing([AbelianZPlusRing(rk ker B), ⊗_a R(SU(N_a))])` "
                "— the repo's own ring-growth rule — and the SECTIONS must strip "
                "`γ`'s kernel component too, which is not built.  Refused rather "
                "than silently discarding the abelian half: the residual "
                "direction projects to flavour weight 0, so it would be "
                "conflated with the trivial character, and "
                "`verify_orthonormality` does NOT catch it (it tests only the "
                "χ₀ component).  Note any antisymmetric pairing of ODD rank is "
                "degenerate, so this is the common case at odd reduced rank, "
                "not an edge case.")
        self.quiver = quiver
        self.rank = quiver.rank
        self.max_irrep = int(max_irrep)
        self._B, self._charges, self._labels = quiver.gauge_weight_chart()
        self._unfolded = None
        self._ring = None
        self._chart_wts = None
        self._irreps = None

    # ----- the chart on `Gamma_gauge (+) P` ------------------------------
    @property
    def unfolded(self):
        """The ordinary `BPSKAlgebra` on `Γ_gauge ⊕ P` (lazy).

        The flavour weight lattice is a **factor** of the charge lattice, not a sublattice generated by differences of node
        charges — see `FlavouredQuiver.gauge_weight_chart`.
        """
        if self._unfolded is None:
            from bps_kalgebra import BPSKAlgebra
            self._unfolded = BPSKAlgebra(pairing=self._B,
                                         node_charges=self._charges)
        return self._unfolded

    # ----- labels --------------------------------------------------------
    def admissible_irreps(self, charge: Vec) -> list[Irrep]:
        """The irreps of `G` enumerated up to `max_irrep` — **every** one.

        With the flavour weight lattice a factor of the charge lattice there is
        no gauge-N-ality condition left to impose: every `(γ, w)` is a charge,
        so every irrep lives over every gauge charge.  The `charge` argument is
        kept because the enumeration is a property of the algebra at a charge,
        and earlier frames did filter here; it is now unused.
        """
        del charge                       # every irrep is admissible
        if self._irreps is None:
            F = self.quiver.flavour
            per_factor: list[list[tuple[int, ...]]] = []
            for i, N in enumerate(F.factors):
                if N <= 1:
                    per_factor.append([()])
                    continue
                parts = [()]
                for length in range(1, N):
                    for lam in itertools.combinations_with_replacement(
                            range(self.max_irrep, 0, -1), length):
                        parts.append(tuple(lam))
                per_factor.append(sorted(set(parts)))
            out: list[Irrep] = []
            for combo in itertools.product(*per_factor):
                try:
                    rep = F._normalise(tuple(combo))
                except ValueError:
                    continue
                if F.irrep_weights(rep) and rep not in out:
                    out.append(rep)
            self._irreps = out
        return list(self._irreps)

    def expand(self, label: Label) -> dict[Vec, int]:
        """`L_{(γ,r)}` as `{chart canonical label: multiplicity}`.

        On `Γ_gauge ⊕ P` this is the direct sum over `r`'s weight diagram: the
        chart charge is the concatenation `γ ⊕ w`, with no integrality side
        condition to satisfy (the earlier frame needed `enlarged_charge` here,
        which refused whenever `γ` and `w` disagreed mod `N`).
        """
        charge, rep = label
        F = self.quiver.flavour
        charge = tuple(int(x) for x in charge)
        if len(charge) != self.rank:
            raise ValueError(
                f"GBPSKAlgebra: gauge charge must have rank {self.rank}: {charge}")
        rep = F._normalise(rep)
        out: dict[Vec, int] = {}
        for w, mult in F.irrep_weights(rep).items():
            key = charge + tuple(w)
            out[key] = out.get(key, 0) + mult
        return out

    def _project(self, chart_label: Vec) -> tuple[Vec, Weight]:
        """A chart charge on `Γ_gauge ⊕ P` ↦ `(gauge charge, flavour weight)`."""
        g = self.rank
        lab = tuple(int(x) for x in chart_label)
        return lab[:g], lab[g:]

    def _collect(self, terms: dict[Vec, LaurentPoly]) -> Element:
        """Unfolded coefficients ↦ an `Element` on `(charge, irrep)` labels.

        Groups by reduced charge, then decomposes the weight function into
        irreps one `\\fq`-power at a time.  Honest-fails (through
        `FlavourSpace.decompose`) if a weight function is not `W`-invariant —
        which would mean the product had left the flavoured span, a real finding
        rather than something to round away.
        """
        F = self.quiver.flavour
        by_charge: dict[Vec, dict[int, dict[Weight, int]]] = {}
        for lab, poly in terms.items():
            if poly.is_zero():
                continue
            charge, weight = self._project(lab)
            slot = by_charge.setdefault(charge, {})
            for e, c in poly._coeffs.items():
                if c:
                    slot.setdefault(e, {})[weight] = \
                        slot.setdefault(e, {}).get(weight, 0) + c
        out: dict[Label, LaurentPoly] = {}
        for charge, per_power in by_charge.items():
            for e, fn in per_power.items():
                for rep, mult in F.decompose(fn).items():
                    if not mult:
                        continue
                    key = (charge, rep)
                    prev = out.get(key, LaurentPoly({}))
                    out[key] = prev + LaurentPoly({e: mult})
        return Element({k: v for k, v in out.items() if not v.is_zero()})

    # ----- the KAlgebra contract ----------------------------------------
    def coefficient_ring(self):
        """`⊗_a R(SU(N_a))` — the non-Abelian flavour ring, natively.

        Taken from the flavour datum (`FlavourSpace.zplus_ring`) rather than
        assembled here, so one product-`Z₊`-ring object serves the whole
        construction.  An earlier version built it pairwise in a loop, which
        both duplicated the datum's own factor list and ignored
        `TensorZPlusRing`'s n-ary constructor.
        """
        if self._ring is None:
            self._ring = self.quiver.flavour.zplus_ring()
        return self._ring

    def identity(self) -> Label:
        return (tuple([0] * self.rank), self.quiver.flavour.trivial_irrep())

    def multiply(self, a: Label, b: Label) -> Element:
        """`L_a · L_b`, with the irreps **fusing**.

        Expand both onto the unfolded chart, multiply there, collect back.  The
        fusion is not imposed: it falls out of the collection, because the
        product's weight function decomposes into whatever irreps it contains.
        """
        terms: dict[Vec, LaurentPoly] = {}
        for ua, ma in self.expand(a).items():
            for ub, mb in self.expand(b).items():
                prod = self.unfolded.multiply(ua, ub)
                for lab, poly in prod.terms.items():
                    scaled = poly * (ma * mb) if ma * mb != 1 else poly
                    prev = terms.get(lab)
                    terms[lab] = scaled if prev is None else prev + scaled
        return self._collect(terms)

    def rho(self, a: Label) -> Label:
        """`ρ` — transported through the unfolded chart's `σ`.

        Legitimate because `σ` is equivariant for the node-permutation action
        (measured, a probe in the source repository), so it
        carries whole reps to whole reps and descends to `(γ, r)`.
        """
        return self._transport(a, self.unfolded.rho)

    def rho_inverse(self, a: Label) -> Label:
        return self._transport(a, self.unfolded.rho_inverse)

    def _transport(self, a: Label, op) -> Label:
        F = self.quiver.flavour
        images = [op(lab) for lab in self.expand(a)]
        charges = {self._project(x)[0] for x in images}
        if len(charges) != 1:
            raise ValueError(
                f"GBPSKAlgebra: the image of {a} spans several reduced charges "
                f"{sorted(charges)} — it is not a single rep")
        fn: dict[Weight, int] = {}
        for lab, mult in zip(images, self.expand(a).values()):
            w = self._project(lab)[1]
            fn[w] = fn.get(w, 0) + mult
        decomposed = F.decompose(fn)
        if len(decomposed) != 1:
            raise ValueError(
                f"GBPSKAlgebra: the image of {a} is not a single irrep: "
                f"{decomposed}")
        (rep, mult), = decomposed.items()
        if mult != 1:
            raise ValueError(f"GBPSKAlgebra: image of {a} has multiplicity {mult}")
        return (charges.pop(), rep)

    def trace(self, a: Label, K: int = 20) -> RPowerSeries:
        """`Tr(L_a) ∈ R(G)((\fq))` — the unfolded traces summed over the rep's
        weights, then **un-branched into `coefficient_ring()`**.

        The lift is not cosmetic.  The unfolded chart's traces are valued in its
        own Cartan ring `Z[U(1)^f]`, and returning them verbatim breaks the
        contract's `Tr(L_a) ∈ R((\fq))` for `R = coefficient_ring()`: every
        derived consumer that pairs two traces — `inner_product`,
        `verify_orthonormality`, i.e. the orthonormality axiom itself — then
        dies on an `RPowerSeries` ring mismatch rather than computing.  Each
        `\fq`-power's weight function is `W`-invariant (it is a sum over whole
        weight diagrams) and so decomposes into characters; a function that is
        not is an honest failure out of `FlavourSpace.decompose`, since it would
        mean the trace had left the flavoured span.
        """
        total = None
        for lab, mult in self.expand(a).items():
            piece = self.unfolded.trace(lab, K)
            for _ in range(mult):
                total = piece if total is None else total + piece
        if total is None:
            raise ValueError(f"GBPSKAlgebra: empty expansion for {a}")
        return self._lift_series(total, K)

    def _chart_flavour_weights(self) -> list:
        """The `FlavourSpace` weight of each unfolded-chart flavour direction.

        The two flavour coordinate systems are **not** the same lattice and must
        be calibrated rather than identified.  The chart keys its flavour ring
        `Z[U(1)^f]` by coordinates in an SNF basis of `Γ_f = ker(B)`, i.e. by
        *primitive* kernel generators, while `FlavourSpace` weights are the
        `to_abelian` coordinates of the `SU(N_a)` weight lattices.  For one
        `SU(2)` the two differ by a factor of two — the doublet's weights are
        `±1`, differing by `2`, where the chart's primitive generator is `1` —
        so identifying them silently decomposes the wrong weight function.

        Read off through the chart's **public** `embed_R`, which maps a flavour
        character to the canonical element at that kernel charge, so the basis
        is obtained without reaching into the chart's internals.
        """
        if self._chart_wts is None:
            F = self.quiver.flavour
            U = self.unfolded
            R = U.coefficient_ring()
            rank = getattr(R, "rank", 0)
            wts = []
            for j in range(rank):
                key = tuple(1 if i == j else 0 for i in range(rank))
                elt = U.embed_R(R.basis_element(key))
                labs = list(elt.terms)
                if len(labs) != 1:
                    raise ValueError(
                        "GBPSKAlgebra: the chart's flavour character "
                        f"{key} is not a single canonical: {labs}")
                wts.append(F.add(F.zero, self._project(labs[0])[1]))
            self._chart_wts = wts
        return self._chart_wts

    def _key_to_weight(self, key) -> Weight:
        """An unfolded-chart flavour key ↦ the `FlavourSpace` weight it denotes."""
        F = self.quiver.flavour
        wts = self._chart_flavour_weights()
        out = F.zero
        for j, n in enumerate(tuple(key)):
            if n:
                out = F.add(out, F.scale(wts[j], n))
        return out

    def _lift_series(self, series: RPowerSeries, K: int) -> RPowerSeries:
        """An unfolded-chart series (`Z[U(1)^f]` coefficients) ↦ one over `R(G)`."""
        F = self.quiver.flavour
        R = self.coefficient_ring()
        out: dict[int, RElement] = {}
        for e, rc in series.coeffs.items():
            fn: dict[Weight, int] = {}
            for k, c in rc.terms.items():
                if c:
                    w = self._key_to_weight(k)
                    fn[w] = fn.get(w, 0) + c
            fn = {w: c for w, c in fn.items() if c}
            if not fn:
                continue
            terms: dict[object, int] = {}
            for rep, m in F.decompose(fn).items():
                if m:
                    key = F.irrep_to_ring_key(rep)
                    terms[key] = terms.get(key, 0) + m
            terms = {k: v for k, v in terms.items() if v}
            if terms:
                out[e] = RElement(R, terms)
        return RPowerSeries(R, out, K)

    def r_label_decompose(self, label: Label):
        """The flavour-lift coordinate `(section, R-basis-label)`.

        `L_{(γ,r)} = χ_r · L_{(γ, triv)}`, so the section is the same gauge
        charge carrying the trivial irrep and the second component is the
        **`coefficient_ring()` basis key** the contract requires — obtained
        through `FlavourSpace.irrep_to_ring_key`, since an `Irrep` (one slot per
        factor, trivial factors included) is a different object from a ring key
        and passing one silently produces ring-invalid data.

        With the flavour weight lattice a factor of the charge lattice the
        section exists at **every** gauge charge; in the earlier frame it did
        not at a charge of non-zero N-ality, and the fallback there returned the
        label as its own section with the trivial character — which made
        `forget()` weight a doublet by `1` instead of `dim = 2`.
        """
        charge, rep = label
        F = self.quiver.flavour
        return ((tuple(int(x) for x in charge), F.trivial_irrep()),
                F.irrep_to_ring_key(rep))

    def r_label_compose(self, section, r_basis_label) -> Label:
        """Inverse of `r_label_decompose`: `(γ, triv), χ_r ↦ (γ, r)`."""
        charge, _triv = section
        F = self.quiver.flavour
        return (tuple(int(x) for x in charge), F.ring_key_to_irrep(r_basis_label))

    def embed_R(self, r: RElement) -> Element:
        """`ι : R(G) ↪ A` — `χ_r ↦ L_{(0, r)}`, the zero-gauge-charge canonical.

        Total, because the flavour weight lattice is a factor of the charge
        lattice: every irrep sits over the zero gauge charge.  (In the earlier
        frame only the N-ality-`0` characters did, so the flavour ring embedded
        only in degree zero and this could not be written.)
        """
        R = self.coefficient_ring()
        if not isinstance(r, RElement) or r.ring != R:
            raise TypeError(
                "GBPSKAlgebra.embed_R: argument must be an RElement over "
                "coefficient_ring()")
        F = self.quiver.flavour
        zero = tuple([0] * self.rank)
        out = Element.zero()
        for key, c in r.terms.items():
            if c:
                out = out + Element.basis((zero, F.ring_key_to_irrep(key))) * c
        return out

    # ----- convenience ---------------------------------------------------
    def basis_iter(self, max_degree: int = 2):
        """Labels `(γ, r)` with `Σ|γ_a| ≤ max_degree`, admissible irreps only."""
        for combo in itertools.product(
                range(-max_degree, max_degree + 1), repeat=self.rank):
            if sum(abs(x) for x in combo) > max_degree:
                continue
            for rep in self.admissible_irreps(tuple(combo)):
                yield (tuple(combo), rep)
