"""The type-A seam: how the general `(G, N)` machinery relates to the
purpose-built `UNNfKAlgebra`.

This is the integration point between the two `AbeKAlgebra` families that both
present U(N)+N_f:

  * `GNAbeKAlgebra(u_n(N), fund, nf=nf)` — the **group-general** construction
    specialised to type A (built here by the `_un_fund` helper; there is no
    `UNFund…` class, because a special name is earned by an optimized algorithm
    and this route has none).  Flavour ring `R(U(nf))` (ruling TM7).
  * `UNNfKAlgebra(N, nf)` — the **purpose-built** type-A class (ruling T4).
    Flavour ring `R(SU(nf))` (rulings D5/D8b).

They present *almost* the same algebra, and the word "almost" is the whole
content of this module.

What the relation is — and three things it is NOT
------------------------------------------------
The relation is a **surjective algebra homomorphism**, obtained by setting the
central (`det`) flavour fugacity to 1:

    GNAbeKAlgebra(u_n(N), fund, nf)   ──φ──▶   UNNfKAlgebra(N, nf)

with `φ` the identity on gauge labels and `zplus_ring.un_to_sun_hom(nf)` on the
flavour irrep.  That is exactly ruling **D8b**: the central direction is removed
by *specializing a fugacity*, never by identifying labels — and specialization is
an algebra homomorphism, so associativity, bar and orthonormality survive it.

Three tempting stronger claims are **false**, and each was ruled out by
measurement rather than by taste:

1. *"they are isomorphic"* — no: `φ` is not injective.  At `nf = 1` the source
   carries a full `R(U(1))` of flavour-charged canonicals
   (`L_{(g,(k,))} = μ^k·L_{(g,(0,))}`) and the target's ring is
   `TrivialZPlusRing`, so an entire `Z[μ^±]`-worth of distinct canonicals
   collapses onto each target canonical.

2. *"they are isomorphic after adjoining a spectator `U(1)`"*, i.e.
   `GN(u_n(N),fund,nf) ≅ UNNfKAlgebra(N,nf).add_flavour(AbelianZPlusRing(1))` — false,
   and it fails **twice over**, for two independent reasons:

   *(a) at `nf ≥ 2`, already on structure constants.*  `add_flavour` builds the
   *untwisted* tensor product `R(SU(n)) ⊗ Z[det^±]`, whereas `R(U(n))` is the
   `Z_n`-invariant **sublattice** of it — `U(n) = (SU(n) × U(1))/Z_n`.  Measured
   witness at `U(2)+2`: the flavour fundamental squares to
   `χ_{(1,0)}² = χ_{(2,0)} + χ_{(1,1)}` and `χ_{(1,1)}` is `det`, i.e. det-degree
   1, while in the tensor product both factors have det-degree 0 and the product
   must too.  Every disagreement observed was exactly this one-unit det shift
   (9 of 36 products, and 3 of 9 ρ images).

   *(b) at `nf = 1`, on the **trace** — the sharper and more interesting
   failure.*  Multiplicativity, ρ-equivariance, the unit and the round trip all
   agree there (measured: 36/36 products, 6/6 ρ), so it looks like an iso until
   the trace is checked.  It is not: the matter `μ` is a **fugacity grading
   matter zero modes**, so it appears in the trace of a flavour-*neutral*
   canonical — measured, the `U(2)+1` Wilson line has
   `Tr = −μ⁻¹𝖖 + 2μ⁻¹𝖖³` (physically: a gauge Wilson line screens against the
   matter and picks up flavour charge).  `UNNfKAlgebra` has *specialized that
   fugacity to 1* (D8b), giving `−𝖖 + 2𝖖³`, and a spectator `U(1)` adjoined
   afterwards is **inert** — it grades a new label slot and cannot resurrect the
   matter grading.  Since `trace` is a contract primitive, this is not an
   isomorphism of `KAlgebra`s.

   The moral: the difference between the two type-A classes is not a spectator
   flavour that could be tensored back on.  It is a genuine **μ-refinement of the
   index**, and the only structure-preserving map between them is the
   specialization that forgets it.

3. *"`base_change` implements it"* — no: `base_change(φ_R)` is a
   *coefficient-only* pushforward, and on both these realisations the flavour
   lives in a **label slot**, not in the coefficients (`r_label_decompose`
   returns `(gauge section, irrep)`).  So the specialization has to act on
   labels, which is what this module does.

Why the honest version is still worth having
--------------------------------------------
Because it is what certifies that the group-general construction did not quietly
build a *different* algebra in the one corner where an independent oracle exists.
`φ` being a homomorphism on the nose — unit, multiplicativity, ρ-equivariance and
trace compatibility, including on det-charged labels — pins the general machinery
against `UNNfKAlgebra` without pretending the flavour conventions agree.

Measured (the suite in the source repository): U(2)+1 36/36 products, U(2)+2
**81/81**, U(3)+1 36/36; ρ-equivariance 6/9/6; trace compatibility on every label
tried, *after* pushing the source trace through `un_to_sun_hom` — which is the
whole point, since that hom is what sets the central fugacity to 1.  Zero
disagreements.  The refuted spectator-iso reading is pinned in the same suite so
it cannot quietly come back.

Run `PYTHONPATH=$(ls -d src/* | paste -sd:) python3 src/gn/g_matter_un_nf_seam.py` for a tour.
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

from kalgebra import Element
from zplus_ring import un_to_sun_hom

from gn_abe_kalgebra import GNAbeKAlgebra
from un_nf_kalgebra import UNNfKAlgebra


def _un_fund(N: int, nf: int, **kw) -> GNAbeKAlgebra:
    """`U(N)` with `nf` fundamentals through the general `(G, N)` class.

    There is deliberately no `UNFundAbeKAlgebra` class: a special name is earned
    by an optimized algorithm (user ruling, 2026-07-28), and the optimized type-A
    class is `UNNfKAlgebra` — which is exactly the target of the seam below."""
    import root_datum as rd
    from matter_wrq_torus import defining_weight
    datum = rd.u_n(int(N))
    return GNAbeKAlgebra(datum, tuple(defining_weight(datum)), nf=int(nf), **kw)


__all__ = ["UNNfSpecialization"]


class UNNfSpecialization:
    """The surjection `GNAbeKAlgebra(u_n(N), fund, nf) ↠ UNNfKAlgebra(N, nf)` that sets
    the central `det` flavour fugacity to 1 (ruling D8b).

    Deliberately **not** a `KAlgebraIso`: it has no inverse, and pretending
    otherwise is the error this class exists to prevent (see the module
    docstring).  It carries the same *shape* of verifier battery — unit,
    multiplicativity, ρ-equivariance, trace compatibility — minus everything
    that presupposes bijectivity.
    """

    def __init__(self, N: int, nf: int = 1, **kw) -> None:
        self.N = int(N)
        self.nf = int(nf)
        self.source = _un_fund(self.N, self.nf, **kw)
        self.target = UNNfKAlgebra(self.N, self.nf)
        self._h = un_to_sun_hom(self.nf)

    def __repr__(self) -> str:
        return (f"UNNfSpecialization(U({self.N})+{self.nf}: "
                f"{self.source.coefficient_ring()} → "
                f"{self.target.coefficient_ring()})")

    # ----- the map -------------------------------------------------------

    def flavour_hom(self):
        """The `RingHom` on flavour rings: `R(U(nf)) → R(SU(nf))`
        (`TrivialZPlusRing` at `nf = 1`)."""
        return self._h

    def apply(self, label):
        """A source canonical label → the target canonical label.

        The gauge part is carried across unchanged (the two classes were measured
        to use the same lower-Kapustin `(m, e)` frame — ruling TM6,
        `chart == UNNfKAlgebra.chart`); the flavour irrep goes through
        `un_to_sun_hom`, whose image is always a *single* irrep with
        multiplicity 1 because the quotient is by a 1-dimensional rep."""
        g, lam = label
        img = self._h.apply_basis(lam)
        terms = {b: c for b, c in img.terms.items() if c}
        if len(terms) != 1:
            raise AssertionError(
                f"{self!r}.apply: un_to_sun_hom sent {lam} to {len(terms)} "
                "irreps — the central quotient must be by a 1-dim rep")
        (w, c), = terms.items()
        if c != 1:
            raise AssertionError(
                f"{self!r}.apply: multiplicity {c} ≠ 1 on {lam}")
        gauge = (tuple(g[0]), tuple(g[1]))
        return self.target._mk_label(gauge, w if self.nf > 1 else ())

    def apply_element(self, x: Element) -> Element:
        """`φ` extended `𝖖`-linearly.  Note this genuinely **merges** terms:
        source canonicals differing by a `det` twist have the same image, so the
        output can have fewer labels than the input.  That merging is the
        surjectivity, not a bug."""
        out: dict = {}
        for lab, c in x.terms.items():
            if hasattr(c, "is_zero") and c.is_zero():
                continue
            k = self.apply(lab)
            out[k] = (out[k] + c) if k in out else c
        return Element({k: v for k, v in out.items()
                        if not (hasattr(v, "is_zero") and v.is_zero())})

    # ----- verifiers -----------------------------------------------------

    def verify_unit(self) -> bool:
        """`φ(1) = 1`."""
        return self.apply(self.source.identity()) == self.target.identity()

    def verify_multiplicative(self, a, b) -> bool:
        """`φ(L_a · L_b) = φ(L_a) · φ(L_b)` — the homomorphism property, and the
        leg that carries the evidence."""
        lhs = self.apply_element(self.source.multiply(a, b))
        rhs = self.target.multiply(self.apply(a), self.apply(b))
        return _clean(lhs) == _clean(rhs)

    def verify_rho(self, a) -> bool:
        """`φ(ρ_S(a)) = ρ_T(φ(a))` — ρ is a label permutation on both sides, so
        this is a label identity."""
        return self.apply(self.source.rho(a)) == self.target.rho(self.apply(a))

    def verify_trace(self, a, K: int = 4) -> bool:
        """`h(Tr_S(L_a)) = Tr_T(L_{φ(a)})`, `h` the flavour ring hom — the trace
        is computed independently on the two sides (each chart-side, from its own
        Schur-measure residue), so this is a genuine cross-check."""
        return self._h.apply_RPowerSeries(self.source.trace(a, K=K)) == \
            self.target.trace(self.apply(a), K=K)

    def verify_not_injective(self) -> bool:
        """Positively confirm the thing the docstring claims: two *distinct*
        source canonicals with the same image.  A `det`-twisted pair does it, so
        this is only meaningful — and only defined — because the source keeps the
        central flavour direction the target specializes away."""
        idn = self.source.identity()
        g, lam = idn
        det = tuple(x + 1 for x in lam)          # λ + (1,…,1) = λ ⊗ det
        twisted = (g, det)
        return twisted != idn and self.apply(twisted) == self.apply(idn)

    def sample_labels(self):
        """A small spanning-ish sample: identity, a Wilson line, a monopole, each
        at trivial / fundamental / `det` flavour."""
        A = self.source
        d = A.datum
        z = (0,) * d.dim
        fund = tuple(A.matter[0])
        mono = tuple([-1] + [0] * (d.dim - 1))
        flavs = [(0,) * self.nf, (1,) + (0,) * (self.nf - 1)]
        if self.nf > 1:
            flavs.append((1,) * self.nf)          # det
        out = []
        for m, e in [(z, z), (z, fund), (mono, z)]:
            for w in flavs:
                out.append(A.fold(m, e, w))
        return out

    def verify_all(self, labels=None, K: int = 4) -> dict:
        """The whole battery; returns a `{check: bool}` report."""
        labels = list(labels if labels is not None else self.sample_labels())
        return {
            "unit": self.verify_unit(),
            "not_injective": self.verify_not_injective(),
            "rho": all(self.verify_rho(a) for a in labels),
            "multiplicative": all(self.verify_multiplicative(a, b)
                                  for a in labels for b in labels),
            "trace": all(self.verify_trace(a, K=K) for a in labels),
        }


def _clean(x: Element) -> dict:
    return {lab: c for lab, c in x.terms.items()
            if not (hasattr(c, "is_zero") and c.is_zero())}


if __name__ == "__main__":
    for N, nf in [(2, 1), (2, 2), (3, 1)]:
        S = UNNfSpecialization(N, nf)
        print("=" * 70)
        print(f"{S!r}")
        print("=" * 70)
        rep = S.verify_all()
        for k, v in rep.items():
            print(f"   {k:16s} {v}")

    print()
    print("=" * 70)
    print("why there is no spectator-U(1) iso, even at N_f = 1: the trace")
    print("=" * 70)
    S = UNNfSpecialization(2, 1)
    A = S.source
    lab = A.fold((0, 0), (1, 0))          # the Wilson line, flavour-NEUTRAL
    print(f"  L = {lab}  (flavour-neutral)")
    print(f"    Tr_GN  = {A.trace(lab, K=4)}")
    print(f"    Tr_UNNf    = {S.target.trace(S.apply(lab), K=4)}")
    print("  the matter μ grades matter zero modes, so it shows up in the trace")
    print("  of a flavour-neutral canonical; UNNfKAlgebra specialized it to 1")
    print("  (D8b), and an adjoined spectator U(1) is inert.")
