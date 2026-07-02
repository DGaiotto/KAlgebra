"""`Grading` — a `Γ_RG`-grading on a `KAlgebra`, as sidecar data.

A `Grading` equips an *auxiliary* (IR) `KAlgebra` with a charge in a
lattice `Γ_RG` on each canonical-basis label, additive under
multiplication, plus a supplied **height** functional
`h : Γ_RG → Z` — an integral height functional (a linearized proxy for
`Im Z_γ`, the physical central charge).

Design decision: the grading is **sidecar
data, NOT a `KAlgebra` subclass**.  It is *flow-relative* — one algebra
admits many gradings (e.g. `Q_q(Γ)` graded by `Γ/Γ'` for any sublattice
`Γ'` of "small" charges) — so the grading belongs to the RG flow that
uses the algebra, not to the algebra's type.  This mirrors how the repo
carries optional structure as a `cone_data()` sidecar rather than baking
it into the hierarchy.

What the grading buys (consumed by `graded_rg_solver`):

* **`Γ_RG` is a bare lattice** — no pairing / inner product.  The charge
  `deg(L_b) ∈ Γ_RG` is *additive under multiplication*:
  `C^c_{ab} ≠ 0 ⇒ deg(c) = deg(a) + deg(b)` (`verify_additive`).  So the
  auxiliary is a `Γ_RG`-graded algebra `B = ⊕_p B_p`,
  `B_p · B_{p'} ⊆ B_{p+p'}`; the degree-0 part is the "small" surviving
  IR algebra.

* **Height-positivity.** `h` is a linear functional on `Γ_RG`, given by
  its coefficient vector (`h(p) = Σ_i height[i]·p[i]`), required strictly
  positive on every nonzero charge that actually appears.  A single such
  functional forces the appearing charges into a **pointed cone** (if
  `h(p)>0` and `h(−p)>0` for `p≠0`, contradiction), and it drives the
  `q`-expansion's convergence, so the RG-solver's cone-walk terminates /
  truncates.  This is the abstract replacement for BPS's
  pointed-positive-cone + doubly-tropical bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

Label = tuple
Charge = tuple  # an element of Γ_RG: a length-`rank` tuple of ints


@dataclass(frozen=True)
class Grading:
    """A `Γ_RG`-grading on a `KAlgebra`'s canonical basis.

    Parameters
    ----------
    rank
        Rank of the grading lattice `Γ_RG`; charges are length-`rank`
        integer tuples.
    deg
        Maps an auxiliary canonical-basis `Label` to its charge in
        `Γ_RG`.  Must be additive under the auxiliary's `multiply`
        (checked by `verify_additive`).
    height
        Coefficient vector of the integral height functional
        `h : Γ_RG → Z` (a linearized proxy for `Im Z_γ`, the physical
        central charge).  `h(p) = Σ_i height[i]·p[i]`.
        Must be strictly positive on every nonzero appearing charge
        (height-positivity).
    cone_gens
        Optional generators of the `Γ_RG` **positive cone** (the rays on
        which `S_RG` and the RG images are supported), each a length-`rank`
        charge tuple.  When supplied, `in_cone(p)` tests membership.  This
        lets a consumer (i) enumerate the grading-height window of `S_RG`
        (`∪ _s_rg_component(p)` over `{p ∈ cone : h(p) ≤ K}`) and (ii)
        recognise the **off-cone** labels produced by `ρ` (the deep-edge
        mirror artifacts) so the `solve_rg` completion test can filter them
        out.  Default `None` (no cone info — consumers fall back to height
        / cutoff-stability heuristics).
    """

    rank: int
    deg: Callable[[Label], Charge]
    height: tuple[int, ...]
    cone_gens: tuple[Charge, ...] | None = None

    def __post_init__(self) -> None:
        if self.rank < 1:
            # rk Γ_RG ≥ 1 — "the grading torus cannot be fully trivial"
            # (need at least one direction to carry the cone / height).
            raise ValueError(f"Grading.rank must be ≥ 1, got {self.rank}")
        if len(self.height) != self.rank:
            raise ValueError(
                f"height vector has length {len(self.height)}, "
                f"expected rank {self.rank}"
            )
        if self.cone_gens is not None:
            for g in self.cone_gens:
                if len(tuple(g)) != self.rank:
                    raise ValueError(
                        f"cone generator {g} has length {len(tuple(g))}, "
                        f"expected rank {self.rank}"
                    )

    # ----- positive cone --------------------------------------------------

    def in_cone(self, p: Charge) -> bool:
        """Is charge `p` in the `Γ_RG` positive cone (a non-negative integer
        combination of `cone_gens`)?  Requires `cone_gens`."""
        if self.cone_gens is None:
            raise ValueError(
                "Grading.in_cone needs cone_gens (the positive-cone "
                "generators); none were supplied to this Grading."
            )
        from lattice import cone_contains
        return cone_contains(tuple(p), [tuple(g) for g in self.cone_gens])

    def label_in_cone(self, label: Label) -> bool:
        """Whether a label's charge `deg(label)` is in the positive cone."""
        return self.in_cone(self.charge(label))

    # ----- charge + height ------------------------------------------------

    def charge(self, label: Label) -> Charge:
        """The `Γ_RG`-charge `deg(L_label)`, as a length-`rank` tuple."""
        c = tuple(self.deg(label))
        if len(c) != self.rank:
            raise ValueError(
                f"deg({label!r}) = {c} has length {len(c)}, "
                f"expected rank {self.rank}"
            )
        return c

    def h(self, p: Charge) -> int:
        """Height `h(p) = Σ_i height[i]·p[i]` of a charge `p ∈ Γ_RG`."""
        return sum(hi * pi for hi, pi in zip(self.height, p))

    def height_of(self, label: Label) -> int:
        """Height of a canonical-basis label: `h(deg(L_label))`."""
        return self.h(self.charge(label))

    # ----- axiom verifiers ------------------------------------------------

    def verify_additive(self, algebra, a: Label, b: Label) -> bool:
        """Grading is an algebra grading: every `L_c` appearing in
        `L_a · L_b` has `deg(c) = deg(a) + deg(b)`."""
        prod = algebra.multiply(a, b)
        target = self.charge_sum(self.charge(a), self.charge(b))
        for c, coeff in prod.terms.items():
            if coeff.is_zero():
                continue
            if self.charge(c) != target:
                return False
        return True

    def verify_height_positive(self, charges: Iterable[Charge]) -> bool:
        """Height-positivity over a set of charges: `h(p) > 0` for every
        nonzero `p` (and `h(0) = 0`).  Pass the charges that actually
        appear (e.g. the support of `S_RG` and the RG images)."""
        zero = (0,) * self.rank
        for p in charges:
            p = tuple(p)
            if p == zero:
                continue
            if self.h(p) <= 0:
                return False
        return True

    # ----- small lattice helpers (Γ_RG arithmetic) -----------------------

    @staticmethod
    def charge_sum(p: Charge, q: Charge) -> Charge:
        return tuple(pi + qi for pi, qi in zip(p, q))

    @staticmethod
    def charge_neg(p: Charge) -> Charge:
        return tuple(-pi for pi in p)

    def zero_charge(self) -> Charge:
        return (0,) * self.rank


def trivial_grading(rank: int = 1) -> Grading:
    """The grading sending every label to the zero charge, with the
    coordinate-sum height.  Useful as a degenerate baseline / test stub
    (every product is trivially additive)."""
    return Grading(rank=rank, deg=lambda label: (0,) * rank,
                   height=(1,) * rank)
