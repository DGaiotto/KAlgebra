"""Z₊-rings (Lusztig-Ostrik) — the natural class of coefficient rings for
K-theoretic Coulomb branch algebras with flavour.

A *Z₊-ring* is a unital ring R together with a Z-basis B such that:

  1. 1 ∈ B (the identity is a basis element);
  2. structure constants are non-negative integers,
       b · b' = Σ_{b''} N^{b''}_{b, b'} · b''   with   N^{b''}_{b, b'} ∈ Z_{≥0};
  3. there is an involution ⋆ : B → B with (b · b')⋆ = b'⋆ · b⋆ and 1⋆ = 1.

By Lusztig-Ostrik (and Tannakian reconstruction), commutative Z₊-rings of
finite type with rigid duals are *exactly* the Grothendieck rings R(G) of
representation categories of compact (algebraic) groups G.  In our setting
these are the coefficient rings of FKAlgebras: the canonical basis of R(G)
is the set of irreducible representations, multiplication is the tensor
product (with non-negative multiplicities given by Clebsch-Gordan), the
identity is the trivial rep, and ⋆ is the duality V ↦ V*.

Concrete cases shipped here:

  * `TrivialZPlusRing()`: R = Z, basis = {()}, ⋆ trivial.  Used by an
    ordinary unflavoured KAlgebra (G trivial).
  * `AbelianZPlusRing(rank=n)`: R = R(U(1)^n) = Z[μ_1^±, …, μ_n^±], basis =
    Z^n (characters), ⋆(f) = -f (the rep-ring duality `V ↦ V*`).  Used by
    FKAlgebras whose flavour symmetry has been broken to its maximal
    torus — the typical mass-deformation picture, and what BPS-quiver
    realizations produce by default.
  * `SU2ZPlusRing()`: R = R(SU(2)) = Z[χ_1] ⊂ Z[μ^±], basis = ℕ (the
    irreducible characters χ_n of highest weight n / spin n/2 / dim n+1),
    multiplication = Clebsch-Gordan
    `χ_n · χ_m = χ_{n+m} + χ_{n+m-2} + … + χ_{|n-m|}`, ⋆ = id (SU(2)
    reps are self-dual).  Embeds in `AbelianZPlusRing(rank=1)` as the
    Weyl-invariant subring via `χ_n ↦ μ^n + μ^{n-2} + … + μ^{-n}`
    (weight convention: μ carries weight 1 = the spin-1/2 fundamental
    weight; in particular χ_1 = μ + μ⁻¹, χ_2 = μ² + 1 + μ⁻²).
  * `SO3ZPlusRing()`: R = R(SO(3)) = Z[χ_1] ⊂ Z[μ^±], basis = ℕ_0
    (the irreducible characters χ_j of spin j / dim 2j+1; only integer
    spins, since SO(3) reps lift to even-n SU(2) reps),
    multiplication = Clebsch-Gordan
    `χ_j · χ_k = χ_{j+k} + χ_{j+k-1} + … + χ_{|j-k|}` (step 1 in j),
    ⋆ = id.  Embeds in `AbelianZPlusRing(rank=1)` as the Weyl-invariant
    subring via `χ_j ↦ μ^j + μ^{j-1} + … + μ^{-j}` (weight convention:
    μ carries weight 1 of the SO(3) Cartan U(1); χ_1 = μ + 1 + μ⁻¹ is
    the 3-dim vector rep).  Embeds in `SU2ZPlusRing` as the even-n
    subring via `χ_j^{SO(3)} ↦ χ_{2j}^{SU(2)}` (set μ_{SO(3)} = μ_{SU(2)}²).
    Used when the U(1) flavour symmetry of an FKAlgebra is enhanced to
    SO(3) — the natural enhancement when the σ-symmetrised lattice
    points carry integer weights, e.g. the σ-invariant subalgebra of a
    quantum torus where σ exchanges two generators at equal flavour
    weight (the simplest case: B = [[0,1,1],[-1,0,0],[-1,0,0]],
    σ : g_2 ↔ g_3).

    Two involutions on `R[q^±]` are kept structurally distinct:

      - **The FKAlgebra's bar** acts only on q: `q ↔ q^{-1}`, R untouched.
        Structure constants are palindromic in q; μ-dependence is
        transparent to bar.  Implemented by `RLaurent.bar()`.

      - **The Z₊-ring's ⋆** is the Lusztig-Ostrik rep-ring duality on
        R itself: `μ^f ↦ μ^{-f}`.  *Not* the FKAlgebra's bar.  This
        coincides with the action of `ρ` (the canonical algebra
        automorphism) restricted to the central flavour subalgebra:
        in any QT realization `ρ(X_γ) = X_{-γ}`, so on central
        elements `μ^f = X_{γ_f}` we get `ρ(μ^f) = μ^{-f}`.  Implemented
        by `RElement.star()` and surfaced through `ZPlusRing.star_basis`.

  * `SU3ZPlusRing()`: R = R(SU(3)) = ⊕_{(p,q) ∈ ℕ²} Z·χ_{(p,q)}, basis =
    Dynkin labels (p, q) (dim (p+1)(q+1)(p+q+2)/2), multiplication via
    Klimyk's formula + Freudenthal weight multiplicities, ⋆(p,q) = (q,p)
    (complex conjugation, 3 ↔ 3̄).  Embeds in `AbelianZPlusRing(rank=2)`
    as the S_3-Weyl-symmetric subring.  Used when a BPS quiver carries an
    S_3-orbit of three rays at the fundamental weights of SU(3); see
    `SU3BPSKAlgebra`.

The corresponding KAlgebra coefficient ring is `R[q^±]`, implemented here as
`RLaurent[R]`.  Two distinct involutions live on this ring:

  * The KAlgebra's **bar involution** on `R[q^±]` is `q ↔ q^{-1}` acting
    trivially on R.  Structure constants `C^c_{a,b}(q, μ)` of FKAlgebras
    are palindromic in q, with μ-dependence transparent to bar.  This is
    `RLaurent.bar()`.

  * The Lusztig-Ostrik **⋆ involution** on R is the rep-ring duality
    `V ↦ V*` — non-trivial in general (`μ^f ↦ μ^{-f}` on the abelian
    ring).  This is *not* the FKAlgebra's bar; the two are genuinely
    different operations and should not be conflated.  Structurally,
    Z₊-ring `⋆` coincides with the action of `ρ` (canonical algebra
    automorphism) restricted to the central flavour subalgebra: in any
    QT realization `ρ(X_γ) = X_{-γ}`, so on `μ^f = X_{γ_f}` we get
    `ρ(μ^f) = μ^{-f}`.  Surfaced as `RElement.star()` and through
    `ZPlusRing.star_basis`; used by FBPSKAlgebra to define `ρ`'s action
    on the central subalgebra.

Non-abelian rep rings are first-class: `SU2ZPlusRing`, `SO3ZPlusRing`, and
`SU3ZPlusRing` are shipped above, each fitting the same `ZPlusRing` protocol
with non-trivial (Clebsch-Gordan / Klimyk) multiplicities.  We don't
construct rep rings programmatically — adding a further non-abelian case
(higher-rank R(SU(N)), other simple groups) means hand-writing its basis
enumeration, structure constants, and ⋆.

References:
    V. Ostrik, "Module categories, weak Hopf algebras and modular invariants",
        Transform. Groups 8 (2003).
    G. Lusztig, "Leading coefficients of character values of Hecke algebras",
        Proc. Sympos. Pure Math. 47 (1987).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Hashable
from functools import lru_cache


BasisElement = Hashable


# ---------------------------------------------------------------------------
# ZPlusRing ABC
# ---------------------------------------------------------------------------


class ZPlusRing(ABC):
    """A unital Z₊-ring (Lusztig-Ostrik).

    Subclasses implement the three abstract primitives:
        one_basis, multiply_basis, star_basis.
    Element-level arithmetic is derived (via `RElement`).
    """

    @abstractmethod
    def one_basis(self) -> BasisElement:
        """The identity basis element  1 ∈ B."""

    @abstractmethod
    def multiply_basis(
        self, b1: BasisElement, b2: BasisElement,
    ) -> dict[BasisElement, int]:
        """Structure constants  b1 · b2 = Σ N^c_{b1, b2} · c .  All N values
        are non-negative integers; zero coefficients omitted."""

    @abstractmethod
    def star_basis(self, b: BasisElement) -> BasisElement:
        """The ⋆ involution on the basis (rep-ring duality V ↦ V*).
        Subclasses must satisfy  star_basis(star_basis(b)) == b  and
        star_basis(one_basis()) == one_basis() ."""

    @abstractmethod
    def dim(self, b: BasisElement) -> int:
        """The **augmentation** ε(b) ∈ ℤ_{≥1}: the dimension of the irrep `b`
        (= χ_b at the identity = the Frobenius–Perron dimension of the based
        ring; = the sum of torus-weight multiplicities).

        This is the per-basis datum of the **forgetful map** `ε : R → Z`
        (`augmentation()`) — physically *forget the flavour group `G_f`*
        (`K_𝖖[T; G_f] → K_𝖖[T]` at the coefficient-ring level; Plan 32).  It is
        a Z₊-ring homomorphism on the basis:

            dim(b₁·b₂) = dim(b₁)·dim(b₂),   dim(b⋆) = dim(b),   dim(1) = 1

        (checked by `verify_augmentation_is_hom`).  A torus has only
        1-dimensional irreps, so `dim ≡ 1` on `AbelianZPlusRing` / `Trivial`;
        the **1-dim reps** (`dim(b) == 1`, the lift torsor of Plan 32) are
        exactly the group-like/unit characters.
        """

    @abstractmethod
    def one_dim_rep_rank(self) -> int:
        """Rank `c` of the group of **1-dimensional representations**
        `Λ ≅ Z^c` — the *lift torsor* of Plan 32.  `Λ = R(G_f^ab)` is the rep
        ring of the abelianization; within the connected-reductive scope
        (Plan 32 A2) it is free abelian, `c` = number of central `U(1)`
        factors of `G_f`.  `0` for semisimple / trivial `G_f`."""

    @abstractmethod
    def embed_one_dim_rep(self, f: "tuple[int, ...]") -> BasisElement:
        """The inclusion `ι : Λ = AbelianZPlusRing(c) ↪ R` on a character:
        the **group-like** basis element `ι(μ^f) ∈ B` for `f ∈ Z^c`
        (`len(f) == one_dim_rep_rank()`).  Its image is a 1-dim rep
        (`dim == 1`, invertible `b·b⋆ == 1`), with `ι(0)=1`,
        `ι(f+g)=ι(f)·ι(g)`, `ι(−f)=ι(f)⋆` (checked by `verify_one_dim_reps`)."""

    # ------- element constructors -------

    def zero(self) -> "RElement":
        return RElement(self, {})

    def one(self) -> "RElement":
        return RElement(self, {self.one_basis(): 1})

    def basis_element(self, b: BasisElement) -> "RElement":
        return RElement(self, {b: 1})

    # ------- the forgetful map  ε : R → Z = TrivialZPlusRing  (Plan 32) -------

    def augmentation(self) -> "RingHom":
        """The forgetful Z₊-ring homomorphism `ε : R → Z` (= `TrivialZPlusRing`),
        `b ↦ dim(b)·()`.  Physically: forget the flavour group `G_f`.
        Generalises the abelian `augmentation_hom` (where every `dim` is 1)."""
        return augmentation_hom(self)

    def verify_augmentation_is_hom(self, samples: "list") -> bool:
        """Check that `dim` is a unital, ⋆-invariant, multiplicative,
        ℤ_{≥1}-valued ring homomorphism on the given basis `samples`."""
        if self.dim(self.one_basis()) != 1:
            return False
        for b in samples:
            if self.dim(b) < 1 or self.dim(self.star_basis(b)) != self.dim(b):
                return False
        for b1 in samples:
            for b2 in samples:
                prod = self.multiply_basis(b1, b2)
                if sum(n * self.dim(c) for c, n in prod.items()) != self.dim(b1) * self.dim(b2):
                    return False
        return True

    # ------- the 1-dim-rep subring  Λ = R(G_f^ab) ↪ R  (Plan 32 lift torsor) -------

    def one_dim_reps(self) -> "AbelianZPlusRing":
        """`Λ` as an abstract ring: `AbelianZPlusRing(c)`, `c = one_dim_rep_rank()`
        (free abelian within the connected-reductive scope; Plan 32 A2)."""
        return AbelianZPlusRing(self.one_dim_rep_rank())

    def one_dim_rep_inclusion(self) -> "RingHom":
        """The group-like inclusion `ι : AbelianZPlusRing(c) → R`,
        `μ^f ↦ embed_one_dim_rep(f)` — the handle the lift/section machinery
        (Plan 32 T4) uses to range over and apply twists by `Λ`."""
        Lam = self.one_dim_reps()
        return RingHom(
            Lam, self,
            lambda f: self.basis_element(self.embed_one_dim_rep(f)),
        )

    def verify_one_dim_reps(self, char_samples: "list") -> bool:
        """Check the 1-dim-rep structure on the given `Λ`-characters `f ∈ Z^c`:
        each `ι(f)` is group-like (`dim == 1`, invertible `ι(f)·ι(f)⋆ == 1`),
        and `ι` is a unital ⋆-compatible ring hom (`ι(0)=1`, `ι(f+g)=ι(f)·ι(g)`,
        `ι(−f)=ι(f)⋆`)."""
        c = self.one_dim_rep_rank()
        if any(len(f) != c for f in char_samples):
            return False
        one = self.one_basis()
        if self.embed_one_dim_rep((0,) * c) != one:
            return False
        for f in char_samples:
            b = self.embed_one_dim_rep(tuple(f))
            if self.dim(b) != 1:
                return False
            if self.multiply_basis(b, self.star_basis(b)) != {one: 1}:
                return False
            if self.embed_one_dim_rep(tuple(-x for x in f)) != self.star_basis(b):
                return False
        for f in char_samples:
            for g in char_samples:
                fg = tuple(a + b for a, b in zip(f, g))
                if self.multiply_basis(
                    self.embed_one_dim_rep(tuple(f)),
                    self.embed_one_dim_rep(tuple(g)),
                ) != {self.embed_one_dim_rep(fg): 1}:
                    return False
        return True


# ---------------------------------------------------------------------------
# RElement: a Z-linear combination of Z₊-ring basis elements.
# ---------------------------------------------------------------------------


class RElement:
    """An element of a Z₊-ring, stored as `dict[BasisElement, int]` with
    zero coefficients dropped.

    Coefficients are *signed* integers (the ring R itself is a Z-module, not
    a Z_{≥0}-module — only the basis-on-basis structure constants are
    non-negative).
    """

    __slots__ = ("ring", "terms")

    def __init__(
        self,
        ring: ZPlusRing,
        terms: dict[BasisElement, int] | None = None,
    ):
        self.ring = ring
        self.terms: dict[BasisElement, int] = {}
        if terms:
            for b, c in terms.items():
                if c == 0:
                    continue
                self.terms[b] = self.terms.get(b, 0) + c
            self.terms = {b: c for b, c in self.terms.items() if c != 0}

    def is_zero(self) -> bool:
        return not self.terms

    def is_one(self) -> bool:
        ob = self.ring.one_basis()
        return self.terms == {ob: 1}

    def __add__(self, other: "RElement") -> "RElement":
        if not isinstance(other, RElement):
            return NotImplemented
        if other.ring != self.ring:
            raise ValueError("RElement: ring mismatch in __add__")
        out = dict(self.terms)
        for b, c in other.terms.items():
            out[b] = out.get(b, 0) + c
        return RElement(self.ring, out)

    def __sub__(self, other: "RElement") -> "RElement":
        return self + (-other)

    def __neg__(self) -> "RElement":
        return RElement(self.ring, {b: -c for b, c in self.terms.items()})

    def __mul__(self, other) -> "RElement":
        if isinstance(other, int):
            if other == 0:
                return self.ring.zero()
            return RElement(self.ring, {b: c * other for b, c in self.terms.items()})
        if not isinstance(other, RElement):
            return NotImplemented
        if other.ring != self.ring:
            raise ValueError("RElement: ring mismatch in __mul__")
        # Fast path: multiplication by the ring identity is a no-op (skips the
        # Clebsch–Gordan fusion loop).  RElement is functionally immutable, so
        # returning the operand is safe.
        if other.is_one():
            return self
        if self.is_one():
            return other
        out: dict[BasisElement, int] = {}
        for b1, c1 in self.terms.items():
            for b2, c2 in other.terms.items():
                products = self.ring.multiply_basis(b1, b2)
                for b3, n in products.items():
                    out[b3] = out.get(b3, 0) + c1 * c2 * n
        return RElement(self.ring, out)

    __rmul__ = __mul__

    def star(self) -> "RElement":
        """The ⋆-involution (Lusztig-Ostrik rep-ring duality) extended
        linearly to elements.  Structurally this is the action of ρ
        restricted to the central flavour subalgebra `R ⊂ A_𝖖[T]`:
        `ρ(μ^f) = μ^{-f}`.  *Not* the FKAlgebra's bar involution."""
        out: dict[BasisElement, int] = {}
        for b, c in self.terms.items():
            sb = self.ring.star_basis(b)
            out[sb] = out.get(sb, 0) + c
        return RElement(self.ring, out)

    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            if other == 0:
                return self.is_zero()
            return self.terms == {self.ring.one_basis(): other}
        if not isinstance(other, RElement):
            return NotImplemented
        return self.ring == other.ring and self.terms == other.terms

    def __hash__(self):
        # Mutable-style; keep dict semantics (cf. LaurentPoly, Element).
        raise TypeError("RElement is unhashable")

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        ob = self.ring.one_basis()
        for b in sorted(self.terms.keys(), key=str):
            c = self.terms[b]
            if b == ob:
                parts.append(str(c))
            elif c == 1:
                parts.append(f"[{b}]")
            elif c == -1:
                parts.append(f"-[{b}]")
            else:
                parts.append(f"{c}*[{b}]")
        s = parts[0]
        for p in parts[1:]:
            if p.startswith("-"):
                s += " - " + p[1:]
            else:
                s += " + " + p
        return s


# ---------------------------------------------------------------------------
# TrivialZPlusRing  (R = Z; corresponds to G = trivial)
# ---------------------------------------------------------------------------


class TrivialZPlusRing(ZPlusRing):
    """The trivial Z₊-ring  R = Z .  Single basis element  () , ⋆ identity."""

    def one_basis(self) -> tuple:
        return ()

    def multiply_basis(self, b1: tuple, b2: tuple) -> dict[tuple, int]:
        if b1 != () or b2 != ():
            raise ValueError(f"TrivialZPlusRing basis is {{()}}; got {b1}, {b2}")
        return {(): 1}

    def star_basis(self, b: tuple) -> tuple:
        if b != ():
            raise ValueError(f"TrivialZPlusRing basis is {{()}}; got {b}")
        return ()

    def dim(self, b: tuple) -> int:
        return 1

    def one_dim_rep_rank(self) -> int:
        return 0   # G = trivial: only the trivial 1-dim rep

    def embed_one_dim_rep(self, f) -> tuple:
        if tuple(f) != ():
            raise ValueError(f"TrivialZPlusRing: Λ has rank 0; got {f!r}")
        return self.one_basis()

    def __repr__(self) -> str:
        return "TrivialZPlusRing()"

    def __eq__(self, other) -> bool:
        return isinstance(other, TrivialZPlusRing)

    def __hash__(self) -> int:
        return hash(("TrivialZPlusRing",))


# ---------------------------------------------------------------------------
# AbelianZPlusRing  (R = R(U(1)^rank) = Z[μ_1^±, ..., μ_rank^±])
# ---------------------------------------------------------------------------


class AbelianZPlusRing(ZPlusRing):
    """The Z₊-ring  R(U(1)^rank) = Z[μ_1^±, ..., μ_rank^±] .

    Basis: Z^rank (the character lattice; basis tuple  f  ↔ character  μ^f ).
    Multiplication: group multiplication on characters,  μ^f · μ^g = μ^{f+g}
    (one term, coefficient 1).  Identity: the zero tuple.

    Star: rep-ring duality `⋆(f) = -f` (= `μ^f ↦ μ^{-f}`).  This is the
    Lusztig-Ostrik `⋆` of the Z₊-ring, *not* the FKAlgebra's bar — the
    latter acts only on q, leaving μ alone.  The Z₊-ring `⋆` here matches
    the action of `ρ` restricted to the central flavour subalgebra in
    any FBPSKAlgebra realization (`ρ(μ^f) = μ^{-f}` since
    `ρ(X_γ) = X_{-γ}`).
    """

    def __init__(self, rank: int):
        if rank < 0:
            raise ValueError(f"rank must be >= 0, got {rank}")
        self.rank = rank

    def one_basis(self) -> tuple[int, ...]:
        return (0,) * self.rank

    def multiply_basis(
        self, b1: tuple[int, ...], b2: tuple[int, ...],
    ) -> dict[tuple[int, ...], int]:
        if len(b1) != self.rank or len(b2) != self.rank:
            raise ValueError(
                f"basis tuples must have length {self.rank}; got {b1}, {b2}"
            )
        return {tuple(a + b for a, b in zip(b1, b2)): 1}

    def star_basis(self, b: tuple[int, ...]) -> tuple[int, ...]:
        if len(b) != self.rank:
            raise ValueError(f"basis tuple must have length {self.rank}; got {b}")
        return tuple(-x for x in b)

    def dim(self, b: tuple[int, ...]) -> int:
        return 1   # every U(1)^n character μ^f is 1-dimensional

    def one_dim_rep_rank(self) -> int:
        return self.rank   # R(U(1)^n): every character is a 1-dim rep, Λ = R itself

    def embed_one_dim_rep(self, f) -> tuple[int, ...]:
        if len(f) != self.rank:
            raise ValueError(f"AbelianZPlusRing: Λ has rank {self.rank}; got {f!r}")
        return tuple(f)   # ι = identity (the ring is its own abelianization)

    def __repr__(self) -> str:
        return f"AbelianZPlusRing(rank={self.rank})"

    def __eq__(self, other) -> bool:
        return isinstance(other, AbelianZPlusRing) and self.rank == other.rank

    def __hash__(self) -> int:
        return hash(("AbelianZPlusRing", self.rank))


# ---------------------------------------------------------------------------
# SU2ZPlusRing  (R = R(SU(2)) = Z[χ_1] = ⊕_{n ≥ 0} Z · χ_n)
# ---------------------------------------------------------------------------


class SU2ZPlusRing(ZPlusRing):
    """The Z₊-ring  R(SU(2)) = ⊕_{n ≥ 0} Z · χ_n .

    Basis: ℕ.  The basis element `n` is the irreducible SU(2) character
    of highest weight `n` (= the spin-`n/2` rep, of dimension `n + 1`).
    `χ_0 = 1` is the trivial rep (the identity).

    Multiplication: Clebsch-Gordan,
        χ_n · χ_m = χ_{n+m} + χ_{n+m-2} + … + χ_{|n-m|}.
    All structure constants are 0 or 1.

    ⋆ = identity: every SU(2) rep is self-dual (V ≅ V* canonically via
    the invariant symplectic form on the fundamental).

    Embedding in the maximal-torus ring.  `R(SU(2)) ↪ R(U(1)) = Z[μ^±]`
    as the Weyl-invariant subring (μ ↔ μ⁻¹), via
        χ_n ↦ μ^n + μ^{n-2} + … + μ^{-n}
              = (μ^{n+1} - μ^{-n-1}) / (μ - μ^{-1}).
    Equivalently χ_1 = μ + μ⁻¹ and the rest is generated by χ_1.  This
    embedding intertwines the Z₊-ring structures: the Clebsch-Gordan
    structure constants here are exactly the Weyl-orbit recombination
    of the U(1) `μ^a · μ^b = μ^{a+b}` rule.

    Use: coefficient ring for an FKAlgebra whose U(1) flavour symmetry
    is enhanced to SU(2) (the universal cover; integer + half-integer
    spins both appear as basis elements).  Use `SO3ZPlusRing` instead
    when only integer-spin reps appear naturally — which is the more
    common case for BPS realisations like [A_1, D_3] where the
    "half-integer-spin" canonical basis elements emerge as σ-orbit
    chains over BPS F's rather than as elements of the coefficient
    ring proper.  See the SU(2) realisations (`su2_bps_kalgebra`,
    `a1d3_kalg`) for the [A_1, D_3] setup.
    """

    def one_basis(self) -> int:
        return 0

    def multiply_basis(self, b1: int, b2: int) -> dict[int, int]:
        if not (isinstance(b1, int) and isinstance(b2, int)):
            raise TypeError(
                f"SU2ZPlusRing basis is ℕ (int ≥ 0); got {b1!r}, {b2!r}"
            )
        if b1 < 0 or b2 < 0:
            raise ValueError(
                f"SU2ZPlusRing basis is ℕ (int ≥ 0); got {b1}, {b2}"
            )
        lo, hi = abs(b1 - b2), b1 + b2
        # Range step 2 from |b1-b2| up to b1+b2, all with coefficient 1.
        return {k: 1 for k in range(lo, hi + 1, 2)}

    def star_basis(self, b: int) -> int:
        if not isinstance(b, int):
            raise TypeError(f"SU2ZPlusRing basis is ℕ (int ≥ 0); got {b!r}")
        if b < 0:
            raise ValueError(f"SU2ZPlusRing basis is ℕ (int ≥ 0); got {b}")
        return b

    def dim(self, b: int) -> int:
        return b + 1   # b = 2j ⇒ dim of the spin-(b/2) irrep is b+1

    def one_dim_rep_rank(self) -> int:
        return 0   # SU(2) is semisimple: only the trivial 1-dim rep

    def embed_one_dim_rep(self, f) -> int:
        if tuple(f) != ():
            raise ValueError(f"SU2ZPlusRing: Λ has rank 0; got {f!r}")
        return self.one_basis()

    def to_abelian(
        self, elt: "RElement", target: "AbelianZPlusRing" | None = None,
    ) -> "RElement":
        """Embed an `RElement` over `SU2ZPlusRing` into `AbelianZPlusRing(rank=1)`
        as Weyl-symmetric Laurent polynomials in μ:
            χ_n ↦ μ^n + μ^{n-2} + … + μ^{-n}.
        Mostly diagnostic — useful for cross-checking SU(2) Clebsch-Gordan
        against U(1) multiplication followed by Weyl-symmetrisation.
        """
        if elt.ring is not self:
            raise ValueError("to_abelian: element's ring is not this SU2ZPlusRing")
        if target is None:
            target = AbelianZPlusRing(rank=1)
        if not (isinstance(target, AbelianZPlusRing) and target.rank == 1):
            raise ValueError("to_abelian: target must be AbelianZPlusRing(rank=1)")
        out: dict[tuple[int], int] = {}
        for n, c in elt.terms.items():
            for k in range(-n, n + 1, 2):
                key = (k,)
                out[key] = out.get(key, 0) + c
        return RElement(target, out)

    def from_abelian(self, u_relt: "RElement",
                     allow_virtual: bool = True) -> "RElement":
        """Inverse of `to_abelian`: convert a Weyl(`μ ↦ μ⁻¹`)-symmetric
        μ-Laurent over `AbelianZPlusRing(rank=1)` into an `RElement` over this
        SU(2) ring by highest-weight peeling `χ_n = μ^n + μ^{n-2} + … + μ^{-n}`
        (top down).  Accepts **virtual** characters (signed coefficients —
        Wilson-line Schur indices are signed); raises only if the input is not
        symmetric under `μ ↦ μ⁻¹`.  The SU(2) sibling of `SO3ZPlusRing` /
        `SU3ZPlusRing.from_abelian`."""
        if not isinstance(u_relt.ring, AbelianZPlusRing) or u_relt.ring.rank != 1:
            raise TypeError(
                "from_abelian expected RElement over AbelianZPlusRing(rank=1)")
        rem: dict[int, int] = {}
        for basis, c in u_relt.terms.items():
            rem[int(basis[0])] = rem.get(int(basis[0]), 0) + c
        rem = {k: v for k, v in rem.items() if v}
        for k in list(rem):
            if rem.get(-k, 0) != rem[k]:
                raise ValueError(
                    f"from_abelian: input not μ↦μ⁻¹ symmetric (coef at {k}="
                    f"{rem[k]} but at {-k}={rem.get(-k, 0)})")
        out: dict[int, int] = {}
        while rem:
            n = max(rem)
            if n < 0:
                break
            c = rem[n]
            out[n] = out.get(n, 0) + c
            for k in range(-n, n + 1, 2):
                rem[k] = rem.get(k, 0) - c
                if rem[k] == 0:
                    rem.pop(k, None)
        return RElement(self, {n: c for n, c in out.items() if c})

    def __repr__(self) -> str:
        return "SU2ZPlusRing()"

    def __eq__(self, other) -> bool:
        return isinstance(other, SU2ZPlusRing)

    def __hash__(self) -> int:
        return hash(("SU2ZPlusRing",))


# ---------------------------------------------------------------------------
# TensorZPlusRing  (R = R_1 ⊗ ... ⊗ R_k; e.g. SU(2)^n = [SU2ZPlusRing] * n)
# ---------------------------------------------------------------------------


class TensorZPlusRing(ZPlusRing):
    """Tensor-product Z₊-ring  R = R_1 ⊗ R_2 ⊗ ... ⊗ R_k  of factor
    Z₊-rings.  The canonical use is **SU(2)^n** =
    `TensorZPlusRing([SU2ZPlusRing()] * n)` -- the flavour ring of the
    closed n-punctured sphere `Sk(S^2_{0,n})` (one SU(2) per puncture).

    Basis: tuples `(b_1, ..., b_k)` of factor basis elements -- the
    character `χ_{b_1} ⊗ ... ⊗ χ_{b_k}`.  Identity: `(1_1, ..., 1_k)`.
    Multiplication fuses **factor-wise** (Clebsch-Gordan in each factor),
    the outer product of the per-factor results:
        `(b)·(b') = ⊗_i (b_i ·_{R_i} b'_i)`,
    with structure constants the products of the factors' (non-negative)
    structure constants.  `⋆` acts factor-wise.

    This is the 'missing infrastructure piece' for SU(2)^n-flavoured
    KAlgebras (`skein_sphere/KALGEBRA_NOTES.md`); the per-factor
    un-branch from the `U(1)^k` Cartan (`AbelianZPlusRing(k)`) up to the
    non-abelian product is a `FlavourEnhancementKAlgebra`-style wrapper
    (Weyl group `∏_i W(R_i)`; for SU(2)^n the per-puncture `μ_p ↔ μ_p⁻¹`).
    """

    def __init__(self, *factors):
        # Unified surface: accept both the list form
        # `TensorZPlusRing([R1, R2, ...])` and the binary positional form
        # `TensorZPlusRing(R_a, R_b)` (the former `tensor_zplus_ring`
        # class, now a re-export shim — Plan 32 streamline).
        if len(factors) == 1 and isinstance(factors[0], (list, tuple)):
            factors = tuple(factors[0])
        self.factors = tuple(factors)
        if not self.factors:
            raise ValueError("TensorZPlusRing needs at least one factor")

    @property
    def factor_a(self):
        return self.factors[0]   # binary-form back-compat (Plan 32 streamline)

    @property
    def factor_b(self):
        return self.factors[1]   # binary-form back-compat (Plan 32 streamline)

    def one_basis(self) -> tuple:
        return tuple(f.one_basis() for f in self.factors)

    def multiply_basis(self, b1: tuple, b2: tuple) -> dict[tuple, int]:
        k = len(self.factors)
        if len(b1) != k or len(b2) != k:
            raise ValueError(
                f"TensorZPlusRing basis tuples must have length {k}; "
                f"got {b1!r}, {b2!r}"
            )
        out: dict[tuple, int] = {(): 1}
        for i, f in enumerate(self.factors):
            partial = f.multiply_basis(b1[i], b2[i])
            nxt: dict[tuple, int] = {}
            for pre, pc in out.items():
                for b, n in partial.items():
                    key = pre + (b,)
                    nxt[key] = nxt.get(key, 0) + pc * n
            out = nxt
        return out

    def star_basis(self, b: tuple) -> tuple:
        k = len(self.factors)
        if len(b) != k:
            raise ValueError(
                f"TensorZPlusRing basis tuple must have length {k}; got {b!r}"
            )
        return tuple(f.star_basis(b[i]) for i, f in enumerate(self.factors))

    def dim(self, b: tuple) -> int:
        out = 1
        for f, bi in zip(self.factors, b):
            out *= f.dim(bi)
        return out

    def one_dim_rep_rank(self) -> int:
        return sum(f.one_dim_rep_rank() for f in self.factors)   # Λ = ∏_i Λ_i

    def embed_one_dim_rep(self, f) -> tuple:
        if len(f) != self.one_dim_rep_rank():
            raise ValueError(
                f"TensorZPlusRing: Λ has rank {self.one_dim_rep_rank()}; got {f!r}"
            )
        out, i = [], 0
        for fac in self.factors:
            ci = fac.one_dim_rep_rank()
            out.append(fac.embed_one_dim_rep(tuple(f[i:i + ci])))
            i += ci
        return tuple(out)

    def __repr__(self) -> str:
        return f"TensorZPlusRing([{', '.join(repr(f) for f in self.factors)}])"

    def __eq__(self, other) -> bool:
        return isinstance(other, TensorZPlusRing) and self.factors == other.factors

    def __hash__(self) -> int:
        return hash(("TensorZPlusRing", self.factors))


# ---------------------------------------------------------------------------
# SU2xU1ZPlusRing  (R = R(SU(2)) ⊗ Z[μ^±] -- SU(2) × U(1) flavor)
# ---------------------------------------------------------------------------


class SU2xU1ZPlusRing(ZPlusRing):
    """The Z₊-ring  R(SU(2)) ⊗ Z[μ, μ⁻¹]  of SU(2) × U(1) characters.

    Basis: pairs ``(k, m)`` with ``k ≥ 0`` (SU(2) χ-index, spin k/2) and
    ``m ∈ ℤ`` (U(1) charge).  ``(0, 0)`` is the trivial character.

    Multiplication: Clebsch-Gordan on χ, additive on μ:
        (k_1, m_1) · (k_2, m_2)
          = Σ_{r ∈ {|k_1-k_2|, |k_1-k_2|+2, ..., k_1+k_2}}  (r, m_1+m_2).

    Star ⋆: SU(2) self-dual; U(1) negation.  (k, m) ↦ (k, -m).

    Use: coefficient ring for an FKAlgebra whose flavour symmetry is
    SU(2) × U(1) (e.g. A_1D_{2k} for k ≥ 2 — chain + doublet, with the
    chain-sum gauged U(1) on top of the doublet SU(2)).
    """

    def one_basis(self) -> tuple[int, int]:
        return (0, 0)

    def multiply_basis(
        self, b1: tuple[int, int], b2: tuple[int, int],
    ) -> dict[tuple[int, int], int]:
        k1, m1 = b1
        k2, m2 = b2
        if k1 < 0 or k2 < 0:
            raise ValueError(f"SU2xU1: χ-index must be ≥ 0, got {b1}, {b2}")
        m = m1 + m2
        out: dict[tuple[int, int], int] = {}
        # Clebsch-Gordan: χ_{k_1} · χ_{k_2} = Σ χ_r for r in CG range.
        r_min, r_max = abs(k1 - k2), k1 + k2
        for r in range(r_min, r_max + 1, 2):
            out[(r, m)] = 1
        return out

    def star_basis(self, b: tuple[int, int]) -> tuple[int, int]:
        k, m = b
        return (k, -m)

    def dim(self, b: tuple[int, int]) -> int:
        return b[0] + 1   # SU(2) spin-(k/2) is (k+1)-dim; the U(1) factor μ^m is 1-dim

    def one_dim_rep_rank(self) -> int:
        return 1   # the U(1) factor (det); SU(2) is semisimple

    def embed_one_dim_rep(self, f) -> tuple[int, int]:
        if len(f) != 1:
            raise ValueError(f"SU2xU1ZPlusRing: Λ has rank 1; got {f!r}")
        return (0, f[0])   # trivial SU(2) ⊗ μ^{f0}

    def __repr__(self) -> str:
        return "SU2xU1ZPlusRing()"

    def __eq__(self, other) -> bool:
        return isinstance(other, SU2xU1ZPlusRing)

    def __hash__(self) -> int:
        return hash("SU2xU1ZPlusRing")


# ---------------------------------------------------------------------------
# SO3ZPlusRing  (R = R(SO(3)) = ⊕_{j ≥ 0} Z · χ_j ⊂ R(SU(2)))
# ---------------------------------------------------------------------------


class SO3ZPlusRing(ZPlusRing):
    """The Z₊-ring  R(SO(3)) = ⊕_{j ≥ 0} Z · χ_j  (integer spins only).

    Basis: ℕ_0.  The basis element `j` is the irreducible SO(3)
    character of spin `j`, dimension `2j + 1`.  Equivalently, the j-th
    SO(3) rep is the SU(2) spin-j rep restricted from the cover; only
    integer spins descend to SO(3), so `j ∈ ℕ_0`.  `χ_0 = 1` is the
    trivial rep; `χ_1` is the 3-dim vector rep.

    Multiplication: Clebsch-Gordan with step 1,
        χ_j · χ_k = χ_{j+k} + χ_{j+k-1} + … + χ_{|j-k|}.
    All structure constants are 0 or 1.

    ⋆ = identity: SO(3) reps are self-dual (they are real / orthogonal).

    Embedding in the maximal-torus ring.  `R(SO(3)) ↪ R(U(1)) = Z[μ^±]`
    as the Weyl-invariant subring (μ ↔ μ⁻¹) with **integer** weights:
        χ_j ↦ μ^j + μ^{j-1} + … + μ^{-j}.
    In particular `χ_1 = μ + 1 + μ⁻¹` (the vector rep weights {-1, 0, +1}).
    Equivalently χ_j = sin((2j+1)θ/2) / sin(θ/2) at μ = e^{iθ}.

    Embedding in `SU2ZPlusRing`.  `R(SO(3)) ↪ R(SU(2))` as the even-n
    subring via `χ_j^{SO(3)} ↦ χ_{2j}^{SU(2)}` (set μ_{SO(3)} = μ_{SU(2)}²).
    See `to_su2()`.

    Use: coefficient ring for an FKAlgebra whose U(1) flavour symmetry
    is enhanced to SO(3) — the natural enhancement when the
    σ-symmetrised lattice elements all carry integer flavour weights.
    Canonical example: the [A_1, D_3] Argyres-Douglas theory in the
    D_3 (= central node + 2 leaves) BPS-quiver presentation, where
    Γ = Z·g_1 ⊕ Z·g_2 ⊕ Z·g_3, `B = [[0,1,1], [-1,0,0], [-1,0,0]]`,
    σ = (g_2 ↔ g_3) the D_3 leaf-exchange Poisson involution; then
    ker(B) = Z·(g_3 − g_2) is the SO(3) Cartan direction, and the
    σ-invariant combination `X_{g_3-g_2} + X_0 + X_{g_2-g_3}` realises
    the SO(3) vector character χ_1 = μ + 1 + μ⁻¹.  D_3 ≅ A_3 as
    Dynkin diagrams, so [A_1, D_3] = [A_1, A_3] as a physical theory;
    the D_3 presentation manifests SO(3) (= leaf-exchange Weyl) at the
    cost of hiding the Z_6 (= half-monodromy ρ) which would be
    manifest in a cyclic A_3 presentation.
    """

    def one_basis(self) -> int:
        return 0

    def multiply_basis(self, b1: int, b2: int) -> dict[int, int]:
        if not (isinstance(b1, int) and isinstance(b2, int)):
            raise TypeError(
                f"SO3ZPlusRing basis is ℕ_0 (int ≥ 0); got {b1!r}, {b2!r}"
            )
        if b1 < 0 or b2 < 0:
            raise ValueError(
                f"SO3ZPlusRing basis is ℕ_0 (int ≥ 0); got {b1}, {b2}"
            )
        lo, hi = abs(b1 - b2), b1 + b2
        # Step 1: every spin in [|j-k|, j+k] appears once.
        return {k: 1 for k in range(lo, hi + 1)}

    def star_basis(self, b: int) -> int:
        if not isinstance(b, int):
            raise TypeError(f"SO3ZPlusRing basis is ℕ_0 (int ≥ 0); got {b!r}")
        if b < 0:
            raise ValueError(f"SO3ZPlusRing basis is ℕ_0 (int ≥ 0); got {b}")
        return b

    def dim(self, b: int) -> int:
        return 2 * b + 1   # integer spin j ⇒ dim 2j+1

    def one_dim_rep_rank(self) -> int:
        return 0   # SO(3) is semisimple: only the trivial 1-dim rep

    def embed_one_dim_rep(self, f) -> int:
        if tuple(f) != ():
            raise ValueError(f"SO3ZPlusRing: Λ has rank 0; got {f!r}")
        return self.one_basis()

    def to_abelian(
        self, elt: "RElement", target: "AbelianZPlusRing" | None = None,
    ) -> "RElement":
        """Embed an `RElement` over `SO3ZPlusRing` into `AbelianZPlusRing(rank=1)`
        via integer-weight Weyl-symmetrisation,
            χ_j ↦ μ^j + μ^{j-1} + … + μ^{-j}.
        Diagnostic — cross-checks Clebsch-Gordan against U(1) multiplication.
        """
        if elt.ring is not self:
            raise ValueError("to_abelian: element's ring is not this SO3ZPlusRing")
        if target is None:
            target = AbelianZPlusRing(rank=1)
        if not (isinstance(target, AbelianZPlusRing) and target.rank == 1):
            raise ValueError("to_abelian: target must be AbelianZPlusRing(rank=1)")
        out: dict[tuple[int], int] = {}
        for j, c in elt.terms.items():
            for k in range(-j, j + 1):
                key = (k,)
                out[key] = out.get(key, 0) + c
        return RElement(target, out)

    def to_su2(
        self, elt: "RElement", target: "SU2ZPlusRing" | None = None,
    ) -> "RElement":
        """Embed an `RElement` over `SO3ZPlusRing` into `SU2ZPlusRing` as
        the even-n subring: `χ_j^{SO(3)} ↦ χ_{2j}^{SU(2)}`.

        Intertwines the Z₊-ring structures: the SO(3) step-1 Clebsch-Gordan
        on j matches the SU(2) step-2 Clebsch-Gordan on n = 2j.
        """
        if elt.ring is not self:
            raise ValueError("to_su2: element's ring is not this SO3ZPlusRing")
        if target is None:
            target = SU2ZPlusRing()
        if not isinstance(target, SU2ZPlusRing):
            raise ValueError("to_su2: target must be SU2ZPlusRing")
        return RElement(target, {2 * j: c for j, c in elt.terms.items()})

    def __repr__(self) -> str:
        return "SO3ZPlusRing()"

    def __eq__(self, other) -> bool:
        return isinstance(other, SO3ZPlusRing)

    def __hash__(self) -> int:
        return hash(("SO3ZPlusRing",))


# ---------------------------------------------------------------------------
# SU3ZPlusRing  (R = R(SU(3)) = ⊕_{(p,q) ∈ ℕ²} Z · χ_{(p,q)})
# ---------------------------------------------------------------------------


class SU3ZPlusRing(ZPlusRing):
    """The Z₊-ring  R(SU(3)) = ⊕_{(p, q) ∈ ℕ²} Z · χ_{(p,q)} .

    Basis: ℕ².  The basis element `(p, q)` is the irreducible SU(3)
    character with Dynkin labels `(p, q)` — the rep of dimension
    `(p+1)(q+1)(p+q+2)/2`.  `(0, 0)` = trivial rep, `(1, 0)` =
    fundamental 3, `(0, 1)` = antifundamental 3̄, `(1, 1)` = adjoint 8,
    `(2, 0)` = symmetric 6.

    Multiplication: tensor product, computed via Klimyk's formula:
        χ_{λ} · χ_{μ}  =  Σ_{ν ∈ wt(V_μ)}  ε(ν) · χ_{λ ⊕ ν}
    where `λ ⊕ ν` is the affine-shift-and-Weyl-reflect of `λ + ν` (drop
    if on a wall, else reflect to dominant chamber with sign).
    Equivalent to Steinberg's formula (and to Littlewood-Richardson for
    sl_3) — produces non-negative integer multiplicities by general
    rep-theory.  Cached.

    Star: complex conjugation,  `⋆(p, q) = (q, p)`.  The fundamental
    `(1, 0) = 3` is dual to the antifundamental `(0, 1) = 3̄`; the
    adjoint `(1, 1) = 8` is self-dual; etc.

    Embedding in the maximal-torus ring.  `R(SU(3)) ↪ R(U(1)²) =
    Z[μ_1^±, μ_2^±]` as the Weyl-symmetric subring (S_3 acting by
    permuting the three weights of the fundamental).  In the
    "fundamental orbit basis" used by SU3BPSKAlgebra — where the
    three weights of the fundamental are `(1, 0)`, `(0, 1)`,
    `(-1, -1)` — the Weyl group S_3 is generated by:

        σ_3 : (a, b) → (-b, a - b)    (Z_3 cyclic, order 3)
        τ   : (a, b) → (b, a)         (Z_2 reflection)

    with the dominant chamber being `{0 ≤ a ≤ b}` and the map from
    SU(3) Dynkin labels to dominant lattice points being
    `(p, q) ↦ (q, p + q) = q·(1, 1) + p·(0, 1)`
    (i.e. ω_1 ↔ (0, 1), ω_2 ↔ (1, 1)).  Implemented by
    `to_abelian()`.

    Use: coefficient ring for an FKAlgebra whose flavour symmetry is
    enhanced to SU(3) — the natural enhancement for a BPS quiver
    containing an S_3-orbit of three rays at the three fundamental
    weights of SU(3).  See `SU3BPSKAlgebra` for the worked example
    (gauge node + 3-orbit of flavour nodes).
    """

    # --- Weyl group S_3 in the fundamental-orbit basis ---
    # σ_3 cycles (1,0) → (0,1) → (-1,-1) → (1,0) (the three flavour weights),
    # τ swaps (1,0) ↔ (0,1).  Both order-3 σ_3 and order-2 τ generate S_3.

    _S3_ELEMENTS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
        # (matrix as ((row0), (row1)), sign)
        # 1 (identity)
        ((1, 0), (0, 1)),
        # σ_3 = [[0,-1],[1,-1]]  (rotation)
        ((0, -1), (1, -1)),
        # σ_3² = [[-1,1],[-1,0]]
        ((-1, 1), (-1, 0)),
        # τ = [[0,1],[1,0]]  (reflection, sign -1)
        ((0, 1), (1, 0)),
        # τ σ_3 = [[1,-1],[0,-1]]
        ((1, -1), (0, -1)),
        # τ σ_3² = [[-1,0],[-1,1]]
        ((-1, 0), (-1, 1)),
    )

    def __init__(self):
        # Cache for tensor-product structure constants.
        self._mul_cache: dict[
            tuple[tuple[int, int], tuple[int, int]],
            dict[tuple[int, int], int],
        ] = {}
        # Cache for weight diagrams of irreducibles.
        self._weights_cache: dict[
            tuple[int, int], dict[tuple[int, int], int],
        ] = {}

    # -- abstract primitives ---------------------------------------------

    def one_basis(self) -> tuple[int, int]:
        return (0, 0)

    def multiply_basis(
        self, b1: tuple[int, int], b2: tuple[int, int],
    ) -> dict[tuple[int, int], int]:
        self._validate(b1)
        self._validate(b2)
        # Symmetrise on (p, q): χ_a · χ_b = χ_b · χ_a; canonicalise key.
        key = (b1, b2) if b1 <= b2 else (b2, b1)
        if key in self._mul_cache:
            return dict(self._mul_cache[key])
        # Klimyk: tensor with the smaller-dim rep on the right.
        lam = b1 if self._dim(b1) >= self._dim(b2) else b2
        mu = b2 if lam is b1 else b1
        out: dict[tuple[int, int], int] = {}
        for w, m in self._irrep_weights(mu).items():
            sign, refl = self._reflect_to_dominant_shifted(lam, w)
            if sign == 0:
                continue
            out[refl] = out.get(refl, 0) + sign * m
        out = {k: v for k, v in out.items() if v != 0}
        # Sanity: non-negative (true by rep theory).
        for v in out.values():
            if v < 0:
                raise RuntimeError(
                    f"Klimyk produced negative multiplicity for "
                    f"χ_{b1} · χ_{b2}: {out}"
                )
        self._mul_cache[key] = dict(out)
        return out

    def star_basis(self, b: tuple[int, int]) -> tuple[int, int]:
        self._validate(b)
        p, q = b
        return (q, p)

    def dim(self, b: tuple[int, int]) -> int:
        self._validate(b)
        p, q = b
        return (p + 1) * (q + 1) * (p + q + 2) // 2   # Weyl dimension (A_2)

    def one_dim_rep_rank(self) -> int:
        return 0   # SU(3) is semisimple: only the trivial 1-dim rep

    def embed_one_dim_rep(self, f) -> tuple[int, int]:
        if tuple(f) != ():
            raise ValueError(f"SU3ZPlusRing: Λ has rank 0; got {f!r}")
        return self.one_basis()

    # -- validation helpers ----------------------------------------------

    @staticmethod
    def _validate(b) -> None:
        if not (isinstance(b, tuple) and len(b) == 2
                and isinstance(b[0], int) and isinstance(b[1], int)):
            raise TypeError(
                f"SU3ZPlusRing basis is (p, q) ∈ ℕ²; got {b!r}"
            )
        if b[0] < 0 or b[1] < 0:
            raise ValueError(
                f"SU3ZPlusRing basis is (p, q) ∈ ℕ²; got {b}"
            )

    @staticmethod
    def _dim(b: tuple[int, int]) -> int:
        p, q = b
        return (p + 1) * (q + 1) * (p + q + 2) // 2

    # -- Weyl-reflection helper (in fundamental-orbit basis) ---

    @classmethod
    def _reflect_to_dominant_shifted(
        cls, lam: tuple[int, int], nu: tuple[int, int],
    ) -> tuple[int, tuple[int, int]]:
        """Klimyk's shift-and-reflect: take (λ + ν) in the *dominant* sense.

        Work in the (p_dyn, q_dyn) Dynkin coords directly.  Shift by ρ =
        (1, 1), apply Weyl simple reflections until in strict dominant
        chamber (both coords ≥ 1), then unshift.  Return (sign, dominant
        result) or (0, (0,0)) if (λ+ν) sits on a wall.

        Klimyk's algorithm needs the *Dynkin* coords (p, q), not the
        fundamental-orbit lattice coords.  The weights ν of a rep V_μ
        are stored in the fundamental-orbit lattice basis; convert via
        the standard SU(3) weight-to-Dynkin map:
            lattice (a, b)  ⟶  Dynkin (b - a, a)
        (this is the inverse of  (p, q) ↦ dominant lattice (q, p+q),
        applied locally to a single weight as a displacement).
        """
        # Convert ν (fundamental-orbit lattice) → Dynkin displacement.
        a, b = nu
        # Lattice basis: ω_1 = (0, 1), ω_2 = (1, 1) in fundamental-orbit
        # coords.  So (a, b) = a·(1, 0) + b·(0, 1) (the user's lattice
        # basis e_3, e_4) = (a, b) in user's basis.  We need to express
        # (a, b) in the ω-basis: ω_1 = (0, 1), ω_2 = (1, 1) (lattice).
        # So (a, b) = c_1·(0, 1) + c_2·(1, 1) ⟹ c_2 = a, c_1 = b - a.
        nu_dyn = (b - a, a)  # = (p_dyn, q_dyn) displacement

        p = lam[0] + nu_dyn[0] + 1  # shift by ρ = (1, 1)
        q = lam[1] + nu_dyn[1] + 1
        sign = 1
        # Reflect to strict dominant: both p, q ≥ 1.
        # Walls: p = 0 or q = 0.
        for _ in range(20):
            if p == 0 or q == 0:
                return 0, (0, 0)
            if p > 0 and q > 0:
                return sign, (p - 1, q - 1)
            # Reflect.  s_1: (p, q) → (-p, p + q).  s_2: (p, q) → (p + q, -q).
            if p < 0:
                p, q = -p, p + q
                sign = -sign
                continue
            if q < 0:
                p, q = p + q, -q
                sign = -sign
                continue
        raise RuntimeError(
            f"Klimyk reflection did not terminate for "
            f"λ={lam}, ν={nu} (Dynkin shift {nu_dyn})"
        )

    # -- weight diagram of V_{(p,q)} (in fundamental-orbit lattice) ------

    def _irrep_weights(
        self, mu: tuple[int, int],
    ) -> dict[tuple[int, int], int]:
        """Weight diagram of V_{(p, q)} as `{lattice_point: multiplicity}`,
        in the fundamental-orbit lattice basis (where the 3 fundamental
        weights are (1, 0), (0, 1), (-1, -1)).

        Algorithm: enumerate dominant lattice weights (those satisfying
        `0 ≤ γ_3 ≤ γ_4`) reachable from the highest weight by subtracting
        positive roots, compute multiplicity by Freudenthal recursion,
        then S_3-symmetrise.  Cached.
        """
        if mu in self._weights_cache:
            return dict(self._weights_cache[mu])
        p, q = mu
        # Highest weight in fundamental-orbit lattice coords:
        # (p, q) Dynkin = p·ω_1 + q·ω_2; ω_1 = (0, 1), ω_2 = (1, 1) (lattice).
        # So highest weight (lattice) = (q, p + q).
        hw = (q, p + q)
        # Positive roots in fundamental-orbit lattice coords.
        # Dynkin α_1 = (2, -1), α_2 = (-1, 2), α_1+α_2 = (1, 1).
        # Convert each to lattice via (p, q) → (q, p + q):
        #   α_1 = (2, -1) → (-1, 1)
        #   α_2 = (-1, 2) → (2, 1)
        #   α_1+α_2 = (1, 1) → (1, 2)
        pos_roots_lattice = [(-1, 1), (2, 1), (1, 2)]
        # Inner-product Gram matrix on lattice basis (in user's coords).
        # The standard SU(3) Killing form, in ω-basis, has Gram G_ω = (1/3)[[2,1],[1,2]].
        # Convert to lattice basis: change-of-basis from lattice (e_3, e_4)
        # to ω (ω_1, ω_2).  We have ω_1 = (0, 1), ω_2 = (1, 1) in lattice;
        # so lattice = M_ω, where M = [[0, 1], [1, 1]] (columns are ω_1, ω_2
        # in lattice).  Then for vectors u, v in lattice coords:
        #   ⟨u, v⟩ = u^T · M^{-T} · G_ω · M^{-1} · v
        # M^{-1} = [[-1, 1], [1, 0]] (cofactor formula).
        # G_lat = M^{-T} G_ω M^{-1} = [[-1, 1], [1, 0]]^T · (1/3)[[2,1],[1,2]] · [[-1, 1],[1, 0]]
        # = [[-1, 1], [1, 0]] · (1/3)[[2,1],[1,2]] · [[-1, 1],[1, 0]]
        # Freudenthal needs |λ+ρ|² - |ν+ρ|², which is simplest in Dynkin
        # coords: (p, q), ρ = (1, 1), inner product = G_ω.  Work in Dynkin
        # coords throughout for Freudenthal, then convert back to lattice
        # for the output dict.

        def lattice_to_dyn(v: tuple[int, int]) -> tuple[int, int]:
            a, b = v
            return (b - a, a)

        def dyn_to_lattice(d: tuple[int, int]) -> tuple[int, int]:
            x, y = d
            return (y, x + y)

        # Positive roots in Dynkin coords.
        pos_roots_dyn = [(2, -1), (-1, 2), (1, 1)]

        def dyn_inner(u: tuple[int, int], v: tuple[int, int]) -> int:
            # Use G_ω · 3 = [[2, 1], [1, 2]] to avoid fractions; we'll
            # multiply by 3 wherever needed.  Returns 3·⟨u, v⟩.
            return 2 * u[0] * v[0] + u[0] * v[1] + u[1] * v[0] + 2 * u[1] * v[1]

        rho_dyn = (1, 1)
        lam_plus_rho = (p + 1, q + 1)
        norm_lam = dyn_inner(lam_plus_rho, lam_plus_rho)

        # Enumerate dominant Dynkin weights ν with ν ≤ λ in dominance order.
        # Dominance: λ - ν = n_1·α_1 + n_2·α_2 with n_1, n_2 ≥ 0.
        # α_1 = (2, -1), α_2 = (-1, 2).  Solve:
        #   p - ν_p = 2n_1 - n_2;  q - ν_q = -n_1 + 2n_2.
        #   ⟹ n_1 = (2(p - ν_p) + (q - ν_q)) / 3
        #     n_2 = ((p - ν_p) + 2(q - ν_q)) / 3
        dominant_weights_dyn: list[tuple[int, int]] = []
        for vp in range(p + q + 1):
            for vq in range(p + q + 1):
                # Require vp + vq ≤ p + q (rough cone bound).
                num1 = 2 * (p - vp) + (q - vq)
                num2 = (p - vp) + 2 * (q - vq)
                if num1 < 0 or num2 < 0:
                    continue
                if num1 % 3 != 0 or num2 % 3 != 0:
                    continue
                dominant_weights_dyn.append((vp, vq))

        # Sort by total depth (n_1 + n_2 — distance from λ in root metric).
        def depth(nu_dyn: tuple[int, int]) -> int:
            return (
                (2 * (p - nu_dyn[0]) + (q - nu_dyn[1])) // 3 +
                ((p - nu_dyn[0]) + 2 * (q - nu_dyn[1])) // 3
            )
        dominant_weights_dyn.sort(key=depth)

        # Freudenthal recursion: mult(λ) = 1; for ν dominant, ν ≠ λ:
        #   (|λ+ρ|² - |ν+ρ|²) · mult(ν) =
        #     2 · Σ_{α ∈ Φ_+} Σ_{k≥1} mult(ν + kα) · ⟨ν + kα, α⟩
        # (Both sides multiplied by 3 via dyn_inner; the 3's cancel.)
        mults: dict[tuple[int, int], int] = {(p, q): 1}
        for nu in dominant_weights_dyn:
            if nu == (p, q):
                continue
            nu_plus_rho = (nu[0] + 1, nu[1] + 1)
            denom = norm_lam - dyn_inner(nu_plus_rho, nu_plus_rho)
            if denom == 0:
                continue  # nu+ρ on the Weyl-orbit boundary
            numer = 0
            for alpha in pos_roots_dyn:
                k = 1
                while True:
                    nuk_dyn = (nu[0] + k * alpha[0], nu[1] + k * alpha[1])
                    # Multiplicity is Weyl-invariant; reflect nuk to dominant
                    # and look it up.
                    mult_at_nuk = _su3_lookup_mult(mults, nuk_dyn)
                    if mult_at_nuk == 0:
                        break  # outside the W-orbit hull
                    numer += mult_at_nuk * dyn_inner(nuk_dyn, alpha)
                    k += 1
            value = 2 * numer
            # Both numer and denom are 3·(true values), so the ratio is exact.
            if value % denom != 0:
                raise RuntimeError(
                    f"Freudenthal: non-integer multiplicity at ν={nu} "
                    f"for V_{mu}: {value}/{denom}"
                )
            mults[nu] = value // denom

        # S_3-symmetrise: each dominant weight nu has 6/|Stab(nu)| Weyl images.
        out_lattice: dict[tuple[int, int], int] = {}
        for nu_dyn, m in mults.items():
            for g_idx, (row0, row1) in enumerate(self._S3_ELEMENTS):
                # Apply g to the dominant weight (in lattice coords)
                nu_lat = dyn_to_lattice(nu_dyn)
                gnu = (
                    row0[0] * nu_lat[0] + row0[1] * nu_lat[1],
                    row1[0] * nu_lat[0] + row1[1] * nu_lat[1],
                )
                # Avoid double-counting: "first wins" — the first image of nu
                # to land in out_lattice keeps the multiplicity (orbit-rep).
                if gnu not in out_lattice:
                    out_lattice[gnu] = m
                # otherwise it's a stabiliser image — same multiplicity, skip.

        self._weights_cache[mu] = dict(out_lattice)
        return out_lattice

    # -- to_abelian: embed into AbelianZPlusRing(rank=2) -----------------

    def to_abelian(
        self, elt: "RElement", target: "AbelianZPlusRing" | None = None,
    ) -> "RElement":
        """Embed an `RElement` over `SU3ZPlusRing` into `AbelianZPlusRing(rank=2)`
        as Weyl-symmetric Laurent polynomials in (μ_1, μ_2):

            χ_{(p,q)}  ↦  Σ_{w ∈ S_3 Weyl orbit}  m_w · μ_1^{w_1} · μ_2^{w_2}

        Weights are expressed in the fundamental-orbit basis (where the
        three fundamental weights are (1, 0), (0, 1), (-1, -1)) — the
        same basis used by SU3BPSKAlgebra.  Mostly diagnostic / used by
        SU3BPSKAlgebra.trace for the reverse direction.
        """
        if elt.ring is not self:
            raise ValueError("to_abelian: element's ring is not this SU3ZPlusRing")
        if target is None:
            target = AbelianZPlusRing(rank=2)
        if not (isinstance(target, AbelianZPlusRing) and target.rank == 2):
            raise ValueError("to_abelian: target must be AbelianZPlusRing(rank=2)")
        out: dict[tuple[int, ...], int] = {}
        for pq, c in elt.terms.items():
            for wt, m in self._irrep_weights(pq).items():
                key = (wt[0], wt[1])
                out[key] = out.get(key, 0) + c * m
        out = {k: v for k, v in out.items() if v != 0}
        return RElement(target, out)

    # -- inverse: μ-Laurent (Weyl-symmetric) → χ_(p,q) basis ------------

    def from_abelian(self, u1_relt: "RElement",
                     allow_virtual: bool = False) -> "RElement":
        """Convert a Weyl(S_3)-symmetric μ-Laurent over `AbelianZPlusRing(rank=2)`
        (in the fundamental-orbit basis) into an `RElement` over this
        SU3 ring, via highest-weight peeling.

        `allow_virtual=False` (default) requires a genuine (positive)
        character and raises on a negative highest-weight multiplicity.
        `allow_virtual=True` permits **virtual** characters (signed
        Z-combinations of irreducibles) — the irreducible characters are
        a Z-basis of the virtual representation ring, so a Weyl-symmetric
        input decomposes uniquely with possibly-negative multiplicities.
        Schur indices carry a `(-1)^F` grading and are virtual characters
        at each q-order, so the trace / inner-product path uses
        `allow_virtual=True`.

        Raises ValueError if the input is not S_3-symmetric.
        """
        if not isinstance(u1_relt.ring, AbelianZPlusRing) or u1_relt.ring.rank != 2:
            raise TypeError(
                "from_abelian expected RElement over AbelianZPlusRing(rank=2)"
            )
        # Collect into a mutable {lattice_point → coeff}.
        coeffs: dict[tuple[int, int], int] = {}
        for basis, c in u1_relt.terms.items():
            key = (int(basis[0]), int(basis[1]))
            coeffs[key] = coeffs.get(key, 0) + c
        coeffs = {k: v for k, v in coeffs.items() if v != 0}
        # Verify S_3 symmetry: for every (a, b) ∈ coeffs, the entire
        # S_3 orbit has the same coefficient.
        seen_orbits: set[tuple[int, int]] = set()
        for k in list(coeffs.keys()):
            if k in seen_orbits:
                continue
            orbit_imgs = set()
            for row0, row1 in self._S3_ELEMENTS:
                gk = (
                    row0[0] * k[0] + row0[1] * k[1],
                    row1[0] * k[0] + row1[1] * k[1],
                )
                orbit_imgs.add(gk)
            base = coeffs.get(k, 0)
            for gk in orbit_imgs:
                if coeffs.get(gk, 0) != base:
                    raise ValueError(
                        f"from_abelian: input not S_3-symmetric; "
                        f"coef at {k}={base} but at {gk}={coeffs.get(gk, 0)}"
                    )
            seen_orbits.update(orbit_imgs)

        chi_coeffs: dict[tuple[int, int], int] = {}
        # Highest-weight peeling: find dominant (0 ≤ a ≤ b) point with
        # max (b, then a) (= largest in dominance), read off (p, q), subtract.
        while coeffs:
            # Find lex-max dominant (a, b) with 0 ≤ a ≤ b.
            dominant = [(a, b) for (a, b), c in coeffs.items()
                        if 0 <= a <= b and c != 0]
            if not dominant:
                raise ValueError(
                    f"from_abelian: residue is non-zero off the dominant "
                    f"chamber: {coeffs}"
                )
            # The maximal dominant weight is the one with largest b (most
            # extreme in any positive direction); break ties by smaller a.
            dominant.sort(key=lambda ab: (-ab[1], ab[0]))
            top = dominant[0]
            a, b = top
            # SU(3) Dynkin: lattice (a, b) dominant ↔ (p, q) = (b - a, a).
            p, q = b - a, a
            c_top = coeffs[top]
            if c_top < 0 and not allow_virtual:
                raise ValueError(
                    f"from_abelian: negative coefficient at dominant "
                    f"weight {top}: {c_top} (not a positive S_3 character; "
                    f"pass allow_virtual=True for signed/index characters)"
                )
            chi_coeffs[(p, q)] = chi_coeffs.get((p, q), 0) + c_top
            # Subtract c_top · weight-diagram of V_{(p, q)}.
            for wt, m in self._irrep_weights((p, q)).items():
                new = coeffs.get(wt, 0) - c_top * m
                if new == 0:
                    coeffs.pop(wt, None)
                else:
                    coeffs[wt] = new
        chi_coeffs = {k: v for k, v in chi_coeffs.items() if v != 0}
        return RElement(self, chi_coeffs)

    def __repr__(self) -> str:
        return "SU3ZPlusRing()"

    def __eq__(self, other) -> bool:
        return isinstance(other, SU3ZPlusRing)

    def __hash__(self) -> int:
        return hash(("SU3ZPlusRing",))


def _su3_lookup_mult(
    mults: dict[tuple[int, int], int], nu_dyn: tuple[int, int],
) -> int:
    """Look up Freudenthal multiplicity at Dynkin coords `nu_dyn`,
    Weyl-reflecting into the dominant chamber if needed.  Returns 0 if
    outside the convex hull of W·λ (= not yet populated in `mults`)."""
    p, q = nu_dyn
    # Reflect into the dominant chamber.  Dominant: p ≥ 0 and q ≥ 0.
    # Reflections s_1: (p,q) → (-p, p+q); s_2: (p,q) → (p+q, -q).
    for _ in range(20):
        if p >= 0 and q >= 0:
            return mults.get((p, q), 0)
        if p < 0:
            p, q = -p, p + q
            continue
        if q < 0:
            p, q = p + q, -q
            continue
    return 0

# ---------------------------------------------------------------------------
# SU4ZPlusRing  (R = R(SU(4)) = ⊕_{(p,q,r) ∈ ℕ³} Z · χ_{(p,q,r)})
# ---------------------------------------------------------------------------


class SU4ZPlusRing(ZPlusRing):
    """The Z₊-ring  R(SU(4)) = ⊕_{(p, q, r) ∈ ℕ³} Z · χ_{(p,q,r)} .

    Basis: ℕ³.  The basis element `(p, q, r)` is the irreducible SU(4)
    character with Dynkin labels `(p, q, r)`, dimension

        dim V_{(p,q,r)} = (p+1)(q+1)(r+1)(p+q+2)(q+r+2)(p+q+r+3)/12.

    Examples: `(0, 0, 0)` = trivial 1; `(1, 0, 0)` = fund 4;
    `(0, 1, 0)` = ∧² fund = 6 (= Spin(6) vector, self-dual);
    `(0, 0, 1)` = antifund 4̄; `(1, 0, 1)` = adjoint 15;
    `(2, 0, 0)` = Sym² fund = 10; `(0, 0, 2)` = Sym² antifund = 10̄.

    Multiplication: Klimyk's formula — tensor with the smaller-dim
    rep on the right, decomposing into a sum of shift-and-reflect
    contributions weighted by the multiplicity of each weight of the
    smaller rep.  Identical algorithm to `SU3ZPlusRing` but in rank 3.
    Cached.

    Star: complex conjugation, `⋆(p, q, r) = (r, q, p)`.  The fund
    `(1, 0, 0) = 4` is dual to the antifund `(0, 0, 1) = 4̄`; the
    `(0, 1, 0) = 6` and adjoint `(1, 0, 1) = 15` are self-dual.

    Embedding in the maximal-torus ring.  `R(SU(4)) ↪ R(U(1)³) =
    Z[μ_1^±, μ_2^±, μ_3^±]` as the Weyl-symmetric subring (S_4
    permuting the 4 weights of the fund).  In the **Dynkin lattice
    basis** used here (the ω-basis), the 4 fund weights are

        w_1 = ( 1,  0,  0),    w_2 = (-1,  1,  0),
        w_3 = ( 0, -1,  1),    w_4 = ( 0,  0, -1),

    and lattice coords ≡ Dynkin coords.  (Unlike `SU3ZPlusRing` whose
    "fundamental-orbit lattice basis" differs from the ω-basis.)
    Implemented by `to_abelian()`.

    Use: coefficient ring for an FKAlgebra whose flavour symmetry is
    SU(4) — natural for SU(2)+N_f=3 with manifest SU(4) flavour
    symmetry (see `bps_su2_nf3` / `su2_nf3_kalgebra`).
    """

    # --- Weyl group S_4 in Dynkin coords -------------------------------
    # Simple reflections in Dynkin coords for A_3:
    #   s_1: (p, q, r) → (-p, p + q, r)
    #   s_2: (p, q, r) → (p + q, -q, q + r)
    #   s_3: (p, q, r) → (p, q + r, -r)
    # The 24 = |S_4| Weyl elements are generated by closure under s_1 s_2 s_3.

    def __init__(self):
        self._mul_cache: dict[
            tuple[tuple[int, int, int], tuple[int, int, int]],
            dict[tuple[int, int, int], int],
        ] = {}
        self._weights_cache: dict[
            tuple[int, int, int], dict[tuple[int, int, int], int],
        ] = {}

    # -- abstract primitives ---------------------------------------------

    def one_basis(self) -> tuple[int, int, int]:
        return (0, 0, 0)

    def multiply_basis(
        self, b1: tuple[int, int, int], b2: tuple[int, int, int],
    ) -> dict[tuple[int, int, int], int]:
        self._validate(b1)
        self._validate(b2)
        key = (b1, b2) if b1 <= b2 else (b2, b1)
        if key in self._mul_cache:
            return dict(self._mul_cache[key])
        # Klimyk: tensor with the smaller-dim rep on the right.
        lam = b1 if self._dim(b1) >= self._dim(b2) else b2
        mu = b2 if lam is b1 else b1
        out: dict[tuple[int, int, int], int] = {}
        for w, m in self._irrep_weights(mu).items():
            sign, refl = self._reflect_to_dominant_shifted(lam, w)
            if sign == 0:
                continue
            out[refl] = out.get(refl, 0) + sign * m
        out = {k: v for k, v in out.items() if v != 0}
        for v in out.values():
            if v < 0:
                raise RuntimeError(
                    f"Klimyk produced negative multiplicity for "
                    f"χ_{b1} · χ_{b2}: {out}"
                )
        self._mul_cache[key] = dict(out)
        return out

    def star_basis(
        self, b: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        self._validate(b)
        p, q, r = b
        return (r, q, p)

    # -- validation / dimension ------------------------------------------

    @staticmethod
    def _validate(b) -> None:
        if not (isinstance(b, tuple) and len(b) == 3
                and all(isinstance(x, int) for x in b)):
            raise TypeError(
                f"SU4ZPlusRing basis is (p, q, r) ∈ ℕ³; got {b!r}"
            )
        if b[0] < 0 or b[1] < 0 or b[2] < 0:
            raise ValueError(
                f"SU4ZPlusRing basis is (p, q, r) ∈ ℕ³; got {b}"
            )

    @staticmethod
    def _dim(b: tuple[int, int, int]) -> int:
        p, q, r = b
        return ((p + 1) * (q + 1) * (r + 1)
                * (p + q + 2) * (q + r + 2) * (p + q + r + 3)) // 12

    def dim(self, b: tuple[int, int, int]) -> int:
        self._validate(b)
        return self._dim(b)

    def one_dim_rep_rank(self) -> int:
        return 0   # SU(4) is semisimple: only the trivial 1-dim rep

    def embed_one_dim_rep(self, f) -> tuple[int, int, int]:
        if tuple(f) != ():
            raise ValueError(f"SU4ZPlusRing: Λ has rank 0; got {f!r}")
        return self.one_basis()

    # -- Klimyk shift-and-reflect (Dynkin coords) ------------------------

    @classmethod
    def _reflect_to_dominant_shifted(
        cls,
        lam: tuple[int, int, int],
        nu: tuple[int, int, int],
    ) -> tuple[int, tuple[int, int, int]]:
        """Klimyk: take (λ + ν + ρ), reflect via simple reflections
        until in the strict dominant chamber (all coords ≥ 1), then
        unshift.  Returns (sign, dominant_result) or (0, _) on a wall.

        Lattice coords == Dynkin coords here (ω-basis), so ν is used
        directly.  Reflections:
          s_1: (p, q, r) → (-p, p + q, r)
          s_2: (p, q, r) → (p + q, -q, q + r)
          s_3: (p, q, r) → (p, q + r, -r)
        """
        p = lam[0] + nu[0] + 1
        q = lam[1] + nu[1] + 1
        r = lam[2] + nu[2] + 1
        sign = 1
        # Reflect to strict dominant: p, q, r ≥ 1.
        for _ in range(40):
            if p == 0 or q == 0 or r == 0:
                return 0, (0, 0, 0)
            if p > 0 and q > 0 and r > 0:
                return sign, (p - 1, q - 1, r - 1)
            if p < 0:
                p, q, r = -p, p + q, r
                sign = -sign
                continue
            if q < 0:
                p, q, r = p + q, -q, q + r
                sign = -sign
                continue
            if r < 0:
                p, q, r = p, q + r, -r
                sign = -sign
                continue
        raise RuntimeError(
            f"SU4 Klimyk reflection did not terminate for λ={lam}, ν={nu}"
        )

    # -- weight diagram of V_{(p, q, r)} (in Dynkin/lattice coords) ------

    def _irrep_weights(
        self, mu: tuple[int, int, int],
    ) -> dict[tuple[int, int, int], int]:
        """Weight diagram of V_{(p, q, r)} as `{lattice_point: mult}`,
        in Dynkin/lattice coords.

        Algorithm: enumerate dominant Dynkin weights ν with ν ≤ λ in
        dominance order, compute mult by Freudenthal recursion, then
        S_4-symmetrise.  Cached.
        """
        if mu in self._weights_cache:
            return dict(self._weights_cache[mu])
        p, q, r = mu

        # Positive roots of A_3 in Dynkin coords (6 = 4·3/2).
        pos_roots = [
            (2, -1, 0),                 # α_1
            (-1, 2, -1),                # α_2
            (0, -1, 2),                 # α_3
            (1, 1, -1),                 # α_1 + α_2
            (-1, 1, 1),                 # α_2 + α_3
            (1, 0, 1),                  # α_1 + α_2 + α_3
        ]

        # Killing-form Gram in ω-basis (Dynkin): G_ω = (1/4)·M where
        # M = [[3, 2, 1], [2, 4, 2], [1, 2, 3]].  We use 4·⟨u, v⟩ to
        # avoid fractions — the 4-factor cancels in (Freudenthal LHS):
        # `(|λ+ρ|² - |ν+ρ|²) · mult(ν) = 2 · numer`.
        def dyn_inner4(u, v):
            # 4·⟨u, v⟩ in ω-basis (Dynkin coords).
            return (
                3 * u[0] * v[0] + 2 * u[0] * v[1] + u[0] * v[2]
                + 2 * u[1] * v[0] + 4 * u[1] * v[1] + 2 * u[1] * v[2]
                + u[2] * v[0] + 2 * u[2] * v[1] + 3 * u[2] * v[2]
            )

        lam_plus_rho = (p + 1, q + 1, r + 1)
        norm_lam = dyn_inner4(lam_plus_rho, lam_plus_rho)

        # Enumerate dominant Dynkin weights ν ≤ λ.  Dominance:
        # λ - ν = n_1 α_1 + n_2 α_2 + n_3 α_3 with n_i ≥ 0.
        # Solve for (n_1, n_2, n_3) from (p - ν_p, q - ν_q, r - ν_r):
        #   p - ν_p = 2 n_1 - n_2
        #   q - ν_q = -n_1 + 2 n_2 - n_3
        #   r - ν_r = -n_2 + 2 n_3
        # Inverse of A_3 Cartan matrix × 4 = [[3,2,1],[2,4,2],[1,2,3]].
        dominant_weights: list[tuple[int, int, int]] = []
        bound = p + q + r + 1
        for vp in range(bound + 1):
            for vq in range(bound + 1):
                for vr in range(bound + 1):
                    dp, dq, dr = p - vp, q - vq, r - vr
                    if dp < 0 or dq < 0 or dr < 0:
                        # λ - ν lies in the positive cone of the α's, so its
                        # Dynkin coords may be negative; the actual constraint
                        # is n_1, n_2, n_3 ≥ 0, filtered via the n_i's below.
                        pass
                    # n_1 = (3 dp + 2 dq + dr) / 4
                    # n_2 = (2 dp + 4 dq + 2 dr) / 4
                    # n_3 = (dp + 2 dq + 3 dr) / 4
                    num1 = 3 * dp + 2 * dq + dr
                    num2 = 2 * dp + 4 * dq + 2 * dr
                    num3 = dp + 2 * dq + 3 * dr
                    if num1 < 0 or num2 < 0 or num3 < 0:
                        continue
                    if num1 % 4 != 0 or num2 % 4 != 0 or num3 % 4 != 0:
                        continue
                    dominant_weights.append((vp, vq, vr))

        # Sort by depth (n_1 + n_2 + n_3): process top-down from λ.
        def depth(nu):
            dp, dq, dr = p - nu[0], q - nu[1], r - nu[2]
            return ((3 * dp + 2 * dq + dr) // 4
                    + (2 * dp + 4 * dq + 2 * dr) // 4
                    + (dp + 2 * dq + 3 * dr) // 4)
        dominant_weights.sort(key=depth)

        # Freudenthal recursion in Dynkin coords.
        mults: dict[tuple[int, int, int], int] = {(p, q, r): 1}
        for nu in dominant_weights:
            if nu == (p, q, r):
                continue
            nu_plus_rho = (nu[0] + 1, nu[1] + 1, nu[2] + 1)
            denom = norm_lam - dyn_inner4(nu_plus_rho, nu_plus_rho)
            if denom == 0:
                continue
            numer = 0
            for alpha in pos_roots:
                k = 1
                while True:
                    nuk = (nu[0] + k * alpha[0],
                           nu[1] + k * alpha[1],
                           nu[2] + k * alpha[2])
                    mult_at_nuk = _su4_lookup_mult(mults, nuk)
                    if mult_at_nuk == 0:
                        break
                    numer += mult_at_nuk * dyn_inner4(nuk, alpha)
                    k += 1
            value = 2 * numer
            if value % denom != 0:
                raise RuntimeError(
                    f"SU4 Freudenthal: non-integer multiplicity at ν={nu} "
                    f"for V_{mu}: {value}/{denom}"
                )
            mults[nu] = value // denom

        # S_4-symmetrise: orbit of each dominant weight via 24 Weyl matrices.
        out_lattice: dict[tuple[int, int, int], int] = {}
        for nu, m in mults.items():
            seen_imgs = set()
            for mat in self._s4_matrices():
                gnu = (
                    mat[0][0] * nu[0] + mat[0][1] * nu[1] + mat[0][2] * nu[2],
                    mat[1][0] * nu[0] + mat[1][1] * nu[1] + mat[1][2] * nu[2],
                    mat[2][0] * nu[0] + mat[2][1] * nu[1] + mat[2][2] * nu[2],
                )
                if gnu in seen_imgs:
                    continue
                seen_imgs.add(gnu)
                out_lattice[gnu] = m  # all orbit members share the multiplicity

        self._weights_cache[mu] = dict(out_lattice)
        return out_lattice

    # -- Weyl group S_4 generation (24 elements) -------------------------

    _S4_CACHE: list[tuple[tuple[int, int, int], ...]] = []

    @classmethod
    def _s4_matrices(cls):
        if cls._S4_CACHE:
            return cls._S4_CACHE
        # Simple reflections in Dynkin coords (as 3×3 matrices).
        s1 = ((-1, 0, 0), (1, 1, 0), (0, 0, 1))
        s2 = ((1, 1, 0), (0, -1, 0), (0, 1, 1))
        s3 = ((1, 0, 0), (0, 1, 1), (0, 0, -1))
        ident = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

        def mat_mul(A, B):
            return tuple(
                tuple(sum(A[i][k] * B[k][j] for k in range(3))
                      for j in range(3))
                for i in range(3)
            )

        gens = [s1, s2, s3]
        elements = [ident]
        frontier = [ident]
        while frontier:
            new_frontier = []
            for g in frontier:
                for h in gens:
                    gh = mat_mul(g, h)
                    if gh not in elements:
                        elements.append(gh)
                        new_frontier.append(gh)
            frontier = new_frontier
        if len(elements) != 24:
            raise RuntimeError(
                f"SU4 Weyl S_4 generation produced {len(elements)} ≠ 24 elements"
            )
        cls._S4_CACHE = elements
        return elements

    # -- to_abelian / from_abelian ---------------------------------------

    def to_abelian(
        self, elt: "RElement", target: "AbelianZPlusRing | None" = None,
    ) -> "RElement":
        """Embed an `RElement` over `SU4ZPlusRing` into `AbelianZPlusRing(rank=3)`
        as Weyl-symmetric Laurent polynomials in (μ_1, μ_2, μ_3):

            χ_{(p,q,r)}  ↦  Σ_{w ∈ S_4 orbit}  m_w · μ^w

        Weights are in Dynkin/lattice coords (the ω-basis).  Mostly
        diagnostic / used for cross-checking against the BPS oracle
        which outputs in the same rank-3 SU(4) Cartan.
        """
        if elt.ring is not self:
            raise ValueError("to_abelian: element's ring is not this SU4ZPlusRing")
        if target is None:
            target = AbelianZPlusRing(rank=3)
        if not (isinstance(target, AbelianZPlusRing) and target.rank == 3):
            raise ValueError("to_abelian: target must be AbelianZPlusRing(rank=3)")
        out: dict[tuple[int, ...], int] = {}
        for pqr, c in elt.terms.items():
            for wt, m in self._irrep_weights(pqr).items():
                key = (wt[0], wt[1], wt[2])
                out[key] = out.get(key, 0) + c * m
        out = {k: v for k, v in out.items() if v != 0}
        return RElement(target, out)

    def from_abelian(self, u_relt: "RElement") -> "RElement":
        """Convert a Weyl(S_4)-symmetric μ-Laurent over `AbelianZPlusRing(rank=3)`
        (in the SU(4) Dynkin/lattice basis) into an `RElement` over this
        SU(4) ring via highest-weight peeling.

        Accepts **virtual** characters: coefficients (in the SU(4) χ_(p,q,r)
        basis) may be negative — Schur indices of Wilson lines are signed.
        The peeling itself proceeds regardless of sign.

        Raises ValueError only if the input is not S_4-symmetric.
        """
        if not isinstance(u_relt.ring, AbelianZPlusRing) or u_relt.ring.rank != 3:
            raise TypeError(
                "from_abelian expected RElement over AbelianZPlusRing(rank=3)"
            )
        coeffs: dict[tuple[int, int, int], int] = {}
        for basis, c in u_relt.terms.items():
            key = (int(basis[0]), int(basis[1]), int(basis[2]))
            coeffs[key] = coeffs.get(key, 0) + c
        coeffs = {k: v for k, v in coeffs.items() if v != 0}

        # Verify S_4 symmetry.
        seen_orbits: set[tuple[int, int, int]] = set()
        for k in list(coeffs.keys()):
            if k in seen_orbits:
                continue
            orbit_imgs = set()
            for mat in self._s4_matrices():
                gk = (
                    mat[0][0] * k[0] + mat[0][1] * k[1] + mat[0][2] * k[2],
                    mat[1][0] * k[0] + mat[1][1] * k[1] + mat[1][2] * k[2],
                    mat[2][0] * k[0] + mat[2][1] * k[1] + mat[2][2] * k[2],
                )
                orbit_imgs.add(gk)
            base = coeffs.get(k, 0)
            for gk in orbit_imgs:
                if coeffs.get(gk, 0) != base:
                    raise ValueError(
                        f"from_abelian: input not S_4-symmetric; "
                        f"coef at {k}={base} but at {gk}={coeffs.get(gk, 0)}"
                    )
            seen_orbits.update(orbit_imgs)

        chi_coeffs: dict[tuple[int, int, int], int] = {}
        while coeffs:
            # Dominant chamber in Dynkin coords: p, q, r ≥ 0.
            dominant = [(p, q, r) for (p, q, r), c in coeffs.items()
                        if p >= 0 and q >= 0 and r >= 0 and c != 0]
            if not dominant:
                raise ValueError(
                    f"from_abelian: residue off the dominant chamber: {coeffs}"
                )
            # Pick deepest dominant (furthest from origin in pos-root direction).
            dominant.sort(key=lambda pqr: -(pqr[0] + pqr[1] + pqr[2]))
            top = dominant[0]
            c_top = coeffs[top]
            # Virtual characters allowed: c_top may be negative.
            chi_coeffs[top] = chi_coeffs.get(top, 0) + c_top
            for wt, m in self._irrep_weights(top).items():
                new = coeffs.get(wt, 0) - c_top * m
                if new == 0:
                    coeffs.pop(wt, None)
                else:
                    coeffs[wt] = new
        chi_coeffs = {k: v for k, v in chi_coeffs.items() if v != 0}
        return RElement(self, chi_coeffs)

    def __repr__(self) -> str:
        return "SU4ZPlusRing()"

    def __eq__(self, other) -> bool:
        return isinstance(other, SU4ZPlusRing)

    def __hash__(self) -> int:
        return hash(("SU4ZPlusRing",))


def _su4_lookup_mult(
    mults: dict[tuple[int, int, int], int],
    nu: tuple[int, int, int],
) -> int:
    """Look up Freudenthal multiplicity at Dynkin coords `nu`, reflecting
    into the dominant chamber if needed.  Returns 0 if outside the
    Weyl-orbit hull (= not yet populated)."""
    p, q, r = nu
    for _ in range(40):
        if p >= 0 and q >= 0 and r >= 0:
            return mults.get((p, q, r), 0)
        if p < 0:
            p, q, r = -p, p + q, r
            continue
        if q < 0:
            p, q, r = p + q, -q, q + r
            continue
        if r < 0:
            p, q, r = p, q + r, -r
            continue
    return 0


# ---------------------------------------------------------------------------
# RLaurent: R[q^±] for a Z₊-ring R.
# ---------------------------------------------------------------------------


class RLaurent:
    """Elements of  R[q^±] = R[q, q^{-1}]  for a Z₊-ring  R .

    Stored as `dict[int, RElement]` (q-exponent → R-coefficient), with
    zero R-coefficients dropped.

    Bar involution: `q^n · r ↦ q^{-n} · r`.  Acts only on q; the R-side is
    untouched.  The FKAlgebra's bar fixes μ-fugacities (their conjugation
    is part of ρ, not bar — see `RElement.star()`).
    """

    __slots__ = ("ring", "coeffs")

    def __init__(
        self,
        ring: ZPlusRing,
        coeffs: dict[int, "RElement | int"] | None = None,
    ):
        self.ring = ring
        self.coeffs: dict[int, RElement] = {}
        if not coeffs:
            return
        for q_exp, c in coeffs.items():
            if isinstance(c, int):
                if c == 0:
                    continue
                rc = RElement(ring, {ring.one_basis(): c})
            elif isinstance(c, RElement):
                if c.ring != ring:
                    raise ValueError("RLaurent: R-coefficient ring mismatch")
                rc = c
            else:
                raise TypeError(
                    f"RLaurent: unsupported coefficient type {type(c).__name__}"
                )
            if not rc.is_zero():
                self.coeffs[q_exp] = rc

    # ------- constructors -------

    @classmethod
    def zero(cls, ring: ZPlusRing) -> "RLaurent":
        return cls(ring)

    @classmethod
    def one(cls, ring: ZPlusRing) -> "RLaurent":
        return cls(ring, {0: 1})

    @classmethod
    def q(
        cls, ring: ZPlusRing, n: int = 1, c: "RElement | int" = 1,
    ) -> "RLaurent":
        """The q-monomial  c · q^n ."""
        return cls(ring, {n: c})

    @classmethod
    def from_basis(
        cls, ring: ZPlusRing, b: BasisElement, n: int = 0,
    ) -> "RLaurent":
        """The R[q^±] element  q^n · [b]  (single basis element, no q-mixing)."""
        return cls(ring, {n: ring.basis_element(b)})

    # ------- predicates -------

    def is_zero(self) -> bool:
        return not self.coeffs

    # ------- arithmetic -------

    def __add__(self, other: "RLaurent") -> "RLaurent":
        if not isinstance(other, RLaurent):
            return NotImplemented
        if other.ring != self.ring:
            raise ValueError("RLaurent: ring mismatch in __add__")
        out = dict(self.coeffs)
        for n, c in other.coeffs.items():
            out[n] = out[n] + c if n in out else c
        return RLaurent(
            self.ring, {n: c for n, c in out.items() if not c.is_zero()},
        )

    def __sub__(self, other: "RLaurent") -> "RLaurent":
        return self + (-other)

    def __neg__(self) -> "RLaurent":
        return RLaurent(self.ring, {n: -c for n, c in self.coeffs.items()})

    def __mul__(self, other) -> "RLaurent":
        if isinstance(other, int):
            if other == 0:
                return RLaurent.zero(self.ring)
            return RLaurent(
                self.ring, {n: c * other for n, c in self.coeffs.items()},
            )
        if isinstance(other, RElement):
            if other.ring != self.ring:
                raise ValueError("RLaurent: ring mismatch in __mul__ with RElement")
            return RLaurent(
                self.ring, {n: c * other for n, c in self.coeffs.items()},
            )
        # LaurentPoly (Z[q^±]) lifts coefficient-wise into R[q^±]
        # (#231 widening: Element coefficients may mix the two types).
        if type(other).__name__ == "LaurentPoly":
            out: dict[int, RElement] = {}
            one_b = self.ring.one_basis()
            for n2, c2 in other._coeffs.items():
                if c2 == 0:
                    continue
                lifted = RElement(self.ring, {one_b: c2})
                for n1, c1 in self.coeffs.items():
                    n = n1 + n2
                    p = c1 * lifted
                    if p.is_zero():
                        continue
                    out[n] = out[n] + p if n in out else p
            return RLaurent(
                self.ring, {n: c for n, c in out.items() if not c.is_zero()},
            )
        if not isinstance(other, RLaurent):
            return NotImplemented
        if other.ring != self.ring:
            raise ValueError("RLaurent: ring mismatch in __mul__")
        # Fast path: a single-term (q-monomial) multiplicand needs no
        # convolution — the shift n1 ↦ n1+n2 is injective, so no key collides
        # and no accumulation is required.  When the lone coefficient is the
        # ring identity (the ubiquitous `q^e·1` phase factors in the cone
        # cyclicity engine), it is a *pure exponent shift* — skip the per-term
        # R-multiply entirely.  Provably identical to the general loop below.
        if len(other.coeffs) == 1:
            (n2, c2), = other.coeffs.items()
            if c2.is_one():
                return RLaurent(
                    self.ring, {n1 + n2: c1 for n1, c1 in self.coeffs.items()},
                )
            shifted = {}
            for n1, c1 in self.coeffs.items():
                p = c1 * c2
                if not p.is_zero():
                    shifted[n1 + n2] = p
            return RLaurent(self.ring, shifted)
        out: dict[int, RElement] = {}
        for n1, c1 in self.coeffs.items():
            for n2, c2 in other.coeffs.items():
                n = n1 + n2
                p = c1 * c2
                if p.is_zero():
                    continue
                out[n] = out[n] + p if n in out else p
        return RLaurent(
            self.ring, {n: c for n, c in out.items() if not c.is_zero()},
        )

    __rmul__ = __mul__

    def bar(self) -> "RLaurent":
        """`q ↔ q^{-1}`, R-coefficients untouched.  This is the FKAlgebra's
        bar involution: structure constants are palindromic in q, with
        μ-dependence transparent.  μ-conjugation is the action of ρ, not
        bar (see `RElement.star()`)."""
        return RLaurent(
            self.ring, {-n: c for n, c in self.coeffs.items()},
        )

    def star(self) -> "RLaurent":
        """Apply ⋆ to each R-coefficient; q untouched.  This is the R-side
        of `ρ`'s twisted-linear action — `ρ(c · u) = c.star() · ρ(u)`."""
        return RLaurent(
            self.ring, {n: c.star() for n, c in self.coeffs.items()},
        )

    # ------- coefficient access -------

    def coefficient(self, q_exp: int) -> RElement:
        """The R-coefficient at q^{q_exp} (zero element if absent)."""
        return self.coeffs.get(q_exp, self.ring.zero())

    # ------- equality -------

    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            if other == 0:
                return self.is_zero()
            return self.coeffs == {0: RElement(self.ring, {self.ring.one_basis(): other})}
        if not isinstance(other, RLaurent):
            return NotImplemented
        return self.ring == other.ring and self.coeffs == other.coeffs

    def __hash__(self):
        raise TypeError("RLaurent is unhashable")

    def __repr__(self) -> str:
        if not self.coeffs:
            return "0"
        parts: list[str] = []
        for n in sorted(self.coeffs.keys()):
            c = self.coeffs[n]
            c_repr = repr(c)
            if n == 0:
                parts.append(c_repr)
                continue
            qstr = "q" if n == 1 else f"q^{n}"
            if c_repr == "1":
                parts.append(qstr)
            elif c_repr == "-1":
                parts.append(f"-{qstr}")
            elif " " in c_repr:
                # multi-term R-coeff: parenthesize
                parts.append(f"({c_repr})*{qstr}")
            else:
                parts.append(f"{c_repr}*{qstr}")
        s = parts[0]
        for p in parts[1:]:
            if p.startswith("-"):
                s += " - " + p[1:]
            else:
                s += " + " + p
        return s


# ---------------------------------------------------------------------------
# RPowerSeries: truncated R((q)) for the trace codomain.
# ---------------------------------------------------------------------------


class RPowerSeries:
    """A truncation of the formal Laurent series ring `R((q))` for a Z₊-ring R.

    Stored as `dict[int, RElement]` (q-exponent → R-coefficient) plus a
    truncation order `K`: only coefficients at q-exponents `≤ K` are kept.
    Negative q-exponents are allowed (Laurent), but no global lower bound
    is enforced — series with deep negative tails are computed only insofar
    as they appear in arithmetic.

    This is the codomain of `KAlgebra.trace` in the parameterized framework:
    the `ρ²`-twisted trace of an FKAlgebra valued in `R((q))`.
    """

    __slots__ = ("ring", "K", "coeffs")

    def __init__(
        self,
        ring: ZPlusRing,
        coeffs: dict[int, "RElement | int"] | None,
        K: int,
    ):
        self.ring = ring
        self.K = K
        self.coeffs: dict[int, RElement] = {}
        if not coeffs:
            return
        for q_exp, c in coeffs.items():
            if q_exp > K:
                continue
            if isinstance(c, int):
                if c == 0:
                    continue
                rc = RElement(ring, {ring.one_basis(): c})
            elif isinstance(c, RElement):
                if c.ring != ring:
                    raise ValueError("RPowerSeries: R-coefficient ring mismatch")
                rc = c
            else:
                raise TypeError(
                    f"RPowerSeries: unsupported coefficient type "
                    f"{type(c).__name__}"
                )
            if not rc.is_zero():
                self.coeffs[q_exp] = rc

    @classmethod
    def zero(cls, ring: ZPlusRing, K: int) -> "RPowerSeries":
        return cls(ring, None, K)

    @classmethod
    def one(cls, ring: ZPlusRing, K: int) -> "RPowerSeries":
        return cls(ring, {0: 1}, K)

    def is_zero(self) -> bool:
        return not self.coeffs

    def __getitem__(self, q_exp: int) -> RElement:
        """The R-coefficient at q^{q_exp} (zero if absent)."""
        return self.coeffs.get(q_exp, self.ring.zero())

    def __add__(self, other: "RPowerSeries") -> "RPowerSeries":
        if not isinstance(other, RPowerSeries):
            return NotImplemented
        if other.ring != self.ring:
            raise ValueError("RPowerSeries: ring mismatch")
        K = min(self.K, other.K)
        out = {n: c for n, c in self.coeffs.items() if n <= K}
        for n, c in other.coeffs.items():
            if n > K:
                continue
            out[n] = out[n] + c if n in out else c
        return RPowerSeries(
            self.ring, {n: c for n, c in out.items() if not c.is_zero()}, K,
        )

    def __sub__(self, other: "RPowerSeries") -> "RPowerSeries":
        return self + (-other)

    def __neg__(self) -> "RPowerSeries":
        return RPowerSeries(
            self.ring, {n: -c for n, c in self.coeffs.items()}, self.K,
        )

    def __mul__(self, other) -> "RPowerSeries":
        if isinstance(other, int):
            if other == 0:
                return RPowerSeries.zero(self.ring, self.K)
            return RPowerSeries(
                self.ring, {n: c * other for n, c in self.coeffs.items()}, self.K,
            )
        if isinstance(other, RElement):
            if other.ring != self.ring:
                raise ValueError("RPowerSeries: ring mismatch in __mul__")
            return RPowerSeries(
                self.ring, {n: c * other for n, c in self.coeffs.items()}, self.K,
            )
        if isinstance(other, RLaurent):
            # Multiply truncated R((q)) by R[q^±]: shift and accumulate.
            if other.ring != self.ring:
                raise ValueError("RPowerSeries: ring mismatch with RLaurent")
            K = self.K
            out: dict[int, RElement] = {}
            for n1, c1 in self.coeffs.items():
                for n2, c2 in other.coeffs.items():
                    n = n1 + n2
                    if n > K:
                        continue
                    p = c1 * c2
                    if p.is_zero():
                        continue
                    out[n] = out[n] + p if n in out else p
            return RPowerSeries(
                self.ring, {n: c for n, c in out.items() if not c.is_zero()}, K,
            )
        if isinstance(other, RPowerSeries):
            if other.ring != self.ring:
                raise ValueError("RPowerSeries: ring mismatch")
            K = min(self.K, other.K)
            out: dict[int, RElement] = {}
            for n1, c1 in self.coeffs.items():
                for n2, c2 in other.coeffs.items():
                    n = n1 + n2
                    if n > K:
                        continue
                    p = c1 * c2
                    if p.is_zero():
                        continue
                    out[n] = out[n] + p if n in out else p
            return RPowerSeries(
                self.ring, {n: c for n, c in out.items() if not c.is_zero()}, K,
            )
        return NotImplemented

    __rmul__ = __mul__

    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            if other == 0:
                return self.is_zero()
            return self.coeffs == {0: RElement(self.ring, {self.ring.one_basis(): other})}
        if not isinstance(other, RPowerSeries):
            return NotImplemented
        return (
            self.ring == other.ring
            and self.K == other.K
            and self.coeffs == other.coeffs
        )

    def __hash__(self):
        raise TypeError("RPowerSeries is unhashable")

    def __repr__(self) -> str:
        if not self.coeffs:
            return f"0 + O(q^{self.K + 1})"
        parts: list[str] = []
        for n in sorted(self.coeffs.keys()):
            c_repr = repr(self.coeffs[n])
            qstr = "" if n == 0 else "q" if n == 1 else f"q^{n}"
            if not qstr:
                parts.append(c_repr)
            elif c_repr == "1":
                parts.append(qstr)
            elif c_repr == "-1":
                parts.append(f"-{qstr}")
            elif " " in c_repr:
                parts.append(f"({c_repr})*{qstr}")
            else:
                parts.append(f"{c_repr}*{qstr}")
        s = parts[0]
        for p in parts[1:]:
            if p.startswith("-"):
                s += " - " + p[1:]
            else:
                s += " + " + p
        return s + f" + O(q^{self.K + 1})"


# ---------------------------------------------------------------------------
# RingHom: a Z₊-ring homomorphism phi : R_1 -> R_2.
#
# Physically, this corresponds to a map of compact groups α : H → G; the
# induced ring hom α* : R(G) → R(H) sends each irrep V ∈ Rep(G) to its
# restriction to H.  For abelian groups (= maximal tori), it sends a
# G-character to its pullback under α^* : Γ̂_G → Γ̂_H.
#
# Used to base-change a `KAlgebra over R_1` into a `KAlgebra over R_2`
# (cf. `kalgebra.KAlgebra.base_change`).
# ---------------------------------------------------------------------------


class RingHom:
    """A homomorphism of Z₊-rings `phi : source -> target`, defined by its
    action on canonical-basis elements (and extended Z-linearly).

    Required properties (caller-enforced; not auto-checked):

      * `phi(1_source) = 1_target`
      * `phi(b_1 · b_2) = phi(b_1) · phi(b_2)`
      * `phi(b⋆) = phi(b)⋆`  (compatibility with Lusztig-Ostrik ⋆)

    The homomorphism extends naturally to `RElement[source] → RElement[target]`,
    `RLaurent[source] → RLaurent[target]` (q-grading preserved), and
    `RPowerSeries[source] → RPowerSeries[target]`.
    """

    __slots__ = ("source", "target", "_on_basis")

    def __init__(
        self,
        source: ZPlusRing,
        target: ZPlusRing,
        on_basis,   # callable: BasisElement → RElement (target)
    ):
        self.source = source
        self.target = target
        self._on_basis = on_basis

    # ------- application to elements -------

    def apply_basis(self, b: BasisElement) -> RElement:
        """`phi(b)` for a single source basis element b."""
        out = self._on_basis(b)
        if not isinstance(out, RElement):
            raise TypeError(
                f"RingHom.on_basis must return an RElement, got {type(out).__name__}"
            )
        if out.ring != self.target:
            raise ValueError("RingHom.on_basis output ring mismatch")
        return out

    def apply_RElement(self, r: RElement) -> RElement:
        """Linear extension of phi to `RElement[source] -> RElement[target]`."""
        if r.ring != self.source:
            raise ValueError("RingHom.apply_RElement: source ring mismatch")
        out = self.target.zero()
        for b, c in r.terms.items():
            out = out + self.apply_basis(b) * c
        return out

    def apply_RLaurent(self, L: RLaurent) -> RLaurent:
        """Apply phi to each R-coefficient; q-grading preserved."""
        if L.ring != self.source:
            raise ValueError("RingHom.apply_RLaurent: source ring mismatch")
        new_coeffs: dict[int, RElement] = {}
        for q_exp, c in L.coeffs.items():
            mapped = self.apply_RElement(c)
            if not mapped.is_zero():
                new_coeffs[q_exp] = mapped
        return RLaurent(self.target, new_coeffs)

    def apply_RPowerSeries(self, P: RPowerSeries) -> RPowerSeries:
        """Apply phi to each R-coefficient; q-grading + truncation order preserved."""
        if P.ring != self.source:
            raise ValueError("RingHom.apply_RPowerSeries: source ring mismatch")
        new_coeffs: dict[int, RElement] = {}
        for q_exp, c in P.coeffs.items():
            mapped = self.apply_RElement(c)
            if not mapped.is_zero():
                new_coeffs[q_exp] = mapped
        return RPowerSeries(self.target, new_coeffs, P.K)

    # ------- compatibility with augmentation + 1-dim reps (Plan 32 T1b / D8) -------
    #
    # A Z₊-ring hom that "forgets part of a flavour symmetry" is a restriction
    # φ = α* of a compact-group hom α : H → G.  Such a φ PRESERVES DIMENSION
    # (ε_target ∘ φ = ε_source, since restriction preserves dim) and carries
    # 1-dim reps to 1-dim reps (so it induces a map Λ(source) → Λ(target) on the
    # lift torsors).  These are *verifiers*, not enforced invariants — a RingHom
    # is a low-level tool (ruling D8); the shipped flavour homs all pass them.

    def verify_preserves_augmentation(self, samples) -> bool:
        """`ε_target ∘ φ == ε_source` on the given source basis `samples`
        (i.e. φ preserves representation dimension)."""
        for b in samples:
            img = self.apply_basis(b)
            d = sum(self.target.dim(c) * coeff for c, coeff in img.terms.items())
            if d != self.source.dim(b):
                return False
        return True

    def verify_preserves_one_dim_reps(self, char_samples=None) -> bool:
        """φ carries 1-dim reps to 1-dim reps: for each `Λ_source` character `f`,
        `φ(ι_source(f))` is a single **group-like** basis element of the target
        (`dim == 1`, invertible) — so φ induces `Λ(source) → Λ(target)`.
        Defaults to the `0` character and the `c` lattice generators."""
        c = self.source.one_dim_rep_rank()
        if char_samples is None:
            char_samples = [(0,) * c] + [
                tuple(1 if i == k else 0 for i in range(c)) for k in range(c)
            ]
        one_t = self.target.one_basis()
        for f in char_samples:
            if len(f) != c:
                return False
            gl = self.source.embed_one_dim_rep(tuple(f))       # group-like in source
            img = self.apply_basis(gl)                          # RElement over target
            if len(img.terms) != 1:
                return False
            (b, coeff), = img.terms.items()
            if coeff != 1 or self.target.dim(b) != 1:
                return False
            if self.target.multiply_basis(b, self.target.star_basis(b)) != {one_t: 1}:
                return False
        return True

    def __repr__(self) -> str:
        return f"RingHom({self.source!r} -> {self.target!r})"


# ------------- concrete factories ----------------



# ---------------------------------------------------------------------------
# R(SU(N)) — the general unitary-flavour rep ring (Plan 30 D8).
# ---------------------------------------------------------------------------


def _distinct_perms(items: tuple):
    """Yield each distinct permutation of `items` exactly once.

    Avoids the `N!` blow-up of `itertools.permutations(...)` + a dedup set
    when `items` has repeats — the common case for SU(N) weight contents
    (e.g. `(1,1,0,0,0)` has `5! = 120` permutations but only `10` distinct)."""
    items = sorted(items)
    n = len(items)
    used = [False] * n
    cur: list = []

    def rec():
        if len(cur) == n:
            yield tuple(cur)
            return
        prev = None
        for i in range(n):
            if used[i] or items[i] == prev:
                continue
            prev = items[i]
            used[i] = True
            cur.append(items[i])
            yield from rec()
            cur.pop()
            used[i] = False

    yield from rec()


@lru_cache(maxsize=None)
def _sun_kostka(lam: tuple, mu: tuple) -> int:
    """Kostka number K_{λμ} (# SSYT of shape λ, content μ) — small DP,
    self-contained (zplus_ring must not import implementations/).

    Memoized two ways: `@lru_cache` reuses repeated `(λ, μ)` across the whole
    process, and the inner horizontal-strip recursion is memoized per call on
    `(shape, idx)` — many strip placements reach the same residual sub-shape,
    so the naive recount was re-deriving identical subproblems (the dominant
    cost in profiling)."""
    lam = tuple(x for x in lam if x)
    mu = tuple(x for x in mu if x)
    if sum(lam) != sum(mu):
        return 0
    if not lam:
        return 1
    memo: dict = {}
    # letters processed last-to-first so each placement is a horizontal strip
    def fill_all(shape, idx):
        if idx < 0:
            return 1 if not any(shape) else 0
        cached = memo.get((shape, idx))
        if cached is not None:
            return cached
        want = mu[idx] if idx < len(mu) else 0
        total = 0
        n = len(shape)

        def rec(i, left, eta):
            nonlocal total
            if i == n:
                if left == 0:
                    total += fill_all(tuple(x for x in eta if x), idx - 1)
                return
            lo = shape[i + 1] if i + 1 < n else 0
            hi = shape[i]
            for e in range(min(lo, hi), hi + 1):
                if e < lo:
                    continue
                take = shape[i] - e
                if take > left:
                    continue
                rec(i + 1, left - take, eta + (e,))

        rec(0, want, ())
        memo[(shape, idx)] = total
        return total

    return fill_all(lam, len(mu) - 1)


class SUNZPlusRing(ZPlusRing):
    """The Z₊-ring `R(SU(N))` for arbitrary `N ≥ 1` (Plan 30 D8: the
    faithful flavour ring of a U-gauge node with `N` fundamentals).

    Basis: dominant SU(N) weights as **partitions with < N rows**
    (tuples, weakly decreasing, trailing zeros stripped; a full
    `N`-row column is the determinant ≡ 1 and is reduced away).
    `()` = trivial; `(1,)` = fundamental; `(1,)*(N−1)` = anti-fundamental.

    Multiplication: Littlewood–Richardson computed by monomial
    convolution + greedy dominance peel in `N` variables (exact,
    self-contained Kostka DP), then column reduction.  Cached.

    Star: complex conjugation `λ* = (λ₁−λ_N, λ₁−λ_{N−1}, …)` (pad to
    `N`, reflect, strip).

    `SUNZPlusRing(2)` ≅ `SU2ZPlusRing` under `(n,) ↔ n`;
    `SUNZPlusRing(3)` ≅ `SU3ZPlusRing` under `(a,b) ↔ Dynkin (a−b, b)`
    — both certified in `tests/test_sun_zplus_ring.py`.
    """

    def __init__(self, N: int) -> None:
        if N < 1:
            raise ValueError(f"SUNZPlusRing: N >= 1 required, got {N}")
        self._N = int(N)
        self._mono_cache: dict = {}
        self._mult_cache: dict = {}

    @property
    def N(self) -> int:
        return self._N

    def __eq__(self, other) -> bool:
        return isinstance(other, SUNZPlusRing) and other._N == self._N

    def __hash__(self):
        return hash(("SUNZPlusRing", self._N))

    def __repr__(self) -> str:
        return f"SUNZPlusRing({self._N})"

    # ----- basis hygiene ------------------------------------------------
    def reduce(self, lam) -> tuple:
        """Normalize a weakly-decreasing non-negative tuple: strip full
        columns (`det ≡ 1` in SU(N)) and trailing zeros."""
        lam = tuple(int(x) for x in lam)
        if any(lam[i] < lam[i + 1] for i in range(len(lam) - 1)) or \
                (lam and lam[-1] < 0):
            raise ValueError(f"SUNZPlusRing: not a partition: {lam}")
        if len(lam) > self._N:
            raise ValueError(
                f"SUNZPlusRing(N={self._N}): more than N rows: {lam}")
        if len(lam) == self._N and lam[-1] > 0:
            lam = tuple(x - lam[-1] for x in lam)
        return tuple(x for x in lam if x)

    # ----- torus embedding (the general SU(N) sibling of SU2/SU3.to_abelian) --
    def to_abelian(self, elt: "RElement",
                   target: "AbelianZPlusRing | None" = None) -> "RElement":
        """Weight diagram of an `RElement` over `AbelianZPlusRing(rank=N-1)`:
        `χ_λ ↦ Σ_{weights w} μ^{proj(w)}` with `proj(w) = (w_1-w_N, …,
        w_{N-1}-w_N)` (kill the diagonal / det — the rank-(N-1) SU(N) torus).
        The general-N sibling of `SU2ZPlusRing` / `SU3ZPlusRing.to_abelian`."""
        if elt.ring is not self:
            raise ValueError("to_abelian: element's ring is not this SUNZPlusRing")
        N = self._N
        if target is None:
            target = AbelianZPlusRing(rank=N - 1)
        if not (isinstance(target, AbelianZPlusRing) and target.rank == N - 1):
            raise ValueError(
                f"to_abelian: target must be AbelianZPlusRing(rank={N - 1})")
        out: dict = {}
        for lam, c in elt.terms.items():
            for w, mult in self._mono(tuple(lam)).items():
                key = tuple(int(w[i]) - int(w[N - 1]) for i in range(N - 1))
                out[key] = out.get(key, 0) + c * mult
        return RElement(target, {k: v for k, v in out.items() if v})

    def from_abelian(self, u_relt: "RElement",
                     allow_virtual: bool = True) -> "RElement":
        """Inverse of `to_abelian`: un-branch a Weyl(`S_N`)-symmetric μ-Laurent
        over `AbelianZPlusRing(rank=N-1)` into SU(N) characters (lift each
        projected coordinate to a U(N) weight, `decompose`, drop the level via
        `split_su_level`).  Accepts **virtual** (signed) characters."""
        import sun_characters as _SC
        N = self._N
        if not isinstance(u_relt.ring, AbelianZPlusRing) or u_relt.ring.rank != N - 1:
            raise TypeError(
                f"from_abelian expected RElement over AbelianZPlusRing(rank={N - 1})")
        poly: dict = {}
        for basis, c in u_relt.terms.items():
            w = tuple(int(x) for x in basis) + (0,)   # lift to U(N) weight
            poly[w] = poly.get(w, 0) + int(c)
        poly = {w: c for w, c in poly.items() if c}
        out: dict = {}
        for wdom, mult in _SC.decompose(N, poly).items():
            if not mult:
                continue
            part, _lvl = _SC.split_su_level(tuple(wdom))
            out[tuple(part)] = out.get(tuple(part), 0) + int(mult)
        return RElement(self, {p: c for p, c in out.items() if c})

    def _mono(self, lam: tuple) -> dict:
        """`s_λ(x_1..x_N)` as `{N-tuple weight: multiplicity}` (Kostka)."""
        key = tuple(lam)
        hit = self._mono_cache.get(key)
        if hit is not None:
            return hit
        N = self._N
        full = key + (0,) * (N - len(key))
        out: dict = {}
        total = sum(full)
        # enumerate dominant contents μ (partitions of |λ| with ≤ N parts)
        def parts(nrem, maxp, acc):
            if len(acc) > N:
                return
            if nrem == 0:
                mu = tuple(acc)
                k = _sun_kostka(full, mu)
                if k:
                    for p in _distinct_perms(mu + (0,) * (N - len(mu))):
                        out[p] = out.get(p, 0) + k
                return
            for p in range(min(maxp, nrem), 0, -1):
                parts(nrem - p, p, acc + [p])

        parts(total, total if total else 1, [])
        if total == 0:
            out[(0,) * N] = 1
        self._mono_cache[key] = out
        return out

    # ----- the ZPlusRing contract ----------------------------------------
    def one_basis(self) -> tuple:
        return ()

    def multiply_basis(self, b1, b2) -> dict:
        b1 = self.reduce(b1)
        b2 = self.reduce(b2)
        key = (b1, b2)
        hit = self._mult_cache.get(key)
        if hit is not None:
            return dict(hit)
        N = self._N
        # Racah–Speiser (Brauer–Klimyk): iterate the weight diagram of the
        # *smaller-dimensional* factor and reflect each `λ + w + ρ` into the
        # dominant chamber — O(dim) terms, versus the O(dim·dim) full-character
        # product-then-peel this replaces.  Same algorithm the bespoke SU3/SU4
        # rings use (Klimyk shift-and-reflect), generalised to all N.
        if self.dim(b1) <= self.dim(b2):
            mu, lam = b1, b2          # iterate the smaller factor (mu)
        else:
            mu, lam = b2, b1
        lam_full = lam + (0,) * (N - len(lam))
        rho = tuple(range(N - 1, -1, -1))          # ρ = (N-1, …, 1, 0)
        out: dict = {}
        for w, m in self._mono(mu + (0,) * (N - len(mu))).items():
            v = [lam_full[i] + w[i] + rho[i] for i in range(N)]
            if len(set(v)) < N:
                continue              # λ+w+ρ on a Weyl wall ⇒ the term vanishes
            # sign of the permutation sorting v into strictly-decreasing order
            sign = 1
            for i in range(N):
                for j in range(i + 1, N):
                    if v[i] < v[j]:
                        sign = -sign
            dom = sorted(v, reverse=True)
            nu_full = tuple(dom[i] - rho[i] for i in range(N))
            nu = self.reduce(tuple(x - nu_full[-1] for x in nu_full))  # det-reduce
            out[nu] = out.get(nu, 0) + sign * m
        out = {k: c for k, c in out.items() if c}
        if any(c < 0 for c in out.values()):
            raise RuntimeError(
                f"SUNZPlusRing: negative multiplicity in {b1}⊗{b2}: {out}")
        self._mult_cache[key] = dict(out)
        return out

    def star_basis(self, b) -> tuple:
        b = self.reduce(b)
        if not b:
            return ()
        full = b + (0,) * (self._N - len(b))
        top = full[0]
        return self.reduce(tuple(top - x for x in reversed(full)))

    # ----- the merged surface (Plan 30; the sun_characters twin folded
    # in here 2026-06-12 — one ring, two cross-certifying engines: this
    # class's Kostka-DP LR vs sun_characters' Weyl-denominator toolkit,
    # pinned equal in tests/test_sun_zplus_ring.py) -------------------

    @property
    def m(self) -> int:
        """Alias of `N` (the `sun_characters` attribute name)."""
        return self._N

    def _validate(self, b):
        """STRICT basis check (raises on anything but a trimmed partition
        with < N positive parts) — the `sun_characters` semantics; the
        lenient input normalizer is `reduce`."""
        ok = (isinstance(b, tuple) and len(b) < self._N
              and all(isinstance(x, int) and x > 0 for x in b)
              and all(b[i] >= b[i + 1] for i in range(len(b) - 1)))
        if not ok:
            raise ValueError(
                f"SUNZPlusRing({self._N}) basis is a partition with "
                f"< {self._N} positive parts; got {b!r}")

    def character(self, b) -> dict:
        """The torus character of irrep `b` as `{N-tuple weight: int}`
        (the `λ_N = 0` representative) — the embedding
        `R(SU(N)) ↪ R(T)^{S_N}`."""
        return dict(self._mono(self.reduce(b)))

    def dim(self, b) -> int:
        return sum(self.character(b).values())

    def one_dim_rep_rank(self) -> int:
        return 0   # SU(N) is semisimple: only the trivial 1-dim rep

    def embed_one_dim_rep(self, f) -> tuple:
        if tuple(f) != ():
            raise ValueError(f"SUNZPlusRing({self._N}): Λ has rank 0; got {f!r}")
        return self.one_basis()

def identity_hom(R: ZPlusRing) -> RingHom:
    """The identity ring homomorphism R → R."""
    return RingHom(R, R, lambda b: R.basis_element(b))


def augmentation_hom(source: ZPlusRing) -> RingHom:
    """The **augmentation** (forgetful) homomorphism `ε : R → Z`
    (= `TrivialZPlusRing`), sending each canonical basis element to its
    **dimension**:  `b ↦ dim(b)·()`.

    Physically: forget the flavour group `G_f` — `α : {e} ↪ G_f` the inclusion
    of the trivial subgroup, and `ε = α^*` is the representation-ring
    augmentation (the Frobenius–Perron dimension of the Z₊-ring).  For an
    abelian torus every irrep is 1-dimensional, so this is the classic
    `μ^f ↦ 1` "gauge quotient / set μ → 1" specialization — the previous
    `AbelianZPlusRing`-only signature is exactly that special case.  Aligns
    with the bundle stack's `CoulombAlgebra.schur_index` output.

    Generalised to any `ZPlusRing` for the Plan-32 forgetful map.
    """
    target = TrivialZPlusRing()
    ob = target.one_basis()    # the single basis element ()
    return RingHom(source, target, lambda b: RElement(target, {ob: source.dim(b)}))


def restriction_hom(
    source: AbelianZPlusRing,
    target: AbelianZPlusRing,
    matrix,   # rank_source × rank_target int matrix; row i = α*(e_i)
) -> RingHom:
    """The restriction `R(U(1)^n) → R(U(1)^k)` induced by a group hom
    `α : U(1)^k → U(1)^n` (as a `k × n` integer matrix `M`, with the
    standard convention that `α^*(e_i) = M[i, :]` ∈ Γ̂_H = Z^k).

    Note the dual direction: the ring hom is parametrized by the
    *pullback* `α^*`, which goes Γ̂_G = Z^n → Γ̂_H = Z^k.  We supply it
    here as the matrix `α^*` directly (a `n × k` matrix would also work
    — caller convention).  Concretely: a source basis tuple `v ∈ Z^n` is
    sent to `M @ v ∈ Z^k`.
    """
    if len(matrix) != target.rank or any(len(row) != source.rank for row in matrix):
        raise ValueError(
            f"restriction_hom: matrix must be {target.rank}×{source.rank}; "
            f"got {len(matrix)}×{len(matrix[0]) if matrix else 0}"
        )

    def _on_basis(b: tuple[int, ...]) -> RElement:
        if len(b) != source.rank:
            raise ValueError(
                f"restriction_hom.apply: basis tuple has length {len(b)}, "
                f"expected {source.rank}"
            )
        new_b = tuple(
            sum(matrix[i][j] * b[j] for j in range(source.rank))
            for i in range(target.rank)
        )
        return target.basis_element(new_b)

    return RingHom(source, target, _on_basis)


# ---------------------------------------------------------------------------
# Flavour-GROWING homs (Plan 32): the duals of augmentation / restriction.
#
# `augmentation` (ε : R → Z) and `restriction` (R(G) → R(H)) *forget* flavour
# — they shrink the coefficient ring.  The two homs below *grow* it, so that a
# flow whose IR is "the same algebra over a bigger flavour ring" is reached
# from a smaller-ring sample by `base_change` (NOT by `add_flavour`, which adds
# flavour *charge* / label coordinates — see `KAlgebra.add_flavour`).  Both are
# coefficient-ring inclusions carrying the unit to the unit, so `base_change`
# through them is a faithful enlargement.
# ---------------------------------------------------------------------------


def unit_hom(target: ZPlusRing) -> RingHom:
    """The **unit** (Z-algebra structure) hom `Z = TrivialZPlusRing ↪ R`,
    embedding the trivial ring as the identity character: `() ↦ χ₀ =
    R.one_basis()` (a single group-like basis element, `dim = 1`).

    The canonical **flavour-growing** hom and the **section** of the
    augmentation `ε : R → Z` (`ε ∘ unit = id_Z`, since `dim(χ₀) = 1`): where
    `augmentation` / `restriction_hom` *forget* a flavour symmetry (shrinking
    the coefficient ring), `unit_hom` *adjoins* a whole flavour ring to an
    unflavoured algebra.  `A.base_change(unit_hom(R))` reinterprets a KAlgebra
    over `Z` (unflavoured) as one over `R`, every `Z`-coefficient `c` carried
    as the multiple `c·χ₀` — the functorial way to grow flavour *without*
    adding flavour charge (contrast `add_flavour`, which adds label
    coordinates).

    Augmentation- and 1-dim-rep-preserving, so it passes the `RingHom` flavour
    verifiers for any `target`."""
    source = TrivialZPlusRing()
    ob_s = source.one_basis()                  # the single basis element ()
    one_t = target.one_basis()                 # χ₀ in the target

    def _on_basis(b):
        if b != ob_s:
            raise ValueError(
                f"unit_hom: source is TrivialZPlusRing (basis {{()}}); got {b!r}"
            )
        return target.basis_element(one_t)

    return RingHom(source, target, _on_basis)


def tensor_inclusion_hom(tensor: "TensorZPlusRing", slot: int = 0) -> RingHom:
    """The **tensor-factor inclusion** `R_slot ↪ R_1 ⊗ ⋯ ⊗ R_k` of one factor
    of a `TensorZPlusRing`, `b ↦ χ₀ ⊗ ⋯ ⊗ b (slot) ⊗ ⋯ ⊗ χ₀` (the spectator
    factors held at their identity character).

    A **flavour-growing** hom — the coproduct-side inclusion: it adjoins the
    spectator factors `R_{j≠slot}` to the coefficient ring without touching
    `R_slot`.  `A.base_change(tensor_inclusion_hom(TensorZPlusRing(R, R'), 0))`
    grows a KAlgebra over `R` to one over `R ⊗ R'` (the "any `R` into a tensor
    of `R` and `R'`" case).  Augmentation-preserving (`dim(χ₀) = 1` on every
    spectator) and 1-dim-rep-preserving.

    `slot` selects the factor (default the first); the source ring is
    `tensor.factors[slot]`."""
    if not isinstance(tensor, TensorZPlusRing):
        raise TypeError("tensor_inclusion_hom: `tensor` must be a TensorZPlusRing")
    k = len(tensor.factors)
    if not (0 <= slot < k):
        raise ValueError(
            f"tensor_inclusion_hom: slot {slot} out of range [0, {k})"
        )
    source = tensor.factors[slot]
    ones = list(tensor.one_basis())

    def _on_basis(b):
        key = tuple(ones[:slot]) + (b,) + tuple(ones[slot + 1:])
        return tensor.basis_element(key)

    return RingHom(source, tensor, _on_basis)


def so3_to_u1_hom(
    source: "SO3ZPlusRing | None" = None,
    target: "AbelianZPlusRing | None" = None,
) -> RingHom:
    """The "forget SO(3) → keep U(1) maximal torus" Z₊-ring homomorphism
    `R(SO(3)) → R(U(1)) = Z[μ^±]`.

    Physically: H = U(1) maximal torus, α : H ↪ G = SO(3) inclusion;
    the induced ring hom α* : R(SO(3)) → R(U(1)) sends each SO(3) irrep
    to its restriction to the maximal torus, which is a sum of U(1)
    characters:

        χ_j ↦ μ^j + μ^{j-1} + ⋯ + μ^{-j}.

    This is exactly `SO3ZPlusRing.to_abelian()`, wrapped as a `RingHom`.

    Formalises the "SO(3) → U(1) symmetry reduction" relating an
    SO(3)-flavoured K-algebra to a conventional U(1)-flavoured one
    via base-change of the coefficient ring.
    """
    if source is None:
        source = SO3ZPlusRing()
    if target is None:
        target = AbelianZPlusRing(rank=1)
    if not isinstance(source, SO3ZPlusRing):
        raise TypeError("so3_to_u1_hom: source must be SO3ZPlusRing")
    if not (isinstance(target, AbelianZPlusRing) and target.rank == 1):
        raise TypeError("so3_to_u1_hom: target must be AbelianZPlusRing(rank=1)")

    def _on_basis(j: int) -> RElement:
        if not isinstance(j, int) or j < 0:
            raise ValueError(f"so3_to_u1_hom: basis j must be ℕ_0; got {j}")
        terms: dict[tuple[int], int] = {}
        for k in range(-j, j + 1):
            terms[(k,)] = 1
        return RElement(target, terms)

    return RingHom(source, target, _on_basis)


def u1_weyl_to_so3(
    elt: RElement,
    target: "SO3ZPlusRing | None" = None,
) -> RElement:
    """Convert a Weyl-symmetric `RElement` over `AbelianZPlusRing(rank=1)`
    (i.e. an element of Z[μ^±] satisfying μ ↔ μ⁻¹ symmetry) into the
    equivalent (possibly virtual) element over `SO3ZPlusRing`.

    Inverse of `so3_to_u1_hom` on the Weyl-invariant subring.
    Uses the recursion

        μ^k + μ^{-k}  =  χ_k − χ_{k-1}      (k ≥ 1),
        μ^0           =  χ_0.

    Equivalently, peel from the outermost μ^k inward.  Raises
    `ValueError` if the input is not Weyl-symmetric.
    """
    if not isinstance(elt, RElement):
        raise TypeError(f"u1_weyl_to_so3: expected RElement, got {type(elt).__name__}")
    if not (isinstance(elt.ring, AbelianZPlusRing) and elt.ring.rank == 1):
        raise TypeError(
            "u1_weyl_to_so3: source must be RElement over AbelianZPlusRing(rank=1)"
        )
    if target is None:
        target = SO3ZPlusRing()
    if not isinstance(target, SO3ZPlusRing):
        raise TypeError("u1_weyl_to_so3: target must be SO3ZPlusRing")

    # Extract μ-power → coefficient dict.
    mu_coeffs: dict[int, int] = {}
    for (k,), c in elt.terms.items():
        if c != 0:
            mu_coeffs[k] = c
    # Verify Weyl symmetry.
    for k in list(mu_coeffs.keys()):
        if k != 0 and mu_coeffs.get(k, 0) != mu_coeffs.get(-k, 0):
            raise ValueError(
                f"u1_weyl_to_so3: input not Weyl-symmetric; "
                f"coefficient of μ^{k} = {mu_coeffs.get(k, 0)} but "
                f"coefficient of μ^{-k} = {mu_coeffs.get(-k, 0)}"
            )
    # Convert to χ_j basis via μ^k + μ^{-k} = χ_k − χ_{k-1} for k ≥ 1,
    # and μ^0 = χ_0.
    chi_coeffs: dict[int, int] = {}
    if 0 in mu_coeffs:
        chi_coeffs[0] = mu_coeffs[0]
    for k in sorted(p for p in mu_coeffs if p > 0):
        cnt = mu_coeffs[k]
        chi_coeffs[k] = chi_coeffs.get(k, 0) + cnt
        chi_coeffs[k - 1] = chi_coeffs.get(k - 1, 0) - cnt
    chi_coeffs = {j: c for j, c in chi_coeffs.items() if c != 0}
    return RElement(target, chi_coeffs)


def so3_to_su2_hom(
    source: "SO3ZPlusRing | None" = None,
    target: "SU2ZPlusRing | None" = None,
) -> RingHom:
    """The inclusion `R(SO(3)) ↪ R(SU(2))` as the even-n subring.

    Physically: SO(3) = SU(2) / Z_2 is the quotient by the centre; the
    natural ring map is the inclusion of representation rings going the
    other way: every SO(3) rep lifts to a (necessarily integer-spin =
    even-n) SU(2) rep.

        χ_j^{SO(3)} ↦ χ_{2j}^{SU(2)}    (set μ_{SO(3)} = μ_{SU(2)}²).
    """
    if source is None:
        source = SO3ZPlusRing()
    if target is None:
        target = SU2ZPlusRing()
    if not isinstance(source, SO3ZPlusRing):
        raise TypeError("so3_to_su2_hom: source must be SO3ZPlusRing")
    if not isinstance(target, SU2ZPlusRing):
        raise TypeError("so3_to_su2_hom: target must be SU2ZPlusRing")

    def _on_basis(j: int) -> RElement:
        if not isinstance(j, int) or j < 0:
            raise ValueError(f"so3_to_su2_hom: basis j must be ℕ_0; got {j}")
        return target.basis_element(2 * j)

    return RingHom(source, target, _on_basis)


def su2_to_u1_hom(
    source: "SU2ZPlusRing | None" = None,
    target: "AbelianZPlusRing | None" = None,
) -> RingHom:
    """The Cartan restriction `R(SU(2)) → R(U(1)) = Z[z^±]`:

        χ_n  ↦  Σ_{k=0}^{n} z^{n-2k}     (the weights of the (n+1)-dim irrep)

    in the **fundamental normalisation** (the doublet has weights z^{±1};
    the adjoint-normalised fugacity is μ = z²).  This is the hom that
    restricts an SU(2)-flavoured KAlgebra to its maximal torus, e.g.
    `a1d3.base_change(su2_to_u1_hom())` = the z-refined μ-flavoured
    hexagon face (Finalized.md, hexagon pair)."""
    if source is None:
        source = SU2ZPlusRing()
    if target is None:
        target = AbelianZPlusRing(rank=1)
    if not isinstance(source, SU2ZPlusRing):
        raise TypeError("su2_to_u1_hom: source must be SU2ZPlusRing")
    if not isinstance(target, AbelianZPlusRing) or target.rank != 1:
        raise TypeError("su2_to_u1_hom: target must be AbelianZPlusRing(1)")

    def _on_basis(n: int) -> RElement:
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"su2_to_u1_hom: basis n must be ℕ_0; got {n}")
        return RElement(target,
                        {(n - 2 * k,): 1 for k in range(n + 1)})

    return RingHom(source, target, _on_basis)


def su3_to_su2u1_hom(
    source: "SU3ZPlusRing | None" = None,
    target: "SU2xU1ZPlusRing | None" = None,
) -> RingHom:
    """The branching `R(SU(3)) → R(SU(2)×U(1))` for the block embedding
    `SU(2)×U(1) ⊂ SU(3)` with `U(1) = diag(e^{iθ}, e^{iθ}, e^{-2iθ})`,
    normalised so the fundamental branches as

        3 = χ_{(1,0)}  ↦  2_{+1} ⊕ 1_{-2}     (i.e. (1, +1) + (0, -2)).

    Physically (regen note for a1d4): the D₄ AD theory's SU(3)-enhanced
    flavour restricted to the SU(2)×U(1) visible to the A₁D₄ chart, so
    `a1d4 = SU3ADKAlg.base_change(su3_to_su2u1_hom())`.

    Computed exactly from the weight system: in the fundamental-orbit
    weight basis of `SU3ZPlusRing` (fundamental weights (1,0), (0,1),
    (-1,-1)) the branching is the linear map on the weight lattice

        (a, b)  ↦  (su2_weight, u1_charge) = (a - b, a + b),

    after which the SU(2)-weight multiset at each fixed U(1) charge is
    resolved into χ_k's by top-weight peeling.  Branching positivity
    (multiplicities ≥ 0) and the exact dimension count
    `Σ (k+1)·mult = (p+1)(q+1)(p+q+2)/2` are asserted per irrep.
    """
    if source is None:
        source = SU3ZPlusRing()
    if target is None:
        target = SU2xU1ZPlusRing()
    if not isinstance(source, SU3ZPlusRing):
        raise TypeError("su3_to_su2u1_hom: source must be SU3ZPlusRing")
    if not isinstance(target, SU2xU1ZPlusRing):
        raise TypeError("su3_to_su2u1_hom: target must be SU2xU1ZPlusRing")

    cache: dict[tuple[int, int], RElement] = {}

    def _on_basis(pq: tuple[int, int]) -> RElement:
        pq = (int(pq[0]), int(pq[1]))
        if pq in cache:
            return cache[pq]
        p, q = pq
        if p < 0 or q < 0:
            raise ValueError(f"su3_to_su2u1_hom: Dynkin labels must be "
                             f"≥ 0; got {pq}")
        # weight multiset, branched to (su2 weight, u1 charge)
        by_charge: dict[int, dict[int, int]] = {}
        for (a, b), m in source._irrep_weights(pq).items():
            w, c = a - b, a + b
            by_charge.setdefault(c, {})[w] = (
                by_charge.get(c, {}).get(w, 0) + m)
        # per-charge top-weight peel into SU(2) characters
        out: dict[tuple[int, int], int] = {}
        for c, wts in sorted(by_charge.items()):
            rem = {w: m for w, m in wts.items() if m}
            while rem:
                k = max(abs(w) for w in rem)
                mult = rem.get(k, 0)
                if mult <= 0:
                    raise ValueError(
                        f"su3_to_su2u1_hom: non-positive multiplicity "
                        f"{mult} at top weight {k} (charge {c}) of "
                        f"χ_{pq} — weight bookkeeping error")
                out[(k, c)] = out.get((k, c), 0) + mult
                for w in range(-k, k + 1, 2):
                    nm = rem.get(w, 0) - mult
                    if nm < 0:
                        raise ValueError(
                            f"su3_to_su2u1_hom: negative remainder at "
                            f"weight {w} (charge {c}) of χ_{pq}")
                    if nm:
                        rem[w] = nm
                    else:
                        rem.pop(w, None)
        dim = sum((k + 1) * m for (k, _c), m in out.items())
        want = (p + 1) * (q + 1) * (p + q + 2) // 2
        if dim != want:
            raise ValueError(
                f"su3_to_su2u1_hom: dimension mismatch for χ_{pq}: "
                f"branched {dim} vs dim {want}")
        res = RElement(target, out)
        cache[pq] = res
        return res

    return RingHom(source, target, _on_basis)


__all__ = [
    "BasisElement",
    "ZPlusRing",
    "RElement",
    "RLaurent",
    "RPowerSeries",
    "TrivialZPlusRing",
    "AbelianZPlusRing",
    "SU2ZPlusRing",
    "SU2xU1ZPlusRing",
    "SO3ZPlusRing",
    "SU3ZPlusRing",
    "SU4ZPlusRing",
    "SUNZPlusRing",
    "RingHom",
    "identity_hom",
    "augmentation_hom",
    "restriction_hom",
    "so3_to_u1_hom",
    "so3_to_su2_hom",
    "su2_to_u1_hom",
    "su3_to_su2u1_hom",
    "u1_weyl_to_so3",
]
