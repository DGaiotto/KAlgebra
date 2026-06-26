"""`QuantumTorusKAlg` — the quantum torus K-algebra parameterized by Γ.

The K-algebra `A_𝖖[QT(Γ, B)]` of the quantum torus on a lattice `Γ = Z^n`
with antisymmetric integer pairing `B` (possibly degenerate).  Defining
relations:

    X_γ · X_{γ'} = q^{<γ, γ'>} X_{γ + γ'}.

Conventions:

* Canonical-basis labels are **full Γ-tuples** of length `n` -- matching
  the convention of `QuantumTorusZ2KAlg` and `BPSKAlgebra`.  This means
  two labels `γ` and `γ + γ_f` for `γ_f ∈ Γ_f := ker(B)` are *distinct*
  basis elements that differ by a μ-monomial coefficient under the
  trace.
* Multiplication: `L_a · L_b = q^{<a, b>} L_{a + b}`.  No μ-shifts in
  the structure constants -- flavour appears only through the trace.
* `ρ(L_a) = L_{-a}`.
* Bar = `q ↔ q⁻¹`, identity on labels.
* Trace: `Tr(L_γ) = δ_{[γ], 0_{Γ_g}} · μ^{flav(γ)} · (q²; q²)_∞^{rk Γ_g}`,
  i.e. the trace is non-zero exactly on `Γ_f` (modulo the gauge cosets),
  and on `Γ_f` it carries the μ-monomial of the flavour exponent.

Coefficient ring `R`:

* `R = TrivialZPlusRing()` if `B` is non-degenerate.
* `R = AbelianZPlusRing(rank=f)` with `f = rk(Γ_f)` if `B` is degenerate.

Flavour generators (= `X_{γ_f}` for `γ_f ∈ Γ_f`) are first-class basis
elements `L_{γ_f}`.  Two convenience accessors are provided:

* `flavour_generator_label(flavour_exp)` -- the label `γ_f ∈ Γ` for the
  given exponent tuple in the SNF kernel basis.
* `gauge_class(γ)` / `flavour_part(γ)` -- the SNF decomposition of any
  `γ ∈ Γ` into (gauge-coords, flavour-coords).

Flavour-lift coordinate and forget:

* `r_label_decompose(γ) = (section, flavour_key)` is the flavour-lift
  coordinate.  **Before forgetting**, the section is a *section of the
  projection* `π : Γ ↠ Γ_g = Γ/Γ_f` — the ρ-equivariant Z-linear lift
  `s : Γ_g → Γ` with image `span(sec_basis)`, a copy of `Γ_g` sitting inside
  `Γ` (a genuine source label, as the `embed_R` faithfulness axiom needs).
  `r_label_compose` is `s` (then re-attach the flavour `μ`).
* `embed_R(μ^f) = X_{γ_f} = L_{flavour_generator_label(f)}` is the central
  embedding of the flavour generators (`⟨Γ_f, ·⟩ = 0`, so they are central).
* `forget()` returns `QuantumTorusKAlg(B_g)` — the quantum torus on the
  now-**abstract** lattice `Γ_g`, with the induced (non-degenerate) pairing
  `B_g[i][j] = ⟨sec_basis[i], sec_basis[j]⟩`.  Forgetting *abstracts* the
  section copy into `Γ_g`; the forget map `L_γ ↦ M_{gauge_class(γ)}` is a
  trace-preserving homomorphism.

`QuantumTorusZ2KAlg(KAlgebra)` from `kalgebra_samples.py` is the special
case of this class at `pairing=[[0, 1], [-1, 0]]`.
"""

from __future__ import annotations

from typing import Sequence

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element
from zplus_ring import (
    ZPlusRing,
    RElement,
    RLaurent,
    RPowerSeries,
    TrivialZPlusRing,
    AbelianZPlusRing,
)
from snf_kernel import integer_kernel_and_section, decompose_in_basis
from qpoch import qpoch_infty
from laurent_poly import LaurentPoly


Vec = tuple[int, ...]


def _qpoch_pref_rpowerseries(
    ring: ZPlusRing, exponent: int, K: int,
) -> RPowerSeries:
    """`(q²; q²)_∞^exponent` as `RPowerSeries[ring]` truncated to `q^K`."""
    if exponent == 0:
        return RPowerSeries(ring, {0: 1}, K)
    pref = qpoch_infty(K)
    for _ in range(exponent - 1):
        pref = pref * qpoch_infty(K)
    return RPowerSeries(ring, dict(pref._c), K)


class QuantumTorusKAlg(KAlgebra):
    """The K-algebra of the quantum torus on `(Γ = Z^n, B)`.

    See module docstring for the conventions.  Labels are full
    Γ-tuples; flavour absorbs into the coefficient ring at the trace
    boundary.
    """

    def __init__(self, pairing: Sequence[Sequence[int]]):
        n = len(pairing)
        if any(len(row) != n for row in pairing):
            raise ValueError("pairing must be a square matrix")
        for i in range(n):
            for j in range(n):
                if int(pairing[i][j]) != -int(pairing[j][i]):
                    raise ValueError("pairing must be antisymmetric")
        self._pairing: list[list[int]] = [
            [int(x) for x in row] for row in pairing
        ]
        self._rank = n
        self._ker_basis, self._sec_basis = integer_kernel_and_section(
            self._pairing
        )
        self._flavour_rank = len(self._ker_basis)
        self._gauge_rank = len(self._sec_basis)
        self._R: ZPlusRing = (
            TrivialZPlusRing() if self._flavour_rank == 0
            else AbelianZPlusRing(rank=self._flavour_rank)
        )
        # Cache the per-label SNF gauge/flavour split (the bases are fixed here).
        self._decomp_cache: dict[Vec, tuple[Vec, Vec]] = {}

    # ----- accessors -----------------------------------------------------

    @property
    def pairing(self) -> list[list[int]]:
        return [list(row) for row in self._pairing]

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def gauge_rank(self) -> int:
        return self._gauge_rank

    @property
    def flavour_rank(self) -> int:
        return self._flavour_rank

    @property
    def section_basis(self) -> list[Vec]:
        """Z-basis lifting `Γ_g → Γ` (one Vec per gauge direction)."""
        return [tuple(b) for b in self._sec_basis]

    @property
    def kernel_basis(self) -> list[Vec]:
        """Z-basis of `Γ_f = ker(B)` as Vecs in Γ."""
        return [tuple(b) for b in self._ker_basis]

    # ----- KAlgebra contract ---------------------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self) -> Vec:
        return tuple([0] * self._rank)

    def multiply(self, a, b) -> Element:
        a = self._check_label(a)
        b = self._check_label(b)
        target = tuple(x + y for x, y in zip(a, b))
        q_exp = self._bracket(a, b)
        coeff = LaurentPoly({q_exp: 1})
        return Element({target: coeff})

    def r_label_decompose(self, label):
        """The flavour-lift coordinate `(section, flavour_key)` (replaces the
        retired `_label_section_decompose`).

        **Before forgetting**, the section is a *section of the projection*
        `π : Γ ↠ Γ_g = Γ/Γ_f`: the ρ-equivariant Z-linear lift `s : Γ_g → Γ`
        with image `span(sec_basis)` — a copy of `Γ_g` sitting *inside* `Γ`.
        So the section label is the Γ-tuple `s(π(γ)) = Σ_i c_i·sec_basis[i]`
        (a genuine *source* label, which is what the `embed_R` faithfulness
        axiom `embed_R(r)·L_section == L_a` needs), and the flavour key is the
        SNF kernel-coordinate tuple of `μ^{flav}`.  `Γ_g` becomes an *abstract*
        lattice only after `forget()` (= `QuantumTorusKAlg(B_g)`); `forget`'s
        relabel `π` is the left inverse of this section.
        """
        a = self._check_label(label)
        sec_c, flav_c = self._decompose(a)
        n = self._rank
        sec_label = tuple(
            sum(sec_c[i] * self._sec_basis[i][k] for i in range(len(sec_c)))
            for k in range(n)
        )
        # AbelianZPlusRing / TrivialZPlusRing: flav_c is the basis key itself.
        return sec_label, tuple(flav_c)

    def r_label_compose(self, section, flavour_key):
        """Inverse of `r_label_decompose`: the section map (lift) `s`, then
        re-attach the flavour `μ^{flavour_key}`.  `section` is a Γ-tuple in
        `span(sec_basis)` (a section of the projection); add the kernel lift
        `Σ_j flavour_key[j]·ker_basis[j] ∈ Γ_f`.  A direct lattice sum — no
        `embed_R`/`multiply` round-trip (and `⟨Γ_f, ·⟩ = 0` keeps it
        q-phase-free)."""
        section = self._check_label(section)
        flavour_key = tuple(int(x) for x in flavour_key)
        if len(flavour_key) != self._flavour_rank:
            raise ValueError(
                f"r_label_compose: flavour_key length {len(flavour_key)} != "
                f"flavour_rank {self._flavour_rank}"
            )
        n = self._rank
        flav_lift = tuple(
            sum(flavour_key[j] * self._ker_basis[j][k]
                for j in range(self._flavour_rank))
            for k in range(n)
        )
        return tuple(section[k] + flav_lift[k] for k in range(n))

    def embed_R(self, r) -> Element:
        """Central embedding `ι : R ↪ A_𝖖`: the flavour fugacity `μ^f`
        (an `R`-basis element keyed by the SNF kernel coordinate `f`) maps to
        the flavour generator `X_{γ_f} = L_{flavour_generator_label(f)}`,
        `γ_f ∈ Γ_f = ker(B)`.  Since `⟨Γ_f, ·⟩ = 0` these are central, so
        `embed_R(μ^f)·L_{section} = L_{γ_f + section}` with no q-phase — exactly
        the faithfulness axiom `embed_R(r_coeff(a))·L_{section(a)} == L_a`.

        (The base default only embeds `1_R`; without this override
        `from_R_form` and `verify_embed_section_roundtrip` failed for any
        non-trivial flavour.)"""
        R = self._R
        if not isinstance(r, RElement) or r.ring != R:
            raise TypeError(
                "embed_R: argument must be an RElement over coefficient_ring()"
            )
        out = Element.zero()
        for key, c in r.terms.items():
            if c == 0:
                continue
            gen = self.flavour_generator_label(key)
            out = out + Element({gen: LaurentPoly({0: 1})}) * c
        return out

    def rho(self, a) -> Vec:
        return tuple(-x for x in self._check_label(a))

    def rho_inverse(self, a) -> Vec:
        return self.rho(a)

    def trace(self, a, K: int = 20) -> RPowerSeries:
        """`Tr(L_γ) = δ_{[γ], 0_{Γ_g}} · μ^{flav(γ)} · (q²; q²)_∞^{rk Γ_g}`.

        Non-zero exactly when γ projects to 0 in `Γ_g` (i.e. γ ∈ Γ_f).
        On `Γ_f`, the trace is the prefactor times the μ-monomial
        encoding the flavour exponent.
        """
        a = self._check_label(a)
        sec_c, flav_c = self._decompose(a)
        if any(s != 0 for s in sec_c):
            return RPowerSeries.zero(self._R, K)
        pref = _qpoch_pref_rpowerseries(self._R, self._gauge_rank, K)
        if self._flavour_rank == 0:
            return pref
        mu = self._R.basis_element(tuple(flav_c))
        # Scale every q-coefficient of `pref` by μ^{flav_c}.
        new_terms: dict[int, RElement] = {}
        for q_exp, r in pref.coeffs.items():
            scaled = r * mu
            if not scaled.is_zero():
                new_terms[q_exp] = scaled
        return RPowerSeries(self._R, new_terms, K)

    def forget(self) -> "KAlgebra":
        """The unflavoured algebra with `G_f` forgotten: the quantum torus on
        the **abstract** gauge lattice `Γ_g = Γ/Γ_f`, i.e.
        `QuantumTorusKAlg(B_g)` with `B_g[i][j] = ⟨sec_basis[i], sec_basis[j]⟩`
        the (non-degenerate) induced pairing.

        Before forgetting, the section is a *section of the projection*
        `Γ ↠ Γ_g` (a copy of `Γ_g` inside `Γ`, = `r_label_decompose`'s
        section).  `forget` **abstracts** that copy into the standalone
        lattice `Γ_g`: the forget map `L_γ ↦ M_{gauge_class(γ)}` is a
        trace-preserving homomorphism — `⟨Γ_f, ·⟩ = 0` gives
        `⟨a, b⟩ = B_g(π a, π b)` and `gauge_class` is additive, while `ε(μ) = 1`
        makes the trace prefactor descend.  Unflavoured already (`Γ_f = 0`)
        ⟹ `self`."""
        if self._flavour_rank == 0:
            return self
        g = self._gauge_rank
        B_g = [
            [self._bracket(self._sec_basis[i], self._sec_basis[j])
             for j in range(g)]
            for i in range(g)
        ]
        return QuantumTorusKAlg(B_g)

    def forget_label(self, gamma) -> Vec:
        """The forget image of `γ`'s canonical label: its gauge class in the
        abstract `Γ_g` (= `forget()`'s basis label).  `forget(L_γ) =
        M_{forget_label(γ)}` (the relabel `π`, left inverse of the section
        `r_label_compose(·, 0)`)."""
        return self.gauge_class(gamma)

    # ----- flavour-aware accessors ---------------------------------------

    def flavour_generator_label(self, flavour_exp: Sequence[int]) -> Vec:
        """The label `γ_f ∈ Γ` for the flavour generator with given
        exponent in the SNF kernel basis: `γ_f = Σ flavour_exp[i] · ker_basis[i]`.
        """
        flavour_exp = tuple(int(x) for x in flavour_exp)
        if len(flavour_exp) != self._flavour_rank:
            raise ValueError(
                f"flavour_generator_label: expected length-"
                f"{self._flavour_rank} tuple, got length {len(flavour_exp)}"
            )
        result = [0] * self._rank
        for i, c in enumerate(flavour_exp):
            for k in range(self._rank):
                result[k] += c * self._ker_basis[i][k]
        return tuple(result)

    def gauge_class(self, gamma: Sequence[int]) -> Vec:
        """SNF gauge-coordinate tuple of `γ ∈ Γ` (length `gauge_rank`)."""
        gamma = self._check_label(gamma)
        sec_c, _ = self._decompose(gamma)
        return tuple(sec_c)

    def flavour_part(self, gamma: Sequence[int]) -> Vec:
        """SNF flavour-coordinate tuple of `γ ∈ Γ` (length `flavour_rank`)."""
        gamma = self._check_label(gamma)
        _, flav_c = self._decompose(gamma)
        return tuple(flav_c)

    # ----- internals -----------------------------------------------------

    def _check_label(self, a) -> Vec:
        a = tuple(int(x) for x in a)
        if len(a) != self._rank:
            raise ValueError(
                f"label has length {len(a)}, expected rank={self._rank}"
            )
        return a

    def _decompose(self, a: Vec) -> tuple[Vec, Vec]:
        """Cached SNF split of (checked) label `a` into (gauge_coords,
        flavour_coords).  The section/kernel bases are fixed at construction,
        so the split is a pure function of `a`; memoize it, since the
        trace/orthonormality battery revisits the same labels across many
        pairs."""
        cached = self._decomp_cache.get(a)
        if cached is None:
            sec_c, flav_c = decompose_in_basis(
                list(a), self._sec_basis, self._ker_basis,
            )
            cached = (tuple(sec_c), tuple(flav_c))
            self._decomp_cache[a] = cached
        return cached

    def _bracket(self, gamma: Vec, gamma_prime: Vec) -> int:
        return sum(
            self._pairing[i][j] * gamma[i] * gamma_prime[j]
            for i in range(self._rank)
            for j in range(self._rank)
        )

    def __repr__(self) -> str:
        return (
            f"QuantumTorusKAlg(rank={self._rank}, "
            f"gauge_rank={self._gauge_rank}, "
            f"flavour_rank={self._flavour_rank})"
        )
