"""`SU2QuantumTorusKAlg` -- the SU(2)-flavoured quantum-torus K-algebra.

The SU(2) quantum-torus realisation, presented in a basis where the
SU(2) Cartan direction is a single explicit coordinate:

    Γ  =  Γ^σ  ×  Λ_weight(SU(2))

where Λ_weight(SU(2)) = ℤ is the **SU(2) weight lattice** (= ℤ in
fundamental-weight units; the root lattice 2ℤ is a sublattice of
index 2).  σ acts diagonally: σ|_{Γ^σ} = +id, σ(v_Cartan) = -v_Cartan,
so the σ-fixed sublattice is exactly Γ^σ and the σ-anti-fixed
direction Z·v_Cartan = Λ_weight(SU(2)).

The pairing B factors: it is non-trivial on Γ^σ (gauge_pairing) and
zero on Λ_weight (the Cartan direction is central -- B(γ, v_Cartan) = 0
for all γ).  Hence Λ_weight is contained in ker(B).

Canonical Z[q^±]-basis labels
-----------------------------
Labels are `(α, k)` where α ∈ Γ^σ ≅ Z^{n-1} is the gauge cell and
k ∈ ℕ_0 is the SU(2) chi index (= dim k+1 irreducible SU(2) rep,
spin k/2).  Note: in this Z-form parametrisation, k indexes the
σ-orbit (= |weight|) rather than the signed weight; the σ acts on
the full f_3 direction with weight ±k merging to chi_k.

The corresponding canonical-basis element (in the underlying full
quantum torus) is:

    M_{(α, k)}  =  X_α · χ_k(X_{v_Cartan})
                =  X_α · (X_{k·v} + X_{(k-2)·v} + … + X_{-k·v}).

Coefficient ring: `SU2ZPlusRing` -- both integer-spin (k even, dim 1,
3, 5, …) and half-integer-spin (k odd, dim 2, 4, …) SU(2) reps appear
as basis elements of the coefficient ring directly.  No "half-integer-
spin sector special case" as in SO3ZPlusRing.

Multiplication (closed-form, in the QT realisation):

    M_{(α_1, k_1)} · M_{(α_2, k_2)}
      =  q^{B_gauge(α_1, α_2)}  ·  Σ_{r = |k_1−k_2|, step 2}^{k_1+k_2}
                                      M_{(α_1+α_2, r)}.

NB: this closed form is the **QT realisation's** structure; a BPS
realisation (`SU2BPSKAlgebra`, to be added) with a specific BPS spec
would carry an F-vs-X cocycle modifying the structure constants.
The two realisations are isomorphic K-algebras.

ρ on labels
-----------
ρ(α, k) = (-α, k)  (gauge inverted; chi-index unchanged, since SU(2)
reps are self-dual).  ρ² = id.

Trace
-----
Tr(M_{(α, k)}) = δ_{α, 0} · χ_k · (q²;q²)_∞^{gauge_rank}  ∈  R(SU(2))[[q]].
"""

from __future__ import annotations

from typing import Sequence

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element
from zplus_ring import ZPlusRing, RElement, RPowerSeries, SU2ZPlusRing
from laurent_poly import LaurentPoly
from qpoch import qpoch_infty


Vec = tuple[int, ...]
Label = tuple  # (α_0, …, α_{gauge_rank-1}, k)  -- last entry = chi index


class SU2QuantumTorusKAlg(KAlgebra):
    """The SU(2)-flavoured quantum-torus K-algebra on a gauge sublattice
    Γ^σ ≅ Z^gauge_rank with given antisymmetric pairing `gauge_pairing`,
    tensored with R(SU(2)) via the SU(2) Cartan direction.

    Constructor takes only the GAUGE pairing (not the full lattice
    pairing); the SU(2) Cartan direction is implicit and has zero
    pairing with everything (= it's in the centre).
    """

    def __init__(self, gauge_pairing: Sequence[Sequence[int]]):
        n = len(gauge_pairing)
        if any(len(row) != n for row in gauge_pairing):
            raise ValueError("gauge_pairing must be a square matrix")
        for i in range(n):
            for j in range(n):
                if int(gauge_pairing[i][j]) != -int(gauge_pairing[j][i]):
                    raise ValueError(
                        f"gauge_pairing must be antisymmetric; "
                        f"got [{i}][{j}]={gauge_pairing[i][j]}, "
                        f"[{j}][{i}]={gauge_pairing[j][i]}"
                    )
        self._gauge_pairing: list[list[int]] = [
            [int(x) for x in row] for row in gauge_pairing
        ]
        self._gauge_rank = n
        self._R: SU2ZPlusRing = SU2ZPlusRing()

    # -- accessors --------------------------------------------------------

    @property
    def gauge_pairing(self) -> list[list[int]]:
        return [list(row) for row in self._gauge_pairing]

    @property
    def gauge_rank(self) -> int:
        return self._gauge_rank

    @property
    def rank(self) -> int:
        """Total label dimension: gauge_rank + 1 (chi index)."""
        return self._gauge_rank + 1

    def chi(self, k: int) -> Label:
        """The SU(2) χ_k character label (gauge cell at origin).

        χ_0 = identity (dim 1)
        χ_1 = SU(2) fundamental (dim 2, half-integer spin 1/2)
        χ_2 = SU(2) adjoint (dim 3, integer spin 1)
        χ_3 = dim 4 (half-integer spin 3/2)
        ...
        """
        if k < 0:
            raise ValueError(f"chi(k): k must be ≥ 0, got {k}")
        return tuple([0] * self._gauge_rank + [k])

    def M(self, gauge_cell: Sequence[int], k: int) -> Label:
        """Construct label (gauge_cell, k)."""
        if len(gauge_cell) != self._gauge_rank:
            raise ValueError(
                f"gauge_cell has length {len(gauge_cell)}, "
                f"expected {self._gauge_rank}"
            )
        if k < 0:
            raise ValueError(f"k must be ≥ 0, got {k}")
        return tuple(int(x) for x in gauge_cell) + (int(k),)

    def split_label(self, label: Label) -> tuple[Vec, int]:
        """Split a label into (gauge_cell, k)."""
        label = tuple(int(x) for x in label)
        if len(label) != self._gauge_rank + 1:
            raise ValueError(
                f"label has length {len(label)}, "
                f"expected {self._gauge_rank + 1}"
            )
        return label[:-1], label[-1]

    # -- KAlgebra primitives ---------------------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self) -> Label:
        return tuple([0] * (self._gauge_rank + 1))

    def multiply(self, a: Label, b: Label) -> Element:
        """Closed-form multiplication:

            M_{(α_1, k_1)} · M_{(α_2, k_2)}
              = q^{B_gauge(α_1, α_2)}
               · Σ_{r = |k_1 − k_2|, step 2}^{k_1 + k_2}  M_{(α_1+α_2, r)}.

        Pure QT q-twist on the gauge sublattice tensored with SU(2)
        Clebsch-Gordan on the chi-index (step 2 in 2·spin = chi-label).
        """
        a_gauge, k_a = self.split_label(a)
        b_gauge, k_b = self.split_label(b)
        if k_a < 0 or k_b < 0:
            raise ValueError("chi indices must be ≥ 0")
        # q-twist from gauge pairing.
        q_exp = sum(
            self._gauge_pairing[i][j] * a_gauge[i] * b_gauge[j]
            for i in range(self._gauge_rank)
            for j in range(self._gauge_rank)
        )
        out_gauge = tuple(a_gauge[i] + b_gauge[i]
                          for i in range(self._gauge_rank))
        # SU(2) CG: r ranges over |k_a - k_b|, |k_a - k_b|+2, ..., k_a + k_b.
        out_terms: dict[Label, LaurentPoly] = {}
        for r in range(abs(k_a - k_b), k_a + k_b + 1, 2):
            label = out_gauge + (r,)
            out_terms[label] = LaurentPoly({q_exp: 1})
        return Element(out_terms)

    def rho(self, a: Label) -> Label:
        """ρ on labels: gauge inverted, chi-index unchanged (self-dual)."""
        gauge, k = self.split_label(a)
        return tuple(-x for x in gauge) + (k,)

    def rho_inverse(self, a: Label) -> Label:
        """ρ² = id, so ρ⁻¹ = ρ."""
        return self.rho(a)

    def trace(self, a: Label, K: int = 20, **kwargs) -> RPowerSeries:
        """Tr(M_{(α, k)}) = δ_{α, 0} · χ_k · (q²;q²)_∞^{gauge_rank}.

        Non-zero only when the gauge cell is at the origin (= the label
        lies in the centre = R(SU(2)) subalgebra).
        """
        gauge, k = self.split_label(a)
        if any(x != 0 for x in gauge):
            return RPowerSeries(self._R, {}, K)
        pref = qpoch_infty(K)
        for _ in range(self._gauge_rank - 1):
            pref = pref * qpoch_infty(K)
        chi_k = self._R.basis_element(k)
        out_terms: dict[int, RElement] = {}
        for q_exp, q_coef in pref._c.items():
            r = chi_k * q_coef
            if not r.is_zero():
                out_terms[q_exp] = r
        return RPowerSeries(self._R, out_terms, K)

    def r_label_decompose(self, label: Label):
        """The flavour-lift coordinate `(section, k)` (replaces the retired
        `_label_section_decompose`): peel the SU(2) χ-index `k` (the R-basis
        label, spin `k/2`) off the gauge cell.  The section `(α, 0)` is the
        flavour-trivial source label."""
        gauge, k = self.split_label(label)
        if k < 0:
            raise ValueError(f"chi index must be ≥ 0, got {k}")
        return gauge + (0,), k

    def r_label_compose(self, section, k):
        """Inverse of `r_label_decompose`: write the spin `k` into the
        χ-slot of the (flavour-trivial) gauge section.  Direct slot write —
        no `embed_R`/`multiply` round-trip."""
        gauge, _ = self.split_label(section)
        if int(k) < 0:
            raise ValueError(f"chi index must be ≥ 0, got {k}")
        return gauge + (int(k),)

    def embed_R(self, r: RElement) -> Element:
        """Central embedding `ι : R(SU(2)) ↪ A_𝖖`: `χ_k ↦ M_{(0, k)}` (the
        Weyl character at the gauge origin), extended Z-linearly.  These are
        central (the SU(2) Cartan has zero pairing), so
        `embed_R(χ_k)·L_{(α,0)} = M_{(α,k)} = L_label` — the faithfulness
        axiom.  (The base default only embeds `1_R`, so `from_R_form` and the
        faithfulness verifier failed for any non-trivial spin.)"""
        if not isinstance(r, RElement) or r.ring != self._R:
            raise TypeError(
                "embed_R: argument must be an RElement over coefficient_ring()"
            )
        out = Element.zero()
        for k, c in r.terms.items():
            if c == 0:
                continue
            out = out + Element({self.chi(int(k)): LaurentPoly({0: 1})}) * c
        return out

    def forget(self) -> "KAlgebra":
        """Forget the SU(2) flavour: the (unflavoured) quantum torus on the
        gauge sublattice `Γ^σ`, i.e. `QuantumTorusKAlg(gauge_pairing)`.

        The forget map is `L_{(α,k)} ↦ dim(χ_k)·M_α = (k+1)·M_α` (the SU(2)
        augmentation `χ_k ↦ k+1`); it is a trace-preserving homomorphism
        because `dim` is multiplicative across the Clebsch–Gordan sum
        (`Σ_r dim χ_r = (k_a+1)(k_b+1)`)."""
        from quantum_torus_kalgebra import QuantumTorusKAlg
        return QuantumTorusKAlg(self._gauge_pairing)

    def forget_label(self, label: Label) -> Vec:
        """The forget image of a label: its gauge cell `α` (= `forget()`'s
        basis label).  `forget(L_{(α,k)}) = (k+1)·M_{forget_label}`."""
        return self.split_label(label)[0]

    # -- convenience ------------------------------------------------------

    def L(self, label: Label) -> Element:
        """The canonical-basis element M_a as an `Element`."""
        return Element.basis(tuple(int(x) for x in label))

    def __repr__(self) -> str:
        return (
            f"SU2QuantumTorusKAlg(gauge_rank={self._gauge_rank})"
        )
