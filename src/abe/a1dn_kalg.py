"""`A1DnKAlg` -- the [A_1, D_n] Argyres-Douglas K-algebra family for
n ≥ 3, presented in the D_n BPS-quiver chamber with manifest SU(2)
flavour (Weyl = leaf-exchange Z_2 on the two end leaves of the D_n
Dynkin diagram; the half-integer sectors are honest SU(2) irreps).

D_n Dynkin diagram (Bourbaki convention, 0-indexed for Python):

    n = 3:   0 — 1            (= A_3, same as our hexagon)
             |
             2

    n = 4:   0 — 1 — 2
                     |
                     3
             (with 1 trivalent; here we use stem→fork: 0—1, 1—2, 1—3)

    n = 5:   0 — 1 — 2 — 3    (stem 0—1—2, fork at 2: edges 2—3, 2—4)
                     |
                     4

    n general:  stem 0—1—…—(n−3), then (n−3)—(n−2) and (n−3)—(n−1).
                The two end leaves are nodes (n−2) and (n−1).

Pairing convention (matches the [A_1, D_3] hard-coded case):

    B[i][i+1] = 1       for i = 0, …, n − 4    (stem edges)
    B[n−3][n−2] = 1     (trivalent → first leaf)
    B[n−3][n−1] = 1     (trivalent → second leaf)
    other off-diagonals = 0; B is antisymmetric (B[j][i] = −B[i][j]).

Two leaves (n−2) and (n−1) are "identical nodes": same off-{n−2,n−1}
pairings (only B[n−3][n−2] = B[n−3][n−1] = 1) and mutually commuting
(B[n−2][n−1] = 0).  σ = swap (n−2) ↔ (n−1) is the leaf-exchange
Poisson involution; the rank-1 ker(B) is Z·(e_{n−1} − e_{n−2}) (up to
sign), the SU(2) Cartan direction.

Canonical-basis labels: γ ∈ Z^n with γ_{n−2} ≥ γ_{n−1} (canonical iff
the lowest-weight σ-orbit rep).  Equivalently the (a, b, s)
parametrisation:

    a = (γ_0, …, γ_{n−3})         ∈ Z^{n−2}
    b = (γ_{n−2} + γ_{n−1}) / 2   ∈ (1/2)·Z
    s = (γ_{n−2} − γ_{n−1}) / 2   ∈ (1/2)·Z_{≥0}

with parity 2b ≡ 2s (mod 2).  γ recovered as (a_0, …, a_{n−3}, b+s, b−s).

Closed-form multiplication (verified by associativity; the n = 3 case
is anchored to the BPS realisation):

    L_{(a_1, b_1, s_1)} · L_{(a_2, b_2, s_2)}
      = q^{B(mid_1, mid_2)} · Σ_{s = |s_1 − s_2|}^{s_1 + s_2}
                                L_{(a_1+a_2, b_1+b_2, s)}

where mid_i = (a_{i,0}, …, a_{i,n−3}, b_i, b_i) ∈ Γ ⊗ (1/2)·Z is the
σ-fixed midpoint, and B is the D_n pairing.  Concretely

    B(mid_1, mid_2)
      =   Σ_{i = 0}^{n−4}  (a_{1,i} a_{2,i+1} − a_{1,i+1} a_{2,i})
        + 2 (a_{1,n−3} b_2 − b_1 a_{2,n−3}).

ρ on labels: γ ↦ −γ (canonicalised).  ρ² = id; ρ(χ_j) = χ_j.

Trace: not implemented (delegates the closed-form character-identity
work to a separate investigation, same as for `A1D3KAlg`).
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


Vec = tuple[int, ...]
Label = Vec


def _build_d_n_pairing(n: int) -> list[list[int]]:
    """D_n antisymmetric pairing (Bourbaki, 0-indexed)."""
    if n < 3:
        raise ValueError(f"n must be ≥ 3, got {n}")
    B = [[0] * n for _ in range(n)]
    # Stem edges: 0—1, 1—2, …, (n−4)—(n−3).
    for i in range(n - 3):
        B[i][i + 1] = 1
        B[i + 1][i] = -1
    # Trivalent-to-leaves: (n−3)—(n−2) and (n−3)—(n−1).
    B[n - 3][n - 2] = 1
    B[n - 2][n - 3] = -1
    B[n - 3][n - 1] = 1
    B[n - 1][n - 3] = -1
    return B


class A1DnKAlg(KAlgebra):
    """The [A_1, D_n] Argyres-Douglas K-algebra (n ≥ 3) in the D_n
    BPS-quiver chamber.  Standalone: closed-form multiplication, no
    BPS / QT machinery at runtime.

    !!! UNVERIFIED FOR n > 3 !!!
    -----------------------------------
    The closed-form structure constants here were DERIVED for n = 3
    against the BPS realisation and confirmed.  For n > 3 the same
    formula was *conjectured* by generalising the (a, b, s)
    parametrisation; it is confirmed by associativity but NOT yet
    against actual [A_1, D_n] BPS data for n > 3 (the former n = 5
    quantum-torus cross-check was tautological -- same closed form --
    and has been retired with the SO(3) realisations).

    Additionally, D_n for n even has a rank-2 ker(B): the σ-invariant
    subalgebra is SU(2) × U(1)-flavoured (not just SU(2)), and this
    class's `SU2ZPlusRing` coefficient ring captures only the SU(2)
    slice.  Trace correctly raises in that case but multiplication
    is silently restricted to the SU(2)-symmetric labels.

    Use A1D3KAlg for the verified n = 3 case.  Re-verification of
    this class against actual BPS data for n > 3 is open work.
    """

    def __init__(self, n: int):
        if n < 3:
            raise ValueError(f"A1DnKAlg(n): n must be ≥ 3, got {n}")
        self._n = n
        self._B: list[list[int]] = _build_d_n_pairing(n)
        # v ∈ ker(B): canonical choice = e_{n-1} - e_{n-2} (matches D_3 SNF).
        self._v: Vec = tuple(
            (-1 if i == n - 2 else (1 if i == n - 1 else 0))
            for i in range(n)
        )
        self._R: SU2ZPlusRing = SU2ZPlusRing()

    # -- accessors --------------------------------------------------------

    @property
    def n(self) -> int:
        return self._n

    @property
    def rank(self) -> int:
        return self._n

    @property
    def gauge_rank(self) -> int:
        return self._n - 1

    @property
    def pairing(self) -> list[list[int]]:
        return [list(row) for row in self._B]

    @property
    def flavour_vector(self) -> Vec:
        return self._v

    @property
    def leaf_indices(self) -> tuple[int, int]:
        """The two identical-node indices that σ swaps."""
        return (self._n - 2, self._n - 1)

    def central_node(self) -> Label:
        """The σ-fixed singlet L_{e_{n-3}} (= the trivalent stem node).
        For n = 3 this is L_{g_1} = (1, 0, 0)."""
        return tuple(1 if i == self._n - 3 else 0 for i in range(self._n))

    def leaf_doublet(self) -> Label:
        """L_{e_{n-2}}, the spin-½ doublet F_{e_{n-2}} + F_{e_{n-1}}.
        For n = 3 this is (0, 1, 0)."""
        return tuple(1 if i == self._n - 2 else 0 for i in range(self._n))

    def chi(self, j: int) -> Label:
        """The spin-j chain charge (SU(2) character χ_{2j}; centred at origin).

        Lowest-weight rep = j·(e_{n-2} − e_{n-1}) (= positive flavour
        weight direction in our convention).  For n = 3, χ_j = (0, j, -j).
        """
        if j < 0:
            raise ValueError(f"chi(j): j must be ≥ 0, got {j}")
        return tuple(
            (j if i == self._n - 2 else (-j if i == self._n - 1 else 0))
            for i in range(self._n)
        )

    # -- canonicalisation / (a, b, s) parametrisation --------------------

    def canonicalise(self, gamma: Sequence[int]) -> Label:
        """Lowest-weight σ-orbit rep: canonical iff γ_{n-2} ≥ γ_{n-1}."""
        g = tuple(int(x) for x in gamma)
        if len(g) != self._n:
            raise ValueError(f"label has length {len(g)}, expected {self._n}")
        i_lo, i_hi = self._n - 2, self._n - 1
        if g[i_lo] >= g[i_hi]:
            return g
        # σ-image: swap coordinates i_lo, i_hi.
        out = list(g)
        out[i_lo], out[i_hi] = g[i_hi], g[i_lo]
        return tuple(out)

    def _to_abs(self, label: Label) -> tuple[Vec, int, int]:
        """γ (canonical) → ((a_0,…,a_{n-3}), 2b, 2s)."""
        gamma = self.canonicalise(label)
        a_coord = gamma[: self._n - 2]
        two_b = gamma[self._n - 2] + gamma[self._n - 1]
        two_s = gamma[self._n - 2] - gamma[self._n - 1]
        return a_coord, two_b, two_s

    def _from_abs(
        self, a_coord: Sequence[int], two_b: int, two_s: int,
    ) -> Label:
        if (two_b % 2) != (two_s % 2):
            raise ValueError(
                f"(a, 2b, 2s) = ({a_coord}, {two_b}, {two_s}): "
                f"parity mismatch (2b and 2s must have same parity)"
            )
        if two_s < 0:
            raise ValueError(
                f"(a, 2b, 2s) = ({a_coord}, {two_b}, {two_s}): "
                f"2s must be ≥ 0 for canonical"
            )
        gamma_lo = (two_b + two_s) // 2  # γ_{n-2}
        gamma_hi = (two_b - two_s) // 2  # γ_{n-1}
        return tuple(list(a_coord) + [gamma_lo, gamma_hi])

    # -- KAlgebra primitives ---------------------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self) -> Label:
        return tuple([0] * self._n)

    def multiply(self, a: Label, b: Label) -> Element:
        """Closed-form multiplication.

        Compute the q-twist as B(midpoint_a, midpoint_b) via the full
        D_n pairing matrix, where midpoint = (a_0, …, a_{n-3}, b, b)
        ∈ Γ ⊗ (1/2)·Z (so we work in 2× units to keep integers).

        Sum over the SO(3) Clebsch-Gordan range of s.
        """
        a_coord1, two_b1, two_s1 = self._to_abs(a)
        a_coord2, two_b2, two_s2 = self._to_abs(b)
        # q-twist = B(mid_1, mid_2) = (stem contributions) +
        #           2 (a_{1, n-3} · b_2 − b_1 · a_{2, n-3}).
        # In 2× units: 2*B(mid_1, mid_2) =
        #   Σ stem (2 a_{1,i} a_{2,i+1} − 2 a_{1,i+1} a_{2,i})
        # + (a_{1,n-3} · 2b_2 − 2b_1 · a_{2,n-3}) · 2.
        # But it's simpler to compute directly using the integer
        # midpoint coords scaled by 2 (= "two_mid" with entries 2a_i for
        # stem and 2b for both leaves):
        two_mid1 = list(2 * x for x in a_coord1) + [two_b1, two_b1]
        two_mid2 = list(2 * x for x in a_coord2) + [two_b2, two_b2]
        # B(two_mid_1, two_mid_2) = 4 · B(mid_1, mid_2); q-twist in
        # original units is B(mid_1, mid_2) = (1/4) · sum_{i,j} B[i][j]
        # · two_mid_1[i] · two_mid_2[j].
        four_q_twist = sum(
            self._B[i][j] * two_mid1[i] * two_mid2[j]
            for i in range(self._n) for j in range(self._n)
        )
        if four_q_twist % 4 != 0:
            raise AssertionError(
                f"q-twist not integer: 4·B(mid_1, mid_2) = {four_q_twist}, "
                f"expected divisible by 4 (mid coords are (1/2)·Z)"
            )
        q_twist = four_q_twist // 4
        # SO(3) CG range.
        two_s_lo = abs(two_s1 - two_s2)
        two_s_hi = two_s1 + two_s2
        out_a_coord = tuple(x + y for x, y in zip(a_coord1, a_coord2))
        out_two_b = two_b1 + two_b2
        out_terms: dict[Label, LaurentPoly] = {}
        for two_s in range(two_s_lo, two_s_hi + 1, 2):
            label = self._from_abs(out_a_coord, out_two_b, two_s)
            out_terms[label] = LaurentPoly({q_twist: 1})
        return Element(out_terms)

    def rho(self, a: Label) -> Label:
        """ρ(γ) = −γ (canonicalised)."""
        g = self.canonicalise(a)
        return self.canonicalise(tuple(-x for x in g))

    def rho_inverse(self, a: Label) -> Label:
        return self.rho(a)

    # ==================================================================
    # Two-layer trace (modeled on A1A2kKAlg's structure):
    #   Layer 1  uses cyclicity to reduce Tr(label) to an elementary
    #            trace T_j (or zero).
    #   Layer 2  plugs in the closed-form character formula for each T_j.

    def trace(self, a: Label, K: int = 20, **kwargs) -> RPowerSeries:
        """Two-layer ρ²-twisted trace.  Calls `trace_layer1(label)`
        then `trace_layer2(j, K)`."""
        # Rank-1 ker(B) requirement: trace via SO(3) characters needs
        # the flavour Cartan to be 1-dim.
        from snf_kernel import integer_kernel_and_section
        ker_basis, _ = integer_kernel_and_section(self._B)
        if len(ker_basis) != 1:
            raise ValueError(
                f"A1DnKAlg.trace currently requires rank(ker B) = 1; "
                f"got rank {len(ker_basis)} at n = {self._n}.  This "
                f"happens for n even (D_4, D_6, ...) which carry an "
                f"extra abelian U(1) flavour on top of SU(2); traces "
                f"there need a multi-Cartan realisation."
            )
        j = self.trace_layer1(a)
        if j is None:
            return RPowerSeries(self._R, {}, K)
        return self.trace_layer2(j, K)

    def trace_layer1(self, a: Label) -> int | None:
        """Cyclicity-based Layer 1: identify j such that label = chi_j,
        else return None.

        Cyclicity ρ²(L_b)·L_a = L_a·L_b (since ρ² = id on labels) combined
        with the closed-form multiplication forces Tr(L_γ) = 0 unless
        γ ∈ ker(B) = Z·v.  The non-vanishing labels are exactly the
        χ_j characters (γ = -j·v in our convention).
        """
        a_coord, two_b, two_s = self._to_abs(a)
        if any(x != 0 for x in a_coord) or two_b != 0:
            return None
        assert two_s % 2 == 0
        return two_s // 2

    def derive_T_j_at_q0(self, j: int) -> RElement:
        """Constraint-derived form of `[T_j]_{q^0}`: from orthonormality
        + cyclicity + algebra structure, `[T_j]_{q^0} = χ_{2j}` (= the
        SU(2) basis element of spin j, b = 2j).

        See `A1D3KAlg.derive_T_j_at_q0` for the full derivation; the
        argument generalises verbatim to [A_1, D_n] for n with rank-1
        ker(B).
        """
        if j < 0:
            raise ValueError(f"derive_T_j_at_q0: j must be ≥ 0, got {j}")
        return self._R.basis_element(2 * j)

    def trace_layer2(self, j: int, K: int = 20) -> RPowerSeries:
        """Closed-form Layer 2: the j-th elementary trace T_j.

        For the QT realisation:
            T_j = (q²; q²)_∞^{gauge_rank} · χ_j
        with `gauge_rank = n - 1` for D_n with rank-1 ker(B).

        For different realisations (e.g. BPS Schur index), substitute a
        different T_j character formula.
        """
        if j < 0:
            raise ValueError(f"trace_layer2: j must be ≥ 0, got {j}")
        from qpoch import qpoch_infty
        pref = qpoch_infty(K)
        for _ in range(self.gauge_rank - 1):
            pref = pref * qpoch_infty(K)
        chi_j = self._R.basis_element(2 * j)
        out_terms: dict[int, RElement] = {}
        for q_exp, q_coef in pref._c.items():
            r = chi_j * q_coef
            if not r.is_zero():
                out_terms[q_exp] = r
        return RPowerSeries(self._R, out_terms, K)

    def r_label_decompose(self, label: Label):
        """SU(2) flavour-lift coordinate (single irrep `χ_{two_s}`).  The
        section anchor differs by parity: even spread → the w-fixed midpoint
        `(a, b, b)`; odd spread → the spin-½ doublet anchor.  The single irrep
        key is the spread `two_s` (basis label `b = 2·spin`)."""
        a_coord, two_b, two_s = self._to_abs(label)
        if two_s % 2 == 0:
            b = two_b // 2
            section = tuple(list(a_coord) + [b, b])
        else:
            section = tuple(list(a_coord) + [(two_b + 1) // 2, (two_b - 1) // 2])
        return section, two_s

    def r_label_compose(self, section: Label, r_basis_label):
        # Re-spread the section anchor to the single irrep χ_{two_s}.
        a_coord = tuple(section[: self._n - 2])
        two_b = section[self._n - 2] + section[self._n - 1]
        return self._from_abs(a_coord, two_b, r_basis_label)


    # -- SO(3) → U(1) reduction ------------------------------------------

    def expand_to_quantum_torus(
        self, elt: Element,
    ) -> "tuple[KAlgebra, Element]":
        """L_γ ↦ Σ_k X_{γ + k·v} in `QuantumTorusKAlg(B_full)`."""
        from quantum_torus_kalgebra import QuantumTorusKAlg
        qt = QuantumTorusKAlg(pairing=self.pairing)
        out_terms: dict[Vec, LaurentPoly] = {}
        for label, lp in elt.terms.items():
            gamma = self.canonicalise(label)
            two_s = gamma[self._n - 2] - gamma[self._n - 1]
            for k in range(two_s + 1):
                g = tuple(gamma[i] + k * self._v[i] for i in range(self._n))
                if g in out_terms:
                    out_terms[g] = out_terms[g] + lp
                else:
                    out_terms[g] = lp
        out_terms = {g: c for g, c in out_terms.items() if not c.is_zero()}
        return qt, Element(out_terms)

    # -- convenience ------------------------------------------------------

    def L(self, label: Label) -> Element:
        return Element.basis(self.canonicalise(label))

    def __repr__(self) -> str:
        return (
            f"A1DnKAlg(n={self._n}, [A_1, D_{self._n}] AD, "
            f"D_{self._n}-chamber, manifest SU(2) flavour)"
        )
