"""`SU2BPSKAlgebra` -- SU(2)-flavoured BPS realisation of A_𝖖[T].

Standalone subclass of `KAlgebra`.
Wraps a `BPSKAlgebra` whose lattice carries a Poisson involution w
admitting a clean orthogonal splitting

    Γ  =  Γ^w  ⊕  Γ^{-w}                                          (*)

with Γ^{-w} of rank 1, equal to ker(B), and a BPS spec partitioned
into

    spec  =  (w-fixed singlets)  ⊔  (one w-doublet pair).

The w-doublet pair is the BPS realisation of the SU(2) fundamental
(half-integer-spin) sector.  This generalises "w acts diagonally" --
the splitting (*) is the frame-independent way to encode that w
admits a clean decomposition into ±1 eigenspaces over Z (equivalently:
every entry of `I + w` and of `I - w` is even, so the projectors
`(I±w)/2` are Z-linear maps Γ → Γ).

Worked example: [A_1, D_3] in the w-diagonal coordinate frame.

    pairing = [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]
    spec    = [(1, 0, 0), (0, 1, 1), (0, 1, -1)]
    w       = diag(1, 1, -1)

  - (1, 0, 0)            ∈ Γ^w        (= the central BPS node)
  - (0, 1, 1), (0, 1,-1) ∈ w-doublet  (= the SU(2) doublet leaves)
  - ker(B) = Z·(0, 0, 1) = Γ^{-w}     (rank 1)

This is the same physical theory as the old w-swap presentation
(node-swap w on [g_1, g_2, g_3]), in a coordinate frame where w is
diagonal.  Coordinates: (γ_0, γ_1, γ_2) where γ_2 = the w-anti-fixed
(= SU(2) Cartan) component.  F_{γ} for γ ∈ Z^3 is well-defined in
BPSKAlgebra for the full ambient lattice (including w-anti-fixed
half-integer points like (0, -1, 0) which are NOT in the spec
sublattice {γ_1 + γ_2 even}).  These half-points are the natural
location of the SU(2) χ_1 doublet character; e.g. F_{(0,-1,0)} is
the (currently conjectured) good multiplicative generator of the
half-integer-spin sector.

Canonical Z-form labels
-----------------------
γ ∈ Γ / w represented by lowest-weight w-orbit reps (flavour_diff
≤ 0).  Basis element:

    Ł_γ  =  Σ_{j=0}^{n-1}  F_{γ + j·v},       n = chain length,

where v is the primitive generator of ker(B) = Γ^{-w}.

R-form decomposition
--------------------
Uniform: every canonical Ł_γ maps to (section_rep, χ_k SU(2)) where
section_rep is the w-fixed projection (γ + wγ)/2 ∈ Γ^w and k is the
chain spread (= -flavour_diff = -(γ-wγ_in_v_units), an integer ≥ 0).
No T_j(χ_·) polynomial special case for half-integer-spin labels as
in SO(3) -- SU2ZPlusRing carries half-integer-spin basis elements
directly (χ_1 = dim 2 fundamental).

Trace
-----
Lifts to BPS, sums the BPS traces (which produce μ-Laurent
polynomials over `AbelianZPlusRing(rank=1)` on the w-anti-fixed
direction), and converts to SU(2) characters directly via the
step-2 SU(2) recursion `μ^n + μ^{-n} = χ_n - χ_{n-2}`.

Z_3-symmetric presentation (worked example: [A_1, D_3])
-------------------------------------------------------
The 6 w-fixed canonical generators decompose into 2 ρ-orbits of
length 3 each.  With T_0 = L_{(1, 0, 0)} (central BPS node) and
D_0 = L_{(0, -1, 0)} (the w-fixed leaf at half the gauge step),

    T_i  =  ρ^i(T_0),       D_i  =  ρ^i(D_0),       i ∈ {0, 1, 2},
    ρ:  T_i ↦ T_{i+1 mod 3},   D_i ↦ D_{i+1 mod 3},   χ_k fixed.

T-orbit lattice positions: (1, 0, 0), (-1, -2, 0), (-1, 0, 0).
D-orbit lattice positions: (0, -1, 0), (-1, -1, 0), (0,  1, 0).

Defining relations (all indices mod 3; each line a single Z_3-orbit):

    T_i · T_{i+1}    =  1  +  q^{-1}·χ_1·D_i  +  q^{-2}·D_i²
    T_{i+1} · T_i    =  1  +  q     ·χ_1·D_i  +  q^{2} ·D_i²
    D_i · D_{i+1}    =  1  +  q^{-1}·T_{i+1}
    D_{i+1} · D_i    =  1  +  q     ·T_{i+1}
    T_i · D_{i+1}    =  χ_1  +  q^{-1}·D_i  +  q·D_{i-1}
    D_{i+1} · T_i    =  χ_1  +  q     ·D_i  +  q^{-1}·D_{i-1}

Plus clean q-Laurent products for all other generator pairs (T_i²,
D_i² are clean lattice doublings; T_i·D_j for j ∉ {i+1} is a clean
q^{B(T_i,D_j)}·L_{T_i+D_j} product).

ρ(T_0) and ρ(D_0) are derivable polynomials in the 4-element
subset {T_0, T_2, D_0, D_2} ⊂ generators -- so this presentation is
not minimal in the "fewest generators" sense, but it IS minimal in
the "fewest ρ-orbits" sense (one for the T-family, one for the
D-family) and the Z_3 symmetry is manifest in the relations.

χ_k structure is carried by R-coefficients, not generators:
χ_1 = L_{(0, 0, -1)} (= the SU(2) doublet at the identity gauge
cell), and e.g. L_{(0, 1, -1)} = χ_1 · D_2 (= the doublet at the
D_2 gauge cell).
"""

from __future__ import annotations

from typing import Sequence

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element
from bps_kalgebra import BPSKAlgebra
from zplus_ring import (
    ZPlusRing,
    RElement,
    RPowerSeries,
    AbelianZPlusRing,
    SU2ZPlusRing,
)
from laurent_poly import LaurentPoly


Vec = tuple[int, ...]
Label = Vec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matvec(M: Sequence[Sequence[int]], v: Sequence[int]) -> Vec:
    return tuple(sum(int(M[i][j]) * int(v[j]) for j in range(len(v)))
                 for i in range(len(M)))


def _u1_weyl_to_su2(
    u1_relt: RElement, target_ring: SU2ZPlusRing,
) -> RElement:
    """Convert a Weyl-symmetric μ-Laurent (over `AbelianZPlusRing(rank=1)`)
    to the SU(2) χ-basis directly, using the step-2 SU(2) recursion

        μ^n + μ^{-n}  =  χ_n SU(2)  -  χ_{n-2} SU(2),       n ≥ 1,
        μ^0           =  χ_0 SU(2).

    Valid when the BPS lattice μ-coord equals the SU(2) weight directly
    (= the m=2 SU2BPSKAlgebra setup, where v_chain = 2·e_neg).  Inputs
    must be w-symmetric: coefficients of μ^k and μ^{-k} agree for k ≠ 0.
    """
    if not isinstance(u1_relt.ring, AbelianZPlusRing) or u1_relt.ring.rank != 1:
        raise TypeError(
            "expected RElement over AbelianZPlusRing(rank=1)"
        )
    # Collect coefficients indexed by lattice μ-coord.
    coeffs: dict[int, int] = {}
    for basis, c in u1_relt.terms.items():
        k = basis[0] if isinstance(basis, tuple) and basis else 0
        coeffs[k] = coeffs.get(k, 0) + c
    coeffs = {k: c for k, c in coeffs.items() if c != 0}
    # w-symmetry.
    for k in list(coeffs.keys()):
        if k != 0 and coeffs.get(k, 0) != coeffs.get(-k, 0):
            raise ValueError(
                f"input not w-symmetric at μ^{k}: "
                f"{coeffs.get(k, 0)} vs μ^{-k}: {coeffs.get(-k, 0)}"
            )
    # Direct SU(2) step-2 conversion.
    chi_coeffs: dict[int, int] = {}
    if 0 in coeffs:
        chi_coeffs[0] = coeffs[0]
    for k in sorted(p for p in coeffs if p > 0):
        cnt = coeffs[k]
        chi_coeffs[k] = chi_coeffs.get(k, 0) + cnt
        if k - 2 >= 0:
            chi_coeffs[k - 2] = chi_coeffs.get(k - 2, 0) - cnt
    chi_coeffs = {j: c for j, c in chi_coeffs.items() if c != 0}
    return RElement(target_ring, chi_coeffs)


# ---------------------------------------------------------------------------
# SU2BPSKAlgebra
# ---------------------------------------------------------------------------


class SU2BPSKAlgebra(KAlgebra):
    """SU(2)-flavoured BPS realisation; wraps `BPSKAlgebra` with a
    w-orbit-chain canonical basis on a Γ = Γ^w ⊕ Γ^{-w}-splittable
    lattice."""

    def __init__(
        self,
        pairing: Sequence[Sequence[int]],
        node_charges: Sequence[Sequence[int]],
        w_su2: Sequence[Sequence[int]],
        **bps_kwargs,
    ):
        # Build the underlying BPSKAlgebra; validates pairing, extracts
        # ker(B), section, etc.
        self._bps = BPSKAlgebra(
            pairing=pairing,
            node_charges=node_charges,
            **bps_kwargs,
        )
        n = self._bps.lattice.rank
        self._rank = n
        if hasattr(self._bps, 'pairing'):
            try:
                self._B = [[int(x) for x in row] for row in self._bps.pairing]
            except Exception:
                self._B = [[int(x) for x in row] for row in pairing]
        else:
            self._B = [[int(x) for x in row] for row in pairing]

        # Validate w_su2 shape.
        if len(w_su2) != n or any(len(row) != n for row in w_su2):
            raise ValueError(f"w_su2 must be a {n}x{n} matrix")
        self._w_su2: list[list[int]] = [
            [int(x) for x in row] for row in w_su2
        ]

        # w² = id.
        sig_sq = [
            [sum(self._w_su2[i][k] * self._w_su2[k][j] for k in range(n))
             for j in range(n)]
            for i in range(n)
        ]
        if any(sig_sq[i][j] != (1 if i == j else 0)
               for i in range(n) for j in range(n)):
            raise ValueError(f"w_su2 must square to identity; got w²={sig_sq}")

        # wᵀ B w = B.
        sBs = [
            [sum(self._w_su2[k][i] * self._B[k][l] * self._w_su2[l][j]
                 for k in range(n) for l in range(n))
             for j in range(n)]
            for i in range(n)
        ]
        if sBs != self._B:
            raise ValueError(
                "w_su2 must be Poisson: wᵀ B w = B; "
                f"got wᵀ B w = {sBs}, B = {self._B}"
            )

        # Γ = Γ^w ⊕ Γ^{-w}: equivalent to (I+w)/2 and (I-w)/2 being
        # Z-linear, i.e., every entry of I+w (equivalently I-w) is even.
        for i in range(n):
            for j in range(n):
                diag = 1 if i == j else 0
                if (diag + self._w_su2[i][j]) % 2 != 0:
                    raise ValueError(
                        f"w_su2 must split Γ = Γ^w ⊕ Γ^{{-w}}: "
                        f"(I+w)[{i}][{j}]={diag + self._w_su2[i][j]} is odd; "
                        f"projectors (I±w)/2 are not Z-linear.\n"
                        f"Got w = {self._w_su2}"
                    )

        # ker(B) rank-1, equal to Γ^{-w} (= "pure flavour" w-anti-fixed).
        if self._bps._flavour_rank != 1:
            raise ValueError(
                f"SU2BPSKAlgebra currently requires rank(ker B) == 1; "
                f"got rank {self._bps._flavour_rank}"
            )
        e_neg: Vec = tuple(int(x) for x in self._bps._ker_basis[0])

        # w acts as -id on ker(B).
        w_su2_e_neg = _matvec(self._w_su2, e_neg)
        if w_su2_e_neg != tuple(-x for x in e_neg):
            raise ValueError(
                f"w_su2 must act as -id on ker(B); got w(e_neg) = "
                f"{w_su2_e_neg}, -e_neg = {tuple(-x for x in e_neg)}"
            )

        # rank(Γ^{-w}) = 1  (= "Γ^{-w} is pure flavour"; combined with
        # ker(B) ⊆ Γ^{-w} and rank(ker B) = 1 gives Γ^{-w} = ker(B)).
        tr_w_su2 = sum(self._w_su2[i][i] for i in range(n))
        if (n - tr_w_su2) % 2 != 0:
            raise ValueError(
                f"w_su2 not a well-defined involution: trace(w)={tr_w_su2}, "
                f"n={n}, n - tr(w) = {n - tr_w_su2} is odd"
            )
        rank_neg = (n - tr_w_su2) // 2
        if rank_neg != 1:
            raise ValueError(
                f"SU2BPSKAlgebra requires rank(Γ^{{-w}}) = 1 (= pure flavour, "
                f"= ker(B)); got rank(Γ^{{-w}}) = {rank_neg} (n={n}, "
                f"trace(w)={tr_w_su2})"
            )

        # Spec partition: every node either w-fixed or in w-doublet pair;
        # exactly one w-doublet pair (= the SU(2) doublet).
        spec_tuples = [tuple(int(x) for x in g) for g in self._bps.spec]
        fixed_nodes = []
        doublet_pairs: list[tuple[Vec, Vec]] = []
        seen: set[Vec] = set()
        for g in spec_tuples:
            if g in seen:
                continue
            sg = self.w_su2_apply(g)
            if sg == g:
                fixed_nodes.append(g)
                seen.add(g)
            else:
                if sg not in spec_tuples:
                    raise ValueError(
                        f"w_su2 does not permute the spec: w({g}) = {sg} "
                        f"is not in spec"
                    )
                doublet_pairs.append((g, sg))
                seen.add(g)
                seen.add(sg)
        if len(doublet_pairs) != 1:
            raise ValueError(
                "SU2BPSKAlgebra requires exactly one w-doublet pair in the "
                f"spec (the SU(2) doublet); got {len(doublet_pairs)} pairs.  "
                f"spec = {spec_tuples}, w = {self._w_su2}"
            )
        self._fixed_nodes: list[Vec] = fixed_nodes
        self._doublet: tuple[Vec, Vec] = doublet_pairs[0]

        # Chain step v = doublet displacement.  This is NOT the ker(B)
        # primitive e_neg -- the doublet sits at displacement m·e_neg with
        # m ≥ 1, and the SU(2) chi structure uses the doublet step (so
        # the w-doublet pair forms a length-2 chain = χ_1 fundamental,
        # regardless of how the displacement is normalised in lattice
        # units).  With Γ = Γ^w ⊕ Γ^{-w}, the w-doublet displacement
        # lies in 2·Γ^{-w} = 2·Z·e_neg, so m is even (≥ 2) in this
        # setup; for the [A_1, D_3] worked example m = 2.
        g_d_a, g_d_b = self._doublet
        displacement = tuple(g_d_a[i] - g_d_b[i] for i in range(n))
        # Normalise sign: v points in same direction as e_neg.
        sign = 1
        for i in range(n):
            if e_neg[i] != 0 and displacement[i] != 0:
                sign = 1 if (displacement[i] > 0) == (e_neg[i] > 0) else -1
                break
        v_chain = tuple(sign * x for x in displacement)
        # v_chain = m · e_neg for some positive integer m.
        m_factors = []
        for i in range(n):
            if e_neg[i] != 0:
                if v_chain[i] % e_neg[i] != 0:
                    raise ValueError(
                        f"doublet displacement {displacement} is not an "
                        f"integer multiple of ker(B) primitive {e_neg}"
                    )
                m_factors.append(v_chain[i] // e_neg[i])
            elif v_chain[i] != 0:
                raise ValueError(
                    f"doublet displacement {displacement} has nonzero coord "
                    f"where ker(B) primitive {e_neg} is zero"
                )
        if not m_factors or any(f != m_factors[0] for f in m_factors):
            raise ValueError(
                f"doublet displacement {displacement} inconsistent with "
                f"ker(B) primitive {e_neg}"
            )
        m = m_factors[0]
        if m < 1:
            raise ValueError(
                f"doublet displacement scale m = {m} must be positive"
            )
        self._m: int = m
        self._v: Vec = v_chain  # = m · e_neg, the chain step

        self._gauge_rank = self._bps._gauge_rank
        self._R: SU2ZPlusRing = SU2ZPlusRing()

    # -- accessors --------------------------------------------------------

    @property
    def bps(self) -> BPSKAlgebra:
        """Underlying BPS realisation."""
        return self._bps

    @property
    def w_su2(self) -> list[list[int]]:
        return [list(row) for row in self._w_su2]

    @property
    def flavour_vector(self) -> Vec:
        return self._v

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def gauge_rank(self) -> int:
        return self._gauge_rank

    @property
    def spec(self) -> list[Vec]:
        return [tuple(int(x) for x in g) for g in self._bps.spec]

    @property
    def fixed_nodes(self) -> list[Vec]:
        """BPS spec nodes that lie in Γ^w (w-fixed singlets)."""
        return [tuple(g) for g in self._fixed_nodes]

    @property
    def doublet(self) -> tuple[Vec, Vec]:
        """The w-doublet pair of spec nodes (= SU(2) fundamental sector)."""
        return tuple(self._doublet[0]), tuple(self._doublet[1])

    # -- w-orbit / label utilities ---------------------------------------

    def w_su2_apply(self, gamma: Sequence[int]) -> Vec:
        return _matvec(self._w_su2, gamma)

    def _flavour_diff(self, gamma: Sequence[int]) -> int:
        """k such that γ - wγ = k·v."""
        gamma_t = tuple(int(x) for x in gamma)
        diff = tuple(gamma_t[i] - self.w_su2_apply(gamma_t)[i]
                     for i in range(self._rank))
        k: int | None = None
        for i in range(self._rank):
            if self._v[i] != 0:
                if diff[i] % self._v[i] != 0:
                    raise ValueError(
                        f"γ - wγ = {diff} is not an integer multiple of "
                        f"v = {self._v}"
                    )
                cand = diff[i] // self._v[i]
                if k is None:
                    k = cand
                elif k != cand:
                    raise ValueError(
                        f"γ - wγ = {diff} inconsistent across coords of v"
                    )
            else:
                if diff[i] != 0:
                    raise ValueError(
                        f"γ - wγ = {diff} has non-zero coord where v has zero"
                    )
        if k is None:
            raise ValueError("flavour vector v is zero")
        return k

    def canonicalise(self, gamma: Sequence[int]) -> Label:
        """Lowest-weight w-orbit rep (flavour_diff ≤ 0)."""
        gamma_t = tuple(int(x) for x in gamma)
        k = self._flavour_diff(gamma_t)
        if k <= 0:
            return gamma_t
        return self.w_su2_apply(gamma_t)

    def _chain_length(self, label: Label) -> int:
        k = self._flavour_diff(label)
        return -k + 1

    def _chain_points(self, label: Label) -> list[Vec]:
        """The γ ∈ Γ at which `Ł_label` has F-support: γ, γ+v, ..., wγ."""
        label = tuple(int(x) for x in label)
        steps = self._chain_length(label)
        return [
            tuple(label[i] + k * self._v[i] for i in range(self._rank))
            for k in range(steps)
        ]

    def _w_su2_fixed_projection(self, gamma: Sequence[int]) -> Vec:
        """(γ + wγ)/2 ∈ Γ^w -- Z-linear because Γ = Γ^w ⊕ Γ^{-w}."""
        gamma_t = tuple(int(x) for x in gamma)
        sg = self.w_su2_apply(gamma_t)
        return tuple((gamma_t[i] + sg[i]) // 2 for i in range(self._rank))

    # -- the seven KAlgebra primitives -----------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self) -> Label:
        return tuple([0] * self._rank)

    def rho(self, a: Label) -> Label:
        """ρ on labels descends from BPS ρ; w commutes with BPS ρ."""
        return self.canonicalise(self._bps.rho(a))

    def rho_inverse(self, a: Label) -> Label:
        return self.canonicalise(self._bps.rho_inverse(a))

    def multiply(self, a: Label, b: Label) -> Element:
        """Compute Ł_a · Ł_b in the BPS F-basis, then peel back into Ł's.

        Strategy: lift each Ł to its F-chain, call `_bps.multiply`
        pairwise on every cross-product, then re-decompose the
        w-invariant F-sum into the Ł-basis via outermost-first
        triangular elimination.
        """
        a = self.canonicalise(a)
        b = self.canonicalise(b)
        bps_product: dict[Vec, dict[int, int]] = {}
        for ga in self._chain_points(a):
            for gb in self._chain_points(b):
                cross = self._bps.multiply(ga, gb)
                for f_label, lp in cross.terms.items():
                    f_label_t = tuple(int(x) for x in f_label)
                    slot = bps_product.setdefault(f_label_t, {})
                    for e, c in lp._coeffs.items():
                        slot[e] = slot.get(e, 0) + c
        bps_product = {
            g: {e: c for e, c in d.items() if c != 0}
            for g, d in bps_product.items() if any(c != 0 for c in d.values())
        }
        return self._decompose_F_into_Ł(bps_product)

    def _decompose_F_into_Ł(
        self, bps_terms: dict[Vec, dict[int, int]],
    ) -> Element:
        """Decompose a w-invariant F-basis sum into the Ł-basis via
        outermost-first chain peeling."""
        remaining: dict[Vec, dict[int, int]] = {
            g: dict(d) for g, d in bps_terms.items()
        }
        out_terms: dict[Label, dict[int, int]] = {}

        while remaining:
            best_g = None
            best_spread = -1
            for g in remaining:
                spread = abs(self._flavour_diff(g))
                if spread > best_spread:
                    best_spread = spread
                    best_g = g
            assert best_g is not None
            label = self.canonicalise(best_g)
            chain = self._chain_points(label)
            coef = dict(remaining[best_g])
            for g in chain:
                if g not in remaining:
                    raise ValueError(
                        f"w-invariance / spread peeling failure at γ = {g}; "
                        f"expected chain {chain} but {g} absent."
                    )
                slot = remaining[g]
                for e, c in coef.items():
                    new = slot.get(e, 0) - c
                    if new == 0:
                        slot.pop(e, None)
                    else:
                        slot[e] = new
                if not slot:
                    del remaining[g]
            if label not in out_terms:
                out_terms[label] = {}
            for e, c in coef.items():
                out_terms[label][e] = out_terms[label].get(e, 0) + c
            out_terms[label] = {e: c for e, c in out_terms[label].items() if c != 0}
            if not out_terms[label]:
                del out_terms[label]

        return Element({
            label: LaurentPoly(coefs)
            for label, coefs in out_terms.items()
        })

    def r_label_decompose(self, label):
        """Uniform SU(2) flavour-**folding** lift coordinate: section =
        `(γ + wγ)/2 ∈ Γ^w` (the w-fixed projection), single irrep key k =
        `-flavour_diff(γ) ≥ 0` (the chain spread).  The w-projection loses the
        spread's sign (γ and wγ share a coordinate), so `r_label_compose` raises
        (Q1B pattern)."""
        a = self.canonicalise(label)
        k = -self._flavour_diff(a)  # ≥ 0
        section_rep = self._w_su2_fixed_projection(a)
        return section_rep, k

    def r_label_compose(self, section, r_basis_label):
        raise NotImplementedError(
            "SU2BPSKAlgebra.r_label_compose: the SU(2) w-fixed projection is a "
            "non-invertible fold (γ and wγ share the section) — no single "
            "inverse label."
        )

    def trace(self, a: Label, K: int = 20, **kwargs) -> RPowerSeries:
        """Trace via BPS, then convert μ-monomials to SU(2) characters."""
        a = self.canonicalise(a)
        from collections import defaultdict
        mu_coeffs: dict[int, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for gamma in self._chain_points(a):
            tr = self._bps.trace(gamma, K=K, **kwargs)
            for q_exp, r_elt in tr.coeffs.items():
                for mu_basis, c in r_elt.terms.items():
                    mu_power = (mu_basis[0]
                                 if isinstance(mu_basis, tuple) and mu_basis
                                 else 0)
                    mu_coeffs[q_exp][mu_power] += c
        out_terms: dict[int, RElement] = {}
        for q_exp, mu_at_q in mu_coeffs.items():
            abelian_ring = AbelianZPlusRing(rank=1)
            u1_relt = RElement(
                abelian_ring,
                {(k,): c for k, c in mu_at_q.items() if c != 0},
            )
            su2_relt = _u1_weyl_to_su2(u1_relt, self._R)
            if not su2_relt.is_zero():
                out_terms[q_exp] = su2_relt
        return RPowerSeries(self._R, out_terms, K)

    # -- convenience ------------------------------------------------------

    def L(self, label: Label) -> Element:
        """The canonical basis element Ł_a as an `Element`."""
        return Element.basis(self.canonicalise(label))

    # -- Conversion to / from the underlying U(1) BPS algebra -----------

    def to_u1(self, elt: Element) -> "tuple[BPSKAlgebra, Element]":
        """Map an SU(2)-flavoured `Element` (in Ł-basis) to the
        underlying U(1)-flavoured BPSKAlgebra `Element` (in F-basis).
        For each c_γ · Ł_γ, expand via the w-orbit chain."""
        out_terms: dict[Vec, LaurentPoly] = {}
        for label, lp in elt.terms.items():
            gamma = self.canonicalise(label)
            for g in self._chain_points(gamma):
                if g in out_terms:
                    out_terms[g] = out_terms[g] + lp
                else:
                    out_terms[g] = lp
        out_terms = {g: c for g, c in out_terms.items() if not c.is_zero()}
        return self._bps, Element(out_terms)

    def from_u1(self, bps_elt: Element) -> Element:
        """Map a w-invariant U(1) BPS `Element` (in F-basis) to the
        SU(2) Ł-basis via triangular w-orbit decomposition."""
        bps_terms: dict[Vec, dict[int, int]] = {}
        for label, lp in bps_elt.terms.items():
            coefs = {e: c for e, c in lp._coeffs.items() if c != 0}
            if coefs:
                bps_terms[tuple(int(x) for x in label)] = coefs
        return self._decompose_F_into_Ł(bps_terms)

    def __repr__(self) -> str:
        return (
            f"SU2BPSKAlgebra(rank={self._rank}, "
            f"|spec|={len(self._bps.spec)}, gauge_rank={self._gauge_rank})"
        )
