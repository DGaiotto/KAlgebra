"""
hexagon_kalg.py
================

`HexagonKAlg` — ungauged hexagon K-algebra, μ-flavoured.  Wraps
`U1HexagonKAlg` with the auxiliary E generator promoted to a flavour
fugacity μ on the coefficient ring.

Canonical basis labels
----------------------
`(factors, μ_charge)` where
  - `factors`   = sorted tuple of `(a, i, exp)` triples, all drawn from
                  one of the 14 maximal q-commuting cones of `U1Hex`,
                  satisfying mag(factors) := Σ MU_LETTER_QPOWER[l]·exp_l = 0;
  - `μ_charge ∈ Z` = flavour grading (= U(1)Hex's `e_E`).

Section convention
------------------
`M_{F, μ_charge}  =  μ^{μ_charge} · M_{F, 0}`.  The section rep is
`(F, 0)`; the flavour shift is the singleton `(μ_charge,)`.

Multiplication
--------------
`HexagonKAlg.multiply((F_a, μ_a), (F_b, μ_b))` delegates to
`U1HexagonKAlg.multiply` (which works on full cone monomials including
non-zero μ_charge/e_E), preserves mag-charge by construction, and
returns the Z-form `dict {(F_out, μ_out): LaurentPoly}`.

Trace
-----
`HexagonKAlg.trace(label, K)` will use `U1HexagonKAlg.trace_layer1` plus
a Layer 2 evaluation of the two empirically-observed irreducible
families (`()` and `((2, 0, 1),)`) as M(2, 5)-minimal-model characters.
[Layer 2 implementation pending.]

Mult-generators
---------------
* `L_long(j)`  for `j ∈ {0, 1, 2}` — long diagonal, mag-zero (label
  `(((2, j, 1),), 0)`).
* `L_diam(i)`  for `i ∈ {0, 1, 2}` — "diameter pair" of non-intersecting
  short diagonals, mag-zero (label `(((1, i, 1), (1, i+3, 1)), 0)`).

6 generators total, symmetric under the hexagon's dihedral group.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element
from laurent_poly import LaurentPoly
from zplus_ring import AbelianZPlusRing, RPowerSeries, ZPlusRing
from u1_hexagon_kalg import U1HexagonKAlg, MU_LETTER_QPOWER, charge, MU_CHARGE


class HexagonKAlg(KAlgebra):
    """Ungauged hexagon K-algebra, μ-flavoured wrapper around U1HexagonKAlg."""

    def __init__(self) -> None:
        self._u1 = U1HexagonKAlg()
        self._R = AbelianZPlusRing(rank=1)

    # -- KAlgebra contract --

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return ((), 0)

    def r_label_decompose(self, label):
        # U(1)-flavoured: strip the central μ-charge — section = (factors, 0),
        # single irrep key = (mu_charge,).  (_label_section_decompose now
        # derives from this via the base forward bridge, returning a proper
        # RElement — the old hand-rolled version returned a bare key tuple,
        # which broke to_R_form.)
        factors, mu_charge = label
        return (factors, 0), (mu_charge,)

    def r_label_compose(self, section, r_basis_label):
        factors, _zero = section
        (mu_charge,) = r_basis_label
        return (factors, mu_charge)

    def rho(self, label):
        return self._u1.rho(label)

    def rho_inverse(self, label):
        return self._u1.rho_inverse(label)

    def multiply(self, a, b):
        """Hex multiply: delegate to U1Hex (which handles arbitrary `(factors, e_E)`
        cone-monomial inputs), enforce mag-zero on inputs/outputs.

        Returns an `Element` (Z-form) — wraps the dict from U1Hex.  Convert to
        R-form (section reps + μ-graded RLaurent coefs) via `to_R_form`.
        """
        self._assert_mag_zero(a, "first operand")
        self._assert_mag_zero(b, "second operand")
        out_u1 = self._u1.multiply(a, b)  # Element since U(1)Hex inherits KAlgebra now
        for label_out in out_u1.terms:
            self._assert_mag_zero(label_out, f"output term {label_out}")
        return out_u1

    def trace(self, label, K=20):
        """μ-flavoured (ungauged) trace — **BPS-free**.

        The ungauged hexagon is the centralizer of the gauge generator E=μ in
        `U1HexagonKAlg`; ungauging restores the U(1) vector-multiplet measure,
        so the flavoured index is the gauge-charge-graded sum of the gauged
        Wilson-line traces over the measure:

            Tr_ung(a)(z)  =  [ Σ_n z^n · Tr_gauged(a·μ^n) ] / (fq²;fq²)_∞² .

        Delegates to the general `UngaugedKAlgebra` ungauger; the gauged trace
        `U1HexagonKAlg.trace` is the standalone certified orthonormality
        bootstrap (now adaptive in gauge width, so it reaches the deep
        `Tr(L·μ^n)` the n-sum needs), so this whole path is BPS-free.  Equals
        `ungauge_kalgebra.ungauge_u1polygon(1).trace`."""
        if getattr(self, "_ung", None) is None:
            from ungauge_kalgebra import UngaugedKAlgebra
            self._ung = UngaugedKAlgebra(self._u1, ((), 1), epow=lambda lbl: lbl[1])
        return self._ung.trace(label, K)

    # -- Hex-specific accessors --

    def L_long(self, j):
        """Long diagonal L_{2, j}, j ∈ {0, 1, 2}.  Hex basis element (mag-zero)."""
        j = j % 3
        return (((2, j, 1),), 0)

    def L_diam(self, i):
        """Diameter-pair of short diagonals L_{1, i} · L_{1, i+3} for i ∈ {0, 1, 2}.
        Hex basis element (mag-zero: MU_QP[i] + MU_QP[i+3] = 0)."""
        i = i % 3
        return (((1, i, 1), (1, i + 3, 1)), 0)

    def mult_generators(self):
        """The 6 named mult-generators: 3 longs + 3 diameter-pair short products."""
        return [self.L_long(j) for j in range(3)] + [self.L_diam(i) for i in range(3)]

    # -- Internal helpers --

    @staticmethod
    def _mag(label):
        factors, _ = label
        return sum(MU_LETTER_QPOWER[(a, i)] * e for (a, i, e) in factors)

    def _assert_mag_zero(self, label, context=""):
        m = self._mag(label)
        if m != 0:
            raise ValueError(
                f"HexagonKAlg label {label} has mag-charge {m} ≠ 0 ({context}). "
                f"Hex basis is restricted to mag-zero cone monomials."
            )

    @classmethod
    def charge_of_label(cls, label):
        """Tropical charge of a Hex label (delegates to U(1)Hex's charge-of-label)."""
        factors, mu_charge = label
        ch = [mu_charge * c for c in MU_CHARGE]
        for (a, i, e) in factors:
            fc = charge((a, i))
            for k in range(4):
                ch[k] += e * fc[k]
        return tuple(ch)

    def __repr__(self) -> str:
        return "HexagonKAlg()"
