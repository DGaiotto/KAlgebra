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
`M_{F, μ_charge}  =  μ^{−μ_charge} · M_{F, 0}`.  The section rep is
`(F, 0)`; the flavour key is the singleton `(−μ_charge,)` — the rule of
`ungauge_kalgebra.UngaugedKAlgebra.r_label_decompose`, which this class
delegates to, and the convention its trace obeys (below).  Until 2026-09-23 the
key was `(μ_charge,)`, which the trace contradicts;
the source repository's test `test_section_decompose` pinned it and now pins
`(−μ_charge,)` with R-linearity of the trace and the old key as negative
control.

Multiplication
--------------
`HexagonKAlg.multiply((F_a, μ_a), (F_b, μ_b))` delegates to
`U1HexagonKAlg.multiply` (which works on full cone monomials including
non-zero μ_charge/e_E), preserves mag-charge by construction, and
returns the Z-form `dict {(F_out, μ_out): LaurentPoly}`.  Since
2026-09-23 `U1HexagonKAlg` serves its products, `ρ` and traces through the
closed-form `U1A1AoddKAlg(1)` under `(F, e) ↦ (F, −e)` (see
`u1_hexagon_kalg`), so no frozen table and no bootstrap is on this class's
serving path either.

Trace
-----
`HexagonKAlg.trace(label, K)` is the ungauging of `U1HexagonKAlg`'s trace
(the measure-restored gauge-charge sum of `ungauge_kalgebra.
UngaugedKAlgebra`), closed forms throughout.  In this class's labels
`Tr((F, e)) = μ^{−e}·Tr((F, 0))`; since this class's `E` is
`U1A1AoddKAlg(1)`'s `E⁻¹`, that is the μ-orientation of the named
k ≥ 2 classes: `HexagonKAlg((F, e))` has the trace of
`UngaugedPolygonKAlg(1)((F, −e))` (the suite in the source repository).
The lift agrees: key `(−e,)` (section convention above).

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

    def _ungauged(self):
        """The ungauger over `U1HexagonKAlg` (centralizer of `E = ((), 1)`),
        which serves `trace` and the lift coordinate.  Built on first use."""
        if getattr(self, "_ung", None) is None:
            from ungauge_kalgebra import UngaugedKAlgebra
            self._ung = UngaugedKAlgebra(self._u1, ((), 1), epow=lambda lbl: lbl[1])
        return self._ung

    def r_label_decompose(self, label):
        """`(F, e) ↦ ((F, 0), (−e,))`: `M_{F, e} = μ^{−e}·M_{F, 0}`, the
        section convention the trace obeys (`Tr((F, e)) = μ^{−e}·Tr((F, 0))`).
        The rule of `UngaugedKAlgebra.r_label_decompose`, delegated."""
        return self._ungauged().r_label_decompose(label)

    def r_label_compose(self, section, r_basis_label):
        return self._ungauged().r_label_compose(section, r_basis_label)

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

        Delegates to the general `UngaugedKAlgebra` ungauger over
        `U1HexagonKAlg`, whose trace is served by the closed-form
        `U1A1AoddKAlg(1)` (since 2026-09-23; before, its chord seeds were
        orthonormality-bootstrap solves), so this whole path is BPS-free and
        bootstrap-free."""
        return self._ungauged().trace(label, K)

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

    def geometric_label(self, label):
        """The label `(F, e)` as `(curves, e)`: `curves` the multiset of
        pairwise non-crossing diagonals of the hexagon, as many even–even as
        odd–odd with multiplicity — each letter's diagonal is
        `U1HexagonKAlg.geometric_label` — and `e` unchanged.  Read from
        `ungauge_u1a1aodd(1).geometric_label((F, −e))`: the letters are the
        same diagonals there and this class's `E` is `U1A1AoddKAlg(1)`'s
        `E⁻¹` (the dictionary the source repository's test
        `test_hexagon_shares_the_orientation` certifies).  Not through this
        class's own ungauger, which checks q-commutation on
        `U1HexagonKAlg.cone_data()`, the frozen cross-check surface."""
        if getattr(self, "_geo", None) is None:
            from ungauge_kalgebra import ungauge_u1a1aodd
            self._geo = ungauge_u1a1aodd(1)
        factors, e = label
        curves, _e = self._geo.geometric_label((factors, -e))
        return (curves, e)

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
