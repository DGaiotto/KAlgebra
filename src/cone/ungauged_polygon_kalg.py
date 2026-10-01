"""
ungauged_polygon_kalg.py
========================

`UngaugedPolygonKAlg(k)` — the μ-flavoured (ungauged) `[A_1, A_{2k+1}]`,
presented as the ungauging of the closed-form u(1)-gauged
`u1a1aodd_kalg.U1A1AoddKAlg(k)` on the `(2k+4)`-gon, at any `k ≥ 1`.  The
named classes `OctagonKAlg` (k = 2), `DecagonKAlg` (k = 3) and
`DodecagonKAlg` (k = 4) are thin subclasses; `HexagonKAlg` (k = 1) keeps its
own labels (`hexagon_kalg.py`).

Construction (no frozen data, no BPS, no bootstrap)
---------------------------------------------------
The ungauged algebra is the **centralizer of the gauge generator**
`E = ((), 1)` of `U1A1AoddKAlg(k)` (`ungauge_kalgebra.UngaugedKAlgebra`):
`multiply` is the gauged product (closed-form analytic peel) restricted to it,
`rho` the gauged `ρ`, and `E` becomes the U(1) flavour fugacity μ.  The trace
restores the U(1) vector-multiplet measure,

    Tr((F, e))(μ)  =  [ Σ_n μ^{−n} · Tr_gauged((F, e + n)) ] / (fq²;fq²)²_∞
                   =  μ^{e} · Tr((F, 0))(μ),

with the gauged traces the closed forms of `U1A1AoddKAlg(k)` (Layer-1
cyclicity + `u1_pgon_layer2.singlet_chord_trace`), so every label is traced to
any order.  This is `ungauge_kalgebra.ungauge_u1a1aodd(k).trace` read with
`μ ↦ μ⁻¹`: the orientation in which `E` is the fugacity `μ` itself, which is
what the section convention `M_{F, e} = μ^{e}·M_{F, 0}` of `r_label_decompose`
states (the trace is R-linear for that lift), and the orientation of the
finite zoo's `a3`/`a5`/`a7`.
(`ungauge_u1a1aodd(k)` itself grades `Tr((F, e)) = μ^{−e}·Tr((F, 0))`, and
its lift key is `(−e,)`; this class's lift is that one read through the same
`μ ↦ μ⁻¹`.)

Canonical basis labels
----------------------
`(F, e)` — the labels of `U1A1AoddKAlg(k)` in the centralizer: `F` a sorted
tuple of `(t, i, exp)`, letter `(t, i)` the diagonal `{i, i+t+1}` of the
`(2k+4)`-gon (`t = 1..k+1`; the diameter `t = k+1` has `i < k+2`), the
diagonals pairwise non-crossing and BALANCED — as many even–even diagonals as
odd–odd ones, counted with multiplicity (a letter's magnetic charge is set by
the parity of its endpoints) — and `e ∈ Z` the μ-charge.  `geometric_label`
returns `(curves, e)` with `curves` the multiset of diagonals: the geometric
labelling of this family.  `mult_generators()` are the mixed-parity diagonals and the
non-crossing (even–even, odd–odd) pairs — `6, 24, 65, 144` at `k = 1..4`.

**The labels changed on 2026-09-23.**  Until then the named classes wrapped
the stand-alone gauged polygons `U1OctagonKAlg` / `U1DecagonKAlg` /
`U1DodecagonKAlg` (retired), in those classes' letters.  The old
label `(F, e)` is the new label obtained letter by letter through the
same-diagonal dictionary, each old letter `(a, i)` becoming the new letter
below times `E^{s}` (so the new E-power is `±e + Σ exp·s`):

  * k = 2: `(a, i) ↦ (a, i)`, except the diameters `(3, i) ↦ (3, i − 4)` for
    `i = 4..7` with `s = +1, −1, +1, −1`; and `e ↦ −e`;
  * k = 3: `(a, i) ↦ (a, i)`, with `s = −2, +2, −2` on `(3, 7), (3, 8),
    (3, 9)`; and `e ↦ −e`;
  * k = 4: `(1, i) ↦ (1, i)`, `(2, i) ↦ (2, i + 9)`, `(3, i) ↦ (3, i + 9)`,
    `(4, i) ↦ (4, i + 7)` (mod 12), `(5, i) ↦ (5, i)`; no `s`; `e ↦ e`.

Each dictionary is certified on every ordered product of the old class's
multiplicative generators and on `ρ`, `ρ⁻¹` (676/676, 1369/1369, 3136/3136
products with `E^{±1}` included, against the retired classes as the
oracle).  Under it the μ-graded traces are
equal at k = 2, 3 and related by `μ ↦ μ⁻¹` at k = 4, where the retired class's
`E` was the same diagonal-geometric `E` as here.  The retired standalone
classes served products from frozen tables and chord seeds from
`u1aodd_trace_bootstrap.solve_intermediate`, which has known
certification holes.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

from kalgebra import KAlgebra
from zplus_ring import AbelianZPlusRing, ZPlusRing, restriction_hom


class UngaugedPolygonKAlg(KAlgebra):
    """The ungauged μ-flavoured `[A_1, A_{2k+1}]` — the centralizer of `E` in
    `U1A1AoddKAlg(k)`, with `E` the fugacity μ (see the module docstring)."""

    k: int = None

    def __init__(self, k: int = None) -> None:
        if k is None:
            k = type(self).k
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"{type(self).__name__}: k must be an integer "
                             f">= 1, got {k!r}")
        self.k = k
        from u1a1aodd_kalg import U1A1AoddKAlg
        from ungauge_kalgebra import UngaugedKAlgebra
        self._u1 = U1A1AoddKAlg(k)
        self._ung = UngaugedKAlgebra(self._u1, ((), 1), epow=lambda lbl: lbl[1])
        self._R = AbelianZPlusRing(rank=1)
        # μ ↦ μ⁻¹ on the trace coefficients: from the ungauger's grading
        # (E ≡ μ⁻¹) to this class's (E ≡ μ).
        self._orient = restriction_hom(self._R, self._R, [[-1]])

    # -- KAlgebra contract -------------------------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return ((), 0)

    def _oriented(self, key):
        """A key of the U(1) ring pushed through `μ ↦ μ⁻¹` (`self._orient`,
        the map the trace is read through; an involution)."""
        (k2, c), = self._orient.apply_RElement(self._R.basis_element(key)).terms.items()
        assert c == 1, (key, k2, c)
        return k2

    def r_label_decompose(self, label):
        """`(F, e) ↦ ((F, 0), (e,))`: `M_{F, e} = μ^{e}·M_{F, 0}`, the section
        convention the trace obeys (`Tr((F, e)) = μ^{e}·Tr((F, 0))`) — the
        ungauger's lift `((F, 0), (−e,))` (`UngaugedKAlgebra.r_label_decompose`)
        read through `μ ↦ μ⁻¹`, as the trace is."""
        section, key = self._ung.r_label_decompose(label)
        return section, self._oriented(key)

    def r_label_compose(self, section, r_basis_label):
        return self._ung.r_label_compose(section, self._oriented(r_basis_label))

    def rho(self, label):
        return self._u1.rho(label)

    def rho_inverse(self, label):
        return self._u1.rho_inverse(label)

    def multiply(self, a, b):
        """The product of `U1A1AoddKAlg(k)` (closed-form analytic peel),
        with both operands and every output term checked to lie in the
        centralizer of `E` (magnetic charge 0).  Returns an `Element` (Z-form;
        the μ-charge is the label's second slot)."""
        self._assert_mag_zero(a, "first operand")
        self._assert_mag_zero(b, "second operand")
        out = self._u1.multiply(a, b)
        for label_out in out.terms:
            self._assert_mag_zero(label_out, f"output term {label_out}")
        return out

    def trace(self, label, K=20):
        """μ-flavoured (ungauged) trace — closed forms only (module
        docstring): `ungauge_u1a1aodd(k).trace` with `μ ↦ μ⁻¹`, so that
        `Tr((F, e)) = μ^{e}·Tr((F, 0))`."""
        return self._orient.apply_RPowerSeries(self._ung.trace(label, K))

    # -- generators and geometry --------------------------------------------

    def mult_generators(self):
        """The multiplicative generators: the mixed-parity diagonals and the
        non-crossing (even–even, odd–odd) pairs of diagonals, as labels
        `(F, 0)` — `6, 24, 65, 144` at `k = 1..4`
        (`U1A1AoddKAlg._centralizer_generators`)."""
        return self._ung.mult_generators()

    def geometric_label(self, label):
        """`(curves, e)`: the multiset of diagonals of the `(2k+4)`-gon as
        sorted `((v1, v2), multiplicity)` pairs, and the μ-charge `e`
        (`UngaugedKAlgebra.geometric_label`; raises on a label that is not in
        the algebra)."""
        return self._ung.geometric_label(label)

    def _family_period(self, a):
        """Number of letters of type `a`: `2k+4`, or `k+2` for the diameter."""
        return self._u1.cone_data()._size[a]

    def physical_chord_types(self):
        """The chord types whose single letters are flavour-neutral (magnetic
        charge 0): the even types `t` — a diagonal `{i, i+t+1}` with `t` even
        joins an even and an odd vertex.  Odd types enter only in balanced
        products."""
        return sorted(a for a in self._u1.cone_data()._size if a % 2 == 0)

    def L_chord(self, a, i):
        """Single chord generator `L_{a,i}` as a label `(((a, i, 1),), 0)`
        (index mod the type's size).  In the algebra exactly for the physical
        (even) types; odd types are magnetic."""
        return (((a, i % self._family_period(a), 1),), 0)

    def L_long(self, i):
        """The physical type-2 chord `L_{2,i}` (the diagonal `{i, i+3}`)."""
        return self.L_chord(2, i)

    # -- internals ----------------------------------------------------------

    def _mag(self, label):
        """Magnetic charge of a label: `Σ exp·(letter magnetic charge)`, the
        letter's charge set by the parity of its diagonal's endpoints
        (`U1A1AoddConeData._letter_mag`: −2 both even, +2 both odd, 0 mixed).
        The E-power contributes nothing."""
        cd = self._u1.cone_data()
        letters = self.__dict__.get("_letters")
        if letters is None:
            letters = self.__dict__["_letters"] = frozenset(cd._chords)
        factors, _e = label
        total = 0
        for (t, i, m) in factors:
            if (t, i) not in letters:
                raise ValueError(f"{type(self).__name__}: {label!r} has an "
                                 f"unknown letter {(t, i)!r}")
            total += m * cd._letter_mag((t, i))
        return total

    def _assert_mag_zero(self, label, context=""):
        m = self._mag(label)
        if m != 0:
            raise ValueError(
                f"{type(self).__name__} label {label} has mag-charge {m} ≠ 0 "
                f"({context}).  The ungauged basis is the centralizer of the "
                f"gauge generator: balanced multisets of diagonals (as many "
                f"even–even as odd–odd).")

    def charge_of_label(self, label):
        """Charge of a label in the gauged chart of `U1A1AoddKAlg(k)`
        (`u1a1aodd_kalg.gauged_quiver_bps`): `e·μ_E + Σ exp·charge(letter)`,
        with `μ_E = (1, 0, 1, 0, …)` the charge of `E`."""
        cd = self._u1.cone_data()
        factors, e = label
        ch = [e * c for c in cd._MU]
        for (a, i, x) in factors:
            fc = cd._chg[(a, i)]
            for j in range(len(ch)):
                ch[j] += x * fc[j]
        return tuple(ch)

    def __repr__(self) -> str:
        if type(self) is UngaugedPolygonKAlg:
            return f"UngaugedPolygonKAlg({self.k})"
        return f"{type(self).__name__}()"
