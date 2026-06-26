"""
ungauged_polygon_kalg.py
========================

`UngaugedPolygonKAlg` — shared base for the named ungauged μ-flavoured
even-polygon K-algebras `OctagonKAlg` / `DecagonKAlg` / `DodecagonKAlg`
(k = 2, 3, 4), the k ≥ 2 siblings of the self-contained `HexagonKAlg` (k = 1).

Each is a μ-flavoured wrapper around its standalone gauged polygon
`U1{Polygon}KAlg`.  The ungauged algebra is the **centralizer of the gauge
generator** E (the magnetically-neutral cone monomials, `mag = 0`); ungauging
promotes E to a U(1) flavour fugacity μ and restores the U(1)
**vector-multiplet measure** in the trace,

    Tr_ung(a)(z)  =  [ Σ_n z^n · Tr_gauged(a·μ^n) ] / (fq²;fq²)_∞² ,

delegated to the general `ungauge_kalgebra.UngaugedKAlgebra`.  Because the
gauged polygons are standalone (frozen charges + certified-bootstrap trace —
no BPS / RG-oracle at runtime), the whole path is **BPS-free**, and equals
`ungauge_kalgebra.ungauge_u1polygon(k).trace` term-for-term.

A concrete subclass declares three class attributes:
  * ``k``             — 1..4 (the (2k+4)-gon);
  * ``_GAUGED_CLASS`` — the standalone gauged polygon class (e.g.
                        `U1OctagonKAlg`), which must supply `_mag_charge`,
                        `multiply`, `rho`, `rho_inverse`, `trace`;
  * ``_CHARGES``      — its `u1a1aodd_k{k}_chord_charges` module (frozen:
                        `chord_charge`, `E_CHARGE`, `PRIMITIVE_FAMILY_PERIOD`).

Canonical basis labels
----------------------
`(factors, μ_charge)` — `factors` a sorted tuple of `(a, i, exp)` chord
triples drawn from one q-commuting cone with `mag(factors) = 0`, and
`μ_charge ∈ Z` the flavour grading (= the gauged polygon's `e_E`).  Section
convention: `M_{F, μ_charge} = μ^{μ_charge}·M_{F, 0}`, so the lift coordinate
strips the central μ-charge (section `(F, 0)`, single flavour irrep
`(μ_charge,)`).

`HexagonKAlg` (k=1) predates this base and stays self-contained (it carries
its own `L_long` / `L_diam` hexagon geometry); behaviourally it is the same
wrapper, and `HexagonKAlg().trace == ungauge_u1polygon(1).trace`.

Geometry accessors (uniform, k-general)
---------------------------------------
* `physical_chord_types()` — the flavour-neutral (mag-zero) chord families:
  the **even** types {2, 4, ...} (odd types {1, 3, ...} are magnetic and only
  enter in mag-zero products); a diameter (type k+1) is physical iff k is odd.
* `L_chord(a, i)` / `L_long(i)` — a single chord generator / the canonical
  type-2 long chord (the primary trace seed).
* `mult_generators()` — all single physical-type chords.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra
from zplus_ring import AbelianZPlusRing, ZPlusRing


class UngaugedPolygonKAlg(KAlgebra):
    """Ungauged μ-flavoured even-polygon, wrapping a standalone gauged polygon."""

    k: int = None
    _GAUGED_CLASS = None
    _CHARGES = None

    def __init__(self) -> None:
        if self._GAUGED_CLASS is None or self._CHARGES is None:
            raise TypeError(
                f"{type(self).__name__} is abstract: a concrete subclass must set "
                f"_GAUGED_CLASS, _CHARGES and k.")
        self._u1 = self._GAUGED_CLASS()
        self._R = AbelianZPlusRing(rank=1)
        self._ung = None

    # -- KAlgebra contract -------------------------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return ((), 0)

    def r_label_decompose(self, label):
        # μ-flavoured: strip the central μ-charge — section = (factors, 0),
        # single flavour-irrep key = (μ_charge,).  (`_label_section_decompose`
        # derives from this via the base forward bridge, returning a proper
        # RElement so `to_R_form` works.)
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
        """Delegate to the gauged polygon (which handles arbitrary
        cone-monomial inputs), enforcing mag-zero on inputs and outputs.  The
        centralizer is closed under the gauged product, so the output check is
        a safety net.  Returns the gauged `Element` (Z-form)."""
        self._assert_mag_zero(a, "first operand")
        self._assert_mag_zero(b, "second operand")
        out = self._u1.multiply(a, b)
        for label_out in out.terms:
            self._assert_mag_zero(label_out, f"output term {label_out}")
        return out

    def trace(self, label, K=20):
        """μ-flavoured (ungauged) trace — **BPS-free**.

        Ungauging the U(1) restores the vector-multiplet measure, so the
        flavoured index is the gauge-charge-graded sum of the gauged
        Wilson-line traces over the measure:

            Tr_ung(a)(z)  =  [ Σ_n z^n · Tr_gauged(a·μ^n) ] / (fq²;fq²)_∞² .

        Delegates to the general `UngaugedKAlgebra` over the standalone gauged
        polygon (whose trace is the certified orthonormality bootstrap, adaptive
        in gauge half-width, so it reaches the deep `Tr(L·μ^n)` the n-sum needs).
        Equals `ungauge_kalgebra.ungauge_u1polygon(k).trace`."""
        if self._ung is None:
            from ungauge_kalgebra import UngaugedKAlgebra
            self._ung = UngaugedKAlgebra(self._u1, ((), 1), epow=lambda lbl: lbl[1])
        return self._ung.trace(label, K)

    # -- geometry accessors (k-general) ------------------------------------

    def _family_period(self, a):
        return self._CHARGES.PRIMITIVE_FAMILY_PERIOD[a]

    def physical_chord_types(self):
        """The flavour-neutral (mag-zero) chord families — the even types
        {2, 4, ...} present (odd types are magnetic)."""
        return sorted(a for a in self._CHARGES.PRIMITIVE_FAMILY_PERIOD if a % 2 == 0)

    def L_chord(self, a, i):
        """Single chord generator L_{a,i} as an ungauged label `(((a,i,1),), 0)`.
        Mag-zero exactly for the physical even types (see `physical_chord_types`);
        odd types are magnetic and only enter in mag-zero products."""
        return (((a, i % self._family_period(a), 1),), 0)

    def L_long(self, i):
        """The physical type-2 long chord L_{2,i} — the primary trace seed."""
        return self.L_chord(2, i)

    def mult_generators(self):
        """All single physical (even-type) chords — the flavour-neutral mag-zero
        generators of the ungauged polygon."""
        return [self.L_chord(a, i)
                for a in self.physical_chord_types()
                for i in range(self._family_period(a))]

    # -- internals ---------------------------------------------------------

    def _mag(self, label):
        """Magnetic charge of an ungauged label (via the gauged polygon)."""
        return self._u1._mag_charge(label)

    def _assert_mag_zero(self, label, context=""):
        m = self._mag(label)
        if m != 0:
            raise ValueError(
                f"{type(self).__name__} label {label} has mag-charge {m} ≠ 0 "
                f"({context}).  The ungauged basis is restricted to mag-zero cone "
                f"monomials (the centralizer of the gauge generator).")

    @classmethod
    def charge_of_label(cls, label):
        """Tropical B_GAUGED charge of an ungauged label:
        `e_E·E_CHARGE + Σ exp·chord_charge(a,i)`."""
        chord_charge, E_CHARGE = cls._CHARGES.chord_charge, cls._CHARGES.E_CHARGE
        factors, e_E = label
        ch = [e_E * c for c in E_CHARGE]
        for (a, i, e) in factors:
            fc = chord_charge(a, i)
            for j in range(len(E_CHARGE)):
                ch[j] += e * fc[j]
        return tuple(ch)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
