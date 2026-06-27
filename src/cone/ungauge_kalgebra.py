"""ungauge_kalgebra.py — a general U(1) "ungauger" for gauged KAlgebras.

Sharp definition.  Pick an
**electric generator** `E` — an element that fq-commutes with everything
(a quantum-torus / Laurent direction).  Then for any `x`

    E · x  =  fq^{mag(x)} · (x · E),

so the **magnetic charge** `mag(x)` is read off as that fq-power.  The
**ungauged algebra is the centralizer of E**,

    Z(E)  =  { x : E·x = x·E }  =  { x : mag(x) = 0 },

a sub-KAlgebra (mag is additive, so Z(E) is closed under multiply, ρ).
Inside Z(E) the electric generator `E` is central, so it is **promoted to
a U(1) flavour fugacity** z: the coefficient ring gains an
`AbelianZPlusRing(1)` factor.  `E` itself is then the fugacity `z·𝟙`
(so `E^m = z^m·𝟙`, *not* an independent basis vector); the canonical
basis is the E-free "matter" of Z(E).

Trace — ungauging restores the U(1) **vector-multiplet measure**, so the
ungauged flavoured index is the gauge-charge-graded sum of the gauged
Wilson-line traces divided by `(fq²;fq²)_∞^{measure_power}` (default 2 =
`(q;q)²_∞`):

    Tr_ung(a)(z)  =  [ Σ_n z^n Tr_gauged(a·E^n) ] / (fq²;fq²)_∞^p.

For `U1A1AoddKAlg(1)` this reproduces the independently-built
`A1AoddToEvenRGKAlgebra(1).Tr(1)` (the μ-flavoured `[A_1,A_3]` index)
term-for-term.

`UngaugedKAlgebra(G, E, epow)`:
  * `E`      — the electric-generator label (fq-commutes with all of `G`);
  * `epow`   — `label -> int`, the E-power grading (-> fugacity z).
Magnetic charge is computed intrinsically from the `E`-commutator.
"""
from __future__ import annotations

from kalgebra import KAlgebra, Element, Label
from zplus_ring import (
    ZPlusRing, AbelianZPlusRing, TrivialZPlusRing, RElement, RPowerSeries,
)


def _monomial_qpow(elt: Element):
    """The single fq-exponent of a one-term monomial Element (the form an
    fq-commuting product takes); None if not a clean monomial.  Handles both
    scalar `LaurentPoly` coefficients (`._coeffs`, keyed by fq-power) and
    R-valued `RLaurent` coefficients (`.coeffs`, keyed by fq-power -> base-ring
    character) — the latter is what a *flavoured* gauged algebra gives (e.g. the
    SU(2)-valued U(1)-gauged D-even, whose E-commutator is q^{mag}·χ, a single
    fq-power times an SU(2) character)."""
    if len(elt.terms) != 1:
        return None
    (lbl, lp), = elt.terms.items()
    coeffs = getattr(lp, "_coeffs", None)
    if coeffs is None:
        coeffs = getattr(lp, "coeffs", None)
    if coeffs is None or len(coeffs) != 1:
        return None
    return lbl, next(iter(coeffs))


class UngaugedKAlgebra(KAlgebra):
    def __init__(self, gauged: KAlgebra, E: Label, epow, e_shift=None,
                 measure_power: int = 2) -> None:
        self._G = gauged
        self._E = E
        self._epow = epow
        # `e_shift(label, n)` = label with the E-power raised by n (x·E^n,
        # exact since x∈Z(E) centralizes E).  Default: cone label (f, e_E).
        self._e_shift = e_shift or (lambda lbl, n: (lbl[0], lbl[1] + n))
        self._measure_power = measure_power
        R_G = gauged.coefficient_ring()
        self._flav = AbelianZPlusRing(rank=1)
        self._base_trivial = isinstance(R_G, TrivialZPlusRing)
        if self._base_trivial:
            self._R: ZPlusRing = self._flav
        else:
            from tensor_zplus_ring import TensorZPlusRing
            self._R = TensorZPlusRing(R_G, self._flav)

    # ----- magnetic charge from the E-commutator (intrinsic) --------------
    def mag(self, x: Label) -> int:
        """`mag(x)` such that `E·x = fq^{mag(x)} (x·E)`.  Requires E to
        fq-commute with x (both products one monomial on the same label)."""
        Ex = _monomial_qpow(self._G.multiply(self._E, x))
        xE = _monomial_qpow(self._G.multiply(x, self._E))
        if Ex is None or xE is None or Ex[0] != xE[0]:
            raise ValueError(
                f"electric generator does not fq-commute with {x!r}: "
                f"E·x={self._G.multiply(self._E, x)}, x·E={self._G.multiply(x, self._E)}"
            )
        return Ex[1] - xE[1]

    def in_centralizer(self, x: Label) -> bool:
        """`x ∈ Z(E)` — magnetically neutral (E commutes with x exactly)."""
        return self._G.multiply(self._E, x).terms == self._G.multiply(x, self._E).terms

    # ----- flavour combine (mirror AddFlavourKAlgebra._combine_r) ----------
    def _combine_r(self, r_B: RElement, f: int) -> RElement:
        fb = (f,)
        if self._base_trivial:
            z = r_B.terms.get((), 0)
            return RElement(self._R, {fb: z}) if z else self._R.zero()
        return RElement(self._R, {(b, fb): c for b, c in r_B.terms.items()})

    # ----- KAlgebra contract ----------------------------------------------
    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self) -> Label:
        return self._G.identity()

    def multiply(self, a: Label, b: Label) -> Element:
        """Inherit `G`'s product; the centralizer is closed, so every term
        is magnetically neutral (filtered as a safety net)."""
        prod = self._G.multiply(a, b)
        out = Element.zero()
        for L, lp in prod.terms.items():
            if lp.is_zero():
                continue
            if not self.in_centralizer(L):
                continue
            out = out + Element({L: lp})
        return out

    def rho(self, a: Label) -> Label:
        return self._G.rho(a)

    def rho_inverse(self, a: Label) -> Label:
        return self._G.rho_inverse(a)

    def _inv_measure(self, K: int) -> "dict[int, int]":
        """`1 / (fq²;fq²)_∞^{measure_power}` as an fq-series to order K — the
        inverse U(1) vector-multiplet measure that ungauging restores."""
        inv = {0: 1}
        for _ in range(self._measure_power):
            for j in range(1, K + 1):
                nxt: dict[int, int] = {}
                for e, c in inv.items():
                    m = 0
                    while e + 2 * j * m <= K:
                        nxt[e + 2 * j * m] = nxt.get(e + 2 * j * m, 0) + c
                        m += 1
                inv = nxt
        return inv

    def trace(self, a: Label, K: int = 20) -> RPowerSeries:
        """Flavoured (ungauged) trace.  Ungauging a U(1) restores the
        vector-multiplet measure, so the ungauged index is the
        gauge-charge-graded sum of the gauged Wilson-line traces divided by
        `(fq²;fq²)_∞^{measure_power}`:

            Tr_ung(a)(z)  =  [ Σ_n z^n · Tr_gauged(a·E^n) ]  /  (fq²;fq²)_∞^p.

        (a·E^n is exact since a∈Z(E); the n-sum is finite to order K because
        the conformal weight grows with |n|.)"""
        acc: dict[int, dict[int, RElement]] = {}     # fqexp e -> {gauge charge n -> base char}
        for n in range(-(K + 1), K + 2):
            tg = self._G.trace(self._e_shift(a, n), K)
            terms = tg._coeffs if hasattr(tg, "_coeffs") else tg.coeffs
            for e, rc in terms.items():
                if hasattr(rc, "is_zero") and rc.is_zero():
                    continue
                acc.setdefault(e, {})[n] = rc        # (e, n) is hit once per n
        inv = self._inv_measure(K)
        # combine each base character `rc` at gauge charge `n` with the U(1)
        # fugacity z^n (via `_combine_r`, the trivial/tensor-base flavour merge),
        # scaled by the restored vector-multiplet measure coefficient `fc`.
        out_terms: dict[int, dict] = {}              # e+fe -> {self._R basis key -> coeff}
        for e, nd in acc.items():
            for fe, fc in inv.items():
                if e + fe > K:
                    continue
                for n, rc in nd.items():
                    base_term = self._combine_r(rc, n)
                    d = out_terms.setdefault(e + fe, {})
                    for k, v in base_term.terms.items():
                        d[k] = d.get(k, 0) + v * fc
        out: dict[int, RElement] = {}
        for e, d in out_terms.items():
            d = {k: v for k, v in d.items() if v}
            if d:
                out[e] = RElement(self._R, d)
        return RPowerSeries(self._R, out, K)

    def r_label_decompose(self, label: Label):
        """Ungauge lift coordinate: the base `_G` section + single irrep,
        combined with the new U(1) flavour charge `f` (the E-power offset from
        the section).  Key = `(f,)` (trivial base) or `(b, (f,))` (base irrep
        `b` ⊗ the U(1) charge).  Delegates to `_G.r_label_decompose` (raises if
        `_G` is not yet migrated)."""
        base_sec, b = self._G.r_label_decompose(label)
        f = self._epow(label) - self._epow(base_sec)
        key = (f,) if self._base_trivial else (b, (f,))
        return base_sec, key

    def r_label_compose(self, section: Label, r_basis_label) -> Label:
        if self._base_trivial:
            (f,) = r_basis_label
            b = self._G.coefficient_ring().one_basis()
        else:
            b, (f,) = r_basis_label
        g = self._G.r_label_compose(section, b)
        f0 = self._epow(g) - self._epow(section)
        return self._e_shift(g, f - f0)

    def __repr__(self) -> str:
        return f"UngaugedKAlgebra({self._G!r}, centralizer of {self._E!r})"


def ungauge_u1a1aodd(k: int) -> "UngaugedKAlgebra":
    """Ungauge `U1A1AoddKAlg(k)` -> the μ-flavoured `[A_1, A_{2k+1}]` as the
    centralizer of the gauge generator E = `((), 1)`; E-power -> fugacity.

    Uses the oracle-backed *canonical* `U1A1AoddKAlg(k)`; for a runtime-BPS-free
    ungauging use `ungauge_u1polygon(k)` instead."""
    from u1a1aodd_kalg import U1A1AoddKAlg
    G = U1A1AoddKAlg(k)
    E = ((), 1)                       # the electric generator (one E)
    return UngaugedKAlgebra(G, E, epow=lambda lbl: lbl[1])


# The standalone gauged even-polygons, keyed by k (= the (2k+4)-gon).
_U1POLYGON = {
    1: ("u1_hexagon_kalg", "U1HexagonKAlg"),       # hexagon   [A_1, A_3]
    2: ("u1_octagon_kalg", "U1OctagonKAlg"),       # octagon   [A_1, A_5]
    3: ("u1_decagon_kalg", "U1DecagonKAlg"),       # decagon   [A_1, A_7]
    4: ("u1_dodecagon_kalg", "U1DodecagonKAlg"),   # dodecagon [A_1, A_9]
}


def ungauge_u1polygon(k: int) -> "UngaugedKAlgebra":
    """Ungauge the **standalone** gauged even-polygon `U1{Polygon}KAlg` (k=1..4)
    -> the μ-flavoured `[A_1, A_{2k+1}]`, as the centralizer of the gauge
    generator E = `((), 1)` (the E-power becomes the U(1) flavour fugacity).

    This is the **fully BPS-free** ungauging: the standalone gauged polygon
    imports no BPS / RG-oracle at runtime, and its trace (the
    measure-restored gauge-charge-graded sum) is computed via the certified
    orthonormality bootstrap (now adaptive in gauge half-width, so it reaches
    the deep `Tr(L·E^n)` the ungauging needs).  Reproduces
    `ungauge_u1a1aodd(k).trace` term-for-term on the vacuum index and
    module-for-module on the chords (the hexagon up to its μ-base z-origin)."""
    if k not in _U1POLYGON:
        raise ValueError(f"ungauge_u1polygon: k must be in {sorted(_U1POLYGON)}, got {k}")
    import importlib
    mod, cls = _U1POLYGON[k]
    G = getattr(importlib.import_module(mod), cls)()
    return UngaugedKAlgebra(G, ((), 1), epow=lambda lbl: lbl[1])


def ungauge_u1a1deven(k: int = 1) -> "UngaugedKAlgebra":
    """Ungauge the spine-free `U1A1DevenConeKAlgebra(k)` (the U(1)-gauged
    `[A_1, D_{2k+2}]`) -> the **ungauged** `[A_1, D_{2k+2}]` with SU(2)×U(1)
    flavour, as the centralizer of the gauge generator E = `((), 1)` (the
    `X_{0,1}` Wilson direction; its power becomes the U(1) flavour fugacity z).

    The SU(2) matter flavour rides in the gauged base ring, so the ungauged
    coefficient ring is `SU(2) ⊗ U(1)` (a `TensorZPlusRing`).  Reproduces
    `A1DevenRGKAlgebra(k).trace` **term-for-term** (validated k=1 through q^6 —
    the SU(2)×U(1)-flavoured [A_1,D_4] index).  Fully spine-free: the gauged
    D-even is construction- and runtime-spine-free, and the ungauger touches no
    BPS/RG engine."""
    from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra
    G = U1A1DevenConeKAlgebra(k)
    return UngaugedKAlgebra(G, ((), 1), epow=lambda lbl: lbl[1])
