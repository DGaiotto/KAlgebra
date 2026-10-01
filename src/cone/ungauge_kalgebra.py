"""ungauge_kalgebra.py — a general U(1) "ungauger" for gauged KAlgebras.

Sharp definition ("Parallel ungauging").  Pick an
**electric generator** `E` — an element that fq-commutes with everything
(a quantum-torus / Laurent direction).  Then for any `x`

    E · x  =  fq^{mag(x)} · (x · E),

so the **magnetic charge** `mag(x)` is read off as that fq-power.  The
**ungauged algebra is the centralizer of E**,

    Z(E)  =  { x : E·x = x·E }  =  { x : mag(x) = 0 },

a sub-KAlgebra (mag is additive, so Z(E) is closed under multiply, ρ).
Inside Z(E) the electric generator `E` is central, so it is **promoted to
a U(1) flavour fugacity** z: the coefficient ring gains an
`AbelianZPlusRing(1)` factor.  `E` itself is then the fugacity `z⁻¹·𝟙`
(`E^m = z^{−m}·𝟙`, the orientation the trace below fixes), so the
unflavoured canonical basis is the E-free "matter" of Z(E): the flavour-lift
coordinate is `(F, e) ↦ ((F, 0), (−e,))` (`r_label_decompose`).  Until
2026-09-23 this docstring said `z·𝟙`, which the trace contradicts.

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

Four things are read from the gauged class when it supplies them: the
magnetic charge of a label (`_label_mag`), which then replaces the two
multiplies of `in_centralizer`; the multiplicative generators of the
centralizer (`_centralizer_generators`, served as `mult_generators`); the
geometric label of a letter (`geometric_label`, assembled into the geometric
label of an ungauged label); and the gauge charge of a label
(`_label_gauge_charge`), which centres the trace's gauge-charge window (see
`trace`).  `U1A1AoddKAlg` supplies all four — by the endpoint-parity rule for
the magnetic charge, see `ungauge_u1a1aodd`.
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
        """`x ∈ Z(E)` — magnetically neutral (E commutes with x exactly).

        If the gauged class supplies `_label_mag(x, E)` — the magnetic charge
        of a label read off its letters (`U1A1AoddKAlg`: the endpoint-parity
        rule) — and it covers `x`, the test is `_label_mag(x, E) == 0`, with
        no multiply; `_label_mag` returns `None` on anything it does not
        cover, and the test is then the multiply-based `E·x == x·E`.  On the
        labels it covers the two are equal (argument in
        `U1A1AoddKAlg._label_mag`).  Measured as the filter of `multiply`:
        200 generator products at `k = 3`, 0.098 s -> 0.009 s."""
        label_mag = getattr(self._G, "_label_mag", None)
        if label_mag is not None:
            m = label_mag(x, self._E)
            if m is not None:
                return m == 0
        return self._G.multiply(self._E, x).terms == self._G.multiply(x, self._E).terms

    # ----- generators and geometric labels (delegated to the gauged class) --
    def mult_generators(self) -> "list[Label]":
        """The multiplicative generators of the ungauged algebra, as labels:
        the cone generators of the centralizer `Z(E)` other than `E^{±1}`
        (`E` is the fugacity here).  Delegated to the gauged class's
        `_centralizer_generators(E)`; for `ungauge_u1a1aodd(k)` these are the
        mixed-parity diagonals and the non-crossing (even–even, odd–odd)
        pairs of diagonals of the `(2k+4)`-gon, `6, 24, 65, 144, 280` at
        `k = 1..5`.  `NotImplementedError` when the gauged class does not
        supply them: listing only the single gauged generators that happen to
        lie in `Z(E)` would silently miss the pairs."""
        gens = getattr(self._G, "_centralizer_generators", None)
        out = gens(self._E) if gens is not None else None
        if out is None:
            raise NotImplementedError(
                f"{type(self._G).__name__} does not supply the generators of "
                f"the centralizer of {self._E!r} (_centralizer_generators)")
        return list(out)

    def geometric_label(self, label: Label):
        """The label `(F, e)` as `(curves, e)`: `curves` the sorted tuple of
        pairs `(geometric label of the letter, multiplicity)`, from the gauged
        class's `geometric_label` letter by letter, and `e` the label's
        E-power, unchanged (the trace carries it as `z^{−e}`:
        `Tr((F, e)) = z^{−e}·Tr((F, 0))`).  The layout of `A1DnKAlg`'s
        `(curves, κ)` labels.  A factor of `F` is `(letter, m)`, or the
        flattened `(t, i, m)` of `U1A1AoddKAlg`'s letters `(t, i)`; a label with
        a slot after the E-power (the SU(2) weight `κ` of
        `U1A1DevenConeKAlgebra`'s `(curves, e, κ)`) keeps it:
        `(curves, e, κ)`.

        For `ungauge_u1a1aodd(k)`, `curves` is a multiset of pairwise
        non-crossing diagonals `(v1, v2)` of the `(2k+4)`-gon holding as many
        even–even diagonals as odd–odd ones, counted with multiplicity; that
        multiset is the ungauged algebra's geometric label (see
        `ungauge_u1a1aodd`).  For `A1DevenKAlg(k)` it is the balanced multiset
        of curves `(x, ℓ)` of the once-punctured `(2k+2)`-gon —
        the gauged letters are the curves themselves, so the geometric label
        is the label.

        `NotImplementedError` when the gauged class has no `geometric_label`;
        `ValueError` on a label that is not in this algebra (an exponent below
        1, two letters that do not q-commute, a letter without a geometric
        label, or a label outside `Z(E)`)."""
        letter_label = getattr(self._G, "geometric_label", None)
        if letter_label is None:
            raise NotImplementedError(
                f"{type(self._G).__name__} supplies no geometric_label")
        factors, e, rest = label[0], label[1], tuple(label[2:])

        def split(f):
            letter, m = tuple(f[:-1]), f[-1]
            return (letter[0] if len(letter) == 1 else letter), m

        factors = [split(f) for f in factors]
        letters = [g for g, _m in factors]
        if any(m < 1 for _g, m in factors):
            raise ValueError(f"{label!r}: an exponent below 1")
        cone_data = getattr(self._G, "cone_data", None)
        if cone_data is not None:
            cd = cone_data()
            for a in range(len(letters)):
                for b in range(a + 1, len(letters)):
                    if not cd.q_commute(letters[a], letters[b]):
                        raise ValueError(
                            f"{label!r}: letters {letters[a]} and {letters[b]} "
                            f"do not q-commute (their curves cross)")
        if not self.in_centralizer(label):
            raise ValueError(
                f"{label!r} is not in the centralizer of {self._E!r} "
                f"(for ungauge_u1a1aodd: not balanced)")
        curves = []
        for g, m in factors:
            c = letter_label(g)
            if c is None:
                raise ValueError(f"{label!r}: letter {g} has no geometric label")
            curves.append((c, m))
        return (tuple(sorted(curves)), e) + rest

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
        the conformal weight grows with the TOTAL E-power epow(a) + n, so the
        window is centred at n = −epow(a).  Until 2026-09-23 it was centred at
        n = 0, which silently dropped every term of a label whose own E-power
        is near or beyond K + 1: `trace(E², 0)` returned 0 instead of z⁻².)

        When the gauged class supplies `_label_gauge_charge(label)` — the gauge
        charge `g` of the label, its E-power included (`U1A1AoddKAlg`: the
        e_1-component of the gauged charge) — the window also covers
        `|g + n| ≤ K + 1`: the terms `Tr_gauged(a·E^n)` concentrate where the
        total gauge charge `g + n` vanishes, not where the E-power does, and
        `val Tr_gauged(a·E^n) ≥ |g + n|` on every label measured (the
        generators at k = 1..3, their powers 2..4 and products of two
        generators).  The E-power window alone dropped terms at k = 3 (fixed
        2026-09-26): the cube of the pair generator `{(1, 8), (3, 7)}`, gauge
        charge 6, lost `−fq³·z⁻⁶` at K = 3, 4, a term the exported a7 cone
        table's own trace carries.  The window is the union of the two, so it
        only ever adds terms of the sum."""
        acc: dict[int, dict[int, RElement]] = {}     # fqexp e -> {gauge charge n -> base char}
        e0 = self._epow(a)
        lo, hi = -e0 - (K + 1), -e0 + K + 1
        charge = getattr(self._G, "_label_gauge_charge", None)
        g = charge(a) if charge is not None else None
        if g is not None:
            lo, hi = min(lo, -g - (K + 1)), max(hi, -g + K + 1)
        for n in range(lo, hi + 1):
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
        """The flavour-lift coordinate: the section is the label with its
        E-power removed, and the U(1) weight is MINUS that E-power,

            (F, e)  ↦  ((F, 0), (−e,))        L_{(F,e)} = z^{−e}·L_{(F,0)},

        with `(F, 0)` read after the gauged class's own lift and its single
        irrep `b` kept beside the U(1) weight: key `(−e,)` over a trivial base,
        `(b, (−e,))` over a flavoured one (the SU(2) of the D-even base, whose
        own lift reads `b = κ` off the label `(curves, e, κ)` since
        2026-09-24; the section is then `(curves, 0, 0)`).

        The sign is the trace's.  `trace` sums `z^n·Tr_G(a·E^n)`, and `a·E^n`
        is the label with its E-power raised by `n` (`e_shift`), so reindexing
        by the total E-power gives `Tr((F, e)) = z^{−e}·Tr((F, 0))`: `E` enters
        as `z^{−1}`, and the trace is R-linear for this lift and not for the
        opposite sign.  (Until 2026-09-23 the section was the gauged class's,
        E-power included, so every E-power was its own section and `z` never
        appeared in a key.)  A class reading this trace with `z ↦ z^{−1}`
        (`UngaugedPolygonKAlg`) pushes the key through that map too."""
        base_sec, b = self._G.r_label_decompose(label)
        e = self._epow(base_sec)
        section = self._e_shift(base_sec, -e)
        return section, ((-e,) if self._base_trivial else (b, (-e,)))

    def r_label_compose(self, section: Label, r_basis_label) -> Label:
        """Inverse of `r_label_decompose`: `L_{χ_b ⊗ z^w · section}`, i.e. the
        gauged class's `χ_b·L_section` with its E-power lowered by `w`."""
        R_G = self._G.coefficient_ring()
        if self._base_trivial:
            (w,) = r_basis_label
            b = R_G.one_basis()
        else:
            b, (w,) = r_basis_label
        g = section if b == R_G.one_basis() else self._G.r_label_compose(section, b)
        return self._e_shift(g, -w)

    def __repr__(self) -> str:
        return f"UngaugedKAlgebra({self._G!r}, centralizer of {self._E!r})"


def ungauge_u1a1aodd(k: int) -> "UngaugedKAlgebra":
    """Ungauge `U1A1AoddKAlg(k)` -> the μ-flavoured `[A_1, A_{2k+1}]` as the
    centralizer of the gauge generator E = `((), 1)`; E-power -> fugacity.

    Uses the closed-form `U1A1AoddKAlg(k)` (`u1a1aodd_kalg`, since 2026-08-31:
    no frozen data, no RG oracle, any `k`).  The named ungauged polygons
    `OctagonKAlg` / `DecagonKAlg` / `DodecagonKAlg`
    (`ungauged_polygon_kalg.UngaugedPolygonKAlg(k)`) are this algebra with
    the fugacity read as `μ ↦ μ⁻¹`, so that `Tr((F, e)) = μ^{e}·Tr((F, 0))`.
    (`ungauge_u1polygon(k)`, the same ungauger over the stand-alone gauged
    polygons of `k ≤ 4` with their frozen tables and bootstrap seeds, was
    retired on 2026-09-23 with those classes.)  The gauged class's trace is defined on every label: since 2026-09-23 every
    seed comes from the single rule `u1_pgon_layer2.singlet_chord_trace`
    (the fitted long-chord forms it replaced were wrong from about 𝖖²⁶).  (Until 2026-09-23 this docstring
    recommended the opposite, which was right only while this class was the
    oracle-backed predecessor.)

    **Labels.**  A label is a gauged label `(F, e)` in the centralizer of
    `E`: `F` a multiset of pairwise non-crossing diagonals of the
    `(2k+4)`-gon (letter `(t, i)` = diagonal `{i, i+t+1}`), `e` the E-power.
    A letter's magnetic charge is set by the parities of its endpoints —
    `−2` both even, `+2` both odd, `0` mixed (`U1A1AoddConeData._letter_mag`,
    derived from `charge_formula`; equal to `mag` on every letter at
    `k = 1..5`) — so `(F, e)` is in the centralizer iff `F` is BALANCED: as
    many even–even diagonals as odd–odd ones, counted with multiplicity.
    That balanced multiset is the geometric label (`geometric_label`); no
    labelling on a polygon of the ungauged theory's own is known.  The
    multiplicative generators (`mult_generators`) are the mixed-parity
    diagonals and the non-crossing (even–even, odd–odd) PAIRS — `6, 24, 65,
    144, 280` at `k = 1..5` — so some generators are pairs of diagonals, not
    single ones, and from `k = 3` two different words in the pairs can name
    one label (`(A1B1)·(A2B2)` and `(A1B2)·(A2B1)`)."""
    from u1a1aodd_kalg import U1A1AoddKAlg
    G = U1A1AoddKAlg(k)
    E = ((), 1)                       # the electric generator (one E)
    return UngaugedKAlgebra(G, E, epow=lambda lbl: lbl[1])


def ungauge_u1a1deven(k: int = 1) -> "UngaugedKAlgebra":
    """Ungauge `U1A1DevenConeKAlgebra(k)` (the U(1)-gauged
    `[A_1, D_{2k+2}]`) -> the **ungauged** `[A_1, D_{2k+2}]` with SU(2)×U(1)
    flavour, as the centralizer of the gauge generator E = `((), 1, 0)` (the
    `X_{0,1}` Wilson direction; its power becomes the U(1) flavour fugacity z).
    `a1deven_kalg.A1DevenKAlg(k)` is this algebra with its generators listed.

    The gauged labels are `(curves, e, κ)` (since 2026-09-24): the E-power `e`
    is the second slot and the SU(2) weight `κ` rides along in the third, so
    the ungauged labels are Z-form too, with coefficient ring `SU(2) ⊗ U(1)`
    (a `TensorZPlusRing`).  Reproduces `A1DevenRGKAlgebra(k).trace`
    **term-for-term** on `Tr(1)` (k = 1, 2, 3 through q^6 — the
    SU(2)×U(1)-flavoured [A_1, D_{2k+2}] index).  The ungauger touches no
    BPS/RG engine; the gauged traces it sums are the exact transport of the
    gauged class's closed-form RG image (`u1a1deven_trace_transport`), whose
    limit on the length of an A1Dodd word bounds the depth: past it `trace`
    raises `ValueError` (per-k limit and measured coverage: `a1deven_kalg`,
    module docstring, "Depth")."""
    from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra
    G = U1A1DevenConeKAlgebra(k)
    return UngaugedKAlgebra(G, ((), 1, 0), epow=lambda lbl: lbl[1],
                            e_shift=lambda lbl, n: (lbl[0], lbl[1] + n, lbl[2]))
