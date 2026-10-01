"""
a1deven_kalg.py
===============

`A1DevenKAlg(k)` — the **ungauged** `[A_1, D_{2k+2}]` Argyres–Douglas algebra,
obtained by **ungauging the U(1)** of `U1A1DevenConeKAlgebra(k)`.

Construction (mirrors the ungauged A-polygons, over a flavoured base)
--------------------------------------------------------------------
`U1A1DevenConeKAlgebra(k)` is the U(1)-gauged `[A_1, D_{2k+2}]` with SU(2)
flavour, on labels `(curves, e, κ)`: curves of the once-punctured `(2k+2)`-gon,
the power `e` of the gauge letter `E = X_{0,1}`, the SU(2) weight `κ`.  `E =
((), 1, 0)` `𝖖`-commutes with everything; the magnetic charge `mag(x)` (the
`X_{1,0}` 't Hooft charge) is read off the `E`-commutator (the gauged class
supplies it in closed form, `_label_mag`: `−2·Σ m·c0`).  The **ungauged**
algebra is the centralizer `Z(E) = {mag = 0}`, with `E` promoted to a U(1)
flavour fugacity `z`, `E = z⁻¹`.  So the ungauged coefficient ring is
`SU(2) ⊗ U(1)`.

Labels (in the curve frame since 2026-09-24)
--------------------------------------------
A label is a gauged label `(F, e, κ)` in the centralizer: `F` a BALANCED
multiset of pairwise non-crossing curves of the once-punctured `(2k+2)`-gon —
as many curves of magnetic charge `+1` as of `−1`, counted with multiplicity
(the charge of a curve is set by the parity of its endpoints; odd-`ℓ` curves
have charge 0) — that multiset being the geometric labelling of this
family; `e` the U(1) weight and `κ` the SU(2) weight:

    L_{(F, e, κ)} = z^{−e} · χ_κ · L_{(F, 0, 0)}.

The flavour is in the label (the Z-form of the `KAlgebra` contract):
`multiply` returns integral `LaurentPoly` coefficients on full labels, and the
flavour-lift coordinate is `UngaugedKAlgebra`'s,
`r_label_decompose((F, e, κ)) = ((F, 0, 0), (κ, (−e,)))` — the section the
`E`-free, `SU(2)`-singlet label, the U(1) weight MINUS the `E`-power, the sign
the trace fixes (`Tr((F, e, κ)) = z^{−e}·χ_κ·Tr((F, 0, 0))`).
Until 2026-09-24 `multiply` stripped the `E`-power off every term into an
`RLaurent` coefficient over `SU(2) ⊗ U(1)` (an "`E`-free canonical basis"),
because the gauged base carried SU(2) in `RLaurent` coefficients and that
kept products and traces in one ring; with the gauged base in Z-form that
reason is gone, and the R-form is `to_R_form`, as for every other class
(the old encoding made `to_R_form` raise).

The E-normalisation is the gauged class's (its edge-{1,2} convention).
Against the retired table frame the same canonical elements moved by a power
of `z` on 3 / 14 / 37 of the 8 / 39 / 120 generators at k = 1 / 2 / 3 (measured
2026-09-24): a table-frame
generator `(F_tab, 0)` is `z^{s}·L_{(F, 0, 0)}` with `s ∈ {−1, 0, +1}` — the
section choice the flavour lift leaves free.

Validation
----------
`trace` reproduces `A1DevenRGKAlgebra(k).trace` **term-for-term** on `Tr(1)`
(k = 1, 2, 3 through 𝖖⁶ — the SU(2)×U(1)-flavoured `[A_1, D_{2k+2}]` index);
that reference has no label map to these labels beyond the identity.  The
ungauger itself touches no BPS/RG engine, and neither does the gauged trace
it sums: the exact transport (`u1a1deven_trace_transport`) on the gauged
class's closed-form RG image, which builds the auxiliary algebra and `S_RG`
itself and imports no RG module (the source repository's test `test_reference_flows_not_imported_at_runtime`).  The `k = 2` orthonormality pole of the retired table frame stays gone
(the source repository's test `test_k2_pole_pair_regression`).

Depth.  `trace(a, K)` sums the gauged traces of `a·Eⁿ` over the total
`E`-power window `|n| ≤ K + 1`.  Every generator is a seed of the gauged
class — an odd curve, or a non-crossing pair of a +1 and a −1 curve — so
`a·Eⁿ` is a seed at every `n`, and since 2026-09-24 the gauged class traces
seeds from their closed forms (`u1a1deven_seed_characters`, MEASURED against
the exact transport, not derived; recorded in the design notes): no
depth limit on the generators.  Measured 2026-09-24, every
generator in one process: through `𝖖⁴⁰` in 0.2 s (k = 1, 8 generators, peak
RSS 23 MB), 0.8 s (k = 2, 39, 43 MB), 2.2 s (k = 3, 120, 81 MB); equal to the
transport route (`seed_closed_forms=False` on the gauged class) on every
generator at `𝖖¹⁰` / `𝖖⁶` / `𝖖⁴`.  A label that is not a generator (a product,
so every pairing) is traced by the gauged class's Layer-1 reduction onto the
seeds (the same day), again with no depth limit and no transport:
every ordered pair of the identity and the generators, `I(a, b) = δ + O(𝖖)`
at `K = 3`, in 11 s at k = 2 (1,600 pairs) and 144 s at k = 3 (14,641 pairs)
— through the transport the k = 2 run took about 15 minutes.  The transport
route stays the witness (`seed_closed_forms=False` on the gauged class; its
limit on the length of an A1Dodd word, `u1a1deven_trace_transport.
_MAX_WORD_DEGREE`, raises `ValueError` past it, never a silent truncation).
Before the closed forms — every generator through the transport, measured
2026-09-23 in the table frame — the served depths were k = 1 `𝖖¹⁶` (8/8,
116 s, longest word 19 letters, peak RSS 39 MB), k = 2 `𝖖¹²` (39/39, 771 s,
16 letters, 99 MB), k = 3 `𝖖⁸` (120/120, 1027 s, 13 letters, 145 MB).
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

from zplus_ring import RPowerSeries
from ungauge_kalgebra import UngaugedKAlgebra


class A1DevenKAlg(UngaugedKAlgebra):
    """Ungauged `[A_1, D_{2k+2}]` (SU(2)×U(1) flavour) — the centralizer of the
    gauge letter `E = ((), 1, 0)` in `U1A1DevenConeKAlgebra(k)`, `E` promoted
    to the U(1) fugacity (`E = z⁻¹`).  Labels `(F, e, κ)`, `F` a balanced
    multiset of curves; Z-form (module docstring)."""

    def __init__(self, k: int = 1) -> None:
        from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra
        G = U1A1DevenConeKAlgebra(k)
        super().__init__(G, ((), 1, 0), epow=lambda lbl: lbl[1],
                         e_shift=lambda lbl, n: (lbl[0], lbl[1] + n, lbl[2]))
        self.k = k
        self._section_trace: dict = {}        # section -> (K, RPowerSeries)

    def trace(self, a, K: int = 20):
        """`Tr((F, e, κ)) = z^{−e}·χ_κ·Tr((F, 0, 0))` — the flavour lift
        (`r_label_decompose`) — with `Tr((F, 0, 0))` the inherited sum over the
        `E`-window, memoised per section and extending in depth.  The two
        sides sum the same gauged traces `Tr((F, t, κ))`, `|t| ≤ K + 1`, so
        this equals the inherited `UngaugedKAlgebra.trace` on every label
        (compared in the source repository's test `test_lift_follows_the_trace`);
        it only avoids re-summing the window for every `E`-power and `SU(2)`
        weight of one section, which the Z-form keeps as separate labels."""
        sec, key = self.r_label_decompose(a)
        hit = self._section_trace.get(sec)
        if hit is None or hit[0] < K:
            hit = (K, UngaugedKAlgebra.trace(self, sec, K))
            self._section_trace[sec] = hit
        base = RPowerSeries(self._R, {q: c for q, c in hit[1].coeffs.items()
                                      if q <= K}, K)
        return base * self._R.basis_element(key)

    def mult_generators(self):
        """The multiplicative generators of the ungauged algebra, as E-free
        `SU(2)`-singlet labels `(F, 0, 0)`: the curves of magnetic charge 0
        (odd `ℓ`) and the pairs of a `+1` and a `−1` curve that do not cross.
        Order: the single curves, then the pairs,
        each sorted.  Counts 8 / 39 / 120 at k = 1 / 2 / 3 (4 + 4, 12 + 27,
        24 + 96).

        Why these generate: every cone of the gauged class is a set of
        pairwise non-crossing curves with `E^{±1}` (simplicial — its canonical
        elements are the monomials in its letters), and the curves carry
        magnetic charges `0, ±1` only, so a balanced monomial splits into
        charge-0 curves and (`+1`, `−1`) pairs of the same cone.  `E` itself
        is the fugacity here.  The test suite recovers the same set from the
        cones and the `E`-commutator, without this rule."""
        from u1a1deven_geometric_frame import _charge, _curves_cross
        G = self._G
        cs = sorted(G._curve_set)
        ch = {c: _charge(c, G.k) for c in cs}
        singles = [(((c, 1),), 0, 0) for c in cs if ch[c] == 0]
        pairs = sorted((tuple(sorted(((a, 1), (b, 1)))), 0, 0)
                       for a in cs for b in cs
                       if ch[a] == 1 and ch[b] == -1 and not _curves_cross(a, b, G.n))
        return singles + pairs

    def __repr__(self) -> str:
        return f"A1DevenKAlg(k={self.k})"
