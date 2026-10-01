"""`KAlgebraObject` for **SU(2) + N_f = 1**.

Realizations:

* ``'cone'`` — `SU2Nf1KAlgebra`: the standalone `ConeKAlgebra`
  (H-tower + Wilson cones over `AbelianZPlusRing(1)`, literal-word
  multiply, μ-twisted ρ `H_n ↦ μ⁻¹H_{n−3}`, standalone cyclicity
  trace).
* ``'abe'`` — `SU2Nf1FlavourInCoefficients` (the retired `SU2Nf1Abe` until
  2026-09-19): the genuine **abelianized** realization, on
  the abelianized tier (the general `(G, N)` tier's `roster('su2-nf1')` since
  2026-09-19; before, `MatterURQTorus` via the retired U(2)+N_f=1
  `UNNfKAlgebra(2,1)`).  The flavour μ is the **matter
  level** of the rational quantum torus (the hyper's U(1)_F — a
  coordinate genuinely separate from the gauge centre, so NO Z₂; the
  gauge centre `λ₁+λ₂` is collapsed).  Correct flavour: `H_n`, `W_n` at
  μ⁰, μ only in `H0·W1 = q·H1 + q⁻¹·H-1 + μ·1`; ρ μ-shifts; generators
  carry no flavour charge.  `multiply`/`ρ` and the **native trace** are
  all certified against cone.  The trace is the **rank-1 SU(2)**
  Schur-measure residue — the pure-U(2) residue with **one fewer**
  `(q²;q²)_∞²` factor (the ungauged central-U(1) photon), keeping the
  matter level as the flavour fugacity μ — so this leg now carries the
  genuine SU(2)+N_f=1 trace, not the U(2)-measure index.
* ``'bps-rform'`` — `SU2Nf1BpsRForm`: the **R-form of the BPS
  realization** — the BPS quantum-torus chart with the `ker(B)` flavour
  folded into `R` (`μ ∈ R`).  Correct flavour; full trace.  This is
  *not* a URQTorus `AbeKAlgebra` (a façade over `BPSKAlgebra`); it
  carries the correct SU(2)+N_f=1 **trace** (the BPS chart measure) — as
  does `abe` (which computes the same trace natively on the rational
  quantum torus, the rank-1 SU(2) measure residue).
* ``'bps'``  — `build_bps_su2_nf1()`: the canonical `BPSKAlgebra` on
  the 3-node quiver `[(1,0,0), (−1,2,0), (0,−1,1)]` (Kronecker pair +
  the flavour-carrying matter node; 5-step strong-coupling spec).
* ``'rg-to-pure-su2'`` — `SubquiverRG(bps, [2])`: the SAME algebra
  presented by its RG flow that decouples the matter dyon
  `γ₃ = (0,−1,1)` (large flavour mass); the IR auxiliary is the
  Kronecker pair = **pure SU(2)**.  The whole KAlgebra API is derived
  from IR + RG data (apex-peel multiply, transported trace) — an
  independent presentation, NOT a UV delegate.
* ``'rg-to-pentagon'`` — `SubquiverRG(bps, [0])`: the flow that
  decouples the 't Hooft node `γ₁ = (1,0,0)`; the IR auxiliary is the
  A₂ pair `(−1,2,0), (0,−1,1)` with `⟨γ₂,γ₃⟩ = 1` in its 3-step
  strong chamber = the **pentagon** (the classic SU(2)+N_f → AD
  decoupling).
* ``'bps-mut'`` — `mutate_bpskalgebra(bps, 2)`: the adjacent BPS
  chamber, necklaced forward at the spec head (the matter dyon) —
  same algebra, *different flow*; witness = the μ_g tropical
  label map.
* ``'abe'``  — **PENDING the correct-flavour rebuild** (see below); not
  wired into the object.
* ``'skein'`` — OPEN SLOT (same deferral as `pure_su2_object`: the
  once-marked annulus family needs the stated self-gluing primitive).

**The two R-form legs (`abe`, `bps-rform`) and the Z₂ resolution
(2026-06-13).**  `abe` (`SU2Nf1FlavourInCoefficients`; `SU2Nf1Abe` until
2026-09-19) is the genuine abelianized
realization — until 2026-09-19 on the **rational quantum torus**
(`MatterURQTorus`, via the retired `UNNfKAlgebra(2,1)`; now `roster('su2-nf1')`):
the U(2)+N_f=1 lines `((m⃗,λ),k)` were mapped to
SU(2) `(m=m⃗[0], e=λ₁−λ₂)` with μ = the **matter level** `k` (the gauge
centre `λ₁+λ₂` collapsed).  Crucially the matter level is a coordinate
**separate from the gauge centre**, so there is **no Z₂** — `H0·W1 =
q·H1 + q⁻¹·H-1 + μ·1` with the meson at matter-level 1 and the dressed
monopoles/Wilsons at level 0 (μ⁰).  This is exactly what the earlier
U(2)-trace-zero abelianized attempt got wrong: it read the **gauge
centre** as the flavour, which IS Z₂-entangled with the SU(2) spinor
charge (that earlier reading stays frozen/provisional).

`bps-rform` (`SU2Nf1BpsRForm`) is the complementary R-form on the **BPS
chart** (a façade over `build_bps_su2_nf1()`, μ in `R`); it is *not* a
URQTorus `AbeKAlgebra` but carries the correct SU(2)+N_f=1 **trace** (the
BPS chart measure).  `abe` has the genuine rational-quantum-torus
presentation **and** the correct trace, computed natively as the rank-1
**SU(2)** Schur-measure residue (the pure-U(2) residue with one fewer
`(q²;q²)_∞²` factor — the ungauged central-U(1) photon — keeping the
matter level as the flavour fugacity μ).  So
both R-forms carry multiply/ρ/trace; `abe` is the rational-quantum-torus
presentation, `bps-rform` the BPS-chart one.

Both R-form witnesses are certified in the **flavour-canonical (folded)
frame**, exactly as `cone ↔ bps` is: cone keeps μ in a label slot
(`μ_p`) while the R-forms keep it in `R`, so the flavour-carrying checks
(multiplicativity forward, and ρ / round-trip on μ-shifting samples) are
compared after folding both sides through `_label_section_decompose`
(the raw Z-form dicts differ for the SAME element).  Unit, raw forward
multiplicativity, and the trace pass directly.

A fully self-contained `abe` not leaning on `UNNfKAlgebra` (the recorded
alternative before the 2026-09-19 retirement) — ungauging a
from-scratch `[SU(2)×U(1)]+hyper` base (RGKAlgebra over
`pure SU(2) ⊗ pure U(1)`, matter `S_RG = E_q(vw)E_q(w/v)`, then
`UngaugedKAlgebra` at the U(1) Wilson) — is the recorded refinement
(pickup board); the native rank-1 trace above already gives the genuine
SU(2)-measure index without it.

Lattice caveat on the flow legs: the node lattice
`⟨γ₁,γ₂,γ₃⟩ = {(m,e,f) : e+f even}` has **index 2** in the charge
lattice Z³ (the SU(2) Wilson class `(0,1,0)` generates the quotient),
and the *derived* flow arithmetic (multiply / trace) needs node-basis
integrality — so the flow legs' derived ops run on the node-integral
sector, while ρ (the UV chart's half-monodromy σ, shared) and the
identity witnesses are total.  The dedicated flow battery in the tests
certifies derived ≡ bps on node-integral baskets.

Witnesses:

* ``cone ↔ bps`` — the pre-existing `su2_nf1_h_iso` (piecewise H-table
  + μ in the `ker(B)` direction); full battery.
* ``cone ↔ abe`` — cone native `(h_factors, μ_p)` ↦ SU(2) `(m, e)` with
  μ^{μ_p}; multiplicativity (forward) + trace raw, round-trip/ρ folded.
* ``cone ↔ bps-rform`` — cone↔bps composed with the trivial f ↔ μ fold.
* ``abe ↔ bps-rform`` — the direct **R-form ↔ R-form** witness; both keep
  μ in R (ρ μ-shifts), so it is **RAW-clean** (no fold: unit / round-trip
  / multiplicativity / ρ / trace all pass directly), unlike the two
  cone↔R-form edges.  Built through cone via the total balanced-H-monomial
  `_psu2nf1_to_native`, so the cone/abe/bps-rform triangle commutes.
* ``bps ↔ {rg-to-pure-su2, rg-to-pentagon}`` — identity-on-labels
  witnesses (the flows live on the BPS charge labels).
* ``bps ↔ bps-mut`` — the μ_g tropical label map.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from kalgebra_object import KAlgebraObject
from laurent_poly import LaurentPoly
from zplus_ring import RElement, RLaurent, RPowerSeries, AbelianZPlusRing

from su2_nf1_h_iso import su2_nf1_h_iso
from su2_nf1_bps_rform import SU2Nf1BpsRForm
from g_matter_roster import roster
from kalgebra import KAlgebra
from su2_nf1_cone_data import _native_to_psu2nf1, _psu2nf1_to_native



def su2_fold(m: int, e: int) -> tuple[int, int]:
    """Fold `(m, e)` into the SU(2) Weyl chamber: `m ≥ 0`, and `e ≥ 0` at `m = 0`."""
    if m < 0:
        m, e = -m, -e
    if m == 0 and e < 0:
        e = -e
    return (m, e)


class SU2Nf1FlavourInCoefficients(KAlgebra):
    """SU(2)+N_f=1 on the general `(G, N)` tier, `roster('su2-nf1')`, presented
    with its U(1) flavour IN THE COEFFICIENTS: labels are the SU(2) lines
    `(m, e)` (folded), the flavour label `w` of the general class's
    `(((m,),(e,)), (w,))` becomes the coefficient `μ^w ∈ R(U(1))`.

    This is the regrouping `L_{(s,w)} = χ_w · L_{(s,0)}` (the design notes
    "Flavour in KAlgebra"; the flavour-lift primitive), applied so the
    object's witnesses written against the retired `SU2Nf1Abe` (flavour =
    the matter level μ in R; retired 2026-09-19 with the type-A layer) keep
    their shape.  ρ is the label-level ρ; its flavour twist sits in the
    coefficients as for every flavour-in-coefficients presentation."""

    def __init__(self) -> None:
        self._G = roster("su2-nf1")
        # The object's other legs (cone, bps-rform) carry the matter level over
        # `AbelianZPlusRing(rank=1)`; the preset's `UNZPlusRing(1)` has the same
        # `(k,)` basis keys, so the regrouping re-keys onto the object's ring.
        self._R = AbelianZPlusRing(rank=1)
        self._w0 = self._G.identity()[1]

    def _to_R(self, rps, K):
        """An `RPowerSeries` over the preset's ring, re-keyed onto `self._R`."""
        return RPowerSeries(self._R, {e: RElement(self._R, dict(c.terms))
                                      for e, c in rps.coeffs.items()
                                      if not c.is_zero()}, K)

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return (0, 0)

    def _lift(self, label):
        m, e = su2_fold(*label)
        return self._G.fold((m,), (e,))          # (((m,),(e,)), one_basis)

    def _regroup(self, x) -> Element:
        out: dict = {}
        for (g, w), C in x.terms.items():
            me = su2_fold(g[0][0], g[1][0])
            row = out.setdefault(me, {})
            coeffs = C._coeffs if hasattr(C, "_coeffs") else C.coeffs
            for qe, z in coeffs.items():
                if not z:
                    continue
                term = RElement(self._R, {tuple(w): z})
                row[qe] = row[qe] + term if qe in row else term
        return Element({me: RLaurent(self._R, row) for me, row in out.items() if row})

    def multiply(self, a, b):
        return self._regroup(self._G.multiply(self._lift(a), self._lift(b)))

    # ρ / ρ⁻¹ return ELEMENTS, as the retired `SU2Nf1Abe` did (`_read(chart.rho())`):
    # the general class's ρ carries the flavour label `w' ` of the image, which
    # in this presentation is the coefficient twist `μ^{w'}` — the ⋆-semilinear
    # `rho_element` of a flavour-in-coefficients presentation (the design notes,
    # "Axiom 5 — ρ-equivariance of the trace", the a3 encoding split).
    def rho(self, label):
        return self._regroup(Element({self._G.rho(self._lift(label)): _ONE}))

    def rho_inverse(self, label):
        return self._regroup(Element({self._G.rho_inverse(self._lift(label)): _ONE}))

    def trace(self, label, K: int = 20):
        return self._to_R(self._G.trace(self._lift(label), K=K), K)

    def _label_section_decompose(self, label):
        return label, self._R.one()

__all__ = ["su2_nf1_object"]

_ONE = LaurentPoly.one()


def su2_nf1_object(with_flows: bool = True) -> KAlgebraObject:
    """The abstract SU(2)+N_f=1 algebra as a `KAlgebraObject`.

    Realizations: cone, bps, abe, bps-rform (+ the RG-flow and mutation
    legs when `with_flows`).  `abe` (`SU2Nf1FlavourInCoefficients`) is the
    genuine abelianized leg, on the general `(G, N)` tier's `roster('su2-nf1')`
    (until 2026-09-19 `SU2Nf1Abe` on the rational quantum torus); `bps-rform` (`SU2Nf1BpsRForm`) is the
    BPS-chart R-form.  Both carry the correct SU(2)+N_f=1 trace.

    `with_flows=False` builds only the cone/bps/abe/bps-rform core (the
    flow and mutation legs add a few seconds of directional-machinery
    construction)."""
    obj = KAlgebraObject("A_q[SU(2)+Nf=1]")

    iso_cb = su2_nf1_h_iso()
    cone = iso_cb.source
    bps = iso_cb.target
    obj.add_realization("cone", cone, {"multiply-fast", "closed-form",
                                       "trace-exact"})
    obj.add_realization("bps", bps, {"chart", "rg", "trace-exact"})
    obj.add_iso("cone", "bps", iso_cb)

    # The R-form of the BPS realization: the same BPS quantum-torus chart
    # with the ker(B) flavour folded into the coefficient ring R (μ ∈ R
    # instead of in the label).  A façade over BPSKAlgebra — NOT an
    # AbeKAlgebra / URQTorus.  Correct flavour (inherited): H_n, W_n at
    # μ⁰; μ only in the monopole×Wilson → meson OPE.  Witness = the
    # cone↔bps iso composed with the (trivial) f ↔ μ fold.
    rform = SU2Nf1BpsRForm()
    obj.add_realization("bps-rform", rform, {"multiply-fast", "trace-exact"})

    def _cone_to_rform(cl):
        return rform._fold_element(iso_cb.map(Element({cl: _ONE})))

    def _rform_to_cone(al):
        return iso_cb.inverse(Element({rform._lift(al): _ONE}))

    obj.add_iso("cone", "bps-rform", KAlgebraIso(
        cone, rform,
        forward_label_map=_cone_to_rform,
        inverse_label_map=_rform_to_cone,
        name="su2nf1[cone→bps-rform]  (BPS chart; flavour folded to R)"))
    # (cone↔bps-rform↔bps is the composite through cone, NOT stored — the
    # raw verify_coherence would compare it against the direct cone↔bps in
    # raw Z-forms, which differ by μ-in-label vs μ-in-R; folded-equal.)

    # The abelianized leg: SU(2)+N_f=1 on the general (G, N) tier
    # (`roster('su2-nf1')`, regrouped with flavour in the coefficients — until
    # 2026-09-19 `SU2Nf1Abe` over the retired `UNNfKAlgebra(2,1)`), flavour =
    # the matter level (μ ∈ R), no Z₂.  Witness:
    # cone native `(h_factors, μ_p)` ↦ SU(2) `(m, e)` with μ^{μ_p}.
    abe = SU2Nf1FlavourInCoefficients()
    obj.add_realization("abe", abe, {"chart", "f-presentation",
                                     "trace-exact"})

    def _cone_to_abe(cl):
        m, e, mp = _native_to_psu2nf1(cl)   # (m, e, μ_p): the third IS μ_p
        return Element({su2_fold(m, e):
                        RLaurent(abe._R, {0: RElement(abe._R, {(mp,): 1})})})

    def _abe_to_cone(al):
        m, e = su2_fold(*al)
        return Element({_psu2nf1_to_native(m, e, 0): _ONE})   # bare line, μ_p=0

    obj.add_iso("cone", "abe", KAlgebraIso(
        cone, abe,
        forward_label_map=_cone_to_abe,
        inverse_label_map=_abe_to_cone,
        name="su2nf1[cone→abe]  (rational quantum torus; μ = matter level)"))

    # Direct R-form ↔ R-form witness: abe (rational quantum torus) ↔
    # bps-rform (BPS chart).  Both keep μ in R with ρ μ-shifting, so — unlike
    # cone↔abe / cone↔bps-rform (μ-in-label vs μ-in-R → folded round-trip/ρ)
    # — this edge is **RAW-clean**: unit / round-trip / multiplicativity / ρ /
    # the (native) trace all verify with no fold.  The forward uses the total
    # balanced-H-monomial `_psu2nf1_to_native` (so abe's gap labels like
    # `(2,−1)` map cleanly); the maps compose through cone, so the triangle
    # commutes (coherence holds).
    def _abe_to_rform(al):
        m, e = su2_fold(*al)
        return _cone_to_rform(_psu2nf1_to_native(m, e, 0))

    def _rform_to_abe(rl):
        out = Element({})
        for cl, C in _rform_to_cone(rl).terms.items():
            m, e, mp = _native_to_psu2nf1(cl)
            coeff = RLaurent(abe._R, {qe: RElement(abe._R, {(mp,): z})
                                      for qe, z in C._coeffs.items()})
            out = out + Element({su2_fold(m, e): coeff})
        return out

    obj.add_iso("abe", "bps-rform", KAlgebraIso(
        abe, rform,
        forward_label_map=_abe_to_rform,
        inverse_label_map=_rform_to_abe,
        name="su2nf1[abe↔bps-rform]  (R-form ↔ R-form; μ in R, raw-clean)"))

    if with_flows:
        from bps_chart_object import mutate_bpskalgebra
        from rg_flow import SubquiverRG

        def _id_iso(src, dst, name):
            fw = lambda label: Element({tuple(label): _ONE})
            return KAlgebraIso(src, dst, fw, fw, name=name)

        # RG-flow presentations: the same
        # algebra realized via its node-deletion flows — matter dyon out
        # (IR = Kronecker = pure SU(2)) and 't Hooft out (IR = A₂ =
        # pentagon).  Labels are the UV charge labels; witness = identity.
        rg_pure = SubquiverRG(bps, [2])
        obj.add_realization("rg-to-pure-su2", rg_pure,
                            {"rg", "chart", "derived"})
        obj.add_iso("bps", "rg-to-pure-su2",
                    _id_iso(bps, rg_pure, "su2nf1[bps→flow(pure SU(2))]"))

        rg_pent = SubquiverRG(bps, [0])
        obj.add_realization("rg-to-pentagon", rg_pent,
                            {"rg", "chart", "derived"})
        obj.add_iso("bps", "rg-to-pentagon",
                    _id_iso(bps, rg_pent, "su2nf1[bps→flow(pentagon)]"))

        # The mutation pair: necklace forward at the
        # spec head (the matter dyon γ₃) — the adjacent BPS chamber.
        # Same algebra, different flow; the witness is the μ_g tropical
        # label map produced by the chart graph.
        bps_mut, mu_w = mutate_bpskalgebra(bps, 2)
        obj.add_realization("bps-mut", bps_mut,
                            {"chart", "rg", "trace-exact"})
        obj.add_iso("bps", "bps-mut", mu_w)

    return obj


if __name__ == "__main__":
    obj = su2_nf1_object()
    print(obj)
    for lab in [(((0, 1),), 0), (((1, 1),), 0), ((((("W", 1)), 1),), 0),
                ((), 1)]:
        print(f"  cone {lab} → bps {dict(obj.transport(lab, 'cone', 'bps').terms)}")
