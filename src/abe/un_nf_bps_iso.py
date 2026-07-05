"""`un_nf_bps_iso` — the all-(N, N_f) **fundamental-matter** `KAlgebraIso`
between the abe shell `UNNfKAlgebra(N, N_f)` and the BPS chart of
`pure_ade.UN_Nf(N, N_f)`, on the anchored + rigidified tropical-map
architecture (user directions 2026-07-02).

Two regimes:

  * **N_f = 1** (trivial flavour, SU(1)): abe labels are bare `(m, λ)` and
    the BPS chart is taken **flavour-free** (`u2_nf1_object.
    flavour_free_un_nf_bps` — the ker-B gauge-centre direction dropped;
    rank-2N lattice, same interleaved coordinates as pure U(N)); the map is
    `un_bps_chamber.UNTropicalMap` unchanged.  Certified at U(2)+N_f=1 and
    U(3)+N_f=1 (`tests/test_un_nf_bps_iso.py`).

  * **N_f ≥ 2** (user ruling 2026-07-02: "flavour-reduce SU(N_f) or U(N_f)
    to Cartan before you iso with BPS"): the iso is between the two
    **Cartan-lowered** presentations —
    `UNNfKAlgebra(N, N_f).lower_flavour(sun_to_cartan_hom)` (SU multiplets
    split into Cartan weights) and the **native flavoured** BPS chart
    (ker B KEPT, coefficient ring `R(U(1)^{N_f})`) lowered along
    `central_collapse_hom` (the central flavour U(1) killed — on the abe
    side it lives inside the gauge centre per D5/D8b, so only after the
    collapse do the two label lattices have the same rank).  Labels on both
    sides are `(section, weight)` pairs with the weight in
    consecutive-difference coordinates; the map is
    `UNNfLoweredTropicalMap`: the `UNTropicalMap` gauge architecture
    (anchors / σ-transport / product derivation) with the weight passing
    through the identity.  The residual weight-sign choice is a genuine
    flavour-Weyl automorphism at N_f = 2, pinned to the identity by
    convention; a mismatch at higher N_f would honest-fail as a
    weight-bucket collision in the derivation.  Certified at U(2)+N_f=2
    (`tests/test_un_nf2_bps_iso.py`).

Matter enters through the algebras (matter-twisted ρ, matter M-factors in
the traces, monopole-bubbling flavour doublets in the products) — the MAP
architecture is unchanged: Wilson anchors (antidominant electric),
bare-monopole anchors (dominant magnetic), σ-transport, product
derivation."""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.dirname(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly
from bps_kalgebra import BPSKAlgebra
import pure_ade as pa
from u2_nf1_object import flavour_free_un_nf_bps
from un_nf_kalgebra import UNNfKAlgebra
from un_bps_chamber import UNTropicalMap, _levi_fold
from sun_cartan_reduction import sun_to_cartan_hom, central_collapse_hom


__all__ = ["un_nf_bps_iso", "un_nf_lowered_bps_iso",
           "flavoured_un_nf_bps_chart", "UNNfLoweredTropicalMap"]

_ONE = LaurentPoly.one()


def flavoured_un_nf_bps_chart(N: int, Nf: int) -> BPSKAlgebra:
    """The **native flavoured** U(N)+N_f BPS chart: `pure_ade.UN_Nf(N, N_f)`
    with the full rank-(2N+N_f) lattice — ker B (the flavour Cartan of
    U(N_f)) kept, coefficient ring `R(U(1)^{N_f})`."""
    t = pa.UN_Nf(N, Nf)
    return BPSKAlgebra(pairing=[list(r) for r in t.B],
                       node_charges=[tuple(g) for g in t.nodes],
                       spec=[tuple(g) for g in t.spec],
                       verify="off").shorten_spec()


class UNNfLoweredTropicalMap:
    """Anchored + rigidified `(section, weight)` map between the
    Cartan-lowered abe shell and the central-collapsed flavoured BPS chart.

    Label shapes: abe `((g, ()), w)` with `g = (m, λ)` the lower-Kapustin
    gauge label; BPS `(γ, w)` with `γ` a section of the rank-(2N+N_f)
    lattice (flavour slots zero on this chart's `_sec_basis`).  The weight
    `w ∈ Z^{N_f−1}` is central on both sides and passes through the
    identity, so the map factorizes through a gauge-level cache
    `g ↦ γ_sec`; weight-charged product terms (monopole bubbling) join the
    derivation with their weight as part of the bucket key — an orientation
    mismatch would surface as a collision, not a silent mispairing."""

    def __init__(self, abe_low, bps_low, N: int, Nf: int):
        self._A, self._B = abe_low, bps_low
        self._N, self._Nf = int(N), int(Nf)
        self._w0 = (0,) * (self._Nf - 1)
        self._g: dict = {}
        self._inv: dict = {}

    # ----- label plumbing --------------------------------------------

    def _a_low(self, g, w=None):
        return ((tuple(g[0]), tuple(g[1])), ()), (self._w0 if w is None
                                                  else tuple(w))

    def _learn(self, g, gam):
        self._g[g] = gam
        self._inv[gam] = g

    def _interleave(self, ms, es):
        out = []
        for i in range(self._N):
            out.extend((ms[i], es[i]))
        return tuple(out) + (0,) * self._Nf

    # ----- forward: abe lowered label → bps lowered label ------------

    def gamma(self, label):
        (g, _su), w = label
        g = (tuple(g[0]), tuple(g[1]))
        return (self._gauge_gamma(g), tuple(w))

    def _gauge_gamma(self, g):
        N = self._N
        m, e = g
        hit = self._g.get(g)
        if hit is not None:
            return hit
        zero = (0,) * N
        if e == zero or len(set(m)) == 1:      # bare monopole / central+Wilson
            if e == zero:
                gam = self._interleave(sorted(m, reverse=True), zero)
            else:
                gam = self._interleave(m, sorted(e))
            self._learn(g, gam)
            return gam
        if m == zero:                          # Wilson anchor
            gam = self._interleave(zero, sorted(e))
            self._learn(g, gam)
            return gam

        # σ-transport first (ρ-images of anchored/cached gauge labels)
        def _cheap(s):
            return (s in self._g or s[0] == zero or s[1] == zero
                    or len(set(s[0])) == 1)
        for step, b_step in ((self._A.rho_inverse, self._B.rho),
                             (self._A.rho, self._B.rho_inverse)):
            try:
                (gs, _), ws = step(self._a_low(g))
            except (NotImplementedError, ValueError):
                continue
            gs = (tuple(gs[0]), tuple(gs[1]))
            if _cheap(gs):
                gam, wb = b_step((self._gauge_gamma(gs), tuple(ws)))
                if tuple(wb) != self._w0:
                    raise NotImplementedError(
                        f"UNNfLoweredTropicalMap: σ-transport of {g} lands "
                        f"on weight {wb} != 0 (weight-transport mismatch)")
                self._learn(g, gam)
                return gam

        # derived: match L_{m,0}·L_{0,e↓} against the BPS anchor product
        a, b = (m, zero), (zero, tuple(sorted(e, reverse=True)))
        ma = self._A.multiply(self._a_low(a), self._a_low(b))
        gb = self._B.multiply((self._gauge_gamma(a), self._w0),
                              (self._gauge_gamma(b), self._w0))
        by_c: dict = {}
        for (gam, wc), c in gb.terms.items():
            by_c.setdefault((tuple(wc), str(c)), []).append(gam)
        for lab, c in sorted(ma.terms.items()):
            (gc, _), wc = lab
            gc = (tuple(gc[0]), tuple(gc[1]))
            bucket = by_c.get((tuple(wc), str(c)), [])
            known = self._g.get(gc)
            if known is None and (gc[0] == zero or gc[1] == zero
                                  or len(set(gc[0])) == 1):
                # anchor-shaped term: closed form, no recursion into the
                # derivation — resolves (weight, coeff) collisions where one
                # colliding term is a Wilson/monopole/central anchor (first
                # seen at U(2)+N_f=3: the Wilson ((0,0),(3,3)) vs the
                # doubly-tropical monopole term in the same bucket)
                known = self._gauge_gamma(gc)
            if known is not None:
                if known in bucket:
                    bucket.remove(known)
                continue
            if len(bucket) != 1:
                raise NotImplementedError(
                    f"UNNfLoweredTropicalMap: coefficient collision deriving "
                    f"{gc} at weight {wc} from {a}·{b} (bucket {bucket})")
            self._learn(gc, bucket.pop(0))
        hit = self._g.get(g)
        if hit is None:
            raise NotImplementedError(
                f"UNNfLoweredTropicalMap: {g} absent from its anchor product")
        return hit

    # ----- inverse: bps lowered label → abe lowered label ------------

    def label(self, lab):
        gam, w = tuple(lab[0]), tuple(lab[1])
        hit = self._inv.get(gam)
        if hit is not None:
            return self._a_low(hit, w)
        N, Nf = self._N, self._Nf
        if any(gam[2 * N:]):
            raise NotImplementedError(
                f"UNNfLoweredTropicalMap: section {gam} has nonzero flavour "
                f"slots — not a section of this chart's _sec_basis")
        from itertools import permutations
        ms = tuple(sorted((gam[2 * i] for i in range(N)), reverse=True))
        es = [gam[2 * i + 1] for i in range(N)]
        seen = set()
        for perm in permutations(es):
            e = _levi_fold(ms, tuple(perm))
            if (ms, e) in seen:
                continue
            seen.add((ms, e))
            try:
                if self._gauge_gamma((ms, e)) == gam:
                    return self._a_low((ms, e), w)
            except NotImplementedError:
                continue
        raise NotImplementedError(
            f"UNNfLoweredTropicalMap: no label derived for {gam}")


def un_nf_lowered_bps_iso(N: int, Nf: int,
                          abe: UNNfKAlgebra | None = None) -> KAlgebraIso:
    """The N_f ≥ 2 iso between the Cartan-lowered presentations (see the
    module docstring).  `iso.source` / `iso.target` are the LOWERED
    algebras; the un-lowered pair is kept on `iso.abe_native` /
    `iso.bps_native`."""
    A0 = abe if abe is not None else UNNfKAlgebra(N, Nf)
    A = A0.lower_flavour(sun_to_cartan_hom(Nf))
    B0 = flavoured_un_nf_bps_chart(N, Nf)
    B = B0.lower_flavour(central_collapse_hom(Nf))
    T = UNNfLoweredTropicalMap(A, B, N, Nf)

    def forward(label):
        return Element({T.gamma(label): _ONE})

    def inverse(lab):
        return Element({T.label(lab): _ONE})

    iso = KAlgebraIso(A, B, forward, inverse,
                      name=f"UNNfKAlgebra({N},{Nf})↓Cartan ≅ "
                           f"BPS U({N})+Nf={Nf}↓central [anchored]")
    iso.tropical_map = T
    iso.abe_native = A0
    iso.bps_native = B0
    return iso


def un_nf_bps_iso(N: int, Nf: int = 1,
                  abe: UNNfKAlgebra | None = None) -> KAlgebraIso:
    """`UNNfKAlgebra(N, N_f) ≅ BPS U(N)+N_f` (anchored map; N_f ≥ 2 via the
    Cartan-lowered presentations — see the module docstring)."""
    if Nf != 1:
        return un_nf_lowered_bps_iso(N, Nf, abe)
    A = abe if abe is not None else UNNfKAlgebra(N, Nf)
    B = flavour_free_un_nf_bps(N, Nf)
    T = UNTropicalMap(A, B, N=N)

    def forward(label):
        return Element({T.gamma(label): _ONE})

    def inverse(g):
        return Element({T.label(g): _ONE})

    iso = KAlgebraIso(A, B, forward, inverse,
                      name=f"UNNfKAlgebra({N},{Nf}) ≅ BPS U({N})+Nf={Nf} "
                           f"[anchored]")
    iso.tropical_map = T
    return iso
