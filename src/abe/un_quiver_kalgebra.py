"""`UNQuiverKAlgebra` — the native linear-quiver `A_𝖖[U(N₁)×⋯×U(N_n)]`
on the `AbeKAlgebra` contract (Plan 30 T4 increment 3).

Labels are **tuples of per-node engine `(m, λ)` pairs** — per the D8/D8b
rulings the link U(1)'s are inside the gauge centres of their two nodes
(no flavour coordinate; the link μ-levels are internal torus bookkeeping,
specialized to 1 at the label boundary — a central specialization per
link, hence an algebra homomorphism: associativity, bar, and
orthonormality survive exactly as in the single-node N_f case).  Per-node
fundamentals would add SU(M_a) weights per D8; this increment covers the
pure-node chains (`Nf⃗ = 0` — the U(2)×U(1) / U(2)×U(2) / U(1)ⁿ family
certified in `tests/test_quiver_over_pure.py`).

The substrate is the group-general **`QuiverWRQTorus`** (D9/D10: the WRQTorus
selected by `torus_shape()` — over the product `u_n` datum).  The contract
triple:

  * `chart(label)` — the canonical as a `QuiverWRQTorus`: the certified
    `QuiverOverPure` flow image de-dressed onto the enriched atoms
    (`_urq_chart`, per-node registry residuals tensored per RG term, then the
    triangular de-dressing) **transported** onto the WRQ substrate (`vr_to_tr`
    over the product datum).  Cached.  *Provenance note*: this is the
    Route-A realisation (the flow IS the certified construction); the
    fully native product-and-peel build on the product lattice is the
    recorded open item (Plan 30 notes), exactly as `PureUNKAlgebra`
    predated its closed-form engine.
  * `decompose(x)` — level-ascending attribution: per base-level slice, the
    **joint recognizer falls out of the generic product-datum
    `recognize_leading`** (the product datum's block structure makes joint
    quiver canonicals into product-datum canonicals — no bespoke joint read),
    normalize by the image's own leading monomial, subtract the level-shifted
    chart, recurse; the link levels drop at the label boundary (D8b per link).
    Honest-fails off-scope.
  * `torus_shape()` = the U(N_i)-chain `TorusShape`.

Derived through the contract: multiply, ρ/ρ⁻¹, trace / inner product
(product-measure residue with the link M-factors; levels collapse),
W1 + `certify_canonical`.
"""
from __future__ import annotations

import os
import sys
from itertools import product as _iproduct

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from abe_kalgebra import AbeKAlgebra
from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing

__all__ = ["UNQuiverKAlgebra"]


class UNQuiverKAlgebra(AbeKAlgebra):
    """Native linear quiver of U(N_a) nodes on the abelianized contract
    (pure-node increment: no per-node fundamentals)."""

    def __init__(self, ranks, **flow_kwargs) -> None:
        from quiver_over_pure import QuiverOverPure
        self._ranks = tuple(int(N) for N in ranks)
        if any(x for x in flow_kwargs.get("Nf", ())):
            raise NotImplementedError(
                "UNQuiverKAlgebra: per-node fundamentals (SU(M_a) labels per "
                "D8) are the next increment")
        self._flow = QuiverOverPure(self._ranks, **flow_kwargs)
        self._pures = self._flow.pures()
        self._n = len(self._ranks)
        self._L = self._n - 1
        from quiver_wrq_torus import _offsets
        self._off = _offsets(self._ranks)
        self._charts: dict = {}        # URQ transport-source cache
        self._wrq_charts: dict = {}    # QuiverWRQTorus cache (the live substrate)
        self._lead_cache: dict = {}    # per-node image-lead monomials (decompose)

    # ----- KAlgebra basics -------------------------------------------

    def coefficient_ring(self):
        return TrivialZPlusRing()

    def identity(self):
        return tuple(P.identity() for P in self._pures)

    def _label_section_decompose(self, label):
        """**Aspirationally obsolete (Plan 32)** — the pure-node quiver is
        unflavoured (`TrivialZPlusRing`); superseded by the inherited (trivial)
        `r_label_decompose` from `AbeKAlgebra`.  Kept while `to_R_form` routes
        through it.  (Per-node SU(M_a) flavour — the next increment — will
        override `r_label_decompose` to peel the per-node weights.)"""
        return (label, self.coefficient_ring().one())

    # ----- the contract triple ---------------------------------------

    def torus_shape(self):
        from abe_kalgebra import TorusShape
        return TorusShape.from_ranks_nf(self._ranks, (0,) * self._n)

    def _embed_tensor(self, labs):
        """`⊗_a urqt(lab_a)` as `{flat atom: VRational}` (per-node registry
        residuals embedded and multiplied)."""
        from quiver_urq_torus import _offsets, _embed_vr
        from abelianized_torus import VRational
        off = _offsets(self._ranks)
        M = off[-1]
        parts = [self._pures[a].urqt(labs[a]).residuals()
                 for a in range(self._n)]
        out: dict = {}
        for combo in _iproduct(*[list(p.items()) for p in parts]):
            atom = tuple(x for (ma, _f) in combo for x in ma)
            vr = VRational.from_scalar(LaurentPoly({0: 1}), n=M)
            for a, (_ma, fa) in enumerate(combo):
                vr = (vr * _embed_vr(fa, off[a], M)).simplify()
            out[atom] = vr if atom not in out else (out[atom] + vr).simplify()
        return {a: f for a, f in out.items() if not f.is_zero()}

    def _urq_chart(self, label):
        """The canonical as a `QuiverURQTorus` (the URQ transport source) — the
        certified flow image de-dressed onto the enriched atoms (cached)."""
        key = tuple((tuple(g[0]), tuple(g[1])) for g in label)
        x = self._charts.get(key)
        if x is None:
            from quiver_urq_torus import QuiverURQTorus
            fam: dict = {}
            for (labs, k), c in self._flow.RG((key, (0,) * self._L)).terms.items():
                lp = c if isinstance(c, LaurentPoly) else c.expand(48)
                if hasattr(c, "expand"):
                    assert c.expand(48) == c.expand(56), \
                        "flow RG coefficient not finite-Laurent"
                add = self._embed_tensor(labs)
                dst = fam.setdefault(tuple(k), {})
                for atom, vr in add.items():
                    from abelianized_torus import VRational
                    t = (vr * VRational.from_scalar(
                        lp, n=sum(self._ranks))).simplify()
                    dst[atom] = t if atom not in dst else (
                        dst[atom] + t).simplify()
            fam = {k: {a: f for a, f in row.items() if not f.is_zero()}
                   for k, row in fam.items()}
            x = QuiverURQTorus.from_family(
                {k: row for k, row in fam.items() if row},
                self._ranks, (0,) * self._n)
            self._charts[key] = x
        return x

    def _transport(self, ch):
        """A `QuiverURQTorus` flow chart → the group-general `QuiverWRQTorus`
        over `self.torus().datum` (the product `u_n` datum).  The transport is
        an isomorphism (`tests/test_quiver_wrq_torus.py`)."""
        from quiver_wrq_torus import QuiverWRQTorus
        from matter_wrq_torus import vr_to_tr
        D = self.torus().datum
        return QuiverWRQTorus(self._ranks, (0,) * self._n, {
            m: {k: vr_to_tr(D, vr) for k, vr in row.items()}
            for m, row in ch.residuals().items()}, datum=D)

    def chart(self, label):
        """`L_label` as a `QuiverWRQTorus` — the certified `_urq_chart` flow
        image transported onto the WRQ substrate (Route-A provenance)."""
        key = tuple((tuple(g[0]), tuple(g[1])) for g in label)
        x = self._wrq_charts.get(key)
        if x is None:
            x = self._transport(self._urq_chart(label))
            self._wrq_charts[key] = x
        return x

    # ----- the joint registry peel (WRQ substrate) --------------------

    def _flat_to_pernode(self, m, e):
        """Split a flat product-datum leading `(m, e)` into per-node
        `((m_a, e_a), …)` engine labels by the rank offsets."""
        off = self._off
        return tuple((tuple(m[off[a]:off[a + 1]]), tuple(e[off[a]:off[a + 1]]))
                     for a in range(self._n))

    def _image_lead(self, pernode, m, e):
        """The image chart's (unit q-monomial) leading coefficient at the flat
        `(m, e)` — the joint recognizer falls out of the generic product-datum
        `recognize_leading` on the base μ-level slice (no bespoke joint read)."""
        key = (pernode, m, e)
        c0 = self._lead_cache.get(key)
        if c0 is None:
            fam = self.chart(pernode).to_family()
            k = min(fam, key=lambda kk: (sum(kk), kk))
            c0 = fam[k].recognize_leading().get((tuple(m), tuple(e)))
            if c0 is None or len(c0._coeffs) != 1:
                raise NotImplementedError(
                    f"UNQuiverKAlgebra: image lead of {pernode} not a monomial")
            self._lead_cache[key] = c0
        return c0

    def decompose(self, x) -> Element:
        """Level-ascending attribution on the WRQ substrate: at the global
        lowest μ-level every term is the base of some canonical (a
        decomposition has no target — unambiguous), read by the generic
        product-datum `recognize_leading`; normalize by the image's own leading
        monomial, subtract the full level-shifted chart, recurse.  Link levels
        drop at the label boundary (D8b per link)."""
        neg1 = LaurentPoly({0: -1})
        cur = x
        out: dict = {}
        for _ in range(512):
            fam = cur.to_family()
            if not fam:
                return Element({lab: C for lab, C in out.items()
                                if not C.is_zero()})
            k0 = min(fam, key=lambda k: (sum(k), k))
            rec = fam[k0].recognize_leading()
            if not rec:
                return Element({lab: C for lab, C in out.items()
                                if not C.is_zero()})
            for (m, e), C in rec.items():
                pernode = self._flat_to_pernode(m, e)
                (p0, z0), = self._image_lead(pernode, m, e)._coeffs.items()
                Ct = LaurentPoly({e0 - p0: z0 * c
                                  for e0, c in C._coeffs.items()})
                out[pernode] = out.get(pernode, LaurentPoly.zero()) + Ct
                cur = cur + self.chart(pernode)._scaled(Ct * neg1, k0)
        raise NotImplementedError(
            "UNQuiverKAlgebra.decompose: attribution did not terminate")

    # ----- flavour packaging (links collapse) --------------------------

    def _flavour_element(self, ring, lev, c):
        return c


if __name__ == "__main__":
    A = UNQuiverKAlgebra((2, 1))
    id2, id1 = ((0, 0), (0, 0)), ((0,), (0,))
    E1 = (((1, 0), (0, 0)), id1)
    F1 = (((0, -1), (0, 0)), id1)
    F2 = (id2, ((-1,), (0,)))
    print("U(2)×U(1) native on the contract; labels = per-node (m, λ)")
    print("  E1·F1 =", {str(k): str(v) for k, v in A.multiply(E1, F1).terms.items()})
    print("  F1·F2 =", {str(k): str(v) for k, v in A.multiply(F1, F2).terms.items()})
    print("  ρ(E1) =", A.rho(E1))
