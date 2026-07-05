"""`PureSUNKAlgebra` — pure SU(N) as an `AbeKAlgebra` on the group-general
**WRQTorus** substrate (`root_datum.su_n(N)`, ω-basis, `d = N−1`).

Pure SU(N) is pure U(N) restricted to the **trace-zero magnetic sublattice**
`{m⃗ : Σ mᵢ = 0}` with the central U(1)/det frozen (Plan 24's U(N)→SU(N)
microscope).  SU(N) has **no minuscule coweight**, so the canonicals cannot be
built from dressed minuscules as in the U(N) keystone; the minimal monopole is
the **adjoint** `m⃗ = (1,0,…,0,−1)`, which bubbles.

This realisation is **oracle-backed** (the "oracle now" leg, D9/D10 on WRQ): the
canonical charts come from the certified `PureUNKAlgebra(N)` build (restricted to
the trace-zero sector, det-collapsed) — the single structural input from U(N),
encapsulating the adjoint-monopole bubbling and the coupled Levi-block fibers —
**transported onto `WRQTorus(su_n(N))`** and everything else runs natively on the
SU torus: `multiply` / `ρ` / `ρ⁻¹` are the `AbeKAlgebra`-derived WRQTorus product +
read-back, `decompose` is the generic `wrq_torus` leading-orbit peel, and `trace` /
`inner_product` are the datum-general WRQTorus SU(N) Schur pairing.  (A fully
native SU(N) build — the general-N adjoint-monopole bubbling closed form — is the
tracked refinement; the generic `wrq_torus.build_canonical` is cone/minuscule-only
and does not reach the bubbling adjoint.)

The transport is the dual-basis U(N)→SU(N) restriction: magnetic cochars project
to the **coroot basis** (partial sums `cᵢ = Σ_{j<i} mⱼ`), electric weights to the
**fundamental-weight basis** (consecutive differences `ωᵢ = eᵢ − e_{i+1}`).  The
pure-SU(N) ρ agrees with the U(N) ρ (the trace-zero monopole is a U(N) monopole)
once `root_datum.su_n` supplies the U(N)-restricted `rho_sign` (the rank-≥2 fix).

Labels are the SU(N) 't Hooft–Wilson `(m, λ)`: `m` an anti-dominant **trace-zero**
magnetic cocharacter (`Σ mᵢ = 0`), `λ` a Levi-anti-dominant electric weight taken
**modulo det** (normalized to `min(λ) = 0`).  `det` is frozen, so pure SU(N) is
unflavoured (`TrivialZPlusRing`).
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from abe_kalgebra import AbeKAlgebra
from kalgebra import Element, Label
from laurent_poly import LaurentPoly
from zplus_ring import ZPlusRing, TrivialZPlusRing, RPowerSeries
from root_datum import su_n
from weyl_torus_ring import TorusLaurent, TorusRational
from wrq_torus import WRQTorus, decompose as _wrq_decompose
from pure_un_kalgebra import PureUNKAlgebra, default_rays


__all__ = ["PureSUNKAlgebra"]


def _vr_to_sun_tr(d, vr):
    """A trace-zero-U(N) `VRational` → `su_n(N)` `TorusRational`: electric
    weights project to the fundamental-weight basis `ωᵢ = eᵢ − e_{i+1}` (the
    denominator root `(v_i − 𝖖^M v_j)` → `(ω(e_j) − ω(e_i), M)` with the
    `v_i^{−mult}` monomial correction), collapsing the det direction."""
    vr = vr.simplify()
    N = d.dim + 1

    def P(a):
        return tuple(a[t] - a[t + 1] for t in range(N - 1))

    comp = [0] * (N - 1)
    den: dict = {}
    for (i, j, M), mult in vr.den.items():
        ei = tuple(1 if t == i else 0 for t in range(N))
        ej = tuple(1 if t == j else 0 for t in range(N))
        alpha = tuple(P(ej)[t] - P(ei)[t] for t in range(N - 1))
        den[(alpha, M)] = den.get((alpha, M), 0) + mult
        pei = P(ei)
        for t in range(N - 1):
            comp[t] -= mult * pei[t]
    num = TorusLaurent(d, {
        tuple(P(ve)[t] + comp[t] for t in range(N - 1)):
            LaurentPoly(dict(lp._coeffs))
        for ve, lp in vr.num._terms.items()})
    return TorusRational(d, num, den)


class PureSUNKAlgebra(AbeKAlgebra):
    """Pure SU(N) on `WRQTorus(su_n(N))`; charts from the `PureUNKAlgebra(N)`
    oracle, transported onto the SU torus."""

    _MAX_PEEL = 600

    def __init__(self, N: int, max_len: int = 4, K: int = 12) -> None:
        if N < 2:
            raise ValueError("PureSUNKAlgebra needs N >= 2")
        self._N = int(N)
        self.datum = su_n(self._N)
        self._u = PureUNKAlgebra(N, default_rays(N), max_len=max_len, K=K)
        self._chart_cache: dict = {}

    # ----- contract: coefficient ring / identity / shape --------------------
    def coefficient_ring(self) -> ZPlusRing:
        return TrivialZPlusRing()

    def identity(self) -> Label:
        return ((0,) * self._N, (0,) * self._N)

    def torus_shape(self):
        """One native SU(N) node, no fundamentals — the honest root-datum
        shape."""
        from abe_kalgebra import TorusShape
        return TorusShape.from_root_data((self.datum,))

    def _label_section_decompose(self, label: Label):
        return (label, self.coefficient_ring().one())

    # ----- label canonicalisation (Weyl-fold + det-collapse) ----------------
    def _su_label(self, m, e) -> Label:
        """The canonical SU(N) **u_n public label**: jointly Weyl-fold `(m, e)`
        to (anti-dominant m, Levi-anti-dominant e), then det-collapse
        `e ↦ e − min(e)·𝟙`."""
        N = self._N
        m = [int(x) for x in m]
        e = [int(x) for x in e]
        order = sorted(range(N), key=lambda i: m[i])      # m ascending (anti-dom)
        m_anti = tuple(m[i] for i in order)
        e1 = [e[i] for i in order]
        e_anti = list(e1)
        i = 0
        while i < N:
            j = i
            while j < N and m_anti[j] == m_anti[i]:
                j += 1
            e_anti[i:j] = sorted(e1[i:j])
            i = j
        c = min(e_anti)
        return (m_anti, tuple(x - c for x in e_anti))

    # ----- su_n internal label -> u_n public label --------------------------
    def _sun_to_un(self, cw) -> Label:
        """A `su_n(N)` label `(c, w)` (coroot magnetic, ω electric) → the u_n
        public label: lift the coroot cochar `c` to the trace-zero cochar
        `mₜ = c_{t+1} − cₜ`, the ω weight `w` to `eₜ = Σ_{s≥t} wₛ`, then
        canonicalise (`_su_label` folds + det-collapses)."""
        N = self._N
        c, w = cw
        cpad = (0,) + tuple(c) + (0,)
        m = tuple(cpad[t + 1] - cpad[t] for t in range(N))
        e = [0] * N
        for i in range(N - 2, -1, -1):
            e[i] = e[i + 1] + int(w[i])
        return self._su_label(m, tuple(e))

    # ----- chart (oracle-backed, transported to su_n WRQ) -------------------
    def _transport(self, urqt) -> WRQTorus:
        """A trace-zero-U(N) `URQTorus` chart → `WRQTorus(su_n(N))`: magnetic
        atoms to the coroot basis (partial sums), residuals via `_vr_to_sun_tr`."""
        N = self.datum.dim
        f = {}
        for atom, vr in urqt.residuals().items():
            catom = tuple(sum(atom[:i + 1]) for i in range(N))
            f[catom] = _vr_to_sun_tr(self.datum, vr)
        return WRQTorus(self.datum, f)

    def _oracle_urqt(self, label):
        """The `PureUNKAlgebra(N)` canonical (trace-zero sector) as a `URQTorus` —
        the u_n oracle image, before transport (engine public label = joint w0)."""
        m, e = self._su_label(*label)
        return self._u.urqt((tuple(reversed(m)), tuple(reversed(e))))

    def _urq_chart(self, label):
        """The u_n (`SUNRQTorus`) chart — the oracle image before transport;
        `VRational` residuals in the trace-zero-U(N) coordinates, for cross-checks
        and bubbling experiments that read the U(N) frame (`chart` is the su_n
        WRQTorus transport of this)."""
        from sun_rq_torus import SUNRQTorus
        return SUNRQTorus(dict(self._oracle_urqt(label).residuals()), self._N)

    def chart(self, label: Label) -> WRQTorus:
        """`L_label` as a `WRQTorus(su_n(N))`: the `PureUNKAlgebra(N)` canonical at
        the lifted U(N) label (trace-zero sector), transported onto the SU torus."""
        m, e = self._su_label(*label)
        key = (m, e)
        ch = self._chart_cache.get(key)
        if ch is None:
            ch = self._transport(self._oracle_urqt(label))
            self._chart_cache[key] = ch
        return ch

    # ----- decompose (generic WRQ leading-orbit peel + label map) -----------
    def decompose(self, x) -> Element:
        """Canonical-basis read on the WRQ substrate: the generic `wrq_torus`
        leading-orbit peel against the (oracle-built, transported) chart images,
        the su_n leading labels mapped back to the u_n public labels.  Honest-fails
        off scope."""
        out = _wrq_decompose(
            self.datum, x,
            build=lambda dd, c, w: self.chart(self._sun_to_un((c, w))))
        acc: dict = {}
        for (c, w), C in out.items():
            lab = self._sun_to_un((c, w))
            acc[lab] = acc.get(lab, LaurentPoly.zero()) + C
        return Element({lab: C for lab, C in acc.items() if not C.is_zero()})

    # ----- chart-level certificates -----------------------------------------
    def verify_chart_bar(self, a: Label) -> bool:
        """W1: `L_a`'s chart is bar-invariant (every residual q-palindromic)."""
        ch = self.chart(a)
        return ch.bar() == ch

    def certify_canonical(self, a: Label):
        """W1 + W2 (the executable KL acceptance) on the SU-torus chart, the
        su_n label mapped back to the u_n public label."""
        wf = self.chart(a).well_formed()
        return self._sun_to_un(wf) if wf is not False else False

    # ----- ρ / ρ⁻¹: read the image label via well_formed --------------------
    def _wf_label(self, img, what: str) -> Label:
        wf = img.well_formed()
        if wf is False:
            raise NotImplementedError(f"PureSUNKAlgebra.{what}: image not "
                                      f"well-formed")
        return self._sun_to_un(wf)

    def rho(self, a: Label) -> Label:
        return self._wf_label(self.chart(a).rho(), "rho")

    def rho_inverse(self, a: Label) -> Label:
        return self._wf_label(self.chart(a).rho_inverse(), "rho_inverse")

    # ----- trace / inner product (SU(N) measure) ----------------------------
    def trace(self, a: Label, K: int = 12) -> RPowerSeries:
        """`Tr(L_a)` — the datum-general WRQTorus SU(N) Schur trace of the chart.
        Negative q-exponents are passed through (not clipped): the contract's
        `verify_orthonormality` negative-window check must see them."""
        lp = self.chart(a).trace(K)
        return RPowerSeries(self.coefficient_ring(),
                            {e: c for e, c in lp._coeffs.items() if e <= K}, K)

    def inner_product(self, a: Label, b: Label, K: int = 12) -> RPowerSeries:
        """`I_{a,b} = Tr(ρ(L_a)·L_b)` — the §6b chart pairing with the SU(N)
        measure.  Negative q-exponents are passed through (see `trace`)."""
        lp = self.chart(a).inner(self.chart(b), K)
        return RPowerSeries(self.coefficient_ring(),
                            {e: c for e, c in lp._coeffs.items() if e <= K}, K)
