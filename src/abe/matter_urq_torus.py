"""MatterURQTorus — the matter-enriched rational quantum torus for U(N)+N_f.

Elements are stored **natively in the f presentation**

    x  =  Σ_m  f_m(𝔮^m v) · U_m ,        f_m = Σ_k f_m^{(k)}·μ^k,

the same half-shift convention as the pure `URQTorus`.  `U_m` here is
the matter theory's own monopole atom — normalized by the square
root of the **full** Schur measure (gauge × matter), exactly as the pure
`U_m = u^m ψ_m(v)` absorbs the gauge half; the matter part of that
normalization (the finite, bar-centered rung products the absorption
theorem certified) is **internal
bookkeeping** of the dress/de-dress converters and is deliberately not
part of the API.

Verified structure this class encodes:

* canonicals on the anti-dominant cone have `f = f^{pure}` **verbatim**
  (`from_pure`) — flavour level 0 only;
* the product is the pure product per flavour level in bare coordinates
  (dress → pure `URQTorus` products → de-dress): the deformed cocycle
  without ever naming it;
* bar is the q-flip of each `f` component (the W1 frame); `well_formed_w1`
  is the first half of the O(𝔮) acceptance;
* the trace inserts the matter Schur factor `M(μ,v)` into the pure
  measure residue (the author's directive, certified in
  `un_nf_dressed_generators.matter_schur_trace`, retired 2026-09-19).

`rho` is the pure GTwist followed by the per-image-atom matter factor
(q-free top-monomial division + flavour star) — the per-atom extension of
the certified label closed form.

RETIRED ROUTES (2026-09-19).  `from_family` / `rg_image_from_pure` / `_recognize_chart`
/ `_build_inner` are the chart-assembly routes of the type-A U(N)+N_f layer
(`UNNfKAlgebra`, `UNNfViaRG`, `pure_via_n2star`) that was retired to the source repository's archive
that day; their chart recognition imports lazily from
`legacy.pure_un_closed_form` — the one lazy spine→legacy edge, kept only so
the archived tree still runs.  Nothing on the canonical
surface (`kalgebra_samples.Sqed1KAlg`, `aux_space`) reaches them.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laurent_poly import LaurentPoly
from urq_torus import URQTorus
from abelianized_torus import VRational, VLaurent


def _matter_rung_levels(atom, Nf: int, N: int) -> dict:
    """The matter normalization of the atom, per flavour level
    `{k_vec: VRational}` — the finite bar-centered rung expansion.
    Delegates to the quiver-torus core: the fundamental
    cells are the 1-node case of the cell table (`_rung_levels` with
    shape `((N,), (Nf,))`); keys coincide (flavour slots = the N_f
    slots)."""
    from quiver_urq_torus import _rung_levels
    return _rung_levels(atom, (N,), (Nf,))


_W_CACHE: dict = {}
_BUILD_STACK: set = set()       # cycle guard (cf. pure _CF_BUILDING)


def _matter_cocycle(m, mp, Nf: int, N: int) -> dict:
    """The matter cocycle `W_{m,m′} = T_{−m′}(Z_m)·T_m(Z_{m′}) / Z_{m+m′}`
    per flavour level `{r: VRational}` — the rule for multiplying atoms:

        U_m·U_{m′} = R̃_{m,m′}·W_{m,m′}·U_{m+m′}.

    Finite (net numerators).  Delegates to the quiver-torus core: `_quiver_cocycle` with shape `((N,), (Nf,))` — same
    triangular division, same cache, same rung-budget honest-fail."""
    from quiver_urq_torus import _quiver_cocycle
    return _quiver_cocycle(m, mp, (N,), (Nf,))


class MatterURQTorus:
    """Matter-enriched rational quantum torus element: native `f`-residual
    storage `{atom: {k_vec: VRational}}` (nonzero only); immutable value
    object."""

    __slots__ = ("_f", "_N", "_Nf")

    def __init__(self, f_residuals: dict, N: int, Nf: int):
        self._N = int(N)
        self._Nf = int(Nf)
        f: dict = {}
        for m, row in f_residuals.items():
            keep = {}
            for k, vr in row.items():
                vr = vr.simplify()
                if not vr.is_zero():
                    keep[tuple(k)] = vr
            if keep:
                f[tuple(m)] = keep
        self._f = f

    # ----- constructors / converters ---------------------------------
    @classmethod
    def zero(cls, N: int, Nf: int) -> "MatterURQTorus":
        return cls({}, N, Nf)

    @classmethod
    def rg_image_from_pure(cls, x: URQTorus, Nf: int) -> "MatterURQTorus":
        """**A statement about `RG(a)`, not a native constructor**: the absorption theorem says the flow image
        of an anti-dominant-cone canonical has the pure residuals
        verbatim on U atoms (`f = f^{pure}`, flavour level 0) — so this
        map *describes* `RG(L_{(m,e)})` for those labels.  It is a
        cross-check against the flow, and **dangerous as a constructor**:
        it imports a pure-theory element across theories on the strength
        of an empirically-scoped theorem (it is FALSE for mixed labels).
        Native canonicals come from `minuscule`/`wilson` + the native
        product + the build."""
        N = x._N
        zero = (0,) * Nf
        return cls({m: {zero: f} for m, f in x.residuals().items()}, N, Nf)

    @classmethod
    def minuscule(cls, m, e, N: int, Nf: int) -> "MatterURQTorus":
        """NATIVE dressed-minuscule generator: the pure-gauge leading-orbit
        closed FORMULA executed on U atoms — atoms = the joint Weyl orbit
        of `(m, e)` (`m` minuscule/regular or central), residual at atom
        `σ(m)` the monomial `∏_j v_j^{σ(e)_j}`; flavour level 0.  No
        pure-theory element is imported — this is the same closed form
        that defines the pure generator, applied in the U algebra."""
        from itertools import permutations
        m = tuple(int(x) for x in m)
        e = tuple(int(x) for x in e)
        zero = (0,) * Nf
        out: dict = {}
        seen = set()
        for sig in permutations(range(N)):
            am = tuple(m[i] for i in sig)
            ae = tuple(e[i] for i in sig)
            if (am, ae) in seen:
                continue
            seen.add((am, ae))
            mono = VRational.from_vlaurent(
                VLaurent({ae: LaurentPoly({0: 1})}, n=N))
            dst = out.setdefault(am, {})
            dst[zero] = mono if zero not in dst else (
                dst[zero] + mono).simplify()
        return cls(out, N, Nf)

    @classmethod
    def wilson(cls, e, N: int, Nf: int) -> "MatterURQTorus":
        """NATIVE Wilson generator: `χ_e(v)` at the zero atom (the pure
        closed formula on U; `U_0 = U_0`)."""
        from itertools import permutations
        e = tuple(int(x) for x in e)
        # χ_e as the monomial-symmetrized character via the Schur formula
        # for the (possibly det-shifted) dominant weight — reuse the
        # certified Kostka machinery
        sys.path.insert(0, os.path.join(_HERE, "implementations"))
        from urq_torus import _schur_monomials
        terms = {}
        for ve, z in _schur_monomials(tuple(sorted(e, reverse=True))).items():
            terms[ve] = LaurentPoly({0: z})
        vr = VRational.from_vlaurent(VLaurent(terms, n=N))
        return cls({(0,) * N: {(0,) * Nf: vr}}, N, Nf)

    @classmethod
    def from_family(cls, family: dict, N: int, Nf: int) -> "MatterURQTorus":
        """De-dress a per-flavour-level bare-chart family `{k: URQTorus}`
        into native `f` (triangular inversion of the atom normalization)."""
        atoms = sorted({a for U in family.values() for a in U.residuals()})
        levels = sorted(family, key=lambda k: (sum(k), k))
        out: dict = {}
        for atom in atoms:
            rungs = _matter_rung_levels(atom, Nf, N)
            ft: dict = {}
            for k in levels:
                b = family[k].residuals().get(atom)
                acc = b if b is not None else VRational.from_scalar(
                    LaurentPoly.zero(), n=N)
                for kp, fv in ft.items():
                    dk = tuple(x - y for x, y in zip(k, kp))
                    if any(x < 0 for x in dk):
                        continue
                    z = rungs.get(dk)
                    if z is None:
                        continue
                    acc = (acc + (fv * z * VRational.from_scalar(
                        LaurentPoly({0: -1}), n=N))).simplify()
                if not acc.is_zero():
                    ft[k] = acc
            if ft:
                out[atom] = ft
        return cls(out, N, Nf)

    def to_family(self) -> dict:
        """Dress back to per-level bare charts `{k: URQTorus}`."""
        N, Nf = self._N, self._Nf
        out: dict = {}
        for atom, row in self._f.items():
            rungs = _matter_rung_levels(atom, Nf, N)
            for k0, fv in row.items():
                for dk, z in rungs.items():
                    k = tuple(x + y for x, y in zip(k0, dk))
                    term = (fv * z).simplify()
                    if term.is_zero():
                        continue
                    add = URQTorus.from_f({atom: term}, N)
                    out[k] = add if k not in out else (out[k] + add)
            # drop levels that cancelled to zero
        return {k: U for k, U in out.items() if U.residuals()}

    # ----- reads -------------------------------------------------------
    def residuals(self) -> dict:
        return {m: dict(row) for m, row in self._f.items()}

    def support(self):
        return sorted(self._f)

    # ----- algebra -----------------------------------------------------
    def __add__(self, other: "MatterURQTorus") -> "MatterURQTorus":
        assert (self._N, self._Nf) == (other._N, other._Nf)
        out = {m: dict(row) for m, row in self._f.items()}
        for m, row in other._f.items():
            dst = out.setdefault(m, {})
            for k, vr in row.items():
                dst[k] = (dst[k] + vr).simplify() if k in dst else vr
        return MatterURQTorus(out, self._N, self._Nf)

    def _as_quiver(self):
        """The 1-node quiver-core view (same residual dict, shape
        `((N,), (Nf,))`) — delegation."""
        from quiver_urq_torus import QuiverURQTorus
        return QuiverURQTorus(self._f, (self._N,), (self._Nf,))

    @classmethod
    def _from_quiver(cls, q) -> "MatterURQTorus":
        (N,), (Nf,) = q.shape
        return cls(q.residuals(), N, Nf)

    def __mul__(self, other: "MatterURQTorus") -> "MatterURQTorus":
        """The native U-product — delegated to the quiver-torus core
        (the 1-node view; `R̃` gauge × finite matter `W` cocycles).
        (`_mul_via_dress` retains the dress→de-dress route as a
        cross-check.)"""
        assert (self._N, self._Nf) == (other._N, other._Nf)
        return MatterURQTorus._from_quiver(
            self._as_quiver() * other._as_quiver())

    def _mul_via_dress(self, other: "MatterURQTorus") -> "MatterURQTorus":
        """The dress → pure per-level products → de-dress route (the
        original implementation) — kept as the cross-check of the native
        cocycle product."""
        assert (self._N, self._Nf) == (other._N, other._Nf)
        N, Nf = self._N, self._Nf
        fam: dict = {}
        for k1, U1 in self.to_family().items():
            for k2, U2 in other.to_family().items():
                k = tuple(x + y for x, y in zip(k1, k2))
                prod = U1 * U2
                fam[k] = prod if k not in fam else (fam[k] + prod)
        fam = {k: U for k, U in fam.items() if U.residuals()}
        return MatterURQTorus.from_family(fam, N, Nf)

    def bar(self) -> "MatterURQTorus":
        """`𝔮 → 𝔮⁻¹` on each `f` component (`v`, atoms, μ fixed) — the W1
        frame (user-pinned: the O(𝔮) conditions live on `f`)."""
        from urq_torus import _qbar
        return MatterURQTorus(
            {m: {k: _qbar(vr, self._N) for k, vr in row.items()}
             for m, row in self._f.items()}, self._N, self._Nf)

    def well_formed_w1(self) -> bool:
        """W1 on `f`: every component bar-palindromic."""
        return self.bar() == self

    def well_formed(self):
        """The full acceptance ("peel until bubbling is O(𝔮)") —
        delegated to the quiver-torus core; the 1-node joint label
        unwraps to the matter `(label, k_vec)` form.  Same scope:
        in-span single-target build outputs."""
        wf = self._as_quiver().well_formed()
        if wf is False:
            return False
        (joint, k0) = wf
        return (joint[0], k0)

    def rho(self) -> "MatterURQTorus":
        """ρ — conjugation by the √(full measure): delegated to the
        quiver-torus core (block GTwist + Z-top star at 1 node = the
        certified matter closed form)."""
        return MatterURQTorus._from_quiver(self._as_quiver().rho())

    def rho_inverse(self) -> "MatterURQTorus":
        """ρ⁻¹ — the exact inverse (delegated; roundtrip-certified in
        the core)."""
        return MatterURQTorus._from_quiver(self._as_quiver().rho_inverse())

    # ----- trace (matter measure) --------------------------------------
    def trace(self, K: int = 8, W: int = 4, Kq_margin: int = 10) -> dict:
        """μ-refined Schur trace (matter factor in the measure residue) —
        delegated to the quiver-torus core (1-node product measure =
        the pure measure; flavour cells = the matter M-factor)."""
        return self._as_quiver().trace(K=K, W=W, Kq_margin=Kq_margin)

    # ----- recognition + the native build loop -------------------------
    def _recognize_chart(self):
        """Lowest-flavour-level leading read on the CHART (the pure
        engine's `_recognize_leading`, which ratios against `dom_atom`
        and so handles rational/bubbled residuals): returns
        `(k0, m_dom, e, cAmono, slice_chart)` — the dominant-frame label
        of the leading canonical at the lowest nonempty flavour level."""
        if not self._f:
            return None
        from legacy.pure_un_closed_form import _recognize_leading   # retired route (see module note)
        levels = sorted({k for row in self._f.values() for k in row},
                        key=lambda k: (sum(k), k))
        k0 = levels[0]
        slice_f = {m: row[k0] for m, row in self._f.items() if k0 in row}
        chart = URQTorus.from_f(slice_f, self._N).to_chart()
        mm, ee, m_dom, cAmono = _recognize_leading(chart, self._N)
        return k0, m_dom, ee, cAmono, chart

    def _level_chart(self, k):
        slice_f = {m: row[k] for m, row in self._f.items() if k in row}
        return URQTorus.from_f(slice_f, self._N).to_chart()

    def _scaled(self, C: LaurentPoly, lev) -> "MatterURQTorus":
        cv = VRational.from_scalar(C, n=self._N)
        return MatterURQTorus(
            {m: {tuple(a + b for a, b in zip(k, lev)): (vr * cv).simplify()
                 for k, vr in row.items()}
             for m, row in self._f.items()}, self._N, self._Nf)

    @classmethod
    def build(cls, m, e, N: int, Nf: int, _depth: int = 0,
              top: int = None) -> "MatterURQTorus":
        """The NATIVE build: canonicals constructed entirely inside the
        matter algebra — generators direct; everything else by seed
        product and **peeling until the bubbling is O(𝔮)** (the
        `well_formed` acceptance), with lower canonicals built
        recursively.  No solver, no flow, no pure-theory elements; the
        oracle is test-only.

        Labels in the engine frame (dominant `m`, Levi-dominant `e`)."""
        from itertools import permutations
        if _depth > 24:
            raise NotImplementedError(f"build({m},{e}): recursion depth")
        bkey = (tuple(m), tuple(e), N, Nf)
        if bkey in _BUILD_STACK:
            raise NotImplementedError(
                f"build({m},{e}): sibling-dressing cycle — same-magnetic "
                f"dressing siblings recognize each other; the unitriangular "
                f"joint solve (the pure engine's documented cf_build_full "
                f"gap (2)) is required.  The label-level engine "
                f"(matter_image) covers most of these shapes meanwhile.")
        m = tuple(int(x) for x in m)
        e = tuple(int(x) for x in e)
        lk_target = (tuple(reversed(m)), tuple(reversed(e)))
        # generators: minuscule / central magnetic — leading orbit only
        depths = [x for x in m if x != 0]
        is_minuscule = (set(depths) <= {1} or set(depths) <= {-1}
                        or len(set(m)) == 1)
        if is_minuscule:
            return cls.minuscule(m, e, N, Nf)
        _BUILD_STACK.add(bkey)
        try:
            try:
                return cls._build_inner(m, e, N, Nf, _depth, top)
            except NotImplementedError as ex:
                if "seed exhausted" not in str(ex) or not (
                        any(x < 0 for x in m) and any(x > 0 for x in m)):
                    raise
                # the negatives-first factorization's product lacks the
                # target (its content there arrived via an above-target
                # canonical's bubbling) — retry peeling the positive side
                return cls._build_inner(m, e, N, Nf, _depth, top,
                                        positive_first=True)
        finally:
            _BUILD_STACK.discard(bkey)

    @classmethod
    def _build_inner(cls, m, e, N, Nf, _depth, top, positive_first=False):
        lk_target = (tuple(reversed(m)), tuple(reversed(e)))
        # factorization: peel one anti-fundamental (or fundamental) step
        if any(x < 0 for x in m) and not (
                positive_first and any(x > 0 for x in m)):
            j = min(range(N), key=lambda i: m[i])         # deepest negative
            psi = tuple(-1 if i == j else 0 for i in range(N))
        else:
            j = max(range(N), key=lambda i: m[i])
            psi = tuple(1 if i == j else 0 for i in range(N))
        rest_m = tuple(x - y for x, y in zip(m, psi))
        g = cls.minuscule(psi, (0,) * N, N, Nf)
        rest = cls.build(rest_m, e, N, Nf, _depth + 1, top=top)
        seed = rest * g                     # E-part first: admixture
        seed_rev = g * rest                 # constants positive-q-supported
        # level-windowing: subtractions only ever need the parent's flavour
        # window; the budget strictly decreases through recursion — this is
        # what tames the Wilson-dressing ladders (cf. engine_image)
        own_max = max((sum(k) for row in seed._f.values() for k in row),
                      default=0)
        eff_top = own_max if top is None else min(int(top), own_max)
        # the true canonical fits inside the seed's level window (it appears
        # in the seed with coefficient 1), so windowing is exact — at the
        # top level too; subtractions beyond the window are artifacts

        def _win(x):
            return MatterURQTorus(
                {mm: {k: vr for k, vr in row.items() if sum(k) <= eff_top}
                 for mm, row in x._f.items()}, N, Nf)
        seed = _win(seed)
        seed_rev = _win(seed_rev)
        from urq_torus import _struct_const
        # phase 0: the seed's level-0 leading canonical can sit dominance-
        # ABOVE the target (cross-orbit atom sums, e.g. (1,0,-2) over
        # (1,-1,-1)) — peel those first, as the pure engine does
        for _ in range(64):
            rec0 = seed._recognize_chart()
            if rec0 is None:
                raise NotImplementedError(
                    f"build({m},{e}): seed exhausted in the top-peel")
            k0, m_dom0, e0, cAmono0, chart0 = rec0
            if (m_dom0, e0) == (m, e) and k0 == (0,) * Nf:
                break
            if k0 != (0,) * Nf:
                raise NotImplementedError(
                    f"build({m},{e}): seed leads at level {k0}")
            Lc = cls.build(m_dom0, e0, N, Nf, _depth + 1, top=eff_top)
            C = _struct_const(chart0, Lc._level_chart((0,) * Nf),
                              m_dom0, e0, cAmono0, N)
            Cb = LaurentPoly({-x: z for x, z in C._coeffs.items()})
            seed = seed + _win(Lc._scaled(C * LaurentPoly({0: -1}),
                                          (0,) * Nf))
            seed_rev = seed_rev + _win(Lc._scaled(
                Cb * LaurentPoly({0: -1}), (0,) * Nf))
        else:
            raise NotImplementedError(
                f"build({m},{e}): phase-0 top-peel did not reach the target")
        lead_t = cls.minuscule(m, e, N, Nf)
        c_t = _struct_const(chart0, lead_t._level_chart((0,) * Nf),
                            m_dom0, e0, cAmono0, N)
        if len(c_t._coeffs) != 1 or abs(next(iter(c_t._coeffs.values()))) != 1:
            raise NotImplementedError(
                f"build({m},{e}): seed leading coefficient {c_t} is not a "
                f"monomial unit (needs a different factorization)")
        (p, sgn), = c_t._coeffs.items()
        L = seed._scaled(LaurentPoly({-p: sgn}), (0,) * Nf)
        # peel: level-ascending; subtract recognized lower canonicals; a
        # level whose leading content is the target itself or its bubbling
        # tail (denom-mismatch on the dom_atom ratio) is done — the final
        # well_formed acceptance is the safety net for anything shadowed
        P = L + lead_t._scaled(LaurentPoly({0: -1}), (0,) * Nf)
        # the reverse-ordering twin (same centering), for tail-killing:
        # the target's content is identical in both orderings, admixture
        # constants are bar-conjugates — P − P_rev exposes shadowed seeds
        L_rev = seed_rev._scaled(LaurentPoly({-p: sgn}), (0,) * Nf)
        P_rev = L_rev + lead_t._scaled(LaurentPoly({0: -1}), (0,) * Nf)
        from legacy.pure_un_closed_form import _recognize_leading   # retired route (see module note)
        done_levels: set = set()
        for _ in range(256):
            live = sorted({k for row in P._f.values() for k in row
                           if k not in done_levels},
                          key=lambda k: (sum(k), k))
            if not live:
                break
            kp = live[0]
            chartP = P._level_chart(kp)
            try:
                mm_, ee_, m_dom_, cAmono_ = _recognize_leading(chartP, N)
                tail_or_target = (kp == (0,) * Nf and (m_dom_, ee_) == (m, e))
            except RuntimeError:
                tail_or_target = True                      # bubbling tail
            if not tail_or_target:
                Lc = cls.build(m_dom_, ee_, N, Nf, _depth + 1,
                               top=eff_top - sum(kp))
                C = _struct_const(chartP, Lc._level_chart((0,) * Nf),
                                  m_dom_, ee_, cAmono_, N)
                sub = _win(Lc._scaled(C * LaurentPoly({0: -1}), kp))
                Cbar = LaurentPoly({-x: z for x, z in C._coeffs.items()})
                L = L + sub
                P = P + sub
                P_rev = P_rev + _win(
                    Lc._scaled(Cbar * LaurentPoly({0: -1}), kp))
                continue
            # tail (possibly shadowing seeds at shared atoms): two-ordering
            # difference kills the target content; recognize seeds on it
            D = P + P_rev._scaled(LaurentPoly({0: -1}), (0,) * Nf)
            Dlvl = {mm: row[kp] for mm, row in D._f.items() if kp in row}
            if not Dlvl:
                done_levels.add(kp)                        # pure tail, done
                continue
            chartD = URQTorus.from_f(Dlvl, N).to_chart()
            try:
                mmD, eeD, m_domD, cAmonoD = _recognize_leading(chartD, N)
            except RuntimeError:
                raise NotImplementedError(
                    f"build({m},{e}): unattributable shared-atom content "
                    f"at level {kp}")
            Lc = cls.build(m_domD, eeD, N, Nf, _depth + 1,
                           top=eff_top - sum(kp))
            diffC = _struct_const(chartD, Lc._level_chart((0,) * Nf),
                                  m_domD, eeD, cAmonoD, N)
            C1 = LaurentPoly({x: z for x, z in diffC._coeffs.items() if x > 0})
            if C1.is_zero():
                done_levels.add(kp)         # palindromic constant: the
                continue                    # wf-driven repair loop's job
            C2 = LaurentPoly({-x: z for x, z in C1._coeffs.items()})
            L = L + _win(Lc._scaled(C1 * LaurentPoly({0: -1}), kp))
            P = P + _win(Lc._scaled(C1 * LaurentPoly({0: -1}), kp))
            P_rev = P_rev + _win(Lc._scaled(C2 * LaurentPoly({0: -1}), kp))
        else:
            raise NotImplementedError(f"build({m},{e}): peel did not finish")
        if top is not None:
            return L                        # windowed inner build: the
        #                                     parent's acceptance gates it
        # wf-driven repair: while the q-extreme is dominated by a non-target
        # canonical, subtract it with its struct-constant — this IS the
        # W2-determined palindromic part of the shared-label constants
        for _ in range(16):
            wf = L.well_formed()
            if wf == (lk_target, (0,) * Nf):
                return L
            if wf is False or wf is None:
                break
            (lk_bad, k_bad) = wf
            mb = tuple(reversed(lk_bad[0]))
            eb = tuple(reversed(lk_bad[1]))
            if (mb, eb) == (m, e) and k_bad == (0,) * Nf:
                break
            Lc = cls.build(mb, eb, N, Nf, _depth + 1,
                           top=max(eff_top - sum(k_bad), 0))
            chartL = L._level_chart(k_bad)
            from legacy.pure_un_closed_form import _recognize_leading as _rl   # retired route (see module note)
            try:
                _, eeL, m_domL, cAmonoL = _rl(chartL, N)
            except RuntimeError:
                break
            C = _struct_const(chartL, Lc._level_chart((0,) * Nf),
                              m_domL, eeL, cAmonoL, N)
            L = L + _win(Lc._scaled(C * LaurentPoly({0: -1}), k_bad))
        wf = L.well_formed()
        if wf != (lk_target, (0,) * Nf):
            raise NotImplementedError(
                f"build({m},{e}): acceptance failed — well_formed gave "
                f"{wf}, expected {(lk_target, (0,) * Nf)}")
        return L

    # ----- equality ----------------------------------------------------
    def __eq__(self, other) -> bool:
        if not isinstance(other, MatterURQTorus):
            return NotImplemented
        if (self._N, self._Nf) != (other._N, other._Nf):
            return False
        keys = set(self._f) | set(other._f)
        for m in keys:
            r1 = self._f.get(m, {})
            r2 = other._f.get(m, {})
            for k in set(r1) | set(r2):
                a = r1.get(k)
                b = r2.get(k)
                if a is None or b is None:
                    return False
                if not (URQTorus.from_f({m: a}, self._N)
                        == URQTorus.from_f({m: b}, self._N)):
                    return False
        return True

    def __repr__(self) -> str:
        n_terms = sum(len(r) for r in self._f.values())
        return (f"MatterURQTorus(N={self._N}, Nf={self._Nf}, "
                f"atoms={len(self._f)}, components={n_terms})")
