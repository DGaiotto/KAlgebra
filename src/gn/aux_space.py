"""The Schur-quantization auxiliary space on the enriched rational quantum
torus — the vacuum-state formulation of the Schur pairing.

The definition (derived and certified 2026-07-24; record and derivation in
`aux_vacuum_pairing.md`, plan in the design record):

    |1>      =  (q^2;q^2)_oo^dim . prod_alpha (q^2 v^alpha; q^2)_oo . U_0
    L_a|1>   =  chart(a) . |1>            (plain torus LEFT multiplication)
    |1>.L_a  =  right multiplication      (= the rho-image left action)

    <x, y>   =  (1/|W|) sum_m  CT_v[ w_m(v) . x_m(1/v) . y_m(v) ]

with the per-sector weight, per root pair {alpha, -alpha}, s = |<m, alpha>|:

    s = 0 :   (1 - v^alpha)(1 - v^-alpha)          (undeformed Levi Vandermonde)
    s >= 1:   prod_{k=1}^{s-1} 1 / [(1 - q^{s-2k} v^alpha)(1 - q^{s-2k} v^-alpha)]

(equivalently, datum-generally:  w_m = prod_{alpha>0} (-q)^{-|<m,alpha>|}
(1 - q^{|<m,alpha>|} v^alpha)(1 - q^{|<m,alpha>|} v^{-alpha})
. R_{-m,m}(q^m v) — the charge-shifted full Vandermonde times the U_{-m}U_m
cocycle at the half-shifted argument; the prefactor exponent is the GNO
monopole dimension).  Then

    I_{a,b}  =  Tr(rho(a).b)  =  <L_a.1, L_b.1>        (Schur / Gram pairing)
    (a|Pi]   =  <L_a.1, Pi>                            (half-index)          [*]
    (Pi|Pi') =  <Pi, Pi'>                              (3d index)            [*]

✅ [*] RESOLVED ✅

    The two rows marked [*] ARE inner products, against the SOLVED pure-U(2)
    Neumann wavefunction (the design notes Sections 2k-2l; battery
    a probe in the source repository):

        f_m = (-q^-1 x^-1)^k / [(q^2;q^2)^rk prod_alpha (q^{2+|<m,alpha>|} v^alpha;q^2)],
        k = (m1-m2)/2, constant along the U(1) centre, f_odd = 0,
        f_0 = 1/|1>  (physical normalization -- the exact reciprocal of THIS
        module's vacuum residual; in main's [1]=1 convention f carries one
        extra 1/(q^2;q^2), and the 42/42 class-map match below is stated in
        that convention).

    (a|N] = I(a, f) reproduces main's rank-one class-map (L|N] on 42/42
    labels; (a|N(k)] = I(a, v^{km} f); and [N|N(k)] is the boundary
    flux sum verbatim (letter map x_flux = q, kappa(m) = k m, q^{-Delta_GNO} =
    the squared tail prefactor; exactly 1 at k >= h_dual, matching).
    The rk exponent is confirmed at U(3) against the affine Weyl-Kac
    Wilson row.  The historical "measured obstruction" (w_m = q^{2tr(m)}
    Z_gauge M^(0)_m vs |Delta_m|^2 = q^{(N-1)tr(m)} Z_gauge) is understood:
    one Delta_m per Delta-dressed slot of the superseded Pi_f candidate
    (|Delta_m|^2 in all across the pairing's two slots) -- a property of the
    dead candidate, not of the pairing.

    Still open as an ordinary Plan-23 item (stage gamma, H5): the B2 trade
    axiom -- DERIVING the boundary state from the axiomatics; the solved f
    is a verified representative anchored on the quotient ground truth.

NOT affected by the above: the Schur pairing row `I_{a,b}` (certified
`== inner_product` at U(1)/U(2)/U(3), the suite in the source repository), the weight
derivation below (verified by exact factor algebra at U(2), d <= 4 --
`aux_vacuum_pairing.md`), and the Abelian boundary-state usage.

Exact-arithmetic implementation notes:

* All computations are exact q-Laurent arithmetic through a trusted window
  (no numeric grids).  The vacuum is truncated at Pochhammer depth `KP`;
  results are exact through q-order `2*KP + 1`.
* Weight poles at level 0 (even differences) sit ON the unit contour; they
  are removed by EXACT DIVISION of the sector integrand — the states'
  vacuum-window zeros must cancel them, and division success is asserted
  (it is the admissibility/genuineness certificate).  Poles at level != 0
  sit off the contour and are expanded in the unique direction convergent
  on |v| = 1 (all-positive q-powers).
* The Cartan letters (q^2;q^2)^dim per state slot and the 1/|W| Haar factor
  are carried in the pairing normalization (mathematically part of |1> and
  the measure respectively); the final |W|-divisibility is asserted.

Datum generality: the weight only reads the root list and the pairings
<m, alpha> (dot products in matched cocharacter/weight coordinates).  Two
constructions:

* `AuxSpace(N, ...)` — the pure-U(N) form on the `URQTorus` substrate;
  `roots` optionally overrides the root list as ordered index pairs (i, j)
  meaning v^alpha = v_i/v_j (default: the N(N-1) roots of U(N)).
* `AuxSpace.from_datum(datum, KP)` — any `root_datum.RootDatum` on the
  group-general `WRQTorus` substrate (roots = exponent vectors in the
  datum's own weight coordinates, |W| from the datum).  Certified at SU(2)
  against `PureSU2KAlgebra` (whose basic monopole has <m, alpha> = 2 — the
  first level-0 gauge window, and the m=1 BUBBLING rational residual: its
  denominators are cleared exactly against the vacuum zeros by q-shifted
  binomial division, another genuineness certificate).

Scope / honest-fail: sector residuals must be polynomial after `simplify`
and the exact denominator division — a residual with uncancellable
denominators raises (off-scope, not guessed).

Matter: the cotangent-matter extension multiplies the weight by the derived
per-cell window Xi (c = <m, w_cell>: trivial for c >= 0, the mu-dressed
double window prod_{s<|c|}(1+q^{1-|c|+2s} mu^-1 v^-w)(1+q^{1-|c|+2s} mu v^w)
for c < 0) — derivation a probe in the source repository,
validated mu-refined at N=2*, U(1)xU(1) bifundamental, and N_f = 1, 2;
promotion of the matter surface into this class is staged.
"""

import math
from collections import defaultdict

from laurent_poly import LaurentPoly
from abelianized_torus import VRational, VLaurent
from urq_torus import URQTorus


# ---------------------------------------------------------------------------
# small exact q-Laurent helpers ({q-power: int} dicts)
# ---------------------------------------------------------------------------
def _lp_add(a, b):
    out = dict(a)
    for p, c in b.items():
        out[p] = out.get(p, 0) + c
        if not out[p]:
            del out[p]
    return out


def _lp_mul(a, b, cap):
    out = {}
    for p, c in a.items():
        for p2, c2 in b.items():
            if p + p2 <= cap:
                out[p + p2] = out.get(p + p2, 0) + c * c2
    return {p: c for p, c in out.items() if c}


def _qpoch_sq(cap):
    """(q^2; q^2)_oo as {qpow: int} through q^cap (finite Euler product)."""
    out = {0: 1}
    for k in range(1, cap // 2 + 2):
        out = _lp_mul(out, {0: 1, 2 * k: -1}, cap)
    return out


def _root_level(m, a):
    """`|⟨m, α⟩|` as a Python **int**, for use as a `range()` bound in the sector
    weight.

    Why the coercion: a cocharacter of a **non-simply-connected** form lives in
    the coweight lattice `P^∨ ⊋ Q^∨`, so its coordinates are `Fraction`s and this
    pairing comes back as a `Fraction` even when its value is a plain integer —
    which `range()` rejects on *type* alone.  Since `P^∨` is by definition the
    dual of the root lattice, `⟨α, m⟩ ∈ Z` for every root whenever `m ∈ P^∨`, so
    the coercion is always legitimate there.  This was the only thing standing
    between the vacuum-state pairing and the exotic forms (measured at `SO(5)`
    and `Sp(4)/Z₂`, whose fundamental coweights are integral in VALUE and
    `Fraction` in TYPE).  Same discipline as `wrq_torus._root_pairing_count`.

    Honest-fails when the value is genuinely non-integral: then `m ∉ P^∨`, the
    per-root window `∏_{k<s}` has no meaning, and no coercion can manufacture
    one."""
    v = abs(sum(x * y for x, y in zip(m, a)))
    iv = int(v)
    if iv != v:
        raise NotImplementedError(
            f"aux_space: |⟨m, α⟩| = {v} is not an integer at α={tuple(a)}, "
            f"m={tuple(m)} — the cocharacter is not in the coweight lattice "
            f"P^∨, so the per-root window has no meaning.")
    return iv


class AuxSpace:
    """The auxiliary space H_q[G] over the enriched rational quantum torus.

    Parameters
    ----------
    N : rank (number of `v` variables / diagonal magnetic components) for
        the pure-U(N) form.
    KP : vacuum Pochhammer truncation depth.  Pairings are exact through
         q-order `2*KP + 1`; `pair(..., K)` refuses larger K.
    roots : optional explicit root list as ordered index pairs (i, j)
         meaning v^alpha = v_i/v_j; default = the N(N-1) roots of U(N).
    datum : optional `root_datum.RootDatum` — the group-general form on
         the WRQTorus substrate (use `AuxSpace.from_datum`).
    """

    def __init__(self, N, KP=8, roots=None, datum=None):
        self.KP = KP
        self.datum = datum
        if datum is not None:
            self.dim = int(datum.dim)
            self.N = self.dim
            self.alphas = [tuple(a) for a in datum.roots()]
            self.weyl_order = len(list(datum.weyl_elements()))
        else:
            self.dim = self.N = N
            pairs = (list(roots) if roots is not None
                     else [(i, j) for i in range(N) for j in range(N)
                           if i != j])
            self.alphas = []
            for (i, j) in pairs:
                e = [0] * N
                e[i] += 1
                e[j] -= 1
                self.alphas.append(tuple(e))
            self.weyl_order = math.factorial(N)
        self._vacuum = None

    @classmethod
    def from_datum(cls, datum, KP=8):
        """Group-general construction from a `root_datum.RootDatum`."""
        return cls(int(datum.dim), KP=KP, datum=datum)

    # -- states -------------------------------------------------------------
    def vacuum(self):
        """|1> as a torus element: residual = prod_{alpha} prod_{k=1..KP}
        (1 - q^{2k} v^alpha) at magnetic charge 0.  (The Cartan letters
        (q^2;q^2)^dim per slot are carried by the pairing normalization.)"""
        if self._vacuum is None:
            zero = tuple([0] * self.dim)
            if self.datum is not None:
                from weyl_torus_ring import TorusLaurent, TorusRational
                from wrq_torus import WRQTorus
                acc = TorusLaurent(self.datum, {zero: LaurentPoly({0: 1})})
                for a in self.alphas:
                    for k in range(1, self.KP + 1):
                        acc = acc * TorusLaurent(self.datum, {
                            zero: LaurentPoly({0: 1}),
                            tuple(a): LaurentPoly({2 * k: -1})})
                self._vacuum = WRQTorus(self.datum, {
                    zero: TorusRational.from_laurent(acc)})
            else:
                acc = VRational.one(self.dim)
                for a in self.alphas:
                    for k in range(1, self.KP + 1):
                        acc = acc * VRational.from_vlaurent(VLaurent(
                            {zero: LaurentPoly({0: 1}),
                             tuple(a): LaurentPoly({2 * k: -1})},
                            n=self.dim))
                self._vacuum = URQTorus.from_f({zero: acc}, self.dim)
        return self._vacuum

    def state(self, chart):
        """L_a|1> for a chart element (torus element of the matching
        substrate): plain left multiplication."""
        return chart * self.vacuum()

    def act_left(self, chart, psi):
        """Left action of a line on a state:  chart . psi."""
        return chart * psi

    def act_right(self, psi, chart):
        """Right action of a line on a state:  psi . chart  (the rho-image
        left action; genuinely different from act_left — opposite shifts)."""
        return psi * chart

    # -- the pairing ----------------------------------------------------------
    def _monos(self, torus_elt, cap):
        """Sector residuals as {m: [(e_tuple, {qpow: int}), ...]}, pruned to
        q-order <= cap.  Rational residuals are simplified, then surviving
        denominator binomials (1 - q^k v^alpha) are divided out EXACTLY
        (the states' vacuum zeros must cancel them — bubbling residuals);
        an uncancellable denominator honest-fails."""
        out = {}
        for m, vr in torus_elt.residuals().items():
            if hasattr(vr, "simplify"):
                try:
                    vr = vr.simplify()
                except Exception:
                    pass
            den = dict(vr.den or {})
            num = getattr(vr.num, "_terms", None)
            if num is None:
                num = vr.num._t          # TorusLaurent storage
            lst = []
            for e, lp in num.items():
                cf = {p: c for p, c in lp._coeffs.items() if p <= cap}
                if cf:
                    lst.append((tuple(e), cf))
            for key, mult in den.items():
                if not (isinstance(key, tuple) and len(key) == 2
                        and isinstance(key[0], tuple)):
                    raise ValueError(
                        f"aux_space: sector {m} residual has denominators "
                        "in an unsupported format — off the certified "
                        "scope (honest-fail)")
                alpha, ksh = key
                for _ in range(mult):
                    lst = self._divide_qbinom(lst, tuple(alpha), ksh,
                                              self.dim, cap)
                    if lst is None:
                        raise ValueError(
                            f"aux_space: sector {m} denominator "
                            f"(1 - q^{ksh} v^{alpha}) does not divide the "
                            "residual — off-scope (honest-fail)")
            if lst:
                out[m] = lst
        return out

    def _sector_weight_parts(self, m, cap):
        """(numerator monomial list, level-0 divisor binomials, expanded
        off-contour inverse factors) of w_m."""
        zero = tuple([0] * self.dim)
        num = [(zero, {0: 1})]
        div0 = []          # (1 - v^alpha) binomials to divide out exactly
        expand = [(zero, {0: 1})]   # product of expanded inverses

        def times(acc, fac):
            out = {}
            for eA, cA in acc:
                for eB, cB in fac:
                    e = tuple(eA[t] + eB[t] for t in range(self.dim))
                    cc = _lp_mul(cA, cB, cap)
                    if cc:
                        out[e] = _lp_add(out.get(e, {}), cc)
            return [(e, c) for e, c in out.items() if c]

        seen = set()
        for a in self.alphas:
            aneg = tuple(-x for x in a)
            if aneg in seen:
                continue
            seen.add(a)
            s = _root_level(m, a)
            if s == 0:
                # Levi factor (1 - v^alpha)(1 - v^-alpha)
                num = times(num, [(zero, {0: 1}), (a, {0: -1})])
                num = times(num, [(zero, {0: 1}), (aneg, {0: -1})])
                continue
            for k in range(1, s):
                lev = s - 2 * k
                if lev == 0:
                    div0.append(a)
                    div0.append(aneg)
                else:
                    # 1/(1 - q^{lev} v^{+-alpha}) expanded convergently on
                    # |v|=1: lev > 0: sum_r q^{lev r} z^r; lev < 0 mirrored.
                    for base in (a, aneg):
                        fac = []
                        if lev > 0:
                            r = 0
                            while lev * r <= cap:
                                fac.append((tuple(r * x for x in base),
                                            {lev * r: 1}))
                                r += 1
                        else:
                            L = -lev
                            # -q^L z^{-1} * sum_r q^{L r} z^{-r}
                            r = 0
                            while L * (r + 1) <= cap:
                                fac.append((tuple(-(r + 1) * x for x in base),
                                            {L * (r + 1): -1}))
                                r += 1
                        expand = times(expand, fac)
        return num, div0, expand

    @staticmethod
    def _divide_binom(entries, d, N):
        """Exact division of a Laurent dict list by (1 - v^d): peel from the
        minimal <e, d> end.  Returns the quotient list or None."""
        return AuxSpace._divide_qbinom(entries, d, 0, N, None)

    @staticmethod
    def _divide_qbinom(entries, d, ksh, N, cap):
        """Exact division by (1 - q^{ksh} v^d): peel from the minimal
        <e, d> end, shifting the peeled coefficient by q^{ksh} at e + d.
        Returns the quotient list or None."""
        rem = {}
        for e, c in entries:
            rem[e] = _lp_add(rem.get(e, {}), c)
        rem = {e: c for e, c in rem.items() if c}
        quot = {}
        shift = {ksh: 1}
        for _ in range(2000000):
            if not rem:
                return [(e, c) for e, c in quot.items() if c]
            e = min(rem, key=lambda ee: sum(x * y for x, y in zip(ee, d)))
            c = rem.pop(e)
            quot[e] = _lp_add(quot.get(e, {}), c)
            e2 = tuple(e[t] + d[t] for t in range(N))
            c2 = (_lp_mul(c, shift, cap) if cap is not None else
                  {p + ksh: v for p, v in c.items()})
            rem[e2] = _lp_add(rem.get(e2, {}), c2)
            rem = {ee: cc for ee, cc in rem.items() if cc}
        return None

    def pair(self, x, y, K):
        """The bare Hermitian pairing <x, y> through q-order K, exact.

        `x`, `y` are torus elements (states: `state(chart)`, or any boundary
        vector).  Left slot conjugated (v -> 1/v).  Returns a LaurentPoly.
        """
        if K > 2 * self.KP + 1:
            raise ValueError(
                f"aux_space: K={K} exceeds the trusted window 2*KP+1="
                f"{2 * self.KP + 1}; raise KP")
        # Internal q-headroom: sectors with off-contour windows (interior
        # levels |s-2k| >= 1, i.e. some |<m,alpha>| >= 3) convolve expansion
        # towers whose products transiently exceed q^{K+2} before the CT
        # projects back down — measured at SU(2) s=4, where cap = K+2 left a
        # spurious q^8 while a deepened cap is exact.  No multiplication or
        # division step ever LOWERS q-order, so extra headroom only adds
        # true terms.
        extra = 0
        for elt in (x, y):
            for m in elt.residuals():
                e_m = 0
                seen = set()
                for a in self.alphas:
                    if tuple(-t for t in a) in seen:
                        continue
                    seen.add(a)
                    s = _root_level(m, a)
                    e_m += 2 * max(0, s - 2)
                extra = max(extra, e_m)
        cap = K + 2 + extra
        xm = self._monos(x, cap)
        ym = self._monos(y, cap)
        total = {}
        for m in set(xm) & set(ym):
            num, div0, expand = self._sector_weight_parts(m, cap)
            # integrand = conj(x_m) * y_m * num * expand, then exact-divide
            # the level-0 binomials, then CT (e = 0 coefficient).
            left = [(tuple(-t for t in e), c) for e, c in xm[m]]
            prod = {}
            for eA, cA in left:
                for eB, cB in ym[m]:
                    e = tuple(eA[t] + eB[t] for t in range(self.dim))
                    cc = _lp_mul(cA, cB, cap)
                    if cc:
                        prod[e] = _lp_add(prod.get(e, {}), cc)
            entries = [(e, c) for e, c in prod.items() if c]
            for fac in (num, expand):
                out = {}
                for eA, cA in entries:
                    for eB, cB in fac:
                        e = tuple(eA[t] + eB[t] for t in range(self.dim))
                        cc = _lp_mul(cA, cB, cap)
                        if cc:
                            out[e] = _lp_add(out.get(e, {}), cc)
                entries = [(e, c) for e, c in out.items() if c]
            for d in div0:
                q = self._divide_binom(entries, d, self.dim)
                if q is None:
                    raise ValueError(
                        f"aux_space: level-0 weight pole at sector {m} not "
                        "cancelled by the states' zeros — off-span/off-scope "
                        "input (honest-fail)")
                entries = q
            ct = {}
            zero = tuple([0] * self.dim)
            for e, c in entries:
                if e == zero:
                    ct = _lp_add(ct, c)
            total = _lp_add(total, ct)
        # Cartan letters (one (q^2;q^2)^dim per slot) and Haar 1/|W|:
        poch = _qpoch_sq(cap)
        cn = dict(poch)
        for _ in range(2 * self.dim - 1):
            cn = _lp_mul(cn, poch, cap)
        total = _lp_mul(total, cn, cap)
        fact = self.weyl_order
        out = {}
        for p, c in total.items():
            if p > K:
                continue
            if c % fact:
                raise ValueError(
                    f"aux_space: q^{p} coefficient {c} not divisible by |W| "
                    f"= {fact} — normalization inconsistency")
            out[p] = c // fact
        return LaurentPoly(out)

    # -- convenience ----------------------------------------------------------
    def schur_pairing(self, algebra, a, b, K):
        """I_{a,b} = <L_a.1, L_b.1> for canonical labels of a chart-bearing
        algebra (`algebra.urqt(label)` where available — the certified
        engine image — else the contract `algebra.chart(label)`)."""
        get = getattr(algebra, "urqt", None) or algebra.chart
        return self.pair(self.state(get(a)), self.state(get(b)), K)

    def half_index(self, algebra, a, boundary_state, K):
        """(a | Pi] = <L_a.1, Pi> — one dressed slot, one boundary vector."""
        get = getattr(algebra, "urqt", None) or algebra.chart   # WRQ charts since 2026-09-19 (the URQ keystone is in the source repository's archive)
        return self.pair(self.state(get(a)), boundary_state, K)
