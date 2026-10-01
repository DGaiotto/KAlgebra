"""The u(1)-gauged [A_1, D_{2k+2}] trace, transported EXACTLY down its RG flow.

The D-even counterpart of the gauged-A trace transport of the source
repository.  The gauge-tower control is this module's `__main__`:

    PYTHONPATH=$(ls -d src/* | paste -sd:) python3 src/cone/u1a1deven_trace_transport.py [k] [K]

Role since 2026-09-24: `U1A1DevenConeKAlgebra.trace` takes every SEED — an odd
curve, or a non-crossing pair of a +1 and a -1 curve, times a power of E —
from its closed form (`u1a1deven_seed_characters.seed_trace`) and reduces
every other label onto the seeds by the cone data's Layer-1 reduction, and its
pairing is multiply-then-trace; so this transport serves nothing on that route
(it is not even imported there).  It is the witness the closed forms and the
reduction are checked against (`seed_closed_forms=False` on the gauged class),
and its limits below (`max_word_degree`, `max_aux_terms`) bound only that
route.

The formula
-----------
The flow `U1A1DevenViaDoddRG(k)` goes to `A1DoddConeKAlg(k-1) (x) QT(Z^2)` with

    S = E_fq(X_{0,1} L) = sum_{n>=0} c_n C^n,   c_n = (-fq)^n / (fq^2; fq^2)_n,

C = (L^n, (0, n)), L the length-k doublet chord (k, 1, 0) of A1Dodd(k-1).  The
QT trace is delta_{charge,0} (fq^2;fq^2)^2, and the torus parts of rho(C^m),
RG(b) and C^n are (0,-m), (c0, c) and (0, n), so only c0 = 0 terms survive and
m = n + c:

    Tr_UV(b) = (fq^2;fq^2)^2 sum_{(x,(0,c)) in RG(b)} sum_n c_{n+c} c_n
               Tr_{A1Dodd}( rho(L^{n+c}) . x . L^n ),

every A1Dodd trace being its closed form (`A1DoddConeKAlg.trace`: the Layer-1
reduction, then `a1dodd_layer2`).  `trace(b, K)` evaluates this for a flow
label `b`; `trace_aux(X, K)` evaluates the same sum for an auxiliary element
`X` in place of `RG(b)` (so `trace(b, K) == trace_aux(RG(b), K)`).  Output:
`{fq-power: {SU(2) highest weight n: int}}` through fq^K, every order <= K
present if nonzero.

Construction (2026-09-24)
-------------------------
The transport builds what the formula needs itself: the auxiliary algebra
`A1DoddConeKAlg(k-1) (x) QT(Z^2)` with its A1Dodd factor (`_auxiliary`, built
as the flow builds them) and the closed-form parts `C^m = (L^m, (0, m))`,
`c_m` of `S` (`_C`, `_e_q_coeff`; `L = (k, 1, 0)`).  The flow
`U1A1DevenViaDoddRG(k)` is imported only by the flow-label entry point
`trace(b, K)`, which needs its `RG`, and by `_from_flow` (the controls with
another dressing chord).  `trace_aux` — the route of
`U1A1DevenConeKAlgebra.trace`, on the class's closed-form RG image — imports
no RG module.  The
parts equal the flow's and the traces are unchanged
(the source repository's test `test_flow_free_parts_equal_the_flow`).

The per-term skip (valid by A1Dodd orthonormality)
--------------------------------------------------
`Tr_{A1Dodd}(L_l) = delta_{l,1} + O(fq)`, so a summand `coef * L_l` of the
n-th product contributes at valuation >= `val(c_{n+c} c_n) + val(coef) +
[word(l) != ()]` (and `val(c_{n+c} c_n) = 2n + c`).  A summand whose bound
exceeds K is not traced; every traced summand is asked only to the depth its
coefficient needs.  The premise is CHECKED on every traced summand: an A1Dodd
trace below its bound raises `ValueError`.

The stopping rule (a measured hypothesis, with a guard)
-------------------------------------------------------
The sum over n in each c-sector stops after three consecutive terms whose
contribution through fq^K is zero (the U1A1Aodd transport's rule, section J).
This is NOT proven: the product-only bound above stays flat in n at k = 1
(`lb(n) = 1` at the seed (1,0,2), 2026-09-23), and the sharper bound
`val Tr_{A1Dodd}(L_w) >= deg w` is false (50 violations in 1250 cone monomials
at A1Dodd k = 1).  Measured: the term valuations grow by 2 per n at the seeds
(1,0,2) (k = 1) and (1,1,0) (k = 2) and by 4 at (1,0,0) (k = 2).

GUARD, part of the stopping condition: a sector stops only when, besides those
three terms, the valuations of its last four terms (fewer if the sector has
fewer) are strictly increasing; while they are not, the sum goes on to the next
term (which is added if it contributes after all), and the n-cap below raises
if the condition never holds.  A term's valuation is exact whenever its
contribution is nonzero through the depth evaluated (the coefficient series
leads with +-fq^{2n+c}, so the leading term cannot cancel); a term that
contributes nothing through fq^K is re-evaluated for this with a look-ahead of
16 orders, reusing its products.  A valuation not found within the look-ahead
counts as larger than every found one (two unfound ones are not compared); a
found valuation after an unfound one, or a non-increasing pair of found ones,
fails the condition (and a look-ahead summand longer than `max_word_degree`
lowers that term's reach, which can leave a comparison undecided and so also
delay the stop).  What the guard cannot see is a term after the stop: a sector
whose three terms beyond fq^K have strictly increasing valuations, and whose
next term would contribute after all, is cut without notice (a scripted
valuation sequence of exactly that shape is cut, 2026-09-23 review) — the
guard turns the non-monotone failures of the hypothesis into an extension or a
raise, not all of them.  Measured (2026-09-23): the condition holds at the
first stop on every single-generator label tried (|c| <= 3; 964 logged
evaluations: k = 1 at fq^16 and fq^32, k = 2 at fq^16 and fq^28, k = 3 at
fq^12 and fq^24); on some labels of two to four letters at k = 2, 3 it fails
there (at k = 3, fq^4: term valuations 11, 11, 13, the first two tie) and holds
a term or two later (it did so on hundreds of the calls behind the 2026-09-23
orthonormality runs); the extended results compared with the flow's own
windowed trace agree, 37/37 (22 at k = 2, 15 at k = 3), and a trace at fq^K
equals a fresh fq^(K+6) evaluation truncated to fq^K on every label of the
review's sample (single- and two-letter labels at k = 1, 2, 3, including the
k = 3 label above).  Until 2026-09-23 a failure raised at once.

Controls (the suite in the source repository): the gauge tower Tr(X_{0,1}^n),
n = 0..3, equals Creutzig's closed form (`exact_characters.deven_gauged_xn_qn`)
at k = 1, 2 through fq^24 and at k = 3 through fq^16 (and at k = 1, 2 through
fq^40, k = 3 through fq^32 in the deep run); the shorter dressing chord
(1,1,0) at k = 2 differs from it first at fq^6, and the singlet chord (1,0,0)
first at fq^4 (k = 1) and fq^6 (k = 2, 3); skipping disabled gives the same
series, skipping one order too early does not.

A1Dodd depth.  The A1Dodd closed forms are certified (coverage-checked
bootstrap, section I) through fq^40 at A1Dodd k = 1, 2 (D-even k = 2, 3) and
through fq^24 at A1Dodd k = 3 (D-even k = 4); A1Dodd k = 0 (D-even k = 1) is
not in that record, beyond the gauge-tower control above.  A long word asks
them for much more: its Layer-1 coefficients reach fq^-(about deg^2/2) at
k = 1 and fq^-(about 0.9 deg^2) at k = 2 (fq^-112 for 15 letters, fq^-169 for
14), so the seeds are evaluated that far beyond the requested depth and
cancel down to it (measured 2026-09-23).  An error at order d of a seed's
closed form would show in the word's trace from fq^(d + e) on, e the lowest
exponent of that seed's Layer-1 coefficient; below the word's floor it raises
at the per-term skip's check (below), so only an error within about K orders of
the far depth could pass unnoticed.

Honest fails (all `ValueError`, naming the label, K, c, n and the reason;
none of these truncates without notice — the stopping rule above is the one
hypothesis):
  * the n-sum of a sector runs more than `max_steps` terms (default
    K + |c| + 20) without stopping;
  * a summand that cannot be skipped has an A1Dodd word of more than
    `max_word_degree` letters.  The default depends on k (`_MAX_WORD_DEGREE`,
    set from measured memory, 2026-09-23; 12 at k >= 4, unmeasured); the
    measurements are in the table's comment below;
  * one auxiliary product has more than `max_aux_terms` terms (default
    200000), checked after each of the two products of a term;
  * an A1Dodd trace starting below the per-term skip's bound (the assumption
    the skip rests on);
  * a coefficient that is not an integer (none is expected; it is checked
    rather than floored).
A1Dodd's ladder-law guard (`a1dodd_layer2._to_irrep_guarded`, RuntimeError) is
propagated, never caught.

Memos: the A1Dodd trace per label and the transport result per flow label
both extend in depth (a shallower request reads a truncation); the Layer-2
closed forms have their own (`a1dodd_layer2._memo_read`).
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from laurent_poly import LaurentPoly
from habiro import HabiroElement
# The flow `U1A1DevenViaDoddRG` is NOT imported here: the transport builds the
# auxiliary algebra and the closed-form `S_RG` components itself
# (`_auxiliary`, `_e_q_coeff`), and imports the flow only for the flow-label
# entry point `trace(b, K)` (its `RG`) — see "Construction" in the docstring.


def _e_q_coeff(m: int) -> HabiroElement:
    """`c_m = (−𝖖)^m / (𝖖²;𝖖²)_m`, the m-th coefficient of `E_𝖖(X_{0,1}·L)`
    (the flow's `_e_q_coeff`, restated so that the transport needs no RG
    module; equal to it for m = 0..12 at k = 1..4)."""
    return HabiroElement(LaurentPoly({m: (-1) ** m}), {j: 1 for j in range(1, m + 1)})


def _auxiliary(k: int):
    """`(A1DoddConeKAlg(k − 1) ⊗ QT(Z²), A1DoddConeKAlg(k − 1))` — the flow's
    auxiliary algebra and its A1Dodd factor, built as the flow builds them."""
    from a1dodd_kalg import A1DoddConeKAlg
    from quantum_torus_kalgebra import QuantumTorusKAlg
    from tensor_kalgebra import TensorKAlgebra
    surv = A1DoddConeKAlg(k - 1)
    return TensorKAlgebra(surv, QuantumTorusKAlg([[0, 1], [-1, 0]])), surv


def _as_int(v, what):
    """`v` as an int; a non-integer raises `ValueError` (never floored)."""
    iv = int(v)
    if iv != v:
        raise ValueError(f"DevenTraceTransport: non-integer {what} coefficient {v!r}")
    return iv


def _irreps(c):
    """A coefficient (LaurentPoly or RLaurent over SU(2)) as {q: {irrep: int}}."""
    co = getattr(c, "coeffs", None)
    if co is not None:
        out = {}
        for q, r in co.items():
            d = {n: _as_int(v, "auxiliary") for n, v in r.terms.items()}
            d = {n: v for n, v in d.items() if v}
            if d:
                out[q] = d
        return out
    out = {}
    for q, v in c._coeffs.items():
        iv = _as_int(v, "auxiliary")
        if iv:
            out[q] = {0: iv}
    return out


def _fuse(d1, d2):
    out = {}
    for a, ca in d1.items():
        for b, cb in d2.items():
            for c in range(abs(a - b), a + b + 1, 2):
                out[c] = out.get(c, 0) + ca * cb
    return {c: v for c, v in out.items() if v}


def _add(tot, e, d, s=1):
    slot = tot.setdefault(e, {})
    for n, c in d.items():
        slot[n] = slot.get(n, 0) + s * c


def _clean(tot):
    out = {e: {n: v for n, v in d.items() if v} for e, d in tot.items()}
    return {e: d for e, d in out.items() if d}


def _truncate(series, K):
    return {q: dict(d) for q, d in series.items() if q <= K}


_GUARD_LOOKAHEAD = 16

# The default `max_word_degree` per k: the longest A1Dodd word a traced summand
# may have.  Set from measured memory (2026-09-23).  The workload it was set for
# — historical since 2026-09-24, when the generators (all seeds) moved to their
# closed forms and stopped reaching this transport — was every
# `A1DevenKAlg(k)` generator traced with no letter limit, one process; it needed
#   k = 1: 19 letters through fq^16 (8/8, peak RSS 39 MB),
#   k = 2: 16 letters through fq^12 (39/39, 99 MB),
#   k = 3: 13 letters through fq^8 (120/120, 145 MB);
# a single cone monomial x.L^n of the dressing chord L, fresh process each:
#   k = 1: 24 / 28 / 32 letters at fq^16: 42 / 70 / 125 MB (42 / 99 / 212 s),
#   k = 2: 17 / 19 letters at fq^12: 33 / 52 MB (69 / 136 s); 17 at fq^24: 33 MB,
#   k = 3: 15 / 17 letters at fq^12: 33 / 50 MB (199 / 380 s).
# Depth costs too: an earlier run at k = 2 met 527 MB on a 15-letter summand at
# fq^24.  Memory is the limit's reason; time
# grows fast with the length (the seeds are evaluated far beyond fq^K, above).
# k >= 4: 12, not measured.  Beyond the limit a summand raises `ValueError`; a
# larger `max_word_degree` may be passed.
_MAX_WORD_DEGREE = {1: 32, 2: 18, 3: 16}


def _default_word_degree(k: int) -> int:
    return _MAX_WORD_DEGREE.get(k, 12)


class DevenTraceTransport:
    """Exact `Tr_UV(b)` of the gauged `[A_1, D_{2k+2}]` by transport down the flow
    `U1A1DevenViaDoddRG(k)` (see the module docstring for the formula, the
    per-term skip, the stopping rule and its guard, and the honest fails)."""

    def __init__(self, k: int, *, max_word_degree=None,
                 max_aux_terms: int = 200_000, max_steps=None):
        self.k = k
        self.max_word_degree = (_default_word_degree(k) if max_word_degree is None
                                else max_word_degree)
        self.max_aux_terms = max_aux_terms
        self.max_steps = max_steps
        aux, surv = _auxiliary(k)
        self._set_parts(aux, surv, (k, 1, 0), flow=None)

    @classmethod
    def _from_flow(cls, flow, **kw):
        """A transport down a given flow instance (e.g. another dressing chord,
        for a negative control): its auxiliary algebra, A1Dodd factor and
        dressing chord."""
        self = cls.__new__(cls)
        self.k = flow.k
        mwd = kw.get("max_word_degree")
        self.max_word_degree = _default_word_degree(flow.k) if mwd is None else mwd
        self.max_aux_terms = kw.get("max_aux_terms", 200_000)
        self.max_steps = kw.get("max_steps")
        self._set_parts(flow.auxiliary(), flow._surv, flow._L, flow=flow)
        return self

    @property
    def F(self):
        """The flow `U1A1DevenViaDoddRG(k)`, built on first use: only the
        flow-label entry point `trace(b, K)` needs it (its `RG`).
        `trace_aux` never does."""
        if self._F is None:
            from u1a1deven_via_dodd_rg import U1A1DevenViaDoddRG
            self._F = U1A1DevenViaDoddRG(self.k)
        return self._F

    def _set_parts(self, aux, surv, chord, flow):
        self._F = flow
        self.aux = aux
        self.D = surv
        self._L = tuple(chord)             # the dressing chord L of S_RG
        self._trc = {}             # A1Dodd label -> (K, {q: {n: int}})
        self._flow_memo = {}       # flow label -> (K, result)
        self._C_cache = {}         # m -> (label of C^m, Habiro c_m)
        self._rhoC_cache = {}      # m -> rho(C^m) as an aux Element
        self._hc_cache = {}        # (m, n) -> (need, {q: int}) of c_m c_n
        self._skip_slack = 0       # controls only: > 0 skips less, < 0 more
        self._stats = {"traced": 0, "skipped": 0, "max_word": 0, "max_terms": 0,
                       "guard_extended": 0}

    # -- the two entry points ---------------------------------------------
    def trace(self, b, K: int, dry_stop: int = 3):
        """`{power: {SU(2) irrep: int}}` of `Tr_UV(b)` through fq^K, for a flow
        label `b = ((word, kappa), (c0, c1))`.  Memoised per label, extending
        in depth."""
        hit = self._flow_memo.get(b)
        if hit is not None and hit[0] >= K:
            return _truncate(hit[1], K)
        res = self._transport(self.F.RG(b), K, dry_stop, what=f"flow label {b!r}")
        self._flow_memo[b] = (K, res)
        return _truncate(res, K)

    def trace_aux(self, X, K: int, dry_stop: int = 3):
        """The same transport sum for an element `X` of the auxiliary algebra
        `A1DoddConeKAlg(k-1) (x) QT(Z^2)` in place of `RG(b)`:
        `trace(b, K) == trace_aux(self.F.RG(b), K)`.  Not memoised."""
        return self._transport(X, K, dry_stop, what="auxiliary element")

    # -- pieces ------------------------------------------------------------
    def _C(self, m):
        """`C^m = (L^m, (0, m))` with its coefficient `c_m`: the degree-`m`
        part of `S_RG = E_𝖖(X_{(0,1)}·L)` (the flow's `_s_rg_component((m,))`,
        in closed form)."""
        hit = self._C_cache.get(m)
        if hit is None:
            lab = (self.aux.identity() if m == 0
                   else ((((self._L, m),), 0), (0, m)))
            hit = self._C_cache[m] = (lab, _e_q_coeff(m))
        return hit

    def _rhoC(self, m):
        hit = self._rhoC_cache.get(m)
        if hit is None:
            lab, _ = self._C(m)
            hit = self._rhoC_cache[m] = self.aux.rho_element(
                Element({lab: LaurentPoly.one()}))
        return hit

    def _hc(self, m, n, need):
        """`c_m c_n` expanded through fq^need, as {q: int}."""
        hit = self._hc_cache.get((m, n))
        if hit is None or hit[0] < need:
            _, cm = self._C(m)
            _, cn = self._C(n)
            ser = {e: v for e, v in (cm * cn).expand(need)._coeffs.items()
                   if e <= need and v}
            hit = self._hc_cache[(m, n)] = (need, ser)
        return {e: v for e, v in hit[1].items() if e <= need}

    def _dtrace(self, lab, K):
        """A1Dodd trace of one label through fq^K (depth-extending memo)."""
        if K < 0:
            return {}
        hit = self._trc.get(lab)
        if hit is not None and hit[0] >= K:
            return {q: d for q, d in hit[1].items() if q <= K}
        r = self.D.trace(lab, K)
        t = {}
        for q, c in r.coeffs.items():
            d = {n: _as_int(v, "A1Dodd trace") for n, v in c.terms.items()}
            d = {n: v for n, v in d.items() if v}
            if d:
                t[q] = d
        self._trc[lab] = (K, t)
        return t

    def _product(self, Xc, n, c, what, K):
        """`rho(C^{n+c}) . Xc . C^n` in the auxiliary algebra, with the term cap."""
        lab_n, _ = self._C(n)
        prod = self.aux.multiply_elements(self._rhoC(n + c), Xc)
        self._check_terms(prod, what, K, c, n)
        prod = self.aux.multiply_elements(prod, Element({lab_n: LaurentPoly.one()}))
        self._check_terms(prod, what, K, c, n)
        return prod

    def _check_terms(self, prod, what, K, c, n):
        nt = len(prod.terms)
        self._stats["max_terms"] = max(self._stats["max_terms"], nt)
        if nt > self.max_aux_terms:
            raise ValueError(
                f"DevenTraceTransport(k={self.k}): {what} at fq^{K}, sector c={c}, "
                f"term n={n}: an auxiliary product has {nt} terms > max_aux_terms="
                f"{self.max_aux_terms} (seeds are served by their closed forms, "
                f"u1a1deven_seed_characters; a larger max_aux_terms may be passed)")

    def _term_tr(self, prod, val_h, depth, strict, what, K, c, n):
        """`sum coef * Tr_{A1Dodd}(l)` over the QT-neutral summands of `prod`,
        through fq^(depth - val_h), skipping summands bounded out of fq^depth.

        Returns `(tr, reach)`: `tr` is exact through fq^(reach - val_h).  With
        `strict`, a summand longer than `max_word_degree` raises; otherwise (the
        guard's look-ahead) it lowers `reach` below its bound instead."""
        tr = {}
        reach = depth
        over = []
        slack = self._skip_slack
        for (dl, qt), co in prod.terms.items():
            if tuple(qt) != (0, 0):
                continue
            cd = _irreps(co)
            if not cd:
                continue
            vco = min(cd)
            word = dl[0]
            floor = 1 if word else 0
            lb = val_h + vco + floor
            if lb > depth + slack:
                self._stats["skipped"] += 1
                continue
            deg = sum(e for _, e in word)
            if deg > self.max_word_degree:
                if strict:
                    raise ValueError(
                        f"DevenTraceTransport(k={self.k}): {what} at fq^{K}, sector "
                        f"c={c}, term n={n}: a summand that cannot be skipped has an "
                        f"A1Dodd word of {deg} letters > max_word_degree="
                        f"{self.max_word_degree} (seeds are served by their closed "
                        f"forms, u1a1deven_seed_characters; a larger max_word_degree "
                        f"may be passed)")
                over.append(lb)
                continue
            t = self._dtrace(dl, max(depth - val_h - vco, -1))
            self._stats["traced"] += 1
            self._stats["max_word"] = max(self._stats["max_word"], deg)
            if t and min(t) < floor:
                raise ValueError(
                    f"DevenTraceTransport(k={self.k}): {what} at fq^{K}, sector c={c}, "
                    f"term n={n}: Tr_A1Dodd({dl!r}) starts at fq^{min(t)} < {floor}; "
                    f"the per-term skip assumes A1Dodd orthonormality")
            for e1, r1 in cd.items():
                for e2, r2 in t.items():
                    if e1 + e2 <= depth - val_h:
                        _add(tr, e1 + e2, _fuse(r1, r2))
        if over:
            reach = min(reach, min(over) - 1)
        tr = _clean({e: d for e, d in tr.items() if e <= reach - val_h})
        return tr, reach

    def _contribution(self, tr, m, n, depth):
        """`c_m c_n * tr` through fq^depth."""
        out = {}
        if not tr:
            return out
        need = depth - min(tr)
        for e1, v1 in self._hc(m, n, need).items():
            for e2, d2 in tr.items():
                if e1 + e2 <= depth:
                    _add(out, e1 + e2, d2, v1)
        return _clean(out)

    # -- the sum -------------------------------------------------------------
    def _transport(self, X, K, dry_stop, what):
        by_c = {}
        for (dl, (c0, c1)), coef in X.terms.items():
            if c0 == 0:
                by_c.setdefault(c1, {})[(dl, (c0, c1))] = coef
        total = {}
        for c in sorted(by_c):
            Xc = Element(by_c[c])
            n0 = max(0, -c)
            cap = self.max_steps if self.max_steps is not None else K + abs(c) + 20
            n, dry = n0, 0
            recent = []                    # (n, val_h, prod, valuation or None)
            lookahead = {}                 # n -> (valuation or reach, found)
            while True:
                if n - n0 >= cap:
                    state = ""
                    if dry >= dry_stop:
                        _, seq = self._guard_status(recent, K, c, what, lookahead)
                        state = (f"; {dry} terms contribute nothing through fq^{K} "
                                 f"but the last term valuations {self._shown(seq)} "
                                 f"are not strictly increasing (the guard)")
                    raise ValueError(
                        f"DevenTraceTransport(k={self.k}): {what} at fq^{K}, sector "
                        f"c={c}: the n-sum reached n={n} ({cap} terms, the n-cap) "
                        f"without stopping{state}; the stopping rule is a measured "
                        f"hypothesis (module docstring)")
                prod = self._product(Xc, n, c, what, K)
                val_h = 2 * n + c
                tr, _ = self._term_tr(prod, val_h, K, True, what, K, c, n)
                v = None
                if tr:
                    for e, d in self._contribution(tr, n + c, n, K).items():
                        _add(total, e, d)
                    v = val_h + min(tr)
                recent.append((n, val_h, prod, v))
                recent = recent[-4:]
                dry = dry + 1 if v is None else 0
                n += 1
                if dry >= dry_stop:
                    ok, _ = self._guard_status(recent, K, c, what, lookahead)
                    if ok:
                        break
                    self._stats["guard_extended"] += 1
        pref = {0: 1}                                 # (fq^2; fq^2)_inf^2
        for j in range(1, K // 2 + 1):
            for _ in range(2):
                new = dict(pref)
                for e, v in pref.items():
                    if e + 2 * j <= K:
                        new[e + 2 * j] = new.get(e + 2 * j, 0) - v
                pref = new
        out = {}
        for e1, v1 in pref.items():
            for e2, d2 in total.items():
                if v1 and e1 + e2 <= K:
                    _add(out, e1 + e2, d2, v1)
        return _clean(out)

    def _guard_status(self, recent, K, c, what, lookahead=None):
        """`(ok, seq)`: whether the last (up to) four term valuations are
        strictly increasing; a term that contributes nothing through fq^K is
        re-evaluated with a look-ahead of `_GUARD_LOOKAHEAD` orders (memoised in
        `lookahead`, keyed by n)."""
        seq = []
        for (n, val_h, prod, v) in recent:
            if v is None:
                hit = lookahead.get(n) if lookahead is not None else None
                if hit is None:
                    depth = K + _GUARD_LOOKAHEAD
                    tr, reach = self._term_tr(prod, val_h, depth, False, what, K, c, n)
                    hit = (val_h + min(tr), True) if tr else (reach, False)
                    if lookahead is not None:
                        lookahead[n] = hit
                v = hit
            else:
                v = (v, True)
            seq.append((n, v))
        ok = True
        for (_, (a, fa)), (_, (b, fb)) in zip(seq, seq[1:]):
            if fa and fb:
                ok &= b > a
            elif fa:                       # found, then unfound: > its reach >= a
                ok &= b >= a
            elif fb:                       # a found valuation after an unfound one
                ok = False
            # two unfound valuations are not compared
        return ok, seq

    @staticmethod
    def _shown(seq):
        return [(n, v if f else f"> {v}") for n, (v, f) in seq]


_TRANSPORTS: dict = {}


def _shared_transport(k: int) -> DevenTraceTransport:
    """One transport per k per process (its memos are shared by every caller)."""
    T = _TRANSPORTS.get(k)
    if T is None:
        T = _TRANSPORTS[k] = DevenTraceTransport(k)
    return T


if __name__ == "__main__":
    # Positive control (moved here 2026-09-23 from the removed
    # a probe in the source repository): the gauge tower Tr(X01^n),
    # n = 0..3, against Creutzig's closed form.
    import time
    from exact_characters import deven_gauged_xn_qn
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    T = DevenTraceTransport(k)
    ok = True
    for n in range(0, 4):
        t0 = time.time()
        got = T.trace((T.D.identity(), (0, n)), K)
        want = _clean({q: dict(d) for q, d in deven_gauged_xn_qn(k, n, K).items() if q <= K})
        ok &= got == want
        print(f"[{time.time() - t0:5.1f}s] k={k}: Tr(X01^{n}) transport == closed form "
              f"through fq^{K}: {got == want}")
    print("control", "PASSED" if ok else "FAILED")
