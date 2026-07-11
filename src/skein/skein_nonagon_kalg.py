"""`SkeinNonagonKAlg` — the [A₁,A₆] nonagon (M(2,9)) as a `KAlgebra`
contract instance realized by the stated-SKEIN engine on the PINNED
torus (the nonagon step of the pinned / unpin programme —
the odd-gon series pentagon → heptagon → nonagon).

The 9-gon is the [A₁,A₆] irregular puncture — RANK 6, the A₆
Argyres–Douglas chart, with THREE chord families (the p+2 shorts
D2_p, the p+3 mediums D3_p, the p+4 longs D4_p; 27 chords) and the
Y-system periodicity n = h+2 = 9.  It is ODD-marked, so it carries no
flavour and unpins CLEANLY: the pentagon story at rank 6.

Realization (measured, exact — all certificates re-asserted at
construction)
-----------------------------------------------------
Everything lives in `PinnedPolygon(9)` (the quantum torus over
Γ~ = Z^15 = 6 internal ⊕ 9 boundary, `pinned_polygon.py`); the side
quantum traces are single MONOMIALS (units — localization is free on
the pinned tier), with pins pin(s_r) = e_r + e_{r+1} and
pin(D_{k,p}) = e_p + e_{p+k}.  The pin-cancellation system
Σ_t w[t]·pin(s_t) = pin(D_{k,p}) is rank 9 with a UNIQUE INTEGER
solution per chord, rotation-covariant: w_{k,p}[t] = w_{k,0}[(t−p)%9],
with base words

    w2_0 = (0,0,1,−1,1,−1,1,−1,1)
    w3_0 = (1,−1,1,0,0,0,0,0,0)
    w4_0 = (0,0,0,0,1,−1,1,−1,1)

The dressed chords are

    T^(k)_p = lq^{x_k} · F(D_{k,p}) · NO_p(sides^{−w_{k,p}}),
    x = {2: −3, 3: −1, 4: −2}   (lq-exponents; q_chart = lq^{−2}),

where NO_p multiplies s_r^{−w[r]} sequentially for r = p, p+1, …,
p+8 (mod 9).  THE ROTATION-COVARIANT ORDER NO_p IS LOAD-BEARING — a
fixed ascending 0..8 order breaks the p-uniformity of the gauges —
and the gauge enters as lq^{x_k} (= lq^{−δ_k}; the sign matters).
Certified at construction (< 1 s): every base word solves its
pin-cancellation system exactly, all 27 dressed chords are pin-0
(the dressing IS the definition of the pinned canonicals: they live
on the internal sublattice Γ_A6), and every T commutes exactly with
every side monomial (243 checks) — the factorization
Pinned = Nonagon ⊗ T(sides) at the pinned level, no localization.

Labels are the intrinsic `A1A2kKAlg(3)` labels ``((a,i,e), …)``
(sorted tuples; () = identity).  Dictionary offset 0 for all three
families: D2_p ↔ (1,p), D3_p ↔ (2,p), D4_p ↔ (3,p).  A general
canonical forwards to lq^{−ORI·c} times the canonical-cone-ordered
product of its generators' dressed chords, c the accumulated
intrinsic phase (each incremental intrinsic product asserted
single-term) — the same cone packaging as `SkeinHeptagonKAlg`, but
with plain pinned-torus arithmetic instead of token bookkeeping.

`multiply(a, b)` is computed GENUINELY ON THE PINNED SIDE: the torus
product of the two forward images, peeled against the forward images
of the intrinsic product's labels (the intrinsic supplies only the
candidate LABEL SET; every peeled coefficient is measured in the
torus), and each peeled coefficient is ASSERTED to equal the
intrinsic coefficient VERBATIM under q → lq^{−2} — a standing guard,
never a fit.  NO per-label normalization (δ) table: δ' = 0
everywhere, STRONGER than the heptagon packaging.  Measured battery:
all 784 ordered generator-pair
products (28×28 over {identity} ∪ {(a,i)}), 1036 structure
constants — every label set matches `A1A2kKAlg(3)` exactly, every
peel ratio a single +1 lq-monomial, every coefficient verbatim.
Flagship relations (K[lbl] = forward image):

    T2_0·T2_1 = 1 + lq^2·K[(2,0)]
    T3_0·T3_1 = 1 + lq^2·K[(1,1),(3,0)]
    T4_0·T4_1 = 1 + lq^2·K[(2,1),(3,5)]
    T2_0·T3_1 = lq^{−2}·K[(1,2)] + K[(3,0)]
    T4_0·T4_3 = lq^2·K[(1,7)] + K[(2,0),(2,4)]

Transported primitives (pending the intrinsic stated-side
trace/half-index): `rho` / `rho_inverse` (the intrinsic label
rotation (a,i,e) → (a, i±1 mod 9, e) — the fan triangulation is not
rotation-invariant, so there is no geometric torus automorphism; same
status as pentagon/heptagon) and `trace` (the intrinsic two-layer
trace, landing on the four M(2,9) Andrews–Gordon characters).
`coefficient_ring` is Trivial (odd marks: no flavour).  No
`BPSKAlgebra` / F-solve anywhere on this path.

The `KAlgebraIso` to the intrinsic is `build_iso()` (identity label
correspondence; its endpoints ARE `self` and the intrinsic instance
this class was constructed on).  Registered as the ``'skein-pinned'``
realization of the nonagon object.
"""

from __future__ import annotations

from fractions import Fraction

import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import Element, KAlgebra
from laurent_poly import LaurentPoly

from half_laurent import HalfLaurent
from pinned_polygon import PinnedPolygon
from skein_pentagon_kalg import SkeinPentagonKAlg

Q = HalfLaurent.monomial
N = 9


def _f9(x):
    return x % 9


# unpin dressing base words (rotation-covariant:
# w_{k,p}[t] = _W0[k][(t-p) % 9]) — the unique integer solutions of
# the pin-cancellation systems, re-asserted at construction.
_W0 = {
    2: (0, 0, 1, -1, 1, -1, 1, -1, 1),
    3: (1, -1, 1, 0, 0, 0, 0, 0, 0),
    4: (0, 0, 0, 0, 1, -1, 1, -1, 1),
}

# unpin gauges (lq-exponents; q_chart = lq^{-2}): T = lq^{x_k}·B
# with x_k = -delta_k — the SIGN matters.
_X = {2: Fraction(-3), 3: Fraction(-1), 4: Fraction(-2)}


class SkeinNonagonKAlg(KAlgebra):
    """The pinned-torus stated-skein realization of the [A₁,A₆]
    nonagon, on the `A1A2kKAlg(3)` labels."""

    # q_int = lq^{ORI} (the bigon dictionary Q = q²; q_chart = lq^{-2})
    ORI = -2

    def __init__(self, intrinsic=None):
        if intrinsic is None:
            from a1a2k_kalg import A1A2kKAlg
            intrinsic = A1A2kKAlg(3)
        self._intr = intrinsic
        self._cd = intrinsic.cone_data()
        self.pp = PinnedPolygon(N)
        pp = self.pp
        self._sides = [pp.arc_F(r, _f9(r + 1), 1, 1) for r in range(N)]
        self._sides_inv = [pp.inverse_monomial(s) for s in self._sides]
        self._chords = {(k, p): pp.arc_F(p, _f9(p + k), 1, 1)
                        for k in (2, 3, 4) for p in range(N)}
        # geometric certificate 1: the hard-coded base words solve the
        # pin-cancellation system Σ_t w_{k,p}[t]·pin(s_t) = pin(D_{k,p})
        # EXACTLY, for all 27 chords (rotation covariance included).
        pin_s = [pp.pinning_vector(s) for s in self._sides]
        for (k, p), D in self._chords.items():
            want = pp.pinning_vector(D)
            acc = [0] * N
            for t in range(N):
                w = _W0[k][_f9(t - p)]
                for j in range(N):
                    acc[j] += w * pin_s[t][j]
            assert tuple(acc) == want, \
                f"pin-cancellation fails at D{k}_{p}"
        self._T = {(k, p): self._dress(k, p)
                   for k in (2, 3, 4) for p in range(N)}
        # geometric certificate 2: all 27 dressed chords are pin-0 —
        # the dressing cancels the pin exactly.
        for (k, p), T in self._T.items():
            assert pp.pinning_vector(T) == (0,) * N, \
                f"dressed chord T{k}_{p} not pin-0"
        # geometric certificate 3: every T commutes exactly with every
        # side monomial (243 checks) — Pinned = Nonagon ⊗ T(sides).
        for (k, p), T in self._T.items():
            for r in range(N):
                s = self._sides[r]
                assert pp.multiply(T, s) == pp.multiply(s, T), \
                    f"T{k}_{p} does not commute with side {r}"
        self._fwd_cache = {}
        self._mult_cache = {}

    # ---- the dressed chords ---------------------------------------------

    def _dress(self, k: int, p: int) -> dict:
        """T^(k)_p = lq^{x_k} · F(D_{k,p}) · NO_p(sides^{-w_{k,p}}),
        where NO_p multiplies s_r^{-w[r]} sequentially for r = p, p+1,
        …, p+8 (mod 9) — the rotation-covariant order (LOAD-BEARING:
        fixed ascending 0..8 breaks the p-uniformity of the gauges)."""
        pp = self.pp
        out = self._chords[(k, p)]
        for j in range(N):
            r = _f9(p + j)
            e = -_W0[k][j]
            for _ in range(abs(e)):
                out = pp.multiply(
                    out, self._sides[r] if e > 0 else self._sides_inv[r])
        return pp.scale(out, Q(_X[k]))

    def T(self, k: int, p: int) -> dict:
        """The dressed chord T^(k)_p, k ∈ {2,3,4} (a pin-0
        pinned-torus element)."""
        return self._T[(k, _f9(p))]

    def side(self, r: int) -> dict:
        """The pinned side monomial F(s_r(+,+))."""
        return self._sides[_f9(r)]

    # ---- forward map (canonical label -> pinned element) ------------------

    @staticmethod
    def _qexp1(poly):
        items = list(poly._coeffs.items())
        assert len(items) == 1 and items[0][1] == 1, poly
        return items[0][0]

    def forward(self, lbl) -> dict:
        """K(lbl): the pinned image of the intrinsic canonical label —
        lq^{-ORI·c} times the canonical-cone-ordered product of the
        generators' dressed chords ((a,i) -> T^(a+1)_i, offset 0), c
        the accumulated intrinsic phase (each incremental intrinsic
        product asserted single-term)."""
        lbl = tuple(lbl)
        if lbl in self._fwd_cache:
            return self._fwd_cache[lbl]
        pp = self.pp
        if lbl == self._intr.identity():
            out = pp.one()
        else:
            gens, powers = self._cd.to_cone_label(lbl)
            order = self._cd.canonical_cone_order(gens)
            acc = None
            img = None
            for g in order:
                glbl = self._cd.from_cone_label(frozenset({g}), {g: 1})
                gT = self._T[(g[0] + 1, g[1])]
                for _ in range(powers.get(g, 0)):
                    if acc is None:
                        acc = (glbl, Fraction(0))
                    else:
                        prod = self._intr.multiply(acc[0], glbl)
                        terms = dict(prod.terms)
                        assert len(terms) == 1, (lbl, g, terms)
                        nl = next(iter(terms))
                        acc = (nl, acc[1] + self._qexp1(terms[nl]))
                    img = gT if img is None else pp.multiply(img, gT)
            assert acc is not None and tuple(acc[0]) == lbl, (lbl, acc)
            out = pp.scale(img, Q(Fraction(-self.ORI * acc[1])))
        self._fwd_cache[lbl] = out
        return out

    # ---- KAlgebra primitives ----------------------------------------------

    def coefficient_ring(self):
        return self._intr.coefficient_ring()

    def identity(self):
        return self._intr.identity()

    def multiply(self, a, b) -> Element:
        a = tuple(a)
        b = tuple(b)
        key = (a, b)
        if key in self._mult_cache:
            return self._mult_cache[key]
        pp = self.pp
        prod = pp.multiply(self.forward(a), self.forward(b))
        # candidates from the intrinsic expansion (LABELS only; every
        # peeled coefficient below is measured in the pinned torus)
        hint = self._intr.multiply(a, b)
        hterms = {tuple(l): c for l, c in hint.terms.items()}
        pool = {l: self.forward(l) for l in hterms}
        rem = dict(prod)
        out = {}
        while rem:
            hit = None
            for u in sorted(rem):
                owners = [l for l, c in pool.items() if u in c]
                if len(owners) == 1:
                    hit = (u, owners[0])
                    break
            assert hit is not None, f"peel stuck for {a} * {b}"
            u, lam = hit
            r = SkeinPentagonKAlg._mono_ratio_hl(rem[u], pool[lam][u])
            assert r is not None, f"non-monomial peel at {lam}"
            e, sg = r
            # THE STANDING GUARD: the measured pinned coefficient must
            # equal the intrinsic coefficient VERBATIM under
            # q -> lq^{-2} (δ' = 0 everywhere — no normalization
            # table).  Asserted, never fitted.
            want = SkeinPentagonKAlg._chart_to_hl(hterms[lam])
            assert Q(e) * sg == want, (
                f"pinned/intrinsic coefficient mismatch at {lam} in "
                f"{a} * {b}: measured lq^{e}·({sg}), intrinsic "
                f"{hterms[lam]}")
            piece = pp.scale(pool.pop(lam), Q(e) * sg)
            rem = pp.add(rem, {v: -c for v, c in piece.items()})
            out[lam] = hterms[lam]
        assert not pool, (
            f"pinned/intrinsic label-set mismatch at {a} * {b}: "
            f"intrinsic labels {sorted(pool)} absent from the torus "
            f"product")
        el = Element(out)
        self._mult_cache[key] = el
        return el

    def rho(self, a):
        r = self._intr.rho(tuple(a))
        if hasattr(r, "terms"):
            ts = dict(r.terms)
            assert len(ts) == 1
            return next(iter(ts))
        return r

    def rho_inverse(self, a):
        r = self._intr.rho_inverse(tuple(a))
        if hasattr(r, "terms"):
            ts = dict(r.terms)
            assert len(ts) == 1
            return next(iter(ts))
        return r

    def trace(self, a, K: int = 20):
        return self._intr.trace(tuple(a), K)

    def _label_section_decompose(self, label):
        """The trivial section coordinate, matching `r_label_decompose`
        (odd marks: no flavour); `to_R_form` routes through it."""
        return (tuple(label), self.coefficient_ring().one())

    def r_label_decompose(self, label):
        """Trivial flavour-lift coordinate `(label, χ₀)`: the nonagon skein
        chart is unflavoured (odd marks).  Implemented directly
        (independent of `_label_section_decompose`;
        `forget()` / ring-hom promotion read this)."""
        return tuple(label), self.coefficient_ring().one_basis()

    def r_label_compose(self, section, r_basis_label):
        return tuple(section)

    # ---- the iso witness ----------------------------------------------------

    def build_iso(self):
        from kalgebra_iso import KAlgebraIso
        one = LaurentPoly.one()

        def _id(lbl):
            return Element({tuple(lbl): one})

        return KAlgebraIso(self, self._intr, _id, _id,
                           name="nonagon[skein-pinned→a1a2k]")
