"""`SkeinOddPolygonKAlg` — the general odd `(2k+3)`-gon `[A₁, A_{2k}]`
as a `KAlgebra` realized by the stated-SKEIN engine on the PINNED torus
(the general-n step of the pinned / unpin programme — the generic-n
unpin frontier, odd half).

This generalizes `SkeinNonagonKAlg` (n = 9) to every odd n = 2k+3 ≥ 5.
The two per-n data that were previously *fitted per polygon* are now
**derived at construction**:

* **Unpin dressing words — closed form.**  On the pinned tier the
  pin-cancellation system `Σ_t w[t]·pin(s_t) = pin(D_{j,p})`
  (`pin(s_t) = e_t + e_{t+1}`, `pin(D_{j,p}) = e_p + e_{p+j}`) is the
  recurrence `w[r] + w[r-1] = δ_{r,0} + δ_{r,j}` on `Z/n`, which for
  odd `n` has the **unique** solution

      w_{j,0}[r] = (−1)^r on [0, j),  0 elsewhere     (j odd)
      w_{j,0}[r] = (−1)^r on [j, n),  0 elsewhere     (j even)

  rotated per chord (`w_{j,p}[t] = w_{j,0}[(t−p) mod n]`).  At n = 9
  this reproduces the recorded `SkeinNonagonKAlg._W0` verbatim; the
  solution property is re-asserted at construction for every chord.

* **Unpin gauges — meson-calibrated.**  The per-family gauge
  `T = lq^{x_j}·B` is fixed by the *unit side-resolution term* of the
  family meson: for adjacent crossing chords the Kauffman resolution of
  `D_{j,p}·D_{j,p+1}` contains the two-sides smoothing, which the
  dressing cancels to the identity charge — its coefficient is
  normalized to exactly 1 (the intrinsic mesons are `1 + q·(…)`), i.e.
  `x_j = −e/2` where `lq^e` is the measured identity coefficient of the
  bare meson `B_{j,0}·B_{j,1}`.  One scalar per family, derived from a
  single designated product, then **asserted on every product** by the
  standing guard — the same epistemic status as the heptagon's one-time
  δ-calibration, but construction-cheap.  At n = 9 the calibration
  reproduces `SkeinNonagonKAlg._X = {2:−3, 3:−1, 4:−2}` exactly
  (regression-pinned in the tests); the measured general pattern is
  `x_j = −(j−1)/2` (odd spans) / `−(n−j−1)/2` (even spans).

Everything else is the nonagon construction verbatim, at general n:
`PinnedPolygon(n)` (the quantum torus over `Γ~ = Z^{n−3} ⊕ Z^n`), side
quantum traces single-monomial units, the dressed chords pin-0 (the
dressing IS the definition of the pinned canonicals — they live on the
internal sublattice `Γ_{A_{2k}}`), every `T` commuting exactly with
every side monomial (the factorization `Pinned = A1A2k ⊗ T(sides)`),
the rotation-covariant dressing order `NO_p` (LOAD-BEARING — a fixed
ascending order breaks the p-uniformity of the gauges).

Labels are the intrinsic `A1A2kKAlg(k)` labels `((a,i,e), …)` (sorted
tuples; `()` = identity); the dictionary is offset-0 in every family —
`(a, i) ↔ D_{a+1, i}` — pinned by `A1A2kKAlg`'s own natural labeling
("L_{a,j} has chord endpoints (j, j+a+1)").  `multiply(a, b)` is
computed GENUINELY ON THE PINNED SIDE: the torus product of the two
forward images, peeled against the forward images of the intrinsic
product's labels (the intrinsic supplies only the candidate LABEL SET),
and each peeled coefficient is ASSERTED to equal the intrinsic
coefficient VERBATIM under `q → lq^{−2}` — a standing guard, never a
fit; δ ≡ 0 everywhere (no per-label normalization table).

Transported primitives (pending the intrinsic stated-side trace):
`rho`/`rho_inverse` (the intrinsic label rotation) and `trace` (the
intrinsic two-layer closed form, the M(2, 2k+3) Andrews–Gordon
characters).  `coefficient_ring` is Trivial (odd marks: no flavour).
No `BPSKAlgebra` / F-solve anywhere on this path.

Validated: n = 5, 7 full generator-pair sweeps; n = 9 word/gauge
regression vs
`SkeinNonagonKAlg` + sweep; n = 11 ([A₁, A₈] — the first NEW odd
instance) full 45×45 generator sweep.
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


def closed_form_word(n: int, j: int) -> tuple:
    """The unique solution of `w[r] + w[r-1] = δ_{r,0} + δ_{r,j}` on
    `Z/n` (n odd): the alternating block on `[0, j)` for odd span j, on
    `[j, n)` for even span j."""
    w = [0] * n
    if j % 2 == 1:
        for r in range(j):
            w[r] = (-1) ** r
    else:
        for r in range(j, n):
            w[r] = (-1) ** r
    return tuple(w)


class SkeinOddPolygonKAlg(KAlgebra):
    """The pinned-torus stated-skein realization of the odd
    `(2k+3)`-gon `[A₁, A_{2k}]`, on the `A1A2kKAlg(k)` labels — the
    generic-n unpin engine (words closed-form, gauges
    meson-calibrated, standing guard verbatim)."""

    # q_int = lq^{ORI} (the bigon dictionary Q = q²; q_chart = lq^{-2})
    ORI = -2

    def __init__(self, k: int = None, intrinsic=None):
        if intrinsic is None:
            if k is None:
                raise ValueError("need k or an A1A2kKAlg intrinsic")
            from a1a2k_kalg import A1A2kKAlg
            intrinsic = A1A2kKAlg(k)
        self.k = intrinsic.k
        n = 2 * self.k + 3
        self.n = n
        self._intr = intrinsic
        self._cd = intrinsic.cone_data()
        self.pp = PinnedPolygon(n)
        pp = self.pp
        self._sides = [pp.arc_F(r, (r + 1) % n, 1, 1) for r in range(n)]
        self._sides_inv = [pp.inverse_monomial(s) for s in self._sides]
        self.spans = tuple(range(2, (n - 1) // 2 + 1))
        self._chords = {(j, p): pp.arc_F(p, (p + j) % n, 1, 1)
                        for j in self.spans for p in range(n)}
        # geometric certificate 1: the closed-form base words solve the
        # pin-cancellation system exactly, for every chord (rotation
        # covariance included).
        self._W0 = {j: closed_form_word(n, j) for j in self.spans}
        pin_s = [pp.pinning_vector(s) for s in self._sides]
        for (j, p), D in self._chords.items():
            want = pp.pinning_vector(D)
            acc = [0] * n
            for t in range(n):
                w = self._W0[j][(t - p) % n]
                for x in range(n):
                    acc[x] += w * pin_s[t][x]
            assert tuple(acc) == want, \
                f"pin-cancellation fails at D{j}_{p} (n={n})"
        # bare dressed chords (gauge-free)
        bare = {(j, p): self._dress_bare(j, p)
                for j in self.spans for p in range(n)}
        # geometric certificate 2: all dressed chords are pin-0.
        for (j, p), B in bare.items():
            assert pp.pinning_vector(B) == (0,) * n, \
                f"dressed chord B{j}_{p} not pin-0 (n={n})"
        # gauge calibration: the unit side-resolution term of the family
        # meson B_{j,0}·B_{j,1} — its identity-charge coefficient must be
        # a unit lq-monomial; x_j = −e/2 normalizes it to exactly 1.
        self._X = {}
        idc = (0,) * pp.n_edges
        for j in self.spans:
            prod = pp.multiply(bare[(j, 0)], bare[(j, 1)])
            assert idc in prod, \
                f"family-{j} meson has no identity term (n={n})"
            items = prod[idc].items()
            assert len(items) == 1 and items[0][1] == 1, \
                f"family-{j} meson identity coefficient not a unit " \
                f"monomial (n={n}): {items}"
            self._X[j] = Fraction(-items[0][0], 2)
        self._T = {(j, p): pp.scale(B, Q(self._X[j]))
                   for (j, p), B in bare.items()}
        # geometric certificate 3: every T commutes exactly with every
        # side monomial — the factorization Pinned = A1A2k ⊗ T(sides).
        for (j, p), T in self._T.items():
            for r in range(n):
                s = self._sides[r]
                assert pp.multiply(T, s) == pp.multiply(s, T), \
                    f"T{j}_{p} does not commute with side {r} (n={n})"
        self._fwd_cache = {}
        self._mult_cache = {}

    # ---- the dressed chords ----------------------------------------------

    def _dress_bare(self, j: int, p: int) -> dict:
        """B_{j,p} = F(D_{j,p}) · NO_p(sides^{-w_{j,p}}), NO_p the
        rotation-covariant order r = p, p+1, …, p+n−1 (mod n)."""
        n = self.n
        pp = self.pp
        out = self._chords[(j, p)]
        for t in range(n):
            r = (p + t) % n
            e = -self._W0[j][t]
            for _ in range(abs(e)):
                out = pp.multiply(
                    out, self._sides[r] if e > 0 else self._sides_inv[r])
        return out

    def T(self, j: int, p: int) -> dict:
        """The dressed chord T^(j)_p, j ∈ spans (a pin-0 pinned-torus
        element)."""
        return self._T[(j, p % self.n)]

    def side(self, r: int) -> dict:
        """The pinned side monomial F(s_r(+,+))."""
        return self._sides[r % self.n]

    def gauges(self) -> dict:
        """The calibrated per-family unpin gauges {span j: lq-exponent}."""
        return dict(self._X)

    def unpin_words(self) -> dict:
        """The closed-form base dressing words {span j: w_{j,0}}."""
        return dict(self._W0)

    # ---- forward map (canonical label -> pinned element) -------------------

    @staticmethod
    def _qexp1(poly):
        items = list(poly._coeffs.items())
        assert len(items) == 1 and items[0][1] == 1, poly
        return items[0][0]

    def forward(self, lbl) -> dict:
        """K(lbl): lq^{-ORI·c} times the canonical-cone-ordered product
        of the generators' dressed chords ((a,i) -> T^(a+1)_i, offset
        0), c the accumulated intrinsic phase (each incremental
        intrinsic product asserted single-term)."""
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

    # ---- KAlgebra primitives -----------------------------------------------

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
            # unique-charge peel, anchored at a unit-monomial charge of
            # the owner where available so multi-term coefficients peel
            # genuinely (the SkeinSquareKAlg loop)
            hit = None
            for u in sorted(rem):
                owners = [l for l, c in pool.items() if u in c]
                if len(owners) != 1:
                    continue
                lam = owners[0]
                anchor = pool[lam][u].items()
                if len(anchor) == 1 and anchor[0][1] in (1, -1):
                    hit = (u, lam, anchor[0], None)
                    break
                r = SkeinPentagonKAlg._mono_ratio_hl(rem[u], pool[lam][u])
                if r is not None:
                    hit = (u, lam, None, r)
                    break
            assert hit is not None, f"peel stuck for {a} * {b}"
            u, lam = hit[0], hit[1]
            if hit[2] is not None:
                be, bv = hit[2]
                h = rem[u] * Q(-be, bv)      # bv^{-1} = bv for ±1
            else:
                e, sg = hit[3]
                h = Q(e) * sg
            # THE STANDING GUARD: the measured pinned coefficient must
            # equal the intrinsic coefficient VERBATIM under q -> lq^{-2}
            # (δ ≡ 0 everywhere — no normalization table).  Asserted,
            # never fitted.
            want = SkeinPentagonKAlg._chart_to_hl(hterms[lam])
            assert h == want, (
                f"pinned/intrinsic coefficient mismatch at {lam} in "
                f"{a} * {b}: measured {h}, intrinsic {hterms[lam]}")
            piece = pp.scale(pool.pop(lam), h)
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
        return (tuple(label), self.coefficient_ring().one())

    def r_label_decompose(self, label):
        """Trivial flavour-lift coordinate `(label, χ₀)`: odd marks — no
        flavour."""
        return tuple(label), self.coefficient_ring().one_basis()

    def r_label_compose(self, section, r_basis_label):
        return tuple(section)

    # ---- the iso witness -----------------------------------------------------

    def build_iso(self):
        from kalgebra_iso import KAlgebraIso
        one = LaurentPoly.one()

        def _id(lbl):
            return Element({tuple(lbl): one})

        return KAlgebraIso(self, self._intr, _id, _id,
                           name=f"{2 * self.k + 3}-gon[skein-pinned→a1a2k]")
