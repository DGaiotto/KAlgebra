"""
u1a1deven_dodd_build.py
=======================

`DevenDoddConeTables` — build the self-contained `U1A1DevenConeKAlgebra` tables
from the **reliable A1Dodd×QT oracle** `U1A1DevenViaDoddRG(k)` instead of the
legacy `A1A2k⊗QT⊗SU(2)` oracle (whose deep matter trace is wrong; see repo_audit
A16).  This makes the cone *fully oracle-consistent and legacy-free*: multiply/ρ
AND the trace freeze all come from the same (correct) oracle, so no cross-oracle
matching is needed (user, 2026-06-25 "rebuild cone on new oracle").

Strategy (subclass `DevenConeTables`, override only the oracle-touching pieces;
reuse the generic ρ-closure / cliques / cocycle / cross / decode machinery):

  * **Sections** are kept in the legacy-compatible nesting
    `((dodd_word, (c0, c1)), kappa)` — `c0` = magnetic ('t Hooft, the trace
    selection charge), `c1` = X_{0,1} matter tower — so the module-level
    `_modT` / `_c2_of` / `_is_pure_x01` and the cone-class section readers work
    unchanged.  The new oracle's native labels are `((dodd_word, kappa), (c0,c1))`;
    `_fmt_section` / `_to_label` convert between the two nestings.
  * Section decomposition uses the **public `r_label_decompose`** (the (section,
    SU(2)-label) flavour-lift primitive) — NOT the internal
    `_label_section_decompose`.
  * Seeds are the **A1Dodd chords** `(a, p, i)` made clean (RG-monomial, dressed
    by one X_{1,0}=c0 if the bare RG is not monomial), plus the clean QT gens.
  * `_monomial_section` multiplies the ray sections through the oracle (the new
    multiply is fast) instead of the legacy fast-`_B` chord product.
"""
from __future__ import annotations
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import SU2ZPlusRing, RLaurent
from u1a1deven_cone_build import DevenConeTables, E_GEN, E_INV


class DevenDoddConeTables(DevenConeTables):
    """`DevenConeTables` built on `U1A1DevenViaDoddRG` (A1Dodd×QT)."""

    def __init__(self, oracle=None, decode_degree: int = 4, frozen=None):
        if frozen is not None:                 # spine-free load (no oracle)
            self.__dict__.update(frozen)
            from a1dodd_kalg import A1DoddConeKAlg
            self._surv = A1DoddConeKAlg(self.k - 1)   # A1Dodd gauge factor, oracle-free
            self._B = None
            return
        self.oracle = oracle
        self._surv = oracle._surv          # A1Dodd(k-1) cone (the gauge chord factor)
        self._oracle_kind = "dodd"         # marker: frozen load picks this class
        self.R = SU2ZPlusRing()
        self.aux = oracle.auxiliary()
        self.k = oracle.k
        self.H = 2 * oracle.k + 3          # dodd position period (unused by build)
        self._B = None                     # no legacy gauge factor
        self._decode_degree = decode_degree
        self._build(decode_degree)
        self._precompute_torus_cocycles()

    def __getattr__(self, name):
        """Lazily rebuild the new (A1Dodd×QT) oracle for off-path comparisons
        after a spine-free frozen load (runtime never needs it)."""
        if name in ("oracle", "aux"):
            from u1a1deven_via_dodd_rg import U1A1DevenViaDoddRG
            self.oracle = U1A1DevenViaDoddRG(self.__dict__["k"])
            self.aux = self.oracle.auxiliary()
            return self.__dict__[name]
        raise AttributeError(name)

    # -- section <-> oracle-label nesting conversion ----------------------
    @staticmethod
    def _fmt_section(label):
        """oracle label `((word, kappa), (c0, c1))` -> section
        `((word, (c0, c1)), kappa)` (legacy-compatible nesting)."""
        (word, kappa), (c0, c1) = label
        return ((word, (c0, c1)), kappa)

    @staticmethod
    def _to_label(section):
        """section `((word, (c0, c1)), kappa)` -> oracle label
        `((word, kappa), (c0, c1))`."""
        (word, (c0, c1)), kappa = section
        return ((word, kappa), (c0, c1))

    # -- oracle helpers (public API: r_label_decompose / RG / multiply / rho)
    def _sec(self, label):
        strip, _su2 = self.oracle.r_label_decompose(label)
        return self._fmt_section(strip)

    def _sec_terms(self, elt):
        """`elt` (Z-form: {oracle-label: LaurentPoly}) regrouped into
        SU(2)-stripped sections with `RLaurent[SU(2)]` coeffs."""
        out = {}
        for lab, c in elt.terms.items():
            strip, su2 = self.oracle.r_label_decompose(lab)
            s = self._fmt_section(strip)
            rc = self.R.basis_element(su2)
            acc = out.get(s, RLaurent(self.R))
            coeffs = c._coeffs if isinstance(c, LaurentPoly) else c.coeffs
            for e, co in coeffs.items():
                acc = acc + RLaurent(self.R, {e: rc * co})
            out[s] = acc
        return {s: v for s, v in out.items() if not v.is_zero()}

    def _mul_labels(self, labels):
        """Product of Z-form oracle labels -> Element {oracle-label: LaurentPoly}."""
        cur = {labels[0]: LaurentPoly({0: 1})}
        for lab in labels[1:]:
            nxt = {}
            for l, c in cur.items():
                p = self.oracle.multiply(l, lab)
                for l2, c2 in p.terms.items():
                    c2lp = c2 if isinstance(c2, LaurentPoly) else LaurentPoly(dict(c2._coeffs))
                    nxt[l2] = nxt.get(l2, LaurentPoly({})) + c * c2lp
            cur = {l: c for l, c in nxt.items() if not c.is_zero()}
        return Element(cur)

    def _secmul(self, gs, hs):
        return self._sec_terms(self.oracle.multiply(self._to_label(gs),
                                                    self._to_label(hs)))

    def _secrho(self, s):
        return self._sec(self.oracle.rho(self._to_label(s)))

    def _secrho_inv(self, s):
        return self._sec(self.oracle.rho_inverse(self._to_label(s)))

    def _mrg(self, lab):
        """nterms of RG of an oracle label."""
        return self._nterms(self.oracle.RG(lab))

    # -- seeds: A1Dodd chords (a,p,i) made clean + QT gens ----------------
    def _make_clean(self, chord):
        """Oracle label for the chord `(a,p,i)` in its clean (RG-monomial) frame:
        bare at c0=0 if monomial, else dressed by one X_{1,0} (c0=±1)."""
        for c0 in (0, 1, -1):
            lbl = ((((chord, 1),), 0), (c0, 0))
            if self._mrg(lbl) == 1:
                return lbl
        return None

    def _seed_sections(self):
        surv = self.oracle._surv          # A1Dodd cone (chords live here)
        chords = surv.cone_data().mult_gens()
        seeds = set()
        for chord in chords:
            cl = self._make_clean(chord)
            if cl is not None:
                seeds.add(self._sec(cl))
        for q in [(((), 0), (1, 0)), (((), 0), (-1, 0)),
                  (((), 0), (0, 1)), (((), 0), (0, -1))]:
            if self._mrg(q) == 1:
                seeds.add(self._sec(q))
        return seeds

    def _chord_nletters(self, chord):
        """A1Dodd word = tuple of `((a,p,i), exp)`; letters = sum of exps."""
        return sum(exp for (_c, exp) in chord)

    def _surv_mul(self, la, lb):
        """Cached A1Dodd chord product `{label: LaurentPoly}` (chord words repeat
        heavily across cone combos during decode)."""
        cache = self.__dict__.setdefault("_survmul_cache", {})
        key = (la, lb)
        hit = cache.get(key)
        if hit is None:
            p = self._surv.multiply(la, lb)
            hit = {l: (c if isinstance(c, LaurentPoly) else LaurentPoly(dict(c._coeffs)))
                   for l, c in p.terms.items() if not c.is_zero()}
            cache[key] = hit
        return hit

    # -- monomial section via the fast (cached) A1Dodd chord product ------
    def _monomial_section(self, ray_ids):
        """Section of a q-commuting ray monomial via the **fast A1Dodd chord
        product** (the gauge factor of the aux; the dodd analog of the legacy
        `_B`) + QT-charge addition — no slow full-oracle/RG call.  Clean-frame
        rays multiply monomially on the chord skeleton; SU(2) CG only varies the
        kappa (same section), so q-commute ⟺ a single chord-WORD survives.
        None if more than one word appears (rays don't all q-commute).
        Memoized by the (sorted) ray-id tuple."""
        cache = self.__dict__.setdefault("_monsec_cache", {})
        key = tuple(ray_ids)
        if key in cache:
            return cache[key]
        s0 = self.rays[ray_ids[0]]
        (word0, (C0, C1)), kap0 = s0
        cur = {(word0, kap0): LaurentPoly({0: 1})}     # A1Dodd element
        ok = True
        for r in ray_ids[1:]:
            rs = self.rays[r]
            (w, (rc0, rc1)), rk = rs
            nxt = {}
            for lab, cc in cur.items():
                for l2, c2 in self._surv_mul(lab, (w, rk)).items():
                    nxt[l2] = nxt.get(l2, LaurentPoly({})) + cc * c2
            cur = {l: c for l, c in nxt.items() if not c.is_zero()}
            C0 += rc0; C1 += rc1
            if len({lab[0] for lab in cur}) > 1:        # early exit: >1 word
                ok = False; break
        res = None
        if ok:
            words = {lab[0] for lab in cur}
            if len(words) == 1:
                res = ((words.pop(), (C0, C1)), 0)
        cache[key] = res
        return res
