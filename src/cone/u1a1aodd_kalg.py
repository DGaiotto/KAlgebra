"""`U1A1AoddKAlg(k)` — the u(1)-gauged `[A_1, A_{2k+1}]` as a
SELF-CONTAINED `ConeKAlgebra`: no frozen pickles, no RG oracle, at any `k`.

**This class replaced the oracle-extracted `U1A1AoddKAlg`**, which loaded
four frozen cone tables (`u1a1aodd_tables_k{1..4}.pkl`, computed by an
RG-flow derivation not included in this repository) and so stopped at
`k = 4`.  The predecessor stays in the source repository as the
**certification oracle** of this class.  Labels are the clean geometric
convention below; the predecessor's per-type offsets were an artifact of its
fitted alignment, see `geometric_label`.

Where the predecessor extracted its cone tables once from the
`U1A1AoddGaugedRG(k)` oracle and froze them (`u1a1aodd_tables_k{k}.pkl`),
this class *derives* every table lazily by **pure analytic combinatorics**
(closed-form analytic formulae, avoiding the embedding): the even
family's two arc rules (the q-commute arc-parity exponent and the fewer-odd-arcs quantum Ptolemy), the
rank-2 torus pairing `⟨a,b⟩ = a₀b₁ − a₁b₀` of the charge law's `(c0, c1)`
labels, and exact multiset subtraction — no `TensorKAlgebra`, no
even-algebra multiplication, no oracle, at any `k`.  The generator
embedding into the even family's closed-form algebra remains the
*derivation* of those rules and the independent cross-check
(`_peel_via_embedding` / `verify_cross_product_analytic_vs_embedding` —
identical on every crossing pair, k = 1..3 in-sample and k = 4 out of
sample, 1,580 pairs):

    Φ :  u(1)-gauged [A_1, A_{2k+1}]  ↪  A1A2kKAlg(k) ⊗ QT[Z^2]

(leg 2 of the `A1Aeven / U1A1Aodd` chain; `S_RG` a single quantum
dilogarithm).  Measured k = 1..5 and held out at k = 6 (21,068 identities, 0
failures — a probe in the source repository): the image of
EVERY chord generator is a 1- or 2-term sum with all coefficients exactly 1,
given by the vertex merge of the `(2k+4)`-gon onto the `(2k+3)`-gon
(collapse the edge `(H−1, 0)`; a chord ending on the merged vertex resolves
both ways) with the `QT` charges of `phi_letter` below.

**Labels (clean geometric convention).**  Chord letter `(t, i)` := the `(2k+4)`-gon diagonal
`{i, i+t+1}`, `t = 1..k+1`, `i ∈ Z_H` (`Z_{H/2}` for the diameter
`t = k+1`); `E = (0,0)`, `E⁻¹ = (0,1)` the gauge `QTCone` Laurent letters;
native labels `(factors, e_E)` exactly as in the retired predecessor.
In this gauge each letter names the SAME
canonical element as the predecessor's letter at the same diagonal (no
E-shift — certified in the source repository's tests), and `ρ`'s label map concentrates
its E-drift at ONE position per type:

    ρ(M(t,i)) = E^{d}·M(t, i+1),   ρ(E) = E⁻¹,
    d = +2 at i = H−1   (t = 1)
        −2 at i = H−t−1 (t odd, 3 ≤ t ≤ k)
        +1 at i = H/2−1 (t = k+1, the diameter)
         0 otherwise                       (measured k = 1..4; certified vs
                                            the predecessor's rho in the
                                            source repository's tests)

**Magnetic charge = endpoint parity.**  A letter's magnetic charge — the
`𝖖`-power `m` in `E·L_g = 𝖖^m·L_g·E`, which `ungauge_kalgebra.
UngaugedKAlgebra.mag` reads off the `E`-commutator — is set by the parities
of the two endpoints of its diagonal: `−2` when both are even, `+2` when both
are odd, `0` when they differ (exactly the letters of even type `t`).  It is
`2·cocycle(E, g)`, i.e. twice the last coordinate of `charge_formula(t, i)`;
see `U1A1AoddConeData._letter_mag` for the derivation.  So a label lies in
the centralizer of `E` — the ungauged `[A_1, A_{2k+1}]` of
`ungauge_kalgebra.ungauge_u1a1aodd(k)` — iff its multiset of diagonals is
BALANCED: as many even–even diagonals as odd–odd ones, counted with
multiplicity (the `E`-power is unconstrained).  Its multiplicative generators,
besides `E^{±1}` (which the ungauging turns into the fugacity), are the
mixed-parity diagonals and the non-crossing (even–even, odd–odd) pairs
(`U1A1AoddKAlg._centralizer_generators`); that multiset of diagonals is the
geometric label of the ungauged algebra.

**What is derived, and from what.**

* chord charges (`B_GAUGED = A_{2k+2}` chain coordinates): the CLOSED FORM
  `charge_formula(t, i)` — two arithmetic progressions plus an alternating
  tail for a non-wrapping chord, four explicit shapes indexed by
  `r = t − w` for a wrapping one, and the single `(t, w) = (1, 1)`
  exception `−2μ` that is the charge-level face of the `E²` drift above.
  No sweep and no iteration.  Compressed out of the ORIGINAL route, which is kept as
  the independent cross-check (`_charges` /
  `verify_charge_formula_vs_sweep`): the even chord charges by the BPS-free
  closed-form `ρ`-sweep (`A1A2k_naming_audit.a2k_rho` from
  `natural_orbit_seeds` — the `a1a2k_bps_iso._compute_chord_charges`
  route), pushed through the flow's charge map (the `φ` lattice map of
  the design notes "Iso gate": even `γ ↦ (0, γ, 0)`,
  dressed leg `↦ MU = (1,0,1,0,…)`, gauge leg `↦ (0,−1,0,−1,…)`), reading
  the seam letters through their LEFT resolution (`{u, H−2}`, the
  `c1r − 1` term) — the convention that reproduces the canonical labels'
  charges (verified against the predecessor's).  The two
  routes agree on all 1,065 chord letters at `k = 1..10`; the formula was
  fitted on `k ≤ 5`, so 870 of those are out of sample.
* `q_commute` = the certified non-crossing dictionary (E commutes with all).
* `cocycle` = the `A_{2k+2}` chain pairing of those charges (the
  predecessor's own closed-form rule, now computed rather than loaded).
* `cross_product` = the ANALYTIC peel (`_peel_analytic`): the daughters are
  the quantum-Ptolemy resolutions of the crossing (geometric), and each
  term's `𝖖`- and `E`-power is read off exactly by matching the SYMBOLIC
  `Φ`-images — even arc rules + torus pairing, no algebra multiplied — and
  subtracting multisets to zero (honest-fail on any residual).  Lazy,
  cached per ordered pair.  `_peel_via_embedding` does the same peel in the
  genuine auxiliary and is the cross-check, not the product.
* trace: the generic Layer-1 `ρ²`-cyclicity reduction over the cones, with
  EVERY seed one closed form (2026-09-23): `u1_pgon_layer2.
  singlet_chord_trace(p, j, n, K)` for the `E`-tower (`j = 0`) and each even
  chord type `2j` — a difference of two `M(1, p)` singlet module characters
  with a `𝖖`-power prefactor, the gauged analogue of `A1A2kKAlg`'s
  `T_a = (−1)^{m+1}𝖖^{−m}(χ_m − χ_{m+1})`; odd chord types are gauge-charged
  and vanish.  Defined for every label to any order; certified against the
  exact RG transport down `U1A1AoddToEvenQTRGKAlgebra` and the
  orthonormality bootstrap (`u1aodd_trace_bootstrap`, now a cross-check).
  The per-family fitted forms it replaced were wrong from about `𝖖²⁶–𝖖⁴³`.

Construction cost: none — the cone data is combinatorial and every table is
computed on demand and cached (a cross-product pair costs a handful of
symbolic ≤2-term multiplies).  The production path imports NO even-family
module: `a1a2k_kalg` / `tensor_kalgebra` / `quantum_torus_kalgebra` /
`a1a2k_bps_iso` appear only inside `aux()` and `_charges()`, i.e. only when
a cross-check verifier is called.

The source repository's tests certify, against
the retired predecessor under the diagonal-matching letter dictionary:
identical structure constants on every generator pair, identical `ρ`,
charges, cocycles, and trace seeds, plus the intrinsic axiom checks (bar
involution, associativity spot-window, `ρ` automorphism, orthonormality
spot checks).
"""
from __future__ import annotations

import sys
import os

# APPEND, never insert(0): a front insertion of this directory could shadow
# same-named modules elsewhere on the path.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

from kalgebra import Element
from cone_kalgebra import ConeKAlgebra
from cone_data import CrossProductTerm, FiniteConeData, Cone
from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing, RPowerSeries

from u1a1aodd_mult_table import chain_pairing
import u1_pgon_layer2 as _gp

E_GEN = (0, 0)
E_INV = (0, 1)

_ONE = LaurentPoly.one()


# ---------------------------------------------------------------------------
# the closed-form generator embedding (the law of
# a probe in the source repository, verbatim)
# ---------------------------------------------------------------------------

def _phi_letter_data(k, u, v):
    """`Φ` of the chord `{u, v}` (sorted, `0 ≤ u < v ≤ H−1`) as a list of
    `(even_cone_label_or_(), (c0, c1))` with coefficient 1 each.  Even cone
    labels are `A1A2kKAlg(k)` letters `((s, j, 1),)`; `()` is the identity
    (an `Hp`-gon edge after the merge)."""
    H = 2 * k + 4
    Hp = H - 1

    def ev(x, y):
        x, y = x % Hp, y % Hp
        if (y - x) % Hp == 1 or (x - y) % Hp == 1:
            return ()
        for s in range(1, k + 1):
            if (x + s + 1) % Hp == y:
                return ((s, x, 1),)
            if (y + s + 1) % Hp == x:
                return ((s, y, 1),)
        raise ValueError((x, y))

    ell = v - u
    t = min(ell, H - ell) - 1
    c0 = ((-1) ** u) if t % 2 == 1 else 0
    if v <= H - 2:
        if 2 <= ell <= H // 2:
            c1 = 0
        elif t % 2 == 0:
            c1 = (-1) ** u
        else:
            c1 = 1 if u == 0 else -((-1) ** u)
        return [(ev(u, v), (c0, c1))]
    # seam chord {u, H-1}: the two vertex-merge resolutions; LEFT = c1r - 1
    if u == 1:
        c1r = -1
    elif 2 <= u <= k:
        c1r = 1
    else:
        c1r = 0
    return [(ev(u, 0), (c0, c1r)), (ev(u, H - 2), (c0, c1r - 1))]


# ---------------------------------------------------------------------------
# The ANALYTIC route (closed-form analytic formulae, avoiding the
# embedding).  The base table is computed by pure integer
# combinatorics — the even family's two arc rules,
# the rank-2 torus pairing, and the charge law of `_phi_letter_data` — with
# NO TensorKAlgebra and NO even-algebra multiplication.  The auxiliary
# expansion survives only as symbolic bookkeeping over dicts
# `(even_key, (c0,c1)) -> multiset of q-exponents`, where the even factors
# are expanded by the ARC RULES, never by an algebra.  Verified identical to
# the embedding peel on every pair at k = 1..3 (590 crossings + 590
# q-commuting checks) and out-of-sample at k = 4; the embedding peel is kept
# below as `_peel_via_embedding`, the independent cross-check.
# ---------------------------------------------------------------------------

def _ev_is_edge(ch, Hp):
    x, y = sorted(ch)
    return (y - x) % Hp in (1, Hp - 1)


def _ev_half(c1, c2, Hp):
    """Even-family half-exponent `c` with `L_{c1} L_{c2} = q^{2c} L_{c2}
    L_{c1}` for a non-crossing pair — the arc-parity rule of the even
    family (control-verified against `A1A2kKAlg` products, 904/904 at
    k = 1..3)."""
    a = sorted(c1)[0]
    pts = sorted(set(tuple(c1)) | set(tuple(c2)), key=lambda p: (p - a) % Hp)
    arcs = [((pts[(m + 1) % len(pts)] - pts[m]) % Hp) for m in range(len(pts))]
    odd = [m for m, L in enumerate(arcs) if L % 2 == 1]
    if len(odd) != 1:
        return 0
    m = odd[0]
    p0, p1 = pts[m], pts[(m + 1) % len(pts)]
    s1, s2 = set(c1), set(c2)
    if (p0 in s1 - s2) and (p1 in s2 - s1):
        return -1
    if (p0 in s2 - s1) and (p1 in s1 - s2):
        return +1
    return 0


def _ev_cross(c1, c2):
    (a, b), (c, d) = sorted(c1), sorted(c2)
    if len({a, b, c, d}) < 4:
        return False
    return (a < c < b) != (a < d < b)


def _ev_ptolemy(c1, c2, Hp):
    """(alpha, D1, beta, D2) for the ORDERED even product with the chord
    containing the least quadrilateral vertex first; fewer-odd-arcs rule."""
    vs = sorted(set(tuple(c1)) | set(tuple(c2)))
    a, b, c, d = vs
    arcs = [(b - a) % Hp, (c - b) % Hp, (d - c) % Hp, (a - d) % Hp]
    al, be = ((1, 0) if (arcs[0] % 2) + (arcs[2] % 2)
              < (arcs[1] % 2) + (arcs[3] % 2) else (0, -1))
    D1 = [p for p in ((a, b), (c, d)) if not _ev_is_edge(p, Hp)]
    D2 = [p for p in ((a, d), (b, c)) if not _ev_is_edge(p, Hp)]
    return al, D1, be, D2


def _ev_key(chords):
    from collections import Counter
    return frozenset(Counter(frozenset(c) for c in chords).items())


def _ev_literal(letters, Hp):
    """Ordered literal product of 0..2 even chords -> {even_key: exponent},
    by the arc rules alone."""
    letters = [tuple(sorted(l)) for l in letters]
    letters = [l for l in letters if not _ev_is_edge(l, Hp)]
    if not letters:
        return {_ev_key([]): 0}
    if len(letters) == 1:
        return {_ev_key(letters): 0}
    if len(letters) == 2:
        A, B = letters
        if not _ev_cross(A, B):
            return {_ev_key([A, B]): _ev_half(A, B, Hp)}
        al, D1, be, D2 = _ev_ptolemy(A, B, Hp)
        vs = sorted(set(A) | set(B))
        if set(A) != {vs[0], vs[2]}:
            al, be = -al, -be          # reversed order = bar = negation
        return {_ev_key(D1): al, _ev_key(D2): be}
    raise NotImplementedError(letters)


def _qt_pair(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _aux_mul(x, y, Hp):
    """Symbolic product of aux dicts `lab -> tuple of q-exponents`."""
    out = {}
    for (ek1, a), es1 in x.items():
        for (ek2, b), es2 in y.items():
            l1 = [c for c, m in ek1 for _ in range(m)]
            l2 = [c for c, m in ek2 for _ in range(m)]
            ev_exp = _ev_literal([tuple(sorted(c)) for c in l1 + l2], Hp)
            ph = _qt_pair(a, b)
            qt = (a[0] + b[0], a[1] + b[1])
            for ekey, ee in ev_exp.items():
                lab = (ekey, qt)
                cur = list(out.get(lab, ()))
                cur += [e1 + e2 + ph + ee for e1 in es1 for e2 in es2]
                out[lab] = tuple(sorted(cur))
    return out


def _phi_dict(k, chord_or_E):
    if chord_or_E == 'E+':
        return {(_ev_key([]), (0, 1)): (0,)}
    if chord_or_E == 'E-':
        return {(_ev_key([]), (0, -1)): (0,)}
    u, v = sorted(chord_or_E)
    out = {}
    for evc, qt in _phi_letter_data(k, u, v):
        key = _ev_key([tuple(sorted((evc[0][1], (evc[0][1] + evc[0][0] + 1)
                                     % (2 * k + 3))))] if evc else [])
        lab = (key, qt)
        out[lab] = tuple(list(out.get(lab, ())) + [0])
    return out


def _sub_multiset(remaining, img, alpha):
    rem = {lab: list(v) for lab, v in remaining.items()}
    for lab, exps in img.items():
        for e in exps:
            if lab not in rem or (e + alpha) not in rem[lab]:
                return None
            rem[lab].remove(e + alpha)
    return {lab: tuple(v) for lab, v in rem.items() if v}


class U1A1AoddConeData(FiniteConeData):
    """Cone data for `U1A1AoddKAlg` — every table derived
    analytically (production: the arc/charge combinatorics above; the
    embedding peel kept as the independent cross-check); nothing loaded."""

    def __init__(self, alg, k):
        self._alg = alg
        self._k = k
        self._H = 2 * k + 4
        self._n = 2 * k + 2
        self._MU = tuple(1 if j % 2 == 0 else 0 for j in range(self._n))
        H = self._H
        self._size = {t: (H // 2 if t == k + 1 else H) for t in range(1, k + 2)}
        self._types = {t: list(range(self._size[t])) for t in range(1, k + 2)}
        self._chords = tuple(
            (t, i) for t in range(1, k + 2) for i in range(self._size[t])
        )
        self._mult_gens = self._chords + (E_GEN, E_INV)
        self._aux = None            # A1A2kKAlg(k) (x) QT[Z^2], built lazily
        self._phi_cache = {}
        self._sweep_chg = None      # the CROSS-CHECK route's cache (see _charges)
        self._chg_cache = None      # the production letter -> charge dict
        self._cross_cache = {}
        self._cones = None

    @property
    def _chg(self):
        """`{(t, i): charge}` for every chord letter — the attribute name the
        consumers of this class read (`e6_rgkalgebra`, `e8_rgkalgebra`,
        `skein_atlas`, …).  Computed from
        `charge_formula`, memoized.  Kept as a mapping, not a method, because
        that is the surface those consumers were written against."""
        if self._chg_cache is None:
            self._chg_cache = {g: self.charge_formula(*g) for g in self._chords}
        return self._chg_cache

    # -- geometry ----------------------------------------------------------

    def chord(self, g):
        """The diagonal `{i, i+t+1}` of the letter `g = (t, i)`."""
        t, i = g
        return (i % self._H, (i + t + 1) % self._H)

    @staticmethod
    def _crossing(c1, c2, H):
        a, b = c1
        c, d = c2
        if len({a, b, c, d}) < 4:
            return False
        lo, hi = min(a, b), max(a, b)
        # interior arc test on the cycle: exactly one of {c, d} strictly
        # between a and b going the short way is not well-defined on a cycle;
        # use the standard interleaving test
        def between(x):
            return lo < x < hi
        return between(c) != between(d)

    def _letter_of_chord(self, x, y):
        """The letter `(t, i)` naming the diagonal `{x, y}` (None for an
        edge)."""
        H = self._H
        x, y = x % H, y % H
        ell = (y - x) % H
        if ell in (1, H - 1):
            return None
        t = min(ell, H - ell) - 1
        if t == self._k + 1:                      # diameter: position < H/2
            i = x if x < H // 2 else y
            return (t, i % (H // 2))
        if (x + t + 1) % H == y:
            return (t, x)
        return (t, y)

    def chord_length(self, a):
        """Geometric length of chord-type `a` on the `(2k+4)`-gon: `a+1`."""
        return a + 1

    def _geom(self):
        """`(sgn, {type: base-vertex offset})` in the affine form
        `v = (sgn·i + off[t]) % H`, `diag = (v, v + chord_length(t))`.

        Kept because consumers re-derive the letter↔diagonal correspondence
        from it rather than calling `geometric_label`
        (`skein_sphere/skein_evengon_kalg.py` inlines exactly this formula).
        Here it is trivial — `sgn = +1`, every offset `0` — because the clean
        labelling IS `(t,i) ↦ {i, i+t+1}`.  The oracle-extracted predecessor
        had to SEARCH for these values against its q-commutation table."""
        return (1, {t: 0 for t in self._size})

    def geometric_label(self, g):
        """Chord `g = (t, i)` as a sorted `(2k+4)`-gon diagonal `(v1, v2)`;
        `None` for the gauge torus direction `E`/`E⁻¹`.

        Here this is a DEFINITION — `chord` sorted — not a fitted alignment:
        the labelling `(t,i) ↦ {i, i+t+1}` is imposed, and
        `verify_qcommute_is_noncrossing` then *measures* that the cocycle's
        geometry agrees with it.  (Contrast the oracle-extracted predecessor,
        which had to brute-force search a per-type offset and could fail to
        find one at all.)"""
        if g in (E_GEN, E_INV):
            return None
        return tuple(sorted(self.chord(g)))

    def verify_qcommute_is_noncrossing(self):
        """The cocycle's geometric content: `q_commute(g,h)` iff the diagonals
        `geometric_label(g), geometric_label(h)` do not cross."""
        for g in self._chords:
            for h in self._chords:
                if g == h:
                    continue
                nc = not self._crossing(self.geometric_label(g),
                                        self.geometric_label(h), self._H)
                if self.q_commute(g, h) != nc:
                    return False
        return True

    def _letter_mag(self, g):
        """The magnetic charge of the letter `g` — the `𝖖`-power `m` in
        `E·L_g = 𝖖^m·L_g·E`, the quantity `UngaugedKAlgebra.mag` reads off
        the `E`-commutator — from the parities of the endpoints of its
        diagonal `{i, i+t+1}`:

            −2   both endpoints even,
            +2   both endpoints odd,
             0   one even, one odd   (exactly the letters of even type t);

        `0` for `E`/`E⁻¹`.

        Derivation.  `E` `𝖖`-commutes with every letter, and in the cone
        convention (`_sort_within_cone`) `L_g·L_h = 𝖖^{2·cocycle(g,h)}·L_h·L_g`,
        so `m = 2·cocycle(E, g)`.  `cocycle(E, g) =
        chain_pairing(μ, charge_formula(t, i))` with `μ = (1,0,1,0,…)` the
        charge of `E`, and that pairing telescopes to the LAST coordinate of
        `charge_formula(t, i)`.  The closed form sets it to `−1 / +1 / 0` for
        both-even / both-odd / mixed endpoints: on an interior chord only the
        alternating tail reaches `e_n`, with sign `(−1)^{u+1}`, and only for
        same-parity endpoints; on a wrapping chord the smaller endpoint is
        `w`, and the three shapes reach `e_n` exactly when the two endpoints
        have the same parity, with sign `−` for `w` even and `+` for `w` odd;
        the `(t, w) = (1, 1)` exception `−2μ` has last coordinate `0`.  Checked: `2·charge_formula(t, i)[n−1]` equals
        this rule on all 1,714 letters of `k = 1..12`, and `UngaugedKAlgebra.
        mag` equals it on every letter at `k = 1..5`
        (the suite in the source repository)."""
        if g in (E_GEN, E_INV):
            return 0
        u, v = self.chord(g)
        if (u - v) % 2:
            return 0
        return -2 if u % 2 == 0 else 2

    # -- the auxiliary and Phi ----------------------------------------------

    def aux(self):
        if self._aux is None:
            from a1a2k_kalg import A1A2kKAlg
            from quantum_torus_kalgebra import QuantumTorusKAlg
            from tensor_kalgebra import TensorKAlgebra
            self._aux = TensorKAlgebra(
                A1A2kKAlg(self._k), QuantumTorusKAlg([[0, 1], [-1, 0]])
            )
        return self._aux

    def phi(self, g):
        """`Φ(letter)` as an auxiliary `Element`."""
        if g not in self._phi_cache:
            if g == E_GEN:
                el = Element({((), (0, 1)): _ONE})
            elif g == E_INV:
                el = Element({((), (0, -1)): _ONE})
            else:
                u, v = sorted(self.chord(g))
                el = Element({(evl, qt): _ONE
                              for evl, qt in _phi_letter_data(self._k, u, v)})
            self._phi_cache[g] = el
        return self._phi_cache[g]

    def phi_label(self, lbl):
        """`Ψ` — the closed-form image of a canonical cone label
        `(factors, e_E)`: `q^{-T}·∏ Φ(letter) · X_{(0, e_E)}` in sorted
        letter order, `T` the pairwise-cocycle phase (the universal
        bar-invariance normalisation, `cone_data.ConeData.cone_label_phase`)."""
        factors, e = lbl
        letters = []
        for (t, i, ex) in factors:
            letters += [(t, i)] * ex
        T = 0
        for a in range(len(letters)):
            for b in range(a + 1, len(letters)):
                T += self.cocycle(letters[a], letters[b])
            T += e * self.cocycle(letters[a], E_GEN)
        aux = self.aux()
        out = Element({((), (0, e)): _ONE})
        for g in reversed(letters):
            out = aux.multiply_elements(self.phi(g), out)
        return Element({lab: c * LaurentPoly({-T: 1})
                        for lab, c in out.terms.items()})

    # -- charges: THE CLOSED FORM (production) ------------------------------
    #
    # Fitted on k = 1..5, held out at k = 6; the wrap rows
    # are cross-checked against the even-sweep route on every letter.

    def charge_formula(self, t, i):
        """The chord charge in `B_GAUGED = A_{2k+2}` chain coordinates,
        as an explicit formula — no sweep, no iteration.

        Basis `e_1..e_n`, `n = 2k+2`; `μ = (1,0,1,0,…)`.  Chord `(t, i)` =
        diagonal `{i, i+t+1}` of the `H`-gon, `H = 2k+4`.

        INTERIOR (`i + t + 1 ≤ H−1`), with `u = i`, `v = i+t+1`:

            charge = − Σ { e_j : n+2−v ≤ j ≤ n−1−u,  j ≡ v (mod 2) }
                     − [u ≡ v (2)] · (e_{n−u} − e_{n−u+1} + … ± e_n)

        (the second line — the alternating telescope — is present exactly
        for the same-parity-endpoint chords, i.e. odd `t`).

        WRAP (`i + t + 1 ≥ H`), with `w = i − (H−t−1) ∈ 0..t`, `r = t−w`:

          * `w = 0`, `t` odd:   +(e_1 + e_3 + … + e_t)
                                − (e_{t+1} − e_{t+2} + … ± e_n)
          * `r` even:           +(e_1 + e_3 + … + e_{r−1})
                                + (e_{n−t+1+r} + e_{n−t+3+r} + … ≤ e_n)
          * `r` odd, `w ≥ 1`:   −(e_1 + e_3 + … + e_{r−2})
                                − (e_r + e_{r+1} + … + e_{n−t+r})
                                − (e_{n−t+r+2} + e_{n−t+r+4} + … ≤ e_n)
          * plus the single exception `(t, w) = (1, 1)`: an extra `−2μ`
            — the charge-level face of the `E²` drift anchored at
            `(1, H−1)` in the ρ law.
        """
        H, n, k = self._H, self._n, self._k
        g = [0] * n

        def add(j, s):
            if 1 <= j <= n:
                g[j - 1] += s

        u, v = i, i + t + 1
        if v <= H - 1:
            j = n + 2 - v
            if (j - v) % 2 == 1:
                j += 1
            while j <= n - 1 - u:
                add(j, -1)
                j += 2
            if (u - v) % 2 == 0:
                s = -1
                for j in range(n - u, n + 1):
                    add(j, s)
                    s = -s
            return tuple(g)
        w = i - (H - t - 1)
        r = t - w
        if w == 0 and t % 2 == 1:
            for j in range(1, t + 1, 2):
                add(j, +1)
            s = -1
            for j in range(t + 1, n + 1):
                add(j, s)
                s = -s
        elif r % 2 == 0:
            for j in range(1, r, 2):
                add(j, +1)
            for j in range(n - t + 1 + r, n + 1, 2):
                add(j, +1)
        else:
            for j in range(1, r - 1, 2):
                add(j, -1)
            for j in range(r, n - t + r + 1):
                add(j, -1)
            for j in range(n - t + r + 2, n + 1, 2):
                add(j, -1)
        if t == 1 and w == 1:
            for j in range(n):
                g[j] -= 2 * self._MU[j]
        return tuple(g)

    # -- charges via the even sweep (the original route; now the cross-check)
    #
    # This route reads the even-family chord charges off
    # `a1a2k_bps_iso._compute_chord_charges` and pushes them through the
    # generator embedding's charge matrix.  It is no longer on the
    # production path — `_charge` uses `charge_formula` — and is kept as
    # the independent second derivation, compared letter by letter by
    # `verify_charge_formula_vs_sweep`.

    def _charges(self):
        if self._sweep_chg is None:
            from a1a2k_bps_iso import _compute_chord_charges
            k, n, H = self._k, self._n, self._H
            evch = _compute_chord_charges(k, {a: 0 for a in range(1, k + 1)})
            gauge = tuple(0 if j % 2 == 0 else -1 for j in range(n))
            chg = {}
            for g in self._mult_gens:
                if g in (E_GEN, E_INV):
                    continue
                u, v = sorted(self.chord(g))
                terms = _phi_letter_data(k, u, v)
                evl, (c0, c1) = terms[-1]        # seam: the LEFT resolution
                vec = [c0 * gauge[j] + c1 * self._MU[j] for j in range(n)]
                if evl:
                    s, j0, _ = evl[0]
                    gamma = evch[(s, j0)]
                    for j in range(2 * k):
                        vec[1 + j] += gamma[j]
                chg[g] = tuple(vec)
            self._sweep_chg = chg
        return self._sweep_chg

    def verify_charge_formula_vs_sweep(self):
        """Emergent verifier: the closed form `charge_formula` agrees with
        the even-sweep route on every chord letter.  Measured 1 065/1 065
        letters at `k = 1..10` (the wrap shapes were fitted on `k ≤ 5`, so
        `k ≥ 6` is out of sample)."""
        return all(tuple(self.charge_formula(*g)) == tuple(self._charges()[g])
                   for g in self._mult_gens if g not in (E_GEN, E_INV))

    def _charge(self, g):
        if g == E_GEN:
            return self._MU
        if g == E_INV:
            return tuple(-x for x in self._MU)
        return self.charge_formula(*g)

    # -- FiniteConeData surface ---------------------------------------------

    def mult_gens(self):
        return self._mult_gens

    def cones(self):
        if self._cones is None:
            V = list(self._mult_gens)
            adj = {v: frozenset(u for u in V if u != v and self.q_commute(v, u))
                   for v in V}
            out = []

            def bk(R, P, X):
                if not P and not X:
                    out.append(R)
                    return
                piv = max(P | X, key=lambda u: len(P & adj[u]))
                for v in list(P - adj[piv]):
                    bk(R | {v}, P & adj[v], X & adj[v])
                    P = P - {v}
                    X = X | {v}

            bk(frozenset(), frozenset(V), frozenset())
            self._cones = tuple(out)
        return self._cones

    def q_commute(self, g, h):
        if g == h or {g, h} == {E_GEN, E_INV}:
            return True
        if g in (E_GEN, E_INV) or h in (E_GEN, E_INV):
            return True
        return not self._crossing(self.chord(g), self.chord(h), self._H)

    def cocycle(self, g, h):
        if g == h or {g, h} == {E_GEN, E_INV}:
            return 0
        return chain_pairing(self._charge(g), self._charge(h))

    def cross_product(self, g, h):
        if self.q_commute(g, h):
            raise ValueError(f"cross_product on q-commuting {g},{h}")
        key = (g, h)
        if key not in self._cross_cache:
            self._cross_cache[key] = self._peel_analytic(g, h)
        return self._cross_cache[key]

    # -- the analytic route (production): pure arc/charge combinatorics ------

    def _word_image(self, word):
        """Symbolic aux image of a literal class word (E letters first per
        the sorted-word convention), by the arc rules alone."""
        Hp = self._H - 1
        out = {(_ev_key([]), (0, 0)): (0,)}
        for w in word:
            if w == E_GEN:
                f = _phi_dict(self._k, 'E+')
            elif w == E_INV:
                f = _phi_dict(self._k, 'E-')
            else:
                f = _phi_dict(self._k, tuple(sorted(self.chord(w))))
            out = _aux_mul(out, f, Hp)
        return out

    def _peel_analytic(self, g, h):
        """The crossing pair's `CrossProductTerm`s from the analytic rules —
        no TensorKAlgebra, no even-algebra multiply.  Verified identical to
        `_peel_via_embedding` on every pair at k = 1..3 and out-of-sample at
        k = 4 (590 + 990 crossings)."""
        H, k = self._H, self._k
        Hp = H - 1
        gu, hu = tuple(sorted(self.chord(g))), tuple(sorted(self.chord(h)))
        vs = sorted(set(gu) | set(hu))
        if len(vs) != 4:
            raise NotImplementedError(
                f"_peel_analytic({g},{h}): crossing without 4 vertices")
        p, q_, r, s = vs
        prod = _aux_mul(_phi_dict(k, gu), _phi_dict(k, hu), Hp)

        def odd_edge(x, y):
            return (y - x) % H in (1, H - 1)

        daughters = []
        for pair in (((p, q_), (r, s)), ((q_, r), (p, s))):
            ch = [tuple(sorted(xy)) for xy in pair if not odd_edge(*sorted(xy))]
            daughters.append(tuple(sorted(
                self._letter_of_chord(*c2) for c2 in ch)))

        def word_of(D, e):
            wl = list(D) + ([E_GEN] * e if e > 0 else [E_INV] * (-e))
            return tuple(sorted(wl))

        cands = []
        for D in daughters:
            base = self._word_image(word_of(D, 0))
            seen = set()
            for (bek, bqt), bexps in base.items():
                for (pek, pqt), pexps in prod.items():
                    if pek == bek and pqt[0] == bqt[0]:
                        e = pqt[1] - bqt[1]
                        if (D, e) in seen:
                            continue
                        seen.add((D, e))
                        img = self._word_image(word_of(D, e))
                        ilab = (bek, (bqt[0], bqt[1] + e))
                        if ilab not in img:
                            continue
                        for pe in pexps:
                            for ie in img[ilab]:
                                cands.append((word_of(D, e), pe - ie, img))
        uniq, seenc = [], set()
        for c2 in cands:
            if (c2[0], c2[1]) not in seenc:
                seenc.add((c2[0], c2[1]))
                uniq.append(c2)
        for i1 in range(len(uniq)):
            r1 = _sub_multiset(prod, uniq[i1][2], uniq[i1][1])
            if r1 is None:
                continue
            if not r1:
                return ((LaurentPoly({uniq[i1][1]: 1}), uniq[i1][0]),)
            for i2 in range(len(uniq)):
                r2 = _sub_multiset(r1, uniq[i2][2], uniq[i2][1])
                if r2 is not None and not r2:
                    return ((LaurentPoly({uniq[i1][1]: 1}), uniq[i1][0]),
                            (LaurentPoly({uniq[i2][1]: 1}), uniq[i2][0]))
        raise NotImplementedError(
            f"_peel_analytic({g},{h}): no exact resolution — honest fail")

    def _torus_inverse_letter(self, g):
        if g == E_GEN:
            return E_INV
        if g == E_INV:
            return E_GEN
        return None

    def iter_cones(self):
        for cone in self.cones():
            yield Cone(self, cone, torus_gens=frozenset({E_GEN, E_INV}) & cone)

    def to_cone_label(self, native_label):
        factors, e_E = native_label
        gens = set()
        powers = {}
        for (a, i, ex) in factors:
            if ex <= 0:
                continue
            gg = (a, i)
            gens.add(gg)
            powers[gg] = powers.get(gg, 0) + ex
        if e_E > 0:
            gens.add(E_GEN)
            powers[E_GEN] = e_E
        elif e_E < 0:
            gens.add(E_INV)
            powers[E_INV] = -e_E
        return frozenset(gens), powers

    def from_cone_label(self, gens, powers):
        ch = sorted((g, powers[g]) for g in gens if g not in (E_GEN, E_INV))
        factors = tuple((g[0], g[1], e) for (g, e) in ch)
        e_E = powers.get(E_GEN, 0) - powers.get(E_INV, 0)
        return factors, e_E

    def cycle_period_bound(self):
        return 2 * self._H

    # -- rho (clean-gauge closed form) ---------------------------------------

    def _drift(self, t, i):
        H, k = self._H, self._k
        if t == 1 and i == H - 1:
            return 2
        if t % 2 == 1 and 3 <= t <= k and i == H - t - 1:
            return -2
        if t == k + 1 and i == H // 2 - 1:
            return 1
        return 0

    def _rho_label_cone(self, lbl):
        factors, e_E = lbl
        nf = []
        drift = 0
        for (a, i, exp) in factors:
            s = self._size[a]
            drift += exp * self._drift(a, i)
            nf.append((a, (i + 1) % s, exp))
        return tuple(sorted(nf)), -e_E + drift

    def _rho_inverse_label_cone(self, lbl):
        factors, e_E = lbl
        nf = []
        drift = 0
        for (a, i, exp) in factors:
            s = self._size[a]
            j = (i - 1) % s
            drift += exp * self._drift(a, j)
            nf.append((a, j, exp))
        return tuple(sorted(nf)), -(e_E - drift)

    # -- the peel -------------------------------------------------------------

    def verify_cross_product_analytic_vs_embedding(self, g, h):
        """Cross-check: the production analytic table entry equals the
        embedding-peel entry (the two independent routes)."""
        return tuple(self._peel_analytic(g, h)) == tuple(
            self._peel_via_embedding(g, h))

    def _peel_via_embedding(self, g, h):
        """The ORIGINAL derivation route, kept as the independent
        cross-check: expand `Φ(g)·Φ(h)` in the genuine auxiliary
        `A1A2kKAlg(k) ⊗ QT[Z²]` and match against the quantum-Ptolemy
        daughters' `Ψ`-images.  Exact; honest-fails on any residual.  Not on
        the production path since the analytic route landed."""
        aux = self.aux()
        H = self._H
        prod = aux.multiply_elements(self.phi(g), self.phi(h))

        # the crossing quadrilateral p < q < r < s
        cg, ch = self.chord(g), self.chord(h)
        vs = sorted({*cg, *ch})
        if len(vs) != 4:
            raise NotImplementedError(
                f"_peel({g},{h}): crossing without 4 distinct vertices")
        p, q_, r, s = vs
        daughters = []
        for pair in (((p, q_), (r, s)), ((q_, r), (p, s))):
            letters = [self._letter_of_chord(*xy) for xy in pair]
            letters = tuple(sorted(l for l in letters if l is not None))
            factors = tuple((t, i, 1) for (t, i) in letters)
            # merge equal letters (a resolution can square a chord)
            merged = {}
            for (t, i, ex) in factors:
                merged[(t, i)] = merged.get((t, i), 0) + ex
            factors = tuple((t, i, ex) for (t, i), ex in sorted(merged.items()))
            daughters.append(factors)

        # Candidate literal words: each daughter with an E-block, i.e. the
        # word `sorted(chord letters + E^{±|e|})`.  Match `Φ(g)·Φ(h)` against
        # the LITERAL word images `∏ Φ(w)` (in the word's canonical sorted
        # order, the same order `derived_multiply` uses) — one pathway, no
        # phase-convention seam.  `e` is read off exactly from the dressed-leg
        # (`c1`) charge of any matched auxiliary term.
        def lit_image(word):
            out = Element({self.aux().identity(): _ONE})
            for w in word:
                out = aux.multiply_elements(out, self.phi(w))
            return out

        def word_of(D, e):
            wl = [gg for (a, i, ex) in D for gg in ((a, i),) * ex]
            if e > 0:
                wl += [E_GEN] * e
            elif e < 0:
                wl += [E_INV] * (-e)
            return tuple(sorted(wl))

        base = {D: lit_image(word_of(D, 0)) for D in daughters}
        cands = []
        for D in daughters:
            for blab, bco in base[D].terms.items():
                for plab, pco in prod.terms.items():
                    if plab[0] == blab[0] and plab[1][0] == blab[1][0]:
                        e = plab[1][1] - blab[1][1]
                        w = word_of(D, e)
                        shifted = lit_image(w)
                        slab = (blab[0], (blab[1][0], blab[1][1] + e))
                        if slab not in shifted.terms:
                            continue
                        ratio = self._mono_ratio(pco, shifted.terms[slab])
                        if ratio is not None:
                            cands.append((w, ratio, shifted))
        seen = set()
        uniq = []
        for c in cands:
            if (c[0], c[1]) not in seen:
                seen.add((c[0], c[1]))
                uniq.append(c)
        for a_i in range(len(uniq)):
            for b_i in range(a_i, len(uniq)):
                A, B = uniq[a_i], uniq[b_i]
                total = Element({})
                for (w, al, shifted) in (A, B):
                    total = total + Element(
                        {l: c * LaurentPoly({al: 1})
                         for l, c in shifted.terms.items()})
                if total == prod:
                    return (
                        (LaurentPoly({A[1]: 1}), A[0]),
                        (LaurentPoly({B[1]: 1}), B[0]),
                    )
        raise NotImplementedError(
            f"_peel({g},{h}): no exact 2-term resolution found "
            f"(the closed-form law does not cover this pair — honest fail)")

    @staticmethod
    def _mono_ratio(pco, sco):
        """`q`-exponent `a` with `pco == q^a·sco`, else None."""
        pi = sorted((e, c) for e, c in pco._coeffs.items() if c)
        si = sorted((e, c) for e, c in sco._coeffs.items() if c)
        if len(pi) != len(si) or not si:
            return None
        a = pi[0][0] - si[0][0]
        for (pe, pc), (se, sc) in zip(pi, si):
            if pe - se != a or pc != sc:
                return None
        return a

    def _terms_out(self, *canonical_terms):
        """Convert `(daughter_factors, e_E, canonical q-power)` triples to the
        literal `CrossProductTerm` convention (`c_lit = c_can + phase`)."""
        out = []
        for (factors, e, alpha) in canonical_terms:
            wl = [gg for (a, i, ex) in factors for gg in ((a, i),) * ex]
            if e > 0:
                wl += [E_GEN] * e
            elif e < 0:
                wl += [E_INV] * (-e)
            word = tuple(sorted(wl))
            gset, pdict = self.to_cone_label((factors, e))
            phase = self.cone_label_phase(gset, pdict)
            out.append((LaurentPoly({alpha + phase: 1}), word))
        return tuple(out)


class U1A1AoddKAlg(ConeKAlgebra):
    """Self-contained gauged `[A_1, A_{2k+1}]` — see the module docstring.
    Native labels `(factors, e_E)`; letters `(t, i)` = diagonal `{i, i+t+1}`."""

    _R = TrivialZPlusRing()

    def __init__(self, k: int = 1):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._cd = U1A1AoddConeData(self, k)
        self._intermediate_cache = {}
        self._rep_cache = {}

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return ((), 0)

    def cone_data(self):
        return self._cd

    def multiply(self, a, b):
        return self._multiply_via_cone_data(a, b)

    def rho(self, lbl):
        return self._cd._rho_label_cone(lbl)

    def rho_inverse(self, lbl):
        return self._cd._rho_inverse_label_cone(lbl)

    def L(self, a, i):
        return (((a, i, 1),), 0)

    def geometric_label(self, g):
        """Chord mult-gen `g = (t, i)` as a sorted `(2k+4)`-gon diagonal
        `(v1, v2)` (length `t+1`, `ρ_UV` = rotate-by-one); `None` for
        `E`/`E⁻¹`."""
        return self._cd.geometric_label(g)

    # -- the centralizer of E (read by ungauge_kalgebra.UngaugedKAlgebra) ---

    def _label_mag(self, label, E=((), 1)):
        """The magnetic charge of a canonical label with respect to the
        electric generator `E = ((), n)`, `n ≠ 0` — the `𝖖`-power `m` in
        `E·L = 𝖖^m·L·E` — by the endpoint-parity rule:
        `m = n·Σ_g (exponent of g)·_letter_mag(g)` over the label's letters
        (the `E`-power contributes nothing).  `None` unless `label` is a canonical label of
        this class (known letters, integer exponents `≥ 1`, letters pairwise
        non-crossing) and `E` a nonzero power of the gauge letter;
        `UngaugedKAlgebra.in_centralizer` then keeps its two multiplies.

        On that domain `m == 0` is exactly the multiply-based test
        `E·L == L·E`: the letters of `L` and `E` lie in one cone, so both
        products are single terms on the same label, with `𝖖`-powers
        differing by `2·n·Σ_g (exponent of g)·cocycle(E, g) = m`.  Checked on
        14,241 canonical labels at `k = 1..3`
        (the suite in the source repository)."""
        if not (isinstance(E, tuple) and len(E) == 2 and E[0] == ()
                and isinstance(E[1], int) and E[1] != 0):
            return None
        cd = self._cd
        letter_set = self.__dict__.get("_letter_set")
        if letter_set is None:
            letter_set = self.__dict__["_letter_set"] = frozenset(cd._chords)
        try:
            factors, e = label
            letters, total = [], 0
            for (t, i, m) in factors:
                g = (t, i)
                if g not in letter_set or not isinstance(m, int) or m < 1:
                    return None
                letters.append(g)
                total += m * cd._letter_mag(g)
        except (TypeError, ValueError):
            return None
        if not isinstance(e, int):
            return None
        for a in range(len(letters)):
            for b in range(a + 1, len(letters)):
                if not cd.q_commute(letters[a], letters[b]):
                    return None
        return E[1] * total

    def _label_gauge_charge(self, label):
        """The gauge charge of a canonical label — the e_1-component of its
        total charge (`E` included, with charge `μ`), the `g` of the trace
        seeds' character index — or `None` off the label domain.  Read by
        `ungauge_kalgebra.UngaugedKAlgebra.trace`, whose gauge-charge sum
        concentrates where `g + n` vanishes (see its docstring)."""
        try:
            factors, e = label
            if not isinstance(e, int):
                return None
            letters = self.__dict__.get("_letter_set")
            if letters is None:
                letters = self.__dict__["_letter_set"] = frozenset(self._cd._chords)
            if any((t, i) not in letters for (t, i, _m) in factors):
                return None
            return self._seed_charge(label)[0]
        except (TypeError, ValueError):
            return None

    def _centralizer_generators(self, E=((), 1)):
        """The multiplicative generators of the ungauged `[A_1, A_{2k+1}]` —
        the cone generators of the centralizer of `E` other than `E^{±1}`
        (which the ungauging turns into the fugacity) — as labels
        `(factors, 0)`, in this order:

          * each diagonal of mixed endpoint parity (magnetic charge 0), in
            letter order — `k(k+2)` of them;
          * each non-crossing pair of one even–even diagonal (charge −2) and
            one odd–odd diagonal (charge +2).

        Why these: a canonical label is a multiset of pairwise non-crossing
        diagonals, i.e. lies in one triangulation (plus `E`), and by the
        endpoint-parity rule (`U1A1AoddConeData._letter_mag`) it is in the
        centralizer iff it is balanced — as many even–even as odd–odd
        diagonals, counted with multiplicity.  Inside one triangulation the
        balanced multisets are generated by its mixed diagonals and its
        (even–even, odd–odd) pairs, none of which is a product of the others;
        two diagonals lie in a common triangulation iff they do not cross.
        Counts: `k(k+2)` mixed diagonals (the `(k+2)²` even–odd vertex pairs
        minus the `2k+4` edges) and `(k+2)·C(k+2, 3)` pairs (every pair of
        even vertices is a diagonal; one with `d` odd vertices on one side
        is left uncrossed by exactly the odd–odd diagonals with both ends on
        one side, `C(d,2) + C(k+2−d,2)` of them, and `k+2−d` even–even
        diagonals have that `d`; the sum over `d` is `(k+2)·C(k+2, 3)`) —
        `6, 24, 65, 144, 280` at `k = 1..5`.  At `k = 1, 2, 3` they match the
        finite zoo's `a3`/`a5`/`a7` generators one-to-one
        (the suite in the source repository).

        Inside one triangulation the balanced multisets do not form a free
        monoid once the triangulation holds two even–even diagonals `A1, A2`
        and two odd–odd ones `B1, B2` (possible from `k = 3`): the products `(A1B1)·(A2B2)` and `(A1B2)·(A2B1)` are
        single terms on ONE canonical label.  So a label is the multiset of
        diagonals, never a word in the pairs.

        `None` unless `E` is a nonzero power of the gauge letter.  Cached."""
        if not (isinstance(E, tuple) and len(E) == 2 and E[0] == ()
                and isinstance(E[1], int) and E[1] != 0):
            return None
        gens = self.__dict__.get("_centralizer_gens")
        if gens is None:
            cd = self._cd
            mixed = [g for g in cd._chords if cd._letter_mag(g) == 0]
            even = [g for g in cd._chords if cd._letter_mag(g) < 0]
            odd = [g for g in cd._chords if cd._letter_mag(g) > 0]
            gens = tuple(
                [(((t, i, 1),), 0) for (t, i) in mixed]
                + [(tuple(sorted(((a[0], a[1], 1), (b[0], b[1], 1)))), 0)
                   for a in even for b in odd if cd.q_commute(a, b)])
            self.__dict__["_centralizer_gens"] = gens
        return gens

    # -- rho^2 canonicalisation (drift quotient; same scheme as the retired
    #    predecessor, on the clean-gauge rho) --------------------------------------

    def rho_squared_is_identity(self):
        return False

    def _canonical_rho2_orbit_rep(self, label):
        factors, e = label
        if len(factors) == 0:
            return ((), e)
        if len(factors) == 1:
            from math import gcd
            a = factors[0][0]
            m0 = factors[0][2]
            s = self._cd._size[a]
            P = s // gcd(s, 2)
            members, cur = [], label
            for _ in range(P):
                members.append(cur)
                cur = self.rho(self.rho(cur))
            D = cur[1] - e
            if D == 0:
                return min(members, key=lambda m: (m[0][0][1], m[1]))
            istar = min(m[0][0][1] for m in members)
            estar = next(m[1] for m in members if m[0][0][1] == istar)
            return (((a, istar, m0),), estar % abs(D))
        return label

    # -- trace: Layer-1 reduction + the general-p closed-form seeds ----------

    def trace(self, a, K: int = 20):
        """`Tr`: the generic Layer-1 `ρ²`-cyclicity reduction over the cones,
        with every seed a closed form (`_trace_residual`), defined for every
        label to any order.

        ρ²-canonical input (manifest ρ²-invariance): `Tr` is ρ²-invariant and ρ
        is a clean basis permutation here, so the input's ρ²-orbit
        representative is reduced before tracing.

        **History (2026-09-23).**  Until this date the chord seeds came from
        per-family fitted closed forms that were wrong from about `𝖖²⁶–𝖖⁴³`;
        the reduction multiplies seeds by powers down to `𝖖⁻⁴⁵`, so those
        errors reached low orders — deep diameter powers at `k = 3` were
        refused, and 37 labels at `K = 16` across `k = 2, 3` came back silently
        wrong.  A `ρ²`-orbit retry worked around a position-dependence of the
        fitted form.  With the uniform rule `u1_pgon_layer2.singlet_chord_trace`
        (certified against the exact RG transport and the orthonormality
        bootstrap) the retry is gone.  The well-formedness guard stays as an
        ASSERTION — a Schur index is a power series, and leading
        multiplicativity `val Tr(gen^n) = n·val Tr(gen)` holds — which should
        never fire; if it does, it is a bug to report, not a regime."""
        a = self._canonical_rho2_orbit_rep(a)
        result = super().trace(a, K)
        self._guard_trace_wellformed(a, result)
        return result

    def _rho2_orbit_labels(self, a):
        """The other labels in `a`'s `ρ²`-orbit (the canonical representative
        itself excluded), in orbit order.  `Tr` is constant on this set."""
        out, cur = [], a
        for _ in range(2 * self._cd._H):
            cur = self.rho(self.rho(cur))
            if cur == a:
                break
            out.append(cur)
        return out

    @staticmethod
    def _coeff_is_zero(c) -> bool:
        if hasattr(c, "is_zero"):
            return c.is_zero()
        if hasattr(c, "terms"):
            return not any(v for v in c.terms.values())
        return c == 0

    @staticmethod
    def _single_generator_power(a):
        """If label `a` is a single generator raised to a power `n ≥ 1` — the
        v-tower `E^e` (`((), e)`) or a single chord `(((t, i, n),), 0)` —
        return `(gen1_label, n)`; else `None`.  These are exactly the
        generators carrying a leading-multiplicativity law."""
        factors, e = a
        if not factors:
            return None if e == 0 else (((), 1), abs(e))
        if len(factors) == 1 and e == 0:
            t, i, n = factors[0]
            if n >= 1:
                return ((((t, i, 1),), 0), n)
        return None

    def _gen1_trace_valuation(self, gen1):
        """Valuation of `Tr(gen1)` — cached; via the base trace (bypassing the
        guard: `n = 1` is low-order and reliable).  `None` if `Tr(gen1) = 0`
        (a flavour-charged generator)."""
        cache = self.__dict__.setdefault("_gen1_val_cache", {})
        if gen1 not in cache:
            r = super().trace(gen1, 4 * (self.k + 2) + 8)
            nz = [e for e, c in r.coeffs.items()
                  if not self._coeff_is_zero(c)]
            cache[gen1] = min(nz) if nz else None
        return cache[gen1]

    def _guard_trace_wellformed(self, a, result) -> None:
        """Raise `NotImplementedError` if `result = Tr(a)` is mathematically
        impossible.  Two provable invariants of a flavour-neutral trace (a
        Schur index): (1) it is a power series, so its valuation is `≥ 0`;
        (2) for a single generator raised to a power, leading multiplicativity
        forces `val(Tr(gen^n)) = n·val(Tr(gen))`."""
        nz = [e for e, c in result.coeffs.items()
              if not self._coeff_is_zero(c)]
        if not nz:
            return                                  # flavour-charged: Tr = 0
        val = min(nz)
        if val < 0:                                 # (1) Schur-index floor
            raise NotImplementedError(
                f"U1A1AoddKAlg(k={self.k}): Tr({a!r}) has a negative "
                f"𝖖-power (valuation {val}) — a Schur index cannot.  The "
                f"seeds are closed forms (u1_pgon_layer2.singlet_chord_trace), "
                f"so this is a bug to report; honest-fail rather than return "
                f"silently wrong.")
        gp = self._single_generator_power(a)
        if gp is not None:                          # (2) leading multiplicativity
            gen1, n = gp
            if n >= 2:
                d = self._gen1_trace_valuation(gen1)
                if d is not None and val != d * n:
                    raise NotImplementedError(
                        f"U1A1AoddKAlg(k={self.k}): Tr({a!r}) has "
                        f"valuation {val}, but leading multiplicativity "
                        f"requires {d * n} (= {n} x the valuation {d} of "
                        f"Tr(gen)).  The seeds are closed forms "
                        f"(u1_pgon_layer2.singlet_chord_trace), so this is a "
                        f"bug to report; honest-fail rather than return "
                        f"silently wrong.")

    def _seed_charge(self, seed):
        cd = self._cd
        n, MU = cd._n, cd._MU
        factors, e = seed
        g = [e * MU[j] for j in range(n)]
        for (a, i, ex) in factors:
            c = cd._charge((a, i))
            for j in range(n):
                g[j] += ex * c[j]
        return g

    def _orbit_has_physical(self, seed):
        cd = self._cd
        n, MU = cd._n, cd._MU
        factors = seed[0]
        if not factors:
            return True
        P = cd._size[factors[0][0]]
        cur = seed
        for _ in range(P + 1):
            cg = self._seed_charge(cur)
            if not any(cg[j] for j in range(n) if MU[j] == 0):
                return True
            cur = self.rho(self.rho(cur))
        return False

    def _lp_to_rps(self, lp, K):
        return RPowerSeries(
            self._R, {e: c for e, c in lp._coeffs.items() if 0 <= e <= K}, K)

    def _trace_residual(self, seed_label, K):
        """Same dispatch as the predecessor's `_trace_residual` (the `M(1,p)`
        singlet-character closed forms of `u1_pgon_layer2`, `p = k+2`),
        driven by this class's own charges and `ρ` — certified seed-by-seed
        against the retired predecessor."""
        factors, _e = seed_label
        g0 = self._seed_charge(seed_label)[0]
        if not self._orbit_has_physical(seed_label):
            return RPowerSeries(self._R, {}, K)
        if not factors:
            return self._lp_to_rps(_gp.tr_v_n(self.k + 2, g0, K), K)
        if len(factors) == 1 and factors[0][2] == 1:
            a, istar, _ = factors[0]
            base = (-1) ** istar * g0
            if a % 2 == 0:
                # Every physical chord is of even type a = 2j (the long chord
                # j = 1, the intermediate chords, the diameter j = (p-1)/2 at
                # odd k): ONE closed form, `singlet_chord_trace` — the
                # difference of two M(1, p) singlet module characters with a
                # fq-power prefactor, the gauged analogue of A1A2kKAlg's
                # T_a = (-1)^{m+1} fq^{-m} (chi_m - chi_{m+1}).  The character
                # index is n = -base for every type and every k (at k = 1,
                # where the long chord is the diameter, the predecessor's
                # base - 1 is its reflection n <-> -1-n, built into the rule).
                # Measured 2026-09-23: 40/40 seeds of types 2 and 4 at k = 4
                # against the orthonormality bootstrap, all positions.
                return self._lp_to_rps(
                    _gp.singlet_chord_trace(self.k + 2, a // 2, -base, K), K)
            if a == 1:
                # type 1 has no closed-form family of its own: its seeds are
                # gauge-charged (odd t => c0 = ±1) and vanish, which the
                # _orbit_has_physical gate above already returns; reaching
                # here means a flavour-neutral type-1 seed — a reduction gap.
                raise NotImplementedError(
                    f"_trace_residual: flavour-neutral type-1 seed "
                    f"{seed_label!r} (k={self.k})")
        raise NotImplementedError(
            f"_trace_residual: unreduced seed {seed_label!r} (k={self.k})")


def base_table_predict_odd(k: int):
    """The closed-form base product table for the u(1)-gauged
    `[A_1, A_{2k+1}]` family — the follow-up named (and left unwritten) in
    the design notes.  Returns
    `{(g, h): ("qc", cocycle)}` for q-commuting ordered letter pairs and
    `{(g, h): ("cross", CrossProductTerms)}` for crossing ones, in the clean
    geometric labels of this class — derived entirely from the
    even family's closed form through the embedding; no oracle, no frozen
    tables."""
    A = U1A1AoddKAlg(k)
    cd = A.cone_data()
    gens = [g for g in cd.mult_gens() if g[0] != 0]
    out = {}
    for g in gens:
        for h in gens:
            if g == h:
                continue
            if cd.q_commute(g, h):
                out[(g, h)] = ("qc", cd.cocycle(g, h))
            else:
                out[(g, h)] = ("cross", cd.cross_product(g, h))
    return out


def gauged_quiver_bps(k: int):
    """The BPS chart of the u(1)-gauged `[A_1, A_{2k+1}]`: the `A_{2k+2}`
    linear chain (`B[i][i+1] = +1`, the pairing of `chain_pairing`) with
    `2k+1` dynamical nodes and the last node frozen (the flavour direction).
    `charge_formula` and `cone_data()._chg` / `_MU` are written in these
    coordinates, with the gauge letter `E` of charge `+_MU` (certified by
    the source repository's tests against this chart, and by the skein
    atlas).  The chart is the witness the skein atlas compares against; it
    replaced the builder of the `k ≤ 4` BPS-bootstrap predecessor of this
    class (`u1a1aodd_general._build_bps`), which is retired."""
    from bps_kalgebra import BPSKAlgebra
    n = 2 * k + 2
    B = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        B[i][i + 1] = 1
        B[i + 1][i] = -1
    node_charges = [tuple(1 if j == i else 0 for j in range(n))
                    for i in range(2 * k + 1)]
    return BPSKAlgebra(pairing=B, node_charges=node_charges, verify="off")
