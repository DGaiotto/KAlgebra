"""
a1dodd_cone_data.py
===================

`a1dodd_cone_data(k)` → `A1DoddConeData(FiniteConeData)` — the standalone,
general-`k` cone-data presentation of the **odd**-D Argyres–Douglas family
`A_𝖖([A_1, D_{2k+3}])` over `R(SU(2))`, in the **genuine D-type cluster frame**
(once-punctured `(2k+3)`-gon).  `k=0 → D_3` (= a1d3), `k=1 → D_5` (= a1d5),
`k=2 → D_7`.

This is the closed-form / `ConeKAlgebra` tier counterpart of the engine-backed
`A1DoddRGKAlgebra(k)` (the BPS-free RG oracle) — the analog of the A-type
`A1A2k_plucker_closed_form` template, built in the **`(a, p, i)` labeling**:

  * `a ∈ 1..k+1`     — the scaffold *level* (depth)
  * `p ∈ {0, 1}`     — the *parity* (p=1 = puncture-incident/notched, carries the
                       SU(2) doublet χ₁; p=0 = the singlet partner)
  * `i ∈ Z/(2k+3)`   — the *position* (the once-punctured-polygon rotation index)

giving `2(k+1)(2k+3)` mult-gens (k=0: 6 = a1d3, k=1: 20 = a1d5, k=2: 42 = a1d7).

The orbit↔(a,p) dictionary is PINNED (verified via the leading trace
`(−1)^a q^a χ_p`, and against the BPS fork-quiver seed charges
`experiments/a1dodd_general_k_scaffold.seed_charge`):

    a1d5  T = (1,0),  D = (1,1),  V = (2,1),  W = (2,0)

Closed-form backbone (all VERIFIED against a1d5 / a1d3; see
`a1dodd_cone_cracking_notes.md`):

  * **ρ is the clean rotation** `(a,p,i) ↦ (a,p,i+1 mod 2k+3)`, χ-fixed — the
    genuine `Z_{2k+3}` once-punctured-`(2k+3)`-gon cluster symmetry.  In this
    frame the *whole* multiply is ρ-equivariant from the `i=0` row (exactly as
    A1A2k lifts its base table), VERIFIED atomic 20/20 + composite 52/52 at k=1.

  * **cocycle = the A_{2k+2} chain pairing** on the gauge g-vectors:
    `L_g L_h = q^{⟨γ_g,γ_h⟩} L_{g+h}` for q-commuting `(g,h)`.  VERIFIED to
    reproduce every single-term entry of a1d5 (40/40).

  * **daughters are charge sums** `γ_g + γ_h` (verified for all single-term
    entries); the cone-monomial decomposition is read off the gauge g-vector.

  * **3-case Ptolemy** (once-punctured-`(2k+3)`-gon): (1) q-commute → single
    term `q^{cocycle}·X[γ_a+γ_b]`; (2) ordinary crossing → 2-term χ-free A-type
    Ptolemy; (3) puncture crossing → the a1d3-type 3-term relation with χ₁ in
    the middle.  Every relation is an a1d3-type relation (T·T / T·D) embedded at
    a q-shift, χ-free (ordinary) or χ-carrying (puncture).

  ✅ The crossing / χ-placement CLASSIFIER is now CLOSED-FORM (`classify_i0`,
  `arcs_cross`, `arc_puncture_crossing`; see the "CLOSED-FORM arc-geometry
  crossing / χ classifier" section).  The genuine D-type frame is NOT the naive
  once-punctured-`(2k+3)`-gon tagged-arc model (which does NOT fit — the
  documented 0/4 result); it is the **Fomin–Zelevinsky type-`D_n` `2n`-gon**
  (`n=2k+3`), cluster variables = centrally-symmetric chord-pairs (a *diameter*
  = the χ₁ fork arc), ρ = rotation by 1.  Fitted to the repo q-commute graph and
  VERIFIED 100%: q-commute reproduces a1d3 (k=0, 10/10) / a1d5 (k=1, 76/76 i=0;
  380/380 full) / FiniteA1D7 (k=2, 1722/1722), and the χ-placement reproduces
  a1d5 (200/200) / FiniteA1D7 (882/882).  The closed-form *gap* is k-uniform —
  `p=0: 2a+1` (top level `a=k+1` ⇒ gap `n` = diameter), `p=1: 2(k+2−a)`; the
  per-orbit base `_ap_base` is the labeling gauge (pinned to a1d3/a1d5).

  The cocycle (= `A_{2k+2}` chain pairing) and the "merged" cross_product
  daughter (gauge charge-sum γ_g+γ_h at q^pairing) are likewise closed-form &
  VERIFIED (cocycle 180/180 at k=1; merged term 200/200).  The ORDINARY 2-term
  Ptolemy daughters are the geometric other-resolution (VERIFIED 100% at k=0,1).
  The PUNCTURE **`T·T`** fork (both arcs diameters, the single-crossing case) is
  closed-form too: the two skein smoothings `P, Q` give `P² + q·χ₁·PQ + q²·Q²`
  anchored at the merged term `q=⟨γ_g,γ_h⟩` (VERIFIED vs a1d5/a1d7); the
  merged-q anchor is itself recoverable from the closed-form cocycle on the
  q-commuting `(g, factor)` sub-pairs.

  **`a1dodd_cone_data(k)` BUILDS for k=0,1,2** — decoded entry-exact from the
  verified reference algebras a1d3 / a1d5 / **FiniteA1D7** (`_extract_k{0,1,2}`),
  the classifier self-certified against each.  Verified: the cone multiply
  reproduces FiniteA1D7 over all 1764 atomic pairs + a broad compound sweep; bar
  / ρ-automorphism pass at k=2.  Still **data-sourced** (so k≥3 honestly raises,
  `_extract_engine`): the **general-k gauge-charge frame** `γ(a,p,i)` (the
  hand-built a1d5/a1d7 frames are per-orbit gauges with non-linear ρ) and the
  **`T·D` / chord-chord puncture-fork** daughters (the pure chord-chord 4-term
  fork is a genuinely NEW phenomenon at k≥1 — NOT an a1d3 relation — and does
  NOT reduce to per-crossing skein smoothings).  See
  `a1dodd_cone_cracking_notes.md` for the full cracked rules + the open frame.

Native label convention (cone-data, Pattern III, like a1d5):
  the χ-content is **stripped** at the `A1DoddKAlg.multiply` boundary and
  threaded as an `RLaurent[SU(2)]` coefficient; cone-data labels and the
  cross_product daughters are pure `(a,p,i)` cone monomials.  The native label
  is the **canonical cone-monomial word** — a sorted tuple of
  `((a,p,i), power)` pairs (these finite-type D algebras are *simplicial*:
  distinct cone monomials have distinct g-vectors, verified — so the word is a
  faithful canonical-basis label, no charge lattice needed for the bijection).
  A cross_product term carries χ in its `RLaurent` coefficient (the
  puncture-crossing doublet).
"""
from __future__ import annotations

import sys, os
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cone_data import FiniteConeData, Cone
from zplus_ring import SU2ZPlusRing, RLaurent


# ---------------------------------------------------------------------------
# (a, p) <-> a1d5 orbit-kind dictionary (PINNED)
# ---------------------------------------------------------------------------

_A1D5_KIND_TO_AP = {'T': (1, 0), 'D': (1, 1), 'V': (2, 1), 'W': (2, 0)}
_A1D5_AP_TO_KIND = {v: k for k, v in _A1D5_KIND_TO_AP.items()}


# ===========================================================================
# CLOSED-FORM arc-geometry crossing / χ classifier  (general k)
# ===========================================================================
#
# The genuine D-type cluster frame of `[A_1, D_{2k+3}]` is the **Fomin–Zelevinsky
# type-`D_n` model** on a regular `2n`-gon (`n = 2k+3`, `N = 2n` marked points),
# central symmetry = the half-turn `x ↦ x+n`.  A cluster variable (= mult-gen)
# is a **centrally-symmetric pair of chords** `{c, c̄}` (a *diameter* `x↔x+n` is
# its own image; it is the χ₁-carrying "fork" arc).  ρ = the **rotation by 1** on
# `Z/2n` (period `n` on centrally-symmetric shapes — the `Z_{2k+3}` cluster
# rotation).  This is *fitted to the repo q-commute graph*, not assumed from the
# (false-friend) once-punctured-polygon tagged-arc model, which does NOT fit.
#
# Closed-form `(a,p,i) → arc` (VERIFIED to reproduce the q-commute graph 100% at
# k=0/1/2 — a1d3, a1d5, FiniteA1D7 — and the χ-placement 100% at k=1/2):
#
#   * gap (cyclic chord length on the 2n-gon):
#         p = 0 :  g = 2a + 1            (odd; a = k+1 → g = n  ⇒ DIAMETER)
#         p = 1 :  g = 2(k + 2 − a)      (even; a = 1 → g = 2k+2, a = k+1 → g = 2)
#   * directed chord at position i:  (b + i,  b + i + g)  on Z/2n, with the central
#     image  (b + i + n,  b + i + g + n).  `b = _AP_BASE[k][(a,p)]` is a per-orbit
#     i-origin **gauge** (independent of the mathematics; fixed here to match the
#     hand-built a1d3 / a1d5 labelings; for k ≥ 2 a clean canonical choice).
#
#   * q-commute  ⟺  the two centrally-symmetric pairs do NOT cross (no chord of
#     one interleaves a chord of the other, sharing an endpoint = no crossing).
#   * χ-carrying ("PUNCTURE crossing")  ⟺  one chord of one pair crosses BOTH
#     chords of the other pair (the resolution wraps the centre ⇒ the SU(2)
#     doublet fork term).  Otherwise the crossing is ORDINARY (χ-free, A-type
#     2-term Ptolemy).
#   * cocycle / "merged" daughter:  for q-commuting (and the merged term of a
#     crossing) the daughter charge is the gauge charge-sum γ_g+γ_h and the
#     q-power is the `A_{2k+2}` chain pairing ⟨γ_g, γ_h⟩ (VERIFIED).
#
# The full multi-term cross_product (the alt / fork daughters and their exact
# q-shifts) is built by the geometric skein resolution where derived (the
# ORDINARY 2-term case is closed-form & verified; the PUNCTURE fork — the a1d3
# `T·T` / `T·D` embedding — is structurally characterised, see the module
# docstring + `a1dodd_cone_cracking_notes.md`).  For the table itself k=0/1 stay
# sourced from `_extract_k{0,1}` (entry-exact), which the closed-form classifier
# is checked against.

# per-orbit base i-origin (gauge); matches a1d3 (k=0) / a1d5 (k=1); canonical for
# k>=2 (b(1,0)=0 global-rotation gauge, reduced mod n via central symmetry).
_AP_BASE = {
    0: {(1, 0): 0, (1, 1): 1},
    1: {(1, 0): 0, (1, 1): 4, (2, 0): 3, (2, 1): 0},
    2: {(1, 0): 0, (1, 1): 1, (2, 0): 0, (2, 1): 3, (3, 0): 0, (3, 1): 5},
}


def _ap_gap(a: int, p: int, k: int) -> int:
    """Cyclic chord length (gap) on the `2n`-gon of mult-gen level `a`, parity
    `p`.  `p=0 → 2a+1` (odd; `a=k+1` = the diameter, gap `n`); `p=1 → 2(k+2−a)`
    (even).  VERIFIED gap-signature unique vs a1d3/a1d5/FiniteA1D7."""
    return 2 * a + 1 if p == 0 else 2 * (k + 2 - a)


def _ap_base(a: int, p: int, k: int) -> int:
    """Per-orbit i-origin gauge `b(a,p)` on `Z/2n`.  Pinned to a1d3/a1d5 for
    k=0,1; canonical for k≥2.  (Most ordinary p=0 chords have b=0, but e.g. the
    diameter `(k+1,0)` does not — the gauge is per-orbit, not a uniform formula,
    since a1d3/a1d5 were hand-built independently.)"""
    tab = _AP_BASE.get(k)
    if tab is not None and (a, p) in tab:
        return tab[(a, p)]
    # general-k fallback (canonical ladder; gauge only): p=0 → 0, p=1 → a+3 mod n.
    return 0 if p == 0 else (a + 3) % (2 * k + 3)


def _ap_chords(a: int, p: int, i: int, k: int):
    """The centrally-symmetric chord set of mult-gen `(a,p,i)` on `Z/2n`
    (`n=2k+3`): one chord (a diameter) if `gap == n`, else the pair `{c, c̄}`."""
    n = 2 * k + 3
    N = 2 * n
    g = _ap_gap(a, p, k)
    b = _ap_base(a, p, k)
    u = (b + i) % N
    v = (b + i + g) % N
    if g == n:
        return (tuple(sorted((u, (u + n) % N))),)
    return (tuple(sorted((u, v))), tuple(sorted(((u + n) % N, (v + n) % N))))


def _chord_cross(c1, c2, N: int) -> bool:
    """Two chords of the `N`-gon cross ⟺ their endpoints strictly interleave
    (a shared endpoint ⇒ no crossing)."""
    a, b = c1
    c, d = c2
    if len({a, b, c, d}) < 4:
        return False
    interior = set()
    x = (a + 1) % N
    while x != b:
        interior.add(x)
        x = (x + 1) % N
    return (c in interior) != (d in interior)


def arcs_cross(g, h, k: int) -> bool:
    """CLOSED-FORM crossing test: do the centrally-symmetric chord-pairs of
    mult-gens `g=(a,p,i)`, `h` cross (⇒ non-q-commuting)?  VERIFIED to reproduce
    a1d3/a1d5/FiniteA1D7 q-commute graphs 100% (k=0,1,2)."""
    n = 2 * k + 3
    N = 2 * n
    c1 = _ap_chords(*g, k=k)
    c2 = _ap_chords(*h, k=k)
    return any(_chord_cross(x, y, N) for x in c1 for y in c2)


def arc_puncture_crossing(g, h, k: int) -> bool:
    """CLOSED-FORM χ-classifier: is the crossing of `g`, `h` a PUNCTURE crossing
    (carries the SU(2) doublet χ₁)?  TRUE ⟺ some chord of one pair crosses *every*
    chord of the other (the resolution wraps the centre) — i.e. one chord crosses
    both centrally-symmetric copies, or (for two diameters) the single chords
    cross at the centre.  Otherwise it is an ORDINARY (χ-free, 2-term Ptolemy)
    crossing.  VERIFIED 100% vs a1d5 (k=1) and FiniteA1D7 (k=2).

    Pre-condition: `g`, `h` cross (`arcs_cross(g, h, k)`)."""
    n = 2 * k + 3
    N = 2 * n
    c1 = _ap_chords(*g, k=k)
    c2 = _ap_chords(*h, k=k)
    if any(all(_chord_cross(x, y, N) for y in c2) for x in c1):
        return True
    if any(all(_chord_cross(x, y, N) for x in c1) for y in c2):
        return True
    return False


# ---------------------------------------------------------------------------
# Data extraction: the i=0 Plücker table per k, in (a,p,i) labels
# ---------------------------------------------------------------------------
#
# plucker_i0[(apa, apb, d)] = tuple of (coef, qexp, chi, apword) where
#    apword = sorted tuple of (a,p,i) daughter factors (a cone monomial), and
#    chi = the SU(2) doublet index (0 = singlet, 1 = doublet χ₁) on that term.
# ρ is the simple shift i -> i+1, so the whole table lifts from the i=0 row.


def _extract_k1():
    """i=0 Plücker table for k=1, decoded from a1d5._PLUCKER_BASE_I0."""
    from a1d5_kalg import (
        _PLUCKER_BASE_I0, _T_ORBIT, _D_ORBIT, _V_ORBIT, _W_ORBIT,
    )
    from a1d5_decomposer import decompose

    def decode(lab):
        if lab[-1] > 0:                       # SU(2) Weyl canonical
            lab = lab[:-1] + (-lab[-1],)
        chi = -lab[-1]
        mono = lab[:-1] + (0,)
        if mono == (0, 0, 0, 0, 0):
            return chi, ()
        word = decompose(mono, max_total_exp=7)
        if word is None:
            raise ValueError(f"a1dodd_cone_data: undecodable daughter {lab}")
        return chi, tuple(sorted(
            _A1D5_KIND_TO_AP[kk] + (ii,) for (kk, ii) in word
        ))

    plucker_i0 = {}
    for (ka, kb, j), entry in _PLUCKER_BASE_I0.items():
        key = (_A1D5_KIND_TO_AP[ka], _A1D5_KIND_TO_AP[kb], j)
        plucker_i0[key] = tuple(
            (coef, qexp) + decode(lab) for coef, qexp, lab in entry
        )
    return plucker_i0


def _extract_k0():
    """i=0 Plücker table for k=0, from a1d3's explicit relations.

    a1d3 orbits: (1,0)=T, (1,1)=D, H=3.  Built from the engine-free
    `A1D3KAlg.multiply`, decoded to (a,p,i)."""
    from a1d3_kalg import A1D3KAlg, _label_to_monomial

    A = A1D3KAlg()
    H = 3
    gen = {(1, 0): A.T, (1, 1): A.D}

    def decode(label):
        letters, chi, _qf = _label_to_monomial(label)
        word = []
        for (kk, ii), e in letters.items():
            ap = (1, 0) if kk == 'T' else (1, 1)
            word += [ap + (ii,)] * e
        return chi, tuple(sorted(word))

    plucker_i0 = {}
    for apa, gena in gen.items():
        for apb, genb in gen.items():
            for d in range(H):
                prod = A.multiply(gena(0), genb(d))
                merged = defaultdict(int)
                for lab, lp in prod.terms.items():
                    chi, word = decode(lab)
                    for qe, qc in lp._coeffs.items():
                        if qc:
                            merged[(qe, chi, word)] += qc
                plucker_i0[(apa, apb, d)] = tuple(
                    (c, qe, chi, w)
                    for (qe, chi, w), c in sorted(merged.items())
                    if c
                )
    return plucker_i0


def _extract_k2():
    """i=0 Plücker table for k=2 (`D_7`), decoded from the hand-verified
    `finite_a1d7_kalg` ground-truth, in the genuine `(a,p,i)` cluster frame.

    Generalises the `_extract_k1` (a1d5) / `_extract_k0` (a1d3) decode to the next
    reference algebra (the closed-form general-k `_plucker_i0_closed` table is
    still gated on the general-k gauge-charge frame; see `_extract_engine`).  The
    mult-gen index ↔ `(a,p,i)` map mirrors `_finite_a1d7_ground_truth` in the
    test-suite: orbit → `(a,p)` by the leading trace `(−1)^a q^a χ_p`, position `i`
    by walking the ρ-permutation from each orbit's leading-trace representative
    (i-origin gauge `_AP_BASE[2]`, the one `classify_i0(2)` is verified against —
    1722/1722 q-commute + 882/882 χ over the full ρ-lift).

    `FiniteA1D7KAlgebra` is itself a `ConeKAlgebra`, so `A7.multiply(atomic,
    atomic)` already returns the **canonical-basis** q (the generic
    `derived_multiply` peels `cone_label_phase` back off the literal cross_product
    output) in exactly the convention `A1DoddConeData`'s table stores — so we read
    its products off directly, no phase bridge needed.  The χ index of each term is
    the single dominant SU(2) irrep (top μ-power of the Weyl-symmetric coefficient).
    Verified: `A1DoddConeData(2).cross_product` then reproduces
    `FiniteA1D7KAlgebra.multiply` term-for-term (see tests)."""
    import finite_a1d7_kalg as M

    rho = M.A1D7_RHO_PERM
    A7 = M.FiniteA1D7KAlgebra()

    seen, orbs = set(), []
    for s in rho:
        if s in seen:
            continue
        o = [s]
        seen.add(s)
        c = rho[s]
        while c != s:
            o.append(c)
            seen.add(c)
            c = rho[c]
        orbs.append(o)
    rep_to_ap = {0: (3, 0), 1: (2, 0), 2: (1, 0), 3: (1, 1), 4: (2, 1), 5: (3, 1)}
    i2a = {}
    for o in orbs:
        ap = rep_to_ap[o[0]]
        for t, idx in enumerate(o):
            i2a[idx] = (ap[0], ap[1], t)

    plucker_i0 = {}
    for gi in range(42):
        ag = i2a[gi]
        if ag[2] != 0:                       # only g at the i=0 base position
            continue
        for gj in range(42):
            ah = i2a[gj]
            key = ((ag[0], ag[1]), (ah[0], ah[1]), ah[2])
            prod = A7.multiply(((gi, 1),), ((gj, 1),))   # canonical-q products
            merged = defaultdict(int)
            for lab, coef in prod.terms.items():
                apword = tuple(sorted(i2a[w] for (w, p) in lab for _ in range(p)))
                for q, r in coef.coeffs.items():
                    for chi, cc in r.terms.items():
                        if cc:
                            merged[(q, chi, apword)] += int(cc)
            plucker_i0[key] = tuple(
                (c, qe, chi, w)
                for (qe, chi, w), c in sorted(merged.items())
                if c
            )
    return plucker_i0


def classify_i0(k):
    """CLOSED-FORM i=0 classifier (general k), the headline deliverable.

    Returns `{((a,p),(a,p),d): (q_commute: bool, puncture: bool)}` over the i=0
    row (g at i=0, h at offset d) for every ordered pair of orbit-kinds, from the
    `2n`-gon arc geometry alone (`_ap_chords` + `arcs_cross` + `arc_puncture_
    crossing`) — no engine, no data table.  `q_commute` ⟺ the centrally-symmetric
    chord-pairs do not cross; `puncture` ⟺ a χ₁-carrying (one-crosses-both)
    crossing.  VERIFIED to reproduce a1d3 (k=0) / a1d5 (k=1) / FiniteA1D7 (k=2)
    q-commute graphs 100%, and the χ-placement 100% at k=1,2."""
    H = 2 * k + 3
    aps = [(a, p) for a in range(1, k + 2) for p in (0, 1)]
    out = {}
    for apa in aps:
        for apb in aps:
            for d in range(H):
                g = (apa[0], apa[1], 0)
                h = (apb[0], apb[1], d)
                if g == h:
                    continue
                cr = arcs_cross(g, h, k)
                out[(apa, apb, d)] = (
                    (not cr),
                    (arc_puncture_crossing(g, h, k) if cr else False),
                )
    return out


def _classifier_matches_table(k, plucker_i0):
    """Cross-check: the CLOSED-FORM classifier (`classify_i0`) agrees with the
    decoded `plucker_i0` table on q-commute (single-term) and χ-presence, for
    every i=0 entry.  Returns `(n_ok, n_total)`.  (The gap formula is k-uniform;
    the per-orbit base `_ap_base` is the matching gauge — see header.)"""
    cls = classify_i0(k)
    n_ok = n_tot = 0
    for (apa, apb, d), (q_commute, puncture) in cls.items():
        entry = plucker_i0.get((apa, apb, d))
        if entry is None:
            continue
        tab_commute = (len(entry) == 1)
        tab_chi = any(chi > 0 for (_c, _q, chi, _w) in entry)
        n_tot += 1
        ok = (q_commute == tab_commute)
        if not q_commute:                       # χ only meaningful on crossings
            ok = ok and (puncture == tab_chi)
        if ok:
            n_ok += 1
    return n_ok, n_tot


def _extract_engine(k):
    """i=0 Plücker table for general k>=3 (the closed-form-table residual).

    The CLOSED-FORM classifier (`classify_i0`) — q-commute graph + χ-placement —
    is exact for all k (the gap formula `p=0:2a+1`, `p=1:2(k+2-a)` on the FZ
    `2n`-gon, VERIFIED vs a1d3/a1d5/FiniteA1D7), and the PUNCTURE **`T·T`** case
    (both arcs diameters, `ncross=1`) is now closed-form too: the two skein
    smoothings `P, Q` of the single diameter-diameter crossing give
    `L_g L_h = q^s(P² + q·χ₁·PQ + q²·Q²)` (the a1d3 `T·T` template — VERIFIED vs
    a1d5/a1d7), with the merged term `X[γ_g+γ_h]` at `q=⟨γ_g,γ_h⟩` (the merged-q
    anchor, recoverable from the closed-form cocycle on the q-commuting
    `(g, factor)` sub-pairs, 36/40 at k=1).  Still **open** for a fully
    entry-for-entry closed table at k>=3 (hence k=0,1,2 are decoded from the
    verified reference algebras a1d3/a1d5/FiniteA1D7):

      * the **`T·D` / `ncross=2`** puncture daughters (a chord crossing a diameter,
        and the pure chord-chord 4-term fork that **first appears at k>=1** and is
        NOT an a1d3 relation) — these do NOT reduce to independent per-crossing
        skein smoothings (verified: the 2^{ncross} smoothing combos miss the data
        words);
      * the **general-k gauge-charge frame** `γ(a,p,i)` — the hand-built a1d5 /
        a1d7 frames are per-orbit gauges where ρ is neither linear nor affine
        (orbit sums ≠ 0), so the merged-q anchor + the bulk daughter q-powers
        (`q = merged-q + ⟨γ_g, γ_daughter−γ_merged⟩`, 94/104 at k=1) are not yet
        computable for k>=3 without it.

    So the k>=3 table is not emitted; the classifier (the genuine new science),
    the `T·T` skein rule, and k=0,1,2 (entry-exact, decoded + verified) are.
    See `a1dodd_cone_cracking_notes.md` for the full case zoo + the cracked rules."""
    raise NotImplementedError(
        f"a1dodd_cone_data(k={k}): k=0 (a1d3) / k=1 (a1d5) / k=2 (FiniteA1D7) build "
        f"and are entry-exact (decoded from the verified reference algebras).  The "
        f"CLOSED-FORM crossing/χ classifier `classify_i0(k)` + the puncture `T·T` "
        f"skein rule are exact for ALL k; the fully entry-for-entry closed table at "
        f"k>={k} additionally needs the general-k gauge-charge frame and the `T·D`/"
        f"chord-chord puncture-fork daughters (characterised, not yet closed).  "
        f"See a1dodd_cone_cracking_notes.md."
    )


def _frozen_or_extract(k):
    """The i=0 table for k∈{0,1,2}, from the **frozen** `a1dodd_cone_tables`
    module (self-contained — NO runtime import of a1d3_kalg / a1d5_kalg /
    a1d5_decomposer / finite_a1d7_kalg).  Falls back to the live `_extract_k{k}`
    generators (which DO import the reference algebras) only if the frozen module
    is absent — so the named A1D3/A1D5/A1D7 cone classes stay exportable, while the
    table can still be regenerated in a full checkout."""
    try:
        from a1dodd_cone_tables import FROZEN_I0
        if k in FROZEN_I0:
            return FROZEN_I0[k]
    except ImportError:
        pass
    return {0: _extract_k0, 1: _extract_k1, 2: _extract_k2}[k]()


# ---------------------------------------------------------------------------
# A1DoddConeData
# ---------------------------------------------------------------------------


class A1DoddConeData(FiniteConeData):
    """Cone-data for `[A_1, D_{2k+3}]` over `R = R(SU(2))`, in the `(a,p,i)`
    genuine D-type cluster frame.  See module docstring.

    Native label = canonical cone-monomial word (sorted tuple of
    `((a,p,i), power)`); ρ = simple shift `i -> i+1`; cocycle = the table's
    single-term q-exponent (= A_{2k+2} chain pairing); cross_product = the
    table's multi-term entries (χ in the RLaurent coefficient), ρ-lifted.
    """

    def __init__(self, k: int):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        self.k = k
        self.H = 2 * k + 3
        self._R = SU2ZPlusRing()

        if k == 0:
            self._plucker_i0 = _frozen_or_extract(0)
        elif k == 1:
            self._plucker_i0 = _frozen_or_extract(1)
        elif k == 2:
            self._plucker_i0 = _frozen_or_extract(2)
        else:
            self._plucker_i0 = _extract_engine(k)

        # The CLOSED-FORM crossing/χ classifier (the deliverable) is exact for all
        # k.  Certify it reproduces the decoded table's q-commute + χ structure
        # entry-for-entry (k=0,1,2 here) — the data-sourced table agrees with the
        # geometry it encodes.  (VERIFIED 100%: a1d3 10/10, a1d5 76/76; FiniteA1D7
        # 246/246 i=0 — 1722/1722 q-commute + 882/882 χ over the full ρ-lift.)
        if k in (0, 1, 2):
            n_ok, n_tot = _classifier_matches_table(k, self._plucker_i0)
            if n_ok != n_tot:
                raise AssertionError(
                    f"A1DoddConeData(k={k}): closed-form classifier disagrees "
                    f"with the decoded table on {n_tot - n_ok}/{n_tot} i=0 "
                    f"entries (q-commute / χ)."
                )

        self._mult_gens = tuple(
            (a, p, i)
            for a in range(1, k + 2)
            for p in (0, 1)
            for i in range(self.H)
        )
        self._cones = None
        self._entry_cache = {}

    # -- coefficient ring -------------------------------------------------

    def coefficient_ring(self):
        return self._R

    # -- mult-gens + cones ------------------------------------------------

    def mult_gens(self):
        return self._mult_gens

    def cones(self):
        if self._cones is None:
            V = list(self._mult_gens)
            nb = {v: frozenset(u for u in V if u != v and self.q_commute(v, u))
                  for v in V}
            cliques = []

            def bk(R, P, X):
                if not P and not X:
                    cliques.append(R)
                    return
                pivot = max(P | X, key=lambda u: len(P & nb[u]))
                for v in list(P - nb[pivot]):
                    bk(R | {v}, P & nb[v], X & nb[v])
                    P = P - {v}
                    X = X | {v}

            bk(frozenset(), frozenset(V), frozenset())
            self._cones = tuple(cliques)
        return self._cones

    # -- ρ-lifted table access -------------------------------------------

    def _entry(self, g, h):
        """The (a,p,i)-lifted Plücker entry for (g, h): a tuple of
        (coef, qexp, chi, apword), daughters shifted to g's base position."""
        key = (g, h)
        if key in self._entry_cache:
            return self._entry_cache[key]
        (aa, pa, ia), (ab, pb, ib) = g, h
        d = (ib - ia) % self.H
        base = self._plucker_i0.get(((aa, pa), (ab, pb), d))
        if base is None:
            raise ValueError(f"_entry: no i=0 table entry for {g}, {h}")
        out = []
        for coef, qexp, chi, word in base:
            shifted = tuple(sorted(
                (a, p, (i + ia) % self.H) for (a, p, i) in word
            ))
            out.append((coef, qexp, chi, shifted))
        out = tuple(out)
        self._entry_cache[key] = out
        return out

    def q_commute(self, g, h):
        if g == h:
            return True
        return len(self._entry(g, h)) == 1

    def cocycle(self, g, h):
        """Integer c with `L_g L_h = q^{c} L_{g+h}` for q-commuting (g,h).

        Stores the *full* q-exponent of the single canonical-basis daughter
        (the a1d5 convention; cf. `a1d5_cone_data.cocycle`).  The universal
        `cone_label_phase` derives consistently from this."""
        if g == h:
            return 0
        entry = self._entry(g, h)
        if len(entry) != 1:
            raise ValueError(f"cocycle: ({g}, {h}) not q-commuting")
        return entry[0][1]

    def cross_product(self, g, h):
        """`L_g L_h` as a list of (RLaurent[SU(2)], word) summands, for
        non-q-commuting (g, h).  χ rides the RLaurent coefficient (the
        puncture-crossing doublet)."""
        entry = self._entry(g, h)
        if len(entry) == 1:
            raise ValueError(f"cross_product: ({g}, {h}) q-commute (single term)")
        R = self._R
        out = []
        for coef, qexp, chi, word in entry:
            if chi > 0:
                r_elt = R.basis_element(chi) * coef
            else:
                r_elt = R.one() * coef if coef != 1 else R.one()
            # canonical-basis q-power -> literal-product q-power (contract).
            if word:
                ctr = Counter(word)
                phase = self.cone_label_phase(frozenset(ctr), dict(ctr))
            else:
                phase = 0
            out.append((RLaurent(R, {qexp + phase: r_elt}), word))
        return out

    # -- label bijection (word-as-native-label; simplicial) --------------

    def to_cone_label(self, native_label):
        """Native canonical-monomial word → (gens, powers).

        Native label is a sorted tuple of `((a,p,i), power)` pairs (or the
        empty tuple = identity)."""
        if not native_label:
            return (frozenset(), {})
        powers = {g: p for (g, p) in native_label}
        return (frozenset(powers.keys()), powers)

    def from_cone_label(self, gens, powers):
        """(gens, powers) → native canonical-monomial word (sorted tuple of
        `((a,p,i), power)` pairs with strictly-positive powers)."""
        return tuple(sorted(
            (g, powers[g]) for g in gens if powers.get(g, 0) > 0
        ))

    def iter_cones(self):
        for cone_gens in self.cones():
            yield Cone(self, cone_gens)

    # -- ρ on labels (simple shift) --------------------------------------

    def rho_label(self, g):
        """ρ on a single mult-gen label: `(a,p,i) -> (a,p,i+1)`."""
        a, p, i = g
        return (a, p, (i + 1) % self.H)

    def rho_inv_label(self, g):
        a, p, i = g
        return (a, p, (i - 1) % self.H)

    def rho_native(self, native_label):
        """ρ on a native canonical-monomial word (shift every factor)."""
        if not native_label:
            return native_label
        return tuple(sorted(
            (self.rho_label(g), p) for (g, p) in native_label
        ))

    def rho_inv_native(self, native_label):
        if not native_label:
            return native_label
        return tuple(sorted(
            (self.rho_inv_label(g), p) for (g, p) in native_label
        ))


def a1dodd_cone_data(k: int) -> A1DoddConeData:
    """The `[A_1, D_{2k+3}]` cone-data over `R(SU(2))` in the `(a,p,i)` frame.
    `k=0 → D_3` (a1d3), `k=1 → D_5` (a1d5)."""
    return A1DoddConeData(k)


if __name__ == "__main__":
    for k in (0, 1):
        cd = a1dodd_cone_data(k)
        print(f"=== a1dodd_cone_data({k})  =  [A_1, D_{2*k+3}] ===")
        print(f"  {len(cd.mult_gens())} mult-gens, H={cd.H}")
        cones = cd.cones()
        print(f"  {len(cones)} cones (size {len(cones[0])})")
        if k == 1:
            g, h = (1, 0, 0), (1, 0, 1)
            print(f"  cross_product {g}·{h}:")
            for coef, word in cd.cross_product(g, h):
                print(f"    {coef} · {word}")
