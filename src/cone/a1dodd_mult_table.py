"""
a1dodd_mult_table.py
====================

Cone-table extraction for the **odd** D-type Argyres–Douglas family
`A_𝖖([A_1, D_{2k+3}])` over `R(SU(2))`, read off the fast, BPS-free
`A1DoddRGKAlgebra(k)` oracle (whose auxiliary is the self-contained
`U1A1AoddKAlg(k).add_flavour(SU2ZPlusRing())`, `S_RG = E_𝖖(μL)·E_𝖖(μ⁻¹L)`).

Cracked structure (verified k=1,2; see the design notes)
---------------------------------------------------------------------
The SU(2) flavour `κ` is a **Z-form label index** in the oracle (that is the
form the RG finder consumes) and folds to an `RLaurent[SU2]` **coefficient**
only at the presentation boundary (via `r_label_decompose`, the single-irrep
lift coordinate — *not* the retired `_label_section_decompose`).  At the
**clean E-frame** `eE*(a,i)` of each chord — the unique E-power at which the
chord's monomial `RG` is a single term — the χ-stripped (`κ=0`) products form
the **same** clean cone skeleton as `U1A1AoddKAlg(k)`:

  * **q-commute graph == u1a1aodd** (k=1 72/72, k=2 380/380);
  * **cocycle == the A_{2k+2} chain pairing** of the clean charge
    `γ(a,i) = u1aodd_charge(a,i) + eE*(a,i)·μ`  (k=1 42/42, k=2 240/240);
  * **Plücker ≤ 2 terms** (clean cluster exchange), 0 undecodable daughters.

The only new content vs u1a1aodd is the **SU(2) character** riding the
doublet-dressed Plücker daughters (the two-`E_𝖖` doublet `S_RG`): 30/81
chord products at k=1 carry a non-trivial χ (the rest are χ₀ = the bare
u1a1aodd Plücker).

This module supplies the per-`k` extraction primitives; `a1dodd_kalg.py`
assembles them into the self-contained `A1DoddKAlg(ConeKAlgebra)`.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from zplus_ring import RLaurent


# ---------------------------------------------------------------------------
# Clean E-frames + chord charges
# ---------------------------------------------------------------------------

def clean_frames(T, ucd, scan=5):
    """`eE*(a,i)`: the smallest-|·| E-power at which the chord `(a,i)` has a
    single-term monomial `RG` (the un-dressed / chord-ray frame).  `T` is the
    `A1DoddRGKAlgebra(k)` oracle, `ucd` the `U1A1AoddKAlg(k)` cone data."""
    clean = {}
    for (a, i) in ucd._chords:
        for eE in sorted(range(-scan, scan + 1), key=abs):
            if len(T.RG(((((a, i, 1),), eE), 0)).terms) == 1:
                clean[(a, i)] = eE
                break
        else:
            raise RuntimeError(f"no clean E-frame for chord {(a, i)} (k scan={scan})")
    return clean


def chord_gamma(ucd, clean):
    """Clean B_GAUGED charge `γ(a,i) = u1aodd_charge(a,i) + eE*(a,i)·μ`."""
    chg, MU, n = ucd._chg, ucd._MU, ucd._n
    return {(a, i): tuple(chg[(a, i)][j] + clean[(a, i)] * MU[j] for j in range(n))
            for (a, i) in ucd._chords}


def clean_thresholds(T, ucd, scan=8):
    """`thr(a,i)`: the **minimum** E-power at which chord `(a,i)` has single-term
    monomial `RG` — the un-dressed threshold.  The clean (RG-monomial) region of
    a chord is the half-line `e_E ≥ thr(a,i)` (the dilog dressing `S_RG` carries
    only non-negative powers of the magnetic `L`, so dressing kicks in only
    *below* threshold).  A general cone monomial `(factors, e_E)` is clean iff
    `e_E ≥ Σ exp·thr(a,i)` (the threshold is additive over factors) — verified
    exact against the oracle's `RG` (k=1: 300/300)."""
    thr = {}
    for (a, i) in ucd._chords:
        cleans = [e for e in range(-scan, scan + 1)
                  if len(T.RG(((((a, i, 1),), e), 0)).terms) == 1]
        if not cleans:
            raise RuntimeError(f"no clean E-frame for chord {(a, i)} (scan={scan})")
        thr[(a, i)] = min(cleans)
    return thr


# ---------------------------------------------------------------------------
# Z-form → R-form fold via the lift coordinate (r_label_decompose)
# ---------------------------------------------------------------------------

def rform_fold(T, elt):
    """Fold an oracle Z-form `Element` (labels `((factors,e_E), κ)`) onto the
    flavour-stripped sections, with the SU(2) character `χ_κ` moved into an
    `RLaurent[SU2]` coefficient — using `r_label_decompose` (the single-irrep
    lift coordinate), the modern replacement for `_label_section_decompose`.

    Returns `dict[section, RLaurent]`, `section = ((factors,e_E), 0)`."""
    R = T.coefficient_ring()
    out: dict = {}
    for label, lp in elt.terms.items():
        sec, r_basis = T.r_label_decompose(label)        # (section, irrep label κ)
        chi = R.basis_element(r_basis)
        terms = {q: chi * c for q, c in lp._coeffs.items()}
        rl = RLaurent(R, terms)
        if rl.is_zero():
            continue
        out[sec] = out[sec] + rl if sec in out else rl
    return {s: v for s, v in out.items() if not v.is_zero()}


# ---------------------------------------------------------------------------
# Charge bookkeeping on the χ-stripped monomial part
# ---------------------------------------------------------------------------

def section_charge(ucd, section):
    """B_GAUGED charge of a folded section `((factors,e_E), 0)` in **bare**
    u1a1aodd coordinates (`Σ exp·u1aodd_charge(a,i) + e_E·μ`).  Used only for
    cross-checks — the daughter decomposition itself is read off the section's
    `factors` directly (see `to_chord_cone`)."""
    chg, MU, n = ucd._chg, ucd._MU, ucd._n
    (factors, eE), _flav = section
    g = [eE * MU[j] for j in range(n)]
    for (a, i, ex) in factors:
        c = chg[(a, i)]
        for j in range(n):
            g[j] += ex * c[j]
    return tuple(g)


def to_chord_cone(factors, e_E, clean):
    """Convert a **bare** u1a1aodd cone monomial `(factors, e_E)` to the
    clean-frame **chord** cone monomial: the chord factors are exactly
    `factors`, and the residual `E`-power is `e_E − Σ exp·eE*(a,i)` (each
    chord's clean frame is folded into the `E` torus).  Unambiguous — read off
    the factors directly, so it sidesteps the genuine B_GAUGED charge collisions
    between distinct cone monomials (e.g. clean `(1,4)·(2,2)` vs `(1,2)·E`)."""
    res = e_E
    for (a, i, ex) in factors:
        res -= ex * clean[(a, i)]
    return tuple(factors), res


# ---------------------------------------------------------------------------
# Top-level selection
# ---------------------------------------------------------------------------

def select_chords(k):
    """Build the per-`k` extraction context for `A1DoddKAlg(k)`.

    Returns a dict with:
      * `T`        — the `A1DoddRGKAlgebra(k)` oracle (BPS-free),
      * `ucd`      — the `U1A1AoddKAlg(k)` cone data (the χ-stripped skeleton),
      * `chords`   — sorted list of chord ids `(a, i)` (== u1a1aodd's),
      * `clean`    — `{(a,i): eE*}` clean E-frames,
      * `gamma`    — `{(a,i): γ}` clean B_GAUGED charges,
      * `decode`   — charge → (chord_factors, e_E) decoder,
      * `MU`, `n`  — the μ vector and B_GAUGED rank.
    """
    from a1dodd_rgkalgebra import A1DoddRGKAlgebra
    from u1a1aodd_kalg import U1A1AoddKAlg
    T = A1DoddRGKAlgebra(k)
    U = U1A1AoddKAlg(k)
    ucd = U.cone_data()
    clean = clean_frames(T, ucd)
    gamma = chord_gamma(ucd, clean)
    return {
        "T": T, "U": U, "ucd": ucd, "chords": list(ucd._chords),
        "clean": clean, "gamma": gamma,
        "MU": ucd._MU, "n": ucd._n,
    }


def clean_ray(clean, a, i, kappa=0):
    """The oracle label of chord `(a,i)` at its clean E-frame, SU(2) index
    `kappa` (default the flavour-singlet section rep `κ=0`)."""
    return ((((a, i, 1),), clean[(a, i)]), kappa)


if __name__ == "__main__":
    ctx = select_chords(1)
    print("A1Dodd k=1 = [A_1, D_5]")
    print("  chords:", ctx["chords"])
    print("  clean eE*:", ctx["clean"])
    T, ucd = ctx["T"], ctx["ucd"]
    # A doublet-dressed Plücker, folded to RLaurent[SU2]:
    clean = ctx["clean"]
    cr = lambda a, i: clean_ray(clean, a, i)
    prod = T.multiply(cr(1, 0), cr(2, 1))
    print("  L(1,0)·L(2,1) folded (RLaurent[SU2] coeffs):")
    for sec, coef in sorted(rform_fold(T, prod).items(), key=str):
        (factors, e_E), _ = sec
        print(f"    {sec[0]}  *  {coef}   -> chord-cone {to_chord_cone(factors, e_E, clean)}")
