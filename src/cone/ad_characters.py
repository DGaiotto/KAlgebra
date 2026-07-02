"""Closed-form AD chiral-algebra characters for the finite zoo.

Exact closed forms for the elementary (Layer-2) seed traces of
finite-type entries, independent of the embedded-BPS Nahm engine:

* ``su2_m43_trace_table`` — the three su(2)_{−4/3} admissible-module
  characters of [A₁,A₃] = the a3/hexagon entry (z-refined; admissible
  reps Φ₀ = [−4/3,0], Φ₁ = [−2/3,−2/3], Φ₂ = [0,−4/3]), assembled into
  the three line-defect Schur traces

      I    = χ₀,
      I_A  = q^{−1/2} z⁻¹ (−χ₁ + χ₂),
      I_B  = q^{−1/2} (χ₀ − χ₁ + z⁻²χ₂).

  χ₁, χ₂ individually have infinite z-support per q-order (the
  horizontal su(2) acts non-integrably on admissible modules); only
  the trace combinations are finite, so they are computed as exact
  numerator arithmetic followed by exact Laurent division by (1−z⁻²)
  — no z-truncation anywhere.

* ``m2_2np3_character`` — all n+1 Virasoro M(2,2n+3) characters in
  the theta/eta vortex-residue form (pentagon n=1, heptagon n=2,
  nonagon n=3, …).

Conventions (the paper↔repo dictionary): ``q_paper = 𝖖²`` and
``z² = μ`` with charge lattices identified verbatim; a line defect may
carry its own flavour charge, so each zoo seed matches a character
combination up to a per-seed monomial ``z^c`` (``A3_SEED_DICTIONARY``,
re-derived by the tests from leading terms).

Provenance: the su(2)_{−4/3} characters and vortex residues are
transcribed from standard results on admissible-level sl(2) characters.
Certification: bulk agreement with the frozen a3 table, full-row
agreement with an embedded-quiver oracle at enlarged window (an
author-side computation not included in this repository), internal
ρ²-cyclicity, and the M(2,5)/M(2,7) match of the pentagon/heptagon
frozen tables.

Why this module exists (the trapezoid defect): the flavoured BPS trace
at q-window K is exact only on a trapezoid in (q-order, μ-charge) —
flavour tails at orders near K need internal q-orders beyond K and
come out clipped or with spurious edge terms (observed: an early
frozen a3 table violated the cyclicity identities T₀ = T₃, T₄ = μ·T₀
from 𝖖²³ on; an oracle run at K=24 corrupts the 𝖖²⁴ row down to
|μ| ≥ 3, while at K=44 it reproduces these characters exactly).  The
closed forms have no window at all.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# exact (paper-q, z) series engine
# ---------------------------------------------------------------------------
# A series is a list (q-power 0..kq) of dicts {z-exponent: int coeff}.


def _zero(kq):
    return [{} for _ in range(kq + 1)]


def _add(s, k, j, c):
    if k <= len(s) - 1 and c:
        s[k][j] = s[k].get(j, 0) + c
        if not s[k][j]:
            del s[k][j]


def _mul(a, b):
    kq = len(a) - 1
    out = _zero(kq)
    for ka, da in enumerate(a):
        if not da:
            continue
        for kb in range(kq + 1 - ka):
            db = b[kb]
            if not db:
                continue
            tgt = out[ka + kb]
            for ja, ca in da.items():
                for jb, cb in db.items():
                    j = ja + jb
                    tgt[j] = tgt.get(j, 0) + ca * cb
    for d in out:
        for j in [j for j, c in d.items() if not c]:
            del d[j]
    return out


def _geom(s, n, a):
    """Multiply by 1/(1 − z^a q^n) = Σ_k z^{ak} q^{nk} (n ≥ 1)."""
    kq = len(s) - 1
    g = _zero(kq)
    k = 0
    while n * k <= kq:
        g[n * k][a * k] = 1
        k += 1
    return _mul(s, g)


def _div_1mzm2(s):
    """Exact division by (1 − z⁻²) per q-order; raises if inexact."""
    out = _zero(len(s) - 1)
    for k, d in enumerate(s):
        if not d:
            continue
        y = {}
        for j in range(max(d) + 2, min(d) - 1, -1):
            c = d.get(j, 0) + y.get(j + 2, 0)
            if c:
                y[j] = c
        chk = {}
        for j, c in y.items():
            chk[j] = chk.get(j, 0) + c
            chk[j - 2] = chk.get(j - 2, 0) - c
        if {j: c for j, c in chk.items() if c} != d:
            raise ValueError(f"(1−z⁻²) ∤ q^{k} coefficient")
        out[k] = y
    return out


def _shift_z(s, c):
    return [{j + c: v for j, v in d.items()} for d in s]


def _sub(a, b):
    out = [dict(d) for d in a]
    for k, d in enumerate(b):
        for j, c in d.items():
            out[k][j] = out[k].get(j, 0) - c
            if not out[k][j]:
                del out[k][j]
    return out


def _addS(a, b):
    return _sub(a, [{j: -v for j, v in d.items()} for d in b])


# ---------------------------------------------------------------------------
# su(2)_{−4/3} ([A₁,A₃] = a3/hexagon)
# ---------------------------------------------------------------------------

def _su2_m43_qz(kq):
    """(χ₀, I_A·q^{1/2}, I_B·q^{1/2}) as exact (paper-q, z) series."""
    n0, n1, n2 = _zero(kq), _zero(kq), _zero(kq)
    m = 0
    while 3 * m * (m + 1) // 2 <= kq:
        for j in range(-m, m + 1):
            _add(n0, 3 * m * (m + 1) // 2, 2 * j, (-1) ** m)
        m += 1
    _add(n1, 0, 0, 1)
    _add(n2, 0, 0, 1)
    n = 1
    while n * (3 * n - 1) // 2 <= kq:
        sgn = (-1) ** n
        _add(n1, n * (3 * n - 1) // 2, -2 * n, sgn)
        _add(n1, n * (3 * n + 1) // 2, 2 * n, sgn)
        # χ₂ = z ↔ z⁻¹ in the numerator sum ONLY (prefactor unchanged)
        _add(n2, n * (3 * n - 1) // 2, 2 * n, sgn)
        _add(n2, n * (3 * n + 1) // 2, -2 * n, sgn)
        n += 1
    invD = _zero(kq)
    invD[0][0] = 1
    for n in range(1, kq + 1):
        for a in (0, 2, -2):
            invD = _geom(invD, n, a)
    chi0 = _mul(n0, invD)
    ia = _shift_z(_mul(_div_1mzm2(_sub(n2, n1)), invD), -1)
    ib = _mul(_addS(n0, _div_1mzm2(_sub(_shift_z(n2, -2), n1))), invD)
    return chi0, ia, ib


def su2_m43_trace_table(K: int) -> dict:
    """``{"I": …, "I_A": …, "I_B": …}`` through 𝖖-order ``K``, each as
    ``{𝖖-order: {z-exponent: int}}`` (z² = μ; I at even orders, I_A/I_B
    at odd orders from the q^{−1/2} prefactor)."""
    kq = K // 2 + 1
    chi0, ia, ib = _su2_m43_qz(kq)
    out = {"I": {}, "I_A": {}, "I_B": {}}
    for k, d in enumerate(chi0):
        if d and 2 * k <= K:
            out["I"][2 * k] = dict(d)
    for name, s in (("I_A", ia), ("I_B", ib)):
        for k, d in enumerate(s):
            if d and 2 * k - 1 <= K:
                out[name][2 * k - 1] = dict(d)
    return out


#: zoo mg-seed ↔ character dictionary for the a3 entry: per-seed
#: (combination, z-shift), i.e. ``T_seed = z^c · I_X(q→𝖖², z²→μ)``.
#: Tied to the current `finite_a3_kalg` mg indexing; re-derived from
#: leading terms by the tests.  ρ²-orbits {0,3,4}
#: and {1,2,5} carry the expected unit-character (μ^δ) twists.
A3_SEED_DICTIONARY = {
    0: ("I_B", 0),
    1: ("I_A", -1),
    2: ("I_A", 1),
    3: ("I_B", 0),
    4: ("I_B", 2),
    5: ("I_A", 1),
}


def a3_elem_entry(K: int) -> dict:
    """The frozen-format ``ELEM_TRACE_DATA['a3']`` entry, generated
    from the closed-form characters (μ-exponents = z-exponents/2 after
    the per-seed torsor shift; integrality asserted)."""
    tab = su2_m43_trace_table(K)

    def to_mu(series, shift):
        out = {}
        for k, d in series.items():
            row = {}
            for j, c in d.items():
                jj = j + shift
                if jj % 2:
                    raise ValueError(
                        f"odd μ-doubled exponent {jj} at 𝖖^{k}")
                row[(jj // 2,)] = c
            if row:
                out[k] = row
        return out

    return {
        "K": K,
        "flavor": "u1",
        "fold": "none",
        "identity": to_mu(tab["I"], 0),
        "orbits": {
            seed: to_mu(tab[name], c)
            for seed, (name, c) in A3_SEED_DICTIONARY.items()
        },
    }


# ---------------------------------------------------------------------------
# Virasoro M(2,2n+3) characters (vortex-residue theta/eta form)
# ---------------------------------------------------------------------------

def m2_2np3_character(n: int, rbar: int, K: int) -> dict:
    """``χ^{(2,2n+3)}_{(1, rbar+1)}`` as ``{𝖖-order: int}`` through
    𝖖-order ``K`` (q_paper → 𝖖²), normalised to start at 1:

        χ = (1/(q)_∞) Σ_{ℓ∈Z} (q^{2ℓ²(2n+3)+(2n−2r+1)ℓ}
                                − q^{2ℓ²(2n+3)+(2n+2r+5)ℓ+r+1}),  r = rbar

    with 0 ≤ rbar ≤ 2n (rbar+1 ≡ 0 mod 2n+3 would vanish identically).
    """
    if not (0 <= rbar <= 2 * n):
        raise ValueError("need 0 <= rbar <= 2n")
    kq = K // 2 + 1
    p = 2 * n + 3
    theta = [0] * (kq + 1)
    ell = 0
    while True:
        hit = False
        for sgn_l in ((ell,) if ell == 0 else (ell, -ell)):
            e1 = 2 * ell * ell * p + (2 * n - 2 * rbar + 1) * sgn_l
            e2 = (2 * ell * ell * p + (2 * n + 2 * rbar + 5) * sgn_l
                  + rbar + 1)
            if e1 <= kq:
                theta[e1] += 1
                hit = True
            if e2 <= kq:
                theta[e2] -= 1
                hit = True
        if not hit and 2 * ell * ell * p > kq + abs(2 * n + 2 * rbar + 5) * ell:
            break
        ell += 1
    # multiply by 1/(q)_∞
    out = list(theta)
    for m in range(1, kq + 1):
        for k in range(m, kq + 1):
            out[k] += out[k - m]
    # the previous loop IS the standard in-place expansion of
    # ∏ 1/(1−q^m): after processing m, out is theta · ∏_{i≤m} (q^i)⁻¹.
    return {2 * k: c for k, c in enumerate(out) if c and 2 * k <= K}
