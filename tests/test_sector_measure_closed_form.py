"""The per-sector measure of the Schur pairing in CLOSED FORM, for all `G`
and all `(G, N)`.

Pure (`wrq_torus.sector_measure_closed_form`):

    B_m(u) = u^{Σ_{α>0}⟨m,α⟩α} · ∏_{α>0} b_{|⟨m,α⟩|}(u^α),
    b_t(z) = z^t / [(1−𝖖^t z)(1−𝖖^{−t}z) ∏_{j=1}^{t−1}(1−𝖖^{t−2j}z)²],

matter (`matter_wrq_torus.matter_sector_measure_closed_form`):

    B^N_m(u,μ) = B_m(u) · ∏_{i,w} Ξ_{⟨m,w⟩}(μ_i u^w),
    Ξ_c(x) = x^{−max(c,0)} ∏_{j<|c|}(1 + 𝖖^{|c|−1−2j} x).

Both are DERIVED in their docstrings; this file pins them against the
derived objects (`sector_measure`, `matter_sector_measure`) exactly — no
truncation, no states — and pins the two facts the derivation turns into
theorems: the collapse of `Q_m = T_{+m}μ/T_{−m}μ` to a monomial and the
inversion certificates (axiom 5, ρ-equivariance of the trace, manifest per
sector), on the matter tier with `⋆` (μ ↦ 1/μ) included.

Legs, positive control first:

  0. CONTROL — at `m = 0` every object is `1` (pure) / `{0⃗: 1}` (matter), and
     the closed form agrees there; abort otherwise.
  1. PURE closed form == derived `sector_measure`, 415 cocharacters over
     eleven data (u_2, u_3, su_2, su_3, sp_4, so_3, so_5, Spin(5), g_2, U(1)²,
     U(2)×U(1)) — odd height and non-dominant `m` included; the prefactor
     `sector_measure_prefactor` is `(1, 0)` on all of them.
  2. `Q_m` finite product == the monomial `u^{−2Σ⟨m,α⟩α}`; `b_t(1/z) = b_t(z)`;
     the certificate `v̄(B_m) = Q_m B_m` through the closed form; and the
     pole data of `B_m` (every `(α, k)` with `k ≡ ⟨m,α⟩ mod 2`, multiplicity
     ≤ 2), which is what makes `M_m = T_{+m}μ·B_m` entire.
  3. The trace's Nahm factor is `1/(−𝖖x;𝖖²)_∞` (the input of the matter
     ratio), to `𝖖^{12}`, `x^{6}`.
  4. MATTER closed form == derived `matter_sector_measure` over the thirteen
     roster presets (U(2)+1 … G₂+7, two-slot Spin(5) spinors, N = 2* SU(2)).
  5. MATTER certificate `v̄⋆(B^N_m) = Q^N_m B^N_m` exactly, same presets.
  6. THE DRAFT'S FORM: the author's `K_𝖖-algebras` draft (eq. Iexplicit) writes
     the pairing measure with ABSOLUTE-VALUE shifts, `(𝖖^{|⟨m,α⟩|}v^α;𝖖²)_∞
     (𝖖^{2+|⟨m,α⟩|}v^α;𝖖²)_∞` per root and `1/(−𝖖^{1+|⟨m,w⟩|}μv^w;𝖖²)_∞(…)` per
     hyper weight, and no rational factor.  `T_{+m}μ·B_m` and
     `T_{+m}μ_N·∏Ξ_c` are exactly that: `B_m` and `Ξ_c` are the finite ratios
     converting the signed shifts into `|·|` shifts, pinned per root pair and
     per weight.  (Matched after the fact — the closed form here was derived
     from the code's definitions, not read off the draft.)

Run:  `python3 run_tests.py` [--full]
(`--full` widens the matter grid; default keeps the file under ~1 min.)
"""
from __future__ import annotations

import itertools
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "implementations")

import root_datum as rd
from g_matter_roster import roster_spec
from habiro import HabiroElement
from laurent_poly import LaurentPoly
from matter_wrq_torus import (
    matter_sector_measure, matter_sector_measure_closed_form,
    matter_sector_measure_inversion_ratio, matter_weight_factor, slot_weights,
    verify_matter_sector_measure_closed_form,
    verify_matter_sector_measure_inversion_symmetric,
)
from weyl_torus_ring import TorusLaurent, TorusRational
from wrq_torus import (
    sector_measure, sector_measure_closed_form, sector_measure_prefactor,
    sector_measure_inversion_ratio, sector_measure_inversion_ratio_closed_form,
    sector_root_factor, verify_sector_measure_closed_form,
    verify_sector_measure_inversion_symmetric,
)

FULL = "--full" in sys.argv
FAILURES = []
COUNTS = {}


def check(name, ok, group=None):
    if group is not None:
        COUNTS[group] = COUNTS.get(group, [0, 0])
        COUNTS[group][0] += bool(ok)
        COUNTS[group][1] += 1
        if ok:
            return
    print(("  PASS  " if ok else "  FAIL  ") + name)
    if not ok:
        FAILURES.append(name)


def _grid(dim, lo, hi):
    return list(itertools.product(range(lo, hi + 1), repeat=dim))


PURE_DATA = [
    ("u_2", rd.u_n(2), 3), ("u_3", rd.u_n(3), 2), ("su_2", rd.su_2(), 4),
    ("su_3", rd.su_n(3), 2), ("sp_4", rd.sp_n(2), 2), ("so_5", rd.so_n(5), 2),
    ("so_3", rd.so_n(3), 3), ("spin_5", rd.b_n_simply_connected(2), 2),
    ("g_2", rd.g_2(), 2),
    ("u1xu1", rd.product_datum([rd.u_n(1), rd.u_n(1)]), 2),
    ("u2xu1", rd.product_datum([rd.u_n(2), rd.u_n(1)]), 1),
]

PRESETS = ["u2-nf1", "u2-nf2", "su2-nf1", "su2-nf2", "su3-nf1", "u3-nf1",
           "sp4-nf1", "sp4-nf2", "spin5-vector-nf1", "spin5-spinor-nf1",
           "spin5-spinor-nf2", "g2-nf1", "su2-adjoint"]


def leg_control():
    print("[0] control: m = 0")
    for name, dat, _ in PURE_DATA:
        z = (0,) * dat.dim
        one = TorusRational.one(dat)
        ok = ((sector_measure(dat, z) - one).simplify().is_zero()
              and (sector_measure_closed_form(dat, z) - one).simplify().is_zero()
              and sector_measure_prefactor(dat, z) == (1, 0))
        check(f"pure control {name}: B_0 == 1 both ways", ok)
    for name in PRESETS:
        dat, lam, nf = roster_spec(name)
        slots = slot_weights(dat, (tuple(lam),) * nf)
        z = (0,) * dat.dim
        a = matter_sector_measure(dat, slots, z)
        b = matter_sector_measure_closed_form(dat, slots, z)
        k0 = (0,) * len(slots)
        ok = (set(a) == {k0} == set(b)
              and (a[k0] - TorusRational.one(dat)).simplify().is_zero()
              and (b[k0] - TorusRational.one(dat)).simplify().is_zero())
        check(f"matter control {name}: B^N_0 == {{0⃗: 1}} both ways", ok)
    if FAILURES:
        print("CONTROL FAILED — aborting before any scan is trusted")
        sys.exit(1)


def leg_pure():
    print("\n[1] pure closed form == derived sector_measure")
    t0 = time.time()
    n = 0
    for name, dat, R in PURE_DATA:
        for m in _grid(dat.dim, -R, R):
            n += 1
            check(f"{name} m={m} closed == derived",
                  verify_sector_measure_closed_form(dat, m), group="pure")
            check(f"{name} m={m} prefactor is (1, 0)",
                  sector_measure_prefactor(dat, m) == (1, 0), group="pref")
    g, tot = COUNTS["pure"]
    check(f"pure closed form == derived: {g}/{tot} cocharacters "
          f"[{time.time()-t0:.1f}s]", g == tot)
    g, tot = COUNTS["pref"]
    check(f"prefactor (1, 0) on every shipped convention: {g}/{tot}", g == tot)


def leg_theorems():
    print("\n[2] the two theorems: Q_m collapses; b_t and B_m invert")
    for name, dat, R in PURE_DATA:
        for m in _grid(dat.dim, -R, R):
            q = sector_measure_inversion_ratio(dat, m)
            qc = sector_measure_inversion_ratio_closed_form(dat, m)
            check(f"{name} m={m} Q_m == u^(-2C)",
                  (q - qc).simplify().is_zero(), group="Q")
            check(f"{name} m={m} certificate via closed form",
                  verify_sector_measure_inversion_symmetric(dat, m), group="cert")
    g, tot = COUNTS["Q"]
    check(f"Q_m finite product == monomial: {g}/{tot}", g == tot)
    g, tot = COUNTS["cert"]
    check(f"v̄(B_m) == Q_m·B_m: {g}/{tot}", g == tot)
    # M_m = T_{+m}μ·B_m is entire: every denominator factor (α, k) of B_m has
    # k ≡ ⟨m,α⟩ (mod 2) and multiplicity ≤ 2, where the shifted measure has a
    # double zero.
    for name, dat, R in PURE_DATA:
        for m in _grid(dat.dim, -R, R):
            B = sector_measure_closed_form(dat, m)
            ok = all((k - dat.shift_pairing(m, a)) % 2 == 0 and mult <= 2
                     for (a, k), mult in B.den.items())
            check(f"{name} m={m} B_m poles under double zeros of T_(+m)μ", ok,
                  group="entire")
    g, tot = COUNTS["entire"]
    check(f"M_m = T_(+m)μ·B_m entire (pole data of B_m): {g}/{tot}", g == tot)
    dat = rd.g_2()
    for a in dat.positive_roots():
        for t in range(0, 5):
            b = sector_root_factor(dat, a, t)
            check(f"g_2 α={a} t={t}: b_t(1/z) == b_t(z)",
                  (b.vinv() - b).simplify().is_zero(), group="bt")
    g, tot = COUNTS["bt"]
    check(f"b_t inversion-invariant at every g_2 root, t ≤ 4: {g}/{tot}",
          g == tot)


def leg_nahm_factor():
    print("\n[3] the trace's Nahm factor is 1/(−𝖖x;𝖖²)_∞")
    K, W = 12, 6
    # Σ_n a_n x^n, a_n = nahm_term((−1)^n, n, [n]) expanded (the trace's `a_n`)
    lhs = [HabiroElement.nahm_term((-1) ** n, n, [n]).expand(K + 2)
           for n in range(W + 1)]
    lhs = [LaurentPoly({e: c for e, c in a._coeffs.items() if e <= K})
           for a in lhs]
    # ∏_{k≥0} 1/(1 + 𝖖^{2k+1} x) = ∏_k Σ_r (−1)^r 𝖖^{(2k+1)r} x^r
    rhs = [LaurentPoly({0: 1})] + [LaurentPoly.zero()] * W
    k = 0
    while 2 * k + 1 <= K:
        fac = [LaurentPoly({(2 * k + 1) * r: (-1) ** r}) for r in range(W + 1)]
        new = [LaurentPoly.zero() for _ in range(W + 1)]
        for i in range(W + 1):
            for j in range(W + 1 - i):
                new[i + j] = new[i + j] + rhs[i] * fac[j]
        rhs = [LaurentPoly({e: c for e, c in p._coeffs.items() if e <= K})
               for p in new]
        k += 1
    ok = all((lhs[i] - rhs[i]).is_zero() for i in range(W + 1))
    check(f"Σ (−𝖖x)^n/(𝖖²;𝖖²)_n == ∏_k 1/(1+𝖖^(2k+1)x) to 𝖖^{K}, x^{W}", ok)


def _preset_grid(dat, lam, nf):
    slots = slot_weights(dat, (tuple(lam),) * nf)
    d = dat.dim
    if FULL:
        R = 2 if d <= 2 else 1
    else:
        big = sum(len(s) for s in slots) >= 7 or (len(slots) == 2 and d >= 2)
        R = 1 if (d >= 3 or big) else 2
    return slots, _grid(d, -R, R)


def leg_matter():
    print("\n[4] matter closed form == derived matter_sector_measure")
    t0 = time.time()
    for name in PRESETS:
        dat, lam, nf = roster_spec(name)
        slots, grid = _preset_grid(dat, lam, nf)
        for m in grid:
            check(f"{name} m={m} closed == derived",
                  verify_matter_sector_measure_closed_form(dat, slots, m),
                  group="matter")
    g, tot = COUNTS["matter"]
    check(f"matter closed form == derived over {len(PRESETS)} presets: "
          f"{g}/{tot} [{time.time()-t0:.1f}s]", g == tot)


def leg_matter_certificate():
    print("\n[5] matter certificate: v̄⋆(B^N_m) == Q^N_m · B^N_m")
    t0 = time.time()
    nonvac = 0
    for name in PRESETS:
        dat, lam, nf = roster_spec(name)
        slots, grid = _preset_grid(dat, lam, nf)
        for m in grid:
            check(f"{name} m={m} certificate",
                  verify_matter_sector_measure_inversion_symmetric(dat, slots, m),
                  group="mcert")
            lev, _ = matter_sector_measure_inversion_ratio(dat, slots, m)
            nonvac += any(lev)
    g, tot = COUNTS["mcert"]
    check(f"matter certificate exact: {g}/{tot}, {nonvac} with a non-zero "
          f"μ-level shift (non-vacuous ⋆) [{time.time()-t0:.1f}s]",
          g == tot and nonvac > 0)


def _abs_shift_over_signed_gauge(dat, m):
    """Per root pair, [the |⟨m,α⟩|-shifted Pochhammers of the draft's pairing
    measure] / [the signed-shift `T_{+m}μ`]: a FINITE ratio.  With `s = ⟨m,α⟩`
    and `t = |s|`, the two agree on the side where the shift is `+t` and differ
    on the other side, `y = u^{∓α}`, by
    `(𝖖^{t}y;𝖖²)(𝖖^{2+t}y;𝖖²) / (𝖖^{−t}y;𝖖²)(𝖖^{2−t}y;𝖖²)
     = 1/∏_{j<t}(1−𝖖^{−t+2j}y)(1−𝖖^{2−t+2j}y)`."""
    out = TorusRational.one(dat)
    for a in dat.positive_roots():
        s = int(dat.shift_pairing(m, a))
        if s == 0:
            continue
        t = abs(s)
        base = tuple(-x for x in a) if s > 0 else tuple(a)
        for j in range(t):
            for k in (-t + 2 * j, 2 - t + 2 * j):
                out = out * TorusRational.factor_inv(dat, base, k)
    return out.simplify()


def _abs_shift_over_signed_hyper(dat, w, c, slot, n):
    """Per hyper weight, [the draft's `1/(−𝖖^{1+|c|}x)_∞(−𝖖^{1+|c|}/x)_∞`] /
    [the signed `1/(−𝖖^{1+c}x)_∞(−𝖖^{1−c}/x)_∞`], `x = μ_slot u^w`: for `c > 0`
    `∏_{j<c}(1+𝖖^{1−c+2j}/x)`, for `c < 0` `∏_{j<|c|}(1+𝖖^{1+c+2j}x)`."""
    levels = {(0,) * n: TorusRational.one(dat)}
    if c == 0:
        return levels
    for j in range(abs(c)):
        if c > 0:
            mono = TorusRational.from_laurent(TorusLaurent.monomial(
                dat, tuple(-x for x in w), LaurentPoly({1 - c + 2 * j: 1})))
            dl = -1
        else:
            mono = TorusRational.from_laurent(TorusLaurent.monomial(
                dat, tuple(w), LaurentPoly({1 + c + 2 * j: 1})))
            dl = +1
        out = {}
        for k, v in levels.items():
            out[k] = (out[k] + v).simplify() if k in out else v
            k2 = tuple(x + (dl if t == slot else 0) for t, x in enumerate(k))
            term = (v * mono).simplify()
            out[k2] = (out[k2] + term).simplify() if k2 in out else term
        levels = {k: v for k, v in out.items() if not v.is_zero()}
    return levels


def leg_draft_form():
    print("\n[6] the closed form == the |⟨m,α⟩|, |⟨m,w⟩| shifts of the draft's "
          "pairing measure (K_𝖖-algebras draft, eq. Iexplicit)")
    for name, dat, R in PURE_DATA:
        for m in _grid(dat.dim, -R, R):
            ok = (_abs_shift_over_signed_gauge(dat, m)
                  - sector_measure_closed_form(dat, m)).simplify().is_zero()
            check(f"{name} m={m} |·|-shift / signed-shift == B_m", ok,
                  group="draftG")
    g, tot = COUNTS["draftG"]
    check(f"gauge: T_(+m)μ·B_m == the |⟨m,α⟩|-shifted measure: {g}/{tot}",
          g == tot)
    for name in PRESETS:
        dat, lam, nf = roster_spec(name)
        slots, grid = _preset_grid(dat, lam, nf)
        n = len(slots)
        for m in grid:
            for i, wts in enumerate(slots):
                for w in wts:
                    c = int(dat.shift_pairing(m, tuple(w)))
                    a = _abs_shift_over_signed_hyper(dat, w, c, i, n)
                    b = matter_weight_factor(dat, w, c, i, n)
                    ok = set(a) == set(b) and all(
                        (a[k] - b[k]).simplify().is_zero() for k in a)
                    check(f"{name} m={m} w={w}: |·|-shift / signed == Ξ_c", ok,
                          group="draftH")
    g, tot = COUNTS["draftH"]
    check(f"hyper: signed-shift measure × Ξ_c == the |⟨m,w⟩|-shifted one: "
          f"{g}/{tot}", g == tot)


if __name__ == "__main__":
    t0 = time.time()
    leg_control()
    leg_pure()
    leg_theorems()
    leg_nahm_factor()
    leg_matter()
    leg_matter_certificate()
    leg_draft_form()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print("  " + f)
        sys.exit(1)
    print(f"All sector-measure closed-form tests passed.  [{time.time()-t0:.1f}s]")
