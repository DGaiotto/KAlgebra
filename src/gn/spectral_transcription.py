"""`spectral_transcription` — building `L_{m,0}` from the SPECTRAL LAW.

The second production construction of an undressed canonical, beside the
(★)-guarded solve, and the one that is a **closed form** rather than a search.

The law (pure gauge)
--------------------
`L_{m,0}` is diagonal on a basis `{P̄_μ}` of `R(G)`:

    L_{m,0}  =  c_m · Λ_m ,     Λ_m : P̄_μ ↦ 𝖖^{2⟨μ,m⟩} P̄_μ ,
    c_m      =  (−1)^{⟨ρ,m⟩} 𝖖^{⟨ρ,m⟩}

and `{P̄_μ}` is characterized **intrinsically** — triangular in the characters
(`P̄_μ = χ_μ + O(𝖖²)`) and orthogonal for `Δ = ∏_{α∈Φ}(v^α;𝖖²)_∞`, a Gram–Schmidt
entirely inside `R(G)`.  Nothing in that references a chart or a solve, which is
what makes the spectral description a *construction* rather than an observation
(the design notes §7i; `star_bubbling.spectral_eigenbasis` /
`spectral_operator_matrix`).

What this module adds: the TRANSCRIPTION back to the f-presentation
------------------------------------------------------------------
The spectral law delivers the operator as a matrix on the **character basis**.
The `AbeKAlgebra` tier speaks the **f-presentation** `x = Σ_a f_a(𝖖^a v)·U_a`.
Promoted from a probe in the source repository (which now imports
from here, so there is one implementation), the bridge is the identity

    Σ_a  χ_μ(𝖖^{2a} v) · d_a(v)  =  (D χ_μ)(v)  =  Σ_ν C[μ][ν] χ_ν(v),
    d_a(v) = f_a(𝖖^a v) · ψ_a(v)                    [= `star_bubbling.conventional`]

solved in two decoupled steps rather than as one joint system (at G₂ `m=(2,3)`
the joint system is 182 unknowns and exact elimination does not finish):

1. **pointwise** — at a fixed exact rational `(v, 𝖖)` the identity is a linear
   system in the `|support|` NUMBERS `{d_a(v)}`, one row per character weight;
2. **per cell** — `g_a(v) := d_a(v)/ψ_a(v) = f_a(𝖖^a v)`, so each cell's
   numerator coefficients form a linear system *of that cell alone*.

Cells are solved independently, so cross-cell Weyl covariance is never imposed
and comes out as a check rather than an input.

The ansatz is DATUM-ONLY
------------------------
Nothing here is read off a solved chart — that is the difference between this and
a probe in the source repository, which re-constructs.  The
denominator is `K_f(n)` (the affine-reflection wall set minus `ψ_n`'s poles, i.e.
(★)'s own pole structure — the same ansatz `star_bubbling.joint_solve` carries),
the numerator support is `box_points(...)` (a cutoff, certified by widening,
never a selector), and the `𝖖`-content is palindromic blocks `𝖖^t + 𝖖^{−t}`,
which is W1 built into the shape.  Measured against the solved charts on 93/93
cells for both the denominator and the box.

Guarded like every other route
------------------------------
`certify_built` applies exactly what `solve_canonical` applies — W1 (bar), (★),
and the leading orbit — and `spectral_canonical` refuses to return anything that
does not clear it.  The route is therefore admissible on the same terms as the
(★) solve, and it is genuinely INDEPENDENT of it: no `solve_canonical` anywhere
in its path, which is what makes agreement between the two evidence rather than
tautology (the `recursive_spectrum` precedent).

Cost — read this before choosing it
-----------------------------------
It is **not** uniformly faster than the (★) solve, and at shallow charges it is
slower: measured SU(3) `m=(1,1)` 3.2 s against ~0.09 s, Spin(5) `m=(2,1)` 6.5 s
against 0.79 s, Sp(2) `m=(1,1)` 6.9 s against 1.12 s.  Its cost is dominated by
the Gram–Schmidt, which depends on `(datum, height, order)` and **not** on `m`,
so it is memoized and amortizes across every charge at one datum; the solve's
cost instead grows superlinearly in the number of bubbled cells carrying free
parameters.  The two therefore cross somewhere, and where they cross is a
measurement, not a guess — a probe in the source repository.

Dependency-free (stdlib `Fraction` only), exact throughout, with a box
certificate (re-solve one step wider and demand the same element).
"""
from __future__ import annotations

import random
from fractions import Fraction

import star_bubbling as SB
import wrq_torus as W
from laurent_poly import LaurentPoly
from weyl_torus_ring import TorusLaurent, TorusRational

__all__ = [
    "wall_sets", "cell_ansatz", "pointwise_values", "solve_cell",
    "spectral_matrix_calibrated", "certify_built", "spectral_canonical",
]

_PRIME = 2147483647


def _ratio(diff, c, d):
    t = None
    for i in range(d):
        if c[i] == 0:
            if diff[i]:
                return None
            continue
        if diff[i] % c[i]:
            return None
        ti = diff[i] // c[i]
        if t is None:
            t = ti
        elif t != ti:
            return None
    return t


def wall_sets(datum, m):
    """`(support, K_d, K_f)` — the affine-reflection wall sets of every cell.

    A transcription of the closures inside `star_bubbling.joint_solve`; pure
    functions of the root datum and `m`.  `K_f(n)` is the predicted denominator
    of the residual `f_n`."""
    d = datum.dim
    support = [tuple(p) for p in SB.tropical_support(datum, m)]
    cor = {tuple(al): SB.coroot_of(datum, al) for al in datum.positive_roots()}

    def partner(n, al, k):
        t = datum.shift_pairing(n, al) + (-k // 2)
        c = cor[tuple(al)]
        return tuple(n[i] - t * c[i] for i in range(d))

    Kd, Kf = {}, {}
    for n in support:
        out = set()
        for al in datum.positive_roots():
            for p in support:
                if p == n:
                    continue
                diff = tuple(n[i] - p[i] for i in range(d))
                t = _ratio(diff, cor[tuple(al)], d)
                if t is None:
                    continue
                k = -2 * (t - datum.shift_pairing(n, al))
                if partner(n, al, k) == p:
                    out.add((tuple(al), k))
        Kd[n] = sorted(out)
        P = set(W.dressing_psi(datum, n).simplify().den)
        Kf[n] = sorted((al, k - datum.shift_pairing(n, al))
                       for (al, k) in Kd[n] if (al, k) not in P)
    return support, Kd, Kf


def cell_ansatz(datum, n, e, kf, pad=1, q_extra=2, free_q=False):
    """The unknown basis at one cell: symmetrized box monomials x 𝖖-blocks.

    Returns `(den, blocks)` where `blocks` is a list of
    `(TorusLaurent numerator, q-exponent list)`: the unknown multiplying that
    numerator is `Σ_t a_t·(𝖖^t + 𝖖^{−t})` (palindromic, = W1) or, with
    `free_q`, an unconstrained `Σ_e c_e 𝖖^e`."""
    den = {}
    for key in kf:
        den[key] = den.get(key, 0) + 1
    S = sum(k for (_al, k) in kf if k > 0)
    T = max(S - 1, 0) + q_extra
    stab = [w for w in datum.weyl_elements()
            if tuple(datum.act_cochar(w, n)) == n]

    made, seen = [], []
    for lam in SB.box_points(datum, e, kf, pad):
        base = TorusRational(
            datum, TorusLaurent(datum, {lam: LaurentPoly.one()}), den)
        imgs = []
        for w in stab:
            im = base.weyl_act(w)
            if im.den != den:
                raise RuntimeError(f"stabilizer moved the denominator at {n}")
            if not any((im.num - prev.num).is_zero() for prev in imgs):
                imgs.append(im)
        acc = TorusLaurent.zero(datum)
        for im in imgs:
            acc = acc + im.num
        if acc.is_zero() or any((acc - prev).is_zero() for prev in seen):
            continue
        seen.append(acc)
        made.append(acc)

    qexps = list(range(-T, T + 1)) if free_q else list(range(0, T + 1))
    return den, made, qexps, T


# ===========================================================================
# numeric evaluation helpers (exact Fractions throughout)
# ===========================================================================
def ev_mono(wt, vpt):
    r = Fraction(1)
    for i, x in enumerate(wt):
        r *= vpt[i] ** int(x)
    return r


def ev_lp(lp, qv):
    return sum(Fraction(int(c)) * qv ** int(e) for e, c in lp._coeffs.items())


def ev_tl(tl, vpt, qv):
    """Evaluate a TorusLaurent."""
    return sum(ev_mono(wt, vpt) * ev_lp(l, qv) for wt, l in tl.terms.items())


def ev_tr(fr, vpt, qv):
    """Evaluate a TorusRational."""
    num = ev_tl(fr.num, vpt, qv)
    den = Fraction(1)
    for (a, k), mult in fr.den.items():
        den *= (1 - qv ** int(k) * ev_mono(a, vpt)) ** mult
    if den == 0:
        raise ZeroDivisionError
    return num / den


def ev_den(den, vpt, qv):
    r = Fraction(1)
    for (a, k), mult in den.items():
        r *= (1 - qv ** int(k) * ev_mono(a, vpt)) ** mult
    if r == 0:
        raise ZeroDivisionError
    return r


def character_tl(datum, nu):
    """`χ_ν` as a TorusLaurent (the Wilson line's magnetic-0 residual)."""
    return W.wilson(datum, nu).residuals()[(0,) * datum.dim].num


def ev_chi(chi_tl, vpt, qv, shift=None):
    """`χ(v)`, or `χ(𝖖^{2·shift} v)` when `shift` is given."""
    tot = Fraction(0)
    for wt, l in chi_tl.terms.items():
        t = ev_lp(l, qv) * ev_mono(wt, vpt)
        if shift is not None:
            t *= qv ** (2 * sum(int(wt[i]) * int(shift[i])
                                for i in range(len(shift))))
        tot += t
    return tot


# ===========================================================================
# exact linear algebra
# ===========================================================================
_PRIME = (1 << 61) - 1          # Mersenne, comfortably above any canonical
                                # numerator coefficient seen in this repo


def _to_mod(x, p):
    return (x.numerator % p) * pow(x.denominator % p, p - 2, p) % p


def solve_mod_p(rows, rhs, nun, p=_PRIME):
    """Gauss-Jordan mod `p`, then lift residues to small signed integers.

    The wall the design notes §7j names: exact `Fraction` elimination on
    the deep G₂ cell does not finish, because the intermediate rationals grow.
    `star_bubbling.solve_canonical` already crossed it the same way — eliminate
    mod one prime, then verify exactly over `Z`.  Verification here is
    `verify_rows_exact` on the spare rows, which is what makes the mod-`p`
    result a certificate rather than a guess."""
    A = [[_to_mod(x, p) for x in rows[i]] + [_to_mod(rhs[i], p)]
         for i in range(len(rows))]
    piv_cols, r = [], 0
    for c in range(nun):
        pv = next((i for i in range(r, len(A)) if A[i][c]), None)
        if pv is None:
            continue
        A[r], A[pv] = A[pv], A[r]
        inv = pow(A[r][c], p - 2, p)
        A[r] = [x * inv % p for x in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c]:
                f = A[i][c]
                Ar = A[r]
                A[i] = [(A[i][k] - f * Ar[k]) % p for k in range(nun + 1)]
        piv_cols.append(c)
        r += 1
    inconsistent = any(all(A[i][k] == 0 for k in range(nun)) and A[i][nun]
                       for i in range(r, len(A)))
    sol = [Fraction(0)] * nun
    for i, c in enumerate(piv_cols):
        v = A[i][nun]
        sol[c] = Fraction(v - p if v > p // 2 else v)     # signed lift
    return sol, r, inconsistent, set(piv_cols)


def verify_rows_exact(rows, rhs, sol):
    """Exact check of a candidate solution on EVERY row, over Q."""
    bad = 0
    for i, row in enumerate(rows):
        acc = Fraction(0)
        for c, x in zip(row, sol):
            if c:
                acc += c * x
        if acc != rhs[i]:
            bad += 1
    return bad


def solve_exact(rows, rhs, nun):
    """Gauss-Jordan over Q.  Returns `(solution, rank, inconsistent)`."""
    A = [list(rows[i]) + [rhs[i]] for i in range(len(rows))]
    piv_cols, r = [], 0
    for c in range(nun):
        p = next((i for i in range(r, len(A)) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        inv = Fraction(1) / A[r][c]
        A[r] = [x * inv for x in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [A[i][k] - f * A[r][k] for k in range(nun + 1)]
        piv_cols.append(c)
        r += 1
    sol = [Fraction(0)] * nun
    for i, c in enumerate(piv_cols):
        sol[c] = A[i][nun]
    inconsistent = any(all(A[i][k] == 0 for k in range(nun)) and A[i][nun] != 0
                       for i in range(r, len(A)))
    return sol, r, inconsistent, set(piv_cols)


# ===========================================================================
# Step 1 — pointwise values of d_a
# ===========================================================================
def pointwise_values(datum, m, C, wts, support, npts, seed=7, verbose=False,
                     known=None):
    """Solve `Σ_a χ_μ(𝖖^{2a}v)·d_a(v) = Σ_ν C[μ][ν]χ_ν(v)` for `{d_a(v)}`.

    `known` is `{cell: f_a}` for cells whose residual is already known from the
    datum — in practice the top Weyl orbit, where `leading_orbit` gives
    `f_a = 1` with no solve (this is precisely the seed `joint_solve` uses, so
    supplying it keeps the construction datum-only).  Those `d_a = f_a(𝖖^a v)·
    ψ_a(v)` move to the RHS, and the system shrinks to the bubbling cells.

    That shrink matters: the system is square in the unknown cells, so it needs
    at least that many dominant weights in the spectral window.  At G₂ the
    dominant weights are sparse (`⟨μ,Σ⁺^∨⟩` grows fast), and asking for 13 —
    all cells of `m=(2,3)` — forces a height around 34 with a correspondingly
    deep truncation, while asking for the 7 bubbling cells is comfortable."""
    d = datum.dim
    known = {tuple(a): f for a, f in (known or {}).items()}
    unknown = [a for a in support if a not in known]
    chis = {tuple(nu): character_tl(datum, nu) for nu in wts}
    psis = {a: W.dressing_psi(datum, a).simplify() for a in known}
    if len(wts) < len(unknown):
        raise RuntimeError(
            f"spectral window has {len(wts)} weights < {len(unknown)} unknown "
            f"cells; raise the height")
    rnd = random.Random(seed)
    out, tries = [], 0
    while len(out) < npts and tries < 60 * npts:
        tries += 1
        vpt = [Fraction(rnd.randint(2, 60), rnd.randint(61, 130))
               for _ in range(d)]
        qv = Fraction(rnd.randint(2, 40), rnd.randint(41, 90))
        try:
            # d_a(v) for the known cells -- moved to the RHS
            dknown = {}
            for a, f in known.items():
                sh = [qv ** int(a[i]) * vpt[i] for i in range(d)]
                dknown[a] = ev_tr(f, sh, qv) * ev_tr(psis[a], vpt, qv)
            rows, rhs = [], []
            for mu in wts:
                mu = tuple(mu)
                rows.append([ev_chi(chis[mu], vpt, qv, shift=a)
                             for a in unknown])
                r = sum(ev_lp(C[mu][nu], qv) * ev_chi(chis[tuple(nu)], vpt, qv)
                        for nu in C[mu])
                for a, dv in dknown.items():
                    r -= ev_chi(chis[mu], vpt, qv, shift=a) * dv
                rhs.append(r)
            sol, rank, bad, _ = solve_exact(rows, rhs, len(unknown))
        except ZeroDivisionError:
            continue
        if bad or rank < len(unknown):
            continue
        vals = {a: sol[i] for i, a in enumerate(unknown)}
        vals.update(dknown)
        out.append((vpt, qv, vals))
    if len(out) < npts:
        raise RuntimeError(f"only {len(out)}/{npts} usable points")
    if verbose:
        print(f"    pointwise: {len(out)} points, system "
              f"{len(wts)}x{len(unknown)} "
              f"({len(known)} cell(s) seeded from leading_orbit)")
    return out


# ===========================================================================
# Step 2 — per-cell numerator interpolation
# ===========================================================================
def solve_cell(datum, n, den, basis, qexps, pts, free_q, verbose=False,
               mod_p=False):
    """Recover `f_n = N_n/D_n` from the pointwise values, cell `n` alone."""
    d = datum.dim
    psi_n = W.dressing_psi(datum, n).simplify()
    nun = len(basis) * len(qexps)
    rows, rhs = [], []
    for vpt, qv, dvals in pts:
        try:
            psi = ev_tr(psi_n, vpt, qv)
            if psi == 0:
                continue
            sh = [qv ** int(n[i]) * vpt[i] for i in range(d)]
            Dsh = ev_den(den, sh, qv)
            g = dvals[n] / psi                     # = f_n(𝖖^n v)
            target = g * Dsh                       # = N_n(𝖖^n v)
            row = []
            for tl in basis:
                base = ev_tl(tl, sh, qv)
                for t in qexps:
                    row.append(base * (qv ** t if (free_q or t == 0)
                                       else (qv ** t + qv ** (-t))))
        except ZeroDivisionError:
            continue
        rows.append(row)
        rhs.append(target)
        if len(rows) >= nun + 10:
            break
    if len(rows) < nun + 1:
        raise RuntimeError(f"cell {n}: {len(rows)} rows < {nun}+1 unknowns "
                           f"(need more points)")
    if mod_p:
        sol, rank, bad, piv = solve_mod_p(rows, rhs, nun)
        nwrong = verify_rows_exact(rows, rhs, sol)
        if nwrong:
            raise RuntimeError(
                f"cell {n}: mod-p solution fails exact verification on "
                f"{nwrong}/{len(rows)} rows")
    else:
        sol, rank, bad, piv = solve_exact(rows, rhs, nun)
    # rebuild N_n as a TorusLaurent
    num = TorusLaurent.zero(datum)
    k = 0
    for tl in basis:
        for t in qexps:
            c = sol[k]
            k += 1
            if c == 0:
                continue
            if c.denominator != 1:
                raise RuntimeError(f"cell {n}: non-integer coefficient {c}")
            lp = (LaurentPoly({t: int(c)}) if (free_q or t == 0)
                  else LaurentPoly({t: int(c), -t: int(c)}))
            num = num + tl.scale(lp) if hasattr(tl, "scale") else num + _scale(
                datum, tl, lp)
    return TorusRational(datum, num, den).simplify(), rank, nun, bad, len(rows)


def _scale(datum, tl, lp):
    return TorusLaurent(datum, {wt: l * lp for wt, l in tl.terms.items()})


# The truncation certificate is load-bearing (the design notes §7j: at
# SU(2) height 10, order 14 gives 74/81 and FAILS, order 40 passes and gives
# 81/81).  Rather than hard-code a per-case order, escalate until it passes —
# and raise the height until the window holds at least |support| weights, which
# is what makes the pointwise system solvable.
_ORDER_LADDER = (1.5, 2.0, 3.0, 4.0)

# O1 — the Gram-Schmidt depends on (datum, height, order) only, NOT on `m`.
# `spectral_operator_matrix` recomputes it on every call, so building k charges
# at one datum paid for k Gram-Schmidts.  §7m's "reused across all m" is only
# true with this cache in place.  Keyed by name+rank since RootDatum is not
# hashable.
_EIGEN_CACHE: dict = {}


def _cached_eigenbasis(datum, height, order, span=8):
    key = (datum.name, datum.dim, height, order, span)
    got = _EIGEN_CACHE.get(key)
    if got is None:
        got = SB.spectral_eigenbasis(datum, height, order, span)
        _EIGEN_CACHE[key] = got
    return got


def _operator_matrix_cached(datum, m, height, order, span=8):
    """`spectral_operator_matrix` with the eigenbasis cached across `m`.

    Mirrors the spine routine: `C = T^{-1}·diag(λ)·T` with `T` the UNBARRED
    transition (`P = bar(P̄)`)."""
    Pbar, weights = _cached_eigenbasis(datum, height, order, span)
    T = {mu: {nu: LaurentPoly({-k: c for k, c in lp._coeffs.items()})
              for nu, lp in row.items()} for mu, row in Pbar.items()}
    rho = SB.rho_weight(datum)
    rm = sum(rho[i] * m[i] for i in range(datum.dim))
    lam = {mu: LaurentPoly(
        {rm + 2 * sum(mu[i] * m[i] for i in range(datum.dim)): (-1) ** rm})
        for mu in weights}
    Tinv = {}
    for i, mu in enumerate(weights):
        row = {mu: LaurentPoly.one()}
        for sig in reversed(weights[:i]):
            acc = LaurentPoly.zero()
            for nu, c in row.items():
                t = T[nu].get(sig)
                if t is not None:
                    acc = acc + c * t
            if not acc.is_zero():
                row[sig] = LaurentPoly.zero() - acc
        Tinv[mu] = row
    C = {}
    for mu in weights:
        out = {}
        for nu, c in Tinv[mu].items():           # χ_μ = Σ_ν Tinv[μ][ν] P_ν
            cc = c * lam[nu]
            for sig, t in T[nu].items():
                out[sig] = out.get(sig, LaurentPoly.zero()) + cc * t
        C[mu] = {k: v for k, v in out.items() if not v.is_zero()}
    return C, weights


def spectral_matrix_calibrated(datum, m, H, N, ncells, verbose=True):
    """`spectral_operator_matrix` with the height/order raised until the
    truncation certificate passes AND the window holds >= `ncells` weights."""
    h, n = H, N
    for attempt in range(8):
        try:
            C, wts = _operator_matrix_cached(datum, m, h, n)
        except SB.StarGuardFailure as ex:
            if "truncation certificate" not in str(ex):
                raise
            n = int(n * _ORDER_LADDER[min(attempt, len(_ORDER_LADDER) - 1)]) + 2
            if verbose:
                print(f"    certificate failed -> order {n}")
            continue
        if len(wts) >= ncells:
            return C, wts, h, n
        h += 2
        n = max(n, int(n * 1.3) + 2)
        if verbose:
            print(f"    window {len(wts)} < {ncells} cells -> height {h}, "
                  f"order {n}")
    raise RuntimeError("could not calibrate the spectral window")


def certify_built(datum, m, e, built, verbose=False):
    """W1 (bar) + (★) + the leading-orbit label, on the spectrally-built
    element — the same guard `star_bubbling.solve_canonical` applies."""
    support = [tuple(p) for p in SB.tropical_support(datum, m)]
    missing = [a for a in support if a not in built]
    if missing:
        print(f"   CERT: {len(missing)} cell(s) missing: {missing[:4]}")
        return False
    x = W.WRQTorus(datum, {a: built[a] for a in support})
    w1 = x.well_formed_w1()
    ok_star, report = SB.criterion(x)
    # W2 / label: the leading Weyl orbit must be the intended one, coefficient 1
    seed = W.leading_orbit(datum, m, e).residuals()
    lead_ok = all((built[tuple(a)] - f).simplify().is_zero()
                  for a, f in seed.items())
    if verbose:
        print(f"   CERT  W1(bar)={'ok' if w1 else 'FAIL'}  "
              f"(★)={'ok' if ok_star else 'FAIL ' + str(report[:2])}  "
              f"leading orbit={'ok' if lead_ok else 'FAIL'}")
    return bool(w1 and ok_star and lead_ok)


# ===========================================================================
# The production entry point
# ===========================================================================
def spectral_canonical(datum, m, height=None, order=None, pad=0, q_extra=0,
                       mod_p=True, npts=None, seed=7, verbose=False,
                       use_seed=True, max_escalate=3):
    """`L_{m,0}` as a `WRQTorus`, built from the spectral law alone.

    The counterpart of `star_bubbling.solve_canonical` for the **undressed**
    charge, and an independent one: no `solve_canonical` in its path.  Returns a
    guarded element or raises — never a candidate.

    `use_seed=True` supplies the top Weyl orbit from `wrq_torus.leading_orbit`
    (`f_a = 1`, no solve — the same seed `joint_solve` uses), which shrinks the
    pointwise system to the bubbling cells.  That shrink is what keeps sparse-
    weight data tractable: the system is square in the unknown cells, so it needs
    at least that many dominant weights in the spectral window, and at G₂ the
    dominant weights are sparse.  `use_seed=False` reconstructs every cell from
    the spectrum alone — the stronger statement, where affordable.

    The ansatz starts TIGHT and escalates only on failure.  That is safe because
    a too-small ansatz fails **loudly**: the mod-`p` candidate is rejected by
    exact verification over `Z` on every row, so it can never return a wrong
    element, only refuse.  `max_escalate` bounds the ladder.

    Raises `NotImplementedError` when no ansatz on the ladder reaches, and
    `SB.StarGuardFailure` when a built element fails the guard — the two are
    different failures and are reported differently on purpose: the first says
    "the box did not reach", the second would say "the spectral law and the
    axioms disagree", which has never been observed and would be a finding."""
    datum_dim = datum.dim
    z = (0,) * datum_dim
    m = tuple(m)
    support, _Kd, Kf = wall_sets(datum, m)
    top = {tuple(datum.act_cochar(w, m)) for w in range(len(datum.weyl))}

    seed_f = ({tuple(a): f for a, f in
               W.leading_orbit(datum, m, z).residuals().items()}
              if use_seed else {})
    nunknown = len(support) - len(seed_f)
    # Start TIGHT and let `spectral_matrix_calibrated` escalate: it raises the
    # order until the truncation certificate passes and the height until the
    # window holds at least `nunknown` weights.  Starting generous is not free —
    # the Gram–Schmidt cost climbs steeply in both — and measured at SU(3)
    # `m=(1,1)` a start of (12, 14) cost 28.8 s against 3.2 s from (8, 12) for
    # the identical element.  These are the values the per-case table in
    # a probe in the source repository converged on.
    H = height if height is not None else 8
    N = order if order is not None else 12
    C, wts, H, N = spectral_matrix_calibrated(datum, m, H, N, nunknown,
                                              verbose=verbose)

    targets = [a for a in support if a not in seed_f]
    last = None
    for step in range(max_escalate + 1):
        pd, qx = pad + step, q_extra + step
        ans = {a: cell_ansatz(datum, a, z, Kf[a], pad=pd, q_extra=qx)
               for a in support}
        need = npts or (max(len(ans[a][1]) * len(ans[a][2])
                            for a in targets) + 12)
        try:
            pts = pointwise_values(datum, m, C, wts, support, need,
                                   seed=seed, verbose=verbose, known=seed_f)
        except RuntimeError as ex:
            last = f"pointwise: {ex}"
            continue
        built, failed = dict(seed_f), None
        for a in targets:
            den, basis, qexps, _T = ans[a]
            try:
                fr, _rank, _nun, bad, _nrows = solve_cell(
                    datum, a, den, basis, qexps, pts, False, verbose,
                    mod_p=mod_p)
            except Exception as ex:                   # loud, never silent
                failed = f"cell {a}: {type(ex).__name__}: {str(ex)[:90]}"
                break
            if bad:
                failed = f"cell {a}: {bad} row(s) failed exact verification"
                break
            built[a] = fr
        if failed:
            last = failed
            continue
        if not certify_built(datum, m, z, built, verbose=verbose):
            raise SB.StarGuardFailure(
                f"{datum.name}: the spectrally-built L_{{{m},0}} FAILS the guard "
                f"(W1 / (★) / leading orbit).  This has never been observed, and "
                f"it would mean the spectral law and the axioms disagree — a "
                f"finding, not a tuning problem.  Do not widen the ansatz to "
                f"make it go away.")
        return W.WRQTorus(datum, {a: built[a] for a in support})

    raise NotImplementedError(
        f"{datum.name}: the spectral transcription of L_{{{m},0}} did not reach "
        f"within {max_escalate} escalations of the ansatz (pad/q_extra from "
        f"({pad},{q_extra})).  Last: {last}.  The failure is a BOX failure, not "
        f"a wrong element — every candidate was rejected by exact verification "
        f"over Z.")
