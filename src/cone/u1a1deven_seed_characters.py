"""Closed forms for the seed traces of the u(1)-gauged [A_1, D_{2k+2}].

Part of the self-contained realization of the ADE finite-type algebras
(fully functional, arbitrary in principle precision and coverage).  The
research record — data generation, the fits, the held-out checks and the
controls, with the numbers — is a probe in the source repository.  Serving
since 2026-09-24: `U1A1DevenConeKAlgebra.trace` takes every seed from
`seed_trace` and reduces every other label onto the seeds by the cone data's
Layer-1 reduction (the transport is the witness: `seed_closed_forms=False`),
so `A1DevenKAlg(k)` and the zoo a1d6 / a1d8 trace their generators, products
and pairings at any depth; the suite in the source repository checks the
closed forms against the exact transport.  `SU3ADKAlg` serves its seeds
(Tr_1, Tr_T, Tr_D) from here: `sl3_su3_traces`.

Labels and seeds
----------------
The curve frame of `u1a1deven_geometric_frame`: a curve `(x, l)` of the
`(2k+2)`-gon with one interior puncture, from marked point `x` to `x + l`
with `l` boundary edges on the side without the puncture (`l = 2k+2` the loop
around it), times a power `E^n` of the gauge letter `E = X_{0,1}`.  A curve
of odd `l` has magnetic charge 0; a curve of even `l` has charge -1 when its
endpoints have the parity of marked point 2, +1 otherwise — marked point 2 is
the one the RG flow into A1Dodd removes, its two boundary edges becoming one
(the "merged vertex" of `u1a1deven_geometric_frame`, section B), and the
boundary edge from marked point 1 to it is where the curve frame's E-power is
anchored.  rho is
`rho((x, l)) = E^d (x + 1, l)`, `d = -#(endpoints at marked point 1)`,
`rho(E) = E^{-1}`, and the trace is rho-invariant.  The seeds — the elements
the traces of the generators of the ungauged `A1DevenKAlg(k)` (so of the zoo
a1d6 / a1d8 seeds) reduce to, one per generator — are

  * a curve of odd length, times `E^n`;
  * a non-crossing pair of a charge +1 and a charge -1 curve, times `E^n`.

A seed is taken to the representative of its rho-orbit (the lexicographically
smallest curve set in the orbit, the one the rules were measured on), its
E-power carried along; rho-invariance of the trace does the rest.

The rule
--------
Write `p = k + 1`, `z` the SU(2) fugacity, `chi_r(w) = w^r + w^{r-2} + ... +
w^{-r}` (`chi_{-1} = 0`), and the Weyl numerator

    N(z) := Tr(seed . E^n) (z - 1/z) prod_{i>=1} (1 - z^2 q^{2i})(1 - z^-2 q^{2i})
          = sum_{e>=1} c_e(n) (z^e - z^-e)

(q the repo's fq).  Then, with `w = q^e` and `sigma = q^n`,

    c_e(n) = (-1)^{p n} q^{p (e^2 - n^2) / 2} F(q^e, q^n)

on the support `e >= nu + |n - mu|`, `e - nu = n - mu (mod 2)`, and 0
elsewhere, where the corner `(mu, nu)` of the support and the Laurent
polynomial F are:

  * the gauge tower `E^n` (Creutzig, `exact_characters.deven_gauged_xn_qn`):
        F = q^{-p/2},  corner (0, 1);
  * the curve `(0, 2j + 1)`, `1 <= j <= k`, with `A = k + 1 - j`:
        F = (-1)^{p+j} q^{-A} sigma^A (chi_j(w) - sigma^{-1} chi_{j-1}(w)),  corner (1, 1);
  * a nested pair — the charge -1 curve of length `2j` inside the other
    curve, `g1` / `g2` marked points strictly between the two at their
    starts / ends:
        F = q^{-p/2} eps sigma^{(g1-g2)/2} sum_{i=0}^{j-1} q^{2i} B_{(g1+g2)/2 + 1 + 2i},
        B_r = q chi_{r+1}(w) + q^{-1} chi_{r-1}(w) - chi_1(sigma) chi_r(w),
        eps = (-1)^{(g1-g2)/2},  corner (0, 1);
  * a side-by-side pair — the charge -1 curve of length `2a`, the charge +1
    curve of length `2b`, `g1` marked points from the end of the first to the
    start of the second, `g2` the other gap, `g = g1 + g2`:
        F = (-1)^{g/2} q^{-g/2 - 2} sigma^{(g2-g1)/2} chi_{a-1}(w) chi_{b-1}(w)
            (chi_1(w) - chi_1(sigma)),  corner (0, 0).

Summed over the lattice points of the support, `n = a - b + mu`,
`e = a + b + nu` with `a, b >= 0`, each monomial of F contributes a double sum
of the shape `sum_{a,b>=0} q^{2pab + (linear in a, b)} (xz)^a (z/x)^b`
(`x` the variable of the sum over n).  That shape resembles the partial sums
in Kronecker's identity for Appell-Lerch sums; the resemblance is of the double
sum's shape only, and no identification with specific module characters is
claimed.  The loop around the puncture enters exactly as an arc of the same
length.  The rules are uniform in k: the configuration (lengths and gaps)
fixes F up to the factors shown.

Status: MEASURED, not derived.  Against the exact transport
(`DevenTraceTransport`, whose own stopping rule is a measured hypothesis with
a guard, and whose A1Dodd closed forms are certified to q^40 at A1Dodd k = 1, 2
and to q^24 at A1Dodd k = 3 — A1Dodd k = 0 is not in that record beyond the
gauge-tower control — while the transport evaluates them far beyond the
requested depth; its docstring): 1,583 evaluations at k = 1..5 plus a k = 6
sample, 0 disagreeing; every k = 5 evaluation and the k = 4 curves and nested
pairs were never used in forming the rules (the research script has the
windows).  `seed_trace` returns None on anything that is not a seed.
"""
from __future__ import annotations

from functools import lru_cache

# Marked point 2 of the (2k+2)-gon: the one the RG flow into A1Dodd removes
# (module docstring; `u1a1deven_geometric_frame._MERGED_VERTEX`).
MERGED_VERTEX = 2


# ---------------------------------------------------------------------------
# curve-frame conventions
# ---------------------------------------------------------------------------

def n_marked(k: int) -> int:
    return 2 * k + 2


def curve_charge(c, k: int) -> int:
    """Magnetic charge `c0` of the curve `c = (x, l)` of the (2k+2)-gon."""
    x, l = c
    if l % 2:
        return 0
    return -1 if (x - MERGED_VERTEX) % 2 == 0 else 1


def _canon(curves, k):
    """Curves as a sorted tuple of `((x mod n, l), m)`."""
    n = n_marked(k)
    acc = {}
    for c, m in curves:
        key = (c[0] % n, c[1])
        acc[key] = acc.get(key, 0) + m
    return tuple(sorted((c, m) for c, m in acc.items() if m))


def rho(curves, e: int, k: int):
    """`rho(L_curves . E^e) = L_{rotated curves} . E^{-e + drift}`."""
    n = n_marked(k)
    out, drift = [], 0
    for (x, l), m in curves:
        d = -sum(1 for v in (x % n, (x + l) % n) if v == 1)
        out.append((((x + 1) % n, l), m))
        drift += m * d
    return _canon(out, k), -e + drift


# ---------------------------------------------------------------------------
# the polynomials P
# ---------------------------------------------------------------------------

def _curve_S(j: int):
    """`S_j(x, y) = sum_{i=0}^{2j} (-1)^i x^{j - 2 ceil(i/2)} y^{j - 2 floor(i/2)}`
    as {(ex, ey): coefficient}: the curve rule written in the lattice
    variables (`x = q X`, `y = Y`, `S_j = chi_j(xy) - (y/x) chi_{j-1}(xy)`),
    kept as a cross-check of `curve_F`."""
    out = {}
    for i in range(2 * j + 1):
        ex = j - 2 * ((i + 1) // 2)
        ey = j - 2 * (i // 2)
        out[(ex, ey)] = out.get((ex, ey), 0) + (-1) ** i
    return out


def curve_F(k: int, j: int) -> dict:
    """F of the curve `(0, 2j + 1)`, `1 <= j <= k` (module docstring)."""
    if not 1 <= j <= k:
        raise ValueError(f"curve_F: need 1 <= j <= k, got k={k}, j={j}")
    p = k + 1
    A = k + 1 - j
    sg = -1 if (p + j) % 2 else 1
    F: dict = {}
    for we, c in _chi(j).items():
        F[(-2 * A, we, A)] = F.get((-2 * A, we, A), 0) + sg * c
    for we, c in _chi(j - 1).items():
        F[(-2 * A, we, A - 1)] = F.get((-2 * A, we, A - 1), 0) - sg * c
    return {key: c for key, c in F.items() if c}


@lru_cache(maxsize=None)
def curve_P(k: int, j: int):
    """`(corner, P)` of the curve `(0, 2j + 1)`."""
    return (1, 1), _F_to_P(curve_F(k, j), (1, 1))


def gauge_P(k: int):
    """`(corner, P)` of the gauge tower `E^n` (Creutzig's closed form)."""
    return (0, 1), ((0, 0, -(k + 1), 1),)


def _chi(r: int) -> dict:
    """SU(2) character `chi_r(w) = w^r + w^{r-2} + ... + w^{-r}` as {exponent: 1}
    (`chi_{-1} = 0`)."""
    return {r - 2 * m: 1 for m in range(r + 1)} if r >= 0 else {}


def _pmul(A: dict, B: dict) -> dict:
    out: dict = {}
    for a, ca in A.items():
        for b, cb in B.items():
            out[a + b] = out.get(a + b, 0) + ca * cb
    return {k: v for k, v in out.items() if v}


def _F_to_P(F: dict, corner) -> tuple:
    """`F = {(q2, w, s): c}` meaning `c q^{q2/2} w^w sigma^s` (w = q^e,
    sigma = q^n) -> P as `(i, j, m2, c)` in the lattice coordinates of the
    support with corner `corner = (mu, nu)`:
    `w^w sigma^s = X^{w+s} Y^{w-s} q^{-[(i+j) nu + (i-j) mu]/2}`."""
    mu, nu = corner
    P: dict = {}
    for (q2, we, se), c in F.items():
        i, j = we + se, we - se
        m2 = q2 + (i + j) * nu + (i - j) * mu
        P[(i, j, m2)] = P.get((i, j, m2), 0) + c
    return tuple(sorted((i, j, m2, c) for (i, j, m2), c in P.items() if c))


def nested_F(k: int, inner: int, g1: int, g2: int) -> dict:
    """F of a nested pair (module docstring): inner curve of even length
    `inner = 2j`, `g1` / `g2` marked points strictly between the outer and the
    inner curve at their starts / ends; the corner of the support is (0, 1).

        F = q^{-p/2} eps sigma^{(g1-g2)/2} sum_{i<j} q^{2i} B_{(g1+g2)/2 + 1 + 2i},
        B_r = q chi_{r+1}(w) + q^{-1} chi_{r-1}(w) - chi_1(sigma) chi_r(w),
        eps = (-1)^{(g1-g2)/2}."""
    p = k + 1
    j = inner // 2
    eps = -1 if ((g1 - g2) // 2) % 2 else 1
    sh = (g1 - g2) // 2
    F: dict = {}

    def add(q2, we, se, c):
        key = (q2 - p, we, se + sh)
        F[key] = F.get(key, 0) + eps * c

    for i in range(j):
        r = (g1 + g2) // 2 + 1 + 2 * i
        for we, c in _chi(r + 1).items():
            add(2 * (2 * i + 1), we, 0, c)
        for we, c in _chi(r - 1).items():
            add(2 * (2 * i - 1), we, 0, c)
        for we, c in _chi(r).items():
            add(2 * (2 * i), we, 1, -c)
            add(2 * (2 * i), we, -1, -c)
    return {key: c for key, c in F.items() if c}


def side_F(k: int, l1: int, l2: int, g1: int, g2: int) -> dict:
    """F of a side-by-side pair: the charge -1 curve of length `l1 = 2a`, the
    charge +1 curve of length `l2 = 2b`, `g1` marked points from the end of
    the first to the start of the second, `g2` the other gap; the corner of
    the support is (0, 0):

        F = (-1)^{g/2} q^{-g/2 - 2} sigma^{(g2-g1)/2} chi_{a-1}(w) chi_{b-1}(w)
            (chi_1(w) - chi_1(sigma)),     g = g1 + g2."""
    a, b = l1 // 2, l2 // 2
    g = g1 + g2
    sg = -1 if (g // 2) % 2 else 1
    sh = (g2 - g1) // 2
    wpart = _pmul(_chi(a - 1), _chi(b - 1))
    F: dict = {}
    q2 = 2 * (-(g // 2) - 2)
    for we, c in _pmul(wpart, _chi(1)).items():
        key = (q2, we, sh)
        F[key] = F.get(key, 0) + sg * c
    for we, c in wpart.items():
        for se in (1, -1):
            key = (q2, we, sh + se)
            F[key] = F.get(key, 0) - sg * c
    return {key: c for key, c in F.items() if c}


def _arc_points(c, n):
    x, l = c
    return {(x + t) % n for t in range(l + 1)} if l < n else set(range(n))


def pair_configuration(k: int, curves):
    """`("nested", inner length, g1, g2, charge of the inner curve)` or
    `("side", l1, l2, g1, g2)` (`l1` the charge -1 curve, `g1` the gap after
    it) of a pair of curves of charges -1 and +1, or None."""
    n = n_marked(k)
    if len(curves) != 2 or any(m != 1 for _, m in curves):
        return None
    ch = {c: curve_charge(c, k) for c, _ in curves}
    if sorted(ch.values()) != [-1, 1]:
        return None
    cm = next(c for c, v in ch.items() if v == -1)
    cp = next(c for c, v in ch.items() if v == 1)
    sm, sp = _arc_points(cm, n), _arc_points(cp, n)
    if cp[1] == n or (sm < sp):
        outer, inner = cp, cm
    elif cm[1] == n or (sp < sm):
        outer, inner = cm, cp
    else:
        if sm & sp:
            return None                      # overlapping arcs: not a seed
        g1 = (cp[0] - (cm[0] + cm[1])) % n - 1
        g2 = (cm[0] - (cp[0] + cp[1])) % n - 1
        return ("side", cm[1], cp[1], g1, g2)
    y, L = outer
    x, l = inner
    g1 = (x - y) % n - 1
    g2 = ((y + L) - (x + l)) % n - 1
    return ("nested", l, g1, g2, ch[inner])


def _pair_rule(k: int, rep):
    """`(corner, P)` of a representative pair, or None (not covered)."""
    conf = pair_configuration(k, rep)
    if conf is None:
        return None
    if conf[0] == "nested":
        _, inner, g1, g2, inner_charge = conf
        if inner_charge != -1:          # measured with the charge -1 curve inside
            return None
        return (0, 1), _F_to_P(nested_F(k, inner, g1, g2), (0, 1))
    _, l1, l2, g1, g2 = conf
    return (0, 0), _F_to_P(side_F(k, l1, l2, g1, g2), (0, 0))


def _rep_P(k: int, curves):
    """`(corner, P)` of an orbit representative, or None."""
    if len(curves) == 1:
        (c, m), = curves
        if m == 1 and c[0] == 0 and c[1] % 2 == 1 and 3 <= c[1] <= 2 * k + 1:
            return curve_P(k, (c[1] - 1) // 2)
        return None
    return _pair_rule(k, curves)


def orbit_rep(curves, e: int, k: int):
    """`(rep, e')` with `Tr(L_curves E^e) = Tr(L_rep E^{e'})`: `rep` the
    lexicographically smallest element of the rho-orbit of the curve set (the
    representative the rules were measured on), reached by powers of rho;
    None if that representative has no known closed form."""
    cur, ee = _canon(curves, k), e
    best = None
    for _ in range(n_marked(k)):
        if best is None or cur < best[0]:
            best = (cur, ee)
        cur, ee = rho(cur, ee, k)
    if _rep_P(k, best[0]) is None:
        return None
    return best


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def numerator_from_P(corner, P, p: int, n: int, K: int):
    """`{e: {E: c}}` (e >= 1): the numerator coefficients `c_e` through q^K
    at E-power n, for the corner `(mu, nu)` of the support and the Laurent
    polynomial `P` (`(i, j, m2, c)`: `c q^{m2/2} X^i Y^j`, `X = q^a`,
    `Y = q^b`) of the module docstring's rule, summed over the lattice points
    of the support."""
    mu, nu = corner
    sg = -1 if (p * n) % 2 else 1
    out: dict = {}
    b0 = max(0, mu - n)
    # E2(b) is convex in b (leading coefficient 4p); stop past every vertex once
    # every monomial is beyond q^K
    c1 = nu - mu
    c2 = 2 * n + nu - mu
    bstar = b0
    for (i, j, m2, c) in P:
        # d/db [p (2b + c1)(2b + c2) + 2 i a + 2 j b], a = b + n - mu
        num = -(2 * p * (c1 + c2) + 2 * (i + j))
        bv = -(-num // (8 * p))
        bstar = max(bstar, bv)
    b = b0
    while True:
        a = b + n - mu
        e = a + b + nu
        base = p * (e * e - n * n)
        beyond = True
        for (i, j, m2, c) in P:
            E2 = base + m2 + 2 * i * a + 2 * j * b
            if E2 <= 2 * K:
                beyond = False
                if E2 % 2:
                    raise ValueError("numerator_from_P: half-integer exponent")
                if e == 0:
                    continue
                E = E2 // 2
                ee, ss = (e, sg * c) if e > 0 else (-e, -sg * c)
                slot = out.setdefault(ee, {})
                slot[E] = slot.get(E, 0) + ss
        if beyond and b >= bstar:
            break
        b += 1
    return {e: {E: c for E, c in d.items() if c}
            for e, d in out.items() if any(d.values())}


@lru_cache(maxsize=None)
def _inverse_weyl_product(K: int):
    """`1 / prod_{i>=1} (1 - z^2 q^{2i})(1 - z^-2 q^{2i})` through q^K, as a
    tuple of `(q, ((zpow, c), ...))`."""
    P = {0: {0: 1}}
    for i in range(1, K // 2 + 1):
        for s in (2, -2):
            new: dict = {}
            for q, zd in P.items():
                t = 0
                while q + 2 * i * t <= K:
                    slot = new.setdefault(q + 2 * i * t, {})
                    for w, c in zd.items():
                        slot[w + s * t] = slot.get(w + s * t, 0) + c
                    t += 1
            P = new
    return tuple((q, tuple(sorted(zd.items()))) for q, zd in sorted(P.items()))


def numerator_to_trace(ce: dict, K: int) -> dict:
    """`{q: {SU(2) highest weight: int}}` through q^K from the numerator
    coefficients: `N / (z - 1/z) = sum_e c_e chi_{e-1}`, times the inverse
    Weyl product."""
    Z: dict = {}
    for e, d in ce.items():
        for E, c in d.items():
            if E > K:
                continue
            zd = Z.setdefault(E, {})
            for w in range(-(e - 1), e, 2):
                zd[w] = zd.get(w, 0) + c
    T: dict = {}
    for q1, z1 in Z.items():
        for q2, z2 in _inverse_weyl_product(K):
            if q1 + q2 > K:
                break
            slot = T.setdefault(q1 + q2, {})
            for w1, c1 in z1.items():
                if not c1:
                    continue
                for w2, c2 in z2:
                    slot[w1 + w2] = slot.get(w1 + w2, 0) + c1 * c2
    out = {}
    for q, zd in T.items():
        if not any(zd.values()):
            continue
        top = max((w for w, c in zd.items() if c), default=None)
        if top is None:
            continue
        irr = {}
        for w in range(0, top + 1):
            mlt = zd.get(w, 0) - zd.get(w + 2, 0)
            if mlt:
                irr[w] = mlt
        if irr:
            out[q] = irr
    return out


def seed_numerator(k: int, curves, n: int, K: int):
    """Numerator `{e: {E: c}}` of `Tr(L_curves . E^n)` through q^K, or None
    when the label is not a seed with a known closed form."""
    hit = orbit_rep(curves, n, k)
    if hit is None:
        return None
    rep, nn = hit
    corner, P = _rep_P(k, rep)
    return numerator_from_P(corner, P, k + 1, nn, K)


def seed_trace(k: int, curves, n: int, K: int):
    """`{q: {SU(2) highest weight: int}}` of `Tr(L_curves . E^n)` through
    q^K from the closed form, or None when the label is not a seed with a
    known closed form (the caller then uses the transport)."""
    ce = seed_numerator(k, curves, n, K)
    if ce is None:
        return None
    return numerator_to_trace(ce, K)


def gauge_trace(k: int, n: int, K: int):
    """`Tr(E^n)` from the same machinery (the positive control against
    Creutzig's closed form)."""
    corner, P = gauge_P(k)
    return numerator_to_trace(numerator_from_P(corner, P, k + 1, n, K), K)
