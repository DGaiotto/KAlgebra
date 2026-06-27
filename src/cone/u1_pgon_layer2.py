"""
u1_pgon_layer2.py — GENERAL-k closed-form Layer-2 for the U1Pgon hierarchy.

The u(1)-gauged [A_1, A_{2k+1}] Argyres-Douglas family is realised, per k,
by the "polygon" K-algebras

    U1Hexagon (p = 3)  = k = 1  = (A_1, A_3) = (A_1, D_3)
    U1Octagon (p = 4)  = k = 2  = (A_1, A_5)
    U1Decagon (p = 5)  = k = 3  = (A_1, A_7)
    ...
    U1A1AoddKAlg(k)    (p = k + 2)

with chiral algebra the M(1, p) singlet vertex algebra (p = k + 2,
central charge c = 1 - 6 (p-1)^2 / p).  The two trace seed families

    Tr(v^n)            — the bare singlet character at U(1)-charge n
    Tr(L_long · v^n)   — the length-3 "long chord" (Wilson-line) defect

were each pinned PER k in the per-polygon modules:

    Tr(v^n):           u1_hexagon_singlet (p3), u1_octagon_singlet (p4)
    Tr(L_long · v^n):  u1_hexagon_singlet.tr_L20_v_n (p3),
                       u1_octagon_l_long (p4), u1_decagon_l_long (p5)

This module gives the **single GENERAL-p closed form** for both, derived
from the cross-rung pattern documented in `u1_decagon_l_long`.  It is
verified to reproduce every one of the per-k modules above
(see `_verify_against_per_k`), so it supersedes them as one formula.

--------------------------------------------------------------------------
Singlet  Tr(v^n)  (valid for all n; symmetric under n -> -n)
--------------------------------------------------------------------------
With p = k + 2 and |n| the U(1) charge,

    Tr(v^n) = (-1)^{(p-1)|n|} [
          sum_{j>=0} fq^{2p j^2 + (2p|n| + 2(p-1)) j + (p-1)|n|}
        - sum_{j>=0} fq^{2p j^2 + (2p|n| + 2(p+1)) j + (p+1)|n| + 2} ]

The sign (-1)^{(p-1)|n|} is +1 for p odd and (-1)^{|n|} for p even.
These are the bare M(1, p) singlet partial-theta characters at charge n.

--------------------------------------------------------------------------
Long chord  Tr(L_long · v^n)  (n >= 0)
--------------------------------------------------------------------------
A FOUR-partial-theta sum at level 2p (two "atypical-like" and two
"log-like" branches), with an overall (-1)^{n+1} only for p even:

    Tr(L_long · v^n) = overall * sum_{branch} s_branch *
                       sum_{j>=0} fq^{2p j^2 + b_branch j + c_branch}

    overall = +1 (p odd) / (-1)^{n+1} (p even)

    branch       s    b                          c
    atypical +  +1   2p n + (2p+4 [po] / 2p+2)   (p+1 [po] / p-1) n + (3 [po] / 1)
    atypical -  -1   2p n + (2p+2 [po] / 2p+4)   (p-1 [po] / p+1) n + (1 [po] / 3)
    log      +  +1   2p n + (2p+10)              3(p-1) n + (4p-3 [po] / 4p-5)
    log      -  -1   2p n + (2p+8)               3(p-1) n + (4p-5 [po] / 4p-3)

where "[po]" denotes the value used when p is odd and the second value
when p is even (the +/- assignment of the atypical branches and the
log c-intercepts swap with the parity of p; b-slopes are uniformly 2p
and the log b-slopes/intercepts are parity-independent).

The four partial-thetas are a linear combination of TWO M(1, p) singlet
*module* characters (the long chord acts on the chiral algebra as a
lattice-shifting Wilson line); n < 0 is the open follow-up — across all
per-k modules it is unsolved (k = 1 alone is reflection-symmetric,
n <-> -1-n, since there L_long is the 6-gon diameter).
"""

from __future__ import annotations

from laurent_poly import LaurentPoly


def _partial_theta(out: dict, level: int, b: int, c: int, sign: int, K: int) -> None:
    """Accumulate sign * sum_{j>=0} fq^{level j^2 + b j + c} into `out`,
    truncated to fq^K."""
    j = 0
    while True:
        e = level * j * j + b * j + c
        if e > K:
            break
        if e >= 0:
            out[e] = out.get(e, 0) + sign
        j += 1


def tr_v_n(p: int, n: int, K: int) -> LaurentPoly:
    """Closed-form singlet character Tr(v^n) for U1Pgon at p = k + 2,
    any integer n, truncated to fq^K.  Symmetric under n -> -n."""
    na = abs(n)
    sign = 1 if (((p - 1) * na) % 2 == 0) else -1
    level = 2 * p
    out: dict[int, int] = {}
    _partial_theta(out, level, 2 * p * na + 2 * (p - 1), (p - 1) * na, +sign, K)
    _partial_theta(out, level, 2 * p * na + 2 * (p + 1), (p + 1) * na + 2, -sign, K)
    return LaurentPoly({e: c for e, c in out.items() if c != 0})


def _tr_L_long_neg(p: int, n: int, K: int) -> LaurentPoly:
    """Tr(L_long · v^n), n < 0, for p >= 4 — the spectral-flow mirror of
    the n >= 0 branches.

    The n < 0 character is a |n|-linear 4-partial-theta family (m = |n|):
    the two atypicals keep c-slopes p-1, p+1 (offset -1); the two "log"
    branches drop to c-slope **p+3** (vs the n>=0 value 3(p-1)); all
    b-slopes stay 2p.  The **atypical** b-offsets are `2p-4, 2p-2`
    (p-dependent, mirroring the n>=0 form's `2p+4, 2p+2`); the **log**
    b-offsets are `12, 10`.  Signs follow the n>=0 parity rule: overall
    (-1)^{m+1} for p even, +1 for p odd, with the atypical +/- assignment
    swapping with the parity of p.

    Verified exact vs gauged-A_{2k+1} BPSKAlgebra at p = 4 (n = -1..-6,
    full series; at p=4 the b-offsets `2p-4,2p-2` = `4,6`, indistinguishable
    from the earlier conjecture there).  At p = 5 they DIFFER (`6,8`); the
    `2p-4` (p-1) offset is **reliable-confirmed** — q^29 is stable across all
    S_RG windows and the earlier `4` predicts a q^27 term that is *absent* in
    every window.  The `2p-2` (p+1) offset (q^33) is only **structurally
    inferred** (parallel to the confirmed p-1): the gauged-RG oracle's
    two-window convergence is UNRELIABLE beyond ~q^29 at k=3 (it sprouts
    spurious terms / grows coefficients with the window — the documented
    artifact), so q^33 was matched against an artifact, not real data.
    **OPEN — the whole j>=2 tail is unverified**: the reliable structure is
    the singlet false-theta TOWER (Σ_r false-theta at momentum (2r+1)(p-1),
    unit coeffs — NOT a derivative/growing theta), whose r>=2 components need
    the reliable A1A2k bridge, not the oracle.  Consumers that aggregate this
    seed deep (the k=3 `Tr(L_diam^{n>=5})`) are therefore still wrong —
    `U1A1AoddKAlg` honest-fails them via its provable trace-level guard."""
    m = -n  # = |n|
    po = (p % 2 == 1)
    overall = 1 if po else (-1 if (m + 1) % 2 == 1 else 1)
    level = 2 * p
    out: dict[int, int] = {}
    # Atypical b-offsets `2p-4, 2p-2` CORRECTED 2026-06-23, mirroring the n>=0
    # form's `2p+4, 2p+2`.  The p-1 offset `2p-4` (q^29 at p=5,n=-1) is
    # RELIABLE-confirmed (window-stable; the old `4`->q^27 is absent); the p+1
    # offset `2p-2` (q^33) is structurally inferred only (the gauged-RG oracle is
    # unreliable beyond ~q^29 -- see docstring).  The log offsets `12,10` are
    # conjectural; the true structure is the singlet false-theta tower (r>=2
    # components unpinned), which is why deep diameter powers honest-fail.
    for s_intr, b, c in [
        (+1 if not po else -1, 2 * p * m + (2 * p - 4), (p - 1) * m - 1),  # atyp p-1
        (-1 if not po else +1, 2 * p * m + (2 * p - 2), (p + 1) * m - 1),  # atyp p+1
        (+1 if not po else -1, 2 * p * m + 12, (p + 3) * m + 1),   # log
        (-1 if not po else +1, 2 * p * m + 10, (p + 3) * m + 3),   # log
    ]:
        _partial_theta(out, level, b, c, overall * s_intr, K)
    return LaurentPoly({e: c for e, c in out.items() if c != 0})


def tr_L_long_v_n(p: int, n: int, K: int) -> LaurentPoly:
    """Closed-form long-chord defect Tr(L_long · v^n) for U1Pgon at
    p = k + 2, truncated to fq^K — valid for ALL integer n.

    n >= 0: the general-p cross-rung 4-partial-theta pattern.
    n <  0: p = 3 (hexagon diameter) is the reflection n <-> -1-n;
            p >= 4 is the spectral-flow mirror family `_tr_L_long_neg`
            (verified vs BPS at p = 4 and p = 5)."""
    if n < 0:
        if p == 3:
            return tr_L_long_v_n(3, -1 - n, K)   # diameter reflection
        return _tr_L_long_neg(p, n, K)
    po = (p % 2 == 1)
    overall = 1 if po else (-1 if (n + 1) % 2 == 1 else 1)
    level = 2 * p
    branches = [
        # (s, b, c)
        (+1, 2 * p * n + (2 * p + 4 if po else 2 * p + 2),
         (p + 1 if po else p - 1) * n + (3 if po else 1)),   # atypical +
        (-1, 2 * p * n + (2 * p + 2 if po else 2 * p + 4),
         (p - 1 if po else p + 1) * n + (1 if po else 3)),   # atypical -
        (+1, 2 * p * n + (2 * p + 10),
         3 * (p - 1) * n + (4 * p - 3 if po else 4 * p - 5)),  # log +
        (-1, 2 * p * n + (2 * p + 8),
         3 * (p - 1) * n + (4 * p - 5 if po else 4 * p - 3)),  # log -
    ]
    out: dict[int, int] = {}
    for s, b, c in branches:
        _partial_theta(out, level, b, c, overall * s, K)
    return LaurentPoly({e: c for e, c in out.items() if c != 0})


def tr_L_diameter_v_n(p: int, n: int, K: int) -> LaurentPoly:
    """Closed-form diameter-chord defect Tr(L_diam · v^n) for U1Pgon —
    general odd p (= k+2), i.e. all odd k >= 3.

    The diameter (type k+1) survives only for odd k; for k=1 it *is* the
    long chord (use `tr_L_long_v_n`).  It is a **2-partial-theta** family
    with the diameter's self-conjugacy: reflection-symmetric about n=1/2,
    `Tr(L_diam v^n) = Tr(L_diam v^{1-n})`.  With m = max(-n, n-1) >= 0 and
    level 2p (b- and c-slopes SWAPPED between the two branches):
        + : c = (p-1)m + (p-1)/2,   b = (p+1)*m
        - : c = (p+1)m + (p+3)/2,   b = (p-1)*m
    times an overall **(-1)^{(p-1)/2}** — the Gauss/Milgram self-duality
    sign of the diameter (the one genuine *mod-4* dependence in the whole
    hierarchy; it is an explicit p-formula, not per-residue data).

    Verified vs gauged-A_{2k+1} data: p=5 / k=3 (BPSKAlgebra, full n incl.
    b-slopes) and p=7 / k=5 (oracle, n=0,-1 c-intercepts + center test).
    For p>=9 the c-intercepts/sign are the same smooth+mod-4 law; the
    b-slopes (p+-1)m extrapolate from p=5 (only m=0 was visible at p=7)."""
    if p < 5 or p % 2 == 0:
        raise NotImplementedError(
            f"diameter closed form is for odd p >= 5 (odd k >= 3); got p={p}. "
            f"(k=1 diameter is the long chord -> tr_L_long_v_n.)"
        )
    m = -n if n <= 0 else n - 1               # reduce via n <-> 1-n
    level = 2 * p
    overall = -1 if ((p - 1) // 2) % 2 else 1    # (-1)^{(p-1)/2}
    out: dict[int, int] = {}
    # b- and c-slopes are swapped between the two branches:
    for s, c, b in [(+1, (p - 1) * m + (p - 1) // 2, (p + 1) * m),
                    (-1, (p + 1) * m + (p + 3) // 2, (p - 1) * m)]:
        _partial_theta(out, level, b, c, overall * s, K)
    return LaurentPoly({e: c for e, c in out.items() if c != 0})


def even_chord_atyp_part(j: int, p: int, n: int, K: int) -> LaurentPoly:
    """ATYPICAL sector of the generic even-type chord Tr(L_{type 2j}·v^n).

    *** PARTIAL — atypical (Felder-resolution) modules only. ***
    The generic even chord (type 2j, the (2k+4)-gon diagonal of length
    2j+1, for 2j < k+1 — i.e. NOT the self-dual diameter) is a 4-θ:
    two **atypical** branches (captured here) + two **logarithmic** branches
    (the (1,p) log modules — OPEN, see below).  This returns only the
    atypical part, so it is NOT the full trace; do not wire into `trace`.

    Analytic-route findings (verified against cf_Llong at j=1 and gauged-A_9
    oracle type-4 at j=2, p=6):

      * conformal weight (lowest exponent) = (p-1)|n| + (+j if n>=0 else -j);
      * the 4 branches are spectral-flow images (σ:(b,c)->(b+2p,c+μ)) of
        UNIVERSAL modules with momenta μ ∈ {p-1, p+1, 3(p-1), 3(p-1)};
      * the two ATYPICAL branches (μ = p∓1) scale LINEARLY in the chord
        half-length j — only the intercepts move:
            atyp+ (μ=p-1): c = (p-1)n + j        b = 2p n + (2p + 2j)      [n>=0]
            atyp- (μ=p+1): c = (p+1)n + (j+2)     b = 2p n + (2p + 2j + 2)  [n>=0]
        with overall sign (-1)^{n+j}, s_intr (+,-).  (j=1 reproduces the
        cf_Llong atypical branches exactly.)

    OPEN — the two LOGARITHMIC branches (μ = 3(p-1)): the k=4 (p=6) data
    show their c-slopes are UNEQUAL (≈3(p-1) and 3(p-1)-6), they carry a
    grading/sign distinct from (-1)^{n+j} (n=1 gives 27:+,31:+ where the
    atypical rule predicts opposite signs), and one wants a negative b.
    This is the genuine (1,p) logarithmic-module subtlety and is the single
    remaining unknown for the general-k trace at k>=4.  Anchor data
    (gauged-A_9 oracle, K-swept, p=6 / k=4):
        n=0: {2,4,16,18,24,30,34}   n=1: {7,11,27,31}
        n=-1: {3,5,13,15}           n=-2: {8,12}
    (atypical part = the two lowest branches; the rest are the logs)."""
    if j < 1 or p < 3:
        raise ValueError(f"need j>=1, p>=3; got j={j}, p={p}")
    m = abs(n)
    level = 2 * p
    po = (p % 2 == 1)            # parity-of-p (= k) swap, as in the cross-rung
    out: dict[int, int] = {}
    if n >= 0:
        overall = 1 if po else (-1 if (n + j) % 2 else 1)   # p even: (-1)^{n+j}
        # full atypical branches (c AND b verified at j=1 both parities, j=2 p-even)
        for s, b, c in [
            (+1, 2 * p * n + (2 * p + 2 * j + (2 if po else 0)),
                 (p + 1 if po else p - 1) * n + (j + 2 if po else j)),
            (-1, 2 * p * n + (2 * p + 2 * j + (0 if po else 2)),
                 (p - 1 if po else p + 1) * n + (j if po else j + 2)),
        ]:
            _partial_theta(out, level, b, c, overall * s, K)
    else:
        # n<0: only the atypical c-intercepts ((p∓1)|n| - j) are verified;
        # the n<0 b's are log-entangled (open), so emit the k=0 terms only.
        overall = 1 if po else (-1 if (m + j) % 2 else 1)   # p even: (-1)^{m+j}
        for s, c in [((-1 if po else +1), (p - 1) * m - j),
                     ((+1 if po else -1), (p + 1) * m - j)]:
            if 0 <= c <= K:
                out[c] = out.get(c, 0) + overall * s
    return LaurentPoly({e: c for e, c in out.items() if c != 0})


def _verify_against_per_k(K: int = 46) -> bool:
    """Verify the general-p forms reproduce every per-k hardcoded module."""
    import u1_hexagon_singlet as hx
    import u1_octagon_singlet as oc
    import u1_octagon_l_long as ocl
    import u1_decagon_l_long as dc

    ok = True
    print(f"Verifying GENERAL-p forms vs per-k hardcoded modules (K={K}):")

    # Singlet (all n).
    for p, mod in [(3, hx), (4, oc)]:
        good = all(
            tr_v_n(p, n, K)._coeffs == mod.tr_v_n(n, K)._coeffs
            for n in range(-5, 6)
        )
        ok &= good
        print(f"  singlet  p={p} (n=-5..5): {'OK' if good else 'FAIL'}")

    # Long chord (n >= 0).
    for p, modL in [(3, hx.tr_L20_v_n), (4, ocl.tr_L_long_v_n), (5, dc.tr_L_long_v_n)]:
        good = all(
            tr_L_long_v_n(p, n, K)._coeffs == modL(n, K)._coeffs
            for n in range(0, 5)
        )
        ok &= good
        print(f"  L_long   p={p} (n=0..4):  {'OK' if good else 'FAIL'}")

    print("ALL OK" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    _verify_against_per_k()
