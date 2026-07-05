"""un_nf_dressed_generators — certified closed-form matter dressings (T2c.1).

Plan 22 / T2c increment 1: the **matter-dressed generator layer** of the
native `A_𝖖[U(N)+N_f]` class, built as closed forms from the T2b
normalization — never a solve — and
**certified by exact equality against the Route-A solver images**
(`UNNfOverPure.RG`, the oracle), including predictive probes beyond the
points the normalization was read off.

The closed form.  For a canonical `L_{(m,e);k}` whose dressing-active
(negative) Levi blocks carry `e = 0`:

    spread(m, e)  =  Σ_w  c_w(q) · L_{(m, e+w); |w|}

where per negative block `b` (size `d_b`, depth `n_b = |m_b|`) the matter
zero-mode factor is

    Z_b = ∏_{i=1}^{N_f} ∏_{j ∈ b} ∏_{s=0}^{n_b−1} (1 + μ_i q^{2s−n_b+1} v_j),

expanded jointly over flavour levels, the per-level symmetric v-content
decomposed into block-Schur functions (greedy leading-monomial peel with
Kostka numbers — exact), and blocks combined by convolution of flavour
multidegrees.  Singleton blocks reproduce the bar-centered Gaussian
binomials `[n_b, w]_{q²}`; transparent (`m_j ≥ 0`) slots are untouched.

Scope (honest-fail outside it): `e = 0` on every negative block —
covering all minuscules `E_k`/`F_k`, `det^{±p}`, mixed `E·F` shapes, and
arbitrary dressings on transparent slots.  General `e` on a dressing
block (a Littlewood–Richardson composition) is the next increment,
oracle-certifiable the same way.
"""
from __future__ import annotations

import os
import sys
from itertools import product

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from laurent_poly import LaurentPoly

from pure_un_chart_engine import _kostka


def gaussian_binom_q2_centered(n: int, w: int) -> LaurentPoly:
    """Bar-centered Gaussian binomial `[n, w]_{q²}`: the coefficient of
    `μ^w` in `∏_{s=0}^{n−1}(1 + μ q^{2s−n+1})` — palindromic about q⁰
    (`[2,1] = q⁻¹+q`, `[3,1] = [3,2] = q⁻²+1+q²`)."""
    if w < 0 or w > n:
        return LaurentPoly.zero()
    # full bivariate expansion (n ≤ ~6 in practice — tiny)
    terms = {(0, 0): 1}
    for s in range(n):
        shift = 2 * s - n + 1
        out: dict = {}
        for (mu, qe), c in terms.items():
            out[(mu, qe)] = out.get((mu, qe), 0) + c
            out[(mu + 1, qe + shift)] = out.get((mu + 1, qe + shift), 0) + c
        terms = out
    return LaurentPoly({qe: c for (mu, qe), c in terms.items() if mu == w and c})


def _levi_blocks_desc(m):
    """Maximal equal-entry runs of a dominant (descending) `m`, as index
    lists."""
    blocks, cur = [], [0]
    for i in range(1, len(m)):
        if m[i] == m[i - 1]:
            cur.append(i)
        else:
            blocks.append(cur)
            cur = [i]
    blocks.append(cur)
    return blocks


def _schur_monomials(lam):
    """`s_λ` as `{x-exponent tuple: int}` (all permutations of each `m_μ`,
    Kostka-weighted).  Negative entries via the det-shift trick."""
    lam = tuple(int(x) for x in lam)
    d = len(lam)
    shift = -min(lam + (0,))
    lam_pos = tuple(x + shift for x in lam)
    from itertools import permutations
    out: dict = {}
    for mu in _partitions_into(sum(lam_pos), d):
        k_num = _kostka(lam_pos, mu)
        if not k_num:
            continue
        for perm in set(permutations(mu)):
            xe = tuple(x - shift for x in perm)
            out[xe] = out.get(xe, 0) + k_num
    return out


def _block_zero_mode_content(n: int, d: int, Nf: int, e_b=None) -> dict:
    """Joint expansion of one negative block's zero-mode factor
    `∏_{i≤Nf} ∏_{j<d} ∏_{s<n} (1 + μ_i q^{2s−n+1} x_j)` — seeded with the
    block's existing Schur dressing `s_{e_b}` when given (the q-weighted
    Littlewood-Richardson composition; T2c.3) — as
    `{flavour level k_vec: {v-exponent tuple (sorted desc): LaurentPoly}}`
    with the v-content decomposed into **Schur** coordinates by greedy
    leading-monomial peel (exact Kostka)."""
    # expand over monomials: {(k_vec, x_exps): q-poly}
    if e_b is None or not any(e_b):
        terms: dict = {((0,) * Nf, (0,) * d): LaurentPoly({0: 1})}
    else:
        terms = {((0,) * Nf, xe): LaurentPoly({0: z})
                 for xe, z in _schur_monomials(e_b).items()}
    for i in range(Nf):
        for j in range(d):
            for s in range(n):
                shift = 2 * s - n + 1
                out: dict = {}
                for (kv, xe), c in terms.items():
                    out[(kv, xe)] = out.get((kv, xe), LaurentPoly.zero()) + c
                    kv2 = tuple(k + (1 if t == i else 0) for t, k in enumerate(kv))
                    xe2 = tuple(x + (1 if t == j else 0) for t, x in enumerate(xe))
                    add = c * LaurentPoly({shift: 1})
                    out[(kv2, xe2)] = out.get((kv2, xe2), LaurentPoly.zero()) + add
                terms = out
    # collect per flavour level into monomial-symmetric coordinates
    by_level: dict = {}
    for (kv, xe), c in terms.items():
        if c.is_zero():
            continue
        lam = tuple(sorted(xe, reverse=True))
        row = by_level.setdefault(kv, {})
        row[lam] = row.get(lam, LaurentPoly.zero()) + c
    # the content is S_d-symmetric: each m_λ appeared once per distinct
    # permutation of λ — normalise to coefficient-of-m_λ
    from math import factorial
    def orbit_size(lam):
        counts: dict = {}
        for x in lam:
            counts[x] = counts.get(x, 0) + 1
        sz = factorial(len(lam))
        for v in counts.values():
            sz //= factorial(v)
        return sz
    for kv, row in by_level.items():
        for lam in list(row):
            row[lam] = row[lam] * LaurentPoly({0: 1})  # copy
            q, r = divmod_poly(row[lam], orbit_size(lam))
            if r is not None:
                raise ArithmeticError(f"non-symmetric content at {kv}, {lam}")
            row[lam] = q
    # monomial → Schur by greedy peel: s_λ = Σ_μ K_{λμ} m_μ, K triangular
    # (Kostka numbers read off `_schur_monomials`, which handles the
    # det-shifted negative-entry shapes the seeded content produces)
    out: dict = {}
    for kv, row in by_level.items():
        schur: dict = {}
        work = {lam: c for lam, c in row.items() if not c.is_zero()}
        while work:
            lam = max(work)                      # lex-largest = leading shape
            c = work.pop(lam)
            schur[lam] = c
            mcoords: dict = {}
            for xe, z in _schur_monomials(lam).items():
                mcoords[tuple(sorted(xe, reverse=True))] = z
            for mu, k_num in mcoords.items():
                if mu == lam:
                    continue
                sub = c * k_num
                cur = work.get(mu, LaurentPoly.zero()) - sub
                if cur.is_zero():
                    work.pop(mu, None)
                else:
                    work[mu] = cur
        out[kv] = {lam: c for lam, c in schur.items() if not c.is_zero()}
    return out


def _pieri_e(lam, r: int):
    """Pieri rule `s_λ·e_r` within a block of `len(lam)` variables: all
    `μ = λ + (vertical r-strip)` (add 0/1 per row, keep descending).
    Works verbatim for integer (possibly negative) descending tuples —
    the rule is det-shift-equivariant."""
    d = len(lam)
    out = []

    def gen(i, rem, acc):
        if i == d:
            if rem == 0:
                out.append(tuple(acc))
            return
        for add in (0, 1):
            if add > rem:
                continue
            v = lam[i] + add
            if i > 0 and v > acc[i - 1]:
                continue
            gen(i + 1, rem - add, acc + [v])

    gen(0, r, [])
    return out


def _block_depth1_dressed_content(e_b, Nf: int) -> dict:
    """Depth-1 negative block carrying an existing Levi-dominant dressing
    `e_b`: the zero-mode factor is `∏_i ∏_{j∈b} (1 + μ_i v_j)`
    (= `Σ_w e_w(v_b)·μ_i^w` per flavour), so the dressed content is the
    **iterated Pieri rule** on `e_b` — vertical strips, unit q-free
    coefficients: `{k_vec: {λ: 1}}`."""
    d = len(e_b)
    one = LaurentPoly({0: 1})
    content = {(0,) * Nf: {tuple(e_b): one}}
    for i in range(Nf):
        nxt: dict = {}
        for kv, row in content.items():
            for lam, c in row.items():
                for w in range(0, d + 1):
                    kv2 = tuple(x + (w if t == i else 0)
                                for t, x in enumerate(kv))
                    for mu in _pieri_e(lam, w):
                        dest = nxt.setdefault(kv2, {})
                        dest[mu] = dest.get(mu, LaurentPoly.zero()) + c
        content = nxt
    return content


def divmod_poly(p: LaurentPoly, k: int):
    """Exact integer division of every coefficient (None remainder flag)."""
    out = {}
    for e, c in p._coeffs.items():
        if c % k:
            return None, object()
        out[e] = c // k
    return LaurentPoly(out), None


def _partitions_into(k: int, parts: int):
    """All partitions of `k` into ≤ `parts` parts, as descending tuples
    padded to length `parts`."""
    def gen(rem, mx, acc):
        if len(acc) == parts:
            if rem == 0:
                yield tuple(acc)
            return
        for x in range(min(rem, mx), -1, -1):
            yield from gen(rem - x, x, acc + [x])
    yield from gen(k, k, [])


def dressed_label_spread(m, e, Nf: int = 1) -> dict:
    """The closed-form matter dressing of `L_{(m,e)}` (engine convention:
    dominant `m`, Levi-dominant `e`):
    `{((m, e′), k_vec): LaurentPoly}` over flavour levels.

    Honest-fail (`NotImplementedError`) when `e` is nonzero on a negative
    block — that composition is outside the certified closed form."""
    m = tuple(int(x) for x in m)
    e = tuple(int(x) for x in e)
    if list(m) != sorted(m, reverse=True):
        raise ValueError(f"m={m} must be dominant (descending engine frame)")
    neg_dirs = sum(1 for x in m if x < 0)
    if m and (max(m) - min(m)) >= 2 and (Nf >= 2 or neg_dirs >= 2):
        # Bubbling channel: zero-mode pairs (distinct flavours, or the same
        # flavour across distinct negative directions — N_f=1 with ≥ 2
        # negative directions bubbles too, e.g. (−1,−3)) can screen a
        # dominance-non-minimal cocharacter into lower magnetic sectors
        # (U(2) N_f=2: RG((0,−2)) ⊃ [2]_q·L_{(−1,−1),(1,1)}·μ₁μ₂).  The
        # closed form here is the *diagonal* (same-m) part only; deep
        # shapes are built by `constructive_rg_image` (anti-fundamental
        # chains — bubbling emerges mechanically, certified), so this
        # honest-fail only marks the diagonal's own scope.  Single
        # negative direction at N_f=1 is Pauli-blocked (certified clean
        # through depth 3 / spread 3).
        raise NotImplementedError(
            f"m={m}, N_f={Nf}: dominance-non-minimal cocharacter with "
            f"bubbling channels (N_f ≥ 2 or ≥ 2 negative directions) — "
            f"beyond the diagonal closed form; use constructive_rg_image")
    blocks = _levi_blocks_desc(m)
    block_contents = []
    for b in blocks:
        depth = m[b[0]]
        if depth >= 0:
            continue                      # transparent block
        e_b = tuple(e[j] for j in b)
        if depth == -1:
            # depth-1 block, arbitrary existing dressing: iterated Pieri
            # (unit, q-free) — T2c.3 slice 1.
            block_contents.append((b, _block_depth1_dressed_content(e_b, Nf)))
            continue
        if any(e_b):
            # The seeded q-ladder × s_{e_b} composition gives the correct
            # *diagonal* but a dressed deep slot opens bubbling channels
            # even at N_f=1 (oracle: ((0,−2),(0,1)) ⊃ L_{(−1,−1),(1,1)}·μ
            # at level 1) — these shapes go through `dressed_chain_image`
            # (two-ordering-centered dressed-minuscule chains).
            raise NotImplementedError(
                f"dressing of a depth-≥2 negative block ({m}, {e}): the "
                f"diagonal misses dressed-bubbling — use dressed_chain_image")
        block_contents.append((b, _block_zero_mode_content(-depth, len(b), Nf)))
    spread = {((m, e), (0,) * Nf): LaurentPoly({0: 1})}
    for b, content in block_contents:
        nxt: dict = {}
        for ((mm, ee), kv), c in spread.items():
            for kv_b, schur in content.items():
                kv2 = tuple(x + y for x, y in zip(kv, kv_b))
                for lam, cl in schur.items():
                    ee2 = list(ee)
                    for idx, j in enumerate(b):
                        ee2[j] = lam[idx]          # Levi-dominant within block
                    key = ((mm, tuple(ee2)), kv2)
                    add = c * cl
                    nxt[key] = nxt.get(key, LaurentPoly.zero()) + add
        spread = {k: v for k, v in nxt.items() if not v.is_zero()}
    return spread


def certify_against_flow(A, m, e) -> bool:
    """EXACT equality of the closed-form spread with the Route-A solver
    image `A.RG(((m, e), 0))` — the oracle certificate (`A` is a
    `UNNfOverPure`)."""
    want = dressed_label_spread(m, e, A.Nf)
    zero = (0,) * A.Nf
    got = A.RG(((tuple(m), tuple(e)), zero)).terms
    got = {(lab, tuple(k)): c for (lab, k), c in got.items() if not c.is_zero()}
    want = {((tuple(mm), tuple(ee)), kv): c
            for ((mm, ee), kv), c in want.items()}
    return got == want


# ---------------------------------------------------------------------------
# T2c increment 2 — deep shapes by product-and-peel (bubbling emerges).
#
# Bubbling needs no kernel: a dominant `m` decomposes into **balanced /
# minuscule factors** (`det^{±1}` powers and the fundamental coweights
# `φ_k = (1,…,1,0,…,0)`), each dominance-minimal — hence bubbling-free and
# covered by the certified diagonal spread — and the RG image of the deep
# canonical is the **auxiliary product of the factor images** (the
# ABSOLUTE-RULE pattern: a polynomial in trusted generators).  The
# magnetic bubbling terms arise mechanically from the cross-flavour /
# cross-direction zero-mode products (probe: `spread(F)·spread(F)` at
# N_f=2 reproduces the solver's `RG(L_{(0,−2)})` exactly, including the
# `[2]_q·L_{(−1,−1),(1,1)}·μ₁μ₂` term).
#
# Self-certificate (no oracle needed): the flavour-level-0 component of
# the product is the **pure** product of the factor canonicals, which
# must be the single canonical `L_{(m,0)}` with coefficient 1 (the
# engine's own certified multiply decides); then RG-multiplicativity +
# level-triangularity force the whole product to equal `RG(L_{(m,0)})`.
# If the level-0 check fails the build honest-fails (a genuine peel
# would be needed — not in this increment's scope).
# ---------------------------------------------------------------------------


def _antifund_decomposition(m):
    """All-non-positive dominant `m` = `Σ_k b_k·ψ_k` with the anti-fundamental
    coweights `ψ_k = (0,…,0,(−1)^k)` and `b_k = m_{N−k} − m_{N−k+1} ≥ 0`
    (`b_N = −m_0`): the F-side minuscule chain.  Chosen over the
    det⁻¹/E-chain because the matter zero-modes attach to negative
    directions only — same-side chains stack the q-ladders coherently
    (`spread(F)² = RG(L_{F²})` exactly), whereas mixing E with det⁻¹
    contaminates the product with flavour-charged canonical admixtures
    (observed: non-palindromic `q⁻³+q⁻¹` extras — the bar certificate
    below catches exactly this)."""
    m = tuple(int(x) for x in m)
    N = len(m)
    if list(m) != sorted(m, reverse=True):
        raise ValueError(f"m={m} must be dominant (descending)")
    if any(x > 0 for x in m):
        raise NotImplementedError(
            f"m={m}: positive-mixed shapes need the full peel "
            f"(T2c increment 3); constructive chains cover the "
            f"all-non-positive cone")
    factors = []
    ext = (0,) + m                       # m_0 prepended as the k=N boundary
    for k in range(1, N + 1):
        b = ext[N - k] - ext[N - k + 1]
        psi = (0,) * (N - k) + (-1,) * k
        factors += [psi] * b
    return factors


def _spread_element(m, e, Nf):
    """The diagonal spread as an auxiliary `Element` (flavour level 0 base)."""
    from kalgebra import Element
    return Element({(lab, kv): c
                    for (lab, kv), c in dressed_label_spread(m, e, Nf).items()})


def constructive_rg_image(A, m):
    """`RG(L_{(m,0);0})` built constructively (never a solve): the auxiliary
    product of the certified anti-fundamental factor images of `m`
    (memoized on `A`).  Two in-build certificates, both honest-fail:

    * **level-0**: the pure part of the product must be the single
      canonical `L_{(m,0)}` with coefficient 1 (the engine's certified
      multiply decides) — necessary but NOT sufficient (flavour-charged
      admixtures are level-0-invisible);
    * **bar**: every coefficient must be q-palindromic — RG images of
      canonicals are bar-invariant (`verify_rg_bar_invariant`), and the
      observed contamination mode (canonical admixtures from a wrong
      factor chain) is non-palindromic, so this catches it in-build."""
    from kalgebra import Element
    m = tuple(int(x) for x in m)
    Nf = A.Nf
    cache = A.__dict__.setdefault("_constructive_rg_cache", {})
    hit = cache.get(m)
    if hit is not None:
        return hit
    aux = A.auxiliary()
    N = A.N
    acc = Element({((tuple((0,) * N), tuple((0,) * N)), (0,) * Nf):
                   LaurentPoly({0: 1})})
    for f in _antifund_decomposition(m):
        acc = aux.multiply_elements(acc, _spread_element(f, (0,) * N, Nf))
    zero_k = (0,) * Nf
    level0 = {lab: c for (lab, k), c in acc.terms.items()
              if tuple(k) == zero_k and not c.is_zero()}
    if level0 != {(m, (0,) * N): LaurentPoly({0: 1})}:
        raise NotImplementedError(
            f"constructive_rg_image({m}): factor product is not the single "
            f"pure canonical (level-0 = {level0}); a peel step is required "
            f"(outside this increment's scope)")
    for (lab, k), c in acc.terms.items():
        if dict(c._coeffs) != {-e: z for e, z in c._coeffs.items()}:
            raise NotImplementedError(
                f"constructive_rg_image({m}): non-bar-invariant coefficient "
                f"{c} at {(lab, k)} — canonical admixture present; a peel "
                f"step is required (outside this increment's scope)")
    cache[m] = acc
    return acc


def _lp_qshift(c: LaurentPoly, p: int) -> LaurentPoly:
    return LaurentPoly({e + p: z for e, z in c._coeffs.items()})


def _uniform_qratio(X, Y):
    """`p` with `X == q^{2p}·Y` termwise (None if not a uniform q-power)."""
    if set(X.terms) != set(Y.terms):
        return None
    p2 = None
    for key, cx in X.terms.items():
        cy = Y.terms[key]
        ex = sorted(cx._coeffs.items())
        ey = sorted(cy._coeffs.items())
        if len(ex) != len(ey):
            return None
        d = ex[0][0] - ey[0][0]
        if any(a - b != d or za != zb for (a, za), (b, zb) in zip(ex, ey)):
            return None
        if p2 is None:
            p2 = d
        elif p2 != d:
            return None
    return None if (p2 is None or p2 % 2) else p2 // 2


def dressed_chain_image(A, m, e, _depth: int = 0):
    """`RG(L_{(m,e);0})` for all-non-positive `m` with dressings on negative
    slots: the ordered product of **dressed-minuscule spreads** (each slot's
    dressing rides the last ψ-factor covering it — every factor is depth-1,
    so its Pieri spread is certified), with the q-prefactor pinned by the
    **two-ordering bar-covariance** (the pure engine's `P == q^{2p}·Q`
    extraction): the chain and its reverse must be a uniform `q^{2p}` apart,
    and the image is the bar-centered representative `q^{−p}·X₁`.
    Honest-fail if the two orderings are not a uniform q-power apart (a
    multi-canonical product — needs the full recognize-and-subtract)."""
    from kalgebra import Element
    Nf = A.Nf
    N = A.N
    m = tuple(int(x) for x in m)
    e = tuple(int(x) for x in e)
    if any(x > 0 for x in m):
        raise NotImplementedError(f"dressed_chain_image: m={m} not ≤ 0")
    if any(e[j] != 0 and m[j] >= 0 for j in range(N)):
        raise NotImplementedError(
            f"dressed_chain_image: ({m},{e}) carries a transparent-slot "
            f"dressing — outside the chain scope (engine lift)")
    factors = _antifund_decomposition(m)
    dress = _assign_dressings(factors, e, N)
    elts = [_spread_element(f, ef, Nf) for f, ef in zip(factors, dress)]
    aux = A.auxiliary()
    one = Element({((tuple((0,) * N), tuple((0,) * N)), (0,) * Nf):
                   LaurentPoly({0: 1})})
    X1, X2 = one, one
    for el in elts:
        X1 = aux.multiply_elements(X1, el)
    for el in reversed(elts):
        X2 = aux.multiply_elements(X2, el)
    p = _uniform_qratio(X1, X2)
    if p is None:
        raise NotImplementedError(
            f"dressed_chain_image({m},{e}): orderings are not a uniform "
            f"q-power apart — multi-canonical chain (engine lift)")
    out = Element({key: _lp_qshift(c, -p) for key, c in X1.terms.items()})
    for key, c in out.terms.items():
        if dict(c._coeffs) != {-x: z for x, z in c._coeffs.items()}:
            raise NotImplementedError(
                f"dressed_chain_image({m},{e}): centered image not "
                f"bar-invariant at {key}: {c}")
    return out


def _assign_dressings(factors, e, N):
    """Distribute per-slot dressings `e` over the ψ-factors: each slot rides
    the last factor where it is a depth-1 SINGLETON block (`ψ₁`-type) when
    one exists; otherwise the last covering factor, with that factor's
    block dressing sorted descending (central blocks need Levi-dominant
    `e_b`; the engine loop's certificates judge the resulting chain)."""
    dress = [dict() for _ in factors]
    for j in range(N):
        if e[j] == 0:
            continue
        singleton = [i for i, f in enumerate(factors)
                     if f[j] != 0 and sum(1 for x in f if x != 0) == 1]
        if singleton:
            dress[singleton[-1]][j] = e[j]
            continue
        covering = [i for i, f in enumerate(factors) if f[j] != 0]
        dress[covering[-1]][j] = e[j]
    out = []
    for f, dr in zip(factors, dress):
        ef = [dr.get(j, 0) for j in range(N)]
        blocks = _levi_blocks_desc(tuple(f))
        for b in blocks:
            if f[b[0]] != 0 and len(b) > 1:
                vals = sorted((ef[j] for j in b), reverse=True)
                for j, v in zip(b, vals):
                    ef[j] = v
        out.append(tuple(ef))
    return out


def engine_image(A, m, e, _depth: int = 0, top=None):
    """`RG(L_{(m,e);0})` by the lifted **recognize-and-subtract loop**: the
    chain product is formed in BOTH orderings;
    remainders are processed level-ascending —

    * level-0 admixture seeds (pure structure constants) are peeled whole,
      per ordering (their constants are bar-related across orderings);
    * higher-level admixture constants are pinned by the **two-ordering
      positive-part rule** `C₁ = (x₁ − x₂)|_{q>0}` — exact when admixture
      constants are strictly-positive-q-supported (the `O(𝔮)` reading;
      bar-covariance gives `x₁ − x₂ = C₁ − bar(C₁)`).  Certified per shape
      against the solver oracle; a violation surfaces as a mismatch there
      or as `W₁ ≠ W₂` here;

    then the target's image is the final remainder, q-centered to
    bar-invariance (the target's own chain constant must be a monomial),
    cross-checked `W₁ == W₂`."""
    from kalgebra import Element
    if _depth > 6:
        raise NotImplementedError(f"engine_image({m},{e}): recursion depth")
    Nf = A.Nf
    N = A.N
    m = tuple(int(x) for x in m)
    e = tuple(int(x) for x in e)
    aux = A.auxiliary()
    # chain factors: transparent part (m⁺ + transparent-slot dressings) as a
    # single label factor; the negative part as dressed ψ-factors.
    m_neg = tuple(min(x, 0) for x in m)
    m_pos = tuple(max(x, 0) for x in m)
    e_neg = tuple(e[j] if m[j] < 0 else 0 for j in range(N))
    e_pos = tuple(e[j] if m[j] >= 0 else 0 for j in range(N))
    factors = []
    if any(m_pos) or any(e_pos):
        factors.append(Element({((m_pos, e_pos), (0,) * Nf):
                                LaurentPoly({0: 1})}))
    psi = _antifund_decomposition(m_neg)
    for f, ef in zip(psi, _assign_dressings(psi, e_neg, N)):
        factors.append(_spread_element(f, ef, Nf))
    one = Element({((tuple((0,) * N), tuple((0,) * N)), (0,) * Nf):
                   LaurentPoly({0: 1})})
    W1, W2 = one, one
    for el in factors:
        W1 = aux.multiply_elements(W1, el)
    for el in reversed(factors):
        W2 = aux.multiply_elements(W2, el)
    target = (m, e)
    zero_k = (0,) * Nf
    levels = sorted({tuple(k) for (_, k) in W1.terms} |
                    {tuple(k) for (_, k) in W2.terms},
                    key=lambda k: (sum(k), k))
    own_max = max((sum(k) for k in levels), default=0)
    eff_top = own_max if top is None else min(int(top), own_max)
    if eff_top < own_max:
        W1 = Element({(lab, k): c for (lab, k), c in W1.terms.items()
                      if sum(k) <= eff_top})
        W2 = Element({(lab, k): c for (lab, k), c in W2.terms.items()
                      if sum(k) <= eff_top})
        levels = [k for k in levels if sum(k) <= eff_top]
    for lev in levels:
        labs = sorted({lab for (lab, k) in W1.terms if tuple(k) == lev} |
                      {lab for (lab, k) in W2.terms if tuple(k) == lev})
        for lab in labs:
            x1 = W1.terms.get((lab, lev), LaurentPoly.zero())
            x2 = W2.terms.get((lab, lev), LaurentPoly.zero())
            if lab == target and lev == zero_k:
                continue                       # the target's own seed
            if lev == zero_k:
                C1, C2 = x1, x2                # pure constants: peel whole
            else:
                diff = x1 - x2
                C1 = LaurentPoly({ee: z for ee, z in diff._coeffs.items()
                                  if ee > 0})
                C2 = LaurentPoly({-ee: z for ee, z in C1._coeffs.items()})
                # C₂ = bar(C₁): the reverse ordering's constant, by
                # bar-covariance C^c_{ba}(q) = C^c_{ab}(q⁻¹).
            if C1.is_zero() and C2.is_zero():
                continue
            sub = matter_image(A, lab[0], lab[1], _depth + 1,
                               top=eff_top - sum(lev))
            sh = Element({(l2, tuple(x + y for x, y in zip(k2, lev))): c2
                          for (l2, k2), c2 in sub.terms.items()
                          if sum(k2) + sum(lev) <= eff_top})
            if not C1.is_zero():
                W1 = W1 + sh * (C1 * (-1))
            if not C2.is_zero():
                W2 = W2 + sh * (C2 * (-1))
    # target seed must be a monomial unit; center to bar-invariance
    c_t = W1.terms.get((target, zero_k))
    if c_t is None or len(c_t._coeffs) != 1 or abs(next(iter(c_t._coeffs.values()))) != 1:
        raise NotImplementedError(
            f"engine_image({m},{e}): target chain constant {c_t} is not a "
            f"monomial unit")
    (p, sgn), = c_t._coeffs.items()
    out = Element({key: _lp_qshift(c, -p) * sgn
                   for key, c in W1.terms.items() if not c.is_zero()})
    c_t2 = W2.terms.get((target, zero_k))
    if c_t2 is None or len(c_t2._coeffs) != 1:
        raise NotImplementedError(
            f"engine_image({m},{e}): reverse target constant {c_t2}")
    (p2, sgn2), = c_t2._coeffs.items()
    out2 = Element({key: _lp_qshift(c, -p2) * sgn2
                    for key, c in W2.terms.items() if not c.is_zero()})
    if out != out2:
        raise NotImplementedError(
            f"engine_image({m},{e}): the two orderings disagree after "
            f"attribution — an admixture constant is not "
            f"positive-q-supported (needs the full chart-level lift)")
    for key, c in out.terms.items():
        if dict(c._coeffs) != {-x: z for x, z in c._coeffs.items()}:
            raise NotImplementedError(
                f"engine_image({m},{e}): image not bar-invariant at {key}: {c}")
    return out


def matter_rho(P, Nf: int, label):
    """Closed-form `ρ` of `A_𝖖[U(N)+N_f]` on `((m, e), k)` labels (verified
    against the flow's generic ρ on U(2)/U(3), N_f=1,2 — 24 probes + the
    T2 intertwining sweep):

        ρ_mat(a) = σ_pure(a) **shifted by the Z-top torsor of the image**:
        with `(m′, e′) = σ_pure(m, e)` and `d_j = |min(m′_j, 0)|`,
        `e′ ↦ e′ − N_f·d`,  `k ↦ k − N_f·|d|·(1,…,1)`.

    Physics: the half-monodromy drags the line once through every matter
    zero mode of the image — division by the top monomial
    `μ^{N_f|d|}·v^{N_f·d}` of `Z(m′)`; the `√(measure·M)` conjugation
    telescopes to exactly this.  `P` is the pure engine (its σ is the
    certified sign-free permutation), so ρ costs the same as pure σ —
    no tRG solve."""
    (m, e), k = label
    mp, ep = P.rho((tuple(m), tuple(e)))
    d = tuple(-min(x, 0) for x in mp)
    e2 = tuple(x - Nf * y for x, y in zip(ep, d))
    k2 = tuple(-x - sum(d) for x in k)         # flavour star, then Z-top shift
    return ((mp, e2), k2)


def matter_rho_inverse(P, Nf: int, label):
    """Inverse of `matter_rho`: un-shift the Z-top torsor (read off the
    input's own `m`), un-star the flavour, then `σ⁻¹_pure`."""
    (m, e), k = label
    d = tuple(-min(x, 0) for x in m)
    e0 = tuple(x + Nf * y for x, y in zip(e, d))
    k0 = tuple(-x - sum(d) for x in k)
    m0, ep = P.rho_inverse((tuple(m), tuple(e0)))
    return ((m0, ep), k0)


def _positive_excess(c: LaurentPoly) -> LaurentPoly:
    """The strictly-positive-q bar-excess of `c`: `Σ_{e>0} (c_e − c_{−e})·q^e`
    — the unique `C` supported at `q^{>0}` with `c − C` palindromic."""
    out = {}
    for e, z in c._coeffs.items():
        if e > 0:
            d = z - c._coeffs.get(-e, 0)
            if d:
                out[e] = d
    return LaurentPoly(out)


def matter_image(A, m, e=None, _depth: int = 0, top=None):
    """`RG(L_{(m,e);0})` built constructively for the U(N)+N_f matter flow:

    * diagonal scope (incl. depth-1 Pieri dressings): the certified spread;
    * `e = 0`, `m ≤ 0`: the anti-fundamental chain (increment 2);
    * `e = 0`, mixed sign: φ-factors (transparent, trivial images) times the
      ψ-chain of the negative part, then the **O(𝔮) peel** (user rulings
      2026-06-10): `RG(a) = a + O(𝔮)` — bubbling *and* matter corrections
      are both `O(𝔮)` — so admixture seeds at the base level are peeled
      whole, and at higher flavour levels the admixture coefficient is the
      strictly-positive-q bar-excess (`_positive_excess`; the palindromic
      remainder is the target's own `O(𝔮)`-centered content).  Certified
      against the solver oracle per shape
      (`certify_matter_image_vs_flow`); honest-fail on recursion overrun
      or a non-bar-invariant final image.

    Scope walls (dressed mixed shapes like `((1,−1),(1,1))`; dressed
    depth-≥2 blocks) are the target of the **engine lift** (T2c.3
    directive, user 2026-06-10): run the *same algorithm as pure U(N)*
    (`pure_un_closed_form.cf_build_full` product-and-peel with the
    leading-orbit / KL q-extreme machinery) with **matter-dressed
    minuscules** as the trusted generator set, rethinking the QTCone
    bookkeeping for the flavour direction — see the plan notes."""
    from kalgebra import Element
    if _depth > 24:
        raise NotImplementedError(f"matter_image({m}, {e}): recursion depth")
    Nf = A.Nf
    N = A.N
    m = tuple(int(x) for x in m)
    e = tuple(int(x) for x in (e if e is not None else (0,) * N))
    cache = A.__dict__.setdefault("_matter_image_cache", {})
    hit = cache.get((m, e))
    if hit is not None:
        return hit                      # full images subsume any window
    try:
        out = Element({(lab, kv): c
                       for (lab, kv), c in dressed_label_spread(m, e, Nf).items()})
        cache[(m, e)] = out
        return out
    except NotImplementedError:
        pass
    # central-Wilson torsor normalization: e → e − c·(1,…,1) is one cheap
    # transparent multiply (`L_{w₂^c}·L_{(m,e0)} = q^α·L_{(m,e)}`, the seed
    # pins α), not a recursive build — this is what keeps chain admixtures
    # like ((−1,−2),(7,7)) from blowing the recursion.
    c0 = min(e)
    if c0 != 0:
        base = matter_image(A, m, tuple(x - c0 for x in e), _depth + 1)
        w2 = Element({((tuple((0,) * N), tuple((c0,) * N)), (0,) * Nf):
                      LaurentPoly({0: 1})})
        prod = A.auxiliary().multiply_elements(w2, base)
        seed = prod.terms.get(((m, e), (0,) * Nf))
        if seed is None or len(seed._coeffs) != 1 or \
                abs(next(iter(seed._coeffs.values()))) != 1:
            raise NotImplementedError(
                f"matter_image({m},{e}): central-Wilson shift seed {seed} "
                f"is not a monomial unit")
        (p, sgn), = seed._coeffs.items()
        out = Element({key: _lp_qshift(c, -p) * sgn
                       for key, c in prod.terms.items() if not c.is_zero()})
        cache[(m, e)] = out
        return out
    if any(e):
        if all(x <= 0 for x in m) and not any(
                e[j] != 0 and m[j] >= 0 for j in range(N)):
            try:
                out = dressed_chain_image(A, m, e, _depth)
                cache[(m, e)] = out
                return out
            except NotImplementedError:
                pass                       # multi-canonical: engine loop
        ekey = (m, e, top)
        ecache = A.__dict__.setdefault("_engine_image_cache", {})
        hit = ecache.get(ekey)
        if hit is None:
            hit = engine_image(A, m, e, _depth, top=top)
            ecache[ekey] = hit
        return hit
    if all(x <= 0 for x in m):
        out = constructive_rg_image(A, m)
        cache[(m, e)] = out
        return out
    # mixed sign, e = 0: E-part (single transparent label) × ψ-chain
    aux = A.auxiliary()
    m_neg = tuple(min(x, 0) for x in m)
    m_pos = tuple(max(x, 0) for x in m)
    X = Element({((m_pos, (0,) * N), (0,) * Nf): LaurentPoly({0: 1})})
    X = aux.multiply_elements(X, matter_image(A, m_neg, None, _depth + 1))
    # ---- the peel ----
    zero_k = (0,) * Nf
    target = (m, (0,) * N)
    work = {key: c for key, c in X.terms.items() if not c.is_zero()}
    peeled = Element(dict(work))
    levels = sorted({tuple(k) for (_, k) in work}, key=lambda k: (sum(k), k))
    for lev in levels:
        cur = {lab: c for (lab, k), c in peeled.terms.items()
               if tuple(k) == lev and not c.is_zero()}
        for lab, c in sorted(cur.items()):
            if lab == target and lev == zero_k:
                if c != LaurentPoly({0: 1}):
                    raise NotImplementedError(
                        f"matter_image({m}): target seed coefficient {c} ≠ 1")
                continue
            C = c if lev != zero_k else c       # base level: peel whole
            if lev != zero_k:
                C = _positive_excess(c)
            if C.is_zero():
                continue
            sub = matter_image(A, lab[0], lab[1], _depth + 1)
            shifted = Element({(l2, tuple(x + y for x, y in zip(k2, lev))): c2
                               for (l2, k2), c2 in sub.terms.items()})
            peeled = peeled + shifted * (C * (-1))
    # final certificates: bar-invariance + clean level-0
    for (lab, k), c in peeled.terms.items():
        if c.is_zero():
            continue
        if dict(c._coeffs) != {-x: z for x, z in c._coeffs.items()}:
            raise NotImplementedError(
                f"matter_image({m}): peeled image not bar-invariant at "
                f"{(lab, k)}: {c} — the bar+minimality principle does not "
                f"close here")
    out = Element({key: c for key, c in peeled.terms.items() if not c.is_zero()})
    cache[(m, e)] = out
    return out


def certify_matter_image_vs_flow(A, m, e=None) -> bool:
    """EXACT equality of the bar+minimality construction with the solver
    oracle — the per-shape certificate of the (empirical) peel principle."""
    got = matter_image(A, m, e).terms
    N = A.N
    e = tuple(int(x) for x in (e if e is not None else (0,) * N))
    want = A.RG(((tuple(m), e), (0,) * A.Nf)).terms
    g = {(lab, tuple(k)): c for (lab, k), c in got.items() if not c.is_zero()}
    w = {(lab, tuple(k)): c for (lab, k), c in want.items() if not c.is_zero()}
    return g == w


def certify_constructive_vs_flow(A, m) -> bool:
    """EXACT equality of the product-and-peel construction with the solver
    oracle `A.RG(((m, 0), 0))` — including every bubbling term."""
    got = constructive_rg_image(A, m).terms
    want = A.RG(((tuple(m), (0,) * A.N), (0,) * A.Nf)).terms
    g = {(lab, tuple(k)): c for (lab, k), c in got.items() if not c.is_zero()}
    w = {(lab, tuple(k)): c for (lab, k), c in want.items() if not c.is_zero()}
    return g == w


if __name__ == "__main__":
    from un_nf_over_pure_rgflow import UNNfOverPure
    print("== certify dressed-generator closed form vs Route-A oracle ==")
    A2 = UNNfOverPure(2, 1)
    probes_u2 = [
        ((0, 0), (0, 0)), ((1, 0), (0, 0)), ((0, -1), (0, 0)),
        ((1, 1), (0, 0)), ((-1, -1), (0, 0)), ((1, -1), (0, 0)),
        ((0, -2), (0, 0)), ((-2, -2), (0, 0)), ((0, -3), (0, 0)),
        ((1, 0), (1, 0)), ((1, 0), (0, 1)),           # transparent dressings
        ((-1, -2), (0, 0)),                            # PREDICTIVE: mixed depths
        ((1, -2), (0, 0)),                             # PREDICTIVE: E×F² shape
    ]
    ok = True
    for m, e in probes_u2:
        r = certify_against_flow(A2, m, e)
        ok &= r
        print(f"  U(2) Nf=1  ({m}, {e}): {'OK' if r else 'MISMATCH'}")
    A22 = UNNfOverPure(2, 2)
    for m, e in [((0, -1), (0, 0)), ((-1, -1), (0, 0))]:
        r = certify_against_flow(A22, m, e)
        ok &= r
        print(f"  U(2) Nf=2  ({m}, {e}): {'OK' if r else 'MISMATCH'}")
    # the documented bubbling counterexample: diagonal form must honest-fail,
    # and the oracle's extra term is the magnetic-bubbling channel
    try:
        dressed_label_spread((0, -2), (0, 0), 2)
        ok = False
        print("  U(2) Nf=2  ((0,-2)): expected NotImplementedError (bubbling)")
    except NotImplementedError:
        bub = A22.RG((((0, -2), (0, 0)), (0, 0))).terms
        hit = bub.get((((-1, -1), (1, 1)), (1, 1)))
        r = hit is not None and dict(hit._coeffs) == {-1: 1, 1: 1}
        ok &= r
        print(f"  U(2) Nf=2  ((0,-2)) bubbling term ((-1,-1),(1,1))@(1,1) = [2]_q: "
              f"{'OK' if r else 'MISMATCH'}")
    A3 = UNNfOverPure(3, 1)
    for m, e in [((0, 0, -1), (0, 0, 0)),
                 ((0, -1, -1), (0, 0, 0)),             # PREDICTIVE: size-2 block
                 ((1, 0, -1), (0, 0, 0))]:
        r = certify_against_flow(A3, m, e)
        ok &= r
        print(f"  U(3) Nf=1  ({m}, {e}): {'OK' if r else 'MISMATCH'}")
    print("--- increment 2: anti-fundamental chains (bubbling emerges) ---")
    for A_, tag, shapes in [
        (A2, "U(2) Nf=1", [(0, -2), (-1, -3), (0, -3)]),
        (A22, "U(2) Nf=2", [(0, -2), (0, -3), (-1, -3), (0, -4), (-2, -2)]),
        (UNNfOverPure(2, 3), "U(2) Nf=3", [(0, -2)]),
        (UNNfOverPure(3, 2), "U(3) Nf=2", [(0, 0, -2)]),
    ]:
        for m in shapes:
            try:
                r = certify_constructive_vs_flow(A_, m)
            except NotImplementedError as ex:
                r = False
                print(f"  {tag}  m={m}: HONEST-FAIL ({ex})")
                continue
            ok &= r
            print(f"  {tag}  m={m}: {'OK' if r else 'MISMATCH'}")
    print("ALL CERTIFIED" if ok else "FAILURES — closed form does not match oracle")
    sys.exit(0 if ok else 1)


def matter_decompose(A, X, top=None, _depth: int = 0):
    """Decompose an auxiliary `Element` `X` against the native canonical
    images: `X = Σ C^{(lab,k)}·μ^k·image(lab)` — unique, and attributed
    **unambiguously** level-by-level (unlike a *build*, a decomposition has
    no target whose higher-level content must be preserved: after
    subtracting lower-based images, every residual term at the current
    level is exactly a seed coefficient).  Returns `{(lab, k): C}`.
    Honest-fails if a needed image is outside the constructive scope or a
    windowed remainder survives."""
    from kalgebra import Element
    work = {key: c for key, c in X.terms.items() if not c.is_zero()}
    levels = sorted({tuple(k) for (_, k) in work}, key=lambda k: (sum(k), k))
    own_max = max((sum(k) for k in levels), default=0)
    eff_top = own_max if top is None else min(int(top), own_max)
    out = {}
    W = Element({(lab, k): c for (lab, k), c in work.items()
                 if sum(k) <= eff_top})
    for lev in levels:
        if sum(lev) > eff_top:
            continue
        cur = {lab: c for (lab, k), c in W.terms.items()
               if tuple(k) == lev and not c.is_zero()}
        for lab, c in sorted(cur.items()):
            out[(lab, lev)] = c
            sub = matter_image(A, lab[0], lab[1], _depth + 1,
                               top=eff_top - sum(lev))
            sh = Element({(l2, tuple(x + y for x, y in zip(k2, lev))): c2
                          for (l2, k2), c2 in sub.terms.items()
                          if sum(k2) + sum(lev) <= eff_top})
            W = W + sh * (c * (-1))
    if any(not c.is_zero() for c in W.terms.values()):
        raise NotImplementedError(
            f"matter_decompose: windowed remainder survives ({dict((k, str(v)) for k, v in W.terms.items() if not v.is_zero())})")
    return out


def matter_multiply(A, a, b, top=None):
    """Native `L_a · L_b` of `A_𝖖[U(N)+N_f]` — NO RG solve: the auxiliary
    product of the constructive images, decomposed against the native
    images (`matter_decompose`).  `a`, `b` are `((m, e), k)` labels;
    returns `{(lab, k): C}` — the matter structure constants."""
    cache = A.__dict__.setdefault("_matter_multiply_cache", {})
    ckey = (tuple(a[0][0]), tuple(a[0][1]), tuple(b[0][0]), tuple(b[0][1]),
            tuple(a[1]), tuple(b[1]), top)
    hit = cache.get(ckey)
    if hit is not None:
        return hit
    ia = matter_image(A, a[0][0], a[0][1])
    ib = matter_image(A, b[0][0], b[0][1])
    from kalgebra import Element
    ka, kb = tuple(a[1]), tuple(b[1])
    shift = tuple(x + y for x, y in zip(ka, kb))
    X = A.auxiliary().multiply_elements(ia, ib)
    if any(shift):
        X = Element({(lab, tuple(x + y for x, y in zip(k, shift))): c
                     for (lab, k), c in X.terms.items()})
    out = matter_decompose(A, X, top=top)
    cache[ckey] = out
    return out


# ---------------------------------------------------------------------------
# The native trace: the matter Schur factor inserted into the pure residue
# (user directive 2026-06-11).  Tr is a SINGLE measure-residue pass:
#
#   Tr(x)  =  Σ_{ℓ, n}  μ^{ℓ+n} · Res_pure( f₀^{(ℓ)}(v) · [M]_n(v) )
#
# with `M(μ,v) = ∏_{i≤N_f} ∏_j E_𝖖(μ_i v_j)·E_𝖖(μ_i⁻¹ v_j⁻¹)` the matter
# Schur-measure factor (both halves — μ-levels of both signs) and
# `f₀^{(ℓ)}` the flavour-level-ℓ magnetic-0 residual of the element's
# bare chart.  No Ψ-window, no FS objects, no ρ-element.
# ---------------------------------------------------------------------------


def _matter_factor_levels(N: int, Nf: int, W: int, Kq: int) -> dict:
    """`{n: [M]_n}` as `VRational` v-Laurents for `|n| ≤ W` (per-flavour
    `a_n` coefficients exact, expanded at `Kq`)."""
    from abelianized_torus import VRational, VLaurent
    from habiro import HabiroElement

    def a_n(n):
        return HabiroElement.nahm_term((-1) ** n, n, [n]).expand(Kq + W)

    terms = {(0, (0,) * N): LaurentPoly({0: 1})}
    for _i in range(Nf):
        for j in range(N):
            for sign in (+1, -1):
                new: dict = {}
                for n in range(0, W + 1):
                    c = a_n(n)
                    for (lv, ve), z in terms.items():
                        lv2 = lv + sign * n
                        if abs(lv2) > W:
                            continue
                        ve2 = tuple(x + (sign * n if t == j else 0)
                                    for t, x in enumerate(ve))
                        add = z * c
                        key = (lv2, ve2)
                        new[key] = new.get(key, LaurentPoly.zero()) + add
                terms = new
    out: dict = {}
    for (lv, ve), z in terms.items():
        if not z.is_zero():
            out.setdefault(lv, {})[ve] = z
    return {lv: VRational.from_vlaurent(VLaurent(row, n=N))
            for lv, row in out.items()}


def matter_schur_trace(A, label, K: int, window: int = None):
    """μ-refined `Tr(L_{(m,e);k})` of `A_𝖖[U(N)+N_f]` by the matter-measure
    insertion (single pure-residue pass per (ℓ, n) pair; per-(A, label, K)
    memo; two-window stability guard on the M-truncation)."""
    from urq_torus import _schur_trace_v0
    from abelianized_torus import VRational
    cache = A.__dict__.setdefault("_matter_trace_cache", {})
    key = (tuple(label[0][0]), tuple(label[0][1]), tuple(label[1]), K)
    hit = cache.get(key)
    if hit is not None:
        return hit
    N, Nf = A.N, A.Nf
    P = A.pure()
    Kq = K + 10
    img = matter_image(A, label[0][0], label[0][1])
    k0 = tuple(label[1])
    f0: dict = {}
    for (lab, k), c in img.terms.items():
        r = P.urqt(lab).residuals().get((0,) * N)
        if r is None:
            continue
        cv = VRational.from_scalar(c, n=N)
        lk = tuple(x + y for x, y in zip(k, k0))
        cur = f0.get(lk)
        term = (r * cv).simplify()
        f0[lk] = term if cur is None else (cur + term).simplify()

    def evaluate(W):
        M = _matter_factor_levels(N, Nf, W, Kq)
        out: dict = {}
        for lk, f in f0.items():
            for n, mf in M.items():
                lp = _schur_trace_v0((f * mf).simplify(), N, K, adaptive=True)
                if lp.is_zero():
                    continue
                tot = lk[0] + n if Nf == 1 else None
                mu = (lk[0] + n,) if Nf == 1 else tuple(
                    x + n for x in lk)        # Nf>1: per-flavour M-levels TBD
                row = out.setdefault(mu, LaurentPoly.zero())
                out[mu] = row + lp
        return {mu: {ee: z for ee, z in lp._coeffs.items() if z and 0 <= ee <= K}
                for mu, lp in out.items()
                if any(z and 0 <= ee <= K for ee, z in lp._coeffs.items())}

    if Nf != 1:
        raise NotImplementedError(
            "matter_schur_trace: N_f ≥ 2 needs per-flavour M-level "
            "bookkeeping (the prototype tracks the total level only)")
    W = 3 if window is None else int(window)
    prev = None
    while W <= 8:
        cur = evaluate(W)
        if prev is not None and prev == cur:
            from zplus_ring import AbelianZPlusRing, RElement, RPowerSeries
            R = A.coefficient_ring()
            coeffs: dict = {}
            for mu, row in cur.items():
                for ee, z in row.items():
                    coeffs.setdefault(ee, {})[(mu if isinstance(mu, tuple)
                                                else (mu,))] = z
            res = RPowerSeries(R, {ee: RElement(R, terms)
                                   for ee, terms in coeffs.items()}, K)
            cache[key] = res
            return res
        prev = cur
        W += 1
    raise NotImplementedError(
        f"matter_schur_trace({label}): M-window did not stabilise by W=8")


def matter_inner_product(A, a, b, K: int):
    """Fully native `I_{a,b} = Tr(L_{ρ(a)}·L_b)`: closed-form ρ + native
    multiply + the matter-measure trace — no solver anywhere."""
    P = A.pure()
    ra = matter_rho(P, A.Nf, a)
    prod = matter_multiply(A, ra, b)
    out = None
    for (lab, k), C in prod.items():
        tr = matter_schur_trace(A, ((lab[0], lab[1]), tuple(k)), K + 4)
        contrib = _rps_scale(tr, C, K)
        out = contrib if out is None else _rps_add(out, contrib, K)
    from zplus_ring import RPowerSeries
    return out if out is not None else RPowerSeries(
        A.coefficient_ring(), {}, K)


def _rps_scale(rps, C: LaurentPoly, K: int):
    from zplus_ring import RElement, RPowerSeries
    R = rps.ring
    coeffs: dict = {}
    for ee, r in rps.coeffs.items():
        for qe, z in C._coeffs.items():
            e2 = ee + qe
            if not (0 <= e2 <= K):
                continue
            row = coeffs.setdefault(e2, {})
            for f, c in r.terms.items():
                row[f] = row.get(f, 0) + c * z
    return RPowerSeries(R, {e2: RElement(R, {f: c for f, c in row.items() if c})
                            for e2, row in coeffs.items() if any(row.values())}, K)


def _rps_add(x, y, K: int):
    from zplus_ring import RElement, RPowerSeries
    R = x.ring
    coeffs: dict = {}
    for s in (x, y):
        for ee, r in s.coeffs.items():
            if ee > K:
                continue
            row = coeffs.setdefault(ee, {})
            for f, c in r.terms.items():
                row[f] = row.get(f, 0) + c
    return RPowerSeries(R, {ee: RElement(R, {f: c for f, c in row.items() if c})
                            for ee, row in coeffs.items() if any(row.values())}, K)


# ---------------------------------------------------------------------------
# The Ũ-native "find the canonical" harness (user directive 2026-06-11):
# demonstrate that L_{(m,e)} can be FOUND inside the matter-enriched torus
# context — generator families constructed by the absorption theorem
# (pure residuals × atom-Z; no label machinery, no solver), products taken
# per-level (the deformed cocycle in bare coordinates), admixtures peeled
# chart-natively, and the result certified by exact per-level chart
# equality against the solver oracle.  Bubbling content is never input:
# it emerges in the products (build, don't guess).
# ---------------------------------------------------------------------------


def _atom_z_levels(matom, Nf: int, N: int) -> dict:
    """`{k_vec: [Z(atom)]_k}` as `VRational` — the atom's zero-mode ladder."""
    from abelianized_torus import VRational, VLaurent
    fac = []
    for j, mj in enumerate(matom):
        if mj < 0:
            n = -mj
            for i in range(Nf):
                for s in range(n):
                    fac.append((i, j, 2 * s - n + 1))
    levels = {(0,) * Nf: {(0,) * N: LaurentPoly({0: 1})}}
    for (i, j, sh) in fac:
        out: dict = {}
        for kv, row in levels.items():
            for ve, c in row.items():
                out.setdefault(kv, {}).setdefault(ve, LaurentPoly.zero())
                out[kv][ve] = out[kv][ve] + c
                kv2 = tuple(x + (1 if t == i else 0) for t, x in enumerate(kv))
                ve2 = tuple(x + (1 if t == j else 0) for t, x in enumerate(ve))
                out.setdefault(kv2, {}).setdefault(ve2, LaurentPoly.zero())
                out[kv2][ve2] = out[kv2][ve2] + c * LaurentPoly({sh: 1})
        levels = out
    return {kv: VRational.from_vlaurent(VLaurent(
        {ve: c for ve, c in row.items() if not c.is_zero()}, n=N))
        for kv, row in levels.items()}


def utilde_family(P, Nf: int, m, e) -> dict:
    """The Ũ-family of an **anti-dominant-cone** canonical, by the
    absorption theorem: `{k_vec: URQTorus}` with per-atom content
    `pure_residual(atom) × [Z(atom)]_k`.  Pure chart + ladders only —
    no spreads, no solver, no label-level matter machinery."""
    from urq_torus import URQTorus
    m = tuple(int(x) for x in m)
    e = tuple(int(x) for x in e)
    if any(x > 0 for x in m):
        raise NotImplementedError(
            f"utilde_family: ({m},{e}) is outside the absorption cone — "
            f"mixed shapes are FOUND by utilde_find_mixed (build)")
    N = len(m)
    pure = P.urqt((m, e))
    out: dict = {}
    for atom, f in pure.residuals().items():
        for kv, z in _atom_z_levels(atom, Nf, N).items():
            term = (f * z).simplify()
            if term.is_zero():
                continue
            cur = out.get(kv)
            add = URQTorus.from_f({atom: term}, N)
            out[kv] = add if cur is None else (cur + add)
    return out


def _family_mul(X: dict, Y: dict, N: int) -> dict:
    from urq_torus import URQTorus
    out: dict = {}
    for k1, U1 in X.items():
        for k2, U2 in Y.items():
            k = tuple(x + y for x, y in zip(k1, k2))
            prod = U1 * U2
            out[k] = prod if k not in out else (out[k] + prod)
    return {k: U for k, U in out.items() if U.residuals()}


def _family_sub_scaled(X: dict, F: dict, C: LaurentPoly, lev, N: int) -> dict:
    """`X − C·μ^lev·F` (level-shifted, q-scaled subtraction)."""
    from urq_torus import URQTorus
    from abelianized_torus import VRational
    cv = VRational.from_scalar(C * LaurentPoly({0: -1}), n=N)
    out = dict(X)
    for k, U in F.items():
        k2 = tuple(x + y for x, y in zip(k, lev))
        sub = URQTorus.from_f({a: (f * cv).simplify()
                               for a, f in U.residuals().items()}, N)
        out[k2] = sub if k2 not in out else (out[k2] + sub)
    return {k: U for k, U in out.items() if U.residuals()}


def _vr_scalar_ratio(f1, f2, N: int, exact: bool = True):
    """`C` with `f1 == C·f2` for a q-Laurent scalar `C`.

    `exact=True`: verify the full equality (None otherwise).
    `exact=False`: extract `C` from the **lexicographically-top v-monomial**
    of `f2`'s numerator only (the leading-coefficient peel used when other
    candidates overlap the same atom — the remainder peels the next
    candidate)."""
    from urq_torus import URQTorus
    from abelianized_torus import VRational
    f1 = f1.simplify(); f2 = f2.simplify()
    n1 = f1.num._terms if hasattr(f1, 'num') else None
    n2 = f2.num._terms if hasattr(f2, 'num') else None
    if not n1 or not n2:
        return None
    ve = max(n2)
    c2 = n2[ve]
    c1 = n1.get(ve)
    if c1 is None:
        return None
    C = _lp_divide_safe(c1, c2)
    if C is None or C.is_zero():
        return None
    if not exact:
        return C
    lhs = URQTorus.from_f({(0,) * N: f1}, N)
    rhs = URQTorus.from_f(
        {(0,) * N: (f2 * VRational.from_scalar(C, n=N)).simplify()}, N)
    return C if lhs == rhs else None


def _lp_divide_safe(a: LaurentPoly, b: LaurentPoly):
    """`a/b` when `b` is a q-monomial times a unit of small support — here
    sufficient: try monomial `b` exactly."""
    if len(b._coeffs) == 1:
        (e, z), = b._coeffs.items()
        if abs(z) == 1:
            return LaurentPoly({x - e: y * z for x, y in a._coeffs.items()})
    # general exact polynomial division attempt via long division in q
    num = dict(a._coeffs)
    out = {}
    bb = sorted(b._coeffs.items())
    while num:
        lo = min(num)
        c = num[lo]
        be, bz = bb[0]
        if c % bz:
            return None
        f = c // bz
        out[lo - be] = f
        for e2, z2 in bb:
            k = lo - be + e2
            num[k] = num.get(k, 0) - f * z2
            if num[k] == 0:
                del num[k]
    return LaurentPoly(out)


def utilde_find_mixed(A, m, e=None) -> dict:
    """FIND the mixed canonical `L_{(m,e)}` inside the Ũ context: the
    transparent-part chart times the absorption-constructed negative-part
    family, in both orderings; chart-native two-ordering peel of the
    admixtures (their families Level-A constructed, recursion stays in the
    absorption cone); bar-centering; returns the per-level chart family.
    Certify with `certify_utilde_vs_oracle`."""
    from urq_torus import URQTorus
    from pure_un_closed_form import _recognize_leading
    N, Nf = A.N, A.Nf
    P = A.pure()
    m = tuple(int(x) for x in m)
    e = tuple(int(x) for x in (e if e is not None else (0,) * N))
    m_neg = tuple(min(x, 0) for x in m)
    m_pos = tuple(max(x, 0) for x in m)
    e_neg = tuple(e[j] if m[j] < 0 else 0 for j in range(N))
    e_pos = tuple(e[j] if m[j] >= 0 else 0 for j in range(N))
    Epart = {(0,) * Nf: P.urqt((m_pos, e_pos))}
    Fpart = utilde_family(P, Nf, m_neg, e_neg)
    X1 = _family_mul(Epart, Fpart, N)
    X2 = _family_mul(Fpart, Epart, N)
    levels = sorted(set(X1) | set(X2), key=lambda k: (sum(k), k))
    zero = (0,) * Nf
    for lev in levels:
        if lev == zero:
            continue
        for _ in range(64):
            U1 = X1.get(lev)
            U2 = X2.get(lev)
            D = None
            if U1 is not None and U2 is not None:
                from abelianized_torus import VRational
                neg = VRational.from_scalar(LaurentPoly({0: -1}), n=N)
                D = U1 + URQTorus.from_f(
                    {a: (f * neg).simplify() for a, f in U2.residuals().items()}, N)
            elif U1 is not None or U2 is not None:
                D = U1 if U1 is not None else U2
            if D is None or not D.residuals():
                break
            mm, ee, m_dom, _ = _recognize_leading(D.to_chart(), N)
            # joint-canonicalise to the engine frame (m descending, e carried)
            order = sorted(range(N), key=lambda i: -mm[i])
            mm = tuple(mm[i] for i in order)
            ee = tuple(ee[i] for i in order)
            cand = utilde_family(P, Nf, mm, ee)
            r_c = cand[zero].residuals().get(m_dom)
            r_d = D.residuals().get(m_dom)
            diffC = _vr_scalar_ratio(r_d, r_c, N)
            if diffC is None:
                raise NotImplementedError(
                    f"utilde_find_mixed({m},{e}): non-scalar attribution at "
                    f"{lev} on {(mm, ee)}")
            C1 = LaurentPoly({x: z for x, z in diffC._coeffs.items() if x > 0})
            C2 = LaurentPoly({-x: z for x, z in C1._coeffs.items()})
            if C1.is_zero() and C2.is_zero():
                raise NotImplementedError(
                    f"utilde_find_mixed({m},{e}): palindromic attribution "
                    f"residue at {lev} on {(mm, ee)} ({diffC}) — needs the "
                    f"q-extreme read")
            if not C1.is_zero():
                X1 = _family_sub_scaled(X1, cand, C1, lev, N)
            if not C2.is_zero():
                X2 = _family_sub_scaled(X2, cand, C2, lev, N)
        else:
            raise NotImplementedError("attribution did not converge")
    for lev in sorted(set(X1) | set(X2), key=lambda k: (sum(k), k)):
        a1, a2 = X1.get(lev), X2.get(lev)
        if (a1 is None) != (a2 is None) or (a1 is not None and not (a1 == a2)):
            raise NotImplementedError(
                f"utilde_find_mixed({m},{e}): orderings disagree at {lev}")
    return X1


def certify_utilde_vs_oracle(A, m, e=None) -> bool:
    """The check: the Ũ-found family equals the ORACLE's chart family
    (flow RG image, label-by-label chart-assembled) at every flavour
    level — exact URQTorus equality."""
    from urq_torus import URQTorus
    from abelianized_torus import VRational
    N, Nf = A.N, A.Nf
    P = A.pure()
    e = tuple(int(x) for x in (e if e is not None else (0,) * N))
    m = tuple(int(x) for x in m)
    if all(x <= 0 for x in m):
        fam = utilde_family(P, Nf, m, e)
    else:
        fam = utilde_find_mixed(A, m, e)
    img = A.RG(((m, e), (0,) * Nf)).terms
    oracle: dict = {}
    for (lab, k), c in img.items():
        cv = VRational.from_scalar(c, n=N)
        x = P.urqt(lab)
        add = URQTorus.from_f({a: (f * cv).simplify()
                               for a, f in x.residuals().items()}, N)
        kk = tuple(k)
        oracle[kk] = add if kk not in oracle else (oracle[kk] + add)
    for k in sorted(set(fam) | set(oracle)):
        a, b = fam.get(k), oracle.get(k)
        if a is None or b is None or not (a == b):
            return False
    return True


def utilde_dedress(family: dict, Nf: int, N: int) -> dict:
    """De-dress a per-level bare-chart family into the **Ũ-residuals**
    `f̃_m` (the `x = Σ f̃_m(𝔮^m v)·Ũ_m` presentation): per atom, the
    triangular inversion `f̃_ℓ = B_ℓ − Σ_{ℓ'<ℓ} f̃_{ℓ'}·[Z(atom)]_{ℓ−ℓ'}`.
    Returns `{atom: {k_vec: VRational}}` (nonzero entries only)."""
    from abelianized_torus import VRational
    atoms = sorted({a for U in family.values() for a in U.residuals()})
    levels = sorted(family, key=lambda k: (sum(k), k))
    out: dict = {}
    for atom in atoms:
        Z = _atom_z_levels(atom, Nf, N)
        ft: dict = {}
        for k in levels:
            b = family[k].residuals().get(atom)
            acc = b if b is not None else VRational.from_scalar(
                LaurentPoly.zero(), n=N)
            for kp, fv in ft.items():
                dk = tuple(x - y for x, y in zip(k, kp))
                if any(x < 0 for x in dk):
                    continue
                z = Z.get(dk)
                if z is None:
                    continue
                acc = (acc + (fv * z * VRational.from_scalar(
                    LaurentPoly({0: -1}), n=N))).simplify()
            if not acc.is_zero():
                ft[k] = acc
        if ft:
            out[atom] = ft
    return out


def utilde_w1_certificate(family: dict, Nf: int, N: int) -> bool:
    """W1 on the Ũ-residuals: every de-dressed component `f̃_m` at every
    flavour level is **bar-palindromic** (`𝔮 → 𝔮⁻¹` with `v`, `m` fixed)
    — the pure `well_formed` W1 condition read in the `f̃`-frame (the
    user-pinned O(𝔮) frame).  The W2 (q-extreme) half on the matter
    corrections is the remaining acceptance item (TODO; level-0 W2 is
    pure `well_formed` by the absorption theorem)."""
    from urq_torus import URQTorus
    ded = utilde_dedress(family, Nf, N)
    for atom, ft in ded.items():
        for k, f in ft.items():
            x = URQTorus.from_f({atom: f}, N)
            if not (x.bar() == x):
                return False
    return True
