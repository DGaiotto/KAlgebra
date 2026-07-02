"""Phase 2e: clean recursive H·H multiplication via w_1 walk.

Algorithm (much cleaner than the σ-difference variant):

For fixed b and decreasing a from b downward, compute H_a · H_b
recursively using the W_1 associativity at anchor (a+1, b):

   (w_1 · H_{a+1}) · H_b  =  w_1 · (H_{a+1} · H_b)

LHS:    q^{-1}·(H_a · H_b) + q·(H_{a+2}·H_b) + [a+1 odd: H_b]
RHS:    w_1 acting on each canonical seed of H_{a+1}·H_b

Solving for H_a·H_b:
   H_a · H_b  =  q · w_1·(H_{a+1}·H_b)
                 − q^2 · (H_{a+2}·H_b)
                 − q · [a+1 odd: H_b]

Inputs: H_{a+1}·H_b (gap k-1, known by induction), H_{a+2}·H_b (gap k-2,
known), and w_1 acting on canonical seeds.

w_1·(canonical seed) by seed type:
  (0, e) (Wilson χ_e): w_1·χ_e = χ_{e-1} + χ_{e+1}, with χ_{-1} = 0
                       and χ_0 = identity.
  (1, n) (single H_n): w_1·H_n = q·H_{n-1} + q^{-1}·H_{n+1}
                                  + [n odd: W_0 = identity]
  (m≥2):              decompose via h_mul_h iteratively.

Bootstrap: gap 0 (H_a^2), gap 1, 2, 3 known explicitly.
"""
from __future__ import annotations
from laurent_poly import LaurentPoly


_cache: dict = {}


def _add(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, LaurentPoly.zero()) + v
    return {k: v for k, v in out.items() if not v.is_zero()}


def _scale(d, factor):
    return {k: factor * v for k, v in d.items() if not v.is_zero()}


def _neg(d):
    return {k: LaurentPoly.zero() - v for k, v in d.items() if not v.is_zero()}


# ---------- Bootstrap: explicit gap 0, 1, 2, 3 -----------------------

def _gap0(a):
    return {(2, 2 * a): LaurentPoly({0: 1})}


def _gap1(a):
    # H_a · H_{a+1} = q · L_{(2, 2a+1)} (within-cone, cocycle 1).
    return {(2, 2 * a + 1): LaurentPoly({1: 1})}


def _gap2(a):
    if a % 2 == 0:
        return {(2, 2 * (a + 1)): LaurentPoly({2: 1})}
    else:
        return {
            (1, a + 1): LaurentPoly({1: 1}),
            (2, 2 * (a + 1)): LaurentPoly({2: 1}),
        }


def _gap3(a):
    bridge_e = a + 1 if (a + 1) % 2 == 1 else a + 2
    return {
        (1, bridge_e): LaurentPoly({1: 1}),
        (2, 2 * a + 3): LaurentPoly({3: 1}),
    }


# ---------- w_1 acting on canonical seeds ---------------------------

def w1_mul_seed(seed: tuple) -> dict:
    """Compute w_1 · L_seed as a canonical decomposition.

    Handles m=0, m=1, m=2 seeds.  For m=2 (= some H_alpha · H_beta), we
    use h_mul_h to reduce.
    """
    m, e = seed
    if m == 0:
        # w_1 · χ_e = χ_{e-1} + χ_{e+1} (SU(2) Clebsch); χ_{-1} = 0.
        out = {}
        if e >= 1:
            out[(0, e - 1)] = LaurentPoly({0: 1})
        out[(0, e + 1)] = out.get((0, e + 1), LaurentPoly.zero()) + LaurentPoly({0: 1})
        return out
    if m == 1:
        # w_1 · H_n = q·H_{n-1} + q^{-1}·H_{n+1} + [n odd: 1·W_0]
        n = e
        out = {
            (1, n - 1): LaurentPoly({1: 1}),
            (1, n + 1): LaurentPoly({-1: 1}),
        }
        if n % 2 == 1:
            out[(0, 0)] = LaurentPoly({0: 1})
        return out
    if m == 2:
        # Canonical seed L_{(2, e)} = q^{phase} · H_alpha · H_beta (literal)
        # where phase = -cocycle(H_alpha, H_beta) = -(beta - alpha)
        # for alpha < beta; phase = 0 if alpha == beta (H_alpha^2).
        alpha, beta = _seed_to_h_pair(seed)
        phase = -(beta - alpha) if alpha != beta else 0
        # Compute w_1 · H_alpha · H_beta (literal product), then apply
        # q^{phase} factor to convert to canonical basis.
        w1_Halpha = w1_mul_seed((1, alpha))
        literal: dict = {}
        for s, coef in w1_Halpha.items():
            prod = _mul_by_H_beta(s, beta)
            for s_out, c_out in prod.items():
                literal[s_out] = literal.get(s_out, LaurentPoly.zero()) + coef * c_out
        # Multiply by q^{phase}.
        out = {k: LaurentPoly({phase: 1}) * v for k, v in literal.items()
               if not v.is_zero()}
        return out
    # Invariant: w1_mul_seed is only ever applied to canonical seeds of
    # H*H products, whose magnetic charge m = (#H-factors) is at most 2.
    # m>2 cannot arise on any caller path -- this guards the invariant.
    raise AssertionError(
        f"w1_mul_seed: invariant m in {{0,1,2}} violated (m={m}); "
        f"seeds of H*H products carry m = #H-factors <= 2")


def _seed_to_h_pair(seed: tuple) -> tuple:
    """For a canonical (m=2, e) seed, return the underlying H_alpha·H_beta
    pair (alpha, beta) in max-diagonal cone-monomial form.

    From the cone-data convention: cone-monomial (a, b, c) at cone C_k
    has lattice (2a+b, b+2c) and (m, e) with m = a+b+c, e = 2k·m + (b+2c).
    For m=2: a+b+c=2, lattice (2a+b, b+2c) — max-diagonal: b = min(n1, n2),
    a = (n1-b)/2, c = (n2-b)/2.

    Returns (alpha, beta) = pair of H indices (alpha ≤ beta).
    """
    m, e = seed
    assert m == 2
    # In cone C_k: e = 2k·m + (b + 2c).  Need to find (k, a, b, c) with
    # a+b+c=m=2 and the lattice consistent.
    # Native: (alpha, exp_alpha=1), (beta, exp_beta=1) or single H_x^2.
    # Use: alpha = 2k·1·(a + b/2)/... actually easier just to deduce.
    # For m=2, possible cone monomials: H_{2k}^2, H_{2k}·H_{2k+1},
    # H_{2k+1}^2, H_{2k}·H_{2k+2} = q²·H_{2k+1}², H_{2k+1}·H_{2k+2},
    # H_{2k+2}^2.
    # The canonical max-diagonal picks (a, b, c) so that one of a, c
    # is 0 (= "diagonal" lattice positions).
    # In pSU2 (m=2, e): the H-pair is determined uniquely up to syzygy.
    # Specifically, (alpha, beta) with alpha+beta = e and alpha ≤ beta,
    # and beta - alpha ≤ 1 (max-diagonal = adjacent in H-tower).
    # alpha = (e - 1) // 2 if e odd, e // 2 if e even (and not H^2).
    # For e = 2n: H_n^2.  For e = 2n+1: H_n · H_{n+1}.
    if e % 2 == 0:
        # H_{e/2}^2
        return (e // 2, e // 2)
    else:
        n = (e - 1) // 2
        return (n, n + 1)


_mul_cache: dict = {}


def _mul_by_H_beta(left_seed: tuple, beta: int) -> dict:
    """Compute (left_seed) · H_beta canonical decomposition.

    left_seed can be (0, e) [Wilson] or (1, n) [single H].  Memoized.
    """
    key = (left_seed, beta)
    if key in _mul_cache:
        return _mul_cache[key]
    result = _mul_by_H_beta_impl(left_seed, beta)
    _mul_cache[key] = result
    return result


def _mul_by_H_beta_impl(left_seed: tuple, beta: int) -> dict:
    """Uncached implementation."""
    m, e = left_seed
    if m == 1:
        n = e
        if n <= beta:
            return h_mul_h(n, beta)
        # n > beta: H_n · H_beta.  If they q-commute (within a cone),
        #   H_n · H_beta = q^{2·cocycle(H_n, H_beta)} · H_beta · H_n
        #                = q^{2·(beta - n)} · h_mul_h(beta, n).
        # Otherwise (cross-cluster), this would need the reverse-order
        # cross_product, which is not implemented.
        from pure_su2_h_cone_data import cone_index_for
        n_lo, n_hi = min(n, beta), max(n, beta)
        cone_idx = cone_index_for(n_lo, n_hi)
        if cone_idx is not None:
            # Within a cone: q-commute, simple cocycle reorder.
            forward = h_mul_h(beta, n)
            factor = LaurentPoly({2 * (beta - n): 1})
            return {k: factor * v for k, v in forward.items()
                    if not v.is_zero()}
        # Invariant: the left seed here is a canonical term of w_1*H_alpha,
        # so |n - beta| <= 1 (adjacent in the H-tower); a cross-cluster
        # (|n-beta|>=3) reversed pair never reaches here.  Guard the invariant.
        raise AssertionError(
            f"_mul_by_H_beta: invariant |n-beta|<=1 violated "
            f"(H_{n}*H_{beta}, n>beta cross-cluster); cannot arise from the "
            f"w_1*H_alpha caller (adjacent seeds only)")
    if m == 0:
        # χ_e · H_beta — this is W·H, need empirical Clebsch.
        # For e=0 (= identity = W_0): result = H_beta.
        if e == 0:
            return {(1, beta): LaurentPoly({0: 1})}
        # For e=1 (= w_1): use h_times_w1 ordering... actually we want
        # χ_1 · H_beta which is w_1·H_beta = q·H_{beta-1} + q^{-1}·H_{beta+1}
        # + [beta odd: 1·W_0].
        if e == 1:
            out = {
                (1, beta - 1): LaurentPoly({1: 1}),
                (1, beta + 1): LaurentPoly({-1: 1}),
            }
            if beta % 2 == 1:
                out[(0, 0)] = LaurentPoly({0: 1})
            return out
        # For e ≥ 2: use χ_e · H_beta = (χ_{e-1} · χ_1)·H_beta − χ_{e-2}·H_beta
        # iteratively, via the χ Chebyshev recursion: χ_e = w_1·χ_{e-1} − χ_{e-2}.
        # So χ_e·H_beta = w_1·(χ_{e-1}·H_beta) − χ_{e-2}·H_beta.
        # And w_1·X = w1_mul_seed iteratively over canonical decomp.
        recur_em1 = _mul_by_H_beta((0, e - 1), beta)
        recur_em2 = _mul_by_H_beta((0, e - 2), beta) if e >= 2 else {}
        # w_1 · recur_em1: apply w1_mul_seed to each canonical term.
        w1_recur: dict = {}
        for s, c in recur_em1.items():
            ws = w1_mul_seed(s)
            for s_out, c_out in ws.items():
                w1_recur[s_out] = w1_recur.get(s_out, LaurentPoly.zero()) + c * c_out
        # Result = w_1·recur_em1 - recur_em2.
        return _add(w1_recur, _neg(recur_em2))
    raise AssertionError(
        f"_mul_by_H_beta: invariant left-seed m in {{0,1}} violated (m={m}); "
        f"left seeds are Wilson (m=0) or single H (m=1)")


# ---------- Main: H_a · H_b via downward walk -----------------------

def h_mul_h(a: int, b: int) -> dict:
    """Compute H_a · H_b in canonical pSU2 (m, e) labels.

    Total in (a, b).  For a <= b: recursion downward in the gap
    `b - a >= 0` (the decreasing termination variable), bootstrapping from
    the gap <= 3 explicit formulas.  For a > b: the bar-conjugate of the
    ordered pair (b, a).  Cached.
    """
    if a > b:
        # Total by bar-symmetry: the bar involution is antimultiplicative and
        # fixes the canonical basis (bar(H_n)=H_n), so
        #   H_a*H_b = bar(bar(H_a)*bar(H_b)) = bar(H_b*H_a).
        # Reduce the ordered pair (b, a) -- gap = a - b > 0 is the decreasing
        # recursion variable, the explicit termination guarantee -- and
        # conjugate q -> q^{-1}.  No precondition; this branch never raises.
        return {seed: LaurentPoly({-e: c for e, c in coef._coeffs.items()})
                for seed, coef in h_mul_h(b, a).items()}
    if (a, b) in _cache:
        return _cache[(a, b)]
    gap = b - a
    if gap == 0:
        result = _gap0(a)
    elif gap == 1:
        result = _gap1(a)
    elif gap == 2:
        result = _gap2(a)
    elif gap == 3:
        result = _gap3(a)
    else:
        # From (w_1·H_{a+1})·H_b = w_1·(H_{a+1}·H_b):
        #   q·H_a·H_b + q^{-1}·H_{a+2}·H_b + [(a+1) odd: H_b]
        #     = w_1·(H_{a+1}·H_b)
        # Solving for H_a·H_b:
        #   H_a·H_b = q^{-1}·w_1·(H_{a+1}·H_b) − q^{-2}·H_{a+2}·H_b
        #             − q^{-1}·[(a+1) odd: H_b]
        Hap1_b = h_mul_h(a + 1, b)
        Hap2_b = h_mul_h(a + 2, b)
        w1_term: dict = {}
        for s, c in Hap1_b.items():
            ws = w1_mul_seed(s)
            for s_out, c_out in ws.items():
                w1_term[s_out] = w1_term.get(s_out, LaurentPoly.zero()) + c * c_out
        result = _add(
            _scale(w1_term, LaurentPoly({-1: 1})),       # +q^{-1} · w_1·(H_{a+1}·H_b)
            _scale(Hap2_b, LaurentPoly({-2: -1})),       # −q^{-2} · H_{a+2}·H_b
        )
        if (a + 1) % 2 == 1:
            result = _add(result, {(1, b): LaurentPoly({-1: -1})})  # −q^{-1}·H_b
    _cache[(a, b)] = result
    return result


def clear_cache():
    _cache.clear()
    _mul_cache.clear()
