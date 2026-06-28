"""
Cluster mutations in the quantum torus algebra.

Implements conjugation by the quantum dilogarithm E_q(x_gamma):

    mutate(A, gamma)   = E_q(x_gamma)^{-1} * A * E_q(x_gamma)
    unmutate(A, gamma) = E_q(x_gamma) * A * E_q(x_gamma)^{-1}

Key identities (Section 5 of the reference):

- <gamma, alpha> = m <= 0:
    mutate(x_alpha) = sum_{k=0}^{|m|} [|m| choose k]_q  x_{alpha + k*gamma}

- <gamma, alpha> = m > 0:
    mutate maps the q-binomial packet  sum_{k=0}^m [m,k]_q x_{alpha+k*gamma}
    back to x_alpha.  Single monomials give infinite series.

unmutate swaps the roles: positive pairing expands, negative contracts.

Convention: <(a,b),(c,d)> = ad - bc  (Dirac pairing).
"""

from __future__ import annotations
from laurent_poly import LaurentPoly, QuantumTorus, x as xt
from collections import defaultdict
from functools import lru_cache


# ── q-Binomial coefficients ─────────────────────────────────────

@lru_cache(maxsize=None)
def q_binomial(n: int, k: int) -> LaurentPoly:
    """Symmetric q-binomial coefficient [n choose k]_q.

    [n,k]_q = q^k [n-1,k]_q + q^{-(n-k)} [n-1,k-1]_q

    Examples:
        [2,1]_q = q + q^-1
        [3,1]_q = q^2 + 1 + q^-2
    """
    if k < 0 or k > n or n < 0:
        return LaurentPoly.zero()
    if k == 0 or k == n:
        return LaurentPoly.one()
    return (LaurentPoly.q(k) * q_binomial(n - 1, k)
            + LaurentPoly.q(-(n - k)) * q_binomial(n - 1, k - 1))


def q_integer(n: int) -> LaurentPoly:
    """Quantum integer [n]_q = q^{n-1} + q^{n-3} + ... + q^{-(n-1)}."""
    if n == 0:
        return LaurentPoly.zero()
    return q_binomial(n, 1)


# ── Internal helpers ─────────────────────────────────────────────

def _xgcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended GCD: returns (g, s, t) with g = s*a + t*b, g >= 0."""
    if a == 0:
        s = 1 if b >= 0 else -1
        return (abs(b), 0, s)
    g, s, t = _xgcd(b % a, a)
    return (g, t - (b // a) * s, s)


def _dirac(alpha: tuple[int, int], beta: tuple[int, int]) -> int:
    """Dirac pairing <alpha, beta> = a*d - b*c."""
    return alpha[0] * beta[1] - alpha[1] * beta[0]


def _decompose(elem: QuantumTorus, gamma: tuple[int, int]):
    """Split element by gamma-lines.

    Returns (lines, bezout) where:
        lines:  dict  line_key -> {offset: LaurentPoly}
        bezout: (p, r) with p*g1 + r*g2 = 1

    line_key = g2*a - g1*b  =  -<gamma, (a,b)>
    offset   = p*a + r*b
    """
    g1, g2 = gamma
    g, p, r = _xgcd(g1, g2)
    if g != 1:
        raise ValueError(f"gamma={gamma} must be primitive (gcd={g})")

    lines: dict[int, dict[int, LaurentPoly]] = defaultdict(dict)
    for (a, b), c in elem._terms.items():
        key = g2 * a - g1 * b
        off = p * a + r * b
        lines[key][off] = c

    return dict(lines), (p, r)


def _recover(lk: int, off: int,
             gamma: tuple[int, int], bez: tuple[int, int]) -> tuple[int, int]:
    """Recover charge (a, b) from line_key and offset."""
    g1, g2 = gamma
    p, r = bez
    return (r * lk + g1 * off, -p * lk + g2 * off)


def _bpoly(m: int) -> dict[int, LaurentPoly]:
    """B_m(z) = sum_{k=0}^m [m,k]_q z^k."""
    return {k: q_binomial(m, k) for k in range(m + 1)}


def _conv(A: dict[int, LaurentPoly],
          B: dict[int, LaurentPoly]) -> dict[int, LaurentPoly]:
    """Convolution (polynomial multiplication) in z."""
    out: dict[int, LaurentPoly] = {}
    for i, ai in A.items():
        for j, bj in B.items():
            k = i + j
            out[k] = out.get(k, LaurentPoly.zero()) + ai * bj
    return {k: v for k, v in out.items() if not v.is_zero()}


def _divmod_desc(C: dict[int, LaurentPoly],
                 B: dict[int, LaurentPoly]):
    """Descending polynomial long division.

    B must be monic (leading coefficient = 1).
    Handles Laurent polynomials in z by shifting to non-negative exponents.
    Returns (quotient, remainder) with deg(remainder) < deg(B).
    """
    C = {k: v for k, v in C.items() if not v.is_zero()}
    if not C:
        return {}, {}

    sh = min(C)
    Cs = {k - sh: v for k, v in C.items()}
    dB = max(B)

    Q: dict[int, LaurentPoly] = {}
    while True:
        Cs = {k: v for k, v in Cs.items() if not v.is_zero()}
        if not Cs:
            break
        dC = max(Cs)
        if dC < dB:
            break
        lc = Cs[dC]
        s = dC - dB
        Q[s] = lc
        for j, bj in B.items():
            Cs[j + s] = Cs.get(j + s, LaurentPoly.zero()) - lc * bj

    Cs = {k: v for k, v in Cs.items() if not v.is_zero()}
    return ({k + sh: v for k, v in Q.items()},
            {k + sh: v for k, v in Cs.items()})


def _ascend(C: dict, B: dict, kmax):
    """Ascending division up to kmax.

    Process offsets k_min..kmax, setting Q[k] = C'[k] at each step.
    Returns (Q, G) where G is the correction needed above kmax.
    Property: C + G = B * Q  with G supported on (kmax, ...).
    Keys may be int or Fraction (for weight-lattice charges).
    """
    C = dict(C)
    Q: dict = {}

    if not C:
        return {}, {}
    kmin = min(C)

    # Build list of integer-spaced keys from kmin to kmax
    nsteps = int(kmax - kmin)
    keys = [kmin + i for i in range(nsteps + 1)]

    for k in keys:
        C = {j: v for j, v in C.items() if not v.is_zero()}
        dk = C.get(k, LaurentPoly.zero())
        if dk.is_zero():
            continue
        Q[k] = dk
        for j, bj in B.items():
            C[k + j] = C.get(k + j, LaurentPoly.zero()) - dk * bj

    C = {k: v for k, v in C.items() if not v.is_zero()}
    G = {k: -v for k, v in C.items()}
    return Q, {k: v for k, v in G.items() if not v.is_zero()}


# ── Core conjugation engine ─────────────────────────────────────

def _conjugate(element: QuantumTorus, gamma: tuple[int, int],
               forward: bool = True) -> QuantumTorus:
    """Shared engine for mutate/unmutate.

    forward=True:  E_q^{-1} A E_q   (expand when m<=0, contract when m>0)
    forward=False: E_q A E_q^{-1}   (expand when m>=0, contract when m<0)
    """
    if element.is_zero():
        return element

    lines, bez = _decompose(element, gamma)
    out: dict[tuple[int, int], LaurentPoly] = {}

    def _add(ch, c):
        out[ch] = out.get(ch, LaurentPoly.zero()) + c

    for lk, offs in lines.items():
        m = -lk  # m = <gamma, alpha>

        if m == 0:
            for k, c in offs.items():
                _add(_recover(lk, k, gamma, bez), c)
            continue

        # Determine expand vs contract
        if forward:
            expand = (m < 0)
        else:
            expand = (m > 0)

        if expand:
            img = _conv(offs, _bpoly(abs(m)))
            for k, c in img.items():
                _add(_recover(lk, k, gamma, bez), c)
        else:
            Q, R = _divmod_desc(offs, _bpoly(abs(m)))
            if R:
                raise ValueError(
                    f"Not finitely mutable: gamma-line with Dirac pairing "
                    f"m={m} has nonzero remainder after q-binomial division"
                )
            for k, c in Q.items():
                _add(_recover(lk, k, gamma, bez), c)

    return QuantumTorus(out)


# ── Public API ───────────────────────────────────────────────────

def mutate(element: QuantumTorus, gamma: tuple[int, int]) -> QuantumTorus:
    """Conjugation  E_q(x_gamma)^{-1} * element * E_q(x_gamma).

    Raises ValueError if the result is not a finite sum.
    """
    return _conjugate(element, gamma, forward=True)


def unmutate(element: QuantumTorus, gamma: tuple[int, int]) -> QuantumTorus:
    """Inverse conjugation  E_q(x_gamma) * element * E_q(x_gamma)^{-1}.

    Raises ValueError if the result is not a finite sum.
    """
    return _conjugate(element, gamma, forward=False)


def can_mutate(element: QuantumTorus, gamma: tuple[int, int]) -> bool:
    """Check if conjugation by E_q(x_gamma) produces a finite result."""
    try:
        mutate(element, gamma)
        return True
    except ValueError:
        return False


def complete(element: QuantumTorus, gamma: tuple[int, int],
             forward: bool = True) -> tuple[QuantumTorus, QuantumTorus]:
    """Find the minimal completion so that mutation/unmutation is finite.

    forward=True:  complete for mutate   (positive-pairing lines need packets)
    forward=False: complete for unmutate (negative-pairing lines need packets)

    Uses ascending division to add correction terms above the existing span.

    Returns (completed, correction) with  completed = element + correction.
    """
    if element.is_zero():
        return element, QuantumTorus.zero()

    lines, bez = _decompose(element, gamma)
    corr: dict[tuple[int, int], LaurentPoly] = {}

    for lk, offs in lines.items():
        m = -lk  # <gamma, alpha>

        # Which lines need completion?
        if forward:
            needs_completion = (m > 0)
        else:
            needs_completion = (m < 0)

        if not needs_completion:
            continue

        Bm = _bpoly(abs(m))
        kmax = max(offs)
        _, G = _ascend(offs, Bm, kmax)

        for k, c in G.items():
            corr[_recover(lk, k, gamma, bez)] = c

    correction = QuantumTorus(corr)
    return element + correction, correction


def joint_complete(element: QuantumTorus,
                   fwd_gamma: tuple[int, int],
                   back_gamma: tuple[int, int],
                   max_iter: int = 50) -> tuple[QuantumTorus, QuantumTorus]:
    """Complete element so it's simultaneously mutable by fwd_gamma
    and unmutable by back_gamma.

    Iterates: complete for fwd_gamma mutation, then for back_gamma
    unmutation, until no more corrections are needed.

    Returns (completed, total_correction).
    """
    original = element
    F = element
    for _ in range(max_iter):
        F, c1 = complete(F, fwd_gamma, forward=True)
        F, c2 = complete(F, back_gamma, forward=False)
        if c1.is_zero() and c2.is_zero():
            return F, F - original
    raise RuntimeError("Joint completion did not converge")


def canonical_element(a: int, b: int,
                      gamma1: tuple[int, int] = (1, 0),
                      gamma2: tuple[int, int] = (0, 1)) -> QuantumTorus:
    """Compute the canonical basis element at charge (a, b).

    Pipeline: complete+mutate by gamma1, joint-complete+mutate by gamma2,
    then unmutate back through gamma2 and gamma1.
    """
    F = QuantumTorus.x(a, b)

    # Forward: complete and mutate by gamma1
    F1, _ = complete(F, gamma1, forward=True)
    M1 = mutate(F1, gamma1)

    # Forward: joint-complete for gamma2 mutation + gamma1 unmutation
    F2, _ = joint_complete(M1, fwd_gamma=gamma2, back_gamma=gamma1)
    M2 = mutate(F2, gamma2)

    # Backward: unmutate cleanly (no completions needed)
    U1 = unmutate(M2, gamma2)
    U2 = unmutate(U1, gamma1)

    return U2


def tropical_mutate(charges: list[tuple[int, int]], i: int
                    ) -> list[tuple[int, int]]:
    """Tropical mutation of mutable charges at index i.

    gamma_i  ->  -gamma_i
    gamma_j  ->  gamma_j + max(<gamma_j, gamma_i>, 0) * gamma_i   (j != i)
    """
    result = list(charges)
    gi = charges[i]
    result[i] = (-gi[0], -gi[1])
    for j in range(len(charges)):
        if j == i:
            continue
        d = _dirac(charges[j], gi)
        if d > 0:
            result[j] = (charges[j][0] + d * gi[0],
                         charges[j][1] + d * gi[1])
    return result


# ── Demo ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from laurent_poly import x, q, q_inv

    print("=== Cluster Mutations in the Quantum Torus ===\n")

    # ── q-Binomials ──
    print("--- q-Binomial coefficients ---")
    for n in range(5):
        row = [q_binomial(n, k) for k in range(n + 1)]
        print(f"  n={n}: {[repr(c) for c in row]}")
    print()

    # ── q-Weyl example (Section 6) ──
    print("--- q-Weyl algebra: mutation at gamma=(0,1) ---")
    gamma = (0, 1)

    F_v  = x(0, 1)
    F_up = x(1, 0)
    F_um = x(-1, 0) + x(-1, 1)

    print(f"  F_v  = {F_v}")
    print(f"  F_u+ = {F_up}")
    print(f"  F_u- = {F_um}")

    print(f"\n  can_mutate(F_u+, (0,1)) = {can_mutate(F_up, gamma)}")
    print(f"  can_mutate(F_u-, (0,1)) = {can_mutate(F_um, gamma)}")
    print(f"  can_mutate(x(-1,0), (0,1)) = {can_mutate(x(-1, 0), gamma)}")

    print(f"\n  mu(F_u+) = {mutate(F_up, gamma)}")
    print(f"  mu(F_u-) = {mutate(F_um, gamma)}")
    print()

    # ── Completion ──
    print("--- Completion ---")
    comp, corr = complete(x(-1, 0), gamma)
    print(f"  complete(x(-1,0), (0,1)):")
    print(f"    completed  = {comp}")
    print(f"    correction = {corr}")
    print(f"    mu(completed) = {mutate(comp, gamma)}")
    print()

    # ── Pentagon example (Section 7) ──
    print("--- Pentagon algebra: mutation at gamma=(0,1) ---")
    L0 = x(1, 0)
    L1 = x(0, -1)
    L2 = x(-1, -1) + x(-1, 0)
    L3 = x(-1, 0) + x(-1, 1) + x(0, 1)

    for name, F in [("L0", L0), ("L1", L1), ("L2", L2), ("L3", L3)]:
        mutable = can_mutate(F, gamma)
        img = mutate(F, gamma) if mutable else "INFINITE"
        print(f"  mu({name} = {F})  =  {img}")
    print()

    # ── Tropical mutation ──
    print("--- Tropical mutation ---")
    charges = [(1, 0), (0, 1)]
    print(f"  charges = {charges}")
    print(f"  mutate at index 1: {tropical_mutate(charges, 1)}")
    print(f"  mutate at index 0: {tropical_mutate(charges, 0)}")
    print()

    # ── Round-trip ──
    print("--- Round-trip: mutate then unmutate ---")
    for name, F in [("F_u+", F_up), ("F_u-", F_um)]:
        F1 = mutate(F, gamma)
        F2 = unmutate(F1, gamma)
        print(f"  {name} = {F}  -->  mu = {F1}  -->  mu^-1 = {F2}  "
              f"  ok={F2 == F}")
