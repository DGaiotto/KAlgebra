"""
u1_hexagon_singlet.py — M(1, 3) singlet algebra characters as closed
forms for the U1Hexagon Tr(v^n) seed family.

The U(1)-gauged hexagon K-algebra `U1HexagonKAlg` has Layer-1 trace
reduction (in `u1_hexagon_kalg.py`) to two irreducible seed families:

    Tr(v^n)             for n ∈ Z          (label `((), n)`)
    Tr(L_{2,j} · v^n)   for n ∈ Z, j ∈ Z_3 (label `(((2, j, 1),), n)`)

(All L_{1,*} seeds have vanishing trace by the U(1)-mag-charge rule;
all (m, j) seeds with m != 0, 2 vanish similarly.)

This module gives the closed form for the FIRST family (Tr(v^n)),
identifying it as a partial-theta character of the **M(1, 3) singlet
vertex algebra** at U(1)-charge n.  Empirically verified through
fq^70 for n = 0, 1, 2, 3 and through fq^31 for n = 4, 5; symmetric
under n -> -n.

Closed form
-----------
Let q = fq² (the conventional CFT variable).  Then

    Tr_U1Hex(v^n)
        =  Σ_{k ≥ 0} fq^{6k² + (6|n|+4)k + 2|n|}
         − Σ_{k ≥ 0} fq^{6k² + (6|n|+8)k + 4|n|+2}
        =  Σ_{k ≥ 0} q^{3k² + (3|n|+2)k + |n|}
         − Σ_{k ≥ 0} q^{3k² + (3|n|+4)k + 2|n|+1}.

Completing the square exposes the partial-theta structure:

    fq^{(9n²+4)/6} · Tr_U1Hex(v^n)
        =  Σ_{m ≡ 3|n|+2 (mod 6), m > 0} fq^{m²/6}
         − Σ_{m ≡ 3|n|+4 (mod 6), m > 0} fq^{m²/6}.

These are the bare characters of the M(1, 3) singlet algebra
(central charge c = -7, the chiral algebra of the (A_1, A_3)
Argyres-Douglas theory) at U(1)-charge n, lattice √6 · Z, with
half-arguments at residue cosets {3|n|+2, 3|n|+4} mod 6.  These are
standard results on M(1, p) singlet characters and the chiral-algebra
description of Argyres-Douglas theories.
"""

from __future__ import annotations

from laurent_poly import LaurentPoly


def tr_v_n(n: int, K: int) -> LaurentPoly:
    """Closed-form Tr_U1Hex(v^n) as a fq-Laurent polynomial truncated to
    coefficients of fq^k for k <= K.

    Args:
        n: U(1) charge.  Symmetric under  n -> -n.
        K: maximum fq-power to retain.

    Returns:
        LaurentPoly representing the M(1, 3) singlet partial-theta
        character at charge n, in our fq variable.
    """
    n_abs = abs(n)
    out: dict[int, int] = {}
    k = 0
    while True:
        e = 6 * k * k + (6 * n_abs + 4) * k + 2 * n_abs
        if e > K:
            break
        out[e] = out.get(e, 0) + 1
        k += 1
    k = 0
    while True:
        e = 6 * k * k + (6 * n_abs + 8) * k + 4 * n_abs + 2
        if e > K:
            break
        out[e] = out.get(e, 0) - 1
        k += 1
    out = {e: c for e, c in out.items() if c != 0}
    return LaurentPoly(out)


def tr_L20_v_n(n: int, K: int) -> LaurentPoly:
    """Empirical closed form for Tr_U1Hex(L_{2,0} · v^n) as a fq-Laurent
    polynomial, valid for n >= 0.

    The series decomposes as a sum of FOUR partial-theta sums at
    level 6 (two with + sign, two with - sign), consistent with
    Tr(L · v^n) being a linear combination of TWO M(1, 3) singlet
    module characters (the Wilson line L_{2, 0} acts on the chiral
    algebra as a Z-module-shifting operator).

    Verified through fq^40 for n in [0, 3]:

        Tr_U1Hex(L_{2,0} v^n)
          =   sum_{k>=0} fq^{6k^2 + (6n+10)k + (4n+3)}
            + sum_{k>=0} fq^{6k^2 + (6n+16)k + (6n+9)}
            - sum_{k>=0} fq^{6k^2 + (6n+8)k  + (2n+1)}
            - sum_{k>=0} fq^{6k^2 + (6n+14)k + (6n+7)}.

    Note the heterogeneous linear-in-n slopes on the constant term:
    {2, 4, 6, 6}.  This indicates the four partial-thetas belong to
    TWO distinct M(1, 3) module characters with different shifts,
    rather than a single module character.  The decomposition as
    `Tr(L_{2,0} v^n) = a(n) chi_{r_1(n)} + b(n) chi_{r_2(n)}` for
    integer a, b and module-shift functions r_i is open follow-up.

    For n < 0 use the symmetry  Tr(L_{2,0} v^{-n}) = ?  (needs check).
    """
    if n < 0:
        raise NotImplementedError(
            "Tr_U1Hex(L_{2,0} v^n) for n < 0 not yet characterised."
        )
    out: dict[int, int] = {}
    for sign, b_off, c_off in [
        (+1, 6 * n + 10, 4 * n + 3),
        (+1, 6 * n + 16, 6 * n + 9),
        (-1, 6 * n + 8,  2 * n + 1),
        (-1, 6 * n + 14, 6 * n + 7),
    ]:
        k = 0
        while True:
            e = 6 * k * k + b_off * k + c_off
            if e > K:
                break
            out[e] = out.get(e, 0) + sign
            k += 1
    out = {e: c for e, c in out.items() if c != 0}
    return LaurentPoly(out)


def _verify_against_bps(K: int = 40, n_range: range = range(-4, 5)) -> None:
    """Compute Tr_U1Hex(v^n) both ways (BPS-backed `.trace` and the
    closed form here) and report match status."""
    from u1_hexagon_kalg import U1HexagonKAlg

    H = U1HexagonKAlg()
    print(f"Verifying closed form vs BPS-backed trace at K = {K}:")
    for n in n_range:
        bps = H._bps_trace(((), n), K=K)
        closed = tr_v_n(n, K)
        # Compare coefficient dicts.
        bps_coeffs = dict(bps.coeffs)
        closed_coeffs = closed._coeffs
        match = bps_coeffs == closed_coeffs
        print(f"  Tr(v^{n:>3}):              match: {match}")
        if not match:
            print(f"    BPS:    {bps}")
            print(f"    closed: {closed_coeffs}")

    print(f"\nVerifying Tr(L_{{2,0}} v^n) closed form at K = {K}, n in [0, 3]:")
    for n in range(0, 4):
        label = (((2, 0, 1),), n)
        bps = H._bps_trace(label, K=K)
        closed = tr_L20_v_n(n, K)
        bps_coeffs = dict(bps.coeffs)
        closed_coeffs = closed._coeffs
        match = bps_coeffs == closed_coeffs
        print(f"  Tr(L_2_0 v^{n}):  match: {match}")
        if not match:
            diff_keys = set(bps_coeffs) ^ set(closed_coeffs)
            print(f"    BPS extra:    {sorted((k, bps_coeffs.get(k, 0)) for k in diff_keys if k in bps_coeffs)}")
            print(f"    closed extra: {sorted((k, closed_coeffs.get(k, 0)) for k in diff_keys if k in closed_coeffs)}")


if __name__ == "__main__":
    _verify_against_bps(K=30)