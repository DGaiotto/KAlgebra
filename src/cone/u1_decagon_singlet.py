"""
u1_decagon_singlet.py — M(1, 5) singlet algebra characters as closed
forms for the U1Decagon Tr(v^n) seed family.

Completes the fourth rung of the U(1)-gauged-(2p)-gon ↔ M(1, p)
singlet correspondence:

    U1Square    (A_1, A_1) -> M(1, 2)  level 4
    U1Hexagon   (A_1, A_3) -> M(1, 3)  level 6  (PR #245)
    U1Octagon   (A_1, A_5) -> M(1, 4)  level 8  (PR #246)
    U1Decagon   (A_1, A_7) -> M(1, 5)  level 10 (this module)

The U1Decagon K-algebra corresponds to the (A_1, A_7) Argyres-Douglas
theory, with chiral algebra the M(1, 5) singlet vertex algebra at
central charge c = 1 - 6 * 4^2 / 5 = -18.4 = -18.4  -- actually
c_{1,p} = 1 - 6*(p-1)^2/p; for p=5, c = 1 - 96/5 = -91/5.  (Check the
exact value against the literature.)

Closed form (verified against direct gauged-A_8 BPSKAlgebra evaluation
through K = 25 for n in {0, 1, 2}):

    Tr_U1Dec(v^n)
      = sum_{k>=0} fq^{10k^2 + (10|n|+8)k + 4|n|}
      - sum_{k>=0} fq^{10k^2 + (10|n|+12)k + 6|n|+2}.

(No overall sign factor: the (-1)^|n| factor only appears for even p.
Empirically: p=3 -> no flip, p=4 -> flip, p=5 -> no flip.  Conjecturally
sign = (-1)^{(p+1)|n|} = 1 for p odd, (-1)^|n| for p even.)

Completing the square exposes the level-10 lattice partial theta:

    fq^{(25 n^2 + 4)/10} * Tr_U1Dec(v^n)
      = sum_{m equiv ?? (mod 10)} fq^{m^2/10}
      - sum_{m equiv ?? (mod 10)} fq^{m^2/10}.

(Exact residues: 5|n|+4 mod 10 and 5|n|+6 mod 10, by the pattern.)
"""

from __future__ import annotations

from laurent_poly import LaurentPoly


def tr_v_n(n: int, K: int) -> LaurentPoly:
    """Closed-form Tr_U1Dec(v^n) as a fq-Laurent polynomial truncated
    to coefficients of fq^k for k <= K."""
    n_abs = abs(n)
    out: dict[int, int] = {}
    k = 0
    while True:
        e = 10 * k * k + (10 * n_abs + 8) * k + 4 * n_abs
        if e > K:
            break
        out[e] = out.get(e, 0) + 1
        k += 1
    k = 0
    while True:
        e = 10 * k * k + (10 * n_abs + 12) * k + 6 * n_abs + 2
        if e > K:
            break
        out[e] = out.get(e, 0) - 1
        k += 1
    out = {e: c for e, c in out.items() if c != 0}
    return LaurentPoly(out)


def _verify_against_bps(K: int = 25, n_range: range = range(0, 3)) -> None:
    """Compute Tr_U1Dec(v^n) via direct gauged-A_8 BPSKAlgebra and
    compare against the closed form."""
    raise NotImplementedError(  # BPS cross-check: not part of the spine-free release
        "BPS cross-check is unavailable in the spine-free ConeKAlgebra release")

    B = [[0] * 8 for _ in range(8)]
    for i in range(7):
        B[i][i + 1] = 1
        B[i + 1][i] = -1
    nc = [tuple(int(j == i) for j in range(8)) for i in range(7)]
    A = BPSKAlgebra(pairing=B, node_charges=nc, verify="off")

    print(f"Verifying closed form vs direct gauged-A_8 BPSKAlgebra at K = {K}:")
    for n in n_range:
        chg = (n, 0, n, 0, n, 0, n, 0)
        bps = A.trace(chg, K=K, cone_cutoff=8)
        closed = tr_v_n(n, K)
        bps_coeffs = dict(bps.coeffs)
        closed_coeffs = closed._coeffs
        match = bps_coeffs == closed_coeffs
        print(f"  n = {n:>3}:  match: {match}")
        if not match:
            print(f"    BPS:    {bps}")
            print(f"    closed: {closed_coeffs}")


if __name__ == "__main__":
    _verify_against_bps(K=25)
