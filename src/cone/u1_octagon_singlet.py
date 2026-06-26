"""
u1_octagon_singlet.py — M(1, 4) singlet algebra characters as closed
forms for the U1Octagon Tr(v^n) seed family.

Mirrors `u1_hexagon_singlet.py` (M(1, 3)) one level up.

The U(1)-gauged octagon K-algebra `U1OctagonKAlg` corresponds to the
(A_1, A_5) Argyres-Douglas theory.  Its chiral algebra is the
M(1, 4) singlet vertex algebra at central charge c = -25/2.  The
trace seed family Tr_U1Oct(v^n) for n in Z gives the bare partial-
theta characters of this algebra at U(1)-charge n.

Closed form (verified against direct gauged-A_6 BPSKAlgebra evaluation
through K = 30 for n in {0, 1, 2}):

    Tr_U1Oct(v^n)
        =  (-1)^|n| · [ sum_{k >= 0} fq^{8k^2 + (8|n|+6)k + 3|n|}
                       - sum_{k >= 0} fq^{8k^2 + (8|n|+10)k + 5|n|+2} ]

Completing the square exposes the level-8 lattice partial theta with
insertions at the residue cosets 4|n|+3 and 4|n|+5 mod 8:

    fq^{(16n^2 + 9)/8} · Tr_U1Oct(v^n)
        =  (-1)^|n| · [ sum_{m equiv 4|n|+3 (mod 8), m > 0} fq^{m^2/8}
                       - sum_{m equiv 4|n|+5 (mod 8), m > 0} fq^{m^2/8} ]

For n even the residues mod 8 are {3, 5} (vacuum-sector modules); for
n odd they are {7, 1} (twisted-sector modules).
"""

from __future__ import annotations

from laurent_poly import LaurentPoly


def tr_v_n(n: int, K: int) -> LaurentPoly:
    """Closed-form Tr_U1Oct(v^n) as a fq-Laurent polynomial truncated
    to coefficients of fq^k for k <= K."""
    n_abs = abs(n)
    sign = -1 if (n_abs % 2 == 1) else 1
    out: dict[int, int] = {}
    # + partial theta (multiplied by sign).
    k = 0
    while True:
        e = 8 * k * k + (8 * n_abs + 6) * k + 3 * n_abs
        if e > K:
            break
        out[e] = out.get(e, 0) + sign
        k += 1
    # - partial theta (multiplied by sign).
    k = 0
    while True:
        e = 8 * k * k + (8 * n_abs + 10) * k + 5 * n_abs + 2
        if e > K:
            break
        out[e] = out.get(e, 0) - sign
        k += 1
    out = {e: c for e, c in out.items() if c != 0}
    return LaurentPoly(out)


def _verify_against_bps(K: int = 30, n_range: range = range(-2, 3)) -> None:
    """Compute Tr_U1Oct(v^n) via direct gauged-A_6 BPSKAlgebra and
    compare against the closed form."""
    raise NotImplementedError(  # BPS cross-check: not part of the spine-free export
        "BPS cross-check is unavailable in the spine-free ConeKAlgebra export")

    # gauged A_6 (= U1Octagon's hidden BPS):  A_6 antisymmetric pairing,
    # first 5 standard basis vectors as node_charges, MU_CHARGE = (1,0,1,0,1,0).
    B = [[0] * 6 for _ in range(6)]
    for i in range(5):
        B[i][i + 1] = 1
        B[i + 1][i] = -1
    nc = [tuple(int(j == i) for j in range(6)) for i in range(5)]
    A = BPSKAlgebra(pairing=B, node_charges=nc, verify="off")

    print(f"Verifying closed form vs direct gauged-A_6 BPSKAlgebra at K = {K}:")
    for n in n_range:
        chg = (n, 0, n, 0, n, 0)
        bps = A.trace(chg, K=K)
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
