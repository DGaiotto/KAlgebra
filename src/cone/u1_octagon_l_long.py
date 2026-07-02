"""
u1_octagon_l_long.py — closed form for Tr_U1Octagon(L_long * v^n).

Empirical 4-partial-theta formula (verified against K=35 BPSKAlgebra
trace data for n in [0, 3]):

    Tr_U1Oct(L_long * v^n)  =  (-1)^{n+1} * [
        + sum_k fq^{8k^2 + (8n+10)k + (3n+1)}      (atypical-like +)
        + sum_k fq^{8k^2 + (8n+18)k + (9n+11)}     (log-like +)
        - sum_k fq^{8k^2 + (8n+12)k + (5n+3)}      (atypical-like -)
        - sum_k fq^{8k^2 + (8n+16)k + (9n+13)}     (log-like -) ]

Structural notes
----------------
The two "atypical-like" branches have b-coefficients = 2p*n + 10 (+)
and 2p*n + 12 (-), with c-slopes p-1 and p+1 respectively.  These
generalise the analogous U1Hex (p=3) branches in u1_hexagon_singlet
.tr_L20_v_n.

The two "log-like" branches have c-slopes 9 = 3(p-1) for both signs,
with c-intercepts 11 (+) and 13 (-).  Their b-coefficients are
CONJECTURAL (slope = 2p, intercepts 18 and 16, by analogy with
U1Hex's 2p-slope log-like branches); the K=35 data only verifies
the k=0 values directly.

Cross-rung pattern (U1Hex vs U1Oct):

                     U1Hex (p=3)            U1Oct (p=4)
    atypical +       c=4n+3, b=6n+10        c=3n+1, b=8n+10
    atypical -       c=2n+1, b=6n+8         c=5n+3, b=8n+12
    log +            c=6n+9, b=6n+16        c=9n+11, b=8n+18 (conj)
    log -            c=6n+7, b=6n+14        c=9n+13, b=8n+16 (conj)

Note the c-slopes for the LOWEST atypical-like branch swap between
p=3 and p=4 (4 -> 3 for +; 2 -> 5 for -).  The atypical b-intercepts
{10, 12} are constant across p (matching U1Hex's {10, 8} apart from
the - intercept which increases by 4 per p).
"""

from __future__ import annotations

from laurent_poly import LaurentPoly


def tr_L_long_v_n(n: int, K: int) -> LaurentPoly:
    """Closed-form Tr_U1Oct(L_long * v^n) as a fq-Laurent polynomial
    truncated to coefficients of fq^k for k <= K.

    Args:
        n: U(1) charge (non-negative; the n < 0 case is not implemented).
        K: maximum fq-power to retain.
    """
    if n < 0:
        raise NotImplementedError("n < 0 case for U1Oct L_long TBD")
    sign = -1 if ((n + 1) % 2 == 1) else 1     # (-1)^{n+1}
    out: dict[int, int] = {}
    for s_intr, b, c in [
        (+1, 8 * n + 10, 3 * n + 1),    # atypical-like +
        (+1, 8 * n + 18, 9 * n + 11),   # log-like +
        (-1, 8 * n + 12, 5 * n + 3),    # atypical-like -
        (-1, 8 * n + 16, 9 * n + 13),   # log-like -
    ]:
        k = 0
        while True:
            e = 8 * k * k + b * k + c
            if e > K:
                break
            out[e] = out.get(e, 0) + sign * s_intr
            k += 1
    out = {e: v for e, v in out.items() if v != 0}
    return LaurentPoly(out)


def _verify_against_observed() -> None:
    observed = {
        0: {1: -1, 3: 1, 11: -1, 13: 1, 19: -1, 23: 1},
        1: {4: 1, 8: -1, 20: 1, 22: -1, 30: 1},
        2: {7: -1, 13: 1, 29: -1, 31: 1},
        3: {10: 1, 18: -1},
    }
    print("Verifying tr_L_long_v_n against observed K=35 data:")
    for n, obs in observed.items():
        pred_lp = tr_L_long_v_n(n, K=35)
        pred = pred_lp._coeffs
        match = pred == obs
        print(f"  n={n}: match = {match}")
        if not match:
            print(f"    predicted: {sorted(pred.items())}")
            print(f"    observed:  {sorted(obs.items())}")


if __name__ == "__main__":
    _verify_against_observed()
