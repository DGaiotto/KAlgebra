"""
u1_decagon_l_long.py — closed form for Tr_U1Decagon(L_long * v^n).

Completes the L_long Layer 2 across the U1Pgon hierarchy:

    U1Hexagon (p=3): u1_hexagon_singlet.tr_L20_v_n   (PR #245)
    U1Octagon (p=4): u1_octagon_l_long.tr_L_long_v_n (PR #255)
    U1Decagon (p=5): tr_L_long_v_n (this module)

Closed form (verified vs direct gauged-A_8 BPSKAlgebra for n in {1, 2}
through K=30; n=0 verification pending due to BPS memory cost):

    Tr_U1Dec(L_long * v^n)  =  + sum_k fq^{10k^2 + (10n+14)k + (6n+3)}
                              + sum_k fq^{10k^2 + (10n+20)k + (12n+17)}
                              - sum_k fq^{10k^2 + (10n+12)k + (4n+1)}
                              - sum_k fq^{10k^2 + (10n+18)k + (12n+15)}

(NO overall (-1)^{n+1} sign: like U1Hex (p=3), the p=5 case has no
sign-flip; only even p flips.)

Cross-rung pattern (now complete for p = 3, 4, 5):

                     U1Hex (p=3)            U1Oct (p=4)            U1Dec (p=5)
    sign factor      +1                     (-1)^{n+1}              +1
    atypical +       c=4n+3, b=6n+10        c=3n+1, b=8n+10         c=6n+3, b=10n+14
    atypical -       c=2n+1, b=6n+8         c=5n+3, b=8n+12         c=4n+1, b=10n+12
    log +            c=6n+9, b=6n+16        c=9n+11, b=8n+18        c=12n+17, b=10n+20
    log -            c=6n+7, b=6n+14        c=9n+13, b=8n+16        c=12n+15, b=10n+18

Atypical c-slopes interpolate as {p+1 if p odd else p-1} (for +)
and {p-1 if p odd else p+1} (for -); the assignment swaps with
parity of p.  Log c-slope = 3(p-1), b-slope = 2p (= b-slope of
atypical).  c-intercept for log + and - tracks (4p-3, 4p-5) or
(4p-5, 4p-3) by parity of p.
"""

from __future__ import annotations

from laurent_poly import LaurentPoly


def tr_L_long_v_n(n: int, K: int) -> LaurentPoly:
    """Closed-form Tr_U1Dec(L_long * v^n) as a fq-Laurent polynomial
    truncated to fq^k for k <= K.  For n < 0 the symmetry is TBD."""
    if n < 0:
        raise NotImplementedError("n < 0 case for U1Dec L_long TBD")
    sign = 1  # p=5 odd -> no flip
    out: dict[int, int] = {}
    for s_intr, b, c in [
        (+1, 10 * n + 14, 6 * n + 3),    # atypical-like +
        (+1, 10 * n + 20, 12 * n + 17),  # log-like +
        (-1, 10 * n + 12, 4 * n + 1),    # atypical-like -
        (-1, 10 * n + 18, 12 * n + 15),  # log-like -
    ]:
        k = 0
        while True:
            e = 10 * k * k + b * k + c
            if e > K:
                break
            out[e] = out.get(e, 0) + sign * s_intr
            k += 1
    out = {e: v for e, v in out.items() if v != 0}
    return LaurentPoly(out)


def _verify_against_observed() -> None:
    """Compare against the K=30 BPSKAlgebra data for n in {1, 2}."""
    observed = {
        # n=0 pending (BPS OOMs).
        1: {5: -1, 9: 1, 27: -1, 29: 1},
        2: {9: -1, 15: 1},
    }
    print("Verifying tr_L_long_v_n against K=30 observed data:")
    for n, obs in observed.items():
        pred_lp = tr_L_long_v_n(n, K=30)
        pred = pred_lp._coeffs
        pred_in_range = {k: v for k, v in pred.items() if k <= max(obs.keys())}
        match = pred_in_range == obs
        print(f"  n={n}: match = {match}")
        if not match:
            print(f"    predicted: {sorted(pred_in_range.items())}")
            print(f"    observed:  {sorted(obs.items())}")


if __name__ == "__main__":
    _verify_against_observed()
