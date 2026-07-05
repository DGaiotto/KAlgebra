"""BPS-label ↔ (m, e, μ_pow) seed decoder for SU(2) + N_f = 1.

The auxiliary `bps_su2_nf1` produces multiply outputs labelled by BPS
Z^3 vectors `(n_1, n_2, n_3)`.  The "matter" / μ-content is the 3rd
coordinate (since γ_3 is the kernel of the pairing matrix).  The
"pure" part `(n_1, n_2, 0)` identifies the canonical-basis seed
`(m, e)` per the H-tower piecewise structure.

Single-letter seeds:
    (0, 0, 0)                       → (m=0, e=0)  identity
    (0, -e, 0) for e ≥ 1            → (m=0, e)    χ_e Wilson
    (1, n, 0) for n ≤ 0             → (m=1, e=n)  H_n (piece n ≤ 0)
    (0, 1, 0)                       → (m=1, e=1)  H_1 (piece n=1)
    (-1, 4-n, 0) for n ≥ 2          → (m=1, e=n)  H_n (piece n ≥ 2)

Multi-letter seeds (m ≥ 2): identified by their BPS-sum = sum of
single-letter BPS labels of the constituent H-rays (within a single
max-diagonal cone {3k, 3k+1, 3k+2}).
"""
from __future__ import annotations


def _h_bps(n: int) -> tuple:
    if n <= 0:
        return (1, n, 0)
    if n == 1:
        return (0, 1, 0)
    return (-1, 4 - n, 0)


def _decode_single(pure: tuple):
    """Decode pure-part (n_1, n_2, 0) → (m, e) seed if it's a single-letter
    canonical-basis element, else None."""
    n1, n2, n3 = pure
    if n3 != 0:
        raise ValueError(f"_decode_single: expected n_3=0, got {pure}")
    if pure == (0, 0, 0):
        return (0, 0)
    if n1 == 0 and n2 >= 1:
        # H_n single: only n=1 (BPS (0, 1, 0)).  n ≥ 2 multi-letter (sum of m=2 H's).
        if n2 == 1:
            return (1, 1)
        return None
    if n1 == 0 and n2 <= -1:
        return (0, -n2)                          # χ_e Wilson
    if n1 == 1 and n2 <= 0:
        return (1, n2)                           # H_n, n ≤ 0
    if n1 == -1 and n2 <= 2:
        return (1, 4 - n2)                       # H_n, n ≥ 2 (n2 ≤ 2)
    return None


def bps_to_seed(v: tuple) -> tuple:
    """Decode BPS Z^3 label `(n_1, n_2, n_3)` to `((m, e), μ_pow)`.

    Single-letter case: returns the single-letter seed + μ_pow.
    Multi-letter case (m ≥ 2): returns the cone-monomial seed (m, e)
    where m = number of H-letters, e = sum of H-letter indices.
    Identifies the H-content by exhaustively decomposing the pure
    part into a sum of single-letter H-labels in a single cone.
    """
    n1, n2, n3 = v
    mu_pow = n3
    pure = (n1, n2, 0)
    seed = _decode_single(pure)
    if seed is not None:
        return (seed, mu_pow)
    # Multi-letter: search rank-2 cones {H_a, H_{a+1}} for all a, and
    # higher-m chains.  Each canonical (m, e) chain corresponds to a
    # multi-set of H-letters in CONSECUTIVE indices (pair cones) summing
    # to `pure`.  We enumerate (start_idx, counts) over a small window.
    for total in range(2, 8):
        for start in range(-8, 8):
            # Distribute `total` H-letters across H_start, H_{start+1}, ...
            # Use a small recursion / nested loops up to total length.
            # Simpler: try chains of consecutive indices with multiplicities.
            for n_distinct in range(1, min(total, 4) + 1):
                # Enumerate multiplicity vectors of length n_distinct summing to total.
                def enum_mult(rem, length):
                    if length == 1:
                        if rem >= 1:
                            yield (rem,)
                        return
                    for k in range(1, rem - length + 2):
                        for tail in enum_mult(rem - k, length - 1):
                            yield (k,) + tail
                for mults in enum_mult(total, n_distinct):
                    indices = [start + i for i in range(n_distinct)]
                    s = (0, 0, 0)
                    e_sum = 0
                    for idx, mult in zip(indices, mults):
                        b_ = _h_bps(idx)
                        s = (s[0] + mult * b_[0], s[1] + mult * b_[1], 0)
                        e_sum += mult * idx
                    if s == pure:
                        return ((total, e_sum), mu_pow)
    raise ValueError(f"bps_to_seed: cannot decode {v}")


def bps_dict_to_seed_dict(d: dict, R) -> dict:
    """Convert `{BPS_tuple: LaurentPoly}` to `{(m, e) seed: RLaurent[R]}`
    by tagging μ_pow on each coefficient."""
    from zplus_ring import RLaurent
    out: dict = {}
    for v, lp in d.items():
        seed, mu_pow = bps_to_seed(v)
        # lp is LaurentPoly in q; convert to RLaurent with μ^{mu_pow}.
        rl = RLaurent(R, {e: R.basis_element((mu_pow,)) * c
                          for e, c in lp._coeffs.items()})
        if seed in out:
            out[seed] = out[seed] + rl
        else:
            out[seed] = rl
    return out
