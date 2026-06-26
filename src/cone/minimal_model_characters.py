"""
minimal_model_characters.py — Characters of the M(2, 2k+3) Virasoro
minimal models (the chiral algebras of the [A_1, A_{2k}] Argyres-Douglas
theories).

By Andrews-Gordon (1974), the bare character (no q^{h-c/24} prefactor)
of the s-th irreducible module of M(2, 2k+3), with s in {1, ..., k+1},
admits two equivalent forms:

    chi_s^{(k)}(q)
        =  prod_{n >= 1,  n mod (2k+3) not in {0, s, -s}}  1/(1 - q^n)        (product side)

        =  sum_{n_1, ..., n_k >= 0}
               q^{N_1^2 + ... + N_k^2 + N_s + N_{s+1} + ... + N_k}
               -------------------------------------------------------
                       (q;q)_{n_1} (q;q)_{n_2} ... (q;q)_{n_k}                (sum side)

with  N_j = n_j + n_{j+1} + ... + n_k.

The k=1 case is the Rogers-Ramanujan / Lee-Yang case M(2,5):
    chi_1 = G(q) = prod 1/((q^2;q^5)_inf (q^3;q^5)_inf)
                 = sum q^{n^2 + n} / (q;q)_n        (vacuum module)
    chi_2 = H(q) = prod 1/((q;q^5)_inf (q^4;q^5)_inf)
                 = sum q^{n^2}     / (q;q)_n        (h = -1/5 module)

The k=2 case is M(2,7), with three modules whose characters appear in
the chiral algebra of [A_1, A_4].
"""

from __future__ import annotations


def char_product(k: int, s: int, MAX: int) -> list[int]:
    """Coefficients [q^0, q^1, ..., q^MAX] of chi_s^{(k)} via the product side.

    Args:
        k: positive integer; M(2, 2k+3) minimal model.
        s: module label in {1, ..., k+1}.
        MAX: highest q-power to compute.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not (1 <= s <= k + 1):
        raise ValueError(f"s must be in 1..{k+1}, got {s}")
    p = 2 * k + 3
    excluded = {0, s % p, (-s) % p}
    out = [0] * (MAX + 1)
    out[0] = 1
    for n in range(1, MAX + 1):
        if n % p in excluded:
            continue
        # Multiply by 1/(1 - q^n).
        for j in range(n, MAX + 1):
            out[j] += out[j - n]
    return out


def _qpoch_reciprocals(MAX: int) -> list[list[int]]:
    """Return [1/(q;q)_0, 1/(q;q)_1, ..., 1/(q;q)_MAX] each as a length-(MAX+1) list."""
    out = [[0] * (MAX + 1) for _ in range(MAX + 1)]
    out[0][0] = 1
    cur = [0] * (MAX + 1)
    cur[0] = 1
    for n in range(1, MAX + 1):
        # cur *= 1/(1 - q^n)
        for j in range(n, MAX + 1):
            cur[j] += cur[j - n]
        out[n] = list(cur)
    return out


def char_sum(k: int, s: int, MAX: int) -> list[int]:
    """Coefficients of chi_s^{(k)} via the Andrews-Gordon sum side.

    Slower than `char_product` for large MAX, but useful as an
    independent check.  Enumerates (n_1, ..., n_k) with bounded
    exponent.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not (1 <= s <= k + 1):
        raise ValueError(f"s must be in 1..{k+1}, got {s}")
    qpoch_inv = _qpoch_reciprocals(MAX)
    out = [0] * (MAX + 1)

    # Each n_i is bounded: N_i^2 <= exponent <= MAX implies N_i <= sqrt(MAX),
    # and N_1 >= N_2 >= ... >= N_k = n_k >= 0, so each n_i <= sqrt(MAX).
    import math
    max_n = int(math.isqrt(MAX)) + 1

    def recurse(j: int, ns: list[int], cur_prod: list[int]):
        if j == k:
            # Compute N_i and the exponent.
            N = [0] * (k + 2)
            for i in range(k, 0, -1):
                N[i] = N[i + 1] + ns[i - 1]
            exponent = sum(N[i] ** 2 for i in range(1, k + 1))
            for i in range(s, k + 1):
                exponent += N[i]
            if exponent > MAX:
                return
            for a in range(MAX + 1 - exponent):
                out[a + exponent] += cur_prod[a]
            return
        for ni in range(max_n + 1):
            # Quick pruning: the contribution to the exponent from later N_i's
            # already includes at least n_i^2 (since N_j contains n_i for j <= i).
            if ni * ni > MAX:
                break
            new_prod = [0] * (MAX + 1)
            inv = qpoch_inv[ni]
            for a in range(MAX + 1):
                if cur_prod[a] == 0:
                    continue
                ca = cur_prod[a]
                for b in range(MAX + 1 - a):
                    if inv[b]:
                        new_prod[a + b] += ca * inv[b]
            ns.append(ni)
            recurse(j + 1, ns, new_prod)
            ns.pop()

    init = [0] * (MAX + 1)
    init[0] = 1
    recurse(0, [], init)
    return out


def _format_series(coeffs: list[int], N: int = 16) -> str:
    out = []
    for n in range(min(N, len(coeffs))):
        c = coeffs[n]
        if c == 0:
            continue
        sgn = "+" if c > 0 else "-"
        ac = abs(c)
        if n == 0:
            tok = f"{ac}"
        elif n == 1:
            tok = "q" if ac == 1 else f"{ac}q"
        else:
            tok = f"q^{n}" if ac == 1 else f"{ac}q^{n}"
        if not out and sgn == "+":
            out.append(tok)
        else:
            out.append(f" {sgn} {tok}")
    return "".join(out) if out else "0"


def _demo() -> None:
    MAX = 20

    print("=" * 64)
    print("M(2,5)  (Lee-Yang, k=1):  two characters")
    print("=" * 64)
    for s in (1, 2):
        prod = char_product(1, s, MAX)
        summ = char_sum(1, s, MAX)
        agree = (prod == summ)
        label = "chi_1 = G (vacuum)" if s == 1 else "chi_2 = H (h=-1/5)"
        print(f"  {label}")
        print(f"    product: {_format_series(prod)} ...")
        print(f"    sum:     {_format_series(summ)} ...")
        print(f"    agree: {agree}")
        print()

    print("=" * 64)
    print("M(2,7)  (k=2):  three characters")
    print("=" * 64)
    for s in (1, 2, 3):
        prod = char_product(2, s, MAX)
        summ = char_sum(2, s, MAX)
        agree = (prod == summ)
        print(f"  chi_{s}^{{(2,7)}}:")
        print(f"    product: {_format_series(prod)} ...")
        print(f"    sum:     {_format_series(summ)} ...")
        print(f"    agree: {agree}")
        print()

    print("=" * 64)
    print("M(2,9)  (k=3):  four characters")
    print("=" * 64)
    for s in (1, 2, 3, 4):
        prod = char_product(3, s, MAX)
        print(f"  chi_{s}^{{(2,9)}}:  {_format_series(prod)} ...")


if __name__ == "__main__":
    _demo()
