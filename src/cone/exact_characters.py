"""
exact_characters.py
===================

Computable, **arbitrary-q** closed-form Schur indices / VOA vacuum characters,
harvested from the literature.  Pure Python, no repo dependencies — each routine
returns plain dicts so the module is reusable standalone (the repo bridges wrap
the output into `RPowerSeries` over the appropriate `ZPlusRing`).

`SU(2)` characters are returned in the **highest-weight** convention: an irrep
is keyed by its highest weight `2j ∈ {0,1,2,…}` (matching `SU2ZPlusRing`), so
`ch[ρ_m]` (the `m`-dimensional irrep) has key `m-1`.

Currently implemented
---------------------
* `deven_gauged_Tr1_qn` — the U(1)-gauged `[A1, D_{2k+2}]` vacuum index `Tr(1)`
  (the `x⁰` slice of a known closed form for the `(A1, D_{2p})` index).
  Cross-verified against the repo's INDEPENDENT `sl(3)₋₃/₂` Kac–Wakimoto vacuum
  character (theta sums, no cone/BPS) at every order to `q_d²⁸`; this exposed a
  defect in the cone oracle from `q_d²⁴` on.
* `creutzig_Aodd_Tr1_z` — `(A1, A_{2p-3})` ↔ `B_p` vacuum index (a known
  closed form from standard results on admissible-level sl(2) characters),
  as a `z`-Laurent q-series.
"""
from __future__ import annotations


# --------------------------------------------------------------------------
# z-Laurent × q-series helpers  (dicts: {q_power: {z_power: int}})
# --------------------------------------------------------------------------
def _sym_laurent_to_su2(zd):
    """Decompose a *symmetric* (`c_w = c_{-w}`) Laurent polynomial `{z_power:int}`
    into `SU(2)` characters `{highest_weight: multiplicity}`.

    For `P(z) = Σ_j a_j χ_j` with `χ_j = z^j+z^{j-2}+…+z^{-j}`, the coefficient of
    `z^w` (`w ≥ 0`) is `c_w = Σ_{j≥w, j≡w (2)} a_j`, hence `a_w = c_w - c_{w+2}`.
    """
    if not zd:
        return {}
    maxw = max(zd)
    irr = {}
    for w in range(0, maxw + 1):          # ALL lattice points 0..maxw, incl. c_w=0
        mult = zd.get(w, 0) - zd.get(w + 2, 0)
        if mult:
            irr[w] = mult
    return irr


def _ch_rho(m):
    """`ch[ρ_m](z)` = character of the `m`-dimensional `sl2` irrep, as
    `{z_power: int}`:  exponents `m-1, m-3, …, -(m-1)`."""
    return {(m - 1) - 2 * j: 1 for j in range(m)}


# --------------------------------------------------------------------------
# (A1, D_{2p}) — the deven character  (a known closed form)
# --------------------------------------------------------------------------
def _deven_zz_product(MQ):
    """`P[Qpow] = {zpow:int}` :  ∏_{n≥1} (1-z² Qⁿ)⁻¹ (1-z⁻² Qⁿ)⁻¹  up to `Q^MQ`."""
    P = {0: {0: 1}}
    for n in range(1, MQ + 1):
        for sgn in (+2, -2):                       # the two geometric factors
            newP = {}
            for qp, zd in P.items():
                a = 0
                while qp + n * a <= MQ:
                    tq = qp + n * a
                    shift = sgn * a
                    tgt = newP.setdefault(tq, {})
                    for zp, c in zd.items():
                        tgt[zp + shift] = tgt.get(zp + shift, 0) + c
                    a += 1
            P = newP
    return P


def deven_gauged_xn_qn(k, n, K):
    """U(1)-gauged `[A1, D_{2k+2}]` v-tower trace `Tr(X_{0,1}ⁿ)`, **arbitrary q**.

    The `[xⁿ]` slice of the known `(A1, D_{2p})` index closed form times the
    `U(1)` vector
    factor `(q;q)_∞²` (`p = k+1`).  `Tr(X01ⁿ) = (Q;Q)²·[xⁿ]ℐ`, and since
    `[xⁿ]f̃ᵖ_{ρ_m}` picks the weight `h=n` of `ρ_m` (present iff `m ≥ |n|+1` and
    `m ≡ n+1 mod 2`):

        Tr(X01ⁿ)(z;Q) = Σ_{m ≥ |n|+1, m ≡ n+1 (2)}  Q^{p(m²-1-n²)/4} · ch[ρ_m](z)
                        · ∏_{n'≥1} (1 - z² Q^{n'})⁻¹ (1 - z⁻² Q^{n'})⁻¹

    `n=0` reduces to `Tr(1)` (odd `m`).  `ℐ` is `x↦x⁻¹` symmetric so the sign of
    `n` is immaterial.  Returns `{q_d_power: {su2_highest_weight: int}}`.  The
    prefactor's q_d-power is `p(m²-1-n²)/2` — always an integer (`m≡n+1 (2)` ⇒
    `m²-1-n²` even) but **odd** when `p` is odd and `n` is odd (even `k`,
    odd v-power): the gauge sector then carries genuine odd-q_d terms (for `k=1`,
    `p=2`, every term is even-q_d, which is why D4 sees only even powers).
    """
    n = abs(n)
    p = k + 1
    MQ = K // 2                                   # P in Q-powers; Q = q_d²
    P = _deven_zz_product(MQ)
    acc = {}                                       # {q_d power: {zpow:int}}
    m = n + 1                                      # smallest m with weight n
    # The m-prefactor q_d-exponent is `p(m²-1-n²)/2` (NOT 2·⌊p(m²-1-n²)/4⌋: for
    # odd p — i.e. even k — and n≠0 the numerator is ≢0 mod 4, so the old floor
    # truncated a half-integer Q-power = an *odd* q_d-power, silently mangling the
    # whole v-tower at k≥2; k=1 has even p=2 so 2·(even) is always ÷4, which is
    # why it was latent).  `m ≡ n+1 (2)` ⇒ `m²-1-n²` is even, so `p(m²-1-n²)/2` is
    # an exact integer.
    # Sign: importing the closed-form character means choosing the branch
    # `q_paper^{1/2} = -𝖖`.  The m-prefactor `q_paper^{p(m²-1-n²)/4}`
    # is a *half-integer* q_paper power exactly when `sh = p(m²-1-n²)/2` is odd
    # (⟺ `n·p` odd ⟺ odd q_d), and a half-integer power picks up one factor of
    # `q_paper^{1/2} = -𝖖`, i.e. a `(-1)` on those (odd-q_d) terms.  P and ch carry
    # integer q_paper / no q, so no sign.  (`n=0`/even-k: never half-integer ⇒ no
    # sign ⇒ the bug was latent.)  Here `sh` has constant parity across the tower,
    # so the per-term `(-1)^{[odd q_d]}` is the uniform `(-1)^{n·p}`.
    sgn = -1 if (n % 2 and p % 2) else 1
    while p * (m * m - 1 - n * n) // 2 <= K:
        num = p * (m * m - 1 - n * n)
        assert num % 2 == 0, (k, n, m, num)
        sh = num // 2                              # q_d units (exact)
        chm = _ch_rho(m)
        for qp, zd in P.items():                   # qp in Q = q_d² units
            tq = sh + 2 * qp                        # q_d units
            if tq > K:
                continue
            tgt = acc.setdefault(tq, {})
            for zp, c in zd.items():
                for w, cm in chm.items():
                    tgt[zp + w] = tgt.get(zp + w, 0) + sgn * c * cm
        m += 2
    return {qd: _sym_laurent_to_su2(zd) for qd, zd in acc.items()}


def _bernoulli(n):
    """Bernoulli numbers `B_0 … B_n` (with `B_1 = -1/2`), as `Fraction`s."""
    from fractions import Fraction as Fr
    from math import comb
    B = [Fr(0)] * (n + 1)
    B[0] = Fr(1)
    for m in range(1, n + 1):
        B[m] = -sum(Fr(comb(m + 1, j)) * B[j] for j in range(m)) / (m + 1)
    return B


def qpochhammer(K):
    """`(q;q)_∞ = ∏_{n≥1}(1-qⁿ)` as `{q_power: int}` to order `q^K` (Euler)."""
    P = {0: 1}
    for n in range(1, K + 1):
        nP = {}
        for e, c in P.items():
            nP[e] = nP.get(e, 0) + c
            if e + n <= K:
                nP[e + n] = nP.get(e + n, 0) - c
        P = nP
    return P


def eisenstein_E2k(kk, K):
    """The standard (holomorphic) Eisenstein series `E_{2k}(q)` as
    `{q_power: Fraction}` to order `q^K`:
    `E_{2k} = 1 - (4k/B_{2k}) Σ_{n≥1} σ_{2k-1}(n) qⁿ`, `σ` the divisor power sum.
    (`E_2,E_4,E_6,…`; building block for the unflavoured `N=4` SU(N) indices and
    the genus indices.)"""
    from fractions import Fraction as Fr
    B = _bernoulli(2 * kk)
    pref = Fr(-4 * kk) / B[2 * kk]
    out = {0: Fr(1)}
    for n in range(1, K + 1):
        sig = sum(d ** (2 * kk - 1) for d in range(1, n + 1) if n % d == 0)
        out[n] = pref * sig
    return out


def deven_gauged_Tr1_qn(k, K):
    """U(1)-gauged `[A1, D_{2k+2}]` vacuum Schur index `Tr(1)` = `Tr(X01⁰)`
    (`deven_gauged_xn_qn(k, 0, K)`).  Closed form, **arbitrary q**; the `x⁰`
    slice of the `(A1, D_{2p})` index closed form (`p=k+1`):

        Tr(1)(z; Q) = Σ_{m odd ≥ 1}  Q^{p(m²-1)/4} · ch[ρ_m](z)
                      · ∏_{n≥1} (1 - z² Qⁿ)⁻¹ (1 - z⁻² Qⁿ)⁻¹
    """
    return deven_gauged_xn_qn(k, 0, K)


# --------------------------------------------------------------------------
# (A1, A_{2p-3}) — the B_p character  (a known closed form)
# --------------------------------------------------------------------------
def creutzig_Aodd_Tr1_z(p, K, zpow_max=None):
    """`(A1, A_{2p-3})` ↔ `B_p` vacuum Schur index, **arbitrary q**.

    Returns `q^{-c_p/24}·ℐ` as `{q_power: {z_power: int_or_fraction}}` to order
    `q^K`, with half-integer `q`-powers cleared by working in `q^{1/?}` — here we
    return integer `q`-powers only when `p | …`; otherwise powers are rationals.

    Formula (domain `|z^{±1} q^{(p-1)/2}| < 1`):

        q^{-c_p/24} ℐ = 1/η(q)² Σ_{n∈ℤ}
            [ q^{p(n+½-1/2p)²}/(1 - z q^{p(n+½-1/2p)})
            - q^{p(n+½+1/2p)²}/(1 - z q^{p(n+½+1/2p)}) ].

    `c_p = 2 - 6(p-1)²/p`.  This is a `z`-Laurent series; SU(2) char
    decomposition is *not* applied (the flavour here is `U(1)`, fugacity `z`).
    NOTE: returns rational q-powers as-is (Fraction keys) — see __main__ demo.
    """
    from fractions import Fraction as Fr
    # 1/eta(q)^2 = q^{-1/12} / (q;q)_inf^2 ;  we strip the q^{-1/12} (absorbed in
    # the q^{-c/24} bookkeeping) and return the (q;q)^{-2} part times the sum.
    # (q;q)_inf^{-2} = sum over partitions-into-2-colours; build to order K.
    inv_qq2 = _inv_eta_sq_series(K)               # {q_pow:int}, q-integer powers
    sum_terms = {}                                # {Fr q_pow: {z_pow:int}}
    # geometric: 1/(1 - z q^e) = sum_{j>=0} z^j q^{e j}  (need e>0 for convergence)
    n = 0
    # symmetric in n around the saddle; sweep until exponents exceed K
    nrange = range(-(K + 2), K + 3)
    for n in nrange:
        for sgn, sign in ((Fr(-1, 2), +1), (Fr(+1, 2), -1)):
            e0 = p * (Fr(n) + Fr(1, 2) + sgn / p) ** 2          # leading q-power
            estep = p * (Fr(n) + Fr(1, 2) + sgn / p)            # geometric step
            if e0 > K + 1:
                continue
            j = 0
            while e0 + estep * j <= K + 1 and (estep > 0 or j == 0):
                qp = e0 + estep * j
                if qp <= K + 1:
                    d = sum_terms.setdefault(qp, {})
                    d[j] = d.get(j, 0) + sign
                if estep <= 0:
                    break
                j += 1
    # convolve with inv_qq2 (integer q-powers)
    out = {}
    for qp_s, zd in sum_terms.items():
        for qp_i, ci in inv_qq2.items():
            qp = qp_s + qp_i
            if qp > K:
                continue
            tgt = out.setdefault(qp, {})
            for zp, c in zd.items():
                tgt[zp] = tgt.get(zp, 0) + c * ci
    return out


def _inv_eta_sq_series(K):
    """`(q;q)_∞^{-2} = ∏_{n≥1}(1-qⁿ)^{-2}` as `{q_power: int}` to order `q^K`."""
    P = {0: 1}
    for n in range(1, K + 1):
        for _ in range(2):                         # squared
            newP = {}
            for qp, c in P.items():
                a = 0
                while qp + n * a <= K:
                    t = qp + n * a
                    newP[t] = newP.get(t, 0) + c
                    a += 1
            P = newP
    return P


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("deven_gauged_xn_qn(k=1, n=0, K=10)  [= Tr(1)]:")
    t1 = deven_gauged_Tr1_qn(1, 10)
    for qp in sorted(t1):
        print(f"  q_d^{qp}: {t1[qp]}")
    print("deven v-tower deven_gauged_xn_qn(k=1, n=1, K=8)  [= Tr(X01)]:")
    tv = deven_gauged_xn_qn(1, 1, 8)
    for qp in sorted(tv):
        print(f"  q_d^{qp}: {tv[qp]}")
    print("building blocks:")
    print("  E_2:", [str(eisenstein_E2k(1, 4)[n]) for n in range(5)])
    print("  E_4:", [str(eisenstein_E2k(2, 3)[n]) for n in range(4)])
    print("  (q;q)_inf to q^6:", qpochhammer(6))
