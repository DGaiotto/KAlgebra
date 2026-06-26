"""`QmuLaurent` — q-Laurent polynomial with `RatFuncMu` (= Q(μ))
coefficients.

Carrier for the SU(2)+Nf=1 cyclicity trace solve.  Each matrix entry
in the cyclicity linear system is a finite-degree polynomial in q and
q^{-1} with coefficients in Q(μ) — exactly the class we need to
avoid the "multi-μ-monomial pivot" blocker that breaks the naïve
Z[μ, μ^{-1}] series-inverse.

Series inversion: for `D ∈ QmuLaurent` with a unique lowest q-exponent
`k0` and leading Q(μ) coefficient `c0` (invertible in Q(μ) as long as
`c0 ≠ 0`), `1/D = q^{-k0} · (1/c0) · 1/(1 + p)` where `p = (D − q^{k0}
· c0) / (q^{k0} · c0)`.  We expand `1/(1 + p) = Σ_n (-p)^n` truncated
at `q^{q_max}`.

Conversion from `RLaurent[AbelianZPlusRing(rank=1)]`:
`fromRLaurent(r)` walks `r.coeffs.items()` and packs each μ-monomial-
collection into a `RatFuncMu`.
"""
from __future__ import annotations

from fractions import Fraction

from ratfunc_mu import RatFuncMu


class QmuLaurent:
    """q-Laurent polynomial with `RatFuncMu` coefficients.

    Stored as `coeffs: dict[int, RatFuncMu]` (q-exponent → Q(μ) coef).
    Zero coefficients are pruned.
    """

    __slots__ = ('coeffs',)

    def __init__(self, coeffs: dict | None = None):
        if coeffs is None:
            self.coeffs = {}
            return
        self.coeffs = {e: c for e, c in coeffs.items() if not c.is_zero()}

    @staticmethod
    def zero() -> 'QmuLaurent':
        return QmuLaurent({})

    @staticmethod
    def one() -> 'QmuLaurent':
        return QmuLaurent({0: RatFuncMu.one()})

    @staticmethod
    def q(n: int) -> 'QmuLaurent':
        return QmuLaurent({n: RatFuncMu.one()})

    @staticmethod
    def from_rlaurent(r) -> 'QmuLaurent':
        """Convert `RLaurent[AbelianZPlusRing(rank=1)]` to `QmuLaurent`.

        Each `RElement` coefficient is a Z-linear combination of
        `basis_element((k,)) = μ^k`.  We pack it into a RatFuncMu by
        building the corresponding bare-polynomial dict.
        """
        out = {}
        for q_exp, r_elem in r.coeffs.items():
            # r_elem.terms maps μ-labels to Z-coefficients.
            mu_dict = {}
            min_shift = 0
            for label, coef in r_elem.terms.items():
                mu_pow = label[0] if isinstance(label, tuple) else label
                if mu_pow < min_shift:
                    min_shift = mu_pow
            for label, coef in r_elem.terms.items():
                mu_pow = label[0] if isinstance(label, tuple) else label
                shifted = mu_pow - min_shift
                mu_dict[shifted] = mu_dict.get(shifted, Fraction(0)) + Fraction(coef)
            mu_dict = {k: v for k, v in mu_dict.items() if v != 0}
            if not mu_dict:
                continue
            num = mu_dict
            # RatFuncMu(shift=min_shift, num=num, den={0: 1}).
            r_fm = RatFuncMu(min_shift, num, {0: Fraction(1)})
            out[q_exp] = r_fm
        return QmuLaurent(out)

    def is_zero(self) -> bool:
        return not self.coeffs

    def __neg__(self) -> 'QmuLaurent':
        return QmuLaurent({e: -c for e, c in self.coeffs.items()})

    def __add__(self, o: 'QmuLaurent') -> 'QmuLaurent':
        out = dict(self.coeffs)
        for e, c in o.coeffs.items():
            if e in out:
                s = out[e] + c
                if s.is_zero():
                    del out[e]
                else:
                    out[e] = s
            else:
                out[e] = c
        return QmuLaurent(out)

    def __sub__(self, o: 'QmuLaurent') -> 'QmuLaurent':
        return self + (-o)

    def __mul__(self, o) -> 'QmuLaurent':
        if isinstance(o, RatFuncMu):
            return QmuLaurent({e: c * o for e, c in self.coeffs.items()})
        if not isinstance(o, QmuLaurent):
            return NotImplemented
        out: dict = {}
        for e1, c1 in self.coeffs.items():
            for e2, c2 in o.coeffs.items():
                prod = c1 * c2
                if prod.is_zero():
                    continue
                key = e1 + e2
                if key in out:
                    s = out[key] + prod
                    if s.is_zero():
                        del out[key]
                    else:
                        out[key] = s
                else:
                    out[key] = prod
        return QmuLaurent(out)

    def truncate(self, q_max: int) -> 'QmuLaurent':
        return QmuLaurent({e: c for e, c in self.coeffs.items() if e <= q_max})

    def series_inverse(self, q_max: int) -> 'QmuLaurent':
        """1/self as truncated formal q-Laurent series with Q(μ) coefs."""
        if not self.coeffs:
            raise ZeroDivisionError
        k0 = min(self.coeffs.keys())
        c0 = self.coeffs[k0]
        inv_c0 = c0.inverse() if hasattr(c0, 'inverse') else _ratfunc_inv(c0)
        # p = (self / (q^k0 · c0)) − 1 = Σ_{e > k0} (c_e / c0) · q^{e − k0}.
        p_coeffs = {}
        for e, c in self.coeffs.items():
            if e == k0:
                continue
            p_coeffs[e - k0] = c * inv_c0
        p = QmuLaurent(p_coeffs)
        # 1/(1+p) = 1 − p + p^2 − ...
        inner_qmax = q_max + (-k0 if k0 < 0 else 0) + 4
        inv = QmuLaurent.one()
        term = QmuLaurent.one()
        sign = 1
        while True:
            term = (term * p).truncate(inner_qmax)
            if term.is_zero():
                break
            sign = -sign
            if sign > 0:
                inv = inv + term
            else:
                inv = inv - term
        # 1/self = q^{-k0} · (1/c0) · 1/(1+p).
        scaled = QmuLaurent({e - k0: c * inv_c0 for e, c in inv.coeffs.items()})
        return scaled.truncate(q_max)

    def __repr__(self):
        if not self.coeffs:
            return 'Qmu(0)'
        return 'Qmu(' + ', '.join(f'q^{e}: {c}' for e, c in
                                   sorted(self.coeffs.items())) + ')'


def _ratfunc_inv(r: RatFuncMu) -> RatFuncMu:
    """Multiplicative inverse of RatFuncMu (= swap num/den and re-normalise)."""
    if r.is_zero():
        raise ZeroDivisionError("inverse of zero RatFuncMu")
    # Swap num/den, shift sign flipped.  RatFuncMu's __init__ re-normalises.
    return RatFuncMu(-r.shift, r.den, r.num)
