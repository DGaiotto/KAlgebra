"""QNumberPoly --- integral linear combinations of quantum integers [n]_q.

The palindromic Laurent polynomials in Z[q,q^{-1}] (those P with
P(q) = P(q^{-1})) form a Z-module with basis {[n]_q : n >= 1}, where

    [n]_q = (q^n - q^{-n}) / (q - q^{-1}) = q^{-(n-1)} + q^{-(n-3)} + ... + q^{n-1}

(n terms, exponents from -(n-1) to (n-1) in steps of 2).

This module is also closed under multiplication: the SU(2) Clebsch--Gordan
identity gives

    [m]_q * [n]_q = sum_{i=0..min(m,n)-1} [|m - n| + 1 + 2i]_q,

so {[n]_q} spans a commutative ring (the representation ring of U_q(sl_2)).
This is exactly the structure of the F-coefficients in the BPS K_𝖖-algebra realisation:
each f_delta is palindromic, and palindromic stays palindromic under
addition and (untwisted) multiplication.

QNumberPoly stores P = sum_n c_n [n]_q as a dict {n: c_n} with n >= 1.
Operations are O(support * something cheap) so peeling a single [n]_q
contribution off, or scaling by an integer, are O(1).
"""

from __future__ import annotations

from typing import Mapping, Union

from laurent_poly import LaurentPoly


class QNumberPoly:
    """Element of Z[[n]_q : n >= 1] (palindromic Laurent polys, ring).

    Stored as a dict {n: c_n} representing sum_n c_n * [n]_q. Keys
    with c_n == 0 are dropped on construction.
    """

    __slots__ = ("_coeffs", "_laurent_cache")

    def __init__(self, coeffs: Mapping[int, int] | None = None):
        self._coeffs: dict[int, int] = {}
        self._laurent_cache: LaurentPoly | None = None
        if coeffs:
            for n, c in coeffs.items():
                if c == 0:
                    continue
                n = int(n)
                if n < 1:
                    raise ValueError(f"q-number index must be >= 1, got {n}")
                self._coeffs[n] = self._coeffs.get(n, 0) + int(c)
            # Drop zeros introduced by accumulation.
            self._coeffs = {n: c for n, c in self._coeffs.items() if c != 0}

    # ---------- Constructors ----------

    @staticmethod
    def zero() -> "QNumberPoly":
        return QNumberPoly()

    @staticmethod
    def one() -> "QNumberPoly":
        return QNumberPoly({1: 1})

    @staticmethod
    def from_int(c: int) -> "QNumberPoly":
        if c == 0:
            return QNumberPoly()
        return QNumberPoly({1: int(c)})

    @staticmethod
    def q_number(n: int, c: int = 1) -> "QNumberPoly":
        """Return c * [n]_q.  n must be >= 1."""
        if n < 1:
            raise ValueError(f"q-number index must be >= 1, got {n}")
        if c == 0:
            return QNumberPoly()
        return QNumberPoly({int(n): int(c)})

    @staticmethod
    def from_palindromic_laurent(p: LaurentPoly) -> "QNumberPoly":
        """Decompose a palindromic LaurentPoly into the q-number basis.

        A palindromic poly with nonzero exponents in {-M, ..., M} expands
        uniquely as sum_n c_n [n]_q.  Algorithm: peel from the outside
        in.  The largest exponent M with nonzero coefficient c means
        c * [M+1]_q contributes to the basis decomposition; subtract
        and recurse.  Raises ValueError if `p` is not palindromic.
        """
        if p.is_zero():
            return QNumberPoly()
        coeffs = dict(p._coeffs)
        # Verify palindromy (lightweight check; we expect inputs to be
        # palindromic by construction in the F-solver).
        for e, c in p._coeffs.items():
            if coeffs.get(-e, 0) != c:
                raise ValueError(
                    f"from_palindromic_laurent: input is not palindromic "
                    f"({e}: {c}, {-e}: {coeffs.get(-e, 0)})"
                )
        out: dict[int, int] = {}
        while coeffs:
            # Find largest exponent with nonzero coefficient.
            M = max(coeffs)
            c = coeffs[M]
            if c == 0:
                del coeffs[M]
                continue
            n = M + 1
            out[n] = out.get(n, 0) + c
            # Subtract c * [n]_q: contributes c at exponents M, M-2, ..., -M.
            e = M
            while e >= -M:
                coeffs[e] = coeffs.get(e, 0) - c
                if coeffs[e] == 0:
                    del coeffs[e]
                e -= 2
        return QNumberPoly(out)

    # ---------- Predicates / accessors ----------

    def is_zero(self) -> bool:
        return not self._coeffs

    def items(self):
        """Iterate (n, c_n) pairs in ascending n."""
        return sorted(self._coeffs.items())

    def to_laurent(self) -> LaurentPoly:
        """Expand to LaurentPoly: sum_n c_n * (q^{-(n-1)} + ... + q^{n-1}).

        Memoised on the QNumberPoly instance: a QNumberPoly's
        ``_coeffs`` are never mutated after construction, so the
        Laurent expansion is a pure function of identity.  Repeated
        F-cache reads (the warm path through
        ``BPSKAlgebra._decompose_in_F_basis`` and friends) hit the
        cached value instead of re-expanding [n]_q's every time.
        """
        if self._laurent_cache is not None:
            return self._laurent_cache
        out: dict[int, int] = {}
        for n, c in self._coeffs.items():
            # [n]_q has exponents -(n-1), -(n-3), ..., n-1, all with coeff 1.
            e = -(n - 1)
            while e <= n - 1:
                out[e] = out.get(e, 0) + c
                e += 2
        lp = LaurentPoly({e: v for e, v in out.items() if v != 0})
        self._laurent_cache = lp
        return lp

    # ---------- Arithmetic ----------

    def __neg__(self) -> "QNumberPoly":
        return QNumberPoly({n: -c for n, c in self._coeffs.items()})

    def __add__(self, other: Union["QNumberPoly", int]) -> "QNumberPoly":
        if isinstance(other, int):
            other = QNumberPoly.from_int(other)
        if not isinstance(other, QNumberPoly):
            return NotImplemented
        result = dict(self._coeffs)
        for n, c in other._coeffs.items():
            result[n] = result.get(n, 0) + c
        return QNumberPoly({n: c for n, c in result.items() if c != 0})

    def __radd__(self, other: int) -> "QNumberPoly":
        return self.__add__(other)

    def __sub__(self, other: Union["QNumberPoly", int]) -> "QNumberPoly":
        return self + (-other if isinstance(other, QNumberPoly) else -int(other))

    def __rsub__(self, other: int) -> "QNumberPoly":
        return (-self) + int(other)

    def __mul__(self, other: Union["QNumberPoly", int]) -> "QNumberPoly":
        if isinstance(other, int):
            if other == 0:
                return QNumberPoly()
            return QNumberPoly({n: c * other for n, c in self._coeffs.items()})
        if not isinstance(other, QNumberPoly):
            return NotImplemented
        # SU(2) Clebsch-Gordan: [m]*[n] = sum_{i=0..min(m,n)-1} [|m-n|+1+2i]
        out: dict[int, int] = {}
        for m, cm in self._coeffs.items():
            for n, cn in other._coeffs.items():
                w = cm * cn
                lo = abs(m - n) + 1
                hi = m + n - 1
                k = lo
                while k <= hi:
                    out[k] = out.get(k, 0) + w
                    k += 2
        return QNumberPoly({n: c for n, c in out.items() if c != 0})

    def __rmul__(self, other: int) -> "QNumberPoly":
        if isinstance(other, int):
            return self.__mul__(other)
        return NotImplemented

    def __pow__(self, n: int) -> "QNumberPoly":
        if n < 0:
            raise ValueError("QNumberPoly: negative powers not supported")
        if n == 0:
            return QNumberPoly.one()
        result = QNumberPoly.one()
        base = self
        while n > 0:
            if n & 1:
                result = result * base
            n >>= 1
            if n:
                base = base * base
        return result

    # ---------- Equality / hashing ----------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            other = QNumberPoly.from_int(other)
        if not isinstance(other, QNumberPoly):
            return NotImplemented
        return self._coeffs == other._coeffs

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._coeffs.items())))

    # ---------- Display ----------

    def __repr__(self) -> str:
        if not self._coeffs:
            return "0"
        parts = []
        for n in sorted(self._coeffs):
            c = self._coeffs[n]
            sign = "+" if c >= 0 else "-"
            mag = abs(c)
            if n == 1:
                term = "1" if mag == 1 else str(mag)
            else:
                term = f"[{n}]_q" if mag == 1 else f"{mag}*[{n}]_q"
            if not parts:
                parts.append(("-" if c < 0 else "") + term)
            else:
                parts.append(f" {sign} {term}")
        return "".join(parts)
