"""LaurentPoly — elements of Z[q, q^{-1}].

Canonical-surface migration of `quantum_torus.LaurentPoly` (Plan 07
Stage A1).  Self-contained: no imports from the preliminary stack.
This is the universal coefficient ring for K-theoretic Coulomb
branch algebras.

A `LaurentPoly` is a finite formal sum `Σ_n c_n q^n` with
`c_n ∈ Z`, stored as a sparse `dict[int, int]` of non-zero
coefficients.
"""

from __future__ import annotations
from typing import Union


class LaurentPoly:
    """An element of Z[q, q^{-1}]: a finite sum of c_n * q^n with c_n in Z."""

    __slots__ = ("_coeffs",)

    def __init__(self, coeffs: dict[int, int] | None = None):
        # Map from exponent -> coefficient, dropping zeros
        self._coeffs: dict[int, int] = {}
        if coeffs:
            for exp, c in coeffs.items():
                if c != 0:
                    self._coeffs[int(exp)] = int(c)

    @classmethod
    def _from_clean_dict(cls, d: dict[int, int]) -> LaurentPoly:
        """Fast-path constructor.  Caller MUST ensure:
          - d is a dict[int, int]
          - no zero coefficients

        Skips the iteration / int() casting in __init__.  Used in hot
        inner arithmetic loops where these invariants are guaranteed."""
        out = cls.__new__(cls)
        out._coeffs = d
        return out

    # --- Constructors ---

    @staticmethod
    def zero() -> LaurentPoly:
        return LaurentPoly()

    @staticmethod
    def one() -> LaurentPoly:
        return LaurentPoly({0: 1})

    @staticmethod
    def q(n: int = 1) -> LaurentPoly:
        """Return q^n as a Laurent polynomial."""
        return LaurentPoly({n: 1})

    @staticmethod
    def from_int(c: int) -> LaurentPoly:
        """Return the constant polynomial c."""
        return LaurentPoly({0: c})

    # --- Arithmetic ---

    def is_zero(self) -> bool:
        return len(self._coeffs) == 0

    def __neg__(self) -> LaurentPoly:
        # All -c are nonzero iff all c were nonzero (which is the invariant).
        return LaurentPoly._from_clean_dict(
            {e: -c for e, c in self._coeffs.items()})

    def __add__(self, other: Union[LaurentPoly, int]) -> LaurentPoly:
        if isinstance(other, int):
            if other == 0:
                return self
            other = LaurentPoly.from_int(other)
        # Fast-path: zero side.
        if not other._coeffs:
            return self
        if not self._coeffs:
            return other
        # Fast-path: monomial other side (most common case).
        if len(other._coeffs) == 1:
            e, c = next(iter(other._coeffs.items()))
            result = dict(self._coeffs)
            v = result.get(e, 0) + c
            if v == 0:
                if e in result:
                    del result[e]
            else:
                result[e] = v
            return LaurentPoly._from_clean_dict(result)
        result: dict[int, int] = dict(self._coeffs)
        for e, c in other._coeffs.items():
            v = result.get(e, 0) + c
            if v == 0:
                del result[e]
            else:
                result[e] = v
        return LaurentPoly._from_clean_dict(result)

    def __radd__(self, other: int) -> LaurentPoly:
        return self.__add__(other)

    def __sub__(self, other: Union[LaurentPoly, int]) -> LaurentPoly:
        if isinstance(other, int):
            other = LaurentPoly.from_int(other)
        return self + (-other)

    def __rsub__(self, other: int) -> LaurentPoly:
        return LaurentPoly.from_int(other) - self

    def __mul__(self, other: Union[LaurentPoly, int]) -> LaurentPoly:
        if isinstance(other, int):
            other = LaurentPoly.from_int(other)
        if not isinstance(other, LaurentPoly):
            return NotImplemented
        a = self._coeffs
        b = other._coeffs
        la = len(a)
        lb = len(b)
        # Fast-path: monomial * polynomial (single coefficient on one side).
        if la == 1:
            e1, c1 = next(iter(a.items()))
            return LaurentPoly._from_clean_dict(
                {e1 + e2: c1 * c2 for e2, c2 in b.items()})
        if lb == 1:
            e2, c2 = next(iter(b.items()))
            return LaurentPoly._from_clean_dict(
                {e1 + e2: c1 * c2 for e1, c1 in a.items()})
        # General polynomial multiplication.
        result: dict[int, int] = {}
        for e1, c1 in a.items():
            for e2, c2 in b.items():
                e = e1 + e2
                result[e] = result.get(e, 0) + c1 * c2
        # Drop zero coefficients that arose from cancellation.
        if any(v == 0 for v in result.values()):
            result = {e: c for e, c in result.items() if c != 0}
        return LaurentPoly._from_clean_dict(result)

    def __rmul__(self, other: int) -> LaurentPoly:
        if isinstance(other, int):
            return self.__mul__(other)
        return NotImplemented

    def __pow__(self, n: int) -> LaurentPoly:
        if n < 0:
            raise ValueError("Negative powers not supported for general Laurent polynomials")
        if n == 0:
            return LaurentPoly.one()
        # Binary exponentiation: O(log n) multiplications.
        result = LaurentPoly.one()
        base = self
        while n > 0:
            if n & 1:
                result = result * base
            n >>= 1
            if n:
                base = base * base
        return result

    def exact_div(self, other: Union[LaurentPoly, int]) -> LaurentPoly:
        """Exact division in Z[q, q^{-1}].

        Returns the unique Laurent polynomial r with ``self == other * r``.
        Raises ``ZeroDivisionError`` if ``other`` is zero and ``ValueError``
        if the division is not exact in Z[q, q^{-1}].
        """
        if isinstance(other, int):
            other = LaurentPoly.from_int(other)
        if not isinstance(other, LaurentPoly):
            return NotImplemented
        if other.is_zero():
            raise ZeroDivisionError("exact_div by zero LaurentPoly")
        if self.is_zero():
            return LaurentPoly.zero()
        num: dict[int, int] = dict(self._coeffs)
        den = other._coeffs
        den_min = min(den)
        den_max = max(den)
        den_lc = den[den_min]
        num_min = min(num)
        num_max = max(num)
        q_min = num_min - den_min
        q_max = num_max - den_max
        if q_max < q_min:
            raise ValueError("division not exact in Z[q,q^-1]: degree mismatch")
        out: dict[int, int] = {}
        for k in range(q_min, q_max + 1):
            c = num.get(k + den_min, 0)
            if c == 0:
                continue
            if c % den_lc != 0:
                raise ValueError(
                    f"division not exact in Z[q,q^-1]: {c} / {den_lc}"
                )
            qk = c // den_lc
            out[k] = qk
            for j, bj in den.items():
                key = k + j
                nv = num.get(key, 0) - bj * qk
                if nv == 0:
                    num.pop(key, None)
                else:
                    num[key] = nv
        if any(v != 0 for v in num.values()):
            raise ValueError("division not exact in Z[q,q^-1]: nonzero remainder")
        return LaurentPoly(out)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            other = LaurentPoly.from_int(other)
        if not isinstance(other, LaurentPoly):
            return NotImplemented
        return self._coeffs == other._coeffs

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._coeffs.items())))

    def __repr__(self) -> str:
        if not self._coeffs:
            return "0"
        parts = []
        for e in sorted(self._coeffs):
            c = self._coeffs[e]
            if c == 0:
                continue
            if e == 0:
                parts.append(str(c))
            elif e == 1:
                if c == 1:
                    parts.append("q")
                elif c == -1:
                    parts.append("-q")
                else:
                    parts.append(f"{c}*q")
            else:
                if c == 1:
                    parts.append(f"q^{e}")
                elif c == -1:
                    parts.append(f"-q^{e}")
                else:
                    parts.append(f"{c}*q^{e}")
        if not parts:
            return "0"
        s = parts[0]
        for p in parts[1:]:
            if p.startswith("-"):
                s += " - " + p[1:]
            else:
                s += " + " + p
        return s


# Convenient aliases
q = LaurentPoly.q()
q_inv = LaurentPoly.q(-1)
class QuantumTorus:
    """
    An element of the quantum torus algebra T_q over Z[q, q^{-1}].

    Internally stored as a dict mapping (a, b) -> LaurentPoly coefficient.

    Note: the canonical `A_𝖖[T]` quantum torus is `QuantumTorusKAlg`
    (`quantum_torus_kalgebra.py`); this rank-2 `QuantumTorus` is a separate,
    lower-level Laurent-arithmetic helper type kept alongside it.
    """

    __slots__ = ("_terms",)

    def __init__(self, terms: dict[tuple[int, int], LaurentPoly] | None = None):
        self._terms: dict[tuple[int, int], LaurentPoly] = {}
        if terms:
            for idx, coeff in terms.items():
                idx = (int(idx[0]), int(idx[1]))
                if not coeff.is_zero():
                    self._terms[idx] = coeff

    # --- Constructors ---

    @staticmethod
    def zero() -> QuantumTorus:
        return QuantumTorus()

    @staticmethod
    def x(a: int, b: int) -> QuantumTorus:
        """Return the basis element x_{a,b}."""
        return QuantumTorus({(a, b): LaurentPoly.one()})

    @staticmethod
    def from_laurent(f: LaurentPoly) -> QuantumTorus:
        """Embed a Laurent polynomial as f * x_{0,0}."""
        return QuantumTorus({(0, 0): f})

    @staticmethod
    def from_int(c: int) -> QuantumTorus:
        """Embed an integer as c * x_{0,0}."""
        return QuantumTorus.from_laurent(LaurentPoly.from_int(c))

    # --- Accessors ---

    def is_zero(self) -> bool:
        return len(self._terms) == 0

    def coeff(self, a: int, b: int) -> LaurentPoly:
        """Return the Laurent polynomial coefficient of x_{a,b}."""
        return self._terms.get((a, b), LaurentPoly.zero())

    # --- Arithmetic ---

    def __neg__(self) -> QuantumTorus:
        return QuantumTorus({idx: -c for idx, c in self._terms.items()})

    def __add__(self, other: Union[QuantumTorus, int]) -> QuantumTorus:
        if isinstance(other, int):
            other = QuantumTorus.from_int(other)
        result: dict[tuple[int, int], LaurentPoly] = dict(self._terms)
        for idx, c in other._terms.items():
            if idx in result:
                result[idx] = result[idx] + c
            else:
                result[idx] = c
        return QuantumTorus(result)

    def __radd__(self, other: int) -> QuantumTorus:
        return self.__add__(other)

    def __sub__(self, other: Union[QuantumTorus, int]) -> QuantumTorus:
        if isinstance(other, int):
            other = QuantumTorus.from_int(other)
        return self + (-other)

    def __rsub__(self, other: int) -> QuantumTorus:
        return QuantumTorus.from_int(other) - self

    def __mul__(self, other: Union[QuantumTorus, LaurentPoly, int]) -> QuantumTorus:
        # Scalar multiplication by LaurentPoly or int
        if isinstance(other, int):
            other = LaurentPoly.from_int(other)
        if isinstance(other, LaurentPoly):
            return QuantumTorus({idx: c * other for idx, c in self._terms.items()})
        # Algebra multiplication
        if not isinstance(other, QuantumTorus):
            return NotImplemented
        result: dict[tuple[int, int], LaurentPoly] = {}
        for (a, b), f in self._terms.items():
            for (c, d), g in other._terms.items():
                # x_{a,b} * x_{c,d} = q^{ad - bc} * x_{a+c, b+d}
                twist = a * d - b * c
                new_idx = (a + c, b + d)
                contribution = f * g * LaurentPoly.q(twist)
                if new_idx in result:
                    result[new_idx] = result[new_idx] + contribution
                else:
                    result[new_idx] = contribution
        return QuantumTorus(result)

    def __rmul__(self, other: Union[LaurentPoly, int]) -> QuantumTorus:
        if isinstance(other, int):
            other = LaurentPoly.from_int(other)
        if isinstance(other, LaurentPoly):
            return QuantumTorus({idx: other * c for idx, c in self._terms.items()})
        return NotImplemented

    def __pow__(self, n: int) -> QuantumTorus:
        if n < 0:
            raise ValueError("Negative powers not supported")
        if n == 0:
            return QuantumTorus.from_int(1)
        result = QuantumTorus.from_int(1)
        for _ in range(n):
            result = result * self
        return result

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            other = QuantumTorus.from_int(other)
        if not isinstance(other, QuantumTorus):
            return NotImplemented
        return self._terms == other._terms

    def __repr__(self) -> str:
        if not self._terms:
            return "0"
        parts = []
        for (a, b) in sorted(self._terms):
            c = self._terms[(a, b)]
            if c.is_zero():
                continue
            c_str = repr(c)
            if a == 0 and b == 0:
                parts.append(c_str)
            elif c == LaurentPoly.one():
                parts.append(f"x({a},{b})")
            elif c == -LaurentPoly.one():
                parts.append(f"-x({a},{b})")
            else:
                if "+" in c_str or ("- " in c_str):
                    parts.append(f"({c_str})*x({a},{b})")
                else:
                    parts.append(f"{c_str}*x({a},{b})")

        if not parts:
            return "0"
        s = parts[0]
        for p in parts[1:]:
            if p.startswith("-"):
                s += " - " + p[1:]
            else:
                s += " + " + p
        return s


# --- Convenience for interactive use ---

def x(a: int, b: int) -> QuantumTorus:
    """Shorthand to create the basis element x_{a,b}."""
    return QuantumTorus.x(a, b)
