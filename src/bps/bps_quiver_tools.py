"""
bps_quiver_tools.py
===================

Self-contained computational toolkit for the K_𝖖-algebra  A_q  (the fusion
algebra of rotation-equivariant BPS line defects) of a 4d  N=2  theory
specified by a *decorated BPS quiver* Q.

A decorated BPS quiver is the data

    (Gamma, <,>, {gamma_i^Q})

where

  * Gamma is a finite-rank integer lattice,
  * <,> is an antisymmetric pairing on Gamma (the Dirac pairing),
  * gamma_i^Q in Gamma are the charges attached to the nodes of Q.

The arrows of Q encode the pairings  B_{ij} = <gamma_i^Q, gamma_j^Q>  in
the usual way (positive = outgoing arrows, negative = incoming).

Implements the BPS-quiver routines underlying the K_𝖖-algebra realisation.

Zero external dependencies beyond  fractions.Fraction  and  collections.

What this module exports
------------------------

* ``LaurentPoly``              -- elements of Z[q, q^{-1}]
* ``Lattice``                  -- integer lattice with antisymmetric pairing
* ``qt_multiply``              -- product in the quantum torus Q_Gamma
* ``rho_Q``                    -- canonical automorphism  X_gamma -> X_{-gamma}

* ``q_binomial`` / ``q_integer``   -- [n,k]_q and [n]_q over Z[q,q^{-1}]
* ``solve`` / ``solve_inverse``    -- conjugation by E_q(X_gamma):
                                     push an element across one E_q factor
* ``can_solve`` / ``can_solve_inverse`` -- cheap finiteness checks
* ``complete_to_solvable``         -- minimal correction to make solve finite
* ``packet_decomposition``         -- diagnose gamma-line structure

* ``sigma`` / ``sigma_inverse``    -- tropical sigma along a spectrum
* ``solve_F``                      -- principal canonical-basis F_gamma solver

* ``BPSQuiver``                    -- quiver, tropical mutation, S-finder
* ``find_spectrum_generator``      -- convenience wrapper around BPSQuiver
* ``commute_F_across``             -- F^{(i)}_a from commuting F_a through
                                     the first i factors of S
* ``CoulombAlgebra``               -- end-to-end driver: F-basis, products,
                                     F-decomposition (structure constants),
                                     Schur indices

* ``PowerSeries`` / ``qpoch_finite`` / ``qpoch_infty`` / ``inv_qpoch_finite``
* ``schur_index_nahm``             -- exact-Nahm Schur index I_{a,b}(q)

* ``PRESETS``                      -- a handful of pre-registered theories

Run  ``python bps_quiver_tools.py``  for a short demo on the Pentagon theory.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from collections import defaultdict, deque
from functools import lru_cache
from itertools import product as _iproduct
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Union

Vec = tuple[int, ...]

__all__ = [
    # ring / lattice
    "LaurentPoly", "Lattice", "Vec",
    "qt_multiply", "rho_Q",
    # q-combinatorics
    "q_binomial", "q_integer",
    # E_q commutation
    "solve", "solve_inverse",
    "can_solve", "can_solve_inverse",
    "complete_to_solvable", "complete_to_inverse_solvable",
    "packet_decomposition",
    "bezout_cofactor",
    # tropical / canonical basis
    "sigma", "sigma_inverse",
    "solve_F",
    # quiver and spectrum
    "BPSQuiver", "find_spectrum_generator",
    # F^{(i)} and algebra driver
    "commute_F_across", "CoulombAlgebra",
    # Schur index helpers
    "PowerSeries", "qpoch_finite", "qpoch_infty", "inv_qpoch_finite",
    "schur_index_nahm",
    # presets
    "PRESETS",
]


# =====================================================================
# SECTION 1 -- LaurentPoly (the coefficient ring Z[q, q^{-1}])
# =====================================================================
class LaurentPoly:
    """An element of  Z[q, q^{-1}]  : a finite sum  sum_n c_n q^n  with c_n in Z.

    Stored as a dict ``{exponent: coefficient}`` with zero entries dropped.
    Supports the full ring structure plus ``exact_div`` (division when the
    result lives in Z[q, q^{-1}]).  This is the coefficient ring of the
    quantum torus Q_Gamma.
    """

    __slots__ = ("_coeffs",)

    def __init__(self, coeffs: dict[int, int] | None = None):
        self._coeffs: dict[int, int] = {}
        if coeffs:
            for exp, c in coeffs.items():
                if c != 0:
                    self._coeffs[int(exp)] = int(c)

    # --- constructors -------------------------------------------------
    @staticmethod
    def zero() -> "LaurentPoly":
        return LaurentPoly()

    @staticmethod
    def one() -> "LaurentPoly":
        return LaurentPoly({0: 1})

    @staticmethod
    def q(n: int = 1) -> "LaurentPoly":
        """Return  q^n  as a Laurent polynomial."""
        return LaurentPoly({n: 1})

    @staticmethod
    def from_int(c: int) -> "LaurentPoly":
        return LaurentPoly({0: c})

    # --- queries ------------------------------------------------------
    def is_zero(self) -> bool:
        return len(self._coeffs) == 0

    # --- arithmetic ---------------------------------------------------
    def __neg__(self) -> "LaurentPoly":
        return LaurentPoly({e: -c for e, c in self._coeffs.items()})

    def __add__(self, other: Union["LaurentPoly", int]) -> "LaurentPoly":
        if isinstance(other, int):
            other = LaurentPoly.from_int(other)
        result: dict[int, int] = dict(self._coeffs)
        for e, c in other._coeffs.items():
            result[e] = result.get(e, 0) + c
        return LaurentPoly(result)

    def __radd__(self, other: int) -> "LaurentPoly":
        return self.__add__(other)

    def __sub__(self, other: Union["LaurentPoly", int]) -> "LaurentPoly":
        if isinstance(other, int):
            other = LaurentPoly.from_int(other)
        return self + (-other)

    def __rsub__(self, other: int) -> "LaurentPoly":
        return LaurentPoly.from_int(other) - self

    def __mul__(self, other: Union["LaurentPoly", int]) -> "LaurentPoly":
        if isinstance(other, int):
            other = LaurentPoly.from_int(other)
        if not isinstance(other, LaurentPoly):
            return NotImplemented
        result: dict[int, int] = {}
        for e1, c1 in self._coeffs.items():
            for e2, c2 in other._coeffs.items():
                e = e1 + e2
                result[e] = result.get(e, 0) + c1 * c2
        return LaurentPoly(result)

    def __rmul__(self, other: int) -> "LaurentPoly":
        if isinstance(other, int):
            return self.__mul__(other)
        return NotImplemented

    def __pow__(self, n: int) -> "LaurentPoly":
        if n < 0:
            raise ValueError("negative powers not supported for general LaurentPoly")
        # Binary exponentiation: O(log n) multiplications.
        r = LaurentPoly.one()
        base = self
        while n > 0:
            if n & 1:
                r = r * base
            n >>= 1
            if n:
                base = base * base
        return r

    def exact_div(self, other: Union["LaurentPoly", int]) -> "LaurentPoly":
        """Return the unique r with  self == other * r  in Z[q, q^{-1}].

        Raises ZeroDivisionError / ValueError if no such r exists.
        """
        if isinstance(other, int):
            other = LaurentPoly.from_int(other)
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
                raise ValueError(f"division not exact in Z[q,q^-1]: {c} / {den_lc}")
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

    # --- equality / hashing / repr ------------------------------------
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
                parts.append("q" if c == 1 else ("-q" if c == -1 else f"{c}*q"))
            else:
                if c == 1:
                    parts.append(f"q^{e}")
                elif c == -1:
                    parts.append(f"-q^{e}")
                else:
                    parts.append(f"{c}*q^{e}")
        s = parts[0]
        for p in parts[1:]:
            s += " - " + p[1:] if p.startswith("-") else " + " + p
        return s


# =====================================================================
# SECTION 1B -- HabiroElement (Z[q,q^{-1}] localised at all 1-q^{2k})
# =====================================================================
#
# Z[q,q^{-1}][1/(1-q^{2k}) : k>=1] is the home of Nahm sums.  Elements
# are stored as (numerator: LaurentPoly, denom: {k: m_k}) meaning
# numerator / prod_k (1-q^{2k})^{m_k}.  The (1-q^{2k})-multiset basis
# is used (not cyclotomic) because divisibility tests become a pure
# coefficient fold.  Keeping Schur-index intermediates as exact
# HabiroElements removes the K_internal-truncation failure mode that
# the old PowerSeries-throughout pipeline had for deep F_a.
#
# See `habiro.py` for the standalone version with full tests.

# ---------------------------------------------------------------------------
# Low-level helpers: divisibility / division by (1 - q^{2k})
# ---------------------------------------------------------------------------

def _fold_is_zero(P: LaurentPoly, k: int) -> bool:
    """Return True iff (1 - q^{2k}) divides P in Z[q, q^{-1}].

    Equivalent to: sum of P's coefficients in each residue class mod 2k is zero.
    """
    if P.is_zero():
        return True
    twok = 2 * k
    fold: dict[int, int] = {}
    for exp, c in P._coeffs.items():
        r = exp % twok
        fold[r] = fold.get(r, 0) + c
    return all(v == 0 for v in fold.values())


def _try_divide_1mq2k(P: LaurentPoly, k: int) -> LaurentPoly | None:
    """Divide P by (1 - q^{2k}) in Z[q, q^{-1}].  Return None if not exact.

    Uses the forward recurrence q_j = p_j + q_{j-2k}, which is what you get
    from P = (1 - q^{2k}) * Q expanded term-by-term.  The walk consumes each
    p_j once; the tail (j beyond the quotient's support) must vanish, which
    is the same as the fold condition.
    """
    if P.is_zero():
        return LaurentPoly.zero()
    twok = 2 * k
    p = P._coeffs
    j_min = min(p)
    j_max = max(p)
    # Quotient has support in [j_min, j_max - 2k].  For j > j_max - 2k the
    # recurrence must produce zero.
    q_coeffs: dict[int, int] = {}
    hi = j_max - twok
    for j in range(j_min, j_max + 1):
        val = p.get(j, 0) + q_coeffs.get(j - twok, 0)
        if j <= hi:
            if val != 0:
                q_coeffs[j] = val
        else:
            if val != 0:
                return None
    return LaurentPoly(q_coeffs)


def _one_minus_q2k(k: int) -> LaurentPoly:
    return LaurentPoly({0: 1, 2 * k: -1})


# ---------------------------------------------------------------------------
# HabiroElement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HabiroElement:
    """Element of Z[q, q^{-1}][ 1/(1-q^{2k}) : k >= 1 ].

    Stored as ``numerator / prod_k (1 - q^{2k})^{denom[k]}``.

    Invariant after `simplify()`: for every ``k`` with ``denom[k] > 0``,
    ``(1 - q^{2k})`` does not divide ``numerator``.

    Construct via the classmethods `zero`, `one`, `from_int`, `from_laurent`,
    `pochhammer_inverse`, `nahm_term`; or pass `(numerator, denom)` directly
    and call `.simplify()`.
    """

    numerator: LaurentPoly
    denom: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise denom: drop zero multiplicities, freeze as a plain dict.
        clean = {int(k): int(m) for k, m in self.denom.items() if m}
        for k, m in clean.items():
            if k <= 0:
                raise ValueError(f"denom index must be >= 1, got {k}")
            if m < 0:
                raise ValueError(f"denom multiplicity must be >= 0, got {m} at k={k}")
        # Bypass frozen-dataclass in __setattr__:
        object.__setattr__(self, "denom", clean)

    # ---------- Constructors ----------

    @staticmethod
    def zero() -> "HabiroElement":
        return HabiroElement(LaurentPoly.zero(), {})

    @staticmethod
    def one() -> "HabiroElement":
        return HabiroElement(LaurentPoly.one(), {})

    @staticmethod
    def from_int(c: int) -> "HabiroElement":
        return HabiroElement(LaurentPoly.from_int(c), {})

    @staticmethod
    def from_laurent(p: LaurentPoly) -> "HabiroElement":
        return HabiroElement(p, {})

    @staticmethod
    def q_power(n: int, c: int = 1) -> "HabiroElement":
        """Return c * q^n."""
        return HabiroElement(LaurentPoly({n: c}), {})

    @staticmethod
    def pochhammer_inverse(n: int) -> "HabiroElement":
        """Return 1 / (q^2; q^2)_n = 1 / prod_{k=1..n} (1 - q^{2k})."""
        if n < 0:
            raise ValueError("pochhammer_inverse requires n >= 0")
        return HabiroElement(LaurentPoly.one(), {k: 1 for k in range(1, n + 1)})

    @staticmethod
    def sum(elements) -> "HabiroElement":
        """Sum an iterable of HabiroElements with a single simplify at the end.

        Equivalent to `reduce(operator.add, elements, HabiroElement.zero())`
        but ~n times faster: puts every summand over the common denominator
        in one pass, sums the scaled numerators, and simplifies once -- vs.
        per-addition simplify which repeats O(|denom|) fold tests on each
        intermediate sum.

        Use this whenever you know you are summing many same-ish-shape
        elements (e.g. Nahm-sum assembly, or accumulating [F S|0>]_eta
        contributions across delta in F).
        """
        elts = [e for e in elements if not e.is_zero()]
        if not elts:
            return HabiroElement.zero()
        # Common denominator (max multiplicity per k).
        common: dict[int, int] = {}
        for e in elts:
            for k, m in e.denom.items():
                if m > common.get(k, 0):
                    common[k] = m
        # Sum scaled numerators.
        num = LaurentPoly.zero()
        for e in elts:
            num = num + e._scale_numerator_to_denom(common)
        return HabiroElement(num, common).simplify()

    @staticmethod
    def nahm_term(c: int, shift: int, ns: list[int]) -> "HabiroElement":
        """Return c * q^shift / prod_i (q^2; q^2)_{n_i}.

        Each factor `(q^2; q^2)_{n_i}` contributes one `(1-q^{2k})` for
        every `k in 1..n_i`, so `denom[k] = #{i : n_i >= k}`.
        """
        num = LaurentPoly({shift: c}) if c else LaurentPoly.zero()
        denom: dict[int, int] = {}
        for n in ns:
            if n < 0:
                raise ValueError(f"nahm_term: n_i must be >= 0, got {n}")
            for k in range(1, n + 1):
                denom[k] = denom.get(k, 0) + 1
        return HabiroElement(num, denom)

    # ---------- Predicates / accessors ----------

    def is_zero(self) -> bool:
        return self.numerator.is_zero()

    def is_polynomial(self) -> bool:
        """True iff the element is a Laurent polynomial (no denominator)."""
        return not self.denom

    def k_min(self) -> int | None:
        """Lowest exponent of the Laurent expansion (== lowest exp of numerator,
        since every denominator factor has constant term 1).  None if zero."""
        if self.numerator.is_zero():
            return None
        return min(self.numerator._coeffs)

    def leading_term(self) -> tuple[int, int] | None:
        """Return (k_min, coefficient) of the Laurent expansion, or None if zero."""
        k = self.k_min()
        if k is None:
            return None
        return (k, self.numerator._coeffs[k])

    def has_positive_q_order(self) -> bool:
        """True iff the Laurent expansion lies in q * Z[[q]] (positive q-order)."""
        k = self.k_min()
        return k is not None and k >= 1

    # ---------- Canonical form ----------

    def simplify(self) -> "HabiroElement":
        """Strip common (1-q^{2k}) factors between numerator and denominator.

        Returns a new HabiroElement with the invariant that no `(1-q^{2k})`
        in `denom` divides the numerator.
        """
        num = self.numerator
        denom = dict(self.denom)
        # Process by k.  A single division lowers m_k by one; retest until no
        # further cancellation.  Order doesn't affect the fixed point: division
        # by (1-q^{2k1}) and (1-q^{2k2}) commute since they are both factors of
        # the numerator.
        changed = True
        while changed:
            changed = False
            for k in list(denom.keys()):
                if denom[k] <= 0:
                    del denom[k]
                    continue
                q = _try_divide_1mq2k(num, k)
                if q is None:
                    continue
                num = q
                denom[k] -= 1
                if denom[k] == 0:
                    del denom[k]
                changed = True
        # Zero numerator -> drop denom entirely.
        if num.is_zero():
            denom = {}
        return HabiroElement(num, denom)

    # ---------- Arithmetic ----------

    def __neg__(self) -> "HabiroElement":
        return HabiroElement(-self.numerator, dict(self.denom))

    def _scale_numerator_to_denom(self, target: Mapping[int, int]) -> LaurentPoly:
        """Multiply self.numerator by the factors of `target` missing from self.denom.

        Used to put two elements over a common denominator that dominates both.
        Precondition: target[k] >= self.denom.get(k, 0) for every k.
        """
        num = self.numerator
        for k, m_target in target.items():
            extra = m_target - self.denom.get(k, 0)
            if extra > 0:
                # (1 - q^{2k})^extra via binary exp on the factor.
                num = num * (_one_minus_q2k(k) ** extra)
        return num

    def __add__(self, other: Union["HabiroElement", LaurentPoly, int]) -> "HabiroElement":
        if isinstance(other, int):
            other = HabiroElement.from_int(other)
        elif isinstance(other, LaurentPoly):
            other = HabiroElement.from_laurent(other)
        if not isinstance(other, HabiroElement):
            return NotImplemented
        keys = set(self.denom) | set(other.denom)
        common = {k: max(self.denom.get(k, 0), other.denom.get(k, 0)) for k in keys}
        a = self._scale_numerator_to_denom(common)
        b = other._scale_numerator_to_denom(common)
        return HabiroElement(a + b, common).simplify()

    def __radd__(self, other: Union[LaurentPoly, int]) -> "HabiroElement":
        return self.__add__(other)

    def __sub__(self, other: Union["HabiroElement", LaurentPoly, int]) -> "HabiroElement":
        if isinstance(other, int):
            other = HabiroElement.from_int(other)
        elif isinstance(other, LaurentPoly):
            other = HabiroElement.from_laurent(other)
        if not isinstance(other, HabiroElement):
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other: Union[LaurentPoly, int]) -> "HabiroElement":
        if isinstance(other, int):
            other = HabiroElement.from_int(other)
        elif isinstance(other, LaurentPoly):
            other = HabiroElement.from_laurent(other)
        else:
            return NotImplemented
        return other - self

    def __mul__(
        self, other: Union["HabiroElement", LaurentPoly, int]
    ) -> "HabiroElement":
        if isinstance(other, int):
            return HabiroElement(self.numerator * other, dict(self.denom)).simplify()
        if isinstance(other, LaurentPoly):
            return HabiroElement(self.numerator * other, dict(self.denom)).simplify()
        if not isinstance(other, HabiroElement):
            return NotImplemented
        new_denom: dict[int, int] = {}
        for k, m in self.denom.items():
            new_denom[k] = new_denom.get(k, 0) + m
        for k, m in other.denom.items():
            new_denom[k] = new_denom.get(k, 0) + m
        return HabiroElement(self.numerator * other.numerator, new_denom).simplify()

    def __rmul__(self, other: Union[LaurentPoly, int]) -> "HabiroElement":
        return self.__mul__(other)

    def __pow__(self, n: int) -> "HabiroElement":
        if n < 0:
            raise ValueError("Negative powers not supported")
        if n == 0:
            return HabiroElement.one()
        result = HabiroElement.one()
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
            other = HabiroElement.from_int(other)
        elif isinstance(other, LaurentPoly):
            other = HabiroElement.from_laurent(other)
        if not isinstance(other, HabiroElement):
            return NotImplemented
        # Cross-multiply: a/d1 == b/d2  iff  a*d2 == b*d1 in Z[q, q^{-1}].
        # Robust regardless of the (non-canonical) simplification state.
        a_times_d2 = self.numerator * other._denom_as_poly()
        b_times_d1 = other.numerator * self._denom_as_poly()
        return a_times_d2 == b_times_d1

    # `simplify()` is not canonical in the `(1-q^{2k})`-basis (equal elements can
    # have different simplified forms, e.g. 1/(1-q^2) vs (1+q^2)/(1-q^4)).
    # We therefore mark the class unhashable rather than risk a hash/eq mismatch.
    # If you need to key dicts by HabiroElements, either convert to a canonical
    # form yourself (e.g. via minimum cyclotomic form) or key by `expand(K)` for
    # a sufficiently large K.
    __hash__ = None  # type: ignore[assignment]

    # ---------- Expansion to a truncated power series ----------

    def _denom_as_poly(self) -> LaurentPoly:
        D = LaurentPoly.one()
        for k, m in self.denom.items():
            f = _one_minus_q2k(k)
            for _ in range(m):
                D = D * f
        return D

    def expand(self, K: int) -> LaurentPoly:
        """Laurent expansion truncated to exponents <= K.

        The denominator has constant term 1, so the expansion is a Laurent
        series with lowest exponent equal to k_min(numerator).  The walk is
        the standard recurrence for series division, O((K - k_min) * |denom poly|).
        """
        N = self.numerator
        if N.is_zero():
            return LaurentPoly.zero()
        D = self._denom_as_poly()
        d = D._coeffs  # d_0 == 1; other nonzero entries at positive exponents
        n = N._coeffs
        j_min = min(n)
        out: dict[int, int] = {}
        for j in range(j_min, K + 1):
            val = n.get(j, 0)
            for i, d_i in d.items():
                if i <= 0:
                    continue
                prev = out.get(j - i, 0)
                if prev:
                    val -= d_i * prev
            if val:
                out[j] = val
        return LaurentPoly(out)

    def coefficient(self, j: int) -> int:
        """The coefficient of q^j in the Laurent expansion.

        Uses the series-division recurrence; cheaper than a full `expand(j)` if
        j is close to k_min, more expensive otherwise (it still walks from k_min).
        """
        N = self.numerator
        if N.is_zero():
            return 0
        D = self._denom_as_poly()
        d = D._coeffs
        n = N._coeffs
        j_min = min(n)
        if j < j_min:
            return 0
        cache: dict[int, int] = {}
        for jj in range(j_min, j + 1):
            val = n.get(jj, 0)
            for i, d_i in d.items():
                if i <= 0:
                    continue
                prev = cache.get(jj - i, 0)
                if prev:
                    val -= d_i * prev
            if val:
                cache[jj] = val
        return cache.get(j, 0)

    # ---------- Display ----------

    def __repr__(self) -> str:
        num = repr(self.numerator)
        if not self.denom:
            return num
        factors = []
        for k in sorted(self.denom):
            m = self.denom[k]
            if m == 1:
                factors.append(f"(1-q^{2*k})")
            else:
                factors.append(f"(1-q^{2*k})^{m}")
        den = " * ".join(factors)
        if " " in num and not (num.startswith("(") and num.endswith(")")):
            num = f"({num})"
        return f"{num} / ({den})"


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def qpoch_inverse(n: int) -> HabiroElement:
    """Return 1 / (q^2; q^2)_n.  Convenience alias for HabiroElement.pochhammer_inverse."""
    return HabiroElement.pochhammer_inverse(n)


def nahm(c: int, shift: int, ns: list[int]) -> HabiroElement:
    """Return c * q^shift / prod_i (q^2; q^2)_{n_i}.  Alias for HabiroElement.nahm_term."""
    return HabiroElement.nahm_term(c, shift, ns)



# =====================================================================
# SECTION 2 -- Lattice, quantum-torus product, rho_Q automorphism
# =====================================================================
class Lattice:
    """A finite-rank integer lattice Gamma with a (typically antisymmetric)
    bilinear pairing  <gamma, gamma'> = sum_{ij} gamma_i B_{ij} gamma'_j.

    The pairing matrix  B  is stored as a tuple of tuples of ints.  For a
    BPS quiver the rows of B are exactly the exchange-matrix rows.
    """

    __slots__ = ("rank", "pairing")

    def __init__(self, pairing: Sequence[Sequence[int]]):
        rows = [tuple(int(x) for x in row) for row in pairing]
        n = len(rows)
        for row in rows:
            if len(row) != n:
                raise ValueError("pairing matrix must be square")
        self.rank: int = n
        self.pairing: tuple[tuple[int, ...], ...] = tuple(rows)

    # --- convenience constructors ------------------------------------
    @staticmethod
    def symplectic(r: int) -> "Lattice":
        """Standard rank-2r symplectic lattice with block  [[0,1],[-1,0]]."""
        n = 2 * r
        P = [[0] * n for _ in range(n)]
        for k in range(r):
            P[2 * k][2 * k + 1] = 1
            P[2 * k + 1][2 * k] = -1
        return Lattice(P)

    @staticmethod
    def from_quiver(B: Sequence[Sequence[int]]) -> "Lattice":
        """Build the ambient lattice directly from a quiver exchange matrix.

        The node charges are then the standard basis vectors  e_i ,
        and  <e_i, e_j> = B_{ij}  by construction.
        """
        return Lattice(B)

    # --- core pairing -------------------------------------------------
    def bracket(self, a: Sequence[int], b: Sequence[int]) -> int:
        """Compute  <a, b>  =  sum_{ij} a_i B_{ij} b_j ."""
        B = self.pairing
        s = 0
        for i, ai in enumerate(a):
            if ai == 0:
                continue
            for j, bj in enumerate(b):
                if bj:
                    s += ai * B[i][j] * bj
        return s

    def check(self, gamma) -> Vec:
        """Normalise a charge to a length-``rank`` tuple of ints."""
        g = tuple(int(x) for x in gamma)
        if len(g) != self.rank:
            raise ValueError(f"expected length {self.rank}, got {len(g)}")
        return g

    def __repr__(self) -> str:
        rows = ", ".join("[" + ",".join(str(x) for x in row) + "]"
                          for row in self.pairing)
        return f"Lattice(rank={self.rank}, B=[{rows}])"


# ---------------------------------------------------------------------
# Quantum torus  Q_Gamma  and canonical automorphism rho_Q
# ---------------------------------------------------------------------

# Convention for the quantum torus:
#
#     X_gamma * X_{gamma'}  =  q^{<gamma, gamma'>}  X_{gamma + gamma'}
#
# with scalars in Z[q, q^{-1}] (our ``LaurentPoly``).  A general element
# is represented as a ``dict[Vec, LaurentPoly]`` -- a finite formal sum
# of monomials.  All computations below are exact (no q-expansions).

def qt_multiply(
    A: dict[Vec, LaurentPoly],
    B: dict[Vec, LaurentPoly],
    lattice: Lattice,
) -> dict[Vec, LaurentPoly]:
    """Return the quantum-torus product  A * B  in  Q_Gamma.

    The factor of  q^{<a, b>}  from the twist is absorbed into the
    LaurentPoly coefficient at the combined charge  a + b.
    """
    result: dict[Vec, dict[int, int]] = {}
    for g1, c1 in A.items():
        for g2, c2 in B.items():
            twist = lattice.bracket(g1, g2)
            ng = tuple(a + b for a, b in zip(g1, g2))
            bucket = result.setdefault(ng, {})
            for e1, v1 in c1._coeffs.items():
                for e2, v2 in c2._coeffs.items():
                    e = e1 + e2 + twist
                    bucket[e] = bucket.get(e, 0) + v1 * v2
    out: dict[Vec, LaurentPoly] = {}
    for g, c in result.items():
        lp = LaurentPoly(c)
        if not lp.is_zero():
            out[g] = lp
    return out


def rho_Q(A: dict[Vec, LaurentPoly]) -> dict[Vec, LaurentPoly]:
    """Canonical automorphism of Q_Gamma:   X_gamma  |->  X_{-gamma} .

    Used e.g. in the intertwining identity
    ``F_a * S  =  S * rho_Q(F_{sigma(a)})``.
    Acts as the identity on LaurentPoly coefficients.
    """
    return {tuple(-x for x in g): c for g, c in A.items() if not c.is_zero()}


# Alias sometimes used in the code below.
qt_mul = qt_multiply


# =====================================================================
# SECTION 3 -- q-binomials and E_q(X_gamma) commutation on Q_Gamma
# =====================================================================
# ---------------------------------------------------------------------
# q-numbers and q-binomials, symmetric convention.
# ---------------------------------------------------------------------
#
#   [n]_q  = q^{n-1} + q^{n-3} + ... + q^{-(n-1)}
#   [n,k]_q = q^k [n-1,k]_q + q^{-(n-k)} [n-1,k-1]_q
#
# These are bar-invariant (palindromic) elements of Z[q, q^{-1}] and form
# the natural basis for canonical-basis coefficients.

@lru_cache(maxsize=None)
def q_binomial(n: int, k: int) -> LaurentPoly:
    """Symmetric q-binomial  [n, k]_q  in  Z[q, q^{-1}]."""
    if k < 0 or k > n or n < 0:
        return LaurentPoly.zero()
    if k == 0 or k == n:
        return LaurentPoly.one()
    return (LaurentPoly.q(k) * q_binomial(n - 1, k)
            + LaurentPoly.q(-(n - k)) * q_binomial(n - 1, k - 1))


def q_integer(n: int) -> LaurentPoly:
    """Symmetric q-integer  [n]_q  =  q^{n-1} + ... + q^{-(n-1)} ."""
    if n == 0:
        return LaurentPoly.zero()
    return q_binomial(n, 1)


# ---------------------------------------------------------------------
# Commutation past E_q(X_gamma) :    O * E_q(X_gamma) = E_q(X_gamma) * O'
# ---------------------------------------------------------------------
#
# With <gamma, gamma> = 0, the conjugation problem decouples along the
# gamma-lines of the support of O.  A gamma-line is an orbit of
# alpha -> alpha + gamma inside Gamma; pick an auxiliary Bezout vector u
# with u . gamma = 1 and parametrise
#
#     alpha = beta + k * gamma,        k = u . alpha,   u . beta = 0.
#
# Since <gamma, gamma> = 0 the pairing  m := <gamma, alpha>  is constant
# along a line and equals  <gamma, beta> .  Letting z be a formal shift
# variable along the line, the coefficient polynomial  C(z) = sum_k c_k z^k
# transforms as
#
#     m == 0 :  C'(z) = C(z)                           (pass through)
#     m <  0 :  C'(z) = C(z) * B_{|m|}(z)              (expand, always ok)
#     m >  0 :  must have  B_m(z)  |  C(z), then
#               C'(z) = C(z) / B_m(z)                  (contract)
#
# with  B_n(z) = sum_{k=0}^n [n,k]_q z^k .  ``solve_inverse`` swaps the
# roles of m > 0 and m < 0.
#
# Everything below (``solve``, ``solve_inverse``, ``can_solve`` and the
# completion helpers) is a thin wrapper around this 1d bookkeeping.

def _xgcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended GCD: returns  (g, s, t)  with  g = s*a + t*b ,  g >= 0."""
    if a == 0:
        s = 1 if b >= 0 else -1
        return (abs(b), 0, s)
    g, s, t = _xgcd(b % a, a)
    return (g, t - (b // a) * s, s)


def bezout_cofactor(gamma: Sequence[int]) -> tuple[int, ...]:
    """Return an integer vector  u  with  sum_i u_i * gamma_i = 1.

    Such a u exists iff  gamma  is primitive (gcd of components = 1).
    It is what lets us parametrise every alpha in Z^n as
    ``alpha = beta + k * gamma``  with  ``u . beta = 0`` .
    """
    g = 0
    n = len(gamma)
    u = [0] * n
    for i, gi in enumerate(gamma):
        if gi == 0:
            continue
        if g == 0:
            g = abs(gi)
            u[i] = 1 if gi > 0 else -1
        else:
            ng, s, t = _xgcd(g, gi)
            u = [s * x for x in u]
            u[i] += t
            g = ng
        if g == 1:
            break
    if g != 1:
        raise ValueError(f"gamma={tuple(gamma)} is not primitive (gcd={g})")
    assert sum(ui * gi for ui, gi in zip(u, gamma)) == 1
    return tuple(u)


# --- 1d polynomial-ring helpers over LaurentPoly coefficients ---------

def _bpoly(m: int) -> dict[int, LaurentPoly]:
    """The q-binomial packet  B_m(z) = sum_{k=0}^m [m,k]_q  z^k ."""
    return {k: q_binomial(m, k) for k in range(m + 1)}


def _conv(A: dict[int, LaurentPoly], B: dict[int, LaurentPoly]
          ) -> dict[int, LaurentPoly]:
    """Polynomial multiplication in z (with LaurentPoly coefficients)."""
    out: dict[int, LaurentPoly] = {}
    for i, ai in A.items():
        for j, bj in B.items():
            k = i + j
            out[k] = out.get(k, LaurentPoly.zero()) + ai * bj
    return {k: v for k, v in out.items() if not v.is_zero()}


def _divmod_desc(C: dict[int, LaurentPoly], B: dict[int, LaurentPoly]
                 ) -> tuple[dict[int, LaurentPoly], dict[int, LaurentPoly]]:
    """Descending long division of  C(z)  by  B(z)  (B monic).

    Returns ``(Q, R)`` with  C = B * Q + R  and ``deg(R) < deg(B)``.
    Handles Laurent polynomials in z by shifting first.
    """
    C = {k: v for k, v in C.items() if not v.is_zero()}
    if not C:
        return {}, {}
    sh = min(C)
    Cs = {k - sh: v for k, v in C.items()}
    dB = max(B)
    Q: dict[int, LaurentPoly] = {}
    while True:
        Cs = {k: v for k, v in Cs.items() if not v.is_zero()}
        if not Cs:
            break
        dC = max(Cs)
        if dC < dB:
            break
        lc = Cs[dC]
        s = dC - dB
        Q[s] = lc
        for j, bj in B.items():
            Cs[j + s] = Cs.get(j + s, LaurentPoly.zero()) - lc * bj
    Cs = {k: v for k, v in Cs.items() if not v.is_zero()}
    return ({k + sh: v for k, v in Q.items()},
            {k + sh: v for k, v in Cs.items()})


def _ascend(C: dict[int, LaurentPoly], B: dict[int, LaurentPoly], kmax: int
            ) -> tuple[dict[int, LaurentPoly], dict[int, LaurentPoly]]:
    """Ascending division up to ``kmax``.

    Produces  Q  with  ``C + G = B * Q`` , G supported strictly above kmax.
    Used to *complete* a line so it becomes a full B_m-packet.
    """
    C = dict(C)
    Q: dict[int, LaurentPoly] = {}
    if not C:
        return {}, {}
    kmin = min(C)
    nsteps = int(kmax - kmin)
    keys = [kmin + i for i in range(nsteps + 1)]
    for k in keys:
        C = {j: v for j, v in C.items() if not v.is_zero()}
        dk = C.get(k, LaurentPoly.zero())
        if dk.is_zero():
            continue
        Q[k] = dk
        for j, bj in B.items():
            C[k + j] = C.get(k + j, LaurentPoly.zero()) - dk * bj
    C = {k: v for k, v in C.items() if not v.is_zero()}
    G = {k: -v for k, v in C.items()}
    return Q, {k: v for k, v in G.items() if not v.is_zero()}


# --- decomposition of an element into gamma-lines --------------------

def _require_isotropic(lattice: Lattice, gamma: Vec) -> None:
    s = lattice.bracket(gamma, gamma)
    if s != 0:
        raise ValueError(
            f"conjugation by E_q(X_gamma) requires <gamma, gamma> = 0; "
            f"got <{gamma},{gamma}> = {s}"
        )


def _prepare(lattice: Lattice, gamma: Sequence[int]
             ) -> tuple[Vec, tuple[int, ...]]:
    g = lattice.check(gamma)
    _require_isotropic(lattice, g)
    u = bezout_cofactor(g)
    return g, u


def _decompose(element: dict[Vec, LaurentPoly], lattice: Lattice,
               gamma: Vec, u: tuple[int, ...]
               ) -> tuple[dict[Vec, dict[int, LaurentPoly]], dict[Vec, int]]:
    """Split ``element`` into gamma-lines keyed by the orthogonal rep beta.

    Returns
    -------
    lines     : {beta : {k : LaurentPoly}}     -- C(z) per line
    m_of_line : {beta : m = <gamma, beta>}     -- constant along the line
    """
    lines: dict[Vec, dict[int, LaurentPoly]] = defaultdict(dict)
    m_of_line: dict[Vec, int] = {}
    for alpha, coeff in element.items():
        if coeff.is_zero():
            continue
        k = sum(ui * ai for ui, ai in zip(u, alpha))
        beta = tuple(ai - k * gi for ai, gi in zip(alpha, gamma))
        lines[beta][k] = coeff
        if beta not in m_of_line:
            m_of_line[beta] = lattice.bracket(gamma, beta)
    return dict(lines), m_of_line


def _recover(beta: Vec, k: int, gamma: Vec) -> Vec:
    return tuple(bi + k * gi for bi, gi in zip(beta, gamma))


def _add_terms(acc: dict[Vec, LaurentPoly],
               ch: Vec, c: LaurentPoly) -> None:
    if c.is_zero():
        return
    if ch in acc:
        s = acc[ch] + c
        if s.is_zero():
            del acc[ch]
        else:
            acc[ch] = s
    else:
        acc[ch] = c


# ---------------------------------------------------------------------
# Public API for E_q commutation
# ---------------------------------------------------------------------

def solve(element: dict[Vec, LaurentPoly], lattice: Lattice,
          gamma: Sequence[int]) -> dict[Vec, LaurentPoly]:
    """Push ``element`` to the right across  E_q(X_gamma) :

        element * E_q(X_gamma)  =  E_q(X_gamma) * result.

    Raises ``ValueError`` if no finite ``result`` exists (i.e. some
    gamma-line with m > 0 is not already a B_m-packet).
    """
    g, u = _prepare(lattice, gamma)
    if not element:
        return {}
    lines, m_of = _decompose(element, lattice, g, u)
    acc: dict[Vec, LaurentPoly] = {}
    for beta, offs in lines.items():
        m = m_of[beta]
        if m == 0:
            for k, c in offs.items():
                _add_terms(acc, _recover(beta, k, g), c)
        elif m < 0:
            img = _conv(offs, _bpoly(-m))
            for k, c in img.items():
                _add_terms(acc, _recover(beta, k, g), c)
        else:  # m > 0 : require divisibility by B_m(z)
            Q, R = _divmod_desc(offs, _bpoly(m))
            if any(not v.is_zero() for v in R.values()):
                raise ValueError(
                    f"not finitely conjugable: gamma-line at beta={beta} "
                    f"has <gamma, beta>={m} > 0 and is not a q-binomial packet"
                )
            for k, c in Q.items():
                _add_terms(acc, _recover(beta, k, g), c)
    return acc


def solve_inverse(element: dict[Vec, LaurentPoly], lattice: Lattice,
                  gamma: Sequence[int]) -> dict[Vec, LaurentPoly]:
    """Push ``element`` to the LEFT across  E_q(X_gamma) :

        result * E_q(X_gamma)  =  E_q(X_gamma) * element,

    equivalently  result = E_q(X_gamma) * element * E_q(X_gamma)^{-1} .
    """
    g, u = _prepare(lattice, gamma)
    if not element:
        return {}
    lines, m_of = _decompose(element, lattice, g, u)
    acc: dict[Vec, LaurentPoly] = {}
    for beta, offs in lines.items():
        m = m_of[beta]
        if m == 0:
            for k, c in offs.items():
                _add_terms(acc, _recover(beta, k, g), c)
        elif m > 0:  # now expand
            img = _conv(offs, _bpoly(m))
            for k, c in img.items():
                _add_terms(acc, _recover(beta, k, g), c)
        else:  # m < 0 : require divisibility by B_{|m|}
            Q, R = _divmod_desc(offs, _bpoly(-m))
            if any(not v.is_zero() for v in R.values()):
                raise ValueError(
                    f"not inverse-conjugable: gamma-line at beta={beta} "
                    f"has <gamma, beta>={m} < 0 and is not a q-binomial packet"
                )
            for k, c in Q.items():
                _add_terms(acc, _recover(beta, k, g), c)
    return acc


def can_solve(element: dict[Vec, LaurentPoly], lattice: Lattice,
              gamma: Sequence[int]) -> bool:
    """Return True iff ``solve`` would succeed (cheap check)."""
    g, u = _prepare(lattice, gamma)
    if not element:
        return True
    lines, m_of = _decompose(element, lattice, g, u)
    for beta, offs in lines.items():
        m = m_of[beta]
        if m > 0:
            _, R = _divmod_desc(offs, _bpoly(m))
            if any(not v.is_zero() for v in R.values()):
                return False
    return True


def can_solve_inverse(element: dict[Vec, LaurentPoly], lattice: Lattice,
                      gamma: Sequence[int]) -> bool:
    """Return True iff ``solve_inverse`` would succeed."""
    g, u = _prepare(lattice, gamma)
    if not element:
        return True
    lines, m_of = _decompose(element, lattice, g, u)
    for beta, offs in lines.items():
        m = m_of[beta]
        if m < 0:
            _, R = _divmod_desc(offs, _bpoly(-m))
            if any(not v.is_zero() for v in R.values()):
                return False
    return True


def complete_to_solvable(
    element: dict[Vec, LaurentPoly],
    lattice: Lattice,
    gamma: Sequence[int],
) -> tuple[dict[Vec, LaurentPoly], dict[Vec, LaurentPoly]]:
    """Minimal correction making ``solve`` finite.

    Every gamma-line with m > 0 that is not already a B_m-packet is
    completed by ascending division: new monomials are added *above*
    the current top offset to turn the line into a multiple of B_m(z).
    Lines with m <= 0 are untouched.

    Returns ``(completed, correction)``  with  ``completed = element + correction``
    and  ``can_solve(completed, ...) = True`` .
    """
    g, u = _prepare(lattice, gamma)
    if not element:
        return {}, {}
    lines, m_of = _decompose(element, lattice, g, u)
    corr: dict[Vec, LaurentPoly] = {}
    for beta, offs in lines.items():
        m = m_of[beta]
        if m <= 0:
            continue
        Bm = _bpoly(m)
        kmax = max(offs)
        _, G = _ascend(offs, Bm, kmax)
        for k, c in G.items():
            _add_terms(corr, _recover(beta, k, g), c)
    completed = dict(element)
    for ch, c in corr.items():
        _add_terms(completed, ch, c)
    return completed, corr


def complete_to_inverse_solvable(
    element: dict[Vec, LaurentPoly],
    lattice: Lattice,
    gamma: Sequence[int],
) -> tuple[dict[Vec, LaurentPoly], dict[Vec, LaurentPoly]]:
    """Mirror of ``complete_to_solvable`` for  solve_inverse  (lines m < 0)."""
    g, u = _prepare(lattice, gamma)
    if not element:
        return {}, {}
    lines, m_of = _decompose(element, lattice, g, u)
    corr: dict[Vec, LaurentPoly] = {}
    for beta, offs in lines.items():
        m = m_of[beta]
        if m >= 0:
            continue
        Bm = _bpoly(-m)
        kmax = max(offs)
        _, G = _ascend(offs, Bm, kmax)
        for k, c in G.items():
            _add_terms(corr, _recover(beta, k, g), c)
    completed = dict(element)
    for ch, c in corr.items():
        _add_terms(completed, ch, c)
    return completed, corr


def packet_decomposition(
    element: dict[Vec, LaurentPoly],
    lattice: Lattice,
    gamma: Sequence[int],
) -> list[dict]:
    """Diagnose the gamma-line structure of ``element``.

    For each nonempty gamma-line the returned list contains a dict with

        'line'      : the orthogonal representative beta
        'm'         : <gamma, beta>  (constant along the line)
        'type'      : one of {'pass', 'expand', 'packet', 'obstruction'}
        'offsets'   : raw  C(z)  along the line
        'quotient'  : (packet only)      C(z) / B_m(z)
        'remainder' : (obstruction only) residue modulo  B_m(z)

    The element is finitely forward-conjugable iff no entry has type
    'obstruction'.
    """
    g, u = _prepare(lattice, gamma)
    if not element:
        return []
    lines, m_of = _decompose(element, lattice, g, u)
    report: list[dict] = []
    for beta, offs in lines.items():
        m = m_of[beta]
        entry: dict = {"line": beta, "m": m, "offsets": dict(offs)}
        if m == 0:
            entry["type"] = "pass"
        elif m < 0:
            entry["type"] = "expand"
        else:
            Q, R = _divmod_desc(offs, _bpoly(m))
            R = {k: v for k, v in R.items() if not v.is_zero()}
            if R:
                entry["type"] = "obstruction"
                entry["remainder"] = R
            else:
                entry["type"] = "packet"
                entry["quotient"] = Q
        report.append(entry)
    return report


# =====================================================================
# SECTION 4 -- tropical sigma, Nahm data, solve_F (canonical basis)
# =====================================================================
# --- §4a : tropical sigma, cone tools -------------------------------
# Tropical mutation along a single spectrum charge g:
#
#     mu_g^t(a)  =  a + max(<a, g>, 0) * g
#
# and the full tropical sigma along a spectrum  [g_1, ..., g_N]  :
#
#     sigma(gamma)  =  - ( mu_{g_N}^t o ... o mu_{g_1}^t )(gamma).
#
# The doubly-tropical interval  [gamma, -sigma^{-1}(gamma)]  (intersected
# with the positive cone) is the support of the canonical basis element
# F_gamma.  ``_support_bfs`` enumerates it in BFS order from ``gamma``.

def sigma(lattice: Lattice, spec: Sequence[Sequence[int]],
          gamma: Sequence[int]) -> Vec:
    """Apply the tropical sigma of the spectrum ``spec`` to ``gamma``."""
    c = lattice.check(gamma)
    spec_t = [lattice.check(g) for g in spec]
    for gk in spec_t:
        m = lattice.bracket(c, gk)
        if m > 0:
            c = tuple(a + m * gi for a, gi in zip(c, gk))
    return tuple(-x for x in c)


def sigma_inverse(lattice: Lattice, spec: Sequence[Sequence[int]],
                  gamma: Sequence[int]) -> Vec:
    """Inverse of ``sigma``: negate first, then undo tropical mutations."""
    c: Vec = tuple(-x for x in lattice.check(gamma))
    spec_t = [lattice.check(g) for g in spec]
    for gk in reversed(spec_t):
        m = lattice.bracket(c, gk)
        if m > 0:
            c = tuple(a - m * gi for a, gi in zip(c, gk))
    return c


# Internal helper: is  v  a non-negative integer combination of ``gens`` ?
# Uses rational Gaussian elimination + enumeration over free variables.

def _cone_contains(v: Vec, gens: list[Vec]) -> bool:
    n = len(gens)
    if n == 0:
        return all(x == 0 for x in v)
    r = len(v)
    A = [[Fraction(gens[j][i]) for j in range(n)] + [Fraction(v[i])]
         for i in range(r)]
    pivots: list[int] = []
    row = 0
    for col in range(n):
        piv = None
        for rr in range(row, r):
            if A[rr][col] != 0:
                piv = rr
                break
        if piv is None:
            continue
        pivots.append(col)
        A[row], A[piv] = A[piv], A[row]
        for rr in range(r):
            if rr != row and A[rr][col] != 0:
                f = A[rr][col] / A[row][col]
                for cc in range(n + 1):
                    A[rr][cc] -= f * A[row][cc]
        row += 1
    for rr in range(row, r):
        if A[rr][n] != 0:
            return False
    pivot_set = set(pivots)
    free_vars = [j for j in range(n) if j not in pivot_set]
    if not free_vars:
        coeffs = [Fraction(0)] * n
        for idx, col in enumerate(pivots):
            coeffs[col] = A[idx][n] / A[idx][col]
        return all(c >= 0 and c.denominator == 1 for c in coeffs)
    # Enumerate non-negative integer values of the free variables.
    pivot_rhs = []
    pivot_free_coeff = []
    for idx, col in enumerate(pivots):
        d = A[idx][col]
        pivot_rhs.append(A[idx][n] / d)
        pivot_free_coeff.append([-A[idx][fv] / d for fv in free_vars])
    nf = len(free_vars)
    v_norm = sum(abs(x) for x in v) + 1
    g_min_norm = min(max(1, sum(abs(x) for x in g)) for g in gens)
    max_t = max(v_norm // g_min_norm + 2, 10)

    def _search(idx: int, ts: list[int]) -> bool:
        if idx == nf:
            for k in range(len(pivots)):
                val = pivot_rhs[k]
                for jj in range(nf):
                    val += pivot_free_coeff[k][jj] * ts[jj]
                if val < 0 or val.denominator != 1:
                    return False
            return True
        for t in range(max_t + 1):
            ts[idx] = t
            if _search(idx + 1, ts):
                return True
        return False

    return _search(0, [0] * nf)


def _support_bfs(lower: Vec, upper: Vec, gens: list[Vec]) -> list[Vec]:
    """BFS enumeration of the lattice interval  [lower, upper]  in  Gamma_+ ."""
    if not _cone_contains(tuple(u - l for u, l in zip(upper, lower)), gens):
        return [lower] if lower == upper else []
    visited: set[Vec] = {lower}
    frontier = [lower]
    order = [lower]
    while frontier:
        nxt_frontier: list[Vec] = []
        for d in frontier:
            for g in gens:
                n = tuple(x + gi for x, gi in zip(d, g))
                if n in visited:
                    continue
                if _cone_contains(tuple(u - x for u, x in zip(upper, n)), gens):
                    visited.add(n)
                    nxt_frontier.append(n)
                    order.append(n)
        frontier = nxt_frontier
    return order

# --- §4b : Nahm index solver ----------------------------------------
# For a spectrum  spec = [g_1, ..., g_N]  the coefficient of  X_gamma
# in  S |0>  is a finite Nahm sum
#
#    [S|0>]_gamma  =  sum_{n_a >= 0, sum_a n_a g_a = gamma}
#                          (-1)^{sum n_a}  q^{shift(n)} / prod_a (q^2;q^2)_{n_a}
#
# with the shift
#
#    shift(n)  =  sum_a n_a  +  sum_{a<b} n_a n_b <g_a, g_b>.
#
# The routines below enumerate the non-negative integer tuples  (n_a)
# solving  sum_a n_a g_a = gamma  and compute the shift.  Since N can
# exceed the rank, the system is solved by Gaussian elimination over Q
# with enumeration of the free variables.

def _solve_nahm_indices(
    gamma: Vec,
    spec_t: list[Vec],
    max_n: int = 50,
) -> list[tuple[int, ...]]:
    """All non-negative integer tuples  (n_1,...,n_N)  with

        sum_a  n_a  g_a  =  gamma.

    Returns a list (possibly empty) of tuples of length ``N``.

    The Gauss-Jordan reduction uses ``Fraction`` (done once; correct
    for every rank / spec combination we've encountered), but the
    hot inner enumeration over free variables is switched to pure
    integer arithmetic by scaling each pivot row by a common
    denominator ``D = lcm(M[pivot_row][pivot_col] for pivot_row)``.
    On the su2_k3 Schur index at K=4 this reclaims most of the
    ``Fraction.__new__`` cost that dominated the prior profile.
    """
    N = len(spec_t)
    r = len(gamma)
    M = [[Fraction(spec_t[j][i]) for j in range(N)] + [Fraction(gamma[i])]
         for i in range(r)]

    pivots: list[int] = []
    row = 0
    for col in range(N):
        piv = None
        for rr in range(row, r):
            if M[rr][col] != 0:
                piv = rr
                break
        if piv is None:
            continue
        pivots.append(col)
        M[row], M[piv] = M[piv], M[row]
        for rr in range(r):
            if rr != row and M[rr][col] != 0:
                f = M[rr][col] / M[row][col]
                for cc in range(N + 1):
                    M[rr][cc] -= f * M[row][cc]
        row += 1

    for rr in range(row, r):
        if M[rr][N] != 0:
            return []

    free = [j for j in range(N) if j not in pivots]
    if not free:
        base = [Fraction(0)] * N
        for idx, col in enumerate(pivots):
            base[col] = M[idx][N] / M[idx][col]
        if all(v.denominator == 1 and v >= 0 for v in base):
            return [tuple(int(v) for v in base)]
        return []

    # --- Switch to integer arithmetic for the enumeration.
    #
    # For each pivot row, the equation is
    #
    #     M[row][col] * n_col + sum_{fi in free} M[row][fi] * n_fi
    #                           = M[row][N].
    #
    # Scale both sides by the common denominator  D  of all the
    # Fractions appearing in that row (row-local LCM).  Store the
    # integer recipe  (d_col, c_fi_col, b_col)  per pivot col; then
    # the enumeration is integer-only.
    from math import lcm

    d_col: dict[int, int] = {}
    b_col: dict[int, int] = {}
    pivot_effect: dict[int, dict[int, int]] = {fi: {} for fi in free}
    for idx, col in enumerate(pivots):
        denoms = [M[idx][col].denominator, M[idx][N].denominator]
        for fi in free:
            denoms.append(M[idx][fi].denominator)
        D = 1
        for x in denoms:
            D = lcm(D, x)
        d_col[col] = int(M[idx][col] * D)
        b_col[col] = int(M[idx][N] * D)
        for fi in free:
            c = int(M[idx][fi] * D)
            if c != 0:
                pivot_effect[fi][col] = c

    # Accumulator  acc[col]  starts at  b_col[col]  and is decremented
    # by  c_{fi,col} * t_fi  for each free variable.  A solution exists
    # iff  acc[col]  is divisible by  d_col[col]  and the quotient >= 0.
    results: list[tuple[int, ...]] = []

    def _enumerate(fi_pos: int, acc: dict[int, int], ts: list[int]) -> None:
        if fi_pos == len(free):
            ordered: list[int] = [0] * N
            for fi_local, t in zip(free, ts):
                ordered[fi_local] = t
            for col in pivots:
                a = acc[col]
                d = d_col[col]
                if d == 0 or (a % d) != 0:
                    return
                v = a // d
                if v < 0:
                    return
                ordered[col] = v
            results.append(tuple(ordered))
            return
        fi = free[fi_pos]
        dec = pivot_effect[fi]
        old = {col: acc[col] for col in dec}
        for t in range(max_n + 1):
            ts.append(t)
            for col, c in dec.items():
                acc[col] = old[col] - c * t
            abort = False
            for col in pivots:
                a = acc[col]
                d = d_col[col]
                if (d > 0 and a < 0) or (d < 0 and a > 0):
                    fixable = False
                    # Can later t at the current fi fix it?  Increasing
                    # t by 1 changes  acc[col]  by  -dec[col]; want this
                    # to move  a  back toward valid sign.
                    cur_c = dec.get(col, 0)
                    if (d > 0 and cur_c < 0) or (d < 0 and cur_c > 0):
                        fixable = True
                    else:
                        for lf in free[fi_pos + 1:]:
                            ce = pivot_effect[lf].get(col, 0)
                            if (d > 0 and ce < 0) or (d < 0 and ce > 0):
                                fixable = True
                                break
                    if not fixable:
                        abort = True
                        break
            if abort:
                ts.pop()
                break
            _enumerate(fi_pos + 1, acc, ts)
            ts.pop()
        for col, v in old.items():
            acc[col] = v

    initial_acc = {col: b_col[col] for col in pivots}
    _enumerate(0, initial_acc, [])
    return results


def _nahm_shift(ns: tuple[int, ...], kmat: list[list[int]]) -> int:
    """shift(n) = sum_a n_a + sum_{a<b} n_a n_b * <g_a, g_b>."""
    N = len(ns)
    s = sum(ns)
    for i in range(N):
        for j in range(i + 1, N):
            s += ns[i] * ns[j] * kmat[i][j]
    return s

# --- §4c : exact Nahm-term arithmetic (for solve_F) -----------------
# ``solve_F`` works by examining the *non-positive* q-powers of each
# Nahm-series coefficient  [F S|0>]_eta  to read off the coefficients
# of F_gamma.  The helpers below represent a single Nahm term
#
#      sign * q^{shift} / prod_i (q^2;q^2)_{n_i}
#
# exactly, and extract its non-positive q-powers without ever expanding
# to a truncated PowerSeries (which causes premature truncation artifacts).

# Exact [S|0>]_gamma as a HabiroElement (one common-denominator sum of Nahm terms).
_sket_habiro_cache: dict[tuple, HabiroElement] = {}


def _s_ket_habiro(gamma: Vec, spec_t: list[Vec],
                  kmat: list[list[int]]) -> HabiroElement:
    """``[S|0>]_gamma`` as an exact HabiroElement (cached)."""
    # The Nahm shift depends on the pairings between spec charges, so
    # two quivers sharing the same ``spec_t`` but with different ``kmat``
    # produce different HabiroElements.  Include ``kmat`` in the cache
    # key to avoid cross-quiver contamination.
    key = (gamma, tuple(tuple(g) for g in spec_t),
           tuple(tuple(row) for row in kmat))
    cached = _sket_habiro_cache.get(key)
    if cached is not None:
        return cached
    terms = []
    for ns in _solve_nahm_indices(gamma, spec_t):
        sign = 1 if sum(ns) % 2 == 0 else -1
        terms.append(HabiroElement.nahm_term(sign, _nahm_shift(ns, kmat), list(ns)))
    result = HabiroElement.sum(terms)
    _sket_habiro_cache[key] = result
    return result


def _state_nonpos_at(
    F_dict: dict[Vec, dict[int, int]],
    lattice: Lattice,
    eta: Vec,
    spec_t: list[Vec],
    kmat: list[list[int]],
) -> dict[int, int]:
    """Non-positive q-powers of  [F * S|0>]_eta  via HabiroElement.

    All (delta, F_delta) contributions are accumulated into a single
    HabiroElement using the batched `sum(...)` (one simplify at the end),
    and we extract the non-positive-q part with a single `expand(0)`.
    Much faster than the per-term convolution used in the pre-Habiro
    `_NahmTerm` path.
    """
    contribs = []
    for delta, c_dict in F_dict.items():
        if not c_dict:
            continue
        mu = tuple(e - d for e, d in zip(eta, delta))
        s_el = _s_ket_habiro(mu, spec_t, kmat)
        if s_el.is_zero():
            continue
        twist = lattice.bracket(delta, mu)
        # Fold c_dict * q^{twist} into s_el's numerator (avoids two per-mult
        # HabiroElement simplifies).
        scale = LaurentPoly(c_dict) * LaurentPoly({twist: 1})
        contribs.append(HabiroElement(scale * s_el.numerator, dict(s_el.denom)))
    total = HabiroElement.sum(contribs)
    if total.is_zero():
        return {}
    k_min = total.k_min()
    if k_min is None or k_min > 0:
        return {}
    np = total.expand(0)
    return {e: c for e, c in np._coeffs.items() if e <= 0 and c != 0}


def _make_palindromic(neg_part: dict[int, int]) -> dict[int, int]:
    """Complete a non-positive-degree dict into a bar-invariant LaurentPoly.

    ``F_gamma`` is required to have palindromic coefficients, so we
    mirror the strictly-negative q-powers around 0.
    """
    result = dict(neg_part)
    for s, v in neg_part.items():
        if s < 0 and v != 0:
            result[-s] = result.get(-s, 0) + v
    return {k: v for k, v in result.items() if v != 0}


def clear_caches() -> None:
    """Clear the internal Nahm / S|0> caches."""
    _sket_cache.clear()

# --- §4d : solve_F (principal canonical-basis F_gamma solver) -------
def solve_F(
    lattice: Lattice,
    spec: Sequence[Sequence[int]],
    cone_gens: Sequence[Sequence[int]],
    gamma: Sequence[int],
) -> dict[Vec, LaurentPoly]:
    """Principal canonical-basis solver.

    Determines

        F_gamma  =  X_gamma  +  sum_{delta > 0}  C_delta(q)  X_{gamma + delta}

    from the defining condition

        F_gamma * S |0>  =  |gamma>  +  O(q)

    with bar-invariant (palindromic) coefficients  C_delta(q)  in Z[q,q^{-1}].

    Single forward pass:

      * Compute the doubly-tropical interval  [gamma, -sigma^{-1}(gamma)]
        (intersected with the positive cone  Gamma_+ ) and walk it in
        BFS order from gamma.
      * At each new  delta , the *non-positive* q-powers of
        ``[F S|0>]_delta``  (computed exactly from the Nahm expansion of
        S|0>) read off the negative-degree part of  C_delta ; completing
        palindromically gives  C_delta  itself.
      * Constraints at charges outside the support are automatically
        satisfied by q-holonomicity of  S .

    Parameters
    ----------
    lattice    : ambient lattice with pairing
    spec       : spectrum charges [g_1, ..., g_N] defining
                 S = E_q(X_{g_1}) * ... * E_q(X_{g_N})
    cone_gens  : generators of the positive cone  Gamma_+
    gamma      : target tropical charge

    Returns
    -------
    F : dict[Vec, LaurentPoly]  -- the canonical basis element F_gamma.
    """
    gamma_t = lattice.check(gamma)
    spec_t = [lattice.check(g) for g in spec]
    cone_t = [lattice.check(g) for g in cone_gens]
    N = len(spec_t)
    kmat = [[lattice.bracket(spec_t[i], spec_t[j]) for j in range(N)]
            for i in range(N)]

    sinv = sigma_inverse(lattice, spec_t, gamma_t)
    upper: Vec = tuple(-x for x in sinv)
    support = _support_bfs(gamma_t, upper, cone_t)

    # Work with raw dict[Vec, dict[int, int]] (a LaurentPoly-like) for speed;
    # wrap in LaurentPoly only at the end.
    F_dict: dict[Vec, dict[int, int]] = {gamma_t: {0: 1}}
    for delta in support:
        if delta == gamma_t:
            continue
        nonpos = _state_nonpos_at(F_dict, lattice, delta, spec_t, kmat)
        if nonpos:
            # Negate to compensate for [F S|0>]_delta: F_delta must kill this
            # non-positive residue; then complete palindromically.
            F_dict[delta] = _make_palindromic({s: -v for s, v in nonpos.items()})

    return {d: LaurentPoly(c) for d, c in F_dict.items() if c}


def verify_F(
    F: dict[Vec, LaurentPoly],
    lattice: Lattice,
    spec: Sequence[Sequence[int]],
    gamma: Sequence[int],
) -> tuple[bool, list[tuple[Vec, dict[int, int], dict[int, int]]]]:
    """Check that a candidate  F  is the canonical-basis element  F_gamma .

    Verifies the defining condition

        F * S|0>  =  |gamma>  +  O(q)

    by computing the non-positive q-powers of  [F * S|0>]_eta  for every
    eta in the support of  F , via the same Habiro path that  ``solve_F``
    uses internally.  A valid  F_gamma  has:

      * [F S|0>]_gamma  =  1 + O(q)   (nonpos part = ``{0: 1}``)
      * [F S|0>]_eta    =  O(q)  for  eta != gamma   (nonpos part = ``{}``)

    Crucially, this is a pure *forward* computation -- one exact Nahm
    expansion per eta, no linear-algebra solve.  It is O(|support(F)|)
    Habiro convolutions, typically an order of magnitude faster than
    ``solve_F``  itself, and independent of whether  F  was computed
    from scratch, lifted from a parent chart, or assembled from a
    Pieri / Jacobi-Trudi product decomposition.

    Parameters
    ----------
    F        : candidate ``dict[Vec, LaurentPoly]`` to verify.
    lattice  : ambient lattice.
    spec     : spectrum charges defining  S .
    gamma    : expected leading tropical charge.

    Returns
    -------
    (ok, defects) where:
      * ``ok``  is True iff every eta in  support(F)  satisfies the
        canonical-basis condition.
      * ``defects``  is a list of ``(eta, actual_nonpos, expected_nonpos)``
        triples for each violation -- empty when ``ok`` is True.
    """
    gamma_t = lattice.check(gamma)
    spec_t = [lattice.check(g) for g in spec]
    N = len(spec_t)
    kmat = [[lattice.bracket(spec_t[i], spec_t[j]) for j in range(N)]
            for i in range(N)]
    F_dict: dict[Vec, dict[int, int]] = {
        lattice.check(d): dict(c._coeffs) for d, c in F.items()
    }
    defects: list[tuple[Vec, dict[int, int], dict[int, int]]] = []
    for eta in F_dict:
        nonpos = _state_nonpos_at(F_dict, lattice, eta, spec_t, kmat)
        expected = {0: 1} if eta == gamma_t else {}
        if nonpos != expected:
            defects.append((eta, nonpos, expected))
    return (not defects), defects


# =====================================================================
# SECTION 5 -- PowerSeries, q-Pochhammer, Schur index via exact Nahm
# =====================================================================
# The Schur index  I_{a,b}(q)  of the theory is a q-series, not a
# Laurent polynomial.  We represent q-series as truncated ``PowerSeries``
# objects (dict[int, int] with a degree cap K) and compute  I_{a,b}
# via the exact Nahm expansion
#
#    I_{a,b}  =  (q^2;q^2)_inf^r  *  sum_gamma  c_a(gamma) * c_b(gamma)
#
# with  c_a(gamma)  =  [F_a S |0>]_gamma  and  r = rk Gamma_g  the gauge
# rank.  ``schur_index_nahm`` is the end-to-end driver; the helper routines
# compute the q-Pochhammer prefactor and the individual  c_a(gamma) .

class PowerSeries:
    """Truncated formal power series  sum_{e >= 0} c_e q^e  mod q^{K+1} ,
    with ``c_e in Z``.  Negative powers are allowed up to  ``-K`` (e.g.
    shifted terms arising from charge twists in Nahm sums)."""

    __slots__ = ("_c", "K")

    def __init__(self, coeffs: dict[int, int] | None = None, K: int = 40):
        self.K = K
        self._c: dict[int, int] = {}
        if coeffs:
            for e, v in coeffs.items():
                if v != 0 and e <= K:
                    self._c[e] = v

    @classmethod
    def zero(cls, K: int = 40) -> "PowerSeries":
        return cls(K=K)

    @classmethod
    def one(cls, K: int = 40) -> "PowerSeries":
        return cls({0: 1}, K=K)

    @classmethod
    def qpow(cls, n: int, K: int = 40) -> "PowerSeries":
        return cls.zero(K) if n > K else cls({n: 1}, K=K)

    def is_zero(self) -> bool:
        return not self._c

    def __getitem__(self, e: int) -> int:
        return self._c.get(e, 0)

    def __neg__(self) -> "PowerSeries":
        return PowerSeries({e: -v for e, v in self._c.items()}, self.K)

    def __add__(self, other: "PowerSeries") -> "PowerSeries":
        K = min(self.K, other.K)
        out: dict[int, int] = dict(self._c)
        for e, v in other._c.items():
            if e > K:
                continue
            s = out.get(e, 0) + v
            if s == 0:
                out.pop(e, None)
            else:
                out[e] = s
        return PowerSeries(out, K)

    def __sub__(self, other: "PowerSeries") -> "PowerSeries":
        return self + (-other)

    def __mul__(self, other: "PowerSeries") -> "PowerSeries":
        K = min(self.K, other.K)
        out: dict[int, int] = {}
        for e1, v1 in self._c.items():
            if e1 > K:
                continue
            for e2, v2 in other._c.items():
                e = e1 + e2
                if e > K:
                    continue
                out[e] = out.get(e, 0) + v1 * v2
        return PowerSeries({e: v for e, v in out.items() if v != 0}, K)

    def __rmul__(self, other: int) -> "PowerSeries":
        if other == 0:
            return PowerSeries.zero(self.K)
        return PowerSeries({e: other * v for e, v in self._c.items()}, self.K)

    def shift(self, n: int) -> "PowerSeries":
        """Multiply by  q^n  (in place of the q^{twist} dressing)."""
        return PowerSeries(
            {e + n: v for e, v in self._c.items() if e + n <= self.K},
            self.K,
        )

    def __repr__(self) -> str:
        if not self._c:
            return "0"
        parts = []
        for e in sorted(self._c):
            v = self._c[e]
            if e == 0:
                parts.append(str(v))
            elif v == 1:
                parts.append(f"q^{e}")
            elif v == -1:
                parts.append(f"-q^{e}")
            else:
                parts.append(f"{v}*q^{e}")
        return " + ".join(parts).replace("+ -", "- ")


# ---------------------------------------------------------------------
# q-Pochhammer symbols used in Schur-index prefactors and Nahm denominators
# ---------------------------------------------------------------------

def qpoch_finite(k: int, K: int) -> PowerSeries:
    """``(q^2; q^2)_k = prod_{j=1}^k (1 - q^{2j})``  truncated to  q^K ."""
    result = PowerSeries.one(K)
    for j in range(1, k + 1):
        result = result * PowerSeries({0: 1, 2 * j: -1}, K)
    return result


def qpoch_infty(K: int) -> PowerSeries:
    """``(q^2; q^2)_infty``  truncated to  q^K ."""
    result = PowerSeries.one(K)
    for j in range(1, K // 2 + 1):
        if 2 * j > K:
            break
        result = result * PowerSeries({0: 1, 2 * j: -1}, K)
    return result


def _invert_series(f: PowerSeries, K: int) -> PowerSeries:
    """Invert a power series with  f[0] = 1 , truncated to order K."""
    assert f[0] == 1, "constant term must be 1 to invert"
    inv: dict[int, int] = {0: 1}
    for n in range(1, K + 1):
        s = 0
        for m in range(1, n + 1):
            fm = f[m]
            if fm != 0 and (n - m) in inv:
                s += fm * inv[n - m]
        if s != 0:
            inv[n] = -s
    return PowerSeries(inv, K)


def inv_qpoch_finite(k: int, K: int) -> PowerSeries:
    """``1 / (q^2; q^2)_k``  as a truncated power series."""
    return _invert_series(qpoch_finite(k, K), K)


# ---------------------------------------------------------------------
# Nahm-based computation of  [F S |0>]_gamma  and the Schur overlap.
#
# Each c_F(gamma) is carried as an exact HabiroElement until the final
# overlap -- no `K_internal` truncation knob, no deep-F_a failure mode.
# We only expand to PowerSeries(K) when multiplying into the
# (q^2;q^2)_inf^r prefactor.
# ---------------------------------------------------------------------


def _c_gamma_habiro(
    gamma: Vec,
    F: "dict[Vec, LaurentPoly] | None",
    spec_t: list[Vec],
    kmat: list[list[int]],
    lattice: Lattice,
) -> HabiroElement:
    """``[F S |0>]_gamma`` as an exact HabiroElement.

    Sum over delta-in-F of  f_delta * q^{<delta, eta>} * [S|0>]_eta  with
    eta = gamma - delta.  Batched via `HabiroElement.sum(...)` (one
    simplify at the end).
    """
    if F is None:
        return _s_ket_habiro(gamma, spec_t, kmat)
    contribs = []
    for delta, f_coeff in F.items():
        if f_coeff.is_zero():
            continue
        eta = tuple(g - d for g, d in zip(gamma, delta))
        s_eta = _s_ket_habiro(eta, spec_t, kmat)
        if s_eta.is_zero():
            continue
        twist = lattice.bracket(delta, eta)
        scale = f_coeff * LaurentPoly({twist: 1})
        contribs.append(HabiroElement(scale * s_eta.numerator, dict(s_eta.denom)))
    return HabiroElement.sum(contribs)


def _habiro_to_ps_trunc(h: HabiroElement, K: int) -> PowerSeries:
    """Expand HabiroElement to a PowerSeries of order K (non-negative part)."""
    if h.is_zero():
        return PowerSeries.zero(K)
    lp = h.expand(K)
    return PowerSeries({e: v for e, v in lp._coeffs.items() if e <= K}, K)


def schur_index_nahm(
    lattice: Lattice,
    spec: Sequence[Sequence[int]],
    F_a: dict[Vec, LaurentPoly] | None = None,
    F_b: dict[Vec, LaurentPoly] | None = None,
    K: int = 20,
    r: int | None = None,
    cone_cutoff: int = 12,
    K_internal: int | None = None,  # accepted for back-compat; ignored
) -> PowerSeries:
    """Compute

        I_{a,b}(q)  =  (q^2;q^2)_inf^r  *  sum_gamma  c_a(gamma) c_b(gamma)

    to order  q^K  via exact HabiroElement arithmetic internally.  No
    `K_internal` is needed (the intermediate representation is exact, not
    truncated).  ``r`` defaults to ``lattice.rank``; for flavoured
    theories pass ``r = gauge rank``.  ``F_a = None`` means the identity.
    """
    if r is None:
        r = lattice.rank

    spec_t = [lattice.check(g) for g in spec]
    N = len(spec_t)
    kmat = [[lattice.bracket(spec_t[i], spec_t[j]) for j in range(N)]
            for i in range(N)]

    # Enumerate eta-charges that can possibly contribute to the overlap.
    all_deltas: set[Vec] = set()
    for F in (F_a, F_b):
        if F is None:
            all_deltas.add(tuple(0 for _ in range(lattice.rank)))
        else:
            for delta in F:
                all_deltas.add(delta)
    zero = tuple(0 for _ in range(lattice.rank))
    output: set[Vec] = set()
    for delta in all_deltas:
        visited: set[Vec] = {zero}
        queue: deque[Vec] = deque([zero])
        while queue:
            eta = queue.popleft()
            g = tuple(d + e for d, e in zip(delta, eta))
            if sum(abs(x) for x in g) <= cone_cutoff:
                output.add(g)
            for gi in spec_t:
                nxt = tuple(e + v for e, v in zip(eta, gi))
                if nxt in visited or sum(abs(x) for x in nxt) > cone_cutoff:
                    continue
                visited.add(nxt)
                queue.append(nxt)

    pf = qpoch_infty(K)
    pf_r = pf
    for _ in range(r - 1):
        pf_r = pf_r * pf

    overlap = PowerSeries.zero(K)
    for gamma in output:
        ca_h = _c_gamma_habiro(gamma, F_a, spec_t, kmat, lattice)
        if ca_h.is_zero():
            continue
        cb_h = _c_gamma_habiro(gamma, F_b, spec_t, kmat, lattice)
        if cb_h.is_zero():
            continue
        ca = _habiro_to_ps_trunc(ca_h, K)
        cb = _habiro_to_ps_trunc(cb_h, K)
        overlap = overlap + ca * cb

    return pf_r * overlap


# =====================================================================
# SECTION 6 -- BPSQuiver and the S-finder (negating mutation sequence)
# =====================================================================
# A decorated BPS quiver  Q  is stored as
#
#   * a list of node charges  gamma_i^Q  in some ambient lattice,
#   * a flag per node (frozen = flavour, unfrozen = gauge),
#   * the exchange matrix  B_{ij} = <gamma_i^Q, gamma_j^Q>  (the "arrows").
#
# The spectrum generator  S  is obtained by finding a *negating mutation
# sequence* -- a sequence of tropical mutations that sends every gauge
# node charge to its negative -- and reading off the charges at which the
# sequence mutates.  These charges, in the order they appear, are the
# factors of
#
#     S  =  E_q(X_{g_1}) * E_q(X_{g_2}) * ... * E_q(X_{g_N}).
#
# Note N can exceed the gauge rank and  g_k  can be a composite (a sum of
# node charges).  ``BPSQuiver.find_negating_sequence`` performs a BFS over
# mutation sequences restricted to nodes whose charge lies in the cone of
# the original gauge generators.

def _in_positive_cone_int(charge: tuple, generators: list[tuple]) -> bool:
    """Is ``charge`` a non-negative integer combination of ``generators`` ?
    Used by the mutation BFS to prune to admissible mutations only.

    Delegates to :func:`_cone_contains` , which correctly enumerates the
    free variables when ``len(generators) > rank``  -- i.e. when the
    generators are linearly dependent (the "drop flavour" case).  An
    earlier version read coefficients only at pivot columns and would
    falsely reject charges like  ``(-1, 1)``  in the cone of
    ``[(1,0), (0,1), (-1,1)]``  because the pivot-only solution sets
    the free variable to zero.
    """
    return _cone_contains(tuple(charge), [tuple(g) for g in generators])


def _compositions(k: int, d: int):
    """Yield non-negative integer tuples of length ``k`` summing to ``d``."""
    if k == 1:
        yield (d,)
        return
    for first in range(d + 1):
        for rest in _compositions(k - 1, d - first):
            yield (first,) + rest


def _spec_pair_matrix(spec_charges: list[tuple],
                      ambient_B: Sequence[Sequence[int]]) -> list[list[int]]:
    """Gram matrix ``M[i][j] = <g_i, g_j>`` for ``g_i in spec_charges``
    under the ambient pairing ``ambient_B``."""
    n = len(spec_charges)
    rank = len(ambient_B)
    mat = [[0] * n for _ in range(n)]
    for i in range(n):
        gi = spec_charges[i]
        for j in range(n):
            gj = spec_charges[j]
            s = 0
            for a in range(rank):
                gia = gi[a]
                if gia == 0:
                    continue
                row = ambient_B[a]
                for b in range(rank):
                    s += row[b] * gia * gj[b]
            mat[i][j] = int(s)
    return mat


def _cone_depth_of_first_nonpos_shift(pair_mat: list[list[int]],
                                      max_d: int) -> int:
    """Smallest depth  d in [1, max_d]  such that some  n in Z_{>=0}^k
    with  sum(n) = d  has  shift(n) <= 0 , where

        shift(n) = sum n_i  +  sum_{i<j}  n_i n_j  pair_mat[i][j].

    Returns  ``max_d + 1``  if no such  n  is found up to  ``max_d`` .

    Interpretation:  shift(n) is the  q-power of the Nahm term in the
    ordered product  E_q(X_{g_1}) ... E_q(X_{g_k}) |0>  attached to the
    multi-index  n ; the full partial state is  1 + O(q^{D(spec)}) ·
    {non-trivial}.  For a *complete* spectrum generator the leading
    q-power at every  gamma != 0  must be  >= 1 , but that only holds
    after Nahm cancellations over multiple multi-indices; a partial
    product can have genuine non-positive shifts, and  D  measures how
    deep into the cone they first appear.
    """
    k = len(pair_mat)
    if k == 0 or max_d < 1:
        return max_d + 1
    # d = 1 : a single  n_i = 1 , all others 0, gives  shift = 1 > 0 .
    for d in range(2, max_d + 1):
        for n in _compositions(k, d):
            s = d
            # active indices (n_i > 0)
            active = []
            for i in range(k):
                if n[i]:
                    active.append(i)
            if len(active) < 2:
                continue  # single-index  n  always has  shift = d > 0
            for ai in range(len(active)):
                i = active[ai]
                ni = n[i]
                row = pair_mat[i]
                for aj in range(ai + 1, len(active)):
                    j = active[aj]
                    s += ni * n[j] * row[j]
            if s <= 0:
                return d
    return max_d + 1


class BPSQuiver:
    """Decorated BPS quiver: charges, frozen flags, exchange matrix.

    Supports Fomin-Zelevinsky (tropical) mutation  :meth:`mutate` , finite
    mutation-sequence composition  :meth:`mutation_sequence` , a BFS
    search  :meth:`find_negating_sequence`  for a sequence that negates
    all gauge charges, and  :meth:`build_spectrum_generator`  which reads
    off the charges of  E_q  factors along such a sequence.
    """

    def __init__(self, charges: list[tuple], frozen: list[bool] | None = None,
                 exchange_matrix: list[list[int]] | None = None,
                 ambient_pairing: Sequence[Sequence[int]] | None = None):
        self.charges = [tuple(c) for c in charges]
        self.n_nodes = len(charges)
        self.frozen = list(frozen) if frozen is not None else [False] * self.n_nodes
        if exchange_matrix is not None:
            self.exchange = [list(row) for row in exchange_matrix]
        else:
            self.exchange = [[0] * self.n_nodes for _ in range(self.n_nodes)]
        # Optional: the ambient lattice pairing that produced  exchange .
        # Carried through  :meth:`mutate`  unchanged (mutation is a lattice
        # automorphism, so the ambient pairing is invariant).  Used by the
        # opt-in heuristic in :meth:`find_negating_sequence` to evaluate
        # pairings between spec charges from different mutation steps.
        if ambient_pairing is not None:
            self.ambient_pairing: tuple[tuple[int, ...], ...] | None = tuple(
                tuple(int(x) for x in row) for row in ambient_pairing
            )
        else:
            self.ambient_pairing = None

    @classmethod
    def from_pairing(cls, charges: Sequence[Sequence[int]],
                     pairing_matrix: Sequence[Sequence[int]],
                     frozen: Sequence[bool] | None = None) -> "BPSQuiver":
        """Build a quiver from node charges + ambient lattice pairing.

        Sets  exchange[i][j] = <gamma_i, gamma_j>  with the ambient pairing.
        Stores the ambient pairing on the instance so the heuristic branch
        of  :meth:`find_negating_sequence`  can evaluate pairings between
        spec charges sampled at different mutation steps.
        """
        rank = len(charges[0])
        n = len(charges)
        exchange = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                val = sum(
                    pairing_matrix[a][b] * charges[i][a] * charges[j][b]
                    for a in range(rank) for b in range(rank)
                )
                exchange[i][j] = int(val)
        fr = list(frozen) if frozen is not None else [False] * n
        return cls([tuple(c) for c in charges], fr, exchange,
                   ambient_pairing=pairing_matrix)

    def mutate(self, k: int) -> "BPSQuiver":
        """Fomin-Zelevinsky tropical mutation at node k (must be unfrozen).

        Applies  mu_k^t  to every unfrozen node charge and updates the
        exchange matrix by the standard FZ formula.  Returns a new quiver.
        """
        if self.frozen[k]:
            raise ValueError(f"cannot mutate frozen node {k}")
        n = self.n_nodes
        new_exchange = [row[:] for row in self.exchange]
        new_charges = [list(c) for c in self.charges]
        rank = len(self.charges[0])
        for i in range(n):
            if i == k:
                continue
            for j in range(n):
                if j == k:
                    continue
                new_exchange[i][j] = (self.exchange[i][j]
                    + max(0, self.exchange[i][k]) * max(0, self.exchange[k][j])
                    - max(0, -self.exchange[i][k]) * max(0, -self.exchange[k][j]))
        for i in range(n):
            if i != k:
                new_exchange[i][k] = -self.exchange[i][k]
                new_exchange[k][i] = -self.exchange[k][i]
        ck = list(self.charges[k])
        for j in range(n):
            if j == k:
                new_charges[j] = [-x for x in ck]
            else:
                shift = max(0, self.exchange[j][k])
                for a in range(rank):
                    new_charges[j][a] = self.charges[j][a] + shift * ck[a]
        return BPSQuiver([tuple(c) for c in new_charges], self.frozen[:],
                         new_exchange, ambient_pairing=self.ambient_pairing)

    def mutation_sequence(self, seq: Sequence[int]) -> "BPSQuiver":
        q = self
        for k in seq:
            q = q.mutate(k)
        return q

    def reverse_mutate(self, k: int) -> "BPSQuiver":
        """Tropical **inverse** mutation at node k.

        Inverts :meth:`mutate`:  ``Q.mutate(k).reverse_mutate(k) == Q`` .
        This is the mutation ``μ^{-1}_{-γ_k}``,
        obtained from the forward formula by flipping the sign inside
        ``max()``:

            γ_k → -γ_k
            γ_j → γ_j + max(-⟨γ_j, γ_k⟩, 0) γ_k     (j ≠ k)

        The exchange-matrix update is the same as :meth:`mutate` — the
        matrix update is an involution on ``B`` — so the two methods
        differ only in the charge update.  Used by
        :meth:`_find_negating_sequence_bidirectional` to walk backward
        from the negated endpoint.
        """
        if self.frozen[k]:
            raise ValueError(f"cannot mutate frozen node {k}")
        n = self.n_nodes
        new_exchange = [row[:] for row in self.exchange]
        new_charges = [list(c) for c in self.charges]
        rank = len(self.charges[0])
        for i in range(n):
            if i == k:
                continue
            for j in range(n):
                if j == k:
                    continue
                new_exchange[i][j] = (self.exchange[i][j]
                    + max(0, self.exchange[i][k]) * max(0, self.exchange[k][j])
                    - max(0, -self.exchange[i][k]) * max(0, -self.exchange[k][j]))
        for i in range(n):
            if i != k:
                new_exchange[i][k] = -self.exchange[i][k]
                new_exchange[k][i] = -self.exchange[k][i]
        ck = list(self.charges[k])
        for j in range(n):
            if j == k:
                new_charges[j] = [-x for x in ck]
            else:
                shift = max(0, -self.exchange[j][k])
                for a in range(rank):
                    new_charges[j][a] = self.charges[j][a] + shift * ck[a]
        return BPSQuiver([tuple(c) for c in new_charges], self.frozen[:],
                         new_exchange, ambient_pairing=self.ambient_pairing)

    def _free_cover(self) -> "BPSQuiver":
        """Return the "free cover" of this quiver: same exchange matrix,
        same frozen flags, but with charges replaced by the standard basis
        e_1, ..., e_N  of  Z^N .  The ambient pairing of the cover is the
        exchange matrix itself (so that  <e_i, e_j>_cover = exchange[i][j]).

        Mutation (forward and reverse) is an abstract index-level operation
        whose evolution depends only on the exchange matrix, so the cover
        and  self  evolve in lockstep up to the relabelling
        ``e_i ↔ self.charges[i]`` .  In particular a mutation sequence is
        a negating sequence in the cover iff it is one in  self  -- but the
        cover is also free of the pathology where two cover states project
        to the same multiset of  Γ -charges (when ``self.charges`` are
        linearly dependent), or where dependent generators enlarge the
        positive cone admissibility test in  :meth:`find_negating_sequence`
        and admit "shortcut" sequences that don't lift to any independent
        cover.  Running the search in the cover therefore guarantees the
        sequence we extract corresponds to an honest factorisation of  S
        into  E_q  factors -- the flavour truncation of the unique cover
        spec, never a spurious shortcut.
        """
        n = self.n_nodes
        cover_charges = [tuple(1 if i == j else 0 for i in range(n))
                         for j in range(n)]
        # The cover ambient pairing on  Z^N  satisfies  <e_i, e_j> = B_{ij} ,
        # i.e. it equals the exchange matrix of  self .
        return BPSQuiver(
            cover_charges,
            frozen=self.frozen[:],
            exchange_matrix=[row[:] for row in self.exchange],
            ambient_pairing=[row[:] for row in self.exchange],
        )

    def find_negating_sequence(self, max_depth: int = 20,
                               allow_permutation: bool = True,
                               heuristic: bool = False,
                               heuristic_max_depth: int = 5,
                               bidirectional: bool = True,
                               edge_mult_heuristic: bool = True,
                               edge_mult_threshold: int = 2,
                               _cover_search: bool = True,
                               ) -> list[int] | None:
        """Search for a mutation sequence negating every unfrozen charge.

        Only mutations at nodes whose *current* charge still lies in the
        non-negative cone of the *original* gauge charges are considered
        (this guarantees positivity of the spectrum factors).

        Parameters
        ----------
        bidirectional
            If True (default), use bidirectional BFS with meet-in-the-
            middle up to permutation of mutable charges.  Forward BFS
            from  self  (mutate + positive-cone constraint), backward
            BFS from the reflected endpoint (reverse_mutate +
            negative-cone constraint), matched on multisets of mutable
            charges.  Dramatically faster than plain BFS on larger
            theories (e.g. ~170x on pure SU(4)) while finding the same
            minimal length.
        edge_mult_heuristic
            If True (default), add a priority penalty for quiver states
            whose exchange matrix has off-diagonal entries exceeding
            ``edge_mult_threshold``.  Conventional BPS quivers have
            ``|B_{ij}| ≤ 2``, and the correct negating sequence
            preserves this bound at every step; detours through
            higher multiplicities are dead-end branches.  Only active
            when ``bidirectional=True``.
        edge_mult_threshold
            Threshold above which edge multiplicities incur a penalty
            (default 2).
        heuristic
            If True, replace plain BFS by a best-first search ordered
            by the "cone-depth of the first non-positive q-power in the
            partial  E_q  product" (see :func:`_cone_depth_of_first_nonpos_shift`).
            Requires ``self.ambient_pairing`` to be set (e.g. via
            :meth:`from_pairing`); otherwise falls back to plain BFS.
            Superseded by ``bidirectional=True`` as the recommended
            default, but kept available for special cases.
        heuristic_max_depth
            Cap on the cone-depth enumerated when evaluating  D ; branches
            that remain clean up to this depth are treated as equally
            good (priority value  ``heuristic_max_depth + 1`` ).

        Returns the mutation sequence (list of node indices) or None.

        Internal flag ``_cover_search``  (default True) routes the search
        through :meth:`_free_cover` -- a quiver with the same exchange
        matrix but standard-basis charges in  Z^N .  This avoids the two
        pathologies that arise when the actual ``self.charges`` are
        linearly dependent: (i) the positive-cone admissibility test
        ``_in_positive_cone_int(charge, originals)``  becomes too
        permissive (a charge has multiple cone decompositions, only one
        need be non-negative for the test to pass, even if the canonical
        decomposition has negative coordinates), and (ii) two distinct
        cover states can collapse to the same multiset of  Γ -charges,
        causing the visited set to merge unrelated branches.  Either
        effect can yield a sequence whose induced  Γ -spec is not the
        flavour truncation of any honest cover spec.  Searching in the
        cover and replaying the resulting indices on  self  (via
        :meth:`build_spectrum_generator`) sidesteps both.  The mutation
        rule is purely index-level, so the sequence is identical in cover
        and self; only the search dynamics differ in the dependent case.
        """
        if _cover_search:
            cover = self._free_cover()
            # Forward all parameters; the cover does the search natively.
            return cover.find_negating_sequence(
                max_depth=max_depth,
                allow_permutation=allow_permutation,
                heuristic=heuristic,
                heuristic_max_depth=heuristic_max_depth,
                bidirectional=bidirectional,
                edge_mult_heuristic=edge_mult_heuristic,
                edge_mult_threshold=edge_mult_threshold,
                _cover_search=False,
            )

        initial_mutable = {i: tuple(c) for i, c in enumerate(self.charges)
                           if not self.frozen[i]}
        mutable_indices = sorted(initial_mutable.keys())
        originals = [initial_mutable[i] for i in mutable_indices]
        neg_set = frozenset(tuple(-x for x in c) for c in originals)

        def is_done(q: "BPSQuiver") -> bool:
            cur = frozenset(q.charges[i] for i in mutable_indices)
            if allow_permutation:
                return cur == neg_set
            return all(tuple(-x for x in initial_mutable[i]) == tuple(q.charges[i])
                       for i in mutable_indices)

        if bidirectional:
            return self._find_negating_sequence_bidirectional(
                mutable_indices=mutable_indices,
                originals=originals,
                max_depth=max_depth,
                allow_permutation=allow_permutation,
                use_edge_mult=edge_mult_heuristic,
                edge_mult_threshold=edge_mult_threshold,
            )

        if heuristic and self.ambient_pairing is not None:
            return self._find_negating_sequence_best_first(
                mutable_indices=mutable_indices,
                originals=originals,
                is_done=is_done,
                max_depth=max_depth,
                heuristic_max_depth=heuristic_max_depth,
            )

        queue_bfs = deque([(self, [])])
        visited = {self._charge_key()}
        while queue_bfs:
            current, path = queue_bfs.popleft()
            if len(path) > max_depth:
                return None
            if is_done(current):
                return path
            for k in mutable_indices:
                if not _in_positive_cone_int(current.charges[k], originals):
                    continue
                new_q = current.mutate(k)
                key = new_q._charge_key()
                if key in visited:
                    continue
                visited.add(key)
                queue_bfs.append((new_q, path + [k]))
        return None

    def _find_negating_sequence_best_first(
        self,
        *,
        mutable_indices: list[int],
        originals: list[tuple],
        is_done,
        max_depth: int,
        heuristic_max_depth: int,
    ) -> list[int] | None:
        """Best-first variant of :meth:`find_negating_sequence`.

        The priority ordering is  (-D, path_len, insertion_counter)
        where  D = :func:`_cone_depth_of_first_nonpos_shift` (spec_so_far).
        The visited set is keyed by quiver state so the same state is
        only processed once, at its best-priority discovery; downstream
        spec extensions from an alternative path to the same state are
        therefore skipped.  This trades some coverage for speed; the
        plain-BFS path remains available when correctness-of-coverage
        matters.
        """
        import heapq

        B = self.ambient_pairing
        assert B is not None

        counter = 0
        start_priority = (-(heuristic_max_depth + 1), 0, counter)
        heap = [(start_priority, self, [], [])]
        # Visited-at-enqueue (rather than at-pop) so each quiver state
        # is pushed onto the heap exactly once.  This keeps the heap
        # small -- otherwise a state with many predecessors gets
        # enqueued once per predecessor and the heap overhead can
        # dominate for large theories (observed on pure SU(4) before
        # the fix).  The cost: the priority ordering only matters at
        # discovery time, not subsequent re-visits -- acceptable since
        # the heuristic is an ordering bias, not a correctness filter.
        visited: set[tuple] = {self._charge_key()}

        while heap:
            _prio, current, path, spec = heapq.heappop(heap)
            if len(path) > max_depth:
                continue
            if is_done(current):
                return path
            for k in mutable_indices:
                if not _in_positive_cone_int(current.charges[k], originals):
                    continue
                new_q = current.mutate(k)
                nkey = new_q._charge_key()
                if nkey in visited:
                    continue
                visited.add(nkey)
                g_new = tuple(current.charges[k])
                new_spec = spec + [g_new]
                pair_mat = _spec_pair_matrix(new_spec, B)
                D = _cone_depth_of_first_nonpos_shift(
                    pair_mat, heuristic_max_depth)
                counter += 1
                new_prio = (-D, len(path) + 1, counter)
                heapq.heappush(heap, (new_prio, new_q, path + [k], new_spec))
        return None

    def _find_negating_sequence_bidirectional(
        self,
        *,
        mutable_indices: list[int],
        originals: list[tuple],
        max_depth: int,
        allow_permutation: bool,
        use_edge_mult: bool = True,
        edge_mult_threshold: int = 2,
    ) -> list[int] | None:
        """Bidirectional BFS with meet-in-the-middle up to permutation.

        Forward BFS from  self  uses :meth:`mutate` (positive-cone
        constraint); backward BFS from the reflected endpoint (all
        mutable charges negated, same B) uses :meth:`reverse_mutate`
        (negative-cone constraint).  Midpoints match when their
        mutable-charge *multisets* coincide — sound under
        ``allow_permutation=True`` because any permutation of the
        reflected endpoint is a valid end state.  The combined forward
        sequence is  ``fwd_path + [π^{-1}(j) for j in reversed(bwd_path)]``
        where  π  is the node permutation relating the matched states.

        The ``use_edge_mult`` priority penalty  ``max(0, max|B_ij|-k)``
        deprioritises detours through quivers with edge multiplicity
        above ``edge_mult_threshold``.  Conventional BPS quivers keep
        ``|B_{ij}| ≤ 2``  along the correct negating sequence, so
        higher-multiplicity states are dead-end detours.

        Priority when edge-mult heuristic is active:
          ``(edge_mult_penalty, path_len)``
        else plain level-synchronized BFS.
        """
        from collections import deque
        import heapq

        n = self.n_nodes
        # Reflected endpoint: negate every mutable charge, keep B.
        neg_charges = [tuple(-x for x in self.charges[i]) if not self.frozen[i]
                        else tuple(self.charges[i]) for i in range(n)]
        Q_neg = BPSQuiver(neg_charges, self.frozen[:],
                          self.exchange, ambient_pairing=self.ambient_pairing)

        fwd_originals = originals
        back_originals = [tuple(-x for x in c) for c in originals]

        def multiset_key(q: "BPSQuiver") -> tuple:
            return tuple(sorted(q.charges[i] for i in mutable_indices))

        def find_permutation(q_f: "BPSQuiver", q_b: "BPSQuiver") -> dict | None:
            """Find a bijection  mutable node i → mutable node j  such
            that ``q_f.charges[i] == q_b.charges[j]`` for every i.
            Backtracking; ``mutable_indices`` is small in practice."""
            f_charges = {i: q_f.charges[i] for i in mutable_indices}
            b_charges = {j: q_b.charges[j] for j in mutable_indices}
            perm: dict[int, int] = {}
            used: set[int] = set()
            fidx = sorted(mutable_indices)

            def backtrack(pos: int) -> bool:
                if pos == len(fidx):
                    return True
                i = fidx[pos]
                for j in mutable_indices:
                    if j in used:
                        continue
                    if b_charges[j] == f_charges[i]:
                        perm[i] = j
                        used.add(j)
                        if backtrack(pos + 1):
                            return True
                        used.remove(j)
                        del perm[i]
                return False

            return perm if backtrack(0) else None

        def edge_mult_penalty(q: "BPSQuiver") -> int:
            m = 0
            for i in range(q.n_nodes):
                row = q.exchange[i]
                for j in range(q.n_nodes):
                    if i == j:
                        continue
                    a = abs(row[j])
                    if a > m:
                        m = a
            return max(0, m - edge_mult_threshold)

        def combine(fwd_q, fwd_path, bwd_q, bwd_path):
            perm = find_permutation(fwd_q, bwd_q)
            if perm is None:
                return None
            inv_perm = {v: k for k, v in perm.items()}
            tail = [inv_perm[j] for j in reversed(bwd_path)]
            return list(fwd_path) + tail

        fwd_visited: dict[tuple, tuple["BPSQuiver", list[int]]] = {
            multiset_key(self): (self, [])
        }
        bwd_visited: dict[tuple, tuple["BPSQuiver", list[int]]] = {
            multiset_key(Q_neg): (Q_neg, [])
        }

        if allow_permutation and multiset_key(self) == multiset_key(Q_neg):
            return []

        half = (max_depth + 1) // 2 + 1

        # Priority-based search (edge-multiplicity heuristic).
        if use_edge_mult:
            counter = 0
            init_prio = (0, 0)  # (edge_mult_penalty, path_len)
            fwd_heap = [(init_prio, counter, self, [])]
            counter += 1
            bwd_heap = [(init_prio, counter, Q_neg, [])]
            counter += 1

            while fwd_heap or bwd_heap:
                if fwd_heap:
                    _, _, cur, path = heapq.heappop(fwd_heap)
                    if len(path) < half:
                        for k in mutable_indices:
                            if not _in_positive_cone_int(cur.charges[k], fwd_originals):
                                continue
                            new_q = cur.mutate(k)
                            key = multiset_key(new_q)
                            if key in fwd_visited:
                                continue
                            new_path = path + [k]
                            fwd_visited[key] = (new_q, new_path)
                            if key in bwd_visited:
                                bwd_q, bwd_path = bwd_visited[key]
                                combined = combine(new_q, new_path, bwd_q, bwd_path)
                                if combined is not None:
                                    return combined
                            em = edge_mult_penalty(new_q)
                            prio = (em, len(new_path))
                            heapq.heappush(fwd_heap,
                                           (prio, counter, new_q, new_path))
                            counter += 1

                if bwd_heap:
                    _, _, cur, path = heapq.heappop(bwd_heap)
                    if len(path) < half:
                        for k in mutable_indices:
                            if not _in_positive_cone_int(cur.charges[k], back_originals):
                                continue
                            new_q = cur.reverse_mutate(k)
                            key = multiset_key(new_q)
                            if key in bwd_visited:
                                continue
                            new_path = path + [k]
                            bwd_visited[key] = (new_q, new_path)
                            if key in fwd_visited:
                                fwd_q, fwd_path = fwd_visited[key]
                                combined = combine(fwd_q, fwd_path, new_q, new_path)
                                if combined is not None:
                                    return combined
                            em = edge_mult_penalty(new_q)
                            prio = (em, len(new_path))
                            heapq.heappush(bwd_heap,
                                           (prio, counter, new_q, new_path))
                            counter += 1
            return None

        # Plain level-synchronized bidirectional BFS.
        fwd_queue = deque([(self, [])])
        bwd_queue = deque([(Q_neg, [])])

        for _ in range(half):
            next_fwd = []
            for cur, path in fwd_queue:
                if len(path) >= half:
                    continue
                for k in mutable_indices:
                    if not _in_positive_cone_int(cur.charges[k], fwd_originals):
                        continue
                    new_q = cur.mutate(k)
                    key = multiset_key(new_q)
                    if key in fwd_visited:
                        continue
                    fwd_visited[key] = (new_q, path + [k])
                    next_fwd.append((new_q, path + [k]))
                    if key in bwd_visited:
                        bwd_q, bwd_path = bwd_visited[key]
                        combined = combine(new_q, path + [k], bwd_q, bwd_path)
                        if combined is not None:
                            return combined
            fwd_queue = deque(next_fwd)

            next_bwd = []
            for cur, path in bwd_queue:
                if len(path) >= half:
                    continue
                for k in mutable_indices:
                    if not _in_positive_cone_int(cur.charges[k], back_originals):
                        continue
                    new_q = cur.reverse_mutate(k)
                    key = multiset_key(new_q)
                    if key in bwd_visited:
                        continue
                    bwd_visited[key] = (new_q, path + [k])
                    next_bwd.append((new_q, path + [k]))
                    if key in fwd_visited:
                        fwd_q, fwd_path = fwd_visited[key]
                        combined = combine(fwd_q, fwd_path, new_q, path + [k])
                        if combined is not None:
                            return combined
            bwd_queue = deque(next_bwd)

            if not fwd_queue and not bwd_queue:
                break

        return None

    def build_spectrum_generator(self, seq: Sequence[int]) -> list[tuple]:
        """Charges of the  E_q  factors along a negating sequence.

        Returns ``[g_1, ..., g_N]``  so that
        ``S = E_q(X_{g_1}) * ... * E_q(X_{g_N})`` .
        """
        charges_for_S = []
        current = self
        for k in seq:
            charges_for_S.append(tuple(current.charges[k]))
            current = current.mutate(k)
        return charges_for_S

    def verify_spectrum_generator(
        self,
        spec: Sequence[Sequence[int]],
        *,
        allow_permutation: bool = True,
    ) -> tuple[bool, dict]:
        """Verify that an ordered list of charges ``[g_1, ..., g_N]``
        defines a valid spectrum generator on this quiver.

        Simulates ``mutate`` step by step: at each step, finds the
        unfrozen node whose *current* charge equals  g_k , mutates
        there, and continues.  The spec is valid iff the simulation
        runs to completion and the terminating multiset of mutable
        charges matches the negation of the initial one (up to
        permutation when ``allow_permutation=True``; position-wise
        otherwise).

        This is the terminating condition of
        :meth:`find_negating_sequence` lifted to user-supplied specs
        -- use when building an  S  from a recipe (e.g. RG-flow
        matter dressing) and you want to check it's a genuine
        spectrum generator without searching for one.

        Returns ``(ok, info)`` where ``info`` contains ``steps``
        (mutations successfully simulated), ``missing_step``  (first
        step where no node has the required charge, or None), and
        ``initial`` / ``final`` / ``expected``  charge lists for
        debugging.
        """
        from collections import Counter
        initial = [tuple(c) for c in self.charges]
        frozen = list(self.frozen)
        current = self
        steps = 0
        missing = None
        for k, gk in enumerate(spec):
            gk_t = tuple(gk)
            idx = None
            for i, c in enumerate(current.charges):
                if not current.frozen[i] and tuple(c) == gk_t:
                    idx = i
                    break
            if idx is None:
                missing = (k, gk_t,
                           [tuple(c) for c in current.charges])
                break
            current = current.mutate(idx)
            steps += 1
        final = [tuple(c) for c in current.charges]
        neg_mutable = Counter(
            tuple(-x for x in a) for a, f in zip(initial, frozen) if not f
        )
        final_mutable = Counter(
            a for a, f in zip(final, frozen) if not f
        )
        frozen_ok = (
            missing is None
            and all(tuple(initial[i]) == tuple(final[i])
                    for i, f in enumerate(frozen) if f)
        )
        if allow_permutation:
            mutable_ok = (missing is None and final_mutable == neg_mutable)
        else:
            mutable_ok = (
                missing is None
                and all(
                    tuple(-x for x in initial[i]) == tuple(final[i])
                    for i, f in enumerate(frozen) if not f
                )
            )
        ok = frozen_ok and mutable_ok
        return ok, {
            "steps": steps,
            "missing_step": missing,
            "initial": initial,
            "final": final,
            "expected": neg_mutable,
            "frozen_match": frozen_ok,
        }

    def _charge_key(self) -> tuple:
        return tuple(tuple(c) for c in self.charges)

    def __repr__(self) -> str:
        lines = [f"BPSQuiver({self.n_nodes} nodes)"]
        for i, c in enumerate(self.charges):
            kind = "frozen" if self.frozen[i] else "mutable"
            lines.append(f"  [{i}] g = {c}  ({kind})")
        return "\n".join(lines)


def reflected_endpoint(Q: BPSQuiver) -> BPSQuiver:
    """Return a copy of ``Q`` with every unfrozen charge negated.

    This is the target state of a negating mutation sequence: a valid
    ``S``  takes ``Q``  to this state (up to permutation of mutable
    nodes).  Useful as the ``target``  argument of
    :func:`find_mutation_path`  when the caller wants to search for
    head/tail-constrained negating sequences.
    """
    neg = [tuple(-x for x in Q.charges[i]) if not Q.frozen[i]
           else tuple(Q.charges[i]) for i in range(Q.n_nodes)]
    return BPSQuiver(neg, Q.frozen[:], Q.exchange,
                      ambient_pairing=Q.ambient_pairing)


def _apply_mutation_seq(Q: BPSQuiver, seq: Sequence[int]) -> BPSQuiver:
    for k in seq:
        Q = Q.mutate(k)
    return Q


def _apply_reverse_mutation_seq(Q: BPSQuiver, seq: Sequence[int]) -> BPSQuiver:
    for k in reversed(list(seq)):
        Q = Q.reverse_mutate(k)
    return Q


def _full_charge_key(Q: BPSQuiver) -> tuple:
    return tuple(tuple(c) for c in Q.charges)


def _mutable_multiset_key(Q: BPSQuiver, mutable_indices: list[int]) -> tuple:
    return tuple(sorted(Q.charges[i] for i in mutable_indices))


def _mp_edge_mult_penalty(Q: BPSQuiver, threshold: int) -> int:
    m = 0
    for row in Q.exchange:
        for v in row:
            a = abs(v)
            if a > m:
                m = a
    return max(0, m - threshold)


def _mp_find_permutation(q_f: BPSQuiver, q_b: BPSQuiver,
                          mutable_indices: list[int]) -> dict[int, int] | None:
    """Find bijection mutable i -> mutable j with q_f.charges[i] == q_b.charges[j]."""
    f_charges = {i: q_f.charges[i] for i in mutable_indices}
    b_charges = {j: q_b.charges[j] for j in mutable_indices}
    perm: dict[int, int] = {}
    used: set[int] = set()
    fidx = sorted(mutable_indices)

    def backtrack(pos: int) -> bool:
        if pos == len(fidx):
            return True
        i = fidx[pos]
        for j in mutable_indices:
            if j in used:
                continue
            if b_charges[j] == f_charges[i]:
                perm[i] = j
                used.add(j)
                if backtrack(pos + 1):
                    return True
                used.remove(j)
                del perm[i]
        return False

    return perm if backtrack(0) else None


def find_mutation_path(
    Q_start: BPSQuiver,
    Q_end: BPSQuiver,
    *,
    head: Sequence[int] = (),
    tail: Sequence[int] = (),
    max_depth: int = 30,
    allow_permutation: bool = False,
    cone_fwd: Sequence[Sequence[int]] | None = None,
    cone_bwd: Sequence[Sequence[int]] | None = None,
    edge_mult_threshold: int = 2,
    use_edge_mult: bool = True,
) -> list[int] | None:
    """Bidirectional BFS for a mutation path from ``Q_start``  to ``Q_end`` .

    Generalises :meth:`BPSQuiver.find_negating_sequence`  with three
    extra degrees of freedom: (a) the endpoint is a user-supplied
    quiver rather than the reflected endpoint, (b) a fixed ``head``
    and/or ``tail``  can be supplied, (c) matching can be by full
    charge vector (``allow_permutation=False``) or by multiset of
    mutable charges (``allow_permutation=True`` ; the tail is then
    relabelled by the discovered permutation, mirroring the existing
    bidirectional S-finder).

    Parameters
    ----------
    Q_start, Q_end
        Same number of nodes and same frozen pattern.
    head, tail
        Fixed prefix / suffix of the returned path.  The search only
        looks for the middle.  ``len(head) + len(tail) + middle ≤ max_depth``.
    max_depth
        Total path length cap (head + middle + tail).
    allow_permutation
        If True, match on multiset of mutable charges; the tail is
        relabelled by the permutation witness at the meet point.
    cone_fwd, cone_bwd
        Optional positive-cone constraints on forward / backward BFS.
        ``cone_fwd``  restricts forward-search mutations to nodes whose
        *current* charge lies in the non-negative cone of ``cone_fwd`` ;
        ``cone_bwd``  does the same for backward (reverse) mutations
        relative to ``cone_bwd``.  Pass ``None``  to disable.  The
        standard spectrum-generator search uses the mutable
        ``Q_start.charges``  and their negation, respectively.
    edge_mult_threshold, use_edge_mult
        Edge-multiplicity priority heuristic (same as
        :meth:`BPSQuiver.find_negating_sequence`).

    Returns
    -------
    The full path ``list(head) + middle + list(tail)``  (or a
    relabelled ``tail``  when ``allow_permutation=True``) or ``None``
    if no path within ``max_depth``.

    Examples
    --------
    Find a full negating sequence (reproduces the standard S-finder)::

        Q = BPSQuiver.from_pairing(nodes, B)
        S = find_mutation_path(
                Q, reflected_endpoint(Q),
                allow_permutation=True,
                cone_fwd=[c for c, f in zip(Q.charges, Q.frozen) if not f],
                cone_bwd=[tuple(-x for x in c) for c, f
                           in zip(Q.charges, Q.frozen) if not f],
            )

    Find ``S``  that ends with a known ``s_sub``  (e.g. after freezing a
    node and solving the sub-quiver)::

        S = find_mutation_path(Q, reflected_endpoint(Q),
                                tail=s_sub, allow_permutation=True, ...)

    Find a state-matching path to a specific Q_target::

        S = find_mutation_path(Q, Q_target)  # allow_permutation=False
    """
    from collections import deque
    import heapq

    if Q_start.n_nodes != Q_end.n_nodes:
        raise ValueError("Q_start and Q_end must have the same number of nodes")
    if Q_start.frozen != Q_end.frozen:
        raise ValueError("Q_start and Q_end must have the same frozen pattern")

    mutable_indices = [i for i, f in enumerate(Q_start.frozen) if not f]

    fwd_root = _apply_mutation_seq(Q_start, head)
    bwd_root = _apply_reverse_mutation_seq(Q_end, tail)

    middle_budget = max_depth - len(head) - len(tail)
    if middle_budget < 0:
        return None

    if allow_permutation:
        def key(Q):
            return _mutable_multiset_key(Q, mutable_indices)
    else:
        def key(Q):
            return _full_charge_key(Q)

    if key(fwd_root) == key(bwd_root):
        if allow_permutation:
            perm = _mp_find_permutation(fwd_root, bwd_root, mutable_indices)
            if perm is None:
                return None
            inv = {v: k for k, v in perm.items()}
            return list(head) + [inv[j] for j in tail]
        return list(head) + list(tail)

    def combine(mid_fwd_path: list[int], mid_bwd_path: list[int],
                 fwd_q: BPSQuiver, bwd_q: BPSQuiver) -> list[int] | None:
        if not allow_permutation:
            return (list(head) + list(mid_fwd_path)
                    + list(reversed(mid_bwd_path)) + list(tail))
        perm = _mp_find_permutation(fwd_q, bwd_q, mutable_indices)
        if perm is None:
            return None
        inv = {v: k for k, v in perm.items()}
        return (list(head) + list(mid_fwd_path)
                + [inv[j] for j in reversed(mid_bwd_path)]
                + [inv[j] for j in tail])

    fwd_visited: dict[tuple, tuple[BPSQuiver, list[int]]] = {
        key(fwd_root): (fwd_root, [])
    }
    bwd_visited: dict[tuple, tuple[BPSQuiver, list[int]]] = {
        key(bwd_root): (bwd_root, [])
    }
    half = (middle_budget + 1) // 2 + 1

    if use_edge_mult:
        counter = 0
        fwd_heap: list = [((0, 0), counter, fwd_root, [])]
        counter += 1
        bwd_heap: list = [((0, 0), counter, bwd_root, [])]
        counter += 1

        while fwd_heap or bwd_heap:
            if fwd_heap:
                _, _, cur, path = heapq.heappop(fwd_heap)
                if len(path) < half:
                    for k in mutable_indices:
                        if cone_fwd is not None and not _in_positive_cone_int(
                                cur.charges[k], cone_fwd):
                            continue
                        nxt = cur.mutate(k)
                        ck = key(nxt)
                        if ck in fwd_visited:
                            continue
                        new_path = path + [k]
                        fwd_visited[ck] = (nxt, new_path)
                        if ck in bwd_visited:
                            bwd_q, bwd_path = bwd_visited[ck]
                            c = combine(new_path, bwd_path, nxt, bwd_q)
                            if c is not None:
                                return c
                        em = _mp_edge_mult_penalty(nxt, edge_mult_threshold)
                        heapq.heappush(fwd_heap,
                                        ((em, len(new_path)), counter, nxt, new_path))
                        counter += 1

            if bwd_heap:
                _, _, cur, path = heapq.heappop(bwd_heap)
                if len(path) < half:
                    for k in mutable_indices:
                        if cone_bwd is not None and not _in_positive_cone_int(
                                cur.charges[k], cone_bwd):
                            continue
                        nxt = cur.reverse_mutate(k)
                        ck = key(nxt)
                        if ck in bwd_visited:
                            continue
                        new_path = path + [k]
                        bwd_visited[ck] = (nxt, new_path)
                        if ck in fwd_visited:
                            fwd_q, fwd_path = fwd_visited[ck]
                            c = combine(fwd_path, new_path, fwd_q, nxt)
                            if c is not None:
                                return c
                        em = _mp_edge_mult_penalty(nxt, edge_mult_threshold)
                        heapq.heappush(bwd_heap,
                                        ((em, len(new_path)), counter, nxt, new_path))
                        counter += 1
        return None

    # Plain level-synchronised BFS.
    fwd_queue = deque([(fwd_root, [])])
    bwd_queue = deque([(bwd_root, [])])
    for _ in range(half):
        nxt_fwd: list = []
        for cur, path in fwd_queue:
            if len(path) >= half:
                continue
            for k in mutable_indices:
                if cone_fwd is not None and not _in_positive_cone_int(
                        cur.charges[k], cone_fwd):
                    continue
                nxt = cur.mutate(k)
                ck = key(nxt)
                if ck in fwd_visited:
                    continue
                fwd_visited[ck] = (nxt, path + [k])
                if ck in bwd_visited:
                    bwd_q, bwd_path = bwd_visited[ck]
                    c = combine(path + [k], bwd_path, nxt, bwd_q)
                    if c is not None:
                        return c
                nxt_fwd.append((nxt, path + [k]))
        fwd_queue = deque(nxt_fwd)

        nxt_bwd: list = []
        for cur, path in bwd_queue:
            if len(path) >= half:
                continue
            for k in mutable_indices:
                if cone_bwd is not None and not _in_positive_cone_int(
                        cur.charges[k], cone_bwd):
                    continue
                nxt = cur.reverse_mutate(k)
                ck = key(nxt)
                if ck in bwd_visited:
                    continue
                bwd_visited[ck] = (nxt, path + [k])
                if ck in fwd_visited:
                    fwd_q, fwd_path = fwd_visited[ck]
                    c = combine(fwd_path, path + [k], fwd_q, nxt)
                    if c is not None:
                        return c
                nxt_bwd.append((nxt, path + [k]))
        bwd_queue = deque(nxt_bwd)
        if not fwd_queue and not bwd_queue:
            break
    return None


def find_spectrum_generator(
    pairing: Sequence[Sequence[int]],
    node_charges: Sequence[Sequence[int]],
    frozen: Sequence[bool] | None = None,
    max_depth: int = 20,
) -> tuple[list[tuple], list[int]]:
    """Convenience wrapper: given pairing + node charges, return
    ``(spec, negating_sequence)``  or raise if none is found."""
    quiver = BPSQuiver.from_pairing(node_charges, pairing, frozen)
    seq = quiver.find_negating_sequence(max_depth=max_depth)
    if seq is None:
        raise ValueError("could not find a negating mutation sequence")
    spec = quiver.build_spectrum_generator(seq)
    return spec, seq


def alt_spec_with_head(
    Q: "BPSQuiver",
    spec: Sequence[Sequence[int]],
    k: int,
    *,
    max_iterations: int = 200,
    use_edge_mult: bool = True,
) -> tuple[tuple, ...] | None:
    """Bubble ``Q.charges[k]``  to the head of ``spec``  using local
    moves only: adjacent commuting swaps (pairing 0) and pentagon
    expansions (pairing ±1).

    The intended use is preparing a chart-mutation: ``mutate(k)`` on a
    chart with finite ``S``  yields a finite-``S`` child iff some
    factorisation of  S  starts with ``Q.charges[k]`` .  Since node
    charges always appear as factors in  S , the question reduces to
    whether the target factor can be moved to position 0 by adjacent
    rewrites that preserve  S  as an operator.

    Algorithm
    ---------
    Locate the target factor; while it is not at position 0, rewrite
    its left neighbour pair  ``[a, target]``  in place:

    * ``<a, target> == 0``  →  commuting swap, length unchanged.
    * ``|<a, target>| == 1`` →  pentagon expansion, length grows by 1
      (the composite charge ``a + target``  is inserted; the original
      neighbour reappears one slot to the right).

    A pair with ``|<a, target>| ≥ 2``  has no length-bounded local
    move, so the bubble blocks and we return ``None`` .  Empirically
    this is the only obstruction: SU(2)-pure node 1 and SU(3)-pure
    nodes 1, 3 fail here (chambers without a finite-``S``  child).

    Each rewrite is realised by a single ``find_mutation_path``  call
    on the 2-step subproblem  ``Q_{i-1} → Q_{i+1}``  with the new head
    fixed; this lets ``BPSQuiver``  do the bookkeeping (mutation-index
    relabelling, etc.) instead of re-deriving the pentagon
    coordinate-by-coordinate.

    Parameters
    ----------
    Q
        Starting BPS quiver; ``Q.charges``  are the current node
        charges.
    spec
        Current factorisation of  S  (charge tuples in ``Q`` 's
        lattice).  ``Q.charges[k]``  must appear somewhere in ``spec`` .
    k
        Index of the unfrozen node whose charge should be bubbled to
        the head.
    max_iterations
        Safety cap on the number of bubble steps.  Worst-case bound
        is ``len(spec) - 1`` (one step per leftward shift), so 200 is
        comfortable for any quiver we have hit.

    Returns
    -------
    The rewritten spec, or ``None``  if a pairing-≥ 2 pair blocks the
    bubble.
    """
    if Q.frozen[k]:
        raise ValueError(f"node {k} is frozen")
    target = tuple(int(x) for x in Q.charges[k])
    cur_spec = [tuple(int(x) for x in g) for g in spec]

    B = Q.ambient_pairing
    if B is None:
        raise ValueError(
            "Q.ambient_pairing must be set; pentagon rewrites use "
            "the lattice pairing on charges"
        )
    n = len(B)

    def bracket(g1: tuple, g2: tuple) -> int:
        return sum(g1[a] * B[a][b] * g2[b]
                   for a in range(n) for b in range(n))

    for _ in range(max_iterations):
        try:
            i = cur_spec.index(target)
        except ValueError:
            return None  # target absent from spec
        if i == 0:
            return tuple(cur_spec)

        prev = cur_spec[i - 1]
        p = bracket(prev, target)
        if p == 0:
            # Adjacent commute:  E(prev) E(target) = E(target) E(prev) .
            cur_spec[i - 1], cur_spec[i] = cur_spec[i], cur_spec[i - 1]
        elif p == 1:
            # Pentagon expand:  E(prev) E(target) =
            #     E(target) E(prev + target) E(prev)   for  <prev, target> = +1 .
            composite = tuple(prev[a] + target[a] for a in range(n))
            cur_spec = (
                cur_spec[:i - 1]
                + [target, composite, prev]
                + cur_spec[i + 1:]
            )
        elif p == -1:
            # Pentagon collapse:  if spec[i-2:i+1]  matches the RHS
            # pattern  [b, a+b, a]  with a = target , b = spec[i-2] ,
            # and  <a, b> = +1 , collapse to  [a, b]  --  target moves
            # leftward by 2 and length drops by 1.
            if i >= 2:
                b = cur_spec[i - 2]
                ab = tuple(b[a] + target[a] for a in range(n))
                if ab == prev and bracket(target, b) == 1:
                    cur_spec = (
                        cur_spec[:i - 2]
                        + [target, b]
                        + cur_spec[i + 1:]
                    )
                    continue
            return None  # pairing -1 with no collapse pattern
        else:
            # |<prev, target>| >= 2 has no length-bounded pentagon
            # rewrite.
            return None
    return None


def alt_spec_with_tail(
    Q: "BPSQuiver",
    spec: Sequence[Sequence[int]],
    k: int,
    *,
    max_iterations: int = 200,
) -> tuple[tuple, ...] | None:
    """Bubble  ``Q.charges[k]``  to the tail of ``spec``  using local
    moves only -- the tail-side mirror of :func:`alt_spec_with_head` .

    Useful for inverse necklace steps: once the target factor is at
    the END of the spec, an inverse mutation at node ``k``  consumes
    it and the remainder is the child chart's spec (with appropriate
    tail-side necklace shift).

    Algorithm (mirror of :func:`alt_spec_with_head` )
    -------------------------------------------------
    Locate target's position; while it is not at the last slot,
    rewrite the pair  ``[target, next]``  at positions  ``[i, i+1]`` :

    * ``<target, next> == 0``  →  commute, length unchanged.
    * ``<target, next> == +1`` →  pentagon expand
      ``E(target) E(next) = E(next) E(target+next) E(target)`` ;
      target moves from  i  to  i+2 , length +1.
    * ``<target, next> == -1`` →  pentagon collapse if
      ``spec[i:i+3]``  matches the RHS pattern  ``[b, a+b, a]``  with
      ``b = target`` ,  ``a = spec[i+2]`` ,  ``<a, b> = +1`` ; collapse
      to  ``[a, b]`` , target moves from  i  to  i+1 , length -1.
      Otherwise stuck.
    * ``|<target, next>| >= 2`` →  stuck.

    Returns the rewritten spec, or ``None``  on a stuck pair.
    """
    if Q.frozen[k]:
        raise ValueError(f"node {k} is frozen")
    target = tuple(int(x) for x in Q.charges[k])
    cur_spec = [tuple(int(x) for x in g) for g in spec]

    B = Q.ambient_pairing
    if B is None:
        raise ValueError(
            "Q.ambient_pairing must be set; pentagon rewrites use "
            "the lattice pairing on charges"
        )
    n = len(B)

    def bracket(g1: tuple, g2: tuple) -> int:
        return sum(g1[a] * B[a][b] * g2[b]
                   for a in range(n) for b in range(n))

    for _ in range(max_iterations):
        # Rightmost occurrence: tail-bubble may pass through duplicate
        # target factors (rare for node charges, but defensive).
        i = None
        for j in range(len(cur_spec) - 1, -1, -1):
            if cur_spec[j] == target:
                i = j
                break
        if i is None:
            return None
        if i == len(cur_spec) - 1:
            return tuple(cur_spec)

        nxt = cur_spec[i + 1]
        p = bracket(target, nxt)
        if p == 0:
            cur_spec[i], cur_spec[i + 1] = cur_spec[i + 1], cur_spec[i]
        elif p == 1:
            # Pentagon expand:  E(target) E(next) =
            #   E(next) E(target+next) E(target)   for  <target, next> = +1 .
            composite = tuple(target[a] + nxt[a] for a in range(n))
            cur_spec = (
                cur_spec[:i]
                + [nxt, composite, target]
                + cur_spec[i + 2:]
            )
        elif p == -1:
            # Pentagon collapse:  spec[i:i+3] = [b, a+b, a]  with
            # b = target , a = spec[i+2] , and  <a, b> = +1  collapses
            # to  [a, b]  -- target moves rightward by 1, length -1.
            if i + 2 < len(cur_spec):
                a = cur_spec[i + 2]
                ab = tuple(a[c] + target[c] for c in range(n))
                if ab == nxt and bracket(a, target) == 1:
                    cur_spec = (
                        cur_spec[:i]
                        + [a, target]
                        + cur_spec[i + 3:]
                    )
                    continue
            return None
        else:
            return None
    return None


# =====================================================================
# SECTION 7 -- commute_F_across (F^(i)_a), CoulombAlgebra, presets, demo
# =====================================================================
# --- §7a : commute_F_across (F^(i)_a intermediates) ------------------
# The canonical element  F_a  satisfies the intertwining identity
#
#     F_a * S  =  S * rho_Q(F_{sigma(a)})
#
# which can be unwrapped into a sequence of partial commutations:
#
#     F_a    * E_q(X_{g_1})  =  E_q(X_{g_1}) * F^{(1)}_a
#     F^{(1)}_a * E_q(X_{g_2})  =  E_q(X_{g_2}) * F^{(2)}_a
#     ...
#     F^{(N)}_a  =  rho_Q(F_{sigma(a)}).
#
# The intermediate element  F^{(i)}_a  is obtained by pushing  F^{(i-1)}_a
# through  E_q(X_{g_i})  from left to right -- exactly what ``solve`` does.
# The "backward" direction (pushing from right to left) uses ``solve_inverse``
# and proceeds  g_N, g_{N-1}, ... .

def commute_F_across(
    F: dict[Vec, LaurentPoly],
    lattice: Lattice,
    spec: Sequence[Sequence[int]],
    i: int,
    *,
    direction: str = "forward",
    collect_intermediates: bool = False,
) -> dict[Vec, LaurentPoly] | list[dict[Vec, LaurentPoly]]:
    """Commute ``F`` across the first  ``i``  factors of  S .

    Parameters
    ----------
    F         : a finite element of  Q_Gamma  (e.g. a canonical  F_a ).
    lattice   : ambient lattice.
    spec      : the ordered list of spectrum charges  [g_1, ..., g_N] .
    i         : number of  E_q  factors to cross  (0 <= i <= len(spec)) .
    direction :
        "forward"  : solve  F^{(k-1)} * E_q(X_{g_k})  =  E_q(X_{g_k}) * F^{(k)}
                     for  k = 1, ..., i   (left-to-right).
        "backward" : solve  E_q(X_{g_{N-k+1}}) * F^{(k)}  =  F^{(k-1)} * E_q(X_{g_{N-k+1}})
                     for  k = 1, ..., i   (right-to-left, using solve_inverse).
    collect_intermediates :
        if True, return the list  [F^{(0)}, F^{(1)}, ..., F^{(i)}] ;
        otherwise return only  F^{(i)} .

    Raises
    ------
    ValueError  if any intermediate fails to be finitely conjugable
    (which should not happen for genuine canonical  F_a ).
    """
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be 'forward' or 'backward'")
    spec_t = [lattice.check(g) for g in spec]
    N = len(spec_t)
    if not (0 <= i <= N):
        raise ValueError(f"i must be in [0, {N}], got {i}")

    current: dict[Vec, LaurentPoly] = {g: c for g, c in F.items()
                                       if not c.is_zero()}
    trail: list[dict[Vec, LaurentPoly]] = [dict(current)]

    if direction == "forward":
        order = spec_t[:i]
    else:
        order = list(reversed(spec_t))[:i]

    for idx, gk in enumerate(order, start=1):
        try:
            if direction == "forward":
                current = solve(current, lattice, gk)
            else:
                current = solve_inverse(current, lattice, gk)
        except ValueError as e:
            raise ValueError(
                f"commute_F_across: step {idx} across E_q(X_{gk}) failed: {e}"
            ) from e
        trail.append(dict(current))

    return trail if collect_intermediates else current

# --- §7b : CoulombAlgebra helpers (validation, persistence) -----------

def _verify_pointed_cone(
    rank: int,
    gens: Sequence[Sequence[int]],
    *,
    witness: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Check that the positive cone generated by ``gens`` is **pointed**,
    and return a strict integer witness  f  with  f(g) >= 1  for every
    generator  g .

    Pointed means: no non-trivial non-negative real combination of
    generators sums to zero.  Equivalently, there exists a linear
    functional  f : Gamma -> R  with  f(g) > 0  for every generator  g .
    By density, one can always find such an  f  with integer coordinates
    if the cone is pointed at all.

    This routine searches for an integer witness  f  with coordinates in
    a box whose size scales with the largest absolute entry of the
    generators.  A witness proves the cone is pointed.  If no witness is
    found in the box, we raise -- in practice, genuinely pointed cones
    built from small BPS-quiver charges admit witnesses well inside the
    search box.

    Returns
    -------
    tuple[int, ...]
        A strict witness  f  of length ``rank``  satisfying
        ``sum(f[i] * g[i]) >= 1``  for every  ``g``  in  ``gens`` .  An
        empty ``gens`` returns the zero vector vacuously.

    Raises
    ------
    ValueError
        If no witness is found.  The error message suggests the most
        common diagnoses: a generator equal to zero, two generators
        summing to zero, or a degenerate quiver.
    """
    gens = [tuple(int(x) for x in g) for g in gens]
    if not gens:
        return tuple([0] * rank)

    for g in gens:
        if all(x == 0 for x in g):
            raise ValueError(
                "Positive cone is not pointed: a generator equals zero."
            )

    # --- Supplied witness (fast path).  A caller like PureADE that
    # *knows* the cone is pointed (e.g. from a closed-form recipe)
    # can hand over a certificate; we verify it here in O(rank * n_gens)
    # and skip the search entirely.
    if witness is not None:
        w = tuple(int(x) for x in witness)
        if len(w) != rank:
            raise ValueError(
                f"supplied witness has length {len(w)}, expected rank={rank}"
            )
        if all(sum(wi * gi for wi, gi in zip(w, g)) >= 1 for g in gens):
            return w
        # else: witness fails; fall through to normal search (also
        # a sanity check on the caller's certificate).

    # --- Pre-reduction: drop ambient coordinates that are zero in
    # every generator.  When the ambient lattice contains "padding"
    # dimensions (e.g. U(1) slots in a PureADE with U(1) factors,
    # which have no quiver nodes of their own) those coords carry
    # zero information for the cone check but inflate the box-search
    # rank exponentially.  Projecting to the non-zero subspace is a
    # cheap, sound optimisation: a witness for the reduced problem
    # lifts back to one for the full ambient by padding with zeros.
    nonzero_coords = [k for k in range(rank)
                      if any(g[k] != 0 for g in gens)]
    if len(nonzero_coords) < rank:
        gens_proj = [tuple(g[k] for k in nonzero_coords) for g in gens]
        w_proj = _verify_pointed_cone(len(nonzero_coords), gens_proj)
        # Lift back to the full ambient by padding zero on dropped coords.
        w_full = [0] * rank
        for j, k in enumerate(nonzero_coords):
            w_full[k] = w_proj[j]
        return tuple(w_full)

    # --- Fast path: try a handful of cheap witness candidates before
    # the full box search.  For every "nice" positive cone in practice
    # -- i.e. all the theories in PRESETS and linear_quiver_presets --
    # one of these candidates succeeds and the cost drops from
    # O((4M+1)^rank)  to  O(rank * n_gens) .
    def _witnesses(f):
        return all(sum(fi * gi for fi, gi in zip(f, g)) >= 1 for g in gens)

    # Candidate 1: sum of generators (average direction).
    s = tuple(sum(g[i] for g in gens) for i in range(rank))
    if any(x != 0 for x in s) and _witnesses(s):
        return s
    # Candidate 2..n+1: each generator itself (only works if pairwise
    # pairings are non-negative with this choice, but check is cheap).
    for g in gens:
        if _witnesses(g):
            return g
    # Candidate n+2: signum of the sum -- gives a bounded-entry
    # alternative when the raw sum has tiny entries.
    sg = tuple((1 if x > 0 else -1 if x < 0 else 0) for x in s)
    if any(x != 0 for x in sg) and _witnesses(sg):
        return sg

    # Slow path: exhaustive box search (kept as safety net for exotic
    # cones -- still used in practice for the paper's SU(3) chambers).
    M = max(max(abs(x) for x in g) for g in gens)
    bound = max(3, 2 * M)
    for f in _iproduct(range(-bound, bound + 1), repeat=rank):
        if _witnesses(f):
            return tuple(f)

    # Exact LP fallback: the box can be too small (a near-antipodal generator
    # pair forces a witness coordinate outside [-bound, bound] -- e.g. a mutated
    # flavoured chamber needing a -4 entry).  `sigma_iso._lp_feasible_strict`
    # is a two-phase rational simplex that finds an integer witness iff the cone
    # is pointed, with no box (sound positive + sound negative by LP duality).
    # Mirrors the same fallback in bps_kalgebra_internals.compute_strict_cone_witness.
    from sigma_iso import _lp_feasible_strict
    feasible, w = _lp_feasible_strict(gens, rank)
    if feasible and w is not None and _witnesses(w):
        return tuple(int(x) for x in w)

    # No witness found -- the exact LP confirms the cone is not pointed.
    for i, gi in enumerate(gens):
        for j, gj in enumerate(gens):
            if j <= i:
                continue
            if all(a + b == 0 for a, b in zip(gi, gj)):
                raise ValueError(
                    f"Positive cone is not pointed: generators {gi} and "
                    f"{gj} are antipodal."
                )
    raise ValueError(
        f"Positive cone is not pointed: the exact LP strict-witness solve "
        f"(sigma_iso._lp_feasible_strict) found no f with f(g) > 0 for every "
        f"generator (box search to [-{bound},{bound}]^{rank} also empty).  "
        f"Check that the unfrozen node charges generate a strictly positive cone."
    )


def _canonical_inputs(
    pairing: Sequence[Sequence[int]],
    node_charges: Sequence[Sequence[int]],
    frozen: Sequence[bool],
) -> dict[str, Any]:
    """Deterministic JSON-serialisable representation of the inputs that
    fully determine a ``CoulombAlgebra``.  Used for cache signatures."""
    return {
        "pairing": [[int(x) for x in row] for row in pairing],
        "node_charges": [[int(x) for x in g] for g in node_charges],
        "frozen": [bool(f) for f in frozen],
    }


def _signature_hash(canonical: dict[str, Any]) -> str:
    blob = json.dumps(canonical, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _laurent_to_pairs(lp: "LaurentPoly") -> list[list[int]]:
    """Serialise a LaurentPoly to a sorted list of ``[exponent, coeff]`` pairs."""
    return [[int(e), int(c)] for e, c in sorted(lp._coeffs.items())]


def _laurent_from_pairs(pairs: Sequence[Sequence[int]]) -> "LaurentPoly":
    return LaurentPoly({int(e): int(c) for e, c in pairs})


# --- §7b : CoulombAlgebra driver class -------------------------------
class CoulombAlgebra:
    """End-to-end driver for the K_𝖖-algebra  A_q  of a BPS quiver.

    (The class name ``CoulombAlgebra`` is historical: the same machinery
    serves BPS-quiver realisations of general K_𝖖-algebras, most of them
    non-Lagrangian Argyres–Douglas theories, not only K-theoretic Coulomb
    branch algebras of conventional gauge theories.)

    Construction from a decorated BPS quiver  (B, node_charges, frozen) :

      1. Wrap  node_charges  into the lattice with pairing  B .
      2. Find a negating mutation sequence via ``BPSQuiver`` and read off
         the spectrum  S = E_q(X_{g_1}) * ... * E_q(X_{g_N}) .
      3. Cache canonical basis elements  F_a  on demand via ``solve_F`` .

    Public API:

      * ``F(gamma)``                 -- cached canonical basis element.
      * ``product(a, b)``            -- quantum-torus product  F_a * F_b .
      * ``multiply(a, b)``           -- F-basis decomposition  {c : C_{ab}^c(q)},
                                        i.e. the structure constants of  A_q .
      * ``commute_F_across(a, i)``   -- F^{(i)}_a : commute  F_a  through the
                                        first  i  factors of  S .
      * ``schur_index(a, b, K=...)`` -- Schur index  I_{a,b}(q)  via exact Nahm.
      * ``sigma(a)`` / ``sigma_orbit(a)``  -- tropical sigma on charges.
      * ``rho_Q(elt)``               -- canonical automorphism  X_g -> X_{-g} .
    """

    def __init__(
        self,
        pairing: Sequence[Sequence[int]],
        node_charges: Sequence[Sequence[int]],
        frozen: Sequence[bool] | None = None,
        *,
        cone_witness: Sequence[int] | None = None,
        negating_sequence: Sequence[int] | None = None,
        spec: Sequence[Sequence[int]] | None = None,
        skip_spec_cone_check: bool = False,
        skip_cone_check: bool = False,
    ):
        self.lattice = Lattice(pairing)
        self.node_charges = [self.lattice.check(g) for g in node_charges]
        self.frozen = list(frozen) if frozen else [False] * len(self.node_charges)

        self.quiver = BPSQuiver.from_pairing(
            self.node_charges, pairing, self.frozen
        )
        if spec is not None and negating_sequence is not None:
            raise ValueError(
                "pass either `spec` OR `negating_sequence`, not both"
            )
        if spec is not None:
            # User-provided spectrum-generator factorization.  This is
            # the "escape hatch" for theories where BFS on
            # ``find_negating_sequence``  times out but a physics-
            # informed S is known (e.g. the Wilson-line-chamber
            # amalgamation   S_{gauge+matter} = M(W) * S_{gauge} , where
            # M(W) is the matter contribution expanded in characters).
            #
            # We only do a *local* sanity check here: every factor
            # charge must lie in the non-negative cone spanned by the
            # unfrozen node charges (otherwise  X_gamma  is not even a
            # positive-cone element of the quantum torus).  The *global*
            # condition that the ordered product genuinely negates the
            # cone is non-local and not practically checkable without
            # re-running BFS on the spec -- users supplying  spec
            # directly are taking responsibility for it.  Orthonormality
            # of the resulting Schur matrix is the canonical downstream
            # cross-check  ``I_{a,b}[q^0] = delta_{ab}``.
            spec_t = [self.lattice.check(g) for g in spec]
            self.spec: list[tuple] = [tuple(g) for g in spec_t]
            self.negating_sequence = []  # empty: we didn't mutate through one
            if not skip_spec_cone_check:
                unfrozen = [
                    g for g, f in zip(self.node_charges, self.frozen) if not f
                ]
                for idx, g in enumerate(self.spec):
                    if not _cone_contains(g, unfrozen):
                        raise ValueError(
                            f"supplied spec[{idx}] = {g} is not in the "
                            f"positive cone spanned by the unfrozen node "
                            f"charges {unfrozen}"
                        )
        elif negating_sequence is not None:
            # User-provided sequence -- skip the expensive search.  We
            # still verify it actually negates all mutable charges, so
            # a stale cached sequence for a different theory fails
            # loudly rather than producing nonsense.
            seq = [int(k) for k in negating_sequence]
            Q = self.quiver
            for k in seq:
                Q = Q.mutate(k)
            expected = [
                tuple(-x for x in g) if not f else tuple(g)
                for g, f in zip(self.node_charges, self.frozen)
            ]
            got = [tuple(c) for c in Q.charges]
            if sorted(got) != sorted(expected):
                raise ValueError(
                    "supplied negating_sequence does not negate the "
                    "mutable node charges of this quiver"
                )
            self.negating_sequence = seq
            self.spec = self.quiver.build_spectrum_generator(seq)
        else:
            seq = self.quiver.find_negating_sequence()
            if seq is None:
                raise ValueError("could not find a negating mutation sequence")
            self.negating_sequence = seq
            self.spec = self.quiver.build_spectrum_generator(seq)
        self.cone_gens: list[tuple] = [
            g for g, f in zip(self.node_charges, self.frozen) if not f
        ]

        # Pointed-cone check on the unfrozen generators.  See
        # `_verify_pointed_cone` for details; raises on failure.
        # Caller can skip when they've verified the spec is a valid
        # negating sequence via  ``verify_spectrum_generator``  (which
        # implies the cone is pointed) and want to avoid the box
        # search that runs when ``cone_witness`` fails validation.
        #
        # The strict witness  f  with  <f, g> >= 1  on every cone gen
        # is also the linear functional that ``_find_lowest`` uses to
        # rank candidate cone-minima during F-basis decomposition.
        # The supplied  ``cone_witness``  is *not* required to be
        # strict (e.g. PureADE.U_N(2) hands over (1,1,0,0), which is
        # only >= 0 on the cone), so we always recompute via
        # ``_verify_pointed_cone``  -- which uses the supplied one as
        # a fast-path candidate but falls through to the search if it
        # fails the strict test.  When ``skip_cone_check=True``  the
        # caller has waived the pointedness check; we defer the
        # witness search to the first ``multiply``  call (lazily, via
        # ``_strict_cone_witness``), so the construction stays cheap.
        self._cone_witness: tuple[int, ...] | None = None
        if not skip_cone_check:
            self._cone_witness = _verify_pointed_cone(
                self.lattice.rank, self.cone_gens, witness=cone_witness,
            )
        else:
            # Stash the supplied witness; ``_strict_cone_witness``
            # validates / upgrades to a strict one on demand.
            self._cone_witness_supplied = (
                tuple(int(x) for x in cone_witness)
                if cone_witness is not None else None
            )

        # Gauge rank (for the Schur-index prefactor).
        self.gauge_rank = sum(1 for f in self.frozen if not f)

        # F cache.
        self._F_cache: dict[Vec, dict[Vec, LaurentPoly]] = {}

    def _strict_cone_witness(self) -> tuple[int, ...]:
        """Lazy strict witness  f  with  <f, g> >= 1  on every cone
        generator.  Used by ``_find_lowest``  to rank candidate
        cone-minima -- the early-break in that loop is only sound
        when  f  is *strictly* positive on the cone (otherwise charges
        sharing an  f -level can dominate each other, which is exactly
        the failure mode that hangs the algorithm on theories like
        pure  U(2)  whose cone generators are  f -orthogonal under the
        naive  f = (1, 1, ..., 1)  choice).
        """
        if self._cone_witness is not None:
            return self._cone_witness
        supplied = getattr(self, "_cone_witness_supplied", None)
        self._cone_witness = _verify_pointed_cone(
            self.lattice.rank, self.cone_gens, witness=supplied,
        )
        return self._cone_witness

    # --- convenience constructors -------------------------------------
    @classmethod
    def from_exchange_matrix(
        cls,
        B: Sequence[Sequence[int]],
        *,
        frozen: Sequence[bool] | None = None,
    ) -> "CoulombAlgebra":
        """Build a ``CoulombAlgebra`` with node charges = standard basis.

        For a bare BPS quiver there is no additional data beyond the
        exchange matrix  B  : the lattice is  Z^n  with pairing  B ,
        the node charges are the basis vectors  e_i , and  <e_i, e_j> =
        B_{ij}  by construction.  This cuts the construction boilerplate

            A = CoulombAlgebra(B, [(1, 0, ..., 0), (0, 1, ..., 0), ...])

        down to  A = CoulombAlgebra.from_exchange_matrix(B) .  Pass
        ``frozen`` to mark individual nodes as non-mutable (for
        flavoured chambers where a physicist wants to freeze a specific
        node -- note that the kernel of  B  is a separate concept, which
        the spectrum-generator finder handles transparently).
        """
        n = len(B)
        if any(len(row) != n for row in B):
            raise ValueError("exchange matrix must be square")
        nodes = [tuple(1 if j == i else 0 for j in range(n)) for i in range(n)]
        return cls(B, nodes, list(frozen) if frozen is not None else None)

    # --- canonical basis F_a ------------------------------------------
    def F(self, gamma: Sequence[int]) -> dict[Vec, LaurentPoly]:
        g = self.lattice.check(gamma)
        if g not in self._F_cache:
            self._F_cache[g] = solve_F(
                self.lattice, self.spec, self.cone_gens, g
            )
        return self._F_cache[g]

    # --- quantum-torus product ----------------------------------------
    def product(self, a: Sequence[int], b: Sequence[int]
                ) -> dict[Vec, LaurentPoly]:
        """Quantum-torus product  F_a * F_b  in  Q_Gamma  (not F-decomposed)."""
        return qt_multiply(self.F(a), self.F(b), self.lattice)

    # --- F-basis decomposition / structure constants ------------------
    def multiply(self, a: Sequence[int], b: Sequence[int],
                 *, max_steps: int = 500,
                 verbose: bool = False,
                 ) -> dict[Vec, LaurentPoly]:
        """Decompose  F_a * F_b  in the F-basis:

            F_a * F_b  =  sum_c  C_{ab}^c(q)  F_c.

        Returns the dict  ``{c : C_{ab}^c}``  of structure constants.

        The algorithm: repeatedly pick a charge  c  that is minimal
        (in the positive-cone order) among the remaining monomial
        charges of the product, record its coefficient as  C_{ab}^c,
        and subtract  C_{ab}^c * F_c  from the running element.  Since
        each step strictly reduces the lower boundary of the support,
        termination is guaranteed for finite F_a, F_b.
        """
        prod = self.product(a, b)
        result: dict[Vec, LaurentPoly] = {}

        for step in range(max_steps):
            prod = {g: c for g, c in prod.items() if not c.is_zero()}
            if not prod:
                break

            charges = list(prod.keys())
            lowest = self._find_lowest(charges)
            coeff = prod[lowest]
            if verbose:
                print(f"  step {step}: leading charge={lowest}  coeff={coeff}")

            if lowest in result:
                s = result[lowest] + coeff
                if s.is_zero():
                    del result[lowest]
                else:
                    result[lowest] = s
            else:
                result[lowest] = coeff

            # Subtract  coeff * F_lowest  from the running element.
            F_low = self.F(lowest)
            neg = LaurentPoly({e: -v for e, v in coeff._coeffs.items()})
            zero = tuple(0 for _ in range(self.lattice.rank))
            sub = qt_multiply({zero: neg}, F_low, self.lattice)
            for g, nc in sub.items():
                if g in prod:
                    s = prod[g] + nc
                    if s.is_zero():
                        del prod[g]
                    else:
                        prod[g] = s
                else:
                    prod[g] = nc
        else:
            remaining = {g: c for g, c in prod.items() if not c.is_zero()}
            if remaining:
                raise RuntimeError(
                    f"F-decomposition did not terminate after {max_steps} steps; "
                    f"{len(remaining)} residual terms remain."
                )

        return {g: c for g, c in result.items() if not c.is_zero()}

    def _find_lowest(self, charges: list[Vec]) -> Vec:
        """Pick a charge minimal in the positive-cone partial order.

        Sorts ``charges`` in ascending order of  ``L(c) = <f, c>`` ,
        where  ``f``  is a strict cone witness ( ``<f, g> >= 1``  on
        every cone generator).  Strict positivity of  ``f``  on the
        cone makes  ``L``  strictly monotone in the cone partial
        order, so any potential dominator  ``other``  of  ``cand``
        satisfies  ``L(other) < L(cand)`` , and the inner loop can
        early-break once  ``L(other) >= L(cand)``.

        The previous version used  ``L(c) = sum(c_i)`` , which is
        strictly positive on the cone of every preset whose node
        charges sit in a positive orthant -- but fails on theories
        like pure  U(2)  where both cone generators have zero
        component sum.  In that regime the early-break terminates
        immediately on every tie in  ``sum``  and the routine
        misidentifies non-minimal charges as minima, which sends
        ``multiply``  into a non-terminating loop along the cone
        ray.  Switching to a strict witness fixes this for any
        pointed cone.
        """
        cone_t = [self.lattice.check(g) for g in self.cone_gens]
        f = self._strict_cone_witness()

        def L_of(c):
            return sum(fi * ci for fi, ci in zip(f, c))

        # Index by key  (L, tiebreak_tuple)  for a stable order.
        ranked = sorted(charges, key=lambda c: (L_of(c), c))
        L = {c: L_of(c) for c in ranked}
        for cand in ranked:
            Lc = L[cand]
            is_lowest = True
            for other in ranked:
                if other is cand:
                    continue
                # Potential dominators have strictly smaller L; stop
                # the inner loop once we reach our own L-level.
                if L[other] >= Lc:
                    break
                diff = tuple(c - o for c, o in zip(cand, other))
                if _cone_contains(diff, cone_t):
                    is_lowest = False
                    break
            if is_lowest:
                return cand
        return min(charges)

    # --- F^(i)_a : push F_a through a prefix of S ---------------------
    def commute_F_across(
        self,
        a: Sequence[int],
        i: int,
        *,
        direction: str = "forward",
        collect_intermediates: bool = False,
    ):
        """F^{(i)}_a from commuting F_a across the first  i  factors of  S .

        See the module-level ``commute_F_across`` for full semantics.
        """
        return commute_F_across(
            self.F(a), self.lattice, self.spec, i,
            direction=direction,
            collect_intermediates=collect_intermediates,
        )

    # --- Schur index via exact Nahm -----------------------------------
    def schur_index(
        self,
        a: Sequence[int] | None = None,
        b: Sequence[int] | None = None,
        K: int = 20,
        *,
        r: int | None = None,
        cone_cutoff: int | None = None,
        K_internal: int | None = None,
    ) -> PowerSeries:
        """``I_{a,b}(q)`` to order ``q^K``.  ``a=None`` means vacuum."""
        if r is None:
            r = self.gauge_rank
        if K_internal is None:
            K_internal = 3 * K + 30
        if cone_cutoff is None:
            cone_cutoff = K + 5
        F_a = self.F(a) if a is not None else None
        F_b = self.F(b) if b is not None else None
        return schur_index_nahm(
            self.lattice, self.spec,
            F_a=F_a, F_b=F_b, K=K, r=r,
            cone_cutoff=cone_cutoff, K_internal=K_internal,
        )

    # --- sigma utilities ----------------------------------------------
    def sigma(self, gamma: Sequence[int]) -> Vec:
        return sigma(self.lattice, self.spec, gamma)

    def sigma_orbit(self, gamma: Sequence[int],
                    max_period: int = 50) -> list[Vec]:
        """Iterate ``sigma`` up to a return, or ``max_period`` if infinite."""
        g = self.lattice.check(gamma)
        orbit = [g]
        current = g
        for _ in range(max_period):
            current = self.sigma(current)
            if current == g:
                break
            orbit.append(current)
        return orbit

    # --- canonical automorphism ---------------------------------------
    @staticmethod
    def rho_Q(elt: dict[Vec, LaurentPoly]) -> dict[Vec, LaurentPoly]:
        return rho_Q(elt)

    # --- alt-spec for chart navigation --------------------------------
    def alt_spec_with_head(
        self,
        k: int,
        *,
        max_iterations: int = 200,
    ) -> list[tuple] | None:
        """Bubble  ``self.node_charges[k]``  to the head of the spec
        via local commute / pentagon moves, for use before mutating at
        node ``k``  of a parent chart.

        Thin wrapper around :func:`alt_spec_with_head` ; see that
        function for the full semantics.  Returns ``None``  when a
        pairing-≥2 pair blocks the bubble.
        """
        alt = alt_spec_with_head(
            self.quiver, self.spec, k, max_iterations=max_iterations,
        )
        if alt is None:
            return None
        return [tuple(g) for g in alt]

    # --- persistence --------------------------------------------------
    def _canonical_inputs(self) -> dict[str, Any]:
        return _canonical_inputs(
            self.lattice.pairing, self.node_charges, self.frozen
        )

    def signature(self) -> str:
        """SHA-256 hex hash of the inputs that fully determine this algebra.

        Two ``CoulombAlgebra`` instances with equal signatures have the
        same lattice, node charges, and frozen flags -- and hence the
        same negating sequence, spec, and F-basis.  Used by
        ``save`` / ``load_cache`` to refuse to load a cache built for a
        different theory.
        """
        return _signature_hash(self._canonical_inputs())

    def save(self, path: str) -> None:
        """Write negating sequence, spec, and F-cache to ``path`` as JSON.

        The file is keyed by a hash of the inputs (``signature()``), so
        ``load_cache`` refuses to populate an algebra whose inputs differ.
        JSON (not pickle) so the file is human-inspectable and safe to
        load from untrusted sources.
        """
        data = {
            "format": "bps_quiver_tools.CoulombAlgebra/v1",
            "signature": self.signature(),
            "inputs": self._canonical_inputs(),
            "negating_sequence": [int(k) for k in self.negating_sequence],
            "spec": [[int(x) for x in c] for c in self.spec],
            "F_cache": [
                {
                    "gamma": [int(x) for x in g],
                    "terms": [
                        [[int(x) for x in d], _laurent_to_pairs(c)]
                        for d, c in F.items()
                    ],
                }
                for g, F in self._F_cache.items()
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_cache(self, path: str, *, strict: bool = True) -> int:
        """Populate the F-cache from a file written by ``save``.

        Returns the number of F-entries loaded.  Raises ``ValueError``
        if the signature does not match this algebra's inputs (i.e. the
        file was saved for a different theory), unless ``strict=False``,
        in which case the mismatch is ignored and entries are loaded
        anyway -- caller's responsibility.

        Only the F-cache is populated.  The negating sequence and spec
        are re-derived on construction and not overwritten (they are
        deterministic given the inputs; the file stores them for
        diagnostics).
        """
        with open(path) as f:
            data = json.load(f)
        if strict and data.get("signature") != self.signature():
            raise ValueError(
                f"load_cache: signature mismatch at {path}.  The cache "
                f"was built for a different CoulombAlgebra (different "
                f"pairing, node_charges, or frozen flags).  Pass "
                f"strict=False to override at your own risk."
            )
        count = 0
        for entry in data.get("F_cache", []):
            g = tuple(int(x) for x in entry["gamma"])
            F: dict[Vec, LaurentPoly] = {}
            for d_raw, c_raw in entry["terms"]:
                d = tuple(int(x) for x in d_raw)
                F[d] = _laurent_from_pairs(c_raw)
            self._F_cache[g] = F
            count += 1
        return count

    def __repr__(self) -> str:
        return (f"CoulombAlgebra(rank={self.lattice.rank}, "
                f"nodes={len(self.node_charges)}, "
                f"|spec|={len(self.spec)}, "
                f"gauge_rank={self.gauge_rank})")

# --- §7c : preset theories and demo __main__ -------------------------
# A small catalogue of tested theories.  Each entry provides the pairing
# matrix  B  (which doubles as the exchange matrix of the quiver, since
# node charges are the standard basis in these examples) and the list
# of node charges.  Use as
#
#     spec = PRESETS["pentagon"]
#     A = CoulombAlgebra(spec["B"], spec["nodes"], spec.get("frozen"))
#
# The PRESETS dict below is the full catalogue.

PRESETS: dict[str, dict] = {
    "pentagon": {
        "name": "Pentagon  [A1, A2]  Argyres-Douglas",
        "B":     [[0, 1], [-1, 0]],
        "nodes": [(1, 0), (0, 1)],
    },
    "su2": {
        "name": "Pure SU(2)  Seiberg-Witten",
        "B":     [[0, 1], [-1, 0]],
        "nodes": [(1, 0), (-1, 2)],
    },
    "hexagon": {
        "name": "Hexagon  [A1, A3]  Argyres-Douglas (flavoured)",
        "B":     [[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
        "nodes": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    },
    "su3": {
        "name": "Pure SU(3)",
        "B":     [[0, 2, 0, -1], [-2, 0, 1, 0],
                  [0, -1, 0, 2], [1, 0, -2, 0]],
        "nodes": [(1, 0, 0, 0), (0, 1, 0, 0),
                  (0, 0, 1, 0), (0, 0, 0, 1)],
    },
    "su2nf1": {
        "name": "SU(2) Nf=1",
        "B":     [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]],
        "nodes": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    },
    "su2nf2": {
        "name": "SU(2) Nf=2  (chamber 1->2, 1->3, 2->4, 3->4)",
        "B":     [[0, 1, 1, 0], [-1, 0, 0, 1],
                  [-1, 0, 0, 1], [0, -1, -1, 0]],
        "nodes": [(1, 0, 0, 0), (0, 1, 0, 0),
                  (0, 0, 1, 0), (0, 0, 0, 1)],
    },
}


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

def _demo() -> None:
    """Short self-test: run the full pipeline on the Pentagon theory."""
    print("=== bps_quiver_tools demo : Pentagon  [A1, A2] ===\n")
    spec_info = PRESETS["pentagon"]
    A = CoulombAlgebra(spec_info["B"], spec_info["nodes"])
    print(A)
    print(f"  negating sequence : {A.negating_sequence}")
    print(f"  spectrum          : S = E_q(X_{A.spec[0]}) * E_q(X_{A.spec[1]})")
    print()

    orbit = A.sigma_orbit((1, 0))
    print(f"sigma orbit of (1,0) : {orbit}  (period {len(orbit)})\n")

    print("canonical F_a for orbit charges:")
    for g in orbit:
        Fa = A.F(g)
        summary = " + ".join(f"{c}*X_{d}" if str(c) not in ("1",) else f"X_{d}"
                              for d, c in Fa.items())
        print(f"  F_{g} = {summary}")
    print()

    print("intertwining check  F_a * S = S * rho_Q(F_sigma(a)) :")
    for g in orbit:
        Fa = A.F(g)
        lhs = A.commute_F_across(g, len(A.spec))
        rhs = A.rho_Q(A.F(A.sigma(g)))
        print(f"  a = {str(g):<10}  matches? {lhs == rhs}")
    print()

    print("structure constants F_a * F_b (pentagon orbit, nonzero results):")
    for a in orbit:
        for b in orbit:
            dec = A.multiply(a, b)
            rhs = " + ".join(f"({c})*F_{g}" for g, c in dec.items()) or "0"
            print(f"  F_{a} * F_{b} = {rhs}")
    print()

    print("Schur indices  I_{a,a}(q)  to order q^6 :")
    for g in orbit:
        print(f"  I_({g}, {g}) = {A.schur_index(g, g, K=6)}")


if __name__ == "__main__":
    _demo()
