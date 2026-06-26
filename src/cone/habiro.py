"""HabiroElement — exact arithmetic in Z[q,q^{-1}][1/(1-q^{2k}) : k>=1].

Canonical-surface migration of `habiro.HabiroElement` (Plan 07
Stage A6, renamed from `habiro_ring.py` to `habiro.py` at Stage D flatten.

Self-contained: only depends on `laurent_poly.LaurentPoly`
and stdlib.

The Habiro ring is the localisation of Z[q,q^{-1}] inverting all
`(1-q^{2k})` for k>=1.  All Nahm-sum / Schur-index intermediates
in this project live exactly in this ring.

(Original docstring follows.)"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Union

from laurent_poly import LaurentPoly


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
    def sum(elements, *, simplify: bool = True) -> "HabiroElement":
        """Sum an iterable of HabiroElements with a single simplify at the end.

        Equivalent to `reduce(operator.add, elements, HabiroElement.zero())`
        but ~n times faster: puts every summand over the common denominator
        in one pass, sums the scaled numerators, and simplifies once -- vs.
        per-addition simplify which repeats O(|denom|) fold tests on each
        intermediate sum.

        Use this whenever you know you are summing many same-ish-shape
        elements (e.g. Nahm-sum assembly, or accumulating [F S|0>]_eta
        contributions across delta in F).

        Pass ``simplify=False`` to skip the closing
        :meth:`HabiroElement.simplify` step.  Useful when the caller
        only needs the Laurent expansion / leading coefficient of the
        result (both of which are independent of whether common
        ``(1-q^{2k})`` factors have been cancelled), e.g. inside the
        F-solver's per-delta materialization.
        """
        elts = [e for e in elements if not e.is_zero()]
        if not elts:
            return HabiroElement.zero()
        if len(elts) == 1:
            # Single-element fast path: no common-denom unification, no
            # numerator scaling, no copy needed.  Common in the F-solver
            # at shallow BFS depth where only one prior delta has
            # propagated to the current eta.
            sole = elts[0]
            return sole.simplify() if simplify else sole
        # Common denominator (max multiplicity per k).
        common: dict[int, int] = {}
        for e in elts:
            for k, m in e.denom.items():
                if m > common.get(k, 0):
                    common[k] = m
        # Pre-sort the common-denom factors by k (smallest first); the
        # downstream ``_scale_numerator_to_denom`` accepts the sorted
        # tuple directly and skips its own sort per contrib.
        sorted_common = tuple(sorted(common.items()))
        # Scale all numerators to the common denom first, then sum.
        # Sum via balanced tree-reduce (pairwise merging) rather than
        # left-fold over a running accumulator: for ``N`` contribs of
        # roughly uniform size, left-fold costs O(N^2 * avg_size)
        # while tree-reduce costs O(N log N * avg_size).  The win is
        # measurable on deep F-solves where each delta has ~10-20
        # contribs of multi-hundred-term polynomials.
        scaled = [e._scale_numerator_to_denom(sorted_common) for e in elts]
        while len(scaled) > 1:
            merged = []
            for i in range(0, len(scaled) - 1, 2):
                merged.append(scaled[i] + scaled[i + 1])
            if len(scaled) % 2:
                merged.append(scaled[-1])
            scaled = merged
        num = scaled[0] if scaled else LaurentPoly.zero()
        result = HabiroElement(num, common)
        return result.simplify() if simplify else result

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

        Iterates denom keys from largest k to smallest: when (1-q^{2k})
        divides num, dividing by larger factors first opens up more
        downstream simplifications (a smaller-k factor of a higher-degree
        numerator is more likely to still divide after a high-k division
        than the reverse).
        """
        num = self.numerator
        denom = dict(self.denom)
        changed = True
        while changed:
            changed = False
            for k in sorted(denom.keys(), reverse=True):
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

        Fast path for ``extra == 1`` (by far the dominant case in the
        F-solver / Schur path: contribs typically lack exactly one factor
        of the common denominator).  Multiplication by ``(1 - q^{2k})``
        is a single in-place merge of ``{e: c}`` and ``{e + 2k: -c}``,
        bypassing the generic ``LaurentPoly.__mul__``.  Zero-filtering
        is deferred to the end (passing a dict with zero entries into
        another ``extra=1`` step costs the same; one final filter is
        cheaper than one filter per step).
        """
        coeffs = self.numerator._coeffs
        scaled = False
        # Apply factors smallest-k first.  Measured ~2x improvement on
        # multi-factor scalings vs descending or arbitrary order:
        # smaller k keeps the intermediate q-degree range tight,
        # reducing the dict-size growth at every step (factor
        # (1-q^{2k}) offsets by 2k, so small-k-first shifts at
        # smaller stride).  Callers in hot loops (HabiroElement.sum)
        # can pass a tuple of (k, m) pairs already sorted to skip the
        # per-call sort; a plain dict triggers the sorted() fallback.
        if isinstance(target, dict):
            target_iter = sorted(target.items())
        else:
            target_iter = target
        for k, m_target in target_iter:
            extra = m_target - self.denom.get(k, 0)
            if extra <= 0:
                continue
            twok = 2 * k
            if extra == 1:
                # Multiply by (1 - q^{2k}) in one merge.  Zeros may
                # accumulate; we filter once at the end of the multi-
                # factor loop.
                new: dict[int, int] = dict(coeffs)
                _new_get = new.get
                for e, c in coeffs.items():
                    e2 = e + twok
                    new[e2] = _new_get(e2, 0) - c
                coeffs = new
            else:
                # Generic case: (1 - q^{2k})^extra via binary exp on the
                # 2-term factor.  Rare (extra >= 2 only when a single
                # contrib is missing multiple copies of the same k).
                num_lp = LaurentPoly({e: v for e, v in coeffs.items() if v != 0})
                num_lp = num_lp * (_one_minus_q2k(k) ** extra)
                coeffs = num_lp._coeffs
            scaled = True
        if not scaled:
            return self.numerator
        return LaurentPoly({e: v for e, v in coeffs.items() if v != 0})

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
            # Integer scaling cannot introduce a new (1-q^{2k}) factor in
            # the numerator (1-q^{2k}) has integer-coprime constant term 1).
            if other == 0:
                return HabiroElement.zero()
            return HabiroElement(self.numerator * other, dict(self.denom))
        if isinstance(other, LaurentPoly):
            # Monomial fast path: multiplying num by c * q^n preserves the
            # divisibility structure --- (1-q^{2k}) divides num iff it
            # divides num * c * q^n.  So simplify() is a guaranteed no-op.
            if len(other._coeffs) <= 1:
                return HabiroElement(self.numerator * other, dict(self.denom))
            return HabiroElement(self.numerator * other, dict(self.denom)).simplify()
        if not isinstance(other, HabiroElement):
            return NotImplemented
        new_denom: dict[int, int] = {}
        for k, m in self.denom.items():
            new_denom[k] = new_denom.get(k, 0) + m
        for k, m in other.denom.items():
            new_denom[k] = new_denom.get(k, 0) + m
        # Numerator monomial fast path: a monomial multiplier cannot
        # alter divisibility, even when both operands carry denominators.
        if len(self.numerator._coeffs) <= 1 or len(other.numerator._coeffs) <= 1:
            return HabiroElement(self.numerator * other.numerator, new_denom)
        return HabiroElement(self.numerator * other.numerator, new_denom).simplify()

    def times_q_number(self, n: int) -> "HabiroElement":
        """Multiply by `[n]_q = q^{1-n}(1-q^{2n})/(1-q^2)`.

        Fast path when `n >= 2` and `n in self.denom`: a (1-q^{2n}) factor
        cancels with one copy from `denom`, and the result is just
        `self.numerator * q^{1-n}` over `(self.denom - {n: 1}) + {1: 1}`.
        No fold tests needed (point (e) of the F-solver optimization plan).

        General case (`n == 1` or `n` not in denom): expressed via the
        cyclotomic factor `(1 + q^2 + ... + q^{2(n-1)})` and the (palindromic-
        preserving) shift `q^{1-n}`, then simplified.
        """
        if n < 1:
            raise ValueError(f"times_q_number requires n >= 1, got {n}")
        if n == 1:
            return self
        if self.is_zero():
            return HabiroElement.zero()
        if n in self.denom:
            # `q^{1-n} * (1-q^{2n}) / (1-q^2)` against `(1-q^{2n})` in denom:
            # the (1-q^{2n}) factor cancels and a (1-q^2) factor is added.
            new_denom = dict(self.denom)
            new_denom[n] -= 1
            if new_denom[n] == 0:
                del new_denom[n]
            new_denom[1] = new_denom.get(1, 0) + 1
            shifted = self.numerator * LaurentPoly({1 - n: 1})
            return HabiroElement(shifted, new_denom)
        # General case: multiply numerator by [n]_q expanded as a polynomial.
        # [n]_q = q^{1-n} + q^{3-n} + ... + q^{n-1}.
        qn_poly = LaurentPoly({1 - n + 2 * i: 1 for i in range(n)})
        return HabiroElement(self.numerator * qn_poly, dict(self.denom))

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

    def expand_to_power_series(self, K: int):
        """Same as `expand(K)`, but returned as a `qpoch.PowerSeries`
        (= explicit truncated q-power-series view).  Useful when the
        caller wants the q-coefficient sequence visible directly,
        rather than the hidden Nahm-sum / rational form."""
        from qpoch import PowerSeries
        lp = self.expand(K)
        # PowerSeries wants dict[int, int] with non-negative exponents
        # (Laurent expansions with negative exponents lose info under
        # this conversion -- the caller should shift first if needed).
        coeffs = dict(lp._coeffs)
        return PowerSeries(coeffs, K)

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
# RHabiroElement — R-coefficient generalisation of HabiroElement.
#
# Element of R[q^±][1/(1-q^{2k}) : k >= 1] for a Z_+-ring R (typically
# `AbelianZPlusRing` or `SO3ZPlusRing`).  Numerator is an `RLaurent`
# over R; denominator is a multiset of `(1 - q^{2k})` factors (only
# even pochhammer exponents, same as the integer-coefficient case).
#
# Use case: the BPS / Schur-index sum S has integer numerator
# coefficients (and so fits HabiroElement), but when the SO(3) chi
# basis or other R-side fugacities are mixed in (e.g. by expanding
# F-products on the R-flavoured side), the coefficients become
# RElement-valued and we need R[q^±][1/(1-q^{2k})] = "RHabiro".
# ---------------------------------------------------------------------------

from zplus_ring import ZPlusRing, RElement, RLaurent  # noqa: E402


def _rlaurent_fold_is_zero(P: "RLaurent", k: int) -> bool:
    """Return True iff (1 - q^{2k}) divides P in R[q^pm]: sum of coefficients
    in each residue class mod 2k is the zero RElement.
    """
    if P.is_zero():
        return True
    twok = 2 * k
    fold: "dict[int, RElement]" = {}
    R = P.ring
    for exp, c in P.coeffs.items():
        r = exp % twok
        if r in fold:
            fold[r] = fold[r] + c
        else:
            fold[r] = c
    return all(v.is_zero() for v in fold.values())


def _rlaurent_try_divide_1mq2k(P: "RLaurent", k: int) -> "RLaurent | None":
    """Divide RLaurent P by (1 - q^{2k}) exactly in R[q^pm]; return None if
    inexact.  Same forward recurrence q_j = p_j + q_{j-2k} as the
    integer case, but with RElement arithmetic.
    """
    R = P.ring
    if P.is_zero():
        return RLaurent(R, {})
    twok = 2 * k
    p = P.coeffs
    j_min = min(p)
    j_max = max(p)
    hi = j_max - twok
    q_coeffs: "dict[int, RElement]" = {}
    for j in range(j_min, j_max + 1):
        val = p.get(j, R.zero())
        prev = q_coeffs.get(j - twok)
        if prev is not None:
            val = val + prev
        if j <= hi:
            if not val.is_zero():
                q_coeffs[j] = val
        else:
            if not val.is_zero():
                return None
    return RLaurent(R, q_coeffs)


def _rlaurent_one(R: "ZPlusRing") -> "RLaurent":
    """The constant 1 in R[q^pm]."""
    return RLaurent(R, {0: R.one()})


def _rlaurent_zero(R: "ZPlusRing") -> "RLaurent":
    return RLaurent(R, {})


def _rone_minus_q2k(R: "ZPlusRing", k: int) -> "RLaurent":
    return RLaurent(R, {0: R.one(), 2 * k: -R.one()})


@dataclass(frozen=True)
class RHabiroElement:
    """Element of R[q, q^{-1}][1/(1-q^{2k}) : k >= 1] for a Z_+-ring R.

    Stored as ``numerator / prod_k (1 - q^{2k})^{denom[k]}``.

    Mirrors `HabiroElement` exactly; arithmetic uses `RLaurent` /
    `RElement` operations on the R-coefficient side.  Same simplify
    invariant: after `.simplify()`, no (1 - q^{2k}) factor in denom
    divides the numerator.

    Construct via `zero(R)`, `one(R)`, `from_rlaurent`,
    `from_relement`, `q_power`, `pochhammer_inverse`, `sum`, or by
    passing `(R, numerator, denom)` directly and calling `.simplify()`.
    """

    ring: "ZPlusRing"
    numerator: "RLaurent"
    denom: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.numerator.ring is not self.ring:
            raise ValueError("RHabiroElement: numerator.ring ≠ self.ring")
        clean = {int(k): int(m) for k, m in self.denom.items() if m}
        for k, m in clean.items():
            if k <= 0:
                raise ValueError(f"denom index must be >= 1, got {k}")
            if m < 0:
                raise ValueError(f"denom multiplicity must be >= 0, got {m} at k={k}")
        object.__setattr__(self, "denom", clean)

    # ---------- Constructors ----------

    @staticmethod
    def zero(ring: "ZPlusRing") -> "RHabiroElement":
        return RHabiroElement(ring, _rlaurent_zero(ring), {})

    @staticmethod
    def one(ring: "ZPlusRing") -> "RHabiroElement":
        return RHabiroElement(ring, _rlaurent_one(ring), {})

    @staticmethod
    def from_rlaurent(p: "RLaurent") -> "RHabiroElement":
        return RHabiroElement(p.ring, p, {})

    @staticmethod
    def from_relement(c: "RElement") -> "RHabiroElement":
        """Lift an RElement c to the constant Habiro element c * q^0."""
        return RHabiroElement(c.ring, RLaurent(c.ring, {0: c}), {})

    @staticmethod
    def q_power(ring: "ZPlusRing", n: int, c: "RElement | None" = None) -> "RHabiroElement":
        """Return c * q^n in R[q^pm][1/(1-q^{2k})] (default c = R.one())."""
        if c is None:
            c = ring.one()
        return RHabiroElement(ring, RLaurent(ring, {n: c}), {})

    @staticmethod
    def pochhammer_inverse(ring: "ZPlusRing", n: int) -> "RHabiroElement":
        """1 / (q^2; q^2)_n  ∈  R[q^pm][1/(1-q^{2k})]."""
        if n < 0:
            raise ValueError("pochhammer_inverse requires n >= 0")
        return RHabiroElement(
            ring, _rlaurent_one(ring), {k: 1 for k in range(1, n + 1)},
        )

    @staticmethod
    def sum(elements, *, simplify: bool = True) -> "RHabiroElement":
        """Sum an iterable of RHabiroElements.  All must share the same `ring`."""
        elts = [e for e in elements if not e.is_zero()]
        if not elts:
            raise ValueError(
                "RHabiroElement.sum: need at least one summand (cannot infer ring)"
            )
        ring = elts[0].ring
        for e in elts[1:]:
            if e.ring is not ring:
                raise ValueError("RHabiroElement.sum: ring mismatch")
        if len(elts) == 1:
            return elts[0].simplify() if simplify else elts[0]
        common: dict[int, int] = {}
        for e in elts:
            for k, m in e.denom.items():
                if m > common.get(k, 0):
                    common[k] = m
        sorted_common = tuple(sorted(common.items()))
        scaled = [e._scale_numerator_to_denom(sorted_common) for e in elts]
        out = _rlaurent_zero(ring)
        for s in scaled:
            out = out + s
        result = RHabiroElement(ring, out, common)
        return result.simplify() if simplify else result

    # ---------- Predicates ----------

    def is_zero(self) -> bool:
        return self.numerator.is_zero()

    # ---------- Canonical form ----------

    def simplify(self) -> "RHabiroElement":
        """Strip common (1-q^{2k}) factors between numerator and denominator."""
        num = self.numerator
        denom = dict(self.denom)
        changed = True
        while changed:
            changed = False
            for k in sorted(denom.keys(), reverse=True):
                if denom[k] <= 0:
                    del denom[k]
                    continue
                q = _rlaurent_try_divide_1mq2k(num, k)
                if q is None:
                    continue
                num = q
                denom[k] -= 1
                if denom[k] == 0:
                    del denom[k]
                changed = True
        if num.is_zero():
            denom = {}
        return RHabiroElement(self.ring, num, denom)

    # ---------- Arithmetic ----------

    def __neg__(self) -> "RHabiroElement":
        return RHabiroElement(self.ring, -self.numerator, dict(self.denom))

    def _scale_numerator_to_denom(self, target: Mapping[int, int]) -> "RLaurent":
        """Multiply self.numerator by the (1 - q^{2k}) factors of `target`
        missing from self.denom."""
        out = self.numerator
        ring = self.ring
        if isinstance(target, dict):
            items = sorted(target.items())
        else:
            items = target
        for k, m_target in items:
            extra = m_target - self.denom.get(k, 0)
            for _ in range(extra):
                out = out * _rone_minus_q2k(ring, k)
        return out

    def __add__(self, other: "RHabiroElement") -> "RHabiroElement":
        if isinstance(other, RHabiroElement):
            if other.ring is not self.ring:
                raise ValueError("RHabiroElement: ring mismatch in __add__")
        elif isinstance(other, RLaurent):
            other = RHabiroElement.from_rlaurent(other)
            if other.ring is not self.ring:
                raise ValueError("RHabiroElement: ring mismatch in __add__")
        else:
            return NotImplemented
        keys = set(self.denom) | set(other.denom)
        common = {k: max(self.denom.get(k, 0), other.denom.get(k, 0)) for k in keys}
        a = self._scale_numerator_to_denom(common)
        b = other._scale_numerator_to_denom(common)
        return RHabiroElement(self.ring, a + b, common).simplify()

    def __sub__(self, other: "RHabiroElement") -> "RHabiroElement":
        return self.__add__(-other)

    def __mul__(self, other) -> "RHabiroElement":
        if isinstance(other, RHabiroElement):
            if other.ring is not self.ring:
                raise ValueError("RHabiroElement: ring mismatch in __mul__")
            num = self.numerator * other.numerator
            denom: dict[int, int] = dict(self.denom)
            for k, m in other.denom.items():
                denom[k] = denom.get(k, 0) + m
            return RHabiroElement(self.ring, num, denom).simplify()
        if isinstance(other, RLaurent):
            if other.ring is not self.ring:
                raise ValueError("RHabiroElement: ring mismatch in __mul__")
            num = self.numerator * other
            return RHabiroElement(self.ring, num, dict(self.denom)).simplify()
        if isinstance(other, RElement):
            if other.ring is not self.ring:
                raise ValueError("RHabiroElement: ring mismatch in __mul__")
            new_coeffs: "dict[int, RElement]" = {}
            for e, c in self.numerator.coeffs.items():
                prod = c * other
                if not prod.is_zero():
                    new_coeffs[e] = prod
            return RHabiroElement(
                self.ring, RLaurent(self.ring, new_coeffs), dict(self.denom),
            ).simplify()
        if isinstance(other, int):
            if other == 0:
                return RHabiroElement.zero(self.ring)
            return RHabiroElement(
                self.ring, self.numerator * other, dict(self.denom),
            )
        return NotImplemented

    # ---------- Expansion ----------

    def _denom_as_rlaurent(self) -> "RLaurent":
        D = _rlaurent_one(self.ring)
        for k, m in self.denom.items():
            f = _rone_minus_q2k(self.ring, k)
            for _ in range(m):
                D = D * f
        return D

    def expand(self, K: int) -> "RLaurent":
        """RLaurent expansion truncated to q-exponents <= K."""
        N = self.numerator
        if N.is_zero():
            return _rlaurent_zero(self.ring)
        D = self._denom_as_rlaurent()
        d = D.coeffs  # d_0 == R.one(); other entries at positive exponents
        n = N.coeffs
        j_min = min(n)
        out: "dict[int, RElement]" = {}
        ring = self.ring
        for j in range(j_min, K + 1):
            val = n.get(j, ring.zero())
            for i, d_i in d.items():
                if i <= 0:
                    continue
                prev = out.get(j - i)
                if prev is None:
                    continue
                # val -= d_i * prev.  d_i is RElement.
                val = val + (-(d_i * prev))
            if not val.is_zero():
                out[j] = val
        return RLaurent(self.ring, out)

    def expand_to_power_series(self, K: int):
        """Same as `expand(K)`, returned as an `RPowerSeries[ring]`
        (= explicit truncated q-power-series view, with the K cap
        carried).  Preferred output form for user-facing display of
        traces / Schur indices: the q-coefficient sequence is visible
        directly, rather than hidden behind the Nahm-sum / rational
        form."""
        from zplus_ring import RPowerSeries
        lp = self.expand(K)
        # Drop entries with q-exponent > K (shouldn't be any) and
        # negative q-exponents (RPowerSeries is non-negative-grading).
        coeffs = {e: c for e, c in lp.coeffs.items() if 0 <= e <= K}
        return RPowerSeries(self.ring, coeffs, K)

    # ---------- Equality ----------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RHabiroElement):
            return NotImplemented
        if self.ring is not other.ring:
            return False
        # Compare in canonical (simplified) form by cross-multiplying:
        # a / D_a == b / D_b  iff  a · D_b == b · D_a.
        a_db = self.numerator * other._denom_as_rlaurent()
        b_da = other.numerator * self._denom_as_rlaurent()
        diff = a_db + (-b_da)
        return diff.is_zero()

    def __hash__(self) -> int:
        raise TypeError("RHabiroElement is unhashable")

    def __repr__(self) -> str:
        if self.is_zero():
            return "0"
        num_repr = str(self.numerator)
        if not self.denom:
            return num_repr
        den_parts = []
        for k in sorted(self.denom):
            m = self.denom[k]
            if m == 1:
                den_parts.append(f"(1 - q^{2*k})")
            else:
                den_parts.append(f"(1 - q^{2*k})^{m}")
        den = " * ".join(den_parts)
        if " + " in num_repr or " - " in num_repr.lstrip("-"):
            num_repr = f"({num_repr})"
        return f"{num_repr} / ({den})"


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def qpoch_inverse(n: int) -> HabiroElement:
    """Return 1 / (q^2; q^2)_n.  Convenience alias for HabiroElement.pochhammer_inverse."""
    return HabiroElement.pochhammer_inverse(n)


def nahm(c: int, shift: int, ns: list[int]) -> HabiroElement:
    """Return c * q^shift / prod_i (q^2; q^2)_{n_i}.  Alias for HabiroElement.nahm_term."""
    return HabiroElement.nahm_term(c, shift, ns)


if __name__ == "__main__":
    # Quick sanity demo: 1/(1-q^2) expands to 1 + q^2 + q^4 + ...
    h = HabiroElement.pochhammer_inverse(1)
    print("1/(q^2;q^2)_1 =", h)
    print("expanded to K=10:", h.expand(10))
    # Pentagon Nahm coefficient at (a,b)=(1,1): +q^{1+1+1} / ((q^2)_1 (q^2)_1)
    t = HabiroElement.nahm_term(1, 3, [1, 1])
    print("Nahm (+,1,[1,1]) =", t)
    print("expanded to K=10:", t.expand(10))
    # Telescoping cancellation sanity: 1/(1-q^2) + 1/(1-q^2) = 2/(1-q^2)
    s = h + h
    print("2 * 1/(q^2;q^2)_1 =", s)
