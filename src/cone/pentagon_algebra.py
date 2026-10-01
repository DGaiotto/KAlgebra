"""
Pentagon algebra over Z[q, q^{-1}] -- standalone.

Generators L_i for i in Z/5 with relations
    L_{i+1} L_i     = q^2 L_i L_{i+1}
    L_{i+1} L_{i-1} = 1 + q   L_i
    L_{i-1} L_{i+1} = 1 + q^{-1} L_i.

Linear basis L_{i;a,b} = q^{ab} L_i^a L_{i+1}^b for i in Z/5, a, b >= 0
(with the convention that the unit 1 is uniquely represented as L_{0;0,0}
and pure powers L_i^a as L_{i;a,0}).  The basis label set is

    {(0, 0, 0)} cup { (i, a, 0) : i in 0..4, a >= 1 }
                cup { (i, a, b) : i in 0..4, a >= 1, b >= 1 }.

Products are computed by concatenating monomials and then iteratively
collapsing each three-letter run via the relations above; the only
moves are:

    di = 0  (same index)              merge powers
    di = 4  (descending adjacent)     swap with q^{2 e f}    (deg unchanged)
    di = 1  (ascending adjacent)      basis-form for length 2; for longer
                                      monomials swap the rightmost di=1
                                      pair with q^{-2 e f} to expose a
                                      di=2 pair on its left
    di = 2  (L_i ... L_{i+2})         pentagon: peel one of each and
                                      apply L_i L_{i+2} = 1 + q^{-1} L_{i+1}
    di = 3  (L_i ... L_{i-2})         pentagon: peel and apply
                                          L_{i+1} L_{i-1} = 1 + q L_i
                                      (so L_X L_{X-2} = 1 + q L_{X-1} with
                                      X = i, giving the L_{i-1} term).

Each pentagon move strictly lowers total degree by 2; degree is bounded
below by 0; and the only non-pentagon moves (merge, swap) terminate
internally because they either shorten the monomial or strictly reduce
the number of "out of order" pairs.  Hence reduction terminates.

This module has no in-repo dependencies beyond `quantum_torus.LaurentPoly`
(the coefficient ring Z[q, q^{-1}]).
"""

from __future__ import annotations

from typing import Iterable

from laurent_poly import LaurentPoly, q, q_inv


# ----------------------------------------------------------------------
# Internal monomial layer.
#
# A monomial here is an ordered tuple ((i_1, e_1), ..., (i_k, e_k)) with
# all e_j >= 1 (no two consecutive same index, after normalisation).
# It represents L_{i_1}^{e_1} ... L_{i_k}^{e_k}.
# ----------------------------------------------------------------------

Mono = tuple[tuple[int, int], ...]


def _idx(i: int) -> int:
    return i % 5


def _normalize_mono(letters: Iterable[tuple[int, int]]) -> Mono:
    out: list[tuple[int, int]] = []
    for i, e in letters:
        if e == 0:
            continue
        ii = _idx(i)
        if out and out[-1][0] == ii:
            out[-1] = (ii, out[-1][1] + e)
        else:
            out.append((ii, e))
    return tuple(out)


def _is_basis_form(m: Mono) -> bool:
    if len(m) == 0 or len(m) == 1:
        return True
    if len(m) == 2 and _idx(m[1][0] - m[0][0]) == 1:
        return True
    return False


def _step(coeff: LaurentPoly, m: Mono) -> list[tuple[LaurentPoly, Mono]] | None:
    """One reduction step.  Returns None iff `m` is already in basis form."""
    n = len(m)
    if n <= 1:
        return None
    if n == 2 and _idx(m[1][0] - m[0][0]) == 1:
        return None

    # First pass: scan for a "real" reduction (merge, descending swap,
    # or pentagon).  Apply the leftmost one we find.
    for k in range(n - 1):
        (i, e), (j, f) = m[k], m[k + 1]
        di = _idx(j - i)
        if di == 0:
            new = m[:k] + ((i, e + f),) + m[k + 2:]
            return [(coeff, _normalize_mono(new))]
        if di == 4:
            tw = LaurentPoly.q(2 * e * f)
            new = m[:k] + ((j, f), (i, e)) + m[k + 2:]
            return [(coeff * tw, _normalize_mono(new))]
        if di == 2:
            # L_i L_{i+2} = 1 + q^{-1} L_{i+1}.
            left = m[:k] + (((i, e - 1),) if e > 1 else ())
            right = (((j, f - 1),) if f > 1 else ()) + m[k + 2:]
            mid_one = _normalize_mono(left + right)
            mid_L = _normalize_mono(left + ((i + 1, 1),) + right)
            return [(coeff,                    mid_one),
                    (coeff * LaurentPoly.q(-1), mid_L)]
        if di == 3:
            # L_X L_{X-2} = 1 + q L_{X-1}, with X = i.
            left = m[:k] + (((i, e - 1),) if e > 1 else ())
            right = (((j, f - 1),) if f > 1 else ()) + m[k + 2:]
            mid_one = _normalize_mono(left + right)
            mid_L = _normalize_mono(left + ((i - 1, 1),) + right)
            return [(coeff,                   mid_one),
                    (coeff * LaurentPoly.q(1), mid_L)]
        # di == 1: ascending adjacent.  Skip in this pass.
    # Every adjacent pair is di=1 and n >= 3.  Swap the rightmost pair
    # to expose a di=2 pair to its left.
    k = n - 2
    (i, e), (j, f) = m[k], m[k + 1]
    tw = LaurentPoly.q(-2 * e * f)
    new = m[:k] + ((j, f), (i, e)) + m[k + 2:]
    return [(coeff * tw, _normalize_mono(new))]


def _reduce(coeff: LaurentPoly, m: Mono) -> dict[tuple[int, int, int], LaurentPoly]:
    """Reduce coeff * m to a basis-coordinate dict {(i,a,b): LaurentPoly}."""
    work: list[tuple[LaurentPoly, Mono]] = [(coeff, _normalize_mono(m))]
    out: dict[tuple[int, int, int], LaurentPoly] = {}
    while work:
        c, mm = work.pop()
        if c.is_zero():
            continue
        if _is_basis_form(mm):
            key = _basis_key_from_mono(mm)
            i, a, b = key
            # L_i^a L_{i+1}^b = q^{-ab} L_{i;a,b}.
            adj = c * LaurentPoly.q(-a * b)
            cur = out.get(key, LaurentPoly.zero())
            s = cur + adj
            if s.is_zero():
                out.pop(key, None)
            else:
                out[key] = s
            continue
        nxt = _step(c, mm)
        assert nxt is not None
        work.extend(nxt)
    return out


def _basis_key_from_mono(m: Mono) -> tuple[int, int, int]:
    if len(m) == 0:
        return (0, 0, 0)
    if len(m) == 1:
        i, a = m[0]
        return (_idx(i), a, 0)
    (i, a), (j, b) = m
    assert _idx(j - i) == 1
    return (_idx(i), a, b)


# ----------------------------------------------------------------------
# Basis label canonicalisation.
# ----------------------------------------------------------------------

Key = tuple[int, int, int]


def _canon_key(i: int, a: int, b: int) -> Key:
    if a == 0 and b == 0:
        return (0, 0, 0)
    if b == 0:
        return (_idx(i), a, 0)
    if a == 0:
        # L_{i;0,b} = L_{i+1}^b = L_{i+1; b, 0}.
        return (_idx(i + 1), b, 0)
    return (_idx(i), a, b)


# ----------------------------------------------------------------------
# Public class.
# ----------------------------------------------------------------------

class PentagonAlgebra:
    """An element of the pentagon algebra, stored as a dict
    ``{(i, a, b): LaurentPoly}``  representing
    sum_{(i,a,b)} c_{i,a,b}(q) L_{i;a,b}.

    Keys are kept canonical via `_canon_key`.
    """

    __slots__ = ("_terms",)

    def __init__(self, terms: dict[Key, LaurentPoly] | None = None):
        self._terms: dict[Key, LaurentPoly] = {}
        if terms:
            for k, c in terms.items():
                if c.is_zero():
                    continue
                ck = _canon_key(*k)
                cur = self._terms.get(ck, LaurentPoly.zero())
                s = cur + c
                if s.is_zero():
                    self._terms.pop(ck, None)
                else:
                    self._terms[ck] = s

    # --- constructors ---

    @staticmethod
    def zero() -> "PentagonAlgebra":
        return PentagonAlgebra()

    @staticmethod
    def one() -> "PentagonAlgebra":
        return PentagonAlgebra({(0, 0, 0): LaurentPoly.one()})

    @staticmethod
    def L(i: int) -> "PentagonAlgebra":
        return PentagonAlgebra({(_idx(i), 1, 0): LaurentPoly.one()})

    @staticmethod
    def basis(i: int, a: int, b: int,
              coeff: LaurentPoly | int = 1) -> "PentagonAlgebra":
        if isinstance(coeff, int):
            coeff = LaurentPoly.from_int(coeff)
        if coeff.is_zero():
            return PentagonAlgebra.zero()
        return PentagonAlgebra({_canon_key(i, a, b): coeff})

    @staticmethod
    def from_laurent(p: LaurentPoly) -> "PentagonAlgebra":
        if p.is_zero():
            return PentagonAlgebra.zero()
        return PentagonAlgebra({(0, 0, 0): p})

    # --- accessors ---

    def is_zero(self) -> bool:
        return len(self._terms) == 0

    def coeff(self, i: int, a: int, b: int) -> LaurentPoly:
        return self._terms.get(_canon_key(i, a, b), LaurentPoly.zero())

    def terms(self) -> list[tuple[Key, LaurentPoly]]:
        return sorted(self._terms.items())

    # --- arithmetic ---

    def __neg__(self) -> "PentagonAlgebra":
        return PentagonAlgebra({k: -c for k, c in self._terms.items()})

    @staticmethod
    def _coerce(x) -> "PentagonAlgebra | None":
        if isinstance(x, PentagonAlgebra):
            return x
        if isinstance(x, LaurentPoly):
            return PentagonAlgebra.from_laurent(x)
        if isinstance(x, int):
            return PentagonAlgebra.from_laurent(LaurentPoly.from_int(x))
        return None

    def __add__(self, other) -> "PentagonAlgebra":
        oth = PentagonAlgebra._coerce(other)
        if oth is None:
            return NotImplemented
        out = dict(self._terms)
        for k, c in oth._terms.items():
            cur = out.get(k, LaurentPoly.zero())
            s = cur + c
            if s.is_zero():
                out.pop(k, None)
            else:
                out[k] = s
        return PentagonAlgebra(out)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        oth = PentagonAlgebra._coerce(other)
        if oth is None:
            return NotImplemented
        return self + (-oth)

    def __rsub__(self, other):
        oth = PentagonAlgebra._coerce(other)
        if oth is None:
            return NotImplemented
        return oth + (-self)

    def __mul__(self, other) -> "PentagonAlgebra":
        if isinstance(other, int):
            other = LaurentPoly.from_int(other)
        if isinstance(other, LaurentPoly):
            return PentagonAlgebra({k: c * other for k, c in self._terms.items()})
        if not isinstance(other, PentagonAlgebra):
            return NotImplemented
        out: dict[Key, LaurentPoly] = {}
        for (i1, a1, b1), c1 in self._terms.items():
            for (i2, a2, b2), c2 in other._terms.items():
                # L_{i;a,b} L_{j;c,d} = q^{ab+cd} L_i^a L_{i+1}^b L_j^c L_{j+1}^d.
                shift = LaurentPoly.q(a1 * b1 + a2 * b2)
                m = _normalize_mono([
                    (i1, a1), (i1 + 1, b1),
                    (i2, a2), (i2 + 1, b2),
                ])
                contrib = _reduce(c1 * c2 * shift, m)
                for k, v in contrib.items():
                    cur = out.get(k, LaurentPoly.zero())
                    s = cur + v
                    if s.is_zero():
                        out.pop(k, None)
                    else:
                        out[k] = s
        return PentagonAlgebra(out)

    def __rmul__(self, other):
        if isinstance(other, int):
            other = LaurentPoly.from_int(other)
        if isinstance(other, LaurentPoly):
            return PentagonAlgebra({k: other * c for k, c in self._terms.items()})
        return NotImplemented

    def __pow__(self, n: int) -> "PentagonAlgebra":
        if n < 0:
            raise ValueError("negative powers not supported")
        result = PentagonAlgebra.one()
        base = self
        while n > 0:
            if n & 1:
                result = result * base
            n >>= 1
            if n:
                base = base * base
        return result

    def __eq__(self, other) -> bool:
        oth = PentagonAlgebra._coerce(other)
        if oth is None:
            return NotImplemented
        return self._terms == oth._terms

    def __hash__(self):
        return hash(tuple(sorted(
            (k, tuple(sorted(c._coeffs.items())))
            for k, c in self._terms.items()
        )))

    # --- the rho automorphism ---

    def rho(self) -> "PentagonAlgebra":
        """rho(L_i) = L_{i+2}, extended as an algebra map.

        On basis: L_{i;a,b} = q^{ab} L_i^a L_{i+1}^b
                          -> q^{ab} L_{i+2}^a L_{i+3}^b = L_{i+2;a,b}.
        """
        return PentagonAlgebra({
            _canon_key(i + 2, a, b): c
            for (i, a, b), c in self._terms.items()
        })

    # --- display ---

    def __repr__(self) -> str:
        if not self._terms:
            return "0"
        parts = []
        for (i, a, b) in sorted(self._terms):
            c = self._terms[(i, a, b)]
            if (i, a, b) == (0, 0, 0):
                label = "1"
            elif b == 0 and a == 1:
                label = f"L_{i}"
            elif b == 0:
                label = f"L_{i}^{a}"
            else:
                label = f"L_{{{i};{a},{b}}}"
            cs = repr(c)
            if c == LaurentPoly.one():
                parts.append(label)
            elif c == -LaurentPoly.one():
                parts.append("-" + label)
            else:
                parts.append(f"({cs})*{label}" if label != "1" else cs)
        s = parts[0]
        for p in parts[1:]:
            if p.startswith("-"):
                s += " - " + p[1:]
            else:
                s += " + " + p
        return s
