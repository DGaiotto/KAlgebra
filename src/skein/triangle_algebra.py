"""The triangle Chekhov-Fock algebra Y(T).

The defining presentation is:

    Y(T) = R<y_a^{+/-1}, y_b^{+/-1}, y_c^{+/-1}>
           / (y_a y_b = lq * y_b y_a,
              y_b y_c = lq * y_c y_b,
              y_c y_a = lq * y_a y_c).

`R` is HalfLaurent (= Z[lq^{1/2}, lq^{-1/2}]); the relations only
generate integer powers of lq, so half-integer powers can only enter
through coefficients (e.g. via Weyl normalization).

Internal storage: dict[(i, j, k), HalfLaurent], where (i, j, k) is
the exponent triple of the ordered monomial y_a^i y_b^j y_c^k.
Multiplication brings the result back to this ordered form using the
ccw commutation relations.

Self-contained: depends only on half_laurent.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping

from half_laurent import HalfLaurent, ONE as HL_ONE


# Exponent triple type for clarity
Triple = tuple[int, int, int]


def _lq_power_for_product(
    i: int, j: int, k: int, ip: int, jp: int, kp: int
) -> int:
    """Integer lq-exponent picked up when multiplying

        (y_a^i y_b^j y_c^k) * (y_a^{ip} y_b^{jp} y_c^{kp})

    and reordering the result to (y_a^{i+ip} y_b^{j+jp} y_c^{k+kp}).

    Derived from the defining relations:
        y_a y_b = lq y_b y_a   (=> y_a moves left past y_b: factor lq^{-1})
        y_c y_a = lq y_a y_c   (=> y_a moves left past y_c: factor lq^{+1})
        y_b y_c = lq y_c y_b   (=> y_b moves left past y_c: factor lq^{-1})

    Step 1: pull y_a^{ip} past y_b^j y_c^k     -> ip * (k - j)
    Step 2: pull y_b^{jp} past y_c^k          -> -jp * k
    Total: ip*k - ip*j - jp*k.

    NOTE: phi : cSs(T) -> Y(T) is an algebra HOMOMORPHISM.
    Empirically, the
    relations hold only when multiplication of phi-values is done in
    REVERSE order, i.e. when phi behaves as an anti-homomorphism in
    our representation:
            phi(alpha_1 . alpha_2)  =  phi(alpha_2) . phi(alpha_1).
    Callers must respect this convention.  See the docstring of
    `phi_map.phi`.
    """
    return ip * k - ip * j - jp * k


class TriangleAlgebra:
    """Element of Y(T).

    Immutable.  Exponent triples are integer; coefficients are HalfLaurent.
    """

    __slots__ = ("_terms",)

    def __init__(self, terms: Mapping[Triple, HalfLaurent] | None = None):
        clean: dict[Triple, HalfLaurent] = {}
        if terms is not None:
            for exp, coef in terms.items():
                if coef.is_zero():
                    continue
                if not (isinstance(exp, tuple) and len(exp) == 3
                        and all(isinstance(x, int) for x in exp)):
                    raise TypeError(f"exponent must be a 3-tuple of ints, got {exp!r}")
                if exp in clean:
                    s = clean[exp] + coef
                    if s.is_zero():
                        del clean[exp]
                    else:
                        clean[exp] = s
                else:
                    clean[exp] = coef
        object.__setattr__(self, "_terms", clean)

    # constructors

    @classmethod
    def zero(cls) -> "TriangleAlgebra":
        return cls()

    @classmethod
    def one(cls) -> "TriangleAlgebra":
        return cls({(0, 0, 0): HL_ONE})

    @classmethod
    def from_scalar(cls, c: HalfLaurent) -> "TriangleAlgebra":
        if c.is_zero():
            return cls.zero()
        return cls({(0, 0, 0): c})

    @classmethod
    def y_a(cls, exp: int = 1) -> "TriangleAlgebra":
        return cls({(exp, 0, 0): HL_ONE})

    @classmethod
    def y_b(cls, exp: int = 1) -> "TriangleAlgebra":
        return cls({(0, exp, 0): HL_ONE})

    @classmethod
    def y_c(cls, exp: int = 1) -> "TriangleAlgebra":
        return cls({(0, 0, exp): HL_ONE})

    @classmethod
    def monomial(cls, exp: Triple, coef: HalfLaurent | int = 1) -> "TriangleAlgebra":
        if isinstance(coef, int):
            coef = HalfLaurent.from_int(coef)
        if coef.is_zero():
            return cls.zero()
        return cls({tuple(exp): coef})

    # accessors

    def is_zero(self) -> bool:
        return not self._terms

    def coefficient(self, exp: Triple) -> HalfLaurent:
        return self._terms.get(tuple(exp), HalfLaurent.zero())

    def items(self) -> list[tuple[Triple, HalfLaurent]]:
        return sorted(self._terms.items())

    # arithmetic

    def __add__(self, other) -> "TriangleAlgebra":
        if isinstance(other, int):
            other = TriangleAlgebra.from_scalar(HalfLaurent.from_int(other))
        if isinstance(other, HalfLaurent):
            other = TriangleAlgebra.from_scalar(other)
        if not isinstance(other, TriangleAlgebra):
            return NotImplemented
        out: dict[Triple, HalfLaurent] = dict(self._terms)
        for e, c in other._terms.items():
            if e in out:
                s = out[e] + c
                if s.is_zero():
                    del out[e]
                else:
                    out[e] = s
            else:
                out[e] = c
        return _raw(out)

    __radd__ = __add__

    def __neg__(self) -> "TriangleAlgebra":
        return _raw({e: -c for e, c in self._terms.items()})

    def __sub__(self, other) -> "TriangleAlgebra":
        return self + (-other if isinstance(other, TriangleAlgebra) else -other)

    def __rsub__(self, other) -> "TriangleAlgebra":
        return -self + other

    def __mul__(self, other) -> "TriangleAlgebra":
        if isinstance(other, int):
            other = HalfLaurent.from_int(other)
        if isinstance(other, HalfLaurent):
            if other.is_zero():
                return TriangleAlgebra.zero()
            return _raw({e: c * other for e, c in self._terms.items()})
        if not isinstance(other, TriangleAlgebra):
            return NotImplemented
        out: dict[Triple, HalfLaurent] = {}
        for (i, j, k), c1 in self._terms.items():
            for (ip, jp, kp), c2 in other._terms.items():
                lq_pow = _lq_power_for_product(i, j, k, ip, jp, kp)
                new_exp = (i + ip, j + jp, k + kp)
                # multiply HalfLaurent coefficient by lq^{lq_pow}
                new_coef = (c1 * c2).shift(lq_pow)
                if new_exp in out:
                    s = out[new_exp] + new_coef
                    if s.is_zero():
                        del out[new_exp]
                    else:
                        out[new_exp] = s
                else:
                    if not new_coef.is_zero():
                        out[new_exp] = new_coef
        return _raw(out)

    __rmul__ = __mul__

    def __pow__(self, n: int) -> "TriangleAlgebra":
        if n < 0:
            if len(self._terms) != 1:
                raise ValueError("negative power requires a monomial")
            ((exp, c),) = self._terms.items()
            i, j, k = exp
            # Inverse of (c * y_a^i y_b^j y_c^k):
            # we need lq^x * y_a^{-i} y_b^{-j} y_c^{-k} on the right such that
            # (y_a^i y_b^j y_c^k) * (y_a^{-i} y_b^{-j} y_c^{-k}) = 1.
            # Multiplication formula gives lq_pow = (-i)*k - (-i)*j - (-j)*k
            #                                    = -i*k + i*j + j*k,
            # so the compensating lq factor on the right is lq^{i*k - i*j - j*k}.
            inv_lq_pow = i * k - i * j - j * k
            # c must be a unit in HalfLaurent: a single (+/-1)-coefficient monomial.
            inv_c = c ** -1
            inv_mono = _raw({(-i, -j, -k): inv_c.shift(inv_lq_pow)})
            # Then raise to |n|.
            result = TriangleAlgebra.one()
            base = inv_mono
            e = -n
            while e > 0:
                if e & 1:
                    result = result * base
                base = base * base
                e >>= 1
            return result
        result = TriangleAlgebra.one()
        base = self
        e = n
        while e > 0:
            if e & 1:
                result = result * base
            base = base * base
            e >>= 1
        return result

    # Weyl normalization: [y_c^eps y_b^{eps'}] = lq^{eps eps' / 2} y_c^eps y_b^{eps'}
    #
    # Internally we store ordered monomials.  Using y_c y_b = lq^{-1} y_b y_c,
    #     y_c^eps y_b^{eps'} = lq^{-eps eps'} y_b^{eps'} y_c^eps,
    # so the Weyl bracket equals
    #     lq^{eps eps' / 2 - eps eps'} y_b^{eps'} y_c^eps
    #     = lq^{-eps eps' / 2} y_b^{eps'} y_c^eps  (in ordered form).
    #
    # This static helper just constructs the right element directly.

    @classmethod
    def weyl_cb(cls, eps: int, eps_prime: int) -> "TriangleAlgebra":
        """[y_c^eps y_b^{eps'}], ordered form."""
        if eps not in (1, -1) or eps_prime not in (1, -1):
            raise ValueError("eps, eps' must be in {+1, -1}")
        coef = HalfLaurent.monomial(Fraction(-eps * eps_prime, 2))
        return _raw({(0, eps_prime, eps): coef})

    # equality

    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            other = TriangleAlgebra.from_scalar(HalfLaurent.from_int(other))
        if not isinstance(other, TriangleAlgebra):
            return NotImplemented
        return self._terms == other._terms

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._terms.items())))

    # display

    def __repr__(self) -> str:
        if not self._terms:
            return "0"
        parts: list[str] = []
        for exp in sorted(self._terms):
            i, j, k = exp
            c = self._terms[exp]
            mono_parts: list[str] = []
            for label, e in (("y_a", i), ("y_b", j), ("y_c", k)):
                if e == 0:
                    continue
                if e == 1:
                    mono_parts.append(label)
                else:
                    mono_parts.append(f"{label}^{e}")
            mono = " ".join(mono_parts) if mono_parts else "1"
            crep = repr(c)
            if mono == "1":
                term = crep
            elif crep == "1":
                term = mono
            elif crep == "-1":
                term = f"-{mono}"
            else:
                term = f"({crep})*{mono}"
            if not parts:
                parts.append(term)
            else:
                if term.startswith("-"):
                    parts.append(f" - {term[1:]}")
                else:
                    parts.append(f" + {term}")
        return "".join(parts)


def _raw(terms: dict[Triple, HalfLaurent]) -> TriangleAlgebra:
    obj = TriangleAlgebra.__new__(TriangleAlgebra)
    object.__setattr__(obj, "_terms", terms)
    return obj
