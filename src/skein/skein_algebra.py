"""The intrinsic Kauffman-bracket skein algebra Sk(S^2_{0,n}).

Elements are `Z[A^{+/-1}]`-combinations of **basis multicurves**; the
product is computed topologically by `skein_resolve.resolve_product`
(stack, smooth, tighten) -- no quantum torus, no BPS chamber.

Basis and labels
----------------
A *basis label* is a pair `(core, k)`:

* `core` -- normal coordinates of a multicurve with no peripheral and
  no contractible component;
* `k`    -- a tuple of `n_punctures` non-negative ints, the number of
  peripheral loops around each puncture.

By normal-curve theory these labels biject with simple multicurves,
i.e. with the Kauffman-bracket basis of the skein module.  Products
close on `Z[A^{+/-1}]`-combinations of labels (Przytycki).

The trivial-loop value is `delta = -A^2 - A^{-2}`; peripheral loops are
*not* evaluated (they are honest central basis elements -- the algebra
of a puncture's flavour symmetry).

Parallel copies vs bracelets: the basis here is the **multicurve
basis** (a core with multiplicity m on some component means m literal
parallel copies).  Chebyshev / bracelet re-expression, flavour
characters, and the relation to the K-theoretic *canonical* basis
(which on FST charts differs from bare multicurves by a computable
"bubbling" dressing -- see the suite in the source repository) live in the
K-algebra layer, not here.

`A` vs the Coulomb-branch `q`: pinned empirically against the quantum
trace (Y_Delta) oracle; see the suite in the source repository.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from multicurve import is_admissible, peripheral_split
from skein_resolve import resolve_product
from triangulation import Triangulation


Coords = tuple[int, ...]
Label = tuple[Coords, tuple[int, ...]]   # (core coords, peripheral counts)


class APoly:
    """Laurent polynomial in A with integer coefficients (exact)."""

    __slots__ = ("_c",)

    def __init__(self, coeffs: Mapping[int, int] | None = None):
        c: dict[int, int] = {}
        if coeffs:
            for e, v in coeffs.items():
                if v:
                    c[e] = c.get(e, 0) + int(v)
                    if c[e] == 0:
                        del c[e]
        self._c = c

    @classmethod
    def zero(cls) -> "APoly":
        return cls()

    @classmethod
    def one(cls) -> "APoly":
        return cls({0: 1})

    @classmethod
    def monomial(cls, e: int, c: int = 1) -> "APoly":
        return cls({e: c})

    def is_zero(self) -> bool:
        return not self._c

    def items(self):
        return sorted(self._c.items())

    def __add__(self, o):
        if isinstance(o, int):
            o = APoly({0: o})
        if not isinstance(o, APoly):
            return NotImplemented
        c = dict(self._c)
        for e, v in o._c.items():
            c[e] = c.get(e, 0) + v
            if c[e] == 0:
                del c[e]
        out = APoly.__new__(APoly)
        out._c = c
        return out

    __radd__ = __add__

    def __neg__(self):
        out = APoly.__new__(APoly)
        out._c = {e: -v for e, v in self._c.items()}
        return out

    def __sub__(self, o):
        return self + (-o if isinstance(o, APoly) else APoly({0: -o}))

    def __mul__(self, o):
        if isinstance(o, int):
            o = APoly({0: o})
        if not isinstance(o, APoly):
            return NotImplemented
        c: dict[int, int] = {}
        for e1, v1 in self._c.items():
            for e2, v2 in o._c.items():
                e = e1 + e2
                c[e] = c.get(e, 0) + v1 * v2
                if c[e] == 0:
                    del c[e]
        out = APoly.__new__(APoly)
        out._c = c
        return out

    __rmul__ = __mul__

    def __pow__(self, n: int) -> "APoly":
        if n < 0:
            if len(self._c) != 1:
                raise ValueError("negative power needs a monomial")
            ((e, v),) = self._c.items()
            if v not in (1, -1):
                raise ValueError("negative power needs a unit coefficient")
            return APoly({n * e: v if n % 2 == 0 or v == 1 else -1})
        r = APoly.one()
        b = self
        while n:
            if n & 1:
                r = r * b
            b = b * b
            n >>= 1
        return r

    def bar(self) -> "APoly":
        """A -> A^{-1}."""
        return APoly({-e: v for e, v in self._c.items()})

    def __eq__(self, o):
        if isinstance(o, int):
            o = APoly({0: o})
        if not isinstance(o, APoly):
            return NotImplemented
        return self._c == o._c

    def __hash__(self):
        return hash(tuple(sorted(self._c.items())))

    def __repr__(self):
        if not self._c:
            return "0"
        parts = []
        for e in sorted(self._c):
            v = self._c[e]
            if e == 0:
                term = f"{v}"
            else:
                av = "" if abs(v) == 1 else f"{abs(v)}*"
                sign = "-" if v < 0 else ""
                term = f"{sign}{av}A^{e}" if e != 1 else f"{sign}{av}A"
            parts.append(term if not parts and not term.startswith("+")
                         else (f" + {term}" if not term.startswith("-")
                               else f" - {term[1:]}"))
        return "".join(parts)


DELTA = APoly({2: -1, -2: -1})    # trivial loop = -A^2 - A^{-2}


class SkeinElement:
    """A finite Z[A^{+/-1}]-combination of basis multicurve labels."""

    __slots__ = ("algebra", "terms")

    def __init__(self, algebra: "SkeinAlgebra", terms: Mapping[Label, APoly] | None = None):
        self.algebra = algebra
        clean: dict[Label, APoly] = {}
        if terms:
            for lbl, c in terms.items():
                if c.is_zero():
                    continue
                if lbl in clean:
                    s = clean[lbl] + c
                    if s.is_zero():
                        del clean[lbl]
                    else:
                        clean[lbl] = s
                else:
                    clean[lbl] = c
        self.terms = clean

    def __add__(self, o):
        if not isinstance(o, SkeinElement) or o.algebra is not self.algebra:
            return NotImplemented
        t = dict(self.terms)
        for lbl, c in o.terms.items():
            s = t.get(lbl, APoly.zero()) + c
            if s.is_zero():
                t.pop(lbl, None)
            else:
                t[lbl] = s
        return SkeinElement(self.algebra, t)

    def __sub__(self, o):
        return self + (-1) * o

    def __mul__(self, o):
        if isinstance(o, (int, APoly)):
            return SkeinElement(self.algebra, {l: c * o for l, c in self.terms.items()})
        if not isinstance(o, SkeinElement) or o.algebra is not self.algebra:
            return NotImplemented
        out = SkeinElement(self.algebra, {})
        for la, ca in self.terms.items():
            for lb, cb in o.terms.items():
                prod = self.algebra.multiply_labels(la, lb)
                out = out + prod * (ca * cb)
        return out

    __rmul__ = __mul__

    def bar(self) -> "SkeinElement":
        """The bar involution: A -> A^{-1} on coefficients, labels fixed.

        Antimultiplicative on products of curves (mirror of the
        stacking order), which is exactly the K_q bar axiom.
        """
        return SkeinElement(self.algebra, {l: c.bar() for l, c in self.terms.items()})

    def __eq__(self, o):
        if not isinstance(o, SkeinElement):
            return NotImplemented
        return self.algebra is o.algebra and self.terms == o.terms

    def __repr__(self):
        if not self.terms:
            return "0"
        parts = []
        for lbl in sorted(self.terms):
            core, k = lbl
            name_bits = []
            if any(core):
                name_bits.append(f"C{tuple(core)}")
            for p, kp in enumerate(k):
                if kp:
                    name_bits.append(f"P{p}" + (f"^{kp}" if kp > 1 else ""))
            name = "*".join(name_bits) if name_bits else "1"
            parts.append(f"({self.terms[lbl]!r})*{name}")
        return " + ".join(parts)


class SkeinAlgebra:
    """Sk(S^2_{0,n}) over Z[A^{+/-1}], on a fixed ideal triangulation.

    The triangulation is a coordinate system only; algebras over
    different triangulations of the same surface are canonically
    isomorphic (not implemented here).
    """

    def __init__(self, triangulation: Triangulation, *, a_smoothing: str = "ccw"):
        # Both construction paths work since the id-based edge refactor
        # (2026-07-07): the engine consumes triangle_edges /
        # edge_incidences / slot_forward only.
        self.t = triangulation
        self.a_smoothing = a_smoothing
        self._mult_cache: dict[tuple[Coords, Coords], list] = {}

    # -- element constructors -----------------------------------------

    def zero(self) -> SkeinElement:
        return SkeinElement(self, {})

    def one(self) -> SkeinElement:
        return SkeinElement(self, {self.identity_label(): APoly.one()})

    def identity_label(self) -> Label:
        return ((0,) * self.t.n_edges, (0,) * self.t.n_punctures)

    def curve(self, coords: Sequence[int]) -> SkeinElement:
        """The basis multicurve with the given (admissible) coordinates."""
        lbl = self.label_of_coords(coords)
        return SkeinElement(self, {lbl: APoly.one()})

    def peripheral(self, p: int, power: int = 1) -> SkeinElement:
        """`power` parallel peripheral loops around puncture p."""
        k = [0] * self.t.n_punctures
        k[p] = power
        return SkeinElement(self, {((0,) * self.t.n_edges, tuple(k)): APoly.one()})

    def label_of_coords(self, coords: Sequence[int]) -> Label:
        coords = tuple(coords)
        if not is_admissible(self.t, coords):
            raise ValueError(f"{coords} is not an admissible multicurve")
        core, k = peripheral_split(self.t, coords)
        return (core, k)

    # -- product -------------------------------------------------------

    def multiply_labels(self, a: Label, b: Label) -> SkeinElement:
        """Product of two basis multicurves (a placed ABOVE b)."""
        core_a, k_a = a
        core_b, k_b = b
        # peripheral loops are central and disjoint from everything:
        # they pass through to the result's peripheral counts.
        k_tot = tuple(x + y for x, y in zip(k_a, k_b))
        key = (core_a, core_b)
        if key not in self._mult_cache:
            self._mult_cache[key] = resolve_product(
                self.t, core_a, core_b, a_smoothing=self.a_smoothing,
            )
        out: dict[Label, APoly] = {}
        for term in self._mult_cache[key]:
            core, k_new = peripheral_split(self.t, term.coords)
            lbl = (core, tuple(x + y for x, y in zip(k_tot, k_new)))
            w = APoly.monomial(term.a_exponent) * (DELTA ** term.n_trivial)
            s = out.get(lbl, APoly.zero()) + w
            if s.is_zero():
                out.pop(lbl, None)
            else:
                out[lbl] = s
        return SkeinElement(self, out)
