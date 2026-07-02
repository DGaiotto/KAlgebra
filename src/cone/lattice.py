"""Lattice and LatticeTorus — the integer lattice with pairing and its quantum torus.
Quantum torus algebra of a lattice Gamma over Z[q, q^{-1}].

Given a lattice Gamma of rank n with linear basis (gamma_1, ..., gamma_n) and
an integer pairing <gamma_i, gamma_j> = B[i][j], the quantum torus algebra
Q_Gamma has linear generators X_gamma for gamma in Gamma and relations

    X_gamma * X_gamma'  =  q^{<gamma, gamma'>} * X_{gamma + gamma'}

The pairing is extended bilinearly:
    <sum a_i gamma_i, sum b_j gamma_j>  =  sum_{i,j} a_i B[i][j] b_j.

In the physical applications B is antisymmetric, but the code does not assume
that; any integer matrix works and the product is sesquilinear with respect
to the bar involution q <-> q^{-1}.

An element is a finite Z[q, q^{-1}]-linear combination

    sum_gamma  f_gamma(q) * X_gamma.

Internally it is stored as a dict mapping the tuple of coordinates
gamma = (a_1, ..., a_n) in Z^n to a LaurentPoly coefficient f_gamma.

The module also provides tools for the positive cone
    Gamma_+ = { sum_i a_i gamma_i : a_i >= 0 in Z }
and for extracting candidate lower / upper tropical charges from an element,
i.e. monomials which are not strictly above / below any other monomial of
the element in the partial order induced by Gamma_+.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Sequence, Union

from laurent_poly import LaurentPoly


Vec = tuple[int, ...]
Coeff = Union[LaurentPoly, int]


# ---------------------------------------------------------------------------
# Lattice with integer pairing
# ---------------------------------------------------------------------------


class Lattice:
    """A finite-rank integer lattice together with an integer pairing matrix.

    The pairing is stored as an n x n tuple of tuples ``pairing`` with
    ``pairing[i][j] = <gamma_i, gamma_j>``.  It is typically antisymmetric.
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

    # --- basic helpers ---

    def zero(self) -> Vec:
        return (0,) * self.rank

    def basis_vec(self, i: int) -> Vec:
        if not 0 <= i < self.rank:
            raise IndexError(i)
        return tuple(1 if k == i else 0 for k in range(self.rank))

    def check(self, gamma) -> Vec:
        from fractions import Fraction
        if any(isinstance(x, Fraction) for x in gamma):
            g = tuple(Fraction(x) for x in gamma)
        else:
            g = tuple(int(x) for x in gamma)
        if len(g) != self.rank:
            raise ValueError(
                f"expected vector of length {self.rank}, got {len(g)}"
            )
        return g

    # --- pairing ---

    def bracket(self, a: Sequence[int], b: Sequence[int]) -> int:
        """Compute <a, b> = sum_{i,j} a_i * B[i][j] * b_j."""
        # Materialise to tuple to avoid consuming generators
        a = tuple(a)
        b = tuple(b)
        B = self.pairing
        s = 0
        for i, ai in enumerate(a):
            if ai == 0:
                continue
            row = B[i]
            for j, bj in enumerate(b):
                if bj:
                    s += ai * row[j] * bj
        return s

    def gauge_rank(self) -> int:
        """Rank of lattice minus dimension of kernel of the pairing.

        For theories with flavour, gauge_rank = rank - dim(ker B).
        Uses integer row reduction to find the rank of B.
        """
        from fractions import Fraction
        n = self.rank
        # Row-reduce B over Q
        M = [[Fraction(self.pairing[i][j]) for j in range(n)] for i in range(n)]
        pivots = 0
        for col in range(n):
            # Find pivot
            pivot_row = None
            for row in range(pivots, n):
                if M[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                continue
            M[pivots], M[pivot_row] = M[pivot_row], M[pivots]
            scale = M[pivots][col]
            for row in range(n):
                if row == pivots or M[row][col] == 0:
                    continue
                factor = M[row][col] / scale
                for j in range(n):
                    M[row][j] -= factor * M[pivots][j]
            pivots += 1
        return pivots

    # --- standard Dirac pairing of physical rank 2r ---

    @staticmethod
    def symplectic(r: int) -> "Lattice":
        """The standard symplectic lattice of rank 2r with blocks [[0,1],[-1,0]]."""
        n = 2 * r
        P = [[0] * n for _ in range(n)]
        for k in range(r):
            P[2 * k][2 * k + 1] = 1
            P[2 * k + 1][2 * k] = -1
        return Lattice(P)

    def __repr__(self) -> str:
        return f"Lattice(rank={self.rank}, pairing={self.pairing})"


# ---------------------------------------------------------------------------
# Elements of the quantum torus algebra Q_Gamma
# ---------------------------------------------------------------------------


def _as_laurent(c: Coeff) -> LaurentPoly:
    if isinstance(c, LaurentPoly):
        return c
    if isinstance(c, int):
        return LaurentPoly.from_int(c)
    raise TypeError(f"cannot convert {type(c).__name__} to LaurentPoly")


class LatticeTorus:
    """An element of the quantum torus algebra of a given :class:`Lattice`.

    Stored as a dict ``{gamma: f_gamma}`` where ``gamma`` is a tuple of ints
    of length ``lattice.rank`` and ``f_gamma`` is a nonzero ``LaurentPoly``.
    """

    __slots__ = ("lattice", "_terms")

    def __init__(
        self,
        lattice: Lattice,
        terms: dict[Sequence[int], Coeff] | None = None,
    ):
        self.lattice = lattice
        self._terms: dict[Vec, LaurentPoly] = {}
        if terms:
            for g, c in terms.items():
                gv = lattice.check(g)
                cp = _as_laurent(c)
                if cp.is_zero():
                    continue
                # accumulate in case the user passes duplicate keys
                if gv in self._terms:
                    cp = self._terms[gv] + cp
                    if cp.is_zero():
                        del self._terms[gv]
                        continue
                self._terms[gv] = cp

    # --- constructors ---

    @classmethod
    def zero(cls, lattice: Lattice) -> "LatticeTorus":
        return cls(lattice)

    @classmethod
    def one(cls, lattice: Lattice) -> "LatticeTorus":
        return cls(lattice, {lattice.zero(): LaurentPoly.one()})

    @classmethod
    def monomial(
        cls,
        lattice: Lattice,
        gamma: Sequence[int],
        coeff: Coeff = 1,
    ) -> "LatticeTorus":
        """Return coeff * X_gamma."""
        return cls(lattice, {lattice.check(gamma): _as_laurent(coeff)})

    @classmethod
    def generator(cls, lattice: Lattice, i: int) -> "LatticeTorus":
        """Return X_{gamma_i}, the i-th basis generator of Q_Gamma."""
        return cls.monomial(lattice, lattice.basis_vec(i))

    # --- introspection ---

    def is_zero(self) -> bool:
        return not self._terms

    def charges(self) -> list[Vec]:
        """Return the list of lattice charges supporting this element."""
        return list(self._terms.keys())

    def coeff(self, gamma: Sequence[int]) -> LaurentPoly:
        g = self.lattice.check(gamma)
        return self._terms.get(g, LaurentPoly.zero())

    def terms(self) -> Iterable[tuple[Vec, LaurentPoly]]:
        return self._terms.items()

    def __len__(self) -> int:
        return len(self._terms)

    def __iter__(self):
        return iter(self._terms)

    # --- lattice compatibility guard ---

    def _check_same(self, other: "LatticeTorus") -> None:
        if self.lattice is not other.lattice and self.lattice.pairing != other.lattice.pairing:
            raise ValueError("operands live in quantum tori of different lattices")

    # --- additive structure ---

    def __neg__(self) -> "LatticeTorus":
        out = LatticeTorus(self.lattice)
        out._terms = {g: -c for g, c in self._terms.items()}
        return out

    def __add__(self, other: Union["LatticeTorus", Coeff]) -> "LatticeTorus":
        if isinstance(other, (int, LaurentPoly)):
            other = LatticeTorus.monomial(self.lattice, self.lattice.zero(), other)
        if not isinstance(other, LatticeTorus):
            return NotImplemented
        self._check_same(other)
        result: dict[Vec, LaurentPoly] = dict(self._terms)
        for g, c in other._terms.items():
            if g in result:
                s = result[g] + c
                if s.is_zero():
                    del result[g]
                else:
                    result[g] = s
            else:
                result[g] = c
        out = LatticeTorus(self.lattice)
        out._terms = result
        return out

    def __radd__(self, other: Coeff) -> "LatticeTorus":
        return self.__add__(other)

    def __sub__(self, other: Union["LatticeTorus", Coeff]) -> "LatticeTorus":
        return self + (-other if isinstance(other, LatticeTorus) else -_as_laurent(other))

    def __rsub__(self, other: Coeff) -> "LatticeTorus":
        return (-self) + other

    # --- multiplication ---

    def __mul__(self, other: Union["LatticeTorus", Coeff]) -> "LatticeTorus":
        # Scalar (Z[q,q^{-1}] or int) multiplication.
        if isinstance(other, (int, LaurentPoly)):
            s = _as_laurent(other)
            if s.is_zero():
                return LatticeTorus.zero(self.lattice)
            out = LatticeTorus(self.lattice)
            out._terms = {g: c * s for g, c in self._terms.items()}
            return out
        if not isinstance(other, LatticeTorus):
            return NotImplemented
        self._check_same(other)
        bracket = self.lattice.bracket
        result: dict[Vec, LaurentPoly] = {}
        for g1, c1 in self._terms.items():
            for g2, c2 in other._terms.items():
                twist = bracket(g1, g2)
                # new charge g1 + g2
                ng = tuple(a + b for a, b in zip(g1, g2))
                contrib = c1 * c2 * LaurentPoly.q(twist)
                if ng in result:
                    s = result[ng] + contrib
                    if s.is_zero():
                        del result[ng]
                    else:
                        result[ng] = s
                else:
                    result[ng] = contrib
        out = LatticeTorus(self.lattice)
        out._terms = result
        return out

    def __rmul__(self, other: Coeff) -> "LatticeTorus":
        if isinstance(other, (int, LaurentPoly)):
            s = _as_laurent(other)
            if s.is_zero():
                return LatticeTorus.zero(self.lattice)
            out = LatticeTorus(self.lattice)
            out._terms = {g: s * c for g, c in self._terms.items()}
            return out
        return NotImplemented

    def __pow__(self, n: int) -> "LatticeTorus":
        if n < 0:
            raise ValueError("negative powers not supported")
        if n == 0:
            return LatticeTorus.one(self.lattice)
        # fast exponentiation
        result = LatticeTorus.one(self.lattice)
        base = self
        k = n
        while k > 0:
            if k & 1:
                result = result * base
            k >>= 1
            if k:
                base = base * base
        return result

    # --- equality and display ---

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            other = LatticeTorus.monomial(self.lattice, self.lattice.zero(), other)
        if not isinstance(other, LatticeTorus):
            return NotImplemented
        return self.lattice.pairing == other.lattice.pairing and self._terms == other._terms

    def __hash__(self) -> int:
        return hash((self.lattice.pairing, tuple(sorted(self._terms.items()))))

    def __repr__(self) -> str:
        if not self._terms:
            return "0"
        parts: list[str] = []
        for g in sorted(self._terms):
            c = self._terms[g]
            c_str = repr(c)
            g_str = "X" + repr(g).replace(" ", "")
            if all(k == 0 for k in g):
                parts.append(c_str)
                continue
            if c == LaurentPoly.one():
                parts.append(g_str)
            elif c == -LaurentPoly.one():
                parts.append("-" + g_str)
            else:
                if "+" in c_str or " - " in c_str:
                    parts.append(f"({c_str})*{g_str}")
                else:
                    parts.append(f"{c_str}*{g_str}")
        s = parts[0]
        for p in parts[1:]:
            if p.startswith("-"):
                s += " - " + p[1:]
            else:
                s += " + " + p
        return s


# ---------------------------------------------------------------------------
# Positive cone and tropical-charge candidates
# ---------------------------------------------------------------------------


class PositiveCone:
    """The positive cone Gamma_+ of non-negative integer combinations of the
    basis (gamma_1, ..., gamma_n) of a :class:`Lattice`.

    In coordinates with respect to that basis, ``gamma in Gamma_+`` iff every
    component is >= 0.  The induced partial order on Gamma is

        gamma'  >  gamma   iff   gamma' - gamma  in  Gamma_+ \\ {0}.
    """

    __slots__ = ("lattice",)

    def __init__(self, lattice: Lattice):
        self.lattice = lattice

    def contains(self, gamma: Sequence[int]) -> bool:
        g = self.lattice.check(gamma)
        return all(x >= 0 for x in g)

    def strictly_above(self, a: Sequence[int], b: Sequence[int]) -> bool:
        """Return True iff a > b, i.e. a - b in Gamma_+ \\ {0}."""
        la = self.lattice.check(a)
        lb = self.lattice.check(b)
        saw_positive = False
        for x, y in zip(la, lb):
            d = x - y
            if d < 0:
                return False
            if d > 0:
                saw_positive = True
        return saw_positive

    # --- tropical-charge candidates ---

    def lower_tropical_candidates(self, element: LatticeTorus) -> list[Vec]:
        """Monomials of ``element`` which are minimal in the cone order.

        A charge gamma supporting ``element`` is a *lower* candidate iff no
        other charge gamma' of ``element`` satisfies gamma > gamma', i.e.
        gamma is not strictly above any other monomial.  Equivalently, gamma
        is a minimal element of ``element.charges()`` in the Gamma_+-order.
        Such a gamma can serve as a lower tropical charge of ``element``.
        """
        if element.lattice.pairing != self.lattice.pairing:
            raise ValueError("element lives in a different lattice")
        charges = element.charges()
        result: list[Vec] = []
        for g in charges:
            is_minimal = True
            for h in charges:
                if h is g:
                    continue
                # g is above h  iff  g - h in Gamma_+ \ {0}  iff  h < g
                # We want: g is NOT above any other h, i.e. g is minimal.
                if self.strictly_above(g, h):
                    is_minimal = False
                    break
            if is_minimal:
                result.append(g)
        return result

    def upper_tropical_candidates(self, element: LatticeTorus) -> list[Vec]:
        """Monomials of ``element`` which are maximal in the cone order.

        A charge gamma supporting ``element`` is an *upper* candidate iff no
        other charge gamma' of ``element`` is strictly above gamma.  Such a
        gamma can serve as an upper tropical charge of ``element``.
        """
        if element.lattice.pairing != self.lattice.pairing:
            raise ValueError("element lives in a different lattice")
        charges = element.charges()
        result: list[Vec] = []
        for g in charges:
            is_maximal = True
            for h in charges:
                if h is g:
                    continue
                if self.strictly_above(h, g):
                    is_maximal = False
                    break
            if is_maximal:
                result.append(g)
        return result


# ---------------------------------------------------------------------------
# Cone-containment predicate
# ---------------------------------------------------------------------------


def make_cone_predicate(cone_gens: Sequence[Sequence[int]]) -> "Callable[[Sequence[int]], bool]":
    """Precompile a fast membership test for the cone spanned by ``cone_gens``.

    Detects the common case of a (possibly permuted) sub-basis of
    standard unit vectors --- each generator has exactly one positive
    entry equal to 1, the supports are pairwise disjoint --- and
    returns a closure that bypasses :func:`cone_contains`'s rational
    Gaussian elimination entirely (membership becomes ``v[i] >= 0`` at
    covered positions, ``v[i] == 0`` elsewhere).

    For non-unit-basis cones, returns a thin wrapper around
    :func:`cone_contains`.  Use this at the top of any hot loop (e.g.
    the F-solver's propagation step) that issues many ``cone_contains``
    calls with a fixed ``cone_gens``.
    """
    gens = [tuple(int(x) for x in g) for g in cone_gens]
    n = len(gens)
    if n == 0:
        def _empty(v: Sequence[int]) -> bool:
            return all(int(x) == 0 for x in v)
        return _empty

    covered: list[int] = []
    is_unit_basis = True
    for g in gens:
        positions = [i for i, x in enumerate(g) if x != 0]
        if len(positions) != 1 or g[positions[0]] != 1:
            is_unit_basis = False
            break
        covered.append(positions[0])
    if is_unit_basis and len(set(covered)) == n:
        rank_dim = len(gens[0]) if gens else 0
        if n == rank_dim:
            # All ambient indices covered: membership is just non-negativity.
            # ``min(v) >= 0`` short-circuits to a single C-level builtin
            # call, beating a Python ``for`` loop with comparisons.
            def _all_covered(v: Sequence[int]) -> bool:
                return min(v) >= 0
            return _all_covered
        covered_set = frozenset(covered)
        def _fast(v: Sequence[int]) -> bool:
            for i, x in enumerate(v):
                xi = int(x)
                if i in covered_set:
                    if xi < 0:
                        return False
                else:
                    if xi != 0:
                        return False
            return True
        return _fast

    # Square-matrix fast path: when ``len(gens) == rank`` and the
    # generators are linearly independent (det != 0), membership
    # reduces to solving ``M c = v`` and checking ``c >= 0`` with
    # integer entries.  We precompute ``adj(M)`` and ``det(M)`` once
    # (integer arithmetic via cofactor expansion); each check is a
    # matrix-vector product and ``n`` divisibility tests --- no
    # rational Gaussian elimination per call.  Covers the typical
    # non-unit-basis BPS-quiver case (e.g. pure SU(2)'s
    # ``[(1, 0), (-1, 2)]``, theories with non-positive-orthant
    # cones).
    rank_dim = len(gens[0])
    if n == rank_dim and n <= 6:  # cofactor cost grows as n!, cap at 6
        # gens is a list of rows (generators); we want
        # c = v · M^{-1} = v · adj(M) / det.
        # Per coordinate i: c_i = sum_j v[j] · adj[j][i].
        adj, det = _adjugate_and_det(gens)
        if det != 0:
            if det > 0:
                def _square(v: Sequence[int]) -> bool:
                    for i in range(n):
                        s = 0
                        for j in range(n):
                            s += v[j] * adj[j][i]
                        if s < 0 or s % det != 0:
                            return False
                    return True
            else:
                # det < 0: c = v · adj / det, with det negative.
                # c >= 0 ⇔ v · adj <= 0 (and divisible by |det|).
                neg_det = -det
                def _square(v: Sequence[int]) -> bool:
                    for i in range(n):
                        s = 0
                        for j in range(n):
                            s += v[j] * adj[j][i]
                        if s > 0 or (-s) % neg_det != 0:
                            return False
                    return True
            return _square

    # General cone: fall back to the full predicate.
    def _general(v: Sequence[int]) -> bool:
        return cone_contains(v, gens)
    return _general


def _adjugate_and_det(M: list[tuple[int, ...]]) -> tuple[list[list[int]], int]:
    """Adjugate (classical adjoint) and determinant of an integer matrix M.

    M is a list of rows.  Returns ``(adj, det)`` such that
    ``adj @ M = M @ adj = det * I``.  Used by
    :func:`make_cone_predicate`'s square-matrix fast path to express
    ``M^{-1} v = adj v / det`` purely in integer arithmetic.

    Cofactor expansion --- O(n!) for an n×n matrix.  ``make_cone_predicate``
    caps the path at n <= 6 to keep this preprocessing bounded.
    """
    n = len(M)
    if n == 1:
        return [[1]], M[0][0]

    def _det(mat: list[list[int]]) -> int:
        size = len(mat)
        if size == 1:
            return mat[0][0]
        if size == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        # Cofactor expansion along the first row.
        total = 0
        for j in range(size):
            if mat[0][j] == 0:
                continue
            minor = [row[:j] + row[j + 1:] for row in mat[1:]]
            sign = 1 if j % 2 == 0 else -1
            total += sign * mat[0][j] * _det(minor)
        return total

    full = [list(row) for row in M]
    det = _det(full)
    # adj[i][j] = cofactor at (j, i) = (-1)^{j+i} * det(M_{j,i})
    adj: list[list[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            minor = [row[:i] + row[i + 1:]
                     for k, row in enumerate(full) if k != j]
            sign = 1 if (i + j) % 2 == 0 else -1
            adj[i][j] = sign * _det(minor)
    return adj, det


_WITNESS_CACHE: dict = {}


def _strict_witness_box(gens, rank):
    """A strict cone witness `f` with `<f, g> >= 1` for every generator,
    by small box search (cached per generator set); `None` if none found
    in the box (e.g. a non-pointed cone)."""
    key = tuple(sorted(gens))
    if key in _WITNESS_CACHE:
        return _WITNESS_CACHE[key]
    from itertools import product as _iproduct
    M = max(max(abs(x) for x in g) for g in gens)
    bound = max(3, 2 * M)
    found = None
    for f in _iproduct(range(-bound, bound + 1), repeat=rank):
        if all(sum(a * b for a, b in zip(f, g)) >= 1 for g in gens):
            found = f
            break
    _WITNESS_CACHE[key] = found
    return found


def cone_contains(v: Sequence[int], cone_gens: Sequence[Sequence[int]]) -> bool:
    """Return True iff ``v`` is a non-negative integer linear combination of
    ``cone_gens``.

    Uses rational Gaussian elimination to find the solution space, then
    searches over free variables (if any) for a non-negative integer
    solution.  This correctly handles overcomplete cones (more generators
    than the ambient dimension).

    For hot loops with a fixed ``cone_gens``, prefer
    :func:`make_cone_predicate`, which short-circuits the unit-basis
    case (common in BPS-quiver theories) without any rational
    arithmetic.
    """
    v = tuple(int(x) for x in v)
    gens = [tuple(int(x) for x in g) for g in cone_gens]
    n = len(gens)
    if n == 0:
        return all(x == 0 for x in v)
    r = len(v)

    # Unit-vector fast path: each generator is e_i for a distinct i.
    covered: list[int] = []
    is_unit_basis = True
    for g in gens:
        positions = [i for i, x in enumerate(g) if x != 0]
        if len(positions) != 1 or g[positions[0]] != 1:
            is_unit_basis = False
            break
        covered.append(positions[0])
    if is_unit_basis and len(set(covered)) == n:
        covered_set = set(covered)
        for i in range(r):
            x = v[i]
            if i in covered_set:
                if x < 0:
                    return False
            else:
                if x != 0:
                    return False
        return True
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

    pivot_rhs = []
    pivot_free_coeff = []
    for idx, col in enumerate(pivots):
        d = A[idx][col]
        rhs_const = A[idx][n] / d
        fcoeffs = [-A[idx][fv] / d for fv in free_vars]
        pivot_rhs.append(rhs_const)
        pivot_free_coeff.append(fcoeffs)

    nf = len(free_vars)

    # Proven enumeration bound (pointed case): a strict witness `f` with
    # `<f, g_i> >= 1` for every generator gives, for ANY representation
    # `v = sum λ_i g_i` with `λ_i >= 0`, `sum λ_i <= <f, v>` — so the
    # free-variable search below is COMPLETE with `max_t = <f, v>`.  Only
    # when no witness is found in the search box (e.g. a non-pointed cone)
    # does the earlier norm heuristic remain, which can in principle
    # under-bound a strongly sheared overcomplete cone.
    witness = _strict_witness_box(gens, r)
    if witness is not None:
        fv = sum(w * x for w, x in zip(witness, v))
        if fv < 0:
            return False                    # <f, v> < 0 excludes membership
        max_t = int(fv)
    else:
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
            feasible = True
            for k in range(len(pivots)):
                val = pivot_rhs[k]
                for jj in range(idx + 1):
                    val += pivot_free_coeff[k][jj] * ts[jj]
                if val < 0:
                    all_nonpos = all(
                        pivot_free_coeff[k][jj] <= 0
                        for jj in range(idx + 1, nf)
                    )
                    if all_nonpos:
                        feasible = False
                        break
            if not feasible:
                if all(pivot_free_coeff[k][idx] <= 0 for k in range(len(pivots))
                       if pivot_rhs[k] + sum(pivot_free_coeff[k][jj] * ts[jj]
                                             for jj in range(idx + 1)) < 0):
                    break
                continue
            if _search(idx + 1, ts):
                return True
        return False

    return _search(0, [0] * nf)


# ---------------------------------------------------------------------------
# Small demonstration
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Rank 2 lattice with the standard symplectic pairing: reproduces the
    # two-variable quantum torus of quantum_torus.py.
    L = Lattice.symplectic(1)
    print("Lattice:", L)

    X = lambda *g: LatticeTorus.monomial(L, g)
    e1, e2 = X(1, 0), X(0, 1)
    print("e1 * e2 =", e1 * e2)
    print("e2 * e1 =", e2 * e1)
    print("[e1, e2] =", e1 * e2 - e2 * e1)

    # Build a typical cluster-type element and extract tropical charge candidates.
    elem = X(1, 0) + X(0, 1) + X(1, 1) + X(2, 3) + X(3, 1) + X(0, 0)
    print("\nelement =", elem)

    cone = PositiveCone(L)
    lo = cone.lower_tropical_candidates(elem)
    hi = cone.upper_tropical_candidates(elem)
    print("lower tropical candidates:", lo)
    print("upper tropical candidates:", hi)

    # A 3-d example with an arbitrary antisymmetric pairing.
    L3 = Lattice([[0, 1, -2], [-1, 0, 3], [2, -3, 0]])
    a = LatticeTorus.generator(L3, 0)
    b = LatticeTorus.generator(L3, 1)
    c = LatticeTorus.generator(L3, 2)
    print("\nrank-3 example: a*b =", a * b, "  b*a =", b * a)
    print("(a + b + c)^2 =", (a + b + c) ** 2)
