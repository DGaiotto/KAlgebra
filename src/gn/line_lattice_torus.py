"""The quantum torus of a `LineLattice`, in two frames.

Written to the user's direction (2026-08-24): *"the label space of `L_{m,e}`
should be the Weyl quotient of a lattice which is a sublattice of coweights x
weights containing coroots x roots"*, admissible whenever *"the pairing is
integral"*, with the traditional global forms being the product case
`(coweights of G) x (weights of G)`.  Two exercises, in the order the user set
them, both preceding the rational variants:

1. **The conventional quantum torus** (`conventional_quantum_torus`) — pick a
   `Z`-basis of the lattice, read the Dirac pairing on it, hand the matrix to
   the repo's existing `QuantumTorusKAlg`.  The pairing matrix is integral
   *exactly* when the lattice is admissible, so the construction refuses an
   inadmissible lattice by arithmetic rather than by a separate check.

2. **`DRationalTorus`** — the *demonstrative* step towards the rational quantum
   torus the `AbeKAlgebra` tier will ultimately use (user, 2026-08-24: that one
   represents `Σ_m f_m(𝖖^m v)·U_m` on the **dressed atoms**; this one represents
   `Σ_m d_m(𝖖^m v)·u^m` on the **plain generators**).  The letters are the
   user's and carry the distinction — `f` on `U_m`, `d` on `u^m` — and because
   no dressing `ψ` appears here, the question of whether the atom is
   `ψ_m(v)·u^m` or `u^m·ψ_m(v)` does not arise, which is what makes this the
   clean preliminary step.

   Only the **bare** residuals `d_n(v)` are stored, never the shifted argument
   `𝖖^n v` (user: *"using `\\fq^n v` can be dangerous"*).  The product law is
   the user's, derived the same day:

       [dd']_n(𝖖^n v) = Σ_{n'} d_{n'}(𝖖^{n'} v) · d'_{n−n'}(𝖖^{n+n'} v)
       [dd']_n(v)     = Σ_{n'} d_{n'}(𝖖^{n'−n} v) · d'_{n−n'}(𝖖^{n'} v)

   which follows from `u^n·h(v) = h(𝖖^{2n} v)·u^n` and the scaling convention
   `(𝖖^p v)^λ = 𝖖^{⟨p,λ⟩} v^λ` (the repo's `T_p`).  On monomials
   `d_n(v) = δ_{n,m} v^e` it reproduces frame 1 exactly:

       [dd']_{m+m'}(v) = 𝖖^{⟨m,e'⟩ − ⟨m',e⟩} · v^{e+e'}

   the exponent being the Dirac pairing — so the two frames are one arithmetic,
   not two, and `test_line_lattice_torus.py` asserts that rather than assuming it.

**Only integral powers of `𝖖` occur** (user, 2026-08-24), which is the standing
repo ruling (`the design notes`, "NO square roots of `𝖖` — ever") realised here as a
theorem about the lattice: a residual is `v^e` times a function of `v^α`, the
shifts turn `v^{weight}` into a power of `𝖖` given by the lattice pairing, and
that pairing is integral precisely by admissibility.  `DRationalTorus.multiply`
therefore raises on a fractional exponent instead of rounding — an inadmissible
lattice is reported, never silently accepted.

Coordinates — TWO conventions, deliberately
-------------------------------------------
The **lattice** functions (`lattice_basis`, `dirac_matrix`,
`conventional_quantum_torus`) work in the **fundamental-(co)weight
coordinates**, where the lattice inclusions are simply `Q^∨ = C·Z^r ⊆ P^∨ = Z^r`
and `Q = C^T·Z^r ⊆ P = Z^r`.  `DRationalTorus` works in the **datum's own**
coordinates instead, because that is what `TorusRational` and
`LineLattice.admits` both speak; `charge_of_label` converts between them.  Do not
assume one convention throughout — that is why this section exists.

In the fundamental-(co)weight coordinates: a coweight `m`
is written `Σ_i m_i ω_i^∨` and a weight `e` is written `Σ_j e_j ω_j`, so that

    P^∨ = Z^r ⊇ Q^∨ = C·Z^r        (column span)
    P   = Z^r ⊇ Q   = C^T·Z^r      (row span of C)
    ⟨m, e⟩ = Σ_{i,j} m_i · e_j · (C^{-1})_{ji}

with `C` the Cartan matrix.  `global_form._coweight_omega_coords` /
`_weight_omega_coords` convert from a datum's own coordinates, exactly, through
the datum's own `shift_pairing` — so no coordinate convention is assumed here
either (the constructors disagree; see `global_form.fundamental_weights`).
"""
from __future__ import annotations

from fractions import Fraction

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from global_form import (
    LineLattice,
    cartan_matrix,
    centre_labelling,
    _coweight_omega_coords,
    _weight_omega_coords,
)
from quantum_torus_kalgebra import QuantumTorusKAlg


# ---------------------------------------------------------------------------
# Linear algebra over Z / Q
# ---------------------------------------------------------------------------


def cartan_inverse(datum):
    """`C^{-1}` as a matrix of `Fraction`s.

    This is the pairing matrix between the fundamental coweights and the
    fundamental weights: `⟨ω_i^∨, ω_j⟩ = (C^{-1})_{ji}`, measured against the
    datum's own `shift_pairing` on every pair for ten root data (88/88, SO(8)
    included) before anything was built on it."""
    C = cartan_matrix(datum)
    r = len(C)
    A = [[Fraction(C[i][j]) for j in range(r)]
         + [Fraction(1 if i == k else 0) for k in range(r)] for i in range(r)]
    for c in range(r):
        piv = next((i for i in range(c, r) if A[i][c] != 0), None)
        if piv is None:
            raise ValueError("Cartan matrix is singular")
        A[c], A[piv] = A[piv], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for i in range(r):
            if i != c and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[c])]
    return [[A[i][r + j] for j in range(r)] for i in range(r)]


def _unimodular_inverse(M):
    """Inverse of a unimodular integer matrix, as integers."""
    n = len(M)
    A = [[Fraction(M[i][j]) for j in range(n)]
         + [Fraction(1 if i == k else 0) for k in range(n)] for i in range(n)]
    for c in range(n):
        piv = next((i for i in range(c, n) if A[i][c] != 0), None)
        if piv is None:
            raise ValueError("matrix is singular")
        A[c], A[piv] = A[piv], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for i in range(n):
            if i != c and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[c])]
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            v = A[i][n + j]
            if v.denominator != 1:
                raise ValueError("matrix is not unimodular")
            row.append(int(v))
        out.append(row)
    return out


def _lattice_basis_from_generators(gens, dim):
    """A `Z`-basis of the lattice spanned by integer vectors `gens`.

    Column-style Hermite reduction: repeatedly pick the pivot row, run the
    Euclidean algorithm across the columns that touch it, and retire the single
    surviving column.  Returns at most `dim` vectors."""
    cols = [list(int(x) for x in g) for g in gens]
    cols = [c for c in cols if any(c)]
    basis = []
    pivot = 0
    while cols and pivot < dim:
        live = [c for c in cols if c[pivot] != 0]
        if not live:
            pivot += 1
            continue
        while len(live) > 1:
            live.sort(key=lambda c: abs(c[pivot]))
            p = live[0]
            for c in live[1:]:
                q = c[pivot] // p[pivot]
                if q:
                    for i in range(dim):
                        c[i] -= q * p[i]
            live = [c for c in cols if c[pivot] != 0]
        p = live[0]
        basis.append(tuple(p))
        cols = [c for c in cols if c is not p and any(c)]
        pivot += 1
    return basis


# ---------------------------------------------------------------------------
# The lattice itself
# ---------------------------------------------------------------------------


def pairing_omega(datum, m, e):
    """`⟨m, e⟩` for `m`, `e` in fundamental-(co)weight coordinates."""
    Cinv = cartan_inverse(datum)
    r = len(Cinv)
    return sum(Fraction(m[i]) * Fraction(e[j]) * Cinv[j][i]
               for i in range(r) for j in range(r))


def dirac_omega(datum, label, other):
    """`⟨(m,e), (m',e')⟩ = ⟨m,e'⟩ − ⟨m',e⟩` in the same coordinates."""
    (m, e), (m2, e2) = label, other
    return pairing_omega(datum, m, e2) - pairing_omega(datum, m2, e)


def lattice_generators(lines: LineLattice):
    """Generators of the line lattice as integer `2r`-vectors `(m | e)`.

    Three families: the simple coroots (the columns of `C`, magnetic), the
    simple roots (the rows of `C`, electric) — together spanning `Q^∨ × Q`,
    which every admissible lattice contains — and one lift per generator of the
    class subgroup, which is what enlarges `Q^∨ × Q` to the lattice."""
    datum = lines.datum
    C = cartan_matrix(datum)
    r = len(C)
    divisors, U, V, slots = centre_labelling(datum)
    gens = []
    for j in range(r):                       # simple coroots: columns of C
        gens.append(tuple(C[i][j] for i in range(r)) + tuple(0 for _ in range(r)))
    for i in range(r):                       # simple roots: rows of C
        gens.append(tuple(0 for _ in range(r)) + tuple(C[i][j] for j in range(r)))
    if divisors:
        Uinv = _unimodular_inverse(U)
        VTinv = _unimodular_inverse([[V[j][i] for j in range(r)] for i in range(r)])
        for (k, l) in sorted(lines.class_pairs()):
            y = [0] * r
            z = [0] * r
            for t, slot in enumerate(slots):
                y[slot] = k[t]
                z[slot] = l[t]
            m = tuple(sum(Uinv[i][j] * y[j] for j in range(r)) for i in range(r))
            e = tuple(sum(VTinv[i][j] * z[j] for j in range(r)) for i in range(r))
            gens.append(tuple(m) + tuple(e))
    return gens


def lattice_basis(lines: LineLattice):
    """A `Z`-basis of the line lattice, as `2r` integer `(m | e)` vectors."""
    datum = lines.datum
    r = len(cartan_matrix(datum))
    basis = _lattice_basis_from_generators(lattice_generators(lines), 2 * r)
    if len(basis) != 2 * r:
        raise ValueError(
            f"{lines.name}: expected a rank-{2*r} lattice, got {len(basis)} "
            f"basis vectors")
    return basis


def split(vec, r):
    """Split a `2r`-vector into its magnetic and electric halves."""
    return tuple(vec[:r]), tuple(vec[r:])


def dirac_matrix(lines: LineLattice, basis=None):
    """The Dirac pairing on a `Z`-basis of the lattice.

    Antisymmetric, and **integral exactly when the lattice is admissible** —
    the user's condition, here as the arithmetic that either closes or does
    not.  Returned with `Fraction` entries so the caller can see a violation
    rather than have it rounded away."""
    datum = lines.datum
    r = len(cartan_matrix(datum))
    basis = basis if basis is not None else lattice_basis(lines)
    labels = [split(b, r) for b in basis]
    return [[dirac_omega(datum, a, b) for b in labels] for a in labels]


def conventional_quantum_torus(lines: LineLattice):
    """`(torus, basis)` — the conventional quantum torus of the line lattice.

    The torus is the repo's `QuantumTorusKAlg` on `Γ = Λ` with the Dirac
    pairing, so `X_γ·X_{γ'} = 𝖖^{⟨γ,γ'⟩}X_{γ+γ'}`, bar fixes each `X_γ`,
    `ρ(γ) = −γ` and the trace is the usual `(𝖖²;𝖖²)_∞^{rk}` at `γ = 0`.  Nothing
    new is realised here: the point of the exercise is that a *general* line
    lattice already has a conventional quantum torus, and that building it is
    what tests the lattice layer.

    Raises if the Dirac pairing is not integral on the basis — i.e. if the
    lattice is not admissible."""
    basis = lattice_basis(lines)
    B = dirac_matrix(lines, basis)
    for row in B:
        for x in row:
            if Fraction(x).denominator != 1:
                raise ValueError(
                    f"{lines.name}: the Dirac pairing is not integral on this "
                    f"lattice ({x}), so it is not a legal set of mutually local "
                    f"lines and has no conventional quantum torus")
    return QuantumTorusKAlg([[int(x) for x in row] for row in B]), basis


def charge_of_label(lines: LineLattice, m, e, basis=None):
    """The `Γ`-coordinates of a line `(m, e)` in the lattice basis.

    `m`, `e` are in the *datum's own* coordinates; they are converted here.
    Returns `None` if the label is not in the lattice."""
    datum = lines.datum
    r = len(cartan_matrix(datum))
    basis = basis if basis is not None else lattice_basis(lines)
    target = list(_coweight_omega_coords(datum, m)) + \
        list(_weight_omega_coords(datum, e))
    if any(x.denominator != 1 for x in target):
        return None
    return _solve_integer(basis, [int(x) for x in target], 2 * r)


def _solve_integer(basis, target, dim):
    """Integer coordinates of `target` in `basis`, or `None`."""
    cols = [list(b) for b in basis]
    aug = list(target)
    coeffs = [[1 if i == j else 0 for j in range(len(cols))]
              for i in range(len(cols))]
    # Gaussian elimination over Q, then check integrality.
    n = len(cols)
    A = [[Fraction(cols[j][i]) for j in range(n)] + [Fraction(aug[i])]
         for i in range(dim)]
    row = 0
    where = [-1] * n
    for c in range(n):
        piv = next((i for i in range(row, dim) if A[i][c] != 0), None)
        if piv is None:
            continue
        A[row], A[piv] = A[piv], A[row]
        pv = A[row][c]
        A[row] = [x / pv for x in A[row]]
        for i in range(dim):
            if i != row and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[row])]
        where[c] = row
        row += 1
    for i in range(row, dim):
        if A[i][n] != 0:
            return None
    out = []
    for c in range(n):
        v = A[where[c]][n] if where[c] >= 0 else Fraction(0)
        if v.denominator != 1:
            return None
        out.append(int(v))
    return tuple(out)


# ---------------------------------------------------------------------------
# The `u^m` torus — `Σ_m d_m(𝖖^m v)·u^m`, stored as the bare `d_m(v)`
# ---------------------------------------------------------------------------


def _is_root_weight(datum, e) -> bool:
    """Whether a weight lies in the root lattice `Q`.

    A non-integral weight is refused outright rather than coerced: roots are
    integral in a datum's own coordinates, so `int()` here would have accepted
    a half-integral weight by truncating it (the `so_n(5)` spinor `(½,½)`)."""
    from global_form import in_root_lattice
    vals = [Fraction(x) for x in e]
    if any(v.denominator != 1 for v in vals):
        return False
    return in_root_lattice(datum, tuple(int(v) for v in vals))


def _root_echelon_basis(datum):
    """A row-echelon `Z`-basis of the root lattice `Q`, as `(vector, pivot)`.

    Used to reduce a weight modulo `Q`.  Echelon order matters: once the pivot
    `p_i` is fixed, every later basis vector has a zero there, so reducing in
    order leaves earlier coordinates alone and the reduction is well defined."""
    rows = [[Fraction(x) for x in a] for a in datum.simple_roots]
    d = datum.dim
    basis = []
    col = 0
    r = 0
    while r < len(rows) and col < d:
        # integer (Euclidean) elimination in this column
        while True:
            live = [i for i in range(r, len(rows)) if rows[i][col] != 0]
            if len(live) <= 1:
                break
            live.sort(key=lambda i: abs(rows[i][col]))
            piv = live[0]
            for i in live[1:]:
                q = rows[i][col] // rows[piv][col]
                if q:
                    rows[i] = [a - q * b for a, b in zip(rows[i], rows[piv])]
            rows[r], rows[piv] = rows[piv], rows[r]
        live = [i for i in range(r, len(rows)) if rows[i][col] != 0]
        if not live:
            col += 1
            continue
        i = live[0]
        rows[r], rows[i] = rows[i], rows[r]
        if rows[r][col] < 0:
            rows[r] = [-a for a in rows[r]]
        basis.append((tuple(rows[r]), col))
        r += 1
        col += 1
    return basis


_ROOT_BASIS_CACHE = {}


def reduce_mod_roots(datum, e):
    """`(e_reduced, shift)` with `e = e_reduced + shift` and `shift ∈ Q`.

    A canonical representative of the coset `e + Q`.  This is what makes the
    residual storage well defined: the same residual can be written `v^e·rat(v)`
    or `v^{e−nα}·(v^{nα}rat(v))` — the root monomial may sit in either factor —
    so without a canonical `e` two equal elements would compare unequal and
    would fail to merge under addition (user, 2026-08-24: *"make sure to allow
    comparison of `v^e` (function) and `v^{e−nα}` (`v^{nα}` function)"*).

    Reduction is modulo the **root lattice**, not by electric centre class: at a
    non-semisimple datum such as `U(N)` the class does not determine `e` mod `Q`
    (`P/Q` is infinite there), so a class-based representative would wrongly
    identify distinct cosets."""
    key = datum.name
    basis = _ROOT_BASIS_CACHE.get(key)
    if basis is None:
        basis = _ROOT_BASIS_CACHE[key] = _root_echelon_basis(datum)
    cur = [Fraction(x) for x in e]
    for vec, p in basis:
        q = cur[p] // vec[p]          # floor division on Fractions
        if q:
            cur = [a - q * b for a, b in zip(cur, vec)]
    red = tuple(cur)
    shift = tuple(Fraction(a) - b for a, b in zip(e, red))
    return red, shift


class _BareResidualTorus:
    """The quantum torus of a `LineLattice`, written on the plain generators
    `u^m` — elements `Σ_m d_m(𝖖^m v)·u^m`, stored through the **bare** residuals
    `d_m(v)` alone.

    ⚠ **PROVISIONAL NAME**, pending the user's approval; the object is *the
    quantum torus of a line lattice written on the generators `u^m`*, and that
    description travels with the name until it is ratified.  It deliberately
    does **not** reuse the design notes's "f-presentation" or "atom", both of which
    name the `U_m` object this one is defined in contrast to.

    Built to the user's direction (2026-08-24): *"the rational quantum torus we
    will ultimately use in AbeKAlgebra is the one representing `Σ_m f(\\fq^m v)
    U_m`.  As a preliminary step, I'd like to build first a demonstrative class
    representing `Σ_m d(\\fq^m v) u^m` with the properties we discussed."*  The
    letters are the user's and mark the distinction: `f` for residuals on the
    **dressed atoms** `U_m`, `d` for residuals on the **plain generators** `u^m`.
    No dressing `ψ` appears here at all, so the question of whether the atom is
    `ψ_m(v)·u^m` or `u^m·ψ_m(v)` does not arise — which is what makes this the
    clean preliminary step.

    Storage
    -------
    ⚠ The rational factor is deliberately **not** written `R_{m,e}` and its
    parameter is **not** named `R` (user, 2026-08-24).  `R_{a,b}` is settled
    notation on this very tier for the atom cocycle,
    `U_a U_b = R_{a,b}(v)·U_{a+b}` (`wrq_torus.cocycle_R` / `CC`;
    the design notes writes `R_{m,m'}`), so a two-index `R` for the residual's
    rational factor would collide head-on and read as established.  It is
    described here rather than given a symbol.

    `{m: {e: TorusRational}}`, meaning

        d_m(v)  =  Σ_e  v^e · (a rational function of the root characters v^α)

    with `(m, e)` a **line of the lattice** and that second factor carrying
    **root** weights only (`TorusRational` = `num / ∏(1 − 𝖖^k v^α)`,
    numerator weights required to be in `Q`).  That is the user's own structural
    requirement — *"`f(v)` must be a rational function of `v^root` multiplying a
    `v^{allowed e}`"* — and it is **load-bearing, not decoration**:

    * a shift `v ↦ 𝖖^c v` sends `v^α ↦ 𝖖^{⟨c,α⟩}v^α`, and `⟨c, α⟩ ∈ Z` for any
      coweight `c` and any root `α`, so the rational part never produces a
      fractional power;
    * the whole fractional content therefore sits in the single scalar
      `𝖖^{⟨c, e⟩}` per residual — and in the product law the two such scalars
      combine into the **Dirac pairing**, which is an integer exactly because
      the lattice is admissible.

    Without that split the frame would not close over `Z[𝖖^{±1}]` at a
    correlated form: at the second SO(3) the dyon has `⟨ω^∨, ω⟩ = ½`, so even
    `d_m(𝖖^m v)` read literally is `𝖖^{1/2}v^ω`.  Storing `d_m(v)` and combining
    the scalars per term is what keeps everything in integral powers of `𝖖`
    (user, 2026-08-24; the standing `the design notes` ruling).  A fractional total
    **raises** — it is never rounded.

    Arithmetic
    ----------
    The product law is the user's, derived by them and confirmed here:

        [dd']_n(𝖖^n v) = Σ_{n'} d_{n'}(𝖖^{n'} v)·d'_{n−n'}(𝖖^{n+n'} v)
        [dd']_n(v)     = Σ_{n'} d_{n'}(𝖖^{n'−n} v)·d'_{n−n'}(𝖖^{n'} v)

    from `u^n·h(v) = h(𝖖^{2n}v)·u^n` with `(𝖖^p v)^λ = 𝖖^{⟨p,λ⟩}v^λ`.  Split on
    the storage above, with `n = m_1 + m_2` so that `n' − n = −m_2`:

        [dd']_n = Σ  𝖖^{⟨m_1,e_2⟩ − ⟨m_2,e_1⟩} · v^{e_1+e_2}
                     · R_1(𝖖^{−m_2} v) · R_2(𝖖^{m_1} v)

    the exponent being the Dirac pairing of the two lines.  On monomials
    (`rat ≡ 1`, `d_m(v) = δ_{m,·}v^e`) this is `X_{m,e}` of the conventional
    quantum torus — the user's acceptance test, asserted in the battery rather
    than assumed.

    **The algebra depends on the lattice, not just the datum** (user,
    2026-08-24).  The `LineLattice` is a constructor argument and gates every
    residual: which `(m, e)` may carry a `v^e`, hence which residuals exist at
    all, hence the algebra.  The same root datum with a different lattice is a
    different `DRationalTorus` — at `su(2)` the three forms admit three
    different sets of residuals, and the correlated one admits a residual at the
    fractional coweight that neither other form does.  So a `DRationalTorus` is
    never "the torus of `G`"; it is the torus of a *global form*.

    **Mixing terms that do not fit one consistent lattice leads to trouble**
    (user, 2026-08-24), and the class is built so the trouble is *reported*
    rather than silent.  Two guards, at the two places it can enter:

    * `term`/`monomial` refuse an `(m, e)` that is not a line of this lattice
      (`check_admissible=True`, the default), so an element cannot be assembled
      out of residuals belonging to different global forms;
    * `multiply` recomputes the Dirac pairing per term and **raises** on a
      fractional `𝖖`-power, so even with the check disabled the inconsistency
      surfaces as an error at the first product rather than as a wrong answer.

    The escape hatch `check_admissible=False` exists only to exhibit that second
    failure in the battery; it is not a supported way to build elements.

    Coordinates are the **datum's own** here (not the fundamental-(co)weight
    ones the lattice functions above use), because that is what `TorusRational`
    and `LineLattice.admits` both speak; `charge_of_label` converts when the
    conventional torus is wanted.
    """

    def __init__(self, lines: LineLattice, check_admissible: bool = True):
        from weyl_torus_ring import TorusRational, TorusLaurent
        self.lines = lines
        self.datum = lines.datum
        self.dim = lines.datum.dim
        self.check_admissible = check_admissible
        self._TR = TorusRational
        self._TL = TorusLaurent

    # ----- coordinates ----------------------------------------------------
    def pair(self, m, e) -> Fraction:
        """`⟨m, e⟩` through the datum's own pairing."""
        return Fraction(self.datum.shift_pairing(tuple(m), tuple(e)))

    def _norm_cochar(self, m):
        return tuple(Fraction(x) for x in m)

    def _norm_weight(self, e):
        """Weights are kept as `Fraction`s, never coerced with `int`.

        A weight can be genuinely half-integral in a datum's own coordinates —
        `so_n(5)`'s spinor fundamental weight is `(½, ½)` in the standard `B_2`
        basis — so `int()` here silently truncated it to `(0, 0)`, turning
        `v^{(½,½)}` into `v^0`.  Caught by the conventional-torus comparison."""
        return tuple(Fraction(x) for x in e)

    # ----- construction ---------------------------------------------------
    # ----- what distinguishes the two tori -------------------------------
    def _cocycle_factor(self, a, b):
        """The factor the atoms contribute to `x·y` beyond the plain shifts.

        This single hook is the entire difference between the two tori of this
        module: trivial on the plain generators `u^m`, and the cocycle
        `R̃_{a,b}` on the dressed atoms `U_m`.  Subclassing here is a shared
        **representation**, not a subtype relation between the algebras — they
        are different algebras that happen to store residuals the same way,
        which is why the base is private."""
        raise NotImplementedError

    def zero(self) -> dict:
        return {}

    def rational_one(self):
        """`rat ≡ 1` as a `TorusRational` over this datum."""
        return self._TR.one(self.datum)

    def root_factor_inv(self, alpha, k):
        """`1 / (1 − 𝖖^k v^α)` — a legal residual factor, `α` a root."""
        return self._TR.factor_inv(self.datum, tuple(alpha), int(k))

    def _check_rational_is_root_only(self, rat):
        """A residual's rational part must be a function of `v^α` only.

        Denominators are root factors by `TorusRational`'s own construction; the
        numerator is checked here.  This is what confines the fractional
        `𝖖`-content to the single `v^e` scalar."""
        for w in rat.num.terms:
            if not _is_root_weight(self.datum, w):
                raise ValueError(
                    f"{self.lines.name}: the residual's rational part carries "
                    f"the weight {w}, which is not in the root lattice.  A "
                    f"residual must be `v^e` times a rational function of "
                    f"`v^root`; otherwise the 𝖖-powers do not stay integral.")

    def _int_exponents(self, rat):
        """Coerce `𝖖`-exponents to `int`, raising on a genuine fraction.

        `q_shift(c)` computes `⟨c, λ⟩` through the datum's pairing, so a
        **fractional** cocharacter (any `P^∨ ∖ Q^∨` coweight — every
        non-simply-connected form has them) yields `Fraction`-typed exponents
        even where the value is integral.  Numerically that is harmless —
        `Fraction(-2)` and `-2` are equal and hash alike — but it silently
        defeats the type-based checks that enforce the repo's "no fractional
        powers of `𝖖`" ruling, so the coercion is done explicitly and a real
        fraction is raised rather than carried."""
        from weyl_torus_ring import TorusLaurent, TorusRational
        from laurent_poly import LaurentPoly

        def as_int(k, what):
            kf = Fraction(k)
            if kf.denominator != 1:
                raise ValueError(
                    f"{self.lines.name}: {what} carries 𝖖^{kf}, a fractional "
                    f"power.  The lattice is not admissible: its Dirac pairing "
                    f"is not integral.")
            return int(kf)

        terms = {}
        for w, lp in rat.num.terms.items():
            terms[tuple(w)] = LaurentPoly(
                {as_int(k, "a residual"): c for k, c in lp._coeffs.items()})
        den = {(tuple(a), as_int(k, "a denominator factor")): mult
               for (a, k), mult in rat.den.items()}
        return TorusRational(self.datum, TorusLaurent(self.datum, terms), den)

    def _canonicalise(self, e, rat):
        """Move the root-lattice part of `v^e` into the rational factor.

        `v^e·rat(v) = v^{e_red}·(v^{e−e_red}rat(v))` with `e − e_red ∈ Q`, so the two
        writings denote the SAME residual and only the reduced one is stored.
        Without this, `v^e` and `v^{e−nα}` times a `v^{nα}` function would be
        different keys for one element — equality would fail and sums would not
        merge (user, 2026-08-24)."""
        from weyl_torus_ring import TorusLaurent
        e_red, shift = reduce_mod_roots(self.datum, e)
        if any(s != 0 for s in shift):
            mono = TorusLaurent.monomial(
                self.datum, tuple(int(s) for s in shift))
            rat = rat * self._TR.from_laurent(mono)
        return e_red, rat

    def term(self, m, e, rat=None) -> dict:
        """One residual `d_m(v) = v^e·rat(v)`, `rat` a rational function of `v^α`."""
        m = self._norm_cochar(m)
        e = self._norm_weight(e)
        if self.check_admissible and not self.lines.admits(m, e):
            raise ValueError(
                f"{self.lines.name}: ({m}, {e}) is not a line of this lattice, "
                f"so `v^{e}` is not an allowed residual at `u^{m}`")
        rat = self.rational_one() if rat is None else rat
        self._check_rational_is_root_only(rat)
        e_red, rat = self._canonicalise(e, rat)
        return {m: {e_red: rat}}

    def monomial(self, m, e) -> dict:
        """`d_n(v) = δ_{n,m}·v^e` — the user's acceptance test, which must behave
        as `X_{m,e}` of the conventional quantum torus."""
        return self.term(m, e, None)

    def one(self) -> dict:
        z = tuple(0 for _ in range(self.dim))
        return self.term(z, z, None)

    # ----- arithmetic -----------------------------------------------------
    def add(self, x: dict, y: dict) -> dict:
        out = {m: dict(row) for m, row in x.items()}
        for m, row in y.items():
            dst = out.setdefault(m, {})
            for e, rat in row.items():
                dst[e] = (dst[e] + rat) if e in dst else rat
            for e in [e for e, rat in dst.items() if rat.is_zero()]:
                del dst[e]
            if not dst:
                del out[m]
        return out

    def multiply(self, x: dict, y: dict) -> dict:
        """The user's product law, on the storage split `d_m = Σ_e v^e·rat`."""
        from laurent_poly import LaurentPoly
        out: dict = {}
        for m1, row1 in x.items():
            for m2, row2 in y.items():
                n = tuple(a + b for a, b in zip(m1, m2))
                neg_m2 = tuple(-a for a in m2)
                for e1, rat1 in row1.items():
                    for e2, rat2 in row2.items():
                        expo = self.pair(m1, e2) - self.pair(m2, e1)
                        if Fraction(expo).denominator != 1:
                            raise ValueError(
                                f"{self.lines.name}: the product of ({m1},{e1}) "
                                f"and ({m2},{e2}) needs 𝖖^{expo} — a fractional "
                                f"power.  The lattice is not admissible: its "
                                f"Dirac pairing is not integral.")
                        # The rational parts shift by INTEGER powers whatever the
                        # cocharacter, because they carry only root weights.
                        rat = rat1.q_shift(neg_m2) * rat2.q_shift(m1)
                        rat = rat * self._cocycle_factor(m1, m2)
                        scal = self._TR.from_scalar(
                            self.datum, LaurentPoly({int(expo): 1}))
                        rat = self._int_exponents(rat * scal)
                        e = tuple(a + b for a, b in zip(e1, e2))
                        # `e1` and `e2` are canonical but their sum need not be
                        e, rat = self._canonicalise(e, rat)
                        dst = out.setdefault(n, {})
                        dst[e] = (dst[e] + rat) if e in dst else rat
                if n in out:
                    for e in [e for e, rat in out[n].items() if rat.is_zero()]:
                        del out[n][e]
                    if not out[n]:
                        del out[n]
        return out

    def equal(self, x: dict, y: dict) -> bool:
        """Equality up to `TorusRational`'s own num/den comparison."""
        if set(x) != set(y):
            return False
        for m in x:
            if set(x[m]) != set(y[m]):
                return False
            for e in x[m]:
                if not (x[m][e] == y[m][e]):
                    return False
        return True

    def simplify(self, x: dict) -> dict:
        """Cancel common root factors in every residual."""
        out = {}
        for m, row in x.items():
            r = {e: rat.simplify() for e, rat in row.items()}
            r = {e: rat for e, rat in r.items() if not rat.is_zero()}
            if r:
                out[m] = r
        return out

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.lines.name})"


def format_element(x: dict, generator: str = "u") -> str:
    """`Σ_m (residual)·u^m` as readable text, residual by residual.

    `generator` picks the symbol: `"u"` for the plain generators, `"U"` for the
    dressed atoms.  They are different objects and the printout says which."""
    if not x:
        return "0"
    parts = []
    for m in sorted(x, key=lambda t: tuple(float(a) for a in t)):
        terms = " + ".join(f"v^{e}·({rat})" for e, rat in sorted(x[m].items()))
        parts.append(f"[{terms}]·{generator}^{m}")
    return "  +  ".join(parts)


class DRationalTorus(_BareResidualTorus):
    """Elements `Σ_m d_m(𝖖^m v)·u^m` on the **plain generators** `u^m`.

    Name the user's (2026-08-24).  The atoms are undressed, so the product law
    is the bare shift law with no cocycle — see `_BareResidualTorus` for the
    storage discipline and the reason the `𝖖`-powers stay integral."""

    def _cocycle_factor(self, a, b):
        return self._TR.one(self.datum)


class FRationalTorus(_BareResidualTorus):
    """Elements `Σ_m f_m(𝖖^m v)·U_m` on the **dressed atoms** `U_m`.

    Name ratified by the user (2026-08-24), following their own letter
    convention: `f` for residuals on the dressed atoms `U_m`, `d` for residuals
    on the plain generators `u^m`.

    This is the torus the `AbeKAlgebra` tier ultimately wants (user: *"the
    rational quantum torus we will ultimately use in AbeKAlgebra is the one
    representing `Σ_m f(\\fq^m v) U_m`"*), and `DRationalTorus` was the
    demonstrative step towards it.

    **The only difference is the cocycle.**  `U_a` commutes past a function of
    `v` exactly as `u^a` does — the dressing is itself a function of `v` — so
    the shifts are unchanged, and the whole content of the dressing enters
    through `U_a U_b = R_{a,b}(v)·U_{a+b}`:

        [fg]_n(𝖖^n v) = Σ_a f_a(𝖖^a v)·g_{n−a}(𝖖^{n+a} v)·R_{a,n−a}(v)
        [fg]_n(v)     = Σ_a f_a(𝖖^{a−n} v)·g_{n−a}(𝖖^{a} v)·CC_{a,n−a}(v)

    with `CC_{a,b}` the cocycle in the f-representation (`wrq_torus.CC`; the
    pure-gauge case, `CC[N]` being the one that also carries matter).  `CC` is
    the **primitive** closed form, so this
    class exercises it directly: the product's associativity *is* the twisted
    2-cocycle condition, which is why the battery tests it here.

    `CC` carries only **root** weights, so it drops into the residual's rational
    factor without disturbing the storage discipline, and its `𝖖`-powers are
    integral, so nothing here can introduce a fractional power that the plain
    shifts did not already have.

    Cross-checked against `wrq_torus.WRQTorus`, whose own `__mul__` is this law
    (`fm.q_shift(−mp)·gmp.q_shift(m)·CC`) — an independent oracle,
    not a restatement."""

    def _cocycle_factor(self, a, b):
        from wrq_torus import CC
        return CC(self.datum, tuple(a), tuple(b))

    def as_wrq_residuals(self, x: dict) -> dict:
        """`{m: TorusRational}` — the residuals unsplit, as `WRQTorus` holds them.

        The split storage keeps `v^e` and the root-character rational factor
        apart (that is what confines the fractional `𝖖`-content to one scalar);
        `WRQTorus` keeps a single residual per atom.  This recombines them so the
        two can be compared."""
        from weyl_torus_ring import TorusLaurent
        out = {}
        for m, row in x.items():
            acc = self._TR.zero(self.datum)
            for e, rat in row.items():
                mono = self._TR.from_laurent(
                    TorusLaurent.monomial(self.datum, tuple(e)))
                acc = acc + (mono * rat)
            acc = acc.simplify()
            if not acc.is_zero():
                out[m] = acc
        return out
