"""Integer kernel + Z-linear section utilities for flavoured BPS realizations.

For a finitely-generated free Z-module `Γ = Z^n` with an antisymmetric
integer pairing `B`, computes:

  * `Γ_f := ker_Z(B)`: the integer kernel, a saturated sublattice of Γ.
  * A Z-basis of `Γ_f`.
  * A Z-linear section `s : Γ_g → Γ` where `Γ_g := Γ / Γ_f`, supplied as
    a complementary Z-basis.

Saturation of `Γ_f = ker(B)` follows from `B` being integer-valued and
Z being torsion-free: `N · γ ∈ Γ_f` ⟹ `N · ⟨γ, ·⟩ = 0` ⟹ `⟨γ, ·⟩ = 0`
⟹ `γ ∈ Γ_f`.  Saturation gives `Γ_g` torsion-free, hence free as a
Z-module, hence the SES `0 → Γ_f → Γ → Γ_g → 0` splits and a *Z-linear*
section always exists (no cocycle obstruction).

Linearity of the section also makes it ρ-equivariant for the canonical
involution `ρ_Q : γ ↦ -γ` on Γ: `s(-g) = -s(g)` automatically.

Algorithm: integer Smith-style column reduction on `B` with column
operations tracked in a unimodular `V`.  After reduction, `B · V` has a
prefix of nonzero columns (rank-many) and a suffix of zero columns
(kernel-many).  The columns of `V` itself, indexed by zero positions in
`B · V`, give a Z-basis of `ker(B)`.  The remaining columns of `V`
provide a complement.

This file has no repo dependencies — pure Python integer arithmetic.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence


def integer_kernel_and_section(
    B: Sequence[Sequence[int]],
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """For an integer square matrix B, compute (ker_basis, sec_basis):

      * ker_basis: list of n-tuples spanning `ker_Z(B)` as a Z-module.
      * sec_basis: list of n-tuples whose union with `ker_basis` is a
        Z-basis of `Z^n` (and hence a section of `Γ → Γ_g`).

    The output is a pair (kernel-basis, section-basis); both lists are
    integer-tuple lists.  `len(ker_basis) + len(sec_basis) == n`.
    """
    n = len(B)
    if n == 0:
        return ([], [])
    if any(len(row) != n for row in B):
        raise ValueError("integer_kernel_and_section: B must be square")

    M = [list(row) for row in B]
    V = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    def swap_cols(j1: int, j2: int) -> None:
        for i in range(n):
            M[i][j1], M[i][j2] = M[i][j2], M[i][j1]
            V[i][j1], V[i][j2] = V[i][j2], V[i][j1]

    def add_col(j_from: int, j_to: int, k: int) -> None:
        if k == 0:
            return
        for i in range(n):
            M[i][j_to] += k * M[i][j_from]
            V[i][j_to] += k * V[i][j_from]

    def swap_rows(i1: int, i2: int) -> None:
        M[i1], M[i2] = M[i2], M[i1]

    def add_row(i_from: int, i_to: int, k: int) -> None:
        if k == 0:
            return
        for j in range(n):
            M[i_to][j] += k * M[i_from][j]

    pos = 0
    while pos < n:
        # Find smallest |entry| in submatrix [pos:, pos:] for the pivot.
        best: tuple[int, int] | None = None
        for i in range(pos, n):
            for j in range(pos, n):
                if M[i][j] != 0:
                    if best is None or abs(M[i][j]) < abs(M[best[0]][best[1]]):
                        best = (i, j)
        if best is None:
            break  # rest of submatrix is zero — done
        bi, bj = best
        if bi != pos:
            swap_rows(pos, bi)
        if bj != pos:
            swap_cols(pos, bj)
        # Reduce the pivot row + column.  Iterate until both are clean.
        while True:
            piv = M[pos][pos]
            if piv == 0:
                break
            changed = False
            for i in range(pos + 1, n):
                if M[i][pos] != 0:
                    q = M[i][pos] // piv
                    if q != 0:
                        add_row(pos, i, -q)
                    if M[i][pos] != 0:
                        if abs(M[i][pos]) < abs(M[pos][pos]):
                            swap_rows(pos, i)
                            changed = True
            for j in range(pos + 1, n):
                if M[pos][j] != 0:
                    q = M[pos][j] // piv
                    if q != 0:
                        add_col(pos, j, -q)
                    if M[pos][j] != 0:
                        if abs(M[pos][j]) < abs(M[pos][pos]):
                            swap_cols(pos, j)
                            changed = True
            if not changed:
                break
        pos += 1

    rank = pos
    ker_basis = [tuple(V[i][j] for i in range(n)) for j in range(rank, n)]
    sec_basis = [tuple(V[i][j] for i in range(n)) for j in range(rank)]
    return ker_basis, sec_basis


def decompose_in_basis(
    gamma: Sequence[int],
    sec_basis: Sequence[Sequence[int]],
    ker_basis: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Decompose `γ ∈ Γ` against the union basis (sec_basis ⊔ ker_basis):
    return `(sec_coords, flavour_coords)` such that

        γ = Σ_i sec_coords[i] · sec_basis[i] + Σ_j flavour_coords[j] · ker_basis[j]

    Both coordinate tuples are integer.  Raises `ValueError` if the
    decomposition isn't integer (= γ doesn't lie in the integer span,
    which shouldn't happen if the bases come from
    `integer_kernel_and_section`).
    """
    n = len(gamma)
    g = len(sec_basis)
    f = len(ker_basis)
    if g + f != n:
        raise ValueError(
            f"decompose_in_basis: sec_basis ({g}) + ker_basis ({f}) ≠ rank {n}"
        )

    # V: columns are sec_basis followed by ker_basis.
    V = [[0] * n for _ in range(n)]
    for j, v in enumerate(sec_basis):
        if len(v) != n:
            raise ValueError("sec_basis vector has wrong length")
        for i in range(n):
            V[i][j] = v[i]
    for j, v in enumerate(ker_basis):
        if len(v) != n:
            raise ValueError("ker_basis vector has wrong length")
        for i in range(n):
            V[i][g + j] = v[i]

    # Solve V · c = γ via integer-rational Gaussian elimination.
    aug = [[Fraction(V[i][j]) for j in range(n)] + [Fraction(gamma[i])]
           for i in range(n)]
    for col in range(n):
        piv_row = None
        for r in range(col, n):
            if aug[r][col] != 0:
                piv_row = r
                break
        if piv_row is None:
            raise ValueError("decompose_in_basis: union basis is singular")
        if piv_row != col:
            aug[col], aug[piv_row] = aug[piv_row], aug[col]
        piv = aug[col][col]
        for r in range(n):
            if r == col:
                continue
            if aug[r][col] != 0:
                factor = aug[r][col] / piv
                aug[r] = [aug[r][k] - factor * aug[col][k] for k in range(n + 1)]

    c = [aug[i][n] / aug[i][i] for i in range(n)]
    c_int = [int(x) for x in c]
    if any(c[i] != c_int[i] for i in range(n)):
        raise ValueError(f"decompose_in_basis: non-integer coords {c} for γ={tuple(gamma)}")

    return tuple(c_int[:g]), tuple(c_int[g:])


__all__ = ["integer_kernel_and_section", "decompose_in_basis", "int_det"]


def int_det(M: Sequence[Sequence[int]]) -> int:
    """Determinant of an integer square matrix.

    Laplace for `n ≤ 3`, Bareiss (fraction-free integer Gaussian
    elimination) otherwise.  Returns an int; never raises on singular
    inputs (returns 0).  Pure integer arithmetic, no `Fraction`.
    """
    n = len(M)
    if n == 0:
        return 1
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    if n == 3:
        return (
            M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
        )
    M = [list(row) for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            piv = None
            for r in range(k + 1, n):
                if M[r][k] != 0:
                    piv = r
                    break
            if piv is None:
                return 0
            M[k], M[piv] = M[piv], M[k]
            sign = -sign
        for r in range(k + 1, n):
            for c in range(k + 1, n):
                M[r][c] = (M[r][c] * M[k][k] - M[r][k] * M[k][c]) // prev
            M[r][k] = 0
        prev = M[k][k]
    return sign * M[n - 1][n - 1]
