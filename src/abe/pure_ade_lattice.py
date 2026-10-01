"""Closed-form lattice data for pure simply-laced ADE gauge theories.

Physical picture
----------------
For any lattice V the cotangent construction gives a natural unimodular
symplectic lattice

    Γ = V^∨ ⊕ V,   ⟨(a,b),(c,d)⟩ = a(d) − c(b),   V^∨ = Hom(V, Z).

For pure ADE gauge theories there are two canonical choices:

  * **Simply-connected G̃**: take V = P (weight lattice), giving
    Γ = P^∨ ⊕ P.
  * **Adjoint G_ad**: take V = Q (root lattice), giving Γ = Q^∨ ⊕ Q.

Both are unimodular.  For ADE (simply-laced) P^∨ = Q, so in both cases the
node charges live in Q ⊕ Q and differ only by the ambient Γ.

This module implements the **simply-connected** form Γ = P^∨ ⊕ P by default.
The ambient Z^{2r} lattice uses the interleaved basis
    e_{2i}   = ω_i^∨  (i-th fundamental coweight, magnetic)
    e_{2j+1} = ω_j    (j-th fundamental weight,  electric)
with unit symplectic pairing ⟨e_{2i}, e_{2j+1}⟩ = ω_i^∨(ω_j) = δ_{ij}.

In these coordinates (SC form):
    γ_{i,+} = e_{2i}
    γ_{i,−} = −e_{2i} + Σ_j C_{ij} e_{2j+1}    (= −ω_i^∨ + α_i)

and the exchange ⟨γ_{i,+}, γ_{j,−}⟩ = ω_i^∨(α_j) = C_{ij} (Cartan).

Public API
----------

    data = pure_ade_lattice_data([("A", 2)])         # dict of raw lattice data
    alg  = pure_ade_kalgebra([("A", 2)])             # BPSKAlgebra or None

    data = pure_ade_lattice_data([("A", 2), ("D", 4)])
    data = pure_ade_lattice_data([("A", 2)], global_form="adj")
    data = pure_ade_lattice_data([("A", 2)], global_form=[((1, 0), (0, 0))])

The returned ``data`` dict has keys:
    ``B``, ``nodes``, ``spec``, ``cone_witness``, ``dim``, ``factor_info``,
    and (explicit-Lagrangian path only) ``lagrangian_generators``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence, Union
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bps_kalgebra import BPSKAlgebra
from dynkin_data import _factor_data    # (rank, Coxeter_h, Dynkin_edges) per ADE type
import bps_quiver_tools as _bps


Vec = tuple[int, ...]
FactorSpec = Union[tuple, str]


# ---------------------------------------------------------------------------
# Factor-spec parsing
# ---------------------------------------------------------------------------


def _parse_factor(f: FactorSpec) -> tuple[str, int]:
    """Normalise a factor spec to ``(kind, n)`` where kind ∈ {A, D, E, U}.

    Accepts:
      ``("A", n)``, ``("D", n)``, ``("E", n)`` — simple ADE factor.
      ``"U1"`` / ``"U(1)"`` / ``("U1",)``       — abelian U(1) factor.
    """
    if isinstance(f, str):
        s = f.replace("(", "").replace(")", "").upper()
        if s in ("U1", "U"):
            return ("U", 1)
        raise ValueError(f"unknown factor spec {f!r}")
    if isinstance(f, (tuple, list)):
        if len(f) == 1:
            s = str(f[0]).replace("(", "").replace(")", "").upper()
            if s in ("U1", "U"):
                return ("U", 1)
        if len(f) == 2:
            tp = str(f[0]).upper()
            if tp in ("A", "D", "E"):
                return (tp, int(f[1]))
            if tp in ("U", "U1"):
                return ("U", 1)
    raise ValueError(
        f"unrecognised factor spec {f!r}; "
        f"expected ('A', n), ('D', n), ('E', n), or 'U1'"
    )


# ---------------------------------------------------------------------------
# Symplectic-basis building blocks
# ---------------------------------------------------------------------------


def _cartan_matrix(r: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Simply-laced Cartan matrix: 2 on the diagonal, −1 across Dynkin edges."""
    C = [[0] * r for _ in range(r)]
    for i in range(r):
        C[i][i] = 2
    for (a, b) in edges:
        C[a][b] = -1
        C[b][a] = -1
    return C


def _standard_symplectic_pairing(r: int) -> list[list[int]]:
    """2r × 2r standard symplectic matrix: block-diag([[0, 1], [−1, 0]])."""
    B = [[0] * (2 * r) for _ in range(2 * r)]
    for i in range(r):
        B[2 * i][2 * i + 1] = 1
        B[2 * i + 1][2 * i] = -1
    return B


def _sc_node_charges(r: int, C: list[list[int]]) -> list[Vec]:
    """BPS-quiver node charges for the simply-connected embedding.

    γ_{i,+}  = α_i∨              = e_{2i}
    γ_{i,-}  = −α_i∨ + α_i      = −e_{2i} + Σ_j C_{ij} e_{2j+1}

    Returned in interleaved order [γ_{0,+}, γ_{0,−}, γ_{1,+}, γ_{1,−}, …].
    """
    dim = 2 * r
    nodes: list[Vec] = []
    for i in range(r):
        gp = [0] * dim
        gp[2 * i] = 1
        gm = [0] * dim
        gm[2 * i] = -1
        for j in range(r):
            gm[2 * j + 1] += C[i][j]
        nodes.append(tuple(gp))
        nodes.append(tuple(gm))
    return nodes


def _bipartite_coxeter_sequence(r: int, h: int) -> list[int]:
    """Mutation-index sequence for the bipartite Coxeter element.

    h alternating passes, each mutating one colour class of nodes:
      even pass (k even): nodes 0, 2, 4, … = γ_{0,+}, γ_{1,+}, …
      odd  pass (k odd ): nodes 1, 3, 5, … = γ_{0,−}, γ_{1,−}, …
    Total length: r·h.
    """
    seq: list[int] = []
    for k in range(h):
        parity = k % 2
        seq.extend(2 * i + parity for i in range(r))
    return seq


def _gauss_jordan_solve(C: list[list[int]], rhs: list) -> list:
    """Solve C · x = rhs via Gauss-Jordan over Fraction; return x."""
    n = len(C)
    A = [[Fraction(C[i][j]) for j in range(n)] + [Fraction(rhs[i])]
         for i in range(n)]
    for col in range(n):
        piv = next((row for row in range(col, n) if A[row][col] != 0), None)
        if piv is None:
            raise ValueError("Cartan matrix is singular")
        A[col], A[piv] = A[piv], A[col]
        A[col] = [x / A[col][col] for x in A[col]]
        for row in range(n):
            if row != col and A[row][col] != 0:
                fac = A[row][col]
                A[row] = [a - fac * b for a, b in zip(A[row], A[col])]
    return [A[i][n] for i in range(n)]


def _center_order(C: list[list[int]]) -> int:
    """Return |P/Q| = det(C) for a simply-laced Cartan matrix."""
    n = len(C)
    A = [[Fraction(C[i][j]) for j in range(n)] for i in range(n)]
    det = Fraction(1)
    for col in range(n):
        piv = next((row for row in range(col, n) if A[row][col] != 0), None)
        if piv is None:
            return 0
        if piv != col:
            A[col], A[piv] = A[piv], A[col]
            det = -det
        det *= A[col][col]
        A[col] = [x / A[col][col] for x in A[col]]
        for row in range(n):
            if row != col and A[row][col] != 0:
                fac = A[row][col]
                A[row] = [a - fac * b for a, b in zip(A[row], A[col])]
    return abs(int(det))


def _weyl_cone_witness(r: int, C: list[list[int]]) -> Vec:
    """Linear form evaluating to 1 on every SC node charge.

    f[2i]   = 1          (from γ_{i,+} = e_{2i})
    f[2j+1] = (Σ⁺)_j     where Σ⁺ = C⁻¹ · (2, …, 2) (the sum of the positive roots
                          expressed in the simple-root basis).

    Verification:
      f · γ_{i,+} = f[2i] = 1.
      f · γ_{i,−} = −f[2i] + Σ_j C_{ij} f[2j+1]
                  = −1 + Σ_j C_{ij} (Σ⁺)_j = −1 + 2 = 1.

    For all simply-laced ADE, Σ⁺ has positive integer coordinates in the
    simple-root basis (verified numerically for A/D/E up to E_8).
    """
    sigma_plus = _gauss_jordan_solve(C, [2] * r)
    f = [0] * (2 * r)
    for i in range(r):
        assert sigma_plus[i].denominator == 1, (
            f"Σ⁺[{i}] = {sigma_plus[i]} is not an integer; "
            f"unexpected for simply-laced ADE"
        )
        f[2 * i] = 1
        f[2 * i + 1] = int(sigma_plus[i])
    return tuple(f)


def _adj_node_charges(r: int, C: list[list[int]]) -> list[Vec]:
    """BPS-quiver node charges for the adjoint (Q^∨ ⊕ Q) embedding.

    In the adjoint basis (e_{2i} = ω_i, e_{2j+1} = α_j):

        γ_{i,+}  = Σ_j C_{ij} e_{2j}          (Cartan row in magnetic)
        γ_{i,−}  = −Σ_j C_{ij} e_{2j} + e_{2i+1}

    Returned in interleaved order [γ_{0,+}, γ_{0,−}, γ_{1,+}, γ_{1,−}, …].
    Pairing: ⟨e_{2i}, e_{2j+1}⟩ = ω_i(α_j) = δ_{ij} — same standard
    symplectic B as the SC embedding.
    """
    dim = 2 * r
    nodes: list[Vec] = []
    for i in range(r):
        gp = [0] * dim
        for j in range(r):
            gp[2 * j] = C[i][j]
        gm = [-x for x in gp]
        gm[2 * i + 1] += 1
        nodes.append(tuple(gp))
        nodes.append(tuple(gm))
    return nodes


def _adj_cone_witness(r: int, C: list[list[int]]) -> Vec:
    """Linear form with f · γ > 0 for every adjoint node charge.

    Let m = det(C) = |P/Q| (the center order).  Solve C · f_m = m·(1,…,1)
    — this has an integer solution for all simply-laced ADE since the
    denominators of C⁻¹ divide m.

    f[2i]   = (f_m)_i     → f · γ_{i,+} = Σ_j C_{ij} f[2j] = m > 0.
    f[2i+1] = m + 1       → f · γ_{i,−} = −m + (m+1) = 1 > 0.
    """
    m = _center_order(C)
    f_m = _gauss_jordan_solve(C, [m] * r)
    f = [0] * (2 * r)
    for i in range(r):
        assert f_m[i].denominator == 1, (
            f"adj witness f_m[{i}] = {f_m[i]} is not an integer; "
            f"unexpected for simply-laced ADE"
        )
        f[2 * i] = int(f_m[i])
        f[2 * i + 1] = m + 1
    return tuple(f)


# ---------------------------------------------------------------------------
# General Γ via explicit Lagrangian L ⊆ (P/Q)²
# ---------------------------------------------------------------------------
#
# A Lagrangian L ≤ (P/Q)² is supplied as a list of (m_vec, e_vec) integer
# r-vectors interpreted modulo CZ^r in each slot.  In SC coordinates each
# pair lifts to the rational vector
#
#     (C⁻¹·m_vec  in magnetic slots,   e_vec  in electric slots).
#
# The lattice Γ in SC coords is then the Z-span of Λ₀'s integer basis
# together with these rational lifts.  We compute a Z-basis of Γ via
# integer Hermite Normal Form (after clearing denominators by det(C)),
# then change coordinates so that Γ becomes Z^{2r} in the new basis.


def _hnf_columns(M_in: list[list[int]]) -> tuple[list[list[int]], int]:
    """Column Hermite Normal Form of an integer matrix.

    Returns ``(H, rank)`` where ``H`` contains the leftmost ``rank``
    columns (the Z-basis of the column span).
    """
    n = len(M_in)
    if n == 0:
        return [], 0
    M = [row[:] for row in M_in]
    cols = len(M[0])
    p = 0
    for r in range(n):
        if p >= cols:
            break
        # Repeated Euclidean reduction on row r over columns [p, cols).
        while True:
            cands = [c for c in range(p, cols) if M[r][c] != 0]
            if not cands:
                break
            mc = min(cands, key=lambda c: abs(M[r][c]))
            if mc != p:
                for rr in range(n):
                    M[rr][p], M[rr][mc] = M[rr][mc], M[rr][p]
            if M[r][p] < 0:
                for rr in range(n):
                    M[rr][p] = -M[rr][p]
            piv = M[r][p]
            for c in range(p + 1, cols):
                if M[r][c] != 0:
                    q = M[r][c] // piv  # floor div: 0 ≤ remainder < piv
                    if q != 0:
                        for rr in range(n):
                            M[rr][c] -= q * M[rr][p]
            if all(M[r][c] == 0 for c in range(p + 1, cols)):
                break
        if M[r][p] != 0:
            p += 1
    return [row[:p] for row in M], p


def _matrix_inverse_q(M: list[list]) -> list[list[Fraction]]:
    """Inverse of a square rational matrix via Gauss-Jordan."""
    n = len(M)
    A = [[Fraction(M[i][j]) for j in range(n)] +
         [Fraction(1) if i == j else Fraction(0) for j in range(n)]
         for i in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if A[r][col] != 0), None)
        if piv is None:
            raise ValueError("singular matrix")
        A[col], A[piv] = A[piv], A[col]
        d = A[col][col]
        A[col] = [x / d for x in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [a - f * b for a, b in zip(A[r], A[col])]
    return [row[n:] for row in A]


def _det_q(M: list[list]) -> Fraction:
    """Determinant of a rational square matrix."""
    n = len(M)
    A = [[Fraction(M[i][j]) for j in range(n)] for i in range(n)]
    det = Fraction(1)
    for col in range(n):
        piv = next((r for r in range(col, n) if A[r][col] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != col:
            A[col], A[piv] = A[piv], A[col]
            det = -det
        det *= A[col][col]
        d = A[col][col]
        A[col] = [x / d for x in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [a - f * b for a, b in zip(A[r], A[col])]
    return det


def _block_diag_cartan(C_blocks: list[list[list[int]]]) -> list[list[int]]:
    """Block-diagonal combined Cartan from per-factor Cartan matrices."""
    r_total = sum(len(C) for C in C_blocks)
    out = [[0] * r_total for _ in range(r_total)]
    off = 0
    for C in C_blocks:
        r = len(C)
        for i in range(r):
            for j in range(r):
                out[off + i][off + j] = C[i][j]
        off += r
    return out


def _verify_lagrangian_linking_form(
    C: list[list[int]],
    generators: list[tuple[Sequence[int], Sequence[int]]],
) -> None:
    """Raise ValueError if any pairwise linking value is non-integer.

    λ((m,e),(m',e')) = m^T C⁻¹ e' − m'^T C⁻¹ e ∈ Z.
    """
    r = len(C)
    if any(len(m) != r or len(e) != r for (m, e) in generators):
        raise ValueError(
            f"generator length mismatch: each (m, e) must be length r={r}"
        )
    # C^{-1} as a rational matrix (column j is C^{-1} · e_j).
    Cinv_cols = [_gauss_jordan_solve(
        C, [Fraction(1) if k == j else Fraction(0) for k in range(r)]
    ) for j in range(r)]
    Cinv = [[Cinv_cols[j][i] for j in range(r)] for i in range(r)]

    def Cinv_dot(u, v):
        return sum(Fraction(u[i]) * Cinv[i][j] * Fraction(v[j])
                   for i in range(r) for j in range(r))

    n = len(generators)
    for i in range(n):
        m_i, e_i = generators[i]
        for j in range(i, n):
            m_j, e_j = generators[j]
            val = Cinv_dot(m_i, e_j) - Cinv_dot(m_j, e_i)
            if val.denominator != 1:
                raise ValueError(
                    f"Lagrangian linking form λ(g_{i}, g_{j}) = {val} "
                    f"is not in Z; the supplied generators are not "
                    f"isotropic for the linking form on (P/Q)²"
                )


def _lambda0_basis_in_sc(C: list[list[int]]) -> list[list[int]]:
    """Z-basis of Λ₀ = P^∨ ⊕ Q in SC coordinates, as a 2r × 2r integer matrix.

    Columns 0..r-1: e_{2i} (magnetic basis: fundamental coweights).
    Columns r..2r-1: roots in SC coords (electric: Σ_j C[k][j] e_{2j+1}).
    """
    r = len(C)
    M = [[0] * (2 * r) for _ in range(2 * r)]
    for i in range(r):
        M[2 * i][i] = 1
    for k in range(r):
        for j in range(r):
            M[2 * j + 1][r + k] = C[k][j]
    return M


def _gamma_basis_in_sc(
    C: list[list[int]],
    generators: list[tuple[Sequence[int], Sequence[int]]],
) -> list[list[Fraction]]:
    """Compute a Z-basis of Γ in SC coordinates as a 2r × 2r rational matrix.

    Procedure
    ---------
    1. Form the 2r × (2r + |generators|) generator matrix in SC coords:
       Λ₀ basis (integer) plus lifted Lagrangian generators
       (C⁻¹·m magnetic, e electric — rational with denom dividing det(C)).
    2. Multiply every entry by m = det(C) → integer matrix.
    3. Compute column HNF → 2r × 2r integer basis of m·Γ.
    4. Divide by m → Z-basis of Γ in SC coords.

    Raises
    ------
    ValueError
        If Γ is not unimodular (|det(basis)| ≠ 1) — happens when the
        generators do not span a Lagrangian of the right size.
    """
    r = len(C)
    m = _center_order(C)
    # Λ₀ basis (integer 2r × 2r).
    cols: list[list[Fraction]] = []
    M_lambda0 = _lambda0_basis_in_sc(C)
    for k in range(2 * r):
        col = [Fraction(M_lambda0[i][k]) for i in range(2 * r)]
        cols.append(col)
    # Lagrangian lifts (rational).
    for (m_vec, e_vec) in generators:
        y = _gauss_jordan_solve(C, list(m_vec))   # C⁻¹ · m
        col = [Fraction(0)] * (2 * r)
        for j in range(r):
            col[2 * j] = y[j]
            col[2 * j + 1] = Fraction(int(e_vec[j]))
        cols.append(col)
    # Multiply by m to get integer matrix.
    n_cols = len(cols)
    int_mat = [[0] * n_cols for _ in range(2 * r)]
    for c in range(n_cols):
        for i in range(2 * r):
            v = cols[c][i] * m
            assert v.denominator == 1, (
                f"unexpected non-integer entry after scaling by m={m}: {v}"
            )
            int_mat[i][c] = int(v)
    # Column HNF.
    H, rank = _hnf_columns(int_mat)
    if rank != 2 * r:
        raise ValueError(
            f"Lagrangian generators span a rank-{rank} extension of Λ₀; "
            f"expected rank {2 * r} for unimodular Γ"
        )
    # Divide by m to recover Γ-basis in SC coords.
    M = [[Fraction(H[i][j], m) for j in range(2 * r)] for i in range(2 * r)]
    # Sanity: |det M| must equal 1 (Γ is unimodular).  In general,
    # |det M_Γ_in_SC| = m / |L|, so |L| = m / |det M|.
    d = _det_q(M)
    if abs(d) != 1:
        L_size = Fraction(m) / abs(d)
        raise ValueError(
            f"Γ is not unimodular: |det(Γ-basis)| = {abs(d)} ≠ 1.  "
            f"The supplied Lagrangian has size |L| = m/|det| = {L_size} "
            f"but a Lagrangian must have size m = det(C) = {m}; enlarge "
            f"or shrink the generator set"
        )
    return M


def _apply_basis_change(
    M_gamma: list[list[Fraction]],
    B_sc: list[list[int]],
    nodes_sc: list[Vec],
    spec_sc: list[Vec],
    witness_sc: Vec,
) -> tuple[list[list[int]], list[Vec], list[Vec], Vec]:
    """Re-express B, nodes, spec, witness in the Γ-basis given by ``M_gamma``.

    SC coords v_SC = M_gamma · v_Γ;   v_Γ = M_gamma⁻¹ · v_SC.
    Pairing: B_Γ = M_gamma^T · B_sc · M_gamma  (integer for unimodular Γ).
    Witness: f_Γ = M_gamma^T · f_sc.
    """
    n = len(M_gamma)
    M_inv = _matrix_inverse_q(M_gamma)

    def transport_vec(v: Vec) -> Vec:
        v_q = [Fraction(int(x)) for x in v]
        out_q = [sum(M_inv[i][j] * v_q[j] for j in range(n)) for i in range(n)]
        out: list[int] = []
        for i, x in enumerate(out_q):
            if x.denominator != 1:
                raise ValueError(
                    f"vector {v} has non-integer Γ-coordinate {x} at "
                    f"position {i}; not in Γ"
                )
            out.append(int(x))
        return tuple(out)

    nodes_g = [transport_vec(g) for g in nodes_sc]
    spec_g = [transport_vec(g) for g in spec_sc]

    # B_Γ = M^T B_sc M.
    BM = [[sum(B_sc[i][k] * M_gamma[k][j] for k in range(n)) for j in range(n)]
          for i in range(n)]
    B_g = [[sum(M_gamma[k][i] * BM[k][j] for k in range(n)) for j in range(n)]
           for i in range(n)]
    B_int = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            v = B_g[i][j]
            if v.denominator != 1:
                raise ValueError(
                    f"transported pairing has non-integer entry "
                    f"B_Γ[{i}][{j}] = {v}"
                )
            B_int[i][j] = int(v)

    # f_Γ = M^T f_sc; clear denominators by scaling — positivity is
    # preserved under positive integer rescaling.
    f_q = [sum(Fraction(witness_sc[k]) * M_gamma[k][i] for k in range(n))
           for i in range(n)]
    from math import lcm
    denom = 1
    for x in f_q:
        denom = lcm(denom, x.denominator)
    f_int = [int(x * denom) for x in f_q]
    # Sanity-check positivity on every node.
    for g in nodes_g:
        val = sum(f_int[k] * g[k] for k in range(n))
        if val <= 0:
            # Should never happen if the SC witness was valid and Γ contains
            # Λ₀.  Raise loudly rather than silently produce a bad witness.
            raise ValueError(
                f"transported cone witness {tuple(f_int)} is not strictly "
                f"positive on node {g} (got {val}); SC witness {witness_sc} "
                f"and basis change M = {M_gamma}"
            )
    witness_g = tuple(f_int)
    return B_int, nodes_g, spec_g, witness_g


# ---------------------------------------------------------------------------
# Per-simple-factor data
# ---------------------------------------------------------------------------


def _simple_factor_data(tp: str, n: int, global_form: str = "sc") -> dict:
    """(B, nodes, spec, cone_witness) for a single simple ADE factor.

    Parameters
    ----------
    global_form
        ``"sc"`` (default) — simply-connected Γ = P^∨ ⊕ P.
        ``"adj"``           — adjoint Γ = Q^∨ ⊕ Q.

    Lattice: Z^{2r} with standard symplectic pairing B (same for both forms).
    Nodes:   differ by global form (see _sc_node_charges, _adj_node_charges).
    Spec:    bipartite Coxeter sequence applied to the chosen node charges.
    Witness: positive linear form certifying the cone is pointed.
    """
    if global_form not in ("sc", "adj"):
        raise ValueError(
            f"global_form must be 'sc' or 'adj', got {global_form!r}"
        )
    r, h, edges = _factor_data(tp, n)
    C = _cartan_matrix(r, edges)
    B = _standard_symplectic_pairing(r)
    if global_form == "sc":
        nodes = _sc_node_charges(r, C)
        witness = _weyl_cone_witness(r, C)
    else:
        nodes = _adj_node_charges(r, C)
        witness = _adj_cone_witness(r, C)
    seq = _bipartite_coxeter_sequence(r, h)
    Q = _bps.BPSQuiver.from_pairing(nodes, B)
    spec = list(Q.build_spectrum_generator(seq))
    return {"r": r, "h": h, "C": C, "B": B, "nodes": nodes,
            "spec": spec, "cone_witness": witness}


# ---------------------------------------------------------------------------
# Block-diagonal assembly
# ---------------------------------------------------------------------------


def _block_pad(v: Sequence[int], start: int, total: int) -> Vec:
    """Embed a local vector into position [start : start + len(v)] in R^total."""
    out = [0] * total
    for k, x in enumerate(v):
        out[start + k] = int(x)
    return tuple(out)


def _is_lagrangian_pair_list(spec) -> bool:
    """True if *spec* looks like a list of ``(m_vec, e_vec)`` integer pairs."""
    if not isinstance(spec, (list, tuple)) or len(spec) == 0:
        return False
    first = spec[0]
    if not isinstance(first, (list, tuple)) or len(first) != 2:
        return False
    m, e = first
    return (isinstance(m, (list, tuple)) and isinstance(e, (list, tuple))
            and not isinstance(m, str) and not isinstance(e, str))


def _assemble_blocks(
    factors: Sequence[tuple[str, int]],
    global_forms=None,
) -> dict:
    """Block-diagonal (B, nodes, spec, cone_witness) for a list of parsed ADE factors.

    Parameters
    ----------
    factors
        List of ``(type, rank)`` pairs (already parsed via ``_parse_factor``).
    global_forms
        Choice of charge lattice Γ.  Accepts:

        * ``None`` or ``"sc"`` — all factors use simply-connected.
        * ``"adj"``            — all ADE factors use adjoint.
        * A list of strings (one ``"sc"``/``"adj"`` per factor).
        * A list of ``(m_vec, e_vec)`` integer-tuple pairs — a combined
          Lagrangian L ⊆ (P/Q)² acting on the **product** of the simple
          factors' centres.  Each ``m_vec`` and ``e_vec`` has length
          equal to the total rank of the ADE factors (excluding U(1)).
          U(1) factors are unaffected.

    Each simple ADE factor contributes a 2r-dimensional symplectic block.
    U(1) factors contribute a rank-2 symplectic block with no quiver nodes.
    """
    # Distinguish the explicit-Lagrangian path from the per-factor named
    # path before normalising.
    explicit_lagrangian = _is_lagrangian_pair_list(global_forms)

    if explicit_lagrangian:
        forms = ["sc"] * len(factors)   # build SC assembly, then change basis
    elif global_forms is None or global_forms == "sc":
        forms = ["sc"] * len(factors)
    elif global_forms == "adj":
        forms = ["adj"] * len(factors)
    else:
        forms = list(global_forms)
        if any(not isinstance(f, str) for f in forms):
            raise ValueError(
                "global_forms must be 'sc', 'adj', a list of those, or "
                "a list of (m_vec, e_vec) integer-tuple pairs"
            )
        if len(forms) != len(factors):
            raise ValueError(
                f"global_forms has length {len(forms)} but there are "
                f"{len(factors)} factors"
            )

    # Collect per-factor raw data and dimensions.
    per_factor: list[dict | None] = []
    dims: list[int] = []
    for (tp, n), gf in zip(factors, forms):
        if tp == "U":
            per_factor.append(None)
            dims.append(2)
        else:
            d = _simple_factor_data(tp, n, global_form=gf)
            per_factor.append(d)
            dims.append(2 * d["r"])

    total = sum(dims)
    B: list[list[int]] = [[0] * total for _ in range(total)]
    nodes: list[Vec] = []
    spec: list[Vec] = []
    witness_parts: list[int] = []
    factor_info: list[dict] = []

    offset = 0
    for (tp, n), dim, d, gf in zip(factors, dims, per_factor, forms):
        if tp == "U":
            # Rank-2 symplectic block; no quiver nodes.
            B[offset][offset + 1] = 1
            B[offset + 1][offset] = -1
            witness_parts.extend([0, 0])
            factor_info.append({"tp": "U", "n": 1, "r": 0, "h": 0,
                                 "start": offset, "dim": 2})
        else:
            # Copy local symplectic block into global matrix.
            for i in range(dim):
                for j in range(dim):
                    B[offset + i][offset + j] = d["B"][i][j]
            # Pad local node / spec charges.
            s = offset
            for g in d["nodes"]:
                nodes.append(_block_pad(g, s, total))
            for g in d["spec"]:
                spec.append(_block_pad(g, s, total))
            witness_parts.extend(d["cone_witness"])
            factor_info.append({"tp": tp, "n": n, "r": d["r"], "h": d["h"],
                                 "start": offset, "dim": dim,
                                 "global_form": gf})
        offset += dim

    result = {
        "B": B,
        "nodes": nodes,
        "spec": spec,
        "cone_witness": tuple(witness_parts),
        "dim": total,
        "factor_info": factor_info,
    }

    # Explicit-Lagrangian path: change basis from SC to Γ on the combined
    # symplectic system, leaving U(1) blocks untouched.
    if explicit_lagrangian:
        result = _apply_explicit_lagrangian(result, per_factor, global_forms)

    return result


def _apply_explicit_lagrangian(
    sc_result: dict,
    per_factor: list,
    generators: list,
) -> dict:
    """Change basis on the SC assembly to the Γ specified by an explicit L.

    The Lagrangian generators act on the *product* of simple-factor centres;
    U(1) blocks are left unchanged (they have no quiver / no centre).
    """
    # Extract per-factor Cartan blocks and the contiguous range of SC
    # coordinates corresponding to ADE blocks.
    ade_blocks = [d for d in per_factor if d is not None]
    if not ade_blocks:
        if generators:
            raise ValueError(
                "explicit Lagrangian generators given, but no ADE factors"
            )
        return sc_result

    C_total = _block_diag_cartan([d["C"] for d in ade_blocks])
    r_total = len(C_total)

    # Validate generator shapes.
    for (m_vec, e_vec) in generators:
        if len(m_vec) != r_total or len(e_vec) != r_total:
            raise ValueError(
                f"each Lagrangian generator (m, e) must have length "
                f"r_total={r_total} (sum of ADE-factor ranks); got "
                f"({len(m_vec)}, {len(e_vec)})"
            )
    norm_gens = [(tuple(int(x) for x in m), tuple(int(x) for x in e))
                 for (m, e) in generators]

    # Verify the linking form is Z-valued on all pairs.
    _verify_lagrangian_linking_form(C_total, norm_gens)

    # Compute Γ-basis on the ADE part (2 r_total × 2 r_total rational matrix)
    # — this also raises if |L| ≠ det(C_total).
    M_ade = _gamma_basis_in_sc(C_total, norm_gens)

    # Build the global change-of-basis matrix M (block-diagonal): identity
    # on U(1) slots, M_ade on each ADE slot.  ADE blocks live at the SC
    # coordinates between the relevant per-factor offsets, *interleaved*
    # with the SC magnetic/electric pattern.  In _assemble_blocks the ADE
    # block of factor f occupies a contiguous range [offset, offset + 2r_f);
    # we splice M_ade across these by position.
    fi = sc_result["factor_info"]
    total = sc_result["dim"]
    M_full = [[Fraction(1) if i == j else Fraction(0) for j in range(total)]
              for i in range(total)]
    # Assemble M_ade block-by-block by mapping each ADE factor's local
    # 2r_f × 2r_f sub-block onto its (offset, offset+2r_f) slice.  M_ade
    # is built from the *concatenated* simple-factor C's, so the i-th ADE
    # block in M_ade corresponds to the i-th ADE factor in fi.
    ade_offsets: list[tuple[int, int]] = []  # (factor_offset, local_dim)
    for d_info in fi:
        if d_info["tp"] == "U":
            continue
        ade_offsets.append((d_info["start"], d_info["dim"]))
    # M_ade rows/cols are also block-diagonal across simple factors (Λ₀
    # has no cross-block coupling, and the author's combined Lagrangian
    # mixes only via the explicit generators — which still respect block
    # structure since C_total is block-diagonal).  So we copy each ADE
    # diagonal block into the corresponding global slice.
    local_off = 0
    for (g_off, g_dim) in ade_offsets:
        # Copy M_ade[local_off:local_off+g_dim, local_off:local_off+g_dim]
        # into M_full[g_off:g_off+g_dim, g_off:g_off+g_dim].
        # Off-diagonal mixing across factors *can* exist in M_ade if the
        # Lagrangian mixes factors; we copy the whole block in two passes.
        local_off += g_dim
    # Two-pass copy of full M_ade (handles cross-factor mixing too).
    # Map: ADE-factor i's local SC-coords [start_i, start_i + 2r_i) in
    # M_ade ↔ global SC-coords [g_off_i, g_off_i + 2r_i).
    local_starts: list[int] = []
    s = 0
    for (g_off, g_dim) in ade_offsets:
        local_starts.append(s)
        s += g_dim
    for i_blk, (g_off_i, g_dim_i) in enumerate(ade_offsets):
        ls_i = local_starts[i_blk]
        for j_blk, (g_off_j, g_dim_j) in enumerate(ade_offsets):
            ls_j = local_starts[j_blk]
            for u in range(g_dim_i):
                for v in range(g_dim_j):
                    M_full[g_off_i + u][g_off_j + v] = M_ade[ls_i + u][ls_j + v]

    B_new, nodes_new, spec_new, witness_new = _apply_basis_change(
        M_full,
        sc_result["B"],
        sc_result["nodes"],
        sc_result["spec"],
        sc_result["cone_witness"],
    )

    new_factor_info = []
    for d_info in fi:
        d2 = dict(d_info)
        d2["global_form"] = "explicit" if d_info["tp"] != "U" else "U1"
        new_factor_info.append(d2)

    return {
        "B": B_new,
        "nodes": nodes_new,
        "spec": spec_new,
        "cone_witness": witness_new,
        "dim": total,
        "factor_info": new_factor_info,
        "lagrangian_generators": norm_gens,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pure_ade_lattice_data(
    factors: Sequence[FactorSpec],
    global_form=None,
) -> dict:
    """Return the raw lattice data for a pure ADE gauge theory.

    Parameters
    ----------
    factors
        Ordered list of factor specs.  Each entry may be:

        * ``("A", n)`` — type-A simple factor, rank n  (n ≥ 1).
        * ``("D", n)`` — type-D simple factor, rank n  (n ≥ 4).
        * ``("E", n)`` — type-E simple factor          (n ∈ {6, 7, 8}).
        * ``"U1"``     — abelian U(1); adds a rank-2 symplectic block
                         to the lattice but no quiver nodes.

    global_form
        Choice of charge lattice Γ.  Valid Γ are unimodular lattices with
        Λ₀ = P^∨ ⊕ Q ⊆ Γ ⊆ Q^∨ ⊕ P, classified by Lagrangian subgroups
        L ⊆ (P/Q)².

        * ``None`` or ``"sc"`` (default) — simply-connected: Γ = P^∨ ⊕ P.
        * ``"adj"``                       — adjoint: Γ = Q^∨ ⊕ Q.
        * A sequence of per-factor strings ``["sc", "adj", …]``.
        * A list of ``(m_vec, e_vec)`` integer-tuple pairs describing a
          Lagrangian L on the product of all simple-factor centres.

    Returns
    -------
    dict with keys:
        ``B``            — symplectic pairing matrix (list[list[int]]).
        ``nodes``        — BPS-quiver node charges (list[Vec]).
        ``spec``         — spectrum-generator charges (list[Vec]).
        ``cone_witness`` — positivity witness (Vec).
        ``dim``          — total lattice rank (int).
        ``factor_info``  — per-factor metadata (list[dict]).
        ``lagrangian_generators`` — (explicit-Lagrangian path only).
    """
    if not factors:
        raise ValueError("pure_ade_lattice_data: at least one factor is required")
    parsed = [_parse_factor(f) for f in factors]
    return _assemble_blocks(parsed, global_forms=global_form)


def pure_ade_kalgebra(
    factors: Sequence[FactorSpec],
    global_form=None,
) -> "BPSKAlgebra | None":
    """Build a ``BPSKAlgebra`` for a pure ADE gauge theory.

    Parameters
    ----------
    factors, global_form
        Same as ``pure_ade_lattice_data``.

    Returns
    -------
    BPSKAlgebra
        The K-theoretic Coulomb-branch algebra for the theory.
    None
        For pure U(1)^s theories (no BPS quiver nodes).

    Examples
    --------
    >>> alg = pure_ade_kalgebra([("A", 2)])          # SU(3)
    >>> alg = pure_ade_kalgebra([("D", 4)])          # SO(8)
    >>> alg = pure_ade_kalgebra([("A", 2)], global_form="adj")
    >>> alg = pure_ade_kalgebra([("A", 1), ("D", 4)])
    """
    data = pure_ade_lattice_data(factors, global_form=global_form)
    if not data["nodes"]:
        return None
    return BPSKAlgebra(
        pairing=data["B"],
        node_charges=data["nodes"],
        spec=data["spec"],
        cone_witness=data["cone_witness"],
        verify="off",
        # A10 η-shell prune: opt-in here ONLY — these are exactly the
        # presentations where the joint bound was validated byte-exact
        # (e6 984 stratified samples, e8 banked seeds, zoo/AD suites);
        # it is unsound in general frames.
        k_joint_prune=True,
    )


# ---------------------------------------------------------------------------
# Physical naming: magnetic / electric decomposition
# ---------------------------------------------------------------------------
#
# The SC/adj interleaved basis always has:
#   even index 2i   → magnetic slot (ω_i^∨ or α_i direction)
#   odd  index 2i+1 → electric slot (ω_i or α_i direction)
#
# These two extractors work for the full coordinate vector, including
# multi-factor theories.  For theories with U(1) blocks the even/odd
# interpretation applies to every pair of coordinates, including the U(1)
# pair (which contributes a standard symplectic block).


def magnetic_charge(gamma: Vec) -> Vec:
    """Magnetic part (v_1, …, v_r) of a tropical charge in SC/adj interleaved basis.

    Returns the even-indexed coordinates of *gamma*, i.e. the ω_i^∨
    (fundamental coweight) components.  For pure ADE theories this gives
    the full cocharacter label; for product theories each factor's magnetic
    slots appear in block order.
    """
    r = len(gamma) // 2
    return tuple(gamma[2 * i] for i in range(r))


def electric_charge(gamma: Vec) -> Vec:
    """Electric part (e_1, …, e_r) of a tropical charge in SC/adj interleaved basis.

    Returns the odd-indexed coordinates of *gamma*, i.e. the ω_i
    (fundamental weight) components.  A charge is a Wilson-line charge iff
    all magnetic components are zero and all electric components ≤ 0.
    """
    r = len(gamma) // 2
    return tuple(gamma[2 * i + 1] for i in range(r))


# ---------------------------------------------------------------------------
# Physical naming: Wilson lines
# ---------------------------------------------------------------------------
#
# A **Wilson line** L_{(0,w)} is a canonical-basis element whose tropical
#
# A **Wilson line** L_{(0,w)} is a canonical-basis element whose tropical
# charge has zero in every magnetic slot.  Its physical label is the
# representation of G whose lowest weight is w = (w_1, …, w_r).
#
# These functions work for the SC and adj global forms.  For an explicit-
# Lagrangian form the basis may be rotated and "zero magnetic" is not
# generally the same condition — callers should verify form="sc" or "adj".


def is_wilson_line(gamma: Vec) -> bool:
    """True if *gamma* is a Wilson-line charge in the SC basis.

    Conditions (SC/adj interleaved basis, magnetic at even indices):
      1. All magnetic coordinates are zero.
      2. At least one electric coordinate is non-zero.
      3. All electric coordinates are ≤ 0  (anti-dominant = lowest-weight
         condition: ⟨w, α_i^∨⟩ = w_i ≤ 0 for all i in the SC basis).

    Purely-electric charges with any w_i > 0 are not valid lowest weights
    of finite-dimensional irreps, so they do not label Wilson lines.
    Returns False for the zero vector (identity element).
    """
    r = len(gamma) // 2
    elec = [gamma[2 * i + 1] for i in range(r)]
    return (
        all(gamma[2 * i] == 0 for i in range(r))
        and any(x != 0 for x in elec)
        and all(x <= 0 for x in elec)
    )


def wilson_weight(gamma: Vec) -> Vec:
    """Extract the electric weight w = (w_1, …, w_r) from a Wilson-line charge.

    For *gamma* = (0, w_1, 0, w_2, …, 0, w_r) this returns (w_1, …, w_r).
    The weight w is the lowest weight of the representation labeling the
    Wilson line; for dominant representations all w_i ≤ 0.

    Does not check that *gamma* is actually a Wilson line.
    """
    r = len(gamma) // 2
    return tuple(gamma[2 * i + 1] for i in range(r))


# ---------------------------------------------------------------------------
# Physical naming: 't Hooft lines
# ---------------------------------------------------------------------------
#
# In pure ADE gauge theory, 't Hooft lines are labelled by *dominant
# cocharacters* (= dominant coweights) of the gauge group.  Under the
# simply-laced identification P^∨ ≅ Q induced by the inner product (with
# basis matching ω_i^∨ ↔ α_i), an integer vector v = (v_1, …, v_r) ∈ P^∨
# corresponds to the element  v ↦ ṽ = Σ_i v_i α_i ∈ Q.  Then ṽ is dominant
# iff ⟨α_i, ṽ⟩ ≥ 0 for every simple root α_i, i.e.
#
#     C · v  ≥  0   componentwise,
#
# where C is the (block-diagonal) Cartan matrix of the ADE factors.
#
# In the SC interleaved basis (e_{2i} = ω_i^∨, e_{2i+1} = ω_i), a
# tropical charge of the form
#
#     γ  =  (v_1, 0, v_2, 0, …, v_r, 0)        (purely magnetic)
#
# is the canonical-basis label of the 't Hooft loop  T(ṽ)  iff C·v ≥ 0.
# Empirical fact (verified for A_1, A_2, A_3 across all of [1,4]^r and
# beyond): for such γ, F_γ = X_γ is a single quantum-torus monomial,
# and the family { F_γ : γ ↔ dominant cocharacter } is a *commutative*
# subalgebra (matching the physical expectation for 't Hooft lines).
#
# The boundary cases C·v on a wall (some component = 0) include the
# fundamental coweights ω_i^∨ themselves, which are the SC node charges
# γ_{i,+}.  Those have non-monomial F's with quantum corrections and do
# *not* literally commute — they satisfy only the bar-twisted relation
# C_{ab}^c(q) = C_{ba}^c(q^{-1}).


def _block_diag_cartan_from_factor_info(factor_info: list[dict]) -> list[list[int]]:
    """Block-diagonal Cartan matrix from ``factor_info`` (ADE blocks only).

    U(1) factors contribute no rows/columns.  Used to evaluate the
    dominance condition ``C·v ≥ 0`` for 't Hooft-line detection.
    """
    blocks = []
    for fi in factor_info:
        if fi["tp"] == "U":
            continue
        r, _, edges = _factor_data(fi["tp"], fi["n"])
        blocks.append(_cartan_matrix(r, edges))
    return _block_diag_cartan(blocks)


def is_thooft_line(gamma: Vec, factor_info: list[dict]) -> bool:
    """True if *gamma* is the tropical charge of a 't Hooft line in SC ADE.

    Conditions (SC interleaved basis, magnetic at even indices):
      1. All electric coordinates are zero (purely magnetic).
      2. At least one magnetic coordinate is non-zero.
      3. ``C · v ≥ 0`` componentwise, where v = (v_1, …, v_r) are the
         magnetic coordinates and C is the block-diagonal ADE Cartan
         from *factor_info*.  This is the dominance condition for the
         corresponding cocharacter under the P^∨ ≅ Q identification.

    Returns False for the identity (zero vector).  Only meaningful for
    SC global form.
    """
    r_total = sum(fi["r"] for fi in factor_info if fi["tp"] != "U")
    # Magnetic / electric components in the ADE part of the lattice.
    # U(1) blocks don't host quiver charges, so we ignore them here.
    v: list[int] = []
    elec: list[int] = []
    for fi in factor_info:
        if fi["tp"] == "U":
            continue
        s = fi["start"]
        for i in range(fi["r"]):
            v.append(gamma[s + 2 * i])
            elec.append(gamma[s + 2 * i + 1])
    if any(x != 0 for x in elec):
        return False
    if all(x == 0 for x in v):
        return False
    C = _block_diag_cartan_from_factor_info(factor_info)
    Cv = [sum(C[i][j] * v[j] for j in range(r_total)) for i in range(r_total)]
    return all(x >= 0 for x in Cv)


def thooft_cocharacter(gamma: Vec, factor_info: list[dict]) -> Vec:
    """Extract the cocharacter v = (v_1, …, v_r) of a 't Hooft-line charge.

    For *gamma* = (v_1, 0, v_2, 0, …, v_r, 0) (purely magnetic in the SC
    basis) this returns (v_1, …, v_r).  Under P^∨ ≅ Q this labels the
    cocharacter ṽ = Σ_i v_i α_i ∈ Q; *gamma* corresponds to a 't Hooft
    line iff ṽ is dominant (C·v ≥ 0).

    Does not check that *gamma* is actually a 't Hooft line.
    """
    out: list[int] = []
    for fi in factor_info:
        if fi["tp"] == "U":
            continue
        s = fi["start"]
        for i in range(fi["r"]):
            out.append(gamma[s + 2 * i])
    return tuple(out)


# ---------------------------------------------------------------------------
# Physical naming: dyon monomials
# ---------------------------------------------------------------------------
#
# A canonical-basis element F_γ is a single quantum-torus monomial (F_γ = X_γ)
# exactly when its tropical charge satisfies:
#
#   (i)  v ≠ 0                (has magnetic charge)
#   (ii) C · v ≥ 0            (dominant cocharacter — the 't Hooft condition)
#   (iii) −C · v ≤ e ≤ 0     componentwise  (electric part in the "box")
#
# where  v = magnetic part of γ  and  e = electric part of γ.
#
# This generalises the pure 't Hooft case (e = 0, which satisfies (iii)
# trivially) and rules out pure Wilson lines (v = 0 forces C·v = 0, so
# (iii) becomes e = 0, leaving only the identity).
#
# Physical picture: F_γ for such γ is the "dyon operator" — a 't Hooft loop
# T(v̄) dressed by the lowest-weight Wilson line at weight e.  Concretely,
# T(v) · W(e) = D(v; e) + lower-magnetic-charge corrections.
#
# This monomial family is the coarsest piece of the filtration: the algebra
# A_q[T] is filtered by magnetic charge with Wilson lines at level 0 (the
# gr⁰ piece) and 't Hooft / dyon monomials generating the filtration.
#
# Equivalently labelled by dominant weights of the GNO-dual group and their
# electric dressings in the weight box [−ρ(v), 0].


def is_dyon_monomial(gamma: Vec, factor_info: list[dict]) -> bool:
    """True if *gamma* is the tropical charge of a genuine dyon monomial.

    "Genuine dyon" means  v ≠ 0  (magnetic)  **and**  e ≠ 0  (electric),
    with F_γ = X_γ a single monomial.  The monomial conditions are:

      1. At least one magnetic coordinate non-zero.
      2. At least one electric coordinate non-zero (all ≤ 0).
      3. ``C · v ≥ 0``  ('t Hooft dominance condition).
      4. ``e ≥ −C · v``  componentwise  (electric within the box).

    Returns False for 't Hooft lines (e = 0; use ``is_thooft_line``),
    Wilson lines (v = 0), and the identity.

    *factor_info* is the ``"factor_info"`` entry from
    ``pure_ade_lattice_data``.  Only ADE factors contribute to the
    Cartan check; U(1) blocks are ignored.
    """
    r_total = sum(fi["r"] for fi in factor_info if fi["tp"] != "U")
    v_list: list[int] = []
    e_list: list[int] = []
    for fi in factor_info:
        if fi["tp"] == "U":
            continue
        s = fi["start"]
        for i in range(fi["r"]):
            v_list.append(gamma[s + 2 * i])
            e_list.append(gamma[s + 2 * i + 1])
    # 1. Must have magnetic charge
    if all(x == 0 for x in v_list):
        return False
    # 2. Must have electric charge (and all ≤ 0 — fast reject for > 0)
    if all(x == 0 for x in e_list):
        return False
    if any(x > 0 for x in e_list):
        return False
    # 3. C · v ≥ 0  (dominance)
    C = _block_diag_cartan_from_factor_info(factor_info)
    Cv = [sum(C[i][j] * v_list[j] for j in range(r_total)) for i in range(r_total)]
    if any(cv < 0 for cv in Cv):
        return False
    # 4. e ≥ −C · v  (lower box bound)
    for i in range(r_total):
        if e_list[i] < -Cv[i]:
            return False
    return True


def element_name(gamma: Vec, factor_info: list[dict] | None = None) -> str:
    """Human-readable name for a canonical-basis element.

    The algebra is filtered by magnetic charge; each named family lives at
    a definite filter level:

    * Identity (zero vector)            → ``"1"``             (level 0)
    * Wilson line charge (0, w)         → ``"W(w)"``          (level 0)
      w is the lowest weight, e.g. ``"W(-1)"`` for A_1 fundamental or
      ``"W(0,-1)"`` for A_2 **3**.
    * 't Hooft line  (C·v ≥ 0, e = 0)  → ``"T(v)"``          (level v)
      v is the dominant cocharacter under P^∨ ≅ Q (ṽ = Σ v_i α_i).
      E.g. A_2 Weyl vector ρ^∨ = (1,1) → ``"T(1,1)"``.
    * Dyon monomial  (C·v ≥ 0,         → ``"D(v;e)"``        (level v)
                      −C·v ≤ e < 0)
      v is the dominant cocharacter, e is the electric weight.
      E.g. ``"D(1,1;-1,0)"`` for the A_2 ρ^∨-dyon at weight (−ω_1).
    * Other charges                     → ``str(gamma)``      (unknown)

    *factor_info* is the ``"factor_info"`` entry from
    ``pure_ade_lattice_data`` and is needed to detect 't Hooft lines and
    dyon monomials (both require the C·v dominance test).  Without it the
    function falls back to the Wilson + identity + raw cases.
    """
    if all(x == 0 for x in gamma):
        return "1"
    if is_wilson_line(gamma):
        w = wilson_weight(gamma)
        if len(w) == 1:
            return f"W({w[0]})"
        return "W(" + ",".join(str(x) for x in w) + ")"
    if factor_info is not None:
        if is_thooft_line(gamma, factor_info):
            v = thooft_cocharacter(gamma, factor_info)
            if len(v) == 1:
                return f"T({v[0]})"
            return "T(" + ",".join(str(x) for x in v) + ")"
        if is_dyon_monomial(gamma, factor_info):
            v = thooft_cocharacter(gamma, factor_info)
            e = electric_charge(gamma)
            vs = ",".join(str(x) for x in v)
            es = ",".join(str(x) for x in e)
            return f"D({vs};{es})"
    return str(gamma)


# ---------------------------------------------------------------------------
# Magnetic filtration of the algebra
# ---------------------------------------------------------------------------
#
# The algebra A_q[T] is filtered by **σ-orbits of dominant cocharacters**:
#
#     F_0  ⊆  F_{[v_1]}  ⊆  F_{[v_2]}  ⊆  …          (F_{[v]} ⊆ F_{[u]} for v ≤ u)
#     F_{[v]} · F_{[u]}  ⊆  F_{[v+u]}                (algebra filtration)
#
# with the following structural facts (confirmed empirically for SU(2)
# and SU(3)):
#
#   • Wilson lines W(w) live at level 0; F_0 is a closed subalgebra (the
#     Chebyshev / character ring of G in the SU(2) case).
#   • 't Hooft lines T(v) live at level [v] (a σ-orbit of dominant
#     cocharacters in P^∨_+).  In the SC interleaved basis T(v) has
#     tropical charge (v_1, 0, v_2, 0, …) whenever C·v ≥ 0, and
#     F_{T(v)} = X_{T(v)} is a single quantum-torus monomial.
#   • T(v_1) · T(v_2) = T(v_1 + v_2) **exactly** (no quantum corrections):
#     the 't Hooft subalgebra is a free commutative monoid on the
#     dominant cocharacters.
#   • σ (induced by ρ on tropical labels) **conjugates magnetic weights**:
#     σ moves between σ-conjugate dominant cocharacters within the same
#     filtration level.  E.g. SU(3): σ swaps T(2,1) ↔ T(1,2) along its
#     orbit, with electric shifts; both share the level [{(2,1),(1,2)}].
#     Wilson lines are σ-fixed.
#
# Filtration-level extractor (working principle, verified):
#
#     level(L_γ) = the σ-orbit of the dominant magnetic charge that the
#                  σ-orbit of γ passes through.
#
# Algorithmically: σ-iterate γ until the magnetic part of σ^k(γ) lands
# in the dominant chamber (C·v ≥ 0); that v's σ-orbit is the level.
#
# IMPORTANT: the filtration level of a canonical basis element L_γ is
# **NOT** read off from the magnetic part of γ alone.  Two elements with
# the same tropical magnetic charge can sit at very different filtration
# levels, and elements with negative tropical magnetic charge (e.g. the
# σ⁻¹ image of T(v)) sit at level [v], not level [-v].


def _ade_magnetic(gamma: Vec, factor_info: list[dict]) -> Vec:
    """Extract the magnetic part (v_1, …, v_r) of *gamma* across all ADE
    factors of *factor_info*, ignoring U(1) blocks."""
    out: list[int] = []
    for fi in factor_info:
        if fi["tp"] == "U":
            continue
        s = fi["start"]
        for i in range(fi["r"]):
            out.append(gamma[s + 2 * i])
    return tuple(out)


def _is_dominant_cocharacter(v: Vec, factor_info: list[dict]) -> bool:
    """True if *v* is a dominant cocharacter (C·v ≥ 0) for the ADE part."""
    r = sum(fi["r"] for fi in factor_info if fi["tp"] != "U")
    if not v or all(x == 0 for x in v):
        return True  # zero is dominant
    C = _block_diag_cartan_from_factor_info(factor_info)
    Cv = [sum(C[i][j] * v[j] for j in range(r)) for i in range(r)]
    return all(cv >= 0 for cv in Cv)


def sigma_orbit_canonical(
    gamma: Vec, factor_info: list[dict], alg, max_steps: int = 30
) -> Vec | None:
    """Canonical representative of the σ-orbit of *gamma* within its level.

    σ-iterates *gamma* forward and backward and identifies the orbit by:
      1. Restricting to σ-images whose magnetic part equals the level
         (= lex-max dominant magnetic in the orbit).
      2. Among those, picking the one with smallest absolute electric
         charge (then lex-min as tiebreak).

    Two ``gamma`` and ``gamma'`` lie in the same σ-orbit iff this function
    returns the same canonical ID for both (within the orbit-scan window).

    Returns ``None`` if no dominant magnetic is reached within
    *max_steps* iterations in either direction.

    Used to count distinct σ-orbits at each filtration level.  For SU(2)
    the count at level v is **4v** (= 2·C·v): 2v+1 monomial-dyon orbits
    anchored at (v, e) for e ∈ [−2v, 0], plus 2v−1 "rogue" orbits at
    (v, e) for e ∈ [−4v+1, −2v−1] (in the band complementary to the
    dyon box).
    """
    visited: list[Vec] = []
    cur = gamma
    for _ in range(max_steps):
        visited.append(cur)
        cur = alg.rho(cur)
    cur = gamma
    for _ in range(max_steps):
        cur = alg.rho_inverse(cur)
        visited.append(cur)
    dom_vs = [
        _ade_magnetic(x, factor_info)
        for x in visited
        if _is_dominant_cocharacter(_ade_magnetic(x, factor_info), factor_info)
    ]
    if not dom_vs:
        return None
    target_v = max(dom_vs)
    candidates = [
        x for x in visited if _ade_magnetic(x, factor_info) == target_v
    ]
    return min(candidates, key=lambda x: (sum(abs(c) for c in x), x))


def filtration_level_A1_closed_form(gamma: Vec) -> int:
    """Closed-form filtration level for SU(2) (A_1) tropical charges.

    For *gamma* = (a, b) in the SU(2) SC interleaved basis (a = magnetic,
    b = electric):

        level((a, b))  =  max(|a|, a + b, 0)

    This formula reads the filtration level directly off the tropical
    charge — no σ-iteration required.  Equivalent to ``filtration_level``
    on A_1 (verified across (a, b) ∈ [-5,5] × [-15,14], 330 pairs, 0
    mismatches).

    Geometric interpretation: the level is the larger of three linear
    functions  a,  −a,  a + b  (and 0 floor for the identity / Wilson
    region).  These three correspond to:

      • a    — the dominant magnetic (when γ is already in the dominant
                wedge)
      • −a   — σ pushes negative magnetic to positive (Weyl reflection
                via σ case 2)
      • a+b  — σ case 3 pushes (a, b > 0) to magnetic a+b

    The level is the lex-max dominant cocharacter that the σ-orbit of
    γ visits.

    Raises ``ValueError`` if *gamma* is not a length-2 vector.
    """
    if len(gamma) != 2:
        raise ValueError(f"A_1 tropical charge must have length 2, got {len(gamma)}")
    a, b = gamma
    return max(abs(a), a + b, 0)


def filtration_level_A2_closed_form(gamma: Vec) -> tuple[int, int]:
    """Closed-form filtration level for pure SU(3) (A_2) tropical charges.

    For *gamma* = (v1, e1, v2, e2) in the SU(3) SC interleaved basis
    (v1, v2 = magnetic; e1, e2 = electric):

        u1  =  max(v1, −v1, v1+e1, −v1+v2, −v1+v2+e2, −v2, 0)
        u2  =  max(v2, −v2, v2+e2, −v2+v1, −v2+v1+e1, −v1, 0)

    where the level is the pair  (u1, u2).  The formula is symmetric under
    the A_2 outer automorphism (swap nodes 1 ↔ 2), which exchanges
    (u1, u2) ↔ (u2, u1).

    This is a piecewise-linear formula with **6 linear pieces** per component
    (plus a 0 floor), compared to 3 pieces for the A_1 (SU(2)) formula.
    No σ-iteration is required.

    Verified against Wilson-product filtration-layer assignments on all 305
    canonical-basis elements with V_MAX = 4, M_W = 5 (0 mismatches).

    The 6 pieces for u1 reduce to the SU(2) formula when v2 = e2 = 0:

        u1|_{v2=e2=0}  =  max(v1, −v1, v1+e1, −v1, −v1, 0)
                       =  max(|v1|, v1+e1, 0)    (the A_1 formula)

    Geometric interpretation of the 6 pieces for u1:
      • v1, −v1, v1+e1  — the three A_1 pieces restricted to node 1
      • −v1+v2          — level from the off-diagonal magnetic difference
      • −v1+v2+e2       — off-diagonal piece augmented by node-2 electric
      • −v2             — Weyl reflection of node-2 magnetic (long Weyl element)

    Raises ``ValueError`` if *gamma* is not a length-4 vector.
    """
    if len(gamma) != 4:
        raise ValueError(f"A_2 tropical charge must have length 4, got {len(gamma)}")
    v1, e1, v2, e2 = gamma
    u1 = max(v1, -v1, v1 + e1, -v1 + v2, -v1 + v2 + e2, -v2, 0)
    u2 = max(v2, -v2, v2 + e2, -v2 + v1, -v2 + v1 + e1, -v1, 0)
    return (u1, u2)


def _weyl_orbit_of_basis_vector(i: int, C: list[list[int]]) -> list[Vec]:
    """W-orbit of the i-th fundamental coweight ω_i^∨, in the coweight basis.

    Simple reflections act as
        s_j(λ)_k  =  λ_k  −  λ_j · C_{jk},
    where λ = Σ_k λ_k ω_k^∨ and C is the Cartan matrix.  Generated by BFS
    starting from the standard basis vector e_i (= ω_i^∨ in the coweight
    basis).

    For simply-laced ADE the orbit is finite (|W·ω_i^∨| = |W| / |Stab(ω_i^∨)|).
    """
    r = len(C)
    start = tuple(1 if k == i else 0 for k in range(r))
    seen: set[Vec] = {start}
    frontier = [start]
    while frontier:
        nxt: list[Vec] = []
        for lam in frontier:
            for j in range(r):
                lj = lam[j]
                if lj == 0:
                    continue
                new = tuple(lam[k] - lj * C[j][k] for k in range(r))
                if new not in seen:
                    seen.add(new)
                    nxt.append(new)
        frontier = nxt
    return sorted(seen)


def filtration_level_ADE_closed_form(
    gamma: Vec, factor_info: list[dict]
) -> Vec:
    """Conjectural closed-form magnetic filtration level for pure simply-laced
    (ADE) gauge theories in the SC interleaved basis.

    **Conjecture (ADE, simply-connected gauge group).**
    Let G be a product ∏_α G_α of simply-laced simple factors, with each G_α
    of type X_{r_α} and Cartan matrix C^{(α)}.  Write the tropical charge in
    the SC interleaved basis as

        γ  =  ( v_1^{(α)}, e_1^{(α)}, …, v_{r_α}^{(α)}, e_{r_α}^{(α)} )_α

    (magnetic v at even indices, electric e at odd indices, U(1) factors
    contribute their own (v, e) pair untouched by the formula).  The
    magnetic filtration level u^{(α)} of the canonical-basis element L_γ
    decomposes blockwise; on each ADE block of rank r the i-th component
    (1 ≤ i ≤ r) is

        u_i(γ)  =  max( S_i(γ) )

    where S_i(γ) ⊂ ℤ collects:

      • 0                                  — vacuum / Wilson floor
      • −v_i                               — extra anti-coweight piece
      • ⟨w·ω_i^∨, v⟩ for each w ∈ W       — magnetic Weyl orbit of ω_i^∨
      • ⟨w·ω_i^∨, v⟩ + Σ_{j ∈ S} e_j for each
        w ∈ W and each non-empty subset
        S ⊆ {j : [w·ω_i^∨]_j > 0}          — electric enhancements

    Here W is the Weyl group of X_r, ω_i^∨ is the i-th fundamental coweight
    (the standard basis vector e_i in the coweight basis), and ⟨·, ·⟩ is the
    standard pairing of coweight-basis coordinates with v.  The full
    filtration level is the tuple (u^{(α)})_α concatenated across factors;
    U(1) factors contribute no level component.

    **Geometric content.**
    The 2r + 2 (counted with multiplicity) linear pieces in S_i define
    hyperplanes in the (v, e)-space; in each region (chamber) one piece
    dominates and gives the level.  This is a Weyl-chamber-style
    decomposition of the full magnetic+electric tropical space.

    **Verified.**  A_1 (SU(2)), A_2 (SU(3)) on 305 elements, and A_3 (SU(4))
    on 190 Wilson-product first-appearance assignments — 0 mismatches in all
    cases.  See the tests in `the suite in the source repository`.

    **Conjectured.**  All ADE; awaiting Wilson-product verification beyond
    A_3.  The non-empty-subset enhancement is essential for A_3 ω_2^∨: the
    orbit element (1, −1, 1) has positives {1, 3} and the piece
    +e_1 + e_3 is needed to fix γ = (0, 1, 0, 1, 0, 1) (level (1, 2, 1)).
    A_2 orbits all have ≤ 1 positive per element, so the subset rule
    reduces to single-element enhancements there.

    Parameters
    ----------
    gamma
        Tropical charge in the SC interleaved basis.  Length = ``sum(d["dim"]
        for d in factor_info)``.
    factor_info
        ``factor_info`` from ``pure_ade_lattice_data``.  Used for the per-
        factor Cartan matrices and offsets.

    Returns
    -------
    Tuple of integers (u^{(α)}_i)_{α, i}, one entry per ADE node across all
    factors.  U(1) factors contribute no entries.

    Raises
    ------
    ValueError
        If ``len(gamma)`` does not match ``factor_info``, or if the formula
        is invoked on a non-simply-laced factor (currently never, since
        ``pure_ade_lattice_data`` only produces ADE blocks).
    """
    expected_dim = sum(d["dim"] for d in factor_info)
    if len(gamma) != expected_dim:
        raise ValueError(
            f"gamma length {len(gamma)} does not match factor_info dim {expected_dim}"
        )

    levels: list[int] = []
    for fi in factor_info:
        if fi["tp"] == "U":
            # U(1): no level component
            continue
        # Simply-laced ADE block.
        tp, n = fi["tp"], fi["n"]
        if tp not in ("A", "D", "E"):
            raise ValueError(f"non-ADE factor type {tp!r} not supported")
        r = fi["r"]
        start = fi["start"]
        # Reconstruct the Cartan matrix for this block.
        _, _, edges = _factor_data(tp, n)
        C = _cartan_matrix(r, edges)
        # Extract magnetic / electric coordinates for this block.
        v = [gamma[start + 2 * k] for k in range(r)]
        e = [gamma[start + 2 * k + 1] for k in range(r)]
        for i in range(r):
            orbit = _weyl_orbit_of_basis_vector(i, C)
            pieces: list[int] = [0, -v[i]]
            for m in orbit:
                base = sum(m[k] * v[k] for k in range(r))
                pieces.append(base)
                # Electric enhancements: one piece per non-empty subset of the
                # positive-component nodes of m.  For |positives| = 1 this
                # reduces to a single +e_j piece (the only subset).  For
                # higher rank, multi-positive orbit elements (like (1,-1,1)
                # in W·ω_2^∨ for A_3) contribute partial sums and the full
                # sum, which is needed to recover the SU(4) middle-node level.
                positives = [j for j in range(r) if m[j] > 0]
                np = len(positives)
                for mask in range(1, 1 << np):
                    inc = 0
                    for k in range(np):
                        if mask & (1 << k):
                            inc += e[positives[k]]
                    pieces.append(base + inc)
            levels.append(max(pieces))
    return tuple(levels)


def filtration_level(
    gamma: Vec, factor_info: list[dict], alg, max_steps: int = 50
) -> Vec | None:
    """Filtration level of *L_γ*, returned as a representative dominant
    cocharacter of its σ-orbit.

    σ-iterates *gamma* forward and backward, scanning every σᵏ(γ) whose
    magnetic part is a dominant cocharacter (C·v ≥ 0).  Returns the
    **lex-maximum** such v across the orbit window — the "highest weight"
    in the orbit.  Examples:

      • SU(2) (0,1):  σ-orbit visits magnetics 0 (e.g. (0,1)) and 1
        (e.g. (1,-3)).  Both dominant.  Lex-max = (1,) → level 1.
      • SU(2) Wilson (0,-1): σ-fixed, magnetic 0, level 0.
      • SU(3) T(2,1): σ-orbit visits (2,1) and (1,2).  Both dominant
        and incomparable in the dominance partial order; lex-max =
        (2,1) gives the canonical representative of the σ-orbit.

    Parameters
    ----------
    gamma : Vec
        Tropical charge of an L_γ.
    factor_info : list[dict]
        ``factor_info`` from ``pure_ade_lattice_data``.
    alg : BPSKAlgebra
        For accessing σ via ``alg.rho`` / ``alg.rho_inverse``.
    max_steps : int
        How many σ-iterates to scan in each direction.  For deep σ-orbits
        increase if the function returns ``None``.

    Returns
    -------
    A Vec representing the σ-orbit's dominant cocharacter, or ``None``
    if no dominant magnetic part is found within the window.
    """
    candidates: list[Vec] = []
    cur = gamma
    for _ in range(max_steps):
        v = _ade_magnetic(cur, factor_info)
        if _is_dominant_cocharacter(v, factor_info):
            candidates.append(v)
        cur = alg.rho(cur)
    cur = gamma
    for _ in range(max_steps):
        cur = alg.rho_inverse(cur)
        v = _ade_magnetic(cur, factor_info)
        if _is_dominant_cocharacter(v, factor_info):
            candidates.append(v)
    if not candidates:
        return None
    return max(candidates)  # lex-max: highest weight in the σ-orbit


def compute_filtration_layers(
    alg,
    factor_info: list[dict],
    T_atoms: Sequence[Vec],
    W_charges: Sequence[Vec],
) -> dict[Vec, Vec]:
    """Organise canonical-basis elements into magnetic-filtration layers.

    Compute T(v) · W(w) and W(w) · T(v) for each *T_atom* v and each Wilson
    charge w in *W_charges*; record every L_γ that appears, tagged with the
    smallest *T_atoms* element v that produced it.  The identity and all
    Wilson lines are placed at level (0, …, 0).

    *T_atoms* must be passed in dominance order (smallest first); the
    function tags each L_γ with the FIRST *T_atom* whose product produces
    it.  For the SU(N) cases this is the natural "level" assignment.

    Parameters
    ----------
    alg : BPSKAlgebra
        The algebra (built from ``pure_ade_kalgebra`` typically).
    factor_info : list[dict]
        ``factor_info`` from ``pure_ade_lattice_data`` (used for naming).
    T_atoms : sequence of Vec
        Dominant cocharacters labelling 't Hooft generators T(v),
        in dominance order.  Each Vec gives the magnetic part v; the
        full tropical charge is ``(v_1, 0, v_2, 0, …)``.
    W_charges : sequence of Vec
        Tropical charges of the Wilson lines to use as multipliers.

    Returns
    -------
    dict mapping L_γ → level (a Vec, namely the cocharacter of the
    *T_atom* at which γ first appears).  Wilson lines and the identity
    map to the zero cocharacter ``(0, …, 0)``.

    Verifies the **filtration property** ``F_v · F_u ⊆ F_{v+u}`` was
    confirmed for SU(2) up to v ≤ 4 (2778 product term occurrences,
    no violations).  σ preserves levels: σ(L_γ) and L_γ have the same
    level.
    """
    r_dom = sum(fi["r"] for fi in factor_info if fi["tp"] != "U")
    zero = tuple([0] * r_dom)
    seen: set[Vec] = {tuple([0] * (2 * r_dom))}
    for w in W_charges:
        seen.add(tuple(w))
    for v in T_atoms:
        # Build the tropical charge of T(v) from its magnetic part v
        # by interleaving with zero electric.  Skip U(1) blocks (no
        # quiver charges).  Assumes pure ADE for now.
        gamma_T = tuple(x for vi in v for x in (vi, 0))
        seen.add(gamma_T)
        for w in (zero,) + tuple(W_charges):
            gamma_W = w if w != zero else tuple([0] * (2 * r_dom))
            for prod in (alg.multiply(gamma_T, gamma_W),
                         alg.multiply(gamma_W, gamma_T)):
                for g in prod.terms:
                    seen.add(g)
    # Compute σ-orbit-canonical level for each seen γ.
    # filtration_level returns the lex-minimum dominant magnetic charge
    # in the σ-orbit, giving a single representative per σ-orbit of
    # cocharacters (e.g. SU(3) (2,1) and (1,2) collapse to (1,2)).
    levels: dict[Vec, Vec] = {}
    for g in seen:
        lv = filtration_level(g, factor_info, alg)
        if lv is None:
            lv = zero  # fallback: treat as level 0 if no dominant magnetic found
        levels[g] = lv
    return levels
