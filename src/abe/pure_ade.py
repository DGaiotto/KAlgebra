"""Pure-gauge ADE theories with closed-form spectrum generators.

``PureADE(factors, K=None)`` is a constructor for K-theoretic Coulomb-branch
theories of the form

    G  =  ( G̃_1 × ... × G̃_M × U(1)^s ) / K

where each G̃_a is a simply-laced simple Lie group (type A/D/E) and K is a
finite subgroup of the product of centers together with the U(1)^s factor.
Instead of searching for a spectrum generator via bidirectional BFS (which
walls out around rank 11 in the quiver), we use the **bipartite-Coxeter
mutation sequence** on the doubled Dynkin quiver.  This gives the ordered
ordered spectrum-generator factorisation

    S = E_q(X_{g_1}) · ... · E_q(X_{g_{r·h}})

directly, with no search.

The constructor returns a ``PureADE`` object with:

    .factors        -- normalized list of factor specs
    .K              -- normalized K matrix (tuple of tuples), or empty tuple
    .B              -- block-diagonal exchange matrix of the BPS quiver
    .nodes          -- standard-basis node charges (in Γ_sc coordinates)
    .spec           -- ordered spectrum-generator charges
    .Gamma_sc       -- K-blind charge lattice  ⊕ (Q^∨⊕Q)_a ⊕ Z^{2s}
    .Gamma_G        -- physical charge lattice (uses K)
    .quiver         -- BPSQuiver in Γ_sc coordinates
    .algebra        -- CoulombAlgebra built via ``spec=`` (no BFS)
    .embed_sc(γ)    -- inclusion  Γ_sc ↪ Γ_G
    .allowed(γ)     -- membership test on Γ_G (γ in extended-fractional coords)

This module does not depend on ``BPSQuiver.find_negating_sequence``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Iterable, Optional, Sequence, Union

from bps_quiver_tools import BPSQuiver, CoulombAlgebra


# =====================================================================
# Per-factor Dynkin data
# =====================================================================

# Bourbaki numbering, 0-indexed.  Dynkin edges.
_E_EDGES: dict[int, list[tuple[int, int]]] = {
    6: [(0, 2), (2, 3), (3, 4), (4, 5), (1, 3)],
    7: [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 3)],
    8: [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 3)],
}
_E_COXETER: dict[int, int] = {6: 12, 7: 18, 8: 30}


FactorSpec = Union[tuple, str]


def _factor_data(tp: str, n: int) -> tuple[int, int, list[tuple[int, int]]]:
    """Rank r, Coxeter number h, and 0-indexed Dynkin edges for (tp, n)."""
    if tp == "A":
        if n < 1:
            raise ValueError(f"A_n requires n >= 1, got n={n}")
        return n, n + 1, [(i, i + 1) for i in range(n - 1)]
    if tp == "D":
        if n < 4:
            raise ValueError(f"D_n requires n >= 4, got n={n}")
        return (
            n,
            2 * n - 2,
            [(i, i + 1) for i in range(n - 2)] + [(n - 3, n - 1)],
        )
    if tp == "E":
        if n not in _E_EDGES:
            raise ValueError(f"E_n requires n in {{6,7,8}}, got n={n}")
        return n, _E_COXETER[n], _E_EDGES[n]
    raise ValueError(f"unknown simple-factor type {tp!r}; expected A/D/E")


def _doubled_dynkin_B(r: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Exchange matrix of the doubled Dynkin quiver (2r x 2r).

    Layout: index 2i is γ_{i,+}, 2i+1 is γ_{i,-}.
    """
    sz = 2 * r
    B = [[0] * sz for _ in range(sz)]
    for i in range(r):
        B[2 * i][2 * i + 1] = 2
        B[2 * i + 1][2 * i] = -2
    for (a, b) in edges:
        B[2 * a + 1][2 * b] = 1
        B[2 * b][2 * a + 1] = -1
        B[2 * b + 1][2 * a] = 1
        B[2 * a][2 * b + 1] = -1
    return B


def _bipartite_coxeter_sequence(r: int, h: int) -> list[int]:
    """Strong-coupling MGS: h passes of alternating μ_+, μ_-.

    In the doubled-quiver index convention of §2, μ_+ mutates all even
    indices and μ_- mutates all odd indices.  Within each class no two
    nodes are adjacent so the individual mutations commute.
    """
    seq: list[int] = []
    for k in range(h):
        base = 0 if (k % 2 == 0) else 1
        seq.extend(2 * i + base for i in range(r))
    return seq


def pure_ADE_block(tp: str, n: int) -> dict:
    """Closed-form  (B, nodes, spec)  for a single pure simply-laced factor.

    No BFS — the negating sequence is the bipartite Coxeter MGS.  Returns a dict with keys
    ``{"tp","n","r","h","edges","B","nodes","spec"}``.

    Validates the three §5 conditions at build time:
      (i)  running the sequence negates the node charges (as a multiset),
      (ii) every spec charge lies in the positive cone (non-negative
           coordinates), and
      (iii) the spec charges are distinct and of length ``r·h``.
    """
    r, h, edges = _factor_data(tp, n)
    B = _doubled_dynkin_B(r, edges)
    sz = 2 * r
    nodes = [tuple(1 if k == j else 0 for k in range(sz)) for j in range(sz)]
    seq = _bipartite_coxeter_sequence(r, h)

    Q = BPSQuiver.from_pairing(nodes, B)
    spec = Q.build_spectrum_generator(seq)
    final = Q.mutation_sequence(seq)

    # (i) negates up to permutation
    expected_neg = sorted(tuple(-x for x in g) for g in nodes)
    got = sorted(tuple(c) for c in final.charges)
    if got != expected_neg:
        raise AssertionError(
            f"bipartite-Coxeter sequence for {tp}_{n} does not negate "
            f"node charges: expected {expected_neg}, got {got}"
        )
    # (ii) positive cone
    for idx, g in enumerate(spec):
        if any(x < 0 for x in g):
            raise AssertionError(
                f"{tp}_{n}: spec[{idx}] = {g} has a negative coordinate "
                f"in the standard basis"
            )
    # (iii) distinct, correct length
    if len(spec) != r * h:
        raise AssertionError(
            f"{tp}_{n}: expected spec of length r*h = {r*h}, got {len(spec)}"
        )
    if len(set(spec)) != len(spec):
        raise AssertionError(f"{tp}_{n}: spec contains duplicates")

    return {
        "tp": tp, "n": n, "r": r, "h": h, "edges": edges,
        "B": B, "nodes": [tuple(v) for v in nodes],
        "spec": [tuple(v) for v in spec],
    }


# =====================================================================
# Factor-spec parsing + block-diagonal assembly
# =====================================================================

def _normalize_factor(f: FactorSpec) -> tuple[str, int]:
    """Canonicalise a factor spec.

    Accepts: ``("A", n)``, ``("D", n)``, ``("E", n)``, ``"U1"`` / ``"U(1)"``
    / ``("U1",)``.  Returns ``(kind, n)`` with ``kind`` one of
    ``{"A","D","E","U"}`` and ``n=1`` for U(1).
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
            raise ValueError(f"unknown 1-tuple factor spec {f!r}")
        if len(f) != 2:
            raise ValueError(f"factor spec must be length 2, got {f!r}")
        tp = str(f[0]).upper()
        if tp in ("A", "D", "E"):
            n = int(f[1])
            # Validate via _factor_data (raises on bad n)
            _factor_data(tp, n)
            return (tp, n)
        if tp in ("U", "U1"):
            return ("U", 1)
        raise ValueError(f"unknown factor type {tp!r}")
    raise ValueError(f"factor spec must be tuple or string, got {type(f).__name__}")


def _assemble_product(factors: Sequence[tuple[str, int]]) -> dict:
    """Block-diagonal assembly.

    Returns a dict with:
        "B_sc"     : (2r+2s) x (2r+2s) block-diagonal pairing
        "nodes"    : unfrozen BPS-quiver node charges in Γ_sc (length 2r+2s each)
        "spec"     : spectrum-generator charges in Γ_sc
        "factor_slots" : per-factor
                        {"tp","n","r","h","edges","start","end","kind"}
                        with kind in {"simple","U1"}, indices into Γ_sc
                        (simple factors occupy [start,end) = 2r dims, U(1)
                        factors occupy 2 dims).
        "simple_dim"   : 2 * Σ r_a        (quiver-carrying slots)
        "u1_count"     : s                (number of U(1) factors)
        "total_dim"    : 2*Σr_a + 2*s     (= rank of Γ_sc)

    U(1) factors contribute a rank-2 symplectic block ``[[0,1],[-1,0]]`` to
    ``B_sc`` but no BPS-quiver nodes and no spec factors.
    """
    # Simple factors first (to group the quiver-carrying slots contiguously),
    # U(1) factors after.  Within each group, preserve the input order.
    simple_factors = [(i, f) for i, f in enumerate(factors) if f[0] != "U"]
    u1_factors = [(i, f) for i, f in enumerate(factors) if f[0] == "U"]

    # Per-factor block data
    simple_blocks = [(orig_idx, pure_ADE_block(tp, n))
                     for (orig_idx, (tp, n)) in simple_factors]
    simple_dim = sum(blk["r"] * 2 for _, blk in simple_blocks)
    u1_count = len(u1_factors)
    total_dim = simple_dim + 2 * u1_count

    # Build block-diagonal pairing
    B_sc = [[0] * total_dim for _ in range(total_dim)]
    offset = 0
    factor_slots: list[dict] = [None] * len(factors)  # type: ignore[list-item]
    nodes: list[tuple[int, ...]] = []
    spec: list[tuple[int, ...]] = []

    def pad(v: Sequence[int], start: int) -> tuple[int, ...]:
        out = [0] * total_dim
        for k, x in enumerate(v):
            out[start + k] = int(x)
        return tuple(out)

    for orig_idx, blk in simple_blocks:
        sz = 2 * blk["r"]
        for i in range(sz):
            for j in range(sz):
                B_sc[offset + i][offset + j] = blk["B"][i][j]
        for v in blk["nodes"]:
            nodes.append(pad(v, offset))
        for v in blk["spec"]:
            spec.append(pad(v, offset))
        factor_slots[orig_idx] = {
            "tp": blk["tp"], "n": blk["n"], "r": blk["r"], "h": blk["h"],
            "edges": blk["edges"], "start": offset, "end": offset + sz,
            "kind": "simple",
        }
        offset += sz

    for orig_idx, (_tp, _n) in u1_factors:
        # Rank-2 symplectic [[0,1],[-1,0]]
        B_sc[offset][offset + 1] = 1
        B_sc[offset + 1][offset] = -1
        factor_slots[orig_idx] = {
            "tp": "U", "n": 1, "r": 0, "h": 0, "edges": [],
            "start": offset, "end": offset + 2, "kind": "U1",
        }
        offset += 2

    assert offset == total_dim
    assert all(slot is not None for slot in factor_slots)

    return {
        "B_sc": B_sc,
        "nodes": nodes,
        "spec": spec,
        "factor_slots": factor_slots,
        "simple_dim": simple_dim,
        "u1_count": u1_count,
        "total_dim": total_dim,
    }


# =====================================================================
# Cartan data and K-slot schema
# =====================================================================

def _cartan_matrix(r: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Symmetric simply-laced Cartan: 2 on the diagonal, -1 across edges."""
    C = [[0] * r for _ in range(r)]
    for i in range(r):
        C[i][i] = 2
    for (a, b) in edges:
        C[a][b] = -1
        C[b][a] = -1
    return C


def _invert_rational_matrix(M: Sequence[Sequence[int]]) -> list[list[Fraction]]:
    """Exact rational inverse via Gauss-Jordan over Fraction."""
    n = len(M)
    A = [[Fraction(M[i][j]) for j in range(n)] + [Fraction(1 if j == i else 0)
                                                  for j in range(n)]
         for i in range(n)]
    for col in range(n):
        # pivot
        piv = None
        for r_ in range(col, n):
            if A[r_][col] != 0:
                piv = r_
                break
        if piv is None:
            raise ValueError("matrix is singular")
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        A[col] = [x / pv for x in A[col]]
        for r_ in range(n):
            if r_ == col:
                continue
            f = A[r_][col]
            if f != 0:
                A[r_] = [a - f * b for a, b in zip(A[r_], A[col])]
    return [row[n:] for row in A]


def _slot_schema_for_factor(tp: str, n: int) -> list[dict]:
    """Per-factor slot metadata.

    Each slot is a dict ``{"period": p, "coweight_index": k}``.  For a U(1)
    factor ``period=0`` and ``coweight_index=None`` (the lift is the unit
    cocharacter on the U(1)'s m-slot, not a simple-factor coweight).
    """
    if tp == "A":
        return [{"period": n + 1, "coweight_index": 0}]
    if tp == "D":
        if n % 2 == 1:
            return [{"period": 4, "coweight_index": n - 1}]  # spinor
        # n even
        return [
            {"period": 2, "coweight_index": 0},     # vector (Bourbaki ω_1)
            {"period": 2, "coweight_index": n - 1}, # spinor (Bourbaki ω_n)
        ]
    if tp == "E":
        if n == 6:
            return [{"period": 3, "coweight_index": 0}]     # Bourbaki ω_1
        if n == 7:
            return [{"period": 2, "coweight_index": n - 1}] # Bourbaki ω_7
        if n == 8:
            return []
        raise ValueError(f"E_{n} not supported")
    if tp == "U":
        return [{"period": 0, "coweight_index": None}]
    raise ValueError(f"unknown factor type {tp!r}")


def _slot_layout(factors: Sequence[tuple[str, int]]) -> dict:
    """Total slot schema for a factor list, flattened in input order.

    Returns::
        {"slots"   : [{"factor": i, "in_factor": j, "period": p,
                       "coweight_index": k, "kind": "simple" or "U1"}, ...],
         "C"       : total slot count,
         "per_factor": [[slot indices], ...]    # one list per factor
        }
    """
    slots: list[dict] = []
    per_factor: list[list[int]] = []
    for i, (tp, n) in enumerate(factors):
        idxs: list[int] = []
        for j, meta in enumerate(_slot_schema_for_factor(tp, n)):
            s = {
                "factor": i, "in_factor": j,
                "period": meta["period"],
                "coweight_index": meta["coweight_index"],
                "kind": "U1" if tp == "U" else "simple",
            }
            idxs.append(len(slots))
            slots.append(s)
        per_factor.append(idxs)
    return {"slots": slots, "C": len(slots), "per_factor": per_factor}



def _integer_column_hnf(A: list[list[int]]) -> list[list[int]]:
    """Column Hermite normal form of an integer matrix.

    Returns the HNF as a list of columns (each a length-``rows`` list of
    ints).  Uses unimodular column operations; zero columns are kept at
    the right (the caller strips them).  Non-canonical but gives a valid
    ℤ-basis of the column lattice.
    """
    if not A or not A[0]:
        return []
    rows = len(A)
    cols = len(A[0])
    # Work with mutable copy
    H = [[A[r_][c] for c in range(cols)] for r_ in range(rows)]
    # For each row, eliminate entries to the right using Euclidean column ops.
    piv = 0
    for r_ in range(rows):
        if piv >= cols:
            break
        # Find any column >= piv with nonzero entry in row r_; if none, skip
        nonzero = [c for c in range(piv, cols) if H[r_][c] != 0]
        if not nonzero:
            continue
        # Use Euclidean reduction: repeatedly GCD columns down so that
        # only column `piv` has nonzero entry in row r_ among cols >= piv.
        while True:
            nonzero = [c for c in range(piv, cols) if H[r_][c] != 0]
            if len(nonzero) <= 1:
                break
            # Among nonzero, sort so smallest absolute is last
            nonzero.sort(key=lambda c: abs(H[r_][c]))
            c_small = nonzero[0]
            # Swap into piv slot if needed
            if c_small != piv:
                for rr in range(rows):
                    H[rr][piv], H[rr][c_small] = H[rr][c_small], H[rr][piv]
            # Reduce all other nonzero columns mod H[r_][piv]
            for c in nonzero:
                if c == c_small:
                    continue
                # After the swap, c_small is now at piv position
                c_eff = piv if c == c_small else c
                if c_eff == piv:
                    continue
                q = H[r_][c_eff] // H[r_][piv]
                if q != 0:
                    for rr in range(rows):
                        H[rr][c_eff] -= q * H[rr][piv]
        # Now only column piv has a nonzero entry in row r_ (among cols >= piv)
        # If that entry is negative, negate the column
        if H[r_][piv] < 0:
            for rr in range(rows):
                H[rr][piv] = -H[rr][piv]
        if H[r_][piv] != 0:
            # Reduce columns to the LEFT of piv so their row-r_ entries
            # are in [0, H[r_][piv])
            for c in range(piv):
                if H[r_][piv] == 0:
                    break
                q = H[r_][c] // H[r_][piv]
                if q != 0:
                    for rr in range(rows):
                        H[rr][c] -= q * H[rr][piv]
            piv += 1
    # Return columns (dropping trailing zeros is caller's job)
    return [[H[r_][c] for r_ in range(rows)] for c in range(cols)]




# =====================================================================
# Explicit integer embedding  Q^v + Q + Z^{2s}  ->  Gamma = Z^{2(r+s)}
# =====================================================================
#
# Parallel to the (K, slot-schema) path above: the *physical* charge
# lattice  Gamma  is ALWAYS  Z^{2(r+s)}  with the standard symplectic
# pairing
#
#     Lambda  =  block-diag( [[0,1], [-1,0]],  2*(r+s) times )
#
# and the "global form" of the gauge group is encoded in an integer
# embedding
#
#     iota : Q^v + Q + Z^{2s}  ->  Gamma
#
# given by two integer matrices  ``coroot_embedding``  and
# ``root_embedding`` :
#
#     coroot_embedding[i]  =  iota(alpha_i^v, 0)   in Gamma
#     root_embedding[i]    =  iota(0, alpha_i)     in Gamma
#
# Node charges then live as honest integer Z-vectors in Gamma:
#
#     gamma_{i,+}  =   coroot_embedding[i]
#     gamma_{i,-}  =  -coroot_embedding[i]  +  root_embedding[i]
#
# (see the recipe discussion in the chat log for the minus sign on
# gamma_{i,-} -- it is what reconciles the unimodular-Gamma picture
# with the existing SU(2) preset  spec = [(1,0), (-1,2)] .)
#
# U(1) factors contribute a rank-2 standard-symplectic block to Gamma
# and two basis rows to each embedding matrix (one magnetic, one
# electric unit).  No BPS-quiver nodes.
#
# This "explicit integer" layer coexists with the K-matrix / Gamma_G
# path above; pick whichever matches your use case.  The unimodular
# representation is exposed on the theory object as
#   .B_uni, .nodes_uni, .spec_uni, .coroot_embedding, .root_embedding.
# ---------------------------------------------------------------------


def _standard_symplectic(n_ranks: int) -> list[list[int]]:
    """2n x 2n standard symplectic matrix:  block-diagonal copies of
    ``[[0,1], [-1,0]]``.  ``n_ranks``  is the number of rank-2 blocks
    (equal to  r + s  for a theory of simple-rank  r  and  U(1)-count
    s).  Returned as a plain list of lists of ints.
    """
    N = 2 * n_ranks
    J = [[0] * N for _ in range(N)]
    for k in range(n_ranks):
        J[2 * k][2 * k + 1] = 1
        J[2 * k + 1][2 * k] = -1
    return J


def _sc_embedding_for_simple(tp: str, n: int) -> tuple[
    list[list[int]], list[list[int]]
]:
    """Simply-connected coroot/root embedding for a simple factor.

    Returns  (coroot_emb, root_emb)  both of shape (r, 2r).  In this
    default embedding  alpha_i^v  = e_{2i}  (the i-th magnetic slot of
    a rank-r block with interleaved symplectic pairs) and  alpha_i  =
    sum_j C_{ij} e_{2j+1}  (electric slot).  The standard-symplectic
    pairing then reproduces the Cartan:  Lambda(alpha_i^v, alpha_j) =
    C_{ij}.
    """
    r, _, edges = _factor_data(tp, n)
    C = _cartan_matrix(r, edges)
    coroot = [[0] * (2 * r) for _ in range(r)]
    root = [[0] * (2 * r) for _ in range(r)]
    for i in range(r):
        coroot[i][2 * i] = 1  # alpha_i^v in the i-th magnetic slot
        for j in range(r):
            root[i][2 * j + 1] = C[i][j]  # alpha_i in electric slots
    return coroot, root


def _apply_embedding_to_quiver_vector(
    v: Sequence[int],
    coroot_emb: Sequence[Sequence[int]],
    root_emb: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Map a vector in the doubled-Dynkin standard basis (dim 2r) to
    its image in  Gamma  (also dim 2r for this simple factor).

    The doubled-Dynkin basis indexes  gamma_{i,+} = e_{2i}  and
    gamma_{i,-} = e_{2i+1}.  The image of  gamma_{i,+}  in  Gamma  is
    ``coroot_emb[i]``; the image of  gamma_{i,-}  is
    ``- coroot_emb[i] + root_emb[i]`` (the sign convention from the
    chat log).
    """
    r = len(coroot_emb)
    assert len(v) == 2 * r, f"expected length {2*r}, got {len(v)}"
    dim = len(coroot_emb[0])
    out = [0] * dim
    for i in range(r):
        a_plus = int(v[2 * i])
        a_minus = int(v[2 * i + 1])
        # Contribution from gamma_{i,+} :   a_plus * coroot_emb[i]
        # Contribution from gamma_{i,-} :   a_minus * (-coroot_emb[i] + root_emb[i])
        for k in range(dim):
            out[k] += a_plus * coroot_emb[i][k]
            out[k] += a_minus * (-coroot_emb[i][k] + root_emb[i][k])
    return tuple(out)


def _check_embedding_gram(
    coroot_emb: Sequence[Sequence[int]],
    root_emb: Sequence[Sequence[int]],
    Lambda: Sequence[Sequence[int]],
    C: Sequence[Sequence[int]],
) -> None:
    """Validation: the Gram matrix of  (coroot, root)  under Lambda must
    reproduce the canonical pairing on  Q^v + Q ,

        Lambda(alpha_i^v, alpha_j^v) = 0,
        Lambda(alpha_i^v, alpha_j)   = C_{ij},
        Lambda(alpha_i,   alpha_j)   = 0   (simply-laced).

    Raises ``AssertionError`` on mismatch.  Called from the SC default
    and from any user-supplied explicit embedding.
    """
    r = len(coroot_emb)

    def lam(u, w):
        s = 0
        for a, ua in enumerate(u):
            if ua == 0:
                continue
            for b, wb in enumerate(w):
                if wb == 0:
                    continue
                s += ua * Lambda[a][b] * wb
        return s

    for i in range(r):
        for j in range(r):
            mv = lam(coroot_emb[i], coroot_emb[j])
            assert mv == 0, (
                f"coroot-coroot pairing nonzero at (i={i},j={j}): {mv}"
            )
            ev = lam(root_emb[i], root_emb[j])
            assert ev == 0, (
                f"root-root pairing nonzero at (i={i},j={j}): {ev}"
            )
            ce = lam(coroot_emb[i], root_emb[j])
            assert ce == C[i][j], (
                f"coroot-root pairing mismatch at (i={i},j={j}): "
                f"got {ce}, expected C_{{ij}} = {C[i][j]}"
            )


def _smith_normal_form_diagonal(M: Sequence[Sequence[int]]) -> list[int]:
    """Diagonal entries (elementary divisors) of the Smith normal form
    of an integer matrix  M .  Returns a sorted list of positive ints
    of length min(rows, cols) (with trailing zeros omitted).

    Algorithm: alternate row + column Euclidean reduction.  Slow for
    large matrices but adequate for the small embedding matrices used
    by  embedding_invariant  (size at most ~16 for E_8).
    """
    if not M or not M[0]:
        return []
    A = [[int(x) for x in row] for row in M]
    rows = len(A)
    cols = len(A[0])
    diag: list[int] = []
    pos = 0
    while pos < rows and pos < cols:
        # Find smallest nonzero |entry| in submatrix [pos:, pos:] and
        # bring it to (pos, pos) via row/column swaps.
        best = None
        for i in range(pos, rows):
            for j in range(pos, cols):
                if A[i][j] != 0:
                    if best is None or abs(A[i][j]) < abs(A[best[0]][best[1]]):
                        best = (i, j)
        if best is None:
            break  # rest of matrix is zero
        bi, bj = best
        # Swap into position (pos, pos)
        if bi != pos:
            A[pos], A[bi] = A[bi], A[pos]
        if bj != pos:
            for r_ in range(rows):
                A[r_][pos], A[r_][bj] = A[r_][bj], A[r_][pos]
        # Reduce rest of row pos and column pos against A[pos][pos].
        # Repeat until row and col are clean.
        while True:
            piv = A[pos][pos]
            # Reduce column pos
            changed = False
            for i in range(pos + 1, rows):
                if A[i][pos] == 0:
                    continue
                q = A[i][pos] // piv
                if q != 0:
                    for c in range(pos, cols):
                        A[i][c] -= q * A[pos][c]
                if A[i][pos] != 0:
                    # Bring smaller into pivot
                    if abs(A[i][pos]) < abs(A[pos][pos]):
                        A[pos], A[i] = A[i], A[pos]
                    changed = True
            # Reduce row pos
            piv = A[pos][pos]
            for j in range(pos + 1, cols):
                if A[pos][j] == 0:
                    continue
                q = A[pos][j] // piv
                if q != 0:
                    for r_ in range(rows):
                        A[r_][j] -= q * A[r_][pos]
                if A[pos][j] != 0:
                    if abs(A[pos][j]) < abs(A[pos][pos]):
                        for r_ in range(rows):
                            A[r_][pos], A[r_][j] = A[r_][j], A[r_][pos]
                    changed = True
            if not changed:
                break
        if A[pos][pos] < 0:
            for r_ in range(rows):
                A[r_][pos] = -A[r_][pos]
        diag.append(A[pos][pos])
        pos += 1
    # Enforce divisibility chain  d_1 | d_2 | ...   by gcd / lcm passes
    for i in range(len(diag)):
        for j in range(i + 1, len(diag)):
            g = gcd(diag[i], diag[j])
            if g == 0:
                continue
            lcm_ij = diag[i] * diag[j] // g
            diag[i], diag[j] = g, lcm_ij
    return diag


def _symplectic_normal_basis(Lambda: Sequence[Sequence[int]]) -> list[list[int]]:
    """Given a 2n x 2n antisymmetric integer matrix ``Lambda`` with
    |det| = 1, return an integer unimodular matrix ``U`` such that
    ``U^T @ Lambda @ U`` equals the block-diagonal standard symplectic
    matrix  J_std  with blocks  [[0, 1], [-1, 0]] .

    This is the symplectic-basis theorem for unimodular antisymmetric
    integer forms, implemented as a Gauss-Jordan-style reduction that
    applies matching column + row operations to preserve antisymmetry.

    Raises ``ValueError``  if  ``Lambda``  is not antisymmetric, has
    odd dimension, or is not unimodular.
    """
    n2 = len(Lambda)
    if n2 == 0:
        return []
    if n2 % 2 != 0:
        raise ValueError(f"Lambda dimension {n2} is odd; expected 2n")
    for i in range(n2):
        for j in range(n2):
            if Lambda[i][j] + Lambda[j][i] != 0:
                raise ValueError(
                    f"Lambda not antisymmetric at ({i},{j})"
                )

    # Work with a mutable copy; accumulate U as we go.
    L = [[int(Lambda[i][j]) for j in range(n2)] for i in range(n2)]
    U = [[1 if i == j else 0 for j in range(n2)] for i in range(n2)]

    def col_op(add_to: int, from_: int, coeff: int) -> None:
        """col[add_to] += coeff * col[from_]  (and matching row op)."""
        if coeff == 0:
            return
        for r_ in range(n2):
            L[r_][add_to] += coeff * L[r_][from_]
            U[r_][add_to] += coeff * U[r_][from_]
        for c_ in range(n2):
            L[add_to][c_] += coeff * L[from_][c_]

    def col_swap(i: int, j: int) -> None:
        if i == j:
            return
        for r_ in range(n2):
            L[r_][i], L[r_][j] = L[r_][j], L[r_][i]
            U[r_][i], U[r_][j] = U[r_][j], U[r_][i]
        for c_ in range(n2):
            L[i][c_], L[j][c_] = L[j][c_], L[i][c_]

    def col_negate(i: int) -> None:
        for r_ in range(n2):
            L[r_][i] = -L[r_][i]
            U[r_][i] = -U[r_][i]
        for c_ in range(n2):
            L[i][c_] = -L[i][c_]

    for k in range(0, n2, 2):
        # The current sub-block starts at (k, k).  We want
        # L[k][k+1] = 1 and all other L[k][j], L[j][k] for j != k+1 zero,
        # as well as the symmetric conditions for row/col k+1.
        # Step 1: find a nonzero entry in row k, columns >= k+1 ; bring
        # to column k+1 .
        pivot_col = None
        for j in range(k + 1, n2):
            if L[k][j] != 0:
                pivot_col = j
                break
        if pivot_col is None:
            raise ValueError(
                "Lambda is degenerate (singular sub-block); not unimodular"
            )
        col_swap(k + 1, pivot_col)

        # Step 2: if L[k][k+1] < 0, negate column k+1 to make it positive.
        if L[k][k + 1] < 0:
            col_negate(k + 1)

        # Step 3: reduce L[k][j] for j > k+1 by Euclidean GCD with
        # column k+1 , then zero them out when gcd reaches L[k][k+1].
        while True:
            # Find any j > k+1 with L[k][j] != 0 .  If L[k][j] not
            # divisible by L[k][k+1] , reduce via  col_op(k+1, j, -q)
            # where q = L[k][j] // L[k][k+1] ; if there's still a
            # remainder, swap columns so that the smaller |entry| is
            # at k+1 .  Repeat until all L[k][j] for j > k+1 are 0 .
            dirty = False
            for j in range(k + 2, n2):
                if L[k][j] == 0:
                    continue
                dirty = True
                q = L[k][j] // L[k][k + 1]
                if q != 0:
                    col_op(j, k + 1, -q)
                if L[k][j] == 0:
                    continue
                # Residue is smaller than |L[k][k+1]| by Euclid.  Swap.
                col_swap(k + 1, j)
                if L[k][k + 1] < 0:
                    col_negate(k + 1)
                break  # restart inner sweep after the swap
            if not dirty:
                break
        if L[k][k + 1] != 1:
            raise ValueError(
                f"Lambda is not unimodular: sub-block (k={k}) yielded "
                f"pivot {L[k][k + 1]}, expected 1"
            )

        # Step 4: clear row k+1 entries L[k+1][j] for j > k+1 using
        # L[k+1][k] = -1 (antisymmetry):  col_op(j, k, L[k+1][j])  makes
        # L[k+1][j] = 0 (since L[k+1][k] = -1 ).  This also modifies
        # column k entries, but we don't care -- future iterations work
        # on (k+2, ...).
        for j in range(k + 2, n2):
            v = L[k + 1][j]
            if v != 0:
                col_op(j, k, v)

        # Sanity: L[k+1][j] = 0 for j > k+1 now.
        for j in range(k + 2, n2):
            assert L[k + 1][j] == 0, (
                f"row k+1 not cleared: L[{k+1}][{j}] = {L[k+1][j]}"
            )
        # And L[k][k] = L[k+1][k+1] = 0 by antisymmetry.
        assert L[k][k] == 0
        assert L[k + 1][k + 1] == 0

    # Verify: L should now equal J_std.  Just sanity-check.
    for i in range(n2):
        for j in range(n2):
            expect = 0
            if i % 2 == 0 and j == i + 1:
                expect = 1
            elif j % 2 == 0 and i == j + 1:
                expect = -1
            if L[i][j] != expect:
                raise AssertionError(
                    f"symplectic reduction failed: L[{i}][{j}] = "
                    f"{L[i][j]}, expected {expect}"
                )
    return U


def _d_even_single_gen_embedding(
    n: int,
    gen: tuple[int, int],
    eta_val: Fraction,
) -> tuple[list[list[int]], list[list[int]]]:
    """Build (coroot_embedding, root_embedding) for a D_n factor with
    n even, given a single magnetic-subgroup generator
    ``gen = (v, s)``  in the (vector, spinor) slot basis of period 2
    each, and an eta value (Fraction, default 0 for this pass).

    The D_even center is  Z/2 x Z/2  with non-diagonal linking form
    ``[[0, 1/2], [1/2, 0]]`` (for n even, the ``vec*vec`` and
    ``spin*spin`` self-pairings are integer and collapse to zero).
    So  Gamma^perp  of  <gen>  contains elements  (a, b)  satisfying
    ``v*b + s*a = 0 mod 2`` ; for a single nonzero gen the generator
    of  Gamma^perp  is  gen  itself -- Γ^⊥ = Γ as subgroups of Z.
    The electric-side perp generator is then the corresponding
    ``v * omega_1 + s * omega_n``  (in α basis via Cinv).
    """
    if n < 4 or n % 2 != 0:
        raise ValueError(f"_d_even_single_gen_embedding requires n even, >= 4; got n={n}")
    if eta_val != 0:
        raise NotImplementedError(
            f"D_{n}: eta != 0 not yet supported in the D_even single-gen builder"
        )
    r = n
    _, _, edges = _factor_data("D", n)
    C = _cartan_matrix(r, edges)
    Cinv = _invert_rational_matrix(C)

    v, s = int(gen[0]) % 2, int(gen[1]) % 2
    # Magnetic lift: v * omega_1^v + s * omega_n^v, in alpha^v basis.
    lift_m = [Fraction(0)] * r
    for i in range(r):
        lift_m[i] += Fraction(v) * Cinv[i][0]
        lift_m[i] += Fraction(s) * Cinv[i][n - 1]

    # Electric perp: same direction (Gamma^perp = Gamma as subgroups
    # of Z for D_even).  v * omega_1 + s * omega_n in alpha basis.
    e_perp = [Fraction(0)] * r
    for i in range(r):
        e_perp[i] += Fraction(v) * Cinv[i][0]
        e_perp[i] += Fraction(s) * Cinv[i][n - 1]

    # Joint generators in (alpha^v, alpha) 2r-dim ambient.
    gens: list[list[Fraction]] = []
    for i in range(r):
        row = [Fraction(0)] * (2 * r)
        row[i] = Fraction(1)
        gens.append(row)
    for i in range(r):
        row = [Fraction(0)] * (2 * r)
        row[r + i] = Fraction(1)
        gens.append(row)
    if any(x != 0 for x in lift_m):
        row = [Fraction(0)] * (2 * r)
        for i in range(r):
            row[i] = lift_m[i]
        gens.append(row)
    if any(x != 0 for x in e_perp):
        row = [Fraction(0)] * (2 * r)
        for i in range(r):
            row[r + i] = e_perp[i]
        gens.append(row)

    # Clear denominators + HNF.
    denom = 1
    for row in gens:
        for x in row:
            denom = denom * x.denominator // gcd(denom, x.denominator)
    ncol = len(gens)
    A = [[int(gens[c][r_] * denom) for c in range(ncol)] for r_ in range(2 * r)]
    H_cols = _integer_column_hnf(A)
    H_cols = [col for col in H_cols if any(x != 0 for x in col)]
    if len(H_cols) != 2 * r:
        raise AssertionError(
            f"joint HNF for D_{n} with gen={gen} yielded {len(H_cols)} "
            f"nonzero columns; expected {2 * r}"
        )

    def lam_bilinear(b_a, b_b):
        m_a = [Fraction(b_a[i], denom) for i in range(r)]
        e_a = [Fraction(b_a[r + i], denom) for i in range(r)]
        m_b = [Fraction(b_b[i], denom) for i in range(r)]
        e_b = [Fraction(b_b[r + i], denom) for i in range(r)]
        res = Fraction(0)
        for i in range(r):
            for j in range(r):
                res += m_a[i] * C[i][j] * e_b[j]
                res -= m_b[i] * C[i][j] * e_a[j]
        return res

    Lambda_new = [[0] * (2 * r) for _ in range(2 * r)]
    for i in range(2 * r):
        for j in range(2 * r):
            val = lam_bilinear(H_cols[i], H_cols[j])
            if val.denominator != 1:
                raise AssertionError(
                    f"Λ not integer on HNF basis at ({i},{j}): {val}; "
                    f"D_{n} with gen={gen} is not Lagrangian"
                )
            Lambda_new[i][j] = int(val)
    U = _symplectic_normal_basis(Lambda_new)

    final_basis = [[Fraction(0)] * (2 * r) for _ in range(2 * r)]
    for j in range(2 * r):
        for k in range(2 * r):
            ukj = U[k][j]
            if ukj == 0:
                continue
            for row_ in range(2 * r):
                final_basis[j][row_] += Fraction(H_cols[k][row_] * ukj, denom)

    B_mat = [[final_basis[j][row_] for j in range(2 * r)]
             for row_ in range(2 * r)]
    B_inv = _invert_fraction_matrix(B_mat)

    coroot_emb: list[list[int]] = []
    for i in range(r):
        target = [Fraction(0)] * (2 * r)
        target[i] = Fraction(1)
        coords = [sum(B_inv[j][col] * target[col] for col in range(2 * r))
                  for j in range(2 * r)]
        row = []
        for x in coords:
            if x.denominator != 1:
                raise AssertionError(
                    f"alpha_{i}^v has non-integer coords: {coords}"
                )
            row.append(int(x))
        coroot_emb.append(row)
    root_emb: list[list[int]] = []
    for i in range(r):
        target = [Fraction(0)] * (2 * r)
        target[r + i] = Fraction(1)
        coords = [sum(B_inv[j][col] * target[col] for col in range(2 * r))
                  for j in range(2 * r)]
        row = []
        for x in coords:
            if x.denominator != 1:
                raise AssertionError(
                    f"alpha_{i} has non-integer coords: {coords}"
                )
            row.append(int(x))
        root_emb.append(row)
    return coroot_emb, root_emb


def _single_factor_global_form_embedding(
    tp: str,
    n: int,
    gen: Sequence[int],
    eta_val: Fraction,
) -> tuple[list[list[int]], list[list[int]]]:
    """Build (coroot_embedding, root_embedding) for a single simple
    factor (tp, n) given a single magnetic-subgroup generator and an
    eta value (Fraction).

    General algorithm (works for any eta):

      1. Form the rank-2r sublattice  tilde{Gamma} ⊆ P^v ⊕ P  generated
         by  {(alpha_i^v, 0)}_i , {(0, alpha_i)}_i , and the glue vector
         (lift(g), eta_lift(g)) , where  lift(g)  is the cocharacter
         lift of g and  eta_lift(g)  is the electric companion such
         that ⟨lift(g), eta_lift(g)⟩ = eta_val mod Z .
      2. Scale by a common denominator D , compute column HNF to get a
         ℤ-basis of D*tilde{Gamma}  inside  ℤ^{2r} , then divide by D .
      3. Compute the Λ-Gram of this basis (an integer antisymmetric
         matrix with det = ±1 when the input is a valid Lagrangian).
      4. Run symplectic normal-basis reduction to get U integer
         unimodular with  U^T @ Lambda_new @ U = J_std .
      5. Re-express alpha_i^v and alpha_i in the U-transformed basis.
    """
    r, _, edges = _factor_data(tp, n)
    layout = _slot_layout([(tp, n)])
    slots = layout["slots"]
    C = _cartan_matrix(r, edges)
    Cinv = _invert_rational_matrix(C)

    # ---- magnetic lift of gen in alpha^v basis -----------------
    lift_m = [Fraction(0)] * r
    for j, val in enumerate(gen):
        if val == 0:
            continue
        slot = slots[j]
        if slot["kind"] != "simple":
            continue
        k = slot["coweight_index"]
        for i in range(r):
            lift_m[i] += Fraction(val) * Cinv[k][i]

    # ---- electric lift in alpha basis --------------------------
    # Two conditions must hold for the Lagrangian glue (lift_m, lift_e)
    # to sit inside a unimodular Gamma_tilde inside P^v + P :
    #
    #   (a) Pairing condition:  ⟨lift_m, lift_e⟩ ≡ eta_val  mod 1 .
    #   (b) Integrality:         ⟨alpha_i^v, lift_e⟩ ∈ ℤ  for all i .
    #
    # (a) is the Lagrangian gluing constraint.  (b) is needed so that
    # ``(alpha_i^v, 0)`` pairs integer with the glue vector.
    #
    # Condition (b) says  lift_e  is in the integer weight lattice  P .
    # So write  lift_e = sum_j c_j * omega_j  with  c_j ∈ ℤ .  In alpha
    # basis,  omega_j[i] = (C^{-1})[i][j] , so
    # ``lift_e_in_alpha[i] = sum_j c_j * (C^{-1})[i][j]`` .
    #
    # Condition (a) becomes  sum_j c_j * ⟨lift_m, omega_j⟩ ≡ eta_val mod 1
    # where  ⟨lift_m, omega_j⟩ = sum_k lift_m[k] * (C * C^{-1})[k][j] =
    #                           = sum_k lift_m_in_omega[k] * delta_{kj}
    #                           = lift_m_in_omega[j]  (since omega-alpha
    # duality gives  ⟨sum_k a_k * alpha_k^v, omega_j⟩ = a_j ).
    #
    # So we solve  sum_j c_j * lift_m_in_omega[j] ≡ eta_val mod 1 over ℤ.
    # A solution exists iff  eta_val  is in the (1/|Z|)ℤ-valued lattice
    # generated by the  lift_m_in_omega[j] 's (always true when eta is
    # restricted to the center ring).  We pick the minimum-norm integer
    # solution via simple search.
    lift_e = [Fraction(0)] * r
    if eta_val != 0:
        # Coweight duality  ⟨alpha_i^v, omega_j⟩ = delta_{ij}  gives
        # ⟨lift_m, omega_j⟩ = lift_m[j]  (lift_m is in alpha^v basis).
        # So the constraint becomes
        #     sum_j c_j * lift_m[j] ≡ eta_val  mod 1 ,  c_j ∈ ℤ .
        # Integer solutions always exist when eta_val ∈ (1/|Z|)ℤ .
        # Minimum-|c_j| one-hot search is sufficient for a single
        # cyclic center (all simple ADE cases).
        bound = 1
        for x in lift_m:
            if x != 0:
                bound = max(bound, x.denominator)
        c_solution = [0] * r
        found = False
        for j in range(r):
            d_j = lift_m[j]
            if d_j == 0:
                continue
            for c in range(-bound, bound + 1):
                if (Fraction(c) * d_j - eta_val).denominator == 1:
                    c_solution[j] = c
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError(
                f"Could not find integer solution to lift_e equation "
                f"for gen={gen}, eta={eta_val}, lift_m={lift_m}."
            )
        # lift_e = sum_j c_j * omega_j ;  in alpha basis:
        # lift_e_in_alpha[i] = sum_j c_j * (C^{-1})[i][j] .
        for i in range(r):
            s = Fraction(0)
            for j in range(r):
                s += c_solution[j] * Cinv[i][j]
            lift_e[i] = s

    # ---- electric perp generator -------------------------------
    # Lagrangian L = {(m, e) : m in <g> and e determined by eta mod
    # Gamma^perp}.  For a cyclic Gamma = <g> of order d_g in ZZ/p, the
    # "orthogonal complement" Gamma^perp on the electric side is also
    # cyclic with generator  (p / d_g) * omega_{k0}  where  k0  is the
    # coweight index of the active slot and  p  is its period.  This
    # factor (0, e_perp) must be added to Gamma_tilde  -- it is the
    # "other half" of the Lagrangian orbit beyond the magnetic glue.
    e_perp = [Fraction(0)] * r
    for j, val in enumerate(gen):
        if val == 0:
            continue
        slot = slots[j]
        if slot["kind"] != "simple":
            continue
        p_j = slot["period"]
        if p_j == 0:
            continue
        # Order d_g of val in ZZ/p_j
        d_g = p_j // gcd(val, p_j)
        # |Gamma^perp| = p_j / d_g .  A generator of order p_j/d_g
        # in the cyclic P/Q (= ZZ/p_j) is d_g * omega_{k0} .  For
        # |Gamma^perp| = 1 (i.e. d_g = p_j), d_g * omega_{k0} lies in
        # Q and contributes nothing new.
        scale = d_g
        k0 = slot["coweight_index"]
        for i in range(r):
            e_perp[i] += Fraction(scale) * Cinv[i][k0]

    # ---- joint generators in (alpha^v, alpha) basis ------------
    # Rows: (alpha_i^v, 0), (0, alpha_i), plus glue (lift_m, lift_e)
    # and electric perp (0, e_perp).
    gens: list[list[Fraction]] = []
    for i in range(r):
        row = [Fraction(0)] * (2 * r)
        row[i] = Fraction(1)
        gens.append(row)
    for i in range(r):
        row = [Fraction(0)] * (2 * r)
        row[r + i] = Fraction(1)
        gens.append(row)
    if any(x != 0 for x in lift_m) or any(x != 0 for x in lift_e):
        row = [Fraction(0)] * (2 * r)
        for i in range(r):
            row[i] = lift_m[i]
            row[r + i] = lift_e[i]
        gens.append(row)
    if any(x != 0 for x in e_perp):
        row = [Fraction(0)] * (2 * r)
        for i in range(r):
            row[r + i] = e_perp[i]
        gens.append(row)

    # ---- scale to clear denominators and HNF -------------------
    denom = 1
    for row in gens:
        for x in row:
            denom = denom * x.denominator // gcd(denom, x.denominator)
    # Integer generator matrix with each generator as a column.
    # Shape: (2r rows, #gens cols).
    ncol = len(gens)
    A = [[int(gens[c][r_] * denom) for c in range(ncol)] for r_ in range(2 * r)]
    H_cols = _integer_column_hnf(A)
    H_cols = [col for col in H_cols if any(x != 0 for x in col)]
    if len(H_cols) != 2 * r:
        raise AssertionError(
            f"joint HNF for ({tp}, {n}, gen={gen}, eta={eta_val}) "
            f"yielded {len(H_cols)} nonzero columns; expected {2 * r}"
        )

    # ---- Λ-Gram of the HNF basis (in (alpha^v, alpha) coords /denom)
    # Lambda((m, e), (m', e')) = <m, e'> - <m', e>
    # = sum_{i,j} m[i] * C[i][j] * e'[j] - m'[i] * C[i][j] * e[j] .
    # Basis vector j (as rational Fraction vector of length 2r) is
    # H_cols[j][...] / denom ; its magnetic and electric halves are
    # basis[j][:r] / denom  and  basis[j][r:] / denom .
    def lam_bilinear(b_a: Sequence[int], b_b: Sequence[int]) -> Fraction:
        """Λ-pairing of two columns from the scaled HNF basis.  The
        result is (exact) Fraction; for unimodular Γ̃ it comes out as
        an integer after dividing by denom^2.
        """
        m_a = [Fraction(b_a[i], denom) for i in range(r)]
        e_a = [Fraction(b_a[r + i], denom) for i in range(r)]
        m_b = [Fraction(b_b[i], denom) for i in range(r)]
        e_b = [Fraction(b_b[r + i], denom) for i in range(r)]
        s = Fraction(0)
        for i in range(r):
            for j in range(r):
                s += m_a[i] * C[i][j] * e_b[j]
                s -= m_b[i] * C[i][j] * e_a[j]
        return s

    Lambda_new: list[list[int]] = [[0] * (2 * r) for _ in range(2 * r)]
    for i in range(2 * r):
        for j in range(2 * r):
            v = lam_bilinear(H_cols[i], H_cols[j])
            if v.denominator != 1:
                raise AssertionError(
                    f"Λ not integer on HNF basis at ({i},{j}): {v}; "
                    f"input (Γ, η) is not Lagrangian. gen={gen}, "
                    f"eta={eta_val}"
                )
            Lambda_new[i][j] = int(v)

    # ---- symplectic normal-basis reduction ---------------------
    U = _symplectic_normal_basis(Lambda_new)
    # The final basis is "HNF basis" * U  (columns-of-columns composition):
    # final_basis[j] = sum_k HNF_cols[k] * U[k][j] .  Each final_basis[j]
    # lives in (alpha^v, alpha) rational coordinates (divide by denom).
    final_basis = [[Fraction(0)] * (2 * r) for _ in range(2 * r)]
    for j in range(2 * r):
        for k in range(2 * r):
            ukj = U[k][j]
            if ukj == 0:
                continue
            for row in range(2 * r):
                final_basis[j][row] += Fraction(H_cols[k][row] * ukj, denom)

    # ---- express alpha_i^v and alpha_i in final basis ----------
    # Final basis  B  has columns = final_basis[j] (in (alpha^v, alpha)
    # coords).  Invert to get coords of any rational vector in final
    # basis.  Build B as a 2r x 2r Fraction matrix with columns = final_basis[j].
    B = [[final_basis[j][row] for j in range(2 * r)]
         for row in range(2 * r)]
    B_inv = _invert_fraction_matrix(B)

    coroot_emb: list[list[int]] = []
    for i in range(r):
        target = [Fraction(0)] * (2 * r)
        target[i] = Fraction(1)        # alpha_i^v = e_i in (alpha^v, alpha) coords
        coords = [sum(B_inv[j][col] * target[col] for col in range(2 * r))
                  for j in range(2 * r)]
        row = []
        for x in coords:
            if x.denominator != 1:
                raise AssertionError(
                    f"alpha_{i}^v has non-integer coords in final basis: "
                    f"{coords}; gen={gen}, eta={eta_val}"
                )
            row.append(int(x))
        coroot_emb.append(row)

    root_emb: list[list[int]] = []
    for i in range(r):
        target = [Fraction(0)] * (2 * r)
        target[r + i] = Fraction(1)    # alpha_i = e_{r+i}
        coords = [sum(B_inv[j][col] * target[col] for col in range(2 * r))
                  for j in range(2 * r)]
        row = []
        for x in coords:
            if x.denominator != 1:
                raise AssertionError(
                    f"alpha_{i} has non-integer coords in final basis: "
                    f"{coords}; gen={gen}, eta={eta_val}"
                )
            row.append(int(x))
        root_emb.append(row)

    return coroot_emb, root_emb


def _invert_fraction_matrix(M: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    """Exact inverse of a Fraction matrix via Gauss-Jordan."""
    n = len(M)
    A = [list(M[i]) + [Fraction(1 if j == i else 0) for j in range(n)]
         for i in range(n)]
    for col in range(n):
        piv = None
        for r_ in range(col, n):
            if A[r_][col] != 0:
                piv = r_
                break
        if piv is None:
            raise ValueError("matrix is singular")
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        A[col] = [x / pv for x in A[col]]
        for r_ in range(n):
            if r_ == col:
                continue
            f = A[r_][col]
            if f != 0:
                A[r_] = [a - f * b for a, b in zip(A[r_], A[col])]
    return [row[n:] for row in A]


# =====================================================================
# PureADE theory object
# =====================================================================

class PureADE:
    """Pure-gauge simply-laced K-theoretic Coulomb-branch theory.

    Constructor::

        PureADE(factors, *, wrap_theory=False)

    ``factors`` is a list whose entries are any of:

      * ``("A", n)``, ``("D", n)``, ``("E", n)`` -- a simple ADE factor
        of type A/D/E and rank n.  Validation: ``n >= 1`` for A,
        ``n >= 4`` for D, ``n in {6,7,8}`` for E.
      * ``"U1"`` (or ``"U(1)"``) -- an abelian U(1) factor.  Contributes
        a rank-2 symplectic block to  Gamma  but no BPS-quiver node
        and no spectrum-generator factor.

    The default embedding is the **simply-connected** symplectic one:
    every charge lives in  Gamma = Z^{2(r+s)}  with standard symplectic
    pairing  Lambda((m, e), (m', e')) = m*e' - m'*e , the node charges
    are the explicit integer vectors

        gamma_{i,+}  =   coroot_embedding[i]
        gamma_{i,-}  =  -coroot_embedding[i]  +  root_embedding[i]

    (see the chat-log discussion), and the spec  S = E_q(X_{g_1}) ...
    E_q(X_{g_N})  factors are also integer vectors in  Gamma .  Other
    global forms (SO(2N), PSU(N), U(N), etc.) are constructed via the
    classmethods  ``simply_connected`` (same as the default),
    ``from_global_form`` , ``from_kac_label`` , and ``U_N`` .

    The returned object has:

      .factors          list of normalized (tp, n) tuples
      .B                2(r+s) x 2(r+s) standard symplectic pairing
      .nodes            BPS-quiver node charges in Gamma (integer)
      .spec             ordered spectrum-generator charges in Gamma
      .total_dim        rank of Gamma  (=  2(r+s) )
      .factor_slots     per-factor geometric slot info (list of dicts)
      .coroot_embedding iota(alpha_i^v, 0)  rows (integer matrix)
      .root_embedding   iota(0, alpha_i)    rows (integer matrix)
      .u1_norms         per-U(1) norm Lambda(coroot_U1, root_U1) (int)
      .quiver           BPSQuiver built from (B, nodes)
      .algebra          CoulombAlgebra built via spec=... (no BFS)

    The previous K-matrix path ( ``K=[[..]]`` argument,  ``.Gamma_sc``,
    ``.Gamma_G``, ``.K_lifts``, ``.allowed`` , ``.embed_sc`` ) lived in
    the rational ambient  P^v + P + Z^{2s}  and used  ``ChargeLattice``
    objects with  Fraction  entries.  That path has been retired.

    Optional ``wrap_theory=True`` also builds a theory wrapper
    around ``self.algebra`` and exposes it as ``self.theory`` for
    multi-chart work.
    """

    def __init__(
        self,
        factors: Sequence[FactorSpec],
        *,
        wrap_theory: bool = False,
        _install_sc: bool = True,
    ):
        """Default constructor installs the simply-connected symplectic
        embedding.  The private  ``_install_sc=False``  flag is used
        internally by  ``from_global_form`` / ``U_N`` / ``from_kac_label``
        to defer the embedding install until they have computed their
        specific (coroot, root) matrices.
        """
        if not factors:
            raise ValueError("PureADE requires at least one factor")

        self.factors: list[tuple[str, int]] = [
            _normalize_factor(f) for f in factors
        ]

        # --- K-blind doubled-Dynkin assembly (internal use only) -----
        asm = _assemble_product(self.factors)
        self._asm = asm
        self.total_dim: int = asm["total_dim"]
        self.factor_slots: list[dict] = asm["factor_slots"]

        # --- Embedding placeholders (installed below) ---------------
        self.coroot_embedding: Optional[tuple[tuple[int, ...], ...]] = None
        self.root_embedding: Optional[tuple[tuple[int, ...], ...]] = None
        self.B: Optional[list[list[int]]] = None
        self.nodes: Optional[list[tuple[int, ...]]] = None
        self.spec: Optional[list[tuple[int, ...]]] = None
        self.u1_norms: tuple[int, ...] = ()
        self.quiver = None
        self.algebra: Optional[CoulombAlgebra] = None
        self.theory = None

        if _install_sc:
            self._install_embedding(*self._build_sc_embedding())
            self._build_quiver_and_algebra(wrap_theory=wrap_theory)

    # --- explicit-integer-embedding classmethods ---------------------
    @classmethod
    def simply_connected(cls, factors: Sequence[FactorSpec]) -> "PureADE":
        """Construct a PureADE with the canonical simply-connected
        embedding installed.  This is the same as the default
        ``PureADE(factors)`` and is provided as an explicit constructor
        for clarity / parallelism with  ``U_N``  etc.

        For  SU(2)  this yields  ``.coroot_embedding = ((1, 0),)`` ,
        ``.root_embedding = ((0, 2),)`` , ``.nodes = [(1, 0), (-1, 2)]``
        under standard symplectic  ``.B = [[0, 1], [-1, 0]]`` .
        """
        return cls(factors)

    @classmethod
    def U_N(cls, N: int) -> "PureADE":
        """U(N)  =  ( SU(N) x U(1) ) / Z_N  built as an explicit
        integer embedding into  Gamma = Z^{2N}  (standard symplectic).

        Coordinates: interleaved magnetic/electric pairs
        ``(m_1, e_1, m_2, e_2, ..., m_N, e_N)``.  The embedding sends

            alpha_i^v   ->   eps_i - eps_{i+1}      (magnetic,  i=1..N-1)
            alpha_i     ->   eps_i^* - eps_{i+1}^*  (electric,  i=1..N-1)
            U(1) coroot ->   eps_1 + eps_2 + ... + eps_N        (mag, "trace")
            U(1) root   ->   eps_1^* + ... + eps_N^*            (elec, "trace dual")

        where  eps_k  is the k-th magnetic basis vector and  eps_k^*  the
        k-th electric basis vector.  This is the standard  U(N)
        cocharacter lattice  Z^N  inside  Gamma .

        The U(1) "norm" detected by the Gram check is
        ``Lambda(t, t^*) = N`` , reflecting the non-trivial  Lambda_Ab
        block  ``[[0, N], [-N, 0]]`` discussed in the design notes.
        """
        if N < 1:
            raise ValueError(f"U(N) requires N >= 1, got N={N}")
        if N == 1:
            # Just U(1)
            return cls.simply_connected(["U1"])

        factors = [("A", N - 1), "U1"]
        obj = cls(factors, _install_sc=False)
        total = obj.total_dim   # 2*(N-1) + 2  =  2N

        coroot = [[0] * total for _ in range(N)]
        root = [[0] * total for _ in range(N)]

        # SU(N) coroots and roots in interleaved (m,e) coordinates.
        # The simple factor occupies columns [0, 2(N-1)) ; the U(1)
        # occupies [2(N-1), 2N) -- but for the U(N) coordinate system
        # we want all slots to play the role of one of N "epsilon"
        # directions.  We use the SU factor's 2(N-1) coords as
        # (m_1, e_1, m_2, e_2, ..., m_{N-1}, e_{N-1}) and the U(1)
        # block's 2 coords as (m_N, e_N).
        for i in range(N - 1):
            coroot[i][2 * i] = 1
            coroot[i][2 * (i + 1)] = -1
            root[i][2 * i + 1] = 1
            root[i][2 * (i + 1) + 1] = -1
        # U(1) trace: sum over all magnetic / electric slots.
        for k in range(N):
            coroot[N - 1][2 * k] = 1
            root[N - 1][2 * k + 1] = 1

        obj._install_embedding(coroot, root)
        obj._build_quiver_and_algebra()
        return obj

    @classmethod
    def from_global_form(
        cls,
        factors: Sequence[FactorSpec],
        magnetic_subgroup: Optional[Sequence[Sequence[int]]] = None,
        eta: Optional[Sequence[Sequence]] = None,
    ) -> "PureADE":
        """Build a PureADE from the discrete  (Gamma, eta)  data of
        Aharony et al. ("reading between the lines"):

          * ``magnetic_subgroup``  -- list of integer-tuple generators of
            the subgroup  Gamma  of the non-abelian center  Z_nonAb .
            Each generator is a tuple of length C = number of slots
            (one per simple factor for A/E_6/E_7, one per spinor/vector
            tower for D, etc.).  Entries are read mod the slot periods.
            Empty / None means simply-connected (Gamma = trivial).
          * ``eta``  -- symmetric bilinear form on  Gamma  ->  R/Z ,
            given as a square matrix of  ``Fraction``  entries (or
            integers, treated as themselves).  ``None`` means  eta = 0 .

        The Lagrangian L is then
        ``L = {(m, eta(m, .) + Gamma^perp) : m in Gamma}``
        and the embedding  iota : Q^v + Q  ->  Gamma  is built so that
        iota(L) sits inside the integer  Gamma_uni .

        Restrictions in this initial pass:
          * Single simple factor (no products yet).
          * Up to one magnetic generator.
          * eta = 0  (no theta-twist) for general factors.  Single-slot
            ZZ/p factors with nontrivial eta land via a hardcoded path.

        For general (factors, Gamma, eta) input, follow up with
        ``from_kac_label`` for known names, or extend this routine.
        """
        # Normalise factors and route trivial cases through SC.
        factors_norm = [_normalize_factor(f) for f in factors]
        if magnetic_subgroup is None or len(magnetic_subgroup) == 0:
            return cls.simply_connected(factors_norm)
        if len(factors_norm) != 1 or factors_norm[0][0] not in ("A", "D", "E"):
            raise NotImplementedError(
                "from_global_form currently supports a single non-abelian "
                "factor; got factors=" + repr(factors_norm)
            )
        if len(magnetic_subgroup) > 1:
            raise NotImplementedError(
                "from_global_form currently supports up to one magnetic "
                "generator; got " + str(len(magnetic_subgroup))
            )

        tp, n = factors_norm[0]
        layout = _slot_layout([(tp, n)])
        slots = layout["slots"]
        # For now we require single-slot factors (A_n, E_6, E_7): the
        # multi-slot linking form on D_even centers needs a proper
        # non-diagonal treatment that this builder does not yet
        # implement.  D-type Kac labels go through a separate
        # hardcoded path.
        if layout["C"] != 1:
            raise NotImplementedError(
                f"from_global_form currently supports single-slot simple "
                f"factors (A_n, E_6, E_7); got {tp}_{n} with "
                f"{layout['C']} slots.  Use from_kac_label for D-type."
            )
        gen_row = tuple(int(x) for x in magnetic_subgroup[0])
        if len(gen_row) != layout["C"]:
            raise ValueError(
                f"magnetic_subgroup row has length {len(gen_row)}, "
                f"expected {layout['C']} (slots = {slots})"
            )
        # Reduce mod periods
        gen_reduced = tuple(
            (gen_row[j] % slots[j]["period"]) if slots[j]["period"] > 0
            else gen_row[j]
            for j in range(layout["C"])
        )
        if all(v == 0 for v in gen_reduced):
            return cls.simply_connected(factors_norm)

        # eta normalisation
        if eta is None:
            eta_val = Fraction(0)
        else:
            eta_mat = [[Fraction(eta[i][j]) for j in range(len(eta[i]))]
                       for i in range(len(eta))]
            if len(eta_mat) != 1 or len(eta_mat[0]) != 1:
                raise NotImplementedError(
                    "from_global_form: eta with more than one generator "
                    "not yet supported"
                )
            eta_val = eta_mat[0][0]
            # Require eta_val * period_of_generator to be integer.
            # For a single generator, period_of_generator = lcm of slot
            # periods on nonzero positions.  (The Lagrangian condition
            # per_g * eta(g, g) in Z is equivalent.)
            per = 1
            for j in range(layout["C"]):
                if gen_reduced[j] != 0 and slots[j]["period"] > 0:
                    per = per * slots[j]["period"] // gcd(per,
                                                          slots[j]["period"])
            if per > 0 and (eta_val * per).denominator != 1:
                raise ValueError(
                    f"eta value {eta_val} incompatible with generator "
                    f"{gen_reduced} (period {per}): (eta * per) must be "
                    f"integer"
                )

        # Delegate to a single-factor (Gamma, eta) -> embedding builder.
        coroot, root = _single_factor_global_form_embedding(
            tp, n, gen_reduced, eta_val
        )
        obj = cls(factors_norm, _install_sc=False)
        obj._install_embedding(coroot, root)
        obj._build_quiver_and_algebra()
        return obj

    @classmethod
    def from_kac_label(cls, name: str) -> "PureADE":
        """Construct a PureADE from a string label.

        Supported names (case-insensitive, spaces optional):

          * ``SU(N)``                  -- simply-connected A_{N-1}
          * ``SO(3)``  (=PSU(2))       -- A_1 / Z_2
          * ``SO(3)_theta``            -- A_1 / Z_2 with eta = 1/2
          * ``PSU(N)``                 -- A_{N-1} / Z_N, eta = 0
          * ``PSU(N)_theta_k``         -- A_{N-1} / Z_N, eta = k/N
          * ``U(N)``                   -- (SU(N) x U(1)) / Z_N, diagonal
          * ``Spin(2N)`` / ``SO(2N)``  -- D_N, simply-connected (Spin),
                                          or vector-quotient (SO).
          * ``E_6`` / ``E_7`` / ``E_8`` -- simply-connected.

        Anything else raises ``ValueError``.  This dispatcher is meant
        to cover the most common globally-named theories;
        ``from_global_form`` and the explicit-embedding constructor
        cover the rest.
        """
        s = name.strip().replace(" ", "")
        s_l = s.lower()

        # --- A-type ---------------------------------------------------
        if s.startswith("SU(") and s.endswith(")"):
            N = int(s[3:-1])
            return cls.simply_connected([("A", N - 1)])
        if s == "SO(3)":
            return cls.from_global_form([("A", 1)], magnetic_subgroup=[[1]])
        if s == "SO(3)_theta":
            return cls.from_global_form(
                [("A", 1)], magnetic_subgroup=[[1]],
                eta=[[Fraction(1, 2)]],
            )
        if s.startswith("PSU(") and s.endswith(")"):
            N = int(s[4:-1])
            return cls.from_global_form(
                [("A", N - 1)], magnetic_subgroup=[[1]],
            )
        if s_l.startswith("psu(") and "_theta_" in s_l:
            # e.g. "PSU(3)_theta_2" -> A_{N-1} / Z_N with eta = k/N .
            head, _, tail = s_l.partition("_theta_")
            N = int(head[4:-1])
            k = int(tail)
            return cls.from_global_form(
                [("A", N - 1)], magnetic_subgroup=[[1]],
                eta=[[Fraction(k, N)]],
            )
        if s.startswith("U(") and s.endswith(")"):
            N = int(s[2:-1])
            return cls.U_N(N)

        # --- D-type ---------------------------------------------------
        if s.startswith("Spin(") and s.endswith(")"):
            twoN = int(s[5:-1])
            if twoN < 8 or twoN % 2 != 0:
                raise ValueError(f"Spin({twoN}) requires 2N >= 8 and even")
            return cls.simply_connected([("D", twoN // 2)])
        # D-type Kac labels.  Single-slot D_odd (period 4) works via
        # the general A/E-style single-generator builder; multi-slot
        # D_even (two Z/2 slots with non-diagonal linking form) is
        # still TODO and the single-gen D_even helper has bugs for
        # n >= 6 / all triality variants.  Raise NotImplementedError
        # for D_even Kac labels until that's fixed.
        if s.startswith("SO(") and s.endswith(")") and s != "SO(3)":
            twoN = int(s[3:-1])
            if twoN < 8 or twoN % 2 != 0:
                raise ValueError(
                    f"SO({twoN}) Kac label only handled for 2N >= 8 even; "
                    f"for SO(3) use the literal 'SO(3)' label."
                )
            N_ = twoN // 2
            if N_ % 2 == 0:
                raise NotImplementedError(
                    f"SO({twoN}) (D_{N_} vector quotient) requires "
                    f"multi-slot D_even handling; not yet supported."
                )
            # D_odd: single slot of period 4, vector = order-2 = 2.
            return cls.from_global_form(
                [("D", N_)], magnetic_subgroup=[[2]],
            )
        if s.startswith("Ss(") and s.endswith(")"):
            raise NotImplementedError(
                f"Ss(..) requires multi-slot D_even handling; "
                f"not yet supported."
            )
        if s.startswith("Sc(") and s.endswith(")"):
            raise NotImplementedError(
                f"Sc(..) requires multi-slot D_even handling; "
                f"not yet supported."
            )
        if s.startswith("PSO(") and s.endswith(")"):
            twoN = int(s[4:-1])
            if twoN < 8 or twoN % 2 != 0:
                raise ValueError(f"PSO({twoN}) requires 2N >= 8 and even")
            N_ = twoN // 2
            if N_ % 2 == 0:
                raise NotImplementedError(
                    f"PSO({twoN}) with N even requires multi-generator + "
                    f"multi-slot D_even handling; not yet supported."
                )
            return cls.from_global_form(
                [("D", N_)], magnetic_subgroup=[[1]],
            )

        # --- E-type ---------------------------------------------------
        if s in ("E_6", "E6"):
            return cls.simply_connected([("E", 6)])
        if s in ("E_7", "E7"):
            return cls.simply_connected([("E", 7)])
        if s in ("E_8", "E8"):
            return cls.simply_connected([("E", 8)])

        raise ValueError(f"unknown Kac label: {name!r}")

    # --- embedding internals -----------------------------------------
    def _build_sc_embedding(self) -> tuple[
        list[list[int]], list[list[int]]
    ]:
        """Block-diagonal assembly of the simply-connected embedding
        matrices over self.factor_slots.  Simple factors use
        ``_sc_embedding_for_simple``; U(1) factors contribute one
        magnetic and one electric unit in their own 2-dim block.
        """
        total_dim = self.total_dim
        # Total non-abelian rank = number of coroot generators.
        rk_simple = sum(
            slot["r"] for slot in self.factor_slots if slot["kind"] == "simple"
        )
        u1_count = sum(
            1 for slot in self.factor_slots if slot["kind"] == "U1"
        )
        rk_full = rk_simple + u1_count  # one extra generator per U(1)

        coroot = [[0] * total_dim for _ in range(rk_full)]
        root = [[0] * total_dim for _ in range(rk_full)]

        row = 0
        for slot in self.factor_slots:
            start = slot["start"]
            if slot["kind"] == "simple":
                cr, rr = _sc_embedding_for_simple(slot["tp"], slot["n"])
                for i in range(slot["r"]):
                    for k in range(2 * slot["r"]):
                        coroot[row + i][start + k] = cr[i][k]
                        root[row + i][start + k] = rr[i][k]
                row += slot["r"]
            else:
                # U(1): one generator, magnetic at start, electric at start+1.
                coroot[row][start] = 1
                root[row][start + 1] = 1
                row += 1
        return coroot, root

    def _install_embedding(
        self,
        coroot_emb: Sequence[Sequence[int]],
        root_emb: Sequence[Sequence[int]],
    ) -> None:
        """Validate and install an integer embedding.  Populates
        ``.coroot_embedding``, ``.root_embedding``, ``.u1_norms``,
        and the symplectic  ``.B / .nodes / .spec``  attributes
        (obtained by mapping the K-blind doubled-Dynkin node and spec
        data through the embedding).
        """
        total_dim = self.total_dim
        rk = len(coroot_emb)
        if rk != len(root_emb):
            raise ValueError(
                f"coroot_embedding has {rk} rows but root_embedding has "
                f"{len(root_emb)}; they must match."
            )
        for name, M in (("coroot_embedding", coroot_emb),
                        ("root_embedding", root_emb)):
            for i, r in enumerate(M):
                if len(r) != total_dim:
                    raise ValueError(
                        f"{name} row {i} has length {len(r)}, "
                        f"expected total_dim={total_dim}"
                    )

        n_ranks = total_dim // 2
        assert n_ranks * 2 == total_dim
        Lambda = _standard_symplectic(n_ranks)

        # Per-factor Cartan + U(1) norm (detected from embedding).
        C_big = [[0] * rk for _ in range(rk)]
        row = 0
        u1_norms: list[int] = []
        for slot in self.factor_slots:
            if slot["kind"] == "simple":
                C_local = _cartan_matrix(slot["r"], slot["edges"])
                for i in range(slot["r"]):
                    for j in range(slot["r"]):
                        C_big[row + i][row + j] = C_local[i][j]
                row += slot["r"]
            else:
                d = 0
                for a in range(total_dim):
                    if coroot_emb[row][a] == 0:
                        continue
                    for b in range(total_dim):
                        if root_emb[row][b] == 0:
                            continue
                        d += (coroot_emb[row][a] * Lambda[a][b]
                              * root_emb[row][b])
                if d <= 0:
                    raise AssertionError(
                        f"U(1) factor at row {row}: Lambda(coroot, root) = "
                        f"{d}; expected positive integer (embedding is "
                        f"degenerate)"
                    )
                C_big[row][row] = d
                u1_norms.append(d)
                row += 1
        self.u1_norms = tuple(u1_norms)

        _check_embedding_gram(coroot_emb, root_emb, Lambda, C_big)

        self.coroot_embedding = tuple(tuple(r) for r in coroot_emb)
        self.root_embedding = tuple(tuple(r) for r in root_emb)
        self.B = Lambda

        # Map the K-blind doubled-Dynkin nodes and spec through the
        # embedding to get symplectic Gamma-coordinates.
        row_cursor = 0
        simple_blocks = []
        for slot in self.factor_slots:
            if slot["kind"] == "simple":
                r_ = slot["r"]
                cr = [list(self.coroot_embedding[row_cursor + i])
                      for i in range(r_)]
                rr = [list(self.root_embedding[row_cursor + i])
                      for i in range(r_)]
                simple_blocks.append({
                    "start": slot["start"], "r": r_, "cr": cr, "rr": rr
                })
                row_cursor += r_
            else:
                row_cursor += 1

        def map_padded(v: Sequence[int]) -> tuple[int, ...]:
            out = [0] * total_dim
            for blk in simple_blocks:
                start, r_ = blk["start"], blk["r"]
                local = [int(v[start + k]) for k in range(2 * r_)]
                if not any(local):
                    continue
                image = _apply_embedding_to_quiver_vector(
                    local, blk["cr"], blk["rr"]
                )
                for k in range(total_dim):
                    out[k] += image[k]
            return tuple(out)

        self.nodes = [map_padded(v) for v in self._asm["nodes"]]
        self.spec = [map_padded(v) for v in self._asm["spec"]]

    def _build_quiver_and_algebra(self, *, wrap_theory: bool = False) -> None:
        """After the embedding is installed, wire up  .quiver  and
        .algebra .  Uses the symplectic  .B  pairing, the integer node /
        spec charges, and a canonical cone-pointedness witness derived
        from the embedding itself (so  CoulombAlgebra  skips the slow
        box search).
        """
        if self.nodes:
            self.quiver = BPSQuiver.from_pairing(self.nodes, self.B)
            witness = self._canonical_cone_witness()
            self.algebra = CoulombAlgebra(
                pairing=self.B,
                node_charges=self.nodes,
                spec=self.spec,
                cone_witness=witness,
            )
        else:
            # Pure U(1)^s: no quiver, no algebra.
            self.quiver = None
            self.algebra = None
        if wrap_theory and self.algebra is not None:
            from coulomb_algebra_theory import CoulombAlgebraTheory
            self.theory = CoulombAlgebraTheory(self.algebra)

    def _canonical_cone_witness(self) -> tuple[int, ...]:
        """A linear functional  f  on  Gamma  that is >= 1 on every
        node charge.  Existence is guaranteed by the closed-form
        spectrum-generator recipe.  The construction:

          1. For each simple factor, use the Weyl-type witness
             ``f_mag = 1 , f_elec = 2 rho``  where  ``2 rho = C^-1 * (2,2,...,2)``
             in alpha-basis.  In the simply-connected embedding this
             gives  f . gamma_{i,+} = 1  and  f . gamma_{i,-} = 1 .
          2. For non-simply-connected embeddings the coroot/root rows
             are rotated integer combinations of the SC ones; compute
             the pullback of the SC witness.  This is just the SC
             witness evaluated on  coroot_embedding / root_embedding
             coordinates.
          3. For U(1) factors, no nodes exist, so f can be zero on
             those coords.

        For the SC case (A_n, any n), the resulting f is exactly the
        one derived in the design notes:  ``f[2j] = 1`` ,
        ``f[2j+1] = (j+1)(n-j)`` (Weyl-vector formula).
        """
        total_dim = self.total_dim
        f = [0] * total_dim
        # Walk simple factors: build local SC witness in the factor's
        # own (doubled-Dynkin) basis, then transform it through the
        # embedding to get the Gamma-basis witness.
        row_cursor = 0
        for slot in self.factor_slots:
            if slot["kind"] == "simple":
                r_ = slot["r"]
                edges = slot["edges"]
                C = _cartan_matrix(r_, edges)
                Cinv = _invert_rational_matrix(C)
                # 2*rho in alpha basis = Cinv . (2, 2, ..., 2)
                # which is a Fraction vector; actually integer for
                # simply-laced because (2,2,...,2) = sum of rows of C.
                two_rho_alpha = [Fraction(0)] * r_
                for i in range(r_):
                    s = Fraction(0)
                    for j in range(r_):
                        s += Cinv[i][j] * 2
                    two_rho_alpha[i] = s
                # In the doubled-Dynkin basis with convention
                #   f.gamma_{i,+} = f[2i] = 1
                #   f.gamma_{i,-} = f[2i+1 via electric map] = 2*rho_i
                # we need  f_uni.coroot_embedding[i] = 1  (magnetic row)
                # and      f_uni.root_embedding[i]   = 2*rho_i.
                # Since the embedding rows are integer vectors in Gamma
                # we can solve:  f_uni . coroot_emb[i] = 1  for each i,
                # and             f_uni . root_emb[i]   = 2*rho_i  for each i.
                # This is a 2r x 2r linear system on the (2r mag+elec)
                # slots of Gamma that the simple factor's embedding rows
                # occupy; for the SC default it is block-identity and
                # the solution is trivially  f[2i] = 1 , f[2i+1] = 2 rho_i.
                # For non-SC embeddings the embedding matrix is not
                # identity but is still integer unimodular on its block.
                # Solve via Fraction Gauss-Jordan, then round-verify.
                coroot_block = [list(self.coroot_embedding[row_cursor + i])
                                for i in range(r_)]
                root_block = [list(self.root_embedding[row_cursor + i])
                              for i in range(r_)]
                # Stack: each row of the 2r x 2r system corresponds to
                # "f . coroot_block[i] = 1" (first r rows) and
                # "f . root_block[i] = 2 rho_i" (last r rows).  But f
                # has 2*total_dim components (global).  So we look
                # at the columns of the block -- the nonzero ones --
                # and solve only on those.
                block_cols = set()
                for row in coroot_block + root_block:
                    for k in range(total_dim):
                        if row[k] != 0:
                            block_cols.add(k)
                block_cols = sorted(block_cols)
                # Reduced system: 2r equations in len(block_cols) unknowns.
                A_sys = []
                rhs = []
                for i in range(r_):
                    A_sys.append([Fraction(coroot_block[i][k]) for k in block_cols])
                    rhs.append(Fraction(1))
                for i in range(r_):
                    A_sys.append([Fraction(root_block[i][k]) for k in block_cols])
                    rhs.append(two_rho_alpha[i])
                # Solve (overdetermined-but-consistent).  Use Gauss-
                # Jordan: find a particular solution with free vars = 0.
                n_cols = len(block_cols)
                aug = [A_sys[i] + [rhs[i]] for i in range(len(A_sys))]
                row_idx = 0
                pivot_col = []
                for col in range(n_cols):
                    piv = None
                    for r2 in range(row_idx, len(aug)):
                        if aug[r2][col] != 0:
                            piv = r2
                            break
                    if piv is None:
                        continue
                    aug[row_idx], aug[piv] = aug[piv], aug[row_idx]
                    pv = aug[row_idx][col]
                    aug[row_idx] = [x / pv for x in aug[row_idx]]
                    for r2 in range(len(aug)):
                        if r2 == row_idx:
                            continue
                        if aug[r2][col] != 0:
                            coef = aug[r2][col]
                            aug[r2] = [a - coef * b
                                       for a, b in zip(aug[r2], aug[row_idx])]
                    pivot_col.append(col)
                    row_idx += 1
                # Extract particular solution (free vars = 0)
                sol = [Fraction(0)] * n_cols
                for idx, col in enumerate(pivot_col):
                    sol[col] = aug[idx][-1]
                # Scale by lcm of denominators to get an integer
                # witness.  Multiplying f by k > 0 preserves
                # f . g >= k >= 1 , so the scaled f is still a valid
                # pointedness certificate.
                scale = 1
                for val in sol:
                    if val != 0:
                        d = val.denominator
                        scale = scale * d // gcd(scale, d)
                for idx, k in enumerate(block_cols):
                    val = sol[idx] * scale
                    assert val.denominator == 1, (
                        f"witness solve produced non-integer after scaling: {val}"
                    )
                    f[k] += int(val)
                row_cursor += r_
            else:
                # U(1) slot: no nodes; f=0 on these coords is fine.
                row_cursor += 1
        return tuple(f)

    # --- lattice helpers ---------------------------------------------
    # --- embedding invariant + hash ----------------------------------
    def embedding_invariant(self) -> tuple:
        """Deterministic invariant of the integer embedding
        ``Q^v + Q + Z^{2s}  ->  Gamma_uni`` , designed so that two
        embeddings related by  ``GL(2(r+s), Z)``  on  Gamma_uni  give the
        same invariant.  This includes the Sp(2(r+s), Z) automorphisms
        of the standard symplectic form, but is a coarser equivalence.

        Returned tuple has three parts:

          1. ``factor_signature``  -- a canonical-form tuple of factors
             (sorted by type then rank) with each entry  ``(tp, n)`` .
          2. ``smith_diag`` -- elementary divisors (Smith normal form
             diagonal) of the (rk x 2(r+s)) integer matrix
             ``[coroot_embedding ; root_embedding]`` regarded as a map
             ``Z^{rk}  ->  Gamma_uni`` .  Captures the lattice index of
             ``image(Q^v + Q + Z^{2s})``  in  ``Gamma_uni`` .
          3. ``u1_norms`` -- the U(1) "Cartan" entries detected at
             install time, as a tuple of ints.

        **Caveat**: NOT invariant under Weyl group action on Q^v+Q
        (which permutes simple roots) nor under outer automorphisms.
        For canonical equivalence under ``Sp x Weyl x Out`` , a deeper
        normal-form algorithm is needed -- a follow-up.
        """
        if self.coroot_embedding is None:
            raise ValueError(
                "embedding_invariant requires an installed embedding; "
                "call .simply_connected(..) / .from_global_form(..) / "
                "._install_embedding(..) first."
            )
        factor_signature = tuple(sorted(self.factors))
        # Smith invariants for coroot, root, and the joint stacked matrix.
        # The coroot-only and root-only Smith forms detect WHICH side
        # (magnetic vs electric) any non-trivial lift goes to:  SO(3)
        # has  coroot Smith = (2,)  but root Smith = (1,) , while
        # SU(2) is the opposite, so the pair of invariants
        # distinguishes them.
        coroot_smith = _smith_normal_form_diagonal(
            [list(r) for r in self.coroot_embedding]
        )
        root_smith = _smith_normal_form_diagonal(
            [list(r) for r in self.root_embedding]
        )
        joint_smith = _smith_normal_form_diagonal(
            [list(r) for r in self.coroot_embedding]
            + [list(r) for r in self.root_embedding]
        )
        return (
            factor_signature,
            tuple(coroot_smith),
            tuple(root_smith),
            tuple(joint_smith),
            tuple(self.u1_norms) if hasattr(self, "u1_norms") else (),
        )

    def embedding_hash(self) -> str:
        """SHA-256 hex digest of ``embedding_invariant()`` .  Stable
        across runs; useful as a cache key for downstream computations
        keyed by global form.
        """
        import hashlib
        import json
        inv = self.embedding_invariant()
        # Serialise via JSON (all entries are tuples / ints / strs)
        blob = json.dumps(inv, sort_keys=True, default=list).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    # --- summary -----------------------------------------------------
    def summary(self) -> str:
        lines = [f"PureADE({self.factors})"]
        n_nodes = 0 if self.nodes is None else len(self.nodes)
        n_spec = 0 if self.spec is None else len(self.spec)
        lines.append(
            f"  total_dim={self.total_dim}   |nodes|={n_nodes}"
            f"   |spec|={n_spec}"
        )
        if self.u1_norms:
            lines.append(f"  u1_norms={self.u1_norms}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# =====================================================================
# Matter-coupled gauge theories via character -> Wilson-line substitution
# =====================================================================
#
def _fundamental_wilson_support_UN(N: int) -> list[tuple[int, ...]]:
    """Closed-form support of the pure-U(N) fundamental Wilson line
    ``F_{eps_N^*}``  in the ``PureADE.U_N(N)`` interleaved (m, e)
    coordinates.

    Pattern observed from ``alg.F((0,...,0,+1))`` and verified by
    the S-test: ``2N - 1``  monomials, all with coefficient 1, in
    BFS-from-lowest order:

        - ``(0, ..., 0, +1)``                  (lowest = eps_N^*)
        - for ``j = N-1, N-2, ..., 1`` :
            - ``(..., -1, +1, +1, 0, ...)``    at positions (2j-2, 2j-1, 2j)
            - ``(..., 0, +1, 0, ...)``         at position  (2j-1) = eps_j^*

    The sign convention follows the dominant chamber eps_1 > ... > eps_N
    of U(N); the "lowest weight" in that chamber is  eps_N^* , sitting
    at the top (positive) of the last electric coord.
    """
    if N < 1:
        raise ValueError(f"N >= 1 required, got N={N}")
    r = 2 * N
    supp: list[tuple[int, ...]] = [
        tuple(1 if k == r - 1 else 0 for k in range(r))
    ]
    for j in range(N - 1, 0, -1):
        # transition:  -1 at 2j-2,  +1 at 2j-1,  +1 at 2j
        v = [0] * r
        v[2 * j - 2] = -1
        v[2 * j - 1] = 1
        v[2 * j] = 1
        supp.append(tuple(v))
        # weight:  +1 at 2j-1
        v = [0] * r
        v[2 * j - 1] = 1
        supp.append(tuple(v))
    return supp


def _antifund_wilson_support_UN(N: int) -> list[tuple[int, ...]]:
    """Closed-form support of the pure-U(N) antifundamental Wilson line
    ``F_{-eps_1^*}``  in the ``PureADE.U_N(N)`` interleaved (m, e)
    coordinates.

    Pattern observed from ``alg.F((0, -1, 0, ..., 0))`` and verified by
    ``verify_spectrum_generator``: ``2N - 1``  monomials, all with
    coefficient 1, in BFS-from-lowest order:

        - ``(0, -1, 0, ..., 0)``                  (lowest = -eps_1^*)
        - for ``j = 1, 2, ..., N-1`` :
            - ``(..., -1, 0, +1, -1, ...)``  at positions (2j-2, 2j, 2j+1)
            - ``(..., -1, ...)``              at position  2j+1 = -eps_{j+1}^*

    Companion to :func:`_fundamental_wilson_support_UN` , which has the
    fund traversing the dominant chamber from lowest weight ``eps_N^*``
    up to highest ``eps_1^*``; here the anti-fund traverses from
    ``-eps_1^*`` down to ``-eps_N^*``.  The two supports share the
    boundary weights ``+/- eps_i^*`` but their "transition" charges are
    different — in particular, the right-placed fundamental block in
    ``UN_Nf(matter_side='right')`` uses
    ``[(-alpha) for alpha in _antifund_wilson_support_UN(N)]`` , not the
    fund support itself.  See ``tests/test_un_nf_right_fundamental.py``.
    """
    if N < 1:
        raise ValueError(f"N >= 1 required, got N={N}")
    r = 2 * N
    supp: list[tuple[int, ...]] = [
        tuple(-1 if k == 1 else 0 for k in range(r))
    ]
    for j in range(1, N):
        # transition:  -1 at 2j-2,  +1 at 2j,  -1 at 2j+1
        v = [0] * r
        v[2 * j - 2] = -1
        v[2 * j] = 1
        v[2 * j + 1] = -1
        supp.append(tuple(v))
        # weight:  -1 at 2j+1
        v = [0] * r
        v[2 * j + 1] = -1
        supp.append(tuple(v))
    return supp


# ---------------------------------------------------------------------------
# w_0 :  longest-Weyl-element Z_2 automorphism of the pure-U(N) ambient
# lattice (slot-reversal  i <-> N + 1 - i  preserving (m, e) type).
#
# Used as the algebraic substrate for chirality flips in the U(N)
# matter constructors: applying  w_0  to a given (chirality, position)
# spec matter factor yields the (opposite-chirality, same-position)
# construction with the  w_0 -conjugated pure-gauge presentation -- a
# genuinely different BPS quiver presenting the same theory.
#
# Confirmed empirically (`experiments/chirality_placement_audit.py`):
# this single operation lets us realise every combination of
# chirality x spec-position for fund / antisym^2 / bifund matter, with
# `BPSQuiver.verify_spectrum_generator` passing on each.
#
# Note: the codebase's  `sigma`  (`A.sigma()` / `spec_sigma`) is a
# spec-induced involution distinct from  w_0 ; on the pure-U(N)
# canonical spec one has  sigma(e) = -w_0(e)  on the electric part,
# but in general they differ.
# ---------------------------------------------------------------------------


def _w0_perm_UN(N: int) -> list[int]:
    """Position-permutation implementing the longest Weyl element  w_0
    of  A_{N-1}  on the pure-U(N) interleaved (m, e) ambient lattice.

    Slot index  i <-> N + 1 - i , preserving (m, e) type:

        m_i  at position  2(i - 1)     ↦  m_{N+1-i}  at position  2(N-i)
        e_i  at position  2(i - 1) + 1 ↦  e_{N+1-i}  at position  2(N-i)+1

    Properties (verified for N <= 4 ; expected for all N):

      * Involution:  w_0 ∘ w_0 = id .
      * Lattice automorphism:  preserves the pure-U(N) pairing  B .
      * Sends fund support to negated antifund support, position-by-position:

            w_0 applied to  _fundamental_wilson_support_UN(N)[k]
                = -1 * _antifund_wilson_support_UN(N)[k]   for all k .

    Returns
    -------
    perm : list of length  2N
        perm[k] = new position of the basis vector currently at
        position  k .  Apply to a vector  v  via
        ``new_v[perm[k]] = v[k]`` ; to a matrix  M  via similarity
        ``new_M[perm[i]][perm[j]] = M[i][j]`` .
    """
    if N < 1:
        raise ValueError(f"N >= 1 required, got N={N}")
    p = [0] * (2 * N)
    for i in range(1, N + 1):
        j = N + 1 - i
        p[2 * (i - 1)] = 2 * (j - 1)
        p[2 * (i - 1) + 1] = 2 * (j - 1) + 1
    return p


def _apply_perm_to_vec(v: Sequence[int], perm: Sequence[int]) -> tuple[int, ...]:
    """Apply a position-permutation `perm` to vector  v .
    Returns a tuple  out  with  out[perm[k]] = v[k] ."""
    out = [0] * len(v)
    for k, x in enumerate(v):
        out[perm[k]] = int(x)
    return tuple(out)


def _apply_perm_to_mat(M: Sequence[Sequence[int]], perm: Sequence[int]) -> list[list[int]]:
    """Apply a position-permutation to a square matrix as similarity:
    out[perm[i]][perm[j]] = M[i][j] ."""
    n = len(M)
    out = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[perm[i]][perm[j]] = M[i][j]
    return out


def _embed_perm_with_identity(
    perm_local: Sequence[int],
    full_rank: int,
    offset: int = 0,
) -> list[int]:
    """Lift a position-permutation acting on positions
    ``[offset, offset + len(perm_local))``  to a length-`full_rank`
    permutation that is the identity outside that range.

    Used to apply  w_0  to a single gauge slot of a multi-slot
    construction (e.g. one leg of a bifund) without disturbing the
    other slots / flavour mus.
    """
    perm = list(perm_local)
    out = list(range(full_rank))
    for k in range(len(perm)):
        out[offset + k] = offset + perm[k]
    return out


def apply_bps_aut_to_charge_UN(
    charge: Sequence[int],
    N: int,
    *,
    gauge_offset: int = 0,
    flavor_sign: int = 1,
) -> tuple[int, ...]:
    """Apply the  BPS-quiver Z_2 automorphism  to a single lattice charge.

    Concretely, on the pure-U(N) gauge slot of  charge  occupying
    positions  ``[gauge_offset, gauge_offset + 2N)`` :  negate AND
    slot-reverse  i <-> N + 1 - i  (preserving (m, e) type).  This is
    the lattice automorphism bps_aut that REALISES the Z_2 permutation
    automorphism of the canonical pure-gauge BPS quiver -- it permutes
    the canonical pure-gauge nodes (sends a node to another node, not
    to its negation).

    The flavour direction (other positions outside the gauge slot) is
    multiplied by ``flavor_sign``  (default +1, i.e. identity on
    flavour);  -1 implements a flavour-charge sign flip which is
    sometimes used to keep matter nodes in the "same" cone after
    bps_aut-conjugation.  Either sign produces a verifier-valid spec on
    canonical pure-gauge -- they correspond to two different lattice
    embeddings of the same physical matter factor.
    """
    perm = _w0_perm_UN(N)
    out = [0] * len(charge)
    for k in range(len(charge)):
        if gauge_offset <= k < gauge_offset + 2 * N:
            new_pos = gauge_offset + perm[k - gauge_offset]
            out[new_pos] = -int(charge[k])
        else:
            out[k] = flavor_sign * int(charge[k])
    return tuple(out)


def apply_bps_aut_UN(
    B: Sequence[Sequence[int]],
    nodes: Sequence[Sequence[int]],
    spec: Sequence[Sequence[int]],
    N: int,
    *,
    gauge_offset: int = 0,
) -> tuple[list[list[int]], list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Apply the BPS-quiver Z_2 lattice automorphism bps_aut (negation +
    slot-reversal on the pure-U(N) gauge slot, identity on other
    positions) to a whole construction  (B, nodes, spec) .

    bps_aut is the lattice automorphism that REALISES the Z_2 permutation
    automorphism of the canonical pure-gauge BPS quiver of U(N):
    it preserves the pure-U(N) pairing  B  AND permutes the canonical
    pure-gauge nodes among themselves (no sign-flips on the node set).
    Concretely:

      * On the gauge slot:  γ → -w_0(γ)  where w_0 is the longest
        Weyl element of A_{N-1} acting as slot-reversal i <-> N+1-i
        on the interleaved (m, e) coords.
      * On flavour mus / other slots:  identity.

    Returns  ``(B', nodes', spec')``  -- a DIFFERENT verifier-valid
    presentation of the same physical theory:

      * ``B'``      has the same block structure (bps_aut preserves B).
      * ``nodes'``  ARE the same set as ``nodes``  for the pure-gauge
                    subset (just permuted);  matter nodes are bps_aut-images.
      * ``spec'``   is a valid spec generator on the new quiver.

    For matter blocks, bps_aut implements the chirality flip
    fund <-> antifund (in the BPS-positive-cone convention).

    Notes
    -----

    * bps_aut is an involution :  applying it twice recovers the original.
    * It is the codebase's notion of "BPS-quiver chamber-flip"
      automorphism;  the codebase's `sigma` (`A.sigma()` / `spec_sigma`)
      is the SPEC-INDUCED involution that DIFFERS from bps_aut in general
      (on the canonical pure-gauge spec one has the relation
       sigma(e) = -w_0(e)  on the electric part).
    """
    n_total = len(B)
    return (
        _apply_perm_to_mat_with_signs(B, N, gauge_offset, n_total),
        [apply_bps_aut_to_charge_UN(v, N, gauge_offset=gauge_offset)
         for v in nodes],
        [apply_bps_aut_to_charge_UN(v, N, gauge_offset=gauge_offset)
         for v in spec],
    )


def _apply_perm_to_mat_with_signs(M, N, gauge_offset, n_total):
    """Helper for `apply_bps_aut_UN` : applies the negation+permutation
    pattern to a matrix as similarity.  Two negations on B cancel, so
    the matrix transformation is just the slot-reversal permutation
    (no extra sign).  This is the same as `_apply_perm_to_mat` with
    the embedded permutation."""
    perm_gauge = _w0_perm_UN(N)
    perm = _embed_perm_with_identity(perm_gauge, n_total, gauge_offset)
    return _apply_perm_to_mat(M, perm)


# Back-compat shim: keep the old `apply_w0_UN` name as a deprecated
# alias for the slot-reversal-only operation (no sign).  It is NOT a
# symmetry of (B, pure_nodes) -- it carries pure_nodes to their
# negations -- so callers should use `apply_bps_aut_UN` instead.
def apply_w0_UN(
    B: Sequence[Sequence[int]],
    nodes: Sequence[Sequence[int]],
    spec: Sequence[Sequence[int]],
    N: int,
    *,
    gauge_offset: int = 0,
) -> tuple[list[list[int]], list[tuple[int, ...]], list[tuple[int, ...]]]:
    """DEPRECATED:  applies just the slot-reversal permutation (w_0
    as a lattice permutation).  This is NOT the BPS-quiver symmetry
    of canonical pure-U(N) -- use `apply_bps_aut_UN` for that.

    Kept as a back-compat alias for the earlier version of this module
    where w_0 alone was (incorrectly) used as the BPS-quiver
    automorphism.  Will be removed in a future cleanup pass.
    """
    perm_gauge = _w0_perm_UN(N)
    perm = _embed_perm_with_identity(perm_gauge, len(B), gauge_offset)
    return (
        _apply_perm_to_mat(B, perm),
        [_apply_perm_to_vec(v, perm) for v in nodes],
        [_apply_perm_to_vec(v, perm) for v in spec],
    )


def _antifund_wilson_support_AN(N: int) -> list[tuple[int, ...]]:
    """Closed-form support of the pure-SU(N) antifundamental Wilson line
    ``F_{-omega_1}``  in PureADE doubled-Dynkin coordinates.

    Observed from ``alg.F((0, -1, 0, ..., 0))`` and cross-checked via
    ``verify_spectrum_generator``: ``2N - 1``  monomials, all unit,
    in BFS-from-lowest order.  Structure mirrors
    :func:`_fundamental_wilson_support_AN`: for each  j = 1, ..., N-1 ,
    two entries "odd" and "even" carrying a  +1  at  2j - 1  (the
    j-th electric slot) and  -1  markers at  2j - 2  and  2j + 1
    where they lie within the lattice.
    """
    if N < 2:
        raise ValueError(f"N >= 2 required, got N={N}")
    r = 2 * (N - 1)
    supp: list[tuple[int, ...]] = [
        tuple(-1 if k == 1 else 0 for k in range(r))
    ]
    for j in range(1, N):
        v = [0] * r
        v[2 * j - 2] = -1
        v[2 * j - 1] = 1
        if 2 * j + 1 < r:
            v[2 * j + 1] = -1
        supp.append(tuple(v))
        v = [0] * r
        v[2 * j - 1] = 1
        if 2 * j + 1 < r:
            v[2 * j + 1] = -1
        supp.append(tuple(v))
    return supp


def _fundamental_wilson_support_AN(N: int) -> list[tuple[int, ...]]:
    """Closed-form support of the pure-SU(N) fundamental Wilson line
    ``F_{-omega_{N-1}}``  in PureADE doubled-Dynkin coordinates.

    Pattern observed from ``alg.F((0,...,0,-1))`` and verified by
    the S-test on  SU(2..6) N_f=1..4 : ``2N - 1``  monomials, all
    with coefficient 1, in the following BFS-from-lowest order
    (which matches the order for N=3):

        - ``(0, ..., 0, -1)``           (lowest weight = -omega_{N-1})
        - for ``j = N-2, N-3, ..., 1`` (so N-2 "gap" blocks):
            - ``(..., -1, -1, +1, ...)``  at positions  (2j-1, 2j, 2j+1)
            - ``(..., -1,  0, +1, ...)``  at positions  (2j-1,     2j+1)
        - ``(-1, +1, 0, ..., 0)``      (coroot shift)
        - ``(0, +1, 0, ..., 0)``       (highest weight = omega_1)

    Hardcoded rather than computed: ``solve_F``  on pure SU(N)  takes
    ~11s for N=6 (and worse for N=7+), whereas the recipe is
    trivially O(N).
    """
    if N < 2:
        raise ValueError(f"N >= 2 required, got N={N}")
    r = 2 * (N - 1)
    supp: list[tuple[int, ...]] = []
    # lowest weight
    supp.append(tuple(-1 if k == r - 1 else 0 for k in range(r)))
    # N-2 transition blocks, from rightmost (j = N-2) down to j = 1
    for j in range(N - 2, 0, -1):
        # (..., -1, -1, +1, ...)  at positions (2j-1, 2j, 2j+1)
        v = [0] * r
        v[2 * j - 1] = -1
        v[2 * j] = -1
        v[2 * j + 1] = 1
        supp.append(tuple(v))
        # (..., -1,  0, +1, ...)  at positions (2j-1, 2j+1)
        v = [0] * r
        v[2 * j - 1] = -1
        v[2 * j + 1] = 1
        supp.append(tuple(v))
    # coroot shift
    v = [0] * r
    v[0] = -1
    v[1] = 1
    supp.append(tuple(v))
    # highest weight
    v = [0] * r
    v[1] = 1
    supp.append(tuple(v))
    return supp


# Recipe (partial RG flows):  if pure gauge theory T[G, 0]
# has spectrum generator  S_IR  with Wilson-line images  F_{w_R}, then
# T[G, T^*N]  has
#
#     S_UV  =  f_N(F_{w_R}, mu) * S_IR
#
# where  prod_{w in weights(R)} E_q(mu * zeta^w)  =  f_N(chi_R(zeta), mu).
#
# For one copy of fundamental of  SU(N) , the recipe simplifies
# dramatically (verified on SU(2), SU(3) and empirically for all
# tested N):  the matter prefactor factorises as
#
#     f_N(F_{w_N}, mu)  =  prod_{alpha in support(F_{w_N})} E_q(mu * X_alpha)
#
# with the factors ordered as they appear in  F_{w_N}  (the
# doubly-tropical BFS order from solve_F).  Each such alpha is a
# tropical charge in  Gamma_sc ; the coefficient of every monomial
# in  F_{w_N}  is 1 (observed for SU(2..5)).  The total number of
# matter factors is  2N - 1 .
#
# Implementation: we use  PureADE  for pure  SU(N)  in the doubled-
# Dynkin coordinates (where all charges are integral), compute
# F((0,...,0,-1))  -- the Wilson line at the lowest weight of the
# fundamental -- and assemble the UV spec by prepending the matter
# factors (one copy of fundamental per N_f).
#
# Validation:  F_a * S_UV = X_{gamma_a} + O(q)  automatically
# guarantees  I_{a,b} = delta_{a,b} + O(q)  on the node basis.  This
# has been checked explicitly on SU(3) N_f=1 (all 15 Gram entries
# up to q^0) and holds by construction for SU(2) N_f=1.


class SUN_Nf:
    """SU(N) gauge theory coupled to ``Nf`` fundamental hypermultiplets.

    Same API as :class:`PureADE`:  ``.B`` , ``.nodes`` , ``.spec`` ,
    ``.algebra`` , optional ``.theory``  (set via ``wrap_theory=True``).
    The pure-gauge part is exposed as ``.pure`` (a ``PureADE`` object)
    and the matter structure as ``.matter_spec`` / ``.matter_nodes``.

    The UV spectrum generator is assembled as
    ``S_UV = prod_j prod_{alpha in F_{w_N}} E_q(X_{mu_j + alpha}) * S_pure``
    via a partial RG flow, with
    ``F_{w_N} = F_{-omega_{N-1}}``  hardcoded via the closed-form
    support ``_fundamental_wilson_support_AN`` (so we never run
    ``solve_F`` on the pure theory).

    The cone-pointedness check is bypassed (``skip_cone_check=True``):
    the recipe produces a provably valid spectrum generator, checkable
    in  O(|spec|)  via
    ``A.quiver.verify_spectrum_generator(A.spec)`` .  Users who want the
    belt-and-braces check can call that explicitly.
    """

    def __init__(self, N: int, Nf: int = 1, *, wrap_theory: bool = False):
        if N < 2:
            raise ValueError(f"N >= 2 required, got N={N}")
        if Nf < 0:
            raise ValueError(f"Nf >= 0 required, got Nf={Nf}")
        self.N = N
        self.Nf = Nf
        self.pure = PureADE([("A", N - 1)])
        B_pure = [list(r) for r in self.pure.B]
        nodes_pure = [tuple(g) for g in self.pure.nodes]
        spec_pure = [tuple(g) for g in self.pure.spec]
        r_pure = len(B_pure)
        if Nf == 0:
            self.B = B_pure
            self.nodes = nodes_pure
            self.spec = spec_pure
            self.matter_spec: list[tuple[int, ...]] = []
            self.matter_nodes: list[tuple[int, ...]] = []
            self.algebra = self.pure.algebra
            self.cone_witness = self.pure._canonical_cone_witness()
            if wrap_theory:
                from coulomb_algebra_theory import CoulombAlgebraTheory
                self.theory = CoulombAlgebraTheory(self.algebra)
            return

        lowest = tuple(-1 if k == r_pure - 1 else 0
                       for k in range(r_pure))
        alphas = _fundamental_wilson_support_AN(N)
        r_UV = r_pure + Nf
        B_UV = [row + [0] * Nf for row in B_pure]
        B_UV += [[0] * r_UV for _ in range(Nf)]

        def _ext(v, extra):
            return tuple(list(v) + list(extra))

        def _mu(j):
            return tuple(1 if k == r_pure + j else 0 for k in range(r_UV))

        matter_spec: list[tuple[int, ...]] = []
        matter_nodes: list[tuple[int, ...]] = []
        for j in range(Nf):
            mu_j = _mu(j)
            for a in alphas:
                matter_spec.append(
                    tuple(x + y for x, y in zip(_ext(a, [0] * Nf), mu_j))
                )
            matter_nodes.append(
                tuple(x + y for x, y in zip(_ext(lowest, [0] * Nf), mu_j))
            )
        pure_spec_ext = [_ext(g, [0] * Nf) for g in spec_pure]
        nodes_UV = [_ext(g, [0] * Nf) for g in nodes_pure] + matter_nodes
        spec_UV = matter_spec + pure_spec_ext

        f_pure = list(self.pure._canonical_cone_witness())
        f_lowest = sum(fi * li for fi, li in zip(f_pure, lowest))
        c = max(1 - f_lowest, 0)
        witness = tuple(f_pure + [c] * Nf)

        self.B = B_UV
        self.nodes = nodes_UV
        self.spec = spec_UV
        self.matter_spec = matter_spec
        self.matter_nodes = matter_nodes
        self.cone_witness = witness
        self.algebra = CoulombAlgebra(
            pairing=B_UV, node_charges=nodes_UV, spec=spec_UV,
            cone_witness=witness,
            skip_spec_cone_check=True,
            skip_cone_check=True,
        )
        if wrap_theory:
            from coulomb_algebra_theory import CoulombAlgebraTheory
            self.theory = CoulombAlgebraTheory(self.algebra)

    def verify(self) -> bool:
        """Run  ``verify_spectrum_generator``  on  ``.spec`` . Returns
        True iff the recipe produced a valid negating sequence.  Useful
        as a safety net since we skip the cone-pointedness check at
        construction."""
        ok, _ = self.algebra.quiver.verify_spectrum_generator(self.spec)
        return ok

    def summary(self) -> str:
        return (
            f"SUN_Nf(N={self.N}, Nf={self.Nf})\n"
            f"  rank={len(self.B)}  |nodes|={len(self.nodes)}  "
            f"|spec|={len(self.spec)}"
        )

    def __repr__(self) -> str:
        return self.summary()


class SUN_Nf_mixed:
    """SU(N) gauge theory with ``n_f``  fundamental and  ``n_af``
    antifundamental hypermultiplets.  Different splittings
    ``(n_f, n_af)``  with  ``n_f + n_af = N_f``  give different BPS
    quivers (different "charts") for the same physical theory.

    The UV spectrum generator is

        S_UV = prod_f f_fund(F_{w_N}, mu_f) * prod_a f_af(F_{w_{bar N}}, mu_a) * S_pure

    where  f_fund  is the fund matter prefactor (``2N-1`` ordered
    E_q factors indexed by the fund Wilson-line support) and
    f_af  is the antifund analogue (support from
    :func:`_antifund_wilson_support_AN`).  Fundamental blocks come
    first, anti-fundamental blocks second, then pure.  Each copy
    gets its own flavour direction  mu  in  ker B .

    Conjectural: for every  (n_f, n_af) , this emits a valid
    spectrum generator -- a "mixed" chart of the same
    SU(N) / N_f = n_f + n_af  theory.  Verified via
    ``BPSQuiver.verify_spectrum_generator``  for a wide  (N, n_f, n_af)
    sweep -- see ``tests/test_sun_nf_fundamental.py``.
    """

    def __init__(self, N: int, n_f: int = 1, n_af: int = 0,
                 *, wrap_theory: bool = False):
        if N < 2:
            raise ValueError(f"N >= 2 required, got N={N}")
        if n_f < 0 or n_af < 0:
            raise ValueError(f"n_f, n_af >= 0 required")
        self.N = N
        self.n_f = n_f
        self.n_af = n_af
        self.Nf = n_f + n_af
        self.pure = PureADE([("A", N - 1)])
        B_pure = [list(r) for r in self.pure.B]
        nodes_pure = [tuple(g) for g in self.pure.nodes]
        spec_pure = [tuple(g) for g in self.pure.spec]
        r_pure = len(B_pure)
        Nf_total = n_f + n_af
        if Nf_total == 0:
            self.B = B_pure
            self.nodes = nodes_pure
            self.spec = spec_pure
            self.matter_spec: list[tuple[int, ...]] = []
            self.matter_nodes: list[tuple[int, ...]] = []
            self.algebra = self.pure.algebra
            self.cone_witness = self.pure._canonical_cone_witness()
            if wrap_theory:
                from coulomb_algebra_theory import CoulombAlgebraTheory
                self.theory = CoulombAlgebraTheory(self.algebra)
            return

        low_f = tuple(-1 if k == r_pure - 1 else 0 for k in range(r_pure))
        low_af = tuple(-1 if k == 1 else 0 for k in range(r_pure))
        alphas_f = _fundamental_wilson_support_AN(N)
        alphas_af = _antifund_wilson_support_AN(N)

        r_UV = r_pure + Nf_total
        B_UV = [row + [0] * Nf_total for row in B_pure]
        B_UV += [[0] * r_UV for _ in range(Nf_total)]

        def _ext(v, extra):
            return tuple(list(v) + list(extra))

        def _mu(j, sign: int = 1):
            return tuple(sign if k == r_pure + j else 0
                         for k in range(r_UV))

        matter_spec: list[tuple[int, ...]] = []
        matter_nodes: list[tuple[int, ...]] = []
        for f in range(n_f):
            mu_f = _mu(f)
            for a in alphas_f:
                matter_spec.append(
                    tuple(x + y for x, y in zip(_ext(a, [0] * Nf_total), mu_f))
                )
            matter_nodes.append(
                tuple(x + y for x, y in zip(_ext(low_f, [0] * Nf_total), mu_f))
            )
        for a_idx in range(n_af):
            # Convention: antifundamentals carry flavour charge -1  (so
            # that mutating through a fund matter block flips it onto
            # an antifund matter node with the matching antifund-lowest
            # + flavour=-1  position).
            mu_a = _mu(n_f + a_idx, sign=-1)
            for a in alphas_af:
                matter_spec.append(
                    tuple(x + y for x, y in zip(_ext(a, [0] * Nf_total), mu_a))
                )
            matter_nodes.append(
                tuple(x + y for x, y in zip(_ext(low_af, [0] * Nf_total), mu_a))
            )

        pure_spec_ext = [_ext(g, [0] * Nf_total) for g in spec_pure]
        nodes_UV = [_ext(g, [0] * Nf_total) for g in nodes_pure] + matter_nodes
        spec_UV = matter_spec + pure_spec_ext

        # Cone witness: f_pure extended by constants on each flavour
        # slot.  Need c_f >= 1 - f_pure(low_f)  for fund matter nodes,
        # c_af >= 1 - f_pure(low_af)  for antifund matter nodes.
        f_pure = list(self.pure._canonical_cone_witness())
        f_low_f = sum(fi * li for fi, li in zip(f_pure, low_f))
        f_low_af = sum(fi * li for fi, li in zip(f_pure, low_af))
        c_f = max(1 - f_low_f, 0)
        c_af = max(1 - f_low_af, 0)
        witness = tuple(f_pure + [c_f] * n_f + [c_af] * n_af)

        self.B = B_UV
        self.nodes = nodes_UV
        self.spec = spec_UV
        self.matter_spec = matter_spec
        self.matter_nodes = matter_nodes
        self.cone_witness = witness
        self.algebra = CoulombAlgebra(
            pairing=B_UV, node_charges=nodes_UV, spec=spec_UV,
            cone_witness=witness,
            skip_spec_cone_check=True,
            skip_cone_check=True,
        )
        if wrap_theory:
            from coulomb_algebra_theory import CoulombAlgebraTheory
            self.theory = CoulombAlgebraTheory(self.algebra)

    def verify(self) -> bool:
        ok, _ = self.algebra.quiver.verify_spectrum_generator(self.spec)
        return ok

    def summary(self) -> str:
        return (
            f"SUN_Nf_mixed(N={self.N}, n_f={self.n_f}, n_af={self.n_af})\n"
            f"  rank={len(self.B)}  |nodes|={len(self.nodes)}  "
            f"|spec|={len(self.spec)}"
        )

    def __repr__(self) -> str:
        return self.summary()


class SUN_Nf_total:
    """Full theory wrapper for  SU(N) / N_f  registering all  N_f + 1
    charts  ``(n_f, n_af)``  with  ``n_f + n_af = N_f``  and the
    block-mutation paths connecting them.

    Root chart is  ``(N_f, 0)``  (all fundamentals).  Each subsequent
    chart  ``(n_f - 1, n_af + 1)``  is reached by mutating through one
    fund matter block (``2N - 1`` node mutations) of the previous
    chart's quiver.  Intermediate mutations use ``lazy=True``  so they
    don't pay the CoulombAlgebra-construction cost; only landing charts
    hold algebras (via the parent's, through transport-only F's).

    Access named charts via ``self.chart(n_f, n_af)``.  The underlying
    theory wrapper is at ``self.theory`` .
    """

    def __init__(self, N: int, Nf: int):
        if N < 2:
            raise ValueError(f"N >= 2 required, got N={N}")
        if Nf < 0:
            raise ValueError(f"Nf >= 0 required, got Nf={Nf}")
        self.N = N
        self.Nf = Nf
        self._splits: dict[tuple[int, int], str] = {}

        from coulomb_algebra_theory import CoulombAlgebraTheory

        root = SUN_Nf_mixed(N, Nf, 0)
        self.root = root
        self.theory = CoulombAlgebraTheory(root.algebra)
        self._splits[(Nf, 0)] = "root"

        r_pure = 2 * (N - 1)
        fund_alphas = _fundamental_wilson_support_AN(N)

        def build_fund_block(slot: int) -> list[tuple[int, ...]]:
            out = []
            for a in fund_alphas:
                g = list(a) + [0] * Nf
                g[r_pure + slot] = 1
                out.append(tuple(g))
            return out

        def find_fund_matter_slot(chart) -> Optional[int]:
            for c in chart.node_charges:
                flav = c[r_pure:]
                nz = [(j, v) for j, v in enumerate(flav) if v != 0]
                if len(nz) == 1 and nz[0][1] > 0:
                    return nz[0][0]
            return None

        current_key = "root"
        for step in range(Nf):
            chart = self.theory.chart(current_key)
            slot = find_fund_matter_slot(chart)
            if slot is None:
                break
            block = build_fund_block(slot)
            # Mutate through  2N - 2  intermediates lazily, then name the
            # landing chart.
            for i, gk in enumerate(block):
                idx = next(
                    j for j, c in enumerate(
                        self.theory.chart(current_key).node_charges
                    )
                    if tuple(c) == tuple(gk)
                )
                is_last = (i == len(block) - 1)
                n_f_new = Nf - step - 1
                n_af_new = Nf - n_f_new
                name = (f"({n_f_new}f+{n_af_new}af)"
                        if is_last else None)
                current_key = self.theory.mutate(
                    current_key, idx, name=name, lazy=True
                )
            self._splits[(n_f_new, n_af_new)] = current_key

    def chart(self, n_f: int, n_af: int) -> str:
        """Return the theory-wrapper chart key for the
        ``(n_f, n_af)``  splitting."""
        if (n_f, n_af) not in self._splits:
            raise KeyError(f"no registered chart for ({n_f}, {n_af})")
        return self._splits[(n_f, n_af)]

    def splits(self) -> list[tuple[int, int]]:
        """Return the list of registered  (n_f, n_af)  splittings."""
        return sorted(self._splits.keys(), key=lambda p: -p[0])

    def summary(self) -> str:
        lines = [f"SUN_Nf_total(N={self.N}, Nf={self.Nf})"]
        lines.append(f"  charts: {len(self._splits)} landing + "
                     f"{len(self.theory.charts()) - len(self._splits)} "
                     f"intermediate")
        for split in self.splits():
            lines.append(f"    {split}: {self._splits[split]}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _bifund_pair_order_AN(n1: int, n2: int) -> list[tuple[int, int]]:
    """Closed-form pair ordering for an  SU(N_1) x SU(N_2)  bifundamental
    matter correction, with  n_i = 2 N_i - 1  the size of the fund
    Wilson-line support on factor  i .

    For each anti-diagonal  k = i + j  ascending:

      * C_even(k) = { (i, k-i) : i even, 0 <= i < n_1, 0 <= k - i < n_2 }
        emitted in  i  ASCENDING order  (starts at the i=0 boundary,
        moves inward);
      * C_odd (k) = { (i, k-i) : i odd,  ...  }
        emitted in  i  DESCENDING order  (starts at the j=0 boundary,
        moves inward);
      * emit  C_even(k)  then  C_odd(k) .

    Rationale: within a fixed anti-diagonal, exactly the pairs
    ``(i, j)``  and  ``(i+2, j-2)``  have non-zero quantum-torus
    bracket  (+/- 1 ), all other same-anti-diag pairs commute.  So
    the anti-diagonal decomposes into two parity-indexed chains
    (even/odd  i ); each chain has a totally-ordered non-commutation
    structure that pins the direction.  Chains commute pairwise so
    the inter-chain interleaving is free.  Tested deterministic on
    ``SU(N_1) x SU(N_2)``  for  ``N_i <= 12`` .
    """
    out: list[tuple[int, int]] = []
    for k in range(n1 + n2 - 1):
        pairs_k = [(i, k - i) for i in range(n1) if 0 <= k - i < n2]
        even_chain = sorted([p for p in pairs_k if p[0] % 2 == 0])
        odd_chain = sorted([p for p in pairs_k if p[0] % 2 == 1],
                           reverse=True)
        out.extend(even_chain + odd_chain)
    return out


class SUN_bifund:
    """``SU(N_1) x SU(N_2)``  gauge theory with one bifundamental
    hypermultiplet and optional per-factor fundamentals.

    This is a drop-in analogue of :class:`SUN_Nf` that takes a second
    gauge factor and adds a bifundamental.  Internally it composes two
    :class:`SUN_Nf`  factors (for the per-factor  ``Nf_i``  fundamental
    matter, using the hardcoded fund Wilson-line recipe), then prepends
    a bifund matter block with ``(2N_1 - 1)(2N_2 - 1)`` factors ordered
    by :func:`_bifund_pair_order_AN`.

    API mirrors :class:`SUN_Nf`:  ``.B`` , ``.nodes`` , ``.spec`` ,
    ``.algebra`` , ``.bifund_spec`` , ``.bifund_node`` , ``.pure1`` /
    ``.pure2``  (the per-factor ``SUN_Nf`` wrappers), ``.cone_witness`` ,
    optional ``.theory``  with ``wrap_theory=True`` .

    Asymptotic freedom is enforced:  ``N_2 + Nf1 <= 2 N_1``  and
    ``N_1 + Nf2 <= 2 N_2``  (with the bifund contributing  ``N_2``
    fundamentals to the SU(N_1) count and vice versa).  Pass ``strict=False``  to relax the check.
    """

    def __init__(
        self,
        N1: int,
        N2: int,
        *,
        Nf1: int = 0,
        Nf2: int = 0,
        wrap_theory: bool = False,
        strict: bool = True,
    ):
        if N1 < 2 or N2 < 2:
            raise ValueError(f"N_i >= 2 required, got N1={N1}, N2={N2}")
        if Nf1 < 0 or Nf2 < 0:
            raise ValueError(
                f"Nf_i >= 0 required, got Nf1={Nf1}, Nf2={Nf2}"
            )
        if strict:
            if N2 + Nf1 > 2 * N1:
                raise ValueError(
                    f"asymptotic freedom on SU({N1}) side: "
                    f"N2 + Nf1 = {N2 + Nf1} > 2 * N1 = {2 * N1}"
                )
            if N1 + Nf2 > 2 * N2:
                raise ValueError(
                    f"asymptotic freedom on SU({N2}) side: "
                    f"N1 + Nf2 = {N1 + Nf2} > 2 * N2 = {2 * N2}"
                )
        self.N1 = N1
        self.N2 = N2
        self.Nf1 = Nf1
        self.Nf2 = Nf2

        # Per-factor  SU(N_i) + Nf_i  fundamentals  (hardcoded, fast).
        self.pure1 = SUN_Nf(N1, Nf1)
        self.pure2 = SUN_Nf(N2, Nf2)
        r1_full = len(self.pure1.B)     # 2(N_1 - 1) + Nf_1
        r2_full = len(self.pure2.B)     # 2(N_2 - 1) + Nf_2
        r_pure_1 = 2 * (N1 - 1)
        r_pure_2 = 2 * (N2 - 1)
        total = r1_full + r2_full + 1   # +1 for the bifund flavour slot

        # Block-diagonal ambient pairing.
        B = [[0] * total for _ in range(total)]
        for i in range(r1_full):
            for j in range(r1_full):
                B[i][j] = int(self.pure1.B[i][j])
        for i in range(r2_full):
            for j in range(r2_full):
                B[r1_full + i][r1_full + j] = int(self.pure2.B[i][j])
        # Last row/column = bifund flavour  (in ker B).

        def pad_1(v):
            return tuple(list(v) + [0] * (total - r1_full))

        def pad_2(v):
            return tuple([0] * r1_full + list(v) + [0])

        # Bifund charges in the doubled-Dynkin coordinates of the
        # per-factor ``PureADE``  parents (NOT shifted by  Nf_i  --
        # they live on the gauge block only).
        alphas1 = _fundamental_wilson_support_AN(N1)
        alphas2 = _fundamental_wilson_support_AN(N2)
        n1 = 2 * N1 - 1
        n2 = 2 * N2 - 1
        pair_order = _bifund_pair_order_AN(n1, n2)

        def bifund_charge(i: int, j: int) -> tuple[int, ...]:
            out = [0] * total
            # alpha_1 in positions 0 .. r_pure_1 - 1
            for k, x in enumerate(alphas1[i]):
                out[k] = int(x)
            # alpha_2 in positions r1_full .. r1_full + r_pure_2 - 1
            for k, x in enumerate(alphas2[j]):
                out[r1_full + k] = int(x)
            out[-1] = 1  # bifund flavour
            return tuple(out)

        bifund_spec = [bifund_charge(i, j) for (i, j) in pair_order]
        bifund_node = bifund_charge(0, 0)  # lowest weight on both sides

        # Assemble nodes, spec.
        nodes_UV: list[tuple[int, ...]] = (
            [pad_1(g) for g in self.pure1.nodes]
            + [pad_2(g) for g in self.pure2.nodes]
            + [bifund_node]
        )
        spec_pure = (
            [pad_1(g) for g in self.pure1.spec]
            + [pad_2(g) for g in self.pure2.spec]
        )
        spec_UV = bifund_spec + spec_pure

        # Cone witness: combine per-factor witnesses, extend by
        # constants on the bifund flavour slot so that f . bifund_node
        # is  >= 1 .
        f1 = list(self.pure1.cone_witness)
        f2 = list(self.pure2.cone_witness)
        f_bifund = f1 + f2 + [0]
        # Evaluate on bifund_node.
        bn_val = sum(fi * gi for fi, gi in zip(f_bifund, bifund_node))
        c = max(1 - bn_val, 0)
        f_bifund[-1] = c
        witness = tuple(f_bifund)

        self.B = B
        self.nodes = nodes_UV
        self.spec = spec_UV
        self.bifund_spec = bifund_spec
        self.bifund_node = bifund_node
        self.cone_witness = witness
        self.algebra = CoulombAlgebra(
            pairing=B,
            node_charges=nodes_UV,
            spec=spec_UV,
            cone_witness=witness,
            skip_spec_cone_check=True,
            skip_cone_check=True,
        )
        if wrap_theory:
            from coulomb_algebra_theory import CoulombAlgebraTheory
            self.theory = CoulombAlgebraTheory(self.algebra)

    def verify(self) -> bool:
        """Run  ``verify_spectrum_generator``  on  ``.spec`` ; returns
        True iff the recipe produced a valid negating sequence on
        the combined quiver.  Cheap (O(|spec|)) safety net since the
        construction skips all cone checks."""
        ok, _ = self.algebra.quiver.verify_spectrum_generator(self.spec)
        return ok

    def summary(self) -> str:
        return (
            f"SUN_bifund(N1={self.N1}, Nf1={self.Nf1}, "
            f"N2={self.N2}, Nf2={self.Nf2})\n"
            f"  rank={len(self.B)}  |nodes|={len(self.nodes)}  "
            f"|spec|={len(self.spec)}  |bifund|={len(self.bifund_spec)}"
        )

    def __repr__(self) -> str:
        return self.summary()


class UN_Nf:
    """U(N) gauge theory coupled to ``Nf`` fundamental hypermultiplets.

    Same API / conventions as :class:`SUN_Nf`  but built on
    ``PureADE.U_N(N)`` .  The lowest weight of the fundamental of
    U(N) is  eps_N^*  (sign +1 in the PureADE interleaved-(m,e)
    coordinates).  Wilson-line support hardcoded via
    ``_fundamental_wilson_support_UN``.

    Per-matter-factor choices (independent of each other in principle,
    realised in this implementation via the constraints noted below):

      * ``position``    in  ``{"before", "after"}`` :  whether the matter
                                                       block sits before
                                                       or after the
                                                       pure-gauge spec.
      * ``w0_applied``  in  ``{False, True}`` :        whether the
                                                       longest-Weyl-element
                                                        w_0  is applied to
                                                       this matter factor
                                                       (flipping its
                                                       chirality:
                                                        fund <-> w_0(fund)
                                                        = "antifund"
                                                       in the BPS-positive-
                                                       cone convention).

    The 4 = 2 (position) x 2 (w0_applied) per-hyper choices give 4^Nf
    distinct BPS quivers + spec presentations of the same theory (up
    to flavour-charge sign flips).

    Parameters
    ----------
    N             : U(N) rank (>= 1).
    Nf            : number of fundamental hypers (>= 0).
    wrap_theory   : if True, also build a theory wrapper at ``.theory`` .
    matter_side   : LEGACY.  Per-hyper  "left" / "right"  enum coupling
                    chirality + position the OLD way:
                      "left"  = (position="before", w0_applied=False)
                              = matter alphas = fund (in BPS-positive cone)
                              ; block placed before pure-gauge spec.
                      "right" = (position="after",  w0_applied=False)
                              = matter alphas = -_antifund_wilson_support
                              ; block placed after pure-gauge spec.
                    Mutually exclusive with `positions` / `w0_applied` .
    positions     : NEW.  Per-hyper  "before" / "after"  position for
                    each matter block.  Pass a string for uniform
                    placement or a length-Nf sequence.  Defaults to all
                    "before".
    w0_applied    : NEW.  Per-hyper  True / False .  When True, the
                    matter alphas become  w_0(fund) (= antifund alphas
                    in the BPS-positive-cone convention) and the matter
                    node is the  w_0 -image of the canonical.  Pass a
                    bool for uniform behaviour or a length-Nf sequence.
                    Defaults to all False (= today's behaviour).

    Constraint (Phase 1):  the current implementation builds the
    w_0 -applied case by applying  w_0  globally to the construction
    (B, nodes, spec), so every hyper that has  w0_applied=True  must
    agree (all True or all False).  Mixed cases (some hypers True,
    others False) live on the same physical theory but require
    local-moves/commute machinery to convert into a verifier-passing
    spec — not yet implemented.  Such requests raise
    ``NotImplementedError`` with a clear pointer.

    Both "matched" choices (canonical 'left' / 'right' = today's
    behaviour) realise the same algebra: the two specs are related by
    a sequence of pentagon + commute local moves (verified explicitly
    for  N <= 3  by ``find_local_moves_chain`` ).  The two w_0 -applied
    presentations (when matter positions are flipped accordingly)
    realise distinct BPS quivers + specs of the same theory up to
    flavour-charge sign flip.
    """

    def __init__(
        self,
        N: int,
        Nf: int = 1,
        *,
        wrap_theory: bool = False,
        matter_side: Union[str, Sequence[str], None] = None,
        positions: Union[str, Sequence[str], None] = None,
        bps_aut_applied: Union[bool, Sequence[bool], None] = None,
    ):
        if N < 1:
            raise ValueError(f"N >= 1 required, got N={N}")
        if Nf < 0:
            raise ValueError(f"Nf >= 0 required, got Nf={Nf}")

        # --- Argument resolution: (matter_side) XOR (positions, bps_aut_applied) ---
        if matter_side is not None and (
            positions is not None or bps_aut_applied is not None
        ):
            raise ValueError(
                "Specify either `matter_side` (legacy) OR "
                "(`positions`, `bps_aut_applied`) (new API), not both."
            )
        if matter_side is None and positions is None and bps_aut_applied is None:
            # Pure-default: today's behaviour, matter_side='left'.
            matter_side = "left"

        if matter_side is not None:
            # Legacy path: matter_side string or per-hyper sequence.
            if isinstance(matter_side, str):
                if matter_side not in ("left", "right"):
                    raise ValueError(
                        f"matter_side must be 'left' or 'right' "
                        f"(got {matter_side!r})"
                    )
                sides = [matter_side] * Nf
            else:
                sides = list(matter_side)
                if len(sides) != Nf:
                    raise ValueError(
                        f"matter_side list length {len(sides)} != Nf={Nf}"
                    )
                for s in sides:
                    if s not in ("left", "right"):
                        raise ValueError(
                            f"each matter_side entry must be 'left' or 'right' "
                            f"(got {s!r})"
                        )
            positions_list = [
                "before" if s == "left" else "after" for s in sides
            ]
            bps_aut_list = [False] * Nf
        else:
            # New path: positions + bps_aut_applied (per-hyper independent).
            if positions is None:
                positions_list = ["before"] * Nf
            elif isinstance(positions, str):
                if positions not in ("before", "after"):
                    raise ValueError(
                        f"position must be 'before' or 'after' "
                        f"(got {positions!r})"
                    )
                positions_list = [positions] * Nf
            else:
                positions_list = list(positions)
                if len(positions_list) != Nf:
                    raise ValueError(
                        f"positions list length {len(positions_list)} "
                        f"!= Nf={Nf}"
                    )
                for p in positions_list:
                    if p not in ("before", "after"):
                        raise ValueError(
                            f"each position entry must be 'before' or "
                            f"'after' (got {p!r})"
                        )
            if bps_aut_applied is None:
                bps_aut_list = [False] * Nf
            elif isinstance(bps_aut_applied, bool):
                bps_aut_list = [bps_aut_applied] * Nf
            else:
                bps_aut_list = list(bps_aut_applied)
                if len(bps_aut_list) != Nf:
                    raise ValueError(
                        f"bps_aut_applied list length {len(bps_aut_list)} "
                        f"!= Nf={Nf}"
                    )
                for x in bps_aut_list:
                    if not isinstance(x, bool):
                        raise ValueError(
                            f"each bps_aut_applied entry must be bool "
                            f"(got {x!r})"
                        )
            sides = [
                "left" if p == "before" else "right" for p in positions_list
            ]

        self.N = N
        self.Nf = Nf
        # Back-compat attribute exposing matter_side when the legacy
        # path was used.  When any bps_aut is applied, matter_side is
        # set to None (no direct legacy equivalent).
        if matter_side is not None:
            self.matter_side = matter_side
        elif not any(bps_aut_list):
            # No bps_aut applied -> matter_side derivable from positions.
            self.matter_side = (
                sides[0] if Nf > 0 and all(s == sides[0] for s in sides)
                else list(sides)
            )
        else:
            self.matter_side = None
        self.positions = tuple(positions_list)
        self.bps_aut_applied = tuple(bps_aut_list)
        self._matter_sides = sides
        self.pure = PureADE.U_N(N)
        B_pure = [list(r) for r in self.pure.B]
        nodes_pure = [tuple(g) for g in self.pure.nodes]
        spec_pure = [tuple(g) for g in self.pure.spec]
        r_pure = len(B_pure)
        if Nf == 0:
            self.B = B_pure
            self.nodes = nodes_pure
            self.spec = spec_pure
            self.matter_spec = []
            self.matter_nodes = []
            self.algebra = self.pure.algebra
            self.cone_witness = self.pure._canonical_cone_witness()
            if wrap_theory:
                from coulomb_algebra_theory import CoulombAlgebraTheory
                self.theory = CoulombAlgebraTheory(self.algebra)
            return

        lowest = tuple(1 if k == r_pure - 1 else 0 for k in range(r_pure))
        alphas_left = _fundamental_wilson_support_UN(N)
        alphas_right = [
            tuple(-x for x in a) for a in _antifund_wilson_support_UN(N)
        ]
        r_UV = r_pure + Nf
        B_UV = [row + [0] * Nf for row in B_pure]
        B_UV += [[0] * r_UV for _ in range(Nf)]

        def _ext(v, extra):
            return tuple(list(v) + list(extra))

        def _mu(j):
            return tuple(1 if k == r_pure + j else 0 for k in range(r_UV))

        matter_left: list[tuple[int, ...]] = []
        matter_right: list[tuple[int, ...]] = []
        matter_nodes: list[tuple[int, ...]] = []
        per_hyper_blocks: list[list[tuple[int, ...]]] = []
        for j in range(Nf):
            mu_j = _mu(j)
            alphas_j = alphas_left if sides[j] == "left" else alphas_right
            block_charges = []
            for a in alphas_j:
                charge = tuple(
                    x + y for x, y in zip(_ext(a, [0] * Nf), mu_j)
                )
                block_charges.append(charge)
            matter_node_j = tuple(
                x + y for x, y in zip(_ext(lowest, [0] * Nf), mu_j)
            )
            # Per-hyper independent bps_aut application: σ on this hyper's
            # matter charges + matter node, leaves pure-gauge and other
            # hypers untouched.
            if bps_aut_list[j]:
                block_charges = [
                    apply_bps_aut_to_charge_UN(c, N) for c in block_charges
                ]
                matter_node_j = apply_bps_aut_to_charge_UN(matter_node_j, N)
            per_hyper_blocks.append(block_charges)
            (matter_left if sides[j] == "left" else matter_right).extend(
                block_charges
            )
            matter_nodes.append(matter_node_j)
        pure_spec_ext = [_ext(g, [0] * Nf) for g in spec_pure]
        nodes_UV = [_ext(g, [0] * Nf) for g in nodes_pure] + matter_nodes
        spec_UV = matter_left + pure_spec_ext + matter_right
        matter_spec = matter_left + matter_right

        f_pure = list(self.pure._canonical_cone_witness())
        # Cone witness per matter node: pick c_j large enough that
        # f_pure . gauge_part(matter_node_j) + c_j >= 1.
        c_values = []
        for mn in matter_nodes:
            gauge_part = mn[:r_pure]
            val = sum(fi * gi for fi, gi in zip(f_pure, gauge_part))
            c_values.append(max(1 - val, 0))
        witness = tuple(f_pure + c_values)

        self.B = B_UV
        self.nodes = nodes_UV
        self.spec = spec_UV
        self.matter_spec = matter_spec
        self.matter_nodes = matter_nodes
        self.cone_witness = witness
        self.algebra = CoulombAlgebra(
            pairing=B_UV, node_charges=nodes_UV, spec=spec_UV,
            cone_witness=witness,
            skip_spec_cone_check=True,
            skip_cone_check=True,
        )
        if wrap_theory:
            from coulomb_algebra_theory import CoulombAlgebraTheory
            self.theory = CoulombAlgebraTheory(self.algebra)

    def verify(self) -> bool:
        ok, _ = self.algebra.quiver.verify_spectrum_generator(self.spec)
        return ok

    def gamma_of_label(self, m, e):
        """Canonical-chamber BPS charge for label  [m, e]  in pure U(N).

        For ``Nf == 0`` (pure gauge) the chamber map is the standard
        sort-pairs-by-(m decreasing, e increasing) rule from
        ``tropical_chamber_map.gamma_canonical`` :  the BPS lattice is
        rank-2N with interleaved coords  (m_1, e_1, ..., m_N, e_N) ;
        the canonical chamber  gamma(m, e)  is obtained by sorting the
        N pairs  (m_i, e_i)  with primary key  m  decreasing and
        secondary  e  increasing.  For  Nf > 0  the gauge block is
        followed by  Nf  flavour slots; matter directions are not part
        of the L-label here, so they are returned as zeros.

        Args:
          m: tuple of length  N  (magnetic charges).
          e: tuple of length  N  (electric charges).

        Returns:
          tuple of length  rank(B)  -- canonical-chamber gamma.
        """
        from tropical_chamber_map import gamma_canonical
        if len(m) != self.N or len(e) != self.N:
            raise ValueError(
                f"m and e must be length N = {self.N}; got {len(m)}, {len(e)}"
            )
        gauge = gamma_canonical(m, e)
        return gauge + (0,) * self.Nf

    def label_of_gamma(self, gamma):
        """Decode BPS charge  gamma  to canonical-form label  [m, e] .

        For ``Nf == 0``: returns  ((m_canon, e_canon), ())  with the
        gauge-block canonical (m, e) per the simultaneous-Weyl-orbit
        sort rule.  For  ``Nf > 0`` : gauge block is canonicalized;
        flavour-block coordinates pass through unchanged.

        Args:
          gamma: tuple of length  rank(B) .

        Returns:
          ((m_canon, e_canon), flavour_part)  : the canonical-form
          label  [m, e]  (gauge) plus the flavour-direction coords.
        """
        from tropical_chamber_map import label_of_gamma as _label
        if len(gamma) != len(self.B):
            raise ValueError(
                f"gamma length must match BPS rank {len(self.B)}; got {len(gamma)}"
            )
        gauge = tuple(gamma[: 2 * self.N])
        flavour = tuple(gamma[2 * self.N :])
        m_canon, e_canon = _label(gauge)
        return ((m_canon, e_canon), flavour)

    def summary(self) -> str:
        return (
            f"UN_Nf(N={self.N}, Nf={self.Nf}, matter_side={self.matter_side!r})\n"
            f"  rank={len(self.B)}  |nodes|={len(self.nodes)}  "
            f"|spec|={len(self.spec)}"
        )

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# UN_antisym :  U(N) gauge with antisymmetric-tensor matter blocks.
# ---------------------------------------------------------------------------
# Recipe (see experiments/twoindex_tensor_search.py for derivation):
#   For one antisym^2(fund) hyper of  U(N) , the matter spec is a product
#   of  |P_N| = 2(N-1)(N-2) + 1  E_q factors, with charges
#
#       left  :   fund[a] + fund[b] + mu       , (a, b) in P_N
#       right : -antifund[a] - antifund[b] + mu , (a, b) in P_N
#
#   where  fund / antifund = _fundamental_wilson_support_UN / _antifund_wilson_
#   support_UN  (size  2N-1 ) and
#
#       P_N = { (a, b)  :  0 <= b <= a <= 2N - 2,
#                          a >= b + 2 + (b mod 2) } .
#
#   The natural priority  (a + b, -a) ASC  gives the order; for  N >= 5
#   a small  O(1)-per-anti-diagonal swap inside one anti-diagonal is needed,
#   which a guided DFS resolves in milliseconds.
#
#   matter_node initial charge :  lowest weight of antisym^2  +  mu
#                              =  eps_{N-1}^* + eps_N^*  +  mu  .

def _antisym2_pair_set_UN(N: int) -> list[tuple[int, int]]:
    """Closed-form  P_N  index pairs for the antisym^2(U(N))  matter
    Wilson-line support.

    Returns the list of pairs  (a, b) with  0 <= b <= a <= 2N - 2  and
    a >= b + 2 + (b mod 2) , sorted by  (a + b, -a) ASCENDING .

    |P_N| = 2(N-1)(N-2) + 1 = (N-1)^2 + (N-2)^2 .
    """
    if N < 2:
        raise ValueError(f"N >= 2 required for antisym^2(U(N)); got N={N}")
    fund_max = 2 * N - 2
    pairs = []
    for b in range(fund_max + 1):
        for a in range(b + 2 + (b % 2), fund_max + 1):
            pairs.append((a, b))
    pairs.sort(key=lambda p: (p[0] + p[1], -p[0]))
    return pairs


def _antisym2_dfs_order(
    N: int,
    B: list[list[int]],
    nodes: list[tuple[int, ...]],
    pure_spec_ext: list[tuple[int, ...]],
    matter_charges_unordered: list[tuple[int, ...]],
    pair_of_charge: dict,
    side: str,
    *,
    max_branches: int = 5_000_000,
) -> list[tuple[int, ...]]:
    """Find an ordering of  matter_charges_unordered  that completes a
    valid negating sequence on the given BPS quiver.

    Implements the guided-DFS strategy from  experiments/twoindex_tensor_
    search.py : at each step, try mutating the unique node whose current
    charge lies in the un-used matter set; prioritise candidates by the
    (a + b, -a)  pair-key.

    Returns the ordered matter spec.  Raises  RuntimeError  on failure.
    """
    from collections import Counter
    matter_set = frozenset(matter_charges_unordered)
    q_orig = BPSQuiver.from_pairing(nodes, B)
    if side == "right":
        q_init = q_orig
        for g in pure_spec_ext:
            idx = next(
                (k for k, c in enumerate(q_init.charges)
                 if tuple(c) == tuple(g) and not q_init.frozen[k]),
                None,
            )
            if idx is None:
                raise RuntimeError(
                    "pure-spec replay failed during right-placement setup"
                )
            q_init = q_init.mutate(idx)
    else:
        q_init = q_orig
    orig_neg = Counter(
        tuple(-x for x in q_orig.charges[k])
        for k in range(q_orig.n_nodes) if not q_orig.frozen[k]
    )
    visited: set = set()
    branches = [0]
    result: list = [None]

    def dfs(Q: BPSQuiver, used: frozenset, path: list) -> None:
        if result[0] is not None:
            return
        branches[0] += 1
        if branches[0] > max_branches:
            return
        if used == matter_set:
            if side == "left":
                Q2 = Q
                for g in pure_spec_ext:
                    idx = next(
                        (k for k, c in enumerate(Q2.charges)
                         if tuple(c) == tuple(g) and not Q2.frozen[k]),
                        None,
                    )
                    if idx is None:
                        return
                    Q2 = Q2.mutate(idx)
                final = Counter(
                    tuple(Q2.charges[k])
                    for k in range(Q2.n_nodes) if not Q2.frozen[k]
                )
            else:
                final = Counter(
                    tuple(Q.charges[k])
                    for k in range(Q.n_nodes) if not Q.frozen[k]
                )
            if final == orig_neg:
                result[0] = list(path)
            return
        key = (used, tuple(tuple(c) for c in Q.charges))
        if key in visited:
            return
        visited.add(key)
        candidates = []
        for k, c in enumerate(Q.charges):
            if Q.frozen[k]:
                continue
            ct = tuple(c)
            if ct in matter_set - used:
                p = pair_of_charge[ct]
                candidates.append(((p[0] + p[1], -p[0]), k, ct))
        candidates.sort()
        for _, k, ct in candidates:
            dfs(Q.mutate(k), used | {ct}, path + [ct])
            if result[0] is not None:
                return

    dfs(q_init, frozenset(), [])
    if result[0] is None:
        raise RuntimeError(
            f"antisym^2 DFS ordering failed: {branches[0]} branches, "
            f"|matter|={len(matter_set)}"
        )
    return result[0]


def _antisym2_charges_for_block(
    N: int,
    supp: list[tuple[int, ...]],
    sign: int,
    pairs: list[tuple[int, int]],
) -> tuple[list[tuple[int, ...]], dict]:
    """Build raw matter charges (no flavour) and the  charge -> (a, b)  map."""
    r_pure = 2 * N
    charges = []
    pair_of = {}
    for (a, b) in pairs:
        ch = tuple(sign * (supp[a][i] + supp[b][i]) for i in range(r_pure))
        charges.append(ch)
        pair_of[ch] = (a, b)
    return charges, pair_of


class UN_antisym:
    """U(N) gauge theory with  n_a  antisym^2(fund) hypermultiplets and
    ``Nf``  fundamental hypermultiplets.

    The antisym^2 matter block uses the closed-form recipe verified for
    N >= 2 :

        matter charges  =  fund[a] + fund[b] + mu        ('left'  block)
                        =  -antifund[a] - antifund[b] + mu ('right' block)
        (a, b) in  P_N  with the  (a+b, -a) ASC  pair order.
        matter_node     =  eps_{N-1}^* + eps_N^* + mu .

    Each antisym^2 block adds ONE flavour direction (a fresh  mu  in
    ker(B));  each fundamental block similarly adds one (built via the
    same recipe as :class:`UN_Nf`).

    Asymptotic-freedom strict check (default  ``strict=True`` ):

        n_a * (N - 2)  +  Nf  <=  2 N  .

    Parameters
    ----------
    N            : U(N) rank (must be >= 2).
    n_a          : number of antisym^2 hypers (default 1).
    Nf           : number of fundamental hypers (default 0).
    antisym_sides: per-antisym 'left'/'right' placement (default 'left').
                   Pass a string to apply uniformly, or a sequence of
                   length  n_a .
    fund_sides   : per-fund 'left'/'right' placement (default 'left'),
                   same semantics as :class:`UN_Nf` .
    strict       : if True (default), raise if AF inequality is violated.
    wrap_theory  : if True, also build a theory wrapper at
                   ``.theory`` .

    Spec assembly order  (each block in its block's order):

        S  =  [ antisym^2 left blocks ]  *  [ fund left blocks ]
              *  S_pure_U(N)
              *  [ fund right blocks ]  *  [ antisym^2 right blocks ]

    Attributes :
        N, n_a, Nf, antisym_sides, fund_sides
        B, nodes, spec, algebra, cone_witness
        antisym_specs   : list of length n_a , one matter block each
        antisym_nodes   : list of length n_a , one matter node each
        fund_specs      : list of length Nf , one matter block each
        fund_nodes      : list of length Nf , one matter node each
        pure            : the underlying  PureADE.U_N(N)  object
        theory (optional)
    """

    def __init__(
        self,
        N: int,
        n_a: int = 1,
        Nf: int = 0,
        *,
        antisym_sides: Union[str, Sequence[str], None] = None,
        fund_sides: Union[str, Sequence[str], None] = None,
        antisym_positions: Union[str, Sequence[str], None] = None,
        antisym_bps_aut_applied: Union[bool, Sequence[bool], None] = None,
        fund_positions: Union[str, Sequence[str], None] = None,
        fund_bps_aut_applied: Union[bool, Sequence[bool], None] = None,
        strict: bool = True,
        wrap_theory: bool = False,
    ):
        if N < 1:
            raise ValueError(f"N >= 1 required (got N={N})")
        if n_a < 0:
            raise ValueError(f"n_a >= 0 required (got n_a={n_a})")
        if n_a > 0 and N < 2:
            raise ValueError(
                f"antisym^2 of U(N) requires N >= 2 (got N={N}, n_a={n_a}); "
                f"antisym^2 of U(1) is trivial / undefined"
            )
        if Nf < 0:
            raise ValueError(f"Nf >= 0 required (got Nf={Nf}")
        if strict and n_a * (N - 2) + Nf > 2 * N:
            raise ValueError(
                f"asymptotic freedom violated for U({N}) with "
                f"n_a={n_a} antisym^2 and Nf={Nf} fund: "
                f"n_a*(N-2) + Nf = {n_a*(N-2)+Nf} > 2N = {2*N}.  "
                "Pass strict=False to override."
            )

        # --- Resolve antisym kwargs: (antisym_sides) XOR (antisym_positions, antisym_bps_aut_applied) ---
        if antisym_sides is not None and (
            antisym_positions is not None or antisym_bps_aut_applied is not None
        ):
            raise ValueError(
                "Specify either `antisym_sides` (legacy) OR "
                "(`antisym_positions`, `antisym_bps_aut_applied`) (new API)."
            )
        if (
            antisym_sides is None
            and antisym_positions is None
            and antisym_bps_aut_applied is None
        ):
            antisym_sides = "left"

        # --- Resolve fund kwargs: (fund_sides) XOR (fund_positions, fund_bps_aut_applied) ---
        if fund_sides is not None and (
            fund_positions is not None or fund_bps_aut_applied is not None
        ):
            raise ValueError(
                "Specify either `fund_sides` (legacy) OR "
                "(`fund_positions`, `fund_bps_aut_applied`) (new API)."
            )
        if (
            fund_sides is None
            and fund_positions is None
            and fund_bps_aut_applied is None
        ):
            fund_sides = "left"

        def _normalize_sides(arg, count, name):
            if isinstance(arg, str):
                if arg not in ("left", "right"):
                    raise ValueError(
                        f"{name} must be 'left' or 'right' (got {arg!r})"
                    )
                return [arg] * count
            sides = list(arg)
            if len(sides) != count:
                raise ValueError(
                    f"{name} list length {len(sides)} != {count}"
                )
            for s in sides:
                if s not in ("left", "right"):
                    raise ValueError(
                        f"each {name} entry must be 'left' or 'right' "
                        f"(got {s!r})"
                    )
            return sides

        def _normalize_positions(arg, count, name):
            if arg is None:
                return ["before"] * count
            if isinstance(arg, str):
                if arg not in ("before", "after"):
                    raise ValueError(
                        f"{name} must be 'before' or 'after' (got {arg!r})"
                    )
                return [arg] * count
            pos_list = list(arg)
            if len(pos_list) != count:
                raise ValueError(
                    f"{name} list length {len(pos_list)} != {count}"
                )
            for p in pos_list:
                if p not in ("before", "after"):
                    raise ValueError(
                        f"each {name} entry must be 'before' or 'after' "
                        f"(got {p!r})"
                    )
            return pos_list

        def _normalize_bps_flags(arg, count, name):
            if arg is None:
                return [False] * count
            if isinstance(arg, bool):
                return [arg] * count
            flags = list(arg)
            if len(flags) != count:
                raise ValueError(
                    f"{name} list length {len(flags)} != {count}"
                )
            for f in flags:
                if not isinstance(f, bool):
                    raise ValueError(
                        f"each {name} entry must be bool (got {f!r})"
                    )
            return flags

        # Resolve antisym: legacy or new path.
        if antisym_sides is not None:
            antisym_side_list = _normalize_sides(antisym_sides, n_a, "antisym_sides")
            antisym_positions_list = [
                "before" if s == "left" else "after" for s in antisym_side_list
            ]
            antisym_bps_aut_list = [False] * n_a
        else:
            antisym_positions_list = _normalize_positions(
                antisym_positions, n_a, "antisym_positions"
            )
            antisym_bps_aut_list = _normalize_bps_flags(
                antisym_bps_aut_applied, n_a, "antisym_bps_aut_applied"
            )
            antisym_side_list = [
                "left" if p == "before" else "right"
                for p in antisym_positions_list
            ]

        # Resolve fund: legacy or new path.
        if fund_sides is not None:
            fund_side_list = _normalize_sides(fund_sides, Nf, "fund_sides")
            fund_positions_list = [
                "before" if s == "left" else "after" for s in fund_side_list
            ]
            fund_bps_aut_list = [False] * Nf
        else:
            fund_positions_list = _normalize_positions(
                fund_positions, Nf, "fund_positions"
            )
            fund_bps_aut_list = _normalize_bps_flags(
                fund_bps_aut_applied, Nf, "fund_bps_aut_applied"
            )
            fund_side_list = [
                "left" if p == "before" else "right"
                for p in fund_positions_list
            ]

        self.N = N
        self.n_a = n_a
        self.Nf = Nf
        # Back-compat attribute surface.
        any_bps_aut = (
            any(antisym_bps_aut_list) or any(fund_bps_aut_list)
        )
        self.antisym_sides = (
            antisym_sides if antisym_sides is not None
            else (antisym_side_list if not any_bps_aut else None)
        )
        self.fund_sides = (
            fund_sides if fund_sides is not None
            else (fund_side_list if not any_bps_aut else None)
        )
        self.antisym_positions = tuple(antisym_positions_list)
        self.antisym_bps_aut_applied = tuple(antisym_bps_aut_list)
        self.fund_positions = tuple(fund_positions_list)
        self.fund_bps_aut_applied = tuple(fund_bps_aut_list)
        self._antisym_side_list = antisym_side_list
        self._fund_side_list = fund_side_list

        self.pure = PureADE.U_N(N)
        B_pure = [list(r) for r in self.pure.B]
        nodes_pure = [tuple(g) for g in self.pure.nodes]
        spec_pure = [tuple(g) for g in self.pure.spec]
        r_pure = len(B_pure)

        n_flavours = n_a + Nf
        if n_flavours == 0:
            self.B = B_pure
            self.nodes = nodes_pure
            self.spec = spec_pure
            self.antisym_specs = []
            self.antisym_nodes = []
            self.fund_specs = []
            self.fund_nodes = []
            self.matter_spec = []
            self.matter_nodes = []
            self.algebra = self.pure.algebra
            self.cone_witness = self.pure._canonical_cone_witness()
            if wrap_theory:
                from coulomb_algebra_theory import CoulombAlgebraTheory
                self.theory = CoulombAlgebraTheory(self.algebra)
            return

        r_UV = r_pure + n_flavours
        B_UV = [row + [0] * n_flavours for row in B_pure]
        B_UV += [[0] * r_UV for _ in range(n_flavours)]

        def _ext(v, extra):
            return tuple(list(v) + list(extra))

        # Flavour slot layout: antisym^2 blocks first (slots [0, n_a)), then
        # fundamental blocks (slots [n_a, n_a + Nf)).  Each block adds 1 mu.
        def _mu(slot_idx):
            return tuple(1 if k == r_pure + slot_idx else 0
                         for k in range(r_UV))

        # ----- Build antisym^2 blocks ------------------------------------
        fund_supp = _fundamental_wilson_support_UN(N)
        antifund_supp = _antifund_wilson_support_UN(N)
        pairs = _antisym2_pair_set_UN(N)

        antisym_left: list[tuple[int, ...]] = []
        antisym_right: list[tuple[int, ...]] = []
        antisym_specs_by_block: list[list[tuple[int, ...]]] = []
        antisym_nodes: list[tuple[int, ...]] = []

        # Lowest weight of antisym^2 in pure-U(N) coords: eps_{N-1} + eps_N
        lw_antisym_gauge = [0] * r_pure
        lw_antisym_gauge[2 * (N - 1) - 1] = 1
        lw_antisym_gauge[2 * N - 1] = 1

        # Per-antisym DFS ordering happens in a quiver with only THIS block's
        # mu present (the other flavour directions are decoupled, so we can
        # build each block in isolation on a (pure + 1 mu)-extended quiver).
        for j in range(n_a):
            side = antisym_side_list[j]
            mu_j = _mu(j)
            # Build the standalone (pure + this-mu) quiver for the DFS.
            r_local = r_pure + 1
            B_local = [row + [0] for row in B_pure]
            B_local += [[0] * r_local]

            if side == "left":
                supp = fund_supp
                sign = +1
            else:
                supp = antifund_supp
                sign = -1
            raw_charges, pair_of = _antisym2_charges_for_block(
                N, supp, sign, pairs
            )
            matter_charges_local = [
                tuple(list(c) + [1]) for c in raw_charges
            ]
            pair_of_with_mu = {
                tuple(list(c) + [1]): pair_of[c] for c in raw_charges
            }
            matter_node_local = tuple(lw_antisym_gauge + [1])
            nodes_local = (
                [tuple(list(g) + [0]) for g in nodes_pure] + [matter_node_local]
            )
            pure_spec_local = [tuple(list(g) + [0]) for g in spec_pure]
            ordered_local = _antisym2_dfs_order(
                N, B_local, nodes_local, pure_spec_local,
                matter_charges_local, pair_of_with_mu, side,
            )

            # Lift ordered local charges back into the full UV lattice.
            extra_pad_before = j  # flavour slots before this block
            extra_pad_after = (n_a - 1 - j) + Nf
            def _lift(v_local):
                # v_local has length r_pure + 1 (last is this block's mu).
                gauge_part = list(v_local[:r_pure])
                mu_val = v_local[r_pure]
                pad_before = [0] * extra_pad_before
                pad_after = [0] * extra_pad_after
                return tuple(
                    gauge_part + pad_before + [mu_val] + pad_after
                )

            ordered_full = [_lift(c) for c in ordered_local]
            matter_node_full = _lift(matter_node_local)
            # Per-block independent bps_aut application: σ on this block's
            # matter charges + matter node.  Other blocks and pure-gauge
            # untouched.
            if antisym_bps_aut_list[j]:
                ordered_full = [
                    apply_bps_aut_to_charge_UN(c, N) for c in ordered_full
                ]
                matter_node_full = apply_bps_aut_to_charge_UN(
                    matter_node_full, N
                )
            antisym_specs_by_block.append(ordered_full)
            antisym_nodes.append(matter_node_full)
            (antisym_left if side == "left" else antisym_right).extend(
                ordered_full
            )

        # ----- Build fundamental blocks (parallel to UN_Nf) --------------
        fund_left: list[tuple[int, ...]] = []
        fund_right: list[tuple[int, ...]] = []
        fund_specs_by_block: list[list[tuple[int, ...]]] = []
        fund_nodes: list[tuple[int, ...]] = []
        # Lowest weight of fundamental of U(N) :  eps_N
        lw_fund = [0] * r_pure
        lw_fund[r_pure - 1] = 1
        alphas_left = _fundamental_wilson_support_UN(N)
        alphas_right = [
            tuple(-x for x in a) for a in _antifund_wilson_support_UN(N)
        ]
        for j in range(Nf):
            slot = n_a + j
            mu_j = _mu(slot)
            side = fund_side_list[j]
            alphas_j = alphas_left if side == "left" else alphas_right
            block_charges = []
            for a in alphas_j:
                charge = tuple(
                    x + y for x, y in zip(_ext(a, [0] * n_flavours), mu_j)
                )
                block_charges.append(charge)
            matter_node_j = tuple(
                x + y for x, y in zip(_ext(lw_fund, [0] * n_flavours), mu_j)
            )
            # Per-fund-block independent bps_aut application.
            if fund_bps_aut_list[j]:
                block_charges = [
                    apply_bps_aut_to_charge_UN(c, N) for c in block_charges
                ]
                matter_node_j = apply_bps_aut_to_charge_UN(matter_node_j, N)
            for charge in block_charges:
                (fund_left if side == "left" else fund_right).append(charge)
            fund_specs_by_block.append(block_charges)
            fund_nodes.append(matter_node_j)

        pure_spec_ext = [_ext(g, [0] * n_flavours) for g in spec_pure]
        nodes_UV = (
            [_ext(g, [0] * n_flavours) for g in nodes_pure]
            + antisym_nodes + fund_nodes
        )
        # S = (antisym^2 left) (fund left) S_pure (fund right) (antisym^2 right)
        spec_UV = (
            antisym_left + fund_left + pure_spec_ext + fund_right
            + antisym_right
        )
        matter_spec_all = (
            antisym_left + fund_left + fund_right + antisym_right
        )

        # Cone witness (matches UN_Nf style: extend pure witness, top up mu's)
        f_pure = list(self.pure._canonical_cone_witness())
        # Witness must be non-negative on every node charge.  Each matter node
        # has gauge part = (lowest-weight-of-block).  Pick mu coefficient large
        # enough that  f_pure . gauge_part + c >= 1 .
        witness_extra = []
        for matter_node in (antisym_nodes + fund_nodes):
            gauge = matter_node[:r_pure]
            val = sum(fi * gi for fi, gi in zip(f_pure, gauge))
            witness_extra.append(max(1 - val, 0))
        witness = tuple(f_pure + witness_extra)

        self.B = B_UV
        self.nodes = nodes_UV
        self.spec = spec_UV
        self.antisym_specs = antisym_specs_by_block
        self.antisym_nodes = antisym_nodes
        self.fund_specs = fund_specs_by_block
        self.fund_nodes = fund_nodes
        self.matter_spec = matter_spec_all
        self.matter_nodes = antisym_nodes + fund_nodes
        self.cone_witness = witness
        self.algebra = CoulombAlgebra(
            pairing=B_UV, node_charges=nodes_UV, spec=spec_UV,
            cone_witness=witness,
            skip_spec_cone_check=True,
            skip_cone_check=True,
        )
        if wrap_theory:
            from coulomb_algebra_theory import CoulombAlgebraTheory
            self.theory = CoulombAlgebraTheory(self.algebra)

    def verify(self) -> bool:
        ok, _ = self.algebra.quiver.verify_spectrum_generator(self.spec)
        return ok

    def summary(self) -> str:
        return (
            f"UN_antisym(N={self.N}, n_a={self.n_a}, Nf={self.Nf}, "
            f"antisym_positions={self.antisym_positions}, "
            f"antisym_bps_aut_applied={self.antisym_bps_aut_applied}, "
            f"fund_positions={self.fund_positions}, "
            f"fund_bps_aut_applied={self.fund_bps_aut_applied})\n"
            f"  rank={len(self.B)}  |nodes|={len(self.nodes)}  "
            f"|spec|={len(self.spec)}  "
            f"|antisym^2 block|=|P_N|={2*(self.N-1)*(self.N-2)+1}"
        )

    def __repr__(self) -> str:
        return self.summary()


class UN_bifund:
    """``U(N_1) x U(N_2)``  gauge theory with one bifundamental
    hypermultiplet and optional per-factor fundamentals.

    Exact  U -analogue of :class:`SUN_bifund` :  uses
    :class:`UN_Nf`  per factor and the U(N) fund Wilson-line support
    (:func:`_fundamental_wilson_support_UN` ) instead of the SU(N)
    one.  Same parity-chain ordering rule  (:func:`_bifund_pair_order_AN` )
    works without modification -- verified on  ``N_i <= 5`` .

    API mirrors :class:`SUN_bifund` : ``.B`` , ``.nodes`` , ``.spec`` ,
    ``.algebra`` , ``.bifund_spec`` , ``.bifund_node`` , ``.pure1`` /
    ``.pure2``  (the per-factor ``UN_Nf`` wrappers), ``.cone_witness`` ,
    optional ``.theory``  with ``wrap_theory=True`` , ``.verify()`` .

    Asymptotic freedom is the same constraint as for SU:
    ``N_2 + Nf_1 <= 2 N_1``  and  ``N_1 + Nf_2 <= 2 N_2`` .

    Placement variants
    ------------------
    ``bifund_side`` controls the position of the bifundamental matter
    block relative to the two pure-gauge spec blocks:

      * ``"left"``    (default, canonical):  ``S = M_bifund * S_pure_1 * S_pure_2``
        with bifund charges built from
        ``(fund(N_1), fund(N_2))``  (left of both factors).
      * ``"between"``:                       ``S = S_pure_1 * M_bifund * S_pure_2``
        with bifund charges from
        ``(-antifund(N_1), fund(N_2))``  (right of factor 1, left of factor 2).
      * ``"right"``:                         ``S = S_pure_1 * S_pure_2 * M_bifund``
        with bifund charges from
        ``(-antifund(N_1), -antifund(N_2))``  (right of both factors).

    The same parity-chain  ``_bifund_pair_order_AN``  ordering is used
    for all three placements (it depends only on the support sizes,
    ``n_i = 2N_i - 1`` , not on the specific charges).  Per-factor
    fundamentals  Nf_i  are placed canonically  (``UN_Nf(..., matter_side='left')``)
    for backwards compatibility — pass ``fund1_sides`` /
    ``fund2_sides``  to override per-mu.
    """

    def __init__(
        self,
        N1: int,
        N2: int,
        *,
        Nf1: int = 0,
        Nf2: int = 0,
        wrap_theory: bool = False,
        strict: bool = True,
        bifund_side: str = "left",
        fund1_sides: Optional[Sequence[str]] = None,
        fund2_sides: Optional[Sequence[str]] = None,
        # New per-leg BPS-quiver-automorphism flags for the bifund block:
        bifund_bps_aut_1: bool = False,
        bifund_bps_aut_2: bool = False,
        # Per-hyper bps_aut flags for the per-factor fundamental hypers:
        fund1_bps_aut_applied: Union[bool, Sequence[bool], None] = None,
        fund2_bps_aut_applied: Union[bool, Sequence[bool], None] = None,
    ):
        if N1 < 1 or N2 < 1:
            raise ValueError(f"N_i >= 1 required, got N1={N1}, N2={N2}")
        if Nf1 < 0 or Nf2 < 0:
            raise ValueError(
                f"Nf_i >= 0 required, got Nf1={Nf1}, Nf2={Nf2}"
            )
        if strict:
            if N2 + Nf1 > 2 * N1:
                raise ValueError(
                    f"asymptotic freedom on U({N1}) side: "
                    f"N2 + Nf1 = {N2 + Nf1} > 2 * N1 = {2 * N1}"
                )
            if N1 + Nf2 > 2 * N2:
                raise ValueError(
                    f"asymptotic freedom on U({N2}) side: "
                    f"N1 + Nf2 = {N1 + Nf2} > 2 * N2 = {2 * N2}"
                )
        if bifund_side not in ("left", "between", "right"):
            raise ValueError(
                f"bifund_side must be 'left', 'between', or 'right' "
                f"(got {bifund_side!r})"
            )
        if not isinstance(bifund_bps_aut_1, bool):
            raise ValueError(
                f"bifund_bps_aut_1 must be bool (got {bifund_bps_aut_1!r})"
            )
        if not isinstance(bifund_bps_aut_2, bool):
            raise ValueError(
                f"bifund_bps_aut_2 must be bool (got {bifund_bps_aut_2!r})"
            )
        self.N1 = N1
        self.N2 = N2
        self.Nf1 = Nf1
        self.Nf2 = Nf2
        self.bifund_side = bifund_side
        self.bifund_bps_aut_1 = bool(bifund_bps_aut_1)
        self.bifund_bps_aut_2 = bool(bifund_bps_aut_2)

        # Per-factor fund hypers: pass through legacy fund*_sides OR new
        # fund*_bps_aut_applied to UN_Nf's new API.
        if fund1_sides is not None and fund1_bps_aut_applied is not None:
            raise ValueError(
                "Specify either `fund1_sides` (legacy) OR "
                "`fund1_bps_aut_applied` (new), not both."
            )
        if fund2_sides is not None and fund2_bps_aut_applied is not None:
            raise ValueError(
                "Specify either `fund2_sides` (legacy) OR "
                "`fund2_bps_aut_applied` (new), not both."
            )
        if fund1_bps_aut_applied is not None:
            # Treat all-'before' positions (matter_side='left' equivalent)
            # plus per-hyper bps_aut flips.  User can also pass bare side
            # legacy if they want positions != all-'before'.
            self.pure1 = UN_Nf(
                N1, Nf1,
                positions="before",
                bps_aut_applied=fund1_bps_aut_applied,
            )
        else:
            fund1_arg = list(fund1_sides) if fund1_sides is not None else "left"
            self.pure1 = UN_Nf(N1, Nf1, matter_side=fund1_arg)
        if fund2_bps_aut_applied is not None:
            self.pure2 = UN_Nf(
                N2, Nf2,
                positions="before",
                bps_aut_applied=fund2_bps_aut_applied,
            )
        else:
            fund2_arg = list(fund2_sides) if fund2_sides is not None else "left"
            self.pure2 = UN_Nf(N2, Nf2, matter_side=fund2_arg)
        r1_full = len(self.pure1.B)   # 2 N_1 + Nf_1
        r2_full = len(self.pure2.B)   # 2 N_2 + Nf_2
        r_pure_1 = 2 * N1
        r_pure_2 = 2 * N2
        total = r1_full + r2_full + 1

        B = [[0] * total for _ in range(total)]
        for i in range(r1_full):
            for j in range(r1_full):
                B[i][j] = int(self.pure1.B[i][j])
        for i in range(r2_full):
            for j in range(r2_full):
                B[r1_full + i][r1_full + j] = int(self.pure2.B[i][j])

        def pad_1(v):
            return tuple(list(v) + [0] * (total - r1_full))

        def pad_2(v):
            return tuple([0] * r1_full + list(v) + [0])

        # Choose bifund slot-i alphas based on placement.  "left" means
        # bifund sits to the left of the corresponding pure-gauge spec
        # (use the fundamental support); "right" means right (use the
        # charge-reversed anti-fund).  "between" is right of pure_1,
        # left of pure_2.
        side1 = "left" if bifund_side == "left" else "right"
        side2 = "left" if bifund_side in ("left", "between") else "right"
        if side1 == "left":
            alphas1 = _fundamental_wilson_support_UN(N1)
        else:
            alphas1 = [tuple(-x for x in a)
                       for a in _antifund_wilson_support_UN(N1)]
        if side2 == "left":
            alphas2 = _fundamental_wilson_support_UN(N2)
        else:
            alphas2 = [tuple(-x for x in a)
                       for a in _antifund_wilson_support_UN(N2)]
        n1 = 2 * N1 - 1
        n2 = 2 * N2 - 1
        pair_order = _bifund_pair_order_AN(n1, n2)

        def bifund_charge(i: int, j: int) -> tuple[int, ...]:
            out = [0] * total
            for k, x in enumerate(alphas1[i]): out[k] = int(x)
            for k, x in enumerate(alphas2[j]): out[r1_full + k] = int(x)
            out[-1] = 1
            return tuple(out)

        bifund_spec = [bifund_charge(i, j) for (i, j) in pair_order]
        # The bifund "starter" node:  (lowest_1, lowest_2, +1)  -- same
        # convention as the canonical (left) placement; the matter node
        # mutates through the spec to the appropriate state regardless
        # of bifund_side, exactly as in :class:`UN_Nf` .
        lowest_node = [0] * total
        lowest_node[r_pure_1 - 1] = 1
        lowest_node[r1_full + r_pure_2 - 1] = 1
        lowest_node[-1] = 1
        bifund_node = tuple(lowest_node)

        # Per-leg independent bps_aut application on the bifund block.
        # bps_aut on slot 1 acts on positions [0, 2*N1); on slot 2 acts on
        # [r1_full, r1_full + 2*N2).  Per-factor flavour mus (if any) and
        # the bifund flavour mu (last position) are untouched.
        if bifund_bps_aut_1:
            bifund_spec = [
                apply_bps_aut_to_charge_UN(c, N1, gauge_offset=0)
                for c in bifund_spec
            ]
            bifund_node = apply_bps_aut_to_charge_UN(
                bifund_node, N1, gauge_offset=0,
            )
        if bifund_bps_aut_2:
            bifund_spec = [
                apply_bps_aut_to_charge_UN(c, N2, gauge_offset=r1_full)
                for c in bifund_spec
            ]
            bifund_node = apply_bps_aut_to_charge_UN(
                bifund_node, N2, gauge_offset=r1_full,
            )

        nodes_UV: list[tuple[int, ...]] = (
            [pad_1(g) for g in self.pure1.nodes]
            + [pad_2(g) for g in self.pure2.nodes]
            + [bifund_node]
        )
        spec_1 = [pad_1(g) for g in self.pure1.spec]
        spec_2 = [pad_2(g) for g in self.pure2.spec]
        if bifund_side == "left":
            spec_UV = bifund_spec + spec_1 + spec_2
        elif bifund_side == "between":
            spec_UV = spec_1 + bifund_spec + spec_2
        else:  # right
            spec_UV = spec_1 + spec_2 + bifund_spec

        f1 = list(self.pure1.cone_witness)
        f2 = list(self.pure2.cone_witness)
        f_bifund = f1 + f2 + [0]
        # Use gauge part of (possibly bps_aut'd) bifund_node for the
        # cone-witness top-up.
        bn_val = sum(fi * gi for fi, gi in zip(f_bifund, bifund_node))
        f_bifund[-1] = max(1 - bn_val, 0)
        witness = tuple(f_bifund)

        self.B = B
        self.nodes = nodes_UV
        self.spec = spec_UV
        self.bifund_spec = bifund_spec
        self.bifund_node = bifund_node
        self.cone_witness = witness
        self.algebra = CoulombAlgebra(
            pairing=B,
            node_charges=nodes_UV,
            spec=spec_UV,
            cone_witness=witness,
            skip_spec_cone_check=True,
            skip_cone_check=True,
        )
        if wrap_theory:
            from coulomb_algebra_theory import CoulombAlgebraTheory
            self.theory = CoulombAlgebraTheory(self.algebra)

    def verify(self) -> bool:
        ok, _ = self.algebra.quiver.verify_spectrum_generator(self.spec)
        return ok

    def summary(self) -> str:
        return (
            f"UN_bifund(N1={self.N1}, Nf1={self.Nf1}, "
            f"N2={self.N2}, Nf2={self.Nf2}, "
            f"bifund_side={self.bifund_side!r}, "
            f"bifund_bps_aut_1={self.bifund_bps_aut_1}, "
            f"bifund_bps_aut_2={self.bifund_bps_aut_2})\n"
            f"  rank={len(self.B)}  |nodes|={len(self.nodes)}  "
            f"|spec|={len(self.spec)}  |bifund|={len(self.bifund_spec)}"
        )

    def __repr__(self) -> str:
        return self.summary()


class LinearUQuiver:
    """Linear quiver gauge theory  ``U(N_1) - U(N_2) - ... - U(N_k)`` ,
    with  ``M_i``  fundamental hypermultiplets attached to node  i
    and bifundamentals between each consecutive pair  (i, i+1) .

    Parameters
    ----------
    N_list
        Sequence of gauge ranks  ``[N_1, N_2, ..., N_k]``  (each  >= 1 ).
    M_list
        Fundamental counts ``[M_1, ..., M_k]``  (default: all zero).
    fund_sides
        Per-factor list of per-mu placements:  ``fund_sides[i]``  is a
        list of length  ``M_list[i]``  with each entry  ``"left"``  or
        ``"right"`` , specifying whether the corresponding fundamental
        of factor  i  sits to the left of  S_pure_i  (canonical) or to
        the right (see :class:`UN_Nf` ).  Default: all ``"left"`` .
    bifund_placements
        Per-edge placement ``["left" | "between" | "right"]``  of length
        ``k - 1`` , specifying the position of each edge's bifund block
        relative to the two pure-gauge spec blocks it couples:

          * ``"left"``  → before  F_i        (= left of both factors)
          * ``"between"`` → between F_i and F_{i+1}
          * ``"right"`` → after  F_{i+1}    (= right of both factors)

        Bifund block charges for edge  (i, i+1)  use:

          * ``"left"``    →  (fund(N_i),       fund(N_{i+1}))
          * ``"between"`` →  (-antifund(N_i),  fund(N_{i+1}))
          * ``"right"``   →  (-antifund(N_i),  -antifund(N_{i+1}))

        Default: all ``"left"``  (canonical).
    wrap_theory
        If True, also build a theory wrapper at
        ``self.theory`` .
    strict
        If True (default), enforce the asymptotic-freedom gate on
        every internal node:

            N_{i-1}  +  M_i  +  N_{i+1}  <=  2 N_i

        (with  N_0 = N_{k+1} = 0  at the ends).

    Construction (canonical default): for each consecutive pair
    ``(i, i+1)``  emit a parity-chain-ordered bifundamental block
    (as in :class:`UN_bifund` , with its own flavour direction), then
    concatenate all per-factor :class:`UN_Nf` specs:

        S  =  bifund(1,2) * bifund(2,3) * ... * bifund(k-1,k)
              * S_pure_1 * S_pure_2 * ... * S_pure_k .

    With placement variants, the assembly interleaves bifund blocks
    into per-factor "gaps":  before  F_1 ,  between  F_i  and
    F_{i+1} ,  after  F_k .  Multiple bifund blocks routed to the
    same gap are emitted in increasing edge order; within each
    factor block, left-side fundamentals precede the pure-gauge
    spec which precedes the right-side fundamentals.

    Attributes
    ----------
    .factors
        List of the per-node :class:`UN_Nf` theories.
    .bifund_specs
        List of per-edge bifund spec blocks, ``len == k - 1`` .
    .bifund_nodes
        List of per-edge bifund matter nodes.
    .B, .nodes, .spec, .algebra, .cone_witness, .theory (optional)
        Standard :class:`SUN_Nf`-style surface.
    """

    def __init__(
        self,
        N_list,
        M_list=None,
        *,
        wrap_theory: bool = False,
        strict: bool = True,
        fund_sides: Optional[Sequence[Sequence[str]]] = None,
        bifund_placements: Optional[Sequence[str]] = None,
    ):
        N_list = list(N_list)
        if M_list is None:
            M_list = [0] * len(N_list)
        else:
            M_list = list(M_list)
        if len(N_list) != len(M_list):
            raise ValueError(
                f"len(N_list) ({len(N_list)}) != len(M_list) ({len(M_list)})"
            )
        k = len(N_list)
        if k < 1:
            raise ValueError("need at least one gauge factor")
        for i, Ni in enumerate(N_list):
            if Ni < 1:
                raise ValueError(
                    f"N_list[{i}] = {Ni}; each N_i >= 1 required"
                )
        for i, Mi in enumerate(M_list):
            if Mi < 0:
                raise ValueError(f"M_list[{i}] = {Mi}; each M_i >= 0")
        if strict:
            N_ext = [0] + list(N_list) + [0]
            for i in range(k):
                lhs = N_ext[i] + M_list[i] + N_ext[i + 2]
                rhs = 2 * N_list[i]
                if lhs > rhs:
                    raise ValueError(
                        f"asymptotic freedom at node {i+1} (U({N_list[i]})): "
                        f"N_{i} + M_{i+1} + N_{i+2} = {lhs} > "
                        f"2 N_{i+1} = {rhs}"
                    )

        # Per-factor fund placements
        if fund_sides is None:
            fund_sides_norm: list[list[str]] = [["left"] * Mi for Mi in M_list]
        else:
            fund_sides_norm = [list(s) for s in fund_sides]
            if len(fund_sides_norm) != k:
                raise ValueError(
                    f"fund_sides length {len(fund_sides_norm)} != k = {k}"
                )
            for i, s in enumerate(fund_sides_norm):
                if len(s) != M_list[i]:
                    raise ValueError(
                        f"fund_sides[{i}] length {len(s)} != M_list[{i}] = "
                        f"{M_list[i]}"
                    )
        # Per-edge bifund placements
        n_bifund = k - 1
        if bifund_placements is None:
            bifund_placements_norm = ["left"] * n_bifund
        else:
            bifund_placements_norm = list(bifund_placements)
            if len(bifund_placements_norm) != n_bifund:
                raise ValueError(
                    f"bifund_placements length {len(bifund_placements_norm)} "
                    f"!= k - 1 = {n_bifund}"
                )
            for p in bifund_placements_norm:
                if p not in ("left", "between", "right"):
                    raise ValueError(
                        f"bifund_placements entries must be 'left', "
                        f"'between', or 'right' (got {p!r})"
                    )

        self.k = k
        self.N_list = N_list
        self.M_list = M_list
        self.fund_sides = fund_sides_norm
        self.bifund_placements = bifund_placements_norm

        # Per-factor theories (hardcoded, fast).  Each per-factor block
        # arrives as  fund_left + S_pure + fund_right  by virtue of the
        # per-mu  matter_side  inside  UN_Nf .
        self.factors = [
            UN_Nf(Ni, Mi, matter_side=fund_sides_norm[i] if Mi > 0 else "left")
            for i, (Ni, Mi) in enumerate(zip(N_list, M_list))
        ]
        r_full = [len(f.B) for f in self.factors]   # 2 N_i + M_i
        r_pure = [2 * Ni for Ni in N_list]

        factor_start = [0]
        for rf in r_full:
            factor_start.append(factor_start[-1] + rf)
        gauge_dim = factor_start[-1]          # sum of r_full
        n_bifund = k - 1
        total = gauge_dim + n_bifund          # + one flavour per bifund

        # Block-diagonal pairing from the per-factor B matrices.
        B = [[0] * total for _ in range(total)]
        for fi, f in enumerate(self.factors):
            s = factor_start[fi]
            for i in range(r_full[fi]):
                for j in range(r_full[fi]):
                    B[s + i][s + j] = int(f.B[i][j])
        # Bifund flavour slots are in ker B (zeros everywhere).

        # Per-factor nodes / spec, padded to the full ambient.
        def pad(v, start):
            out = [0] * total
            for k_, x in enumerate(v):
                out[start + k_] = int(x)
            return tuple(out)

        nodes_per_factor = [
            [pad(g, factor_start[fi]) for g in f.nodes]
            for fi, f in enumerate(self.factors)
        ]
        spec_per_factor = [
            [pad(g, factor_start[fi]) for g in f.spec]
            for fi, f in enumerate(self.factors)
        ]

        # Bifund blocks.  For the edge (i, i+1), use the parity-chain
        # recipe on the appropriate slot-i and slot-(i+1) supports,
        # chosen per-placement:
        #   "left"    -> ( fund(N_i),       fund(N_{i+1})    )
        #   "between" -> (-antifund(N_i),   fund(N_{i+1})    )
        #   "right"   -> (-antifund(N_i),  -antifund(N_{i+1}))
        # Same parity-chain pair-order is used in all three cases (it
        # depends only on the support sizes n_i = 2N_i - 1).
        self.bifund_specs: list[list[tuple[int, ...]]] = []
        self.bifund_nodes: list[tuple[int, ...]] = []
        for edge in range(n_bifund):
            placement = bifund_placements_norm[edge]
            Ni, Nj = N_list[edge], N_list[edge + 1]
            side_i = "left" if placement == "left" else "right"
            side_j = "left" if placement in ("left", "between") else "right"
            if side_i == "left":
                alphas_i = _fundamental_wilson_support_UN(Ni)
            else:
                alphas_i = [tuple(-x for x in a)
                            for a in _antifund_wilson_support_UN(Ni)]
            if side_j == "left":
                alphas_j = _fundamental_wilson_support_UN(Nj)
            else:
                alphas_j = [tuple(-x for x in a)
                            for a in _antifund_wilson_support_UN(Nj)]
            n_i, n_j = 2 * Ni - 1, 2 * Nj - 1
            pair_order = _bifund_pair_order_AN(n_i, n_j)
            si = factor_start[edge]
            sj = factor_start[edge + 1]
            mu_slot = gauge_dim + edge

            def bifund_charge(a, b, si=si, sj=sj, mu_slot=mu_slot,
                              alphas_i=alphas_i, alphas_j=alphas_j):
                out = [0] * total
                for k_, x in enumerate(alphas_i[a]):
                    out[si + k_] = int(x)
                for k_, x in enumerate(alphas_j[b]):
                    out[sj + k_] = int(x)
                out[mu_slot] = 1
                return tuple(out)

            block = [bifund_charge(a, b) for (a, b) in pair_order]
            self.bifund_specs.append(block)
            # Bifund "starter" node: (lowest_i, lowest_{i+1}, +1) in the
            # corresponding slots.  Independent of placement -- same
            # rationale as in :class:`UN_Nf` and :class:`UN_bifund` .
            starter = [0] * total
            starter[si + r_pure[edge] - 1] = 1
            starter[sj + r_pure[edge + 1] - 1] = 1
            starter[mu_slot] = 1
            self.bifund_nodes.append(tuple(starter))

        # Route each bifund block to a "gap": gap_g sits between factor
        # g-1 and factor g (g=0 .. k).  For edge e=(i, i+1):
        #   "left"    -> gap i        (before  F_i)
        #   "between" -> gap (i + 1)  (between F_i and F_{i+1})
        #   "right"   -> gap (i + 2)  (after   F_{i+1})
        gaps: list[list[list[tuple[int, ...]]]] = [[] for _ in range(k + 1)]
        for edge in range(n_bifund):
            placement = bifund_placements_norm[edge]
            if placement == "left":
                gap_idx = edge          # before F_{edge+1}, i.e. before F_i
            elif placement == "between":
                gap_idx = edge + 1
            else:  # "right"
                gap_idx = edge + 2
            gaps[gap_idx].append(self.bifund_specs[edge])

        # Assemble nodes, spec.
        nodes_UV: list[tuple[int, ...]] = []
        for fi in range(k):
            nodes_UV.extend(nodes_per_factor[fi])
        nodes_UV.extend(self.bifund_nodes)

        spec_UV: list[tuple[int, ...]] = []
        for fi in range(k):
            for block in gaps[fi]:
                spec_UV.extend(block)
            spec_UV.extend(spec_per_factor[fi])
        for block in gaps[k]:
            spec_UV.extend(block)

        # Cone witness: per-factor witnesses, then 0 on each bifund slot
        # adjusted so  f . bifund_node_edge  >= 1 .
        f_lin: list[int] = []
        for fi, f in enumerate(self.factors):
            f_lin.extend(int(x) for x in f.cone_witness)
        f_lin.extend([0] * n_bifund)
        for edge in range(n_bifund):
            bn = self.bifund_nodes[edge]
            val = sum(fi_ * gi for fi_, gi in zip(f_lin, bn))
            c = max(1 - val, 0)
            f_lin[gauge_dim + edge] = c
        witness = tuple(f_lin)

        self.B = B
        self.nodes = nodes_UV
        self.spec = spec_UV
        self.cone_witness = witness
        self.algebra = CoulombAlgebra(
            pairing=B,
            node_charges=nodes_UV,
            spec=spec_UV,
            cone_witness=witness,
            skip_spec_cone_check=True,
            skip_cone_check=True,
        )
        if wrap_theory:
            from coulomb_algebra_theory import CoulombAlgebraTheory
            self.theory = CoulombAlgebraTheory(self.algebra)

    def verify(self) -> bool:
        ok, _ = self.algebra.quiver.verify_spectrum_generator(self.spec)
        return ok

    def cluster_seed_basis(self, chart):
        """Cluster-seed basis (mutable + frozen), with explicit chart Laurent monomials and
        BPS lattice gammas.  See :func:`cluster_seed.cluster_seed_basis`."""
        from cluster_seed import cluster_seed_basis as _csb
        return _csb(self, chart)

    def bps_lattice_coords(self):
        """Coord map for the BPS lattice.  See :func:`cluster_seed.bps_lattice_coords`."""
        from cluster_seed import bps_lattice_coords as _blc
        return _blc(self)

    def verify_cluster_seed_basis(self, chart, q_max=14):
        """Verify pairwise BPS-vs-chart q-commutation on the cluster-seed basis.

        Returns  (n_pairs, n_inconsistencies, report) ."""
        from cluster_seed import verify_cluster_seed_basis as _vcsb
        return _vcsb(self, chart, q_max=q_max)

    def register_cluster_seed_with_theory(self, chart):
        """Integrate the cluster-seed basis with the theory
        wrapper:  every cluster-seed name (s_i_m, t_i_m, bf_i_j, fl_i_k,
        frozen_*) is registered as an intrinsic label = the BPS gamma.

        Requires  wrap_theory=True  on construction.

        Returns the dict  name -> (gamma, chart_D) ."""
        if not hasattr(self, 'theory'):
            raise RuntimeError(
                "LinearUQuiver was built without wrap_theory=True; "
                "no .theory to register against.  Re-build with wrap_theory=True."
            )
        seeds = self.cluster_seed_basis(chart)
        for name, (g, _D) in seeds.items():
            self.theory.register(name, g)
        return seeds

    def gamma_to_chart_monomial(self, chart, gamma):
        """Convert a BPS gamma to its single-Laurent chart-image."""
        from cluster_seed import gamma_to_chart_monomial as _g2c
        return _g2c(self, chart, gamma)

    def chart_monomial_to_gamma(self, chart_D):
        """Convert a single-Laurent chart DOp to its BPS gamma."""
        from cluster_seed import chart_monomial_to_gamma as _c2g
        return _c2g(self, chart_D)

    def E_dressed(self, chart, n_tuple, f_partitions=None):
        """Dressed minuscule monopole  E_{n, f}  per coulomb-2026 eq. 691."""
        from coulomb_2026 import E_dressed
        return E_dressed(self, chart, n_tuple, f_partitions)

    def F_dressed(self, chart, n_tuple, f_partitions=None):
        """Dressed minuscule monopole  F_{n, f}  per coulomb-2026 eq. 702."""
        from coulomb_2026 import F_dressed
        return F_dressed(self, chart, n_tuple, f_partitions)

    def label_registry(self, chart, populate_simple_roots=True):
        """Return a TropicalLabelRegistry for this LinearUQuiver, optionally
        pre-populated with the simple-root E and F cluster A-variables."""
        from coulomb_2026 import TropicalLabelRegistry, standard_label_registry
        if populate_simple_roots:
            return standard_label_registry(self, chart)
        return TropicalLabelRegistry(self, chart)

    def cE(self, chart, n_tuple, f_partitions=None):
        """cE_{n, f}  (calligraphic dressed-monopole;
        distinct from F_gamma BPS canonical-basis element)."""
        from coulomb_2026 import cE
        return cE(self, chart, n_tuple, f_partitions)

    def cF(self, chart, n_tuple, f_partitions=None):
        """cF_{n, f}  (calligraphic dressed-anti-monopole)."""
        from coulomb_2026 import cF
        return cF(self, chart, n_tuple, f_partitions)

    def cH(self, chart, l, n):
        """cH_n  Toda Hamiltonian at gauge node l."""
        from coulomb_2026 import cH
        return cH(self, chart, l, n)

    def is_canonical_basis(self, chart, side, n_tuple, f_partitions=None,
                           gamma_search=None):
        """Test whether  cE_{n, f}  (side='E') or  cF_{n, f}  (side='F')
        equals a single BPS canonical-basis element  F_gamma  under the
        algebra isomorphism.

        Compares cE/cF directly against the chart-image of F_γ (= sum of
        chart Laurent monomials with q-coefficients per A.F(γ)) for each
        candidate γ.

        Returns  (is_single, gamma_or_None, q_power_or_None, info)
        per  coulomb_2026.is_canonical_basis_element."""
        from coulomb_2026 import is_canonical_basis_element
        return is_canonical_basis_element(self, chart, side, n_tuple,
                                          f_partitions,
                                          gamma_search=gamma_search)

    def F_gamma_chart_image(self, chart, gamma):
        """Chart-image of the BPS canonical-basis element  F_γ ."""
        from coulomb_2026 import F_gamma_chart_image
        return F_gamma_chart_image(self, chart, gamma)

    def F_gamma_dictionary(self, chart, populate_seeds=True):
        """Return an FGammaChartDict caching γ → chart_image(F_γ).

        Initially populated with the cluster-seed primitives.  Extends via
        BPS F·F structure constants  (A.multiply)  on demand."""
        from F_gamma_chart_dict import FGammaChartDict
        return FGammaChartDict(self, chart, populate_seeds=populate_seeds)

    def summary(self) -> str:
        chain = " - ".join(
            f"U({Ni})[M={Mi}]" for Ni, Mi in zip(self.N_list, self.M_list)
        )
        return (
            f"LinearUQuiver({chain})\n"
            f"  rank={len(self.B)}  |nodes|={len(self.nodes)}  "
            f"|spec|={len(self.spec)}  "
            f"|bifunds|={sum(len(b) for b in self.bifund_specs)}"
        )

    def __repr__(self) -> str:
        return self.summary()


def sun_nf_fundamental(N: int, Nf: int = 1):
    """Build ``SU(N)`` coupled to ``Nf`` fundamental hypermultiplets
    and return just the ``CoulombAlgebra``.  Thin wrapper around
    :class:`SUN_Nf` -- use the class directly if you want
    ``.pure`` / ``.theory`` / ``.matter_nodes`` access.

    Returns a ``CoulombAlgebra`` on the extended lattice
    ``Gamma = Gamma_sc^{pure SU(N)} (+) Z^{Nf}``  (the Nf extra
    directions are flavour, in ``ker B``).  The spectrum generator is
    assembled as

        S_UV = (prod_j prod_{alpha in F_{w_N}} E_q(X_{mu_j + alpha}))
               * S_{pure SU(N)}

    -- the matter factors run in BFS order of the support of
    ``F_{w_N}`` (= paper order), and  mu_j  is the j-th flavour
    direction.  ``Nf`` extra quiver nodes are added, one per flavour
    copy, at the lowest-weight tropical charge  (0,...,0,-1, 1_j) .

    For  Nf = 0  this reduces to the pure  SU(N)  ``CoulombAlgebra`` .

    Conjectural (verified on SU(2,3) N_f=1):  this equals the
    honest UV  CoulombAlgebra  of  SU(N)  with  Nf  fundamental
    hypers, in particular  F_a * S = X_{gamma_a} + O(q)  on the
    N+1-or-whatever node basis (so the Schur matrix is orthonormal
    at  q^0 ).

    Performance note: building the algebra is cheap (sub-second up to
    N=5); the expensive step is ``A.F(gamma)`` for coroot nodes
    when  N >= 4  (the doubly-tropical interval grows fast with the
    matter factors prepended to ``spec``).
    """
    if N < 2:
        raise ValueError(f"N >= 2 required, got N={N}")
    if Nf < 0:
        raise ValueError(f"Nf >= 0 required, got Nf={Nf}")

    pure = PureADE([("A", N - 1)])
    if Nf == 0:
        return pure.algebra

    B_pure = [list(r) for r in pure.B]
    nodes_pure = [tuple(g) for g in pure.nodes]
    spec_pure = [tuple(g) for g in pure.spec]
    r_pure = len(B_pure)  # = 2(N-1)

    lowest = tuple(-1 if k == r_pure - 1 else 0 for k in range(r_pure))
    alphas = _fundamental_wilson_support_AN(N)

    # Extend by Nf flavour directions.
    r_UV = r_pure + Nf
    B_UV = [row + [0] * Nf for row in B_pure]
    B_UV += [[0] * r_UV for _ in range(Nf)]

    def _ext(v, extra):
        return tuple(list(v) + list(extra))

    def _mu(j):
        return tuple(1 if k == r_pure + j else 0 for k in range(r_UV))

    matter_spec: list[tuple[int, ...]] = []
    matter_nodes: list[tuple[int, ...]] = []
    for j in range(Nf):
        mu_j = _mu(j)
        for a in alphas:
            matter_spec.append(
                tuple(x + y for x, y in zip(_ext(a, [0] * Nf), mu_j))
            )
        matter_nodes.append(
            tuple(x + y for x, y in zip(_ext(lowest, [0] * Nf), mu_j))
        )

    pure_spec_ext = [_ext(g, [0] * Nf) for g in spec_pure]
    nodes_UV = [_ext(g, [0] * Nf) for g in nodes_pure] + matter_nodes
    spec_UV = matter_spec + pure_spec_ext

    # Cone witness: extend PureADE's pure-gauge witness to the UV.  f_pure
    # is >= 1 on every pure node; it evaluates to some integer on the
    # lowest-weight  lowest = (0,...,0,-1)  (for A_n this is -n).  Pick
    # c_j = 1 - f_pure(lowest)  (= 1+n for A_n) so that
    # f_UV(lowest + mu_j) = f_pure(lowest) + c_j = 1.  For every other
    # generator / spec factor  alpha + mu_j  with alpha a non-zero
    # non-negative combination of pure nodes,  f_pure(alpha) >= 1  so
    # f_UV(alpha + mu_j) >= 1 + c_j >= 1.  Pure node charges are unchanged
    # (mu_j-component 0) so f_UV = f_pure there.
    f_pure = list(pure._canonical_cone_witness())
    f_lowest = sum(fi * li for fi, li in zip(f_pure, lowest))
    c = max(1 - f_lowest, 0)
    witness = tuple(f_pure + [c] * Nf)

    return CoulombAlgebra(
        pairing=B_UV, node_charges=nodes_UV, spec=spec_UV,
        cone_witness=witness, skip_spec_cone_check=True,
    )


def un_nf_fundamental(N: int, Nf: int = 1):
    """Same recipe as :func:`sun_nf_fundamental`, but for  U(N)  built
    via :meth:`PureADE.U_N`.  The fundamental of  U(N)  has  N  weights
    (eps_1^*, ..., eps_N^*);  lowest-weight in the dominant chamber is
    eps_N^*.  The Wilson line support has  ``2N - 1``  monomials (same
    count as SU(N)) but a different pattern -- see
    :func:`_fundamental_wilson_support_UN`.
    """
    if N < 1:
        raise ValueError(f"N >= 1 required, got N={N}")
    if Nf < 0:
        raise ValueError(f"Nf >= 0 required, got Nf={Nf}")
    pure = PureADE.U_N(N)
    if Nf == 0:
        return pure.algebra
    B_pure = [list(r) for r in pure.B]
    nodes_pure = [tuple(g) for g in pure.nodes]
    spec_pure = [tuple(g) for g in pure.spec]
    r_pure = len(B_pure)  # = 2N
    lowest = tuple(1 if k == r_pure - 1 else 0 for k in range(r_pure))
    alphas = _fundamental_wilson_support_UN(N)
    r_UV = r_pure + Nf
    B_UV = [row + [0] * Nf for row in B_pure]
    B_UV += [[0] * r_UV for _ in range(Nf)]
    def _ext(v, extra):
        return tuple(list(v) + list(extra))
    def _mu(j):
        return tuple(1 if k == r_pure + j else 0 for k in range(r_UV))
    matter_spec: list[tuple[int, ...]] = []
    matter_nodes: list[tuple[int, ...]] = []
    for j in range(Nf):
        mu_j = _mu(j)
        for a in alphas:
            matter_spec.append(
                tuple(x + y for x, y in zip(_ext(a, [0] * Nf), mu_j))
            )
        matter_nodes.append(
            tuple(x + y for x, y in zip(_ext(lowest, [0] * Nf), mu_j))
        )
    pure_spec_ext = [_ext(g, [0] * Nf) for g in spec_pure]
    nodes_UV = [_ext(g, [0] * Nf) for g in nodes_pure] + matter_nodes
    spec_UV = matter_spec + pure_spec_ext
    f_pure = list(pure._canonical_cone_witness())
    f_lowest = sum(fi * li for fi, li in zip(f_pure, lowest))
    c = max(1 - f_lowest, 0)
    witness = tuple(f_pure + [c] * Nf)
    return CoulombAlgebra(
        pairing=B_UV, node_charges=nodes_UV, spec=spec_UV,
        cone_witness=witness, skip_spec_cone_check=True,
    )


def un_nf_fundamental_raw(N: int, Nf: int = 1):
    """Raw  (B, nodes, spec)  variant of :func:`un_nf_fundamental`."""
    if N < 1:
        raise ValueError(f"N >= 1 required, got N={N}")
    if Nf < 0:
        raise ValueError(f"Nf >= 0 required, got Nf={Nf}")
    pure = PureADE.U_N(N)
    B_pure = [list(r) for r in pure.B]
    nodes_pure = [tuple(g) for g in pure.nodes]
    spec_pure = [tuple(g) for g in pure.spec]
    r_pure = len(B_pure)
    if Nf == 0:
        return B_pure, nodes_pure, spec_pure
    lowest = tuple(1 if k == r_pure - 1 else 0 for k in range(r_pure))
    alphas = _fundamental_wilson_support_UN(N)
    r_UV = r_pure + Nf
    B_UV = [row + [0] * Nf for row in B_pure]
    B_UV += [[0] * r_UV for _ in range(Nf)]
    def _ext(v, extra):
        return tuple(list(v) + list(extra))
    def _mu(j):
        return tuple(1 if k == r_pure + j else 0 for k in range(r_UV))
    matter_spec: list[tuple[int, ...]] = []
    matter_nodes: list[tuple[int, ...]] = []
    for j in range(Nf):
        mu_j = _mu(j)
        for a in alphas:
            matter_spec.append(
                tuple(x + y for x, y in zip(_ext(a, [0] * Nf), mu_j))
            )
        matter_nodes.append(
            tuple(x + y for x, y in zip(_ext(lowest, [0] * Nf), mu_j))
        )
    pure_spec_ext = [_ext(g, [0] * Nf) for g in spec_pure]
    nodes_UV = [_ext(g, [0] * Nf) for g in nodes_pure] + matter_nodes
    spec_UV = matter_spec + pure_spec_ext
    return B_UV, nodes_UV, spec_UV


def sun_nf_fundamental_raw(N: int, Nf: int = 1):
    """Lightweight variant of  ``sun_nf_fundamental``  that returns the
    raw ingredients  ``(pairing, node_charges, spec)``  without
    constructing a ``CoulombAlgebra``.

    Useful when you only need the BPS-quiver / spec data -- e.g. to
    verify the spectrum generator via
    ``BPSQuiver.verify_spectrum_generator`` -- and want to skip the
    (sometimes costly)  CoulombAlgebra  plumbing (positive-cone
    verification etc.).
    """
    if N < 2:
        raise ValueError(f"N >= 2 required, got N={N}")
    if Nf < 0:
        raise ValueError(f"Nf >= 0 required, got Nf={Nf}")
    pure = PureADE([("A", N - 1)])
    B_pure = [list(r) for r in pure.B]
    nodes_pure = [tuple(g) for g in pure.nodes]
    spec_pure = [tuple(g) for g in pure.spec]
    r_pure = len(B_pure)
    if Nf == 0:
        return B_pure, nodes_pure, spec_pure
    lowest = tuple(-1 if k == r_pure - 1 else 0 for k in range(r_pure))
    alphas = _fundamental_wilson_support_AN(N)
    r_UV = r_pure + Nf
    B_UV = [row + [0] * Nf for row in B_pure]
    B_UV += [[0] * r_UV for _ in range(Nf)]
    def _ext(v, extra):
        return tuple(list(v) + list(extra))
    def _mu(j):
        return tuple(1 if k == r_pure + j else 0 for k in range(r_UV))
    matter_spec: list[tuple[int, ...]] = []
    matter_nodes: list[tuple[int, ...]] = []
    for j in range(Nf):
        mu_j = _mu(j)
        for a in alphas:
            matter_spec.append(
                tuple(x + y for x, y in zip(_ext(a, [0] * Nf), mu_j))
            )
        matter_nodes.append(
            tuple(x + y for x, y in zip(_ext(lowest, [0] * Nf), mu_j))
        )
    pure_spec_ext = [_ext(g, [0] * Nf) for g in spec_pure]
    nodes_UV = [_ext(g, [0] * Nf) for g in nodes_pure] + matter_nodes
    spec_UV = matter_spec + pure_spec_ext
    return B_UV, nodes_UV, spec_UV
