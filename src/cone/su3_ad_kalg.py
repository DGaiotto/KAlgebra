"""`SU3ADKAlg` — the SU(3)-flavoured Argyres-Douglas K-algebra: the
[A_1, D_4] AD theory with SU(3) symmetry enhancement (special to D_4
due to triality).

Standalone hard-coded Z_4-symmetric realisation, analogous to
`A1D3KAlg` but with:

  * 8 multiplicative generators {T_0, T_1, T_2, T_3, D_0, D_1, D_2, D_3}
    indexed by Z_4 (= the BPS quiver's half-monodromy ρ orbit) instead
    of Z_3.
  * SU(3) χ_{(p, q)} coefficient ring R(SU(3)) = `SU3ZPlusRing` in
    place of SU(2)'s R(SU(2)).  The Weyl S_3 = Weyl(SU(3)) is generated
    by the cyclic σ_3 (which permutes the three flavour-charged spec
    nodes) and an order-2 reflection that lives in R(SU(3)) (= the
    "complex conjugation" ⋆ : (p, q) ↔ (q, p)).

Companion to a BPS-quiver realisation (`SU3BPSKAlgebra`, from a
derivation not included in this repository): the two are isomorphic on
their canonical bases (a `KAlgebraIso`).  Distinguished from the
`[A_1, D_{2k}]` SU(2)-flavoured family (`u1a1deven_cone_kalgebra` /
`a1deven_kalg`), which carries SU(2) flavour + gauged U(1) for general
D_{2k}; D_4 is exceptional precisely because of triality, lifting the
SU(2) ↪ SU(3) enhancement.

BPS quiver
----------
4-node BPS quiver with one central gauge node (1, 0; 0, 0) and three
flavour nodes at the SU(3) fundamental weights (0, 1; 1, 0),
(0, 1; 0, 1), (0, 1; -1, -1).  Pairing (lattice basis):

    B = [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

ker(B) is rank-2 (the SU(3) Cartan = pure flavour); rank(B) = 2 (one
gauge factor).  Coxeter-like ρ has order 4.

Multiplicative generators
-------------------------
    T_i := ρ^i · L_{(1, 0, 0, 0)}        i ∈ Z_4   (central gauge ρ-orbit)
    D_i := ρ^i · L_{(0, -1, 0, 0)}       i ∈ Z_4   ("negative gauge" ρ-orbit)

Both σ-fixed singletons (= intrinsic χ_{(0, 0)}).  Lattice positions:

    T_0 = (1, 0)        T_1 = (-1, -3)    T_2 = (-2, -3)    T_3 = (-1, 0)
    D_0 = (0, -1)       D_1 = (-1, -2)    D_2 = (-1, -1)    D_3 = (0, 1)

(Only the σ-fixed gauge sublattice (γ_1, γ_2) shown; γ_3 = γ_4 = 0
identically.)  The SU(3) σ-triplet at lattice (0, 1; ±, ±) — the
"obvious" fundamental — sits at the SAME gauge position as D_3 but with
χ_{(1, 0)} on top, distinguishing it as a separate basis element.

q-commute structure
-------------------
Eight tiles (max q-commuting cliques in the letter graph):

    tile 0:  T_0 · D_0   (T·D q-twist q^{-1})
    tile 1:  T_0 · D_3   (q^{+1})
    tile 2:  T_1 · D_1   (q^{-1})
    tile 3:  T_1 · D_0   (q^{+1})
    tile 4:  T_2 · D_2   (q^{-1})
    tile 5:  T_2 · D_1   (q^{+1})
    tile 6:  T_3 · D_3   (q^{-1})
    tile 7:  T_3 · D_2   (q^{+1})

T-T and D-D never q-commute.  T_i · D_j q-commute iff (j - i) mod 4 ∈
{0, 3}.

Defining relations (each line is a single Z_4 ρ-orbit; indices mod 4)
-------------------------------------------------------------------

    T_i · T_{i+1}      =  1  +  q^{-1}·χ_{(0,1)}·D_i  +  q^{-2}·χ_{(1,0)}·D_i^2  +  q^{-3}·D_i^3
    T_{i+1} · T_i      =  1  +  q^{+1}·χ_{(0,1)}·D_i  +  q^{+2}·χ_{(1,0)}·D_i^2  +  q^{+3}·D_i^3

    T_i · T_{i+2}      =  q^{-3}·T_{i+1}  +  q^{-2}·1
                         +  q^{-1}·(χ_{(1,0)}·D_{i+1}  +  χ_{(0,1)}·D_i)
                         +  4 + 2·χ_{(1,1)}
                         +  q^{+1}·(χ_{(0,1)}·D_{i+2}  +  χ_{(1,0)}·D_{i-1})
                         +  q^{+2}·1  +  q^{+3}·T_{i-1}        (palindromic)

    D_i · D_{i+1}      =  1  +  q^{-1}·T_{i+1}
    D_{i+1} · D_i      =  1  +  q^{+1}·T_{i+1}

    D_i · D_{i+2}      =  q^{-1}·D_{i+1}  +  χ_{(0,1)}  +  q^{+1}·D_{i-1}     (palindromic)

    T_i · D_{i+1}      =  q^{-2}·D_i^2  +  q^{-1}·χ_{(1,0)}·D_i  +  χ_{(0,1)}  +  q^{+1}·D_{i-1}
    D_{i+1} · T_i      =  q^{-1}·D_{i-1}  +  χ_{(0,1)}  +  q^{+1}·χ_{(1,0)}·D_i  +  q^{+2}·D_i^2

    T_i · D_{i+2}      =  q^{-1}·D_i  +  χ_{(1,0)}  +  q^{+1}·χ_{(0,1)}·D_{i-1}  +  q^{+2}·D_{i-1}^2
    D_{i+2} · T_i      =  q^{-2}·D_{i-1}^2  +  q^{-1}·χ_{(0,1)}·D_{i-1}  +  χ_{(1,0)}  +  q^{+1}·D_i

These were obtained from the BPS-quiver realisation by pattern matching
(no runtime bootstrap); the standalone class encodes them as static
Python data, parametrised by the ρ-orbit index i and exploiting Z_4
cyclic symmetry + q ↔ q^{-1} bar palindromy.

**ρ-orbit χ-parity.**  ρ = (tile shift) ∘ ⋆ where
⋆ = the rep-ring duality χ_(p,q) ↦ χ_(q,p) (= 3 ↔ 3̄, which the BPS
realisation confirms ρ performs), so the relation at tile i is ρ^i
of the base (i ≡ 0) relation and its χ-labels are ⋆^i-conjugated —
UNCHANGED for even i, SWAPPED for odd i.  The base χ-labels above are
the i ≡ 0 form; `_chi_parity` supplies the per-position ⋆.  (A table
emitting the same χ for every i would break ρ-equivariance of
`multiply` on the 32 odd-position generator pairs; the parity-⋆ tables
are certified product-for-product against the BPS realisation.)

Canonical labels
----------------
    Label = (tile, a, b, p, q)
        tile ∈ {0, ..., 7}     (which q-commuting tile (T_i, D_j))
        a, b ∈ ℕ                (T- and D-exponents in this tile)
        p, q ∈ ℕ                (SU(3) Dynkin labels of χ-coefficient)

M(tile, a, b, p, q) := q^{-a·b·twist} · T_i^a · D_j^b · χ_{(p,q)},
so expanding M into literal letters pulls out a q^{-a·b·twist} factor.

ρ on labels: ρ-shift on tiles 0→2→4→6→0 and 1→3→5→7→1 (= shift both
i_T and i_D by 1 mod 4).

Trace: `trace` / `inner_product` / `trace_word` compute the ρ²-twisted
Schur trace (virtual SU(3) characters) **BPS-free**, via `sl3_su3_traces`.
Any canonical monomial Layer-1-reduces to the three elementary seeds
{Tr_1, Tr_T, Tr_D}; the seed *values* are exact and engine-free —
`Tr_1` is the closed-form Kac–Wakimoto vacuum character of
`\widehat{sl}(3)_{-3/2}` (the [A_1,D_4] VOA; `Tr(1)=1+χ_(1,1)·q²+…`,
the SU(3) flavour current), and `Tr_T`/`Tr_D` come from the
orthonormality bootstrap seeded by `Tr_1`.  All work is carried in
Cartan fugacities (weights, not characters) and Weyl-symmetrized to
SU(3) characters only on the *total*, so genuinely non-self-dual product
content (e.g. `T₀·T₂`'s `3+3̄`) is handled correctly.  Arbitrary q-order;
no BPS / RG engine on the trace path — the BPS realisation is consulted
lazily only as a cross-check oracle (where that derivation is available).
`trace_layer1` (the character-level Layer-1 reduction) is retained for
inspection.
"""

from __future__ import annotations

from typing import Sequence

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element, ElementOverR
from cone_kalgebra import ConeKAlgebra
from zplus_ring import RLaurent
from zplus_ring import (ZPlusRing, RElement, RPowerSeries, SU3ZPlusRing,
                        TrivialZPlusRing)
from tensor_zplus_ring import TensorZPlusRing
from laurent_poly import LaurentPoly


Label = tuple  # (tile, a, b, p, q)

_N = 4  # Z_4 cyclic order


# ---------------------------------------------------------------------------
# q-commute / tile catalog
# ---------------------------------------------------------------------------


def _q_commute_twist(L1, L2):
    """q-twist e such that L1·L2 = q^e · L_{lattice sum}, or None if
    (L1, L2) is non-q-commuting.

    T_i · D_i: returns -1.    T_i · D_{i-1 mod 4}: returns +1.
    Reversed orders flip sign.  T-T, D-D, T_i · D_{j ∈ {i+1, i+2}}: None.
    """
    (k1, i1), (k2, i2) = L1, L2
    if L1 == L2:
        return 0
    if k1 == k2:
        return None  # T-T or D-D never q-commute
    if k1 == 'T' and k2 == 'D':
        t_i, d_j, sign = i1, i2, +1
    else:
        t_i, d_j, sign = i2, i1, -1
    diff = (d_j - t_i) % _N
    if diff == 0:
        return -sign           # T_i · D_i = q^{-1} · lattice
    if diff == _N - 1:
        return +sign           # T_i · D_{i-1} = q^{+1} · lattice
    return None                # diff ∈ {1, 2}: interaction


_TILE_LETTERS = {
    0: (('T', 0), ('D', 0)),
    1: (('T', 0), ('D', 3)),
    2: (('T', 1), ('D', 1)),
    3: (('T', 1), ('D', 0)),
    4: (('T', 2), ('D', 2)),
    5: (('T', 2), ('D', 1)),
    6: (('T', 3), ('D', 3)),
    7: (('T', 3), ('D', 2)),
}
_TILE_TWIST = {t: _q_commute_twist(L_T, L_D)
               for t, (L_T, L_D) in _TILE_LETTERS.items()}


def _tile_for_pair(L_T, L_D):
    for t, pair in _TILE_LETTERS.items():
        if pair == (L_T, L_D):
            return t
    return None


def _tile_for_T_letter(L_T):
    (_, i_T) = L_T
    return {0: 0, 1: 2, 2: 4, 3: 6}[i_T]


def _tile_for_D_letter(L_D):
    (_, i_D) = L_D
    return {0: 0, 1: 2, 2: 4, 3: 6}[i_D]


_TILE_RHO = {}
for tile, (L_T, L_D) in _TILE_LETTERS.items():
    (_, i_T), (_, i_D) = L_T, L_D
    new_pair = (('T', (i_T + 1) % _N), ('D', (i_D + 1) % _N))
    _TILE_RHO[tile] = _tile_for_pair(*new_pair)
_TILE_RHO_INV = {v: k for k, v in _TILE_RHO.items()}


def _rho_n_letter(L, n):
    return (L[0], (L[1] + n) % _N)


# ---------------------------------------------------------------------------
# Canonical form: letters_dict ↔ (tile, a, b, p, q)
# ---------------------------------------------------------------------------


def _monomial_to_label(letters, pq, q_factor):
    letters = {L: e for L, e in letters.items() if e > 0}
    t_entries = [(L, e) for L, e in letters.items() if L[0] == 'T']
    d_entries = [(L, e) for L, e in letters.items() if L[0] == 'D']
    if len(t_entries) > 1 or len(d_entries) > 1:
        raise ValueError(f"monomial has multiple T or D letters: {letters}")
    if not t_entries and not d_entries:
        return (0, 0, 0, pq[0], pq[1]), LaurentPoly({q_factor: 1})
    if t_entries and not d_entries:
        L_T, a = t_entries[0]
        return ((_tile_for_T_letter(L_T), a, 0, pq[0], pq[1]),
                LaurentPoly({q_factor: 1}))
    if d_entries and not t_entries:
        L_D, b = d_entries[0]
        return ((_tile_for_D_letter(L_D), 0, b, pq[0], pq[1]),
                LaurentPoly({q_factor: 1}))
    L_T, a = t_entries[0]
    L_D, b = d_entries[0]
    tile = _tile_for_pair(L_T, L_D)
    if tile is None:
        raise ValueError(
            f"{L_T}, {L_D} non-q-commuting; cannot form single-tile monomial"
        )
    twist = _TILE_TWIST[tile]
    q_factor += a * b * twist
    return ((tile, a, b, pq[0], pq[1]), LaurentPoly({q_factor: 1}))


def _label_to_monomial(label):
    tile, a, b, p, q = label
    L_T, L_D = _TILE_LETTERS[tile]
    letters = {}
    if a > 0:
        letters[L_T] = a
    if b > 0:
        letters[L_D] = b
    twist = _TILE_TWIST[tile]
    q_factor = -a * b * twist
    return letters, (p, q), q_factor


# ---------------------------------------------------------------------------
# Interaction relations (hardcoded, pattern-matched, ρ-equivariant)
# ---------------------------------------------------------------------------


def _T(i):
    return ('T', i % _N)


def _D(i):
    return ('D', i % _N)


def _chi_parity(terms, base_i):
    """Apply the χ-parity of the ρ-orbit position `base_i` to a base
    (`i ≡ 0`) relation `terms` list.

    ρ = (tile shift by 1) ∘ (⋆ on the SU(3) centre), where ⋆ is the
    rep-ring duality `χ_(p,q) ↦ χ_(q,p)` (= complex conjugation 3 ↔ 3̄;
    `_SU3_RING.star_basis`).  The relation at tile `i` is therefore
    `ρ^i` of the base relation, so its χ-labels are `⋆^i`-conjugated:
    UNCHANGED for even `i`, swapped `(p,q) ↦ (q,p)` for odd `i`.  The
    `_tt_*` / `_dd_*` / `_td_*` families below encode the base (`i ≡ 0`)
    χ-labels and the correct `i`-shifted letter words; this wrapper
    supplies the missing parity-⋆ that makes ρ a genuine algebra
    automorphism (verified against the BPS realisation; a static table
    emitting the *same* χ for every `i` would break ρ-equivariance of
    multiply).
    """
    if base_i % 2 == 0:
        return terms
    return [(q_delta, letters, (pq[1], pq[0]), coef)
            for (q_delta, letters, pq, coef) in terms]


def _interaction(L1, L2):
    """For non-q-commuting (L1, L2), return list of
    (q_delta, letters_dict, chi_(p,q), coef) terms encoding
    L1 · L2 = Σ q^{q_delta} · coef · letters_dict · χ_{(p,q)}.

    Eight ρ-orbit families, hardcoded at the base (`i ≡ 0`) χ-labels and
    closed under Z_4 ρ-equivariance by `_chi_parity` (the per-orbit-
    position parity-⋆).  No runtime BPS bootstrap.
    """
    (k1, i1), (k2, i2) = L1, L2
    if k1 == 'T' and k2 == 'T':
        diff = (i2 - i1) % _N
        if diff == 1:
            return _chi_parity(_tt_dist1_fwd(i1), i1)
        if diff == _N - 1:           # = 3 mod 4
            return _chi_parity(_tt_dist1_bwd(i2), i2)
        if diff == 2:
            return _chi_parity(_tt_dist2(i1), i1)
    if k1 == 'D' and k2 == 'D':
        diff = (i2 - i1) % _N
        if diff == 1:
            return _chi_parity(_dd_dist1_fwd(i1), i1)
        if diff == _N - 1:
            return _chi_parity(_dd_dist1_bwd(i2), i2)
        if diff == 2:
            return _chi_parity(_dd_dist2(i1), i1)
    if k1 == 'T' and k2 == 'D':
        diff = (i2 - i1) % _N
        if diff == 1:
            return _chi_parity(_td_dist1_fwd(i1), i1)
        if diff == 2:
            return _chi_parity(_td_dist2_fwd(i1), i1)
    if k1 == 'D' and k2 == 'T':
        # (k1, i1) = ('D', j); (k2, i2) = ('T', i).
        j_D, i_T = i1, i2
        diff = (j_D - i_T) % _N    # j - i
        if diff == 1:
            return _chi_parity(_td_dist1_bwd(i_T), i_T)   # D_{i+1} · T_i
        if diff == 2:
            return _chi_parity(_td_dist2_bwd(i_T), i_T)   # D_{i+2} · T_i
    return None


def _tt_dist1_fwd(i):
    """T_i · T_{i+1} = 1 + q^{-1}·χ_(0,1)·D_i + q^{-2}·χ_(1,0)·D_i² + q^{-3}·D_i³."""
    return [
        (0, {}, (0, 0), 1),
        (-1, {_D(i): 1}, (0, 1), 1),
        (-2, {_D(i): 2}, (1, 0), 1),
        (-3, {_D(i): 3}, (0, 0), 1),
    ]


def _tt_dist1_bwd(i):
    """T_{i+1} · T_i (= reverse of fwd; q → q^{-1})."""
    return [
        (0, {}, (0, 0), 1),
        (+1, {_D(i): 1}, (0, 1), 1),
        (+2, {_D(i): 2}, (1, 0), 1),
        (+3, {_D(i): 3}, (0, 0), 1),
    ]


def _tt_dist2(i):
    """T_i · T_{i+2} (palindromic / self-reverse).

        = q^{-3}·T_{i+1} + q^{-2}·1
        + q^{-1}·(χ_(1,0)·D_{i+1} + χ_(0,1)·D_i)
        + 2 + χ_(1,1)
        + q^{+1}·(χ_(0,1)·D_{i+2} + χ_(1,0)·D_{i-1})
        + q^{+2}·1 + q^{+3}·T_{i-1}.

    The gauge-neutral term is `2 + χ_(1,1)`, NOT `4 + 2·χ_(1,1)`: the genuine
    content is the six adjoint root-weights + two singlets, and the six roots
    are `χ_(1,1) − 2` as a *virtual* SU(3) character (the adjoint minus its two
    zero-weights), so `(χ_(1,1)−2) + 2·𝟙 = χ_(1,1) + 2·𝟙`.  The old `4+2χ`
    double-counted the adjoint by `χ_(1,1)+2` — invisible to the character-
    level iso cert (its `to_R_form` fold double-counts identically) but exposed
    by the ρ²-twisted trace / the U(1)² reference.
    """
    return [
        (-3, {_T(i + 1): 1}, (0, 0), 1),
        (-2, {}, (0, 0), 1),
        (-1, {_D(i + 1): 1}, (1, 0), 1),
        (-1, {_D(i): 1}, (0, 1), 1),
        (0, {}, (0, 0), 2),
        (0, {}, (1, 1), 1),
        (+1, {_D(i + 2): 1}, (0, 1), 1),
        (+1, {_D(i - 1): 1}, (1, 0), 1),
        (+2, {}, (0, 0), 1),
        (+3, {_T(i - 1): 1}, (0, 0), 1),
    ]


def _dd_dist1_fwd(i):
    """D_i · D_{i+1} = 1 + q^{-1}·T_{i+1}."""
    return [
        (0, {}, (0, 0), 1),
        (-1, {_T(i + 1): 1}, (0, 0), 1),
    ]


def _dd_dist1_bwd(i):
    """D_{i+1} · D_i = 1 + q^{+1}·T_{i+1}."""
    return [
        (0, {}, (0, 0), 1),
        (+1, {_T(i + 1): 1}, (0, 0), 1),
    ]


def _dd_dist2(i):
    """D_i · D_{i+2} = q^{-1}·D_{i+1} + χ_(0,1) + q^{+1}·D_{i-1}."""
    return [
        (-1, {_D(i + 1): 1}, (0, 0), 1),
        (0, {}, (0, 1), 1),
        (+1, {_D(i - 1): 1}, (0, 0), 1),
    ]


def _td_dist1_fwd(i):
    """T_i · D_{i+1} = q^{-2}·D_i² + q^{-1}·χ_(1,0)·D_i + χ_(0,1) + q·D_{i-1}."""
    return [
        (-2, {_D(i): 2}, (0, 0), 1),
        (-1, {_D(i): 1}, (1, 0), 1),
        (0, {}, (0, 1), 1),
        (+1, {_D(i - 1): 1}, (0, 0), 1),
    ]


def _td_dist1_bwd(i):
    """D_{i+1} · T_i = q^{-1}·D_{i-1} + χ_(0,1) + q·χ_(1,0)·D_i + q²·D_i²."""
    return [
        (-1, {_D(i - 1): 1}, (0, 0), 1),
        (0, {}, (0, 1), 1),
        (+1, {_D(i): 1}, (1, 0), 1),
        (+2, {_D(i): 2}, (0, 0), 1),
    ]


def _td_dist2_fwd(i):
    """T_i · D_{i+2} = q^{-1}·D_i + χ_(1,0) + q·χ_(0,1)·D_{i-1} + q²·D_{i-1}²."""
    return [
        (-1, {_D(i): 1}, (0, 0), 1),
        (0, {}, (1, 0), 1),
        (+1, {_D(i - 1): 1}, (0, 1), 1),
        (+2, {_D(i - 1): 2}, (0, 0), 1),
    ]


def _td_dist2_bwd(i):
    """D_{i+2} · T_i = q^{-2}·D_{i-1}² + q^{-1}·χ_(0,1)·D_{i-1} + χ_(1,0) + q·D_i."""
    return [
        (-2, {_D(i - 1): 2}, (0, 0), 1),
        (-1, {_D(i - 1): 1}, (0, 1), 1),
        (0, {}, (1, 0), 1),
        (+1, {_D(i): 1}, (0, 0), 1),
    ]


# ---------------------------------------------------------------------------
# Multiplication: letter-sequence reduction (A1D3-style)
# ---------------------------------------------------------------------------


_SU3_RING = SU3ZPlusRing()


def _su3_cg(pq1, pq2):
    return _SU3_RING.multiply_basis(pq1, pq2)


def _reduce_letter_seq(letter_seq, pq, q_factor, depth=0):
    """Reduce [L_1, …, L_n] · χ_{(p,q)} · q^{q_factor} to {Label: LaurentPoly}.

    Same algorithm as `A1D3KAlg._reduce_letter_seq` with SU(3) Clebsch-
    Gordan via `_su3_cg`.
    """
    if depth > 80:
        raise RecursionError(
            f"_reduce_letter_seq exceeded depth {depth}; "
            f"seq = {letter_seq}, pq = {pq}"
        )
    seq = [L for L in letter_seq if L is not None]
    n = len(seq)

    distinct = list(set(seq))
    all_commute = True
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            if _q_commute_twist(distinct[i], distinct[j]) is None:
                all_commute = False
                break
        if not all_commute:
            break

    first_bad = None
    if not all_commute:
        for idx in range(n - 1):
            if _q_commute_twist(seq[idx], seq[idx + 1]) is None:
                first_bad = idx
                break
        if first_bad is None:
            # Bubble offending letters adjacent.
            target_i = target_j = None
            for i_idx in range(n):
                for j_idx in range(i_idx + 1, n):
                    if _q_commute_twist(seq[i_idx], seq[j_idx]) is None:
                        target_i, target_j = i_idx, j_idx
                        break
                if target_i is not None:
                    break
            new_seq = list(seq)
            q_pre = 0
            pos = target_j
            while pos > target_i + 1:
                L_a, L_b = new_seq[pos - 1], new_seq[pos]
                t = _q_commute_twist(L_a, L_b)
                if t is None:
                    break
                q_pre += 2 * t
                new_seq[pos - 1], new_seq[pos] = L_b, L_a
                pos -= 1
            return _reduce_letter_seq(
                new_seq, pq, q_factor + q_pre, depth + 1,
            )

    if first_bad is None:
        sequence = list(seq)
        target = (sorted([L for L in sequence if L[0] == 'T']) +
                  sorted([L for L in sequence if L[0] == 'D']))
        q_delta = 0
        for tp in range(len(target)):
            tgt_L = target[tp]
            sp = tp
            while sp < len(sequence) and sequence[sp] != tgt_L:
                sp += 1
            while sp > tp:
                L_a, L_b = sequence[sp - 1], sequence[sp]
                t = _q_commute_twist(L_a, L_b)
                q_delta += 2 * t
                sequence[sp - 1], sequence[sp] = L_b, L_a
                sp -= 1
        out_letters = {}
        for L in sequence:
            out_letters[L] = out_letters.get(L, 0) + 1
        new_label, new_lp = _monomial_to_label(
            out_letters, pq, q_factor + q_delta,
        )
        return {new_label: new_lp}

    # Apply interaction at first_bad.
    L1, L2 = seq[first_bad], seq[first_bad + 1]
    left = seq[:first_bad]
    right = seq[first_bad + 2:]
    interactions = _interaction(L1, L2)
    if interactions is None:
        raise RuntimeError(
            f"no interaction for ({L1}, {L2}) — should be q-commute"
        )
    acc = {}
    for q_delta, new_letters_dict, chi_pq, mult in interactions:
        mid = []
        for L, e in new_letters_dict.items():
            mid += [L] * e
        for new_pq, cg_mult in _su3_cg(pq, chi_pq).items():
            total = mult * cg_mult
            if total == 0:
                continue
            sub = _reduce_letter_seq(
                left + mid + right, new_pq,
                q_factor + q_delta, depth + 1,
            )
            for lab, lp in sub.items():
                acc[lab] = acc.get(lab, LaurentPoly({})) + lp * total
    return {l: lp for l, lp in acc.items() if not lp.is_zero()}


# ---------------------------------------------------------------------------
# Layer-1 trace reduction (tag-move-cycle-Plücker, A1D3-style)
# ---------------------------------------------------------------------------


def _base_qdict(q_lp, chi_re):
    out = {}
    for q_exp, q_c in q_lp._coeffs.items():
        if q_c == 0:
            continue
        contrib = chi_re * q_c
        if not contrib.is_zero():
            out[q_exp] = contrib
    return out


def _merge(target, contrib):
    for q_exp, r in contrib.items():
        if r.is_zero():
            continue
        if q_exp in target:
            new_r = target[q_exp] + r
            if new_r.is_zero():
                del target[q_exp]
            else:
                target[q_exp] = new_r
        else:
            target[q_exp] = r


def _trace_reduce_word(alg, word, chi_pq, q_factor, depth=0, max_depth=80):
    """Reduce Tr(word · χ · q^{q_factor}) via ρ²-twisted cyclicity +
    Plücker interactions."""
    if depth > max_depth:
        return {('Tr_max_depth', tuple(word), chi_pq):
                _base_qdict(LaurentPoly({q_factor: 1}),
                            alg._R.basis_element(chi_pq))}
    if not word:
        return {('Tr_1',): _base_qdict(
            LaurentPoly({q_factor: 1}), alg._R.basis_element(chi_pq))}
    if len(word) == 1:
        L = word[0]
        if L[0] == 'T':
            key = ('Tr_T',)                  # χ-self-dual ⇒ parity-independent
        else:
            # ρ inverts flavour fugacities, so Tr(D_odd) = ⋆Tr(D_even) (the
            # ⋆-dual character).  The D-letter index parity (D₀,D₂ even;
            # D₁,D₃ odd) tags which: Layer 2 plugs Tr_D for parity 0 and
            # ⋆Tr_D for parity 1.  ρ² shifts the index by 2, preserving
            # parity (consistent with ρ²-invariance of the trace).
            key = ('Tr_D', L[1] % 2)
        return {key: _base_qdict(
            LaurentPoly({q_factor: 1}), alg._R.basis_element(chi_pq))}
    n = len(word)
    cur = list(word)
    qa = q_factor
    for _cycle in range(_N):
        L_last = cur[-1]
        cur = [_rho_n_letter(L_last, 2)] + cur[:-1]
        pos = 0
        while pos < n - 1:
            L_tag, L_next = cur[pos], cur[pos + 1]
            t = _q_commute_twist(L_tag, L_next)
            if t is None:
                left = cur[:pos]; right = cur[pos + 2:]
                inters = _interaction(L_tag, L_next)
                result = {}
                for q_delta, ld, chi_inter, mult in inters:
                    mid = []
                    for L, e in ld.items():
                        mid += [L] * e
                    new_word = left + mid + right
                    for new_pq, cg in _su3_cg(chi_pq, chi_inter).items():
                        total = mult * cg
                        if total == 0:
                            continue
                        sub = _trace_reduce_word(
                            alg, new_word, new_pq, qa + q_delta,
                            depth + 1, max_depth,
                        )
                        for key, qdict in sub.items():
                            scaled = {qe: r * total for qe, r in qdict.items()
                                      if not (r * total).is_zero()}
                            bucket = result.setdefault(key, {})
                            _merge(bucket, scaled)
                            if not bucket:
                                del result[key]
                return result
            qa += 2 * t
            cur[pos], cur[pos + 1] = L_next, L_tag
            pos += 1
    return {('Tr_irreducible', tuple(cur), chi_pq):
            _base_qdict(LaurentPoly({qa: 1}),
                        alg._R.basis_element(chi_pq))}


# ---------------------------------------------------------------------------
# SU3ADKAlg
# ---------------------------------------------------------------------------


class SU3ADKAlg(ConeKAlgebra):
    """[A_1, D_4] AD K-algebra with SU(3) symmetry enhancement,
    standalone Z_4-symmetric, with χ_{(p,q)} ∈ R(SU(3)) coefficients.

    A `ConeKAlgebra` that is **fully free over `R = R(SU(3))`** (trivial
    `R_lab`; SU(3) characters fuse, never invert).  Canonical labels are
    5-tuples `(tile, a, b, p, q)`, related to the freeness
    `(free_char, section)` split by `free_char = (p, q)` (the SU(3)
    weight) and `section = (tile, a, b)`, via `_pack_label` /
    `_unpack_label`.

    * **Freeness fusion** — `multiply` (Z-valued), `rho`, `identity`,
      `section_decompose`, `embed_R` are derived from the section engine
      below.  The character-fusion logic is implemented in place
      (fully-free / trivial-`R_lab` case) rather than inherited from a
      shared base class, keeping this class self-contained.
    * `ConeKAlgebra` keeps the cone-data presentation (`cone_data` →
      `SU3ADConeData`) — which the section engine `section_multiply`
      drives — and the `isinstance(·, ConeKAlgebra)` identity used by
      the cone registry.

    The section engine is the cone-data product on `(p,q)`-stripped
    labels; the fusion wrapper fuses the input `χ_{(p,q)}` characters
    (SU(3) Clebsch–Gordan) and re-expands to 5-tuples.  ρ on the centre
    is the rep-ring `⋆` `(p,q) ↔ (q,p)`; the section ρ permutes tiles.

    Layer-1 / Layer-2 trace pipeline (`trace_layer1`, `trace`) is
    preserved verbatim — it has its own tag-move-cycle-Plücker
    algorithm independent of the cone-data Layer-1 reducer.
    """

    _RANK = 4
    _GAUGE_RANK = 2     # = len(BPS sec_basis); rank(B) = 2 (one gauge pair)

    def __init__(self):
        self._R = _SU3_RING

    @property
    def rank(self) -> int:
        return self._RANK

    @property
    def gauge_rank(self) -> int:
        return self._GAUGE_RANK

    # -- generators -------------------------------------------------------

    def T(self, i: int) -> Label:
        return ({0: 0, 1: 2, 2: 4, 3: 6}[i % _N], 1, 0, 0, 0)

    def D(self, i: int) -> Label:
        return ({0: 0, 1: 2, 2: 4, 3: 6}[i % _N], 0, 1, 0, 0)

    def chi(self, p: int, q: int) -> Label:
        if p < 0 or q < 0:
            raise ValueError(f"chi(p, q): need p, q ≥ 0; got ({p}, {q})")
        return (0, 0, 0, p, q)

    # -- section engine (the freeness structure) -------------------------
    #
    # Fully free over R_free = R(SU(3)) (trivial R_lab).  Abstract
    # `(free_char, section)` ↔ 5-tuple label via _pack/_unpack:
    #   free_char = (p, q)   (SU(3) weight),  section = (tile, a, b).
    # The copied fusion below derives multiply / rho / identity /
    # section_decompose / embed_R from these.

    def free_ring(self) -> ZPlusRing:
        return self._R

    def section_identity(self):
        return (0, 0, 0)

    def _unpack_label(self, label):
        tile, a, b, p, q = self.canonicalise(label)
        return (p, q), (tile, a, b)

    def _pack_label(self, free_char, section):
        p, q = free_char
        tile, a, b = section
        return self.canonicalise((tile, a, b, p, q))

    def section_multiply(self, s1, s2) -> ElementOverR:
        """The R-form section product `M_{s1}·M_{s2}`: the cone-data
        product on `(p,q)`-stripped 5-tuples, reinterpreted as an
        `ElementOverR` keyed by `(tile, a, b)` sections (RLaurent[SU(3)]
        coefficients)."""
        t1, a1, b1 = s1
        t2, a2, b2 = s2
        res = self.cone_data().derived_multiply(
            (t1, a1, b1, 0, 0), (t2, a2, b2, 0, 0),
        )
        R = self.coefficient_ring()
        out: dict = {}
        for lab5, coef in res.terms.items():
            rl = (RLaurent(R, dict(coef._coeffs))
                  if isinstance(coef, LaurentPoly) else coef)
            sec = (lab5[0], lab5[1], lab5[2])
            out[sec] = rl if sec not in out else out[sec] + rl
        return ElementOverR(R, out)

    def section_rho(self, s):
        tile, a, b = s
        return (_TILE_RHO[tile], a, b)

    def section_rho_inverse(self, s):
        tile, a, b = s
        return (_TILE_RHO_INV[tile], a, b)

    # ---- KAlgebra contract: freeness fusion (copied from RKAlgebra) ----
    # Fully-free case (R_lab trivial); the custom 5-tuple ↔
    # (free_char, section) maps `_pack_label`/`_unpack_label` live above.

    def label_ring(self) -> ZPlusRing:
        return TrivialZPlusRing()

    def _label_is_trivial(self) -> bool:
        return isinstance(self.label_ring(), TrivialZPlusRing)

    def coefficient_ring(self):
        cached = getattr(self, "_rk_coeff_ring", None)
        if cached is not None:
            return cached
        R = (self.free_ring() if self._label_is_trivial()
             else TensorZPlusRing(self.free_ring(), self.label_ring()))
        self._rk_coeff_ring = R
        return R

    def _split(self, full_basis):
        if self._label_is_trivial():
            return full_basis, None
        free_b, lab_b = full_basis
        return free_b, lab_b

    def _lift_free_elem(self, e: RElement) -> RElement:
        if self._label_is_trivial():
            return e
        R = self.coefficient_ring()
        lab_one = self.label_ring().one_basis()
        return RElement(R, {(fb, lab_one): n for fb, n in e.terms.items()})

    def _section_of(self, label_basis):
        if label_basis is None:
            return self.section_identity()
        return self.shift_section(self.section_identity(), label_basis)

    def identity(self):
        return self._pack_label(self.free_ring().one_basis(),
                                self.section_identity())

    def multiply(self, a, b) -> Element:
        w1, s1 = self._unpack_label(a)
        w2, s2 = self._unpack_label(b)
        Rf = self.free_ring()
        section = self.section_multiply(s1, s2)            # ElementOverR
        cw = self._lift_free_elem(
            Rf.basis_element(w1) * Rf.basis_element(w2)
        )                                                   # RElement over full R
        out: dict = {}
        for u, rl in section.terms.items():                 # rl : RLaurent
            for qpow, relem in rl.coeffs.items():           # relem : RElement
                fused = cw * relem                          # RElement over full R
                for full_b, n in fused.terms.items():
                    if n == 0:
                        continue
                    fb, lb = self._split(full_b)
                    u2 = u if lb is None else self.shift_section(u, lb)
                    lab = self._pack_label(fb, u2)
                    term = LaurentPoly({qpow: n})
                    out[lab] = term if lab not in out else out[lab] + term
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    def rho(self, label):
        w, s = self._unpack_label(label)
        return self._pack_label(self.free_ring().star_basis(w),
                                self.section_rho(s))

    def rho_inverse(self, label):
        w, s = self._unpack_label(label)
        return self._pack_label(self.free_ring().star_basis(w),
                                self.section_rho_inverse(s))

    def r_label_decompose(self, label):
        """The single-irrep flavour-lift coordinate: peel the SU(3)
        flavour character `χ_{(p,q)}` off the gauge section, preserving
        the free/label split of the coefficient ring.  `r_label_compose` is inherited (the central
        `embed_R` rebuilds the canonical, since the section is
        flavour-trivial)."""
        w, s = self._unpack_label(label)
        free_one = self.free_ring().one_basis()
        r_coeff = self._lift_free_elem(self.free_ring().basis_element(w))
        # r_coeff is a single basis element (irrep); read its key off.
        (r_basis_label, _n), = r_coeff.terms.items()
        return self._pack_label(free_one, s), r_basis_label

    def embed_R(self, r: RElement) -> Element:
        R = self.coefficient_ring()
        if not isinstance(r, RElement) or r.ring != R:
            raise TypeError(
                "embed_R: argument must be an RElement over coefficient_ring()"
            )
        out: dict = {}
        for full_b, n in r.terms.items():
            if n == 0:
                continue
            fb, lb = self._split(full_b)
            lab = self._pack_label(fb, self._section_of(lb))
            term = LaurentPoly({0: n})
            out[lab] = term if lab not in out else out[lab] + term
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    # -- other primitives / helpers --------------------------------------

    def canonicalise(self, x) -> Label:
        x = tuple(int(c) for c in x)
        if len(x) != 5:
            raise ValueError(f"label must be 5-tuple, got {x}")
        tile, a, b, p, q = x
        if not (0 <= tile <= 7 and a >= 0 and b >= 0 and p >= 0 and q >= 0):
            raise ValueError(f"invalid label {x}")
        L_T, L_D = _TILE_LETTERS[tile]
        if a == 0 and b == 0:
            return (0, 0, 0, p, q)
        if a == 0:
            return (_tile_for_D_letter(L_D), 0, b, p, q)
        if b == 0:
            return (_tile_for_T_letter(L_T), a, 0, p, q)
        return x

    def cone_data(self):
        """Cone-data presentation on (p,q)-stripped 5-tuple labels.
        Lazy-built."""
        if not hasattr(self, "_cone_data_cache"):
            from su3_ad_cone_data import SU3ADConeData
            self._cone_data_cache = SU3ADConeData()
        return self._cone_data_cache

    def _canonical_rho2_orbit_rep(self, label):
        """ρ² has order 4 on tiles; default orbit walk."""
        return KAlgebra._canonical_rho2_orbit_rep(self, label)

    def _trace_residual(self, seed_label, K):
        """Required abstract on ConeKAlgebra.  Unused: SU3AD's public
        `trace` uses its own `trace_layer1` pipeline (tag-move-cycle-
        Plücker on letters); cone-data trace seeds are not routed here."""
        raise NotImplementedError(
            "SU3ADKAlg._trace_residual: SU3AD uses its own trace_layer1 "
            "pipeline; cone-data trace seeds are not routed here."
        )

    # -- Layer 1 trace ---------------------------------------------------

    def trace_layer1(self, label: Label) -> dict:
        tile, a_e, b_e, p, q = self.canonicalise(label)
        if a_e == 0 and b_e == 0:
            return {('Tr_1',): {0: self._R.basis_element((p, q))}}
        letters, chi_pq, q_factor = _label_to_monomial(label)
        word = []
        for L in sorted([L for L in letters if L[0] == 'T']):
            word += [L] * letters[L]
        for L in sorted([L for L in letters if L[0] == 'D']):
            word += [L] * letters[L]
        return _trace_reduce_word(self, word, chi_pq, q_factor)

    # -- cross-check trace via the BPS oracle -----------------------------
    #
    # The standalone ↔ BPS-realisation correspondence (gauge charge =
    # the (tile, a, b) letter-monomial position; SU(3) character carried
    # in the flavour direction) is certified product-for-product against
    # the BPS engine (exact coefficients on all 64 generator pairs).
    # The `_bps_*` helpers below transport a canonical label to the BPS
    # chart and read off its analytic trace there; they serve only as a
    # cross-check oracle (the derivation module is not included in this
    # repository) — the production `trace` is the BPS-free
    # `sl3_su3_traces` route.  `trace_layer1` (the algebraic Layer-1
    # reduction) is retained for the elementary-trace
    # tag-move-cycle-Plücker pipeline.

    _TPOS = {0: (1, 0), 1: (-1, -3), 2: (-2, -3), 3: (-1, 0)}
    _DPOS = {0: (0, -1), 1: (-1, -2), 2: (-1, -1), 3: (0, 1)}

    def _bps_engine(self):
        """Memoised companion BPS-quiver realisation (the analytic-trace
        cross-check oracle; its module `su3_bps_kalgebra` is not included
        in this repository).  Built lazily — `multiply`/`rho` and the
        production trace path never touch it."""
        if not hasattr(self, "_bps_eng"):
            from su3_bps_kalgebra import _user_quiver
            self._bps_eng = _user_quiver()
        return self._bps_eng

    def _std_to_bps_label(self, label):
        """Transport a standalone canonical label `(tile, a, b, p, q)` to
        its BPS-oracle canonical label: gauge charge from the
        letter-monomial position, SU(3) character `χ_(p,q)` carried by
        its highest weight `(-(p+q), -p)` in the BPS flavour basis (the
        Weyl orbit rep — the trace is Weyl-invariant, so any orbit
        representative gives the same value)."""
        tile, a, b, p, q = self.canonicalise(label)
        (_, i_T), (_, i_D) = _TILE_LETTERS[tile]
        g0 = a * self._TPOS[i_T][0] + b * self._DPOS[i_D][0]
        g1 = a * self._TPOS[i_T][1] + b * self._DPOS[i_D][1]
        B = self._bps_engine()
        return B.canonicalise((g0, g1, -(p + q), -p))

    def _bps_trace_to_K(self, bps_label, K, **kwargs):
        """BPS analytic trace of `bps_label`, returned over `R(SU(3))`
        truncated to `q^K`.

        Flavour-charged elements draw on `S_RG` levels *above* K (the
        flavoured-trace "trapezoid": the χ-content near the window top is
        clipped and momentarily non-Weyl-symmetric), so we compute with a
        growing internal margin and truncate back.  If even a generous
        margin leaves the q^≤K content clipped, we raise (rather
        than return a silently wrong / non-symmetric character) — that
        regime is what the exact closed-form Layer-2 (the SU(3)_{−3/2}
        chiral characters) is for."""
        B = self._bps_engine()
        last_err = None
        for margin in (0, 4):
            try:
                full = B.trace(bps_label, K=K + margin, **kwargs)
                return RPowerSeries(
                    self._R,
                    {q: r for q, r in full.coeffs.items() if q <= K},
                    K,
                )
            except ValueError as e:   # non-S_3-symmetric clipped top
                last_err = e
        raise NotImplementedError(
            f"SU3ADKAlg.trace: the analytic Schur trace of {bps_label} clips "
            f"non-symmetrically at q^≤{K} even with margin (BPS-window "
            f"trapezoid: {last_err}).  The exact closed-form Layer-2 for this "
            f"flavour-charged element needs the SU(3)_{{-3/2}} chiral "
            f"characters."
        )

    def trace(self, a: Label, K: int = 20, **kwargs) -> RPowerSeries:
        """ρ²-twisted Schur trace `Tr(L_a)` over `R(SU(3))((q))`, **BPS-free**.

        Layer-1 reduces the (flavour-stripped) gauge monomial to the three
        elementary seeds {Tr_1, Tr_T, Tr_D} in Cartan fugacities; the exact,
        engine-free seed values are the closed-form Kac–Wakimoto vacuum
        character (`Tr_1`) and the orthonormality bootstrap (`Tr_T`, `Tr_D`).
        The flavour `χ_(p,q)` is a genuine spectator over `R(SU(3))`, so it is
        multiplied back (R(SU(3))-linearity), Weyl-symmetrizing the total to
        SU(3) characters only at the end.  Arbitrary q-order; see
        `sl3_su3_traces`.  `kwargs` are accepted for back-compat and ignored
        (no BPS cutoff on the engine-free path)."""
        from sl3_su3_traces import trace as _trace
        return _trace(self, a, K)

    def trace_word(self, factors, K: int = 8, margin: int = 4) -> RPowerSeries:
        """Axiom-correct `Tr(L_{f₁}·…·L_{fₙ})` for a word of canonical
        generators (each `factors[i]` a standalone label), **BPS-free**.

        The ρ²-twisted trace of a *product* must not symmetrize the
        intermediate non-self-dual flavour content.  We therefore fugacity-
        multiply the whole product (`sl3_su3_traces.fug_multiply`: weights,
        not characters), single-label-trace each resulting canonical monomial,
        and Weyl-symmetrize the *total* to SU(3) characters only at the end —
        so genuinely non-self-dual content (e.g. `T₀·T₂`'s `3+3̄`) is handled
        correctly.  An alternative U(1)²-substrate route through the BPS
        oracle gives equal results (verified); `margin` is accepted for
        back-compat and ignored — the route computes its own q-depth."""
        from sl3_su3_traces import product_trace as _pt
        return _pt(self, factors, K)

    def inner_product(self, a, b, K: int = 20, **kwargs) -> RPowerSeries:
        """Schur pairing `I_{a,b} = Tr(ρ(L_a)·L_b)`, **BPS-free** via
        `sl3_su3_traces.inner_product` (the same fugacity-multiply product
        trace as `trace_word`).  `kwargs` are accepted for back-compat and
        ignored."""
        from sl3_su3_traces import inner_product as _ip
        return _ip(self, a, b, K)

    def L(self, label) -> Element:
        return Element.basis(self.canonicalise(label))

    def __repr__(self) -> str:
        return (f"SU3ADKAlg(rank={self._RANK}, "
                f"gauge_rank={self._GAUGE_RANK}, Z_4-cyclic, R=R(SU(3)))")


if __name__ == "__main__":
    A = SU3ADKAlg()
    print(A)
    print("T_0 =", A.T(0), "  D_0 =", A.D(0))
    print("ρ(T_0) =", A.rho(A.T(0)))
    print("ρ(D_0) =", A.rho(A.D(0)))
    print()
    print("T_0 · T_1 =", A.multiply(A.T(0), A.T(1)))
    print("D_0 · D_1 =", A.multiply(A.D(0), A.D(1)))
    print("T_0 · D_1 =", A.multiply(A.T(0), A.D(1)))
    print("D_0 · D_2 =", A.multiply(A.D(0), A.D(2)))
