"""`A1D3KAlg` -- the [A_1, D_3] Argyres-Douglas K_𝖖-algebra,
fully standalone SU(2)-flavoured realisation with manifest Z_3
cyclic symmetry.

This is the *truly autonomous* hard-coded version: no BPS-quiver
realisation (`BPSKAlgebra`) runtime dependency.  The algebra is presented by 6
multiplicative generators {T_0, T_1, T_2, D_0, D_1, D_2} and the
8 Z_3-symmetric defining relations, plus the q-commutation rules
that hold within each q-commuting tile.  Multiplication is computed
by a recursive reduction algorithm using only this base data.

It follows the same reduction-algorithm design as the `[A_1, A_{2k}]`
polygon algebras (`a1a2k_kalg.py`).

Canonical label structure
-------------------------
Each canonical basis element is a monomial in q-commuting generators,
tensored with an SU(2) χ_k coefficient:

    Label  =  (tile, a, b, k)
      tile ∈ {0, 1, 2, 3, 4, 5}    -- which q-commuting (T_i, D_j) pair
      a, b ∈ ℤ_{≥0}                 -- exponents in this tile
      k ∈ ℤ_{≥0}                    -- chi index (R(SU(2)) coefficient)

Tile catalog:

    tile 0:  T_0 · D_0   (T·D q-twist q^{-1})
    tile 1:  T_0 · D_2   (T·D q-twist q^{+1})
    tile 2:  T_1 · D_1   (q^{-1})
    tile 3:  T_1 · D_0   (q^{+1})
    tile 4:  T_2 · D_2   (q^{-1})
    tile 5:  T_2 · D_1   (q^{+1})

The basis element corresponding to label (tile, a, b, k) is
    M(tile, a, b, k)  :=  q^{-a*b*twist}  ·  T_{tile_T}^a  ·  D_{tile_D}^b  ·  χ_k
(the Weyl-symmetric normalization — bar-invariant since T^a·D^b picks up
exactly q^{+a*b*twist} relative to the symmetric-ordered atom).

ρ (cyclic monodromy) is a permutation of canonical labels:
    ρ-shift on tiles:  0→2, 1→3, 2→4, 3→5, 4→0, 5→1.
    (= shifts both i_T and i_D by 1 mod 3.)

Defining relations (each line is a Z_3-orbit of 3 concrete relations):

    T_i · T_{i+1}    =  1  +  q^{-1}·χ_1·D_i  +  q^{-2}·D_i²
    T_{i+1} · T_i    =  1  +  q     ·χ_1·D_i  +  q^{ 2}·D_i²
    D_i · D_{i+1}    =  1  +  q^{-1}·T_{i+1}
    D_{i+1} · D_i    =  1  +  q     ·T_{i+1}
    T_i · D_{i+1}    =  χ_1  +  q^{-1}·D_i  +  q·D_{i-1}
    D_{i+1} · T_i    =  χ_1  +  q     ·D_i  +  q^{-1}·D_{i-1}

Trace: two-layer pipeline.  Layer 1 (pure algebra) reduces Tr(label) via
ρ²-twisted cyclicity + the Plücker relations to the elementary basis
{Tr(1), Tr(T_i), Tr(D_i)} over R(SU(2))[q^±]; Layer 2 substitutes the
closed-form affine sl(2)_{−4/3} characters
    Tr 1 = κ_0,   Tr D = −q⁻¹·κ_1^anti,
    Tr T = +q⁻¹·(κ_0 − κ_1^sym − χ_1·κ_1^anti)
(so Tr(T_i), Tr(D_i) are nonzero; only the q⁰ coefficient is δ-supported
on the identity gauge cell).  See the Layer-2 section below.
"""

from __future__ import annotations

from typing import Sequence

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element
from cone_kalgebra import ConeKAlgebra
from zplus_ring import ZPlusRing, RElement, RLaurent, RPowerSeries, SU2ZPlusRing
from laurent_poly import LaurentPoly
from qpoch import qpoch_infty


Label = tuple  # (tile, a, b, k) 4-tuple of ints


# ---------------------------------------------------------------------------
# Letter algebra (6 multiplicative generators)
# ---------------------------------------------------------------------------


def _q_commute_twist(L1, L2):
    """q-twist exponent e such that L1·L2 = q^e · L_{lattice sum}, or
    None if (L1, L2) is non-q-commuting (= interaction relation).

    Same-letter L·L: returns 0 (clean square).
    T_i · D_i: returns -1.
    T_i · D_{i-1 mod 3}: returns +1.
    Reversed orders flip sign.
    """
    (k1, i1), (k2, i2) = L1, L2
    if L1 == L2:
        return 0
    if k1 == 'T' and k2 == 'T':
        return None
    if k1 == 'D' and k2 == 'D':
        return None
    if k1 == 'T' and k2 == 'D':
        t_i, d_j, sign = i1, i2, +1
    else:
        t_i, d_j, sign = i2, i1, -1
    diff = (d_j - t_i) % 3
    if diff == 0:
        return -sign
    if diff == 2:
        return +sign
    return None  # diff == 1: interaction


# ---------------------------------------------------------------------------
# Tile catalog
# ---------------------------------------------------------------------------


_TILE_LETTERS = {
    0: (('T', 0), ('D', 0)),
    1: (('T', 0), ('D', 2)),
    2: (('T', 1), ('D', 1)),
    3: (('T', 1), ('D', 0)),
    4: (('T', 2), ('D', 2)),
    5: (('T', 2), ('D', 1)),
}

_TILE_TWIST = {tile: _q_commute_twist(L_T, L_D)
               for tile, (L_T, L_D) in _TILE_LETTERS.items()}


def _tile_for_pair(L_T, L_D):
    for tile, pair in _TILE_LETTERS.items():
        if pair == (L_T, L_D):
            return tile
    return None


def _tile_for_T_letter(L_T):
    """Pick a canonical tile containing T-letter L_T (= one with both
    indices matching the ρ-orbit, default T_i · D_i)."""
    (_, i_T) = L_T
    return {0: 0, 1: 2, 2: 4}[i_T]


def _tile_for_D_letter(L_D):
    """Pick a canonical tile containing D-letter L_D (T_i · D_i)."""
    (_, i_D) = L_D
    return {0: 0, 1: 2, 2: 4}[i_D]


_TILE_RHO = {}
for tile, (L_T, L_D) in _TILE_LETTERS.items():
    (_, i_T), (_, i_D) = L_T, L_D
    new_pair = (('T', (i_T + 1) % 3), ('D', (i_D + 1) % 3))
    _TILE_RHO[tile] = _tile_for_pair(*new_pair)

_TILE_RHO_INV = {v: k for k, v in _TILE_RHO.items()}


def _rho_letter(L):
    """ρ on a letter: (k, i) → (k, (i+1) mod 3)."""
    return (L[0], (L[1] + 1) % 3)


def _rho_n_letter(L, n):
    """ρ^n on a letter."""
    return (L[0], (L[1] + n) % 3)


# ---------------------------------------------------------------------------
# Canonical-form conversion: letters_dict ↔ (tile, a, b, k)
# ---------------------------------------------------------------------------


def _monomial_to_label(letters, k, q_factor):
    """Convert a letter dict {letter: exp} + chi k + q_factor (= integer
    exponent of q) to (Label, LaurentPoly q-coefficient).

    The letter dict must have ≤ 1 T-letter and ≤ 1 D-letter forming
    a q-commuting pair (or be a single letter or empty).
    """
    letters = {L: e for L, e in letters.items() if e > 0}
    t_entries = [(L, e) for L, e in letters.items() if L[0] == 'T']
    d_entries = [(L, e) for L, e in letters.items() if L[0] == 'D']
    if len(t_entries) > 1 or len(d_entries) > 1:
        raise ValueError(f"monomial has multiple T or D letters: {letters}")

    if not t_entries and not d_entries:
        return (0, 0, 0, k), LaurentPoly({q_factor: 1})
    if t_entries and not d_entries:
        L_T, a = t_entries[0]
        return (_tile_for_T_letter(L_T), a, 0, k), LaurentPoly({q_factor: 1})
    if d_entries and not t_entries:
        L_D, b = d_entries[0]
        return (_tile_for_D_letter(L_D), 0, b, k), LaurentPoly({q_factor: 1})

    L_T, a = t_entries[0]
    L_D, b = d_entries[0]
    tile = _tile_for_pair(L_T, L_D)
    if tile is None:
        raise ValueError(
            f"{L_T}, {L_D} non-q-commuting; cannot form single-tile monomial"
        )
    # M(tile, a, b, k) := q^{-a·b·twist} · T^a · D^b · χ_k, so
    #   T^a · D^b = q^{a·b·twist} · M(tile, a, b, k).
    twist = _TILE_TWIST[tile]
    q_factor += a * b * twist
    return (tile, a, b, k), LaurentPoly({q_factor: 1})


def _label_to_monomial(label):
    """Inverse of _monomial_to_label: M(tile, a, b, k) = q^{-a·b·twist}·T^a·D^b·χ_k,
    so expanding M into literal letters pulls out a q^{-a·b·twist} factor."""
    tile, a, b, k = label
    L_T, L_D = _TILE_LETTERS[tile]
    letters = {}
    if a > 0:
        letters[L_T] = a
    if b > 0:
        letters[L_D] = b
    twist = _TILE_TWIST[tile]
    q_factor = -a * b * twist
    return letters, k, q_factor


# ---------------------------------------------------------------------------
# Interaction relations
# ---------------------------------------------------------------------------


def _interaction(L1, L2):
    """For a non-q-commuting (L1, L2), return list of
    (q_delta, output_letters_dict, chi_delta) terms representing
    L1·L2 = Σ q^{q_delta}·(letters_dict)·χ_{chi_delta}.
    """
    if _q_commute_twist(L1, L2) is not None:
        return None
    (k1, i1), (k2, i2) = L1, L2

    if k1 == 'T' and k2 == 'T':
        # T_i · T_j (i ≠ j).
        if (i2 - i1) % 3 == 1:  # forward: T_i · T_{i+1}
            i = i1
            return [
                (0, {}, 0),
                (-1, {('D', i): 1}, 1),
                (-2, {('D', i): 2}, 0),
            ]
        # backward: T_{i+1} · T_i, with i = i2.
        i = i2
        return [
            (0, {}, 0),
            (1, {('D', i): 1}, 1),
            (2, {('D', i): 2}, 0),
        ]

    if k1 == 'D' and k2 == 'D':
        if (i2 - i1) % 3 == 1:
            ip1 = (i1 + 1) % 3
            return [(0, {}, 0), (-1, {('T', ip1): 1}, 0)]
        ip1 = (i2 + 1) % 3
        return [(0, {}, 0), (1, {('T', ip1): 1}, 0)]

    if k1 == 'T':
        # T_i · D_{i+1} = χ_1 + q^{-1}·D_i + q·D_{i-1}.
        i = i1
        return [
            (0, {}, 1),
            (-1, {('D', i): 1}, 0),
            (1, {('D', (i - 1) % 3): 1}, 0),
        ]
    # D_{i+1} · T_i.
    i = i2
    return [
        (0, {}, 1),
        (1, {('D', i): 1}, 0),
        (-1, {('D', (i - 1) % 3): 1}, 0),
    ]


# ---------------------------------------------------------------------------
# Multiplication via letter expansion + recursive reduction
# ---------------------------------------------------------------------------


def _reduce_letter_seq(letter_seq, k, q_factor, depth=0):
    """Reduce [L1, L2, ...] · χ_k · q^{q_factor} to a dict {Label: LaurentPoly}.

    Algorithm:
      - If all letters pairwise q-commute → reorder to canonical (T-first,
        sorted by index), accumulate q-shift from crossings, emit one term.
      - Else → find first non-q-commuting adjacent pair, apply interaction,
        recurse on each output sub-sequence.
    """
    if depth > 50:
        raise RecursionError(
            f"_reduce_letter_seq exceeded depth {depth}; "
            f"letter_seq = {letter_seq}, k = {k}"
        )

    # Drop accidental empties.
    seq = [L for L in letter_seq if L is not None]

    # Check all pairs (not just adjacent) q-commute.
    n = len(seq)
    first_bad = None
    distinct = list(set(seq))
    all_commute = True
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            if _q_commute_twist(distinct[i], distinct[j]) is None:
                all_commute = False
                break
        if not all_commute:
            break
    if not all_commute:
        # Find first adjacent non-q-commuting pair (after some bubbling
        # to make at least one such pair adjacent).  Greedy: scan from
        # left, find first pair that doesn't q-commute (looking ahead).
        # Simplest: scan adjacent first; if all adjacent commute, walk
        # right and bring the offending letter adjacent.
        for idx in range(n - 1):
            if _q_commute_twist(seq[idx], seq[idx + 1]) is None:
                first_bad = idx
                break
        if first_bad is None:
            # All adjacent commute but some non-adjacent don't.  Move
            # the conflicting letters adjacent by bubbling.  Find any
            # non-q-commuting pair (i, j) with i < j, then bubble them
            # adjacent through commuting intermediates.
            target_i = None
            target_j = None
            for i_idx in range(n):
                for j_idx in range(i_idx + 1, n):
                    if _q_commute_twist(seq[i_idx], seq[j_idx]) is None:
                        target_i, target_j = i_idx, j_idx
                        break
                if target_i is not None:
                    break
            # Bubble seq[target_j] left to position target_i + 1.
            new_seq = list(seq)
            q_pre = 0
            pos = target_j
            while pos > target_i + 1:
                L_a, L_b = new_seq[pos - 1], new_seq[pos]
                t = _q_commute_twist(L_a, L_b)
                if t is None:
                    # Shouldn't happen if (a, j) was the chosen pair,
                    # but safe-guard.
                    break
                q_pre += 2 * t
                new_seq[pos - 1], new_seq[pos] = L_b, L_a
                pos -= 1
            # Now recurse with the reorganised sequence.
            return _reduce_letter_seq(
                new_seq, k, q_factor + q_pre, depth + 1,
            )

    if first_bad is None:
        # Bubble-sort to canonical order, accumulating q-crossings.
        sequence = list(seq)
        target = (sorted([L for L in sequence if L[0] == 'T']) +
                  sorted([L for L in sequence if L[0] == 'D']))
        q_delta = 0
        for target_pos in range(len(target)):
            target_L = target[target_pos]
            src_pos = target_pos
            while src_pos < len(sequence) and sequence[src_pos] != target_L:
                src_pos += 1
            while src_pos > target_pos:
                L_a, L_b = sequence[src_pos - 1], sequence[src_pos]
                t = _q_commute_twist(L_a, L_b)
                # L_a · L_b = q^{2t} · L_b · L_a (since L_a·L_b = q^t·M,
                # L_b·L_a = q^{-t}·M).  Swapping the sequence reinterpretation
                # picks up q^{+2t}.
                q_delta += 2 * t
                sequence[src_pos - 1], sequence[src_pos] = L_b, L_a
                src_pos -= 1
        # Now in canonical order; collapse to letter dict.
        out_letters = {}
        for L in sequence:
            out_letters[L] = out_letters.get(L, 0) + 1
        new_label, new_lp = _monomial_to_label(
            out_letters, k, q_factor + q_delta,
        )
        return {new_label: new_lp}

    # Apply interaction at position first_bad.
    L1, L2 = seq[first_bad], seq[first_bad + 1]
    left = seq[:first_bad]
    right = seq[first_bad + 2:]
    interactions = _interaction(L1, L2)
    acc = {}
    for q_delta, new_letters_dict, chi_delta in interactions:
        mid = []
        for L, e in new_letters_dict.items():
            mid += [L] * e
        # SU(2) Clebsch-Gordan fusion of the current running χ_k with
        # χ_{chi_delta}: χ_k · χ_{chi_delta} = Σ_{r=|k-chi_delta|, step 2}^{k+chi_delta} χ_r.
        for new_k in range(abs(k - chi_delta), k + chi_delta + 1, 2):
            sub = _reduce_letter_seq(
                left + mid + right, new_k, q_factor + q_delta, depth + 1,
            )
            for lab, lp in sub.items():
                acc[lab] = acc.get(lab, LaurentPoly({})) + lp
    return {l: lp for l, lp in acc.items() if not lp.is_zero()}


# ---------------------------------------------------------------------------
# Layer-1 trace reduction (tag-move-cycle-Plücker algorithm)
# ---------------------------------------------------------------------------


def _base_to_qdict(q_lp, chi_re):
    """Convert a (LaurentPoly, RElement) base-case entry into the
    dict[q_exp, RElement] uniform representation."""
    out = {}
    for q_exp, q_c in q_lp._coeffs.items():
        if q_c == 0:
            continue
        contrib = chi_re * q_c
        if not contrib.is_zero():
            out[q_exp] = contrib
    return out


def _merge_qdict(target_dict, contrib_dict):
    """Merge contrib_dict into target_dict (both are dict[q, RElement])."""
    for q_exp, r in contrib_dict.items():
        if r.is_zero():
            continue
        if q_exp in target_dict:
            new_r = target_dict[q_exp] + r
            if new_r.is_zero():
                del target_dict[q_exp]
            else:
                target_dict[q_exp] = new_r
        else:
            target_dict[q_exp] = r


def _accumulate_trace_result(acc, key, q_lp_or_dict, chi_re=None):
    """Accumulate contribution under `key` in `acc`.
    Backward-compatible: accepts either (LaurentPoly, RElement) or dict[q, RElement]."""
    if chi_re is not None:
        contrib = _base_to_qdict(q_lp_or_dict, chi_re)
    else:
        contrib = q_lp_or_dict
    bucket = acc.setdefault(key, {})
    _merge_qdict(bucket, contrib)
    if not bucket:
        del acc[key]


def _trace_reduce_word(alg, word, chi_idx, q_factor, depth=0, max_depth=40):
    """Reduce Tr(word · χ_{chi_idx} · q^{q_factor}) to a linear
    combination over elementary trace symbols, using the ρ²-twisted
    cyclicity contract Tr(a · b) = Tr(ρ²(b) · a) applied to a SINGLE
    TAGGED FACTOR.

    Algorithm (the same tagged-letter style as `a1a2k_kalg`):
      1. Tag the LAST letter of the word.  Keep this same tagged
         factor across all cycles.
      2. Cycle: apply ρ² to the tagged factor and move it to the
         FRONT.  q_factor unchanged (cyclicity is exact).
      3. Walk the tagged factor RIGHT through the word via q-commute:
         at each position, if it's non-q-commuting with its right
         neighbour, apply the Plücker interaction (recurse on each
         output term).  Else, commute past (accumulate q-twist),
         continue.
      4. When the tagged factor reaches the end again, cycle (step 2).
      5. After 3 cyclicity applications, the tagged factor has had
         ρ^6 = id applied, so it returns to its starting letter — if
         no Plücker fired in any of the 3 cycles, mark as an
         irreducible elementary trace.

    Returns dict {elem_key: (LaurentPoly q_coef, RElement chi_coef)}.
    """
    if depth > max_depth:
        return {('Tr_max_depth', tuple(word), chi_idx):
                _base_to_qdict(LaurentPoly({q_factor: 1}),
                               alg._R.basis_element(chi_idx))}

    if not word:
        return {('Tr_1',): _base_to_qdict(
            LaurentPoly({q_factor: 1}), alg._R.basis_element(chi_idx))}

    if len(word) == 1:
        L = word[0]
        key = ('Tr_T',) if L[0] == 'T' else ('Tr_D',)
        return {key: _base_to_qdict(
            LaurentPoly({q_factor: 1}), alg._R.basis_element(chi_idx))}

    n = len(word)
    cur_word = list(word)
    q_accum = q_factor

    for cycle_num in range(3):
        # Cycle: ρ²-twisted cyclicity moves the LAST letter to the
        # FRONT with ρ² applied (= a single cyclicity step on the
        # tagged factor).
        L_last = cur_word[-1]
        cur_word = [_rho_n_letter(L_last, 2)] + cur_word[:-1]
        # cur_word[0] is the tagged factor (ρ²-shifted).
        # q_accum unchanged (cyclicity is q-exact).

        # Walk tagged factor RIGHT through the word, checking each
        # adjacent pair for a Plücker.
        tagged_pos = 0
        while tagged_pos < n - 1:
            L_tag = cur_word[tagged_pos]
            L_next = cur_word[tagged_pos + 1]
            t = _q_commute_twist(L_tag, L_next)
            if t is None:
                # Plücker at (tagged_pos, tagged_pos + 1).
                left = cur_word[:tagged_pos]
                right = cur_word[tagged_pos + 2:]
                interactions = _interaction(L_tag, L_next)
                result = {}
                for q_delta, new_letters_dict, chi_delta in interactions:
                    mid = []
                    for L, e in new_letters_dict.items():
                        mid += [L] * e
                    new_word = left + mid + right
                    for new_chi in range(abs(chi_idx - chi_delta),
                                         chi_idx + chi_delta + 1, 2):
                        sub = _trace_reduce_word(
                            alg, new_word, new_chi,
                            q_accum + q_delta, depth + 1, max_depth,
                        )
                        for key, qdict in sub.items():
                            _accumulate_trace_result(result, key, qdict)
                return result
            # Commute past: swap, accumulate q-twist.
            q_accum += 2 * t
            cur_word[tagged_pos], cur_word[tagged_pos + 1] = L_next, L_tag
            tagged_pos += 1
        # Tagged factor now at position n-1 (= end).  Next iteration
        # of the for-loop will cycle it again.

    # 3 cycles done, no Plücker.  Irreducible.
    return {('Tr_irreducible', tuple(cur_word), chi_idx):
            _base_to_qdict(LaurentPoly({q_accum: 1}),
                           alg._R.basis_element(chi_idx))}


# ---------------------------------------------------------------------------
# Layer-2 chiral-algebra characters
# ---------------------------------------------------------------------------
#
# Closed form for the three elementary traces (verified against a
# BPS-quiver reference realisation at K=30 to every probed q-order):
#
#     Tr_1  =        κ_0
#     Tr_D  =  −q⁻¹·                                  κ_1^anti
#     Tr_T  =  +q⁻¹·(κ_0  −  κ_1^sym  −  χ_1·κ_1^anti)
#
# where the basis characters are admissible irreducibles of affine
# sl(2)_{−4/3}:
#     κ_0      = ch(L_{1,0})  (vacuum module).
#     κ_1^sym  = ch(D⁺_{1,1}) + ch(D⁻_{1,1}).
#     κ_1^anti = (ch(D⁺_{1,1}) − ch(D⁻_{1,1})) / (μ − μ⁻¹).
#
# Construction: each character = BPS-Verma denominator × an explicit
# numerator constructed from the Creutzig–Ridout Σ_j formulas (known
# closed forms for admissible-level sl(2) characters), with BPS Schur
# convention q_CR = q² and z = μ.


def _laurent_clean(d):
    """Drop zero entries from a {q_pow: {mu_pow: int}} dict."""
    out = {}
    for q, mud in d.items():
        cleaned = {m: c for m, c in mud.items() if c != 0}
        if cleaned:
            out[q] = cleaned
    return out


def _laurent_mul(a, b, K):
    """Multiply two (q, μ)-Laurent dicts, truncated at q^K."""
    out = {}
    for qa, mua in a.items():
        for qb, mub in b.items():
            if qa + qb > K:
                continue
            qab = qa + qb
            row = out.setdefault(qab, {})
            for ma, ca in mua.items():
                for mb, cb in mub.items():
                    row[ma+mb] = row.get(ma+mb, 0) + ca*cb
    return _laurent_clean(out)


def _laurent_combine(*sigs):
    """Sum a sequence of (q, μ)-Laurent dicts."""
    out = {}
    for sig in sigs:
        for q, mud in sig.items():
            row = out.setdefault(q, {})
            for m, c in mud.items():
                row[m] = row.get(m, 0) + c
    return _laurent_clean(out)


def _laurent_negate(sig):
    return {q: {m: -c for m, c in mud.items()} for q, mud in sig.items()}


def _laurent_reflect_mu(sig):
    """Apply μ → μ⁻¹."""
    return {q: {-m: c for m, c in mud.items()} for q, mud in sig.items()}


def _laurent_shift_mu(sig, shift):
    return {q: {m+shift: c for m, c in mud.items()} for q, mud in sig.items()}


def _divide_by_1_minus_mu2(mud, max_iter=500):
    """Divide a μ-Laurent dict by (1 − μ²), assuming clean divisibility.
    Returns the quotient as a μ-Laurent dict (or raises if not divisible)."""
    rem = dict(mud)
    quot = {}
    for _ in range(max_iter):
        nz = [(p, c) for p, c in rem.items() if c != 0]
        if not nz:
            return {p: c for p, c in quot.items() if c != 0}
        p_max = max(p for p, _ in nz)
        c = rem[p_max]
        qp = p_max - 2
        quot[qp] = quot.get(qp, 0) + (-c)
        rem[p_max] = 0
        rem[qp] = rem.get(qp, 0) + c
    raise ValueError("_divide_by_1_minus_mu2 did not converge")


def _divide_by_mu_minus_muinv(mud, max_iter=500):
    """Divide a μ-Laurent dict by (μ − μ⁻¹), assuming clean divisibility."""
    rem = dict(mud)
    quot = {}
    for _ in range(max_iter):
        nz = [(p, c) for p, c in rem.items() if c != 0]
        if not nz:
            return {p: c for p, c in quot.items() if c != 0}
        p_max = max(p for p, _ in nz)
        c = rem[p_max]
        quot[p_max - 1] = quot.get(p_max - 1, 0) + c
        rem[p_max] = 0
        rem[p_max - 2] = rem.get(p_max - 2, 0) + c
    raise ValueError("_divide_by_mu_minus_muinv did not converge")


def _divide_each(sig, divider):
    return {q: divider(mud) for q, mud in sig.items() if mud}


def _bps_verma(K):
    """BPS Verma denominator inverted (= the symmetric q-Pochhammer-like
    product) as a (q, μ)-Laurent dict truncated at q^K:

        BPS_Verma = ∏_{n ≥ 1} 1 / [(1 − q^{2n})(1 − q^{2n} μ²)(1 − q^{2n} μ⁻²)].
    """
    result = {0: {0: 1}}
    n = 1
    while 2 * n <= K:
        for fpow in (0, 2, -2):
            geom = {}
            m = 0
            while 2 * n * m <= K:
                geom[2 * n * m] = {fpow * m: 1}
                m += 1
            result = _laurent_mul(result, geom, K)
        n += 1
    return result


def _sigma_j_add(out, e, mu_pow, sign):
    """Helper: add `sign` to `out[e][mu_pow]`, creating entries as needed."""
    row = out.setdefault(e, {})
    row[mu_pow] = row.get(mu_pow, 0) + sign


def _sigma_j_L10(K):
    """Σ_j numerator for L_{1,0} = σ⁻¹·D⁺_{1,2} at k = −4/3 (BPS q)."""
    out = {}
    j_bound = max(5, int((K // 6) ** 0.5) + 5)
    for j in range(-j_bound, j_bound + 1):
        e1 = 12 * j * j - 6 * j
        if 0 <= e1 <= K:
            _sigma_j_add(out, e1, 4 * j, +1)
        e2 = 12 * j * j - 18 * j + 6
        if 0 <= e2 <= K:
            _sigma_j_add(out, e2, 4 * j - 2, -1)
    return _laurent_clean(out)


def _sigma_j_Dplus11(K):
    """Σ_j numerator for D⁺_{1,1} at k = −4/3 (BPS q)."""
    out = {}
    j_bound = max(5, int((K // 6) ** 0.5) + 5)
    for j in range(-j_bound, j_bound + 1):
        e1 = 12 * j * j + 2 * j
        if 0 <= e1 <= K:
            _sigma_j_add(out, e1, 4 * j, +1)
        e2 = 12 * j * j - 10 * j + 2
        if 0 <= e2 <= K:
            _sigma_j_add(out, e2, 4 * j - 2, -1)
    return _laurent_clean(out)


def _layer2_basis_mu(K):
    """Compute κ_0, κ_1^sym, κ_1^anti, χ_1·κ_1^anti as (q, μ)-Laurent dicts
    truncated at q^K."""
    verma = _bps_verma(K)
    sig_L10 = _sigma_j_L10(K)
    sig_Dp = _sigma_j_Dplus11(K)
    sig_Dm = _laurent_reflect_mu(sig_Dp)

    # κ_0 numerator = Σ_j[L_{1,0}] / (1 − μ²)
    num_k0 = _divide_each(sig_L10, _divide_by_1_minus_mu2)
    kappa0 = _laurent_mul(verma, num_k0, K)

    # κ_1^sym numerator = (Σ_+ − μ² · Σ_−) / (1 − μ²)
    raw_sym = _laurent_combine(
        sig_Dp, _laurent_shift_mu(_laurent_negate(sig_Dm), 2)
    )
    num_k1sym = _divide_each(raw_sym, _divide_by_1_minus_mu2)
    kappa1_sym = _laurent_mul(verma, num_k1sym, K)

    # κ_1^anti numerator = (Σ_+ − Σ_−) / (μ − μ⁻¹)
    raw_anti = _laurent_combine(sig_Dp, _laurent_negate(sig_Dm))
    num_k1anti = _divide_each(raw_anti, _divide_by_mu_minus_muinv)
    kappa1_anti = _laurent_mul(verma, num_k1anti, K)

    # χ_1 · κ_1^anti  (= (μ + μ⁻¹) · κ_1^anti, shifts μ-powers by ±1).
    chi1_k1anti = _laurent_combine(
        _laurent_shift_mu(kappa1_anti, 1),
        _laurent_shift_mu(kappa1_anti, -1),
    )

    return kappa0, kappa1_sym, kappa1_anti, chi1_k1anti


def _mu_dict_to_chi_relement(mud, R):
    """Convert a Weyl-symmetric μ-Laurent dict {μ_pow: int} to an RElement
    over SU(2) by greedy χ-decomposition.  Returns None if not Weyl-symmetric."""
    md = dict(mud)
    coeffs: dict[int, int] = {}
    iters = 0
    while True:
        iters += 1
        if iters > 200:
            return None
        nz = [(m, c) for m, c in md.items() if c != 0]
        if not nz:
            break
        max_m = max(m for m, _ in nz)
        if max_m < 0:
            return None  # not μ-symmetric — μ⁻¹ tail without μ head
        c = md[max_m]
        coeffs[max_m] = coeffs.get(max_m, 0) + c
        for k in range(max_m, -max_m - 1, -2):
            md[k] = md.get(k, 0) - c
    return RElement(R, {n: c for n, c in coeffs.items() if c != 0})


def _mu_laurent_to_rpowerseries(sig, R, K):
    """Convert a (q, μ)-Laurent dict to RPowerSeries over R(SU(2)), truncated at q^K."""
    coeffs: dict[int, RElement] = {}
    for q, mud in sig.items():
        if q > K:
            continue
        re = _mu_dict_to_chi_relement(mud, R)
        if re is None:
            raise ValueError(
                f"Layer 2: q^{q} coefficient is not Weyl-symmetric: {mud}"
            )
        if not re.is_zero():
            coeffs[q] = re
    return RPowerSeries(R, coeffs, K)


def _shift_q_rpowerseries(rps, shift):
    """Multiply an RPowerSeries by q^{shift} (positive or negative shift)."""
    return RPowerSeries(
        rps.ring,
        {q + shift: c for q, c in rps.coeffs.items() if q + shift <= rps.K},
        rps.K,
    )


# ---------------------------------------------------------------------------
# A1D3KAlg
# ---------------------------------------------------------------------------


class A1D3KAlg(ConeKAlgebra):
    """[A_1, D_3] K-algebra, standalone SU(2)-flavoured, Z_3-symmetric.

    Inherits `ConeKAlgebra`: `multiply` is routed through
    `A1D3ConeData` (cone-data presentation on 3-tuple labels
    `(tile, a, b)` over `SU2ZPlusRing`, with χ-content in the
    RLaurent coefficient).  The public API stays on 4-tuple labels
    `(tile, a, b, k)`: `multiply` splits χ at the boundary, performs
    the cone-data multiplication, then re-expands the RLaurent
    output into 4-tuple labels with LaurentPoly coefficients.

    The Layer-1 / Layer-2 trace pipeline (`trace_layer1`,
    `trace_layer2`, `trace`) is preserved verbatim — it has its own
    tag-move-cycle-Plücker algorithm independent of the cone-data
    Layer-1 reducer.
    """

    _RANK = 3
    _GAUGE_RANK = 2

    def __init__(self):
        self._R = SU2ZPlusRing()

    @property
    def rank(self) -> int:
        return self._RANK

    @property
    def gauge_rank(self) -> int:
        return self._GAUGE_RANK

    # -- generators -------------------------------------------------------

    def T(self, i: int) -> Label:
        i = i % 3
        return ({0: 0, 1: 2, 2: 4}[i], 1, 0, 0)

    def D(self, i: int) -> Label:
        i = i % 3
        return ({0: 0, 1: 2, 2: 4}[i], 0, 1, 0)

    def chi(self, k: int) -> Label:
        if k < 0:
            raise ValueError(f"chi(k): k must be ≥ 0, got {k}")
        return (0, 0, 0, k)

    # -- KAlgebra primitives ---------------------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self) -> Label:
        return (0, 0, 0, 0)

    def canonicalise(self, x) -> Label:
        x = tuple(int(c) for c in x)
        if len(x) != 4:
            raise ValueError(f"A1D3KAlg label must be 4-tuple, got {x}")
        tile, a, b, k = x
        if not (0 <= tile <= 5 and a >= 0 and b >= 0 and k >= 0):
            raise ValueError(f"invalid label {x}")
        # If a == 0 or b == 0, choose canonical tile (= tile-i-i for that
        # letter's i).  This ensures unique representation for single
        # letters and the identity.
        L_T, L_D = _TILE_LETTERS[tile]
        if a == 0 and b == 0:
            return (0, 0, 0, k)
        if a == 0:
            return (_tile_for_D_letter(L_D), 0, b, k)
        if b == 0:
            return (_tile_for_T_letter(L_T), a, 0, k)
        return x

    def cone_data(self):
        """Cone-data presentation on 3-tuple labels (χ-index absorbed
        into RLaurent coefficients).  Lazy-built."""
        if not hasattr(self, "_cone_data_cache"):
            from a1d3_cone_data import A1D3ConeData
            self._cone_data_cache = A1D3ConeData()
        return self._cone_data_cache

    def _canonical_rho2_orbit_rep(self, label):
        """ρ² has order 3 on tiles; default orbit walk with safety."""
        return KAlgebra._canonical_rho2_orbit_rep(self, label)

    def _trace_residual(self, seed_label, K):
        """Required abstract on ConeKAlgebra.  Unused: A1D3's
        public `trace` uses its own `trace_layer1` + `trace_layer2`
        pipeline (tag-move-cycle-Plücker on letters, not on cones).
        Raising rather than wiring keeps the two pipelines clearly
        separated."""
        raise NotImplementedError(
            "A1D3KAlg._trace_residual: A1D3 uses its own trace_layer1 "
            "/ trace_layer2 pipeline; cone-data trace seeds are not "
            "routed here."
        )

    def multiply(self, a: Label, b: Label) -> Element:
        """Cone-data multiplication, with χ-index folding at the
        4-tuple ↔ 3-tuple boundary.

        Algorithm:
          1. Canonicalise inputs; split each 4-tuple `(tile, a, b, k)`
             into a 3-tuple `(tile, a, b)` and a χ-index `k`.
          2. Multiply the 3-tuple labels via `cone_data().derived_multiply`
             — this returns an `Element` over `RLaurent[SU(2)]`.
          3. Multiply the RLaurent coefficients by `χ_{k_a}·χ_{k_b}`
             (SU(2) Clebsch-Gordan, handled by `RElement.__mul__`).
          4. Re-expand each `(3-tuple, RLaurent)` output term into a
             sum of `(4-tuple, LaurentPoly)` terms: for each q-power
             and each SU(2) basis index `k_out`, emit a 4-tuple
             `(tile, a, b, k_out)` with the corresponding integer
             q-coefficient.
        """
        a = self.canonicalise(a)
        b = self.canonicalise(b)
        tile_a, a_exp, b_exp_a, k_a = a
        tile_b, a_exp_b, b_exp_b, k_b = b

        # Cone-data multiply on 3-tuples.
        result_cone = self.cone_data().derived_multiply(
            (tile_a, a_exp, b_exp_a), (tile_b, a_exp_b, b_exp_b),
        )

        # χ_{k_a} · χ_{k_b} via SU(2) CG (handled by RElement.__mul__).
        chi_a = self._R.basis_element(k_a)
        chi_b = self._R.basis_element(k_b)
        chi_prod = chi_a * chi_b  # RElement: Σ χ_j over CG decomposition

        # Re-expand to 4-tuple labels.
        out: dict[Label, LaurentPoly] = {}
        for lab3, coef in result_cone.terms.items():
            # Coerce to RLaurent for uniform handling.
            if isinstance(coef, LaurentPoly):
                rl = RLaurent(self._R, dict(coef._coeffs))
            else:
                rl = coef
            # Scale by chi_prod on the R-side.
            for q_exp, r_elt in rl.coeffs.items():
                scaled = r_elt * chi_prod
                if scaled.is_zero():
                    continue
                # Expand the R-element into SU(2)-basis components.
                for k_out, coef_int in scaled.terms.items():
                    if coef_int == 0:
                        continue
                    lab4 = (lab3[0], lab3[1], lab3[2], k_out)
                    lp_add = LaurentPoly({q_exp: int(coef_int)})
                    out[lab4] = out.get(lab4, LaurentPoly({})) + lp_add
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    def rho(self, a: Label) -> Label:
        tile, a_exp, b_exp, k = self.canonicalise(a)
        return self.canonicalise(
            (_TILE_RHO[tile], a_exp, b_exp, k)
        )

    def rho_inverse(self, a: Label) -> Label:
        tile, a_exp, b_exp, k = self.canonicalise(a)
        return self.canonicalise(
            (_TILE_RHO_INV[tile], a_exp, b_exp, k)
        )

    # -- two-layer trace ------------------------------------------------
    #
    # Layer 1 (pure algebra): reduce Tr(label) to the elementary basis
    # {Tr(1), Tr_T, Tr_D} over R(SU(2))[q^±] via the tag-move-cycle-
    # Plücker algorithm (ρ²-twisted cyclicity + the interaction
    # relations).  Only the q⁰ coefficient is δ-supported on the
    # identity gauge cell (the vacuum_only=True shortcut).
    #
    # Layer 2 (analytic plug-in): the affine sl(2)_{−4/3} admissible
    # characters — Tr_1 = κ_0, Tr_D = −q⁻¹·κ_1^anti,
    # Tr_T = +q⁻¹·(κ_0 − κ_1^sym − χ_1·κ_1^anti).

    def trace_layer1(
        self, label: Label,
    ) -> dict:
        """Layer-1 trace reduction via tag-move-cycle-Plücker algorithm.

        Algorithm:
          1. Tag a letter in the monomial; χ_n factors are pulled out
             (they're in the coefficient ring R(SU(2))).
          2. Move the tagged letter right via q-commute (collecting
             q-twists).
          3. Use ρ²-cyclicity: Tr(L·rest) = Tr(rest·ρ²(L)) — brings
             the tagged letter (now ρ²-shifted) back to the right end.
          4. If the tagged letter and its left neighbor Plücker-relate,
             apply the interaction relation, recurse on each output.
          5. Else, cycle again (up to 3 times since ρ has order 3 on
             letters).  If no Plücker after 3 cycles, the trace is
             irreducible by this algorithm.

        Returns a dict `{elementary_trace_key: (LaurentPoly q-coef,
        RElement chi-coef)}` decomposing Tr(label) as a R(SU(2))[q^±]-
        linear combination of elementary traces.

        Elementary trace keys:
          ('Tr_1',)                -- Tr(1)                  (= identity)
          ('Tr_T',)                -- Tr(T_0) = Tr(T_1) = Tr(T_2)
                                      (ρ²-orbit-averaged)
          ('Tr_D',)                -- Tr(D_0) = Tr(D_1) = Tr(D_2)
          ('Tr_irreducible', word) -- irreducible higher trace
                                      (= cannot Plücker-reduce in
                                       3 cyclic shifts)

        Closed form only for `label` that reduces fully to identity-
        gauge-cell content (= same as before).  General reduction is
        algorithmic.
        """
        tile, a_exp, b_exp, k = self.canonicalise(label)
        if a_exp == 0 and b_exp == 0:
            # Identity gauge cell: Tr = χ_k · Tr(1).
            return {('Tr_1',): {0: self._R.basis_element(k)}}

        # Expand label as letter word + chi + q-factor.
        letters, chi_idx, q_factor = _label_to_monomial(label)
        # Build the word as a sequence of letters (one per multiplicity).
        word = []
        for L in sorted([L for L in letters if L[0] == 'T']):
            word += [L] * letters[L]
        for L in sorted([L for L in letters if L[0] == 'D']):
            word += [L] * letters[L]

        # Run the recursive reduction.
        return _trace_reduce_word(self, word, chi_idx, q_factor, depth=0)

    def trace_layer1_element(self, elt: Element) -> dict:
        """Layer-1 trace reduction applied to an Element (sum of L's).
        Returns dict {elem_key: dict[q_exp, RElement]} accumulating
        per-term reductions, with q-Laurent coefficients from elt.terms
        multiplied through.
        """
        result = {}
        for lab, lp in elt.terms.items():
            sub = self.trace_layer1(lab)
            for key, qdict in sub.items():
                # Multiply qdict by lp (LaurentPoly).
                for q_exp_lp, q_c_lp in lp._coeffs.items():
                    if q_c_lp == 0:
                        continue
                    shifted = {q + q_exp_lp: r * q_c_lp
                               for q, r in qdict.items()
                               if not (r * q_c_lp).is_zero()}
                    _accumulate_trace_result(result, key, shifted)
        return result

    def trace_layer2(
        self, K: int = 20,
    ) -> dict[str, RPowerSeries]:
        """Layer-2 elementary traces as R(SU(2))[[q]] series truncated at q^K.

        Closed form (verified against a BPS-quiver reference realisation
        at K=30 at every odd q-power up to q^17):

            Tr_1  =        κ_0
            Tr_D  =  −q⁻¹·                                  κ_1^anti
            Tr_T  =  +q⁻¹·(κ_0  −  κ_1^sym  −  χ_1·κ_1^anti)

        where {κ_0, κ_1^sym, κ_1^anti} are the BPS-Verma-renormalised
        characters of admissible irreducibles {L_{1,0}, D⁺_{1,1},
        D⁻_{1,1}} of affine sl(2)_{−4/3} (the chiral algebra of
        [A_1, D_3]).  See the module-level Layer-2 docstring.

        Returns dict mapping elementary trace key → RPowerSeries:
            'Tr_1', 'Tr_T', 'Tr_D'.
        """
        # Need κ basis up to q^{K+1} since Tr_T and Tr_D pick up q^{-1}.
        kappa0_mu, k1sym_mu, k1anti_mu, chi1_k1anti_mu = _layer2_basis_mu(K + 1)

        kappa0_rps = _mu_laurent_to_rpowerseries(kappa0_mu, self._R, K + 1)
        k1sym_rps = _mu_laurent_to_rpowerseries(k1sym_mu, self._R, K + 1)
        k1anti_rps = _mu_laurent_to_rpowerseries(k1anti_mu, self._R, K + 1)
        chi1_k1anti_rps = _mu_laurent_to_rpowerseries(
            chi1_k1anti_mu, self._R, K + 1,
        )

        # Tr_1 = κ_0 (truncate to K).
        tr1 = RPowerSeries(
            self._R,
            {q: c for q, c in kappa0_rps.coeffs.items() if q <= K},
            K,
        )

        # Tr_D = −q⁻¹ · κ_1^anti.
        tr_d_inner = -k1anti_rps
        tr_d = _shift_q_rpowerseries(tr_d_inner, -1)
        tr_d = RPowerSeries(
            self._R,
            {q: c for q, c in tr_d.coeffs.items() if q <= K},
            K,
        )

        # Tr_T = +q⁻¹ · (κ_0 − κ_1^sym − χ_1·κ_1^anti).
        tr_t_inner = kappa0_rps - k1sym_rps - chi1_k1anti_rps
        tr_t = _shift_q_rpowerseries(tr_t_inner, -1)
        tr_t = RPowerSeries(
            self._R,
            {q: c for q, c in tr_t.coeffs.items() if q <= K},
            K,
        )

        return {('Tr_1',): tr1, ('Tr_T',): tr_t, ('Tr_D',): tr_d}

    def trace(
        self, a: Label, K: int = 20, vacuum_only: bool = False, **kwargs,
    ) -> RPowerSeries:
        """Trace via two-layer decomposition: layer 1 reduces to the
        elementary basis {Tr_1, Tr_T, Tr_D}, layer 2 substitutes the
        closed-form chiral-algebra characters.

        With `vacuum_only=True`, substitutes Tr_1 = 1, Tr_T = Tr_D = 0
        (= leading vacuum contribution of the Schur index), giving only
        the algebraically well-defined leading piece.

        With `vacuum_only=False` (default), uses the full Layer-2 closed
        form for all three elementary traces.
        """
        reduction = self.trace_layer1(a)
        if vacuum_only:
            tr1_qdict = reduction.get(('Tr_1',))
            if not tr1_qdict:
                return RPowerSeries(self._R, {}, K)
            out_terms = {q: r for q, r in tr1_qdict.items()
                         if not r.is_zero() and q <= K}
            return RPowerSeries(self._R, out_terms, K)

        # Determine the most-negative q-shift in the layer-1 reduction;
        # negative shifts require knowing Layer 2 up to q^{K + |shift|}.
        min_q_exp = 0
        for qdict in reduction.values():
            if not qdict:
                continue
            min_q_exp = min(min_q_exp, min(qdict.keys()))
        K_eff = K - min_q_exp  # = K + |min_q_exp| when min_q_exp < 0

        layer2 = self.trace_layer2(K_eff)
        out = RPowerSeries.zero(self._R, K)
        for key in (('Tr_1',), ('Tr_T',), ('Tr_D',)):
            qdict = reduction.get(key)
            if not qdict:
                continue
            base = layer2[key]
            for q_exp, r_coef in qdict.items():
                shifted = _shift_q_rpowerseries(base, q_exp)
                # Truncate to target K.
                shifted_truncated = RPowerSeries(
                    shifted.ring,
                    {q: c for q, c in shifted.coeffs.items() if q <= K},
                    K,
                )
                contrib = shifted_truncated * r_coef
                out = out + contrib
        return out

    def r_label_decompose(self, label):
        """The single-irrep flavour-lift coordinate: peel the SU(2) spin
        index `k` (the R-basis-label) off the gauge monomial
        `(tile, a, b)`."""
        tile, a, b, k = self.canonicalise(label)
        return (tile, a, b, 0), k

    def r_label_compose(self, section, r_basis_label):
        """Inverse of `r_label_decompose`: insert the spin `r_basis_label`
        into the (flavour-trivial) gauge section's χ-slot.  A direct slot
        write — no `embed_R`/`multiply` round-trip."""
        tile, a, b, _ = self.canonicalise(section)
        return self.canonicalise((tile, a, b, r_basis_label))

    def embed_R(self, r: RElement) -> Element:
        """Central embedding `R(SU(2)) ↪ A_𝖖`: each character `χ_k` maps
        to the central canonical basis element `L_{(0,0,0,k)}` (the pure
        spin-`k/2` character), extended Z-linearly.  Inverse-compatible
        with `r_label_decompose`: `embed_R(χ_k) · L_{(tile,a,b,0)}
        == L_{(tile,a,b,k)}` since `χ_k` is central.  (Still backs the
        default `from_R_form`; `r_label_compose` no longer needs it.)"""
        R = self.coefficient_ring()
        if not isinstance(r, RElement) or r.ring != R:
            raise TypeError(
                "embed_R: argument must be an RElement over coefficient_ring()"
            )
        out = Element.zero()
        for k, coeff in r.terms.items():
            if coeff == 0:
                continue
            out = out + Element.basis(self.canonicalise((0, 0, 0, k))) * coeff
        return out

    # -- convenience ------------------------------------------------------

    def L(self, label) -> Element:
        return Element.basis(self.canonicalise(label))

    def __repr__(self) -> str:
        return (
            f"A1D3KAlg(rank={self._RANK}, gauge_rank={self._GAUGE_RANK}, "
            f"standalone, Z_3-symmetric)"
        )
