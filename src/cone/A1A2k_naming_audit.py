"""
A1A2k_naming_audit.py
=====================

Generalises the heptagon naming-and-Plücker audit to the family
$[A_1, A_{2k}]$ for arbitrary k.

Conjecture (heptagon-derived):
  * orbit i (for i = 1, ..., k)  ↔  length-(i+1) chord on the (2k+3)-gon
  * a per-orbit ρ-shift  d_i  selects which lattice charge sits at the
    'starting vertex' index 0.

User's full geometric rule:
  * chords disjoint  (no shared vertex, no interior cross)  ↔
        L_a · L_b commute exactly (q-power 0)
  * chords share a vertex  (no cross)  ↔  q-commute with q^{2⟨γ_a, γ_b⟩}
  * chords interior-cross  ↔  multi-term Plücker

This module:

  * iteratively determines the consistent shifts d_1, ..., d_k (anchoring
    d_1 = 0 by global-ρ symmetry) for each k,
  * verifies the cross ↔ Plücker correspondence has zero mismatches,
  * verifies the q-power-based q-commute rule has zero mismatches,
  * reports the resulting (length, shift) per orbit.

Per-orbit alphabet:  L_{i, j}  for  i ∈ {1, ..., k}, j ∈ Z/(2k+3),
with  L_{0, j} := 1  (edge normalisation).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from itertools import product as iproduct


def A2k(k: int) -> "BPSKAlgebra":
    # Lazy: BPSKAlgebra (the realization spine) is needed ONLY by the
    # audit/seed-finding helpers (A2k / search_consistent_shifts /
    # natural_orbit_seeds), NOT by the closed-form character path
    # (`predicted_lengths_and_shifts`, which is purely combinatorial).  A
    # module-level import here gated the whole spine-free A1A_even character
    # path behind the BPS engine.
    raise NotImplementedError(  # BPS cross-check: not part of the spine-free release
        "BPS cross-check is unavailable in the spine-free ConeKAlgebra release")
    n = 2 * k
    B = [[0] * n for _ in range(n)]
    for a in range(n - 1):
        B[a][a + 1] = 1
        B[a + 1][a] = -1
    nodes = [tuple(1 if p == a else 0 for p in range(n)) for a in range(n)]
    return BPSKAlgebra(pairing=B, node_charges=nodes, verify="off")


def rho_orbit(A, g):
    o = [tuple(g)]
    while True:
        nx = tuple(A.rho(o[-1]))
        if nx == o[0]:
            return o
        o.append(nx)


def natural_seed(k, i):
    """Alternating-pattern seed:  k - i + 1 ones at even positions."""
    n = 2 * k
    v = [0] * n
    for s in range(k - i + 1):
        v[2 * s] = 1
    return tuple(v)


def classify(ep1, ep2, cyc):
    """Geometric classification of two chords on a cyc-gon."""
    if set(ep1) == set(ep2):
        return "same"
    if set(ep1) & set(ep2):
        return "share"
    a, b = ep1
    arc1, arc2 = set(), set()
    x = (a + 1) % cyc
    while x != b:
        arc1.add(x); x = (x + 1) % cyc
    x = (b + 1) % cyc
    while x != a:
        arc2.add(x); x = (x + 1) % cyc
    c, d = ep2
    if (c in arc1 and d in arc2) or (c in arc2 and d in arc1):
        return "cross"
    return "disjoint"


def antisym_pair(B, a, b):
    n = len(a)
    return sum(a[i] * B[i][j] * b[j] for i in range(n) for j in range(n))


def search_consistent_shifts(k: int, conjecture_lengths: dict[int, int] | None = None):
    """Iteratively determine the orbit-i ρ-shift d_i (for i = 2, ..., k)
    such that cross↔Plücker holds against orbit 1 (anchored at shift 0).
    Returns (lengths, shifts, full_mismatch_count, q_pairing_mismatch_count)."""
    cyc = 2 * k + 3
    A = A2k(k)
    if conjecture_lengths is None:
        conjecture_lengths = {i: i + 1 for i in range(1, k + 1)}
    n = 2 * k
    # Local copy of the pairing matrix
    B = [[0] * n for _ in range(n)]
    for a in range(n - 1):
        B[a][a + 1] = 1; B[a + 1][a] = -1
    orbits = {i: rho_orbit(A, natural_seed(k, i)) for i in range(1, k + 1)}

    # Lattice charge of L((i, j)) is just  orbits[i][j]  (natural-seed
    # indexing).  The shift d_i only re-interprets the GEOMETRIC chord
    # assigned to L((i, j)), as chord (j+d_i, j+d_i + length_i).
    def charge(i, j):
        return orbits[i][j % cyc]
    def chord_ep(i, j, d, L):
        a = (j + d) % cyc
        return (a, (a + L) % cyc)

    # Anchor orbit 1 at shift 0
    shifts = {1: 0}
    L1 = conjecture_lengths[1]

    for i in range(2, k + 1):
        Li = conjecture_lengths[i]
        good = []
        for d in range(cyc):
            ok = True
            for j1 in range(cyc):
                for j2 in range(cyc):
                    ga = charge(1, j1); gb = charge(i, j2)
                    ep_a = chord_ep(1, j1, shifts[1], L1)
                    ep_b = chord_ep(i, j2, d, Li)
                    geom = classify(ep_a, ep_b, cyc)
                    prod = A.multiply(ga, gb)
                    alg = "PLÜCKER" if len(prod.terms) > 1 else "monomial"
                    if geom == "cross" and alg != "PLÜCKER":
                        ok = False; break
                    if geom in ("share", "disjoint") and alg != "monomial":
                        ok = False; break
                if not ok:
                    break
            if ok:
                good.append(d)
        if not good:
            return None
        shifts[i] = good[0]

    # Full verification across ALL ordered pairs (every orbit × every orbit)
    labels = [(i, j) for i in range(1, k + 1) for j in range(cyc)]
    full_mism = 0
    q_mism = 0
    for la in labels:
        for lb in labels:
            if la == lb:
                continue
            ia, ja = la; ib, jb = lb
            ga = charge(ia, ja); gb = charge(ib, jb)
            La = conjecture_lengths[ia]; Lb = conjecture_lengths[ib]
            ep_a = chord_ep(ia, ja, shifts[ia], La)
            ep_b = chord_ep(ib, jb, shifts[ib], Lb)
            geom = classify(ep_a, ep_b, cyc)
            prod = A.multiply(ga, gb)
            alg = "PLÜCKER" if len(prod.terms) > 1 else "monomial"
            if geom == "cross" and alg != "PLÜCKER":
                full_mism += 1
            elif geom in ("share", "disjoint") and alg != "monomial":
                full_mism += 1
            if alg == "monomial":
                rev = A.multiply(gb, ga)
                if len(rev.terms) == 1:
                    (fch, fcoef), = prod.terms.items()
                    (rch, rcoef), = rev.terms.items()
                    if fch == rch and len(fcoef._coeffs) == 1 and len(rcoef._coeffs) == 1:
                        fp = next(iter(fcoef._coeffs))
                        rp = next(iter(rcoef._coeffs))
                        q_exp = fp - rp
                        predicted = 2 * antisym_pair(B, ga, gb)
                        if q_exp != predicted:
                            q_mism += 1
    return (conjecture_lengths, shifts, full_mism, q_mism)


def predicted_lengths_and_shifts(k: int) -> tuple[dict[int, int], dict[int, int]]:
    """Closed-form length + shift assignment for orbit i in [A_1, A_{2k}],
    verified by `search_consistent_shifts` at k = 2, 3, 4, 5.

    Rule:
      * Pre-peak orbits 1, ..., ⌈(k+1)/2⌉  get even lengths 2, 4, 6, ...
        capped at the largest even ≤ k+1.
      * Post-peak orbits get the remaining (odd) lengths in descending
        order, ending at 3.
      * Shift for an even-length orbit is 0.
      * Shift for length  2m+1  (m ≥ 1)  is  2(k - m + 1)  (mod 2k+3).

    This labeling is a vestige of the alternating-pattern lattice seed
    `natural_seed(k, i)` and is kept for internal seed-finding only.
    The public algebra interface uses the natural labeling (see
    `natural_orbit_seeds`):  orbit a ∈ {1, ..., k} has length a+1 and
    shift 0,  so  L_{a, j}  has chord endpoints  (j, j+a+1)  on the
    (2k+3)-gon.
    """
    lengths_in_order = [2 * i for i in range(1, k + 2) if 2 * i <= k + 1]
    used_evens = set(lengths_in_order)
    all_odds_desc = [L for L in range(k + 1, 1, -1)
                     if L % 2 == 1 and L not in used_evens]
    seq = lengths_in_order + all_odds_desc
    assert len(seq) == k, (k, seq)
    lengths = {i + 1: L for i, L in enumerate(seq)}
    shifts = {}
    cyc = 2 * k + 3
    for i in range(1, k + 1):
        L = lengths[i]
        if L % 2 == 0:
            shifts[i] = 0
        else:
            m = (L - 1) // 2
            shifts[i] = (2 * (k - m + 1)) % cyc
    return lengths, shifts


def natural_orbit_seeds(k: int) -> dict[int, tuple[int, ...]]:
    """Lattice seeds for the natural labeling of `[A_1, A_{2k}]`.

    In the natural labeling, orbit a ∈ {1, ..., k} corresponds to the
    chord of length a+1 on the (2k+3)-gon, and the "j = 0" representative
    L_{a, 0} has chord endpoints (0, a+1).  ρ-action on labels is just
    `L_{a, j} ↦ L_{a, j+1}`, so all per-orbit shifts are zero.

    Returns `{a: seed_charge for a = 1, ..., k}`, where `seed_charge`
    is the BPS-lattice charge γ such that  L_{a, 0} = X[γ].  Computed
    by taking the alternating-pattern seed of the (old) length-(a+1)
    orbit and applying ρ^{-S_old} on the lattice, where S_old is the
    legacy shift of that orbit.
    """
    A = A2k(k)
    old_lengths, old_shifts = predicted_lengths_and_shifts(k)
    # Invert old_lengths: map length → old orbit index.
    old_idx_of_length = {L: i for i, L in old_lengths.items()}
    seeds: dict[int, tuple[int, ...]] = {}
    for a in range(1, k + 1):
        old_i = old_idx_of_length[a + 1]
        seed_old = natural_seed(k, old_i)
        shift = old_shifts[old_i]
        # Apply ρ^{-shift} on the lattice (since old seed has chord
        # starting at vertex shift, we want chord starting at 0).
        cur = tuple(seed_old)
        for _ in range(shift):
            cur = tuple(A.rho_inverse(cur))
        seeds[a] = cur
    return seeds


if __name__ == "__main__":
    print("Closed-form prediction (verified at k = 2..5):")
    for k_pred in (2, 3, 4, 5, 6, 7):
        L_p, S_p = predicted_lengths_and_shifts(k_pred)
        print(f"  k = {k_pred}:  lengths = {L_p},  shifts = {S_p}")
    print()
    for k in (2, 3, 4):
        print(f"\n{'='*68}\n  [A_1, A_{2*k}]  ((2k+3) = {2*k+3}-gon)\n{'='*68}")
        # Try lengths = {i: i+1 for i in 1..k}
        result = search_consistent_shifts(k)
        if result is None:
            print("  NO consistent shifts found with lengths = (2, 3, ..., k+1)")
            # Search over all possible length assignments
            from itertools import permutations
            possible_lengths = list(range(2, k + 2))  # avoid length 1 = edges
            found = False
            for perm in permutations(possible_lengths):
                lengths = {i: L for i, L in zip(range(1, k + 1), perm)}
                result = search_consistent_shifts(k, lengths)
                if result is not None:
                    found = True
                    print(f"  Found with lengths permutation {perm}:")
                    break
            if not found:
                print("  No length permutation works either.")
                continue
        lengths, shifts, full_mism, q_mism = result
        print(f"  Length per orbit:  {lengths}")
        print(f"  ρ-shift per orbit (orbit 1 anchored at 0):  {shifts}")
        print(f"  Cross↔Plücker mismatches:  {full_mism}")
        print(f"  q-power = 2⟨,⟩ mismatches:  {q_mism}")
