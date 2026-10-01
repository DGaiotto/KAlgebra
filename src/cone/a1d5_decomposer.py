"""
a1d5_decomposer.py
==================

Cluster-monomial decomposition for A1D5KAlg: given a 5-tuple target
g-vector γ (with γ[-1] = 0 = the monomial part, χ-content stripped),
find atomic mult-gens whose g-vectors sum to γ and which all pairwise
q-commute (= live in a single cluster cone).

Used by `a1d5_cone_data.A1D5ConeData` to build `cross_product` daughter
words and to define `to_cone_label` / `from_cone_label`.
"""
from __future__ import annotations
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from itertools import combinations_with_replacement
from a1d5_kalg import (
    _T_ORBIT, _D_ORBIT, _V_ORBIT, _W_ORBIT,
    _PLUCKER_BASE_I0, _LATTICE_TO_MULTGEN,
)


# 20 atomic mult-gens labelled (kind, i).
ATOMIC_GENS: tuple = tuple(
    (K, i) for K in 'TDVW' for i in range(5)
)


def gvec(mg: tuple) -> tuple[int, ...]:
    """Atomic mult-gen → 5-tuple g-vector (last coord = 0)."""
    K, i = mg
    return {'T': _T_ORBIT, 'D': _D_ORBIT, 'V': _V_ORBIT, 'W': _W_ORBIT}[K][i % 5]


def _q_commute(mg1: tuple, mg2: tuple) -> bool:
    """Two atomic mult-gens q-commute iff their _PLUCKER_BASE_I0 entry
    has a single term (no Plücker)."""
    if mg1 == mg2:
        return True
    (k1, i1), (k2, i2) = mg1, mg2
    j = (i2 - i1) % 5
    entry = _PLUCKER_BASE_I0.get((k1, k2, j))
    return entry is not None and len(entry) == 1


def _qcommute_clique(mgs: tuple) -> bool:
    n = len(mgs)
    for i in range(n):
        for j in range(i + 1, n):
            if mgs[i] != mgs[j] and not _q_commute(mgs[i], mgs[j]):
                return False
    return True


def decompose(target: tuple, max_total_exp: int = 5) -> tuple | None:
    """Find a q-commuting multiset of atomic mult-gens whose g-vectors
    sum to `target`.  Returns a tuple of (kind, i) mult-gens (with
    repetition) in sorted order, or None if no decomposition exists
    within `max_total_exp` factors.

    Brute search over multisets of size 1..max_total_exp.  The cone
    constraint (pairwise q-commute) is checked before the lattice sum.
    """
    # Special: identity / zero vector (no factors).
    if target == (0, 0, 0, 0, 0):
        return ()
    # Atomic shortcut.
    if target in _LATTICE_TO_MULTGEN:
        K, i = _LATTICE_TO_MULTGEN[target]
        return ((K, i),)
    # Brute search.
    gens_sorted = sorted(ATOMIC_GENS)
    for n in range(2, max_total_exp + 1):
        for combo in combinations_with_replacement(gens_sorted, n):
            if not _qcommute_clique(combo):
                continue
            total = [0] * 5
            for mg in combo:
                gv = gvec(mg)
                for k in range(5):
                    total[k] += gv[k]
            if tuple(total) == target:
                return combo
    return None


def all_nonatomic_plucker_outputs():
    """Return the set of distinct non-atomic monomial-part g-vectors
    that appear in `_PLUCKER_BASE_I0`."""
    out = set()
    for terms in _PLUCKER_BASE_I0.values():
        for _coef, _q, lab in terms:
            # canonicalise: flip last coord if > 0
            mono = lab[:-1] + (0,)
            # Also handle case where lab[-1] > 0 (rare): canon flips,
            # mono = abs-value last coord but for atomic-detection we
            # use the mono part with 0 in last slot.
            if mono == (0, 0, 0, 0, 0):
                continue
            if mono in _LATTICE_TO_MULTGEN:
                continue
            out.add(mono)
    return out


if __name__ == "__main__":
    targets = all_nonatomic_plucker_outputs()
    print(f"Decomposing {len(targets)} non-atomic Plücker outputs...")
    n_ok = n_fail = 0
    examples = []
    for t in sorted(targets):
        d = decompose(t)
        if d is None:
            n_fail += 1
            if len(examples) < 5:
                examples.append((t, None))
        else:
            n_ok += 1
            if len(examples) < 5:
                examples.append((t, d))
    print(f"  OK: {n_ok}    Failed: {n_fail}")
    for t, d in examples:
        print(f"  {t} -> {d}")
