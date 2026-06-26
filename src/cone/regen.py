"""Regeneration + isomorphism tools for the ``finite_kalgebras`` package.

The frozen standalones (``finite_pentagon_kalg.py``, …) embed the
multiplication data for one specific finite ConeKAlgebra.  This module
provides the safety net Phase 0:

* :func:`rebuild_data(short_id)` — re-runs the FiniteBPSKAlgebra build
  pipeline from the BPS pairing + node-charges literals embedded in the
  standalone, and returns the resulting cone-algebra data dict.
* :func:`native_data(short_id)` — reads the same data dict directly from
  the frozen standalone (no rebuild).
* :func:`kalgebra_isomorphism(data_a, data_b)` — searches for a
  permutation ``π`` of mg-indices that intertwines cones, cocycle, ρ,
  and cross-table.  Returns the permutation as a dict if one exists,
  ``None`` otherwise.

This is what we call when we want to verify that a freshly-rebuilt
algebra is the same KAlgebra (up to canonical-basis relabeling) as the
frozen standalone.
"""
from __future__ import annotations

import importlib
from typing import Optional


# short_id → (PREFIX, standalone_module_name, flavor, max_charts)
REGEN_SPECS: dict[str, tuple[str, str, str, int]] = {
    "pentagon": ("PENTAGON", "finite_pentagon_kalg", "trivial",  2000),
    "heptagon": ("HEPTAGON", "finite_heptagon_kalg", "trivial",  2000),
    "a3":       ("A3",       "finite_a3_kalg",       "u1",       2000),
    "hexagon":  ("A3",       "finite_a3_kalg",       "u1",       2000),
    "a5":       ("A5",       "finite_a5_kalg",       "u1",       5000),
    "octagon":  ("A5",       "finite_a5_kalg",       "u1",       5000),
    "a7":       ("A7",       "finite_a7_kalg",       "u1",      10000),
    "decagon":  ("A7",       "finite_a7_kalg",       "u1",      10000),
    "a1d3":     ("A1D3",     "finite_a1d3_kalg",     "su2",      2000),
    "a1d4":     ("A1D4",     "finite_a1d4_kalg",     "su2u1",    2000),
    "a1d5":     ("A1D5",     "finite_a1d5_kalg",     "su2",      2000),
    "a1d6":     ("A1D6",     "finite_a1d6_kalg",     "su2u1",    5000),
    "a1d7":     ("A1D7",     "finite_a1d7_kalg",     "su2",     10000),
    "a1d8":     ("A1D8",     "finite_a1d8_kalg",     "su2u1",   20000),
    "e6":       ("E6",       "finite_e6_kalg",       "trivial", 20000),
    "e7":       ("E7",       "finite_e7_kalg",       "u1",      20000),
    "e8":       ("E8",       "finite_e8_kalg",       "trivial", 50000),
}


def _spec(short_id: str) -> tuple[str, str, str, int]:
    if short_id not in REGEN_SPECS:
        raise KeyError(
            f"Unknown short_id {short_id!r}; expected one of "
            f"{sorted(REGEN_SPECS)}."
        )
    return REGEN_SPECS[short_id]


def _load_standalone(short_id: str):
    """Import the standalone module + return (module, prefix)."""
    prefix, modname, _, _ = _spec(short_id)
    return importlib.import_module(modname), prefix


def _normalize_cross_entry(entry):
    """Normalize a cross-table entry to the modern 3-tuple form.

    Older standalones (E_6) used 2-tuples ``(coeffs, word)`` for
    LaurentPoly-only entries; the modern form is ``(kind, data, word)``
    with ``kind ∈ {'LP', 'RL'}``.
    """
    out = []
    for t in entry:
        if len(t) == 3:
            out.append(t)
        elif len(t) == 2:
            coeffs, word = t
            out.append(("LP", coeffs, word))
        else:
            raise ValueError(f"unrecognized cross-entry tuple shape: {t!r}")
    return out


# Standalones not currently covered by the regen check:
# - "e8":  cross-table is ρ-orbit-reduced (``CROSS_TABLE_REDUCED``),
#          which we don't yet expand.
# (a1d4 was here: the cluster BFS gives only 3 gauge cone rays — the D_4
#  gauge cone is not ρ-closed, since R = SU(2)×U(1) is the Z_2 that D_4's
#  S_3 triality extends.  RESOLVED by the ρ-orbit completion in
#  `cluster_cone_builder.to_cone_kalgebra` (complete gens + cones under ρ
#  → the two length-4 σ-orbits = the honest 8 mg's; cross-products
#  decomposed over the completed cones).  a1d4 now regenerates in the
#  standard expanded form like its A1D_even siblings.)
REGEN_UNSUPPORTED: set[str] = {"e8"}


def native_data(short_id: str) -> dict:
    """Read the frozen cone-algebra data dict from the standalone module."""
    if short_id in REGEN_UNSUPPORTED:
        raise NotImplementedError(
            f"native_data({short_id!r}): standalone stores cross-table in a "
            f"reduced form not yet expanded by the regen path."
        )
    mod, prefix = _load_standalone(short_id)
    raw_cross = getattr(mod, f"{prefix}_CROSS_TABLE")
    cross = {k: _normalize_cross_entry(v) for k, v in raw_cross.items()}
    return {
        "mult_gens_lattice": tuple(getattr(mod, f"{prefix}_MULT_GENS_LATTICE")),
        "cones":             tuple(
            tuple(sorted(c)) for c in getattr(mod, f"{prefix}_CONES")
        ),
        "cocycle_table":     dict(getattr(mod, f"{prefix}_COCYCLE_TABLE")),
        "rho_perm":          dict(getattr(mod, f"{prefix}_RHO_PERM")),
        "cross_table":       cross,
    }


def rebuild_data(short_id: str, *, verbose: bool = False) -> dict:
    """Re-run the FiniteBPSKAlgebra build pipeline for ``short_id`` and
    return its cone-algebra data dict (same shape as :func:`native_data`,
    minus the ``cone_canon`` field which is internal to the build).

    Uses the BPS pairing + node-charges embedded in the standalone as
    the input spec — so a successful round-trip ``native_data ↔
    rebuild_data`` is a true regression check.
    """
    mod, prefix = _load_standalone(short_id)
    _, _, flavor, max_charts = _spec(short_id)
    pairing = getattr(mod, f"{prefix}_BPS_PAIRING")
    node_charges = getattr(mod, f"{prefix}_BPS_NODE_CHARGES")
    from generate_finite_kalg import build_finite_kalg_data
    data = build_finite_kalg_data(
        pairing=pairing,
        node_charges=node_charges,
        flavor=flavor,
        max_charts=max_charts,
        prefix=prefix,
        verbose=verbose,
    )
    return {
        "mult_gens_lattice": data["mult_gens_lattice"],
        "cones":             tuple(tuple(sorted(c)) for c in data["cones"]),
        "cocycle_table":     dict(data["cocycle_table"]),
        "rho_perm":          dict(data["rho_perm"]),
        "cross_table":       dict(data["cross_table"]),
    }


# --------------------------------------------------------------------
# KAlgebra isomorphism (permutation of canonical basis).
# --------------------------------------------------------------------

def _validate_rho_bijection(rho_perm: dict[int, int], n: int) -> None:
    """ρ is an algebra automorphism, so its action on mg-indices must
    be a bijection on ``range(n)`` (missing keys = fixed points).
    Raise ``ValueError`` with a precise message if not.

    Known offender: ``finite_a1d4_kalg.py`` stores ``{0: 2}`` only,
    which yields ρ(0)=2 and ρ(2)=2 — not a permutation.  Root cause
    is in ``cluster_cone_builder.py`` (BPS step 4 silently skips
    mg's whose ρ-image isn't found in the canonical-lift lookup).
    """
    images = [rho_perm.get(i, i) for i in range(n)]
    if sorted(images) != list(range(n)):
        bad = [i for i in range(n)
               if images.count(images[i]) > 1]
        raise ValueError(
            f"rho_perm is not a bijection on range({n}): "
            f"images={images}, collisions at {sorted(set(bad))}. "
            f"This indicates a bug in the BPS-pipeline rho_perm "
            f"computation; see cluster_cone_builder.py step 4."
        )


def _rho_cycles(
    rho_perm: dict[int, int], n: int,
) -> list[tuple[int, ...]]:
    """Decompose ρ-permutation on ``range(n)`` into disjoint cycles.

    Missing keys in ``rho_perm`` are treated as fixed points (identity).
    Each cycle is rotated so its smallest element comes first.

    Caller is expected to have validated ρ as a bijection via
    :func:`_validate_rho_bijection`.
    """
    def rho(i):
        return rho_perm.get(i, i)
    seen: set[int] = set()
    out: list[tuple[int, ...]] = []
    for i in range(n):
        if i in seen:
            continue
        cyc = [i]
        seen.add(i)
        j = rho(i)
        while j != i:
            cyc.append(j)
            seen.add(j)
            j = rho(j)
        m = min(cyc)
        k = cyc.index(m)
        cyc = cyc[k:] + cyc[:k]
        out.append(tuple(cyc))
    out.sort(key=lambda c: (len(c), c))
    return out


def _mg_signature(
    i: int,
    n: int,
    cones: tuple[tuple[int, ...], ...],
    cocycle: dict[tuple[int, int], int],
    orbit_len: int,
) -> tuple:
    """Rotation-invariant local invariant of mg ``i``.

    Contains: ρ-orbit length, number of cones containing ``i``, multiset
    of (cone-size, sorted-cocycle-tuple) over those cones.
    """
    cone_data = []
    for c in cones:
        if i not in c:
            continue
        sig_c = tuple(sorted(cocycle.get((i, j), 0) for j in c if j != i))
        cone_data.append((len(c), sig_c))
    cone_data.sort()
    return (orbit_len, len(cone_data), tuple(cone_data))


def _verify_permutation(
    perm: dict[int, int], data_a: dict, data_b: dict,
) -> bool:
    """Check that ``perm`` is a KAlgebra isomorphism: it intertwines
    cones (as sets), cocycle, ρ, and cross_table."""
    inv = {v: k for k, v in perm.items()}
    if len(inv) != len(perm):
        return False

    # Cones (as sets) must map to cones (as sets).
    target_cones = {frozenset(c) for c in data_b["cones"]}
    for c in data_a["cones"]:
        mapped = frozenset(perm[i] for i in c)
        if mapped not in target_cones:
            return False

    # Cocycle
    cocy_a = data_a["cocycle_table"]
    cocy_b = data_b["cocycle_table"]
    for (i, j), v in cocy_a.items():
        if cocy_b.get((perm[i], perm[j])) != v:
            return False
    # Reverse direction (catches keys present only in B).
    for (i, j), v in cocy_b.items():
        if cocy_a.get((inv[i], inv[j])) != v:
            return False

    # ρ-permutation: π must commute with ρ on the *raw* rho_perm dict
    # (taking missing keys as fixed points), so that ρ_A and ρ_B agree
    # as algebra automorphisms.  Use rho_perm.get(i, i).
    rho_a = data_a["rho_perm"]
    rho_b = data_b["rho_perm"]
    n = len(data_a["mult_gens_lattice"])
    for i in range(n):
        ri = rho_a.get(i, i)
        rpi = rho_b.get(perm[i], perm[i])
        if rpi != perm[ri]:
            return False

    # Cross-product table
    cross_a = data_a["cross_table"]
    cross_b = data_b["cross_table"]
    for (i, j), entry_a in cross_a.items():
        key_b = (perm[i], perm[j])
        entry_b = cross_b.get(key_b, [])
        if not _cross_entry_eq(entry_a, entry_b, perm, cocy_a, cocy_b):
            return False
    return True


def _canonicalize_word(word, cocycle_table):
    """Bubble-sort ``word`` (a tuple of mg-indices) to ascending order,
    accumulating q-phase ``2·cocycle(i, j)`` for each adjacent swap of
    distinct gens.

    Returns ``(sorted_word, q_phase)`` so that the original word equals
    ``q^{q_phase}`` times the sorted word.
    """
    arr = list(word)
    phase = 0
    n = len(arr)
    for i in range(n):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                phase += 2 * cocycle_table.get((arr[j], arr[j + 1]), 0)
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return tuple(arr), phase


def _shift_q_data(kind, data, q_phase: int):
    """Shift coefficient by q^{q_phase}."""
    if q_phase == 0:
        return data
    if kind == "LP":
        return {e + q_phase: c for e, c in data.items()}
    if kind == "RL":
        return {q + q_phase: dict(r) for q, r in data.items()}
    raise TypeError(kind)


def _canonical_entry(entry, cocycle_table):
    """Canonicalize a cross-product entry: sort each word, fold q-phase
    into the coefficient, then return a hashable canonical form."""
    by_word: dict = {}
    for kind, data, word in entry:
        cw, phase = _canonicalize_word(word, cocycle_table)
        sd = _shift_q_data(kind, data, phase)
        # Merge with any existing term of same kind+word.
        key = (kind, cw)
        if key in by_word:
            existing = by_word[key]
            if kind == "LP":
                for e, c in sd.items():
                    existing[e] = existing.get(e, 0) + c
            elif kind == "RL":
                for q, rt in sd.items():
                    if q in existing:
                        d2 = existing[q]
                        for ch, c in rt.items():
                            d2[ch] = d2.get(ch, 0) + c
                    else:
                        existing[q] = dict(rt)
        else:
            by_word[key] = (
                dict(sd) if kind == "LP"
                else {q: dict(r) for q, r in sd.items()}
            )
    # Strip zeros and canonicalize.
    out = []
    for (kind, cw), d in by_word.items():
        if kind == "LP":
            d = {e: c for e, c in d.items() if c != 0}
            if not d:
                continue
            d_canon = tuple(sorted(d.items()))
        else:
            d2 = {}
            for q, rt in d.items():
                rt2 = {ch: c for ch, c in rt.items() if c != 0}
                if rt2:
                    d2[q] = rt2
            if not d2:
                continue
            d_canon = tuple(sorted(
                (q, tuple(sorted(rt.items()))) for q, rt in d2.items()
            ))
        out.append((kind, d_canon, cw))
    out.sort()
    return tuple(out)


def _cross_entry_eq(entry_a, entry_b, perm, cocycle_a, cocycle_b) -> bool:
    """Compare cross-table entries from A and B under permutation ``perm``.

    Apply ``perm`` to each word in ``entry_a``, then canonicalize both
    sides (sort each word ascending, fold q-phase into coefficient).
    """
    permuted_a = [
        (kind, data, tuple(perm[g] for g in word))
        for kind, data, word in entry_a
    ]
    # After permutation, A's words live in B's coordinate system, so
    # canonicalize against B's cocycle.
    return (_canonical_entry(permuted_a, cocycle_b)
            == _canonical_entry(entry_b, cocycle_b))


def kalgebra_isomorphism(
    data_a: dict, data_b: dict, *, max_attempts: int = 100000,
) -> Optional[dict[int, int]]:
    """Search for a KAlgebra isomorphism A → B as a permutation of
    multi-gen indices.  Returns the permutation dict if one exists,
    ``None`` otherwise.

    Uses ρ-orbit structure and mg-signatures to prune the search.
    """
    n_a = len(data_a["mult_gens_lattice"])
    n_b = len(data_b["mult_gens_lattice"])
    if n_a != n_b:
        return None
    cones_a = data_a["cones"]
    cones_b = data_b["cones"]
    if len(cones_a) != len(cones_b):
        return None
    # Cone-size histogram must match.
    from collections import Counter
    if (Counter(len(c) for c in cones_a)
            != Counter(len(c) for c in cones_b)):
        return None

    n = n_a
    rho_a = data_a["rho_perm"]
    rho_b = data_b["rho_perm"]
    _validate_rho_bijection(rho_a, n)
    _validate_rho_bijection(rho_b, n)
    orbits_a = _rho_cycles(rho_a, n)
    orbits_b = _rho_cycles(rho_b, n)
    if [len(c) for c in orbits_a] != [len(c) for c in orbits_b]:
        return None

    # mg signatures
    orbit_len_a = {}
    for cyc in orbits_a:
        for i in cyc:
            orbit_len_a[i] = len(cyc)
    orbit_len_b = {}
    for cyc in orbits_b:
        for i in cyc:
            orbit_len_b[i] = len(cyc)

    sig_a = {i: _mg_signature(i, n, cones_a, data_a["cocycle_table"],
                              orbit_len_a[i]) for i in range(n)}
    sig_b = {i: _mg_signature(i, n, cones_b, data_b["cocycle_table"],
                              orbit_len_b[i]) for i in range(n)}
    if Counter(sig_a.values()) != Counter(sig_b.values()):
        return None

    # Group orbits by their representative's signature
    # (all elements of an orbit share the same signature).
    def _orbit_sig(cyc, sig):
        return (len(cyc), sig[cyc[0]])

    a_orbits_by_sig: dict = {}
    for cyc in orbits_a:
        a_orbits_by_sig.setdefault(_orbit_sig(cyc, sig_a), []).append(cyc)
    b_orbits_by_sig: dict = {}
    for cyc in orbits_b:
        b_orbits_by_sig.setdefault(_orbit_sig(cyc, sig_b), []).append(cyc)
    if set(a_orbits_by_sig) != set(b_orbits_by_sig):
        return None
    for k_sig, lst_a in a_orbits_by_sig.items():
        if len(lst_a) != len(b_orbits_by_sig[k_sig]):
            return None

    # Search: for each signature class, try every (orbit-perm × cyclic-shift)
    # assignment of A's orbits to B's orbits.
    from itertools import permutations, product
    attempts = 0

    sig_classes = sorted(a_orbits_by_sig)

    def assignments():
        per_class = []
        for s in sig_classes:
            a_list = a_orbits_by_sig[s]
            b_list = b_orbits_by_sig[s]
            L = len(a_list[0])
            class_options = []
            for b_perm in permutations(range(len(b_list))):
                for shifts in product(range(L), repeat=len(a_list)):
                    class_options.append((b_perm, shifts))
            per_class.append(class_options)
        for combo in product(*per_class):
            yield combo

    for combo in assignments():
        attempts += 1
        if attempts > max_attempts:
            return None
        perm: dict[int, int] = {}
        ok = True
        for s, (b_perm, shifts) in zip(sig_classes, combo):
            a_list = a_orbits_by_sig[s]
            b_list = b_orbits_by_sig[s]
            L = len(a_list[0])
            for k, a_cyc in enumerate(a_list):
                b_cyc = b_list[b_perm[k]]
                sh = shifts[k]
                for idx, a_elem in enumerate(a_cyc):
                    perm[a_elem] = b_cyc[(idx + sh) % L]
            if not ok:
                break
        if not ok:
            continue
        if len(set(perm.values())) != n:
            continue
        if _verify_permutation(perm, data_a, data_b):
            return perm
    return None
