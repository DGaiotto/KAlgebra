"""Builder for the sharpened BPS-quiver dictionaries.

Walks the necklace-mutation + single-node-RG graph from a seed set
of UQuiver theories (`LinearUQuiver` / `CircularUQuiver` /
`DShapeUQuiver` / `E6/E7/E8UQuiver`) and registers every distinct
`(B, spec)` chart into a `SharpenedBPSQuiverDictionary`.

Identification predicate (`bpskalgebra_iso.find_isomorphism`): two
entries are merged iff there is a unimodular `A` intertwining their
pairings AND `A · spec_1` is local-move-reachable from `spec_2`.

Anywhere an exchange-matrix bucket holds two or more entries that
do *not* satisfy the iso predicate, we have *research-interest*
flagging: same quiver, two specs not related by local moves.

Outputs (relative to repo root):

  * `dictionaries/n_NNN.json` -- one file per node
    count, flat list of `QuiverEntry` dicts in the
    standard-basis convention (matches
    the archived tree shape).
  * `dictionaries/flags.json` -- list of
    research-interest collisions, each carrying the exchange
    matrix and the colliding spec list with provenance.
  * `dictionaries/build.log` -- summary of what was
    explored and how many entries / flags surfaced.

Run from the repo root:

    PYTHONPATH=. python dictionaries/build.py

CLI flags let you tune the bounds:
  --max-necklace : per-seed forward-necklace BFS depth (default 6)
  --max-rg-depth : single-node-RG recursion depth (default 2)
  --quiet        : skip per-step logging
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESTRUCT = os.path.dirname(_HERE)
_REPO = _RESTRUCT  # post-flatten: _RESTRUCT is the repo root
sys.path.insert(0, _REPO)
sys.path.insert(0, _RESTRUCT)

from bps_quiver_dictionary import QuiverEntry, signature as exchange_signature
from sharpened_dictionary import SharpenedBPSQuiverDictionary
from rg_flow import (
    decompose_nonneg_in_node_basis,
    decompose_nonneg_in_node_basis_bulk,
    subquiver_rg_flow,
    subquiver_rg_flow_raw,
)
from chart_graph import (
    _necklace_forward, _necklace_inverse, _bracket,
    _mutate_node_charges, _inverse_mutate_node_charges,
)
from quiver_constructors import (
    LinearUQuiver,
    CircularUQuiver,
    DShapeUQuiver,
    E6UQuiver, E7UQuiver, E8UQuiver,
)
from bps_kalgebra import BPSKAlgebra
import bps_quiver_tools as _bps


Vec = tuple[int, ...]


# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------


def _invert_unimodular_columns(cols):
    """Invert the matrix whose columns are `cols` (n vectors in Z^n).
    Returns the inverse as `dict[(i, j)] -> Fraction`, or `None` if
    singular.  Caller is responsible for ensuring the inverse is
    integer (= the cols are unimodular)."""
    n = len(cols)
    if n == 0 or len(cols[0]) != n:
        return None
    M = [[Fraction(cols[c][r]) for c in range(n)] for r in range(n)]
    inv = [[Fraction(1 if r == c else 0) for c in range(n)] for r in range(n)]
    for c in range(n):
        piv = None
        for r in range(c, n):
            if M[r][c] != 0:
                piv = r
                break
        if piv is None:
            return None
        if piv != c:
            M[c], M[piv] = M[piv], M[c]
            inv[c], inv[piv] = inv[piv], inv[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        inv[c] = [x / pv for x in inv[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [M[r][k] - f * M[c][k] for k in range(n)]
                inv[r] = [inv[r][k] - f * inv[c][k] for k in range(n)]
    return inv


def _restandardize(
    pairing: list[list[int]],
    nodes: list[Vec],
    spec: list[Vec],
) -> tuple[list[list[int]], list[Vec]] | None:
    """Re-express `(pairing, nodes, spec)` in the basis where `nodes`
    becomes the standard basis of an `n`-dim integer lattice.  Each
    spec entry must lie in the non-negative integer span of `nodes`
    (which is the case for UQuiver constructions by recipe).

    Returns `(B_std, spec_std)` (with implicit `node_charges_std =
    e_1, …, e_n`), or `None` if any spec entry can't be expressed.
    """
    n = len(nodes)
    if n == 0:
        return [[]], list(spec)
    rank = len(nodes[0])
    # B_std[i][j] = pairing on the i-th and j-th node charges.
    B_std = [
        [
            sum(
                pairing[a][b] * nodes[i][a] * nodes[j][b]
                for a in range(rank)
                for b in range(rank)
            )
            for j in range(n)
        ]
        for i in range(n)
    ]
    # Bulk-decompose the entire spec against `nodes` in one
    # Gauss-Jordan pass: the per-spec-entry decomposition is the
    # hot spot of the dictionary builder's mutation steps because
    # each entry was previously inverting `nodes` from scratch.
    decomps = decompose_nonneg_in_node_basis_bulk(nodes, spec)
    if decomps is None:
        return None
    spec_std: list[Vec] = []
    for d in decomps:
        if d is None:
            return None
        spec_std.append(tuple(d))
    return B_std, spec_std


def _entry_from_algebra(
    A: BPSKAlgebra, *, name: str, provenance: str,
) -> QuiverEntry | None:
    """Build a standard-basis QuiverEntry from a BPSKAlgebra."""
    pairing = [list(row) for row in A.lattice.pairing]
    nodes = [tuple(g) for g in A.node_charges]
    spec = [tuple(g) for g in A.spec]
    res = _restandardize(pairing, nodes, spec)
    if res is None:
        return None
    B_std, spec_std = res
    return QuiverEntry(
        name=name,
        exchange=tuple(tuple(r) for r in B_std),
        spec=tuple(spec_std),
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Necklace step in the standard-basis frame
# ---------------------------------------------------------------------------


def _necklace_step_entry(
    entry: QuiverEntry, *, direction: str = "forward",
) -> QuiverEntry | None:
    """Apply one necklace step to a standard-basis entry.

    `direction = "forward"`: spec rotates head→tail with sign flip;
    nodes mutate via FZ at the head's node.

    `direction = "inverse"`: spec rotates tail→head with sign flip;
    nodes inverse-mutate by `-tail`.

    Closed-form (no BFS).  After the step we re-standardize so the
    new `nodes` becomes the standard basis again.  Returns `None`
    if the precondition fails (head/tail not in the standard basis,
    or re-standardisation rejects the result).
    """
    if not entry.spec:
        return None
    n = entry.n_nodes
    pairing = [list(r) for r in entry.exchange]
    nodes = [tuple(1 if k == i else 0 for k in range(n)) for i in range(n)]
    spec = [tuple(g) for g in entry.spec]

    if direction == "forward":
        head = tuple(entry.spec[0])
        try:
            k0 = nodes.index(head)
        except ValueError:
            return None
        l = nodes[k0]
        u = nodes[k0]
        new_spec, new_nodes, _, _, _ = _necklace_forward(
            spec, nodes, l, u, pairing,
        )
        suffix = "mut+1"
    elif direction == "inverse":
        tail = tuple(entry.spec[-1])
        try:
            k0 = nodes.index(tail)
        except ValueError:
            return None
        l = nodes[k0]
        u = nodes[k0]
        new_spec, new_nodes, _, _, _ = _necklace_inverse(
            spec, nodes, l, u, pairing,
        )
        suffix = "mut-1"
    else:
        raise ValueError(f"direction must be 'forward' or 'inverse', got {direction!r}")

    res = _restandardize(pairing, new_nodes, new_spec)
    if res is None:
        return None
    B_new, spec_new = res
    return QuiverEntry(
        name=f"{entry.name}[{suffix}]",
        exchange=tuple(tuple(r) for r in B_new),
        spec=tuple(spec_new),
        provenance=f"necklace_{direction}({entry.name!r})",
    )


# ---------------------------------------------------------------------------
# General FZ mutation at an arbitrary node (not just spec head/tail)
# ---------------------------------------------------------------------------


def _fz_mutation_step_entry(
    entry: QuiverEntry, node_index: int, *,
    direction: str = "forward",
) -> QuiverEntry | None:
    """Apply a Fomin-Zelevinsky tropical mutation at `node_index`.

    The spec's operator-level expression (= product of `E_q`'s) is
    invariant under quiver mutation; only the *coordinates* of the
    spec entries change because the node-charge basis changes.  We
    re-standardize so the new chart again has standard-basis nodes.

    `direction = "forward"` is the standard FZ; `direction = "inverse"`
    is its inverse.

    Returns `None` if re-standardisation rejects the result (e.g.
    a spec entry now has fractional coordinates in the new basis).
    """
    n = entry.n_nodes
    if not (0 <= node_index < n):
        return None
    pairing = [list(r) for r in entry.exchange]
    nodes = [tuple(1 if k == i else 0 for k in range(n)) for i in range(n)]
    spec = [tuple(g) for g in entry.spec]

    if direction == "forward":
        new_nodes = _mutate_node_charges(list(nodes), nodes[node_index], pairing)
        suffix = f"fz+{node_index}"
    elif direction == "inverse":
        neg_node = tuple(-x for x in nodes[node_index])
        new_nodes = _inverse_mutate_node_charges(
            list(nodes), neg_node, pairing,
        )
        suffix = f"fz-{node_index}"
    else:
        raise ValueError(
            f"direction must be 'forward' or 'inverse', got {direction!r}"
        )

    res = _restandardize(pairing, new_nodes, spec)
    if res is None:
        return None
    B_new, spec_new = res
    return QuiverEntry(
        name=f"{entry.name}[{suffix}]",
        exchange=tuple(tuple(r) for r in B_new),
        spec=tuple(spec_new),
        provenance=f"fz_{direction}({entry.name!r}, k={node_index})",
    )


# ---------------------------------------------------------------------------
# Single-node-RG on a standard-basis entry
# ---------------------------------------------------------------------------


def _drop_node_entry(entry: QuiverEntry, j: int) -> QuiverEntry | None:
    """Return the entry obtained by single-node RG flow at index `j`,
    or `None` if the recipe doesn't apply.

    Operates entirely at the `(pairing, nodes, spec)` tuple level via
    `subquiver_rg_flow_raw` -- skips both `BPSKAlgebra` constructions
    (input wrap + output wrap) that the `BPSKAlgebra`-level wrapper
    `subquiver_rg_flow` would do.  Profile showed
    `BPSKAlgebra.__init__` accounting for ~70 % of build wall-clock;
    this refactor eliminates it from the RG path.
    """
    n = entry.n_nodes
    if n == 0:
        return None
    pairing = [list(r) for r in entry.exchange]
    # Standard-basis nodes for a QuiverEntry -- the build pass always
    # operates in this convention.
    nodes = [tuple(1 if k == i else 0 for k in range(n)) for i in range(n)]
    spec = [tuple(g) for g in entry.spec]
    try:
        new_pairing, new_nodes, new_spec, _ = subquiver_rg_flow_raw(
            pairing, nodes, spec, [j],
        )
    except (ValueError, RuntimeError, IndexError):
        # IndexError fires when the IR is 0-node (downstream code can't
        # handle empty node lists).
        return None
    if not new_nodes:
        return None
    # `_restandardize` re-expresses (pairing, new_nodes, new_spec) in
    # the standard basis convention so the resulting `QuiverEntry` is
    # consumable by the rest of the build pass without any
    # `BPSKAlgebra` wrapping.
    res = _restandardize(new_pairing, new_nodes, new_spec)
    if res is None:
        return None
    B_std, spec_std = res
    return QuiverEntry(
        name=f"{entry.name}[drop {j}]",
        exchange=tuple(tuple(r) for r in B_std),
        spec=tuple(spec_std),
        provenance=f"single_node_rg({entry.name!r}, j={j})",
    )


# ---------------------------------------------------------------------------
# Seed enumeration
# ---------------------------------------------------------------------------


def _seed_theories():
    """Yield seed `(name, BPSKAlgebra)` pairs from the UQuiver
    constructors with bounded parameters."""
    cases: list[tuple[str, callable]] = []

    # Single-factor U(N) (pure or with fundamentals), small N.
    for N1 in [2, 3, 4]:
        cases.append((f"linear_A1_U{N1}",
                      lambda N1=N1: LinearUQuiver([N1])))
    # SU(2) Nf, SU(3) Nf, SU(4) Nf -- all asymp-free combos with small n.
    cases.append(("linear_A1_U2_Nf1", lambda: LinearUQuiver([2], [1])))
    cases.append(("linear_A1_U2_Nf2", lambda: LinearUQuiver([2], [2])))
    cases.append(("linear_A1_U3_Nf1", lambda: LinearUQuiver([3], [1])))
    cases.append(("linear_A1_U3_Nf2", lambda: LinearUQuiver([3], [2])))
    cases.append(("linear_A1_U4_Nf1", lambda: LinearUQuiver([4], [1])))

    # Linear A_2 with N_i ∈ {2, 3} and lengths up to 3.
    for N1, N2 in [(2, 2), (2, 3), (3, 2)]:
        cases.append((f"linear_A2_U{N1}-U{N2}",
                      lambda N1=N1, N2=N2: LinearUQuiver([N1, N2])))
    # Linear A_2 with matter.
    cases.append(("linear_A2_U2-U2_Nf[1,0]",
                  lambda: LinearUQuiver([2, 2], [1, 0])))
    cases.append(("linear_A2_U2-U2_Nf[0,1]",
                  lambda: LinearUQuiver([2, 2], [0, 1])))
    cases.append(("linear_A2_U2-U2_Nf[1,1]",
                  lambda: LinearUQuiver([2, 2], [1, 1])))

    # Linear A_3.
    cases.append(("linear_A3_U2-U2-U2",
                  lambda: LinearUQuiver([2, 2, 2])))

    # Circular Â_{k-1}.
    cases.append(("circular_2_U2-U2", lambda: CircularUQuiver([2, 2])))
    cases.append(("circular_3_U2^3", lambda: CircularUQuiver([2, 2, 2])))
    cases.append(("circular_4_U2^4", lambda: CircularUQuiver([2, 2, 2, 2])))
    cases.append(("circular_3_U4^3", lambda: CircularUQuiver([4, 4, 4])))
    cases.append(("circular_2_U2-U2_Nf[1,0]",
                  lambda: CircularUQuiver([2, 2], [1, 0])))

    # D-shape (D_4 with Dynkin labels) and variants.
    cases.append(("D4_U[1,2,1,1]",
                  lambda: DShapeUQuiver([1, 2, 1, 1])))
    cases.append(("D4_U[2,3,2,2]",
                  lambda: DShapeUQuiver([2, 3, 2, 2])))
    cases.append(("D5_U[1,2,2,1,1]",
                  lambda: DShapeUQuiver([1, 2, 2, 1, 1])))

    # E_6 / E_7 / E_8 (Dynkin-label N's).
    cases.append(("E6_U[1,2,3,2,1,2]",
                  lambda: E6UQuiver([1, 2, 3, 2, 1, 2])))
    cases.append(("E7_U[2,3,4,3,2,1,2]",
                  lambda: E7UQuiver([2, 3, 4, 3, 2, 1, 2])))
    cases.append(("E8_U[2,4,6,5,4,3,2,3]",
                  lambda: E8UQuiver([2, 4, 6, 5, 4, 3, 2, 3])))

    for name, ctor in cases:
        try:
            U = ctor()
            yield name, U.algebra
        except Exception as exc:
            print(f"  SKIP seed {name!r}: {exc}", file=sys.stderr)


def _seed_theories_stable():
    """Yield seed ``(name, BPSKAlgebra)`` pairs with the *rewritten*
    Z-admitting spec when one is available from
    ``dictionaries/seed_stable_specs.json``.

    Falls back to the original constructor-produced spec for any seed
    whose record is missing or marked ``failed``/``todo``.  Skips none —
    every seed yielded by :func:`_seed_theories` is yielded here too,
    just possibly with a phase-orderable spec.

    Callers that want strict-only-stable behaviour can filter the
    output themselves.
    """
    try:
        import seed_stable_specs as _sss
    except ImportError:
        for name, A in _seed_theories():
            yield name, A
        return
    payload = _sss.load()
    by_name = {r["name"]: r for r in payload["entries"]}
    for name, A in _seed_theories():
        rec = by_name.get(name)
        if (rec is None
                or rec["bucket"] in ("failed", "todo")
                or rec["rewritten_spec"] is None):
            yield name, A
            continue
        rewritten = [tuple(g) for g in rec["rewritten_spec"]]
        if [tuple(g) for g in A.spec] == rewritten:
            yield name, A   # spec already matches; no rebuild needed
            continue
        # Rebuild the algebra with the rewritten spec.  All other data
        # (pairing, nodes, cone witness) is unchanged across the move
        # chain, so the new BPSKAlgebra shares structure with `A`.
        from bps_kalgebra import BPSKAlgebra
        try:
            A_stable = BPSKAlgebra(
                pairing=[list(r) for r in A.lattice.pairing],
                node_charges=A.node_charges,
                spec=rewritten,
                verify="off",
            )
        except Exception as exc:
            print(f"  WARN seed {name!r}: could not rebuild with "
                  f"rewritten spec: {exc}", file=sys.stderr)
            yield name, A
            continue
        yield name, A_stable


# ---------------------------------------------------------------------------
# Explore one seed
# ---------------------------------------------------------------------------


def _explore(
    A: BPSKAlgebra,
    seed_name: str,
    *,
    d: SharpenedBPSQuiverDictionary,
    edges: list[dict],
    max_necklace: int,
    max_rg_depth: int,
    fz_mutate_at_each_node: bool = False,
    quiet: bool = False,
) -> int:
    """Walk the necklace-mutation + single-node-RG graph from `A`,
    registering every visited chart in `d` and appending each
    successful mutation step (parent, child, kind, params) to
    `edges`.  Returns the number of charts visited."""
    seed_entry = _entry_from_algebra(
        A, name=seed_name, provenance=f"seed {seed_name!r}",
    )
    if seed_entry is None:
        if not quiet:
            print(f"  seed {seed_name}: failed to standardize")
        return 0

    visited_keys: set[tuple] = set()
    visited_count = 0

    def _key(entry: QuiverEntry) -> tuple:
        return (entry.exchange, entry.spec)

    def _edge(parent: QuiverEntry | None,
              child: QuiverEntry,
              kind: str,
              params: dict) -> None:
        if parent is None:
            return
        edges.append({
            "kind": kind,                        # "necklace" or "rg"
            "source_n": parent.n_nodes,
            "source_exchange": [list(r) for r in parent.exchange],
            "source_spec": [list(g) for g in parent.spec],
            "target_n": child.n_nodes,
            "target_exchange": [list(r) for r in child.exchange],
            "target_spec": [list(g) for g in child.spec],
            "params": params,
        })

    def _register_and_recurse(
        entry: QuiverEntry, rg_depth_left: int,
        parent: QuiverEntry | None = None,
        edge_kind: str | None = None,
        edge_params: dict | None = None,
    ) -> None:
        nonlocal visited_count
        k = _key(entry)
        is_new = k not in visited_keys
        if is_new:
            visited_keys.add(k)
            d.register(entry)
            visited_count += 1
        # Always log the edge (parent → entry), even if `entry` was
        # already visited -- the edge itself is fresh information.
        if parent is not None:
            _edge(parent, entry, edge_kind, edge_params or {})
        if not is_new:
            return

        # Single-node RG: drop each node, recurse with reduced depth.
        if rg_depth_left > 0:
            for j in range(entry.n_nodes):
                child = _drop_node_entry(entry, j)
                if child is not None:
                    _register_and_recurse(
                        child, rg_depth_left - 1,
                        parent=entry,
                        edge_kind="rg",
                        edge_params={"node_index": j},
                    )

    # Bidirectional BFS: necklace at head/tail of spec, and (if
    # `fz_mutate_at_each_node`) general FZ mutations at every node
    # index in both directions.  The chart-graph closure at small n
    # is then exhaustive in the strong sense.
    frontier = [seed_entry]
    _register_and_recurse(seed_entry, max_rg_depth)
    for step in range(max_necklace):
        next_frontier: list[QuiverEntry] = []
        for cur in frontier:
            children: list[tuple[QuiverEntry, str, dict]] = []
            for direction in ("forward", "inverse"):
                ch = _necklace_step_entry(cur, direction=direction)
                if ch is not None:
                    children.append(
                        (ch, f"necklace_{direction}",
                         {"step": step + 1})
                    )
            if fz_mutate_at_each_node:
                for j in range(cur.n_nodes):
                    for direction in ("forward", "inverse"):
                        ch = _fz_mutation_step_entry(
                            cur, j, direction=direction,
                        )
                        if ch is not None:
                            children.append(
                                (ch, f"fz_{direction}",
                                 {"step": step + 1, "node_index": j})
                            )
            for child, kind, params in children:
                already = _key(child) in visited_keys
                _register_and_recurse(
                    child, max_rg_depth,
                    parent=cur,
                    edge_kind=kind,
                    edge_params=params,
                )
                if not already:
                    next_frontier.append(child)
        frontier = next_frontier
        if not frontier:
            break
    return visited_count


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _flags_from_dict(d: SharpenedBPSQuiverDictionary) -> list[dict]:
    """Surface buckets with ≥ 2 non-iso-equivalent entries.

    Each flag carries the exchange matrix and the list of colliding
    specs with provenance.
    """
    flags: list[dict] = []
    for sig, bucket in d._buckets.items():
        if len(bucket) >= 2:
            flags.append({
                "n_nodes": bucket[0].n_nodes,
                "exchange_signature": [list(r) for r in sig],
                "colliding_entries": [
                    {
                        "name": e.name,
                        "exchange": [list(r) for r in e.exchange],
                        "spec": [list(g) for g in e.spec],
                        "provenance": e.provenance,
                    }
                    for e in bucket
                ],
            })
    return flags


def _high_multiplicity_flags(
    d: SharpenedBPSQuiverDictionary,
) -> list[dict]:
    """Surface every entry whose exchange matrix has any
    `|B_{ij}| > 2`.

    Conventional BPS quivers satisfy `|B_{ij}| ≤ 2` everywhere; an
    entry violating this is research-interesting (it sits outside
    the "standard" cluster-algebra regime, per the edge-multiplicity
    heuristic in `bps_quiver_tools`).
    """
    out: list[dict] = []
    for entry in d:
        n = entry.n_nodes
        max_mult = 0
        worst: tuple[int, int] | None = None
        for i in range(n):
            for j in range(i + 1, n):
                m = abs(entry.exchange[i][j])
                if m > max_mult:
                    max_mult = m
                    worst = (i, j)
        if max_mult > 2:
            out.append({
                "n_nodes": n,
                "name": entry.name,
                "exchange": [list(r) for r in entry.exchange],
                "spec": [list(g) for g in entry.spec],
                "max_multiplicity": max_mult,
                "worst_pair": list(worst) if worst else None,
                "provenance": entry.provenance,
            })
    return out


def _entry_payload(e: QuiverEntry) -> dict:
    return {
        "name": e.name,
        "exchange": [list(r) for r in e.exchange],
        "spec": [list(g) for g in e.spec],
        "negating_sequence": (
            list(e.negating_sequence) if e.negating_sequence else None
        ),
        "provenance": e.provenance,
        "known_specs": [
            [list(g) for g in s] for s in e.known_specs
        ],
    }


def _save_dict_by_n(
    d: SharpenedBPSQuiverDictionary, out_dir: Path,
) -> dict[int, int]:
    """Write `n_NNN.json` files; return `{n: count}` summary."""
    by_n: dict[int, list[QuiverEntry]] = {}
    for entry in d:
        by_n.setdefault(entry.n_nodes, []).append(entry)
    counts: dict[int, int] = {}
    for n, entries in by_n.items():
        path = out_dir / f"n_{n:03d}.json"
        with open(path, "w") as f:
            json.dump([_entry_payload(e) for e in entries], f, indent=2)
        counts[n] = len(entries)
    return counts


def _save_graph_by_n(
    d: SharpenedBPSQuiverDictionary,
    edges: list[dict],
    out_dir: Path,
) -> tuple[dict[int, int], int]:
    """Write `graph_nNNN.json` per node count (entries + within-n
    necklace edges) and `rg_edges.json` (cross-n RG edges).

    Mirrors the shape of the archived tree
    (`{entries, edges}`) for the within-n part, with an explicit
    cross-n companion file for the RG-flow edges.

    Returns `({n: edge_count_within_n}, n_rg_edges)`.
    """
    # Group entries by n.
    by_n: dict[int, list[QuiverEntry]] = {}
    for entry in d:
        by_n.setdefault(entry.n_nodes, []).append(entry)
    # Group edges.
    within_n_edges: dict[int, list[dict]] = {}
    cross_n_edges: list[dict] = []
    for ed in edges:
        if ed["source_n"] == ed["target_n"]:
            within_n_edges.setdefault(ed["source_n"], []).append(ed)
        else:
            cross_n_edges.append(ed)
    # graph_nNNN.json per n.
    edge_counts: dict[int, int] = {}
    for n, entries in by_n.items():
        edges_for_n = within_n_edges.get(n, [])
        path = out_dir / f"graph_n{n:03d}.json"
        with open(path, "w") as f:
            json.dump({
                "entries": [_entry_payload(e) for e in entries],
                "edges": edges_for_n,
            }, f, indent=2)
        edge_counts[n] = len(edges_for_n)
    # rg_edges.json: cross-n.
    with open(out_dir / "rg_edges.json", "w") as f:
        json.dump(cross_n_edges, f, indent=2)
    return edge_counts, len(cross_n_edges)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-necklace", type=int, default=6,
                   help="forward-necklace BFS depth per seed (default 6)")
    p.add_argument("--max-rg-depth", type=int, default=2,
                   help="single-node-RG recursion depth (default 2)")
    p.add_argument("--max-node-count", type=int, default=8,
                   help="skip seeds whose root algebra has more nodes "
                        "than this (default 8); iso-witnessing on larger "
                        "entries is too slow for a build pass")
    p.add_argument("--iso-max-states", type=int, default=512,
                   help="`find_isomorphism` state-search budget (default 512)")
    p.add_argument("--iso-max-extra", type=int, default=2,
                   help="`find_isomorphism` extra-length budget (default 2)")
    p.add_argument("--exhaustive-fz-up-to", type=int, default=0,
                   help="for seeds with n_nodes ≤ this, use general "
                        "FZ mutations at every node (not just necklace "
                        "head/tail).  Set to 0 (default) to disable.  "
                        "Larger values substantially expand the chart "
                        "graph closure but are only tractable for "
                        "small n.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    out_dir = Path(_HERE)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = SharpenedBPSQuiverDictionary(
        iso_max_states=args.iso_max_states,
        iso_max_extra_length=args.iso_max_extra,
    )
    edges: list[dict] = []
    log_lines: list[str] = []

    t0 = time.time()
    for seed_name, A in _seed_theories():
        t_seed = time.time()
        if len(A.node_charges) > args.max_node_count:
            line = (
                f"  seed {seed_name}: skipped (n_nodes="
                f"{len(A.node_charges)} > max-node-count="
                f"{args.max_node_count})"
            )
            log_lines.append(line)
            if not args.quiet:
                print(line)
            continue
        use_fz = (
            args.exhaustive_fz_up_to > 0
            and len(A.node_charges) <= args.exhaustive_fz_up_to
        )
        n_visited = _explore(
            A, seed_name,
            d=d, edges=edges,
            max_necklace=args.max_necklace,
            max_rg_depth=args.max_rg_depth,
            fz_mutate_at_each_node=use_fz,
            quiet=args.quiet,
        )
        dt = time.time() - t_seed
        line = (
            f"  seed {seed_name}: {n_visited} charts visited "
            f"in {dt:.1f}s"
        )
        log_lines.append(line)
        if not args.quiet:
            print(line)

    counts = _save_dict_by_n(d, out_dir)
    edge_counts, n_rg_edges = _save_graph_by_n(d, edges, out_dir)
    flags = _flags_from_dict(d)
    with open(out_dir / "flags.json", "w") as f:
        json.dump(flags, f, indent=2)
    mult_flags = _high_multiplicity_flags(d)
    with open(out_dir / "multiplicity_flags.json", "w") as f:
        json.dump(mult_flags, f, indent=2)

    summary = [
        f"=== sharpened-dictionary build complete ===",
        f"Total charts: {len(d)}.",
        f"Per-n counts: {sorted(counts.items())}.",
        f"Per-n within-n edges: {sorted(edge_counts.items())}.",
        f"Cross-n RG edges: {n_rg_edges}.",
        f"Research-flagged buckets: {len(flags)}.",
        f"High-|B_ij| flagged entries: {len(mult_flags)}.",
        f"Total time: {time.time() - t0:.1f}s.",
        f"Bounds: max_necklace={args.max_necklace}, "
        f"max_rg_depth={args.max_rg_depth}, "
        f"exhaustive_fz_up_to={args.exhaustive_fz_up_to}.",
    ]
    log_text = "\n".join(log_lines + [""] + summary)
    with open(out_dir / "build.log", "w") as f:
        f.write(log_text + "\n")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
