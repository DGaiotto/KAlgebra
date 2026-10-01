"""Builder for the FLAVOURED BPS-quiver dictionary.

The author's request, 2026-09-14: *"Code a flavoured version of the BPS quiver
dictionary.  For example, I could specify SU(2) and expect a dictionary of
quivers which have exactly one node doublet, with SU(2)-covariant specs when
available and otherwise SU(2)-covariant crystal S's"*.

WHAT AN ENTRY IS
----------------

A **derived** dictionary over the enumerated one (`dictionaries/enumerated/`),
not a second enumeration.  The entry is an ordinary BPS quiver `B` from that
set which **has a node doublet**: a transposition of two nodes that is an
automorphism of `B`, so the two are identical and permutable and the theory
carries a manifest `SU(2)` rotating them (the author's *"as soon as you have N
identical nodes in the quiver, the theory has manifest SU(N) symmetry"*).
`SU(2)` is `doublets = 1`; the same code selects `∏_a SU(N_a)` by asking for
other node-orbit shapes, which is why the selector is a parameter.

Keying on the unfolded quiver — the honest BPS quiver of the theory — rather
than on the folded one buys two things.  Cells line up `(n, e)` with the
unflavoured dictionary, and **every entry is by construction already in it**,
so the flavoured dictionary is a refinement whose membership is checkable
against a set built by a different route.

THE VALUE, in the author's order of preference
--------------------------------------------

  1. an **`SU(2)`-covariant spec** where one is found — a flavoured spec in
     the sense of `flavoured_spec`, whose factors for one irrep are
     consecutive, so the whole factorisation is a product of
     `E^{(s;r)}_𝖖(X_γ) = ∏_{w∈r} E^{(s)}_𝖖(μ^w X_γ)` and the flavour
     symmetry is manifest factor by factor;
  2. otherwise the **`SU(2)`-covariant crystal `S`** to a stated depth —
     `Ω(γ, r)`, a palindromic `𝖖`-polynomial per irrep at each reduced charge
     (`FlavouredFactorSpectrum.multiplicities`).  That this is a *character*
     at every charge is the substantive flavoured claim, so an entry that
     honest-fails there is recorded as such rather than dropped.

ACYCLIC IS FREE, AND IT IS THE BULK.  On an acyclic flavoured quiver the
source/sink (strip) order is a flavoured spec attaining the floor — one
generator per reduced node — so those entries are certified without a search.
Note the shipped shards **code** the acyclic quivers rather than storing them
(`SHARD_FORMAT = 2`), so a scan that forgets `include_acyclic=True` silently
omits them — and with them the `[A_1, D_3]` anchor, the canonical one-`SU(2)`
example.  `flavoured_entries` passes the flag.

CERTIFICATION STRINGS, mirroring the unflavoured builder's discipline of
recording exactly what held:

  * `covariant_spec:strip`        — acyclic; the strip order IS a spec (floor).
  * `covariant_spec:order@C`      — found by the order search, confirmed to
                                    cone degree `C` (`FlavouredSpec.confirmed_to`).
  * `covariant_spec:blocks`       — found by the block-mutation bidirectional
                                    BFS (`find_flavoured_negating_sequence`),
                                    the cluster-side definition.
  * `crystalline_only@D`          — no covariant spec found; `Ω(γ, r)` to
                                    depth `D`.
  * `crystalline_failed@D`        — `Ω` is not a character at some charge.
                                    A conjecture counterexample: recorded
                                    loudly, never dropped.

WHAT THE SHIPPED TIER MEASURES.  `dictionaries/flavoured/` is the `SU(2)` set
(`doublets = 1`), complete through total arrow weight `≤ 9` with four complete
weight-10 cells beside it: **156,225 quivers**, of which 155,937 carry an
`SU(2)`-covariant spec — 128,703 by the acyclic construction (coded, not stored),
24,585 by the block BFS and 2,649 by the order search — and **288** fall through
to `crystalline_only@3`.  The manifest marks weight 10 PARTIAL and names the
cells; continuing it is the design record.  **No entry is
`crystalline_failed`**, so on this set `Ω` is a `G`-character at every charge
reached, which is the substantive flavoured claim
(`flavoured_s_axioms.md` §3) holding on 139,371 quivers.

Every one of the 27,522 stored specs was re-checked against `S` on a cone
strictly wider than the search's by
a probe in the source repository — 0 failures at depth 4
(2026-09-15/16); a sample of them is
re-checked in CI (the suite in the source repository).
Reading the shipped tier goes through `dictionary_loader`
(`default_flavoured_dictionary_dir`, `iter_flavoured_entries`,
`lookup_flavoured`) — a consumer should not have to import this builder.

An earlier version of this docstring reported the weight-`≤ 7` figures and said
the crystalline branch was "never reached by the build"; at weight 9 it is
reached 53 times, so that reading is retracted here.

Run:  PYTHONPATH=. python dictionaries/build_flavoured.py --max-weight 6
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.path.join(_REPO, "implementations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quiver_enumeration import canonical_form, is_acyclic          # noqa: E402
from flavoured_factor_spectrum import (                            # noqa: E402
    FlavouredQuiver, FlavouredFactorSpectrum,
)
from flavoured_spec import (                                       # noqa: E402
    find_flavoured_spec, find_flavoured_negating_sequence,
    spec_from_negating_sequence,
)

Matrix = tuple[tuple[int, ...], ...]
SHARD_FORMAT = 2


# --------------------------------------------------------------------------
# node doublets: which quivers carry a manifest SU(2)
# --------------------------------------------------------------------------

def node_doublets(B: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    """Pairs `(i, j)` whose TRANSPOSITION is an automorphism of `B`.

    Those two nodes are identical and permutable, so the theory has a manifest
    `SU(2)` rotating them and the pair is a doublet.  The pair must also have
    `B[i][j] = 0`: the two doublet weights span an isotropic direction, which
    is what lets the flavour weight lattice sit in `ker B` as a factor of the
    charge lattice (the 2026-09-14 ruling).  A pair with `B[i][j] ≠ 0` is not a
    doublet however symmetric it looks, so the test is part of the definition
    and not an optimisation.
    """
    n = len(B)
    out: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if B[i][j] != 0:
                continue
            swap = lambda k, i=i, j=j: j if k == i else (i if k == j else k)
            if all(B[swap(k)][swap(l)] == B[k][l]
                   for k in range(n) for l in range(n)):
                out.append((i, j))
    return out


def node_orbits(B: Sequence[Sequence[int]], size: int) -> list[tuple[int, ...]]:
    """Sets of `size` nodes that are mutually identical and permutable.

    The `SU(N)` generalisation of `node_doublets`, which is exactly the
    `size = 2` case (asserted in the suite in the source repository).  A set
    qualifies when

    * its nodes **pairwise do not pair**, `B[i][j] = 0` — the weights of the
      `N` copies must span an isotropic subspace, which is what lets the
      flavour weight lattice sit in `ker B` as a factor of the charge lattice
      (the 2026-09-14 ruling, stated there for `N = 2`); and
    * **every** permutation of the set is an automorphism of `B`.

    Only the transpositions have to be tested, since they generate the
    symmetric group and the automorphisms are a group; and within the set the
    charge shift of a mutation vanishes (the pairings are zero), so the copies
    really are interchangeable.  Checking generators rather than all `size!`
    permutations is what keeps this cheap at `N = 3, 4`.

    ⚠ **A `size`-set is NOT `size` doublets.**  Three identical nodes contain
    three doublet PAIRS, so a selector counting doublets would put an `SU(3)`
    quiver in the "3 doublets" bucket beside a quiver with three unrelated
    pairs.  The two are different objects, which is why the selector is the
    orbit size and not a doublet count.
    """
    import itertools

    n = len(B)
    if size < 2 or size > n:
        return []
    out: list[tuple[int, ...]] = []
    for combo in itertools.combinations(range(n), size):
        if any(B[i][j] != 0 for i, j in itertools.combinations(combo, 2)):
            continue
        ok = True
        for i, j in itertools.combinations(combo, 2):
            swap = lambda k, i=i, j=j: j if k == i else (i if k == j else k)
            if not all(B[swap(k)][swap(l)] == B[k][l]
                       for k in range(n) for l in range(n)):
                ok = False
                break
        if ok:
            out.append(combo)
    return out


def fold_orbit(B: Sequence[Sequence[int]],
               orbit: Sequence[int]) -> FlavouredQuiver:
    """Quotient `B` by a node orbit — the flavoured quiver it unfolds from.

    Keep the first node of the orbit and drop the rest; the survivor carries
    the fundamental of `SU(len(orbit))`.  `fold_doublet` is the `len == 2`
    case.  Inverse to `FlavouredQuiver.unfold()` up to node permutation.
    """
    orbit = sorted(int(x) for x in orbit)
    keep_node, drop = orbit[0], set(orbit[1:])
    keep = [k for k in range(len(B)) if k not in drop]
    reduced = [[int(B[a][b]) for b in keep] for a in keep]
    slot = keep.index(keep_node)
    factors = [len(orbit) if t == slot else 1 for t in range(len(keep))]
    return FlavouredQuiver(reduced, factors)


def unfold_orbit_at(B: Sequence[Sequence[int]], a: int, size: int) -> list[list[int]]:
    """Replace node `a` of `B` by `size` identical copies.

    `unfold_at` is the `size = 2` case.  The copies pair to zero with one
    another and identically with every other node, so they are a node orbit of
    the result and `fold_orbit` inverts this.  The weight grows by
    `(size - 1) * deg(a)`, so on a connected quiver with more than one node an
    `SU(N)` unfolding of weight `<= E` comes from a fold of weight
    `<= E - (N - 1)`.
    """
    n = len(B)
    extra = size - 1
    out = [[int(B[i][j]) for j in range(n)] + [int(B[i][a])] * extra
           for i in range(n)]
    for _ in range(extra):
        out.append([int(B[a][j]) for j in range(n)] + [0] * extra)
    for t in range(n, n + extra):
        out[a][t] = 0
        out[t][a] = 0
    return out


def fold_doublet(B: Sequence[Sequence[int]],
                 pair: tuple[int, int]) -> FlavouredQuiver:
    """Quotient `B` by a node doublet — the flavoured quiver it unfolds from.

    Drop one of the two nodes; the survivor carries the `SU(2)` fundamental.
    Inverse to `FlavouredQuiver.unfold()` up to node permutation, asserted in
    the suite in the source repository.
    """
    i, j = pair
    keep = [k for k in range(len(B)) if k != j]
    reduced = [[int(B[a][b]) for b in keep] for a in keep]
    slot = keep.index(i)
    factors = [2 if t == slot else 1 for t in range(len(keep))]
    return FlavouredQuiver(reduced, factors)


def node_orbit_families(B: Sequence[Sequence[int]],
                        sizes: Sequence[int]) -> list[tuple[tuple[int, ...], ...]]:
    """Families of **pairwise disjoint** node orbits with the given sizes.

    `sizes = (2,)` is one `SU(2)`, `(3,)` one `SU(3)`, `(2, 2)` an `SU(2) x
    SU(2)` — two disjoint pairs, which is a different object from one orbit of
    size 4 (four mutually identical nodes) and from three overlapping pairs
    inside a triple.  Returned sorted, each family as a tuple of orbits in
    increasing order of their first node, so the list is a canonical answer.

    Disjointness is imposed, not hoped for.  It is also automatic for equal
    sizes: if swapping `(i, j)` and swapping `(i, k)` are both automorphisms
    then so is their composite, a 3-cycle, so `(j, k)` is an orbit too and the
    three nodes are one orbit of size 3 rather than two overlapping pairs.  The
    code still checks, because the argument does not cover unequal sizes.
    """
    import itertools

    pools = [node_orbits(B, k) for k in sizes]
    if any(not pool for pool in pools):
        return []
    out: list[tuple[tuple[int, ...], ...]] = []
    seen: set = set()
    for combo in itertools.product(*pools):
        nodes: set[int] = set()
        ok = True
        for orb in combo:
            if nodes & set(orb):
                ok = False
                break
            nodes |= set(orb)
        if not ok:
            continue
        key = tuple(sorted(combo))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return sorted(out)


def fold_orbits(B: Sequence[Sequence[int]],
                family: Sequence[Sequence[int]]) -> FlavouredQuiver:
    """Quotient `B` by SEVERAL disjoint node orbits at once.

    Keep the first node of each orbit and drop the rest; each survivor carries
    the fundamental of `SU(len(orbit))`, so the flavour group of the fold is
    `prod_a SU(N_a)`.  `fold_orbit` is the one-orbit case.

    This is what `_entry_for` could not do while it folded `d[0]` alone, which
    is why `SU(2) x SU(2)` was out of reach although the `S` / spec / F engines
    below were always `prod_a SU(N_a)`-general.
    """
    orbits = [sorted(int(x) for x in orb) for orb in family]
    drop = {k for orb in orbits for k in orb[1:]}
    keep = [k for k in range(len(B)) if k not in drop]
    reduced = [[int(B[a][b]) for b in keep] for a in keep]
    factors = [1] * len(keep)
    for orb in orbits:
        factors[keep.index(orb[0])] = len(orb)
    return FlavouredQuiver(reduced, factors)


def unfold_orbits_at(B: Sequence[Sequence[int]],
                     marks: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Replace each marked node by that many identical copies.

    `marks` is `[(node, size), …]` on the ORIGINAL node indices; the unfoldings
    are applied in decreasing node order so the earlier indices do not move.
    Inverse to `fold_orbits` up to node permutation.
    """
    out = [[int(x) for x in row] for row in B]
    for node, size in sorted(marks, reverse=True):
        out = unfold_orbit_at(out, node, size)
    return out


# --------------------------------------------------------------------------
# the value: covariant spec, else covariant crystalline S
# --------------------------------------------------------------------------

@dataclass
class FlavouredEntry:
    """One dictionary entry.  `spec` is `None` exactly when `omega` carries it.

    ⚠ **Two labellings, and only one of them is canonical.**  `exchange` is the
    canonical form of the unfolded quiver — it is the KEY, so it must be.  The
    other three are in the labelling the builder actually folded, which is the
    walk's and not the canonical one, and they are mutually consistent there.
    So `doublet` does NOT index `exchange`, and folding `exchange` at those
    indices is meaningless.

    What the entry guarantees, and what the shipped verifier checks, is the
    indexing-independent statement:

        canonical_form(unfold(reduced, factors)) == canonical_form(exchange)

    A consumer wanting the orbit in `exchange`'s own labelling computes it —
    `node_orbit_families(exchange, sizes)[0]`, unique because a tier admits a
    quiver only when its family is unique.  `doublet` is the builder's
    provenance record, not that.
    """

    exchange: Matrix                     # the UNFOLDED quiver, canonical form
    doublet: tuple[int, ...]             # the node ORBIT, in the BUILDER's order
    reduced: Matrix                      # the folded quiver, same order as doublet
    factors: tuple[int, ...]             # per reduced node: 1, or N at the orbit
    certification: str
    spec: tuple | None = None            # ((charge, irrep), …)
    omega: dict | None = None            # {charge: {irrep: {q-exp: mult}}}
    note: str = ""

    def as_json(self) -> dict:
        out = {
            "exchange": [list(r) for r in self.exchange],
            # `doublet` is the shipped SU(2) spelling and is kept so the tracked
            # tier's schema does not move; `orbit` is the same data under the
            # general name, written whenever the orbit is not a pair.
            "doublet": list(self.doublet),
            "reduced": [list(r) for r in self.reduced],
            "factors": list(self.factors),
            "certification": self.certification,
        }
        if len(self.doublet) != 2:
            out["orbit"] = list(self.doublet)
        if self.spec is not None:
            out["spec"] = [[list(g), [list(p) for p in r]] for g, r in self.spec]
        if self.omega is not None:
            out["omega"] = [
                [list(k), [[[list(p) for p in rep], sorted(poly.items())]
                           for rep, poly in per.items()]]
                for k, per in sorted(self.omega.items())
            ]
        if self.note:
            out["note"] = self.note
        return out


def acyclic_covariant_spec(fq: FlavouredQuiver) -> tuple | None:
    """The `SU(2)`-covariant spec of an ACYCLIC flavoured quiver — a construction.

    The author, 2026-09-14: *"for acyclic it is easy to write an SU(2) covariant spec,
    by placing the two identical nodes consecutive"*.  On an acyclic quiver the
    source/sink order is already a spec; the two doublet nodes are identical and
    non-adjacent (`B[i][j] = 0` — `node_doublets`), so they have the same
    relations to every other node and can always be moved next to each other
    without disturbing that order.  Consecutive is exactly what covariance
    needs: the two factors `E_𝖖(μ^{+}X_γ)` and `E_𝖖(μ^{-}X_γ)` are then
    adjacent, so they group into the single flavoured generator
    `E^{(s;r)}_𝖖(X_γ) = ∏_{w∈r} E^{(s)}_𝖖(μ^w X_γ)`.

    At the REDUCED level that is simply the source/sink order of the folded
    quiver, one generator per reduced node carrying that node's rep — the floor.
    Returned directly rather than searched for: a search would confirm on a cone
    of some finite degree and report a claim relative to it, where this is a
    theorem.  Asserted against the search in the suite in the source repository.
    """
    from quiver_enumeration import source_sink_order

    order = source_sink_order([list(r) for r in fq.pairing])
    if order is None:
        return None
    reps = fq.node_reps
    rank = fq.rank
    return tuple((tuple(1 if j == i else 0 for j in range(rank)), tuple(reps[i]))
                 for i in order)


# Raised by a caller's time limit or by exhausted memory: never a search outcome
# (covariant_value re-raises them).
_NOT_A_RESULT = (TimeoutError, MemoryError)


def covariant_value(fq: FlavouredQuiver, *, cutoff: int = 4, depth: int = 4,
                    trials: int = 8, use_blocks: bool = True,
                    order_first: bool = False, use_order: bool = True,
                    timings: dict | None = None):
    """`(certification, spec | None, omega | None, note)` for one flavoured quiver.

    Tried in the author's order of preference — a covariant spec first, the
    crystalline `Ω(γ, r)` only when no spec is found — with the acyclic case
    free, since the strip order attains the floor there by a theorem.

    **Between the two spec finders the block BFS goes first** (2026-09-15), on a
    measurement rather than a preference.  On 25 cyclic weight-9 rank-6 entries
    the old order took 11.8 s and resolved 23 by blocks and 2 by the order
    search; blocks-first took **0.5 s and resolved 25/25 by blocks** — a 23x
    speedup, because the order search was being run and failing on ~90 % of the
    entries before the block BFS succeeded on them.  The shipped weight-9 census
    shows the same shape from the other side: 19,680 `blocks` against 2,649
    `order@5`.

    It also yields a *better* spec.  A block-move negating sequence is a green
    sequence, and the acceptance discipline prefers a green spec to one accepted
    in an arbitrary factor order; the two entries the old order
    recorded as `order@5` in that sample are `blocks` under this one.

    `order_first=True` restores the previous order, which is what the shipped
    weight-9 tier was built with.

    `use_order=False` skips the order search, the way `use_blocks=False` skips
    the block BFS: an entry the block BFS does not resolve goes straight to the
    crystalline `Ω`.  It is the flavoured counterpart of `build_enumerated`'s
    `--no-order-search` (ruled there for the UNFLAVOURED tier), and
    a builder that uses it must record the skip, since a crystalline entry then
    means "no block spec" rather than "no spec found by either finder".

    `timings`, when a dict is passed, receives the wall seconds each stage took
    (`blocks`, `order_search`, `crystalline`) -- a measurement hook only; the
    returned value does not depend on it.

    **A caller's time limit, or exhausted memory, is never a result.**  A
    `TimeoutError` (or subclass) or `MemoryError` raised inside any step
    propagates to the caller.  It is not read as "no block spec", as "no order
    spec", or, in the crystalline step, as `crystalline_failed`, which is the
    counterexample flag.  A time limit must therefore raise one of those, or a
    `BaseException`.  (2026-09-29: the Symmetry session's harness found that a
    SIGALRM helper raising a plain `Exception` subclass was swallowed.  In a
    local probe a 300 s limit turned a pending SU(2) weight-12 quiver into
    `crystalline_only@3`.  No shipped build sets a time limit here.)
    """
    import time as _time

    # ACYCLIC: a construction, not a search.
    built = acyclic_covariant_spec(fq)
    if built is not None:
        return "covariant_spec:strip", built, None, ""

    def by_order():
        if not use_order:
            return None
        try:
            res = find_flavoured_spec(fq, cutoff, trials=trials)
        except _NOT_A_RESULT:
            raise                                  # the caller's limit, not a search outcome
        except Exception as exc:                   # honest: record, never hide
            return ("spec_search_failed", None, None,
                    f"{type(exc).__name__}: {exc}"[:200])
        if res.spec is not None:
            return (f"covariant_spec:order@{res.spec.confirmed_to}",
                    tuple(res.spec.entries), None, "")
        return None

    def by_blocks():
        if not use_blocks:
            return None
        try:
            seq = find_flavoured_negating_sequence(fq)
        except _NOT_A_RESULT:
            raise
        except Exception:
            return None
        if seq is None:
            return None
        try:
            sp = spec_from_negating_sequence(fq, seq)
        except _NOT_A_RESULT:
            raise
        except Exception:
            return None
        if sp is None:
            return None
        entries = tuple(sp.entries) if hasattr(sp, "entries") else tuple(sp)
        return "covariant_spec:blocks", entries, None, ""

    def timed(name, fn):
        if timings is None:
            return fn()
        t0 = _time.perf_counter()
        try:
            return fn()
        finally:
            timings[name] = timings.get(name, 0.0) + _time.perf_counter() - t0

    failure = None
    finders = ((("order_search", by_order), ("blocks", by_blocks)) if order_first
               else (("blocks", by_blocks), ("order_search", by_order)))
    for name, finder in finders:
        if name == "order_search" and not use_order:
            continue                      # skipped, so not timed as a "run" either
        got = timed(name, finder)
        if got is None:
            continue
        if got[0] == "spec_search_failed":
            failure = got                 # remember, but let the other finder try
            continue
        return got
    if failure is not None:
        return failure

    def crystalline():
        try:
            spectrum = FlavouredFactorSpectrum(fq, depth)
            om = spectrum.multiplicities()
        except _NOT_A_RESULT:
            raise                                  # never a crystalline_failed: that is a finding
        except Exception as exc:
            return (f"crystalline_failed@{depth}", None, None,
                    f"{type(exc).__name__}: {exc}"[:200])
        return f"crystalline_only@{depth}", None, om, ""
    return timed("crystalline", crystalline)


# --------------------------------------------------------------------------
# the walk
# --------------------------------------------------------------------------

def flavoured_quivers(*, doublets: int = 1, max_weight: int = 6,
                      max_rank: int | None = None, orbit_size: int = 2,
                      dict_dir: str | Path | None = None) -> Iterator[dict]:
    """Connected quivers of weight `≤ max_weight` with exactly `doublets` doublets.

    Enumerated directly at the walk's OWN bound (`quiver_enumeration.
    enumerate_connected_quivers`) rather than read out of the shipped
    dictionary.  Two reasons.  The shipped shards CODE the acyclic quivers
    instead of storing them, and regenerating the coded half re-enumerates the
    whole shipped set — weight 9 — however small a slice is asked for; and the
    acyclic quivers are the bulk of the flavoured set, so they cannot simply be
    skipped (the `[A_1, D_3]` anchor is one).  Enumerating here is both faster
    and complete by construction.

    That leaves membership in the unflavoured dictionary as an independent
    CHECK rather than the source of the walk — the stronger arrangement, and
    what the suite in the source repository asserts.

    `dict_dir` is accepted and unused; kept so a caller can pin a dictionary for
    that check without the signature changing.
    """
    from quiver_enumeration import enumerate_connected_quivers

    del dict_dir
    records = enumerate_connected_quivers(max_weight, max_rank=max_rank)
    for key in records:
        B = [[int(x) for x in row] for row in key]
        n = len(B)
        if n < 2:
            continue
        w = sum(abs(B[i][j]) for i in range(n) for j in range(i + 1, n))
        d = node_orbits(B, orbit_size)
        if len(d) != doublets:
            continue
        yield {"exchange": B, "n": n, "e": w, "doublets": d}


def unfold_at(B: Sequence[Sequence[int]], a: int) -> list[list[int]]:
    """Duplicate node `a` of `B`, giving the quiver `a` is a doublet node of.

    The twin pairs to zero with `a` and identically with every other node —
    which is exactly the `node_doublets` condition — so `(a, a')` is a doublet
    of the result and `fold_doublet` inverts this.  The weight grows by `a`'s
    degree `Σ_{k≠a} |B[a][k]|`, which is `≥ 1` on a connected quiver with more
    than one node.
    """
    n = len(B)
    out = [[int(B[i][j]) for j in range(n)] + [int(B[i][a])] for i in range(n)]
    out.append([int(B[a][j]) for j in range(n)] + [0])
    out[a][n] = 0
    out[n][a] = 0
    return out


def flavoured_quivers_via_fold(*, doublets: int = 1, max_weight: int = 10,
                               min_weight: int = 0,
                               max_rank: int | None = None,
                               only_rank: int | None = None,
                               orbit_size: int = 2,
                               folds=None, workers: int = 1,
                               progress: bool = False) -> Iterator[dict]:
    """The same set as `flavoured_quivers`, reached by enumerating the FOLDS.

    Why this route exists.  `flavoured_quivers` enumerates every connected
    quiver of weight `≤ max_weight` and keeps the ones with a doublet, so it
    pays for the whole unflavoured walk — 4,934,371 quivers at weight 10, whose
    dedup dictionary does not fit in a 15 GB container (measured: 2.70 GB peak
    at weight 9, and the set grows ~7× per unit of weight).

    Unfolding is what makes the smaller walk exact.  A doublet node's twin has
    the same neighbours, so deleting it keeps the quiver connected and
    `fold_doublet` inverts `unfold_at`; and unfolding at a node of degree `d`
    adds exactly `d ≥ 1` to the weight.  Hence **every** flavoured quiver of
    unfolded weight `≤ E` is `unfold_at(B_red, a)` for a connected `B_red` of
    weight `≤ E − 1` — 668,926 of those at `E = 10` rather than 4.9 million.

    Distinct `(B_red, a)` can unfold to the same quiver, so results are
    deduplicated by the unfolded canonical form; and the doublet count is
    checked on the unfolded quiver, since duplicating a node can create a
    second doublet (two nodes that were already identical become two pairs).

    `only_rank` restricts the yield to unfoldings with exactly that many nodes,
    and `folds` accepts an already-enumerated fold set.  Together they are what
    lets a big weight be built **one cell at a time in one process**: enumerate
    the folds once, then sweep the ranks, so only one cell's entries and one
    cell's dedup set are ever resident.  The sweep costs almost nothing extra,
    because a candidate's rank is `g + 1` — known before `canonical_form`, which
    is the only expensive step — so each rank's pass canonicalises only its own
    candidates and the total canonicalisation work is that of a single walk.
    This is not an optimisation but a requirement at weight 10: the single walk
    was measured growing ~0.1 GB/min past 6.5 GB with the cell dictionary and
    the dedup set both live.

    `min_weight` skips candidates below a weight — for EXTENDING a shipped tier,
    where the lower cells are already on disk.  It is applied before the
    canonical form, which is the expensive step, so extending by one weight does
    not re-canonicalise the whole set underneath it.  It narrows the yield only:
    the walk over folds is unchanged, since a fold of any weight can produce a
    high-weight unfolding.

    Positive control: at `max_weight ≤ 9` and `min_weight = 0` this must
    reproduce `flavoured_quivers` exactly, cell by cell — asserted in
    the suite in the source repository.
    """
    from quiver_enumeration import enumerate_connected_quivers

    if max_weight < 1:
        return
    span = orbit_size - 1            # unfolding at degree d adds (N-1)*d
    if folds is None:
        if progress:
            print(f"enumerating FOLDS to weight {max_weight - span} …", flush=True)
        folds = enumerate_connected_quivers(max_weight - span, workers=workers)
        if progress:
            print(f"  {len(folds)} folds", flush=True)

    seen: set = set()
    for key in folds:
        g = len(key)
        if max_rank is not None and g + span > max_rank:
            continue
        if only_rank is not None and g + span != only_rank:
            continue
        w_red = sum(abs(key[i][j]) for i in range(g) for j in range(i + 1, g))
        for a in range(g):
            deg = sum(abs(key[a][k]) for k in range(g) if k != a)
            w = w_red + span * deg
            if deg < 1 or w > max_weight or w < min_weight:
                continue
            B = unfold_orbit_at(key, a, orbit_size)
            # The ORBIT COUNT is tested before `canonical_form`, and the two
            # cannot be swapped back for free: the canonical form is the one
            # expensive step here, its leaf count runs with |Aut| (a star on n
            # nodes has (n-1)! leaves), and the quivers with the largest |Aut|
            # are exactly the ones this test throws away -- a 12-node star has
            # hundreds of orbits of any given size, so no tier admits it.
            # Canonicalising it first is what raised `canonical_form: leaf cap
            # exceeded` in rank 12 of the SU(5) and SU(6) weight-11 builds
            # (2026-09-22), and what made the SU(2) rank-11 walk take 1.8 h.
            # The yielded sequence is unchanged: the orbit count is a property
            # of the quiver, not of its labelling, so a rejected quiver is
            # rejected at every fold that reaches it, and an accepted one is
            # still first reached by the same fold.  Held to that by
            # the suite in the source repository.
            d = node_orbits(B, orbit_size)
            if len(d) != doublets:
                continue
            cf = canonical_form(B)
            if cf.key in seen:
                continue
            seen.add(cf.key)
            n = g + span
            yield {"exchange": [[int(x) for x in row] for row in B], "n": n,
                   "e": w, "doublets": d}


def flavoured_quivers_multi(*, sizes: Sequence[int], max_weight: int,
                            min_weight: int = 0, only_rank: int | None = None,
                            families: int = 1, folds=None, workers: int = 1,
                            progress: bool = False) -> Iterator[dict]:
    """The fold walk for SEVERAL orbits — `prod_a SU(N_a)`.

    `sizes = (2, 2)` is `SU(2) x SU(2)`: two DISJOINT node pairs, which is not
    one orbit of size 4 (four mutually identical nodes) and not three
    overlapping pairs inside a triple.  `sizes = (N,)` is the single-orbit walk
    and is asserted to agree with it.

    Same fold argument as the one-orbit case, applied once per orbit: unfolding
    a node of degree `d` into `N` copies adds `(N-1)*d >= N-1` to the weight, so
    a quiver of unfolded weight `<= E` carrying these orbits folds to weight
    `<= E - sum(N_a - 1)`.  The walk therefore enumerates folds at that lower
    bound and marks `len(sizes)` distinct nodes of each.

    The acceptance is `node_orbit_families(B, sizes)` having exactly `families`
    elements.  That excludes the degenerate neighbours by itself: four mutually
    identical nodes admit three different pairings into two pairs, so they are
    rejected here rather than silently filed as `SU(2) x SU(2)`.  It does NOT
    exclude a quiver that also carries an unrelated orbit of some other size —
    that is a judgement about what the entry IS, and it is left to the caller
    rather than decided here.
    """
    import itertools
    from quiver_enumeration import enumerate_connected_quivers

    sizes = tuple(int(x) for x in sizes)
    span = sum(k - 1 for k in sizes)
    if max_weight < span:
        return
    if folds is None:
        if progress:
            print(f"enumerating FOLDS to weight {max_weight - span} …", flush=True)
        folds = enumerate_connected_quivers(max_weight - span, workers=workers)
        if progress:
            print(f"  {len(folds)} folds", flush=True)

    seen: set = set()
    for key in folds:
        g = len(key)
        if g < len(sizes):
            continue
        if only_rank is not None and g + span != only_rank:
            continue
        w_red = sum(abs(key[i][j]) for i in range(g) for j in range(i + 1, g))
        for nodes in itertools.permutations(range(g), len(sizes)):
            if sizes.count(sizes[0]) == len(sizes) and list(nodes) != sorted(nodes):
                continue        # equal sizes: the marking is unordered
            if any(sum(abs(key[a][k]) for k in range(g) if k != a) < 1
                   for a in nodes):
                continue
            B = unfold_orbits_at(key, list(zip(nodes, sizes)))
            n = g + span
            w = sum(abs(B[i][j]) for i in range(n) for j in range(i + 1, n))
            if w > max_weight or w < min_weight:
                continue
            # The family test goes before `canonical_form`, for the reason
            # given in `flavoured_quivers_via_fold`: the most symmetric quivers
            # are the expensive ones to canonicalise and the ones this rejects.
            fams = node_orbit_families(B, sizes)
            if len(fams) != families:
                continue
            cf = canonical_form(B)
            if cf.key in seen:
                continue
            seen.add(cf.key)
            # NOTE the nesting: `doublets` is a LIST of admissible foldings, as
            # in the single-orbit walk, and each element is a whole FAMILY of
            # disjoint orbits.  A caller takes `doublets[0]` and folds all of
            # it.  Yielding the family bare instead made `doublets[0]` one
            # ORBIT, so the builder folded half of an `SU(2) x SU(2)` and
            # emitted entries carrying a single `SU(2)` -- caught by the
            # per-tier factor check in the suite in the source repository.
            yield {"exchange": [[int(x) for x in row] for row in B], "n": n,
                   "e": w, "doublets": [fams[0]]}


def _cell_compute(B, pair, cutoff, depth, trials, *, use_order: bool = True,
                  timings: dict | None = None):
    """One quiver -> its entry, as a worker-friendly plain tuple.

    The result depends only on the arguments -- including the LABELLING of `B`,
    since `reduced`, `doublet` and the spec are written in it -- so a resumed or
    split build reproduces a plain one exactly when it feeds the walk's own
    `(B, pair)`.  `use_order` and `timings` are passed to `covariant_value`.
    """
    # `pair` is either one orbit (a tuple of node indices) or a FAMILY of them
    fq = (fold_orbits(B, pair) if pair and isinstance(pair[0], (tuple, list))
          else fold_orbit(B, pair))
    cert, spec, om, note = covariant_value(fq, cutoff=cutoff, depth=depth,
                                           trials=trials, use_order=use_order,
                                           timings=timings)
    return (cert, spec, om, note,
            tuple(tuple(r) for r in fq.pairing), tuple(fq.flavour.factors))


def _cell_task(task):
    """One quiver -> its entry, as a worker-friendly plain tuple."""
    B, pair, cutoff, depth, trials = task
    return _cell_compute(B, pair, cutoff, depth, trials)


def _entry_for(B, pair, cutoff, depth, trials) -> FlavouredEntry:
    cert, spec, om, note, reduced, factors = _cell_task(
        (B, pair, cutoff, depth, trials))
    return FlavouredEntry(exchange=canonical_form(B).key, doublet=pair,
                          reduced=reduced, factors=factors,
                          certification=cert, spec=spec, omega=om, note=note)


def build(*, doublets: int = 1, max_weight: int = 6, max_rank: int | None = None,
          cutoff: int = 4, depth: int = 4, trials: int = 8,
          orbit_size: int = 2, dict_dir: str | Path | None = None,
          progress: bool = True) -> list[FlavouredEntry]:
    """Build the flavoured dictionary up to `max_weight`, in memory.

    Convenient for small weights and for the tests; `build_cells` is the one to
    use at scale, since it checkpoints and parallelises.
    """
    out: list[FlavouredEntry] = []
    for i, rec in enumerate(flavoured_quivers(doublets=doublets,
                                              max_weight=max_weight,
                                              max_rank=max_rank,
                                              dict_dir=dict_dir)):
        out.append(_entry_for(rec["exchange"], rec["doublets"][0],
                              cutoff, depth, trials))
        if progress and (i + 1) % 25 == 0:
            print(f"    … {i + 1} entries", flush=True)
    return out


def flavour_group_name(orbit_size) -> str:
    """How a shard names the flavour group of its tier.

    One orbit of size `N` is `SU(N)`.  A FAMILY of disjoint orbits is the
    product of their factors, so `(3, 2)` reads `SU(3)xSU(2)` -- and never
    `SU((3, 2))`, which names nothing.
    """
    sizes = orbit_size if isinstance(orbit_size, (tuple, list)) else [orbit_size]
    return "x".join(f"SU({int(k)})" for k in sizes)


def flavour_orbit_phrase(orbit_size, doublets: int = 1) -> str:
    """How a shard names the SELECTOR -- which orbits a member must carry."""
    if isinstance(orbit_size, (tuple, list)):
        sizes = [int(k) for k in orbit_size]
        listed = (" and ".join(str(k) for k in sizes) if len(sizes) == 2
                  else ", ".join(str(k) for k in sizes[:-1]) + f" and {sizes[-1]}")
        return (f"{doublets} famil{'y' if doublets == 1 else 'ies'} of "
                f"pairwise-disjoint node orbits of sizes {listed}")
    return f"{doublets} node orbit(s) of size {int(orbit_size)}"


def shard_header(n: int, e: int, *, count: int, acyclic: int, stored: int,
                 acyclic_coded: bool, doublets: int = 1,
                 orbit_size=2, skipped_stages: Sequence[str] = ()) -> dict:
    """The self-describing header of one cell shard.

    A single cell has to be usable on its own — the same discipline as the
    unflavoured tier — so the completeness statement and the coding convention
    travel with the file rather than only with `manifest.json`.

    `skipped_stages` names a search the build did NOT run for this cell (today
    only `"order_search"`, from `--no-order-search`); it is written only when
    non-empty, so every header built with both searches is unchanged.  It rides
    in the cell because a crystalline entry means something weaker without the
    order search, and a single cell must say so on its own.
    """
    group = flavour_group_name(orbit_size)
    head = {
        "plan": "41_enumerated_quiver_dictionary",
        "format": SHARD_FORMAT, "orbit_size": orbit_size, "n": n, "e": e,
        "completeness": (f"every connected quiver with {n} nodes and {e} arrows "
                         f"(counted with multiplicity) having exactly "
                         f"{flavour_orbit_phrase(orbit_size, doublets)}, up to "
                         f"node permutation, each with an {group}-covariant "
                         "spec or the covariant crystalline Omega"),
        "count": count, "acyclic": acyclic, "stored": stored,
        "acyclic_coded": bool(acyclic_coded),
    }
    if acyclic_coded:
        which = ("each orbit's identical nodes"
                 if isinstance(orbit_size, (tuple, list)) else "the identical nodes")
        head["acyclic_note"] = (
            "the acyclic quivers of this cell are CODED, not stored: their "
            f"{group}-covariant spec is the source/sink order with "
            f"{which} consecutive (a theorem), so a lookup "
            "recomputes it rather than reading it; count = acyclic + stored")
    if skipped_stages:
        head["skipped_stages"] = sorted(set(skipped_stages))
    return head


def build_cells(*, doublets: int = 1, max_weight: int = 10,
                max_rank: int | None = None, cutoff: int = 4, depth: int = 3,
                trials: int = 6, out_dir: str | Path = "dictionaries/flavoured",
                workers: int = 1, acyclic_coded: bool = True,
                via_fold: bool = False, min_weight: int = 0,
                orbit_size: int = 2, progress: bool = True) -> Path:
    """Build cell by cell, writing each shard as it completes — the scale route.

    Three things this does that `build` does not, all of which weight 10 needs.

    **Acyclic quivers are CODED, not stored** (`acyclic_coded`, the default),
    mirroring the shipped unflavoured tier's own `SHARD_FORMAT = 2` convention.
    It is licensed here by the same kind of theorem: an acyclic flavoured quiver's
    covariant spec is its source/sink order with the two identical nodes placed
    consecutively, so a lookup RECOMPUTES it with
    `acyclic_covariant_spec` instead of reading it.  They are ~87 % of the set at
    weight 8 and the share falls only slowly, so this is most of the file size.
    Each cell header keeps `count = acyclic + stored`, so the completeness
    statement is unchanged and checkable.

    **Parallel**, because only the cyclic minority costs anything: the acyclic
    bulk is a construction.  Workers are forked per cell.

    **Checkpointed**: a cell whose shard already exists is skipped, so an
    interrupted build resumes at the cell it died in rather than at the start.
    That matters — the enumeration alone is ~2 h at weight 10, and this session's
    container has already restarted once mid-job.  With `via_fold` the skip
    happens *during* enumeration, so an already-shipped cell costs no memory
    either — which is what makes extending a shipped tier by one weight cheap.

    **`via_fold`** switches the walk to `flavoured_quivers_via_fold`, which
    enumerates the FOLDS at weight `max_weight − 1` instead of the unfolded
    quivers at `max_weight`.  Required from weight 10 up: the unfolded walk
    holds 4,934,371 quivers there, ≈ 20 GB by the measured 2.70 GB at weight 9,
    against 15 GB of container RAM.  The two routes are asserted to give the
    same set (the suite in the source repository, and measured equal at weights
    4–8: 19 / 95 / 514 / 3,080 / 20,097).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if progress:
        print(f"enumerating to weight {max_weight} …", flush=True)
    cells: dict[tuple[int, int], list] = {}
    walk = (flavoured_quivers_via_fold(doublets=doublets, max_weight=max_weight,
                                       min_weight=min_weight, max_rank=max_rank,
                                       orbit_size=orbit_size, workers=workers,
                                       progress=progress)
            if via_fold else
            flavoured_quivers(doublets=doublets, max_weight=max_weight,
                              max_rank=max_rank, orbit_size=orbit_size))
    skipped: set[tuple[int, int]] = set()
    for rec in walk:
        cell = (rec["n"], rec["e"])
        if cell in skipped:
            continue
        if cell not in cells and (
                out / f"f_n{cell[0]:02d}_e{cell[1]:02d}.json.gz").exists():
            # already shipped: drop it here rather than carrying it to the loop
            skipped.add(cell)
            continue
        cells.setdefault(cell, []).append(
            (rec["exchange"], rec["doublets"][0]))
    if progress and skipped:
        print(f"  {len(skipped)} cells already shipped, skipped during the walk",
              flush=True)
    if progress:
        tot = sum(len(v) for v in cells.values())
        print(f"  {tot} flavoured quivers in {len(cells)} cells", flush=True)

    pool = None
    if workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(workers)
    try:
        for (n, w), items in sorted(cells.items()):
            path = out / f"f_n{n:02d}_e{w:02d}.json.gz"
            if path.exists():
                if progress:
                    print(f"  cell ({n},{w}): {len(items)} — shard present, skipped",
                          flush=True)
                continue
            acyclic, cyclic = [], []
            for B, pair in items:
                (acyclic if is_acyclic(B) else cyclic).append((B, pair))

            entries: list[FlavouredEntry] = []
            store = cyclic if acyclic_coded else (acyclic + cyclic)
            if not acyclic_coded:
                store = items
            tasks = [(B, pair, cutoff, depth, trials) for B, pair in store]
            if pool is not None and len(tasks) >= 4 * workers:
                results = pool.map(_cell_task, tasks, chunksize=8)
            else:
                results = [_cell_task(t) for t in tasks]
            for (B, pair), (cert, spec, om, note, reduced, factors) in zip(store, results):
                entries.append(FlavouredEntry(
                    exchange=canonical_form(B).key, doublet=pair,
                    reduced=reduced, factors=factors,
                    certification=cert, spec=spec, omega=om, note=note))

            body = shard_header(n, w, count=len(items), acyclic=len(acyclic),
                                stored=len(entries), acyclic_coded=acyclic_coded,
                                doublets=doublets, orbit_size=orbit_size)
            body["entries"] = [x.as_json() for x in entries]
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump(body, fh, separators=(",", ":"))
            if progress:
                print(f"  cell ({n},{w}): {len(items)} quivers "
                      f"({len(acyclic)} acyclic coded, {len(entries)} stored) "
                      f"-> {path.stat().st_size} B", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    write_manifest(out, max_weight=max_weight, doublets=doublets,
                   acyclic_coded=acyclic_coded, orbit_size=orbit_size)
    return out


def reheader_cells(out_dir: str | Path, *, doublets: int = 1,
                   orbit_size=2) -> int:
    """Rewrite every shard's HEADER in place, leaving its entries verbatim.

    A shard is self-describing, so its prose is part of what it ships -- and
    prose gets corrected (the product tiers first wrote `SU((3, 2))`, which
    names nothing).  The header is a pure function of the shard's own
    `(n, e, count, acyclic, stored, acyclic_coded)` plus the tier's selector,
    all of which the shard already carries, so regenerating it cannot change
    what the tier CLAIMS about any quiver -- and it costs seconds where a
    rebuild costs half an hour.  Returns the number of shards rewritten.

    The entries list is copied across untouched and asserted unchanged.
    """
    out = Path(out_dir)
    n_written = 0
    for path in sorted(out.glob("f_n*_e*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            old = json.load(fh)
        body = shard_header(old["n"], old["e"], count=old["count"],
                            acyclic=old["acyclic"], stored=old["stored"],
                            acyclic_coded=old["acyclic_coded"],
                            doublets=doublets, orbit_size=orbit_size,
                            skipped_stages=old.get("skipped_stages", ()))
        body["entries"] = old["entries"]
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(body, fh, separators=(",", ":"))
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            back = json.load(fh)
        if back["entries"] != old["entries"]:       # never seen; the point is
            raise AssertionError(f"{path}: entries changed")   # that it cannot
        n_written += 1
    return n_written


def write_manifest(out_dir: str | Path, *, max_weight: int, doublets: int,
                   acyclic_coded: bool, orbit_size=2,
                   complete_through: int | None = None) -> Path:
    """Census over whatever shards are present.

    `complete_through` is the largest weight the build actually FINISHED.  It
    matters because the completeness statement is the whole value of a
    dictionary: a run that is interrupted part-way through its top weight has
    some of that weight's cells on disk and not others, and a manifest claiming
    `e <= max_weight` there would be false.  When it is given and smaller than
    the top weight present, the manifest says the top weight is PARTIAL and
    lists the cells it does have, so a reader can tell exactly what is claimed.
    Passing `None` keeps the old behaviour and asserts full coverage.
    """
    from collections import Counter

    out = Path(out_dir)
    cells: dict[str, dict] = {}
    census: Counter = Counter()
    skipped: dict[str, list[str]] = {}
    total = stored = acyclic = 0
    for path in sorted(out.glob("f_n*_e*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            d = json.load(fh)
        cells[f"{d['n']},{d['e']}"] = {"count": d.get("count", len(d["entries"])),
                                       "acyclic": d.get("acyclic", 0),
                                       "stored": d.get("stored", len(d["entries"]))}
        for stage in d.get("skipped_stages", ()):
            skipped.setdefault(stage, []).append(f"{d['n']},{d['e']}")
        total += cells[f"{d['n']},{d['e']}"]["count"]
        acyclic += cells[f"{d['n']},{d['e']}"]["acyclic"]
        stored += cells[f"{d['n']},{d['e']}"]["stored"]
        for x in d["entries"]:
            census[x["certification"]] += 1
    if acyclic_coded:
        census[f"covariant_spec:strip (coded, not stored)"] = acyclic
    top = max((int(k.split(",")[1]) for k in cells), default=0)
    full = max_weight if complete_through is None else min(complete_through, top)
    if full >= top:
        statement = ("every connected quiver with total arrow weight e <= "
                     f"{full} having exactly "
                     f"{flavour_orbit_phrase(orbit_size, doublets)}, up to "
                     "node permutation")
        partial = None
    else:
        have = sorted(int(k.split(",")[0]) for k in cells
                      if int(k.split(",")[1]) > full)
        statement = ("every connected quiver with total arrow weight e <= "
                     f"{full} having exactly "
                     f"{flavour_orbit_phrase(orbit_size, doublets)}, up to "
                     f"node permutation; weight {full + 1}..{top} is PARTIAL — "
                     "only the cells listed under partial_cells are present, "
                     "and each of those is itself complete")
        partial = {"complete_through_weight": full,
                   "top_weight_present": top,
                   "partial_cells": sorted(
                       k for k in cells if int(k.split(",")[1]) > full),
                   "partial_ranks": have}
    manifest = {
        "format": SHARD_FORMAT, "doublets": doublets, "orbit_size": orbit_size,
        "max_weight": max_weight, "acyclic_coded": acyclic_coded,
        "completeness": statement,
        "acyclic_note": ("the acyclic quivers are CODED, not stored: their "
                         f"{flavour_group_name(orbit_size)}-covariant spec is "
                         "the source/sink order with "
                         + ("each orbit's identical nodes"
                            if isinstance(orbit_size, (tuple, list))
                            else "the identical nodes")
                         + " consecutive (a theorem), so a lookup recomputes "
                           "it with acyclic_covariant_spec"),
        "quivers": total, "acyclic": acyclic, "stored": stored,
        "cells": cells, "certification_census": dict(sorted(census.items())),
    }
    if partial is not None:
        manifest["coverage"] = partial
    if skipped:
        # A search the build did not run, and the cells it was not run for --
        # the tier never claims a search it did not make (the promise,
        # carried to the flavoured tiers).  Absent when every cell ran both.
        manifest["skipped_stages"] = {k: sorted(v) for k, v in sorted(skipped.items())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out / "manifest.json"


def ship(entries: Sequence[FlavouredEntry], out_dir: str | Path) -> Path:
    """Write per-cell shards from an in-memory entry list, storing ALL of them.

    The small-scale counterpart of `build_cells` (which codes the acyclic half
    and checkpoints); kept because it is the simplest thing that round-trips,
    and the tests use it.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells: dict[tuple[int, int], list] = {}
    for ent in entries:
        n = len(ent.exchange)
        w = sum(abs(ent.exchange[i][j])
                for i in range(n) for j in range(i + 1, n))
        cells.setdefault((n, w), []).append(ent)
    for (n, w), items in sorted(cells.items()):
        body = shard_header(n, w, count=len(items), acyclic=0,
                            stored=len(items), acyclic_coded=False)
        body["entries"] = [x.as_json() for x in items]
        with gzip.open(out / f"f_n{n:02d}_e{w:02d}.json.gz", "wt",
                       encoding="utf-8") as fh:
            json.dump(body, fh, separators=(",", ":"))
    write_manifest(out, max_weight=max((w for _n, w in cells), default=0),
                   doublets=1, acyclic_coded=False)
    return out


def flavoured_entry(B: Sequence[Sequence[int]], *, cutoff: int = 4,
                    depth: int = 3, trials: int = 6,
                    doublets: int = 1,
                    orbit_size: int = 2) -> FlavouredEntry | None:
    """The entry for ONE quiver, computed directly — the lookup primitive.

    Needs no enumeration and no dictionary, which is what makes coding the
    acyclic half free: an acyclic quiver's entry is recomputed here in
    microseconds rather than read from a file.  `None` if `B` does not have
    exactly `doublets` node doublets.
    """
    B = [[int(x) for x in row] for row in B]
    d = node_orbits(B, orbit_size)
    if len(d) != doublets:
        return None
    return _entry_for(B, d[0], cutoff, depth, trials)


def load_flavoured(dict_dir: str | Path, *, include_acyclic: bool = False,
                   cutoff: int = 4, depth: int = 3, trials: int = 6,
                   doublets: int = 1, orbit_size: int = 2) -> list[dict]:
    """Entries of a shipped flavoured dictionary.

    The shards STORE only the cyclic quivers (`SHARD_FORMAT = 2`); the acyclic
    ones are coded, their covariant spec being the construction.  So this yields
    the stored entries by default, and with `include_acyclic=True` re-enumerates
    the set and regenerates the coded half — complete, at the cost of the
    enumeration.  Each shard header's `count` / `acyclic` / `stored` is what
    makes the completeness statement checkable either way.
    """
    d = Path(dict_dir)
    out: list[dict] = []
    max_w = 0
    for path in sorted(d.glob("f_n*_e*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            body = json.load(fh)
        out.extend(body["entries"])
        max_w = max(max_w, int(body.get("e", 0)))
    if not include_acyclic:
        return out
    seen = {tuple(tuple(r) for r in e["exchange"]) for e in out}
    for rec in flavoured_quivers(doublets=doublets, max_weight=max_w,
                                 orbit_size=orbit_size):
        key = canonical_form(rec["exchange"]).key
        if key in seen:
            continue
        ent = _entry_for(rec["exchange"], rec["doublets"][0], cutoff, depth, trials)
        out.append(ent.as_json())
    return out


# --------------------------------------------------------------------------
# the strongly connected flavoured family
# --------------------------------------------------------------------------
#
# The author, 2026-09-28, accepting the proposal in the design notes:
# store only strongly connected flavoured quivers and answer every other one by
# composition; the key is the canonical form of the unfolded quiver coloured by
# orbit size; a composed record reads as the unflavoured one does; the
# covariant specs come from the enumerated tier's where they group, and otherwise
# from the block search.  The functions below read the strongly connected
# flavoured quivers off the enumerated tier and bring the shipped tiers' specs into
# the same labelling.

def equal_row_classes(B: Sequence[Sequence[int]]) -> list[list[int]]:
    """The classes of identical nodes with two or more members.

    Two nodes are identical — pairwise non-paired and exchanged by an
    automorphism — exactly when their rows of `B` are equal: equal rows force
    `B[i][j] = B[j][j] = 0` and make the swap an automorphism, and a swap that
    is an automorphism makes the rows equal.  So `node_orbits(B, N)` is the set
    of `N`-subsets of these classes (5,706 shipped entries agree,
    a probe in the source repository)."""
    rows: dict[tuple, list[int]] = {}
    for i, r in enumerate(B):
        rows.setdefault(tuple(int(x) for x in r), []).append(i)
    return [c for c in rows.values() if len(c) >= 2]


def orbit_family_choices(B: Sequence[Sequence[int]],
                         sizes: Sequence[int]) -> list[list[tuple[int, ...]]]:
    """Every family of disjoint node orbits of the given sizes, ONE per choice
    up to the permutations inside each class of identical nodes (those are
    automorphisms, so the families they relate are isomorphic).  A class may
    hold several orbits; orbits of equal size are placed in non-decreasing
    class order, so no choice is listed twice."""
    classes = equal_row_classes(B)
    sizes = sorted((int(s) for s in sizes), reverse=True)
    out: list[list[tuple[int, ...]]] = []

    def rec(k, cap, placed, last):
        if k == len(sizes):
            fam: list[tuple[int, ...]] = []
            for c, ss in sorted(placed.items()):
                nodes = list(classes[c])
                for s in ss:
                    fam.append(tuple(nodes[:s]))
                    nodes = nodes[s:]
            out.append(fam)
            return
        s = sizes[k]
        for c in range(last.get(s, 0), len(classes)):
            if cap[c] < s:
                continue
            cap[c] -= s
            placed.setdefault(c, []).append(s)
            prev = last.get(s)
            last[s] = c
            rec(k + 1, cap, placed, last)
            if prev is None:
                del last[s]
            else:
                last[s] = prev
            placed[c].pop()
            if not placed[c]:
                del placed[c]
            cap[c] += s

    rec(0, [len(c) for c in classes], {}, {})
    return out


def strongly_connected_flavoured_key(B: Sequence[Sequence[int]],
                                     family: Sequence[Sequence[int]]):
    """The key: the canonical form of the unfolded quiver with every node
    coloured by the size of its orbit (0 outside the family), returned as
    `((canonical matrix, colours in canonical order), CanonicalForm)`.

    Complete for these objects: orbits are subsets of classes of equal rows, so
    the colours fix which nodes carry which orbit up to permutations inside a
    class, which are automorphisms.  (`flavoured_s_axioms.md` §9's open case —
    two nodes under one `SU(2)` against two under `SU(2) × SU(2)` — cannot
    arise: each orbit carries its own factor.)"""
    colours = [0] * len(B)
    for orb in family:
        for v in orb:
            colours[int(v)] = len(orb)
    cf = canonical_form([[int(x) for x in r] for r in B], colours=colours)
    return (cf.key, tuple(colours[p] for p in cf.perm)), cf


def canonical_family(cf, family: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    """`family` (in the caller's node indices) in the canonical labelling of
    `cf` (canonical node `i` is the caller's `cf.perm[i]`), each orbit sorted,
    orbits in order of their first node."""
    where = {int(v): i for i, v in enumerate(cf.perm)}
    return sorted(tuple(sorted(where[int(v)] for v in orb)) for orb in family)


def _fold_index(B_len: int, family: Sequence[Sequence[int]]) -> dict[int, int]:
    """Unfolded node -> its node of `fold_orbits(B, family)` (an orbit's nodes all
    map to the fold node of its first member)."""
    orbits = [sorted(int(x) for x in orb) for orb in family]
    drop = {v for orb in orbits for v in orb[1:]}
    keep = [v for v in range(B_len) if v not in drop]
    index = {v: keep.index(v) for v in keep}
    for orb in orbits:
        for v in orb[1:]:
            index[v] = index[orb[0]]
    return index


def regroup_unfolded_spec(B: Sequence[Sequence[int]], family: Sequence[Sequence[int]],
                          spec: Sequence[Sequence[int]]):
    """An unfolded spec of `B` (its charges over `B`'s nodes) regrouped into a
    flavoured spec on `fold_orbits(B, family)`.

    Tried literally (`group_unfolded_spec`), then after sliding commuting factors
    (`group_unfolded_spec_up_to_commuting` — sufficient, not complete).  Returns
    `(flavoured spec, how)`, `how` being ``"as is"`` or ``"after commuting"``, or
    `None`.  The product is the same element either way, so the unfolded spec's
    record carries over."""
    from flavoured_spec import group_unfolded_spec, group_unfolded_spec_up_to_commuting
    fq = fold_orbits(B, family)
    orbits = [sorted(int(x) for x in orb) for orb in family]
    fold_of = _fold_index(len(B), family)
    _, _, labels = fq.unfold()
    by_node: dict[int, list] = {}
    for a, w in labels:
        by_node.setdefault(a, []).append(w)
    weight_of: dict[int, tuple] = {}
    for v in range(len(B)):
        a = fold_of[v]
        orb = next((o for o in orbits if v in o), None)
        weight_of[v] = by_node[a][orb.index(v)] if orb is not None else by_node[a][0]
    dim = len(labels[0][1]) if labels else 0
    seq = []
    for g in spec:
        gam = [0] * fq.rank
        w = [0] * dim
        for v, x in enumerate(g):
            if x:
                gam[fold_of[v]] += int(x)
                for t in range(dim):
                    w[t] += int(x) * weight_of[v][t]
        seq.append((tuple(gam), tuple(w)))
    got = group_unfolded_spec(fq, seq)
    if got is not None:
        return got, "as is"
    got = group_unfolded_spec_up_to_commuting(fq, seq)
    if got is not None:
        return got, "after commuting"
    return None


def fold_map(src_fold_of: dict[int, int], dst_fold_of: dict[int, int],
             node_map: dict[int, int]) -> dict[int, int]:
    """Fold node -> fold node, from an unfolded node map `node_map` (source node
    -> destination node) and each side's `_fold_index`."""
    out: dict[int, int] = {}
    for v, a in src_fold_of.items():
        b = dst_fold_of[node_map[v]]
        if out.setdefault(a, b) != b:
            raise ValueError("fold_map: the node map does not carry orbits to orbits")
    return out


def move_flavoured_spec(spec, fmap: dict[int, int], rank: int, trivial) -> list:
    """A flavoured spec `((charge, irrep), …)` carried along the fold map `fmap`
    (source fold node -> destination fold node) into a fold of rank `rank`, the
    other factors trivial (`trivial` = the destination's `trivial_irrep()`)."""
    out = []
    for g, rep in spec:
        charge = [0] * rank
        for a, x in enumerate(g):
            if x:
                charge[fmap[a]] += int(x)
        r = list(trivial)
        for a, part in enumerate(rep):
            if a in fmap:
                r[fmap[a]] = tuple(part)
        out.append((tuple(charge), tuple(r)))
    return out


def shipped_entry_in_key_labelling(ent: dict):
    """A shipped (format-2) flavoured entry brought into the labelling: its
    unfolded quiver with the canonical colouring by orbit size.

    The shipped `reduced` / `factors` / `spec` / `omega` are in the builder's
    labelling, NOT `exchange`'s (FlavouredEntry's warning), so the map is taken
    through the entry's own fold: unfold it, key it, and carry the fold nodes
    across.  Returns `(key, canonical matrix, canonical family, spec | None,
    omega | None)` with the spec on the canonical fold; the fold matrices are
    asserted to correspond."""
    fq_b = FlavouredQuiver(ent["reduced"], ent["factors"])
    B_b, _, labels = fq_b.unfold()
    fam_b = []
    for a, n in enumerate(ent["factors"]):
        if int(n) > 1:
            fam_b.append(tuple(i for i, (aa, _w) in enumerate(labels) if aa == a))
    key, cf = strongly_connected_flavoured_key(B_b, fam_b)
    B_c = [list(r) for r in cf.key]
    fam_c = canonical_family(cf, fam_b)
    src_fold_of = {i: labels[i][0] for i in range(len(labels))}
    dst_fold_of = _fold_index(len(B_c), fam_c)
    where = {int(v): i for i, v in enumerate(cf.perm)}
    fmap = fold_map(src_fold_of, dst_fold_of, where)
    fq_c = fold_orbits(B_c, fam_c)
    for a in range(fq_b.rank):
        for b in range(fq_b.rank):
            assert fq_c.pairing[fmap[a]][fmap[b]] == fq_b.pairing[a][b], "fold map is not an isomorphism"
    spec = None
    if ent.get("spec") is not None:
        raw = [(tuple(g), tuple(tuple(p) for p in r)) for g, r in ent["spec"]]
        spec = move_flavoured_spec(raw, fmap, fq_c.rank, fq_c.flavour.trivial_irrep())
    omega = None
    if ent.get("omega") is not None:
        omega = crystalline_omega_json(fq_c, _crystalline_depth(ent["certification"]))
    return key, B_c, fam_c, spec, omega


def _crystalline_depth(cert: str, default: int = 3) -> int:
    try:
        return int(str(cert).rsplit("@", 1)[1])
    except (IndexError, ValueError):
        return default


def crystalline_omega_json(fq: FlavouredQuiver, depth: int) -> list:
    """The covariant crystalline content `Ω(γ, r)` of a fold to `depth`, in the
    shipped JSON shape, RECOMPUTED in the fold's own labelling.

    It cannot be carried across a relabelling.  The content is that of the
    engine's order, and the default order depends on the node labelling.
    Measured 2026-09-28 on the five strongly connected crystalline `SU(4)`
    entries: a permuted `Ω` never equals the fresh one, while `S` agrees.  A
    carried `Ω` would be the content of the transported order, and a check
    against a fresh run in the stored labelling
    (`checks_rg.check_flavoured_quivers`) rejects it.  Raises if the fresh `Ω`
    is not a `G`-character: that is the flavoured counterexample, and it must
    not be written as `crystalline_only`."""
    eng = FlavouredFactorSpectrum(fq, int(depth))
    eng.run()
    bad = eng.verify_omega_is_a_character()
    if bad:
        raise ValueError(f"crystalline_omega_json: Omega is not a G-character on {fq!r}: {bad}")
    return [[list(k), [[[list(p) for p in rep], sorted(poly.items())] for rep, poly in per.items()]]
            for k, per in sorted(eng.multiplicities().items())]


FLAVOURED_TIERS = {"flavoured": (2,), "flavoured_su3": (3,), "flavoured_su4": (4,),
                   "flavoured_su5": (5,), "flavoured_su6": (6,), "flavoured_su2su2": (2, 2),
                   "flavoured_su3su2": (3, 2), "flavoured_su3su3": (3, 3),
                   "flavoured_su4su2": (4, 2), "flavoured_su2su2su2": (2, 2, 2)}

# a record reads as the unflavoured one (the grammar); the shipped strings
# name a PRODUCER, which is kept beside it as the entry's `source`.
_SHIPPED_TO_RECORD = {"covariant_spec:blocks": "green", "covariant_spec:strip": "green"}


def shipped_record(cert: str) -> str:
    """A shipped flavoured certification in the grammar: a block-mutation
    spec is a negating sequence (`green`), an order-search spec was accepted by
    the check to degree D (`axioms@D`), and the crystalline records keep their
    spelling."""
    if cert in _SHIPPED_TO_RECORD:
        return _SHIPPED_TO_RECORD[cert]
    if cert.startswith("covariant_spec:order@"):
        return "axioms@" + cert.split("@", 1)[1]
    return cert


def _flavoured_entry_json(B_c, fam_c, spec, omega, record, source):
    from quiver_enumeration import arrows_string
    fq = fold_orbits(B_c, fam_c)
    colours = [0] * len(B_c)
    for orb in fam_c:
        for v in orb:
            colours[v] = len(orb)
    out = {"q": arrows_string(B_c), "colours": colours, "family": [list(o) for o in fam_c],
           "reduced": [list(r) for r in fq.pairing], "factors": list(fq.flavour.factors),
           "certification": record, "source": source}
    if spec is not None:
        out["spec"] = [[list(g), [list(p) for p in r]] for g, r in spec]
    if omega is not None:
        out["omega"] = omega
    return out


def assemble_strongly_connected_flavoured(out_dir: str | Path, *, max_weight: int = 12,
                                          tiers: dict | None = None,
                                          enumerated_dir: str | Path | None = None,
                                          flavoured_root: str | Path | None = None,
                                          log=print) -> dict:
    """Every strongly connected flavoured quiver of each tier through
    unfolded weight `max_weight`, read off the enumerated tier, each with the best
    covariant spec available WITHOUT a search:

      1. the shipped tier's entry for it — its covariant spec or its crystalline
         `Ω`, moved into the labelling, its record in the grammar;
      2. else the enumerated tier's spec, regrouped as is or after sliding
         commuting factors, with the enumerated record (the same element);
      3. else ``pending``: the block search (on Symmetry), listed with the
         reason in `pending.json`.

    Writes `out_dir/<tier>/f_n{NN}_e{EE}.json` (lists of entries) and
    `out_dir/<tier>/manifest.json`, and returns the counts per tier and source."""
    import re as _re
    from dictionary_loader import expand_entry, iter_flavoured_entries
    from quiver_enumeration import is_strongly_connected, matrix_from_arrows
    tiers = dict(FLAVOURED_TIERS if tiers is None else tiers)
    enum_dir = Path(enumerated_dir) if enumerated_dir is not None else Path(_REPO) / "dictionaries" / "enumerated"
    root = Path(flavoured_root) if flavoured_root is not None else Path(_REPO) / "dictionaries"
    out = Path(out_dir)
    shipped: dict[str, dict] = {}
    for t, sizes in tiers.items():
        shipped[t] = {}
        for ent in iter_flavoured_entries(root / t):
            if not is_strongly_connected(ent["exchange"]):
                continue
            key, B_c, fam_c, spec, omega = shipped_entry_in_key_labelling(ent)
            shipped[t][key] = (B_c, fam_c, spec, omega, ent["certification"])
        log(f"  {t}: {len(shipped[t])} strongly connected shipped entries")
    cells: dict[str, dict] = {t: {} for t in tiers}
    seen: dict[str, set] = {t: set() for t in tiers}
    counts: dict[str, dict] = {t: {} for t in tiers}
    pending: dict[str, list] = {t: [] for t in tiers}
    for path in sorted(enum_dir.glob("q_n*_e*.json.gz")):
        n, e = map(int, _re.match(r"q_n(\d+)_e(\d+)", path.name).groups())
        if e > max_weight:
            continue
        with gzip.open(path, "rt") as fh:
            body = json.load(fh)
        for x in body["entries"]:
            B = matrix_from_arrows(n, x["q"])
            if not equal_row_classes(B):
                continue
            spec_u = None
            for t, sizes in tiers.items():
                for fam in orbit_family_choices(B, sizes):
                    key, cf = strongly_connected_flavoured_key(B, fam)
                    if key in seen[t]:
                        continue
                    seen[t].add(key)
                    B_c = [list(r) for r in cf.key]
                    fam_c = canonical_family(cf, fam)
                    if key in shipped[t]:
                        _B, _f, spec, omega, cert = shipped[t][key]
                        rec = _flavoured_entry_json(B_c, fam_c, spec, omega, shipped_record(cert),
                                                    f"shipped {t} ({cert})")
                        src = "shipped"
                    else:
                        got = None
                        if x.get("s") is not None or x.get("v") is not None:
                            if spec_u is None:
                                spec_u = expand_entry(x, n)["spec"]
                            spec_c = [tuple(int(g[p]) for p in cf.perm) for g in spec_u]
                            got = regroup_unfolded_spec(B_c, fam_c, spec_c)
                        if got is not None:
                            rec = _flavoured_entry_json(B_c, fam_c, got[0], None, x["c"],
                                                        f"enumerated tier, grouped {got[1]}")
                            src = f"enumerated, grouped {got[1]}"
                        else:
                            why = ("no spec in the enumerated tier" if spec_u is None
                                   else "the enumerated spec does not group")
                            rec = _flavoured_entry_json(B_c, fam_c, None, None, "pending",
                                                        f"pending: block search ({why})")
                            src = "pending"
                            pending[t].append({"q": rec["q"], "family": rec["family"], "n": n, "e": e,
                                               "why": why})
                    counts[t][src] = counts[t].get(src, 0) + 1
                    cells[t].setdefault((n, e), []).append(rec)
        log(f"  read {path.name}")
    for t in tiers:
        d = out / t
        d.mkdir(parents=True, exist_ok=True)
        for (n, e), ents in sorted(cells[t].items()):
            ents.sort(key=lambda r: (r["q"], r["colours"]))
            tmp = d / f"f_n{n:02d}_e{e:02d}.json.tmp"
            tmp.write_text(json.dumps(ents))
            os.replace(tmp, d / f"f_n{n:02d}_e{e:02d}.json")
        missing = sorted(set(map(str, shipped[t])) - {str(k) for k in seen[t]})
        manifest = {"plan": "41_enumerated_quiver_dictionary", "decision": "D55",
                    "what": "the strongly connected flavoured quivers of this tier (a strongly "
                            "connected quiver with a family of disjoint node orbits of sizes "
                            f"{list(tiers[t])}), unfolded weight <= {max_weight}, read off the "
                            "enumerated tier; not shipped",
                    "sizes": list(tiers[t]), "max_weight": max_weight,
                    "counts": counts[t], "n_entries": sum(counts[t].values()),
                    "shipped_entries_not_found": len(missing)}
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
        (d / "pending.json").write_text(json.dumps(pending[t]))
        log(f"  {t}: {manifest['n_entries']} entries {counts[t]}; shipped entries not found: {len(missing)}")
    return counts


def verify_strongly_connected_flavoured(tier_dir: str | Path, *, depth: int = 5, workers: int = 1,
                                        timeout: int = 600, stop_at: float | None = None,
                                        log=print) -> dict:
    """The verification pass, on the strongly connected entries only (composed
    answers inherit it): every entry with a covariant spec and no axiom record at
    `depth` or deeper is checked with `is_flavoured_spec` on the cone to degree
    `depth` (so found on `depth - 1`, confirmed one wider), and its record
    becomes `green;axioms@depth` / `axioms@depth`, or `…;axiom_mismatch@depth`
    listed in the manifest.  Cell by cell, each cell written atomically as it
    finishes, so a stop loses at most the cell in flight; resumable."""
    import time as _time
    from quiver_enumeration import fork_pool
    d = Path(tier_dir)
    counts: dict[str, int] = {}

    def needs(rec) -> bool:
        c = rec.get("certification", "")
        if rec.get("spec") is None or "mismatch" in c:
            return False
        if "@" in c and not c.startswith("crystalline"):
            try:
                if int(c.rsplit("@", 1)[1]) >= depth:
                    return False
            except ValueError:
                pass
        return True

    for path in sorted(d.glob("f_n*_e*.json")):
        if stop_at is not None and _time.time() >= stop_at:
            log(f"  stop time reached before {path.name}")
            break
        ents = json.loads(path.read_text())
        todo = [i for i, r in enumerate(ents) if needs(r)]
        if not todo:
            continue
        tasks = [(ents[i]["reduced"], ents[i]["factors"], ents[i]["spec"], depth, timeout) for i in todo]
        if workers > 1:
            with fork_pool(workers) as pool:
                results = pool.map(_verify_flavoured_task, tasks, chunksize=1)
        else:
            results = [_verify_flavoured_task(t) for t in tasks]
        for i, status in zip(todo, results):
            r = ents[i]
            base = r["certification"].split(";")[0] if r["certification"] != "pending" else "green"
            if status == "ok":
                r["certification"] = (f"green;axioms@{depth}" if base.startswith("green")
                                      else f"axioms@{depth}")
            elif status == "mismatch":
                r["certification"] = (f"green;axiom_mismatch@{depth}" if base.startswith("green")
                                      else f"axiom_mismatch@{depth}")
            counts[status] = counts.get(status, 0) + 1
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ents))
        os.replace(tmp, path)
        log(f"  verified {path.name}: {len(todo)} entries; so far {counts}")
    return counts


def _verify_flavoured_task(task) -> str:
    """One entry's check in a forked worker: 'ok', 'mismatch', 'timeout' or 'error'."""
    import signal as _signal
    from flavoured_spec import is_flavoured_spec
    reduced, factors, spec, depth, timeout = task
    fq = FlavouredQuiver(reduced, factors)
    sp = [(tuple(g), tuple(tuple(p) for p in r)) for g, r in spec]

    def _alarm(*_):
        raise TimeoutError

    old = _signal.signal(_signal.SIGALRM, _alarm)
    _signal.alarm(int(timeout))
    try:
        ok, _d = is_flavoured_spec(fq, sp, depth - 1)
        return "ok" if ok else "mismatch"
    except TimeoutError:
        return "timeout"
    except Exception:  # noqa: BLE001 — recorded, never hidden
        return "error"
    finally:
        _signal.alarm(0)
        _signal.signal(_signal.SIGALRM, old)


FLAVOURED_SC_FORMAT = 3

_FLAVOURED_SC_COVERAGE = (
    "every flavoured quiver - a quiver with a family of disjoint node orbits of these sizes, "
    "connected or not, of any rank - whose strongly connected components are answered: a plain "
    "component by the enumerated tier (dictionaries/enumerated/), a component carrying orbits, with "
    "the orbits it carries, by the flavoured tier of those sizes, through its own weight; "
    "lookup_flavoured composes them sources first, each orbit of single-node components placed "
    "consecutively (the design notes section 1)")


def _group_label(sizes) -> str:
    return "x".join(f"SU({s})" for s in sizes)


def ship_strongly_connected_flavoured(assembled_dir: str | Path, ship_dir: str | Path, *,
                                      max_weight: int, gzip_level: int = 9) -> Path:
    """A flavoured tier in shard format 3, holding ONLY the
    strongly connected flavoured quivers of one orbit-size multiset.  Every
    other flavoured quiver is answered by composition in the loader
    (`dictionary_loader.lookup_flavoured`).

    The entries come from an assembled set (`assemble_strongly_connected_flavoured`),
    one tier's directory of it, through unfolded weight `max_weight`.  That
    weight must be SETTLED there: every entry at or below it has a covariant spec
    or its crystalline `Ω`, and none is `pending`.  This is checked before
    anything is written.

    Cells `f_n{NN}_e{EE}.json.gz` (unfolded rank and weight) hold compact
    entries `{q, family, c, source, spec?, omega?}`, in the labelling: `q` is
    the arrow string of the canonical form coloured by orbit size, and `family`
    is the orbits in its node indices.  The directory ends up holding exactly
    these cells, so a format-2 tier's cells are removed.  A format-2 manifest
    already in `ship_dir` supplies the census of every flavoured quiver it
    held, kept as a record (`census_of_every_flavoured_quiver`); a format-3
    manifest carries its own forward."""
    import gzip as _gzip
    import io as _io
    import re as _re
    src, dst = Path(assembled_dir), Path(ship_dir)
    am = json.loads((src / "manifest.json").read_text())
    sizes = [int(s) for s in am["sizes"]]
    cells_in = []
    unsettled = {}
    for path in sorted(src.glob("f_n*_e*.json")):
        m = _re.match(r"f_n(\d+)_e(\d+)\.json$", path.name)
        if not m:
            continue
        n, e = int(m.group(1)), int(m.group(2))
        if e > max_weight:
            continue
        ents = json.loads(path.read_text())
        bad = sum(1 for r in ents if r.get("certification") == "pending"
                  or (r.get("spec") is None and not str(r.get("certification", "")).startswith("crystalline")))
        if bad:
            unsettled[f"n={n},e={e}"] = bad
        cells_in.append((n, e, ents))
    if unsettled:
        raise ValueError(f"ship_strongly_connected_flavoured(): weight <= {max_weight} is not settled "
                         f"in {src} - entries with neither a covariant spec nor the crystalline Omega, "
                         f"per cell: {unsettled}")
    census = None
    old_manifest = dst / "manifest.json"
    if old_manifest.exists():
        om = json.loads(old_manifest.read_text())
        if om.get("format") == 2:
            census = {"note": "the retired format-2 tier's census of every flavoured quiver it held "
                              "(count = every connected quiver with the orbit family, acyclic = those "
                              "with no directed cycle, stored = the cyclic ones), kept as a record and "
                              "as the cross-check of the composition",
                      "max_weight": om.get("max_weight"), "quivers": om.get("quivers"),
                      "acyclic": om.get("acyclic"), "stored": om.get("stored"),
                      "certification_census": om.get("certification_census"),
                      "cells": om.get("cells")}
        elif om.get("format") == FLAVOURED_SC_FORMAT:
            census = om.get("census_of_every_flavoured_quiver")
    dst.mkdir(parents=True, exist_ok=True)
    label = _group_label(sizes)
    completeness = (f"every strongly connected BPS quiver with total unfolded arrow weight "
                    f"e <= {max_weight}, together with a family of disjoint node orbits of sizes "
                    f"{sizes} (a manifest {label}), up to isomorphism")
    written = set()
    cells = {}
    by_cert, by_source = {}, {}
    n_entries = n_spec = n_cry = 0
    for n, e, ents in cells_in:
        out = []
        for r in sorted(ents, key=lambda r: (r["q"], r["family"])):
            c = {"q": r["q"], "family": r["family"], "c": r["certification"], "source": r["source"]}
            if r.get("spec") is not None:
                c["spec"] = r["spec"]
                n_spec += 1
            if r.get("omega") is not None:
                c["omega"] = r["omega"]
            if str(r["certification"]).startswith("crystalline"):
                n_cry += 1
            by_cert[r["certification"]] = by_cert.get(r["certification"], 0) + 1
            head = r["source"].split(" (")[0]
            by_source[head] = by_source.get(head, 0) + 1
            out.append(c)
        shard = {"plan": "41_enumerated_quiver_dictionary", "format": FLAVOURED_SC_FORMAT,
                 "strongly_connected": True, "sizes": sizes,
                 "orbit_size": sizes[0] if len(sizes) == 1 else sizes,
                 "n": n, "e": e, "completeness": completeness,
                 "entry_keys": "q = arrow string of the unfolded quiver in the canonical form (coloured "
                               "by orbit size); family = its orbits, in those node indices; c = "
                               "certification (the grammar); source = where the spec or Omega came "
                               "from; spec = [[charge on the fold, irrep], ...]; omega = the covariant "
                               "crystalline content when no spec",
                 "count": len(out), "entries": out}
        path = dst / f"f_n{n:02d}_e{e:02d}.json.gz"
        raw = _io.BytesIO()
        with _gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=gzip_level, mtime=0) as f:
            f.write(json.dumps(shard, separators=(",", ":")).encode())
        path.write_bytes(raw.getvalue())
        written.add(path)
        cells[f"n={n},e={e}"] = {"stored": len(out)}
        n_entries += len(out)
    for stale in list(dst.glob("f_n*_e*.json")) + list(dst.glob("f_n*_e*.json.gz")):
        if stale not in written:
            stale.unlink()
    manifest = {
        "plan": "41_enumerated_quiver_dictionary",
        "format": FLAVOURED_SC_FORMAT,
        "strongly_connected": True,
        "decision": "D55, D57",
        "doublets": len(sizes),
        "orbit_size": sizes[0] if len(sizes) == 1 else sizes,
        "sizes": sizes,
        "group": label,
        "max_weight": int(max_weight),
        "completeness": completeness,
        "coverage": _FLAVOURED_SC_COVERAGE,
        "composed_record": "as the enumerated tier's (dictionary_loader.compose_certification), "
                           "over the components of two or more nodes: green iff every one is; axioms@D "
                           "at the least depth; the least crystalline_only@D if one is; a mismatch "
                           "carried",
        "certification": "green;axioms@D (a block-mutation or regrouped negating sequence, checked on "
                         "the cone to degree D) | axioms@D (an order-search spec accepted to degree D) "
                         "| crystalline_only@D (no covariant spec found; Omega to depth D) | "
                         "crystalline_failed@D (Omega not a G-character: a counterexample)",
        "n_entries": n_entries,
        "n_with_spec": n_spec,
        "n_crystalline_only": n_cry,
        "certification_census": dict(sorted(by_cert.items())),
        "source_census": dict(sorted(by_source.items())),
        "census_of_every_flavoured_quiver": census,
        "cells": cells,
        "builder": "dictionaries/build_flavoured.py (ship_strongly_connected_flavoured), from "
                   "assemble_strongly_connected_flavoured",
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--doublets", type=int, default=1,
                    help="how many node ORBITS the quiver must have (one = 1)")
    ap.add_argument("--orbit-size", type=int, default=2,
                    help="size of the node orbit: 2 for SU(2), 3 for SU(3), ... "
                         "NOT a doublet count -- three identical nodes contain "
                         "three doublet PAIRS and are one orbit of size 3")
    ap.add_argument("--max-weight", type=int, default=6)
    ap.add_argument("--max-rank", type=int, default=None)
    ap.add_argument("--cutoff", type=int, default=4, help="spec-search cone degree")
    ap.add_argument("--depth", type=int, default=3, help="crystalline Omega depth")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-weight", type=int, default=0,
                    help="skip candidates below this weight (with --via-fold): "
                         "for extending a shipped tier, whose lower cells are "
                         "already on disk")
    ap.add_argument("--via-fold", action="store_true",
                    help="enumerate the FOLDS at max_weight-1 instead of the "
                         "unfolded quivers at max_weight; required from weight "
                         "10 up, where the unfolded walk does not fit in memory")
    ap.add_argument("--store-acyclic", action="store_true",
                    help="store the acyclic entries too instead of coding them")
    ap.add_argument("--ship", default="dictionaries/flavoured")
    args = ap.parse_args(argv)

    print(f"flavoured dictionary: SU({args.orbit_size}) "
          f"orbits={args.doublets} "
          f"max_weight={args.max_weight} workers={args.workers} "
          f"acyclic={'stored' if args.store_acyclic else 'coded'}", flush=True)
    path = build_cells(doublets=args.doublets, max_weight=args.max_weight,
                       max_rank=args.max_rank, cutoff=args.cutoff,
                       depth=args.depth, trials=args.trials,
                       out_dir=args.ship, workers=args.workers,
                       acyclic_coded=not args.store_acyclic,
                       via_fold=args.via_fold, min_weight=args.min_weight,
                       orbit_size=args.orbit_size)
    man = json.loads((path / "manifest.json").read_text())
    total = sum(f.stat().st_size for f in path.glob("*.gz"))
    print(f"\n{man['quivers']} quivers  ({man['acyclic']} acyclic coded, "
          f"{man['stored']} stored)")
    for k, v in man["certification_census"].items():
        print(f"   {k:44s} {v}")
    print(f"\nshipped to {path}   gzipped total {total} B ({total/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
