"""Builder for the ENUMERATED BPS-quiver dictionary.

The entry is the quiver: every connected quiver with total arrow weight
`e = Σ_{i<j}|b_ij| ≤ E` (arrows with multiplicity), up to node permutation,
enumerated by `quiver_enumeration` — plus a *provenance tail* of quivers
beyond `E` reached from physical seeds.  Specs are annotations, finite exact
certificates for `S`, populated by three producers in cost order, never
accepted by their origin:

  1. propagation from seeds with a known spec — local moves (necklacing at
     the head / tail), Fomin–Zelevinsky mutation transport, single-node drops
     (the entry-level primitives of `dictionaries/build.py`);
  2. the cluster-side bidirectional BFS (`BPSQuiver.find_negating_sequence`);
  3. order search on the crystalline `S` (`recursive_spectrum.
     extract_spec_from_quiver`), accepted with its depth.

ACCEPTANCE.  *"It is probably sensible to try build
assuming every conjecture is true, and then any contradiction encountered when
building is a conjecture test, and after the build happened one can use it to
test further"*; *"checking for greenness of a spec is easier than checking the
axioms!"*  So a candidate spec is accepted by whichever of two routes holds:

  * **green** — it replays as a genuine negating sequence
    (`spec_acceptance.is_green_sequence`, the cluster-side check: exact,
    finite, combinatorial).  The build ASSUMES the conjecture that a green
    spec satisfies the `S` axioms (`S_cluster = S`, measured 2381/2381 and
    now on this whole set).
  * **axioms@D** — for a spec that is NOT green (any factor order; user:
    *"it does not even need to be green … Any spec which matches the S
    axiomatics, with whatever factor order"*): the axiom check
    `spec_acceptance.accept_spec` at cone depth `D` — the BPS quiver
    constraint `S = 1 − 𝖖 Σ_a X_a + O(𝖖²)` on the truncated cone + agreement
    with the crystalline `S`.  Finite-depth, and the expensive one.

The axiom check is ALSO the post-build verification pass
(`stage_verify_axioms`, `--verify-axioms D`): every green spec re-tested at
depth `D`, a mismatch recorded loudly in the manifest (`axiom_mismatches`) —
a conjecture counterexample, never dropped silently.  The certification string
records exactly what held: `green`, `green;axioms@D`, `green;axiom_mismatch@D`,
`axioms@D`, `crystalline_only@D`.

Entries no producer reaches carry the crystalline factor content to a stated
depth (`certification = "crystalline_only@D"`).  The mutation-class index is
DERIVED: components of the mutation graph inside the enumerated set
(`quiver_enumeration.mutation_components`), refined by the `I_{id,id}`
fingerprint (the design record, μ → 1 layer) where computed.

THREE TIERS OF OUTPUT:

  * **permanent** — the shipped set, `--ship` (default
    `dictionaries/enumerated/`, TRACKED): per cell a gzipped
    `q_n{NN}_e{EE}.json.gz` of minimal entries `{exchange, spec | null,
    certification}` plus a small `manifest.json`; 0.17 MB for `E = 7`.
    `dictionary_loader` reads it transparently.
  * **semi-permanent, ancillary** — the rich build, `--out` (default
    `dictionaries/enumerated_build/`, untracked, regenerable): everything
    below, useful while the dictionary is being built or studied.
  * **transient** — timings, the propagation log.

Rich build outputs (under `--out`):

  * `q_n{NN}_e{EE}.json` — one file per cell, a flat list of entry dicts whose
    keys EXTEND the `dictionary_loader` shape (`name`, `exchange`, `spec` —
    `null` when no accepted spec —, `negating_sequence`, `provenance`) with
    `n`, `e`, `aut_order`, `flavour_rank`, `acyclic`, `exits`, `specs`
    (the accepted family, each with producer / provenance / acceptance
    record / green-sequence flag), `content` + `content_depth`,
    `certification`, `class_id`, `fingerprint`, `fingerprint_status`;
  * `tail_n{NN}.json` — provenance-tail entries (`e > E`), same shape;
  * `classes_E{E}.json` — the class table;
  * `manifest.json` — parameters, counts per cell, the completeness statement,
    the propagation log, and any spec REJECTED by the axiom check (a
    propagated spec failing the axioms is a finding, not noise).

Run from the repo root:

    PYTHONPATH=. python dictionaries/build_enumerated.py --E 5
    PYTHONPATH=. python dictionaries/build_enumerated.py --E 7 --verify-axioms 5 --verify-strip-depth 3 --ship
    PYTHONPATH=. python dictionaries/build_enumerated.py --ship-from dictionaries/enumerated_build --ship
    PYTHONPATH=. python dictionaries/build_enumerated.py --E 9 --verify-axioms 5 --resume   # continue an interrupted build

The rich build in `--out` is a CHECKPOINT: written after every stage and every
`--checkpoint-every` entries inside the long per-entry stages, and `--resume`
loads it and continues (stages done are skipped; an interrupted stage re-runs
over the entries it had not finished).

Every stage is idempotent on its inputs and whole cells are written at once.
"""

from __future__ import annotations

import argparse
import gc
import heapq
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from quiver_enumeration import (  # noqa: E402
    ALPHABET, CanonicalForm, Matrix, QuiverRecord, Vec, arrows_string, canonical_form,
    canonical_spec, cell_counts, compact_record, enumerate_connected_quivers,
    enumerate_strongly_connected_quivers, fork_pool, intern_vec, is_acyclic, is_connected,
    is_strongly_connected, mutation_components, relabel_spec, sequence_string, standard_nodes,
    total_weight,
)
from spec_acceptance import (  # noqa: E402
    accept_spec, content_record, crystalline_spectrum, is_green_sequence,
    sequence_from_spec, spec_from_sequence, strip_spec,
)
from bps_quiver_dictionary import QuiverEntry  # noqa: E402
from snf_kernel import integer_kernel_and_section  # noqa: E402

DEFAULT_OUT = Path(_HERE) / "enumerated_build"     # rich, untracked
DEFAULT_SHIP = Path(_HERE) / "enumerated"          # permanent, tracked, gzipped
SHIPPED_KEYS = ("exchange", "spec", "certification")     # shard format 1
COMPACT_KEYS = ("q", "s", "v", "c")                      # shard format 2 (below)
SHARD_FORMAT = 2


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


_AXIOMS_INTERN: dict[tuple, dict] = {}


def _intern_axioms(d: dict | None) -> dict | None:
    """One shared dict per distinct axiom-check record.  The records are
    read (`["ok"]`, `["depth"]`) or replaced whole, never mutated, and
    take a handful of distinct values across a build, so sharing them by
    content costs nothing and saves a dict per spec."""
    if d is None:
        return None
    k = tuple(sorted(d.items()))
    r = _AXIOMS_INTERN.get(k)
    if r is None:
        r = _AXIOMS_INTERN[k] = d
    return r


_CONTENT_INNER: dict[tuple, dict] = {}


def _intern_content(c: dict | None) -> dict | None:
    """A content record (`content_record`: charge string -> {exponent:
    coefficient}) with its charge strings and inner dicts SHARED by value.

    Content records are only ever assigned whole or read, never mutated, so
    sharing is safe, and the JSON written is byte-identical.  Unshared, a
    weight-13 record costs ~50-65 KB in the parent: 2.6 M of them would not
    fit a 180 GB node, and the build could not be loaded afterwards
    (measured 2026-09-28: 2,191 distinct charge strings and 41 distinct inner
    dicts over the probe's weight-13 records, ~7 KB per record shared)."""
    if c is None:
        return None
    out = {}
    for k, inner in c.items():
        t = tuple(inner.items())
        r = _CONTENT_INNER.get(t)
        if r is None:
            r = _CONTENT_INNER[t] = inner
        out[sys.intern(k)] = r
    return out


@dataclass
class SpecRecord:
    spec: tuple[Vec, ...]           # canonical node coordinates
    producer: str                   # 'strip' | 'propagation' | 'bfs' | 'order_search' | 'seed'
    provenance: str
    green: bool                     # replays as a negating sequence (exact, finite)
    axioms: dict | None = None      # the axiom check record (spec_acceptance.Acceptance.as_dict), if run
    propagated: bool = False        # already used as a propagation source (stage 1b / 2b / 3b)

    def __post_init__(self) -> None:
        # share the charge vectors (unit vectors, repeated across every
        # spec), the producer / provenance strings and the axiom record —
        # the spec families are the largest item of a build's memory
        # (1.5 KB of 3.4 KB per entry at weight 8 before sharing)
        self.spec = intern_vec(tuple(intern_vec(tuple(g)) for g in self.spec))
        self.producer = sys.intern(self.producer)
        self.provenance = sys.intern(self.provenance)
        self.axioms = _intern_axioms(self.axioms)

    @property
    def certification(self) -> str:
        if self.green:
            if self.axioms is None:
                return "green"
            d = self.axioms["depth"]
            return f"green;axioms@{d}" if self.axioms["ok"] else f"green;axiom_mismatch@{d}"
        return f"axioms@{self.axioms['depth']}"

    def as_dict(self) -> dict:
        return {
            "spec": [list(g) for g in self.spec],
            "producer": self.producer,
            "provenance": self.provenance,
            "green": self.green,
            "axioms": self.axioms,
            "propagated": self.propagated,
            "certification": self.certification,
        }


@dataclass
class EntryData:
    key: Matrix
    n: int
    e: int
    aut_order: int
    automorphisms: tuple[Vec, ...]
    acyclic: bool
    flavour_rank: int
    exits: Vec
    in_set: bool
    specs: list[SpecRecord] = field(default_factory=list)
    content: dict | None = None
    content_depth: int | None = None
    provenance: list[str] = field(default_factory=list)
    class_id: int | None = None
    fingerprint: str | None = None
    fingerprint_status: str | None = None
    # The producers that have already FAILED on this entry, and under which
    # budget: e.g. {"bfs": {"outcome": "none", "depth": 12, "timeout": 30}}.
    # A stage that spans several nights re-derives only what was not stored, and
    # a failure used to store nothing, so every resume re-ran every earlier
    # failure (7,613 BFS failures at E = 10 on 2026-09-21).  Written only when
    # non-empty, so an entry without one serialises exactly as before; None
    # until the first failure, since an empty dict on each of E10's 4.9 M
    # entries would cost ~300 MB for nothing.
    attempts: dict | None = None
    # The family test's record (stage_verify_axioms, families=True): the depth it
    # compared every alternative spec with the primary at, and how many specs the
    # entry had then.  A re-run skips the entry while both still hold, so the
    # test resumes across job links instead of starting over (2026-09-28).
    # Written only when set, like `attempts`.
    family_checked: dict | None = None
    owned: bool = True              # on a shard: this entry's per-entry stages run HERE (not serialised)

    @property
    def primary(self) -> SpecRecord | None:
        """The spec that ships: green preferred over non-green, then
        the simplest — shortest, then lexicographic."""
        if not self.specs:
            return None
        return min(self.specs, key=lambda s: (not s.green, len(s.spec), s.spec))

    @property
    def certification(self) -> str:
        p = self.primary
        if p is not None:
            return p.certification
        if self.content is not None:
            return f"crystalline_only@{self.content_depth}"
        return "none"

    def name(self) -> str:
        return f"n{self.n}_e{self.e}_" + "_".join(
            "".join(f"{x:+d}" for x in row) for row in self.key)

    def as_dict(self) -> dict:
        p = self.primary
        d = {
            "name": self.name(),
            "exchange": [list(r) for r in self.key],
            "spec": [list(g) for g in p.spec] if p is not None else None,
            "negating_sequence": None,
            "provenance": "; ".join(self.provenance),
            "n": self.n, "e": self.e,
            "aut_order": self.aut_order,
            "flavour_rank": self.flavour_rank,
            "acyclic": self.acyclic,
            "exits": list(self.exits),
            "in_set": self.in_set,
            "specs": [s.as_dict() for s in
                      sorted(self.specs, key=lambda s: (not s.green, len(s.spec), s.spec))],
            "content": self.content,
            "content_depth": self.content_depth,
            "certification": self.certification,
            "class_id": self.class_id,
            "fingerprint": self.fingerprint,
            "fingerprint_status": self.fingerprint_status,
        }
        if self.attempts:
            d["attempts"] = self.attempts
        if self.family_checked:
            d["family_checked"] = self.family_checked
        return d


# ---------------------------------------------------------------------------
# Timeouts (Unix main thread; a no-op elsewhere)
# ---------------------------------------------------------------------------


class _Timeout(TimeoutError):
    """`_with_timeout`'s alarm.  A `TimeoutError`, so a callee that re-raises time
    limits (`build_flavoured.covariant_value`) lets it through instead of
    reading it as a search outcome."""


class StopRequested(Exception):
    """Raised inside a per-entry loop when the build has been asked to stop —
    by `--stop-at`, or by SIGTERM / SIGINT.  `run()` catches it, writes a full
    checkpoint and returns, so an overnight build that is interrupted at a
    fixed hour (or by a machine restart's shutdown signal) loses nothing but
    the entries since the last write.  The stage it was in is NOT recorded as
    done, and every stage is idempotent per entry, so `--resume` re-derives
    only what was not stored."""


def _with_timeout(seconds: int | None, fn: Callable, *args, **kwargs):
    """Run `fn` with a SIGALRM deadline; returns `(result, "ok")`,
    `(None, "timeout")` or `(None, "err: …")`."""
    if not seconds or not hasattr(signal, "SIGALRM"):
        try:
            return fn(*args, **kwargs), "ok"
        except Exception as exc:  # noqa: BLE001 — recorded, never hidden
            return None, f"err: {type(exc).__name__}: {str(exc)[:80]}"

    def handler(*_):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        return fn(*args, **kwargs), "ok"
    except _Timeout:
        return None, "timeout"
    except Exception as exc:  # noqa: BLE001
        return None, f"err: {type(exc).__name__}: {str(exc)[:80]}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# Verification workers (stage 3b with `workers > 1`)
# ---------------------------------------------------------------------------


def _verify_primary_task(task):
    """One entry's primary spec through the axiom check at its depth, in a
    forked worker; returns `(key, record | None, status)` — the record is
    `spec_acceptance.Acceptance.as_dict()`, exactly what the sequential pass
    stores."""
    key, spec, depth, timeout = task
    acc, status = _with_timeout(timeout, accept_spec, key, spec, depth)
    if acc is None:
        return key, None, status
    return key, acc.as_dict(), "ok"


def _bfs_task(task):
    """One entry through the bidirectional negating-sequence search, in a
    forked worker; returns `(key, spec | None, status)`."""
    key, bfs_depth, timeout = task[:3]
    strategy = task[3] if len(task) > 3 else "bfs"
    from bps_quiver_tools import BPSQuiver
    n = len(key)

    def run():
        Q = BPSQuiver(charges=standard_nodes(n), frozen=[False] * n,
                      exchange_matrix=[list(r) for r in key])
        seq = Q.find_negating_sequence(max_depth=bfs_depth, strategy=strategy)
        if seq is None:
            return None
        return [tuple(g) for g in Q.build_spectrum_generator(seq)]

    spec, status = _with_timeout(timeout, run)
    return key, spec, status


def _content_task(task):
    """One entry's crystalline content (`stage_content`), in a forked worker;
    returns `(key, in_set, depth, content record | None, status)`."""
    key, depth, timeout, in_set = task
    res, status = _with_timeout(timeout, crystalline_spectrum, key, depth,
                                with_multiplicities=True)
    # only the record travels back: the spectrum itself is not stored
    return key, in_set, depth, (content_record(res[1]) if status == "ok" else None), status


def _order_search_task(task):
    """One entry through the order search on the crystalline `S`, in a
    forked worker; returns `(key, spec | None, status)`."""
    key, cutoff, timeout = task
    from recursive_spectrum import extract_spec_from_quiver
    spec, status = _with_timeout(timeout, extract_spec_from_quiver,
                                 [list(r) for r in key], standard_nodes(len(key)), cutoff=cutoff)
    return key, (None if spec is None else [tuple(g) for g in spec]), status


def _verify_family_task(task):
    """One entry's alternative specs against its primary on the cone to
    `depth`, in a forked worker; returns `(key, [(spec, n_differing | None,
    status), …])`."""
    from spec_acceptance import spec_product, spectra_agree
    key, primary, others, depth, timeout = task
    n = len(key)
    Sp, status = _with_timeout(timeout, spec_product, key, primary, depth)
    if status != "ok":
        return key, [(o, None, f"primary_{status}") for o in others]
    out = []
    for o in others:
        So, status = _with_timeout(timeout, spec_product, key, o, depth)
        if status != "ok":
            out.append((o, None, status))
            continue
        out.append((o, len(spectra_agree(Sp, So, n, depth)), "ok"))
    return key, out


# ---------------------------------------------------------------------------
# The enumeration cache (T16): the stage-0 records saved and reloaded
# ---------------------------------------------------------------------------


def save_records(records: dict, path: str | Path, *, kind: str = "connected") -> Path:
    """Write the enumeration records (canonical matrix, automorphisms,
    acyclicity, exits) as compact JSON, so a restart or a shard reloads them
    in seconds instead of re-enumerating (14 min at weight 9).  `kind` names
    the set — every `connected` quiver, or the `strongly connected` ones
 — so a cache is never read into a build of the other kind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"key": [list(r) for r in rec.key], "n": rec.n, "e": rec.e,
             "aut_order": rec.aut_order, "automorphisms": [list(a) for a in rec.automorphisms],
             "acyclic": rec.acyclic, "exits": list(rec.exits)}
            for rec in records.values()]
    path.write_text(json.dumps({"E": max((r.e for r in records.values()), default=0),
                                "kind": kind, "count": len(rows), "records": rows}))
    return path


def load_records(path: str | Path, E: int, *, kind: str = "connected") -> dict:
    """Read a records cache written by `save_records`; refuses a cache built
    for a different weight bound or a different set."""
    d = json.loads(Path(path).read_text())
    if d.get("E") != E:
        raise ValueError(f"enumeration cache {path} is for E = {d.get('E')}, not {E}")
    if d.get("kind", "connected") != kind:
        raise ValueError(f"enumeration cache {path} holds the {d.get('kind', 'connected')} "
                         f"quivers, not the {kind} ones")
    out = {}
    for row in d["records"]:
        key = tuple(tuple(int(x) for x in r) for r in row["key"])
        rec = compact_record(QuiverRecord(
            key=key, n=row["n"], e=row["e"], aut_order=row["aut_order"],
            automorphisms=tuple(tuple(int(x) for x in a) for a in row["automorphisms"]),
            acyclic=bool(row["acyclic"]), exits=tuple(row["exits"])))
        out[rec.key] = rec
    if len(out) != d["count"]:
        raise ValueError(f"enumeration cache {path}: {len(out)} records read, {d['count']} announced")
    return out


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


def _flavour_rank(B: Matrix) -> int:
    if len(B) == 0:
        return 0
    ker, _ = integer_kernel_and_section([list(r) for r in B])
    return len(ker)


def _restandardize_fast(pairing: Sequence[Sequence[int]], nodes: Sequence[Vec],
                        spec: Sequence[Vec]):
    """`dictionaries.build._restandardize` for the basis a necklace step
    produces — the standard basis with one node `k` negated and every other
    node `j` shifted by an integer multiple `m_j` of `e_k` (the
    Fomin–Zelevinsky mutation of the node charges at `k`).  There the new
    pairing is `M·P·Mᵀ` in integers, and a charge `g` has coordinates
    `c_a = g_a` (`a ≠ k`), `c_k = Σ_{j≠k} m_j g_j − g_k` in the new basis —
    closed form, no rational elimination.  Any other basis shape goes to
    the generic routine.  Returns `(B_std, spec_std)` or `None` when a spec
    entry is not in the non-negative span, exactly as the generic one
    (asserted on every child at weight 5 in the test suite)."""
    from dictionaries.build import _restandardize
    n = len(nodes)

    def generic():
        return _restandardize([list(r) for r in pairing], list(nodes), list(spec))

    if any(len(row) != n for row in nodes):
        return generic()
    k = None
    for j, row in enumerate(nodes):
        if row[j] == -1:
            if k is not None:
                return generic()
            k = j
    if k is None:
        return generic()
    m = [0] * n
    for j, row in enumerate(nodes):
        for a, x in enumerate(row):
            if j == k:
                if x != (-1 if a == k else 0):
                    return generic()
            elif a == j:
                if x != 1:
                    return generic()
            elif a == k:
                m[j] = x
            elif x != 0:
                return generic()
    spec_std = []
    for g in spec:
        c = list(g)
        c[k] = sum(m[j] * g[j] for j in range(n) if j != k) - g[k]
        if any(x < 0 for x in c):
            return None
        spec_std.append(tuple(int(x) for x in c))
    # M·P·Mᵀ for this basis IS the Fomin–Zelevinsky mutation of the pairing
    # at k (3,252 of 3,252 necklace bases at weight 5 agree with the cubic
    # product; the control test keeps checking against the generic routine)
    B_std = [list(r) for r in _mutate(pairing, k)]
    return B_std, spec_std


def propagation_children(B: Matrix, spec: Sequence[Vec], *, max_states: int = 64,
                         max_extra_length: int = 4, drops: bool = True, fast: bool = True,
                         canonical: bool = False, replay: bool = False):
    """The closed-form children of one source `(B, spec)` in the standard
    basis: `(child_B, child_spec, move)`.

    * `head{k}` / `tail{k}`: the necklace step at node `k` — mutation at `k`
      with the spec rotated — after the local-move rewrite (commuting swaps
      and pentagon expansions, `chart_graph.find_local_moves_for_targets`,
      one bulk search per source) that brings `k`'s charge to the head
      (forward step) or the tail (inverse step).  The necklace only ever
      acts at the head or tail, so without the rewrite a source hands a spec
      to two of its mutation neighbours; with it, to every neighbour whose
      charge the search can bring to an end within `max_states` states and
      `max_extra_length` net expansions.
    * `drop{k}`: the single-node RG flow (induced subquiver).

    Every move carries a green spec to a green spec (the local moves by the
    ruling, measured 395/395 + 1000/1000; the necklace and the
    drop by construction), so a child of a green source is stored on the
    replay alone.  The Fomin–Zelevinsky transport of a spec through a
    mutation at `k` is NOT a move here: a negating sequence contains every
    node charge and the transported spec must be re-expressed in the
    non-negative span of the new nodes, in which `k`'s charge is negated —
    it never applies (measured 0 of 23,598 at weight 6).

    `fast=True` takes the necklace steps through `_restandardize_fast`
    (closed form) instead of the generic re-standardisation — the same
    children (control test); `canonical=True` appends each child's
    `CanonicalForm` and, with `replay=True`, its green replay (`None`
    otherwise), so the parent's ingest need not recompute them.  The
    replay is worth doing here only in a worker pool: it runs on every
    child, while the parent replays only the children that survive the
    duplicate and family filters (about a third at weight 6)."""
    from chart_graph import (_apply_local_move, _necklace_forward, _necklace_inverse,
                             find_local_moves_for_targets)
    from dictionaries.build import _drop_node_entry, _necklace_step_entry
    n = len(B)
    spec_t = [tuple(int(x) for x in g) for g in spec]
    nodes = standard_nodes(n)
    Bl = [list(r) for r in B]
    out: list = []

    def emit(child_B, child_spec, move):
        child_B = tuple(tuple(int(x) for x in r) for r in child_B)
        child_spec = tuple(tuple(int(x) for x in g) for g in child_spec)
        if canonical:
            # the canonical form and the green replay (invariant under the
            # relabelling to canonical coordinates) are computed here, in
            # the worker, so the parent's ingest is left with the store
            out.append((child_B, child_spec, move, canonical_form(child_B),
                        is_green_sequence(child_B, child_spec) if replay else None))
        else:
            out.append((child_B, child_spec, move))

    heads, tails = find_local_moves_for_targets(
        spec_t, nodes, nodes, Bl, max_states=max_states, max_extra_length=max_extra_length)
    for which, chains, direction in (("head", heads, "forward"), ("tail", tails, "inverse")):
        for target, chain in chains.items():
            if chain is None:
                continue
            sp = [tuple(g) for g in spec_t]
            for mv in chain:
                sp = [tuple(g) for g in _apply_local_move(sp, mv)]
            k0 = nodes.index(tuple(target))
            move = f"{which}{k0}"
            if fast:
                if direction == "forward":
                    if sp[0] != nodes[k0]:
                        continue
                    new_spec, new_nodes, _, _, _ = _necklace_forward(sp, nodes, nodes[k0], nodes[k0], Bl)
                else:
                    if sp[-1] != nodes[k0]:
                        continue
                    new_spec, new_nodes, _, _, _ = _necklace_inverse(sp, nodes, nodes[k0], nodes[k0], Bl)
                res = _restandardize_fast(Bl, new_nodes, new_spec)
                if res is not None and res[1]:
                    emit(res[0], res[1], move)
            else:
                src = QuiverEntry(name="s", exchange=B, spec=sp, provenance="")
                ch = _necklace_step_entry(src, direction=direction)
                if ch is not None and ch.spec:
                    emit(ch.exchange, ch.spec, move)
    if drops and n > 1:
        src = QuiverEntry(name="s", exchange=B, spec=spec_t, provenance="")
        for k in range(n):
            ch = _drop_node_entry(src, k)
            if ch is not None and ch.spec:
                emit(ch.exchange, ch.spec, f"drop{k}")
    return out


def _propagate_task(task) -> list:
    """Worker unit of the propagation stage: the children of one source,
    each with its canonical form (computed here, not in the parent)."""
    B, spec, max_states, max_extra, drops, replay = task
    try:
        return propagation_children(B, spec, max_states=max_states, max_extra_length=max_extra,
                                    drops=drops, canonical=True, replay=replay)
    except Exception:            # a defective source must not kill the stage
        return []


def _entry_from_record(r: QuiverRecord, in_set: bool) -> EntryData:
    r = compact_record(r)                   # shared rows / automorphisms / exits
    return EntryData(key=r.key, n=r.n, e=r.e, aut_order=r.aut_order,
                     automorphisms=r.automorphisms, acyclic=r.acyclic,
                     flavour_rank=_flavour_rank(r.key), exits=r.exits,
                     in_set=in_set)



def _write_json_list(path: Path, items) -> None:
    """Write `items` (an iterable of JSON-serialisable dicts) as one JSON
    list, streamed one item at a time — byte-identical to
    `path.write_text(json.dumps(list(items)))`, without materialising the
    list or the whole string (T27).

    ATOMIC: streamed into a sibling `.tmp` and then `os.replace`d onto
    `path`, so no reader ever sees a half-written file.  The old writer
    truncated in place, and the resume path parses these with a bare
    `json.loads`: a kill landing inside a multi-minute checkpoint left an
    unparseable cell or manifest, after which *every* later `--resume` died
    on it and the whole build was unrecoverable (2026-09-21)."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("[")
            first = True
            for item in items:
                if not first:
                    f.write(", ")
                first = False
                f.write(json.dumps(item))
            f.write("]")
        os.replace(tmp, path)
    except BaseException:
        # Never leave a stray .tmp behind to be mistaken for a shard.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _fork_pool(workers: int, maxtasksperchild: int):
    """A fork-context `Pool` whose workers do not copy the parent's heap.

    A forked worker shares the parent's memory copy-on-write, but Python's
    cyclic garbage collector in the worker walks every object it inherited and
    writes to each one's header, which copies every page it touches.  Measured
    2026-09-21 on the E = 10 resume: with the parent at 14 GB and three workers
    recycled every 256 tasks, one worker reached 9.5 GB, the parent's heap went
    into the compressor, and swap grew by 10 GB in 48 minutes.  `gc.freeze()`
    right before the fork (the recipe in the `gc` documentation) puts every
    existing object beyond the collector's reach; measured on a 1.5 GB heap, a
    child's full collection then copied nothing (7 MB before and after, against
    7 MB -> 1,516 MB without).  It is O(1), so callers may repeat it.  Nothing
    is unfrozen: the frozen objects are the build's long-lived state and are
    still freed by reference counting; only cyclic garbage alive at the moment
    of a freeze is kept.  The workers die on SIGTERM however early it comes
    (`quiver_enumeration.fork_pool`)."""
    gc.freeze()
    return fork_pool(workers, maxtasksperchild)


def resolve_stop_at(text: str, now: float | None = None) -> float:
    """`--stop-at` as epoch seconds, resolved when the process STARTS.

    Two forms.  ``YYYY-MM-DDTHH:MM`` (a space instead of the ``T`` also
    works) is that moment exactly and never rolls over — the form the nightly
    driver passes, because only the driver knows which morning closes the
    window (Monday, for a start on Friday evening).  ``HH:MM`` is the next
    such time after `now`: today if it is still ahead, else tomorrow.

    It used to be resolved AFTER the resume load, which took about an hour at
    E = 10 under the old Background priority, so a build started shortly
    before its stop time rolled over to the NEXT morning and would have run
    through the working day.  Raises `ValueError` on anything else."""
    import datetime as _dt
    now = time.time() if now is None else now
    s = text.strip().replace(" ", "T")
    if "T" in s:
        return _dt.datetime.strptime(s, "%Y-%m-%dT%H:%M").timestamp()
    hh, mm = (int(x) for x in s.split(":"))
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError(f"--stop-at {text!r}: not a time of day")
    lt = time.localtime(now)
    at = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
    if at <= now:                       # passed today: tomorrow (mktime normalises the day)
        at = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday + 1, hh, mm, 0, 0, 0, -1))
    return at


_THERMAL_LEVELS = {0: "nominal", 1: "moderate", 2: "heavy", 3: "trapping", 4: "sleeping"}


def peak_rss_gb() -> float:
    """This process's peak resident memory in GB (macOS reports `ru_maxrss` in
    bytes, Linux in KiB) — the parent's, which holds the build; logged at each
    checkpoint for the footprint question."""
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r if sys.platform == "darwin" else r * 1024) / 2**30


def thermal_pressure() -> str | None:
    """macOS's thermal-pressure level — ``nominal``, ``moderate``, ``heavy``,
    ``trapping`` or ``sleeping`` — or None off macOS or on any failure.

    Logged with every checkpoint because the nightly build's heat cap is its
    WORKER COUNT (the author chose 3 of 10 cores for the heat), and a level
    above ``nominal`` is macOS saying the machine is heat-limited, i.e. that
    the cap should come down.  Temperature and power readings need
    administrator rights; this one does not."""
    if sys.platform != "darwin":
        return None
    try:
        import subprocess
        out = subprocess.run(["/usr/bin/notifyutil", "-g", "com.apple.system.thermalpressurelevel"],
                             capture_output=True, text=True, timeout=5).stdout.split()
        level = int(out[-1])
    except Exception:  # noqa: BLE001 — a missing reading never stops a build
        return None
    return _THERMAL_LEVELS.get(level, str(level))


def _write_text_atomic(path: Path, text: str) -> None:
    """`path.write_text(text)` via a sibling `.tmp` + `os.replace`, for the
    same reason as `_write_json_list` — `manifest.json` in particular is the
    file `--resume` reads first, so a torn write there wedges the build."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise

class EnumeratedBuild:
    """The pipeline.  Call the `stage_*` methods in order, or `run()`."""

    def __init__(
        self, E: int, *,
        accept_depth: int = 5,
        verify_depth: int = 0,
        verify_strip_depth: int | None = None,
        verify_workers: int = 1,
        workers: int | None = None,
        content_depth: int = 6,
        tail_max_rank: int = 12,
        tail_max_weight: int | None = None,
        tail_accept_depth: int = 3,
        max_tail_entries: int = 5000,
        specs_per_entry: int = 3,
        family_max: int | None = None,
        mutation_max_rank: int = 12,
        bfs_depth: int = 12,
        per_entry_timeout: int = 20,
        local_move_states: int = 64,
        local_move_extra: int = 4,
        log: Callable[[str], None] | None = print,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int = 2000,
        checkpoint_interval: float = 3600.0,
        enumeration_cache: str | Path | None = None,
        shard: tuple[int, int] | None = None,
        order_search_timeout: int | None = None,
        order_search_cutoff: int | None = None,
        strongly_connected: bool = False,
    ):
        self.E = E
        # the entries are the STRONGLY CONNECTED quivers only — every
        # other quiver is answered by composing its components — found by the
        # walk that climbs past weight 10.
        self.strongly_connected = strongly_connected
        self.accept_depth = accept_depth          # the axiom check for NON-green candidates
        self.verify_depth = verify_depth          # 0 = no post-build axiom verification of green specs
        self.verify_strip_depth = verify_strip_depth   # None = min(3, verify_depth)
        self.verify_workers = verify_workers           # > 1: the pass runs in forked workers
        # `workers` drives every per-entry stage that has a worker task (the
        # BFS fallback, the order search, the verification pass); it defaults
        # to `verify_workers` so the older flag keeps its meaning.
        self.workers = workers if workers is not None else verify_workers
        self.content_depth = content_depth
        self.tail_max_rank = tail_max_rank
        self.tail_max_weight = 2 * E if tail_max_weight is None else tail_max_weight
        self.tail_accept_depth = tail_accept_depth
        self.max_tail_entries = max_tail_entries
        self.specs_per_entry = specs_per_entry
        self.family_max = family_max              # hard cap on specs per entry (None: 2n + 1)
        self.mutation_max_rank = mutation_max_rank
        self.bfs_depth = bfs_depth
        self.per_entry_timeout = per_entry_timeout
        self.local_move_states = local_move_states    # the local-move search budget per propagation source
        self.local_move_extra = local_move_extra
        self._log = log or (lambda s: None)
        # Checkpointing (for the long builds): the rich build is written to
        # `checkpoint_dir` after every stage and every `checkpoint_every`
        # entries inside the per-entry loops; `EnumeratedBuild.load()` reads
        # it back and `run(resume=True)` skips the stages already done.
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.checkpoint_every = checkpoint_every
        # WALL-CLOCK is the unit the durability rule is stated in, so it is the
        # unit the guard uses.  `checkpoint_every` counts ENTRIES, so whether it
        # honours "save at least hourly" depends entirely on how fast entries
        # happen to flow — which is why `--checkpoint-every 0` (chosen at E = 10
        # because per-entry writes cost more than the work) produced a 51-hour
        # stretch with nothing on disk.  `checkpoint_interval` is independent of
        # throughput: 0 disables it, and it composes with `checkpoint_every`.
        self.checkpoint_interval = float(checkpoint_interval or 0.0)
        self._last_checkpoint = time.time()
        # T17: `(i, K)` — this builder owns the entries with enumeration index
        # ≡ i (mod K); the per-entry stages run on those only, propagation
        # every shard holds EVERY entry and runs the strip + propagation
        # closure over all of them (a deterministic function of the
        # enumeration, so the shards agree); it OWNS the entries with index
        # ≡ i mod K, on which alone the per-entry stages (BFS, order search,
        # verification, content) run; the class index (whole set) is left to
        # `merge_shards`, which takes each entry from its owner.
        if shard is not None:
            i, K = shard
            if not (K >= 1 and 0 <= i < K):
                raise ValueError(f"shard must be (i, K) with 0 <= i < K, got {shard}")
        self.shard = shard
        # T18: the order search on the crystalline S (producer 3) with its own
        # budget — None = the per-entry timeout / the content depth.  It is the
        # one producer that can yield a NON-green spec, and the only door
        # left for the entries the BFS and the node drops never reach.
        self.order_search_timeout = order_search_timeout
        self.order_search_cutoff = order_search_cutoff
        # T16: where stage 0 saves / reloads its records (None = always enumerate)
        self.enumeration_cache = Path(enumeration_cache) if enumeration_cache else None
        # A build asked to stop (a wall-clock deadline, or a signal) raises
        # StopRequested from the next per-entry checkpoint hook; `run()` flushes.
        self.stop_at: float | None = None          # epoch seconds
        self._stop_signal: str | None = None
        self.stopped_early: str | None = None
        # A stage that cannot be interrupted is not STARTED when the deadline
        # is closer than this: stage_class_index's mutation_components
        # call runs to completion whatever the clock says.
        self.class_index_reserve: float = 3600.0
        # A recorded timeout, error or unaccepted find is skipped on resume at
        # the same depth and time limit; set this to try those again, e.g.
        # after the machine got faster.  A recorded "none" is final at
        # its depth whatever this says: only a deeper search retries it.
        self.retry_timeouts: bool = False
        # "bfs 1234/81502" while a long per-entry stage runs, for the
        # checkpoint and stop log lines.
        self._progress: str | None = None
        # The family comparison of the verification pass (every alternative
        # spec of an entry against its primary, at the verify depth) is a
        # consistency test, not a condition of shipping; `--no-family-test`
        # leaves it out.  At weight 9 it is ≈ 2 M spec products at depth 5 —
        # days on three workers, with no progress line and no checkpoint.
        self.family_test: bool = True
        # The order search (producer 3) can be switched off (the author, 2026-09-22:
        # "skip the order search at weight 10"): at weight 9 it found 0 specs in 653
        # entries, every one a timeout or an exhaustive none.  A skipped stage is
        # recorded in `skipped_stages`, and so in both manifests, so a tier never
        # claims a search it did not run.
        self.order_search: bool = True
        self.skipped_stages: list[str] = []
        # BFS's own per-entry time limit (None: the build's `per_entry_timeout`),
        # and the stage reopenings already applied to this build, by id.
        self.bfs_timeout: int | None = None
        # the search behind producer 2: "bfs", the bidirectional BFS to
        # `bfs_depth`, or "auto", `find_spec_auto`'s seeded recursion (depth
        # `bfs_depth`), which reaches the long green sequences of high-rank
        # strongly connected quivers a depth-12 BFS cannot (2n - 2 at n >= 8)
        self.bfs_strategy: str = "bfs"
        self.reruns_applied: list[dict] = []
        # T15: incremental mid-stage checkpoints — the cells touched since the
        # last write to `checkpoint_dir`; a stage-boundary checkpoint and the
        # first checkpoint of a fresh build write everything.
        self._dirty_cells: set[tuple[int, int]] = set()
        self._tail_dirty = False
        self._checkpoint_complete = False
        self.stages_done: list[str] = []

        self.records: dict[Matrix, QuiverRecord] = {}
        self.entries: dict[Matrix, EntryData] = {}        # the set, e <= E
        self.tail: dict[Matrix, EntryData] = {}           # e > E, reached from seeds
        self.crystalline_cache: dict[tuple[Matrix, int], dict] = {}
        self.rejected: list[dict] = []                    # non-green candidates failing the axiom check
        self.axiom_mismatches: list[dict] = []            # GREEN specs failing the axiom check: findings
        self.family_mismatches: list[dict] = []           # two accepted specs of one quiver with different S: findings
        self.log_lines: list[str] = []
        self.stats: dict[str, int] = {}
        self.classes: list[dict] = []
        self.timings: dict[str, float] = {}

    # ----------------------------------------------------------------- utils

    def log(self, msg: str) -> None:
        self.log_lines.append(msg)
        self._log(msg)

    def _bump(self, k: str, by: int = 1) -> None:
        self.stats[k] = self.stats.get(k, 0) + by

    CRYSTALLINE_CACHE_SIZE = 256

    def _crystalline(self, key: Matrix, depth: int):
        """The crystalline `S` of an entry, cached across the candidate specs
        of the SAME entry only: a bounded LRU, because an unbounded cache of
        exact spectra at `E = 7` (14,593 entries × ~10³ Habiro elements) is
        what silently killed the first full build."""
        ck = (key, depth)
        cache = self.crystalline_cache
        if ck in cache:
            cache[ck] = cache.pop(ck)
            return cache[ck]
        S = crystalline_spectrum(key, depth)
        cache[ck] = S
        while len(cache) > self.CRYSTALLINE_CACHE_SIZE:
            cache.pop(next(iter(cache)))
        return S

    def _lookup(self, key: Matrix) -> EntryData | None:
        return self.entries.get(key) or self.tail.get(key)

    def _checkpoint(self, stage: str | None) -> None:
        """Write the rich build to `checkpoint_dir` (no-op without one); with
        a `stage` name, record that stage as done first."""
        if stage is not None and stage not in self.stages_done:
            self.stages_done.append(stage)
        if self.checkpoint_dir is None:
            return
        t = time.time()
        # Read before the write so the manifest counts this checkpoint's level
        # (stats `thermal_<level>`: how many checkpoints saw each level).
        level = thermal_pressure()
        if level is not None:
            self._bump(f"thermal_{level}")
        incremental = stage is None and self._checkpoint_complete
        n_cells = self.write(self.checkpoint_dir, incremental=incremental)
        # Measured from the END of the write: a full checkpoint took 1025 s on
        # the E10 build, and timing the interval from the start would re-arm it
        # the instant it finished.
        self._last_checkpoint = time.time()
        extra = (f"; {self._progress}" if self._progress else "") + \
                (f"; thermal {level}" if level is not None else "") + \
                f"; peak RSS {peak_rss_gb():.1f} GB"
        self.log(f"  checkpoint{' after ' + stage if stage else ''} → {self.checkpoint_dir} "
                 f"({'%d cells' % n_cells if incremental else 'full'}, {time.time() - t:.1f}s){extra}")

    def install_stop_handlers(self, stop_at: float | None = None) -> None:
        """Ask this build to stop at `stop_at` (epoch seconds) and on
        SIGTERM / SIGINT.  The stop is cooperative: the next per-entry
        checkpoint hook raises `StopRequested` and `run()` flushes a full
        checkpoint, so nothing is lost but the entries since the last write.
        A second signal is left to the default handler, so an impatient
        operator can still kill the process outright."""
        self.stop_at = stop_at

        def _handler(signum, frame):
            self._stop_signal = {getattr(signal, n, None): n
                                 for n in ("SIGTERM", "SIGINT")}.get(signum, str(signum))
            self.log(f"  {self._stop_signal} received; stopping at the next entry")
            signal.signal(signum, signal.SIG_DFL)      # a second one kills

        for name in ("SIGTERM", "SIGINT"):
            sig = getattr(signal, name, None)
            if sig is not None:
                signal.signal(sig, _handler)

    def _stop_reason(self) -> str | None:
        if self._stop_signal is not None:
            return self._stop_signal
        if self.stop_at is not None and time.time() >= self.stop_at:
            return f"the --stop-at deadline ({time.strftime('%H:%M', time.localtime(self.stop_at))})"
        return None

    PIPELINE = ("strip", "propagate", "bfs", "propagate_after_bfs", "order_search",
                "propagate_after_order_search", "verify", "content", "class_index")

    def is_complete(self) -> bool:
        """Every stage of this build's pipeline done, or switched off
        (`skipped_stages`; `verify` only counts when a verify depth is set).
        The nightly driver reads it, as `complete` in the manifest, to tell a
        finished build from one with a reopened stage."""
        done = set(self.stages_done) | set(self.skipped_stages)
        return all(st in done for st in self.PIPELINE
                   if st != "verify" or self.verify_depth)

    def reopen(self, stages: Sequence[str], rerun_id: str) -> bool:
        """Reopen finished `stages`, so that the next `run(resume=True)` does
        them again — at most ONCE per `rerun_id`: the id is recorded in
        `reruns_applied`, so the nightly driver can pass the same request every
        night and only the first launch reopens anything; later launches just
        resume the reopened work.

        Reopening `propagate` marks every stored spec as a source again, so the
        closure runs once more from the whole set, with whatever tail cap and
        seeds the launch has (the weight-10 build's tail of 5,000 discarded
        12.8 million higher-weight quivers with specs).  A reopened `bfs`
        retries only what the recorded failures allow: a larger
        `bfs_timeout` retries the timeouts, never an exhausted `none`.
        Returns whether anything was reopened."""
        if any(r.get("id") == rerun_id for r in self.reruns_applied):
            self.log(f"rerun {rerun_id!r} was already applied; nothing reopened")
            return False
        unknown = [st for st in stages if st not in self.PIPELINE]
        if unknown:
            raise ValueError(f"cannot reopen unknown stage(s) {unknown}; "
                             f"the stages are {list(self.PIPELINE)}")
        for st in stages:
            if st in self.stages_done:
                self.stages_done.remove(st)
        if "propagate" in stages:
            n = 0
            for ent in list(self.entries.values()) + list(self.tail.values()):
                for rec in ent.specs:
                    if rec.propagated:
                        rec.propagated = False
                        n += 1
                if ent.specs:
                    self._touch(ent)
            self.log(f"rerun {rerun_id!r}: {n:,} specs are propagation sources again")
        self.reruns_applied.append({"id": rerun_id, "stages": list(stages),
                                    "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.log(f"rerun {rerun_id!r}: reopened {', '.join(stages)}")
        return True

    def _refuse_if_deadline_near(self, stage: str, reserve: float) -> None:
        """Stop INSTEAD of starting `stage` when the `--stop-at` deadline is
        less than `reserve` seconds away.  `stage` cannot be interrupted
        once it has started, so a late start would run on past the deadline and
        into the author's working day.  The stop is the ordinary one: a full
        checkpoint now, and `stage` runs first thing on the next resume."""
        if self.stop_at is not None and self.stop_at - time.time() < reserve:
            left = (self.stop_at - time.time()) / 60
            raise StopRequested(f"{stage} not started: the --stop-at deadline is {left:.0f} min away, "
                                f"under the {reserve / 60:.0f} min it is given, and it cannot be "
                                f"interrupted once started")

    def _already_failed(self, ent: EntryData, producer: str, depth: int,
                        timeout: int | None, strategy: str | None = None) -> bool:
        """Has `producer` already failed on `ent` under a budget at least as
        large as this one?  A ``none`` is final at its depth, since the
        search exhausted it, so only a deeper search tries again.  A timeout,
        an error or an unaccepted find also depends on the time limit and on
        how fast the machine was, so a larger time limit or `retry_timeouts`
        tries again."""
        a = (ent.attempts or {}).get(producer)
        if not a or a.get("depth", -1) < depth:
            return False                       # never tried, or a deeper search now
        if strategy is not None and a.get("strategy", "bfs") != strategy:
            return False                       # tried by another search
        if a.get("outcome") == "none":
            return True
        if self.retry_timeouts:
            return False
        inf = float("inf")                     # a time limit of 0 / None is no limit
        return (a.get("timeout") or inf) >= (timeout or inf)

    def _record_failure(self, ent: EntryData, producer: str, outcome: str, depth: int,
                        timeout: int | None, strategy: str | None = None) -> None:
        """Remember on the entry, persisted with it, that `producer` failed
        on it and under which budget — and which search — so a
        resumed stage can skip it."""
        if ent.attempts is None:
            ent.attempts = {}
        ent.attempts[producer] = {"outcome": outcome, "depth": depth, "timeout": int(timeout or 0)}
        if strategy is not None and strategy != "bfs":
            ent.attempts[producer]["strategy"] = strategy
        self._touch(ent)

    @staticmethod
    def _forget_failure(ent: EntryData, producer: str) -> None:
        """`producer` has now given `ent` a spec: drop its failure record."""
        if ent.attempts:
            ent.attempts.pop(producer, None)
            if not ent.attempts:
                ent.attempts = None

    def _maybe_checkpoint(self, i: int) -> None:
        reason = self._stop_reason()
        if reason is not None:
            raise StopRequested(reason)
        if self.checkpoint_dir is None:
            return
        by_count = self.checkpoint_every and i % self.checkpoint_every == 0
        by_clock = (self.checkpoint_interval
                    and time.time() - self._last_checkpoint >= self.checkpoint_interval)
        if by_count or by_clock:
            self._checkpoint(None)

    # --------------------------------------------------------------- stage 0

    def stage_enumerate(self) -> None:
        t = time.time()
        cache = self.enumeration_cache
        kind = "strongly connected" if self.strongly_connected else "connected"
        if cache is not None and cache.exists():
            self.records = load_records(cache, self.E, kind=kind)
            source = f"from the cache {cache}"
        else:
            enumerate_set = (enumerate_strongly_connected_quivers if self.strongly_connected
                             else enumerate_connected_quivers)
            self.records = enumerate_set(
                self.E, progress=lambda e, c: self.log(f"  enumerate e={e}: +{c}"),
                workers=self.workers)
            source = "enumerated"
            if cache is not None:
                save_records(self.records, cache, kind=kind)
                source += f", cached to {cache}"
        if not self.entries:
            for idx, (key, r) in enumerate(self.records.items()):
                ent = _entry_from_record(r, in_set=True)
                if self.shard is not None:
                    ent.owned = idx % self.shard[1] == self.shard[0]
                self.entries[key] = ent
        else:
            for idx, key in enumerate(self.records):
                ent = self.entries.get(key)
                if ent is not None:
                    ent.owned = self.shard is None or idx % self.shard[1] == self.shard[0]
        self.timings["enumerate"] = time.time() - t
        n_owned = sum(1 for ent in self.entries.values() if ent.owned)
        shard_note = (f"; shard {self.shard[0]}/{self.shard[1]}: {n_owned} entries owned here"
                      if self.shard is not None else "")
        self.log(f"stage 0: {len(self.records)} {kind} quivers with e <= {self.E} "
                 f"({source}, {self.timings['enumerate']:.1f}s){shard_note}")

    # ------------------------------------------------- dirty tracking (T15)

    def _touch(self, ent: EntryData) -> None:
        """Record that `ent`'s cell must be rewritten at the next checkpoint."""
        if ent.in_set:
            self._dirty_cells.add((ent.n, ent.e))
        else:
            self._tail_dirty = True

    # --------------------------------------------------------- spec storage

    def _store_spec(self, ent: EntryData, spec_canon: tuple[Vec, ...], producer: str,
                    provenance: str, *, depth: int | None = None,
                    green: bool | None = None) -> bool:
        """Canonicalise under `Aut`, apply the family policy, run the
        acceptance check, store.  Returns True iff a NEW spec was stored.
        `green` is the replay's verdict when the caller already has it (the
        propagation workers replay each child; the verdict is invariant
        under relabelling)."""
        spec_c = canonical_spec(spec_canon, ent.automorphisms)
        if any(s.spec == spec_c for s in ent.specs):
            self._bump("spec_duplicate")
            return False
        n = ent.n
        heads = {s.spec[0] for s in ent.specs if s.spec}
        tails = {s.spec[-1] for s in ent.specs if s.spec}
        if self.family_max is not None and len(ent.specs) >= self.family_max:
            self._bump("spec_family_full")           # the hard cap (--family-max)
            return False
        if len(ent.specs) >= self.specs_per_entry:
            new_end = (spec_c and (spec_c[0] not in heads or spec_c[-1] not in tails))
            if not new_end or len(ent.specs) >= 2 * n + 1:
                self._bump("spec_family_full")
                return False
        # Route 1 — green: the exact, finite replay.  Accepted on the
        # conjecture that a green spec satisfies the S axioms.
        if green is None:
            green, gstatus = _with_timeout(self.per_entry_timeout, is_green_sequence, ent.key, spec_c)
            if gstatus != "ok":
                self._bump(f"green_check_{gstatus.split(':')[0]}")
                green = False
        axioms = None
        if green:
            # A green spec is stored on the replay alone; the axiom check on it
            # is the post-build verification stage's business (right depth per
            # producer, checkpointed) unless a caller asks for it here with
            # `depth=`.  Checking inline at `verify_depth` cost the weight-8
            # build ~0.4 s per strip spec, uncheckpointed (the design notes 2026-09-13).
            if depth:
                axioms = self._axiom_check(ent, spec_c, depth, producer, provenance)
                if axioms is None:
                    return False          # timeout / error, recorded
            rec = SpecRecord(spec=spec_c, producer=producer, provenance=provenance,
                             green=True, axioms=axioms)
            if axioms is not None and not axioms["ok"]:
                self._bump("axiom_mismatch_green")
            else:
                self._bump(f"spec_accepted_{producer}")
            ent.specs.append(rec)
            ent.family_checked = None            # a new spec: the family is compared again
            self._touch(ent)
            return True
        # Route 2 — not green: only the axiom check can accept it.
        d = depth if depth is not None else (
            self.accept_depth if ent.in_set else self.tail_accept_depth)
        axioms = self._axiom_check(ent, spec_c, d, producer, provenance)
        if axioms is None:
            return False
        if not axioms["ok"]:
            self._bump("spec_rejected")
            self.rejected.append({
                "name": ent.name(), "exchange": [list(r) for r in ent.key],
                "spec": [list(g) for g in spec_c], "producer": producer,
                "provenance": provenance, "acceptance": axioms,
            })
            return False
        if producer == "propagation":
            self._bump("propagated_non_green")     # worth knowing: a necklace/drop result that is not green
        self._touch(ent)
        ent.specs.append(SpecRecord(spec=spec_c, producer=producer, provenance=provenance,
                                    green=False, axioms=axioms))
        ent.family_checked = None                # a new spec: the family is compared again
        self._bump(f"spec_accepted_{producer}")
        return True

    def _axiom_check(self, ent: EntryData, spec_c: tuple[Vec, ...], depth: int,
                     producer: str, provenance: str) -> dict | None:
        """`spec_acceptance.accept_spec` at `depth`, timed; a GREEN spec that
        fails is a conjecture counterexample and goes to `axiom_mismatches`."""
        acc, status = _with_timeout(self.per_entry_timeout, accept_spec, ent.key,
                                    spec_c, depth, crystalline=self._crystalline(ent.key, depth))
        if acc is None:
            self._bump(f"axiom_check_{status.split(':')[0]}")
            self.log(f"  axiom check {status} on {ent.name()} ({producer})")
            return None
        rec = acc.as_dict()
        if not rec["ok"] and is_green_sequence(ent.key, spec_c):
            self.axiom_mismatches.append({
                "name": ent.name(), "exchange": [list(r) for r in ent.key],
                "spec": [list(g) for g in spec_c], "producer": producer,
                "provenance": provenance, "acceptance": rec,
            })
            self.log(f"  !! AXIOM MISMATCH on a green spec: {ent.name()} ({producer}) at depth {depth}")
        return rec

    # --------------------------------------------------------------- stage 1

    def stage_seed_acyclic(self) -> None:
        """The free producer: every acyclic quiver's source/sink strip spec
        (a theorem — `S = ∏ E_𝖖(X_{γ_a})` in that order), accepted as green by
        the replay; the axiom check on it is the verification pass's business
        (the acyclic entries are ~85 % of the set and the check at depth 5
        costs ~0.4 s each at rank 8 — measured, the design notes)."""
        t = time.time()
        n_ok = 0
        for i, ent in enumerate(self.entries.values(), 1):
            self._maybe_checkpoint(i)
            if not ent.acyclic:
                continue
            sp = strip_spec(ent.key)
            if sp is None:
                continue
            if self._store_spec(ent, tuple(sp), "strip", "source/sink order (acyclic)"):
                n_ok += 1
        self.timings["strip"] = time.time() - t
        self.log(f"stage 1a: strip specs on {n_ok} acyclic entries ({self.timings['strip']:.1f}s)")

    def _ingest(self, entry: QuiverEntry, producer: str, provenance: str,
                cf: CanonicalForm | None = None, green: bool | None = None
                ) -> tuple[EntryData | None, bool]:
        """A standard-basis `(exchange, spec)` candidate → its entry (set or
        tail), storing the spec if it is new and accepted.  `cf` is the
        candidate's canonical form when the caller already has it (the
        propagation workers compute it)."""
        B = entry.exchange
        n = len(B)
        if n == 0 or not is_connected(B):
            self._bump("ingest_disconnected")
            return None, False
        if self.strongly_connected and not is_strongly_connected(B):
            self._bump("ingest_not_strongly_connected")    # answered by composition
            return None, False
        e = total_weight(B)
        if e > self.E and (n > self.tail_max_rank or e > self.tail_max_weight):
            self._bump("ingest_out_of_range")
            return None, False
        if cf is None:
            cf = canonical_form(B)
        ent = self._lookup(cf.key)
        if ent is None and cf.key in self.records:
            self._bump("ingest_other_shard")        # in the set, owned by another shard
            return None, False
        if ent is None:
            if len(self.tail) >= self.max_tail_entries:
                self._bump("tail_full")
                return None, False
            rec = QuiverRecord(key=cf.key, n=n, e=e, aut_order=cf.aut_order,
                               automorphisms=cf.automorphisms,
                               acyclic=_is_acyclic(cf.key),
                               exits=tuple(total_weight(_mutate(cf.key, k)) for k in range(n)))
            ent = _entry_from_record(rec, in_set=False)
            self.tail[cf.key] = ent
            self._tail_dirty = True
        if provenance and provenance not in ent.provenance and len(ent.provenance) < 8:
            ent.provenance.append(provenance)
            self._touch(ent)
        spec_c = relabel_spec(entry.spec, cf.perm)
        stored = self._store_spec(ent, spec_c, producer, provenance, green=green)
        return ent, stored

    def _to_quiver_entry(self, ent: EntryData, spec: SpecRecord, name: str) -> QuiverEntry:
        return QuiverEntry(name=name, exchange=ent.key, spec=spec.spec, provenance=spec.provenance)

    def seed_from_tier(self, dict_dir: str | Path) -> list[tuple[str, QuiverEntry]]:
        """An existing tier (shipped, or a rich build) as a starting point: its
        specs become propagation seeds, and the entries it has NO spec for are
        recorded as BFS failures, so the BFS stage does not repeat the tier's
        search on them (their crystalline content is still computed).  A
        strongly connected build reads only the tier's strongly connected
        entries — the rest are answered by composition.  A seed goes
        through the ordinary acceptance, the green replay: its origin carries
        no trust (A2).  A later `--bfs-timeout` larger than this build's does
        retry the imported failures (they are recorded as `tier`, not as an
        exhausted `none`).  Needs the enumeration (stage 0) loaded."""
        from dictionary_loader import iter_enumerated_entries
        name = f"tier:{Path(dict_dir).name}"
        bt = self.bfs_timeout or self.per_entry_timeout
        depth = self.bfs_depth
        mpath = Path(dict_dir) / "manifest.json"
        unsearched = False
        if mpath.exists():
            # the failure is the TIER's search, recorded at the depth and limit
            # it ran under (a format-3 manifest says), not this build's
            search = json.loads(mpath.read_text()).get("search") or {}
            # a floor of 0: some spec-less entries of the tier were never
            # searched, and a shipped entry does not say which, so none is
            # recorded as a failure (each is searched here instead)
            unsearched = search.get("bfs_depth") == 0 or bool(search.get("n_spec_less_unsearched"))
            depth = search.get("bfs_depth") or depth
            if isinstance(search.get("bfs_time_limit_s"), int):
                bt = search["bfs_time_limit_s"]
        seeds: list[tuple[str, QuiverEntry]] = []
        n_failed = 0
        for d in iter_enumerated_entries(dict_dir):
            B = tuple(tuple(int(x) for x in row) for row in d["exchange"])
            if self.strongly_connected and not is_strongly_connected(B):
                continue
            if d.get("spec"):
                spec = tuple(tuple(int(x) for x in g) for g in d["spec"])
                seeds.append((name, QuiverEntry(name=name, exchange=B, spec=spec, provenance=name)))
                continue
            if unsearched:
                continue
            ent = self._lookup(canonical_form(B).key)
            if ent is not None and not ent.specs and not (ent.attempts or {}).get("bfs"):
                self._record_failure(ent, "bfs", "tier", depth, bt)
                n_failed += 1
        self.log(f"  seeds: {len(seeds)} specs from {dict_dir}; {n_failed} entries it has no "
                 f"spec for are recorded as BFS failures")
        return seeds

    def import_attempts(self, dict_dir: str | Path) -> dict:
        """Search FACTS from builds run elsewhere, never specs: an
        attempts export — `n_*.json` lists of `{exchange, attempts: {bfs:
        {outcome, depth, timeout, strategy}}, source}` — is merged into the
        entries that still have no spec.  Per entry the stronger of the local
        and the imported BFS attempt is kept, deeper first, then `auto` above
        `bfs`, then the longer time limit, with the build it came from
        (`source`).  A spec always supersedes a failure record, and an imported
        `tier` stamp (a seeded build recording only that its tier had no spec)
        is not a search and is ignored.  Returns the counts."""
        def strength(a):
            return (a.get("depth") or 0, 1 if a.get("strategy", "bfs") == "auto" else 0,
                    a.get("timeout") or 0)
        counts = {"read": 0, "no_record": 0, "unknown": 0, "has_spec": 0, "kept_local": 0, "imported": 0}
        for path in sorted(Path(dict_dir).glob("n_*.json")):
            for d in json.loads(path.read_text()):
                counts["read"] += 1
                a = (d.get("attempts") or {}).get("bfs")
                if not a or a.get("outcome") == "tier":
                    counts["no_record"] += 1
                    continue
                ent = self._lookup(canonical_form(d["exchange"]).key)
                if ent is None or not ent.in_set:
                    counts["unknown"] += 1
                    continue
                if ent.specs:
                    counts["has_spec"] += 1
                    continue
                local = (ent.attempts or {}).get("bfs")
                if local and strength(local) >= strength(a):
                    counts["kept_local"] += 1
                    continue
                rec = {"outcome": a["outcome"], "depth": int(a["depth"]), "timeout": int(a.get("timeout") or 0)}
                if a.get("strategy", "bfs") != "bfs":
                    rec["strategy"] = a["strategy"]
                rec["source"] = d.get("source") or str(dict_dir)
                if ent.attempts is None:
                    ent.attempts = {}
                ent.attempts["bfs"] = rec
                self._touch(ent)
                counts["imported"] += 1
        self.log(f"  attempts from {dict_dir}: {counts}")
        return counts

    def import_verification(self, dict_dir: str | Path) -> dict:
        """Verification FACTS from a build run elsewhere, never specs: an
        export of `n_*.json` lists `{exchange, spec, axioms, source}`, `spec` a
        verified spec in canonical coordinates.  Its `axioms` record is attached
        to the local spec only where it IS that spec (canonical spec equality
        under the entry's automorphisms), never to "the same quiver": a quiver
        whose local build kept another spec stays as it is.  An equal or deeper
        local record is kept.  A local record that DISAGREES at the same depth
        (ok against not ok) is counted and left alone — a finding for a person,
        not a merge.  The attached record keeps its `source`.  Returns counts."""
        counts = {"read": 0, "unknown": 0, "no_spec_here": 0, "spec_differs": 0,
                  "kept_local": 0, "disagree": 0, "attached": 0}
        for path in sorted(Path(dict_dir).glob("n_*.json")):
            for d in json.loads(path.read_text()):
                counts["read"] += 1
                ent = self._lookup(canonical_form(d["exchange"]).key)
                if ent is None or not ent.in_set:
                    counts["unknown"] += 1
                    continue
                if not ent.specs:
                    counts["no_spec_here"] += 1
                    continue
                want = canonical_spec(tuple(tuple(int(x) for x in g) for g in d["spec"]),
                                      ent.automorphisms)
                rec = next((sp for sp in ent.specs if sp.spec == want), None)
                if rec is None:
                    counts["spec_differs"] += 1
                    continue
                a = dict(d["axioms"])
                local = rec.axioms
                if local is not None:
                    if local.get("depth") == a.get("depth") and bool(local.get("ok")) != bool(a.get("ok")):
                        counts["disagree"] += 1
                        self.log(f"  !! verification DISAGREES on {ent.name()} at depth {a.get('depth')}: "
                                 f"local ok={local.get('ok')}, {d.get('source')} ok={a.get('ok')}")
                        continue
                    if local.get("depth", 0) >= a.get("depth", 0):
                        counts["kept_local"] += 1
                        continue
                a["source"] = d.get("source") or str(dict_dir)
                rec.axioms = _intern_axioms(a)
                self._touch(ent)
                counts["attached"] += 1
        self.log(f"  verification from {dict_dir}: {counts}")
        return counts

    def stage_propagate(self, seeds: Iterable[tuple[str, QuiverEntry]] = (), *,
                        max_candidates: int | None = None, workers: int | None = None,
                        label: str = "1b") -> None:
        """Producer 1: EVERY stored spec is a source, not only the gauge-theory seeds.  The `seeds` are
        ingested first; then the frontier — every spec record in the set or
        the tail not yet marked `propagated` — is processed in rounds, best
        first by `(e, n)` so the enumerated set fills before the tail: each
        source's children (`propagation_children`: the necklace step at
        every node reachable by a local-move rewrite, and the node drops)
        are ingested, and every child entry that receives its FIRST spec
        joins the next round.  An entry is a source once, through its first
        spec (the alternative specs of a family regenerate the same
        children).  The closure stops when a round stores nothing.  Sources are
        generated in forked workers (`_propagate_task`) when `workers > 1`;
        the ingest is in the parent.  Re-runnable: `run()` calls it again
        after the BFS and the order search so their specs propagate too, and
        the `propagated` flag is checkpointed, so a resumed build does not
        redo the closure.  Measured (weight 6 / 7, no tail): every cyclic
        entry whose mutation class inside the bound has an acyclic member is
        reached, 220/220 and 1618/1618, all children green."""
        t = time.time()
        workers = self.workers if workers is None else workers
        n_seeds = 0
        for name, qe in seeds:
            n_seeds += 1
            self._ingest(qe, "seed", f"seed:{name}")
        processed = 0
        rounds = 0
        n_stored = 0
        budget_hit = False
        pool = None
        if workers > 1:
            pool = _fork_pool(workers, 4096)
        try:
            # one source per ENTRY: its first spec, unless the entry has already
            # been a source (a family's alternative specs regenerate the same
            # children — measured 4× the sources for nothing at weight 7)
            frontier = [(ent, ent.specs[0]) for ent in list(self.entries.values()) + list(self.tail.values())
                        if ent.specs and not any(s.propagated for s in ent.specs)]
            while frontier:
                frontier.sort(key=lambda es: (es[0].e, es[0].n))
                if max_candidates is not None:
                    room = max_candidates - processed
                    if room <= 0:
                        budget_hit = True
                        break
                    frontier = frontier[:room]
                rounds += 1
                in_pool = pool is not None and len(frontier) >= 4 * workers
                tasks = [(ent.key, s.spec, self.local_move_states, self.local_move_extra,
                          ent.n <= self.mutation_max_rank, in_pool) for ent, s in frontier]
                if in_pool:
                    # this round's new objects too, for workers recycled mid-round
                    gc.freeze()
                    results = pool.imap(_propagate_task, tasks, chunksize=32)
                else:
                    results = map(_propagate_task, tasks)
                nxt: list[tuple[EntryData, SpecRecord]] = []
                for (ent, s), children in zip(frontier, results):
                    processed += 1
                    # the hook BEFORE the flag: a stop raised here, or a checkpoint
                    # written here, leaves this source unflagged with its children
                    # not yet stored, so a resume redoes it; flagged first, a stop
                    # lost its children for good
                    self._maybe_checkpoint(processed)
                    s.propagated = True
                    self._touch(ent)
                    src_name = ent.name()
                    for child_B, child_spec, move, cf, green in children:
                        prov = f"{move}({src_name})"
                        qe = QuiverEntry(name=src_name, exchange=child_B, spec=child_spec, provenance=prov)
                        cent, stored = self._ingest(qe, "propagation", prov, cf=cf, green=green)
                        if cent is None or not stored:
                            continue
                        n_stored += 1
                        self._bump(f"propagation_{move.rstrip('0123456789')}")
                        if len(cent.specs) == 1:            # a NEW entry with a spec: a source next round
                            nxt.append((cent, cent.specs[-1]))
                frontier = nxt
        except BaseException:
            # A stop (or any error) must not wait for the rest of the round.
            # close() + join() lets the pool finish every task `imap` has queued,
            # whose results nobody will read: measured 2026-09-21, a stop 3 s into
            # a pooled round at E = 7 returned 20.5 s after its deadline, and a
            # round at E = 10 is of the order of an hour — past 07:00.
            # terminate() kills the workers at once; they die on its SIGTERM
            # because the pool initializer restored SIG_DFL in them.
            if pool is not None:
                pool.terminate()
                pool.join()
                pool = None
            raise
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        self.timings["propagate"] = self.timings.get("propagate", 0.0) + (time.time() - t)
        self.log(f"stage {label}: propagation from {n_seeds} seeds + every stored spec: "
                 f"{processed} sources in {rounds} rounds, {n_stored} new specs "
                 f"({time.time() - t:.1f}s); tail {len(self.tail)}"
                 + ("; candidate budget reached" if budget_hit else ""))

    # --------------------------------------------------------------- stage 2

    def _family_done(self, ent: EntryData, depth: int) -> bool:
        """The family test already compared this entry's specs at `depth` or
        deeper, and the entry has gained no spec since."""
        fc = ent.family_checked
        return bool(fc) and fc.get("depth", 0) >= depth and fc.get("n_specs") == len(ent.specs)

    def _mark_family_done(self, ent: EntryData, depth: int) -> None:
        ent.family_checked = {"depth": depth, "n_specs": len(ent.specs)}
        self._touch(ent)

    def _entries_without_spec(self) -> list[EntryData]:
        return [ent for ent in self.entries.values() if not ent.specs and ent.owned]

    def stage_bfs_fallback(self, workers: int | None = None, *,
                           select: Callable[[EntryData], bool] | None = None,
                           key: Callable[[EntryData], tuple] | None = None) -> None:
        """Producer 2: the cluster-side bidirectional BFS, budgeted.  With
        `workers > 1` the searches run in forked workers (`_bfs_task`) and
        the results are stored here in arrival order — the same specs, the
        same acceptance, the same checkpoints.  The search is
        `self.bfs_strategy`.  `select` limits the pass to the entries it
        accepts and `key` orders them (default weight, then rank): a partial
        pass, called directly, which the pipeline does not mark done."""
        from bps_quiver_tools import BPSQuiver
        t = time.time()
        n_ok = 0
        # BFS's own time limit: `bfs_timeout` when set, else the per-entry one
        # (the weight-10 timeouts retried at 300 s).
        bt = self.bfs_timeout or self.per_entry_timeout
        # Skip what BFS has already failed on under this depth and time limit
        #  The stage is not recorded as done until it finishes, and at
        # E = 10 it spans several nights, so without this every resume re-ran
        # every earlier failure first.
        todo_all = self._entries_without_spec()
        if select is not None:
            todo_all = [ent for ent in todo_all if select(ent)]
        strategy = self.bfs_strategy
        todo = [ent for ent in todo_all
                if not self._already_failed(ent, "bfs", self.bfs_depth, bt, strategy)]
        # Weight first, then rank.  Low rank is where BFS succeeds fast;
        # the strongly connected walk lists a weight's HIGH-rank quivers first
        # (long ears on small parents), and on the first SC12 night that put
        # ~10,000 rank 7-12 searches, most of the rank >= 10 ones 30 s timeouts,
        # ahead of ~500,000 rank 3-8 ones.  Stable: the enumeration order
        # stays within a cell.
        todo.sort(key=key or (lambda ent: (ent.e, ent.n)))
        known = (f" ({len(todo_all) - len(todo)} skipped: BFS already failed on them at this "
                 f"depth and time limit)" if len(todo) < len(todo_all) else "")
        workers = self.workers if workers is None else workers

        def absorb(ent, spec, status) -> int:
            if status != "ok":
                outcome = status.split(":")[0]            # "timeout" or "err"
                self._bump(f"bfs_{outcome}")
                self._record_failure(ent, "bfs", outcome, self.bfs_depth, bt, strategy)
                return 0
            if spec is None:
                self._bump("bfs_none")
                self._record_failure(ent, "bfs", "none", self.bfs_depth, bt, strategy)
                return 0
            how = (f"find_negating_sequence(max_depth={self.bfs_depth})" if strategy == "bfs" else
                   f"find_negating_sequence(max_depth={self.bfs_depth}, strategy={strategy!r})")
            if self._store_spec(ent, tuple(spec), "bfs", how):
                self._forget_failure(ent, "bfs")
                if strategy != "bfs":
                    self._bump(f"spec_accepted_bfs_{strategy}")
                return 1
            if not ent.specs:                             # found, but not accepted
                self._record_failure(ent, "bfs", "not_accepted", self.bfs_depth, bt, strategy)
            return 0

        if workers > 1:
            tasks = [(ent.key, self.bfs_depth, bt, strategy) for ent in todo]
            self.log(f"  bfs: {len(tasks)} entries on {workers} workers ({bt} s each"
                     f"{', ' + strategy + ' to depth ' + str(self.bfs_depth) if strategy != 'bfs' else ''})"
                     f"{known}")
            with _fork_pool(workers, 256) as pool_:
                for i, (key, spec, status) in enumerate(
                        pool_.imap_unordered(_bfs_task, tasks, chunksize=1), 1):
                    self._maybe_checkpoint(i)
                    n_ok += absorb(self.entries[key], spec, status)
                    self._progress = f"bfs {i}/{len(tasks)}"
            self._progress = None
            self.timings["bfs"] = time.time() - t
            self.log(f"stage 2: BFS on {len(todo)} entries, {n_ok} specs "
                     f"({self.timings['bfs']:.1f}s){known}")
            return
        for i, ent in enumerate(todo, 1):
            self._maybe_checkpoint(i)
            B = [list(r) for r in ent.key]

            def run(B=B, n=ent.n):
                Q = BPSQuiver(charges=standard_nodes(n), frozen=[False] * n, exchange_matrix=B)
                seq = Q.find_negating_sequence(max_depth=self.bfs_depth, strategy=strategy)
                if seq is None:
                    return None
                return [tuple(g) for g in Q.build_spectrum_generator(seq)]

            spec, status = _with_timeout(bt, run)
            n_ok += absorb(ent, spec, status)
            self._progress = f"bfs {i}/{len(todo)}"
        self._progress = None
        self.timings["bfs"] = time.time() - t
        self.log(f"stage 2: BFS on {len(todo)} entries, {n_ok} specs "
                 f"({self.timings['bfs']:.1f}s){known}")

    def stage_order_search_fallback(self, workers: int | None = None) -> None:
        """Producer 3: order search on the crystalline `S`; accepted with its
        depth (a fixed depth is never a certificate — the audit).

        Each result is absorbed AS IT ARRIVES, behind a `_maybe_checkpoint`
        hook.  This stage used to collect every result into a list and store
        them only afterwards, which meant it carried no stop hook at all — a
        SIGTERM was latched and then ignored for the stage's whole duration,
        and a stop would have thrown away everything the stage had computed."""
        from recursive_spectrum import extract_spec_from_quiver
        t = time.time()
        n_ok = 0
        workers = self.workers if workers is None else workers
        cutoff = self.order_search_cutoff or self.content_depth
        timeout = self.order_search_timeout or self.per_entry_timeout
        # As for BFS: what the order search already failed on at this
        # cutoff and time limit is not searched again on resume.
        todo_all = self._entries_without_spec()
        todo = [ent for ent in todo_all
                if not self._already_failed(ent, "order_search", cutoff, timeout)]
        known = (f" ({len(todo_all) - len(todo)} skipped: the order search already failed on "
                 f"them at this cutoff and time limit)" if len(todo) < len(todo_all) else "")

        def absorb(ent, spec, status) -> int:
            if status != "ok":
                outcome = status.split(":")[0]            # "timeout" or "err"
                self._bump(f"order_search_{outcome}")
                self._record_failure(ent, "order_search", outcome, cutoff, timeout)
                return 0
            if spec is None:
                self._bump("order_search_none")
                self._record_failure(ent, "order_search", "none", cutoff, timeout)
                return 0
            if self._store_spec(
                    ent, tuple(tuple(g) for g in spec), "order_search",
                    f"extract_spec_from_quiver(cutoff={cutoff})",
                    depth=max(cutoff, self.accept_depth)):
                self._forget_failure(ent, "order_search")
                return 1
            if not ent.specs:                             # found, but not accepted
                self._record_failure(ent, "order_search", "not_accepted", cutoff, timeout)
            return 0

        if workers > 1:
            tasks = [(ent.key, cutoff, timeout) for ent in todo]
            self.log(f"  order search: {len(tasks)} entries on {workers} workers "
                     f"(cutoff {cutoff}, {timeout} s each){known}")
            with _fork_pool(workers, 64) as pool_:
                for i, (key, spec, status) in enumerate(
                        pool_.imap_unordered(_order_search_task, tasks, chunksize=1), 1):
                    self._maybe_checkpoint(i)
                    n_ok += absorb(self.entries[key], spec, status)
                    self._progress = f"order search {i}/{len(tasks)}"
        else:
            for i, ent in enumerate(todo, 1):
                self._maybe_checkpoint(i)
                spec, status = _with_timeout(
                    timeout, extract_spec_from_quiver,
                    [list(r) for r in ent.key], standard_nodes(ent.n), cutoff=cutoff)
                n_ok += absorb(ent, spec, status)
                self._progress = f"order search {i}/{len(todo)}"
        self._progress = None
        self.timings["order_search"] = time.time() - t
        self.log(f"stage 3: order search on {len(todo)} entries, {n_ok} specs "
                 f"({self.timings['order_search']:.1f}s){known}")

    # --------------------------------------------------------------- stage 3b

    def stage_verify_axioms(self, depth: int, *, sample: int | None = None,
                            seed: int = 41, families: bool = True,
                            strip_depth: int | None = None, workers: int = 1,
                            select: Callable[[EntryData], bool] | None = None) -> None:
        """The post-build test of the conjecture: the axiom check at `depth`
        on every green spec of the set (or a deterministic `sample` of
        entries) not already checked to that depth.  Mismatches are findings
        (`axiom_mismatches`), never dropped.

        With `families=True` the entries carrying several accepted specs are a
        second test: every spec of the family must give the SAME `S` as the
        primary on the cone to `depth` (`spec_acceptance.spec_product` +
        `spectra_agree`); a disagreement is a finding (`family_mismatches`).

        With `workers > 1` the per-entry checks run in a pool of forked
        worker processes (`_verify_primary_task`, `_verify_family_task`) and
        the results are applied here in arrival order — the same records,
        the same findings, the same checkpoints; only the wall clock differs
        (the exhaustive re-verification of a whole set on the free cores)."""
        from spec_acceptance import spec_product, spectra_agree
        t = time.time()
        if strip_depth is None:
            strip_depth = min(3, depth)

        def wanted(ent):
            # the strip specs (an acyclic quiver's source/sink order, ~85 % of
            # the set, a theorem) are checked at `strip_depth`; the rest at
            # `depth`.  `strip_depth = 0` SKIPS them: shard format 2 codes the
            # acyclic entries rather than storing them, and on a strip
            # spec the crystalline comparison is near-tautological anyway —
            # `crystalline_spectrum` builds the product in the same source/sink
            # order, so agreement there is a regression guard, not evidence
            # certification is then the replay alone, `green`.
            #
            # The test is whether the entry will be CODED (`_split_codable`), not
            # whether its spec came from the strip producer.  Keying on the
            # producer misses almost all of them: propagation routinely stores
            # another source/sink order of the same length `n`, which wins the
            # primary's lexicographic tie-break, so at weight 9 only 28,667 of
            # 522,969 acyclic entries still had a `strip` primary and the rest
            # were verified in full (measured 2026-09-13, mid-build).
            if ent.acyclic and is_source_sink_order(ent.key, ent.primary.spec):
                return strip_depth
            return depth

        todo = [ent for ent in self.entries.values()
                if ent.owned and ent.primary is not None and ent.primary.green
                and (select is None or select(ent))
                and wanted(ent) > 0
                and (ent.primary.axioms is None or ent.primary.axioms["depth"] < wanted(ent))]
        if sample is not None and sample < len(todo):
            import random
            todo = random.Random(seed).sample(todo, sample)
        n_ok = n_bad = 0
        if workers > 1:
            tasks = [(ent.key, ent.primary.spec, wanted(ent), self.per_entry_timeout) for ent in todo]
            self.log(f"  verify: {len(tasks)} entries on {workers} workers")
            with _fork_pool(workers, 256) as pool_:
                for i, (key, rec, status) in enumerate(
                        pool_.imap_unordered(_verify_primary_task, tasks, chunksize=1), 1):
                    self._maybe_checkpoint(i)
                    ent = self.entries[key]
                    p = ent.primary
                    if rec is None:
                        self._bump(f"axiom_check_{status.split(':')[0]}")
                        self.log(f"  axiom check {status} on {ent.name()} ({p.producer})")
                        continue
                    p.axioms = _intern_axioms(rec)
                    self._touch(ent)
                    if rec["ok"]:
                        n_ok += 1
                    else:
                        n_bad += 1
                        # every entry in `todo` is green: a failure is a conjecture counterexample
                        self.axiom_mismatches.append({
                            "name": ent.name(), "exchange": [list(r) for r in ent.key],
                            "spec": [list(g) for g in p.spec], "producer": p.producer,
                            "provenance": p.provenance, "acceptance": rec,
                        })
                        self.log(f"  !! AXIOM MISMATCH on a green spec: {ent.name()} ({p.producer}) "
                                 f"at depth {rec['depth']}")
                    if i % 500 == 0:
                        self.log(f"  verify {i}/{len(todo)} ({time.time() - t:.0f}s)")
        else:
            for i, ent in enumerate(todo, 1):
                self._maybe_checkpoint(i)
                p = ent.primary
                rec = self._axiom_check(ent, p.spec, wanted(ent), p.producer, p.provenance)
                if rec is None:
                    continue
                p.axioms = _intern_axioms(rec)
                self._touch(ent)
                if rec["ok"]:
                    n_ok += 1
                else:
                    n_bad += 1
                if i % 500 == 0:
                    self.log(f"  verify {i}/{len(todo)} ({time.time() - t:.0f}s)")
        n_fam = n_fam_bad = 0
        if families and workers > 1:
            # the family test takes the same entries as the primary check: all
            # (or `select`'s), or the sample (`select` used to reach only the
            # primaries, so a weight-11 pass re-compared every family of the set)
            pool = ([ent for ent in self.entries.values() if select is None or select(ent)]
                    if sample is None else todo)
            tasks = [(ent.key, ent.primary.spec,
                      [o.spec for o in ent.specs if o is not ent.primary], depth, self.per_entry_timeout)
                     for ent in pool if ent.owned and len(ent.specs) >= 2 and ent.primary is not None
                     and not self._family_done(ent, depth)]
            self.log(f"  family test: {len(tasks)} entries on {workers} workers")
            with _fork_pool(workers, 256) as pool_:
                for i, (key, results) in enumerate(
                        pool_.imap_unordered(_verify_family_task, tasks, chunksize=1), 1):
                    # the stop and checkpoint hook: without it a stop waited
                    # for the whole family test — 95,596 entries, ~2.5 h on the iMac
                    self._maybe_checkpoint(i)
                    self._progress = f"family test {i}/{len(tasks)}"
                    ent = self.entries[key]
                    p = ent.primary
                    by_spec = {o.spec: o for o in ent.specs}
                    if all(status == "ok" for _, _, status in results):
                        self._mark_family_done(ent, depth)
                    for o_spec, n_diff, status in results:
                        if status != "ok":
                            self._bump(f"family_check_{status.split(':')[0]}")
                            continue
                        n_fam += 1
                        if n_diff:
                            n_fam_bad += 1
                            other = by_spec.get(o_spec)
                            self.family_mismatches.append({
                                "name": ent.name(), "exchange": [list(r) for r in ent.key],
                                "spec_primary": [list(g) for g in p.spec],
                                "spec_other": [list(g) for g in o_spec],
                                "producers": [p.producer, other.producer if other else None],
                                "depth": depth, "n_differing_charges": n_diff,
                            })
                            self.log(f"  !! FAMILY MISMATCH on {ent.name()}: two accepted specs give "
                                     f"different S at depth {depth}")
        elif families:
            # the family test takes the same entries as the primary check: all
            # (or `select`'s), or the sample (`select` used to reach only the
            # primaries, so a weight-11 pass re-compared every family of the set)
            pool = ([ent for ent in self.entries.values() if select is None or select(ent)]
                    if sample is None else todo)
            for i, ent in enumerate(pool, 1):
                self._maybe_checkpoint(i)
                if len(ent.specs) < 2 or self._family_done(ent, depth):
                    continue
                p = ent.primary
                Sp, status = _with_timeout(self.per_entry_timeout, spec_product, ent.key, p.spec, depth)
                if status != "ok":
                    continue
                all_ok = True
                for other in ent.specs:
                    if other is p:
                        continue
                    So, status = _with_timeout(self.per_entry_timeout, spec_product, ent.key, other.spec, depth)
                    if status != "ok":
                        all_ok = False
                        continue
                    n_fam += 1
                    diff = spectra_agree(Sp, So, ent.n, depth)
                    if diff:
                        n_fam_bad += 1
                        self.family_mismatches.append({
                            "name": ent.name(), "exchange": [list(r) for r in ent.key],
                            "spec_primary": [list(g) for g in p.spec],
                            "spec_other": [list(g) for g in other.spec],
                            "producers": [p.producer, other.producer], "depth": depth,
                            "n_differing_charges": len(diff),
                        })
                        self.log(f"  !! FAMILY MISMATCH on {ent.name()}: two accepted specs give different S at depth {depth}")
                if all_ok:
                    self._mark_family_done(ent, depth)
        self._progress = None
        self.timings["verify_axioms"] = time.time() - t
        self.stats["verified_ok"] = self.stats.get("verified_ok", 0) + n_ok
        self.stats["family_pairs_checked"] = self.stats.get("family_pairs_checked", 0) + n_fam
        self.log(f"stage 3b: axiom check at depth {depth} (strip specs at {strip_depth}) on {len(todo)} green specs: "
                 f"{n_ok} ok, {n_bad} MISMATCH; {n_fam} family pairs compared, {n_fam_bad} differ "
                 f"({self.timings['verify_axioms']:.1f}s)")

    # --------------------------------------------------------------- stage 4

    def stage_content(self, max_weight: int | None = None, *, workers: int | None = None,
                      select: Callable[[EntryData], bool] | None = None) -> None:
        """The crystalline factor content, to `content_depth`, for entries with
        no accepted spec (set and tail).  `max_weight` limits it to entries of
        weight `≤ max_weight`: the weights a climbing build has finished
        searching can be certified and shipped before the stage itself runs; an entry that has its content is skipped when the stage runs.
        `select` limits it further, as in `stage_bfs_fallback`.

        With `workers > 1` the entries run in forked workers (`_content_task`)
        and each result is stored as it arrives.  Every entry's result depends
        only on its key, depth and time limit, so the records and the
        `content_*` counts are those of the serial run: at weight 13 the
        serial stage is ~2.6 M entries at 0.245 s each, about a week."""
        t = time.time()
        n_ok = 0
        workers = self.workers if workers is None else workers

        def wanted(ent: EntryData) -> bool:
            if ent.specs or ent.content is not None or not ent.owned:
                return False
            if max_weight is not None and ent.e > max_weight:
                return False
            if not ent.in_set and ent.n > self.tail_max_rank:
                return False
            return select is None or select(ent)

        def absorb(ent: EntryData, depth: int, record: dict | None, status: str) -> int:
            if status != "ok":
                self._bump(f"content_{status.split(':')[0]}")
                return 0
            ent.content = _intern_content(record)
            ent.content_depth = depth
            self._touch(ent)
            return 1

        if workers > 1:
            todo = [ent for ent in list(self.entries.values()) + list(self.tail.values())
                    if wanted(ent)]
            tasks = [(ent.key, self.content_depth if ent.in_set else self.tail_accept_depth,
                      self.per_entry_timeout, ent.in_set) for ent in todo]
            self.log(f"  content: {len(tasks)} entries on {workers} workers")
            with _fork_pool(workers, 256) as pool_:
                for i, (key, in_set, depth, record, status) in enumerate(
                        pool_.imap_unordered(_content_task, tasks, chunksize=4), 1):
                    self._maybe_checkpoint(i)
                    ent = self.entries[key] if in_set else self.tail[key]
                    n_ok += absorb(ent, depth, record, status)
                    self._progress = f"content {i}/{len(tasks)}"
                    if i % 4096 == 0:
                        # the stored records beyond the collector's reach, so a
                        # worker recycled later does not walk and copy them
                        gc.freeze()
            self._progress = None
        else:
            # The hook fires on every iteration, cheap ones included: it only reads
            # the clock and the stop flag.  Without it this stage was stop-blind and
            # unwritable for its whole duration.
            for i, ent in enumerate(list(self.entries.values()) + list(self.tail.values()), 1):
                self._maybe_checkpoint(i)
                if not wanted(ent):
                    continue
                depth = self.content_depth if ent.in_set else self.tail_accept_depth
                res, status = _with_timeout(
                    self.per_entry_timeout, crystalline_spectrum, ent.key, depth,
                    with_multiplicities=True)
                n_ok += absorb(ent, depth, content_record(res[1]) if status == "ok" else None, status)
        self.timings["content"] = time.time() - t
        self.log(f"stage 4: crystalline content on {n_ok} entries ({self.timings['content']:.1f}s)")

    # --------------------------------------------------------------- stage 5

    def stage_fingerprints(self, K: int = 8, timeout: int | None = None) -> None:
        """`I_{id,id}` mod `𝖖^K`, μ → 1 (the design record layer (a)), on entries with an
        accepted spec."""
        from bps_kalgebra import BPSKAlgebra
        from dictionaries.build_fingerprint import _mu_to_one_qseries
        t = time.time()
        timeout = self.per_entry_timeout if timeout is None else timeout
        n_ok = 0
        todo = [ent for ent in self.entries.values()]
        slow: list[tuple[float, str]] = []
        for i, ent in enumerate(todo, 1):
            self._maybe_checkpoint(i)
            p = ent.primary
            if p is None:
                ent.fingerprint_status = "no_spec"
                continue
            t_ent = time.time()

            def run(ent=ent, p=p):
                A = BPSKAlgebra(pairing=[list(r) for r in ent.key],
                                node_charges=standard_nodes(ent.n),
                                spec=[tuple(g) for g in p.spec], verify="off")
                return _mu_to_one_qseries(A.inner_product(A.identity(), A.identity(), K=K))

            fp, status = _with_timeout(timeout, run)
            ent.fingerprint_status = status
            dt = time.time() - t_ent
            if dt > 2.0:
                slow.append((round(dt, 1), ent.name()))
            if status == "ok":
                ent.fingerprint = fp
                n_ok += 1
            if i % 100 == 0:
                self.log(f"  fingerprints {i}/{len(todo)} ({time.time() - t:.0f}s)")
        self.timings["fingerprints"] = time.time() - t
        self.stats["fingerprint_slow_entries"] = len(slow)
        self.fingerprint_slow = sorted(slow, reverse=True)[:20]
        self.log(f"stage 5: fingerprints on {n_ok} entries (K={K}, {self.timings['fingerprints']:.1f}s; "
                 f"{len(slow)} entries over 2s)")

    # --------------------------------------------------------------- stage 6

    def stage_class_index(self) -> None:
        t = time.time()
        if self.shard is not None:
            self.log("stage 6: class index skipped on a shard (run by merge_shards on the whole set)")
            return
        # NOTE: `mutation_components` is a single call and cannot be hooked from
        # here, so the stage stays unresponsive to a stop for that call's
        # duration; the loops below are hooked.  Checkpointing mid-stage is safe
        # because the stage is not marked done until it completes, so `--resume`
        # re-runs it in full and overwrites any partial class assignment.
        gc.freeze()        # mutation_components forks its own pool
        rep_of = mutation_components(self.records, self.E, workers=self.workers)
        reps = sorted(set(rep_of.values()), key=lambda k: (self.records[k].e, k))
        idx = {r: i for i, r in enumerate(reps)}
        members: dict[Matrix, list[Matrix]] = {r: [] for r in reps}
        for i, (key, r) in enumerate(rep_of.items(), 1):
            self._maybe_checkpoint(i)
            self.entries[key].class_id = idx[r]
            members[r].append(key)
        by_fp: dict[str, set[int]] = {}
        self.classes = []
        for r in reps:
            mem = sorted(members[r], key=lambda k: (self.records[k].e, k))
            fps = sorted({self.entries[k].fingerprint for k in mem
                          if self.entries[k].fingerprint is not None})
            for fp in fps:
                by_fp.setdefault(fp, set()).add(idx[r])
            self.classes.append({
                "class_id": idx[r],
                "representative": [list(row) for row in r],
                "n": self.records[r].n,
                "min_e": self.records[r].e,
                "size": len(mem),
                "has_acyclic_member": any(self.records[k].acyclic for k in mem),
                "has_exit": any(x > self.E for k in mem for x in self.records[k].exits),
                "members": [self.entries[k].name() for k in mem],
                "fingerprints": fps,
                "fingerprint_conflict": len(fps) > 1,
                "spec_verified": any(self.entries[k].specs for k in mem),
            })
        merges = [sorted(v) for v in by_fp.values() if len(v) > 1]
        self.class_merge_candidates = merges
        self.timings["class_index"] = time.time() - t
        self.log(f"stage 6: {len(reps)} components inside e <= {self.E}; "
                 f"{sum(c['fingerprint_conflict'] for c in self.classes)} fingerprint conflicts; "
                 f"{len(merges)} same-fingerprint component groups "
                 f"({self.timings['class_index']:.1f}s)")

    # ----------------------------------------------------------------- write

    def manifest(self) -> dict:
        cells = cell_counts(self.records.values())
        return {
            "plan": "41_enumerated_quiver_dictionary",
            "E": self.E,
            "shard": list(self.shard) if self.shard is not None else None,
            "strongly_connected": self.strongly_connected,
            "completeness": (f"every {'strongly ' if self.strongly_connected else ''}connected "
                             f"BPS quiver with total arrow weight "
                             f"e = sum_(i<j)|b_ij| <= {self.E}, up to node permutation"
                             + (f" — SHARD {self.shard[0]} of {self.shard[1]} (enumeration index "
                                f"≡ {self.shard[0]} mod {self.shard[1]}); merge_shards joins the shards"
                                if self.shard is not None else "")),
            "accept_depth": self.accept_depth,
            "verify_depth": self.verify_depth,
            "verify_strip_depth": (self.verify_strip_depth if self.verify_strip_depth is not None
                                   else min(3, self.verify_depth)),
            "content_depth": self.content_depth,
            "acceptance": "green (exact replay; the S axioms ASSUMED) | axioms@D (non-green, "
                          "the finite-depth axiom check); verification = the axiom check on green "
                          "specs, a mismatch being a conjecture counterexample",
            "tail_max_rank": self.tail_max_rank,
            "tail_max_weight": self.tail_max_weight,
            "tail_accept_depth": self.tail_accept_depth,
            "max_tail_entries": self.max_tail_entries,
            "specs_per_entry": self.specs_per_entry,
            "family_max": self.family_max,
            "bfs_depth": self.bfs_depth,
            "per_entry_timeout": self.per_entry_timeout,
            "bfs_timeout": self.bfs_timeout,
            "bfs_strategy": self.bfs_strategy,
            "local_move_states": self.local_move_states,
            "local_move_extra": self.local_move_extra,
            "cells": {f"n={n},e={e}": {"quivers": q, "acyclic": a}
                      for (n, e), (q, a) in cells.items()},
            "n_entries": len(self.entries),
            "n_tail": len(self.tail),
            "n_with_spec": sum(1 for x in self.entries.values() if x.specs),
            "n_green": sum(1 for x in self.entries.values() if x.primary is not None and x.primary.green),
            "n_axioms_checked": sum(1 for x in self.entries.values()
                                    if x.primary is not None and x.primary.axioms is not None),
            "n_spec_verified": sum(1 for x in self.entries.values() if x.specs),   # kept for readers of older manifests
            "n_crystalline_only": sum(1 for x in self.entries.values()
                                      if not x.specs and x.content is not None),
            "n_uncertified": sum(1 for x in self.entries.values()
                                 if not x.specs and x.content is None),
            "n_classes": len(self.classes),
            "class_merge_candidates": getattr(self, "class_merge_candidates", []),
            "stats": dict(sorted(self.stats.items())),
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
            "fingerprint_slow": getattr(self, "fingerprint_slow", []),
            "rejected_specs": self.rejected,
            "axiom_mismatches": self.axiom_mismatches,
            "family_mismatches": self.family_mismatches,
            "stages_done": list(self.stages_done),
            "skipped_stages": list(self.skipped_stages),
            "reruns_applied": list(self.reruns_applied),
            # every stage of the pipeline done or switched off — what the nightly
            # driver reads to tell a finished build from a reopened one
            "complete": self.is_complete(),
            "stopped_early": self.stopped_early,
            "log": self.log_lines,
        }

    def write(self, out_dir: str | Path, *, incremental: bool = False) -> Path | int:
        """Write the rich build.  With `incremental=True` only the cells (and
        the tail) touched since the last write to `checkpoint_dir` are
        rewritten — a mid-stage checkpoint — and the number of cells written
        is returned; a full write returns the directory.  Either way the
        manifest is rewritten and the dirty set is cleared when `out_dir` is
        the checkpoint directory."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # One cell at a time, streamed entry by entry (T27): a weight-9 cell
        # is 0.5-0.7 GB of JSON and several GB as a materialised list of
        # dicts; holding every dirty cell's list at once was what pushed the
        # builder over the memory limit in the verification pass.  The bytes
        # written are those of `json.dumps(list)` exactly (the default
        # separators), and the order is the old sort on `exchange` — which is
        # the entry key.
        cells: dict[tuple[int, int], list[EntryData]] = {}
        for ent in self.entries.values():
            if incremental and (ent.n, ent.e) not in self._dirty_cells:
                continue
            cells.setdefault((ent.n, ent.e), []).append(ent)
        for (n, e), ents in cells.items():
            ents.sort(key=lambda ent: ent.key)
            _write_json_list(out / f"q_n{n:02d}_e{e:02d}.json", (ent.as_dict() for ent in ents))
        if not incremental or self._tail_dirty:
            tails: dict[int, list[EntryData]] = {}
            for ent in self.tail.values():
                tails.setdefault(ent.n, []).append(ent)
            for n, ents in tails.items():
                ents.sort(key=lambda ent: (ent.e, ent.key))
                _write_json_list(out / f"tail_n{n:02d}.json", (ent.as_dict() for ent in ents))
        if not incremental:
            _write_text_atomic(out / f"classes_E{self.E}.json", json.dumps(self.classes))
        _write_text_atomic(out / "manifest.json", json.dumps(self.manifest(), indent=1))
        if self.checkpoint_dir is not None and out.resolve() == Path(self.checkpoint_dir).resolve():
            self._dirty_cells.clear()
            self._tail_dirty = False
            if not incremental:
                self._checkpoint_complete = True
        return len(cells) if incremental else out

    # ------------------------------------------------------------------- run

    STAGES = ("strip", "propagate", "bfs", "order_search", "verify", "content",
              "fingerprints", "class_index")

    def run(self, seeds: Iterable[tuple[str, QuiverEntry]] | None = None, *,
            fingerprints: bool = False, K: int = 8,
            max_candidates: int | None = None, resume: bool = False) -> None:
        """All stages in order.  With `resume=True` (on a builder returned by
        `load()`) the stages recorded as done are skipped; a stage that was
        interrupted mid-way re-runs over the entries it had not finished,
        since every stage is idempotent on its per-entry results."""
        def todo(stage: str) -> bool:
            return not (resume and stage in self.stages_done)

        try:
            self._run_stages(todo, seeds, fingerprints, K, max_candidates)
        except StopRequested as stop:
            # The stage in flight is NOT recorded as done: every stage is
            # idempotent per entry, so --resume re-derives only what was not
            # stored.  A FULL write, not an incremental one, so the checkpoint
            # is complete however far the stage had got.
            self.log(f"STOPPING: {stop}.  Writing a full checkpoint; "
                     f"stages done: {self.stages_done}"
                     + (f"; {self._progress}" if self._progress else ""))
            t0 = time.time()
            self.stopped_early = str(stop)        # before the write, so the manifest records it
            if self.checkpoint_dir is not None:
                self._checkpoint_complete = False
                self.write(self.checkpoint_dir)
            self.log(f"  checkpoint written in {time.time() - t0:.0f}s; resume with --resume")

    def _run_stages(self, todo: Callable[[str], bool], seeds, fingerprints: bool,
                    K: int, max_candidates: int | None) -> None:
        """The stages in order.  Split out of `run()` so a stop raised in any
        per-entry loop unwinds to one place."""
        if not self.records:
            self.stage_enumerate()
        if todo("strip"):
            self.stage_seed_acyclic()
            self._checkpoint("strip")
        if todo("propagate"):
            self.stage_propagate(seeds if seeds is not None else default_seeds(self.tail_max_rank, self.log),
                                 max_candidates=max_candidates)
            self._checkpoint("propagate")
        if todo("bfs"):
            self.stage_bfs_fallback()
            self._checkpoint("bfs")
        if todo("propagate_after_bfs"):
            self.stage_propagate(label="2b")            # the BFS specs are sources too
            self._checkpoint("propagate_after_bfs")
        if todo("order_search"):
            if self.order_search:
                self.stage_order_search_fallback()
                self._checkpoint("order_search")
            else:
                if "order_search" not in self.skipped_stages:
                    self.skipped_stages.append("order_search")
                self.log("stage 3: order search SKIPPED (--no-order-search); the entries "
                         "still without a spec go on to the crystalline content")
        if todo("propagate_after_order_search"):
            self.stage_propagate(label="3b")
            self._checkpoint("propagate_after_order_search")
        if self.verify_depth and todo("verify"):
            self.stage_verify_axioms(self.verify_depth, strip_depth=self.verify_strip_depth,
                                     workers=self.workers, families=self.family_test)
            self._checkpoint("verify")
        if todo("content"):
            self.stage_content()
            self._checkpoint("content")
        if fingerprints and todo("fingerprints"):
            self.stage_fingerprints(K=K)
            self._checkpoint("fingerprints")
        if todo("class_index"):
            if self.strongly_connected:
                # mutation leaves the set, and the mutation classes belong to
                # the theory dictionary (strongly_connected_redesign.md §4)
                if "class_index" not in self.skipped_stages:
                    self.skipped_stages.append("class_index")
                self.log("stage 6: class index SKIPPED — a strongly connected build")
            else:
                self._refuse_if_deadline_near("class_index", self.class_index_reserve)
                self.stage_class_index()
                self._checkpoint("class_index")

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, build_dir: str | Path, *, log: Callable[[str], None] | None = print,
             checkpoint: bool = True, enumeration_cache: str | Path | None = None,
             workers: int = 1) -> "EnumeratedBuild":
        """Rebuild a builder from a rich build directory (a checkpoint or a
        finished build): parameters and findings from `manifest.json`, the
        entries with their spec records from the per-cell files, the tail,
        the class table.  The enumeration is recomputed (it is fast and is
        what supplies the automorphisms); with `checkpoint=True` the builder
        keeps checkpointing into the same directory."""
        src = Path(build_dir)
        m = json.loads((src / "manifest.json").read_text())
        b = cls(m["E"], accept_depth=m.get("accept_depth", 5),
                verify_depth=m.get("verify_depth", 0),
                verify_strip_depth=m.get("verify_strip_depth"),
                content_depth=m.get("content_depth", 6),
                tail_max_rank=m.get("tail_max_rank", 12),
                tail_max_weight=m.get("tail_max_weight"),
                tail_accept_depth=m.get("tail_accept_depth", 3),
                max_tail_entries=m.get("max_tail_entries", 5000),
                specs_per_entry=m.get("specs_per_entry", 3),
                family_max=m.get("family_max"),
                bfs_depth=m.get("bfs_depth", 12),
                local_move_states=m.get("local_move_states", 64),
                local_move_extra=m.get("local_move_extra", 4),
                log=log, checkpoint_dir=src if checkpoint else None,
                shard=tuple(m["shard"]) if m.get("shard") else None, workers=workers,
                strongly_connected=bool(m.get("strongly_connected", False)))
        b.enumeration_cache = Path(enumeration_cache) if enumeration_cache else None
        b.stage_enumerate()
        b._checkpoint_complete = checkpoint          # the directory holds every cell
        b.stats = dict(m.get("stats", {}))
        b.rejected = list(m.get("rejected_specs", []))
        b.axiom_mismatches = list(m.get("axiom_mismatches", []))
        b.family_mismatches = list(m.get("family_mismatches", []))
        b.stages_done = list(m.get("stages_done", []))
        b.skipped_stages = list(m.get("skipped_stages", []))
        b.reruns_applied = list(m.get("reruns_applied", []))
        b.bfs_timeout = m.get("bfs_timeout")
        # the manifest records it; left at the constructor's default of 20 s, every
        # resume of a build made with another limit had to set it again by hand
        b.per_entry_timeout = m.get("per_entry_timeout", b.per_entry_timeout)
        b.bfs_strategy = m.get("bfs_strategy", "bfs")
        b.timings = {k: float(v) for k, v in m.get("timings", {}).items()}
        b.log_lines = list(m.get("log", []))
        n_loaded = 0
        for path in sorted(src.glob("q_n*_e*.json")):
            for d in json.loads(path.read_text()):
                key = tuple(tuple(int(x) for x in row) for row in d["exchange"])
                ent = b.entries.get(key)
                if ent is None:
                    raise ValueError(f"{path}: entry {d.get('name')} is not in the enumeration at E={b.E}")
                _restore_entry(ent, d)
                n_loaded += 1
        for path in sorted(src.glob("tail_n*.json")):
            for d in json.loads(path.read_text()):
                key = tuple(tuple(int(x) for x in row) for row in d["exchange"])
                cf = canonical_form(key)
                ent = EntryData(key=tuple(intern_vec(row) for row in cf.key), n=d["n"],
                                e=d["e"], aut_order=cf.aut_order,
                                automorphisms=intern_vec(tuple(intern_vec(a) for a in cf.automorphisms)),
                                acyclic=bool(d["acyclic"]),
                                flavour_rank=int(d["flavour_rank"]),
                                exits=intern_vec(tuple(d["exits"])), in_set=False)
                _restore_entry(ent, d)
                b.tail[ent.key] = ent
        cls_path = src / f"classes_E{b.E}.json"
        if cls_path.exists():
            b.classes = json.loads(cls_path.read_text())
        b.log(f"loaded {n_loaded} entries + {len(b.tail)} tail from {src}; stages done: {b.stages_done}")
        return b


def _restore_entry(ent: EntryData, d: dict) -> None:
    """Fill an `EntryData` from its rich `as_dict` form.  Spec records written
    before the green-first acceptance carry (`green_sequence`, `acceptance`)
    in place of (`green`, `axioms`) — the same shape under the old names
    (the first weight-7 rich build) — and are translated, as `ship()` does."""
    ent.specs = [SpecRecord(spec=tuple(tuple(int(x) for x in g) for g in r["spec"]),
                            producer=r["producer"], provenance=r.get("provenance", ""),
                            green=bool(r["green"] if "green" in r else r.get("green_sequence", False)),
                            axioms=r["axioms"] if "axioms" in r else r.get("acceptance"),
                            propagated=bool(r.get("propagated", False)))
                 for r in d.get("specs", [])]
    ent.content = _intern_content(d.get("content"))
    ent.content_depth = d.get("content_depth")
    prov = d.get("provenance") or ""
    ent.provenance = [x for x in prov.split("; ") if x]
    ent.class_id = d.get("class_id")
    ent.fingerprint = d.get("fingerprint")
    ent.fingerprint_status = d.get("fingerprint_status")
    ent.attempts = dict(d["attempts"]) if d.get("attempts") else None
    ent.family_checked = dict(d["family_checked"]) if d.get("family_checked") else None


# ---------------------------------------------------------------------------
# Sharding across machines (T17): joining the shard builds
# ---------------------------------------------------------------------------


def merge_shards(shard_dirs: Sequence[str | Path], out_dir: str | Path, *,
                 log: Callable[[str], None] | None = print,
                 enumeration_cache: str | Path | None = None,
                 workers: int = 1) -> "EnumeratedBuild":
    """Join the rich builds of the shards `0/K … K-1/K` of one weight bound
    into one rich build at `out_dir`: the entries (disjoint by construction,
    together the whole set), the tails (same key → the spec families
    united), the counters and findings summed; then the class index, which
    needs the whole set, runs once; the result is a complete rich build,
    ready for `ship()`."""
    log = log or (lambda s: None)
    parts = [EnumeratedBuild.load(d, log=None, checkpoint=False, enumeration_cache=enumeration_cache)
             for d in shard_dirs]
    if not parts:
        raise ValueError("merge_shards: no shard directories")
    Ks = {p.shard[1] if p.shard else None for p in parts}
    if len(Ks) != 1 or None in Ks:
        raise ValueError(f"merge_shards: the builds are not shards of one split (K = {Ks})")
    K = Ks.pop()
    seen = sorted(p.shard[0] for p in parts)
    if seen != list(range(K)):
        raise ValueError(f"merge_shards: shards present {seen}, need all of 0..{K - 1}")
    if len({p.E for p in parts}) != 1:
        raise ValueError("merge_shards: the shards have different E")
    first = parts[0]
    b = EnumeratedBuild(first.E, accept_depth=first.accept_depth, verify_depth=first.verify_depth,
                        verify_strip_depth=first.verify_strip_depth, content_depth=first.content_depth,
                        tail_max_rank=first.tail_max_rank, tail_max_weight=first.tail_max_weight,
                        tail_accept_depth=first.tail_accept_depth, max_tail_entries=first.max_tail_entries,
                        specs_per_entry=first.specs_per_entry, family_max=first.family_max,
                        bfs_depth=first.bfs_depth, local_move_states=first.local_move_states,
                        local_move_extra=first.local_move_extra, log=log, checkpoint_dir=None,
                        enumeration_cache=enumeration_cache, workers=workers)
    b.records = first.records
    by_shard = {p.shard[0]: p for p in parts}
    # each entry from its OWNER (every shard holds every entry — the strip +
    # propagation closure is the same in all of them — but only the owner ran
    # the per-entry stages on it)
    for idx, key in enumerate(b.records):
        owner = by_shard[idx % K]
        ent = owner.entries.get(key)
        if ent is None:
            raise ValueError(f"merge_shards: entry {key} is missing from its owner shard {idx % K}")
        ent.in_set = True
        ent.owned = True
        b.entries[key] = ent
    # counters of the shared (closure) stages are identical in every shard —
    # take them once; the per-entry stages' counters add up
    SHARED_PREFIXES = ("spec_accepted_strip", "spec_accepted_propagation", "spec_accepted_seed",
                       "propagation_", "ingest_", "spec_duplicate", "spec_family_full", "tail_full",
                       "propagated_non_green")
    for k, v in first.stats.items():
        if k.startswith(SHARED_PREFIXES):
            b.stats[k] = v
    for p in parts:
        for key, ent in p.tail.items():
            mine = b.tail.get(key)
            if mine is None:
                b.tail[key] = ent
                continue
            have = {s.spec for s in mine.specs}
            for rec in ent.specs:
                if rec.spec not in have:
                    mine.specs.append(rec)
                    have.add(rec.spec)
            for prov in ent.provenance:
                if prov not in mine.provenance and len(mine.provenance) < 8:
                    mine.provenance.append(prov)
        for k, v in p.stats.items():
            if not k.startswith(SHARED_PREFIXES):
                b.stats[k] = b.stats.get(k, 0) + v
        for k, v in p.timings.items():
            b.timings[k] = b.timings.get(k, 0.0) + v
        b.rejected += p.rejected
        b.axiom_mismatches += p.axiom_mismatches
        b.family_mismatches += p.family_mismatches
    if len(b.entries) != len(b.records):
        missing = len(b.records) - len(b.entries)
        raise ValueError(f"merge_shards: {missing} entries of the set are in no shard")
    # the stages every shard completed
    done = set(parts[0].stages_done)
    for p in parts[1:]:
        done &= set(p.stages_done)
    b.stages_done = [st for st in EnumeratedBuild.STAGES if st in done]
    # the specs the per-entry producers found propagate ACROSS shards only
    # here: a shard's rounds 2b / 3b reached entries it does not own, and
    # those were discarded above — so re-run them on the whole set, in the
    # unsharded order (the BFS specs first, then the order-search ones), and
    # verify whatever they add
    for producer, label in (("bfs", "propagate_after_bfs"), ("order_search", "propagate_after_order_search")):
        if label not in done:
            continue
        for ent in list(b.entries.values()) + list(b.tail.values()):
            for rec in ent.specs:
                if rec.producer == producer:
                    rec.propagated = False
        b.stage_propagate(label=f"merge:{label}")
    if b.verify_depth and "verify" in done:
        b.stage_verify_axioms(b.verify_depth, strip_depth=b.verify_strip_depth, workers=workers)
    if "content" in done:
        b.stage_content()
    b.stage_class_index()
    b._checkpoint("class_index")
    b.log(f"merged {len(parts)} shards: {len(b.entries)} entries, {len(b.tail)} tail, "
          f"stages done {b.stages_done}")
    out = b.write(out_dir)
    b.log(f"wrote the merged build to {out}")
    return b


# ---------------------------------------------------------------------------
# Shipping — the permanent tier
# ---------------------------------------------------------------------------


def shipped_entry(entry: dict) -> dict:
    """The permanent form of an entry: the quiver, one spec (or `null`), and
    the certification string — `green` (exact replay, axioms assumed),
    `green;axioms@D` (also verified at depth `D`), `green;axiom_mismatch@D`
    (a finding), `axioms@D` (non-green, accepted by the finite-depth check;
    user: *"it does not even need to be green … Any spec which matches the S
    axiomatics, with whatever factor order"*), `crystalline_only@D`.
    Everything else is ancillary."""
    out = {k: entry[k] for k in SHIPPED_KEYS}
    specs = entry.get("specs") or []
    if out["spec"] is not None and specs:
        primary = min(specs, key=lambda s: (not (s.get("green") if "green" in s else s.get("green_sequence")),
                                            len(s["spec"]), s["spec"]))
        out["spec"] = [list(g) for g in primary["spec"]]     # the spec and its certification come from ONE record
        if "certification" in primary:
            out["certification"] = primary["certification"]
        else:
            # a rich build written by the earlier record shape
            # (`green_sequence`, `acceptance`): translate
            ax = primary.get("acceptance")
            green = primary.get("green_sequence") is True
            if green:
                out["certification"] = ("green" if not ax else
                                        (f"green;axioms@{ax['depth']}" if ax.get("ok")
                                         else f"green;axiom_mismatch@{ax['depth']}"))
            elif ax:
                out["certification"] = f"axioms@{ax['depth']}"
    return out


def compact_entry(full: dict) -> dict:
    """Shard format 2: a shipped entry `{exchange, spec, certification}` in its
    compact form — the quiver as its **arrow string** and a green spec as its
    **node-index string** (`quiver_enumeration.arrows_string` /
    `sequence_string`; the loader inverts both).

    | key | content |
    |---|---|
    | `q` | the arrow string: one character pair per arrow, `2e` characters |
    | `s` | the node-index string of the green replay, `len(spec)` characters |
    | `v` | the explicit charge vectors — ONLY when the spec is not a negating sequence in its stored order, so no index string exists (none has appeared) |
    | `c` | the certification string, unchanged |

    Neither `s` nor `v` means no accepted spec (`crystalline_only@D`).  The
    conversion is **checked per entry**: the index string is kept only if
    replaying it rebuilds the stored charges exactly, so shipping cannot
    silently lose a spec.  Measured on the weight-8 set: 5.7 gzipped bytes per
    cyclic entry against 11.9 for format 1."""
    B = full["exchange"]
    out = {"q": arrows_string(B), "c": full["certification"]}
    spec = full["spec"]
    if spec is not None:
        spec_t = [tuple(int(x) for x in g) for g in spec]
        seq = sequence_from_spec(B, spec_t)
        if seq is not None and spec_from_sequence(B, seq) == spec_t:
            out["s"] = sequence_string(seq)
        else:
            out["v"] = [list(g) for g in spec_t]
    return out


def is_source_sink_order(B: Sequence[Sequence[int]], spec: Sequence[Sequence[int]]) -> bool:
    """Is `spec` a source/sink order of the acyclic quiver `B` — the node
    charges, each node mutated before any node it points to?

    This is the licence to CODE an acyclic entry rather than store it: two
    source/sink orders of one quiver differ by transpositions of nodes with no
    arrow between them, whose `E_𝖖` factors commute, so they give the SAME `S`
    and the build's certification transfers to the order a lookup computes.
    An acyclic entry whose stored spec is anything else is stored, not coded."""
    n = len(B)
    if len(spec) != n:
        return False
    order = []
    for g in spec:
        g = [int(x) for x in g]
        if sum(g) != 1 or max(g) != 1 or min(g) != 0:
            return False
        order.append(g.index(1))
    if len(set(order)) != n:
        return False
    done: set[int] = set()
    for k in order:
        if any(B[j][k] > 0 for j in range(n) if j != k and j not in done):
            return False
        done.add(k)
    return True


CELL_COMPLETENESS = ("every connected BPS quiver with n nodes and e arrows counted with "
                     "multiplicity (e = sum_(i<j)|b_ij|), up to node permutation, with one "
                     "accepted spec each or the crystalline certification")

ACYCLIC_CODED = ("the acyclic quivers of this cell are CODED, not stored: the source/sink "
                 "order is a spec (a theorem), so `strip_spec(B)` answers a lookup without "
                 "the file.  `count` counts every connected quiver of the cell, `acyclic` "
                 "the coded ones, `stored` the entries below (the cyclic ones).")


def _split_codable(full: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """Split a cell's shipped entries into the CODED half (acyclic, and whose
    stored spec is a source/sink order, so the computed one gives the same
    `S` — `is_source_sink_order`) and the half that is STORED (every cyclic
    entry, plus any acyclic entry whose spec is something else, which is kept
    rather than silently replaced)."""
    coded, stored = [], []
    for x in full:
        B, spec = x["exchange"], x["spec"]
        if is_acyclic(B) and spec is not None and is_source_sink_order(B, spec):
            coded.append(x)
        else:
            stored.append(x)
    return coded, stored


def compact_shipped_tier(src_dir: str | Path, dst_dir: str | Path, *,
                         gzip_level: int = 9, acyclic_control: int = 200,
                         log: Callable[[str], None] = print) -> Path:
    """Re-encode an ALREADY SHIPPED tier from shard format 1 to format 2,
    without a rich build and without re-deriving anything.

    This is how the committed tier changes format: every certification string
    is carried across verbatim — re-shipping from a fresh rich build would
    silently downgrade the weight-7 cells, whose `green;axioms@7` records came
    from the T13 re-verification merged in elsewhere.  The transformation is
    checked entry by entry: a stored spec must round-trip through its
    node-index string, and an entry is coded away only where
    `is_source_sink_order` licenses it.  `manifest.json` keeps its counts and
    findings, gaining the format-2 census fields."""
    import gzip as _gzip
    import random as _random
    import re as _re
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text())
    if manifest.get("format", 1) >= SHARD_FORMAT:
        raise ValueError(f"{src}: already shard format {manifest.get('format')}")
    rng = _random.Random(41)
    cells: dict[str, dict] = {}
    n_entries = n_acyclic = n_stored = n_spec = 0
    acyclic_certs: dict[str, int] = {}
    ctrl_checked = ctrl_ok = 0
    written: set[Path] = set()
    for path in sorted(list(src.glob("q_n*_e*.json.gz")) + list(src.glob("q_n*_e*.json"))):
        m = _re.match(r"q_n(\d+)_e(\d+)\.json(\.gz)?$", path.name)
        if not m:
            continue
        n, e = int(m.group(1)), int(m.group(2))
        if path.name.endswith(".gz"):
            with _gzip.open(path, "rt", encoding="utf-8") as f:
                shard_in = json.load(f)
        else:
            shard_in = json.loads(path.read_text())
        full = sorted(shard_in["entries"], key=lambda d: d["exchange"])
        coded, stored = _split_codable(full)
        sample = coded if len(coded) <= acyclic_control else rng.sample(coded, acyclic_control)
        for x in sample:
            sp = strip_spec(x["exchange"])
            ctrl_checked += 1
            if sp is not None and is_green_sequence(x["exchange"], sp):
                ctrl_ok += 1
        cell_acyclic_certs: dict[str, int] = {}
        for x in coded:
            acyclic_certs[x["certification"]] = acyclic_certs.get(x["certification"], 0) + 1
            cell_acyclic_certs[x["certification"]] = cell_acyclic_certs.get(x["certification"], 0) + 1
        entries = [compact_entry(x) for x in stored]
        for x, c in zip(stored, entries):            # the per-entry control
            if x["spec"] is not None and "s" not in c and "v" not in c:
                raise AssertionError(f"{path}: spec lost while compacting {x['exchange']}")
        n_sv = sum(1 for x in entries if "s" in x or "v" in x)
        shard = {
            "plan": shard_in.get("plan", "41_enumerated_quiver_dictionary"),
            "format": SHARD_FORMAT,
            "n": n, "e": e,
            "completeness": shard_in.get("completeness", CELL_COMPLETENESS),
            "acyclic_coded": ACYCLIC_CODED,
            "entry_keys": "q = arrow string (2e chars, one character pair per arrow, "
                          "node index in base 36); s = node-index string of the green replay "
                          "(absent with v = no accepted spec); v = explicit charges when the "
                          "spec is not a negating sequence in its order; c = certification",
            "count": len(full),
            "acyclic": len(coded),
            "stored": len(entries),
            "acyclic_certifications": dict(sorted(cell_acyclic_certs.items())),
            "spec_verified": n_sv + len(coded),
            "accept_depth": shard_in.get("accept_depth"),
            "verify_depth": shard_in.get("verify_depth"),
            "verify_strip_depth": shard_in.get("verify_strip_depth"),
            "entries": entries,
        }
        out = dst / f"q_n{n:02d}_e{e:02d}.json.gz"
        with open(out, "wb") as raw:
            with _gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                compresslevel=gzip_level, mtime=0) as f:
                f.write(json.dumps(shard, separators=(",", ":")).encode())
        written.add(out)
        cells[f"n={n},e={e}"] = {"quivers": len(full), "acyclic": len(coded),
                                 "stored": len(entries), "spec_verified": n_sv + len(coded)}
        n_entries += len(full)
        n_acyclic += len(coded)
        n_stored += len(entries)
        n_spec += n_sv + len(coded)
    if src.resolve() == dst.resolve():
        for stale in list(dst.glob("q_n*_e*.json")):
            if stale not in written:
                stale.unlink()
    manifest.update({
        "format": SHARD_FORMAT,
        "tier": "permanent — every connected BPS quiver with total arrow weight "
                "e <= E, up to node permutation, with one accepted spec each "
                "(or the crystalline certification where no producer reached one); "
                "the acyclic quivers are CODED, not stored (see acyclic_coded)",
        "acyclic_coded": ACYCLIC_CODED,
        "acyclic_certifications": dict(sorted(acyclic_certs.items())),
        "acyclic_strip_control": {"checked": ctrl_checked, "green": ctrl_ok,
                                  "note": "a deterministic sample per cell: the source/sink "
                                          "order replayed as a negating sequence"},
        "shard_shape": "a dict {plan, format, n, e, completeness, acyclic_coded, entry_keys, "
                       "count, acyclic, stored, spec_verified, accept_depth, verify_depth, "
                       "verify_strip_depth, entries: [...]}, gzipped",
        "entry_keys": list(COMPACT_KEYS),
        "cells": cells,
        "n_entries": n_entries,
        "n_acyclic_coded": n_acyclic,
        "n_stored": n_stored,
        "n_with_spec": n_spec,
        "n_spec_verified": n_spec,
        "recoded": "shard format 1 -> 2 by compact_shipped_tier(); certifications carried "
                   "across verbatim, nothing re-derived",
    })
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"re-encoded {n_entries} entries -> {n_stored} stored + {n_acyclic} coded "
        f"({ctrl_ok}/{ctrl_checked} strip controls green) in {dst}")
    return dst


def _certification_depth(c: str) -> int:
    """The cone depth a certification string claims; 0 for a bare `green` or a
    recorded mismatch."""
    if not c or "@" not in c or "mismatch" in c:
        return 0
    try:
        return int(c.split("@")[1])
    except ValueError:
        return 0


def _deeper_cell(prev_dir, n: int, e: int) -> dict | None:
    """The cell `(n, e)` of an already-shipped tier, or None."""
    if prev_dir is None:
        return None
    import gzip as _gzip
    for cand in (prev_dir / f"q_n{n:02d}_e{e:02d}.json.gz", prev_dir / f"q_n{n:02d}_e{e:02d}.json"):
        if cand.exists():
            if cand.suffix == ".gz":
                with _gzip.open(cand, "rt", encoding="utf-8") as f:
                    return json.load(f)
            return json.loads(cand.read_text())
    return None


def ship(build_dir: str | Path, ship_dir: str | Path = DEFAULT_SHIP, *,
         gzip_level: int = 9, acyclic_control: int = 200,
         keep_deeper_from: str | Path | None = None) -> Path:
    """Convert a rich build (`--out`) into the shipped set: one
    `q_n{NN}_e{EE}.json.gz` per cell, plus `manifest.json`.  The shipped
    directory is made to hold EXACTLY this build's cells: any other cell file
    in `ship_dir` — a plain-`.json` twin, or a shard of a cell beyond this
    build's `E` left by an earlier, larger build — is removed, so the manifest
    and the shards never disagree (`iter_enumerated_entries` yields every cell
    file present).

    **Shard format 2** (2026-09-13), two changes, both measured on the
    weight-8 set (1.15 MB → 0.13 MB, 8.8×):

    * the entries are **compact** — the quiver as an arrow string, a green
      spec as a node-index string (`compact_entry`), 5.7 gzipped bytes per
      entry against 11.9;
    * the **acyclic quivers are coded, not stored**: their source/sink order
      is a spec by a theorem, so `dictionary_loader.lookup_enumerated`
      answers an acyclic quiver from `strip_spec(B)` without opening a file.
      They are 82 % of the set at weight 8 and their census stays in every
      cell header (`count` / `acyclic` / `stored`), so the completeness
      statement is unchanged and checkable.  `acyclic_control` entries per
      cell are re-replayed here as the positive control on that code path.

    `keep_deeper_from` names an already-shipped tier whose certifications are
    kept where they are STRICTLY DEEPER than this build's: a larger-`E` build
    re-derives the small cells from scratch and would otherwise replace a
    `green;axioms@7` record — earned by a verification pass run elsewhere —
    with its own shallower one.  A stored entry's record is carried across only
    when the quiver AND the spec are the same; the coded half's per-cell
    histogram is carried when the cell's membership is unchanged (the
    enumeration is deterministic, so a cell with the same `count` and `acyclic`
    holds the same quivers) and every depth the earlier histogram records is
    at least every depth this build's records — the coded entries are not
    stored individually, so that is the condition under which carrying the
    earlier histogram whole downgrades no entry.
    """
    import gzip as _gzip
    import random as _random
    import re as _re
    src = Path(build_dir)
    dst = Path(ship_dir)
    if src.resolve() == dst.resolve():
        raise ValueError("ship(): the shipped directory must differ from the build directory "
                         "(shipping in place would destroy the rich build)")
    if json.loads((src / "manifest.json").read_text()).get("strongly_connected"):
        raise ValueError(f"ship(): {src} is a strongly connected build, whose tier is shard "
                         f"format 3 (strongly_connected_redesign.md §6); ship() "
                         f"writes format 2 only")
    dst.mkdir(parents=True, exist_ok=True)
    build_manifest = json.loads((src / "manifest.json").read_text())
    cells: dict[str, dict] = {}
    n_entries = n_spec = n_cry = n_green = n_uncert = 0
    n_acyclic = n_stored = 0
    acyclic_certs: dict[str, int] = {}
    ctrl_checked = ctrl_ok = 0
    rng = _random.Random(41)
    prev_dir = Path(keep_deeper_from) if keep_deeper_from is not None else None
    n_kept_deeper = n_kept_spec = 0
    written: set[Path] = set()
    for path in sorted(src.glob("q_n*_e*.json")):
        m = _re.match(r"q_n(\d+)_e(\d+)\.json$", path.name)
        if not m:
            continue
        n, e = int(m.group(1)), int(m.group(2))
        prev = _deeper_cell(prev_dir, n, e)
        full = sorted((shipped_entry(x) for x in json.loads(path.read_text())),
                      key=lambda d: d["exchange"])
        acyclic, cyclic = _split_codable(full)
        # Positive control on the CODED half: the source/sink order must replay
        # as a negating sequence, since that is what licenses dropping these
        # entries.  A deterministic sample per cell (the check is a theorem, so
        # this guards the code path, not the mathematics).
        sample = acyclic if len(acyclic) <= acyclic_control else rng.sample(acyclic, acyclic_control)
        for x in sample:
            sp = strip_spec(x["exchange"])
            ctrl_checked += 1
            if sp is not None and is_green_sequence(x["exchange"], sp):
                ctrl_ok += 1
        cell_acyclic_certs: dict[str, int] = {}
        for x in acyclic:
            acyclic_certs[x["certification"]] = acyclic_certs.get(x["certification"], 0) + 1
            cell_acyclic_certs[x["certification"]] = cell_acyclic_certs.get(x["certification"], 0) + 1
        entries = [compact_entry(x) for x in cyclic]
        if prev is not None:
            # Keep what an earlier tier certified MORE DEEPLY.  A larger-`E`
            # build re-derives the small cells from scratch and, seeded
            # differently, usually arrives at another (often shorter) spec for
            # the same quiver — so carrying only the certification string would
            # attach an old record to a new spec, and carrying nothing would
            # trade a depth-7 statement for a depth-3 one.  The whole ENTRY is
            # carried instead: the earlier spec with the record that belongs to
            # it.  Both specs are green and accepted; the deeper-verified one is
            # the better thing to ship, and the shorter one stays in the rich
            # build.  An earlier entry with NO spec never displaces a new one
            # that has one.
            by_q = {x["q"]: x for x in prev.get("entries", [])}
            for c in entries:
                o = by_q.get(c["q"])
                if o is None or (o.get("s") is None and o.get("v") is None):
                    continue
                if _certification_depth(o.get("c", "")) <= _certification_depth(c["c"]):
                    continue
                same_spec = o.get("s") == c.get("s") and o.get("v") == c.get("v")
                c["c"] = o["c"]
                if not same_spec:
                    for k in ("s", "v"):
                        c.pop(k, None)
                        if o.get(k) is not None:
                            c[k] = o[k]
                    n_kept_spec += 1
                n_kept_deeper += 1
            # the coded half: unchanged membership means the same quivers, so
            # the earlier cell's recorded depths carry across
            # The coded entries are not stored individually, so the two
            # histograms cannot be merged entry by entry; the earlier one is
            # carried whole exactly when that is entry-wise safe — every depth
            # it records is at least every depth this build recorded — and it
            # changes something.  (A cell minimum against a cell minimum, the
            # earlier rule, left the weight-8 cell n=7 at depth 3 because 3 of
            # its 23,134 coded entries were, although 23,131 were at 7.)
            old_certs = prev.get("acyclic_certifications") or {}
            if (old_certs and prev.get("count") == len(full) and prev.get("acyclic") == len(acyclic)
                    and old_certs != cell_acyclic_certs
                    and min(_certification_depth(k) for k in old_certs)
                    >= max((_certification_depth(k) for k in cell_acyclic_certs), default=0)):
                for k, v in cell_acyclic_certs.items():
                    acyclic_certs[k] -= v
                    if acyclic_certs[k] <= 0:
                        acyclic_certs.pop(k, None)
                cell_acyclic_certs = dict(old_certs)
                for k, v in old_certs.items():
                    acyclic_certs[k] = acyclic_certs.get(k, 0) + v
        n_sv = sum(1 for x in entries if "s" in x or "v" in x)
        shard = {
            "plan": "41_enumerated_quiver_dictionary",
            "format": SHARD_FORMAT,
            "n": n, "e": e,
            "completeness": CELL_COMPLETENESS,
            "acyclic_coded": ACYCLIC_CODED,
            "entry_keys": "q = arrow string (2e chars, one character pair per arrow, "
                          "node index in base 36); s = node-index string of the green replay "
                          "(absent with v = no accepted spec); v = explicit charges when the "
                          "spec is not a negating sequence in its order; c = certification",
            "count": len(full),
            "acyclic": len(acyclic),
            "stored": len(entries),
            "acyclic_certifications": dict(sorted(cell_acyclic_certs.items())),
            "spec_verified": n_sv,
            "accept_depth": build_manifest.get("accept_depth"),
            "verify_depth": build_manifest.get("verify_depth"),
            "verify_strip_depth": build_manifest.get("verify_strip_depth"),
            "entries": entries,
        }
        payload = json.dumps(shard, separators=(",", ":")).encode()
        out = dst / f"q_n{n:02d}_e{e:02d}.json.gz"
        # reproducible bytes: no timestamp and no name in the gzip header,
        # so an unchanged cell re-ships byte-identical and git sees no churn
        with open(out, "wb") as raw:
            with _gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                compresslevel=gzip_level, mtime=0) as f:
                f.write(payload)
        written.add(out)
        cells[f"n={n},e={e}"] = {"quivers": len(full), "acyclic": len(acyclic),
                                 "stored": len(entries), "spec_verified": n_sv + len(acyclic)}
        n_entries += len(full)
        n_acyclic += len(acyclic)
        n_stored += len(entries)
        n_spec += n_sv + len(acyclic)
        n_cry += sum(1 for x in cyclic if x["spec"] is None
                     and x["certification"].startswith("crystalline_only"))
        n_green += sum(1 for x in full if x["certification"].startswith("green"))
        n_uncert += sum(1 for x in full if x["certification"] == "none")
    for stale in list(dst.glob("q_n*_e*.json")) + list(dst.glob("q_n*_e*.json.gz")):
        if stale not in written:
            stale.unlink()
    manifest = {
        "plan": build_manifest.get("plan", "41_enumerated_quiver_dictionary"),
        "format": SHARD_FORMAT,
        "tier": "permanent — every connected BPS quiver with total arrow weight "
                "e <= E, up to node permutation, with one accepted spec each "
                "(or the crystalline certification where no producer reached one); "
                "the acyclic quivers are CODED, not stored (see acyclic_coded)",
        "acyclic_coded": ACYCLIC_CODED,
        "acyclic_certifications": dict(sorted(acyclic_certs.items())),
        "acyclic_strip_control": {"checked": ctrl_checked, "green": ctrl_ok,
                                  "note": "a deterministic sample per cell: the source/sink "
                                          "order replayed as a negating sequence"},
        "certifications_kept_from": (str(prev_dir) if prev_dir is not None else None),
        "n_certifications_kept_deeper": n_kept_deeper,
        "n_entries_kept_with_their_spec": n_kept_spec,
        "kept_note": ("an entry the earlier tier certified more deeply is carried across WITH its "
                      "spec, since a differently-seeded build usually reaches another spec for the "
                      "same quiver and an old record must not be attached to a new spec; the "
                      "shorter spec, where there is one, stays in the rich build"),
        "acyclic_certifications_earlier_tier": (
            (json.loads((prev_dir / "manifest.json").read_text()).get("acyclic_certifications")
             if prev_dir is not None and (prev_dir / "manifest.json").exists() else None)),
        "acyclic_certifications_note": ("the CODED half carries its certification per cell, and an "
                                        "earlier tier shipped before that field existed has only a "
                                        "global histogram — recorded above verbatim rather than "
                                        "spread over cells, so no per-cell claim is invented"),
        "E": build_manifest["E"],
        "completeness": build_manifest["completeness"],
        "accept_depth": build_manifest.get("accept_depth"),
        "verify_depth": build_manifest.get("verify_depth"),
        "verify_strip_depth": build_manifest.get("verify_strip_depth"),
        "depths_note": "accept_depth: the axiom check for NON-green candidates; verify_depth / "
                       "verify_strip_depth: the post-build check on green specs (all others / the acyclic "
                       "strip specs).  The depth that actually held for an entry is in its certification.",
        "content_depth": build_manifest.get("content_depth"),
        "shard_shape": "a dict {plan, format, n, e, completeness, acyclic_coded, entry_keys, "
                       "count, acyclic, stored, spec_verified, accept_depth, verify_depth, "
                       "verify_strip_depth, entries: [...]}, gzipped",
        "entry_keys": list(COMPACT_KEYS),
        "certification": "green | green;axioms@D | green;axiom_mismatch@D | axioms@D | crystalline_only@D",
        "certification_semantics": "green = the spec replays as a negating sequence (exact, finite); "
            "the S axioms are ASSUMED for it.  axioms@D = the BPS quiver constraint (every non-node "
            "coefficient of S is O(q^2)) and the crystalline agreement checked on cone charges of "
            "degree <= D only; no all-degree certificate exists (RESEARCH_GOALS 3.12).  A green spec "
            "failing the check is a conjecture counterexample (axiom_mismatches).",
        "axiom_mismatches": build_manifest.get("axiom_mismatches", []),
        "family_mismatches": build_manifest.get("family_mismatches", []),
        # a stage the build did not run — e.g. the order search at weight 10
        "skipped_stages": build_manifest.get("skipped_stages", []),
        "n_entries": n_entries,          # every connected quiver of the set (the census)
        "n_acyclic_coded": n_acyclic,    # answered by the strip rule, not stored
        "n_stored": n_stored,            # the entries actually in the shards (the cyclic ones)
        "n_with_spec": n_spec,
        "n_spec_verified": n_spec,   # kept for readers of older manifests: entries WITH a spec
        "n_green": n_green,
        "n_crystalline_only": n_cry,
        "n_uncertified": n_uncert,     # entries with neither a spec nor crystalline content; 0 on a complete build
        "cells": cells,
        "rejected_specs": build_manifest.get("rejected_specs", []),
        "builder": "dictionaries/build_enumerated.py",
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return dst


SC_SHARD_FORMAT = 3
SC_CELL_COMPLETENESS = ("every STRONGLY CONNECTED BPS quiver with n nodes and e arrows counted "
                        "with multiplicity (e = sum_(i<j)|b_ij|), up to node permutation, with one "
                        "accepted spec each or the crystalline certification")
SC_COVERAGE = ("every BPS quiver — connected or not, of any rank and any total weight — whose "
               "strongly connected components have total arrow weight <= E: its spec is its "
               "components' specs concatenated in a source-first order, a single node "
               "contributing its own charge (strongly_connected_redesign.md §1.1; "
               "dictionary_loader.lookup_enumerated)")
SC_COMPOSED_RECORD = ("the record of a composed answer, over its components of two or more nodes "
                      "(a single node satisfies the constraint exactly, at every depth): green iff "
                      "every one is; axioms@D at the least depth among them; crystalline_only@D "
                      "(no spec) if one is, at the least such depth; a component's "
                      "axiom_mismatch@D is carried.  An acyclic quiver — every component a "
                      "single node — is answered `green`, its source/sink order, and its S "
                      "satisfies the constraint at every depth by §1.2, a theorem")


def ship_strongly_connected(build_dir: str | Path, ship_dir: str | Path = DEFAULT_SHIP, *,
                            max_weight: int | None = None, gzip_level: int = 9,
                            keep_deeper_from: str | Path | None = None) -> Path:
    """**Shard format 3**: the tier of a STRONGLY
    CONNECTED build.  One `q_n{NN}_e{EE}.json.gz` per cell of strongly
    connected quivers with `e <= max_weight` (default: the build's `E`), in
    format 2's compact entry shape (`compact_entry`), plus `manifest.json`.
    Every other quiver is answered by composition in the loader; the one-node
    quiver needs no entry.  The directory is made to hold exactly these cells:
    any other cell file in `ship_dir` — the format-2 tier's among them — is
    removed.

    A climbing build is shipped only through a weight it has SETTLED: every
    entry of every shipped cell has a spec or the crystalline certification.
    The check runs before anything is written, and names the unsettled cells.

    `keep_deeper_from` names an already-shipped tier (format 2 or 3) whose
    entries are kept where they are STRICTLY more deeply certified — the entry
    whole, spec and record together, exactly as `ship()` does — or, both green,
    AS deeply certified with a strictly SHORTER spec: the shortest known, which
    a build at its family cap (`family_max`) can refuse as a seed.  The
    earlier tier's manifest also supplies, for the weights it covers, the
    census of every connected quiver, kept as a record
    (`census_of_every_connected_quiver`)."""
    import gzip as _gzip
    import io as _io
    import re as _re
    src, dst = Path(build_dir), Path(ship_dir)
    if src.resolve() == dst.resolve():
        raise ValueError("ship_strongly_connected(): the shipped directory must differ from the "
                         "build directory (shipping in place would destroy the rich build)")
    bm = json.loads((src / "manifest.json").read_text())
    if not bm.get("strongly_connected"):
        raise ValueError(f"ship_strongly_connected(): {src} is not a strongly connected build; "
                         f"ship() writes its format 2")
    E = bm["E"] if max_weight is None else int(max_weight)
    if E > bm["E"]:
        raise ValueError(f"ship_strongly_connected(): max_weight {E} is beyond the build's E = {bm['E']}")
    prev_dir = Path(keep_deeper_from) if keep_deeper_from is not None else None
    if prev_dir is not None and not (prev_dir / "manifest.json").exists():
        # a missing tier would ship silently without its deeper records and census
        raise ValueError(f"ship_strongly_connected(): keep_deeper_from {prev_dir} is not a shipped tier "
                         f"(no manifest.json)")
    cells_in: list[tuple[int, int, list[dict]]] = []
    unsettled: dict[str, int] = {}
    searched: dict[str, int] = {}        # the spec-less entries' BFS attempts, as searched
    depths: set = set()                  # the depths those attempts searched to
    limits: set = set()                  # the time limits they ran under
    for path in sorted(src.glob("q_n*_e*.json")):
        m = _re.match(r"q_n(\d+)_e(\d+)\.json$", path.name)
        if not m:
            continue
        n, e = int(m.group(1)), int(m.group(2))
        if e > E or n < 2:
            continue
        rich = [x for x in json.loads(path.read_text()) if x.get("in_set", True)]
        for x in rich:
            if x.get("spec") is None:
                a = (x.get("attempts") or {}).get("bfs")
                k = (f"{a.get('outcome')} at depth {a.get('depth')}, {a.get('timeout')} s"
                     + (f", {a['strategy']}" if a.get("strategy", "bfs") != "bfs" else "")
                     if a else "no BFS attempt recorded")
                searched[k] = searched.get(k, 0) + 1
                if a and a.get("depth") is not None:
                    depths.add(int(a["depth"]))
                if a and a.get("timeout"):
                    limits.add(int(a["timeout"]))
        full = sorted((shipped_entry(x) for x in rich), key=lambda d: d["exchange"])
        bad = sum(1 for x in full if x["spec"] is None
                  and not str(x["certification"]).startswith("crystalline_only"))
        if bad:
            unsettled[f"n={n},e={e}"] = bad
        cells_in.append((n, e, full))
    if unsettled:
        raise ValueError(f"ship_strongly_connected(): weight <= {E} is not settled — entries with "
                         f"neither a spec nor the crystalline certification, per cell: {unsettled}")
    n_unsearched = searched.get("no BFS attempt recorded", 0)
    dst.mkdir(parents=True, exist_ok=True)
    cells: dict[str, dict] = {}
    n_entries = n_spec = n_cry = n_green = 0
    n_kept_deeper = n_kept_spec = n_kept_shorter = 0
    written: set[Path] = set()

    def spec_len(x):
        return len(x["s"]) if x.get("s") is not None else len(x["v"]) if x.get("v") is not None else None

    for n, e, full in cells_in:
        entries = [compact_entry(x) for x in full]
        prev = _deeper_cell(prev_dir, n, e)
        if prev is not None:
            # as ship(): the earlier entry WHOLE, spec with its record, where the
            # earlier tier certified it strictly more deeply — or as deeply, both
            # green, with a strictly shorter spec
            by_q = {x["q"]: x for x in prev.get("entries", [])}
            for c in entries:
                o = by_q.get(c["q"])
                if o is None or (o.get("s") is None and o.get("v") is None):
                    continue
                d_old, d_new = _certification_depth(o.get("c", "")), _certification_depth(c["c"])
                shorter = (d_old == d_new and spec_len(c) is not None and spec_len(o) < spec_len(c)
                           and str(o.get("c", "")).startswith("green") and str(c["c"]).startswith("green"))
                if d_old <= d_new and not shorter:
                    continue
                same_spec = o.get("s") == c.get("s") and o.get("v") == c.get("v")
                c["c"] = o["c"]
                if not same_spec:
                    for k in ("s", "v"):
                        c.pop(k, None)
                        if o.get(k) is not None:
                            c[k] = o[k]
                    n_kept_spec += 1
                if shorter:
                    n_kept_shorter += 1
                else:
                    n_kept_deeper += 1
        n_sv = sum(1 for x in entries if "s" in x or "v" in x)
        n_c = sum(1 for x in entries if str(x["c"]).startswith("crystalline_only"))
        shard = {
            "plan": "41_enumerated_quiver_dictionary",
            "format": SC_SHARD_FORMAT,
            "strongly_connected": True,
            "n": n, "e": e,
            "completeness": SC_CELL_COMPLETENESS,
            "entry_keys": "q = arrow string (2e chars, one character pair per arrow, "
                          "node index in base 36); s = node-index string of the green replay "
                          "(absent with v = no accepted spec); v = explicit charges when the "
                          "spec is not a negating sequence in its order; c = certification",
            "count": len(entries),
            "stored": len(entries),
            "spec_verified": n_sv,
            "crystalline_only": n_c,
            "accept_depth": bm.get("accept_depth"),
            "verify_depth": bm.get("verify_depth"),
            "content_depth": bm.get("content_depth"),
            "entries": entries,
        }
        out = dst / f"q_n{n:02d}_e{e:02d}.json.gz"
        raw = _io.BytesIO()
        # reproducible bytes: no timestamp and no name in the gzip header
        with _gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                            compresslevel=gzip_level, mtime=0) as f:
            f.write(json.dumps(shard, separators=(",", ":")).encode())
        out.write_bytes(raw.getvalue())
        written.add(out)
        cells[f"n={n},e={e}"] = {"stored": len(entries), "spec_verified": n_sv, "crystalline_only": n_c}
        n_entries += len(entries)
        n_spec += n_sv
        n_cry += n_c
        n_green += sum(1 for x in entries if str(x["c"]).startswith("green"))
    for stale in list(dst.glob("q_n*_e*.json")) + list(dst.glob("q_n*_e*.json.gz")):
        if stale not in written:
            stale.unlink()
    census = None
    if prev_dir is not None and (prev_dir / "manifest.json").exists():
        pm = json.loads((prev_dir / "manifest.json").read_text())
        pcells = cE = cfrom = None
        if pm.get("census_of_every_connected_quiver"):          # a format-3 tier: carry its record
            pc = pm["census_of_every_connected_quiver"]
            pcells, cE, cfrom = pc.get("cells"), pc.get("E"), pc.get("from")
        elif pm.get("format") == SHARD_FORMAT:                     # the format-2 tier's own census
            pcells, cE, cfrom = pm.get("cells"), pm.get("E"), str(prev_dir)
        if pcells:
            census = {"E": min(E, cE if cE is not None else E),
                      "from": cfrom,
                      "note": "count = every connected quiver of the cell, acyclic = those with "
                              "no directed cycle; a record of the retired format-2 tier, the "
                              "cross-check of the composition (strongly_connected_redesign.md §6.1)",
                      "cells": {k: {kk: v.get(kk) for kk in ("quivers", "acyclic", "stored")}
                                for k, v in pcells.items()
                                if int(k.split("e=")[1]) <= E}}
    manifest = {
        "plan": bm.get("plan", "41_enumerated_quiver_dictionary"),
        "format": SC_SHARD_FORMAT,
        "strongly_connected": True,
        "tier": "permanent — every strongly connected BPS quiver with total arrow weight "
                "e <= E, up to node permutation, with one accepted spec each (the shortest "
                "known) or the crystalline certification; every other quiver is answered by "
                "composition (coverage)",
        "E": E,
        "build_E": bm["E"],
        "completeness": (f"every strongly connected BPS quiver with total arrow weight "
                         f"e = sum_(i<j)|b_ij| <= {E}, up to node permutation"),
        "coverage": SC_COVERAGE.replace("<= E", f"<= {E}"),
        "composed_record": SC_COMPOSED_RECORD,
        "accept_depth": bm.get("accept_depth"),
        "verify_depth": bm.get("verify_depth"),
        "content_depth": bm.get("content_depth"),
        "shard_shape": "a dict {plan, format, strongly_connected, n, e, completeness, entry_keys, "
                       "count, stored, spec_verified, crystalline_only, accept_depth, verify_depth, "
                       "content_depth, entries: [...]}, gzipped",
        "entry_keys": list(COMPACT_KEYS),
        "certification": "green | green;axioms@D | green;axiom_mismatch@D | axioms@D | crystalline_only@D",
        "certification_semantics": "green = the spec replays as a negating sequence (exact, finite); "
            "the S axioms are ASSUMED for it.  axioms@D = the BPS quiver constraint (every non-node "
            "coefficient of S is O(q^2)) and the crystalline agreement checked on cone charges of "
            "degree <= D only; no all-degree certificate exists (RESEARCH_GOALS 3.12).  A green spec "
            "failing the check is a conjecture counterexample (axiom_mismatches).",
        "search": {
            "producers": "seeds (an earlier tier, the weak dictionaries and harvests: the green "
                         "replay decides), propagation (necklace moves and node drops, tail cap "
                         "max_tail_entries), BFS (find_negating_sequence; "
                         + (f"{n_unsearched} spec-less entries have NO BFS attempt, so the floor "
                            f"bfs_depth is 0; the others were searched at least to depth "
                            f"{min(depths) if depths else '?'}, "
                            if n_unsearched else
                            "every spec-less entry searched at least to depth bfs_depth for "
                            "bfs_time_limit_s, ")
                         + "the deeper passes in spec_less_entries_by_bfs_attempt); "
                         # "ran" only when the pipeline marked it done: a build driven by
                         # partial passes past `propagate` never reaches it
                         "the order search " + ("SKIPPED" if "order_search" in bm.get("skipped_stages", [])
                                                else "ran" if "order_search" in bm.get("stages_done", [])
                                                else "not run"),
            # the FLOOR of the spec-less entries' own attempts — what every one of
            # them was searched to at least, which `seed_from_tier` records for
            # them — not the build's last settings, which a deeper pass over
            # a subset leaves in its manifest; the build's, with no attempts
            # 0 when any spec-less entry was never searched: a floor over the
            # searched ones alone would claim a search they did not get, and
            # seed_from_tier would record them as failures (2026-09-28)
            "bfs_depth": 0 if n_unsearched else (min(depths) if depths else bm.get("bfs_depth")),
            "bfs_time_limit_s": 0 if n_unsearched else (
                min(limits) if limits else bm.get("bfs_timeout") or bm.get("per_entry_timeout")),
            "n_spec_less_unsearched": n_unsearched,
            "max_tail_entries": bm.get("max_tail_entries"),
            "spec_less_entries_by_bfs_attempt": dict(sorted(searched.items())),
            "note": ("crystalline_only@D means no accepted spec was FOUND by these producers, and "
                     "the crystalline S was computed to depth D.  It is NOT a proof that no spec "
                     "exists, at ANY rank: a depth-bounded search rules out only sequences up to its "
                     "depth, and the shortest negating sequence can be far longer than 2n - 2, which "
                     "holds for the oriented n-cycle, not in general.  Measured 2026-09-26 on SC12: of "
                     "156 specs at rank <= 7 found by propagation from weight 13, on quivers where a "
                     "depth-12 BFS had exhausted, 154 are longer than 12 (up to 21 at rank 7).  "
                     "spec_less_entries_by_bfs_attempt says how deeply each spec-less entry was "
                     "searched, and by which strategy."),
        },
        "certifications_kept_from": (str(prev_dir) if prev_dir is not None else None),
        "n_certifications_kept_deeper": n_kept_deeper,
        "n_entries_kept_shorter": n_kept_shorter,
        "n_entries_kept_with_their_spec": n_kept_spec,
        "census_of_every_connected_quiver": census,
        "axiom_mismatches": bm.get("axiom_mismatches", []),
        "skipped_stages": bm.get("skipped_stages", []),
        "n_entries": n_entries,
        "n_with_spec": n_spec,
        "n_green": n_green,
        "n_crystalline_only": n_cry,
        "n_uncertified": 0,
        "cells": cells,
        "builder": "dictionaries/build_enumerated.py (ship_strongly_connected)",
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return dst


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def _is_acyclic(B):
    from quiver_enumeration import is_acyclic
    return is_acyclic(B)


def _mutate(B, k):
    from quiver_enumeration import mutate
    return mutate(B, k)


def uquiver_seeds(max_rank: int, log: Callable[[str], None] = print
                  ) -> Iterable[tuple[str, QuiverEntry]]:
    """The UQuiver constructor seeds of `dictionaries/build.py` (linear /
    circular / D-shape / E-type gauge theories with a constructor-supplied
    spec), those with rank `≤ max_rank`."""
    from dictionaries.build import _entry_from_algebra, _seed_theories
    for name, A in _seed_theories():
        qe = _entry_from_algebra(A, name=name, provenance=f"seed {name!r}")
        if qe is None:
            log(f"  seed {name}: could not standardise")
            continue
        if qe.n_nodes > max_rank:
            log(f"  seed {name}: rank {qe.n_nodes} > {max_rank}, skipped")
            continue
        yield name, qe


def canonical_tree_seeds(dict_dir: str | Path | None = None, max_rank: int | None = None,
                         max_weight: int | None = None
                         ) -> Iterable[tuple[str, QuiverEntry]]:
    """Entries of an entry-list dictionary tree (`n_*.json`: the previous
    canonical tree, the weak dictionary's shards, a harvest of an older build)
    as seeds with their specs.

    `max_weight` drops the seeds too heavy to be worth canonicalising: a
    candidate beyond `E` can only land in the provenance tail, which is capped
    (`max_tail_entries`), so harvesting a dictionary that reaches weight 48
    would spend its time on canonical forms that are then discarded.  The
    origin of a seed carries NO trust either way — it is a candidate through
    the ordinary acceptance check."""
    from dictionary_loader import default_dictionary_dir
    d = Path(dict_dir) if dict_dir is not None else default_dictionary_dir()
    for path in sorted(d.glob("n_*.json")):
        for entry in json.loads(path.read_text()):
            B = tuple(tuple(int(x) for x in row) for row in entry["exchange"])
            if max_rank is not None and len(B) > max_rank:
                continue
            if max_weight is not None and total_weight(B) > max_weight:
                continue
            spec = tuple(tuple(int(x) for x in g) for g in entry.get("spec") or ())
            if not spec:
                continue
            yield entry.get("name", path.stem), QuiverEntry(
                name=entry.get("name", path.stem), exchange=B, spec=spec,
                provenance=f"canonical tree {path.name}")


def default_seeds(max_rank: int, log: Callable[[str], None] = print
                  ) -> list[tuple[str, QuiverEntry]]:
    seeds = list(uquiver_seeds(max_rank, log))
    seeds += list(canonical_tree_seeds(max_rank=max_rank))
    return seeds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--E", type=int, default=5, help="max total arrow weight (default 5)")
    p.add_argument("--accept-depth", type=int, default=5)
    p.add_argument("--verify-axioms", type=int, default=0, metavar="DEPTH",
                   help="run the axiom check at DEPTH on every green spec after the build "
                        "(0 = skip; the build accepts green specs on the conjecture)")
    p.add_argument("--verify-strip-depth", type=int, default=None, metavar="DEPTH",
                   help="depth for the acyclic strip specs in that pass (default min(3, DEPTH))")
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help="run the enumeration walk, the per-entry stages (BFS fallback, order "
                        "search, verification) and the class index in N forked worker "
                        "processes (default 1)")
    p.add_argument("--verify-workers", type=int, default=1, metavar="N",
                   help="older spelling of --workers (kept; --workers wins when both are given)")
    p.add_argument("--reverify", action="store_true",
                   help="with --resume: run the verification pass again at the requested depths "
                        "(entries already checked to that depth are skipped)")
    p.add_argument("--no-order-search", action="store_true",
                   help="skip the order search (producer 3) and record the skip in the manifests; the entries still without a spec go on to the crystalline content.  At weight 9 it found 0 specs in 653 entries")
    p.add_argument("--no-family-test", action="store_true",
                   help="leave the family comparison out of the verification pass (every "
                        "alternative spec against its primary at the verify depth — a "
                        "consistency test, not a condition of shipping; at weight 9 it is "
                        "≈ 2 M spec products with no checkpoint)")
    p.add_argument("--content-depth", type=int, default=6)
    p.add_argument("--tail-max-rank", type=int, default=12)
    p.add_argument("--tail-max-weight", type=int, default=None)
    p.add_argument("--max-tail-entries", type=int, default=None, metavar="N",
                   help="entries beyond E kept as propagation intermediates (default 5000; the "
                        "weight-7 and weight-8 builds hit it)")
    p.add_argument("--specs-per-entry", type=int, default=3)
    p.add_argument("--family-max", type=int, default=None, metavar="K",
                   help="hard cap on the specs stored per entry (default 2n+1; the alternatives "
                        "propagation stores triple the rich build on disk — 315 MB at weight 8)")
    p.add_argument("--bfs-depth", type=int, default=None,
                   help="the BFS depth (default 12 for a fresh build; on --resume the build's own "
                        "unless given — a resumed build used to ignore it)")
    p.add_argument("--bfs-strategy", choices=("bfs", "auto"), default=None,
                   help="producer 2's search: the bidirectional BFS (default for a fresh build) or "
                        "find_spec_auto's seeded recursion ('auto'), which reaches the long "
                        "sequences at high rank; on --resume the build's own unless given")
    p.add_argument("--timeout", type=int, default=20, help="per-entry seconds")
    p.add_argument("--bfs-timeout", type=int, default=None, metavar="SECONDS",
                   help="BFS's own per-entry time limit (default: --timeout); recorded in the "
                        "manifest.  A recorded BFS timeout under a smaller limit is retried")
    p.add_argument("--rerun", action="append", default=None, metavar="STAGE",
                   help="with --resume and --rerun-id: reopen a finished STAGE (repeatable) so this "
                        "run does it again, once per --rerun-id.  Reopening 'propagate' makes every "
                        "stored spec a source again (use with a larger --max-tail-entries)")
    p.add_argument("--rerun-id", default=None, metavar="ID",
                   help="names a --rerun request; a request whose id the manifest already records "
                        "is not applied again, so the same arguments can be passed every night")
    p.add_argument("--order-search-timeout", type=int, default=None, metavar="SECONDS",
                   help="budget per entry for the order search on the crystalline S (default: --timeout)")
    p.add_argument("--order-search-cutoff", type=int, default=None, metavar="DEPTH",
                   help="cone depth for the order search (default: --content-depth)")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="propagation sources to process at most (per call of the stage)")
    p.add_argument("--local-move-states", type=int, default=64, metavar="N",
                   help="local-move search budget per propagation source (states; default 64 — "
                        "256 and 1024 reached the same entries at weight 6)")
    p.add_argument("--no-seeds", action="store_true", help="skip propagation seeds")
    p.add_argument("--strongly-connected", action="store_true",
                   help="store the STRONGLY CONNECTED quivers only, found by the walk "
                        "that climbs past weight 10; every other quiver is answered by "
                        "composing its components.  Recorded in the manifest; no class index")
    p.add_argument("--seeds-tier", metavar="DIR", default=None,
                   help="start from an existing tier (shipped, or a rich build): its specs are "
                        "propagation seeds and the entries it has no spec for skip the BFS; with "
                        "--strongly-connected only its strongly connected entries are read")
    p.add_argument("--seeds-dir", action="append", default=None, metavar="DIR",
                   help="an extra directory of (quiver, spec) seed files to propagate from, in "
                        "the entry-list shape `n_*.json` that `canonical_tree_seeds` reads "
                        "(repeatable).  This is how a dictionary built elsewhere is HARVESTED: "
                        "every seed is a candidate through the ordinary acceptance check, its "
                        "origin carrying no trust")
    p.add_argument("--resume", action="store_true",
                   help="load the rich build in --out and continue from its last checkpoint")
    p.add_argument("--checkpoint-interval", type=float, default=60.0, metavar="MINUTES",
                   help="also checkpoint whenever this many MINUTES have passed since the last "
                        "write, independently of entry throughput (0 = off; default 60). This is "
                        "the durability guarantee: --checkpoint-every counts entries, so at high "
                        "weight it can leave a stage running for many hours unwritten.")
    p.add_argument("--checkpoint-every", type=int, default=2000,
                   help="also checkpoint every N entries inside the long per-entry stages (0 = stages only)")
    p.add_argument("--shard", default=None, metavar="i/K",
                   help="build only the entries with enumeration index i (mod K) — one of K machines; "
                        "join the K builds afterwards with --merge-shards")
    p.add_argument("--merge-shards", nargs="+", default=None, metavar="BUILD_DIR",
                   help="join the shard builds into one rich build at --out (runs the class index), "
                        "then --ship if given; no build is run")
    p.add_argument("--enumeration-cache", default=None, metavar="FILE",
                   help="save the stage-0 records to FILE on the first run and reload them "
                        "on every later run or shard (14 min at weight 9 becomes seconds)")
    p.add_argument("--fingerprints", action="store_true")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help="the rich build directory (semi-permanent, untracked)")
    p.add_argument("--ship", nargs="?", const=str(DEFAULT_SHIP), default=None,
                   help="also write the permanent gzipped set (default dictionaries/enumerated/)")
    p.add_argument("--ship-from", default=None, metavar="BUILD_DIR",
                   help="skip the build; ship an existing rich build directory")
    p.add_argument("--keep-deeper-from", default=None, metavar="SHIPPED_DIR",
                   help="when shipping, keep a certification from this already-shipped tier "
                        "where it is strictly deeper than this build's (same quiver, same "
                        "spec) — so a larger-E build does not replace records earned by a "
                        "verification pass run elsewhere")
    p.add_argument("--recompact", default=None, metavar="SHIPPED_DIR",
                   help="skip the build; re-encode an already-shipped tier from shard format 1 "
                        "to format 2 (in place unless --ship names a target).  Certifications "
                        "are carried across verbatim and nothing is re-derived, so a tier whose "
                        "records were verified deeper elsewhere keeps them")
    p.add_argument("--stop-at", default=None, metavar="YYYY-MM-DDTHH:MM|HH:MM",
                   help="stop cleanly at this local time and write a full checkpoint — for an "
                        "unattended overnight window.  A date and time is that moment exactly "
                        "(what the nightly driver passes); HH:MM alone is the next such time "
                        "after the process starts.  SIGTERM and SIGINT do the same at any time")
    p.add_argument("--retry-timeouts", action="store_true",
                   help="try again the entries on which BFS or the order search already timed "
                        "out, errored or found an unaccepted spec at this depth and time limit "
                        "(recorded per entry; skipped on resume otherwise).  A recorded 'none' "
                        "is retried only by a deeper search")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    log = (lambda s: None) if args.quiet else print
    # The deadline is fixed NOW, before the resume load, which took about an
    # hour at E = 10: resolved after it, an HH:MM that passed in the meantime
    # rolled over to the next morning.
    if args.rerun and not (args.resume and args.rerun_id):
        p.error("--rerun needs --resume and --rerun-id")
    stop_at = None
    if args.stop_at:
        try:
            stop_at = resolve_stop_at(args.stop_at)
        except ValueError as exc:
            p.error(f"--stop-at {args.stop_at!r}: {exc}")
    if args.recompact is not None:
        compact_shipped_tier(args.recompact, args.ship or args.recompact, log=log)
        return 0
    if args.ship_from is not None:
        dst = ship(args.ship_from, args.ship or str(DEFAULT_SHIP),
                   keep_deeper_from=args.keep_deeper_from)
        m = json.loads((dst / "manifest.json").read_text())
        log(f"shipped {m['n_entries']} entries ({m['n_with_spec']} with a spec) to {dst}")
        return 0
    if args.merge_shards is not None:
        merged = merge_shards(args.merge_shards, args.out, log=log,
                              enumeration_cache=args.enumeration_cache,
                              workers=args.workers or 1)
        if args.ship is not None:
            dst = ship(args.out, args.ship, keep_deeper_from=args.keep_deeper_from)
            log(f"shipped the permanent set to {dst}")
        return 0
    shard = None
    if args.shard is not None:
        i, K = (int(x) for x in args.shard.split("/"))
        shard = (i, K)
    if args.resume:
        b = EnumeratedBuild.load(args.out, log=log, enumeration_cache=args.enumeration_cache,
                                 workers=args.workers or args.verify_workers)
        b.per_entry_timeout = args.timeout
        if args.max_tail_entries is not None:          # a resumed build takes a new cap too
            b.max_tail_entries = args.max_tail_entries
        b.checkpoint_every = args.checkpoint_every
        b.checkpoint_interval = max(0.0, args.checkpoint_interval) * 60.0
        if args.verify_axioms:
            b.verify_depth = args.verify_axioms
            b.verify_strip_depth = args.verify_strip_depth
        b.verify_workers = args.verify_workers
        b.workers = args.workers if args.workers is not None else args.verify_workers
        b.order_search_timeout = args.order_search_timeout
        b.order_search_cutoff = args.order_search_cutoff
        if args.bfs_depth is not None:                  # a resume takes a new depth too
            b.bfs_depth = args.bfs_depth
        if args.reverify and "verify" in b.stages_done:
            b.stages_done.remove("verify")
    else:
        b = EnumeratedBuild(args.E, accept_depth=args.accept_depth, verify_depth=args.verify_axioms,
                            verify_strip_depth=args.verify_strip_depth,
                            content_depth=args.content_depth,
                            tail_max_rank=args.tail_max_rank,
                            tail_max_weight=args.tail_max_weight,
                            max_tail_entries=(args.max_tail_entries if args.max_tail_entries is not None
                                              else 5000),
                            verify_workers=args.verify_workers, workers=args.workers,
                            specs_per_entry=args.specs_per_entry, family_max=args.family_max,
                            bfs_depth=args.bfs_depth if args.bfs_depth is not None else 12,
                            per_entry_timeout=args.timeout,
                            local_move_states=args.local_move_states, log=log,
                            checkpoint_dir=args.out, checkpoint_every=args.checkpoint_every,
                            checkpoint_interval=max(0.0, args.checkpoint_interval) * 60.0,
                            enumeration_cache=args.enumeration_cache, shard=shard,
                            order_search_timeout=args.order_search_timeout,
                            order_search_cutoff=args.order_search_cutoff,
                            strongly_connected=args.strongly_connected)
    if args.strongly_connected and not b.strongly_connected:
        p.error(f"--strongly-connected: the build in {args.out} is not a strongly connected build")
    t = time.time()
    seeds = None
    if args.no_seeds:
        seeds = []
    elif args.seeds_dir:
        seeds = default_seeds(b.tail_max_rank, log)
        for sd in args.seeds_dir:
            n0 = len(seeds)
            seeds += list(canonical_tree_seeds(sd, max_rank=b.tail_max_rank,
                                               max_weight=b.E + 1))
            log(f"  seeds: +{len(seeds) - n0} from {sd}")
    if args.seeds_tier and "propagate" not in b.stages_done:
        if not b.records:
            b.stage_enumerate()
        base = seeds if seeds is not None else default_seeds(b.tail_max_rank, log)
        seeds = base + b.seed_from_tier(args.seeds_tier)
    if stop_at is not None:
        log(f"will stop at {time.strftime('%a %Y-%m-%d %H:%M', time.localtime(stop_at))} "
            f"({(stop_at - time.time()) / 3600:.1f} h from now)")
    b.family_test = not args.no_family_test
    b.order_search = not args.no_order_search
    if args.bfs_timeout is not None:
        b.bfs_timeout = args.bfs_timeout
    if args.bfs_strategy is not None:
        b.bfs_strategy = args.bfs_strategy
    if args.rerun and b.reopen(args.rerun, args.rerun_id):
        # durable before any work: the reopened stages and the reset source
        # marks reach disk now, so an interruption resumes the reopened work
        b._checkpoint(None)
    b.retry_timeouts = args.retry_timeouts
    b.install_stop_handlers(stop_at)
    b.run(seeds=seeds, fingerprints=args.fingerprints, K=args.K,
          max_candidates=args.max_candidates, resume=args.resume)
    if b.stopped_early:
        log(f"stopped early: {b.stopped_early}")
    out = b.write(args.out)
    m = b.manifest()
    log(f"wrote {out}: {m['n_entries']} entries ({m['n_with_spec']} with a spec, {m['n_green']} green, "
        f"{m['n_axioms_checked']} axiom-checked, {m['n_crystalline_only']} crystalline-only, "
        f"{m['n_uncertified']} uncertified), {m['n_tail']} tail, {m['n_classes']} classes, "
        f"{len(m['rejected_specs'])} rejected candidates, {len(m['axiom_mismatches'])} AXIOM MISMATCHES; "
        f"{time.time() - t:.1f}s total")
    if args.ship is not None:
        dst = ship(out, args.ship, keep_deeper_from=args.keep_deeper_from)
        log(f"shipped the permanent set to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
