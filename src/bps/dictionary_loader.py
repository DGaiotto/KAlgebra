"""Loader for a BPS-quiver dictionary tree.

The canonical tree lives at `restructuring/dictionaries/` and is
returned by `default_dictionary_dir()`.  A pre-sharpening raw tree
is preserved at the archived tree for cross-checking
the canonical sharpening pipeline.

Two file shapes (plus the ENUMERATED dictionary of the design record — see
`default_enumerated_dictionary_dir` / `iter_enumerated_entries`: its shipped
shards `q_n{NN}_e{EE}.json.gz` are self-describing dicts whose entries carry
ONLY `exchange`, `spec` (possibly `null`) and `certification` — no `name`,
`negating_sequence` or `provenance`; the rich build's entries carry the
`n_NNN.json` shape with extra keys):

  * `n_NNN.json` (3-digit): a flat list of quiver entries on `n`
    active nodes.  Each entry has
    `{name, exchange, spec, negating_sequence, provenance}`.

  * `graph_nNNN.json` (3-digit): a mutation graph wrapper, with
    `{entries, edges}`.  `entries` is the same shape as
    `n_NNN.json`; `edges` is a list of
    `{source, target, via, target_via}` records describing a
    mutation between two `entries` (matched by `exchange` matrix
    or weak-iso key).

Active-only convention: each entry is a BPS quiver on `n` *unfrozen*
nodes, represented in the standard basis (`node_charges = e_1, …, e_n`).
Frozen-node descendants project to fewer active nodes by removing the
frozen row/column from `exchange` (cf. `bps_quiver_dictionary.freeze` in
the repo root).

Many entries have *degenerate* exchange matrices.  Concretely, all
3×3 antisymmetric integer matrices have determinant 0 (odd dim), so
every `n=3` entry has non-trivial `ker(B)`; for `n=5` similarly all
canonical-tree entries are flavoured.  Our flavoured BPSKAlgebra picks
up `Γ_f = ker(B)` automatically and produces an abelian flavour
structure.

This loader has no repo dependencies — it parses the JSON directly and
constructs `BPSKAlgebra` instances, so it works equally well after the
restructuring/ folder is moved to a new repo.

Usage:

    from dictionary_loader import (
        load_dictionary, entry_to_bpskalgebra,
        iter_bpskalgebras_from_dictionary,
        load_mutation_graph, iter_bpskalgebras_from_graph,
        default_dictionary_dir,
    )

    root = default_dictionary_dir()
    for name, A in iter_bpskalgebras_from_dictionary(root / "n_005.json"):
        # A is a BPSKAlgebra; flavoured if det(B) = 0.
        ...

    graph = load_mutation_graph(root / "graph_n005.json")
    for name, A in iter_bpskalgebras_from_graph(graph):
        ...
    # Edges are exposed raw for the caller to consume:
    for edge in graph["edges"]:
        src_excg = edge["source"]   # 2D int matrix
        ...
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterator

# The dictionaries live at the top of the release tree (`dictionaries/`), two
# levels above this module's `src/bps/`; every default path below hangs off it.
_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bps_kalgebra import BPSKAlgebra


Vec = tuple[int, ...]


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def load_dictionary(path: str | Path) -> list[dict]:
    """Load and return the list of entries in a dictionary JSON file.
    A `.gz` path is read through gzip transparently (the shipped enumerated
    shards are `q_n{NN}_e{EE}.json.gz`)."""
    path = Path(path)
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path) as f:
            data = json.load(f)
    if _is_shard(data):
        # a shipped enumerated shard: self-describing, `entries` inside
        return expand_shard_entries(data)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of entries (a graph_nNNN.json "
                         f"mutation graph loads through load_mutation_graph)")
    return data


def _is_shard(data) -> bool:
    """A shipped enumerated shard: a dict with the cell header `n`, `e` and an
    `entries` list — NOT any dict with an `entries` key (a `graph_nNNN.json`
    mutation graph has `entries` and `edges` and is not an entry list)."""
    return (isinstance(data, dict) and isinstance(data.get("entries"), list)
            and isinstance(data.get("n"), int) and isinstance(data.get("e"), int))


def expand_entry(e: dict, n: int) -> dict:
    """One shard entry in the standard `{exchange, spec, certification}` shape.

    **Shard format 2** stores the quiver as an arrow string `q` and a green
    spec as the node-index string `s` of its replay (explicit charges under
    `v` only where no index string exists — a spec that is not a negating
    sequence in its stored order).  Rebuilding the charges is the replay
    `spec_from_sequence`, a few dozen mutations.  A format-1 entry (explicit
    `exchange` / `spec`) is returned unchanged."""
    if "q" not in e:
        return e
    from quiver_enumeration import matrix_from_arrows, sequence_from_string
    B = matrix_from_arrows(n, e["q"])
    if e.get("s") is not None:
        from spec_acceptance import spec_from_sequence
        spec = [list(g) for g in spec_from_sequence(B, sequence_from_string(e["s"]))]
    elif e.get("v") is not None:
        spec = [list(g) for g in e["v"]]
    else:
        spec = None
    return {"exchange": [list(r) for r in B], "spec": spec, "certification": e.get("c", "")}


def expand_shard_entries(shard: dict) -> list[dict]:
    """The entries of a shard in the standard shape (format 1 or 2)."""
    n = shard["n"]
    return [expand_entry(e, n) for e in shard["entries"]]


def acyclic_entry(B, certification: str = "green") -> dict:
    """The entry of an ACYCLIC quiver, which the shipped tier CODES rather
    than stores (shard format 2): its source/sink order is a spec by a
    theorem, so this is computed, never read.  `B` in any node order; the
    spec comes back in the same one."""
    from spec_acceptance import strip_spec
    spec = strip_spec(B)
    if spec is None:
        raise ValueError("acyclic_entry: the quiver has a directed cycle")
    return {"exchange": [[int(x) for x in row] for row in B],
            "spec": [list(g) for g in spec], "certification": certification}


def load_shard(path: str | Path, *, expand: bool = True) -> dict:
    """A shipped enumerated shard WITH its header (`n`, `e`, `completeness`,
    `count`, `acyclic`, `stored`, `spec_verified`, depths, `entries`); a plain
    list file is wrapped as `{"entries": [...]}`.

    With `expand=True` (the default) the entries come back in the standard
    `{exchange, spec, certification}` shape whatever the shard format;
    `expand=False` returns the compact format-2 entries as stored."""
    path = Path(path)
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path) as f:
            data = json.load(f)
    if isinstance(data, list):
        return {"entries": data}
    if _is_shard(data):
        if expand:
            data = dict(data)
            data["entries"] = expand_shard_entries(data)
        return data
    raise ValueError(f"{path}: expected a shard dict (with n, e, entries) or a list of entries")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _standard_basis(n: int) -> list[Vec]:
    return [tuple(1 if j == i else 0 for j in range(n)) for i in range(n)]


def _content_depth(entry: dict, default: int = 6) -> int:
    d = entry.get("content_depth")
    if isinstance(d, int):
        return d
    cert = entry.get("certification") or ""
    if "@" in cert:
        try:
            return int(cert.rsplit("@", 1)[1].split(";")[0])
        except ValueError:
            pass
    return default


def entry_to_bpskalgebra(
    entry: dict,
    *,
    verify: str = "off",
) -> BPSKAlgebra:
    """Construct a `BPSKAlgebra` from a single dictionary entry.

    The entry's `exchange` matrix becomes the pairing; node charges are
    the standard basis (active-only convention); the `spec` is passed
    directly.  Verification is set to `"off"` by default since dictionary
    entries are pre-validated by their authoring pipeline.

    For entries with `det(exchange) = 0`, the constructor automatically
    picks up `Γ_f = ker(exchange)` via SNF and yields a flavoured BPSKAlgebra
    with `coefficient_ring = AbelianZPlusRing(rank=f)`.
    """
    if "exchange" not in entry or "spec" not in entry:
        raise ValueError(
            f"dictionary entry missing 'exchange' or 'spec': "
            f"keys = {sorted(entry.keys())}"
        )
    n = len(entry["exchange"])
    nodes = _standard_basis(n)
    if entry["spec"] is None:
        # An ENUMERATED-dictionary entry with no accepted spec:
        # construct in spec-free mode — the crystalline `S` from the leading
        # data at the entry's recorded content depth, ρ via the tRG fallback
        # (the principled σ = −upper(F) does not close where no finite chamber
        # exists — Markov — so it is not the default here).  The depth comes
        # from `content_depth` (rich entries) or the certification string
        # `crystalline_only@D` (shipped entries); 6 if neither is present —
        # never `None`, which would auto-stabilise and cannot terminate where
        # σ does not close.
        return BPSKAlgebra(
            pairing=entry["exchange"],
            node_charges=nodes,
            build_S=True,
            build_S_cutoff=_content_depth(entry),
            extract_spec=False,
            spec_free_sigma="trg",
            verify=verify,
        )
    return BPSKAlgebra(
        pairing=entry["exchange"],
        node_charges=nodes,
        spec=[tuple(g) for g in entry["spec"]],
        verify=verify,
    )


def iter_bpskalgebras_from_dictionary(
    path: str | Path,
    *,
    limit: int | None = None,
    verify: str = "off",
) -> Iterator[tuple[str, BPSKAlgebra]]:
    """Iterate over `(name, BPSKAlgebra)` pairs from a dictionary file.

    `limit`: if set, stops after that many entries.
    """
    entries = load_dictionary(path)
    for i, entry in enumerate(entries):
        if limit is not None and i >= limit:
            break
        name = entry.get("name", f"<entry {i}>")
        yield name, entry_to_bpskalgebra(entry, verify=verify)


# ---------------------------------------------------------------------------
# Defaults — the canonical dictionary lives next to this file.
# ---------------------------------------------------------------------------


def default_dictionary_dir() -> Path:
    """Path to the canonical BPS-quiver dictionary tree.

    Resolves to `restructuring/dictionaries/` (next to this loader),
    which contains the iso-witness-deduplicated, mutation-closed
    canonical entries.  The pre-sharpening raw tree is preserved at
    the archived tree (relative to the repo root) and
    extends to `n = 9, 10` where the canonical build currently caps
    at `n = 8`."""
    return Path(_HERE) / "dictionaries"


def default_enumerated_dictionary_dir() -> Path:
    """Path to the **enumerated** BPS-quiver dictionary, the
    PERMANENT tier: `dictionaries/enumerated/`, tracked — one gzipped
    `q_n{NN}_e{EE}.json.gz` per (rank, total-arrow-weight) cell holding the
    minimal entries `{exchange, spec | null, certification}` (every connected
    quiver with `e ≤ E`, up to node permutation, one accepted spec each), plus
    `manifest.json`.  `entry_to_bpskalgebra` builds `spec = null` entries
    spec-free.  The rich, semi-permanent build lives beside it (untracked):
    `default_enumerated_build_dir()`."""
    return Path(_HERE) / "dictionaries" / "enumerated"


def default_enumerated_build_dir() -> Path:
    """The SEMI-PERMANENT tier of the enumerated dictionary:
    `dictionaries/enumerated_build/`, untracked and regenerable
    (`PYTHONPATH=. python dictionaries/build_enumerated.py --E 7`) — the rich
    per-cell files (accepted spec families with producer / provenance /
    acceptance records, exits, class ids, fingerprints), the provenance tail,
    the class table and the full manifest."""
    return Path(_HERE) / "dictionaries" / "enumerated_build"


def _cell_files(d: Path) -> list[Path]:
    """The cell files of an enumerated tree in `(n, e)` order, a gzipped
    file taking precedence over a plain one for the same cell."""
    seen: dict[str, Path] = {}
    for path in sorted(d.glob("q_n*_e*.json")) + sorted(d.glob("q_n*_e*.json.gz")):
        stem = path.name[:-3] if path.name.endswith(".gz") else path.name
        seen[stem] = path
    return [seen[k] for k in sorted(seen)]


def iter_enumerated_entries(dict_dir: str | Path | None = None, *,
                            include_tail: bool = False,
                            include_acyclic: bool = False) -> Iterator[dict]:
    """Iterate the entry dicts of an enumerated dictionary (shipped or rich),
    cells in `(n, e)` order, then the tail files if `include_tail` (rich
    builds only).

    **Shard format 2 STORES only the cyclic quivers** — an acyclic quiver's
    spec is its source/sink order, a theorem, so those entries are coded
    rather than written (`ship()`).  This iterator therefore yields the
    stored entries by default; `include_acyclic=True` regenerates the coded
    ones by re-enumerating the set (`quiver_enumeration`, seconds at weight 8,
    minutes at weight 9 and up) and yields them with their strip specs, so
    the iteration is over every connected quiver of the set.  The per-cell
    census in each shard header (`count` / `acyclic` / `stored`) is what makes
    the completeness statement checkable either way."""
    d = Path(dict_dir) if dict_dir is not None else default_enumerated_dictionary_dir()
    if include_acyclic and _strongly_connected_tier(d) is not None:
        raise ValueError(
            f"{d} is a strongly connected tier (shard format 3): it stores the strongly "
            f"connected quivers, and every other quiver is ANSWERED by composition "
            f"(lookup_enumerated), not coded per cell — iterating every connected quiver "
            f"means enumerating them (quiver_enumeration.enumerate_connected_quivers) and "
            f"looking each one up")
    stored_keys: set = set()
    for path in _cell_files(d):
        for entry in load_dictionary(path):
            if include_acyclic:
                stored_keys.add(tuple(tuple(int(x) for x in row) for row in entry["exchange"]))
            yield entry
    if include_acyclic:
        # Regenerate the CODED half.  An acyclic quiver whose stored spec was
        # not a source/sink order is kept in the shards (`_split_codable`), so
        # it is already above and must not be yielded twice.
        from quiver_enumeration import enumerate_connected_quivers, is_acyclic
        manifest = d / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"{d}: include_acyclic needs manifest.json for the weight bound E")
        header = json.loads(manifest.read_text())
        cert = _acyclic_certification(header)
        for key in enumerate_connected_quivers(header["E"]):
            if is_acyclic(key) and key not in stored_keys:
                yield acyclic_entry([list(r) for r in key], cert)
    if include_tail:
        for path in sorted(d.glob("tail_n*.json")) + sorted(d.glob("tail_n*.json.gz")):
            yield from load_dictionary(path)


def _acyclic_certification(header: dict) -> str:
    """What the build recorded for the coded acyclic entries: `green` (the
    replay alone) or `green;axioms@D` where the verification pass ran at
    strip depth `D`.  The spec a lookup returns may be a DIFFERENT source/sink
    order than the one the build verified; the two differ by commuting swaps
    of non-adjacent nodes, so they give the same `S` and the certification
    transfers (pinned in the suite in the source repository)."""
    certs = header.get("acyclic_certifications") or {}
    if certs:
        # several depths in one cell: report the WEAKEST, since the caller's
        # quiver could be any of them
        def depth(c: str) -> int:
            return int(c.split("@")[1]) if "@" in c else 0
        return min(certs, key=depth)
    d = header.get("verify_strip_depth") or 0
    return f"green;axioms@{d}" if d else "green"


_SC_TIERS: dict = {}


def _strongly_connected_tier(d: Path) -> dict | None:
    """The in-memory index of a format-3 tier — `{"E", "entries": {(n, q):
    compact entry}}` — read once per directory and manifest version; `None`
    for a tier of another format (or no manifest)."""
    mpath = Path(d) / "manifest.json"
    if not mpath.exists():
        return None
    stamp = (str(mpath.resolve()), mpath.stat().st_mtime_ns)
    hit = _SC_TIERS.get(stamp)
    if hit is None:
        m = json.loads(mpath.read_text())
        if m.get("format") != 3:
            hit = False
        else:
            entries = {}
            for path in _cell_files(Path(d)):
                shard = load_shard(path, expand=False)
                for raw in shard["entries"]:
                    entries[(shard["n"], raw["q"])] = raw
            hit = {"E": m["E"], "entries": entries}
        _SC_TIERS[stamp] = hit
    return hit or None


def compose_certification(certs) -> str:
    """The record of a composed answer from the records of its components of
    two or more nodes (a single node satisfies the BPS quiver
    constraint exactly, at every depth, so it lowers nothing): green iff every
    component is; `axioms@D` at the least depth among them; a component's
    `axiom_mismatch@D` carried (the shallowest); `crystalline_only@D` (no spec)
    if one is, at the least such depth.  No component — an acyclic quiver — is
    `green`: its source/sink order, whose `S` satisfies the constraint at every
    depth (the design notes §1.2, a theorem)."""
    certs = [str(c) for c in certs]
    if not certs:
        return "green"

    def depth(c):
        return int(c.rsplit("@", 1)[1]) if "@" in c else None
    mismatch = [c for c in certs if "axiom_mismatch@" in c]
    if mismatch:
        return min(mismatch, key=depth)
    crystalline = [c for c in certs if c.startswith("crystalline_only")]
    if crystalline:
        return f"crystalline_only@{min(depth(c) for c in crystalline)}"
    axioms = [depth(c) for c in certs if "axioms@" in c]
    if all(c.startswith("green") for c in certs):
        return f"green;axioms@{min(axioms)}" if len(axioms) == len(certs) else "green"
    if len(axioms) == len(certs):
        return f"axioms@{min(axioms)}"         # a non-green component, every one checked
    return "none"                              # a non-green part next to an unchecked one: no claim


def _lookup_composed(B, tier) -> tuple[dict, list[Vec] | None] | None:
    """`lookup_enumerated` on a format-3 tier: compose from the components."""
    from quiver_enumeration import (apply_perm, arrows_string, canonical_form,
                                    strongly_connected_components, total_weight)
    from spec_acceptance import spec_from_components
    if len(B) == 0:
        return None
    parts = []
    for comp in strongly_connected_components(B):
        if len(comp) == 1:
            continue
        sub = [[B[i][j] for j in comp] for i in comp]
        if total_weight(sub) > tier["E"]:
            return None                    # a component beyond the tier: outside the coverage
        cf = canonical_form(sub)
        raw = tier["entries"].get((len(comp), arrows_string(cf.key)))
        if raw is None:
            return None                    # not stored: the tier is not complete at this cell
        parts.append((comp, cf, expand_entry(raw, len(comp))))
    todo = iter(parts)

    def component_spec(sub):
        # `spec_from_components` walks the same components in the same order
        comp, cf, ent = next(todo)
        if ent["spec"] is None:
            return None
        return [apply_perm(g, cf.perm) for g in ent["spec"]]
    spec = spec_from_components(B, component_spec)
    entry = {"exchange": [list(r) for r in B],
             "spec": None if spec is None else [list(g) for g in spec],
             "certification": compose_certification(ent["certification"] for _, _, ent in parts),
             "components": [{"nodes": list(comp), "exchange": ent["exchange"],
                             "certification": ent["certification"]} for comp, _, ent in parts]}
    return entry, (None if spec is None else [tuple(g) for g in spec])


def lookup_enumerated(exchange, dict_dir: str | Path | None = None
                      ) -> tuple[dict, list[Vec] | None] | None:
    """Find the entry of a BPS quiver given in ANY node order.

    Returns `(entry, spec)` with `spec` in the CALLER's node coordinates
    (`None` when the quiver has no accepted spec, `crystalline_only`), or
    `None` when the quiver is outside the tier's coverage.

    **Shard format 3** (the strongly connected tier):
    the quiver is split into its strongly connected components; each one of
    two or more nodes is looked up in the tier (an in-memory index, loaded
    once), and the spec is the components' specs concatenated in a
    source-first order, a single node contributing its own charge
    (`spec_acceptance.spec_from_components`; the design notes
    §1.1).  The entry is the composed answer: `exchange` in the caller's
    order, the record `compose_certification` gives, and the stored
    `components` it came from.  Coverage: every quiver — connected or not —
    whose strongly connected components have total arrow weight `<= E`.

    **Shard formats 1 and 2** (the tier of every connected quiver):
    canonicalises `exchange`, opens the one cell file `(n, e)` it belongs to
    (an acyclic quiver is answered by its source/sink order, format 2 coding
    it), and returns the stored entry; outside the tree (disconnected, or `e`
    beyond the shipped `E`) is `None`."""
    from quiver_enumeration import (apply_perm, arrows_string, canonical_form, is_acyclic,
                                    is_connected, total_weight)
    B = [[int(x) for x in row] for row in exchange]
    n = len(B)
    d = Path(dict_dir) if dict_dir is not None else default_enumerated_dictionary_dir()
    tier = _strongly_connected_tier(d)
    if tier is not None:
        return _lookup_composed(B, tier)
    if n == 0 or not is_connected(B):
        return None
    e = total_weight(B)
    candidates = [d / f"q_n{n:02d}_e{e:02d}.json.gz", d / f"q_n{n:02d}_e{e:02d}.json"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None          # outside the tree: the cell is beyond the shipped E
    shard = load_shard(path, expand=False)
    # An ACYCLIC quiver is coded, not stored (shard format 2): its source/sink
    # order is a spec, so it is computed here, in the caller's own node order,
    # with no search through the cell.  (A format-1 shard stores it; computing
    # it is still a correct answer, and the two give the same S.)
    if is_acyclic(B):
        return acyclic_entry(B, _acyclic_certification(shard)), \
            [tuple(g) for g in acyclic_entry(B)["spec"]]
    cf = canonical_form(B)
    key = [list(r) for r in cf.key]
    want_q = arrows_string(cf.key)
    for raw in shard["entries"]:
        if raw.get("q", None) == want_q or raw.get("exchange") == key:
            entry = expand_entry(raw, n)
            spec = entry.get("spec")
            if spec is None:
                return entry, None
            # canonical coordinate i is the caller's node perm[i]
            return entry, [apply_perm(g, cf.perm) for g in spec]
    return None


# --------------------------------------------------------------------------
# the FLAVOURED tier — quivers carrying a manifest SU(2)
# --------------------------------------------------------------------------

def default_flavoured_dictionary_dir(orbit_size=2) -> Path:
    """Path to a **flavoured** BPS-quiver dictionary.  All of them are tracked:
    one gzipped `f_n{NN}_e{EE}.json.gz` per (rank, total-arrow-weight) cell, plus
    `manifest.json`.

    `orbit_size` is the tier's selector, and it names the directory:

    * an **int** `N` — one node orbit of size `N`, a manifest `SU(N)`:
      `dictionaries/flavoured/` at `N = 2`, `dictionaries/flavoured_su{N}/` above.
    * a **sequence** — a family of pairwise-DISJOINT orbits of those sizes, a
      manifest product: `(2, 2)` is `SU(2)xSU(2)` in
      `dictionaries/flavoured_su2su2/`, `(3, 2)` is `SU(3)xSU(2)` in
      `dictionaries/flavoured_su3su2/`.

    An entry is a quiver of the enumerated tier carrying that orbit structure —
    a set of `N` pairwise non-paired nodes every permutation of which is an
    automorphism, so they are identical and interchangeable and the theory has a
    manifest `SU(N)` rotating them — together with a covariant spec where one was
    found and the covariant crystalline `Ω(γ, r)` otherwise.

    ⚠ The selector is the ORBIT SIZE, not a doublet count: three identical
    nodes are ONE orbit of size 3 and contain three doublet PAIRS, so `SU(3)` is
    not "three doublets", and two DISJOINT pairs (`SU(2)xSU(2)`) is a third
    object again, distinct from both.  Each tier's `manifest.json` records its
    own `orbit_size`.

    The dictionary is keyed on the **unfolded** quiver (the honest BPS quiver),
    so it refines `default_enumerated_dictionary_dir()` as far as that tier
    reaches.  Builder: `dictionaries/build_flavoured.py`."""
    sizes = ((orbit_size,) if isinstance(orbit_size, int)
             else tuple(int(k) for k in orbit_size))
    if sizes == (2,):
        name = "flavoured"
    else:
        name = "flavoured_" + "".join(f"su{k}" for k in sizes)
    return Path(_HERE) / "dictionaries" / name


def _flavoured_cell_files(d: Path) -> list[Path]:
    seen: dict[str, Path] = {}
    for path in sorted(d.glob("f_n*_e*.json")) + sorted(d.glob("f_n*_e*.json.gz")):
        stem = path.name[:-3] if path.name.endswith(".gz") else path.name
        seen[stem] = path
    return [seen[k] for k in sorted(seen)]


# The FLAVOURED tiers in shard format 3: only the strongly
# connected flavoured quivers are stored; every other one is composed.
_FLAV_SC_TIERS: dict = {}


def _expand_flavoured_sc(raw: dict, n: int) -> tuple[tuple, dict]:
    """A stored format-3 flavoured entry in the format-2 entry's shape
    (`exchange`, `reduced`, `factors`, `spec`, `omega`, `certification`) plus
    `family` and `source`; returned with the key."""
    from quiver_enumeration import matrix_from_arrows
    from dictionaries.build_flavoured import fold_orbits
    B = matrix_from_arrows(n, raw["q"])
    fam = [tuple(int(v) for v in o) for o in raw["family"]]
    colours = [0] * n
    for o in fam:
        for v in o:
            colours[v] = len(o)
    fq = fold_orbits(B, fam)
    ent = {"exchange": [list(r) for r in B], "family": [list(o) for o in fam],
           "reduced": [list(r) for r in fq.pairing], "factors": list(fq.flavour.factors),
           "certification": raw["c"], "source": raw.get("source", "")}
    if raw.get("spec") is not None:
        ent["spec"] = raw["spec"]
    if raw.get("omega") is not None:
        ent["omega"] = raw["omega"]
    return (tuple(tuple(r) for r in B), tuple(colours)), ent


def _flavoured_sc_tier(d: Path) -> dict | None:
    """The in-memory index of a format-3 flavoured tier — `{"sizes",
    "max_weight", "entries": {key: entry}}` with the key — read once per
    directory and manifest version; `None` for a tier of another format (or no
    manifest)."""
    import gzip
    mpath = Path(d) / "manifest.json"
    if not mpath.exists():
        return None
    stamp = (str(mpath.resolve()), mpath.stat().st_mtime_ns)
    hit = _FLAV_SC_TIERS.get(stamp)
    if hit is None:
        m = json.loads(mpath.read_text())
        if m.get("format") != 3:
            hit = False
        else:
            entries = {}
            for path in _flavoured_cell_files(Path(d)):
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    body = json.load(fh)
                for raw in body["entries"]:
                    key, ent = _expand_flavoured_sc(raw, body["n"])
                    entries[key] = ent
            hit = {"sizes": [int(s) for s in m["sizes"]], "max_weight": m["max_weight"],
                   "entries": entries}
        _FLAV_SC_TIERS[stamp] = hit
    return hit or None


def _lookup_flavoured_composed(B, tier) -> dict | None:
    """`lookup_flavoured` on a format-3 tier.

    The quiver's orbit family (the tier's own admission: exactly one family of
    its sizes) fixes the canonical form.  Its components are then composed
    source-first, each orbit of single-node components placed consecutively
    (the design notes §1.2):
    - a plain component from the enumerated tier;
    - a component carrying orbits from the flavoured tier of the sizes it
      carries;
    - an orbit of single nodes as its fundamental generator.
    The record follows `compose_certification`."""
    from quiver_enumeration import strongly_connected_components, total_weight
    from dictionaries.build_flavoured import (canonical_family, fold_map, fold_orbits,
                                              move_flavoured_spec, node_orbit_families,
                                              strongly_connected_flavoured_key, _fold_index)
    sizes = list(tier["sizes"])
    fams = node_orbit_families(B, sizes)
    if len(fams) != 1:
        return None
    _key, cf = strongly_connected_flavoured_key(B, fams[0])
    Bc = [list(r) for r in cf.key]
    famc = canonical_family(cf, fams[0])
    fq = fold_orbits(Bc, famc)
    F = fq.flavour
    fold_of = _fold_index(len(Bc), famc)
    orbit_of = {v: tuple(o) for o in famc for v in o}
    comps = strongly_connected_components(Bc)
    pieces, parts, done = [], [], set()
    omega = None
    for comp in comps:
        if len(comp) == 1:
            v = comp[0]
            if v in done:
                continue
            e = [0] * fq.rank
            if v in orbit_of:
                done.update(orbit_of[v])
                a = fold_of[v]
                e[a] = 1
                pieces.append([(tuple(e), F.fundamental(a))])
            else:
                e[fold_of[v]] = 1
                pieces.append([(tuple(e), F.trivial_irrep())])
            continue
        inside = sorted({orbit_of[v] for v in comp if v in orbit_of})
        sub = [[Bc[i][j] for j in comp] for i in comp]
        if not inside:
            got = lookup_enumerated(sub)
            if got is None:
                return None                       # beyond the enumerated tier: outside the coverage
            entry, spec = got
            parts.append({"nodes": list(comp), "certification": entry["certification"],
                          "tier": "enumerated"})
            if spec is None:
                pieces.append(None)
                continue
            gens = []
            for g in spec:
                r = [0] * fq.rank
                for i, x in enumerate(g):
                    if x:
                        r[fold_of[comp[i]]] += int(x)
                gens.append((tuple(r), F.trivial_irrep()))
            pieces.append(gens)
            continue
        loc = [tuple(comp.index(v) for v in o) for o in inside]
        ckey, ccf = strongly_connected_flavoured_key(sub, loc)
        csizes = sorted((len(o) for o in inside), reverse=True)
        ctier = (tier if csizes == sorted(sizes, reverse=True) else
                 _flavoured_sc_tier(default_flavoured_dictionary_dir(
                     csizes[0] if len(csizes) == 1 else tuple(csizes))))
        if ctier is None or total_weight(sub) > ctier["max_weight"]:
            return None                           # no tier of those sizes reaches it: outside the coverage
        rec = ctier["entries"].get(ckey)
        if rec is None:
            return None
        parts.append({"nodes": list(comp), "certification": rec["certification"],
                      "tier": "x".join(f"SU({s})" for s in csizes)})
        if len(comp) == len(Bc) and rec.get("omega") is not None:
            omega = rec["omega"]                  # the quiver IS the stored one: same canonical fold
        if rec.get("spec") is None:
            pieces.append(None)
            continue
        canon_fold_of = _fold_index(len(rec["exchange"]), rec["family"])
        node_map = {i: comp[int(ccf.perm[i])] for i in range(len(comp))}
        fmap = fold_map(canon_fold_of, fold_of, node_map)
        spec = [(tuple(g), tuple(tuple(p) for p in r)) for g, r in rec["spec"]]
        pieces.append(move_flavoured_spec(spec, fmap, fq.rank, F.trivial_irrep()))
    out = {"exchange": [list(r) for r in Bc], "family": [list(o) for o in famc],
           "reduced": [list(r) for r in fq.pairing], "factors": list(fq.flavour.factors),
           "certification": compose_certification([p["certification"] for p in parts]),
           "perm": [int(p) for p in cf.perm], "components": parts}
    if all(p is not None for p in pieces):
        out["spec"] = [[list(g), [list(x) for x in r]] for piece in pieces for g, r in piece]
    if omega is not None:
        out["omega"] = omega
    return out


def iter_flavoured_entries(dict_dir: str | Path | None = None, *,
                           include_acyclic: bool = False,
                           orbit_size=2) -> Iterator[dict]:
    """Iterate the entries of the flavoured dictionary, cells in `(n, e)` order.

    As in the enumerated tier, **the acyclic quivers are coded, not stored**:
    an acyclic flavoured quiver's covariant spec is the source/sink order with
    the two doublet nodes consecutive, which is a construction rather than a
    search, so those entries are recomputed instead of written.  This iterator
    therefore yields the stored (cyclic) entries by default, and with
    `include_acyclic=True` re-enumerates the set and regenerates the coded half
    — complete, at the cost of the enumeration.  A scan that forgets the flag
    silently omits the bulk of the set (84 % of it at weight 9) and with it the
    `[A_1, D_3]` anchor.  Each shard header's `count` / `acyclic` / `stored` is
    what makes the completeness statement checkable either way."""
    import gzip
    d = (Path(dict_dir) if dict_dir is not None
         else default_flavoured_dictionary_dir(orbit_size))
    tier = _flavoured_sc_tier(d)
    if tier is not None:
        # shard format 3: the stored entries are the strongly
        # connected flavoured quivers; every other one is composed on lookup
        if include_acyclic:
            raise ValueError(
                f"{d} is a strongly connected flavoured tier (shard format 3): it stores only the "
                "strongly connected flavoured quivers and composes the rest, so there is no coded "
                "half to regenerate; iterate by enumerating the quivers wanted and calling "
                "lookup_flavoured on each")
        for ent in tier["entries"].values():
            yield dict(ent)
        return
    for path in _flavoured_cell_files(d):
        with (gzip.open(path, "rt", encoding="utf-8") if path.name.endswith(".gz")
              else open(path, "r", encoding="utf-8")) as fh:
            body = json.load(fh)
        yield from body["entries"]
    if not include_acyclic:
        return
    from dictionaries.build_flavoured import flavoured_quivers, _entry_for
    manifest = d / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{d}: include_acyclic needs manifest.json for the weight bound")
    header = json.loads(manifest.read_text())
    for rec in flavoured_quivers(doublets=header.get("doublets", 1),
                                 max_weight=header["max_weight"],
                                 orbit_size=header.get("orbit_size", 2)):
        from quiver_enumeration import is_acyclic
        if not is_acyclic(rec["exchange"]):
            continue
        yield _entry_for(rec["exchange"], rec["doublets"][0], 4, 3, 6).as_json()


def lookup_flavoured(exchange, dict_dir: str | Path | None = None, *,
                     orbit_size=2) -> dict | None:
    """The flavoured entry of a BPS quiver given in ANY node order.

    Returns the entry dict augmented with `perm` — `canonical_form(B).perm`,
    so canonical coordinate `i` is the caller's node `perm[i]` — or `None` when
    the quiver is outside the tree: disconnected, beyond the shipped weight, or
    carrying a number of node doublets other than the manifest's `doublets`.
    Note that the entry's `spec` lives on the **reduced** (folded) quiver
    recorded in the same entry, not on the caller's unfolded one; `perm` is what
    relates the caller's nodes to the canonical unfolded ones.

    An **acyclic** quiver is answered by construction rather than by opening a
    file, exactly as in `lookup_enumerated`: its covariant spec is the reduced
    source/sink order, a theorem.

    **Shard format 3** is read first.  The tier stores only
    the strongly connected flavoured quivers, and every other quiver with the
    tier's orbit family — acyclic, cyclic but not strongly connected, or
    disconnected — is composed from its strongly connected components
    (`_lookup_flavoured_composed`).  The entry comes in the same shape, with
    three additions:
    - `family`: the orbits in the canonical node indices;
    - `components`: what it was composed from, with each component's record;
    - the composed record in `certification`.

    `perm` relates the caller's nodes to the canonical form: the canonical
    form coloured by orbit size.  `omega` is given only for a stored
    (strongly connected) crystalline entry."""
    import gzip
    from quiver_enumeration import (canonical_form, is_acyclic, is_connected,
                                    total_weight)
    from dictionaries.build_flavoured import flavoured_entry, node_orbits

    B = [[int(x) for x in row] for row in exchange]
    n = len(B)
    d = (Path(dict_dir) if dict_dir is not None
         else default_flavoured_dictionary_dir(orbit_size))
    tier = _flavoured_sc_tier(d)
    if tier is not None:
        return _lookup_flavoured_composed(B, tier) if n else None
    if n == 0 or not is_connected(B):
        return None
    manifest = d / "manifest.json"
    header = json.loads(manifest.read_text()) if manifest.exists() else {}
    doublets = header.get("doublets", 1)
    size = header.get("orbit_size", orbit_size)
    if len(node_orbits(B, size)) != doublets:
        return None
    e = total_weight(B)
    if header and e > header.get("max_weight", e):
        return None
    cf = canonical_form(B)
    if is_acyclic(B):
        ent = flavoured_entry([list(r) for r in cf.key], doublets=doublets,
                              orbit_size=size)
        if ent is None:
            return None
        out = ent.as_json()
        out["perm"] = list(cf.perm)
        return out
    path = next((p for p in (d / f"f_n{n:02d}_e{e:02d}.json.gz",
                             d / f"f_n{n:02d}_e{e:02d}.json") if p.exists()), None)
    if path is None:
        return None
    with (gzip.open(path, "rt", encoding="utf-8") if path.name.endswith(".gz")
          else open(path, "r", encoding="utf-8")) as fh:
        body = json.load(fh)
    key = [list(r) for r in cf.key]
    for entry in body["entries"]:
        if entry["exchange"] == key:
            out = dict(entry)
            out["perm"] = list(cf.perm)
            return out
    return None


def default_z_dictionary_dir() -> Path:
    """Path to the **Z-admissible** BPS-quiver dictionary tree.

    Resolves to `dictionaries_z/` (sibling of `dictionaries/`),
    which contains entries that carry an exact-rational central
    charge `Z = (a, b)` alongside `(B, spec)`.  Built by
    `dictionaries_z/build.py` and `dictionaries_z/full_closure.py`
    from ACVRV seeds via necklace + RG closures.

    Per-rank shards use the prefix ``z_n_NNN.json``; the entry
    schema includes a ``Z`` field (rational `a` and `b` as
    ``[num, den]`` pairs) plus the standard ``B``, ``spec``,
    ``provenance``.

    Distinct from the canonical and weak trees: each entry
    represents a *physically realized chamber* with its own Z, so
    necklace-rotation-related variants are kept as distinct
    entries rather than merged.  The three trees grow
    independently.
    """
    return Path(_HERE) / "dictionaries_z"


def default_weak_dictionary_dir() -> Path:
    """Path to the **weak** BPS-quiver dictionary tree.

    Resolves to `dictionaries_weak/` (sibling of `dictionaries/`),
    which contains the weak-merged entries produced by
    `dictionaries_weak/build_weak.py`.  The weak tree uses the same
    file shapes (`n_NNN.json`, `graph_nNNN.json`, `rg_edges.json`) as
    the canonical tree, so every reader in this loader works
    unchanged on the weak tree.

    Distinct from the canonical tree: the weak builder merges entries
    on `S_n`-equivalence of the exchange matrix alone (no spec-level
    iso witness), per the weak-dictionary assumption that any two
    certified specs on the same BPS quiver give isomorphic algebras.
    The three trees grow independently.
    """
    return Path(_HERE) / "dictionaries_weak"


# ---------------------------------------------------------------------------
# Mutation-graph loaders
# ---------------------------------------------------------------------------


def load_mutation_graph(path: str | Path) -> dict:
    """Load a `graph_n*.json` file: a dict with `entries` (list of quiver
    entries) and `edges` (list of `{source, target, via, target_via}`
    records).  Returns the dict as-is for caller inspection."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a dict (mutation graph), got {type(data).__name__}")
    if "entries" not in data or "edges" not in data:
        raise ValueError(
            f"{path}: missing 'entries' or 'edges'; "
            f"keys = {sorted(data.keys())}"
        )
    return data


def iter_bpskalgebras_from_graph(
    graph: dict,
    *,
    limit: int | None = None,
    verify: str = "off",
) -> Iterator[tuple[str, BPSKAlgebra]]:
    """Iterate over `(name, BPSKAlgebra)` pairs from a mutation-graph's
    `entries`.  Edges live on `graph["edges"]` for the caller to consume
    separately."""
    entries = graph["entries"]
    for i, entry in enumerate(entries):
        if limit is not None and i >= limit:
            break
        name = entry.get("name", f"<entry {i}>")
        yield name, entry_to_bpskalgebra(entry, verify=verify)


def find_entry_by_exchange(
    graph: dict, exchange: list[list[int]],
) -> int | None:
    """Locate the index of a graph entry whose `exchange` matches.
    Returns `None` if not found.  Useful for resolving mutation edges
    back to entry indices."""
    target = [list(row) for row in exchange]
    for i, entry in enumerate(graph["entries"]):
        if [list(row) for row in entry["exchange"]] == target:
            return i
    return None


__all__ = [
    "load_dictionary",
    "entry_to_bpskalgebra",
    "iter_bpskalgebras_from_dictionary",
    "load_mutation_graph",
    "iter_bpskalgebras_from_graph",
    "find_entry_by_exchange",
    "default_dictionary_dir",
    "default_z_dictionary_dir",
    "default_weak_dictionary_dir",
]
