"""`SharpenedBPSQuiverDictionary` — a dictionary that distinguishes
`(B, spec)` entries which are not iso-witness-equivalent.

The existing `BPSQuiverDictionary` (in `bps_quiver_dictionary.py`)
identifies two entries when they share an `S_n`-permutation-invariant
exchange-matrix signature, **regardless of spec** -- it keeps just
the shortest-spec representative per quiver.

This module provides a stricter alternative: two entries are
identified iff they are witnessed isomorphic by
`bpskalgebra_iso.find_isomorphism`, which means there is a unimodular
`A : Γ_1 → Γ_2` with `A^T B_2 A = B_1` mapping `nodes_1` to `nodes_2`
as a multiset, **and** `A · spec_1` is reachable from `spec_2` by a
finite chain of algebra-preserving local moves (pentagon collapses +
commute swaps + pentagon expansions).

Distinct specs on the same quiver that are *not* local-move-related
are kept as separate entries.

Bucketing.  We still bucket by the cheap exchange-matrix signature
(`bps_quiver_dictionary.signature`); within a bucket we run pairwise
`find_isomorphism` against the existing entries.  This keeps
registration cheap on quivers that don't collide, and pays the
iso-search cost only on the small bucket of same-quiver candidates.

Persistence.  JSON, mirroring the archived tree
shape but with a list of *all* entries (no internal dedup beyond what
`register` performed at build time).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Sequence
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bps_quiver_dictionary import (
    QuiverEntry,
    signature as _exchange_signature,
    _spec_sort_key,
)
from bpskalgebra_iso import find_isomorphism_raw, Iso


Vec = tuple[int, ...]


# ---------------------------------------------------------------------------
# Iso-witness predicate at the QuiverEntry level
# ---------------------------------------------------------------------------


def iso_equivalent_entries(
    e1: QuiverEntry,
    e2: QuiverEntry,
    *,
    max_states: int = 4096,
    max_extra_length: int = 4,
    bidirectional: bool = True,
    cache: Optional[dict] = None,
) -> bool:
    """True iff there is an isomorphism witness `(A, local-moves chain)`
    relating `e1` and `e2`.

    Calls `find_isomorphism_raw` directly on the `(exchange, nodes,
    spec)` tuples extracted from each `QuiverEntry`, skipping the
    `BPSKAlgebra` wrapper construction the legacy path used to pay
    twice per call.  Both entries are assumed to use the standard-
    basis convention (which `QuiverEntry`s in the dictionary always
    do); the implicit coefficient-ring agreement is therefore
    automatic.

    Returns False on any structural mismatch (rank) or when no
    witness is found within the search budget.
    """
    if e1.n_nodes != e2.n_nodes:
        return False
    n = e1.n_nodes
    nodes_std = [
        tuple(1 if k == i else 0 for k in range(n)) for i in range(n)
    ]
    iso = find_isomorphism_raw(
        e1.exchange, nodes_std, e1.spec,
        e2.exchange, nodes_std, e2.spec,
        max_states=max_states, max_extra_length=max_extra_length,
        bidirectional=bidirectional,
        cache=cache,
    )
    return iso is not None


# ---------------------------------------------------------------------------
# SharpenedBPSQuiverDictionary
# ---------------------------------------------------------------------------


class SharpenedBPSQuiverDictionary:
    """Dictionary of `QuiverEntry` objects identified by the iso-witness
    predicate `(S_n × local-moves)` rather than by exchange-matrix alone.

    Internal layout: `dict[exchange_signature, list[QuiverEntry]]`,
    where each list contains pairwise non-iso-equivalent entries.

    Registration policy on iso-equivalence collision: keep the
    shortest-spec representative (lexicographic tiebreaker), and union
    the `known_specs` field across the merged pair.
    """

    def __init__(
        self,
        *,
        iso_max_states: int = 4096,
        iso_max_extra_length: int = 4,
        iso_bidirectional: bool = True,
        iso_bucket_cache: bool = True,
    ):
        self._buckets: dict[tuple, list[QuiverEntry]] = {}
        self._iso_max_states = iso_max_states
        self._iso_max_extra_length = iso_max_extra_length
        self._iso_bidirectional = iso_bidirectional
        # Per-bucket BFSExplorer caches.  Each bucket gets its own
        # `dict[(B_tuple, spec_tuple), BFSExplorer]`, populated lazily
        # by `iso_equivalent_entries` and reused across registrations
        # in the same bucket.  The shared `(B, spec)` key works because
        # all entries in a SharpenedBPSQuiverDictionary bucket have
        # exchange matrices that agree as a WL invariant; pairs that
        # share the exact `B` (the common case after WL bucketing)
        # also reuse `src_explorer` since `spec_1_mapped == spec_1`
        # under the identity aut.
        self._iso_bucket_cache = iso_bucket_cache
        self._bucket_caches: dict[tuple, dict] = {}

    # ----- public API ----------------------------------------------------

    def __len__(self) -> int:
        return sum(len(b) for b in self._buckets.values())

    def __iter__(self) -> Iterator[QuiverEntry]:
        for bucket in self._buckets.values():
            for entry in bucket:
                yield entry

    def __contains__(self, entry: QuiverEntry) -> bool:
        sig = _exchange_signature(entry)
        cache = (
            self._bucket_caches.setdefault(sig, {})
            if self._iso_bucket_cache else None
        )
        for existing in self._buckets.get(sig, []):
            if iso_equivalent_entries(
                entry, existing,
                max_states=self._iso_max_states,
                max_extra_length=self._iso_max_extra_length,
                bidirectional=self._iso_bidirectional,
                cache=cache,
            ):
                return True
        return False

    def entries(self) -> list[QuiverEntry]:
        return list(self)

    def register(self, entry: QuiverEntry) -> tuple[tuple, int]:
        """Register `entry`.  Returns `(exchange_signature, slot_index)`
        where `slot_index` is the position of the merged-or-appended
        entry within its bucket.  Disconnected entries are skipped
        (return `(sig, -1)`).
        """
        sig, slot, _canonical, _iso = self.register_with_witness(entry)
        return sig, slot

    def register_with_witness(
        self, entry: QuiverEntry,
    ) -> tuple[tuple, int, QuiverEntry, Iso | None]:
        """Like :meth:`register`, but additionally returns the canonical
        ``QuiverEntry`` currently occupying the slot and the iso witness
        ``Iso(A, local_move_chain)`` from
        :func:`bpskalgebra_iso.find_isomorphism_raw` proving the merge
        equivalence.

        Returns ``(exchange_signature, slot_index, canonical_entry,
        iso_witness)``.

        For new entries (no iso-equivalent existing entry found within
        budget), ``iso_witness`` is ``None`` and ``canonical_entry`` is
        the entry itself.  For merged entries, ``iso_witness.A`` is a
        unimodular matrix and ``iso_witness.local_move_chain`` is the
        (possibly empty) sequence of local moves that takes
        ``A * entry.spec`` to ``canonical_entry.spec``.

        Disconnected entries return ``(sig, -1, entry, None)``.

        NB: this is the strong-side analogue of
        :meth:`weak_dictionary.WeakBPSQuiverDictionary.register_with_witness`,
        which returns an ``S_n`` permutation instead of an iso witness.
        """
        sig = _exchange_signature(entry)
        if not entry.is_connected():
            return sig, -1, entry, None
        bucket = self._buckets.setdefault(sig, [])
        cache = (
            self._bucket_caches.setdefault(sig, {})
            if self._iso_bucket_cache else None
        )
        for i, existing in enumerate(bucket):
            if entry.n_nodes != existing.n_nodes:
                continue
            n = entry.n_nodes
            nodes_std = [
                tuple(1 if k == j else 0 for k in range(n))
                for j in range(n)
            ]
            iso = find_isomorphism_raw(
                entry.exchange, nodes_std, entry.spec,
                existing.exchange, nodes_std, existing.spec,
                max_states=self._iso_max_states,
                max_extra_length=self._iso_max_extra_length,
                bidirectional=self._iso_bidirectional,
                cache=cache,
            )
            if iso is not None:
                merged = _merge_keep_shorter(existing, entry, iso)
                bucket[i] = merged
                return sig, i, merged, iso
        bucket.append(entry)
        return sig, len(bucket) - 1, entry, None

    def replace_entry(
        self, sig: tuple, slot: int, new_entry: QuiverEntry,
    ) -> bool:
        """Replace the bucket slot ``(sig, slot)`` with ``new_entry``,
        preserving the slot index.  Used by tools that mutate an entry
        in place (e.g. :mod:`dictionaries_weak.spec_optimize` shortening
        the primary spec).

        The caller is responsible for ensuring ``new_entry`` belongs in
        the same bucket as the old entry (same exchange-signature) — no
        re-bucketing is performed.

        Returns ``True`` if the slot was found and replaced, ``False``
        otherwise.
        """
        bucket = self._buckets.get(sig)
        if bucket is None or not (0 <= slot < len(bucket)):
            return False
        bucket[slot] = new_entry
        return True

    def replace_known_specs(
        self,
        sig: tuple, slot: int,
        known_specs: tuple[tuple[Vec, ...], ...],
    ) -> bool:
        """Update the ``known_specs`` of the entry at bucket slot
        ``(sig, slot)``.  The primary ``spec`` and other fields stay
        unchanged.  Useful for trusted-closure builders that grow an
        entry's alt-spec store.
        """
        bucket = self._buckets.get(sig)
        if bucket is None or not (0 <= slot < len(bucket)):
            return False
        e = bucket[slot]
        bucket[slot] = QuiverEntry(
            name=e.name,
            exchange=e.exchange,
            spec=e.spec,
            negating_sequence=e.negating_sequence,
            provenance=e.provenance,
            known_specs=known_specs,
        )
        return True

    # ----- persistence ---------------------------------------------------

    def to_json(self) -> list[dict]:
        """Flat list of entries as JSON-ready dicts."""
        out: list[dict] = []
        for bucket in self._buckets.values():
            for e in bucket:
                out.append({
                    "name": e.name,
                    "exchange": [list(r) for r in e.exchange],
                    "spec": [list(g) for g in e.spec],
                    "negating_sequence":
                        list(e.negating_sequence) if e.negating_sequence
                        else None,
                    "provenance": e.provenance,
                    "known_specs": [
                        [list(g) for g in s] for s in e.known_specs
                    ],
                })
        return out

    def save(self, path: str | Path) -> None:
        """Write to a JSON file as a flat list of entries (matches
        the shape of the archived tree)."""
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        iso_max_states: int = 4096,
        iso_max_extra_length: int = 4,
        iso_bidirectional: bool = True,
    ) -> "SharpenedBPSQuiverDictionary":
        """Load a JSON file produced by `save` (or compatible).
        Each entry is registered through `register`; the iso-witness
        predicate is *not* re-run at load time -- entries written
        out are assumed pairwise non-iso, so they re-bucket
        cheaply (one find_isomorphism check per same-signature pair,
        all returning False)."""
        with open(path) as f:
            data = json.load(f)
        d = cls(
            iso_max_states=iso_max_states,
            iso_max_extra_length=iso_max_extra_length,
            iso_bidirectional=iso_bidirectional,
        )
        for raw in data:
            entry = QuiverEntry(
                name=raw.get("name", ""),
                exchange=tuple(tuple(r) for r in raw["exchange"]),
                spec=tuple(tuple(g) for g in raw["spec"]),
                negating_sequence=(
                    tuple(raw["negating_sequence"])
                    if raw.get("negating_sequence") else None
                ),
                provenance=raw.get("provenance", ""),
                known_specs=tuple(
                    tuple(tuple(g) for g in s)
                    for s in raw.get("known_specs", [])
                ),
            )
            d.register(entry)
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_A_to_spec(
    spec: Sequence[Vec], A: Sequence[Sequence[int]],
) -> tuple[Vec, ...]:
    """Apply a unimodular charge-basis-change matrix ``A`` to every
    charge in a spec.

    Convention matches :class:`bpskalgebra_iso.Iso`: the witness
    ``A`` satisfies ``A · alg_1.node_charges = alg_2.node_charges``
    and ``A · alg_1.spec`` is reachable from ``alg_2.spec`` via the
    ``local_move_chain``.  So a spec in basis-1 is transported into
    basis-2 by applying ``A`` to each charge.
    """
    n = len(A)
    return tuple(
        tuple(
            sum(A[i][j] * g[j] for j in range(n))
            for i in range(n)
        )
        for g in spec
    )


def _merge_keep_shorter(
    a: QuiverEntry,
    b: QuiverEntry,
    iso: Optional["Iso"] = None,
) -> QuiverEntry:
    """Merge two iso-equivalent entries.

    Keeps the shorter spec (lexicographic tiebreak); unions the
    ``known_specs`` after transporting the loser's specs through the
    iso witness ``A`` into the winner's basis.

    ``iso`` is the witness returned by ``find_isomorphism_raw(a, b)``:
    ``A · a.spec`` is local-move-reachable from ``b.spec``.

    If ``iso is None`` (i.e. ``a == b`` exactly, no iso search ran),
    no transport is needed.  If ``A`` is the identity matrix, the
    transport is a no-op.  Otherwise the loser's known_specs must be
    transported, otherwise specs from the loser's basis would
    contaminate the winner's known_specs with charges that don't
    represent valid negating sequences on the winner's quiver.
    """
    a_key = (len(a.spec), a.spec)
    b_key = (len(b.spec), b.spec)
    a_wins = a_key <= b_key
    winner = a if a_wins else b
    loser = b if a_wins else a

    if iso is None:
        loser_known = loser.known_specs
    else:
        # Convention from `bpskalgebra_iso.Iso` and the call site in
        # `register_with_witness`: `iso` was computed with `entry` as
        # alg_1 and `existing` as alg_2, so `A` transports
        # `entry -> existing`.  In this function `a = existing`,
        # `b = entry`, so `A` transports `b -> a`.
        A_t = tuple(tuple(int(x) for x in row) for row in iso.A)
        n = len(A_t)
        identity = tuple(
            tuple(1 if i == j else 0 for j in range(n)) for i in range(n)
        )
        if A_t == identity:
            # Common case: same exchange matrix, local-moves-only iso.
            loser_known = loser.known_specs
        elif a_wins:
            # Loser is b = entry; transport it forward via A into a's basis.
            loser_known = tuple(
                _apply_A_to_spec(s, A_t) for s in loser.known_specs
            )
        else:
            # Loser is a = existing; transport it back via A^{-1} into b's basis.
            inv = _invert_signed_perm(A_t)
            if inv is None:
                # Non-signed-perm A is rare in std-basis dictionaries.
                # Fall back: drop the loser's known_specs rather than
                # contaminate the winner.
                loser_known = ()
            else:
                loser_known = tuple(
                    _apply_A_to_spec(s, inv) for s in loser.known_specs
                )

    merged_known = tuple(sorted(
        set(winner.known_specs) | set(loser_known),
        key=_spec_sort_key,
    ))
    if merged_known == winner.known_specs:
        return winner
    return QuiverEntry(
        name=winner.name,
        exchange=winner.exchange,
        spec=winner.spec,
        negating_sequence=winner.negating_sequence,
        provenance=winner.provenance,
        known_specs=merged_known,
    )


def _invert_signed_perm(
    A: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...] | None:
    """Invert a signed permutation matrix.  Returns ``None`` if
    ``A`` is not a signed permutation."""
    n = len(A)
    inv = [[0] * n for _ in range(n)]
    for i in range(n):
        # find the unique nonzero column j in row i
        col = -1
        v = 0
        for j in range(n):
            if A[i][j] != 0:
                if col != -1 or A[i][j] not in (1, -1):
                    return None
                col = j
                v = A[i][j]
        if col == -1:
            return None
        # The inverse of a signed perm matrix has inv[j][i] = v
        inv[col][i] = v
    return tuple(tuple(row) for row in inv)
