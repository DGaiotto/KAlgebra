"""Permanent dictionary of BPS quivers with known finite spectrum
generators.

Scope (deliberately minimal): a dictionary entry is

    (exchange_matrix, spec)

for the  active (mutable) nodes only .  No ambient lattice, no Γ
embedding, no frozen-node metadata, no chart identity, no theory
attribution.  An entry is a pure combinatorial BPS quiver with a
known finite  S , modulo permutation of nodes.

This is strictly less refined than the per-chart classes used
elsewhere in the codebase:

    dictionary entry     -- BPS quiver + S factorization (this file)
            ↓ add Γ embedding, F cache, lattice pairing
    CoulombAlgebra       -- single chart of a specific theory
            ↓ add chart graph, intrinsic labels, automorphisms
    CoulombAlgebraTheory -- space of charts for one theory

The dictionary is the common root: many charts and many theories
collapse onto the same quiver entry (e.g.  SUN_Nf(2,1)  and the
``su2nf1``  preset are the same entry here; their embeddings differ
but the quiver and  S  do not).

Three primitives:

* :func:`freeze`  --  RG-flow freezing of a node.  Drops every spec
  factor whose coefficient on the frozen node is nonzero, drops that
  coordinate from each surviving factor, takes the sub-exchange-
  matrix (rows/cols removed).  The frozen node is removed from the
  entry entirely -- the dictionary holds active-node data only.
  Verified by  ``BPSQuiver.verify_spectrum_generator``  before
  return.  Empirical fact (`a probe in the source repository`):
  produces a valid finite S on every tested theory × node.

* :func:`necklace`  --  chart-siblings obtained by mutating at the
  first node of the negating sequence; re-identify mutated charges
  as a fresh standard basis; re-derive the spec.  Stops at
  signature closure.

* :func:`signature`  --  permutation-invariant key under  S_n .
  Brute over permutations for  n ≤ 8 ; weak multiset fallback for
  larger.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from typing import Iterable, Sequence

from bps_quiver_tools import BPSQuiver, CoulombAlgebra, Lattice, PRESETS, sigma

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "bps_quiver_dictionary"


# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------

Vec = tuple[int, ...]


def _spec_sort_key(spec: Sequence[Vec]) -> tuple:
    """Canonical sort key for finite specs:  (length, lex)  primary
    ordering.  Shorter specs always come first; ties broken by the
    standard tuple-of-tuples lex order.  Used by  ``known_specs`` ,
    pentagon-orbit merges, and dictionary register-collision unions
    so that "the K shortest known specs" is always the prefix of
    ``entry.known_specs[:K]`` ."""
    return (len(spec), tuple(tuple(g) for g in spec))


def is_connected(exchange: Sequence[Sequence[int]]) -> bool:
    """Underlying graph of the exchange matrix is connected.

    Two nodes  i  and  j  are adjacent iff  exchange[i][j]  or
    exchange[j][i]  is nonzero.  A 0- or 1-node quiver is trivially
    connected.

    Disconnected quivers represent factorized BPS data -- the
    dictionary stores connected quivers only, and the builders reject
    disconnected outputs (whether they arise as initial seeds or as
    freezing descendants).
    """
    n = len(exchange)
    if n <= 1:
        return True
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in range(n):
            if j not in seen and (exchange[i][j] != 0 or exchange[j][i] != 0):
                seen.add(j)
                stack.append(j)
    return len(seen) == n


@dataclass(frozen=True)
class QuiverEntry:
    """One dictionary entry: a BPS quiver with a known finite S.

    ``exchange[i][j] = <gamma_i, gamma_j>``  for the active nodes.
    ``spec[k]``  is the k-th factor of  S , expressed as an integer
    combination of the active-node basis.

    ``known_specs``  carries every finite spec for this quiver that
    has been observed so far (the seed  ``spec``  always belongs to
    it by invariant).  New specs enter via :func:`pentagon_close`
    (orbit closure) or via the :meth:`BPSQuiverDictionary.register`
    merge on signature collision.  The conjectural pentagon-
    connectedness theorem says one closed orbit covers every finite
    spec; a freeze descendant whose seed sits outside the existing
    orbit is therefore a candidate counterexample worth keeping.
    """
    name: str
    exchange: tuple[tuple[int, ...], ...]
    spec: tuple[Vec, ...]
    negating_sequence: tuple[int, ...] | None = None
    provenance: str = ""
    known_specs: tuple[tuple[Vec, ...], ...] = ()

    def __post_init__(self):
        # Invariant:  known_specs  contains  spec  and is sorted by
        # (length, lex) so that equality on entries is stable across
        # construction paths and  ``known_specs[:K]``  gives the K
        # shortest known factorisations.
        merged = set(self.known_specs)
        merged.add(tuple(self.spec))
        canon = tuple(sorted(merged, key=_spec_sort_key))
        if canon != self.known_specs:
            object.__setattr__(self, "known_specs", canon)

    @property
    def n_nodes(self) -> int:
        return len(self.exchange)

    @property
    def first_factors(self) -> frozenset[Vec]:
        """Distinct first factors across  ``known_specs`` .  Each one
        is the source of one outgoing necklace edge:
        ``multi_necklace_step``  produces (at most) one sibling per
        first factor.  This is the future graph's  out-arrow set.
        """
        return frozenset(s[0] for s in self.known_specs if s)

    @property
    def last_factors(self) -> frozenset[Vec]:
        """Distinct last factors across  ``known_specs`` .  Each one
        is the source of one incoming necklace edge (via the
        inverse-necklace move:  the last factor of  S  is the first
        factor of  S^{-1} ).  This is the future graph's  in-arrow
        set, computable without leaving the entry.
        """
        return frozenset(s[-1] for s in self.known_specs if s)

    def is_connected(self) -> bool:
        return is_connected(self.exchange)

    def _bps_quiver(self) -> BPSQuiver:
        """Materialize as a  BPSQuiver  in a fresh standard basis with
        no frozen nodes.  Used for verification and mutation."""
        n = self.n_nodes
        std_basis = [tuple(1 if k == j else 0 for k in range(n))
                     for j in range(n)]
        return BPSQuiver(
            charges=std_basis,
            frozen=[False] * n,
            exchange_matrix=[list(r) for r in self.exchange],
        )

    def verify(self) -> bool:
        Q = self._bps_quiver()
        ok, _ = Q.verify_spectrum_generator([tuple(g) for g in self.spec])
        return ok

    @classmethod
    def from_algebra(cls, A: CoulombAlgebra, *, name: str = "",
                     provenance: str = "") -> "QuiverEntry":
        """Extract intrinsic  (exchange, spec-in-active-node-basis)
        from a  CoulombAlgebra  and drop the ambient lattice and any
        frozen-node metadata.

        The stored exchange matrix and spec are on the active nodes.
        Frozen-node rows/cols of the parent exchange matrix are
        dropped; spec factors whose expression in the full node basis
        has nonzero coefficient on a frozen node are rejected
        (``ValueError``) -- such a spec involves flavour directions
        the dictionary does not retain.
        """
        all_exchange = [[int(x) for x in row] for row in A.quiver.exchange]
        active = [i for i, f in enumerate(A.frozen) if not f]
        coeffs_full = _coeffs_in_node_basis(
            [tuple(g) for g in A.spec],
            [tuple(g) for g in A.node_charges],
        )
        active_set = set(active)
        spec_active: list[Vec] = []
        for c in coeffs_full:
            for j, v in enumerate(c):
                if j not in active_set and v != 0:
                    raise ValueError(
                        f"spec factor {c} has nonzero coefficient on frozen "
                        f"node {j} in algebra {name!r}; this dictionary "
                        f"only stores active-node data"
                    )
            reduced = tuple(c[j] for j in active)
            if any(x.denominator != 1 for x in reduced):
                raise ValueError(
                    f"spec factor has non-integer active-node coefficient "
                    f"in algebra {name!r}"
                )
            spec_active.append(tuple(int(x) for x in reduced))
        sub_exchange = tuple(
            tuple(all_exchange[i][j] for j in active)
            for i in active
        )
        if not is_connected(sub_exchange):
            raise ValueError(
                f"BPS quiver of algebra {name!r} is disconnected; the "
                f"dictionary stores connected quivers only"
            )
        return cls(
            name=name,
            exchange=sub_exchange,
            spec=tuple(spec_active),
            negating_sequence=(tuple(A.negating_sequence)
                               if getattr(A, "negating_sequence", None) is not None
                               else None),
            provenance=provenance,
        )


def _coeffs_in_node_basis(spec: Sequence[Vec],
                          node_charges: Sequence[Vec]) -> list[tuple[Fraction, ...]]:
    """Express each ``g in spec`` as  sum_j c_j * node_charges[j] ."""
    rank = len(node_charges[0])
    n = len(node_charges)
    A = [[Fraction(node_charges[j][r]) for j in range(n)] for r in range(rank)]
    out: list[tuple[Fraction, ...]] = []
    for g in spec:
        M = [row[:] + [Fraction(g[r])] for r, row in enumerate(A)]
        row = 0
        pivots: list[tuple[int, int]] = []
        for col in range(n):
            piv = None
            for r in range(row, rank):
                if M[r][col] != 0:
                    piv = r
                    break
            if piv is None:
                continue
            M[row], M[piv] = M[piv], M[row]
            for r in range(rank):
                if r != row and M[r][col] != 0:
                    f = M[r][col] / M[row][col]
                    for c in range(col, n + 1):
                        M[r][c] -= f * M[row][c]
            pivots.append((row, col))
            row += 1
        for r in range(row, rank):
            if M[r][-1] != 0:
                raise ValueError(f"spec charge {g} not in span of node charges")
        x = [Fraction(0)] * n
        for pr, pc in pivots:
            x[pc] = M[pr][-1] / M[pr][pc]
        out.append(tuple(x))
    return out


# ---------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------

def find_relabeling(B_target: Sequence[Sequence[int]],
                     B_source: Sequence[Sequence[int]]
                     ) -> tuple[int, ...] | None:
    """Find a node-index permutation  ``perm``  such that

        ``B_target[i][j] == B_source[perm[i]][perm[j]]``

    for all  ``i, j`` .  Returns the permutation as a tuple, or
    ``None``  if the two matrices are not  S_n -equivalent.

    For  n ≤ 8  we brute-force over all  n!  permutations (the same
    enumeration  :func:`signature`  uses).  For larger  n  we don't
    have a complete invariant, so we still try permutations
    consistent with the per-node multiset of brackets, and give up
    if none works.

    Used by the necklace graph: when a multi-necklace step lands at
    a sibling whose intrinsic quiver matches an already-stored
    entry, we use the relabeling to translate the new sibling's
    via /  target_via  / inherited  known_specs  into the stored
    entry's basis.
    """
    import itertools
    n = len(B_target)
    if n != len(B_source):
        return None
    if n == 0:
        return ()
    target = tuple(tuple(row) for row in B_target)
    source = tuple(tuple(row) for row in B_source)
    if n <= 8:
        candidates = itertools.permutations(range(n))
    else:
        # Weak-fallback: filter by per-node bracket multiset.
        def node_key(B, i):
            return tuple(sorted(B[i]))
        target_keys = [node_key(target, i) for i in range(n)]
        source_keys = [node_key(source, i) for i in range(n)]
        # Per-target node, list candidate source nodes with same key.
        candidates_per = []
        for tk in target_keys:
            candidates_per.append(
                [j for j, sk in enumerate(source_keys) if sk == tk]
            )

        def gen():
            from itertools import product
            for combo in product(*candidates_per):
                if len(set(combo)) == n:  # must be a permutation
                    yield combo
        candidates = gen()
    for perm in candidates:
        ok = True
        for i in range(n):
            for j in range(n):
                if target[i][j] != source[perm[i]][perm[j]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return tuple(perm)
    return None


def _relabel_charge(gamma: Sequence[int],
                    perm: Sequence[int]) -> Vec:
    """Relabel a charge under a node-index permutation.

    If  ``perm``  satisfies  ``B_target[i][j] = B_source[perm[i]][perm[j]]`` ,
    a charge written in the source basis as  ``gamma_source = sum_j a_j e_j^source``
    becomes, in the target basis,  ``gamma_target[i] = a_{perm[i]}`` .
    """
    return tuple(gamma[perm[i]] for i in range(len(perm)))


def _relabel_spec(spec: Sequence[Vec],
                  perm: Sequence[int]) -> tuple[Vec, ...]:
    return tuple(_relabel_charge(g, perm) for g in spec)


def _wl_canonical_signature(B, n):
    """1-Weisfeiler-Lehman canonical multiset signature of the
    exchange matrix `B` (n×n).

    Refines node colors iteratively until the color partition
    stabilises; the colors are *tuple-valued* throughout (not
    integer-relabelled), so the final per-node color is an intrinsic
    structural fingerprint independent of node ordering.  Returns
    `tuple(sorted(final_colors))` -- a true `S_n`-invariant.

    Soundness for dictionary bucketing: any `π ∈ S_n` produces an
    iso quiver whose WL colors are the same multiset, so isomorphic
    entries share a signature (no false splits, ever).  The
    signature *may* collide for non-isomorphic but WL-equivalent
    quivers (1-WL incompleteness, mostly on highly regular graphs);
    in that case the in-bucket pairwise iso check correctly rejects
    the false collision, so correctness is preserved -- only a small
    amount of extra iso work is paid.  For BPS quivers (small graphs
    with mostly asymmetric integer-labelled edges) WL is empirically
    near-complete and the speedup vs the brute `n!` approach is
    several orders of magnitude at `n >= 7`.
    """
    # Initial colors: out-edge multiset and in-edge multiset (excluding
    # self-loops).  Diagonal is zero in BPS quivers so this is just the
    # full row/column.
    colors = [
        (
            tuple(sorted(B[i][j] for j in range(n) if j != i)),
            tuple(sorted(B[j][i] for j in range(n) if j != i)),
        )
        for i in range(n)
    ]
    # Refine until the partition stabilises.
    for _ in range(n):  # at most n rounds suffice
        new_colors = [
            (
                colors[i],
                tuple(sorted(
                    (B[i][j], colors[j]) for j in range(n) if j != i
                )),
                tuple(sorted(
                    (B[j][i], colors[j]) for j in range(n) if j != i
                )),
            )
            for i in range(n)
        ]
        if _same_partition_tuples(colors, new_colors):
            break
        colors = new_colors
    return tuple(sorted(colors))


def _same_partition_tuples(a, b):
    """True iff the equivalence partitions induced by `a` and `b`
    (sequences of hashable values) agree."""
    if len(a) != len(b):
        return False
    map_ab: dict = {}
    map_ba: dict = {}
    for x, y in zip(a, b):
        if x in map_ab:
            if map_ab[x] != y:
                return False
        else:
            map_ab[x] = y
        if y in map_ba:
            if map_ba[y] != x:
                return False
        else:
            map_ba[y] = x
    return True


def signature(entry: QuiverEntry) -> tuple:
    """Permutation-invariant signature of the BPS quiver under the
    `S_n` action on nodes.  Keys on the  **exchange matrix alone** :
    isomorphic entries share a signature regardless of their spec.

    The dictionary holds at most one entry per quiver.  If two
    entries with the same signature meet at registration time, the
    shorter-spec one is kept (see :meth:`BPSQuiverDictionary.register`).
    Alternative specs for the same quiver are not retained: once the
    necklace and freezing descendants have been spawned off a given
    chart, the chart itself contributes nothing more to the dictionary.

    Implementation: 1-Weisfeiler-Lehman color refinement on the
    exchange matrix.  The signature is the sorted multiset of
    stable WL colors -- intrinsic to the unordered quiver and
    computable in `O(n^3 log n)` instead of the `O(n! · n^2)` of
    the previous brute-force lex-min canonical form.  WL is a true
    invariant (no false splits between isomorphic quivers); on
    rare highly-symmetric pairs of *non-isomorphic* quivers it can
    over-bucket, in which case the per-bucket iso check correctly
    rejects -- correctness preserved, at the cost of a few extra
    iso evaluations per false-collision bucket.
    """
    n = entry.n_nodes
    B = entry.exchange
    if n == 0:
        return ((),)
    return _wl_canonical_signature(B, n)


# ---------------------------------------------------------------------
# Freezing (RG flow)
# ---------------------------------------------------------------------

def freeze(entry: QuiverEntry, node_index: int, *,
           name: str | None = None) -> QuiverEntry:
    """RG-flow freeze of node  ``node_index`` .

    For each spec in  ``entry.known_specs`` :

      * drop every factor whose coordinate on  ``node_index``  is
        nonzero;
      * drop the  ``node_index``  coordinate from each surviving
        factor.

    Each candidate is verified on the sub-quiver; valid ones become
    the child's  ``known_specs`` .  The primary  ``spec``  is the one
    derived from  ``entry.spec`` , preserving the seed-projection
    semantics.  The sub-exchange matrix has the frozen node's row /
    column removed.

    Raises  ``ValueError``  if the primary spec is invalid or the
    sub-quiver is disconnected.
    """
    n = entry.n_nodes
    if not (0 <= node_index < n):
        raise ValueError(f"node_index {node_index} out of range [0, {n})")
    if n <= 1:
        raise ValueError("cannot freeze: quiver would become empty")
    sub_B = tuple(
        tuple(entry.exchange[r][c] for c in range(n) if c != node_index)
        for r in range(n) if r != node_index
    )
    if not is_connected(sub_B):
        raise ValueError(
            f"freezing node {node_index} of {entry.name!r} produced a "
            f"disconnected quiver"
        )

    def project(spec):
        kept = [g for g in spec if g[node_index] == 0]
        return tuple(
            tuple(g[j] for j in range(n) if j != node_index) for g in kept
        )

    primary = project(entry.spec)

    # Materialise the child quiver in std basis to verify candidate
    # specs from inherited orbit.
    sub_n = n - 1
    std = [tuple(1 if k == j else 0 for k in range(sub_n))
           for j in range(sub_n)]
    child_Q = BPSQuiver(
        charges=std, frozen=[False] * sub_n,
        exchange_matrix=[list(r) for r in sub_B],
    )
    ok_primary, _ = child_Q.verify_spectrum_generator(list(primary))
    if not ok_primary:
        raise ValueError(
            f"freezing node {node_index} of {entry.name!r} produced an "
            f"invalid spectrum generator"
        )

    inherited: set = {primary}
    for s in entry.known_specs:
        if tuple(s) == tuple(entry.spec):
            continue  # already handled
        cand = project(s)
        if not cand:
            continue
        ok, _ = child_Q.verify_spectrum_generator(list(cand))
        if ok:
            inherited.add(cand)
    return QuiverEntry(
        name=(name or f"{entry.name}[freeze {node_index}]"),
        exchange=sub_B,
        spec=primary,
        negating_sequence=None,
        provenance=f"freeze({entry.name!r}, {node_index})",
        known_specs=tuple(sorted(inherited, key=_spec_sort_key)),
    )


# ---------------------------------------------------------------------
# Necklace
# ---------------------------------------------------------------------

def necklace_step(entry: QuiverEntry) -> QuiverEntry:
    """Mutate at the first node of the spec; re-identify the mutated
    charges as a fresh standard basis; re-derive the spec.

    Works on any entry with a valid spec -- falls back to replaying
    the spec on the standard-basis quiver when ``negating_sequence``
    is not stored (e.g. entries produced by :func:`freeze`).

    Returns a fresh  QuiverEntry .
    """
    if len(entry.spec) == 0:
        raise ValueError("entry has empty spec; nothing to rotate")
    parent = entry._bps_quiver()
    # Derive node indices from the spec (works without a stored
    # negating_sequence).
    replay = _spec_to_indices(parent, list(entry.spec))
    if replay is None:
        raise ValueError(f"spec does not apply to quiver of {entry.name!r}")
    indices, _ = replay
    k0 = indices[0]
    mutated = parent.mutate(k0)
    # Re-identify: treat mutated charges as a new standard basis.
    n = entry.n_nodes
    std_basis = [tuple(1 if k == j else 0 for k in range(n))
                 for j in range(n)]
    new_Q = BPSQuiver(
        charges=std_basis,
        frozen=[False] * n,
        exchange_matrix=[list(r) for r in mutated.exchange],
    )
    seq = new_Q.find_negating_sequence()
    if seq is None:
        raise ValueError("mutated quiver has no finite S within BFS budget")
    spec = new_Q.build_spectrum_generator(seq)
    return QuiverEntry(
        name=f"{entry.name}[necklace +1]",
        exchange=tuple(tuple(r) for r in mutated.exchange),
        spec=tuple(tuple(g) for g in spec),
        negating_sequence=tuple(seq),
        provenance=f"necklace_step({entry.name!r})",
    )


def necklace(entry: QuiverEntry, max_steps: int | None = None) -> list[QuiverEntry]:
    from itertools import count
    seen = {signature(entry)}
    out = [entry]
    current = entry
    for step in count(1):
        if max_steps is not None and step > max_steps:
            break
        current = necklace_step(current)
        sig = signature(current)
        if sig in seen:
            break
        seen.add(sig)
        out.append(current)
    return out


# ---------------------------------------------------------------------
# Sigma and uniqueness hypothesis testing
# ---------------------------------------------------------------------

def compute_sigma(entry: QuiverEntry, gamma: Sequence[int]) -> Vec:
    """Apply the tropical  σ  of  ``entry.spec``  to  ``gamma`` .
    Thin wrapper around :func:`bps_quiver_tools.sigma`  using the
    entry's exchange matrix as the lattice pairing and its spec (in
    node-basis) as the sequence of factor charges."""
    L = Lattice([list(r) for r in entry.exchange])
    return sigma(L, list(entry.spec), list(gamma))


def compute_sigma_on_vectors(entry: QuiverEntry,
                              vectors: Sequence[Sequence[int]]) -> list[Vec]:
    L = Lattice([list(r) for r in entry.exchange])
    spec_list = list(entry.spec)
    return [sigma(L, spec_list, list(v)) for v in vectors]


def lattice_ball(n: int, *, radius: int) -> list[Vec]:
    """All  γ ∈ Z^n  with  ``sum(|γ_i|) ≤ radius``  (L1 ball).
    Enumerated by total L1 weight then lex.  Excludes the zero
    vector (σ(0) = 0 trivially)."""
    out: list[Vec] = []
    def gen(prefix: list[int], remaining: int, slots: int):
        if slots == 0:
            if remaining == 0 and any(x != 0 for x in prefix):
                out.append(tuple(prefix))
            return
        for v in range(-remaining, remaining + 1):
            gen(prefix + [v], remaining - abs(v), slots - 1)
    for w in range(1, radius + 1):
        def gen_exact(prefix: list[int], remaining: int, slots: int):
            if slots == 0:
                if remaining == 0:
                    out.append(tuple(prefix))
                return
            for v in range(-remaining, remaining + 1):
                gen_exact(prefix + [v], remaining - abs(v), slots - 1)
        gen_exact([], w, n)
    return out


def check_uniqueness_hypothesis(entry: QuiverEntry, *,
                                 radius: int = 3,
                                 retry_bfs: bool = True) -> dict:
    """Verification hook for the one-S-per-quiver hypothesis.

    Compute  σ  on the L1-ball of radius ``radius``  from the stored
    spec.  If ``retry_bfs=True`` , re-run  ``find_negating_sequence``
    on the quiver to get an independently-discovered spec; compute
    its  σ  on the same ball; compare pointwise.

    Returns a dict::

        {
            "match": bool,
            "first_disagreement": Vec or None,
            "spec_1": tuple,       # stored spec
            "spec_2": tuple or None,  # independent spec, if retried
            "n_vectors": int,
            "sigma_1": list[Vec],
            "sigma_2": list[Vec] or None,
        }

    Intended for occasional / opt-in use, not every build.  σ is
    piecewise-linear (tropical), so checking only on the standard
    basis is insufficient -- the L1-ball covers enough neighbouring
    pieces to catch genuine discrepancies.
    """
    n = entry.n_nodes
    vectors = lattice_ball(n, radius=radius)
    sigma_1 = compute_sigma_on_vectors(entry, vectors)

    spec_2: tuple | None = None
    sigma_2: list[Vec] | None = None
    if retry_bfs:
        Q = entry._bps_quiver()
        seq = Q.find_negating_sequence()
        if seq is not None:
            alt_spec = tuple(
                tuple(g) for g in Q.build_spectrum_generator(seq)
            )
            if alt_spec != entry.spec:
                spec_2 = alt_spec
                alt_entry = QuiverEntry(
                    name=entry.name + "[alt]",
                    exchange=entry.exchange,
                    spec=alt_spec,
                    negating_sequence=tuple(seq),
                )
                sigma_2 = compute_sigma_on_vectors(alt_entry, vectors)

    match = True
    first_bad: Vec | None = None
    if sigma_2 is not None:
        for v, s1, s2 in zip(vectors, sigma_1, sigma_2):
            if s1 != s2:
                match = False
                first_bad = v
                break
    return {
        "match": match,
        "first_disagreement": first_bad,
        "spec_1": entry.spec,
        "spec_2": spec_2,
        "n_vectors": len(vectors),
        "sigma_1": sigma_1,
        "sigma_2": sigma_2,
    }


# ---------------------------------------------------------------------
# Spec shortening  (head/tail-constrained BFS over sub-ranges)
# ---------------------------------------------------------------------

def _spec_to_indices(Q: BPSQuiver,
                     spec: Sequence[Vec]) -> tuple[list[int], BPSQuiver] | None:
    """Replay a spec (list of charge tuples) on  Q  and return the
    equivalent list of mutation indices + the final quiver state."""
    current = Q
    indices: list[int] = []
    for gk in spec:
        gk_t = tuple(gk)
        idx = None
        for i, c in enumerate(current.charges):
            if not current.frozen[i] and tuple(c) == gk_t:
                idx = i
                break
        if idx is None:
            return None
        indices.append(idx)
        current = current.mutate(idx)
    return indices, current


def _indices_to_spec(Q: BPSQuiver, indices: Sequence[int]) -> list[Vec]:
    """Replay a list of mutation indices on  Q  and return the charges
    that were mutated at each step."""
    current = Q
    out: list[Vec] = []
    for k in indices:
        out.append(tuple(current.charges[k]))
        current = current.mutate(k)
    return out


def try_shorten_range(entry: QuiverEntry, i: int, j: int) -> QuiverEntry | None:
    """Try to replace ``entry.spec[i..j]``  (inclusive) with a
    strictly shorter middle found by head/tail-constrained
    bidirectional BFS.

    Returns a new :class:`QuiverEntry`  with shorter spec, or ``None``
    if no shortening of that sub-range exists within the budget.
    Uses ``allow_permutation=True``  and  ``use_edge_mult=True`` .
    """
    from bps_quiver_tools import find_mutation_path
    N = len(entry.spec)
    if not (0 <= i <= j < N):
        raise ValueError(f"bad range ({i},{j}) for N={N}")
    Q0 = entry._bps_quiver()
    full = _spec_to_indices(Q0, list(entry.spec))
    if full is None:
        return None
    full_indices, Q_end = full
    head_indices = full_indices[:i]
    tail_indices = full_indices[j + 1:]
    orig_mid = j - i + 1
    path = find_mutation_path(
        Q0, Q_end,
        head=head_indices, tail=tail_indices,
        max_depth=N - 1,
        allow_permutation=True,
        use_edge_mult=True,
    )
    if path is None:
        return None
    new_mid = len(path) - len(head_indices) - len(tail_indices)
    if new_mid >= orig_mid:
        return None
    new_spec = _indices_to_spec(Q0, list(path))
    out = QuiverEntry(
        name=f"{entry.name}[short {i}:{j+1}]",
        exchange=entry.exchange,
        spec=tuple(tuple(g) for g in new_spec),
        negating_sequence=tuple(path),
        provenance=f"shorten({entry.name!r}, {i}, {j})",
    )
    return out if out.verify() else None


def alternative_subseq_factorizations(
    entry: QuiverEntry,
    *,
    max_range_len: int = 5,
    require_strictly_different: bool = True,
    same_length_only: bool = False,
) -> list[tuple[tuple[int, int], tuple[Vec, ...]]]:
    """Find alternative factorizations of  S  by replacing a
    contiguous subsequence  ``entry.spec[i..j]``  with an
    alternative middle discovered via head/tail-constrained
    bidirectional BFS.

    For every contiguous range  (i, j)  of length  2..max_range_len ,
    runs :func:`bps_quiver_tools.find_mutation_path`  with the
    untouched  ``head = spec[:i]``  and  ``tail = spec[j+1:]``  to
    look for an alternative middle that lands in the same end state.
    Each such hit yields a fresh factorization of  S  (same operator,
    different ordering / pentagon-rewrite of the middle).

    Returns a list of  ``((i, j), new_spec)``  tuples.

    Parameters
    ----------
    max_range_len
        Cap on the range length  L = j - i + 1  to scan.  Wider
        ranges expose more pentagon rewrites but cost more.
    require_strictly_different
        If True (default), only return alternatives whose new spec
        differs from the original (in some position).  If False,
        identical-spec replacements are also returned (e.g. when the
        BFS rediscovers the same middle).
    same_length_only
        If True, only accept alternatives whose middle has the same
        length as the original.  Default False (shorter and longer
        middles are both allowed).
    """
    from bps_quiver_tools import find_mutation_path
    N = len(entry.spec)
    if N == 0:
        return []
    Q0 = entry._bps_quiver()
    full = _spec_to_indices(Q0, list(entry.spec))
    if full is None:
        return []
    full_indices, Q_end = full
    seen_specs: set[tuple[Vec, ...]] = {tuple(entry.spec)}
    out: list[tuple[tuple[int, int], tuple[Vec, ...]]] = []
    for L in range(2, min(max_range_len, N) + 1):
        for i in range(0, N - L + 1):
            j = i + L - 1
            head_indices = full_indices[:i]
            tail_indices = full_indices[j + 1:]
            orig_mid = j - i + 1
            try:
                path = find_mutation_path(
                    Q0, Q_end,
                    head=head_indices, tail=tail_indices,
                    max_depth=N + 2,   # allow some growth, capped
                    allow_permutation=True,
                    use_edge_mult=True,
                )
            except Exception:
                continue
            if path is None:
                continue
            new_mid = len(path) - len(head_indices) - len(tail_indices)
            if same_length_only and new_mid != orig_mid:
                continue
            new_spec = tuple(
                tuple(g) for g in _indices_to_spec(Q0, list(path))
            )
            if require_strictly_different and new_spec == tuple(entry.spec):
                continue
            if new_spec in seen_specs:
                continue
            # Sanity check: the candidate must be a valid negating
            # sequence on the same quiver.
            cand = QuiverEntry(
                name=f"{entry.name}[alt subseq {i}:{j+1}]",
                exchange=entry.exchange,
                spec=new_spec,
                negating_sequence=tuple(path),
            )
            if not cand.verify():
                continue
            seen_specs.add(new_spec)
            out.append(((i, j), new_spec))
    return out


def shorten_spec(entry: QuiverEntry, *,
                 max_range_len: int = 5,
                 max_iterations: int = 20) -> QuiverEntry:
    """Iteratively shorten  ``entry``  by sweeping all contiguous
    sub-ranges of length  ``2..max_range_len`` , taking the first win,
    and repeating until a full sweep finds no shortening.

    Empirical finding (`a probe in the source repository`): wins cluster
    at boundaries between structural blocks (matter↔gauge, bifund↔
    pure) and almost always come from ranges of length 3–5 saving 1
    or 2.  Wider sweeps rarely add new savings and cost more.

    Returns a  QuiverEntry  with the shortest spec found (possibly
    unchanged).
    """
    current = entry
    for _ in range(max_iterations):
        improved = False
        N = len(current.spec)
        for L in range(2, min(max_range_len, N) + 1):
            for i in range(0, N - L + 1):
                j = i + L - 1
                try:
                    better = try_shorten_range(current, i, j)
                except Exception:
                    better = None
                if better is not None:
                    current = better
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return current


# ---------------------------------------------------------------------
# Alternative factorizations + alternative-first-factor necklace mining
# ---------------------------------------------------------------------

def _ambient_bracket(B: tuple[tuple[int, ...], ...],
                     g1: Sequence[int], g2: Sequence[int]) -> int:
    n = len(B)
    return sum(g1[i] * B[i][j] * g2[j]
               for i in range(n) for j in range(n))


def alternative_first_factors(entry: QuiverEntry) -> list[tuple[int, tuple[Vec, ...]]]:
    """Yield alternative factorizations of  S  that share the same
    operator but put a different factor first.

    **Scope.** Only alternatives reachable from the stored
    factorization via chained *adjacent commuting swaps* are
    returned.  Two consecutive factors commute iff their charges
    pair to zero under the ambient (B) pairing; factor  i  can be
    moved to position 0 iff it commutes with every earlier  g_j .
    This covers all first-factor choices reachable without touching
    the E_q-length (pentagon-identity rewrites that change length
    are excluded).

    Returns a list of  (i, swapped_spec)  pairs with  i > 0 .  The
    swapped_spec is ``[g_i, g_0, ..., g_{i-1}, g_{i+1}, ..., g_{N-1}]`` .

    **Limitation.** The set of all operator-equivalent factorizations
    is the set of linear extensions of the non-commutation partial
    order, which is combinatorially large.  We only sample the
    first-factor slice.  Further coverage comes implicitly: once a
    sibling is registered and closed over (necklace, alt, freezing),
    its own first-factor alternatives are explored in a later pass.
    The dictionary therefore converges toward -- but does not in
    general attain -- full closure under arbitrary reorderings.
    """
    out: list[tuple[int, tuple[Vec, ...]]] = []
    spec = entry.spec
    B = entry.exchange
    for i in range(1, len(spec)):
        # Check  g_i  commutes with every earlier  g_j .
        ok = all(_ambient_bracket(B, spec[i], spec[j]) == 0
                 for j in range(i))
        if not ok:
            continue
        swapped = (spec[i],) + tuple(spec[j] for j in range(len(spec))
                                      if j != i)
        out.append((i, swapped))
    return out


def alternative_last_factors(entry: QuiverEntry) -> list[tuple[int, tuple[Vec, ...]]]:
    """Mirror of :func:`alternative_first_factors`  for the tail:
    yield  (i, swapped_spec)  pairs where factor  g_i  can be moved
    to the LAST position by adjacent commuting swaps with every
    later factor  g_j  ( j > i ).  Two factors commute iff they pair
    to zero under the ambient  B .

    Useful in combination with an *inverse* necklace step
    (``surface_catalog.paired_inverse_necklace_step`` ): after moving
    factor  g_i  to the tail, inverse-necklacing pulls  -g_i  to the
    head and flips the triangulation at  g_i 's node.  Pairs with
    head-side  alternative_first_factors  for symmetric coverage of
    head / tail reorganizations of  S .
    """
    out: list[tuple[int, tuple[Vec, ...]]] = []
    spec = entry.spec
    B = entry.exchange
    N = len(spec)
    for i in range(N - 1):
        ok = all(_ambient_bracket(B, spec[i], spec[j]) == 0
                 for j in range(i + 1, N))
        if not ok:
            continue
        swapped = tuple(spec[j] for j in range(N) if j != i) + (spec[i],)
        out.append((i, swapped))
    return out


def mine_alt_necklaces(entry: QuiverEntry) -> list[QuiverEntry]:
    """Legacy entry point: kept for backward compatibility, delegates
    to :func:`node_targeted_necklaces`  with ``directions=('head',)``.

    The original implementation generated alts via
    :func:`alternative_first_factors`  (commute-only) and then ran
    ``necklace_step``  -- which in turn called
    :meth:`BPSQuiver.find_negating_sequence`  (BFS).  The new function
    iterates *node by node* using :func:`bps_quiver_tools.alt_spec_with_head`
    (commute + pentagon expand + pentagon collapse) and does the
    necklace step at the charge level (no BFS), so it produces
    strictly more siblings on strictly less work.
    """
    return node_targeted_necklaces(entry, directions=("head",))


def node_targeted_necklaces(
    entry: QuiverEntry,
    *,
    directions: Sequence[str] = ("head", "tail"),
) -> list[QuiverEntry]:
    """Targeted node-by-node necklace generator.

    For each unfrozen node  k  of  ``entry``  and each requested
    direction:

    * ``'head'``  --  use :func:`bps_quiver_tools.alt_spec_with_head`
      to bubble the standard basis vector  e_k  to the head of the
      spec via local moves (commute / pentagon expand / collapse),
      then necklace-shift (drop head, append  -e_k  at tail) and
      tropically translate to the post-mutation std basis.  Equivalent
      to a forward mutation at node  k  with no BFS.
    * ``'tail'``  --  mirror via :func:`alt_spec_with_tail` , giving an
      inverse necklace step (drop tail, prepend  -e_k  at head,
      reverse-tropical translate, reverse-mutate exchange matrix).

    Replacement for :func:`mine_alt_necklaces` : strictly more siblings
    (the commuting-swap slice is the  ``<a, b> == 0``  branch of the
    bubble), and the necklace step itself bypasses
    ``find_negating_sequence`` -- the principal cost driver at large
    rank.

    Skips nodes blocked by a pairing-≥ 2 wall or pairing-1 without a
    matching collapse triple (those signal genuine transport-only
    chambers).  Verified siblings only -- unverified candidates are
    dropped.
    """
    from bps_quiver_tools import (
        alt_spec_with_head as _alt_head,
        alt_spec_with_tail as _alt_tail,
    )
    from a1_from_topology import (
        _tropical_mutate_charge,
        _tropical_reverse_mutate_charge,
    )

    n = entry.n_nodes
    B = [list(row) for row in entry.exchange]
    # The dictionary normalises every entry to standard basis, so the
    # ambient pairing equals the exchange matrix.  ``alt_spec_with_*``
    # needs ``ambient_pairing``  set on the quiver to evaluate
    # ``<a, b>``  on charges, so build it via ``from_pairing`` .
    std_basis = [tuple(1 if k == j else 0 for k in range(n))
                 for j in range(n)]
    Q = BPSQuiver.from_pairing(std_basis, B)
    siblings: list[QuiverEntry] = []
    seen: set = set()

    for k in range(n):
        # Skip frozen (currently dictionary entries don't carry frozen
        # flags, but be safe).
        if Q.frozen[k]:
            continue

        for direction in directions:
            if direction == "head":
                alt = _alt_head(Q, list(entry.spec), k)
                if alt is None:
                    continue
                head = tuple(alt[0])
                neg_head = tuple(-x for x in head)
                new_spec_old = list(alt[1:]) + [neg_head]
                new_spec = tuple(
                    _tropical_mutate_charge(tuple(g), B, k)
                    for g in new_spec_old
                )
                Qm = Q.mutate(k)
            elif direction == "tail":
                alt = _alt_tail(Q, list(entry.spec), k)
                if alt is None:
                    continue
                tail = tuple(alt[-1])
                neg_tail = tuple(-x for x in tail)
                new_spec_old = [neg_tail] + list(alt[:-1])
                new_spec = tuple(
                    _tropical_reverse_mutate_charge(tuple(g), B, k)
                    for g in new_spec_old
                )
                Qm = Q.reverse_mutate(k)
            else:
                raise ValueError(
                    f"unknown direction {direction!r} -- "
                    f"expected 'head' or 'tail'"
                )

            new_B = tuple(tuple(int(x) for x in row) for row in Qm.exchange)
            sib = QuiverEntry(
                name=f"{entry.name}[node-{direction} {k}]",
                exchange=new_B,
                spec=new_spec,
                negating_sequence=None,
                provenance=(
                    f"node_targeted_necklaces({entry.name!r}, "
                    f"k={k}, dir={direction!r})"
                ),
            )
            if not sib.verify():
                continue
            sig = signature(sib)
            if sig in seen:
                continue
            seen.add(sig)
            siblings.append(sib)
    return siblings


def _transport_to_sibling(gamma: Sequence[int],
                            k0: int,
                            parent_B: Sequence[Sequence[int]]) -> Vec:
    """Express a parent-basis charge  ``gamma``  in the sibling
    basis after necklace-mutating at parent node  ``k0`` .

    Sibling node charges, in parent's basis, are
    ``c_{k0}' = -e_{k0}``  and ``c_j' = e_j + max(B[j,k0], 0) e_{k0}``
    for  ``j != k0`` .  Solving  ``M b = gamma``  for the matrix
    whose columns are the  ``c_i'``  yields:

        b_i      = gamma_i                                  for i != k0
        b_{k0}   = -gamma_{k0} + sum_{j!=k0} max(B[j,k0],0) gamma_j
    """
    n = len(gamma)
    b = list(gamma)
    b[k0] = -gamma[k0]
    for j in range(n):
        if j == k0:
            continue
        b[k0] += max(parent_B[j][k0], 0) * gamma[j]
    return tuple(b)


def multi_necklace_edges(entry: QuiverEntry
                          ) -> list[tuple[QuiverEntry, Vec, Vec]]:
    """For every distinct first-factor across  ``entry.known_specs`` ,
    perform one necklace mutation and return  (sibling, via,
    target_via)  triples -- one per first-factor group, *not*
    deduplicated by signature.

    ``via``  is the parent-basis charge that drove the mutation
    (the first factor of the group's specs).  ``target_via``  is
    the sibling-basis charge that, in an inverse-necklace step,
    sends the sibling back to the parent: in the standard re-
    identification, ``target_via = e_{k_0}``  where  ``k_0``  is
    the parent-node index of  ``via`` .

    Each sibling inherits, in its own  ``known_specs`` , the
    rotation of every parent orbit spec sharing  ``via`` .

    Inheritance scheme (per group):

      * The rotated spec in *parent's* basis is
        ``[γ_2, γ_3, ..., γ_N, -γ_1]`` -- MGS rotation.
      * Each charge is transported to *sibling's* basis via
        :func:`_transport_to_sibling` .
      * Each candidate is verified on the sibling's quiver.  Only
        valid ones are stored.
    """
    n = entry.n_nodes
    parent_Q = entry._bps_quiver()
    std_basis = [tuple(1 if k == j else 0 for k in range(n))
                 for j in range(n)]

    # Group parent orbit specs by first factor.
    by_first: dict[Vec, list[tuple[Vec, ...]]] = {}
    for s in entry.known_specs:
        if not s:
            continue
        by_first.setdefault(s[0], []).append(s)

    out: list[tuple[QuiverEntry, Vec, Vec]] = []
    for first_factor, group in by_first.items():
        replay = _spec_to_indices(parent_Q, list(group[0]))
        if replay is None:
            continue
        indices0, _ = replay
        k0 = indices0[0]

        Q1 = parent_Q.mutate(k0)
        new_Q = BPSQuiver(
            charges=std_basis,
            frozen=[False] * n,
            exchange_matrix=[list(r) for r in Q1.exchange],
        )

        inherited: list[tuple[Vec, ...]] = []
        neg_first = tuple(-x for x in first_factor)
        for s in group:
            rotated_parent = list(s[1:]) + [neg_first]
            sib_spec = tuple(
                _transport_to_sibling(g, k0, entry.exchange)
                for g in rotated_parent
            )
            ok, _info = new_Q.verify_spectrum_generator(list(sib_spec))
            if ok:
                inherited.append(sib_spec)

        seq = new_Q.find_negating_sequence()
        if seq is not None:
            canonical_spec = tuple(
                tuple(g) for g in new_Q.build_spectrum_generator(seq)
            )
            primary = canonical_spec
            seq_tup = tuple(seq)
        elif inherited:
            primary = inherited[0]
            seq_tup = None
        else:
            continue

        all_specs = set(inherited)
        all_specs.add(primary)
        merged_specs = tuple(sorted(all_specs, key=_spec_sort_key))

        sib = QuiverEntry(
            name=entry.name,
            exchange=tuple(tuple(r) for r in new_Q.exchange),
            spec=primary,
            negating_sequence=seq_tup,
            provenance=f"multi_necklace({entry.name!r})",
            known_specs=merged_specs,
        )
        target_via = std_basis[k0]
        out.append((sib, first_factor, target_via))

    return out


def multi_necklace_step(entry: QuiverEntry) -> list[QuiverEntry]:
    """Backward-compatible wrapper around :func:`multi_necklace_edges` :
    returns just the sibling entries, deduplicated by signature.

    On signature collision we keep the first sibling seen for that
    signature; if the colliding sibling has the **same** exchange
    matrix, its  ``known_specs``  are merged into the kept entry.
    Different exchange matrices (same signature up to permutation)
    are treated as distinct -- spec coordinates are not basis-
    invariant, so cross-exchange merges would be unsound.
    """
    siblings_by_sig: dict[tuple, QuiverEntry] = {}
    for sib, _via, _target_via in multi_necklace_edges(entry):
        sig = signature(sib)
        existing = siblings_by_sig.get(sig)
        if existing is None:
            siblings_by_sig[sig] = sib
        elif existing.exchange == sib.exchange:
            merged = tuple(sorted(
                set(existing.known_specs) | set(sib.known_specs),
                key=_spec_sort_key,
            ))
            siblings_by_sig[sig] = QuiverEntry(
                name=existing.name,
                exchange=existing.exchange,
                spec=existing.spec,
                negating_sequence=existing.negating_sequence,
                provenance=existing.provenance,
                known_specs=merged,
            )
        # else: drop the new sibling -- different exchange, can't merge.
    return list(siblings_by_sig.values())


# ---------------------------------------------------------------------
# Pentagon orbit
# ---------------------------------------------------------------------
#
# The conjectural pentagon-connectedness theorem (Keller for finite-
# type / acyclic; conjectured in general): any two finite spectrum
# generators for the same BPS quiver are connected by elementary
# moves on adjacent factors driven by the pairing  ⟨γ_i, γ_{i+1}⟩ :
#
#   ⟨a, b⟩ = 0   commuting swap        (a, b) ↔ (b, a)
#   ⟨a, b⟩ = 1   pentagon expand 2→3   (a, b) → (b, a+b, a)
#                pentagon contract 3→2 (b, a+b, a) → (a, b)
#
# Asymmetric:  ⟨a, b⟩ = -1  is **not** a forward move — its expand
# would put the two factors in the "wrong" pentagon order.  The
# corresponding contract still triggers when the pattern shows up
# inside a longer spec.
#
# This is the smallest move set that is conjecturally complete; if a
# freeze descendant produces a spec disconnected from the pentagon
# orbit of every existing entry for the same quiver, that descendant
# is a candidate counterexample.

def pentagon_neighbours(spec: Sequence[Vec],
                         exchange: Sequence[Sequence[int]]
                         ) -> list[tuple[Vec, ...]]:
    """All specs reachable from  ``spec``  by one elementary move on
    the pairing  ``exchange`` .  Three move classes:

    * commuting swap at gap  i  when  ⟨γ_i, γ_{i+1}⟩ = 0
    * pentagon expand at gap  i  when  ⟨γ_i, γ_{i+1}⟩ = +1 :
      ``(a, b) → (b, a+b, a)``
    * pentagon contract at triple  (i, i+1, i+2)  when
      ``γ_{i+1} = γ_i + γ_{i+2}``  and  ``⟨γ_{i+2}, γ_i⟩ = +1`` :
      ``(b, a+b, a) → (a, b)``

    Output is deduplicated; order is gap-first, then triple, both
    left-to-right.
    """
    spec_t = tuple(tuple(g) for g in spec)
    N = len(spec_t)
    out: list[tuple[Vec, ...]] = []
    seen: set[tuple[Vec, ...]] = set()

    def emit(new_spec: tuple[Vec, ...]) -> None:
        if new_spec not in seen:
            seen.add(new_spec)
            out.append(new_spec)

    # Gaps: swap (pairing 0) and expand (pairing +1).
    for i in range(N - 1):
        a, b = spec_t[i], spec_t[i + 1]
        p = _ambient_bracket(exchange, a, b)
        if p == 0:
            emit(spec_t[:i] + (b, a) + spec_t[i + 2:])
        elif p == 1:
            ab = tuple(x + y for x, y in zip(a, b))
            emit(spec_t[:i] + (b, ab, a) + spec_t[i + 2:])

    # Triples: contract  (b, a+b, a) → (a, b)  when  ⟨a, b⟩ = +1 .
    for i in range(N - 2):
        b, m, a = spec_t[i], spec_t[i + 1], spec_t[i + 2]
        if m != tuple(x + y for x, y in zip(a, b)):
            continue
        if _ambient_bracket(exchange, a, b) != 1:
            continue
        emit(spec_t[:i] + (a, b) + spec_t[i + 3:])

    return out


def _orbit_bfs(seeds: Sequence[Sequence[Vec]],
               exchange: Sequence[Sequence[int]],
               *,
               max_states: int | None,
               max_length: int | None,
               same_length_only: bool,
               ) -> list[tuple[Vec, ...]]:
    """Length-priority BFS over the pentagon-move graph.

    Pops specs in  (length, lex)  order and returns them in the same
    order.  Stops gracefully when  ``len(out) >= max_states`` , so
    the result is the  ``max_states``  shortest specs *reachable
    along length-priority paths from any seed*.

    Caveat: a strict "K shortest specs in the orbit" enumeration
    would require running to completion and post-sorting, since a
    short spec can occasionally only be reached via a longer detour.
    For pentagon orbits this corner case is rare in practice; the
    length-priority traversal collects the canonical-length specs
    first and degrades gracefully on truncation.
    """
    import heapq
    seen: set[tuple[Vec, ...]] = set()
    heap: list[tuple[tuple, tuple[Vec, ...]]] = []
    seed_lens: set[int] = set()
    for s in seeds:
        t = tuple(tuple(g) for g in s)
        if t in seen:
            continue
        seen.add(t)
        seed_lens.add(len(t))
        heapq.heappush(heap, (_spec_sort_key(t), t))
    out: list[tuple[Vec, ...]] = []
    while heap:
        if max_states is not None and len(out) >= max_states:
            break
        _, s = heapq.heappop(heap)
        out.append(s)
        for nbr in pentagon_neighbours(s, exchange):
            if nbr in seen:
                continue
            if same_length_only and len(nbr) not in seed_lens:
                continue
            if max_length is not None and len(nbr) > max_length:
                continue
            seen.add(nbr)
            heapq.heappush(heap, (_spec_sort_key(nbr), nbr))
    return out


def pentagon_close(entry: QuiverEntry, *,
                    max_states: int | None = 10_000,
                    max_length: int | None = None,
                    seeds: Sequence[Sequence[Vec]] | None = None,
                    ) -> QuiverEntry:
    """Return a copy of  ``entry``  with  ``known_specs``  closed
    under :func:`pentagon_neighbours` .

    Closure is a length-priority BFS from every spec in
    ``entry.known_specs``  (or  ``seeds``  if supplied), unioned
    together.  Stops gracefully when the number of collected specs
    reaches  ``max_states`` , returning the shortest specs found so
    far.  Pass  ``max_length``  to cap the longest spec considered.

    Idempotent within a fixed cap: applying twice yields the same
    orbit (the second call rediscovers the same set in the same
    length-priority order).
    """
    seed_specs = (seeds if seeds is not None
                  else entry.known_specs or (entry.spec,))
    discovered = _orbit_bfs(
        seed_specs, entry.exchange,
        max_states=max_states, max_length=max_length,
        same_length_only=False,
    )
    merged = tuple(discovered)  # already in (length, lex) order
    if merged == entry.known_specs:
        return entry
    return QuiverEntry(
        name=entry.name,
        exchange=entry.exchange,
        spec=entry.spec,
        negating_sequence=entry.negating_sequence,
        provenance=entry.provenance,
        known_specs=merged,
    )


def pentagon_close_variety(entry: QuiverEntry, *,
                            max_states: int | None = 1_000,
                            max_length: int | None = None,
                            plateau: int = 50,
                            seeds: Sequence[Sequence[Vec]] | None = None,
                            ) -> QuiverEntry:
    """Variety-prioritised  :func:`pentagon_close` .

    Stops the orbit BFS as soon as it stops discovering new first or
    last factors:  if  ``plateau``  consecutive specs are popped
    without enlarging  ``first_factors``  or  ``last_factors`` , the
    BFS terminates.  This keeps  known_specs  small while preserving
    the boundary-mutation diversity that drives the necklace graph's
    edge population.

    The hard cap  ``max_states``  is still honoured (graceful stop).
    For graph-building this is dramatically cheaper than full
    pentagon closure; for palindromic search you'll want the full
    closure instead.
    """
    import heapq
    seed_specs = (seeds if seeds is not None
                  else entry.known_specs or (entry.spec,))
    seen: set[tuple[Vec, ...]] = set()
    heap: list[tuple[tuple, tuple[Vec, ...]]] = []
    for s in seed_specs:
        t = tuple(tuple(g) for g in s)
        if t in seen:
            continue
        seen.add(t)
        heapq.heappush(heap, (_spec_sort_key(t), t))
    out: list[tuple[Vec, ...]] = []
    first_factors: set[Vec] = set()
    last_factors: set[Vec] = set()
    plateau_count = 0
    while heap:
        if max_states is not None and len(out) >= max_states:
            break
        _, s = heapq.heappop(heap)
        out.append(s)
        ff = s[0] if s else None
        lf = s[-1] if s else None
        new = False
        if ff is not None and ff not in first_factors:
            first_factors.add(ff)
            new = True
        if lf is not None and lf not in last_factors:
            last_factors.add(lf)
            new = True
        if new:
            plateau_count = 0
        else:
            plateau_count += 1
            if plateau_count >= plateau:
                break
        for nbr in pentagon_neighbours(s, entry.exchange):
            if nbr in seen:
                continue
            if max_length is not None and len(nbr) > max_length:
                continue
            seen.add(nbr)
            heapq.heappush(heap, (_spec_sort_key(nbr), nbr))
    merged = tuple(out)  # already in (length, lex) order
    if merged == entry.known_specs:
        return entry
    return QuiverEntry(
        name=entry.name,
        exchange=entry.exchange,
        spec=entry.spec,
        negating_sequence=entry.negating_sequence,
        provenance=entry.provenance,
        known_specs=merged,
    )


def pentagon_orbit(spec: Sequence[Vec],
                    exchange: Sequence[Sequence[int]],
                    *,
                    max_states: int | None = 10_000,
                    max_length: int | None = None,
                    same_length_only: bool = False,
                    ) -> list[tuple[Vec, ...]]:
    """Length-priority BFS of the orbit of  ``spec``  under
    :func:`pentagon_neighbours` .

    Returns the discovered specs in  (length, lex)  order (so the
    seed comes first, then any other length-N specs, then length-N+1
    specs, etc.).  Stops gracefully at  ``max_states``  -- a partial
    orbit is returned, not an exception, and the partial is the
    ``max_states``  shortest specs reachable along length-priority
    paths.

    Use  ``same_length_only=True``  to restrict to specs whose
    length equals the seed's (commuting-swap orbit), or
    ``max_length``  to cap the longest spec considered.
    """
    return _orbit_bfs(
        [spec], exchange,
        max_states=max_states, max_length=max_length,
        same_length_only=same_length_only,
    )


# ---------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class NecklaceEdge:
    """A directed necklace-mutation edge between two dictionary
    entries.

    ``source``  and  ``target``  are signature keys.
    ``via``  is the first-factor charge (in  source 's basis) that
    drives the necklace step from  source  to  target .
    ``target_via``  is the corresponding first factor for the
    inverse-necklace edge  target → source , expressed in  target 's
    basis.  Storing both directions lets a graph walk navigate
    forward (multi-necklace) or backward (inverse necklace) without
    re-running the mutation.
    """
    source: tuple
    target: tuple
    via: Vec
    target_via: Vec


class BPSQuiverDictionary:
    """In-memory map from  S_n -signature to  QuiverEntry , plus a
    directed multigraph of necklace-mutation edges (one edge per
    distinct first factor in each entry's  ``known_specs`` ).

    The graph is populated on demand by
    :meth:`close_under_multi_necklace`  -- the dictionary itself can
    operate in either pure-dict or graph-aware mode.
    """

    def __init__(self):
        self._entries: dict[tuple, QuiverEntry] = {}
        self._edges_out: dict[tuple, dict[Vec, NecklaceEdge]] = {}
        self._edges_in: dict[tuple, list[NecklaceEdge]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, item) -> bool:
        if isinstance(item, QuiverEntry):
            return signature(item) in self._entries
        return item in self._entries

    def __iter__(self):
        return iter(self._entries.values())

    def register(self, entry: QuiverEntry, *, replace: bool = False) -> tuple:
        """Register an entry.  The signature is keyed by the quiver
        (exchange matrix) alone, so at most one entry per quiver is
        retained.  On collision:

        * ``replace=True`` : the new entry overwrites unconditionally.
        * ``replace=False``  (default): the entry with the **shorter
          spec** is kept (the shortest-known factorization of  S
          wins).  Ties broken by lexicographic spec order for
          determinism.  The merged entry retains the **union** of
          the two  ``known_specs``  fields, so alternative specs
          from either source are preserved across collisions.

        Disconnected entries are silently skipped: the dictionary
        stores connected BPS quivers only.  The signature is still
        returned (so callers iterating the result aren't surprised by
        ``None`` ), but the entry is not stored.
        """
        sig = signature(entry)
        if not entry.is_connected():
            return sig
        existing = self._entries.get(sig)
        if existing is None or replace:
            self._entries[sig] = entry
            return sig
        # Keep the shorter spec (or the lexicographically smaller one
        # when tied).
        cur_key = (len(existing.spec), existing.spec)
        new_key = (len(entry.spec), entry.spec)
        winner = entry if new_key < cur_key else existing
        # Merge known_specs only when exchanges match exactly.  Same
        # signature with different exchange = same quiver up to node
        # permutation, but spec charges are coordinates, not labels,
        # so a spec valid on one exchange does not transfer to the
        # other.
        if existing.exchange == entry.exchange:
            merged = tuple(sorted(
                set(existing.known_specs) | set(entry.known_specs),
                key=_spec_sort_key,
            ))
            if merged != winner.known_specs:
                self._entries[sig] = QuiverEntry(
                    name=winner.name,
                    exchange=winner.exchange,
                    spec=winner.spec,
                    negating_sequence=winner.negating_sequence,
                    provenance=winner.provenance,
                    known_specs=merged,
                )
                return sig
        self._entries[sig] = winner
        return sig

    def register_verified(self, entry: QuiverEntry, **kw) -> tuple:
        if not entry.verify():
            raise ValueError(f"entry {entry.name!r} fails spec verification")
        return self.register(entry, **kw)

    def lookup(self, key) -> QuiverEntry | None:
        sig = key if isinstance(key, tuple) else signature(key)
        return self._entries.get(sig)

    def entries(self) -> list[QuiverEntry]:
        return list(self._entries.values())

    # --- necklace-graph accessors -----------------------------------

    def add_edge(self, edge: NecklaceEdge) -> None:
        """Register a necklace edge.  Idempotent: adding the same
        edge twice is a no-op.  Asserts that both endpoints exist.

        The graph is keyed  (source, via) → edge :  the destination is
        a function of  (source, via) , so two edges from the same
        source via the same first factor must agree.
        """
        if edge.source not in self._entries or edge.target not in self._entries:
            raise ValueError("both endpoints of a necklace edge must "
                             "be registered entries first")
        out_bucket = self._edges_out.setdefault(edge.source, {})
        existing = out_bucket.get(edge.via)
        if existing is not None:
            if existing != edge:
                raise ValueError(
                    f"conflicting edge from {edge.source!r} via {edge.via}: "
                    f"existing target={existing.target}, "
                    f"new target={edge.target}"
                )
            return
        out_bucket[edge.via] = edge
        self._edges_in.setdefault(edge.target, []).append(edge)

    def successors(self, key) -> list[NecklaceEdge]:
        """Outgoing necklace edges from the given entry / signature."""
        sig = key if isinstance(key, tuple) else signature(key)
        return list(self._edges_out.get(sig, {}).values())

    def predecessors(self, key) -> list[NecklaceEdge]:
        """Incoming necklace edges to the given entry / signature.
        These correspond to inverse-necklace moves: an edge
        ``u -> v via gamma``  is recorded both as an out-edge of  u
        and an in-edge of  v ."""
        sig = key if isinstance(key, tuple) else signature(key)
        return list(self._edges_in.get(sig, []))

    def all_edges(self) -> list[NecklaceEdge]:
        out: list[NecklaceEdge] = []
        for bucket in self._edges_out.values():
            out.extend(bucket.values())
        return out

    def connected_components(self) -> list[frozenset]:
        """Weakly-connected components of the necklace graph.

        Each component is a  ``frozenset``  of signatures whose
        entries are mutation-connected through the recorded
        necklace edges.  Components are theory candidates: a
        connected component is the set of BPS-quiver charts
        reachable from any one seed by repeated necklacing -- i.e.
        the chart space of one (conjectural) physical theory.

        Returned components are sorted by descending size.  Run
        :meth:`close_under_multi_necklace_and_freezing`  before
        querying components, otherwise edges may be missing and
        the partition will be coarser than reality.
        """
        sigs = set(self._entries.keys())
        # Union-find.
        parent: dict[tuple, tuple] = {s: s for s in sigs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for edge in self.all_edges():
            if edge.source in parent and edge.target in parent:
                union(edge.source, edge.target)

        groups: dict[tuple, set[tuple]] = {}
        for s in sigs:
            groups.setdefault(find(s), set()).add(s)
        return sorted(
            (frozenset(g) for g in groups.values()),
            key=len,
            reverse=True,
        )

    def close_under_freezing(self, *, verbose: bool = False) -> int:
        added = 0
        for e in list(self._entries.values()):
            for i in range(e.n_nodes):
                try:
                    child = freeze(e, i)
                except ValueError:
                    # Includes disconnected results: freeze rejects them.
                    continue
                sig = signature(child)
                if sig not in self._entries:
                    self._entries[sig] = child
                    added += 1
                    if verbose:
                        print(f"  + {child.name}  "
                              f"(n={child.n_nodes}, "
                              f"|spec|={len(child.spec)})")
        return added

    def shorten_all(self, *,
                    max_range_len: int = 5,
                    verbose: bool = False) -> int:
        """For every entry, try :func:`shorten_spec`  and replace in
        place.  The signature is keyed by the quiver alone, so the
        shortened spec lives under the same key.

        Returns the number of entries whose spec strictly shortened.
        """
        n_shortened = 0
        total_saved = 0
        for sig in list(self._entries):
            e = self._entries[sig]
            better = shorten_spec(e, max_range_len=max_range_len)
            if len(better.spec) < len(e.spec):
                saved = len(e.spec) - len(better.spec)
                total_saved += saved
                n_shortened += 1
                self._entries[sig] = QuiverEntry(
                    name=e.name,
                    exchange=e.exchange,
                    spec=better.spec,
                    negating_sequence=better.negating_sequence,
                    provenance=e.provenance + f" [shortened -{saved}]",
                )
                if verbose:
                    print(f"  shortened {e.name}: "
                          f"{len(e.spec)} -> {len(better.spec)}  "
                          f"(saved {saved})")
        if verbose:
            print(f"  total entries shortened: {n_shortened}, "
                  f"total factors saved: {total_saved}")
        return n_shortened

    def close_under_freezing_transitively(self, *, verbose: bool = False) -> int:
        total = 0
        while True:
            added = self.close_under_freezing(verbose=verbose)
            if added == 0:
                break
            total += added
        return total

    def close_under_alt_necklace(self, *, verbose: bool = False) -> int:
        """Single pass: for every entry, try every alternative first
        factor (reachable via :func:`alternative_first_factors`);
        register the sibling quiver produced by one necklace step.

        **Not a closure** in the strict sense -- this only explores
        the commuting-swap slice of factorization-equivalent first
        factors, and only does one necklace step per alternative.
        Wider coverage comes from repeatedly interleaving this with
        ``close_under_necklace`` and  ``close_under_freezing_
        transitively`` ; the dictionary grows monotonically but is
        not guaranteed to reach a full closure under arbitrary
        factorization-equivalent reorderings.

        Returns the number of new quivers added this pass.
        """
        added = 0
        for e in list(self._entries.values()):
            for sib in mine_alt_necklaces(e):
                sig = signature(sib)
                if sig not in self._entries:
                    self._entries[sig] = sib
                    added += 1
                    if verbose:
                        print(f"  + alt-necklace sib of {e.name}: "
                              f"n={sib.n_nodes}, |spec|={len(sib.spec)}")
                else:
                    self.register(sib)
        return added

    def close_under_necklace(self, *, verbose: bool = False,
                              budget_per_entry: int | None = None) -> int:
        """Walk the necklace of every entry; register each sibling
        quiver.  The dictionary dedupes by quiver signature (shortest
        spec wins), so a necklace that revisits an already-known
        quiver is silently absorbed.

        Returns the number of new quivers added.
        """
        added = 0
        for e in list(self._entries.values()):
            try:
                cycle = necklace(e, max_steps=budget_per_entry)
            except ValueError:
                continue
            for c in cycle[1:]:
                sig = signature(c)
                if sig not in self._entries:
                    self._entries[sig] = c
                    added += 1
                    if verbose:
                        print(f"  + necklace of {e.name}: n={c.n_nodes}, "
                              f"|spec|={len(c.spec)}")
                else:
                    # Use the standard collision rule in case the
                    # sibling has a shorter spec than the stored entry.
                    self.register(c)
        return added

    # --- top-down layout: close each n-shard, then freeze to n-1 ----

    def entries_with_n(self, n: int) -> list[QuiverEntry]:
        return [e for e in self._entries.values() if e.n_nodes == n]

    def close_for_n(self, n: int, *,
                    with_necklace: bool = True,
                    with_shortening: bool = True,
                    with_alt_s: bool = False,
                    verbose: bool = False) -> int:
        """Run all  n -preserving closures on the shard with  n  nodes.
        Returns the number of new entries added (at this same  n ).

        Stages within a single call (each iterated to local fixed
        point if the previous added anything):

        * shortening (if requested)
        * necklace  (if requested): one step forward for every entry
        * alt-S necklace (if requested): commuting-swap alternative
          first factors, one necklace step each
        """
        added = 0
        if with_shortening:
            self.shorten_all(verbose=False)
        while True:
            step_added = 0
            if with_necklace:
                for e in self.entries_with_n(n):
                    try:
                        cycle = necklace(e)
                    except ValueError:
                        continue
                    for c in cycle[1:]:
                        if c.n_nodes != n:
                            continue
                        sig = signature(c)
                        if sig not in self._entries:
                            self._entries[sig] = c
                            step_added += 1
                            if verbose:
                                print(f"  + necklace sib of {e.name} at n={n}")
                        else:
                            self.register(c)
            if with_alt_s:
                for e in self.entries_with_n(n):
                    for sib in mine_alt_necklaces(e):
                        if sib.n_nodes != n:
                            continue
                        sig = signature(sib)
                        if sig not in self._entries:
                            self._entries[sig] = sib
                            step_added += 1
                            if verbose:
                                print(f"  + alt-necklace sib of {e.name} "
                                      f"at n={n}")
                        else:
                            self.register(sib)
            added += step_added
            if step_added == 0:
                break
            if with_shortening:
                self.shorten_all(verbose=False)
        return added

    def build_top_down(self, *,
                       with_necklace: bool = True,
                       with_shortening: bool = True,
                       verbose: bool = False) -> dict[int, int]:
        """Close each n-shard top-down:  max  n  first, then  freeze
        to populate  n-1 , close that, and so on to  n = 1 .

        Returns  ``{n: count_at_that_n_after_closing}`` .
        """
        result: dict[int, int] = {}
        while True:
            ns = sorted({e.n_nodes for e in self._entries.values()}, reverse=True)
            if not ns:
                break
            # Work at the highest n not yet finalised.
            n = ns[0]
            if n in result:
                break
            before = len(self._entries)
            self.close_for_n(n, with_necklace=with_necklace,
                             with_shortening=with_shortening,
                             verbose=verbose)
            # Freeze every unfrozen node of every entry at this n
            # to populate  n - 1 .
            self.close_under_freezing_transitively(verbose=False)
            result[n] = len(self.entries_with_n(n))
            if verbose:
                print(f"  n={n}: {result[n]} entries  (+{len(self._entries) - before} net across all n)")
            # Check if any n < n_max remain unfinalised.
            lower = [m for m in ns if m < n]
            if not lower:
                break
            # Next iteration handles the next lower n.
            # (entries_with_n(n) is final at this point since all
            # closures at  n  are complete; further mutations can't
            # increase  n .)
        # Also handle every remaining n that was populated by freezing
        # but didn't have any own entries at max-n start.
        done = set(result.keys())
        while True:
            ns = sorted({e.n_nodes for e in self._entries.values()}, reverse=True)
            todo = [m for m in ns if m not in done]
            if not todo:
                break
            m = todo[0]
            self.close_for_n(m, with_necklace=with_necklace,
                             with_shortening=with_shortening,
                             verbose=verbose)
            self.close_under_freezing_transitively(verbose=False)
            result[m] = len(self.entries_with_n(m))
            done.add(m)
            if verbose:
                print(f"  n={m}: {result[m]} entries")
        return result

    def verify_uniqueness(self, *,
                          radius: int = 3,
                          sample: int | None = None,
                          verbose: bool = False) -> list[dict]:
        """Run :func:`check_uniqueness_hypothesis`  on  ``sample``
        entries (or all).  Returns a list of the check results with
        ``match=False`` ; empty list means the hypothesis holds on
        every tested entry."""
        import random
        items = list(self._entries.values())
        if sample is not None and sample < len(items):
            items = random.sample(items, sample)
        failures: list[dict] = []
        for e in items:
            r = radius
            # Reduce radius for high- n  quivers to keep the L1-ball
            # tractable.  |ball| = O(r^n / n!).
            if e.n_nodes >= 7:
                r = min(r, 2)
            if e.n_nodes >= 9:
                r = min(r, 1)
            info = check_uniqueness_hypothesis(e, radius=r)
            if not info["match"]:
                info["name"] = e.name
                failures.append(info)
                if verbose:
                    print(f"  FAIL  {e.name}  at γ={info['first_disagreement']}")
            elif verbose and info["spec_2"] is not None:
                print(f"  ok    {e.name}  (matched on {info['n_vectors']} "
                      f"vectors, radius={r}, independent spec found)")
        return failures

    def prune_disconnected(self, *, verbose: bool = False) -> int:
        """Drop every stored entry whose underlying graph is
        disconnected.  Returns the number of entries removed.

        Used as a backstop when loading a dictionary that may predate
        the connectedness guard, or after a hand-rolled bulk insert.
        Builders going through :meth:`register`,  :func:`freeze` , or
        :meth:`QuiverEntry.from_algebra`  already filter at the source.
        """
        bad = [sig for sig, e in self._entries.items() if not e.is_connected()]
        for sig in bad:
            if verbose:
                print(f"  - {self._entries[sig].name}  (disconnected)")
            del self._entries[sig]
        return len(bad)

    def close_under_pentagon(self, *,
                              max_states: int | None = 10_000,
                              max_length: int | None = None,
                              variety: bool = False,
                              plateau: int = 50,
                              verbose: bool = False,
                              only_signatures: set | None = None,
                              ) -> int:
        """Run :func:`pentagon_close`  on every entry in place.  No
        new quivers are created; this only enriches each entry's
        ``known_specs``  with alternative factorisations of the same
        S .

        ``variety=True``  switches to :func:`pentagon_close_variety` :
        BFS stops when the orbit no longer adds new first/last
        factors, so memory stays bounded.  Recommended for graph
        building (where only first/last factor diversity matters)
        on  n >= 5  entries; ``variety=False``  is the default for
        compatibility / palindromic-search uses.

        ``only_signatures``  if given restricts the pass to those
        entries -- useful for incremental by-n closures.
        """
        added = 0
        for sig, e in list(self._entries.items()):
            if only_signatures is not None and sig not in only_signatures:
                continue
            before = len(e.known_specs)
            if variety:
                e2 = pentagon_close_variety(
                    e, max_states=max_states, max_length=max_length,
                    plateau=plateau,
                )
            else:
                e2 = pentagon_close(e, max_states=max_states,
                                    max_length=max_length)
            delta = len(e2.known_specs) - before
            if delta > 0:
                self._entries[sig] = e2
                added += delta
                if verbose:
                    flag = (" (capped)"
                            if max_states is not None
                            and len(e2.known_specs) >= max_states
                            else "")
                    print(f"  + {e.name}: {before} -> "
                          f"{len(e2.known_specs)} specs{flag}")
        return added

    def close_under_multi_necklace(self, *,
                                    with_pentagon: bool = True,
                                    max_states: int | None = 10_000,
                                    max_length: int | None = None,
                                    variety: bool = False,
                                    plateau: int = 50,
                                    only_signatures: set | None = None,
                                    verbose: bool = False) -> int:
        """For every entry, run :func:`multi_necklace_edges`  using
        every spec in  ``known_specs`` , register each sibling
        quiver, and add the corresponding necklace edge to the
        dictionary's graph.  Pentagon-close each entry first when
        ``with_pentagon=True``  (default) so the multi-necklace walk
        sees the full orbit.

        When a multi-necklace step lands at a sibling whose intrinsic
        quiver matches an already-stored entry but with a different
        exchange-matrix node-labeling, the new sibling's
        ``target_via``  /  ``known_specs``  are translated into the
        stored entry's basis via :func:`find_relabeling` , and the
        edge is added with the translated  ``target_via`` .  This way
        every distinct first-factor of the parent's known specs
        contributes one outgoing edge -- the dictionary captures the
        full intrinsic neighbourhood of every node.

        Returns the number of new quivers added in this pass.  Not a
        closure on its own; alternate with
        :meth:`close_under_freezing`  to fixed point (or use
        :meth:`close_under_multi_necklace_and_freezing` ).
        """
        if with_pentagon:
            self.close_under_pentagon(
                max_states=max_states, max_length=max_length,
                variety=variety, plateau=plateau,
                only_signatures=only_signatures, verbose=verbose,
            )
        added = 0
        for source_sig, e in list(self._entries.items()):
            if only_signatures is not None and source_sig not in only_signatures:
                continue
            for sib, via, target_via in multi_necklace_edges(e):
                target_sig = signature(sib)
                stored = self._entries.get(target_sig)
                if stored is None:
                    self._entries[target_sig] = sib
                    stored = sib
                    added += 1
                    if verbose:
                        print(f"  + multi-necklace sib of {e.name}: "
                              f"n={sib.n_nodes}, |spec|={len(sib.spec)}")
                    edge_target_via = target_via
                elif stored.exchange == sib.exchange:
                    # Same basis -- merge known_specs directly.
                    self.register(sib)
                    edge_target_via = target_via
                else:
                    # Same intrinsic quiver, different node labels:
                    # translate sib's data into stored's basis.
                    perm = find_relabeling(stored.exchange, sib.exchange)
                    if perm is None:
                        # Should not happen for n <= 8 (signature
                        # equivalence implies a permutation exists),
                        # but defensively skip the edge.
                        continue
                    edge_target_via = _relabel_charge(target_via, perm)
                    translated_specs = tuple(
                        _relabel_spec(s, perm) for s in sib.known_specs
                    )
                    translated_sib = QuiverEntry(
                        name=sib.name,
                        exchange=stored.exchange,
                        spec=_relabel_spec(sib.spec, perm),
                        negating_sequence=None,  # original tied to sib's basis
                        provenance=sib.provenance + " [relabeled]",
                        known_specs=translated_specs,
                    )
                    self.register(translated_sib)

                edge = NecklaceEdge(
                    source=source_sig,
                    target=target_sig,
                    via=via,
                    target_via=edge_target_via,
                )
                try:
                    self.add_edge(edge)
                except ValueError:
                    pass
        return added

    def close_under_necklace_and_freezing(self, *,
                                          verbose: bool = False) -> int:
        """Alternate necklace and freezing closures until neither
        produces new entries.  Returns total added across all passes."""
        total = 0
        while True:
            k_neck = self.close_under_necklace(verbose=verbose)
            k_free = self.close_under_freezing_transitively(verbose=verbose)
            if verbose:
                print(f"  outer pass: +{k_neck} necklace, +{k_free} freezing, "
                      f"total={len(self)}")
            if k_neck == 0 and k_free == 0:
                break
            total += k_neck + k_free
        return total

    def close_under_multi_necklace_and_freezing(
            self, *,
            with_pentagon: bool = True,
            max_states: int | None = 10_000,
            max_length: int | None = None,
            variety: bool = False,
            plateau: int = 50,
            only_signatures: set | None = None,
            verbose: bool = False) -> int:
        """Alternate :meth:`close_under_multi_necklace`  and freezing
        closures until neither adds new entries.  Each outer pass
        only takes one necklace step per entry per first-factor;
        siblings discovered this round expand their own first-factor
        sets in the next round, so the closure converges to the full
        finite-spec-chart-space without ever needing per-entry
        closure-from-scratch.

        Returns total entries added across all passes.  Edges
        accumulate in the dictionary's graph as they're discovered.

        ``variety=True`` switches the per-entry pentagon BFS to the
        plateau-stop variant (cheap, retains first/last factor
        coverage).  ``only_signatures``  scopes the closure to a
        subset of entries (incremental graph builds).
        """
        total = 0
        while True:
            k_neck = self.close_under_multi_necklace(
                with_pentagon=with_pentagon,
                max_states=max_states,
                max_length=max_length,
                variety=variety,
                plateau=plateau,
                only_signatures=only_signatures,
                verbose=verbose,
            )
            k_free = self.close_under_freezing_transitively(verbose=verbose)
            if verbose:
                print(f"  outer pass: +{k_neck} multi-necklace, "
                      f"+{k_free} freezing, total={len(self)}, "
                      f"edges={len(self.all_edges())}")
            if k_neck == 0 and k_free == 0:
                break
            total += k_neck + k_free
        return total

    # --- persistence ---------------------------------------------------

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the dictionary.  Returns a list-of-entries when
        there are no graph edges (back-compat with the original
        format), or a  ``{"entries": [...], "edges": [...]}``  dict
        when edges are present."""
        entries = [self._entry_to_dict(e) for e in self._entries.values()]
        edges = self.all_edges()
        if not edges:
            return json.dumps(entries, indent=indent)
        return json.dumps({
            "entries": entries,
            "edges": [self._edge_to_dict(e) for e in edges],
        }, indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "BPSQuiverDictionary":
        d = cls()
        obj = json.loads(s)
        if isinstance(obj, list):
            entries = obj
            edges: list[dict] = []
        else:
            entries = obj.get("entries", [])
            edges = obj.get("edges", [])
        for entry_obj in entries:
            # ``register`` silently skips disconnected entries.
            d.register(cls._entry_from_dict(entry_obj))
        for edge_obj in edges:
            edge = cls._edge_from_dict(edge_obj)
            # Skip edges whose endpoints aren't in the dictionary
            # (e.g. dropped by the connectedness filter).
            if edge.source in d._entries and edge.target in d._entries:
                try:
                    d.add_edge(edge)
                except ValueError:
                    pass
        return d

    @classmethod
    def load_default(cls, path: Path | str | None = None) -> "BPSQuiverDictionary":
        """Load the committed seed dictionary.

        Layout: ``data/bps_quiver_dictionary/`` containing

            index.json              summary (counts per n, total)
            n_001.json  n_002.json  ... one file per node-count
            edges.json              optional necklace-graph edges

        Entries are split by node count so adding a new entry only
        touches the relevant file -- git diffs stay surgical.  Edges
        live in a single file because they cross node-counts.

        Pass ``path``  to override the directory; defaults to
        :data:`DEFAULT_DATA_DIR` .  Rebuild with
        ``python scripts/build_bps_quiver_dictionary.py`` .
        """
        p = Path(path) if path is not None else DEFAULT_DATA_DIR
        d = cls()
        for shard in sorted(p.glob("n_*.json")):
            for obj in json.loads(shard.read_text()):
                # ``register`` silently skips disconnected entries
                # (defends against any stale data on disk).
                d.register(cls._entry_from_dict(obj))
        edges_path = p / "edges.json"
        if edges_path.exists():
            for edge_obj in json.loads(edges_path.read_text()):
                edge = cls._edge_from_dict(edge_obj)
                if edge.source in d._entries and edge.target in d._entries:
                    try:
                        d.add_edge(edge)
                    except ValueError:
                        pass
        return d

    @classmethod
    def load_for_n(cls, n_nodes: int,
                   path: Path | str | None = None) -> "BPSQuiverDictionary":
        """Load only the shard of entries with  ``n_nodes``  nodes.
        Returns an empty dict if no such shard exists.  Edges are
        not loaded (they cross n-shards)."""
        p = Path(path) if path is not None else DEFAULT_DATA_DIR
        shard = p / f"n_{n_nodes:03d}.json"
        d = cls()
        if shard.exists():
            for obj in json.loads(shard.read_text()):
                d.register(cls._entry_from_dict(obj))
        return d

    @staticmethod
    def _entry_to_dict(e: QuiverEntry) -> dict:
        out: dict = {
            "name": e.name,
            "exchange": [list(r) for r in e.exchange],
            "spec": [list(g) for g in e.spec],
            "negating_sequence": (list(e.negating_sequence)
                                  if e.negating_sequence is not None else None),
            "provenance": e.provenance,
        }
        # Persist alternative specs only when there are extras beyond
        # the seed -- keeps shards small for entries that haven't been
        # pentagon-closed yet.
        extras = [s for s in e.known_specs if tuple(s) != tuple(e.spec)]
        if extras:
            out["known_specs"] = [
                [list(g) for g in s] for s in extras
            ]
        return out

    @staticmethod
    def _sig_to_json(sig):
        """Convert a signature (tuple of tuples / weak fallback tuple)
        to JSON-friendly nested lists.  Strings (from the n>8 weak
        fallback header) pass through unchanged."""
        if isinstance(sig, (tuple, list)):
            return [BPSQuiverDictionary._sig_to_json(x) for x in sig]
        return sig

    @staticmethod
    def _sig_from_json(obj):
        if isinstance(obj, list):
            return tuple(BPSQuiverDictionary._sig_from_json(x) for x in obj)
        return obj

    @staticmethod
    def _edge_to_dict(e: NecklaceEdge) -> dict:
        return {
            "source": BPSQuiverDictionary._sig_to_json(e.source),
            "target": BPSQuiverDictionary._sig_to_json(e.target),
            "via": list(e.via),
            "target_via": list(e.target_via),
        }

    @staticmethod
    def _edge_from_dict(d: dict) -> NecklaceEdge:
        return NecklaceEdge(
            source=BPSQuiverDictionary._sig_from_json(d["source"]),
            target=BPSQuiverDictionary._sig_from_json(d["target"]),
            via=tuple(d["via"]),
            target_via=tuple(d["target_via"]),
        )

    @staticmethod
    def _entry_from_dict(d: dict) -> QuiverEntry:
        seed = tuple(tuple(g) for g in d["spec"])
        extras = tuple(
            tuple(tuple(g) for g in s) for s in d.get("known_specs", ())
        )
        return QuiverEntry(
            name=d["name"],
            exchange=tuple(tuple(r) for r in d["exchange"]),
            spec=seed,
            negating_sequence=(tuple(d["negating_sequence"])
                               if d.get("negating_sequence") else None),
            provenance=d.get("provenance", ""),
            known_specs=(seed,) + extras,
        )


# ---------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------

def seed_from_presets(d: BPSQuiverDictionary) -> int:
    added = 0
    for name, preset in PRESETS.items():
        A = CoulombAlgebra(preset["B"], preset["nodes"])
        try:
            e = QuiverEntry.from_algebra(A, name=name, provenance="PRESETS")
        except ValueError:
            # Disconnected or has frozen-flavour spec: skip.
            continue
        sig = signature(e)
        if sig not in d._entries:
            d._entries[sig] = e
            added += 1
    return added


def seed_from_sun_nf(d: BPSQuiverDictionary,
                     pairs: Iterable[tuple[int, int]]) -> int:
    from pure_ade import SUN_Nf
    added = 0
    for (N, Nf) in pairs:
        A = SUN_Nf(N, Nf).algebra
        try:
            e = QuiverEntry.from_algebra(
                A, name=f"SUN_Nf({N},{Nf})", provenance="pure_ade.SUN_Nf",
            )
        except ValueError:
            continue
        sig = signature(e)
        if sig not in d._entries:
            d._entries[sig] = e
            added += 1
    return added


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

def _demo() -> None:
    d = BPSQuiverDictionary()
    n_pre = seed_from_presets(d)
    n_sun = seed_from_sun_nf(d, [(2, 1), (2, 2), (3, 1), (3, 2)])
    print(f"seeded: {n_pre} presets + {n_sun} SUN_Nf  =>  |dict| = {len(d)}")
    total = d.close_under_freezing_transitively(verbose=True)
    print(f"  + {total} by freezing closure  =>  |dict| = {len(d)}")
    print("sample verification:")
    for e in list(d)[:5]:
        print(f"  {e.name}: n={e.n_nodes}, |spec|={len(e.spec)}, "
              f"verify={e.verify()}")


if __name__ == "__main__":
    _demo()
