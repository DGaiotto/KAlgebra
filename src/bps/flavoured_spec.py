"""Flavoured specs — `S = ∏_a E^{(0;r_a)}_\\fq(X_{γ_a})`, and finding them.

The author's direction, 2026-09-14: *"we need a spec finder which finds specs of the
flavoured form, where factors for the same irrep are consecutive"*.

What a flavoured spec is
------------------------

A **spec**, repo-wide, is a finite exact certificate for `S`: an ordered finite
sequence of charges whose `E_\\fq` product is `S`, with spin `0` and positive
multiplicities (`spec_acceptance.py`, `BPSKAlgebra(spec=…)`).  A **flavoured**
spec is the same thing over `\\cE_G`: an ordered finite sequence of pairs
`(γ_i, r_i)` — a reduced gauge charge and an irrep of `G` — with

    S  =  ∏_i E^{(0;r_i)}_\\fq(X_{γ_i}) ,

each generator entering once, in the given order.

The consecutiveness condition, and why it is the whole point
------------------------------------------------------------

`E^{(0;r)}_\\fq(X_γ) = ∏_{w∈r} E^{(0)}_\\fq(μ^w X_γ)` is a product of `dim r`
ordinary factors of the **unfolded** quiver.  So a flavoured spec is exactly an
unfolded spec in which, for each generator, those `dim r` factors sit
**consecutively** — that is what lets them be collected into one
`E^{(0;r)}_\\fq`.  The author's condition is therefore not an extra demand on top
of "is a spec": it is precisely the condition under which an unfolded spec
descends to `\\cE_G` at all.

Two facts make the condition well posed rather than order-dependent:

* the `dim r` factors of one generator **commute** — same gauge charge, flavour
  central, `⟨γ,γ⟩ = 0` — so a consecutive block may be permuted internally
  without changing anything, and "consecutive" is the only real constraint;
* an unfolded spec whose blocks are *not* consecutive still factors `S` (it is a
  perfectly good spec of the unfolded quiver) but is **not** a factorization in
  `\\cE_G` — the same phenomenon measured on the `S` side, where a weight-level
  order leaves `S` unchanged and destroys the character property
  (`flavoured_s_axioms.md` §4).

Acceptance: a wider cone, never a wider margin
----------------------------------------------

The repo's hard-won rule applies verbatim (the audit, three fixed
depths refuted in two days): *"factors stopped early"* inside a truncation is a
**screen**, not a certificate, because in-cone rebuilds admit impostors —
contents exact at every degree checked and wrong at the next.  So acceptance
here rebuilds the candidate product on a **strictly wider** cone and the verdict
is reported relative to `confirmed_to`, the degree actually paid for.  Read the
depth, not the boolean.

What is settled and what is not
-------------------------------

On an **acyclic** flavoured quiver the strip order gives the spec outright — one
generator per reduced node carrying that node's own rep, attaining the floor
(`flavoured_s_axioms.md` §6b).  Where the strip leaves a core, this module
searches; it inherits the unflavoured situation, where a finite spin-`0`
factorization need not exist at all (`factor_order_search`), and honest-fails
rather than returning something weaker.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from habiro import HabiroElement

from bps_factor_spectrum import acyclic_node_order, mean_rank_key
from flavoured_factor_spectrum import (
    FlavouredFactorSpectrum,
    FlavouredQuiver,
    Irrep,
    Vec,
    Weight,
)

H0 = HabiroElement.zero()
H1 = HabiroElement.one()


@dataclass(frozen=True)
class FlavouredSpec:
    """An ordered finite sequence of `(charge, irrep)` generators.

    `confirmed_to` is the cone degree the acceptance actually paid to check;
    a spec is a claim **relative to it**, never absolutely (the audit).
    """

    entries: tuple[tuple[Vec, Irrep], ...]
    confirmed_to: int = 0

    def __len__(self) -> int:
        return len(self.entries)

    def charges(self) -> tuple[Vec, ...]:
        return tuple(g for g, _ in self.entries)

    def dims(self, quiver: FlavouredQuiver) -> tuple[int, ...]:
        return tuple(quiver.flavour.irrep_dim(r) for _, r in self.entries)


# --------------------------------------------------------------------------
# rebuilding `S` from a spec — the independent check
# --------------------------------------------------------------------------

def product_of_generators(quiver: FlavouredQuiver, spec: Sequence,
                          cutoff: int) -> dict[tuple[Vec, Weight], HabiroElement]:
    """`∏_i E^{(0;r_i)}_\\fq(X_{γ_i})` on the reduced cone, in the given order.

    Built by multiplying the generators out directly — it does **not** run the
    `S` recursion, so comparing the result against `S` is a genuine check of the
    spec rather than a restatement of how it was found.
    """
    eng = FlavouredFactorSpectrum(quiver, cutoff)
    F = quiver.flavour
    zero = tuple([0] * quiver.rank)
    acc: dict[Vec, dict[Weight, HabiroElement]] = {zero: {F.zero: H1}}
    for charge, rep in spec:
        charge = tuple(charge)
        deg = sum(charge)
        if deg < 1:
            raise ValueError("a spec entry must have a positive charge")
        blk = eng.generator_weight_content(rep, 0, 1)
        acc = eng._factor_multiply(acc, charge, deg, blk)
    out: dict[tuple[Vec, Weight], HabiroElement] = {}
    for k, block in acc.items():
        for w, h in block.items():
            if not h.is_zero():
                out[(k, w)] = h
    return out


def _spectrum(quiver: FlavouredQuiver, cutoff: int):
    eng = FlavouredFactorSpectrum(quiver, cutoff)
    eng.run()
    return eng.spectrum_generator()


def is_flavoured_spec(quiver: FlavouredQuiver, spec: Sequence, cutoff: int, *,
                      confirm_extra: int = 1) -> tuple[bool, int]:
    """Does the ordered product reproduce `S`?  Returns `(verdict, depth)`.

    Checked on `cutoff + confirm_extra` — **strictly wider** than any cone the
    candidate was found on — because an in-cone rebuild is a screen and not a
    certificate.  The depth is returned beside the verdict and is the thing to
    read; `confirm_extra` buys more of it.
    """
    depth = cutoff + max(0, int(confirm_extra))
    want = _spectrum(quiver, depth)
    got = product_of_generators(quiver, spec, depth)
    keys = set(want) | set(got)
    ok = all(want.get(k, H0) == got.get(k, H0) for k in keys)
    return ok, depth


# --------------------------------------------------------------------------
# the bridge to the unfolded quiver — the consecutiveness condition
# --------------------------------------------------------------------------

def unfold_spec(quiver: FlavouredQuiver, spec: Sequence) -> list[tuple[Vec, Weight]]:
    """Expand each generator into its `dim r` unfolded factors, in place.

    The result is an ordered sequence of enlarged charges in which each
    generator's factors are, by construction, **consecutive** — which is the
    shape the condition names.
    """
    F = quiver.flavour
    out: list[tuple[Vec, Weight]] = []
    for charge, rep in spec:
        for w, mult in sorted(F.irrep_weights(rep).items()):
            out.extend([(tuple(charge), w)] * mult)
    return out


def group_unfolded_spec(quiver: FlavouredQuiver,
                        sequence: Sequence[tuple[Vec, Weight]]
                        ) -> list[tuple[Vec, Irrep]] | None:
    """Collect an unfolded sequence into flavoured generators, or `None`.

    Walks the sequence and greedily takes maximal runs sharing a gauge charge;
    a run is accepted when its weight multiset is exactly the weight diagram of
    some irrep of `G`.  Returns `None` — an honest failure, not a partial
    answer — as soon as a run is not a complete rep, which is precisely the
    case where the same-irrep factors are **not** consecutive and the unfolded
    spec does not descend to `\\cE_G`.
    """
    F = quiver.flavour
    out: list[tuple[Vec, Irrep]] = []
    i = 0
    n = len(sequence)
    while i < n:
        charge = tuple(sequence[i][0])
        j = i
        run: dict[Weight, int] = {}
        while j < n and tuple(sequence[j][0]) == charge:
            w = tuple(sequence[j][1])
            run[w] = run.get(w, 0) + 1
            j += 1
            rep = _rep_with_weights(F, run)
            if rep is not None:
                out.append((charge, rep))
                run = {}
                i = j
                break
        else:
            return None
        if run:
            return None
    return out


def group_unfolded_spec_up_to_commuting(
        quiver: FlavouredQuiver,
        sequence: Sequence[tuple[Vec, Weight]]) -> list[tuple[Vec, Irrep]] | None:
    """Group an unfolded sequence, allowing factors to be slid past commuting ones.

    **Why this exists, and why it is not the same as the literal test.**  Literal
    consecutiveness is a property of the *presentation*, not an invariant of the
    spec: two unfolded factors at gauge charges `γ, γ'` commute when
    `⟨γ, γ'⟩ = 0`, so an intervening commuting factor can be slid out without
    changing the product.  Measured on `A1 × A1` with `SU(2)` at each node,
    where every factor commutes with every other: interleaving the two
    generators' factors leaves the product **equal to `S`** while
    :func:`group_unfolded_spec` — correctly, for what it tests — returns `None`.

    So this routine answers the invariant question: can the same-irrep factors be
    **made** consecutive without changing the element?  It gathers each block by
    sliding intervening factors out, and only when every one of them commutes
    with the block's gauge charge.

    **Sufficient, not complete**, and honestly so: the sequence lives in a trace
    monoid and deciding reachability there can need moves this greedy gather does
    not make.  A `None` therefore means *"this procedure could not group it"*,
    not *"it does not descend"*.
    """
    F = quiver.flavour
    remaining = [(tuple(c), tuple(w)) for c, w in sequence]
    out: list[tuple[Vec, Irrep]] = []
    while remaining:
        charge = remaining[0][0]
        run: dict[Weight, int] = {}
        taken: list[int] = []
        rep = None
        for idx, (c, w) in enumerate(remaining):
            if c != charge:
                # May we slide this one past the block being gathered?
                if quiver.bracket(charge, c) != 0:
                    break
                continue
            run[w] = run.get(w, 0) + 1
            taken.append(idx)
            rep = _rep_with_weights(F, run)
            if rep is not None:
                break
        if rep is None:
            return None
        out.append((charge, rep))
        drop = set(taken)
        remaining = [x for i, x in enumerate(remaining) if i not in drop]
    return out


def _rep_with_weights(flavour, run: dict[Weight, int]) -> Irrep | None:
    """The irrep whose weight diagram is exactly `run`, or `None`."""
    try:
        decomposed = flavour.decompose({w: c for w, c in run.items()})
    except ValueError:
        return None                      # not Weyl-invariant, so not a rep
    if len(decomposed) != 1:
        return None
    (rep, mult), = decomposed.items()
    return rep if mult == 1 else None


# --------------------------------------------------------------------------
# the finder
# --------------------------------------------------------------------------

@dataclass
class SpecSearchResult:
    spec: FlavouredSpec | None
    order_tried: int
    reason: str
    exhaustive: bool = False


def find_flavoured_spec(quiver: FlavouredQuiver, cutoff: int, *,
                        trials: int = 24, seed: int = 20260914,
                        confirm_extra: int = 1) -> SpecSearchResult:
    """Search the order on `Γ_+ × ½N × Irr(G)` for a flavoured spec.

    The strip order is tried first where the quiver is acyclic — it is this
    repo's mantle statement and it attains the floor, one generator per node —
    and random orders after that.  A candidate is screened by *all spins zero,
    all exponents `+1`, factors stopping strictly inside the cone*, and then
    **confirmed on a wider cone** before being returned.

    Honest failure: returns `spec=None` with a reason.  A finite spin-`0`
    factorization need not exist (the unflavoured situation, `factor_order_search`),
    and nothing weaker is returned in its place.
    """
    rng = random.Random(seed)
    rank = quiver.rank
    orders: list[tuple[str, object]] = []

    charges = [tuple(1 if j == i else 0 for j in range(rank)) for i in range(rank)]
    node_order = acyclic_node_order(quiver.pairing, charges)
    if node_order is not None:
        rank_key = mean_rank_key(node_order)
        orders.append(("strip", lambda k, two_s, r, key=rank_key: (key(k), k, two_s, r)))

    for t in range(trials):
        memo: dict = {}

        def key(*args, memo=memo, rng=rng):
            if args not in memo:
                memo[args] = rng.random()
            return memo[args]

        orders.append((f"random[{t}]", key))

    for name, piece_key in orders:
        eng = FlavouredFactorSpectrum(quiver, cutoff, piece_key=piece_key)
        eng.run()
        placements = eng.placements
        if not placements:
            continue
        if any(two_s != 0 or amount != 1 for _, _, two_s, amount in placements):
            continue                       # spin, or a non-unit exponent
        top = max(sum(k) for k, _, _, _ in placements)
        if top >= cutoff:
            continue                       # did not stop inside the cone
        spec = [(k, rep) for k, rep, _, _ in placements]
        ok, depth = is_flavoured_spec(quiver, spec, cutoff,
                                      confirm_extra=confirm_extra)
        if ok:
            return SpecSearchResult(
                spec=FlavouredSpec(entries=tuple(spec), confirmed_to=depth),
                order_tried=orders.index((name, piece_key)) + 1,
                reason=f"found in the {name} order",
            )
    return SpecSearchResult(
        spec=None, order_tried=len(orders),
        reason="no order tried gave a finite spin-0 factorization with unit "
               "exponents; a flavoured spec may not exist",
    )


# --------------------------------------------------------------------------
# the bidirectional BFS over BLOCK mutations
# --------------------------------------------------------------------------
#
# The author, 2026-09-14: *"We should have a bidirectional flavoured BFS for specs"*.
#
# The cluster-side spec finder is `BPSQuiver.find_negating_sequence` — a
# explicitly because it is *the cluster definition of the DT invariant*.  Its
# flavoured counterpart changes the MOVE SET, not the search:
#
#   * a move is a **block mutation** `μ_a = ∏_{i ∈ group a} μ_i`, mutating every
#     unfolded node of one flavoured node at once;
#   * within a group the unfolded nodes carry no arrows between them
#     (`exchange[i][j] = 0`), so their mutations **commute** and the block move
#     is well defined independently of the internal order — measured, and with a
#     mechanism: the charge shift is `max(0, −exchange[j][i])`, which vanishes
#     inside a block, so a node's blockmates do not move its charge at all.
#     That also lets the positive-cone admissibility be tested once per block.
#
# A negating sequence of block moves replays to a sequence of `dim r_a` charges
# per move, consecutive by construction — which is exactly the flavoured spec
# condition (`group_unfolded_spec`).  So the block-move BFS searches the
# flavoured specs directly, rather than searching unfolded specs and filtering.


def _unfolded_cover(quiver: FlavouredQuiver):
    """The unfolded quiver as a free-cover `BPSQuiver`, plus its node groups.

    Free cover — charges the standard basis — for the reason
    `BPSQuiver._free_cover` gives: searching in the cover avoids both the
    over-permissive cone test and the collapse of distinct states onto one
    charge multiset when the charges are linearly dependent.
    """
    from bps_quiver_tools import BPSQuiver

    B, _charges, labels = quiver.unfold()
    n = len(labels)
    cover = BPSQuiver(
        [tuple(1 if i == j else 0 for i in range(n)) for j in range(n)],
        frozen=[False] * n,
        exchange_matrix=[row[:] for row in B],
        ambient_pairing=[row[:] for row in B],
    )
    blocks: dict[int, list[int]] = {}
    for i, (a, _w) in enumerate(labels):
        blocks.setdefault(a, []).append(i)
    return cover, [blocks[a] for a in sorted(blocks)], labels


def _block_mutate(q, block, *, reverse=False):
    for i in block:
        q = q.reverse_mutate(i) if reverse else q.mutate(i)
    return q


def find_flavoured_negating_sequence(quiver: FlavouredQuiver, *,
                                     max_depth: int = 12,
                                     bidirectional: bool = True
                                     ) -> list[int] | None:
    """A negating sequence of BLOCK mutations — the flavoured green sequence.

    Returns reduced node indices (one per block move), or `None` — an honest
    failure, since a finite chamber need not exist.

    Bidirectional by default: forward from the quiver under block `mutate` with
    the positive-cone constraint, backward from the reflected endpoint (every
    charge negated) under block `reverse_mutate` with the negative-cone
    constraint, meeting on the multiset of charges.  This is the flavoured form
    of `BPSQuiver._find_negating_sequence_bidirectional`; the moves are blocks
    and everything else follows it.
    """
    from collections import deque

    from bps_quiver_tools import BPSQuiver, _in_positive_cone_int

    cover, blocks, _labels = _unfolded_cover(quiver)
    n = cover.n_nodes
    idx = list(range(n))
    originals = [cover.charges[i] for i in idx]
    back_originals = [tuple(-x for x in c) for c in originals]
    neg = BPSQuiver([tuple(-x for x in c) for c in cover.charges],
                    frozen=cover.frozen[:], exchange_matrix=cover.exchange,
                    ambient_pairing=cover.ambient_pairing)

    def key(q):
        return tuple(sorted(q.charges[i] for i in idx))

    def admissible(q, block, cone):
        # One test per block: inside a block the nodes do not move each other's
        # charges, so the state each sees is the state at the block's start.
        return all(_in_positive_cone_int(q.charges[i], cone) for i in block)

    if not bidirectional:
        queue = deque([(cover, [])])
        seen = {key(cover)}
        goal = key(neg)
        while queue:
            cur, path = queue.popleft()
            if key(cur) == goal:
                return path
            if len(path) >= max_depth:
                continue
            for a, block in enumerate(blocks):
                if not admissible(cur, block, originals):
                    continue
                nxt = _block_mutate(cur, block)
                k = key(nxt)
                if k in seen:
                    continue
                seen.add(k)
                queue.append((nxt, path + [a]))
        return None

    def permutation(q_f, q_b):
        """A bijection `i ↦ j` with `q_f.charges[i] == q_b.charges[j]`."""
        f = {i: q_f.charges[i] for i in idx}
        b = {j: q_b.charges[j] for j in idx}
        perm: dict[int, int] = {}
        used: set[int] = set()

        def walk(pos):
            if pos == len(idx):
                return True
            i = idx[pos]
            for j in idx:
                if j in used or b[j] != f[i]:
                    continue
                perm[i] = j
                used.add(j)
                if walk(pos + 1):
                    return True
                used.remove(j)
                del perm[i]
            return False

        return perm if walk(0) else None

    def block_of(node, table):
        for a, blk in enumerate(table):
            if node in blk:
                return a
        return None

    fwd = {key(cover): (cover, [])}
    bwd = {key(neg): (neg, [])}
    fwd_front, bwd_front = [(cover, [])], [(neg, [])]

    for _depth in range(max_depth):
        for front, store, other, cone, rev in (
                (fwd_front, fwd, bwd, originals, False),
                (bwd_front, bwd, fwd, back_originals, True)):
            nxt_front = []
            for cur, path in front:
                for a, block in enumerate(blocks):
                    if not admissible(cur, block, cone):
                        continue
                    nq = _block_mutate(cur, block, reverse=rev)
                    k = key(nq)
                    if k in store:
                        continue
                    store[k] = (nq, path + [a])
                    nxt_front.append((nq, path + [a]))
                    hit = other.get(k)
                    if hit is None:
                        continue
                    oq, opath = hit
                    if rev:
                        f_q, f_path, b_q, b_path = oq, opath, nq, path + [a]
                    else:
                        f_q, f_path, b_q, b_path = nq, path + [a], oq, opath
                    perm = permutation(f_q, b_q)
                    if perm is None:
                        continue
                    inv = {v: kk for kk, v in perm.items()}
                    # A backward block move at reduced node `a` becomes, on the
                    # forward side, the block through the matched permutation.
                    tail = []
                    ok = True
                    for j in reversed(b_path):
                        moved = {inv.get(i) for i in blocks[j]}
                        target = block_of(next(iter(moved)), blocks)
                        if target is None or set(blocks[target]) != moved:
                            ok = False
                            break
                        tail.append(target)
                    if ok:
                        return list(f_path) + tail
            if rev:
                bwd_front = nxt_front
            else:
                fwd_front = nxt_front
        if not fwd_front and not bwd_front:
            break
    return None


def spec_from_negating_sequence(quiver: FlavouredQuiver, sequence: Sequence[int]
                                ) -> list[tuple[Vec, Irrep]] | None:
    """Replay a block-move sequence into a flavoured spec, or `None`.

    Each block move contributes `dim r_a` unfolded charges, consecutive by
    construction; `group_unfolded_spec` then collects them into one generator
    per move.  A `None` means the replayed charges did not group — which would
    be a real finding, since the block structure is supposed to guarantee it.
    """
    cover, blocks, labels = _unfolded_cover(quiver)
    F = quiver.flavour
    unfolded: list[tuple[Vec, Weight]] = []
    cur = cover
    for a in sequence:
        for i in blocks[a]:
            raw = cur.charges[i]
            gauge = [0] * quiver.rank
            wt = F.zero
            for j, m in enumerate(raw):
                if not m:
                    continue
                node, w = labels[j]
                gauge[node] += m
                wt = F.add(wt, F.scale(w, m))
            unfolded.append((tuple(gauge), wt))
            cur = cur.mutate(i)
    return group_unfolded_spec(quiver, unfolded)
