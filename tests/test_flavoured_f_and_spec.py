"""The flavoured F-finder and the flavoured spec finder.

The author, 2026-09-14: *"agreed, flavoured F-finder.  But also, we need a spec finder
which finds specs of the flavoured form, where factors for the same irrep are
consecutive"*.

As in `tests/test_flavoured_factor_spectrum.py`, each test name is the claim and
its docstring is the claim in the paper's notation.  Enforced / emergent /
independent are labelled, per the audit — a green check is evidence
only where the property is not built in.

* **Enforced** — the defining relation `F·S = μ^w X_γ + O(\\fq)`.  The recursion
  forces exactly it, so a pass is a regression guard on the arithmetic.
* **Independent** — agreement with the **unfolded** chart, whose F-solver shares
  the defining relation with this one and nothing else: it uses the
  doubly-tropical support window and the `[n]_q` peeling solver
  (`bps_kalgebra_internals`), while the flavoured route uses a weight-graded
  recursion on the reduced cone with a closed-form forcing and no support window
  at all.
* **The spec side** — a spec is accepted only after rebuilding the product on a
  **strictly wider** cone (the audit: three fixed depths were
  refuted, so a verdict is relative to the depth paid for).

Run:  `python3 run_tests.py`
"""
import sys

sys.path.insert(0, ".")

from bps_kalgebra import BPSKAlgebra
from habiro import HabiroElement
from flavoured_factor_spectrum import FlavouredQuiver
from flavoured_f_solver import (
    FlavouredFSolver,
    enlarged_charge,
    unfolded_F,
)
from flavoured_spec import (
    find_flavoured_negating_sequence,
    find_flavoured_spec,
    spec_from_negating_sequence,
    group_unfolded_spec,
    group_unfolded_spec_up_to_commuting,
    is_flavoured_spec,
    product_of_generators,
    unfold_spec,
)

H0 = HabiroElement.zero()

CORPUS = [
    ("A2, SU(2) on node 0", [[0, 1], [-1, 0]], [2, 1], 6),
    ("A2, SU(2) x SU(2)",   [[0, 1], [-1, 0]], [2, 2], 5),
    ("A2, SU(3) on node 0", [[0, 1], [-1, 0]], [3, 1], 5),
    ("Kronecker-2, SU(2)",  [[0, 2], [-2, 0]], [2, 1], 5),
]


def _same(a, b):
    return all(a.get(k, H0) == b.get(k, H0) for k in set(a) | set(b))


# --------------------------------------------------------------------------
# the F-finder
# --------------------------------------------------------------------------

def test_trivial_flavour_reproduces_the_unflavoured_F():
    """`G` trivial ⇒ the flavoured F-finder IS the repo's F-solver.

    The positive control.  Bit-identical, not approximate.
    """
    for B, gamma, D in (([[0, 1], [-1, 0]], (1, 0), 6),
                        ([[0, 1], [-1, 0]], (0, 1), 6),
                        ([[0, 2], [-2, 0]], (1, 0), 6)):
        n = len(B)
        Q = FlavouredQuiver(B, [1] * n)
        got = {k: v for (k, _w), v in FlavouredFSolver(Q, D).solve(gamma).items()}
        A = BPSKAlgebra(
            pairing=B,
            node_charges=[tuple(1 if j == i else 0 for j in range(n))
                          for i in range(n)])
        ref = {k: HabiroElement.from_laurent(v.to_laurent())
               for k, v in A.F_qn(gamma).items()}
        assert set(got) == set(ref) and _same(got, ref), (B, gamma)


def test_F_equals_the_unfolded_F_on_node_charges():
    """`F_{(γ_a, w)}` agrees with the unfolded chart's own `F`.

    INDEPENDENT: the unfolded side knows nothing of flavour and reaches `F` by
    the tropical window plus `[n]_q` peeling.
    """
    for name, B, fac, D in CORPUS:
        Q = FlavouredQuiver(B, fac)
        sol = FlavouredFSolver(Q, D)
        _, _, labels = Q.unfold()
        for a, w in labels:
            gamma = tuple(1 if i == a else 0 for i in range(Q.rank))
            got = sol.solve(gamma, w)
            ref = {k: v for k, v in unfolded_F(Q, gamma, w).items()
                   if sum(k[0]) <= D}
            assert set(got) == set(ref) and _same(got, ref), (name, a, w)


def test_F_equals_the_unfolded_F_on_composite_charges():
    """The same, away from the nodes — where the recursion actually does work."""
    import itertools

    for name, B, fac, D in CORPUS[:3]:
        Q = FlavouredQuiver(B, fac)
        F = Q.flavour
        sol = FlavouredFSolver(Q, D)
        _, _, labels = Q.unfold()
        n = len(labels)
        checked = 0
        for deg in (2, 3):
            for combo in itertools.combinations_with_replacement(range(n), deg):
                k = [0] * n
                for i in combo:
                    k[i] += 1
                gauge = [0] * Q.rank
                wt = F.zero
                for i, m in enumerate(k):
                    if not m:
                        continue
                    a, w = labels[i]
                    gauge[a] += m
                    wt = F.add(wt, F.scale(w, m))
                gauge = tuple(gauge)
                if sum(gauge) > D - 2:
                    continue
                got = sol.solve(gauge, wt)
                ref = {kk: v for kk, v in unfolded_F(Q, gauge, wt).items()
                       if sum(kk[0]) <= D}
                assert set(got) == set(ref) and _same(got, ref), (name, gauge, wt)
                checked += 1
        assert checked, name


def test_the_defining_relation_holds():
    """`F·S = μ^w X_γ + O(\\fq)` with `f_{(γ,w)} = 1`.

    ENFORCED, not emergent: the recursion forces exactly this.  Present as a
    regression guard on the arithmetic, and labelled so it is not read as
    evidence.
    """
    for name, B, fac, D in CORPUS:
        Q = FlavouredQuiver(B, fac)
        sol = FlavouredFSolver(Q, D)
        _, _, labels = Q.unfold()
        for a, w in labels:
            gamma = tuple(1 if i == a else 0 for i in range(Q.rank))
            assert sol.verify_defining_relation(gamma, w) == [], (name, a, w)


def test_the_enlarged_charge_projection_inverts():
    """`(enlarged charge) ↦ (reduced charge, flavour weight)` is a bijection.

    The ranks match — `Σ_a dim r_a = g + Σ_a (N_a − 1)` — so a canonical label of
    the unfolded chart is recovered from its gauge charge and flavour weight, and
    the two descriptions are interchangeable.  A `(γ, w)` violating the N-ality
    condition is the image of no enlarged charge, and is refused.
    """
    Q = FlavouredQuiver([[0, 1], [-1, 0]], [3, 1])
    F = Q.flavour
    _, _, labels = Q.unfold()
    for k in ((1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0), (1, 0, 1, 1), (2, 1, 0, 1)):
        gauge = [0] * Q.rank
        wt = F.zero
        for i, n in enumerate(k):
            if not n:
                continue
            a, w = labels[i]
            gauge[a] += n
            wt = F.add(wt, F.scale(w, n))
        assert enlarged_charge(Q, tuple(gauge), wt) == k, k
    try:
        enlarged_charge(Q, (1, 0), (1, 1))      # wrong N-ality for SU(3)
    except ValueError:
        return
    raise AssertionError("expected the N-ality condition to refuse")


def test_the_rep_grouped_F_satisfies_the_character_form_of_the_relation():
    """`(Σ_{w∈r} F_{(γ,w)}) · S = χ_r(μ) X_γ + O(\\fq)`.

    The `R`-form readout.  It is a sum of canonical-basis elements, **not**
    itself one — the canonical basis stays indexed by enlarged charges
    (`kalgebra.md`, "Freeness over `R` — a convention, not a contract").
    """
    Q = FlavouredQuiver([[0, 1], [-1, 0]], [2, 1])
    F = Q.flavour
    sol = FlavouredFSolver(Q, 5)
    rep = F.fundamental(0)
    grouped = sol.solve_irrep((1, 0), rep)
    prod = sol.product_with_S(grouped)
    want = {((1, 0), w): m for w, m in F.irrep_weights(rep).items()}
    for (k, w), h in prod.items():
        nonpos = {j: c for j, c in
                  __import__("bps_factor_spectrum").expansion(h).items()
                  if j <= 0 and c}
        if (k, w) in want:
            assert nonpos == {0: want[(k, w)]}, (k, w, nonpos)
        else:
            assert not nonpos, (k, w, nonpos)


# --------------------------------------------------------------------------
# the spec finder
# --------------------------------------------------------------------------

def test_the_strip_order_gives_the_flavoured_spec_on_an_acyclic_quiver():
    """`S = ∏_a E^{(0;r_a)}_\\fq(X_{γ_a})` — one generator per reduced node.

    The flavoured mantle statement, read out as a spec: finite, spin-`0`, unit
    exponents, and attaining the floor of one generator per node.
    """
    for name, B, fac, D in CORPUS:
        Q = FlavouredQuiver(B, fac)
        res = find_flavoured_spec(Q, 4, trials=4)
        assert res.spec is not None, (name, res.reason)
        assert "strip" in res.reason, (name, res.reason)
        assert len(res.spec) == Q.rank, (name, len(res.spec))
        nodes = {tuple(1 if j == i else 0 for j in range(Q.rank))
                 for i in range(Q.rank)}
        assert set(res.spec.charges()) == nodes, (name, res.spec.charges())
        # and each node's generator carries that node's own rep
        by_charge = dict(res.spec.entries)
        for i in range(Q.rank):
            gamma = tuple(1 if j == i else 0 for j in range(Q.rank))
            assert by_charge[gamma] == Q.node_reps[i], (name, i)


def test_a_flavoured_spec_reproduces_S_on_a_strictly_wider_cone():
    """Acceptance rebuilds on a wider cone, and the verdict carries its depth.

    An in-cone rebuild is a screen, not a certificate: it admits impostors —
    contents exact at every degree checked and wrong at the next
    (the audit, where three successive fixed depths were refuted).
    So `confirmed_to` is the number to read, not the boolean.
    """
    for name, B, fac, D in CORPUS:
        Q = FlavouredQuiver(B, fac)
        res = find_flavoured_spec(Q, 4, trials=4)
        assert res.spec is not None, name
        assert res.spec.confirmed_to > 4, (name, res.spec.confirmed_to)
        ok, depth = is_flavoured_spec(Q, res.spec.entries, 4, confirm_extra=2)
        assert ok and depth == 6, (name, ok, depth)


def test_the_finder_reaches_a_quiver_where_the_strip_leaves_a_core():
    """The search does work beyond the acyclic case.

    On the 3-cycle the strip determines nothing and honest-fails, and a spec is
    found in a random order instead — so the finder is not merely re-reading the
    mantle statement.
    """
    from bps_factor_spectrum import acyclic_node_order

    B = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
    Q = FlavouredQuiver(B, [2, 1, 1])
    assert acyclic_node_order(B, [(1, 0, 0), (0, 1, 0), (0, 0, 1)]) is None
    res = find_flavoured_spec(Q, 4, trials=12)
    assert res.spec is not None, res.reason
    assert "strip" not in res.reason, res.reason
    ok, _ = is_flavoured_spec(Q, res.spec.entries, 4, confirm_extra=1)
    assert ok


# --------------------------------------------------------------------------
# the consecutiveness condition
# --------------------------------------------------------------------------

def test_unfolding_a_flavoured_spec_gives_consecutive_same_irrep_factors():
    """A flavoured spec IS an unfolded spec whose same-irrep factors adjoin.

    `E^{(0;r)}_\\fq(X_γ) = ∏_{w∈r} E^{(0)}_\\fq(μ^w X_γ)`, so unfolding lays each
    generator's `dim r` factors down consecutively — and grouping recovers the
    flavoured spec exactly.
    """
    for name, B, fac, _D in CORPUS:
        Q = FlavouredQuiver(B, fac)
        res = find_flavoured_spec(Q, 4, trials=4)
        spec = list(res.spec.entries)
        seq = unfold_spec(Q, spec)
        assert len(seq) == sum(res.spec.dims(Q)), name
        assert group_unfolded_spec(Q, seq) == spec, name


def test_separating_a_reps_factors_breaks_the_grouping():
    """Split a rep's weights and the sequence no longer descends to `\\cE_G`.

    The condition is doing work: a sequence in which a generator's factors are
    separated by a **non-commuting** factor is refused by both the literal test
    and the up-to-commuting one.
    """
    Q = FlavouredQuiver([[0, 1], [-1, 0]], [2, 1])
    spec = list(find_flavoured_spec(Q, 4, trials=4).spec.entries)
    seq = unfold_spec(Q, spec)
    assert len(seq) == 3
    separated = [seq[0], seq[2], seq[1]]
    assert Q.bracket(seq[0][0], seq[2][0]) != 0      # genuinely non-commuting
    assert group_unfolded_spec(Q, separated) is None
    assert group_unfolded_spec_up_to_commuting(Q, separated) is None


def test_literal_consecutiveness_is_a_property_of_the_presentation():
    """…and NOT an invariant of the spec — so the two tests differ, honestly.

    Factors at gauge charges with `⟨γ, γ'⟩ = 0` commute, so an intervening
    commuting factor can be slid out without changing the element.  On
    `A1 × A1` with an `SU(2)` at each node — where every factor commutes with
    every other — interleaving the two generators leaves the product **equal to
    `S`**, while the literal grouping refuses it and the up-to-commuting one
    recovers the spec.

    (The up-to-commuting routine is sufficient, not complete: the sequence lives
    in a trace monoid and deciding reachability there can need moves its greedy
    gather does not make.  A `None` from it means "not grouped by this
    procedure", never "does not descend".)
    """
    Q = FlavouredQuiver([[0, 0], [0, 0]], [2, 2])
    spec = list(find_flavoured_spec(Q, 4, trials=4).spec.entries)
    seq = unfold_spec(Q, spec)
    assert len(seq) == 4
    interleaved = [seq[0], seq[2], seq[1], seq[3]]

    assert group_unfolded_spec(Q, interleaved) is None
    assert group_unfolded_spec_up_to_commuting(Q, interleaved) == spec

    # And the interleaved sequence really does still equal `S`.
    import flavoured_factor_spectrum as ffs
    eng = ffs.FlavouredFactorSpectrum(Q, 4)
    acc = {tuple([0] * Q.rank): {Q.flavour.zero: HabiroElement.one()}}
    for ch, w in interleaved:
        acc = eng._factor_multiply(acc, tuple(ch), sum(ch), {w: {0: 1}})
    prod = {(k, w): h for k, b in acc.items() for w, h in b.items()
            if not h.is_zero()}
    assert _same(prod, product_of_generators(Q, spec, 4))


# --------------------------------------------------------------------------
# the bidirectional BFS over block mutations
# --------------------------------------------------------------------------

def test_block_mutations_commute_within_a_group():
    """A block move `μ_a = ∏_{i∈group a} μ_i` is well defined.

    The unfolded nodes of one flavoured node carry no arrows between them, so
    the charge shift `max(0, −exchange[j][i])` vanishes inside a block and a
    node's blockmates never move its charge.  Hence the block move does not
    depend on the internal order — which is what makes a search over block
    moves a search over *flavoured* specs, and lets the positive-cone
    admissibility be tested once per block rather than per node.
    """
    import itertools

    from bps_quiver_tools import BPSQuiver
    from flavoured_spec import _block_mutate, _unfolded_cover

    for B, fac in (([[0, 1], [-1, 0]], [2, 1]),
                   ([[0, 1], [-1, 0]], [3, 1]),
                   ([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], [2, 1, 1])):
        Q = FlavouredQuiver(B, fac)
        cover, blocks, _ = _unfolded_cover(Q)
        for block in blocks:
            seen = set()
            for order in itertools.permutations(block):
                q = _block_mutate(cover, list(order))
                seen.add((tuple(q.charges),
                          tuple(tuple(r) for r in q.exchange)))
            assert len(seen) == 1, (B, fac, block)


def test_trivial_flavour_reproduces_the_repo_negating_sequence():
    """`G` trivial ⇒ blocks are single nodes ⇒ the repo's own green sequence.

    The positive control for the block-move search, against
    `BPSQuiver.find_negating_sequence` — the cluster-side bidirectional BFS that
    the design notes keeps as *the cluster definition of the DT invariant*.  Lengths
    must agree; the sequences themselves need not, since a quiver generally has
    several maximal green sequences.
    """
    from bps_quiver_tools import BPSQuiver

    for B in ([[0, 1], [-1, 0]],
              [[0, 2], [-2, 0]],
              [[0, 1, 0], [-1, 0, 1], [0, -1, 0]],
              [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]):
        n = len(B)
        Q = FlavouredQuiver(B, [1] * n)
        got = find_flavoured_negating_sequence(Q, max_depth=12)
        cover = BPSQuiver(
            [tuple(1 if i == j else 0 for i in range(n)) for j in range(n)],
            exchange_matrix=[r[:] for r in B],
            ambient_pairing=[r[:] for r in B])
        ref = cover.find_negating_sequence(max_depth=12)
        assert got is not None and ref is not None, B
        assert len(got) == len(ref), (B, got, ref)


def test_the_block_bfs_finds_a_flavoured_spec():
    """A negating sequence of block moves replays to a flavoured spec.

    Each move contributes its `dim r_a` charges consecutively, so the replay
    groups into one generator per move by construction — and the product is then
    checked against `S` on a strictly wider cone.
    """
    cases = [("A2, SU(2)", [[0, 1], [-1, 0]], [2, 1]),
             ("A2, SU(2)xSU(2)", [[0, 1], [-1, 0]], [2, 2]),
             ("A2, SU(3)", [[0, 1], [-1, 0]], [3, 1]),
             ("A2, SU(4)", [[0, 1], [-1, 0]], [4, 1]),
             ("Kronecker-2, SU(2)", [[0, 2], [-2, 0]], [2, 1]),
             ("A3, SU(2) middle", [[0, 1, 0], [-1, 0, 1], [0, -1, 0]], [1, 2, 1]),
             ("3-cycle, SU(2)", [[0, 1, -1], [-1, 0, 1], [1, -1, 0]], [2, 1, 1])]
    for name, B, fac in cases:
        Q = FlavouredQuiver(B, fac)
        seq = find_flavoured_negating_sequence(Q, max_depth=10)
        assert seq is not None, name
        spec = spec_from_negating_sequence(Q, seq)
        assert spec is not None, (name, seq)
        assert len(spec) == len(seq), (name, len(spec), len(seq))
        ok, depth = is_flavoured_spec(Q, spec, 4, confirm_extra=1)
        assert ok and depth == 5, (name, ok, depth)


def test_the_bidirectional_search_agrees_with_the_plain_one():
    """Meet-in-the-middle finds the same length as the plain forward BFS.

    The permutation matching is the subtle part of the bidirectional search —
    a backward block move has to be carried to the forward side through the
    bijection relating the matched states — so it is pinned against the
    unidirectional route, which has no matching at all.
    """
    for name, B, fac in (("A2, SU(2)", [[0, 1], [-1, 0]], [2, 1]),
                         ("A2, SU(3)", [[0, 1], [-1, 0]], [3, 1]),
                         ("3-cycle, SU(2)",
                          [[0, 1, -1], [-1, 0, 1], [1, -1, 0]], [2, 1, 1])):
        Q = FlavouredQuiver(B, fac)
        bi = find_flavoured_negating_sequence(Q, max_depth=10)
        plain = find_flavoured_negating_sequence(Q, max_depth=10,
                                                 bidirectional=False)
        assert bi is not None and plain is not None, name
        assert len(bi) == len(plain), (name, bi, plain)
        for seq in (bi, plain):
            spec = spec_from_negating_sequence(Q, seq)
            assert spec is not None and is_flavoured_spec(Q, spec, 4)[0], (name, seq)


def test_the_two_spec_finders_agree_in_length():
    """The block-move BFS and the order search reach specs of the same length.

    INDEPENDENT: the two share no mechanism.  One searches *green sequences* by
    block mutation of the unfolded quiver (the cluster side); the other searches
    the *order* on `Γ_+ × ½N × Irr(G)` and reads `Ω` off the recursion.  They
    need not return the same spec — different chambers — so length is what is
    compared.
    """
    for name, B, fac in (("A2, SU(2)", [[0, 1], [-1, 0]], [2, 1]),
                         ("A2, SU(3)", [[0, 1], [-1, 0]], [3, 1]),
                         ("Kronecker-2, SU(2)", [[0, 2], [-2, 0]], [2, 1]),
                         ("A3, SU(2) middle",
                          [[0, 1, 0], [-1, 0, 1], [0, -1, 0]], [1, 2, 1]),
                         ("3-cycle, SU(2)",
                          [[0, 1, -1], [-1, 0, 1], [1, -1, 0]], [2, 1, 1])):
        Q = FlavouredQuiver(B, fac)
        seq = find_flavoured_negating_sequence(Q, max_depth=10)
        from_bfs = spec_from_negating_sequence(Q, seq)
        from_order = find_flavoured_spec(Q, 4, trials=8).spec
        assert from_bfs is not None and from_order is not None, name
        assert len(from_bfs) == len(from_order), (
            name, len(from_bfs), len(from_order))


TESTS = [
    test_trivial_flavour_reproduces_the_unflavoured_F,
    test_F_equals_the_unfolded_F_on_node_charges,
    test_F_equals_the_unfolded_F_on_composite_charges,
    test_the_defining_relation_holds,
    test_the_enlarged_charge_projection_inverts,
    test_the_rep_grouped_F_satisfies_the_character_form_of_the_relation,
    test_the_strip_order_gives_the_flavoured_spec_on_an_acyclic_quiver,
    test_a_flavoured_spec_reproduces_S_on_a_strictly_wider_cone,
    test_the_finder_reaches_a_quiver_where_the_strip_leaves_a_core,
    test_unfolding_a_flavoured_spec_gives_consecutive_same_irrep_factors,
    test_separating_a_reps_factors_breaks_the_grouping,
    test_literal_consecutiveness_is_a_property_of_the_presentation,
    test_block_mutations_commute_within_a_group,
    test_trivial_flavour_reproduces_the_repo_negating_sequence,
    test_the_block_bfs_finds_a_flavoured_spec,
    test_the_bidirectional_search_agrees_with_the_plain_one,
    test_the_two_spec_finders_agree_in_length,
]


if __name__ == "__main__":
    for fn in TESTS:
        fn()
        print(f"  PASS: {fn.__name__}")
    print(f"All {len(TESTS)} flavoured F / spec tests passed.")
