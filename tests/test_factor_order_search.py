"""`factor_order_search` — the BFS over BPS-factor insertion points.

What is actually load-bearing here, stated up front because the module reports a
*search result* and a search that finds nothing is easy to mistake for a search
that proved nothing exists:

* Every reported factorisation is checked against a Nahm-sum expansion sharing
  no code with the factor recursion — `Theory.S_from_spec` for a spin-0 spec, and
  `nahm_local.general_nahm_habiro` for a content carrying spin.  That is the
  evidence, and since 2026-08-14 it reaches **every** route the module returns,
  the sparse fallback included.
* `exhaustive` is checked to be honest — `False` whenever a cap bit — so a
  negative result is never read as a proof unless it says so.
* The prunes (`require_spin_zero`, `require_positive`) are checked to be EXACT,
  by verifying that turning them off does not find something they excluded.

Run:  `python3 run_tests.py`
"""
import sys
sys.path.insert(0, ".")

from habiro import HabiroElement

import factor_order_search

from bps_quiver_tools import BPSQuiver
from factor_order_search import (
    Content,
    FactorOrderSearch,
    _normal_form,
    default_cost,
    find_simple_factorisation,
    leading_data_from_spectrum,
    simplify_factorisation,
)
from bps_factor_spectrum import BPSFactorSpectrum, spin_decompose
from recursive_spectrum import Theory, extract_spec_from_quiver

H0 = HabiroElement.zero()
PENTAGON = [[0, 1], [-1, 0]]
KRONECKER2 = [[0, 2], [-2, 0]]
B2 = [(1, 0), (0, 1)]
B3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
B4 = [tuple(1 if i == k else 0 for i in range(4)) for k in range(4)]
SU3_CYCLIC = [[0, 1, 0, -2], [-1, 0, 2, 0], [0, -2, 0, 1], [2, 0, -1, 0]]
FIVE_CYCLE = [[0, 1, 0, 0, -1], [-1, 0, 1, 0, 0], [0, -1, 0, 1, 0],
              [0, 0, -1, 0, 1], [1, 0, 0, -1, 0]]
NODES5 = [tuple(1 if j == k else 0 for j in range(5)) for k in range(5)]


def three_cycle(a, b, c):
    return [[0, a, -c], [-a, 0, b], [c, -b, 0]]


def reference_S(pairing, nodes, degree):
    built = BPSFactorSpectrum(pairing, nodes, degree, order="lex")
    built.run()
    return built.spectrum_generator()


def rebuilds(pairing, nodes, degree, spec):
    """`∏ E_𝖖(X_{spec_i})` reproduces `S` in-cone, by the independent route."""
    theory = Theory("t", pairing, nodes, CONE=degree)
    rebuilt = theory.S_from_spec(list(spec))
    target = reference_S(pairing, nodes, degree)
    return all(rebuilt.get(g, H0) == target.get(g, H0)
               for g in set(rebuilt) | set(target) if theory.in_cone(g))


# ----------------------------------------------------------------------
# the specs it finds are real
# ----------------------------------------------------------------------

CORED = (
    ("3-cycle(1,1,1)", three_cycle(1, 1, 1), B3, 5, 4),
    ("3-cycle(2,1,1)", three_cycle(2, 1, 1), B3, 5, 4),
    ("3-cycle(2,2,1)", three_cycle(2, 2, 1), B3, 5, 4),
    ("SU(3) cyclic", SU3_CYCLIC, B4, 4, 6),
)


def test_found_specs_rebuild_S_independently():
    """The claim is `S = ∏ E_𝖖(X_{spec_i})` in this order — so rebuild it.

    Checked against `Theory.S_from_spec`, which expands the ordered product as a
    Nahm sum and shares no code with the factor recursion.  Without this the module
    would only be asserting its own arithmetic back to itself; with it, a wrong
    order is caught, which is exactly how the strip shortcut's
    dict-order-vs-placement-order defect was found.
    """
    for name, pairing, nodes, degree, expected in CORED:
        found = find_simple_factorisation(pairing, nodes, degree)
        assert found.is_spec, name
        assert len(found.spec) == expected, (name, found.spec)
        assert rebuilds(pairing, nodes, degree, found.spec), name


def test_the_search_beats_the_default_order_where_the_strip_has_no_answer():
    """The reason the module exists, in one measurement.

    On a quiver whose source/sink strip leaves a core the engine's default falls
    back to random placement, which is a legitimate total order but a poor
    factorisation: many factors, and genuinely higher spin.  The search finds a
    spin-0 one.  Both halves are asserted, because "fewer factors" alone would be
    satisfied by a lucky draw while "spin 0" is the property consumers need.
    """
    for name, pairing, nodes, degree, expected in CORED:
        default = BPSFactorSpectrum(pairing, nodes, degree)
        assert default.order == "random", f"{name} should have a core"
        default.run()
        content = default.multiplicities()
        default_spin = max((two_s for om in content.values()
                            for two_s in spin_decompose(om)), default=0)

        found = find_simple_factorisation(pairing, nodes, degree)
        assert found.content.max_spin_doubled == 0, name
        assert len(found.spec) == expected < len(content), (
            name, len(found.spec), len(content))
        assert default_spin > 0 or len(content) > expected, (
            f"{name}: the default order was already fine, so this measured "
            f"nothing")


def test_agrees_with_the_two_independent_spec_routes():
    """Three routes to a finite factorisation must agree on whether one exists.

    `extract_spec_from_quiver` is the crystalline extractor (consumes `S`,
    recovers factors by insertion); `BPSQuiver.find_negating_sequence` is the
    CLUSTER-side finder (a green-sequence BFS over mutations — the cluster
    definition of the DT invariant, explicitly retained by the user on
    2026-08-13).  This search is a third, over factor orders.

    Agreement is asserted on **existence and length**, NOT on the charges: specs
    are not unique — different chambers give different ones, and at the 3-cycles
    the three routes measurably return three *different* valid 4-factor specs.
    Asserting equality of the charge sets would be asserting something false.
    """
    for name, pairing, nodes, degree, expected in CORED:
        found = find_simple_factorisation(pairing, nodes, degree)
        extracted = extract_spec_from_quiver(pairing, nodes, cutoff=degree)
        quiver = BPSQuiver(nodes, exchange_matrix=pairing,
                           ambient_pairing=pairing)
        sequence = quiver.find_negating_sequence(max_depth=12)
        clustered = (quiver.build_spectrum_generator(sequence)
                     if sequence is not None else None)

        assert extracted is not None and clustered is not None, name
        assert len(found.spec) == len(extracted) == len(clustered) == expected, (
            name, found.spec, extracted, clustered)


def test_no_finite_factorisation_where_none_exists():
    """Markov and the wild 3-cycle admit no finite `E_𝖖` product, and the search
    says so — with `exhaustive=True`, so the negative is complete over the
    searched space rather than an artifact of the search width.  The two independent routes are
    asserted to agree that nothing is there, so this cannot pass by the search
    merely being weak."""
    for name, pairing in (("Markov", three_cycle(2, 2, 2)),
                          ("3-cycle(3,3,3)", three_cycle(3, 3, 3))):
        # The spin-0 pass alone: nothing, and it really did cut branches.
        pruned = find_simple_factorisation(pairing, B3, 4, fallback=False)
        assert not pruned.is_spec and pruned.spec is None, name
        assert pruned.branches_cut > 0, f"{name}: nothing was actually cut"

        # Completeness is asserted from the setting that actually confers it —
        # `keep_per_degree=None`, which discards nothing — rather than from the
        # default, whose negative is only "not found".  Reading the default's
        # negative as a proof is exactly the mistake `exhaustive` exists to
        # prevent, so the test must not make it either.
        complete = find_simple_factorisation(pairing, B3, 4, fallback=False,
                                             keep_per_degree=None)
        assert not complete.is_spec, name
        assert complete.exhaustive, f"{name}: negative result is not complete"


def test_the_sparse_fallback_returns_something_usable_where_no_spec_exists():
    """No spec is not the same as nothing worth reporting.

    A sparse `Ω` is an optimization goal in its own right — it makes downstream
    calculation cheaper — so on a quiver admitting no spin-0
    factorisation the search drops the prune rather than returning empty, and the
    result is measurably better than the order the engine would otherwise use.

    What is asserted is the *gain*, not merely that something came back: strictly
    fewer populated charges AND no higher spin than the default order, plus
    `is_spec = False` and `route = "search-sparse"` so the weaker claim cannot be
    read as the stronger one.
    """
    for name, pairing, nodes, degree in (
        ("Markov", three_cycle(2, 2, 2), B3, 4),
        ("3-cycle(3,3,3)", three_cycle(3, 3, 3), B3, 4),
    ):
        default = BPSFactorSpectrum(pairing, nodes, degree)
        default.run()
        baseline = default.multiplicities()
        baseline_spin = max((two_s for om in baseline.values()
                             for two_s in spin_decompose(om)), default=0)

        found = find_simple_factorisation(pairing, nodes, degree)
        assert found.route == "search-sparse", (name, found.route)
        assert not found.is_spec and found.spec is None, name
        assert found.omega, f"{name}: the fallback returned nothing"

        spin = max((two_s for om in found.omega.values()
                    for two_s in spin_decompose(om)), default=0)
        assert len(found.omega) < len(baseline), (
            name, len(found.omega), len(baseline))
        assert spin <= baseline_spin, (name, spin, baseline_spin)

        assert extract_spec_from_quiver(pairing, B3, cutoff=4) is None, name
        quiver = BPSQuiver(B3, exchange_matrix=pairing, ambient_pairing=pairing)
        assert quiver.find_negating_sequence(max_depth=10) is None, name


# ----------------------------------------------------------------------
# the acyclic short-circuit
# ----------------------------------------------------------------------

ACYCLIC = (
    ("pentagon", PENTAGON, B2),
    ("pure SU(2)", KRONECKER2, B2),
    ("A3 linear", [[0, 1, 0], [-1, 0, 1], [0, -1, 0]], B3),
    ("D4 star", [[0, 1, 1, 1], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0]],
     B4),
)


def test_the_cycle_rank_law():
    """`cycle_n` factors as `2(n−1)` spin-0 factors topping out at degree `n−1`.

    Read off ranks 3–6 and then confirmed out of sample at ranks 7 and 8.
    Pinned here at the cheap ranks only — 7 and 8 cost minutes, which belongs in
    the probes in the source repository, not in a suite.

    The law's *structural* half (`2(n−1)` factors, top degree `n−1`) holds at
    every rank measured.  Its *cutoff* half did not: the original reading also
    predicted `D ≥ n+1` suffices, and cycle6 refuted exactly that — at `D = 7`
    the content is the right shape and clears the margin, yet is wrong at degree
    8, so rank 6 needs `D ≥ n+2`.  That is the counterexample which retired the
    margin as a certificate (`test_the_cycle6_counterexample_that_refuted_the_
    margin`), and the reason the cutoff assertions below are pinned only at the
    ranks where they were verified against a wider cone.

    Worth a test rather than a note because it is the one structural prediction
    the search makes about a family, so a change that quietly broke it would
    otherwise show up only as "different but still valid" output.
    """
    for rank in (3, 4, 5):
        pairing = [[0] * rank for _ in range(rank)]
        for i in range(rank):
            j = (i + 1) % rank
            pairing[i][j], pairing[j][i] = 1, -1
        nodes = [tuple(1 if j == k else 0 for j in range(rank))
                 for k in range(rank)]

        found = find_simple_factorisation(pairing, nodes, rank + 1)
        assert found.is_spec, rank
        assert len(found.spec) == 2 * (rank - 1), (rank, len(found.spec))
        assert found.top_degree == rank - 1, (rank, found.top_degree)
        assert found.margin >= 2, (rank, found.margin)
        assert rebuilds(pairing, nodes, rank + 1, found.spec), rank

        # ...and the cutoff the law names is the smallest that certifies: one
        # lower leaves margin 1, which STOP_MARGIN refuses.
        tight = find_simple_factorisation(pairing, nodes, rank)
        assert not tight.is_spec, rank


def test_a_margin_of_one_is_not_enough_to_certify_stopping():
    """The measured law behind `STOP_MARGIN`, and the case that forced it.

    A factorisation certified at cutoff `D` can be **wrong at `D+1`**: the
    rebuild is in-cone, so it certifies only what it saw.  At 4-cycle(1) with
    `D = 4` the search finds a 6-factor spin-0 content topping out at degree 3 —
    one empty degree — which rebuilds `S` correctly up to 4 and disagrees from
    degree 5 on.  The valid factorisation differs from it *only by the order of
    two adjacent charges*, so nothing about the content itself gives it away.

    Measured over 84 checks (find at `D`, re-verify at `D+1, D+2, D+3`): every
    failure had margin `≤ 1`, every margin `≥ 2` passed.  Hence `STOP_MARGIN = 2`
    — and hence this test, which pins both halves: the margin-1 case is refused,
    and the margin-2 case is accepted and really does hold at larger cutoffs.
    """
    four_cycle = [[0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0]]

    tight = find_simple_factorisation(four_cycle, B4, 4)
    assert tight.spec is not None, "the candidate itself should still be found"
    assert tight.margin == 1, tight.margin
    assert not tight.stopped and not tight.is_spec, (
        "a margin of 1 is being accepted again; it is measurably unsound")
    assert not rebuilds(four_cycle, B4, 5, tight.spec), (
        "the margin-1 candidate now survives at D+1 — re-derive STOP_MARGIN")

    ample = find_simple_factorisation(four_cycle, B4, 5)
    assert ample.margin >= 2 and ample.stopped and ample.is_spec
    for cutoff in (5, 6, 7):
        assert rebuilds(four_cycle, B4, cutoff, ample.spec), cutoff


def test_acyclic_short_circuits_to_the_optimal_node_charge_spec():
    """No search is run where the strip already attains the provable floor.

    The floor is a theorem, not a measurement (`minimum_factors`): at cone degree
    1 the accumulated product is empty, so `Ω` is forced nonzero at every node
    whatever the order, and no factorisation has fewer than `rank` factors.  The
    strip order attains it, so searching there could only rediscover it — 33.7 s
    at A6 before this short-circuit, milliseconds after.
    """
    for name, pairing, nodes in ACYCLIC:
        found = find_simple_factorisation(pairing, nodes, 5)
        assert found.route == "strip", name
        assert found.states_examined == 0, f"{name}: it searched anyway"
        assert found.is_spec and found.optimal, name
        assert len(found.spec) == len(nodes), (name, found.spec)
        assert set(found.spec) == {tuple(g) for g in nodes}, (name, found.spec)
        assert rebuilds(pairing, nodes, 5, found.spec), name


def test_the_short_circuit_returns_the_placement_order_not_the_dict_order():
    """A spec is an ORDERED product, and the two orders genuinely differ.

    `Ω` is filled in cone order (degree, then lexicographic) while the factors
    are placed by the engine's key; at the pentagon those are exact reverses.
    Reading the dict order returned `[(0,1),(1,0)]`, whose `E_𝖖` product is a
    different element — caught only by the independent rebuild, which is why the
    rebuild is inside the short-circuit and not merely in this file.
    """
    found = find_simple_factorisation(PENTAGON, B2, 6)
    assert found.spec == [(1, 0), (0, 1)], found.spec
    assert not rebuilds(PENTAGON, B2, 6, list(reversed(found.spec))), (
        "the reversed order rebuilds too, so this test cannot detect the bug "
        "it exists for")


# ----------------------------------------------------------------------
# the order is reusable, which is the point of returning one
# ----------------------------------------------------------------------

def test_the_found_order_replays_through_the_engine():
    """`piece_key()` hands the winning order back to `bps_factor_spectrum`.

    Both `S` and `Ω` must come back identical: `S` alone would prove nothing
    (it is order-independent, so *any* key reproduces it), and `Ω` is precisely
    the order-dependent half — so the content check is the whole test.
    """
    for name, pairing, nodes, degree, _expected in CORED:
        found = find_simple_factorisation(pairing, nodes, degree)
        replay = BPSFactorSpectrum(pairing, nodes, degree,
                             piece_key=found.piece_key())
        replay.run()
        assert replay.spectrum_generator() == reference_S(pairing, nodes,
                                                          degree), name
        assert replay.multiplicities() == found.omega, name
        assert replay.verify_leading_data() == [], name


# ----------------------------------------------------------------------
# the prunes are exact; the search width is not, and says so
# ----------------------------------------------------------------------

def test_the_found_order_is_usable_at_a_LARGER_cutoff():
    """`piece_key()` must be a total order on the whole cone, not just what was seen.

    Replaying the order at the cutoff that produced it exercises nothing: every
    pair is known.  The method exists to be carried to a *larger* cutoff, and
    there it meets pairs the search never placed — which is where a key returning
    a bare `int` for placed pairs and a tuple for unplaced ones makes the
    engine's `bisect` compare `int` against `tuple` and raise.  So the test runs
    it at a strictly larger cone.

    ⚠ It asserts only that the key is USABLE, not that the replayed
    factorisation is still the best one — measured, it often is not, because an
    order says nothing about where pairs it never saw belong.
    """
    pairing, nodes = three_cycle(1, 1, 1), B3
    found = find_simple_factorisation(pairing, nodes, 4)
    assert found.is_spec

    for cutoff in (5, 6, 7):
        replay = BPSFactorSpectrum(pairing, nodes, cutoff,
                             piece_key=found.piece_key())
        replay.run()                      # must not raise
        assert replay.verify_leading_data() == [], cutoff


def test_the_spin_prune_excludes_nothing_a_wider_search_finds():
    """`require_spin_zero` is a DECISION, not a heuristic.

    `Ω` is forced by the strictly-lower-degree factors, so a branch that has
    emitted a nonzero spin can never become spin-0 later — cutting it loses no
    spin-0 factorisation.  Checked by running with the prune off and confirming
    the answer does not improve.
    """
    for name, pairing, nodes, degree, expected in CORED[:2]:
        pruned = find_simple_factorisation(pairing, nodes, degree)
        loose = find_simple_factorisation(pairing, nodes, degree,
                                          require_spin_zero=False)
        assert pruned.is_spec, name
        assert not loose.is_spec or len(loose.spec) >= expected, (
            f"{name}: the unpruned search found a SHORTER spin-0 "
            f"factorisation, so the prune is not exact")


def test_exhaustive_is_honest_about_the_search_width():
    """A capped run must not claim completeness.

    `keep_per_degree` is the one heuristic in the module: carrying only that many
    candidate orders forward from each cone degree can
    drop the branch that would have won, so `exhaustive` exists to keep a
    negative result from being read as a proof.  Keeping only 1 on a quiver that
    needs more must report `False`.
    """
    narrow = find_simple_factorisation(three_cycle(1, 1, 1), B3, 5, keep_per_degree=1,
                                       branch_cap=2)
    assert not narrow.exhaustive

    # ...and the short-circuit, which searches nothing, is genuinely complete.
    assert find_simple_factorisation(PENTAGON, B2, 5).exhaustive


def test_the_default_width_is_the_measured_one_and_widening_buys_nothing():
    """The default `keep_per_degree` is a measurement, pinned.

    Widening to 24 costs ~10× on the expensive quivers and returns the same spec
    everywhere measured, so the default is small on purpose.
    """
    for name, pairing, nodes, degree, expected in CORED:
        default = find_simple_factorisation(pairing, nodes, degree)
        wide = find_simple_factorisation(pairing, nodes, degree, keep_per_degree=24)
        assert default.is_spec and wide.is_spec, name
        assert len(default.spec) == len(wide.spec) == expected, (
            name, default.spec, wide.spec)


def test_too_narrow_a_search_really_does_lose_an_answer():
    """The positive control for `exhaustive`, and the default's lower edge.

    A negative result from this search means "not found", not "does not exist",
    and that distinction is only worth reporting if it can actually bite.  It
    bites two different ways, and both are pinned:

    * at `keep_per_degree=2` the SU(3)-cyclic 6-factor spec — which the default
      finds, and which two independent routes confirm exists — is missed
      entirely, and the run correctly reports `exhaustive=False` while doing so;
    * at the same width the 5-cycle still *returns* a spin-0 factorisation, and
      that one is not merely worse, it is **wrong** — see
      `test_a_narrow_search_can_return_an_in_cone_impostor` for the measurement.
      A too-narrow search does not only fail loudly.

    ⚠ The first bullet is asserted with `escalate=False`, because it is a claim
    about the WIDTH and escalation now repairs exactly this case — recovering the
    6-factor spec from width 2, which is the whole point of `_escalate`.  Both
    halves are pinned: the raw width loses it, and escalation gets it back.
    """
    lost = find_simple_factorisation(SU3_CYCLIC, B4, 4, keep_per_degree=2,
                                     escalate=False)
    assert not lost.is_spec, "keep_per_degree=2 no longer loses it; re-tune"
    assert not lost.exhaustive, "it lost an answer while claiming completeness"

    rescued = find_simple_factorisation(SU3_CYCLIC, B4, 4, keep_per_degree=2)
    assert rescued.is_spec and len(rescued.spec) == 6, "escalation lost its case"
    assert rescued.route == "search-wide"
    assert rebuilds(SU3_CYCLIC, B4, 4, rescued.spec)

    found = find_simple_factorisation(SU3_CYCLIC, B4, 4)
    assert found.is_spec and len(found.spec) == 6
    assert rebuilds(SU3_CYCLIC, B4, 4, found.spec)

    # Cutoff 6, not 5: at 5 the 5-cycle's factorisation clears only a margin of
    # 1 above its top factor, which `STOP_MARGIN` refuses — correctly, since a
    # margin-1 acceptance is measurably unsound.
    default = find_simple_factorisation(FIVE_CYCLE, NODES5, 6)
    assert default.is_spec and len(default.spec) == 8
    assert rebuilds(FIVE_CYCLE, NODES5, 6, default.spec)


def test_a_narrow_search_can_return_an_in_cone_impostor():
    """Rebuilding `S` in-cone does NOT certify a factorisation.  Measured twice.

    Both counterexamples were accepted by `STOP_MARGIN` alone and are refused by
    `CONFIRM_EXTRA`, and the second is the sharp one because it holds two
    answers to the *same* quiver at the *same* top degree side by side:

    * cycle6 at `D = 7`: a 10-factor spin-0 content topping out at degree 5
      (margin 2) rebuilds `S` on the whole cone up to 7 and is wrong at 8.  This
      is what refuted the 84-check derivation of `STOP_MARGIN = 2`, which had
      covered seven quivers not including this one.
    * the 5-cycle at `D = 6`: `keep_per_degree=2` returns 9 factors and the
      default returns 8, **both** topping out at degree 4, so no margin can tell
      them apart.  The 9-factor product is exact on the cone up to 6 and wrong
      from 7 on; the 8-factor one holds at 6, 7 and 8.

    So the narrow search's failure mode is not "returns something suboptimal" —
    it is "returns an impostor that agrees with `S` everywhere you looked".  A
    proxy for termination cannot see that; only a rebuild on a strictly wider
    cone can, which is why `is_spec` now requires one.
    """
    # `confirm_extra=0` reproduces the margin-only rule that was refuted, and
    # `escalate=False` keeps the restart ladder from replacing the failing draw.
    impostor = find_simple_factorisation(FIVE_CYCLE, NODES5, 6,
                                         keep_per_degree=2, confirm_extra=0,
                                         escalate=False)

    assert impostor.is_spec and len(impostor.spec) == 9, (
        "the margin-only rule no longer accepts the impostor; re-measure")
    assert impostor.top_degree == 4, "same top degree as the valid answer"
    assert rebuilds(FIVE_CYCLE, NODES5, 6, impostor.spec), "exact in-cone …"
    assert not rebuilds(FIVE_CYCLE, NODES5, 7, impostor.spec), "… wrong at 7"

    # and with the certificate on, the same search refuses it.
    refused = find_simple_factorisation(FIVE_CYCLE, NODES5, 6, keep_per_degree=2,
                                        escalate=False)
    assert refused.stopped and not refused.is_spec, (
        "CONFIRM_EXTRA no longer catches the impostor")


def test_the_cycle6_counterexample_that_refuted_the_margin():
    """cycle6 is refused at `D = 7` and granted at `D = 8`.

    The margin-2 rule granted it at 7, where its 10-factor content is exact on
    the cone and wrong one degree up.  Pinned here so a future retuning of
    `STOP_MARGIN` cannot silently re-admit it.
    """
    pairing = [[0] * 6 for _ in range(6)]
    for i in range(6):
        pairing[i][(i + 1) % 6] = 1
        pairing[(i + 1) % 6][i] = -1
    nodes = [tuple(1 if j == k else 0 for j in range(6)) for k in range(6)]

    # escalate=False: with restarts on, this quiver's failed draw is retried and
    # a VALID 10-factor spec is found at the next seed — which is the feature
    # working, but it would hide the counterexample this test exists to pin.
    tight = find_simple_factorisation(pairing, nodes, 7, escalate=False)
    assert tight.stopped and tight.top_degree == 5, "the margin-2 screen passes"
    assert not tight.is_spec, "but the wider-cone rebuild refuses it"

    wider = find_simple_factorisation(pairing, nodes, 8)
    assert wider.is_spec and len(wider.spec) == 10
    assert rebuilds(pairing, nodes, 8, wider.spec)


# ----------------------------------------------------------------------
# the pieces
# ----------------------------------------------------------------------

def test_trace_normal_form_merges_commuting_reorderings_only():
    """Placements form a trace monoid: factors commute iff `⟨γ,δ⟩ = 0`.

    The normal form must merge exactly the reorderings that leave the product
    alone — so a commuting swap collapses, and a non-commuting one does not.
    """
    commuting = lambda a, b: True
    never = lambda a, b: False
    left, right = ((1, 0), 0, 1), ((0, 1), 0, 1)

    assert (_normal_form([left, right], commuting)
            == _normal_form([right, left], commuting))
    assert (_normal_form([left, right], never)
            != _normal_form([right, left], never))

    # And the real predicate: on the pentagon the two nodes pair to 1, so they
    # do NOT commute and their two orders are genuinely different states.
    search = FactorOrderSearch(PENTAGON, B2, 4)
    assert not search._commutes((1, 0), (0, 1))
    assert search._commutes((1, 0), (1, 0))      # ⟨γ,γ⟩ = 0, always


def test_minimum_factors_is_the_rank_under_bps_leading_data():
    for pairing, nodes in ((PENTAGON, B2), (three_cycle(1, 1, 1), B3),
                           (SU3_CYCLIC, B4)):
        assert FactorOrderSearch(pairing, nodes, 4).minimum_factors() == len(nodes)

    # With a leading datum switched off at one node that node needs no factor,
    # so the bound drops — it is a statement about the leading data, not a
    # constant.
    def one_node_only(charge):
        return -1 if charge == (1, 0) else 0

    reduced = FactorOrderSearch(PENTAGON, B2, 4, leading_data=one_node_only)
    assert reduced.minimum_factors() == 1


def test_default_cost_counts_terms_and_dislikes_spin():
    """The cost is *the number of terms, with a dislike for `s > 0`* — pinned.

    Three properties, each a claim the search depends on:

    1. **Fewer terms is better**, all else equal.
    2. **Spin is disliked but not forbidden** — a spin-carrying content loses to
       a slightly larger spin-0 one, and beats a *much* larger one.  That is the
       difference between a dislike and a hard prune, and the search offers both
       (`require_spin_zero` is the prune).
    3. **The terms forced at the next degree count too**, because they are forced
       — a candidate that has committed itself to many is ranked worse than one
       that has not, at equal placed size.  This is what the earlier
       placed-terms-only version missed, and it cost the SU(3)-cyclic spec.
    """
    def content(terms, spin_weight=0, negatives=0, forced_next=0, factors=1):
        return Content(factors=factors, terms=terms, spin_weight=spin_weight,
                       max_spin_doubled=2 if spin_weight else 0,
                       negatives=negatives, forced_next=forced_next)

    assert default_cost(content(4)) < default_cost(content(6))

    # A dislike, not a prohibition: one unit of spin is worth SPIN_PENALTY terms.
    assert default_cost(content(5)) < default_cost(content(4, spin_weight=1))
    assert default_cost(content(4, spin_weight=1)) < default_cost(content(40))

    # Forced-next is inside the count, not a tiebreak after it.
    assert (default_cost(content(4, forced_next=9))
            > default_cost(content(6, forced_next=0)))

    # Negatives break ties but never outrank the count.
    assert default_cost(content(4, negatives=1)) < default_cost(content(9))
    assert default_cost(content(4)) < default_cost(content(4, negatives=1))


# ----------------------------------------------------------------------
# factoring an S you already have
# ----------------------------------------------------------------------

def test_a_computed_S_supplies_its_own_leading_data():
    """`expansion(S[γ])[1] == BPSFactorSpectrum.target(k)` at every cone charge.

    This is the identity `simplify_factorisation` rests on, so it is checked
    directly rather than only through its consumer.  The SU(2) case is the one
    that matters: at weak coupling the W boson carries `Ω = 𝖖⁻¹ + 𝖖`, so an
    implementation that read the whole element instead of the `𝖖¹` coefficient
    would pass the spin-free quivers and fail here.
    """
    for pairing, nodes, degree in ((PENTAGON, B2, 4), (KRONECKER2, B2, 5),
                                   (three_cycle(1, 1, 1), B3, 4)):
        built = BPSFactorSpectrum(pairing, nodes, degree, order="lex")
        built.run()
        leading = leading_data_from_spectrum(built.spectrum_generator())
        for k in built.cone:
            assert leading(built.charge(k)) == built.target(k), (pairing, k)


def test_factoring_a_supplied_S_agrees_with_factoring_from_scratch():
    """The two front doors are the same search entered from opposite ends.

    `find_simple_factorisation` takes the quiver and derives `S`;
    `simplify_factorisation` takes `S` and derives the leading data.  They must
    land on the identical factorisation, and do — including on SU(3)-cyclic,
    where the answer is 6 factors and not the floor.
    """
    for pairing, nodes, degree in ((PENTAGON, B2, 5), (KRONECKER2, B2, 5),
                                   (three_cycle(1, 1, 1), B3, 5),
                                   (SU3_CYCLIC, B4, 4)):
        built = BPSFactorSpectrum(pairing, nodes, degree, order="lex")
        built.run()
        scratch = find_simple_factorisation(pairing, nodes, degree)
        supplied = simplify_factorisation(built.spectrum_generator(),
                                          pairing, nodes, degree)
        assert supplied.spec == scratch.spec, (pairing, degree)
        assert supplied.is_spec == scratch.is_spec


def test_factoring_refuses_an_S_that_is_not_this_quivers():
    """The verify pass, and the measurement that makes it load-bearing.

    A re-factorisation tool takes `S` on trust, so the failure it invites is
    factoring the wrong element.  It is not hypothetical: handed 3-cycle(2,1,1)'s
    `S` against 3-cycle(1,1,1)'s quiver, `verify=False` returns a **valid
    4-factor spec** — of 3-cycle(1,1,1), silently, because the two share their
    leading data and only the pairing differs.  Nothing in the result says so.

    So `verify=True` is the default and refuses both a foreign `S` (20 charges
    differ) and one perturbed at a single charge.
    """
    host, other = three_cycle(1, 1, 1), three_cycle(2, 1, 1)
    built = BPSFactorSpectrum(host, B3, 5, order="lex")
    built.run()
    good = built.spectrum_generator()
    foreign = BPSFactorSpectrum(other, B3, 5, order="lex")
    foreign.run()

    nudged = dict(good)
    where = min(nudged)
    nudged[where] = nudged[where] + HabiroElement.one()

    for bad in (foreign.spectrum_generator(), nudged):
        try:
            simplify_factorisation(bad, host, B3, 5)
        except ValueError as err:
            assert "not this quiver's crystalline S" in str(err)
        else:
            raise AssertionError("a wrong S was accepted")

    # and the bypass really is a bypass — this is the silent answer it gives.
    slipped = simplify_factorisation(foreign.spectrum_generator(), host, B3, 5,
                                     verify=False)
    assert slipped.is_spec and len(slipped.spec) == 4, (
        "the silent-wrong-answer measurement changed; re-check the docstring")
    assert rebuilds(host, B3, 5, slipped.spec), (
        "and it is a genuine factorisation — of the HOST, which is the trap")


def test_a_single_run_is_a_DRAW_and_the_seed_decides_what_it_finds():
    """At one fixed width the outcome spans no-spec / valid spec / impostor.

    The measurement behind `_escalate`, and a correction to how this was first
    read.  `_positions` samples once the position count exceeds its cap, so a run
    is a *draw* — and the draws are not close to each other.  cycle6 at `D = 7`,
    at the **default** width, varying only the seed:

        seed 1 -> certified 10-factor spec
        seed 2 -> nothing found
        seed 3 -> certified 10-factor spec
        seed 5 -> a 10-factor content exact to degree 7 and WRONG at 8

    All three outcomes at one width.  The first reading of this data attributed
    it to `keep_per_degree` — "width 2 beats width 4 here" — and that was a
    sampling artifact: width 4 wins too, at other seeds.  It was caught only
    because two runs of the same quantity disagreed, which is why the module
    escalates by RESTARTING rather than by widening.

    What is *not* seed-dependent, and is asserted separately below: the refused
    draw really is exact in-cone and really is wrong above it.  That is a fact
    about that factorisation, however it was drawn.
    """
    pairing = [[0] * 6 for _ in range(6)]
    for i in range(6):
        pairing[i][(i + 1) % 6] = 1
        pairing[(i + 1) % 6][i] = -1
    nodes = [tuple(1 if j == k else 0 for j in range(6)) for k in range(6)]

    outcomes = set()
    for seed in (1, 2, 3, 5):
        run = FactorOrderSearch(pairing, nodes, 7, escalate=False,
                                seed=seed)._run()
        if run.spec is None:
            outcomes.add("none")
        elif run.is_spec:
            outcomes.add("certified")
            # Verified only as far as `confirmed_to` — NOT to 9.  Asserting 9
            # here is what exposed the third refuted constant; see
            # `test_a_certified_factorisation_can_still_fail_above_its_depth`.
            assert run.confirmed_to == 7 + factor_order_search.CONFIRM_EXTRA
            assert rebuilds(pairing, nodes, run.confirmed_to, run.spec), seed
        else:
            outcomes.add("refused")
            # the refused draw: exact in its own cone, wrong one degree up.
            assert rebuilds(pairing, nodes, 7, run.spec), seed
            assert not rebuilds(pairing, nodes, 8, run.spec), seed

    assert outcomes == {"none", "certified", "refused"}, (
        f"the seed spread changed: {outcomes}; re-measure before trusting any "
        f"single-run claim about this quiver")


def test_a_certified_factorisation_can_still_fail_above_its_depth():
    """The third refuted constant: `CONFIRM_EXTRA = 1` is not sufficient either.

    cycle6 at `D = 7`, **seed 1**, is granted `is_spec` — so the wider-cone check
    ran and the product was verified exact at degree 8 — and it is **wrong at 9
    and 10**.  Same shape as the two `STOP_MARGIN` refutations before it: a check
    that stops at a fixed depth is defeated by a counterexample living past that
    depth.

    So this test does NOT assert that some larger constant works; that is the
    error being pinned, three times over.  It asserts the honest reading: the
    result is verified exactly as far as `confirmed_to` and no further, which is
    why that field exists and why `is_spec` is documented as relative to it.

    ⚠ Seed-sensitive by construction (A55) — a run is a draw, and this is the
    draw that fails.  `escalate=False` keeps the restart ladder from replacing it.
    """
    pairing = [[0] * 6 for _ in range(6)]
    for i in range(6):
        pairing[i][(i + 1) % 6] = 1
        pairing[(i + 1) % 6][i] = -1
    nodes = [tuple(1 if j == k else 0 for j in range(6)) for k in range(6)]

    run = FactorOrderSearch(pairing, nodes, 7, escalate=False, seed=1)._run()
    assert run.is_spec, "seed 1 no longer produces the certified draw; re-measure"
    assert run.confirmed_to == 8, run.confirmed_to

    assert rebuilds(pairing, nodes, 7, run.spec), "exact in its own cone"
    assert rebuilds(pairing, nodes, 8, run.spec), "and at the depth it paid for"
    assert not rebuilds(pairing, nodes, 9, run.spec), (
        "CONFIRM_EXTRA=1 is no longer refuted here — re-measure before relaxing "
        "anything that depends on it")

    # Paying for more depth refuses it, which is the point of the parameter.
    deeper = FactorOrderSearch(pairing, nodes, 7, escalate=False, seed=1,
                               confirm_extra=2)._run()
    assert not deeper.is_spec, "confirm_extra=2 should reject this draw"


def test_escalation_fires_only_on_failure_and_is_certified():
    """`_escalate` must buy answers without touching successes or the guard.

    Both halves matter.  A search that already found a spec must return the same
    one by the same route — escalation is not allowed to cost time on the ~97 %
    of cases that succeed first time.  And anything the wider pass returns goes
    through the identical `CONFIRM_EXTRA` certificate, so widening can only find
    a right answer, never introduce a wrong one.
    """
    for pairing, nodes, degree, length in ((three_cycle(1, 1, 1), B3, 5, 4),
                                           (SU3_CYCLIC, B4, 4, 6)):
        plain = find_simple_factorisation(pairing, nodes, degree)
        assert plain.route == "search" and len(plain.spec) == length
        off = find_simple_factorisation(pairing, nodes, degree, escalate=False)
        assert off.spec == plain.spec, "escalation changed a success"

    # A quiver where the default width finds nothing and a wider one does.  The
    # 5-cycle at cutoff 5 is the cheap stand-in: at width 1 the beam is too
    # narrow to reach its 8-factor spec, and escalation recovers it.
    starved = find_simple_factorisation(FIVE_CYCLE, NODES5, 6, keep_per_degree=1,
                                        escalate=False)
    rescued = find_simple_factorisation(FIVE_CYCLE, NODES5, 6, keep_per_degree=1)
    if starved.spec is None:
        assert rescued.spec is not None, "escalation bought nothing"
        assert rescued.route == "search-wide"
        assert rebuilds(FIVE_CYCLE, NODES5, 6, rescued.spec)


def test_replaying_an_order_at_its_own_cutoff_is_a_fixed_point():
    """`_replay` of a result's own `piece_key()` reproduces that result exactly.

    The direct test of the replay, which the seeded-search tests only exercise
    end to end.  At the cutoff that produced it every pair is known to the key,
    so the replay should place every piece back where it was, survive to the top
    (`resume > degree_cap`), and hand back the same factorisation — a fixed
    point.  Anything less means the bisect placement is not reconstructing the
    seed's interleaving, which would show up only as a slower search, never as a
    wrong answer, and so would otherwise go unnoticed.
    """
    for pairing, nodes, degree in ((three_cycle(1, 1, 1), B3, 5),
                                   (SU3_CYCLIC, B4, 4)):
        found = find_simple_factorisation(pairing, nodes, degree)
        search = FactorOrderSearch(pairing, nodes, degree)
        state, resume = search._replay(found.piece_key())

        assert resume > search.degree_cap, (pairing, "the seed's own order died")
        assert [k for k, _s, _a in state.placed] == [
            tuple(k) for k, _s, _a in found._pieces], (pairing, "placement moved")

        replayed = find_simple_factorisation(pairing, nodes, degree,
                                             seed_order=found.piece_key())
        assert replayed.spec == found.spec and replayed.is_spec == found.is_spec


def test_a_seed_order_is_donated_but_never_trusted():
    """`seed_order` must change cost, never correctness.

    Two things are asserted, and the second is the one that matters: a seed from
    the cutoff below reaches the *same* factorisation as the cold search, and a
    **deliberately bad** seed still does.  The replay stops donating at the first
    degree where the seed's forced `Ω` goes bad, and the certificate runs either
    way, so a stale seed can only cost time.
    """
    low = find_simple_factorisation(SU3_CYCLIC, B4, 4)
    cold = find_simple_factorisation(SU3_CYCLIC, B4, 5)
    warm = find_simple_factorisation(SU3_CYCLIC, B4, 5,
                                     seed_order=low.piece_key())
    assert warm.spec == cold.spec and warm.is_spec == cold.is_spec

    # A seed from a DIFFERENT quiver — no reason for its order to be any good.
    stale = find_simple_factorisation(three_cycle(1, 1, 1), B3, 4).piece_key()
    misled = find_simple_factorisation(SU3_CYCLIC, B4, 5,
                                       seed_order=lambda k, s: stale(k[:3], s))
    assert misled.spec is not None and rebuilds(SU3_CYCLIC, B4, 5, misled.spec)


def test_the_older_extractors_default_cutoff_verified_two_degrees_short():
    """`Theory.extract_spec_insert`'s old `CONE − 2` default, pinned (A54).

    Found by benchmarking against it, not by looking for it: at `CONE = 4` on
    SU(3)-cyclic the old default returned **5 factors** to the search's 6 — and 5
    is not a shorter answer, since the 5-factor product reproduces `S` at no cone
    degree at all.  The DFS keeps the *shortest* candidate, so the unverified two
    degrees are exactly where a too-short answer survives; the bias is towards
    looking better than the truth.

    Both directions are asserted so neither the defect nor the fix can drift.
    """
    old = Theory("t", SU3_CYCLIC, B4, CONE=4)
    stale = old.extract_spec_insert(reference_S(SU3_CYCLIC, B4, 4), cutoff=2)
    assert stale is not None and len(stale) == 5
    assert not rebuilds(SU3_CYCLIC, B4, 4, stale), (
        "the old default's answer is valid after all; re-check A54")
    assert not rebuilds(SU3_CYCLIC, B4, 3, stale), "wrong even one degree down"

    fixed = Theory("t", SU3_CYCLIC, B4, CONE=4).extract_spec_insert(
        reference_S(SU3_CYCLIC, B4, 4))
    assert len(fixed) == 6 and rebuilds(SU3_CYCLIC, B4, 4, fixed)
    # …and it now agrees with the search on the LENGTH, which is the quantity
    # the defect corrupted.  Not on the charges: specs are not unique.
    assert len(fixed) == len(find_simple_factorisation(SU3_CYCLIC, B4, 4).spec)


def test_the_sparse_fallback_is_now_independently_verified():
    """The one route whose output rested on the search's own arithmetic.

    `"search-sparse"` fires where no spin-0 factorisation exists, and its content
    CARRIES SPIN — so the spec-only rebuild could not touch it, `spec` is `None`
    and `is_spec` is `False`, and none of that said whether the answer was right.
    It became checkable when the Nahm shift gained its diagonal `n²`
    (`nahm_local.general_nahm_habiro`), and `Result.content_verified` now reports
    it.

    3-cycle(3,3,3) is the case worth having: its content carries spin up to
    `2s = 5`, which is exactly what the spec route cannot express.
    """
    for pairing, degree, top_spin in ((three_cycle(2, 2, 2), 4, 1),
                                      (three_cycle(3, 3, 3), 4, 5)):
        found = find_simple_factorisation(pairing, B3, degree)
        assert found.route == "search-sparse", (pairing, found.route)
        assert found.spec is None and not found.is_spec
        assert max(s for _c, s, _a in found._pieces) == top_spin, (
            "the fallback's spin content moved; re-check what is being verified")
        assert found.content_verified is True, (pairing, "sparse route unverified")

    # …and the spec routes leave it None, since `is_spec` already subsumes it.
    assert find_simple_factorisation(PENTAGON, B2, 5).content_verified is None


def test_the_content_route_agrees_with_the_spec_route_and_rejects_damage():
    """`_rebuilds_content` is `_rebuilds` without the spin-0 restriction.

    Positive: the two agree wherever both apply — otherwise the general route
    would be verifying something other than what the spec route verifies, and
    the sparse-route result above would mean nothing.  Negative: bumping a
    multiplicity or dropping a factor must be caught, so a `True` is a check and
    not a tautology.
    """
    for pairing, nodes, degree in ((PENTAGON, B2, 5), (KRONECKER2, B2, 5),
                                   (three_cycle(1, 1, 1), B3, 5),
                                   (SU3_CYCLIC, B4, 4)):
        found = find_simple_factorisation(pairing, nodes, degree)
        search = FactorOrderSearch(pairing, nodes, degree)
        target = reference_S(pairing, nodes, degree)
        assert search._rebuilds(found.spec, target)
        assert search._rebuilds_content(found._pieces, target), (pairing, degree)

    search = FactorOrderSearch(three_cycle(1, 1, 1), B3, 5)
    found = find_simple_factorisation(three_cycle(1, 1, 1), B3, 5)
    target = reference_S(three_cycle(1, 1, 1), B3, 5)
    bumped = list(found._pieces)
    bumped[0] = (bumped[0][0], bumped[0][1], bumped[0][2] + 1)
    assert not search._rebuilds_content(bumped, target), "a bumped multiplicity passed"
    assert not search._rebuilds_content(found._pieces[1:], target), "a dropped factor passed"


TESTS = [
    test_found_specs_rebuild_S_independently,
    test_the_search_beats_the_default_order_where_the_strip_has_no_answer,
    test_agrees_with_the_two_independent_spec_routes,
    test_no_finite_factorisation_where_none_exists,
    test_the_sparse_fallback_returns_something_usable_where_no_spec_exists,
    test_the_cycle_rank_law,
    test_a_margin_of_one_is_not_enough_to_certify_stopping,
    test_acyclic_short_circuits_to_the_optimal_node_charge_spec,
    test_the_short_circuit_returns_the_placement_order_not_the_dict_order,
    test_the_found_order_replays_through_the_engine,
    test_the_found_order_is_usable_at_a_LARGER_cutoff,
    test_the_spin_prune_excludes_nothing_a_wider_search_finds,
    test_exhaustive_is_honest_about_the_search_width,
    test_the_default_width_is_the_measured_one_and_widening_buys_nothing,
    test_too_narrow_a_search_really_does_lose_an_answer,
    test_a_narrow_search_can_return_an_in_cone_impostor,
    test_the_cycle6_counterexample_that_refuted_the_margin,
    test_trace_normal_form_merges_commuting_reorderings_only,
    test_minimum_factors_is_the_rank_under_bps_leading_data,
    test_default_cost_counts_terms_and_dislikes_spin,
    test_a_computed_S_supplies_its_own_leading_data,
    test_factoring_a_supplied_S_agrees_with_factoring_from_scratch,
    test_factoring_refuses_an_S_that_is_not_this_quivers,
    test_a_single_run_is_a_DRAW_and_the_seed_decides_what_it_finds,
    test_a_certified_factorisation_can_still_fail_above_its_depth,
    test_escalation_fires_only_on_failure_and_is_certified,
    test_replaying_an_order_at_its_own_cutoff_is_a_fixed_point,
    test_a_seed_order_is_donated_but_never_trusted,
    test_the_older_extractors_default_cutoff_verified_two_degrees_short,
    test_the_sparse_fallback_is_now_independently_verified,
    test_the_content_route_agrees_with_the_spec_route_and_rejects_damage,
]


if __name__ == "__main__":
    for fn in TESTS:
        fn()
        print(f"  PASS: {fn.__name__}")
    print(f"\nAll {len(TESTS)} factor-order-search tests passed.")
