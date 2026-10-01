"""Tests for `spec_acceptance`: the axiom check that accepts a
spec — the BPS quiver constraint on the truncated cone plus agreement with
the crystalline `S` — with its positive AND negative controls (the four wrong
pentagon specs of a probe in the source repository, the reversed strip).

Run:  `python3 run_tests.py`
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

from spec_acceptance import (  # noqa: E402
    accept_spec, content_record, crystalline_spectrum, is_green_sequence,
    leading_data_violations, spec_from_components, spec_product, strip_spec,
)

PENTAGON = [[0, 1], [-1, 0]]
KRONECKER2 = [[0, 2], [-2, 0]]
THREE_CYCLE = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
A3 = [[0, 1, 0], [-1, 0, 1], [0, -1, 0]]
MARKOV = [[0, 2, -2], [-2, 0, 2], [2, -2, 0]]


def test_pentagon_genuine_specs_accepted():
    for spec in ([(1, 0), (0, 1)], [(0, 1), (1, 1), (1, 0)]):
        a = accept_spec(PENTAGON, spec, 6)
        assert a.ok and a.leading_ok and a.matches_crystalline, a.as_dict()


def test_pentagon_wrong_specs_rejected():
    wrong = {
        "reversed": [(0, 1), (1, 0)],
        "node missing": [(1, 0)],
        "spurious charge": [(1, 0), (0, 1), (1, 1)],
        "node repeated": [(1, 0), (1, 0), (0, 1)],
    }
    for name, spec in wrong.items():
        a = accept_spec(PENTAGON, spec, 6)
        assert not a.ok and not a.leading_ok, (name, a.as_dict())


def test_leading_data_check_sweeps_the_whole_cone():
    # a node dropped is invisible on the support: the check must see it
    S = spec_product(PENTAGON, [(1, 0)], 4)
    bad = leading_data_violations(S, 2, 4)
    assert any(g == (0, 1) and e == 1 for g, e, _ in bad), bad


def test_acyclic_strip_spec_and_its_reverse():
    for B in (KRONECKER2, A3):
        sp = strip_spec(B)
        assert accept_spec(B, sp, 6).ok
        assert is_green_sequence(B, sp)
        assert not accept_spec(B, list(reversed(sp)), 6).ok
    assert strip_spec(THREE_CYCLE) is None


def test_spec_from_components():
    """A spec composed from the strongly connected components' specs,
    source-first (`strongly_connected_redesign.md` §1.1–1.2).  Positive
    controls: the composed spec replays green and passes the axiom check;
    negative control: the components in the reverse order fail both."""
    # no component source: acyclic quivers compose to strip_spec exactly,
    # a quiver with a directed cycle to None
    for B in (A3, [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], KRONECKER2):
        assert spec_from_components(B) == strip_spec(B)
    assert spec_from_components(THREE_CYCLE) is None

    # the replay of the negating sequence [1, 0, 2, 1] of the oriented 3-cycle
    three_cycle_spec = [(0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 0)]
    assert is_green_sequence(THREE_CYCLE, three_cycle_spec)

    def component_spec(sub):
        assert len(sub) == 3
        return three_cycle_spec if sub == THREE_CYCLE else None

    # node 0 → 3-cycle {1, 2, 3} → node 4, the joining arrows single or triple
    for mult in (1, 3):
        B = [[0] * 5 for _ in range(5)]
        for i, j, m in [(1, 2, 1), (2, 3, 1), (3, 1, 1), (0, 1, mult), (3, 4, mult)]:
            B[i][j], B[j][i] = m, -m
        spec = spec_from_components(B, component_spec)
        assert spec == [(1, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 1, 1, 0, 0),
                        (0, 0, 0, 1, 0), (0, 1, 0, 0, 0), (0, 0, 0, 0, 1)], spec
        assert is_green_sequence(B, spec)
        assert accept_spec(B, spec, 3).ok
        reverse = spec[5:] + spec[1:5] + spec[:1]
        assert not is_green_sequence(B, reverse)
        assert not accept_spec(B, reverse, 3).ok

    # a component source with no answer makes the whole composition None
    assert spec_from_components(
        [[0, 1, -1, 1], [-1, 0, 1, 0], [1, -1, 0, 0], [-1, 0, 0, 0]],
        lambda sub: None) is None


def test_three_cycle_chamber():
    spec = [(0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 0)]
    assert accept_spec(THREE_CYCLE, spec, 6).ok
    assert is_green_sequence(THREE_CYCLE, spec)


def test_bad_charge_rejected_without_arithmetic():
    a = accept_spec(PENTAGON, [(1, 0), (0, -1)], 4)
    assert not a.ok and a.violations and a.violations[0][0] == "bad charge"


def test_markov_content_only():
    S, omega = crystalline_spectrum(MARKOV, 4, with_multiplicities=True)
    assert leading_data_violations(S, 3, 4) == []
    rec = content_record(omega)
    assert rec["1,0,0"] == {"0": 1} and rec["0,1,0"] == {"0": 1}
    assert any(sum(int(x) for x in k.split(",")) > 1 for k in rec)   # bubbled content beyond the nodes


if __name__ == "__main__":
    import traceback
    failures = 0
    for name in sorted(globals()):
        fn = globals()[name]
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS: {name}")
        except Exception:
            failures += 1
            print(f"  FAIL: {name}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures} failure(s).")
        sys.exit(1)
    print("\nAll spec-acceptance tests passed.")
