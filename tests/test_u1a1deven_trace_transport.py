"""Tests for `src/cone/u1a1deven_trace_transport.py::DevenTraceTransport`
— the exact trace of the u(1)-gauged `[A_1, D_{2k+2}]` by transport down the
flow `U1A1DevenViaDoddRG(k)` (the design record `finite_type_trace_machinery.md`
section J).

Positive controls first: the gauge tower `Tr(X_{0,1}^n)`, n = 0..3, equals
Creutzig's closed form (k = 1, 2 through 𝖖²⁴; k = 3 through 𝖖¹⁶).  Negative
controls: the shorter dressing chord (1,1,0) at k = 2 differs from it, first
at 𝖖⁶; the singlet chord (1,0,0) agrees through 𝖖² and differs first at 𝖖⁴
(k = 1) and 𝖖⁶ (k = 2, 3).  Then the transport's own machinery: the per-term
skip equals the unmodified sum (skipping off) on a sample and skipping one
order too early does not; the flow-label memo; the magnetic selection rule;
`trace_aux` against `trace`; the honest-fail caps (each forced small must raise
`ValueError`) and the per-k default of the A1Dodd word-length limit; the
stopping-rule guard (it holds at the first stop on every
single-generator label sampled, and fails on a non-increasing valuation
sequence; where it fails at the first stop the sum goes on, the result equals
a deeper evaluation, and an extension that does not settle hits the n-cap).

`deep_gauge_tower` (k = 1, 2 through 𝖖⁴⁰, k = 3 through 𝖖³²) is not in the
default run: `python3 run_tests.py` deep`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "implementations"))

from kalgebra import Element
from u1a1deven_trace_transport import DevenTraceTransport, _clean

_T = {}


def _transport(k):
    if k not in _T:
        _T[k] = DevenTraceTransport(k)
    return _T[k]


def _creutzig(k, n, K):
    from exact_characters import deven_gauged_xn_qn
    return _clean({q: dict(d) for q, d in deven_gauged_xn_qn(k, n, K).items() if q <= K})


def _tower(k, K):
    T = _transport(k)
    for n in range(4):
        got = T.trace((T.D.identity(), (0, n)), K)
        assert got == _creutzig(k, n, K), (k, n, K)


def test_gauge_tower_matches_creutzig():
    """POSITIVE CONTROL: Tr(X01^n) == Creutzig, k=1,2 through 𝖖²⁴, k=3 through 𝖖¹⁶."""
    for k, K in ((1, 24), (2, 24), (3, 16)):
        _tower(k, K)


def test_wrong_chord_differs_at_order_6():
    """NEGATIVE CONTROL: dressing by the length-1 doublet (1,1,0) at k=2 gets
    Tr(1) right through 𝖖⁵ and wrong at 𝖖⁶ — the comparison can fail."""
    from u1a1deven_via_dodd_rg import U1A1DevenViaDoddRG
    W = DevenTraceTransport._from_flow(U1A1DevenViaDoddRG(2, chord=(1, 1, 0)))
    got = W.trace((W.D.identity(), (0, 0)), 12)
    want = _creutzig(2, 0, 12)
    differ = sorted(q for q in set(got) | set(want) if got.get(q) != want.get(q))
    assert differ and differ[0] == 6, differ


def test_singlet_chord_differs():
    """NEGATIVE CONTROL: dressing by the singlet chord (1,0,0) gets Tr(1) right
    through 𝖖² and wrong from 𝖖⁴ at k = 1, from 𝖖⁶ at k = 2, 3 (measured
    2026-09-23; the flow module's "right only to q²" is the k = 1 case)."""
    from u1a1deven_via_dodd_rg import U1A1DevenViaDoddRG
    for k, first in ((1, 4), (2, 6), (3, 6)):
        W = DevenTraceTransport._from_flow(U1A1DevenViaDoddRG(k, chord=(1, 0, 0)))
        got = W.trace((W.D.identity(), (0, 0)), 10)
        want = _creutzig(k, 0, 10)
        differ = sorted(q for q in set(got) | set(want) if got.get(q) != want.get(q))
        assert differ and differ[0] == first, (k, differ)


def _sample(T, cs):
    gens = sorted(T.D.cone_data().mult_gens())
    out = [(T.D.identity(), (0, n)) for n in (0, 1, 2)]
    return out + [((((g, 1),), 0), (0, c)) for g in gens for c in cs]


def test_skip_equals_unmodified():
    """The per-term skip changes nothing: with skipping OFF (every summand
    traced) the series are identical on 59 labels at k = 1, 2, 3.  Control:
    skipping one order too early changes Tr(X01) at k = 1 at 𝖖¹⁴."""
    n = 0
    for k, K, cs in ((1, 12, (-1, 0, 1)), (2, 10, (0, 1)), (3, 8, (0,))):
        S = DevenTraceTransport(k)
        U = DevenTraceTransport(k, max_word_degree=10 ** 6)
        U._skip_slack = 10 ** 9
        for b in _sample(S, cs)[: 15 if k == 1 else 22]:
            assert S.trace(b, K) == U.trace(b, K), (k, b, K)
            n += 1
    assert n == 59, n
    S = DevenTraceTransport(1)
    O = DevenTraceTransport(1)
    O._skip_slack = -1
    b = (S.D.identity(), (0, 1))
    a, o = S.trace(b, 14), O.trace(b, 14)
    assert a == _creutzig(1, 1, 14)
    assert a != o and [q for q in a if a.get(q) != o.get(q)] == [14]


def test_flow_label_memo_truncates():
    """A shallower request after a deeper one reads the truncation, equal to a
    fresh transport at the shallower order."""
    T = DevenTraceTransport(2)
    b = ((((sorted(T.D.cone_data().mult_gens())[0], 1),), 0), (0, 1))
    deep = T.trace(b, 16)
    shallow = T.trace(b, 9)
    assert shallow == {q: d for q, d in deep.items() if q <= 9}
    assert shallow == DevenTraceTransport(2).trace(b, 9)


def test_magnetic_selection_rule():
    """c0 != 0 (magnetic charge) gives the zero series; its c0 = 0 partner
    does not vanish."""
    T = _transport(2)
    g = sorted(T.D.cone_data().mult_gens())[0]
    assert T.trace((((((g, 1),), 0)), (1, 0)), 10) == {}
    assert T.trace((((((g, 1),), 0)), (-1, 2)), 10) == {}
    assert T.trace((T.D.identity(), (1, 0)), 10) == {}
    assert T.trace((((((g, 1),), 0)), (0, 0)), 10) != {}


def test_trace_aux_entry():
    """`trace_aux(RG(b)) == trace(b)`, and on a product: RG is an algebra map,
    so `trace_aux(RG(g)·RG(h)) = 𝖖^c·Tr(L)` when `g·h = 𝖖^c L`."""
    T = _transport(2)
    gens = sorted(T.D.cone_data().mult_gens())
    for b in [(T.D.identity(), (0, 2)), ((((gens[1], 1),), 0), (0, -1))]:
        assert T.trace_aux(T.F.RG(b), 12) == T.trace(b, 12), b
    cone = sorted(next(iter(T.D.cone_data().cones())))
    g = ((((cone[0], 1),), 0), (0, 1))
    h = ((((cone[1], 1),), 0), (0, 0))
    P = {L: c for L, c in T.F.multiply(g, h).terms.items() if not c.is_zero()}
    assert len(P) == 1, P
    (L, cf), = P.items()
    (c, one), = [(e, v) for e, v in cf._coeffs.items() if v]
    assert one == 1
    K = 12
    X = T.aux.multiply_elements(T.F.RG(g), T.F.RG(h))
    want = {q + c: d for q, d in T.trace(L, K - c).items() if q + c <= K}
    assert T.trace_aux(X, K) == want


def test_caps_raise():
    """Honest fails, each forced small: word degree, n-cap, aux-product terms."""
    cases = [
        (dict(max_word_degree=3), 2, 3, 24, "max_word_degree"),
        (dict(max_steps=2), 1, 0, 12, "n-cap"),
        (dict(max_aux_terms=3), 1, 0, 12, "max_aux_terms"),
    ]
    for kw, k, n, K, why in cases:
        T = DevenTraceTransport(k, **kw)
        try:
            T.trace((T.D.identity(), (0, n)), K)
        except ValueError as e:
            assert why in str(e) and f"fq^{K}" in str(e), str(e)
        else:
            raise AssertionError(f"{kw}: no ValueError")
    # the defaults do not fire on the same requests
    _tower(2, 24)


def test_word_limit_default_by_k():
    """The default `max_word_degree` depends on k (set from measured memory;
    12 at k >= 4).  The k = 1 flow label the old fixed 12-letter limit refused
    at 𝖖¹² (a summand with a 14-letter A1Dodd word; 2026-09-23 review) is
    served by the default and vanishes through 𝖖¹²; with the limit forced to 12
    it still raises."""
    from u1a1deven_trace_transport import _MAX_WORD_DEGREE, _default_word_degree
    assert _default_word_degree(4) == 12 and _default_word_degree(1) == _MAX_WORD_DEGREE[1]
    T = _transport(1)
    assert T.max_word_degree == _MAX_WORD_DEGREE[1] > 12
    b = (((((1, 0, 0), 1),), 0), (0, -13))
    assert T.trace(b, 12) == {}
    try:
        DevenTraceTransport(1, max_word_degree=12).trace(b, 12)
    except ValueError as e:
        assert "max_word_degree=12" in str(e), str(e)
    else:
        raise AssertionError("a 12-letter limit no longer refuses the label")


def test_stopping_rule_guard():
    """The guard condition holds at the first stop on every single-generator
    label at k = 1, 2, |c| <= 1 (no sum is extended there); it fails on a
    non-increasing sequence of term valuations, and on a found valuation after
    one beyond the look-ahead."""
    for k, K in ((1, 12), (2, 10)):
        T = DevenTraceTransport(k)
        for b in _sample(T, (-1, 0, 1)):
            T.trace(b, K)
        assert T._stats["guard_extended"] == 0, (k, T._stats)
    T = DevenTraceTransport(1)
    empty = Element({})
    ok, seq = T._guard_status([(0, 0, None, 3), (1, 2, None, 5), (2, 4, empty, None)],
                              8, 0, "x")
    assert ok, seq
    for bad in ([(0, 0, None, 3), (1, 2, None, 5), (2, 4, None, 5)],
                [(0, 0, None, 3), (1, 2, empty, None), (2, 4, None, 6)]):
        ok, seq = T._guard_status(bad, 8, 0, "x")
        assert not ok, (bad, seq)


def test_guard_extends_the_sum():
    """Where the guard condition fails at the first stop the n-sum goes on: at
    k = 3 the flow label below at 𝖖⁴ has three terms contributing nothing
    through 𝖖⁴, with valuations 11, 11, 13 (the first two tie), and the sector
    stops two terms later (13, 15, 17 after the 11).  The result equals the 𝖖¹² evaluation truncated to 𝖖⁴.  An
    extension that does not settle within the n-cap raises."""
    b = (((((3, 0, 0), 1), ((3, 1, 3), 1)), 0), (0, 0))
    T = DevenTraceTransport(3)
    r = T.trace(b, 4)
    assert T._stats["guard_extended"] == 2, T._stats
    deep = DevenTraceTransport(3).trace(b, 12)
    assert r == {q: d for q, d in deep.items() if q <= 4}, (r, deep)
    try:
        DevenTraceTransport(3, max_steps=4).trace(b, 4)
    except ValueError as e:
        assert "n-cap" in str(e) and "the guard" in str(e), str(e)
    else:
        raise AssertionError("the n-cap did not end an unsettled extension")


class _FlowReadTransport(DevenTraceTransport):
    """The transport as it was until 2026-09-24: `C^m` read from the flow's
    `_s_rg_component`, the auxiliary algebra the flow's own (the reference for
    `test_flow_free_parts_equal_the_flow`)."""

    def _C(self, m):
        hit = self._C_cache.get(m)
        if hit is None:
            (lab, c), = self.F._s_rg_component((m,)).items()
            hit = self._C_cache[m] = (lab, c)
        return hit


def test_flow_free_parts_equal_the_flow():
    """The transport builds its auxiliary algebra and the parts of `S` without
    the flow (2026-09-24).  POSITIVE CONTROL first, k = 1..4: `C^m` and `c_m`
    equal the flow's `_s_rg_component((m,))` for m = 0..12 (label and the
    coefficient's expansion through 𝖖³⁰), and the two auxiliary algebras have
    the same identity and the same products on the sampled labels.  Then the
    traces are unchanged: `trace_aux` of the flow-free transport equals that of
    a transport reading the flow as before, on the RG images of the gauge tower
    and of four curve labels at k = 1, 2, 3 (through 𝖖⁸ / 𝖖⁸ / 𝖖⁶), and the
    gauge tower equals Creutzig's closed form.  A fresh process running
    `trace_aux` imports no RG module (while one calling `trace(b)`, the
    flow-label entry point, does — the control that the check can see it)."""
    import json
    import subprocess
    from u1a1deven_via_dodd_rg import U1A1DevenViaDoddRG
    from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra
    for k in (1, 2, 3, 4):
        T, Fl = DevenTraceTransport(k), U1A1DevenViaDoddRG(k)
        for m in range(13):
            (lab, c), = Fl._s_rg_component((m,)).items()
            tl, tc = T._C(m)
            assert tl == lab, (k, m)
            assert tc.expand(30)._coeffs == c.expand(30)._coeffs, (k, m)
        assert T.aux.identity() == Fl.auxiliary().identity()
        a = ((((1, 0, 0), 1),), 0), (0, 1)
        b = ((((k, 1, 0), 1),), 1), (1, -1)
        for x, y in ((a, b), (b, a), (a, a)):
            assert T.aux.multiply(x, y) == Fl.auxiliary().multiply(x, y), (k, x, y)
    for k, K in ((1, 8), (2, 8), (3, 6)):
        T = DevenTraceTransport(k)
        W = _FlowReadTransport._from_flow(U1A1DevenViaDoddRG(k))
        A = U1A1DevenConeKAlgebra(k)
        labs = [((), e, 0) for e in range(3)] + [A.curve(1, 3), A.curve(0, 3, e=1),
                                                 A.curve(2, 3, kappa=1), A.curve(0, 2 * k + 1)]
        for lab in labs:
            X = A._phi(lab)
            assert T.trace_aux(X, K) == W.trace_aux(X, K), (k, lab)
        for n in range(3):
            got = T.trace_aux(A._phi(((), n, 0)), K)
            assert got == _creutzig(k, n, K), (k, n)
    code = ("import sys, json\n"
            "from u1a1deven_trace_transport import DevenTraceTransport\n"
            "from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra\n"
            "A = U1A1DevenConeKAlgebra(2); T = DevenTraceTransport(2)\n"
            "T.trace_aux(A._phi(A.curve(1, 3)), 4)\n"
            "before = [m for m in ('u1a1deven_via_dodd_rg', 'rgkalgebra') if m in sys.modules]\n"
            "T.trace(A._flow_label(A.curve(1, 3)), 4)\n"
            "after = [m for m in ('u1a1deven_via_dodd_rg', 'rgkalgebra') if m in sys.modules]\n"
            "print(json.dumps([before, after]))\n")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=root), cwd=root, timeout=600)
    assert out.returncode == 0, out.stderr[-1500:]
    before, after = json.loads(out.stdout.strip().splitlines()[-1])
    assert before == [] and after == ["u1a1deven_via_dodd_rg", "rgkalgebra"], (before, after)


def deep_gauge_tower():
    """DEEP: k = 1, 2 through 𝖖⁴⁰ and k = 3 through 𝖖³² (not in the default run)."""
    for k, K in ((1, 40), (2, 40), (3, 32)):
        _tower(k, K)


_TESTS = [test_gauge_tower_matches_creutzig, test_wrong_chord_differs_at_order_6,
          test_singlet_chord_differs,
          test_skip_equals_unmodified, test_flow_label_memo_truncates,
          test_magnetic_selection_rule, test_trace_aux_entry, test_caps_raise,
          test_word_limit_default_by_k,
          test_stopping_rule_guard, test_guard_extends_the_sum,
          test_flow_free_parts_equal_the_flow]

if __name__ == "__main__":
    import time
    run = _TESTS + ([deep_gauge_tower] if "deep" in sys.argv[1:] else [])
    for t in run:
        t0 = time.time()
        t()
        print(f"  PASS: {t.__name__} [{time.time() - t0:.1f}s]", flush=True)
    print(f"\nAll {len(run)} DevenTraceTransport tests passed.")
