"""Tests for `directional_subquiver_rg` — the directional node-drop
RGKAlgebra + spec arranger + certification harness.

Pentasquare (pentagon, drop γ₂) and A₄ (drop γ₂ — the a1a2k k=2 inner
flow) instances:
  * closed-form `S_RG` (stuff product, ket-peel) ≡ the validated
    `rg_flow.SubquiverRG` extraction, per cutoff and per graded
    component;
  * the *derived* `multiply` / `rho` / `rho_inverse` / `trace` agree
    with an independently built UV `BPSKAlgebra` — ρ and trace are the
    checks the old wrap-UV / first-cut classes could not make
    non-circular;
  * version (b): the local-move BFS arranger reaches a stuff-first
    chamber from the natural spec WITHOUT moving labels (the
    certificate passes against the natural-chamber UV);
  * the F-oracle path accelerates `RG` and is cross-checked against
    the exact solve by `rho_inverse` (a wrong oracle raises).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kalgebra import Element
from laurent_poly import LaurentPoly
from bps_kalgebra import BPSKAlgebra
from rg_flow import SubquiverRG
from directional_subquiver_rg import (
    DirectionalSubquiverRG,
    DirectionalSingleNodeRG,
    arrange_spec_stuff_first,
    certify_directional_vs_bps,
    stuff_split,
    uv_f_oracle,
)

_ONE = LaurentPoly.one()

PENTA_PAIRING = [[0, 1], [-1, 0]]
PENTA_NODES = [(1, 0), (0, 1)]
PENTA_STUFF_FIRST = [(0, 1), (1, 1), (1, 0)]
PENTA_NATURAL = [(1, 0), (0, 1)]

A4_PAIRING = [[0, 1, 0, 0], [-1, 0, 1, 0], [0, -1, 0, 1], [0, 0, -1, 0]]
A4_NODES = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
A4_NATURAL = list(A4_NODES)


def _pentasquare():
    return DirectionalSingleNodeRG(
        PENTA_PAIRING, PENTA_NODES, PENTA_STUFF_FIRST, gamma_drop=(0, 1),
    )


def _uv_pentagon(spec):
    return BPSKAlgebra(
        pairing=PENTA_PAIRING, node_charges=PENTA_NODES,
        spec=spec, verify="off",
    )


def test_s_rg_closed_form_matches_extractor():
    """Version (a): the chart-free closed-form `S_RG` equals the
    validated UV-wrapping extraction, cutoff by cutoff."""
    D = _pentasquare()
    UV = _uv_pentagon(PENTA_STUFF_FIRST)
    EX = SubquiverRG(UV, [1])
    for cut in (0, 1, 2, 3, 4):
        a = D.rg_generator(cut)
        b = EX.rg_generator(cut)
        assert set(a) == set(b), (cut, sorted(a), sorted(b))
        for k in a:
            assert (a[k] - b[k]).is_zero(), (cut, k, a[k], b[k])


def test_s_rg_component_exact():
    """`_s_rg_component`: exact graded slices, `{}` off-cone, identity
    at zero."""
    D = _pentasquare()
    assert D._s_rg_component((-1,)) == {}
    c0 = D._s_rg_component((0,))
    assert set(c0) == {(0, 0)}
    c1 = D._s_rg_component((1,))
    assert set(c1) == {(0, 1)}
    c2 = D._s_rg_component((2,))
    assert set(c2) == {(0, 2)}


def test_pentasquare_derived_api_vs_uv():
    """Derived multiply / ρ / ρ⁻¹ / trace / inner product against the
    independent UV pentagon — including the two checks the old classes
    could not make meaningful (ρ is tRG-derived, trace is transported)."""
    D = _pentasquare()
    UV = _uv_pentagon(PENTA_STUFF_FIRST)
    labels = [(a, b) for a in range(0, 2) for b in range(0, 2)]
    for a in labels:
        assert D.rho(a) == UV.rho(a), a
        assert D.rho_inverse(a) == UV.rho_inverse(a), a
        for b in labels:
            assert D.multiply(a, b) == UV.multiply(a, b), (a, b)
    for a in [(0, 0), (1, 0), (1, 1)]:
        assert D.trace(a, K=5) == UV.trace(a, K=5), a
    assert D.inner_product((1, 0), (1, 0), K=6) \
        == UV.inner_product((1, 0), (1, 0), K=6)


def test_pentagon_certificate_stuff_first():
    D = _pentasquare()
    UV = _uv_pentagon(PENTA_STUFF_FIRST)
    rep = certify_directional_vs_bps(D, UV, trace_K=5)
    failures = {k: v for k, v in rep.items() if k != "iso" and v is not True}
    assert not failures, failures


def test_arranger_pentagon():
    """Version (b): the natural pentagon spec is rearranged stuff-first
    by ONE pentagon expansion; the result is the pentasquare spec."""
    arr = arrange_spec_stuff_first(
        PENTA_PAIRING, PENTA_NODES, PENTA_NATURAL, [1])
    assert arr == [(0, 1), (1, 1), (1, 0)], arr
    stuff, ir = stuff_split(PENTA_NODES, arr, [1])
    assert stuff == [(0, 1), (1, 1)] and ir == [(1, 0)]


def test_arranger_budget_returns_none():
    arr = arrange_spec_stuff_first(
        PENTA_PAIRING, PENTA_NODES, PENTA_NATURAL, [1], max_states=1)
    assert arr is None


def test_certificate_across_chambers():
    """The labels do not move under local-move arrangement: the
    directional algebra built with `arrange=True` certifies against the
    UV in the ORIGINAL (natural) chamber — the chamber-calibration trap
    is structurally absent."""
    D = DirectionalSingleNodeRG(
        PENTA_PAIRING, PENTA_NODES, PENTA_NATURAL, gamma_drop=(0, 1),
        arrange=True,
    )
    assert D.spec == [(0, 1), (1, 1), (1, 0)]
    UV = _uv_pentagon(PENTA_NATURAL)
    rep = certify_directional_vs_bps(D, UV, trace_K=5)
    failures = {k: v for k, v in rep.items() if k != "iso" and v is not True}
    assert not failures, failures


def test_strict_split_raises_on_interleaving():
    try:
        stuff_split(PENTA_NODES, [(1, 0), (0, 1), (1, 1)], [1])
    except ValueError as e:
        assert "stuff" in str(e)
    else:
        raise AssertionError("interleaved spec must raise")


def test_constructor_validation():
    # dropping every node
    try:
        DirectionalSubquiverRG(
            PENTA_PAIRING, PENTA_NODES, PENTA_STUFF_FIRST, [0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("dropping every node must raise")
    # unknown drop charge
    try:
        DirectionalSingleNodeRG(
            PENTA_PAIRING, PENTA_NODES, PENTA_STUFF_FIRST,
            gamma_drop=(5, 5))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown drop charge must raise")
    # index and charge forms agree
    D1 = DirectionalSingleNodeRG(
        PENTA_PAIRING, PENTA_NODES, PENTA_STUFF_FIRST, gamma_drop=1)
    D2 = _pentasquare()
    assert D1.drop_indices == D2.drop_indices == (1,)
    assert D1.gamma_drop == (0, 1)


def test_a4_certificate():
    """A₄, drop γ₂, arranged from the natural simple-root spec (the
    a1a2k k=2 inner flow in its honest stuff-first chamber
    `[γ₂, γ₁+γ₂, γ₁, γ₃, γ₄]`)."""
    D = DirectionalSingleNodeRG(
        A4_PAIRING, A4_NODES, A4_NATURAL, gamma_drop=(0, 1, 0, 0),
        arrange=True, rg_window=6,
    )
    assert D.spec[:2] == [(0, 1, 0, 0), (1, 1, 0, 0)], D.spec
    assert D.ir_spec == [(1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    UV = BPSKAlgebra(
        pairing=A4_PAIRING, node_charges=A4_NODES, spec=A4_NATURAL,
        verify="off",
    )
    labels = [(0, 0, 0, 0)] + A4_NODES \
        + [(1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1)]
    pairs = [(a, b) for a in A4_NODES for b in A4_NODES]
    rep = certify_directional_vs_bps(
        D, UV, labels=labels, pairs=pairs, trace_K=4)
    failures = {k: v for k, v in rep.items() if k != "iso" and v is not True}
    assert not failures, failures


def test_a4_f_oracle_cross_check():
    """F-oracle accelerates RG; `rho_inverse` re-derives via the exact
    solve and cross-checks the oracle.  A wrong oracle raises."""
    UV = BPSKAlgebra(
        pairing=A4_PAIRING, node_charges=A4_NODES, spec=A4_NATURAL,
        verify="off",
    )
    D = DirectionalSingleNodeRG(
        A4_PAIRING, A4_NODES, A4_NATURAL, gamma_drop=(0, 1, 0, 0),
        arrange=True, rg_window=6,
    )
    D2 = DirectionalSingleNodeRG(
        A4_PAIRING, A4_NODES, A4_NATURAL, gamma_drop=(0, 1, 0, 0),
        arrange=True, rg_window=6,
        f_oracle=uv_f_oracle(UV, D.auxiliary()),
    )
    for a in [(0, 1, 0, 0), (1, 1, 0, 0), (0, 1, 1, 0)]:
        assert D2.RG(a) == D.RG(a), a
    assert D2.multiply((1, 0, 0, 0), (0, 1, 0, 0)) \
        == UV.multiply((1, 0, 0, 0), (0, 1, 0, 0))
    # rho_inverse triggers the exact mirror solve + oracle cross-check
    assert D2.rho_inverse((1, 1, 0, 0)) == UV.rho_inverse((1, 1, 0, 0))

    # A wrong oracle must fail loudly at the cross-check.  (Doubled
    # coefficient: can never be right — the apex coefficient of a
    # genuine RG image is 1.)
    def bad_oracle(label):
        return Element({tuple(label): LaurentPoly({0: 2})})

    D3 = DirectionalSingleNodeRG(
        A4_PAIRING, A4_NODES, A4_NATURAL, gamma_drop=(0, 1, 0, 0),
        arrange=True, rg_window=6, f_oracle=bad_oracle,
    )
    D3.RG((0, 1, 1, 0))
    try:
        D3.rho_inverse((0, 1, 1, 0))
    except ValueError as e:
        assert "oracle" in str(e)
    else:
        raise AssertionError("wrong F-oracle must raise on cross-check")


def test_unguarded_mode_certifies():
    """`require_stuff_first=False` (version (c) — open question): the
    order-by-order ket factorization on interleaved specs certifies
    all-True on the probed cases — pentagon drop γ₂ (stuff second) and
    A₄ drop γ₃ (middle drop: non-commuting IR factors on BOTH sides of
    the stuff factor).  General validity remains open; the staged
    certificate is the gate (a failure here would be a *finding*, not
    just a regression)."""
    D = DirectionalSingleNodeRG(
        PENTA_PAIRING, PENTA_NODES, PENTA_NATURAL, gamma_drop=(0, 1),
        require_stuff_first=False,
    )
    assert D.stuff_spec == [(0, 1)] and D.ir_spec == [(1, 0)]
    UV = _uv_pentagon(PENTA_NATURAL)
    rep = certify_directional_vs_bps(D, UV, trace_K=5)
    bad = {k: v for k, v in rep.items() if k != "iso" and v is not True}
    assert not bad, bad

    D3 = DirectionalSingleNodeRG(
        A4_PAIRING, A4_NODES, A4_NATURAL, gamma_drop=(0, 0, 1, 0),
        require_stuff_first=False, rg_window=6,
    )
    assert D3.stuff_spec == [(0, 0, 1, 0)]
    UV4 = BPSKAlgebra(
        pairing=A4_PAIRING, node_charges=A4_NODES, spec=A4_NATURAL,
        verify="off",
    )
    labels = [(0, 0, 0, 0)] + A4_NODES + [(1, 1, 0, 0), (0, 1, 1, 0)]
    pairs = [(a, b) for a in A4_NODES for b in A4_NODES]
    rep = certify_directional_vs_bps(
        D3, UV4, labels=labels, pairs=pairs, trace_K=4)
    bad = {k: v for k, v in rep.items() if k != "iso" and v is not True}
    assert not bad, bad


def test_composition_of_node_drops():
    """Sequential single-node drops compose (`then` / `ComposedRG`) and
    agree with the direct two-node drop: A₄, drop γ₁ then drop γ₂ of
    the survivors ≡ drop {γ₁, γ₂} at once.  Exercises `from_uv` (the
    second flow's `starting_algebra()` must BE the first flow's
    auxiliary, by identity) and the squashed
    `S^UV_IR = RG²(S¹)·S²` generator."""
    from directional_subquiver_rg import DirectionalSubquiverRG
    D1 = DirectionalSubquiverRG(
        A4_PAIRING, A4_NODES, A4_NATURAL, drop=[0], rg_window=6)
    # Drop the first survivor (γ₂) of D1's auxiliary.
    D2 = DirectionalSubquiverRG.from_uv(
        D1.auxiliary(), [0], attach_f_oracle=False, rg_window=6)
    assert D2.starting_algebra() is D1.auxiliary()
    C = D1.then(D2)
    D12 = DirectionalSubquiverRG(
        A4_PAIRING, A4_NODES, A4_NATURAL, drop=[0, 1], rg_window=6)
    # Same flow target (the [γ₃, γ₄] = A₂ theory).
    assert D2.auxiliary().node_charges == D12.auxiliary().node_charges
    # RG images agree label-by-label.
    for a in [(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (1, 1, 0, 0),
              (0, 0, 1, 1)]:
        assert C.RG(a) == D12.RG(a), a
    # Squashed S^UV_IR ≡ the direct two-drop generator (cutoff 2).
    sa = C.rg_generator(2)
    sb = D12.rg_generator(2)
    assert set(sa) == set(sb), (sorted(sa), sorted(sb))
    for k in sa:
        assert (sa[k] - sb[k]).is_zero(), (k, sa[k], sb[k])
    # Structure constants through the composition ≡ UV.
    UV = BPSKAlgebra(
        pairing=A4_PAIRING, node_charges=A4_NODES, spec=A4_NATURAL,
        verify="off",
    )
    for a, b in [((1, 0, 0, 0), (0, 1, 0, 0)), ((0, 0, 1, 0), (0, 0, 0, 1))]:
        assert C.multiply(a, b) == UV.multiply(a, b), (a, b)
    assert C.verify_rg_unital()
    assert C.verify_rg_multiplicative((0, 0, 1, 0), (0, 0, 0, 1))


def test_axiom_battery_pentasquare():
    """The full axiom battery (KAlgebra axioms + RG axioms + the
    Schur-side trace axioms at small K) on a derived directional flow —
    the executable form of the RG axiomatics on a presentation where
    nothing is delegated."""
    from directional_subquiver_rg import verify_axioms
    D = _pentasquare()
    labels = [(1, 0), (0, 1), (1, 1)]
    pairs = [(a, b) for a in labels for b in labels][:6]
    checks = verify_axioms(D, labels, pairs, trace_K=4)
    bad = {k: v for k, v in checks.items() if v is not True}
    assert not bad, bad


def test_replaced_extractors_are_directional():
    """`rg_flow.SubquiverRG` / `SingleNodeRG` are now directional: KAlgebra ops derived (not UV-delegated), UV retained
    as `starting_algebra()` + F-oracle, historical introspection
    surface intact, values at parity with the UV."""
    from rg_flow import SingleNodeRG, SubquiverRG
    from directional_subquiver_rg import DirectionalSubquiverRG
    UV = _uv_pentagon(PENTA_STUFF_FIRST)
    E = SubquiverRG(UV, [1])
    assert isinstance(E, DirectionalSubquiverRG)
    assert E.starting_algebra() is UV
    assert E.uv_algebra is UV
    assert E.node_indices == (1,)
    assert E.multiply((1, 0), (0, 1)) == UV.multiply((1, 0), (0, 1))
    assert E.rho((1, 0)) == UV.rho((1, 0))
    assert E.trace((0, 0), K=4) == UV.trace((0, 0), K=4)
    S = SingleNodeRG(UV, 1)
    assert S._j == 1 and S._gamma_j == (0, 1)
    # the F-oracle is attached and cross-checked by the mirror
    assert S.RG((1, 1)) is not None
    assert S.rho_inverse((1, 1)) == UV.rho_inverse((1, 1))


if __name__ == "__main__":
    # Discovery harness: every `test_*` in the module runs — the previous
    # hand-maintained list had silently omitted four tests (among them
    # `test_composition_of_node_drops`, the recorded RED — the composed
    # rg_generator box-vs-simplex windowing bug, fixed 2026-07-10).
    _funcs = [v for k, v in sorted(globals().items())
              if k.startswith("test_")]
    for _f in _funcs:
        _f()
        print(f"OK  {_f.__name__}")
    print(f"\nAll {len(_funcs)} directional_subquiver_rg tests pass.")
