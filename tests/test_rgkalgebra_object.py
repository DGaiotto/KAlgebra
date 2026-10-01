"""Tests for `RGKAlgebraObject` / `RGKAlgebraIso` — parallel
descriptions of one RG flow (`rgkalgebra_object.py`), populated by
unimodular frame changes of the pentagon BPS flow
(`bps_chart_object.bps_frame_change`).

The sharpest check is the algebra/flow distinction: a chart-mutation
witness is a fully certified iso of *algebras* but FAILS the flow
battery (different chamber = different `S_RG` = different flow).

Run:  `python3 run_tests.py`
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bps_kalgebra import BPSKAlgebra
from bps_chart_object import bps_frame_change, mutate_bpskalgebra
from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly
from rgkalgebra_object import RGKAlgebraIso, RGKAlgebraObject


PASS = []
FAIL = []
ONE = LaurentPoly.one()
SAMPLES = [(1, 0), (0, 1), (1, 1)]


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


def _pentagon():
    return BPSKAlgebra(pairing=[[0, 1], [-1, 0]],
                       node_charges=[(1, 0), (0, 1)])


def test_frame_change_is_a_flow_iso():
    """A unimodular re-coordinatization passes BOTH the algebra battery
    and the flow battery."""
    A = _pentagon()
    T = [[1, 1], [0, 1]]
    A2, w = bps_frame_change(A, T)
    se = [Element({l: ONE}) for l in [(0, 0)] + SAMPLES]
    te = [w.map(e) for e in se]
    alg = w.verify_all(se, te,
                       [(a, b) for a in se[1:] for b in se[1:]],
                       [(a, b) for a in te[1:] for b in te[1:]],
                       trace_K=8)
    check("frame change: algebra battery"
          + ("" if all(alg.values()) else f" - {alg}"), all(alg.values()))
    aux = w.aux_iso.verify_all(se, te,
                               [(a, b) for a in se[1:] for b in se[1:]],
                               [(a, b) for a in te[1:] for b in te[1:]],
                               trace_K=8)
    check("frame change: aux battery"
          + ("" if all(aux.values()) else f" - {aux}"), all(aux.values()))
    flow = w.verify_flow(SAMPLES, cutoff=4)
    check("frame change: flow battery (RG-intertwine, S_RG-match, apex)"
          + ("" if all(flow.values()) else f" - {flow}"),
          all(flow.values()))
    # Gradings: BPS's Gamma_RG is the charge lattice itself, so the
    # frame change intertwines gradings via Lambda = T - and the
    # identity map must FAIL (the verifier detects the re-coordinatization)
    aux_lbls = [(1, 0), (0, 1), (1, 1)]

    def LT(p):
        return tuple(sum(T[i][j] * p[j] for j in range(2))
                     for i in range(2))

    check("frame change: grading intertwines via Lambda = T",
          w.verify_grading_intertwine(aux_lbls, charge_map=LT))
    check("frame change: identity charge map correctly rejected",
          not w.verify_grading_intertwine(aux_lbls))


def test_mutation_is_not_a_flow_iso():
    """The algebra/flow distinction, executable: the chart-mutation
    witness is a certified KAlgebraIso of algebras (tested in
    test_bps_chart_object), but wrapped as a flow witness its S_RG
    match FAILS — different chamber, different flow."""
    A = _pentagon()
    A2, mu = mutate_bpskalgebra(A, 0)

    def idmap(label):
        return Element({label: ONE})

    aux_id = KAlgebraIso(A.auxiliary(), A2.auxiliary(), idmap, idmap,
                         name="aux-id")
    w = RGKAlgebraIso(A, A2, mu._forward, mu._inverse, aux_id,
                      name="mutation-as-flow?")
    flow = w.verify_flow(SAMPLES, cutoff=4)
    check("mutation wrapped as flow witness FAILS s_rg_match",
          not flow["s_rg_match"])


def test_rgkalgebra_object_frames():
    """Three frames of one pentagon flow as an RGKAlgebraObject:
    flow batteries green; the composed 0→2 witness stays flow-typed
    and itself passes the flow battery."""
    A0 = _pentagon()
    T1 = [[1, 1], [0, 1]]
    T2 = [[1, 0], [1, 1]]
    A1, w01 = bps_frame_change(A0, T1)
    A2, w12 = bps_frame_change(A1, T2)

    O = RGKAlgebraObject("pentagon-flow")
    O.add_realization("frame-0", A0, {"chart", "rg"})
    O.add_realization("frame-1", A1, {"chart", "rg"})
    O.add_realization("frame-2", A2, {"chart", "rg"})
    O.add_iso("frame-0", "frame-1", w01)
    O.add_iso("frame-1", "frame-2", w12)

    samples = {
        "frame-0": SAMPLES,
        "frame-1": [next(iter(w01.map(Element({l: ONE})).terms))
                    for l in SAMPLES],
    }
    samples["frame-2"] = [next(iter(w12.map(Element({l: ONE})).terms))
                          for l in samples["frame-1"]]

    flows = O.verify_flow_pairwise(samples, cutoff=4)
    bad = {e: r for e, r in flows.items() if not all(r.values())}
    check("frames: flow batteries" + ("" if not bad else f" - {bad}"),
          not bad)

    w02 = O.iso("frame-0", "frame-2")
    check("composed witness stays flow-typed",
          isinstance(w02, RGKAlgebraIso))
    check("composed witness passes the flow battery",
          all(w02.verify_flow(SAMPLES, cutoff=4).values()))

    img = O.transport((1, 1), "frame-0", "frame-2")
    check("transport along frames is one canonical label",
          len(img.terms) == 1)
    aux = O.aux_object()
    aux_samples = {k: [(0, 0)] + list(v) for k, v in samples.items()}
    res = aux.verify_pairwise(aux_samples, pairs=True, trace_K=6)
    bad = {e: r for e, r in res.items() if not all(r.values())}
    check("aux projection: pairwise batteries"
          + ("" if not bad else f" - {bad}"), not bad)


def test_subquiver_flow_pair():
    """One node-deletion flow, two live descriptions: the SubquiverRG
    F-oracle vs the bare DirectionalSubquiverRG co-solver.  The flow
    battery on the identity-shaped witness IS the historical
    'F-oracle cross-checked against the generic solve' validation,
    now curated."""
    from rgkalgebra_object import subquiver_flow_object
    A = _pentagon()
    O = subquiver_flow_object(A, [0], name="pentagon-drop0")
    check("flow pair: realizations + capability routing",
          set(O.keys()) == {"f-oracle", "co-solver"}
          and O.preferred("generic-solve") is O.realization("co-solver"))
    uv_samples = [(1, 0), (0, 1), (1, 1)]
    flows = O.verify_flow_pairwise({"f-oracle": uv_samples}, cutoff=3)
    bad = {e: r for e, r in flows.items() if not all(r.values())}
    check("flow pair: RG-intertwine + S_RG-match + apex"
          + ("" if not bad else f" - {bad}"), not bad)
    w_pair = O.iso("f-oracle", "co-solver")
    aux_lbls = [tuple(g) for g in
                O.realization("f-oracle").auxiliary().node_charges]
    check("flow pair: grading intertwines (identity charge map)",
          w_pair.verify_grading_intertwine(aux_lbls))
    # algebra-level battery across the two derived KAlgebra surfaces
    w = O.iso("f-oracle", "co-solver")
    se = [Element({l: ONE}) for l in uv_samples]
    te = [w.map(e) for e in se]
    alg = w.verify_all(se, te,
                       [(a, b) for a in se for b in se],
                       [(a, b) for a in te for b in te],
                       trace_K=4)
    check("flow pair: derived-KAlgebra battery"
          + ("" if all(alg.values()) else f" - {alg}"),
          all(alg.values()))
    aux = O.aux_object()
    check("flow pair: aux projection carries the IR algebra",
          set(aux.keys()) == {"f-oracle", "co-solver"})


def test_type_guards():
    A = _pentagon()
    O = RGKAlgebraObject("guarded")
    from kalgebra_samples import PentagonKAlg
    try:
        O.add_realization("x", PentagonKAlg())
        check("non-RG realization rejected", False)
    except TypeError:
        check("non-RG realization rejected", True)
    O.add_realization("a", A, {"rg"})
    A2, w = bps_frame_change(A, [[1, 1], [0, 1]])
    O.add_realization("b", A2, {"rg"})
    try:
        O.add_iso("a", "b", KAlgebraIso(A, A2, w._forward, w._inverse))
        check("plain KAlgebraIso witness rejected", False)
    except TypeError:
        check("plain KAlgebraIso witness rejected", True)
    O.add_iso("a", "b", w)
    check("identity iso is flow-typed",
          isinstance(O.iso("a", "a"), RGKAlgebraIso))


if __name__ == "__main__":
    test_frame_change_is_a_flow_iso()
    test_mutation_is_not_a_flow_iso()
    test_rgkalgebra_object_frames()
    test_subquiver_flow_pair()
    test_type_guards()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED, {len(PASS)} passed")
        sys.exit(1)
    print(f"All {len(PASS)} RGKAlgebraObject tests passed.")
