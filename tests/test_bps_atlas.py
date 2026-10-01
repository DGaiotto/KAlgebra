"""`BPSAtlas` — ensemble of BPSKAlgebra charts + automated certified isos.

Run:  `python3 run_tests.py`
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bps_kalgebra import BPSKAlgebra
from bps_atlas import BPSAtlas
from kalgebra import Element
from laurent_poly import LaurentPoly

PASS = []
FAIL = []
ONE = LaurentPoly.one()


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


def _pentagon():
    return BPSKAlgebra(pairing=[[0, 1], [-1, 0]],
                       node_charges=[(1, 0), (0, 1)])


_LABELS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def _battery(iso, labels, trace_K=8):
    """Run the full KAlgebraIso battery on a label window of the source."""
    se = [Element({l: ONE}) for l in labels]
    te = [iso.map(e) for e in se]
    return iso.verify_all(
        se, te,
        [(a, b) for a in se[1:] for b in se[1:]],
        [(a, b) for a in te[1:] for b in te[1:]],
        trace_K=trace_K,
    )


def test_root():
    At = BPSAtlas(_pentagon())
    check("root-only at construction", At.keys() == [()])
    check("chart() returns the seed", At.chart() is At.root)


def test_single_mutate_certified():
    """One head necklace: the composed root→chart iso passes the full battery."""
    At = BPSAtlas(_pentagon())
    key, iso = At.mutate((), 0)            # node 0 = (1,0) = spec head
    check("chart materialized", key in At.keys())
    res = _battery(iso, _LABELS)
    check("single-mutate battery"
          + ("" if all(res.values()) else f" - {res}"), all(res.values()))


def test_path_iso_certified():
    """A 2-step rotation chain: root→end iso passes the full battery."""
    At = BPSAtlas(_pentagon())
    k1, _ = At.mutate_head(())
    k2, iso = At.mutate_head(k1)
    res = _battery(iso, _LABELS)
    check("path-iso battery"
          + ("" if all(res.values()) else f" - {res}"), all(res.values()))


def test_transition_between_charts():
    """iso(src, dst) for two non-root charts (composed via root) is certified."""
    At = BPSAtlas(_pentagon())
    k1, _ = At.mutate_head(())
    k2, _ = At.mutate_head(k1)
    iso12 = At.iso(k1, k2)
    # samples must live in k1's label space: transport the root window into k1.
    k1_labels = [next(iter(At.transport(l, (), k1).terms)) for l in _LABELS]
    res = _battery(iso12, k1_labels, trace_K=6)
    check("transition iso k1→k2 battery"
          + ("" if all(res.values()) else f" - {res}"), all(res.values()))


def test_monodromy_is_rho2():
    """The rotation-loop automorphism is ρ² (certified + label check)."""
    A = _pentagon()
    At = BPSAtlas(A)
    mono = At.monodromy()
    res = _battery(mono, _LABELS)
    check("monodromy battery"
          + ("" if all(res.values()) else f" - {res}"), all(res.values()))
    lab = (1, 0)
    img = next(iter(mono.map(Element({lab: ONE})).terms))
    rho2 = A.rho(A.rho(lab))
    check(f"monodromy = ρ² on {lab} (got {img}, ρ²={rho2})", img == rho2)


def test_as_object_pairwise():
    """The materialized charts as a KAlgebraObject: pairwise batteries green."""
    At = BPSAtlas(_pentagon())
    k1, _ = At.mutate_head(())
    k2, _ = At.mutate_head(k1)
    O = At.as_object()
    check("object has 3 charts",
          set(O.keys()) == {"root", At._key_str(k1), At._key_str(k2)})
    samples = {
        At._key_str(k): [next(iter(At.transport(l, (), k).terms)) for l in _LABELS]
        for k in At.keys()
    }
    res = O.verify_pairwise(samples, pairs=True, trace_K=6)
    bad = {e: r for e, r in res.items() if not all(r.values())}
    check("as_object pairwise batteries"
          + ("" if not bad else f" - {bad}"), not bad)


def test_F_transport_matches_native():
    """T3a: F solved once at root, transported to a chart, equals that chart's
    native F at the transported label (S-free quantum-torus transport)."""
    At = BPSAtlas(_pentagon())
    k1, _ = At.mutate_head(())
    a = (1, 1)
    F_trans = At.F(a, k1)
    a_in_k1 = next(iter(At.transport(a, (), k1).terms))
    F_native = At.chart(k1).F(a_in_k1)
    check(f"F transport root→k1 == native chart F (label {a_in_k1})",
          F_trans == F_native)


def test_multiply_shared_across_charts():
    """T3a: multiply computed once at root, served in a chart via the certified
    label iso, equals that chart's native multiply at the transported labels."""
    At = BPSAtlas(_pentagon())
    k1, _ = At.mutate_head(())
    shared = At.multiply((1, 0), (0, 1), k1)
    a1 = next(iter(At.transport((1, 0), (), k1).terms))
    b1 = next(iter(At.transport((0, 1), (), k1).terms))
    native = At.chart(k1).multiply(a1, b1)
    check("shared multiply == native chart multiply", shared == native)


def test_cache_solved_once():
    """T3a: F/multiply are solved once at root and reused across charts."""
    At = BPSAtlas(_pentagon())
    k1, _ = At.mutate_head(())
    k2, _ = At.mutate_head(k1)
    for key in ((), k1, k2):
        At.F((1, 1), key)
        At.multiply((1, 0), (0, 1), key)
    check("F solved once at root, served to 3 charts", len(At._F_root) == 1)
    check("multiply computed once at root, served to 3 charts",
          len(At._mult_root) == 1)


def test_certificate_pentagon():
    """T5: the one-call axiomatics-vs-cluster certificate over the rotation
    chamber chain — per-edge battery, multiply + Schur-index chart-invariance,
    monodromy = ρ²."""
    At = BPSAtlas(_pentagon())
    cert = At.certificate(trace_K=8)
    check("certificate: rotation closes (period ≥ 1)", cert["period"] >= 1)
    check("certificate: every edge battery-green",
          all(all(r.values()) for r in cert["edge_batteries"]))
    check("certificate: multiply chart-invariant", cert["multiply_chart_invariant"])
    check("certificate: Schur index chart-invariant", cert["trace_chart_invariant"])
    check("certificate: monodromy = ρ²", cert["monodromy"]["is_rho2"])
    check("certificate: all_ok", cert["all_ok"])


def test_intrinsic_trace_and_inner_product():
    """T3b: trace / inner_product are intrinsic — atlas value == root value,
    cached once by intrinsic label."""
    A = _pentagon()
    At = BPSAtlas(A)
    check("atlas.trace == root.trace", At.trace((1, 0), 8) == A.trace((1, 0), 8))
    check("atlas.inner_product == root.inner_product",
          At.inner_product((1, 0), (1, 0), 8) == A.inner_product((1, 0), (1, 0), 8))
    At.trace((1, 0), 8)  # second call hits the cache
    check("trace cached once", len(At._trace_cache) == 1)


def test_intrinsic_memoization_layer():
    """T3b/D9: ρ / ρ⁻¹ are intrinsic and memoized alongside multiply / trace /
    inner_product / F; the cache reflects use and clears."""
    A = _pentagon()
    At = BPSAtlas(A)
    check("atlas.rho == root.rho", At.rho((1, 0)) == tuple(A.rho((1, 0))))
    check("atlas.rho_inverse == root.rho_inverse",
          At.rho_inverse((1, 0)) == tuple(A.rho_inverse((1, 0))))
    At.rho((1, 0))  # memoized
    check("rho memoized once", At.intrinsic_cache_info()["rho"] == 1)
    # populate several intrinsic caches, then confirm clear
    At.multiply((1, 0), (0, 1)); At.trace((1, 0), 6); At.inner_product((1, 0), (1, 0), 6)
    info = At.intrinsic_cache_info()
    check("intrinsic caches populated",
          info["multiply"] >= 1 and info["trace"] >= 1 and info["inner_product"] >= 1)
    At.clear_intrinsic_caches()
    check("clear_intrinsic_caches empties them",
          all(v == 0 for v in At.intrinsic_cache_info().values()))


def test_cross_validate_spec_charts_agree():
    """T3b: the Schur index is chart-invariant by axiom; across verified (spec)
    charts a disagreement would be a truncation error — here they agree."""
    At = BPSAtlas(_pentagon())
    cv = At.cross_validate((1, 0), K=8)
    check("cross_validate: spec charts agree", cv["agree"])
    check("cross_validate: provenance all 'spec'",
          set(cv["provenance"].values()) == {"spec"})


def test_build_S_charts_provenance_and_conjecture():
    """T4: spec-free chart materialization (conjectural direct-S); the iso still
    certifies, and cross_validate tests the conjecture against the verified root
    (agreement on the pentagon = conjecture consistent), reporting provenance so
    a failure could be read as a conjecture objection, not assumed truncation."""
    A = _pentagon()
    At = BPSAtlas(A, build_S_charts=True)
    k1, iso = At.mutate_head(())
    check("build_S chart provenance", At.provenance(k1) == "build_S")
    res = _battery(iso, _LABELS, trace_K=6)
    check("build_S chart iso still battery-green"
          + ("" if all(res.values()) else f" - {res}"), all(res.values()))
    cv = At.cross_validate((1, 0), K=8)
    check("cross_validate (incl build_S) agrees on pentagon", cv["agree"])
    check("cross_validate: root='spec', mutated='build_S'",
          cv["provenance"]["root"] == "spec"
          and any(v == "build_S" for v in cv["provenance"].values()))


def test_loop_automorphisms():
    """T9: discover chart-graph loops and their automorphisms. Even the pentagon
    is non-trivial — its chart graph carries a loop whose automorphism is ρ²,
    discovered from the loop (not hand-built) and certified by the full battery."""
    A = _pentagon()
    At = BPSAtlas(A)
    auts = At.discover_automorphisms(max_depth=6)
    check("pentagon: exactly one non-identity loop automorphism", len(auts) == 1)
    aut = auts[0]
    img = tuple(next(iter(aut.map(Element({l: ONE})).terms)) for l in _LABELS)
    rho2 = tuple(A.rho(A.rho(l)) for l in _LABELS)
    check(f"loop automorphism = ρ² (got {img})", img == rho2)
    res = _battery(aut, _LABELS, trace_K=6)
    check("loop automorphism is a certified automorphism"
          + ("" if all(res.values()) else f" - {res}"), all(res.values()))


def test_summary():
    """T13 helper: the one-call catalogue record (certificate + loop
    automorphisms) on the pentagon."""
    s = BPSAtlas(_pentagon()).summary(trace_K=6)
    check("summary all_ok", s["all_ok"])
    check("summary period 4", s["period"] == 4)
    check("summary monodromy ρ²", s["monodromy_is_rho2"])
    check("summary finds 1 loop automorphism", s["n_loop_automorphisms"] == 1)
    check("summary chart-invariant (multiply + trace)",
          s["multiply_chart_invariant"] and s["trace_chart_invariant"])


def test_automorphism_group_is_Z5():
    """T9: for a FINITE-TYPE chart (pentagon = A₂) the chart-graph loop
    automorphisms close into the *finite* cluster modular group ⟨ρ²⟩ = Z/5 —
    reconciling the period-4 rotation 4-cycle with A₂'s periodicity-5 (the
    4-cycle's ρ² monodromy has order 5).  (Gauge theories are the opposite
    regime — typically infinite; see the module/method docstring.)"""
    A = _pentagon()
    At = BPSAtlas(A)
    g = At.automorphism_group()
    check("automorphism group is finite (finite type)", g["finite"])
    check(f"|Aut_loop| = 5 = Z/5 (got {g['order']})", g["order"] == 5)
    check("generated by ρ² (one generator)", g["n_generators"] == 1)
    # the 5 elements act on the window as the 5 powers of ρ
    window = g["signatures"][0]               # each sig is a tuple of (l, img)
    labels = [l for l, _ in window]
    rho_powers = set()
    for j in range(5):
        sig = []
        for l in labels:
            x = l
            for _ in range(j):
                x = At.rho(x)
            sig.append((l, x))
        rho_powers.add(tuple(sig))
    got = set(tuple(s) for s in g["signatures"])
    check("group elements = {ρ^j : j=0..4}", got == rho_powers)
    # the ρ-orbit window is genuinely closed under ρ (5-cycle)
    check("ρ-orbit window has size 5", len(At._rho_orbit_window()) == 5)


def test_chart_isomorphism_recognizes_quiver_iso():
    """T9-bis: a new chart that is a node-perm + Γ-automorphism image of an old
    one is recognized, the witness is a certified KAlgebraIso, and for the
    pentagon the Γ-automorphism relating root and the first necklace IS ρ²."""
    A = _pentagon()
    At = BPSAtlas(A)
    nk, _ = At.mutate((), 0, "fwd")                 # C1
    iso = At.chart_isomorphism((), nk)
    check("root and first necklace are quiver-isomorphic", iso is not None)
    res = _battery(iso, _LABELS, trace_K=6)
    check("chart-iso witness is battery-green"
          + ("" if all(res.values()) else f" - {res}"), all(res.values()))
    g = {l: tuple(next(iter(iso.map(Element({l: ONE})).terms))) for l in [(1, 0), (0, 1)]}
    rho2 = {l: tuple(A.rho(A.rho(l))) for l in [(1, 0), (0, 1)]}
    check(f"witnessing Γ-automorphism = ρ² (got {g})", g == rho2)


def test_recognize_folds_chart_onto_representative():
    """T9-bis: `recognize` is the work-saver — a freshly reached chart is folded
    onto an already-built representative; an empty pool returns None."""
    A = _pentagon()
    At = BPSAtlas(A)
    nk, _ = At.mutate((), 0, "fwd")
    rec = At.recognize(nk, among=[()])
    check("recognize folds the new chart onto root", rec is not None and rec[0] == ())
    check("recognize among an empty pool is None", At.recognize(nk, among=[]) is None)


def test_chart_isomorphism_none_when_quivers_differ():
    """T9-bis: charts with genuinely non-isomorphic BPS quivers are NOT folded
    (linear A₃: a necklace flips an arrow → not a node-perm + Γ-auto image)."""
    A = BPSKAlgebra(pairing=[[0, 1, 0], [-1, 0, 1], [0, -1, 0]],
                    node_charges=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                    spec=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    At = BPSAtlas(A)
    nk, _ = At.mutate((), 0, "fwd")
    check("non-isomorphic quivers give no chart-iso",
          At.chart_isomorphism((), nk) is None)
    check("self chart-iso always exists", At.chart_isomorphism((), ()) is not None)


def test_verify_spec_equivalence():
    """T9-bis: the canonical-spec verifier — chart-isomorphic charts have
    equivalent specs (same S), and the spec-S coincides with the chamber's
    recursive (direct build_S) S.  (Both conjectural; this *tests* them.)"""
    A = _pentagon()
    At = BPSAtlas(A)
    nk, _ = At.mutate((), 0, "fwd")
    r = At.verify_spec_equivalence((), nk, K=8)
    check("verify_spec_equivalence: isomorphic", r["isomorphic"])
    check("verify_spec_equivalence: specs equivalent (same S)", r["spec_equivalent"])
    check("verify_spec_equivalence: full leading-order index agreement", r["full"])
    rr = At.verify_spec_equivalence((), nk, K=6, vs_recursive=True)
    check("spec-S coincides with the recursive direct-S",
          rr["recursive"]["built"] and rr["recursive"]["coincides"])


def test_iso_check_at_add_time_folds_atlas():
    """The author, 2026-06-28: 'iso-checking new bps charts should be done when the
    charts are added.'  The pentagon's 4 charts fold to ONE quiver-iso class as
    they are materialized (fundamental domain = {root}); each non-root chart is
    recorded as an automorphism-image of root.  `iso_check=False` defers it."""
    A = _pentagon()
    At = BPSAtlas(A)
    for k in [((0, "fwd"),), ((1, "inv"),), ((0, "fwd"), (1, "fwd"))]:
        At.chart(k)                              # materialize → classify at add
    check("4 charts materialized", len(At.keys()) == 4)
    check("all fold to one iso-class (fundamental domain = {root})",
          At.representatives() == [()])
    check("non-root charts recorded as automorphism-images of root",
          all(not At.iso_class(k)["new_class"]
              and At.iso_class(k)["representative"] == ()
              for k in At.keys() if k != ()))
    check("orbit(root) is all 4 charts", len(At.orbit(())) == 4)
    check("recognize() reads the add-time classification",
          At.recognize(((0, "fwd"),)) is not None
          and At.recognize(((0, "fwd"),))[0] == ())
    # iso_check=False defers classification (cheap adds); still queryable lazily
    At2 = BPSAtlas(A, iso_check=False)
    At2.chart(((0, "fwd"),))
    check("iso_check=False: no eager fold (only root classified)",
          len(At2.representatives()) == 1 and len(At2.keys()) == 2)
    check("iso_class still classifies lazily on demand",
          At2.iso_class(((0, "fwd"),))["representative"] == ())


def test_folded_graph_self_loops_are_automorphisms():
    """The author, 2026-06-28: merging isomorphic charts must carefully transport the
    mutation edges.  The pentagon's chart graph folds to ONE representative
    (root); the quotient self-loops are genuine automorphisms reproducing ρ² and
    ρ³=ρ⁻² (so the edge transport `transition∘witness⁻¹` is correct), and image
    charts are never re-expanded (C3 is never built — the work-saving)."""
    A = _pentagon()
    At = BPSAtlas(A)
    fg = At.folded_graph()
    check("pentagon folds to one representative", fg["representatives"] == [()])
    check("quotient closed (finite type)", fg["closed"])
    check("fold_ratio > 1 (charts saved)", fg["fold_ratio"] > 1.0)
    check("two rotation self-loops",
          len(fg["self_loop_automorphisms"]) == 2)

    def rho_pow(l, j):
        for _ in range(j):
            l = tuple(A.rho(l))
        return l
    powers = set()
    for sl in fg["self_loop_automorphisms"]:
        iso = sl["automorphism"]
        res = _battery(iso, _LABELS, trace_K=6)
        check("self-loop is a certified automorphism"
              + ("" if all(res.values()) else f" - {res}"), all(res.values()))
        sig = {l: tuple(next(iter(iso.map(Element({l: ONE})).terms)))
               for l in [(1, 0), (0, 1)]}
        j = next((j for j in range(5)
                  if all(sig[l] == rho_pow(l, j) for l in [(1, 0), (0, 1)])), None)
        check(f"self-loop is a ρ-power (got ρ^{j})", j is not None)
        powers.add(j)
    check("the two self-loops are ρ² and ρ³ (=ρ⁻²)", powers == {2, 3})
    check("image charts not re-expanded (C3 never built)",
          ((0, "fwd"), (1, "fwd")) not in At.keys())


def test_automorphism_lazy_memoization():
    """The author, 2026-06-28: 'automorphisms greatly expand power of memoized info'
    (+ mult, + ρ), but 'orbits are infinite — be careful', so the sharing is
    LAZY (bounded BFS on a miss; orbits never enumerated).  A root automorphism
    is exactly trace/inner-product-invariant and multiply/ρ-covariant, so one
    compute serves the orbit-neighbours within depth."""
    A = _pentagon()
    At = BPSAtlas(A)
    info = At.register_automorphisms()
    check("registered automorphism generators", info["n_generators"] >= 1)
    Ref = BPSAtlas(A)                       # no automorphisms = direct reference
    phi = At._aut_gens[0]

    def lab(x):
        return At._aut_label(phi, x)
    a, b = (1, 0), (0, 1)
    # invariants
    check("Tr matches direct compute", At.trace(a, 8) == Ref.trace(a, 8))
    check("Tr(φa) == Tr(a) (invariant, lazily shared)",
          At.trace(lab(a), 8) == At.trace(a, 8))
    check("I matches direct, and I(φa,φb) == I(a,b)",
          At.inner_product(a, b, 8) == Ref.inner_product(a, b, 8)
          and At.inner_product(lab(a), lab(b), 8) == At.inner_product(a, b, 8))
    # covariants
    ra = Ref.rho(a)
    check("ρ(a) ok and ρ(φa) == φ(ρa) (covariant)",
          At.rho(a) == ra and At.rho(lab(a)) == lab(ra))
    prod = Ref.multiply(a, b)
    check("mult(a,b) ok and L_{φa}·L_{φb} == φ(L_a·L_b) (covariant)",
          At.multiply(a, b) == prod and At.multiply(lab(a), lab(b)) == phi.map(prod))
    # the reuse actually saves compute: the whole ρ-orbit → one real root.trace
    At2 = BPSAtlas(A)
    At2.register_automorphisms()
    calls = {"n": 0}
    orig = At2._root.trace

    def counting(a, K, _o=orig, _c=calls):
        _c["n"] += 1
        return _o(a, K)
    At2._root.trace = counting
    orbit, x = [a], a
    for _ in range(6):
        x = At2._aut_label(At2._aut_gens[0], x)
        orbit.append(x)
    for x in orbit:
        At2.trace(x, 8)
    check(f"ρ-orbit of {len(set(orbit))} labels served by 1 real compute "
          f"(got {calls['n']})", calls["n"] == 1)


def test_complete_finite_type_atlas():
    """The author, 2026-06-28: for a finite-type KAlgebra the atlas can be completed —
    every chart materialized.  The pentagon's exchange graph closes (finite);
    a tiny cap leaves it open (closed=False), and complete() and folded_graph()
    agree on the class count."""
    A = _pentagon()
    At = BPSAtlas(A)
    c = At.complete()
    check("pentagon completes (finite type)", c["closed"])
    check(f"pentagon n_charts == 4 (got {c['n_charts']})", c["n_charts"] == 4)
    check("every chart materialized", set(c["keys"]) <= set(At.keys()))
    # the class count agrees with the fundamental-domain (folded_graph) view
    # (pentagon is a full-rank square quiver, so the recognition applies)
    check("pentagon quiver is square (classification available)", c["classified"])
    check(f"one quiver-iso class (got {c['n_classes']})", c["n_classes"] == 1)
    # a tiny cap leaves the graph open (closed=False, the infinite/large signal)
    capped = BPSAtlas(A).complete(max_charts=2)
    check("max_charts cap reported as not-closed", not capped["closed"])


def test_mutation_complete_folds_to_finite_charts():
    """The author, 2026-06-28: the **mutation-complete folded atlas** — every chart
    reachable by mutating any node, folded by chart-isomorphism (node-perm +
    Γ-automorphism).  Mutating the pentagon's BPS quiver yields an *isomorphic*
    quiver, so by the atlas's definitions the pentagon is a **single chart with
    two outgoing mutations back to itself** — finite even though the raw node
    charges grow without bound.  Runs on the charge-space quiver mutation
    directly (no necklace budget), so the fold always closes for a finite
    theory."""
    mc = BPSAtlas(_pentagon()).mutation_complete()
    check("pentagon: a single chart", mc["n_charts"] == 1)
    check("pentagon: two outgoing mutations", len(mc["edges"]) == 2)
    check("pentagon: both mutations back to itself (self-loops)",
          mc["self_loops"] == 2)
    check("pentagon: mutation-complete (closed)", mc["closed"])
    # completeness certificate: rank-regular (one edge per node) + square fold
    check("pentagon: rank-regular, classified fold",
          mc["rank_regular"] and mc["classified"])
    # A₃: the interior-node mutation reaches a fourth chart
    a3 = BPSKAlgebra(pairing=[[0, 1, 0], [-1, 0, 1], [0, -1, 0]],
                     node_charges=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                     spec=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    mc3 = BPSAtlas(a3).mutation_complete()
    check(f"A₃: four charts (got {mc3['n_charts']})", mc3["n_charts"] == 4)
    check("A₃: every chart has 3 node-mutations", len(mc3["edges"]) == 12)
    check("A₃: mutation-complete (closed)", mc3["closed"])
    check("A₃: rank-regular, classified fold",
          mc3["rank_regular"] and mc3["classified"])
    # [A1,E6] — the keystone: the necklace fold could not close it (E-type charts
    # need an unbounded local-move budget); the charge-mutation fold closes it to
    # 67 charts, rank-regular, ~0.3s.
    e6 = BPSKAlgebra(
        pairing=[[0, 1, 0, 0, 0, 0], [-1, 0, -1, 0, 0, 0], [0, 1, 0, 1, 0, 1],
                 [0, 0, -1, 0, -1, 0], [0, 0, 0, 1, 0, 0], [0, 0, -1, 0, 0, 0]],
        node_charges=[(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0),
                      (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1)],
        spec=[(1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1),
              (0, 0, 0, 0, 1, 0), (0, 0, 0, 1, 0, 0), (0, 1, 0, 0, 0, 0)])
    mce6 = BPSAtlas(e6).mutation_complete()
    check(f"[A1,E6]: closes to 67 charts (got {mce6['n_charts']})",
          mce6["n_charts"] == 67 and mce6["closed"] and mce6["rank_regular"])
    # non-square (frozen flavour node) → chart-iso fold undefined, classified=False
    sq = BPSAtlas(BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=[(1, 0)],
                              spec=[(1, 0)])).mutation_complete()
    check("non-square (sqed1): classified=False, not closed",
          (not sq["classified"]) and (not sq["closed"])
          and sq["n_charts"] is None)
    # the `keep` predicate restricts the fold to a sub-class of charts; excluded
    # charts are walls.  keep≡True reproduces the unrestricted fold (0
    # walls); keep≡False leaves just the root with every mutation a wall.
    A2 = BPSAtlas(_pentagon())
    mk_t = A2.mutation_complete(keep=lambda m: True)
    check("keep≡True ≡ unrestricted fold, 0 walls",
          mk_t["n_charts"] == 1 and len(mk_t["edges"]) == 2 and mk_t["walls"] == 0)
    mk_f = A2.mutation_complete(keep=lambda m: False)
    check("keep≡False → root only, every mutation a wall",
          mk_f["n_charts"] == 1 and len(mk_f["edges"]) == 0 and mk_f["walls"] == 2)


def test_raw_quiver_iso_matches_chart_isomorphism():
    """The charge-level `_raw_quiver_iso` (used by `mutation_complete`, no chart
    materialization) must agree with the canonical materializing
    `chart_isomorphism` — the faithfulness guarantee that lets the fold run on
    pure quiver data."""
    agree = mism = 0
    for pairing, charges, spec in [
        ([[0, 1, 0], [-1, 0, 1], [0, -1, 0]],
         [(1, 0, 0), (0, 1, 0), (0, 0, 1)], None),
        ([[0, 1, 0, 0], [-1, 0, 1, 0], [0, -1, 0, 1], [0, 0, -1, 0]],
         [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)], None),
    ]:
        kw = dict(pairing=pairing, node_charges=charges)
        A = BPSKAlgebra(spec=charges if spec is None else spec, **kw)
        P = [list(r) for r in A.lattice.pairing]
        At = BPSAtlas(A, max_local_moves=3 * len(P), iso_check=False)
        rootc = [tuple(v) for v in A.node_charges]
        for k in range(len(P)):
            for d in ("fwd", "inv"):
                try:
                    nk, _ = At.mutate((), k, d)
                except Exception:
                    continue
                mc = [tuple(v) for v in At.chart(nk).node_charges]
                raw = BPSAtlas._raw_quiver_iso(P, rootc, mc)
                mat = At.chart_isomorphism((), nk) is not None
                if raw == mat:
                    agree += 1
                else:
                    mism += 1
    check(f"_raw_quiver_iso agrees with chart_isomorphism ({agree} pairs)",
          mism == 0 and agree > 0)


def _showcase():
    """Pentagon (A₂) showcase — the export demonstrator (printed, not asserted)."""
    A = _pentagon()
    cert = BPSAtlas(A).certificate(trace_K=8)
    print("\n--- pentagon (A₂) showcase: cluster mutation preserves the K_𝖖 axioms ---")
    print(f"  rotation period (chambers to return) : {cert['period']}")
    print(f"  every mutation edge battery-green    : "
          f"{all(all(r.values()) for r in cert['edge_batteries'])}")
    print(f"  multiply chart-invariant             : {cert['multiply_chart_invariant']}")
    print(f"  Schur index I_a identical in every chamber : {cert['trace_chart_invariant']}")
    print(f"  rotation monodromy = ρ²              : {cert['monodromy']['is_rho2']}")
    print(f"  Schur index I_(1,0)(q)               : {A.trace((1, 0), 8)}")


if __name__ == "__main__":
    for fn in [test_root, test_single_mutate_certified, test_path_iso_certified,
               test_transition_between_charts, test_monodromy_is_rho2,
               test_as_object_pairwise, test_F_transport_matches_native,
               test_multiply_shared_across_charts, test_cache_solved_once,
               test_certificate_pentagon, test_intrinsic_trace_and_inner_product,
               test_intrinsic_memoization_layer,
               test_cross_validate_spec_charts_agree,
               test_build_S_charts_provenance_and_conjecture,
               test_loop_automorphisms, test_summary,
               test_automorphism_group_is_Z5,
               test_chart_isomorphism_recognizes_quiver_iso,
               test_recognize_folds_chart_onto_representative,
               test_chart_isomorphism_none_when_quivers_differ,
               test_verify_spec_equivalence,
               test_iso_check_at_add_time_folds_atlas,
               test_folded_graph_self_loops_are_automorphisms,
               test_automorphism_lazy_memoization,
               test_complete_finite_type_atlas,
               test_mutation_complete_folds_to_finite_charts,
               test_raw_quiver_iso_matches_chart_isomorphism]:
        fn()
    _showcase()
    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print(f"All {len(PASS)} BPSAtlas tests passed.")
