"""`BPSAtlas` for the SU(2) gauge-theory family (the design record catalogue).

Pins the `src/abe/su2_family_atlas.py` builders and the atlas surface for

    pure SU(2),  SU(2)+N_f=1..4,  SU(2)-SU(2) quiver,

each seeded from the known BPS quiver (spec = matter·pure-gauge):
`pure_ade.SUN_Nf(2,Nf)` / `SUN_bifund(2,2)` transcribed into the canonical
`BPSKAlgebra`.  Certifies:

  * every theory builds an atlas with the expected rank;
  * pure SU(2) rotates with period 4 and rotation monodromy = ρ²;
  * the chamberless higher-rank corners pass a bounded mutation-window iso
    battery (the infinite-automorphism regime);
  * the **node-drop RG iso** (identity-on-labels, guaranteed) passes the light
    structural battery on every matter theory;
  * the **cone iso** is ray-certified (flavour-neutral gauge rays) for pure SU(2)
    and N_f=1, and the cone-presentation availability is as expected;
  * intrinsic memoization is consistent (cached == fresh).

Run:  `python3 run_tests.py`
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalgebra import Element
from laurent_poly import LaurentPoly
from bps_atlas import BPSAtlas
from su2_family_atlas import (
    su2_family_names, su2_family_entry, su2_family_charts,
    su2_family_rg_iso, su2_family_cone, su2_family_cone_iso,
    su2_family_cone_rays, su2_family_object,
    su2_family_cone_cartan_hom, su2_family_cone_reduced,
    su2_family_cone_index_matches_bps, su2_family_cone_bps_iso,
)

PASS = []
FAIL = []
ONE = LaurentPoly.one()

_EXPECTED_RANK = {
    "pure_SU2": 2, "SU2_Nf1": 3, "SU2_Nf2": 4,
    "SU2_Nf3": 5, "SU2_Nf4": 6, "SU2_SU2": 5,
}
_HAS_CONE = {"pure_SU2", "SU2_Nf1", "SU2_Nf2", "SU2_Nf3"}
_RAY_CERTIFIED_CONE = {"pure_SU2", "SU2_Nf1"}


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


def _ray_battery(iso, rays, *, with_mult=True):
    se = [Element({tuple(r): ONE}) for r in rays]
    te = [iso.map(e) for e in se]
    out = {"unit": iso.verify_unit(),
           "round_trip": iso.verify_round_trip(se, te),
           "rho": iso.verify_rho_equivariant(se, te)}
    if with_mult:
        pairs = [(se[i], se[j]) for i in range(len(se)) for j in range(len(se))]
        tpairs = [(iso.map(a), iso.map(b)) for a, b in pairs]
        out["multiplicative"] = iso.verify_multiplicative(pairs, tpairs)
    return out


# ---------------------------------------------------------------------------
def test_family_builds_with_expected_rank():
    charts = su2_family_charts()
    check("family has all six theories",
          list(charts) == su2_family_names() and len(charts) == 6)
    ok = True
    for nm, A in charts.items():
        At = BPSAtlas(A)
        ok = ok and (len(A.lattice.pairing) == _EXPECTED_RANK[nm])
        ok = ok and (At.root is A)
    check("every theory builds a BPSAtlas with the expected rank", ok)


def test_nf1_equals_handbuilt_chart():
    # SUN_Nf(2,1) must reproduce the hand-built canonical bps_su2_nf1 chart.
    from bps_su2_nf1 import build_bps_su2_nf1
    h = build_bps_su2_nf1()
    a = su2_family_entry("SU2_Nf1").chart
    same_nodes = [tuple(g) for g in h.node_charges] == [tuple(g) for g in a.node_charges]
    same_spec = [tuple(g) for g in h.spec] == [tuple(g) for g in a.spec]
    check("SU2_Nf1 chart == hand-built bps_su2_nf1 (nodes + spec)",
          same_nodes and same_spec)


def test_pure_su2_rotation_monodromy_is_rho2():
    A = su2_family_entry("pure_SU2").chart
    At = BPSAtlas(A)
    cert = At.certificate(trace_K=4)
    check("pure SU(2) rotates with period 4 (chart-data periodicity)",
          cert["period"] == 4)
    check("pure SU(2) multiply is chart-invariant", cert["multiply_chart_invariant"])
    # The rotation realizes the *automorphism* ρ² on the sampled labels — and
    # ρ² ≠ id (ρ is infinite-order, below); "period 4" is NOT an automorphism order.
    check("pure SU(2) rotation monodromy = ρ² (on sampled labels)",
          cert["monodromy"]["is_rho2"])


def _rho_orbit_closes(A, start, steps=40):
    cur = tuple(start)
    for i in range(steps):
        cur = tuple(A.rho(cur))
        if cur == tuple(start):
            return i + 1
    return None


def test_rho_is_infinite_order_on_asymptotically_free():
    # The asymptotically-free theories (β>0) have ρ of INFINITE order — the
    # Witten effect: the running θ-angle shifts a dyon's electric charge each ρ,
    # so the orbit of a gauge node drifts and never closes
    # (pure SU(2): (1,0)→(1,-4)→(1,-8)→…).
    ok = True
    for nm in ("pure_SU2", "SU2_Nf1", "SU2_Nf2", "SU2_Nf3", "SU2_SU2"):
        A = su2_family_entry(nm).chart
        ok = ok and (_rho_orbit_closes(A, A.node_charges[0], steps=40) is None)
    check("ρ is infinite-order on the asymptotically-free SU(2) theories", ok)


def test_nf4_conformal_rho_is_an_involution():
    # The conformal SU(2)+N_f=4 (β=0) is genuinely different: ρ²=id on every
    # label — ρ is an *involution* (finite order 2), NOT infinite-order.  Physical
    # reason: it is a *conformal* gauge theory — no running, so
    # **no Witten effect**, so the electric charge does not drift and ρ²=id.  The
    # AF/conformal distinction shows up cleanly as ord(ρ): ∞ for the
    # asymptotically-free SU(2) theories, 2 for the conformal one.
    A = su2_family_entry("SU2_Nf4").chart
    sweep = [tuple(g) for g in A.node_charges] + [
        (2, -1, 1, 0, 1, 0), (1, -2, 1, 1, 0, 0), (3, 0, 0, 1, 1, 1),
        (-2, 3, 1, 0, 0, 1), (1, -1, 1, 1, 1, 1)]
    check("N_f=4 (conformal): ρ² = id — ρ is an involution",
          all(tuple(A.rho(A.rho(l))) == l for l in sweep))


def test_chamberless_corners_window_iso_certified():
    # The higher-rank corners are chamberless within a small bound; a bounded
    # mutation-window iso battery must still certify (infinite-automorphism
    # regime).  Only spec-cooperating nodes necklace at max_local_moves=0, so we
    # try each node and require at least one certified mutation per theory.
    ok = True
    for nm in ("SU2_Nf1", "SU2_Nf3", "SU2_Nf4", "SU2_SU2"):
        A = su2_family_entry(nm).chart
        At = BPSAtlas(A)
        win = [tuple(A.identity())] + [tuple(g) for g in A.node_charges[:2]]
        certified = 0
        for k in range(len(A.node_charges)):
            try:
                _, iso = At.mutate((), k)
            except ValueError:
                continue                       # node does not necklace here
            if all(_ray_battery(iso, win).values()):
                certified += 1
            if certified >= 1:
                break
        ok = ok and (certified >= 1)
    check("chamberless corners: a bounded mutation-window iso battery certifies",
          ok)


def test_rg_nodedrop_iso_light_all_matter():
    # The node-drop RG presentation is identity-on-labels to the BPS chart and
    # iso by construction (guaranteed if aux match) — light structural battery.
    ok = True
    for nm in ("SU2_Nf1", "SU2_Nf2", "SU2_Nf3", "SU2_Nf4", "SU2_SU2"):
        e = su2_family_entry(nm)
        iso = su2_family_rg_iso(nm)
        matter = tuple(e.chart.node_charges[e.matter_indices[0]])
        gauge = tuple(e.chart.node_charges[0])
        res = _ray_battery(iso, [tuple(e.chart.identity()), matter, gauge])
        ok = ok and all(res.values())
    check("node-drop RG iso (light) certified on all 5 matter theories", ok)


def test_pure_su2_has_no_matter_to_drop():
    e = su2_family_entry("pure_SU2")
    check("pure SU(2) has empty matter_indices", e.matter_indices == ())


def test_cone_iso_ray_certified():
    ok = True
    for nm in _RAY_CERTIFIED_CONE:
        iso = su2_family_cone_iso(nm)
        rays = su2_family_cone_rays(nm)
        ok = ok and (iso is not None) and all(_ray_battery(iso, rays).values())
    check("cone iso ray-certified on pure SU(2) and N_f=1", ok)


def test_cone_index_matches_bps_via_flavour_reduction():
    # `BPSKAlgebra` carries only abelian (Cartan) flavour, but the N_f=2/3 cones
    # carry non-abelian flavour (Spin(4)=SU(2)×SU(2), SU(4)).  The correct cone↔BPS
    # comparison reduces the cone to its Cartan first (base_change along the
    # Cartan restriction) — then the reduced cone's Schur index reproduces the BPS
    # chart's.  Certified on the vacuum (the flavoured index) for N_f=1/2/3.
    ok = True
    for nm in ("SU2_Nf1", "SU2_Nf2", "SU2_Nf3"):
        red = su2_family_cone_reduced(nm)
        # the reduction lands in the abelian Cartan (matching BPS's flavour group)
        ok = ok and type(red.coefficient_ring()).__name__ == "AbelianZPlusRing"
        ok = ok and su2_family_cone_index_matches_bps(nm, K=6)
    # N_f=2 needs a genuine non-abelian→abelian hom; N_f=1 is already abelian
    ok = ok and (su2_family_cone_cartan_hom("SU2_Nf2") is not None)
    ok = ok and (su2_family_cone_cartan_hom("SU2_Nf1") is None)
    check("cone Schur index matches BPS via flavour reduction (N_f=1/2/3)", ok)


def test_su2_nf2_explicit_cone_bps_iso():
    # The non-abelian-flavour keystone, built the author's way (2026-06-28): match
    # the two seeds — 't Hooft H_0 → tropical (1,0;0,0) and Wilson w_1 →
    # (0,-1;0,0) — and the rest is *determined* by the BPS multiply.  The gauge
    # tower is the piecewise-linear σ map (ρ is NOT linear on tropical charges):
    # H_n → (1,n) for n≤0, (-1,2-n) for n≥1, so H_1 → (-1,1), H_{-1} → (1,-1)
    # (W·H_0 ∋ H_1, H_{-1}).  The Cartan flavour weight rides into the BPS slots.
    iso = su2_family_cone_bps_iso("SU2_Nf2")
    gens = [((), (0, 0)), (((0, 1),), (0, 0)), (((("W", 1), 1),), (0, 0)),
            (((1, 1),), (0, 0)), (((-1, 1),), (0, 0)), (((2, 1),), (0, 0)),
            ((), (1, 0)), ((), (0, 1)), ((), (1, 1))]
    S = [Element({l: ONE}) for l in gens]
    T = [iso.map(s) for s in S]
    pairs = [(S[i], S[j]) for i in range(len(S)) for j in range(len(S))]
    tpairs = [(iso.map(a), iso.map(b)) for a, b in pairs]
    # Structural battery is EXACT: the cone keeps flavour as a Cartan weight in
    # the label (q-Laurent coefficients) — same shape as the BPS flavour slots —
    # so multiply needs no coefficient bridge.  81/81 products close.
    structural = (iso.verify_unit()
                  and iso.verify_round_trip(S, T)
                  and iso.verify_multiplicative(pairs, tpairs)
                  and iso.verify_rho_equivariant(S, T))
    check("N_f=2 explicit cone↔BPS iso: structural battery exact "
          "(unit/round-trip/multiplicative/ρ)", structural)
    # Trace (the Schur index) on the flavour-neutral gauge sector — clean at K=4.
    # (The charged-label trace is the cone's documented Weyl-recentering
    # convention; the vacuum index to q^6 is certified in
    # test_cone_index_matches_bps_via_flavour_reduction.)
    neutral = [Element({l: ONE}) for l in gens[:6]]
    nt = [iso.map(s) for s in neutral]
    check("N_f=2 explicit cone↔BPS iso: trace-equivariant on the gauge sector "
          "(K=4, via flavour reduction)",
          iso.verify_trace_equivariant(neutral, nt, K=4))


def test_explicit_cone_bps_iso_scope():
    # pure SU(2) / N_f=1 delegate to the already-certified object-layer isos;
    # N_f=3 is None (its BPS gauge H-tower is matter-dressed — H_0·w_1 yields a
    # single clean gauge term, not the H_1,H_{-1} bifurcation — so the
    # flavour-neutral σ-table does not close); N_f=4 has no cone.
    ok = (su2_family_cone_bps_iso("pure_SU2") is not None
          and su2_family_cone_bps_iso("SU2_Nf1") is not None
          and su2_family_cone_bps_iso("SU2_Nf3") is None
          and su2_family_cone_bps_iso("SU2_Nf4") is None)
    check("explicit cone↔BPS iso scope: pure/N_f=1 delegate, N_f=3/4 None", ok)


def test_cone_presentation_availability():
    ok = True
    for nm in su2_family_names():
        has = su2_family_cone(nm) is not None
        ok = ok and (has == (nm in _HAS_CONE))
    check("cone presentation present exactly for pure SU(2) + N_f=1/2/3", ok)


def test_intrinsic_memoization_consistent():
    A = su2_family_entry("pure_SU2").chart
    At = BPSAtlas(A)
    a, b = (1, 0), (-1, 2)
    At.clear_intrinsic_caches()
    first = At.multiply(a, b)
    cached = At.multiply(a, b)
    check("intrinsic memoized multiply is consistent (cached == fresh)",
          first == cached)


def test_chart_iso_recognition_folds_su2():
    # Chart-iso recognition genuinely folds the SU(2) atlases.  The SU(2) gauge
    # pairing is Kronecker `⟨γ_1,γ_2⟩=2`, so the node-charge matrix has det ±2
    # (not unimodular) — but the Γ-automorphism g = M'σ·M⁻¹ is still integer
    # (e.g. g = M·M⁻¹ = I for root↔root), so recognition works once it checks the
    # integrality of g rather than of M.  pure SU(2) folds onto a single class.
    A = su2_family_entry("pure_SU2").chart
    At = BPSAtlas(A)
    # root is quiver-iso to itself (identity witness) — not None
    ok = At.chart_isomorphism((), ()) is not None
    fg = At.folded_graph(max_steps=12)
    # genuine folding: more charts visited than classes, with self-loop
    # automorphisms (the cluster-modular self-loops)
    ok = ok and fg["fold_ratio"] > 1.0 and len(fg["self_loop_automorphisms"]) >= 1
    check("chart-iso recognition folds the pure-SU(2) atlas (fold_ratio>1)", ok)


def test_family_objects_build():
    # Each theory packages as a KAlgebraObject (Stage-4 export shape).
    ok = True
    for nm in su2_family_names():
        obj = su2_family_object(nm)
        ok = ok and ("bps" in obj.keys())
    # pure SU(2) reuses the richer object (abe/bps/cone); N_f=1 too (incl. a
    # node-drop RG leg); the rest get a fresh bps+rg object.
    o_pure = su2_family_object("pure_SU2")
    o_nf1 = su2_family_object("SU2_Nf1")
    o_nf2 = su2_family_object("SU2_Nf2")
    ok = ok and {"abe", "bps", "cone"} <= set(o_pure.keys())
    ok = ok and {"cone", "bps"} <= set(o_nf1.keys())
    ok = ok and set(o_nf2.keys()) == {"bps", "rg"}
    check("every theory packages as a KAlgebraObject (export shape)", ok)


def test_fresh_objects_rg_bps_witness_certified():
    # The fresh bps+rg objects (N_f=2/3/4, SU(2)-SU(2)) carry a node-drop RG
    # witness that passes the light structural battery.
    ok = True
    for nm in ("SU2_Nf2", "SU2_Nf3", "SU2_Nf4", "SU2_SU2"):
        obj = su2_family_object(nm)
        A = obj.realization("bps")
        win = [tuple(A.identity())] + [tuple(g) for g in A.node_charges[:2]]
        iso = obj.iso("rg", "bps")
        ok = ok and all(_ray_battery(iso, win).values())
    check("fresh family objects' rg↔bps witness certified (structural)", ok)


if __name__ == "__main__":
    for fn in [
        test_family_builds_with_expected_rank,
        test_nf1_equals_handbuilt_chart,
        test_pure_su2_rotation_monodromy_is_rho2,
        test_rho_is_infinite_order_on_asymptotically_free,
        test_nf4_conformal_rho_is_an_involution,
        test_chamberless_corners_window_iso_certified,
        test_rg_nodedrop_iso_light_all_matter,
        test_pure_su2_has_no_matter_to_drop,
        test_cone_iso_ray_certified,
        test_cone_index_matches_bps_via_flavour_reduction,
        test_su2_nf2_explicit_cone_bps_iso,
        test_explicit_cone_bps_iso_scope,
        test_cone_presentation_availability,
        test_intrinsic_memoization_consistent,
        test_chart_iso_recognition_folds_su2,
        test_family_objects_build,
        test_fresh_objects_rg_bps_witness_certified,
    ]:
        fn()
    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print(f"All {len(PASS)} BPSAtlas SU(2)-family tests passed.")
