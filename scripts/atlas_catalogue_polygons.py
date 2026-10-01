"""`BPSAtlas` catalogue over the polygon / SQED tower, with the
**RG node-deletion builder + isos**, the **cone-presentation ray-multiply/ρ
test**, and a **benchmark** of the routes.

Ten theories in the (2n+1)-/(2n)-gon family, each an `A_𝖖[T]` with three
certified presentations tied by `KAlgebraIso`s:

    BPS chart  ──(node-drop RG, identity iso)──  RG realization
        │
        └────(object-layer iso, section change)──  Cone presentation

The families (polygon ↔ AD type):

  * trivial AD      pentagon [A1,A2] · heptagon [A1,A4] · nonagon [A1,A6]
  * u1-flavoured    hexagon a3 [A1,A3] · octagon a5 [A1,A5] · decagon a7 [A1,A7]
  * u(1)-gauged     U1hexagon · u1octagon · u1decagon = U1A1AoddKAlg(1/2/3)
  * base            sqed1 = U(1)+N_f=1

Per theory:
  1. **BPS chart** seeds a `BPSAtlas`; the rotation `certificate` (period /
     per-edge battery / multiply- & Schur-index chart-invariance / monodromy=ρ²)
     where tractable (rank ≤ 5; rank ≥ 6 is cost-prohibitive — a7's full
     certificate is >30 min, so the heavy tier reports the cone test only).
  2. **RG realization** = `DirectionalSingleNodeRG` (drop the terminal node γ₁ —
     the IR sub-chamber lifted by a closed-form S_RG, *no UV F-solve*), with the
     **identity `KAlgebraIso`** to the BPS chart certified (full battery incl.
     trace on the light tier; structural-only beyond, the RG trace being the
     slow generic exact-FS).
  3. **Cone presentation** (`U1SquareKAlg` / `PentagonKAlg` / `U1A1AoddKAlg(k)` /
     finite a3/a5/a7 / `A1A2kKAlg(3)`): `BPSAtlas.test_cone_presentation` checks
     the cone's **ray-multiplication** and **ρ** against the BPS root via the
     certified iso — cheap (generators only) and flavour-aware (canonical R-form
     / section change; ρ on flavoured cones holds up to the μ-unit section
     torsor, reported as `rho_mod_flavour`).
  4. **Benchmark**: closed-form cone multiply vs BPS Schur vs the RG route.

Run (background; rank ≥ 5 is slow):

    PYTHONPATH=. python scripts/atlas_catalogue_polygons.py            # all 10
    PYTHONPATH=. python scripts/atlas_catalogue_polygons.py --light    # light tier only
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "implementations"))

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly
from bps_kalgebra import BPSKAlgebra
from bps_atlas import BPSAtlas
from directional_subquiver_rg import (
    DirectionalSingleNodeRG, certify_directional_vs_bps)

_ONE = LaurentPoly.one()


# ---------------------------------------------------------------------------
# per-theory bundle: (bps chart, cone presentation, cone↔bps iso, cone label)
# ---------------------------------------------------------------------------
def _u1gauged_bundle(k):
    """U(1)-gauged [A1,A_{2k+1}] = U1A1AoddKAlg(k): bps via the gauged-quiver
    chart, cone↔bps iso via the octagon_objects E-sign machinery (any k)."""
    from u1a1aodd_kalg import U1A1AoddKAlg, gauged_quiver_bps
    from octagon_objects import _pick_e_sign, _charge_maps
    intrinsic = U1A1AoddKAlg(k)
    bps = gauged_quiver_bps(k)
    s = _pick_e_sign(intrinsic, bps)
    fwd, inv = _charge_maps(intrinsic, bps, s)
    iso = KAlgebraIso(intrinsic, bps,
                      lambda l: Element({fwd(l): _ONE}),
                      lambda c: Element({inv(c): _ONE}), name=f"u1[{k}][cone→bps]")
    return bps, intrinsic, iso, f"U1A1AoddKAlg({k})"


def bundle(name):
    if name == "sqed1":
        from sqed1_object import sqed1_object
        o = sqed1_object(); return o.realization("bps"), o.realization("cone"), o.iso("cone", "bps"), "U1SquareKAlg"
    if name == "pentagon":
        from finite_kalgebra_objects import kalgebra_object
        o = kalgebra_object("pentagon"); return o.realization("bps"), o.realization("closed-form"), o.iso("closed-form", "bps"), "PentagonKAlg"
    if name == "hexagon":
        from finite_kalgebra_objects import kalgebra_object
        o = kalgebra_object("a3"); return o.realization("bps"), o.realization("cone-frozen"), o.iso("cone-frozen", "bps"), "FiniteA3 (u1)"
    if name == "U1hexagon":
        return _u1gauged_bundle(1)
    if name == "heptagon":
        from finite_kalgebra_objects import kalgebra_object
        o = kalgebra_object("heptagon"); return o.realization("bps"), o.realization("closed-form"), o.iso("closed-form", "bps"), "HeptagonKAlg"
    if name == "octagon":
        from finite_kalgebra_objects import kalgebra_object
        o = kalgebra_object("a5"); return o.realization("bps"), o.realization("cone-frozen"), o.iso("cone-frozen", "bps"), "FiniteA5 (u1)"
    if name == "u1octagon":
        return _u1gauged_bundle(2)
    if name == "nonagon":
        from nonagon_objects import nonagon_object
        o = nonagon_object(); return o.realization("bps"), o.realization("a1a2k"), o.iso("a1a2k", "bps"), "A1A2kKAlg(3)"
    if name == "decagon":
        from finite_kalgebra_objects import kalgebra_object
        o = kalgebra_object("a7"); return o.realization("bps"), o.realization("cone-frozen"), o.iso("cone-frozen", "bps"), "FiniteA7 (u1)"
    if name == "u1decagon":
        return _u1gauged_bundle(3)
    raise KeyError(name)


# (name, tier, atlas_K, atlas_win, rg_trace_K)  tier ∈ {light, medium, heavy}
PLAN = [
    ("sqed1", "light", 8, 0, 6),
    ("pentagon", "light", 8, 0, 6),
    ("hexagon", "light", 6, 0, 6),
    ("U1hexagon", "light", 6, 0, 4),
    ("heptagon", "light", 6, 2, 4),
    ("octagon", "medium", 5, 2, None),
    ("u1octagon", "medium", 4, 2, None),
    ("nonagon", "heavy", None, None, None),
    ("decagon", "heavy", None, None, None),
    ("u1decagon", "heavy", None, None, None),
]


def _window(A, win):
    return None if not win else [A.identity()] + [tuple(g) for g in A.node_charges[:win]]


def _time(fn):
    t0 = time.time()
    r = fn()
    return r, time.time() - t0


def certify_theory(name, tier, K, win, rg_K):
    rec = {"name": name, "tier": tier}
    bps, cone, cone_iso, clabel = bundle(name)
    rec["rank"] = len(bps.lattice.pairing)
    rec["n_nodes"] = len(bps.node_charges)
    rec["flavour"] = str(bps.coefficient_ring())
    rec["cone"] = clabel
    At = BPSAtlas(bps)

    # ---- (0) complete the atlas — materialize the whole chart graph ----
    # Finite-type ⇒ finite chart graph, so a complete atlas is cheap (mutate +
    # classify only, no trace solves).  Records n_charts / n_classes / closed.
    comp, t_comp = _time(lambda: At.complete())
    rec["complete"] = {k: comp[k] for k in
                       ("n_charts", "n_classes", "classified", "closed")}
    rec["t_complete"] = t_comp

    # ---- (3) cone-presentation ray-multiply / ρ test (all tiers; cheap) ----
    # The cost is the BPS ground-truth side (per-ray-pair F-solve); at rank ≥ 7
    # the *flavoured* full sweep is slow, so the heavy tier samples a ray subset
    # (still the generators-only "efficient test" — full coverage on lighter tiers).
    cd = cone.cone_data()
    all_gens = [cd.from_cone_label(frozenset({g}), {g: 1})
                for g in sorted(cd.mult_gens())]
    gens = all_gens if tier != "heavy" else all_gens[:8]
    cres, t_cone = _time(lambda: At.test_cone_presentation(cone, cone_iso, gens=gens))
    rec["cone_test"] = cres
    rec["cone_total_rays"] = len(all_gens)
    rec["cone_subset"] = (tier == "heavy" and len(gens) < len(all_gens))
    rec["t_cone"] = t_cone

    # ---- (1) atlas rotation certificate (light/medium only) ----
    if tier in ("light", "medium"):
        def _cert():
            return At.certificate(trace_K=K, labels=_window(bps, win))
        try:
            cert, t_cert = _time(_cert)
            rec["period"] = cert["period"]
            rec["edges_ok"] = all(all(r.values()) for r in cert["edge_batteries"])
            rec["mult_inv"] = cert["multiply_chart_invariant"]
            rec["trace_inv"] = cert["trace_chart_invariant"]
            rec["rho2"] = cert["monodromy"]["is_rho2"]
            rec["all_ok"] = cert["all_ok"]
            rec["t_cert"] = t_cert
        except Exception as e:
            rec["atlas_error"] = f"{type(e).__name__}: {e}"
    else:
        rec["atlas_note"] = "rotation certificate cost-prohibitive (rank>=6)"

    # ---- (2) RG node-drop realization + identity iso ----
    nodes = [tuple(g) for g in bps.node_charges]
    if len(nodes) < 2:
        rec["rg"] = "base"
    else:
        try:
            R, t_rg = _time(lambda: DirectionalSingleNodeRG(
                [list(r) for r in bps.lattice.pairing], nodes,
                [tuple(s) for s in bps.spec], gamma_drop=nodes[0],
                rg_window=(rg_K or 4), arrange=True))
            rec["t_rg_build"] = t_rg
            cert_labels = [bps.identity()] + nodes[:max(2, (win or 3))]
            pairs = [(cert_labels[1], cert_labels[2])] if len(cert_labels) > 2 else None
            cv, t_cv = _time(lambda: certify_directional_vs_bps(
                R, bps, labels=cert_labels, pairs=pairs, trace_K=rg_K))
            rec["t_rg_certify"] = t_cv
            rec["rg_iso_ok"] = all(v for k, v in cv.items() if k != "iso")
            rec["rg_iso_trace"] = (rg_K is not None)
            rec["rg"] = "ok"
        except Exception as e:
            rec["rg"] = f"ERROR {type(e).__name__}: {e}"

    # ---- (4) benchmark: cone (closed-form) vs BPS vs RG, multiply on nodes ----
    a, b = nodes[0], nodes[-1]
    rec["bench"] = {}
    try:
        _, rec["bench"]["bps_mult"] = _time(lambda: bps.multiply(a, b))
    except Exception:
        rec["bench"]["bps_mult"] = None
    try:
        _cd = cone.cone_data()
        cg = [_cd.from_cone_label(frozenset({g}), {g: 1})
              for g in sorted(_cd.mult_gens())][:2]
        ca, cb = (cg + cg)[0], (cg + cg)[1]
        _, rec["bench"]["cone_mult"] = _time(lambda: cone.multiply(ca, cb))
    except Exception:
        rec["bench"]["cone_mult"] = None
    if rec.get("rg") == "ok" and rg_K is not None:
        try:
            R2 = DirectionalSingleNodeRG(
                [list(r) for r in bps.lattice.pairing], nodes,
                [tuple(s) for s in bps.spec], gamma_drop=nodes[0],
                rg_window=(rg_K or 4), arrange=True)
            _, rec["bench"]["rg_mult"] = _time(lambda: R2.multiply(a, b))
        except Exception:
            rec["bench"]["rg_mult"] = None
    return rec


def _ms(x):
    return "   n/a " if x is None else f"{x*1e3:7.2f}ms"


def fmt(rec):
    lines = [f"  · {rec['name']:11s} [{rec['tier']:6s}] rank={rec['rank']} "
             f"flav={rec['flavour']:22s} cone={rec['cone']}"]
    if "complete" in rec:
        cp = rec["complete"]
        ncl = "-" if cp["n_classes"] is None else str(cp["n_classes"])
        lines.append(
            f"      complete: n_charts={cp['n_charts']:3d} n_classes={ncl:>3s} "
            f"classified={cp['classified']!s:5s} closed={cp['closed']!s:5s}  "
            f"[{rec['t_complete']:.2f}s]")
    c = rec["cone_test"]
    sub = f"/{rec['cone_total_rays']} (subset)" if rec.get("cone_subset") else ""
    lines.append(
        f"      cone: rays={c['n_rays']:3d}{sub} prods={c['n_products']:5d} "
        f"ray_multiply_ok={c['ray_multiply_ok']!s:5s} rho_ok={c['rho_ok']!s:5s} "
        f"rho_mod_flavour={c['rho_ok_mod_flavour']!s:5s}  [{rec['t_cone']:.2f}s]")
    if "all_ok" in rec:
        lines.append(
            f"      atlas: period={rec['period']:2d} edges_ok={rec['edges_ok']!s:5s} "
            f"mult_inv={rec['mult_inv']!s:5s} trace_inv={rec['trace_inv']!s:5s} "
            f"rho2={rec['rho2']!s:5s} all_ok={rec['all_ok']!s:5s}  [{rec['t_cert']:.1f}s]")
    elif "atlas_error" in rec:
        lines.append(f"      atlas: ERROR {rec['atlas_error']}")
    else:
        lines.append(f"      atlas: {rec.get('atlas_note', '(skipped)')}")
    if rec.get("rg") == "base":
        lines.append("      RG  : tower BASE (1 node)")
    elif rec.get("rg") == "ok":
        lines.append(
            f"      RG  : node-drop iso_ok={rec['rg_iso_ok']!s:5s} "
            f"(trace={'yes' if rec['rg_iso_trace'] else 'structural'}) "
            f"build={rec['t_rg_build']:.3f}s certify={rec['t_rg_certify']:.1f}s")
    else:
        lines.append(f"      RG  : {rec.get('rg')}")
    bm = rec["bench"]
    lines.append(f"      bench multiply: cone={_ms(bm.get('cone_mult'))}  "
                 f"bps={_ms(bm.get('bps_mult'))}  rg={_ms(bm.get('rg_mult'))}")
    return "\n".join(lines)


def main():
    light = "--light" in sys.argv
    plan = [p for p in PLAN if p[1] == "light"] if light else PLAN
    print("=== BPSAtlas polygon/SQED catalogue: atlas + RG-builder + cone test + benchmark ===",
          flush=True)
    recs = []
    for name, tier, K, win, rg_K in plan:
        try:
            rec = certify_theory(name, tier, K, win, rg_K)
        except Exception as e:
            import traceback
            rec = {"name": name, "tier": tier, "fatal": f"{type(e).__name__}: {e}"}
            print(f"  · {name}: FATAL {rec['fatal']}", flush=True)
            traceback.print_exc()
            recs.append(rec)
            continue
        recs.append(rec)
        print(fmt(rec), flush=True)

    print("\n=== summary ===", flush=True)
    ct = [r for r in recs if "cone_test" in r]
    print(f"  cone ray-multiply OK: {sum(1 for r in ct if r['cone_test']['ray_multiply_ok'])}/{len(ct)}",
          flush=True)
    print(f"  cone ρ OK (strict): {sum(1 for r in ct if r['cone_test']['rho_ok'])}/{len(ct)} ; "
          f"ρ OK mod flavour: {sum(1 for r in ct if r['cone_test']['rho_ok_mod_flavour'])}/{len(ct)}",
          flush=True)
    atl = [r for r in recs if "all_ok" in r]
    print(f"  atlas certificate all_ok: {sum(1 for r in atl if r['all_ok'])}/{len(atl)}", flush=True)
    rg = [r for r in recs if r.get("rg") == "ok"]
    print(f"  RG node-drop iso certified: {sum(1 for r in rg if r['rg_iso_ok'])}/{len(rg)}", flush=True)
    return recs


if __name__ == "__main__":
    main()
