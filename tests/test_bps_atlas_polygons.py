"""`BPSAtlas` polygon/SQED catalogue regression.

Ten theories in the (2n)-/(2n+1)-gon family, each `A_𝖖[T]` with three certified
presentations tied by `KAlgebraIso`s — BPS chart, RG node-deletion realization,
and a cone presentation.  This test asserts, per theory:

  * **cone ray-multiplication / ρ** — `BPSAtlas.test_cone_presentation` validates
    the cone's cocycle ray-product and ρ against the BPS root via the certified
    iso (flavour-aware: ray-multiply for all; strict ρ for unflavoured, ρ up to
    the μ-unit section torsor `rho_ok_mod_flavour` for flavoured cones);
  * **atlas rotation certificate** — `all_ok` with the expected period and
    monodromy = ρ² (the axiomatics-vs-cluster statement), light/medium tier;
  * **RG node-drop identity iso** — `DirectionalSingleNodeRG` (drop γ₁) certifies
    the identity `KAlgebraIso` to the BPS chart.

The light tier runs by default (~1 min); `--full` adds the rank-5/6 octagon /
u1octagon (slow atlas certificate) and the rank-6/7/8 nonagon / decagon /
u1decagon (cone test only — the rotation certificate is cost-prohibitive there).

Run:  `python3 run_tests.py`
      `python3 run_tests.py` --full
"""
import sys
import os

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "implementations"))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

from bps_atlas import BPSAtlas
from kalgebra import Element
from directional_subquiver_rg import (
    DirectionalSingleNodeRG, certify_directional_vs_bps)
from atlas_catalogue_polygons import bundle

PASS = []
FAIL = []


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


# (name, expected_period_or_None, atlas_K, atlas_win, rg_K)
LIGHT = [
    ("sqed1", 2, 6, 0, 6),
    ("pentagon", 4, 6, 0, 6),
    ("hexagon", 6, 6, 0, 6),
    ("U1hexagon", 6, 6, 0, 4),
    ("heptagon", 8, 6, 2, 4),
]
MEDIUM = [
    ("octagon", 10, 5, 2, None),
    ("u1octagon", 10, 4, 2, None),
]
# cone test only (atlas certificate cost-prohibitive at rank >= 6).
# (name, ray_subset, check_multiply) — decagon = flavoured a7: the BPS
# ground-truth ray-multiply hits the a7 F-solve pathology (>30 min), so only
# the cone build + ρ are checked there (the reachability wall).
HEAVY_CONE = [
    ("nonagon", 8, True),
    ("decagon", 6, False),
    ("u1decagon", 8, True),
]


def _window(A, win):
    return None if not win else [A.identity()] + [tuple(g) for g in A.node_charges[:win]]


def test_theory(name, period, K, win, rg_K, *, do_atlas):
    bps, cone, cone_iso, clabel = bundle(name)
    At = BPSAtlas(bps)

    # complete the atlas — the whole (finite-type) chart graph (cheap: no traces)
    comp = At.complete()
    check(f"{name}: atlas completes (closed, {comp['n_charts']} charts)",
          comp["closed"])
    if do_atlas:
        check(f"{name}: n_charts == period {period} (got {comp['n_charts']})",
              comp["n_charts"] == period)

    # cone ray-multiply / ρ test (all tiers)
    cres = At.test_cone_presentation(cone, cone_iso)
    check(f"{name}: cone ray-multiplication OK ({clabel})"
          + ("" if cres["ray_multiply_ok"] else f" - {cres['mismatches']}"),
          cres["ray_multiply_ok"])
    check(f"{name}: cone ρ OK (mod flavour torsor)", cres["rho_ok_mod_flavour"])
    if str(bps.coefficient_ring()) == "TrivialZPlusRing()":
        check(f"{name}: cone ρ OK (strict, unflavoured)", cres["rho_ok"])

    # atlas rotation certificate (light/medium)
    if do_atlas:
        cert = At.certificate(trace_K=K, labels=_window(bps, win))
        check(f"{name}: atlas period == {period} (got {cert['period']})",
              cert["period"] == period)
        check(f"{name}: monodromy = ρ²", cert["monodromy"]["is_rho2"])
        check(f"{name}: atlas certificate all_ok", cert["all_ok"])

    # RG node-drop identity iso
    nodes = [tuple(g) for g in bps.node_charges]
    if len(nodes) < 2:
        check(f"{name}: tower base (1 node — no RG drop)", len(nodes) == 1)
    else:
        R = DirectionalSingleNodeRG(
            [list(r) for r in bps.lattice.pairing], nodes,
            [tuple(s) for s in bps.spec], gamma_drop=nodes[0],
            rg_window=(rg_K or 4), arrange=True)
        labels = [bps.identity()] + nodes[:max(2, (win or 3))]
        pairs = [(labels[1], labels[2])] if len(labels) > 2 else None
        cv = certify_directional_vs_bps(R, bps, labels=labels, pairs=pairs, trace_K=rg_K)
        checks = {k: v for k, v in cv.items() if k != "iso"}
        check(f"{name}: RG node-drop identity iso"
              + ("" if all(checks.values()) else f" - {checks}"),
              all(checks.values()))


def test_cone_only(name, subset, check_mult):
    """Heavy tier: just the cone ray-multiply/ρ test on a ray subset (the atlas
    rotation certificate is cost-prohibitive at rank >= 6)."""
    bps, cone, cone_iso, clabel = bundle(name)
    At = BPSAtlas(bps)
    # complete the atlas (cheap even at the heavy tier — chart graph only)
    comp = At.complete()
    check(f"{name}: atlas completes (closed, {comp['n_charts']} charts)",
          comp["closed"])
    cd = cone.cone_data()
    gens = [cd.from_cone_label(frozenset({g}), {g: 1})
            for g in sorted(cd.mult_gens())][:subset]
    cres = At.test_cone_presentation(cone, cone_iso, gens=gens,
                                     check_multiply=check_mult)
    n = len(cd.mult_gens())
    if check_mult:
        check(f"{name}: cone ray-multiplication OK ({clabel}, {subset}/{n} rays)"
              + ("" if cres["ray_multiply_ok"] else f" - {cres['mismatches']}"),
              cres["ray_multiply_ok"])
    check(f"{name}: cone ρ OK (mod flavour torsor, {subset}/{n} rays)",
          cres["rho_ok_mod_flavour"])


def test_u1e7_flow_composition():
    """U1E7 (u(1)-gauged E₇) has no native gauged-E₇ BPS quiver, but its RG flow
    `U1A1E7RGKAlgebra` lands on `A1A2kKAlg(3) = nonagon` (which HAS a BPS chart) ⊗
    QT(Z²) — so flow composition makes it BPS-backed.  The
    generalized `verify_cone_presentation` validates the U1E7 cone against that
    flow root (identity iso), confirming the cone test is first-class on
    flow-composed gauge theories, not just `BPSKAlgebra` seeds."""
    from kalgebra_iso import KAlgebraIso
    from laurent_poly import LaurentPoly
    from bps_atlas import verify_cone_presentation
    from u1e7_cone_kalgebra import U1E7ConeKAlgebra
    from u1a1e7_rgkalgebra import U1A1E7RGKAlgebra
    one = LaurentPoly.one()
    cone = U1E7ConeKAlgebra(use_frozen=True)
    root = U1A1E7RGKAlgebra()                      # IR = nonagon ⊗ QT(Z²): BPS-backed
    ident = KAlgebraIso(cone, root,
                        lambda l: Element({l: one}),
                        lambda l: Element({l: one}), name="u1e7[cone->rg]")
    cd = cone.cone_data()
    gens = [cd.from_cone_label(frozenset({g}), {g: 1})
            for g in sorted(cd.mult_gens())][:4]
    res = verify_cone_presentation(root, cone, ident, gens=gens)
    check("U1E7: cone ray-multiply vs BPS-backed RG flow (flow composition)"
          + ("" if res["ray_multiply_ok"] else f" - {res['mismatches']}"),
          res["ray_multiply_ok"])
    check("U1E7: cone ρ vs BPS-backed RG flow", res["rho_ok_mod_flavour"])


def main():
    full = "--full" in sys.argv
    print("=== BPSAtlas polygon/SQED catalogue regression"
          + (" (--full: all 10)" if full else " (light tier; --full adds the heavy 5)")
          + " ===")
    for name, period, K, win, rg_K in LIGHT:
        test_theory(name, period, K, win, rg_K, do_atlas=True)
    if full:
        for name, period, K, win, rg_K in MEDIUM:
            test_theory(name, period, K, win, rg_K, do_atlas=True)
        for name, subset, check_mult in HEAVY_CONE:
            test_cone_only(name, subset, check_mult)
        test_u1e7_flow_composition()      # gauge theory BPS-backed via flow composition
    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print(f"All {len(PASS)} BPSAtlas polygon-catalogue checks passed.")


if __name__ == "__main__":
    main()
