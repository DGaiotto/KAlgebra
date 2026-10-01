"""`BPSAtlas` U(1)-gauged AD cone test regression.

The U(1)-gauged Argyres–Douglas theories (u1a1d4 = gauged [A₁,D₄], u1e7 = gauged
E₇) have **no native gauged-AD BPS quiver**, but each is realized as an
`RGKAlgebra` whose IR auxiliary has a BPS chart — so "**an RG flow to something
which has a BPS chart fixes immediately a BPS chart via flow composition**".  The RG realization is then a BPS-backed `root`, and
`bps_atlas.verify_cone_presentation(root, cone, iso)` validates the cone's
ray-multiplication and ρ against it.

Unlike the ungauged A1D catalogue (`test_bps_atlas_a1d.py`, BPS-chart ground
truth + SU(2)→U(1) rebase), here the RG root carries the **full** non-abelian
flavour — identical to the cone — so the comparison is apples-to-apples in
canonical R-form and **strict ρ** holds (no rebase, no μ-torsor slack).

u1a1d4 (full 14 generators: 12 curves + X_{0,1}^±) + u1e7 (4-ray sample)
asserted by default; `--full` adds D₆ (u1a1d6, k=2; 32 generators / 1024
products) and D₈ (u1a1d8, k=3; 58 generators / 3364 products, strict ρ).  Every
u1a1d* root is the flow `U1A1DevenViaDoddRG(k)` the curve frame
`U1A1DevenConeKAlgebra(k)` is derived from (the public class since 2026-09-24),
and the iso is the closed-form label bijection
`u1a1deven_cone_dodd_section_iso(k)`.

Run:  `python3 run_tests.py` [--full]
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "implementations"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from atlas_catalogue_gauged_ad import certify

PASS = []
FAIL = []


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


# (name, ray_subset, gen_mode, strict_rho) — full-SU(2)-flavoured RG roots give
# strict ρ.  D₆ (k=2) and D₈ (k=3) certify on the FULL generator set via the
# label-bijection iso onto the flow.
LIGHT = [("u1a1d4", None, "conedata", True), ("u1e7", 4, "conedata", True)]
FULL = [("u1a1d6", None, "conedata", True), ("u1a1d8", None, "conedata", True)]


def test_gauged(name, nray, gen_mode, strict_rho):
    rec = certify(name, nray, gen_mode)
    check(f"{name}: flow-composition cone↔RG-root iso + cone test builds "
          f"(root={rec['root']}, ring={rec['root_ring']})",
          rec["ray_multiply_ok"] is not None)
    check(f"{name}: cone ray-multiplication OK ({rec['n_rays']}/{rec['total']} rays, "
          f"{rec['n_products']} products)"
          + ("" if rec["ray_multiply_ok"] else f" - {rec['mismatches']}"),
          rec["ray_multiply_ok"])
    if strict_rho:
        check(f"{name}: cone ρ OK (strict — full-flavour RG root)", rec["rho_ok"])
    else:
        check(f"{name}: cone ρ OK (mod flavour torsor)", rec["rho_ok_mod_flavour"])


def main():
    full = "--full" in sys.argv
    plan = LIGHT + (FULL if full else [])
    print("=== BPSAtlas U(1)-gauged AD cone test regression"
          + (" (--full)" if full else " (u1a1d4 + u1e7; --full adds u1a1d6 + u1a1d8)")
          + " ===")
    for name, nray, gen_mode, strict_rho in plan:
        test_gauged(name, nray, gen_mode, strict_rho)
    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print(f"All {len(PASS)} U(1)-gauged AD cone-test checks passed.")


if __name__ == "__main__":
    main()
