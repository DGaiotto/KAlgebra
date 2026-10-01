"""`BPSAtlas` A1D-type cone test regression.

The [A₁,Dₙ] Argyres–Douglas theories are flavoured, but a `BPSKAlgebra` chart
sees only the *abelian* flavour `ker(B)` — it does **not see the flavour
promotion**.  So to validate the flavoured cone against the
BPS chart you must first **flavour-rebase the cone onto that abelian view** (the
Cartan restriction), *then* establish the cone↔BPS iso and run the
ray-multiply/ρ cone test:

  * **D-odd** a1d3/5/7 are **pure SU(2)** — `Su2ToU1Rebase` (`su2_to_u1_hom`),
    abelian Cartan `U(1) = ker B` (`AbelianZPlusRing(rank=1)`).
  * **D-even** a1d4/6/8 are **SU(2)×U(1)** — `CoeffRebase` by an
    `SU(2)×U(1)→U(1)²` hom, abelian Cartan `U(1)² = ker B`
    (`AbelianZPlusRing(rank=2)`), with an auto-detected rank-2 section change.

`base_change(...)` is **trace-only** (the A1D cones carry flavour in the
`RLaurent` cross-product coefficients), so the rebases (`scripts/atlas_catalogue_a1d.py`)
push the **multiply AND trace** coeffs through the hom while keeping labels.  The
direct `_bps_oracle(short)` chart is already in the fundamental (z) normalization
`su2_to_u1_hom` produces, so no μ=z² re-refinement is needed.  Each chart's atlas
is also completed (`BPSAtlas(bps).complete()`) — the finite-type chart graph closes.

a1d3 (su2) + a1d4 (su2u1) are fast and asserted by default.  The feasibility
frontier (measured 2026-06-28): a1d5/a1d6 run under `--full` on a ray subset
(a1d6 = 864 s for 4 rays, strict ρ — just under the 900 s budget); **a1d7 is the
wall** (rank 7 exceeded 900 s even at 3 rays), a1d8 beyond it.  The D-type fork
makes the rank-≥7 BPS F-solve dense (like E-type) — unlike the linear-A polygons
that reached the dodecagon — so a1d7/a1d8 are recorded as the frontier, not run.

Run:  `python3 run_tests.py` [--full]
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "implementations"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from atlas_catalogue_a1d import certify_a1d

PASS = []
FAIL = []


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


# (short, ray_subset, flavour)  — D-odd = pure SU(2); D-even = SU(2)×U(1).
# The ungauged A1Dₙ feasibility frontier (measured 2026-06-28): a1d5/a1d6 are
# feasible (a1d6 = 864 s for 4 rays / 16 products, strict ρ — just under the
# 900 s budget), **a1d7 is the wall** (rank 7 timed out >900 s even at 3 rays),
# a1d8 beyond it.  The D-type fork makes the rank-≥7 BPS F-solve dense (like
# E-type), unlike the cheap linear-A polygons that reached the dodecagon — the
# genuine compute wall (the corrected, non-environmental answer to "why is A1Dₙ
# harder").  So `--full` asserts only the feasible a1d5/a1d6 (a1d6 small ray
# subset to stay ~minutes); a1d7/a1d8 are recorded as the frontier, not run.
LIGHT = [("a1d3", None, "su2"), ("a1d4", 4, "su2u1")]
FULL = [("a1d5", 3, "su2"), ("a1d6", 2, "su2u1")]


def test_a1d(short, nray, flavour):
    rec = certify_a1d(short, nray, flavour)
    expect_rank = 1 if flavour == "su2" else 2
    check(f"{short}: flavour rebase + cone↔bps iso builds "
          f"(bps={rec['bps_ring']}, order={rec.get('flavour_order')})",
          rec["bps_ring"] == f"AbelianZPlusRing(rank={expect_rank})")
    cp = rec["complete"]
    check(f"{short}: atlas completes (closed, {cp['n_charts']} charts)", cp["closed"])
    check(f"{short}: cone ray-multiplication OK ({rec['rays']}/{rec['total']} rays)"
          + ("" if rec["ray_multiply_ok"] else f" - {rec['mismatches']}"),
          rec["ray_multiply_ok"])
    check(f"{short}: cone ρ OK (mod flavour torsor)", rec["rho_ok_mod_flavour"])


def main():
    full = "--full" in sys.argv
    plan = LIGHT + (FULL if full else [])
    print("=== BPSAtlas A1D-type cone test regression"
          + (" (--full)" if full else " (a1d3 su2 + a1d4 su2u1; --full adds a1d5/a1d6 — a1d7 is the wall)")
          + " ===")
    for short, nray, flavour in plan:
        test_a1d(short, nray, flavour)
    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print(f"All {len(PASS)} A1D-type cone-test checks passed.")


if __name__ == "__main__":
    main()
