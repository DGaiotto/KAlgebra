"""`BPSAtlas` for the U(1)-gauged AD theories (the design record Stage-4 catalogue J).

A gauged theory has the **same BPS quiver as the ungauged one, with a larger Γ**: gauging a flavour U(1) promotes it from flavour to gauge,
enlarging the charge lattice but leaving the BPS *quiver* unchanged.  So each
U(1)-gauged AD theory shares the ungauged AD theory's BPS chart, and
`BPSAtlas(chart).mutation_complete()` closes the same finite folded cluster graph
(`src/bps/gauged_ad_atlas.py`):

  * u1a1d4 = u(1)-gauged [A₁,D₄] → the a1d4 quiver → 10 charts
  * u1a1d6 = u(1)-gauged [A₁,D₆] → the a1d6 quiver → 80 charts
  * u1a1d8 = u(1)-gauged [A₁,D₈] → the a1d8 quiver → 810 charts
  * u1e7   = u(1)-gauged E₇      → the [A₁,E₇] quiver → 416 charts

So "build a BPSAtlas for the gauged-AD theories" is the ungauged AD quiver's
atlas with the gauge Γ — there is no separate gauged-theory quiver to complete.
(The flow-composition cone test in `atlas_catalogue_gauged_ad.py` is the
*matter-content* cross-check; this is the *chart-graph* side.)

Run:  `python3 run_tests.py` [--full]
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "implementations"))

from bps_atlas import BPSAtlas
from gauged_ad_atlas import gauged_ad_names, gauged_ad_entry, gauged_ad_charts

PASS = []
FAIL = []


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


# u1a1d8 (810 charts, ~27 s) and u1e7 (416 charts, ~7 s) under --full.
_LIGHT = ["u1a1d4", "u1a1d6"]
_HEAVY = ["u1e7", "u1a1d8"]


def test_family_builds():
    charts = gauged_ad_charts()
    check("gauged-AD family builds (u1a1d4/6/8, u1e7)",
          list(charts) == gauged_ad_names() and len(charts) == 4)


def test_shares_ungauged_quiver_and_completes(name):
    """The gauged theory's BPS chart = the ungauged AD quiver (larger Γ), and
    mutation_complete closes to the ungauged AD folded-chart count."""
    e = gauged_ad_entry(name)
    mc = BPSAtlas(e.chart).mutation_complete(max_charts=5000)
    check(f"{name} = {e.description}: shares the {e.source} quiver (rank {e.rank}), "
          f"mutation_complete -> {mc['n_charts']} charts (expect {e.folded_charts})",
          mc["closed"] and mc["classified"] and mc["n_charts"] == e.folded_charts)


def main():
    full = "--full" in sys.argv
    print("=== BPSAtlas U(1)-gauged AD (same quiver, larger Γ)"
          + (" (--full)" if full else " (u1a1d4/u1a1d6; --full adds u1e7/u1a1d8)") + " ===")
    test_family_builds()
    for nm in _LIGHT + (_HEAVY if full else []):
        test_shares_ungauged_quiver_and_completes(nm)
    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print(f"All {len(PASS)} gauged-AD BPSAtlas checks passed.")


if __name__ == "__main__":
    main()
