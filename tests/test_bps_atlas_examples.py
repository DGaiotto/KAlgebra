"""Regression for the complete `BPSAtlas` example gallery (Stage-4 export).

The Argyres–Douglas zoo built from self-contained BPS-chart literals
(`bps_atlas_examples`): every example must build from its literal and **complete**
(finite chart graph → `closed=True`).  The unflavoured square-quiver members fold
onto a fundamental domain; the flavoured / frozen-flavour-node ones complete but
report `classified=False`.

Run:  `python3 run_tests.py`
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bps_atlas_examples import (
    build_all_complete, EXAMPLE_NAMES, complete_atlas, mutation_complete_atlas,
)

PASS = []
FAIL = []


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


def main():
    print("=== complete BPSAtlas example gallery (Argyres-Douglas zoo) ===")
    recs = build_all_complete()
    check(f"{len(recs)} examples (== {len(EXAMPLE_NAMES)} names)",
          len(recs) == len(EXAMPLE_NAMES) == 16)
    for r in recs:
        # every example completes — finite chart graph, small (n_charts ~ 2*rank)
        check(f"{r['name']:9s}: completes ({r['n_charts']} charts, "
              f"closed)", r["closed"] and r["n_charts"] <= 4 * r["rank"])
    # a non-trivial chart graph (not just the seed) for the rank>=2 theories
    check("pentagon has a 4-chart graph", build_rec(recs, "pentagon")["n_charts"] == 4)
    # square unflavoured member folds; sqed1 (frozen flavour node) does not
    penta = build_rec(recs, "pentagon")
    check("pentagon is square → folds to 1 class",
          penta["classified"] and penta["n_classes"] == 1)
    sqed = build_rec(recs, "sqed1")
    check("sqed1 is non-square → classified=False, n_classes=None",
          (not sqed["classified"]) and sqed["n_classes"] is None)
    # re-completing is idempotent (deterministic)
    _, rec = complete_atlas("heptagon")
    check("heptagon re-completes to 8 charts", rec["n_charts"] == 8 and rec["closed"])

    # --- mutation-complete folded atlas -------------------
    # pentagon: a single chart with two outgoing mutation self-loops.
    _, mc = mutation_complete_atlas("pentagon")
    check("pentagon mutation-completes to 1 chart, 2 self-loops",
          mc["n_charts"] == 1 and mc["self_loops"] == 2
          and len(mc["edges"]) == 2 and mc["closed"])
    # every edge of the single pentagon chart loops back to itself
    check("pentagon mutation edges are all self-loops",
          all(e["src"] == e["dst"] == 0 for e in mc["edges"]))
    # [A1,A3]: four charts, mutation-complete
    _, mc3 = mutation_complete_atlas("a3")
    check("a3 mutation-completes to 4 charts (closed)",
          mc3["n_charts"] == 4 and mc3["closed"])
    # [A1,D3]: four charts
    _, mcd3 = mutation_complete_atlas("a1d3")
    check("a1d3 mutation-completes (closed)", mcd3["closed"])

    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print(f"All {len(PASS)} BPSAtlas-example checks passed.")


def build_rec(recs, name):
    return next(r for r in recs if r["name"] == name)


if __name__ == "__main__":
    main()
