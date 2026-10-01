"""Bulk certification of directional node-drop RGKAlgebras over the
BPS-quiver dictionary — the "countless examples supporting the RG
axiomatics" harness.

For every dictionary entry and every single-node drop, build the
directional flow (`DirectionalSingleNodeRG`, unguarded
`require_stuff_first=False` — so every run is also a probe of the
open version-(c) validity question) and run:

  * the **axiom battery** (`directional_subquiver_rg.verify_axioms`):
    bar involution, ρ-automorphism, ρ∘ρ⁻¹, RG-unital/-multiplicative/
    -bar-invariant, tRG intertwining; with `--trace-K` also ρ²-twisted
    trace cyclicity and orthonormality (the Schur-side axioms);
  * the **identity-on-labels iso battery** against the dictionary's own
    UV `BPSKAlgebra` (round-trip, multiplicativity, ρ-equivariance).

Any False is a counterexample candidate and is printed in full.

Usage (from the repo root):

    PYTHONPATH=. python scripts/certify_directional_on_dictionary.py \
        the source repository's archive \
        [--limit 20] [--per-case-seconds 90] [--trace-K 3] [--drops 2]

Tally buckets: PASS / FAIL (some check False — printed) / SKIP (flow
not in scope: empty stuff, empty IR, non-unimodular node basis) /
ERROR (exception — printed) / TIMEOUT (per-case alarm).
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

sys.path.insert(0, ".")

from kalgebra import Element
from laurent_poly import LaurentPoly
from dictionary_loader import iter_bpskalgebras_from_dictionary
from directional_subquiver_rg import (
    DirectionalSingleNodeRG,
    certify_directional_vs_bps,
    verify_axioms,
)

_ONE = LaurentPoly.one()


class _CaseTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise _CaseTimeout()


def run_case(B, j: int, *, trace_K, rg_window: int,
             engine: str = "compat") -> tuple[str, str]:
    """Certify entry `B` with node `j` dropped.  Returns (bucket, detail).

    `engine="pure"`: the fully-derived `DirectionalSingleNodeRG` —
    every check (incl. tRG-route ρ) is computed from IR + RG data; the
    counterexample-hunting mode, but the flavoured-ρ solver follow-up
    (the design record notes 2026-06-10) makes ρ-dependent checks slow/failing on
    flavoured entries for now.  `engine="compat"` (default): the
    replaced `rg_flow.SingleNodeRG` — F-oracle RG + the UV's σ as ρ
    (flow data, contract-sanctioned), so the axiom battery runs fast
    and is still a genuine test of the flow relations; the iso battery
    then skips ρ-equivariance (vacuous under σ-delegation)."""
    nodes = [tuple(g) for g in B.node_charges]
    try:
        if engine == "compat":
            from rg_flow import SingleNodeRG
            D = SingleNodeRG(B, j)
        else:
            D = DirectionalSingleNodeRG(
                [list(r) for r in B.lattice.pairing],
                nodes,
                [tuple(g) for g in B.spec],
                gamma_drop=j,
                require_stuff_first=False,
                rg_window=rg_window,
            )
    except ValueError as e:
        return "SKIP", f"build: {e}"
    except Exception as e:
        return "ERROR", f"build: {type(e).__name__}: {e}"

    labels = [tuple(B.identity())] + nodes
    # pairs: node pairs with nonzero pairing (the interacting ones), capped.
    def br(a, b):
        n = len(B.lattice.pairing)
        return sum(a[p] * B.lattice.pairing[p][q] * b[q]
                   for p in range(n) for q in range(n))
    pairs = [(a, b) for a in nodes for b in nodes
             if a != b and br(a, b) != 0][:6]
    if not pairs:
        pairs = [(a, b) for a in nodes for b in nodes if a != b][:4]
    pairs += [(nodes[0], nodes[0])]

    checks = verify_axioms(D, nodes, pairs, trace_K=trace_K,
                           deep_rho=(engine == "pure"))
    iso_rep = certify_directional_vs_bps(
        D, B, labels=labels, pairs=pairs,
        check_rho=(engine == "pure"), trace_K=None)
    checks.update({f"iso_{k}": v for k, v in iso_rep.items() if k != "iso"})
    bad = {k: v for k, v in checks.items() if v is not True}
    if bad:
        return "FAIL", str(bad)
    return "PASS", f"{len(pairs)} pairs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--limit", type=int, default=None,
                    help="max dictionary entries per file")
    ap.add_argument("--drops", type=int, default=None,
                    help="max node-drops per entry (default: all nodes)")
    ap.add_argument("--per-case-seconds", type=int, default=90)
    ap.add_argument("--trace-K", type=int, default=None,
                    help="also run the Schur-side axioms at this K")
    ap.add_argument("--rg-window", type=int, default=6)
    ap.add_argument("--engine", choices=("compat", "pure"),
                    default="compat")
    args = ap.parse_args()

    signal.signal(signal.SIGALRM, _alarm)
    tally: dict[str, int] = {}
    t_start = time.time()
    for path in args.paths:
        for idx, (name, B) in enumerate(
            iter_bpskalgebras_from_dictionary(path, limit=args.limit)
        ):
            n_nodes = len(B.node_charges)
            drops = range(n_nodes) if args.drops is None \
                else range(min(args.drops, n_nodes))
            for j in drops:
                case = f"{name} [drop {j}]"
                signal.alarm(args.per_case_seconds)
                t0 = time.time()
                try:
                    bucket, detail = run_case(
                        B, j, trace_K=args.trace_K,
                        rg_window=args.rg_window, engine=args.engine)
                except _CaseTimeout:
                    bucket, detail = "TIMEOUT", \
                        f"> {args.per_case_seconds}s"
                except Exception as e:
                    bucket, detail = "ERROR", f"{type(e).__name__}: {e}"
                finally:
                    signal.alarm(0)
                tally[bucket] = tally.get(bucket, 0) + 1
                mark = "OK " if bucket == "PASS" else bucket
                if bucket in ("FAIL", "ERROR"):
                    print(f"{mark}  {case}: {detail}", flush=True)
                else:
                    print(f"{mark}  {case} ({time.time()-t0:.1f}s)"
                          + (f" [{detail}]" if bucket == "SKIP" else ""),
                          flush=True)
    print(f"\n== tally ({time.time()-t_start:.0f}s): "
          + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())),
          flush=True)
    return 1 if tally.get("FAIL", 0) or tally.get("ERROR", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
