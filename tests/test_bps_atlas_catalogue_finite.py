"""`BPSAtlas` catalogue over the finite-type / Argyres–Douglas zoo.

For each finite-type theory we build a `BPSAtlas` on its BPS-quiver chart (the
embedded `*_BPS_*` literals, served by the source repository's BPS oracle) and run
the one-call rotation **certificate** (`BPSAtlas.certificate`).  The certificate
walks the rotation chamber chain and checks, per theory:

  * ``period``                   — necklace steps for the chart to return to root;
  * ``edges_ok``                 — every mutation edge is a full ``KAlgebraIso``
    (unit / round-trip / multiplicative / ρ- / trace-equivariant);
  * ``multiply_chart_invariant`` — ``L_a·L_b`` is the same in every chamber;
  * ``trace_chart_invariant``    — the Schur index ``I_a(𝖖)`` is identical in
    every chamber (the headline wall-crossing-invariance statement);
  * ``monodromy.is_rho2``        — the full rotation loop composes to ``ρ²``.

This is the demonstrative dataset for **one abstract algebra, many
certified presentations** and **wall-crossing as algebraic identities
on ``A_𝖖``**.  Two findings hold on *every* probed theory:

  1. the rotation monodromy is **always ``ρ²``** — the same ``ρ²`` that twists
     the trace-cyclicity axiom;
  2. the Schur index is **chart-invariant** (a wall-crossing invariant), once
     ``K`` is large enough to clear truncation.

**D8 truncation discipline.**  ``Tr`` / ``I_{a,b}`` / ``multiply`` / ``ρ`` are
chart-independent *by axiom*, so any ``trace_chart_invariant=False`` on a
**spec** (verified) chart is a finite-``K`` **truncation** artefact, not a bug:
the test raises ``trace_K`` and records the ``K`` at which it recovers.  A
genuine anomaly would be one that does **not** recover with ``K``.

Run the cheap tier (asserted regression, ~1 min):

    `python3 run_tests.py`

Run the **full** catalogue (all 14 theories; rank-7/8 traces are slow —
minutes each — so run it in the background):

    `python3 run_tests.py` --full
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elem_traces import _bps_oracle
from regen import REGEN_SPECS
from bps_atlas import BPSAtlas

PASS = []
FAIL = []


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}", flush=True)


# ---------------------------------------------------------------------------
# The catalogue.  `tier`: "cheap" runs by default (regression, full label
# window); "full" is the rank-≥5 tail, opt-in via --full.  `period` is the
# recorded rotation period (asserted as a regression anchor; None = record-only).
# `K` is the trace order; `win` is the certificate's label-window size (0 = the
# default full window = identity + every node charge; a positive `win` uses
# identity + the first `win` node charges — a smaller but genuine demonstration
# window, used to keep the rank-≥5 batteries tractable since the per-edge battery
# is ~win² trace-equivariance checks per chamber).
# ---------------------------------------------------------------------------
CATALOGUE = [
    # short_id   family                tier    period  K  win
    ("pentagon", "[A1,A2] (A2)",      "cheap",   4,    6, 0),
    ("heptagon", "[A1,A4]",           "cheap",   8,    6, 0),
    ("a3",       "A3 / hexagon (u1)", "cheap",   6,    6, 0),
    ("a1d3",     "[A1,D3] (su2)",     "cheap",   6,    6, 0),
    ("a1d4",     "[A1,D4] (su2u1)",   "full",    8,    4, 2),
    ("a5",       "A5 / octagon (u1)", "full",   10,    5, 2),
    ("a1d5",     "[A1,D5] (su2)",     "full",   10,    5, 2),
    ("e6",       "E6",                "full",   12,    5, 2),
    ("e7",       "E7 (u1)",           "full",   14,    4, 2),
    ("e8",       "E8",                "full",   16,    4, 2),
]
# Note: a7 (A7 / decagon) and the rank-≥6 su2u1 D-types (a1d6/a1d7/a1d8) are
# omitted — their F-solve / Schur-trace is pathologically slow at this rank
# (a7 > 30 min CPU even at the 2-label window, vs e7's ~7 min at rank 7), so
# they are not routine-runnable.  The rank-7/8 regime is already represented by


def certify_finite(short_id, *, K=6, win=0, max_K=16):
    """Build the atlas and run the certificate, applying the D8 truncation
    discipline: if the Schur index is not chart-invariant at the modest `K`
    (a truncation artefact on a verified spec chart), raise `K` until it
    recovers.  `win` selects the certificate label window (0 = full).  Returns
    `(cert, used_K, recovered_from)` where `recovered_from` is the `K` at which
    truncation first showed (or None)."""
    bps = _bps_oracle(short_id)
    At = BPSAtlas(bps)
    # complete the atlas — the whole finite-type chart graph (cheap: no traces)
    comp = At.complete()
    check(f"{short_id}: atlas completes (closed, {comp['n_charts']} charts)",
          comp["closed"])
    labels = None
    if win:
        labels = [bps.identity()] + [tuple(g) for g in bps.node_charges[:win]]
    cert = At.certificate(trace_K=K, labels=labels)
    used_K = K
    recovered_from = None
    while not cert["trace_chart_invariant"] and used_K < max_K:
        if recovered_from is None:
            recovered_from = used_K
        used_K += 2
        cert = At.certificate(trace_K=used_K, labels=labels)
    return cert, used_K, recovered_from


def run_entry(short_id, family, expected_period, K=6, win=0):
    t0 = time.time()
    cert, used_K, trunc_from = certify_finite(short_id, K=K, win=win)
    dt = time.time() - t0
    edges_ok = all(all(r.values()) for r in cert["edge_batteries"])
    flav = REGEN_SPECS[short_id][2]
    note = "" if trunc_from is None else f" (trace_inv recovered at K={used_K})"
    print(f"  · {short_id:9s} {family:20s} flav={flav:6s} "
          f"period={cert['period']:2d} edges_ok={edges_ok} "
          f"mult_inv={cert['multiply_chart_invariant']} "
          f"trace_inv={cert['trace_chart_invariant']} "
          f"rho2={cert['monodromy']['is_rho2']} "
          f"all_ok={cert['all_ok']} [{dt:.1f}s]{note}", flush=True)

    # The universal claims — asserted for every catalogued theory.
    check(f"{short_id}: every mutation edge is a full KAlgebraIso", edges_ok)
    check(f"{short_id}: multiply chart-invariant",
          cert["multiply_chart_invariant"])
    check(f"{short_id}: Schur index chart-invariant (D8 K={used_K})",
          cert["trace_chart_invariant"])
    check(f"{short_id}: rotation monodromy = ρ²",
          cert["monodromy"]["is_rho2"])
    check(f"{short_id}: certificate all_ok", cert["all_ok"])
    if expected_period is not None:
        check(f"{short_id}: rotation period == {expected_period}",
              cert["period"] == expected_period)
    return cert


def main(full=False):
    print(f"BPSAtlas finite-type catalogue ({'full' if full else 'cheap'} tier)\n",
          flush=True)
    for short_id, family, tier, period, K, win in CATALOGUE:
        if tier == "full" and not full:
            continue
        run_entry(short_id, family, period, K=K, win=win)
    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print(f"All {len(PASS)} BPSAtlas finite-type catalogue checks passed.")


if __name__ == "__main__":
    main(full="--full" in sys.argv)
