"""Compute I_{id, id}(q) fingerprints for sharpened-dict entries.

Per the design record (fingerprint registry) — and per the author's directive to use
``I_{id, id}`` as the canonical chart-invariant fingerprint:

  - ``I_{id, id}(q)`` is intrinsic to the algebra (not the chart).
  - Distinct fingerprints ⇒ definitely non-equivalent algebras.
  - Equal fingerprints ⇒ candidates for equivalence (need deeper test).

Flavour handling.  ``I_{id, id}`` is valued in ``R[[q]]`` where
``R = Z[μ_1^±, …, μ_r^±]`` carries flavour fugacities indexed by a
chosen Z-basis of Γ_f.  A different basis acts as ``GL(r, Z)`` on
the exponent vectors — so ``repr(iid)`` is **not** an algebra
invariant for flavoured entries.

This script implements the cheap (μ→1) layer of a planned
two-layer fingerprint:

  - Layer (a): specialise μ_i → 1.  This collapses each q-coefficient
    (an element of R) to an integer, giving a pure q-series.
    Manifestly invariant under any GL(r,Z) basis change of Γ_f.
    Distinct (a)-fingerprints ⇒ definitely non-equivalent algebras
    (free non-equivalence certificate, even with flavour).
    Equal (a)-fingerprints in flavoured entries are candidates for
    equivalence, to be refined by a follow-up GL(r,Z)-canonical form
    on the multivariate polynomial (layer (b), not yet implemented).

Run from repo root:

    PYTHONPATH=. python dictionaries/build_fingerprint.py
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from collections import defaultdict
from typing import Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESTRUCT = os.path.dirname(_HERE)
_REPO = _RESTRUCT  # post-flatten: _RESTRUCT is the repo root
for p in (_REPO, _RESTRUCT):
    if p not in sys.path:
        sys.path.insert(0, p)

from bps_kalgebra import BPSKAlgebra


class TimeoutError_(Exception):
    pass


def _alarm(seconds: int) -> None:
    def h(*_):
        raise TimeoutError_("timeout")
    signal.signal(signal.SIGALRM, h)
    signal.alarm(seconds)


def _disarm() -> None:
    signal.alarm(0)


def standard_nodes(n: int) -> list[tuple[int, ...]]:
    return [tuple(1 if i == k else 0 for k in range(n)) for i in range(n)]


def _mu_to_one_qseries(iid) -> str:
    """Collapse ``I_{id,id}`` to its μ→1 specialisation as a string.

    The result is a sorted-by-q list of ``(q_exp, int)`` pairs; the
    integer at q^k is ``Σ_v c_v`` over all flavour-exponent monomials
    ``μ^v`` appearing at that q-order.  GL(rk Γ_f, Z)-invariant.

    Works uniformly for unflavoured (TrivialZPlusRing → single
    ``()``-keyed term) and flavoured (AbelianZPlusRing) cases.
    """
    pairs: list[tuple[int, int]] = []
    for k, rel in iid.coeffs.items():
        s = sum(rel.terms.values())
        if s != 0:
            pairs.append((int(k), int(s)))
    pairs.sort()
    return repr(pairs)


def fingerprint_for_entry(
    entry: dict, K: int = 10, timeout_s: int = 60,
) -> tuple[str, str]:
    """Return ``(fingerprint_str, status)`` where:

    - ``fingerprint_str`` is the μ→1 specialisation of
      ``I_{id, id}(q) mod q^K`` as a hashable string (a sorted list of
      ``(q_exp, int_coeff)`` pairs), or ``""`` if computation failed.
    - ``status`` is one of ``"ok"``, ``"timeout"``, or
      ``"err: <message>"``.
    """
    n = len(entry["pi"]) if "pi" in entry else len(entry["exchange"])
    nodes = standard_nodes(n)
    spec = [tuple(b) for b in entry["spec"]]
    try:
        _alarm(timeout_s)
        A = BPSKAlgebra(
            pairing=entry["exchange"], node_charges=nodes, spec=spec,
        )
        iid = A.inner_product(A.identity(), A.identity(), K=K)
        _disarm()
        return (_mu_to_one_qseries(iid), "ok")
    except TimeoutError_:
        _disarm()
        return ("", "timeout")
    except Exception as exc:
        _disarm()
        return ("", f"err: {type(exc).__name__}: {str(exc)[:80]}")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-min", type=int, default=2)
    p.add_argument("--n-max", type=int, default=8)
    p.add_argument("--K", type=int, default=10,
                   help="q-truncation order for I_{id,id}")
    p.add_argument("--timeout", type=int, default=60,
                   help="per-entry timeout in seconds")
    p.add_argument("--out", default=os.path.join(_HERE, "fingerprint.json"))
    args = p.parse_args(argv)

    by_fingerprint: dict[str, list[dict]] = defaultdict(list)
    timeouts: list[dict] = []
    errors: list[dict] = []

    total = 0
    for n in range(args.n_min, args.n_max + 1):
        path = os.path.join(_HERE, f"n_{n:03d}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            entries = json.load(f)
        n_ok = 0
        t_n = time.time()
        for e in entries:
            total += 1
            fp, status = fingerprint_for_entry(e, K=args.K, timeout_s=args.timeout)
            if status == "ok":
                by_fingerprint[fp].append({"name": e["name"], "n": n})
                n_ok += 1
            elif status == "timeout":
                timeouts.append({"name": e["name"], "n": n})
            else:
                errors.append({"name": e["name"], "n": n, "err": status})
        print(f"n={n}: {n_ok}/{len(entries)} ok in {time.time()-t_n:.1f}s")

    # Summary.
    fingerprints = sorted(by_fingerprint.keys())
    n_total_ok = sum(len(v) for v in by_fingerprint.values())
    n_distinct = len(fingerprints)
    n_collisions = sum(1 for v in by_fingerprint.values() if len(v) > 1)

    print()
    print(f"Total entries: {total}")
    print(f"Computed (ok): {n_total_ok}")
    print(f"Timeouts: {len(timeouts)}")
    print(f"Errors: {len(errors)}")
    print()
    print(f"Distinct fingerprints: {n_distinct}")
    print(f"Fingerprints with collisions (>= 2 entries): {n_collisions}")
    print()
    largest_buckets = sorted(
        by_fingerprint.values(), key=lambda v: len(v), reverse=True
    )[:5]
    print("Top-5 largest buckets (potential equivalence classes):")
    for bucket in largest_buckets:
        print(f"  size {len(bucket)}: {[e['name'][:40] for e in bucket[:5]]}"
              f"{'...' if len(bucket) > 5 else ''}")

    out = {
        "K": args.K,
        "by_fingerprint": dict(by_fingerprint),
        "timeouts": timeouts,
        "errors": errors,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Wrote fingerprint dictionary to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
