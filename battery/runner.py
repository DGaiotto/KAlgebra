"""runner.py — the runner of the battery of checks for the paper's claims.

One claim id of `claims.json` maps to one adapter (a callable returning the
checks it performed); the runner validates the id against the registry, checks
that the claim can run in the requested ENVIRONMENT (`web` = the tracked
repository is the whole input; `local` = needs the local machine's gitignored
data, named under the claim's `requires`), refuses a run whose inputs are
absent (a skip goes to the log, never to `results/`), and writes ONE dated
result record per run:

    results/<id-slug>/<environment>/<UTC date-time>.json

with the fields of README section 5 — id, date, commit, environment, machine,
inputs actually read, population, counts {pass, fail}, controls, elapsed_s,
notes, and the list of individual checks (name, ok, note).  Latest record per
(claim, environment) wins when the renderer merges records into the appendix.

Run from the repo root:

    PYTHONPATH=. python3 battery/runner.py --list
    PYTHONPATH=. python3 battery/runner.py --id def:penta/implementation
    PYTHONPATH=. python3 battery/runner.py --section sec:pentagon
    PYTHONPATH=. python3 battery/runner.py --all-implemented --no-write

Pure Python, no dependencies.  Adapters live in sibling modules (today
`checks_kq.py`) and register themselves in an `ADAPTERS` dict keyed by claim id.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))     # the release root: battery/ sits at its top
REG = os.path.join(HERE, "claims.json")
RESULTS = os.path.join(HERE, "results")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ENVIRONMENTS = ("web", "local")


def slug(claim_id: str) -> str:
    return claim_id.replace(":", ".").replace("/", "__")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001 — a record without a commit is still a record
        return "unknown"


def launch_commit() -> str:
    """The commit whose code this process runs: HEAD when the adapters were imported, with `+dirty` when a
    tracked file differed from it then (new, untracked records do not count).  Read once, at launch — a row
    can run for hours, and HEAD read at the end would name commits made meanwhile, not the code that ran."""
    if "commit" not in _LAUNCH:
        c = git_commit()
        try:
            dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
        except Exception:  # noqa: BLE001
            dirty = ""
        _LAUNCH["commit"] = c + ("+dirty" if dirty else "")
    return _LAUNCH["commit"]


_LAUNCH: dict = {}


def code_changed_since_launch() -> list:
    """Tracked .py files that differ between the launch commit and the working tree now.  The runner pins the code
    it imported at launch, but a module first imported later (by the runner or by a forked worker) is read from the
    working tree, so a record whose run saw code change says so, by name, rather than claim the launch commit alone."""
    base = _LAUNCH.get("commit", "").split("+")[0]
    if not base or base == "unknown":
        return []
    try:
        out = subprocess.check_output(["git", "diff", "--name-only", base, "--", "*.py"], cwd=ROOT, text=True)
    except Exception:  # noqa: BLE001
        return []
    return [line for line in out.splitlines() if line.strip()]


def load_registry() -> dict:
    return json.load(open(REG, encoding="utf-8"))


def load_adapters() -> dict:
    adapters: dict = {}
    import checks_kq  # noqa: E402  (sibling modules; each exposes ADAPTERS)
    import checks_flavoured  # noqa: E402
    import checks_uq_a1d3  # noqa: E402
    import checks_coulomb  # noqa: E402
    import checks_rg  # noqa: E402
    import checks_extra_examples  # noqa: E402
    import checks_skein  # noqa: E402
    import checks_finite  # noqa: E402
    adapters.update(checks_kq.ADAPTERS)
    adapters.update(checks_flavoured.ADAPTERS)
    adapters.update(checks_uq_a1d3.ADAPTERS)
    adapters.update(checks_coulomb.ADAPTERS)
    adapters.update(checks_rg.ADAPTERS)
    adapters.update(checks_extra_examples.ADAPTERS)
    adapters.update(checks_skein.ADAPTERS)
    adapters.update(checks_finite.ADAPTERS)
    return adapters


def extensive_capable() -> set:
    """Claim ids whose adapter grows its windows at the extensive depth; a module lists them in EXTENSIVE."""
    import checks_coulomb  # noqa: E402
    import checks_extra_examples  # noqa: E402
    import checks_flavoured  # noqa: E402
    import checks_kq  # noqa: E402
    import checks_rg  # noqa: E402
    import checks_skein  # noqa: E402
    import checks_finite  # noqa: E402
    import checks_uq_a1d3  # noqa: E402
    out: set = set()
    for mod in (checks_kq, checks_flavoured, checks_uq_a1d3, checks_coulomb, checks_rg, checks_extra_examples, checks_skein,
                checks_finite):
        out |= set(getattr(mod, "EXTENSIVE", ()))
    return out


def inputs_present(claim: dict, environment: str) -> tuple[bool, list[str]]:
    """The `requires[environment]` entries are prose naming paths; a path-like
    token (containing '/') must resolve to at least one existing file or
    directory under the repo root (globs allowed).  Returns (ok, missing)."""
    missing: list[str] = []
    for entry in claim.get("requires", {}).get(environment, []):
        for tok in entry.replace("(", " ").replace(")", " ").replace(",", " ").replace(";", " ").split():
            if "/" in tok and not tok.startswith("http"):
                pat = os.path.join(ROOT, tok.rstrip(":"))
                if not glob.glob(pat) and not glob.glob(pat + "*"):
                    missing.append(tok)
    return (not missing, missing)


def make_record(claim: dict, environment: str, machine: str, result: dict, elapsed: float, depth: str = "fast") -> dict:
    checks = result.get("checks", [])
    n_pass = sum(1 for c in checks if c["ok"])
    n_fail = len(checks) - n_pass
    changed = code_changed_since_launch()
    return {
        "id": claim["id"],
        "labels": claim["labels"],
        "date": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": launch_commit(),
        **({"code_changed_during_run": changed[:20]} if changed else {}),
        "environment": environment,
        "depth": depth,
        "machine": machine,
        "inputs": result.get("inputs", []),
        "population": result.get("population", {}),
        "counts": {"pass": n_pass, "fail": n_fail},
        "controls": result.get("controls", {}),
        "elapsed_s": round(elapsed, 3),
        "notes": result.get("notes", ""),
        "checks": checks,
    }


def write_record(rec: dict) -> str:
    d = os.path.join(RESULTS, slug(rec["id"]), rec["environment"] + ("-extensive" if rec.get("depth") == "extensive" else ""))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, rec["date"].replace(":", "") + ".json")
    json.dump(rec, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open(path, "a").write("\n")
    return path


def latest_records() -> dict:
    """{claim id: {environment[ (extensive)]: record}} — the latest record per (claim, environment, depth); an
    extensive run is kept beside the fast one, never superseding it or superseded by it."""
    out: dict = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "*", "*", "*.json"))):
        rec = json.load(open(path, encoding="utf-8"))
        key = rec["environment"] + (" (extensive)" if rec.get("depth") == "extensive" else "")
        out.setdefault(rec["id"], {})[key] = rec
    return out


def run_one(claim: dict, adapters: dict, environment: str, machine: str, write: bool, depth: str = "fast") -> dict | None:
    cid = claim["id"]
    if cid not in adapters:
        print(f"[skip] {cid}: no adapter yet (standing {claim['standing']})")
        return None
    if depth == "extensive" and cid not in extensive_capable():
        print(f"[skip] {cid}: its adapter has no extensive windows (a fast run relabelled would mislead)")
        return None
    if environment not in claim["environment"]:
        print(f"[skip] {cid}: not runnable in environment {environment!r} (registry: {claim['environment']})")
        return None
    ok, missing = inputs_present(claim, environment)
    if not ok:
        print(f"[skip] {cid}: input absent for environment {environment!r}: {missing}")
        return None
    t0 = time.time()
    os.environ["BATTERY_DEPTH"] = depth
    try:
        result = adapters[cid](environment)
    except ModuleNotFoundError as exc:
        # a probe of the source repository this release does not carry: refused, never failed, never written
        print(f"[skip] {cid}: needs a module this release does not carry: {exc.name}")
        return None
    rec = make_record(claim, environment, machine, result, time.time() - t0, depth)
    if not rec["checks"]:
        # nothing was compared (e.g. every theory's worker died): like a skip, never written, so it can neither count
        # as a pass nor supersede an earlier record
        print(f"[skip] {cid}: the run compared nothing (population: {str(rec['population'])[:300]}); no record written")
        return None
    fails = [c for c in rec["checks"] if not c["ok"]]
    print(f"[run ] {cid}{' (extensive)' if depth == 'extensive' else ''}: {rec['counts']['pass']} pass, {rec['counts']['fail']} fail, {rec['elapsed_s']} s"
          + (f"; controls {rec['controls']}" if rec["controls"] else ""))
    for c in fails:
        print(f"        FAIL {c['name']}: {c.get('note', '')}")
    if write:
        path = write_record(rec)
        print(f"        -> {os.path.relpath(path, ROOT)}")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--environment", choices=ENVIRONMENTS, default="web")
    ap.add_argument("--machine", default=os.environ.get("BATTERY_MACHINE") or platform.node() or "unknown")
    ap.add_argument("--id", action="append", default=[], help="claim id (repeatable)")
    ap.add_argument("--section", default=None, help="run every implemented claim whose first label is at or after this draft label and before the next section head")
    ap.add_argument("--tier", choices=("fast", "extensive"), default=None)
    ap.add_argument("--depth", choices=("fast", "extensive"), default="fast",
                    help="the window depth the adapters run at; an extensive record is kept beside the fast one")
    ap.add_argument("--all-implemented", action="store_true")
    ap.add_argument("--list", action="store_true", help="list claims with an adapter")
    ap.add_argument("--no-write", action="store_true", help="run without writing a result record")
    args = ap.parse_args(argv)

    reg = load_registry()
    claims = {c["id"]: c for c in reg["claims"]}
    os.environ["BATTERY_LAUNCH_COMMIT"] = launch_commit()   # before the adapters are imported: the code that runs is the code at this commit (also keys the extensive checkpoints)
    adapters = load_adapters()
    unknown = sorted(set(adapters) - set(claims))
    assert not unknown, f"adapters for ids absent from the registry: {unknown}"
    if args.list:
        for cid in claims:
            if cid in adapters:
                c = claims[cid]
                print(f"{cid:34s} {c['sweep']:10s} {'+'.join(c['environment']):10s} {c['standing']}")
        return 0
    selected: list[str] = []
    if args.id:
        for cid in args.id:
            if cid not in claims:
                print(f"unknown claim id {cid!r}"); return 2
            selected.append(cid)
    if args.section:
        labels = reg["draft"]["labels"]
        heads = [l for l in labels if l.startswith("sec:") or l.startswith("app:")]
        if args.section not in labels:
            print(f"unknown draft label {args.section!r}"); return 2
        start = labels.index(args.section)
        later_heads = [labels.index(h) for h in heads if labels.index(h) > start]
        stop = min(later_heads) if later_heads else len(labels)
        window = set(labels[start:stop])
        selected += [cid for cid, c in claims.items() if cid in adapters and any(l in window for l in c["labels"])]
    if args.all_implemented:
        selected += [cid for cid in claims if cid in adapters]
    if args.tier:
        selected = [cid for cid in selected if claims[cid]["sweep"] == args.tier]
    seen: set[str] = set()
    selected = [cid for cid in selected if not (cid in seen or seen.add(cid))]
    if not selected:
        print("nothing selected (use --id, --section, --all-implemented or --list)"); return 1
    total_fail = 0
    for cid in selected:
        rec = run_one(claims[cid], adapters, args.environment, args.machine, write=not args.no_write, depth=args.depth)
        if rec:
            total_fail += rec["counts"]["fail"]
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
