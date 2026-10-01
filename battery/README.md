# The battery of checks for the paper's claims

`claims.json` is the registry: one entry per claim of the paper, keyed by the
paper's labels, with its standing and the checks that bear on it.  `runner.py` runs
a claim's adapter (in the `checks_*.py` modules) and writes one dated result record
per run under `results/<claim>/<environment>/`; `render_battery.py` merges the
latest record of each claim into the companion's appendix and into `battery.md`,
its Markdown twin.  `results/` holds the records the published appendix was
rendered from.

Run from the repository root:

    PYTHONPATH=. python3 battery/runner.py --list
    PYTHONPATH=. python3 battery/runner.py --id def:penta/implementation --no-write
    PYTHONPATH=. python3 battery/runner.py --section sec:pentagon --no-write

`--environment web` (the default) needs nothing beyond this repository; a `local`
row needs data built on a local machine, named under the claim's `requires`, and
is refused (not failed) when that data is absent.  `--no-write` runs without
adding a record.

The research probes the adapters import ship in `experiments/`.
`test_ade_serving_guard.py` here holds the serving table that the row for the
table of finite-type algebras reads; as a test it audits the source repository's
layout, so it is not part of `run_tests.py`.  Should an adapter need a module this
release does not carry, the runner refuses that row, naming the module, writes no
record, and goes on with the next.
