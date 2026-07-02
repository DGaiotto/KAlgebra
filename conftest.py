"""pytest guard: this repository's validation gate is ``run_tests.py``.

pytest is **not** a supported entry point, and a pytest run silently
produces a weaker-and-wrong gate: it skips ``tests/test_cones.py`` and
``tests/test_sample_cone_iso.py`` entirely (their checks live in
``main()``-style runners, not ``test_*`` functions), and it
collection-imports every test module into a single process — the BPS
suite's module-level imports then make every ``test_spine_free``
assertion fail spuriously (spine-freeness is a per-process property,
which is why ``run_tests.py`` runs the BPS suite last).  Rather than let
a green-looking partial run masquerade as the gate, refuse loudly.

Run the real gate:

    python3 run_tests.py
"""


def pytest_configure(config):
    raise RuntimeError(
        "pytest is not a supported entry point for this repository: it "
        "skips the runner-style suites (test_cones.py, "
        "test_sample_cone_iso.py) and its single-process collection "
        "defeats the per-process spine-freeness assertions.  Run the "
        "validation gate instead:  python3 run_tests.py"
    )
