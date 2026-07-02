"""Validation gate — pure Python 3, no third-party dependencies.

Puts every ``src/<layer>/`` directory on ``sys.path`` (the project's bare-name
import convention), then runs the contract self-tests for all four layers:

    python3 run_tests.py

Step 1 / Step 2 (core, samples, cones, isomorphism witnesses):

  * ``tests/test_samples.py``          — Step-1 sample algebras
  * ``tests/test_cones.py``            — Step-2 ConeKAlgebra realizations + zoo
  * ``tests/test_sample_cone_iso.py``  — Step-1 ↔ Step-2 KAlgebraIso witnesses

Step 3 (the live RG-flow engine; all but ``test_rg_flows.py`` also assert
spine-freeness):

  * ``tests/test_rg_flows.py``         — the three reference flows
  * ``tests/test_a1an_chain.py``       — the A-type Argyres-Douglas chain
  * ``tests/test_dn_chain.py``         — the D-type gauged chain
  * ``tests/test_e_type.py``           — the E6 / E8 / gauged-E7 flows
  * ``tests/test_flavoured_fork.py``   — the ungauged add_flavour fork
  * ``tests/test_over_pure.py``        — the over-pure SU(2) gauge-theory corner
  * ``tests/test_su2_gauged_chain.py`` — the nested SU(2)-gauged chain
  * ``tests/test_wild.py``             — the "wild" formal flows

Step 4 (the BPS-quiver realisation engine — *this* is the spine):

  * ``tests/test_bps_flows.py``        — the BPS realisation + atlas + node-drop RG

Each prints its own ``PASS`` / ``ALL ... PASSED`` line. (``test_samples`` also
runs under ``pytest`` via ``conftest.py``; the others are runner scripts.)
``test_cones`` is the slowest (a few minutes). ``test_bps_flows`` is run **last**
because it imports the BPS spine — so the spine-freeness assertions in the
Step-3 suites (which require no spine module in ``sys.modules``) hold when they
run earlier in the same process.
"""
import pathlib
import runpy
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
_SRC = _ROOT / "src"
sys.path[:0] = [
    str(p) for p in _SRC.rglob("*") if p.is_dir() and p.name != "__pycache__"
]

for _test in ("tests/test_samples.py", "tests/test_cones.py",
              "tests/test_sample_cone_iso.py",
              "tests/test_rg_flows.py", "tests/test_a1an_chain.py",
              "tests/test_dn_chain.py", "tests/test_e_type.py",
              "tests/test_flavoured_fork.py", "tests/test_over_pure.py",
              "tests/test_su2_gauged_chain.py", "tests/test_wild.py",
              "tests/test_bps_flows.py"):
    print(f"\n=== {_test} ===")
    runpy.run_path(str(_ROOT / _test), run_name="__main__")
