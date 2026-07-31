"""Validation gate — pure Python 3, no third-party dependencies.

Puts every ``src/<layer>/`` directory on ``sys.path`` (the project's bare-name
import convention), then runs the contract self-tests for every layer:

    python3 run_tests.py

Step 1 / Step 2 (core, samples, cones, isomorphism witnesses):

  * ``tests/test_samples.py``          — Step-1 sample algebras
  * ``tests/test_cones.py``            — Step-2 ConeKAlgebra realizations + zoo
  * ``tests/test_sample_cone_iso.py``  — Step-1 ↔ Step-2 KAlgebraIso witnesses

Step 3 (the live RG-flow engine; every suite asserts spine-freeness via the
shared, filesystem-derived list in ``tests/_spine.py``):

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

Step 5 (the abelianized-presentation tier + the object-layer capstone; relies on
Steps 1-4 by design, so it runs after the spine-free suites):

  * ``tests/test_abe_flows.py``        — the AbeKAlgebra tier (pure U(N)/SU(N),
    matter as Abe+RG, N=2* machinery, the KAlgebraObject layer, SU(2)/SU(3)
    families, and the flow-typed RGKAlgebraObjects)

Step 6 (the skein tier — SU(2) skein algebras of marked surfaces realised as
``A_𝖖[T]``; relies on Steps 1-5, so it also runs after the spine-free suites):

  * ``tests/test_skein_flows.py``      — the SkeinKAlgebra tier (the intrinsic
    Kauffman-bracket layer, the two-parent ``SkeinKAlgebra`` class, the named
    roster, the KAlgebraIso / KAlgebraObject legs, the contract axioms, the
    ``SkeinAtlas`` flip charts, and the independent BPS-quiver vacuum anchor)

Step 7 (gauge theory at an arbitrary 4d gauge group with arbitrary matter;
relies on Steps 1-6, so it also runs after the spine-free suites):

  * ``tests/test_gn_flows.py``         — general-`G` pure gauge over an arbitrary
    root datum with the guarded route ladder and the licensed (star) solve, `(G, N)`
    matter at any datum and any matter representation, the 4d gauge group data and
    its Langlands duals, and the vacuum-state Schur pairing

Notes on the order and the entry point:

  * ``pytest`` is **not** a supported entry point and is refused loudly by
    ``conftest.py``.  It would skip ``test_cones.py`` and
    ``test_sample_cone_iso.py`` (neither exposes ``test_``-prefixed
    collectables), and importing the BPS suite at collection time would defeat
    the spine-freeness assertions the Step-3 suites make.  Use this script.
  * ``test_cones.py`` is the slowest suite by a wide margin — it walks the whole
    finite-type cone zoo.
  * the spine-importing tiers (Steps 4-7) run **last**, after every suite that
    asserts no spine module is loaded, so that ordering is what makes those
    assertions meaningful rather than accidental.

"""
import pathlib
import runpy
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
_SRC = _ROOT / "src"
sys.path[:0] = [
    str(p) for p in _SRC.rglob("*") if p.is_dir() and p.name != "__pycache__"
]
# tests/ itself: the shared spine-freeness helper (tests/_spine.py) is
# imported by the Step-3 suites.
sys.path.insert(0, str(_ROOT / "tests"))

for _test in ("tests/test_samples.py", "tests/test_cones.py",
              "tests/test_sample_cone_iso.py",
              "tests/test_rg_flows.py", "tests/test_a1an_chain.py",
              "tests/test_dn_chain.py", "tests/test_e_type.py",
              "tests/test_flavoured_fork.py", "tests/test_over_pure.py",
              "tests/test_su2_gauged_chain.py", "tests/test_wild.py",
              "tests/test_bps_flows.py", "tests/test_abe_flows.py",
              "tests/test_skein_flows.py", "tests/test_gn_flows.py"):
    print(f"\n=== {_test} ===")
    runpy.run_path(str(_ROOT / _test), run_name="__main__")
