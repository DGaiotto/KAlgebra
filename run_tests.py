"""Validation gate — pure Python 3, no third-party dependencies.

Puts every ``src/<layer>/`` directory on ``sys.path`` (the project's bare-name
import convention), then runs the three contract self-tests:

    python3 run_tests.py

  * ``tests/test_samples.py``          — Step-1 sample algebras
  * ``tests/test_cones.py``            — Step-2 ConeKAlgebra realizations + zoo
  * ``tests/test_sample_cone_iso.py``  — Step-1 ↔ Step-2 KAlgebraIso witnesses

Each prints its own ``ALL ... PASSED`` line. (``test_samples`` also runs under
``pytest`` via ``conftest.py``; the other two are runner scripts.)
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
              "tests/test_sample_cone_iso.py"):
    print(f"\n=== {_test} ===")
    runpy.run_path(str(_ROOT / _test), run_name="__main__")
