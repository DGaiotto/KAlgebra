"""Auto-extend sys.path with the tree's `src/<layer>/` directories.

Modules import one another by bare name and live in `src/<layer>/`.  A run from
the repository root with `PYTHONPATH=.` puts the root on `sys.path`; Python then
imports this module at startup and it appends every `src/<layer>/` directory, so
`PYTHONPATH=. python3 tests/test_x.py` — and a child process a test starts the
same way — resolves every module.  `run_tests.py` sets up the same path itself
and does not rely on this file.
"""
import os
import sys

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_src):
    for _d in sorted(os.listdir(_src)):
        _p = os.path.join(_src, _d)
        if os.path.isdir(_p) and _d != "__pycache__" and _p not in sys.path:
            sys.path.append(_p)
