"""pytest path bootstrap.

The library modules under ``src/<layer>/`` import one another by **bare name**
(``from kalgebra import ...``) — the project's convention. This puts every
``src/`` subdirectory on ``sys.path`` so those imports resolve when running
``pytest`` from the repository root.
"""
import sys
import pathlib

_SRC = pathlib.Path(__file__).resolve().parent / "src"
for _d in _SRC.rglob("*"):
    if _d.is_dir() and _d.name != "__pycache__":
        _s = str(_d)
        if _s not in sys.path:
            sys.path.insert(0, _s)
