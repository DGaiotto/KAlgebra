"""Back-compatibility shim — the class is now `GNAbeKAlgebra`.

**Renamed.**  `GMatterAbeKAlgebra` became
`gn_abe_kalgebra.GNAbeKAlgebra`, which takes the **4d gauge group data**
(`global_form.LineLattice` — a maximal set of mutually compatible Kapustin
`(m, e)` labels) and the matter as arguments.  The same ruling retired the
per-theory subclass roster: *"use special names only if you have algorithms
specifically optimized for a G and/or N"*; of the optimized classes it had in
view (`UNNfKAlgebra`, `PureUNKAlgebra`, `PureSU2KAlgebra`), only
`PureSU2KAlgebra` remains — the other two were retired on 2026-09-19.

This module re-exports the class under its old name so existing callers and
`tests/test_g_matter_abe_kalgebra.py` keep working.  New code should import
`GNAbeKAlgebra` from `gn_abe_kalgebra`.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gn_abe_kalgebra import GNAbeKAlgebra

#: Deprecated alias for `GNAbeKAlgebra`.
GMatterAbeKAlgebra = GNAbeKAlgebra

__all__ = ["GNAbeKAlgebra", "GMatterAbeKAlgebra"]
