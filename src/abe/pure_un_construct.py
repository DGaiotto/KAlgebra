"""Back-compat shim — the pure-U(N) canonical basis now lives in the unified
`pure_un_kalgebra` module: the `CanonicalBasis` registry built on demand by the
Principal-QTCone / `w_R·cone` engine (`_build_clean`), plus multiply / trace.

Importing the still-live names from `pure_un_construct` continues to work; new
code should import from `pure_un_kalgebra` directly.  (The obsolete q⁰-Gram-
Schmidt constructor — `construct_basis` / `autobuild` / `register_via_M` — was
removed: q⁰ pairing cannot certify bar-invariance, and it failed on the
asymmetric Levi dressings the cone/`w_R` engine builds correctly.)
"""
from __future__ import annotations

from pure_un_kalgebra import (              # noqa: F401  (re-export)
    _sc, _gen, _rgen, _product, _q0, _lowest_charge, _default_deg, CanonicalBasis,
)
