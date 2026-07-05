"""Back-compat shim — the pure-U(N) canonical surface now lives in the unified
`pure_un_kalgebra` module (Section 1: bar-centered loops + Schur trace + ρ).

Importing from `pure_un_canonical` continues to work; every name is re-exported
from `pure_un_kalgebra`.  New code should import from `pure_un_kalgebra`
directly.  See that module's header for the unification rationale.
"""
from __future__ import annotations

from pure_un_kalgebra import (              # noqa: F401  (re-export)
    _ipair, lam_E, lam_F, lam_det, _qpow, L_E, L_F, L_det, L_W,
    _vdiv, _inv_root, trace, inner,
    witten_shift, _w0neg, rho_label, rho2_label,
    dop_of, rho_pairing_const,
)
