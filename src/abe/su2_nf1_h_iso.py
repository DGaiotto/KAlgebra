"""`su2_nf1_h_iso` — **test tool** `KAlgebraIso` between the standalone
`SU2Nf1KAlgebra` (native `(h_factors, μ_power)` labels) and the
canonical-surface `BPSKAlgebra` companion (BPS tropical `Z³` labels
from `build_bps_su2_nf1()`).

The standalone algebra carries its own primitives (multiply, ρ,
coefficient ring, cone data, label section decomposition, **and trace
via cyclicity reduction to `Tr(W_n)` from Schur F**); the iso exists
to **verify** that these standalone primitives agree term-for-term
with the BPS companion.  Trace is **not** delegated to BPS — see
`su2_nf1_h_trace` for the standalone cyclicity-based derivation.

Label maps
----------

* **Forward** (native → BPS):

      ((h_factors, μ_pow))  ↦  sum_i mult_i · piecewise(H_{n_i})
                               + (0, -e_W, 0) for Wilson letter
                               + (0, 0, μ_pow)

  with the piecewise H_n table from `su2_nf1_bps_decoder._h_bps`.

* **Inverse** (BPS → native):  `bps_to_seed` plus
  `_psu2nf1_to_native(m, e, μ_pow)`.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly
from su2_nf1_kalgebra import SU2Nf1KAlgebra
from bps_su2_nf1 import build_bps_su2_nf1
from su2_nf1_bps_decoder import _h_bps, bps_to_seed
from su2_nf1_h_multiply import _psu2nf1_to_native


__all__ = ["su2_nf1_h_iso"]


def _native_to_bps_tropical(label: tuple) -> tuple:
    """Native `(h_factors, μ_power)` → BPS tropical `Z³` charge.

    Sums the per-H-letter piecewise charges (with multiplicities),
    adds `(0, -e, 0)` for the Wilson letter (if any), and adds
    `(0, 0, μ_power)` for the flavour shift.
    """
    h_factors, mu_pow = label
    n1, n2, n3 = 0, 0, int(mu_pow)
    for entry in h_factors:
        gen, exp = entry
        if isinstance(gen, int):                 # H_n
            h = _h_bps(gen)
            n1 += exp * h[0]
            n2 += exp * h[1]
        elif isinstance(gen, tuple) and gen[0] == 'W':
            assert exp == 1, f"Wilson exp must be 1, got {entry}"
            n2 -= gen[1]                         # W_e = (0, -e, 0)
        else:
            raise ValueError(f"Unrecognised native generator: {entry}")
    return (n1, n2, n3)


def _forward_label_map(label: tuple) -> Element:
    """Source native label → target BPS Element (q-Laurent unit at the
    mapped tropical charge)."""
    bps_charge = _native_to_bps_tropical(label)
    return Element({bps_charge: LaurentPoly.one()})


def _inverse_label_map(label: tuple) -> Element:
    """Target BPS tropical → source native Element.

    Decodes via `bps_to_seed` (gives `((m, e), μ_pow)`) then
    canonicalises to native `(h_factors, μ_pow)` via
    `_psu2nf1_to_native`.
    """
    (m, e), mu_pow = bps_to_seed(label)
    native = _psu2nf1_to_native(m, e, mu_pow)
    return Element({native: LaurentPoly.one()})


def su2_nf1_h_iso() -> KAlgebraIso:
    """`KAlgebraIso(SU2Nf1KAlgebra, BPSKAlgebra for SU(2)+Nf=1)`.

    The standalone-side multiply / ρ / coefficient ring are independent
    of BPS; the iso witnesses they coincide with the canonical BPS
    realisation term-for-term.  Trace delegation: see
    `SU2Nf1KAlgebra.trace`.
    """
    return KAlgebraIso(
        source=SU2Nf1KAlgebra(),
        target=build_bps_su2_nf1(),
        forward_label_map=_forward_label_map,
        inverse_label_map=_inverse_label_map,
        name="SU2Nf1KAlgebra ≅ BPSKAlgebra[SU(2)+Nf=1]",
    )
