"""`su2_nf2_h_iso` — **test tool** `KAlgebraIso` between standalone
`SU2Nf2KAlgebra` (tropical Z⁴ labels) and the canonical-surface
`BPSKAlgebra` companion from `bps_su2_nf2.build_bps_su2_nf2()`.

Since both algebras now use tropical Z⁴ labels directly, the iso is
the IDENTITY on labels.  Verification confirms that multiply / ρ /
trace agree term-for-term between the two presentations.

The standalone algebra is NEVER computed via this iso — it has its
own primitives.  The iso exists strictly to VERIFY them against the
BPS realisation.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly
from su2_nf2_kalgebra import SU2Nf2KAlgebra
from bps_su2_nf2 import build_bps_su2_nf2


__all__ = ["su2_nf2_h_iso"]


def _identity_label_map(label):
    """Both algebras use tropical Z⁴ canonical-basis labels — the iso
    is the identity on labels."""
    return Element({tuple(label): LaurentPoly.one()})


def su2_nf2_h_iso() -> KAlgebraIso:
    """`KAlgebraIso(SU2Nf2KAlgebra, BPSKAlgebra for SU(2)+Nf=2)`.

    Identity on labels (both use tropical Z⁴).  Test tool only.
    """
    return KAlgebraIso(
        source=SU2Nf2KAlgebra(),
        target=build_bps_su2_nf2(),
        forward_label_map=_identity_label_map,
        inverse_label_map=_identity_label_map,
        name="SU2Nf2KAlgebra ≅ BPSKAlgebra[SU(2)+Nf=2]",
    )
