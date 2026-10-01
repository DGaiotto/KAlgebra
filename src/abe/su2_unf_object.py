"""`KAlgebraObject` for **SU(2) + N_f**.

The abstract algebra `A_𝖖[SU(2)+N_f]` at its **Spin(2N_f)-manifest** level:

* N_f=2: the canonical-surface pair `cone` (`SU2Nf2KAlgebra`, standalone
  Spin(4) = SU(2)_L×SU(2)_R) ↔ `bps` (`build_bps_su2_nf2`).  Both carry
  tropical `Z⁴` labels, so the witness is the identity-on-labels
  `su2_nf2_h_iso`; unit / round-trip / multiply / ρ certify **raw**, and the
  trace certifies in the **irrep↔Cartan folded frame** (cone carries flavour
  as Spin(4) irreps `SU(2)_L×SU(2)_R`, bps as Cartan weights
  `AbelianZPlusRing(2)`; the same index in two ring conventions —
  the suite in the source repository).
* N_f=3: the Spin(6)-manifest `bps` is `build_bps_su2_nf3` (manifest
  `SU(4) ≅ SO(6)` flavour, rank-3 Dynkin Cartan).
* N_f=4: there is no BPS generator, so the object fails honestly.

**Retired 2026-09-19 with the type-A layer: the U(N_f)-manifest `abe` leg.**
Until then the object also carried `abe` = `SU2UNf{N_f}AbeKAlgebra`, the
**ungauged** realization on the rational quantum torus (the diagonal U(1) of
`UNNfKAlgebra(2,N_f)` ungauged), whose manifest flavour is
`U(N_f) ⊂ SO(2N_f)`.  It was bridged to the Spin(2N_f) core by its
`SO(2N_f)`-lifted index (`su2_unf_abe_kalgebra.so2nf_index`), equal to
`bps`'s index exactly (`certify_abe_bps_so2nf`, pinned through q⁴ at N_f=2)
and, at N_f=3, by the q-graded dimension match `certify_abe_bps_dim`; at N_f=4
it was the only realization.  See `su2_unf_object` below for what restoring it
on the general tier takes.

Realizations (N_f=2):

* ``'cone'`` — `SU2Nf2KAlgebra`: standalone Spin(4)-manifest KAlgebra
  (tropical `Z⁴`, Spin(4) characters in `R`).
* ``'bps'``  — `build_bps_su2_nf2()`: the canonical `BPSKAlgebra` on the
  Spin(4)-manifest mutated chart.

Witnesses:

* ``cone ↔ bps`` — `su2_nf2_h_iso` (identity on `Z⁴`; unit/round-trip/
  multiply/ρ raw, trace in the irrep↔Cartan folded frame).
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra_object import KAlgebraObject


__all__ = ["su2_unf_object"]


def su2_unf_object(Nf: int = 2) -> KAlgebraObject:
    """The abstract `A_𝖖[SU(2)+N_f]` as a `KAlgebraObject`.

    N_f=2 is the certified Spin(4) pair (`cone ↔ bps`); N_f=3 the Spin(6)
    `bps` (`build_bps_su2_nf3`).  The ungauged, SO(2N_f)-enhanced `abe` leg
    (`SU2UNf{N_f}AbeKAlgebra` over `UNNfKAlgebra(2, N_f)`) was retired to
    the source repository's archive on 2026-09-19 with the type-A layer, and with it the N_f=4
    entry it alone carried — `Nf=4` now fails honestly.  The SO(2N_f)
    recognition itself has since been re-derived on the general tier
    (2026-09-24: the traces, pairings and structure constants of
    `GNAbeKAlgebra(su_2(), (1,), N_f)`, twisted by the det power ρ fixes, lift
    to `R(Spin(2N_f))` — the suite in the source repository); restoring
    the leg on this object is the open step, and
    the archived tree keeps the old leg's science runnable.
    """
    if Nf not in (2, 3):
        raise ValueError(f"su2_unf_object: no realisation on the canonical "
                         f"surface for Nf={Nf} (the SO(2Nf)-enhanced abe leg "
                         f"is in the source repository's archive since 2026-09-19)")
    obj = KAlgebraObject(f"A_q[SU(2)+Nf={Nf}]")

    if Nf == 2:
        from su2_nf2_h_iso import su2_nf2_h_iso
        iso_cb = su2_nf2_h_iso()
        cone = iso_cb.source
        bps = iso_cb.target
        obj.add_realization("cone", cone,
                            {"closed-form", "trace-exact", "spin4-manifest"})
        obj.add_realization("bps", bps,
                            {"chart", "rg", "trace-exact", "spin4-manifest"})
        obj.add_iso("cone", "bps", iso_cb)
    elif Nf == 3:
        from bps_su2_nf3 import build_bps_su2_nf3
        obj.add_realization("bps", build_bps_su2_nf3(),
                            {"chart", "rg", "trace-exact", "spin6-manifest"})

    return obj


if __name__ == "__main__":
    obj = su2_unf_object(2)
    print(obj)
    print("  keys:", obj.keys())
    for k in obj.keys():
        print(f"    {k}: {sorted(obj.capabilities(k))}")
    # cone ↔ bps transport (identity on Z⁴)
    print("  cone→bps transport (0,0,0,0):",
          dict(obj.transport((0, 0, 0, 0), "cone", "bps").terms))
