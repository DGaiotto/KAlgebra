"""
u1e7_gauged_rg.py
=================

**Back-compat shim.**  The u(1)-gauged E7 RG flow is now the canonical
`U1A1E7RGKAlgebra` (`u1a1e7_rgkalgebra.py`) — a pure exact-FS `RGKAlgebra`
over `A1A2kKAlg(3) ⊗ QT(Z²)` with `S_RG = E_𝖖(X_{(0,1)}·L_{(3,0)})`, `(3, 0)`
the central chord `E7RGKAlgebra` dresses (an earlier `(2, 2)` dressing gave
`[A₁,D₇]` with the Cartan `U(1)` of its `SU(2)` gauged; see that module).  The old
name `U1E7GaugedRG` is retained here as an alias so existing callers (the
`u1e7_cone_kalgebra` oracle, `u1e7_cone_derivation`, `test_u1e7_gauged_rg`)
keep working unchanged.

The previous standalone class carried a docstring caveat that its trace was
"slow / non-converging in a tight q-window" — that predated the nested-aux
exact-FS engine; the canonical class's exact-FS trace is now
truncation-safe and fast.  See `u1a1e7_rgkalgebra.py`.
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from u1a1e7_rgkalgebra import U1A1E7RGKAlgebra, _e_q_coeff   # noqa: F401

# Back-compat alias: U1E7GaugedRG is the canonical U1A1E7RGKAlgebra.
U1E7GaugedRG = U1A1E7RGKAlgebra

__all__ = ["U1E7GaugedRG", "U1A1E7RGKAlgebra", "_e_q_coeff"]


if __name__ == "__main__":
    T = U1E7GaugedRG()
    print("U1E7GaugedRG is U1A1E7RGKAlgebra:", U1E7GaugedRG is U1A1E7RGKAlgebra)
    print("  aux =", type(T.auxiliary()).__name__,
          " dressing L = (type", T.DRESS_TYPE, ", i =", T._i0, ") = the central chord")
