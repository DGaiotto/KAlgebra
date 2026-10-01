"""`A1D5KAlg` — the [A_1, D_5] Argyres-Douglas K-algebra,
SU(2)-flavoured realisation with manifest Z_5 cluster symmetry.

Fully self-contained K-algebra: no external oracle (BPS or otherwise)
is imported at runtime.  All multiplications are driven by the
hardcoded `_PLUCKER_BASE_I0` table (i=0 row of mult-gen × mult-gen
products) plus Z_5 ρ-shift via `_RHO_TABLE`.

## Atomic generating set (4 ρ-orbits × 5 = 20 atomic mult-gens)

The atomic generators are the four Z_5 orbits

  T_i  =  ρ^i T_0        T_0 = (1, 0, 0, 0, 0)
  D_i  =  ρ^i D_0        D_0 = (0, 0, 0, -1, 0)
  V_i  =  ρ^i V_0        V_0 = (0, 1, 0, 1, 0)
  W_i  =  ρ^i W_0        W_0 = (1, 0, 1, 0, 0)

These are *atomic* in the sense that no element in any orbit
decomposes as a γ_1 + γ_2 tropical-charge sum with both γ_1, γ_2
nonzero and in a common cone of linearity of ρ^{-1}.  (Equivalently,
none is a positive cluster monomial of the others.)

## Derived cluster monomials

Elements such as the old "U_i" are *not* atomic — they are cluster
monomials of atomic generators.  Specifically

  U_0  =  D_1 · V_0    (in lattice charges: D_1 + V_0)

is the cluster monomial of D_1 and V_0 in their shared cone.  The
multiplication table outputs labels in the canonical basis
(charge-indexed); helper `is_cluster_monomial(label)` /
`as_cluster_monomial(label)` can decompose them on demand.

## Label format

A *label* is an integer 5-tuple representing a canonical-basis
element by its tropical charge in the cluster-algebra g-vector
frame.  This is the natural lattice for the canonical basis of the
K-algebra and is intrinsic to the algebra — not derived from any
auxiliary BPS construction.  (BPSKAlgebra happens to use the same
g-vector frame as its internal labelling, but that's a coincidence
of frame choice, not a dependency.)

## Status

Multiplication, ρ, ρ^{-1}, canonicalisation, identity, χ-shift, and
canonical-basis section decomposition are all implemented in closed
form.  The trace is NOT wired into this class and raises.

For `[A_1, D_5]` use `A1DnKAlg(5)` (`src/cone/a1dn_kalg.py`), rebuilt
2026-09-23 in the once-punctured-polygon frame:
its canonical-basis labels are the simple curves of the pentagon with one
interior puncture, and its products, ρ and trace are those of
`A1DoddConeKAlg(1)` (`src/cone/a1dodd_kalg.py`), to which the certified
iso `a1dn_a1dodd_iso` relates it.  The zoo's `FiniteA1D5KAlgebra`
(`finite_kalgebras`) also serves this algebra, with the explicit admissible
sl(2)_{-8/5} characters of `a1d5_layer2.py` as its traces (status:
the design notes).  This hand-written class is kept as an independent
witness of the products (the design record battery row `comp:a1dodd/implementation`,
through the flavour dictionary `χ_n·L_γ ↔` the label decomposing to
`(γ, n)`).
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element
from cone_kalgebra import ConeKAlgebra
from zplus_ring import ZPlusRing, RElement, RLaurent, SU2ZPlusRing
from laurent_poly import LaurentPoly


# ---------------------------------------------------------------------------
# Hard-coded canonical-basis data (no BPS or oracle dependency at runtime)
# ---------------------------------------------------------------------------

# Auto-generated A1D5KAlg data — atomic basis {T, D, V, W}, with ρ closure.

_T_ORBIT = (
    (1, 0, 0, 0, 0),
    (-1, -1, -1, -2, 0),
    (0, 0, -1, 0, 0),
    (0, -1, 0, 0, 0),
    (-1, 0, 0, 0, 0),
)

_D_ORBIT = (
    (0, 0, 0, -1, 0),
    (0, 0, -1, -1, 0),
    (0, -1, -1, -1, 0),
    (-1, -1, -1, -1, 0),
    (0, 0, 0, 1, 0),
)

_V_ORBIT = (
    (0, 1, 0, 1, 0),
    (0, -1, 0, -1, 0),
    (-1, 0, -1, -1, 0),
    (1, 0, 0, 1, 0),
    (-1, -1, 0, -1, 0),
)

_W_ORBIT = (
    (1, 0, 1, 0, 0),
    (-1, 0, -1, -2, 0),
    (1, 0, -1, 0, 0),
    (-1, -2, -1, -2, 0),
    (-1, 0, -1, 0, 0),
)

_RHO_TABLE = {
    (-2, -4, -2, -4, 0): (-2, 0, -2, 0, 0),
    (-2, -3, -2, -4, 0): (-1, 0, -2, 0, 0),
    (-2, -3, -2, -3, 0): (-1, 0, -1, 1, 0),
    (-2, -3, -1, -3, 0): (-1, 1, -1, 1, 0),
    (-2, -2, -2, -4, 0): (0, 0, -2, 0, 0),
    (-2, -2, -2, -3, 0): (0, 0, -1, 1, 0),
    (-2, -2, -2, -2, 0): (0, 0, 0, 2, 0),
    (-2, -2, -1, -3, -1): (0, 1, -1, 1, -1),
    (-2, -2, -1, -3, 0): (0, 1, -1, 1, 0),
    (-2, -2, -1, -2, 0): (0, 1, 0, 2, 0),
    (-2, -2, 0, -2, 0): (0, 2, 0, 2, 0),
    (-2, -1, -2, -4, 0): (1, 0, -2, 0, 0),
    (-2, -1, -2, -3, 0): (1, 0, -1, 1, 0),
    (-2, -1, -2, -2, 0): (1, 0, 0, 2, 0),
    (-2, -1, -2, -1, 0): (1, 0, 1, 1, 0),
    (-2, -1, -1, -3, 0): (1, 1, -1, 1, 0),
    (-2, -1, -1, -2, -1): (1, 1, 0, 2, -1),
    (-2, -1, -1, -2, 0): (1, 1, 0, 2, 0),
    (-2, -1, -1, -1, 0): (1, 1, 1, 1, 0),
    (-2, -1, 0, -1, 0): (1, 1, 0, 1, 0),
    (-2, 0, -2, -4, 0): (2, 0, -2, 0, 0),
    (-2, 0, -2, -3, 0): (2, 0, -1, 1, 0),
    (-2, 0, -2, -2, 0): (2, 0, 0, 2, 0),
    (-2, 0, -2, -1, 0): (2, 0, 1, 1, 0),
    (-2, 0, -2, 0, 0): (2, 0, 2, 0, 0),
    (-2, 0, -1, -2, 0): (2, 0, -1, 0, 0),
    (-2, 0, -1, -1, -1): (2, 0, 0, 1, -1),
    (-2, 0, -1, -1, 0): (2, 0, 0, 1, 0),
    (-2, 0, -1, 0, 0): (2, 0, 1, 0, 0),
    (-2, 0, 0, 0, 0): (2, 0, 0, 0, 0),
    (-1, -3, -2, -3, 0): (-2, -1, -2, -1, 0),
    (-1, -3, -1, -3, 0): (-2, 0, -2, -1, 0),
    (-1, -3, -1, -2, 0): (-2, 0, -1, 0, 0),
    (-1, -2, -2, -3, 0): (-1, -1, -2, -1, 0),
    (-1, -2, -2, -2, 0): (-1, -1, -1, 0, 0),
    (-1, -2, -1, -3, 0): (-1, 0, -2, -1, 0),
    (-1, -2, -1, -2, 0): (-1, 0, -1, 0, 0),
    (-1, -2, -1, -1, 0): (-1, 0, 0, 1, 0),
    (-1, -2, 0, -2, -1): (-1, 1, -1, 0, -1),
    (-1, -2, 0, -2, 0): (-1, 1, -1, 0, 0),
    (-1, -2, 0, -1, 0): (-1, 1, 0, 1, 0),
    (-1, -1, -2, -3, 0): (0, -1, -2, -1, 0),
    (-1, -1, -2, -2, 0): (0, -1, -1, 0, 0),
    (-1, -1, -2, -1, 0): (0, -1, 0, 1, 0),
    (-1, -1, -1, -3, 0): (0, 0, -2, -1, 0),
    (-1, -1, -1, -2, -1): (0, 0, -1, 0, -1),
    (-1, -1, -1, -2, 0): (0, 0, -1, 0, 0),
    (-1, -1, -1, -1, -1): (0, 0, 0, 1, -1),
    (-1, -1, -1, -1, 0): (0, 0, 0, 1, 0),
    (-1, -1, -1, 0, 0): (0, 0, 1, 0, 0),
    (-1, -1, 0, -2, 0): (0, 1, -1, 0, 0),
    (-1, -1, 0, -1, -1): (0, 1, 0, 1, -1),
    (-1, -1, 0, -1, 0): (0, 1, 0, 1, 0),
    (-1, -1, 0, 0, 0): (0, 1, 0, 0, 0),
    (-1, 0, -2, -3, 0): (1, -1, -2, -1, 0),
    (-1, 0, -2, -2, 0): (1, -1, -1, 0, 0),
    (-1, 0, -2, -1, 0): (1, -1, 0, 1, 0),
    (-1, 0, -2, 0, 0): (1, -1, 1, 0, 0),
    (-1, 0, -1, -3, 0): (1, 0, -2, -1, 0),
    (-1, 0, -1, -2, 0): (1, 0, -1, 0, 0),
    (-1, 0, -1, -1, -1): (1, 0, 0, 1, -1),
    (-1, 0, -1, -1, 0): (1, 0, 0, 1, 0),
    (-1, 0, -1, 0, 0): (1, 0, 1, 0, 0),
    (-1, 0, -1, 1, 0): (1, 0, 1, -1, 0),
    (-1, 0, 0, -1, 0): (1, 0, -1, -1, 0),
    (-1, 0, 0, 0, -1): (1, 0, 0, 0, -1),
    (-1, 0, 0, 0, 0): (1, 0, 0, 0, 0),
    (-1, 0, 0, 1, 0): (1, 0, 0, -1, 0),
    (-1, 1, -1, -1, 0): (1, -1, -1, -1, 0),
    (-1, 1, -1, 0, -1): (1, -1, 0, 0, -1),
    (-1, 1, -1, 0, 0): (1, -1, 0, 0, 0),
    (-1, 1, -1, 1, 0): (1, -1, 1, -1, 0),
    (-1, 1, 0, 1, 0): (1, -1, 0, -1, 0),
    (0, -2, -2, -2, 0): (-2, -2, -2, -2, 0),
    (0, -2, -1, -2, 0): (-2, -1, -2, -2, 0),
    (0, -2, -1, -1, 0): (-2, -1, -1, -1, 0),
    (0, -2, 0, -2, 0): (-2, 0, -2, -2, 0),
    (0, -2, 0, -1, -1): (-2, 0, -1, -1, -1),
    (0, -2, 0, -1, 0): (-2, 0, -1, -1, 0),
    (0, -2, 0, 0, 0): (-2, 0, 0, 0, 0),
    (0, -1, -2, -2, 0): (-1, -2, -2, -2, 0),
    (0, -1, -2, -1, 0): (-1, -2, -1, -1, 0),
    (0, -1, -1, -2, 0): (-1, -1, -2, -2, 0),
    (0, -1, -1, -1, -1): (-1, -1, -1, -1, -1),
    (0, -1, -1, -1, 0): (-1, -1, -1, -1, 0),
    (0, -1, -1, 0, 0): (-1, -1, 0, 0, 0),
    (0, -1, 0, -2, 0): (-1, 0, -2, -2, 0),
    (0, -1, 0, -1, -1): (-1, 0, -1, -1, -1),
    (0, -1, 0, -1, 0): (-1, 0, -1, -1, 0),
    (0, -1, 0, 0, -1): (-1, 0, 0, 0, -1),
    (0, -1, 0, 0, 0): (-1, 0, 0, 0, 0),
    (0, -1, 0, 1, 0): (-1, 0, 0, -1, 0),
    (0, -1, 1, -1, 0): (-1, 1, -1, -1, 0),
    (0, 0, -2, -2, 0): (0, -2, -2, -2, 0),
    (0, 0, -2, -1, 0): (0, -2, -1, -1, 0),
    (0, 0, -2, 0, 0): (0, -2, 0, 0, 0),
    (0, 0, -1, -2, 0): (0, -1, -2, -2, 0),
    (0, 0, -1, -1, -1): (0, -1, -1, -1, -1),
    (0, 0, -1, -1, 0): (0, -1, -1, -1, 0),
    (0, 0, -1, 0, -1): (0, -1, 0, 0, -1),
    (0, 0, -1, 0, 0): (0, -1, 0, 0, 0),
    (0, 0, -1, 1, 0): (0, -1, 1, -1, 0),
    (0, 0, 0, -2, 0): (0, 0, -2, -2, 0),
    (0, 0, 0, -1, -1): (0, 0, -1, -1, -1),
    (0, 0, 0, -1, 0): (0, 0, -1, -1, 0),
    (0, 0, 0, 0, -1): (0, 0, 0, 0, -1),
    (0, 0, 0, 0, 0): (0, 0, 0, 0, 0),
    (0, 0, 0, 1, -1): (0, 0, 0, -1, -1),
    (0, 0, 0, 1, 0): (0, 0, 0, -1, 0),
    (0, 0, 0, 2, 0): (0, 0, 0, -2, 0),
    (0, 0, 1, 0, 0): (0, 0, -1, -2, 0),
    (0, 1, -1, 0, 0): (0, -2, -1, -2, 0),
    (0, 1, -1, 1, -1): (0, -2, 0, -1, -1),
    (0, 1, -1, 1, 0): (0, -2, 0, -1, 0),
    (0, 1, 0, 0, 0): (0, -1, -1, -2, 0),
    (0, 1, 0, 1, -1): (0, -1, 0, -1, -1),
    (0, 1, 0, 1, 0): (0, -1, 0, -1, 0),
    (0, 1, 0, 2, 0): (0, -1, 0, -2, 0),
    (0, 2, 0, 2, 0): (0, -2, 0, -2, 0),
    (1, -1, -2, -1, 0): (-2, -3, -2, -3, 0),
    (1, -1, -1, -1, 0): (-2, -2, -2, -3, 0),
    (1, -1, -1, 0, 0): (-2, -2, -1, -2, 0),
    (1, -1, 0, -1, 0): (-2, -1, -2, -3, 0),
    (1, -1, 0, 0, -1): (-2, -1, -1, -2, -1),
    (1, -1, 0, 0, 0): (-2, -1, -1, -2, 0),
    (1, -1, 0, 1, 0): (-2, -1, 0, -1, 0),
    (1, -1, 1, -1, 0): (-2, 0, -2, -3, 0),
    (1, -1, 1, 0, 0): (-2, 0, -1, -2, 0),
    (1, 0, -2, -1, 0): (-1, -3, -2, -3, 0),
    (1, 0, -2, 0, 0): (-1, -3, -1, -2, 0),
    (1, 0, -1, -1, 0): (-1, -2, -2, -3, 0),
    (1, 0, -1, 0, 0): (-1, -2, -1, -2, 0),
    (1, 0, -1, 1, 0): (-1, -2, 0, -1, 0),
    (1, 0, 0, -1, 0): (-1, -1, -2, -3, 0),
    (1, 0, 0, 0, -1): (-1, -1, -1, -2, -1),
    (1, 0, 0, 0, 0): (-1, -1, -1, -2, 0),
    (1, 0, 0, 1, -1): (-1, -1, 0, -1, -1),
    (1, 0, 0, 1, 0): (-1, -1, 0, -1, 0),
    (1, 0, 0, 2, 0): (-1, -1, 0, -2, 0),
    (1, 0, 1, -1, 0): (-1, 0, -2, -3, 0),
    (1, 0, 1, 0, 0): (-1, 0, -1, -2, 0),
    (1, 0, 1, 1, 0): (-1, 0, -1, -3, 0),
    (1, 1, -1, 1, 0): (-1, -3, -1, -3, 0),
    (1, 1, 0, 1, 0): (-1, -2, -1, -3, 0),
    (1, 1, 0, 2, -1): (-1, -2, 0, -2, -1),
    (1, 1, 0, 2, 0): (-1, -2, 0, -2, 0),
    (1, 1, 1, 1, 0): (-1, -1, -1, -3, 0),
    (2, 0, -2, 0, 0): (-2, -4, -2, -4, 0),
    (2, 0, -1, 0, 0): (-2, -3, -2, -4, 0),
    (2, 0, -1, 1, 0): (-2, -3, -1, -3, 0),
    (2, 0, 0, 0, 0): (-2, -2, -2, -4, 0),
    (2, 0, 0, 1, -1): (-2, -2, -1, -3, -1),
    (2, 0, 0, 1, 0): (-2, -2, -1, -3, 0),
    (2, 0, 0, 2, 0): (-2, -2, 0, -2, 0),
    (2, 0, 1, 0, 0): (-2, -1, -2, -4, 0),
    (2, 0, 1, 1, 0): (-2, -1, -1, -3, 0),
    (2, 0, 2, 0, 0): (-2, 0, -2, -4, 0),
}

_PLUCKER_BASE_I0 = {
    ('D', 'D', 0): ((1, 0, (0, 0, 0, -2, 0)),),
    ('D', 'D', 1): ((1, -1, (0, 0, -1, -2, 0)), (1, 0, (0, 0, 0, 0, 0))),
    ('D', 'D', 2): ((1, -1, (0, -1, -1, -2, 0)), (1, 0, (0, -1, 0, -1, -1)), (1, 0, (0, 0, 0, 0, 0)), (1, 1, (0, -1, 0, 0, 0))),
    ('D', 'D', 3): ((1, -1, (-1, -1, -1, -2, 0)), (1, 0, (-1, -1, 0, -1, -1)), (1, 0, (0, 0, 0, 0, 0)), (1, 1, (-1, -1, 0, 0, 0))),
    ('D', 'D', 4): ((1, 0, (0, 0, 0, 0, 0)), (1, 1, (0, 0, 1, 0, 0))),
    ('D', 'T', 0): ((1, 0, (1, 0, 0, -1, 0)),),
    ('D', 'T', 1): ((1, -1, (-1, -1, -1, -3, 0)), (1, 0, (-1, -1, 0, -1, 0))),
    ('D', 'T', 2): ((1, -1, (0, 0, -1, -1, 0)), (1, 0, (0, 0, 0, 0, -1)), (1, 0, (0, 1, 0, 1, 0)), (1, 1, (0, 0, 0, 1, 0))),
    ('D', 'T', 3): ((1, 0, (0, -1, 0, -1, 0)), (1, 1, (0, -1, 1, -1, 0))),
    ('D', 'T', 4): ((1, 0, (-1, 0, 0, -1, 0)),),
    ('D', 'V', 0): ((1, 0, (0, 1, 0, 0, 0)),),
    ('D', 'V', 1): ((1, 0, (0, -1, 0, -2, 0)),),
    ('D', 'V', 2): ((1, -1, (-1, 0, -1, -2, 0)), (1, 0, (-1, 0, 0, 0, 0))),
    ('D', 'V', 3): ((1, 0, (1, 0, 0, 0, 0)), (1, 1, (1, 0, 1, 0, 0))),
    ('D', 'V', 4): ((1, 0, (-1, -1, 0, -2, 0)),),
    ('D', 'W', 0): ((1, 1, (1, 0, 1, -1, 0)),),
    ('D', 'W', 1): ((1, -1, (-1, 0, -1, -3, 0)),),
    ('D', 'W', 2): ((1, -1, (1, 0, -1, -1, 0)), (1, 0, (1, 0, 0, 0, -1)), (1, 1, (1, 0, 0, 1, 0))),
    ('D', 'W', 3): ((1, -1, (-1, -2, -1, -3, 0)), (1, 0, (-1, -2, 0, -2, -1)), (1, 1, (-1, -2, 0, -1, 0))),
    ('D', 'W', 4): ((1, -1, (-1, 0, -1, -1, 0)), (1, 0, (-1, 0, 0, 0, -1)), (1, 1, (-1, 0, 0, 1, 0))),
    ('T', 'D', 0): ((1, 0, (1, 0, 0, -1, 0)),),
    ('T', 'D', 1): ((1, 0, (1, 0, -1, -1, 0)),),
    ('T', 'D', 2): ((1, -1, (1, -1, -1, -1, 0)), (1, 0, (1, 0, 0, 1, 0))),
    ('T', 'D', 3): ((1, -1, (0, -1, -1, -1, 0)), (1, 0, (0, 0, 0, 0, -1)), (1, 0, (1, 0, 0, 1, 0)), (1, 1, (0, 0, 0, 1, 0))),
    ('T', 'D', 4): ((1, 0, (1, 0, 0, 1, 0)), (1, 1, (1, 1, 1, 1, 0))),
    ('T', 'T', 0): ((1, 0, (2, 0, 0, 0, 0)),),
    ('T', 'T', 1): ((1, -1, (0, -1, -1, -2, 0)), (1, 0, (0, 0, 0, 0, 0))),
    ('T', 'T', 2): ((1, 0, (1, 0, -1, 0, 0)), (1, 1, (1, 1, 0, 2, 0))),
    ('T', 'T', 3): ((1, -1, (1, -1, 0, 0, 0)), (1, 0, (1, 0, 1, 0, 0))),
    ('T', 'T', 4): ((1, 0, (0, 0, 0, 0, 0)), (1, 1, (0, 1, 0, 0, 0))),
    ('T', 'V', 0): ((1, 1, (1, 1, 0, 1, 0)),),
    ('T', 'V', 1): ((1, -1, (1, -1, 0, -1, 0)),),
    ('T', 'V', 2): ((1, 0, (0, 0, -1, -1, 0)), (1, 1, (0, 1, 0, 1, 0))),
    ('T', 'V', 3): ((1, 0, (2, 0, 0, 1, 0)),),
    ('T', 'V', 4): ((1, -1, (0, -1, 0, -1, 0)), (1, 0, (0, 0, 0, -1, 0))),
    ('T', 'W', 0): ((1, 0, (2, 0, 1, 0, 0)),),
    ('T', 'W', 1): ((1, 0, (0, 0, -1, -2, 0)),),
    ('T', 'W', 2): ((1, 0, (2, 0, -1, 0, 0)),),
    ('T', 'W', 3): ((1, -2, (0, -2, -1, -2, 0)), (1, -1, (0, -1, 0, -1, -1)), (1, 0, (0, -1, 0, 0, 0))),
    ('T', 'W', 4): ((1, 0, (0, 0, -1, 0, 0)), (1, 1, (0, 1, 0, 1, -1)), (1, 2, (0, 1, 0, 2, 0))),
    ('V', 'D', 0): ((1, 0, (0, 1, 0, 0, 0)),),
    ('V', 'D', 1): ((1, 0, (0, 1, -1, 0, 0)),),
    ('V', 'D', 2): ((1, -1, (1, 0, -1, 0, 0)), (1, 0, (0, 0, -1, 0, 0))),
    ('V', 'D', 3): ((1, 0, (0, 0, -1, 0, 0)), (1, 1, (-1, 0, -1, 0, 0))),
    ('V', 'D', 4): ((1, 0, (0, 1, 0, 2, 0)),),
    ('V', 'T', 0): ((1, -1, (1, 1, 0, 1, 0)),),
    ('V', 'T', 1): ((1, 0, (0, 0, -1, -1, 0)), (1, 1, (-1, 0, -1, -1, 0))),
    ('V', 'T', 2): ((1, 0, (0, 1, -1, 1, 0)),),
    ('V', 'T', 3): ((1, -1, (1, 0, 0, 1, 0)), (1, 0, (0, 0, 0, 1, 0))),
    ('V', 'T', 4): ((1, 1, (-1, 1, 0, 1, 0)),),
    ('V', 'V', 0): ((1, 0, (0, 2, 0, 2, 0)),),
    ('V', 'V', 1): ((1, -1, (1, 0, 0, 0, 0)), (1, 0, (0, 0, 0, 0, 0))),
    ('V', 'V', 2): ((1, 1, (-1, 1, -1, 0, 0)),),
    ('V', 'V', 3): ((1, -1, (1, 1, 0, 2, 0)),),
    ('V', 'V', 4): ((1, 0, (0, 0, 0, 0, 0)), (1, 1, (-1, 0, 0, 0, 0))),
    ('V', 'W', 0): ((1, -1, (1, 1, 1, 1, 0)),),
    ('V', 'W', 1): ((1, 1, (-1, 1, -1, -1, 0)),),
    ('V', 'W', 2): ((1, -1, (1, 1, -1, 1, 0)),),
    ('V', 'W', 3): ((1, -1, (0, -1, -1, -1, 0)), (1, 0, (0, 0, 0, 0, -1)), (1, 1, (-1, -1, -1, -1, 0))),
    ('V', 'W', 4): ((1, 1, (-1, 1, -1, 1, 0)),),
    ('W', 'D', 0): ((1, -1, (1, 0, 1, -1, 0)),),
    ('W', 'D', 1): ((1, -1, (1, 0, 0, -1, 0)), (1, 0, (1, 0, 0, 0, -1)), (1, 1, (1, 0, 0, 1, 0))),
    ('W', 'D', 2): ((1, -1, (1, -1, 0, -1, 0)), (1, 0, (1, -1, 0, 0, -1)), (1, 1, (1, -1, 0, 1, 0))),
    ('W', 'D', 3): ((1, -1, (0, -1, 0, -1, 0)), (1, 0, (0, -1, 0, 0, -1)), (1, 1, (0, -1, 0, 1, 0))),
    ('W', 'D', 4): ((1, 1, (1, 0, 1, 1, 0)),),
    ('W', 'T', 0): ((1, 0, (2, 0, 1, 0, 0)),),
    ('W', 'T', 1): ((1, -2, (0, -1, 0, -2, 0)), (1, -1, (0, -1, 0, -1, -1)), (1, 0, (0, -1, 0, 0, 0))),
    ('W', 'T', 2): ((1, 0, (1, 0, 0, 0, 0)), (1, 1, (1, 0, 0, 1, -1)), (1, 2, (1, 0, 0, 2, 0))),
    ('W', 'T', 3): ((1, 0, (1, -1, 1, 0, 0)),),
    ('W', 'T', 4): ((1, 0, (0, 0, 1, 0, 0)),),
    ('W', 'V', 0): ((1, 1, (1, 1, 1, 1, 0)),),
    ('W', 'V', 1): ((1, -1, (1, -1, 1, -1, 0)),),
    ('W', 'V', 2): ((1, -1, (0, 0, 0, -1, 0)), (1, 0, (0, 0, 0, 0, -1)), (1, 1, (0, 0, 0, 1, 0))),
    ('W', 'V', 3): ((1, 1, (2, 0, 1, 1, 0)),),
    ('W', 'V', 4): ((1, -1, (0, -1, 1, -1, 0)),),
    ('W', 'W', 0): ((1, 0, (2, 0, 2, 0, 0)),),
    ('W', 'W', 1): ((1, -2, (0, 0, 0, -2, 0)), (1, -1, (0, 0, 0, -1, -1)), (1, 0, (0, 0, 0, 0, 0))),
    ('W', 'W', 2): ((1, 0, (2, 0, 0, 0, 0)), (1, 1, (2, 0, 0, 1, -1)), (1, 2, (2, 0, 0, 2, 0))),
    ('W', 'W', 3): ((1, -2, (0, -2, 0, -2, 0)), (1, -1, (0, -2, 0, -1, -1)), (1, 0, (0, -2, 0, 0, 0))),
    ('W', 'W', 4): ((1, 0, (0, 0, 0, 0, 0)), (1, 1, (0, 0, 0, 1, -1)), (1, 2, (0, 0, 0, 2, 0))),
}


_RHO_INV_TABLE: dict[tuple, tuple] = {v: k for k, v in _RHO_TABLE.items()}


# Reverse map: canonical lattice label → ('T'|'D'|'V'|'W', i).

_LATTICE_TO_MULTGEN: dict[tuple, tuple] = {}
for _kind, _orbit in (('T', _T_ORBIT), ('D', _D_ORBIT),
                      ('V', _V_ORBIT), ('W', _W_ORBIT)):
    for _i, _lab in enumerate(_orbit):
        _LATTICE_TO_MULTGEN[_lab] = (_kind, _i)


# ---------------------------------------------------------------------------
# ρ helpers
# ---------------------------------------------------------------------------


_IDENTITY_LABEL = (0, 0, 0, 0, 0)


def _rho_apply(label: tuple) -> tuple:
    if label == _IDENTITY_LABEL:
        return _IDENTITY_LABEL
    try:
        return _RHO_TABLE[label]
    except KeyError as e:
        raise KeyError(
            f"_RHO_TABLE missing entry for {label!r}.  Label is outside "
            f"the precomputed mult-gen × mult-gen closure."
        ) from e


def _rho_inverse_apply(label: tuple) -> tuple:
    if label == _IDENTITY_LABEL:
        return _IDENTITY_LABEL
    try:
        return _RHO_INV_TABLE[label]
    except KeyError as e:
        raise KeyError(f"_RHO_INV_TABLE missing entry for {label!r}.") from e


def _rho_n_apply(label: tuple, n: int) -> tuple:
    """ρ^n applied to a single label.  Period 5."""
    cur = label
    n = n % 5
    for _ in range(n):
        cur = _rho_apply(cur)
    return cur


# ---------------------------------------------------------------------------
# A1D5KAlg
# ---------------------------------------------------------------------------


class A1D5KAlg(ConeKAlgebra):
    """[A_1, D_5] K-algebra, SU(2)-flavoured, Z_5-symmetric, standalone.

    Atomic generating set: T_i, D_i, V_i, W_i for i ∈ Z/5 (= 4 ρ-orbits
    × 5 = 20 mult-gens).  No BPS oracle imported at any point.

    Labels are integer 5-tuples = canonical-basis g-vectors (last coord
    = χ-shift, ≤ 0 after `canonicalise`; magnitude = SU(2) χ-index).

    Inherits `ConeKAlgebra`: `multiply` is routed through
    `A1D5ConeData` (cone-data presentation on 4-coord lattice +
    χ-stripped, with χ-content threaded as an RLaurent[SU(2)]
    coefficient at the boundary).  This generalises the previous
    atomic-only multiply (which raised NotImplementedError on
    compound labels) to arbitrary cone monomials.
    """

    _RANK = 5
    _GAUGE_RANK = 4

    def __init__(self):
        self._R = SU2ZPlusRing()

    @property
    def rank(self) -> int:
        return self._RANK

    @property
    def gauge_rank(self) -> int:
        return self._GAUGE_RANK

    # -- generators -------------------------------------------------------

    def T(self, i: int) -> tuple:
        return _T_ORBIT[i % 5]

    def D(self, i: int) -> tuple:
        return _D_ORBIT[i % 5]

    def V(self, i: int) -> tuple:
        return _V_ORBIT[i % 5]

    def W(self, i: int) -> tuple:
        return _W_ORBIT[i % 5]

    def chi(self, k: int) -> tuple:
        if k < 0:
            raise ValueError(f"chi(k): k must be >= 0, got {k}")
        return (0, 0, 0, 0, -k)

    # -- KAlgebra primitives ---------------------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self) -> tuple:
        return _IDENTITY_LABEL

    def canonicalise(self, label) -> tuple:
        """SU(2)-Weyl canonical rep: flip last coord if > 0."""
        lab = tuple(int(x) for x in label)
        if lab[-1] > 0:
            return lab[:-1] + (-lab[-1],)
        return lab

    def cone_data(self):
        """Cone-data presentation on χ-stripped 5-tuple lattice (4-coord
        g-vector + last coord 0).  Lazy-built."""
        if not hasattr(self, "_cone_data_cache"):
            from a1d5_cone_data import A1D5ConeData
            self._cone_data_cache = A1D5ConeData()
        return self._cone_data_cache

    def _canonical_rho2_orbit_rep(self, label):
        """ρ² has order 5 on tiles; default orbit walk with safety."""
        return KAlgebra._canonical_rho2_orbit_rep(self, label)

    def _trace_residual(self, seed_label, K):
        """Layer 2 is not wired into this class: `[A_1, D_5]`'s traces are
        served by `A1DnKAlg(5)`, and by the zoo's `FiniteA1D5KAlgebra` from
        the closed-form sl(2)_{-8/5} seeds of `a1d5_layer2.py` (see the
        module docstring)."""
        raise NotImplementedError(
            "A1D5KAlg._trace_residual: Layer 2 is not wired into this class; "
            "for [A_1, D_5] traces use A1DnKAlg(5) (or the zoo's "
            "FiniteA1D5KAlgebra, whose closed-form sl(2)_{-8/5} traces are in "
            "a1d5_layer2.py)."
        )

    def multiply(self, a, b) -> Element:
        """Cone-data multiplication with χ-content folded at the
        5-tuple <-> (4-tuple + RLaurent) boundary.

        Algorithm (parallels A1D3KAlg.multiply):
          1. Canonicalise inputs; split each 5-tuple into a
             χ-stripped 5-tuple `(*lattice, 0)` and a χ-index
             `k = -last_coord`.
          2. Multiply the χ-stripped labels via cone_data on the
             atomic 20-mult-gen presentation, returning Element over
             RLaurent[SU(2)].
          3. Multiply the RLaurent coefficients by `χ_{k_a} · χ_{k_b}`
             (SU(2) Clebsch-Gordan via RElement.__mul__).
          4. Re-expand each (mono-label, RLaurent) into a sum of
             (5-tuple, LaurentPoly) terms: each SU(2) basis index
             k_out gives a 5-tuple `(*mono, -k_out)` with integer
             q-coefficient.
        """
        a = self.canonicalise(a)
        b = self.canonicalise(b)
        # Strip χ from each input (canonical: last coord ≤ 0).
        a_mono = a[:-1] + (0,)
        b_mono = b[:-1] + (0,)
        k_a = -a[-1]  # ≥ 0
        k_b = -b[-1]  # ≥ 0
        # Cone-data multiply on χ-stripped labels.
        cone_result = self.cone_data().derived_multiply(a_mono, b_mono)
        # χ_{k_a} · χ_{k_b} via SU(2) CG.
        chi_a = self._R.basis_element(k_a)
        chi_b = self._R.basis_element(k_b)
        chi_prod = chi_a * chi_b
        # Re-expand to 5-tuple labels.
        out: dict[tuple, LaurentPoly] = {}
        for lab_mono, coef in cone_result.terms.items():
            # Coerce to RLaurent.
            if isinstance(coef, LaurentPoly):
                rl = RLaurent(self._R, dict(coef._coeffs))
            else:
                rl = coef
            for q_exp, r_elt in rl.coeffs.items():
                scaled = r_elt * chi_prod
                if scaled.is_zero():
                    continue
                for k_out, coef_int in scaled.terms.items():
                    if coef_int == 0:
                        continue
                    lab5 = lab_mono[:-1] + (-k_out,)
                    lp_add = LaurentPoly({q_exp: int(coef_int)})
                    out[lab5] = out.get(lab5, LaurentPoly({})) + lp_add
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    def rho(self, a) -> tuple:
        return _rho_apply(self.canonicalise(a))

    def rho_inverse(self, a) -> tuple:
        return _rho_inverse_apply(self.canonicalise(a))

    def trace(self, a, K: int = 20, **kwargs):
        raise NotImplementedError(
            "A1D5KAlg.trace: not wired into this class; for [A_1, D_5] traces "
            "use A1DnKAlg(5) (or the zoo's FiniteA1D5KAlgebra, whose "
            "closed-form sl(2)_{-8/5} traces are in a1d5_layer2.py)."
        )

    def trace_layer1(self, label):
        raise NotImplementedError("A1D5KAlg.trace_layer1: TODO")

    def trace_layer2(self, K: int = 20):
        raise NotImplementedError("A1D5KAlg.trace_layer2: TODO")

    def r_label_decompose(self, label):
        """The single-irrep flavour-lift coordinate (replaces the retired
        `_label_section_decompose`).  Canonical γ has γ[-1] ≤ 0; the SU(2)
        spin `k = -γ[-1] ≥ 0` peels off as the R-basis-label, the section is
        γ with the last (w-fixed) coord zeroed."""
        a = self.canonicalise(label)
        return a[:-1] + (0,), -a[-1]

    def r_label_compose(self, section, r_basis_label):
        """Inverse of `r_label_decompose`: write the spin into the w-fixed
        (last) coord as `γ[-1] = -k`.  A direct slot write — no
        `embed_R`/`multiply` round-trip."""
        s = self.canonicalise(section)
        return self.canonicalise(s[:-1] + (-r_basis_label,))

    def embed_R(self, r: RElement) -> Element:
        """Central embedding `R(SU(2)) ↪ A_𝖖`: each character `χ_k` maps
        to the central canonical basis element `L_{(0,…,0,−k)}` (the
        pure spin-`k/2` character in the w-fixed direction), extended
        Z-linearly.  Inverse-compatible with `r_label_decompose`:
        `embed_R(χ_k) · L_{section} == L_γ` for `γ[-1] = −k` (χ_k is
        central).  (Still backs the default `from_R_form`; `r_label_compose`
        no longer needs it.)"""
        R = self.coefficient_ring()
        if not isinstance(r, RElement) or r.ring != R:
            raise TypeError(
                "embed_R: argument must be an RElement over coefficient_ring()"
            )
        rank = len(self.identity())
        out = Element.zero()
        for k, coeff in r.terms.items():
            if coeff == 0:
                continue
            central = self.canonicalise((0,) * (rank - 1) + (-k,))
            out = out + Element.basis(central) * coeff
        return out


    def atomic_kind(self, label) -> tuple | None:
        """If `label` is in the atomic mult-gen set, return (kind, i);
        else None."""
        return _LATTICE_TO_MULTGEN.get(self.canonicalise(label))

    def __repr__(self) -> str:
        return "A1D5KAlg()"


if __name__ == "__main__":
    A = A1D5KAlg()
    print(f"Constructed: {A}")
    print(f"Rank: {A.rank}, gauge_rank: {A.gauge_rank}")
    print()
    print("Atomic mult-gen orbit reps:")
    for X in 'TDVW':
        for i in range(5):
            print(f"  {X}_{i} = {getattr(A, X)(i)}")
    print(f"  identity = {A.identity()}")
    print(f"  chi(1)   = {A.chi(1)}")
    print()
    print("Sample multiplications:")
    T0, T4 = A.T(0), A.T(4)
    V0 = A.V(0)
    D0 = A.D(0)
    print(f"  T_0 * T_4 = {A.multiply(T0, T4)}")
    print(f"  T_0 * V_0 = {A.multiply(T0, V0)}")
    print(f"  D_0 * V_0 = {A.multiply(D0, V0)}  # was 'U_? in old basis'")
    print(f"  T_0 * W_0 = {A.multiply(T0, A.W(0))}")
    print(f"  identity * V_0 = {A.multiply(A.identity(), V0)}")
