"""``FiniteA1D3ZKAlgebra`` — Z-form (API-compliant) realisation of A₁D₃.

the design record: a flavoured `KAlgebra` must return **Z-valued**
`multiply` (`LaurentPoly`, flavour in labels).  The generated
`FiniteA1D3KAlgebra` instead returns `RLaurent` on bare *section* labels
(flavour in coefficients) — the non-compliant half-state.

This class is the **fully-free** worked example of the freeness
encoding: A₁D₃ is free over `R = R(SU(2))` (no torus / `R_lab` part).
Canonical labels are `(w, u)` = SU(2) irrep `w` × native cone section
`u`; the factorisation is `L_{(w,u)} = embed_R(χ_w)·M_u`.

**the design record, step 1 — decouple `RKAlgebra`.**  This used to subclass
`RKAlgebra` and inherit the free/label fusion.  As of the flavour
re-axiomatization it is a **standalone `KAlgebra`** with the fully-free
fusion **copied in** (no `R_lab` branch — `coefficient_ring()` is just
`R(SU(2))`), so `RKAlgebra` can be redesigned without constraining it.
The realisation data is still only the **section engine** (the native
cone cross-tables); `multiply` / `rho` / `_label_section_decompose` /
`embed_R` are the specialised fully-free copies.

**the design record (2026-09-26) — generalised.**  The fully-free fusion now
lives in `finite_su2_zform.FiniteSU2ZKAlgebra(native)`, the Z-form wrapper
of any SU(2)-flavoured standalone (a1d3, a1d5, a1d7), and this class is that
wrapper over `FiniteA1D3KAlgebra()` — same labels, products, ρ, lift
coordinate and central embedding as before.  One addition: `trace` no longer
raises; it is `Tr((w, u)) = χ_w · Tr_native(u)`, the native standalone's
trace (served through `a1d3_seeds`).

ρ on the centre is `⋆`, which fixes every SU(2) character (self-dual),
and `A1D3_RHO_DELTA` is empty, so section ρ is the native one.

Validated against the native R-form by isomorphism in
`the suite in the source repository`.
"""
from __future__ import annotations

from finite_a1d3_kalg import FiniteA1D3KAlgebra
from finite_su2_zform import FiniteSU2ZKAlgebra


class FiniteA1D3ZKAlgebra(FiniteSU2ZKAlgebra):
    """A₁D₃ as a standalone fully-free `KAlgebra` over `R(SU(2))`:
    `FiniteSU2ZKAlgebra` over `FiniteA1D3KAlgebra()`."""

    def __init__(self):
        super().__init__(FiniteA1D3KAlgebra())

    def __repr__(self) -> str:
        return "FiniteA1D3ZKAlgebra()"
