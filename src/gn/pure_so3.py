"""`PureSO3KAlgebra` — pure SO(3)=PSU(2), the GNO/Langlands dual of pure SU(2).

**A BPS quiver does not by itself name a theory.**  It names a quiver; the theory
is that quiver *plus a choice of how its node charges embed in a charge lattice*
`Γ`, and that choice is the 4d gauge group.  SU(2) and SO(3) are the **same**
Kronecker-2 quiver — `⟨n₀, n₁⟩ = 2` for both, so the pairing distinguishes
nothing — on the **same** canonical `Z²` with unit symplectic form.  All that
differs is where the nodes sit in it:

    SU(2)   nodes (1, 0), (−1, 2)      Γ = P^∨ ⊕ P   (simply connected)
    SO(3)   nodes (2, 0), (−2, 1)      Γ = Q^∨ ⊕ Q   (adjoint)

The node charges of a pure ADE theory live in `Q^∨ ⊕ Q`; the global form is which
unimodular `Λ₀ = P^∨ ⊕ Q ⊆ Γ ⊆ Q^∨ ⊕ P` you embed them in (classified by the
Lagrangian subgroups of `(P/Q)²`).  So this class does **not** hand-build a
lattice — it asks `pure_ade_lattice` for pure `A₁` at the adjoint form, which is
where that classification lives.  Asking the same function for `"sc"` returns the
SU(2) chart.

Writing both forms in one set of coordinates gives the familiar

    (M, E)_SO(3)  =  (2 m, e/2)_SU(2),        (m, e)_SU(2) = (M/2, 2E)_SO(3),

but that is **a description of how the two lattices sit inside each other, not a
rescaling of one into the other** (user correction, 2026-07-29; ruling D22).
Nothing is rescaled and the quiver is untouched.  What the map does show is the
genuinely-new **spinorial** sector of SO(3) at *odd* `M` (no SU(2) preimage — its
`(M/2, 2E)` is half-integer).  The minimal lines:

    H_0   = F_{(1,0)}   minimal 't Hooft  (spinorial coweight ω^∨; magnetic ½ in
                        SU(2) units — the S-dual of SU(2)'s minimal Wilson w_1),
    w_2   = F_{(0,1)}   minimal Wilson    (adjoint; the root lattice — SO(3) has
                        NO w_1, whose electric (0,1)→(0,½) is fractional),
    L_{1,0}= F_{(2,0)} = H_0²             the SU(2) adjoint monopole (even M).

**Why the BPS chart — and why no fractional powers of 𝖖.**  On this chart the
**quiver nodes are the charge basis** and the coweight-torus atom phase never
enters — which is the whole reason `H_0` presents here.  Under the label map
above this is `= SU(2)-BPS(nodes (1,0),(−1,2))` — an
**exact algebra iso** (multiply structure constants AND traces transport verbatim,
𝖖-powers and all; certified in the suite in the source repository).  Everything is over
`Z[𝖖^±]`; `H_0=F_{(1,0)}` is orthonormal (`I=1−𝖖²+𝖖⁴+…`) and `H_0²=L_{1,0}`.

**The torus/atom (AbeKAlgebra) route DOES hold the spinorial line** — since ruling
D31 (2026-07-29).  It used not to: `ω^∨` has `⟨Σ⁺, ω^∨⟩ = 1` (odd), so the
*materialised* atom monomial `M(m) ∝ (−𝖖)^{⟨ρ,m⟩}` — the square root of the
measure — wants a `𝖖^{1/2}` and an `i` there, and the `ε` stand-in that was
adopted instead corrupted the cocycle and broke `bar` on `H_0²`.  But the algebra
never needs that monomial: `wrq_torus.cocycle_R` now restores the honest phase
through the INTEGRAL coboundary `(−𝖖)^{δ(S_honest − S_used)}`, and
`PureGAbeKAlgebra(so_n(3))` builds `H_0` bar-invariantly with
`H_0² = L_{(2,0)}` and `I(H_0,H_0)` equal to this file's value term by term.
**This realisation is the ORACLE that certified that fix**, which is its role
now — an independent BPS presentation of the same algebra, not the only one.
See a probe in the source repository, ruling D31, and the now
historical a probe in the source repository.

The even-`M` (shared) sector coincides with the SU(2) **Abe** presentation
`PureSU2KAlgebra` on the *pure-magnetic* lines (`I(F_{(2m,0)})_SO3 = I(L_{m,0})_SU2`);
dyonic lines differ only by the usual BPS-canonical vs Wilson-character basis
choice already present in SU(2), not by anything SO(3)-specific.

    A = PureSO3KAlgebra()
    A.multiply(A.H0, A.H0)          # = F_{(2,0)} = L_{1,0}
    A.inner_product(A.H0, A.H0, 6)  # 1 − 𝖖² + …  (orthonormal)
    A.so3_to_su2((2, 0))            # (1, 0)  — the SU(2) preimage (even M)
    A.so3_to_su2((1, 0))            # None    — spinorial, no SU(2) preimage
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bps_kalgebra import BPSKAlgebra
from pure_ade_lattice import pure_ade_lattice_data

__all__ = ["PureSO3KAlgebra", "so3_to_su2", "su2_to_so3"]

# Pure `A₁` at the **adjoint** global form, `Γ = Q^∨ ⊕ Q`.  Derived rather than
# hand-written: `pure_ade_lattice` owns the classification of the valid Γ, so the
# global form is a parameter here and not two magic node vectors.  The same call
# with `global_form="sc"` is SU(2), nodes (1,0),(−1,2) — same pairing, same quiver.
_A1_ADJOINT = pure_ade_lattice_data([("A", 1)], global_form="adj")
SO3_PAIRING = [list(row) for row in _A1_ADJOINT["B"]]
SO3_NODES = [tuple(n) for n in _A1_ADJOINT["nodes"]]


def so3_to_su2(charge):
    """`(M, E)_SO(3) → (m, e)_SU(2) = (M/2, 2E)`, or `None` for the spinorial
    (odd-`M`) sector, which has no SU(2) preimage."""
    M, E = charge
    if M % 2 != 0:
        return None
    return (M // 2, 2 * E)


def su2_to_so3(charge):
    """`(m, e)_SU(2) → (M, E)_SO(3) = (2 m, e/2)`, or `None` for odd `e` (the
    SU(2) half-integer-spin sector, e.g. the fundamental Wilson w_1, which is not
    an SO(3) line)."""
    m, e = charge
    if e % 2 != 0:
        return None
    return (2 * m, e // 2)


class PureSO3KAlgebra(BPSKAlgebra):
    """Pure SO(3)=PSU(2): the pure `A₁` quiver at the **adjoint** global form
    (`Γ = Q^∨ ⊕ Q`; nodes `(2,0),(−2,1)` on the canonical `Z²`).  The *simply
    connected* embedding of that same quiver is SU(2).  Charges `(M, E)` =
    (magnetic, electric).  A complete `KAlgebra` (BPS realisation) — `multiply` /
    `ρ` / `trace` / `inner_product` / verifiers all inherited; over `Z[𝖖^±]`, no
    fractional powers."""

    #: named minimal lines (standard-Z² charges)
    H0 = (1, 0)      # minimal 't Hooft (spinorial ω^∨) — the S-dual of SU(2) w_1
    w2 = (0, 1)      # minimal Wilson (adjoint; SO(3) has no w_1)
    L10 = (2, 0)     # SU(2) adjoint monopole = H_0²

    def __init__(self, **kwargs):
        super().__init__(pairing=SO3_PAIRING, node_charges=SO3_NODES, **kwargs)

    # ---- the SU(2) bridge (the (2m, e/2) global-form relabeling) -------------
    @staticmethod
    def so3_to_su2(charge):
        return so3_to_su2(charge)

    @staticmethod
    def su2_to_so3(charge):
        return su2_to_so3(charge)

    def __repr__(self):
        return "PureSO3KAlgebra(A_1 at the adjoint global form; nodes (2,0),(-2,1))"


if __name__ == "__main__":
    A = PureSO3KAlgebra()
    print(A)
    print("  H_0² =", {g: str(c) for g, c in A.multiply(A.H0, A.H0).terms.items()},
          "(= L_{1,0} = F_(2,0))")
    print("  I(H_0,H_0) =", A.inner_product(A.H0, A.H0, 6))
    print("  I(w_2,w_2) =", A.inner_product(A.w2, A.w2, 6))
    print("  so3_to_su2((2,0)) =", A.so3_to_su2((2, 0)),
          "  so3_to_su2((1,0)) =", A.so3_to_su2((1, 0)), "(spinorial: no SU(2) preimage)")
