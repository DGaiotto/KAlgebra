"""
u1a1deven_rgkalgebra.py
=======================

`U1A1DevenRGKAlgebra(k)` — the **U(1)-gauged** presentation of `[A_1,
D_{2k+2}]`: the even D-type matter dressing of `A1A2k(k)` with the U(2) flavour
of `A1DevenRGKAlgebra` split as `U(2) ⊃ U(1) × SU(2)` and the U(1) part
*gauged* into a `QT(Z_2)` symplectic factor, leaving `SU(2)` as flavour.  Built
to be a clean **cone oracle**: the gauged U(1) charge is an honest quantum-torus
generator `X_{0,1}` (a cone-chord direction), not a flavour coefficient — which
matches the existing D-quiver cone presentations (`A1D3`/`A1D5`/`U1A1D4`, all
SU(2)-flavoured).

  * IR auxiliary  =  A1A2k(k) ⊗ QT(Z_2) ⊗ SU(2)-flavour
        `TensorKAlgebra(A1A2kKAlg(k), QuantumTorusKAlg([[0,1],[-1,0]]))
             .add_flavour(SU2ZPlusRing())`
    labels `((chord, (c₁, c₂)), κ)` — `chord` the A1A2k gauge chord, `(c₁,c₂)`
    the QT(Z_2) charge, `κ` the SU(2) χ-index.
  * S_RG  =  E_𝖖(μ · X_{0,1} · L) · E_𝖖(μ⁻¹ · X_{0,1} · L)
    the two doublet weights `μ^{±1}` of the matter coupling to `X_{0,1}·L`
    (`X_{0,1}` = QT charge `(0,1)`, `L` = the short chord `L((1, H−2))`,
    `H = 2k+3`).  Collinear (both in `X_{0,1}·L`), so

        S_RG = Σ_{n₁,n₂≥0} a_{n₁} a_{n₂} μ^{n₁−n₂} (X_{0,1} L)^{n₁+n₂},
        a_n = (−q)^n/(q²;q²)_n,

    and the μ-polynomial at total order `N = n₁+n₂` Weyl-folds into SU(2)
    characters: irrep `κ = a−b` (`a+b=N`, `a≥b≥0`) with coefficient
    `c_a c_b − c_{a+1} c_{b-1}` (the GL(2) Schur peel; same coefficients as the
    U(2) version, now read as SU(2) χ-content with the U(1) charge carried by
    `X_{0,1}^N`).  `(X_{0,1} L)^N = (L^N, (0,N))`.
  * `Γ_RG = Z` = the `X_{0,1}` charge `c₂` (the gauged-U(1) / matter level);
    the survivor (`c₂ = 0`) keeps the A1A2k gauge, the QT `X_{1,0}` direction,
    and the SU(2) flavour.  apex = identity.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from habiro import HabiroElement
from kalgebra import Element
from rgkalgebra import RGKAlgebra
from grading import Grading
from zplus_ring import SU2ZPlusRing
from tensor_kalgebra import TensorKAlgebra
from quantum_torus_kalgebra import QuantumTorusKAlg
from a1a2k_kalg import A1A2kKAlg
from a1aodd_to_even_rgkalgebra import _e_q_coeff          # a_n = (−q)^n/(q²;q²)_n
from a1deven_rgkalgebra import _u2_char_coeff     # c_a c_b − c_{a+1}c_{b-1}


class U1A1DevenRGKAlgebra(RGKAlgebra):
    """`[A_1, D_{2k+2}]` with the U(1) flavour gauged to a `QT(Z_2)` factor and
    SU(2) flavour; `S_RG = E_𝖖(μ X_{0,1} L) E_𝖖(μ⁻¹ X_{0,1} L)`.  See the module
    docstring."""

    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._gauge = A1A2kKAlg(k)
        self._H = self._gauge.H
        self._i0 = self._H - 2                                # short-chord index
        self._tensor = TensorKAlgebra(
            self._gauge, QuantumTorusKAlg([[0, 1], [-1, 0]]))
        self._aux = self._tensor.add_flavour(SU2ZPlusRing())

    # ----- RGKAlgebra contract -------------------------------------------

    def coefficient_ring(self):
        return self._aux.coefficient_ring()

    def identity(self):
        return self._aux.identity()

    def auxiliary(self):
        return self._aux

    def _label_section_decompose(self, label):
        return self._aux._label_section_decompose(label)

    def grading(self) -> Grading:
        # Γ_RG = Z = X_{0,1} charge c₂ = label[0][1][1]; cone Z_{>=0}, height 1.
        return Grading(rank=1, deg=lambda lab: (lab[0][1][1],), height=(1,),
                       cone_gens=((1,),))

    def apex(self, a):
        """Identity apex (the survivor sits at grading-degree 0)."""
        return a

    def _matter_label(self, N: int, kappa: int):
        """`(X_{0,1} L)^N ⊗ χ_κ` = `((L^N, (0,N)), κ)` (`L^N = ()` for N=0)."""
        chord = () if N == 0 else ((1, self._i0, N),)
        return ((chord, (0, N)), kappa)

    def _s_rg_component(self, p):
        """`[S_RG]_{(N,)}` — exact, finite, `{}` off the cone (`N<0`).  At level
        `N` the SU(2) irreps `κ = N, N−2, …, 0/1` with Schur coefficient
        `c_a c_b − c_{a+1} c_{b-1}` (`a=(N+κ)/2`, `b=(N−κ)/2`)."""
        (N,) = p
        if N < 0:
            return {}
        out: dict = {}
        for kappa in range(N, -1, -2):
            a = (N + kappa) // 2
            b = (N - kappa) // 2
            coeff = _u2_char_coeff(a, b)
            if coeff.is_zero():
                continue
            out[self._matter_label(N, kappa)] = coeff
        return out

    def rg_generator(self, cutoff: int) -> dict:
        out: dict = {}
        for N in range(cutoff):
            out.update(self._s_rg_component((N,)))
        return out

    def trace(self, a, K: int = 20):
        """Fast `Tr_UV(L_a) = Tr_aux(ρ(S_RG)·RG(a)·S_RG)`.

        The auxiliary is `A1A2k ⊗ QT(Z²) ⊗ SU(2)`, whose trace carries the
        quantum-torus `δ_{(c1,c2),0}` — so only the **QT-neutral** terms of the
        FS object survive.  Filtering to those *before* the (otherwise dominant)
        `aux.trace_element` is a ~12× speedup over the generic RGKAlgebra trace.

        **Adaptive on BOTH axes (the audit).**  The inner loop grows the
        q-order window with two-window stability; the outer loop grows the
        **matter-level cutoff** `cut` (the number of `S_RG` levels `N`) with
        **two-cutoff stability**.  Matter level `N` first contributes at
        `q_paper^N = q_d^{2N}`, so a *fixed* `cut` (the old `_rg_cutoff()=12`)
        silently truncates the matter tower past `q_d^{2·cut}` — caught against
        the deven closed form / `sl(3)` KW character at `q_d²⁴`.  Growing `cut`
        until two successive cutoffs agree (the A10 idiom on the level axis)
        removes the truncation.  Memoised.
        """
        cache = self.__dict__.setdefault("_fast_trace_cache", {})
        ck = (a, K)
        if ck in cache:
            return cache[ck]
        aux = self.auxiliary()
        rg_a = self.RG(a)

        def _trace_at_cut(cut):
            prev = res = None
            # Start the q-order window SMALL and grow under two-window stability.
            # The order-K trace needs S_RG only to order ~K (low matter levels
            # dominate low q), so the old `win = max(cut, 2)` start over-built to
            # order ~K+cut — ~3.6× the cost per build at cut=12 (verified: ray 4
            # K=8 converges identically at win=2,4,6,8).  The cap stays generous.
            win = 2
            win_max = 2 * K + max(cut, 2) + 10
            while win <= win_max:
                s_rg = self._s_rg_as_aux_element(cut, K + win)
                rho_s = aux.rho_element(s_rg)
                inner = aux.multiply_elements(rg_a, s_rg)
                # QT-neutral EARLY filter: the trace keeps only QT-charge-(0,0)
                # terms of ρ(S_RG)·inner, and the aux multiply is QT-charge
                # ADDITIVE (the QT(Z²) tensor factor: U_a·U_b ∝ U_{a+b}), so only
                # term pairs with complementary charge contribute a neutral term.
                # Multiplying just those — rather than the full O(n²) product then
                # filtering — is the dominant high-cutoff speedup for k≥2 (the
                # S_RG charge classes are the n/C reduction).  Correctness is
                # pinned by the freeze regression (values unchanged).
                by_charge: dict = {}
                for b, cb in inner.terms.items():
                    by_charge.setdefault(b[0][1], []).append((b, cb))
                nt: dict = {}
                for a, ca in rho_s.terms.items():
                    aq = a[0][1]
                    for b, cb in by_charge.get((-aq[0], -aq[1]), ()):
                        scalar = ca * cb
                        if scalar.is_zero():
                            continue
                        for lbl, c in (aux.multiply(a, b) * scalar).terms.items():
                            nt[lbl] = nt[lbl] + c if lbl in nt else c
                neutral = Element({l: c for l, c in nt.items() if not c.is_zero()})
                tr = aux.trace_element(neutral, K)
                if prev is not None and tr.coeffs == prev.coeffs:
                    return tr
                prev = tr
                win += 2
            return prev

        # outer two-cutoff-stability loop on the matter-level axis.  Start with a
        # K-aware floor (level N contributes from q_d^{2N}, so order K needs
        # cut ≳ K//2) and grow by 2 until the q^K result is stable.
        cut = max(self._rg_cutoff(), K // 2 + 2)
        prev = _trace_at_cut(cut)
        cut_cap = K + self._rg_cutoff() + 6
        while cut < cut_cap:
            cut += 2
            cur = _trace_at_cut(cut)
            if cur.coeffs == prev.coeffs:
                prev = cur
                break
            prev = cur
        cache[ck] = prev
        return prev

    def _section_split(self, label):
        # labels are nested tuples ((chord,(c1,c2)),κ); disable the flat-vector
        # flavour-shift cache, fall back to direct from_ir_image multiply.
        return tuple(label), None


if __name__ == "__main__":
    A = U1A1DevenRGKAlgebra(1)
    print(f"U1A1DevenRGKAlgebra(k=1)  [U(1)-gauged [A_1,D_4], QT(Z2)⊗SU(2)]")
    print(f"  aux coefficient ring = {A.coefficient_ring()}")
    print("  S_RG = E(μ X_{0,1} L) E(μ⁻¹ X_{0,1} L), levels ≤ 2:")
    for N in range(3):
        for lab, c in A._s_rg_component((N,)).items():
            print(f"    N={N}  {lab}:  {c}")
    # RG of a survivor: the bare short chord at c2=0.
    g = ((A._gauge.L((1, 0)), (0, 0)), 0)
    rg = A.RG(g)
    print(f"  RG(short chord) has {len(rg.terms)} terms:")
    for lab, c in sorted(rg.terms.items(), key=lambda t: t[0][0][1][1]):
        print(f"    {lab}:  {c}")
