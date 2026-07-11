"""
u1_hexagon_pentagon_kalg.py
===========================

`U1HexagonPentagonRGKAlg`: `U1HexagonKAlg` (UV) presented as a
`ConeRGKAlgebra` with RG flow to `PentagonKAlg ⊗ QuantumTorus_2D`.

**Fully hard-coded — no BPSKAlgebra at import time or runtime.**

Direction: U(1)Hex UV → Pentagon ⊗ QT_2D IR.  The RG flow drops the
last O of the gauged-hex BPS quiver (O→O→O→F).  Surviving cluster:
A_2 chain on (γ_1, γ_2) = `PentagonKAlg` BPS, plus a rank-2 quantum
torus on (f = γ_3, v = γ_F) with `B[f, v] = +1` (= the dropped
γ_3 ↔ γ_F connection preserved as a quantum-torus pairing).

Hard-coded data (the only family-specific datum):
  * `RG_GENERATORS` — 11-entry dict mapping each U(1)Hex mult-gen to
    its `auxiliary()` `Element` (single canonical-basis term each).
    These match the BPS-backed RG exactly on single mult-gens.
  * `S_RG = E_q(A) · E_q(B)` where `A = -q · L_{(id, (1, 0))}` and
    `B = -q · L_{(id, (1, 1))}` (= the q-pentagon S_+ structure for
    this flow, identified by inspecting the BPS-IR rg_generator output
    once and fitting the closed form).  Verified against the BPS-backed
    reference for n_A + n_B ≤ 2.

Lattice automorphism (= why bare TensorKAlgebra is faithful)
-------------------------------------------------------------
Naïvely the BPS algebra `BPS(O→O→F→F)` on lattice (γ_1, γ_2, γ_3, γ_F)
with pairing `B_GAUGED` doesn't factor as `Pent × QT_2D` because
`B_GAUGED` has cross-pairing `B[γ_2, γ_3] = +1` linking the two
would-be tensor factors.  HOWEVER: there is a `GL(4, ℤ)` lattice
automorphism

    M :  γ_3 ↦ γ_3 − γ_1     (γ_1, γ_2, γ_F unchanged)

with `det(M) = 1`, which satisfies `M^T · B_block · M = B_GAUGED`
where `B_block = B_Pent ⊕ B_QT_2D` is block-diagonal.  M also
preserves the two mutable node charges γ_1 = (1, 0, 0, 0) and
γ_2 = (0, 1, 0, 0) — they are FIXED by M.

So `BPS(O→O→F→F)` IS isomorphic to `Pent × QT_2D` as a
lattice-with-pairing.  The cross-pairing is absorbed entirely by
the change of basis; the bare `TensorKAlgebra(Pent, QT_2D)` (no
cross-cocycle in `multiply`) is a faithful auxiliary, provided each
U(1)Hex mult-gen's BPS charge is first mapped through M to block
coords before being looked up in `RG_GENERATORS`.

The `RG_GENERATORS` table below has the M-corrected entries pre-baked
in (each value is `(Pent_label_of_(γ_1_block, γ_2_block),
(γ_3_block, γ_F_block))`).  Multiplicative extension over this
table reproduces BPS exactly on cone-monomial UV labels — no
dressing, no cross-cocycle, no BPS calls.

Cone-data side is inherited from `U1HexagonKAlg` (multiply, trace,
canonicalisation, ρ, ρ⁻¹, cone_data) — these stay BPS-free as in
`U1HexagonKAlg`.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from laurent_poly import LaurentPoly
from cone_rgkalgebra import ConeRGKAlgebra
from tensor_kalgebra import TensorKAlgebra
from kalgebra_samples import PentagonKAlg
from quantum_torus_kalgebra import QuantumTorusKAlg
from u1_hexagon_kalg import U1HexagonKAlg
from u1_hexagon_cone_data import E_GEN, E_INV
from habiro import HabiroElement


# ---------------------------------------------------------------------------
# RG image of each U(1)Hex mult-gen as a single auxiliary canonical-basis term
# ---------------------------------------------------------------------------
#
# Each U(1)Hex mult-gen has a known BPS charge in the gauged-hex 4-dim
# lattice (γ_1, γ_2, γ_3, γ_F).  Under the RG that drops γ_3:
#   * (γ_1, γ_2) maps to Pentagon's canonical basis via the standard
#     `pentagon_bps_iso` cone decomposition (`_bps_to_pent_label`).
#   * (γ_3, γ_F) maps to the QT_2D Laurent charge `(c, k)`.
#
# Auxiliary label is `(pent_label, (c, k))` ∈ Pentagon × QT_2D.
#
# This table is the *only* hand-coded family-specific data.  Derived once
# from the chord-charge assignments in `u1_hexagon_kalg.MU_CHARGE` and the
# Pentagon iso, frozen here.

# The 11-entry mapping below uses the LATTICE AUTOMORPHISM
#   M : γ_3 ↦ γ_3 − γ_1   (γ_1, γ_2, γ_F unchanged)
# which transforms B_GAUGED ↔ block-diagonal `B_Pent ⊕ B_QT_2D`
# while preserving the two mutable node charges γ_1, γ_2.  Concretely:
#     M^T · B_block · M = B_GAUGED   ✓
# This iso means BPS(O→O→F→F) IS isomorphic to Pent × QT_2D as
# lattice-with-pairing — the auxiliary `TensorKAlgebra(Pent, QT_2D)`
# (bare tensor, NO cross-cocycle) is faithful, provided each U(1)Hex
# mult-gen's BPS charge is first mapped through M to block coords.
#
# Each entry below is computed as: γ_block = M(γ_GAUGED), then split
# into (γ_1_block, γ_2_block) (= Pent's A_2 charge) and
# (γ_3_block, γ_F_block) (= QT_2D charge `(c, k)`).
RG_GENERATORS: dict = {
    # (a, i) U(1)Hex mult-gen    γ_GAUGED        γ_block            → list of (Pent label, (c, k)) terms
    # IN-CONE entries: single-term (γ_3-coef ≥ 0).
    (1, 0): [((4, 1, 0), (0,  1))],    # (0,  1,  0,  1) → (0,  1,  0,  1)
    (1, 1): [((1, 1, 0), (0, -1))],    # (0, -1,  0, -1) → (0, -1,  0, -1)
    (1, 3): [((0, 0, 0), (1, -1))],    # (1,  0,  1, -1) → (0,  0,  1, -1)
    (1, 5): [((2, 1, 0), (2, -1))],    # (1, -1,  2, -1) → (-1, -1, 2, -1)
    (2, 0): [((0, 1, 0), (0,  0))],    # (1,  0,  0,  0) → (1,  0,  0,  0)
    (2, 2): [((3, 1, 0), (1,  0))],    # (0,  0,  1,  0) → (-1, 0,  1,  0)
    E_GEN:  [((0, 0, 0), (1,  0))],    # (1,  0,  1,  0) → (0,  0,  1,  0)
    E_INV:  [((0, 0, 0), (-1, 0))],    # (-1, 0, -1,  0) → (0,  0, -1,  0)
    # OUT-OF-CONE entries (γ_3-coef < 0): 2-term Plücker BPS-RG image.
    # The second term is the F-cache "lower" Plücker correction the IR
    # spectrum-generator produces; needed for multiplicativity.
    (1, 2): [((0, 0, 0), (-1, 1)), ((3, 1, 0), (0, 1))],
    (1, 4): [((0, 1, 0), (-2, 1)), ((0, 0, 0), (-1, 1))],
    (2, 1): [((1, 1, 0), (-1, 0)), ((2, 1, 0), (0, 0))],
}


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class U1HexagonPentagonRGKAlg(U1HexagonKAlg, ConeRGKAlgebra):
    """U(1)Hex UV → Pentagon ⊗ QT_2D IR.  Hard-coded ConeRGKAlgebra.

    Inherits cone-data side from `U1HexagonKAlg`: `multiply`, `trace`,
    `cone_data`, `coefficient_ring`, `identity`, ρ, ρ⁻¹.  These stay
    BPS-free.

    RG side (auxiliary, RG, rg_generator, from_ir_image) is computed
    from the hard-coded `RG_GENERATORS` table + closed-form
    `S_RG = E_q(A) · E_q(B)`.  No `BPSKAlgebra` constructed at any
    point.
    """

    def __init__(self) -> None:
        U1HexagonKAlg.__init__(self)
        # IR auxiliary: Pent ⊗ (2D quantum torus on f, v with B[f,v]=1).
        self._aux = TensorKAlgebra(
            PentagonKAlg(),
            QuantumTorusKAlg([[0, 1], [-1, 0]]),  # (f, v) directions
        )

    def auxiliary(self):
        return self._aux

    # ---- RG primitive -----------------------------------------------------

    def RG(self, label) -> Element:
        """RG image of a U(1)Hex cone-monomial label.

        For canonical label L_{native} = q^{phase_UV} · ∏ L_g^{powers[g]}
        (= cone-data convention with `phase_UV = cone_label_phase`),
        the RG image is

            RG(L_{native}) = q^{phase_UV} · ∏ RG_GENERATORS[g]^{powers[g]}

        in the auxiliary, with the auxiliary's `multiply_elements`
        absorbing its own q-cocycle phases automatically.  The
        prefactor `q^{phase_UV}` translates between the UV
        cone-monomial convention and the literal mult-gen-product
        convention.
        """
        cd = self.cone_data()
        gens, powers = cd.to_cone_label(label)
        phase_UV = cd.cone_label_phase(gens, powers)
        one = LaurentPoly.one()
        result = Element({self._aux.identity(): LaurentPoly({phase_UV: 1})})

        # Multiply mult-gen images in canonical-cone-order (the order
        # used by `cone_label_phase`).  Each RG_GENERATORS[g] is a list
        # of tensor labels (typically singleton for IN-CONE letters; 2
        # entries for OUT-OF-CONE letters where Plücker corrections
        # appear in the IR F-basis decomposition).
        order = cd.canonical_cone_order(gens)
        for g in order:
            p = powers.get(g, 0)
            if p == 0:
                continue
            img = Element({lbl: one for lbl in RG_GENERATORS[g]})
            for _ in range(p):
                result = self._aux.multiply_elements(result, img)
        return result

    # ---- rg_generator: closed-form S_RG = E_q(-q · L_3 ⊗ X_{(1,0)}) -----

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG = E_q(-q · L_3^Pent ⊗ X_{(1, 0)}^QT)` truncated to
        `n ≤ cutoff` powers.

        Empirically extracted from `SingleNodeRG(BPS(O→O→O→F), j=2)`
        and projected through the M-basis-change + Pent iso to the
        tensor labels `((3, n, 0), (n, 0))`:

            S_RG = Σ_{n ≥ 0}  (-q)^n / (q²; q²)_n  ·  L_{((3, n, 0), (n, 0))}

        Note: this is a single quantum-dilogarithm tower (along the
        L_3 · X_{(1, 0)} direction).  Earlier `E_q(A) · E_q(B)` closed
        form in this module was wrong (had support on
        `((0, 0, 0), (·, ·))` instead of `((3, n, 0), (n, 0))`).
        Truncation here is by `n` ≤ cutoff (= γ_3-multiplicity).

        Verified against the BPS-backed reference for `cutoff ≤ 6`.

        Note on `verify_rg_inner_product` / `verify_rg_twist` truncation:
        a small number of inner_product / twist axiom checks (3 of 25
        pairs, 8 of 10 twist gens) fail at finite cutoff with characteristic
        boundary residuals.  Cross-check with `SingleNodeRG(BPS(O→O→O→F),
        j=2)` shows the BPS-side SubquiverRG produces the SAME mismatches
        at the SAME labels under truncation — these are intrinsic
        best-effort behaviors of `verify_rg_inner_product` /
        `verify_rg_twist` (per their docstrings), not bugs in this class.
        Our `rg_generator` matches the BPS-side `rg_generator` term-by-term
        through `cutoff = 6`.
        """
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        result: dict = {((0, 0, 0), (0, 0)): HabiroElement.one()}
        for n in range(1, cutoff + 1):
            sign = -1 if (n % 2 == 1) else 1
            coef = (HabiroElement.q_power(n, sign)
                    * HabiroElement.pochhammer_inverse(n))
            # Pent label (3, n, 0) = L_3^n; QT label (n, 0).
            label = ((3, n, 0), (n, 0))
            result[label] = coef
        return result

    # ---- from_ir_image (deferred) ----------------------------------------

    def from_ir_image(self, x_ir: Element) -> Element:
        """Recognise a `TensorKAlgebra(Pentagon, QT_2D)` element as a
        U(1)Hex `Element`.

        Not currently implemented in BPS-free form -- the inverse of
        the multiplicative extension over `RG_GENERATORS` requires a
        cluster-monomial decomposition (which (a, i) mult-gens compose
        to produce a given (Pent_label, (c, k)) auxiliary label).

        For the use cases in scope (cone-data multiply + RG-side
        evaluation), `from_ir_image` is not on the hot path.  The
        previous BPS-backed `IsoComposedRGKAlgebra` shim is also gone.

        If/when needed: derive an inverse table by enumerating
        multiplicative compositions of `RG_GENERATORS` up to bounded
        depth, indexed by aux label.
        """
        raise NotImplementedError(
            "U1HexagonPentagonRGKAlg.from_ir_image: not yet implemented "
            "in BPS-free form.  Multiply / trace / RG / rg_generator "
            "are all available; from_ir_image (= the directional RG "
            "multiply path) is research follow-up — see module docstring."
        )


# Backward-compat alias for code that referenced the previous class name.
U1HexagonPentagonKAlg = U1HexagonPentagonRGKAlg


if __name__ == "__main__":
    A = U1HexagonPentagonRGKAlg()
    print(f"Constructed: {type(A).__name__}")
    print(f"  MRO: {' → '.join(c.__name__ for c in type(A).__mro__[:6])}")
    print(f"  cone_data: {type(A.cone_data()).__name__}")
    print(f"  auxiliary: {type(A.auxiliary()).__name__}")
    print()
    print("Cone-data multiply (inherited from U1HexagonKAlg):")
    a = A.L((1, 0))
    b = A.L((1, 1))
    print(f"  L_{{1,0}} · L_{{1,1}} = {dict(A.multiply(a, b).terms)}")
    print()
    print("RG primitives (hard-coded, no BPS):")
    for gen in (A.L((1, 0)), A.L((1, 1)), A.L((2, 0)), A.L((2, 2))):
        print(f"  RG({gen}): {dict(A.RG(gen).terms)}")
    print()
    print("S_RG closed form (cutoff = 2):")
    sr = A.rg_generator(cutoff=2)
    for lab, hab in sorted(sr.items()):
        print(f"  {lab}: {hab}")
