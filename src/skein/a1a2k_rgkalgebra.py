"""
a1a2k_rgkalgebra.py
===================

`A1A2kRGKAlgebra(k)` — the even Argyres–Douglas chord algebra
`A_𝖖([A_1, A_{2k}])` as a **new-contract `RGKAlgebra`**: a
concrete RG flow supplies only `auxiliary()` + `grading()` +
`rg_generator()` (and the default identity `apex`), and the whole
`KAlgebra` API — `RG` (via `graded_rg_solver.solve_rg`), `multiply`,
`rho`/`rho_inverse`, `trace` — is derived generically.  No UV BPS
F-finder is ever invoked (that is the divergent operation this
construction exists to avoid).

The flow (drop the first two nodes γ₁, γ₂)
------------------------------------------
`[A_1, A_{2k}]` flows to `[A_1, A_{2k-2}]` by the RG that drops the
**first two** nodes of the linear `A_{2k}` BPS quiver `γ₁—γ₂—⋯—γ_{2k}`
(stuff = {γ₁, γ₂}: both dilogarithms live in `S_RG` — the two factors
below).  The chain `γ₃—⋯—γ_{2k} = A_{2k-2}` survives as the gauge
sector, and the `(γ₁,γ₂)` plane survives as a **bare** decoupled
symplectic pair:

    auxiliary  =  [A_1, A_{2k-2}]  ⊗  QT(Z₂)
               =  A1A2kKAlg(k-1)   ⊗  QuantumTorusKAlg([[0,1],[-1,0]])

*(Corrected 2026-06-10: this docstring previously described the flow as
the single-node drop of γ₂ alone.  That flow would leave behind
`[A_1, A_{2k-2}] × [A_1, A_1]` — the isolated γ₁ node keeping its
dilogarithm in `S_IR` — which is NOT the auxiliary used here; the code
has always used the bare-QT tensor above, i.e. the two-node drop.  The
single-node variant exists as an alternative factorization and is
machine-checked in `tests/test_a1a2k_honest_chamber.py`.)*

(`A1A2kKAlg(0) = Trivial`, so `A1A2kRGKAlgebra(1)` is the pentagon
`[A_1, A_2]` over `Trivial ⊗ QT(Z₂)`).  The Darboux symplectic change of
basis makes the bare tensor product (no cross-cocycle) a faithful
auxiliary: the qt direction γ₁ pairs only with `γ₂+γ₄+⋯+γ_{2k}`, which
telescopes to zero against every surviving node.

Grading (decisions: Γ_RG = Z²)
------------------------------
The grading lattice is just **Z²** — the QT `(γ₁,γ₂)` charge — with the
**obvious positive cone** `Z²_{≥0}` and height `(1,1)`.  The surviving
`A1A2kKAlg` factor sits entirely at grading-degree 0 (the "small" IR),
so the height only sees the QT charge.  Because the survivor is
degree-0, **the UV canonical labels are taken to be the auxiliary
labels themselves** `(chord, qt)` and `apex` is the identity — no
charge↔chord identification is needed anywhere (this UV labelling is
RG-flow-specific, reconciled to the intrinsic `A1A2kKAlg(k)` / `BPS`
labelling via the reconstruction iso).

Spectrum generator
------------------
    S_RG  =  E_𝖖(X_{γ₁}) · E_𝖖( X_{γ₂} · L )

— the **first-node** (γ₁, isolated ⟹ bare) and **second-node** (γ₂,
dressed) drop contributions.  The dressing `L` is, in `A1A2kKAlg(k-1)`
chord labels, the **short chord** `L((1, i₀))` at charge
`-(γ₄+γ₆+⋯+γ_{2k})` (verified by the second-node-drop RG: a simple
chord for all k; `i₀` is fixed by cyclic symmetry).  `S_RG` is sourced
exactly as `BPS(A_{2k}, spec=[γ₁,γ₂]).rg_generator` and relabeled charge
`(c₁,c₂,…) ↦ (L^{c₂}, (c₁,c₂))`; its support lies in the positive cone.

Status
------
* k=1: `multiply`/`rho`/`rho_inverse`/`trace` reproduce `BPS(A_2)`
  exactly.  k≥2: `verify_rg_unital` / `verify_rg_multiplicative` hold;
  full `≅ A1A2kKAlg(k)` is the reconstruction iso
  (`a1a2k_induction.a1a2k_bps_iso_inductive`).
* Prototype for the `a1_odd → a1_even` flow: a fast `[A_1, A_{2k+1}]`
  is the single-node (drop γ₁) RG over this cheap even IR.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from laurent_poly import LaurentPoly
from rgkalgebra import RGKAlgebra
from grading import Grading
from tensor_kalgebra import TensorKAlgebra
from quantum_torus_kalgebra import QuantumTorusKAlg
from kalgebra_samples import TrivialKAlg
from bps_kalgebra import BPSKAlgebra
from a1a2k_kalg import A1A2kKAlg
from a1a2k_bps_iso import _compute_chord_charges
from a1a2k_rgkalg import _a_n_pairing, _standard_basis

Vec = tuple


class A1A2kRGKAlgebra(RGKAlgebra):
    """`[A_1, A_{2k}]` as a new-contract `RGKAlgebra` over
    `A1A2kKAlg(k-1) ⊗ QT(Z₂)`, dropping the second node γ₂.

    UV canonical labels are the auxiliary labels `(chord, (c₁,c₂))`
    (`apex = identity`).  `RG`/`multiply`/`rho`/`trace` are the generic
    RGKAlgebra defaults driven by `grading()` (Γ_RG = Z²) and
    `rg_generator()` (`S_RG = E_𝖖(X_{γ₁})·E_𝖖(X_{γ₂}·L)`).
    """

    def __init__(self, k: int, *, rg_cutoff: int = 6) -> None:
        if k < 1:
            raise ValueError(f"A1A2kRGKAlgebra: need k ≥ 1, got {k}")
        self.k = k
        self._cut = rg_cutoff
        n = 2 * k

        # Surviving gauge sector [A_1, A_{2k-2}] (intrinsic chord algebra).
        self._survivor = TrivialKAlg() if k == 1 else A1A2kKAlg(k - 1)
        self._qt = QuantumTorusKAlg([[0, 1], [-1, 0]])
        self._aux = TensorKAlgebra(self._survivor, self._qt)
        self._survivor_id = self._survivor.identity()

        # S_RG source: the two-node sub-spectrum {γ₁, γ₂} on the full
        # A_{2k} lattice — E_𝖖(X_{γ₁})·E_𝖖(X_{γ₂}).  (Small spec ⟹ this
        # terminates; the divergent full-UV F-finder is never used.)
        nodes = _standard_basis(n)
        self._srg_bps = BPSKAlgebra(
            pairing=_a_n_pairing(n), node_charges=nodes,
            spec=[nodes[0], nodes[1]], verify="off",
        )

        # Dressing short chord position i₀ in A1A2kKAlg(k-1): the chord
        # whose charge is the dressing direction -(γ₄+γ₆+⋯+γ_{2k}) =
        # -(even surviving nodes).
        if k >= 2:
            cc = _compute_chord_charges(k - 1, {a: 0 for a in range(1, k)})
            m = 2 * k - 2
            dress = tuple(-1 if (j % 2 == 1) else 0 for j in range(m))
            try:
                self._i0 = next(
                    i for (a, i), ch in cc.items()
                    if a == 1 and tuple(ch) == dress
                )
            except StopIteration:                  # pragma: no cover
                raise RuntimeError(
                    f"A1A2kRGKAlgebra(k={k}): no short chord with dressing "
                    f"charge {dress} in A1A2kKAlg({k-1})."
                )
        else:
            self._i0 = None

    # ----- new-contract defining data -------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z² is the QT (γ₁,γ₂) charge; the survivor factor is
        # degree-0.  Obvious positive cone Z²_{≥0} (gens (1,0),(0,1)),
        # height (1,1).  The cone lets solve_rg use the single-cutoff
        # cone-filtered completion certificate (off-cone ρ-artifacts dropped).
        return Grading(rank=2, deg=lambda label: tuple(label[1]),
                       height=(1, 1), cone_gens=((1, 0), (0, 1)))

    def _rg_cutoff(self) -> int:
        return self._cut

    def _dressing_chord(self, b: int) -> Vec:
        """The auxiliary survivor-label `L^b` for the γ₂-tower power `b`:
        the short chord to power `b` (identity for `b = 0` or `k = 1`)."""
        if self.k == 1 or b == 0:
            return self._survivor_id
        return ((1, self._i0 % self._survivor.H, b),)

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG = E_𝖖(X_{γ₁})·E_𝖖(X_{γ₂}·L)` windowed to **q-order ≤ cutoff**,
        keyed by auxiliary labels.

        Sourced from `BPS(A_{2k}, spec=[γ₁,γ₂]).rg_generator` (q-order window)
        and relabeled charge `(c₁,c₂,…) ↦ (L^{c₂}, (c₁,c₂))`; `[S_RG]_0` is
        the auxiliary identity `(1_survivor, (0,0))`."""
        out = {}
        for chg, h in self._srg_bps.rg_generator(cutoff).items():
            c1, c2 = chg[0], chg[1]
            out[(self._dressing_chord(c2), (c1, c2))] = h
        return out

    def _s_rg_component(self, p):
        """Exact `Γ_RG`-graded component `[S_RG]_p`, `p = (c₁, c₂)` the
        `(γ₁,γ₂)` multiplicity (RGKAlgebra contract).

        The `(γ₁,γ₂)` quantum torus is `deg = id`, so `[S_RG]_p` is the single
        relabeled BPS component at charge `c₁γ₁ + c₂γ₂`: exact Nahm sum, finite,
        `{}` off the cone (`c₁<0` or `c₂<0`).  This is the cutoff-free
        "knowing `S_RG`" oracle (grading-height window = `∪_{c₁+c₂ ≤ K}`)."""
        c1, c2 = int(p[0]), int(p[1])
        if c1 < 0 or c2 < 0:
            return {}
        chg = tuple([c1, c2] + [0] * (2 * self.k - 2))
        comp = self._srg_bps._s_rg_component(chg)
        if not comp:
            return {}
        return {(self._dressing_chord(c2), (c1, c2)): comp[chg]}

    # ----- introspection --------------------------------------------------

    @property
    def survivor(self):
        """The `[A_1, A_{2k-2}]` auxiliary factor (`A1A2kKAlg(k-1)`)."""
        return self._survivor

    @property
    def dressing_chord_index(self):
        """The cyclic index `i₀` of the short dressing chord `L((1,i₀))`
        (None at k=1, where the dressing is trivial)."""
        return self._i0


if __name__ == "__main__":
    A = A1A2kRGKAlgebra(2)
    print(f"A1A2kRGKAlgebra(k=2): aux = "
          f"{type(A.auxiliary().factor_A).__name__} ⊗ "
          f"{type(A.auxiliary().factor_B).__name__}; "
          f"dressing chord L((1,{A.dressing_chord_index}))")
    sid = A.survivor.identity()
    g1 = (sid, (1, 0))            # γ₁
    g3 = A.survivor.L((1, 0))     # a survivor chord
    print("RG(γ₁) =", {k: str(v) for k, v in A.RG(g1).terms.items()})
    print("multiply(γ₁, survivor-chord) =",
          dict(A.multiply(g1, (g3, (0, 0))).terms))
    print("verify_rg_multiplicative:",
          A.verify_rg_multiplicative(g1, (g3, (0, 0))))
