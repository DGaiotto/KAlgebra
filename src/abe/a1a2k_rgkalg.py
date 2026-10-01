"""
a1a2k_rgkalg.py
===============

⚠️  SUPERSEDED by `a1a2k_induction.py`.  This module's `RG` is the
*Darboux relabel* (the tropical/intrinsic label map), NOT the genuine
RG homomorphism `F_a`; consequently `A1A2kRGKAlg.multiply` drops the
S_RG-mediated terms and is a self-consistent but **wrong** algebra (it
is not `[A_1, A_{2k}]`).  It also uses the wrong spec (positive roots,
not the simple-root negating sequence).  The correct, verified
construction — `A1A2kKAlg(k) ≅ BPS(A_{2k})` via the RGKAlgebra
reconstruction theorem with the simple-root negating-sequence spec —
lives in `a1a2k_induction.py` (`a1a2k_bps_iso_inductive`,
`verify_a1a2k_induction_step`, `prove_a1a2k_bps_iso`).  Kept only as a
record of the (instructive) wrong turn.

`A1A2kRGKAlg(k)`: the even Argyres–Douglas chord algebra
`A_𝖖([A_1, A_{2k}])` presented as an `RGKAlgebra` whose auxiliary is
the tensor product

    auxiliary()  =  TensorKAlgebra( A1A2kRGKAlg(k-1),  QT(Z_2) )
                 =  [A_1, A_{2k-2}]  ⊗  QuantumTorusKAlg([[0,1],[-1,0]])

i.e. the construction is **recursive**: the `[A_1, A_{2k-2}]` factor
is itself an `A1A2kRGKAlg`, so the tower descends k → k-1 → … → 1,
bottoming out at `[A_1, A_0] = Trivial` (so `A1A2kRGKAlg(1)` =
pentagon over `Trivial ⊗ QT(Z_2)`).  Every multiply is therefore
BPS-F-free all the way down, and generic-k is cheap (k=4
RG-multiplicativity over all 1296 positive-root pairs runs in ~0.3 s).

It is built via the **single-node RG that drops the second node γ_2**
of the linear A_{2k} BPS quiver (`γ_1 — γ_2 — … — γ_{2k}`):

  * dropping γ_2 disconnects γ_1 from the chain, leaving
    `γ_3 — … — γ_{2k} = A_{2k-2}` as the surviving gauge sector
    (`= [A_1, A_{2k-2}]`) and `(γ_1, γ_2)` as a decoupled symplectic
    pair (`= QT(Z_2)`).

RG map and RG generator (direct, BPS-F-free)
--------------------------------------------
The cross-pairing `⟨γ_2, γ_3⟩ = 1` is absorbed by a Darboux
(symplectic) change of basis `M_k`; the **bare** `TensorKAlgebra`
(no cross-cocycle) is then a faithful auxiliary — verified
term-by-term against the IR `BPSKAlgebra` multiply.

  * `RG(L_a)` is the **Darboux relabel** of the UV charge `a` into
    the auxiliary canonical basis (the "simple IR image").  No UV
    F-element is computed — the integrated-out dynamics live entirely
    in the RG generator.
  * `S_RG = E_𝖖(F_{γ_1}) = E_𝖖(X_{γ_1})`.  Since γ_1 is isolated in
    the IR its F-element is the bare monomial `X_{γ_1}`, so `S_RG` is
    the single isolated-node quantum dilogarithm, living on the
    QT(Z_2) factor at charge `(n, 0)`.  For BPS the canonical-basis
    label *is* the lattice charge and `A1A2kRGKAlg`'s UV labels are
    A_{2k} charges, so the Darboux block-charge is used directly as
    the recursive `A1A2kRGKAlg(k-1)` label — no further iso needed.

Darboux change of basis `M_k` (drop γ_2)
----------------------------------------
For an IR/UV charge `c = (c_1, …, c_{2k})` (in the A_{2k} node basis):

    A_{2k-2} block charge :  for i = 3 … 2k,  coord_i = c_i − c_2·[i even]
    QT(Z_2) charge        :  (c_1, c_2)

The qt direction `γ_1` pairs (`⟨·,·⟩ = 1`) only with
`q1 = γ_2 + γ_4 + γ_6 + … + γ_{2k}`, which telescopes to zero pairing
against every surviving A_{2k-2} node — hence the clean symplectic
split.  The block charge is used directly as the recursive
`A1A2kRGKAlg(k-1)` label (whose UV labels are A_{2k-2} charges).

Base case
---------
`k = 1` ( pentagon `[A_1, A_2]` ): the surviving sector is
`[A_1, A_0] = ` trivial, so the auxiliary is `Trivial ⊗ QT(Z_2)`,
i.e. pentagon ≅ RGKAlgebra over the rank-2 quantum torus with
`S_RG = E_𝖖(X_{γ_1})`.

Status
------
* `RG`, `rg_generator`, `multiply`, `from_ir_image` are all direct
  (no UV F-finder, no hang) and **cheap for generic k** via the
  recursive auxiliary.  Validated by `verify_rg_multiplicative`
  (all positive-root pairs, k = 1..4), `verify_rg_unital`,
  `verify_rg_twist`, and associativity (see
  `tests/test_a1a2k_rgkalg.py`).
* `trace` / `inner_product` delegate to `inner`
  (`SingleNodeRGKAlgebra`); the fast auxiliary-side Schur-transport
  trace (`I^UV_{a,b} = ⟨RG_a·S_RG, RG_b·S_RG⟩_aux`) is the next step
  that makes generic-k iso verification cheap.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from laurent_poly import LaurentPoly
from kalgebra_iso import KAlgebraIso
from iso_composed_rgkalgebra import IsoComposedRGKAlgebra
from single_node_rgkalgebra import SingleNodeRGKAlgebra
from tensor_kalgebra import TensorKAlgebra
from quantum_torus_kalgebra import QuantumTorusKAlg


# ---------------------------------------------------------------------------
# A_n linear-quiver data
# ---------------------------------------------------------------------------

def _a_n_pairing(n: int) -> list[list[int]]:
    """Antisymmetric A_n linear-chain pairing on Z^n: B[i, i+1] = +1."""
    B = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        B[i][i + 1] = 1
        B[i + 1][i] = -1
    return B


def _standard_basis(n: int) -> list[tuple[int, ...]]:
    return [tuple(1 if j == i else 0 for j in range(n)) for i in range(n)]


def _a_n_positive_roots(n: int) -> list[tuple[int, ...]]:
    """Positive roots of A_n = contiguous intervals [i, j], as charges."""
    roots: list[tuple[int, ...]] = []
    for i in range(n):
        for j in range(i, n):
            roots.append(tuple(1 if i <= t <= j else 0 for t in range(n)))
    return roots


def _spec_drop_second(n: int) -> list[tuple[int, ...]]:
    """A_n positive-root spec ordered 'stuff first': all roots with
    γ_2 (index 1) coefficient > 0 before all roots with it 0 — the
    ordering `SingleNodeRGKAlgebra` requires for `gamma_drop = γ_2`."""
    roots = _a_n_positive_roots(n)
    head = [r for r in roots if r[1] > 0]
    tail = [r for r in roots if r[1] == 0]
    return head + tail


# ---------------------------------------------------------------------------
# Darboux split (drop γ_2): A_{2k} charge ↔ (A_{2k-2} charge, QT charge)
# ---------------------------------------------------------------------------

def _darboux_split(k: int, c) -> tuple[tuple[int, ...], tuple[int, int]]:
    """`c = (c_1, …, c_{2k}) → (A_{2k-2} block charge, (c_1, c_2))`.
    Block coord of γ_i (i = 3 … 2k) is `c_i − c_2·[i even]`."""
    c = tuple(c)
    c2 = c[1]
    block = tuple(
        c[i - 1] - (c2 if i % 2 == 0 else 0) for i in range(3, 2 * k + 1)
    )
    return block, (c[0], c[1])


def _darboux_unsplit(k: int, block, qt) -> tuple[int, ...]:
    """Inverse of `_darboux_split`."""
    block = tuple(block)
    c1, c2 = qt
    c = [0] * (2 * k)
    c[0], c[1] = c1, c2
    for idx, i in enumerate(range(3, 2 * k + 1)):
        c[i - 1] = block[idx] + (c2 if i % 2 == 0 else 0)
    return tuple(c)


# ---------------------------------------------------------------------------
# Auxiliary + Darboux relabel iso
# ---------------------------------------------------------------------------

def _build_aux_and_iso(k: int, ir_bps):
    """Return `(aux, iso)` where `aux = TensorKAlgebra(block, QT(Z_2))`
    and `iso: ir_bps → aux` is the label-bijective Darboux relabel.

    `block` is the *recursive* `A1A2kRGKAlg(k-1)` for k ≥ 2, bottoming
    out at `TrivialKAlg` for k = 1.  Because `A1A2kRGKAlg`'s UV labels
    are themselves A_{2k-2} lattice charges, the Darboux block-charge
    *is* the recursive label — no further iso is needed, and the whole
    tower is BPS-F-free down to `Trivial ⊗ QT(Z_2)`.
    """
    one = LaurentPoly.one()
    if k == 1:
        from kalgebra_samples import TrivialKAlg
        factor_A = TrivialKAlg()
    else:
        factor_A = A1A2kRGKAlg(k - 1)
    aux = TensorKAlgebra(factor_A, QuantumTorusKAlg([[0, 1], [-1, 0]]))
    triv_id = factor_A.identity() if k == 1 else None

    def fwd(c):
        block, qt = _darboux_split(k, c)
        a_label = triv_id if k == 1 else tuple(block)
        return Element({(a_label, qt): one})

    def inv(label):
        a_label, qt = label
        block = () if k == 1 else a_label
        return Element({_darboux_unsplit(k, block, qt): one})

    iso = KAlgebraIso(
        source=ir_bps, target=aux,
        forward_label_map=fwd, inverse_label_map=inv,
        name=f"IR_BPS(drop γ_2, A_{2*k}) ≅ "
             + ("Trivial" if k == 1 else f"BPS(A_{2*(k-1)})")
             + " ⊗ QT(Z_2)",
    )
    return aux, iso


# ---------------------------------------------------------------------------
# A1A2kRGKAlg
# ---------------------------------------------------------------------------

class A1A2kRGKAlg(IsoComposedRGKAlgebra):
    """`[A_1, A_{2k}]` as an `RGKAlgebra` over the recursive auxiliary
    `A1A2kRGKAlg(k-1) ⊗ QT(Z_2)` via the single-node RG dropping γ_2.

    UV labels are A_{2k} BPS-charge tuples in Z^{2k}.  `RG`,
    `rg_generator` (= `E_𝖖(X_{γ_1})`), `from_ir_image` and `multiply`
    are direct (no UV F-finder).
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError(f"A1A2kRGKAlg: need k ≥ 1, got {k}")
        self.k = k
        n = 2 * k
        node_charges = _standard_basis(n)
        gamma2 = node_charges[1]

        inner = SingleNodeRGKAlgebra(
            pairing=_a_n_pairing(n),
            node_charges=node_charges,
            spec=_spec_drop_second(n),
            gamma_drop=gamma2,
        )
        aux, aux_iso = _build_aux_and_iso(k, inner.auxiliary())
        super().__init__(inner, aux_iso)

        # E_𝖖(X_{γ_1}) generator: a helper BPS on the QT(Z_2) lattice
        # with spec = [(1, 0)] reproduces the isolated-node quantum
        # dilogarithm coefficients {(n, 0): s_n}.
        from bps_kalgebra import BPSKAlgebra
        self._eq_helper = BPSKAlgebra(
            pairing=[[0, 1], [-1, 0]],
            node_charges=[(1, 0), (0, 1)],
            spec=[(1, 0)],
            verify="off",
        )
        self._factor_A_id = aux.factor_A.identity()

    # ----- RG primitives (override the inner/iso-translated defaults) -----

    def RG(self, a) -> Element:
        """`RG(L_a)` = the Darboux relabel of the UV charge `a` into
        the auxiliary canonical basis (the simple IR image)."""
        one = LaurentPoly.one()
        return self._iso.map(Element({tuple(a): one}))

    def from_ir_image(self, x_ir: Element) -> Element:
        """Inverse Darboux relabel: auxiliary `Element` → UV `Element`."""
        return self._iso.inverse(x_ir)

    def rg_generator(self, cutoff: int):
        """`S_RG = E_𝖖(X_{γ_1})`, keyed by auxiliary labels
        `(1_{A_{2k-2}}, (n, 0))` with Habiro coefficients `s_n`,
        truncated to leading q-order ≤ `cutoff`."""
        eq = self._eq_helper.rg_generator(cutoff)   # {(n, 0): HabiroElement}
        out = {}
        for qt_label, h in eq.items():
            out[(self._factor_A_id, tuple(qt_label))] = h
        return out

    # ----- trace via Schur transport through the (recursive) auxiliary ----

    def trace(self, a, K: int = 20):
        """ρ²-twisted trace via Schur transport:

            Tr_UV(L_a)  =  Tr_aux( ρ_aux(S_RG) · RG(L_a) · S_RG )

        (the b = 1 specialisation of the inner-product transport,
        cross-checked against `RGKAlgebra.verify_rg_inner_product`).
        `aux.trace` recurses into `A1A2kRGKAlg(k-1).trace`, bottoming
        out at the closed-form `Trivial ⊗ QT(Z_2)` trace — so the
        whole computation is BPS-F-free.  `inner_product(a, b)` rides
        on this via the `KAlgebra` default
        `trace_element(multiply(rho(a), b))`.

        EXPERIMENTAL.  The transport is structurally correct and fast
        (recursive, BPS-F-free), but at finite cutoff the q⁰
        normalisation does not yet match the Schur-index convention
        (`Tr(1) = (q²;q²)_∞^{·}` from the QT-factor trace, and the
        ρ-twist / E_𝖖 ordering of the dilog insertion need the
        boundary-aware treatment — cf. the planned
        `08_boundary_aware_intertwining_verifier`).  So `trace` /
        `inner_product` are not yet a validated Schur index; this is
        the open follow-up.  The multiplicative structure (`multiply`,
        `RG`, and the `A1A2kKAlg(k) ≅ A1A2kRGKAlg(k)` iso) is
        independent of it and fully validated.
        """
        aux = self.auxiliary()
        s_rg = self._s_rg_as_aux_element(K, K)
        rho_s = aux.rho_element(s_rg)
        rg_a = self.RG(tuple(a))
        prod = aux.multiply_elements(
            rho_s, aux.multiply_elements(rg_a, s_rg)
        )
        return aux.trace_element(prod, K)


if __name__ == "__main__":
    A = A1A2kRGKAlg(2)
    print(f"Constructed A1A2kRGKAlg(k=2);  aux = "
          f"{type(A.auxiliary()).__name__}"
          f"({type(A.auxiliary().factor_A).__name__}, "
          f"{type(A.auxiliary().factor_B).__name__})")
    print("RG(γ_3) =", dict(A.RG((0, 0, 1, 0)).terms))
    print("RG(γ_1) =", dict(A.RG((1, 0, 0, 0)).terms))
    print("S_RG(cutoff=3) =", {k: str(v) for k, v in A.rg_generator(3).items()})
    print("multiply(γ_3, γ_4) =",
          dict(A.multiply((0, 0, 1, 0), (0, 0, 0, 1)).terms))
    print("verify_rg_multiplicative(γ_3, γ_4):",
          A.verify_rg_multiplicative((0, 0, 1, 0), (0, 0, 0, 1)))
