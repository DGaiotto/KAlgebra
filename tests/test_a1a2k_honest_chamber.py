"""A₂ₖ → A₂ₖ₋₂ flow on the author's standard spec (2026-06-10).

The intended flow drops the **first two nodes** of the linear A_{2k}
quiver; on the standard simple-root spec
`[γ₁, γ₂, γ₃, …, γ_{2k}]` the stuff factors are already the contiguous
head, so the version-(a) closed form applies with NO rearrangement —
`test_two_node_drop_standard_spec` certifies the directional two-node
flow against the UV `BPSKAlgebra` and the validated extractor.  (The
modern exemplar `a1a2k_rgkalgebra.py` is in substance this flow: its
`S_RG = E(X_{γ₁})·E(X_{γ₂}·L)` carries both dropped towers, sourced
from `spec=[γ₁,γ₂]` kets.)

The remaining tests record the related findings:

  * `a1a2k_induction._simple_root_spec_drop_second` builds its UV on
    the reorder `[γ₂, γ₁, γ₃, …]`, which is NOT a presentation of the
    standard A_{2k} spectrum (the kets differ —
    `test_reordered_spec_is_a_different_spectrum`); the reorder looks
    like an artifact of forcing the two-node flow through the
    single-node-only `SingleNodeRGKAlgebra` interface.
  * The honest **single-node** γ₂-drop is also available as an
    alternative factorization (`E(γ₁)` stays in `S_IR`): the standard
    spec arranges by local moves to `[γ₂, γ₁+γ₂, γ₁, γ₃, …]`, the
    exact `S_RG` collapses to the single γ₂ tower (the γ₁+γ₂
    contributions cancel into the IR-canonical dressing), and the
    reconstruction-theorem hypotheses verify for k = 1, 2, 3 with
    *stronger* sampling than `tests/test_a1a2k_induction.py`
    (dropped-direction pairs + full exact S_RG transport).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kalgebra import Element
from laurent_poly import LaurentPoly
from directional_subquiver_rg import (
    DirectionalSingleNodeRG,
    DirectionalSubquiverRG,
    certify_directional_vs_bps,
)
from a1a2k_induction import _darboux_split, _darboux_unsplit, complete_block_iso
from a1a2k_rgkalg import _a_n_pairing, _standard_basis
from tensor_kalgebra import TensorKAlgebra
from quantum_torus_kalgebra import QuantumTorusKAlg
from kalgebra_samples import TrivialKAlg
from kalgebra_iso import KAlgebraIso
from rgkalgebra_iso_via_ir import verify_ir_iso_hypotheses

_ONE = LaurentPoly.one()


def honest_inner(k: int) -> DirectionalSingleNodeRG:
    """`[A₁, A_{2k}]` as the directional drop-γ₂ flow built on the
    standard simple-root spec `1..2k`, arranged stuff-first."""
    n = 2 * k
    nodes = _standard_basis(n)
    return DirectionalSingleNodeRG(
        _a_n_pairing(n), nodes, list(nodes),
        gamma_drop=nodes[1], arrange=True, rg_window=6,
    )


def phi_for(inner: DirectionalSingleNodeRG, k: int) -> KAlgebraIso:
    """The Darboux + block-iso IR correspondence of `a1a2k_phi`, built
    against the honest inner's auxiliary."""
    ir = inner.auxiliary()
    if k == 1:
        A = TrivialKAlg()
        aux = TensorKAlgebra(A, QuantumTorusKAlg([[0, 1], [-1, 0]]))

        def fwd(c):
            _, qt = _darboux_split(1, c)
            return Element({(A.identity(), qt): _ONE})

        def inv(label):
            _, qt = label
            return Element({_darboux_unsplit(1, (), qt): _ONE})
    else:
        from a1a2k_kalg import A1A2kKAlg
        block_iso = complete_block_iso(k - 1, max_degree=6)
        A = A1A2kKAlg(k - 1)
        aux = TensorKAlgebra(A, QuantumTorusKAlg([[0, 1], [-1, 0]]))

        def fwd(c):
            block, qt = _darboux_split(k, c)
            e = block_iso.inverse(Element({tuple(block): _ONE}))
            (lbl, co), = e.terms.items()
            return Element({(lbl, qt): co})

        def inv(label):
            lbl, qt = label
            e = block_iso.map(Element({lbl: _ONE}))
            (chg, co), = e.terms.items()
            return Element({_darboux_unsplit(k, chg, qt): co})

    return KAlgebraIso(
        source=ir, target=aux,
        forward_label_map=fwd, inverse_label_map=inv,
        name=f"IR_BPS(honest drop γ₂, A_{2*k}) ≅ A1A2kKAlg({k-1}) ⊗ QT(Z₂)",
    )


def test_two_node_drop_standard_spec():
    """The intended A₄ → A₂ flow: drop {γ₁, γ₂}, standard spec
    1,2,3,4 already stuff-first (version (a), no arrangement).  Closed
    form ≡ the validated extractor; full staged certificate vs the UV
    `BPSKAlgebra` (ρ- and trace-equivariance included)."""
    from rg_flow import SubquiverRG
    from bps_kalgebra import BPSKAlgebra
    n = 4
    pair = _a_n_pairing(n)
    nodes = _standard_basis(n)
    spec = list(nodes)
    D = DirectionalSubquiverRG(pair, nodes, spec, drop=[0, 1], rg_window=6)
    assert D.stuff_spec == [nodes[0], nodes[1]]
    assert D.ir_spec == [nodes[2], nodes[3]]
    UV = BPSKAlgebra(pairing=pair, node_charges=nodes, spec=spec,
                     verify="off")
    EX = SubquiverRG(UV, [0, 1])
    for cut in (1, 2, 3):
        a = D.rg_generator(cut)
        b = EX.rg_generator(cut)
        assert set(a) == set(b), (cut, sorted(a), sorted(b))
        for kk in a:
            assert (a[kk] - b[kk]).is_zero(), (cut, kk)
    labels = [(0, 0, 0, 0)] + nodes \
        + [(1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1)]
    pairs = [(a, b) for a in nodes for b in nodes]
    rep = certify_directional_vs_bps(
        D, UV, labels=labels, pairs=pairs, trace_K=4)
    failures = {c: v for c, v in rep.items() if c != "iso" and v is not True}
    assert not failures, failures


def test_reordered_spec_is_a_different_spectrum():
    """`[γ₂, γ₁, γ₃, γ₄]` is not a presentation of the standard A₄
    spectrum: the kets differ at γ₁+γ₂ (q-shift 1 vs 3)."""
    from nahm_local import s_gamma_habiro
    pair4 = _a_n_pairing(4)
    std = _standard_basis(4)
    reo = [std[1], std[0], std[2], std[3]]

    def km(spec):
        return [[sum(a[i] * pair4[i][j] * b[j]
                     for i in range(4) for j in range(4)) for b in spec]
                for a in spec]

    g12 = (1, 1, 0, 0)
    a = s_gamma_habiro(g12, [tuple(g) for g in std], km(std))
    b = s_gamma_habiro(g12, [tuple(g) for g in reo], km(reo))
    assert not (a - b).is_zero(), "reorder unexpectedly matches"


def test_honest_arrangement_and_single_tower_s_rg():
    """The arranger fronts `[γ₂, γ₁+γ₂]`; the exact S_RG support is the
    single γ₂ tower (γ₁+γ₂ cancels into the canonical dressing)."""
    for k in (1, 2):
        inner = honest_inner(k)
        n = 2 * k
        g2 = _standard_basis(n)[1]
        g12 = tuple(a + b for a, b in zip(_standard_basis(n)[0], g2))
        assert inner.stuff_spec == [g2, g12], inner.stuff_spec
        for h in range(0, 5):
            comp = inner._s_rg_component((h,))
            expected = tuple(h * x for x in g2)
            assert set(comp) <= {expected}, (k, h, sorted(comp))


def test_honest_induction_step_k1_k2_k3():
    """Reconstruction-theorem hypotheses on the honest chamber, with
    dropped-direction sample pairs and full exact S_RG transport."""
    for k in (1, 2, 3):
        inner = honest_inner(k)
        n = 2 * k
        nodes = _standard_basis(n)
        ir = inner.auxiliary()
        support = []
        for h in range(0, 5):
            support += list(inner._s_rg_component((h,)).keys())
        phi = phi_for(inner, k)
        g2 = nodes[1]
        g12 = tuple(a + b for a, b in zip(nodes[0], nodes[1]))
        label_set = [ir.identity()] \
            + [nodes[i] for i in range(n) if i != 1] + [g2, g12]
        samples = [Element({l: _ONE}) for l in label_set]
        surv = [nodes[i] for i in range(n) if i != 1]
        pair_labels = [(a, g2) for a in surv] + [(g2, a) for a in surv] \
            + [(g2, g2), (g2, g12), (g12, g2), (g12, g12)] \
            + [(surv[0], surv[-1]), (surv[-1], surv[0])]
        pairs = [(Element({a: _ONE}), Element({b: _ONE}))
                 for a, b in pair_labels]
        checks = verify_ir_iso_hypotheses(
            inner, phi, samples, pairs, s_rg_support=support)
        bad = {c: v for c, v in checks.items() if v is not True}
        assert not bad, (k, bad)


def test_two_node_induction_step_k2_k3():
    """Reconstruction-theorem hypotheses for the INTENDED two-node flow:
    inner = drop {γ₁, γ₂} on the standard spec, φ = Darboux split to
    `A1A2kKAlg(k−1) ⊗ QT(Z₂)`, S_RG transport over the full 2-d exact
    graded support, φ-pairs including the dropped (γ₁,γ₂)-plane.
    (k = 1 has no surviving nodes — the IR is a bare QT(Z₂), the
    `_PentagonViaQT` corner validated in `tests/test_rgkalgebra_graded`.)"""
    for k in (2, 3):
        n = 2 * k
        nodes = _standard_basis(n)
        inner = DirectionalSubquiverRG(
            _a_n_pairing(n), nodes, list(nodes), drop=[0, 1], rg_window=6)
        ir = inner.auxiliary()
        support = []
        for h1 in range(0, 4):
            for h2 in range(0, 4):
                support += list(inner._s_rg_component((h1, h2)).keys())
        assert len(support) >= 10, support
        phi = phi_for(inner, k)
        g1, g2 = nodes[0], nodes[1]
        g12 = tuple(a + b for a, b in zip(g1, g2))
        surv = [nodes[i] for i in range(n) if i >= 2]
        label_set = [ir.identity()] + surv + [g1, g2, g12]
        samples = [Element({l: _ONE}) for l in label_set]
        pair_labels = [(a, g2) for a in surv] + [(g2, a) for a in surv] \
            + [(g1, g2), (g2, g1), (g1, g12), (g12, g2), (g12, g12),
               (g1, g1)] \
            + [(surv[0], surv[-1]), (surv[-1], surv[0])]
        pairs = [(Element({a: _ONE}), Element({b: _ONE}))
                 for a, b in pair_labels]
        checks = verify_ir_iso_hypotheses(
            inner, phi, samples, pairs, s_rg_support=support)
        bad = {c: v for c, v in checks.items() if v is not True}
        assert not bad, (k, bad)


if __name__ == "__main__":
    test_two_node_drop_standard_spec()
    print("OK  test_two_node_drop_standard_spec")
    test_two_node_induction_step_k2_k3()
    print("OK  test_two_node_induction_step_k2_k3")
    test_reordered_spec_is_a_different_spectrum()
    print("OK  test_reordered_spec_is_a_different_spectrum")
    test_honest_arrangement_and_single_tower_s_rg()
    print("OK  test_honest_arrangement_and_single_tower_s_rg")
    test_honest_induction_step_k1_k2_k3()
    print("OK  test_honest_induction_step_k1_k2_k3")
    print("\nAll a1a2k honest-chamber tests pass.")
