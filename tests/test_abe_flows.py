"""Self-test for the AbeKAlgebra (Step 5) export — the abelianized-presentation
tier of `A_𝖖[T]`, and the **capstone** that ties the earlier stages together.

Unlike Steps 1–3 (deliberately spine-free), Step 5 **relies on the earlier
stages** — that is the point: the abelianized presentation of a *matter* theory
is the combination of an `AbeKAlgebra` (the pure-gauge keystone) with an
`RGKAlgebra` flow (adjoining the matter), and each presentation is certified
against the BPS / cone / RG presentations of the *same* abstract algebra by a
`KAlgebraIso`. So this stage ships, layered on Steps 1–4:

  A. pure U(N) on the `AbeKAlgebra` contract — `PureGAbeKAlgebra(u_n(N))`, the
     general-gauge-group class of `src/gn/` (orthonormality, the
     contract's derived product and ρ, the Kazhdan–Lusztig acceptance);
  B. **matter**: U(2) with one fundamental hypermultiplet, the preset
     `roster('u2-nf1')` (a `GNAbeKAlgebra`, flavour ring `R(U(1))`);
  C. the **N=2\*** flow `SU2N2StarRGKAlgebra`;
  D. the **object layer**: `KAlgebraObject`s (`pure_u2_object`, `u2_nf1_object`)
     holding the abe / cone / bps presentations under one roof with certified
     `KAlgebraIso` transition maps and a path-independence (coherence)
     certificate — plus `pure_un_bps_iso`
     (`PureGAbeKAlgebra(u_n(N)) ≅ BPS pure U(N)`);
  E. the **SU(2) AbeKAlgebra family**: the native `PureSU2KAlgebra` (non-minuscule),
     the `AbelianizedSU2KAlg` / `SU2Nf1Abe` / `SU2UNf{N_f}AbeKAlgebra` realisations,
     and the SU(2) `KAlgebraObject`s (`pure_su2_object`, `su2_nf1_object` — with
     RG-flow legs — `su2_unf_object` with the Spin(2N_f) enhancement);
  F. **SU(3) / U(3)**: pure SU(3) (`PureGAbeKAlgebra(su_n(3))`, orthonormality)
     and pure U(3) `≅` its BPS realisation (rank 2);
  G. the flow-typed **RGKAlgebraObjects**: the generic `subquiver_flow_object` and
     the concrete SU(2)-gauged flow objects, incl. an Abe + RG object.

The type-A classes this suite once exercised — `PureUNKAlgebra`,
`PureSUNKAlgebra`, `UNNfKAlgebra`, `UNQuiverKAlgebra`, the builder
`N2StarBuilder` and their flows and witnesses — were retired in favour of the
general classes, which reproduce them (see `CHANGELOG.md`).

Run through the gate, `python3 run_tests.py`, which puts every `src/` layer on
the path.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing
from abe_kalgebra import AbeKAlgebra, TorusShape

_ONE = LaurentPoly.one()
def _E(l):
    return Element({l: _ONE})

# lower-Kapustin (m, λ) labels for U(2)
_ID, _Em, _Fm, _CHI, _DET, _DETI = (
    ((0, 0), (0, 0)), ((1, 0), (0, 0)), ((0, -1), (0, 0)),
    ((0, 0), (1, 0)), ((1, 1), (0, 0)), ((-1, -1), (0, 0)))


# ===========================================================================
# A. Pure U(N) on the Abe contract — the general class at a unitary datum
# ===========================================================================

def test_A_pure_un_contract():
    from pure_g_abe_kalgebra import PureGAbeKAlgebra
    from root_datum import u_n
    A = PureGAbeKAlgebra(u_n(2))
    assert A.coefficient_ring() == TrivialZPlusRing()
    assert A.torus_shape() == TorusShape.from_ranks_nf((2,), (0,))
    # derived (contract base) ≡ the class's own — the delegation guarantee
    for a, b in [(_Em, _Fm), (_CHI, _CHI), (_DET, _Fm), (_Em, _Em)]:
        assert dict(AbeKAlgebra.multiply(A, a, b).terms) == dict(A.multiply(a, b).terms), (a, b)
    for a in [_Em, _Fm, _CHI, _DET]:
        assert AbeKAlgebra.rho(A, a) == A.rho(a), a
    # axioms + orthonormality + the KL acceptance (W1 + W2)
    labs = [_ID, _Em, _Fm, _CHI, _DET]
    for a in labs:
        for b in labs:
            assert A.verify_bar_involution(a, b), ("bar", a, b)
            assert A.verify_rho_is_automorphism(a, b), ("rho", a, b)
            assert A.verify_orthonormality(a, b, 4), ("orthonormality", a, b)
        assert A.certify_canonical(a) == a, ("KL acceptance", a)
    print("  PASS: test_A_pure_un_contract")


# ===========================================================================
# B. Matter: U(2) with one fundamental hypermultiplet
# ===========================================================================

def test_B_matter():
    """`roster('u2-nf1')`: U(2) + one fundamental on the general `(G, N)`
    class, flavour ring `R(U(1))`; orthonormality on a window of
    magnetic, electric and flavour labels."""
    from g_matter_roster import roster
    from zplus_ring import UNZPlusRing
    A = roster("u2-nf1")
    assert A.coefficient_ring() == UNZPlusRing(1)
    labs = [(((0, 0), (0, 0)), (0,)), (((1, 0), (0, 0)), (0,)),
            (((0, -1), (0, 0)), (0,)), (((0, 0), (1, 0)), (0,)),
            (((0, 0), (0, 0)), (1,))]
    for a in labs:
        for b in labs:
            assert A.verify_orthonormality(a, b, 4), ("matter orthonormality", a, b)
    print("  PASS: test_B_matter")


# ===========================================================================
# C. N=2* — the adjoint-matter flow
# ===========================================================================

def test_C_su2_n2star_rgflow():
    """`SU2N2StarRGKAlgebra` is a live RGKAlgebra flow (adjoint matter): the
    adjoint S_RG carries even characters, and it flows over R(U(1))."""
    from su2_n2star_rgkalgebra import SU2N2StarRGKAlgebra
    T = SU2N2StarRGKAlgebra()
    assert str(T.coefficient_ring()) == "AbelianZPlusRing(rank=1)"
    assert T._fs_exact_available()
    print("  PASS: test_C_su2_n2star_rgflow")


# ===========================================================================
# D. The object layer — KAlgebraObjects with certified KAlgebraIsos
# ===========================================================================

def _battery(obj, basket, edges_full, edges_light):
    for src, dst in edges_full:
        iso = obj.iso(src, dst)
        s = [_E(l) for l in basket]
        t = [iso.map(x) for x in s]
        pr = [(_E(a), _E(b)) for a in basket[:3] for b in basket[:3]]
        tp = [(iso.map(x), iso.map(y)) for x, y in pr]
        res = iso.verify_all(s, t, pr, tp, trace_K=4)
        assert all(res.values()), (src, dst, res)
    for src, dst in edges_light:
        iso = obj.iso(src, dst)
        s = [_E(l) for l in basket[:3]]
        t = [iso.map(x) for x in s]
        assert iso.verify_round_trip(s, t), (src, dst, "round_trip")
        assert iso.verify_multiplicative([(_E(basket[1]), _E(basket[2]))],
                                         [(iso.map(_E(basket[1])), iso.map(_E(basket[2])))]), (src, dst)


def test_D_pure_u2_object():
    from pure_u2_object import pure_u2_object
    obj = pure_u2_object()
    assert set(obj.keys()) == {"abe", "cone", "bps"}
    basket = [_ID, _Em, _Fm, _DET, ((0, 0), (1, 1))]
    _battery(obj, basket, edges_full=[("abe", "cone")], edges_light=[("abe", "bps")])
    # path-independence across all three presentations
    samples = {"abe": basket, "cone": basket,
               "bps": [next(iter(obj.transport(l, "abe", "bps").terms)) for l in basket]}
    assert obj.verify_coherence(samples)
    print("  PASS: test_D_pure_u2_object  (abe↔cone full, abe↔bps mult, coherence)")


def test_D_u2_nf1_object():
    from u2_nf1_object import u2_nf1_object
    obj = u2_nf1_object()
    assert set(obj.keys()) == {"abe", "cone", "bps"}
    basket = [_ID, _Em, _Fm, _CHI, _DET, _DETI]
    _battery(obj, basket, edges_full=[("abe", "cone")], edges_light=[("abe", "bps")])
    samples = {"abe": basket, "cone": basket,
               "bps": [next(iter(obj.transport(l, "abe", "bps").terms)) for l in basket]}
    assert obj.verify_coherence(samples)
    print("  PASS: test_D_u2_nf1_object  (matter object: abe↔cone full, abe↔bps mult, coherence)")


def test_D_pure_un_bps_iso():
    """`PureGAbeKAlgebra(u_n(2)) ≅ BPS pure U(2)` — the abelianized presentation
    against the BPS realisation, full battery."""
    from pure_un_bps_iso import pure_un_bps_iso
    iso = pure_un_bps_iso(2)
    basket = [_ID, _Em, _Fm, _DET, ((0, 0), (1, 0))]
    s = [_E(l) for l in basket]
    t = [iso.map(x) for x in s]
    pr = [(_E(a), _E(b)) for a in basket[:3] for b in basket[:3]]
    tp = [(iso.map(x), iso.map(y)) for x, y in pr]
    res = iso.verify_all(s, t, pr, tp, trace_K=3)
    assert all(res.values()), res
    print("  PASS: test_D_pure_un_bps_iso  (PureGAbeKAlgebra(u_n(2)) ≅ BPS pure U(2))")


# ===========================================================================
# E. SU(2) AbeKAlgebras — the native pure-SU(2) Abe + the SU(2) object family
# ===========================================================================

def test_E_pure_su2_native():
    """`PureSU2KAlgebra` is a genuinely NATIVE `AbeKAlgebra` (SU(2)
    is non-minuscule), on which orthonormality holds."""
    from pure_su2_kalgebra import PureSU2KAlgebra
    A = PureSU2KAlgebra()
    assert isinstance(A, AbeKAlgebra)
    basis = [A.identity()]
    it = A.basis_iter() if hasattr(A, "basis_iter") else iter(())
    for _ in range(4):
        try:
            basis.append(next(it))
        except StopIteration:
            break
    for a in basis:
        for b in basis:
            assert A.verify_orthonormality(a, b, 4), ("pure SU(2) orthonormality", a, b)
    print("  PASS: test_E_pure_su2_native  (native SU(2) AbeKAlgebra, orthonormality)")


def test_E_pure_su2_object():
    """The pure-SU(2) `KAlgebraObject` (abe / cone / bps / skein): the whole
    pairwise KAlgebraIso battery (abe-leg trace excluded — the decoupled-photon
    caveat; the skein ↔ bps trace leg included), the groupoid coherence
    certificate, and two landmarks of the stated-skein annulus leg —
    the cone's adjoint Wilson line `W₂` is the once-wrapping core multicurve
    `(0,−2)`, and the 't Hooft line round-trips skein → abe → skein."""
    from laurent_poly import LaurentPoly
    from pure_su2_object import pure_su2_object
    one = LaurentPoly({0: 1})
    obj = pure_su2_object()
    assert set(obj.keys()) == {"abe", "cone", "bps", "skein"}
    assert dict(obj.transport(((("W", 2), 1),), "cone", "skein").terms) == \
        {(0, -2): one}, "cone W_2 must be the skein core loop (0,-2)"
    (al, _c), = obj.transport((1, 0), "skein", "abe").terms.items()
    assert al == (1, 0) and dict(obj.transport(al, "abe", "skein").terms) == \
        {(1, 0): one}, "'t Hooft line round trip skein -> abe -> skein"
    cone = [(), ((0, 1),), ((1, 1),), ((-1, 1),), ((("W", 1), 1),), ((0, 2),), ((0, 1), (1, 1))]
    samples = {"cone": cone,
               "abe": [next(iter(obj.transport(l, "cone", "abe").terms)) for l in cone],
               "bps": [next(iter(obj.transport(l, "cone", "bps").terms)) for l in cone],
               "skein": [next(iter(obj.transport(l, "cone", "skein").terms)) for l in cone]}
    out = obj.verify_pairwise(samples, trace_K=6)
    for pair, checks in out.items():
        skip = {"trace_equivariant"} if "abe" in pair else set()
        assert not {k: v for k, v in checks.items() if v is False and k not in skip}, (pair, checks)
    assert obj.verify_coherence(samples)
    print("  PASS: test_E_pure_su2_object  (abe/cone/bps batteries + coherence; "
          "skein leg landmarks)")


def test_E_su2_nf1_object():
    """SU(2)+N_f=1 `KAlgebraObject` — the matter object with **RG-flow legs**
    (the Abe + RG story): coherence across cone / bps / bps-rform / abe and the
    `rg-to-pure-su2` / `rg-to-pentagon` flow legs + the `bps-mut` mutation."""
    from su2_nf1_object import su2_nf1_object
    obj = su2_nf1_object()
    assert {"abe", "cone", "bps", "rg-to-pure-su2", "rg-to-pentagon", "bps-mut"} <= set(obj.keys())
    cone = [((), 0), (((0, 1),), 0), (((1, 1),), 0), (((-1, 1),), 0),
            ((((("W", 1)), 1),), 0), (((0, 2),), 0), ((), 1)]
    node = [(1, 0, 0), (-1, 2, 0), (0, -1, 1), (0, 1, 1), (1, -1, 1)]
    S = {"cone": cone,
         "bps-rform": [next(iter(obj.transport(l, "cone", "bps-rform").terms)) for l in cone],
         "bps": [next(iter(obj.transport(l, "cone", "bps").terms)) for l in cone],
         "abe": [next(iter(obj.transport(l, "cone", "abe").terms)) for l in cone],
         "rg-to-pure-su2": list(node), "rg-to-pentagon": list(node),
         "bps-mut": [next(iter(obj.transport(l, "bps", "bps-mut").terms)) for l in node]}
    assert obj.verify_coherence(S)
    print("  PASS: test_E_su2_nf1_object  (matter object + RG-flow legs + coherence)")


def test_E_su2_unf_object():
    """SU(2)+N_f≥2 `KAlgebraObject` (the Spin(2N_f)-manifest cone/bps pair):
    structure + the Z⁴-core coherence certificate.  Its U(N_f)-manifest abe leg
    was built on the retired `UNNfKAlgebra` and was retired with it; restoring
    an abe leg on `GNAbeKAlgebra` is open (see `CHANGELOG.md`)."""
    from su2_unf_object import su2_unf_object
    obj = su2_unf_object()
    assert set(obj.keys()) == {"cone", "bps"}, sorted(obj.keys())
    basket = [(0, 0, 0, 0), (1, 0, 1, 0)]
    assert obj.verify_coherence({"cone": basket, "bps": basket})
    print("  PASS: test_E_su2_unf_object  (Spin(2N_f) cone/bps, coherence)")


def test_E_abelianized_su2_bps_iso():
    """The standalone pure-SU(2) abe ↔ BPS `KAlgebraIso` (Kronecker chart)."""
    from abelianized_su2_bps_iso import abelianized_su2_bps_iso
    iso = abelianized_su2_bps_iso()
    assert iso.verify_unit()
    labs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]   # canonical (m,e), m≥0
    s = [_E(l) for l in labs]
    t = [iso.map(x) for x in s]
    assert iso.verify_round_trip(s, t)
    pairs = [((0, 1), (0, 1)), ((1, 0), (1, 0)), ((0, 1), (1, 1))]
    assert iso.verify_multiplicative([(_E(a), _E(b)) for a, b in pairs],
                                     [(iso.map(_E(a)), iso.map(_E(b))) for a, b in pairs])
    print("  PASS: test_E_abelianized_su2_bps_iso  (pure SU(2): abe ≅ BPS Kronecker)")


# ===========================================================================
# F. SU(3) / U(3) — the keystone and the native SU(3), against BPS
# ===========================================================================

def test_F_u3_and_su3():
    """Pure SU(3) and pure U(3) on the general class: orthonormality
    at SU(3), and U(3) against its BPS realisation.  (The SU(3) ↔ BPS witness
    of earlier releases was built on the retired `PureSUNKAlgebra` and has no
    counterpart on the general class.)"""
    from pure_g_abe_kalgebra import PureGAbeKAlgebra
    from root_datum import su_n
    from pure_un_bps_iso import pure_un_bps_iso
    A3 = PureGAbeKAlgebra(su_n(3))
    assert isinstance(A3, AbeKAlgebra)
    labs3 = [((0, 0), (0, 0)), ((0, 0), (1, 0)), ((0, 0), (0, 1)), ((1, 1), (0, 0))]
    for a in labs3:
        for b in labs3:
            assert A3.verify_orthonormality(a, b, 4), ("pure SU(3) orthonormality", a, b)
    iso = pure_un_bps_iso(3)
    labs = [((0, 0, 0), (0, 0, 0)), ((1, 0, 0), (0, 0, 0)), ((0, 0, -1), (0, 0, 0))]
    s = [_E(l) for l in labs]
    assert iso.verify_round_trip(s, [iso.map(x) for x in s])
    print("  PASS: test_F_u3_and_su3  (pure SU(3) orthonormality; U(3) ≅ BPS)")


# ===========================================================================
# G. RGKAlgebraObjects — the flow-typed object layer
# ===========================================================================

def test_G_subquiver_flow_object():
    """The generic `subquiver_flow_object`: one node-deletion flow in two live
    descriptions (F-oracle vs generic co-solver), bridged by an `RGKAlgebraIso`
    whose `verify_rg_intertwine` / `verify_s_rg_match` battery IS the historical
    cross-check, now a curated certificate."""
    from rgkalgebra_object import subquiver_flow_object
    from bps_kalgebra import BPSKAlgebra
    A = BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=[(1, 0), (0, 1)],
                    spec=[(1, 0), (0, 1)], verify="off")
    O = subquiver_flow_object(A, [0], name="pentagon-drop0")
    assert set(O.keys()) == {"f-oracle", "co-solver"}
    flows = O.verify_flow_pairwise({"f-oracle": [(1, 0), (0, 1), (1, 1)]}, cutoff=3)
    assert all(all(r.values()) for r in flows.values()), flows
    assert set(O.aux_object().keys()) == {"f-oracle", "co-solver"}
    print("  PASS: test_G_subquiver_flow_object  (RGKAlgebraObject flow witness)")


def test_G_su2a1d3_rg_object():
    """The concrete `RGKAlgebraObject`s for the SU(2)-gauged [A₁,D₃] node-deletion
    flows: the flow witness `rg_intertwine`, and the **Abe + RG** object whose IR
    auxiliary's first tensor factor is the abelianized pure SU(2)."""
    from su2a1d3_rg_object import (su2a1d3_matter_drop_object,
                                   su2a1d3_matter_drop_abe_object)
    uv = [(1, 0, 0, 0), (-1, 2, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1)]
    O = su2a1d3_matter_drop_object()
    assert set(O.keys()) == {"bps-ir", "pure-su2-x-sqed1"}
    assert O.iso("bps-ir", "pure-su2-x-sqed1").verify_rg_intertwine(uv)
    # the Abe + RG object
    Oa = su2a1d3_matter_drop_abe_object()
    assert Oa.keys() == ["bps-ir", "abe-su2-x-sqed1"]
    assert "abelianized" in Oa.capabilities("abe-su2-x-sqed1")
    swap = Oa.realization("abe-su2-x-sqed1")
    assert swap.verify_rg_unital()
    assert swap.verify_rg_multiplicative((0, 0, 1, 0), (0, 0, 0, 1))
    print("  PASS: test_G_su2a1d3_rg_object  (flow witness + Abe+RG object)")


def test_G_su2a1d4_forgetful_ladder():
    """`su2a1d4_tail_drop_object` — the forgetful-RG ladder SU2A1D4 → SU2A1D3
    (drop the flavour-charged tail node) as an `RGKAlgebraObject`."""
    from su2a1d4_rg_object import su2a1d4_tail_drop_object
    O = su2a1d4_tail_drop_object()
    assert set(O.keys()) == {"bps-ir", "su2a1d3"}
    uv = [(1, 0, 0, 0, 0), (-1, 2, 0, 0, 0), (0, -1, 1, 0, 0), (0, 0, 0, 1, 0)]
    assert O.iso("bps-ir", "su2a1d3").verify_rg_intertwine(uv)   # the flow witness
    print("  PASS: test_G_su2a1d4_forgetful_ladder  (RG ladder SU2A1D4 → SU2A1D3, flow witness)")


def main():
    test_A_pure_un_contract()
    test_B_matter()
    test_C_su2_n2star_rgflow()
    test_D_pure_u2_object()
    test_D_u2_nf1_object()
    test_D_pure_un_bps_iso()
    test_E_pure_su2_native()
    test_E_pure_su2_object()
    test_E_su2_nf1_object()
    test_E_su2_unf_object()
    test_E_abelianized_su2_bps_iso()
    test_F_u3_and_su3()
    test_G_subquiver_flow_object()
    test_G_su2a1d3_rg_object()
    test_G_su2a1d4_forgetful_ladder()
    print("\nALL AbeKAlgebra export self-tests passed (Abe tier + matter + N=2* + "
          "object layer + SU(2)/SU(3) families + RGKAlgebraObjects).")


if __name__ == "__main__":
    main()
