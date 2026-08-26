"""Self-test for the AbeKAlgebra (Step 5) export — the abelianized-presentation
tier of `A_𝖖[T]`, and the **capstone** that ties the earlier stages together.

Unlike Steps 1–3 (deliberately spine-free), Step 5 **relies on the earlier
stages** — that is the point: the abelianized presentation of a *matter* theory
is the combination of an `AbeKAlgebra` (the pure-gauge keystone) with an
`RGKAlgebra` flow (adjoining the matter), and each presentation is certified
against the BPS / cone / RG presentations of the *same* abstract algebra by a
`KAlgebraIso`. So this stage ships, layered on Steps 1–4:

  A. the pure-U(N) keystone `PureUNKAlgebra` (Goal 2.1 orthonormality), built
     constructively on the enriched rational quantum torus (the ABSOLUTE RULE);
  B. **matter = Abe + RG**: the native `UNNfKAlgebra(N, N_f)` and the Route-A
     flow `UNNfOverPure`, with the flow ↔ BPS `KAlgebraIso` (Goal 1.3);
  C. the **N=2\*** machinery — `N2StarBuilder` ("solve in N=2\*, RG-flow" — the
     pure-U(N)/SU(N) canonical finder) and the `SU2N2StarRGKAlgebra` flow;
  D. the **object layer**: `KAlgebraObject`s (`pure_u2_object`, `u2_nf1_object`)
     holding the abe / cone / bps presentations under one roof with certified
     `KAlgebraIso` transition maps and a path-independence (coherence)
     certificate — plus `pure_un_bps_iso` (`PureUNKAlgebra(N) ≅ BPS pure U(N)`);
  E. the **SU(2) AbeKAlgebra family**: the native `PureSU2KAlgebra` (non-minuscule),
     the `AbelianizedSU2KAlg` / `SU2Nf1Abe` / `SU2UNf{N_f}AbeKAlgebra` realisations,
     and the SU(2) `KAlgebraObject`s (`pure_su2_object`, `su2_nf1_object` — with
     RG-flow legs — `su2_unf_object` with the Spin(2N_f) enhancement);
  F. **SU(3) / U(3)**: the pure-U(3) keystone and the native pure SU(3)
     `PureSUNKAlgebra`, each `≅` its BPS realisation (Goal 1.3, rank 2);
  G. the flow-typed **RGKAlgebraObjects**: the generic `subquiver_flow_object` and
     the concrete SU(2)-gauged flow objects, incl. an Abe + RG object.

Run with the earlier stages on the path (Step 5 depends on Steps 1–4):

    PYTHONPATH=KAlgebra:ConeKAlgebra:RGKAlgebra:BPSKAlgebra:AbeKAlgebra \
        python AbeKAlgebra/test_abe_flows.py
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
# A. The pure-U(N) keystone — the Abe contract + Goal 2.1 orthonormality
# ===========================================================================

def test_A_pure_keystone_contract():
    from pure_un_kalgebra import PureUNKAlgebra, default_rays
    A = PureUNKAlgebra(2, default_rays(2), max_len=2, K=8)
    assert A.coefficient_ring() == TrivialZPlusRing()
    assert A.torus_shape() == TorusShape.from_ranks_nf((2,), (0,))
    # derived (contract base) ≡ engine (keystone override) — the delegation guarantee
    for a, b in [(_Em, _Fm), (_CHI, _CHI), (_DET, _Fm), (_Em, _Em)]:
        assert dict(AbeKAlgebra.multiply(A, a, b).terms) == dict(A.multiply(a, b).terms), (a, b)
    for a in [_Em, _Fm, _CHI, _DET]:
        assert AbeKAlgebra.rho(A, a) == A.rho(a), a
    # axioms + orthonormality (Goal 2.1) + the KL acceptance (W1 + W2)
    labs = [_ID, _Em, _Fm, _CHI, _DET]
    for a in labs:
        for b in labs:
            assert A.verify_bar_involution(a, b), ("bar", a, b)
            assert A.verify_rho_is_automorphism(a, b), ("rho", a, b)
            assert A.verify_orthonormality(a, b, 4), ("Goal 2.1", a, b)
        assert A.certify_canonical(a) == a, ("KL acceptance", a)
    print("  PASS: test_A_pure_keystone_contract")


# ===========================================================================
# B. Matter = Abe + RG:  UNNfKAlgebra (native) + UNNfOverPure (flow) + the iso
# ===========================================================================

def test_B_matter_native_abe():
    from un_nf_kalgebra import UNNfKAlgebra
    A = UNNfKAlgebra(2, 1)                       # U(2) + N_f = 1 (flavour SU(1) trivial → Z)
    assert A.coefficient_ring() == TrivialZPlusRing()
    labs = [_ID, _Em, _Fm, _CHI]
    for a in labs:
        for b in labs:
            assert A.verify_orthonormality(a, b, 4), ("UNNf orthonormality", a, b)
    print("  PASS: test_B_matter_native_abe")


def test_B_matter_flow_bps_iso():
    """The Route-A flow `UNNf1OverPure(2)` (Abe keystone + RG matter) and the
    BPS realisation of U(2)+N_f=1 are the SAME abstract algebra — the full
    KAlgebraIso battery (Goal 1.3)."""
    from un_nf1_over_pure_iso import build_iso, certify
    iso, A, B = build_iso()
    res = certify(iso, A, B, trace_K=3, verbose=False)
    assert all(res.values()), res
    print("  PASS: test_B_matter_flow_bps_iso  (unit/round_trip/mult/rho/trace all True)")


# ===========================================================================
# C. N=2* machinery — the "solve in N=2*, RG-flow" pure-U(N)/SU(N) finder
# ===========================================================================

def test_C_n2star_builder():
    """`N2StarBuilder(N).pure_chart(points)` builds a torus slice; the pure
    keystone `decompose`s it to a SINGLE coefficient-1 canonical."""
    from pure_via_n2star import N2StarBuilder
    from pure_un_kalgebra import PureUNKAlgebra, default_rays
    reg = PureUNKAlgebra(2, default_rays(2), max_len=3, K=8)
    B = N2StarBuilder(2)
    for points, kind in [([(1, 0), (0, 0)], "H1 minuscule"),
                         ([(2, 0), (0, 0)], "aligned tower L_(2,0)"),
                         ([(1, 0), (-1, 0)], "SU(2) adjoint monopole")]:
        s = B.pure_chart(points)
        assert s.well_formed(), (points, "not well-formed")
        d = reg.decompose(s)
        terms = {lab: str(c) for lab, c in d.terms.items()}
        assert len(terms) == 1 and list(terms.values())[0] == "1", (points, terms)
    print("  PASS: test_C_n2star_builder")


def test_C_su2_n2star_rgflow():
    """`SU2N2StarRGKAlgebra` is a live RGKAlgebra flow (adjoint matter): the
    adjoint S_RG carries even characters, and it flows over R(U(1))."""
    from su2_n2star_rgkalgebra import SU2N2StarRGKAlgebra
    T = SU2N2StarRGKAlgebra()
    assert str(T.coefficient_ring()) == "AbelianZPlusRing(rank=1)"
    assert T._fs_exact_available()
    print("  PASS: test_C_su2_n2star_rgflow")


# ===========================================================================
# D. The object layer — KAlgebraObjects with certified KAlgebraIsos (Goal 1.3)
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
    """PureUNKAlgebra(2) ≅ BPS pure U(2) — the Abe keystone against the Step-4
    realisation, full battery (Goal 1.3)."""
    from pure_un_bps_iso import pure_un_bps_iso
    iso = pure_un_bps_iso(2)
    basket = [_ID, _Em, _Fm, _DET, ((0, 0), (1, 0))]
    s = [_E(l) for l in basket]
    t = [iso.map(x) for x in s]
    pr = [(_E(a), _E(b)) for a in basket[:3] for b in basket[:3]]
    tp = [(iso.map(x), iso.map(y)) for x, y in pr]
    res = iso.verify_all(s, t, pr, tp, trace_K=3)
    assert all(res.values()), res
    print("  PASS: test_D_pure_un_bps_iso  (PureUNKAlgebra(2) ≅ BPS pure U(2))")


# ===========================================================================
# E. SU(2) AbeKAlgebras — the native pure-SU(2) Abe + the SU(2) object family
# ===========================================================================

def test_E_pure_su2_native():
    """`PureSU2KAlgebra` (Plan 24) is a genuinely NATIVE `AbeKAlgebra` (SU(2)
    is non-minuscule), on which orthonormality (Goal 2.1) holds."""
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
            assert A.verify_orthonormality(a, b, 4), ("pure SU(2) Goal 2.1", a, b)
    print("  PASS: test_E_pure_su2_native  (native SU(2) AbeKAlgebra, orthonormality)")


def test_E_pure_su2_object():
    """The pure-SU(2) `KAlgebraObject` (abe / cone / bps): the whole pairwise
    KAlgebraIso battery (abe-leg trace excluded — the decoupled-photon caveat)
    and the groupoid coherence certificate (Goal 1.3)."""
    from pure_su2_object import pure_su2_object
    obj = pure_su2_object()
    assert set(obj.keys()) == {"abe", "cone", "bps"}
    cone = [(), ((0, 1),), ((1, 1),), ((-1, 1),), ((("W", 1), 1),), ((0, 2),), ((0, 1), (1, 1))]
    samples = {"cone": cone,
               "abe": [next(iter(obj.transport(l, "cone", "abe").terms)) for l in cone],
               "bps": [next(iter(obj.transport(l, "cone", "bps").terms)) for l in cone]}
    out = obj.verify_pairwise(samples, trace_K=6)
    for pair, checks in out.items():
        skip = {"trace_equivariant"} if "abe" in pair else set()
        assert not {k: v for k, v in checks.items() if v is False and k not in skip}, (pair, checks)
    assert obj.verify_coherence(samples)
    print("  PASS: test_E_pure_su2_object  (abe/cone/bps batteries + coherence)")


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
    """SU(2)+N_f≥2 `KAlgebraObject` (Spin(2N_f)-manifest cone/bps + the U(N_f)
    ungauged abe leg): structure + the Z⁴-core coherence certificate."""
    from su2_unf_object import su2_unf_object
    obj = su2_unf_object()
    assert {"abe", "cone", "bps"} <= set(obj.keys())
    basket = [(0, 0, 0, 0), (1, 0, 1, 0)]
    assert obj.verify_coherence({"cone": basket, "bps": basket, "abe": []})
    print("  PASS: test_E_su2_unf_object  (Spin(2N_f) cone/bps + U(N_f) abe leg, coherence)")


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
# F. SU(3) / U(3) — the keystone and the native SU(3), against BPS (Goal 1.3)
# ===========================================================================

def test_F_u3_and_su3():
    """Pure U(3) keystone + the genuinely native pure SU(3) `PureSUNKAlgebra`,
    orthonormality (Goal 2.1), and both against their BPS realisations."""
    from pure_un_kalgebra import PureUNKAlgebra
    from pure_sun_kalgebra import PureSUNKAlgebra
    from pure_un_bps_iso import pure_un_bps_iso
    from pure_sun_bps_iso import pure_sun_bps_iso
    # native pure SU(3) — orthonormality
    A3 = PureSUNKAlgebra(3, max_len=4, K=10)
    assert isinstance(A3, AbeKAlgebra)
    assert A3.verify_orthonormality(A3.identity(), A3.identity(), 4)
    # U(3) ≅ BPS pure U(3)
    U3 = PureUNKAlgebra.cached(3)
    iso = pure_un_bps_iso(3, abe=U3)
    labs = [((0, 0, 0), (0, 0, 0)), ((1, 0, 0), (0, 0, 0)), ((0, 0, -1), (0, 0, 0))]
    s = [_E(l) for l in labs]
    assert iso.verify_round_trip(s, [iso.map(x) for x in s])
    # SU(3) ≅ BPS pure SU(3): unit + round-trip on a small basis window
    iso3 = pure_sun_bps_iso(3, abe=A3)
    assert iso3.verify_unit()
    s3 = [A3.identity()]
    it = A3.basis_iter() if hasattr(A3, "basis_iter") else iter(())
    for _ in range(4):
        try:
            s3.append(next(it))
        except StopIteration:
            break
    se3 = [_E(l) for l in s3]
    assert iso3.verify_round_trip(se3, [iso3.map(x) for x in se3])
    print("  PASS: test_F_u3_and_su3  (native SU(3) orthonormality; U(3)/SU(3) ≅ BPS)")


# ===========================================================================
# G. RGKAlgebraObjects — the flow-typed object layer (Plan 25)
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


def test_H_the_primitive_cocycle():
    """**`CC` is the primitive**, and `R` is derived from it — not the other way
    round.  The product law on the enriched torus is written in `CC`:

        U_m · U_{m'}  =  CC[N]_{m,m'} · U_{m+m'},     R = T_{a+b}(CC)

    with `CC = CC[0]` the pure-gauge case, a genuine specialisation matching the
    `(G, N)` / `(G, 0)` convention.  Its closed form absorbs the atom phase, the
    Weyl transport and the half-integral-height correction, so nothing in it
    mentions `ψ`; `ψ` and `Z` are demoted to **trivializations**, which makes
    `δψ = R` and `δZ = W` emergent evidence rather than definitions.

    The reason `CC` is the primitive and not `R` is measurable, and this test is
    that measurement: **`CC` satisfies the bar axiom `bar(CC_{a,b}) = CC_{b,a}`
    and `R` does not.**  Every ingredient of `CC` is symmetric in `|A|, |B|`, so
    the swap and the charge conjugation coincide.  The `R` half is asserted to
    **fail** somewhere — otherwise the `CC` half tests nothing.
    """
    import itertools
    import root_datum as rd
    import wrq_torus as W

    D = rd.su_n(3)
    charges = list(itertools.product(range(-1, 2), repeat=2))
    cc_bad = [(a, b) for a in charges for b in charges
              if W.CC(D, a, b).bar() != W.CC(D, b, a)]
    r_bad = [(a, b) for a in charges for b in charges
             if W.cocycle_R(D, a, b).bar() != W.cocycle_R(D, b, a)]
    assert not cc_bad, ("CC must satisfy the bar axiom everywhere", cc_bad[:3])
    assert r_bad, ("the R half must FAIL somewhere, or the CC half is vacuous")

    print("  PASS: test_H_the_primitive_cocycle  (SU(3): bar(CC_ab) = CC_ba on "
          "%d/%d charge pairs, while R fails on %d — which is why CC is the "
          "primitive)" % (len(charges) ** 2 - len(cc_bad), len(charges) ** 2,
                          len(r_bad)))


def test_H_rho_is_a_label_level_closed_form():
    """`ρ`/`ρ⁻¹` on this tier are now the **explicit label-level closed form**,
    promoted to primary; the old route — read the torus `G`-twist back through
    `decompose` — is demoted to the verifier `verify_rho_via_twist`.

    On Weyl orbits of pairs, with no chamber assumed,

        ρ^{±1}[(m, e)] = [( −m, −e + Σ_{α: ±⟨α,m⟩>0} |⟨α,m⟩|·α
                                  − Σ_{w∈wt(N): ±⟨w,m⟩>0} |⟨w,m⟩|·w )]

    — `ρ` uses the roots positive on `m`, `ρ⁻¹` the negative half.  The check
    that costs nothing and catches a sign error is the **round trip**: `ρ⁻¹ρ` must
    be the identity on labels, for the ρ-automorphism axiom to have a chance.

    A label is a **Weyl orbit of a pair**, not a tuple, so the comparison is on
    the dominant representative.  That is not a technicality: at SU(3),
    `dominant_cochar_rep((1,0)) = (1,1)`, so `L_{(1,0),0}` and `L_{(1,1),0}` are
    the *same line*, and a round trip checked on raw tuples reports a spurious
    failure — which is what happened while writing this test.
    """
    import root_datum as rd
    import wrq_torus as W

    checked = 0
    for datum in (rd.su_2(), rd.su_n(3), rd.sp_n(2)):
        n = datum.dim
        labels = [((1,) + (0,) * (n - 1), (0,) * n),
                  ((1,) * n, (0,) * n),
                  ((0,) * n, (1,) + (0,) * (n - 1))]
        for m, e in labels:
            fwd = W.rho_label(datum, m, e)
            back = W.rho_label(datum, fwd[0], fwd[1], inverse=True)
            want = tuple(datum.dominant_cochar_rep(m))
            got = tuple(datum.dominant_cochar_rep(back[0]))
            assert got == want, ("ρ⁻¹ρ moved the magnetic label",
                                 datum.name, (m, e), fwd, back)
            checked += 1
    print("  PASS: test_H_rho_is_a_label_level_closed_form  (ρ⁻¹ρ = id on %d "
          "labels at SU(2) / SU(3) / Sp(4), compared on dominant "
          "representatives)" % checked)


def test_H_chart_cache_round_trip_and_its_guards():
    """The `L_{m,e}` are **memoized and persistable** — `save_cache` / `load_cache`,
    the same names and shape as `RGKAlgebra`'s, so the chart tier's caches persist
    the way the flow tier's already did.

    The file is **exact**: a residual is an integer numerator over an explicit
    denominator multiset, so a round trip is an identity, not a re-derivation.

    Two guards, because a cache file is untrusted input — it lets an `L_{m,e}`
    enter the algebra without having been built by the guarded solve, and on this
    tier a *fast wrong answer* is the dangerous failure mode:

    * **provenance** — the header fingerprints the presentation, including the
      phase convention **probed rather than described** (a datum's phase is a
      callable and cannot be compared any other way).  This matters because a
      chart is a residual vector in raw coordinates: cross-loading SU(3)'s charts
      into Sp(4) would not raise anywhere downstream, it would simply be wrong.
      So a mismatch must **refuse the file**, which is what is asserted here;
    * **re-verification** — every admitted chart is re-run through the axioms.
      Affordable precisely because verifying is not solving (measured at ~5% of a
      rebuild), and worth it because (★) is exactly the condition that rejects a
      plausible impostor the `𝖖⁰` self-norm cannot see.
    """
    import os
    import tempfile
    import root_datum as rd
    from pure_g_abe_kalgebra import PureGAbeKAlgebra

    A = PureGAbeKAlgebra(rd.su_n(3))
    lab = A.fold((1, 1), (0, 0))
    built = A.chart(lab)

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "su3_charts.json")
    assert A.save_cache(path) >= 1

    B = PureGAbeKAlgebra(rd.su_n(3))
    assert B.load_cache(path) >= 1
    assert B.chart(lab) == built, "the round trip must be an identity"

    refused = False
    try:
        PureGAbeKAlgebra(rd.sp_n(2)).load_cache(path)
    except ValueError:
        refused = True
    assert refused, ("a fingerprint mismatch must REFUSE the file — a "
                     "cross-loaded chart would not raise downstream, it would "
                     "simply be wrong")
    print("  PASS: test_H_chart_cache_round_trip_and_its_guards  (SU(3) charts "
          "save/load exactly; a Sp(4) instance refuses the SU(3) file)")


def test_H_laurent_poly_refuses_a_non_integral_coefficient():
    """`LaurentPoly` is over `Z[q, q⁻¹]`, and it now **says so**.

    Two defects shipped, and both are the same failure class as the `int()`
    truncation that once sent `SO(5)`'s spinor weight `(½,½)` to `v^0`:

    * a non-integral coefficient was **silently truncated** — the zero-test read
      the original `c` while the store kept `int(c)`, so `Fraction(1,2)` passed
      the test and landed as **0**, and `3/2` as `1`;
    * that stored zero then broke this class's documented invariant (a sparse
      dict of *non-zero* coefficients), and `__add__`'s two paths disagreed about
      it — the general path raised `KeyError` where the monomial fast path did
      not, so the same sum crashed or not depending on how many terms the right
      operand happened to have.

    Refusing is the fix for the first; making the paths agree is free.
    """
    from fractions import Fraction
    from laurent_poly import LaurentPoly

    for bad in (Fraction(1, 2), Fraction(3, 2), 0.5):
        raised = False
        try:
            LaurentPoly({0: bad})
        except TypeError:
            raised = True
        assert raised, ("a non-integral coefficient must raise, not truncate", bad)

    # both `__add__` paths agree on an explicit zero entry
    dirty = LaurentPoly._from_clean_dict({0: 0, 1: 2})
    assert LaurentPoly({0: 1}) + dirty == LaurentPoly({0: 1, 1: 2})
    assert LaurentPoly({1: 5}) + dirty == LaurentPoly({1: 7})
    print("  PASS: test_H_laurent_poly_refuses_a_non_integral_coefficient  "
          "(Fraction/float refused; both __add__ paths agree on a stored zero)")


def main():
    test_A_pure_keystone_contract()
    test_B_matter_native_abe()
    test_B_matter_flow_bps_iso()
    test_C_n2star_builder()
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
    test_H_the_primitive_cocycle()
    test_H_rho_is_a_label_level_closed_form()
    test_H_chart_cache_round_trip_and_its_guards()
    test_H_laurent_poly_refuses_a_non_integral_coefficient()
    print("\nALL AbeKAlgebra (Step 5) export self-tests passed (Abe tier + matter "
          "Abe+RG + N=2* + object layer + SU(2)/SU(3) families + RGKAlgebraObjects).")


if __name__ == "__main__":
    main()
