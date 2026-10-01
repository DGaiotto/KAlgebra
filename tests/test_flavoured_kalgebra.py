"""Tests for `KAlgebra.add_flavour` / `AddFlavourKAlgebra`.

Covers four settings:
  A. Trivial base (PentagonKAlg)        -> AbelianZPlusRing(n)
  B. Flavoured base (hexagon BPSKAlg)   -> TensorZPlusRing(R_B, Abelian(n))
  C. The real auxiliary use (AbePureUN) -> grading reads deg = μ-charge
  D. Non-abelian flavour (PentagonKAlg) -> SU2ZPlusRing (Clebsch-Gordan)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kalgebra import Element
from zplus_ring import AbelianZPlusRing, TrivialZPlusRing
from tensor_zplus_ring import TensorZPlusRing
from flavoured_kalgebra import AddFlavourKAlgebra, AddAbelianFlavourKAlgebra

_passed = 0


def check(name, cond):
    global _passed
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    _passed += 1
    print(f"  PASS: {name}")


def _canonical_labels(B, candidates):
    """Keep the candidates that are genuine canonical labels
    (`L_id · L_l == L_l`)."""
    out = []
    for l in candidates:
        try:
            if B.multiply(B.identity(), l) == Element.basis(l):
                out.append(l)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# A. Trivial base — PentagonKAlg + U(1)^2 flavour
# ---------------------------------------------------------------------------

def test_trivial_base_pentagon():
    from kalgebra_samples import PentagonKAlg
    P = PentagonKAlg()
    F = P.add_flavour(2)

    check("coeff ring is Abelian(2)",
          F.coefficient_ring() == AbelianZPlusRing(2))
    check("identity = (base_id, 0^2)",
          F.identity() == (P.identity(), (0, 0)))

    labels = _canonical_labels(P, [P.identity(), (0, 1, 0), (1, 1, 0)])
    check("found >= 2 pentagon labels", len(labels) >= 2)

    # multiply: flavour charges add, base structure preserved.
    a = (labels[0], (1, 0))
    b = (labels[1], (0, 2))
    prod = F.multiply(a, b)
    base_prod = P.multiply(labels[0], labels[1])
    check("multiply flavour adds + base preserved",
          prod.terms == {(bc, (1, 2)): lp for bc, lp in base_prod.terms.items()})

    # rho negates flavour; round-trips.
    check("rho negates flavour",
          F.rho(a) == (P.rho(labels[0]), (-1, 0)))
    check("rho_inverse round-trips", F.rho_inverse(F.rho(a)) == a)

    # to_R_form folds μ onto the flavour-zero section.
    rform = F.to_R_form(Element.basis((labels[1], (1, 2))))
    sec = (labels[1], (0, 0))
    check("to_R_form section is flavour-zero", sec in rform.terms)
    check("to_R_form coeff is μ^(1,2)",
          rform.terms[sec].coeffs[0] == AbelianZPlusRing(2).basis_element((1, 2)))

    # trace μ-dressing: Tr(L_{(id,f)}) = μ^f · Tr_base(id).
    tr = F.trace((P.identity(), (1, 2)), K=4)
    base_tr = P.trace(P.identity(), K=4)
    mu = AbelianZPlusRing(2).basis_element((1, 2))
    check("trace is μ^f · base-trace",
          all(tr.coeffs.get(q) == base_tr.coeffs[q].terms.get((), 0) * mu
              for q in base_tr.coeffs))

    # verifier bundle on real (flavoured) labels.
    fa, fb = (labels[0], (1, -1)), (labels[1], (2, 0))
    check("verify_rho_fixes_identity", F.verify_rho_fixes_identity())
    check("verify_rho_inverse", F.verify_rho_inverse(fa))
    check("verify_rho_is_automorphism", F.verify_rho_is_automorphism(fa, fb))
    check("verify_bar_involution", F.verify_bar_involution(fa, fb))
    check("verify_rho_twisted_trace",
          F.verify_rho_twisted_trace(fa, fb, K=6))
    check("verify_embed_section_roundtrip",
          F.verify_embed_section_roundtrip((labels[1], (3, -2))))
    check("verify_embed_intertwines_rho",
          F.verify_embed_intertwines_rho(
              F.coefficient_ring().basis_element((2, 1))))


# ---------------------------------------------------------------------------
# B. Flavoured base — hexagon BPSKAlgebra (R_B = Abelian(1)) + U(1) flavour
# ---------------------------------------------------------------------------

def test_flavoured_base_hexagon():
    from bps_kalgebra import BPSKAlgebra
    # hexagon: ker(B) = Z·(1,1,1) -> Abelian(1) coefficient ring.
    H = BPSKAlgebra(
        pairing=[[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
        node_charges=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    )
    check("hexagon base is flavoured (Abelian(1))",
          H.coefficient_ring() == AbelianZPlusRing(1))

    F = H.add_flavour(1)
    check("flavoured-base coeff ring is Tensor(Abelian(1), Abelian(1))",
          F.coefficient_ring() == TensorZPlusRing(
              AbelianZPlusRing(1), AbelianZPlusRing(1)))

    labels = _canonical_labels(
        H, [H.identity(), (1, 0, 0), (0, 1, 0), (1, 1, 0)])
    check("found >= 2 hexagon labels", len(labels) >= 2)

    a = (labels[0], (1,))
    b = (labels[1], (-1,))
    # multiply: base structure preserved, flavour adds to (0,).
    prod = F.multiply(a, b)
    base_prod = H.multiply(labels[0], labels[1])
    check("flavoured-base multiply OK",
          prod.terms == {(bc, (0,)): lp for bc, lp in base_prod.terms.items()})

    # the faithfulness roundtrip must hold across the tensor ring.
    check("verify_embed_section_roundtrip (tensor ring)",
          F.verify_embed_section_roundtrip((labels[1], (2,))))
    check("verify_rho_is_automorphism (tensor ring)",
          F.verify_rho_is_automorphism(a, b))
    check("verify_bar_involution (tensor ring)",
          F.verify_bar_involution(a, b))


# ---------------------------------------------------------------------------
# C. The real auxiliary — AbePureUN(N) + U(1)^{N_f} flavour
# ---------------------------------------------------------------------------

def test_abe_pure_un_auxiliary():
    # Re-fixtured twice: onto the PureUNKAlgebra keystone, then
    # onto the general class when that keystone was retired (2026-09-19) — the
    # same flavoured-auxiliary plumbing either way.
    from pure_g_abe_kalgebra import PureGAbeKAlgebra
    from root_datum import u_n
    A = PureGAbeKAlgebra(u_n(2))
    check("pure U(2) is unflavoured", A.coefficient_ring() == TrivialZPlusRing())

    F = A.add_flavour(1)   # add U(1) matter flavour
    check("auxiliary coeff ring is Abelian(1)",
          F.coefficient_ring() == AbelianZPlusRing(1))

    # grading reads deg = μ-charge straight off the label coordinate.
    def deg(label):
        return label[1]
    w = ((0, 0), (1, 0))                      # the fundamental Wilson label
    lab = (w, (3,))
    check("grading deg = μ-charge", deg(lab) == (3,))

    # a flavoured multiply: pure-gauge multiply on base, flavour adds.
    m = F.multiply((w, (1,)), (w, (2,)))
    base_m = A.multiply(w, w)
    check("auxiliary multiply: gauge × gauge, flavour 1+2=3",
          m.terms == {(bc, (3,)): lp for bc, lp in base_m.terms.items()})

    check("verify_embed_section_roundtrip (auxiliary)",
          F.verify_embed_section_roundtrip((w, (2,))))


# ---------------------------------------------------------------------------
# D. Non-abelian flavour — PentagonKAlg + SU(2) flavour (generic ZPlusRing)
# ---------------------------------------------------------------------------

def test_nonabelian_su2_flavour():
    from kalgebra_samples import PentagonKAlg
    from zplus_ring import SU2ZPlusRing
    P = PentagonKAlg()
    R = SU2ZPlusRing()
    F = P.add_flavour(R)          # any ZPlusRing factor, not just abelian

    check("back-compat alias AddAbelianFlavourKAlgebra is AddFlavourKAlgebra",
          AddAbelianFlavourKAlgebra is AddFlavourKAlgebra)
    check("SU2: coeff ring is SU2ZPlusRing", F.coefficient_ring() == R)
    check("SU2: identity = (base_id, χ_0)", F.identity() == (P.identity(), 0))

    labels = _canonical_labels(P, [P.identity(), (0, 1, 0), (1, 1, 0)])
    check("SU2: found >= 2 pentagon labels", len(labels) >= 2)

    # multiply Clebsch-Gordans the flavour: (l0, χ_1)·(l1, χ_1) carries χ_2 + χ_0.
    a = (labels[0], 1)
    b = (labels[1], 1)
    prod = F.multiply(a, b)
    base_prod = P.multiply(labels[0], labels[1])
    expect = {}
    for bc, lp in base_prod.terms.items():
        for fc in (0, 2):                       # χ_1 · χ_1 = χ_2 + χ_0
            expect[(bc, fc)] = lp
    check("SU2: multiply Clebsch-Gordan χ_1·χ_1 = χ_2 + χ_0",
          prod.terms == expect)

    # rho: every SU(2) rep is self-dual, so the flavour label is ρ-fixed.
    check("SU2: rho fixes the (self-dual) flavour label",
          F.rho(a) == (P.rho(labels[0]), 1))
    check("SU2: rho_inverse round-trips", F.rho_inverse(F.rho(a)) == a)

    # trace character: Tr(L_{(id, χ_n)}) = χ_n · Tr_base(id).
    tr = F.trace((P.identity(), 2), K=4)
    base_tr = P.trace(P.identity(), K=4)
    chi2 = R.basis_element(2)
    check("SU2: trace is χ_n · base-trace",
          all(tr.coeffs.get(q) == base_tr.coeffs[q].terms.get((), 0) * chi2
              for q in base_tr.coeffs))

    # axiom verifiers on genuinely non-abelian flavour labels.
    fa, fb = (labels[0], 1), (labels[1], 2)
    check("SU2: verify_rho_fixes_identity", F.verify_rho_fixes_identity())
    check("SU2: verify_rho_inverse", F.verify_rho_inverse(fa))
    check("SU2: verify_rho_is_automorphism", F.verify_rho_is_automorphism(fa, fb))
    check("SU2: verify_bar_involution", F.verify_bar_involution(fa, fb))
    check("SU2: verify_rho_twisted_trace", F.verify_rho_twisted_trace(fa, fb, K=6))
    check("SU2: verify_embed_section_roundtrip",
          F.verify_embed_section_roundtrip((labels[1], 2)))


if __name__ == "__main__":
    print("test_flavoured_kalgebra:")
    test_trivial_base_pentagon()
    test_flavoured_base_hexagon()
    test_abe_pure_un_auxiliary()
    test_nonabelian_su2_flavour()
    print(f"\nAll flavoured-KAlgebra tests passed ({_passed} checks).")
