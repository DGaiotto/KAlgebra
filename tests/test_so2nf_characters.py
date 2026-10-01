"""Tests for `so2nf_characters` — `Spin(2Nf) = D_Nf` character theory and the
flavour-enhancement checker for SU(2)+Nf-type indices presented over the
`U(1)^{Nf}` Cartan.

Covers: D-series rep theory (dimensions, tensor product, round-trip
decomposition), the checker on synthetic genuine/broken/virtual indices, and
the end-to-end recovery of the hand-built `su2_nf2` `Spin(4)` trace from its
Cartan data (`tr_W` ↦ checker == `tr_W_su2xsu2`).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractions import Fraction as F

import so2nf_characters as C


def test_dimensions():
    """Known `Spin(2Nf)` irrep dimensions via Weyl's character formula."""
    # Spin(6) = SU(4)
    assert C.dim(3, (1, 0, 0)) == 6                          # vector
    assert C.dim(3, (F(1, 2), F(1, 2), F(1, 2))) == 4        # spinor = SU(4) fund
    assert C.dim(3, (1, 1, 0)) == 15                         # adjoint so(6)
    # Spin(4) = SU(2)×SU(2)
    assert C.dim(2, (1, 0)) == 4                             # vector (2,2)
    assert C.dim(2, (F(1, 2), F(1, 2))) == 2                 # spinor (2,1)
    assert C.dim(2, (F(1, 2), F(-1, 2))) == 2                # cospinor (1,2)
    # Spin(8)
    assert C.dim(4, (1, 0, 0, 0)) == 8                       # vector
    assert C.dim(4, (F(1,2),)*4) == 8                        # spinor (triality)


def test_tensor_product():
    """`6 ⊗ 6 = 20' ⊕ 15 ⊕ 1` in Spin(6)."""
    vec = C.character(3, (1, 0, 0))
    dec = C.decompose(3, C._lpoly_mul(vec, vec))
    assert dec == {(2, 0, 0): 1, (1, 1, 0): 1, (0, 0, 0): 1}
    assert {k: C.dim(3, k) for k in dec} == {(2, 0, 0): 20, (1, 1, 0): 15, (0, 0, 0): 1}


def test_decompose_roundtrip():
    """`reconstruct ∘ decompose = id` on a genuine character sum, and the
    checker recognizes it."""
    content = {(1, 0, 0): 1, (F(1, 2), F(1, 2), F(1, 2)): 1, (0, 0, 0): 2}
    idx = C.reconstruct(3, content)
    enhanced, dec, genuine = C.verify_flavour_enhancement(3, idx)
    assert enhanced and genuine and dec == content
    assert C.reconstruct(3, dec) == idx


def test_checker_rejects_broken_and_flags_virtual():
    content = {(1, 0, 0): 1, (0, 0, 0): 1}
    idx = C.reconstruct(3, content)

    # Cartan-broken: drop one weight of the vector ⇒ not W-invariant.
    broken = dict(idx)
    e = next(w for w in C.character(3, (1, 0, 0)) if w != (0, 0, 0))
    broken[e] = broken.get(e, 0) - 1
    if broken[e] == 0:
        del broken[e]
    enhanced, _, _ = C.verify_flavour_enhancement(3, broken)
    assert enhanced is False

    # Virtual: −(vector) is W-invariant (enhanced) but not genuine.
    virt = {w: -c for w, c in C.character(3, (1, 0, 0)).items()}
    enhanced, dec, genuine = C.verify_flavour_enhancement(3, virt)
    assert enhanced is True and genuine is False and dec == {(1, 0, 0): -1}


def test_su2_nf2_spin4_recovery():
    """End-to-end: recover the hand-built `su2_nf2` `Spin(4)` trace decomposition
    from its Cartan data — `tr_W(n)` ↦ checker must equal `tr_W_su2xsu2(n)`,
    including the virtual (signed) Wilson-line content."""
    import su2_nf2_h_trace as T2

    def lr_to_orth(a, b):
        return (F(a + b, 2), F(a - b, 2))

    def orth_to_lr(lam):
        return (lam[0] + lam[1], lam[0] - lam[1])     # SU(2)_L × SU(2)_R Dynkin

    QMAX, TRUST = 10, 6
    for n in (0, 2):
        cartan = T2.tr_W(n, q_max=QMAX)               # RLaurent over (q; μ_1, μ_2)
        gt = T2.tr_W_su2xsu2(n, q_max=QMAX)           # RLaurent over R(SU2×SU2)
        for qe, rc in cartan.coeffs.items():
            if qe > TRUST:
                continue                              # q-cutoff boundary: incomplete
            poly = {tuple(k): v for k, v in rc.terms.items() if v}
            if not poly:
                continue
            orth = {lr_to_orth(*k): v for k, v in poly.items()}
            enhanced, dec, _ = C.verify_flavour_enhancement(2, orth)
            assert enhanced, (n, qe, "not W(D_2)-invariant")
            mine = {orth_to_lr(k): v for k, v in dec.items()}
            truth = {tuple(k): v for k, v in gt.coeffs.get(qe).terms.items()} \
                if gt.coeffs.get(qe) is not None else {}
            assert mine == {k: v for k, v in truth.items() if v}, (n, qe, mine, truth)
    # the vacuum index W_0 is a *genuine* (non-virtual) Spin(4) rep at each order
    for qe, rc in T2.tr_W(0, q_max=QMAX).coeffs.items():
        if qe > TRUST:
            continue
        poly = {tuple(k): v for k, v in rc.terms.items() if v}
        if not poly:
            continue
        orth = {lr_to_orth(*k): v for k, v in poly.items()}
        _, _, genuine = C.verify_flavour_enhancement(2, orth)
        assert genuine, (0, qe, "vacuum index not a genuine rep")


def test_so2nf_ring_matches_su4():
    """`SO2NfZPlusRing(3) = R(Spin(6))` reproduces `SU4ZPlusRing = R(SU(4))`:
    Clebsch–Gordan, dualities, dims agree under the weight dictionary
    (ω_1=spinor=4, ω_2=vector=6, ω_3=cospinor=4̄)."""
    import itertools
    from zplus_ring import SU4ZPlusRing, RElement

    R = C.SO2NfZPlusRing(3)
    SU4 = SU4ZPlusRing()
    w1, w2, w3 = (F(1, 2),) * 3, (1, 0, 0), (F(1, 2), F(1, 2), F(-1, 2))

    def to_d3(p, q, r):
        return tuple(p * w1[i] + q * w2[i] + r * w3[i] for i in range(3))

    reps = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0), (1, 1, 0)]
    for a in reps:
        for b in reps:
            su4 = {to_d3(*k): v for k, v in SU4.multiply_basis(a, b).items()}
            assert R.multiply_basis(to_d3(*a), to_d3(*b)) == su4, (a, b)
    # duality + involution
    assert R.star_basis(w1) == w3 and R.star_basis(w2) == w2
    assert all(R.star_basis(R.star_basis(to_d3(*x))) == to_d3(*x) for x in reps)
    # associativity
    def re(x):
        return RElement(R, {to_d3(*x): 1})
    a, b, c = (1, 0, 0), (1, 0, 0), (0, 0, 1)
    assert (re(a) * re(b)) * re(c) == re(a) * (re(b) * re(c))


def _orbit_to_orth_d3():
    """Linear map from `su2_nf3`'s 'fundamental-orbit' Cartan basis (the `w_k`
    weights) to the orthogonal `D_3` `e_i` basis: the three `±(w_i+w_j)` pairs
    (the vector `6_v`) map to `±e_a`."""
    cols = [(0, 1, 0), (1, 0, -1), (1, -1, 1)]      # these orbit-vectors → e_1,e_2,e_3
    A = [[F(cols[j][i]) for j in range(3)] for i in range(3)]
    aug = [[A[i][j] for j in range(3)]
           + [F(1) if i == k else F(0) for k in range(3)] for i in range(3)]
    for c in range(3):
        p = next(r for r in range(c, 3) if aug[r][c] != 0)
        aug[c], aug[p] = aug[p], aug[c]
        pv = aug[c][c]
        aug[c] = [x / pv for x in aug[c]]
        for r in range(3):
            if r != c and aug[r][c] != 0:
                f = aug[r][c]
                aug[r] = [aug[r][k] - f * aug[c][k] for k in range(6)]
    M = [[aug[i][3 + j] for j in range(3)] for i in range(3)]
    return lambda w: tuple(sum(M[i][j] * w[j] for j in range(3)) for i in range(3))


def test_su2_nf3_su4_recovery():
    """End-to-end at Nf=3 (Spin(6)=SU(4), the spinor sector): the live
    `su2_nf3` index is recognized as SU(4)-enhanced — the vacuum `W_0` is a
    genuine SU(4) rep at each order, and `Tr(W_1)[q¹] = −χ_{6_v}` (the
    documented value, the vector = SU(4) Dynkin (0,1,0))."""
    import su2_nf3_h_trace as T3
    to_orth = _orbit_to_orth_d3()

    def d3_to_su4(lam):
        p = lam[1] + lam[2]
        r = lam[1] - lam[2]
        q = lam[0] - p / 2 - r / 2
        return (p, q, r)

    # vacuum index: genuine SU(4) content at every (trusted) q-order
    for qe, rc in T3.tr_W(0, q_max=6).coeffs.items():
        if qe > 4:
            continue
        poly = {to_orth(tuple(k)): v for k, v in rc.terms.items() if v}
        if not poly:
            continue
        enhanced, _, genuine = C.verify_flavour_enhancement(3, poly)
        assert enhanced and genuine, (qe, "vacuum not genuine SU(4)")

    # Tr(W_1)[q^1] = -6_v  (SU(4) Dynkin (0,1,0)) — documented in su2_nf3_h_trace
    rc1 = T3.tr_W(1, q_max=4).coeffs[1]
    poly = {to_orth(tuple(k)): v for k, v in rc1.terms.items() if v}
    enhanced, dec, _ = C.verify_flavour_enhancement(3, poly)
    assert enhanced
    assert {d3_to_su4(k): v for k, v in dec.items()} == {(0, 1, 0): -1}


def test_spin8_triality():
    """Spin(8) triality: the S_3 outer automorphism permutes the three 8s
    (8v, 8s, 8c), fixes the adjoint 28, and is a *ring automorphism*
    (Clebsch–Gordan is triality-covariant)."""
    R = C.SO2NfZPlusRing(4)
    s, t = C.triality_d4_generators()
    vec, spin, cospin = (1, 0, 0, 0), (F(1, 2),) * 4, (F(1, 2), F(1, 2), F(1, 2), F(-1, 2))
    adj = (1, 1, 0, 0)
    # the three 8s are one S_3 orbit
    orbit = set()
    for word in ([], [s], [t], [s, t], [t, s], [s, t, s]):
        w = vec
        for g in word:
            w = C.triality_act(g, w)
        orbit.add(w)
    assert orbit == {vec, spin, cospin}
    # generators permute the 8s as expected, fix the adjoint
    assert C.triality_act(t, vec) == spin and C.triality_act(t, cospin) == cospin
    assert C.triality_act(s, spin) == cospin and C.triality_act(s, vec) == vec
    assert C.triality_act(s, adj) == adj and C.triality_act(t, adj) == adj
    # the famous products, triality-symmetric
    one, c28, c35v = (0, 0, 0, 0), adj, (2, 0, 0, 0)
    assert R.multiply_basis(vec, vec) == {c35v: 1, c28: 1, one: 1}
    assert R.multiply_basis(spin, spin) == {(1, 1, 1, 1): 1, c28: 1, one: 1}
    assert vec in R.multiply_basis(spin, cospin)        # 8s ⊗ 8c ⊇ 8v
    # ring automorphism: CG is triality-covariant
    weights = [vec, spin, cospin, adj, c35v, (1, 1, 1, 1), (1, 1, 1, -1), one]
    assert C.is_triality_automorphism(R, s, weights)
    assert C.is_triality_automorphism(R, t, weights)


def test_spin8_products():
    """Explicit Spin(8) tensor products (textbook decompositions), exercising
    the vector/spinor/cospinor sectors and triality symmetry."""
    R = C.SO2NfZPlusRing(4)
    from zplus_ring import RElement
    # named irreps by highest weight (orthogonal D_4 basis)
    one = (0, 0, 0, 0)
    v8, s8, c8 = (1, 0, 0, 0), (F(1, 2),) * 4, (F(1, 2), F(1, 2), F(1, 2), F(-1, 2))
    a28 = (1, 1, 0, 0)
    g35v, g35s, g35c = (2, 0, 0, 0), (1, 1, 1, 1), (1, 1, 1, -1)
    v56, s56, c56 = (1, 1, 1, 0), (F(3, 2), F(1, 2), F(1, 2), F(1, 2)), (F(3, 2), F(1, 2), F(1, 2), F(-1, 2))
    v160, v112 = (2, 1, 0, 0), (3, 0, 0, 0)
    r300, r350 = (2, 2, 0, 0), (2, 1, 1, 0)

    # the three 8s ⊗ themselves: 1 ⊕ 28 ⊕ 35_x  (triality-related 35s)
    assert R.multiply_basis(v8, v8) == {one: 1, a28: 1, g35v: 1}
    assert R.multiply_basis(s8, s8) == {one: 1, a28: 1, g35s: 1}
    assert R.multiply_basis(c8, c8) == {one: 1, a28: 1, g35c: 1}
    # two different 8s ⊗ → the third 8 ⊕ a 56
    assert R.multiply_basis(v8, s8) == {c8: 1, s56: 1}
    assert R.multiply_basis(v8, c8) == {s8: 1, c56: 1}
    assert R.multiply_basis(s8, c8) == {v8: 1, v56: 1}
    # adjoint ⊗ vector = 8v ⊕ 56v ⊕ 160v
    assert R.multiply_basis(a28, v8) == {v8: 1, v56: 1, v160: 1}
    # adjoint ⊗ adjoint — the fully triality-symmetric one (all three 35s)
    assert R.multiply_basis(a28, a28) == {
        one: 1, a28: 1, g35v: 1, g35s: 1, g35c: 1, r300: 1, r350: 1}
    # 35v ⊗ 8v = 8v ⊕ 160v ⊕ 112v
    assert R.multiply_basis(g35v, v8) == {v8: 1, v160: 1, v112: 1}
    # triple vector 8v⊗8v⊗8v = 3·8v ⊕ 2·160v ⊕ 112v ⊕ 56v
    triple = RElement(R, {v8: 1}) * RElement(R, {v8: 1}) * RElement(R, {v8: 1})
    assert triple.terms == {v8: 3, v160: 2, v112: 1, v56: 1}

    # every decomposition conserves dimension
    for a, b in [(v8, v8), (s8, s8), (v8, s8), (a28, v8), (a28, a28), (g35v, v8)]:
        prod = R.multiply_basis(a, b)
        assert sum(m * C.dim(4, lam) for lam, m in prod.items()) \
            == C.dim(4, a) * C.dim(4, b), (a, b)


def test_so2nf_ring_even_selfdual():
    """Even `Nf` ⇒ `w_0 = -1` ⇒ every irrep self-dual (Spin(4), Spin(8))."""
    for nf in (2, 4):
        R = C.SO2NfZPlusRing(nf)
        for lam in [(1,) + (0,) * (nf - 1), (F(1, 2),) * nf]:
            assert R.star_basis(lam) == lam


if __name__ == "__main__":
    test_dimensions(); print("dimensions: OK")
    test_tensor_product(); print("tensor product 6⊗6: OK")
    test_decompose_roundtrip(); print("decompose round-trip + checker: OK")
    test_checker_rejects_broken_and_flags_virtual(); print("broken/virtual flags: OK")
    test_su2_nf2_spin4_recovery(); print("su2_nf2 Spin(4) recovery from Cartan: OK")
    test_so2nf_ring_matches_su4(); print("SO2NfZPlusRing(3) == R(SU(4)): OK")
    test_su2_nf3_su4_recovery(); print("su2_nf3 SU(4) recovery from Cartan: OK")
    test_spin8_triality(); print("Spin(8) triality (S_3 ring automorphism): OK")
    test_spin8_products(); print("Spin(8) tensor products (incl. 28⊗28 triality): OK")
    test_so2nf_ring_even_selfdual(); print("even-Nf self-duality: OK")
    print("\nAll so2nf_characters tests passed.")
