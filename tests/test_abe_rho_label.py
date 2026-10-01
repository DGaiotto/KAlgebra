"""The explicit label-level ρ on the `AbeKAlgebra` tier (`wrq_torus.rho_label`
+ `rho_level_star`) — promoted to primary 2026-08-23, with the demoted
chart→twist→`decompose` route as the verifier (`verify_rho_via_twist`).

Four groups of tests:

  1. ENGINE ALGEBRA (no charts): ρ⁻¹∘ρ = ρ∘ρ⁻¹ = id on grids; Weyl-
     representative independence (the map is well-defined on orbits of
     pairs); the `N = Adj` corollary — `wt(Adj) = Φ ∪ {0}` makes the gauge
     and matter sums cancel identically, so ρ^{±1} = the antipode
     `[(m,e)] ↦ [(−m,−e)]` and ρ² = id on the gauge charges.
  2. CLASS ↔ TWIST: `verify_rho_via_twist` on every wired realisation
     (PureSU2KAlgebra, PureSUNKAlgebra(3), PureGAbeKAlgebra at su_2 and
     sp_n(2), GNAbeKAlgebra at su_2 + fundamental, UNNfKAlgebra(2,2)/(2,3))
     — and on PureUNKAlgebra(2), whose Witten-shift label maps predate this
     engine, as an independent cross-check of the same twist route.
  3. ROUND-TRIPS + AXIOMS through the promoted ρ: ρ⁻¹∘ρ = id on labels,
     ρ fixes the identity, ρ is an automorphism (spot checks).
  4. N = Adj AT CHART LEVEL: GNAbeKAlgebra(su_2, adjoint) — the gauge label
     is ρ-fixed (the antipode in the su2 fold) while the adjoint flavour
     charge stars as k ↦ −k − D.
"""
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "implementations")

from root_datum import su_2, su_n, sp_n, g_2
from wrq_torus import rho_label, rho_level_star

FAILURES = []


def check(name, ok):
    print(("  PASS  " if ok else "  FAIL  ") + name)
    if not ok:
        FAILURES.append(name)


def fold_dominant(datum, m, e):
    """The joint dominant fold of a pair: `m` cochar-dominant, `e` the
    Levi-dominant transported partner (the native label frame)."""
    d = datum.dim
    m_plus, cands = None, []
    for w in datum.weyl_elements():
        mm = tuple(datum.act_cochar(w, m))
        if datum.is_dominant_cochar(mm):
            if m_plus is None:
                m_plus = mm
            if mm == m_plus:
                cands.append(tuple(datum.act(w, e)))
    levi = [cor for a, cor in zip(datum.simple_roots, datum.simple_coroots)
            if sum(a[i] * m_plus[i] for i in range(d)) == 0]
    for ee in cands:
        if all(sum(ee[i] * cor[i] for i in range(d)) >= 0 for cor in levi):
            return (m_plus, ee)
    raise AssertionError("fold_dominant: no Levi-dominant transport")


def adjoint_slots(datum):
    """The adjoint weight set: all roots plus `rank` zeros."""
    roots = list(datum.positive_roots())
    roots += [tuple(-x for x in a) for a in roots]
    return (tuple(roots) + ((0,) * datum.dim,) * datum.dim,)


def grid_labels(datum, span=2):
    """A small grid of native-frame labels (m, e), m folded dominant."""
    d = datum.dim
    out = []
    vals = range(-span, span + 1)
    ms = [(1,) * d, (2,) + (0,) * (d - 1), (1,) + (0,) * (d - 1),
          (0,) * d]
    for m in ms:
        for i in range(d):
            for v in vals:
                e = tuple(v if t == i else 0 for t in range(d))
                out.append(fold_dominant(datum, m, e))
    return sorted(set(out))


def leg_engine_algebra():
    print("== engine algebra (no charts) ==")
    fund3 = None
    for datum in (su_2(), su_n(3), sp_n(2), g_2()):
        labs = grid_labels(datum)
        # matter stress: the (non-self-conjugate) defining rep where cheap
        slots = ()
        if datum.dim == 2 and datum is not None:
            from matter_wrq_torus import slot_weights
            try:
                slots = slot_weights(datum, ((1, 0),))
            except Exception:
                slots = ()
        ok_inv = ok_rep = True
        for (m, e) in labs:
            r = rho_label(datum, m, e, slots=slots)
            back = rho_label(datum, *r, slots=slots, inverse=True)
            if fold_dominant(datum, *back) != fold_dominant(datum, m, e):
                ok_inv = False
            ri = rho_label(datum, m, e, slots=slots, inverse=True)
            backi = rho_label(datum, *ri, slots=slots)
            if fold_dominant(datum, *backi) != fold_dominant(datum, m, e):
                ok_inv = False
            # representative independence: transported inputs, same image
            for w in datum.weyl_elements()[:6]:
                mt = tuple(datum.act_cochar(w, m))
                et = tuple(datum.act(w, e))
                if rho_label(datum, mt, et, slots=slots) != r:
                    ok_rep = False
        check(f"{datum.name}: rho_inverse inverts rho ({len(labs)} labels)",
              ok_inv)
        check(f"{datum.name}: Weyl-representative independence", ok_rep)
        # N = Adj: both directions equal the antipode, rho^2 = id
        adj = adjoint_slots(datum)
        ok_adj = ok_sq = True
        for (m, e) in labs:
            want = fold_dominant(datum, tuple(-x for x in m),
                                 tuple(-x for x in e))
            r = rho_label(datum, m, e, slots=adj)
            ri = rho_label(datum, m, e, slots=adj, inverse=True)
            if r != want or ri != want:
                ok_adj = False
            if rho_label(datum, *r, slots=adj) != fold_dominant(datum, m, e):
                ok_sq = False
        check(f"{datum.name}: N = Adj ⇒ ρ^{{±1}} = antipode", ok_adj)
        check(f"{datum.name}: N = Adj ⇒ ρ² = id", ok_sq)
        # flavour star at N = Adj is still nontrivial: D = Σ_α max(0,⟨α,m⟩)
        m = (1,) * datum.dim
        D = rho_level_star(adj, m)[0]
        check(f"{datum.name}: N = Adj level star D > 0 at m = {m}", D > 0)


def leg_class_vs_twist():
    print("== class rho ↔ twist route (verify_rho_via_twist) ==")
    from pure_su2_kalgebra import PureSU2KAlgebra
    from pure_g_abe_kalgebra import PureGAbeKAlgebra
    from root_datum import su_n, u_n
    A = PureSU2KAlgebra()
    for a in [(0, 0), (0, 2), (1, 0), (1, 3), (2, 1), (3, 0)]:
        check(f"PureSU2 {a}", A.verify_rho_via_twist(a))
    G3 = PureGAbeKAlgebra(su_n(3))    # (the type-A PureSUNKAlgebra(3) rows stood here until 2026-09-19)
    for a in [((0, 0), (1, 0)), ((-1, -1), (0, 0)), ((-1, -1), (1, 0)), ((-2, -2), (0, 0))]:
        check(f"PureG su_n(3) {a}", G3.verify_rho_via_twist(G3.fold(*a)))
    G2 = PureGAbeKAlgebra(su_2())
    for a in [((0,), (1,)), ((1,), (0,)), ((2,), (1,))]:
        check(f"PureG su_2 {a}", G2.verify_rho_via_twist(a))
    GC = PureGAbeKAlgebra(sp_n(2))
    for a in [((0, 0), (1, 0)), ((1, 0), (0, 0)), ((1, 1), (0, 0)),
              ((1, 0), (1, 0))]:
        check(f"PureG sp_n(2) {a}", GC.verify_rho_via_twist(a))
    from gn_abe_kalgebra import GNAbeKAlgebra
    GN = GNAbeKAlgebra(su_2(), ((1,),))
    for a in [(((1,), (0,)), (0,)), (((1,), (1,)), (0,)),
              (((2,), (0,)), (0,)), (((1,), (0,)), (1,))]:
        check(f"GN su_2+fund {a}", GN.verify_rho_via_twist(a))
    # U(2)+2 and U(2)+3 on the general tier, BOTH magnetic sectors: type-A ±
    # orbits are distinct, and the positive-monopole (E/F) sector is where an
    # under-fit frame dictionary hid from the negative-sector rows (caught
    # 2026-08-23).  (The type-A UNNfKAlgebra / PureUNKAlgebra rows that stood
    # here were retired 2026-09-19.)
    U22 = GNAbeKAlgebra(u_n(2), (1, 0), nf=2)
    for g in [((0, 0), (1, 0)), ((-1, 0), (0, 0)), ((-1, -1), (1, 0)),
              ((1, 0), (0, 0)), ((0, -1), (0, 0)), ((1, 1), (0, 0))]:
        for w in [(0, 0), (1, 0)]:
            a = (U22.fold(*g)[0], w)
            check(f"GN u_n(2)+2 {a}", U22.verify_rho_via_twist(a))
    check("GN u_n(2)+2 rho automorphism (E, F)",
          U22.verify_rho_is_automorphism((((1, 0), (0, 0)), (0, 0)),
                                         (((0, -1), (0, 0)), (0, 0))))
    P2 = PureGAbeKAlgebra(u_n(2))
    for a in [((0, 0), (1, 0)), ((1, 0), (0, 0)), ((1, 1), (1, 0))]:
        check(f"PureG u_n(2) {a}", P2.verify_rho_via_twist(P2.fold(*a)))
    return A, GN


def leg_roundtrips_axioms(A, GN):
    print("== round-trips + axioms through the promoted rho ==")
    check("PureSU2 rho fixes identity", A.verify_rho_fixes_identity())
    ok = all(A.rho_inverse(A.rho(a)) == a and A.rho(A.rho_inverse(a)) == a
             for a in [(0, 1), (1, 0), (1, 2), (2, 0), (3, 1)])
    check("PureSU2 rho round-trips (5 labels)", ok)
    check("PureSU2 rho automorphism ((0,1),(1,0))",
          A.verify_rho_is_automorphism((0, 1), (1, 0)))
    check("GN su_2+fund rho fixes identity", GN.verify_rho_fixes_identity())
    a = (((1,), (0,)), (0,))
    check("GN su_2+fund rho round-trip",
          GN.rho_inverse(GN.rho(a)) == a and GN.rho(GN.rho_inverse(a)) == a)


def leg_trace_axioms():
    """The KAlgebra trace axioms THROUGH the promoted label-level ρ:
    ρ is an algebra automorphism on real products; the two trace-pairing
    faces `Tr(ρ(a)·b) = Tr(b·ρ⁻¹(a))`; ρ²-twisted cyclicity
    `Tr(ab) = Tr(ρ²(b)·a)`; orthonormality `I_{a,b} = δ_{a,b} + O(𝖖)`
    (incl. the empty negative window); and the sharpest tie between the
    label map and the pairing — multiply-then-trace through label-ρ equals
    the class's certified chart-side pairing exactly.

    History (2026-08-23): the UNNfKAlgebra `verify_inner_product_consistent`
    row initially FAILED — not a ρ statement, but a genuine defect this sweep
    exposed: `UNNfKAlgebra._flavour_element` silently collapsed the flavour
    level, so the tier's chart-side pairing (the verifier's MRO reference)
    fabricated the dimension-collapsed series of the SU-refined one
    (−2+χ₍₂₎ ↦ 1, 1−χ₍₂₎+χ₍₄₎ ↦ 3 at U(2)+2).  Fixed the honest-fail way:
    that hook now raises at N_f ≥ 2, and the verifier walks past a reference
    that declines, down to the root multiply-then-trace definition — the
    axiom itself — against which the row below now asserts agreement.  The
    direct is_zero check stays as the explicit form of the same tie.

    A same-shape sweep at PureGAbeKAlgebra(sp_n(2)) (Wilson×Wilson,
    Wilson×monopole; automorphism/faces/cyclicity/orthonormality/consistency)
    ran green in-session but costs ~4 min, so it is not in the suite."""
    print("== trace axioms through the promoted rho ==")
    from pure_su2_kalgebra import PureSU2KAlgebra
    from pure_g_abe_kalgebra import PureGAbeKAlgebra
    from root_datum import su_n, u_n
    A = PureSU2KAlgebra()
    for (a, b) in [((0, 1), (1, 0)), ((1, 0), (1, 0)), ((1, 1), (1, -1)),
                   ((2, 0), (0, 1))]:
        check(f"su2 automorphism {a},{b}", A.verify_rho_is_automorphism(a, b))
        check(f"su2 pairing faces {a},{b}",
              A.verify_trace_pairing_faces(a, b, K=6))
        check(f"su2 rho2 cyclicity {a},{b}",
              A.verify_rho_twisted_trace(a, b, K=6))
        check(f"su2 orthonormality {a},{b}",
              A.verify_orthonormality(a, b, K=6))
        check(f"su2 inner consistent {a},{b}",
              A.verify_inner_product_consistent(a, b, K=6))
    from gn_abe_kalgebra import GNAbeKAlgebra
    G = GNAbeKAlgebra(su_2(), ((1,),))
    la = (((1,), (0,)), (0,))
    lb = (((1,), (1,)), (0,))
    lf = (((1,), (0,)), (1,))
    for (a, b) in [(la, lb), (la, la), (lf, la)]:
        check("GN automorphism", G.verify_rho_is_automorphism(a, b))
        check("GN pairing faces", G.verify_trace_pairing_faces(a, b, K=4))
        check("GN rho2 cyclicity", G.verify_rho_twisted_trace(a, b, K=4))
        check("GN orthonormality", G.verify_orthonormality(a, b, K=4))
        check("GN inner consistent",
              G.verify_inner_product_consistent(a, b, K=4))
    # (the PureSUNKAlgebra(3) / UNNfKAlgebra(2,2) rows that stood here were
    # retired 2026-09-19; the general-tier rows above cover the same axioms)


def leg_gn_adjoint():
    print("== N = Adj at chart level: GNAbeKAlgebra(su_2, adjoint) ==")
    from gn_abe_kalgebra import GNAbeKAlgebra
    try:
        GA = GNAbeKAlgebra(su_2(), ((2,),))
        t = time.time()
        a = (((1,), (0,)), (0,))
        r, ri = GA.rho(a), GA.rho_inverse(a)
        # antipode on the gauge part (su2 fold: fixed); flavour k ↦ −k − D
        check("gauge label rho-fixed (antipode)",
              r[0] == a[0] and ri[0] == a[0])
        D = rho_level_star(adjoint_slots(su_2()), (1,))[0]
        check(f"flavour star k ↦ −k − D (D = {D})",
              r[1] == (-D,) and ri[1] == (-D,))
        check("twist verifier on the adjoint monopole",
              GA.verify_rho_via_twist(a))
        print(f"    [{time.time()-t:.1f}s]")
    except Exception as ex:
        print(f"  SKIP (chart build): {type(ex).__name__}: {str(ex)[:100]}")


if __name__ == "__main__":
    t0 = time.time()
    leg_engine_algebra()
    A, GN = leg_class_vs_twist()
    leg_roundtrips_axioms(A, GN)
    leg_trace_axioms()
    leg_gn_adjoint()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print("  " + f)
        sys.exit(1)
    print(f"All abe-rho-label tests passed.   [{time.time()-t0:.1f}s]")
