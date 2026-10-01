"""Certification battery for `PureGAbeKAlgebra` — pure gauge theory at a general
`G` on the `WRQTorus`, built through the (★)-guarded route (`star_bubbling`).

The load-bearing rows are the **independent-route agreements**: at `su_2()` the
solved chart must equal `PureSU2KAlgebra`'s fully-native constructive build, and
at `su_n(3)` it must equal the `PureSUNKAlgebra` ground truth.  Those are the
only rows where a second, non-solving construction exists, so they are what
certifies the solve rather than merely exercising it.  Everything beyond type A
is self-certified by the axioms — (★), W1+W2, bar, ρ, orthonormality — which is
exactly the epistemic status recorded in the docstrings.

Run:  `python3 run_tests.py`
      `python3 run_tests.py` --full   (adds Sp(6)/SO(7))
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "implementations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import star_bubbling as SB
import wrq_torus as W
from pure_g_abe_kalgebra import PureGAbeKAlgebra
from pure_su2_kalgebra import PureSU2KAlgebra
from root_datum import (su_2, su_n, so_n, sp_n, g_2, b_n_simply_connected,
                        u_n, _positive_roots_from_simple)

_PASS = 0
_FAIL = []


def check(name, cond, detail=""):
    global _PASS
    if cond:
        _PASS += 1
        print(f"  PASS: {name}")
    else:
        _FAIL.append(name)
        print(f"  FAIL: {name}  {detail}")


# ---------------------------------------------------------------------------
def test_root_data():
    print("\n[root data — the general-G factories]")
    # The generic positive-root closure must agree with the type-A fast path.
    for N in (3, 4, 5):
        d = su_n(N)
        gen = _positive_roots_from_simple(d.simple_roots, d.simple_coroots, d.dim)
        check(f"su_n({N}) generic closure == type-A default",
              sorted(gen) == sorted(d.positive_roots()))
    for N in (3, 4):
        d = u_n(N)
        gen = _positive_roots_from_simple(d.simple_roots, d.simple_coroots, d.dim)
        check(f"u_n({N}) generic closure == type-A default",
              sorted(gen) == sorted(d.positive_roots()))

    # Weyl orders: |W(B_n)| = 2^n n!, |W(C_n)| = 2^n n!, |W(D_n)| = 2^{n-1} n!,
    # |W(G_2)| = 12.
    for dat, order in [(b_n_simply_connected(2), 8), (b_n_simply_connected(3), 48),
                       (sp_n(2), 8), (sp_n(3), 48), (so_n(5), 8), (so_n(7), 48),
                       (so_n(8), 192), (g_2(), 12)]:
        check(f"|W({dat.name})| == {order}",
              len(dat.weyl_elements()) == order,
              f"got {len(dat.weyl_elements())}")

    # `b_n_simply_connected(2)` must reproduce the datum solved B2 on.
    b2 = b_n_simply_connected(2)
    check("Spin(5) simple roots == the B2 datum",
          b2.simple_roots == [(2, -2), (-1, 2)], f"got {b2.simple_roots}")
    check("Spin(5) positive roots == the B2 datum",
          sorted(b2.positive_roots()) == sorted([(2, -2), (-1, 2), (1, 0), (0, 2)]))
    g2 = g_2()
    check("G2 positive roots == the G2 datum",
          sorted(g2.positive_roots()) == sorted(
              [(2, -1), (-3, 2), (-1, 1), (1, 0), (3, -1), (0, 1)]))

    # Spin(5) has NO minuscule cocharacter — the reason the (★) route exists.
    check("Spin(5) has no minuscule cocharacter (|m_i| <= 3)",
          not any(W.is_minuscule(b2, (a, b))
                  for a in range(-3, 4) for b in range(-3, 4) if (a, b) != (0, 0)))


def test_label_frame():
    print("\n[label frame — fold vs what well_formed reports]")
    for dat, labels in [(b_n_simply_connected(2),
                         [((1, 0), (0, 0)), ((0, 1), (0, 0)), ((1, 1), (0, 0))]),
                        (g_2(), [((1, 0), (0, 0)), ((0, 1), (0, 0))])]:
        A = PureGAbeKAlgebra(dat)
        for (m, e) in labels:
            lab = A.fold(m, e)
            # the frame is defined by the substrate's own read
            wf = W.leading_orbit(dat, m, e).well_formed()
            check(f"{dat.name} fold{m,e} == well_formed read {wf}", lab == wf,
                  f"fold gave {lab}")
            check(f"{dat.name} fold idempotent at {lab}", A.fold(*lab) == lab)


def test_su2_against_native():
    """The strongest row: the (★) SOLVE vs a fully-native constructive build."""
    print("\n[SU(2): (★) solve == PureSU2KAlgebra native build]")
    A = PureGAbeKAlgebra(su_2())
    N = PureSU2KAlgebra()
    for (m, e) in [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3)]:
        lab = A.fold((m,), (e,))
        try:
            got = A.chart(lab)
        except Exception as ex:                       # pragma: no cover
            check(f"su_2 L_{(m, e)} builds", False, f"{type(ex).__name__}: {ex}")
            continue
        want = N.chart((m, e))
        same = (got + W._scale(want, W.LaurentPoly({0: -1}))).is_zero()
        check(f"su_2 L_{(m, e)} == PureSU2KAlgebra chart (route {A.route(lab)})",
              same)


def test_su3_against_oracle():
    """SU(3): the three labels that the type-A `PureSUNKAlgebra(3)` oracle
    certified 3/3 (2026-07-27) — since that class's retirement (2026-09-19;
    its own transported chart was WRONG at dressed 2·adjoint labels, A75) the
    load-bearing check here is the one the review named: orthonormality of
    the (★)-built charts against each other and the identity, plus the W1+W2
    acceptance.  Chart agreement alone cannot see an off-span chart (the (★)
    guard, W1 and `certify_canonical` all accepted the A75 chart)."""
    print("\n[SU(3): (★)-built charts — orthonormality + acceptance at the labels]")
    A = PureGAbeKAlgebra(su_n(3))
    labs = [A.fold((-1, -1), (0, 0)), A.fold((-1, -1), (1, 0)), A.fold((-1, -1), (0, 1))]
    for lab in labs:
        check(f"su_3 L_{lab} certify_canonical", bool(A.certify_canonical(lab)))
    basket = [A.identity()] + labs
    for a in basket:
        for b in basket:
            check(f"su_3 orthonormality {a} , {b}", A.verify_orthonormality(a, b, K=4))


def test_un_against_keystone():
    """Type A: the axiom route against the retired constructive zoo.

    Until 2026-09-19 this row also compared both against the `PureUNKAlgebra`
    keystone (equal on every label, 411/411 in the retirement audit); with that
    class in the source repository's archive the two independent constructions left are the (★)
    solve and the non-solving zoo behind `constructive_routes=True`."""
    print("\n[U(2): axioms == the retired constructive zoo (no solve at type A)]")
    A = PureGAbeKAlgebra(u_n(2))                            # axioms only
    Z = PureGAbeKAlgebra(u_n(2), constructive_routes=True)   # the retired zoo
    for lab in [((0, 0), (1, 0)), ((1, 0), (0, 0)), ((1, 1), (0, 0)),
                ((1, 0), (1, 0)), ((2, 0), (0, 0))]:
        f = A.fold(*lab)
        try:
            got, route = A.chart(f), A.route(f)
        except Exception as ex:
            check(f"u_2 L_{lab} axiom route", False,
                  f"{type(ex).__name__}: {str(ex)[:100]}")
            continue
        want = got      # (the PureUNKAlgebra keystone this compared against was retired 2026-09-19; 411/411 recorded)
        try:
            zgot, zroute = Z.chart(f), Z.route(f)
        except Exception as ex:
            check(f"u_2 L_{lab} zoo route", False,
                  f"{type(ex).__name__}: {str(ex)[:100]}")
            continue
        check(f"u_2 L_{lab}: the retired zoo needs no solve at type A",
              zroute != "star", f"zoo route was {zroute}")
        check(f"u_2 L_{lab}: axioms == zoo (independent constructions)",
              (zgot + W._scale(want, W.LaurentPoly({0: -1}))).is_zero())


def _axiom_sweep(dat, labels, tag, K=4, do_multiply=False):
    print(f"\n[{tag} — axioms on the (★)-built charts]")
    A = PureGAbeKAlgebra(dat)
    built = []
    for (m, e) in labels:
        lab = A.fold(m, e)
        try:
            x = A.chart(lab)
        except Exception as ex:
            check(f"{tag} L_{lab} builds", False,
                  f"{type(ex).__name__}: {str(ex)[:110]}")
            continue
        built.append(lab)
        check(f"{tag} L_{lab} W1 (bar-invariant)", bool(x.well_formed_w1()))
        check(f"{tag} L_{lab} certify_canonical (W1+W2) == label",
              A.certify_canonical(lab) == lab,
              f"got {A.certify_canonical(lab)}")
        ok, rep = SB.criterion(x)
        check(f"{tag} L_{lab} satisfies (★) (acts on R(G))", ok, str(rep[:2]))
        # ρ is a basis permutation, and ρ⁻¹ inverts it.  Checked at the ELEMENT
        # level: `A.rho_inverse(A.rho(lab))` would rebuild the ρ-image chart,
        # which can be far more expensive than the original (measured at G₂:
        # ρ(L_{(2,3),(−1,1)}) = L_{(2,3),(−13,1)}, whose Wilson anchor is huge),
        # while the substrate round-trip needs no rebuild at all and says the
        # same thing about ρ being invertible on this chart.
        try:
            r = A.rho(lab)
            back = x.rho().rho_inverse()
            same = (back + W._scale(x, W.LaurentPoly({0: -1}))).is_zero()
            check(f"{tag} ρ⁻¹ρ = id on the chart at {lab} (ρ ↦ {r})", same)
        except Exception as ex:
            check(f"{tag} ρ at {lab}", False, f"{type(ex).__name__}: {str(ex)[:90]}")
        # Orthonormality: I_{a,a} = 1 + O(𝖖).
        try:
            I = A.inner_product(lab, lab, K)
            check(f"{tag} orthonormality I_{{{lab},{lab}}}[q^0] == 1",
                  I.coeffs.get(0, 0) == 1, f"I = {I}")
        except Exception as ex:
            check(f"{tag} orthonormality at {lab}", False,
                  f"{type(ex).__name__}: {str(ex)[:90]}")
    # off-diagonal orthogonality on the first two DISTINCT built labels.  The
    # dedup matters: several spellings can fold to one label (at `sp_n(2)`,
    # `m=(0,1)` is not cochar-dominant and folds onto `m=(1,0)`), and comparing
    # a label with itself would read as an orthogonality failure.
    built = list(dict.fromkeys(built))
    if len(built) >= 2:
        a, b = built[0], built[1]
        try:
            I = A.inner_product(a, b, K)
            check(f"{tag} off-diagonal I_{{{a},{b}}}[q^0] == 0",
                  I.coeffs.get(0, 0) == 0, f"I = {I}")
        except Exception as ex:
            check(f"{tag} off-diagonal", False,
                  f"{type(ex).__name__}: {str(ex)[:90]}")
    if do_multiply and built:
        a = built[0]
        try:
            prod = A.multiply(a, a)
            check(f"{tag} multiply({a},{a}) closes in the basis",
                  len(prod.terms) > 0)
            # bar involution: C^c_{ab}(q^-1) == C^c_{ba}(q)
            check(f"{tag} bar involution at ({a},{a})",
                  A.verify_bar_involution(a, a))
        except Exception as ex:
            check(f"{tag} multiply at {a}", False,
                  f"{type(ex).__name__}: {str(ex)[:110]}")
    return A


def test_global_form():
    """The 4d global form = a SUBLATTICE of Kapustin `(m, e)` labels, not a choice of cocharacter lattice — the latter is the 3d
    reading.  The rows below check the datum-general `LineLattice` against
    facts the repo already records independently."""
    print("\n[global form — the Kapustin (m,e) line sublattice]")
    from global_form import (centre_invariants, simply_connected_lines,
                             adjoint_lines)
    # Centres, from the Smith invariants of the Cartan matrix.
    for dat, inv in [(su_2(), (2,)), (su_n(3), (3,)), (su_n(4), (4,)),
                     (b_n_simply_connected(2), (2,)), (sp_n(2), (2,)),
                     (g_2(), ())]:
        check(f"centre of {dat.name} == {inv or 'trivial'}",
              centre_invariants(dat) == inv, f"got {centre_invariants(dat)}")

    # the design notes: SO(3)=PSU(2) is the S-dual form on the SAME
    # su(2) line algebra; its minimal Wilson is the ADJOINT, and SU(2)'s
    # fundamental w_1 is NOT an SO(3) line.  Reproduced independently here.
    adj = adjoint_lines(su_2())
    check("SU(2)-adjoint form REJECTS the fundamental Wilson e=(1,)",
          not adj.elec_admits((1,)))
    check("SU(2)-adjoint form admits the adjoint Wilson e=(2,)",
          adj.elec_admits((2,)))
    sc = simply_connected_lines(su_2())
    check("simply connected form admits the fundamental Wilson",
          sc.elec_admits((1,)))

    # Trivial centre ⇒ exactly one global form (why G₂ is untouched by this).
    check("G2 has a single global form (trivial centre)",
          adjoint_lines(g_2()).H == () and centre_invariants(g_2()) == ())

    # Mutual locality: the Dirac pairing is integral on the lattice.
    labels = [((1,), (0,)), ((0,), (1,)), ((1,), (1,)), ((2,), (0,))]
    check("simply connected su(2) lines are mutually local",
          sc.verify_mutually_local(labels))
    check("Dirac pairing ⟨(1,0),(0,1)⟩ == 1 (a genuine dyonic pair)",
          sc.dirac(((1,), (0,)), ((0,), (1,))) == 1)

    # The class honours the lattice: a label outside it honest-fails.
    A = PureGAbeKAlgebra(su_2(), lines=adj)
    try:
        A.chart(((0,), (1,)))
        check("PureGAbeKAlgebra rejects a label outside its line lattice",
              False, "built a non-line")
    except NotImplementedError:
        check("PureGAbeKAlgebra rejects a label outside its line lattice", True)

    # Which lines this TIER can hold: ALL of them, — odd
    # `⟨Σ⁺, m⟩` included.  The odd-height boundary was an engine artifact
    # (`_psi_monomial_data` materialising the square root of the measure), and
    # `wrq_torus.cocycle_R` now restores the honest phase through its integral
    # coboundary.  See a probe in the source repository.
    L5 = simply_connected_lines(so_n(5))
    L7 = simply_connected_lines(so_n(7))
    check("so_n(5) m=(1,0) IS abe-representable at odd ⟨Σ⁺,m⟩",
          L5.abe_representable((1, 0)))
    check("so_n(5) m=(1,1) IS abe-representable", L5.abe_representable((1, 1)))
    check("so_n(7) m=(1,0,0) IS abe-representable at odd ⟨Σ⁺,m⟩",
          L7.abe_representable((1, 0, 0)))
    check("so_n(7) m=(1,1,0) IS abe-representable",
          L7.abe_representable((1, 1, 0)))
    check("legal_lines no longer removes odd-⟨Σ⁺,m⟩ lines",
          L5.legal_lines([((1, 0), (0, 0)), ((1, 1), (0, 0))])
          == [((1, 0), (0, 0)), ((1, 1), (0, 0))])
    # Representability is a property of the DATUM's atom phase, not of the
    # parity of ⟨Σ⁺,m⟩: `u_n` supplies its own integer phase (the historical
    # U(N) Σ_j j·m_j, differing from the default by a linear central term), so
    # U(2) at m=(1,0) has ODD ⟨Σ⁺,m⟩ and is nonetheless representable.  Testing
    # the parity instead rejected the whole pure-U(N) keystone.
    LU = simply_connected_lines(u_n(2))
    odd_2rho = sum(u_n(2).shift_pairing((1, 0), a)
                   for a in u_n(2).positive_roots()) % 2 == 1
    check("U(2) m=(1,0) has odd ⟨Σ⁺,m⟩ yet IS abe-representable",
          odd_2rho and LU.abe_representable((1, 0)))
    # every simply connected su(2)/su(3) magnetic charge IS representable
    for dat in (su_2(), su_n(3), g_2(), b_n_simply_connected(2)):
        Ld = simply_connected_lines(dat)
        ms = [tuple(x) for x in
              ([(1,), (2,)] if dat.dim == 1 else [(1, 1), (2, 1)])]
        check(f"{dat.name}: coroot-lattice charges are abe-representable",
              all(Ld.abe_representable(m) for m in ms))
    # the class BUILDS a spinorial label — bar-invariant and orthonormal
    B = PureGAbeKAlgebra(so_n(5))
    try:
        ch = B.chart(((1, 0), (0, 0)))
        ip = B.inner_product(((1, 0), (0, 0)), ((1, 0), (0, 0)), 6)
        check("PureGAbeKAlgebra BUILDS an odd-⟨Σ⁺,m⟩ label, bar-invariantly",
              ch.bar() == ch)
        check("…and it is orthonormal, I = 1 + O(𝖖)",
              str(ip).startswith("1 "), f"{ip}")
    except NotImplementedError as ex:
        check("PureGAbeKAlgebra BUILDS an odd-⟨Σ⁺,m⟩ label, bar-invariantly",
              False, f"honest-failed: {str(ex)[:110]}")


def test_two_presentations():
    """`Spin(5) ≅ Sp(2)`: the SAME theory in `B₂` and `C₂` coordinates.

    The two data are built by completely different routes — `b_n_simply_connected`
    from the Cartan matrix, `sp_n` transcribed from `GaugeDatum.sp` — so agreement
    is a genuine cross-presentation check (one algebra, many presentations in miniature), not a
    tautology.  `Tr(1)` is the chart-independent invariant available without a
    label dictionary: it is the vacuum Schur index, fixed by the root system and
    the measure alone."""
    print("\n[Spin(5) == Sp(2): one theory, two coordinate systems]")
    A = PureGAbeKAlgebra(b_n_simply_connected(2))
    B = PureGAbeKAlgebra(sp_n(2))
    ta = A.trace(A.identity(), 10)
    tb = B.trace(B.identity(), 10)
    check("Spin(5) and Sp(2) agree on Tr(1) through q^10",
          dict(ta.coeffs) == dict(tb.coeffs), f"{ta} vs {tb}")
    # both must see the same (absence of) minuscule cocharacters
    check("neither presentation has a minuscule cocharacter",
          not any(W.is_minuscule(A.datum, (a, b))
                  for a in range(-2, 3) for b in range(-2, 3) if (a, b) != (0, 0))
          and not any(W.is_minuscule(B.datum, (a, b))
                      for a in range(-2, 3) for b in range(-2, 3)
                      if (a, b) != (0, 0)))


def test_rank4_solve():
    """The rank-4 solve — `--full` only (~10 min): unique, W1 + (★).  (Until
    2026-09-19 also compared against the `PureSUNKAlgebra(5)` oracle at the
    bare adjoint, where they agreed; that class is retired.)"""
    print("\n[rank 4: SU(5) (★) solve]")
    dat = su_n(5)
    m, e = (-1, -1, -1, -1), (0, 0, 0, 0)
    st, Ff, _supp, _interior, _K_f = SB.joint_solve(dat, m, e, verbose=False)
    check("SU(5) rank-4 system is uniquely solvable", st == "unique", st)
    if st != "unique":
        return
    seed = {tuple(a): f for a, f in
            W.leading_orbit(dat, m, e).residuals().items()}
    x = SB.build_element(dat, seed, Ff)
    ok, _rep = SB.criterion(x)
    check("SU(5) rank-4 chart is W1 + (★)", bool(x.well_formed_w1()) and ok)
    # (the PureSUNKAlgebra(5) oracle comparison that stood here was retired with
    # that class on 2026-09-19; its bare-adjoint agreement was recorded)


def test_cone_generators():
    """The declared generating set.

    `cone_generators` is a **complete** Hilbert-basis search, not a height cut —
    the bound `Σᵢ⟨Σ⁺, rᵢ⟩` over the primitive extreme rays is a proof bound.
    That matters: the height-cut version undercounted `SU(5)` (6 instead of 14)
    and `Spin(7)` (2 instead of 4), which is exactly the failure mode a cut has
    and a bound does not.

    Certified here two ways, neither of which trusts the search:
    * every returned generator is **indecomposable** in the monoid;
    * the returned set **generates** every dominant cocharacter in a height ball
      (BFS over sums), so nothing is missing.

    Plus the two structural facts the final form rests on: the primitive extreme
    rays must appear (they are indecomposable by construction), and a datum with
    a **central torus** has a non-pointed monoid — `U(2)`'s `det` direction is a
    line, so it raises `NonPointedMonoid` rather than returning a wrong finite
    answer."""
    import itertools

    from pure_g_abe_kalgebra import cone_generators, NonPointedMonoid

    print("\n[cone generators — the declared generating set]")

    def dominant_ball(dat, H, span):
        out = []
        for m in itertools.product(range(0, span + 1), repeat=dat.dim):
            if not any(m) or not dat.is_dominant_cochar(m):
                continue
            if 0 < W._cochar_height(dat, m) <= H:
                out.append(tuple(m))
        return out

    expect = {"SU(2)": 1, "SU(3)": 3, "SU(4)": 6, "SU(5)": 14,
              "Spin(5)": 2, "Spin(7)": 4, "Sp(2)": 2, "Sp(3)": 3, "G2": 2}
    for dat, H, span in [(su_2(), 20, 20), (su_n(3), 20, 10), (su_n(4), 24, 8),
                         (su_n(5), 30, 7), (b_n_simply_connected(2), 20, 10),
                         (b_n_simply_connected(3), 24, 8), (sp_n(2), 20, 10),
                         (sp_n(3), 24, 8), (g_2(), 30, 12)]:
        gens = cone_generators(dat)
        d = dat.dim
        check(f"{dat.name} cone_generators count == {expect[dat.name]}",
              len(gens) == expect[dat.name], f"got {len(gens)}: {gens}")

        ball = dominant_ball(dat, H, span)
        seen = set(ball)
        bad = [g for g in gens
               if any(any(t := tuple(g[i] - a[i] for i in range(d)))
                      and t in seen for a in ball if any(a))]
        check(f"{dat.name} every cone generator is indecomposable",
              not bad, f"decomposable: {bad}")

        reach, frontier = {(0,) * d}, [(0,) * d]
        while frontier:
            nxt = []
            for x in frontier:
                for g in gens:
                    y = tuple(x[i] + g[i] for i in range(d))
                    if y in reach or W._cochar_height(dat, y) > H:
                        continue
                    reach.add(y)
                    nxt.append(y)
            frontier = nxt
        missing = [m for m in ball if m not in reach]
        check(f"{dat.name} cone generators generate all {len(ball)} dominant "
              f"cocharacters with ⟨Σ⁺,m⟩ ≤ {H}",
              not missing, f"missing {missing[:4]}")

    # SU(5)'s primitive extreme rays — the ones the height cut lost.
    rays5 = {(4, 3, 2, 1), (3, 6, 4, 2), (2, 4, 6, 3), (1, 2, 3, 4)}
    got5 = set(cone_generators(su_n(5)))
    check("SU(5) cone generators contain all four primitive extreme rays",
          rays5 <= got5, f"missing {sorted(rays5 - got5)}")

    # A central torus makes the monoid non-pointed: no finite Hilbert basis.
    try:
        cone_generators(u_n(2))
        check("U(2) (central torus) raises NonPointedMonoid", False,
              "returned a finite generating set for a non-pointed monoid")
    except NonPointedMonoid as ex:
        check("U(2) (central torus) raises NonPointedMonoid, reporting the "
              f"det direction {ex.lineality}",
              ex.lineality == ((1, 1),), f"lineality {ex.lineality}")


def test_analytic_laws():
    """The two analytic laws of the design notes §7.

    **(i) The `e ↦ e + k·m̄` symmetry**, the load-bearing one:
    `T^k : f_p ↦ v^{k·p̄}` carries the canonical `L_{m,e}` to the canonical
    `L_{m, e + k·m̄}`.  Certified here as the full acceptance — W1, (★), and
    `well_formed` landing at the *predicted* label — so a whole cone of dressed
    lines is the twisted undressed cone and costs **no solve**.

    The shift is by `m̄`, the Weyl-equivariant image of `m` in the weight
    lattice, **not** by `m`'s raw coordinates.  Two traps, both live:
    * at `SU(2)` the coroot `(1)` maps to the root `(2)`, so `T^1` shifts `e` by
      2, not 1;
    * at a non-simply-laced datum the root-length multipliers are essential
      (`Spin(5)` `[1,2]`, `Sp(2)` `[2,1]`, `G₂` `[3,1]`).  With the naive
      `αᵢ^∨ ↦ αᵢ` the twist breaks Weyl symmetry and `_levi_decompose` rejects
      it outright — and simply-laced data give all-ones, so the bug is invisible
      wherever it is cheapest to test.

    **(ii) The one-step bubbling law**, a CONJECTURE pinned with its measured
    scope: exact at the non-deepest one-step cells of `SU(2)`/`SU(3)`/`Spin(5)`/
    `Sp(2)`, and **refuted at `G₂`** (both `m=(2,3)` cells) and at the deepest
    cell of any rank ≥ 2.  The test asserts both the holds and the failures, so
    the recorded scope cannot silently drift."""
    from pure_g_abe_kalgebra import cone_generators

    print("\n[analytic laws — the e↦e+k·m̄ twist and the one-step form]")

    for dat, mult in [(su_2(), [1]), (su_n(3), [1, 1]),
                      (b_n_simply_connected(2), [1, 2]), (sp_n(2), [2, 1]),
                      (g_2(), [3, 1])]:
        check(f"{dat.name} length multipliers == {mult}",
              SB.length_multipliers(dat) == mult,
              f"got {SB.length_multipliers(dat)}")

    for dat in (su_2(), su_n(3), b_n_simply_connected(2), sp_n(2), g_2()):
        d = dat.dim
        # `constructive_routes=True`: this test is ABOUT the retired zoo — it
        # asserts which constructive route a label takes.  Since 2026-08-25 the
        # production default is the axiom route alone, so the zoo has to be
        # asked for explicitly; the statement being tested is unchanged.
        A = PureGAbeKAlgebra(dat, constructive_routes=True)
        for h in cone_generators(dat)[:2]:
            x0 = A.chart(A.fold(h, (0,) * d))
            mbar = SB.cochar_to_weight(dat, h)
            for k in (1, 2, -1):
                t = SB.theta_twist(x0, k)
                want = (tuple(h), tuple(k * c for c in mbar))
                star, _ = SB.criterion(t)
                check(f"{dat.name} T^{k} L_{(tuple(h), (0,) * d)} == L_{want} "
                      f"(W1 + (★) + well_formed)",
                      t.well_formed() == want and t.well_formed_w1() and star,
                      f"well_formed={t.well_formed()} star={star}")
            # the route actually takes it, and the result certifies (W1+W2)
            lab = A.fold(h, mbar)
            check(f"{dat.name} L_{lab} builds by the twist route, no solve",
                  A.route(lab).split("[")[0] == "twist", f"route={A.route(lab)}")
            check(f"{dat.name} L_{lab} certify_canonical == label",
                  A.certify_canonical(lab) == lab)

    # (ii) the one-step law: the measured holds ...
    holds = [(su_2(), (1,), [(0,)]),
             (su_n(3), (1, 2), [(0, 1), (1, 1)]),
             (b_n_simply_connected(2), (2, 1), [(1, 0), (1, 1)]),
             (sp_n(2), (1, 1), [(0, 1), (1, 0)])]
    for dat, m, cells in holds:
        A = PureGAbeKAlgebra(dat)
        f = A.chart(A.fold(m, (0,) * dat.dim)).residuals()
        cand = SB.one_step_cells(dat, m)
        for p in cells:
            a, n = cand[p]
            check(f"{dat.name} L_{m},0 f_{p} == one_step_residual (α={a}, n={n})",
                  (SB.one_step_residual(dat, m, a) - f[p]).simplify().is_zero())

    # ... and the measured REFUTATIONS, asserted so the scope cannot drift.
    A = PureGAbeKAlgebra(g_2())
    fg = A.chart(A.fold((2, 3), (0, 0))).residuals()
    cg = SB.one_step_cells(g_2(), (2, 3))
    for p in [(1, 2), (1, 1)]:
        a, _n = cg[p]
        check(f"G2 one-step law DOES NOT hold at {p} (the open discriminator)",
              not (SB.one_step_residual(g_2(), (2, 3), a) - fg[p]
                   ).simplify().is_zero())
    A2 = PureGAbeKAlgebra(sp_n(2))
    f2 = A2.chart(A2.fold((1, 1), (0, 0))).residuals()
    a2, _ = SB.one_step_cells(sp_n(2), (1, 1))[(0, 0)]
    check("Sp(2) one-step law DOES NOT hold at the deepest cell (0,0)",
          not (SB.one_step_residual(sp_n(2), (1, 1), a2) - f2[(0, 0)]
               ).simplify().is_zero())

    # n = 1 is not a bubbling cell at all — honest-fail, not a wrong answer.
    try:
        SB.one_step_residual(su_n(3), (1, 1), (2, -1))     # ⟨α,m⟩ = 1
        check("one_step_residual honest-fails at ⟨α,m⟩ = 1", False)
    except NotImplementedError:
        check("one_step_residual honest-fails at ⟨α,m⟩ = 1 (p is in the top "
              "Weyl orbit, f = 1)", True)


def test_spectral_and_monoid_laws():
    """The spectral law of the undressed monopoles, and the exact monoid law it
    forces (the design notes §7e).

    Four measured facts, each asserted here:

    1. the `L_{m,0}` **commute** with each other — and do NOT commute with a
       Wilson line, which is the control showing the test has teeth;
    2. on `R(G)` they are triangular with the explicit diagonal
       `(−𝖖)^E·𝖖^{2⟨μ,m⟩}`, `E = −atom_phase(m)` (`star_bubbling.spectral_diagonal`).
       `E = ⟨ρ,m⟩` at even `⟨Σ⁺,m⟩`, which is every charge of a simply connected
       form and so every charge in this test; the `atom_phase` form is what keeps
       it integral at odd `⟨Σ⁺,m⟩` too, where `⟨ρ,m⟩` would need a fourth root of
       unity and a half power of `𝖖`;
    3. every off-diagonal entry strictly drops `⟨·,m⟩`, so degenerate
       eigenvalues never couple and the family is jointly diagonalizable;
    4. hence — eigenvalues multiply and `⟨ρ,·⟩` is linear, so the
       normalizations cancel — **`L_{m,0}·L_{m',0} = L_{m+m',0}` exactly**.

    (4) is what makes the final form work: every undressed monopole is an exact
    monomial in the cone-generator monopoles, so the licensed (★) solve is
    confined to the finitely many cone generators.  The last check asserts
    exactly that: sweeping the undressed labels, the set that reaches `star` IS
    the set of cone generators."""
    import itertools

    from pure_g_abe_kalgebra import cone_generators

    print("\n[spectral law + the exact monoid law]")
    NEG = W.LaurentPoly({0: -1})

    def same(x, y):
        return (x + W._scale(y, NEG)).is_zero()

    def dom_wts(dat, b):
        return [tuple(e) for e in itertools.product(range(0, b + 1), repeat=dat.dim)
                if dat.is_dominant(e)]

    for dat, wb in [(su_2(), 4), (su_n(3), 2), (b_n_simply_connected(2), 2),
                    (sp_n(2), 2), (g_2(), 1)]:
        d = dat.dim
        z = (0,) * d
        A = PureGAbeKAlgebra(dat, constructive_routes=True)   # about the zoo
        gens = cone_generators(dat)
        wts = dom_wts(dat, wb)
        charts = {h: A.chart(A.fold(h, z)) for h in gens}

        # (1) commutativity, with the Wilson control
        for a, b in itertools.combinations(gens, 2):
            check(f"{dat.name} [L_{{{a},0}}, L_{{{b},0}}] == 0",
                  same(charts[a] * charts[b], charts[b] * charts[a]))
        h0 = gens[0]
        wl = W.wilson(dat, tuple(1 if i == 0 else 0 for i in range(d)))
        check(f"{dat.name} [L_{{{h0},0}}, Wilson] != 0 (control: the algebra is "
              f"NOT commutative)",
              not same(charts[h0] * wl, wl * charts[h0]))

        # (2)+(3) the diagonal law and strict triangularity
        for h in gens:
            dok = tok = 0
            for mu in wts:
                row = SB.operator_row(charts[h], mu, wts)
                want = SB.spectral_diagonal(dat, h, mu)
                got = row.get(mu, W.LaurentPoly.zero())
                dok += 1 if (got - want).is_zero() else 0
                tok += 1 if all(
                    sum(nu[i] * h[i] for i in range(d))
                    < sum(mu[i] * h[i] for i in range(d))
                    for nu in row if nu != mu) else 0
            check(f"{dat.name} L_{{{h},0}} diagonal == "
                  f"(-𝖖)^(-atom_phase(m))·𝖖^(2⟨μ,m⟩) on all {len(wts)} characters",
                  dok == len(wts), f"{dok}/{len(wts)}")
            check(f"{dat.name} L_{{{h},0}} strictly triangular in ⟨·,m⟩ "
                  f"(so degenerate eigenvalues never couple)",
                  tok == len(wts), f"{tok}/{len(wts)}")

        # (4) the monoid law, exactly
        for a, b in itertools.combinations_with_replacement(gens, 2):
            s = tuple(a[i] + b[i] for i in range(d))
            prod = charts[a] * charts[b]
            check(f"{dat.name} L_{{{a},0}}·L_{{{b},0}} == L_{{{s},0}} EXACTLY",
                  same(prod, A.chart(A.fold(s, z))))
            check(f"{dat.name} the product is canonical at {(s, z)} (W1+W2)",
                  prod.well_formed() == (s, z), f"got {prod.well_formed()}")

        # The consequence: (★) is confined to the cone generators — and, since
        # `closed_form_undressed` landed, to the strict SUBSET of them it does
        # not reach.  Sums of generators must NEVER need a solve (that is the
        # monoid law); and a generator needs one only where the closed form
        # honest-fails.
        stars, sums = set(), []
        for a, b in itertools.combinations_with_replacement(gens, 2):
            sums.append(tuple(a[i] + b[i] for i in range(d)))
        for m in list(gens) + sums:
            if A.route(A.fold(m, z)).split("[")[0] == "star":
                stars.add(tuple(m))
        check(f"{dat.name} no SUM of cone generators ever needs (★) "
              f"(the monoid law)", not (stars & set(sums) - set(gens)),
              f"solves at sums: {sorted(stars & set(sums) - set(gens))}")
        check(f"{dat.name} every (★) solve is at a cone generator",
              stars <= set(gens), f"stray solves {sorted(stars - set(gens))}")
        cf_fails = set()
        for h in gens:
            try:
                SB.closed_form_undressed(dat, h)
            except NotImplementedError:
                cf_fails.add(tuple(h))
        check(f"{dat.name} the (★) solves are EXACTLY the generators the closed "
              f"form does not reach ({sorted(cf_fails)})",
              stars == cf_fails, f"solves {sorted(stars)}")


def test_closed_form_undressed():
    """`L_{m,0}` in CLOSED FORM — no solve, no peel (the design notes §7d).

    Certified by the **guard alone** (W1 + (★) + `well_formed` == the label), so
    no comparison against a solve is needed and rank 3–4 is affordable.  That is
    a real certification, not a weaker one: (★) is the acceptance the licensed
    solve itself must pass.

    Also pinned here:
    * the **adjoint (quasi-minuscule) generator** — the coroot of the highest
      root — is closed-form at EVERY datum including `G₂`, where its whole
      content is the single scalar-Neumann-row cell;
    * `L_{m,0}|N|` is a **scalar** `(−1)^{⟨ρ,m⟩}𝖖^{⟨ρ,m⟩}`, which is what forces
      the origin cell;
    * the exact scope of §7c, via `stabilizer_root_orbit`: orbit size 1 and 2
      (with `γ = (α+β)/2` a root) are covered, and the three open cases
      **honest-fail** rather than returning a wrong element.  `Sp(3)` `(1,1,1)` is
      the load-bearing row: there `γ` is a weight but not a root and the size-2
      formula is measurably WRONG, so the scope check must reject it."""
    from pure_g_abe_kalgebra import cone_generators

    print("\n[closed-form undressed monopoles — no solve]")

    reach = {"SU(2)": 1, "SU(3)": 3, "SU(4)": 2, "Spin(5)": 2, "Spin(7)": 2,
             "Sp(2)": 2, "Sp(3)": 2, "G2": 1}
    for dat in (su_2(), su_n(3), su_n(4), b_n_simply_connected(2),
                b_n_simply_connected(3), sp_n(2), sp_n(3), g_2()):
        d = dat.dim
        z = (0,) * d
        got = 0
        for m in cone_generators(dat):
            try:
                X = SB.closed_form_undressed(dat, m)
            except NotImplementedError:
                continue                    # a named open case — never wrong
            ok, _ = SB.criterion(X)
            good = (X.well_formed_w1() is True and ok is True
                    and X.well_formed() == (tuple(m), z))
            check(f"{dat.name} closed-form L_{{{m},0}} guard-certified "
                  f"(W1 + (★) + well_formed)", good,
                  f"W1={X.well_formed_w1()} star={ok} wf={X.well_formed()}")
            got += 1 if good else 0
        check(f"{dat.name} closed form reaches {reach[dat.name]} cone generators",
              got == reach[dat.name], f"reached {got}")

    # the adjoint (quasi-minuscule) generator: support = W·h ∪ {0}
    for dat in (su_2(), su_n(3), b_n_simply_connected(2), sp_n(2), g_2()):
        d = dat.dim
        z = (0,) * d
        adj = min((tuple(int(x) for x in SB.coroot_of(dat, a))
                   for a in dat.positive_roots()
                   if dat.is_dominant_cochar(
                       tuple(int(x) for x in SB.coroot_of(dat, a)))),
                  key=lambda c: W._cochar_height(dat, c))
        X = SB.closed_form_undressed(dat, adj)
        orbit = {tuple(dat.act_cochar(w, adj)) for w in range(len(dat.weyl))}
        check(f"{dat.name} adjoint monopole L_{{{adj},0}}: support == W·h ∪ {{0}}",
              set(X.residuals()) == orbit | {z})
        ok, _ = SB.criterion(X)
        check(f"{dat.name} adjoint monopole closed form guard-certified",
              X.well_formed_w1() and ok and X.well_formed() == (adj, z))

    # Sp(3) (1,1,1): γ is a weight but NOT a root -> must honest-fail
    try:
        SB.closed_form_undressed(sp_n(3), (1, 1, 1))
        check("Sp(3) (1,1,1) honest-fails (γ = (α+β)/2 a weight but not a root, "
              "where the size-2 formula is measurably wrong)", False,
              "returned an element instead of failing")
    except NotImplementedError:
        check("Sp(3) (1,1,1) honest-fails (γ = (α+β)/2 a weight but not a root, "
              "where the size-2 formula is measurably wrong)", True)

    # the two-root formula, exactly, where it is verified
    A = PureGAbeKAlgebra(sp_n(3))
    f = A.chart(A.fold((1, 1, 0), (0, 0, 0))).residuals()
    for p, (alpha, n) in SB.one_step_cells(sp_n(3), (1, 1, 0)).items():
        if p == (0, 0, 0) or p not in f:
            continue
        orb = SB.stabilizer_root_orbit(sp_n(3), p, alpha)
        check(f"Sp(3) L_(1,1,0) cell {p} has stabilizer-root-orbit size 2",
              len(orb) == 2, f"got {orb}")
        check(f"Sp(3) L_(1,1,0) cell {p} == two_root_residual{tuple(orb)} exactly",
              (SB.two_root_residual(sp_n(3), orb[0], orb[1], n)
               - f[p]).simplify().is_zero())


def test_spectral_construction():
    """The INTRINSIC spectral construction (the design notes §7i).

    `{P̄_μ}` is built by Gram–Schmidt of the characters against
    `Δ = ∏_{α∈Φ}(v^α;𝖖²)_∞` — entirely inside `R(G)`, with **no (★)-solved chart
    anywhere**.  From it, `spectral_operator_matrix` produces the matrix of
    `L_{m,0}` on `R(G)` for ANY dominant `m`, including labels where
    `closed_form_undressed` honest-fails.

    Two things are asserted:

    * the construction's output matches the `(★)`-solved operator **exactly** —
      two wholly independent routes, so this is a real cross-certification of the
      solver, not a self-consistency check;
    * `P̄_μ = χ_μ + O(𝖖²)` with coefficients in `Z[𝖖²]` — the repo-natural
      normalisation.

    The **truncation certificate** is asserted too, in both directions, because it
    is load-bearing rather than decorative: at `SU(2)` with a height-10 window,
    order 14 gives 74/81 agreement and the certificate fails; order 30 gives
    80/81 and still fails; order 40 passes and gives 81/81.  A test that only
    checked the passing case would not notice if the guard were removed."""
    print("\n[spectral construction — {P̄_μ} intrinsically, no solved chart]")

    for dat, H, N in [(su_n(3), 8, 12), (b_n_simply_connected(2), 8, 12),
                      (sp_n(2), 8, 12)]:
        d = dat.dim
        A = PureGAbeKAlgebra(dat)
        Pbar, wts = SB.spectral_eigenbasis(dat, H, N)

        # P̄_μ = χ_μ + O(𝖖²), coefficients in Z[𝖖²]
        good = all(
            (nu == mu and (lp - W.LaurentPoly({0: 1})).is_zero())
            or (nu != mu and all(k >= 2 and k % 2 == 0 for k in lp._coeffs))
            for mu, row in Pbar.items() for nu, lp in row.items())
        check(f"{dat.name} P̄_μ = χ_μ + O(𝖖²) with coefficients in Z[𝖖²]", good)

        for m in [tuple(g) for g in __import__(
                "pure_g_abe_kalgebra").cone_generators(dat)]:
            C, _ = SB.spectral_operator_matrix(dat, m, H, N)
            L = A.chart(A.fold(m, (0,) * d))
            ok = tot = 0
            for mu in wts:
                row = SB.operator_row(L, mu, wts)
                for sig in wts:
                    a = C[mu].get(sig, W.LaurentPoly.zero())
                    b = row.get(sig, W.LaurentPoly.zero())
                    tot += 1
                    ok += 1 if (a - b).is_zero() else 0
            check(f"{dat.name} spectral matrix of L_{{{m},0}} == the (★)-solved "
                  f"operator on all {tot} entries", ok == tot, f"{ok}/{tot}")

    # the truncation certificate must FIRE when the order is too small ...
    try:
        SB.spectral_eigenbasis(su_2(), 10, 14)
        check("truncation certificate fires at too-small order (SU(2), h=10, "
              "order=14 — where agreement is only 74/81)", False, "did not fire")
    except SB.StarGuardFailure:
        check("truncation certificate fires at too-small order (SU(2), h=10, "
              "order=14 — where agreement is only 74/81)", True)
    # ... and must PASS at an order that is big enough, where agreement is exact
    C, wts = SB.spectral_operator_matrix(su_2(), (1,), 10, 40)
    N2 = PureSU2KAlgebra()
    A2 = PureGAbeKAlgebra(su_2())
    L2 = A2.chart(A2.fold((1,), (0,)))
    ok = tot = 0
    for mu in wts:
        row = SB.operator_row(L2, mu, wts)
        for sig in wts:
            a = C[mu].get(sig, W.LaurentPoly.zero())
            b = row.get(sig, W.LaurentPoly.zero())
            tot += 1
            ok += 1 if (a - b).is_zero() else 0
    check(f"SU(2) at order 40 (certificate passes): spectral == solved on all "
          f"{tot} entries", ok == tot, f"{ok}/{tot}")


def test_honest_failures():
    print("\n[honest failures — the documented scope limits]")
    # (a) odd <2rho, m> on the SO form is NO LONGER a scope limit (2026-07-29).  `atom_phase` is total and integer-valued, and the honest
    #     half-integral phase is restored at the COCYCLE, through the integral
    #     coboundary factor (−𝖖)^{π(a)π(b)} in `wrq_torus.cocycle_R`.  So the row
    #     now asserts the capability: SO(5) — non-simply-laced — builds its odd
    #     charges, bar-invariantly and orthonormally.
    d5 = so_n(5)
    ph = d5.atom_phase((1, 0))
    check("so_n(5) m=(1,0): atom_phase is an exact int at odd <2rho,m> "
          "(no 𝖖^{1/2})",
          isinstance(ph, int) or Fraction(ph).denominator == 1, f"{ph}")
    check("so_n(5) m=(1,0) odd <2rho,m> is now CANONICAL",
          d5.atom_phase_is_canonical((1, 0)))
    A5 = PureGAbeKAlgebra(d5)
    for m, hgt in [((1, 0), 3), ((2, 1), 7)]:
        lab = (m, (0, 0))
        try:
            ch = A5.chart(lab)
            ip = A5.inner_product(lab, lab, 6)
            check(f"so_n(5) odd m={m} (<Σ⁺,m>={hgt}) charts and is bar-invariant",
                  ch.bar() == ch)
            check(f"so_n(5) odd m={m}: I = 1 + O(𝖖) (orthonormal)",
                  str(ip).startswith("1 "), f"{ip}")
        except Exception as ex:
            check(f"so_n(5) odd m={m} builds", False, f"{type(ex).__name__}: {ex}")
    # the even sector of the same datum is unchanged
    try:
        d5.atom_phase((1, 1))
        check("so_n(5) m=(1,1) even sector works", True)
    except Exception as ex:
        check("so_n(5) m=(1,1) even sector works", False, str(ex)[:80])

    # (b) rank >= 4 USED to honest-fail (no dimension-4 facet normal).  The
    # dominance-order support removed that ceiling, so the row now asserts the
    # capability instead of the limitation.
    for dat, m, size in [(su_n(5), (-1, -1, -1, -1), 21),
                         (sp_n(4), (1, 1, 1, 1), 81),
                         (so_n(9), (1, 1, 0, 0), 25)]:
        try:
            got = SB.tropical_support(dat, m)
            check(f"rank 4 support at {dat.name} m={m} == {size}",
                  len(got) == size, f"got {len(got)}")
        except Exception as ex:
            check(f"rank 4 support at {dat.name}", False,
                  f"{type(ex).__name__}: {str(ex)[:80]}")

    # (c) The G₂ `e = 0` regression is FIXED — it was the ansatz box confining
    # the numerator to the roots of that cell's own denominator `K_f(n)`, a
    # residue of the properness intuition.  Stepping along ALL primitive
    # positive-root directions recovers it, and the answer agrees with the
    # superseded properness device on every cell (two independent devices).
    A = PureGAbeKAlgebra(g_2())
    for m in [(1, 0), (0, 1)]:
        lab = A.fold(m, (0, 0))
        try:
            x = A.chart(lab)
        except Exception as ex:
            check(f"G2 e=0 at m={m} now builds", False,
                  f"{type(ex).__name__}: {str(ex)[:110]}")
            continue
        ok, _rep = SB.criterion(x)
        check(f"G2 e=0 at m={m} builds, W1+(★)+W2",
              bool(x.well_formed_w1()) and ok
              and A.certify_canonical(lab) == lab)

    # (d) `allow_solve=False` must honest-fail where ONLY the solve reaches.
    #
    # Note what changed: `Spin(5)` `m=(1,1)` used to be such a label, and is not
    # any more — `closed_form_undressed` builds it with no solve at all, so
    # `allow_solve=False` now succeeds there.  The honest-fail therefore has to be
    # exercised at a label that is still genuinely open, i.e. one of the three
    # named cases of the design notes §7f.  `Sp(3)` `m=(1,1,1)` is the
    # cheapest: `γ = (α+β)/2` is a weight but not a root, so the size-2 formula
    # does not apply.
    Ac = PureGAbeKAlgebra(b_n_simply_connected(2), allow_solve=False,
                          constructive_routes=True)
    lab = Ac.fold((1, 1), (0, 0))
    x = Ac.chart(lab)                       # closed form, no solve
    ok, _ = SB.criterion(x)
    check("allow_solve=False now BUILDS Spin(5) L_((1,1),0) — the closed form "
          "removed the need for a solve",
          bool(x.well_formed_w1()) and ok and Ac.certify_canonical(lab) == lab)
    check("Spin(5) L_((1,1),0) took the closed_form route (no solve)",
          Ac.route(lab) == "closed_form", f"route {Ac.route(lab)}")

    Ao = PureGAbeKAlgebra(sp_n(3), allow_solve=False,
                          constructive_routes=True)
    labo = Ao.fold((1, 1, 1), (0, 0, 0))
    try:
        Ao.chart(labo)
        check("allow_solve=False honest-fails at a still-open label "
              "(Sp(3) (1,1,1))", False, "built")
    except NotImplementedError:
        check("allow_solve=False honest-fails at a still-open label "
              "(Sp(3) (1,1,1))", True)

    # (e) the guard is real: a bubbling-stripped seed must be REJECTED by (★).
    b2 = b_n_simply_connected(2)
    seed = W.leading_orbit(b2, (1, 1), (0, 0))       # extremal orbit only
    ok, _rep = SB.criterion(seed)
    check("(★) REJECTS the bubbling-stripped Spin(5) seed", not ok)
    full = PureGAbeKAlgebra(b2).chart(((1, 1), (0, 0)))
    ok2, _r2 = SB.criterion(full)
    check("(★) ACCEPTS the completed Spin(5) canonical", ok2)


def test_axiom_route_is_the_whole_surface():
    """Stage 1 (2026-08-25): ONE route — the axioms — builds every label.

    ⚠ The instances here are built with `optimizations=()`, and that is the
    POINT rather than a workaround.  Stage 2 added declared optimizations
    (`theta_twist`, `monoid`) which are ON by default, so `route` at a dressed or
    splittable label now names one of them; `optimizations=()` is exactly stage
    1's surface, kept reachable so "the axioms alone build every label" stays a
    RUNNABLE claim rather than a historical one.  That the optimizations agree
    with this surface is pinned separately, in
    the suite in the source repository.

    Three claims, and each is checked against something independent:

    1. every label takes `star` with the optimizations off, `m = 0` included;
    2. the axiom answer EQUALS the retired constructive zoo's answer — the zoo
       is the only independent construction of these elements, which is why
       agreement is evidence rather than tautology;
    3. at `m = 0` the system is EMPTY (single support cell, no unknowns), so the
       Wilson lines are the degenerate case of the same construction rather
       than a special case — with the honest caveat that `χ_e` enters as the
       W2 seed, not as something solved for.
    """
    print("the axiom route is the whole production surface (stage 1)")

    def same(x, y):
        rx = {tuple(k): v for k, v in x.residuals().items()}
        ry = {tuple(k): v for k, v in y.residuals().items()}
        for k in set(rx) | set(ry):
            a, b = rx.get(k), ry.get(k)
            if a is None:
                if not b.simplify().is_zero():
                    return False
            elif b is None:
                if not a.simplify().is_zero():
                    return False
            elif not (a - b).simplify().is_zero():
                return False
        return True

    zoo = [(su_2(), [((0,), (0,)), ((0,), (1,)), ((0,), (2,)),
                     ((1,), (0,)), ((1,), (1,)), ((2,), (0,))]),
           (su_n(3), [((0, 0), (0, 0)), ((0, 0), (1, 0)), ((0, 0), (1, 1)),
                      ((1, 0), (0, 0)), ((1, 1), (0, 0)), ((1, 0), (1, 0))]),
           (u_n(2), [((0, 0), (1, 0)), ((1, 0), (0, 0)), ((1, -1), (0, 0))]),
           (b_n_simply_connected(2), [((0, 0), (1, 0)), ((1, 0), (0, 0)),
                                      ((0, 1), (0, 0))])]
    for dat, labels in zoo:
        A = PureGAbeKAlgebra(dat, optimizations=())            # axioms only
        Z = PureGAbeKAlgebra(dat, constructive_routes=True)     # the retired zoo
        for lab in labels:
            x = A.chart(A.fold(*lab))
            check(f"{dat.name} L_{lab} takes the axiom route",
                  A.route(lab) == "star", f"route={A.route(lab)}")
            check(f"{dat.name} L_{lab} axioms == constructive zoo "
                  f"(zoo route {Z.route(lab)})", same(x, Z.chart(Z.fold(*lab))))

    # --- m = 0: the system is EMPTY, and the seed is the answer -------------
    for dat, e in [(su_2(), (1,)), (su_n(3), (1, 0)), (so_n(5), (1, 0)),
                   (sp_n(2), (1, 0)), (g_2(), (1, 0))]:
        z = (0,) * dat.dim
        sup = SB.tropical_support(dat, z)
        st, Ff, _s, interior, _k = SB.joint_solve(dat, z, e)
        check(f"{dat.name} m=0: tropical support is the single cell",
              [tuple(p) for p in sup] == [z], f"support={sup}")
        check(f"{dat.name} m=0 e={e}: the system has NO unknowns",
              st == "unique" and interior == [] and not Ff,
              f"status={st} interior={interior} Ff={Ff}")
        check(f"{dat.name} m=0 e={e}: the solve returns the Wilson character",
              same(SB.solve_canonical(dat, z, e), W.wilson(dat, e)))
        # (★) and W1 are ACCEPTANCE here, not constraints: both hold for any χ_e
        ok, _rep = SB.criterion(W.wilson(dat, e))
        check(f"{dat.name} m=0 e={e}: W1 and (★) hold automatically",
              ok and W.wilson(dat, e).well_formed_w1())

    # --- the axioms, run as a CHECKLIST on finished elements ---------------
    # Positive controls first, then negatives — a checklist that accepts
    # everything certifies nothing, and each condition must be shown to catch
    # something the others do not.
    for dat, lab in [(su_2(), ((1,), (0,))), (su_2(), ((2,), (1,))),
                     (su_n(3), ((1, 0), (1, 0))), (su_n(3), ((0, 0), (1, 1))),
                     (b_n_simply_connected(2), ((1, 1), (0, 0)))]:
        A = PureGAbeKAlgebra(dat)
        lb = A.fold(*lab)
        rep = SB.verify_axioms(dat, lb[0], lb[1], A.chart(lb))
        check(f"{dat.name} L_{lab} passes all five axioms",
              all(ok for ok, _ in rep.values()),
              str({k: v for k, v in rep.items() if not v[0]}))

    dat = su_2()
    A = PureGAbeKAlgebra(dat)
    good = A.chart(((2,), (0,)))
    orb = {tuple(dat.act_cochar(w, (2,))) for w in dat.weyl_elements()}
    stripped = W.WRQTorus(dat, {n: f for n, f in good.residuals().items()
                                if tuple(n) in orb})
    rep = SB.verify_axioms(dat, (2,), (0,), stripped)
    check("bubbling-stripped SU(2) L_((2,),0) is rejected by (★) ALONE — the "
          "documented discrimination, and the reason (★) is the guard",
          not rep["(★)"][0] and all(ok for k, (ok, _) in rep.items()
                                    if k != "(★)"),
          str({k: v[0] for k, v in rep.items()}))
    rep = SB.verify_axioms(dat, (2,), (1,), good)
    check("a good element under the WRONG label is rejected by W2",
          not rep["seed (W2)"][0], str({k: v[0] for k, v in rep.items()}))
    one = A.chart(((1,), (0,)))
    rep = SB.verify_axioms(dat, (1,), (0,), one * one)
    check("L_1·L_1 asked as L_1 is rejected by support + W2 + O(𝖖)",
          not rep["support"][0] and not rep["seed (W2)"][0]
          and not rep["O(𝖖)"][0], str({k: v[0] for k, v in rep.items()}))

    # --- a non-dominant representative is a LABEL, not a failure ------------
    A = PureGAbeKAlgebra(su_n(3))
    try:
        SB.solve_canonical(su_n(3), (0, 0), (2, -1))
        check("raw solver refuses an un-normalized representative", False,
              "it built one")
    except ValueError as ex:
        check("raw solver refuses an un-normalized representative, and says "
              "so", "not the normalized representative" in str(ex), str(ex)[:80])
    check("chart folds it: SU(3) (0, α₁) is the adjoint Wilson line",
          A.fold((0, 0), (2, -1)) == ((0, 0), (1, 1))
          and same(A.chart(((0, 0), (2, -1))), W.wilson(su_n(3), (1, 1))))


def test_so3_odd_height_frame():
    """SO(3) = PSU(2) on the `so_n(3)` frame — the odd-height, minuscule case.

    Why this suite needed it: `so_n(3)` as a *datum* appeared in no abelianized
    algebra suite (only in `test_pure_neumann_boundary`), so the tier's
    odd-`⟨Σ⁺,m⟩` path was exercised only through `adjoint_lines(su_2())`.  Those
    are the same lattice, but NOT the same coordinates, and the difference is
    exactly what this case is for:

      * `so_n(3)`: `⟨Σ⁺,(1,)⟩ = 1`, so integer `m` covers the whole lattice and
        `m = 1` IS the minuscule spinorial monopole;
      * `adjoint_lines(su_2())`: `⟨Σ⁺,(1,)⟩ = 2`, so integer `m = 1` is the
        even-height SU(2) adjoint monopole and the spinorial line is the
        FRACTIONAL `m = 1/2` — an integer-only sweep there never reaches it.

    SO(3) is also the smallest case with a non-trivial minuscule cocharacter at
    all: SU(2) has none.  Depth (`m ≤ 4`, `|e| ≤ 6`, 27058 checks) lives in
    a probe in the source repository; this leg is the fast one.
    """
    from wrq_torus import is_minuscule
    dat = so_n(3)

    check("SO(3) frame has ODD height at m=(1,)", dat._root_height((1,)) == 1,
          f"got {dat._root_height((1,))}")
    check("su_2() frame has EVEN height at m=(1,) (a different coordinate frame)",
          su_2()._root_height((1,)) == 2)
    check("SO(3) m=(1,) IS minuscule (the spinorial coweight)",
          is_minuscule(dat, (1,)))
    check("SU(2) has NO non-trivial minuscule cocharacter",
          not is_minuscule(su_2(), (1,)))

    labels = [((0,), (0,)), ((0,), (1,)), ((0,), (2,)),
              ((1,), (0,)), ((1,), (1,)), ((2,), (0,))]
    _axiom_sweep(dat, labels, "SO(3)=PSU(2) odd-height", do_multiply=True)

    A = PureGAbeKAlgebra(dat)
    H0 = A.fold((1,), (0,))
    check("SO(3) minuscule monopole has NO bubbling (support = bare Weyl orbit)",
          set(A.chart(H0).residuals()) == {(1,), (-1,)},
          f"support {sorted(A.chart(H0).residuals())}")
    sq = A.multiply(H0, H0)
    check("H_0^2 == L_{(2,),(0,)} exactly (one term, coefficient 1)",
          (list(sq.terms) == [A.fold((2,), (0,))]
           and list(sq.terms.values())[0] == W.LaurentPoly({0: 1})),
          f"got {sq}")

    # the independent BPS presentation, on the sector where labels correspond.
    # `pure_so3.py` records that the two differ by BPS-canonical vs
    # Wilson-character basis off the pure-magnetic sector, so only that sector is
    # compared label-wise here.
    try:
        from pure_so3 import PureSO3KAlgebra
    except Exception as ex:                                  # pragma: no cover
        check("BPS oracle importable", False, f"{type(ex).__name__}")
        return
    O = PureSO3KAlgebra()
    for m in (1, 2, 3):
        lab = A.fold((m,), (0,))
        ok = all(str(A.inner_product(lab, lab, K=k))
                 == str(O.inner_product((m, 0), (m, 0), k)) for k in (6, 8))
        check(f"SO(3) I(L_{{{m},0}}) == the BPS oracle, K-stably", ok)


def test_su2_correlated_form():
    """The THIRD su(2) global form — `(m, e)` centre classes correlated.

    Beside the simply connected form (`m ∈ Q^∨`, `e ∈ P`) and the adjoint one
    (`m ∈ P^∨`, `e ∈ Q`) there is a correlated lattice, admitting a label only
    when the two centre classes AGREE: both even, or both odd.  Its
    distinguishing line is the dyon at `(ω^∨, ω)` — half-integral `m` AND odd
    `e` — which neither product form admits.

    `test_line_lattice_torus` covers this lattice at the TORUS level (residual
    admissibility, the dyon's half-integer shift, its square being integral).
    This leg is the ALGEBRA level: that the tier builds the dyon and that it
    satisfies the axioms.

    Note the frame point again: the dyon is at FRACTIONAL `m` in su(2)
    coordinates, so an integer-only sweep never reaches the both-odd sector —
    the same trap `global_form.adjoint_lines` warns about.
    """
    from fractions import Fraction
    from global_form import (LineLattice, simply_connected_lines, adjoint_lines,
                             fundamental_coweights, fundamental_weights)
    D = su_n(2)
    cor = LineLattice(D, classes=[((1,), (1,))], name="SO(3) correlated")
    hm = tuple(fundamental_coweights(D)[0])       # ω^∨, class 1
    w = tuple(fundamental_weights(D)[0])          # ω,   class 1

    check("correlated form admits BOTH-ODD (ω^∨, ω)", cor.admits(hm, w))
    check("correlated form admits BOTH-EVEN ((1,), (0,))", cor.admits((1,), (0,)))
    check("correlated form REFUSES odd/even (ω^∨, 0)", not cor.admits(hm, (0,)))
    check("correlated form REFUSES even/odd ((0,), (1,))",
          not cor.admits((0,), (1,)))

    # the dyon is what makes this form a third one at all
    check("the dyon is refused by the simply connected form",
          not simply_connected_lines(D).admits(hm, w))
    check("the dyon is refused by the adjoint form",
          not adjoint_lines(D).admits(hm, w))

    labels = [((0,), (0,)), ((0,), (2,)), ((1,), (0,)),
              (hm, w), ((Fraction(3, 2),), (Fraction(1),))]
    A = PureGAbeKAlgebra(D, lines=cor)
    for m, e in labels:
        lab = A.fold(m, e)
        try:
            x = A.chart(lab)
        except Exception as ex:
            check(f"correlated L_{lab} builds", False,
                  f"{type(ex).__name__}: {str(ex)[:100]}")
            continue
        check(f"correlated L_{lab} W1 (bar-invariant)", bool(x.well_formed_w1()))
        check(f"correlated L_{lab} certify_canonical == label",
              A.certify_canonical(lab) == lab, f"got {A.certify_canonical(lab)}")
        ok, rep = SB.criterion(x)
        check(f"correlated L_{lab} satisfies (★)", ok, str(rep[:2]))
        check(f"correlated L_{lab} ρ round-trip", A.verify_rho_inverse(lab))
        check(f"correlated L_{lab} orthonormality [q^0]",
              A.verify_orthonormality(lab, lab, K=4))

    dy = A.fold(hm, w)
    check("the dyon has NO bubbling (support = the bare Weyl orbit)",
          set(A.chart(dy).residuals())
          == {(Fraction(1, 2),), (Fraction(-1, 2),)},
          f"support {sorted(map(str, A.chart(dy).residuals()))}")
    sq = A.multiply(dy, dy)
    check("dyon^2 == L_{(1,),(2,)} exactly (one term, coefficient 1)",
          (list(sq.terms) == [A.fold((Fraction(1),), (2,))]
           and list(sq.terms.values())[0] == W.LaurentPoly({0: 1})), f"got {sq}")

    # EXPLAINED: the two forms are isomorphic by the
    # θ-twist, and the dyon is the image of the minuscule monopole under it —
    # so this equality is a consequence, not a coincidence.  The isomorphism
    # itself is asserted in `test_theta_twist_relates_the_two_so3_forms`.
    B = PureGAbeKAlgebra(so_n(3))
    check("I(dyon,dyon) == I(H_0,H_0) of the adjoint form's minuscule monopole",
          str(A.inner_product(dy, dy, K=8))
          == str(B.inner_product(B.fold((1,), (0,)), B.fold((1,), (0,)), K=8)),
          f"{A.inner_product(dy, dy, K=8)} vs "
          f"{B.inner_product(B.fold((1,), (0,)), B.fold((1,), (0,)), K=8)}")

    # three forms, three different sets of lines
    sets = set()
    for L in (simply_connected_lines(D), adjoint_lines(D), cor):
        sets.add(frozenset((str(m), str(e))
                           for m in (Fraction(0), Fraction(1, 2), Fraction(1))
                           for e in (Fraction(0), Fraction(1), Fraction(2))
                           if L.admits((m,), (e,))))
    check("the three su(2) forms admit three DIFFERENT sets of lines",
          len(sets) == 3, f"got {len(sets)}")


def test_theta_twist_relates_the_two_so3_forms():
    """The correlated form is isomorphic to the adjoint one by the θ-twist.

    The author, 2026-08-25: *"this global form is isomorphic (not canonically) to the
    conventional one via a fractional θ-twist"*, where a **fractional `k` means
    one that maps a global form to a DIFFERENT global form** —
    not a `k` with a fractional value.

    Measured, that is exactly the parity law: **odd `k` lands in the correlated
    form, even `k` back in the adjoint one**.  So the form-changing twists are
    the odd `k`, and `k = 1` is the smallest.  (For completeness, since the
    literal reading is the tempting one: a *literally* fractional `k = 1/2`
    makes `e` fractional and the correlated form refuses it at every `m ≠ 0`,
    because `m̄ = cochar_to_weight(m) = 2m` already carries the factor 2.  That
    is a fact about this parameter's normalisation, not a reading of the author's
    words.)

    NOT CANONICAL, and demonstrably so: every odd `k` gives a valid isomorphism
    with a DIFFERENT image, so there is no preferred identification.
    """
    from fractions import Fraction
    from global_form import LineLattice, adjoint_lines
    from star_bubbling import theta_twist, cochar_to_weight

    D = su_n(2)
    adj = adjoint_lines(D)
    cor = LineLattice(D, classes=[((1,), (1,))], name="SO(3) correlated")
    A = PureGAbeKAlgebra(D, lines=adj)
    C = PureGAbeKAlgebra(D, lines=cor)

    base = [((Fraction(0),), (Fraction(0),)),
            ((Fraction(1, 2),), (Fraction(0),)),
            ((Fraction(1),), (Fraction(0),)),
            ((Fraction(1, 2),), (Fraction(2),))]

    def mbar(m):
        return cochar_to_weight(D, m)[0]

    for k, want_cor in ((0, False), (1, True), (2, False), (3, True), (-1, True)):
        landed = all(cor.admits(m, (e + k * mbar(m),)) for m, (e,) in base)
        check(f"θ-twist k={k}: image in the correlated form == {want_cor}",
              landed == want_cor)
    # k=1/2 is refused wherever it actually shifts anything; at m=0 it shifts
    # nothing (m̄ = 0), so the honest predicate is "at every m ≠ 0".
    # a LITERALLY fractional k is refused; the form-changing ("fractional" in
    # the author's sense) twists are the odd integer k, checked by the law above.
    check("a literally fractional k=1/2 is REFUSED at every m != 0",
          all(not cor.admits(m, (e + Fraction(1, 2) * mbar(m),))
              for m, (e,) in base if mbar(m) != 0))

    def img(lab):
        (m,), (e,) = lab
        return C.fold((m,), (e + mbar((m,)),))

    labs = [A.fold(m, e) for m, e in base]
    for lab in labs:
        check(f"θ-twist carries the chart at {lab}",
              theta_twist(A.chart(lab), 1) == C.chart(img(lab)))
    check("inner products transport under the θ-twist",
          all(str(A.inner_product(a, b, K=6))
              == str(C.inner_product(img(a), img(b), K=6))
              for a in labs for b in labs))
    check("structure constants transport under the θ-twist",
          all({img(c): v for c, v in A.multiply(a, b).terms.items()}
              == dict(C.multiply(img(a), img(b)).terms)
              for a in labs for b in labs))

    h0 = A.fold((Fraction(1, 2),), (Fraction(0),))
    i_plus = C.fold((Fraction(1, 2),), (Fraction(1),))
    i_minus = C.fold((Fraction(1, 2),), (Fraction(-1),))
    check("k=+1 and k=-1 give DIFFERENT images of the same line",
          i_plus != i_minus)
    check("both images are genuine canonicals of the correlated form",
          C.certify_canonical(i_plus) == i_plus
          and C.certify_canonical(i_minus) == i_minus)
    check("so the identification is NOT canonical (>=2 isomorphisms)",
          theta_twist(A.chart(h0), 1) == C.chart(i_plus)
          and theta_twist(A.chart(h0), -1) == C.chart(i_minus))


def main(argv):
    full = "--full" in argv
    test_root_data()
    test_label_frame()
    test_su2_against_native()
    test_su3_against_oracle()
    test_un_against_keystone()
    _axiom_sweep(b_n_simply_connected(2),
                 [((1, 1), (0, 0)), ((1, 1), (1, 0)), ((2, 1), (0, 0))],
                 "Spin(5)=B2", do_multiply=True)
    _axiom_sweep(g_2(), [((1, 0), (1, 0)), ((1, 0), (0, 1))], "G2")
    _axiom_sweep(sp_n(2), [((1, 0), (0, 0)), ((1, 1), (0, 0))], "Sp(2)=USp(4)")
    test_axiom_route_is_the_whole_surface()
    test_global_form()
    test_cone_generators()
    test_analytic_laws()
    test_spectral_and_monoid_laws()
    test_closed_form_undressed()
    test_spectral_construction()
    test_two_presentations()
    test_honest_failures()
    test_so3_odd_height_frame()
    test_su2_correlated_form()
    test_theta_twist_relates_the_two_so3_forms()
    if full:
        _axiom_sweep(sp_n(3), [((1, 1, 1), (0, 0, 0))], "Sp(3)=USp(6)=C3")
        _axiom_sweep(so_n(7), [((1, 1, 0), (0, 0, 0))], "SO(7)=B3")
        test_rank4_solve()

    print(f"\n{'=' * 66}")
    if _FAIL:
        print(f"{_PASS} passed, {len(_FAIL)} FAILED: {_FAIL}")
        return 1
    print(f"All {_PASS} PureGAbeKAlgebra tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
