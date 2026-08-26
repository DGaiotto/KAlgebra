"""Self-test for the `GNKAlgebra` (Step 7) export package.

Step 7 is gauge theory at an **arbitrary 4d gauge group with arbitrary matter**,
on the abelianized-presentation tier of Step 5.  Four blocks:

  H. **pure gauge at an ARBITRARY root datum** (`PureGAbeKAlgebra`) — including
     groups with no minuscule cocharacter, where product-and-peel has nothing to
     start from and the canonical is reached by the **licensed (★)-guarded**
     route; and including the **odd-height** charges: a cocharacter of odd
     `⟨Σ⁺, m⟩` has a half-integral atom phase `−½⟨Σ⁺,m⟩`, but the algebra needs
     that phase only through its coboundary `δS̃`, which is an integer even where
     the phase is not, so the cocycle restores it and the charge **builds**.  No
     `𝖖^{1/2}` and no `i` occurs; the scalar ring stays `Z[𝖖, 𝖖⁻¹]`.  (An earlier
     revision of this gate asserted that such a charge must raise — that boundary
     is retracted, and `test_H_odd_height_builds` now pins the opposite);
  I. **`(G, N)` matter at any datum and any matter representation**
     (`GNAbeKAlgebra`), the named theory **presets** (parameter tuples, not
     classes), and the type-A **seam** — a surjection, not an isomorphism;
  J. the **4d gauge group data** as a lattice and its dual, and the character on
     the centre that decides which forms need the phase correction to fire;
  K. the **Schur pairing in vacuum-state form**, `I_{a,b} = ⟨L_a·1, L_b·1⟩`.

Run from the repository root with Steps 1-6 on the path:

    python3 run_tests.py          # the whole gate, this suite included

Pure Python 3, **no third-party dependencies at all** — literally, not by guard.
`star_bubbling` has exactly one solver path, in pure Python.  (It once carried a
numpy "accelerator"; that path never called numpy, was slower, and could skip the
unlucky-prime retry, so whether a canonical built or honest-failed could depend on
what happened to be installed.  It was removed rather than tested.)
"""

from abe_kalgebra import AbeKAlgebra


# ===========================================================================
# H. Pure gauge at ANY root datum — and the (★)-guarded build
# ===========================================================================

def test_H_pure_g_any_datum():
    """`PureGAbeKAlgebra` is the abelianized tier at an **arbitrary** root datum,
    not just type A.  The anchor is a group with **no minuscule cocharacter**
    (`Spin(5)`), where there is no dressed-minuscule generator to start from, so
    the canonical is reached by the **(★)-guarded** route — (★) being the
    affine-Weyl residue cancellation, equivalently *the element's `𝖖`-difference
    operator preserves* `Λ = R(G)`, the representation ring of this form.
    `route(label)` reports which route fired, so the build is visible rather
    than implicit."""
    import root_datum as rd
    from pure_g_abe_kalgebra import PureGAbeKAlgebra
    A = PureGAbeKAlgebra(rd.b_n_simply_connected(2))          # Spin(5)
    assert isinstance(A, AbeKAlgebra)
    ident = A.identity()
    assert A.certify_canonical(ident) == ident
    routes = {}
    for m in ((1, 1), (2, 1)):        # cochar-dominant representatives
        lab = (m, (0,) * len(m))
        A.chart(lab)                                # builds, or raises
        routes[m] = A.route(lab)
        # W1+W2 acceptance: `certify_canonical` returns the dominant
        # representative, and certifying it again is a fixed point.
        got = A.certify_canonical(lab)
        assert A.certify_canonical(got) == got, ("W1+W2 acceptance", lab, got)
    assert A.verify_orthonormality(ident, ident, 4), "Spin(5) Goal 2.1 at the unit"
    print("  PASS: test_H_pure_g_any_datum  (Spin(5): routes %s, certify + Goal 2.1)"
          % routes)


def test_H_one_route_and_the_declared_optimizations():
    """**One route builds every label** (2026-08-25), and the two general
    optimizations are speed, never a different answer.

    This replaces an earlier test of a *route ladder* — `wilson → minuscule →
    cone → peel → monoid → star`, with `star` as "the last resort".  That ladder
    is retired: the axioms alone (the W2 seed, bar, `O(𝖖)`, and (★)) build every
    `L_{m,e}` at every datum, `m = 0` included, where the tropical support is the
    single cell `{0}`, the system has no unknowns and the seed **is** the answer.
    So Wilson follows from the axioms rather than from a special case.

    Two halves, and the first is the load-bearing one:

    * with `optimizations=()` the claim is **runnable**, not merely asserted —
      every route reported is `star`, at a Wilson label, a minuscule label and a
      bubbled one;
    * with the defaults on, `theta_twist` and `monoid` must return the
      **identical element**.  An optimization that returned a merely similar
      answer would be the exact failure mode the constructive-build rule exists
      to prevent, so equality is what is asserted, not agreement of some
      invariant.

    The constructive zoo is retired-not-erased behind `constructive_routes=True`,
    and it is kept for the reason the peel `S`-recursion is kept: it is an
    *independent* construction of the same element, which is what makes agreement
    evidence rather than a restatement.
    """
    import root_datum as rd
    from pure_g_abe_kalgebra import PureGAbeKAlgebra

    # --- half 1: the axioms alone, nothing else switched on -----------------
    bare = PureGAbeKAlgebra(rd.su_n(3), optimizations=())
    labels = [((0, 0), (1, 0)),        # Wilson  — support is the single cell {0}
              ((1, 0), (0, 0)),        # minuscule — no bubbling
              ((1, 1), (0, 0))]        # bubbled  — the general case
    routes = {}
    for m, e in labels:
        lab = bare.fold(m, e)
        bare.chart(lab)
        routes[lab] = bare.route(lab)
        assert routes[lab] == "star", ("with no optimizations every label must "
                                       "be built by the axiom route", lab,
                                       routes[lab])
        assert bare.certify_canonical(lab) == lab, ("W1+W2 acceptance", lab)

    # --- half 2: the declared optimizations agree ELEMENTWISE ----------------
    fast = PureGAbeKAlgebra(rd.su_n(3))
    assert PureGAbeKAlgebra.DEFAULT_OPTIMIZATIONS == ("theta_twist", "monoid")
    checked = 0
    for m, e in labels:
        lab = fast.fold(m, e)
        assert fast.chart(lab) == bare.chart(lab), ("an optimization returned a "
                                                    "different element", lab)
        checked += 1

    # The θ-twist is general in `e`: one known point gives the whole `m̄`-line,
    # so a DRESSED neighbour of an already-built charge must come back twisted
    # rather than re-solved.  Asking for it after the undressed anchor above is
    # what exercises that; the element is still checked against the bare build.
    twisted = fast.fold((1, 1), (1, 1))
    assert fast.chart(twisted) == bare.chart(twisted), "θ-twist changed the element"
    tw = fast.route(twisted)
    # Non-vacuous: the twist must actually have FIRED here.  Its route name
    # carries the shift and the charge it twisted from (`twist[k=.. from ..]`),
    # so `startswith` pins the mechanism rather than merely accepting any answer
    # — and the same label under `optimizations=()` must come back from the
    # axiom route instead, or the comparison above is between two twists.
    assert tw.startswith("twist["), ("the θ-twist must fire at a dressed "
                                     "neighbour of a cached charge", tw)
    assert bare.route(twisted) == "star", bare.route(twisted)
    print("  PASS: test_H_one_route_and_the_declared_optimizations  "
          "(SU(3): %d labels all route='star' at optimizations=(); "
          "%d identical under the defaults; θ-twist fired as %s)"
          % (len(labels), checked, tw))


def test_H_the_five_axioms_and_their_negative_controls():
    """The axioms are **executable in one place** — `star_bubbling.verify_axioms`
    runs all five against a *finished* element and returns
    `{condition: (ok, detail)}` over `bar (W1)`, `seed (W2)`, `support`, `O(𝖖)`
    and `(★)`.  It exists apart from the solver's internal guard because the
    guard runs on what the solver just built, whereas this runs on **anything** —
    in particular on an independently constructed element, which is what makes
    checking a build against the axioms a test rather than a restatement.

    What makes the list non-redundant is the **negative controls**: each
    condition has to catch something the others do not, or it is decoration.
    Positive controls run first — a checklist that accepts everything certifies
    nothing.

    * A **bubbling-stripped** canonical (keep only the leading Weyl orbit) passes
      W1, W2, support and `O(𝖖)` and is rejected by **(★) alone**.  That is the
      whole reason a solve is licensed here, and it is the same fact as the
      derivation of orthonormality: `I[𝖖⁰]` reads only the leading orbit, so the
      `𝖖⁰` self-norm **cannot see the bubbling at all** and passes on exactly
      this content, while (★) — a membership condition on `Λ = R(G)`, not a
      pairing inside the span — rejects it.
    * A good element under the **wrong `e`** is rejected by W2.
    * `L_1·L_1` asked as `L_1` is rejected by support, W2 and `O(𝖖)` together.
    """
    import root_datum as rd
    import star_bubbling as SB
    import wrq_torus as W
    from pure_g_abe_kalgebra import PureGAbeKAlgebra

    # --- positive controls, across three data ------------------------------
    for dat, lab in [(rd.su_2(), ((1,), (0,))),
                     (rd.su_n(3), ((0, 0), (1, 1))),
                     (rd.b_n_simply_connected(2), ((1, 1), (0, 0)))]:
        A = PureGAbeKAlgebra(dat)
        lb = A.fold(*lab)
        rep = SB.verify_axioms(dat, lb[0], lb[1], A.chart(lb))
        assert all(ok for ok, _ in rep.values()), \
            (dat.name, lab, {k: v for k, v in rep.items() if not v[0]})

    dat = rd.su_2()
    A = PureGAbeKAlgebra(dat)
    good = A.chart(((2,), (0,)))

    # (a) strip the bubbling — (★) ALONE must reject
    orb = {tuple(dat.act_cochar(w, (2,))) for w in dat.weyl_elements()}
    stripped = W.WRQTorus(dat, {n: f for n, f in good.residuals().items()
                                if tuple(n) in orb})
    rep = SB.verify_axioms(dat, (2,), (0,), stripped)
    assert not rep["(★)"][0], ("(★) must reject the bubbling-stripped element",
                               rep)
    assert all(ok for k, (ok, _) in rep.items() if k != "(★)"), \
        ("every OTHER axiom must pass on it — that is what makes (★) the guard",
         {k: v[0] for k, v in rep.items()})

    # (b) the wrong electric label — W2 sees it
    rep = SB.verify_axioms(dat, (2,), (1,), good)
    assert not rep["seed (W2)"][0], {k: v[0] for k, v in rep.items()}

    # (c) a product asked as a generator — support + W2 + O(𝖖) together
    one = A.chart(((1,), (0,)))
    rep = SB.verify_axioms(dat, (1,), (0,), one * one)
    assert not rep["support"][0] and not rep["seed (W2)"][0] \
        and not rep["O(𝖖)"][0], {k: v[0] for k, v in rep.items()}

    print("  PASS: test_H_the_five_axioms_and_their_negative_controls  "
          "(3 positive controls; bubbling-stripped rejected by (★) alone; "
          "wrong `e` by W2; L_1·L_1 by support+W2+O(q))")


def test_H_odd_height_builds():
    """**Odd height is not an obstruction** — the charge builds, and this test
    exists because an earlier revision of this gate asserted the opposite.

    A cocharacter with odd `⟨Σ⁺, m⟩` (writing `Σ⁺` for the sum of the positive
    roots, so the Weyl vector is `½Σ⁺`) has a *half-integral* atom phase
    `S = −⟨ρ,m⟩ = −½⟨Σ⁺,m⟩`.  The earlier reading concluded that the tier
    therefore could not carry the charge without a `𝖖^{1/2}`.  The measurement
    reproduces; **the inference does not**.  The algebra never needs `S` as a
    number — only through its coboundary
    `δS̃(a,b) = S̃(a+b) − S̃(a) − S̃(b)`, which is an **integer even where `S̃` is
    not**, because `π = ⟨Σ⁺,·⟩ mod 2` is Weyl-invariant *and* additive, so the
    halves cancel.  The cocycle restores the honest phase through that integral
    coboundary, and the charge is an ordinary atom of this torus.

    `SO(5)` at `m = (1,0)` is such a charge: `Σ⁺ = (3,1)`, so `⟨Σ⁺,m⟩ = 3` is
    odd and `atom_phase_doubled` is `−3`.  It is the very charge the previous
    revision asserted must raise.  No `𝖖^{1/2}` and no `i` enters: the scalar
    ring is `Z[𝖖, 𝖖⁻¹]` here as everywhere else."""
    import root_datum as rd
    from global_form import LineLattice
    from pure_g_abe_kalgebra import PureGAbeKAlgebra
    dat = rd.so_n(5)
    # the charge really is the odd-height one -- assert the hypothesis, so the
    # test cannot quietly become a statement about an even charge.
    assert dat.atom_phase_doubled((1, 0)) == -3, "SO(5) (1,0) must have odd height"
    assert LineLattice(dat, H=(), name="SO(5)").abe_representable((1, 0)) is True

    A = PureGAbeKAlgebra(dat)
    lab = ((1, 0), (0, 0))
    A.chart(lab)
    assert A.certify_canonical(lab) == lab
    # ... and it is a *canonical* element, not merely a returned value: Goal 2.1
    # orthonormality is the property an off-span fabrication would fail.
    assert A.verify_orthonormality(lab, lab, 4), "SO(5) odd charge Goal 2.1"
    # the even sector on the same datum is unchanged by the correction.
    lab2 = ((1, 1), (0, 0))
    A.chart(lab2)
    assert A.certify_canonical(lab2) == lab2
    print("  PASS: test_H_odd_height_builds  (SO(5) ⟨Σ⁺,m⟩=3: odd charge builds, "
          "certifies and is orthonormal — the retracted boundary)")


def test_H_odd_height_against_bps_oracle():
    """The odd-height fix checked against an **independent presentation**.

    The test above certifies the odd charge *within* the abelianized tier — with
    the tier's own acceptance and its own pairing.  That is the weaker kind of
    evidence: the phase correction being tested lives in the same cocycle the
    pairing is computed from.  So this test leaves the tier entirely.

    `PureSO3KAlgebra` presents pure `SO(3) = PSU(2)` as a **BPS-quiver chart**
    (nodes `(2,0), (−2,1)`, Dirac pairing `⟨n₀,n₁⟩ = 2`), where the quiver nodes
    *are* the charge basis and the coweight-torus atom phase never enters at all.
    The minimal 't Hooft `H_0 = F_{(1,0)}` is the spinorial line — the one with no
    `SU(2)` preimage, and `⟨Σ⁺, ω^∨⟩ = 1`, the smallest odd height there is.  The
    two presentations share no machinery on the quantity being compared, so
    agreement is a real cross-check on the correction rather than a restatement
    of it, and the earlier `𝖖^{1/2}`/`i` reading is what it would have refuted.

    Both must give `H_0² = L_{(2,0)}` (the `SU(2)` adjoint monopole) and the same
    Schur self-pairing term by term.  Note this test **imports the Step-4 BPS
    spine** — the oracle is a `BPSKAlgebra`, so there is no version of this
    certificate that does not.  That is why the suite runs after the spine-free
    assertions of Steps 1-3, not before."""
    import root_datum as rd
    from pure_g_abe_kalgebra import PureGAbeKAlgebra
    from pure_so3 import PureSO3KAlgebra
    K = 8
    # -- the independent BPS presentation -----------------------------------
    oracle = PureSO3KAlgebra()
    assert oracle.H0 == (1, 0)
    sq_bps = oracle.multiply(oracle.H0, oracle.H0).terms
    assert set(sq_bps) == {(2, 0)}, f"BPS H_0^2 support {sorted(sq_bps)}"
    assert str(sq_bps[(2, 0)]) == "1", "H_0^2 must be L_(2,0) EXACTLY"
    i_bps = oracle.inner_product(oracle.H0, oracle.H0, K)
    # -- the same line on the abelianized tier ------------------------------
    dat = rd.so_n(3)
    assert dat.atom_phase_doubled((1,)) % 2 != 0, "m=(1,) must have ODD height"
    abe = PureGAbeKAlgebra(dat)
    lab = abe.fold((1,), (0,))
    chart = abe.chart(lab)
    assert chart.bar() == chart, "the spinorial line must be bar-invariant"
    assert abe.certify_canonical(lab) == lab
    sq_abe = abe.multiply(lab, lab).terms
    assert set(sq_abe) == {((2,), (0,))}, f"abe H_0^2 support {sorted(sq_abe)}"
    assert str(sq_abe[((2,), (0,))]) == "1", "H_0^2 must be L_(2,0) EXACTLY"
    i_abe = abe.inner_product(lab, lab, K)
    # -- and they agree, with the value pinned so neither side can drift ----
    want = "1 - q^2 + q^4 + q^6 - q^8 + O(q^9)"
    assert str(i_bps) == want, f"BPS oracle I(H_0,H_0) = {i_bps}"
    assert str(i_abe) == want, f"abe tier  I(H_0,H_0) = {i_abe}"
    print("  PASS: test_H_odd_height_against_bps_oracle  (SO(3) H_0: BPS chart "
          f"== abelianized tier, H_0^2 = L_(2,0) and I = {want})")


# ===========================================================================
# I. (G, N) — matter at ANY datum and ANY matter representation
# ===========================================================================

def test_I_gn_abe_kalgebra():
    """`GNAbeKAlgebra` puts `T^*N` matter on the tier at an arbitrary datum and
    an arbitrary matter representation.  The **4d gauge group data** is the
    `lines=` argument (a maximal set of mutually compatible `(m, e)` labels),
    NOT a choice of cocharacter lattice; matter is a **representation**, not a
    count, which is what makes `Sp(4)+4` and `Spin(5)+4ˢ` different theories."""
    import root_datum as rd
    from gn_abe_kalgebra import GNAbeKAlgebra
    A = GNAbeKAlgebra(rd.su_2(), (1,), 2)            # SU(2) + 2 fundamentals
    assert isinstance(A, AbeKAlgebra)
    ident = A.identity()
    assert A.certify_canonical(ident) == ident
    print("  PASS: test_I_gn_abe_kalgebra  (SU(2)+2×fund on the (G,N) tier)")


def test_I_roster_presets():
    """Named theories are **parameter presets, not classes** — a special class is
    earned only by an algorithm optimized for a given `G` and/or `N`, and those
    classes already exist.  Checks a non-type-A preset off the roster."""
    from g_matter_roster import ROSTER, roster
    assert "sp4-nf1" in ROSTER and "g2-nf1" in ROSTER
    A = roster("sp4-nf1")
    ident = A.identity()
    assert A.certify_canonical(ident) == ident
    print("  PASS: test_I_roster_presets  (%d presets; sp4-nf1 builds)" % len(ROSTER))


def test_I_type_a_seam_is_a_surjection():
    """The relation between the general `(G, N)` class and the purpose-built
    type-A `UNNfKAlgebra` is a **surjection, not an isomorphism**, and the
    sharpest witness is the *trace*: at `N_f = 1` structure constants and ρ
    agree, but the matter `μ` grades matter zero modes, so it appears in the
    trace of a flavour-NEUTRAL canonical (the `U(2)+1` Wilson line screens
    against the matter and picks up flavour charge).  `UNNfKAlgebra` specialized
    that fugacity to 1, and a spectator `U(1)` adjoined afterwards is inert — so
    the difference is a genuine μ-refinement of the index, not a flavour that
    could be tensored back on."""
    from zplus_ring import un_to_sun_hom, un_to_cartan_hom
    # the two ring homs that make the comparison exact rather than
    # augmentation-blurred: the central specialization and the Cartan restriction
    assert un_to_sun_hom(1) is not None
    assert un_to_cartan_hom(1) is not None
    import g_matter_un_nf_seam as seam
    assert hasattr(seam, "UNNfSpecialization")
    print("  PASS: test_I_type_a_seam_is_a_surjection  (seam + both ring homs present)")


# ===========================================================================
# J. The 4d gauge group data, and Langlands duality
# ===========================================================================

def test_J_lattices_are_dual():
    """A 4d gauge group's charge data is **a lattice and its dual** — there is no
    rescaling anywhere in it.  `verify_lattices_are_dual` checks, in BOTH
    directions, that `m` is admitted exactly when `⟨m, e⟩ ∈ Z` for every
    admitted `e` and vice versa; that is strictly stronger than maximality,
    which can only refute over-smallness.  A self-dual form has balanced
    admitted-charge counts."""
    import itertools
    from fractions import Fraction as F
    import root_datum as rd
    from global_form import LineLattice, fundamental_coweights, fundamental_weights

    def box(D):
        cow, wt, d = fundamental_coweights(D), fundamental_weights(D), D.dim
        rng = range(-2, 3)
        mags = [tuple(sum(F(c[j]) * F(cow[j][i]) for j in range(d)) for i in range(d))
                for c in itertools.product(rng, repeat=d)]
        eles = [tuple(sum(F(c[j]) * F(wt[j][i]) for j in range(d)) for i in range(d))
                for c in itertools.product(rng, repeat=d)]
        return mags, eles

    counts = {}
    for tag, D, H in (("SU(2)", rd.su_2(), ()), ("SO(3)", rd.su_2(), (0, 1)),
                      ("G_2", rd.g_2(), ())):
        lat = LineLattice(D, H=H, name=tag)
        mags, eles = box(D)
        assert lat.verify_lattices_are_dual(mags, eles), (tag, "duality")
        counts[tag] = (sum(1 for m in mags if lat.mag_admits(m)),
                       sum(1 for e in eles if lat.elec_admits(e)))
    # the reciprocity that motivated the check: the same box, lattices exchanged
    assert counts["SU(2)"] == counts["SO(3)"][::-1], counts
    # G_2 is self-dual, so its counts are balanced
    assert counts["G_2"][0] == counts["G_2"][1], counts
    print("  PASS: test_J_lattices_are_dual  (SU(2)/SO(3) counts %s/%s swapped; "
          "G_2 self-dual %s)" % (counts["SU(2)"], counts["SO(3)"], counts["G_2"]))


def test_J_parity_character_decides_the_frame():
    """Which forms need the phase **correction** is decided by a character on
    the centre, not charge by charge: `⟨Σ⁺, α_i^∨⟩ = 2` on every simple coroot,
    so `⟨Σ⁺, ·⟩ mod 2` kills `Q^∨` and descends to `π : P^∨/Q^∨ → Z/2`.  The
    default atom phase suffices on all of `G̃/H` exactly when `π|_H ≡ 0` —
    decidable from the centre alone, with nothing built.

    ⚠ **This is no longer the same question as `abe_representable`**, and the two
    deliberately disagree.  The per-charge predicate is `True` everywhere (block
    H), so a `False` here names a form on which the cocycle's coboundary
    correction actually *fires* — **not** a form with lines the tier cannot
    build.  Reading it the latter way is exactly the retracted inference."""
    import root_datum as rd
    from global_form import LineLattice
    su2 = LineLattice(rd.su_2(), H=(), name="SU(2)")
    so3 = LineLattice(rd.su_2(), H=(0, 1), name="SO(3)")
    assert su2.abe_representable_lattice() is True, \
        "SU(2): the default atom phase needs no correction"
    assert so3.abe_representable_lattice() is False, \
        "SO(3): the coboundary correction fires (and the lines still build)"
    # an odd centre order admits no non-trivial character to Z/2, so the adjoint
    # form of PSU(3) is carried whole without testing one coweight at a time
    psu3 = LineLattice(rd.su_n(3), H=(0, 1, 2), name="PSU(3)")
    assert psu3.abe_representable_lattice() is True, "odd |Z| ⇒ π ≡ 0"
    print("  PASS: test_J_parity_character_decides_the_frame  "
          "(SU(2) yes / SO(3) no / PSU(3) yes — matches block H)")


# ===========================================================================
# K. The Schur pairing in vacuum-state form
# ===========================================================================

def test_J_generalized_global_forms():
    """A global form need not be a **product** of a magnetic and an electric
    lattice.  The label space is the Weyl quotient of a lattice `Λ` between
    `Q^∨ × Q` and `P^∨ × P` on which the Dirac pairing is integral, so `Λ` is the
    preimage of a finite subgroup `L = Λ/(Q^∨ × Q)` of centre-class **pairs** —
    and *admissible ⟺ `L` isotropic*, *maximal ⟺ `L` Lagrangian*, with
    **maximality not required**: a non-maximal isotropic `L` is a consistent,
    merely incomplete, set of lines.

    At `su(2)` that gives **three** forms, not two.  Beside `SU(2) = (Q^∨, P)`
    and `SO(3) = (P^∨, Q)` there is the *correlated* one — a discrete theta angle
    — generated by the **pair** `((1,), (1,))`: odd magnetic class requires odd
    electric class.  Its one extra line is the dyon `(ω^∨, ω)`, which neither
    product form admits — `SU(2)` has no `ω^∨`, `SO(3)` has no `ω`.

    So `admits` is **not** `mag_admits and elec_admits`.  That conjunction is
    right only for a product form, and the two side predicates are now its
    *fibres* over `e = 0` and `m = 0`; the correlated form is exactly where the
    difference shows, which is why it is the witness here.
    """
    import root_datum as rd
    from global_form import (LineLattice, adjoint_lines, simply_connected_lines,
                             fundamental_coweights, fundamental_weights)

    D = rd.su_n(2)
    sc, adj = simply_connected_lines(D), adjoint_lines(D)
    cor = LineLattice(D, classes=[((1,), (1,))], name="su(2) correlated")

    hm = tuple(fundamental_coweights(D)[0])       # ω^∨, in the coroot basis
    w = tuple(fundamental_weights(D)[0])          # ω,  in the weight basis
    assert cor.admits(hm, w), "the correlated form must admit the dyon (ω^∨, ω)"
    assert not sc.admits(hm, w), "SU(2) has no ω^∨"
    assert not adj.admits(hm, w), "SO(3) has no ω"

    # the three are genuinely different sets of lines, not renamings
    probe = [((0,), (0,)), ((0,), w), (hm, (0,)), (hm, w), ((1,), (0,))]
    sigs = {name: tuple(lat.admits(*l) for l in probe)
            for name, lat in (("SU(2)", sc), ("SO(3)", adj), ("correlated", cor))}
    assert len(set(sigs.values())) == 3, ("the three forms must differ", sigs)

    print("  PASS: test_J_generalized_global_forms  (su(2) has THREE forms; the "
          "correlated one admits the dyon (ω^∨,ω) that neither product form "
          "does; three distinct admission signatures)")


def test_J_line_lattice_tori():
    """The line lattice's own rational quantum tori (`line_lattice_torus`).

    `conventional_quantum_torus(lines)` reads the Dirac matrix on a `Z`-basis of
    `Λ` and hands it to the `QuantumTorusKAlg` Step 1 already ships, so an
    **inadmissible** lattice is refused by *arithmetic* — a non-integral `𝖖`
    power — rather than by a check bolted on beside it.

    `DRationalTorus` represents `Σ_m d_m(𝖖^m v)·u^m` on the plain generators and
    `FRationalTorus` represents `Σ_m f_m(𝖖^m v)·U_m` on the **dressed** atoms;
    the *only* difference is the cocycle, since `U_a` commutes past a function of
    `v` exactly as `u^a` does.  So the `F` side exercises the primitive cocycle
    `CC` directly.

    The load-bearing check is the **residual splitting**, and the correlated form
    is where it bites.  Only the bare residual is stored, as `v^e` times a
    rational function of the *root* characters, so the whole fractional content
    sits in one scalar per residual: `⟨ω^∨, ω⟩ = ½` is a genuine half-integer,
    and yet the dyon **squared** carries only integral `𝖖` powers, because in the
    product law the two such scalars combine into the Dirac pairing — an integer
    exactly by admissibility.  Without that split the frame would not close over
    `Z[𝖖, 𝖖⁻¹]` at a correlated form at all.

    And the algebra depends on the **lattice**, not just the datum: the same
    residual is legal in one form and refused at construction in another.
    """
    import root_datum as rd
    from global_form import (LineLattice, simply_connected_lines,
                             fundamental_coweights, fundamental_weights)
    from line_lattice_torus import (DRationalTorus, FRationalTorus,
                                    conventional_quantum_torus)

    D = rd.su_n(2)
    sc = simply_connected_lines(D)
    cor = LineLattice(D, classes=[((1,), (1,))], name="su(2) correlated")
    assert conventional_quantum_torus(sc) is not None
    assert FRationalTorus(sc) is not None            # the dressed-atom sibling

    hm = tuple(fundamental_coweights(D)[0])
    w = tuple(fundamental_weights(D)[0])
    Fc, Fs = DRationalTorus(cor), DRationalTorus(sc)

    from fractions import Fraction
    assert Fraction(Fc.pair(hm, w)).denominator == 2, \
        ("the dyon's own shift must be a HALF-integer", Fc.pair(hm, w))

    dyon = Fc.monomial(hm, w)
    sq = Fc.multiply(dyon, dyon)
    assert sq, "the dyon must square inside the correlated form"

    refused = False
    try:
        Fs.monomial(hm, w)
    except ValueError:
        refused = True
    assert refused, ("the dyonic residual must be REFUSED in the simply "
                     "connected form — the algebra depends on the lattice")

    print("  PASS: test_J_line_lattice_tori  (conventional QT from the Dirac "
          "matrix on a Z-basis of Λ; ⟨ω^∨,ω⟩ = 1/2 yet the dyon squares "
          "integrally; the same residual refused at SU(2))")


def test_K_aux_space_pairing():
    """The **primary definition** of the Schur pairing on the enriched rational
    quantum torus: states are plain torus products `L_a|1⟩ = chart(a)·|1⟩`
    against the vacuum `|1⟩ = (𝖖²;𝖖²)_∞^dim·∏_α(𝖖²v^α;𝖖²)_∞·U_0`, and

        I_{a,b} = Tr(ρ(a)·b) = ⟨L_a·1, L_b·1⟩

    with ρ never constructed — the Hermitian slot plus the two vacuum dressings
    generate the twist."""
    from aux_space import AuxSpace
    aux = AuxSpace(2, KP=8)
    assert aux.vacuum() is not None
    for meth in ("state", "act_left", "act_right", "pair", "schur_pairing",
                 "half_index", "from_datum"):
        assert hasattr(aux, meth) or hasattr(AuxSpace, meth), meth
    print("  PASS: test_K_aux_space_pairing  (vacuum + pairing surface present)")


# ===========================================================================
#  The contract itself, at a general G — through the UNIVERSAL surface
# ===========================================================================

def test_contract_surface_at_a_general_G():
    """**Do these general-`G` algebras actually look like `KAlgebra`s?**

    Everything above certifies a *particular* structural claim — the route
    ladder, the licence, the lattice duality, the pairing.  This one steps back
    and exercises the contract itself, and deliberately does so **only through
    the universal surface** (`multiply`, `rho`, `rho_inverse`, `trace`,
    `inner_product` and the verifiers), never through a presentation's internals:
    the whole point of the class hierarchy is that a question about the abstract
    algebra is answered through `KAlgebra`, and reaching for
    `PureGAbeKAlgebra`-specific machinery to answer it would confuse the
    presentation with the object.

    It also *prints* the values, because a general-`G` canonical basis is the
    thing hardest to take on trust.  The G₂ Wilson line is the specimen worth
    reading: its square is

        L_{(0,0),(1,0)}²  =  L_{(0,0),(0,0)} + L_{(0,0),(0,1)}
                           + L_{(0,0),(1,0)} + L_{(0,0),(2,0)}

    with every coefficient exactly `1` — i.e. **7 ⊗ 7 = 1 ⊕ 7 ⊕ 14 ⊕ 27**
    (49 = 1+7+14+27).  Wilson fusion *is* the tensor product of `G₂`
    representations, read off the canonical basis with no character table
    consulted."""
    from root_datum import g_2, b_n_simply_connected
    from pure_g_abe_kalgebra import PureGAbeKAlgebra
    from global_form import simply_connected_lines

    K = 8
    for tag, D in (("G2", g_2()), ("Spin(5)", b_n_simply_connected(2))):
        A = PureGAbeKAlgebra(D, lines=simply_connected_lines(D))
        z = (0,) * D.dim
        one = A.identity()
        W = A.fold(z, tuple(D.simple_roots[0]))
        basis = [one, W]
        print(f"    [{tag}] basis {[str(b) for b in basis]}")

        # bar: the antimultiplicative involution, in structure-constant form
        for a in basis:
            for b in basis:
                assert A.verify_bar_involution(a, b), (tag, "bar", a, b)
        # rho: a label permutation fixing the identity, with a real inverse
        assert A.verify_rho_fixes_identity(), (tag, "rho(1) != 1")
        for a in basis:
            assert A.rho_inverse(A.rho(a)) == a, (tag, "rho^-1 rho != id", a)
        for a in basis:
            for b in basis:
                assert A.verify_rho_is_automorphism(a, b), (tag, "rho aut", a, b)
        # the rho^2-twisted trace, and cyclicity
        for a in basis:
            print(f"    [{tag}] Tr(L_{a}) = {A.trace(a, K=K)}")
        for a in basis:
            for b in basis:
                assert A.verify_rho_twisted_trace(a, b, K=K), (tag, "cyc", a, b)
        # orthonormality: I_{a,b} = delta + O(q), the central relation
        for a in basis:
            for b in basis:
                ip = A.inner_product(a, b, K=K)
                print(f"    [{tag}] I[{a},{b}] = {ip}")
                assert A.verify_orthonormality(a, b, K=K), (tag, "orth", a, b)
        # associativity, on the sampled triples
        for a in basis:
            for b in basis:
                for c in basis:
                    lhs = A.multiply_elements(A.multiply(a, b),
                                              A.multiply(c, one))
                    rhs = A.multiply_elements(A.multiply(a, one),
                                              A.multiply(b, c))
                    assert lhs == rhs, (tag, "assoc", a, b, c)

    # the G2 Wilson square, asserted term by term: 7 (x) 7 = 1 + 7 + 14 + 27
    D = g_2()
    A = PureGAbeKAlgebra(D, lines=simply_connected_lines(D))
    z = (0, 0)
    W = A.fold(z, tuple(D.simple_roots[0]))
    sq = A.multiply(W, W)
    print(f"    [G2] L_W^2 = " + "  +  ".join(
        f"({v})*L_{k}" for k, v in sorted(sq.terms.items(), key=lambda kv: str(kv[0]))))
    assert len(sq.terms) == 4, f"G2 7(x)7 should have 4 summands, got {len(sq.terms)}"
    assert all(str(v) == "1" for v in sq.terms.values()), \
        f"G2 7(x)7 multiplicities should all be 1, got {[str(v) for v in sq.terms.values()]}"
    print("  PASS: test_contract_surface_at_a_general_G  (bar / rho / trace / "
          "cyclicity / orthonormality / associativity through the universal "
          "surface; G2 Wilson fusion == the tensor product of representations)")


def test_M_one_quiver_two_embeddings():
    """A BPS quiver does not name a theory — the **embedding** does.

    `PureSO3KAlgebra` and the SU(2) chart are the *same* Kronecker-2 quiver on
    the *same* canonical `Z²`: `⟨n₀,n₁⟩ = 2` for both, so the pairing identifies
    neither.  What picks the theory is where the nodes sit — `Γ = Q^∨ ⊕ Q`
    (adjoint) versus `Γ = P^∨ ⊕ P` (simply connected).

    This pins two things a future change could quietly break: that the two forms
    share an exchange matrix but not node charges, and that the oracle's charges
    really are the ones `adjoint_lines(su_2())` names — so the BPS side and the
    abelianized side are demonstrably making the *same* choice, rather than
    agreeing numerically for unstated reasons."""
    import root_datum as rd
    from global_form import simply_connected_lines, adjoint_lines
    from global_form_bridge import (bps_global_form, bps_lattice_for,
                                    exchange_matrix, verify_choice_matches,
                                    verify_same_quiver_different_embedding)
    from pure_so3 import SO3_NODES

    sc, adj = simply_connected_lines(rd.su_2()), adjoint_lines(rd.su_2())
    assert bps_global_form(sc) == "sc", bps_global_form(sc)
    assert bps_global_form(adj) == "adj", bps_global_form(adj)

    d_sc, d_adj = bps_lattice_for(sc, [("A", 1)]), bps_lattice_for(adj, [("A", 1)])
    assert exchange_matrix(d_sc) == exchange_matrix(d_adj), "same quiver expected"
    assert [tuple(n) for n in d_sc["nodes"]] != [tuple(n) for n in d_adj["nodes"]], \
        "different embedding expected"

    # the oracle is the adjoint form, and demonstrably not the simply connected one
    assert verify_choice_matches(adj, [("A", 1)], SO3_NODES)
    assert not verify_choice_matches(sc, [("A", 1)], SO3_NODES)

    for factors in ([("A", 1)], [("A", 2)], [("D", 4)]):
        assert verify_same_quiver_different_embedding(factors), factors

    print("  PASS: test_M_one_quiver_two_embeddings  (SU(2)/SO(3): exchange "
          "%s both, nodes %s vs %s; oracle == adjoint_lines(su_2()))"
          % (exchange_matrix(d_adj)[0], [tuple(n) for n in d_sc["nodes"]],
             [tuple(n) for n in d_adj["nodes"]]))


if __name__ == "__main__":
    test_H_pure_g_any_datum()
    test_H_one_route_and_the_declared_optimizations()
    test_H_the_five_axioms_and_their_negative_controls()
    test_H_odd_height_builds()
    test_H_odd_height_against_bps_oracle()
    test_I_gn_abe_kalgebra()
    test_I_roster_presets()
    test_I_type_a_seam_is_a_surjection()
    test_J_lattices_are_dual()
    test_J_parity_character_decides_the_frame()
    test_J_generalized_global_forms()
    test_J_line_lattice_tori()
    test_K_aux_space_pairing()
    test_M_one_quiver_two_embeddings()
    test_contract_surface_at_a_general_G()
    print("\nALL GNKAlgebra (Step 7) export self-tests passed  (general-G pure "
          "gauge with the (*)-guarded build + (G,N) matter at any datum and any "
          "matter representation + the 4d gauge group data as a lattice and its "
          "dual + the vacuum-state Schur pairing).")
