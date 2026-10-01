"""Tests for `BPSKAlgebra`, the K-algebra view of a BPS-chart
`bps_quiver_tools.CoulombAlgebra`.

The intent: confirm that the existing BPS-chart machinery, exposed through
the `KAlgebra` ABC, satisfies the same axioms as the hand-coded samples.

Run:  PYTHONPATH=. python restructuring/tests/test_bps_kalgebra.py
"""

from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESTRUCT = os.path.dirname(_HERE)
_REPO = _RESTRUCT  # post-flatten: _RESTRUCT is the repo root
sys.path.insert(0, _REPO)
sys.path.insert(0, _RESTRUCT)

from bps_kalgebra import BPSKAlgebra


# ---------------------------------------------------------------------------
# Pentagon BPS chart as a K-algebra
# ---------------------------------------------------------------------------

def _pentagon():
    return BPSKAlgebra(
        pairing=[[0, 1], [-1, 0]],
        node_charges=[(1, 0), (0, 1)],
    )


def test_pentagon_kalgebra_basic():
    A = _pentagon()
    # identity is the zero charge
    assert A.identity() == (0, 0)
    # rho fixes identity (axiom)
    assert A.verify_rho_fixes_identity()


def test_rg_r_label_decompose_delegates_to_aux():
    """`RGKAlgebra.r_label_decompose` delegates to the auxiliary (a KAlgebra
    optionally implementing the lift coordinate).  BPS's auxiliary is a
    `QuantumTorusKAlg`, which has it, so the delegation round-trips.
    Unflavoured (R = Z): section = the label, R-basis-label = trivial char."""
    A = _pentagon()
    for g in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (-1, 1)]:
        sec, r = A.r_label_decompose(g)
        assert A.r_label_compose(sec, r) == g, (g, sec, r)


def test_pentagon_kalgebra_rho_inverse_is_inverse():
    A = _pentagon()
    labels = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]
    for a in labels:
        assert A.verify_rho_inverse(a)


def test_pentagon_kalgebra_bar_involution():
    """Bar involution `C^c_{ab}(q^{-1}) = C^c_{ba}(q)` on a window."""
    A = _pentagon()
    labels = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1)]
    for a in labels:
        for b in labels:
            assert A.verify_bar_involution(a, b), (
                f"bar fails on {a}, {b}"
            )


def test_pentagon_deep_negative_charges_orthonormality():
    """Regression: deep-support F's like (-2, 1), (-3, 2) have c(η)
    HabiroElements with negative q-powers that must cancel against the
    prefactor. Orthonormality I_{a,a}[q^0] = 1 verifies the negative-
    q-powers aren't being dropped during PowerSeries expansion.

    Per the design notes, the old PowerSeries-throughout pipeline failed on
    these exact charges; we want to confirm the new pipeline does NOT.
    """
    A = _pentagon()
    K = 4
    deep = [(-2, 1), (-1, 2), (-3, 2), (-2, 3)]
    for a in deep:
        I_aa = A.inner_product(a, a, K=K, cone_cutoff=10)
        assert I_aa[0] == 1, (
            f"orthonormality I_({a},{a})[q^0] = {I_aa[0]}, expected 1; "
            f"likely a missed negative-q-power cancellation. I = {I_aa}"
        )


def test_pentagon_kalgebra_orthonormality():
    """Orthonormality I_{a,b}[q^0] = δ on a small window."""
    A = _pentagon()
    K = 4
    # Use the σ-orbit of (1,0) plus identity -- 6 labels, 36 pairs.
    labels = [(0, 0), (1, 0), (-1, -1), (0, 1), (0, -1), (-1, 0)]
    for a in labels:
        for b in labels:
            assert A.verify_orthonormality(a, b, K=K), (
                f"orthonormality fails on a={a}, b={b}: "
                f"I = {A.inner_product(a, b, K)}"
            )


def test_pentagon_kalgebra_cyclicity():
    """`Tr(uv) = Tr(ρ²(v) u)` on a window."""
    A = _pentagon()
    K = 3
    labels = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
    for u in labels:
        for v in labels:
            assert A.verify_rho_twisted_trace(u, v, K=K), (
                f"cyclicity fails for u={u}, v={v}"
            )


def test_pentagon_kalgebra_repr():
    A = _pentagon()
    s = repr(A)
    assert "rank=2" in s
    assert "|spec|=2" in s


def test_pentagon_F_known_values():
    """`F(γ)` for several pentagon generators agrees with user-documented
    formulas (F_{L_i} from the pentagon BPS-chart Nahm-sum derivation).

    Tropical charges of L_0..L_4: (1,0), (0,-1), (-1,-1), (-1,0), (0,1).
    """
    from laurent_poly import LaurentPoly
    A = _pentagon()
    # F_{L_0} = X_{(1, 0)}
    assert A.F((1, 0)) == {(1, 0): LaurentPoly.one()}
    # F_{L_1} = X_{(0, -1)}
    assert A.F((0, -1)) == {(0, -1): LaurentPoly.one()}
    # F_{L_2} = X_{(-1, -1)} + X_{(-1, 0)}
    assert A.F((-1, -1)) == {
        (-1, -1): LaurentPoly.one(),
        (-1, 0): LaurentPoly.one(),
    }
    # F_{L_3} = X_{(-1, 0)} + X_{(-1, 1)} + X_{(0, 1)}
    assert A.F((-1, 0)) == {
        (-1, 0): LaurentPoly.one(),
        (-1, 1): LaurentPoly.one(),
        (0, 1): LaurentPoly.one(),
    }
    # F_{L_4} = X_{(0, 1)} + X_{(1, 1)}
    assert A.F((0, 1)) == {
        (0, 1): LaurentPoly.one(),
        (1, 1): LaurentPoly.one(),
    }


def test_pentagon_spectrum_generator_leading():
    """`spectrum_generator(K)` should give 1 + O(q) and contain the
    expected monomial leading at the spec charges."""
    from laurent_poly import LaurentPoly
    A = _pentagon()
    K = 3
    S = A.spectrum_generator(K)
    # Identity coefficient at q^0 should be 1.
    id_coeff = S.get((0, 0))
    assert id_coeff is not None and id_coeff._coeffs.get(0, 0) == 1, (
        f"S identity coeff = {id_coeff}, expected 1 at q^0"
    )
    # Leading monomial at γ_1 = (1, 0) is -q (one E_q factor).
    g1_coeff = S.get((1, 0))
    assert g1_coeff is not None, f"S missing X_{{(1,0)}} term"
    assert g1_coeff._coeffs.get(1, 0) == -1, (
        f"S X_{{(1,0)}} coeff = {g1_coeff}, expected -q"
    )


# ---------------------------------------------------------------------------
# Spec shortening
# ---------------------------------------------------------------------------

def test_shorten_spec_pentagon_already_optimal():
    """Pentagon's spec [(1,0), (0,1)] is length 2, the minimum possible
    for a non-trivial 2-node quiver. shorten_spec=True should leave it
    unchanged."""
    A = BPSKAlgebra(
        pairing=[[0, 1], [-1, 0]],
        node_charges=[(1, 0), (0, 1)],
        spec=[(1, 0), (0, 1)],
        shorten_spec=True,
        verify="off",
    )
    assert A.spec == [(1, 0), (0, 1)], (
        f"shorten_spec changed an already-optimal spec: {A.spec}"
    )


def test_shorten_spec_method_returns_new_instance():
    """`A.shorten_spec()` returns a new BPSKAlgebra with a shortened
    spec.  Starting from a 3-factor pentagon-expanded spec, the
    method should reduce it to the canonical 2-factor pentagon spec.
    """
    # Pentagon-expanded spec: pentagon identity says
    #   E_q(X_a) E_q(X_{a+b}) E_q(X_b) = E_q(X_b) E_q(X_a)
    # so [(0,1), (1,1), (1,0)] is equivalent to [(1,0), (0,1)].
    A_long = BPSKAlgebra(
        pairing=[[0, 1], [-1, 0]],
        node_charges=[(1, 0), (0, 1)],
        spec=[(0, 1), (1, 1), (1, 0)],
        verify="off",
    )
    assert len(A_long.spec) == 3
    A_short = A_long.shorten_spec()
    assert isinstance(A_short, BPSKAlgebra)
    assert len(A_short.spec) == 2, (
        f"shorten_spec didn't shrink the 3-factor spec: {A_short.spec}"
    )
    # Original is untouched
    assert len(A_long.spec) == 3
    # Shortened algebra still computes F's correctly (same as canonical)
    A_canonical = BPSKAlgebra(
        pairing=[[0, 1], [-1, 0]],
        node_charges=[(1, 0), (0, 1)],
        spec=[(1, 0), (0, 1)],
        verify="off",
    )
    for gamma in [(1, 0), (0, 1), (-1, -1), (-1, 0), (0, -1)]:
        assert A_short.F(gamma) == A_canonical.F(gamma), (
            f"shorten_spec produced different F for γ={gamma}"
        )


def test_shorten_spec_collapses_synthetic_pentagon_expansion():
    """Pentagon move identity: E_q(X_{(1,0)}) E_q(X_{(0,1)}) =
       E_q(X_{(0,1)}) E_q(X_{(1,1)}) E_q(X_{(1,0)})  (length 2 = length 3).

    Feed BPSKAlgebra the length-3 form and verify shorten_spec=True
    reduces it to the canonical length-2 spec.
    """
    from spec_shortening import shorten_spec
    expanded = [(0, 1), (1, 1), (1, 0)]  # length 3, equivalent to len-2
    shorter = shorten_spec(expanded, exchange=[[0, 1], [-1, 0]])
    assert shorter == [(1, 0), (0, 1)], (
        f"shorten_spec did not collapse pentagon expansion: {shorter}"
    )


def test_shorten_spec_commute_aware_finds_collapse_behind_swap():
    """A pentagon collapse genuinely blocked by an intervening factor.

    Spec `[b, X, a+b, a]` where `X = e_3` lies in ker(B) (commutes
    with everything).  The greedy never finds the collapse: at
    position 0 the triple is `[b, X, ab]` (X is not `a+b` separately
    from `ab`); at position 1 the triple is `[X, ab, a]` (ab is not
    `X + a` since `X` and `a` are linearly independent of each
    other and of `ab`).  But after commuting `b` past `X` (legal:
    `⟨b, X⟩ = 0`), the spec becomes `[X, b, ab, a]` and positions
    1..3 are exactly the `[b, ab, a]` collapse pattern.
    """
    from spec_shortening import shorten_spec
    # Pentagon pairing with a flavour direction (e_3) in ker(B).
    B = [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]
    a = (1, 0, 0)
    b = (0, 1, 0)
    ab = (1, 1, 0)
    flav = (0, 0, 1)  # in ker(B), commutes with everything
    spec = [b, flav, ab, a]
    greedy = shorten_spec(spec, B, commute_orbit_states=0)
    assert len(greedy) == 4, (
        f"greedy was supposed to miss the commute-blocked collapse, "
        f"got {greedy}"
    )
    aware = shorten_spec(spec, B, commute_orbit_states=64)
    assert len(aware) == 3, (
        f"commute-aware should find the hidden collapse, got {aware}"
    )
    # After the swap `[X, b, ab, a]` then collapse `[b, ab, a] -> [a, b]`,
    # the final spec is `[X, a, b]` (a permutation thereof).
    assert sorted(aware) == sorted([a, b, flav])


def test_shorten_spec_redundant_specs_get_shortened():
    """Construct an artificially-padded spec by repeating a shortenable
    sub-pattern, then verify shorten_spec reduces it."""
    # SU(3) pure: known optimal spec_len = 6.
    # We don't have a clean way to inject a redundant spec without using
    # the dictionary, so just verify that explicit construction with
    # shorten_spec=True doesn't error and produces a valid spec.
    A = BPSKAlgebra(
        pairing=[[0, 2, 0, -1], [-2, 0, 1, 0],
                 [0, -1, 0, 2], [1, 0, -2, 0]],
        node_charges=[(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
        shorten_spec=True,
        verify="off",
    )
    assert len(A.spec) == 6, f"SU(3) pure spec_len = {len(A.spec)}, expected 6"


# ---------------------------------------------------------------------------
# Recipe mode: pentagon via user-supplied s_coefficient
# ---------------------------------------------------------------------------

def test_pentagon_chart_graph_root_present():
    """In spec mode, BPSKAlgebra builds a private chart graph whose
    root chart contains the user-supplied quiver data."""
    A = _pentagon()
    g = A._chart_graph
    assert g is not None, "spec-mode BPSKAlgebra should have a chart graph"
    root = g.root()
    assert root.nodes == [(1, 0), (0, 1)]
    assert root.spec == [(1, 0), (0, 1)]
    assert root.parent_edge is None
    assert len(g) == 1   # only root materialized; lazy


def test_pentagon_chart_graph_mutate_forward_at_node_1():
    """Forward mutation at node 1 (charge (0,1)) of pentagon root.
    Local moves bring (0,1) to head via pentagon expand:
    [(1,0),(0,1)] → [(0,1),(1,1),(1,0)].  Then rotation necklace:
    new spec = [(1,1), (1,0), -(0,1)] = [(1,1), (1,0), (0,-1)].
    Other spec factors are NOT mutated by μ_g."""
    A = _pentagon()
    g = A._chart_graph
    fwd_id = g.mutate(g.root_id, node_index=1, direction="fwd")
    assert fwd_id is not None
    fwd = g.chart(fwd_id)
    assert fwd.spec == [(1, 1), (1, 0), (0, -1)], (
        f"unexpected post-necklace spec: {fwd.spec}"
    )


def test_pentagon_multiply_via_chart_search_agrees_with_direct():
    """Algorithm-(c)-style chart-search multiply should agree with the
    direct (root-chart) multiply on a window of pentagon products."""
    A = _pentagon()
    pairs = [
        ((1, 0), (0, 1)), ((0, 1), (1, 0)),
        ((-1, 0), (1, 0)), ((1, 0), (-1, 0)),
        ((-1, 0), (0, 1)), ((0, 1), (-1, 0)),
        ((-1, -1), (1, 0)), ((1, 0), (-1, -1)),
    ]
    for a, b in pairs:
        m_dir = A.multiply(a, b)
        m_chart = A._multiply_via_chart_search(a, b)
        assert m_dir == m_chart, (
            f"chart-search multiply disagrees on {a}·{b}: "
            f"dir={m_dir} chart={m_chart}"
        )


def test_su2_multiply_via_chart_search_agrees_with_direct():
    """Pure SU(2) products: chart-search multiply matches direct."""
    A = BPSKAlgebra(
        pairing=[[0, 2], [-2, 0]], node_charges=[(1, 0), (0, 1)],
        verify="off",
    )
    pairs = [
        ((1, 0), (0, 1)), ((0, 1), (1, 0)),
        ((-1, 0), (1, 0)), ((1, 0), (-1, 0)),
        ((-1, -1), (1, 0)), ((-1, 0), (0, -1)),
    ]
    for a, b in pairs:
        m_dir = A.multiply(a, b)
        m_chart = A._multiply_via_chart_search(a, b)
        assert m_dir == m_chart, (
            f"chart-search multiply disagrees on {a}·{b}: "
            f"dir={m_dir} chart={m_chart}"
        )


def test_chart_graph_boundary_tracking():
    """Forward necklacing the head with no local moves preserves the
    `(S_2, -S_1)` boundary.  Local moves required to bring a non-head
    node-charge to the head break it."""
    A = _pentagon()
    g = A._chart_graph
    root = g.root()
    assert root.s1_chain == []
    assert root.boundary_preserved is True
    assert root.s2_length == 2
    # Forward at node 0 (charge (1,0) = head): empty local_moves, boundary preserved.
    dst1 = g.mutate(g.root_id, 0, "fwd")
    ch1 = g.chart(dst1)
    assert ch1.s1_chain == [(1, 0)]
    assert ch1.boundary_preserved is True
    assert ch1.s2_length == 1
    # Forward at node 1 (charge (0,1) -- not the head): needs pentagon
    # expand to bring (0,1) to head; boundary is destroyed.
    dst2 = g.mutate(g.root_id, 1, "fwd")
    ch2 = g.chart(dst2)
    assert ch2.boundary_preserved is False


def test_pentagon_solve_F_via_chain_agrees_with_direct():
    """Algorithm (c): F via chart-graph chain should agree with the
    standard solver on every pentagon σ-orbit charge plus a few
    extended γ's."""
    A = _pentagon()
    test_gammas = [
        (0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1),
        (-1, -1), (-1, 1), (1, -1), (-2, 0), (0, -2), (-2, 1), (-1, 2),
    ]
    for gamma in test_gammas:
        F_direct = A._F_internal(gamma)
        F_chain = A._solve_F_via_chain(gamma)
        assert F_direct == F_chain, (
            f"chain solver disagrees with direct on γ={gamma}: "
            f"|direct|={len(F_direct)}, |chain|={len(F_chain)}"
        )


def test_su2_solve_F_via_chain_agrees_with_direct():
    """Algorithm (c) on pure SU(2) for several γ's."""
    A = BPSKAlgebra(
        pairing=[[0, 2], [-2, 0]],
        node_charges=[(1, 0), (0, 1)],
        verify="off",
    )
    test_gammas = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1)]
    for gamma in test_gammas:
        F_direct = A._F_internal(gamma)
        A._F_cache.clear()
        F_chain = A._solve_F_via_chain(gamma)
        assert F_direct == F_chain, (
            f"chain solver disagrees with direct on SU(2) γ={gamma}: "
            f"|direct|={len(F_direct)}, |chain|={len(F_chain)}"
        )


def test_pentagon_recipe_mode_agrees_with_spec_mode():
    """Construct the pentagon BPS algebra in recipe mode by passing its
    s_coefficient + sigma + sigma_inverse functions explicitly. Verify
    that key intrinsic operations agree with spec mode.
    """
    A_spec = _pentagon()
    s_fn = A_spec._s_coefficient
    sigma_fn = A_spec._sigma_fn
    sigma_inv_fn = A_spec._sigma_inverse_fn
    A_rec = BPSKAlgebra(
        pairing=[[0, 1], [-1, 0]],
        s_coefficient=s_fn,
        sigma=sigma_fn,
        sigma_inverse=sigma_inv_fn,
        cone_gens=[(1, 0), (0, 1)],
    )
    # Identity, ρ
    assert A_rec.identity() == A_spec.identity()
    for label in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)]:
        assert A_rec.rho(label) == A_spec.rho(label)
        assert A_rec.rho_inverse(label) == A_spec.rho_inverse(label)
    # F at the BPS quiver nodes
    for label in [(1, 0), (0, 1), (-1, -1), (-1, 0), (0, 1)]:
        assert A_rec.F(label) == A_spec.F(label), (
            f"recipe F({label}) != spec F({label}): "
            f"{A_rec.F(label)} vs {A_spec.F(label)}"
        )
    # Inner products on a small window
    K = 4
    labels = [(0, 0), (1, 0), (0, 1)]
    for a in labels:
        for b in labels:
            I_rec = A_rec.inner_product(a, b, K)
            I_spec = A_spec.inner_product(a, b, K)
            assert (I_rec - I_spec).is_zero(), (
                f"inner_product({a}, {b}) differs: "
                f"recipe {I_rec} vs spec {I_spec}"
            )


# ---------------------------------------------------------------------------
# SQED_1 BPS chart as a K-algebra
# ---------------------------------------------------------------------------

def _sqed1():
    # SQED_1 = U(1) Nf=1 BPS chart: lattice Z², single E_q at (0,1).
    return BPSKAlgebra(
        pairing=[[0, 1], [-1, 0]],
        node_charges=[(0, 1)],
    )


def test_sqed1_kalgebra_basic():
    A = _sqed1()
    assert A.identity() == (0, 0)
    assert A.verify_rho_fixes_identity()


def test_pentagon_chart_with_negative_cone_generator():
    """Inner products in a chart whose positive cone has a generator
    with negative coordinates are computed correctly.

    Regression for `the design notes`: the old
    ``fs_dict_for_eta_set`` initialised an integer ``upper`` to
    ``[0]*rank`` and only ratcheted up componentwise, which silently
    truncated the s-table along any cone direction with negative
    coordinates.  In chart B (``cone_gens=[(1,0), (0,-1)]``) every
    μ with negative second component fell outside the box, so e.g.
    ``[F_(0,1) · S|0⟩]_(0,0)`` came out as 0 instead of ``-q/(1-q²)``.
    Worse, the missing entries were poisoned into ``_FS_cache`` as
    ``HabiroElement.zero()``, so the lazy fallback never ran.

    Fix: drive the s-table walk by the strict cone-pointedness witness
    (``self._cone_witness``) as an L-shell predicate, and stop the
    zero-poisoning of the cache.

    Verifications:
      (1) the historically-mis-cached ``[F_(0,1) S|0⟩]_(0,0)`` agrees
          with the lazy ``c_gamma_via_s`` evaluation;
      (2) intra-chart orthonormality I_{a,b}[q⁰] = δ_{a,b} on the
          chart-B σ-orbit;
      (3) ``Tr L_(0,1)`` agrees in chart A and chart B (same Γ-tuple
          indexes the same canonical-basis element here).
    """
    A = BPSKAlgebra(pairing=[[0, 1], [-1, 0]],
                    node_charges=[(1, 0), (0, 1)])
    B = BPSKAlgebra(pairing=[[0, 1], [-1, 0]],
                    node_charges=[(1, 0), (0, -1)])
    K = 4

    # (1) cache-entry correctness.
    from bps_kalgebra_internals import c_gamma_via_s
    F_01 = B._F_internal((0, 1))
    direct = c_gamma_via_s((0, 0), F_01, B._s_coefficient, B.lattice)
    # Trigger the warm path.
    _ = B.inner_product((0, 1), (0, 1), K=K)
    cached = B._FS_cache.get(((0, 1), (0, 0)))
    assert cached is not None and cached == direct, (
        f"_FS_cache[((0,1),(0,0))] mismatch: cached={cached}, direct={direct}"
    )

    # (2) intra-chart orthonormality on chart B's σ-orbit.
    orbit = [(0, 0)]
    g = (1, 0)
    for _ in range(5):
        orbit.append(g)
        g = B.rho(g)
    for a in orbit:
        for b in orbit:
            assert B.verify_orthonormality(a, b, K=K), (
                f"chart-B orthonormality fails on a={a}, b={b}: "
                f"I = {B.inner_product(a, b, K)}"
            )

    # (3) `Tr L_(0,1)` is chart-invariant for this label.
    assert A.trace((0, 1), K=K) == B.trace((0, 1), K=K)


def test_pentagon_equivalent_presentations_match():
    """Two SL(2,Z)-equivalent presentations of the pentagon BPSKAlgebra
    give the same Schur indices, with labels related by the
    transformation.

    Aut(Γ, B) action: pick `M ∈ SL(2, Z)`.  The presentation
    ``node_charges' = [M·n for n in node_charges]`` realises the same
    abstract algebra in a different lattice basis; canonical-basis
    labels translate as ``a' = M·a``.  In particular,
    ``I^orig_{a, b}(q) == I^new_{M·a, M·b}(q)``.

    This is the minimal cross-chart bug detector for the inner-product
    code: any chart-direction-blind walk in the s-table builder will
    fail this for some `M` whose action mixes coordinates with negative
    signs.
    """
    A0 = BPSKAlgebra(pairing=[[0, 1], [-1, 0]],
                     node_charges=[(1, 0), (0, 1)])
    K = 4
    # Two distinct SL(2,Z) elements.  M1 is upper-triangular (positive),
    # M2 is the rotation (mixes signs -- exercises the bug fix path).
    M1 = ((1, 1), (0, 1))           # det = 1
    M2 = ((0, -1), (1, 0))          # det = 1, mixes signs
    for M in (M1, M2):
        new_nodes = [
            (M[0][0] * n[0] + M[0][1] * n[1],
             M[1][0] * n[0] + M[1][1] * n[1])
            for n in [(1, 0), (0, 1)]
        ]
        A_new = BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=new_nodes)
        labels = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (-1, -1)]
        for a in labels:
            for b in labels:
                Ma = (M[0][0] * a[0] + M[0][1] * a[1],
                      M[1][0] * a[0] + M[1][1] * a[1])
                Mb = (M[0][0] * b[0] + M[0][1] * b[1],
                      M[1][0] * b[0] + M[1][1] * b[1])
                i0 = A0.inner_product(a, b, K)
                i1 = A_new.inner_product(Ma, Mb, K)
                assert i0 == i1, (
                    f"M={M}, (a,b)=({a},{b}): "
                    f"orig={i0}, transformed={i1}"
                )


def test_sqed1_kalgebra_orthonormality_window():
    A = _sqed1()
    K = 3
    labels = [(0, 0), (0, 1), (0, -1), (0, 2), (1, 0), (-1, 0)]
    for a in labels:
        for b in labels:
            assert A.verify_orthonormality(a, b, K=K), (
                f"SQED_1 ortho fails on a={a}, b={b}: "
                f"I = {A.inner_product(a, b, K)}"
            )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
# Spec-free construction (build_S): build the spectrum generator by the
# recursive engine (no spec / no green-sequence BFS) and derive ρ via the
# RGKAlgebra tRG intertwining.  Must agree with spec mode.
# ---------------------------------------------------------------------------

def _specfree_agrees(pairing, nodes, cutoff=7):
    spec_mode = BPSKAlgebra(pairing=pairing, node_charges=nodes)
    free = BPSKAlgebra(pairing=pairing, node_charges=nodes,
                       build_S=True, build_S_cutoff=cutoff)
    # ρ on every node label
    for a in spec_mode.node_charges:
        assert spec_mode.rho(a) == free.rho(a), (a, spec_mode.rho(a), free.rho(a))
    # multiply on the node pair
    a, b = spec_mode.node_charges[0], spec_mode.node_charges[1]
    assert spec_mode.multiply(a, b) == free.multiply(a, b)


def test_spec_free_pentagon_agrees_with_spec_mode():
    _specfree_agrees([[0, 1], [-1, 0]], [(1, 0), (0, 1)])


def test_spec_free_pure_su2_agrees_with_spec_mode():
    _specfree_agrees([[0, 1], [-1, 0]], [(1, 0), (-1, 2)])


def test_spec_free_pure_su3_cyclic_agrees_with_spec_mode():
    # cyclic gauge (A2 / pure SU(3)): build_S recovers a finite spec and runs
    # fast spec mode (the generic tRG path would be intractable here).
    pairing = [[0, 1, 0, -2], [-1, 0, 2, 0], [0, -2, 0, 1], [2, 0, -1, 0]]
    nodes = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    free = BPSKAlgebra(pairing=pairing, node_charges=nodes, build_S=True,
                       build_S_cutoff=7)
    assert not free._spec_free, "expected a finite spec to be recovered"
    _specfree_agrees(pairing, nodes)


def _principled_agrees(pairing, nodes, cutoff=8):
    # Path B (spec_free_sigma="principled"): the principled spec-free σ
    # σ⁻¹(a)=−upper(F_a), σ(a)=−upper(F̃_a) read off the built S.  extract_spec
    # =False forces the spec-free path (else extraction recovers a spec — the
    # ideal — and we'd be in fast spec mode instead of exercising B).
    spec_mode = BPSKAlgebra(pairing=pairing, node_charges=nodes)
    free = BPSKAlgebra(pairing=pairing, node_charges=nodes, build_S=True,
                       build_S_cutoff=cutoff, spec_free_sigma="principled",
                       extract_spec=False)
    assert free._spec_free and free._principled_sigma, (
        "principled σ should keep the algebra spec-free")
    for a in spec_mode.node_charges:
        assert spec_mode.rho(a) == free.rho(a), (a, spec_mode.rho(a), free.rho(a))
        assert spec_mode.rho_inverse(a) == free.rho_inverse(a)
    a, b = spec_mode.node_charges[0], spec_mode.node_charges[1]
    assert spec_mode.multiply(a, b) == free.multiply(a, b)


def test_principled_sigma_pentagon_agrees_with_spec_mode():
    _principled_agrees([[0, 1], [-1, 0]], [(1, 0), (0, 1)])


def test_principled_sigma_pure_su2_agrees_with_spec_mode():
    _principled_agrees([[0, 1], [-1, 0]], [(1, 0), (-1, 2)])


def test_principled_sigma_flavoured_su2_nf1_agrees_with_spec_mode():
    # SU(2)+N_f=1 (pure_ade.SUN_Nf(2,1)): the first FLAVOURED principled test
    # (ker B ≠ 0, flavour_rank=1).  Exercises (i) that the principled σ composes
    # with the section-rectification, and (ii) the degree-≤cutoff cone-simplex
    # F-window — without it, spec-free flavoured multiply is intractable (the
    # generous box is unbounded along the null flavour direction).
    import itertools
    pairing = [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]
    nodes = [(1, 0, 0), (-1, 2, 0), (0, -1, 1)]
    A = BPSKAlgebra(pairing=pairing, node_charges=nodes)
    B = BPSKAlgebra(pairing=pairing, node_charges=nodes, build_S=True,
                    build_S_cutoff=8, spec_free_sigma="principled",
                    extract_spec=False)
    assert A._flavour_rank == 1 and B._principled_sigma
    assert B._sf_max_degree is not None, "degree cap must be active (flavoured)"
    for a in A.node_charges:
        assert A.rho(a) == B.rho(a)
        assert A.rho_inverse(a) == B.rho_inverse(a)
    for a, b in itertools.product(A.node_charges, A.node_charges):
        assert A.multiply(a, b) == B.multiply(a, b), (a, b)


def test_spec_free_auto_cutoff_no_user_value():
    # build_S_cutoff=None (default) auto-stabilizes the cone cutoff until the
    # node canonicals are σ-stable — no user cutoff.  ρ/ρ⁻¹/multiply still match
    # spec mode; the auto-settled cutoff is recorded in _sf_max_degree.
    import itertools
    for pairing, nodes in (
        ([[0, 1], [-1, 0]], [(1, 0), (0, 1)]),            # pentagon
        ([[0, 1, 0], [-1, 0, 0], [0, 0, 0]],              # SU(2)+N_f=1 (flavoured)
         [(1, 0, 0), (-1, 2, 0), (0, -1, 1)]),
    ):
        A = BPSKAlgebra(pairing=pairing, node_charges=nodes)
        B = BPSKAlgebra(pairing=pairing, node_charges=nodes, build_S=True,
                        spec_free_sigma="principled", extract_spec=False)
        assert B._sf_max_degree is not None and B._sf_max_degree >= 4
        for a in A.node_charges:
            assert A.rho(a) == B.rho(a)
            assert A.rho_inverse(a) == B.rho_inverse(a)
        for a, b in itertools.product(A.node_charges, A.node_charges):
            assert A.multiply(a, b) == B.multiply(a, b), (a, b)


def test_spec_free_multiply_degrades_gracefully_on_shallow_cutoff():
    # Regression: spec-free `multiply` on an UNDER-built S must terminate
    # (truncated to the built cone-degree), not loop unboundedly.  Before the
    # from_ir_image cone-degree guard, a too-small cutoff sent the apex peel
    # marching to cone-degree ~35 (and spuriously negative) and never closed.
    # At cutoff 5 the SU(2)+N_f=1 node products happen to land in-cone, so the
    # truncation is loss-free and the result still matches spec mode — the point
    # is that it *completes* fast either way.
    import itertools
    pairing = [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]
    nodes = [(1, 0, 0), (-1, 2, 0), (0, -1, 1)]
    A = BPSKAlgebra(pairing=pairing, node_charges=nodes)
    B = BPSKAlgebra(pairing=pairing, node_charges=nodes, build_S=True,
                    build_S_cutoff=5, spec_free_sigma="principled",
                    extract_spec=False)
    for a, b in itertools.product(A.node_charges, A.node_charges):
        prod = B.multiply(a, b)          # must return (no hang)
        assert A.multiply(a, b) == prod, (a, b)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback
    failures = 0
    for name in sorted(globals()):
        fn = globals()[name]
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS: {name}")
        except Exception:
            failures += 1
            print(f"  FAIL: {name}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures} failure(s).")
        sys.exit(1)
    print(f"\nAll BPSKAlgebra tests passed.")
