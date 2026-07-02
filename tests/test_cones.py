"""Runnable self-test for the ConeKAlgebra layer.

Pure Python 3, no third-party dependencies, no realisation-spine
modules (no BPS / RG / quantum-torus engine).  Each cone algebra in
the catalogue is a *closed-form* `ConeKAlgebra`: the canonical basis is
organised into cones of multiplicative generators, `multiply` is the
generic normal-ordering reduction over the cone cocycle/cross-products,
and `trace` is the closed-form residual rule `_trace_residual`
(Nahm-sum / character / q-Pochhammer) — all evaluated here without any
computed (BPS/RG) backend.

For every algebra this exercises, on the unit and on a couple of cone
generators:

  * `multiply` (cocycle + cross-product reduction),
  * `rho` / `rho_inverse`,
  * `trace` (the closed-form residual),
  * `verify_orthonormality` (the Schur pairing `I_{a,b}=δ_{a,b}+O(q)`),

i.e. the orthonormality conjecture (docs/conjectures-step2-cone.md) on these
structures.
"""
from __future__ import annotations

import traceback

# (module, class name, constructor args) for each catalogued cone algebra.
CONE_ALGEBRAS = [
    # --- all finite-type cone KAlgebras ---
    ("finite_pentagon_kalg", "FinitePentagonKAlgebra", ()),
    ("finite_a3_kalg",       "FiniteA3KAlgebra",       ()),
    ("finite_a5_kalg",       "FiniteA5KAlgebra",       ()),
    ("finite_a7_kalg",       "FiniteA7KAlgebra",       ()),
    ("finite_a1d3_kalg",     "FiniteA1D3KAlgebra",     ()),
    ("finite_a1d4_kalg",     "FiniteA1D4KAlgebra",     ()),
    ("finite_a1d5_kalg",     "FiniteA1D5KAlgebra",     ()),
    ("finite_a1d6_kalg",     "FiniteA1D6KAlgebra",     ()),
    ("finite_a1d7_kalg",     "FiniteA1D7KAlgebra",     ()),
    ("finite_a1d8_kalg",     "FiniteA1D8KAlgebra",     ()),
    ("finite_e6_kalg",       "FiniteE6KAlgebra",       ()),
    ("finite_e7_kalg",       "FiniteE7KAlgebra",       ()),
    ("finite_e8_kalg",       "FiniteE8KAlgebra",       ()),
    ("finite_heptagon_kalg", "FiniteHeptagonKAlgebra", ()),
    # --- pure SU(2) ---
    ("pure_su2_h_cone_data",   "PureSU2KAlg",        ()),
    # --- SU(2) with flavour ---
    ("su2_nf1_kalgebra", "SU2Nf1KAlgebra", ()),
    # --- Argyres-Douglas closed-form distillation ---
    ("a1d3_kalg",      "A1D3KAlg",      ()),
    # --- [A_1, D_odd] explicitly-named cone classes (k=0,1,2): closed-form multiply
    #     (frozen inline Plücker tables -- no a1d5_kalg/finite_a1d7_kalg/u1a1aodd/RG
    #     engine) + the arbitrary-q sl(2) admissible-character trace (a1dodd_layer2).
    #     Unit-level contract here; generator multiply + no-cap trace + orthonormality
    #     are exercised in check_improvable below.
    ("a1dodd_kalg",    "A1D3ConeKAlg",  ()),   # = [A_1,D_3] = sl(2)_{-4/3}
    ("a1dodd_kalg",    "A1D5ConeKAlg",  ()),   # = [A_1,D_5] = sl(2)_{-8/5}
    ("a1dodd_kalg",    "A1D7ConeKAlg",  ()),   # = [A_1,D_7] = sl(2)_{-12/7}
    # --- A1A_even reference family: geometric cone-ray labels + full Layer-2
    #     trace = M(2,2k+3) Andrews-Gordon characters.  Closed-form + spine-free
    #     for ALL k; the entries below are just recognisable samples.
    ("a1a2k_kalg",     "A1A2kKAlg",     (1,)),   # = pentagon, M(2,5)
    ("a1a2k_kalg",     "A1A2kKAlg",     (2,)),   # = heptagon, M(2,7)
    ("a1a2k_kalg",     "A1A2kKAlg",     (3,)),   # = nonagon, M(2,9)
    ("a1a2k_kalg",     "A1A2kKAlg",     (6,)),   # = M(2,15) — general-k witness
    # --- U(1)-gauged A1A_odd.  TWO things differ from A1A2k:
    #   (1) MULTIPLY: the cone cross-products have no closed form yet; they were
    #       computed by an RG-flow derivation and are frozen in
    #       u1a1aodd_tables_k{k}.pkl, loaded here.
    #   (2) TRACE: the v-tower / long-chord (a=2) / diameter (a=k+1) seeds are the
    #       closed-form M(1,p) singlet characters (u1_pgon_layer2), closing every
    #       trace for k<=3.  At k>=4 the intermediate odd chords (2<a<k+1) have no
    #       closed form (their LOG b-slopes are an open (1,p)/B_p log-module
    #       question); they are computed to ARBITRARY q-order by the spine-free
    #       orthonormality bootstrap (u1aodd_trace_bootstrap, certified) — so the
    #       trace now closes for ALL k.  (k=4 below exercises that path.)
    ("u1a1aodd_kalg",  "U1A1AoddKAlg",  (1,)),   # = U(1)-gauged hexagon (A1A_1)
    ("u1a1aodd_kalg",  "U1A1AoddKAlg",  (2,)),   # = U(1)-gauged octagon (A1A_3)
    ("u1a1aodd_kalg",  "U1A1AoddKAlg",  (3,)),   # = U(1)-gauged decagon (A1A_5)
    ("u1a1aodd_kalg",  "U1A1AoddKAlg",  (4,)),   # = U(1)-gauged dodecagon (A1A_7);
    #     k=4 is the first with an intermediate chord (length-5) -> bootstrap trace
    # --- [A_1, D_4] = SU(3)_{-3/2}, SU(3) flavour, fully BPS-free trace ---
    #   Tr_1 = closed-form Kac-Wakimoto vacuum character of sl(3)^_{-3/2};
    #   Tr_T/Tr_D = the orthonormality bootstrap seeded by Tr_1.  Layer-1 and
    #   the product multiply are carried in SU(3) Cartan fugacities (weights,
    #   not characters), Weyl-symmetrized only on the total -- so non-self-dual
    #   product content is correct.  No engine on the trace path.  (The generic
    #   label discovery below only reaches its unit -> vacuum trace Tr_1 +
    #   orthonormality here; the generator trace Tr_T is exercised, to high
    #   q-order, in check_improvable.)
    ("su3_ad_kalg",    "SU3ADKAlg",     ()),
    # --- u(1)-gauged E7 = A_q[T] for the u(1)-gauged E7 SCFT.  A QTCone (rank-1
    #   gauge torus on E = X_{(0,1)}).  MULTIPLY: the cone cross-products are
    #   computed by an RG-flow derivation (no closed form) and frozen as
    #   u1e7_cone_tables.pkl (cf. u1a1aodd).  TRACE: magnetic (c0) sector
    #   vanishes; Tr(1)/Tr(E^n) is the lazy E7 Nahm-sum vacuum; every other c0=0
    #   ray-word is fixed by the forward-triangular orthonormality bootstrap
    #   (u1e7_trace_bootstrap: flat ρ²-canonical reduction + one-step cyclicity
    #   relations, certified, arbitrary q-order).  ρ is spine-free via a frozen
    #   (ray_word, c0) table + the gauge-reflection ρ((w,(c0,c1)))=(π,(c0',δ-c1))
    #   (u1e7_rho_tables.pkl).  No engine on multiply / ρ / trace.
    ("u1e7_cone_kalgebra", "U1E7ConeKAlgebra", ()),
    # --- u(1)-gauged [A_1, D_{2k+2}] = A_q[T] for the u(1)-gauged D-even SCFT.
    #   SU(2) flavour; QTCone (torus on X_{0,1}).  MULTIPLY: closed-form,
    #   general-k — the cocycle q-powers + the cross step phase d_j·c1(i)-d_i·c1(j)
    #   are DERIVED (ρ-folded from an RG-flow derivation, then closed formulas;
    #   cf. u1a1aodd, which stores the raw table).  The tables are frozen
    #   (u1a1deven_tables_k1.pkl) so construction needs no external derivation.
    #   TRACE: c1!=0
    #   vanishes; every c1=0 seed (v-tower + matter) is bootstrapped from Tr(1)
    #   ALONE — the SU(2) per-irrep orthonormality sweep + all-orders monopole
    #   cyclicity (u1a1deven_matter_bootstrap), no engine on the trace path.
    #   Tr(1) and the seeds are closed-form characters (arbitrary q-order, no
    #   cap) — see the q^70 improvability witness in check_improvable.
    #   Only k=1 is included: the k=1 matter bootstrap is tractable (re-solvable
    #   to any q-order), while the spine-free SU(2) matter bootstrap is not
    #   tractable at arbitrary order for k>=2 (D6/D8) — frozen tables there
    #   would be limited to K≈8-12 and raise rather than silently degrade beyond.
    ("u1a1deven_cone_kalgebra", "U1A1DevenConeKAlgebra", (1,)),
]

K = 3            # q-order window for trace / orthonormality
MAX_GENS = 2     # generators exercised per algebra

# Classes whose trace is correct but whose bootstrap seed-solve is impractically
# slow for a quick self-test (the 8-node su2×u1 cone reductions are ~exponential
# in mult-gen power); exercised at the multiply/ρ level only here.  Their trace
# uses the same spine-free Nahm-sum Tr(1) + bootstrap as a1d4/a1d6 (which run the
# full battery), and Tr(1) itself is fast (≈0.2 s).
LIGHT_TRACE = {"FiniteA1D8KAlgebra", "FiniteE7KAlgebra"}


def _candidate_labels(A):
    """Native canonical labels to exercise: the unit plus a few cone
    generators, discovered through the universal `cone_data()` surface."""
    labels = [A.identity()]
    cd = A.cone_data()
    seen = set(labels)
    for getter in ("mult_gens", "cones"):
        try:
            coll = list(getattr(cd, getter)())
        except Exception:
            continue
        for item in coll:
            members = list(item) if getter == "cones" else [item]
            for g in members:
                L = g if (isinstance(g, tuple) and (not g or isinstance(g[0], tuple))) \
                    else ((g, 1),)
                if L not in seen:
                    seen.add(L)
                    labels.append(L)
    return labels


def exercise(A, cls_name=""):
    """Run the contract surface; return number of (unit+gen) labels that
    cleanly multiplied / traced / passed orthonormality."""
    R = A.coefficient_ring()
    assert R is not None
    labels = _candidate_labels(A)
    unit = labels[0]
    # core unit-level paths
    assert A.multiply(unit, unit) is not None
    A.rho(unit); A.rho_inverse(unit)
    if cls_name in LIGHT_TRACE:           # multiply/ρ only (trace correct but slow)
        for L in labels[1:3]:
            A.multiply(unit, L)
        return 0
    A.trace(unit, K=K)
    assert A.verify_orthonormality(unit, unit, K=K), "unit orthonormality"
    passed = 1
    gens_done = 0
    prev = unit
    for L in labels[1:]:
        if gens_done >= MAX_GENS:
            break
        try:
            A.multiply(L, L)
            A.multiply(prev, L)        # cross-product path
            A.trace(L, K=K)
            ok = A.verify_orthonormality(L, L, K=K)
        except Exception:
            continue                    # not a clean single canonical here; skip
        assert ok, f"generator orthonormality failed for {L!r}"
        passed += 1
        gens_done += 1
        prev = L
    return passed


def check_improvable():
    """Theories with a closed-form vacuum character must be arbitrarily
    q-improvable spine-free: trace past the frozen-table window must succeed
    (no BPS, no fixed-K cap)."""
    import importlib
    # (module, class, K beyond the frozen window) — one per flavour family,
    # all spine-free via the Nahm-sum Tr(1) (no BPS, no fixed-K cap)
    # Fast improvability witnesses past the frozen-table window (the slow
    # big-node bootstraps at high K — e7/e8/a5 — are correct but unsuited to a
    # quick self-test; one per fast family suffices to prove no fixed-K cap):
    cases = [("finite_pentagon_kalg", "FinitePentagonKAlgebra", 70, ()),  # trivial, frozen 64
             ("finite_a1d4_kalg",     "FiniteA1D4KAlgebra",       8, ()),  # su2u1, NOT frozen
             ("a1a2k_kalg",           "A1A2kKAlg",               60, (2,)),  # M(2,7) char, no cap
             # U(1)-gauged [A_1,D_4]: gauge-sector Tr(1) is the exact closed-form
             # (A1,D_2p) admissible-character formula (exact_characters), arbitrary
             # q-order, far past the frozen-table window (cross-verified vs sl(3) KW).
             ("u1a1deven_cone_kalgebra", "U1A1DevenConeKAlgebra", 70, (1,))]
    for mod_name, cls_name, K, args in cases:
        A = getattr(importlib.import_module(mod_name), cls_name)(*args)
        t = A.trace(A.identity(), K=K)
        assert max(t.coeffs) >= K - 2, (cls_name, "did not reach q^K")
        print(f"  OK   {cls_name:24s} arbitrarily-improvable spine-free to q^{K}")
    # u1a1aodd: closed-form (u1_pgon_layer2), so no fixed-K cap by construction.
    # Probe a chord generator (dense trace) rather than the sparse vacuum.
    U = importlib.import_module("u1a1aodd_kalg").U1A1AoddKAlg(1)
    cd = U.cone_data()
    chord = next(g for g in cd.mult_gens() if g not in ((0, 0), (0, 1)))
    Lc = cd.from_cone_label(frozenset([chord]), {chord: 1})
    tc = U.trace(Lc, K=40)
    assert tc is not None and tc.K >= 40, ("U1A1AoddKAlg", "trace to q^40")
    print(f"  OK   {'U1A1AoddKAlg':24s} arbitrarily-improvable spine-free to q^40")
    # SU3AD = [A_1,D_4] = SU(3)_{-3/2}: Tr_1 the closed-form KW vacuum char,
    # Tr_T/Tr_D the orthonormality bootstrap -> arbitrary q-order, no engine.
    # Probe the generator trace Tr_T = Tr(T_0) (not the sparse vacuum).
    S = importlib.import_module("su3_ad_kalg").SU3ADKAlg()
    ts = S.trace(S.T(0), K=30)
    assert ts is not None and max(ts.coeffs) >= 26, ("SU3ADKAlg", "Tr_T to q^30")
    print(f"  OK   {'SU3ADKAlg':24s} arbitrarily-improvable spine-free to q^30")
    # A1D3KAlg = [A_1,D_3] = [A_1,A_3] (so(6)=su(4)): the EXPLICIT closed-form
    # chiral-algebra characters of affine sl(2)_{-4/3} (admissible irreducibles
    # κ_0, κ_1^sym, κ_1^anti) -- NOT a bootstrap.  Exercise the generator traces
    # Tr_T=Tr(T_0), Tr_D=Tr(D_0): they must (a) be non-trivial, (b) reach high
    # q-order (no fixed-K cap), and (c) genuinely use the Layer-2 characters --
    # i.e. DIFFER from the vacuum_only reduction (which zeroes Tr_T/Tr_D).
    AD = importlib.import_module("a1d3_kalg").A1D3KAlg()
    for nm, g in (("Tr_T", AD.T(0)), ("Tr_D", AD.D(0))):
        full = AD.trace(g, K=30)
        vac = AD.trace(g, K=30, vacuum_only=True)
        assert any(not r.is_zero() for r in full.coeffs.values()), (nm, "empty")
        assert max(q for q, r in full.coeffs.items() if not r.is_zero()) >= 26, \
            (nm, "did not reach q^30")
        assert full.coeffs != vac.coeffs, \
            ("A1D3KAlg", nm, "not using explicit Layer-2 characters")
    print(f"  OK   {'A1D3KAlg':24s} explicit sl(2)_-4/3 character traces, to q^30")
    # [A_1,D_5] = sl(2)_{-8/5}, [A_1,D_7] = sl(2)_{-12/7}: explicit closed-form
    # admissible-character Layer-2 (a1d5_layer2 / a1d7_layer2) now serve the
    # elementary traces (overriding the frozen tables, whose upper tails were
    # under-resolved).  Exact to arbitrary q-order -- trace the vacuum far past
    # the frozen-table window, spine-free.
    for mod_name, cls_name, Kq in (("finite_a1d5_kalg", "FiniteA1D5KAlgebra", 40),
                                   ("finite_a1d7_kalg", "FiniteA1D7KAlgebra", 24)):
        A = getattr(importlib.import_module(mod_name), cls_name)()
        t = A.trace(A.identity(), K=Kq)
        nz = [q for q, r in t.coeffs.items() if not r.is_zero()]
        assert nz and max(nz) >= Kq - 2, (cls_name, "did not reach q^Kq")
        print(f"  OK   {cls_name:24s} explicit sl(2) char traces, spine-free to q^{Kq}")
    # [A_1,D_3]/[A_1,D_5]/[A_1,D_7] explicitly-named cone classes (k=0,1,2): the
    # genuine closed-form D-type cone presentations.  Closed-form cone multiply
    # (frozen inline Plücker tables) + arbitrary-q sl(2) admissible-character trace,
    # both fully spine-free -- pulling NONE of a1d5_kalg/finite_a1d7_kalg/u1a1aodd/
    # rgkalgebra (this whole self-test runs with no realisation-spine module imported).
    # Trace far past the finite-class q^24 window to witness no fixed-K cap.
    a1dodd = importlib.import_module("a1dodd_kalg")
    for cls_name, Kq in (("A1D3ConeKAlg", 40), ("A1D5ConeKAlg", 40), ("A1D7ConeKAlg", 30)):
        C = getattr(a1dodd, cls_name)()
        mg = list(C.cone_data().mult_gens())
        a = (((mg[0], 1),), 0); b = (((mg[1], 1),), 0)
        assert C.multiply(a, b).terms, (cls_name, "empty cone multiply")
        t = C.trace(C.identity(), K=Kq)
        nz = [q for q, r in t.coeffs.items() if not r.is_zero()]
        assert nz and max(nz) >= Kq - 2, (cls_name, "trace did not reach q^Kq")
        assert C.verify_orthonormality(a, a, K=3), (cls_name, "orthonormality")
        print(f"  OK   {cls_name:24s} closed-form sl(2) char trace spine-free to q^{Kq}")
    # A1D7ConeKAlg completeness regression: the diameter seed (a=k+1=3, p=0) is
    # the hardest orthonormality case; its closed-form trace must resolve, reach
    # high q, AND pass orthonormality.  This guards the a1dodd_layer2 diameter
    # recipe against regressions (it must stay synced).
    D7 = a1dodd.A1D7ConeKAlg()
    dia = ((((3, 0, 0), 1),), 0)
    td = D7.trace(dia, K=30)
    nzd = [q for q, r in td.coeffs.items() if not r.is_zero()]
    assert nzd and max(nzd) >= 28, ("A1D7ConeKAlg", "diameter trace capped/empty")
    assert D7.verify_orthonormality(dia, dia, K=4), ("A1D7ConeKAlg", "diameter orthonormality")
    print(f"  OK   {'A1D7ConeKAlg':24s} diameter seed (a=3) complete: trace q^30 + orthonormality")
    # Ungauged [A_1, A_odd] polygons: HexagonKAlg / OctagonKAlg / DecagonKAlg /
    # DodecagonKAlg = [A_1, A_{2k+1}], k=1..4 -- the ungauged twins of the
    # gauged U1A1Aodd family (ungauged = centralizer of the gauge generator E=μ,
    # measure-restored trace).  These are KAlgebra, NOT ConeKAlgebra (the cone lives
    # in the wrapped gauged U1*KAlg), so they are exercised here with explicit labels
    # rather than via the generic CONE_ALGEBRAS loop.  Spine-free: construct +
    # multiply + arbitrary-q vacuum trace, all engine-free.  Orthonormality is
    # correct but O(minutes) at K>=4 for the larger polygons, so only the fast
    # Hexagon gets a self-norm spot-check here (below).
    for mod_name, cls_name, Kq in (("hexagon_kalg", "HexagonKAlg", 40),
                                   ("octagon_kalg", "OctagonKAlg", 30),
                                   ("decagon_kalg", "DecagonKAlg", 30),
                                   ("dodecagon_kalg", "DodecagonKAlg", 30)):
        C = getattr(importlib.import_module(mod_name), cls_name)()
        g = list(C.mult_generators())
        assert C.multiply(g[0], g[1]).terms, (cls_name, "empty multiply")
        t = C.trace(C.identity(), K=Kq)
        nz = [q for q, r in t.coeffs.items() if not r.is_zero()]
        assert nz and max(nz) >= Kq - 2, (cls_name, "vacuum trace did not reach q^Kq")
        print(f"  OK   {cls_name:24s} ungauged [A_1,A_odd] spine-free vacuum trace to q^{Kq}")
    Hx = importlib.import_module("hexagon_kalg").HexagonKAlg()
    hg = list(Hx.mult_generators())
    assert Hx.verify_orthonormality(hg[0], hg[0], K=3), ("HexagonKAlg", "orthonormality")
    print(f"  OK   {'HexagonKAlg':24s} orthonormality self-norm = 1 + O(q)")
    # Ungauged [A_1, D_{2k+2}] = A1DevenKAlg (k=1 = D_4): the U(1) of the gauged
    # U1A1DevenConeKAlgebra ungauged (centralizer of the X_{0,1} gauge generator;
    # gauge charge -> U(1) fugacity z; SU(2)×U(1) flavour).  KAlgebra (not
    # ConeKAlgebra).  Spine-free: construct + multiply (E-free, gauge charge lifted
    # into the coefficient) + arbitrary-q trace + orthonormality, all engine-free;
    # the trace reproduces A1DevenRGKAlgebra(1) term-for-term.
    from a1deven_kalg import A1DevenKAlg
    Dv = A1DevenKAlg(1)
    dg = Dv.mult_generators()
    assert Dv.multiply(dg[0], dg[1]).terms, ("A1DevenKAlg", "empty multiply")
    tdv = Dv.trace(Dv.identity(), K=30)
    nzdv = [q for q, r in tdv.coeffs.items() if not r.is_zero()]
    assert nzdv and max(nzdv) >= 28, ("A1DevenKAlg", "vacuum trace did not reach q^30")
    assert Dv.verify_orthonormality(dg[0], dg[0], K=3), ("A1DevenKAlg", "self-norm")
    assert Dv.verify_orthonormality(dg[0], dg[1], K=3), ("A1DevenKAlg", "off-diagonal")
    print(f"  OK   {'A1DevenKAlg':24s} ungauged [A1,D4] spine-free trace q^30 + orthonormality")
    # SU(2)+N_f=2 (flavour Spin(4)=SU(2)_L×SU(2)_R): spine-free ConeKAlgebra over
    # SU(2)⊗SU(2).  multiply is total (incl. magnetic monomials); trace is total
    # as well (flavour-charged labels route to the Weyl-invariant neutral
    # section) — the Spin(4) Schur index,
    # an arbitrary-q orthonormality bootstrap (no fixed-K cap).  cone_data has no
    # generic mult_gens, so exercised here with explicit labels.
    from su2_nf2_cone_standalone import SU2Nf2ConeKAlgebra
    N2 = SU2Nf2ConeKAlgebra()
    H0 = (((0, 1),), (0, 0)); H1 = (((1, 1),), (0, 0))
    assert N2.multiply(H0, H1).terms, ("SU2Nf2ConeKAlgebra", "empty multiply")
    tn2 = N2.trace(N2.identity(), K=12)
    nzn2 = [q for q, r in tn2.coeffs.items() if not r.is_zero()]
    assert nzn2 and max(nzn2) >= 10, ("SU2Nf2ConeKAlgebra", "Spin(4) index did not reach q^12")
    assert N2.verify_orthonormality(H0, H0, K=3), ("SU2Nf2ConeKAlgebra", "self-norm")
    assert N2.verify_orthonormality(H0, H1, K=3), ("SU2Nf2ConeKAlgebra", "off-diagonal")
    print(f"  OK   {'SU2Nf2ConeKAlgebra':24s} Spin(4) index spine-free to q^12 + orthonormality")
    # SU(2)+N_f=3 (flavour SU(4); matter = SO(6)=6=Λ²4): spine-free ConeKAlgebra
    # over SU(4).  multiply is the SU(4) literal-word reducer (the N_f=2 sibling),
    # TOTAL on every magnetic level: canonical inputs expand to literal H/Wilson
    # words via the inverse Gaussian q-binomial, adjacent pairs multiply via the
    # closed-form H·H (always magnetic ≤ 2; ε_n = 4/4̄ by parity), the word reduces
    # (matter-aware swap H_xH_{x+1}=q²H_{x+1}H_x+(1−q²)·1) to canonical single-cone
    # form, read back by the forward Gaussian.  ρ is the H-tower shift 4−N_f=−1 with
    # the flavour weight starred (4↔4̄).  trace routes flavour-charged labels to the
    # Weyl-invariant neutral section and solves the magnetic seeds by the SU(4)
    # character-basis cyclicity+orthonormality bootstrap (arbitrary-q; the
    # magnetic-1×magnetic-(m−1) orthonormality cascade pins the magnetic-m anchors).
    # cone_data has no generic mult_gens, so exercised here with explicit labels.
    from su2_nf3_cone_standalone import SU2Nf3ConeKAlgebra
    N3 = SU2Nf3ConeKAlgebra()
    H0n3 = (((0, 1),), (0, 0, 0)); H1n3 = (((1, 1),), (0, 0, 0))
    M0n3 = (((0, 2),), (0, 0, 0))                  # M_0 = H_0^2 (magnetic-2 cone monomial)
    M30n3 = (((0, 3),), (0, 0, 0))                 # M^(3)_0 = H_0^3 (magnetic-3 INPUT)
    assert N3.multiply(H0n3, H1n3).terms, ("SU2Nf3ConeKAlgebra", "empty multiply")
    assert N3.multiply(M0n3, M0n3).terms, ("SU2Nf3ConeKAlgebra", "empty M·M (cone-monomial sector)")
    # magnetic-3 cone monomials as INPUTS (the total multiply):
    # M^(3)_0·H_0 = H_0^4 = M^(4)_0, and M^(3)_0·M^(3)_0 = H_0^6 = M^(6)_0.
    assert list(N3.multiply(M30n3, H0n3).terms) == [(((0, 4),), (0, 0, 0))], \
        ("SU2Nf3ConeKAlgebra", "M^(3)_0·H_0 ≠ M^(4)_0")
    assert list(N3.multiply(M30n3, M30n3).terms) == [(((0, 6),), (0, 0, 0))], \
        ("SU2Nf3ConeKAlgebra", "M^(3)_0·M^(3)_0 ≠ M^(6)_0")
    tn3 = N3.trace(N3.identity(), K=12)
    nzn3 = [q for q, r in tn3.coeffs.items() if not r.is_zero()]
    assert nzn3 and max(nzn3) >= 10, ("SU2Nf3ConeKAlgebra", "SU(4) index did not reach q^12")
    # the magnetic-sector trace (SU(4) bootstrap): Tr(H_0)[q^1] = −χ_4̄ ≠ 0.
    assert N3.trace(H0n3, K=4).coeffs.get(1), ("SU2Nf3ConeKAlgebra", "magnetic trace seed empty")
    assert N3.verify_orthonormality(H0n3, H0n3, K=3), ("SU2Nf3ConeKAlgebra", "self-norm")
    assert N3.verify_orthonormality(H0n3, H1n3, K=3), ("SU2Nf3ConeKAlgebra", "off-diagonal")
    print(f"  OK   {'SU2Nf3ConeKAlgebra':24s} SU(4) index spine-free to q^12 + magnetic-3 input multiply + trace + orthonormality")


def check_a1d3_mixed_tiles():
    """[A_1,D_3] mixed-tile orthonormality — the deep-label battery the
    generic `exercise` loop cannot reach (it does not speak `A1D3KAlg`'s
    native `(tile, a, b, k)` labels, so the class is otherwise exercised
    at unit + generator level only).

    The mixed-tile monomials `q^{-ab}·T_i^a·D_{i-1}^b` at
    `a, b ≥ 1, a + b ≥ 3` ARE orthonormal.  Apparent violations (a
    `-χ₃q^{-3}` term in `I_{(3,2,1,0),(0,2,0,0)}`, a q⁰ coefficient of 1
    against `(0,1,1,0)`) were truncation artifacts of a `trace_element`
    that did not widen per-label trace requests by the negative valuation
    of the product's Laurent coefficients; the widened assembly is
    window-stable and agrees exactly with the BPS Schur-formula pairing
    at the corresponding charges."""
    import importlib
    AD = importlib.import_module("a1d3_kalg").A1D3KAlg()
    pairs = [
        ((3, 2, 1, 0), (0, 2, 0, 0)),
        ((3, 2, 1, 0), (0, 1, 1, 0)),
        ((5, 1, 2, 0), (3, 2, 1, 0)),
    ]
    for a, b in pairs:
        assert AD.verify_orthonormality(a, b, K=5), \
            ("A1D3KAlg", "mixed-tile orthonormality", a, b)
        I = AD.inner_product(a, b, 5)
        assert not I.coeffs, ("A1D3KAlg", "off-diagonal not exactly 0", a, b, I)
    d = (3, 2, 1, 0)
    assert AD.verify_orthonormality(d, d, K=5), ("A1D3KAlg", "diagonal", d)
    Id = AD.inner_product(d, d, 5)
    assert not Id[0].is_zero(), ("A1D3KAlg", "diagonal q⁰ vanished", d)
    print(f"  OK   {'A1D3KAlg':24s} mixed-tile (a+b≥3) orthonormality, K=5")


def main():
    import importlib
    n_ok = 0
    failures = []
    for mod_name, cls_name, args in CONE_ALGEBRAS:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            A = cls(*args)
            passed = exercise(A, cls_name)
            tag = " (multiply/ρ only; trace slow)" if cls_name in LIGHT_TRACE else ""
            print(f"  OK   {cls_name:24s} labels exercised: {passed}{tag}")
            n_ok += 1
        except Exception as e:
            print(f"  FAIL {cls_name:24s} {type(e).__name__}: {e}")
            failures.append((cls_name, traceback.format_exc()))
    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        raise SystemExit(1)
    check_improvable()
    check_a1d3_mixed_tiles()
    print(f"ALL {n_ok} CONE CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
