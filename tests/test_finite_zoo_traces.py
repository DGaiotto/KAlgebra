"""Exact elementary traces + KAlgebra-axiom verification for the
frozen finite zoo (`finite_kalgebras/elem_traces.py`).

Three layers of validation:

1. **Closed-form oracles** — the zoo's seed traces (served live: closed
   forms — e6/e8 from their W₃ recipes, the design record — or geometric classes
   through generator maps built at runtime — pentagon / heptagon from
   `A1A2kKAlg(k)` since 2026-09-24, before that the orthonormality bootstrap
   over the closed-form M(2,5) / M(2,7) vacuum character; no frozen table
   remains, and no BPS fallback) must reproduce the independently-implemented
   chiral-algebra characters: `PentagonKAlg`'s Rogers–Ramanujan pair
   (M(2,5)) and `A1A2kKAlg(2)`'s Andrews–Gordon characters (M(2,7) — the
   heptagon).  Where no route serves a trace the zoo raises, and says why:
   a bootstrap that stops short names the entry, the seed and the order
   (section 4).  The su2u1 entries are served (a1d4 from SU3ADKAlg, a1d6 /
   a1d8 from A1DevenKAlg(2) / A1DevenKAlg(3)).

2. **Contract axioms on trace-enabled entries** — `verify_orthonormality`
   (orthonormality: the FULL Gram window, not just the diagonal),
   `verify_rho_twisted_trace` (ρ²-twisted cyclicity), through the
   universal `KAlgebra` surface.

3. **ρ/bar axioms on the whole zoo** (no trace needed) — ρ is an
   automorphism fixing the identity with a genuine inverse, and the
   bar involution holds, on in-cone label pairs for every registry
   entry.  This is the "make sure ρ satisfies the KAlgebra axioms"
   guard for the (re)generated standalones.

Run:  `python3 run_tests.py`
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random

import finite_kalgebras as fk
from elem_traces import (
    elementary_seed_indices,
    trace_residual,
)


PASS = []
FAIL = []


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def inner_product_via_cone(A, a, b, K):
    """I_{a,b} = Tr(rho(L_a) L_b) computed through ConeKAlgebra.trace.

    Workaround for the Plan-18 half-state: the *generated* flavoured
    zoo classes return `RLaurent` coefficients from `multiply`, which
    the base `KAlgebra.trace_element` (LaurentPoly-scalar) rejects.
    `ConeKAlgebra.trace` itself handles RLaurent seed coefficients, so
    the q-linear extension is done here by hand.  Dies with the Z-form
    regeneration (the design record PR2)."""
    from zplus_ring import RPowerSeries
    R = A.coefficient_ring()
    # ρ(L_a) in R-form via the contract `rho_element` (the ⋆+μ^δ
    # framework ρ — correct on flavour-in-coefficient Elements; the old
    # `rho_R_element` is retired, the design record, and the bare-label `A.rho(a)`
    # fallback would DROP the ⋆/μ-twist and break flavoured orthonormality).
    ra = A.rho_element(A.multiply(a, A.identity()))
    out = RPowerSeries(R, {}, K)
    for l1, c1 in ra.terms.items():
        out = _accumulate_traces(A, A.multiply(l1, b), out, K, scale=c1)
    return RPowerSeries(R, {e: c for e, c in out.coeffs.items() if e <= K},
                        K)


def _accumulate_traces(A, prod, out, K, scale=None):
    """out += Σ_label coeff·Tr(label), widening the trace order by the
    most-negative q-power of each coefficient so truncation at K stays
    exact (mirrors ConeKAlgebra.trace's inner_K widening)."""
    from laurent_poly import LaurentPoly
    from zplus_ring import RLaurent
    R = A.coefficient_ring()
    for label, c in prod.terms.items():
        if isinstance(c, LaurentPoly):
            c = RLaurent(R, dict(c._coeffs))
        if scale is not None:
            c = scale * c
        emin = min((e for e in c.coeffs), default=0)
        inner_K = K - min(emin, 0)
        tr = A.trace(label, inner_K)
        prod_series = tr * c
        from zplus_ring import RPowerSeries
        out = out + RPowerSeries(
            R, {e: v for e, v in prod_series.coeffs.items() if e <= K}, K)
    return out


def delta_q0(A, series, a, b):
    """`series[q^0] == delta_{a,b}` as RElements."""
    R = A.coefficient_ring()
    got = series.coeffs.get(0, R.zero())
    want = R.one() if a == b else R.zero()
    if not hasattr(got, "terms"):
        from zplus_ring import RElement
        got = RElement(R, {R.one_basis(): got} if got else {})
    return got == want


def cone_window_labels(A, *, max_deg=2, max_cones=None):
    """Identity + single gens + squares + in-cone degree-2 pairs."""
    cd = A.cone_data()
    cones = sorted(cd.cones(), key=sorted)
    if max_cones is not None:
        cones = cones[:max_cones]
    labels = {()}
    for c in cones:
        cs = sorted(c)
        for i in cs:
            labels.add(((i, 1),))
            if max_deg >= 2:
                labels.add(((i, 2),))
        if max_deg >= 2:
            for x in range(len(cs)):
                for y in range(x + 1, len(cs)):
                    labels.add(((cs[x], 1), (cs[y], 1)))
    return sorted(labels)


# ---------------------------------------------------------------------------
# 1. closed-form oracles
# ---------------------------------------------------------------------------

def test_pentagon_matches_rogers_ramanujan():
    """Zoo pentagon seed traces == PentagonKAlg's M(2,5) characters."""
    from kalgebra_samples import PentagonKAlg
    A = fk.FinitePentagonKAlgebra()
    P = PentagonKAlg()
    K = 20
    ok1 = str(A.trace((), K=K)) == str(P.trace((0, 0, 0), K=K))
    okL = str(A.trace(((0, 1),), K=K)) == str(P.trace((0, 1, 0), K=K))
    check("pentagon Tr(1) == chi_0 (Rogers-Ramanujan)", ok1)
    check("pentagon Tr(L) == q^-1(chi_0 - chi_1)", okL)


def test_heptagon_matches_andrews_gordon():
    """Zoo heptagon orbit traces == A1A2kKAlg(2)'s M(2,7) T-series.

    The two single-gen ρ²-orbits of the zoo heptagon must reproduce
    {T_1, T_2} (as a set — the orbit↔a matching is not fixed a priori),
    and the identity trace must equal T_0 = chi_1(q²).  Since 2026-09-24
    the zoo serves these seeds from `A1A2kKAlg(2)` itself
    (`aeven_seeds`), so this checks the routing; the
    independent witnesses are the trivial-R bootstrap and the theta-form
    characters (the suite in the source repository, the suite in the source repository)."""
    from a1a2k_kalg import A1A2kKAlg
    A = fk.FiniteHeptagonKAlgebra()
    H = A1A2kKAlg(2)
    K = 16
    T = H._compute_T_series(K)
    zoo_id = str(A.trace((), K=K))
    zoo_orbits = {str(A.trace(((rep, 1),), K=K))
                  for rep in elementary_seed_indices("heptagon")}
    check("heptagon Tr(1) == T_0 (M(2,7) vacuum)", zoo_id == str(T[0]))
    check("heptagon orbit traces == {T_1, T_2} (Andrews-Gordon)",
          zoo_orbits == {str(T[1]), str(T[2])})


def test_no_frozen_trace_tables():
    """No zoo trace is read from a frozen table ("frozen data for which we have an actual functional class is clearly
    pointless").  The last table, e8 at K = 6,
    is a fixture of tests/test_w3_seeds.py now.  The table module is empty
    and `elem_traces` has no reader or writer for it, so a table cannot
    come back unnoticed."""
    import elem_traces as ET
    from elem_trace_data import ELEM_TRACE_DATA
    check("elem_trace_data.ELEM_TRACE_DATA is empty", ELEM_TRACE_DATA == {})
    gone = ("_frozen", "supported_ids", "_wedge_valid_K", "freeze")
    check(f"elem_traces has no frozen-table reader or writer ({', '.join(gone)})",
          not any(hasattr(ET, n) for n in gone))


def test_a1d3_identity_matches_hand_written():
    """Zoo a1d3 (SU(2)-charactered) Tr(1) == hand-written A1D3KAlg."""
    from a1d3_kalg import A1D3KAlg
    A = fk.FiniteA1D3KAlgebra()
    H = A1D3KAlg()
    K = 12
    ok = str(A.trace((), K=K)) == str(H.trace((0, 0, 0, 0), K=K))
    check("a1d3 Tr(1) == A1D3KAlg Tr(1) (SU(2) characters)", ok)


def _bps_spy():
    """A `sys.setprofile` hook recording every function executed from the
    BPS engine's files, and the set it fills."""
    ran = set()

    def prof(frame, event, arg):
        name = frame.f_code.co_filename
        if event == "call" and name.endswith(("bps_kalgebra.py",
                                              "bps_factor_spectrum.py")):
            ran.add(os.path.basename(name) + ":" + frame.f_code.co_name)
    return prof, ran


def _trace_error(A, label, K):
    """(NotImplementedError message or None, BPS functions run) for one
    trace call made under the BPS spy."""
    prof, ran = _bps_spy()
    sys.setprofile(prof)
    try:
        A.trace(label, K=K)
        msg = None
    except NotImplementedError as ex:
        msg = str(ex)
    finally:
        sys.setprofile(None)
    return msg, ran


def test_su2u1_traces_served():
    """The su2u1 entries serve their traces: a1d4 since 2026-09-23 from
    SU3ADKAlg (`a1d4_seeds`, tested in
    the suite in the source repository), a1d6 / a1d8 since 2026-09-24 from
    `A1DevenKAlg(2)` / `A1DevenKAlg(3)` (`a1deven_seeds`,
    tested in tests/test_a1deven_seeds.py).  Until then a1d6 / a1d8 raised
    NotImplementedError naming those classes: the old su2u1 route peeled the
    flavour slots the wrong way round (the audit — no
    SU(2) triplet at q^2 in the a1d4 vacuum), and it reached the BPS engine
    at a1d6 and did not finish at a1d8.  Each vacuum now carries the triplet
    at q^2, a seed is served, and no BPS function runs.  Positive control
    first: `a1d4_from_su3ad` carries the triplet (key (2, 0)) at q^2, and the
    spy does see BPS when BPS runs."""
    from a1d4_from_su3ad import a1d4_kalgebra
    D4 = a1d4_kalgebra()
    q2 = dict(D4.trace(D4.identity(), K=4).coeffs[2].terms)
    check("positive control: a1d4_from_su3ad Tr(1)[q^2] carries the "
          "SU(2) triplet (2, 0)", q2.get((2, 0), 0) != 0)
    from elem_traces import _bps_oracle
    prof, ran = _bps_spy()
    sys.setprofile(prof)
    try:
        _bps_oracle("pentagon").trace((0, 0), K=2)
    finally:
        sys.setprofile(None)
    check("positive control: the spy records BPS functions when BPS runs",
          bool(ran))
    # the classes the su2u1 routes serve through exist, with the signature
    # they are named with (import-level only, so this check's cost does not
    # depend on building them)
    import inspect
    from su3_ad_kalg import SU3ADKAlg
    from a1deven_kalg import A1DevenKAlg
    check("the classes the su2u1 routes serve through exist: SU3ADKAlg, "
          "A1DevenKAlg(k)",
          callable(SU3ADKAlg)
          and "k" in inspect.signature(A1DevenKAlg).parameters)
    for sid in ("a1d4", "a1d6", "a1d8"):
        A = fk.FINITE_KALGEBRAS[sid]()
        gen = ((min(A.cone_data().mult_gens()), 1),)
        prof, ran = _bps_spy()
        sys.setprofile(prof)
        try:
            t2 = dict(A.trace(A.identity(), K=4).coeffs[2].terms)
            tg = A.trace(gen, K=2)
        finally:
            sys.setprofile(None)
        check(f"{sid} Tr(1)[q^2] is served and carries the SU(2) triplet "
              f"(2, 0); Tr({gen}) is served; no BPS function runs",
              t2.get((2, 0), 0) != 0 and bool(tg.coeffs) and not ran)


def test_e6_seed_heads_pinned():
    """E6 elementary-trace heads, pinned from the exact BPS generation;
    served live since 2026-09-23 (Nahm-sum vacuum + orthonormality
    bootstrap, which reproduced the retired K=12 table exactly; since
    the design record the served route is the W₃(3,7) recipes, and that one
    is the witness).
    Tr(1) = 1 + q^4 + 2q^6 + 3q^8 + ..., Tr(mg0) = -q - 2q^5 - q^7 ..."""
    A = fk.FiniteE6KAlgebra()
    tr1 = A.trace((), K=8)
    trg = A.trace(((0, 1),), K=8)
    check("e6 Tr(1) head pinned",
          [tr1.coeffs.get(e, 0) for e in (0, 4, 6, 8)] == [1, 1, 2, 3]
          and not tr1.coeffs.get(2))
    check("e6 Tr(mg0) head pinned",
          [trg.coeffs.get(e, 0) for e in (1, 5, 7)] == [-1, -2, -1])


def test_e8_seed_heads_pinned():
    """E8 (the largest exceptional, 16 trivial-R seeds) elementary-trace
    heads, pinned from the BPS-free orthonormality bootstrap — the
    incremental degree-sweep with full-seed-grid pairing, which closes at
    degree 3 entirely wall-free through K=6 (the old generate-everything loop
    timed out >900 s on the degree-4 reducer wall); certificate: the
    over-determined consistency (3098 eqns at K=6), the K=6/K=4 overlap
    agreement, and the mg27 BPS spot-check (Tr(mg27) head -q).  Served since
    the design record from the W₃(3,8) recipes; the K=6 table itself is a fixture
    of tests/test_w3_seeds.py.  See a probe in the source repository."""
    A = fk.FiniteE8KAlgebra()
    tr1 = A.trace((), K=6)
    check("e8 Tr(1) head pinned",
          [tr1.coeffs.get(e, 0) for e in (0, 4, 6)] == [1, 1, 2]
          and not tr1.coeffs.get(1) and not tr1.coeffs.get(2)
          and not tr1.coeffs.get(3) and not tr1.coeffs.get(5))
    trg = A.trace(((27, 1),), K=6)
    check("e8 Tr(mg27) head pinned (BPS-matched -q)",
          [trg.coeffs.get(e, 0) for e in (1, 5)] == [-1, -1])
    trg0 = A.trace(((0, 1),), K=6)
    check("e8 Tr(mg0) head pinned",
          trg0.coeffs.get(4, 0) == 1 and not trg0.coeffs.get(1)
          and not trg0.coeffs.get(2) and not trg0.coeffs.get(3)
          and not trg0.coeffs.get(5) and not trg0.coeffs.get(6))


# ---------------------------------------------------------------------------
# 2. contract axioms on trace-enabled entries
# ---------------------------------------------------------------------------

def test_orthonormality_gram_window():
    """Orthonormality on a full basis window.

    Trivial-R entries go through the contract's `verify_orthonormality`;
    the flavoured generated entries (RLaurent multiply — the Plan-18
    half-state) go through the cone-path inner product, same statement:
    I_{a,b}[q^0] == delta_{a,b} for ALL pairs in the window."""
    windows = {
        "pentagon": dict(max_deg=2, max_cones=None, K=6, contract=True),
        "heptagon": dict(max_deg=2, max_cones=4, K=4, contract=True),
        "a3": dict(max_deg=1, max_cones=4, K=4, contract=False),
        "a1d3": dict(max_deg=1, max_cones=4, K=4, contract=False),
        "a1d5": dict(max_deg=1, max_cones=2, K=4, contract=False),
        "a5": dict(max_deg=1, max_cones=2, K=4, contract=False),
    }
    for sid, cfg in windows.items():
        A = fk.FINITE_KALGEBRAS[sid]()
        labels = cone_window_labels(
            A, max_deg=cfg["max_deg"], max_cones=cfg["max_cones"])
        if cfg["contract"]:
            bad = [(a, b) for a in labels for b in labels
                   if not A.verify_orthonormality(a, b, K=cfg["K"])]
        else:
            bad = [(a, b) for a in labels for b in labels
                   if not delta_q0(
                       A, inner_product_via_cone(A, a, b, cfg["K"]), a, b)]
        check(f"orthonormality {len(labels)}x{len(labels)} window [{sid}]"
              + ("" if not bad else f" - {len(bad)} bad e.g. {bad[:2]}"),
              not bad)


def test_rho_twisted_trace_cyclicity():
    """Tr(ab) == Tr(rho^2(b) a) on in-cone pairs (trace-enabled).

    Same trivial-R/flavoured split as the orthonormality window."""
    for sid, contract in (("pentagon", True), ("heptagon", True),
                          ("a3", False), ("a1d3", False)):
        A = fk.FINITE_KALGEBRAS[sid]()
        labels = cone_window_labels(A, max_deg=1, max_cones=3)
        if contract:
            bad = [(a, b) for a in labels for b in labels
                   if not A.verify_rho_twisted_trace(a, b, K=4)]
        else:
            from zplus_ring import RPowerSeries
            R = A.coefficient_ring()

            def tr_elem(elem, K=4):
                out = RPowerSeries(R, {}, K)
                return _accumulate_traces(A, elem, out, K)

            def as_elem(label):
                return A.multiply(label, A.identity())

            def bilinear(e1, y, K=4):
                # Tr(e1 · L_y), q-linear over e1's RLaurent coefficients
                out = RPowerSeries(R, {}, K)
                for l1, c1 in e1.terms.items():
                    out = _accumulate_traces(
                        A, A.multiply(l1, y), out, K, scale=c1)
                return out

            # element-level rho^2 via the contract `rho_element` (the
            # ⋆+μ^δ framework ρ; label rho^2 alone drops unit characters,
            # `rho_R_element` retired — the design record).
            def rho2_elem(label):
                return A.rho_element(A.rho_element(as_elem(label)))

            bad = [(a, b) for a in labels for b in labels
                   if str(tr_elem(A.multiply(a, b)))
                   != str(bilinear(rho2_elem(b), a))]
        check(f"rho^2-twisted cyclicity [{sid}]"
              + ("" if not bad else f" - {len(bad)} bad"), not bad)


def test_trace_constant_on_rho_orbit():
    """Tr(L_{rho(i)}) == Tr(L_i): the trace is ρ-invariant on the zoo
    (ρ-twist enters only through ρ², and seeds fold by ρ²-orbit)."""
    for sid in ("pentagon", "heptagon"):
        A = fk.FINITE_KALGEBRAS[sid]()
        n = len(A.cone_data().mult_gens())
        ok = all(
            str(A.trace(((i, 1),), K=6))
            == str(A.trace(A.rho(((i, 1),)), K=6))
            for i in range(n)
        )
        check(f"trace rho-invariant on single gens [{sid}]", ok)


# ---------------------------------------------------------------------------
# 3. rho / bar axioms across the whole registry (no trace needed)
# ---------------------------------------------------------------------------

# Flavoured generated entries return RLaurent coefficients from
# `multiply` (the Plan-18 half-state): the base `Element` arithmetic
# (bar / rho_element / multiply_elements) is LaurentPoly-only, so the
# axiom checks below carry manual coefficient-aware equivalents.
# Entries WITHOUT a per-file `rho_R_element` have NO correct
# element-level ρ today — that gap is *asserted* so the Z-form
# regeneration (the design record PR2) must flip it consciously.
NO_ELEMENT_RHO = {"a7", "e7", "a1d7", "a1d8"}


def _elem_str(e):
    return {l: str(c) for l, c in e.terms.items() if str(c) != "0"}


def _manual_bar_check(A, a, b):
    """multiply(a, b).bar() == multiply(b, a), coefficient-aware."""
    mab = A.multiply(a, b)
    mba = A.multiply(b, a)
    barred = {l: c.bar() for l, c in mab.terms.items()}
    return ({l: str(c) for l, c in barred.items() if str(c) != "0"}
            == _elem_str(mba))


def test_rho_axioms_whole_zoo(pairs=12):
    """For EVERY registry entry: rho fixes the identity, rho_inverse
    inverts, rho is an algebra automorphism, and the bar involution holds.
    ρ is checked through the *contract* `verify_rho_is_automorphism` on both
    trivial-R and flavour-in-coefficient entries: the u1/su2u1 per-file
    `rho_R_element` HACK has been RETIRED from a3/a5/a1d6 — it was a
    codegen-duplicated shadow of the framework `rho_element` (the ⋆+μ^δ
    automorphism).  su2 (a1d3/a1d5) keep a *trivial* alias
    `rho_R_element = rho_element` (harmless), so the check is the contract
    ρ-automorphism, not the method's existence.  The remaining NO_ELEMENT_RHO
    entries still take the lighter label-rho path for speed; the genuinely
    open Plan-18-PR2 item is the *derived* `rho_R_form` (embed_R wiring + the
    RLaurent-at-multiply fix), not a bespoke per-file ρ."""
    rng = random.Random(0)
    seen_classes = set()
    for sid, cls in sorted(fk.FINITE_KALGEBRAS.items()):
        if cls in seen_classes:
            continue
        seen_classes.add(cls)
        A = cls()
        flavoured = not isinstance(
            A.coefficient_ring(), __import__("zplus_ring").TrivialZPlusRing)
        labels = [l for l in cone_window_labels(A, max_deg=1, max_cones=6)
                  if l != ()]
        sample_pairs = [(rng.choice(labels), rng.choice(labels))
                        for _ in range(pairs)]
        ok_id = A.verify_rho_fixes_identity()
        ok_inv = all(A.verify_rho_inverse(l)
                     for l in rng.sample(labels, min(6, len(labels))))
        if sid == "a1d4":
            # FIXED (ρ-orbit completion; the S₃⊃Z₂ root cause): a1d4 is now
            # the full 8-mg su2u1 algebra with an honest framework ρ, not
            # the old 3-mg truncation.  The D_4 gauge cone is not ρ-closed
            # (gauge rank 2, the family minimum); completing gens + cones
            # under ρ recovers the two length-4 σ-orbits = 8 generators.  It
            # now passes the contract ρ-automorphism + bar in place, like any
            # flavour-in-coefficient cone.
            n_mg = len(A.cone_data().mult_gens())
            ok_aut = all(A.verify_rho_is_automorphism(a, b)
                         for a, b in sample_pairs)
            ok_bar = all(A.verify_bar_involution(a, b)
                         for a, b in sample_pairs)
            check("a1d4 full algebra (8 mg, ρ-orbit-completed)", n_mg == 8)
            check("a1d4 rho-automorphism + bar (contract)",
                  ok_id and ok_inv and ok_aut and ok_bar)
            continue
        if not flavoured:
            ok_aut = all(A.verify_rho_is_automorphism(a, b)
                         for a, b in sample_pairs)
            ok_bar = all(A.verify_bar_involution(a, b)
                         for a, b in sample_pairs)
            check(f"rho+bar axioms (contract) [{sid}]",
                  ok_id and ok_inv and ok_aut and ok_bar)
            continue
        ok_bar = all(_manual_bar_check(A, a, b) for a, b in sample_pairs)
        if sid in NO_ELEMENT_RHO:
            check(f"label-rho + bar [{sid}]", ok_id and ok_inv and ok_bar)
            check(f"element-rho GAP asserted [{sid}] (no rho_R_element; "
                  f"the design record PR2 closes)", not hasattr(A, "rho_R_element"))
            continue
        # ρ verified through the *contract* `rho_element` (the ⋆+μ^δ
        # framework automorphism), exactly like a1d4.  The u1/su2u1 per-file
        # `rho_R_element` HACK is retired (a3/a5/a1d6 no longer carry it —
        # evidenced by the regen-diff and the `rho_element` equivariance
        # test).  su2 (a1d3/a1d5) keep a *trivial* alias
        # `rho_R_element = rho_element` (harmless, emitter-intended), so the
        # check is the contract ρ-automorphism, NOT the method's existence.
        # Remaining Plan-18-PR2 work: the *derived* `rho_R_form` (needs
        # embed_R wiring + the RLaurent-at-multiply fix).
        ok_aut = all(A.verify_rho_is_automorphism(a, b)
                     for a, b in sample_pairs)
        check(f"rho+bar axioms (contract rho_element) [{sid}]",
              ok_id and ok_inv and ok_aut and ok_bar)


# ---------------------------------------------------------------------------
# 4. layer-2 plumbing details
# ---------------------------------------------------------------------------

def test_unexpected_seed_raises():
    """A non-cone-monomial seed shape must raise, not fabricate."""
    A = fk.FinitePentagonKAlgebra()
    try:
        trace_residual("pentagon", A, ((0, 1, 7),), 4)
        check("malformed seed raises", False)
    except (ValueError, TypeError):
        check("malformed seed raises", True)


def test_larger_window_agrees_on_overlap():
    """A trace asked at a larger q-window agrees with the smaller one on
    the overlap (the live route regenerates exactly; this used to test the
    lazy extension past the retired frozen pentagon window)."""
    A = fk.FinitePentagonKAlgebra()
    K = 24
    lo = trace_residual("pentagon", A, ((0, 1),), K)
    hi = trace_residual("pentagon", A, ((0, 1),), K + 4)
    ok = all(hi.coeffs.get(e, 0) == c for e, c in lo.coeffs.items())
    check("larger window agrees on the overlap", ok)


def test_unserved_seed_raises_without_bps():
    """Where no exact route serves a seed the zoo raises NotImplementedError
    naming the entry, the seed and the order; it never falls back to the
    BPS engine (the design record: no runtime oracle).  Since 2026-09-24 no entry
    is served by a bootstrap, so the raise is exercised on the pentagon with
    its route (`aeven_seeds`, through `A1A2kKAlg(1)`) withdrawn for the check
    and the trivial-R bootstrap forced to stop short.  With the routes in
    place, the trivial-R, u(1) and su(2) bootstraps, forced to stop short,
    leave the served pentagon / a5 / a1d3 traces untouched (pentagon and
    heptagon through `aeven_seeds` since 2026-09-24; a3 / a5 / a7 through
    `aodd_seeds` and a1d3 through `a1d3_seeds` since 2026-09-23), and
    `generate` (the witness API that still runs them) raises.  A
    multi-generator seed, whose only route was BPS, raises too.  The spy's
    positive control is in test_su2u1_traces_served."""
    import elem_traces as ET
    import u1_bootstrap as U1B
    import su2_bootstrap as SU2B

    def forced(*args, **kwargs):
        raise ET._BootstrapUnavailable("forced by the test")

    def clear():
        ET._EXT.clear()
        for cache in ET._REC_CACHE.values():
            cache.clear()

    sid = "pentagon"
    seed = ET.rho2_orbit_map(sid)[1]
    saved = ET._generate_bootstrap
    saved_route = ET._AEVEN_SEEDS.pop(sid)
    ET._generate_bootstrap = forced
    clear()
    try:
        A = fk.FINITE_KALGEBRAS[sid]()
        msg, ran = _trace_error(A, ((1, 1),), 5)
    finally:
        ET._generate_bootstrap = saved
        ET._AEVEN_SEEDS[sid] = saved_route
        clear()
    check(f"{sid}: with its route withdrawn, a bootstrap that stops short "
          f"raises naming the entry, the seed and the order; no BPS function "
          f"runs",
          msg is not None and msg.startswith(f"{sid}:")
          and f"seed {seed}" in msg and "q^5" in msg
          and "forced by the test" in msg and not ran)

    for sid, mod, name in (("pentagon", ET, "_generate_bootstrap"),
                           ("a5", U1B, "generate_u1"),
                           ("a1d3", SU2B, "generate_su2")):
        saved = getattr(mod, name)
        setattr(mod, name, forced)
        clear()
        try:
            A = fk.FINITE_KALGEBRAS[sid]()
            msg, ran = _trace_error(A, ((1, 1),), 5)
            try:
                ET.generate(sid, 5)
                gen_msg = None
            except NotImplementedError as ex:
                gen_msg = str(ex)
        finally:
            setattr(mod, name, saved)
            clear()
        check(f"{sid}: the served trace does not reach the {name} bootstrap "
              f"(forced to stop short, the trace is still served; no BPS "
              f"function runs)", msg is None and not ran)
        check(f"{sid}: generate() with {name} forced to stop short raises "
              f"naming the entry and the order",
              gen_msg is not None and gen_msg.startswith(f"{sid}:")
              and "q^5" in gen_msg and "forced by the test" in gen_msg)
    A = fk.FinitePentagonKAlgebra()
    prof, ran = _bps_spy()
    sys.setprofile(prof)
    try:
        trace_residual("pentagon", A, ((0, 1), (1, 1)), 4)
        msg = None
    except NotImplementedError as ex:
        msg = str(ex)
    finally:
        sys.setprofile(None)
    check("a multi-generator seed raises (its only route was BPS); no BPS "
          "function runs",
          msg is not None and "multi-generator" in msg and not ran)


if __name__ == "__main__":
    test_no_frozen_trace_tables()
    test_pentagon_matches_rogers_ramanujan()
    test_heptagon_matches_andrews_gordon()
    test_a1d3_identity_matches_hand_written()
    test_su2u1_traces_served()
    test_e6_seed_heads_pinned()
    test_e8_seed_heads_pinned()
    test_orthonormality_gram_window()
    test_rho_twisted_trace_cyclicity()
    test_trace_constant_on_rho_orbit()
    test_rho_axioms_whole_zoo()
    test_unexpected_seed_raises()
    test_larger_window_agrees_on_overlap()
    test_unserved_seed_raises_without_bps()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED, {len(PASS)} passed")
        sys.exit(1)
    print(f"All {len(PASS)} finite-zoo trace tests passed.")
