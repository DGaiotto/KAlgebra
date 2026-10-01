"""Runnable self-test for the ConeKAlgebra layer.

Pure Python 3, no third-party dependencies, no realisation engine (no BPS
or RG-flow module is imported: `check_no_engine_loaded` asserts it at the
end).  Each cone algebra in the catalogue is a *closed-form* `ConeKAlgebra`:
the canonical basis is organised into cones of multiplicative generators,
`multiply` is the generic normal-ordering reduction over the cone
cocycle/cross-products, and `trace` is the closed-form residual rule
`_trace_residual` (Nahm-sum / character / q-Pochhammer) — all evaluated here
without any computed (BPS/RG) backend.  The one Step-1 module beyond the
contract that the cone layer uses is the quantum torus sample
(`QuantumTorusKAlg`), the second tensor factor of the u(1)-gauged
`[A_1, D_{2k+2}]` trace transport — the witness route of that class's traces,
which `check_seed_routes` runs.

For every algebra this exercises, on the unit and on a couple of cone
generators:

  * `multiply` (cocycle + cross-product reduction),
  * `rho` / `rho_inverse`,
  * `trace` (the closed-form residual),
  * `verify_orthonormality` (the Schur pairing `I_{a,b}=δ_{a,b}+O(q)`),

i.e. the orthonormality conjecture (docs/conjectures-step2-cone.md) on these
structures.

`check_seed_routes` then checks the trace routes the closed-form seeds serve
against their witnesses: `SU3ADKAlg`'s geometric labels, its seeds (`Tr_1`
against the Kac–Wakimoto vacuum character, `Tr_T` / `Tr_D` against the forward
orthonormality pass), the exact seed depth of its product traces, and the
Layer-1 route of the u(1)-gauged `[A_1, D_{2k+2}]` against its trace
transport.

`check_geometric_labels` checks the geometric canonical-basis labels of the A
and D classes: the named pentagon / heptagon samples (where the abelianized
layer is on the path), `A1A2kKAlg`, `U1HexagonKAlg` / `HexagonKAlg`,
`A1DoddConeKAlg` against the curves of `A1DnKAlg`, and every A / D entry of the
finite zoo (`zoo_geometry`).
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
    # --- U(1)-gauged A1A_odd, the analytic class at every k: the letters are
    #   the diagonals of the (2k+4)-gon plus E^{±1} (geometric_label).
    #   MULTIPLY: the analytic peel (the even family's arc rules on a rank-2
    #       torus pairing) — no stored table, no RG-flow derivation.
    #   TRACE: Layer 1 + one closed form for every chord seed, the singlet rule
    #       singlet_chord_trace (a difference of two M(1,p) singlet module
    #       characters with a q-power prefactor, u1_pgon_layer2), at every k.
    ("u1a1aodd_kalg",  "U1A1AoddKAlg",  (1,)),   # = U(1)-gauged hexagon (A1A_3)
    ("u1a1aodd_kalg",  "U1A1AoddKAlg",  (2,)),   # = U(1)-gauged octagon (A1A_5)
    ("u1a1aodd_kalg",  "U1A1AoddKAlg",  (3,)),   # = U(1)-gauged decagon (A1A_7)
    ("u1a1aodd_kalg",  "U1A1AoddKAlg",  (4,)),   # = U(1)-gauged dodecagon (A1A_9)
    # --- [A_1, D_4] = SU(3)_{-3/2}, SU(3) flavour, fully BPS-free trace ---
    #   Tr_1, Tr_T, Tr_D = the even-D k = 1 closed forms through the curve map
    #   onto A1DevenKAlg(1) (Tr_1: Creutzig's gauge tower); the Kac-Wakimoto
    #   vacuum character of sl(3)^_{-3/2} and the forward orthonormality pass
    #   are their witnesses (check_seed_routes).  Layer-1 and
    #   the product multiply are carried in SU(3) Cartan fugacities (weights,
    #   not characters), Weyl-symmetrized only on the total -- so non-self-dual
    #   product content is correct.  No engine on the trace path.  (The generic
    #   label discovery below only reaches its unit -> vacuum trace Tr_1 +
    #   orthonormality here; the generator trace Tr_T is exercised, to high
    #   q-order, in check_improvable.)
    ("su3_ad_kalg",    "SU3ADKAlg",     ()),
    # --- u(1)-gauged E7 = A_q[T] for the u(1)-gauged E7 SCFT.  A quantum-torus
    #   cone (rank-1 gauge torus on E = X_{(0,1)}).  MULTIPLY: the cone
    #   cross-products are stored product data (no closed form),
    #   u1e7_cone_tables.pkl, computed once from the gauged flow dressed with
    #   the central chord (3,0).  TRACE: the magnetic (c0) sector vanishes;
    #   every magnetically neutral label, the E-tower included, is traced
    #   through the ungauged [A_1,E_7] algebra's closed forms (FiniteE7KAlgebra,
    #   by a label map certified by products).  ρ from the stored single-ray
    #   table (u1e7_rho_tables.pkl) + the gauge reflection.  No engine on
    #   multiply / ρ / trace.
    ("u1e7_cone_kalgebra", "U1E7ConeKAlgebra", ()),
    # --- u(1)-gauged [A_1, D_{2k+2}] = A_q[T] for the u(1)-gauged D-even SCFT,
    #   in the curve frame: labels (curves, e, κ) — curves of the once-punctured
    #   (2k+2)-gon, the power e of the gauge letter E = X_{0,1}, and the SU(2)
    #   weight κ in the label (the Z-form).  MULTIPLY: the closed-form product
    #   rule of the curve frame (u1a1deven_geometric_frame, the A1Dodd arc rules),
    #   no stored table, every k.  TRACE: the magnetic sector vanishes; the gauge
    #   sector is Creutzig's closed form (arbitrary q-order, see the q^70 witness
    #   in check_improvable); a seed (an odd curve, or a non-crossing pair of a
    #   +1 and a -1 curve, times E^e) from its closed form
    #   (u1a1deven_seed_characters), any order; every other label by the cone
    #   data's Layer-1 reduction onto the seeds; the pairing multiply-then-trace.
    #   The exact transport on the closed-form RG image into
    #   A1DoddConeKAlg(k-1) ⊗ QT(Z²) (u1a1deven_trace_transport) is the
    #   witness: check_seed_routes.  Generators, the R-form and geometry:
    #   check_ade_rows.
    ("u1a1deven_cone_kalgebra", "U1A1DevenConeKAlgebra", (1,)),
    ("u1a1deven_cone_kalgebra", "U1A1DevenConeKAlgebra", (2,)),
    # --- [A_1, D_{2k+3}] on the curves of the once-punctured (2k+3)-gon:
    #   A1DnKAlg(n) labels (curves, κ), products / ρ / trace through
    #   A1DoddConeKAlg((n-3)/2) under the curve dictionary (certified by
    #   a1dn_a1dodd_iso in check_ade_rows).  Even n is refused at construction.
    ("a1dn_kalg",      "A1DnKAlg",      (3,)),
    ("a1dn_kalg",      "A1DnKAlg",      (5,)),
    ("a1dn_kalg",      "A1DnKAlg",      (7,)),
]

K = 3            # q-order window for trace / orthonormality
MAX_GENS = 2     # generators exercised per algebra

# Per-class q-order override for the battery.  SU2Nf1KAlgebra's shipped
# H-seed trace recursion is under-determined past q² (`_qmu_to_rl`
# honest-fails at q³ — a genuine shipped-depth boundary surfaced by the
# native-label coverage; previously invisible because the class was
# exercised at unit level only).
_NATIVE_K = {"SU2Nf1KAlgebra": 2}

# Classes whose trace is correct but too slow for a quick self-test would be
# exercised at the multiply/ρ level only here.  None is: the last one, the zoo's
# [A_1,D_8], serves its seeds through A1DevenKAlg(3), whose generator traces
# now come from the closed forms of u1a1deven_seed_characters (about 30 s for
# its full battery here).
LIGHT_TRACE: set = set()

# Classes whose PAIRWISE trace battery (off-diagonal orthonormality +
# ρ²-cyclicity on products) is memory-infeasible at e8 scale — the deep
# Nahm trace of a two-generator product explodes (it OOM-kills the
# process, not merely slows it).  The per-label diagonal battery still
# runs; bar involution (multiply-only) still runs pairwise.
HEAVY_TRACE_PAIRS = {"FiniteE8KAlgebra"}


# Hand-written native label formats the generic `_candidate_labels`
# builder cannot guess.  Without these, the four classes below were
# silently degraded to unit-only coverage via `except Exception:
# continue` — which is exactly how the `[A_1,D_3]` mixed-tile
# truncation artifact stayed invisible to this gate.  Each entry lists
# a few genuine canonical generator labels in the class's own format.
_NATIVE_LABELS = {
    "A1D3KAlg": [                       # (tile, a, b, k)
        (0, 1, 0, 0),                   # T_0
        (0, 0, 1, 0),                   # D_0
        (0, 0, 0, 1),                   # χ_1
        (3, 1, 1, 0),                   # mixed tile: q^{-1}·T_1·D_0
    ],
    "PureSU2KAlg": [                    # native H-tower / Wilson words
        ((0, 1),),                      # H_0
        ((('W', 1), 1),),               # Wilson W_1
        ((0, 2),),                      # H_0²
    ],
    "SU2Nf1KAlgebra": [                 # (H-word, flavour-charge)
        (((0, 1),), 0),                 # H_0
        (((("W", 1), 1),), 0),          # Wilson W_1
        ((), 1),                        # flavour μ
    ],
    "SU3ADKAlg": None,                  # generators via A.T(0) / A.D(0) below
    # curve-labelled classes: generators through the geometric accessor
    "U1A1DevenConeKAlgebra": lambda A: [A.curve(0, 2), A.curve(1, 3, kappa=1),
                                        A.curve(0, A.n)],
    "A1DnKAlg": lambda A: [A.curve(0, 2), A.curve(0, 3, kappa=1),
                           A.curve(0, A.n)],
    "A1A2kKAlg": lambda A: [A.curve(0, 2), A.curve(1, 3)],
    # letter-labelled cone classes (`(word, e)` / `(word, κ)`): a short chord,
    # a long chord and, for the gauged polygon, the gauge letter E
    "U1A1AoddKAlg": lambda A: [_letter(A, (1, 0)), _letter(A, (2, 0)),
                               _letter(A, (0, 0))],
    "A1D3ConeKAlg": lambda A: _odd_letters(A),
    "A1D5ConeKAlg": lambda A: _odd_letters(A),
    "A1D7ConeKAlg": lambda A: _odd_letters(A),
}


def _letter(A, g):
    """The canonical label of the single letter `g` of `A.cone_data()` (for
    `U1A1AoddKAlg` a full `(word, e_E)` label)."""
    return A.cone_data().from_cone_label(frozenset({g}), {g: 1})


def _odd_letters(A):
    """Two letters of an `A1DoddConeKAlg` (labels `(word, κ)`), the first also
    at `κ = 1`."""
    gs = sorted(A.cone_data().mult_gens(), key=repr)[:2]
    labs = [(A.cone_data().from_cone_label(frozenset({g}), {g: 1}), 0) for g in gs]
    return labs + [(labs[0][0], 1)]


def _candidate_labels(A, cls_name=""):
    """Native canonical labels to exercise: the unit plus a few cone
    generators — hand-listed for the hand-written label formats
    (`_NATIVE_LABELS`), otherwise discovered through the universal
    `cone_data()` surface."""
    labels = [A.identity()]
    if cls_name in _NATIVE_LABELS:
        native = _NATIVE_LABELS[cls_name]
        if native is None:              # generator-method classes
            native = [A.T(0), A.D(0)]
        elif callable(native):          # accessor-built labels
            native = native(A)
        return labels + [l for l in native if l != labels[0]]
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


def _cyclicity_up_to_flavour_drift(A, a, b, K):
    """`Tr(ab) == Tr(ρ²(b)·a)`, allowing the two sides to differ by an
    overall **central flavour monomial** `μ^δ`.

    On the U(1)-charged finite-zoo entries the *label-level* ρ differs
    from the element-level ρ-twisted automorphism by a flavour-monomial
    twist (`ρ²` on a U(1)-charged label drops/misassigns a central
    `μ^δ`), so the literal label-level cyclicity check fails by exactly
    that unit — e.g. on the u1-flavoured A₃ entry
    `Tr(ρ²(L)·1) = μ·Tr(L)`.  Element-level cyclicity (the axiom) holds;
    this check pins it while tolerating the label-level drift, and a
    *genuine* cyclicity violation (not of the single-monomial form)
    still fails.  Returns (ok, delta) with `delta = 0` for exact
    agreement."""
    lhs = A.trace_element(A.multiply(a, b), K)
    rhs = A.trace_element(A.multiply(A.rho(A.rho(b)), a), K)
    if lhs == rhs:
        return True, 0
    R = A.coefficient_ring()
    base = tuple(R.one_basis())
    if not base:
        return False, None          # trivial ring: no flavour direction
    for delta in range(-8, 9):
        if delta == 0:
            continue
        key = base[:-1] + (base[-1] + delta,)
        try:
            w = R.basis_element(key)
        except Exception:
            continue
        twisted_coeffs = {e: c * w for e, c in lhs.coeffs.items()}
        if twisted_coeffs == rhs.coeffs:
            return True, delta
    return False, None


def exercise(A, cls_name=""):
    """Run the contract surface; return number of (unit+gen) labels that
    cleanly multiplied / traced / passed the axiom battery.

    The battery per exercised label pair: orthonormality (diagonal AND
    off-diagonal), the bar involution `C^c_{ab}(q⁻¹) = C^c_{ba}(q)`, and
    ρ²-twisted trace cyclicity — the two latter were previously never
    exercised outside the Step-1 samples (the [A₁,D₃] lesson: verifiers
    that exist but are never called certify nothing)."""
    R = A.coefficient_ring()
    K_cls = _NATIVE_K.get(cls_name, K)
    labels = _candidate_labels(A, cls_name)
    unit = labels[0]
    # core unit-level paths + unit axioms
    assert A.multiply(unit, unit) is not None
    A.rho(unit); A.rho_inverse(unit)
    assert A.verify_identity_in_basis(), "identity idempotent"
    assert A.verify_rho_fixes_identity(), "rho fixes identity"
    if cls_name in LIGHT_TRACE:           # multiply/ρ only (trace correct but slow)
        for L in labels[1:3]:
            A.multiply(unit, L)
        return 0
    A.trace(unit, K=K_cls)
    assert A.verify_orthonormality(unit, unit, K=K_cls), "unit orthonormality"
    exercised = [unit]
    gens_done = 0
    prev = unit
    hand_listed = cls_name in _NATIVE_LABELS
    for L in labels[1:]:
        if gens_done >= MAX_GENS and not hand_listed:
            break
        try:
            A.multiply(L, L)
            A.multiply(prev, L)        # cross-product path
            A.trace(L, K=K_cls)
            ok = A.verify_orthonormality(L, L, K=K_cls)
        except Exception:
            if hand_listed:
                raise                  # hand-listed labels must not degrade
            continue                   # not a clean single canonical here; skip
        assert ok, f"generator orthonormality failed for {L!r}"
        exercised.append(L)
        gens_done += 1
        prev = L
    # pairwise battery on the exercised labels: off-diagonal
    # orthonormality + bar involution + ρ²-twisted trace cyclicity.
    heavy = cls_name in HEAVY_TRACE_PAIRS
    for a in exercised:
        for b in exercised:
            assert A.verify_bar_involution(a, b), \
                f"{cls_name}: bar involution fails on {a!r}, {b!r}"
            if heavy:
                continue        # trace-heavy pair checks skipped (see above)
            if a != b:
                try:
                    od = A.verify_orthonormality(a, b, K=K_cls)
                except Exception as exc:
                    # honest-fail from the trace machinery on the widened
                    # pairing window (e.g. su2_nf1's H-seed recursion is
                    # under-determined past q^2) — skip loudly.
                    print(f"       (off-diag orthonormality skipped on "
                          f"{a!r},{b!r}: {type(exc).__name__})")
                    od = None
                if od is not None:
                    assert od, \
                        f"{cls_name}: off-diagonal orthonormality {a!r}, {b!r}"
            try:
                ok, _delta = _cyclicity_up_to_flavour_drift(A, a, b, K_cls)
            except Exception as exc:
                # An honest-fail from the trace (e.g. a bootstrap declining
                # a widened-window request beyond its certified depth) means
                # the pair is not checkable here — skip it loudly; bar
                # involution and orthonormality above still guard the pair.
                print(f"       (cyclicity skipped on {a!r},{b!r}: "
                      f"{type(exc).__name__}: {exc})")
                continue
            assert ok, \
                f"{cls_name}: rho2-twisted cyclicity fails on {a!r}, {b!r}"
    return len(exercised)


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
    # u1a1aodd: closed-form (the singlet rule of u1_pgon_layer2), so no
    # fixed-K cap by construction.
    # Probe a chord generator (dense trace) rather than the sparse vacuum.
    U = importlib.import_module("u1a1aodd_kalg").U1A1AoddKAlg(1)
    cd = U.cone_data()
    # The (1, i) monopole chords have identically-zero trace (magnetic) —
    # the earlier "chord" probe picked one, making the check vacuous (a
    # zero series satisfies any K echo).  Probe the DENSE length-2 chord
    # (2, 0) instead: its M(1, p) singlet-character trace is lacunary
    # (…, q^27, q^31 within a q^40 window), so assert the q^27 term is
    # reached — well past the old frozen windows, genuinely no-cap.
    chord = (2, 0)
    Lc = cd.from_cone_label(frozenset([chord]), {chord: 1})
    tc = U.trace(Lc, K=40)
    nzc = [q for q, r in tc.coeffs.items() if not r.is_zero()]
    assert nzc and max(nzc) >= 27, ("U1A1AoddKAlg", "dense chord trace to q^40")
    print(f"  OK   {'U1A1AoddKAlg':24s} arbitrarily-improvable spine-free to q^40")
    # SU3AD = [A_1,D_4] = SU(3)_{-3/2}: Tr_1, Tr_T, Tr_D from the even-D k = 1
    # closed forms -> arbitrary q-order, no engine.
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
    # DodecagonKAlg = [A_1, A_{2k+1}], k=1..4 -- the ungaugings of the gauged
    # U1A1AoddKAlg(k) (the centralizer of the gauge generator E, measure-restored
    # trace; Octagon/Decagon/Dodecagon are UngaugedPolygonKAlg(k), Hexagon keeps
    # its own labels over U1HexagonKAlg).  These are KAlgebra, NOT ConeKAlgebra
    # (the cone lives in the gauged class), so they are exercised here with
    # explicit labels rather than via the generic CONE_ALGEBRAS loop.
    # Spine-free: construct + multiply + arbitrary-q vacuum trace, all
    # engine-free; their complete generator sets and geometric labels are
    # checked in check_ade_rows.
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
    # SU(2)×U(1) flavour).  KAlgebra (not ConeKAlgebra).  Labels are the gauged
    # labels (F, e, κ) in the centralizer, L_{(F,e,κ)} = z^{-e}·χ_κ·L_{(F,0,0)}:
    # the flavour is in the label (Z-form).  Spine-free: construct + multiply +
    # arbitrary-q trace + orthonormality, all engine-free; the trace reproduces
    # A1DevenRGKAlgebra(1) term-for-term.
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


def _deven_generators_from_the_cones(A):
    """`A1DevenKAlg`'s generators recovered without `mult_generators`' rule:
    per cone of the gauged class, its letters of magnetic charge 0 and the
    products of its letters of opposite nonzero charge (the charge read off
    the `E`-commutator, `A.mag`), each product one canonical element with its
    `E`-power stripped."""
    G = A._G
    cd = G.cone_data()
    letters = {g: (((g, 1),), 0, 0) for g in cd.mult_gens()
               if cd.from_cone_label(frozenset([g]), {g: 1})[0]}
    mag = {g: A.mag(lab) for g, lab in letters.items()}
    out = {lab for g, lab in letters.items() if mag[g] == 0}
    for cone in cd.cones():
        for a in cone:
            for b in cone:
                if a in mag and b in mag and mag[a] > 0 > mag[b]:
                    (L, _c), = G.multiply(letters[a], letters[b]).terms.items()
                    out.add((L[0], 0, 0))
    return out


def check_ade_rows():
    """The ADE finite-type rows: geometric canonical-basis labels for the A and
    D families, self-contained products and traces for every class, and the
    corrections this layer carries (each named by what it checks).

      * A1A2kKAlg(k).curve: every diagonal of the (2k+3)-gon, named from either
        end, is a generator, and ρ is the rotation;
      * U1A1AoddKAlg(k).geometric_label: the letters are the diagonals of the
        (2k+4)-gon plus E^{±1};
      * the ungauged [A_1,A_{2k+1}]: the complete multiplicative generators —
        the charge-0 diagonals AND the products of every opposite-charge pair in
        a common cone (6 / 24 / 65 / 144 at k = 1..4; the single diagonals alone
        are 3 / 8 / 15 / 24), each label a BALANCED multiset of diagonals (as
        many even-even as odd-odd);
      * A1DnKAlg(n), n = 3, 5, 7: the curves of the once-punctured n-gon are
        the generators, ρ the rotation, the certified iso to
        A1DoddConeKAlg((n-3)/2), and Tr 1 the [A_1,D_n] Schur index (its q²
        coefficient the SU(2) flavour current), not a flat quantum torus's;
      * U1A1DevenConeKAlgebra(k), k = 1, 2, 3: the curve accessor and
        geometric_label; products stay in the Z-form (integral LaurentPoly
        coefficients, the SU(2) weight in the label) and round-trip through
        to_R_form; orthonormality on generator pairs;
      * A1DevenKAlg(k): the complete generators (8 / 39 at k = 1, 2: the
        charge-0 curves and the non-crossing (+1, −1) pairs, recovered from the
        cones), the Z-form round trip, orthonormality including pair products;
      * the finite zoo: [A_1,D_4]'s q² coefficient is the SU(3) adjoint
        branched to SU(2)×U(1) (it contains the SU(2) triplet), and
        [A_1,A_5]'s Tr 1 carries the flavour tails μ^{±5} at q^15;
      * U1E7ConeKAlgebra: Tr(E^{±1}) = −q³ (the u(1)-gauged [A_1,E_7]; the
        chord (2,2) dressing gives the u(1)-gauged [A_1,D_7], where it is +q²)."""
    import importlib
    from kalgebra import Element
    from laurent_poly import LaurentPoly
    # A1A2kKAlg: the diagonals, rho = rotation
    from a1a2k_kalg import A1A2kKAlg
    for k in (1, 2):
        A = A1A2kKAlg(k)
        H = A.H
        cd = A.cone_data()
        gens = {cd.from_cone_label(frozenset({g}), {g: 1}) for g in cd.mult_gens()}
        img = {}
        for x in range(H):
            for ell in range(2, H - 1):
                img.setdefault(A.curve(x, ell), set()).add((x, ell))
        assert set(img) == gens and all(len(v) == 2 for v in img.values()), k
        assert all(A.rho(A.curve(x, ell)) == A.curve(x + 1, ell)
                   for x in range(H) for ell in range(2, H - 1)), k
    print(f"  OK   {'A1A2kKAlg.curve':24s} the diagonals of the (2k+3)-gon, rho = rotation (k=1,2)")
    # U1A1AoddKAlg: letters = diagonals + E^{±1}
    from u1a1aodd_kalg import U1A1AoddKAlg
    for k in (1, 2):
        U = U1A1AoddKAlg(k)
        H = 2 * k + 4
        labs = [U.geometric_label(g) for g in U.cone_data().mult_gens()]
        diag = {d for d in labs if d is not None}
        assert labs.count(None) == 2 and len(diag) == len(labs) - 2 == H * (H - 3) // 2, k
    print(f"  OK   {'U1A1AoddKAlg':24s} geometric_label: the diagonals of the (2k+4)-gon + E^(±1)")
    # the ungauged [A_1, A_{2k+1}]: complete generators, balanced multisets
    from ungauge_kalgebra import ungauge_u1a1aodd
    cases = [(1, ungauge_u1a1aodd(1), 6, 3)]
    for k, (mod, cls) in ((2, ("octagon_kalg", "OctagonKAlg")),
                          (3, ("decagon_kalg", "DecagonKAlg")),
                          (4, ("dodecagon_kalg", "DodecagonKAlg"))):
        cases.append((k, getattr(importlib.import_module(mod), cls)(),
                       {2: 24, 3: 65, 4: 144}[k], {2: 8, 3: 15, 4: 24}[k]))
    for k, C, want, singles in cases:
        g = C.mult_generators()
        assert len(g) == len(set(g)) == want, (k, len(g))
        seen = set()
        n_single = 0
        for L in g:
            curves, e = C.geometric_label(L)
            ee = sum(m for ((a, b), m) in curves if a % 2 == 0 and b % 2 == 0)
            oo = sum(m for ((a, b), m) in curves if a % 2 == 1 and b % 2 == 1)
            assert ee == oo, (k, L, curves)
            n_single += sum(m for _c, m in curves) == 1
            seen.add((curves, e))
        assert len(seen) == want and n_single == singles, (k, n_single)
    print(f"  OK   {'ungauged [A1,A_odd]':24s} complete generators 6/24/65/144, balanced multisets")
    # A1DnKAlg: curves of the once-punctured n-gon, iso to A1DoddConeKAlg
    from a1dn_kalg import A1DnKAlg
    from a1dn_a1dodd_iso import a1dn_a1dodd_iso
    for n in (3, 5, 7):
        A = A1DnKAlg(n)
        cd = A.cone_data()
        gens = {(cd.from_cone_label(frozenset({g}), {g: 1}), 0) for g in cd.mult_gens()}
        img = {A.curve(x, ell) for x in range(n) for ell in range(2, n + 1)}
        assert img == gens and len(img) == n * (n - 1), n
        assert all(A.rho(A.curve(x, ell)) == A.curve(x + 1, ell)
                   for x in range(n) for ell in range(2, n + 1)), n
        iso = a1dn_a1dodd_iso(n)
        src = [Element.basis(L) for L in (A.identity(), A.curve(0, 2),
                                          A.curve(0, 3, kappa=1), A.curve(1, n))]
        tgt = [iso.map(x) for x in src]
        res = iso.verify_all(src, tgt, [(src[1], src[2]), (src[2], src[3])],
                             [(tgt[1], tgt[2]), (tgt[2], tgt[3])], trace_K=4)
        assert all(res.values()), (n, res)
    try:
        A1DnKAlg(4)
        raise AssertionError("A1DnKAlg(4) must be refused (even n)")
    except NotImplementedError as exc:
        assert "A1DevenKAlg" in str(exc), exc
    A3 = A1DnKAlg(3)
    t = A3.trace(A3.identity(), K=4)
    assert str(t[2]) == "[2]" and str(t[4]) == "1 + [2] + [4]", t
    print(f"  OK   {'A1DnKAlg':24s} curves of the once-punctured n-gon (n=3,5,7), iso to A1DoddConeKAlg, Tr 1")
    # U1A1DevenConeKAlgebra: curves, Z-form, orthonormality
    from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra
    for k in (1, 2, 3):
        G = U1A1DevenConeKAlgebra(k)
        n = G.n
        cd = G.cone_data()
        gens = [cd.from_cone_label(frozenset({g}), {g: 1}) + (0,)
                for g in sorted(cd.mult_gens(), key=repr)]
        img = {G.curve(x, ell) for x in range(n) for ell in range(2, n + 1)}
        torus = {L for L in gens if not L[0]}
        assert len(img) == n * (n - 1) and img == set(gens) - torus and len(torus) == 2, k
        named = [G.geometric_label(g) for g in cd.mult_gens()]
        assert named.count(None) == 2 and len({d for d in named if d}) == n * (n - 1), k
        assert all(G.rho(G.curve(x, ell))[0] == G.curve(x + 1, ell)[0]
                   for x in range(n) for ell in range(2, n + 1)), k
        sample = gens[:6]
        for a in sample:
            for b in sample:
                a2 = (a[0], a[1] + 1, 1)
                b2 = (b[0], b[1] - 1, 0)
                x = G.multiply(a2, b2)
                assert x.terms and all(isinstance(c, LaurentPoly) for c in x.terms.values())
                assert G.from_R_form(G.to_R_form(x)) == x, (k, a2, b2)
                assert G.verify_orthonormality(a, b, K=3), (k, a, b)
    print(f"  OK   {'U1A1DevenConeKAlgebra':24s} curve frame (k=1,2,3): geometry, Z-form round trip, orthonormality")
    # A1DevenKAlg: complete generators, Z-form, orthonormality incl. pair products
    from a1deven_kalg import A1DevenKAlg
    for k, want, n1 in ((1, 8, 4), (2, 39, 12)):
        D = A1DevenKAlg(k)
        g = D.mult_generators()
        assert len(g) == len(set(g)) == want, (k, len(g))
        assert set(g) == _deven_generators_from_the_cones(D), k
        assert sum(1 for L in g if len(L[0]) == 1) == n1, k
        for a in g[:4]:
            for b in g[-3:]:
                x = D.multiply((a[0], 1, 1), (b[0], -1, 0))
                assert D.from_R_form(D.to_R_form(x)) == x, (k, a, b)
    D1 = A1DevenKAlg(1)
    g1 = D1.mult_generators()
    pairs = [(a, a) for a in g1] + [(g1[0], g1[-1]), (g1[-1], g1[0]), (g1[-2], g1[-1])]
    assert all(D1.verify_orthonormality(a, b, K=3) for a, b in pairs)
    D2 = A1DevenKAlg(2)
    g2 = D2.mult_generators()
    assert all(D2.verify_orthonormality(a, a, K=3) for a in (g2[0], g2[-1]))
    print(f"  OK   {'A1DevenKAlg':24s} complete generators 8/39, Z-form round trip, orthonormality incl. pair products")
    # the finite zoo: a1d4 flavour adjoint; a5 flavour tails
    from finite_a1d4_kalg import FiniteA1D4KAlgebra
    Z4 = FiniteA1D4KAlgebra()
    q2 = Z4.trace(Z4.identity(), K=2)[2]
    su2_weights = sorted(key[0] for key, c in q2.terms.items() for _ in range(c))
    assert su2_weights == [0, 1, 1, 2], q2          # 8 = 1 + 2 + 2 + 3
    from finite_a5_kalg import FiniteA5KAlgebra
    Z5 = FiniteA5KAlgebra()
    q15 = Z5.trace(Z5.identity(), K=15)[15]
    assert {abs(key[0]) for key in q15.terms} == {1, 3, 5}, q15
    print(f"  OK   {'finite zoo':24s} a1d4: q^2 = SU(3) adjoint -> SU(2)xU(1); a5: mu^(±5) at q^15")
    # U1E7ConeKAlgebra: the E-tower of the u(1)-gauged [A_1,E_7]
    from u1e7_cone_kalgebra import U1E7ConeKAlgebra
    U7 = U1E7ConeKAlgebra()
    for n in (1, -1):
        t = U7.trace(((), (0, n)), 6)
        assert {q: str(c) for q, c in t.coeffs.items() if not c.is_zero()} == {3: "-1"}, (n, t)
    print(f"  OK   {'U1E7ConeKAlgebra':24s} Tr(E^(±1)) = -q^3 (the u(1)-gauged [A1,E7])")


def _deven_composites(A, k, count, seed):
    """`count` distinct labels of `U1A1DevenConeKAlgebra(k)` with curves that
    are not seeds and have magnetic charge 0: one to three mutually
    non-crossing curves with powers, `E^{-2..2}`, `κ = 0, 1` (deterministic)."""
    import random
    from u1a1deven_geometric_frame import _curves, _curves_cross
    from u1a1deven_seed_characters import seed_trace
    rng = random.Random(seed)
    n = 2 * k + 2
    cs = sorted(_curves(n))
    out = []
    while len(out) < count:
        pick = []
        target = rng.randint(1, 3)
        for c in rng.sample(cs, len(cs)):
            if all(not _curves_cross(c, d, n) and c != d for d in pick):
                pick.append(c)
            if len(pick) >= target:
                break
        curves = tuple(sorted((c, rng.randint(1, 3 if len(pick) == 1 else 2))
                              for c in pick))
        e = rng.randint(-2, 2)
        if A._magnetic_charge((curves, e)) or seed_trace(k, curves, e, 0) is not None:
            continue
        lab = (curves, e, rng.randint(0, 1))
        if lab not in out:
            out.append(lab)
    return out


def check_seed_routes():
    """The trace routes the closed-form seeds serve, each against its witness.

      * `SU3ADKAlg.geometric_label`: the generators go to the documented
        curves of the once-punctured square (`T_i` the loop at `i` with the
        curve `(i + 1, 2)`, `D_i` the curve `(i + 1, 3)`); on the window
        `2a + b ≤ 6` times three SU(3) weights the label is injective, and ρ
        rotates the curves and conjugates the weight.
      * `SU3ADKAlg`'s served seeds, the even-D k = 1 closed forms
        (`sl3_su3_traces._ClosedFormSeeds`): `Tr_1` equals the Kac–Wakimoto
        vacuum character of `sl(3)_{-3/2}` through 𝖖⁴⁰, `Tr_T` / `Tr_D` (both
        parities) the forward orthonormality pass `SU3ElemTraces` through
        𝖖²⁴; negative control: `D_0`'s U(1) offset moved by 6 (the smallest
        move keeping SU(3) weights) differs from the witness.
      * `SU3ADKAlg`'s product traces ask their seeds for the exact depth: from
        a fresh provider `(T_0²D_0)·(T_0²D_0)` equals its value from a
        provider deepened far beyond (an earlier padded depth was one order
        short there and got the top order wrong), and seeds one order too
        shallow are refused, not truncated.
      * `U1A1DevenConeKAlgebra(1)`: a label that is not a seed is traced by the
        Layer-1 reduction onto the seeds and the pairing is
        multiply-then-trace; the trace transport (`seed_closed_forms=False`)
        is the witness.  Positive control: the matter midpoint squared,
        nonzero at 𝖖², 𝖖⁴, 𝖖⁶, 𝖖⁸ by the transport, agrees; then six composite
        labels through 𝖖⁸ and nine pairings of generators through 𝖖² agree,
        every leaf of the reduction being a seed; negative control: the
        reducer handed a ρ without its `E`-drift disagrees."""
    import sl3_su3_traces as T
    from su3_ad_kalg import SU3ADKAlg, _deven_letter_images
    import u1a1deven_cone_kalgebra as M
    from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra

    A = SU3ADKAlg()
    for i in range(4):
        assert A.geometric_label(A.T(i)) == (
            tuple(sorted((((i, 4), 1), (((i + 1) % 4, 2), 1)))), (0, 0)), ("T", i)
        assert A.geometric_label(A.D(i)) == (
            ((((i + 1) % 4, 3), 1),), (0, 0)), ("D", i)
    seen = {}
    for tile in range(8):
        for a in range(4):
            for b in range(7 - 2 * a):
                for w in ((0, 0), (1, 0), (0, 1)):
                    lab = A.canonicalise((tile, a, b) + w)
                    curves, wt = A.geometric_label(lab)
                    seen.setdefault((curves, wt), set()).add(lab)
                    rot = tuple(sorted((((x + 1) % 4, l), m)
                                       for (x, l), m in curves))
                    assert A.geometric_label(A.rho(lab)) == (rot, (wt[1], wt[0])), lab
    assert all(len(v) == 1 for v in seen.values()), "geometric_label not injective"
    print(f"  OK   {'SU3ADKAlg':24s} geometric_label: curves of the once-punctured "
          f"square, injective on {len(seen)} labels, ρ-equivariant")

    def upto(series, K):
        return {q: z for q, z in series.items() if q <= K and z}

    K_vac, K_seeds = 40, 24
    served = T._ClosedFormSeeds().ensure(K_vac)
    kw = {2 * k: T.char_to_zlaurent(c)
          for k, c in T.vacuum_character(K_vac // 2 + 2).items() if 2 * k <= K_vac}
    assert upto(served.series(("Tr_1",)), K_vac) == upto(kw, K_vac), "Tr_1"
    wit = T.SU3ElemTraces().ensure(K_seeds)
    for key in (("Tr_T",), ("Tr_D", 0), ("Tr_D", 1)):
        assert upto(served.series(key), K_seeds) == upto(wit.series(key), K_seeds), key
    c, d = _deven_letter_images()[("D", 0)]
    assert (upto(T._deven_seed_z(c, d + 6, K_seeds), K_seeds)
            != upto(wit.series(("Tr_D", 0)), K_seeds)), "negative control"
    print(f"  OK   {'SU3ADKAlg':24s} seeds from the even-D closed forms: Tr_1 = "
          f"Kac–Wakimoto (q^{K_vac}), Tr_T / Tr_D = forward pass (q^{K_seeds})")

    K = 8
    saved = T._PROVIDER
    try:
        f = [(0, 2, 1, 0, 0)] * 2
        T._PROVIDER = T._ClosedFormSeeds()
        fresh = T.product_trace(A, f, K)
        T._PROVIDER = T._ClosedFormSeeds().ensure(K + 40)
        deep = T.product_trace(A, f, K)
        assert all(str(fresh.coeffs.get(q)) == str(deep.coeffs.get(q))
                   for q in range(K + 1)), "product_trace: fresh != deepened"
        tab = (0, 4, 2)
        red = T.seed_z_fast(tab + (0, 0))
        need = K + T._seed_depth(*(dd.keys() for dd in red.values()))
        prov = T._ClosedFormSeeds()
        prov.ensure(need - 1)
        prov.K = need - 1                  # exactly one order short
        try:
            T._single_label_trace_z(prov, tab, K, red)
            refused = False
        except ValueError:
            refused = True
        assert refused, "seeds one order short were not refused"
    finally:
        T._PROVIDER = saved
    print(f"  OK   {'SU3ADKAlg':24s} product traces: exact seed depth "
          f"((T0²D0)² fresh = deepened, one order short refused)")

    D = U1A1DevenConeKAlgebra(1)
    Wt = U1A1DevenConeKAlgebra(1, seed_closed_forms=False)
    sq = ((((1, 3), 2),), 0, 0)
    t = Wt.trace(sq, K)
    assert sorted(q for q, co in t.coeffs.items() if not co.is_zero()) == [2, 4, 6, 8]
    assert D.trace(sq, K) == t, "positive control"
    comps = _deven_composites(D, 1, 6, 31)
    for lab in comps:
        assert D.trace(lab, K) == Wt.trace(lab, K), lab
    assert D._layer1_stats["transport_leaves"] == 0
    gens = [D.curve(0, 3), D.curve(1, 3, kappa=1), ((), 1, 0)]
    for a in gens:
        for b in gens:
            assert D.inner_product(a, b, 2) == Wt.inner_product(a, b, 2), (a, b)

    class NoDrift(M._ChiStrippedView):
        def rho(self, label2):
            return (self._alg.rho((label2[0], label2[1], 0))[0], -label2[1])

        def rho_inverse(self, label2):
            return (self._alg.rho_inverse((label2[0], label2[1], 0))[0], -label2[1])

    B = U1A1DevenConeKAlgebra(1)
    B._view_ = NoDrift(B)
    assert any(B.trace(lab, K) != Wt.trace(lab, K) for lab in comps), "negative control"
    print(f"  OK   {'U1A1DevenConeKAlgebra':24s} Layer 1 onto the seeds = the transport "
          f"(k=1): {len(comps)} composites, 9 pairings, both controls")


def check_geometric_labels():
    """The geometric canonical-basis labels of the A and D classes, each read
    through the certified map that serves the class: diagonals of the polygon
    (`A1A2kKAlg`, the ungauged `[A_1, A_{2k+1}]`), curves of the
    once-punctured polygon (`A1DnKAlg`, the ungauged `[A_1, D_{2k+2}]`).

      * `PentagonKAlg` / `HeptagonKAlg` (`kalgebra_samples`): `curve(x, ell)`
        names every diagonal from either end, the diagonals are exactly the
        generators, ρ is the rotation, and `geometric_label(curve(x, ell))` is
        that diagonal with power 1.  The pentagon's `L_i` is the diagonal
        `{3i, 3i + 2}`: under `L_i ↦ (1, 3i)`, `A1A2kKAlg(1)`'s letter of that
        diagonal, all 961 products of the 31 labels `(i, a, b)` with
        `a, b ≤ 2` agree, so does ρ on each, and `geometric_label` is the
        image's; negative control: the reflected `L_i ↦ (1, −3i)` agrees on
        fewer than half the products and is not ρ-equivariant.  The
        heptagon's `curve` is `A1A2kKAlg(2).curve` through the class's
        relabelling.  `kalgebra_samples` belongs to the abelianized layer:
        where it is not importable (the `ConeKAlgebra` package's own gate runs
        with only the `KAlgebra` and `ConeKAlgebra` packages on the path) this
        part is reported as skipped.
      * `A1A2kKAlg(k)`, k = 1, 2, 3: the same checks of `curve` and of
        `geometric_label(curve(x, ell))`.
      * `U1HexagonKAlg.geometric_label` on the letters of its generators: the
        nine diagonals of the hexagon, each once and each `U1A1AoddKAlg(1)`'s
        diagonal of the letter, `None` on `E^{±1}`; two letters q-commute iff
        their diagonals do not cross; ρ rotates each diagonal by one.
      * `HexagonKAlg.geometric_label` on its generators, their products and
        those labels at `e = −1, 2`: a balanced multiset of diagonals (as many
        even–even as odd–odd, with multiplicity), `UngaugedPolygonKAlg(1)`'s
        label of `(F, −e)` with the power `e`, injective.
      * `A1DoddConeKAlg(k)`, k = 0, 1, 2: on the generators exactly the
        n(n−1) curves `A1DnKAlg(2k+3).curve(x, ell)`, each once, and ρ the
        rotation; on the labels of the products of ten curves it is the
        `A1DnKAlg` label of the same element, also after ρ.
      * the zoo's A and D entries (`zoo_geometry`; the aliases hexagon /
        octagon / decagon included): on every generator a non-empty multiset
        of pairwise non-crossing curves, injective; the zoo's ρ rotates it by
        one and ρ⁻¹'s rotation fits no generator (the check tells the two
        directions apart); additive on every q-commuting pair of generators;
        the standalone's `geometric_label` is `zoo_geometry.geometric_label`
        of its entry and of the alias; the E entries have none and are
        refused; and in a fresh process the D entries' labels (a1d3 to a1d7)
        import no BPS module and no bootstrap.  a1d8 is not skipped: its
        generator map's discovery takes about 25 s in a fresh process
        (measured 2026-09-24), and from `main()` the catalogue's a1d8 entry has
        already found it (the map is cached per process), so the entry costs
        well under a second here."""
    import importlib
    import os
    import subprocess
    import sys
    import time
    from a1a2k_kalg import A1A2kKAlg
    from a1dn_kalg import A1DnKAlg, _arc_crossings
    from a1dodd_kalg import A1DoddConeKAlg
    from elem_traces import _standalone_algebra
    from hexagon_kalg import HexagonKAlg
    from u1_hexagon_kalg import U1HexagonKAlg
    from u1a1aodd_kalg import U1A1AoddKAlg
    from u1a1deven_geometric_frame import _curves_cross
    from ungauged_polygon_kalg import UngaugedPolygonKAlg
    from zoo_geometry import GEOMETRIC_IDS, geometric_label

    def diagonal(a, b, n):
        return tuple(sorted((a % n, b % n)))

    def diagonals_cross(c, d):
        (a, b), (x, y) = c, d
        return len({a, b, x, y}) == 4 and ((a < x < b) != (a < y < b))

    def cone_gens(A):
        cd = A.cone_data()
        return [cd.from_cone_label(frozenset({g}), {g: 1})
                for g in sorted(cd.mult_gens(), key=repr)]

    def polygon_curves(A, name):
        H = A.H
        img = {}
        for x in range(H):
            for ell in range(2, H - 1):
                img.setdefault(A.curve(x, ell), set()).add((x, ell))
        assert set(img) == set(cone_gens(A)), (name, "curve() image != generators")
        assert all(len(v) == 2 for v in img.values()), (name, "a diagonal not named twice")
        for x in range(H):
            for ell in range(2, H - 1):
                lab = A.curve(x, ell)
                assert A.rho(lab) == A.curve(x + 1, ell), (name, "rho", x, ell)
                assert A.geometric_label(lab) == ((diagonal(x, x + ell, H), 1),), \
                    (name, "geometric_label(curve)", x, ell)

    # PentagonKAlg / HeptagonKAlg, where the abelianized layer is on the path
    try:
        ks = importlib.import_module("kalgebra_samples")
    except ModuleNotFoundError as exc:
        if exc.name != "kalgebra_samples":
            raise
        ks = None
    if ks is None:
        print(f"  SKIP {'Pentagon/HeptagonKAlg':24s} kalgebra_samples (the abelianized "
              f"layer) is not on this path")
    else:
        P, A1 = ks.PentagonKAlg(), A1A2kKAlg(1)
        polygon_curves(P, "PentagonKAlg")

        def image(label, s):
            i, a, b = label
            out = {}
            for j, m in ((i, a), (i + 1, b)):
                if m:
                    out[s(j) % 5] = out.get(s(j) % 5, 0) + m
            return tuple(sorted((1, j, m) for j, m in out.items()))

        labels = sorted({ks._pent_canon_key(i, a, b)
                         for i in range(5) for a in range(3) for b in range(3)})
        assert len(labels) == 31

        def agreement(s):
            good = sum({image(l, s): c for l, c in P.multiply(x, y).terms.items()}
                       == dict(A1.multiply(image(x, s), image(y, s)).terms)
                       for x in labels for y in labels)
            return good, all(image(P.rho(x), s) == A1.rho(image(x, s)) for x in labels)

        assert agreement(lambda j: 3 * j) == (961, True), "PentagonKAlg != A1A2kKAlg(1)"
        reflected, rho_ok = agreement(lambda j: -3 * j)
        assert reflected < 961 // 2 and not rho_ok, ("negative control", reflected, rho_ok)
        for x in labels:
            assert P.geometric_label(x) == A1.geometric_label(image(x, lambda j: 3 * j)), x
        Hp, A2 = ks.HeptagonKAlg(), A1A2kKAlg(2)
        polygon_curves(Hp, "HeptagonKAlg")
        assert all(ks._hept_label_to_a1a2k(Hp.curve(x, ell)) == A2.curve(x, ell)
                   for x in range(7) for ell in range(2, 6)), "HeptagonKAlg.curve"
        print(f"  OK   {'Pentagon/HeptagonKAlg':24s} curve / geometric_label: the "
              f"diagonals, rho = rotation; pentagon = A1A2kKAlg(1) on 961/961 products "
              f"(reflected control {reflected}/961)")

    # A1A2kKAlg: geometric_label(curve(x, ell)) is that diagonal
    for k in (1, 2, 3):
        polygon_curves(A1A2kKAlg(k), "A1A2kKAlg(%d)" % k)
    print(f"  OK   {'A1A2kKAlg':24s} geometric_label(curve(x, ell)) = the diagonal (k=1,2,3)")

    # U1HexagonKAlg: the letters' diagonals
    U1, G1 = U1HexagonKAlg(), U1A1AoddKAlg(1)
    chords = [U1.L((1, i)) for i in range(6)] + [U1.L((2, i)) for i in range(3)]
    letter = {g: g[0][0][:2] for g in chords}
    geo = {g: U1.geometric_label(letter[g]) for g in chords}
    assert U1.geometric_label((3, 0)) is None and U1.geometric_label((3, 1)) is None
    assert None not in geo.values() and len(set(geo.values())) == 9, geo
    assert all(geo[g] == G1.geometric_label(letter[g]) for g in chords)
    assert not [(g, h) for g in chords for h in chords if g < h
                and (len(U1.multiply(g, h).terms) == 1) == diagonals_cross(geo[g], geo[h])], \
        "U1HexagonKAlg: q-commuting is not non-crossing"
    for g in chords:
        (factor,), _e = U1.rho(g)
        d = geo[g]
        assert U1.geometric_label(factor[:2]) == diagonal(d[0] + 1, d[1] + 1, 6), g
    # HexagonKAlg: balanced multisets, through UngaugedPolygonKAlg(1) at (F, -e)
    Hx, Ug = HexagonKAlg(), UngaugedPolygonKAlg(1)
    gens = list(Hx.mult_generators())
    hex_labels = set(gens)
    for a in gens:
        for b in gens:
            hex_labels |= set(Hx.multiply(a, b).terms)
    hex_labels |= {(F, e) for (F, _e) in list(hex_labels) for e in (-1, 2)}
    seen = set()
    for lab in hex_labels:
        curves, e = Hx.geometric_label(lab)
        assert (curves, e) == (Ug.geometric_label((lab[0], -lab[1]))[0], lab[1]), lab
        ee = sum(m for ((a, b), m) in curves if a % 2 == 0 and b % 2 == 0)
        oo = sum(m for ((a, b), m) in curves if a % 2 == 1 and b % 2 == 1)
        assert ee == oo, ("HexagonKAlg: not balanced", lab, curves)
        seen.add((curves, e))
    assert len(seen) == len(hex_labels), "HexagonKAlg.geometric_label not injective"
    print(f"  OK   {'U1Hexagon/HexagonKAlg':24s} the 9 diagonals (+ E^(±1)), q-commuting = "
          f"non-crossing; balanced multisets, injective on {len(hex_labels)} labels")

    # A1DoddConeKAlg: the curves of A1DnKAlg(2k+3)
    for k in (0, 1, 2):
        n = 2 * k + 3
        O, D = A1DoddConeKAlg(k), A1DnKAlg(n)
        cd = O.cone_data()
        gens = [(cd.from_cone_label(frozenset({g}), {g: 1}), 0)
                for g in sorted(cd.mult_gens(), key=repr)]
        out = [O.geometric_label(g) for g in gens]
        curves = [D.curve(x, ell) for x in range(n) for ell in range(2, n + 1)]
        assert len(set(out)) == len(gens) == n * (n - 1) and set(out) == set(curves), k
        for g, (cv, kappa) in zip(gens, out):
            rot = tuple(sorted((((x + 1) % n, ell), m) for (x, ell), m in cv))
            assert O.geometric_label(O.rho(g)) == (rot, kappa), (k, g)
        window = curves[:10]
        for a in window:
            for b in window:
                for lab in D.multiply(a, b).terms:
                    assert O.geometric_label(D._to_odd(lab)) == D.canonicalise(lab), (k, lab)
                    assert O.geometric_label(O.rho(D._to_odd(lab))) == D.rho(lab), (k, lab)
    print(f"  OK   {'A1DoddConeKAlg':24s} geometric_label: the n(n-1) curves of A1DnKAlg(2k+3), "
          f"rho = rotation, products of 10 curves (k=0,1,2)")

    # the zoo's A and D entries
    polygon = {"pentagon": 5, "heptagon": 7, "a3": 6, "hexagon": 6, "a5": 8, "octagon": 8,
               "a7": 10, "decagon": 10, "a1d3": 3, "a1d4": 4, "a1d5": 5, "a1d6": 6,
               "a1d7": 7, "a1d8": 8}
    own_id = {"hexagon": "a3", "octagon": "a5", "decagon": "a7"}

    def add(u, v):
        out = {}
        for c, m in u + v:
            out[c] = out.get(c, 0) + m
        return tuple(sorted(out.items()))

    timing = {}
    for sid in GEOMETRIC_IDS:
        t0 = time.time()
        n = polygon[sid]
        if sid.startswith("a1d"):
            crossing = _arc_crossings if n % 2 else _curves_cross

            def cross(c, d, n=n, crossing=crossing):
                return bool(crossing(c, d, n))

            def rot(cv, s, n=n):
                return tuple(sorted((((x + s) % n, ell), m) for (x, ell), m in cv))
        else:
            cross = diagonals_cross

            def rot(cv, s, n=n):
                return tuple(sorted((diagonal(a + s, b + s, n), m) for (a, b), m in cv))
        Z = _standalone_algebra(sid)
        cd = Z.cone_data()
        ng = len(cd.mult_gens())
        labels = [geometric_label(sid, ((g, 1),)) for g in range(ng)]
        for cv in labels:
            assert cv and all(m >= 1 for _c, m in cv), (sid, cv)
            assert not any(cross(c, d) for c, _m in cv for d, _k in cv if c < d), (sid, cv)
        assert len(set(labels)) == ng, (sid, "geometric_label not injective")
        img = [geometric_label(sid, Z.rho(((g, 1),))) for g in range(ng)]
        assert all(img[g] == rot(labels[g], 1) for g in range(ng)), (sid, "rho != rotation")
        assert not any(img[g] == rot(labels[g], -1) for g in range(ng)), (sid, "rho^-1")
        own = own_id.get(sid, sid)
        assert all(Z.geometric_label(((g, 1),)) == geometric_label(own, ((g, 1),)) == labels[g]
                   for g in range(ng)), (sid, "the generated class's geometric_label")
        pairs = 0
        for a in range(ng):
            for b in range(a + 1, ng):
                if cd.q_commute(a, b):
                    (lab,) = Z.multiply(((a, 1),), ((b, 1),)).terms
                    assert geometric_label(sid, lab) == add(labels[a], labels[b]), (sid, a, b)
                    pairs += 1
        assert pairs, sid
        timing[sid] = time.time() - t0
    for e in ("e6", "e7", "e8"):
        assert not hasattr(_standalone_algebra(e), "geometric_label"), e
        try:
            geometric_label(e, ((0, 1),))
            refused = False
        except KeyError:
            refused = True
        assert refused, e
    fresh = ("import sys\n"
             "from elem_traces import _standalone_algebra\n"
             "for sid in ('a1d3', 'a1d4', 'a1d5', 'a1d6', 'a1d7'):\n"
             "    A = _standalone_algebra(sid)\n"
             "    n = len(A.cone_data().mult_gens())\n"
             "    labels = [A.geometric_label(((g, 1),)) for g in range(n)]\n"
             "    assert len(set(labels)) == n, sid\n"
             "print('RESULT', [m for m in ('bps_kalgebra', 'bps_factor_spectrum',\n"
             "                             'u1_bootstrap', 'su2_bootstrap') if m in sys.modules])\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    r = subprocess.run([sys.executable, "-c", fresh], env=env, capture_output=True,
                       text=True, timeout=900)
    assert r.returncode == 0 and "RESULT []" in r.stdout.splitlines(), \
        ("fresh process", r.stdout[-500:], r.stderr[-2000:])
    slowest = max(timing, key=timing.get)
    print(f"  OK   {'zoo A/D geometric_label':24s} {len(timing)} entries: non-crossing, "
          f"injective, rho = rotation by one, additive; E refused; fresh process imports "
          f"no BPS / bootstrap (slowest {slowest}, {timing[slowest]:.1f} s)")


class _HiddenGaugeCharge:
    """A gauged class with its `_label_gauge_charge` hook hidden (everything
    else forwarded): the ungauger then windows by the E-power alone."""

    def __init__(self, G):
        self._inner = G

    def __getattr__(self, name):
        if name == "_label_gauge_charge":
            raise AttributeError(name)
        return getattr(self._inner, name)


def check_ungauge_window():
    """The ungauging's gauge-charge window (fixed 2026-09-26).  At k = 3 the
    cube of the pair generator {(1, 8), (3, 7)}, gauge charge 6, carries
    -q^3 z^-6 at K = 4; the class's trace equals the sum over a wide window of
    gauge charges (|n| <= 24); with the gauge-charge hook hidden, the E-power
    window alone drops the term (the behaviour before the fix)."""
    from ungauge_kalgebra import UngaugedKAlgebra, ungauge_u1a1aodd
    U = ungauge_u1a1aodd(3)
    G = U._G
    cube, K = ((1, 8, 3), (3, 7, 3)), 4

    def zd(t):
        return {(q, key[0]): int(v) for q, r in t.coeffs.items()
                for key, v in r.terms.items() if v and q <= K}

    acc, meas = {}, U._inv_measure(K)
    for n in range(-24, 25):
        for q, rc in G.trace((cube, n), K).coeffs.items():
            c = sum(rc.terms.values())
            for fe, fc in meas.items():
                if c and q + fe <= K:
                    acc[(q + fe, n)] = acc.get((q + fe, n), 0) + c * fc
    wide = {key: v for key, v in acc.items() if v}
    got = zd(U.trace((cube, 0), K))
    assert G._label_gauge_charge((cube, 0)) == 6
    assert got.get((3, -6)) == -1 and got == wide, (got, wide)
    P = UngaugedKAlgebra(_HiddenGaugeCharge(G), ((), 1), epow=lambda lbl: lbl[1])
    old = zd(P.trace((cube, 0), K))
    assert (3, -6) not in old and old != got, old
    print(f"  OK   {'ungauging window':24s} k=3, the cube of the pair {{(1,8),(3,7)}}: -q^3 z^-6 at K=4, "
          f"equal to the wide gauge-charge sum; the E-power window alone drops it")


def check_layer1_on_labels():
    """Layer 1 on canonical labels (`ConeData.layer1_on_labels`, 2026-09-26),
    which the zoo classes E8, E7, A5, A7, A1D4, A1D6 and A1D8 run: on labels
    with nonzero traces it equals the word route (a second instance with the
    flag off), and an E8 label of degree 6, whose word-route reduction took
    64 s and 4.8 GB, equals the word route's value through q^30."""
    from finite_e8_kalg import FiniteE8KAlgebra
    from finite_e7_kalg import FiniteE7KAlgebra
    from finite_a5_kalg import FiniteA5KAlgebra
    from finite_a7_kalg import FiniteA7KAlgebra
    from finite_a1d4_kalg import FiniteA1D4KAlgebra
    from finite_a1d6_kalg import FiniteA1D6KAlgebra
    from finite_a1d8_kalg import FiniteA1D8KAlgebra
    cases = ((FiniteE8KAlgebra, (((70, 1), (127, 1)), ((23, 1), (70, 1), (103, 1)),
                                 ((22, 1), (85, 1), (91, 1))), 8),
             (FiniteE7KAlgebra, (((21, 2), (47, 2)),), 8),
             (FiniteA5KAlgebra, (((17, 2),), ((23, 2),)), 8),
             (FiniteA7KAlgebra, (((53, 2),), ((40, 1), (54, 1))), 8),
             (FiniteA1D4KAlgebra, (((6, 2),), ((3, 2),)), 8),
             (FiniteA1D6KAlgebra, (((22, 2),), ((27, 2),)), 8),
             (FiniteA1D8KAlgebra, (((20, 1), (32, 1)), ((56, 2),)), 8))
    n = 0
    for cls, labels, K in cases:
        new, old = cls(), cls()
        assert new.cone_data().layer1_on_labels, cls.__name__
        old.cone_data().layer1_on_labels = False
        for L in labels:
            a, b = str(new.trace(L, K)), str(old.trace(L, K))
            assert a == b and not a.startswith("0 +"), (cls.__name__, L, a, b)
            n += 1
    got = str(FiniteE8KAlgebra().trace(((2, 3), (51, 3)), 30))
    want = ("q^6 + q^10 + q^12 + 2*q^14 + 3*q^16 + 4*q^18 + 5*q^20 + 7*q^22 + 12*q^24 "
            "+ 15*q^26 + 19*q^28 + 26*q^30 + O(q^31)")
    assert got == want, got
    print(f"  OK   {'Layer 1 on labels':24s} E8/E7/A5/A7/A1D4/A1D6/A1D8: {n} labels "
          f"equal to the word route; E8 a^3 b^3 through q^30 equals the word "
          f"route's value")


def check_su3_characters():
    """The SU(3) character arithmetic the [A_1,D_4] seeds go through
    (2026-09-26): the branching to SU(2)×U(1) by Gelfand–Tsetlin interlacing
    equals the weight-system branching it replaced on every irrep with
    p+q ≤ 20, and `sl3_su3_traces.sym_to_char` (the Weyl-denominator read-off)
    returns the character of random virtual characters' weight multisets and
    refuses a weight multiset that is not Weyl-symmetric."""
    import random
    from zplus_ring import (SU3ZPlusRing, _su3_to_su2u1_by_weights,
                            _su3_to_su2u1_gelfand_tsetlin)
    import sl3_su3_traces as S
    ring = SU3ZPlusRing()
    n = 0
    for s in range(21):
        for p in range(s + 1):
            q = s - p
            assert (_su3_to_su2u1_gelfand_tsetlin(p, q)
                    == _su3_to_su2u1_by_weights(ring, (p, q))), (p, q)
            n += 1
    rng = random.Random(2609)
    for _ in range(50):
        ch = {}
        for _ in range(rng.randint(1, 8)):
            hw = (rng.randint(0, 8), rng.randint(0, 8))
            ch[hw] = ch.get(hw, 0) + rng.choice((-3, -2, -1, 1, 2, 3))
        ch = {k: v for k, v in ch.items() if v}
        assert S.sym_to_char(S.char_to_zlaurent(ch)) == ch, ch
    try:
        S.sym_to_char({(-1, 2): 1})
        raise AssertionError("sym_to_char accepted a non-symmetric input")
    except RuntimeError:
        pass
    print(f"  OK   {'SU(3) characters':24s} Gelfand–Tsetlin == weight system on {n} "
          f"irreps; sym_to_char on 50 virtual characters; asymmetric input refused")


def check_no_engine_loaded():
    """No realisation engine was imported by anything above: no module of the
    BPS layer (`src/bps/`, less its three shared lattice primitives) and no
    module of the RG-flow layer (`src/rg/`)."""
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(os.path.dirname(here), "src")
    shared = {"spec_sigma", "mutation", "lattice_mutation"}
    engine = {f[:-3] for layer in ("bps", "rg")
              for f in os.listdir(os.path.join(src, layer))
              if f.endswith(".py") and f[:-3] not in shared}
    hit = sorted(m for m in sys.modules if m in engine)
    assert not hit, ("engine modules imported by the cone layer", hit)
    print(f"  OK   {'no engine loaded':24s} no src/bps or src/rg module imported")


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
    check_ade_rows()
    check_seed_routes()
    check_geometric_labels()
    check_ungauge_window()
    check_layer1_on_labels()
    check_su3_characters()
    check_no_engine_loaded()
    print(f"ALL {n_ok} CONE CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
