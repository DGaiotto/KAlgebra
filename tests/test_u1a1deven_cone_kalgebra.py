"""Tests for `U1A1DevenConeKAlgebra` — the U(1)-gauged `[A_1, D_{2k+2}]` as a
`ConeKAlgebra` over `SU2ZPlusRing`, on labels `(curves, e, κ)`: curves of the
once-punctured `(2k+2)`-gon, the power of `E = X_{0,1}`, the `SU(2)` weight
(the curve frame since 2026-09-24, the design record; the frame itself is tested in
`tests/test_u1a1deven_geometric_frame.py`, the switch against the retired
tables in the suite in the source repository).

"The oracle" below is the flow `U1A1DevenViaDoddRG(k)` the frame was derived
from; a label's flow label is `A._flow_label(label)`, and the flow is Z-form
too, so products are compared directly.  Labels that earlier versions of these
tests wrote as table rays were carried over once through the table's sections
(the ray the old tests called M3, section `((((1,1,1),1),),(0,0)),0)`, is the
curve (1, 3) at k = 1; the table composites of the audit's errata at k = 2 are
written below as curve labels).  Covered: construction; the section bridges;
`multiply` against the flow on every generator pair (k = 1) and a sample
(k = 2), Z-form (`LaurentPoly` coefficients, `to_R_form` round-trips); the
Clebsch–Gordan of the `SU(2)` weights in the labels; associativity, bar
involution, ρ; `trace` — magnetic vanishing, the gauge sector's closed form,
the closed-form route (2026-09-24: the seeds from `u1a1deven_seed_characters`,
every other label by the generic Layer-1 reduction onto them, the pairing by
multiply-then-trace) against the transport route (`seed_closed_forms=False`,
the witness) — a positive control, composite labels and pairings at k = 1, 2,
3, a negative control, the pair seeds by Layer 1 against their own closed
forms — and the routing (the served route never reaches the transport), the
flow's own windowed trace, the k = 2, 3 matter; the runtime dependencies.

Run from the repo root: `python3 run_tests.py`.
"""
import os
import sys
import itertools
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "implementations"))

from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import SU2ZPlusRing
from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra

_A = None
_FLOWS = {}


def _alg():
    global _A
    if _A is None:
        _A = U1A1DevenConeKAlgebra(1)
    return _A


def _flow(k):
    if k not in _FLOWS:
        from u1a1deven_via_dodd_rg import U1A1DevenViaDoddRG
        _FLOWS[k] = U1A1DevenViaDoddRG(k)
    return _FLOWS[k]


def _gens(A):
    """The generators: every curve, and `E^{±1}`."""
    return [A.curve(x, l) for (x, l) in sorted(A._curve_set)] + [((), 1, 0), ((), -1, 0)]


def _L(n):
    return Element({n: LaurentPoly.one()})


def _terms(el):
    return {l: {q: v for q, v in c._coeffs.items() if v}
            for l, c in el.terms.items() if not c.is_zero()}


def _oracle(A, a, b):
    """The flow's product of the flow labels of `a`, `b`, back on the curve
    labels (ground truth)."""
    k = A.k
    W = _flow(k).multiply(A._flow_label(a), A._flow_label(b))
    return {A._label_of_flow_label(l): d for l, d in _terms(W).items()}


# The ray the table-era tests called M3 (the matter midpoint, section
# ((((1,1,1),1),),(0,0)),0)): the curve (1, 3) at k = 1.
_M3 = ((((1, 3), 1),), 0, 0)
_M3_SEC = (((((1, 1, 1), 1),), (0, 0)), 0)
# Labels of magnetic charge +1 (the table-era magnetic labels, carried over).
_MAGNETIC = [((((1, 4), 1),), 3, 0), ((((1, 2), 1),), 3, 0),
             ((((1, 4), 2), ((2, 2), 1)), 2, 0)]


def test_construction():
    A = _alg()
    cd = A.cone_data()
    assert isinstance(A.coefficient_ring(), SU2ZPlusRing)
    assert len(A._curve_set) == 12                     # 4·3 curves of the square
    assert len(cd.cones()) == 20                       # C(6, 3)
    assert A.identity() == ((), 0, 0)
    assert A.curve(1, 3) == _M3 and A.curve(5, 3, e=2, kappa=1) == ((((1, 3), 1),), 2, 1)
    for bad in [((), 0), ((((1, 3), 1),), 0, -1), ((((0, 2), 1), ((1, 2), 1)), 0, 0),
                ((((0, 5), 1),), 0, 0)]:
        try:
            A.canonicalise(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_bridges_round_trip():
    """`native_of_oracle_section ∘ oracle_section_of == id` on every generator
    (at κ = 0, 1) and every term of their products; the section of M3 is the
    table-era one."""
    A = _alg()
    assert A.oracle_section_of(_M3) == _M3_SEC
    labs = set()
    for kap in (0, 1):
        labs |= {(g[0], g[1], kap) for g in _gens(A)}
    for a, b in itertools.product(_gens(A), _gens(A)):
        labs |= set(A.multiply(a, b).terms)
    for lab in labs:
        assert A.native_of_oracle_section(A.oracle_section_of(lab)) == lab, lab


def test_multiply_matches_oracle_all_generator_pairs():
    """`multiply` reproduces the flow on EVERY generator pair (Z-form: integral
    `LaurentPoly` coefficients, `SU(2)` weight in the label), with no
    fallback."""
    A = _alg()
    gens = _gens(A)
    for a, b in itertools.product(gens, gens):
        P = A.multiply(a, b)
        assert all(isinstance(c, LaurentPoly) for c in P.terms.values())
        assert _terms(P) == _oracle(A, a, b), (a, b)


def test_to_R_form_round_trips():
    """The Z-form contract (`kalgebra.md` "Z-form vs R-form"): on every
    generator product at k = 1 and on 200 at k = 2 (generators at κ = 0, 1),
    `to_R_form` works and `from_R_form(to_R_form(x)) == x`; every label is a
    single-irrep section (`verify_section_is_single_irrep`) and
    `r_label_compose` inverts `r_label_decompose`.  Until 2026-09-24 the
    coefficients were `RLaurent` and `to_R_form` raised (the audit, the item found 2026-09-24)."""
    for k, npairs in ((1, None), (2, 200)):
        A = _alg() if k == 1 else U1A1DevenConeKAlgebra(k)
        gens = [(g[0], g[1], kap) for g in _gens(A) for kap in (0, 1)]
        pairs = list(itertools.product(gens, gens))
        if npairs is not None:
            pairs = random.Random(k).sample(pairs, npairs)
        for a, b in pairs:
            x = A.multiply(a, b)
            assert A.from_R_form(A.to_R_form(x)) == x, (k, a, b)
            for lab in x.terms:
                assert A.verify_section_is_single_irrep(lab), lab
                sec, kap = A.r_label_decompose(lab)
                assert sec == (lab[0], lab[1], 0) and kap == lab[2]
                assert A.r_label_compose(sec, kap) == lab


def test_matter_clebsch_in_the_label():
    """The matter doublet: `M3·M3` is a single cone monomial, and at `SU(2)`
    weight 1 on both factors the product carries `χ₁·χ₁ = χ₀ + χ₂` in the
    labels' third slot, with the same `𝖖`-power."""
    A = _alg()
    R = A.coefficient_ring()
    (lab, c), = A.multiply(_M3, _M3).terms.items()
    assert lab == ((((1, 3), 2),), 0, 0)
    d = A.multiply((_M3[0], 0, 1), (_M3[0], 0, 1))
    assert _terms(d) == {(lab[0], 0, 0): dict(c._coeffs), (lab[0], 0, 2): dict(c._coeffs)}
    assert R.multiply_basis(1, 1) == {0: 1, 2: 1}


def test_associativity():
    A = _alg()
    gens = _gens(A) + [((), 0, 1)]
    random.seed(11)
    for _ in range(150):
        a, b, c = (random.choice(gens) for _ in range(3))
        lhs = A.multiply_elements(A.multiply(a, b), _L(c))
        rhs = A.multiply_elements(_L(a), A.multiply(b, c))
        assert lhs == rhs, (a, b, c)


def test_bar_involution():
    A = _alg()
    gens = _gens(A)
    for a, b in itertools.product(gens, gens):
        assert A.verify_bar_involution(a, b), (a, b)


def test_rho_is_automorphism_and_inverse():
    A = _alg()
    gens = _gens(A)
    for a in gens:
        assert A.rho_inverse(A.rho(a)) == a
        assert A.rho(A.rho_inverse(a)) == a
    for a, b in itertools.product(gens, gens[:6]):
        assert A.rho_element(A.multiply(a, b)) == A.multiply(A.rho(a), A.rho(b))


def test_rho_matches_oracle():
    """Closed-form ρ agrees with the flow's ρ on flow labels."""
    A = _alg()
    T = _flow(1)
    for a in _gens(A) + [A.identity(), ((), 0, 2)]:
        assert A._flow_label(A.rho(a)) == T.rho(A._flow_label(a)), a


def test_rho2_orbit_rep():
    """The closed-form ρ²-orbit representative is constant on orbits,
    idempotent, and reduces the `E`-power mod the drift of a period; `E^e` is
    ρ²-fixed."""
    for k in (1, 2):
        A = _alg() if k == 1 else U1A1DevenConeKAlgebra(k)
        for (x, l) in sorted(A._curve_set):
            for e in (-3, 0, 4):
                lab = A.curve(x, l, e=e)
                r = A._canonical_rho2_orbit_rep(lab)
                assert A._canonical_rho2_orbit_rep(r) == r
                assert A._canonical_rho2_orbit_rep(A.rho(A.rho(lab))) == r
        assert A._canonical_rho2_orbit_rep(((), 5, 1)) == ((), 5, 1)


def test_trace_magnetic_vanishing():
    """`Tr = 0` (exact) for any nonzero X_{1,0} ('t Hooft magnetic) charge."""
    A = _alg()
    for lab in _MAGNETIC:
        assert A._magnetic_charge(lab) != 0
        tr = A.trace(lab, 6)
        assert all(c.is_zero() for c in tr.coeffs.values()), lab


def test_trace_matches_oracle():
    """Neutral traces equal the flow's own windowed trace: the vacuum (Creutzig
    route), the matter midpoint M3 (a seed: its closed form since
    2026-09-24) and M3 at weight 1, and M3 squared (not a seed: the Layer-1
    reduction onto the seeds)."""
    A = _alg()
    T = _flow(1)
    K = 3
    for lab in [A.identity(), _M3, (_M3[0], 0, 1), ((((1, 3), 2),), 0, 0)]:
        assert A.trace(lab, K) == T.trace(A._flow_label(lab), K), lab


def test_seed_traces_from_closed_forms():
    """The seeds — an odd curve, or a non-crossing pair of a +1 and a −1
    curve, times `E^e` — are traced from their closed forms
    (`u1a1deven_seed_characters.seed_trace`) and equal the transport route
    (`seed_closed_forms=False`, the witness) on every seed at k = 1, 2, at
    three `E`-powers and `κ = 0, 1`.  Routing: the served route never
    reaches the transport (its memo stays empty) — neither on a seed nor on a
    label that is not one (a curve squared, reduced by Layer 1) — and the
    witness does."""
    import u1a1deven_cone_kalgebra as M
    from u1a1deven_geometric_frame import _curves, _charge, _curves_cross
    for k, K in ((1, 8), (2, 5)):
        A = U1A1DevenConeKAlgebra(k)
        W = U1A1DevenConeKAlgebra(k, seed_closed_forms=False)
        n = 2 * k + 2
        cs = sorted(_curves(n))
        seeds = [((c, 1),) for c in cs if _charge(c, k) == 0]
        seeds += [tuple(sorted(((a, 1), (b, 1)))) for a in cs for b in cs
                  if _charge(a, k) == 1 and _charge(b, k) == -1
                  and not _curves_cross(a, b, n)]
        assert len(seeds) == {1: 8, 2: 39}[k]         # the A1DevenKAlg(k) generators
        M._TRACE_MEMO.clear()
        for curves in seeds:
            for e in (-1, 0, 2):
                A.trace((curves, e, 0), K)
        assert not M._TRACE_MEMO, list(M._TRACE_MEMO)[:3]
        for curves in seeds:
            for e in (-1, 0, 2):
                for kap in (0, 1):
                    lab = (curves, e, kap)
                    assert A.trace(lab, K) == W.trace(lab, K), (k, lab)
        assert M._TRACE_MEMO                          # the witness did run
        M._TRACE_MEMO.clear()
        A.trace(((((0, 3), 2),), 0, 0), 4)
        assert not M._TRACE_MEMO                      # a non-seed: Layer 1
        W.trace(((((0, 3), 2),), 0, 0), 4)
        assert M._TRACE_MEMO                          # the witness: the transport


def _composites(A, k, count, seed):
    """`count` labels with curves that are not seeds, magnetic charge 0: one to
    three mutually non-crossing curves with powers, `E^{−2..2}`, `κ = 0, 1`."""
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
        out.append((curves, e, rng.randint(0, 1)))
    return out


def test_layer1_route():
    """Every label that is not a seed is traced by the generic Layer-1
    reduction of the cone data (`ConeData.simplify_trace_via_cone_data`,
    through the class's `χ`-stripped view) onto the seeds, and the pairing is
    multiply-then-trace; the transport route (`seed_closed_forms=False`) is
    the witness.
    POSITIVE CONTROL first: M3 squared at k = 1, nonzero at 𝖖², 𝖖⁴, 𝖖⁶, 𝖖⁸ by
    the transport, is equal.  Then composite labels (20 / 15 / 10 at
    k = 1 / 2 / 3 through 𝖖¹⁰ / 𝖖⁶ / 𝖖⁴) and pairings (6 of generators and
    `E^{±1}` per k, 3 of composites at k = 1, 2) equal the witness, and every leaf the reduction
    left was a seed (`_layer1_stats`).  NEGATIVE CONTROL: the reducer handed
    a ρ without its `E`-drift disagrees with the witness.  And the pair seeds,
    reduced by Layer 1 onto single curves and `E^e` (no pair is left as a
    leaf), equal their own closed forms deep (k = 1, 2 through 𝖖⁴⁰, k = 3 through 𝖖³⁰; `E`-powers −1, 0, 1)."""
    import u1a1deven_cone_kalgebra as M
    from u1a1deven_geometric_frame import _curves, _charge, _curves_cross
    A = U1A1DevenConeKAlgebra(1)
    W = U1A1DevenConeKAlgebra(1, seed_closed_forms=False)
    sq = ((((1, 3), 2),), 0, 0)
    t = W.trace(sq, 8)
    assert sorted(q for q, c in t.coeffs.items() if not c.is_zero()) == [2, 4, 6, 8]
    assert A.trace(sq, 8) == t
    for k, count, K in ((1, 20, 10), (2, 15, 6), (3, 10, 4)):
        A = U1A1DevenConeKAlgebra(k)
        W = U1A1DevenConeKAlgebra(k, seed_closed_forms=False)
        labs = _composites(A, k, count, 30 + k)
        for lab in labs:
            assert A.trace(lab, K) == W.trace(lab, K), (k, lab)
        assert A._layer1_stats["transport_leaves"] == 0
        gl = _gens(A)
        rng = random.Random(40 + k)
        pairs = [(rng.choice(gl), rng.choice(gl)) for _ in range(6)]
        if k < 3:     # two composites at k = 3: 28 s here, 118 s for the witness
            pairs += [(labs[i], labs[i + 1]) for i in range(3)]
        for a, b in pairs:
            assert A.inner_product(a, b, 2) == W.inner_product(a, b, 2), (k, a, b)

    class NoDrift(M._ChiStrippedView):
        def rho(self, label2):
            return (self._alg.rho((label2[0], label2[1], 0))[0], -label2[1])

        def rho_inverse(self, label2):
            return (self._alg.rho_inverse((label2[0], label2[1], 0))[0], -label2[1])

    A = U1A1DevenConeKAlgebra(1)
    W = U1A1DevenConeKAlgebra(1, seed_closed_forms=False)
    A._view_ = NoDrift(A)
    labs = _composites(A, 1, 10, 51)
    assert any(A.trace(lab, 8) != W.trace(lab, 8) for lab in labs)
    for k, K in ((1, 40), (2, 40), (3, 30)):
        A = U1A1DevenConeKAlgebra(k)
        n = 2 * k + 2
        cs = sorted(_curves(n))
        pairs = [tuple(sorted(((a, 1), (b, 1)))) for a in cs for b in cs
                 if _charge(a, k) == 1 and _charge(b, k) == -1
                 and not _curves_cross(a, b, n)]
        assert len(pairs) == {1: 4, 2: 27, 3: 96}[k]
        for curves in pairs:
            for e in (-1, 0, 1):
                # not a tautology: the reduction leaves no pair as a leaf
                red = A._layer1_reduction((curves, e))
                assert all(len(leaf[0]) <= 1 for leaf in red.terms), (k, curves, e)
                assert (A._layer1_trace((curves, e), K)
                        == A._seed_trace((curves, e), K)), (k, curves, e)
        assert A._layer1_stats["transport_leaves"] == 0


def test_inner_product_matches_oracle():
    """End-to-end Schur index `I_{a,b} = Tr(ρ(a)·b)` against the flow on M3."""
    A = _alg()
    T = _flow(1)
    lab = A._flow_label(_M3)
    K = 2
    assert A.inner_product(_M3, _M3, K) == T.inner_product(lab, lab, K)


def test_k2_scales_to_D6():
    """k = 2 (U(1)-gauged D₆): multiply matches the flow on a sample of
    generator pairs, plus associativity on a sample of triples."""
    A = U1A1DevenConeKAlgebra(2)
    gens = _gens(A)
    random.seed(2)
    sample = [(random.choice(gens), random.choice(gens)) for _ in range(120)]
    for a, b in sample:
        assert _terms(A.multiply(a, b)) == _oracle(A, a, b), (a, b)
    random.seed(5)
    for _ in range(60):
        a, b, c = (random.choice(gens) for _ in range(3))
        assert (A.multiply_elements(A.multiply(a, b), _L(c))
                == A.multiply_elements(_L(a), A.multiply(b, c))), (a, b, c)


def _blocked(names):
    """Context: `import name` raises for every name in `names`."""
    import contextlib

    @contextlib.contextmanager
    def cm():
        saved = {n: sys.modules.get(n, "__absent__") for n in names}
        for n in names:
            sys.modules[n] = None
        try:
            yield
        finally:
            for n, v in saved.items():
                if v == "__absent__":
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = v
    return cm()


def test_no_tables_on_any_path():
    """The retired table modules (now in the source repository's archive) and the legacy flow
    `u1a1deven_rgkalgebra` blocked: construction, multiply, ρ, the gauge-sector
    trace, a seed's and a composite's trace and a pairing all run at k = 1."""
    import u1a1deven_cone_kalgebra as M
    M._TRACE_MEMO.clear()                  # the memo is per process: start cold
    with _blocked(["u1a1deven_rgkalgebra", "u1a1deven_cone_build",
                   "u1a1deven_dodd_build", "u1a1deven_cone_derivation",
                   "u1a1deven_cone_kalgebra_tables"]):
        A = U1A1DevenConeKAlgebra(1)
        assert len(A.multiply(A.curve(0, 2), A.curve(1, 2)).terms) == 2
        assert A.rho_inverse(A.rho(_M3)) == _M3
        assert A.trace(((), 0, 0), 4).coeffs.get(0).is_one()
        assert not A.trace(_M3, 3).coeffs.get(1).is_zero()        # −χ₁𝖖 + …
        assert A.trace(((((1, 3), 2),), 0, 0), 3).coeffs          # Layer 1
        assert A.inner_product(_M3, _M3, K=2).coeffs.get(0).is_one()


def test_runs_from_an_experiments_script():
    """From a script in the probes in the source repository (its directory first on sys.path, as for
    `python a probe in the source repository), k = 1 constructs and traces a matter label,
    and no module it loads is served from the probes in the source repository or has a same-named
    file there (such a file shadows the module for every script run from
    the probes in the source repository: until 2026-09-23 a stale a probe in the source repository
    made this construction raise, and a copy of the transport module sat there
    too)."""
    import json
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import sys, os, json\n"
        "root = sys.argv[1]\n"
        "sys.path[0] = os.path.join(root, 'experiments')\n"
        "from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra\n"
        "t = U1A1DevenConeKAlgebra(1).trace(((((1, 3), 1),), 0, 0), 4)\n"
        "mods = {n: m.__file__ for n, m in list(sys.modules.items())\n"
        "        if (getattr(m, '__file__', None) or '').startswith(root)}\n"
        "print(json.dumps({'orders': sorted(q for q, c in t.coeffs.items() if not c.is_zero()),\n"
        "                  'mods': mods}))\n")
    env = dict(os.environ, PYTHONPATH=root)
    out = subprocess.run([sys.executable, "-c", code, root], capture_output=True,
                         text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr[-1500:]
    data = json.loads(out.stdout.strip().splitlines()[-1])
    assert data["orders"] and min(data["orders"]) >= 1, data["orders"]
    exp = os.path.join(root, "experiments")
    served = [n for n, f in data["mods"].items() if os.path.dirname(f) == exp]
    twins = [n for n in data["mods"]
             if os.path.exists(os.path.join(exp, n.rsplit(".", 1)[-1] + ".py"))]
    assert not served and not twins, (served, twins)


def test_no_rg_flow_on_the_serving_path():
    """The serving path imports no RG module (the design record; the `imports` check
    of the ADE serving guard).  With the flow `u1a1deven_via_dodd_rg`, the RG
    spine `rgkalgebra` / `graded_rg_solver` and the legacy flow blocked, at
    k = 3: construction, every generator product, ρ, the gauge-sector trace,
    the magnetic zeros, a seed's trace (closed form), a composite's trace
    (Layer 1) and a pairing all run, and so does the witness route
    (`seed_closed_forms=False`: the transport builds its auxiliary algebra and
    `S_RG` itself).  Blocking the transport module instead leaves the served
    route running whole — seeds, composites, pairings (since 2026-09-24 it
    needs no transport) — while the witness raises `ImportError`, the
    documented run-time dependency, not a silent fallback.  And in a fresh
    process the served workload at k = 1, 2, 3 leaves none of those modules,
    nor the transport, in `sys.modules` (positive control: importing the flow
    there shows up)."""
    import json
    import subprocess
    import u1a1deven_cone_kalgebra as M
    M._TRACE_MEMO.clear()                  # the memo is per process: start cold
    rg = ["u1a1deven_rgkalgebra", "u1a1deven_via_dodd_rg", "rgkalgebra",
          "graded_rg_solver"]
    saved_transports = dict(_transports())
    _transports().clear()                  # rebuild the shared transport blocked
    try:
        with _blocked(rg):
            A = U1A1DevenConeKAlgebra(3)
            gens = _gens(A)
            assert len(gens) == 58
            for a, b in itertools.product(gens, gens):
                A.multiply(a, b)
            r = A.curve(2, 3)
            assert A.rho_inverse(A.rho(r)) == r
            from exact_characters import deven_gauged_xn_qn
            for n in range(0, 3):
                assert A.trace(((), n, 0), 8) == A._to_rps(deven_gauged_xn_qn(3, n, 8), 8)
            t = A.trace(A.curve(0, 2), 6)                  # charge −1
            assert all(c.is_zero() for c in t.coeffs.values())
            assert A.trace(A.curve(0, 3), 4).coeffs            # a seed: closed form
            sq = ((((0, 3), 2),), 0, 0)
            assert A.trace(sq, 6).coeffs                       # not a seed: Layer 1
            assert A.inner_product(A.curve(0, 3), A.curve(0, 3), 2).coeffs.get(0).is_one()
            assert not M._TRACE_MEMO                           # no transport served
            Wt = U1A1DevenConeKAlgebra(3, seed_closed_forms=False)
            assert Wt.trace(sq, 6) == A.trace(sq, 6)           # the witness runs too
            assert (3, (sq[0], 0)) in M._TRACE_MEMO
    finally:
        _transports().clear()
        _transports().update(saved_transports)
    M._TRACE_MEMO.clear()
    with _blocked(["u1a1deven_trace_transport"]):
        B = U1A1DevenConeKAlgebra(3)
        assert B.trace(B.curve(0, 3), 4).coeffs            # a seed
        assert B.trace(((((0, 3), 2),), 0, 0), 6).coeffs   # a composite (Layer 1)
        assert B.inner_product(B.curve(0, 3), B.curve(0, 3), 2).coeffs.get(0).is_one()
        Bt = U1A1DevenConeKAlgebra(3, seed_closed_forms=False)
        try:
            Bt.trace(B.curve(0, 3), 4)
        except ImportError:
            pass
        else:
            raise AssertionError("the transport route ran without the transport")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import sys, json\n"
        "RG = ('u1a1deven_via_dodd_rg', 'rgkalgebra', 'graded_rg_solver',\n"
        "      'u1a1deven_rgkalgebra', 'a1deven_rgkalgebra')\n"
        "from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra\n"
        "from a1deven_kalg import A1DevenKAlg\n"
        "for k in (1, 2, 3):\n"
        "    A = U1A1DevenConeKAlgebra(k)\n"
        "    A.multiply(A.curve(0, 3), A.curve(1, 4)); A.rho(A.curve(0, 3))\n"
        "    A.trace(A.curve(1, 3), 4); A.trace(((), 1, 0), 4)\n"
        "    A.trace(((((0, 3), 2),), 0, 0), 4)\n"
        "    A.inner_product(A.curve(1, 3), A.curve(1, 3), 2)\n"
        "    U = A1DevenKAlg(k); g = U.mult_generators()\n"
        "    U.multiply(g[0], g[-1]); U.trace(g[0], 3); U.inner_product(g[0], g[-1], 2)\n"
        "served = [m for m in RG + ('u1a1deven_trace_transport',) if m in sys.modules]\n"
        "import u1a1deven_via_dodd_rg\n"
        "control = [m for m in RG if m in sys.modules]\n"
        "print(json.dumps([served, control]))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=root), cwd=root, timeout=900)
    assert out.returncode == 0, out.stderr[-1500:]
    served, control = json.loads(out.stdout.strip().splitlines()[-1])
    assert served == [], served
    assert "u1a1deven_via_dodd_rg" in control and "rgkalgebra" in control, control


def _transports():
    import u1a1deven_trace_transport as TT
    return TT._TRANSPORTS


def test_gauge_sector_closed_form_arbitrary_q():
    """The whole gauge sector — Tr(1) and the v-tower Tr(X01ⁿ) — equals the
    exact closed-form character `deven_gauged_xn_qn` at **arbitrary q**, well
    past the cone oracle's q_d²⁴ matter-tower-truncation defect (the audit).  This is the regression guard that the defect cannot creep back in."""
    from exact_characters import deven_gauged_xn_qn
    A = _alg()
    K = 24                                            # past the old q_d²⁴ defect
    for n in range(0, 4):
        tr = A.trace(((), n, 0), K)
        cf = deven_gauged_xn_qn(1, n, K)
        for qd in range(0, K + 1):
            got = tr.coeffs.get(qd)
            got = {h: c for h, c in (dict(getattr(got, "terms", {})
                   or getattr(got, "_coeffs", {}))).items() if c} if got else {}
            want = {h: c for h, c in cf.get(qd, {}).items() if c}
            assert got == want, (n, qd, got, want)


def test_gauge_sector_matches_independent_sl3_kw():
    """Cross-check vs the INDEPENDENT `sl(3)₋₃/₂` Kac–Wakimoto vacuum character
    (theta sums, no cone/BPS): the gauged xⁿ slice `(Q;Q)²·[xⁿ]` of the ungauged
    (A1,D_4) index equals the deven Tr(X01ⁿ).  `q_paper = 𝖖² = q_d²`; map
    x-charge `= -(m1+2 m2)/3`, z-power `= m1` (the design notes §2.1)."""
    from sl3_su3_traces import vacuum_character, char_to_zlaurent
    A = _alg()
    MQ = 9                                             # q_d up to 18
    vc = vacuum_character(MQ)
    qq2 = {0: 1}
    for nn in range(1, MQ + 1):
        for _ in range(2):
            nq = {}
            for q, c in qq2.items():
                nq[q] = nq.get(q, 0) + c
                if q + nn <= MQ:
                    nq[q + nn] = nq.get(q + nn, 0) - c
            qq2 = nq

    def to_su2(z):
        if not z:
            return {}
        mx = max(z)
        return {w: z.get(w, 0) - z.get(w + 2, 0)
                for w in range(0, mx + 1) if z.get(w, 0) - z.get(w + 2, 0)}

    for n in range(0, 3):
        G = {}
        for q in range(0, MQ + 1):
            z = {}
            for (m1, m2), c in char_to_zlaurent(vc.get(q, {})).items():
                if (m1 + 2 * m2) == -3 * n:
                    z[m1] = z.get(m1, 0) + c
            G[q] = z
        sl3 = {}
        for q in range(0, MQ + 1):
            zt = {}
            for qi, ci in qq2.items():
                if qi > q:
                    continue
                for w, c in G.get(q - qi, {}).items():
                    zt[w] = zt.get(w, 0) + ci * c
            sl3[2 * q] = {h: c for h, c in to_su2(zt).items() if c}
        tr = A.trace(((), n, 0), 2 * MQ)
        for qd in range(0, 2 * MQ + 1):
            got = tr.coeffs.get(qd)
            got = {h: c for h, c in (dict(getattr(got, "terms", {})
                   or getattr(got, "_coeffs", {}))).items() if c} if got else {}
            assert got == {h: c for h, c in sl3.get(qd, {}).items() if c}, (n, qd)


def test_base_rgkalgebra_fallback_adaptive_cutoff():
    """Guard the base `RGKAlgebra` legacy-fallback hardening (the audit,
    #3b): forcing the deven oracle through the base `_trace_uncached` fallback
    with a deliberately small cutoff *floor* (so the old fixed-cutoff defect
    would bite early, at q_d^{2·floor}) must still reproduce the closed-form
    character — i.e. the fallback now grows the S_RG matter-level cutoff
    (two-cutoff stability) instead of truncating at a fixed level count."""
    from rgkalgebra import RGKAlgebra
    from exact_characters import deven_gauged_Tr1_qn
    from u1a1deven_rgkalgebra import U1A1DevenRGKAlgebra
    orc = U1A1DevenRGKAlgebra(1)                    # the legacy flow (a certification oracle)
    orc._fs_exact_available = lambda: False        # force the legacy fallback
    orc._rg_cutoff = lambda: 4                      # small floor ⇒ defect at q_d^8
    for c in ("_s_rg_elt_cache", "_rho_s_rg_elt_cache", "_rg_S_cache",
              "_fs_certified_cache", "_trace_cache"):
        orc.__dict__.pop(c, None)
    sec = (((), (0, 0)), 0)                         # the legacy flow's vacuum label
    K = 10                                          # crosses the q_d^8 defect
    tr = RGKAlgebra._trace_uncached(orc, sec, K)
    cf = deven_gauged_Tr1_qn(1, K)
    for qd in range(0, K + 1):
        got = tr.coeffs.get(qd)
        got = {h: c for h, c in (dict(getattr(got, "terms", {})
               or getattr(got, "_coeffs", {}))).items() if c} if got else {}
        assert got == {h: c for h, c in cf.get(qd, {}).items() if c}, (qd, got)


def test_gauge_even_k_odd_qd_sign():
    """Even k (odd p=k+1): the v-tower's odd v-powers carry genuine **odd-q_d**
    terms with the √Q=-q_d sign (-1)^{sh}, which the old Q-power `//4` truncation
    silently dropped.  Locks the fix on D6 (k=2) against the oracle-verified
    values (Tr(X01) odd-q_d, all negative); and guards that D4 (k=1, p=2 even)
    stays even-q_d only.  Fast, oracle-free."""
    from exact_characters import deven_gauged_xn_qn
    # k=2 Tr(X01): odd-q_d tower, all negative (verified vs oracle to q_d^14)
    g1 = deven_gauged_xn_qn(2, 1, 14)
    assert g1.get(3) == {1: -1}, g1.get(3)
    assert g1.get(5) == {3: -1}, g1.get(5)
    assert g1.get(7) == {3: -1, 5: -1}, g1.get(7)
    assert g1.get(9) == {1: -1, 3: -1, 5: -1, 7: -1}, g1.get(9)
    assert g1.get(13) == {1: -2, 3: -3, 5: -3, 7: -2, 9: -1, 11: -1}, g1.get(13)
    assert all(qd % 2 == 1 for qd in g1), "k=2 Tr(X01) must be odd-q_d only"
    # k=2 Tr(1): even-q_d, mixed sign (verified vs oracle)
    g0 = deven_gauged_xn_qn(2, 0, 12)
    assert g0.get(0) == {0: 1} and g0.get(2) == {0: -1, 2: 1}
    assert g0.get(12) == {0: 2, 2: 2, 4: 3, 6: 2, 8: 2, 12: 1}, g0.get(12)
    assert all(qd % 2 == 0 for qd in g0), "k=2 Tr(1) must be even-q_d only"
    # k=1 (p=2 even): sh always even ⇒ NO odd-q_d terms (D4 unaffected)
    for n in range(0, 3):
        assert all(qd % 2 == 0 for qd in deven_gauged_xn_qn(1, n, 12)), n


def test_k2_matter_and_gauge():
    """D6 (k = 2): the gauge closed form (incl. the even-k odd-q_d term),
    multiply, and the two legacy composite rays 20 / 24 of the audit's errata.

    Legacy ray 20 at `X01⁰` (the Dodd-table monomial `L_19·L_31·X01⁻¹`) is
    the curve label `((2, 6), (3, 4))` at `E⁻¹`, and legacy ray 24 (`L_8·L_39`)
    is `((3, 6), (4, 4))` — balanced pairs of a loop and a curve.  Their
    traces are the legacy-flow values `−q + 2q³ − (1+χ₂)q⁵ + χ₂q⁷ + …`
    (the audit correction), NOT the A1Dodd-flow values the old k = 2
    freeze held for rays 20 / 24 (`−q + (2−χ₂)q³ + …`, the frame mismatch of
    the audit's errata), and they are orthonormal."""
    A = U1A1DevenConeKAlgebra(2)
    assert len(A._curve_set) == 30
    g3 = A.trace(((), 1, 0), 6).coeffs.get(3)
    assert g3 is not None and not g3.is_zero()
    assert len(A.multiply(A.curve(0, 3), A.curve(1, 4)).terms) >= 1
    want = {1: {0: -1}, 3: {0: 2}, 5: {0: -1, 2: -1}, 7: {2: 1}}
    for lab in [((((2, 6), 1), ((3, 4), 1)), -1, 0), ((((3, 6), 1), ((4, 4), 1)), 0, 0)]:
        t = A.trace(lab, 7)
        got = {q: {int(n): int(v) for n, v in c.terms.items() if v}
               for q, c in t.coeffs.items() if not c.is_zero()}
        assert got == want, (lab, t)
        I = A.inner_product(lab, lab, K=4)
        assert I.coeffs.get(0).is_one(), (lab, I)
        assert not any(q < 0 and not c.is_zero() for q, c in I.coeffs.items())


def test_k3_matter_and_gauge():
    """D8 (k = 3): the gauge closed form against the exact character, and
    canonical orthonormality (`I_{a,a} = 1 + O(𝖖)`, off-diagonal `O(𝖖)`, no
    negative powers) on the charge-0 curves at marked point 0 (ℓ = 3, 5, 7),
    one at marked point 3 and a balanced pair (pairings by multiply-then-trace,
    the products' traces by Layer 1 onto the seeds)."""
    A = U1A1DevenConeKAlgebra(3)
    assert len(A._curve_set) == 56
    from exact_characters import deven_gauged_xn_qn
    for n in range(0, 3):
        assert A.trace(((), n, 0), 8) == A._to_rps(deven_gauged_xn_qn(3, n, 8), 8)
    reps = [A.curve(0, 3), A.curve(0, 5), A.curve(0, 7), A.curve(3, 5),
            ((((0, 8), 1), ((1, 2), 1)), 0, 0)]
    assert all(A._magnetic_charge(r) == 0 for r in reps)
    for r in reps:
        I = A.inner_product(r, r, K=4)
        assert I.coeffs.get(0).is_one(), (r, I.coeffs.get(0))
        assert not any(q < 0 and not I.coeffs[q].is_zero() for q in I.coeffs), (r, "neg-q")
    for a, b in itertools.combinations(reps, 2):
        I = A.inner_product(a, b, K=4)
        c0 = I.coeffs.get(0)
        assert (c0 is None or c0.is_zero()), (a, b, c0)
        assert not any(q < 0 and not I.coeffs[q].is_zero() for q in I.coeffs), (a, b, "neg-q")


def test_section_iso():
    """`u1a1deven_cone_dodd_section_iso(1)` — the iso onto the flow by the
    bijection of labels — is multiplicative on every generator pair and
    ρ-equivariant on every generator (both sides Z-form)."""
    from u1a1deven_cone_kalgebra import u1a1deven_cone_dodd_section_iso
    iso = u1a1deven_cone_dodd_section_iso(1)
    C, T = iso.source, iso.target
    gens = _gens(C) + [((), 0, 1)]
    for a in gens:
        assert iso.map(C.rho_element(_L(a))) == T.rho_element(iso.map(_L(a))), a
        assert iso.inverse(iso.map(_L(a))) == _L(a)
        for b in gens:
            assert (iso.map(C.multiply(a, b))
                    == T.multiply_elements(iso.map(_L(a)), iso.map(_L(b)))), (a, b)


if __name__ == "__main__":
    import time
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, t in tests:
        t0 = time.time()
        t()
        print(f"  PASS: {name} [{time.time() - t0:.1f}s]", flush=True)
    print(f"\nAll {len(tests)} U1A1DevenConeKAlgebra tests passed.")
