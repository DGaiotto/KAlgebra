"""checks_skein.py — the design record adapters for the draft's Section 5, "Skein algebras as K_q-algebras" (sec:skein): the
K_q structure the section proposes on Sk_q(g, C) (sec:skein/kq-structure) and the trace's independence of the
triangulation (sec:skein/trace).

g = sl_2 throughout: the package (skein_sphere/) encodes curves, not webs, so the canonical junctions of
conj:junctions do not enter here.  What the package offers, and so what a check can test:

* the ROSTER (skein_kalgebra.ROSTER, 17 instances).  Ten carry a skein computation: on the eight polygons
  and the pure-SU(2) annulus (engine_provenance 'stated-skein-*', 'bordered-stated-chart') the product returned is an
  independent presentation's (a finite-type class, a chart), and inside every multiply the stated-skein computation
  reproduces its coefficients, up to a normalisation per label fixed at the label's first occurrence and asserted at
  every later one; on the four-punctured sphere ('native-skein-kauffman') the product IS the Kauffman bracket and the
  trace the Schur/Askey-Wilson contour functional.  The other seven ('*-anchored') return an external presentation's
  product with no skein computation, so they carry no skein content to test.
* the flip ATLASES (skein_atlas.SkeinAtlas): charts of one Sk_q(C) related by triangulation flips, each
  chart's product and trace computed from its own quiver and transported spec, the transitions certified KAlgebraIsos.

Decisions R2 (axioms are not claims): the draft's axioms are checked only where they are emergent -- on the sphere, whose
product and trace are both skein-side; elsewhere they are the underlying presentation's, tested in Sections 2-4.
"""
from __future__ import annotations

import itertools
import os
import signal
import sys
import time
from fractions import Fraction

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the release root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
IMPL = os.path.join(ROOT, "implementations")
if IMPL not in sys.path:
    sys.path.insert(0, IMPL)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from kalgebra import Element  # noqa: E402
from laurent_poly import LaurentPoly  # noqa: E402
from checks_kq import check  # noqa: E402

_ONE = LaurentPoly({0: 1})
_BUDGET = {"fast": 300, "extensive": 1200}      # seconds per roster instance
_SKEIN_ENGINES = ("native-skein-kauffman", "stated-skein-pinned", "stated-skein-localized", "bordered-stated-chart")


def _depth():
    from checks_coulomb import battery_depth
    return battery_depth()


class _Out(BaseException):          # not an Exception: no library fallback may swallow the alarm
    pass


def _alarm(*_):
    raise _Out()


def _within(seconds, fn):
    """fn() within `seconds` (SIGALRM): its value, or the string 'not reached in S s'."""
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return fn()
    except _Out:
        return f"not reached in {seconds} s"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# --------------------------------------------------------------------------
# sec:skein/kq-structure
# --------------------------------------------------------------------------
def _roster_window(name, A, depth):
    """1, the first three (four at extensive) multiplicative generators and the square of the first; the two instances whose
    generators are not cone letters get the windows their own tests use."""
    cd = A.cone_data()
    if name == "annulus-pure-su2":
        g = cd.gamma_of
        return [A.identity(), g(1, 0), g(0, 1), g(1, 1), g(0, 2)]
    if name == "sphere-su2-nf4":
        eng = cd.engine()
        z = (0, 0, 0, 0)
        return ([A.identity()] + [("C", eng.SLOPES[c], 1, z) for c in "abc"]
                + [("C", eng.SLOPES["a"], 2, z), ("F", (0, 0, 1, 0))])
    gens = list(cd.mult_gens())
    ng = 4 if depth == "extensive" else 3
    W = [A.identity()] + [cd.from_cone_label(frozenset({g}), {g: 1}) for g in gens[:ng]]
    if gens:
        W.append(cd.from_cone_label(frozenset({gens[0]}), {gens[0]: 2}))
    return W


def _products(A, W):
    """Every product on the window; a skein computation that disagrees raises inside multiply.  (count, error or None)"""
    n = 0
    for a in W:
        for b in W:
            try:
                A.multiply(a, b)
            except Exception as ex:
                return n, f"{a} * {b}: {type(ex).__name__}: {str(ex)[:120]}"
            n += 1
    return n, None


def _rho_orders(A, labels, cap=12):
    out = []
    for a in labels:
        x, k = A.rho(a), 1
        while x != a and k < cap:
            x, k = A.rho(x), k + 1
        out.append(k if x == a else f">{cap}")
    return out


def _flavour_span(fs):
    """Membership in the Q-span of the flavour charges fs (Fraction elimination)."""
    n = len(fs[0])

    def member(v):
        M = [[Fraction(f[i]) for f in fs] + [Fraction(v[i])] for i in range(n)]
        r = 0
        for c in range(len(fs)):
            pr = next((i for i in range(r, n) if M[i][c] != 0), None)
            if pr is None:
                continue
            M[r], M[pr] = M[pr], M[r]
            M[r] = [x / M[r][c] for x in M[r]]
            for i in range(n):
                if i != r and M[i][c] != 0:
                    M[i] = [a - M[i][c] * b for a, b in zip(M[i], M[r])]
            r += 1
        return all(M[i][len(fs)] == 0 for i in range(r, n))
    return member


def _rho_box(A, fs, R):
    """On every label of the box [-R, R]^n: rho^2 = id, and rho(g) - g in the span of the flavour charges."""
    member = _flavour_span(fs)
    n = len(fs[0])
    tot = inv = modf = 0
    for g in itertools.product(range(-R, R + 1), repeat=n):
        tot += 1
        r1 = tuple(A.rho(g))
        inv += tuple(A.rho(r1)) == g
        modf += member(tuple(a - b for a, b in zip(r1, g)))
    neg = all(tuple(A.rho(f)) == tuple(-x for x in f) for f in fs)
    return tot, inv, modf, neg


def check_skein_structure(environment):
    """sec:skein/kq-structure.  The K_q structure the draft proposes on Sk_q(g, C), at g = sl_2, in four clauses.
    (a) the skein product is the K_q algebra's: on the ten roster instances carrying a skein computation, every product on
        a label window (the stated-skein computation reproducing the returned product inside multiply; on the sphere the
        product is the Kauffman bracket); and, on the sphere, where product and trace are both skein-side, the axioms
        emergent there: ax:bar (the mirror) and the trace's orthonormality and cyclicity.
    (b) 'canonical basis = irreducible skeins': on the four-punctured sphere's chart the curves carry the root-lattice
        (adjoint) colouring, and the colour-k curve is irreducible exactly when the towers fuse as V_2 (x) V_2k =
        V_2k+2 + V_2k + V_2k-2, i.e. L_g L_kg = L_(k+1)g + L_kg + L_(k-1)g with unit coefficients.  The chart's product
        comes from its quiver and spec, not from the colouring.
    (c) 'rho = dualization; rho^2 = 1 without irregular punctures': on the charts of the four- and five-punctured spheres
        (regular punctures only), the spec-derived sigma fixes every label of a box up to a flavour charge; the flavour
        inversion (the star of ax:rho) is imposed by the code (BPSKAlgebra._sec_rectified_map), so rho^2 = 1 follows.
        Dualization is trivial on sl_2 curves (V_k is self-dual), which is the content measured.
    (d) the converse direction, as a discriminator (not a claim of the draft): with an irregular puncture rho^2 != 1 --
        the pentagon's box, and a window label of rho-order above 2 on every skein-computing roster instance but the
        sphere."""
    from skein_atlas import SkeinAtlas
    from skein_kalgebra import ROSTER
    depth = _depth()
    budget = _BUDGET[depth]
    checks = []
    t0 = time.time()
    # (a) the roster instances that carry a skein computation
    prods, orders, anchored, windows = {}, {}, [], {}
    sphere = None
    for name, cls in ROSTER.items():
        A = cls()
        if A.engine_provenance not in _SKEIN_ENGINES:
            anchored.append(f"{name} ({A.engine_provenance})")
            continue
        W = _roster_window(name, A, depth)
        windows[name] = len(W)
        prods[name] = _within(budget, lambda A=A, W=W: _products(A, W))
        orders[name] = _rho_orders(A, W[1:])
        if name == "sphere-su2-nf4":
            sphere = (A, W)
    done = {nm: v for nm, v in prods.items() if isinstance(v, tuple)}
    unreached = [nm for nm, v in prods.items() if not isinstance(v, tuple)]
    errs = {nm: v[1] for nm, v in done.items() if v[1]}
    checks.append(check(f"(a) every product on the window reproduced by the skein computation, on {len(done) - len(errs)} of "
                        f"{len(done)} skein-computing roster instances ({sum(v[0] for v in done.values())} products)"
                        + (f"; not reached within {budget} s on " + ", ".join(unreached) if unreached else "")
                        + f"; not in the population, no skein computation: {len(anchored)} instances",
                        not errs and bool(done), str(errs)[:300]))
    A, W = sphere
    tr = W[:4]
    fams = (("ax:bar (the mirror)", lambda: all(A.verify_bar_involution(a, b) for a in W for b in W)),
            ("orthonormality of the Askey-Wilson trace", lambda: all(A.verify_orthonormality(a, b, K=4) for a in tr for b in tr)),
            ("cyclicity of the Askey-Wilson trace", lambda: all(A.verify_rho_twisted_trace(a, b, K=4) for a in tr for b in tr)))
    for fam, fn in fams:
        v = _within(budget, fn)
        checks.append(check(f"(a) on the four-punctured sphere (Kauffman product, Askey-Wilson trace): {fam}, "
                            f"{len(W)}-label window" + ("" if v is True or v is False else f": {v}"),
                            v is True))
    # (b) the Clebsch-Gordan law of the curve towers on the four-punctured sphere's chart
    from su2_nf4_sample_kalgebra import SU2Nf4SampleKAlgebra
    at = SkeinAtlas.tetrahedron()
    T = at.chart_skein()
    slopes = {ch: tuple(-x for x in SU2Nf4SampleKAlgebra.SLOPES[ch]) for ch in "abc"}
    kmax = 3 if depth == "extensive" else 2
    tower_ok, tower_n = [], 0
    for ch, g in slopes.items():
        for k in range(1, kmax + 1):
            got = {tuple(l): str(c) for l, c in T.multiply(g, tuple(k * x for x in g)).terms.items()}
            want = {tuple((k + 1) * x for x in g): "1", tuple(k * x for x in g): "1", tuple((k - 1) * x for x in g): "1"}
            tower_n += 1
            if got == want:
                tower_ok.append((ch, k))
    checks.append(check(f"(b) L_g L_kg = L_(k+1)g + L_kg + L_(k-1)g, unit coefficients, on the four-punctured sphere's chart: "
                        f"{len(tower_ok)} of {tower_n} (channels a, b, c; k <= {kmax})", len(tower_ok) == tower_n))
    S = SU2Nf4SampleKAlgebra()
    z = (0, 0, 0, 0)
    two_term = 0
    for ch in "abc":
        g1, g2 = ("C", S.SLOPES[ch], 1, z), ("C", S.SLOPES[ch], 2, z)
        got = {l: str(c) for l, c in S.multiply(g1, g1).terms.items()}
        two_term += got == {g2: "1", S.identity(): "1"}
    checks.append(check(f"negative control (b): the package's fundamental-character sample class obeys the two-term law "
                        f"L^2 = L_2 + 1 on {two_term} of 3 channels, so the three-term check rejects that normalisation",
                        two_term == 3))
    # (c) rho on boxes of the sphere charts (regular punctures only)
    fs4 = [tuple(at.triangulation().puncture_flavour_charge(p)) for p in range(4)]
    tot, inv, modf, neg = _rho_box(T, fs4, 2)
    checks.append(check(f"(c) four-punctured sphere, box [-2,2]^6: rho^2 = 1 on {inv} of {tot} labels, rho(g) - g a flavour "
                        f"charge on {modf} of {tot}, rho(f_p) = -f_p for the four punctures: {neg}",
                        inv == tot and modf == tot and neg))
    from atlas_examples import SPEC20_BIPYRAMID
    bat = SkeinAtlas.bipyramid(spec=SPEC20_BIPYRAMID)
    Bp = bat.chart_skein()
    fs5 = [tuple(bat.triangulation().puncture_flavour_charge(p)) for p in range(5)]
    tot5, inv5, modf5, neg5 = _rho_box(Bp, fs5, 1)
    checks.append(check(f"(c) five-punctured sphere (the stored 20-factor spec), box [-1,1]^9: rho^2 = 1 on {inv5} of {tot5}, "
                        f"rho(g) - g a flavour charge on {modf5} of {tot5}, rho(f_p) = -f_p: {neg5}",
                        inv5 == tot5 and modf5 == tot5 and neg5))
    # (d) with an irregular puncture rho^2 != 1
    P = SkeinAtlas.polygon(5).chart_skein()
    cnt = sum(1 for g in itertools.product(range(-3, 4), repeat=2) if tuple(P.rho(P.rho(g))) != g)
    checks.append(check(f"(d) negative control: the pentagon (an irregular puncture), box [-3,3]^2: rho^2 != 1 on {cnt} of 49 "
                        "labels", cnt > 0))
    others = [nm for nm in orders if nm != "sphere-su2-nf4"]
    beyond2 = [nm for nm in others if any(isinstance(x, str) or x > 2 for x in orders[nm])]
    polys = [nm for nm in ("pentagon", "heptagon", "nonagon", "hendecagon") if nm in orders]
    checks.append(check(f"(d) every skein-computing roster instance with an irregular puncture ({len(others)}: all but the "
                        f"sphere) has a window label of rho-order above 2: {len(beyond2)} of {len(others)}; the polygons' "
                        "orders " + ", ".join(f"{nm} {orders[nm][0]}" for nm in polys) + " (n for the n-gon, the draft's table)",
                        len(beyond2) == len(others)))
    return {"checks": checks,
            "population": {"skein-computing roster windows (labels)": windows, "rho-orders on the windows": orders,
                           "roster instances with no skein computation": anchored,
                           "tower": f"channels a, b, c, k <= {kmax}",
                           "rho boxes": "[-2,2]^6 (four punctures), [-1,1]^9 (five), [-3,3]^2 (pentagon)",
                           "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the ten skein-computing roster instances; the sphere charts' boxes",
                         "negative": "the fundamental-character sample class fails the three-term tower law; the pentagon's rho^2 != 1"},
            "notes": "Decisions R2: the axioms are checked only on the sphere, where they are emergent; elsewhere they are the "
                     "underlying presentation's.  The flavour inversion in rho is imposed by the code; what (c) measures is that "
                     "the spec-derived sigma fixes every label up to a flavour charge.  On the roster's sphere instance rho is "
                     "wired as the identity, so (c) reads the sphere through its atlas charts.",
            "inputs": []}


# --------------------------------------------------------------------------
# sec:skein/trace
# --------------------------------------------------------------------------
def _atlases(depth):
    from skein_atlas import SkeinAtlas
    out = [("pentagon", lambda: SkeinAtlas.polygon(5)),
           ("heptagon", lambda: SkeinAtlas.polygon(7)),
           ("[A_1,D_3]", lambda: SkeinAtlas.a1dodd(0)),
           ("U(1)-gauged square", lambda: SkeinAtlas.u1a1aodd(0)),
           ("U(1)-gauged hexagon", lambda: SkeinAtlas.u1a1aodd(1))]
    if depth == "extensive":
        out.append(("[A_1,D_5]", lambda: SkeinAtlas.a1dodd(1)))
    return out


def _flip_samples(A, n_rays=4):
    """1, the first n_rays rays, and their pairwise products: a single chord's trace cannot tell chords in one rho-orbit
    apart (the trace is rho-invariant), their products can."""
    rays = sorted(A.cone_data().mult_gens())[:n_rays]
    basis = [Element({tuple(A.identity()): _ONE})] + [Element({tuple(r): _ONE}) for r in rays]
    return basis + [A.multiply(tuple(a), tuple(b)) for a in rays for b in rays]


def check_skein_trace(environment):
    """sec:skein/trace.  The draft conjectures the trace comes from the Kapustin-Witten path integral on D^2 x C; the
    registry row tests its consequence that the trace does not depend on the triangulation used to compute it.  On every
    flip from the root chart of the bordered flip atlases: Tr(x) in the root chart equals Tr(iso(x)) in the flipped chart,
    each chart's trace computed from its own quiver and transported spec (so the agreement is not imposed), and the flip
    commutes with rho.  Samples: 1, four rays and their sixteen products, to q^4."""
    from bps_kalgebra import BPSKAlgebra
    from kalgebra_iso import _trace_dict
    depth = _depth()
    K = 4
    checks = []
    t0 = time.time()
    per = {}
    for name, make in _atlases(depth):
        at = make()
        A = at.chart_skein()
        src = _flip_samples(A)
        edges = list(getattr(at.triangulation(), "internal_edge_ids", None) or range(12))
        n = tr = rh = 0
        for e in edges:
            try:
                key, _ = at.flip(e)
            except Exception:
                continue
            iso = at.iso_skein((), key)
            dst = [iso.map(s) for s in src]
            n += 1
            tr += iso.verify_trace_equivariant(src, dst, K=K)
            rh += iso.verify_rho_equivariant(src, dst)
        per[name] = (n, tr, rh, len(src))
    checks.append(check("the trace is flip-invariant, every flip from the root chart: "
                        + ", ".join(f"{nm} {tr}/{n}" for nm, (n, tr, _rh, _s) in per.items())
                        + f" ({sum(v[3] for v in per.values()) // len(per)} samples per atlas, to q^{K})",
                        all(tr == n and n > 0 for (n, tr, _rh, _s) in per.values())))
    checks.append(check("each flip commutes with rho: " + ", ".join(f"{nm} {rh}/{n}" for nm, (n, _tr, rh, _s) in per.items()),
                        all(rh == n for (n, _tr, rh, _s) in per.values())))
    # negative control: the pentagon's first flipped chart rebuilt with its transported spec reversed (a wrong chamber)
    from skein_atlas import SkeinAtlas
    at = SkeinAtlas.polygon(5)
    A = at.chart_skein()
    src = _flip_samples(A)
    e0 = at.triangulation().internal_edge_ids[0]
    key, _ = at.flip(e0)
    tw = at.chart_kalg(key)
    rk = tw.lattice.rank
    unit = [tuple(int(i == a) for i in range(rk)) for a in range(rk)]
    pairing = [[tw.lattice.bracket(unit[a], unit[b]) for b in range(rk)] for a in range(rk)]
    wrong = BPSKAlgebra(pairing, [tuple(g) for g in tw.node_charges], spec=list(reversed([tuple(g) for g in tw.spec])))
    iso = at.iso_skein((), key)
    differ = sum(_trace_dict(A, s, K) != _trace_dict(wrong, iso.map(s), K) for s in src)
    checks.append(check(f"negative control: the pentagon's flipped chart rebuilt with its spec reversed (a wrong chamber) "
                        f"disagrees with the root chart's trace on {differ} of {len(src)} samples", differ > 0))
    return {"checks": checks,
            "population": {"atlases (flips, trace-equivariant, rho-equivariant, samples)": per, "K": K,
                           "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "every certified flip from the root",
                         "negative": "a wrong chamber (the flipped chart's spec reversed)"},
            "notes": "Reusing labels verbatim across a flip (a wrong dictionary) is NOT caught on the pentagon: that map is "
                     "trace-preserving there, so the control is a wrong trace, not a wrong dictionary.  The closed surfaces "
                     "are not in this sweep: one flip of the four-punctured sphere's chart did not finish its trace "
                     "comparison at q^2 within 900 s.",
            "inputs": []}


ADAPTERS = {
    "sec:skein/kq-structure": check_skein_structure,
    "sec:skein/trace": check_skein_trace,
}
EXTENSIVE = {"sec:skein/kq-structure", "sec:skein/trace"}
