"""SkeinKAlgebra (Step 6) self-test.

Run with Steps 1-5 on the path:

    PYTHONPATH=KAlgebra:ConeKAlgebra:RGKAlgebra:BPSKAlgebra:AbeKAlgebra:SkeinKAlgebra \
        python SkeinKAlgebra/test_skein_flows.py

Sections:
  A. the intrinsic (topological) skein algebra — Kauffman resolution
     on the 4-punctured sphere, deliberately KAlgebra-free;
  B. the unified `SkeinKAlgebra(ConeKAlgebra, BPSKAlgebra)` two-parent
     architecture (chart-ful pentagon vs chartless honest-fails);
  C. the named-instance roster + family dispatchers;
  D. certification spots per engine class (pinned / localized /
     native Kauffman / bordered chart / cone-anchored);
  E. `KAlgebraIso` + `KAlgebraObject` legs;
  F. contract axioms through the universal surface;
  G. `SkeinAtlas` — closed, bordered, gauged and flavoured charts
     (incl. the measured 3-term GNO tower);
  H. the independent vacuum anchor (skein chart vs the
     strong-coupling BPS quiver — no chart machinery shared).

Every check is a hard assert; the run prints one line per section.
"""
import time

t00 = time.time()


def section(tag):
    print(f"[{time.time()-t00:6.1f}s] {tag}", flush=True)


# ---------------------------------------------------------------------------
# A. the intrinsic skein algebra (topological layer, KAlgebra-free)
# ---------------------------------------------------------------------------
section("A. intrinsic Kauffman skein algebra (S^2 with 4 punctures)")
from skein_algebra import SkeinAlgebra
from su2_nf4_sample_kalgebra import SU2Nf4SampleKAlgebra
from triangulation import Triangulation

tri = Triangulation.tetrahedron_S2_4()
Sk = SkeinAlgebra(tri)
sl = SU2Nf4SampleKAlgebra.SLOPES
la = next(iter(Sk.curve(sl["a"]).terms))
lb = next(iter(Sk.curve(sl["b"]).terms))
x = Sk.multiply_labels(la, lb)
assert len(dict(x.terms)) == 4, "a x b Kauffman resolution"
# bar antimultiplicativity: bar(a.b) == b.a on multicurve labels
assert dict(x.bar().terms) == dict(Sk.multiply_labels(lb, la).terms)

# ---------------------------------------------------------------------------
# B. the unified class — two-parent architecture
# ---------------------------------------------------------------------------
section("B. SkeinKAlgebra(ConeKAlgebra, BPSKAlgebra) architecture")
from bps_kalgebra import BPSKAlgebra
from cone_kalgebra import ConeKAlgebra
from skein_kalgebra import (PentagonSkeinKAlgebra, U1SquareSkeinKAlgebra,
                            SkeinKAlgebra)

A = PentagonSkeinKAlgebra()          # chart-ful: labels ARE chart charges
B = U1SquareSkeinKAlgebra()          # chartless: BPS routes honest-fail
for cls in (PentagonSkeinKAlgebra, U1SquareSkeinKAlgebra):
    assert issubclass(cls, ConeKAlgebra) and issubclass(cls, BPSKAlgebra)
assert A.has_chart and not B.has_chart
assert A.cache_identity() != B.cache_identity()
try:
    B.to_bpskalgebra()
    raise AssertionError("chartless to_bpskalgebra must honest-fail")
except NotImplementedError:
    pass
# the chart route of the inner product == the KAlgebra default
assert str(A.inner_product_via_chart((1, 0), (1, 0), 6)) == \
    str(A.inner_product((1, 0), (1, 0), 6))

# ---------------------------------------------------------------------------
# C. roster + dispatchers
# ---------------------------------------------------------------------------
section("C. named-instance roster + polygon/su2_nf dispatchers")
from skein_kalgebra import ROSTER

for key in ("pentagon", "u1square", "u1hexagon", "heptagon", "nonagon",
            "annulus-pure-su2", "sphere-su2-nf4", "a1d3", "u1a1d4",
            "disk-su2-nf2", "su2-2punct-a1d3"):
    assert key in ROSTER, key
P7 = SkeinKAlgebra.polygon(7)
x = P7.multiply(((1, 0, 1),), ((1, 1, 1),))
assert len(dict(x.terms)) == 2       # the meson relation
try:
    SkeinKAlgebra.polygon(3)
    raise AssertionError("polygon(3) must honest-fail")
except NotImplementedError:
    pass
try:
    SkeinKAlgebra.su2_nf(5)
    raise AssertionError("su2_nf(5) must honest-fail")
except NotImplementedError:
    pass

# ---------------------------------------------------------------------------
# D. certification spots per engine class
# ---------------------------------------------------------------------------
section("D. engine certification spots")
from laurent_poly import LaurentPoly

# pinned bridge: derived cone reducer over the stated engine == twin
tw = A.to_bpskalgebra()
for pa, pb in [((1, 0), (0, 1)), ((0, 1), (1, 0)), ((1, 1), (0, -1))]:
    assert dict(A.multiply(pa, pb).terms) == \
        dict(tw.multiply(pa, pb).terms), (pa, pb)
# native Kauffman: the sphere's exact cross-ray resolution (pinned
# coefficients — Kauffman signs + the 8s unit)
from skein_kalgebra import SphereSU2Nf4SkeinKAlgebra
S = SphereSU2Nf4SkeinKAlgebra()
assert S.engine_provenance == "native-skein-kauffman"
eng = S.cone_data().engine()
ga = ("C", eng.SLOPES["a"], 1, (0, 0, 0, 0))
gb = ("C", eng.SLOPES["b"], 1, (0, 0, 0, 0))
p = {tuple(l): c for l, c in S.multiply(ga, gb).terms.items()}
assert p == {
    ("C", (1, 1, 0, 0, 1, 1), 1, (0, 0, 0, 0)): LaurentPoly({1: -1}),
    ("C", (1, 1, 2, 2, 1, 1), 1, (0, 0, 0, 0)): LaurentPoly({-1: -1}),
    ("F", (0, 0, 1, 0)): LaurentPoly({0: 1}),
}, p
for l in (ga, gb, ("F", (0, 0, 1, 0))):
    assert S.rho(l) == l             # rho = id, live
# bordered chart (SU(2)+N_f): units are ker(B) sums; roundtrips
N2 = SkeinKAlgebra.su2_nf(2)
cd2 = N2.cone_data()
a0 = sorted(cd2._cliques[0])[0]
assert cd2.verify_roundtrip(a0)
u0, u1 = cd2._units[0], cd2._units[1]
Bx = N2.to_bpskalgebra()
assert dict(Bx.multiply(tuple(u0), tuple(u1)).terms) == \
    dict(Bx.multiply(tuple(u1), tuple(u0)).terms)
# cone-anchored flavoured tier: native SU(2) Clebsch-Gordan
from skein_kalgebra import A1DoddSkeinKAlgebra
D = A1DoddSkeinKAlgebra(0)
chi1 = ((), 1)
xt = {tuple(l): c for l, c in D.multiply(chi1, chi1).terms.items()}
assert set(xt) == {((), 0), ((), 2)}          # chi1*chi1 = chi0 + chi2
sec, irr = D.r_label_decompose(
    (((sorted(D.intrinsic.cone_data().mult_gens())[0], 1),), 1))
assert irr == 1                               # the R-basis LABEL kappa

# ---------------------------------------------------------------------------
# E. KAlgebraIso + KAlgebraObject legs
# ---------------------------------------------------------------------------
section("E. isos + object legs")
from kalgebra import Element

one = LaurentPoly({0: 1})
iso = A.build_iso()                  # pentagon skein-cone <-> BPS twin
samples = [Element({l: one}) for l in
           [(0, 0), (1, 0), (0, 1), (1, 1), (0, -1)]]
prs = [(x_, y_) for x_ in samples[:3] for y_ in samples[:3]]
assert iso.verify_unit()
assert iso.verify_round_trip(samples, samples)
assert iso.verify_multiplicative(prs, prs)
from sqed1_object import sqed1_object
obj = sqed1_object()
assert "skein-cone" in obj.keys()
oiso = obj.iso("skein-cone", "cone")
osamples = [Element({l: one}) for l in [(0, 0), (1, 0), (-1, 0), (0, 1)]]
assert oiso.verify_round_trip(osamples, osamples)
assert oiso.verify_multiplicative(
    [(x_, y_) for x_ in osamples[:3] for y_ in osamples[:3]],
    [(x_, y_) for x_ in osamples[:3] for y_ in osamples[:3]])
from hexagon_objects import u1hexagon_object
hobj = u1hexagon_object()
assert "skein-cone" in hobj.keys()

# ---------------------------------------------------------------------------
# F. contract axioms through the universal surface
# ---------------------------------------------------------------------------
section("F. contract axioms (bar / rho^2-twisted trace / orthonormality)")
assert A.verify_identity_in_basis()
assert A.verify_bar_involution((1, 0), (0, 1))
assert A.verify_rho_twisted_trace((1, 0), (0, 1), K=6)
assert A.verify_orthonormality((1, 0), (1, 0), K=6)
assert A.verify_orthonormality((1, 0), (0, 1), K=6)
assert B.verify_bar_involution((1, 0), (0, 1))
assert B.verify_rho_twisted_trace((1, 0), (-1, 0), K=6)
assert B.verify_orthonormality((1, 0), (1, 0), K=6)
tr = B.trace((1, 0), 6)
assert all(str(tr[k]) == "0" for k in range(7))    # monopoles vanish

# ---------------------------------------------------------------------------
# G. SkeinAtlas — flip atlases of one Sk(Sigma)
# ---------------------------------------------------------------------------
section("G. SkeinAtlas (closed / bordered / gauged / flavoured)")
from skein_atlas import SkeinAtlas

import sys

att = SkeinAtlas.tetrahedron()       # closed: multicurve dictionary
At = att.chart_skein()
g0 = sl["a"]                         # a channel curve of S^2_{0,4}
cdt = At.cone_data()
f0 = cdt._fps[0]                     # a peripheral (flavour) class
assert cdt.ray_kind(f0) == "torus"
gl = cdt.to_cone_label(f0)           # multicurve component dictionary
assert cdt.from_cone_label(*gl) == tuple(f0)
try:                                  # mixed core+flavour: honest-fail
    cdt.to_cone_label(g0)
    raise AssertionError("mixed label must honest-fail")
except ValueError:
    pass
if "--full" in sys.argv:
    # The measured 3-term GNO tower L_g^2 = L_{2g} + L_g + 1 (unit
    # coefficients) — the flagship closed-chart measurement.  Gated:
    # the closed chart multiply costs tens of minutes on this
    # package's Step-4 spine until the upstream caching layer lands
    # (see SKEINKALGEBRA_EXPORT_STATUS.md); it is certified
    # development-side in the Cluster battery.
    tow = {tuple(l): str(c)
           for l, c in At.multiply(g0, g0).terms.items()}
    two = tuple(2 * v for v in g0)
    zero = tuple(0 for _ in g0)
    assert tow == {two: "1", tuple(g0): "1", zero: "1"}, tow
at5 = SkeinAtlas.polygon(5)          # bordered: flip + lifted iso
key5, iso5 = at5.flip(at5.triangulation().internal_edge_ids[0])
assert iso5.verify_unit()
A5 = at5.chart_skein()
tw5 = at5.chart_kalg(())
r5 = sorted(A5.cone_data().mult_gens())
assert dict(A5.multiply(r5[0], r5[1]).terms) == \
    dict(tw5.multiply(r5[0], r5[1]).terms)
atg = SkeinAtlas.u1a1aodd(0)         # gauged square
Ag = atg.chart_skein()
tg = atg.chart_kalg(())
rg = sorted(Ag.cone_data().mult_gens())
for x_, y_ in [(rg[0], rg[1]), (rg[1], rg[0])]:
    assert dict(Ag.multiply(x_, y_).terms) == \
        dict(tg.multiply(x_, y_).terms)
atd = SkeinAtlas.a1dodd(0)           # flavoured (SU(2) weight expansion)
Ad = atd.chart_skein()
td = atd.chart_kalg(())
rd = sorted(Ad.cone_data().mult_gens())
for x_ in rd[:4]:
    for y_ in rd[:4]:
        if x_ == y_:
            continue
        assert dict(Ad.multiply(x_, y_).terms) == \
            dict(td.multiply(x_, y_).terms), (x_, y_)
try:
    SkeinAtlas.a1dodd(3)
    raise AssertionError("a1dodd(3) must honest-fail")
except NotImplementedError:
    pass

# ---------------------------------------------------------------------------
# H. the independent vacuum anchor
# ---------------------------------------------------------------------------
section("H. vacuum Schur anchor vs the strong-coupling BPS quiver")
from bps_su2_nf1 import build_bps_su2_nf1

N1 = SkeinKAlgebra.su2_nf(1)
tr_skein = N1.trace(N1.identity(), 3)
Bq = build_bps_su2_nf1()
tr_bps = Bq.trace((0, 0, 0), 3)


def _aug(rc):
    """Augment an abelian-refined coefficient to its integer dimension."""
    if hasattr(rc, "terms"):
        return sum(int(v) for v in rc.terms.values())
    return int(rc)


for q in range(4):
    assert _aug(tr_skein[q]) == _aug(tr_bps[q]), q
# refined under the measured integral flavour frame map e -> 2e
for q in range(4):
    ws = {(2 * w[0],): v for w, v in
          (tr_skein[q].terms.items() if hasattr(tr_skein[q], "terms")
           else {(0,): int(tr_skein[q])}.items())}
    wb = {tuple(w): v for w, v in
          (tr_bps[q].terms.items() if hasattr(tr_bps[q], "terms")
           else {(0,): int(tr_bps[q])}.items())}
    assert ws == wb, q

print(f"\nALL SkeinKAlgebra self-tests pass  ({time.time()-t00:.1f}s)")
