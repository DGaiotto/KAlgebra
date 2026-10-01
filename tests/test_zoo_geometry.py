"""Geometric labels of the zoo's A and D entries (`zoo_geometry`,
the design record).

For every A / D zoo entry (the aliases hexagon / octagon / decagon included):
  * on every generator `geometric_label` is a non-empty multiset of pairwise
    non-crossing curves (diagonals of the polygon for A, curves of the
    once-punctured polygon for D), injective; the zoo's ρ rotates it by one,
    and ρ⁻¹'s rotation fits no generator (the check tells the directions apart);
  * additivity: for every q-commuting pair of generators, the product's single
    label is named by the sum of the two multisets;
  * the generated classes' `geometric_label` is `zoo_geometry.geometric_label`
    of their entry, and an alias gets the same labels as its entry;
  * the E entries have no `geometric_label`, and `zoo_geometry` refuses them;
  * in a fresh process the labels of the D entries come with no BPS function
    run and neither `bps_kalgebra` nor a bootstrap imported.

Run:  `python3 run_tests.py`
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "implementations")):
    if _p not in sys.path:
        sys.path.append(_p)

import finite_kalgebras as fk
from zoo_geometry import GEOMETRIC_IDS, geometric_label

PASS, FAIL = [], []

POLYGON = {"pentagon": 5, "heptagon": 7, "a3": 6, "hexagon": 6, "a5": 8, "octagon": 8,
           "a7": 10, "decagon": 10, "a1d3": 3, "a1d4": 4, "a1d5": 5, "a1d6": 6,
           "a1d7": 7, "a1d8": 8}
OWN_ID = {"hexagon": "a3", "octagon": "a5", "decagon": "a7"}


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return ok


def _geometry(sid):
    n = POLYGON[sid]
    if sid.startswith("a1d"):
        if n % 2:
            from a1dn_kalg import _arc_crossings as crossings
        else:
            from u1a1deven_geometric_frame import _curves_cross as crossings

        def cross(c, d):
            return bool(crossings(c, d, n))

        def rot(cv, s):
            return tuple(sorted((((x + s) % n, ell), m) for (x, ell), m in cv))
    else:
        def cross(c, d):
            (a, b), (x, y) = c, d
            return len({a, b, x, y}) == 4 and ((a < x < b) != (a < y < b))

        def rot(cv, s):
            return tuple(sorted((tuple(sorted(((a + s) % n, (b + s) % n))), m)
                                for (a, b), m in cv))
    return cross, rot


def _sum(u, v):
    out = {}
    for c, m in u + v:
        out[c] = out.get(c, 0) + m
    return tuple(sorted(out.items()))


def test_generators():
    for sid in GEOMETRIC_IDS:
        Z = fk.FINITE_KALGEBRAS[sid]()
        n = len(Z.cone_data().mult_gens())
        cross, rot = _geometry(sid)
        labels = [geometric_label(sid, ((g, 1),)) for g in range(n)]
        ok = all(cv and all(m >= 1 for _c, m in cv)
                 and not any(cross(c, d) for c, _m in cv for d, _k in cv if c < d)
                 for cv in labels)
        fwd = sum(geometric_label(sid, Z.rho(((g, 1),))) == rot(labels[g], 1)
                  for g in range(n))
        bwd = sum(geometric_label(sid, Z.rho(((g, 1),))) == rot(labels[g], -1)
                  for g in range(n))
        check(f"{sid}: {n} generators, non-crossing and injective; ρ = rotation on "
              f"{fwd}/{n}, ρ⁻¹'s rotation on {bwd}",
              ok and len(set(labels)) == n and fwd == n and bwd == 0)


def test_additivity():
    for sid in GEOMETRIC_IDS:
        Z = fk.FINITE_KALGEBRAS[sid]()
        cd = Z.cone_data()
        n = len(cd.mult_gens())
        good = tot = 0
        for a in range(n):
            for b in range(a + 1, n):
                if not cd.q_commute(a, b):
                    continue
                (lab,) = Z.multiply(((a, 1),), ((b, 1),)).terms
                tot += 1
                good += geometric_label(sid, lab) == _sum(
                    geometric_label(sid, ((a, 1),)), geometric_label(sid, ((b, 1),)))
        check(f"{sid}: additive on {good}/{tot} q-commuting generator pairs",
              tot > 0 and good == tot)


def test_generated_classes():
    bad = []
    for sid in GEOMETRIC_IDS:
        A = fk.FINITE_KALGEBRAS[sid]()
        own = OWN_ID.get(sid, sid)
        for g in range(len(A.cone_data().mult_gens())):
            lab = ((g, 1),)
            if not (A.geometric_label(lab) == geometric_label(own, lab)
                    == geometric_label(sid, lab)):
                bad.append((sid, g))
    e_free = all(not hasattr(fk.FINITE_KALGEBRAS[e](), "geometric_label")
                 for e in ("e6", "e7", "e8"))
    try:
        geometric_label("e6", ((0, 1),))
        refused = False
    except KeyError:
        refused = True
    check("the generated classes' geometric_label is zoo_geometry's, aliases "
          "included; the E entries have none and are refused",
          not bad and e_free and refused)


_FRESH = r"""
import sys, os
sys.path.insert(0, os.getcwd())
ran = set()
def prof(frame, event, arg):
    if event == "call" and frame.f_code.co_filename.endswith(
            ("bps_kalgebra.py", "bps_factor_spectrum.py")):
        ran.add(frame.f_code.co_name)
sys.setprofile(prof)
import finite_kalgebras as fk
for sid in ("a1d3", "a1d4", "a1d5", "a1d6", "a1d7"):
    A = fk.FINITE_KALGEBRAS[sid]()
    labels = [A.geometric_label(((g, 1),)) for g in range(len(A.cone_data().mult_gens()))]
    assert len(set(labels)) == len(labels), sid
sys.setprofile(None)
bad = [m for m in ("bps_kalgebra", "finite_kalgebras.u1_bootstrap",
                   "finite_kalgebras.su2_bootstrap") if m in sys.modules]
print("RESULT", sorted(ran), bad)
"""


def test_fresh_process():
    env = dict(os.environ, PYTHONPATH=_ROOT)
    r = subprocess.run([sys.executable, "-c", _FRESH], cwd=_ROOT, env=env,
                       capture_output=True, text=True, timeout=900)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT")]
    check("fresh process: the D entries' labels with no BPS function run and "
          "neither bps_kalgebra nor a bootstrap imported",
          r.returncode == 0 and line == ["RESULT [] []"])
    if r.returncode:
        print(r.stderr[-2000:])


if __name__ == "__main__":
    test_generators()
    test_additivity()
    test_generated_classes()
    test_fresh_process()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
