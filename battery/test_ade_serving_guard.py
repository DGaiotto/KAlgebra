"""In this release this module ships in `battery/` for its serving table `ROWS`
(the classes that serve each `[A_1,ADE]` entry, with their builders), which the
battery's row for the table of finite-type algebras reads.  As a test it audits
the source repository's layout -- its file paths and two of its controls name
modules of that repository's archive -- so it is not part of `run_tests.py`.

Regression guard: the classes that serve the ADE finite K-algebras (and the
U(1)-gauged ones where the flavour has a U(1)) are self-contained, fully
functional, and — for the A and D families — geometrically labelled.

"Self-contained" (the session's reading of 2026-09-24, recorded in
the design notes and listed for the author's veto): no frozen trace data and no BPS
engine or RG flow on the serving path.  An orthonormality bootstrap seeded by the
class's own closed-form Tr(1) is self-contained (exact, to any order, from the
class's own data and the orthonormality axiom): the `bootstrap` check still
detects it, and its rows list it in `EXPECTED_FAILURES` as recorded, not as a
defect.  Stored product data (cone / ρ tables) is declared per row — as Python
literals (`stored`) or data files (`data_files`) — not failed.  "Fully
functional": multiply and trace on every label, to any order.  Each row below is one serving
class, run in a FRESH process under the instruments of
the design record (its
`INSTRUMENTS`, loaded by path: the import guard on the frozen trace-table module,
the `open()` spy, the profile hook), switched on before the row's first import.
The checks, by name:

  frozen     no frozen trace data is read: the import guard is not triggered
             and no data file (pickle, json, gz, ...) is opened outside the
             Python installation;
  stored     the modules on the serving path whose numeric-literal density
             marks them as stored data (the audit's `literal_density` and
             threshold) are exactly the row's DECLARED stored data — cone /
             product tables and recipe coefficients kept as Python literals,
             listed in the row, so that a new table on the path fails here.  A
             zoo module that the package `finite_kalgebras/__init__` preloads
             counts only when a method of one of its classes runs;
  bps        no function of `bps_kalgebra.py` / `bps_factor_spectrum.py` runs
             (the audit's `CHART_FILES`);
  imports    no bootstrap module (name containing `bootstrap`, or
             `su2_reliable`) and no RG-flow module (`rgkalgebra`, `rg_flow`,
             `*_rg`, ...) is imported;
  bootstrap  no orthonormality-bootstrap routine runs: a function of a
             bootstrap module, or one of `_BOOTSTRAP_ROUTINES`
             (`elem_traces._generate_bootstrap`, the `SU3ElemTraces` seed
             solves of `sl3_su3_traces`);
  tr1        `Tr(1)` through 𝖖⁴⁰ (the row's window where the class documents a
             depth limit), and `Tr(1) = 1 + O(𝖖)`;
  gens       the trace of every multiplicative generator through 𝖖¹² (the
             row's window), each `O(𝖖)` — `I_{1,g} = Tr(L_g) = δ_{1,g} + O(𝖖)`;
  ortho      `I(L_a, L_b) = δ_ab + O(𝖖)` on generator pairs, in the repo's
             convention (`KAlgebra.verify_orthonormality`: no negative powers,
             the χ₀-component of the 𝖖⁰ coefficient is δ), through
             `inner_product` — or, where the class carries the μ^δ of ρ on
             `rho_element` rather than on the label (the audit), in the
             element form `trace_element(multiply_elements(rho_element(L_a),
             L_b))`;
  geometry   A and D rows: the class's geometric-label accessor is total and
             injective on the generators (and ρ is the rotation on it where the
             class says so).

A row fails a check or passes it; the rows that are known to fail a check are
listed BY NAME in `EXPECTED_FAILURES` (built from `_EXPECTED`: rows, checks,
reason) and the test asserts that exactly those fail: a newly failing check is a
regression, and a listed check that now passes must be removed from its entry.
Every window is recorded in the row; `finite_type_kalgebras.md` §9 carries the
per-row table.

Positive controls run first and abort the run if one does not fire: a synthetic
module with a dense numeric table and a pickle written to a temporary directory,
both read on a serving path (the old frozen classes are retired, so the frozen
positives are synthetic, as `frozen_data_audit.py`'s are); an import of the
frozen trace-table module; the `BPSKAlgebra` pentagon (the oracle positive); an
import of a bootstrap module and of an RG module; the two orthonormality-
bootstrap routines run directly; and the δ + O(𝖖) check itself, which must fail
where it should (zoo `a3`: the label form of the pairing on 3 of its 6
generators, the audit; `Tr(1)` read as a generator's trace).  A clean
snippet is the negative control.

Run:  `python3 run_tests.py` [--slow]
      [--rows NAME,NAME,...] [--survey] [--jobs N]
  (default) the controls and all 47 rows: 259 s with two processes (other
            jobs sharing the machine);
  --slow    every ordered generator pair of every row (`A1DevenKAlg(3)` and the
            zoo's a1d6 / a1d8 keep a sample of 60 generators: 3 minutes for
            `A1DevenKAlg(3)`'s 3660 pairs, 2 for zoo a1d8's), and three times
            the time caps;
  --rows    only the named rows (the controls still run first);
  --survey  print every row's raw record (timings, windows, details) as JSON;
  --jobs    child processes at once (default 2).
(Timings measured 2026-09-24 on a shared four-core machine.)
"""
import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THIS = os.path.abspath(__file__)
_AUDIT_PATH = os.path.join(_ROOT, "battery", "frozen_data_audit.py")


def _load_audit():
    spec = importlib.util.spec_from_file_location("frozen_data_audit", _AUDIT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AUDIT = _load_audit()

CHECKS = ("frozen", "stored", "bps", "imports", "bootstrap", "tr1", "gens", "ortho",
          "geometry")

_BOOTSTRAP_MODULE = re.compile(r"bootstrap|(^|\.)su2_reliable$")
_RG_MODULE = re.compile(r"rgkalgebra|(^|[._])s?rg(_|$)")
_BOOTSTRAP_FILE = re.compile(r"bootstrap|^su2_reliable\.py$")
# Orthonormality-bootstrap routines living in modules not named for it.
_BOOTSTRAP_ROUTINES = (
    ("finite_kalgebras/elem_traces.py", "_generate_bootstrap"),
    ('implementations/sl3_su3_traces.py', "SU3ElemTraces._solve_TrT"),
    ('implementations/sl3_su3_traces.py', "SU3ElemTraces._solve_SD"),
)

_CAPS = {"build": 300, "gens_list": 120, "tr1": 300, "gens": 900, "ortho": 900,
         "geometry": 120}
# `--slow` multiplies every cap: it takes every generator pair (the dodecagon's
# 20736 took 684 s of its 900 on a shared machine) and the A1DevenKAlg rows.
_SLOW_CAP_FACTOR = 3


# ---------------------------------------------------------------------------
# helpers run inside the child (repo imports only inside functions)
# ---------------------------------------------------------------------------

def _cone_gens(A):
    """The generator labels of a cone presentation: each multiplicative
    generator of `cone_data()` as the native label of its first power."""
    cd = A.cone_data()
    return [cd.from_cone_label(frozenset({g}), {g: 1})
            for g in sorted(cd.mult_gens(), key=repr)]


def _word_gens(A):
    """As `_cone_gens`, for the classes whose cone data speaks χ-stripped words
    (`A1DnKAlg`, `A1DoddConeKAlg`): the label is `(word, 0)`."""
    cd = A.cone_data()
    return [(cd.from_cone_label(frozenset({g}), {g: 1}), 0)
            for g in sorted(cd.mult_gens(), key=repr)]


def _deven_gens(A):
    """As `_cone_gens`, for `U1A1DevenConeKAlgebra`, whose cone data speaks the
    χ-stripped `(curves, e)`: the label is `(curves, e, 0)`."""
    cd = A.cone_data()
    return [cd.from_cone_label(frozenset({g}), {g: 1}) + (0,)
            for g in sorted(cd.mult_gens(), key=repr)]


def _geo_polygon_curve(A, gens):
    """`A1A2kKAlg.curve(x, ell)`: the diagonals of the (2k+3)-gon, each named
    twice (from either end), are exactly the generators, and ρ is the rotation."""
    H = A.H
    img = {}
    for x in range(H):
        for ell in range(2, H - 1):
            img.setdefault(A.curve(x, ell), set()).add((x, ell))
    assert set(img) == set(gens), "curve() image != generators"
    assert all(len(v) == 2 for v in img.values()), "a diagonal not named twice"
    bad = [(x, ell) for x in range(H) for ell in range(2, H - 1)
           if A.rho(A.curve(x, ell)) != A.curve(x + 1, ell)]
    assert not bad, "rho is not the rotation on %s" % bad[:3]
    return "curve(x, ell): the %d diagonals of the %d-gon; rho = rotation" % (len(img), H)


def _geo_punctured_curve(A, gens):
    """`A1DnKAlg.curve(x, ell)`: the n(n-1) curves of the once-punctured n-gon
    are exactly the generators, and ρ is the rotation."""
    n = A.n
    img = {A.curve(x, ell): (x, ell) for x in range(n) for ell in range(2, n + 1)}
    assert len(img) == n * (n - 1), "curve() not injective"
    assert set(img) == set(gens), "curve() image != generators"
    bad = [(x, ell) for x in range(n) for ell in range(2, n + 1)
           if A.rho(A.curve(x, ell)) != A.curve(x + 1, ell)]
    assert not bad, "rho is not the rotation on %s" % bad[:3]
    return "curve(x, ell): the %d curves of the once-punctured %d-gon; rho = rotation" % (
        len(img), n)


def _geo_deven_curve(A, gens):
    """`U1A1DevenConeKAlgebra.curve(x, ell)` and `geometric_label(letter)`: the
    n(n-1) curves of the once-punctured n-gon (n = 2k+2) are exactly the
    generators other than E^{±1}; `geometric_label` names each curve letter by
    its curve and gives None on E^{±1}; ρ is the rotation up to a power of E."""
    n = A.n
    img = {A.curve(x, ell): (x, ell) for x in range(n) for ell in range(2, n + 1)}
    assert len(img) == n * (n - 1), "curve() not injective"
    torus = {g for g in gens if not g[0]}
    assert len(torus) == 2, "expected the two gauge letters E, E^-1 without a curve"
    assert set(img) == set(gens) - torus, "curve() image != the curve generators"
    cd = A.cone_data()
    named = 0
    for g in cd.mult_gens():
        d = A.geometric_label(g)
        lab = cd.from_cone_label(frozenset({g}), {g: 1}) + (0,)
        if d is None:
            assert lab in torus, (g, lab)
            continue
        assert A.curve(*d) == lab, (g, d, lab)
        named += 1
    assert named == n * (n - 1), "geometric_label does not name every curve letter"
    bad = [(x, ell) for x in range(n) for ell in range(2, n + 1)
           if A.rho(A.curve(x, ell))[0::2] != A.curve(x + 1, ell)[0::2]]
    assert not bad, "rho is not the rotation up to a power of E on %s" % bad[:3]
    return ("curve(x, ell) / geometric_label(letter): the %d curves of the once-punctured "
            "%d-gon (+ E^{±1}); rho = rotation up to a power of E" % (len(img), n))


def _geo_chord(A, gens):
    """`U1A1AoddKAlg.geometric_label(g)` on the letters: each chord letter a
    distinct diagonal of the (2k+4)-gon (all of them), `None` on `E^{±1}`;
    `cone_data().chord(g)` agrees."""
    H = 2 * A.k + 4
    cd = A.cone_data()
    letters = sorted(cd.mult_gens(), key=repr)
    diag = {}
    torus = 0
    for g in letters:
        d = A.geometric_label(g)
        if d is None:
            torus += 1
            continue
        v1, v2 = d
        assert 0 <= v1 < v2 < H and (v2 - v1) % H not in (1, H - 1), (g, d)
        assert tuple(sorted(cd.chord(g))) == d, (g, d)
        diag[d] = g
    assert torus == 2, "expected the two gauge letters E, E^-1 without a curve"
    assert len(diag) == len(letters) - 2 == H * (H - 3) // 2, "not all diagonals"
    return "geometric_label(letter): the %d diagonals of the %d-gon (+ E^{±1})" % (
        len(diag), H)


def _geo_balanced(A, gens):
    """`geometric_label(label)` of an ungauged class: on every generator a
    multiset of diagonals, balanced (as many even-even as odd-odd, with
    multiplicity, the design record), injective."""
    out = set()
    for g in gens:
        curves, e = A.geometric_label(g)
        ee = sum(m for ((v1, v2), m) in curves if v1 % 2 == 0 and v2 % 2 == 0)
        oo = sum(m for ((v1, v2), m) in curves if v1 % 2 == 1 and v2 % 2 == 1)
        assert ee == oo, (g, curves)
        out.add((curves, e))
    assert len(out) == len(gens), "geometric_label not injective on the generators"
    return "geometric_label(label): balanced multisets of diagonals, injective on %d" % len(gens)


def _geo_su3ad_curves(A, gens):
    """`SU3ADKAlg.geometric_label(label)`: on every generator a balanced multiset
    of pairwise non-crossing curves of the once-punctured square (the curves of
    `A1DevenKAlg(1)`, the design record) with the SU(3) weight (0, 0), injective; ρ
    rotates the curves and conjugates the weight (on the generators and on the
    generators times χ_(1,0)).  Crossing: the chord model of the curve frame
    (a curve (x, ℓ) is the chord {x, x+ℓ} of the 8-gon with its half-turn, the
    loop ℓ = 4 one diameter; two curves cross iff two chords interleave)."""
    from u1a1deven_seed_characters import curve_charge

    def chords(c):
        x, l = c
        u, v = x % 8, (x + l) % 8
        if l == 4:
            return [tuple(sorted((u, v)))]
        return [tuple(sorted((u, v))), tuple(sorted(((u + 4) % 8, (v + 4) % 8)))]

    def cross(c, d):
        return any(a < p < b < q or p < a < q < b
                   for a, b in chords(c) for p, q in chords(d))

    out = set()
    for g in gens:
        for flav in ((0, 0), (1, 0)):
            lab = tuple(g[:3]) + flav
            curves, w = A.geometric_label(lab)
            assert w == flav, (lab, w)
            assert all(m >= 1 and 2 <= c[1] <= 4 and 0 <= c[0] < 4
                       for c, m in curves), (lab, curves)
            assert sum(m * curve_charge(c, 1) for c, m in curves) == 0, (lab, curves)
            assert not any(cross(c, d) for (c, _m) in curves for (d, _n) in curves
                           if c < d), (lab, curves)
            r = A.geometric_label(A.rho(lab))
            rot = tuple(sorted((((x + 1) % 4, l), m) for (x, l), m in curves))
            assert r == (rot, (flav[1], flav[0])), (lab, r)
            if flav == (0, 0):
                out.add(curves)
    assert len(out) == len(gens), "geometric_label not injective on the generators"
    return ("geometric_label(label): balanced non-crossing multisets of curves of the "
            "once-punctured square (+ the SU(3) weight), injective on %d; rho = rotation, "
            "weight conjugated" % len(gens))


def _geo_total_injective(A, gens):
    """`geometric_label(label)` is defined on every generator and injective (no
    claim about the shape of the labels)."""
    out = {A.geometric_label(g) for g in gens}
    assert len(out) == len(gens), "geometric_label not injective on the generators"
    return "geometric_label(label): total and injective on %d generators" % len(gens)


def _geo_polygon_sample(A, gens):
    """`curve(x, ell)` as `_geo_polygon_curve` checks it, and
    `geometric_label(label)` on each: the one diagonal `curve(x, ell)` names,
    with power 1 (`PentagonKAlg`, `HeptagonKAlg`, on their own labels)."""
    out = _geo_polygon_curve(A, gens)
    H = A.H
    bad = [(x, ell) for x in range(H) for ell in range(2, H - 1)
           if A.geometric_label(A.curve(x, ell))
           != ((tuple(sorted((x % H, (x + ell) % H))), 1),)]
    assert not bad, "geometric_label(curve(x, ell)) is not that diagonal at %s" % bad[:3]
    return out + "; geometric_label(label) the multiset of diagonals"


def _geo_hexagon_letters(A, gens):
    """`U1HexagonKAlg.geometric_label(g)` on the letters of its generators (not
    through `cone_data()`, its frozen cross-check surface): the nine diagonals
    of the hexagon, each once, and `None` on `E^{±1}` (`(3, 0)`, `(3, 1)`); two
    letters q-commute — their product is one term — iff their diagonals do not
    cross; ρ rotates each letter's diagonal by one (up to a power of `E`)."""
    chords = [g for g in gens if g[0]]
    letter = {g: g[0][0][:2] for g in chords}
    geo = {g: A.geometric_label(letter[g]) for g in chords}
    assert A.geometric_label((3, 0)) is None and A.geometric_label((3, 1)) is None, \
        "the gauge letters E, E^-1 carry a diagonal"
    assert len(chords) == 9 and None not in geo.values() and len(set(geo.values())) == 9, \
        "not the nine diagonals, each once"

    def cross(c, d):
        (a, b), (x, y) = c, d
        return len({a, b, x, y}) == 4 and ((a < x < b) != (a < y < b))

    bad = [(letter[g], letter[h]) for g in chords for h in chords if g < h
           and (len(A.multiply(g, h).terms) == 1) == cross(geo[g], geo[h])]
    assert not bad, "q-commutation is not non-crossing on %s" % bad[:3]
    for g in chords:
        (factor,), _e = A.rho(g)
        d = geo[g]
        want = tuple(sorted(((d[0] + 1) % 6, (d[1] + 1) % 6)))
        assert A.geometric_label(factor[:2]) == want, (letter[g], factor, want)
    return ("geometric_label(letter): the 9 diagonals of the hexagon (+ E^{±1}); "
            "q-commuting = non-crossing; rho = rotation up to a power of E")


def _geo_dodd_curves(A, gens):
    """`A1DoddConeKAlg.geometric_label(label)`: on the generators exactly the
    labels `A1DnKAlg(2k+3).curve(x, ell)` of the n(n−1) curves of the
    once-punctured n-gon, each once, and ρ is the rotation on them."""
    from a1dn_kalg import A1DnKAlg
    n = 2 * A.k + 3
    D = A1DnKAlg(n)
    out = [A.geometric_label(g) for g in gens]
    assert len(set(out)) == len(gens), "geometric_label not injective on the generators"
    assert set(out) == {D.curve(x, ell) for x in range(n) for ell in range(2, n + 1)}, \
        "geometric_label image != the curves of A1DnKAlg(%d)" % n
    for g, (curves, kappa) in zip(gens, out):
        rot = tuple(sorted((((x + 1) % n, ell), m) for (x, ell), m in curves))
        assert A.geometric_label(A.rho(g)) == (rot, kappa), g
    return ("geometric_label(label): the %d curves of the once-punctured %d-gon, "
            "= A1DnKAlg(%d).curve; rho = rotation" % (len(out), n, n))


def _pair_sample(gens, cap):
    """All ordered generator pairs if there are at most `cap` generators, else
    all pairs among every s-th generator (s = ceil(n/cap)) plus every diagonal
    pair; returns (pairs, description)."""
    n = len(gens)
    if cap is None or n <= cap:
        return [(a, b) for a in gens for b in gens], "all %d ordered pairs" % (n * n)
    s = -(-n // cap)
    S = gens[::s]
    pairs = [(a, b) for a in S for b in S] + [(g, g) for g in gens if g not in S]
    return pairs, ("%d pairs: all ordered pairs of every %d-th generator (%d) and "
                   "the %d other diagonal pairs" % (len(pairs), s, len(S), n - len(S)))


def _delta(A, series, expected):
    """`series = expected·1 + O(𝖖)` in the convention of
    `KAlgebra.verify_orthonormality`: (ok, detail)."""
    R = A.coefficient_ring()
    neg = [e for e, c in series.coeffs.items() if e < 0 and not c.is_zero()]
    if neg:
        return False, "nonzero at q^%d" % min(neg)
    got = series[0].terms.get(R.one_basis(), 0)
    return got == expected, "chi0[q^0] = %s (want %s)" % (got, expected)


def _pairing(A, a, b, K, element_form):
    if element_form:
        from kalgebra import Element
        x = A.multiply_elements(A.rho_element(Element.basis(a)), Element.basis(b))
        return A.trace_element(x, K)
    return A.inner_product(a, b, K)


# ---------------------------------------------------------------------------
# the rows
# ---------------------------------------------------------------------------

class Row:
    """One serving class.  `build()` constructs it; `gens(A)` lists its
    multiplicative generators; `geometry(A, gens)` checks the accessor (None:
    the class has none); `stored` = the declared stored-data modules
    (repo-relative); `data_files` = the declared stored-data FILES the row
    reads (repo-relative) — product or ρ tables, never trace data; `k1`, `kg`
    = the Tr(1) and generator windows; `pair_cap` = the generator-pair sample
    of the default run and `slow_pair_cap` that of `--slow` (None: all
    pairs)."""

    def __init__(self, name, theory, family, module, build, gens, *, geometry=None,
                 accessor=None, element_form=False, k1=40, kg=12, pair_cap=30,
                 slow_pair_cap=None, stored=(), data_files=(), slow=False, caps=None,
                 route="", window_note=""):
        self.name, self.theory, self.family, self.module = name, theory, family, module
        self.build, self.gens, self.geometry, self.accessor = build, gens, geometry, accessor
        self.element_form, self.k1, self.kg = element_form, k1, kg
        self.pair_cap, self.slow_pair_cap = pair_cap, slow_pair_cap
        self.stored, self.slow = tuple(stored), slow
        self.data_files = tuple(data_files)
        self.caps = dict(_CAPS, **(caps or {}))
        self.route, self.window_note = route, window_note


def _b(module, expr):
    """A builder: `from module import *` names, then evaluate `expr`."""
    def build():
        mod = __import__(module)
        return eval(expr, dict(vars(mod)))
    build.__name__ = "build_" + module
    return build


def _zoo(sid):
    def build():
        import finite_kalgebras as fk
        return fk.FINITE_KALGEBRAS[sid]()
    return build


def _mult_generators(A):
    return list(A.mult_generators())


def _u1hexagon_gens(A):
    """The 11 multiplicative generators of `U1HexagonKAlg` (its docstring): the
    six short and three long chords and μ^{±1}.  Not through `cone_data()`,
    which is its frozen cross-check surface."""
    return ([A.L((1, i)) for i in range(6)] + [A.L((2, i)) for i in range(3)]
            + [A.mu, ((), -1)])


def _su3ad_gens(A):
    return [A.T(i) for i in range(4)] + [A.D(i) for i in range(4)]


_ZOO_FLAVOUR = {"pentagon": "trivial", "heptagon": "trivial", "a3": "u1", "hexagon": "u1",
                "a5": "u1", "octagon": "u1", "a7": "u1", "decagon": "u1",
                "a1d3": "su2", "a1d4": "su2u1", "a1d5": "su2", "a1d6": "su2u1",
                "a1d7": "su2", "a1d8": "su2u1", "e6": "trivial", "e7": "u1", "e8": "trivial"}
_ZOO_MODULE = {"pentagon": "finite_pentagon_kalg", "heptagon": "finite_heptagon_kalg",
               "a3": "finite_a3_kalg", "hexagon": "finite_a3_kalg",
               "a5": "finite_a5_kalg", "octagon": "finite_a5_kalg",
               "a7": "finite_a7_kalg", "decagon": "finite_a7_kalg",
               "a1d3": "finite_a1d3_kalg", "a1d4": "finite_a1d4_kalg",
               "a1d5": "finite_a1d5_kalg", "a1d6": "finite_a1d6_kalg",
               "a1d7": "finite_a1d7_kalg", "a1d8": "finite_a1d8_kalg",
               "e6": "finite_e6_kalg", "e7": "finite_e7_kalg", "e8": "finite_e8_kalg"}
_ZOO_THEORY = {"pentagon": "[A1,A2]", "heptagon": "[A1,A4]", "a3": "[A1,A3]",
               "hexagon": "[A1,A3]", "a5": "[A1,A5]", "octagon": "[A1,A5]",
               "a7": "[A1,A7]", "decagon": "[A1,A7]", "a1d3": "[A1,D3]",
               "a1d4": "[A1,D4]", "a1d5": "[A1,D5]", "a1d6": "[A1,D6]",
               "a1d7": "[A1,D7]", "a1d8": "[A1,D8]", "e6": "[A1,E6]",
               "e7": "[A1,E7]", "e8": "[A1,E8]"}


_ZOO_POLYGON = {"pentagon": 5, "heptagon": 7, "a3": 6, "hexagon": 6, "a5": 8,
                "octagon": 8, "a7": 10, "decagon": 10, "a1d3": 3, "a1d4": 4, "a1d5": 5,
                "a1d6": 6, "a1d7": 7, "a1d8": 8}


def _geo_zoo(sid):
    """The check of a zoo `A` / `D` entry's `geometric_label(label)`
    (`zoo_geometry`): on every generator a non-empty multiset
    of pairwise non-crossing curves — diagonals `(v1, v2)` of the polygon for
    `A`, curves `(x, ℓ)` of the once-punctured polygon for `D` — injective, and
    the zoo's ρ rotates it by one.  How many generators ρ⁻¹'s rotation would
    fit is reported (none, measured 2026-09-24: the check tells the two
    directions apart)."""
    n = _ZOO_POLYGON[sid]
    punctured = sid.startswith("a1d")

    def check(A, gens):
        if punctured:
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
        seen, backward = set(), 0
        for g in gens:
            cv = A.geometric_label(g)
            assert cv and all(m >= 1 for _c, m in cv), (g, cv)
            assert not any(cross(c, d) for c, _m in cv for d, _k in cv if c < d), (g, cv)
            r = A.geometric_label(A.rho(g))
            assert r == rot(cv, 1), "rho is not the rotation on %r: %r -> %r" % (g, cv, r)
            backward += r == rot(cv, -1)
            seen.add(cv)
        assert len(seen) == len(gens), "geometric_label not injective on the generators"
        return ("geometric_label(label): non-crossing multisets of curves of the %s%d-gon, "
                "injective on %d; rho = rotation by one (rho^-1's on %d)"
                % ("once-punctured " if punctured else "", n, len(gens), backward))
    return check


def _zoo_row(sid, **kw):
    fam = "E" if sid.startswith("e") else ("D" if sid.startswith("a1d") else "A")
    name = {"e6": "FiniteE6KAlgebra (zoo e6)", "e7": "FiniteE7KAlgebra (zoo e7)",
            "e8": "FiniteE8KAlgebra (zoo e8)"}.get(sid, "zoo " + sid)
    kw.setdefault("element_form", _ZOO_FLAVOUR[sid] in ("u1", "su2u1"))
    kw.setdefault("stored", ('implementations/%s.py' % _ZOO_MODULE[sid],))
    if fam in ("A", "D"):
        kw.setdefault("geometry", _geo_zoo(sid))
        kw.setdefault("accessor", "geometric_label(label): the family class's curves "
                                  "(finite_kalgebras.zoo_geometry)")
    return Row(name, _ZOO_THEORY[sid], fam, 'implementations/%s.py' % _ZOO_MODULE[sid],
               _zoo(sid), _cone_gens, **kw)


# The even-D rows (2026-09-24): every trace of U1A1DevenConeKAlgebra is a
# closed form or the Layer-1 reduction onto the seeds' closed forms
# (u1a1deven_seed_characters), so A1DevenKAlg's pairings — traces of products —
# no longer run the RG transport (before: `ortho` 546 s / 1447 s for
# A1DevenKAlg(2) / (3) under the profile hook, behind --slow).  The zoo's a1d6 /
# a1d8 reach their products by their own Layer 1 over the seeds: 10-21 s /
# 101-131 s in all over two runs (a1d8's `tr1` includes the generator map's
# discovery, 93 s under the hook).  Measured 2026-09-24 under the hook, shared
# machine: the six rows U1A1DevenConeKAlgebra(1..3), A1DevenKAlg(1..3) in 94 s
# (A1DevenKAlg(3)'s 990 pairs 84 s), and with --slow in 188 s (every pair but
# A1DevenKAlg(3)'s, whose 60-generator sample, 3660 pairs, takes 176 s; at the
# default sample's rate all 14400 would take about 20 minutes — not run).
_DEVEN_CAPS = {"build": 900, "tr1": 900, "ortho": 3600}

ROWS = [
    # [A1,A2k]
    *[Row("A1A2kKAlg(%d)" % k, "[A1,A%d]" % (2 * k), "A", 'implementations/a1a2k_kalg.py',
          _b("a1a2k_kalg", "A1A2kKAlg(%d)" % k), _cone_gens, geometry=_geo_polygon_sample,
          accessor="curve(x, ell), geometric_label(label)",
          route="Layer 1 + Andrews-Gordon M(2,2k+3) characters")
      for k in (1, 2, 3)],
    Row("PentagonKAlg", "[A1,A2]", "A", "kalgebra_samples.py",
        _b("kalgebra_samples", "PentagonKAlg()"), _cone_gens, geometry=_geo_polygon_sample,
        accessor="curve(x, ell), geometric_label(label)",
        route="Layer 1 + Rogers-Ramanujan closed forms"),
    Row("HeptagonKAlg", "[A1,A4]", "A", "kalgebra_samples.py",
        _b("kalgebra_samples", "HeptagonKAlg()"), _cone_gens, geometry=_geo_polygon_sample,
        accessor="curve(x, ell), geometric_label(label)",
        route="through A1A2kKAlg(2), relabelled"),
    # [A1,A2k+1], gauged
    *[Row("U1A1AoddKAlg(%d)" % k, "u(1)-gauged [A1,A%d]" % (2 * k + 1), "A",
          'implementations/u1a1aodd_kalg.py', _b("u1a1aodd_kalg", "U1A1AoddKAlg(%d)" % k),
          _cone_gens, geometry=_geo_chord, accessor="geometric_label(letter), cone_data().chord",
          route="Layer 1 + u1_pgon_layer2 closed forms")
      for k in (1, 2, 3)],
    Row("U1HexagonKAlg", "u(1)-gauged [A1,A3]", "A", 'implementations/u1_hexagon_kalg.py',
        _b("u1_hexagon_kalg", "U1HexagonKAlg()"), _u1hexagon_gens,
        geometry=_geo_hexagon_letters, accessor="geometric_label(letter)",
        route="through U1A1AoddKAlg(1), (F, e) -> (F, -e)"),
    # [A1,A2k+1], ungauged
    *[Row("ungauge_u1a1aodd(%d)" % k, "[A1,A%d]" % (2 * k + 1), "A",
          'implementations/ungauge_kalgebra.py', _b("ungauge_kalgebra", "ungauge_u1a1aodd(%d)" % k),
          _mult_generators, geometry=_geo_balanced, accessor="geometric_label(label)",
          route="ungauging of U1A1AoddKAlg(k)")
      for k in (1, 2, 3)],
    Row("HexagonKAlg", "[A1,A3]", "A", 'implementations/hexagon_kalg.py',
        _b("hexagon_kalg", "HexagonKAlg()"), _mult_generators, geometry=_geo_balanced,
        accessor="geometric_label(label)", route="ungauging of U1HexagonKAlg"),
    Row("OctagonKAlg", "[A1,A5]", "A", 'implementations/octagon_kalg.py',
        _b("octagon_kalg", "OctagonKAlg()"), _mult_generators, geometry=_geo_balanced,
        accessor="geometric_label(label)", route="ungauging of U1A1AoddKAlg(2)"),
    Row("DecagonKAlg", "[A1,A7]", "A", 'implementations/decagon_kalg.py',
        _b("decagon_kalg", "DecagonKAlg()"), _mult_generators, geometry=_geo_balanced,
        accessor="geometric_label(label)", route="ungauging of U1A1AoddKAlg(3)"),
    Row("DodecagonKAlg", "[A1,A9]", "A", 'implementations/dodecagon_kalg.py',
        _b("dodecagon_kalg", "DodecagonKAlg()"), _mult_generators, geometry=_geo_balanced,
        accessor="geometric_label(label)", route="ungauging of U1A1AoddKAlg(4)"),
    # [A1,D2k+3]
    *[Row("A1DnKAlg(%d)" % v, "[A1,D%d]" % v, "D", 'implementations/a1dn_kalg.py',
          _b("a1dn_kalg", "A1DnKAlg(%d)" % v), _word_gens,
          geometry=_geo_punctured_curve, accessor="curve(x, ell, kappa)",
          route="through A1DoddConeKAlg((n-3)/2)")
      for v in (3, 5, 7)],
    *[Row("A1DoddConeKAlg(%d)" % k, "[A1,D%d]" % (2 * k + 3), "D",
          'implementations/a1dodd_kalg.py', _b("a1dodd_kalg", "A1DoddConeKAlg(%d)" % k),
          _word_gens, geometry=_geo_dodd_curves,
          accessor="geometric_label(label): A1DnKAlg(2k+3)'s (curves, kappa)",
          route="Layer 1 + a1dodd_layer2 closed forms")
      for k in (0, 1, 2)],
    # [A1,D2k+2]
    *[Row("U1A1DevenConeKAlgebra(%d)" % k, "u(1)-gauged [A1,D%d]" % (2 * k + 2), "D",
          'implementations/u1a1deven_cone_kalgebra.py',
          _b("u1a1deven_cone_kalgebra", "U1A1DevenConeKAlgebra(%d)" % k), _deven_gens,
          geometry=_geo_deven_curve,
          accessor="curve(x, ell, e, kappa), geometric_label(letter)",
          caps=_DEVEN_CAPS,
          route="gauge sector and seeds from closed forms; every other label by "
                "Layer 1 onto the seeds")
      for k in (1, 2, 3)],
    *[Row("A1DevenKAlg(%d)" % k, "[A1,D%d]" % (2 * k + 2), "D", 'implementations/a1deven_kalg.py',
          _b("a1deven_kalg", "A1DevenKAlg(%d)" % k), _mult_generators,
          slow_pair_cap=60, caps=_DEVEN_CAPS,
          geometry=_geo_total_injective,
          accessor="geometric_label(label), from UngaugedKAlgebra over the gauged "
                   "class's letter hook",
          route="ungauging of U1A1DevenConeKAlgebra(k): generators from the seeds' "
                "closed forms, products by Layer 1 onto them")
      for k in (1, 2, 3)],
    # [A1,D4]
    Row("SU3ADKAlg", "[A1,D4]", "D", 'implementations/su3_ad_kalg.py',
        _b("su3_ad_kalg", "SU3ADKAlg()"), _su3ad_gens, geometry=_geo_su3ad_curves,
        accessor="geometric_label(label): the curves of A1DevenKAlg(1) + the SU(3) weight",
        route="Layer 1; Tr(1) and the T/D seeds from the even-D k = 1 closed forms "
              "(u1a1deven_seed_characters: Creutzig's gauge tower, the seeds through "
              "the curve map), summed over the gauge charge"),
    # the zoo, every id (e6 / e7 / e8 are the E-type classes)
    *[_zoo_row(sid) for sid in ("pentagon", "heptagon", "a3", "hexagon", "a5", "octagon",
                                "a7", "decagon", "a1d3", "a1d4", "a1d5", "a1d7", "e6", "e8")],
    # a1d6 / a1d8 are served through A1DevenKAlg(2) / A1DevenKAlg(3)
    # (a1deven_seeds), whose generator traces are the seeds'
    # closed forms: default rows at the standard windows since 2026-09-24.
    # --slow takes 60 generators' pairs, not all: every ordered pair at a1d6
    # (1521), 3660 of the 14400 at a1d8 (all of them never finished while the
    # seeds ran the transport).
    *[_zoo_row(sid, pair_cap=30, slow_pair_cap=60, caps=_DEVEN_CAPS,
               route="through A1DevenKAlg(%d) (finite_kalgebras.a1deven_seeds)" % k)
      for sid, k in (("a1d6", 2), ("a1d8", 3))],
    _zoo_row("e7", stored=('implementations/finite_e7_kalg.py', "finite_kalgebras/e7_seeds.py")),
    # E-type, gauged.  The two pickles hold the cone and rho TABLES (product data,
    # built once on the U1E7GaugedRG flow); the traces are closed forms.  Stored
    # product tables are declared like the zoo's literal tables (a decision of
    # 2026-09-24, the design record; `frozen` is about trace data).
    Row("U1E7ConeKAlgebra", "u(1)-gauged [A1,E7]", "E", 'implementations/u1e7_cone_kalgebra.py',
        _b("u1e7_cone_kalgebra", "U1E7ConeKAlgebra()"), _cone_gens,
        stored=('implementations/u1e7_cone_kalgebra.py', 'implementations/finite_e7_kalg.py',
                "finite_kalgebras/e7_seeds.py"),
        data_files=('implementations/u1e7_cone_tables.pkl',
                    'implementations/u1e7_rho_tables.pkl'),
        route="neutral sector through FiniteE7KAlgebra"),
]

ROWS_BY_NAME = {r.name: r for r in ROWS}
assert len(ROWS_BY_NAME) == len(ROWS), "duplicate row names"


# ---------------------------------------------------------------------------
# expected failures: (rows, checks, reason).  Exactly these fail; a listed
# check that passes must be removed from its entry.
# ---------------------------------------------------------------------------

_EXPECTED = [
    # Empty since 2026-09-24: every A and D row passes `geometry` — the zoo's
    # entries too, through zoo_geometry.
    # An orthonormality bootstrap on a serving path would be RECORDED here, not
    # failed: seeded by the class's own closed-form Tr(1) and the orthonormality
    # axiom, exact to any order, it counts as self-contained as the user
    # worded it ("self-contained realizations which are fully functional
    # (arbitrary in principle precision and coverage)"); decision of 2026-09-24,
    # listed for the author's veto.  Closed forms, where cheap, remain the preferred
    # route, and no row is served by one since 2026-09-24: zoo pentagon / heptagon
    # through the closed-form A1A2kKAlg(1) / A1A2kKAlg(2)
    # (aeven_seeds), and SU3ADKAlg (with zoo a1d4, served through
    # it) through the even-D k = 1 closed forms (sl3_su3_traces._ClosedFormSeeds);
    # the forward pass SU3ElemTraces is their witness.
]

EXPECTED_FAILURES = {(row, check): reason for rows, checks, reason in _EXPECTED
                     for row in rows for check in checks}


# ---------------------------------------------------------------------------
# controls: snippets run in a fresh child under the same instruments; each
# must make the named checks fire (the clean snippet: none), or return the
# named value (the δ-check control)
# ---------------------------------------------------------------------------

_FIXTURE_MODULE = "ade_guard_dense_table"
_FIXTURE_PICKLE = "ade_guard_table.pkl"


def _write_fixtures(d):
    """A module holding a dense numeric table (200 rows of 8 integers) and a
    pickle of a trace-like table, in the temporary directory `d`."""
    import pickle
    rows = ["    (%s)," % ", ".join(str((7 * i + 3 * j) % 97 - 48) for j in range(8))
            for i in range(200)]
    with open(os.path.join(d, _FIXTURE_MODULE + ".py"), "w") as f:
        f.write('"""Synthetic frozen table (test fixture)."""\nTABLE = (\n%s\n)\n'
                % "\n".join(rows))
    with open(os.path.join(d, _FIXTURE_PICKLE), "wb") as f:
        pickle.dump({(i, 1): {q: (-1) ** q * (q + i) for q in range(13)}
                     for i in range(8)}, f)


def _ctl_dense_module():
    sys.path.insert(0, os.environ["ADE_GUARD_FIXTURES"])
    mod = __import__(_FIXTURE_MODULE)
    return mod.TABLE[7][3]


def _ctl_pickle():
    import pickle
    path = os.path.join(os.environ["ADE_GUARD_FIXTURES"], _FIXTURE_PICKLE)
    with open(path, "rb") as f:
        return pickle.load(f)[(3, 1)][5]


def _ctl_trace_table_import():
    import elem_trace_data  # noqa: F401


def _ctl_bps_pentagon():
    from bps_kalgebra import BPSKAlgebra
    A = BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=[(1, 0), (0, 1)])
    return A.trace(A.identity(), K=4)


def _ctl_bootstrap_import():
    import u1_bootstrap  # noqa: F401


def _ctl_rg_import():
    import rgkalgebra  # noqa: F401


def _ctl_generate_bootstrap():
    from elem_traces import _generate_bootstrap
    return _generate_bootstrap("pentagon", 4)


def _ctl_su3_elem_traces():
    # the forward pass directly: it left the serving path (`_provider`) on
    # 2026-09-24, and stays the witness of the closed forms
    from sl3_su3_traces import SU3ElemTraces
    return SU3ElemTraces().ensure(4)


def _ctl_delta_check():
    """The δ + O(𝖖) check is not vacuous: on zoo `a3` the LABEL form of the
    pairing fails on 3 of the 6 generators (the audit: `I[𝖖⁰] = μ` where
    ρ's μ^δ sits on `rho_element`), the element form on none; and `Tr(1)` read
    as a generator's trace (`δ = 0`) fails."""
    import finite_kalgebras as fk
    A = fk.FINITE_KALGEBRAS["a3"]()
    gens = _cone_gens(A)
    label = sum(not _delta(A, _pairing(A, g, g, 2, False), 1)[0] for g in gens)
    elem = sum(not _delta(A, _pairing(A, g, g, 2, True), 1)[0] for g in gens)
    return (label, elem, _delta(A, A.trace(A.identity(), 4), 0)[0])


def _ctl_clean():
    from laurent_poly import LaurentPoly
    return LaurentPoly.q(2) * LaurentPoly.q(-1)


# (name, snippet, {check that must fail: a string its detail must contain} --
#  None: the instruments are not the point --, the snippet's required value)
CONTROLS = [
    ("synthetic dense table read", _ctl_dense_module, {"stored": _FIXTURE_MODULE}, None),
    ("synthetic pickle read", _ctl_pickle, {"frozen": _FIXTURE_PICKLE}, None),
    ("frozen trace-table import", _ctl_trace_table_import, {"frozen": "elem_trace_data"},
     None),
    ("BPSKAlgebra pentagon trace", _ctl_bps_pentagon, {"bps": "bps_kalgebra.py"}, None),
    ("bootstrap module import", _ctl_bootstrap_import, {"imports": "u1_bootstrap"}, None),
    ("RG module import", _ctl_rg_import, {"imports": "rgkalgebra"}, None),
    ("elem_traces._generate_bootstrap run", _ctl_generate_bootstrap,
     {"bootstrap": "_generate_bootstrap"}, None),
    ("SU3ElemTraces seed solves run", _ctl_su3_elem_traces, {"bootstrap": "SU3ElemTraces"},
     None),
    ("delta check fails where it should (zoo a3)", _ctl_delta_check, None, "(3, 0, False)"),
    ("clean snippet", _ctl_clean, {}, None),
]
CONTROLS_BY_NAME = {c[0]: c for c in CONTROLS}


# ---------------------------------------------------------------------------
# the child side
# ---------------------------------------------------------------------------

class _PhaseTimeout(BaseException):
    """Raised by the phase alarm; a BaseException, so that a repo
    `except Exception` cannot swallow it."""


def _phase(res, name, cap, fn, FrozenRead):
    """Run `fn()` under an alarm of `cap` seconds; (ran to completion, value)."""
    import signal

    def _alarm(*_a):
        raise _PhaseTimeout()
    signal.signal(signal.SIGALRM, _alarm)
    t0 = time.time()
    signal.alarm(cap)
    try:
        return True, fn()
    except FrozenRead as ex:
        res["frozen_trace_read"] = str(ex)
        res["errors"][name] = "FrozenRead: %s" % ex
    except _PhaseTimeout:
        res["errors"][name] = "TIMEOUT after %d s" % cap
    except Exception as ex:
        res["errors"][name] = "%s: %s" % (type(ex).__name__, str(ex)[:400])
    finally:
        signal.alarm(0)
        res["seconds"][name] = round(time.time() - t0, 2)
    return False, None


_GEOMETRY_NAMES = ("curve", "geometric_label", "chord")


def _run_row(res, row, FrozenRead, slow):
    checks, info = res["checks"], res["info"]
    caps = {k: v * (_SLOW_CAP_FACTOR if slow else 1) for k, v in row.caps.items()}

    def record(check, phase, ok, out):
        checks[check] = list(out) if ok else [False, "not completed: %s" % res["errors"][phase]]

    ok, A = _phase(res, "build", caps["build"], row.build, FrozenRead)
    if not ok:
        return
    ok, gens = _phase(res, "gens_list", caps["gens_list"], lambda: list(row.gens(A)),
                      FrozenRead)
    if not ok:
        return
    info["n_gens"] = len(gens)
    if not gens:
        res["errors"]["gens_list"] = "no multiplicative generators listed"
        return

    def tr1():
        T = A.trace(A.identity(), row.k1)
        if T.K < row.k1:
            return False, "returned window q^%s < q^%d" % (T.K, row.k1)
        good, d = _delta(A, T, 1)
        nz = sorted(e for e, c in T.coeffs.items() if not c.is_zero())
        info["tr1_orders"] = [len(nz), nz[-1] if nz else None]
        return good, "through q^%d (%d nonzero orders, top q^%s): %s" % (
            row.k1, len(nz), nz[-1] if nz else None, d)
    ok, out = _phase(res, "tr1", caps["tr1"], tr1, FrozenRead)
    record("tr1", "tr1", ok, out)

    def gtr():
        bad, nonzero = [], 0
        for g in gens:
            t = A.trace(g, row.kg)
            if t.K < row.kg:
                bad.append("%r: window q^%s" % (g, t.K))
                continue
            nonzero += any(not c.is_zero() for c in t.coeffs.values())
            good, d = _delta(A, t, 0)
            if not good:
                bad.append("%r: %s" % (g, d))
        info["gens_nonzero"] = nonzero
        return not bad, ("%d generators through q^%d, each O(q) (%d nonzero)"
                         % (len(gens), row.kg, nonzero)
                         + ("; %d failing, e.g. %s" % (len(bad), bad[:2]) if bad else ""))
    ok, out = _phase(res, "gens", caps["gens"], gtr, FrozenRead)
    record("gens", "gens", ok, out)

    pairs, desc = _pair_sample(gens, row.slow_pair_cap if slow else row.pair_cap)
    info["pairs"] = desc

    def ortho():
        bad = []
        for a, b in pairs:
            good, d = _delta(A, _pairing(A, a, b, 2, row.element_form), 1 if a == b else 0)
            if not good:
                bad.append("(%r, %r): %s" % (a, b, d))
        form = "element form" if row.element_form else "inner_product"
        return not bad, ("%s, %s" % (desc, form)
                         + ("; %d failing, e.g. %s" % (len(bad), bad[:2]) if bad else ""))
    ok, out = _phase(res, "ortho", caps["ortho"], ortho, FrozenRead)
    record("ortho", "ortho", ok, out)

    if row.family in ("A", "D"):
        if row.geometry is not None:
            ok, out = _phase(res, "geometry", caps["geometry"], lambda: row.geometry(A, gens),
                             FrozenRead)
            checks["geometry"] = ([True, out] if ok else
                                  [False, "failed: %s" % res["errors"]["geometry"]])
        else:
            have = [n for n in _GEOMETRY_NAMES if callable(getattr(A, n, None))]
            checks["geometry"] = ([False, "UNDECLARED ACCESSOR %s: the row declares no "
                                   "geometry check; add one" % have] if have else
                                  [False, "no geometric-label accessor on this class"])


def child_run(kind, name, prof, FrozenRead, slow=False):
    """The child's work (called after the instruments are installed)."""
    res = {"kind": kind, "name": name, "seconds": {}, "errors": {}, "checks": {}, "info": {}}
    sys.setprofile(prof)
    try:
        if kind == "control":
            ok, value = _phase(res, "control", 600, CONTROLS_BY_NAME[name][1], FrozenRead)
            if ok:
                res["value"] = repr(value)
        else:
            _run_row(res, ROWS_BY_NAME[name], FrozenRead, slow)
    finally:
        sys.setprofile(None)
    return res


_CHILD_TAIL = r'''
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("test_ade_serving_guard", %(this)r)
G = _ilu.module_from_spec(_spec)
sys.modules["test_ade_serving_guard"] = G
_spec.loader.exec_module(G)
preloaded = set(sys.modules)
WATCH = %(watch)r
out = G.child_run(%(kind)r, %(name)r, prof, FrozenRead, %(slow)r)
def inscope(p):
    return bool(p) and (p.startswith(REPO + os.sep) or any(p.startswith(w + os.sep) for w in WATCH))
out["loaded"] = sorted({os.path.abspath(m.__file__) for k, m in list(sys.modules.items())
                        if k not in preloaded and getattr(m, "__file__", None)
                        and inscope(os.path.abspath(m.__file__))})
out["loaded_names"] = sorted(k for k in list(sys.modules) if k not in preloaded)
out["executed"] = sorted(p for p in ran if inscope(p))
out["calls"] = sorted([p, q] for (p, q) in calls if inscope(p))
out["opened"] = sorted(opened)
out["guard_hits"] = list(guard_hits)
print("RESULT " + json.dumps(out, default=repr), flush=True)
'''


# ---------------------------------------------------------------------------
# the parent side
# ---------------------------------------------------------------------------

_SYSTEM_PREFIXES = tuple(sorted({os.path.abspath(p) + os.sep for p in
                                 (sys.prefix, sys.base_prefix, sys.exec_prefix)}))
_ZOO_STANDALONES = {os.path.join(_ROOT, "implementations", m + ".py")
                    for m in set(_ZOO_MODULE.values())}


def _rel(p):
    return os.path.relpath(p, _ROOT) if p.startswith(_ROOT + os.sep) else p


def _is_data_file(p):
    if p.endswith((".py", ".pyc")) or p.startswith(_SYSTEM_PREFIXES):
        return False
    return not p.startswith(("/proc/", "/dev/", "/sys/", "/etc/"))


def run_child(kind, name, fixtures, timeout, slow=False):
    code = (AUDIT.INSTRUMENTS % {"repo": _ROOT}
            + _CHILD_TAIL % {"this": _THIS, "kind": kind, "name": name, "slow": slow,
                             "watch": [fixtures] if fixtures else []})
    env = dict(os.environ, ADE_GUARD_FIXTURES=fixtures or "")
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=timeout, cwd=_ROOT, env=env)
    except subprocess.TimeoutExpired:
        return {"name": name, "child_error": "child timed out after %d s" % timeout,
                "wall": round(time.time() - t0, 1)}
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if not lines:
        return {"name": name, "child_error": "no RESULT (exit %s): %s" % (
            p.returncode, p.stderr[-800:]), "wall": round(time.time() - t0, 1)}
    r = json.loads(lines[-1][len("RESULT "):])
    r["wall"] = round(time.time() - t0, 1)
    return r


def instrument_verdicts(r, stored, data_files=()):
    """The instrument checks of one child record: {check: [ok, detail]}.
    `stored` / `data_files`: the row's declared stored-data modules and files
    (a declared file read is not `frozen`; a declared one never read fails
    `stored`, as a declared module off the path does)."""
    v = {}
    executed = set(r["executed"])
    # A zoo module preloaded by `finite_kalgebras/__init__` is on the path only if
    # a method of one of its classes runs: its import-time code (module-level and
    # class-body comprehensions, table builders) also executes functions.
    used_zoo = {f for f, q in r["calls"] if f in _ZOO_STANDALONES and "." in q
                and not q.rsplit(".", 1)[1].startswith("<")}
    scan = {f for f in set(r["loaded"]) | executed
            if f not in _ZOO_STANDALONES or f in used_zoo}
    dense = {}
    for f in sorted(scan):
        n, lines = AUDIT.literal_density(f)
        if n >= AUDIT.MIN_LITERALS and n / max(lines, 1) >= AUDIT.DENSITY_FROZEN:
            dense[_rel(f)] = "%s (%.1f/line, %d literals)" % (_rel(f), n / lines, n)
    opened = sorted({_rel(p) for p in r["opened"] if _is_data_file(p)})
    data = [p for p in opened if p not in data_files]
    declared_read = [p for p in opened if p in data_files]
    undeclared = sorted(set(dense) - set(stored))
    missing = sorted(set(stored) - set(dense))
    missing_files = sorted(set(data_files) - set(opened))
    trouble = []
    if r.get("frozen_trace_read") or r.get("guard_hits"):
        trouble.append("frozen trace-table import: %s" % (r.get("frozen_trace_read")
                                                         or r.get("guard_hits")))
    if data:
        trouble.append("data files opened: %s" % data)
    v["frozen"] = [not trouble, "; ".join(trouble) or (
        "no frozen trace data read" + ("; declared stored data files read: %s"
                                       % declared_read if declared_read else ""))]
    trouble = []
    if undeclared:
        trouble.append("undeclared stored data: %s" % [dense[f] for f in undeclared])
    if missing:
        trouble.append("declared stored data not on the path: %s" % missing)
    if missing_files:
        trouble.append("declared stored data files not read: %s" % missing_files)
    v["stored"] = [not trouble, "; ".join(trouble) or (
        "stored data: %s" % ([dense[f] for f in sorted(dense)] + declared_read or "none"))]
    chart = sorted(_rel(f) for f in executed if os.path.basename(f) in AUDIT.CHART_FILES)
    v["bps"] = [not chart, ("BPS functions ran in %s" % chart) if chart else "no BPS function ran"]
    bad_imports = [m for m in r["loaded_names"]
                   if _BOOTSTRAP_MODULE.search(m) or _RG_MODULE.search(m.split(".")[-1])]
    v["imports"] = [not bad_imports, ("imported: %s" % bad_imports) if bad_imports
                    else "no bootstrap or RG module imported"]
    boot = sorted({"%s:%s" % (_rel(f), q) for f, q in r["calls"]
                   if _BOOTSTRAP_FILE.search(os.path.basename(f))
                   or (_rel(f), q) in _BOOTSTRAP_ROUTINES})
    v["bootstrap"] = [not boot, ("ran: %s" % boot[:6]) if boot
                      else "no orthonormality-bootstrap routine ran"]
    return v


def row_verdicts(row, r):
    if "child_error" in r:
        return {c: [False, r["child_error"]] for c in CHECKS}
    v = instrument_verdicts(r, row.stored, row.data_files)
    for c in ("tr1", "gens", "ortho", "geometry"):
        if c == "geometry" and row.family not in ("A", "D"):
            v[c] = None
        elif c in r["checks"]:
            v[c] = r["checks"][c]
        else:
            phase = next((p for p in ("build", "gens_list") if p in r["errors"]), None)
            v[c] = [False, "not run: %s failed: %s" % (phase, r["errors"].get(phase))]
    return v


def _check_routines_exist():
    """Each `_BOOTSTRAP_ROUTINES` entry is defined where it is named (a rename
    would blind the detector)."""
    missing = []
    for rel, qual in _BOOTSTRAP_ROUTINES:
        tree = ast.parse(open(os.path.join(_ROOT, rel)).read())
        found = set()

        def walk(node, prefix):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    q = prefix + ch.name
                    found.add(q)
                    walk(ch, q + ".")
        walk(tree, "")
        if qual not in found:
            missing.append("%s:%s" % (rel, qual))
    return missing


def _mark(v, expected):
    if v is None:
        return "n/a"
    if v[0]:
        return "PASS" if not expected else "XPASS"
    return "XFAIL" if expected else "FAIL"


def main(argv):
    slow = "--slow" in argv
    survey = "--survey" in argv
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 2
    only = set(argv[argv.index("--rows") + 1].split(",")) if "--rows" in argv else None
    t_start = time.time()
    problems = []

    missing = _check_routines_exist()
    if missing:
        print("ABORT: bootstrap routines not found where named: %s" % missing)
        return 1
    stale = [k for k in EXPECTED_FAILURES if k[0] not in ROWS_BY_NAME or k[1] not in CHECKS]
    if stale:
        print("ABORT: EXPECTED_FAILURES names unknown rows/checks: %s" % stale)
        return 1

    fixtures = tempfile.mkdtemp(prefix="ade_guard_")
    try:
        _write_fixtures(fixtures)
        # ---- controls first
        print("controls:")
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            recs = list(ex.map(lambda c: run_child("control", c[0], fixtures, 900), CONTROLS))
        for (name, _fn, fires, value), r in zip(CONTROLS, recs):
            if "child_error" in r:
                print("  FAIL  %s: %s" % (name, r["child_error"]))
                problems.append(name)
                continue
            v = instrument_verdicts(r, ())
            fired = {c for c in v if not v[c][0]}
            if fires is None:
                ok = True
            elif fires:
                ok = set(fires) <= fired and all(fires[c] in v[c][1] for c in fires)
            else:
                ok = not fired
            if value is not None:
                ok = ok and r.get("value") == value
            print("  %s  %s: fired %s%s%s" % (
                "PASS" if ok else "FAIL", name, sorted(fired) or "none",
                "; value %s" % r.get("value") if value is not None else "",
                "" if ok else " (wanted %s, value %s) %s %s" % (
                    fires, value, {c: v[c][1] for c in v}, r.get("errors"))))
            if not ok:
                problems.append(name)
        if problems:
            print("\nABORT: positive control(s) failed -- the instruments cannot be trusted: %s"
                  % problems)
            return 1

        # ---- rows
        rows = [r for r in ROWS if (slow or not r.slow) and (only is None or r.name in only)]
        skipped = [r for r in ROWS if r.slow and not slow and (only is None or r.name in only)]
        print("\nrows (%d%s):" % (len(rows), "; %d slow rows skipped, pass --slow" % len(skipped)
                                   if skipped else ""))

        def one(row):
            return row, run_child("row", row.name, fixtures,
                                  sum(row.caps.values()) * (_SLOW_CAP_FACTOR if slow else 1)
                                  + 300, slow)
        n_fail = 0
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            for row, r in ex.map(one, rows):
                v = row_verdicts(row, r)
                marks = {c: _mark(v[c], (row.name, c) in EXPECTED_FAILURES) for c in CHECKS}
                bad = [c for c in CHECKS if marks[c] in ("FAIL", "XPASS")]
                bad += [c for c in CHECKS if v[c] and not v[c][0]
                        and v[c][1].startswith("UNDECLARED ACCESSOR") and c not in bad]
                n_fail += bool(bad)
                print("  %-5s %-30s %s  (%.0f s)" % ("FAIL" if bad else "ok", row.name,
                      " ".join("%s:%s" % (c, marks[c]) for c in CHECKS), r.get("wall", 0)))
                for c in bad:
                    why = ("listed in EXPECTED_FAILURES but passes -- remove it"
                           if marks[c] == "XPASS" else v[c][1])
                    print("        %s: %s" % (c, why))
                if survey:
                    rec = {k: r.get(k) for k in ("seconds", "errors", "info", "wall")}
                    rec["verdicts"] = v
                    rec["windows"] = {"tr1": row.k1, "gens": row.kg}
                    rec.update(theory=row.theory, module=row.module, accessor=row.accessor,
                               route=row.route, window_note=row.window_note,
                               stored=list(row.stored), data_files=list(row.data_files))
                    print("SURVEY " + json.dumps({"row": row.name, **rec}, default=repr))
                if bad:
                    problems.append(row.name)
        print("\n%d rows, %d failing; %.0f s" % (len(rows), n_fail, time.time() - t_start))
    finally:
        shutil.rmtree(fixtures, ignore_errors=True)
    return 1 if problems else 0


def test_ade_serving_guard():
    """pytest entry point: the default tier."""
    assert main([]) == 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
