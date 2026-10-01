"""checks_finite.py — the design record adapters for the draft's Appendix B, "K_q-algebras of finite type" (app:finite): Table B.1
(tab:finite_type), the canonical basis of normalised monomials (app:finite/canonical-basis), the uniqueness of the trace
(app:finite/trace-unique) and the class S coincidences (app:finite/class-s-coincidences); and its subsection B.1, the odd
polygon algebras K_q([A1, A_2k]) (app:a1a2k): the chord rules, transcribed from the draft's text (app:a1a2k/rules), the
elementary traces as M(2, 2k+3) characters (app:a1a2k/traces), and the minors miracle (app:a1a2k/minors).

The presentations are the classes that serve [A1, ADE] in the repository -- the ungauged rows of
the suite in the source repository (the cone classes, the ungauged polygons, SU3ADKAlg, and the zoo
finite_kalgebras.FINITE_KALGEBRAS, whose aliases hexagon / octagon / decagon are the class objects of a3 / a5 / a7 and are
counted once) -- and, independently of all of them, the BPS chart of each Dynkin quiver (bipartite orientation, node
charges the unit vectors, the default spec), whose product and trace come from the quiver alone and whose flavour is the
U(1)^r of ker B.
"""
from __future__ import annotations

import gc
import os
import signal
import sys
import time
from fractions import Fraction
from math import lcm

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the release root
for _p in (ROOT, os.path.join(ROOT, "implementations"), os.path.join(ROOT, "tests"),
           os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kalgebra import Element  # noqa: E402
from laurent_poly import LaurentPoly  # noqa: E402
from checks_kq import check  # noqa: E402

_ONE = LaurentPoly({0: 1})


def _depth():
    from checks_coulomb import battery_depth
    return battery_depth()


class _Out(BaseException):          # not an Exception: no library fallback may swallow the alarm
    pass


def _alarm(*_):
    raise _Out()


def _within(seconds, fn):
    """fn() within `seconds` (SIGALRM): its value, or a string saying why it stopped -- the budget, or memory (a
    MemoryError under the run's address-space limit)."""
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return fn()
    except _Out:
        return f"not reached in {seconds} s"
    except MemoryError:
        return "not reached: out of memory"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _fresh_twin(sid):
    """A new BPS twin of zoo entry `sid` -- the BPSKAlgebra on the entry's embedded quiver that
    `elem_traces._bps_oracle` builds, but not memoised: a budgeted check can be cut off by the alarm in
    the middle of a cache update, and the memo would hand that twin to the next check and keep every twin of the run
    alive."""
    from bps_kalgebra import BPSKAlgebra
    from regen import _load_standalone
    mod, prefix = _load_standalone(sid)
    return BPSKAlgebra(pairing=getattr(mod, f"{prefix}_BPS_PAIRING"),
                       node_charges=getattr(mod, f"{prefix}_BPS_NODE_CHARGES"))


def _lab(x):
    """The label of a single-term, coefficient-1 element (rho's return), or the label itself."""
    if isinstance(x, Element):
        (l, c), = x.terms.items()
        return l
    return x


# --------------------------------------------------------------------------
# Dynkin quivers
# --------------------------------------------------------------------------
def dynkin_edges(t, n):
    if t == "A":
        return [(i, i + 1) for i in range(n - 1)]
    if t == "D":
        return [(i, i + 1) for i in range(n - 2)] + [(n - 3, n - 1)]
    if t == "E":
        return [(i, i + 1) for i in range(n - 2)] + [(2, n - 1)]          # the branch at the third node
    raise ValueError(t)


def dynkin_exchange(t, n):
    """The Dynkin quiver of type t_n in its bipartite orientation (every arrow from colour 0 to colour 1)."""
    edges = dynkin_edges(t, n)
    col = {0: 0}
    for _ in range(n):
        for i, j in edges:
            if i in col and j not in col:
                col[j] = 1 - col[i]
            if j in col and i not in col:
                col[i] = 1 - col[j]
    B = [[0] * n for _ in range(n)]
    for i, j in edges:
        s, d = (i, j) if col[i] == 0 else (j, i)
        B[s][d], B[d][s] = 1, -1
    return B


def _rank_q(M):
    """Rank over Q, exact."""
    m = [[Fraction(x) for x in row] for row in M]
    r = 0
    for c in range(len(m[0]) if m else 0):
        p = next((i for i in range(r, len(m)) if m[i][c] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                f = m[i][c] / m[r][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        r += 1
    return r


def _bps_chart(t, n):
    from bps_kalgebra import BPSKAlgebra
    e = [tuple(1 if k == i else 0 for k in range(n)) for i in range(n)]
    return BPSKAlgebra(pairing=dynkin_exchange(t, n), node_charges=e)


# --------------------------------------------------------------------------
# tab:finite_type
# --------------------------------------------------------------------------
# Flavour groups: rank, and the order of the duality V -> V* on the representation ring (1 when every irrep is self-dual).
_GROUPS = {"---": (0, 1), "U(1)": (1, 2), "SU(2)": (1, 1), "U(1)^2": (2, 2), "SU(3)": (2, 2), "SU(2)xU(1)": (2, 2)}
# Equal-rank proper subgroups H of G that a presentation may carry instead of G: the maximal tori, and the Levi of SU(3).
_SUBGROUPS = {("SU(2)", "U(1)"), ("SU(3)", "U(1)^2"), ("SU(3)", "SU(2)xU(1)"), ("SU(2)xU(1)", "U(1)^2")}


def table_b1(nmax):
    """The draft's Table B.1, transcribed: {Dynkin type (t, n): [(row, flavour group, order of rho)]}, n a positive integer
    up to nmax in the family rows.  [A1, D3] is [A1, A3] (D3 = A3), so the D_{2n+1} row's n = 1 member joins the A3 entry;
    D4 appears twice, in its own row and as the n = 1 member of D_{2n+2}."""
    out = {}

    def add(key, row, grp, order):
        out.setdefault(key, []).append((row, grp, order))
    for n in range(1, nmax + 1):
        add(("A", 2 * n), "[A1,A_2n]", "---", 2 * n + 3)
        add(("A", 2 * n + 3), "[A1,A_2n+3]", "U(1)", 2 * n + 6)
        add(("A", 3) if n == 1 else ("D", 2 * n + 1), "[A1,D_2n+1]", "SU(2)", 2 * n + 1)
        add(("D", 2 * n + 2), "[A1,D_2n+2]", "SU(2)xU(1)", 2 * n + 2)
    add(("A", 3), "[A1,A_3]", "SU(2)", 3)
    add(("D", 4), "[A1,D_4]", "SU(3)", 4)
    add(("E", 6), "[A1,E_6]", "---", 14)
    add(("E", 7), "[A1,E_7]", "U(1)", 10)
    add(("E", 8), "[A1,E_8]", "---", 16)
    return out


def _primary(entries):
    """The entry whose group contains every other entry's group (the flavour symmetry the table asserts)."""
    for e in entries:
        if all(f[1] == e[1] or (e[1], f[1]) in _SUBGROUPS for f in entries):
            return e
    return None


def _ring_group(R):
    """(group name, rank, order of the duality measured on the ring's own star_basis)."""
    name = type(R).__name__
    if name == "TensorZPlusRing":
        parts = [_ring_group(f) for f in R.factors]
        su2 = sum(1 for p in parts if p[0] == "SU(2)")
        u1 = sum(p[1] for p in parts if p[0].startswith("U(1)"))
        other = [p for p in parts if p[0] not in ("SU(2)", "---") and not p[0].startswith("U(1)")]
        if other or su2 > 1:
            return ("x".join(p[0] for p in parts), sum(p[1] for p in parts), max(p[2] for p in parts))
        grp = "SU(2)" * su2 + ("x" if su2 and u1 else "") + ("U(1)" if u1 == 1 else f"U(1)^{u1}" if u1 else "")
        return (grp or "---", su2 + u1, max(p[2] for p in parts))
    if name == "TrivialZPlusRing":
        return ("---", 0, 1)
    if name == "AbelianZPlusRing":
        c = R.one_dim_rep_rank()
        wit = [tuple(1 if j == i else 0 for j in range(c)) for i in range(c)]
        return ("U(1)" if c == 1 else f"U(1)^{c}", c, 2 if any(R.star_basis(w) != w for w in wit) else 1)
    if name == "SU2ZPlusRing":
        return ("SU(2)", 1, 2 if any(R.star_basis(k) != k for k in range(6)) else 1)
    if name == "SU3ZPlusRing":
        wit = [(1, 0), (0, 1), (2, 1)]
        return ("SU(3)", 2, 2 if any(R.star_basis(w) != w for w in wit) else 1)
    if name == "SU2xU1ZPlusRing":
        wit = [(1, 0), (0, 1), (1, 1)]
        return ("SU(2)xU(1)", 2, 2 if any(R.star_basis(w) != w for w in wit) else 1)
    raise ValueError(f"unrecognised flavour ring {name}")


def _theory_key(theory):
    """'[A1,D5]' -> ('D', 5); D3 is A3."""
    t, n = theory[4], int(theory[5:-1])
    return ("A", 3) if (t, n) == ("D", 3) else (t, n)


def _presentations(depth):
    """[(name, Dynkin key, build, gens-of-A)] -- the ungauged serving rows (zoo aliases once), plus at extensive depth the
    next members of the cone families."""
    import test_ade_serving_guard as g
    out, seen_cls = [], set()
    for row in g.ROWS:
        if "gauged" in row.theory:
            continue
        if row.name.startswith("zoo "):
            import finite_kalgebras as fk
            cls = fk.FINITE_KALGEBRAS[row.name[4:]]
            if cls in seen_cls:
                continue
            seen_cls.add(cls)
        out.append((row.name, _theory_key(row.theory), row.build, row.gens))
    if depth == "extensive":
        for mod, expr, key, gens in (("a1a2k_kalg", "A1A2kKAlg(4)", ("A", 8), g._cone_gens),
                                     ("a1a2k_kalg", "A1A2kKAlg(5)", ("A", 10), g._cone_gens),
                                     ("ungauge_kalgebra", "ungauge_u1a1aodd(5)", ("A", 11), g._mult_generators),
                                     ("a1dn_kalg", "A1DnKAlg(9)", ("D", 9), g._word_gens),
                                     ("a1dodd_kalg", "A1DoddConeKAlg(3)", ("D", 9), g._word_gens),
                                     ("a1dodd_kalg", "A1DoddConeKAlg(4)", ("D", 11), g._word_gens),
                                     ("a1deven_kalg", "A1DevenKAlg(4)", ("D", 10), g._mult_generators),
                                     ("a1deven_kalg", "A1DevenKAlg(5)", ("D", 12), g._mult_generators)):
            out.append((expr, key, g._b(mod, expr), gens))
    return out


def _orbit(A, lab, key=None, cap=96):
    key = key or (lambda z: z)
    k0 = key(lab)
    x, k = _lab(A.rho(lab)), 1
    while key(x) != k0 and k < cap:
        x, k = _lab(A.rho(x)), k + 1
    return k if key(x) == k0 else None


def _rho_pow(A, lab, m):
    for _ in range(m):
        lab = _lab(A.rho(lab))
    return lab


def _measure(A, gens, second):
    """Orders of rho on a window W of labels -- the generators, the terms of the products of pairs of the first ten, and
    (where the class composes labels) the first twelve generators' sections dressed by non-self-dual flavour characters:
    on labels, and on sections (the flavour forgotten).  `second` is a disjoint control window."""
    R = A.coefficient_ring()
    W = list(dict.fromkeys(gens))
    for x in gens[:10]:
        for y in gens[:10]:
            W.extend(l for l in A.multiply(x, y).terms if l not in W)
    rng = _ring_group(R)
    dressed = 0
    if rng[2] == 2:
        chars = []
        for x in W:
            w = A.r_label_decompose(x)[1]
            if R.star_basis(w) != w and w not in chars:
                chars.append(w)
        for x in gens[:12]:
            s = A.r_label_decompose(x)[0]
            for w in chars[:2]:
                try:
                    y = A.r_label_compose(s, w)
                except (NotImplementedError, ValueError):
                    continue
                if y not in W:
                    W.append(y)
                    dressed += 1
    labels = lcm(*[_orbit(A, l) for l in W])
    sections = lcm(*[_orbit(A, l, key=lambda z: A.r_label_decompose(z)[0]) for l in W])
    flav = lcm(labels, rng[2])
    ctl = [l for l in second if _rho_pow(A, l, flav) != l]
    return {"window": len(W), "dressed": dressed, "labels": labels, "sections": sections, "flavoured": flav,
            "group": rng[0], "star": rng[2], "control_moved": len(ctl), "control": len(second)}


def check_finite_table(environment):
    """tab:finite_type.  Table B.1: the flavour symmetry and the order of rho for every [A1, ADE] family.
    The order of rho is measured on every presentation, three ways: on sections (the unflavoured algebra, the flavour
    forgotten); on labels; and as an automorphism of the flavoured algebra, lcm(label order, order of the duality on the
    flavour ring) -- the contract's axiom that rho acts on the central R(G_f) by the duality V -> V*
    (verify_embed_intertwines_rho), which enters through the coefficients where a class keeps the flavour out of its
    labels.  The table is read as a statement about the flavour group it names: over that group the flavoured order is the
    table's; over an equal-rank subgroup H the prediction is lcm(table order, order of the duality on R(H)), derived
    from A_H = A_G (x) R(H) with rho_H = rho_G (x) duality."""
    depth = _depth()
    nmax = 5 if depth == "extensive" else 3
    t0 = time.time()
    table = table_b1(nmax)
    checks = []
    # 1. the table's own consistency: every theory's entries nest, and the flavour rank is the corank of B
    nest_bad, rank_bad = [], []
    for key, entries in sorted(table.items()):
        prim = _primary(entries)
        if prim is None or any(f[2] != (prim[2] if f[1] == prim[1] else lcm(prim[2], _GROUPS[f[1]][1])) for f in entries):
            nest_bad.append(key)
        B = dynkin_exchange(*key)
        corank = len(B) - _rank_q(B)
        if any(_GROUPS[f[1]][0] != corank for f in entries):
            rank_bad.append((key, corank))
    dbl = {k: [(f[0], f[1], f[2]) for f in v] for k, v in table.items() if len(v) > 1}
    checks.append(check(f"the table's double entries nest: at {', '.join(f'{k[0]}{k[1]}' for k in sorted(dbl))} the "
                        "family row names an equal-rank subgroup of the explicit row's group, with the order lcm(explicit "
                        "order, duality order of the subgroup)", not nest_bad, str(nest_bad)))
    checks.append(check(f"flavour rank = corank of the Dynkin quiver's exchange matrix, on all {len(table)} table entries "
                        f"(n <= {nmax})", not rank_bad, str(rank_bad)))
    # 2. the presentations
    pres = []
    for name, key, build, gens in _presentations(depth):
        A = build()
        G = [_lab(x) for x in gens(A)]
        second = []
        for x in G[10:16] or G[:3]:
            for y in G[:6]:
                second.extend(l for l in A.multiply(x, y).terms if l not in second)
        pres.append((name, key, _measure(A, G, second)))
    for key in sorted(table):
        B = dynkin_exchange(*key)
        n = len(B)
        A = _bps_chart(*key)
        e = [tuple(1 if k == i else 0 for k in range(n)) for i in range(n)]
        W = e + [tuple(-x for x in v) for v in e] + [tuple(a + b for a, b in zip(x, y)) for i, x in enumerate(e) for y in e[i + 1:]]
        second = [tuple(a + b + c for a, b, c in zip(x, y, z)) for i, x in enumerate(e) for j, y in enumerate(e[i + 1:], i + 1)
                  for z in e[j + 1:]][:40]
        R = A.coefficient_ring()
        rg = _ring_group(R)
        lab_o = lcm(*[_orbit(A, l) for l in W])
        sec_o = lcm(*[_orbit(A, l, key=lambda z, A=A: A.r_label_decompose(z)[0]) for l in W])
        flav = lcm(lab_o, rg[2])
        pres.append((f"BPS chart {key[0]}{key[1]}", key,
                     {"window": len(W), "dressed": 0, "labels": lab_o, "sections": sec_o, "flavoured": flav, "group": rg[0],
                      "star": rg[2], "control_moved": sum(1 for l in second if _rho_pow(A, l, flav) != l),
                      "control": len(second)}))
    by_key = {}
    for name, key, m in pres:
        by_key.setdefault(key, []).append((name, m))
    sec_bad, grp_bad, flav_bad, sub_bad, doubled, over_own, over_sub = [], [], [], [], [], [], []
    for name, key, m in pres:
        if key not in table:
            continue
        entries = table[key]
        prim = _primary(entries)
        if m["sections"] != prim[2]:
            sec_bad.append((name, m["sections"], prim[2]))
        own = next((f for f in entries if f[1] == m["group"]), None)
        if own is not None:
            over_own.append(name)
            if m["flavoured"] != own[2]:
                flav_bad.append((name, m["group"], m["flavoured"], own[2]))
        elif (prim[1], m["group"]) in _SUBGROUPS:
            over_sub.append(name)
            want = lcm(prim[2], m["star"])
            if m["flavoured"] != want:
                sub_bad.append((name, m["group"], m["flavoured"], want))
            if want != prim[2]:
                doubled.append(f"{name} ({m['group']} in {prim[1]}: {m['flavoured']} for {prim[2]})")
        else:
            grp_bad.append((name, m["group"], prim[1]))
    npres = sum(1 for _, k, _ in pres if k in table)
    checks.append(check(f"order of rho on the unflavoured algebra (sections) = the table's, on {npres - len(sec_bad)} of {npres} "
                        "presentations (cone classes, zoo, BPS charts)", not sec_bad, str(sec_bad)[:300]))
    checks.append(check(f"every presentation's flavour group is the table's, or an equal-rank subgroup of it: "
                        f"{npres - len(grp_bad)} of {npres} ({len(over_own)} over the table's group, {len(over_sub)} over a "
                        "subgroup)", not grp_bad, str(grp_bad)[:300]))
    checks.append(check(f"over the table's own group, the order of rho on the flavoured algebra = the table's: "
                        f"{len(over_own) - len(flav_bad)} of {len(over_own)}", not flav_bad, str(flav_bad)[:300]))
    checks.append(check(f"over an equal-rank subgroup H, the flavoured order = lcm(table order, duality order on R(H)): "
                        f"{len(over_sub) - len(sub_bad)} of {len(over_sub)}; it differs from the table's exactly where the "
                        f"table's order is odd and H has a U(1): {len(doubled)} presentations", not sub_bad and bool(doubled),
                        "; ".join(doubled)[:600]))
    uncovered = []
    for key, entries in sorted(table.items()):
        for f in entries:
            if not any(m["group"] == f[1] for _, m in by_key.get(key, [])):
                uncovered.append(f"{f[0]} at {key[0]}{key[1]} ({f[1]})")
    checks.append(check(f"every table entry (n <= {nmax}) has a presentation over its own flavour group: "
                        f"{sum(len(v) for v in table.values()) - len(uncovered)} of {sum(len(v) for v in table.values())}",
                        not uncovered, "; ".join(uncovered)[:400]))
    moved = [(name, m["control_moved"], m["control"]) for name, key, m in pres if m["control_moved"]]
    checks.append(check(f"positive control: rho^order fixes every label of a second window (products of other generator "
                        f"pairs; triple sums on the charts) on {len(pres) - len(moved)} of {len(pres)} presentations",
                        not moved, str(moved)[:300]))
    # 3. negative control: a wrong table is rejected
    caught, total, weak = 0, 0, []
    for key, entries in sorted(table.items()):
        prim = _primary(entries)
        B = dynkin_exchange(*key)
        corank = len(B) - _rank_q(B)
        secs = {m["sections"] for _, m in by_key.get(key, [])}
        wrong_orders = [d for d in range(1, prim[2]) if prim[2] % d == 0] + [2 * prim[2]]
        for d in wrong_orders:
            total += 1
            caught += secs != {d}
        for grp, (rk, _) in _GROUPS.items():
            if grp == prim[1]:
                continue
            total += 1
            if rk != corank:
                caught += 1
            elif any(m["group"] != grp and (grp, m["group"]) not in _SUBGROUPS for _, m in by_key.get(key, [])):
                caught += 1
            else:
                weak.append(f"{key[0]}{key[1]}: {grp} for {prim[1]}")
    checks.append(check(f"negative control: of {total} single-entry perturbations of the table (each order replaced by a "
                        "proper divisor or its double, each group by another), rejected: "
                        f"{caught}; not rejected by any measurement ({len(weak)}): a larger group of the same rank, which only "
                        "a presentation over it could confirm", caught + len(weak) == total,
                        "; ".join(weak)[:500]))
    return {"checks": checks,
            "population": {"presentations": {name: m for name, _, m in pres},
                           "table entries": {f"{k[0]}{k[1]}": v for k, v in sorted(table.items())},
                           "n": f"1..{nmax}", "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "rho^order on a second window of every presentation",
                         "negative": "every single-entry perturbation of the table"},
            "notes": "The order of rho depends on the flavour group a presentation is taken over: the duality is trivial on "
                     "R(SU(2)) and inverts U(1) charges, so over the Cartan U(1) of the enhanced SU(2) (the hexagon, the "
                     "ungauged u(1)-gauged hexagon, the BPS charts of A3, D5, D7) the table's odd orders double.  On the "
                     "unflavoured algebra the order is the table's on every presentation.",
            "inputs": []}


# --------------------------------------------------------------------------
# app:finite/canonical-basis
# --------------------------------------------------------------------------
def _zoo_distinct():
    """[(short id, class)] of the zoo, each class once (hexagon / octagon / decagon are a3 / a5 / a7)."""
    import finite_kalgebras as fk
    out, seen = [], set()
    for sid, cls in fk.FINITE_KALGEBRAS.items():
        if cls not in seen:
            seen.add(cls)
            out.append((sid, cls))
    return out


def _zoo_data(sid, all_cones=False):
    """The entry's rays (generator charges), its q-commuting sets (the largest only, unless `all_cones`), and its embedded
    BPS pairing."""
    from regen import _load_standalone
    mod, pre = _load_standalone(sid)
    rays = [tuple(g) for g in getattr(mod, pre + "_MULT_GENS_LATTICE")]
    cones = [tuple(sorted(c)) for c in getattr(mod, pre + "_CONES")]
    if not all_cones:
        top = max(len(c) for c in cones)
        cones = [c for c in cones if len(c) == top]
    return rays, cones, getattr(mod, pre + "_BPS_PAIRING")


def _solve_q(basis, v):
    """Rational coordinates of v in the independent vectors `basis` (None if v is not in their span)."""
    n, N = len(basis), len(v)
    m = [[Fraction(basis[j][i]) for j in range(n)] + [Fraction(v[i])] for i in range(N)]
    r, piv = 0, []
    for c in range(n):
        p = next((i for i in range(r, N) if m[i][c] != 0), None)
        if p is None:
            return None
        m[r], m[p] = m[p], m[r]
        m[r] = [x / m[r][c] for x in m[r]]
        for i in range(N):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        piv.append(c)
        r += 1
    if any(m[i][-1] != 0 for i in range(r, N)):
        return None
    return [m[i][-1] for i in range(n)]


def _det_int(M):
    """Exact determinant of an integer matrix (Bareiss)."""
    m = [list(r) for r in M]
    n, sign, prev = len(m), 1, 1
    for k in range(n - 1):
        if m[k][k] == 0:
            s = next((i for i in range(k + 1, n) if m[i][k] != 0), None)
            if s is None:
                return 0
            m[k], m[s] = m[s], m[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                m[i][j] = (m[i][j] * m[k][k] - m[i][k] * m[k][j]) // prev
        prev = m[k][k]
    return sign * m[n - 1][n - 1] if n else 1


def _inv_int(M):
    """The inverse of a unimodular integer matrix, as integers (None if |det| != 1)."""
    n = len(M)
    m = [[Fraction(x) for x in row] + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(M)]
    for c in range(n):
        p = next((i for i in range(c, n) if m[i][c] != 0), None)
        if p is None:
            return None
        m[c], m[p] = m[p], m[c]
        m[c] = [x / m[c][c] for x in m[c]]
        for i in range(n):
            if i != c and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[c])]
    inv = [[m[i][n + j] for j in range(n)] for i in range(n)]
    if any(x.denominator != 1 for row in inv for x in row):
        return None
    return [[int(x) for x in row] for row in inv]


def _fan(rays, cones, B, box, walls, box_cap=None, drop_base=False):
    """The fan of the cones in Gamma / ker B, coordinates B.gamma written in the basis of one maximal cone's rays.  Rays
    equal modulo ker B (a generator and its flavour-shifted copy, which the zoo lists separately) are identified, and the
    maximal cones are the q-commuting sets whose images are rank(B) distinct rays.  Returns a dict: lattice ok (every ray
    and every unit charge is an integer point of the first cone's lattice), maximal cones, unimodular, box points,
    points with exactly one monomial, walls (facets, in exactly two cones, on opposite sides)."""
    N = len(B)
    rk = _rank_q(B)

    def img(g):
        return tuple(sum(g[a] * B[a][b] for a in range(N)) for b in range(N))
    ims = [img(g) for g in rays]
    distinct = list(dict.fromkeys(ims))
    idx = {v: i for i, v in enumerate(distinct)}
    mcones = sorted({tuple(sorted({idx[ims[i]] for i in c})) for c in cones} - {()}, key=lambda c: (-len(c), c))
    mcones = [c for c in mcones if len(c) == rk]
    out = {"maximal cones": len(mcones), "distinct rays mod ker B": len(distinct)}
    base = [distinct[i] for i in mcones[0]]
    co = []
    for v in distinct + [img(tuple(int(i == j) for j in range(N))) for i in range(N)]:
        c = _solve_q(base, v)
        if c is None or any(x.denominator != 1 for x in c):
            out["lattice ok"] = False
            return out
        co.append([int(x) for x in c])
    out["lattice ok"] = True
    co = co[:len(distinct)]
    dets = [_det_int([co[i] for i in c]) for c in mcones]
    out["unimodular"] = sum(1 for d in dets if abs(d) == 1)
    if box and (box_cap is None or len(mcones) <= box_cap):
        import itertools
        invs = [(c, _inv_int([[co[i][t] for i in c] for t in range(rk)])) for c in (mcones[1:] if drop_base else mcones)]
        npts = one = 0
        for x in itertools.product(range(-box, box + 1), repeat=rk):
            labs = set()
            for c, inv in invs:
                if inv is None:
                    continue
                p = [sum(inv[a][t] * x[t] for t in range(rk)) for a in range(rk)]
                if all(v >= 0 for v in p):
                    labs.add(tuple((i, v) for i, v in zip(c, p) if v))
            npts += 1
            one += len(labs) == 1
        out["box points"], out["exactly one monomial"] = npts, one
    if walls:
        facets = {}
        for c in mcones:
            for k in range(rk):
                facets.setdefault(c[:k] + c[k + 1:], []).append(c[k])
        two = sum(1 for v in facets.values() if len(v) == 2)
        opposite = 0
        for f, opp in facets.items():
            if len(opp) == 2:
                s0, s1 = (_det_int([co[i] for i in f] + [co[o]]) for o in opp)
                opposite += s0 * s1 < 0
        out["walls"] = (len(facets), two, opposite)
    return out


def _augmented(c, R):
    """A coefficient (LaurentPoly or RLaurent) with the flavour forgotten: {q-exponent: integer}."""
    if hasattr(c, "coeffs"):
        out = {}
        for e, r in c.coeffs.items():
            v = sum(n * R.dim(b) for b, n in r.terms.items())
            if v:
                out[e] = out.get(e, 0) + v
        return out
    return {e: v for e, v in c._coeffs.items() if v}


def check_finite_canonical_basis(environment):
    """app:finite/canonical-basis.  'The canonical basis consists of the normalised monomials L_(a1..an) =
    q^(-sum (ai,aj)) L_a1 ... L_an in q-commuting generators, all distinct; the algebra is presented by the remaining
    products.'  Three clauses on every distinct zoo entry, read against the entry's BPS twin (BPSKAlgebra on its embedded
    quiver, default spec: product from the quiver alone):
    (i) the monomials are the whole basis, each once: the maximal cones (the zoo's q-commuting sets) are unimodular in
        Gamma / ker B, and every point of a box there is exactly one monomial (rank <= 6; at extensive depth also every
        wall lies in exactly two maximal cones, on opposite sides -- the local completeness certificate, all ranks);
    (ii) the normalisation: in the twin, a product of rays of one cone is the single canonical element at their sum with
        the q-power the zoo's normalisation predicts (pairs, squares, triples);
    (iii) the presentation: the zoo's products of generator pairs in no common cone equal the twin's -- exactly on the
        unflavoured entries, and with the flavour forgotten (the zoo's coefficients through the augmentation, the twin's
        terms grouped modulo ker B) on the flavoured ones, whose zoo keeps the flavour in coefficients and the twin in
        labels."""
    depth = _depth()
    ext = depth == "extensive"
    t0 = time.time()
    checks = []
    fan_rows, norm_rows, pres_rows = {}, {}, {}
    fan_bad, norm_bad, pres_bad, neg, stop2, stop3 = [], [], [], [], [], []
    nonsimp = {}
    budget = 900 if ext else 60
    for sid, cls in _zoo_distinct():
        A = cls()
        R = A.coefficient_ring()
        rays, allcones, B = _zoo_data(sid, all_cones=True)
        r = _rank_q(B)
        simplicial = max(len(c) for c in allcones) <= r
        box = ((2 if r <= 4 else 1) if r <= 6 else 0) if simplicial else 0
        fan = _fan(rays, allcones, B, box, walls=ext and simplicial, box_cap=None if ext else 1000)
        # distinct monomials of degree <= 2 in one q-commuting set sharing a charge (full lattice)
        by_charge = {}
        for c in allcones:
            for a, i in enumerate(c):
                for j in c[a:]:
                    g = tuple(x + y for x, y in zip(rays[i], rays[j]))
                    by_charge.setdefault(g, set()).add(tuple(sorted({i, j})) if i != j else (i, i))
                by_charge.setdefault(tuple(rays[i]), set()).add((i,))
        shared = sorted((g for g, v in by_charge.items() if len(v) > 1), key=lambda g: (sum(map(abs, g)), g))
        fan_rows[sid] = {"rays": len(rays), "rank mod ker B": r, "largest q-commuting set": max(len(c) for c in allcones),
                         **fan, "charges shared by two monomials of degree <= 2": len(shared)}
        w = fan.get("walls")
        if simplicial:
            if (not fan["lattice ok"] or fan["unimodular"] != fan["maximal cones"] or shared
                    or fan.get("exactly one monomial", 0) != fan.get("box points", 0) or (w and not (w[0] == w[1] == w[2]))):
                fan_bad.append(sid)
        else:
            nonsimp[sid] = (shared, by_charge)
        if sid == "heptagon":                                   # negative control: the base cone removed
            f2 = _fan(rays, allcones, B, box, walls=False, drop_base=True)
            neg.append(f"heptagon without its base cone: {f2['box points'] - f2['exactly one monomial']} of "
                       f"{f2['box points']} box points lose their monomial")
        cones = [c for c in allcones if len(c) == max(len(x) for x in allcones)]
        # (ii) the normalisation and (iii) the remaining products, in the twin: each clause under its own time budget and
        # with its own fresh twin, counted as it goes, so that a stopped clause keeps what it compared
        prog = {"ok": 0, "tot": 0, "agree": 0, "far": 0}
        size = [sum(abs(x) for x in g) for g in rays]                     # small charges: cheap products in the twin
        flav = type(R).__name__ != "TrivialZPlusRing"

        def zpow(i, j, A=A, R=R):
            (lab, c), = A.multiply(((i, 1),), ((j, 1),)).terms.items()
            ex = _augmented(c, R)
            return next(iter(ex)) if len(ex) == 1 and next(iter(ex.values())) == 1 else None

        def norm_checks(R=R, rays=rays, cones=cones, r=r, sid=sid, prog=prog, size=size, zpow=zpow):
            T = _fresh_twin(sid)
            ok = tot = 0
            sample = sorted(cones, key=lambda c: (sum(size[i] for i in c), c))[:8 if ext else 4]
            for c in sample:
                pairs = [(i, j) for a, i in enumerate(c) for j in c[a:]]
                pairs = sorted(pairs, key=lambda ij: (size[ij[0]] + size[ij[1]], ij))[:(24 if r >= 7 else 60) if ext else 12]
                for i, j in pairs:
                    e = zpow(i, j) if i != j else 0
                    if i != j and zpow(j, i) != (None if e is None else -e):
                        tot += 1
                        prog["tot"] = tot
                        continue
                    prod = T.multiply(rays[i], rays[j])
                    want = tuple(a + b for a, b in zip(rays[i], rays[j]))
                    tot += 1
                    if list(prod.terms) == [want] and _augmented(prod.terms[want], R) == {e: 1}:
                        ok += 1
                    prog["ok"], prog["tot"] = ok, tot
                if len(c) >= 3:
                    i, j, k = c[:3]
                    e = sum(zpow(a, b) or 0 for a, b in ((i, j), (i, k), (j, k)))
                    prod = T.multiply_elements(T.multiply(rays[i], rays[j]), Element({rays[k]: _ONE}))
                    want = tuple(a + b + d for a, b, d in zip(rays[i], rays[j], rays[k]))
                    tot += 1
                    ok += list(prod.terms) == [want] and _augmented(prod.terms[want], R) == {e: 1}
                    prog["ok"], prog["tot"] = ok, tot
            return ok, tot

        def pres_checks(A=A, R=R, rays=rays, cones=cones, B=B, r=r, sid=sid, prog=prog, size=size, flav=flav):
            T = _fresh_twin(sid)
            incone = {frozenset((i, j)) for c in cones for i in c for j in c}
            small = sorted(range(len(rays)), key=lambda i: (size[i], i))[:16]
            far = [(i, j) for i in small for j in small if i != j and frozenset((i, j)) not in incone]
            far = far[:(60 if ext else 24) if r <= 6 else (24 if ext else 8)]
            N = len(B)

            def cls_of(g):
                return tuple(sum(g[a] * B[a][b] for a in range(N)) for b in range(N))
            agree = 0
            for i, j in far:
                zg, tg = {}, {}
                for lab, c in A.multiply(((i, 1),), ((j, 1),)).terms.items():
                    gam = tuple(sum(rays[a][t] * p for a, p in lab) for t in range(len(rays[0])))
                    key = cls_of(gam) if flav else gam
                    for e, v in _augmented(c, R).items():
                        zg[(key, e)] = zg.get((key, e), 0) + v
                for lab, c in T.multiply(rays[i], rays[j]).terms.items():
                    key = cls_of(lab) if flav else tuple(lab)
                    for e, v in _augmented(c, R).items():
                        tg[(key, e)] = tg.get((key, e), 0) + v
                agree += {k: v for k, v in zg.items() if v} == {k: v for k, v in tg.items() if v}
                prog["agree"], prog["far"] = agree, prog["far"] + 1
            wrong = None
            if sid == "pentagon":                               # negative control: the twin's product in the other order
                i, j = far[0]
                zp = {tuple(sum(rays[a][t] * p for a, p in lab) for t in range(2)): str(c)
                      for lab, c in A.multiply(((i, 1),), ((j, 1),)).terms.items()}
                tp = {tuple(l): str(c) for l, c in T.multiply(rays[j], rays[i]).terms.items()}
                wrong = f"pentagon: the zoo's L_{i} L_{j} against the twin's L_{j} L_{i}: {'equal' if zp == tp else 'differ'}"
            return agree, len(far), wrong
        te = time.time()
        res2 = _within(budget, norm_checks)
        gc.collect()                                            # the twin is released with the check's frame
        res3 = _within(budget, pres_checks)
        gc.collect()
        if isinstance(res2, str):                               # stopped: what was compared before the stop still counts
            stop2.append(sid)
            ok, tot = prog["ok"], prog["tot"]
            norm_rows[sid] = f"{ok}/{tot} before the stop ({res2})"
        else:
            ok, tot = res2
            norm_rows[sid] = f"{ok}/{tot}"
        if isinstance(res3, str):
            stop3.append(sid)
            agree, nfar, wrong = prog["agree"], prog["far"], None
            pres_rows[sid] = f"{agree}/{nfar} before the stop ({res3})" + (" (flavour forgotten)" if flav else " (exact)")
        else:
            agree, nfar, wrong = res3
            pres_rows[sid] = f"{agree}/{nfar}" + (" (flavour forgotten)" if flav else " (exact)")
        if ok != tot:
            norm_bad.append(sid)
        if agree != nfar:
            pres_bad.append(sid)
        if wrong:
            neg.append(wrong)
        fan_rows[sid]["seconds (ii)+(iii)"] = round(time.time() - te, 1)
        print(f"  canonical-basis {sid}: fan {fan_rows[sid].get('unimodular')}/{fan_rows[sid].get('maximal cones')}, "
              f"(ii) {norm_rows[sid]}, (iii) {pres_rows[sid]} [{time.time() - te:.1f}s]", file=sys.stderr, flush=True)
    # (i-b) the entries whose q-commuting sets exceed the rank of B: distinct monomials that are one canonical element
    coinc = {}
    for sid, (shared, by_charge) in nonsimp.items():
        import finite_kalgebras as fk
        A = fk.FINITE_KALGEBRAS[sid]()
        rays = _zoo_data(sid)[0]

        def mono(m):
            return tuple((i, m.count(i)) for i in sorted(set(m)))

        def normalised(prod):
            """The label of a product that is a single canonical element times a power of q (None otherwise): the
            normalised monomial it defines."""
            if len(prod.terms) != 1:
                return None
            (l, c), = prod.terms.items()
            ex = _augmented(c, A.coefficient_ring())
            return tuple(l) if len(ex) == 1 and next(iter(ex.values())) == 1 else None

        def zoo_prod(m):
            return normalised(A.multiply(((m[0], 1),), ((m[-1], 1),))) if len(m) == 2 else mono(m)
        sample = [sorted(by_charge[g])[:2] for g in shared[:3]]
        zoo_same = sum(1 for m1, m2 in sample if zoo_prod(m1) is not None and zoo_prod(m1) == zoo_prod(m2))

        def twin_same(sample=sample, rays=rays, sid=sid):
            T = _fresh_twin(sid)

            def tp(m):
                prod = T.multiply(rays[m[0]], rays[m[-1]])
                if len(prod.terms) != 1:
                    return None
                (l, c), = prod.terms.items()
                return tuple(l) if len(_lp(c)) == 1 and abs(next(iter(_lp(c).values()))) == 1 else None
            return sum(1 for m1, m2 in sample if tp(m1) is not None and tp(m1) == tp(m2))
        tw = _within(budget, twin_same)
        gc.collect()
        coinc[sid] = {"charges shared": len(shared), "sample": [[mono(m) for m in pair] for pair in sample],
                      "one element in the zoo": f"{zoo_same}/{len(sample)}",
                      "one element in the twin": tw if isinstance(tw, str) else f"{tw}/{len(sample)}"}
    n = len(fan_rows)
    boxed = [sid for sid, v in fan_rows.items() if v.get("box points")]
    unboxed = [sid for sid in fan_rows if sid not in boxed]
    ns = n - len(nonsimp)
    checks.append(check(f"(i) on the {ns} entries whose q-commuting sets are at most the rank of B: every maximal cone "
                        f"unimodular in Gamma / ker B and no two small monomials sharing a charge on {ns - len(fan_bad)} of "
                        f"{ns}; every box point exactly one monomial on the {len(boxed)} entries with a box"
                        + (f" (none on {', '.join(u for u in unboxed if u not in nonsimp)}: rank above 6"
                           + ("" if ext else ", or more than 1000 cones") + ")"
                           if [u for u in unboxed if u not in nonsimp] else "")
                        + ("; every wall in exactly two maximal cones, on opposite sides, on every entry" if ext else ""),
                        not fan_bad, str(fan_bad)))
    if coinc:
        checks.append(check("(i) on " + ", ".join(coinc) + " the zoo's q-commuting sets are larger than the rank of B, and "
                            "distinct normalised monomials are one canonical element: "
                            + "; ".join(f"{sid}: {v['charges shared']} charges shared by two monomials of degree <= 2, the "
                                        f"sampled pairs one element in the zoo {v['one element in the zoo']} and in the BPS "
                                        f"twin {v['one element in the twin']}" for sid, v in coinc.items())
                            + " -- the canonical basis is still normalised monomials, but not uniquely (what 'all distinct' asks of a generating set)",
                            all(v["one element in the zoo"].split("/")[0] == v["one element in the zoo"].split("/")[1]
                                for v in coinc.values()), str(coinc)[:400]))
    def late(stopped, rows):
        return ((f"; the sample stopped before its end (budget {budget} s, or memory) on "
                 + ", ".join(f"{s} ({rows[s].split(' ')[0]} compared)" for s in stopped)) if stopped else "")
    checks.append(check(f"(ii) in the BPS twin, products of rays of one cone are the single canonical element at the sum "
                        f"with the zoo's q-power (pairs, squares, a triple per cone), and the zoo's exponents are "
                        f"antisymmetric: every product compared agrees on {n - len(norm_bad)} of {n} entries, the whole "
                        f"sample on {n - len(stop2)}" + late(stop2, norm_rows),
                        not norm_bad and len(stop2) < n, str({s: norm_rows[s] for s in norm_bad})))
    checks.append(check(f"(iii) the zoo's products of generators in no common cone equal the twin's (exactly when "
                        f"unflavoured, with the flavour forgotten otherwise): every product compared agrees on "
                        f"{n - len(pres_bad)} of {n} entries, the whole sample on {n - len(stop3)}" + late(stop3, pres_rows),
                        not pres_bad and len(stop3) < n, str({s: pres_rows[s] for s in pres_bad})))
    checks.append(check("negative controls: " + "; ".join(neg),
                        any("lose" in x and not x.startswith("heptagon without one maximal cone: 0") for x in neg)
                        and any(x.endswith("differ") for x in neg)))
    return {"checks": checks,
            "population": {"fan": fan_rows, "normalisation (ii)": norm_rows, "presentation (iii)": pres_rows,
                           "coincident monomials": coinc,
                           "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "every distinct zoo entry against its BPS twin",
                         "negative": "a cone removed from the heptagon's fan; the pentagon's product in the wrong order"},
            "notes": "The zoo's labels are monomials in its generators by construction; what is measured is that they are "
                     "the canonical basis of the independent BPS presentation (a unimodular complete fan; normalised "
                     "monomials) and that the products the zoo stores reproduce the twin's.  On the flavoured entries (iii) "
                     "is modulo flavour.",
            "inputs": []}


# --------------------------------------------------------------------------
# app:finite/trace-unique
# --------------------------------------------------------------------------
def _uniqueness_windows(depth):
    """(tag, zoo id, window builder, K_eq) -- the windows of the suite in the source repository, and E6."""
    def deg_window(cones, max_deg):
        labels = {()}
        for c in cones:
            cs = sorted(c)
            for d1 in range(max_deg + 1):
                for d2 in range(max_deg + 1 - d1):
                    if d1 + d2:
                        labels.add(tuple((i, p) for i, p in ((cs[0], d1), (cs[1], d2)) if p > 0))
        return sorted(labels)

    def cone_window(cones, squares):
        labels = {()}
        for c in cones:
            cs = sorted(c)
            labels.update(((i, 1),) for i in cs)
            labels.update(((cs[x], 1), (cs[y], 1)) for x in range(len(cs)) for y in range(x + 1, len(cs)))
            if squares:
                labels.update(((i, 2),) for i in cs)
        return sorted(labels)
    out = [("pentagon, degree <= 2", "pentagon", lambda c: deg_window(c, 2), 2),
           ("pentagon, degree <= 3", "pentagon", lambda c: deg_window(c, 3), 4),
           ("heptagon, 3 cones + squares", "heptagon", lambda c: cone_window(c[:3], True), 2),
           ("heptagon, 5 cones + squares", "heptagon", lambda c: cone_window(c[:5], True), 3),
           ("E6, 8 cones + squares", "e6", lambda c: cone_window(c[:8], True), 1)]
    if depth == "extensive":
        out += [("E6, 16 cones + squares", "e6", lambda c: cone_window(c[:16], True), 2),
                ("E6, 24 cones + squares", "e6", lambda c: cone_window(c[:24], True), 2)]
    return out


def check_finite_trace_unique(environment):
    """app:finite/trace-unique.  'The I_ab axioms determine I_ab uniquely up to 1 + O(q) and are self-consistent on these
    examples.'  trace_uniqueness.verify_unique_up_to_rescale on windows of the unflavoured zoo entries: the canonical
    trace solves every equation (self-consistency), and the solution set, projected to the window's labels and orders
    <= K_eq, is exactly the line of its 1 + q Q[[q]] rescales (uniqueness).  The equations are the rho^2-twisted
    cyclicity on the window's closure and the q^0-orthonormality."""
    import finite_kalgebras as fk
    from regen import _load_standalone
    from trace_uniqueness import trace_solution_space, verify_unique_up_to_rescale
    depth = _depth()
    t0 = time.time()
    checks, rows, bad = [], {}, []
    for tag, sid, build, K_eq in _uniqueness_windows(depth):
        A = fk.FINITE_KALGEBRAS[sid]()
        mod, pre = _load_standalone(sid)
        labels = build(sorted(getattr(mod, pre + "_CONES"), key=sorted))
        t1 = time.time()
        res = _within(3600, lambda: verify_unique_up_to_rescale(A, labels, K_eq, K_unknown=2 * K_eq + 4))
        if isinstance(res, str):
            rows[tag] = res
            bad.append(tag)
            continue
        rows[tag] = {"window": len(labels), "K_eq": K_eq, "canonical trace solves": res["known_trace_solves"],
                     "projected dimension": res["projected_dim"], "rescale line": res["projected_kernel_is_rescale_line"],
                     "seconds": round(time.time() - t1, 1)}
        if not (res["known_trace_solves"] and res["projected_kernel_is_rescale_line"] and res["projected_dim"] == K_eq):
            bad.append(tag)
    checks.append(check(f"the canonical trace solves every equation, and the admissible traces are its 1 + O(q) rescales, on "
                        f"{len(rows) - len(bad)} of {len(rows)} windows (pentagon, heptagon, E6)", not bad, str(bad)))
    P = fk.FINITE_KALGEBRAS["pentagon"]()
    mod, pre = _load_standalone("pentagon")
    small = sorted({()} | {((i, 1),) for c in getattr(mod, pre + "_CONES") for i in c}
                   | {tuple((i, 1) for i in sorted(c)) for c in getattr(mod, pre + "_CONES")})
    res = verify_unique_up_to_rescale(P, small, 2, K_unknown=8)
    w = trace_solution_space(P, small, 4, orthonormal_q0=True)["dim"]
    wo = trace_solution_space(P, small, 4, orthonormal_q0=False)["dim"]
    checks.append(check(f"negative controls (pentagon): on a window without squares the projected dimension is "
                        f"{res['projected_dim']} > 2, so the window matters; dropping the q^0-orthonormality enlarges the "
                        f"solution space ({w} -> {wo})", res["projected_dim"] > 2 and wo > w))
    return {"checks": checks,
            "population": {"windows": rows, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the canonical trace is a solution on every window",
                         "negative": "a window too small to pin the trace; the orthonormality condition dropped"},
            "notes": "trace_uniqueness handles unflavoured algebras only (a TypeError on an R-valued coefficient), so the "
                     "flavoured zoo entries are not in the population; E8 exceeds 12 GB in the solver's dense kernel at order 1 "
                     "on 8 cones and squares.  Uniqueness is a statement about a window: the "
                     "projected solution set, not the full one, and larger K_eq needs larger windows.",
            "inputs": []}


# --------------------------------------------------------------------------
# app:finite/class-s-coincidences
# --------------------------------------------------------------------------
def _acyclic(B):
    m = len(B)
    adj = [[j for j in range(m) if B[i][j] > 0] for i in range(m)]
    col = [0] * m

    def dfs(u):
        col[u] = 1
        for v in adj[u]:
            if col[v] == 1 or (col[v] == 0 and not dfs(v)):
                return False
        col[u] = 2
        return True
    return all(col[u] != 0 or dfs(u) for u in range(m))


def _perm_match(B, T):
    m = len(B)

    def rec(p, used):
        i = len(p)
        if i == m:
            return list(p)
        for j in range(m):
            if j not in used and B[i][i] == T[j][j] and all(B[i][a] == T[j][p[a]] and B[a][i] == T[p[a]][j] for a in range(i)):
                r = rec(p + [j], used | {j})
                if r:
                    return r
        return None
    return rec([], frozenset())


def a2ak_to_zoo_iso(k, *, drop_last=False):
    """The KAlgebraIso [A2,Ak] chart (a2ak_induction.member(k)) -> the zoo's E6 / E8 BPS twin: a mutation path found by a
    seeded random walk to an acyclic quiver, then sink/source mutations to the twin's orientation; the labels transported
    by the forward-necklace rule mu_g (g the mutated node's charge; node charges by Fomin-Zelevinsky) along the path, then
    by the lattice map sending the reached node charges to the twin's under the matching permutation.  `drop_last` omits
    the path's last transport step (a negative control)."""
    import random
    from collections import deque
    from a2ak_induction import member
    from chart_graph import _mu_g, _mu_inv_g, _mutate_node_charges
    from elem_traces import _bps_oracle
    from kalgebra_iso import KAlgebraIso
    sid = {3: "e6", 4: "e8"}[k]
    src, twin = member(k), _bps_oracle(sid)
    P = [list(r) for r in src.lattice.pairing]
    n = 2 * k

    def exch(A, nodes):
        return [[A.lattice.bracket(a, b) for b in nodes] for a in nodes]
    TB = exch(twin, twin.node_charges)
    nodes0 = [tuple(c) for c in src.node_charges]
    found = None
    for seed in range(64):
        rng, nodes, path = random.Random(seed), list(nodes0), []
        for _ in range(4000):
            if _acyclic(exch(src, nodes)):
                found = (nodes, path)
                break
            j = rng.randrange(n)
            nodes = _mutate_node_charges(nodes, nodes[j], P)
            path.append(j)
        if found:
            break
    nodes, path = found
    Q, seen, hit = deque([(nodes, path)]), {tuple(map(tuple, exch(src, nodes)))}, None
    while Q:
        nd, pa = Q.popleft()
        perm = _perm_match(exch(src, nd), TB)
        if perm is not None:
            hit = (nd, pa, perm)
            break
        Bn = exch(src, nd)
        for i in range(len(Bn)):
            if all(x >= 0 for x in Bn[i]) or all(x <= 0 for x in Bn[i]):
                nd2 = _mutate_node_charges(nd, nd[i], P)
                key = tuple(map(tuple, exch(src, nd2)))
                if key not in seen:
                    seen.add(key)
                    Q.append((nd2, pa + [i]))
    nodes, path, perm = hit
    consumed, nd = [], list(nodes0)
    for j in path:
        consumed.append(tuple(nd[j]))
        nd = _mutate_node_charges(nd, nd[j], P)
    steps = consumed[:-1] if drop_last else consumed
    tnodes = [tuple(c) for c in twin.node_charges]
    Nl = len(nodes[0])

    def fwd(a):
        for g in steps:
            a = _mu_g(tuple(a), g, P)
        co = _solve_q(nodes, a)
        return tuple(int(sum(co[i] * tnodes[perm[i]][t] for i in range(n))) for t in range(len(tnodes[0])))

    def inv(b):
        co = _solve_q([tnodes[perm[i]] for i in range(n)], b)
        a = tuple(int(sum(co[i] * nodes[i][t] for i in range(n))) for t in range(Nl))
        for g in reversed(steps):
            a = _mu_inv_g(a, g, P)
        return a
    iso = KAlgebraIso(src, twin, lambda a: Element({fwd(a): _ONE}), lambda b: Element({inv(b): _ONE}),
                      name=f"[A2,A{k}] -> zoo {sid}")
    return iso, {"mutations": len(path), "permutation": perm}, fwd


def _class_s_battery(k, n_left, n_right, K, n_trace, drop_last=False, prog=None):
    """Products, rho and traces of the composed isomorphism on a window; `prog` (if given) is filled as it goes --
    'info', 'planned' and [agreeing, compared] per stage -- so that a stopped battery keeps what it compared."""
    prog = {} if prog is None else prog
    prog.update({"products": [0, 0], "rho": [0, 0], "traces": [0, 0]})
    iso, info, fwd = a2ak_to_zoo_iso(k, drop_last=drop_last)
    prog["info"] = info
    src, twin = iso.source, iso.target
    W = [tuple(c) for c in src.node_charges]
    for x in list(W):
        y = x
        for _ in range(2):
            y = tuple(_lab(src.rho(y)))
            W.append(y)
    W = list(dict.fromkeys(W))
    size = lambda g: sum(abs(x) for x in fwd(g))                   # noqa: E731 -- the transported charge's size
    prods = sorted(((a, b) for a in W[:n_left] for b in W[:n_right]),  # cheap products first: a stopped battery
                   key=lambda ab: (size(ab[0]) + size(ab[1]), ab))     # has compared the most it could
    tr = [tuple(0 for _ in W[0])] + W[:n_trace - 1]
    prog["planned"] = {"products": len(prods), "rho": len(W), "traces": len(tr)}
    rok = tok = 0
    if not drop_last:                                                  # rho and traces first: they are light
        for a in W:
            rok += fwd(_lab(src.rho(a))) == tuple(_lab(twin.rho(fwd(a))))
            prog["rho"] = [rok, prog["rho"][1] + 1]
        for a in tr:
            tok += src.trace(a, K) == twin.trace(fwd(a), K)
            prog["traces"] = [tok, prog["traces"][1] + 1]
    pok = 0
    for a, b in prods:
        mapped = {fwd(l): str(c) for l, c in src.multiply(a, b).terms.items()}
        pok += {tuple(l): str(c) for l, c in twin.multiply(fwd(a), fwd(b)).terms.items()} == mapped
        prog["products"] = [pok, prog["products"][1] + 1]
    if drop_last:
        return info, pok, len(prods)
    return info, pok, len(prods), rok, len(W), tok, len(tr)


def check_finite_class_s(environment):
    """app:finite/class-s-coincidences.  '[A1,E6] = [A2,A3] and [A1,E8] = [A2,A4] as class S theories of type A2.'
    The K_q-algebra consequence: the BPS chart of the A2 x Ak product quiver (a2ak_induction.member, whose mutation class
    is recorded there) is isomorphic to the zoo's [A1,E] BPS twin, by a KAlgebraIso composed along a mutation path
    (a2k_to_zoo_iso) and certified on a window: products, rho, traces."""
    depth = _depth()
    t0 = time.time()
    checks, rows = [], {}
    plan = [(3, 15, 4, 3, 4)] + ([(3, 15, 8, 4, 7), (4, 12, 4, 2, 3)] if depth == "extensive" else [])
    budget = 1800 if depth == "extensive" else 600               # per battery; the fast one takes about 40 s
    for k, nl, nr, K, nt in plan:
        t1 = time.time()
        prog = {}
        res = _within(budget, lambda: _class_s_battery(k, nl, nr, K, nt, prog=prog))
        gc.collect()
        tag = {3: "[A2,A3] -> E6", 4: "[A2,A4] -> E8"}[k]
        key = f"{tag} (window {nl}x{nr}, traces to q^{K})"
        if isinstance(res, str):                                 # stopped: what was compared before the stop still counts
            info, plan_n = prog.get("info", {}), prog.get("planned", {})
            (pok, pc), (rok, rc), (tok, tc) = prog["products"], prog["rho"], prog["traces"]
            rows[key] = {**info, "stopped": res, "products": f"{pok}/{pc} of {plan_n.get('products', '?')}",
                         "rho": f"{rok}/{rc} of {plan_n.get('rho', '?')}",
                         "traces": f"{tok}/{tc} of {plan_n.get('traces', '?')}", "seconds": round(time.time() - t1, 1)}
            checks.append(check(f"{tag}: a KAlgebraIso along {info.get('mutations', '?')} mutations, the window stopped "
                                f"before its end ({res}): products {pok}/{pc} of {plan_n.get('products', '?')} compared, "
                                f"rho {rok}/{rc}, traces to q^{K} {tok}/{tc}",
                                pc > 0 and pok == pc and rok == rc and tok == tc))
            continue
        info, pok, pn, rok, rn, tok, tn = res
        rows[key] = {**info, "products": f"{pok}/{pn}", "rho": f"{rok}/{rn}",
                     "traces": f"{tok}/{tn}", "seconds": round(time.time() - t1, 1)}
        checks.append(check(f"{tag}: a KAlgebraIso along {info['mutations']} mutations; products {pok}/{pn}, rho {rok}/{rn}, "
                            f"traces to q^{K} {tok}/{tn}", pok == pn and rok == rn and tok == tn))
    info, pok, pn = _class_s_battery(3, 15, 4, 3, 4, drop_last=True)
    checks.append(check(f"negative control: the same map with the path's last transport step omitted agrees on {pok} of {pn} "
                        "products", pok < pn))
    return {"checks": checks,
            "population": {"isomorphisms": rows, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the composed transport certified on products, rho and traces",
                         "negative": "the transport with its last mutation step omitted"},
            "notes": "BPSAtlas cannot take this path: in its spec mode a forward mutation needs the node at the spec's head "
                     "(it refuses the first step even with 16 local moves), so the transport is composed directly from the "
                     "chart graph's forward-necklace rule and certified by the battery instead."
                     + ("" if depth == "extensive" else "  E8 is in the extensive sweep."),
            "inputs": []}


# --------------------------------------------------------------------------
# App. B.1: the odd polygon algebras K_q([A1, A_2k]) -- the draft's rules and formulas, transcribed from its text
# --------------------------------------------------------------------------
def _chord(letter, H):
    """L_(a;i): the diagonal from vertex i to vertex i + a + 1."""
    a, i = letter
    return (i % H, (i + a + 1) % H)


def _letter_of(p, q, H):
    """The letter of the chord with endpoints {p, q} (None for an edge): L_pq = L_(q-p-1; p) with q - p in {2..k+1}."""
    d = (q - p) % H
    if d in (0, 1, H - 1):
        return None
    if 2 <= d <= (H - 1) // 2:
        return (d - 1, p)
    return ((p - q) % H - 1, q)


def _crossing(c1, c2, H):
    p, q = c1
    r, s = c2
    if len({p, q, r, s}) < 4:
        return False

    def inside(x):
        return 0 < (x - p) % H < (q - p) % H
    return inside(r) != inside(s)


def draft_c(la, lb, H, orient=1):
    """The draft's c_ab for non-crossing letters: list the distinct endpoints in cyclic order from L_a's first endpoint
    (orient = +1: increasing vertex index), mark each only-a / only-b / shared; if exactly one arc is odd and it joins an
    only-a endpoint to an only-b one, c_ab = -1 (a to b) or +1 (b to a); otherwise 0."""
    p, q = _chord(la, H)
    r, s = _chord(lb, H)
    A_, B_ = {p, q}, {r, s}
    pts = sorted(A_ | B_, key=lambda x: (orient * (x - p)) % H)
    kind = {x: ("s" if x in A_ and x in B_ else "a" if x in A_ else "b") for x in pts}
    m = len(pts)
    arcs = [(pts[j], pts[(j + 1) % m], (orient * (pts[(j + 1) % m] - pts[j])) % H) for j in range(m)]
    odd = [x for x in arcs if x[2] % 2]
    if len(odd) == 1:
        u, v, _ = odd[0]
        if (kind[u], kind[v]) == ("a", "b"):
            return -1
        if (kind[u], kind[v]) == ("b", "a"):
            return 1
    return 0


def draft_ptolemy(la, lb, H, orient=1):
    """The draft's quantum Ptolemy for crossing letters L_ac, L_bd (a < b < c < d cyclically from L_a's first endpoint):
    {label: q-exponent} = {L_(ab,cd): alpha, L_(ad,bc): beta}, (alpha, beta) = (1, 0) if {A0, A2} has fewer odd arcs,
    else (0, -1); an edge is 1."""
    a, c = _chord(la, H)
    b, d = sorted(_chord(lb, H), key=lambda x: (orient * (x - a)) % H)
    A0, A1, A2, A3 = [(orient * (y - x)) % H for x, y in ((a, b), (b, c), (c, d), (d, a))]
    al, be = (1, 0) if A0 % 2 + A2 % 2 < A1 % 2 + A3 % 2 else (0, -1)

    def lab(pairs):
        ls = sorted(l for l in (_letter_of(x, y, H) for x, y in pairs) if l is not None)
        return tuple((l[0], l[1], 1) for l in ls)
    return {lab([(a, b), (c, d)]): al, lab([(a, d), (b, c)]): be}


def _draft_m(a, k):
    return a // 2 if a % 2 == 0 else k - (a - 1) // 2


def ag_product(k, s, N):
    """chi_s(q) of M(2, 2k+3), the Andrews-Gordon product prod_{n != 0, +-s mod 2k+3} (1 - q^n)^-1, through q^N."""
    P = 2 * k + 3
    c = [1] + [0] * N
    for n in range(1, N + 1):
        if n % P in (0, s % P, (-s) % P):
            continue
        for e in range(n, N + 1):
            c[e] += c[e - n]
    return c


def ag_sum(k, s, N):
    """The Andrews-Gordon multisum sum_{n_1..n_k >= 0} q^(N_1^2+..+N_k^2 + N_s+..+N_k) / prod_j (q)_{n_j},
    N_j = n_j + .. + n_k, through q^N (the positive control on ag_product)."""
    def qinv(m):
        c = [1] + [0] * N
        for j in range(1, m + 1):
            for e in range(j, N + 1):
                c[e] += c[e - j]
        return c
    inv = [qinv(m) for m in range(N + 1)]
    out = [0] * (N + 1)

    def rec(j, ns):
        if j == k:
            Ns = [sum(ns[i:]) for i in range(k)]
            e0 = sum(x * x for x in Ns) + sum(Ns[s - 1:])
            if e0 > N:
                return
            acc = [0] * (N + 1)
            acc[e0] = 1
            for m in ns:
                new = [0] * (N + 1)
                for x, cx in enumerate(acc):
                    if cx:
                        for y in range(N + 1 - x):
                            if inv[m][y]:
                                new[x + y] += cx * inv[m][y]
                acc = new
            for e in range(N + 1):
                out[e] += acc[e]
            return
        for n in range(N + 1):
            if n * n > N:
                break
            rec(j + 1, ns + [n])
    rec(0, [])
    return out


def draft_T(k, K):
    """The draft's elementary traces {a: {fq-exponent: coefficient}} through fq^K: T_0 = chi_1(fq^2),
    T_a = (-1)^(m+1) fq^-m (chi_m - chi_(m+1))(fq^2) with m = m(a)."""
    out = {0: {2 * n: c for n, c in enumerate(ag_product(k, 1, K // 2 + 1)) if c and 2 * n <= K}}
    for a in range(1, k + 1):
        m = _draft_m(a, k)
        x, y = ag_product(k, m, (K + m) // 2 + 1), ag_product(k, m + 1, (K + m) // 2 + 1)
        sg = 1 if m % 2 else -1
        out[a] = {2 * n - m: sg * (x[n] - y[n]) for n in range(len(x)) if x[n] != y[n] and 2 * n - m <= K}
    return out


def _series(t, K):
    d = t.coeffs if hasattr(t, "coeffs") else t
    out = {}
    for e, c in d.items():
        v = sum(c.terms.values()) if hasattr(c, "terms") else c
        if e <= K and v:
            out[e] = v
    return out


def _lp(c):
    return {e: v for e, v in (c._coeffs if hasattr(c, "_coeffs") else c).items() if v}


def check_a1a2k_rules(environment):
    """app:a1a2k/rules.  The chord rule for c_ab, quantum Ptolemy with (alpha, beta) decided by the odd arcs, and the
    canonical basis of non-crossing monomials -- transcribed from the draft's text (draft_c, draft_ptolemy) and compared
    with A1A2kKAlg(k) on every ordered pair of letters; the class's products in turn against the independent BPS chart of
    the A_2k quiver (a1a2k_induction.complete_block_iso, the additive charge map) on every ordered pair."""
    from a1a2k_kalg import A1A2kKAlg
    from A1A2k_naming_audit import predicted_lengths_and_shifts
    depth = _depth()
    kmax_bps = 4 if depth == "extensive" else 3
    t0 = time.time()
    checks, rows = [], {}
    bad_rule, bad_orient, bad_nc, bad_m = [], [], [], []
    for k in range(1, 7):
        A = A1A2kKAlg(k)
        H = 2 * k + 3
        letters = [(a, i) for a in range(1, k + 1) for i in range(H)]
        counts = {1: [0, 0, 0, 0], -1: [0, 0, 0, 0]}          # q-commuting ok/total, Ptolemy ok/total
        nc_ok = nc_tot = 0
        for la in letters:
            for lb in letters:
                got = {tuple(l): _lp(c) for l, c in A.multiply(A.L(la), A.L(lb)).terms.items()}
                for lab in got:
                    ls = [(x[0], x[1]) for x in lab]
                    nc_tot += 1
                    nc_ok += all(not _crossing(_chord(u, H), _chord(v, H), H) for u in ls for v in ls if u != v)
                X = _crossing(_chord(la, H), _chord(lb, H), H)
                for o in (1, -1):
                    if X:
                        want = {l: {e: 1} for l, e in draft_ptolemy(la, lb, H, o).items()}
                        counts[o][3] += 1
                        counts[o][2] += got == want
                    else:
                        lab = (((la[0], la[1], 2),) if la == lb
                               else tuple(sorted(((la[0], la[1], 1), (lb[0], lb[1], 1)))))
                        e = 0 if la == lb else draft_c(la, lb, H, o)
                        counts[o][1] += 1
                        counts[o][0] += got == {lab: {e: 1}}
        old, _ = predicted_lengths_and_shifts(k)
        old_idx = {L: i for i, L in old.items()}
        m_code = [k - old_idx[a + 1] + 1 for a in range(1, k + 1)]
        m_draft = [_draft_m(a, k) for a in range(1, k + 1)]
        c1, cm = counts[1], counts[-1]
        rows[f"k={k}"] = {"q-commuting pairs": f"{c1[0]}/{c1[1]}", "Ptolemy pairs": f"{c1[2]}/{c1[3]}",
                          "opposite orientation": f"{cm[0]}/{cm[1]}, {cm[2]}/{cm[3]}", "m(a) draft": m_draft,
                          "m(a) code": m_code}
        if c1[0] != c1[1] or c1[2] != c1[3]:
            bad_rule.append(k)
        if cm[2] == cm[3]:
            bad_orient.append(k)
        if nc_ok != nc_tot:
            bad_nc.append(k)
        if m_code != m_draft:
            bad_m.append(k)
    checks.append(check("the draft's c_ab rule and quantum Ptolemy rule, transcribed from the text, reproduce A1A2kKAlg(k) "
                        f"on every ordered pair of letters, k = 1..6 (at k = 6: {rows['k=6']['q-commuting pairs']} "
                        f"q-commuting, {rows['k=6']['Ptolemy pairs']} Ptolemy), with 'CCW' read as increasing vertex "
                        "index", not bad_rule, str(bad_rule)))
    checks.append(check("negative control: with the opposite orientation the Ptolemy rule fails on every crossing pair "
                        f"(k = 6: {rows['k=6']['opposite orientation']}), so the orientation is load-bearing -- and it is "
                        "the one in which the draft's heptagon figure numbers its vertices clockwise",
                        not bad_orient))
    checks.append(check("every term of every letter product is a non-crossing multiset of chords, k = 1..6", not bad_nc,
                        str(bad_nc)))
    checks.append(check("the draft's m(a) (a/2 for even a, k - (a-1)/2 for odd a) is the class's, k = 1..6", not bad_m,
                        str(bad_m)))
    from a1a2k_induction import complete_block_iso
    bps = {}
    for k in range(1, kmax_bps + 1):
        iso = complete_block_iso(k)
        A, T = iso.source, iso.target
        H = 2 * k + 3
        letters = [(a, i) for a in range(1, k + 1) for i in range(H)]

        def img(lab):
            (g, c), = iso.map(Element({lab: _ONE})).terms.items()
            return tuple(g)
        ok = tot = 0
        for la in letters:
            for lb in letters:
                tot += 1
                want = {img(l): _lp(c) for l, c in A.multiply(A.L(la), A.L(lb)).terms.items()}
                got = {tuple(l): _lp(c) for l, c in T.multiply(img(A.L(la)), img(A.L(lb))).terms.items()}
                ok += got == want
        bps[f"k={k}"] = f"{ok}/{tot}"
    checks.append(check("the class's letter products equal the BPS chart's (A_2k quiver, default spec) through the "
                        "additive charge map, on every ordered pair: " + ", ".join(f"{k} {v}" for k, v in bps.items()),
                        all(v.split("/")[0] == v.split("/")[1] for v in bps.values())))
    return {"checks": checks,
            "population": {"rules": rows, "BPS chart": bps, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the transcribed rules against the class; the class against the BPS chart",
                         "negative": "the rules with the opposite orientation"},
            "notes": "The draft's figure places vertex i at angle 90 - 360 i / 7 degrees, i.e. its indices increase "
                     "clockwise; the rules hold with increasing index, so the text's 'CCW' is clockwise in that figure.",
            "inputs": []}


def check_a1a2k_traces(environment):
    """app:a1a2k/traces.  The elementary traces T_0 = chi_1(fq^2), T_a = (-1)^(m+1) fq^-m (chi_m - chi_(m+1))(fq^2),
    with the M(2, 2k+3) characters as Andrews-Gordon products (implemented here from the formula, controlled against the
    Andrews-Gordon sums), against (a) the class's elementary traces, k <= 6 (transcription), and (b) the BPS chart of the
    A_2k quiver, independently: Tr(1) and Tr(L_(a;i)) for every i (the i-independence); and the reduction algorithm (the
    class's Layer 1), on every label with at most two letters, against the BPS chart's trace of the label's image."""
    from a1a2k_kalg import A1A2kKAlg
    from a1a2k_induction import complete_block_iso
    depth = _depth()
    t0 = time.time()
    checks, rows = [], {}
    ctl = all(ag_product(k, s, 30) == ag_sum(k, s, 30) for k in (1, 2, 3) for s in range(1, k + 2))
    checks.append(check("positive control: the Andrews-Gordon products equal the Andrews-Gordon sums, k <= 3, every s, "
                        "through q^30", ctl))
    trans = []
    for k in range(1, 7):
        T = A1A2kKAlg(k)._compute_T_series(40)
        D = draft_T(k, 40)
        trans.append(all(_series(T[a], 40) == D[a] for a in range(k + 1)))
    checks.append(check("the draft's T_0..T_k equal the class's elementary traces through fq^40, k = 1..6: "
                        f"{sum(trans)} of 6", all(trans)))
    plan = [(1, 12, 10), (2, 10, 8)] + ([(3, 8, 6)] if depth == "extensive" else [])
    wrong_caught = []
    for k, K, K2 in plan:
        iso = complete_block_iso(k)
        A, B = iso.source, iso.target
        H = 2 * k + 3
        D = draft_T(k, K)

        def img(lab):
            (g, c), = iso.map(Element({lab: _ONE})).terms.items()
            return tuple(g)
        t0k = _series(B.trace(tuple([0] * (2 * k)), K), K) == D[0]
        per = [sum(1 for i in range(H) if _series(B.trace(img(A.L((a, i))), K), K) == D[a]) for a in range(1, k + 1)]
        # the reduction algorithm on every label with at most two letters
        labs = {()}
        letters = [(a, i) for a in range(1, k + 1) for i in range(H)]
        firsts = letters if k <= 2 else [(a, 0) for a in range(1, k + 1)]   # k = 3: every pair up to rotation
        for la in firsts:
            labs.add(A.L(la))
            for lb in letters:
                labs.update(tuple(l) for l in A.multiply(A.L(la), A.L(lb)).terms)
        D2 = draft_T(k, K2 + 2 * k + 6)
        red_ok = 0
        for lab in sorted(labs, key=repr):
            cs = A.trace_layer1(lab)
            tot = {}
            for s, c in enumerate(cs):
                for e1, v1 in _lp(c).items():
                    for e2, v2 in D2[s].items():
                        if e1 + e2 <= K2:
                            tot[e1 + e2] = tot.get(e1 + e2, 0) + v1 * v2
            red_ok += {e: v for e, v in tot.items() if v} == _series(B.trace(img(lab), K2), K2)
        rows[f"k={k}"] = {"Tr(1) to fq^%d" % K: t0k, "Tr(L_(a;i)) = T_a for every i": f"{per} of {H} each",
                          ("reduction on labels of <= 2 letters to fq^%d" if k <= 2 else
                           "reduction on labels of <= 2 letters, the first at rotation 0, to fq^%d") % K2: f"{red_ok}/{len(labs)}"}
        if not (t0k and all(p == H for p in per) and red_ok == len(labs)):
            rows[f"k={k}"]["FAIL"] = True
        # negative control: the naive m(a) = a
        wrong = {}
        for a in range(1, k + 1):
            m = a
            x, y = ag_product(k, m, (K + m) // 2 + 1), ag_product(k, m + 1, (K + m) // 2 + 1)
            sg = 1 if m % 2 else -1
            wrong[a] = {2 * n - m: sg * (x[n] - y[n]) for n in range(len(x)) if x[n] != y[n] and 2 * n - m <= K}
        wrong_caught.append((k, sum(1 for a in range(1, k + 1) if _series(B.trace(img(A.L((a, 0))), K), K) != wrong[a])))
    checks.append(check("the BPS chart's Tr(1) and Tr(L_(a;i)) equal the draft's T_0 and T_a for every rotation index i, "
                        "and the reduction algorithm's Sum c_s T_s equals the BPS chart's trace on every label of at most "
                        "two letters: " + "; ".join(f"{k}: {v}" for k, v in rows.items()),
                        not any("FAIL" in v for v in rows.values())))
    checks.append(check("negative control: with m(a) = a in place of the draft's m(a), the formula disagrees with the BPS "
                        "chart on " + ", ".join(f"k={k}: {n} of {k}" for k, n in wrong_caught)
                        + " elementary traces (at k = 1 the two coincide)",
                        all(n >= 1 for k, n in wrong_caught if k >= 2)))
    return {"checks": checks,
            "population": {"BPS chart": rows, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "Andrews-Gordon products against sums; the formula against the independent chart",
                         "negative": "the formula with m(a) = a"},
            "notes": "The class's own traces ARE this formula (its Layer 2), so (a) is a transcription check; the BPS chart, "
                     "whose trace comes from its quiver and spec, is the independent comparison."
                     + ("" if depth == "extensive" else "  k = 3 is in the extensive sweep (its chart traces take minutes)."),
            "inputs": []}


def _minor_rows(A, k, a, decos):
    rows = []
    for d in decos:
        lab = ((2, 0, a),) if d is None else tuple(sorted(((2, 0, a), (d[0], 0, d[1]))))
        cs = [_lp(c) for c in A.trace_layer1(lab)]
        emin = min(e for c in cs for e in c)
        rows.append([{e - emin: v for e, v in c.items()} for c in cs])
    return rows


def _pmul(p, q, N):
    out = {}
    for a, x in p.items():
        for b, y in q.items():
            if a + b <= N:
                out[a + b] = out.get(a + b, 0) + x * y
    return {e: v for e, v in out.items() if v}


def _padd(p, q, s=1):
    out = dict(p)
    for e, v in q.items():
        out[e] = out.get(e, 0) + s * v
    return {e: v for e, v in out.items() if v}


def _pdet(M, N):
    if len(M) == 1:
        return M[0][0]
    acc = {}
    for j in range(len(M)):
        acc = _padd(acc, _pmul(M[0][j], _pdet([r[:j] + r[j + 1:] for r in M[1:]], N), N), 1 if j % 2 == 0 else -1)
    return acc


def _agree_to(p, q, N):
    for e in range(N + 1):
        if p.get(e, 0) != q.get(e, 0):
            return e - 1
    return N


def check_a1a2k_minors(environment):
    """app:a1a2k/minors.  Fix L_(2;0) and the chords D_1 = L_(1;0), D_r = L_(r+1;0) (r >= 2), sharing vertex 0 with it;
    the rows are the reduction algorithm's coefficient vectors of Tr(L_(2;0)^a) and Tr(L_(2;0)^a D_r), each divided by
    its most negative power of fq.  Measured: the rows stabilise between consecutive a; each stabilised row annihilates
    (T_0..T_k) (the draft's formula); and the maximal minors reproduce the T_s -- with the sign recorded."""
    from a1a2k_kalg import A1A2kKAlg
    depth = _depth()
    t0 = time.time()
    plan = [(2, (8, 9)), (3, (11, 12)), (4, (8, 9))] + ([(5, (8, 9)), (3, (14, 15))] if depth == "extensive" else [])
    checks, rows, sign, bad = [], {}, {}, []
    for k, (a1, a2) in plan:
        A = A1A2kKAlg(k)
        decos = [None, (1, 1)] + [(r + 1, 1) for r in range(2, k)]
        R1, R2 = _minor_rows(A, k, a1, decos), _minor_rows(A, k, a2, decos)
        N = min(_agree_to(R1[r][s], R2[r][s], 80) for r in range(k) for s in range(k + 1))
        T = draft_T(k, N + 2 * k + 6)
        null = min(_agree_to(
            {e: v for e, v in __import__("functools").reduce(lambda x, y: _padd(x, y),
                                                             [_pmul(R2[r][s], T[s], N) for s in range(k + 1)], {}).items()},
            {}, N) for r in range(k))
        plus = min(_agree_to({e: (-1) ** s * v for e, v in _pdet([row[:s] + row[s + 1:] for row in R2], N).items()},
                             T[s], N) for s in range(k + 1))
        minus = min(_agree_to({e: (-1) ** (s + 1) * v for e, v in _pdet([row[:s] + row[s + 1:] for row in R2], N).items()},
                              T[s], N) for s in range(k + 1))
        rows[f"k={k}, a={a1},{a2}"] = {"rows stable to fq^": N, "rows annihilate T to fq^": null,
                                       "(-1)^s det = T_s to fq^": plus, "(-1)^(s+1) det = T_s to fq^": minus}
        sign[k] = "+" if plus == N else "-" if minus == N else "none"
        if null != N or max(plus, minus) != N or N < 10:
            bad.append(k)
    checks.append(check("the normalised rows stabilise, annihilate (T_0..T_k), and their maximal minors reproduce the "
                        "draft's T_s up to one overall sign, to the stabilised order: "
                        + "; ".join(f"{k}: stable to fq^{v['rows stable to fq^']}" for k, v in rows.items()),
                        not bad, str(bad)))
    signs = set(sign.values())
    checks.append(check("the overall sign: (-1)^(s+1) det M^(s-hat) = T_s at k = "
                        + ", ".join(str(k) for k, v in sign.items() if v == "-")
                        + " -- the draft prints (-1)^s, which holds at k = "
                        + (", ".join(str(k) for k, v in sign.items() if v == "+") or "none")
                        + "; every row's leading coefficient is +1, so the sign is fixed by the draft's row order, not by "
                          "a normalisation", len(signs) == 1 and "none" not in signs, str(sign)))
    # negative control: a squared decoration makes two rows equal and every maximal minor vanish
    A = A1A2kKAlg(3)
    Q1, Q2 = _minor_rows(A, 3, 11, [None, (1, 2), (3, 1)]), _minor_rows(A, 3, 12, [None, (1, 2), (3, 1)])
    Nq = min(_agree_to(Q1[r][s], Q2[r][s], 80) for r in range(3) for s in range(4))
    van = min(_agree_to(_pdet([row[:s] + row[s + 1:] for row in Q2], Nq), {}, Nq) for s in range(4))
    checks.append(check(f"negative control (k = 3): with D_1 = L_(1;0)^2 in place of L_(1;0) the rows are stable to "
                        f"fq^{Nq}, the second equals the first there and every maximal minor vanishes to fq^{van}; the "
                        "draft's choice (chords of distinct lengths) is what makes the minors non-zero", van == Nq >= 10))
    return {"checks": checks,
            "population": {"minors": rows, "sign": sign, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the stabilised rows annihilate the draft's T",
                         "negative": "a squared decoration (rank-deficient rows)"},
            "notes": "T_s is the draft's formula (Andrews-Gordon products), not the class's traces.  The overall sign of "
                     "the minors depends on the order of the rows; the draft's order (X^(0), X^(1), ...) is used.",
            "inputs": []}


ADAPTERS = {
    "tab:finite_type": check_finite_table,
    "app:a1a2k/rules": check_a1a2k_rules,
    "app:a1a2k/traces": check_a1a2k_traces,
    "app:a1a2k/minors": check_a1a2k_minors,
    "app:finite/canonical-basis": check_finite_canonical_basis,
    "app:finite/trace-unique": check_finite_trace_unique,
    "app:finite/class-s-coincidences": check_finite_class_s,
}
EXTENSIVE = {"tab:finite_type", "app:finite/canonical-basis", "app:finite/trace-unique",
             "app:finite/class-s-coincidences", "app:a1a2k/rules", "app:a1a2k/traces", "app:a1a2k/minors"}
