"""a1dodd_skein.py — frame-free graded Kauffman-skein cross_product for the
[A_1, D_{2k+3}] cone-KAlgebra (the non-diameter puncture fork).

A puncture crossing of two mult-gens `g`, `h` is two chords meeting at TWO points
(`g`'s chord crosses both of `h`'s centrally-symmetric chords).  Skein-resolving
every crossing of the chord set `cg ∪ ch` (Kauffman planar smoothing) and keeping
the **centrally-symmetric** states gives the FZ type-D double-resolution:

  * daughter words = the centrally-symmetric terminal multicurves
    (edges→unit, central chord-pair→mult-gen via the arc↔label bijection);
  * a terminal closed loop ENCIRCLING the puncture (winding ≠ 0) carries the SU(2)
    doublet **χ₁** (the middle-q daughter);
  * q-grading: `q = arc_cocycle(g,h) + 1 − #B`, where `#B` counts the
    central-symmetric crossing CLASSES smoothed the "B" way, with the smoothing
    labelled by **g-chord identity** (rotate the angular order at each crossing to
    start at a g-end) so the bit is ρ-covariant.

VERIFIED entry-for-entry against the decoded k=1 (a1d5) / k=2 (FiniteA1D7) cross
tables over every NON-diameter puncture crossing: k=1 20/20, k=2 140/140
(full `(word, q, χ, coef)`).  Diameter-involved puncture crossings (the quadratic
`T·T`/`T·D`) are handled separately.
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from a1dodd_cone_data import (_ap_chords, _is_polygon_edge, _arcs_to_multgen,
                              arc_cocycle)


def _vpos(v, N):
    th = 2 * math.pi * v / N
    return (math.cos(th), math.sin(th))


def _seg_cross(a, b, c, d, N):
    if len({a, b, c, d}) < 4:
        return None
    (x1, y1), (x2, y2) = _vpos(a, N), _vpos(b, N)
    (x3, y3), (x4, y4) = _vpos(c, N), _vpos(d, N)
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
    if 1e-9 < t < 1 - 1e-9 and 1e-9 < u < 1 - 1e-9:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def _winding(poly):
    if len(poly) < 3:
        return 0
    tot = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        d = math.atan2(y2, x2) - math.atan2(y1, x1)
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        tot += d
    return round(tot / (2 * math.pi))


def _skein_states(chords, ng, N, n):
    """Yield (arcs, n_puncture_loops, n_contractible_loops, n_B_classes) over the
    centrally-symmetric skein states of the chord set.  `chords[:ng]` are g's.
    Smoothing is g-first labelled (ρ-covariant); B = mask bit 1."""
    crs = []
    for i in range(len(chords)):
        for j in range(i + 1, len(chords)):
            p = _seg_cross(*chords[i], *chords[j], N)
            if p:
                crs.append((i, j, p))
    nC = len(crs)
    ckey = {tuple(sorted(c)): idx for idx, c in enumerate(chords)}

    def img(idx):
        a, b = chords[idx]
        return ckey.get(tuple(sorted(((a + n) % N, (b + n) % N))))

    cls_of = [-1] * nC
    classes = []
    for k in range(nC):
        if cls_of[k] != -1:
            continue
        i, j, _ = crs[k]
        ii, jj = img(i), img(j)
        part = None
        for k2 in range(nC):
            if {crs[k2][0], crs[k2][1]} == {ii, jj}:
                part = k2
                break
        cid = len(classes)
        classes.append([k])
        cls_of[k] = cid
        if part is not None and part != k and cls_of[part] == -1:
            classes[cid].append(part)
            cls_of[part] = cid
    nCl = len(classes)

    def chordpts(ci):
        a, b = chords[ci]
        (ax, ay) = _vpos(a, N)
        (bx, by) = _vpos(b, N)
        o = []
        for k, (i, j, pp) in enumerate(crs):
            if ci in (i, j):
                t = (((pp[0] - ax) * (bx - ax) + (pp[1] - ay) * (by - ay))
                     / ((bx - ax) ** 2 + (by - ay) ** 2))
                o.append((t, k))
        o.sort()
        return o

    chordsegs = {}
    for ci, (a, b) in enumerate(chords):
        cps = chordpts(ci)
        pts = [('v', a)] + [('x', k, crs[k][2]) for (t, k) in cps] + [('v', b)]
        chordsegs[ci] = [(pts[s], pts[s + 1]) for s in range(len(pts) - 1)]
    cross_ends = {k: [] for k in range(nC)}
    endpt = {}
    terminals = []
    for ci, segs in chordsegs.items():
        for s, (pA, pB) in enumerate(segs):
            endpt[(ci, s, 0)] = _vpos(pA[1], N) if pA[0] == 'v' else pA[2]
            endpt[(ci, s, 1)] = _vpos(pB[1], N) if pB[0] == 'v' else pB[2]
            (terminals.append(((ci, s, 0), pA[1])) if pA[0] == 'v'
             else cross_ends[pA[1]].append((ci, s, 0)))
            (terminals.append(((ci, s, 1), pB[1])) if pB[0] == 'v'
             else cross_ends[pB[1]].append((ci, s, 1)))
    term_of = dict(terminals)

    for clmask in range(1 << nCl):
        bit = [(clmask >> c) & 1 for c in range(nCl)]
        mask = {k: bit[cls_of[k]] for k in range(nC)}
        nxt = defaultdict(list)

        def add(u, v):
            nxt[u].append(v)
            nxt[v].append(u)

        for ci, segs in chordsegs.items():
            for s in range(len(segs)):
                add((ci, s, 0), (ci, s, 1))
        bad = False
        for k in range(nC):
            i, j, P = crs[k]
            ends = cross_ends[k]
            if len(ends) != 4:
                bad = True
                break

            def ang(e):
                far = endpt[(e[0], e[1], 1 - e[2])]
                return math.atan2(far[1] - P[1], far[0] - P[0])

            es = sorted(ends, key=ang)
            for r in range(4):                       # rotate to a g-end (g-first)
                if es[r][0] < ng:
                    es = es[r:] + es[:r]
                    break
            if mask[k]:
                add(es[1], es[2])
                add(es[3], es[0])
            else:
                add(es[0], es[1])
                add(es[2], es[3])
        if bad:
            continue
        visited = set()
        arcs = []
        loops = []
        for e0, v0 in terminals:
            if e0 in visited:
                continue
            cur = e0
            prev = None
            visited.add(e0)
            while True:
                cand = [x for x in nxt[cur] if x != prev]
                nb = cand[0] if cand else None
                if nb is None:
                    break
                if nb in term_of:
                    visited.add(nb)
                    cur = nb
                    break
                visited.add(nb)
                prev, cur = cur, nb
            if cur in term_of and cur != e0:
                arcs.append(tuple(sorted((v0, term_of[cur]))))
        for e0 in list(nxt):
            if e0 in visited or e0 in term_of:
                continue
            cur = e0
            prev = None
            cyc = []
            while cur not in visited:
                visited.add(cur)
                cyc.append(endpt[cur])
                cand = [x for x in nxt[cur] if x != prev]
                if not cand:
                    break
                prev, cur = cur, cand[0]
            loops.append(cyc)
        npunct = sum(1 for c in loops if _winding(c) != 0)
        ncontr = sum(1 for c in loops if _winding(c) == 0)
        # central-symmetry filter: arcs invariant under x -> x+n
        sarcs = frozenset(tuple(sorted(c)) for c in arcs)
        if frozenset(tuple(sorted(((c[0] + n) % N, (c[1] + n) % N)))
                     for c in sarcs) != sarcs:
            continue
        yield arcs, npunct, ncontr, sum(bit)


def _arcs_word(arcs, n, N, inv):
    nz = [tuple(sorted(c)) for c in arcs
          if not _is_polygon_edge(tuple(sorted(c)), N) and c[0] != c[1]]
    used = set()
    word = []
    for c in nz:
        if c in used:
            continue
        cc = tuple(sorted(((c[0] + n) % N, (c[1] + n) % N)))
        key = frozenset((c, cc))
        if key not in inv:
            return None
        word.append(inv[key])
        used.add(c)
        used.add(cc)
    return tuple(sorted(word))


def puncture_cross_product_nondiam(g, h, k, inv=None):
    """`L_g L_h` for a NON-diameter puncture crossing → list of (word, qexp, chi).

    Frame-free graded skein: q = arc_cocycle(g,h) + 1 − #B (g-first labelled
    smoothing), χ₁ on the puncture-loop daughter (the middle q).  Coefficients +1.
    VERIFIED vs the decoded k=1/k=2 tables (20/20, 140/140)."""
    n = 2 * k + 3
    N = 2 * n
    if inv is None:
        inv = _arcs_to_multgen(k)
    a = arc_cocycle(g, h, k)
    cg = _ap_chords(*g, k=k)
    ch = _ap_chords(*h, k=k)
    out = {}
    for arcs, npunct, ncontr, nB in _skein_states(list(cg) + list(ch), len(cg), N, n):
        w = _arcs_word(arcs, n, N, inv)
        if w is None:
            continue
        q = a + 1 - nB
        chi = 1 if npunct > 0 else 0
        out[(w, q, chi)] = out.get((w, q, chi), 0) + 1
    return [(w, q, chi, c) for (w, q, chi), c in out.items()]


def _bulk_q(g, word, k):
    """q = Σ_factor arc_cocycle(g, factor) — the bulk rule ⟨γ_g, γ_daughter⟩."""
    s = 0
    for f in word:
        s += arc_cocycle(g, f, k)
    return s


def puncture_cross_product_diam_diam(g, h, k, inv=None):
    """`L_g L_h` for a diameter × diameter (`T·T`) puncture crossing → (word,q,χ,coef).

    The a1d3 quadratic `P² + χ₁ PQ + Q²` where `P,Q` are the two reconnections of
    the 4 diameter endpoints (edges→unit), with q via the bulk rule (the diameter
    `(−1)^(x−u)` cocycle) and χ₁ on the PQ cross-term.  VERIFIED vs decoded k=1
    (20/20) / k=2 (42/42).  Pre-condition: `g`, `h` are both diameters and cross."""
    n = 2 * k + 3
    N = 2 * n
    if inv is None:
        inv = _arcs_to_multgen(k)
    (du,), (dv,) = _ap_chords(*g, k=k), _ap_chords(*h, k=k)
    u1 = du[0] if (du[1] - du[0]) % N == n else du[1]
    v1 = dv[0] if (dv[1] - dv[0]) % N == n else dv[1]
    u2, v2 = (u1 + n) % N, (v1 + n) % N
    P = _arcs_word([tuple(sorted((u1, v1))), tuple(sorted((u2, v2)))], n, N, inv)
    Q = _arcs_word([tuple(sorted((u1, v2))), tuple(sorted((u2, v1)))], n, N, inv)
    if P is None or Q is None:
        return None

    def merge(*ws):
        out = []
        for w in ws:
            out += list(w)
        return tuple(sorted(out))

    res = {}
    for word, chi in [(merge(P, P), 0), (merge(P, Q), 1), (merge(Q, Q), 0)]:
        key = (word, _bulk_q(g, word, k), chi)
        res[key] = res.get(key, 0) + 1
    return [(w, q, chi, c) for (w, q, chi), c in res.items()]


def _central(c, n, N):
    return tuple(sorted(((c[0] + n) % N, (c[1] + n) % N)))


def diam_nondiam_pure_daughters(g, h, k, inv=None):
    """The two χ=0 ("pure") daughters of a diameter × non-diameter (`T·D`) puncture
    crossing `L_g L_h` (exactly one of `g`,`h` a diameter), FRAME-FREE via the FZ
    double-resolution: the diameter is the a=k+1 limit of a centrally-symmetric
    pair, so resolve `D × d` (Ptolemy → R₁,R₂) and take each central-consistent
    state `R_i ∪ central(R_i)`, paired into mult-gens.

    q-grading is the diameter's cocycle pairing against the daughter, oriented by
    multiplication side: `q = ε·Σ_f arc_cocycle(diam, f)` with `ε = +1` if the
    diameter is the LEFT operand (`g`), `−1` if the RIGHT (`h`).  This is the
    genuine bulk rule `⟨γ_diam, γ_daughter⟩` — the diameter q-commutes with every
    daughter factor (VERIFIED) so its bulk is the true pairing, and `ε` is the
    cocycle antisymmetry under swapping operand order.  (The non-diameter operand
    need NOT q-commute with the factors, so its bulk would be wrong — hence the
    rule keys on the diameter.)

    Returns the list of (word, qexp).  VERIFIED to reproduce the decoded χ=0
    daughters of every Dx/xD crossing: k=1 60/60, k=2 210/210.  (The χ₁
    middle/merged daughter is the separate puncture-loop term.)"""
    from a1dodd_cone_data import _chord_cross, _resolve_crossing, _is_diameter
    n = 2 * k + 3
    N = 2 * n
    if inv is None:
        inv = _arcs_to_multgen(k)
    gd = _is_diameter(g, k)
    diam, other = (g, h) if gd else (h, g)
    eps = 1 if gd else -1
    (Dc,) = _ap_chords(*diam, k=k)
    dc = next(c for c in _ap_chords(*other, k=k) if _chord_cross(Dc, c, N))
    out = []
    for reso in _resolve_crossing(Dc, dc, N):
        chords = list(reso) + [_central(c, n, N) for c in reso]
        w = _arcs_word(chords, n, N, inv)
        out.append((w, eps * _bulk_q(diam, w, k)))
    return out


def _seg_cross_pos(pa, pb, pc, pd):
    """Geometric segment×segment intersection point (None if disjoint/parallel)."""
    (x1, y1), (x2, y2) = pa, pb
    (x3, y3), (x4, y4) = pc, pd
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
    if 1e-9 < t < 1 - 1e-9 and 1e-9 < u < 1 - 1e-9:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def _doubled_diam_chi_words(diam, other, k, inv, delta=0.05):
    """The χ₁ daughter word(s) of a diameter × non-diameter puncture crossing, via
    the **doubled-diameter graded skein** (the FZ type-D χ₁-doublet / tagged-arc
    made concrete): the diameter is a fork — two strands flanking the puncture — so
    realise it as two infinitesimally perpendicular-offset copies `D₁,D₂` with the
    puncture BETWEEN them.  The non-diameter chord then crosses the diameter at TWO
    points (one per strand); skein-resolving keeps the centrally-symmetric states,
    and the **mixed** smoothing that sends one strand each way produces a loop
    ENCIRCLING the puncture → the SU(2) doublet χ₁ (exactly the geometric mechanism:
    "skein both crossings ⇒ one output loops the puncture ⇒ χ₁").  The *concerted*
    smoothings reproduce the two pure daughters; the mixed state WITHOUT a puncture
    loop is the cabling artifact of the doubling (it neither encircles the puncture
    nor resolves the original single diameter) and is discarded.

    Returns the set of χ₁ daughter words (the puncture-loop states' surviving arcs).
    The δ offset is a regulator — the combinatorics is δ-independent for small δ.
    VERIFIED: FZ pures + these χ₁ words reproduce the FULL decoded Dx/xD entry
    (word,q,χ,coef) at k=1 60/60, k=2 210/210."""
    n = 2 * k + 3
    N = 2 * n
    (u, w), = _ap_chords(*diam, k=k)
    pu, pw = _vpos(u, N), _vpos(w, N)
    dx, dy = pw[0] - pu[0], pw[1] - pu[1]
    L = math.hypot(dx, dy)
    perp = (-dy / L, dx / L)
    D1 = (u, w, (pu[0] + delta * perp[0], pu[1] + delta * perp[1]),
          (pw[0] + delta * perp[0], pw[1] + delta * perp[1]))
    D2 = (u, w, (pu[0] - delta * perp[0], pu[1] - delta * perp[1]),
          (pw[0] - delta * perp[0], pw[1] - delta * perp[1]))
    chords = [D1, D2] + [(a, b, _vpos(a, N), _vpos(b, N))
                         for (a, b) in _ap_chords(*other, k=k)]
    ng = 2
    crs = []
    for i in range(len(chords)):
        for j in range(i + 1, len(chords)):
            p = _seg_cross_pos(chords[i][2], chords[i][3], chords[j][2], chords[j][3])
            if p:
                crs.append((i, j, p))
    nC = len(crs)
    lab2 = defaultdict(list)
    for idx, (a, b, _pa, _pb) in enumerate(chords):
        lab2[frozenset((a, b))].append(idx)

    def img(idx):                                   # central-image chord index
        a, b, _pa, _pb = chords[idx]
        cand = lab2.get(frozenset(((a + n) % N, (b + n) % N)), [])
        for c in cand:                              # prefer the OTHER copy/partner
            if c != idx:
                return c
        return None

    cls_of = [-1] * nC
    classes = []
    for kk in range(nC):
        if cls_of[kk] != -1:
            continue
        i, j, _ = crs[kk]
        ii, jj = img(i), img(j)
        part = None
        for k2 in range(nC):
            if k2 != kk and {crs[k2][0], crs[k2][1]} == {ii, jj}:
                part = k2
                break
        cid = len(classes)
        classes.append([kk])
        cls_of[kk] = cid
        if part is not None and cls_of[part] == -1:
            classes[cid].append(part)
            cls_of[part] = cid
    nCl = len(classes)

    def chordpts(ci):
        pa, pb = chords[ci][2], chords[ci][3]
        o = []
        for kk, (i, j, pp) in enumerate(crs):
            if ci in (i, j):
                t = (((pp[0] - pa[0]) * (pb[0] - pa[0]) + (pp[1] - pa[1]) * (pb[1] - pa[1]))
                     / ((pb[0] - pa[0]) ** 2 + (pb[1] - pa[1]) ** 2))
                o.append((t, kk))
        o.sort()
        return o

    chordsegs = {}
    for ci, (a, b, pa, pb) in enumerate(chords):
        cps = chordpts(ci)
        pts = [('v', a)] + [('x', kk, crs[kk][2]) for (t, kk) in cps] + [('v', b)]
        chordsegs[ci] = [(pts[s], pts[s + 1]) for s in range(len(pts) - 1)]
    cross_ends = {kk: [] for kk in range(nC)}
    endpt = {}
    terminals = []
    for ci, segs in chordsegs.items():
        for s, (pA, pB) in enumerate(segs):
            endpt[(ci, s, 0)] = chords[ci][2] if pA[0] == 'v' else pA[2]
            endpt[(ci, s, 1)] = chords[ci][3] if pB[0] == 'v' else pB[2]
            (terminals.append(((ci, s, 0), pA[1])) if pA[0] == 'v'
             else cross_ends[pA[1]].append((ci, s, 0)))
            (terminals.append(((ci, s, 1), pB[1])) if pB[0] == 'v'
             else cross_ends[pB[1]].append((ci, s, 1)))
    term_of = dict(terminals)
    out = set()
    for clmask in range(1 << nCl):
        bit = [(clmask >> c) & 1 for c in range(nCl)]
        mask = {kk: bit[cls_of[kk]] for kk in range(nC)}
        nxt = defaultdict(list)

        def add(a, b):
            nxt[a].append(b)
            nxt[b].append(a)

        for ci, segs in chordsegs.items():
            for s in range(len(segs)):
                add((ci, s, 0), (ci, s, 1))
        bad = False
        for kk in range(nC):
            i, j, P = crs[kk]
            ends = cross_ends[kk]
            if len(ends) != 4:
                bad = True
                break

            def ang(e):
                far = endpt[(e[0], e[1], 1 - e[2])]
                return math.atan2(far[1] - P[1], far[0] - P[0])

            es = sorted(ends, key=ang)
            for r in range(4):                       # rotate to a diameter-strand end
                if es[r][0] < ng:
                    es = es[r:] + es[:r]
                    break
            if mask[kk]:
                add(es[1], es[2])
                add(es[3], es[0])
            else:
                add(es[0], es[1])
                add(es[2], es[3])
        if bad:
            continue
        visited = set()
        arcs = []
        loops = []
        for e0, v0 in terminals:
            if e0 in visited:
                continue
            cur = e0
            prev = None
            visited.add(e0)
            while True:
                cand = [x for x in nxt[cur] if x != prev]
                nb = cand[0] if cand else None
                if nb is None:
                    break
                if nb in term_of:
                    visited.add(nb)
                    cur = nb
                    break
                visited.add(nb)
                prev, cur = cur, nb
            if cur in term_of and cur != e0:
                arcs.append(tuple(sorted((v0, term_of[cur]))))
        for e0 in list(nxt):
            if e0 in visited or e0 in term_of:
                continue
            cur = e0
            prev = None
            cyc = []
            while cur not in visited:
                visited.add(cur)
                cyc.append(endpt[cur])
                cand = [x for x in nxt[cur] if x != prev]
                if not cand:
                    break
                prev, cur = cur, cand[0]
            loops.append(cyc)
        npunct = sum(1 for c in loops if _winding(c) != 0)
        sarcs = frozenset(tuple(sorted(c)) for c in arcs)
        if frozenset(tuple(sorted(((c[0] + n) % N, (c[1] + n) % N)))
                     for c in sarcs) != sarcs:
            continue
        if npunct < 1:                               # χ₁ ⟺ a puncture-loop state
            continue
        nz = [c for c in sarcs if not _is_polygon_edge(c, N) and c[0] != c[1]]
        used = set()
        word = []
        ok = True
        for c in nz:
            if c in used:
                continue
            cc = tuple(sorted(((c[0] + n) % N, (c[1] + n) % N)))
            if frozenset((c, cc)) not in inv:
                ok = False
                break
            word.append(inv[frozenset((c, cc))])
            used.add(c)
            used.add(cc)
        if ok:
            out.add(tuple(sorted(word)))
    return out


def puncture_cross_product_diam_nondiam(g, h, k, inv=None):
    """`L_g L_h` for a diameter × non-diameter (`T·D`) puncture crossing → the full
    `(word, qexp, chi, coef)` list (exactly one of `g`,`h` a diameter).

    Two χ=0 pure daughters (FZ double-resolution) + one χ₁ daughter (the
    doubled-diameter puncture loop), all at `q = ε·bulk(diam, word)` with `ε=+1`
    if the diameter is the LEFT operand else `−1`.  Coefficients +1.  VERIFIED
    entry-for-entry vs the decoded k=1 (a1d5) / k=2 (FiniteA1D7) tables over every
    Dx/xD crossing: 60/60, 210/210."""
    from a1dodd_cone_data import _is_diameter
    if inv is None:
        inv = _arcs_to_multgen(k)
    gd = _is_diameter(g, k)
    diam, other = (g, h) if gd else (h, g)
    eps = 1 if gd else -1
    res = {}
    for (word, q) in diam_nondiam_pure_daughters(g, h, k, inv):
        res[(word, q, 0)] = res.get((word, q, 0), 0) + 1
    for word in _doubled_diam_chi_words(diam, other, k, inv):
        q = eps * _bulk_q(diam, word, k)
        res[(word, q, 1)] = res.get((word, q, 1), 0) + 1
    return [(w, q, chi, c) for (w, q, chi), c in res.items()]
