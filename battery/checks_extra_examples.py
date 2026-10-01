"""checks_extra_examples.py — the design record adapters for the COMPANION's own examples
(Section 2 of the companion, "further examples from the finite-type
stand-alone realisations"): the u(1)-gauged odd
polygons (comp:u1a1aodd) and K_q([A_1,D_4]) with its SU(3) flavour
(comp:su3ad).  The K_q([A_1,E_6]) example (comp:e6) was withdrawn: its class is the zoo's frozen standalone, and the E-type
flows belong to the companion's RG section.

Method as for the pentagon: every printed rule or series is transcribed
INDEPENDENTLY here and compared with the class; the axioms are the
instruments (run_k_verifiers); cross-presentation checks use an independent
engine (the even family's product for the embedding, the BPS charts for the
traces).
"""
from __future__ import annotations

import itertools
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the release root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from kalgebra import Element  # noqa: E402
from laurent_poly import LaurentPoly  # noqa: E402
from checks_kq import check  # noqa: E402
from checks_coulomb import run_k_verifiers  # noqa: E402


def _lp(e, c=1):
    return LaurentPoly({e: c})


def _lpc(lp) -> dict:
    """The coefficient dict of a LaurentPoly (it stores `_coeffs`)."""
    return dict(getattr(lp, "_coeffs", None) or getattr(lp, "coeffs", {}))


def _shift(el: Element, k: int) -> Element:
    """q^k · element."""
    return Element({lab: v * LaurentPoly({k: 1}) for lab, v in el.terms.items()})


def _add_terms(acc: dict, el: Element, scale: LaurentPoly | None = None):
    for lab, v in el.terms.items():
        w = v if scale is None else v * scale
        acc[lab] = acc[lab] + w if lab in acc else w


def _elem(acc: dict) -> Element:
    return Element({lab: v for lab, v in acc.items() if not v.is_zero()})


# ============================================================================
# comp:u1a1aodd — the u(1)-gauged odd polygons (H = 2k+4), from the write-up
# ============================================================================
E_GEN, E_INV = (0, 0), (0, 1)


def _chord(t, i, H):
    a, b = i % H, (i + t + 1) % H
    return tuple(sorted((a, b)))


def _is_edge(c, H):
    a, b = c
    return (b - a) % H == 1 or (a - b) % H == 1


def _crossing(c1, c2):
    """Two chords (sorted endpoint pairs) cross iff their endpoints are four distinct vertices and interleave."""
    a, b = c1
    c, d = c2
    if len({a, b, c, d}) < 4:
        return False
    return (a < c < b) != (a < d < b)


def _drift(t, i, k):
    """The printed drift d(t, i) of rho (companion eq:comp-drift), letters indexed as the class indexes them."""
    H = 2 * k + 4
    if t == 1 and i == H - 1:
        return 2
    if t % 2 == 1 and 3 <= t <= k and i == H - t - 1:
        return -2
    if t == k + 1 and i == H // 2 - 1:
        return 1
    return 0


def _charge(t, i, k):
    """The printed closed-form charge (eq:comp-charge and the wrap rows), A_{2k+2}-chain coordinates e_1..e_n."""
    H, n = 2 * k + 4, 2 * k + 2
    g = [0] * n
    mu = [1 if j % 2 == 0 else 0 for j in range(n)]

    def add(j, s):
        if 1 <= j <= n:
            g[j - 1] += s
    u, v = i, i + t + 1
    if v <= H - 1:
        for j in range(n + 2 - v, n - 1 - u + 1):
            if (j - v) % 2 == 0:
                add(j, -1)
        if (u - v) % 2 == 0:
            s = -1
            for j in range(n - u, n + 1):
                add(j, s)
                s = -s
        return tuple(g)
    w = i - (H - t - 1)
    r = t - w
    if w == 0 and t % 2 == 1:
        for j in range(1, t + 1, 2):
            add(j, +1)
        s = -1
        for j in range(t + 1, n + 1):
            add(j, s)
            s = -s
    elif r % 2 == 0:
        for j in range(1, r, 2):
            add(j, +1)
        for j in range(n - t + 1 + r, n + 1, 2):
            add(j, +1)
    else:
        for j in range(1, r - 1, 2):
            add(j, -1)
        for j in range(r, n - t + r + 1):
            add(j, -1)
        for j in range(n - t + r + 2, n + 1, 2):
            add(j, -1)
    if t == 1 and w == 1:
        for j in range(n):
            g[j] -= 2 * mu[j]
    return tuple(g)


def _pairing(g, h):
    """The A_n chain pairing <g,h> = sum_i (g_i h_{i+1} - g_{i+1} h_i) (eq:comp-qcomm)."""
    return sum(g[i] * h[i + 1] - g[i + 1] * h[i] for i in range(len(g) - 1))


def _ev(x, y, k):
    """The (H-1)-gon chord on vertices x, y as an even-family letter ((s, j, 1),); edges -> ()."""
    Hp = 2 * k + 3
    x, y = x % Hp, y % Hp
    if (y - x) % Hp == 1 or (x - y) % Hp == 1:
        return ()
    for s in range(1, k + 1):
        if (x + s + 1) % Hp == y:
            return ((s, x, 1),)
        if (y + s + 1) % Hp == x:
            return ((s, y, 1),)
    raise ValueError((x, y))


def _phi_printed(u, v, k):
    """The printed embedding of the chord {u, v}, 0 <= u < v <= H-1: {(even label, (c0, c1)): 1}."""
    H = 2 * k + 4
    ell = v - u
    t = min(ell, H - ell) - 1
    c0 = (-1) ** u if t % 2 == 1 else 0
    if v <= H - 2:
        if 2 <= ell <= H // 2:
            c1 = 0
        elif t % 2 == 0:
            c1 = (-1) ** u
        else:
            c1 = 1 if u == 0 else (-1) ** (u + 1)
        return {(_ev(u, v, k), (c0, c1)): 1}
    c1r = -1 if u == 1 else (1 if 2 <= u <= k else 0)
    return {(_ev(u, 0, k), (c0, c1r)): 1, (_ev(u, H - 2, k), (c0, c1r - 1)): 1}


def _letters(A):
    cd = A.cone_data()
    return [g for g in cd.mult_gens() if g not in (E_GEN, E_INV)], cd


def _lab(g, e=0):
    return (((g[0], g[1], 1),), e)


def check_u1a1aodd_implementation(environment):
    from u1a1aodd_kalg import U1A1AoddKAlg
    checks = []
    t0 = time.time()
    census = {}
    for k in (1, 2, 3):
        A = U1A1AoddKAlg(k)
        chords, cd = _letters(A)
        H = 2 * k + 4
        # letters: every chord (t, i), t = 1..k with i in Z/H, and the diameter t = k+1 with i in Z/(H/2), exactly once
        seen = sorted(chords)
        want = sorted([(t, i) for t in range(1, k + 1) for i in range(H)] + [(k + 1, i) for i in range(H // 2)])
        ok_letters = seen == want and all(cd.geometric_label(g) == _chord(g[0], g[1], H) for g in chords)
        # the drift
        bad_rho = []
        for (t, i) in chords:
            Ht = H if t <= k else H // 2
            exp = (((t, (i + 1) % Ht, 1),), _drift(t, i, k))
            got = A.rho(_lab((t, i)))
            if got != exp:
                bad_rho.append(((t, i), got, exp))
        ok_rho = not bad_rho and A.rho(((), 1)) == ((), -1)
        # a wrong anchor for the +2 drift is detected (negative control)
        wrong = sum(1 for (t, i) in chords if A.rho(_lab((t, i))) == (((t, (i + 1) % (H if t <= k else H // 2), 1),), 2 if (t == 1 and i == 0) else 0)) == len(chords)
        # q-commutation iff non-crossing, exponent = the chain pairing of the printed charges; the products themselves
        n_pairs = n_qc = n_cross = 0
        bad_qc, bad_prod, bad_ptol = [], [], []
        charge = {g: _charge(g[0], g[1], k) for g in chords}
        for g in chords:
            for h in chords:
                if g == h:
                    continue
                n_pairs += 1
                cg, ch = _chord(*g, H), _chord(*h, H)
                qc = not _crossing(cg, ch)
                if qc != cd.q_commute(g, h):
                    bad_qc.append((g, h, "q-commute" if cd.q_commute(g, h) else "cross"))
                    continue
                if qc:
                    n_qc += 1
                    c = _pairing(charge[g], charge[h])
                    if cd.cocycle(g, h) != c:
                        bad_qc.append((g, h, "cocycle", cd.cocycle(g, h), c))
                    pair = (tuple(sorted([(g[0], g[1], 1), (h[0], h[1], 1)])), 0)
                    if A.multiply(_lab(g), _lab(h)) != Element({pair: _lp(c)}):
                        bad_prod.append((g, h))
                else:
                    n_cross += 1
                    a, b, c_, d = sorted(set(cg) | set(ch))
                    res = []
                    for pair in (((a, b), (c_, d)), ((a, d), (b, c_))):
                        res.append(tuple(sorted(x for x in pair if not _is_edge(x, H))))
                    prod = A.multiply(_lab(g), _lab(h))
                    for lab, coef in prod.terms.items():
                        factors, e = lab
                        ms = tuple(sorted(itertools.chain.from_iterable([_chord(t, i, H)] * ex for (t, i, ex) in factors)))
                        unit = len(_lpc(coef)) == 1 and list(_lpc(coef).values())[0] == 1
                        if ms not in res or not unit:
                            bad_ptol.append((g, h, lab, str(coef)))
                    if len(prod.terms) > 2:
                        bad_ptol.append((g, h, "more than two terms"))
        # E q-commutes with every chord and carries the charge mu: the exponent is the pairing with mu, which
        # vanishes exactly on the even types and is +-1 on the odd ones (so E L = q^{+-2} L E there)
        mu_vec = tuple(1 if j % 2 == 0 else 0 for j in range(2 * k + 2))
        ok_E = all(cd.q_commute(g, E_GEN) and cd.cocycle(g, E_GEN) == _pairing(charge[g], mu_vec)
                   and (_pairing(charge[g], mu_vec) == 0) == (g[0] % 2 == 0)
                   and abs(_pairing(charge[g], mu_vec)) <= 1 for g in chords)
        # the printed embedding
        bad_phi = []
        for g in chords:
            u, v = _chord(*g, H)
            exp = _phi_printed(u, v, k)
            got = {lab: coef for lab, coef in cd.phi(g).terms.items()}
            if set(got) != set(exp) or any(str(coef) != "1" for coef in got.values()):
                bad_phi.append((g, sorted(got), sorted(exp)))
        ok_phi = not bad_phi and cd.phi(E_GEN) == Element({((), (0, 1)): _lp(0)}) and cd.phi(E_INV) == Element({((), (0, -1)): _lp(0)})
        census[k] = dict(letters=len(chords), pairs=n_pairs, qcommuting=n_qc, crossing=n_cross, drift_ok=ok_rho, letters_ok=ok_letters,
                         qc_bad=len(bad_qc), prod_bad=len(bad_prod), ptolemy_bad=len(bad_ptol), phi_bad=len(bad_phi), E_ok=ok_E, wrong_anchor_detected=not wrong)
        checks.append(check(f"k={k}: the letters are the chords (t;i), t<=k with i in Z/{H}, and the {H // 2} diameters, plus E^±1; geometric labels as printed", ok_letters))
        checks.append(check(f"k={k}: rho(L_t;i) = E^d(t,i) L_t;i+1 with the printed drift on all {len(chords)} letters, and rho(E) = E^-1", ok_rho, str(bad_rho[:3])))
        checks.append(check(f"k={k}: q-commutation iff non-crossing on all {n_pairs} ordered pairs, exponent = the chain pairing of the printed charges (interior formula and wrap rows, with the -2mu exception), and the {n_qc} q-commuting products are q^c times the canonical pair monomial", not bad_qc and not bad_prod, str((bad_qc[:2], bad_prod[:2]))))
        checks.append(check(f"k={k}: every crossing product ({n_cross} pairs) has at most two terms, each a unit monomial q^a E^e times a resolution of the crossing quadrilateral", not bad_ptol, str(bad_ptol[:3])))
        checks.append(check(f"k={k}: E q-commutes with every chord, its cocycle the pairing with mu: zero exactly on the even types, +-1 on the odd ones", ok_E))
        checks.append(check(f"k={k}: the printed embedding Phi (single term off the merged vertex, two terms on it, coefficients 1) equals the class's letter images", ok_phi, str(bad_phi[:2])))
    # negative control on the charges: without the -2mu exception the cocycles with E at (t, w) = (1, 1) differ
    A = U1A1AoddKAlg(2)
    chords, cd = _letters(A)
    H = 8
    g = (1, H - 2)      # w = i - (H - t - 1) = 6 - 6 = 0? choose w = 1: i = H - t - 1 + 1 = H - 1
    g = (1, H - 1)
    mu = tuple(1 if j % 2 == 0 else 0 for j in range(6))
    without = list(_charge(1, H - 1, 2))
    for j in range(6):
        without[j] += 2 * mu[j]
    checks.append(check("negative control: dropping the -2mu exception at (t, w) = (1, 1) changes the chord's cocycle with a neighbouring chord (k = 2)",
                        any(cd.cocycle(g, h) != _pairing(tuple(without), _charge(h[0], h[1], 2)) for h in chords if h != g and cd.q_commute(g, h))))
    return {"checks": checks, "population": {"k": [1, 2, 3], "census": census, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the printed k = 1 values", "negative": "a wrong drift anchor and the charges without the -2mu exception are detected"},
            "notes": "The write-up's rules transcribed independently (chords, the drift, crossing test, charge formulas, chain pairing, resolutions, Phi) against the class; the products are the class's analytic route.", "inputs": []}


def check_u1a1aodd_is_kalgebra(environment):
    from u1a1aodd_kalg import U1A1AoddKAlg
    checks = []
    t0 = time.time()
    for k in (1, 2):
        A = U1A1AoddKAlg(k)
        chords, cd = _letters(A)
        W = [((), 0), ((), 1), ((), -1)] + [_lab(g) for g in chords]
        # a few products (q-commuting pairs) as labels
        extra = []
        for g in chords:
            for h in chords:
                if g < h and cd.q_commute(g, h):
                    extra.append((tuple(sorted([(g[0], g[1], 1), (h[0], h[1], 1)])), 0))
            if len(extra) >= 3:
                break
        W += extra[:3]
        K = 4
        ver = run_k_verifiers(A, W, K)
        n_fail = sum(f for f, _x in ver.values())
        closure = all(isinstance(c, LaurentPoly) for a in W for b in W for c in A.multiply(a, b).terms.values())
        checks.append(check(f"k={k}: K1-K5 verifiers on all pairs of a window of {len(W)} labels (letters, E^±1, identity, products) at K={K}, and closure over Z[q^±1]",
                            n_fail == 0 and closure, "; ".join(f"{n} fails {f} (first {x})" for n, (f, x) in ver.items() if f)))
    return {"checks": checks, "population": {"k": [1, 2], "K": 4, "seconds": round(time.time() - t0, 1)}, "controls": {"positive": "the identity row", "negative": "n.a."},
            "notes": "All five axioms emergent: the tables are analytic (arc parities, pairing, charges), not built to satisfy them.", "inputs": []}


def check_u1a1aodd_embedding(environment):
    from u1a1aodd_kalg import U1A1AoddKAlg
    checks = []
    t0 = time.time()
    census = {}
    for k in (1, 2, 3):
        if k == 3 and time.time() - t0 > 120 and environment != "local":
            census[k] = "skipped (web budget)"
            continue
        A = U1A1AoddKAlg(k)
        chords, cd = _letters(A)
        aux = cd.aux()
        letters = chords + [E_GEN, E_INV]
        labs = {g: (_lab(g) if g not in (E_GEN, E_INV) else ((), 1 if g == E_GEN else -1)) for g in letters}
        n_ok = n_bad = 0
        first = ""
        for g in letters:
            for h in letters:
                lhs = aux.multiply_elements(cd.phi(g), cd.phi(h))
                prod = A.multiply(labs[g], labs[h])
                acc = {}
                for c, coef in prod.terms.items():
                    _add_terms(acc, cd.phi_label(c), coef)
                rhs = _elem(acc)
                if lhs == rhs:
                    n_ok += 1
                else:
                    n_bad += 1
                    first = first or f"k={k} {g}·{h}: lhs {lhs} vs rhs {rhs}"[:300]
        census[k] = dict(pairs=n_ok + n_bad, bad=n_bad)
        checks.append(check(f"k={k}: Phi(g)Phi(h) = sum_c C^c_gh Phi(c) on all {n_ok + n_bad} ordered letter pairs, the left side in A^even ⊗ Q_q(Z^2) by the even algebra's own product", n_bad == 0, first))
    return {"checks": checks, "population": {"census": census, "seconds": round(time.time() - t0, 1)}, "controls": {"positive": "the identity pairs", "negative": "n.a."},
            "notes": "The even family A1A2kKAlg(k) and the rank-2 quantum torus are independent algebras; the author's experiment covers k = 1..6.", "inputs": []}


def _singlet(p, n, K):
    """The printed Tr(E^n) (eq:comp-singlet) as {exp: coeff}."""
    na = abs(n)
    sign = 1 if ((p - 1) * na) % 2 == 0 else -1
    out = {}
    for s, b, c in ((+1, 2 * p * na + 2 * p - 2, (p - 1) * na), (-1, 2 * p * na + 2 * p + 2, (p + 1) * na + 2)):
        j = 0
        while 2 * p * j * j + b * j + c <= K:
            e = 2 * p * j * j + b * j + c
            out[e] = out.get(e, 0) + sign * s
            j += 1
    return {e: v for e, v in out.items() if v}


def _long_chord(p, n, K, swap=False):
    """The printed Tr(L_2;i E^n), n >= 0 (eq:comp-longchord) as {exp: coeff}; swap=True exchanges the odd/even columns (negative control)."""
    odd = (p % 2 == 1) != swap
    overall = 1 if odd else (-1) ** (n + 1)
    branches = [
        (+1, 2 * p * n + (2 * p + 4 if odd else 2 * p + 2), ((p + 1) if odd else (p - 1)) * n + (3 if odd else 1)),
        (-1, 2 * p * n + (2 * p + 2 if odd else 2 * p + 4), ((p - 1) if odd else (p + 1)) * n + (1 if odd else 3)),
        (+1, 2 * p * n + 2 * p + 10, 3 * (p - 1) * n + (4 * p - 3 if odd else 4 * p - 5)),
        (-1, 2 * p * n + 2 * p + 8, 3 * (p - 1) * n + (4 * p - 5 if odd else 4 * p - 3)),
    ]
    out = {}
    for s, b, c in branches:
        j = 0
        while 2 * p * j * j + b * j + c <= K:
            e = 2 * p * j * j + b * j + c
            out[e] = out.get(e, 0) + overall * s
            j += 1
    return {e: v for e, v in out.items() if v}


def _chat(p, r, s, K, shift):
    """shift + the printed false-theta numerator chat_{r,s} (eq:comp-longchord) as {exp: coeff}, exponents <= K."""
    out = {}
    if s == 0:
        return out
    for lin, const, sg in ((p * r - s, 0, 1), (p * r + s, 2 * r * s, -1)):
        vertex = max(0, -lin // (2 * p) + 1)
        j = 0
        while True:
            e = shift + const + 2 * p * j * j + 2 * j * lin
            if e <= K:
                out[e] = out.get(e, 0) + sg
            elif j >= vertex:
                break
            j += 1
    return out


def _chord_rule(p, m, n, K):
    """The printed Tr(L_{2m;i} E^e) (eq:comp-longchord) at character index n, as {exp: coeff}."""
    r = n + 1
    shift = (p - 1 - 2 * m) * n - m
    sign = -1 if ((p - 1) * n + m + 1) % 2 else 1
    out = {}
    for s, sigma in ((m, 1), (m + 1, -1)):
        for e, c in _chat(p, r, s, K, shift).items():
            out[e] = out.get(e, 0) + sign * sigma * c
    out = {e: v for e, v in out.items() if v}
    assert not out or min(out) >= 0, ("negative power in the printed chord rule", p, m, n)
    return out


def _lp_dict(x, K):
    """A LaurentPoly / RPowerSeries-like trace value -> {exp: int} through q^K."""
    if hasattr(x, "_coeffs"):
        return {e: v for e, v in _lpc(x).items() if v and e <= K}
    if hasattr(x, "coeffs") and isinstance(x.coeffs, dict) and all(isinstance(v, int) for v in x.coeffs.values()):
        return {e: v for e, v in x.coeffs.items() if v and e <= K}
    out = {}
    for e, v in getattr(x, "coeffs", {}).items():
        if e > K:
            continue
        terms = getattr(v, "terms", None)
        n = int(v) if isinstance(v, int) else (sum(terms.values()) if terms else 0)
        if n:
            out[e] = n
    return out


def check_u1a1aodd_traces(environment):
    from u1a1aodd_kalg import U1A1AoddKAlg
    import u1_pgon_layer2 as gp
    checks = []
    t0 = time.time()
    K = 40
    # the printed rule is the served code, deep: every p = 3..7, every even type, n = -4..4, through q^80
    Kd = 80
    mism = [(p, m, n) for p in range(3, 8) for m in range(0, (p + 1) // 2) for n in range(-4, 5)
            if _chord_rule(p, m, n, Kd) != {e: c for e, c in gp.singlet_chord_trace(p, m, n, Kd)._coeffs.items() if c}]
    checks.append(check(f"the printed rule eq:comp-longchord equals the served singlet_chord_trace for p = 3..7, every even type, n = -4..4, through q^{Kd}; at m = 0 it is eq:comp-singlet",
                        not mism and all(_chord_rule(p, 0, n, Kd) == _singlet(p, n, Kd) for p in range(3, 8) for n in range(-4, 5)), str(mism[:3])))
    # negative control: the retired four-partial-theta form printed before 2026-09-23 differs, from q^39 at p = 4
    first = sorted(e for e in set(_long_chord(4, 0, 60)) | set(_chord_rule(4, 1, 0, 60))
                   if _long_chord(4, 0, 60).get(e, 0) != _chord_rule(4, 1, 0, 60).get(e, 0))
    checks.append(check("negative control: the retired printed long chord differs from the rule, first at q^39 (p = 4, n = 0)", bool(first) and first[0] == 39, str(first[:3])))
    for k in (1, 2, 3):
        p = k + 2
        A = U1A1AoddKAlg(k)
        chords, cd = _letters(A)
        bad = []
        for n in range(-3, 4):
            got = _lp_dict(A.trace(((), n), K), K)
            if got != _singlet(p, n, K):
                bad.append(("E^%d" % n, got, _singlet(p, n, K)))
        checks.append(check(f"k={k} (p={p}): Tr(E^n) = the printed singlet partial-theta characters for n = -3..3 through q^{K}", not bad, str(bad[:2])))
        # every even-type chord L_{2m;i} E^e: the printed rule eq:comp-longchord at the character index
        # n = -(-1)^i g, g the seed's gauge charge (the e_1 component of the printed chord charge plus e)
        bad, tested = [], 0
        for t in range(2, k + 2, 2):
            m = t // 2
            for i in range(0, 2):
                g_chord = _charge(t, i, k)[0]
                for e in range(-4, 5):
                    n = -((-1) ** i) * (g_chord + e)
                    try:
                        got = _lp_dict(A.trace(_lab((t, i), e), K), K)
                    except Exception as ex:  # noqa: BLE001
                        bad.append((t, i, e, f"{type(ex).__name__}: {str(ex)[:80]}"))
                        continue
                    tested += 1
                    if got != _chord_rule(p, m, n, K):
                        bad.append((t, i, e, n, got, _chord_rule(p, m, n, K)))
        checks.append(check(f"k={k} (p={p}): Tr(L_2m;i E^e) = the printed rule eq:comp-longchord at n = -(-1)^i g, every even type, i = 0, 1, e = -4..4 ({tested} seeds) through q^{K}",
                            not bad and tested > 0, str(bad[:2])))
        # the other positions, as printed: Tr o rho^2 = Tr, rho^2 carrying L_{2m;i} E^e to
        # L_{2m;i+2} E^{e + d(2m,i+1) - d(2m,i)} (eq:comp-drift); so walk each position down to i = 0, 1
        H, Kp = 2 * k + 4, 12
        bad, tested, face_fail = [], 0, []
        for t in range(2, k + 2, 2):
            m = t // 2
            for i in range(2, H):
                if t == k + 1 and i >= H // 2:          # the diameter: L_{k+1;i} = L_{k+1;i+H/2}
                    continue
                for e in range(-2, 3):
                    j, ej = i, e
                    while j >= 2:
                        ej -= _drift(t, j - 1, k) - _drift(t, j - 2, k)
                        j -= 2
                    n = -((-1) ** j) * (_charge(t, j, k)[0] + ej)
                    got = _lp_dict(A.trace(_lab((t, i), e), Kp), Kp)
                    tested += 1
                    if got != _chord_rule(p, m, n, Kp):
                        bad.append((t, i, e))
                    n_face = -((-1) ** i) * (_charge(t, i, k)[0] + e)
                    if got != _chord_rule(p, m, n_face, Kp):
                        face_fail.append((t, i))
        checks.append(check(f"k={k} (p={p}): at the other positions, Tr(L_2m;i E^e) = the printed rule at i = 0, 1 after the rho^2 steps of eq:comp-drift ({tested} seeds, e = -2..2, through q^{Kp})",
                            not bad and tested > 0, str(bad[:3])))
        if k == 1:
            checks.append(check("negative control (k=1): read at face value at i = 2 (the chord {2,5}), the index rule fails, as the text says",
                                (2, 2) in face_fail, str(sorted(set(face_fail))[:5])))
        # odd-type single chords vanish
        odd = [g for g in chords if g[0] % 2 == 1]
        bad = [g for g in odd if _lp_dict(A.trace(_lab(g), K), K)]
        checks.append(check(f"k={k}: Tr(L_t;i) = 0 for every odd-type chord ({len(odd)} letters; gauge-charged rho^2-orbits)", not bad, str(bad[:3])))
    return {"checks": checks, "population": {"k": [1, 2, 3], "K": K, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the printed rule equals the served code through q^80", "negative": "the retired printed form differs from q^39"},
            "notes": "The rule eq:comp-longchord transcribed from the companion (2026-09-23 version); the class serves the seeds through u1_pgon_layer2.singlet_chord_trace.", "inputs": []}


# ============================================================================
# comp:su3ad — K_q([A_1, D_4]) with SU(3) flavour
# ============================================================================
def _su3_relations(i):
    """The printed relations at orbit index i (i even; for odd i the chi labels are swapped): list of (lhs pair, [(q-power, letters, (p,q), coeff)])."""
    T, D = ("T",), ("D",)
    rel = []
    rel.append(((("T", i), ("T", i + 1)), [(0, {}, (0, 0), 1), (-1, {("D", i): 1}, (0, 1), 1), (-2, {("D", i): 2}, (1, 0), 1), (-3, {("D", i): 3}, (0, 0), 1)]))
    rel.append(((("T", i + 1), ("T", i)), [(0, {}, (0, 0), 1), (1, {("D", i): 1}, (0, 1), 1), (2, {("D", i): 2}, (1, 0), 1), (3, {("D", i): 3}, (0, 0), 1)]))
    rel.append(((("D", i), ("D", i + 1)), [(0, {}, (0, 0), 1), (-1, {("T", i + 1): 1}, (0, 0), 1)]))
    rel.append(((("D", i + 1), ("D", i)), [(0, {}, (0, 0), 1), (1, {("T", i + 1): 1}, (0, 0), 1)]))
    rel.append(((("D", i), ("D", i + 2)), [(-1, {("D", i + 1): 1}, (0, 0), 1), (0, {}, (0, 1), 1), (1, {("D", i - 1): 1}, (0, 0), 1)]))
    rel.append(((("T", i), ("D", i + 1)), [(-2, {("D", i): 2}, (0, 0), 1), (-1, {("D", i): 1}, (1, 0), 1), (0, {}, (0, 1), 1), (1, {("D", i - 1): 1}, (0, 0), 1)]))
    rel.append(((("D", i + 1), ("T", i)), [(-1, {("D", i - 1): 1}, (0, 0), 1), (0, {}, (0, 1), 1), (1, {("D", i): 1}, (1, 0), 1), (2, {("D", i): 2}, (0, 0), 1)]))
    rel.append(((("T", i), ("D", i + 2)), [(-1, {("D", i): 1}, (0, 0), 1), (0, {}, (1, 0), 1), (1, {("D", i - 1): 1}, (0, 1), 1), (2, {("D", i - 1): 2}, (0, 0), 1)]))
    rel.append(((("D", i + 2), ("T", i)), [(-2, {("D", i - 1): 2}, (0, 0), 1), (-1, {("D", i - 1): 1}, (0, 1), 1), (0, {}, (1, 0), 1), (1, {("D", i): 1}, (0, 0), 1)]))
    rel.append(((("T", i), ("T", i + 2)), [(-3, {("T", i + 1): 1}, (0, 0), 1), (-2, {}, (0, 0), 1), (-1, {("D", i + 1): 1}, (1, 0), 1), (-1, {("D", i): 1}, (0, 1), 1),
                                            (0, {}, (0, 0), 2), (0, {}, (1, 1), 1), (1, {("D", i + 2): 1}, (0, 1), 1), (1, {("D", i - 1): 1}, (1, 0), 1), (2, {}, (0, 0), 1), (3, {("T", i - 1): 1}, (0, 0), 1)]))
    return rel


def check_su3ad_implementation(environment):
    from su3_ad_kalg import SU3ADKAlg, _T, _D, _monomial_to_label
    checks = []
    t0 = time.time()
    A = SU3ADKAlg()
    gen = lambda kind, j: A.T(j % 4) if kind == "T" else A.D(j % 4)   # noqa: E731
    letter = lambda kind, j: _T(j % 4) if kind == "T" else _D(j % 4)   # noqa: E731

    def expected(terms, swap):
        acc = {}
        for e, letters, pq, coeff in terms:
            p, q = (pq[1], pq[0]) if swap else pq
            lab = _monomial_to_label({letter(kind, j): ex for (kind, j), ex in letters.items()}, (p, q), 0)[0] if letters else (0, 0, 0, p, q)
            acc[lab] = acc.get(lab, LaurentPoly({})) + LaurentPoly({e: coeff})
        return Element({lab: v for lab, v in acc.items() if not v.is_zero()})
    bad, n_rel = [], 0
    bad_noswap = 0
    for i in range(4):
        for (l1, l2), terms in _su3_relations(i):
            n_rel += 1
            got = A.multiply(gen(*l1), gen(*l2))
            if got != expected(terms, swap=(i % 2 == 1)):
                bad.append((i, l1, l2, str(got)[:120]))
            if i % 2 == 1 and got != expected(terms, swap=False) and any(pq != (0, 0) and pq != (1, 1) for _e, _l, pq, _c in terms):
                bad_noswap += 1
    checks.append(check(f"the ten printed relation families hold at every i (the {n_rel} products), with the chi labels swapped at odd i", not bad, str(bad[:3])))
    checks.append(check("negative control: at odd i the relations WITHOUT the star swap fail wherever a non-self-conjugate character appears", bad_noswap > 0, f"{bad_noswap} of the odd-i relations distinguish the swap"))
    # the two q-commutations
    ok_qc = True
    for i in range(4):
        TD, DT = A.multiply(A.T(i), A.D(i)), A.multiply(A.D(i), A.T(i))
        ok_qc &= TD == _shift(DT, -2)
        TD1, D1T = A.multiply(A.T(i), A.D((i - 1) % 4)), A.multiply(A.D((i - 1) % 4), A.T(i))
        ok_qc &= TD1 == _shift(D1T, 2)
    checks.append(check("T_i D_i = q^-2 D_i T_i and T_i D_{i-1} = q^2 D_{i-1} T_i for all i", ok_qc))
    # rho
    ok_rho = all(A.rho(A.T(i)) == A.T((i + 1) % 4) and A.rho(A.D(i)) == A.D((i + 1) % 4) for i in range(4)) and A.rho(A.chi(1, 0)) == A.chi(0, 1) and A.rho(A.chi(0, 1)) == A.chi(1, 0)
    ok_order = all(A.rho(A.rho(A.rho(A.rho(x)))) == x for x in [A.T(0), A.D(0)])
    checks.append(check("rho(T_i) = T_{i+1}, rho(D_i) = D_{i+1}, rho(chi_(1,0)) = chi_(0,1) (the star), rho of order 4 on the generators", ok_rho and ok_order))
    return {"checks": checks, "population": {"relations": n_rel, "seconds": round(time.time() - t0, 1)}, "controls": {"positive": "the printed relations", "negative": "the relations without the star swap at odd i fail"},
            "notes": "The constant term of T_i T_{i+2} is 2 + chi_(1,1) (six roots and two singlets); the class's docstring block once said 4 + 2 chi_(1,1).", "inputs": []}


def check_su3ad_is_kalgebra(environment):
    from su3_ad_kalg import SU3ADKAlg, _T, _D, _monomial_to_label
    checks = []
    t0 = time.time()
    A = SU3ADKAlg()
    W = [A.identity()] + [A.T(i) for i in range(4)] + [A.D(i) for i in range(4)] + [A.chi(1, 0), A.chi(0, 1)]
    W += [_monomial_to_label({_T(i): 1, _D(i): 1}, (0, 0), 0)[0] for i in range(4)]
    W = list(dict.fromkeys(W))
    K = 4
    ver = run_k_verifiers(A, W, K)
    n_fail = sum(f for f, _x in ver.values())
    checks.append(check(f"K1-K5 verifiers on all pairs of a window of {len(W)} labels (identity, the eight generators, chi_(1,0), chi_(0,1), the four T_i D_i tiles) at K={K}", n_fail == 0,
                        "; ".join(f"{n} fails {f} (first {x})" for n, (f, x) in ver.items() if f)))
    f5 = getattr(A, "verify_trace_intertwines_rho_star", None)
    if f5 is not None:
        try:
            ok5 = all(f5(a, K=K) for a in W)
        except TypeError:
            ok5 = all(f5(a) for a in W)
        checks.append(check(f"F5: Tr L_rho(a) = rho(Tr L_a) with the outer star on R(SU(3)), on the window at K={K}", ok5))
    return {"checks": checks, "population": {"labels": len(W), "K": K, "seconds": round(time.time() - t0, 1)}, "controls": {"positive": "the identity row", "negative": "n.a."}, "notes": "Emergent: the relations are static data, the trace BPS-free.", "inputs": []}


def _relt_dict(v):
    terms = getattr(v, "terms", None)
    return {tuple(b): int(c) for b, c in terms.items() if c} if terms else ({(): int(v)} if v else {})


def _su3_restricted(char, K=None):
    """An R(SU(3)) element {(p, q): c} (or a {q: {(p, q): c}} series) restricted to SU(2) x U(1) along the printed
    branching 3 -> 2_{+1} + 1_{-2}: the Dynkin weight (m1, m2) carries SU(2) weight -m1 and U(1) charge m1 + 2 m2
    (the fundamental's weights (1,0), (-1,1), (0,-1) give (-1, 1), (1, 1), (0, -2))."""
    import sl3_su3_traces as S3
    out = {}
    for hw, c in char.items():
        for (m1, m2), mult in S3.weight_diagram(tuple(hw)).items():
            key = (-m1, m1 + 2 * m2)
            out[key] = out.get(key, 0) + c * mult
    return {w: c for w, c in out.items() if c}


def check_su3ad_traces(environment):
    from su3_ad_kalg import SU3ADKAlg, _T, _D, _monomial_to_label
    from su3_bps_kalgebra import _user_quiver
    import sl3_su3_traces as S3
    checks = []
    t0 = time.time()
    A = SU3ADKAlg()
    K = 6
    tr1, trT, trD = A.trace(A.identity(), K=K), A.trace(A.T(0), K=K), A.trace(A.D(0), K=K)
    # the printed expansions (basis labels (p, q) = chi_(p,q); (0, 0) the identity of R(SU(3)))
    want1 = {0: {(0, 0): 1}, 2: {(1, 1): 1}, 4: {(0, 0): 1, (1, 1): 1, (2, 2): 1}, 6: {(0, 0): 1, (0, 3): 1, (1, 1): 2, (2, 2): 1, (3, 0): 1, (3, 3): 1}}
    wantT = {1: {(0, 0): -1}, 5: {(1, 1): -1}}
    wantD = {1: {(1, 0): -1}, 3: {(2, 1): -1}, 5: {(0, 2): -1, (1, 0): -1, (2, 1): -1, (3, 2): -1}}
    canon = lambda ps: {e: _relt_dict(v) for e, v in ps.coeffs.items() if _relt_dict(v)}   # noqa: E731
    checks.append(check("Tr 1, Tr T_0, Tr D_0 through q^6 equal the printed expansions", canon(tr1) == want1 and canon(trT) == wantT and canon(trD) == wantD,
                        f"Tr1 {tr1}; TrT {trT}; TrD {trD}"))
    # the served elementary traces are the even-D k = 1 closed forms: the printed ungauged trace (comp:a1deven) of the
    # generator's image, every gauged trace from the printed tower and seed rules (transcribed there), a term
    # chi_kappa z^f standing for U(1) charge 3f + Delta; restricted to SU(2) x U(1), which is injective on class functions
    Kt = 16
    images = {"1": ((), 0, A.identity()), "T_0": (((((0, 4), 1), ((1, 2), 1))), 0, A.T(0)), "D_0": (((((1, 3), 1),)), 1, A.D(0))}

    def transcribed(curves, delta):
        out = {}
        for q, row in _dv_ungauged(1, curves, Kt).items():
            d = {}
            for (m, f), c in row.items():
                d[(m, 3 * f + delta)] = d.get((m, 3 * f + delta), 0) + c
            d = {w: c for w, c in d.items() if c}
            if d:
                out[q] = d
        return out

    def served(label):
        tr = A.trace(label, K=Kt)
        return {e: _su3_restricted(_relt_dict(v)) for e, v in tr.coeffs.items() if e <= Kt and _relt_dict(v)}

    got = {name: served(lab) for name, (_c, _d, lab) in images.items()}
    ok_tr = {name: got[name] == transcribed(curves, delta) for name, (curves, delta, _lab) in images.items()}
    checks.append(check(f"the served Tr 1, Tr T_0, Tr D_0, restricted to SU(2) x U(1), are the even-D k = 1 formulas of comp:a1deven through q^{Kt}: the ungauged trace of the images (), {{(0,4),(1,2)}}, {{(1,3)}} with offsets Delta = 0, 0, 1 (the gauge tower for Tr 1, the seed rules for T_0, D_0; transcribed from the text)",
                        all(ok_tr.values()), str(ok_tr)))
    neg = {"D_0 offset + 3": got["D_0"] != transcribed(images["D_0"][0], 4),
           "D_1's image {(2,3)} for D_0": got["D_0"] != transcribed(((((2, 3), 1),)), 1),
           "z of U(1) charge -3": got["T_0"] != {q: {(m, -y): c for (m, y), c in d.items()} for q, d in transcribed(images["T_0"][0], 0).items()} or
           got["D_0"] != {q: {(m, 2 - y): c for (m, y), c in d.items()} for q, d in transcribed(images["D_0"][0], 1).items()}}
    checks.append(check("negative controls: D_0's offset moved by 3, D_1's image in place of D_0's, and z read with U(1) charge -3 each disagree with the served traces", all(neg.values()), str(neg)))
    # the witnesses are compared on the served seed series in Cartan fugacities (the series the class symmetrizes into
    # SU(3) characters, as tests/test_su3ad_deven_map.py compares them; the symmetrization at q^100 would cost more than
    # the witnesses); first those series are tied to the class's traces
    def upto(series, KK):
        return {q: z for q, z in series.items() if q <= KK and z}

    prov = S3._ClosedFormSeeds().ensure(100)
    Kc = 24
    tie = all({e: {hw: c for hw, c in S3.sym_to_char(z).items() if c} for e, z in upto(prov.series(key), Kc).items()} == canon(A.trace(lab, K=Kc))
              for key, lab in ((("Tr_1",), A.identity()), (("Tr_T",), A.T(0)), (("Tr_D", 0), A.D(0))))
    checks.append(check(f"the served seed series (Cartan fugacities), symmetrized, are the class's Tr 1, Tr T_0, Tr D_0 through q^{Kc}", tie))
    # witness 1: the Kac-Wakimoto vacuum character of sl(3) at level -3/2, through q^100
    K1 = 100
    tw = time.time()
    kw = {2 * g: S3.char_to_zlaurent(ch) for g, ch in S3.vacuum_character(K1 // 2 + 2).items() if 2 * g <= K1}
    tw = round(time.time() - tw, 1)
    ok1 = upto(prov.series(("Tr_1",)), K1) == upto(kw, K1)
    checks.append(check(f"witness: the served Tr 1 equals the Kac-Wakimoto vacuum character of sl(3) at level -3/2 through q^{K1} (witness {tw} s)", ok1))
    # witness 2: the orthonormality bootstrap SU3ElemTraces seeded by that character, through q^80
    K2 = 80
    tw = time.time()
    wit = S3.SU3ElemTraces().ensure(K2)
    tw = round(time.time() - tw, 1)
    ok2 = {key: upto(prov.series(key), K2) == upto(wit.series(key), K2) for key in (("Tr_T",), ("Tr_D", 0), ("Tr_D", 1))}
    checks.append(check(f"witness: the served Tr T_0 and Tr D_0 (both parities) equal the orthonormality bootstrap SU3ElemTraces through q^{K2} (witness {tw} s)", all(ok2.values()), str(ok2)))
    # cross-presentation: the BPS chart's Tr 1
    B = _user_quiver()
    trB = B.trace(B.identity(), K=K)
    same = canon(trB) == canon(tr1)
    checks.append(check("Tr 1 of the BPS chart SU3BPSKAlgebra (the exact Habiro engine) equals the served Tr 1 through q^6", same, f"BPS: {trB}"))
    # the bootstrap's defining constraint on the served traces: Tr(T_0^a), Tr(D_0^a) = O(q^a)
    ok_val = True
    detail = []
    for a in (1, 2, 3):
        for L in (_T(0), _D(0)):
            tr = A.trace(_monomial_to_label({L: a}, (0, 0), 0)[0], K=K)
            v = min((e for e, c in tr.coeffs.items() if _relt_dict(c)), default=None)
            detail.append((L, a, v))
            ok_val &= (v is None or v >= a)
    checks.append(check("Tr(T_0^a) and Tr(D_0^a) are O(q^a) for a = 1, 2, 3 (the constraint the bootstrap deconvolves)", ok_val, str(detail)))
    # negative control: perturbing the vacuum character at q^2 breaks the BPS comparison
    checks.append(check("negative control: the BPS chart's Tr 1 has chi_(1,1) at q^2, so a vacuum character without the adjoint current would fail", canon(trB).get(2) == {(1, 1): 1}))
    return {"checks": checks, "population": {"printed heads K": K, "even-D formulas K": Kt, "Kac-Wakimoto witness K": K1, "bootstrap witness K": K2,
                                             "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the printed heads; the served Tr 1 against the Kac-Wakimoto character",
                         "negative": "D_0's offset moved, D_1's image in place of D_0's, z with the opposite U(1) charge; a vacuum character without the adjoint at q^2 fails the BPS comparison"},
            "notes": "The served route (since 2026-09-24): the even-D k = 1 closed forms through the curve map, summed over the gauge charge with the measure restored (sl3_su3_traces), checked against the formulas of comp:a1deven transcribed here; the Kac-Wakimoto character (Tr 1) and the forward bootstrap SU3ElemTraces (Tr T_0, Tr D_0) are the witnesses, at the depths the companion prints; the BPS chart's Tr 1 is a third, at low order.",
            "inputs": []}


# ============================================================================
# comp:a1dodd — K_q([A_1,D_5]) and K_q([A_1,D_7]): the paper's [A_1,D_3], continued
# ============================================================================
# (zoo module, literal prefix, v with k + 2 = 2/v, number of seeds, hand-written module, class)
_DODD = {"a1d5": ("finite_a1d5_kalg", "A1D5", 5, 4, "a1d5_kalg", "A1D5KAlg"),
         "a1d7": ("finite_a1d7_kalg", "A1D7", 7, 6, "a1d7_kalg", "A1D7KAlg")}

# The printed seed recipes, transcribed from the companion: each term is
# (building block, s, power of q, dressed by chi_1, sign), the building blocks
# being "vac" (kappa_0), "sym" (kappa_s^sym) and "anti" (kappa_s^anti).
_DODD_RECIPES = {
    3: {"T": [("vac", 0, -1, False, 1), ("sym", 1, -1, False, -1), ("anti", 1, -1, True, -1)],
        "D": [("anti", 1, -1, False, -1)]},
    5: {0: [("vac", 0, 0, False, 1), ("sym", 1, -2, False, 1), ("sym", 2, -2, False, -1), ("anti", 2, -2, True, -1)],
        1: [("vac", 0, -1, False, 1), ("sym", 2, -1, False, -1), ("anti", 2, -1, True, -1)],
        2: [("anti", 1, -1, False, -1), ("anti", 2, -1, False, -1)],
        3: [("anti", 1, -2, False, -1)]},
    7: {0: [("vac", 0, -1, False, 1), ("sym", 1, -3, False, 1), ("sym", 2, -3, False, -1), ("anti", 2, -3, True, -1),
            ("sym", 3, -1, False, -1), ("anti", 3, -1, True, -1)],
        1: [("vac", 0, 0, False, 1), ("sym", 1, -2, False, 1), ("sym", 3, -2, False, -1), ("anti", 3, -2, True, -1)],
        2: [("vac", 0, -1, False, 1), ("sym", 3, -1, False, -1), ("anti", 3, -1, True, -1)],
        3: [("anti", 1, -1, False, -1), ("anti", 2, -1, False, -1), ("anti", 3, -1, False, -1)],
        4: [("anti", 1, -2, False, -1), ("anti", 2, -2, False, -1)],
        5: [("anti", 2, -3, False, -1)]},
}


def _qm_add(acc, e, m, c):
    if c:
        row = acc.setdefault(e, {})
        row[m] = row.get(m, 0) + c


def _qm_clean(d):
    return {e: {m: c for m, c in row.items() if c} for e, row in d.items() if any(row.values())}


def _qm_mul(a, b, K):
    out = {}
    for ea, ra in a.items():
        for eb, rb in b.items():
            if ea + eb <= K:
                for ma, ca in ra.items():
                    for mb, cb in rb.items():
                        _qm_add(out, ea + eb, ma + mb, ca * cb)
    return _qm_clean(out)


def _qm_verma(K):
    """V = prod_{m>=1} 1/[(1 - q^{2m})(1 - q^{2m} mu^2)(1 - q^{2m} mu^{-2})], through q^K."""
    out = {0: {0: 1}}
    m = 1
    while 2 * m <= K:
        for f in (0, 2, -2):
            geo = {2 * m * j: {f * j: 1} for j in range(K // (2 * m) + 1)}
            out = _qm_mul(out, geo, K)
        m += 1
    return out


def _qm_sigma(v, s, K, u=2):
    """sigma_s = sum_j [ q^{2uv j^2 + (4us - 2v) j} mu^{2uj} - q^{2uv j^2 + (4us - 6v) j + 2v - 2us} mu^{2uj - 2} ]."""
    out = {}
    jb = int((K / (2 * u * v)) ** 0.5) + 8
    for j in range(-jb, jb + 1):
        e1 = 2 * u * v * j * j + (4 * u * s - 2 * v) * j
        if 0 <= e1 <= K:
            _qm_add(out, e1, 2 * u * j, 1)
        e2 = 2 * u * v * j * j + (4 * u * s - 6 * v) * j + 2 * v - 2 * u * s
        if 0 <= e2 <= K:
            _qm_add(out, e2, 2 * u * j - 2, -1)
    return _qm_clean(out)


def _mu_div(row, kind):
    """Exact division of a mu-Laurent polynomial by (1 - mu^2) or by (mu - mu^{-1})."""
    f = dict(row)
    if not f:
        return {}
    g = {}
    if kind == "1-mu2":           # f_k = g_k - g_{k-2}: solve upward
        lo, hi = min(f), max(f)
        for k in range(lo, hi - 1):
            g[k] = f.get(k, 0) + g.get(k - 2, 0)
        rem = {k: f.get(k, 0) - g.get(k, 0) + g.get(k - 2, 0) for k in range(lo, hi + 1)}
    else:                          # f_k = g_{k-1} - g_{k+1}: solve downward
        lo, hi = min(f), max(f)
        for k in range(hi, lo + 1, -1):
            g[k - 1] = f.get(k, 0) + g.get(k + 1, 0)
        rem = {k: f.get(k, 0) - g.get(k - 1, 0) + g.get(k + 1, 0) for k in range(lo, hi + 1)}
    if any(rem.values()):
        raise ArithmeticError("inexact division by %s" % kind)
    return {k: c for k, c in g.items() if c}


def _qm_block(kind, v, s, K):
    V = _qm_verma(K)
    sp = _qm_sigma(v, s, K)
    sm = {e: {-m: c for m, c in row.items()} for e, row in sp.items()}
    num = {}
    for e in set(sp) | set(sm):
        a, b = sp.get(e, {}), sm.get(e, {})
        if kind == "vac":
            row = _mu_div(a, "1-mu2")
        elif kind == "sym":           # (sigma_s - mu^2 sigma_bar_s)/(1 - mu^2)
            raw = dict(a)
            for m, c in b.items():
                raw[m + 2] = raw.get(m + 2, 0) - c
            row = _mu_div({m: c for m, c in raw.items() if c}, "1-mu2")
        else:                         # (sigma_s - sigma_bar_s)/(mu - mu^{-1})
            raw = dict(a)
            for m, c in b.items():
                raw[m] = raw.get(m, 0) - c
            row = _mu_div({m: c for m, c in raw.items() if c}, "mu-muinv")
        if row:
            num[e] = row
    return _qm_mul(V, num, K)


def _qm_recipe(recipe, v, K):
    out = {}
    for kind, s, sh, dress, sign in recipe:
        base = _qm_block(kind, v, s, K - sh)
        for e, row in base.items():
            if e + sh > K:
                continue
            for m, c in row.items():
                if dress:
                    _qm_add(out, e + sh, m + 1, sign * c)
                    _qm_add(out, e + sh, m - 1, sign * c)
                else:
                    _qm_add(out, e + sh, m, sign * c)
    return _qm_clean(out)


def _su2_to_qm(series, K):
    """An R(SU(2))-valued trace -> the (q, mu)-Laurent it restricts to: chi_n -> sum_k mu^{n - 2k}."""
    out = {}
    for e, r in getattr(series, "coeffs", {}).items():
        if e > K:
            continue
        for n, c in (getattr(r, "terms", None) or {}).items():
            n = n[0] if isinstance(n, tuple) else n
            for k in range(n + 1):
                _qm_add(out, e, n - 2 * k, c)
    return _qm_clean(out)


def _irrep_rows_to_qm(rows, K):
    """{q: {irrep n: c}} (the layer-2 / bootstrap record format) -> (q, mu)-Laurent."""
    out = {}
    for e, row in rows.items():
        if e > K:
            continue
        for n, c in row.items():
            for k in range(n + 1):
                _qm_add(out, e, n - 2 * k, c)
    return _qm_clean(out)


def _dodd_sides(Z, H, gens, i, j):
    """The product L_i L_j on both sides of the flavour dictionary: the zoo writes
    chi_n * L_gamma, the hand-written class a label whose r_label_decompose is
    (gamma, n).  Returns ({(gamma, n): {q: c}}, {(gamma, n): {q: c}}); raises
    NotImplementedError where the hand-written table does not reach."""
    hp = H.multiply(H.canonicalise(gens[i]), H.canonicalise(gens[j]))
    hd = {}
    for l, c in hp.terms.items():
        sec, n = H.r_label_decompose(l)
        d = hd.setdefault((tuple(sec), n), {})
        for e, v in _lpc(c).items():
            d[int(e)] = d.get(int(e), 0) + int(v)
    zd = {}
    for l, c in Z.multiply(((i, 1),), ((j, 1),)).terms.items():
        lat = tuple(sum(gens[a][k] * p for a, p in l) for k in range(len(gens[0])))
        for qe, r in c.coeffs.items():
            for irrep, v in (getattr(r, "terms", None) or {}).items():
                irrep = irrep[0] if isinstance(irrep, tuple) else irrep
                if v:
                    d = zd.setdefault((lat, irrep), {})
                    d[int(qe)] = d.get(int(qe), 0) + int(v)
    clean = lambda D: {k: {e: v for e, v in d.items() if v} for k, d in D.items() if any(d.values())}   # noqa: E731
    return clean(zd), clean(hd)


def _dodd_flavour_aware(Z, H, gens, pairs):
    """Returns (agree, differ, uncovered) over the pairs."""
    agree = differ = uncovered = 0
    for i, j in pairs:
        try:
            zd, hd = _dodd_sides(Z, H, gens, i, j)
        except NotImplementedError:
            uncovered += 1
            continue
        if zd == hd:
            agree += 1
        else:
            differ += 1
    return agree, differ, uncovered


def check_a1dodd_implementation(environment):
    import importlib
    from math import gcd
    from finite_kalgebras import FINITE_KALGEBRAS
    from finite_kalgebra_objects import cone_to_bps_iso
    import elem_traces as ET
    checks = []
    t0 = time.time()
    pop = {}
    for sid, (mname, pref, v, nseeds, hmod, hcls) in _DODD.items():
        M = importlib.import_module(mname)
        Z = FINITE_KALGEBRAS[sid]()
        gens = getattr(M, pref + "_MULT_GENS_LATTICE")
        perm = getattr(M, pref + "_RHO_PERM")
        n = len(gens)
        order, L = 1, 1
        for g in range(n):
            o, cur = 1, perm[g]
            while cur != g:
                cur, o = perm[cur], o + 1
            L = L * o // gcd(L, o)
        ring = type(Z.coefficient_ring()).__name__
        expect_n = {"a1d5": 20, "a1d7": 42}[sid]
        checks.append(check(f"{sid}: {expect_n} generators, rho a permutation of them of order {v} (Table tab:finite_type), SU(2) flavour in the coefficients",
                            n == expect_n and L == v and ring == "SU2ZPlusRing", f"n={n}, order={L}, ring={ring}"))
        # the abelian witness certifies every product that carries no SU(2) character
        B = ET._bps_oracle(sid)
        iso = cone_to_bps_iso(sid, Z, B)
        lab = lambda i: ((i, 1),)   # noqa: E731
        if sid == "a1d5":
            pairs = [(i, j) for i in range(n) for j in range(n)]
            sample = "all generator pairs"
        else:
            # D7: a generic BPS product costs minutes (measured 2026-09-23), so the witness
            # runs on every ordered product of the generators whose BPS charge is a single
            # node charge up to sign -- deterministic, and about ten minutes in all
            node = [i for i in range(n)
                    if sum(abs(x) for x in next(iter(iso.map(Element.basis(lab(i))).terms))) == 1]
            pairs = [(i, j) for i in node for j in node]
            sample = f"every ordered product of the {len(node)} generators whose BPS charge is a single node charge up to sign"
        plain_ok = plain = flav = 0
        for i, j in pairs:
            prod = Z.multiply(lab(i), lab(j))
            flavoured = any(any((k[0] if isinstance(k, tuple) else k) != 0 for k in (getattr(r, "terms", None) or {}))
                            for c in prod.terms.values() for r in c.coeffs.values())
            if flavoured:
                flav += 1
                continue
            plain += 1
            plain_ok += iso.verify_multiplicative([(Element.basis(lab(i)), Element.basis(lab(j)))], [])
        checks.append(check(f"{sid}: the cone-to-BPS witness (U(1) weights) is multiplicative on every sampled generator product without an SU(2) character (sample: {sample}; {plain} of {len(pairs)} pairs; the other {flav} carry a character it cannot translate)",
                            plain_ok == plain and plain > 0, f"{plain_ok}/{plain}"))
        # the products WITH characters: against the independent hand-written class, through the flavour dictionary
        H = getattr(importlib.import_module(hmod), hcls)()
        allpairs = [(i, j) for i in range(n) for j in range(n)]
        ag, df, unc = _dodd_flavour_aware(Z, H, gens, allpairs)
        checks.append(check(f"{sid}: every generator product the hand-written {hcls} covers agrees with the zoo class through the flavour dictionary (chi_n * L_gamma <-> the label decomposing to (gamma, n))",
                            df == 0 and ag > 0 and (sid != "a1d5" or unc == 0), f"agree {ag}, differ {df}, outside the hand-written table {unc}"))
        pop[sid] = {"generators": n, "abelian witness sample": sample, "abelian witness pairs": plain, "hand-written pairs": ag, "uncovered": unc}
    # negative control: relabelling a doublet as a singlet must be caught by the flavour-aware comparison
    import finite_a1d5_kalg as M5
    from a1d5_kalg import A1D5KAlg
    zd, hd = _dodd_sides(FINITE_KALGEBRAS["a1d5"](), A1D5KAlg(), M5.A1D5_MULT_GENS_LATTICE, 0, 4)
    k1 = [k for k in zd if k[1] == 1]
    tampered = dict(zd)
    if k1:
        tampered[(k1[0][0], 0)] = tampered.pop(k1[0])
    ok_neg = zd == hd and bool(k1) and tampered != hd
    checks.append(check("negative control: a product whose doublet coefficient is demoted to a singlet is caught by the flavour-aware comparison", ok_neg))
    return {"checks": checks, "population": {"classes": pop, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the flavour-free products against the BPS chart", "negative": "a doublet demoted to a singlet is caught"},
            "notes": "The zoo's cone-to-BPS witness is abelian: it certifies products without an SU(2) character only.  The products with a character are certified against the independent hand-written classes (all 400 pairs for D5; the pairs inside A1D7KAlg's atomic table for D7).", "inputs": []}


def check_a1dodd_is_kalgebra(environment):
    from finite_kalgebras import FINITE_KALGEBRAS
    checks = []
    t0 = time.time()
    pop = {}
    for sid, (_m, _p, _v, nseeds, _h, _c) in _DODD.items():
        A = FINITE_KALGEBRAS[sid]()
        W = [A.identity()] + [((i, 1),) for i in range(nseeds)]
        cd = A.cone_data()
        for i in range(nseeds):
            for j in range(i + 1, nseeds):
                if cd.q_commute(i, j):
                    W.append(((i, 1), (j, 1)))
                    break
            if len(W) >= nseeds + 3:
                break
        K = 4
        ver = run_k_verifiers(A, W, K)
        n_fail = sum(f for f, _x in ver.values())
        checks.append(check(f"{sid}: K1-K5 verifiers on all pairs of a window of {len(W)} labels (identity, the {nseeds} seeds, products) at K={K}", n_fail == 0,
                            "; ".join(f"{k} fails {f}" for k, (f, _x) in ver.items())))
        pop[sid] = len(W)
    return {"checks": checks, "population": {"window sizes": pop, "K": 4, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the identity row", "negative": "n.a."}, "notes": "", "inputs": []}


def check_a1dodd_traces(environment):
    from finite_kalgebras import FINITE_KALGEBRAS
    import elem_traces as ET
    import su2_reliable as SR
    from a1d3_kalg import A1D3KAlg
    checks = []
    t0 = time.time()
    # positive control: the same transcription at v = 3 reproduces the paper's [A_1,D_3] traces
    K3 = 12
    D3 = A1D3KAlg()
    got = {"vac": _su2_to_qm(D3.trace(D3.identity(), K=K3), K3), "T": _su2_to_qm(D3.trace(D3.T(0), K=K3), K3),
           "D": _su2_to_qm(D3.trace(D3.D(0), K=K3), K3)}
    want = {"vac": _qm_recipe([("vac", 0, 0, False, 1)], 3, K3), "T": _qm_recipe(_DODD_RECIPES[3]["T"], 3, K3),
            "D": _qm_recipe(_DODD_RECIPES[3]["D"], 3, K3)}
    checks.append(check("positive control: the transcribed characters at level -4/3 give A1D3KAlg's Tr 1, Tr T_0, Tr D_0 (the paper's [A_1,D_3]) through q^12",
                        got == want, "" if got == want else str({k: got[k] == want[k] for k in got})))
    pop = {}
    for sid, (_m, _p, v, nseeds, _h, _c) in _DODD.items():
        A = FINITE_KALGEBRAS[sid]()
        K = 16 if sid == "a1d5" else 12
        ok_vac = _su2_to_qm(A.trace(A.identity(), K=K), K) == _qm_recipe([("vac", 0, 0, False, 1)], v, K)
        ok_seeds = {i: _su2_to_qm(A.trace(((i, 1),), K=K), K) == _qm_recipe(_DODD_RECIPES[v][i], v, K) for i in range(nseeds)}
        checks.append(check(f"{sid}: Tr 1 and all {nseeds} seeds equal the printed admissible-character combinations at level {-(4 * (v - 1) // 2)}/{v} through q^{K}",
                            ok_vac and all(ok_seeds.values()), f"vacuum {ok_vac}, seeds {ok_seeds}"))
        # independent witness 1: the vacuum from the exact Nahm sum on the class's embedded BPS spec.
        # `_vacuum_rps` returns the Nahm series keyed by CARTAN WEIGHTS (w,), even over R(SU(2));
        # every consumer in the repo reads it through `_series_to_data`, which un-branches the
        # weights into SU(2) irreps (certified by an exact round-trip) -- so read it the same way.
        nahm = _irrep_rows_to_qm(ET._series_to_data(sid, ET._vacuum_rps(sid, K)), K)
        kap0 = _qm_recipe([("vac", 0, 0, False, 1)], v, K)
        ok_nahm = nahm == kap0
        checks.append(check(f"{sid}: the Nahm-sum vacuum on the embedded BPS spectrum equals kappa_0 through q^{K}", ok_nahm,
                            "" if ok_nahm else f"first differing orders: {sorted(e for e in set(nahm) | set(kap0) if nahm.get(e) != kap0.get(e))[:4]}"))
        # independent witness 2: the orthonormality bootstrap seeded by the Kac-Wakimoto vacuum
        Kb = 12 if sid == "a1d5" else 8
        rs = SR.reliable_seeds(sid, Kb)
        ok_boot = all(_irrep_rows_to_qm(rs[i], Kb) == _qm_recipe(_DODD_RECIPES[v][i], v, Kb) for i in range(nseeds))
        checks.append(check(f"{sid}: the orthonormality bootstrap (su2_reliable, pure powers Tr(seed^a) = O(q^a)) pins the same {nseeds} seeds through q^{Kb}", ok_boot))
        pop[sid] = {"K": K, "bootstrap K": Kb}
    # independent witness 3: the per-seed BPS engine, at low order, for D5 (D7's run is 635 s, recorded in the report)
    rec = ET.generate("a1d5", 6, method="bps")
    ok_bps = all(_irrep_rows_to_qm(rec["orbits"][i], 6) == _qm_recipe(_DODD_RECIPES[5][i], 5, 6) for i in range(4)) and \
        _irrep_rows_to_qm(rec["identity"], 6) == _qm_recipe([("vac", 0, 0, False, 1)], 5, 6)
    checks.append(check("a1d5: the per-seed BPS engine gives the same vacuum and four seeds through q^6", ok_bps))
    # the printed heads
    heads = {0: {0: 1}, 2: {2: 1}, 4: {4: 1, 2: 1, 0: 1}, 6: {6: 1, 4: 1, 2: 3, 0: 1}, 8: {8: 1, 6: 1, 4: 4, 2: 4, 0: 3}}
    for sid in ("a1d5", "a1d7"):
        A = FINITE_KALGEBRAS[sid]()
        tr = A.trace(A.identity(), K=8)
        rows = {e: {(n[0] if isinstance(n, tuple) else n): c for n, c in (getattr(r, "terms", None) or {}).items() if c} for e, r in tr.coeffs.items() if e <= 8}
        rows = {e: r for e, r in rows.items() if r}
        checks.append(check(f"{sid}: the printed head Tr 1 = 1 + chi_2 q^2 + (chi_0+chi_2+chi_4) q^4 + (chi_0+3chi_2+chi_4+chi_6) q^6 + (3chi_0+4chi_2+4chi_4+chi_6+chi_8) q^8",
                            rows == heads, str(rows)))
    # negative control: dropping one building block from a recipe breaks the comparison
    A5 = FINITE_KALGEBRAS["a1d5"]()
    broken = _DODD_RECIPES[5][0][:-1]
    ok_neg = _su2_to_qm(A5.trace(((0, 1),), K=12), 12) != _qm_recipe(broken, 5, 12)
    checks.append(check("negative control: seed 0 of D5 with its chi_1-dressed term dropped no longer matches", ok_neg))
    return {"checks": checks, "population": {"classes": pop, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the level -4/3 transcription against A1D3KAlg", "negative": "a recipe with one term dropped fails"},
            "notes": "Characters transcribed independently of a1d5_layer2 / a1d7_layer2 and compared as (q, mu)-Laurent series; witnesses: the Nahm-sum vacuum, the su2_reliable bootstrap, the per-seed BPS engine (D5).", "inputs": []}


def _dodd_rho_orbits(Z):
    """The ρ-orbits of a zoo class's generators, from its own permutation."""
    perm, seen, orbs = Z._rho_perm, set(), []
    for g in sorted(perm):
        if g in seen:
            continue
        orb, x = [], g
        while x not in seen:
            seen.add(x)
            orb.append(x)
            x = perm[x]
        orbs.append(orb)
    return orbs


def check_a1dodd_geometry(environment):
    """The geometric labels printed in comp:a1dodd ("Geometric labels"),
    transcribed: the curves (x, l), x in Z/(2n+1), 2 <= l <= 2n+1, name the
    generators one each; rho is the rotation (x, l) -> (x + 1, l); a
    q-commuting generator pair is named by the union of its two curves; and the
    generator map onto A1DoddConeKAlg is a certified KAlgebraIso (verify_all:
    unit, round trip, multiplicative, rho- and trace-equivariant)."""
    import importlib
    from finite_kalgebras import FINITE_KALGEBRAS
    import a1dodd_seeds as S
    from zoo_geometry import geometric_label
    from kalgebra_iso import KAlgebraIso
    checks = []
    t0 = time.time()
    pop = {}
    one = LaurentPoly.one()
    K = 6
    for sid, (mname, pref, v, _ns, _h, _c) in _DODD.items():
        n = v                                    # 2n+1 marked points in the companion's n
        N = len(getattr(importlib.import_module(mname), pref + "_MULT_GENS_LATTICE"))
        Z = FINITE_KALGEBRAS[sid]()
        # --- the printed curve set, transcribed ---
        expected = {(x, ell) for x in range(n) for ell in range(2, n + 1)}
        names, bad = {}, []
        for i in range(N):
            g = geometric_label(sid, ((i, 1),))
            if len(g) == 1 and g[0][1] == 1 and tuple(g[0][0]) in expected:
                names[i] = tuple(g[0][0])
            else:
                bad.append((i, g))
        ok = (not bad and N == n * (n - 1) and len(set(names.values())) == N
              and set(names.values()) == expected)
        checks.append(check(f"{sid}: each of the {N} generators is one curve (x, l), x in Z/{n}, "
                            f"2 <= l <= {n}, and the {n}*{n - 1} curves are named once each",
                            ok, f"bad shapes {bad[:3]}"))
        # --- rho is the rotation ---
        rot_bad = [i for i in range(N)
                   if geometric_label(sid, Z.rho(((i, 1),)))
                   != ((((names[i][0] + 1) % n, names[i][1]), 1),)]
        checks.append(check(f"{sid}: rho is the rotation (x, l) -> (x + 1, l) on all {N} generators",
                            not rot_bad, f"fails at {rot_bad[:5]}"))
        # --- a q-commuting pair is named by the union of its curves ---
        cd = Z.cone_data()
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N) if cd.q_commute(i, j)]
        mono_bad = [(i, j) for i, j in pairs
                    if geometric_label(sid, ((i, 1), (j, 1)))
                    != tuple(sorted(((names[i], 1), (names[j], 1))))]
        checks.append(check(f"{sid}: each of the {len(pairs)} q-commuting generator pairs is named "
                            f"by the union of its two curves", pairs and not mono_bad,
                            f"fails at {mono_bad[:5]}"))
        # --- the generator map as a certified KAlgebraIso ---
        iso = S.kalgebra_iso(sid, native=Z)
        D = iso.target
        src = ([iso.source.identity()] + [(0, ((g, 1),)) for g in range(N)] + [(1, ())])
        tgt = ([D.identity()] + [(((g, 1),), 0) for g in sorted(D.cone_data().mult_gens())]
               + [((), 1)])
        se = [Element({l: one}) for l in src]
        te = [Element({l: one}) for l in tgt]
        res = iso.verify_all(se, te, [(a, b) for a in se[1:] for b in se[1:]],
                             [(a, b) for a in te[1:] for b in te[1:]], trace_K=K)
        checks.append(check(f"{sid}: the generator map onto A1DoddConeKAlg({S.K_OF[sid]}) passes "
                            f"KAlgebraIso.verify_all (unit, round trip, multiplicative on "
                            f"{len(se) - 1}^2 ordered pairs each way, rho-equivariant, "
                            f"trace-equivariant through q^{K})",
                            len(res) == 5 and all(res.values()), str(res)))
        # --- negative control: two rho-orbits of generators swapped ---
        orbs = _dodd_rho_orbits(Z)
        a, b = orbs[0], orbs[1]
        swap = {}
        for g, h in zip(a, b):
            swap[g], swap[h] = h, g

        def perm(l, t=swap):
            w, word = l
            return (w, tuple(sorted((t.get(g, g), p) for g, p in word)))

        bent = KAlgebraIso(
            iso.source, iso.target,
            lambda l: iso.map(Element({perm(l): one})),
            lambda l: Element({perm(x): c for x, c in iso.inverse(Element({l: one})).terms.items()}),
            name=f"{sid}[two rho-orbits swapped]")
        mult_ok = bent.verify_multiplicative([(x, y) for x in se[1:] for y in se[1:]], [])
        checks.append(check(f"negative control ({sid}): the map with two rho-orbits of "
                            f"generators swapped fails multiplicativity", not mult_ok))
        pop[sid] = {"generators": N, "q-commuting pairs": len(pairs),
                    "iso samples": len(se), "trace K": K}
    return {"checks": checks, "population": {"classes": pop, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the printed curve set reproduced; the iso's verify_all",
                         "negative": "two rho-orbits swapped in the generator map fail multiplicativity"},
            "notes": "The curve set and the rotation are transcribed from the companion's text; "
                     "the labels are read through the class's geometric_label, i.e. through "
                     "finite_kalgebras.zoo_geometry and the served generator map.", "inputs": []}


# ============================================================================
# comp:e6 — K_q([A_1, E_6]): the exported cone table, and traces with no frozen data
# ============================================================================
def _perm_order(perm):
    k, cur = 1, dict(perm)
    while any(cur[i] != i for i in cur):
        cur = {i: perm[cur[i]] for i in cur}
        k += 1
        if k > 10 ** 4:
            return None
    return k


def check_e6_implementation(environment):
    import finite_e6_kalg as E
    from finite_e6_kalg import FiniteE6KAlgebra
    from finite_kalgebra_objects import cone_to_bps_iso
    from bps_kalgebra import BPSKAlgebra
    import random
    checks = []
    t0 = time.time()
    n = len(E.E6_MULT_GENS_LATTICE)
    cones = E.E6_CONES
    order = _perm_order(E.E6_RHO_PERM)
    ring = type(FiniteE6KAlgebra().coefficient_ring()).__name__
    checks.append(check("42 generators, 833 cones all of size 6, rho a permutation of the generators of order 14, trivial flavour",
                        n == 42 and len(cones) == 833 and all(len(c) == 6 for c in cones) and order == 14 and ring == "TrivialZPlusRing",
                        f"n={n}, cones={len(cones)}, order={order}, ring={ring}"))
    A = FiniteE6KAlgebra()
    B = BPSKAlgebra(E.E6_BPS_PAIRING, E.E6_BPS_NODE_CHARGES)
    iso = cone_to_bps_iso("e6", A, B)
    rnd = random.Random(6)
    pairs = [(rnd.randrange(n), rnd.randrange(n)) for _ in range(40)]
    lab = lambda i: ((i, 1),)   # noqa: E731
    ok_mult = iso.verify_multiplicative([(Element.basis(lab(i)), Element.basis(lab(j))) for i, j in pairs], [])
    ok_rho = iso.verify_rho_equivariant([Element.basis(lab(i)) for i in range(n)], [])
    ok_unit = iso.verify_unit()
    checks.append(check("the cone-to-BPS isomorphism witness: unit, rho-equivariant on all 42 generators, multiplicative on 40 sampled generator pairs",
                        ok_mult and ok_rho and ok_unit, f"unit {ok_unit}, rho {ok_rho}, mult {ok_mult}"))
    return {"checks": checks, "population": {"generators": n, "cones": len(cones), "witness pairs": 40, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the literals' counts", "negative": "n.a."},
            "notes": "The flavour is trivial, so the zoo's abelian cone-to-BPS witness certifies every product it samples.", "inputs": []}


def _e6_window(A, n_products=3):
    import elem_traces as T
    seeds = T.elementary_seed_indices("e6")
    lab = lambda i: ((i, 1),)   # noqa: E731
    W = [A.identity()] + [lab(i) for i in seeds]
    cd = A.cone_data()
    extra = []
    for i in seeds:
        for j in seeds:
            if i < j and cd.q_commute(i, j):
                extra.append(((i, 1), (j, 1)))
        if len(extra) >= n_products:
            break
    return W + extra[:n_products], seeds


def check_e6_is_kalgebra(environment):
    from finite_e6_kalg import FiniteE6KAlgebra
    checks = []
    t0 = time.time()
    A = FiniteE6KAlgebra()
    W, _seeds = _e6_window(A, n_products=1)
    K = 4
    ver = run_k_verifiers(A, W, K)
    n_fail = sum(f for f, _x in ver.values())
    checks.append(check(f"K1-K5 verifiers on all pairs of a window of {len(W)} labels (identity, the six seeds, a product) at K={K}", n_fail == 0,
                        "; ".join(f"{k} fails {f}" for k, (f, _x) in ver.items())))
    return {"checks": checks, "population": {"labels": len(W), "K": K, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the identity row", "negative": "n.a."}, "notes": "", "inputs": []}


_E6_CHILD = r'''
import sys, os, json
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "implementations"))
import elem_traces as ET
import elem_trace_data as ETD
# the e6 table was removed on 2026-09-23, and the frozen-table reader with the last table
no_table = "e6" not in ETD.ELEM_TRACE_DATA and not hasattr(ET, "_frozen")
ET._W3_SEEDS.pop("e6", None)                  # and the served W3 recipes are made absent: this run is the WITNESS route
ran = set()
def prof(frame, event, arg):
    if event == "call" and (frame.f_code.co_flags & 3) == 3:
        ran.add(os.path.basename(frame.f_code.co_filename))
from finite_e6_kalg import FiniteE6KAlgebra
A = FiniteE6KAlgebra()
K = KVAL
seeds = ET.elementary_seed_indices("e6")
def val(c):
    if isinstance(c, int):
        return c
    if isinstance(c, dict):
        return sum(val(v) for v in c.values())
    t = getattr(c, "terms", None)
    return sum(t.values()) if t is not None else int(c)
def d(x):
    return {int(e): val(c) for e, c in x.coeffs.items() if e <= K and val(c)}
sys.setprofile(prof)
out = {"identity": d(A.trace(A.identity(), K=K)), "seeds": {str(i): d(A.trace(((i, 1),), K=K)) for i in seeds}}
cd = A.cone_data()
W = [A.identity()] + [((i, 1),) for i in seeds]
for i in seeds:
    for j in seeds:
        if i < j and cd.q_commute(i, j) and len(W) < 10:
            W.append(((i, 1), (j, 1)))
out["valuations"] = [min((e for e, c in d(A.trace(l, K=K)).items() if c), default=None) for l in W[1:]]
out["orthonormal"] = all(A.verify_orthonormality(a, b, K=4) for a in W for b in W)
sys.setprofile(None)
out["bps_chart_ran"] = sorted(f for f in ran if f in ("bps_kalgebra.py", "bps_factor_spectrum.py"))
out["no_table"] = no_table
print("RESULT " + json.dumps(out))
'''


E6_BPS_WITNESS_K = 6    # the per-seed BPS engine's order for the e6 seed witness (timed 2026-09-23)


def check_e6_traces(environment):
    import json
    import subprocess
    import elem_traces as ET
    from e6_rgkalgebra import E6RGKAlgebra
    checks = []
    t0 = time.time()
    K = 12
    code = _E6_CHILD.replace("ROOT", repr(ROOT)).replace("KVAL", str(K))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1200)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        checks.append(check("the no-frozen-table run completed", False, (p.stderr.strip().splitlines() or ["no output"])[-1][:200]))
        return {"checks": checks, "population": {"K": K}, "controls": {}, "notes": "", "inputs": []}
    r = json.loads(line[-1][7:])
    ident = {int(e): v for e, v in r["identity"].items()}
    seeds = {k: {int(e): v for e, v in s.items()} for k, s in r["seeds"].items()}
    checks.append(check("witness route (no frozen trace table exists for e6; the served W3 recipes made absent), and no BPS chart runs: Tr 1 is the exact Nahm sum on the embedded spectrum, the seeds come from the orthonormality bootstrap",
                        r["no_table"] and r["bps_chart_ran"] == [], f"no table {r['no_table']}; BPS files run {r['bps_chart_ran']}"))
    # the SERVED route: every seed a W3(3,7) character combination (w3_seeds)
    import w3_seeds as W3
    from finite_e6_kalg import FiniteE6KAlgebra
    A6 = FiniteE6KAlgebra()
    served_id = _lp_dict(A6.trace(A6.identity(), K=K), K)
    served = {k: _lp_dict(A6.trace(((int(k), 1),), K=K), K) for k in seeds}
    rec_id = W3.E6.vacuum_trace(K)
    rec = {k: W3.E6.seed_trace(int(k), K) for k in seeds}
    checks.append(check(f"served: Tr 1 and all six seeds of FiniteE6KAlgebra are the printed W3(3,7) character combinations through q^{K}",
                        served_id == rec_id and served == rec))
    checks.append(check(f"witness: the Nahm-sum vacuum and the bootstrap seeds equal the W3(3,7) combinations through q^{K}",
                        ident == rec_id and all(seeds[k] == rec[k] for k in seeds)))
    heads = {0: 1, 4: 1, 6: 2, 8: 3, 10: 3, 12: 6}
    ok_heads = ident == heads and {e: v for e, v in seeds["0"].items() if e <= 7} == {1: -1, 5: -2, 7: -1} and \
        {e: v for e, v in seeds["2"].items() if e <= 7} == {1: -1, 5: -1, 7: -1}
    checks.append(check("the printed heads: Tr 1 = 1 + q^4 + 2q^6 + 3q^8 + 3q^10 + 6q^12, Tr s_0 = -q - 2q^5 - q^7 + ..., Tr s_2 = -q - q^5 - q^7 + ...",
                        ok_heads, f"Tr 1 {ident}; s0 {seeds['0']}; s2 {seeds['2']}"))
    # witness: the per-seed BPS engine, computed here (after the child's BPS-free run)
    Kb = E6_BPS_WITNESS_K
    tb = time.time()
    rec = ET.generate("e6", Kb, method="bps")
    tb = round(time.time() - tb, 1)
    def _cut(dd):
        return {int(e): (sum(v.values()) if isinstance(v, dict) else int(v)) for e, v in dd.items()
                if int(e) <= Kb and (sum(v.values()) if isinstance(v, dict) else int(v))}
    bid = _cut(rec["identity"])
    bseeds = {str(k): _cut(v) for k, v in rec["orbits"].items()}
    ours = {k: {e: v for e, v in sd.items() if e <= Kb and v} for k, sd in seeds.items()}
    same = bid == {e: v for e, v in ident.items() if e <= Kb and v} and bseeds == ours
    checks.append(check(f"witness: the vacuum and all six seeds equal the per-seed BPS engine through q^{Kb} ({tb} s)",
                        same, f"BPS Tr 1 {bid}; BPS seeds {bseeds}"))
    # negative control: one seed perturbed must be caught by the same comparison
    bad = {k: dict(v) for k, v in ours.items()}
    k0 = sorted(bad)[0]
    e0 = min(bad[k0])
    bad[k0][e0] += 1
    checks.append(check("negative control: a seed perturbed by one unit in its leading coefficient is caught by the BPS comparison",
                        bseeds != bad))
    E = E6RGKAlgebra()
    flow = _lp_dict(E.trace(E.identity(), K=K), K)
    checks.append(check(f"independent witness: Tr 1 equals the E6 RG flow's through q^{K}", flow == ident, f"flow {flow}"))
    checks.append(check("Tr L = O(q) for every window label L != 1 and I_ab = delta + O(q) on all window pairs",
                        all(v is not None and v >= 1 for v in r["valuations"]) and r["orthonormal"], f"valuations {r['valuations']}"))
    return {"checks": checks, "population": {"K": K, "window": len(r["valuations"]) + 1, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the per-seed BPS engine as a witness through q^%d; the served W3(3,7) combinations against the witness route" % Kb,
                         "negative": "a seed perturbed by one unit is caught by the same comparison"},
            "notes": "The witness route runs in a subprocess with the served W3 recipes made absent, so no cached or recipe data reaches it; e6 has no frozen trace table (removed 2026-09-23).", "inputs": []}


# ============================================================================
# comp:e7 — K_q([A_1, E_7]) with its U(1) flavour: the exported cone table, the
# Bershadsky–Polyakov vacuum and the nine theta-product recipes
# ============================================================================
# Every printed series is transcribed here from the companion's text, with its own
# exact arithmetic: a series is a dict {(q-power, mu-power): int}, exact through a
# stated q-order.  Nothing below reads finite_kalgebras/e7_seeds.py; the served
# traces are read from FiniteE7KAlgebra().trace, and the witnesses are the pinned
# bootstrap and Nahm fixtures of tests/test_e7_seeds.py and live runs.

_E7_REPS = (0, 1, 2, 3, 5, 7, 12, 13, 21)

# eq:comp-e7-recipes, transcribed from the display: representative ->
# [(q-power, mu-power, {(a, b): coefficient of Theta_ab})]
_E7_RECIPES_TEX = {
    0: [(-3, -1, {(1, 2): 1, (1, 3): -1}), (-2, 0, {(3, 4): 1, (2, 4): -1}),
        (-1, -1, {(0, 1): 1, (0, 4): 1, (1, 2): 1, (0, 2): -1, (0, 3): -1}), (0, 0, {(3, 4): 1})],
    2: [(-2, -1, {(1, 2): 1, (2, 3): 1, (0, 3): -1, (1, 3): -1}), (-1, 0, {(3, 4): 1}), (0, -1, {(0, 4): 1})],
    3: [(-2, 0, {(1, 2): 2, (2, 3): 2, (0, 3): -1, (1, 3): -2, (1, 4): -1}),
        (-1, -1, {(0, 1): 1}), (-1, 1, {(3, 4): 1}),
        (0, 0, {(0, 4): 3, (1, 2): 2, (2, 3): 2, (0, 2): -1, (0, 3): -1, (1, 3): -1, (1, 4): -1, (2, 4): -1}),
        (1, -1, {(0, 1): 1}), (1, 1, {(3, 4): 1}),
        (2, 0, {(0, 4): 2, (1, 2): 1, (2, 3): 1, (0, 3): -1, (1, 3): -1, (1, 4): -1}),
        (4, 0, {(0, 4): 1})],
    5: [(-1, 0, {(0, 4): 1, (1, 2): 1, (2, 3): 1, (0, 3): -1, (1, 3): -1, (1, 4): -1}), (1, 0, {(0, 4): 1})],
    7: [(-1, 0, {(0, 4): 1, (1, 2): 2, (2, 3): 2, (0, 2): -1, (0, 3): -1, (1, 3): -1, (1, 4): -1, (2, 4): -1}),
        (0, -1, {(0, 1): 1}), (0, 1, {(3, 4): 1}),
        (1, 0, {(0, 4): 2, (1, 2): 1, (2, 3): 1, (0, 3): -1, (1, 3): -1, (1, 4): -1}),
        (3, 0, {(0, 4): 1})],
    12: [(-2, 0, {(2, 3): 1, (2, 4): -1}), (-1, -1, {(0, 1): 1})],
    13: [(-1, 0, {(2, 3): 1, (3, 4): 1, (1, 4): -1, (2, 4): -1}), (0, -1, {(0, 1): 1})],
    21: [(-1, -1, {(0, 1): 1, (1, 2): 2, (0, 2): -1, (0, 3): -1, (1, 3): -1}), (0, 0, {(3, 4): 2, (2, 4): -1}),
         (1, -1, {(0, 1): 1, (0, 4): 1, (1, 2): 1, (0, 2): -1, (0, 3): -1}), (2, 0, {(3, 4): 1})],
}
_E7_RECIPES_TEX[1] = _E7_RECIPES_TEX[0]          # "N_0 = N_1"

# eq:comp-e7-heads, transcribed
_E7_HEADS_TEX = {
    "vacuum": ({(0, 0): 1, (2, 0): 1, (3, 1): -1, (3, -1): -1, (4, 0): 3, (5, 1): -2, (5, -1): -2,
                (6, 2): 1, (6, 0): 6, (6, -2): 1}, 6),
    5: ({(1, 0): -1, (3, 0): -1, (4, 1): 1, (4, -1): 1, (5, 0): -3}, 5),
    0: ({(3, -1): -1, (7, -1): -1, (8, -2): 1}, 8),
    1: ({(3, -1): -1, (7, -1): -1, (8, -2): 1}, 8),
}


def _e7_clean(d):
    return {k: v for k, v in d.items() if v}


def _e7_mul(a, b, K):
    out = {}
    for (qa, ma), ca in a.items():
        for (qb, mb), cb in b.items():
            q = qa + qb
            if q <= K:
                k = (q, ma + mb)
                out[k] = out.get(k, 0) + ca * cb
    return _e7_clean(out)


def _e7_times_binomial(s, e, m, c, K):
    """s * (1 + c q^e mu^m)."""
    return _e7_mul(s, {(0, 0): 1, (e, m): c}, K) if e <= K else dict(s)


def _e7_inverse(D, K):
    """1/D for D = 1 + O(q) with Laurent-polynomial mu-coefficients, order by order."""
    if D.get((0, 0)) != 1 or any(q < 0 or (q == 0 and m != 0) for (q, m) in D):
        raise ValueError("_e7_inverse: the series must be 1 + O(q)")
    rows = {}
    for (q, m), c in D.items():
        rows.setdefault(q, {})[m] = c
    inv = {0: {0: 1}}
    for k in range(1, K + 1):
        acc = {}
        for j in range(1, k + 1):
            for m1, c1 in rows.get(j, {}).items():
                for m2, c2 in inv.get(k - j, {}).items():
                    acc[m1 + m2] = acc.get(m1 + m2, 0) - c1 * c2
        inv[k] = {m: c for m, c in acc.items() if c}
    return {(q, m): c for q, r in inv.items() for m, c in r.items()}


def _e7_vacuum_tex(K, ten=10):
    """eq:comp-e7-vacuum: prod_{n>=1} (1-q^{10n})^2 (1-q^{10n-2}) (1-q^{10n+2})
    prod_pm (1+mu^{pm1} q^{10n-1}) (1+mu^{pm1} q^{10n+1}) / [(1-q^{2n}) (1-q^{2n+2}) prod_pm (1+mu^{pm1} q^{2n+1})].
    `ten` = 10 as printed; 8 and 14 are the negative controls (u = 4, 7)."""
    num, n = {(0, 0): 1}, 1
    while ten * n - 2 <= K:
        for (e, m, c) in ((ten * n, 0, -1), (ten * n, 0, -1), (ten * n - 2, 0, -1), (ten * n + 2, 0, -1),
                          (ten * n - 1, 1, 1), (ten * n - 1, -1, 1), (ten * n + 1, 1, 1), (ten * n + 1, -1, 1)):
            num = _e7_times_binomial(num, e, m, c, K)
        n += 1
    den, n = {(0, 0): 1}, 1
    while 2 * n <= K:
        for (e, m, c) in ((2 * n, 0, -1), (2 * n + 2, 0, -1), (2 * n + 1, 1, 1), (2 * n + 1, -1, 1)):
            den = _e7_times_binomial(den, e, m, c, K)
        n += 1
    return _e7_mul(num, _e7_inverse(den, K), K)


def _e7_theta(K):
    """theta(mu) = sum_n q^{n^2} mu^n (eq:comp-e7-blocks)."""
    out, n = {(0, 0): 1}, 1
    while n * n <= K:
        out[(n * n, n)] = 1
        out[(n * n, -n)] = 1
        n += 1
    return out


def _e7_psi(a, K):
    """psi_a = sum_n q^{5n^2 + (2a-4)n} mu^n (eq:comp-e7-blocks)."""
    out, n = {}, 0
    while 5 * n * n - 4 * n <= K:
        for nn in ({n, -n} if n else {0}):
            e = 5 * nn * nn + (2 * a - 4) * nn
            if e <= K:
                out[(e, nn)] = out.get((e, nn), 0) + 1
        n += 1
    return out


def _e7_th(r, K):
    """th_r = prod_{m>=0} (1 - q^{10m+r}) (1 - q^{10m+10-r}) (eq:comp-e7-blocks)."""
    out, m = {(0, 0): 1}, 0
    while 10 * m + min(r, 10 - r) <= K:
        for e in (10 * m + r, 10 * m + 10 - r):
            out = _e7_times_binomial(out, e, 0, -1, K)
        m += 1
    return out


_E7_BLOCKS: dict = {}


def _e7_Theta(K):
    """Theta_ab = th_ab psi_a psi_b, th_ab = th_2 for |a-b| in {1,4} and th_4 for |a-b| in {2,3}."""
    if K not in _E7_BLOCKS:
        psi = [_e7_psi(a, K) for a in range(5)]
        th = {2: _e7_th(2, K), 4: _e7_th(4, K)}
        _E7_BLOCKS[K] = {(a, b): _e7_mul(th[2 if abs(a - b) in (1, 4) else 4], _e7_mul(psi[a], psi[b], K), K)
                         for a in range(5) for b in range(a + 1, 5)}
    return _E7_BLOCKS[K]


def _e7_seed_rep(rep, K, recipes=None):
    """eq:comp-e7-seed: Tr s_rep = N_rep / ((q^2;q^2)_inf theta(mu)) through q^K (N_rep starts at q^{-3})."""
    KK = K + 3
    blocks = _e7_Theta(KK)
    N = {}
    for (s, t, terms) in (recipes or _E7_RECIPES_TEX)[rep]:
        for ab, c in terms.items():
            for (q, m), v in blocks[ab].items():
                if q + s <= K:
                    N[(q + s, m + t)] = N.get((q + s, m + t), 0) + c * v
    qq = {(0, 0): 1}
    for n in range(1, KK // 2 + 1):
        qq = _e7_times_binomial(qq, 2 * n, 0, -1, KK)
    return _e7_mul(_e7_clean(N), _e7_inverse(_e7_mul(qq, _e7_theta(KK), KK), KK), K)


def _e7_all_seeds(K, recipes=None, sign=-1):
    """All 90 seeds from the nine printed recipes by eq:comp-e7-rule,
    Tr s_{rho(i)}(mu) = mu^{-delta_i} Tr s_i(1/mu), with the class's rho-permutation and delta table.
    sign = +1 is the negative control mu^{+delta_i}."""
    from finite_e7_kalg import E7_RHO_PERM, E7_RHO_DELTA
    out = {}
    for rep in _E7_REPS:
        T, i = _e7_seed_rep(rep, K, recipes), rep
        while True:
            if i in out:
                raise AssertionError(f"seed {i} reached twice by the printed rule")
            out[i] = T
            d = E7_RHO_DELTA[i][0]
            T = {(q, -m + sign * d): v for (q, m), v in T.items()}
            i = E7_RHO_PERM[i]
            if i == rep:
                break
    return out


def _e7_rps(x, K):
    """An RPowerSeries over R(U(1)) (or Z) -> {(q, mu-power): int} through q^K."""
    out = {}
    for e, r in x.coeffs.items():
        if e > K:
            continue
        for kb, v in (getattr(r, "terms", None) or {}).items():
            if int(v):
                out[(e, kb[0] if kb else 0)] = out.get((e, kb[0] if kb else 0), 0) + int(v)
    return _e7_clean(out)


def _e7_cut(d, K):
    return {k: v for k, v in d.items() if k[0] <= K and v}


def _e7_fixtures():
    """The pinned witnesses of tests/test_e7_seeds.py: the certified u(1) orthonormality bootstrap's
    eight distinct representatives and the Nahm-sum vacuum, both through q^20, as {(q, mu): c}."""
    tests = os.path.join(ROOT, "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    import test_e7_seeds as TE
    flat = lambda s: {(q, m): c for q, d in s.items() for m, c in d.items() if c}   # noqa: E731
    return {r: flat(s) for r, s in TE.PINNED_20.items()}, flat(TE.NAHM_VACUUM_20)


def _perm_cycles(perm):
    seen, cyc = set(), []
    for s in sorted(perm):
        if s in seen:
            continue
        n, x = 0, s
        while True:
            seen.add(x)
            x, n = perm[x], n + 1
            if x == s:
                break
        cyc.append(n)
    return cyc


def _e7_maximal_cliques(n, qc):
    adj = {i: {j for j in range(n) if j != i and qc(i, j)} for i in range(n)}
    out = []

    def bk(R, P, X):
        if not P and not X:
            out.append(frozenset(R))
            return
        u = max(P | X, key=lambda v: len(adj[v] & P))
        for v in list(P - adj[u]):
            bk(R | {v}, P & adj[v], X & adj[v])
            P, X = P - {v}, X | {v}
    bk(set(), set(range(n)), set())
    return out


def _e7_phi_inverse(elt, Y, sigma=1):
    """phi^{-1} of a FiniteE7KAlgebra element, written from the printed statement phi(L_l) = mu^r L_z
    with E -> mu: a zoo label ((g, p), ...) goes to the gauged label whose chord multiset is sum p * chord(y_g)
    and whose gauge charge is sum p * c1(y_g), and mu^t to E^{sigma t}.  Returns {label: {q: int}}."""
    out = {}
    for zl, c in elt.terms.items():
        chord, c1 = {}, 0
        for (g, p) in zl:
            w, cg = Y[g]
            for (a, i, e) in w:
                chord[(a, i)] = chord.get((a, i), 0) + e * p
            c1 += cg * p
        w = tuple(sorted((a, i, e) for (a, i), e in chord.items() if e))
        coeffs = getattr(c, "coeffs", None)
        items = ([(e, kb[0] if kb else 0, int(v)) for e, r in coeffs.items() for kb, v in r.terms.items()]
                 if coeffs is not None else [(e, 0, int(v)) for e, v in _lpc(c).items()])
        for (e, t, v) in items:
            if v:
                d = out.setdefault((w, (0, c1 + sigma * t)), {})
                d[e] = d.get(e, 0) + v
    return {l: {e: v for e, v in d.items() if v} for l, d in out.items() if any(d.values())}


def _e7_gauged_products(Z, G, Y, sigma=1, stop_at_first=False):
    """Count the ordered generator pairs whose products disagree under phi (0 = all agree)."""
    bad = 0
    for g in range(90):
        for h in range(90):
            ours = _e7_phi_inverse(Z.multiply(((g, 1),), ((h, 1),)), Y, sigma)
            theirs = {l: {e: int(v) for e, v in _lpc(c).items() if int(v)}
                      for l, c in G.multiply((Y[g][0], (0, Y[g][1])), (Y[h][0], (0, Y[h][1]))).terms.items()
                      if not c.is_zero()}
            if ours != theirs:
                bad += 1
                if stop_at_first:
                    return bad
    return bad


def _e7_mu_carrying(prod):
    return any(any(k[0] != 0 for k in r.terms) for c in prod.terms.values() for r in c.coeffs.values())


def check_e7_implementation(environment):
    import finite_e7_kalg as M
    from finite_e7_kalg import FiniteE7KAlgebra
    from fractions import Fraction
    checks = []
    t0 = time.time()
    A = FiniteE7KAlgebra()
    R = A.coefficient_ring()
    n = len(M.E7_MULT_GENS_LATTICE)
    qc = lambda g, h: g == h or (g, h) in M.E7_COCYCLE_TABLE   # noqa: E731
    cones = [frozenset(c) for c in M.E7_CONES]
    sizes = {}
    for c in cones:
        sizes[len(c)] = sizes.get(len(c), 0) + 1
    six = {c for c in cones if len(c) == 6}
    cliques = _e7_maximal_cliques(n, qc)
    faces = [c for c in cones if len(c) < 6]
    ok_faces = all(any(c < d for d in six) for c in faces)
    ok_cocycle = all(M.E7_COCYCLE_TABLE[(h, g)] == -v for (g, h), v in M.E7_COCYCLE_TABLE.items())
    checks.append(check("90 generators; 3300 cones: the 2600 of size 6 are exactly the maximal sets of pairwise q-commuting generators, "
                        "the other 700 are faces of them; the cocycle is antisymmetric",
                        n == 90 and len(cones) == 3300 and sizes.get(6) == 2600 and set(cliques) == six
                        and len(faces) == 700 and ok_faces and ok_cocycle,
                        f"n={n}, cones={len(cones)} {sorted(sizes.items())}, maximal cliques {len(cliques)}, faces ok {ok_faces}, cocycle ok {ok_cocycle}"))
    # the flavour: rank B = 6, ker B spanned by e0 - e2 + e6 (node basis), coefficient ring R(U(1))
    B = [[Fraction(x) for x in row] for row in M.E7_BPS_PAIRING]
    rows, rank = [r[:] for r in B], 0
    for col in range(7):
        piv = next((i for i in range(rank, 7) if rows[i][col] != 0), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(7):
            if i != rank and rows[i][col] != 0:
                f = rows[i][col] / rows[rank][col]
                rows[i] = [x - f * y for x, y in zip(rows[i], rows[rank])]
        rank += 1
    v = [1, 0, -1, 0, 0, 0, 1]
    in_ker = all(sum(B[i][j] * v[j] for j in range(7)) == 0 for i in range(7))
    ring = type(R).__name__
    checks.append(check("the flavour: rank B = 6 with ker B spanned by e0 - e2 + e6; the coefficient ring is R(U(1)) = Z[mu^+-1]",
                        rank == 6 and in_ker and ring == "AbelianZPlusRing" and getattr(R, "rank", None) == 1,
                        f"rank {rank}, e0-e2+e6 in ker {in_ker}, ring {ring}"))
    # the cross-product table: every ordered pair that does not q-commute, mu-powers -2..2
    nonqc = [(g, h) for g in range(n) for h in range(n) if not qc(g, h)]
    mus = set()
    for (g, h) in nonqc:
        for (_kind, data, _w) in M.E7_CROSS_TABLE.get((g, h), []):
            for d in data.values():
                mus |= {k[0] for k in d}
    checks.append(check("the 5880 ordered pairs of generators that do not q-commute all have their product in the table, "
                        "with powers of mu from -2 to 2",
                        len(nonqc) == 5880 and all(p in M.E7_CROSS_TABLE for p in nonqc) and mus == {-2, -1, 0, 1, 2},
                        f"{len(nonqc)} pairs, mu-powers {sorted(mus)}"))
    # rho: nine orbits of length 10 (order 10, Table tab:finite_type: U(1), 10); delta in {0,-1,-2}, 41 nonzero;
    # rho^10 the identity on elements (eq:comp-e7-rho, through the class's rho_element)
    cyc = _perm_cycles(M.E7_RHO_PERM)
    dvals = [M.E7_RHO_DELTA[i][0] for i in range(n)]
    one = R.one_basis()
    from zplus_ring import RLaurent, RElement
    unit = RLaurent(R, {0: RElement(R, {one: 1})})
    back = 0
    for g in range(n):
        x = Element({((g, 1),): unit})
        y = x
        for _ in range(10):
            y = A.rho_element(y)
        back += (y.terms == x.terms) or ({l: str(c) for l, c in y.terms.items()} == {l: str(c) for l, c in x.terms.items()})
    checks.append(check("rho permutes the generators in nine orbits of length 10 (order 10, as Table tab:finite_type lists with U(1)); "
                        "delta_i in {0,-1,-2}, 41 of 90 nonzero; rho^10 (rho_element, eq:comp-e7-rho) is the identity on every generator",
                        sorted(cyc) == [10] * 9 and set(dvals) <= {0, -1, -2} and sum(1 for d in dvals if d) == 41 and back == n,
                        f"cycles {sorted(cyc)}, nonzero delta {sum(1 for d in dvals if d)}, rho^10 = id on {back}/{n}"))
    # negative control for the element-level claim: with one delta_i moved by 1 the powers of mu no longer cancel
    # around that orbit, and rho^10 is caught off the identity (the check reads the twist)
    A2 = FiniteE7KAlgebra()
    A2._rho_delta = dict(M.E7_RHO_DELTA)
    A2._rho_delta[4] = (M.E7_RHO_DELTA[4][0] + 1,)
    x = Element({((4, 1),): unit})
    y = x
    for _ in range(10):
        y = A2.rho_element(y)
    caught = {l: str(c) for l, c in y.terms.items()} != {l: str(c) for l, c in x.terms.items()}
    checks.append(check("negative control: with delta_4 moved by 1, rho^10 is no longer the identity on L_4", caught,
                        f"rho^10(L_4) = {({l: str(c) for l, c in y.terms.items()})}"))
    # the witness through the gauged class: all 8100 generator products agree under phi (E -> mu)
    import u1e7_cone_kalgebra as GM
    from u1e7_cone_kalgebra import U1E7ConeKAlgebra
    G = U1E7ConeKAlgebra()
    Y = {g: GM._E7_GENERATOR_PREIMAGES[tuple(M.E7_MULT_GENS_LATTICE[g])] for g in range(n)}
    tg = time.time()
    bad = _e7_gauged_products(A, G, Y)
    tg = round(time.time() - tg, 1)
    checks.append(check(f"all 8100 ordered generator products agree with U1E7ConeKAlgebra's stored tables (built from the flow "
                        f"U1A1E7RGKAlgebra) under phi, E -> mu ({tg} s)", bad == 0 and G._oracle is None,
                        f"disagreeing pairs {bad}; gauged class served from its stored tables: {G._oracle is None}"))
    checks.append(check("negative control: the same comparison with E -> mu^-1 fails",
                        _e7_gauged_products(A, G, Y, sigma=-1, stop_at_first=True) > 0))
    # the object layer's witness: the BPS chart, through cone_to_bps_iso (lattice charge, coefficient unchanged)
    from finite_kalgebra_objects import kalgebra_object
    obj = kalgebra_object("e7")
    iso = obj.iso("cone-frozen", "bps")
    lab = lambda i: Element.basis(((i, 1),))   # noqa: E731
    plain, flav = [], []
    for g in range(n):
        for h in range(n):
            (flav if _e7_mu_carrying(A.multiply(((g, 1),), ((h, 1),))) else plain).append((g, h))
    # a generic E7 BPS product can exhaust 6 GB (measured 2026-09-26 on a random mu-free pair), so the witness runs on
    # every ordered product of the six generators whose BPS charge is a single node charge up to sign (all mu-free),
    # and on four mu-carrying products timed at 0.7-5.8 s -- deterministic samples
    node = [i for i, gam in enumerate(M.E7_MULT_GENS_LATTICE) if sum(abs(x) for x in gam) == 1]
    ps = [(i, j) for i in node for j in node]
    fs = [(7, 54), (11, 83), (86, 78), (15, 83)]
    samples_ok = all(p in set(plain) for p in ps) and all(p in set(flav) for p in fs)
    tb = time.time()
    ok_plain = sum(bool(iso.verify_multiplicative([(lab(i), lab(j))], [])) for i, j in ps)
    ok_flav = sum(bool(iso.verify_multiplicative([(lab(i), lab(j))], [])) for i, j in fs)
    tb = round(time.time() - tb, 1)
    d0 = [g for g in range(n) if dvals[g] == 0]
    dn = [g for g in range(n) if dvals[g] != 0]
    rho0 = sum(bool(iso.verify_rho_equivariant([lab(g)], [])) for g in d0)
    rhon = sum(bool(iso.verify_rho_equivariant([lab(g)], [])) for g in dn)
    checks.append(check(f"kalgebra_object('e7') holds the cone presentation and the BPS chart; its witness is multiplicative on the {len(ps)} "
                        f"products of the {len(node)} single-node-charge generators, which carry no mu in a coefficient ({len(plain)} of "
                        f"the 8100 generator products are of that kind), and fails on {len(fs)} products with one; rho-equivariant on the "
                        f"{len(d0)} generators with delta = 0 and on none of the {len(dn)} others ({tb} s for the products)",
                        samples_ok and len(plain) == 3540 and len(ps) == 36 and ok_plain == len(ps) and ok_flav == 0
                        and rho0 == len(d0) == 49 and rhon == 0 and iso.verify_unit(),
                        f"mu-free {len(plain)}; multiplicative {ok_plain}/{len(ps)} mu-free, {ok_flav}/{len(fs)} mu-carrying; "
                        f"rho {rho0}/{len(d0)} and {rhon}/{len(dn)}"))
    return {"checks": checks,
            "population": {"generators": n, "cones": len(cones), "generator products (gauged witness)": 8100,
                           "BPS witness sample": {"mu-free": len(ps), "mu-carrying": len(fs), "rho": n},
                           "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the literals' counts and the maximal-clique enumeration; the gauged tables through phi",
                         "negative": "a delta_i moved by one breaks rho^10 = id; phi with E -> mu^-1 fails the product comparison"},
            "notes": "The object layer's cone-to-BPS witness carries coefficients across unchanged, so it cannot read a power of mu "
                     "(the audit); the products with mu are certified through the gauged class's tables, built by a "
                     "different engine (the flow), under the label map of `src/cone/u1e7_cone_kalgebra.py`.",
            "inputs": []}


def _e7_window(A):
    """identity; generators with delta = 0, -1 and -2; two cone monomials."""
    return [A.identity(), ((0, 1),), ((3, 1),), ((4, 1),), ((5, 1),), ((8, 1),), ((12, 1),), ((21, 1),),
            ((22, 1),), ((70, 1),), ((21, 1), (47, 1)), ((5, 2),)]


def check_e7_is_kalgebra(environment):
    from finite_e7_kalg import FiniteE7KAlgebra, E7_RHO_DELTA
    from zplus_ring import RLaurent, RElement
    checks = []
    t0 = time.time()
    A = FiniteE7KAlgebra()
    R = A.coefficient_ring()
    one = RLaurent(R, {0: RElement(R, {R.one_basis(): 1})})
    W = _e7_window(A)
    K = 4
    # the label-safe verifiers of the battery (bar, unit, associativity, rho a bijection; rho automorphism is
    # checked through rho_element by the contract itself)
    safe = ("verify_bar_involution", "verify_identity_in_basis", "verify_unit_law", "verify_associativity",
            "verify_rho_fixes_identity", "verify_rho_inverse", "verify_rho_is_automorphism")
    from checks_kq import VERIFIERS
    ver = {}
    for _ax, name, arity, takesK in VERIFIERS:          # run_k_verifiers' iteration, restricted to the label-safe ones
        if name not in safe:
            continue
        fn = getattr(A, name)
        it = ([()] if arity == 0 else [(a,) for a in W] if arity == 1 else itertools.product(W, W) if arity == 2
              else itertools.product(W[:4], W[:4], W[:4]))
        f, first = 0, None
        for args in it:
            if not (fn(*args, K=K) if takesK else fn(*args)):
                f, first = f + 1, first or args
        ver[name] = (f, first)
    n_fail = sum(f for f, _x in ver.values())
    checks.append(check(f"K1-K3 on all pairs of a window of {len(W)} labels (identity; generators with delta = 0, -1, -2; two cone "
                        f"monomials); rho an automorphism in its element form (the contract's verifier uses rho_element)",
                        n_fail == 0 and len(ver) == len(safe), "; ".join(f"{k} fails {f}" for k, (f, _x) in ver.items())))
    el = lambda l: Element({l: one})   # noqa: E731

    def tr(x, KK):
        return _e7_rps(A.trace_element(x, KK), KK)

    def rho(x):
        return A.rho_element(x)
    # K4, element form: Tr(rho(L_a) L_b) = delta_ab + O(q); rho^2-twisted cyclicity Tr(L_a L_b) = Tr(rho^2(L_b) L_a)
    bad_on, bad_cyc, bad_k5, bad_sym = [], [], [], []
    for a in W:
        ra = rho(el(a))
        ta = A.trace_element(el(a), K)
        if _e7_rps(A.trace_element(ra, K), K) != _e7_rps(ta.star(), K):
            bad_k5.append(a)
        for b in W:
            I = A.trace_element(A.multiply_elements(ra, el(b)), K)
            low = {k: v for k, v in _e7_rps(I, K).items() if k[0] <= 0}
            if low != ({(0, 0): 1} if a == b else {}):
                bad_on.append((a, b))
            Iba = A.trace_element(A.multiply_elements(rho(el(b)), el(a)), K)
            if _e7_rps(Iba, K) != _e7_rps(I.star(), K):
                bad_sym.append((a, b))
            lhs = tr(A.multiply_elements(el(a), el(b)), K)
            rhs = tr(A.multiply_elements(rho(rho(el(b))), el(a)), K)
            if lhs != rhs:
                bad_cyc.append((a, b))
    npairs = len(W) ** 2
    checks.append(check(f"K4, element form: Tr(rho(L_a) L_b) = delta_ab + O(q) (no negative powers) on all {npairs} pairs, with rho = rho_element",
                        not bad_on, f"failures {bad_on[:3]}"))
    checks.append(check(f"K4, element form: rho^2-twisted cyclicity Tr(L_a L_b) = Tr(rho^2(L_b) L_a) through q^{K} on all {npairs} pairs",
                        not bad_cyc, f"failures {bad_cyc[:3]}"))
    checks.append(check(f"F5, element form: Tr(rho(L_a)) = star Tr(L_a) on all {len(W)} labels, and I_ba = star I_ab on all {npairs} pairs",
                        not bad_k5 and not bad_sym, f"trace {bad_k5[:3]}; pairing {bad_sym[:3]}"))
    # negative control: the label-form pairing is off by a unit character on the generators with delta != 0 -- the
    # element form is needed, and the orthonormality check is sensitive to a unit character.  Since
    # rho(L_a) = mu^delta(a) L_rho(a), the label form reads Tr(L_rho(a) L_a) = mu^-delta(a) (1 + O(q)).
    lab_fail, lab_const = [], {}
    for a in W[1:10]:
        ok = A.verify_orthonormality(a, a, K=2)
        d = E7_RHO_DELTA[a[0][0]][0]
        c0 = {k[1]: v for k, v in _e7_rps(A.inner_product(a, a, K=2), 2).items() if k[0] == 0}
        lab_const[a[0][0]] = c0
        if not ok:
            lab_fail.append(a[0][0])
        if c0 != {-d: 1}:
            lab_fail.append(("q0", a[0][0]))
    want = sorted(a[0][0] for a in W[1:10] if E7_RHO_DELTA[a[0][0]][0] != 0)
    checks.append(check("negative control: the label-form verify_orthonormality fails exactly on the window generators with delta != 0, "
                        "where I_aa[q^0] = mu^-delta(a) (the audit)",
                        sorted(x for x in lab_fail if isinstance(x, int)) == want and not [x for x in lab_fail if isinstance(x, tuple)],
                        f"label-form failures {lab_fail}; expected {want}; q^0 terms {lab_const}"))
    return {"checks": checks, "population": {"labels": len(W), "pairs": npairs, "K": K, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the identity row; the q^0 term of I_aa equals 1 in the element form",
                         "negative": "the label-form pairing is caught off by a unit character (q^0 term mu^-delta(a)) on the generators with delta != 0"},
            "notes": "On this flavour-in-coefficients tier the contract's label-level rho is the bare permutation; orthonormality, "
                     "rho^2-twisted cyclicity and rho-equivariance of the trace are checked with rho_element (eq:comp-e7-rho).",
            "inputs": []}


E7_NAHM_LIVE_K = 16     # the live Nahm sum (90 s, timed 2026-09-26)
E7_FLOW_K = 12          # the live E7RGKAlgebra vacuum (16 s; q^14 took 162 s)
E7_RHO2_BOOT_K = 6      # the live bootstrap over rho^2-orbits (42 s)


def check_e7_traces(environment):
    from finite_e7_kalg import FiniteE7KAlgebra
    import elem_traces as ET
    checks = []
    t0 = time.time()
    A = FiniteE7KAlgebra()
    K = 22
    pinned, nahm20 = _e7_fixtures()
    # positive control of the transcription engine: theta(mu) is the Jacobi triple product
    KJ = 30
    jtp = {(0, 0): 1}
    for n in range(1, KJ // 2 + 1):
        jtp = _e7_times_binomial(jtp, 2 * n, 0, -1, KJ)
    for n in range(1, KJ + 1, 2):
        jtp = _e7_times_binomial(_e7_times_binomial(jtp, n, 1, 1, KJ), n, -1, 1, KJ)
    checks.append(check("positive control: the transcription's theta(mu) equals the Jacobi triple product through q^30",
                        _e7_theta(KJ) == jtp))
    # the vacuum: printed product == served == the Nahm sum (pinned through q^20, live through q^16) == the flow (q^12)
    vac = _e7_vacuum_tex(30)
    served_vac = _e7_rps(A.trace(A.identity(), K=30), 30)
    checks.append(check("served: Tr 1 of FiniteE7KAlgebra is the printed product eq:comp-e7-vacuum through q^30", served_vac == vac))
    checks.append(check("witness: the printed product equals the Nahm sum on the BPS spectrum, pinned in tests/test_e7_seeds.py, through q^20",
                        _e7_cut(vac, 20) == nahm20))
    tn = time.time()
    live = _e7_rps(ET._vacuum_rps("e7", E7_NAHM_LIVE_K), E7_NAHM_LIVE_K)
    tn = round(time.time() - tn, 1)
    checks.append(check(f"witness: the Nahm sum computed now equals the printed product through q^{E7_NAHM_LIVE_K} ({tn} s)",
                        live == _e7_cut(vac, E7_NAHM_LIVE_K)))
    from e7_rgkalgebra import E7RGKAlgebra
    tf = time.time()
    F = E7RGKAlgebra()
    flow = _e7_rps(F.trace(F.identity(), K=E7_FLOW_K), E7_FLOW_K)
    tf = round(time.time() - tf, 1)
    checks.append(check(f"independent witness: the RG flow E7RGKAlgebra gives the printed Tr 1 through q^{E7_FLOW_K} ({tf} s)",
                        flow == _e7_cut(vac, E7_FLOW_K)))
    first = {}
    for ten in (8, 14):
        wrong = _e7_vacuum_tex(20, ten=ten)
        diff = sorted({k[0] for k in set(wrong) | set(nahm20) if wrong.get(k, 0) != nahm20.get(k, 0)})
        first[ten] = diff[0] if diff else None
    checks.append(check("negative control: with 10 replaced by 8 or 14 (u = 4, 7) the product disagrees with the Nahm sum, from q^6 and q^8",
                        first == {8: 6, 14: 8}, f"first differing orders {first}"))
    # the seeds: nine printed recipes + the printed rule -> all 90, against the served traces and the pinned bootstrap
    ts = time.time()
    mine = _e7_all_seeds(K)
    ts = round(time.time() - ts, 1)
    served = {i: _e7_rps(A.trace(((i, 1),), K=K), K) for i in range(90)}
    checks.append(check(f"served: all 90 seed traces of FiniteE7KAlgebra equal the printed recipes eq:comp-e7-recipes over "
                        f"(q^2;q^2) theta(mu), transported by eq:comp-e7-rule, through q^{K} (transcription {ts} s)",
                        len(mine) == 90 and all(served[i] == mine[i] for i in range(90)),
                        f"disagreeing seeds {[i for i in range(90) if served[i] != mine.get(i)][:6]}"))
    checks.append(check("every seed is O(q), and the printed recipes N_0 and N_1 give equal traces",
                        all(min(q for q, _m in mine[i]) >= 1 for i in range(90)) and mine[0] == mine[1]))
    over = [(i, q, m) for i in range(90) for (q, m) in mine[i] if abs(m) > 4 * q]
    checks.append(check(f"the printed seeds obey the bootstrap's recorded mu-support hypothesis (|mu-degree| <= 4k at q^k) through q^{K}",
                        not over, f"violations {over[:4]}"))
    checks.append(check("witness: the eight distinct representatives equal the certified u(1) orthonormality bootstrap "
                        "(finite_kalgebras.u1_bootstrap.generate_u1, pinned in tests/test_e7_seeds.py) through q^20",
                        all(_e7_cut(mine[r], 20) == pinned[r] for r in pinned) and sorted(pinned) == [0, 2, 3, 5, 7, 12, 13, 21]))
    heads_ok = {}
    for key, (want, KH) in _E7_HEADS_TEX.items():
        got = _e7_cut(served_vac, KH) if key == "vacuum" else _e7_cut(served[key], KH)
        heads_ok[str(key)] = got == want
    checks.append(check("the printed heads eq:comp-e7-heads (Tr 1 through q^6, Tr s_5 through q^5, Tr s_0 = Tr s_1 through q^8)",
                        all(heads_ok.values()), str(heads_ok)))
    # the bootstrap over rho^2-orbits, run now: it does not use eq:comp-e7-rule, so it checks the 81 transported seeds
    from u1_bootstrap import generate_u1
    tb = time.time()
    rec = generate_u1("e7", E7_RHO2_BOOT_K, fold="rho2")
    tb = round(time.time() - tb, 1)
    boot = {i: {(q, m[0] if isinstance(m, tuple) else m): c for q, d in rec["orbits"][i].items() if q <= E7_RHO2_BOOT_K
                for m, c in d.items() if c}
            for i in range(90)}
    checks.append(check(f"witness: the orthonormality bootstrap over rho^2-orbits (no rho-rule) gives all 90 seeds equal to the printed ones "
                        f"through q^{E7_RHO2_BOOT_K} ({tb} s)",
                        all(boot[i] == _e7_cut(mine[i], E7_RHO2_BOOT_K) for i in range(90))))
    # negative controls on the seeds
    pert = {r: [(s, t, dict(terms)) for (s, t, terms) in v] for r, v in _E7_RECIPES_TEX.items()}
    pert[13][0][2][(3, 4)] = -1          # N_13: + Theta_34 -> - Theta_34 in the q^-1 group
    bad13 = _e7_seed_rep(13, 20, recipes=pert)
    checks.append(check("negative control: N_13 with the sign of one Theta term flipped is caught by the pinned bootstrap",
                        _e7_cut(bad13, 20) != pinned[13]))
    flipped = _e7_all_seeds(12, sign=+1)
    n_caught = sum(1 for i in range(90) if flipped[i] != _e7_cut(served[i], 12))
    checks.append(check("negative control: the rule with mu^{+delta_i} in place of mu^{-delta_i} is caught on the orbits with delta != 0",
                        n_caught > 0, f"{n_caught} seeds differ"))
    return {"checks": checks,
            "population": {"seeds": 90, "K": K, "fixtures": "q^20", "Nahm live K": E7_NAHM_LIVE_K, "flow K": E7_FLOW_K,
                           "rho2 bootstrap K": E7_RHO2_BOOT_K, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "theta(mu) against the Jacobi triple product; the pinned bootstrap and Nahm sum through q^20; the flow",
                         "negative": "the levels u = 4, 7 in the vacuum; one sign flipped in N_13; mu^{+delta} in the rule"},
            "notes": "The vacuum, the blocks, the nine recipes and the rule are transcribed from the companion's text with their own "
                     "exact arithmetic (no import of finite_kalgebras/e7_seeds.py).  The bootstrap witnesses rest on their recorded "
                     "mu-support hypothesis (the q^k coefficient of a seed has mu-degree at most 4k).",
            "inputs": []}


def check_e7_gauged(environment):
    import finite_e7_kalg as M
    import u1e7_cone_kalgebra as GM
    from u1e7_cone_kalgebra import U1E7ConeKAlgebra
    checks = []
    t0 = time.time()
    G = U1E7ConeKAlgebra()
    cd = G.cone_data()
    gens = list(cd.mult_gens())
    c0s = sorted({g[1][0] for g in gens})
    has_E = ((), (0, 1)) in gens and ((), (0, -1)) in gens
    checks.append(check("U1E7ConeKAlgebra: 202 generators (E^{+-1} and 200 others of magnetic charge -3..3), 4160 cones, served from its "
                        "two stored tables (no flow loaded)",
                        len(gens) == 202 and has_E and c0s == [-3, -2, -1, 0, 1, 2, 3] and len(cd.cones()) == 4160
                        and G._oracle is None,
                        f"generators {len(gens)}, magnetic charges {c0s}, cones {len(cd.cones())}"))
    # eq:comp-e7-gauged, transcribed: Tr L_l = [mu^{-r}]((q^2;q^2)^2 Tr_{[A1,E7]} L_z), phi(L_l) = mu^r L_z
    KG = 14
    meas = {(0, 0): 1}
    for n in range(1, KG // 2 + 1):
        meas = _e7_times_binomial(_e7_times_binomial(meas, 2 * n, 0, -1, KG), 2 * n, 0, -1, KG)

    def gauged(series, r, KK, measure=True, sgn=-1):
        s = _e7_mul(series, meas, KK) if measure else series
        return {q: c for (q, m), c in s.items() if m == sgn * r and q <= KK and c}

    def qs(x, KK):
        out = {}
        for e, v in x.coeffs.items():
            c = sum(int(t) for t in v.terms.values())
            if c and e <= KK:
                out[e] = c
        return out
    vac = _e7_vacuum_tex(KG)
    tower = {n: qs(G.trace(((), (0, n)), KG), KG) for n in range(-3, 4)}
    checks.append(check(f"the E-tower Tr E^n (n = -3..3) is the printed formula at z = 1: [mu^-n]((q^2;q^2)^2 Tr 1) through q^{KG}",
                        all(tower[n] == gauged(vac, n, KG) for n in tower), str({n: tower[n] for n in (0, 1)})))
    checks.append(check("the printed heads: Tr 1 = 1 - q^2 + q^6 + q^12 + ... and Tr E^{+-1} = -q^3 + ... (through q^12)",
                        {e: c for e, c in tower[0].items() if e <= 12} == {0: 1, 2: -1, 6: 1, 12: 1}
                        and {e: c for e, c in tower[1].items() if e <= 12} == {3: -1} == {e: c for e, c in tower[-1].items() if e <= 12}))
    # the neutral generator labels shifted by E^n: Tr = [mu^-n]((q^2;q^2)^2 Tr s_g) with the printed seeds
    KS = 10
    seeds = _e7_all_seeds(KS)
    Y = {g: GM._E7_GENERATOR_PREIMAGES[tuple(M.E7_MULT_GENS_LATTICE[g])] for g in range(90)}
    bad, flipped_caught, nomeasure_caught = [], 0, 0
    for g in range(90):
        w, c1 = Y[g]
        for n in (-1, 0, 1):
            got = qs(G.trace((w, (0, c1 + n)), KS), KS)
            if got != gauged(seeds[g], n, KS):
                bad.append((g, n))
            if n and gauged(seeds[g], n, KS, sgn=+1) != got:
                flipped_caught += 1
            if gauged(seeds[g], n, KS, measure=False) != got:
                nomeasure_caught += 1
    checks.append(check(f"the 90 neutral generator labels shifted by E^n (n = -1, 0, 1) trace by eq:comp-e7-gauged with the printed seeds, through q^{KS}",
                        not bad, f"failures {bad[:4]}"))
    checks.append(check("negative controls: the [mu^{+r}] reading, and the formula without (q^2;q^2)^2, are caught",
                        flipped_caught > 0 and nomeasure_caught > 0, f"{flipped_caught} and {nomeasure_caught} labels differ"))
    mag = [g for g in gens if g[1][0] != 0]
    zero = all(not qs(G.trace(g, 8), 8) for g in mag)
    checks.append(check(f"every generator of nonzero magnetic charge ({len(mag)}) has trace 0 through q^8", zero))
    # witness: the gauged Nahm sum (the class's own witness method, not on its serving path) for the E-tower
    P = G._vacuum_mu(10)
    ok_nahm = all({q: md[-n] for q, md in P.items() if md.get(-n, 0)} == qs(G.trace(((), (0, n)), 10), 10) for n in range(-3, 4))
    checks.append(check("witness: the E-tower equals the gauged Nahm sum on the E7 BPS spectrum through q^10", ok_nahm))
    return {"checks": checks, "population": {"E-tower": 7, "neutral labels": 270, "magnetic generators": len(mag),
                                             "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the E-tower against the gauged Nahm sum", "negative": "[mu^{+r}] in place of [mu^{-r}]; the measure dropped"},
            "notes": "The formula and the seeds are transcribed from the companion's text; the label map phi enters only through the class's "
                     "dictionary of generator preimages (_E7_GENERATOR_PREIMAGES), whose product certificate is comp:e7/implementation.",
            "inputs": []}


# ============================================================================
# comp:a1deven — K_q([A_1, D_{2k+2}]) and its u(1) gauging: curves on the once-punctured polygon
# ============================================================================
# Every rule the companion prints for the even-D family is transcribed here from the text, independently of
# the implementation modules (u1a1deven_geometric_frame, u1a1deven_seed_characters, ungauge_kalgebra): the
# curves and their crossing (drawn on the doubled polygon), the magnetic charge, the collapse of the
# boundary edge {1, 2}, rho with its E-drift, the gauge tower, the four seed rules, the ungauging sum, and the
# SU(3) restriction of the k = 1 member.  Series are {q-power: {mu-power: int}} with mu the SU(2) fugacity (the
# _qm helpers of comp:a1dodd); the ungauged ones are {q-power: {(mu-power, z-power): int}}.
_DV_KS = (1, 2, 3)


def _dv_curves(k):
    """The (2k+2)(2k+1) curves (x, l): from marked point x to x + l, l boundary edges away from the puncture."""
    n = 2 * k + 2
    return [(x, l) for x in range(n) for l in range(2, n + 1)]


def _dv_charge(c, k):
    """The magnetic charge c0: 0 on odd length; on even length -1 if the endpoints have the parity of marked
    point 2, +1 if not."""
    x, l = c
    return 0 if l % 2 else (-1 if (x - 2) % 2 == 0 else 1)


def _dv_chords(c, k):
    """A curve drawn on the doubled (2(2k+2))-gon: its chord and the chord's half-turn image (a loop: one diameter)."""
    n = 2 * k + 2
    N = 2 * n
    x, l = c
    if l == n:
        return [tuple(sorted((x % N, (x + n) % N)))]
    return [tuple(sorted((x % N, (x + l) % N))), tuple(sorted(((x + n) % N, (x + l + n) % N)))]


def _dv_cross(g, h, k):
    """Do two curves cross away from their endpoints?  (Some chord of one strictly interleaves one of the other.)"""
    if g == h:
        return False
    for a, b in _dv_chords(g, k):
        for c, d in _dv_chords(h, k):
            if len({a, b, c, d}) == 4 and ((a < c < b) != (a < d < b)):
                return True
    return False


def _dv_collapse(c, k):
    """The curve of the once-punctured (2k+1)-gon obtained by collapsing the boundary edge {1, 2} (marked points
    v >= 2 renumbered v - 1); None where the curve becomes a boundary edge, i.e. the unit."""
    n = 2 * k + 2
    x, l = c
    x %= n
    f = (lambda v: v if v < 2 else v - 1)   # noqa: E731
    if l == n:
        return (f(x), n - 1)
    pts = [(x + s) % n for s in range(l + 1)]
    ll = l - 1 if any(pts[i] == 1 and pts[i + 1] == 2 for i in range(l)) else l
    return None if ll == 1 else (f(x), ll)


def _dv_rho(curves, e, k, drift_point=1):
    """rho on a chi-stripped label (curves, e) as printed: each curve rotated x -> x + 1, and e -> -e + sum m*d,
    d = -(the number of the curve's endpoints at the drift point, marked point 1 in the text; a loop there counts
    two).  drift_point=None drops the drift (a negative control)."""
    n = 2 * k + 2
    out, d = [], 0
    for (x, l), m in curves:
        out.append((((x + 1) % n, l), m))
        if drift_point is not None:
            d -= m * sum(1 for v in (x % n, (x + l) % n) if v == drift_point % n)
    return tuple(sorted(out)), -e + d


def _dv_label(curves, e=0, kappa=0):
    return (tuple(sorted(curves)), e, kappa)


def _dv_chi_at(r, x):
    """chi_r(q^x) = q^{rx} + q^{(r-2)x} + ... + q^{-rx} as {q-power: int}; chi_{-1} = 0."""
    out = {}
    for t in range(r + 1):
        out[(r - 2 * t) * x] = out.get((r - 2 * t) * x, 0) + 1
    return out


def _dv_lp(*terms):
    """sum of coefficient * q^shift * polynomial over (coefficient, {power: int}, shift) terms."""
    out = {}
    for co, lp, sh in terms:
        for e, c in lp.items():
            out[e + sh] = out.get(e + sh, 0) + co * c
    return {e: c for e, c in out.items() if c}


def _dv_lpmul(a, b):
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            out[ea + eb] = out.get(ea + eb, 0) + ca * cb
    return {e: c for e, c in out.items() if c}


def _dv_points(c, k):
    n = 2 * k + 2
    x, l = c
    return {(x + t) % n for t in range(l + 1)} if l < n else set(range(n))


def _dv_configuration(curves, k):
    """('curve', j) for the curve (0, 2j+1); for a pair of a -1 and a +1 curve, ('nested', inner length, g1, g2,
    inner charge) or ('side', length of the -1 curve, length of the +1 curve, g1, g2), the g's the numbers of
    marked points strictly between the two curves (as the text defines them); None otherwise."""
    n = 2 * k + 2
    if len(curves) == 1:
        (c, m), = curves
        return ("curve", (c[1] - 1) // 2) if (m == 1 and c[1] % 2 == 1 and c[0] == 0) else None
    if len(curves) != 2 or any(m != 1 for _c, m in curves):
        return None
    (c1, _m1), (c2, _m2) = curves
    ch = {c1: _dv_charge(c1, k), c2: _dv_charge(c2, k)}
    if sorted(ch.values()) != [-1, 1]:
        return None
    cm = next(c for c in ch if ch[c] == -1)
    cp = next(c for c in ch if ch[c] == 1)
    sm, sp = _dv_points(cm, k), _dv_points(cp, k)
    if cp[1] == n or sm < sp:
        outer, inner = cp, cm
    elif cm[1] == n or sp < sm:
        outer, inner = cm, cp
    else:
        return ("side", cm[1], cp[1], (cp[0] - cm[0] - cm[1]) % n - 1, (cm[0] - cp[0] - cp[1]) % n - 1)
    (y, L), (x, l) = outer, inner
    return ("nested", l, (x - y) % n - 1, (y + L - x - l) % n - 1, ch[inner])


def _dv_representative(curves, e, k, drift_point=1):
    """The representative of the rho-orbit of a seed: among its 2k+2 rotations (rho with its E-drift), the one whose
    sorted list of curves is lexicographically smallest, with the E-power rho carries it to."""
    cur, ee = tuple(sorted(curves)), e
    best = None
    for _ in range(2 * k + 2):
        if best is None or cur < best[0]:
            best = (cur, ee)
        cur, ee = _dv_rho(cur, ee, k, drift_point)
    return best


def _dv_ce(conf, k, e, n, variant=None):
    """c_e(n) of the printed rule for the configuration conf, as {q-power: int}, or None off its support; variant
    perturbs the rule for the negative controls ('nested-rule': the nested rule in place of the curve rule;
    'corner': the curve's support moved to e >= |n| + 2; 'swap': the nested gaps swapped; 'old-side': the retired
    side-by-side power q^{n g/2})."""
    p = k + 1
    sgn = -1 if (p * n) % 2 else 1
    kind = conf[0]
    if kind == "curve" and variant == "nested-rule":
        kind, conf = "nested", ("nested", 2 * conf[1], 0, 0, -1)
    if kind in ("tower", "nested"):
        if not (e >= abs(n) + 1 and (e - n - 1) % 2 == 0):
            return None
        base = (p * (e * e - n * n - 1)) // 2
        if kind == "tower":
            return {base: sgn}
        _kind, inner, g1, g2 = conf[:4]
        if variant == "swap":
            g1, g2 = g2, g1
        S = {}
        for i in range(inner // 2):
            r = (g1 + g2) // 2 + 1 + 2 * i
            S = _dv_lp((1, S, 0), (1, _dv_chi_at(r + 1, e), 2 * i + 1), (1, _dv_chi_at(r - 1, e), 2 * i - 1),
                       (-1, _dv_lpmul(_dv_chi_at(1, n), _dv_chi_at(r, e)), 2 * i))
        eps = -1 if ((g1 - g2) // 2) % 2 else 1
        return _dv_lp((sgn * eps, S, base + n * (g1 - g2) // 2))
    if kind == "curve":
        on = (e >= abs(n) + 2) if variant == "corner" else (e >= abs(n - 1) + 1)
        if not (on and (e - n) % 2 == 0):
            return None
        j = conf[1]
        s2 = -1 if (p + j) % 2 else 1
        F = _dv_lp((1, _dv_chi_at(j, e), 0), (-1, _dv_chi_at(j - 1, e), -n))
        return _dv_lp((sgn * s2, F, (p * (e * e - n * n)) // 2 + (n - 1) * (p - j)))
    _kind, l1, l2, g1, g2 = conf
    if not (e >= abs(n) and (e - n) % 2 == 0):
        return None
    a, b, g = l1 // 2, l2 // 2, g1 + g2
    F = _dv_lpmul(_dv_lpmul(_dv_chi_at(a - 1, e), _dv_chi_at(b - 1, e)),
                  _dv_lp((1, _dv_chi_at(1, e), 0), (-1, _dv_chi_at(1, n), 0)))
    sh = g // 2 if variant == "old-side" else (g2 - g1) // 2
    s = -1 if (g // 2) % 2 else 1
    return _dv_lp((sgn * s, F, (p * (e * e - n * n)) // 2 - g // 2 - 2 + n * sh))


_DV_INVW: dict = {}


def _dv_inv_weyl(K):
    """1 / prod_{j >= 1} (1 - mu^2 q^{2j})(1 - mu^-2 q^{2j}) through q^K."""
    if K not in _DV_INVW:
        out = {0: {0: 1}}
        j = 1
        while 2 * j <= K:
            for f in (2, -2):
                out = _qm_mul(out, {2 * j * t: {f * t: 1} for t in range(K // (2 * j) + 1)}, K)
            j += 1
        _DV_INVW[K] = out
    return _DV_INVW[K]


def _dv_seed_series(k, curves, n, K, variant=None, drift_point=1):
    """Tr(L_curves E^n) through q^K from the printed rules: the gauge tower (curves empty), or the seed carried to
    its representative and its rule; sum_e c_e(n) chi_{e-1}(mu), chi_{e-1} = (mu^e - mu^-e)/(mu - mu^-1), times
    the inverse Weyl product.  ValueError on a label the printed rules do not cover."""
    if curves:
        rep, nn = _dv_representative(curves, n, k, drift_point)
        conf = _dv_configuration(rep, k)
        if conf is None or (conf[0] == "nested" and conf[4] != -1):
            raise ValueError(f"not a seed of the printed rules: {curves!r} (representative {rep!r})")
    else:
        conf, nn = ("tower",), n
    num = {}
    e = above = 0
    while above < 4 or e <= abs(nn) + 4:
        e += 1
        c = _dv_ce(conf, k, e, nn, variant)
        if c:
            for qe, v in c.items():
                if qe <= K:
                    for t in range(e):
                        _qm_add(num, qe, e - 1 - 2 * t, v)
            above = above + 1 if min(c) > K else 0
        if e > 8 * (K + abs(nn) + 4 * k + 16):
            raise RuntimeError(f"the support sum of {curves!r} at E^{n} did not settle")
    return _qm_mul(_qm_clean(num), _dv_inv_weyl(K), K)


def _dv_ungauged(k, curves, K, measure_power=2):
    """The printed ungauged trace of the E-free section L_(F, 0, 0), F = curves: sum_n z^n Tr(L_F E^n) over the
    window |n| <= K + 1, each gauged trace from the printed rules, divided by (q^2; q^2)^{measure_power}."""
    acc = {}
    for n in range(-(K + 1), K + 2):
        for q, row in _dv_seed_series(k, curves, n, K).items():
            for m, c in row.items():
                d = acc.setdefault(q, {})
                d[(m, n)] = d.get((m, n), 0) + c
    inv = _dv_inv_measure_power(K, measure_power)
    out = {}
    for q, row in acc.items():
        for dq, c2 in inv.items():
            if q + dq <= K:
                d = out.setdefault(q + dq, {})
                for key, c in row.items():
                    d[key] = d.get(key, 0) + c * c2
    return {q: {key: c for key, c in d.items() if c} for q, d in out.items() if any(d.values())}


def _dv_inv_measure_power(K, power):
    """1 / (q^2; q^2)^power_infinity through q^K as {q-power: int}."""
    inv = {0: 1}
    for _ in range(power):
        for j in range(1, K // 2 + 1):
            nxt = {}
            for e, c in inv.items():
                m = 0
                while e + 2 * j * m <= K:
                    nxt[e + 2 * j * m] = nxt.get(e + 2 * j * m, 0) + c
                    m += 1
            inv = nxt
    return inv


def _dv_su2u1_to_qm(series, K):
    """An R(SU(2)) x R(U(1))-valued trace (keys (kappa, (f,))) -> {q: {(mu-power, z-power): int}}."""
    out = {}
    for e, r in series.coeffs.items():
        if e > K:
            continue
        for (kap, (f,)), c in r.terms.items():
            for t in range(kap + 1):
                d = out.setdefault(e, {})
                d[(kap - 2 * t, f)] = d.get((kap - 2 * t, f), 0) + c
    return {q: {key: c for key, c in d.items() if c} for q, d in out.items() if any(d.values())}


def _dv_is_seed(curves, k):
    """An odd curve, or a non-crossing pair of a +1 and a -1 curve (the text's seeds, E-power aside)."""
    if len(curves) == 1:
        (c, m), = curves
        return m == 1 and c[1] % 2 == 1
    if len(curves) == 2 and all(m == 1 for _c, m in curves):
        (a, _ma), (b, _mb) = curves
        return sorted((_dv_charge(a, k), _dv_charge(b, k))) == [-1, 1] and not _dv_cross(a, b, k)
    return False


def _dv_composites(k, count, seed):
    """A deterministic sample of charge-0 labels that are not seeds: one to three mutually non-crossing curves of
    one maximal non-crossing set, with powers up to 2, E-powers -2..2."""
    import random
    from itertools import combinations
    rnd = random.Random(seed)
    cs = _dv_curves(k)
    out, seen = [], set()
    tries = 0
    while len(out) < count and tries < 20000:
        tries += 1
        size = rnd.choice((1, 2, 2, 3))
        pick = rnd.sample(cs, size)
        if any(_dv_cross(a, b, k) for a, b in combinations(pick, 2)):
            continue
        F = tuple(sorted((c, rnd.choice((1, 1, 2))) for c in pick))
        if sum(m * _dv_charge(c, k) for c, m in F) != 0 or _dv_is_seed(F, k):
            continue
        lab = (F, rnd.randint(-2, 2), 0)
        if lab not in seen:
            seen.add(lab)
            out.append(lab)
    return out


def _dv_printed_products_k1():
    """The four printed k = 1 products, as (left curve, right curve, {label: {q-power: int}})."""
    L = lambda curves, e=0, kappa=0: (tuple(sorted(curves)), e, kappa)   # noqa: E731
    return [
        ((0, 2), (1, 2), {L(()): {0: 1}, L((((0, 3), 1),), 1): {-1: 1}}),
        ((0, 3), (2, 3), {L((((1, 3), 1),)): {-1: 1}, L((((3, 3), 1),)): {1: 1}, L((), 0, 1): {0: 1}, L((), 1): {0: 1}}),
        ((0, 2), (1, 4), {L((((1, 3), 1),), 1): {-2: 1}, L((((2, 3), 1),)): {0: 1}, L((), 1, 1): {-1: 1}}),
        ((1, 4), (0, 2), {L((((1, 3), 1),), 1): {2: 1}, L((((2, 3), 1),)): {0: 1}, L((), 1, 1): {1: 1}}),
        ((0, 4), (1, 4), {L(()): {0: 1}, L((((1, 3), 1),), 0, 1): {-1: 1}, L((((1, 3), 2),)): {-2: 1}}),
    ]


def _dv_elem(d):
    return Element({lab: LaurentPoly(dict(v)) for lab, v in d.items()})


def check_a1deven_implementation(environment):
    from math import comb
    from itertools import combinations
    import u1a1deven_geometric_frame as FR
    from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra, u1a1deven_cone_dodd_section_iso
    from a1deven_kalg import A1DevenKAlg
    from a1dn_kalg import A1DnKAlg
    checks = []
    t0 = time.time()
    pop = {}
    E, Einv = ((), 1, 0), ((), -1, 0)
    for k in _DV_KS:
        n = 2 * k + 2
        C = U1A1DevenConeKAlgebra(k)
        cs = _dv_curves(k)
        cd = C.cone_data()
        cones = cd.cones()
        ok_count = len(cs) == n * (n - 1) and all(C.curve(*c) == ((((c[0], c[1]), 1),), 0, 0) for c in cs) and \
            len(cones) == comb(4 * k + 2, 2 * k + 1) and all(len(cn) == 2 * k + 1 + 2 for cn in cones)
        checks.append(check(f"k={k}: {n * (n - 1)} curves (x, l), and {comb(4 * k + 2, 2 * k + 1)} = C({4 * k + 2}, {2 * k + 1}) maximal cones, each {2 * k + 1} pairwise non-crossing curves with E^(+-1)",
                            ok_count, f"curves {len(cs)}, cones {len(cones)}"))
        # crossing (transcribed on the doubled polygon) <=> not q-commuting; the cocycle from the collapsed curves
        D = A1DnKAlg(2 * k + 1)
        bad_q, bad_c, bad_anti, n_pairs = [], [], [], 0
        cval = {}
        for g in cs:
            for h in cs:
                if g == h:
                    continue
                crosses = _dv_cross(g, h, k)
                if cd.q_commute(g, h) == crosses:
                    bad_q.append((g, h))
                if crosses:
                    continue
                n_pairs += 1
                (lab, co), = C.multiply(C.curve(*g), C.curve(*h)).terms.items()
                (cexp, one), = _lpc(co).items()
                gb, hb = _dv_collapse(g, k), _dv_collapse(h, k)
                if gb is None or hb is None:
                    want = 0
                else:
                    (_l2, co2), = D.multiply(D.curve(*gb), D.curve(*hb)).terms.items()
                    (want, _one2), = _lpc(co2).items()
                cval[cexp] = cval.get(cexp, 0) + 1
                if one != 1 or cexp != want or lab != _dv_label(((g, 1), (h, 1))):
                    bad_c.append((g, h, cexp, want))
                (_lab3, co3), = C.multiply(C.curve(*h), C.curve(*g)).terms.items()
                if _lpc(co3) != {-cexp: 1}:
                    bad_anti.append((g, h))
        checks.append(check(f"k={k}: two curves q-commute iff they do not cross (every ordered pair of distinct curves, crossing transcribed on the doubled polygon)", not bad_q, str(bad_q[:3])))
        checks.append(check(f"k={k}: for the {n_pairs} ordered non-crossing pairs L_g L_h = q^c(g,h) L_(g,h), c antisymmetric, c = the exponent of A1DnKAlg({2 * k + 1})'s product of the curves after collapsing the edge {{1,2}} (0 when one becomes a boundary edge); values {sorted(cval.items())}",
                            not bad_c and not bad_anti, f"{bad_c[:3]} {bad_anti[:3]}"))
        # E q-commutes: E L_c = q^(-2 c0) L_c E
        bad_e = [c for c in cs
                 if C.multiply(E, C.curve(*c)) != _shift(C.multiply(C.curve(*c), E), -2 * _dv_charge(c, k))]
        checks.append(check(f"k={k}: E L_c = q^(-2 c0(c)) L_c E on every curve, c0 the printed parity rule", not bad_e, str(bad_e[:3])))
        # the canonical normalisation of a cone monomial: L_(F,e) = q^(-sum_{i<j} c(a_i,a_j) - e c0(F)) L_a1...L_am E^e
        bad_norm = 0
        n_norm = 0
        for cn in list(cones)[:25]:
            curves = sorted(x for x in cn if x not in FR._E_LETTERS)
            for e in (-1, 2):
                acc_lab, acc_q = C.identity(), 0
                for c in curves:
                    (acc_lab, co), = C.multiply(acc_lab, C.curve(*c)).terms.items()
                    acc_q += next(iter(_lpc(co)))
                for _ in range(abs(e)):
                    (acc_lab, co), = C.multiply(acc_lab, E if e > 0 else Einv).terms.items()
                    acc_q += next(iter(_lpc(co)))
                cpair = 0
                for a, b in combinations(curves, 2):
                    gb, hb = _dv_collapse(a, k), _dv_collapse(b, k)
                    if gb is not None and hb is not None:
                        (_l, co2), = D.multiply(D.curve(*gb), D.curve(*hb)).terms.items()
                        cpair += next(iter(_lpc(co2)))
                want_q = cpair + e * sum(_dv_charge(c, k) for c in curves)
                n_norm += 1
                if acc_lab != _dv_label(tuple((c, 1) for c in curves), e) or acc_q != want_q:
                    bad_norm += 1
        checks.append(check(f"k={k}: L_a1...L_am E^e = q^(sum_(i<j) c(a_i,a_j) + e c0(F)) L_(F,e) on {n_norm} cone monomials (all curves of 25 maximal cones, e = -1, 2)", bad_norm == 0, f"{bad_norm} fail"))
        # rho as printed, rho(E) = E^-1, the full turn
        bad_rho, n_rho = [], 0
        for c in cs:
            for e in (-1, 0, 2):
                for kap in (0, 1):
                    n_rho += 1
                    cur, ee = _dv_rho((((c[0], c[1]), 1),), e, k)
                    if C.rho(C.curve(c[0], c[1], e, kap)) != (cur, ee, kap):
                        bad_rho.append((c, e, kap))
        ok_E = C.rho(E) == Einv and C.rho(Einv) == E
        full = []
        for c in cs:
            for e in (-1, 0, 2):
                x = C.curve(c[0], c[1], e)
                for _ in range(n):
                    x = C.rho(x)
                if x != C.curve(c[0], c[1], e + 2 * _dv_charge(c, k)):
                    full.append((c, e))
        checks.append(check(f"k={k}: rho(L_(x,l) E^e) = L_(x+1,l) E^(-e+d), d = -(endpoints at marked point 1), on all {n_rho} curve labels (e = -1, 0, 2; kappa = 0, 1); rho(E) = E^-1; the full turn rho^{n}(L_(c,e)) = L_(c,e+2c0) on every curve (e = -1, 0, 2)",
                            not bad_rho and ok_E and not full, f"rho {bad_rho[:3]}; full turn {full[:3]}"))
        wrong = sum(1 for c in cs for e in (-1, 0, 2)
                    if C.rho(C.curve(c[0], c[1], e)) != _dv_label(*_dv_rho((((c[0], c[1]), 1),), e, k, drift_point=3)))
        checks.append(check(f"k={k}: negative control: rho with its drift at marked point 3 disagrees with the class", wrong > 0, f"{wrong} of {3 * len(cs)} labels differ"))
        # every product of two crossing curves: each coefficient a single power of q, the SU(2) weight at most 1
        bad_shape, n_cross = [], 0
        for g in cs:
            for h in cs:
                if _dv_cross(g, h, k):
                    n_cross += 1
                    for lab, co in C.multiply(C.curve(*g), C.curve(*h)).terms.items():
                        if len(_lpc(co)) != 1 or set(_lpc(co).values()) != {1} or lab[2] > 1:
                            bad_shape.append((g, h, lab))
        checks.append(check(f"k={k}: on all {n_cross} ordered crossing pairs every coefficient of L_g L_h is a single power of q, times chi_1 at most", not bad_shape, str(bad_shape[:3])))
        # the printed exponent rules, applied to the skein model's resolution states (the states taken from the frame
        # module; they are certified by the flow comparison below), with the cocycle transcribed above
        ccache = {}

        def cc(a, b):
            if a == b:
                return 0
            if (a, b) not in ccache:
                ab, bb = _dv_collapse(a, k), _dv_collapse(b, k)
                if ab is None or bb is None:
                    ccache[(a, b)] = 0
                else:
                    (_l, co), = D.multiply(D.curve(*ab), D.curve(*bb)).terms.items()
                    ccache[(a, b)] = next(iter(_lpc(co)))
            return ccache[(a, b)]

        def bulk(u, curves, e):
            return sum(m * cc(u, c) for c, m in curves) + e * _dv_charge(u, k)

        bad_rule, kinds_seen = [], {}
        for g in cs:
            for h in cs:
                if not _dv_cross(g, h, k):
                    continue
                kind, states = FR._resolutions(g, h, k)
                kinds_seen[kind] = kinds_seen.get(kind, 0) + 1
                want = {}
                if kind == "two crossings":
                    (chi_state,) = [s_ for s_ in states if s_[2]]
                    A_ = bulk(g, chi_state[0], chi_state[1])
                for curves, e, chi, nB in states:
                    if kind in ("one crossing", "two loops"):
                        qexp = bulk(g, curves, e)
                    elif kind == "two crossings":
                        qexp = A_ + 1 - nB
                    else:
                        lam, eps = (g, 1) if kind == "loop and curve" else (h, -1)
                        qexp = eps * bulk(lam, curves, e)
                    lab = (curves, e, chi)
                    want.setdefault(lab, {})
                    want[lab][qexp] = want[lab].get(qexp, 0) + 1
                if C.multiply(C.curve(*g), C.curve(*h)) != _dv_elem(want):
                    bad_rule.append((g, h, kind))
        checks.append(check(f"k={k}: the printed exponents (q^(sum_f c(g,f)) for one crossing and two loops, q^(+-sum_f c(loop,f)) for a loop and a curve, q^(A+1-#B) for two crossings), on the model's resolution states with E on the edge {{1,2}} and chi_1 for a puncture loop, reproduce every product of two crossing curves ({kinds_seen})",
                            not bad_rule, str(bad_rule[:3])))
        # the products against the flow U1A1DevenViaDoddRG(k), through the closed-form bijection of labels
        iso = u1a1deven_cone_dodd_section_iso(k)
        gens = [C.curve(*c) for c in cs] + [E, Einv]
        pairs = [(Element.basis(a), Element.basis(b)) for a in gens for b in gens]
        ok_flow = iso.verify_multiplicative(pairs, [])
        kpairs = [(Element.basis(C.curve(c[0], c[1], 0, 1)), Element.basis(C.curve(d[0], d[1], 1, 1))) for c in cs[:6] for d in cs]
        ok_flow_k = iso.verify_multiplicative(kpairs, [])
        checks.append(check(f"k={k}: products equal the flow U1A1DevenViaDoddRG({k})'s on all {len(pairs)} ordered generator pairs (curves and E^(+-1)) and on {len(kpairs)} pairs with SU(2) weights and E-powers",
                            ok_flow and ok_flow_k, f"generators {ok_flow}, weighted {ok_flow_k}"))
        # the ungauged generators
        B = A1DevenKAlg(k)
        singles = [(((c, 1),), 0, 0) for c in cs if _dv_charge(c, k) == 0]
        pairs2 = sorted((tuple(sorted(((a, 1), (b, 1)))), 0, 0) for a in cs for b in cs
                        if _dv_charge(a, k) == 1 and _dv_charge(b, k) == -1 and not _dv_cross(a, b, k))
        want = sorted(singles) + pairs2
        got = sorted(B.mult_generators())
        checks.append(check(f"k={k}: A1DevenKAlg({k})'s generators are the {len(singles)} charge-0 curves and the {len(pairs2)} non-crossing (+1, -1) pairs ({len(want)} in all), geometric_label is the label, and rho(z) = z^-1",
                            got == sorted(want) and all(B.geometric_label(g) == g for g in got) and B.rho(((), 1, 0)) == ((), -1, 0),
                            f"{len(got)} generators"))
        pop[k] = {"curves": len(cs), "cones": len(cones), "noncrossing pairs": n_pairs, "crossing pairs": n_cross,
                  "flow pairs": len(pairs) + len(kpairs), "ungauged generators": [len(singles), len(pairs2)]}
    # the printed products at k = 1
    C1 = U1A1DevenConeKAlgebra(1)
    bad_pr = [(g, h) for g, h, rhs in _dv_printed_products_k1() if C1.multiply(C1.curve(*g), C1.curve(*h)) != _dv_elem(rhs)]
    checks.append(check("k=1: the printed products L_(0,2)L_(1,2), L_(0,3)L_(2,3), L_(0,2)L_(1,4), L_(1,4)L_(0,2), L_(0,4)L_(1,4)", not bad_pr, str(bad_pr)))
    # [A_1, D_4] as the k = 1 member: SU3ADKAlg's geometric labels are the printed curves of A1DevenKAlg(1), a monomial
    # of a tile the union of its letters' curves (multiplicities added), with the SU(3) weight
    from su3_ad_kalg import SU3ADKAlg, _monomial_to_label, _T, _D
    S3A = SU3ADKAlg()
    B1 = A1DevenKAlg(1)
    img = {("T", i): {(i % 4, 4): 1, ((i + 1) % 4, 2): 1} for i in range(4)}
    img.update({("D", i): {((i + 1) % 4, 3): 1} for i in range(4)})
    bad_geo, n_geo = [], 0
    for i in range(4):
        for j in (i, (i - 1) % 4):                      # the tiles T_i D_i and T_i D_(i-1)
            for a in range(3):
                for b in range(3):
                    if a == b == 0:
                        continue
                    for w in ((0, 0), (1, 0)):
                        letters = {}
                        if a:
                            letters[_T(i)] = a
                        if b:
                            letters[_D(j)] = b
                        lab = _monomial_to_label(letters, w, 0)[0]
                        cnt = {}
                        for key, m in ((("T", i), a), (("D", j), b)):
                            for c, mult in img[key].items():
                                if m:
                                    cnt[c] = cnt.get(c, 0) + m * mult
                        want = (tuple(sorted(cnt.items())), w)
                        n_geo += 1
                        got = S3A.geometric_label(lab)
                        balanced = sum(m * _dv_charge(c, 1) for c, m in want[0]) == 0
                        try:
                            valid = C1.canonicalise((want[0], 0, 0)) == (want[0], 0, 0) and B1.in_centralizer((want[0], 0, 0))
                        except ValueError:
                            valid = False
                        if got != want or not balanced or not valid:
                            bad_geo.append((i, j, a, b, w, got))
    ok_gen = all(S3A.geometric_label(S3A.T(i)) == (tuple(sorted(img[("T", i)].items())), (0, 0)) and
                 S3A.geometric_label(S3A.D(i)) == (tuple(sorted(img[("D", i)].items())), (0, 0)) for i in range(4))
    checks.append(check(f"k=1: SU3ADKAlg's geometric labels are the printed curves of A1DevenKAlg(1) with the SU(3) weight: T_i the loop at i with the curve (i+1,2), D_i the curve (i+1,3), a tile monomial the union of its letters' curves ({n_geo} labels: 8 tiles, powers up to 2, two weights), each a balanced label of A1DevenKAlg(1)",
                        ok_gen and not bad_geo, str(bad_geo[:3])))
    # negative control: E on the boundary edge {2,3} instead of {1,2} is caught by the flow
    own = FR._E_EDGE
    caught = {}
    try:
        FR._E_EDGE = 2
        for k in (1, 2):
            Cb = U1A1DevenConeKAlgebra(k)
            iso = u1a1deven_cone_dodd_section_iso(k)
            cs = _dv_curves(k)
            caught[k] = sum(1 for g in cs for h in cs if _dv_cross(g, h, k)
                            and not iso.verify_multiplicative([(Element.basis(Cb.curve(*g)), Element.basis(Cb.curve(*h)))], []))
    finally:
        FR._E_EDGE = own
    checks.append(check("negative control: with E on the boundary edge {2,3} instead of {1,2} the products of crossing curves disagree with the flow", all(v > 0 for v in caught.values()),
                        f"disagreeing crossing pairs {caught}"))
    return {"checks": checks, "population": {"k": pop, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the printed k = 1 products; the flow's products on every generator pair",
                         "negative": "rho with the drift at marked point 3; E on the edge {2,3}"},
            "notes": "Transcribed independently: the crossing test on the doubled polygon, the parity rule, the collapse of the edge {1,2} and the A1DnKAlg(2k+1) cocycle, rho with its drift, the printed products.  The flow U1A1DevenViaDoddRG(k) is an independent presentation (its own RG solver); the bijection of labels is the closed-form lowest term of its image.",
            "inputs": []}


def check_a1deven_is_kalgebra(environment):
    from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra
    from a1deven_kalg import A1DevenKAlg
    checks = []
    t0 = time.time()
    pop = {}
    for k in _DV_KS:
        C = U1A1DevenConeKAlgebra(k)
        n = 2 * k + 2
        odd = [c for c in _dv_curves(k) if c[1] % 2]
        W = [C.identity(), ((), 1, 0), ((), -1, 0), C.curve(*odd[0]), C.curve(*odd[1]), C.curve(odd[0][0], odd[0][1], 1, 1),
             C.curve(0, 2), C.curve(1, 2), C.curve(0, n)]
        K = 3
        ver = run_k_verifiers(C, W, K)
        nf = sum(f for f, _x in ver.values())
        checks.append(check(f"k={k}: U1A1DevenConeKAlgebra: K1-K5 verifiers on all pairs of a window of {len(W)} labels (identity, E^(+-1), odd curves, an SU(2)-weighted E-dressed curve, charged curves and a loop) at K={K}",
                            nf == 0, "; ".join(f"{m} fails {f} (first {x})" for m, (f, x) in ver.items() if f)))
        B = A1DevenKAlg(k)
        gens = B.mult_generators()
        WB = [B.identity()] + gens[:5] + [((), 0, 1), ((), 1, 0)]
        ver = run_k_verifiers(B, WB, K)
        nf = sum(f for f, _x in ver.values())
        checks.append(check(f"k={k}: A1DevenKAlg: K1-K5 verifiers on all pairs of a window of {len(WB)} labels (identity, five generators, chi_1, z^-1) at K={K}",
                            nf == 0, "; ".join(f"{m} fails {f} (first {x})" for m, (f, x) in ver.items() if f)))
        step = 1 if k < 3 else 4
        labs = [B.identity()] + gens[::step]
        bad = [(a, b) for a in labs for b in labs if not B.verify_orthonormality(a, b, K=K)]
        what = f"the {len(gens)} generators" if step == 1 else f"every {step}th of the {len(gens)} generators"
        checks.append(check(f"k={k}: A1DevenKAlg: I_(a,b) = delta + O(q) on every ordered pair of the identity and {what} ({len(labs) ** 2} pairs) at K={K}", not bad, str(bad[:3])))
        pop[k] = {"gauged window": len(W), "ungauged window": len(WB), "orthonormality pairs": len(labs) ** 2}
    return {"checks": checks, "population": {"k": pop, "K": 3, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the identity rows", "negative": "n.a."},
            "notes": "Emergent: the products are the skein rule, the traces closed forms measured against an independent route; nothing in either is built to satisfy the axioms.",
            "inputs": []}


def check_a1deven_traces(environment):
    from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra
    from a1deven_kalg import A1DevenKAlg
    checks = []
    t0 = time.time()
    pop = {}
    # positive control first: the printed gauge tower (Creutzig's formula) against the class, k = 1, 2, 3
    tower_bad = {}
    for k, K in ((1, 24), (2, 20), (3, 16)):
        C = U1A1DevenConeKAlgebra(k)
        tower_bad[k] = [nn for nn in range(-3, 4) if _su2_to_qm(C.trace(((), nn, 0), K=K), K) != _dv_seed_series(k, (), nn, K)]
    ok_tower = not any(tower_bad.values())
    checks.append(check("positive control: the printed gauge tower Tr(E^n), n = -3..3, equals the class's trace through q^24 / q^20 / q^16 at k = 1 / 2 / 3",
                        ok_tower, str(tower_bad)))
    if not ok_tower:
        return {"checks": checks, "population": {}, "controls": {}, "notes": "the positive control failed; aborted", "inputs": []}
    # the printed seed rules, on every seed (every A1DevenKAlg generator), E-powers -2..2
    Ks = {1: 16, 2: 12, 3: 10}
    for k in _DV_KS:
        C = U1A1DevenConeKAlgebra(k)
        gens = A1DevenKAlg(k).mult_generators()
        K = Ks[k]
        bad, kinds = [], {}
        for F, _e, _kap in gens:
            conf = _dv_configuration(_dv_representative(F, 0, k)[0], k)
            kinds[conf[0]] = kinds.get(conf[0], 0) + 1
            for nn in range(-2, 3):
                if _su2_to_qm(C.trace((F, nn, 0), K=K), K) != _dv_seed_series(k, F, nn, K):
                    bad.append((F, nn))
        checks.append(check(f"k={k}: every one of the {len(gens)} seeds ({kinds}), at E^n, n = -2..2, equals the printed rule at its representative through q^{K}",
                            not bad, str(bad[:3])))
        pop[k] = {"seeds": len(gens), "kinds": kinds, "K": K}
    # the independent witness: the transport route (seed_closed_forms=False) on every seed, n = -1, 0, 1
    tw = {}
    for k, K in ((1, 10), (2, 6), (3, 4)):
        C = U1A1DevenConeKAlgebra(k)
        W = U1A1DevenConeKAlgebra(k, seed_closed_forms=False)
        gens = A1DevenKAlg(k).mult_generators()
        tb = time.time()
        bad = [(F, nn) for F, _e, _k in gens for nn in (-1, 0, 1) if C.trace((F, nn, 0), K=K) != W.trace((F, nn, 0), K=K)]
        tw[k] = round(time.time() - tb, 1)
        checks.append(check(f"k={k}: witness: the transport route (seed_closed_forms=False) equals the served closed forms on all {3 * len(gens)} seed labels (n = -1, 0, 1) through q^{K} ({tw[k]} s)",
                            not bad, str(bad[:3])))
    # every other label by the Layer-1 reduction: composites against the transport route; the leaves
    comp_pop = {}
    for k, K, cnt in ((1, 8, 20), (2, 6, 15), (3, 4, 10)):
        C = U1A1DevenConeKAlgebra(k)
        W = U1A1DevenConeKAlgebra(k, seed_closed_forms=False)
        labs = _dv_composites(k, cnt, seed=100 + k)
        bad = [l for l in labs if C.trace(l, K=K) != W.trace(l, K=K)]
        pairs = [(F, nn, 0) for F, _e, _k in A1DevenKAlg(k).mult_generators() if len(F) == 2 for nn in (-1, 0, 1)]
        leaves = set()
        for l in labs + pairs:
            for leaf in C._layer1_reduction(l[:2]).terms:
                leaves.add(leaf)
        odd_leaves = all((not F) or (len(F) == 1 and F[0][1] == 1 and F[0][0][1] % 2 == 1) for F, _e in leaves)
        checks.append(check(f"k={k}: {len(labs)} composite labels (one to three non-crossing curves with powers, E^-2..E^2, charge 0, not seeds) traced by the Layer-1 reduction equal the transport route through q^{K}; every leaf of their reductions and of the {len(pairs)} pair-seed labels' is a power of E or a single odd curve times one ({len(leaves)} distinct leaves)",
                            not bad and odd_leaves and C._layer1_stats["transport_leaves"] == 0,
                            f"differ {bad[:2]}; leaves {sorted(leaves)[:4]}..."))
        comp_pop[k] = {"labels": len(labs), "K": K, "leaves": len(leaves)}
    # the pair seeds by the Layer-1 reduction onto the single curves, against their own closed forms
    for k, K in ((1, 12), (2, 12), (3, 10)):
        C = U1A1DevenConeKAlgebra(k)
        pairs = [F for F, _e, _k in A1DevenKAlg(k).mult_generators() if len(F) == 2]
        bad = [(F, nn) for F in pairs for nn in (-1, 0, 1)
               if C._layer1_trace((F, nn), K) != C._seed_trace((F, nn), K)]
        checks.append(check(f"k={k}: each of the {len(pairs)} pair seeds (n = -1, 0, 1), reduced by Layer 1 onto the single curves and E^n, equals its own closed form through q^{K}",
                            not bad, str(bad[:3])))
    # negative controls, each on the class's served traces
    neg = {}
    for k, K in ((1, 12), (2, 12), (3, 10)):
        C = U1A1DevenConeKAlgebra(k)
        gens = A1DevenKAlg(k).mult_generators()
        for variant in ("nested-rule", "corner", "swap"):
            cnt = tot = 0
            for F, _e, _kap in gens:
                conf = _dv_configuration(_dv_representative(F, 0, k)[0], k)
                if (variant in ("nested-rule", "corner")) != (conf[0] == "curve") or (variant == "swap" and conf[0] != "nested"):
                    continue
                for nn in (-1, 0, 1):
                    tot += 1
                    cnt += _su2_to_qm(C.trace((F, nn, 0), K=K), K) != _dv_seed_series(k, F, nn, K, variant=variant)
            neg[(k, variant)] = (cnt, tot)
        cnt = sum(1 for F, _e, _kap in gens for nn in (-1, 0, 1)
                  if _su2_to_qm(C.trace((F, nn, 0), K=K), K) != _dv_seed_series(k, F, nn, K, drift_point=None))
        neg[(k, "no drift")] = (cnt, 3 * len(gens))
    C4 = U1A1DevenConeKAlgebra(4)
    side4 = []
    for a in _dv_curves(4):
        for b in _dv_curves(4):
            if _dv_charge(a, 4) == 1 and _dv_charge(b, 4) == -1 and not _dv_cross(a, b, 4):
                F = tuple(sorted(((a, 1), (b, 1))))
                if _dv_configuration(_dv_representative(F, 0, 4)[0], 4)[0] == "side":
                    side4.append(F)
    ok4 = all(_su2_to_qm(C4.trace((F, nn, 0), K=10), 10) == _dv_seed_series(4, F, nn, 10) for F in side4 for nn in (-1, 0, 1, 2))
    old4 = sum(1 for F in side4 for nn in (-1, 0, 1, 2) if _su2_to_qm(C4.trace((F, nn, 0), K=10), 10) != _dv_seed_series(4, F, nn, 10, variant="old-side"))
    neg[(4, "old-side")] = (old4, 4 * len(side4))
    ok_neg = all(neg[(k, v)][0] > 0 for k in _DV_KS for v in ("nested-rule", "corner", "no drift")) and \
        all(neg[(k, "swap")][0] > 0 for k in (2, 3)) and old4 > 0
    checks.append(check("negative controls, each disagreeing with the served traces: the nested rule in place of the curve rule, the curve's support moved to e >= |n| + 2, the nested gaps swapped (k = 2, 3), rho without its E-drift in the representative, and at k = 4 the retired side-by-side power q^(n g/2)",
                        ok_neg, str({f"{k}/{v}": c for (k, v), c in neg.items()})))
    checks.append(check(f"k=4 (held out from the rows above): the printed side-by-side rule on all {len(side4)} side-by-side seeds, n = -1..2, through q^10", ok4 and len(side4) > 0))
    return {"checks": checks, "population": {"seeds": pop, "composites": comp_pop, "witness seconds": tw, "k=4 side seeds": len(side4), "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the printed gauge tower (Creutzig's formula) against the class",
                         "negative": "five perturbed rules, each disagreeing with the served traces"},
            "notes": "The tower and the seed rules transcribed from the companion's text, independently of u1a1deven_seed_characters; the witness is the transport route, which shares nothing with the closed forms but the algebra (its stopping rule is a measured hypothesis, guarded).",
            "inputs": []}


def check_a1deven_ungauging(environment):
    from a1deven_kalg import A1DevenKAlg
    from a1deven_rgkalgebra import A1DevenRGKAlgebra
    import finite_kalgebras as fk
    import a1deven_seeds as S
    from kalgebra_iso import KAlgebraIso
    checks = []
    t0 = time.time()
    pop = {}
    for k, K in ((1, 10), (2, 8), (3, 6)):
        B = A1DevenKAlg(k)
        labs = [B.identity()] + B.mult_generators()
        bad = [l for l in labs if _dv_su2u1_to_qm(B.trace(l, K=K), K) != _dv_ungauged(k, l[0], K)]
        checks.append(check(f"k={k}: the printed ungauged trace, the sum over n of z^n Tr(L_(F,n,0)) with the measure (q^2;q^2)^-2 restored and every gauged trace from the printed rules, equals A1DevenKAlg({k})'s on the identity and all {len(labs) - 1} generators through q^{K}",
                            not bad, str(bad[:3])))
        g = labs[1]
        base = _dv_su2u1_to_qm(B.trace(g, K=K), K)
        lift = {q: {(m, f - 1): c for (m, f), c in d.items()} for q, d in base.items()}
        wrong = {q: {(m, f + 1): c for (m, f), c in d.items()} for q, d in base.items()}
        got = _dv_su2u1_to_qm(B.trace((g[0], 1, 1), K=K), K)
        chi1 = {q: {} for q in lift}
        for q, d in lift.items():
            for (m, f), c in d.items():
                for s in (1, -1):
                    chi1[q][(m + s, f)] = chi1[q].get((m + s, f), 0) + c
        chi1 = {q: {kk: c for kk, c in d.items() if c} for q, d in chi1.items() if any(d.values())}
        chi1w = {q: {} for q in wrong}
        for q, d in wrong.items():
            for (m, f), c in d.items():
                for s in (1, -1):
                    chi1w[q][(m + s, f)] = chi1w[q].get((m + s, f), 0) + c
        chi1w = {q: {kk: c for kk, c in d.items() if c} for q, d in chi1w.items() if any(d.values())}
        checks.append(check(f"k={k}: the lift L_(F,e,kappa) = z^-e chi_kappa L_(F,0,0) on the trace of a generator at e = 1, kappa = 1; negative control: z^+e fails",
                            got == chi1 and got != chi1w))
        # a negative control on the formula: the measure restored once instead of twice
        once = _dv_ungauged(k, (), K, measure_power=1)
        checks.append(check(f"k={k}: negative control: with (q^2;q^2)^-1 in place of (q^2;q^2)^-2 the identity's trace disagrees", once != _dv_su2u1_to_qm(B.trace(B.identity(), K=K), K)))
        pop[k] = {"labels": len(labs), "K": K}
    # the printed heads of Tr 1
    heads = {1: {0: {(0, 0): 1}, 2: {(0, 0): 1, (2, 0): 1, (1, 1): 1, (1, -1): 1},
                 4: {(0, 0): 3, (2, 0): 2, (4, 0): 1, (1, 1): 2, (1, -1): 2, (3, 1): 1, (3, -1): 1, (2, 2): 1, (2, -2): 1}},
             2: {0: {(0, 0): 1}, 2: {(0, 0): 1, (2, 0): 1}, 3: {(1, 1): -1, (1, -1): -1}, 4: {(0, 0): 3, (2, 0): 2, (4, 0): 1}},
             3: {0: {(0, 0): 1}, 2: {(0, 0): 1, (2, 0): 1}, 4: {(0, 0): 3, (2, 0): 2, (4, 0): 1, (1, 1): 1, (1, -1): 1}}}
    for k in _DV_KS:
        B = A1DevenKAlg(k)
        tr = B.trace(B.identity(), K=4)
        rows = {e: {(kap, f): c for (kap, (f,)), c in r.terms.items() if c} for e, r in tr.coeffs.items() if e <= 4}
        rows = {e: r for e, r in rows.items() if r}
        checks.append(check(f"k={k}: the printed head of the ungauged Tr 1 through q^4", rows == heads[k], str(rows)))
    # independent witness: the flow A1DevenRGKAlgebra(k) (the two fork terminals dropped onto A1A2kKAlg(k))
    for k in _DV_KS:
        R = A1DevenRGKAlgebra(k)
        B = A1DevenKAlg(k)
        tr = R.trace(R.identity(), K=6)
        rf = {e: {(kap, f): c for (kap, f), c in r.terms.items() if c} for e, r in tr.coeffs.items() if e <= 6}
        tb = B.trace(B.identity(), K=6)
        rb = {e: {(kap, f): c for (kap, (f,)), c in r.terms.items() if c} for e, r in tb.coeffs.items() if e <= 6}
        checks.append(check(f"k={k}: independent witness: Tr 1 equals the flow A1DevenRGKAlgebra({k})'s through q^6",
                            {e: r for e, r in rf.items() if r} == {e: r for e, r in rb.items() if r}))
    # the zoo entries a1d4, a1d6: the certified KAlgebraIso through the Z-form wrapper
    one = LaurentPoly.one()
    for sid in ("a1d4", "a1d6", "a1d8"):
        tz = time.time()
        Z = fk.FINITE_KALGEBRAS[sid]()
        iso = S.kalgebra_iso(sid, native=Z)
        ngen = len(S.generator_map(sid)["sigma"])
        src = ([iso.source.identity()] + [(0, (((g, 1),), 0)) for g in range(ngen)]
               + [(1, ((), 0)), (0, ((), 1)), (0, ((), -1))])
        tgt = ([iso.target.identity()] + list(iso.target.mult_generators())
               + [((), 0, 1), ((), -1, 0), ((), 1, 0)])
        se = [Element({l: one}) for l in src]
        te = [Element({l: one}) for l in tgt]
        step = 4 if sid == "a1d8" else 1
        ps, pt = se[1:-3][::step] + se[-3:], te[1:-3][::step] + te[-3:]
        res = {"unit": iso.verify_unit(), "round_trip": iso.verify_round_trip(se, te),
               "multiplicative": iso.verify_multiplicative([(a, b) for a in ps for b in ps], [(a, b) for a in pt for b in pt]),
               "rho_equivariant": iso.verify_rho_equivariant(se, te),
               "trace_equivariant": iso.verify_trace_equivariant(se, te, K=8)}
        what = "all" if step == 1 else f"every {step}th generator's and the flavour characters'"
        checks.append(check(f"{sid}: the KAlgebraIso FiniteSU2U1ZKAlgebra -> A1DevenKAlg({S.K_OF[sid]}) (a1deven_seeds.kalgebra_iso) passes unit, round trip, rho and traces through q^8 on the identity, the {ngen} generators and the flavour characters, and multiplicativity on {what} {len(ps) ** 2} ordered pairs each way ({round(time.time() - tz, 1)} s)",
                            all(res.values()), str(res)))
        flip = (lambda l: (l[0], (l[1][0], -l[1][1])))   # noqa: E731
        bad_iso = KAlgebraIso(iso.source, iso.target,
                              lambda l: iso.map(Element({flip(l): one})),
                              lambda l: Element({flip(x): c for x, c in iso.inverse(Element({l: one})).terms.items()}),
                              name="flipped")
        if sid != "a1d8":
            caught = not (bad_iso.verify_multiplicative([(a, b) for a in ps for b in ps], []) and bad_iso.verify_rho_equivariant(se, []) and
                          bad_iso.verify_trace_equivariant(se, [], K=4))
            checks.append(check(f"{sid}: negative control: the map precomposed with the zoo's mu -> mu^-1 is caught", caught))
        pop[sid] = {"samples": len(src), "product pairs each way": len(ps) ** 2}
    return {"checks": checks, "population": {"k": pop, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the flow A1DevenRGKAlgebra(k)'s Tr 1; the zoo's certified KAlgebraIso",
                         "negative": "the opposite lift sign; the measure restored once; the zoo map with mu -> mu^-1"},
            "notes": "The ungauging sum transcribed from the text over the transcribed gauged rules.  At a1d4 the zoo is served by SU3ADKAlg, so the iso's trace check compares two independent routes; at a1d6 the zoo's seeds are served through this map, so its trace check is not independent there (products and rho are: the zoo's exported cone table).",
            "inputs": []}


# ============================================================================
# comp:a1aodd — the ungauged odd polygons, K_q([A_1, A_{2k+1}]) with its U(1)
# flavour: ungauge_u1a1aodd(k), the centralizer of E in U1A1AoddKAlg(k), E the
# fugacity (L_(F,e) = z^-e L_(F,0)).  Transcribed here from the companion; the
# gauged example's printed rules are the transcriptions of the comp:u1a1aodd
# section above (_chord, _crossing, _drift, _charge, _pairing, _singlet,
# _chord_rule).
# ============================================================================
def _aodd_mag(t, i, k):
    """The printed magnetic charge eq:comp-aodd-mag: -2 both endpoints even, +2 both odd, 0 mixed."""
    H = 2 * k + 4
    u, v = i % H, (i + t + 1) % H
    if (u - v) % 2:
        return 0
    return -2 if u % 2 == 0 else 2


def _aodd_letters(k):
    """The chord letters (t, i): t <= k with i in Z/H, the diameter t = k+1 with i in Z/(H/2)."""
    H = 2 * k + 4
    return [(t, i) for t in range(1, k + 1) for i in range(H)] + [(k + 1, i) for i in range(H // 2)]


def _aodd_size(t, k):
    return (2 * k + 4) // 2 if t == k + 1 else 2 * k + 4


def _aodd_generators(k):
    """The printed generators: the balanced chords and the non-crossing (even-even, odd-odd) pairs, as sections F."""
    H = 2 * k + 4
    L = _aodd_letters(k)
    mixed = [((t, i, 1),) for (t, i) in L if _aodd_mag(t, i, k) == 0]
    ee = [g for g in L if _aodd_mag(*g, k) < 0]
    oo = [g for g in L if _aodd_mag(*g, k) > 0]
    pairs = [tuple(sorted(((a[0], a[1], 1), (b[0], b[1], 1)))) for a in ee for b in oo
             if not _crossing(_chord(a[0], a[1], H), _chord(b[0], b[1], H))]
    return mixed, pairs


def _aodd_rho(label, k, drift=True):
    """The printed rho eq:comp-aodd-rho on a label: (F, e) -> (F', d(F) - e)."""
    F, e = label
    Fr, d = {}, 0
    for (t, i, m) in F:
        j = (i + 1) % _aodd_size(t, k)
        Fr[(t, j)] = Fr.get((t, j), 0) + m
        d += m * _drift(t, i, k) if drift else 0
    return (tuple(sorted((t, j, m) for (t, j), m in Fr.items())), d - e)


def _aodd_zd(t, K):
    """An R(U(1))-valued trace -> {(q, z-power): int} through q^K."""
    out = {}
    for q, r in t.coeffs.items():
        if q > K:
            continue
        for key, v in r.terms.items():
            if v:
                out[(q, key[0])] = out.get((q, key[0]), 0) + int(v)
    return {x: v for x, v in out.items() if v}


def _aodd_gd(t, K):
    """A Z-valued (gauged) trace -> {q: int} through q^K."""
    out = {}
    for q, r in t.coeffs.items():
        v = sum(r.terms.values()) if hasattr(r, "terms") else r
        if v and q <= K:
            out[q] = int(v)
    return out


def _aodd_inv_measure(K, power=2):
    """1 / (q^2; q^2)_inf^power as {exp: coeff} through q^K, by expanding each 1/(1 - q^{2j})."""
    s = {0: 1}
    for _ in range(power):
        for j in range(1, K // 2 + 1):
            out = {}
            for e, c in s.items():
                m = 0
                while e + 2 * j * m <= K:
                    out[e + 2 * j * m] = out.get(e + 2 * j * m, 0) + c
                    m += 1
            s = out
    return s


def _aodd_sum(series_of_n, K, W, zsign=1, zshift=0, power=2):
    """(q^2;q^2)^-power * sum_{|n| <= W} z^(zsign*n + zshift) series_of_n(n), as {(q, z): int} through q^K."""
    meas = _aodd_inv_measure(K, power)
    out = {}
    for n in range(-W, W + 1):
        for q, c in series_of_n(n).items():
            for fe, fc in meas.items():
                if q + fe <= K:
                    key = (q + fe, zsign * n + zshift)
                    out[key] = out.get(key, 0) + c * fc
    return {x: v for x, v in out.items() if v}


def _aodd_gauge_charge(F, k):
    """g of L_(F,0): the e_1-component of its charge, from the printed charge formula."""
    return sum(m * _charge(t, i, k)[0] for (t, i, m) in F)


class _AoddHiddenGaugeCharge:
    """A gauged class with `_label_gauge_charge` hidden: the ungauger then windows by the E-power alone."""

    def __init__(self, G):
        self._inner = G

    def __getattr__(self, name):
        if name == "_label_gauge_charge":
            raise AttributeError(name)
        return getattr(self._inner, name)


def check_a1aodd_implementation(environment):
    from ungauge_kalgebra import ungauge_u1a1aodd
    from math import comb, gcd
    import importlib
    import random
    import finite_kalgebras as fk
    from hexagon_kalg import HexagonKAlg
    checks = []
    t0 = time.time()
    census = {}
    zoo = {1: "a3", 2: "a5", 3: "a7"}
    for k in (1, 2, 3):
        U = ungauge_u1a1aodd(k)
        G = U._G
        H = 2 * k + 4
        n = 2 * k + 2
        mu = tuple(1 if j % 2 == 0 else 0 for j in range(n))
        E = ((), 1)
        # eq:comp-aodd-mag: E L = q^mag L E on every letter, mag by endpoint parity = 2<mu, gamma> (printed charges)
        bad_mag = []
        for (t, i) in _aodd_letters(k):
            L = (((t, i, 1),), 0)
            (l1, c1), = G.multiply(E, L).terms.items()
            (l2, c2), = G.multiply(L, E).terms.items()
            (e1,), (e2,) = _lpc(c1).keys(), _lpc(c2).keys()
            m = _aodd_mag(t, i, k)
            if l1 != l2 or e1 - e2 != m or 2 * _pairing(mu, _charge(t, i, k)) != m:
                bad_mag.append(((t, i), e1 - e2, m))
        checks.append(check(f"k={k}: E L_t;i = q^mag L_t;i E on all {len(_aodd_letters(k))} letters, mag = -2/+2/0 by endpoint parity = 2<mu, gamma_t;i> of the printed charges (eq:comp-aodd-mag)",
                            not bad_mag, str(bad_mag[:3])))
        # the canonical basis = the balanced labels: parity rule == the multiply-based centralizer test == the class's
        r = 3 if k < 3 else 2
        win, nbal = 0, 0
        bad_bal = []
        letters = _aodd_letters(k)
        for s in range(r + 1):
            for S in itertools.combinations(letters, s):
                if any(_crossing(_chord(*a, H), _chord(*b, H)) for a, b in itertools.combinations(S, 2)):
                    continue
                for ms in itertools.product((1, 2), repeat=s):
                    F = tuple(sorted((g[0], g[1], m) for g, m in zip(S, ms)))
                    for e in (-1, 0, 2):
                        x = (F, e)
                        win += 1
                        bal = sum(m * _aodd_mag(t, i, k) for (t, i, m) in F) == 0
                        by_mult = G.multiply(E, x).terms == G.multiply(x, E).terms
                        nbal += bal
                        if not (bal == by_mult == U.in_centralizer(x)):
                            bad_bal.append(x)
        checks.append(check(f"k={k}: a canonical label (F, e) commutes with E iff F is balanced (as many even-even as odd-odd diagonals): the printed rule, the multiply-based test and the class agree on {win} labels ({nbal} balanced; up to {r} letters, exponents 1, 2, e in {{-1, 0, 2}})",
                            not bad_bal and 0 < nbal < win, str(bad_bal[:3])))
        # the generators: the balanced chords and the non-crossing (ee, oo) pairs, k(k+2) + (k+2) C(k+2, 3)
        mixed, pairs = _aodd_generators(k)
        gens = U.mult_generators()
        ours = {(F, 0) for F in mixed + pairs}
        count = k * (k + 2) + (k + 2) * comb(k + 2, 3)
        geo_ok = all(U.geometric_label((F, 0)) == (tuple(sorted((_chord(t, i, H), m) for (t, i, m) in F)), 0) for F in mixed + pairs)
        checks.append(check(f"k={k}: mult_generators = the {len(mixed)} balanced chords and the {len(pairs)} non-crossing (even-even, odd-odd) pairs, {count} = k(k+2) + (k+2) C(k+2,3) in all; geometric labels are the multisets of diagonals",
                            set(gens) == ours and len(gens) == count == len(ours) and len(mixed) == k * (k + 2) and geo_ok, f"{len(gens)} gens, {len(ours)} transcribed, geo {geo_ok}"))
        # the relations: the gauged product restricted, every term balanced
        rng = random.Random(4200 + k)
        prs = [(a, b) for a in gens for b in gens]
        if k == 3:
            prs = rng.sample(prs, 300)
        bad_prod = []
        for a, b in prs:
            pg = G.multiply(a, b)
            pu = U.multiply(a, b)
            if pu != pg or any(sum(m * _aodd_mag(t, i, k) for (t, i, m) in L[0]) != 0 for L in pg.terms):
                bad_prod.append((a, b))
        checks.append(check(f"k={k}: the product is the gauged product (U1A1AoddKAlg) and every term of it is balanced, on {len(prs)} ordered generator pairs",
                            not bad_prod, str(bad_prod[:2])))
        # eq:comp-aodd-lift: L_(F,e) = E^e L_(F,0) = z^-e L_(F,0); the lift and the trace obey it, the opposite sign does not
        R = U.coefficient_ring()
        bad_lift, neg_hit = [], 0
        Fs = [()] + [g[0] for g in gens[:4]] + [g[0] for g in gens[-2:]]
        for F in Fs:
            base = U.trace((F, 0), 6)
            for e in (-2, 1, 3):
                sec, key = U.r_label_decompose((F, e))
                lab_ok = sec == (F, 0) and key == (-e,) and U.multiply(((), e), (F, 0)) == Element({(F, e): _lp(0)})
                tr = U.trace((F, e), 6)
                if not lab_ok or tr != base * R.basis_element((-e,)):
                    bad_lift.append((F, e))
                if not base.is_zero() and tr != base * R.basis_element((e,)):
                    neg_hit += 1
        checks.append(check(f"k={k}: L_(F,e) = E^e L_(F,0) = z^-e L_(F,0) (eq:comp-aodd-lift): r_label_decompose, the product with E^e, and Tr L_(F,e) = z^-e Tr L_(F,0) through q^6 on {3 * len(Fs)} labels",
                            not bad_lift, str(bad_lift[:3])))
        # eq:comp-aodd-rho on the generators and the terms of their products, with E-shifts; rho^-1 inverts it
        labs = set(gens)
        for a, b in prs[:60]:
            labs.update(U.multiply(a, b).terms)
        labs = sorted({(F, e + s) for (F, e) in labs for s in (-1, 0, 2)})
        bad_rho = [x for x in labs if U.rho(x) != _aodd_rho(x, k) or U.rho_inverse(U.rho(x)) != x]
        nodrift_bad = sum(1 for x in labs if U.rho(x) != _aodd_rho(x, k, drift=False))
        checks.append(check(f"k={k}: rho(L_(F,e)) = L_(F', d(F)-e) (eq:comp-aodd-rho, the drift eq:comp-drift) and rho^-1 inverts it, on {len(labs)} labels",
                            not bad_rho, str(bad_rho[:2])))
        # the order of rho: H exactly on every generator; modulo z (the E-power) 3 / 8 / 10, the zoo's rho-permutation order
        orders, orders_mod = [], []
        for g in gens:
            cur, o = U.rho(g), 1
            while cur != g and o < 4 * H:
                cur, o = U.rho(cur), o + 1
            orders.append(o)
            cur, o = U.rho(g), 1
            while cur[0] != g[0] and o < 4 * H:
                cur, o = U.rho(cur), o + 1
            orders_mod.append(o)
        lcm = 1
        for o in orders_mod:
            lcm = lcm * o // gcd(lcm, o)
        mod = importlib.import_module(fk.FINITE_KALGEBRAS[zoo[k]].__module__)
        perm = getattr(mod, zoo[k].upper() + "_RHO_PERM")
        zl = 1
        for g0 in range(len(perm)):
            o, cur = 1, perm[g0]
            while cur != g0:
                cur, o = perm[cur], o + 1
            zl = zl * o // gcd(zl, o)
        want = {1: 3, 2: 8, 3: 10}[k]
        checks.append(check(f"k={k}: rho^H = id on every generator (H = {H}); modulo powers of z the order of rho is {want}, the order of the zoo's {zoo[k]} rho-permutation",
                            set(orders) == {H} and lcm == want == zl, f"orders {sorted(set(orders))}, mod z {lcm}, zoo {zl}"))
        census[k] = dict(letters=len(_aodd_letters(k)), window=win, balanced=nbal, generators=len(gens), product_pairs=len(prs),
                         rho_labels=len(labs), rho_without_drift_differs=nodrift_bad, lift_opposite_sign_fails=neg_hit)
    # the non-simplicial case at k = 3: two words, one label
    U = ungauge_u1a1aodd(3)
    A1, A2, B1, B2 = (1, 0), (1, 2), (1, 5), (1, 7)
    pr = lambda a, b: (tuple(sorted((a + (1,), b + (1,)))), 0)   # noqa: E731
    w1, w2 = U.multiply(pr(A1, B1), pr(A2, B2)), U.multiply(pr(A1, B2), pr(A2, B1))
    target = (tuple(sorted(g + (1,) for g in (A1, A2, B1, B2))), 0)
    checks.append(check("k=3: with A1 = {0,2}, A2 = {2,4}, B1 = {5,7}, B2 = {7,9}, the products L_{A1,B1} L_{A2,B2} and L_{A1,B2} L_{A2,B1} are single terms on the one label {A1, A2, B1, B2}",
                        len(w1.terms) == 1 and len(w2.terms) == 1 and set(w1.terms) == set(w2.terms) == {target}, f"{w1} / {w2}"))
    # positive control: at k = 1 the generators are HexagonKAlg's six hand-named ones
    named = {lbl[0] for lbl in HexagonKAlg().mult_generators()}
    mixed, pairs = _aodd_generators(1)
    checks.append(check("positive control: at k = 1 the transcribed generators are HexagonKAlg's six hand-named ones (3 diameters, 3 short-diagonal pairs)",
                        named == set(mixed + pairs) and len(named) == 6))
    # negative controls: an off-by-one parity rule, rho without the drift, the lift with the opposite sign
    off = {}
    for k in (1, 2, 3):
        U = ungauge_u1a1aodd(k)
        off[k] = sum(1 for (t, i) in _aodd_letters(k) if _aodd_mag(t + 1, i, k) != U.mag((((t, i, 1),), 0)))
    checks.append(check(f"negative control: the parity rule applied to the diagonal one vertex longer, {{i, i+t+2}}, disagrees with the E-commutator on {off} letters at k = 1, 2, 3",
                        all(v > 0 for v in off.values())))
    checks.append(check(f"negative control: rho without the drift differs from the class on {[census[k]['rho_without_drift_differs'] for k in (1, 2, 3)]} of the rho labels at k = 1, 2, 3",
                        all(census[k]["rho_without_drift_differs"] > 0 for k in (1, 2, 3))))
    checks.append(check(f"negative control: the lift with the opposite sign, L_(F,e) = z^e L_(F,0), fails the trace on {[census[k]['lift_opposite_sign_fails'] for k in (1, 2, 3)]} labels at k = 1, 2, 3",
                        all(census[k]["lift_opposite_sign_fails"] > 0 for k in (1, 2, 3))))
    return {"checks": checks, "population": {"k": [1, 2, 3], "census": census, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "at k = 1 the transcribed generators are HexagonKAlg's six hand-named ones",
                         "negative": "an off-by-one parity rule, rho without the drift, and the lift with the opposite sign are each caught"},
            "notes": "The magnetic charge, balance, generators, lift and rho transcribed from the companion (eq:comp-aodd-mag, eq:comp-aodd-lift, eq:comp-aodd-rho) and the gauged example's printed charges and drift; the products are the gauged class's analytic route.",
            "inputs": []}


# the printed k = 1 presentation (eq:comp-aodd-hexqc, eq:comp-aodd-hex, eq:comp-aodd-hexrho):
# symbols ('d', j) = L_{2;j}, ('p', j) = L_{1;j,1;j+3}, ('d2', j) = L_{2;j}^2, '1' = 1; terms (symbol, z-power) -> {q: c}
_AODD_HEX_REL = {
    (("d", 0), ("d", 1)): {("1", 0): {0: 1}, (("p", 1), 1): {-1: 1}},
    (("d", 1), ("p", 0)): {("1", 0): {0: 1}, ("1", 1): {0: 1}, (("d", 0), 0): {1: 1}, (("d", 2), 0): {-1: 1}},
    (("p", 0), ("p", 1)): {("1", 0): {0: 1}, (("d", 0), 0): {-1: 1}, (("d", 0), -1): {-1: 1}, (("d2", 0), -1): {-2: 1}},
}
_AODD_HEX_QC = ((0, 2), (1, -2))          # L_{2;j} L_{1;j+s,1;j+s+3} = q^c L_{1;j+s,...} L_{2;j}
_AODD_HEX_RHO_Z = {"d": -1, "d2": -2, "p": -2}   # rho(X_2) = z^(this) X_0 ; rho(X_j) = X_(j+1) for j = 0, 1


def _aodd_hex_label(sym, zp):
    e = -zp
    if sym == "1":
        return ((), e)
    kind, j = sym
    j %= 3
    if kind == "d":
        return (((2, j, 1),), e)
    if kind == "d2":
        return (((2, j, 2),), e)
    return (tuple(sorted(((1, j, 1), (1, j + 3, 1)))), e)


def _aodd_hex_el(terms):
    out = {}
    for (sym, zp), qd in terms.items():
        L = _aodd_hex_label(sym, zp)
        lp = LaurentPoly(dict(qd))
        out[L] = out[L] + lp if L in out else lp
    return Element({k_: v for k_, v in out.items() if not v.is_zero()})


def _aodd_hex_rho_sym(sym, zp, drift=True):
    if sym == "1":
        return ("1", -zp)
    kind, j = sym
    j %= 3
    extra = _AODD_HEX_RHO_Z[kind] if (drift and j == 2) else 0
    return ((kind, (j + 1) % 3), -zp + extra)


def _aodd_hex_derive(drift=True):
    """All crossing products of the six generators from the three printed ones, by the printed rho and bar."""
    def rho_rel(rel):
        out = {}
        for (a, b), rhs in rel.items():
            (a2, za), (b2, zb) = _aodd_hex_rho_sym(a, 0, drift), _aodd_hex_rho_sym(b, 0, drift)
            new = {}
            for (sym, zp), qd in rhs.items():
                s2, z2 = _aodd_hex_rho_sym(sym, zp, drift)
                d = new.setdefault((s2, z2 - za - zb), {})
                for q, c in qd.items():
                    d[q] = d.get(q, 0) + c
            out[(a2, b2)] = new
        return out
    rels = dict(_AODD_HEX_REL)
    r1 = rho_rel(_AODD_HEX_REL)
    r2 = rho_rel(r1)
    rels.update(r1)
    rels.update(r2)
    rels.update({(b, a): {k_: {-q: c for q, c in qd.items()} for k_, qd in rhs.items()} for (a, b), rhs in list(rels.items())})
    return rels


def _aodd_hex_products(U, drift=True, qc=_AODD_HEX_QC):
    """(agree, differ, n q-commuting, n derived crossing) over the 36 ordered generator products."""
    rels = _aodd_hex_derive(drift)
    gens = [("d", j) for j in range(3)] + [("p", j) for j in range(3)]
    ok = bad = nqc = 0
    for a in gens:
        for b in gens:
            got = U.multiply(_aodd_hex_label(a, 0), _aodd_hex_label(b, 0))
            if (a, b) in rels:
                good = got == _aodd_hex_el(rels[(a, b)])
            elif a == b:
                sq = ("d2", a[1]) if a[0] == "d" else None
                good = len(got.terms) == 1 and (sq is None or set(got.terms) == {_aodd_hex_label(sq, 0)})
            else:
                c = None
                for s, cc in qc:
                    if a[0] == "d" and b == ("p", (a[1] + s) % 3):
                        c = cc
                    if b[0] == "d" and a == ("p", (b[1] + s) % 3):
                        c = -cc
                if c is None:
                    good = False
                else:
                    nqc += 1
                    rev = U.multiply(_aodd_hex_label(b, 0), _aodd_hex_label(a, 0))
                    good = got == Element({L: v * _lp(c) for L, v in rev.terms.items()})
            ok += good
            bad += not good
    return ok, bad, nqc, len(rels)


def check_a1aodd_hexagon(environment):
    from ungauge_kalgebra import ungauge_u1a1aodd
    checks = []
    t0 = time.time()
    U = ungauge_u1a1aodd(1)
    # eq:comp-aodd-hexrho against the class
    rho_ok = True
    for kind in ("d", "p"):
        for j in range(3):
            s2, z2 = _aodd_hex_rho_sym((kind, j), 0)
            rho_ok &= U.rho(_aodd_hex_label((kind, j), 0)) == _aodd_hex_label(s2, z2)
    checks.append(check("the printed rho eq:comp-aodd-hexrho on the six generators: rho(L_2;j) = L_2;j+1, rho(L_{1;j,1;j+3}) = L_{1;j+1,1;j+4} (j = 0, 1), rho(L_2;2) = z^-1 L_2;0, rho(L_{1;2,1;5}) = z^-2 L_{1;0,1;3}",
                        rho_ok))
    ok, bad, nqc, nrel = _aodd_hex_products(U)
    checks.append(check(f"the printed q-commutations eq:comp-aodd-hexqc ({nqc} ordered products), the three printed relations eq:comp-aodd-hex with their rho images and bar images ({nrel} ordered crossing products) and the squares give all 36 ordered products of the six generators",
                        ok == 36 and bad == 0 and nqc == 12 and nrel == 18, f"agree {ok}, differ {bad}"))
    # the canonical basis at k = 1: z^n times the monomials of the six cones
    cd = U._G.cone_data()
    H = 6
    n_bal, not_cone = 0, []
    for s in range(0, 5):
        for S in itertools.combinations(_aodd_letters(1), s):
            if any(_crossing(_chord(*a, H), _chord(*b, H)) for a, b in itertools.combinations(S, 2)):
                continue
            for ms in itertools.product((1, 2, 3), repeat=s):
                F = tuple(sorted((g[0], g[1], m) for g, m in zip(S, ms)))
                if sum(m * _aodd_mag(t, i, 1) for (t, i, m) in F) != 0:
                    continue
                n_bal += 1
                d = {(t, i): m for (t, i, m) in F}
                ok_cone = any(set(d) <= {(2, j), (1, jj), (1, jj + 3)} and d.get((1, jj), 0) == d.get((1, jj + 3), 0)
                              for j in range(3) for jj in (j, (j + 1) % 3))
                if not ok_cone:
                    not_cone.append(F)
    checks.append(check(f"every balanced label at k = 1 ({n_bal}: up to four letters, exponents up to 3) is a monomial of one of the six cones {{L_2;j, L_{{1;j,1;j+3}}}}, {{L_2;j, L_{{1;j+1,1;j+4}}}}",
                        not not_cone and n_bal > 0, str(not_cone[:3])))
    # negative controls
    ok_nd, bad_nd, _q, _r = _aodd_hex_products(U, drift=False)
    ok_sw, bad_sw, _q2, _r2 = _aodd_hex_products(U, qc=((0, -2), (1, 2)))
    checks.append(check(f"negative control: rho without its powers of z (rho(L_2;2) = L_2;0, ...) breaks {bad_nd} of the 36 products", bad_nd > 0))
    checks.append(check(f"negative control: the two q-commutation powers exchanged breaks {bad_sw} of the 36 products", bad_sw > 0))
    return {"checks": checks, "population": {"k": 1, "products": 36, "balanced labels": n_bal, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the printed rho on the six generators, and all 36 products reproduced",
                         "negative": "rho without its powers of z breaks %d products; the q-commutation powers exchanged break %d" % (bad_nd, bad_sw)},
            "notes": "The k = 1 presentation transcribed from the companion (three crossing relations at j = 0, the q-commutations, rho with its fugacity powers, bar) and compared with ungauge_u1a1aodd(1).multiply on every ordered pair of the six generators.",
            "inputs": []}


def check_a1aodd_is_kalgebra(environment):
    from ungauge_kalgebra import UngaugedKAlgebra, ungauge_u1a1aodd
    from u1a1aodd_kalg import U1A1AoddKAlg
    import random
    checks = []
    t0 = time.time()
    K = 4
    pop = {}
    for k in (1, 2):
        U = ungauge_u1a1aodd(k)
        gens = U.mult_generators()
        if k == 2:
            gens = random.Random(22).sample(gens, 7)
        W = [U.identity(), ((), 1)] + list(gens) + [(gens[0][0], -1)]
        extra = []
        for a in gens:
            for b in gens:
                p = U.multiply(a, b)
                if len(p.terms) == 1 and a < b:
                    extra.append(next(iter(p.terms)))
        W += sorted(set(extra))[:2]
        ver = run_k_verifiers(U, W, K)
        n_fail = sum(f for f, _x in ver.values())
        closure = all(isinstance(c, LaurentPoly) for a in W for b in W for c in U.multiply(a, b).terms.values())
        checks.append(check(f"k={k}: K1-K5 verifiers on all pairs of a window of {len(W)} labels (identity, E, generators, a generator times E^-1, products) at K={K}, and closure over Z[q^±1]",
                            n_fail == 0 and closure, "; ".join(f"{nm} fails {f} (first {x})" for nm, (f, x) in ver.items() if f)))
        pop[k] = len(W)

    # negative control: the same algebra with rho stripped of its drift is not an automorphism
    class _NoDrift(UngaugedKAlgebra):
        def rho(self, a):
            return _aodd_rho(a, 1, drift=False)

        def rho_inverse(self, a):
            F, e = a
            Fr = tuple(sorted((t, (i - 1) % _aodd_size(t, 1), m) for (t, i, m) in F))
            return (Fr, -e)
    N = _NoDrift(U1A1AoddKAlg(1), ((), 1), epow=lambda lbl: lbl[1])
    U1 = ungauge_u1a1aodd(1)
    W1 = [U1.identity()] + U1.mult_generators()
    nd_fail = sum(1 for a in W1 for b in W1 if not N.verify_rho_is_automorphism(a, b))
    checks.append(check(f"negative control: at k = 1, rho without the drift fails verify_rho_is_automorphism on {nd_fail} of the {len(W1) ** 2} pairs of the identity and the generators",
                        nd_fail > 0))
    return {"checks": checks, "population": {"window sizes": pop, "K": K, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the identity row", "negative": "rho without the drift fails the automorphism check (%d pairs)" % nd_fail},
            "notes": "All five axioms emergent on the ungauged class: its products and traces are the gauged analytic ones restricted and summed.", "inputs": []}


def check_a1aodd_traces(environment):
    from ungauge_kalgebra import UngaugedKAlgebra, ungauge_u1a1aodd
    from a1d3_kalg import A1D3KAlg
    import random
    checks = []
    t0 = time.time()
    pop = {}
    # eq:comp-aodd-trace transcribed: z^-e (q^2;q^2)^-2 sum_n z^n Tr_gauged L_(F,n), over a wide window, against the class
    Kf = 10
    fails = {}
    for k in (1, 2, 3):
        U = ungauge_u1a1aodd(k)
        G = U._G
        gens = U.mult_generators()
        rng = random.Random(700 + k)
        labs = [U.identity(), ((), 2)] + list(gens) + [(g[0], s) for g in gens[:3] for s in (-2, 3)]
        prs = [(a, b) for a in gens for b in gens]
        for a, b in (prs if k == 1 else rng.sample(prs, 12)):
            labs += sorted(U.multiply(a, b).terms)[:2]
        labs = sorted(set(labs))
        bad = []
        for (F, e) in labs:
            g = _aodd_gauge_charge(F, k)
            W = 2 * Kf + 8 + abs(g) + abs(e)
            want = _aodd_sum(lambda n: _aodd_gd(G.trace((F, n), Kf), Kf), Kf, W, zshift=-e)
            if _aodd_zd(U.trace((F, e), Kf), Kf) != want:
                bad.append((F, e))
        fails[k] = bad
        pop[f"formula labels k={k}"] = len(labs)
    checks.append(check(f"eq:comp-aodd-trace transcribed (the gauged traces summed with z^n over a wide window, times an independent (q^2;q^2)^-2, times z^-e) equals the class's trace through q^{Kf} on {sum(pop[f'formula labels k={k}'] for k in (1, 2, 3))} labels at k = 1, 2, 3 (identity, E^2, every generator, E-shifted generators, terms of generator products)",
                        not any(fails.values()), str({k: v[:2] for k, v in fails.items() if v})))
    # the window: at k = 3 the powers a = 2, 3, 4 of the two pair generators with gauge charge +-2a/... (the labels that lost terms before 2026-09-26)
    U3 = ungauge_u1a1aodd(3)
    G3 = U3._G
    bad_w = []
    n_w = 0
    for pair in (((1, 8), (3, 7)), ((1, 9), (3, 8))):
        for a in (2, 3, 4):
            F = tuple(sorted((t, i, a) for (t, i) in pair))
            g = _aodd_gauge_charge(F, 3)
            for K in range(1, 8):
                n_w += 1
                want = _aodd_sum(lambda n: _aodd_gd(G3.trace((F, n), K), K), K, 2 * K + 8 + abs(g))
                if _aodd_zd(U3.trace((F, 0), K), K) != want or G3._label_gauge_charge((F, 0)) != g:
                    bad_w.append((F, K))
    checks.append(check(f"the window covers |g + n| <= K+1: on the powers 2..4 of the pairs L_{{1;8,3;7}}, L_{{1;9,3;8}} at k = 3 ({n_w} label-order pairs, K = 1..7) the class equals the wide sum, and its gauge charge equals g of the printed charges",
                        not bad_w, str(bad_w[:3])))
    # Tr_gauged L_(F,n) = O(q^|g+n|) on the generators and their powers 2..4
    Kv, Nv = 16, 12
    slack_bad, n_sec = [], 0
    for k in (1, 2, 3):
        U = ungauge_u1a1aodd(k)
        G = U._G
        for (F0, _e) in U.mult_generators():
            for a in (1, 2, 3, 4):
                F = tuple((t, i, m * a) for (t, i, m) in F0)
                g = _aodd_gauge_charge(F, k)
                n_sec += 1
                for n in range(-g - Nv, -g + Nv + 1):
                    d = _aodd_gd(G.trace((F, n), Kv), Kv)
                    if d and min(d) < abs(g + n):
                        slack_bad.append((k, F, n, min(d)))
                        break
    checks.append(check(f"Tr_gauged L_(F,n) = O(q^|g+n|), g the e_1-component of the printed charge of L_(F,0): on {n_sec} labels (the generators and their powers 2..4 at k = 1, 2, 3), n within {Nv} of -g, through q^{Kv}",
                        not slack_bad, str(slack_bad[:3])))
    # eq:comp-aodd-closed: Tr 1 and Tr L_2m;1 from the printed eq:comp-singlet / eq:comp-longchord; Tr L_2m;0 by z -> 1/z; rho^2 for the rest
    Kc = 30
    bad_c, n_c = [], 0
    for k in (1, 2, 3):
        p = k + 2
        U = ungauge_u1a1aodd(k)
        n_c += 1
        if _aodd_zd(U.trace(U.identity(), Kc), Kc) != _aodd_sum(lambda n: _singlet(p, n, Kc), Kc, Kc + 2):
            bad_c.append((k, "Tr 1"))
        for t in range(2, k + 2, 2):
            m = t // 2
            n_c += 2
            if _aodd_zd(U.trace((((t, 1, 1),), 0), Kc), Kc) != _aodd_sum(lambda n: _chord_rule(p, m, n, Kc), Kc, Kc + 8):
                bad_c.append((k, t, 1))
            if _aodd_zd(U.trace((((t, 0, 1),), 0), Kc), Kc) != _aodd_sum(lambda n: _chord_rule(p, m, n, Kc), Kc, Kc + 8, zsign=-1):
                bad_c.append((k, t, 0))
            if _charge(t, 1, k)[0] != 0 or _charge(t, 0, k)[0] != 0:
                bad_c.append((k, t, "gauge charge"))
            s = _aodd_size(t, k)
            for i in range(s):
                n_c += 1
                a = _aodd_zd(U.trace((((t, i, 1),), 0), Kc), Kc)
                b = _aodd_zd(U.trace((((t, (i + 2) % s, 1),), 0), Kc), Kc)
                sh = _drift(t, (i + 1) % s, k) - _drift(t, i, k)
                if b != {(q, z + sh): c for (q, z), c in a.items()}:
                    bad_c.append((k, t, i, "rho^2"))
    checks.append(check(f"eq:comp-aodd-closed: Tr 1 = (q^2;q^2)^-2 sum_n z^n Tr_gauged(E^n) from eq:comp-singlet, Tr L_2m;1 from eq:comp-longchord at index n (the chords L_2m;0, L_2m;1 have gauge charge 0), Tr L_2m;0 with z -> 1/z, and Tr L_2m;i+2 = z^(d(2m,i+1)-d(2m,i)) Tr L_2m;i at every position; k = 1, 2, 3 through q^{Kc} ({n_c} comparisons)",
                        not bad_c, str(bad_c[:4])))
    # the printed heads eq:comp-aodd-heads
    heads = {
        (1, "1"): {(0, 0): 1, (2, -1): 1, (2, 0): 1, (2, 1): 1, (4, -2): 1, (4, -1): 2, (4, 0): 3, (4, 1): 2, (4, 2): 1},
        (1, "L2;0"): {(1, 0): -1, (1, 1): -1, (3, -1): -1, (3, 0): -1, (3, 1): -1, (3, 2): -1},
        (1, "L1;0,1;3"): {(1, 1): -1, (5, 0): -1, (5, 1): -1, (5, 2): -1},
        (2, "1"): {(0, 0): 1, (2, 0): 1, (3, -1): -1, (3, 1): -1, (4, 0): 3, (5, -1): -2, (5, 1): -2, (6, -2): 1, (6, 0): 5, (6, 2): 1},
        (3, "1"): {(0, 0): 1, (2, 0): 1, (4, -1): 1, (4, 0): 3, (4, 1): 1, (6, -1): 2, (6, 0): 5, (6, 1): 2},
    }
    lab = {"1": ((), 0), "L2;0": (((2, 0, 1),), 0), "L1;0,1;3": (((1, 0, 1), (1, 3, 1)), 0)}
    cut = {"1": 6, "L2;0": 4, "L1;0,1;3": 5}
    bad_h = []
    for (k, nm), want in heads.items():
        Kh = 4 if (k == 1 and nm == "1") else cut[nm]
        if _aodd_zd(ungauge_u1a1aodd(k).trace(lab[nm], Kh), Kh) != want:
            bad_h.append((k, nm))
    checks.append(check("the printed heads eq:comp-aodd-heads: Tr 1 at k = 1, 2, 3, Tr L_2;0 and Tr L_{1;0,1;3} at k = 1", not bad_h, str(bad_h)))
    # positive control, and the k = 1 statement: the paper's [A_1, D_3] traces restricted to the Cartan U(1)
    K3 = 30
    D = A1D3KAlg()

    def restrict(t):          # chi_n -> sum_j w^(n - 2j), w = z^(1/2): doubled z-exponents
        out = {}
        for q, r in t.coeffs.items():
            if q > K3:
                continue
            for nn, c in r.terms.items():
                nn = nn[0] if isinstance(nn, tuple) else nn
                for j in range(nn + 1):
                    out[(q, nn - 2 * j)] = out.get((q, nn - 2 * j), 0) + c
        return {x: v for x, v in out.items() if v}
    U1 = ungauge_u1a1aodd(1)
    doubled = lambda d: {(q, 2 * z): c for (q, z), c in d.items()}   # noqa: E731
    shifted = lambda d, s: {(q, w + s): c for (q, w), c in d.items()}   # noqa: E731
    ok_vac = doubled(_aodd_zd(U1.trace(U1.identity(), K3), K3)) == restrict(D.trace(D.identity(), K=K3))
    ok_d = doubled(_aodd_zd(U1.trace((((2, 0, 1),), 0), K3), K3)) == shifted(restrict(D.trace(D.D(0), K=K3)), 1)
    ok_t = doubled(_aodd_zd(U1.trace((((1, 0, 1), (1, 3, 1)), 0), K3), K3)) == shifted(restrict(D.trace(D.T(0), K=K3)), 2)
    checks.append(check(f"positive control and the k = 1 statement: Tr 1 is A1D3KAlg's Tr 1 (the paper's [A_1,D_3]) restricted, chi_n -> z^(-n/2) + ... + z^(n/2); Tr L_2;0 = z^(1/2) Tr D_0 and Tr L_{{1;0,1;3}} = z Tr T_0 after restriction; through q^{K3}",
                        ok_vac and ok_d and ok_t, f"vacuum {ok_vac}, D {ok_d}, T {ok_t}"))
    # negative controls
    Kn = 8
    neg_meas = _aodd_zd(U1.trace(U1.identity(), Kn), Kn) != _aodd_sum(lambda n: _singlet(3, n, Kn), Kn, Kn + 2, power=1)
    neg_orient = _aodd_zd(U1.trace((((2, 0, 1),), 0), Kn), Kn) != _aodd_sum(lambda n: _chord_rule(3, 1, n, Kn), Kn, Kn + 8)
    cube = ((1, 8, 3), (3, 7, 3))
    P = UngaugedKAlgebra(_AoddHiddenGaugeCharge(G3), ((), 1), epow=lambda lbl: lbl[1])
    old = _aodd_zd(P.trace((cube, 0), 4), 4)
    new = _aodd_zd(U3.trace((cube, 0), 4), 4)
    checks.append(check("negative control: the measure (q^2;q^2)^-1 in place of (q^2;q^2)^-2 fails Tr 1 at k = 1 through q^8", neg_meas))
    checks.append(check("negative control: Tr L_2;0 at k = 1 with the orientation of L_2;1 (index n instead of -n) fails through q^8", neg_orient))
    checks.append(check("negative control: with the gauge-charge window hidden (the E-power window alone, the class before 2026-09-26), the cube of L_{1;8,3;7} at k = 3 loses -q^3 z^-6 at K = 4",
                        new.get((3, -6)) == -1 and (3, -6) not in old, f"with the window {new.get((3, -6))}, without {old.get((3, -6))}"))
    pop.update({"window label-orders": n_w, "slack labels": n_sec, "closed-form comparisons": n_c, "seconds": round(time.time() - t0, 1)})
    return {"checks": checks, "population": pop,
            "controls": {"positive": "at k = 1, Tr 1 and the generator traces are the paper's [A_1,D_3] ones (A1D3KAlg) restricted, through q^30",
                         "negative": "the measure (q^2;q^2)^-1, the reflected orientation, and the E-power window alone are each caught"},
            "notes": "eq:comp-aodd-trace, eq:comp-aodd-closed and eq:comp-aodd-heads transcribed from the companion; the gauged seeds eq:comp-singlet / eq:comp-longchord are the comp:u1a1aodd transcriptions (_singlet, _chord_rule), the measure is expanded here.  The gauge-charge window was added to UngaugedKAlgebra.trace on 2026-09-26 (the E-power window alone dropped terms of the k = 3 pair powers).",
            "inputs": []}


def check_a1aodd_presentations(environment):
    from ungauge_kalgebra import ungauge_u1a1aodd
    import finite_kalgebras as fk
    import aodd_seeds as S
    from ad_characters import a3_elem_entry
    from u1_bootstrap import generate_u1
    from kalgebra_iso import KAlgebraIso
    from a1aodd_to_even_rgkalgebra import A1AoddToEvenRGKAlgebra
    from hexagon_kalg import HexagonKAlg
    from ungauged_polygon_kalg import UngaugedPolygonKAlg
    checks = []
    t0 = time.time()
    pop = {}
    one = _lp(0)
    ids = {1: "a3", 2: "a5", 3: "a7"}

    def samples(sid, iso):
        n = len(S.generator_map(sid))
        src = [iso.source.identity()] + [((), (((g, 1),), (0,))) for g in range(n)] + [((), ((), (m,))) for m in (1, -1)]
        tgt = [iso.target.identity()] + list(iso.target.mult_generators()) + [((), e) for e in (1, -1)]
        return src, tgt

    def battery(iso, src, tgt, K):
        se = [Element({l_: one}) for l_ in src]
        te = [Element({l_: one}) for l_ in tgt]
        return iso.verify_all(se, te, [(a, b) for a in se[1:] for b in se[1:]], [(a, b) for a in te[1:] for b in te[1:]], trace_K=K)

    def relabelled(iso, perm):
        return KAlgebraIso(iso.source, iso.target, lambda l_: iso.map(Element({perm(l_): one})),
                           lambda l_: Element({perm(x): c for x, c in iso.inverse(Element({l_: one})).terms.items()}), name="perturbed")
    # the zoo's exported cone tables: the certified KAlgebraIso onto ungauge_u1a1aodd(k)
    for k in (1, 2, 3):
        sid = ids[k]
        tk = time.time()
        iso = S.kalgebra_iso(sid)
        src, tgt = samples(sid, iso)
        res = battery(iso, src, tgt, 12)
        npair = (len(src) - 1) ** 2
        checks.append(check(f"{sid}: the KAlgebraIso from the zoo's Z-form wrapper onto ungauge_u1a1aodd({k}) passes {sorted(c for c, v in res.items() if v)} on the identity, {len(src) - 3} generators and the flavour characters, {npair} ordered pairs each way, traces through q^12",
                            all(res.values()) and len(res) == 5, f"{res}; {time.time() - tk:.1f} s"))
        pop[sid] = {"pairs each way": npair, "seconds": round(time.time() - tk, 1)}
    # negative controls: the flavour normalisation flipped, two zoo generators transposed
    neg = {}
    for sid in ("a3", "a5"):
        iso = S.kalgebra_iso(sid)
        src, tgt = samples(sid, iso)

        def flip(l_):
            w, (word, (m,)) = l_
            return (w, (word, (-m,)))

        def swap(l_):
            w, (word, m) = l_
            tt = {0: 1, 1: 0}
            return (w, (tuple(sorted((tt.get(g, g), p) for g, p in word)), m))
        neg[sid] = (sorted(c for c, v in battery(relabelled(iso, flip), src, tgt, 6).items() if not v),
                    sorted(c for c, v in battery(relabelled(iso, swap), src, tgt, 6).items() if not v))
    checks.append(check(f"negative control: with the flavour normalisation flipped, or two zoo generators transposed, the isomorphism battery fails (a3, a5: {neg})",
                        all(a and b for a, b in neg.values())))
    # the zoo's own trace of composite labels (its Layer-1 reduction over its cone table) against the class, incl. the k = 3 pair cube
    bad_z, n_z = [], 0
    for k, lab, K in ((1, (((1, 0, 1), (1, 3, 1), (2, 0, 2)), 0), 8), (2, (((2, 0, 2),), 0), 8),
                      (3, (((1, 8, 3), (3, 7, 3)), 0), 4), (3, (((1, 9, 3), (3, 8, 3)), 0), 4), (3, (((1, 8, 2), (3, 7, 2)), 0), 6)):
        iso = S.kalgebra_iso(ids[k])
        (zl, _c), = iso.inverse(Element({lab: one})).terms.items()
        n_z += 1
        if _aodd_zd(iso.source.trace(zl, K), K) != _aodd_zd(ungauge_u1a1aodd(k).trace(lab, K), K):
            bad_z.append((k, lab))
    checks.append(check(f"the zoo's own trace (its Layer-1 reduction over its exported cone table) equals the class's on {n_z} composite labels, among them the pair powers at k = 3 that the E-power window alone got wrong",
                        not bad_z, str(bad_z)))
    # the named classes: HexagonKAlg = k = 1 with E inverted; UngaugedPolygonKAlg(k) = the same labels; both read z as 1/z
    inv = lambda d: {(q, -z): c for (q, z), c in d.items()}   # noqa: E731
    Hx, U1 = HexagonKAlg(), ungauge_u1a1aodd(1)
    flipE = lambda l_: (l_[0], -l_[1])   # noqa: E731
    hg = Hx.mult_generators()
    hx_prod = all(Element({flipE(L): c for L, c in Hx.multiply(a, b).terms.items()}) == U1.multiply(flipE(a), flipE(b)) for a in hg for b in hg)
    hx_tr = all(_aodd_zd(Hx.trace(l_, 8), 8) == inv(_aodd_zd(U1.trace(flipE(l_), 8), 8)) for l_ in [((), 0), ((), 1)] + hg + [(hg[0][0], 2)])
    named_ok = {}
    for cls_k in (2, 3, 4):
        A, U = UngaugedPolygonKAlg(cls_k), ungauge_u1a1aodd(cls_k)
        g = A.mult_generators()
        prods = all(A.multiply(a, b) == U.multiply(a, b) for a in g[:5] for b in g[-5:])
        trs = all(_aodd_zd(A.trace(l_, 6), 6) == inv(_aodd_zd(U.trace(l_, 6), 6)) for l_ in [((), 0), ((), 2)] + g[:3] + [(g[-1][0], -1)])
        named_ok[cls_k] = prods and trs
    checks.append(check(f"the named classes: HexagonKAlg is k = 1 under (F, e) -> (F, -e) (all 36 generator products; traces through q^8 with z -> 1/z); UngaugedPolygonKAlg(2, 3, 4) has the same labels, products and, with z -> 1/z, traces ({named_ok})",
                        hx_prod and hx_tr and all(named_ok.values()), f"hexagon products {hx_prod}, traces {hx_tr}"))
    # witness: Tr 1 from the RG flow A1AoddToEvenRGKAlgebra(k)
    Kr = 16
    rg = {}
    for k in (1, 2, 3):
        tk = time.time()
        A = A1AoddToEvenRGKAlgebra(k)
        U = ungauge_u1a1aodd(k)
        ref = {}
        for q, r in A.trace(A.identity(), Kr).coeffs.items():
            if q > Kr:
                continue
            for key, c in r.terms.items():
                z = key[0] if isinstance(key, tuple) else 0
                v = sum(c.terms.values()) if hasattr(c, "terms") else c
                if v:
                    ref[(q, z)] = ref.get((q, z), 0) + v
        rg[k] = (_aodd_zd(U.trace(U.identity(), Kr), Kr) == {x: v for x, v in ref.items() if v}, round(time.time() - tk, 1))
    checks.append(check(f"witness: Tr 1 equals the RG flow A1AoddToEvenRGKAlgebra(k)'s at k = 1, 2, 3 through q^{Kr} ({rg})", all(v[0] for v in rg.values())))
    # witness: the zoo's a3 closed-form characters, through q^48
    Ka = 48
    ref = a3_elem_entry(Ka)
    sd = S.seeds("a3")
    ok_a3 = sd.vacuum_trace(Ka) == ref["identity"] and all(sd.seed_trace(i, Ka) == ref["orbits"][i] for i in range(6))
    Ua = S.ungauged_algebra("a3")
    mu_rows = lambda d: {q: {(k_ if isinstance(k_, tuple) else (k_,)): int(c) for k_, c in row.items() if c} for q, row in d.items() if q <= 12}   # noqa: E731
    wrong_orient = 0
    for i in range(6):
        t12 = Ua.trace(S.to_ungauged_label("a3", ((i, 1),)), 12)
        rb = {}
        for q, r in t12.coeffs.items():
            row = {(z,): int(v) for (z,), v in r.terms.items() if v}
            if row and q <= 12:
                rb[q] = row
        wrong_orient += rb != {q: r for q, r in mu_rows(a3_elem_entry(12)["orbits"][i]).items() if r}
    checks.append(check(f"witness: Tr 1 and the six a3 generator traces equal the closed-form characters of finite_kalgebras.ad_characters through q^{Ka}; negative control: read back with z -> z instead of z -> 1/z, {wrong_orient} of the 6 differ",
                        ok_a3 and wrong_orient > 0))
    # witness: the u(1) orthonormality bootstrap
    boot = {}
    for sid, K in (("a5", 10), ("a7", 6)):
        tk = time.time()
        rec = generate_u1(sid, K)
        sd = S.seeds(sid)
        ns = len(rec["orbits"])
        okk = sum(sd.seed_trace(i, K) == mu_rows(rec["orbits"][i]) or sd.seed_trace(i, K) == {q: r for q, r in mu_rows(rec["orbits"][i]).items() if r} for i in rec["orbits"])
        vac = sd.vacuum_trace(K) == {q: r for q, r in mu_rows(rec["identity"]).items() if r}
        boot[sid] = (vac, okk, ns, round(time.time() - tk, 1))
    checks.append(check(f"witness: the u(1) orthonormality bootstrap generate_u1 gives the same Tr 1 and generator traces (a5 through q^10, a7 through q^6: {boot})",
                        all(v[0] and v[1] == v[2] for v in boot.values())))
    pop.update({"rg": rg, "bootstrap": boot, "seconds": round(time.time() - t0, 1)})
    return {"checks": checks, "population": pop,
            "controls": {"positive": "the certified isomorphism with the zoo's exported cone tables; Tr 1 against the RG flow",
                         "negative": "a flipped flavour normalisation and two transposed zoo generators fail the isomorphism; the wrong read-back orientation fails the a3 characters"},
            "notes": "The zoo's a3/a5/a7 seed traces are served from ungauge_u1a1aodd through this map, so the isomorphism's trace check is not independent there; its product and rho checks are, and so are the zoo's own reduction of composite labels, the RG flow, the a3 characters and the u(1) bootstrap.",
            "inputs": []}


ADAPTERS = {
    "comp:u1a1aodd/implementation": check_u1a1aodd_implementation,
    "comp:u1a1aodd/is-kalgebra": check_u1a1aodd_is_kalgebra,
    "comp:u1a1aodd/embedding": check_u1a1aodd_embedding,
    "comp:u1a1aodd/traces": check_u1a1aodd_traces,
    "comp:a1aodd/implementation": check_a1aodd_implementation,
    "comp:a1aodd/hexagon": check_a1aodd_hexagon,
    "comp:a1aodd/is-kalgebra": check_a1aodd_is_kalgebra,
    "comp:a1aodd/traces": check_a1aodd_traces,
    "comp:a1aodd/presentations": check_a1aodd_presentations,
    "comp:su3ad/implementation": check_su3ad_implementation,
    "comp:su3ad/is-kalgebra": check_su3ad_is_kalgebra,
    "comp:su3ad/traces": check_su3ad_traces,
    "comp:a1deven/implementation": check_a1deven_implementation,
    "comp:a1deven/is-kalgebra": check_a1deven_is_kalgebra,
    "comp:a1deven/traces": check_a1deven_traces,
    "comp:a1deven/ungauging": check_a1deven_ungauging,
    "comp:a1dodd/implementation": check_a1dodd_implementation,
    "comp:a1dodd/is-kalgebra": check_a1dodd_is_kalgebra,
    "comp:a1dodd/traces": check_a1dodd_traces,
    "comp:a1dodd/geometry": check_a1dodd_geometry,
    "comp:e6/implementation": check_e6_implementation,
    "comp:e6/is-kalgebra": check_e6_is_kalgebra,
    "comp:e6/traces": check_e6_traces,
    "comp:e7/implementation": check_e7_implementation,
    "comp:e7/is-kalgebra": check_e7_is_kalgebra,
    "comp:e7/traces": check_e7_traces,
    "comp:e7/gauged": check_e7_gauged,
}
