"""Genuinely self-contained recursive spectrum generator -- NO spec / NO green
sequence anywhere.  S is built purely by:

    build_S([gamma])                = E(X_gamma)                       (base case)
    build_S(nodes), peel gamma:
        S_sub = build_S(nodes \\ gamma)                    (recursion)
        F_gamma :  F_gamma . S_sub = X_gamma + O(q)
                   (solved against the ELEMENT S_sub, s_coeff(d)=[S_sub]_d)
        S = E_q(F_gamma) . S_sub

EXACT Habiro; only truncation = charge sum along the node cone (deg<=CONE).
Ground-truth S_full (from a known chamber spec) is used ONLY as an independent
check.
"""
import sys; sys.path.insert(0, ".")
from fractions import Fraction

from lattice import Lattice
from habiro import HabiroElement
from bps_kalgebra_internals import solve_F_via_s_coefficient
from nahm_local import s_gamma_habiro

H0 = HabiroElement.zero(); H1 = HabiroElement.one()
_cn = {}
def c_n(n):
    if n not in _cn:
        _cn[n] = s_gamma_habiro((n, 0), [(1, 0)], [[0]])
    return _cn[n]

_qp = {}
def q_pow(n):
    """Memoised HabiroElement.q_power: bracket values repeat massively in
    qt_mul (measured 15-37 distinct across thousands of in-cone pairs).
    Shared instances are safe -- HabiroElement is frozen/immutable."""
    if n not in _qp:
        _qp[n] = HabiroElement.q_power(n)
    return _qp[n]


def mat_inverse(M):
    """Exact inverse (Fraction) of a square integer matrix via Gauss-Jordan."""
    n = len(M)
    A = [[Fraction(M[i][j]) for j in range(n)] + [Fraction(i == j) for j in range(n)]
         for i in range(n)]
    for col in range(n):
        piv = next(r for r in range(col, n) if A[r][col] != 0)
        A[col], A[piv] = A[piv], A[col]
        d = A[col][col]
        A[col] = [x / d for x in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [a - f * b for a, b in zip(A[r], A[col])]
    return [[A[i][n + j] for j in range(n)] for i in range(n)]


class Theory:
    def __init__(self, name, B, nodes, CONE, WINDOW=None):
        self.name = name
        self.lat = Lattice(B)
        self.D = len(B)
        self.CONE = CONE
        # WINDOW bounds the F-solver's BFS support box `[gamma, gamma+WINDOW*sum(cone_gens)]`.
        # In a degree-<=CONE cone no single generator can appear more than CONE
        # times, so WINDOW=CONE already covers the whole cone-truncated support
        # of F -- and is the *principled* default.  A larger fixed WINDOW makes
        # the BFS box blow up as WINDOW^(#cone_gens) (catastrophic in higher D:
        # e.g. D=6, 5 gens -> 11^5 points at WINDOW=10 vs 4^5 at WINDOW=CONE=3,
        # a ~4000x build speedup with identical exact results).
        self.WINDOW = CONE if WINDOW is None else WINDOW
        self.NODES = [tuple(v) for v in nodes]
        self.zero = tuple(0 for _ in range(self.D))
        Mcols = [[nodes[c][r] for c in range(self.D)] for r in range(self.D)]  # columns=nodes
        self.Minv = mat_inverse(Mcols)
        # opposite-pairing torus for the right-handed (tail) solve:
        #   S_sub . tildeF = X_gamma + O(q)  <=>  left-solve with pairing -B
        self._lat_op = Lattice([[-x for x in row] for row in B])
        # Fast path when the nodes are the standard basis (Minv == identity):
        # node-basis coords == the charge itself, so cone tests are pure
        # integer ops with no Fraction matrix-vector products (the hot path).
        ident = [[Fraction(i == j) for j in range(self.D)] for i in range(self.D)]
        self._std_basis = (self.Minv == ident)

    def coords(self, p):
        if self._std_basis:
            return p
        return [sum(self.Minv[i][j] * p[j] for j in range(self.D)) for i in range(self.D)]

    def in_cone(self, p):
        if self._std_basis:
            s = 0
            for x in p:
                if x < 0:
                    return False
                s += x
            return s <= self.CONE
        n = self.coords(p)
        return all(x >= 0 for x in n) and sum(n) <= self.CONE

    def add(self, a, b): return tuple(x + y for x, y in zip(a, b))
    def smul(self, n, a): return tuple(n * x for x in a)

    def qt_mul(self, A, Bd):
        """Cone-truncated torus product of two charge dicts.

        Graded fast path (measured 1.6-1.7x on whole guarded builds, exact-
        identical output, 2026-07-16): on the build path every operand key
        lies in the coord-nonnegative orthant (S / F / E_q supports), and
        coords() is linear,
        so in_cone(ga+gb) <=> deg(ga) + deg(gb) <= CONE.  Bucketing Bd by
        cone degree prunes out-of-cone pairs BEFORE any tuple-add / Habiro
        work (the dense loop's per-pair in_cone is the dominant qt_mul cost
        in non-standard-basis frames -- a Fraction matrix-vector product per
        pair); the bracket comes from a per-ga row ((ga.B).gb); q_power is
        memoised (q_pow above); per-charge contributions are summed once via
        HabiroElement.sum (tree-reduce, one simplify) instead of per-pair
        __add__.  A key outside the orthant (never produced on the build
        path) falls back to the dense loop, which stays the reference body.
        """
        coords = self.coords
        ia = []
        for g, c in A.items():
            cs = coords(g)
            if any(x < 0 for x in cs):
                return self._qt_mul_dense(A, Bd)
            ia.append((g, c, sum(cs)))
        buckets = {}
        for g, c in Bd.items():
            cs = coords(g)
            if any(x < 0 for x in cs):
                return self._qt_mul_dense(A, Bd)
            d = sum(cs)
            if d <= self.CONE:
                buckets.setdefault(d, []).append((g, c))
        bitems = sorted(buckets.items())
        Bp = self.lat.pairing; rng = range(self.D)
        out = {}
        for ga, ca, da in ia:
            rem = self.CONE - da
            if rem < 0:
                continue
            row = [0] * self.D                       # row_j = (ga . B)_j
            for i, ai in enumerate(ga):
                if ai:
                    Bi = Bp[i]
                    for j in rng:
                        if Bi[j]:
                            row[j] += ai * Bi[j]
            for d, items in bitems:
                if d > rem:
                    break
                for gb, cb in items:
                    ng = tuple(x + y for x, y in zip(ga, gb))
                    n = 0
                    for wj, bj in zip(row, gb):
                        if bj:
                            n += wj * bj
                    term = ca * cb * q_pow(n)
                    lst = out.get(ng)
                    if lst is None:
                        out[ng] = [term]
                    else:
                        lst.append(term)
        res = {}
        for g, lst in out.items():
            s = HabiroElement.sum(lst)
            if not s.is_zero():
                res[g] = s
        return res

    def _qt_mul_dense(self, A, Bd):
        """The pre-optimization dense double loop -- the graded fast path's
        reference semantics and its fallback off the coord-nonneg orthant."""
        out = {}; br = self.lat.bracket
        for ga, ca in A.items():
            for gb, cb in Bd.items():
                ng = self.add(ga, gb)
                if not self.in_cone(ng):
                    continue
                out[ng] = out.get(ng, H0) + ca * cb * HabiroElement.q_power(br(ga, gb))
        return {g: c for g, c in out.items() if not c.is_zero()}

    def factor_E(self, beta):
        out = {}; n = 0
        while self.in_cone(self.smul(n, beta)):
            out[self.smul(n, beta)] = c_n(n); n += 1
        return out

    def e_q(self, F):
        acc = {self.zero: H1}; Fpow = {self.zero: H1}; n = 0
        while True:
            n += 1
            Fpow = self.qt_mul(Fpow, F)
            if not Fpow:
                break
            cn = c_n(n)
            for g, c in Fpow.items():
                acc[g] = acc.get(g, H0) + cn * c
        return {g: c for g, c in acc.items() if not c.is_zero()}

    def solve_F(self, S_sub, gamma, cone_gens, side="left"):
        """Canonical element for `gamma` against `S_sub`.

        side='left'  : F   with   F . S_sub      = X_gamma + O(q)  (prepend)
        side='right' : tildeF with S_sub . tildeF = X_gamma + O(q)  (append),
                       solved as a left-solve in the opposite-pairing torus.
        """
        lat = self.lat if side == "left" else self._lat_op
        s_coeff = lambda d: S_sub.get(tuple(d), H0)
        csum = tuple(sum(cg[i] for cg in cone_gens) for i in range(self.D))
        sinv = lambda gg: tuple(-(gg[i] + self.WINDOW * csum[i]) for i in range(self.D))
        # Cap the F-support enumeration at the cone-degree SIMPLEX, not the
        # WINDOW*csum hypercube box.  Without a known sigma^{-1} the fallback box
        # has side ~cutoff per generator (~cutoff^#gens, exponential in rank), but
        # the true in-cone support has total cone-degree <= CONE, i.e.
        # deg(delta - gamma) <= CONE - deg(gamma).  The simplex (~md^g/g!) is
        # polynomial in rank at fixed cutoff and yields the IDENTICAL in-cone S
        # (charges past the cone are discarded anyway) -- measured 86x at rank 6,
        # >2700x at rank 8 (stock walls), gentle ~2-3x/2-rank scaling.
        md = max(0, self.CONE - int(sum(self.coords(gamma))))
        degree_fn = lambda d: int(sum(self.coords(d)))
        F = solve_F_via_s_coefficient(lat, cone_gens, gamma, s_coeff, sinv,
                                      max_degree=md, degree_fn=degree_fn)
        out = {}
        for g, c in F.items():
            h = HabiroElement.from_laurent(c.to_laurent())
            if not h.is_zero():
                out[g] = h
        return out

    def _monomial_gate_ok(self, S_sub, gamma, F, cone_gens, side="left"):
        """The monomial-charge gate: `E_q(F_gamma)` is the correct reattach factor
        iff `F` is a monomial RAY -- `F_{2gamma} == F_gamma^2` (checked exactly
        over the cone).  A character-valued charge (matter) fails it and needs the
        chi-expansion recursion instead.

        Measured (2026-07-16): the gate correlates PERFECTLY with build
        wrongness (0 false positives, 0 misses across exhaustive SU(2)+Nf
        orders, targeted SU(3)+Nf builds, and Markov) at 1.03-1.17x build
        cost.  It certifies
        the build AT THE BUILT CUTOFF only -- a pass at a small cutoff does
        not certify deeper (SU(3)+Nf=1 matter-last passes at cutoff 4, fails
        at 5).  The `|F| == 1` bare-monomial shortcut was verified against the
        full gate on every such peel before adoption."""
        if len(F) == 1:
            return True
        F2 = self.solve_F(S_sub, self.smul(2, gamma), cone_gens, side=side)
        return self._eq(F2, self.qt_mul(F, F))

    def build_S_guarded(self, nodes=None, *, chosen=None, _memo=None):
        """Production build: at each level try candidate (node, side) peels in
        smart cost order (prepend cost = incoming arrows, append = outgoing);
        solve F; run the monomial-charge gate; on gate failure try the next
        candidate; if ALL candidates fail, honest-fail with a ValueError
        naming the offending nodes -- the seam where the character-charge
        (chi-expansion) recursion plugs in.  Sub-builds are memoised per
        node-subset (failures too).  `chosen` (optional dict) records the
        (node, side, |F|) picked per subset, for tracing.

        This supersedes order heuristics as the correctness mechanism: on the
        SUN_Nf family the documented 'peel matter last' rule is INVERTED at
        cutoff >= 5 (the gauge top dressed by the matter is itself the
        character charge), so no static order rule is safe -- only the gate
        is."""
        memo = {} if _memo is None else _memo
        nodes = [tuple(n) for n in (self.NODES if nodes is None else nodes)]
        key = tuple(sorted(nodes))
        if key in memo:
            v = memo[key]
            if isinstance(v, Exception):
                raise v
            return v
        if len(nodes) == 1:
            S = self.factor_E(nodes[0])
            memo[key] = S
            return S
        br = self.lat.bracket
        cands = []
        for g in nodes:
            inc = sum(max(-br(g, d), 0) for d in nodes if d != g)
            out = sum(max(br(g, d), 0) for d in nodes if d != g)
            cands.append((inc, 0, g, "left"))
            cands.append((out, 1, g, "right"))
        cands.sort(key=lambda t: (t[0], t[1], t[2]))
        failures = []
        for _cost, _s, g, side in cands:
            sub = [d for d in nodes if d != g]
            try:
                S_sub = self.build_S_guarded(sub, chosen=chosen, _memo=memo)
            except ValueError:
                failures.append((g, side, "sub-build honest-failed"))
                continue
            F = self.solve_F(S_sub, g, cone_gens=sub, side=side)
            if not self._monomial_gate_ok(S_sub, g, F, sub, side=side):
                failures.append((g, side,
                                 f"|F|={len(F)} F_2g != F^2 (character charge)"))
                continue
            S = (self.qt_mul(self.e_q(F), S_sub) if side == "left"
                 else self.qt_mul(S_sub, self.e_q(F)))
            memo[key] = S
            if chosen is not None:
                chosen[key] = (g, side, len(F))
            return S
        err = ValueError(
            f"monomial-charge gate: NO (node, side) peel passes at level "
            f"|nodes|={len(nodes)} (nodes={nodes}); candidates tried: "
            f"{failures}.  These nodes need the character-charge recursion "
            f"(matter chi-expansion) -- the "
            f"naive E_q(F) reattach would build a silently wrong S.")
        memo[key] = err
        raise err

    def build_S(self, nodes, peel="last", trace=None, depth=0):
        if len(nodes) == 1:
            return self.factor_E(nodes[0])
        k = 0 if peel == "first" else len(nodes) - 1
        gamma = nodes[k]
        sub = [n for i, n in enumerate(nodes) if i != k]
        S_sub = self.build_S(sub, peel=peel, trace=trace, depth=depth + 1)
        F = self.solve_F(S_sub, gamma, cone_gens=sub)
        F2 = self.solve_F(S_sub, self.smul(2, gamma), cone_gens=sub)
        mono = self._eq(F2, self.qt_mul(F, F))
        if trace is not None:
            trace.append((len(nodes), gamma, len(F), mono))
        return self.qt_mul(self.e_q(F), S_sub)

    def S_from_spec(self, spec):
        km = [[self.lat.bracket(spec[i], spec[j]) for j in range(len(spec))]
              for i in range(len(spec))]
        out = {}
        order = self.NODES
        def rec(i, p):
            if not self.in_cone(p):
                return
            if i == self.D:
                h = s_gamma_habiro(p, spec, km)
                if not h.is_zero():
                    out[p] = h
                return
            kk = 0
            while True:
                q = self.add(p, self.smul(kk, order[i]))
                if not self.in_cone(q):
                    break
                rec(i + 1, q); kk += 1
        rec(0, self.zero)
        return out

    def _eq(self, A, Bd):
        ch = {g for g in set(A) | set(Bd) if self.in_cone(g)}
        return all(A.get(g, H0) == Bd.get(g, H0) for g in ch)

    def cmp(self, A, Bd, label, charges):
        bad = [g for g in charges if self.in_cone(g) and A.get(g, H0) != Bd.get(g, H0)]
        ok = not bad
        print(f"    {label}: {'MATCH (exact)' if ok else f'DIFFER ({len(bad)})'}")
        for g in sorted(bad)[:4]:
            print(f"        @{g}: got {A.get(g, H0).expand(7)}  want {Bd.get(g, H0).expand(7)}")
        return ok

    # ---- F-minimising peel heuristic ------------------------------------
    def cheap_peel_order(self, nodes):
        """Greedy order minimising |F| at each level: peel the node with the
        fewest UNSAFE incoming arrows (smallest character cone).  A node with
        none (a 'source', `<g,d> >= 0` for all remaining `d`) gives a bare
        monomial F.  order[0] is peeled first (top apex)."""
        cur = list(nodes); order = []
        while len(cur) > 1:
            unsafe = lambda g: sum(max(-self.lat.bracket(g, d), 0)
                                   for d in cur if d != g)
            gamma = min(cur, key=unsafe)
            order.append(gamma); cur = [d for d in cur if d != gamma]
        order.append(cur[0])
        return order

    def build_S_order(self, nodes, order, gate=False):
        """Recursion peeling nodes in the explicit `order` (order[0] first).

        `gate=True` runs the monomial-charge gate at every peel and raises
        `ValueError` on failure instead of building a silently wrong `S`
        (see `_monomial_gate_ok`); production entry points pass it."""
        nodes = list(nodes)
        if len(nodes) == 1:
            return self.factor_E(nodes[0])
        gamma = order[0]
        sub = [n for n in nodes if n != gamma]
        S_sub = self.build_S_order(sub, [o for o in order if o != gamma],
                                   gate=gate)
        F = self.solve_F(S_sub, gamma, cone_gens=sub)
        if gate and not self._monomial_gate_ok(S_sub, gamma, F, sub):
            raise ValueError(
                f"monomial-charge gate failed at the peel of {gamma} "
                f"(|F|={len(F)}, F_2g != F^2): this order treats a "
                f"character-charge (matter) node as a monomial charge -- the built "
                f"S would be silently wrong at this cutoff.  Omit `order` "
                f"(the guarded build finds a passing order automatically) or "
                f"supply one whose every peel is a monomial charge.")
        return self.qt_mul(self.e_q(F), S_sub)

    def build_S_cheap(self, nodes=None):
        nodes = self.NODES if nodes is None else nodes
        return self.build_S_order(nodes, self.cheap_peel_order(nodes))

    # ---- F-minimising peel with prepend/append (tail-tildeF) ------------
    def smart_peel_choice(self, nodes):
        """Pick the (node, side) with the smallest unsafe-dressing cost.
        Prepend cost = incoming arrows (sum max(-<g,d>,0)); append cost =
        outgoing arrows (sum max(<g,d>,0)).  A source -> prepend monomial;
        a sink -> append monomial."""
        best = None
        for g in nodes:
            inc = sum(max(-self.lat.bracket(g, d), 0) for d in nodes if d != g)
            out = sum(max(self.lat.bracket(g, d), 0) for d in nodes if d != g)
            cost, side = (inc, "left") if inc <= out else (out, "right")
            if best is None or cost < best[0]:
                best = (cost, g, side)
        return best[1], best[2]

    def build_S_smart(self, nodes=None, trace=None):
        """Recursion choosing, at each level, the cheapest node AND side
        (prepend `E_q(F).S_sub` or append `S_sub.E_q(tildeF)`).  Acyclic
        quivers peel entirely with monomial F's."""
        nodes = list(self.NODES if nodes is None else nodes)
        if len(nodes) == 1:
            return self.factor_E(nodes[0])
        gamma, side = self.smart_peel_choice(nodes)
        sub = [d for d in nodes if d != gamma]
        S_sub = self.build_S_smart(sub, trace=trace)
        F = self.solve_F(S_sub, gamma, cone_gens=sub, side=side)
        if trace is not None:
            trace.append((len(nodes), gamma, side, len(F)))
        if side == "left":
            return self.qt_mul(self.e_q(F), S_sub)
        return self.qt_mul(S_sub, self.e_q(F))

    # ---- minimal-spec extraction (inverse problem) ----------------------
    def _degree(self, g):
        return sum(self.coords(g))

    def _partial_product(self, spec):
        P = {self.zero: H1}
        for beta in spec:
            P = self.qt_mul(P, self.factor_E(beta))
        return P

    def extract_spec_insert(self, S, cutoff=None, max_factors=24):
        """Minimal-spec extraction by *insertion* (the author's algorithm, 2026-06-27).

        No slope / green-sequence / front-tail assumption.  Walk the positive
        cone in increasing order; build a partial product `prod E_q(X_beta_i)`
        that matches S up to the current charge.  At the first mismatch at
        charge `g` the deficit must be exactly `c_1` (a single new hyper at g);
        try inserting `E_q(X_g)` at every position and recurse.  If the deficit
        is not `c_1` (not fixable by one E factor) we backtrack.  DFS keeps the
        SHORTEST spec (the minimal chamber).  Returns the spec or None.

        ⚠ WHAT `cutoff` MEANS, and why the default changed (2026-08-14).  The
        DFS matches `S` only at charges of degree `<= cutoff`, so the returned
        spec is verified exactly that far and no further.  The default used to be
        `CONE - 2`, leaving a two-degree gap between what was checked and what
        was handed back — and the gap is not theoretical.  Measured over ten
        quivers x three cone sizes, **six returned a spec that does not reproduce
        `S` inside the very cone it was given** — the three 3-cycles,
        SU(3)-cyclic, the 4-cycle and the 5-cycle, every one at `CONE = 4` — and
        three of the six returned a *shorter* wrong answer than the correct one
        (SU(3)-cyclic 5 vs 6, 4-cycle 5 vs 6, 5-cycle 6 vs 8), which is the
        direction a caller comparing lengths would prefer.  `cutoff = CONE` fixed
        all six, broke none, and lost no answer the old default found.

        This is the same error in an older place: an acceptance
        calibrated to a margin *inside* a truncation.  The production route
        `extract_spec_from_quiver` was never exposed to it — it passes `cutoff`
        explicitly and re-verifies over the full cone — so only direct callers
        taking the default were affected.

        ⚠ THE CORRECT DEFAULT IS ALSO THE EXPENSIVE ONE, and on this DFS that
        is not a rounding error: verifying two more degrees deepens the search
        and the memory goes with it — measured **> 4.7 GB RSS** on the 5-cycle,
        which is why a caller running this on rank ≥ 5 should box it in time and
        watch memory (the benchmark's part D does).  `factor_order_search.
        simplify_factorisation` answers the same question in seconds there.  The
        cheaper cutoff is still available explicitly; what is no longer available
        is getting it silently."""
        if cutoff is None:
            cutoff = self.CONE
        c1 = c_n(1)
        Sd = {g: c for g, c in S.items() if not c.is_zero()}
        order = sorted((g for g in self._all_cone_charges(cutoff)),
                       key=lambda g: (self._degree(g), g))
        best = [None]
        # prefix-cached partial products: P(spec) extends the longest cached
        # prefix (specs sharing a prefix reuse the product).  Pruning still cuts
        # children before their P is built (computed lazily per visited node).
        pp = {(): {self.zero: H1}}

        def partial(spec):
            key = tuple(spec)
            P = pp.get(key)
            if P is not None:
                return P
            i = len(spec)
            while tuple(spec[:i]) not in pp:
                i -= 1
            P = pp[tuple(spec[:i])]
            for j in range(i, len(spec)):
                P = self.qt_mul(P, self.factor_E(spec[j]))
                pp[tuple(spec[:j + 1])] = P
            return P

        def dfs(spec, scan):
            if best[0] is not None and len(spec) >= len(best[0]):
                return
            if len(spec) > max_factors:
                return
            P = partial(spec)
            gmis = jmis = None
            for j in range(scan, len(order)):       # charges < scan already matched
                g = order[j]
                if P.get(g, H0) != Sd.get(g, H0):
                    gmis, jmis = g, j
                    break
            if gmis is None:                        # matches S over the interior
                best[0] = list(spec)
                return
            if Sd.get(gmis, H0) - P.get(gmis, H0) != c1:   # not one new hyper
                return
            # try APPEND first: the minimal chamber is built by appending
            # factors in cone order, so this finds a full factorisation fast,
            # which then prunes longer branches by best-length.
            #
            # Commutation dedup: inserting E(gmis) on either side of a factor
            # E(beta) with <gmis,beta>=0 yields the IDENTICAL S (the two
            # E-factors commute), so positions separated only by commuting
            # factors are redundant.  Sweeping right-to-left, the tail is always
            # a representative; an interior position `pos` is a *new* class iff
            # the factor it crosses, spec[pos], does NOT commute with gmis.
            br = self.lat.bracket
            dfs(spec + [gmis], jmis)                 # pos = len  (append)
            for pos in range(len(spec) - 1, -1, -1):
                if br(gmis, spec[pos]) != 0:
                    dfs(spec[:pos] + [gmis] + spec[pos:], jmis)

        dfs([], 0)
        return best[0]

    def _all_cone_charges(self, cutoff):
        # enumerate non-negative node-basis combos with degree <= cutoff
        n = self.D
        res = []
        def rec(i, coeffs):
            if i == n:
                if sum(coeffs) <= cutoff:
                    g = self.zero
                    for k in range(n):
                        g = self.add(g, self.smul(coeffs[k], self.NODES[k]))
                    res.append(g)
                return
            c = 0
            while sum(coeffs) + c <= cutoff:
                rec(i + 1, coeffs + [c]); c += 1
        rec(0, [])
        return res


# --------------------------------------------------------------------------
# the peel engine is RETIRED — temporarily, and not erased
# --------------------------------------------------------------------------
#
# The author's ruling, 2026-08-13: *"The peel algorithm to build S seems a bit
# antiquated now.  Retire it temporarily (but do not erase it)."*  This
# supersedes the 2026-08-12 ruling, which demoted it from the default but left
# it selectable ("replace the peel engine, but leave it accessible").
#
# WHAT RETIREMENT MEANS HERE, precisely, because "retired" is doing real work:
#
#   * every line of the peel recursion stays — `Theory.build_S_guarded`,
#     `build_S_order`, `cheap_peel_order`, the monomial-charge gate, the whole
#     `Theory` machinery.  Nothing is deleted and nothing is moved to the source repository's archive.
#   * the two ways IN are closed: the module's own `build_spectrum_generator`
#     and `engine="peel"` on the dispatcher.  Both raise `RetiredEngineError`,
#     which is a `ValueError`, so a caller that was catching argument errors
#     still catches this.
#   * reversal is ONE flag.  `PEEL_RETIRED = False` restores the previous
#     behaviour exactly; `enable_retired_peel_engine()` does it for a block, and
#     `allow_retired=True` does it for a single call.
#
# The reason to gate rather than delete is that the peel engine is still the
# repo's only *independent* construction of `S` — it shares no mechanism with
# the factor engine, which is exactly what makes their agreement evidence rather
# than a self-check (1108/1108 dictionary entries).  A retired engine that can
# still be switched on keeps that cross-check available; a deleted one does not.
#
# NOT retired, and not to be confused with this: `extract_spec_from_quiver`,
# `build_spectrum_generator_auto` and `principled_sigma_maps` live in this
# module but are **engine-agnostic scaffolding** — they consume an `S`, they do
# not peel.  They keep working, on the factor engine.  Nor is
# `BPSQuiver.build_spectrum_generator` in `bps_quiver_tools` affected: despite
# the identical name it is the cluster-side spec builder (a negating sequence to
# a spec), a different object entirely, and it is explicitly retained.

PEEL_RETIRED = True

#: Engines the dispatcher recognises, including retired ones — so a caller
#: asking for `"peel"` gets the *retirement* message rather than an unhelpful
#: "must be one of [...]" that reads as though the engine never existed.
SPEC_FREE_ENGINES = ("factors", "peel")

#: Engines actually available.  This is what a chooser should offer.
ACTIVE_SPEC_FREE_ENGINES = ("factors",)


class RetiredEngineError(ValueError):
    """Raised when a retired `S`-engine is asked for without an opt-in."""


def _peel_gate(allow_retired: bool) -> None:
    """Refuse the peel route unless the caller has explicitly opted in."""
    if PEEL_RETIRED and not allow_retired:
        raise RetiredEngineError(
            "the peel S-engine is RETIRED.  The active engine is the "
            "crystalline factor one — `bps_factor_spectrum`, reached by "
            "engine='factors' (the default) — which builds everything the peel "
            "engine does and also the quivers its monomial-charge gate honest-fails "
            "on (N=2*/Markov, the wild ones).  The peel code is intact: pass "
            "allow_retired=True for one call, use the "
            "`enable_retired_peel_engine()` context manager for a block, or set "
            "`recursive_spectrum.PEEL_RETIRED = False` to un-retire it wholesale.")


def enable_retired_peel_engine():
    """Context manager re-enabling the retired peel engine for a block.

    For the paths that reach the engine through several layers of keyword —
    `BPSKAlgebra(build_S=True, build_S_engine="peel")` is the one that matters —
    where threading an `allow_retired` flag down would widen three public
    signatures to serve a retired route.  A context manager keeps the retirement
    a single switch, which is what makes it reversible.

    Not thread-safe (it toggles a module global); that is acceptable for its
    purpose, which is cross-checking in tests and one-off comparisons.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        global PEEL_RETIRED
        previous = PEEL_RETIRED
        PEEL_RETIRED = False
        try:
            yield
        finally:
            PEEL_RETIRED = previous

    return _ctx()


def _build_S_by_engine(pairing, node_charges, cutoff, *, engine="factors",
                       order=None, factor_order=None, phases=None, gate=True,
                       allow_retired=False):
    """`S` at a fixed cone cutoff, from whichever spec-free engine is asked for.

    The single dispatch point, so the auto-cutoff loop and the spec extractor
    below have exactly one notion of "build `S`" between them.  `"factors"` is
    `bps_factor_spectrum` (leading data + palindromic BPS factors); `"peel"` is
    `build_spectrum_generator` above (peel a node, solve `F`, reattach `E_𝖖(F)`).

    The two engines take genuinely different knobs and the wrong one silently
    doing nothing would be worse than an error, so a mismatch raises:
    `order` / `gate` are the peel engine's (a peel order over node charges, and
    the monomial-charge gate), while `factor_order` / `phases` are the factor engine's
    (a placement order and, optionally, a central charge).

    **No central charge is imposed here.**  `factor_order=None` leaves the factor
    engine on its own default — the source/sink strip on an acyclic quiver, the
    component order on a cyclic quiver that is not strongly connected (each
    strongly connected component built on its own, user 2026-09-23), and a random
    placement of the individual `E^{(s)}_𝖖(X_γ)^{Ω(γ,s)}` factors on a strongly
    connected one — so a central charge enters only if the caller supplies
    `phases`.  That is safe for every consumer
    of this function because it returns **`S` alone**, and `S` is
    order-independent; it is the *content* that a random order makes
    unphysical, and no caller here reads it.
    """
    if engine not in SPEC_FREE_ENGINES:
        raise ValueError(f"engine must be one of {list(SPEC_FREE_ENGINES)}")
    if engine == "factors":
        if order is not None:
            raise ValueError(
                "order=... is the PEEL engine's peel order over node charges; "
                "the factor engine takes factor_order=... (a placement order among "
                "'random' / 'phase' / 'lex' / 'degree-phase').  Pass "
                "engine='peel' to use a peel order.")
        from bps_factor_spectrum import BPSFactorSpectrum
        builder = BPSFactorSpectrum([list(r) for r in pairing],
                              [tuple(g) for g in node_charges], cutoff,
                              order=factor_order, phases=phases)
        builder.run()
        return builder.spectrum_generator()
    if factor_order is not None or phases is not None:
        raise ValueError(
            "factor_order=/phases= are the RAY engine's; the peel engine takes "
            "order=... (a peel order over node charges).")
    _peel_gate(allow_retired)
    T = Theory("specfree", [list(r) for r in pairing],
               [tuple(g) for g in node_charges], CONE=cutoff, WINDOW=cutoff)
    if order is not None:
        return T.build_S_order(T.NODES, [tuple(g) for g in order], gate=gate)
    if gate:
        return T.build_S_guarded()
    return T.build_S_order(T.NODES, T.cheap_peel_order(T.NODES))


def build_spectrum_generator(pairing, node_charges, cutoff, *,
                             window=None, order=None, gate=True,
                             allow_retired=False):
    """Spec-free spectrum generator `S = {γ: HabiroElement}` of the BPS quiver
    `(pairing, node_charges)`, built by the peel recursion (no spec, no green
    sequence).  Charges are keyed in the same lattice as `node_charges`.

    ⚠ **RETIRED, not erased.**  This is the peel
    engine's own entry point and it now raises `RetiredEngineError` unless
    `allow_retired=True` (or `PEEL_RETIRED` is off / the
    `enable_retired_peel_engine()` block is active).  Use
    `bps_factor_spectrum.build_spectrum_generator_from_factors`, or `engine="factors"` on the
    functions below, which is the default everywhere.  The recursion itself is
    untouched — see the retirement note above `SPEC_FREE_ENGINES` for why gating
    beats deleting (it is the repo's only *independent* construction of `S`, so
    keeping it switchable keeps the cross-check).

    **The monomial-charge gate is on by default**:
    with `order=None` the guarded build (`build_S_guarded`) picks a passing
    (node, side) chain automatically and honest-fails (ValueError) when none
    exists (character-charge / matter-chi nodes, e.g. N=2*/Markov); an explicit
    `order` is built as given but every peel is gate-checked (raise on
    failure).  `gate=False` restores the old unguarded behaviour — measured
    to build a silently WRONG `S` on e.g. SU(3)+matter at cutoff ≥ 5 for
    matter-last / `cheap_peel_order` orders; use it only for
    deliberate experiments.  Truncated only along the positive cone
    (`deg ≤ cutoff`); `window` defaults to `cutoff`.
    """
    _peel_gate(allow_retired)
    T = Theory("specfree", [list(r) for r in pairing],
               [tuple(g) for g in node_charges], CONE=cutoff, WINDOW=window)
    if order is None:
        if gate:
            return T.build_S_guarded()
        return T.build_S_order(T.NODES, T.cheap_peel_order(T.NODES))
    return T.build_S_order(T.NODES, [tuple(g) for g in order], gate=gate)


def _nodes_cone_interior(T, S, nodes, cutoff, certified=None):
    """True iff every node canonical `F_a` / `F̃_a` (left/right solve against `S`)
    is **cone-interior** — its cone-maximal charge has degree `< cutoff`.

    Interior ⟺ the cutoff strictly exceeds the canonical's true cone-degree, so
    `S` fully contains it (no truncation) and `σ(a)=−upper(F̃_a)`,
    `σ⁻¹(a)=−upper(F_a)` are stable.  A boundary `upper` (degree `≥ cutoff`)
    means the cone is too small.  This is the σ-stability certificate the
    auto-builder grows the cutoff until it meets.

    `certified` (optional mutable set of `(node, side)` pairs): interiority is
    **monotone** in the cutoff — a canonical's true cone-degree is fixed, so once
    `(a, side)` is interior at cutoff `c` it stays interior at every `c' > c`.
    The auto-builder passes a persistent `certified` set so already-interior
    pairs are **not re-solved** at wider cutoffs; this cuts the dominant cost of
    the widening loop (the interior checks, not the builds — ~80% on pure SU(3))
    without changing the result.  Pass `None` (default) to check every pair.
    """
    for a in nodes:
        for side in ("left", "right"):
            if certified is not None and (a, side) in certified:
                continue
            F = {k: v for k, v in T.solve_F(S, a, cone_gens=nodes, side=side).items()
                 if T.in_cone(k) and not v.is_zero()}
            if not F:
                return False
            cm = max(F, key=lambda g: (sum(T.coords(g)), g))
            if sum(T.coords(cm)) >= cutoff:
                return False
            if certified is not None:
                certified.add((a, side))
    return True


def build_spectrum_generator_auto(pairing, node_charges, *, engine="factors",
                                  order=None, factor_order=None, phases=None,
                                  base=4, max_iters=8, step=1, gate=True,
                                  allow_retired=False):
    """Spec-free `S`, **auto-growing the cone cutoff** until the node canonicals
    are cone-interior (σ-stable) — no user-supplied cutoff.  Returns `(S, cutoff)`.

    **`engine` is `"factors"` (`bps_factor_spectrum`, leading data + palindromic BPS
    factors) — since 2026-08-12 the default, and since 2026-08-13 the only active
    one**: `engine="peel"` now needs `allow_retired=True` (see the retirement
    note above `SPEC_FREE_ENGINES`).  The reason for both rulings is
    coverage: the peel engine's monomial-charge gate trips on a character-charge
    (matter) node and honest-fails, so N=2\\*/Markov and the wild quivers cannot
    be built through it at all, while the factor engine has no such step.  The
    cutoff loop itself is **engine-agnostic** — `_nodes_cone_interior` is a
    property of `S`, not of how `S` was made — which is why there is one loop
    here rather than one per engine.

    Mirrors the repo's adaptive Schur shell (`BPSKAlgebra._schur_index_stable`):
    build at increasing `cutoff = base, base+step, …`; settle at the first where
    `_nodes_cone_interior` certifies every node `F_a`/`F̃_a` is strictly interior
    (so `S` fully contains them and σ is stable).  On budget exhaustion
    (`max_iters` widenings) it **warns** and returns the widest build — a real
    signal, not a silent under-converged default.

    Scope note.  The certificate stabilises **σ/ρ** (the basis maps).  `multiply`
    is cone-truncated to the *built* degree — exact in-cone, truncated beyond
    (`from_ir_image` drops out-of-cone terms; see `rgkalgebra`); pass an explicit
    larger `cutoff` to a `BPSKAlgebra` when a deeper product is needed.
    """
    # NOTE (2026-07-14): a geometric-bracket + binary-search variant was tried to
    # cut the number of widenings, but it is net-negative — the cost is dominated
    # by the *high-cutoff* interior checks near the threshold (unavoidable to
    # certify σ), so fewer low-cutoff iterations save almost nothing (pure SU(3):
    # 1.26 s either way), and the binary-search rebuilds actually REGRESS
    # small-threshold cases (A4: 3 builds vs the linear scan's 2).  The linear
    # +step scan below is already near-optimal for the monotone-threshold search;
    # the σ-interior cutoff growth (~2·rank ⇒ cone ~4^rank) is an INHERENT
    # high-rank cost, not an iteration-count artifact.
    nodes = [tuple(g) for g in node_charges]
    S = None
    cutoff = base
    certified: set = set()   # (node, side) pairs already cone-interior (monotone)
    for _ in range(max_iters):
        T = Theory("auto", [list(r) for r in pairing], nodes,
                   CONE=cutoff, WINDOW=cutoff)
        S = _build_S_by_engine(pairing, nodes, cutoff, engine=engine,
                               order=order, factor_order=factor_order, phases=phases,
                               gate=gate, allow_retired=allow_retired)
        if _nodes_cone_interior(T, S, nodes, cutoff, certified):
            return S, cutoff
        cutoff += step
    import warnings
    warnings.warn(
        f"build_spectrum_generator_auto: node canonicals did not become "
        f"cone-interior within {max_iters} widenings (cutoff={cutoff - step}); "
        f"returning the widest build — σ may be under-converged.",
        RuntimeWarning, stacklevel=2)
    return S, cutoff - step


def extract_spec_from_quiver(pairing, node_charges, *, cutoff=8, engine="factors",
                             order=None, factor_order=None, phases=None,
                             gate=True, allow_retired=False):
    """Recover a **finite-chamber spec** for the BPS quiver
    `(pairing, node_charges)` spec-free: build `S`, then run the insertion
    extractor and **verify the result rebuilds `S` over the full cone** (not just
    the matching window — a too-low extractor cutoff returns a spurious short
    spec).

    Returns the spec (a list of charges, `S = ∏ E_𝖖(X_{spec_i})`) or **`None`**
    if no finite chamber is found at this `cutoff` (e.g. theories with no finite
    BPS chamber, like N=2\\*/Markov) — the caller then keeps the spec-free
    tRG path.  The extractor cutoff is swept up to `cutoff` until the recovered
    spec is full-cone-stable.

    **`engine` defaults to `"factors"`** (see `build_spectrum_generator_auto`; the
    peel alternative is retired).  Extraction itself is engine-independent — it
    consumes `S`, and the two engines produce the *same* `S` — so the engine only
    changes which route builds the input, and in particular a quiver whose peel
    build honest-fails can still be *asked* for a spec (and correctly answered
    `None` when it has no finite chamber, rather than raising).

    **This function is NOT retired**, and the distinction matters: it is
    engine-agnostic scaffolding that happens to live in the peel engine's module.
    What it returns is a finite factorisation with spin 0 only — which is what
    consumers actually need — verified by
    rebuilding `S` over the full cone.
    """
    T = Theory("specfree", [list(r) for r in pairing],
               [tuple(g) for g in node_charges], CONE=cutoff)
    S = _build_S_by_engine(pairing, node_charges, cutoff, engine=engine,
                           order=order, factor_order=factor_order, phases=phases,
                           gate=gate, allow_retired=allow_retired)
    full = set(S)
    for cut in range(2, cutoff + 1):
        spec = T.extract_spec_insert(S, cutoff=cut)
        if spec is None:
            continue
        Sx = T.S_from_spec(spec)
        if all(Sx.get(g, H0) == S.get(g, H0) for g in full | set(Sx)
               if T.in_cone(g)):
            return [tuple(b) for b in spec]
    return None


def _trop_mu(lat, alpha, p):
    """Lower tropical mutation `μ_p^t(α) = α + max(⟨α,p⟩,0)·p`."""
    m = lat.bracket(alpha, p)
    return tuple(a + max(m, 0) * x for a, x in zip(alpha, p))


def recursive_sigma_map(pairing, node_charges, cutoff, *, order=None):
    """The tropical `σ` of the BPS quiver `(pairing, node_charges)`, derived
    **spec-free, recursively from the built `S`** (no spec, no green-sequence
    BFS, no global tRG).  Returns a callable `σ(charge) -> charge`.

    Mechanism:
    at each peel of `γ` onto the sub-quiver `S_sub`,

        G_a = E_𝖖(F_γ)^{-1} · F^{UV}_a · E_𝖖(F_γ)        (one-factor conjugation)
        σ_UV(a) = min_b σ_sub(b)   over the labels b of G_a's decomposition
                                    in the sub-quiver canonical basis,

    recursing to the base case (single node `p` → `σ(α) = −μ_p^t(α)`).  The
    `min` is in cone order (`deg, lex`).  Verified == spec-σ on pentagon, pure
    SU(2), and **cyclic pure SU(3)** (where the global tRG is intractable).

    `cutoff` must be large enough to surface the deciding labels (the
    SU(3) `(0,1,0,0)` lesson — its label is at degree 7); too small silently
    truncates the decomposition.  Cost: one single-factor conjugation + a finite
    sub-canonical decomposition per peel.
    """
    T = Theory("specfree-sigma", [list(r) for r in pairing],
               [tuple(g) for g in node_charges], CONE=cutoff, WINDOW=cutoff)
    lat = T.lat

    def trim(d):
        return {k: v for k, v in d.items() if T.in_cone(k) and not v.is_zero()}

    def qt_inv(P):
        N = {k: v for k, v in P.items() if k != T.zero}
        acc = {T.zero: H1}; term = {T.zero: H1}; sgn = -1
        for _ in range(cutoff + 1):
            term = T.qt_mul(term, N)
            if not term:
                break
            for k, v in term.items():
                acc[k] = acc.get(k, H0) + sgn * v
            sgn = -sgn
        return trim(acc)

    def apex(d):
        return min(d, key=lambda g: (sum(T.coords(g)), g)) if d else None

    # Precompute the peel chain once: per level, the factor E_𝖖(F_γ), its
    # inverse, the sub-quiver S, and the full S.
    levels = {}

    def build(nodes, peel):
        if len(nodes) == 1:
            return T.factor_E(nodes[0])
        gamma = peel[0]
        sub = [n for n in nodes if n != gamma]
        S_sub = build(sub, [o for o in peel if o != gamma])
        F = trim(T.solve_F(S_sub, gamma, cone_gens=sub))
        E = trim(T.e_q(F))
        S = trim(T.qt_mul(E, S_sub))
        levels[tuple(nodes)] = dict(sub=tuple(sub), S_sub=S_sub, E=E,
                                    Einv=qt_inv(E), S=S)
        return S

    nodes0 = list(T.NODES)
    build(nodes0, list(order) if order is not None else T.cheap_peel_order(nodes0))

    _fir = {}

    def FIR(S_sub, sub, b):
        key = (id(S_sub), b)
        if key not in _fir:
            _fir[key] = trim(T.solve_F(S_sub, b, cone_gens=list(sub)))
        return _fir[key]

    def decompose(G, S_sub, sub):
        G = dict(G); out = []
        while True:
            G = trim(G)
            if not G:
                break
            b = apex(G); Fb = FIR(S_sub, sub, b); c = G.get(b, H0)
            if c.is_zero() or b not in Fb:
                break
            out.append(b)
            for k, v in Fb.items():
                G[k] = G.get(k, H0) - c * v
        return out

    def conemin(vs):
        return min(vs, key=lambda v: (sum(T.coords(v)), v))

    def sigma(nodes, a):
        if len(nodes) == 1:
            return tuple(-x for x in _trop_mu(lat, a, nodes[0]))
        L = levels[nodes]
        F_a = trim(T.solve_F(L['S'], a, cone_gens=list(nodes)))
        G = trim(T.qt_mul(T.qt_mul(L['Einv'], F_a), L['E']))
        labels = decompose(G, L['S_sub'], L['sub'])
        return conemin([sigma(L['sub'], b) for b in labels])

    nkey = tuple(T.NODES)
    return lambda a: sigma(nkey, tuple(a))


def principled_sigma_maps(pairing, node_charges, cutoff, *, order=None,
                          built_S=None, gate=True):
    """The tropical `σ` and `σ⁻¹` of the BPS quiver `(pairing, node_charges)`,
    derived **spec-free** straight from the axioms (no spec, no recursion, no
    global tRG).  Returns `(sigma, sigma_inverse)`, each a callable
    `charge -> charge`.

    Principled derivation.  The auxiliary quantum-torus ρ is negation, `ρ_QT(γ) = −γ`
    (`quantum_torus_kalgebra.py`), an involution.  The σ-axiom
    `F_a · S = S · ρ_QT(F_{σ(a)})` together with the right-solve
    `F_a · S = S · F̃_a` give `F̃_a = ρ_QT(F_{σ(a)})`; reading the repo's
    F-support interval `[a, −σ⁻¹(a)]` off the canonical's support then yields,
    with `upper(·)` the cone-maximal charge in the support,

        σ⁻¹(a) = −upper(F_a)        (F_a   :  F_a · S = X_a + O(𝖖),  left-solve)
        σ(a)   = −upper(F̃_a)        (F̃_a  :  S · F̃_a = X_a + O(𝖖),  right-solve)

    No spec, no tRG — both maps read directly off `solve_F`.  Verified `== spec-σ`
    on pentagon, pure SU(2), and **cyclic pure SU(3)** (all four nodes, both
    directions), where the global tRG is intractable.
    This supersedes the more elaborate `recursive_sigma_map`.

    **Truncation guard (important).**  `upper` is read from the cone-truncated
    support of `F`, so the F-solve cone (`cutoff`) must be large enough to
    contain the *true* `upper`.  A too-small cone returns a `upper` that sits on
    the cone boundary (node-degree `== cutoff`) and is short by one or more
    generators (the CONE=7-vs-9 SU(3) lesson: each truncated coordinate was off
    by exactly 1).  When the read `upper` lands on the boundary we raise
    `ValueError` rather than return a silently-wrong σ — raise `cutoff` (or
    `build_S_cutoff`) until it sits strictly interior.
    """
    T = Theory("principled-sigma", [list(r) for r in pairing],
               [tuple(g) for g in node_charges], CONE=cutoff, WINDOW=cutoff)
    nodes = list(T.NODES)
    if built_S is None:
        if order is not None:
            built_S = T.build_S_order(nodes, [tuple(g) for g in order],
                                      gate=gate)
        elif gate:
            built_S = T.build_S_guarded()
        else:
            built_S = T.build_S_order(nodes, T.cheap_peel_order(nodes))
    S = {tuple(g): c for g, c in built_S.items()}

    def _upper(a, side):
        F = {k: v for k, v in T.solve_F(S, a, cone_gens=nodes, side=side).items()
             if T.in_cone(k) and not v.is_zero()}
        if not F:
            raise ValueError(f"empty F for {a} (side={side}) -- cone too small")
        cm = max(F, key=lambda g: (sum(T.coords(g)), g))
        if sum(T.coords(cm)) >= cutoff:
            raise ValueError(
                f"upper(F_{a}) at cone boundary (deg {sum(T.coords(cm))} >= "
                f"cutoff {cutoff}); raise the cone -- σ would be truncated")
        return tuple(-x for x in cm)

    _cache_f, _cache_i = {}, {}

    def sigma(a):
        a = tuple(a)
        if a not in _cache_f:
            _cache_f[a] = _upper(a, "right")     # σ(a)   = −upper(F̃_a)
        return _cache_f[a]

    def sigma_inverse(a):
        a = tuple(a)
        if a not in _cache_i:
            _cache_i[a] = _upper(a, "left")      # σ⁻¹(a) = −upper(F_a)
        return _cache_i[a]

    return sigma, sigma_inverse


def run(theory, spec, peels=("last", "first")):
    print("\n" + "#" * 70)
    print(f"# {theory.name}   nodes={theory.NODES}")
    print("#" * 70)
    S_full = theory.S_from_spec(spec)
    gt = set(S_full.keys())
    print(f"ground-truth S_full from chamber spec ({len(spec)} factors): {len(gt)} cone charges")
    for peel in peels:
        trace = []
        S_rec = theory.build_S(theory.NODES, peel=peel, trace=trace)
        print(f"  peel={peel}: recursion trace (level |F| mono): " +
              "  ".join(f"L{lvl}:peel{g}|F|={nf}{'OK' if m else '!!'}" for lvl, g, nf, m in trace))
        theory.cmp(S_rec, S_full, f"build_S(peel={peel}) == S_full", gt)


if __name__ == "__main__":
    run(Theory("PENTAGON", [[0, 1], [-1, 0]], [(1, 0), (0, 1)], CONE=10),
        spec=[(1, 0), (0, 1)])
    run(Theory("PURE SU(2)", [[0, 1], [-1, 0]], [(1, 0), (-1, 2)], CONE=10),
        spec=[(1, 0), (-1, 2)])
    run(Theory("HEXAGON (3-cycle)", [[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
               [(1, 0, 0), (0, 1, 0), (0, 0, 1)], CONE=9),
        spec=[(0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 0)])
    run(Theory("PURE SU(3) (cyclic 1,2,1,2)",
               [[0, 1, 0, -2], [-1, 0, 2, 0], [0, -2, 0, 1], [2, 0, -1, 0]],
               [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)], CONE=8),
        spec=[(0, 1, 0, 0), (0, 0, 0, 1), (1, 1, 0, 0), (0, 0, 1, 1), (0, 0, 1, 0), (1, 0, 0, 0)])
