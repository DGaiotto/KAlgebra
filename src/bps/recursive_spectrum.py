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
        out = {}; br = self.lat.bracket
        for ga, ca in A.items():
            for gb, cb in Bd.items():
                ng = self.add(ga, gb)
                if not self.in_cone(ng):
                    continue
                out[ng] = out.get(ng, H0) + ca * cb * HabiroElement.q_power(br(ga, gb))
        return {g: c for g, c in out.items() if not c.is_zero()}

    def ray_E(self, beta):
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
        F = solve_F_via_s_coefficient(lat, cone_gens, gamma, s_coeff, sinv)
        out = {}
        for g, c in F.items():
            h = HabiroElement.from_laurent(c.to_laurent())
            if not h.is_zero():
                out[g] = h
        return out

    def build_S(self, nodes, peel="last", trace=None, depth=0):
        if len(nodes) == 1:
            return self.ray_E(nodes[0])
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

    def build_S_order(self, nodes, order):
        """Recursion peeling nodes in the explicit `order` (order[0] first)."""
        nodes = list(nodes)
        if len(nodes) == 1:
            return self.ray_E(nodes[0])
        gamma = order[0]
        sub = [n for n in nodes if n != gamma]
        S_sub = self.build_S_order(sub, [o for o in order if o != gamma])
        F = self.solve_F(S_sub, gamma, cone_gens=sub)
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
            return self.ray_E(nodes[0])
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
            P = self.qt_mul(P, self.ray_E(beta))
        return P

    def extract_spec_insert(self, S, cutoff=None, max_factors=24):
        """Minimal-spec extraction by *insertion* (user's algorithm, 2026-06-27).

        No slope / green-sequence / front-tail assumption.  Walk the positive
        cone in increasing order; build a partial product `prod E_q(X_beta_i)`
        that matches S up to the current charge.  At the first mismatch at
        charge `g` the deficit must be exactly `c_1` (a single new hyper at g);
        try inserting `E_q(X_g)` at every position and recurse.  If the deficit
        is not `c_1` (not fixable by one E factor) we backtrack.  DFS keeps the
        SHORTEST spec (the minimal chamber).  Returns the spec or None."""
        if cutoff is None:
            cutoff = self.CONE - 2
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
                P = self.qt_mul(P, self.ray_E(spec[j]))
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


def build_spectrum_generator(pairing, node_charges, cutoff, *,
                             window=None, order=None):
    """Spec-free spectrum generator `S = {γ: HabiroElement}` of the BPS quiver
    `(pairing, node_charges)`, built by the recursion (no spec, no green
    sequence).  Charges are keyed in the same lattice as `node_charges`.

    `order` is the peel order (`order[0]` peeled first / outermost); the default
    `cheap_peel_order` is all-monomial for pure-gauge / acyclic quivers.  For
    matter theories pass a gauge-first / matter-last order so every solved `F`
    is a gauge monomial ray.  Truncated only
    along the positive cone (`deg ≤ cutoff`); `window` defaults to `cutoff`.
    """
    T = Theory("specfree", [list(r) for r in pairing],
               [tuple(g) for g in node_charges], CONE=cutoff, WINDOW=window)
    if order is None:
        order = T.cheap_peel_order(T.NODES)
    return T.build_S_order(T.NODES, order)


def _nodes_cone_interior(T, S, nodes, cutoff):
    """True iff every node canonical `F_a` / `F̃_a` (left/right solve against `S`)
    is **cone-interior** — its cone-maximal charge has degree `< cutoff`.

    Interior ⟺ the cutoff strictly exceeds the canonical's true cone-degree, so
    `S` fully contains it (no truncation) and `σ(a)=−upper(F̃_a)`,
    `σ⁻¹(a)=−upper(F_a)` are stable.  A boundary `upper` (degree `≥ cutoff`)
    means the cone is too small.  This is the σ-stability certificate the
    auto-builder grows the cutoff until it meets.
    """
    for a in nodes:
        for side in ("left", "right"):
            F = {k: v for k, v in T.solve_F(S, a, cone_gens=nodes, side=side).items()
                 if T.in_cone(k) and not v.is_zero()}
            if not F:
                return False
            cm = max(F, key=lambda g: (sum(T.coords(g)), g))
            if sum(T.coords(cm)) >= cutoff:
                return False
    return True


def build_spectrum_generator_auto(pairing, node_charges, *, order=None,
                                  base=4, max_iters=8, step=1):
    """Spec-free `S`, **auto-growing the cone cutoff** until the node canonicals
    are cone-interior (σ-stable) — no user-supplied cutoff.  Returns `(S, cutoff)`.

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
    nodes = [tuple(g) for g in node_charges]
    S = None
    cutoff = base
    for _ in range(max_iters):
        T = Theory("auto", [list(r) for r in pairing], nodes,
                   CONE=cutoff, WINDOW=cutoff)
        order_k = order if order is not None else T.cheap_peel_order(T.NODES)
        S = T.build_S_order(T.NODES, order_k)
        if _nodes_cone_interior(T, S, nodes, cutoff):
            return S, cutoff
        cutoff += step
    import warnings
    warnings.warn(
        f"build_spectrum_generator_auto: node canonicals did not become "
        f"cone-interior within {max_iters} widenings (cutoff={cutoff - step}); "
        f"returning the widest build — σ may be under-converged.",
        RuntimeWarning, stacklevel=2)
    return S, cutoff - step


def extract_spec_from_quiver(pairing, node_charges, *, cutoff=8, order=None):
    """Recover a **finite-chamber spec** for the BPS quiver
    `(pairing, node_charges)` spec-free: build `S` by the recursion, then run
    the insertion extractor and **verify the result rebuilds `S` over the full
    cone** (not just the matching window — a too-low extractor cutoff returns a
    spurious short spec).

    Returns the spec (a list of charges, `S = ∏ E_𝖖(X_{spec_i})`) or **`None`**
    if no finite chamber is found at this `cutoff` (e.g. theories with no finite
    BPS chamber, like N=2\\*/Markov) — the caller then keeps the spec-free
    tRG path.  The extractor cutoff is swept up to `cutoff` until the recovered
    spec is full-cone-stable.
    """
    T = Theory("specfree", [list(r) for r in pairing],
               [tuple(g) for g in node_charges], CONE=cutoff)
    if order is None:
        order = T.cheap_peel_order(T.NODES)
    S = T.build_S_order(T.NODES, order)
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

    Mechanism (user, 2026-06-27 — "commute `F^{UV}_a` across
    `S^{UV}=E_𝖖(F^{IR}_γ)·S^{IR}` factor-first, then by recursion"):
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
            return T.ray_E(nodes[0])
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
                          built_S=None):
    """The tropical `σ` and `σ⁻¹` of the BPS quiver `(pairing, node_charges)`,
    derived **spec-free** straight from the axioms (no spec, no recursion, no
    global tRG).  Returns `(sigma, sigma_inverse)`, each a callable
    `charge -> charge`.

    Principled derivation (user, 2026-06-27 — "do a principled analysis of the
    axioms").  The auxiliary quantum-torus ρ is negation, `ρ_QT(γ) = −γ`
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
        if order is None:
            order = T.cheap_peel_order(nodes)
        built_S = T.build_S_order(nodes, order)
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
