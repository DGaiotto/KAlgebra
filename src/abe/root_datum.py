"""`root_datum` — the weight-lattice + root-system datum that parameterizes the
enriched rational quantum torus (Layer 1 of the `AbeKAlgebra` substrate
redesign).

Motivation
----------
The `v`-rational ring the abelianized chart lives in has monomials labelled by a
**weight lattice** and denominators of the uniform form

    1 − 𝖖^k · v^α        (α a root, k ∈ Z).

`RootDatum` is the object that supplies "what the weight lattice is, what the
roots are, and how the Weyl group acts" so the ring code (`WeylTorusRing`, to
come) is **group-agnostic**.  The single rule `1 − 𝖖^k v^α` for arbitrary
integer root vectors `α` subsumes both the U(N) factor `1 − 𝖖^k v_i/v_j`
(root `e_i − e_j`) and the SU(2) factor `1 − 𝖖^k v²` (root `α = 2ω = (2)`),
so the old separate `(1 − 𝖖^m v_i²)` square-factor engine is no longer needed.

Coordinate realization
---------------------------------------------
Weights and roots are integer tuples in a chosen basis `P ↪ Z^d`.  The two
families use the two natural realizations, and `RootDatum` hides the choice:

  * **U(N)** — the standard `e`-basis, `d = N`, `v = (v_1,…,v_N)`, simple roots
    `e_i − e_{i+1}`, Weyl group `S_N` permuting coordinates.  (The central /
    determinant U(1) is the diagonal, not a root.)
  * **SU(N)** — the fundamental-weight `ω`-basis, `d = N−1`, simple roots = the
    Cartan rows, Weyl group `S_N` acting by reflections.  In particular
    **SU(2)** has `d = 1` and the single positive root `α = (2)` — so
    `v^α = v²`, matching `pure_su2_h_abelianized` (`_N = 1`).

Both families here are type `A`, so positive roots are the contiguous partial
sums of the simple roots; a non-`A` extension would generalize
`_positive_roots_type_A`.

The `𝖖 → 1` shadow of the ring's denominators is the **Weyl denominator**
`∏_{α>0}(1 − v^{−α})`, exposed here as `weyl_denominator()`; for type `A` it
matches `sun_characters` up to the `v^ρ` monomial (anchored in the tests).
"""
from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from typing import Sequence


def _as_fraction(x):
    """Coerce an int / Fraction pairing value to `Fraction` (exact, no floats)."""
    return x if isinstance(x, Fraction) else Fraction(x)


Vec = tuple        # a weight / root / cocharacter: tuple[int, ...]
Mat = tuple        # a d×d integer matrix: tuple[tuple[int, ...], ...]


# ---------------------------------------------------------------------------
# small integer linear-algebra helpers (tuples → hashable, no numpy dep)
# ---------------------------------------------------------------------------
def _identity(d: int) -> Mat:
    return tuple(tuple(1 if i == j else 0 for j in range(d)) for i in range(d))


def _matvec(M: Mat, x: Vec) -> Vec:
    """`M·x`.

    Written with explicit loops rather than nested `sum(... for ...)` genexps
    because this is the innermost primitive of every Weyl action in the
    abelianized tier: profiling a single dressed SU(3) chart build measured
    **396k calls** here, spawning 1.19M generator frames and ~2.2 s — the largest
    single `sum()` consumer in the process.  The arithmetic is unchanged."""
    out = []
    for row in M:
        acc = 0
        for a, b in zip(row, x):
            acc += a * b
        out.append(acc)
    return tuple(out)


def _matmul(A: Mat, B: Mat) -> Mat:
    n, k, m = len(A), len(B), len(B[0])
    return tuple(
        tuple(sum(A[i][l] * B[l][j] for l in range(k)) for j in range(m))
        for i in range(n)
    )


def _det(M: Mat) -> int:
    """Integer determinant — `snf_kernel.int_det` (Bareiss fraction-free
    elimination, exact over Z, O(d³)).

    It was plain Laplace expansion, O(d!), on the assumption "d ≤ ~6"; the
    product datum of a longer linear quiver breaks it (`U(1)×U(2)×U(3)×U(4)`
    has d = 10: 10! terms per determinant for each of the 288 Weyl elements —
    constructing it did not finish in an hour, 2026-09-22).  The Laplace
    expansion is kept as `_det_laplace`, the independent construction the
    replacement is checked against (the suite in the source repository)."""
    from snf_kernel import int_det
    return int_det(M)


def _det_laplace(M: Mat) -> int:
    """The original Laplace-expansion determinant (kept as the cross-check)."""
    d = len(M)
    if d == 1:
        return M[0][0]
    total = 0
    for j in range(d):
        minor = tuple(tuple(M[i][c] for c in range(d) if c != j) for i in range(1, d))
        total += ((-1) ** j) * M[0][j] * _det_laplace(minor)
    return total


def _contragredient(M: Mat) -> Mat:
    """`(M^{-1})^T` for an integer matrix with `det = ±1` (so the result is
    integer).  This is how a Weyl element acts on COCHARACTERS given its action
    `M` on weights, so that the character–cocharacter pairing `⟨m, λ⟩` is
    preserved: `⟨(M^{-1})^T m, M λ⟩ = m^T M^{-1} M λ = ⟨m, λ⟩`.  For permutation
    matrices (U(N)) it equals `M` itself (self-dual)."""
    d = len(M)
    det = _det(M)
    if d == 1:
        return ((1 // det,),) if det in (1, -1) else ((M[0][0],),)
    out = []
    for i in range(d):
        row = []
        for j in range(d):
            minor = tuple(tuple(M[r][c] for c in range(d) if c != j)
                          for r in range(d) if r != i)
            row.append(((-1) ** (i + j)) * _det(minor) // det)   # cofactor(i,j)/det
        out.append(tuple(row))
    return tuple(out)


def _lpoly_mul(a: dict, b: dict) -> dict:
    """Multiply two weight-Laurent polynomials `{weight: int}`."""
    out: dict = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = tuple(ea[i] + eb[i] for i in range(len(ea)))
            out[e] = out.get(e, 0) + ca * cb
    return {e: c for e, c in out.items() if c != 0}


# ---------------------------------------------------------------------------
# Weyl-group construction (closure from simple reflections)
# ---------------------------------------------------------------------------
def _simple_reflection(alpha: Vec, coroot: Vec, d: int) -> Mat:
    """`s_α(λ) = λ − ⟨λ, α^∨⟩ α`, as a matrix: M[k][l] = δ_{kl} − α_k·α^∨_l."""
    return tuple(
        tuple((1 if k == l else 0) - alpha[k] * coroot[l] for l in range(d))
        for k in range(d)
    )


def _close_weyl(gens: Sequence[Mat], d: int) -> list[tuple[Mat, int]]:
    """Close a set of generator matrices into the full group; sign = det."""
    I = _identity(d)
    seen = {I}
    order = [I]
    frontier = [I]
    while frontier:
        nxt = []
        for e in frontier:
            for g in gens:
                p = _matmul(e, g)
                if p not in seen:
                    seen.add(p)
                    order.append(p)
                    nxt.append(p)
        frontier = nxt
    return [(M, _det(M)) for M in order]


def _positive_roots_type_A(simple_roots: list[Vec], d: int) -> list[Vec]:
    """Type-A positive roots = contiguous partial sums of the simple roots."""
    pos = []
    r = len(simple_roots)
    for i in range(r):
        acc = tuple(simple_roots[i])
        pos.append(acc)
        for j in range(i + 1, r):
            acc = tuple(acc[t] + simple_roots[j][t] for t in range(d))
            pos.append(acc)
    return pos


# ---------------------------------------------------------------------------
# The datum
# ---------------------------------------------------------------------------
class RootDatum:
    """Weight-lattice + root-system datum in a coordinate realization `P ↪ Z^d`.

    Construct via the `u_n` / `su_n` / `su_2` factories rather than directly.
    """

    __slots__ = ("dim", "simple_roots", "simple_coroots", "_positive",
                 "_pos_set", "_root_set", "weyl", "weyl_cochar", "name", "_pairing",
                 "_atom_phase", "_rho_sign", "_phase_is_canonical",
                 "_act_memo", "_orient_memo")

    def __init__(self, dim: int, simple_roots, simple_coroots, name: str,
                 positive_roots=None, pairing: Mat | None = None,
                 atom_phase=None, rho_sign=None, phase_is_canonical=None):
        self.dim = int(dim)
        self.simple_roots = [tuple(a) for a in simple_roots]
        self.simple_coroots = [tuple(a) for a in simple_coroots]
        if positive_roots is None:
            positive_roots = _positive_roots_type_A(self.simple_roots, self.dim)
        self._positive = [tuple(a) for a in positive_roots]
        self._pos_set = set(self._positive)
        self._root_set = self._pos_set | {tuple(-x for x in a) for a in self._positive}
        self.name = name
        self._pairing = pairing       # None ⇒ standard dot product
        self._atom_phase = atom_phase  # None ⇒ the root-system default (below)
        self._rho_sign = rho_sign      # None ⇒ the staircase default (below)
        #: optional override for `atom_phase_is_canonical`.  A datum that supplies
        #: its OWN `atom_phase` is its own certified convention and answers `True`
        #: by default (that is what keeps `u_n` untouched) — but a datum whose
        #: phase DELEGATES to sub-data, like `product_datum`, must answer the
        #: conjunction over those, or it would claim canonicity at a component's
        #: odd-height coweight.  See `atom_phase_is_canonical`.
        self._phase_is_canonical = phase_is_canonical
        #: memo for `act` — see there for why.  Pure function of `(w, x)`, so this
        #: can only ever change speed, never results.
        self._act_memo: dict = {}
        #: memo for `orient` — pure function of a root, and the root set is finite,
        #: so this is bounded by |Φ| × (k-values seen).  See `orient`.
        self._orient_memo: dict = {}
        gens = [_simple_reflection(a, c, self.dim)
                for a, c in zip(self.simple_roots, self.simple_coroots)]
        self.weyl = _close_weyl(gens, self.dim) if gens else [(_identity(self.dim), 1)]
        # the action on COCHARACTERS = contragredient of the weight action, same
        # index w (so `act`/`act_cochar` and `weyl`/`weyl_cochar` align element-wise).
        self.weyl_cochar = [(_contragredient(M), s) for (M, s) in self.weyl]

    # ----- roots / denominators (what TorusRational consumes) --------------
    def positive_roots(self) -> list[Vec]:
        return list(self._positive)

    def roots(self) -> list[Vec]:
        return list(self._root_set)

    def is_root(self, v) -> bool:
        return tuple(v) in self._root_set

    def orient(self, alpha) -> tuple[Vec, bool]:
        """Canonical positive form of the denominator factor `1 − 𝖖^k v^α`.

        Returns `(α⁺, flipped)`: if `flipped`, the ring stores the factor under
        the positive root `α⁺ = −α` with `k ↦ −k`, and multiplies the numerator
        by the monomial `−𝖖^k v^α` (since `1 − 𝖖^k v^α = −𝖖^k v^α·(1 − 𝖖^{−k}v^{−α})`).
        """
        alpha = tuple(alpha)
        # Memoized: called once per denominator key on EVERY `TorusRational`
        # construction — measured 139k constructions in one dressed SU(3) build.
        # Pure function of `alpha`, and `alpha` ranges over the finite root set,
        # so the memo is bounded by |Φ|.
        hit = self._orient_memo.get(alpha)
        if hit is not None:
            return hit
        if alpha in self._pos_set:
            out = (alpha, False)
            self._orient_memo[alpha] = out
            return out
        neg = tuple(-x for x in alpha)
        if neg in self._pos_set:
            out = (neg, True)
            self._orient_memo[alpha] = out
            return out
        raise ValueError(f"{alpha} is not a root of {self.name}")

    def shift_pairing(self, c, alpha) -> int:
        """`⟨c, α⟩` for the normal-ordering shift `v^λ ↦ 𝖖^{⟨c,λ⟩}v^λ`, which
        sends `1 − 𝖖^k v^α ↦ 1 − 𝖖^{k+⟨c,α⟩} v^α`.  Default: standard dot product."""
        c = tuple(c)
        alpha = tuple(alpha)
        if self._pairing is None:
            return sum(c[i] * alpha[i] for i in range(self.dim))
        return sum(c[i] * self._pairing[i][j] * alpha[j]
                   for i in range(self.dim) for j in range(self.dim))

    # ----- Weyl action (what the dressing / trace / recognize consume) -----
    def weyl_elements(self) -> list[int]:
        """Indices into `self.weyl`; pass to `act` / `sign`."""
        return list(range(len(self.weyl)))

    #: cap on `_act_memo`; on overflow it is dropped wholesale rather than grown
    #: without bound.  Nothing rests on a hit — `act` is a pure function — so the
    #: cap is a memory guard, not a correctness one.
    _ACT_MEMO_CAP = 200_000

    def act(self, w: int, x) -> Vec:
        """`w · x` — applies to weights and roots alike (same matrices).

        Memoized: this is called hundreds of thousands of times per dressed chart
        build (measured 396k `_matvec` calls in ONE SU(3) build), overwhelmingly on
        repeated `(w, weight)` pairs, because `weyl_act` sweeps the same supports
        for every Weyl element.  `act` is a pure function of its arguments, so the
        memo can only change speed."""
        xt = x if type(x) is tuple else tuple(x)
        # Memoize ONLY all-int vectors.  Python hashes `1.5 == Fraction(3,2)` as
        # equal, so a float-typed and a Fraction-typed query for the same
        # mathematical weight would collide and return the other one's TYPE — and
        # `ρ` really is float-valued at the half-integral forms (SO(3) `(0.5,)`,
        # SO(5) `(1.5, 0.5)`), so this is reachable, not hypothetical.  Numerically
        # the answers agree; a downstream `.denominator` would not survive it.
        if not all(type(v) is int for v in xt):
            return _matvec(self.weyl[w][0], xt)
        key = (w, xt)
        memo = self._act_memo
        hit = memo.get(key)
        if hit is not None:
            return hit
        val = _matvec(self.weyl[w][0], xt)
        if len(memo) >= self._ACT_MEMO_CAP:
            memo.clear()
        memo[key] = val
        return val

    def sign(self, w: int) -> int:
        return self.weyl[w][1]

    def weyl_orbit(self, x) -> set:
        return {self.act(w, x) for w in self.weyl_elements()}

    def is_dominant(self, lam) -> bool:
        lam = tuple(lam)
        return all(sum(lam[i] * cor[i] for i in range(self.dim)) >= 0
                   for cor in self.simple_coroots)

    def dominant_rep(self, lam):
        """The (unique) dominant weight in the Weyl orbit of `lam`."""
        lam = tuple(lam)
        for w in self.weyl_elements():
            x = self.act(w, lam)
            if self.is_dominant(x):
                return x
        raise ValueError(f"no dominant representative for {lam} in {self.name}")

    def dominance_key(self, lam):
        """A sortable key (the dominant rep); larger ⇒ more dominant."""
        return self.dominant_rep(lam)

    # ----- the COCHARACTER side (magnetic charges m ∈ X_*(T), dual to P) ----
    def act_cochar(self, w: int, m) -> Vec:
        """`w · m` on a cocharacter — the contragredient of the weight action,
        so `⟨w·m, w·λ⟩ = ⟨m, λ⟩`.  For U(N) (permutation Weyl) this equals `act`."""
        return _matvec(self.weyl_cochar[w][0], tuple(m))

    def is_dominant_cochar(self, m) -> bool:
        """`⟨α_i, m⟩ ≥ 0` for every simple root α_i — cocharacter dominance.
        Distinct from the weight `is_dominant` (`⟨λ, α_i^∨⟩ ≥ 0`) unless self-dual."""
        m = tuple(m)
        return all(sum(a[i] * m[i] for i in range(self.dim)) >= 0
                   for a in self.simple_roots)

    def dominant_cochar_rep(self, m):
        """The dominant cocharacter in the Weyl orbit of `m`."""
        m = tuple(m)
        for w in self.weyl_elements():
            x = self.act_cochar(w, m)
            if self.is_dominant_cochar(x):
                return x
        raise ValueError(f"no dominant cocharacter for {m} in {self.name}")

    # ----- the atom-normalization phase (cocycle-representative convention) --
    def atom_phase(self, m) -> int:
        """The scalar normalization phase `S(m)` of the torus atom `U_m` —
        equivalently, the choice of cocycle representative `R_{a,b}(v)` in
        `U_a U_b = R_{a,b}(v)·U_{a+b}` (rescaling `U_m ↦ ±𝖖^{S(m)}U_m` is the
        only scalar freedom).  The engine evaluates `S` at the DOMINANT
        cocharacter rep and transports along the Weyl orbit.  (The tier's
        language is atoms + residuals + cocycle only — no `u`'s and no
        difference-operator dressing appear on any surface.)

        **Bar-honesty constraint** (measured 2026-07-01): the bar-relevant
        content of `S` must be `−⟨ρ_weyl, m_dom⟩ = −½ Σ_{α>0} ⟨α, m⟩`.  A phase
        violating it makes `bar` fail antimultiplicativity on mixed-chamber
        products: the su_2 datum with `S ≡ 0` had `bar(H·H) ≠ H·H` while every
        residual of `H` was palindromic.

        **Two claims in the older wording were WRONG, both measured 2026-07-29.**

        * *"up to terms linear in `m` (linear pieces are coboundaries that cancel
          in the cocycle)"* — **false as implemented.**  Adding an integral linear
          `λ` and re-solving breaks the certified even sector: at `su_2` `m=(1,)`
          the shifted phase `S=0` makes `solve_canonical` return `inconsistent`,
          and at `b_n_simply_connected(2)` `m=(1,0)` the self-norm goes to `0`
          instead of `1 − 𝖖² + 𝖖⁴`.  `S` is far more rigid than that: on the even
          sublattice it must be `−½⟨Σ⁺,m⟩` **on the nose**.
        * *"`S` is Weyl-invariant — generically NON-linear"* — this was measured
          against the THEN-current code, where `_rho_block` read `_phase_S`
          directly.  **⚠ It no longer describes HEAD**: since
          `atom_phase_doubled` split the ρ path off, instrumenting `_phase_S`
          records **0 non-dominant calls, ever** — SO(3)/SU(2)/U(2), every gauge, in
          `multiply`/`rho`/`bar` (33 evaluations at SU(2), 161 at SU(3), 0
          non-dominant).  So in the ψ path `S` is consumed **only at
          `dominant_cochar_rep(m)`**, i.e. the effective phase really is the
          Weyl-invariant `S̃ = S ∘ dominant_cochar_rep`.  The non-dominant
          arguments now go to `atom_phase_doubled`, and only OVERRIDE data
          (`u_n`, `product_datum`) route them back through `_atom_phase` at the raw
          `k` — which is where the old observation still holds.

          **Consequence, and it is the operative one**: because the ψ path
          sees only `S̃`, the cocycle depends on `S` through `(−𝖖)^{δS̃}`, so a
          *linear* `S` is NOT invisible — `δS̃(a,b) = c·((a+b)_dom − a_dom − b_dom)`
          vanishes for all `c` only when dominant reps add (`a, b, a+b` in one
          closed chamber).  The surviving freedom is the **centre** alone.

        **`S` IS TOTAL AND INTEGER-VALUED — no fractional powers of `𝖖` anywhere**.  Writing
        `−½⟨Σ⁺,m⟩` was an artifact of expressing the scalar in a normalisation with
        no reason to be integral; on the coroot lattice `⟨Σ⁺,m⟩` is always even so
        the half never had to justify itself, and on `P^∨` it produced a spurious
        `𝖖^{1/2}` (and `(−1)^{1/2} = i`, since `S` enters `_psi_monomial_data` in
        both the `𝖖`-power and the sign).  The fix is the *parity correction* `ε`:

            S(m) = −(⟨Σ⁺,m⟩ − ε(m)) / 2 ,

        `ε(m) = 0` where `⟨Σ⁺,m⟩` is even, and `±1` where it is odd, signed by the
        first non-zero coordinate of `m` in the fundamental-coweight basis so that
        `ε(−m) = −ε(m)`.  Verified over a `P^∨` box at `su_2`, `su_n(3)`, `su_n(4)`,
        `b_n_simply_connected(2)`, `b_n_simply_connected(3)`, `sp_n(2)`, `sp_n(3)`,
        `g_2` — **0 failures** on each of: agrees with `−½⟨Σ⁺,m⟩` at every
        even-height `m`; integral everywhere; `S(−m) = −S(m)` everywhere.  The
        certified even sector is **bit-identical** (`su_2` `m=(1,),(2,)`; `su_n(3)`
        `m=(1,0),(1,1)`; `b_n_simply_connected(2)` `m=(1,0)`).

        ⚠ **This does NOT make an odd-height charge canonical**, and the guard for
        that has moved rather than gone — see `atom_phase_is_canonical`.  Making `S`
        total removes fractional powers from the *convention*; it says nothing about
        the construction.

        ⚠ **An earlier version of this paragraph argued the phase "provably cannot
        repair the odd-height self-norm" because an overall factor `c` contributes
        `c·bar(c) = 1`.  That reasoning is WRONG and is withdrawn.**
        `inner(x,y) = Tr(ρ(x)·y)` is **bilinear**, not sesquilinear: rescaling `x`
        by `c` moves `I` by `c²`.  Measured at SO(3) with `ρ` pinned, `S: 0 → −1` at
        `m=(1,)` moves `I` by exactly `𝖖²`, and at even `m=(2,)` **only**
        `S = −⟨ρ,m⟩` gives `𝖖⁰ = 1` (`S = 0` and `S = −⟨Σ⁺,m⟩` both fail).  Any
        earlier "invariant over `S_odd ∈ {0,±1,±2}`" reading came from a probe that
        moved `ρ` along with the phase — the `atom_phase_doubled` trap below.

        So the truth is the opposite of a free convention: **orthonormality PINS
        `S = −⟨ρ,m⟩`**, and independently the cocycle sees `S` through
        `(−𝖖)^{δS̃}` with `S̃ = S ∘ dominant_cochar_rep`, leaving only functions
        whose Weyl-symmetrization is additive — i.e. nothing off the **centre**
        (1484 rows).

        ⚠ **AND THE ODD-HEIGHT CASE IS NOT CLOSED BY THAT — THE PARAGRAPH THAT USED TO END HERE
        IS RETRACTED (2026-07-29).**  It argued that absorbing
        the residual `−𝖖^{−1}` needs `c² = −𝖖^{−1}`, hence the forbidden
        `i·𝖖^{−1/2}`, hence the tier cannot carry odd-`⟨Σ⁺,m⟩` lines.  There is no
        scalar to absorb.  Since orthonormality pins `S = −⟨ρ,m⟩`, the value this
        method returns is the *reduced* integral stand-in `−(⟨Σ⁺,m⟩ − ε)/2`, and
        `wrq_torus.cocycle_R` supplies the difference back through the **integral
        coboundary** `(−𝖖)^{δ(S_honest − S_used)}`.  That is legitimate precisely
        because the algebra never needs `S` as a number — only `δS̃`, which is an
        integer even where `S̃` is a half-integer (`π = ⟨Σ⁺,·⟩ mod 2` is
        Weyl-invariant and additive).  With that, SO(3)'s spinorial `H_0` is
        bar-invariant, squares to `L_{(2,0)}` exactly, and pairs to the
        `PureSO3KAlgebra` BPS oracle's `1 − 𝖖² + 𝖖⁴ + 𝖖⁶ − 𝖖⁸`; SO(5)/SO(7) odd
        charges build and are orthonormal; the even sector is bit-identical.  The
        `ε` correction, meanwhile, was **not** a free convention — used in the
        cocycle it breaks bar antimultiplicativity at odd height.  Battery:
        a probe in the source repository.

        The `u_n` factory overrides with the historical U(N) `Σ_j j·m_j` — equal to
        the default plus the linear central term `(N−1)/2·Σ_j m_j` at dominant
        `m`, i.e. honest, and kept verbatim for continuity with the certified
        `URQTorus` normalization.  **That is exactly the surviving freedom** (the centre is the only direction a Weyl-invariant additive functional can
        live in), which is why U(N) may carry its own convention while a semisimple
        `G` may not.  Overrides are returned untouched."""
        if self._atom_phase is not None:
            return self._atom_phase(m)
        tot = self._root_height(m)
        return -(tot - self._parity_correction(m, tot)) // 2

    def atom_phase_doubled(self, m):
        """`2·S(m)` — and it is **always an integer**, with no parity correction.

        This exists because `2S` and `S` need *different* treatment, which is a bug I
        introduced on 2026-07-29 and the author caught: `ρ` is **defined** so that the
        seed contributes exactly `1` to `I_{a,a}`, so any perturbation of `ρ` breaks
        `I_{a,a} = 1 + O(𝖖)` at once.

        `wrq_torus._rho_block` uses the phase only as `2·S(k)`, and the honest value
        `2S = −⟨Σ⁺,k⟩` is an integer on **all** of `P^∨` — the half never appears
        there, so `ε` must NOT be applied.  Feeding it the `ε`-corrected `atom_phase`
        shifted `ρ`'s `𝖖`-power by `ε = ±1` at odd height, which is exactly the
        `𝖖^{−2}`-type error that made the odd-height self-norm come out as
        `½𝖖⁻² + …` instead of `1 + O(𝖖)`.

        A datum with its own `atom_phase` override keeps it (doubled verbatim), so
        `u_n` and `product_datum` are unaffected."""
        if self._atom_phase is not None:
            return 2 * self._atom_phase(m)
        return -self._root_height(m)

    def _root_height(self, m):
        """`⟨Σ⁺, m⟩ = Σ_{α>0} ⟨α, m⟩`, exactly (a `Fraction` off the coroot
        lattice)."""
        return sum(self.shift_pairing(m, a) for a in self._positive)

    def _parity_correction(self, m, tot=None):
        """`ε(m)` — the term that makes `atom_phase` integral without moving it on
        the even sublattice: `0` where `⟨Σ⁺,m⟩` is even, `±1` where it is odd.

        The sign is **forced to be odd** under `m ↦ −m`, because `S` is used as a
        linear function (the ρ path calls it at `±k`; see `atom_phase`).  Taking
        `ε ≡ +1` at odd height — the obvious first guess — fails exactly there:
        `π(−m) = π(m)` for the parity character, so it gives
        `S(−m) = −S(m) + 1`.  Signing by the first non-zero coordinate of `m` in
        the fundamental-coweight basis is odd by construction, and no coordinate
        can be all-zero at odd height since `⟨Σ⁺,0⟩ = 0` is even."""
        from fractions import Fraction
        if tot is None:
            tot = self._root_height(m)
        tot = Fraction(tot)
        if tot.denominator != 1 or int(tot) % 2 == 0:
            return 0
        for a in self.simple_roots:
            c = Fraction(self.shift_pairing(tuple(m), tuple(a)))
            if c != 0:
                return 1 if c > 0 else -1
        return 0                        # unreachable: m = 0 has even height

    def atom_phase_is_canonical(self, m) -> bool:
        """Whether `atom_phase(m)` is the **canonical** normalisation at `m`, i.e.
        whether it needed no parity correction.

        This is the guard that used to be `atom_phase`'s `NotImplementedError` on
        odd `⟨Σ⁺,m⟩`.  Since `atom_phase` is now total (no fractional powers), the
        gate has to be asked separately — otherwise the tier would silently emit
        non-canonical elements at odd height, which is the `_cf_joint_fiber`
        failure mode the whole tier's discipline exists to prevent.

        **A parity test on `⟨Σ⁺,m⟩` would be WRONG here**, and `u_n` is the
        counterexample: `U(2)` at `m = (1,0)` has `⟨Σ⁺,m⟩ = 1`, odd, yet builds
        perfectly on its own certified convention (`u_n(2).atom_phase((1,0)) = 0`,
        integral).  So the question is *"did THIS datum's convention need a
        correction"*, not *"is the height even"* — a datum carrying its own
        `atom_phase` is its own certified convention and answers `True`.

        ⚠ **SINCE 2026-07-29 THE DEFAULT IS UNCONDITIONALLY `True`.**  The
        parity correction was never a demotion of the *algebra*, only of the
        materialised monomial `M(m) ∝ (−𝖖)^{⟨ρ,m⟩}` — the square root of the
        measure — and `wrq_torus.cocycle_R` now restores the honest phase exactly,
        through the integral coboundary factor `(−𝖖)^{π(a)π(b)}`.  Measured at
        SO(3) against the independent `PureSO3KAlgebra` BPS oracle: the spinorial
        `H_0` is bar-invariant, `H_0² = L_{(2,0)}` exactly, `I(H_0,H_0) =
        1 − 𝖖² + 𝖖⁴ + 𝖖⁶ − 𝖖⁸`, `I(H_0, L_{(2,0)}) = 0`, with the even and Wilson
        sectors bit-identical; and orthonormality's `𝖖⁰` condition holds at every
        odd charge measured at SO(5) — the non-simply-laced case.  So this is no
        longer a gate on which lines the tier carries.  The method is kept
        (returning `True`) because `global_form.LineLattice.abe_representable`
        reads it and because an explicit `phase_is_canonical=` override is still a
        useful escape hatch for probes.  Battery:
        a probe in the source repository."""
        if self._phase_is_canonical is not None:
            return bool(self._phase_is_canonical(m))
        return True

    def rho_sign_exp(self, k) -> int:
        """The sign exponent of the ρ-twist block at magnetic `k` (the engine
        reads `(−1)^{rho_sign_exp(k)}`).  **DERIVED, not a convention** (2026-07-29):

            rho_sign_exp(k)  =  2·S(k) + ⟨Σ⁺, k⟩
                             =  atom_phase_doubled(k) + ⟨Σ⁺, k⟩     (mod 2)

        i.e. the sign compensates exactly the deviation of *this datum's* atom
        phase from the honest `S = −⟨ρ, k⟩`.  Consequences:

        * a datum on the honest phase has exponent **identically 0**, so the
          sign is `+1` — which is what orthonormality requires, and is the
          missing piece that let the odd-`⟨Σ⁺,m⟩` sectors close;
        * `u_n`'s value is this law read on its central shift
          `S_used − S_honest = ½ Σ_j m_j`, and `weyl_vector_rho_sign = ⟨Σ⁺,k⟩`
          is it read on the (now retired) `ε`-shifted convention.

        Measured `≡` the previously shipped values (staircase, the `su_n` /
        `product_datum` / `weyl_vector_rho_sign` overrides) at **0 mismatches**
        over 1410 cocharacters on `u_n(2,3,4)`, `su_n(2,3,4)`, `so_n(3,5,7)`,
        `b_n_simply_connected(2,3)`, `sp_n(2,3)`, `g_2`, plus the
        `product_datum` combinations — a probe in the source repository.

        It also **retires a finding of the audit**: the coordinate staircase
        disagreed with `product_datum([u_n(1), u_n(1)])` on the same group
        `U(1)²` at 12 of 25 cocharacters; this law agrees with the product
        route everywhere, being additive over blocks by construction.

        An explicit `rho_sign=` passed to the constructor still wins, as an
        escape hatch for probes; no factory sets one any more."""
        if self._rho_sign is not None:
            return self._rho_sign(k)
        exp = _as_fraction(self.atom_phase_doubled(k)) + _as_fraction(
            self._root_height(k))
        if exp.denominator != 1:
            raise AssertionError(
                f"{self.name}: rho_sign_exp is non-integral at k={tuple(k)} "
                f"({exp}) — the datum's atom phase is not a half-integer "
                f"multiple of the honest −⟨ρ,k⟩.")
        return int(exp)

    # ----- 𝖖→1 anchor -----------------------------------------------------
    def weyl_vector(self) -> Vec:
        """`ρ = ½ Σ_{α>0} α` (may have half-integer entries)."""
        s = [0] * self.dim
        for a in self._positive:
            for i in range(self.dim):
                s[i] += a[i]
        half = tuple(x / 2 if x % 2 else x // 2 for x in s)
        return half

    def weyl_denominator(self) -> dict:
        """`∏_{α>0}(1 − v^{−α})` as `{weight: int}` (no 𝖖)."""
        poly = {(0,) * self.dim: 1}
        for a in self._positive:
            neg = tuple(-x for x in a)
            poly = _lpoly_mul(poly, {(0,) * self.dim: 1, neg: -1})
        return poly

    def __repr__(self) -> str:
        return f"RootDatum({self.name}, dim={self.dim}, |Φ⁺|={len(self._positive)})"


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def u_n(N: int) -> RootDatum:
    """U(N): `e`-basis, `d = N`, roots `e_i − e_j`, Weyl group `S_N`."""
    N = int(N)
    simple = []
    for i in range(N - 1):
        a = [0] * N
        a[i], a[i + 1] = 1, -1
        simple.append(tuple(a))
    return RootDatum(N, simple, list(simple), name=f"U({N})",
                     atom_phase=lambda m: sum(j * m[j] for j in range(N)))


def _cartan_A(r: int) -> list[list[int]]:
    C = [[0] * r for _ in range(r)]
    for i in range(r):
        C[i][i] = 2
        if i + 1 < r:
            C[i][i + 1] = C[i + 1][i] = -1
    return C


def su_n(N: int) -> RootDatum:
    """SU(N): fundamental-weight `ω`-basis, `d = N−1`, simple roots = Cartan rows,
    simple coroots = unit vectors (so `⟨λ, α_i^∨⟩ = λ_i`)."""
    N = int(N)
    r = N - 1
    if r == 0:
        raise ValueError("SU(1) is trivial")
    C = _cartan_A(r)
    simple = [tuple(C[i]) for i in range(r)]
    coroots = [tuple(1 if k == i else 0 for k in range(r)) for i in range(r)]

    return RootDatum(r, simple, coroots, name=f"SU({N})")


def su_2() -> RootDatum:
    """SU(2): `d = 1`, single positive root `α = (2)` ⇒ `v^α = v²`."""
    return su_n(2)


def product_datum(data) -> RootDatum:
    """The product root datum `G_1 × … × G_n` — block-diagonal embedding:
    dim = Σ dims, simple roots/coroots and positive roots block-embedded
    (no cross-block roots), Weyl = the product group, and the
    atom-normalization phase the SUM of the per-block phases (each factor
    keeps its own convention on its slice)."""
    data = list(data)
    dims = [d.dim for d in data]
    D = sum(dims)
    offs = [0]
    for dd in dims:
        offs.append(offs[-1] + dd)

    def _embed(vec, a):
        return tuple((vec[t - offs[a]] if offs[a] <= t < offs[a + 1] else 0)
                     for t in range(D))

    simple, coroots, positive = [], [], []
    for a, d in enumerate(data):
        for r, c in zip(d.simple_roots, d.simple_coroots):
            simple.append(_embed(r, a))
            coroots.append(_embed(c, a))
        for r in d.positive_roots():
            positive.append(_embed(r, a))

    def _phase(m):
        return sum(d.atom_phase(tuple(m[offs[a]:offs[a + 1]]))
                   for a, d in enumerate(data))

    return RootDatum(D, simple, coroots,
                     name=" × ".join(d.name for d in data),
                     positive_roots=positive, atom_phase=_phase,
                     # `_phase` DELEGATES to the components, so canonicity is the
                     # conjunction over them.  Without this the product would claim
                     # `True` unconditionally (any datum carrying its own
                     # `atom_phase` does) and would silently call a component's
                     # odd-height coweight canonical.
                     phase_is_canonical=lambda m: all(
                         d.atom_phase_is_canonical(tuple(m[offs[a]:offs[a + 1]]))
                         for a, d in enumerate(data)))


def torus(d: int) -> RootDatum:
    """U(1)^d: the rootless datum (empty Weyl group), `dim = d`.  The flavour
    datum of an Abelian flavour symmetry."""
    return RootDatum(int(d), [], [], name=f"U(1)^{int(d)}")


# ---------------------------------------------------------------------------
# Beyond type A — the group-general factories
#
# `RootDatum` itself was never type-A: the constructor takes explicit positive
# roots and closes the Weyl group from the simple reflections.  Only the
# *factories* were (`u_n` / `su_n` / `torus` / `product_datum`), so B/C/D/G data
# had to be hand-built in the probes in the source repository — which is where `b2_datum` / `g2_datum`
# / `datum_from_gauge` lived until this promotion.  The bridge below is the
# inverse of `to_gauge_datum`, so B/C/D come from the repo's own validated
# `root_data.GaugeDatum` factories rather than from fresh arithmetic.
# ---------------------------------------------------------------------------

def weyl_vector_rho_sign(positive_roots):
    """The ρ-twist sign exponent, **DERIVED** (2026-07-27, PR + follow-up):

        rho_sign_exp(k)  =  Σ_{α>0} ⟨α, k⟩  =  ⟨Σ⁺, k⟩,   Σ⁺ = Σ_{α>0} α

    the datum's own Weyl vector (`weyl_vector`, doubled) paired with the
    cocharacter.  Coordinate-free, so it transports to any root datum — unlike
    the `rho_sign_exp` staircase default, which is this same formula written in
    the U(N) `e`-basis (there the `i`-th coordinate of `Σ⁺` IS `N−1−2i`)
    and is therefore only right in those coordinates.

    **It is a character of `π₁(G)`, and that is the whole content.**  MEASURED
    on every datum the repo carries: `k ↦ ⟨Σ⁺,k⟩ mod 2` is additive, and
    it kills every simple coroot (`⟨Σ⁺, α_i^∨⟩ = 2`), so it vanishes on
    `Q^∨` and descends to

        X_*(T)/Q^∨  =  π₁(G)  ⟶  {±1}.

    So the sign is the image of the monopole's GNO class in `π₁(G)`, and it is
    nontrivial exactly on data that are not simply connected AND whose `Σ⁺`
    is not a weight: measured trivial for SU(N), Spin(5), Spin(7), Sp(n) (all
    `π₁ = 1`), trivial for U(3) (`π₁ = Z` but `Σ⁺ = (2,0,−2)` is even),
    and nontrivial for SO(5), SO(7) (`π₁ = Z/2`) and U(2), U(4).

    This SUPERSEDES `_pinned_rho_sign`, the constant `+1` that was pinned by
    demanding orthonormality.  That fit was recovering a theorem: `b2_datum` /
    `g2_datum` / `b_n_simply_connected` carry cocharacters in the **coroot**
    lattice, where the character above is identically trivial — so `+1` was
    never a free choice there, which is exactly why the fit worked and why the
    staircase default failed (`⟨L,L⟩ = 𝖖⁴`: right formula, wrong coordinates).

    **Inert on everything currently computed** (measured): the constant and this
    formula differ only on `so_n(5)`/`so_n(7)` at odd-`⟨Σ⁺,m⟩` cocharacters
    — precisely the sectors where `atom_phase` honest-fails — so no result
    changes.  It stops being inert the moment those sectors are opened.

    NOT the same thing as `atom_phase` half-integrality — a correction to the
    first write-up of this (PR), which claimed the sign was "the missing
    half of `atom_phase`".  **U(2) refutes that**: its sign is nontrivial at
    `m = (1,0)` while `u_n(2).atom_phase((1,0)) = 0`, perfectly integral.  The
    honest statement is that both are governed by the SAME character above.
    `atom_phase`'s default `−½⟨Σ⁺,m⟩` is a half-integer exactly when the
    character is nontrivial, but that can be repaired by the freedom to add
    a **linear** (coboundary) term — and such a term exists iff the group has a
    central torus.  Measured: the space of Weyl-invariant linear functionals is
    1-dimensional for U(2)/U(3) (the direction `Σ_j m_j`, which is what
    `u_n`'s historical `(N−1)/2·Σ_j m_j` shift uses) and **0-dimensional** for
    SU(3), SO(5), SO(7), Sp(3).  So U(N) can repair the parity and a semisimple
    datum cannot.

    **No `𝖖^{1/2}` is involved** — **and none is needed**.

    ⚠ **The conclusion this docstring used to draw is RETRACTED.**  It said the
    route was *measured dead* — a probe in the source repository
    showing that no integer atom phase works for the odd-height spinorial line and
    that simulating `𝖖^{1/2}` fails anyway (`I_WRQ = −𝖖⁻¹·I_BPS`) — and concluded
    that *"the odd-root-height cocharacter is simply not an atom on this torus"* and
    that `atom_phase`'s raise was correct behaviour.  **The measurements reproduce;
    the inference does not.**  The torus never needs the phase as a number, only
    through its coboundary `δS̃`, an integer even where `S̃` is a half-integer (because
    `π = ⟨Σ⁺,·⟩ mod 2` is Weyl-invariant *and* additive), and `wrq_torus.cocycle_R`
    restores the honest phase through it.  The `−𝖖⁻¹` was `_psi_monomial_data`'s
    materialised **square root of the measure**, not the algebra.  So odd-`⟨Σ⁺,m⟩`
    cocharacters *are* atoms of this torus, `atom_phase` no longer raises, and that
    experiment is retained as a HISTORICAL record of the defective route.  The
    standard-`Z²` BPS frame (SO(3) = SU(2) with `(M,E) = (2m, e/2)`,
    `src/gn/pure_so3.py`) remains a correct independent presentation — it is
    the oracle the fix was certified against.
    """
    pos = [tuple(a) for a in positive_roots]

    def _sign(k):
        return sum(sum(k[i] * a[i] for i in range(len(k))) for a in pos)
    return _sign


def from_gauge_datum(gd, rho_sign=None, atom_phase=None) -> RootDatum:
    """`RootDatum` from a `root_data.GaugeDatum` — the inverse of
    `to_gauge_datum`, and the way non-type-A data enter the canonical surface.

    `GaugeDatum` is the repo's validated root-datum surface on the **flux
    lattice**: roots are functionals `α(m) = Σ αᵢ mᵢ`, coroots live in flux
    coordinates, and `α(α^∨) = 2` is enforced at construction — exactly
    `RootDatum`'s convention (`shift_pairing` is the same dot product), so this
    is a straight transcription rather than a re-derivation.  Its `su` / `so` /
    `sp` / `u` / `torus` / `product` factories therefore all become available
    here.

    **The global form is part of the datum, and it matters.**  The cocharacter
    lattice is `X_*(T)`, and `X_*(T)/Q^∨ = π₁(G)`: cocharacters in the *coroot*
    lattice mean the **simply connected** form, while `GaugeDatum.so(N)` is
    documented as "the SO form" (adjoint for `B_n`).  These are physically
    different theories — different GNO/line-operator lattices — so they are
    different data here, not two spellings of one.

    **Scope — the odd-`⟨Σ⁺,m⟩` case is CLOSED.**
    `atom_phase`'s default is `−½Σ_{α>0}⟨α,m⟩`, which is a half-integer exactly
    where `⟨Σ⁺,m⟩` is odd — for the SO-form `B_n` on genuine cocharacters
    (`SO(5)` at `m=(1,0)`; `SO(7)` at `m=(1,0,0)` and `(1,1,1)`).  That is **no
    longer an obstruction and nothing honest-fails for it**: the phase is never
    needed as a number, only through its coboundary `δS̃`, which is an integer
    even where `S̃` is not, and `wrq_torus.cocycle_R` restores the honest phase
    through it.  Those charges build, bar-invariantly, with `I = 1 + O(𝖖)` —
    certified at SO(3) against the `PureSO3KAlgebra` BPS oracle and measured
    orthonormal at SO(5)/SO(7).  No `𝖖^{1/2}` and no `(−1)^B` spin subtlety
    appear anywhere; that reading (and the "open item" this docstring used to
    claim) is retracted."""
    simple_roots = [tuple(gd.roots[i]) for i in gd.simple]
    simple_coroots = [tuple(gd.coroots[i]) for i in gd.simple]
    return RootDatum(gd.rank, simple_roots, simple_coroots, name=gd.name,
                     positive_roots=[tuple(r) for r in gd.positive_roots()],
                     atom_phase=atom_phase,
                     rho_sign=rho_sign)


def _gauge_datum_module():
    import importlib
    import os
    import sys
    # The shim only *adds* a search path, so a `root_data` already reachable by
    # bare name resolves untouched — which is how the flat export tree, where
    # there is no `threed/` beside this module, finds it.  Hence the `isdir`
    # guard: without it that layout gets a nonexistent directory inserted at the
    # front of `sys.path`, which is dead weight on every later import.
    _threed = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threed")
    if os.path.isdir(_threed) and _threed not in sys.path:
        sys.path.insert(0, _threed)
    return importlib.import_module("root_data")


def so_n(N: int) -> RootDatum:
    """`SO(N)` in the **SO form** (`GaugeDatum.so`: flux lattice `Z^n` for
    `T = SO(2)^n`, `n = ⌊N/2⌋`) — type `B_n` for odd `N`, `D_n` for even `N`.

    This is the *adjoint*-ish global form, not the spin cover: see
    `from_gauge_datum` on why that distinction is physical, and on its
    odd-`⟨Σ⁺,m⟩` cocharacters — which **build**, the
    half-integral atom phase reaching the cocycle only through its integral
    coboundary.  For the simply connected `B_n` theory use
    `b_n_simply_connected` / `sp_n` (`Spin(5) ≅ Sp(2)`).

    **This datum's coordinates make that lattice INTEGRAL**, which is why it is
    the frame to sweep in.  `so_n(3)` has `⟨Σ⁺,(1,)⟩ = 1`, so `m = 1` is the
    minuscule spinorial monopole — the line SU(2) does not have (`su_2()` has no
    non-trivial minuscule cocharacter: its `m = 1` is height 2).  The same
    lattice written as `global_form.adjoint_lines(su_2())` puts that line at the
    *fractional* `m = 1/2`, so an integer sweep there misses it entirely; see
    that function's coordinates warning, and
    a probe in the source repository for the axioms measured on this frame."""
    return from_gauge_datum(_gauge_datum_module().GaugeDatum.so(int(N)))


def sp_n(n: int) -> RootDatum:
    """`Sp(n)` in the engine's naming = compact `USp(2n)` = type `C_n`, simply
    connected (flux lattice `Z^n` = the `C_n` coroot lattice).

    NOTE the convention: the argument is the **rank**, matching
    `GaugeDatum.sp`, so `sp_n(3)` is the group physics usually writes `Sp(6)`
    and `sp_n(2)` is `USp(4) ≅ Spin(5)`."""
    return from_gauge_datum(_gauge_datum_module().GaugeDatum.sp(int(n)))


def b_n_simply_connected(n: int) -> RootDatum:
    """`Spin(2n+1)` — type `B_n` with cocharacters in the **coroot lattice**
    (hence simply connected), in the fundamental-coweight coordinates where
    `⟨Σ⁺, m⟩` is always even so the integer atom phase applies at every `m`.

    Simple roots are the Cartan-matrix rows (`α₁ … α_{n−1}` long, `α_n` short)
    and simple coroots the unit vectors, exactly as `su_n` does for type A.  At
    `n = 2` this reproduces the `B₂` datum the solver used — which that
    module called "B2=SO(5)", a misnomer: the coroot lattice is the simply
    connected form, so it is `Spin(5) ≅ Sp(2)` (cross-checkable against
    `sp_n(2)`, the same theory in `C₂` coordinates)."""
    n = int(n)
    if n < 2:
        raise ValueError("B_n needs n >= 2")
    # Cartan matrix of B_n: a_ij = <alpha_i, alpha_j^vee>; the short root is
    # alpha_n, so a_{n-1,n} = -2 and a_{n,n-1} = -1.
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        C[i][i] = 2
        if i + 1 < n:
            C[i][i + 1] = -1
            C[i + 1][i] = -1
    C[n - 2][n - 1] = -2
    simple = [tuple(C[i]) for i in range(n)]
    coroots = [tuple(1 if k == i else 0 for k in range(n)) for i in range(n)]
    positive = _positive_roots_from_simple(simple, coroots, n)
    return RootDatum(n, simple, coroots, name=f"Spin({2 * n + 1})",
                     positive_roots=positive)


def g_2() -> RootDatum:
    """`G₂` — cocharacters in the coroot lattice (`G₂` is both simply connected
    and adjoint, so there is only one form).  Cartan `[[2,−1],[−3,2]]`
    (`α₁` short, `α₂` long); `Φ⁺ = {α₁, α₂, α₁+α₂, 2α₁+α₂, 3α₁+α₂, 3α₁+2α₂}`
    in the fundamental-coweight coordinates the `` solver used."""
    simple = [(2, -1), (-3, 2)]
    coroots = [(1, 0), (0, 1)]
    positive = [(2, -1), (-3, 2), (-1, 1), (1, 0), (3, -1), (0, 1)]
    return RootDatum(2, simple, coroots, name="G2",
                     positive_roots=positive)


def _positive_roots_from_simple(simple, coroots, d) -> list:
    """Positive roots of an arbitrary root system, by closing the simple roots
    under the simple reflections and keeping the ones with non-negative
    coefficients in the simple-root basis.

    Type-agnostic (`_positive_roots_type_A` is the fast path for type A only);
    used by the non-type-A factories above."""
    refl = [_simple_reflection(a, c, d) for a, c in zip(simple, coroots)]
    roots = {tuple(a) for a in simple}
    frontier = list(roots)
    while frontier:
        new = []
        for r in frontier:
            for M in refl:
                img = _matvec(M, r)
                if img not in roots:
                    roots.add(img)
                    new.append(img)
        frontier = new
    # Split into ±: keep one of each {r, -r} pair, choosing the one that is a
    # non-negative combination of the simple roots.
    coeffs = _simple_basis_coeffs(simple, roots, d)
    pos = [r for r in sorted(roots)
           if coeffs[r] is not None and all(c >= 0 for c in coeffs[r])]
    return pos


def _simple_basis_coeffs(simple, roots, d) -> dict:
    """Coordinates of each root in the simple-root basis (Gaussian elimination
    over the rationals, then exact-integer check)."""
    from fractions import Fraction
    out = {}
    r = len(simple)
    for root in roots:
        # solve  Σ_j x_j · simple[j] = root
        M = [[Fraction(simple[j][i]) for j in range(r)] + [Fraction(root[i])]
             for i in range(d)]
        piv_cols = []
        row = 0
        for col in range(r):
            sel = next((i for i in range(row, d) if M[i][col] != 0), None)
            if sel is None:
                continue
            M[row], M[sel] = M[sel], M[row]
            pv = M[row][col]
            M[row] = [x / pv for x in M[row]]
            for i in range(d):
                if i != row and M[i][col] != 0:
                    f = M[i][col]
                    M[i] = [a - f * b for a, b in zip(M[i], M[row])]
            piv_cols.append(col)
            row += 1
        if any(all(M[i][c] == 0 for c in range(r)) and M[i][r] != 0
               for i in range(d)):
            out[root] = None                      # inconsistent: not in the span
            continue
        x = [Fraction(0)] * r
        for i, col in enumerate(piv_cols):
            x[col] = M[i][r]
        out[root] = tuple(x)
    return out


# ---------------------------------------------------------------------------
# Flavour-group contract extensions (the design record
# "a class contracting 3d flavour groups — just reductive Lie groups").
# `RootDatum` is promoted to the contract role; the genuinely new pieces are
# the rep-ring accessor, the Levi hook (for the A23 valuation-floor shape),
# group homs/embeddings, and the certified bridge to the self-contained
# `root_data.GaugeDatum` engine sibling.
# ---------------------------------------------------------------------------

def _rep_ring(datum: "RootDatum"):
    """The representation ring of the group as a `ZPlusRing`.

    Torus (no roots) → `AbelianZPlusRing(dim)` (character lattice = Z^dim);
    `SU(N)` (by factory name) → `SUNZPlusRing(N)`.  Other non-abelian data
    honestly raise until their rings are wired (`NotImplementedError`)."""
    import zplus_ring as _zr
    if not datum.positive_roots():
        return _zr.AbelianZPlusRing(rank=datum.dim)
    if datum.name.startswith("SU(") and datum.name.endswith(")"):
        return _zr.SUNZPlusRing(int(datum.name[3:-1]))
    raise NotImplementedError(
        f"rep_ring for {datum.name}: wire the ZPlusRing for this group "
        "(only torus and SU(N) are connected so far)")


def _levi_positive_roots(datum: "RootDatum", flux) -> list:
    """The positive roots of the Levi `L_flux` (centralizer of the magnetic
    cocharacter `flux`): the roots orthogonal to the flux under the pairing
    `⟨α, m⟩ = Σ αᵢ mᵢ`.  This is the hook for the A23 valuation-floor shape —
    in a flux sector the linear floor may involve the Abelian charges of the
    *Levi* of the magnetic charge (its centre grows as roots are broken)."""
    m = tuple(flux)
    return [a for a in datum.positive_roots()
            if sum(x * y for x, y in zip(a, m)) == 0]


RootDatum.rep_ring = _rep_ring
RootDatum.levi_positive_roots = _levi_positive_roots


class RootDatumHom:
    """A homomorphism of flavour groups at the datum level: a Z-linear map of
    **cocharacter lattices** `matrix : X_*(T_source) → X_*(T_target)`
    (columns = images of the source basis).  Weights/fugacities restrict by
    the transpose.  This is the datum-level counterpart of `RingHom` /
    `base_change`, consumed by the 4d promotion (`G ↪ G_f`) and by
    flavour-breaking boundaries (the design record design §3a″)."""

    __slots__ = ("source", "target", "matrix")

    def __init__(self, source: RootDatum, target: RootDatum, matrix):
        self.source, self.target = source, target
        self.matrix = [tuple(row) for row in matrix]   # target.dim × source.dim
        if len(self.matrix) != target.dim or any(
                len(r) != source.dim for r in self.matrix):
            raise ValueError("RootDatumHom: matrix shape must be "
                             "target.dim × source.dim")

    def map_cochar(self, m):
        """Push a source cocharacter (magnetic flux) into the target."""
        m = tuple(m)
        return tuple(sum(r[j] * m[j] for j in range(self.source.dim))
                     for r in self.matrix)

    def restrict_weight(self, w):
        """Pull a target weight (fugacity charge) back to the source
        (the transpose action)."""
        w = tuple(w)
        return tuple(sum(self.matrix[i][j] * w[i]
                         for i in range(self.target.dim))
                     for j in range(self.source.dim))


def to_gauge_datum(datum: "RootDatum"):
    """Bridge to the self-contained engine sibling
    `root_data.GaugeDatum`, by factory dispatch on the datum name
    (torus / U(N) / SU(N)) — the engine package stays independent; agreement (Weyl orders, dominance) is pinned by the test battery."""
    import importlib
    import os
    import sys
    # `isdir`-guarded for the same reason as `_gauge_datum_module` above.
    _threed = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "threed")
    if os.path.isdir(_threed) and _threed not in sys.path:
        sys.path.insert(0, _threed)
    rd = importlib.import_module("root_data")
    name = datum.name
    if not datum.positive_roots():
        return rd.GaugeDatum.torus(datum.dim)
    if name.startswith("U(") and name.endswith(")"):
        return rd.GaugeDatum.u(int(name[2:-1]))
    if name.startswith("SU(") and name.endswith(")"):
        return rd.GaugeDatum.su(int(name[3:-1]))
    raise NotImplementedError(f"to_gauge_datum: no bridge for {name}")


RootDatum.to_gauge_datum = to_gauge_datum
