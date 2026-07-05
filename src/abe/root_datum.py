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

Coordinate realization (ruling, this session)
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

from itertools import permutations
from typing import Sequence


Vec = tuple        # a weight / root / cocharacter: tuple[int, ...]
Mat = tuple        # a d×d integer matrix: tuple[tuple[int, ...], ...]


# ---------------------------------------------------------------------------
# small integer linear-algebra helpers (tuples → hashable, no numpy dep)
# ---------------------------------------------------------------------------
def _identity(d: int) -> Mat:
    return tuple(tuple(1 if i == j else 0 for j in range(d)) for i in range(d))


def _matvec(M: Mat, x: Vec) -> Vec:
    d = len(M)
    return tuple(sum(M[i][j] * x[j] for j in range(len(x))) for i in range(d))


def _matmul(A: Mat, B: Mat) -> Mat:
    n, k, m = len(A), len(B), len(B[0])
    return tuple(
        tuple(sum(A[i][l] * B[l][j] for l in range(k)) for j in range(m))
        for i in range(n)
    )


def _det(M: Mat) -> int:
    """Integer determinant by fraction-free (Bareiss-free, small-d) expansion.

    Weyl matrices are ±1 orthogonal-ish; d ≤ ~6, so plain Laplace is fine."""
    d = len(M)
    if d == 1:
        return M[0][0]
    total = 0
    for j in range(d):
        minor = tuple(tuple(M[i][c] for c in range(d) if c != j) for i in range(1, d))
        total += ((-1) ** j) * M[0][j] * _det(minor)
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
                 "_atom_phase", "_rho_sign")

    def __init__(self, dim: int, simple_roots, simple_coroots, name: str,
                 positive_roots=None, pairing: Mat | None = None,
                 atom_phase=None, rho_sign=None):
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
        if alpha in self._pos_set:
            return alpha, False
        neg = tuple(-x for x in alpha)
        if neg in self._pos_set:
            return neg, True
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

    def act(self, w: int, x) -> Vec:
        """`w · x` — applies to weights and roots alike (same matrices)."""
        return _matvec(self.weyl[w][0], tuple(x))

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
        difference-operator dressing appear on any surface; user ruling
        2026-07-02.)

        **Bar-honesty constraint** (D10, measured 2026-07-01): the bar-relevant
        content of `S` must be `−⟨ρ_weyl, m_dom⟩ = −½ Σ_{α>0} ⟨α, m⟩` up to
        terms *linear* in `m` (linear pieces are coboundaries that cancel in
        the cocycle).  Because the transport copies `S(m_dom)` along the
        orbit, `S` is Weyl-invariant — generically NON-linear — and a phase
        violating the constraint makes `bar` fail antimultiplicativity on
        mixed-chamber products: the su_2 datum with `S ≡ 0` had
        `bar(H·H) ≠ H·H` while every residual of `H` was palindromic.

        Default: the root-system form `−½ Σ_{α>0} ⟨α, m⟩` (an integer:
        `Σ_{α>0}⟨α,m⟩ = ⟨2ρ, m⟩` is even on the cocharacter lattice).  The
        `u_n` factory overrides with the historical U(N) `Σ_j j·m_j` — equal to
        the default plus the linear central term `(N−1)/2·Σ_j m_j` at dominant
        `m`, i.e. honest, and kept verbatim for continuity with the certified
        `URQTorus` normalization."""
        if self._atom_phase is not None:
            return self._atom_phase(m)
        tot = sum(self.shift_pairing(m, a) for a in self._positive)
        if tot % 2:
            raise NotImplementedError(
                f"atom_phase: odd root-height {tot} at {m} in {self.name}")
        return -(tot // 2)

    def rho_sign_exp(self, k) -> int:
        """The sign exponent of the ρ-twist block at magnetic `k` (the engine
        reads `(−1)^{rho_sign_exp(k)}`).  Default: the coordinate staircase
        `Σ_t (dim−1−2t)·k_t` (validated == the U(N) `_rho_block_data`).
        Coordinate-convention-dependent — a `product_datum` sums the
        per-block values (the global staircase differs by an odd offset
        parity), and non-coordinate data may override."""
        if self._rho_sign is not None:
            return self._rho_sign(k)
        return sum((self.dim - 1 - 2 * t) * k[t] for t in range(self.dim))

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

    def _rho_sign(k):
        """The ρ-twist sign exponent for pure SU(N) — the **U(N) staircase of the
        corresponding trace-zero cochar** (NOT the ω-basis coordinate staircase,
        which is coordinate-dependent and wrong for `r ≥ 2`).  A magnetic
        coroot-basis cochar `k = (c₁,…,c_{N−1})` lifts to the u_n cochar
        `mₜ = c_{t+1} − cₜ` (`c₀ = c_N = 0`); the sign is `Σₜ (N−1−2t)·mₜ =
        −2·Σₜ t·mₜ`, **always even** (`Σ mₜ = 0`) ⇒ sign `+1`, restoring
        consistency with the U(N) ρ (the pure-SU(N) monopole is a trace-zero U(N)
        monopole).  Reduces to the already-correct value at `r = 1` (su_2)."""
        c = (0,) + tuple(k) + (0,)
        m = [c[t + 1] - c[t] for t in range(N)]
        return sum((N - 1 - 2 * t) * m[t] for t in range(N))

    return RootDatum(r, simple, coroots, name=f"SU({N})", rho_sign=_rho_sign)


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

    def _rho_sign(k):
        return sum(d.rho_sign_exp(tuple(k[offs[a]:offs[a + 1]]))
                   for a, d in enumerate(data))

    return RootDatum(D, simple, coroots,
                     name=" × ".join(d.name for d in data),
                     positive_roots=positive, atom_phase=_phase,
                     rho_sign=_rho_sign)
