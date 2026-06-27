"""`sun_characters` — `SU(M)` / `U(M)` (type `A_{M-1}`) character theory for
presenting a `U(1)^M`-Cartan-flavoured algebra over its **correct flavour
ring** `R(SU(M))`.

The flavour symmetry of a `U(N)` gauge theory with `M = N_f` fundamental
hypers is `SU(N_f)` (per node `SU(M_i)` for a quiver with `M_i` fundamentals
at the i-th node) — the fundamental is *complex* (no `Spin(2N_f)`
enhancement; Plan 22 T5), and the diagonal `U(1) ⊂ U(N_f)` sits inside the
gauge centre, so it is a *formal* level bookkeeping, not flavour.  A
Cartan-flavoured presentation (`add_flavour(AbelianZPlusRing(M))`, e.g.
`un_nf_over_pure_rgflow` / `quiver_over_pure`) only manifests the torus
`T = U(1)^M`; the statement that the content is genuinely `SU(M)`-flavoured
is

    R(U(M))  ≅  R(T)^{S_M}                                              (★)

— the `U(M)` characters (Schur Laurent polynomials `s_λ`, `λ ∈ Z^M` weakly
decreasing; negative entries = `det`-twists) are exactly the
`S_M`-invariant `T`-characters, and `U(M)`-content splits as
`(SU(M) irrep, diagonal level)` via `λ ↦ (λ − λ_M·𝟙, Σλ_i)`.

This is the type-`A` sibling of `so2nf_characters` (the `D_Nf` machinery for
the pseudoreal SU(2)+Nf case), with the same surface:

* `weyl_elements` / `is_weyl_invariant` — the `S_M` Weyl action.
* `weyl_denominator` — the Vandermonde `δ = Σ_{σ∈S_M} sgn(σ)·x^{σρ}`,
  `ρ = (M-1,…,1,0)`.
* `decompose` — un-branch an `S_M`-invariant Laurent polynomial into
  irreducible `U(M)` characters by the Weyl-denominator trick
  (`c_λ = [x^{λ+ρ}](I·δ)` at strictly decreasing `λ+ρ`).
* `character` — the Schur Laurent polynomial `s_λ = N_λ/δ` (Weyl/bialternant
  formula), the inverse lift.
* `verify_flavour_assembly` — the recognition `(covariant, content, genuine)`.
* `SUNZPlusRing(M)` — the **general `R(SU(M))` Z₊-ring** (the repo's
  `SU2ZPlusRing` / `SU3ZPlusRing` are its `M = 2, 3` cases).
* `SUNFlavourRing` — `⊗_i R(SU(M_i)) ⊗ R(U(1)^r)`: the coefficient ring of an
  SU-enhanced presentation, the abelian factor carrying the *formal* residue
  (per-group diagonal levels + link slots).

Weights are length-`M` tuples of plain ints (type `A` has no spinors).
"""

from __future__ import annotations

from itertools import permutations


Weight = tuple  # length-M tuple of int


# ---------------------------------------------------------------------------
# The Weyl group S_M: coordinate permutations
# ---------------------------------------------------------------------------


def _perm_sign(perm: tuple[int, ...]) -> int:
    """Sign of a permutation given as an image tuple."""
    seen = [False] * len(perm)
    sign = 1
    for i in range(len(perm)):
        if seen[i]:
            continue
        j, length = i, 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        if length % 2 == 0:
            sign = -sign
    return sign


def weyl_elements(m: int):
    """Yield `(perm, sign)` for every `σ ∈ S_M` (`perm` an image tuple);
    `|S_M| = M!`."""
    for perm in permutations(range(m)):
        yield perm, _perm_sign(perm)


def apply_weyl(w, weight: Weight) -> Weight:
    """`(σ·λ)_i = λ_{σ(i)}` for a Weyl element `w = (perm, sign)`."""
    perm, _ = w
    return tuple(weight[perm[i]] for i in range(len(weight)))


def weyl_orbit(m: int, weight: Weight) -> set:
    """The `S_M` orbit of a weight."""
    return {apply_weyl(w, weight) for w in weyl_elements(m)}


# ---------------------------------------------------------------------------
# Laurent polynomials in x_1,…,x_M  (exponent tuple -> coefficient)
# ---------------------------------------------------------------------------


def _lpoly_mul(a: dict, b: dict) -> dict:
    out: dict = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = tuple(x + y for x, y in zip(ea, eb))
            out[e] = out.get(e, 0) + ca * cb
    return {e: c for e, c in out.items() if c != 0}


def is_weyl_invariant(m: int, poly: dict) -> bool:
    """Whether a Laurent polynomial `{exponent tuple: coeff}` is `S_M`-
    invariant — the recognition criterion for `U(M)`/`SU(M)` covariance."""
    clean = {e: c for e, c in poly.items() if c}
    for w in weyl_elements(m):
        if {apply_weyl(w, e): c for e, c in clean.items()} != clean:
            return False
    return True


def rho(m: int) -> Weight:
    """`ρ` for `A_{M-1}` in the U(M) coordinates: `(M-1, M-2, …, 1, 0)`."""
    return tuple(m - 1 - i for i in range(m))


def weyl_denominator(m: int) -> dict:
    """`δ = Σ_{σ∈S_M} sgn(σ)·x^{σρ}` — the Vandermonde determinant
    `∏_{i<j}(x_i − x_j)` as a Laurent polynomial."""
    r = rho(m)
    out: dict = {}
    for w in weyl_elements(m):
        e = apply_weyl(w, r)
        out[e] = out.get(e, 0) + w[1]
    return {e: c for e, c in out.items() if c != 0}


def _is_strictly_decreasing(beta: Weight) -> bool:
    return all(beta[i] > beta[i + 1] for i in range(len(beta) - 1))


def is_dominant(lam: Weight) -> bool:
    """`λ` a `U(M)`-dominant weight: weakly decreasing integers (negative
    entries allowed — `det`-twisted reps)."""
    return (all(isinstance(x, int) for x in lam)
            and all(lam[i] >= lam[i + 1] for i in range(len(lam) - 1)))


def decompose(m: int, poly: dict) -> dict:
    """Un-branch an `S_M`-invariant Laurent polynomial into irreducible
    `U(M)` characters: returns `{dominant λ ∈ Z^M: multiplicity c_λ}`.

    Uses `I·δ = Σ_λ c_λ·N_λ`, so `c_λ = [x^{λ+ρ}](I·δ)` at each strictly
    decreasing `λ+ρ`.  The caller checks `c_λ ∈ Z_{≥0}` (genuine
    representation) and `S_M`-invariance (`is_weyl_invariant`); together
    these certify the assembly and give its character content."""
    r = rho(m)
    prod = _lpoly_mul(poly, weyl_denominator(m))
    out: dict = {}
    for beta, c in prod.items():
        if c == 0 or not _is_strictly_decreasing(beta):
            continue
        lam = tuple(beta[i] - r[i] for i in range(m))
        out[lam] = c
    return out


def character(m: int, lam: Weight) -> dict:
    """The Schur Laurent polynomial `s_λ(x_1,…,x_M)` (the irreducible `U(M)`
    character at dominant `λ ∈ Z^M`) as `{exponent: coeff}`, via the Weyl /
    bialternant formula `N_λ/δ`.  The division is exact."""
    if not is_dominant(lam) or len(lam) != m:
        raise ValueError(f"need a U({m})-dominant length-{m} weight; got {lam!r}")
    r = rho(m)
    lam_rho = tuple(lam[i] + r[i] for i in range(m))
    numerator: dict = {}
    for w in weyl_elements(m):
        e = apply_weyl(w, lam_rho)
        numerator[e] = numerator.get(e, 0) + w[1]
    return _lpoly_divide(numerator, weyl_denominator(m))


def _lpoly_divide(num: dict, den: dict) -> dict:
    """Exact division of Laurent polynomials when it is exact (here `N_λ/δ`).
    Greedy: repeatedly cancel the lexicographically-highest term of `num`
    against the highest term of `den`."""
    num = {e: c for e, c in num.items() if c != 0}
    den = {e: c for e, c in den.items() if c != 0}
    den_top = max(den)
    den_top_c = den[den_top]
    out: dict = {}
    while num:
        top = max(num)
        e = tuple(top[i] - den_top[i] for i in range(len(top)))
        c = num[top]
        if c % den_top_c != 0:
            raise ValueError("non-exact Laurent division (not a character)")
        c //= den_top_c
        out[e] = out.get(e, 0) + c
        for de, dc in den.items():
            te = tuple(e[i] + de[i] for i in range(len(e)))
            num[te] = num.get(te, 0) - c * dc
            if num[te] == 0:
                del num[te]
    return {e: c for e, c in out.items() if c != 0}


def dim(m: int, lam: Weight) -> int:
    """Dimension of the irrep — `s_λ` evaluated at the identity."""
    return sum(character(m, lam).values())


def su_normalize(lam: Weight) -> tuple:
    """`U(M)` dominant weight → the `SU(M)` irrep label: subtract the
    diagonal (`λ − λ_M·𝟙`) and trim trailing zeros — a partition with
    `< M` parts."""
    base = lam[-1]
    shifted = tuple(x - base for x in lam)
    while shifted and shifted[-1] == 0:
        shifted = shifted[:-1]
    return shifted


def split_su_level(lam: Weight) -> tuple:
    """`U(M)` dominant weight → `(SU(M) partition, diagonal level Σλ_i)` —
    the `U(M) → SU(M)×U(1)` split.  Faithful: at fixed level the map is a
    bijection (`λ − λ' = k·𝟙` with `Σλ = Σλ'` forces `k = 0`)."""
    return su_normalize(lam), sum(lam)


# ---------------------------------------------------------------------------
# The recognition checker
# ---------------------------------------------------------------------------


def verify_flavour_assembly(m: int, index_poly: dict) -> tuple[bool, dict | None, bool]:
    """Recognize whether a `U(1)^M`-Cartan index `index_poly`
    (`{exponent tuple: coeff}`) assembles into **`U(M)`/`SU(M)` characters**.

    By (★) the index lifts to `R(U(M))` **iff it is `S_M`-invariant** — that
    is the recognition.  A *separate* property is whether the lift is a
    *genuine* (non-virtual) representation: `c_λ ∈ Z_{≥0}` (an index with
    gauge-current cancellations, e.g. the vacuum `χ_adj − χ_0` at `q²`, is a
    perfectly covariant *virtual* character).

    Returns `(covariant, content, genuine)` with `content = {dominant
    λ ∈ Z^M: c_λ}` (`None` when not invariant)."""
    if not is_weyl_invariant(m, index_poly):
        return False, None, False
    dec = decompose(m, index_poly)
    genuine = all(isinstance(c, int) and c >= 0 for c in dec.values())
    return True, dec, genuine


def reconstruct(m: int, content: dict) -> dict:
    """`Σ_λ c_λ·s_λ` as a Laurent polynomial — the inverse of `decompose`,
    for round-trip checks."""
    out: dict = {}
    for lam, c in content.items():
        for e, mult in character(m, lam).items():
            out[e] = out.get(e, 0) + c * mult
    return {e: c for e, c in out.items() if c != 0}


# ---------------------------------------------------------------------------
# R(SU(M)) as a ZPlusRing — the correct flavour ring for U(N)+N_f
# ---------------------------------------------------------------------------


from zplus_ring import ZPlusRing


# ---------------------------------------------------------------------------
# The ring itself lives in `zplus_ring.SUNZPlusRing` (the ZPlusRing family
# home) — merged 2026-06-12 (user ruling) from this module's class and the
# Plan 30 twin: ONE class, the union surface (`reduce`/`character`/`dim`/
# `_validate`/`m`), Kostka-DP LR engine, cross-certified against this
# module's Weyl-denominator machinery (`character`/`decompose`).
# Re-exported here for back-compat.
# ---------------------------------------------------------------------------
from zplus_ring import SUNZPlusRing            # noqa: F401  (re-export)


class SUNFlavourRing(ZPlusRing):
    """`⊗_i R(SU(M_i))  ⊗  R(U(1)^r)` — the coefficient ring of an
    SU-enhanced flavour presentation.

    The `SU(M_i)` factors are the **genuine flavour rings** (one per group
    of `M_i` fundamental flavours; user ruling 2026-06-11: `SU(N_f)` for
    `U(N)+N_f`, `SU(M_i)` per quiver node).  The rank-`r` abelian factor is
    the **formal residue** — for the standard layout (see
    `sun_flavour_enhancement`) the per-group diagonal levels followed by any
    untouched abelian slots (quiver link μ's): gauge-centre bookkeeping
    carried for faithfulness, *not* flavour.

    Basis = `(su_parts, w)` with `su_parts` a tuple of per-group partitions
    (basis elements of the respective `SUNZPlusRing`) and `w ∈ Z^r`.
    Fusion = per-group Littlewood–Richardson × additive weights; `⋆` =
    per-group duality × negation.
    """

    def __init__(self, Ms, abelian_rank: int):
        self.Ms = tuple(int(M) for M in Ms)
        if any(M < 1 for M in self.Ms):
            raise ValueError(f"bad SU factor ranks {Ms}")
        self.abelian_rank = int(abelian_rank)
        if self.abelian_rank < 0:
            raise ValueError(f"bad abelian rank {abelian_rank}")
        self.factors = tuple(SUNZPlusRing(M) for M in self.Ms)

    def _validate(self, b):
        ok = (isinstance(b, tuple) and len(b) == 2
              and isinstance(b[0], tuple) and len(b[0]) == len(self.Ms)
              and isinstance(b[1], tuple) and len(b[1]) == self.abelian_rank
              and all(isinstance(x, int) for x in b[1]))
        if not ok:
            raise ValueError(f"{self!r} basis is (su_parts, w); got {b!r}")
        for R, p in zip(self.factors, b[0]):
            R._validate(p)

    def one_basis(self):
        return (((),) * len(self.Ms), (0,) * self.abelian_rank)

    def multiply_basis(self, b1, b2):
        self._validate(b1)
        self._validate(b2)
        (p1, w1), (p2, w2) = b1, b2
        w = tuple(a + b for a, b in zip(w1, w2))
        acc = {(): 1}
        for R, a, b in zip(self.factors, p1, p2):
            fus = R.multiply_basis(a, b)
            nxt: dict = {}
            for parts, c0 in acc.items():
                for p, c in fus.items():
                    nxt[parts + (p,)] = nxt.get(parts + (p,), 0) + c0 * c
            acc = nxt
        return {(parts, w): c for parts, c in acc.items() if c != 0}

    def star_basis(self, b):
        self._validate(b)
        parts, w = b
        return (tuple(R.star_basis(p) for R, p in zip(self.factors, parts)),
                tuple(-x for x in w))

    def dim(self, b) -> int:
        self._validate(b)
        parts, _w = b
        out = 1
        for R, p in zip(self.factors, parts):
            out *= R.dim(p)        # the U(1)^r abelian charges are 1-dimensional
        return out

    def one_dim_rep_rank(self) -> int:
        return self.abelian_rank   # SU(M_i) semisimple; Λ = the U(1)^r factor

    def embed_one_dim_rep(self, f):
        if len(f) != self.abelian_rank:
            raise ValueError(f"{self!r}: Λ has rank {self.abelian_rank}; got {f!r}")
        # trivial SU parts ⊗ the abelian weight f
        return (tuple(R.one_basis() for R in self.factors), tuple(f))

    def __eq__(self, other):
        return (isinstance(other, SUNFlavourRing)
                and other.Ms == self.Ms
                and other.abelian_rank == self.abelian_rank)

    def __hash__(self):
        return hash(("SUNFlavourRing", self.Ms, self.abelian_rank))

    def __repr__(self):
        su = " ⊗ ".join(f"R(SU({M}))" for M in self.Ms) or "Z"
        return f"SUNFlavourRing({su} ⊗ R(U(1)^{self.abelian_rank}))"
