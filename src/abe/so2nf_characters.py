"""`so2nf_characters` — `Spin(2Nf) = D_Nf` character theory for recognizing the
**enhanced flavour symmetry** of an SU(2)+Nf-type algebra presented only over
its Cartan torus `U(1)^{Nf}`.

The flavour symmetry of SU(2) with `Nf` fundamental hypers is `SO(2Nf)`
(pseudoreal fundamental ⇒ the `2Nf` half-hypers form the `SO(2Nf)` vector),
realized on the index as its simply-connected cover `Spin(2Nf)`.  A
`U(1)^{Nf}`-flavoured presentation (e.g. `su2_nf_over_pure_rgflow`) only
*manifests the Cartan torus* `T = U(1)^{Nf}`; the enhancement is the statement

    R(Spin(2Nf))  ≅  R(T)^{W(D_Nf)}                                      (★)

— the `Spin(2Nf)` characters are exactly the `W(D_Nf)`-invariant `T`-characters.
So an index `I(μ_1,…,μ_{Nf})` (a Laurent polynomial in the torus fugacities)
"is secretly `Spin(2Nf)`-flavoured" iff it is `W(D_Nf)`-invariant, and it then
lifts uniquely to a `Spin(2Nf)`-character content.

This module supplies that recognition in the **orthogonal `e_i` basis**:

* `weyl_orbit` / `is_weyl_invariant` — the `W(D_Nf)` action (coordinate
  permutations + *even* numbers of sign flips `x_i ↦ x_i^{-1}`).
* `weyl_denominator` — `δ = Σ_{w∈W} sign(w)·x^{wρ}` (`ρ = (Nf-1,…,1,0)`).
* `decompose` — un-branch a `W`-invariant Laurent polynomial into irreducible
  `Spin(2Nf)` characters by the Weyl-denominator trick:
  `I·δ = Σ_λ c_λ·N_λ` with `N_λ = Σ_w sign(w)·x^{w(λ+ρ)}` the Weyl numerator,
  so `c_λ = [x^{λ+ρ}](I·δ)` read off at each strictly-dominant `λ+ρ`.
  `c_λ ∈ Z_{≥0}` ⟺ a genuine representation ⟺ the enhancement holds.
* `character` — `χ_λ` as the Laurent polynomial `N_λ/δ`, the inverse lift.

Weights/labels are length-`Nf` tuples of either all-integers (tensor reps) or
all-half-integers (spinor reps); both appear because the SU(2)+Nf matter sits
in the `Spin(2Nf)` spinor while the conserved currents/vector sit in the
tensor reps.  Half-integers are carried as `Fraction`s.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations, product
from typing import Iterable


Weight = tuple  # length-Nf tuple of int | Fraction


# ---------------------------------------------------------------------------
# The Weyl group W(D_Nf): coordinate permutations + even sign changes
# ---------------------------------------------------------------------------


def weyl_elements(nf: int):
    """Yield `(perm, signs, sign_of_w)` for every `w ∈ W(D_Nf)`:
    `perm` a permutation of `range(nf)`, `signs` a length-`nf` tuple of `±1`
    with an **even** number of `-1`s, and `sign_of_w = det(w) = sgn(perm)·∏signs`
    (here `∏signs = +1` always for `D`, so `sign_of_w = sgn(perm)`).

    `W(D_Nf)` has order `2^{Nf-1}·Nf!`."""
    for perm in permutations(range(nf)):
        psign = _perm_sign(perm)
        for signs in product((1, -1), repeat=nf):
            if signs.count(-1) % 2 != 0:        # even sign changes only (D-type)
                continue
            yield perm, signs, psign            # ∏signs = +1, so det = psign


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


def apply_weyl(w, weight: Weight) -> Weight:
    """Apply a Weyl element `w = (perm, signs, _)` to a weight (orthogonal basis):
    permute the coordinates then flip signs.  `(w·λ)_i = signs[i]·λ[perm[i]]`."""
    perm, signs, _ = w
    return tuple(signs[i] * weight[perm[i]] for i in range(len(weight)))


def weyl_orbit(nf: int, weight: Weight) -> set:
    """The `W(D_Nf)` orbit of a weight."""
    return {apply_weyl(w, weight) for w in weyl_elements(nf)}


# ---------------------------------------------------------------------------
# Laurent polynomials in x_1,…,x_{Nf}  (exponent tuple -> coefficient)
# ---------------------------------------------------------------------------


def _lpoly_mul(a: dict, b: dict) -> dict:
    out: dict = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = tuple(x + y for x, y in zip(ea, eb))
            out[e] = out.get(e, 0) + ca * cb
    return {e: c for e, c in out.items() if c != 0}


def is_weyl_invariant(nf: int, poly: dict) -> bool:
    """Whether a Laurent polynomial `{exponent tuple: coeff}` is invariant under
    `W(D_Nf)` — the recognition criterion for the `Spin(2Nf)` enhancement."""
    for w in weyl_elements(nf):
        permuted = {apply_weyl(w, e): c for e, c in poly.items()}
        if {e: c for e, c in permuted.items() if c} != {e: c for e, c in poly.items() if c}:
            return False
    return True


def rho(nf: int) -> Weight:
    """`ρ` for `D_Nf` in the orthogonal basis: `(Nf-1, Nf-2, …, 1, 0)`."""
    return tuple(nf - 1 - i for i in range(nf))


def weyl_denominator(nf: int) -> dict:
    """`δ = Σ_{w∈W(D_Nf)} sign(w)·x^{wρ}` as a Laurent polynomial."""
    r = rho(nf)
    out: dict = {}
    for w in weyl_elements(nf):
        e = apply_weyl(w, r)
        out[e] = out.get(e, 0) + w[2]
    return {e: c for e, c in out.items() if c != 0}


def _is_strictly_dominant(beta: Weight) -> bool:
    """`β` strictly dominant for `D_Nf`: `β_1 > β_2 > … > β_{Nf-1} > |β_Nf|`
    (the regular weights `λ+ρ`)."""
    n = len(beta)
    for i in range(n - 2):
        if not beta[i] > beta[i + 1]:
            return False
    if n >= 2 and not beta[n - 2] > abs(beta[n - 1]):
        return False
    return True


def decompose(nf: int, poly: dict) -> dict:
    """Un-branch a `W(D_Nf)`-invariant Laurent polynomial into irreducible
    `Spin(2Nf)` characters: returns `{highest weight λ: multiplicity c_λ}`.

    Uses `I·δ = Σ_λ c_λ·N_λ`, so `c_λ = [x^{λ+ρ}](I·δ)` at each strictly
    dominant `λ+ρ`.  The caller checks `c_λ ∈ Z_{≥0}` (genuine representation)
    and that `poly` is `W`-invariant (`is_weyl_invariant`); together these
    certify the enhancement and give its character content.
    """
    r = rho(nf)
    prod = _lpoly_mul(poly, weyl_denominator(nf))
    out: dict = {}
    for beta, c in prod.items():
        if c == 0 or not _is_strictly_dominant(beta):
            continue
        lam = tuple(beta[i] - r[i] for i in range(nf))
        out[lam] = c
    return out


def character(nf: int, lam: Weight) -> dict:
    """`χ_λ` (the irreducible `Spin(2Nf)` character at dominant highest weight
    `λ`) as a Laurent polynomial `N_λ/δ`, via Weyl's character formula.  The
    division is exact (both are `W`-anti-invariant)."""
    r = rho(nf)
    lam_rho = tuple(lam[i] + r[i] for i in range(nf))
    numerator: dict = {}
    for w in weyl_elements(nf):
        e = apply_weyl(w, lam_rho)
        numerator[e] = numerator.get(e, 0) + w[2]
    return _lpoly_divide(numerator, weyl_denominator(nf))


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


def dim(nf: int, lam: Weight) -> int:
    """Dimension of the irrep — `χ_λ` evaluated at the identity (sum of weight
    multiplicities)."""
    return sum(character(nf, lam).values())


# ---------------------------------------------------------------------------
# The recognition checker
# ---------------------------------------------------------------------------


def verify_flavour_enhancement(nf: int, index_poly: dict) -> tuple[bool, dict | None, bool]:
    """Recognize whether a `U(1)^{Nf}`-Cartan index `index_poly`
    (`{orthogonal-`e_i`-basis exponent tuple: coeff}`) carries the **enhanced
    `Spin(2Nf)` flavour symmetry**.

    By (★) the index is a `Spin(2Nf)` class function — i.e. *lifts to*
    `R(Spin(2Nf))`, manifesting the enhanced symmetry — **iff it is
    `W(D_Nf)`-invariant**.  That is the recognition.  A *separate* property is
    whether the lift is a *genuine* (non-virtual) representation: `c_λ ∈ Z_{≥0}`.
    A vacuum index (a plain partition function) is genuine; a signed/graded
    insertion (e.g. a Wilson line `Tr(W_n)`) is generically a *virtual*
    character (differences of irreps) — still fully `Spin(2Nf)`-covariant.

    Returns `(enhanced, content, genuine)`:

    * `enhanced` — `W(D_Nf)`-invariant (the symmetry is present, the index
      lifts to `R(Spin(2Nf))`).
    * `content`  — `{highest weight λ: c_λ}`, the lift's character content
      (virtual if some `c_λ < 0`); `None` when not `W`-invariant.
    * `genuine`  — `enhanced` and every `c_λ ∈ Z_{≥0}` (a true representation).
    """
    if not is_weyl_invariant(nf, index_poly):
        return False, None, False
    dec = decompose(nf, index_poly)
    genuine = all(isinstance(c, int) and c >= 0 for c in dec.values())
    return True, dec, genuine


def reconstruct(nf: int, content: dict) -> dict:
    """`Σ_λ c_λ·χ_λ` as a Laurent polynomial — the inverse of `decompose`,
    for round-trip checks."""
    out: dict = {}
    for lam, c in content.items():
        for e, m in character(nf, lam).items():
            out[e] = out.get(e, 0) + c * m
    return {e: m for e, m in out.items() if m != 0}


# ---------------------------------------------------------------------------
# R(Spin(2Nf)) as a ZPlusRing — the coefficient ring for the enhanced flavour
# ---------------------------------------------------------------------------


def is_dominant(lam: Weight) -> bool:
    """Whether `λ` is a `D_Nf`-dominant highest weight:
    `λ_1 ≥ λ_2 ≥ … ≥ λ_{Nf-1} ≥ |λ_Nf|`, all-integer or all-half-integer."""
    n = len(lam)
    if n == 0:
        return True
    fracs = [(2 * x) % 2 != 0 for x in lam]
    if any(fracs) and not all(fracs):
        return False
    for i in range(n - 2):
        if not lam[i] >= lam[i + 1]:
            return False
    if n >= 2 and not lam[n - 2] >= abs(lam[n - 1]):
        return False
    # Dominance fully checked above; the last coordinate λ_Nf may legitimately
    # be negative (it distinguishes the two spinor chiralities of D_Nf), so
    # there is no further sign constraint here.
    return True


from zplus_ring import ZPlusRing, RElement


class SO2NfZPlusRing(ZPlusRing):
    """`R(Spin(2Nf)) = D_Nf` representation ring as a `ZPlusRing`.

    Basis = `D_Nf`-dominant highest weights in the orthogonal `e_i` basis
    (`λ_1 ≥ … ≥ λ_{Nf-1} ≥ |λ_Nf|`, all-int [tensor] or all-half-int [spinor]).
    `one_basis = (0,…,0)`.  **Clebsch–Gordan** `multiply_basis` is computed
    exactly for any `Nf` as `decompose(χ_λ · χ_μ)` (character product then
    Weyl-denominator un-branch) — non-negative by rep theory.  `star_basis`
    is rep duality via the longest Weyl element `w_0` (self-dual for even `Nf`;
    flips the last coordinate — swapping the two spinor chiralities — for odd
    `Nf`).  `basis_element`/`character` give the torus character (the embedding
    `R(Spin(2Nf)) ↪ R(T)^{W(D_Nf)}`).

    This is the coefficient ring for the `Spin(2Nf)`-enhanced flavour wrapper."""

    def __init__(self, nf: int):
        if nf < 1:
            raise ValueError("nf must be >= 1")
        self.nf = nf
        self._mul_cache: dict = {}

    def _validate(self, b):
        if not (isinstance(b, tuple) and len(b) == self.nf and is_dominant(b)):
            raise ValueError(f"SO2NfZPlusRing({self.nf}) basis is a D_{self.nf}-"
                             f"dominant weight; got {b!r}")

    def one_basis(self):
        return (0,) * self.nf

    def multiply_basis(self, b1, b2):
        self._validate(b1)
        self._validate(b2)
        key = (b1, b2) if b1 <= b2 else (b2, b1)
        if key in self._mul_cache:
            return dict(self._mul_cache[key])
        prod = _lpoly_mul(character(self.nf, b1), character(self.nf, b2))
        out = decompose(self.nf, prod)
        out = {k: v for k, v in out.items() if v != 0}
        for v in out.values():
            if v < 0:
                raise RuntimeError(f"SO2Nf CG negative multiplicity: {b1}·{b2}={out}")
        self._mul_cache[key] = dict(out)
        return out

    def star_basis(self, b):
        self._validate(b)
        if self.nf % 2 == 0:
            return b                                   # w_0 = -1 ⇒ self-dual
        return b[:-1] + (-b[-1],)                       # odd Nf: flip spinor chirality

    def dim(self, b) -> int:
        # dimension = χ_b at the identity = sum of torus-weight multiplicities
        return sum(self.character(b).values())

    def one_dim_rep_rank(self) -> int:
        return 0   # Spin(2Nf) is semisimple: only the trivial 1-dim rep

    def embed_one_dim_rep(self, f):
        if tuple(f) != ():
            raise ValueError(f"SO2NfZPlusRing({self.nf}): Λ has rank 0; got {f!r}")
        return self.one_basis()

    def character(self, b) -> dict:
        """The torus character of irrep `b` as a Laurent polynomial in the
        orthogonal fugacities (`{exponent: coeff}`)."""
        self._validate(b)
        return character(self.nf, b)

    def __eq__(self, other):
        return isinstance(other, SO2NfZPlusRing) and other.nf == self.nf

    def __hash__(self):
        return hash(("SO2NfZPlusRing", self.nf))

    def __repr__(self):
        return f"SO2NfZPlusRing(nf={self.nf})  # R(Spin({2*self.nf}))"


# ---------------------------------------------------------------------------
# Spin(8) triality — the S_3 outer automorphism of D_4
# ---------------------------------------------------------------------------
#
# D_4 is the unique simple Lie algebra with an order-3 diagram automorphism:
# the three length-8 fundamentals — vector `8_v = (1,0,0,0)`, spinor
# `8_s = (½,½,½,½)`, cospinor `8_c = (½,½,½,-½)` — are permuted by `S_3`, while
# the adjoint `28 = (1,1,0,0)` is fixed.  Realized on the orthogonal weight
# lattice by two order-2 orthogonal generators (`s` swaps `8_s↔8_c`, `t` swaps
# `8_v↔8_s`); `⟨s,t⟩ = S_3`.  `triality_act` carries a dominant weight to its
# triality image (a diagram automorphism preserves the dominant chamber).


def _matvec(M, v: Weight) -> Weight:
    return tuple(sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M)))


def triality_d4_generators() -> tuple[list, list]:
    """The two order-2 generators `(s, t)` of the `D_4` triality `S_3`
    (orthogonal `4×4` matrices over `Fraction`).  `s` swaps `8_s↔8_c` (fixes
    `8_v`); `t` swaps `8_v↔8_s` (fixes `8_c`)."""
    h = Fraction(1, 2)
    s = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
    t = [[h, h, h, h], [h, h, -h, -h], [h, -h, h, -h], [h, -h, -h, h]]
    return s, t


def triality_act(g, lam: Weight) -> Weight:
    """Image of a `D_4`-dominant weight `λ` under a triality element `g` (a
    matrix or a word in the generators applied left-to-right), folded back into
    the dominant chamber by `W(D_4)`."""
    w = _matvec(g, lam)
    for ww in weyl_elements(4):
        cand = apply_weyl(ww, w)
        if is_dominant(cand):
            return cand
    raise ValueError(f"triality image of {lam} not dominant under any Weyl elt")


def is_triality_automorphism(ring: "SO2NfZPlusRing", g, weights) -> bool:
    """Whether the triality element `g` is a **ring automorphism** of
    `R(Spin(8))` on `weights`: `χ_{g·λ}·χ_{g·μ} = g·(χ_λ·χ_μ)` for all
    `λ, μ` — i.e. Clebsch–Gordan is triality-covariant."""
    for lam in weights:
        for mu in weights:
            lhs = ring.multiply_basis(triality_act(g, lam), triality_act(g, mu))
            rhs = {triality_act(g, nu): c
                   for nu, c in ring.multiply_basis(lam, mu).items()}
            if lhs != rhs:
                return False
    return True
