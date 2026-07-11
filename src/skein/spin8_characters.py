"""`Spin8ZPlusRing` — the representation ring R(Spin(8)) = R(D_4), the flavour
ring of SU(2)+N_f=4, with the **triality** S_3 outer automorphism manifest.

D_4 in the orthonormal e-basis (e_1..e_4), Bourbaki simple roots
    α_1 = e_1-e_2,  α_2 = e_2-e_3,  α_3 = e_3-e_4,  α_4 = e_3+e_4,
so α_2 is the central (trivalent) node and **triality permutes {α_1, α_3, α_4}**
(equivalently the three 8-dim reps 8v=ω_1, 8s=ω_3, 8c=ω_4; the adjoint 28=ω_2 is
triality-fixed).  Fundamental weights
    ω_1 = e_1,                 ω_2 = e_1+e_2,
    ω_3 = (e_1+e_2+e_3-e_4)/2,  ω_4 = (e_1+e_2+e_3+e_4)/2.
Irreps are labelled by Dynkin labels (a,b,c,d) ∈ ℕ⁴ (highest weight
Λ = a ω_1 + b ω_2 + c ω_3 + d ω_4).

This module is being built bottom-up; step 1 is the root datum + the Weyl
dimension formula (cross-checked against known irrep dims).  `_irrep_weights`,
the tensor product, `from_abelian`, and the triality map follow.

Work in **doubled** e-coordinates (2·e_i, all integers) so spinor weights
(half-integers) stay integral.

Relationship to `so2nf_characters.SO2NfZPlusRing(4)` (the repo's generic
R(Spin(2N)), orthogonal-coords): same ring, **cross-validated** (dims + tensor,
`tests/test_spin8_characters.py`).  This module is the **Dynkin-basis,
triality-manifest** presentation — triality is a permutation of the three labels
(a,c,d), whereas in orthogonal coords it is a nontrivial lattice rotation — which
is what the SL(2,Z)→triality structure of SU(2)+N_f=4 wants.  Self-contained
(no spine import) so it can ship with the export.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import product as _iproduct


# --- D_4 root datum, in 2·e coordinates (integer 4-tuples) ------------------

def _vadd(u, v):
    return tuple(a + b for a, b in zip(u, v))


def _vsub(u, v):
    return tuple(a - b for a, b in zip(u, v))


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


# Simple roots in e-basis (×1, integer here).
_SIMPLE = (
    (1, -1, 0, 0),    # α_1
    (0, 1, -1, 0),    # α_2  (central node)
    (0, 0, 1, -1),    # α_3
    (0, 0, 1, 1),     # α_4
)

# Positive roots: e_i ± e_j, i<j  (12 of them).
_POS_ROOTS = []
for i in range(4):
    for j in range(i + 1, 4):
        ei = tuple(1 if k == i else 0 for k in range(4))
        ej = tuple(1 if k == j else 0 for k in range(4))
        _POS_ROOTS.append(_vsub(ei, ej))   # e_i - e_j
        _POS_ROOTS.append(_vadd(ei, ej))   # e_i + e_j

# Weyl vector ρ = sum of fundamental weights = (3,2,1,0) in e-basis (D_4).
_RHO = (3, 2, 1, 0)

# Fundamental weights in e-basis (ω_3, ω_4 are half-integer).
_OMEGA = (
    (Fraction(1), Fraction(0), Fraction(0), Fraction(0)),                    # ω_1  8v
    (Fraction(1), Fraction(1), Fraction(0), Fraction(0)),                    # ω_2  28
    (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(-1, 2)),      # ω_3  8s
    (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),       # ω_4  8c
)


def highest_weight(dynkin) -> tuple:
    """Dynkin labels (a,b,c,d) -> highest weight in the e-basis (Fractions)."""
    a, b, c, d = dynkin
    coeffs = (a, b, c, d)
    return tuple(sum(coeffs[k] * _OMEGA[k][i] for k in range(4)) for i in range(4))


def dim(dynkin) -> int:
    """Weyl dimension formula: dim = ∏_{α>0} ⟨Λ+ρ, α⟩ / ⟨ρ, α⟩."""
    lam = highest_weight(dynkin)
    lam_rho = tuple(lam[i] + _RHO[i] for i in range(4))
    num = Fraction(1)
    den = Fraction(1)
    for a in _POS_ROOTS:
        num *= _dot(lam_rho, a)
        den *= _dot(_RHO, a)
    val = num / den
    assert val.denominator == 1, f"non-integer dim for {dynkin}: {val}"
    return int(val)


# Triality: S_3 outer automorphism permuting ω_1,ω_3,ω_4 (Dynkin slots a,c,d),
# fixing ω_2 (slot b).  Generators: the transposition 8v<->8s and the 3-cycle.
def triality(dynkin, perm) -> tuple:
    """Apply a permutation `perm` of the three 8's to Dynkin labels.
    `perm` is a 3-tuple giving where (a,c,d) go, e.g. (0,1,2)=id,
    (1,0,2)=swap 8v<->8s, (1,2,0)=3-cycle."""
    a, b, c, d = dynkin
    triple = (a, c, d)
    new = tuple(triple[perm.index(k)] for k in range(3))
    return (new[0], b, new[1], new[2])


def _refl(v, alpha):
    """Reflection s_alpha(v) = v - <v,alpha> alpha  (all D_4 roots have <a,a>=2)."""
    c = Fraction(_dot(v, alpha))
    return tuple(v[i] - c * alpha[i] for i in range(4))


def dom_rep(v):
    """The dominant weight in the Weyl orbit of v (apply simple reflections
    until all Dynkin labels <v,alpha_i> >= 0)."""
    v = tuple(Fraction(x) for x in v)
    moved = True
    while moved:
        moved = False
        for a in _SIMPLE:
            if _dot(v, a) < 0:
                v = _refl(v, a)
                moved = True
    return v


# Simple-root matrix (columns α_i) inverse, for the dominance test Λ-μ=Σk_iα_i.
def _simple_coeffs(w):
    """Solve w = Σ k_i α_i for the k_i (Fractions).  α_i are a basis of R^4."""
    # Build/solve the 4x4 system M k = w, M columns = simple roots.
    M = [[Fraction(_SIMPLE[j][i]) for j in range(4)] for i in range(4)]
    aug = [row[:] + [Fraction(w[i])] for i, row in enumerate(M)]
    # Gaussian elimination.
    for col in range(4):
        piv = next(r for r in range(col, 4) if aug[r][col] != 0)
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [x / pv for x in aug[col]]
        for r in range(4):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [aug[r][k] - f * aug[col][k] for k in range(5)]
    return tuple(aug[i][4] for i in range(4))


def _leq(mu, lam):
    """Dominance: lam - mu = Σ k_i α_i with all k_i >= 0 (convex-hull test)."""
    k = _simple_coeffs(tuple(lam[i] - mu[i] for i in range(4)))
    return all(x >= 0 for x in k)


@lru_cache(maxsize=None)
def _irrep_weights(dynkin):
    """{weight (tuple of Fractions): multiplicity} for irrep `dynkin`, via the
    convex-hull ∩ coset weight set + Freudenthal multiplicities.

    Memoized (pure function of the Dynkin tuple): this Freudenthal pass is the
    dominant cost of the Spin(8) tensor product, and `tensor`/`decompose_weights`
    call it repeatedly with the same labels.  The returned dict is treated
    read-only by all callers (they iterate it and accumulate into a fresh dict)."""
    lam = highest_weight(dynkin)
    # coset type: all-integer (Q/vector) or all-half-integer (spinor) components.
    half = any(x.denominator == 2 for x in lam)
    M = max((abs(x) for x in lam), default=Fraction(0))
    Mi = int(M) + 1
    # candidate component values in the box, matching the coset parity.
    if half:
        vals = [Fraction(2 * t + 1, 2) for t in range(-Mi - 1, Mi + 1)]
    else:
        vals = [Fraction(t) for t in range(-Mi, Mi + 1)]
    # enumerate, filter by coset (mu-lam in root lattice Q) and convex hull.
    cands = []
    for mu in _iproduct(vals, repeat=4):
        diff = tuple(mu[i] - lam[i] for i in range(4))
        if any(d.denominator != 1 for d in diff):
            continue
        if sum(int(d) for d in diff) % 2 != 0:        # D_4 root lattice = even sum
            continue
        if _leq(mu, lam):
            cands.append(mu)
    # Freudenthal on dominant reps, by decreasing <mu,2rho> (height).
    rho = tuple(Fraction(x) for x in _RHO)
    lam_rho = tuple(lam[i] + rho[i] for i in range(4))
    c_lam = _dot(lam_rho, lam_rho)
    dom = sorted({dom_rep(mu) for mu in cands},
                 key=lambda m: -_dot(m, tuple(2 * x for x in rho)))
    mult = {}
    for mu in dom:
        if mu == lam:
            mult[mu] = 1
            continue
        mu_rho = tuple(mu[i] + rho[i] for i in range(4))
        denom = c_lam - _dot(mu_rho, mu_rho)
        s = Fraction(0)
        for a in _POS_ROOTS:
            k = 1
            while True:
                nu = tuple(mu[i] + k * a[i] for i in range(4))
                nd = dom_rep(nu)
                if nd not in mult and nd != lam:
                    # nu beyond the highest weight in this direction -> stop.
                    if not _leq(nd, lam):
                        break
                m_nu = mult.get(nd, 0)
                if m_nu:
                    s += m_nu * _dot(nu, a)
                if not _leq(nd, lam):
                    break
                k += 1
                if k > 4 * (Mi + 2):
                    break
        mult[mu] = int(2 * s / denom) if denom != 0 else 0
    # expand each dominant weight to its full Weyl orbit (same multiplicity).
    out = {}
    for mu in cands:
        out[mu] = mult[dom_rep(mu)]
    return {w: m for w, m in out.items() if m}


def _is_dominant(w):
    return all(_dot(w, a) >= 0 for a in _SIMPLE)


def weight_to_dynkin(w):
    """Dynkin labels (⟨w,α_i⟩) of a (dominant) weight w."""
    return tuple(int(_dot(w, a)) for a in _SIMPLE)


def decompose_weights(wd):
    """{weight: mult} (a Cartan character) -> {dynkin: coeff} irrep decomposition,
    by greedily peeling the highest dominant weight present."""
    wd = {w: m for w, m in wd.items() if m}
    out = {}
    tworho = (6, 4, 2, 0)
    guard = 0
    while any(m for m in wd.values()):
        doms = [w for w, m in wd.items() if m and _is_dominant(w)]
        if not doms:
            raise ValueError("decompose_weights: no dominant weight left "
                             "(not a genuine character)")
        hw = max(doms, key=lambda w: _dot(w, tworho))
        dl = weight_to_dynkin(hw)
        coeff = wd[hw]
        out[dl] = out.get(dl, 0) + coeff
        for w, m in _irrep_weights(dl).items():
            wd[w] = wd.get(w, 0) - coeff * m
        wd = {w: m for w, m in wd.items() if m}
        guard += 1
        if guard > 100000:
            raise RuntimeError("decompose_weights: runaway")
    return out


def tensor(dl_a, dl_b):
    """Tensor product χ_a · χ_b -> {dynkin: coeff} (Clebsch–Gordan)."""
    wa = _irrep_weights(dl_a)
    wb = _irrep_weights(dl_b)
    prod = {}
    for u, mu in wa.items():
        for v, mv in wb.items():
            w = tuple(u[i] + v[i] for i in range(4))
            prod[w] = prod.get(w, 0) + mu * mv
    return decompose_weights(prod)


from zplus_ring import ZPlusRing, RElement, AbelianZPlusRing


class Spin8ZPlusRing(ZPlusRing):
    """R(Spin(8)) = R(D_4) over the Dynkin-label basis ℕ⁴, the SU(2)+N_f=4
    flavour ring.  Multiplication = tensor (Clebsch–Gordan); star = identity
    (every D_4 irrep is self-dual; the nontrivial outer automorphism is the
    **triality** S_3, exposed separately via `triality`).  8v=(1,0,0,0),
    8s=(0,0,1,0), 8c=(0,0,0,1), adjoint 28=(0,1,0,0)."""

    @staticmethod
    def _validate(b):
        if not (isinstance(b, tuple) and len(b) == 4 and all(
                isinstance(x, int) and x >= 0 for x in b)):
            raise ValueError(f"Spin8ZPlusRing basis must be ℕ⁴; got {b!r}")

    # -- abstract primitives --------------------------------------------
    def one_basis(self):
        return (0, 0, 0, 0)

    def multiply_basis(self, b1, b2):
        self._validate(b1)
        self._validate(b2)
        return tensor(b1, b2)

    def star_basis(self, b):
        self._validate(b)
        return b                      # D_4: all irreps self-dual

    def dim(self, b):
        self._validate(b)
        return dim(b)

    def one_dim_rep_rank(self):
        return 0                      # Spin(8) semisimple: only the trivial 1-dim rep

    def embed_one_dim_rep(self, f):
        if tuple(f) != ():
            raise ValueError(f"Spin8ZPlusRing: Λ has rank 0; got {f!r}")
        return self.one_basis()

    # -- triality (S_3 outer automorphism) ------------------------------
    def triality_basis(self, b, perm):
        self._validate(b)
        return triality(b, perm)

    def triality(self, relt, perm):
        """Apply triality `perm` (a 3-perm of 8v/8s/8c) to an RElement."""
        out = {}
        for b, c in relt.terms.items():
            out[triality(b, perm)] = out.get(triality(b, perm), 0) + c
        return RElement(self, {k: v for k, v in out.items() if v})

    # -- maximal-torus un-branching (for the trace) ---------------------
    def from_abelian(self, relt):
        """R(U(1)⁴) -> R(Spin(8)): decompose a Cartan character into irreps.
        The abelian basis label is the weight in **doubled** e-coordinates
        (2·weight, integer 4-tuple) so spinor (half-integer) weights are
        representable; coefficients are its multiplicities."""
        wd = {}
        for lab, c in relt.terms.items():
            w = tuple(Fraction(x, 2) for x in lab)
            wd[w] = wd.get(w, 0) + c
        wd = {w: m for w, m in wd.items() if m}
        if not wd:
            return RElement(self, {})
        return RElement(self, decompose_weights(wd))

    def to_abelian(self, b):
        """R(Spin(8)) -> R(U(1)⁴): the weight diagram in doubled e-coords."""
        R = AbelianZPlusRing(rank=4)
        terms = {}
        for w, m in _irrep_weights(b).items():
            lab = tuple(int(2 * x) for x in w)
            terms[lab] = terms.get(lab, 0) + m
        return RElement(R, terms)


if __name__ == "__main__":
    # Validate dims against the known Spin(8) irreps.
    known = {
        (0, 0, 0, 0): 1,      # trivial
        (1, 0, 0, 0): 8,      # 8v
        (0, 0, 1, 0): 8,      # 8s
        (0, 0, 0, 1): 8,      # 8c
        (0, 1, 0, 0): 28,     # adjoint
        (2, 0, 0, 0): 35,     # 35v (sym^2 8v - trace)
        (0, 0, 2, 0): 35,     # 35s
        (0, 0, 0, 2): 35,     # 35c
        (1, 0, 1, 0): 56,     # 56 (8v x 8s = 8c + 56)
        (0, 0, 1, 1): 56,
        (1, 0, 0, 1): 56,
        (0, 2, 0, 0): 300,
        (1, 1, 0, 0): 160,
        (3, 0, 0, 0): 112,    # 112v
    }
    ok = True
    for dl, want in known.items():
        got = dim(dl)
        flag = "OK" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  dim{dl} = {got}  (want {want})  {flag}")
    # Triality: dim must be invariant under permuting the three 8's.
    import itertools
    for dl in [(1, 0, 0, 0), (2, 1, 0, 0), (1, 0, 2, 0)]:
        dims = {dim(triality(dl, p)) for p in itertools.permutations(range(3))}
        print(f"  triality-invariance of dim{dl}: {dims} -> {'OK' if len(dims)==1 else 'FAIL'}")
        ok &= len(dims) == 1
    print(f"=== Spin(8) step 1 (root datum + dim): {'ALL OK' if ok else 'FAIL'} ===")

    # Step 2: sum of weight multiplicities must equal dim.
    print("--- _irrep_weights: sum of mults == dim ---")
    ok2 = True
    for dl in [(0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 1, 0, 0),
               (2, 0, 0, 0), (1, 0, 1, 0), (1, 1, 0, 0), (0, 2, 0, 0), (3, 0, 0, 0)]:
        w = _irrep_weights(dl)
        tot = sum(w.values())
        d = dim(dl)
        flag = "OK" if tot == d else "FAIL"
        if tot != d:
            ok2 = False
        print(f"  Σmult{dl} = {tot}  (dim {d})  {flag}")
    # 8v weights must be exactly ±e_i (mult 1); adjoint zero-weight mult = rank 4.
    w8v = _irrep_weights((1, 0, 0, 0))
    e_pm = {tuple(Fraction((1 if k == i else 0) * s) for k in range(4))
            for i in range(4) for s in (1, -1)}
    print(f"  8v weights == {{±e_i}}: {'OK' if set(w8v) == e_pm and all(m==1 for m in w8v.values()) else 'FAIL'}")
    wadj = _irrep_weights((0, 1, 0, 0))
    zero = (Fraction(0),) * 4
    print(f"  adjoint zero-weight mult = 4 (rank): {'OK' if wadj.get(zero) == 4 else 'FAIL'}")
    print(f"=== Spin(8) step 2 (_irrep_weights): {'ALL OK' if ok2 else 'FAIL'} ===")

    # Step 3: tensor products (Clebsch–Gordan) against known decompositions.
    print("--- tensor products ---")
    ok3 = True
    tests = {
        ((1, 0, 0, 0), (1, 0, 0, 0)): {(0, 0, 0, 0): 1, (0, 1, 0, 0): 1, (2, 0, 0, 0): 1},  # 8v⊗8v=1+28+35v
        ((1, 0, 0, 0), (0, 0, 1, 0)): {(0, 0, 0, 1): 1, (1, 0, 1, 0): 1},                    # 8v⊗8s=8c+56
        ((0, 0, 1, 0), (0, 0, 0, 1)): {(1, 0, 0, 0): 1, (0, 0, 1, 1): 1},                    # 8s⊗8c=8v+56'
        ((0, 1, 0, 0), (1, 0, 0, 0)): {(1, 0, 0, 0): 1, (1, 1, 0, 0): 1, (0, 0, 1, 1): 1},   # 28⊗8v=8v+160+56
    }
    for (a, b), want in tests.items():
        got = tensor(a, b)
        # check dim balance too
        dbal = dim(a) * dim(b) == sum(dim(dl) * c for dl, c in got.items())
        flag = "OK" if got == want and dbal else "FAIL"
        if got != want or not dbal:
            ok3 = False
        print(f"  {a}⊗{b} = {got}  {flag}")
    # triality covariance: applying triality to both factors permutes the result.
    perm = (1, 2, 0)  # 3-cycle 8v->8s->8c
    lhs = {triality(dl, perm): c for dl, c in tensor((1, 0, 0, 0), (1, 0, 0, 0)).items()}
    rhs = tensor(triality((1, 0, 0, 0), perm), triality((1, 0, 0, 0), perm))
    print(f"  triality covariance of 8v⊗8v: {'OK' if lhs == rhs else 'FAIL'}")
    ok3 &= lhs == rhs
    # from_abelian round-trip: decompose the weight system of an irrep -> itself.
    rt = decompose_weights(_irrep_weights((1, 1, 0, 0)))
    print(f"  decompose(weights(160)) == {{160:1}}: {'OK' if rt == {(1,1,0,0): 1} else 'FAIL'}")
    ok3 &= rt == {(1, 1, 0, 0): 1}
    print(f"=== Spin(8) step 3 (tensor + decompose): {'ALL OK' if ok3 else 'FAIL'} ===")

    # Step 4: the Spin8ZPlusRing class (RElement arithmetic + un-branching).
    print("--- Spin8ZPlusRing class ---")
    ok4 = True
    R = Spin8ZPlusRing()
    fund = R.basis_element((1, 0, 0, 0))
    prod = (fund * fund).terms
    want = {(0, 0, 0, 0): 1, (0, 1, 0, 0): 1, (2, 0, 0, 0): 1}
    print(f"  RElement 8v*8v == 1+28+35v: {'OK' if prod == want else 'FAIL'}  {prod}")
    ok4 &= prod == want
    print(f"  star(8v) == 8v (self-dual): {'OK' if R.star_basis((1,0,0,0)) == (1,0,0,0) else 'FAIL'}")
    print(f"  dim(8v)=8, dim(28)=28: {'OK' if R.dim((1,0,0,0))==8 and R.dim((0,1,0,0))==28 else 'FAIL'}")
    # from_abelian ∘ to_abelian round-trips to the irrep.
    for dl in [(1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 1, 0, 0), (1, 0, 1, 0)]:
        rt = R.from_abelian(R.to_abelian(dl)).terms
        flag = "OK" if rt == {dl: 1} else "FAIL"
        if rt != {dl: 1}:
            ok4 = False
        print(f"  from_abelian(to_abelian({dl})) == {{{dl}:1}}: {flag}")
    # triality on RElement: 8v -> 8s under the 3-cycle.
    tri = R.triality(fund, (1, 2, 0)).terms
    print(f"  triality(8v, 3-cycle) == 8s: {'OK' if tri == {(0,0,1,0): 1} else 'FAIL'}")
    ok4 &= tri == {(0, 0, 1, 0): 1}
    print(f"=== Spin(8) step 4 (Spin8ZPlusRing): {'ALL OK' if ok4 else 'FAIL'} ===")
