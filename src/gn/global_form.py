"""`global_form` — the type-A global forms U(N) / SU(N) / PSU(N) and their
't Hooft–Wilson charge lattices, in the U(N) `e`-basis `Z^N` (the basis the
enriched torus uses, as did the `pure_via_n2star` builder retired 2026-09-19).

Infrastructure to **talk about PSU(N)'s L's** on
the same footing as U(N)/SU(N), and to relate them to what the U(N) engine
builds.

The charge lattices (a line operator carries magnetic `m` = cocharacter and
electric `e` = character; both live in `Z^N` here):

| form | magnetic `m` | electric `e` | minimal 't Hooft | minimal Wilson |
|---|---|---|---|---|
| **U(N)** | `Z^N` | `Z^N` | minuscule `(1,0,…,0)` | fundamental `(1,0,…,0)` |
| **SU(N)** | coroot: `Σm=0` | weight: `Z^N` | **adjoint** `(1,0,…,−1)` | fundamental |
| **PSU(N)** | coweight: `Z^N / Z·𝟙` | root: `Σe=0` | **minuscule** `(1,0,…,0) mod 𝟙` | adjoint `(1,0,…,−1)` |

Center charges (the `ℤ_N` defect data): magnetic `z_m = Σm mod N`, electric
`z_e = Σe mod N`.  A genuine line of a global form has the *complementary*
center charge trivial:

  * **U(N)** — both `z_m, z_e` free (it is not a simple-gauge-group form; the
    diagonal `U(1)` carries them);
  * **SU(N)** — `z_m = 0` (magnetic center trivial: `Σm = 0`); `z_e` free;
  * **PSU(N)** — `z_e = 0` (electric center trivial: `Σe = 0`); `z_m` free.

**S-duality** `S: SU(N) ↔ PSU(N)` swaps magnetic ↔ electric, i.e. the coroot
(`Σm=0`) magnetic lattice of SU(N) with the root (`Σe=0`) electric lattice of
PSU(N), and the full electric of SU(N) with the mod-`𝟙` magnetic of PSU(N).
This is why the N=2\* slope constructor (`pure_via_n2star`, S-covariant; retired
2026-09-19 with the type-A layer) was
the natural tool for the **minuscule sector of either form**: SU(N)'s
minuscule *Wilson* lines and PSU(N)'s minuscule *'t Hooft* lines are S-dual.

**No fractional powers of 𝖖 are needed**.  PSU(N)'s
minuscule monopoles are fractional *cocharacters* (coweights, the `1/N·𝟙`
shifts), but PSU(N)'s electric charges are the **root lattice** (`Σe = 0`),
so the diagonal/trace part of the magnetic charge **pairs to zero** against
them:  `⟨m, e⟩ = ⟨m_int, e⟩ + (k/N)·Σe = integer + 0`.  The fractional part
never produces a fractional phase, so the **integer U(N) representative**
carries the full PSU(N) physics with only integer powers of 𝖖 — the integer
torus suffices, no √𝖖 / centre extension.

**The U(N) bridge (`un_representative`).**  A PSU(N) label `(m mod 𝟙, e)`
with `Σe = 0` has a canonical integer U(N) representative `(m₀, e)` with
`Σm₀ = z_m ∈ {0,…,N−1}` (the least non-negative diagonal shift); any two
representatives differ by `𝟙`, which pairs to `0` with the traceless `e`.
For the minuscule 't Hooft class `z_m = 1` this is `m₀ = (1,0,…,0)` —
**U(N)'s minuscule monopole**, which the builder produces instantly (integer
𝖖 throughout).  So the PSU(N) minuscule-monopole chart is *reachable from the
U(N) engine directly*.  (Whether the U(N) canonical equals the genuine
PSU(N) canonical once the electric spectrum shrinks to the root lattice is a
build relationship to verify per label; this module is the labeling/charge
layer.)
"""
from __future__ import annotations

from dataclasses import dataclass


def _is_integral(x) -> bool:
    """Whether a coordinate or pairing value is an integer — **exactly**.

    The coordinates here are `int` where integral and `Fraction` otherwise (the
    denominators divide the centre order), so `Fraction(x).denominator == 1` is
    the whole test — the same idiom `fundamental_coweights` normalises with.

    This is a **lattice-membership predicate**, which is why the float round-trip
    it replaces was not merely inelegant: `verify_mutually_local` and
    `mag_admits` decide what a global form *contains* by asking it, so a wrong
    answer admits a line that is not mutually local — i.e. corrupts the 4d gauge
    group data itself.  And the float route can be wrong:
    `float(Fraction(2**53 + 1, 2**53)).is_integer()` is `True` for a value whose
    denominator is not 1, and `float()` of a large `Fraction` raises
    `OverflowError` rather than answering.  (No floating-point arithmetic in the
    algebra core is a standing rule; this is that rule at a decision point.)"""
    from fractions import Fraction
    return Fraction(x).denominator == 1


def _center(vec) -> int:
    return sum(int(x) for x in vec) % len(vec)


def _is_integral(x) -> bool:
    """Whether a coordinate or pairing value is an integer — **exactly**.

    The coordinates here are `int` where integral and `Fraction` otherwise (the
    denominators divide the centre order), so `Fraction(x).denominator == 1` is
    the whole test — the same idiom `fundamental_coweights` normalises its own
    coordinates with below.  This is a
    lattice-membership predicate: `mag_admits` and `verify_mutually_local` decide
    what a global form contains by asking it, and a float round-trip is not a
    sound way to decide membership of a lattice."""
    from fractions import Fraction
    return Fraction(x).denominator == 1


@dataclass(frozen=True)
class GlobalForm:
    """A type-A global form and its 't Hooft–Wilson charge lattices in `Z^N`.

    `mag`/`elec` are `"full"` (all of `Z^N`), `"traceless"` (`Σ = 0`), or
    `"mod_diag"` (`Z^N` modulo `𝟙 = (1,…,1)`)."""

    name: str
    N: int
    mag: str          # "full" | "traceless" | "mod_diag"
    elec: str         # "full" | "traceless"

    # ----- lattice membership / canonical representatives -------------------
    def mag_valid(self, m) -> bool:
        m = tuple(int(x) for x in m)
        if len(m) != self.N:
            return False
        if self.mag == "traceless":
            return sum(m) == 0
        return True          # "full" and "mod_diag" accept any integer vector

    def elec_valid(self, e) -> bool:
        e = tuple(int(x) for x in e)
        if len(e) != self.N:
            return False
        if self.elec == "traceless":
            return sum(e) == 0
        return True

    def mag_rep(self, m) -> tuple:
        """Canonical integer representative of the magnetic charge.  For
        `mod_diag` (PSU), shift by `𝟙` to the least non-negative trace in
        `{0,…,N−1}` (the center-charge normal form)."""
        m = tuple(int(x) for x in m)
        if self.mag == "mod_diag":
            k = sum(m) // self.N          # floor; land trace in [0, N)
            m = tuple(x - k for x in m)
            while sum(m) >= self.N:
                m = tuple(x - 1 for x in m)
            while sum(m) < 0:
                m = tuple(x + 1 for x in m)
        return m

    def mag_center(self, m) -> int:
        """`z_m = Σm mod N` (the magnetic ℤ_N center charge)."""
        return _center(m)

    def elec_center(self, e) -> int:
        return _center(e)

    def label_valid(self, m, e) -> bool:
        """A genuine `(m, e)` line of this form: lattice membership + the
        complementary-center-triviality that defines the form."""
        if not (self.mag_valid(m) and self.elec_valid(e)):
            return False
        if self.name.startswith("SU") and self.mag_center(m) != 0:
            return False   # SU: magnetic center trivial (already Σm=0 ⇒ ok)
        if self.name.startswith("PSU") and self.elec_center(e) != 0:
            return False   # PSU: electric center trivial (already Σe=0 ⇒ ok)
        return True

    # ----- distinguished lines ---------------------------------------------
    def minimal_thooft(self) -> tuple:
        """The minimal (non-trivial, dominant) magnetic charge."""
        if self.mag == "traceless":                 # SU: adjoint
            return (1,) + (0,) * (self.N - 2) + (-1,)
        return (1,) + (0,) * (self.N - 1)            # U/PSU: minuscule

    def minimal_wilson(self) -> tuple:
        """The minimal (non-trivial, dominant) electric charge."""
        if self.elec == "traceless":                # PSU: adjoint
            return (1,) + (0,) * (self.N - 2) + (-1,)
        return (1,) + (0,) * (self.N - 1)            # U/SU: fundamental

    def un_representative(self, m, e) -> tuple:
        """The integer U(N) `(m, e)` representative of a label of this form
        (the least-non-negative-trace magnetic rep + the electric charge
        as-is), from which the U(N) engine can attempt a chart."""
        return (self.mag_rep(m), tuple(int(x) for x in e))

    def __repr__(self) -> str:
        return (f"GlobalForm({self.name}: mag={self.mag}, elec={self.elec}, "
                f"z_m∈ℤ_{self.N} {'free' if self.mag != 'traceless' else '=0'}, "
                f"z_e {'=0' if self.elec == 'traceless' else 'free'})")


# ---------------------------------------------------------------------------
# factories + S-duality
# ---------------------------------------------------------------------------
def u_n_form(N: int) -> GlobalForm:
    return GlobalForm(f"U({N})", int(N), "full", "full")


def su_n_form(N: int) -> GlobalForm:
    return GlobalForm(f"SU({N})", int(N), "traceless", "full")


def psu_n_form(N: int) -> GlobalForm:
    return GlobalForm(f"PSU({N})", int(N), "mod_diag", "traceless")


def s_dual(form: GlobalForm) -> GlobalForm:
    """`S: SU(N) ↔ PSU(N)` (swap magnetic ↔ electric lattices).  `U(N)` is
    self-dual as a lattice datum (the diagonal U(1) is S-invariant here)."""
    if form.name.startswith("SU"):
        return psu_n_form(form.N)
    if form.name.startswith("PSU"):
        return su_n_form(form.N)
    return u_n_form(form.N)


# ---------------------------------------------------------------------------
# Datum-general global forms — the 4d notion
#
# **In 4d the global form is NOT "which cocharacter lattice".**  That reading is
# the 3d-appropriate one.  The 4d datum is a **sublattice of the Kapustin
# `(m, e)` label lattice**: a maximal set of mutually local line operators.  The
# repo already says this in two places — the design notes
# (`(M, E)_{SO(3)} = (2m, e/2)_{SU(2)}`: SO(3) is the S-dual global form on the
# *same* `su(2)` line algebra, magnetic doubled and electric halved) and the
# type-A `GlobalForm` above (each form = a magnetic lattice + an electric
# lattice, pinned by which centre charge is required to vanish).  The classes
# below generalise that to an arbitrary `RootDatum`.
#
# The structure: let `Z = P/Q` be the centre of the simply connected form.  A
# global form is a subgroup `H ⊆ Z` (the centre one quotients by), and
#
#     magnetic lattice  =  { m ∈ P^∨ : [m] ∈ H }        (H = 1 ⇒ Q^∨)
#     electric lattice  =  { e ∈ P   : e|_H = 0 }       (H = 1 ⇒ P)
#
# Mutual locality is integrality of the **Dirac pairing** on labels,
# `⟨(m,e), (m',e')⟩ = ⟨m, e'⟩ − ⟨m', e⟩`, and the two conditions above are
# exactly what makes it hold.  (Discrete theta angles give further maximal
# isotropic choices with the same `H`; not modelled here.)
# ---------------------------------------------------------------------------

def _smith_invariants(M) -> tuple:
    """Elementary divisors of an integer matrix (small ranks; plain Smith)."""
    A = [list(map(int, row)) for row in M]
    rows, cols = len(A), len(A[0]) if A else 0
    inv, r, c = [], 0, 0
    while r < rows and c < cols:
        piv = None
        for i in range(r, rows):
            for j in range(c, cols):
                if A[i][j] and (piv is None or abs(A[i][j]) < abs(A[piv[0]][piv[1]])):
                    piv = (i, j)
        if piv is None:
            break
        pi, pj = piv
        A[r], A[pi] = A[pi], A[r]
        for row in A:
            row[c], row[pj] = row[pj], row[c]
        done = False
        while not done:
            done = True
            for i in range(r + 1, rows):
                if A[i][c] % A[r][c]:
                    q = A[i][c] // A[r][c]
                    A[i] = [x - q * y for x, y in zip(A[i], A[r])]
                    A[r], A[i] = A[i], A[r]
                    done = False
            for j in range(c + 1, cols):
                if A[r][j] % A[r][c]:
                    q = A[r][j] // A[r][c]
                    for row in A:
                        row[j] -= q * row[c]
                    for row in A:
                        row[c], row[j] = row[j], row[c]
                    done = False
        for i in range(r + 1, rows):
            q = A[i][c] // A[r][c]
            A[i] = [x - q * y for x, y in zip(A[i], A[r])]
        for j in range(c + 1, cols):
            q = A[r][j] // A[r][c]
            for row in A:
                row[j] -= q * row[c]
        inv.append(abs(A[r][c]))
        r += 1
        c += 1
    return tuple(x for x in inv if x != 1)


def cartan_matrix(datum) -> list:
    """`a_ij = ⟨α_i, α_j^∨⟩` for a `RootDatum`."""
    n = len(datum.simple_roots)
    return [[datum.shift_pairing(datum.simple_coroots[j], datum.simple_roots[i])
             for j in range(n)] for i in range(n)]


def centre_invariants(datum) -> tuple:
    """The centre `Z(G̃) ≅ P/Q` as elementary divisors — `()` when trivial.

    In the fundamental-weight coordinates the weight lattice is `Z^r` and the
    root lattice is the row span of the Cartan matrix, so `Z = Z^r / C·Z^r`.
    Measured: `SU(N) → (N,)`, `Spin(5) → (2,)`, `G₂ → ()` (trivial — hence a
    single global form, which is why the `G₂` work is untouched by this)."""
    return _smith_invariants(cartan_matrix(datum))


def fundamental_coweights(datum):
    """The fundamental coweights `ω_i^∨`, in the datum's own **cocharacter
    coordinates** — defined by `⟨α_j, ω_i^∨⟩ = δ_ij`.  `P^∨` is their `Z`-span and
    `Q^∨ ⊆ P^∨`, so the *fractional* ones are exactly the magnetic charges a
    non-simply-connected global form adds.

    Coordinates are `Fraction`s in general (denominators divide the centre order),
    which is why the substrate needed the integral-pairing coercion in
    `wrq_torus._root_pairing_count`.

    **Solved against the datum's own `shift_pairing`, not read off `C^{-1}`**
    (corrected 2026-07-28).  Inverting the Cartan matrix requires committing to an
    index convention *and* assuming the cocharacter coordinates are the coroot
    basis, and both assumptions break in the repo:

    * `cartan_matrix[i][j] = ⟨α_i, α_j^∨⟩` is **not symmetric** for non-simply-laced
      `G`, so rows ≠ columns and only the *columns* of `C^{-1}` satisfy the
      defining property — measured: taking rows gives `⟨α_j, ω_i^∨⟩` grids of
      `[[0,1],[-1,3/2]]` at `Spin(5)` and `[[3,-4],[4,-5]]` at `G₂`;
    * `sp_n` states its roots in the standard `C_n` coordinates (`e_i − e_j`,
      `2e_i`) rather than the fundamental-weight basis the other constructors use,
      so at `Sp(4)` **neither** rows nor columns are right.

    Solving `⟨α_j, w⟩ = δ_ij` through `shift_pairing` is convention-free: it uses
    the very pairing the ψ-dressing and the atom machinery use, so a coweight built
    here is a coweight *for the substrate* by construction."""
    from fractions import Fraction
    from wrq_torus import _pairing_root_cochar
    d = datum.dim
    roots = [tuple(a) for a in datum.simple_roots]
    if len(roots) != d:
        raise NotImplementedError(
            f"{datum.name}: {len(roots)} simple roots but cocharacter dimension "
            f"{d} — the coweight system is not square, so ⟨α_j, w⟩ = δ_ij does "
            f"not determine w (a non-semisimple factor needs its own basis).")
    # M[j][k] = ⟨α_j, e_k⟩ through the datum's OWN pairing; then ω_i^∨ solves
    # M·w = e_i.  Bilinearity of shift_pairing is what makes this legitimate.
    basis = [tuple(1 if t == k else 0 for t in range(d)) for k in range(d)]
    M = [[Fraction(_pairing_root_cochar(datum, roots[j], basis[k]))
          for k in range(d)] for j in range(d)]
    A = [M[j] + [Fraction(1 if j == k else 0) for k in range(d)]
         for j in range(d)]
    for c in range(d):
        piv = next((i for i in range(c, d) if A[i][c] != 0), None)
        if piv is None:
            raise NotImplementedError(
                f"{datum.name}: the simple-root pairing matrix is singular, so "
                f"the fundamental coweights are not determined.")
        A[c], A[piv] = A[piv], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for i in range(d):
            if i != c and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[c])]
    # ω_i^∨ = M^{-1}·e_i = the i-th COLUMN of the inverse
    return [tuple(A[k][d + i] for k in range(d)) for i in range(d)]


def is_simply_laced(datum) -> bool:
    """`G` is simply laced ⟺ its Cartan matrix is symmetric ⟺ its root system is
    **Langlands self-dual** (`C^∨ = C^T`)."""
    C = cartan_matrix(datum)
    return C == [list(row) for row in zip(*C)]


def langlands_self_dual_permutation(datum):
    """The node permutation `σ` with `Cᵀ = σ·C·σ^{-1}`, or `None` if none exists.

    A root datum is Langlands **self-dual** exactly when its transposed Cartan matrix
    is its own, up to relabelling the nodes — `C^∨ = Cᵀ`, and a relabelling is an
    isomorphism of root data.  This is strictly weaker than *simply laced* and that
    matters here:

    * simply laced (A, D, E): `Cᵀ = C`, `σ = id`;
    * `B₂ ≅ C₂`: `C = [[2,−2],[−1,2]]`, `Cᵀ = [[2,−1],[−2,2]]`, and swapping the two
      nodes carries one to the other — the exceptional isomorphism, which is why
      `Spin(5)^∨ = PSp(4) ≅ SO(5)` is expressible on `B₂`'s **own** datum;
    * `G₂`: `C = [[2,−1],[−3,2]]`, likewise self-dual under the node swap — so `G₂` is
      Langlands self-dual and `S` is an **automorphism** of it.

    Returned as a tuple `σ` with `σ[i]` the image of node `i`.  Searched by brute
    force over permutations, which is fine at the ranks in play."""
    import itertools
    C = cartan_matrix(datum)
    n = len(C)
    CT = [[C[j][i] for j in range(n)] for i in range(n)]
    for perm in itertools.permutations(range(n)):
        if all(CT[i][j] == C[perm[i]][perm[j]] for i in range(n)
               for j in range(n)):
            return perm
    return None


def langlands_dual_datum(datum):
    """`G^∨` — the Langlands dual **root datum**.

    `C^∨ = C^T`, so a **simply laced** `G` (types A, D, E) is self-dual and this is
    the identity on the datum; what the duality moves in that case is the *global
    form* (`LineLattice.langlands_dual`), not the root system.  That is exactly the
    `SU(N) ↔ PSU(N)` pattern: same `RootDatum`, magnetic and electric lattices
    exchanged.

    **Honest-fails for non-simply-laced `G`.**  `B_n^∨ = C_n` and the repo *has*
    both (`b_n_simply_connected`, `sp_n`), but they are stated in **different
    coordinate conventions** — `sp_n` uses the standard `C_n` coordinates
    (`e_i − e_j`, `2e_i`) while `b_n_simply_connected` uses the fundamental-weight
    basis — so identifying them requires an explicit change of basis that is not
    yet measured.  Guessing it is exactly the error `fundamental_coweights` already
    made once (rows vs columns of `C^{-1}`, which silently produced non-coweights at
    `Spin(5)`/`Sp(4)`), so this refuses rather than assumes.

    Note also that at `n = 2` the repo's `b_n_simply_connected(2)` and `sp_n(2)` are
    the **same group** (`Spin(5) ≅ Sp(4)`, the exceptional `B₂ ≅ C₂` isomorphism),
    not a dual pair — the dual of `Spin(5)` is `SO(5)`, its own adjoint form.

    **Cross-datum since 2026-07-29.**  For `n ≥ 3` the dual really is a different
    datum, `B_n^∨ = C_n`, and both families are in the repo — the obstacle was only
    that they are stated in **different coordinate conventions** (`sp_n` uses the
    standard `C_n` coordinates, the others the fundamental-weight basis).  That is no
    longer an obstacle, because the charge dictionary is now built from the
    fundamental (co)weights of each datum separately (`langlands_label_map`) and so
    never assumes a convention.  The dual is found by **searching the constructors of
    the same rank for one whose Cartan matrix is `Cᵀ` up to a node relabelling** —
    identified, not guessed."""
    if langlands_self_dual_permutation(datum) is not None:
        return datum
    d = datum.dim
    for cand in _dual_candidates(d):
        if langlands_node_permutation(datum, cand) is not None:
            return cand
    raise NotImplementedError(
        f"{datum.name}: no dual datum found.  Self-dual data (A/D/E, B₂≅C₂, G₂, F₄) "
        f"return themselves; `B_n`/`C_n` find each other among the rank-{d} "
        f"constructors.  {datum.name} matched neither, so its dual is a datum this "
        f"module cannot name — which it refuses to guess.")


def _dual_candidates(rank):
    """The rank-`rank` root data to search when looking for a Langlands dual.

    Deliberately a short explicit list rather than a registry: the point is to
    *identify* the dual by its Cartan matrix, and a candidate that does not match is
    simply skipped, so the list only has to be broad enough to contain the answer."""
    import root_datum as _rd
    out = []
    for make in (lambda: _rd.sp_n(rank), lambda: _rd.b_n_simply_connected(rank),
                 lambda: _rd.su_n(rank + 1)):
        try:
            out.append(make())
        except Exception:
            pass
    return out


def langlands_node_permutation(datum, dual):
    """The node relabelling `σ` with `C(datum)ᵀ_{ij} = C(dual)_{σ(i)σ(j)}`, or `None`.

    The cross-datum generalisation of `langlands_self_dual_permutation`, which is
    exactly this with `dual = datum`.  `σ` is what makes the canonical identification
    `ω_i^∨(G) ↔ ω_{σ(i)}(G^∨)` concrete."""
    import itertools
    C, Cv = cartan_matrix(datum), cartan_matrix(dual)
    n = len(C)
    if len(Cv) != n:
        return None
    CT = [[C[j][i] for j in range(n)] for i in range(n)]
    for perm in itertools.permutations(range(n)):
        if all(CT[i][j] == Cv[perm[i]][perm[j]] for i in range(n)
               for j in range(n)):
            return perm
    return None


def langlands_label_map(datum):
    """`(forward, inverse)` — the Langlands charge map `S: (m, e) ↦ (e, −m)` made
    concrete in the repo's coordinates.

    `(m, e) ↦ (e, −m)` is only meaningful once `G`'s **cocharacters** are identified
    with `G^∨`'s **weights** and vice versa.  Langlands duality does exactly that —
    `X_*(T) = X^*(T^∨)` — and the identification is canonical **on the fundamental
    (co)weights**:

        ω_i^∨(G)  ⟷  ω_{σ(i)}(G^∨) ,      ω_i(G)  ⟷  ω_{σ(i)}^∨(G^∨) ,

    with `σ` the node relabelling between `Cᵀ` and the dual datum's Cartan matrix
    (`langlands_node_permutation`).  So the dictionary is: *express the charge in the
    fundamental (co)weight basis, relabel the nodes, read back in the dual's*, and

        forward:  (m, e) ↦ ( ι(e), −ι^∨(m) )
        inverse:  (m, e) ↦ ( −ι(e), ι^∨(m) )

    It carries `P^∨ → P` and `Q^∨ → Q` bijectively — i.e. the simply connected form's
    `(Q^∨, P)` to the adjoint form's `(P^∨, Q)`, the lattice content of the duality.

    **Stated this way it needs no coordinate convention at all**, which is what makes
    it work **cross-datum** (`B_n ↔ C_n` for `n ≥ 3`, where the two data are stated in
    genuinely different conventions).  The earlier formulation `φ = P_σ·C·` assumed
    cocharacters in the coroot basis and weights in the fundamental-weight basis —
    true for `su_n` / `b_n_simply_connected` / `g_2`, false for `sp_n`.  The rewrite
    was checked to reproduce the old map **exactly**, forward and inverse, on all
    five certified self-dual data (SU(2), SU(3), SU(4), Spin(5), G₂), so nothing
    measured with it is disturbed.

    Verified (2026-07-28) at SU(2)/SU(3)/SU(4): `φ(ω_i^∨) = ω_i` exactly, the Dirac
    pairing satisfies `⟨m, e⟩ = −⟨φ^{-1}(e), −φ(m)⟩` on the basis, and the lattice
    bookkeeping lands where stated.  `forward` composed with itself is charge
    conjugation `(m, e) ↦ (−m, −e)`, as `S² = −1` in `SL(2, Z)` requires.

    Coordinates come back as `int` where integral and `Fraction` otherwise — a
    Wilson line of `SU(N)` maps to a *fractional-coweight* monopole of `PSU(N)`, so
    the fractions are the point, not an artifact.

    **NOTATION — read this before comparing against a formula written elsewhere.**
    Both global forms are expressed in **one** coordinate system, the simply
    connected one's: `m` is a cocharacter in the **coroot basis** (multiples of
    `α_i^∨`) and `e` a weight in the **fundamental-weight basis** (multiples of
    `ω_i`).  The two forms are the *same* `RootDatum`; only the admissibility
    predicates differ.  At `SU(2)`:

        SU(2):   m ∈ Z   (= Q^∨)          e ∈ Z   (= P)
        SO(3):   m ∈ ½Z  (= P^∨, gen ω^∨)  e ∈ 2Z  (= Q, gen α)

    so in **this** notation the map reads

        L^{SU(2)}_{m,e}  ⟼  L^{SO(3)}_{e/2, −2m}.

    That is the *same map* as the literal `L_{m,e} ↦ L_{e,−m}` written in SO(3)'s
    **own** units, where both charges are integers — the two notations differ by
    `(M, E)_{SO(3)} = (2m, e/2)_{SU(2)}` (the design notes,
    `src/gn/pure_so3.py`).  Conflating them is a live trap: it pairs
    `SU(2)`'s Wilson with the `SU(2)` *adjoint monopole* instead of with `SO(3)`'s
    minimal 't Hooft, and the wrong pairing still "builds", so nothing flags it."""
    from fractions import Fraction
    C0 = cartan_matrix(datum)
    d = len(C0)
    # φ = P_σ · C, i.e. the rows of C permuted by the self-duality node permutation
    # σ (`langlands_self_dual_permutation`).  σ = id for A/D/E, so this reduces to
    # the plain `C·` there; for B₂/C₂ and G₂, σ is the node swap and the permutation
    # is REQUIRED.  Measured (2026-07-28) at Spin(5) against the four criteria
    # (coweights ↦ fundamental weights, `S² = −1`, and the two lattice inclusions):
    # `P·C` satisfies all four while plain `C·` fails the ELECTRIC one — it sends
    # the coroot generator to `(−2, 1)`, which is not in the root lattice `Q`
    # (`a = −3/2`), so the dual of a Spin(5) monopole was not a line of SO(5).
    #
    # That was measured the other way round earlier the same day and the earlier
    # reading was an artifact: it used `elec_admits` *before* the centre-grading fix,
    # which mis-graded genuine roots and so accepted `(−2,1)`.  Re-measured against
    # `in_root_lattice`, `P·C` is the dictionary.
    dual = langlands_dual_datum(datum)
    sigma = langlands_node_permutation(datum, dual)
    if sigma is None:
        raise NotImplementedError(
            f"{datum.name}: no Langlands charge dictionary — no node relabelling "
            f"carries Cᵀ to the dual datum's Cartan matrix.")
    dv = dual.dim
    cow, wt = fundamental_coweights(datum), fundamental_weights(datum)
    cowv, wtv = fundamental_coweights(dual), fundamental_weights(dual)

    def _norm(v):
        return tuple(int(x) if Fraction(x).denominator == 1 else Fraction(x)
                     for x in v)

    def _coords(basis, v, dim):
        """Coordinates of `v` in `basis`, exactly over `Q`."""
        n = len(basis)
        A = [[Fraction(basis[j][i]) for j in range(n)] + [Fraction(v[i])]
             for i in range(dim)]
        row, piv = 0, []
        for col in range(n):
            p = next((r for r in range(row, dim) if A[r][col] != 0), None)
            if p is None:
                continue
            A[row], A[p] = A[p], A[row]
            pv = A[row][col]
            A[row] = [x / pv for x in A[row]]
            for r in range(dim):
                if r != row and A[r][col] != 0:
                    f = A[r][col]
                    A[r] = [x - f * y for x, y in zip(A[r], A[row])]
            piv.append(col)
            row += 1
        for r in range(row, dim):
            if A[r][n] != 0:
                raise NotImplementedError(
                    f"{datum.name}: {v} is not in the span of the fundamental "
                    f"(co)weights, so it has no Langlands image.")
        out = [Fraction(0)] * n
        for i, c in enumerate(piv):
            out[c] = A[i][n]
        return out

    def _combine(basis, coeffs, dim):
        return _norm(tuple(sum(Fraction(c) * Fraction(basis[j][i])
                               for j, c in enumerate(coeffs))
                           for i in range(dim)))

    def _relabel(coeffs):
        out = [Fraction(0)] * d
        for i, c in enumerate(coeffs):
            out[sigma[i]] = c
        return out

    def _unrelabel(coeffs):
        return [coeffs[sigma[i]] for i in range(d)]

    def forward(m, e):
        a, b = _coords(cow, m, d), _coords(wt, e, d)
        return (_combine(cowv, _relabel(b), dv),
                _combine(wtv, [-x for x in _relabel(a)], dv))

    def inverse(m, e):
        a, b = _coords(cowv, m, dv), _coords(wtv, e, dv)
        return (_combine(cow, [-x for x in _unrelabel(b)], d),
                _combine(wt, _unrelabel(a), d))

    return forward, inverse


def _int_solve(cols, target, d):
    """Is `target` an INTEGER combination of `cols`?  Rational elimination, then
    an integrality test on the solution — the same device `in_root_lattice` uses."""
    from fractions import Fraction
    n = len(cols)
    A = [[Fraction(cols[j][i]) for j in range(n)] + [Fraction(target[i])]
         for i in range(d)]
    row, piv = 0, []
    for col in range(n):
        p = next((r for r in range(row, d) if A[r][col] != 0), None)
        if p is None:
            continue
        A[row], A[p] = A[p], A[row]
        pv = A[row][col]
        A[row] = [x / pv for x in A[row]]
        for r in range(d):
            if r != row and A[r][col] != 0:
                f = A[r][col]
                A[r] = [x - f * y for x, y in zip(A[r], A[row])]
        piv.append(col)
        row += 1
    for r in range(row, d):
        if A[r][n] != 0:
            return False
    sol = [Fraction(0)] * n
    for i, c in enumerate(piv):
        sol[c] = A[i][n]
    return all(x.denominator == 1 for x in sol)


def in_coroot_lattice(datum, m) -> bool:
    """`m ∈ Q^∨` — an integer combination of the simple coroots."""
    return _int_solve([tuple(c) for c in datum.simple_coroots], tuple(m),
                      datum.dim)


def fundamental_weights(datum):
    """The fundamental weights `ω_i`, in the datum's own **character coordinates** —
    defined by `⟨α_j^∨, ω_i⟩ = δ_ij`.

    The exact mirror of `fundamental_coweights`, and solved the same way: against
    the datum's *own* `shift_pairing`, so no coordinate convention is assumed.  That
    matters because the constructors disagree — `su_n`, `b_n_simply_connected` and
    `g_2` state their simple roots in the fundamental-weight basis (`simple_roots` =
    the rows of the Cartan matrix), while **`sp_n` uses the standard `C_n`
    coordinates** (`e_i − e_j`, `2e_i`).  So "the `i`-th basis vector" is `ω_i` for
    three of them and something else for `sp_n`, and anything that needs the actual
    fundamental weights has to solve for them."""
    from fractions import Fraction
    d = datum.dim
    cor = [tuple(c) for c in datum.simple_coroots]
    if len(cor) != d:
        raise NotImplementedError(
            f"{datum.name}: {len(cor)} simple coroots but character dimension "
            f"{d} — `⟨α_j^∨, ω_i⟩ = δ_ij` does not determine `ω_i`.")
    basis = [tuple(1 if t == k else 0 for t in range(d)) for k in range(d)]
    M = [[Fraction(datum.shift_pairing(cor[j], basis[k])) for k in range(d)]
         for j in range(d)]
    A = [M[j] + [Fraction(1 if j == k else 0) for k in range(d)]
         for j in range(d)]
    for c in range(d):
        piv = next((i for i in range(c, d) if A[i][c] != 0), None)
        if piv is None:
            raise NotImplementedError(
                f"{datum.name}: the simple-coroot pairing matrix is singular, so "
                f"the fundamental weights are not determined.")
        A[c], A[piv] = A[piv], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for i in range(d):
            if i != c and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[c])]
    return [tuple(A[k][d + i] for k in range(d)) for i in range(d)]


def _class_order(datum, w, n, in_sub):
    """The order of `[w]` in a cyclic quotient of order `n`, `in_sub` deciding
    membership of the sublattice (`in_coroot_lattice` or `in_root_lattice`)."""
    from fractions import Fraction
    d = datum.dim
    for k in range(1, n + 1):
        if in_sub(datum, tuple(k * Fraction(x) for x in w)):
            return k
    return None


def centre_generators(datum):
    """`(g^∨, g)` — **dual** generators of `P^∨/Q^∨` and `P/Q`, so that a magnetic
    class `k` and an electric class `l` pair to `k·l/n` in `Q/Z`.

    This exists because the obvious shortcut is wrong twice over:

    * `ω_1^∨` **need not generate** `P^∨/Q^∨`.  At `C_n` it is *integral* (class 0),
      so labelling classes by multiples of `ω_1^∨` reported `None` — "not in `P^∨`" —
      for `Sp(4)`'s genuine class-1 coweight `(½,½)`, which `in_coweight_lattice`
      confirms *is* in `P^∨`.  The generator has to be **found**, and at `C_n` it is
      the *last* fundamental coweight.
    * the electric labelling must be the **dual** basis, not an independent choice.
      `elec_admits` asks whether `[e]` annihilates `H`, and it evaluates that as
      `k·h ≡ 0 (mod n)` — a statement about the perfect pairing
      `P^∨/Q^∨ × P/Q → Q/Z`.  If the two labellings are not dual, that congruence is
      testing nothing.  So `g` is normalised by `⟨g^∨, g⟩ ≡ 1/n`: pick any weight of
      full order, read off `⟨g^∨, g⟩ = u/n` with `u` invertible mod `n`, and replace
      `g` by `u^{-1}·g`.

    Honest-fails when no single fundamental (co)weight has full order — possible in
    principle for a cyclic group generated by a set with no single generator, and
    not the case for any datum here (`A_n`: `ω_1^∨`; `B_n`/`C_n`/`E_7`: order 2, any
    non-trivial class; `E_6`: order 3, likewise)."""
    from fractions import Fraction
    inv = centre_invariants(datum)
    if not inv:
        return None, None
    if len(inv) > 1:
        raise NotImplementedError(
            f"{datum.name}: centre {inv} is not cyclic, so a class is not a single "
            f"integer — the intermediate-form machinery here assumes a cyclic "
            f"centre.")
    n = inv[0]
    gv = next((w for w in fundamental_coweights(datum)
               if _class_order(datum, w, n, in_coroot_lattice) == n), None)
    if gv is None:
        raise NotImplementedError(
            f"{datum.name}: no single fundamental coweight generates P^∨/Q^∨ "
            f"(order {n}), so the classes cannot be labelled by multiples of one "
            f"of them.")
    g = None
    for w in fundamental_weights(datum):
        if _class_order(datum, w, n, in_root_lattice) != n:
            continue
        u = (Fraction(datum.shift_pairing(gv, w)) * n) % n
        if u.denominator != 1:
            continue
        u = int(u) % n
        inv_u = next((t for t in range(1, n) if (u * t) % n == 1), None)
        if inv_u is None:
            continue
        g = tuple(inv_u * Fraction(x) for x in w)
        break
    if g is None:
        raise NotImplementedError(
            f"{datum.name}: found no fundamental weight of full order pairing "
            f"invertibly with the coweight generator, so the electric classes "
            f"cannot be labelled dually to the magnetic ones.")
    return gv, g


def parity_character(datum):
    """`π: P^∨/Q^∨ → Z/2`, `[m] ↦ ⟨Σ⁺, m⟩ mod 2` — returned as a dict
    `class -> 0/1` (`{0: 0}` for a trivial centre).

    **Provisional name, pending ratification.**  The repo had a *per-charge* claim —
    "odd `⟨Σ⁺, m⟩` needs a forbidden `𝖖^{1/2}`" — but that claim is **retracted**: the phase is only ever needed through its integral coboundary, so
    those charges build.  What `π` names is the object the parity still descends to,
    which is what makes it decidable per *form* rather than charge by charge:

    `⟨Σ⁺, ·⟩` is linear, and `⟨Σ⁺, α_i^∨⟩ = 2` for every simple coroot, so its
    reduction mod 2 kills `Q^∨` and factors through the finite group `P^∨/Q^∨` as a
    genuine **homomorphism**.  Hence the frame question is answered by a character on
    the centre, not charge by charge — see `LineLattice.abe_representable_lattice`.

    Two immediate corollaries, both measured:

    * an **odd** centre order admits no non-trivial map to `Z/2`, so `π ≡ 0` and the
      adjoint form is carried whole — PSU(3), PSU(5) (and `E_6`) never need the
      atom construction at all;
    * at even order the kernel of `π` is an index-2 subgroup, so the charges this
      frame carries are those of an intermediate form: at `SU(4)`, `ker π = {0,2}`,
      which is exactly **SU(4)/Z_2**.

    Generalised to an arbitrary finite abelian centre (2026-08-24).  The classes
    are keyed by an **int** when the centre is cyclic — every pre-existing caller
    reads them that way — and by a class **tuple** otherwise, the same convention
    `LineLattice`'s `H` uses.  The lift is `mag_class_lift`, which goes through
    the Smith labelling rather than through `centre_generators`, so `Spin(4k)`'s
    `Z_2 × Z_2` is covered.

    `⟨Σ⁺, ·⟩` is well defined on classes because `Σ⁺ ∈ Q` and `⟨Q, P^∨⟩ ⊆ Z`, so
    changing the lift by an element of `Q^∨` changes the value by an even
    integer — `⟨Σ⁺, α_i^∨⟩ = 2`."""
    from fractions import Fraction
    divisors = centre_labelling(datum)[0]
    if not divisors:
        return {0: 0}
    sigma_plus = [0] * datum.dim
    for a in datum.positive_roots():
        for i, x in enumerate(a):
            sigma_plus[i] += x
    cyclic = len(divisors) == 1
    out = {}
    for k in _all_classes(divisors):
        m = mag_class_lift(datum, k)
        h = Fraction(datum.shift_pairing(m, tuple(sigma_plus)))
        key = k[0] if cyclic else k
        out[key] = None if h.denominator != 1 else int(h) % 2
    return out


def coweight_class(datum, m):
    """The class of `m` in `P^∨/Q^∨`, as an integer mod the centre order — the
    unique `k` with `m − k·g^∨ ∈ Q^∨` for the generator `g^∨` of
    `centre_generators`, or `None` if `m ∉ P^∨`.

    Computed by *search over the finitely many classes*, each membership decided by
    an exact integer solve.  Deliberately not via a "centre functional": the
    functional `centre_class` guesses (a row of `C^{-1}` with the right denominator)
    is measured **not to annihilate the root lattice** on non-simply-laced data, and
    that produced a wrong answer two steps downstream.  Searching is slower and
    right.

    The generator is **found, not assumed to be `ω_1^∨`** — see `centre_generators`
    for the `Sp(4)` measurement that forced this.

    Cyclic centres only (which covers `A_n`, `B_n`, `C_n`, `E_6`, `E_7`); a product
    centre such as `Spin(4n)`'s `Z_2 × Z_2` honest-fails rather than pretending the
    class is a single integer."""
    from fractions import Fraction
    inv = centre_invariants(datum)
    if not inv:
        return 0
    n = inv[0]
    if not in_coweight_lattice(datum, m):
        return None
    gv, _ = centre_generators(datum)
    d = datum.dim
    for k in range(n):
        diff = tuple(Fraction(m[i]) - k * Fraction(gv[i]) for i in range(d))
        if in_coroot_lattice(datum, diff):
            return k
    return None


def weight_class(datum, e):
    """The class of `e` in `P/Q` — the unique `k` with `e − k·g ∈ Q`, for the
    generator `g` **dual to** the magnetic one (`centre_generators`).

    Same construction as `coweight_class`, on the electric side: search the classes,
    decide each by `in_root_lattice`.  The duality of the two labellings is what
    makes `elec_admits`' annihilator test (`k·h ≡ 0 mod n`) mean anything; the
    previous version subtracted `k` from the *first coordinate*, which is `k·ω_1`
    only when the datum states its roots in the fundamental-weight basis — true for
    `su_n`/`b_n_simply_connected`/`g_2` and **false for `sp_n`**."""
    from fractions import Fraction
    inv = centre_invariants(datum)
    if not inv:
        return 0
    n = inv[0]
    _, g = centre_generators(datum)
    d = datum.dim
    for k in range(n):
        diff = tuple(Fraction(e[i]) - k * Fraction(g[i]) for i in range(d))
        if in_root_lattice(datum, diff):
            return k
    return None


def in_root_lattice(datum, e) -> bool:
    """`e ∈ Q` — is the weight an **integer** combination of the simple roots?

    Solved directly, by integer elimination on `e = Σ c_i α_i`, rather than through
    a centre functional.  That is deliberate: `centre_class` used to pick "any row of
    `C^{-1}` with denominator `n`" as the functional, and such a row need not
    **annihilate the root lattice**, which is the one property a centre functional
    must have.  At `B₂` it selected `(½, 1)`, which pairs with the simple root
    `(−1, 2)` to `3/2 → 3 ≡ 1 (mod 2)`; the genuine roots `(−1,2)` and `(1,0)` were
    therefore graded centre-charged, and the **adjoint representation was refused as
    matter for SO(5)** — though the adjoint is a representation of *every* global
    form, since the centre acts trivially on it.

    Root-lattice membership is what `elec_admits` actually needs, and it is exactly
    decidable, so this replaces the functional at the point of use."""
    from fractions import Fraction
    roots = [tuple(a) for a in datum.simple_roots]
    d = datum.dim
    n = len(roots)
    # columns are the simple roots; solve  M·c = e  over Q, then require c ∈ Z^n
    A = [[Fraction(roots[j][i]) for j in range(n)] + [Fraction(e[i])]
         for i in range(d)]
    row = 0
    piv_cols = []
    for col in range(n):
        p = next((r for r in range(row, d) if A[r][col] != 0), None)
        if p is None:
            continue
        A[row], A[p] = A[p], A[row]
        pv = A[row][col]
        A[row] = [x / pv for x in A[row]]
        for r in range(d):
            if r != row and A[r][col] != 0:
                f = A[r][col]
                A[r] = [x - f * y for x, y in zip(A[r], A[row])]
        piv_cols.append(col)
        row += 1
    for r in range(row, d):                      # inconsistent rows ⇒ not in span
        if A[r][n] != 0:
            return False
    sol = [Fraction(0)] * n
    for i, col in enumerate(piv_cols):
        sol[col] = A[i][n]
    return all(x.denominator == 1 for x in sol)


def in_coweight_lattice(datum, m) -> bool:
    """`m ∈ P^∨` — integral against every root.  (Testing the *simple* roots
    suffices, but every root is cheap and states the property directly.)"""
    from fractions import Fraction
    from wrq_torus import _pairing_root_cochar
    for a in datum.positive_roots():
        if Fraction(_pairing_root_cochar(datum, a, tuple(m))).denominator != 1:
            return False
    return True


# ---------------------------------------------------------------------------
# General centres — labelling `P^∨/Q^∨` and `P/Q` by Smith normal form
# ---------------------------------------------------------------------------
#
# `centre_generators` above labels a class by a single integer, which works
# only when the centre is cyclic; `Spin(4k)`'s `Z_2 × Z_2` honest-fails there.
# The machinery below labels a class by a TUPLE, one entry per elementary
# divisor, and works for every finite abelian centre.
#
# It also removes the search: `centre_generators` *hunts* for a fundamental
# coweight of full order and then for a weight pairing invertibly with it,
# honest-failing when neither exists.  Smith normal form produces both
# labellings at once, and their duality is a THEOREM about the decomposition
# rather than a normalisation imposed afterwards --
#
#     U C V = D  (D diagonal, U/V unimodular)  =>  C^{-1} = V D^{-1} U
#
# so, in the fundamental-(co)weight coordinates where `Q^∨ = C·Z^r ⊆ P^∨ = Z^r`
# and `Q = C^T·Z^r ⊆ P = Z^r`,
#
#     ⟨m, e⟩  =  e^T C^{-1} m  =  Σ_t (V^T e)_t · (U m)_t / d_t          (†)
#
# and the two class maps `m ↦ U·m mod D`, `e ↦ V^T·e mod D` pair to
# `Σ_t k_t l_t / d_t` in `Q/Z` by construction.  Measured against the datum's
# own `shift_pairing` on every fundamental (co)weight pair: 88/88 exact across
# SU(2..5), SO(5), SO(7), Sp(4), G2, Spin(7) and **SO(8)** -- the `Z_2 × Z_2`
# case that has no cyclic labelling at all.


def _smith_with_transforms(M):
    """`(D, U, V)` with `U·M·V = D` diagonal, `U`/`V` unimodular.

    The elementary-divisor-only `_smith_invariants` above cannot be reused: the
    class maps need the *transforms*, since `U` is the magnetic labelling and
    `V^T` the electric one.  Same algorithm, carrying `U` and `V` along."""
    A = [list(map(int, row)) for row in M]
    n = len(A)
    m = len(A[0]) if A else 0
    U = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    V = [[1 if i == j else 0 for j in range(m)] for i in range(m)]

    def rsub(i, k, q):
        A[i] = [x - q * y for x, y in zip(A[i], A[k])]
        U[i] = [x - q * y for x, y in zip(U[i], U[k])]

    def csub(j, k, q):
        for r in A:
            r[j] -= q * r[k]
        for r in V:
            r[j] -= q * r[k]

    def rswap(i, k):
        A[i], A[k] = A[k], A[i]
        U[i], U[k] = U[k], U[i]

    def cswap(j, k):
        for r in A:
            r[j], r[k] = r[k], r[j]
        for r in V:
            r[j], r[k] = r[k], r[j]

    for t in range(min(n, m)):
        piv = None
        for i in range(t, n):
            for j in range(t, m):
                if A[i][j] and (piv is None
                                or abs(A[i][j]) < abs(A[piv[0]][piv[1]])):
                    piv = (i, j)
        if piv is None:
            break
        rswap(t, piv[0])
        cswap(t, piv[1])
        # Each pass either zeroes an off-pivot entry or strictly decreases
        # |A[t][t]|, so the loop terminates.
        while True:
            for i in range(t + 1, n):
                if A[i][t]:
                    rsub(i, t, A[i][t] // A[t][t])
                    if A[i][t]:
                        rswap(t, i)
            if any(A[i][t] for i in range(t + 1, n)):
                continue
            for j in range(t + 1, m):
                if A[t][j]:
                    csub(j, t, A[t][j] // A[t][t])
                    if A[t][j]:
                        cswap(t, j)
            if (not any(A[t][j] for j in range(t + 1, m))
                    and not any(A[i][t] for i in range(t + 1, n))):
                break
        if A[t][t] < 0:
            A[t] = [-x for x in A[t]]
            U[t] = [-x for x in U[t]]

    # Enforce the divisibility chain d_1 | d_2 | ... so the elementary divisors
    # agree with `_smith_invariants` (and with `centre_invariants`).
    changed = True
    while changed:
        changed = False
        for t in range(min(n, m) - 1):
            a, b = A[t][t], A[t + 1][t + 1]
            if a and b % a:
                csub(t, t + 1, -1)
                while A[t + 1][t]:
                    rsub(t + 1, t, A[t + 1][t] // A[t][t])
                    if A[t + 1][t]:
                        rswap(t, t + 1)
                while A[t][t + 1]:
                    csub(t + 1, t, A[t][t + 1] // A[t][t])
                    if A[t][t + 1]:
                        cswap(t, t + 1)
                for s in (t, t + 1):
                    if A[s][s] < 0:
                        A[s] = [-x for x in A[s]]
                        U[s] = [-x for x in U[s]]
                changed = True
    return A, U, V


_CENTRE_LABELLING_CACHE = {}


def centre_labelling(datum):
    """`(divisors, U, V, slots)` — the dual labelling of `P^∨/Q^∨` and `P/Q`.

    * `divisors` — the elementary divisors `> 1`, i.e. `centre_invariants(datum)`;
    * `slots` — which Smith positions those divisors occupy;
    * `U`, `V` — the Smith transforms of the Cartan matrix, so that the magnetic
      class of `m` is `(U·m)_t mod d_t` and the electric class of `e` is
      `(V^T·e)_t mod d_t`, both read in fundamental-(co)weight coordinates.

    The two labellings are dual by construction — identity `(†)` in the comment
    above — which is what makes "the electric class annihilates the magnetic
    one" a meaningful statement.  `centre_generators` achieves the same duality
    for a cyclic centre by searching for a generator and rescaling its partner;
    this derives it, and is not restricted to cyclic centres."""
    key = (datum.name, tuple(tuple(r) for r in cartan_matrix(datum)))
    hit = _CENTRE_LABELLING_CACHE.get(key)
    if hit is not None:
        return hit
    C = cartan_matrix(datum)
    r = len(C)
    D, U, V = _smith_with_transforms(C)
    diag = [D[i][i] for i in range(r)]
    slots = tuple(i for i in range(r) if diag[i] != 1)
    out = (tuple(diag[i] for i in slots), U, V, slots)
    _CENTRE_LABELLING_CACHE[key] = out
    return out


def _coweight_omega_coords(datum, m):
    """`m` in the fundamental-**coweight** basis: `c_i = ⟨α_i, m⟩`.

    Exact by `⟨α_j, ω_i^∨⟩ = δ_ij`, and read through the datum's own
    `shift_pairing`, so no coordinate convention is assumed (the constructors
    disagree — see `fundamental_weights`)."""
    from fractions import Fraction
    return tuple(Fraction(datum.shift_pairing(tuple(m), tuple(a)))
                 for a in datum.simple_roots)


def _weight_omega_coords(datum, e):
    """`e` in the fundamental-**weight** basis: `c_i = ⟨α_i^∨, e⟩`."""
    from fractions import Fraction
    return tuple(Fraction(datum.shift_pairing(tuple(c), tuple(e)))
                 for c in datum.simple_coroots)


def coweight_classes(datum, m):
    """The class of `m` in `P^∨/Q^∨` as a **tuple**, one entry per elementary
    divisor — or `None` if `m ∉ P^∨`.

    The general-centre counterpart of `coweight_class` (which returns a single
    integer and is defined only for a cyclic centre).  `()` when the centre is
    trivial."""
    divisors, U, _V, slots = centre_labelling(datum)
    c = _coweight_omega_coords(datum, m)
    if any(x.denominator != 1 for x in c):
        return None                     # not in `P^∨` at all
    r = len(c)
    return tuple(int(sum(U[t][j] * c[j] for j in range(r))) % d
                 for t, d in zip(slots, divisors))


def weight_classes(datum, e):
    """The class of `e` in `P/Q` as a **tuple**, dual to `coweight_classes` —
    or `None` if `e ∉ P`."""
    divisors, _U, V, slots = centre_labelling(datum)
    c = _weight_omega_coords(datum, e)
    if any(x.denominator != 1 for x in c):
        return None
    r = len(c)
    return tuple(int(sum(V[j][t] * c[j] for j in range(r))) % d
                 for t, d in zip(slots, divisors))


def centre_pairing_classes(datum, k, l):
    """`Σ_t k_t·l_t / d_t` in `Q/Z`, as a `Fraction` in `[0, 1)`.

    The perfect pairing `P^∨/Q^∨ × P/Q → Q/Z` read on the dual labellings.  It
    is the *fractional part* of `⟨m, e⟩` for any lifts `m`, `e`, which is why
    integrality of the Dirac pairing is a statement about classes alone."""
    from fractions import Fraction
    divisors, _U, _V, _slots = centre_labelling(datum)
    tot = sum(Fraction(int(a) * int(b), d)
              for a, b, d in zip(k, l, divisors)) if divisors else Fraction(0)
    return tot - int(tot)


def _as_class_tuple(x, divisors):
    """Normalise a centre class to a tuple, one entry per elementary divisor.

    A bare `int` is accepted when the centre is cyclic (or trivial), which is
    what keeps every pre-2026-08-24 caller — and the `H=` product forms — working
    unchanged."""
    if isinstance(x, int):
        if not divisors:
            return ()
        if len(divisors) != 1:
            raise ValueError(
                f"centre {divisors} is not cyclic, so the class {x} is not a "
                f"single integer — pass a tuple of length {len(divisors)}")
        return (x % divisors[0],)
    t = tuple(int(a) for a in x)
    if len(t) != len(divisors):
        raise ValueError(
            f"class {x} has length {len(t)}, expected {len(divisors)} "
            f"(centre {divisors})")
    return tuple(a % d for a, d in zip(t, divisors))


def _all_classes(divisors):
    """Every element of `⊕_t Z/d_t`, as tuples."""
    out = [()]
    for d in divisors:
        out = [c + (k,) for c in out for k in range(d)]
    return out


def _add_classes(a, b, divisors):
    return tuple((x + y) % d for x, y, d in zip(a, b, divisors))


def _generated_subgroup(gens, divisors):
    """The subgroup of class **pairs** generated by `gens`, materialised.

    Centres are small (`|Z| ≤ N` for `SU(N)`, `4` for `Spin(4k)`), so the whole
    subgroup fits comfortably in memory and membership becomes a set lookup."""
    zero = (tuple(0 for _ in divisors), tuple(0 for _ in divisors))
    seen = {zero}
    frontier = [zero]
    while frontier:
        cur = frontier.pop()
        for (gk, gl) in gens:
            nxt = (_add_classes(cur[0], gk, divisors),
                   _add_classes(cur[1], gl, divisors))
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return frozenset(seen)


def _product_class_pairs(datum, H, divisors):
    """The class pairs of the **product** form cut out by a centre subgroup `H`.

    `H` is the magnetic half — a set of classes, given as ints for a cyclic
    centre — and the electric half is its annihilator for the linking pairing.
    This reproduces the pre-2026-08-24 `mag_admits` / `elec_admits` exactly:
    `H` trivial gives `Q^∨ × P` (simply connected), `H` the whole centre gives
    `P^∨ × Q` (adjoint), and an intermediate `H` gives the intermediate form."""
    zero = tuple(0 for _ in divisors)
    if not divisors:
        return frozenset({(zero, zero)})
    mag_gens = [_as_class_tuple(h, divisors) for h in H] or [zero]
    mags = _generated_subgroup([(g, zero) for g in mag_gens], divisors)
    mag_set = {k for (k, _l) in mags}
    from fractions import Fraction
    elec_set = [l for l in _all_classes(divisors)
                if all(Fraction(centre_pairing_classes(datum, k, l)) % 1 == 0
                       for k in mag_set)]
    return frozenset((k, l) for k in mag_set for l in elec_set)


def is_semisimple(datum) -> bool:
    """Whether the datum has no central torus — `#simple roots == dim`.

    `U(N)` is the standing counterexample in this repo, and it matters here:
    its cocharacter lattice `Z^N` surjects onto `P^∨/Q^∨` (the minuscule
    monopole `(1,0)` has centre class 1 and is nonetheless an ordinary `U(2)`
    line), so a class filter on the magnetic side would reject genuine lines.
    A datum with a central torus therefore keeps the pre-2026-08-24 integral
    tests, which are the correct ones for it."""
    return len(datum.simple_roots) == datum.dim


def mag_class_lift(datum, k):
    """A coweight of centre class `k`, in the **datum's own** coordinates.

    Well defined up to `Q^∨`, which is integral in those coordinates, so
    "`m − lift` is integral" does not depend on the choice."""
    from fractions import Fraction
    divisors, U, _V, slots = centre_labelling(datum)
    r = len(cartan_matrix(datum))
    if not divisors:
        return tuple(Fraction(0) for _ in range(datum.dim))
    Uinv = _integer_inverse(U)
    y = [0] * r
    for t, slot in enumerate(slots):
        y[slot] = int(k[t])
    coeffs = [sum(Uinv[i][j] * y[j] for j in range(r)) for i in range(r)]
    fcw = fundamental_coweights(datum)
    return tuple(sum(Fraction(coeffs[i]) * Fraction(fcw[i][d])
                     for i in range(r)) for d in range(datum.dim))


def _integer_inverse(M):
    """Inverse of a unimodular integer matrix."""
    from fractions import Fraction
    n = len(M)
    A = [[Fraction(M[i][j]) for j in range(n)]
         + [Fraction(1 if i == k else 0) for k in range(n)] for i in range(n)]
    for c in range(n):
        piv = next((i for i in range(c, n) if A[i][c] != 0), None)
        if piv is None:
            raise ValueError("singular matrix")
        A[c], A[piv] = A[piv], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for i in range(n):
            if i != c and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[c])]
    return [[int(A[i][n + j]) for j in range(n)] for i in range(n)]


def magnetic_classes_of(datum, m):
    """Every centre class `k` with `m ∈ M_0 + lift(k)`, `M_0` the datum's own
    integral cocharacter lattice.

    Usually a single class (or none).  For a datum with a central torus it can
    be several — at `U(2)` the lattice already meets every class — which is
    exactly why the magnetic side is tested by **coset** rather than by class:
    classes *grow* `Q^∨` towards `P^∨`, and a lattice that already reaches a
    class must not be cut back by it."""
    from fractions import Fraction
    divisors, _U, _V, _slots = centre_labelling(datum)
    integral = all(Fraction(x).denominator == 1 for x in m)
    if not divisors:
        return [()] if integral else []
    if not is_semisimple(datum):
        # A central torus makes the cocharacter lattice `M_0` meet every class
        # (at `U(N)` the minuscule monopole already has class 1), so
        # `M_0 + lift(k) = M_0` for every `k` and the coset test degenerates to
        # integrality — which is precisely the pre-2026-08-24 behaviour, and the
        # correct one for such a datum.  `fundamental_coweights` is not even
        # defined here, so this branch is required, not merely an optimisation.
        return list(_all_classes(divisors)) if integral else []
    out = []
    for k in _all_classes(divisors):
        lift = mag_class_lift(datum, k)
        if all(Fraction(m[d]) - lift[d] == int(Fraction(m[d]) - lift[d])
               for d in range(datum.dim)):
            out.append(k)
    return out


class LineLattice:
    """A 4d global form as a **sublattice of Kapustin `(m, e)` labels**.

    `admits(m, e)` is membership; `dirac` is the Dirac pairing whose integrality
    is mutual locality.  The default (`H` trivial) is the **simply connected**
    form, which is what `PureGAbeKAlgebra` realises: in the repo's coordinates
    cocharacters are stored in the *coroot* basis (`Q^∨`) and weights in the
    fundamental-weight basis (`P`), and `⟨Q^∨, P⟩ ⊆ Z` — mutually local, as it
    must be.

    **Which forms this class covers** (updated 2026-07-29).  A form
    with `H ≠ 1` needs magnetic charges in `P^∨ ⊋ Q^∨`, i.e. *fractional*
    coordinates in the coroot basis, and those are presented on the coweight torus:
    PSU(3) at both fractional coweights, PSU(4) at `ω_2^∨`, PSU(5) at all four, the
    non-simply-laced Sp(6)/Z₂, the **intermediate** SU(4)/Z_2 (self-dual), and
    PSU(3) with adjoint matter.

    `admits(m, e)` decides **membership** — whether the label is a line of *this*
    form; `abe_representable(m)` decides **which frame presents it**, and that is a
    routing fact about presentations, *not* a claim that a line cannot be built.  **The coweight torus carries
    them all**, odd `⟨Σ⁺, m⟩` included — PSU(2), PSU(4) `ω_1^∨`/`ω_3^∨`, SO(5),
    Sp(4)/Z₂, SO(7) — because the half-integral atom phase reaches the cocycle only
    through its integral coboundary.  `abe_representable` is therefore `True`
    everywhere; what the parity still decides is whether the **default** phase
    convention needs that correction.  (The standard-`Z²` BPS chart at SU(2)/SO(3),
    `src/gn/pure_so3.py`, remains a correct independent presentation — it is
    the oracle the fix was certified against — but it is no longer *needed* here.)"""

    def __init__(self, datum, H=(), classes=None, name=None):
        """`H` — a subgroup of the centre, giving the **product** lattice
        `{m : [m] ∈ H} × {e : [e] ⊥ H}`; `classes` — generators of an arbitrary
        subgroup of centre-class **pairs**, for a lattice that need not be a
        product.

        Exactly one of the two says what the lattice is (`H` is the default, and
        `H=()` is the simply connected form).  `classes` is the general case
        directed by the author (2026-08-24): *"the label space of `L_{m,e}` should
        be the Weyl quotient of a lattice which is a sublattice of coweights x
        weights containing coroots x roots"*, with the admissibility condition
        *"as long as the pairing is integral"*.

        Each generator is a pair `(k, l)` of class tuples in the labelling of
        `coweight_classes` / `weight_classes`; a bare `int` is accepted for a
        cyclic centre.  The subgroup they generate is materialised (centres are
        small), so membership is a set lookup.

        **Maximality is not required** — a non-maximal
        isotropic subgroup is a consistent, merely incomplete, set of lines, and
        `is_maximal()` reports rather than enforces.  Integrality of the Dirac
        pairing is likewise *reported* by `verify_dirac_integral`, not imposed:
        a caller may build an inadmissible lattice and be told so."""
        self.datum = datum
        # `H` entries are ints for a cyclic centre (every pre-2026-08-24
        # caller) and class tuples for a general one.
        if not H:
            self.H = ()
        elif all(isinstance(h, int) for h in H):
            self.H = tuple(sorted(set(int(h) for h in H)))
        else:
            self.H = tuple(sorted(set(tuple(int(x) for x in h) for h in H)))
        self.invariants = centre_invariants(datum)
        self.divisors = centre_labelling(datum)[0]
        if classes is not None and H:
            raise ValueError(
                "LineLattice: give `H` (a product form) or `classes` (a general "
                "class-pair subgroup), not both")
        if classes is not None:
            gens = [(_as_class_tuple(k, self.divisors),
                     _as_class_tuple(l, self.divisors)) for (k, l) in classes]
            self._pairs = _generated_subgroup(gens, self.divisors)
            self.name = name or f"{datum.name}[{len(self._pairs)} classes]"
        else:
            self._pairs = _product_class_pairs(self.datum, self.H, self.divisors)
            self.name = name or (
                f"{datum.name}"
                + ("" if not self.H else f"/Z_{self.H}"))

    # ----- the class-pair subgroup ----------------------------------------
    def class_pairs(self) -> frozenset:
        """The finite subgroup `Λ / (Q^∨ × Q)` of centre-class pairs.

        This *is* the lattice: `Λ` is its preimage in `P^∨ × P`, which is why a
        finite set determines an infinite lattice.  Every such preimage contains
        `Q^∨ × Q` by construction, so the containment the author requires is
        structural rather than checked."""
        return self._pairs

    def label_classes(self, m, e):
        """`([m], [e])` as class tuples, or `None` if `(m, e) ∉ P^∨ × P`.

        Informational.  Membership goes through `admits`, which tests the
        magnetic side by coset rather than by this class — the two differ when
        the datum has a central torus (`magnetic_classes_of`)."""
        k = coweight_classes(self.datum, tuple(m))
        if k is None:
            return None
        l = weight_classes(self.datum, tuple(e))
        if l is None:
            return None
        return (k, l)

    def verify_dirac_integral(self) -> bool:
        """The author's admissibility condition, made executable: the Dirac
        pairing is integral on the whole lattice.

        Checked on **classes**, not on sampled labels, which makes it a proof
        rather than a spot check: `⟨Q^∨, P⟩ ⊆ Z` and `⟨P^∨, Q⟩ ⊆ Z`, so the
        fractional part of `⟨m, e'⟩ − ⟨m', e⟩` depends only on the four classes.
        Hence integrality on the lattice **is** isotropy of the finite subgroup,
        and the finite check is exhaustive."""
        from fractions import Fraction
        pairs = list(self._pairs)
        for (k, l) in pairs:
            for (k2, l2) in pairs:
                val = (centre_pairing_classes(self.datum, k, l2)
                       - centre_pairing_classes(self.datum, k2, l))
                if Fraction(val) % 1 != 0:
                    return False
        return True

    def is_maximal(self) -> bool:
        """Whether the lattice is a genuine global form — maximal among mutually
        local ones, i.e. the class subgroup is Lagrangian for the linking pairing.

        The order of a Lagrangian subgroup of `Z × Ẑ` is `|Z|`, so this is a
        counting test.  Reported, never required: see `__init__`."""
        order = 1
        for d in self.divisors:
            order *= d
        return len(self._pairs) == order

    def is_product(self) -> bool:
        """Whether the lattice is a product `(magnetic) × (electric)` — the
        *traditional* global forms.

        False exactly for the correlated lattices — the ones carrying a discrete
        theta angle, where a purely magnetic and a purely electric line may both
        be absent although their sum is a line."""
        ks = {k for (k, _l) in self._pairs}
        ls = {l for (_k, l) in self._pairs}
        return len(self._pairs) == len(ks) * len(ls)

    # ----- the pairing -----------------------------------------------------
    def dirac(self, label, other) -> int:
        """`⟨(m,e), (m',e')⟩ = ⟨m, e'⟩ − ⟨m', e⟩` — the Dirac pairing on
        Kapustin labels.  Mutual locality of two lines is its integrality (here
        the coordinates are integral, so it is automatically an integer; the
        check bites once fractional coweights enter)."""
        (m, e), (m2, e2) = label, other
        return (self.datum.shift_pairing(m, e2)
                - self.datum.shift_pairing(m2, e))

    def verify_mutually_local(self, labels) -> bool:
        """Every pair of lines in the lattice pairs integrally — the defining
        property of a global form's line sublattice."""
        labels = list(labels)
        return all(_is_integral(self.dirac(a, b))
                   for a in labels for b in labels)

    # ----- membership ------------------------------------------------------
    def mag_admits(self, m) -> bool:
        """The **purely magnetic** lines: `(m, 0) ∈ Λ`.

        Generalised 2026-08-24 as the fibre of `admits` over `e = 0`; on every
        product form it returns exactly what it did before — `Q^∨` at the simply
        connected form, `P^∨` at the adjoint form, the class filter at an
        intermediate one.  The fibre, not the projection `{m : ∃e, (m,e) ∈ Λ}`:
        the fibre is the cocharacter lattice of the gauge group (charges of
        genuine 't Hooft lines), while the projection also counts the magnetic
        charge of a dyon whose own 't Hooft line is not a line.  A discrete theta
        angle is exactly where the two differ.

        Tested by **coset** (`m ∈ M_0 + lift(k)`), not by class — see
        `magnetic_classes_of` for why `U(N)` forces that."""
        zero = tuple(0 for _ in self.divisors)
        return any((k, zero) in self._pairs
                   for k in magnetic_classes_of(self.datum, tuple(m)))

    def elec_admits(self, e) -> bool:
        """The **purely electric** lines: `(0, e) ∈ Λ` — the character lattice of
        the gauge group, i.e. which representations are representations *of this
        form*.

        The `e = 0` mirror of `mag_admits`, but tested by **class**: weights are
        stored in the fundamental-weight basis, so `P` is the ambient lattice and
        the classes *cut it down* towards `Q` — the opposite direction to the
        magnetic side, where they grow `Q^∨` towards `P^∨`.  `P` at the simply
        connected form, `Q` at the adjoint form (so the adjoint representation is
        admitted at every form, as it must be), the annihilator of `H` at an
        intermediate one — unchanged on every product form.

        This is the reading its callers need: `gn_abe_kalgebra` and
        `star_bubbling` ask it whether a matter weight is a weight of the gauge
        group, a question about the `m = 0` fibre and not about the projection."""
        l = weight_classes(self.datum, tuple(e))
        if l is None:
            return False
        return (tuple(0 for _ in self.divisors), l) in self._pairs

    def _centre_pairing(self, e, h) -> int:
        """The `Z`-valued pairing of a weight class with a centre element.
        With `Z` cyclic of order `n` generated by the image of a fundamental
        weight, this is `⟨e⟩·h mod n` read off the weight's centre class."""
        n = self.invariants[0] if self.invariants else 1
        return (self.centre_class(e) * h) % n if n > 1 else 0

    def verify_centre_grading_kills_roots(self) -> bool:
        """The defining sanity check on any `P/Q` grading: it must vanish on **every
        root**, since the roots generate `Q`.

        Kept as an explicit verifier because this is exactly what silently broke.
        `centre_class`'s functional failed it at `B₂` — grading the roots `(−1,2)`
        and `(1,0)` as class 1 — and the visible symptom was two steps away: the
        adjoint representation being refused as matter for SO(5).  A one-line
        invariant here would have caught it at the source."""
        return all(in_root_lattice(self.datum, tuple(a))
                   for a in self.datum.positive_roots())

    def centre_class(self, e) -> int:
        """The class of a weight in `P/Q` (as an integer mod the first
        elementary divisor; `0` when the centre is trivial).

        ⚠ **Measured unreliable on non-simply-laced data** (2026-07-28) and no longer
        used by `elec_admits`, which decides root-lattice membership directly via
        `in_root_lattice`.  The functional below is chosen as "any row of `C^{-1}`
        with denominator `n`", and such a row need not annihilate `Q` — at `B₂` it
        does not, so genuine roots come back with class 1.  Use `in_root_lattice`
        for membership; treat this as informational until the functional is derived
        properly (Smith normal form of `C`)."""
        if not self.invariants:
            return 0
        n = self.invariants[0]
        C = cartan_matrix(self.datum)
        # solve  e ≡ Σ c_i · (row i of C)  mod n  is not needed: the class is
        # determined by the linear functional dual to the centre generator.
        # For the cyclic case use the standard "comarks" functional: the class
        # of ω_i is the i-th entry of the inverse Cartan's last row scaled to Z.
        from fractions import Fraction
        r = len(C)
        # invert C over Q (small r)
        A = [[Fraction(C[i][j]) for j in range(r)] + [Fraction(1 if i == k else 0)
             for k in range(r)] for i in range(r)]
        for col in range(r):
            piv = next(i for i in range(col, r) if A[i][col] != 0)
            A[col], A[piv] = A[piv], A[col]
            pv = A[col][col]
            A[col] = [x / pv for x in A[col]]
            for i in range(r):
                if i != col and A[i][col] != 0:
                    f = A[i][col]
                    A[i] = [x - f * y for x, y in zip(A[i], A[col])]
        # the centre functional: any row of C^{-1} with denominator n
        func = None
        for i in range(r):
            row = A[i][r:]
            if any(x.denominator == n for x in row):
                func = row
                break
        if func is None:
            return 0
        val = sum(Fraction(int(e[j])) * func[j] for j in range(r))
        return int((val * n) % n)

    def admits(self, m, e) -> bool:
        """Whether `(m, e)` is a line of this lattice.

        ⚠ **This is not `mag_admits(m) and elec_admits(e)`** once the lattice is
        allowed to be non-product (2026-08-24).  It used to be defined as that
        conjunction, which is correct exactly for the traditional product forms;
        a correlated lattice can contain `(m, e)` while containing neither
        `(m, 0)` nor `(0, e)`.  The two side predicates are now the *fibres* of
        this one (see `mag_admits`), so on a product form all three agree with
        the old behaviour, label for label."""
        l = weight_classes(self.datum, tuple(e))
        if l is None:
            return False
        return any((k, l) in self._pairs
                   for k in magnetic_classes_of(self.datum, tuple(m)))

    # ----- the DEFINING property ------------------------
    #
    # "4d gauge group data is a maximal set of mutually compatible `(m, e)`
    # labels, Kapustin style."  `verify_mutually_local` above is the
    # *compatibility* half; this is the **maximality** half, which the module
    # previously only asserted in a comment.

    def verify_lattices_are_dual(self, mags, eles) -> bool:
        """The two charge lattices are **dual to each other** — the sharp form of
        "a maximal set of mutually compatible Kapustin `(m, e)` labels".

        The author's framing (2026-07-29), correcting a story this module had been telling
        itself: a 4d gauge group's charge data is simply *a lattice and its dual*.
        With `L` the character lattice of the form, the magnetic lattice is `L^*`:

            SU(2) = (dual to weights, weights) = (Q^∨, P),
            SO(3) = (dual to roots,   roots)   = (P^∨, Q).

        Concretely this checks, on the two finite samples given,

            mag_admits(m)   ⟺   ⟨m, e⟩ ∈ Z for every admitted e,
            elec_admits(e)  ⟺   ⟨m, e⟩ ∈ Z for every admitted m,

        so it constrains the predicates in **both** directions: `verify_maximal`
        can only refute over-smallness, while this also catches over-admission.

        Measured `True` both ways on 16 forms (2026-07-29) — SU(2)/SO(3),
        SU(3)/PSU(3), SU(4)/SU(4)/Z_2/PSU(4), Spin(5)/SO(5), Sp(4)/Sp(4)/Z_2,
        Sp(6)/Sp(6)/Z_2, Spin(7)/SO(7), G₂ — with no rank or type special cases.

        Finite-sample, like every `verify_*` here: a failure is a genuine
        counterexample, a pass is support over the sample.  Samples should be
        integer combinations of the fundamental (co)weights, so that a
        non-simply-connected form's fractional coweights are actually in range."""
        from fractions import Fraction
        mags = [tuple(m) for m in mags]
        eles = [tuple(e) for e in eles]
        adm_m = [m for m in mags if self.mag_admits(m)]
        adm_e = [e for e in eles if self.elec_admits(e)]

        def integral(m, e):
            return Fraction(self.datum.shift_pairing(m, e)).denominator == 1

        return (all(self.mag_admits(m) == all(integral(m, e) for e in adm_e)
                    for m in mags)
                and all(self.elec_admits(e) == all(integral(m, e) for m in adm_m)
                        for e in eles))

    def verify_maximal(self, outside, inside) -> bool:
        """Maximality: no label **outside** the lattice is mutually compatible
        with all of it.

        A global form is a *maximal* mutually-compatible (Dirac-isotropic) set of
        Kapustin `(m, e)` charges, so being isotropic is not enough — an
        isotropic sublattice that could still be enlarged is not a global form.
        This checks the second half on a finite sample: every candidate in
        `outside` must pair **non-integrally** with at least one label of
        `inside`, which is what forbids adjoining it.

        Finite-sample, like every other `verify_*` in the repo: it can refute
        maximality outright (one candidate compatible with everything is a
        counterexample) but only supports it over the sample given.  The
        candidates must genuinely lie outside — `admits` is used to skip any that
        do not, so a caller passing only interior labels gets a vacuous `True`,
        and `verify_maximal_is_nonvacuous` below says whether that happened.

        For a simply connected form in the repo's coordinates the interesting
        candidates are the **fractional coweights** (`P^∨ ⊋ Q^∨`) — exactly the
        charges the adjoint form would add — since all integral labels pair
        integrally by construction."""
        inside = [(tuple(m), tuple(e)) for m, e in inside]
        for m, e in outside:
            if self.admits(m, e):
                continue                     # not actually outside; skip
            if all(_is_integral(self.dirac((m, e), lab))
                   for lab in inside):
                return False                 # could be adjoined ⇒ not maximal
        return True

    def verify_maximal_is_nonvacuous(self, outside) -> bool:
        """Whether `verify_maximal`'s sample contained any genuinely outside
        label — guards against a vacuous pass."""
        return any(not self.admits(m, e) for m, e in outside)

    # ----- which lines this *tier* can hold --------------------------------
    def abe_representable(self, m) -> bool:
        """Whether a magnetic charge can be an atom of the abelianized
        (WRQTorus) tier — i.e. whether the datum admits an **integer** atom
        phase at `m`.  Asked of the datum (`atom_phase`), not re-derived.

        For the *default* root-system phase `−½⟨Σ⁺,m⟩` that means `⟨Σ⁺, m⟩` must
        be **even**.  But a datum may supply its own integer phase and then odd
        `⟨Σ⁺,m⟩` is no obstruction: `u_n` does exactly this (the historical U(N)
        `Σ_j j·m_j`, which differs from the default by a *linear* central term —
        a coboundary that cancels in the cocycle), so U(2) at `m=(1,0)` has odd
        `⟨Σ⁺,m⟩ = 1` and is perfectly representable.  Testing the parity instead
        of the datum wrongly rejected the entire pure-U(N) keystone — caught by
        the battery.

        ⚠ **SINCE 2026-07-29 THIS RETURNS `True` EVERYWHERE, and
        the "theorem" recorded below is RETRACTED.**  The argument was that the
        effective (Weyl-symmetrized) phase `g` is pinned to `−½⟨Σ⁺,m⟩`, which is
        not an integer at odd `⟨Σ⁺,m⟩`, so the tier cannot carry those charges —
        and that simulating `𝖖^{1/2}` fails because the line then differs from
        the canonical by `−𝖖^{−1}` (an `i` as well as a `𝖖^{1/2}`).  Both
        measurements reproduce; the **inference does not**.  The tier never needs
        `g` as a number: the cocycle sees only the coboundary `δS̃`, which is an
        INTEGER even where `S̃` is a half-integer, because `π = ⟨Σ⁺,·⟩ mod 2` is
        Weyl-invariant *and* additive.  `wrq_torus.cocycle_R` now supplies the
        honest phase back through `(−𝖖)^{δ(S_honest − S_used)}`, so SO(3)'s
        spinorial `H_0` builds bar-invariantly with `H_0² = L_{(2,0)}` and
        `I(H_0,H_0)` equal to the `PureSO3KAlgebra` BPS oracle, and SO(5)/SO(7)
        odd charges build and are orthonormal — with no `𝖖^{1/2}` and no `i`
        anywhere, so the standing no-`Z[𝖖^{±1/2}]`-chart ruling is untouched.
        Battery: a probe in the source repository.

        The method is kept because it is still the right *question* — "does the
        private ψ need a square root this form's torus lacks", i.e. is `ρ` a
        character of the form — and because an explicit `phase_is_canonical=`
        override on a probe datum still routes through it.

        **The obstruction is entirely MAGNETIC.**  `S(m) = −⟨ρ, m⟩` is a function
        of the cocharacter alone; the electric charge `e` never enters it — it
        only sets the residual `f_m = χ_e`, a `𝖖`-free Laurent polynomial in `v`.
        Measured: Spin(5)'s **spinor Wilson line** `χ_{ω₂}` (the 4) builds
        perfectly, bar-invariant, 4 weights.  What fails is a *'t Hooft* charge
        `m ∈ P^∨ ∖ Q^∨` — a coweight that is not a coroot.  (`pure_so3.py` calls
        such a line "spinorial" because magnetic charges of `G` are **weights of
        the GNO-dual group**, and `ω^∨` is the dual SU(2)'s spinor weight; the
        unambiguous phrasing is "magnetic charge outside `Q^∨`".)  Consequence
        for a global form `G̃/H`: the *electric* condition (`e` annihilates `H`)
        costs nothing, and the whole difficulty is the *magnetic* enlargement.

        What *does* carry those lines is a presentation that is **not the atom
        construction** — at SU(2)/SO(3) the standard-`Z²` BPS chart
        (`src/gn/pure_so3.py`, nodes `(2,0),(−2,1)`), a `BPSKAlgebra`, i.e.
        a different tier, where the quiver nodes are the charge basis and the atom
        phase never enters.

        Note what is **not** going on there.  Nothing
        is rescaled, and no charge is fractional: the charge data of a form is just
        **a lattice and its dual** (`verify_lattices_are_dual`), so SO(3)'s minimal
        't Hooft is the ordinary generator of `P^∨`.  The halves appear only because
        this module *writes* both forms in the simply-connected one's coordinates,
        where the index-2 inclusion `Q ⊂ P` reads as "magnetic doubled, electric
        halved".  That is a description of the two lattices, not a construction, and
        there is nothing in it to carry to higher rank.

        **So this predicate is about the frame, never about the line**.  Every label the lattice `admits` is a
        line of the theory and has a presentation.  The coweight torus of
        the abelianized tier presents them **all**, odd `⟨Σ⁺,m⟩` included (the
        half-integral atom phase is restored through its integral coboundary), so
        this predicate is `True` everywhere and the old "even height here, a
        non-atom presentation for odd height" split is retracted.  Reading a
        `False` — from this or from `abe_representable_lattice` — as "this line
        cannot be built" is the error the author's principle forbids."""
        # Asked of the datum via `atom_phase_is_canonical`, NOT by catching an
        # exception from `atom_phase` and NOT by testing the parity of `⟨Σ⁺,m⟩`
        # here.  Both of those were wrong (2026-07-29):
        #   * `atom_phase` is now TOTAL — it applies the parity correction `ε` and
        #     never raises, because a convention has no business producing
        #     `𝖖^{1/2}`.  So there is no exception left to catch.
        #   * a parity test would refuse `u_n`: `U(2)` at `m=(1,0)` has odd
        #     `⟨Σ⁺,m⟩ = 1` yet builds on its own certified convention.  The
        #     question is whether THIS datum's convention needed a correction.
        return self.datum.atom_phase_is_canonical(tuple(m))

    def abe_representable_lattice(self) -> bool:
        """Does this form's **default** atom phase need no coboundary correction — i.e.
        is the private ψ a character of this form's torus?  Decided from the centre
        alone; no charge is built, and none has to be.

        ⚠ **This is NO LONGER the same question as `abe_representable`, and the two deliberately disagree.**  The per-charge predicate is `True`
        everywhere (the honest half-integral phase is restored at the cocycle through
        its integral coboundary), so a `False` here does **not** mean any line is
        outside the tier — it names the forms on which `wrq_torus.cocycle_R`'s
        correction actually fires.  The old phrasing, "does the coweight-torus frame
        present every line of this form", is retracted; reading a `False` as
        a line the tier cannot carry is an error.

        It collapses to the centre because `⟨Σ⁺, ·⟩ mod 2` is a **character**
        `π: P^∨/Q^∨ → Z/2` (`parity_character`).  The magnetic charges of `G̃/H` are
        the classes in `H`, so:

            the default phase suffices on all of `G̃/H`  ⟺  π|_H ≡ 0.

        Measured against 17 forms (2026-07-29, the suite in the source repository),
        agreeing with the build/honest-fail verdict in every one: SU(2), SU(3),
        PSU(3), SU(4), **SU(4)/Z_2**, PSU(5), Spin(5), Sp(4), Sp(6), Sp(6)/Z_2,
        Spin(7), G₂ carried whole; SO(3), PSU(4), SO(5), Sp(4)/Z_2, SO(7) not.

        A `False` therefore names the forms whose **phase convention** needs the coboundary correction, not forms with lines the tier cannot build — every
        admitted charge of every form in that list is carried (asserted per form in
        the suite in the source repository)."""
        pi = parity_character(self.datum)
        return all(pi.get(k) == 0 for k in (self.H or (0,)))

    def legal_lines(self, labels) -> list:
        """The sublattice of `labels` **this frame** presents — historically the
        `m`-even ("legal") span in the sense of the standing ruling for SU(2)
        (the design notes: work in the legal lattice, with odd magnetic
        singletons dropped or paired into it).

        ⚠ **This filter removes nothing**, because it filters on
        `abe_representable`, which is now `True` for every admitted charge — the
        odd-height lines are ordinary atoms of this torus.  That is deliberate and
        pinned (the suite in the source repository, "legal_lines no longer removes
        odd-`⟨Σ⁺,m⟩` lines"); the method is kept because `admits` still filters, and
        because the C3/C4 *convention* remains a meaningful thing to ask for — but
        it must no longer be read as "the buildable ones"."""
        return [(m, e) for (m, e) in labels
                if self.admits(m, e) and self.abe_representable(m)]

    # ----- Langlands duality on the 4d gauge group data -------------------
    def langlands_dual(self) -> "LineLattice":
        """`L^∨` — the Langlands dual 4d gauge group data.

        It does, and this is the natural home for it.  `S: (m, e) ↦ (e, −m)` acts on
        the Kapustin label lattice and is **symplectic** for the Dirac pairing, so it
        carries a maximal set of mutually compatible labels to another one — i.e. a
        `LineLattice` to a `LineLattice`.  In the "Lagrangian sublattice of
        `Z × Z`" reading the statement is immediate: a symplectic map preserves
        Lagrangian-ness, and maximality is exactly Lagrangian-ness here.

        Concretely `S` exchanges the magnetic and electric lattices, so it exchanges
        the two extreme forms:

            simply connected  (H = 1,  magnetic Q^∨, electric P)
                 ↕
            adjoint           (H = Z,  magnetic P^∨, electric Q)

        which is `SU(N) ↔ PSU(N)` and `SU(2) ↔ SO(3)`.  The *datum* moves too in
        general (`langlands_dual_datum`, the identity for simply-laced `G`); the
        charge dictionary is `langlands_label_map`.

        For an **intermediate `H`** the dual is the annihilator
        `H^⊥ = {k : k·h ≡ 0 ∀ h ∈ H}` under the centre pairing — the electric
        condition `elec_admits` imposes, which is what `S` turns into the magnetic
        one.  So `SU(4)/Z_2` is **self-dual**, `H = {0,2} ⊂ Z_4`
        being its own annihilator."""
        if not self.invariants:
            # trivial centre ⇒ a single global form, necessarily self-dual (G₂, F₄, E₈)
            return self
        # GENERAL CASE (2026-08-24): `S` exchanges magnetic and electric, so on
        # class pairs it is simply the SWAP `(k, l) ↦ (l, k)` — which is defined
        # for a correlated lattice and a non-cyclic centre alike, where the
        # `H`-and-annihilator algebra below is not.  Isotropy is preserved
        # because the linking pairing is antisymmetric under the swap, so the
        # image is again a legal lattice; and the swap is an involution, which
        # is `S² = 1` at the level of the lattice.
        if (not self.is_product()) or len(self.divisors) > 1:
            dual_datum = langlands_dual_datum(self.datum)
            if centre_labelling(dual_datum)[0] != self.divisors:
                raise NotImplementedError(
                    f"{self.name}: the Langlands dual datum has centre "
                    f"{centre_labelling(dual_datum)[0]}, not {self.divisors}, so "
                    f"the class-pair swap is not a map between their class "
                    f"groups.")
            swapped = [(l, k) for (k, l) in sorted(self.class_pairs())]
            return LineLattice(dual_datum, classes=swapped,
                               name=f"{self.name} (Langlands dual)")
        full = tuple(range(self.invariants[0]))
        if not self.H:
            return adjoint_lines(langlands_dual_datum(self.datum))
        if set(self.H) == set(full):
            return simply_connected_lines(langlands_dual_datum(self.datum))
        # INTERMEDIATE H: the dual is the ANNIHILATOR `H^⊥ = {k : k·h ≡ 0 ∀h∈H}`
        # under the centre pairing.  Note `SU(4)/Z_2` is therefore SELF-DUAL
        # `H = {0,2} ⊂ Z_4` and `H^⊥ = {k : 2k ≡ 0 mod 4} =
        # {0,2} = H`.
        n = self.invariants[0]
        perp = tuple(k for k in range(n) if all((k * h) % n == 0
                                                for h in self.H))
        return LineLattice(langlands_dual_datum(self.datum), H=perp,
                           name=f"{self.datum.name} (H^⊥ of {self.H})")

    def verify_langlands_involutive(self, labels) -> bool:
        """`S² = −1`: the dual of the dual is this form back, and the charge map
        composed with itself is charge conjugation `(m, e) ↦ (−m, −e)`.

        Both halves are checked because they are independent statements — the first
        about the *lattice*, the second about the *coordinates* — and `S⁴ = 1` in
        `SL(2, Z)` requires them to agree."""
        back = self.langlands_dual().langlands_dual()
        if back.H != self.H or back.datum.name != self.datum.name:
            return False
        fwd, _ = langlands_label_map(self.datum)
        for (m, e) in labels:
            m1, e1 = fwd(tuple(m), tuple(e))
            m2, e2 = fwd(m1, e1)
            if (m2, e2) != (tuple(-x for x in m), tuple(-x for x in e)):
                return False
        return True

    def verify_langlands_maps_lattice(self, labels) -> bool:
        """Every label this form admits maps to one the **dual** form admits — the
        statement that `S` really does carry `L` into `L^∨` and not merely into some
        larger lattice.  Vacuously true on an empty sample, so pair it with
        `labels` drawn from `admits`."""
        dual = self.langlands_dual()
        fwd, _ = langlands_label_map(self.datum)
        for (m, e) in labels:
            if not self.admits(tuple(m), tuple(e)):
                continue
            m1, e1 = fwd(tuple(m), tuple(e))
            if not dual.admits(m1, e1):
                return False
        return True

    def __repr__(self) -> str:
        z = "trivial" if not self.invariants else "×".join(
            f"Z_{d}" for d in self.invariants)
        return (f"LineLattice({self.name}: centre {z}, "
                f"H={self.H or '1'})")


def simply_connected_lines(datum) -> LineLattice:
    """The simply connected form: magnetic `Q^∨`, electric `P`.  This is what
    `PureGAbeKAlgebra` realises in the repo's coordinates."""
    return LineLattice(datum, H=(), name=f"{datum.name} (simply connected)")


def adjoint_lines(datum) -> LineLattice:
    """The adjoint form: magnetic `P^∨`, electric `Q` — the **Langlands dual** of
    the simply connected one (`SU(2) ↔ SO(3)`, `SU(N) ↔ PSU(N)`), which is what
    `LineLattice.langlands_dual` returns.

    Presented on the coweight torus — PSU(3), PSU(4) `ω_2^∨`,
    PSU(5), Sp(6)/Z₂ — and the odd-`⟨Σ⁺,m⟩` charges too (SO(3), PSU(4)
    `ω_1^∨`/`ω_3^∨`, SO(5), Sp(4)/Z₂, SO(7)), so this form is carried whole.

    ⚠ **COORDINATES: the extra cocharacters are FRACTIONAL here, and an
    integer-only sweep silently omits exactly the lines that make the form
    interesting** (trap hit and recorded 2026-08-25).  Charges are in the
    datum's coroot coordinates, so `Q^∨` is the integer lattice; this form's
    magnetic lattice is the strictly larger `P^∨`, and the cocharacters in
    `P^∨ \ Q^∨` are therefore **not integral**.  At `adjoint_lines(su_2())`
    the minimal 't Hooft line — the spinorial `ω^∨`, the whole point of SO(3) —
    is `m = Fraction(1, 2)`, and `⟨Σ⁺, (1,)⟩ = 2` there, i.e. integer `m = 1` is
    the *even-height* SU(2) adjoint monopole, not a new line at all.

    So a sweep over integer `m` on this lattice tests the SU(2) magnetic sector
    with the adjoint form's electric restriction — a real algebra, but **not**
    the spinorial sector, and every odd-`⟨Σ⁺,m⟩` phenomenon is invisible in it.
    Two ways not to be caught:

      * sweep `mag_admits` over fractional `m` as well (it accepts `Fraction`);
      * or work in a datum whose own coordinates make the lattice integral —
        for SO(3) that is `root_datum.so_n(3)`, where `⟨Σ⁺,(1,)⟩ = 1` and
        integer `m` covers the whole lattice, `m = 1` being the minuscule
        spinorial monopole.  This is the frame the BPS oracle
        `src/gn/pure_so3.py` uses.

    The two are the same lattice, so neither is "right"; what is wrong is
    enumerating one in the other's coordinates.  Worked example, with the
    axioms measured on both sides:
    a probe in the source repository.

    Generalised 2026-08-24 to a non-cyclic centre.  `H` used to be
    `range(invariants[0])`, which enumerates only the *first* invariant factor —
    at `Spin(8)` two of the four classes of `Z_2 × Z_2`.  That was a degenerate
    *encoding* rather than a wrong answer: `mag_admits` special-cased
    `set(H) == set(range(invariants[0]))` and took the `P^∨` branch, so the form
    came out right anyway.  The whole class group is now named explicitly, so the
    subgroup means what it says and the special case is not load-bearing."""
    divisors = centre_labelling(datum)[0]
    if not divisors:
        H = ()
    elif len(divisors) == 1:
        # Cyclic: keep the integer encoding, which every pre-2026-08-24 consumer
        # of `.H` reads (`langlands_dual`'s annihilator, the parity diagnostics).
        H = tuple(range(divisors[0]))
    else:
        H = tuple(k for k in _all_classes(divisors) if any(k))
    return LineLattice(datum, H=H, name=f"{datum.name} (adjoint)")
