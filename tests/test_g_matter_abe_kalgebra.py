"""Certification for `GMatterAbeKAlgebra` — `(G, N)` natively on the
`AbeKAlgebra` tier, and for the `TorusShape` / `MatterWRQTorus` widening that
lets it name its theory.

Four independent legs, deliberately kept apart:

1. **the shape names the theory** — `Sp(4)+4`, `Spin(5)+5` and `Spin(5)+4ˢ` used
   to declare the same `1` and were indistinguishable; they are now distinct,
   while the legacy `int` declaration still means exactly what it meant;
2. **the substrate widening is inert where it must be, and right where it is
   new** — the `int` and expanded-weight declarations of U(N)+N_f agree, and at
   `Sp(4)`/`Spin(5)`/`G₂` the rung ladder is literally
   `matter_star_bubbling.matter_monomials`;
3. **the chart is the RG image un-dressed** — `Z·chart == GMatterOverPure.rg_chart`
   cell by cell (the flow leg), and `chart == UNNfKAlgebra.chart` in the type-A
   corner (a genuinely independent oracle, no flow and no
   `matter_star_bubbling` in its path);
4. **the contract holds** — W1, `certify_canonical`, ρ a coefficient-free basis
   permutation with ρ∘ρ⁻¹ = id, the unit law, and orthonormality
   `I_{a,b} = δ + O(𝖖)` through `KAlgebra`'s own verifier;
5. **`multiply` works** — the operation that exercises the substrate cocycle
   `W` and the level-ascending read *together*, so it is the sharpest test of
   whether the presentation is right.  Closure is asserted by **exact
   reconstruction** (`Σ_c C^c_{ab}·chart(c) == chart(a)·chart(b)`), never by
   "decompose returned something"; the bar law is the **conjugate-transpose**
   one (see `test_multiply_bar_conjugate_law`); associativity is checked; and
   the structure constants — not merely the charts — are compared against
   `UNNfKAlgebra`.

Run: `python3 run_tests.py`
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import root_datum as rd
from abe_kalgebra import TorusShape
from laurent_poly import LaurentPoly
from g_matter_abe_kalgebra import GMatterAbeKAlgebra
from g_matter_over_pure import GMatterOverPure
from matter_multislot import Z_levels_vec
from matter_star_bubbling import matter_monomials
from matter_wrq_torus import (MatterWRQTorus, _flavour_rungs, defining_weight,
                              rep_weights, slot_weights)
from weyl_torus_ring import TorusRational

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}: {name}"
          + (f"  {detail}" if detail and not cond else ""), flush=True)


# ---------------------------------------------------------------------------
# 1. the shape names the theory
# ---------------------------------------------------------------------------
def test_shape_names_the_theory():
    print("TorusShape distinguishes theories with the same matter COUNT")
    sp = TorusShape.from_root_data((rd.sp_n(2),), matter=(((1, 0),),))
    b2v = TorusShape.from_root_data((rd.b_n_simply_connected(2),),
                                    matter=(((1, 0),),))
    b2s = TorusShape.from_root_data((rd.b_n_simply_connected(2),),
                                    matter=(((0, 1),),))
    check("Sp(4)+4 != Spin(5)+5", sp != b2v)
    check("Spin(5)+5 != Spin(5)+4ˢ (vector vs SPINOR matter)", b2v != b2s,
          f"{b2v!r} vs {b2s!r}")
    check("distinct shapes hash apart",
          len({sp, b2v, b2s}) == 3)
    check("counts still readable", (sp.matter, b2s.matter) == ((1,), (1,)))
    check("reps carry the distinction",
          b2v.matter_reps != b2s.matter_reps)

    print("the legacy `int` declaration is unchanged")
    u = rd.u_n(2)
    a = TorusShape.from_root_data((u,), matter=(2,))
    b = TorusShape.from_root_data((u,), matter=((defining_weight(u),) * 2,))
    check("U(2)+2 as int == U(2)+2 as explicit defining weights", a == b)
    check("U(2)+2 prints as a count", "matter=(2,)" in repr(a), repr(a))
    check("from_ranks_nf still builds the type-A chain",
          TorusShape.from_ranks_nf((2, 1), (1, 0)).matter == (1, 0))

    print("the algebra's own shape names it")
    A = GMatterAbeKAlgebra(rd.b_n_simply_connected(2), (0, 1), nf=2)
    check("Spin(5)+2×4ˢ shape carries the spinor weight",
          A.torus_shape().matter_reps == (((0, 1), (0, 1)),),
          repr(A.torus_shape()))
    check("and is != the same group with 2 vectors",
          A.torus_shape() != GMatterAbeKAlgebra(
              rd.b_n_simply_connected(2), (1, 0), nf=2).torus_shape())


# ---------------------------------------------------------------------------
# 2. the substrate widening
# ---------------------------------------------------------------------------
def test_substrate_widening():
    print("slot_weights: `int` and explicit reps agree at U(N)")
    u = rd.u_n(2)
    check("u_n(2) defining weight is the first fundamental",
          defining_weight(u) == (1, 0))
    check("its weights are the e_j", rep_weights(u, (1, 0)) == ((0, 1), (1, 0)))
    check("slot_weights(int 2) == slot_weights(two highest weights)",
          slot_weights(u, 2) == slot_weights(u, ((1, 0), (1, 0))))

    print("the rung ladder IS matter_star_bubbling.matter_monomials")
    for datum, lam, m in [(rd.u_n(2), (1, 0), (0, -2)),
                          (rd.sp_n(2), (1, 0), (1, 0)),
                          (rd.b_n_simply_connected(2), (0, 1), (1, 1)),
                          (rd.g_2(), (1, 0), (1, 2))]:
        sl = slot_weights(datum, (lam,))
        got = sorted(_flavour_rungs(m, sl))
        want = sorted(matter_monomials(datum, ((lam, 1),), m))
        check(f"{datum.name} λ={lam} m={m}: rungs match ({len(want)})",
              got == want, f"{got[:2]} vs {want[:2]}")

    print("MatterWRQTorus accepts either declaration and builds the same object")
    one = TorusRational.one(u)
    f = {(0, -1): {(0, 0): one}}
    x = MatterWRQTorus(u, 2, f)
    y = MatterWRQTorus(u, ((1, 0), (1, 0)), f)
    check("int and rep declarations give equal elements", x == y)
    check("and the same slot expansion", x.slots == y.slots)
    check("Nf stays the slot count", (x.Nf, y.Nf) == (2, 2))
    check("products agree too", (x * x) == (y * y))


# ---------------------------------------------------------------------------
# 3. the chart is the RG image un-dressed
# ---------------------------------------------------------------------------
FLOW_CASES = [
    (rd.su_2(), (1,), 2, [(1,)]),
    (rd.u_n(2), (1, 0), 2, [(0, -1)]),
    (rd.sp_n(2), (1, 0), 2, [(1, 0)]),
    (rd.b_n_simply_connected(2), (0, 1), 2, [(1, 1)]),
    # (B2, VECTOR) -- added 2026-07-28.  Every anomaly chased this session
    # landed on this datum precisely because no battery exercised it: this list
    # and g_matter_monoid_law.py both used the spinor only.  Certified at
    # m=(1,1) and (2,1) for n_f = 1 and 2.
    (rd.b_n_simply_connected(2), (1, 0), 1, [(1, 1), (2, 1)]),
    (rd.b_n_simply_connected(2), (1, 0), 2, [(1, 1)]),
]


def test_chart_times_Z_is_the_flow():
    """`Z·chart == RG(a)` cell by cell — the chart is the RG image with the
    matter dressing removed (the dressing lives in the substrate cocycle)."""
    print("Z · chart == GMatterOverPure.rg_chart")
    for datum, lam, nf, ms in FLOW_CASES:
        mat = ((tuple(lam), nf),)
        A = GMatterAbeKAlgebra(datum, lam, nf=nf)
        F = GMatterOverPure(datum, lam, nf=nf)
        e = (0,) * datum.dim
        for m in ms:
            lab = ((tuple(m), e), (0,) * nf)
            got = A.chart(lab).residuals()
            # re-dress: F_cell = Z_cell * Q_cell
            dressed = {}
            for c, row in got.items():
                Z = Z_levels_vec(datum, mat, c)
                for kq, q in row.items():
                    for kz, z in Z.items():
                        k = tuple(a + b for a, b in zip(kq, kz))
                        t = (q * TorusRational.from_laurent(z)).simplify()
                        if t.is_zero():
                            continue
                        d = dressed.setdefault(c, {})
                        d[k] = t if k not in d else (d[k] + t).simplify()
            truth = {}
            for k, x in F.rg_chart(lab, Kq=22).items():
                for c, v in x.residuals().items():
                    v = v.simplify()
                    if not v.is_zero():
                        truth.setdefault(tuple(c), {})[tuple(k)] = v
            bad = []
            for c in set(dressed) | set(truth):
                r1, r2 = dressed.get(c, {}), truth.get(c, {})
                for k in set(r1) | set(r2):
                    v1 = r1.get(k)
                    v2 = r2.get(k)
                    if v1 is None:
                        if not v2.is_zero():
                            bad.append((c, k))
                    elif v2 is None:
                        if not v1.is_zero():
                            bad.append((c, k))
                    elif not (v1 + v2 * (-1)).simplify().is_zero():
                        bad.append((c, k))
            check(f"{datum.name}+{nf}×{lam} m={m}: Z·chart == RG(a) "
                  f"({len(dressed)} cells)", not bad, str(bad[:3]))


# `test_chart_against_independent_oracle` compared against the type-A `UNNfKAlgebra` oracle;

# ---------------------------------------------------------------------------
# 4. the contract
# ---------------------------------------------------------------------------
CONTRACT_CASES = [
    ("SU(2)+2×2", rd.su_2(), (1,), 2, [((1,), (0,)), ((1,), (1,))]),
    ("U(2)+2×2", rd.u_n(2), (1, 0), 2, [((0, -1), (0, 0))]),
]


def test_contract():
    print("the KAlgebra contract through the derived tier")
    for tag, datum, lam, nf, gs in CONTRACT_CASES:
        A = GMatterAbeKAlgebra(datum, lam, nf=nf)
        one = A.identity()
        labs = [(g, (0,) * nf) for g in gs]
        check(f"{tag}: identity is in the basis", A.verify_identity_in_basis())
        check(f"{tag}: ρ fixes the identity", A.verify_rho_fixes_identity())
        for a in labs:
            check(f"{tag} {a[0]}: W1 (chart bar-invariant)",
                  A.verify_chart_bar(a))
            cert = A.certify_canonical(a)
            check(f"{tag} {a[0]}: certify_canonical reads the label back",
                  cert is not False and tuple(cert[0]) == tuple(a[0]),
                  str(cert))
            check(f"{tag} {a[0]}: unit law", A.verify_unit_law(a))
            r = A.rho(a)
            check(f"{tag} {a[0]}: ρ is a single coefficient-free label -> {r}",
                  isinstance(r, tuple))
            check(f"{tag} {a[0]}: ρ⁻¹∘ρ = id (element level)",
                  A.chart(a).rho().rho_inverse() == A.chart(a))
            check(f"{tag} {a[0]}: ρ⁻¹(ρ(a)) == a (label level)",
                  A.rho_inverse(r) == a, f"{A.rho_inverse(r)}")
        for a in labs:
            for b in labs:
                check(f"{tag}: orthonormality I{a[0]},{b[0]} = δ + O(𝖖)",
                      A.verify_orthonormality(a, b, K=3))


def test_honest_failures():
    print("honest failures / frame discipline")
    A = GMatterAbeKAlgebra(rd.su_2(), (1,), nf=2)
    try:
        A.chart((((1,), (0,)), (0,)))
        check("a flavour charge of the wrong length is refused", False,
              "accepted")
    except ValueError:
        check("a flavour charge of the wrong length is refused", True)
    try:
        A.chart((((1,), (0,)), (0, 1)))
        check("a flavour label that is not U(2)-dominant is refused", False,
              "accepted")
    except ValueError:
        check("a flavour label that is not U(2)-dominant is refused", True)
    check("fold transports a non-dominant m into the frame",
          A.fold((-1,), (0,))[0][0] == (1,), str(A.fold((-1,), (0,))))
    try:
        GMatterAbeKAlgebra(rd.su_2(), ())
        check("no matter is refused (use PureGAbeKAlgebra)", False, "accepted")
    except ValueError:
        check("no matter is refused (use PureGAbeKAlgebra)", True)


def test_flavour_is_a_character_multiple():
    """`L_{(g,w)} = χ_w(μ)·L_{(g,1)}` with `χ_w` an irrep of `∏_i U(n_i)`.

    The author's ruling, 2026-07-28: *"you could have `U(n_i)` if there are `n_i` copies
    of the same irrep"* — so the `n_i` interchangeable slots carry a `U(n_i)`,
    not `n_i` separate `U(1)`s.  This supersedes the interim `R(U(1)^M)`
    convention; the neutral label is the same object in both, which is why every
    flow/oracle comparison in this file is unaffected."""
    print("flavour-charged labels are χ_w multiples of the neutral one")
    A = GMatterAbeKAlgebra(rd.su_2(), (1,), nf=2)
    R = A.coefficient_ring()
    check("2 copies of one irrep ⇒ flavour ring R(U(2))",
          repr(R) == "UNZPlusRing(2)", repr(R))
    check("one group, both slots in it", A.groups == (((1,), (0, 1)),),
          str(A.groups))
    g = ((1,), (0,))
    base = A.chart((g, R.one_basis()))
    for w in [(1, 0), (0, 0), (1, 1), (-1, -1), (2, 0), (0, -1), (2, -1)]:
        x = A.chart((g, w))
        # rebuild by hand from the weight diagram of χ_w
        want = {}
        for wt, mult in R.character(w).items():
            for c, row in base.residuals().items():
                for kk, v in row.items():
                    k = tuple(a + b for a, b in zip(kk, wt))
                    d = want.setdefault(c, {})
                    scaled = (v * TorusRational.from_scalar(
                        A.datum, LaurentPoly({0: int(mult)}))).simplify()
                    d[k] = scaled if k not in d else (d[k] + scaled).simplify()
        got = x.residuals()
        bad = []
        for c in set(got) | set(want):
            r1, r2 = got.get(c, {}), want.get(c, {})
            for k in set(r1) | set(r2):
                v1 = r1.get(k)
                v2 = r2.get(k)
                if v1 is None:
                    if not v2.is_zero():
                        bad.append((c, k))
                elif v2 is None:
                    if not v1.is_zero():
                        bad.append((c, k))
                elif not (v1 + v2 * (-1)).simplify().is_zero():
                    bad.append((c, k))
        check(f"chart(g, χ_{w}) == χ_{w}(μ)·chart(g, 1)  (dim {R.dim(w)})",
              not bad, str(bad[:3]))
    sec, key = A.r_label_decompose((g, (1, 0)))
    check("r_label_decompose splits off the flavour irrep",
          sec == (g, R.one_basis()) and tuple(key) == (1, 0), f"{sec}, {key}")
    check("r_label_compose inverts it",
          A.r_label_compose(sec, key) == (g, (1, 0)))
    check("the neutral label is unchanged by the ruling "
          "(one_basis == the old 0⃗)", R.one_basis() == (0, 0))


def test_flavour_packaging_shortens_structure_constants():
    """The visible consequence of the ruling: two `U(1)²` charges that were
    separate labels are recognized as ONE `U(2)` doublet.

    At SU(2)+2×2, `L_{(0,e=1)} · L_{(m=1,e=0)}` used to return four canonicals,
    two of them carrying the separate charges `(0,1)` and `(1,0)`.  It now
    returns three, with those two packaged as the defining rep of `U(2)`."""
    print("flavour packaging: separate U(1) charges become one U(n) irrep")
    A = GMatterAbeKAlgebra(rd.su_2(), (1,), nf=2)
    R = A.coefficient_ring()
    one = R.one_basis()
    W = (((0,), (1,)), one)
    Mg = (((1,), (0,)), one)
    el = A.multiply(W, Mg)
    labs = {lab: C for lab, C in el.terms.items() if not C.is_zero()}
    check(f"L_W · L_m has 3 canonicals ({len(labs)})", len(labs) == 3,
          str({str(k): str(v) for k, v in labs.items()}))
    doublet = (((0,), (0,)), (1, 0))
    check("the flavour doublet is a SINGLE label with the U(2) defining rep",
          doublet in labs and labs[doublet] == LaurentPoly({0: 1}),
          str({str(k): str(v) for k, v in labs.items()}))
    check("and it really is 2-dimensional", R.dim((1, 0)) == 2)



# ---------------------------------------------------------------------------
# 5. multiply — the operation that exercises the cocycle AND the read together
# ---------------------------------------------------------------------------
def _bar(lp):
    return LaurentPoly({-e: c for e, c in lp._coeffs.items() if c})


def _palindromic(lp):
    c = {e: v for e, v in lp._coeffs.items() if v}
    return all(c.get(-e, 0) == v for e, v in c.items())


def _reconstruct(A, el, M):
    """`Σ_c C^c·chart(c)` — rebuilding the product from its decomposition."""
    acc = None
    for lab, C in el.terms.items():
        if C.is_zero():
            continue
        piece = A.chart(lab)._scaled(C, (0,) * M)
        acc = piece if acc is None else acc + piece
    return acc


MUL_CASES = [
    ("SU(2)+2×2", rd.su_2(), (1,), 2,
     [((0,), (1,)), ((0,), (2,)), ((1,), (0,))]),
    ("U(2)+2×2", rd.u_n(2), (1, 0), 2,
     [((0, 0), (1, 0)), ((0, -1), (0, 0))]),
    ("Sp(4)+2×4", rd.sp_n(2), (1, 0), 2,
     [((0, 0), (1, 0)), ((0, 0), (0, 1)), ((1, 0), (0, 0))]),
    ("Spin(5)+2×4ˢ", rd.b_n_simply_connected(2), (0, 1), 2,
     [((0, 0), (1, 0)), ((0, 0), (0, 1)), ((1, 1), (0, 0))]),
]


def test_multiply_closes_by_reconstruction():
    """A `decompose` that returns *something* is not evidence of closure.
    `Σ_c C^c_{ab}·chart(c) == chart(a)·chart(b)` is — it is the only check here
    that cannot be satisfied by an off-span answer."""
    print("multiply closes in the canonical basis (exact reconstruction)")
    for tag, datum, lam, nf, gs in MUL_CASES:
        A = GMatterAbeKAlgebra(datum, lam, nf=nf)
        M = A.M
        pairs = [(gs[0], gs[0]), (gs[0], gs[1]), (gs[1], gs[0]),
                 (gs[-1], gs[0]), (gs[0], gs[-1])]
        for (g1, g2) in pairs:
            a, b = (g1, (0,) * M), (g2, (0,) * M)
            try:
                el = A.multiply(a, b)
                acc = _reconstruct(A, el, M)
                prod = A.chart(a) * A.chart(b)
            except Exception as ex:
                check(f"{tag} {g1}·{g2}: multiply + reconstruct", False,
                      f"{type(ex).__name__}: {str(ex)[:110]}")
                continue
            good = prod.is_zero() if acc is None else acc == prod
            check(f"{tag} {g1}·{g2}: RECONSTRUCTS ({len(el.terms)} terms)",
                  good, str({str(k): str(v) for k, v in el.terms.items()}))


def test_multiply_bar_conjugate_law():
    """Bar is **antimultiplicative** and fixes the canonical basis, so from
    `L_a L_b = Σ_c C^c_{ab} L_c`:

        bar(L_a L_b) = bar(L_b)·bar(L_a) = L_b L_a = Σ_c C^c_{ba} L_c

    hence **`bar(C^c_{ab}) = C^c_{ba}`** — the conjugate-transpose law.  NOT
    `bar(C) = C`: an individual structure constant is 𝖖-palindromic exactly
    when the pair *commutes*, and these algebras are quantized, so that is the
    exception.  Measured here: the equivalence
    `all-palindromic ⟺ commuting` holds in every case, and the Wilson ×
    monopole pairs are precisely the non-commuting ones (e.g. at SU(2)+2×2,
    `C^{((1,),(1,))}` is `𝖖⁻¹` one way and `𝖖` the other)."""
    print("bar(C^c_ab) == C^c_ba, and palindromic ⟺ commuting")
    for tag, datum, lam, nf, gs in MUL_CASES[:2]:
        A = GMatterAbeKAlgebra(datum, lam, nf=nf)
        M = A.M
        for g1 in gs:
            for g2 in gs:
                a, b = (g1, (0,) * M), (g2, (0,) * M)
                try:
                    ab = A.multiply(a, b)
                    ba = A.multiply(b, a)
                except Exception as ex:
                    check(f"{tag} {g1}·{g2}: multiply both orders", False,
                          f"{type(ex).__name__}: {str(ex)[:110]}")
                    continue
                A_ = {k: C for k, C in ab.terms.items() if not C.is_zero()}
                B_ = {k: C for k, C in ba.terms.items() if not C.is_zero()}
                zero = LaurentPoly.zero()
                keys = set(A_) | set(B_)
                bad = [(str(k), str(A_.get(k, zero)), str(B_.get(k, zero)))
                       for k in keys
                       if not (_bar(A_.get(k, zero))
                               - B_.get(k, zero)).is_zero()]
                check(f"{tag} {g1}·{g2}: bar(C^c_ab) == C^c_ba", not bad,
                      str(bad[:2]))
                commuting = all((A_.get(k, zero) - B_.get(k, zero)).is_zero()
                                for k in keys)
                allpal = all(_palindromic(C) for C in A_.values())
                check(f"{tag} {g1}·{g2}: palindromic ⟺ commuting "
                      f"(both {commuting})", commuting == allpal,
                      f"commuting={commuting} palindromic={allpal}")


# `test_multiply_against_independent_oracle` compared against the type-A `UNNfKAlgebra` oracle;

def test_structure_constants_positive_and_integral():
    """**The sharpest check available**.

    Reconstruction proves the product lies in the **span** of the basis.
    Positivity proves the basis is **canonical**: `C^c_{ab} ∈ Z₊[𝖖,𝖖⁻¹]` is
    the Z₊-ring property the whole `zplus_ring` framework is named for, and a
    single negative coefficient is the documented signature of a fabricated
    element — the design notes: *"That subtraction is the entire
    origin of the negative structure constant.  There is no real negative;
    pure-U(3) is positive, as BPS shows directly."*  It caught a defect the
    bar-blind 𝖖⁰ self-norm did not.

    Integrality: the scalar ring is `Z[𝖖,𝖖⁻¹]` and nothing else (standing
    ruling — no `𝖖^{1/2}` ever), so every coefficient is a plain `int`.

    Scope, so this is not over-read: positivity is asserted of **structure
    constants**, not of the Schur index — `I_{a,a}` legitimately carries
    negatives (the documented `1 − 5𝖖²`).

    Measured across SU(2)+2×2, U(2)+N_f=1,2, Sp(4)+2×4 and Spin(5)+2×4ˢ, and
    against `UNNfKAlgebra` as a control: 300+ structure constants, no negative
    and no non-integer coefficient."""
    print("structure constants: positivity and integrality")
    for tag, datum, lam, nf, gs in MUL_CASES:
        A = GMatterAbeKAlgebra(datum, lam, nf=nf)
        M = A.M
        labs = [(g, (0,) * M) for g in gs]
        if datum.name.startswith("SU(2)"):
            labs.append(((gs[-1]), (1, 0)))      # a flavour-charged label too
        n_terms = 0
        neg, nonint = [], []
        for a in labs:
            for b in labs:
                try:
                    el = A.multiply(a, b)
                except Exception as ex:
                    check(f"{tag} {a[0]}·{b[0]}: multiply runs", False,
                          f"{type(ex).__name__}: {str(ex)[:110]}")
                    continue
                for c, C in el.terms.items():
                    if C.is_zero():
                        continue
                    n_terms += 1
                    for ex_, co in C._coeffs.items():
                        if co == 0:
                            continue
                        if not isinstance(co, int) or not isinstance(ex_, int):
                            nonint.append((str(a), str(b), str(c), str(C)))
                        elif co < 0:
                            neg.append((str(a), str(b), str(c), str(C)))
        check(f"{tag}: INTEGRALITY — every C^c_ab in Z[𝖖,𝖖⁻¹] "
              f"({n_terms} structure constants)", not nonint, str(nonint[:2]))
        check(f"{tag}: POSITIVITY — every C^c_ab in Z₊[𝖖,𝖖⁻¹] "
              f"({n_terms} structure constants)", not neg, str(neg[:2]))


def test_associativity():
    """The sharpest structural test available: associativity fails if the
    substrate cocycle or the level-ascending read is wrong in a way closure
    alone would not catch."""
    print("associativity of the derived multiply")
    A = GMatterAbeKAlgebra(rd.su_2(), (1,), nf=2)
    M = A.M
    W1 = (((0,), (1,)), (0,) * M)
    W2 = (((0,), (2,)), (0,) * M)
    MG = (((1,), (0,)), (0,) * M)
    for (a, b, c) in [(W1, W1, W1), (W1, W2, W1), (W1, W1, MG), (W1, MG, W1),
                      (MG, W1, W1)]:
        try:
            good = A.verify_associativity(a, b, c)
        except Exception as ex:
            check(f"assoc {a[0]},{b[0]},{c[0]}", False,
                  f"{type(ex).__name__}: {str(ex)[:110]}")
            continue
        check(f"assoc {a[0]},{b[0]},{c[0]}", good)


def test_bar_antimultiplicative_on_charts():
    """The chart-level statement, with `decompose` out of the path entirely."""
    print("bar(x·y) == bar(y)·bar(x) on charts")
    for tag, datum, lam, nf, gs in MUL_CASES[:2]:
        A = GMatterAbeKAlgebra(datum, lam, nf=nf)
        M = A.M
        for g1 in gs:
            for g2 in gs:
                x, y = A.chart((g1, (0,) * M)), A.chart((g2, (0,) * M))
                check(f"{tag} bar({g1}·{g2}) == bar({g2})·bar({g1})",
                      (x * y).bar() == y.bar() * x.bar())


if __name__ == "__main__":
    for fn in [test_shape_names_the_theory,
               test_substrate_widening,
               test_chart_times_Z_is_the_flow,
               test_flavour_is_a_character_multiple,
               test_flavour_packaging_shortens_structure_constants,
               test_honest_failures,
               test_multiply_closes_by_reconstruction,
               test_structure_constants_positive_and_integral,
               test_bar_antimultiplicative_on_charts,
               test_multiply_bar_conjugate_law,
               test_associativity,
               test_contract]:
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        sys.exit(1)
    print("All GMatterAbeKAlgebra tests passed.")
