"""Certification for `GMatterOverMatter` — **removing matter** as an
`RGKAlgebra`: `(G, N_keep ⊕ N_drop) → (G, N_keep)` at any `RootDatum`.

Five independent legs, deliberately kept apart:

1. **`GMatterOverPure` is the `keep = ∅` case** — not "morally", but term by
   term: `S_RG` identical at several cutoffs and `RG(a)` identical on Wilson and
   monopole labels, at type A *and* outside it (`Sp(4)`).  This is what makes the
   new class a generalization rather than a parallel implementation.

2. **`RG` closes over a MATTER auxiliary.**  The load-bearing question.
   `GMatterOverPure`'s auxiliary is pure `G` (charts from the WRQ engine); here it
   is `(G, N_keep)`, whose charts come from the (★)-guarded constructor.  If the
   generic graded co-solver did not close over that, the class would be a shell.

3. **the RG axiom battery** — `verify_rg_unital`, `_bar_invariant`,
   `_multiplicative` on the partial flow.  Note the enforced-vs-emergent split
   (the audit): `_bar_invariant` is **enforced** for solver-built flows
   (the graded solver peels palindromically), so it is a regression guard here;
   `_multiplicative` is **emergent** and is the leg that carries evidence.

4. **`Ψ` expands on the right basis.**  The construction needs the IR algebra's
   Wilson sector to be the character ring `R(G)` with the *same* `χ_e` as pure
   `G`, undisturbed by the surviving matter.  Checked two ways: the `(G, N_keep)`
   Wilson chart *is* the pure-`G` Wilson chart, and `fuse_characters` agrees with
   the auxiliary's own `multiply` on Wilson pairs (the contract surface).  This is
   (★) as the author states it — the elements act on `G` characters, and
   `L_{(0,e)}|N] = χ_e` — made executable.

5. **the native algebra EMBEDS in the flow's** — and the word is
   "embeds", not "equals".  The flow grades its dropped slots one `U(1)` each
   (`R(U(1)^M)`) while the native class packages identical slots into `R(U(n_i))`
   (TM7), and the map between them is the Cartan expansion
   `χ_λ ↦ Σ_{w ∈ wt(λ)} μ^w` (`zplus_ring.un_to_cartan_hom`).  That is injective
   and multiplicative, but its image is the flavour-**symmetric** part of the
   flow's algebra, so the flow is strictly the finer-graded object whenever an
   irrep repeats.  It becomes an honest isomorphism exactly when all matter irreps
   are distinct (every `n_i = 1`) — see `g_matter_roster`'s `flow_iso()`, certified
   in the suite in the source repository.  Comparing through the hom is in any case
   sharper than comparing after `augmentation_hom`, which would forget the flavour
   refinement entirely.

`--full` adds the expensive rungs (Sp(4) partial flow, deeper monopoles).

Run: `python3 run_tests.py` [--full]`
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import root_datum as rd
from laurent_poly import LaurentPoly
from zplus_ring import un_to_cartan_hom

from gn_abe_kalgebra import GNAbeKAlgebra
from g_matter_over_matter import GMatterOverMatter, matter_removal_tower
from g_matter_over_pure import GMatterOverPure, fuse_characters

FULL = "--full" in sys.argv
PASS, FAIL = [], []


def check(name, ok, detail=""):
    if ok:
        PASS.append(name)
        print(f"  [ok]   {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}{(' — ' + detail) if detail else ''}")


# ===========================================================================
# Leg 1 — GMatterOverPure is the keep = ∅ case
# ===========================================================================


def _pure_pairs():
    """`(tag, datum, weight, nf, [(m, e), …])` — theories where both the old
    `GMatterOverPure` and the new `keep = ∅` flow are cheap."""
    cases = [("SU(2)+1", rd.su_2(), (1,), 1, [((0,), (0,)), ((0,), (1,)),
                                              ((1,), (0,))]),
             ("U(2)+1", rd.u_n(2), (1, 0), 1, [((0, 0), (0, 0)),
                                               ((0, 0), (1, 0)),
                                               ((0, -1), (0, 0))])]
    if FULL:
        cases.append(("Sp(4)+1", rd.sp_n(2), (1, 0), 1,
                      [((0, 0), (0, 0)), ((0, 0), (1, 0))]))
    return cases


def test_keep_empty_is_gmatter_over_pure():
    """`S_RG` and `RG(a)` identical to `GMatterOverPure`, term by term."""
    print("keep = ∅ reproduces GMatterOverPure exactly")
    for tag, datum, lam, nf, labs in _pure_pairs():
        P = GMatterOverPure(datum, lam, nf=nf)
        Q = GMatterOverMatter(datum, keep=(), drop=((lam,) * nf))
        check(f"{tag}: IR is the same pure-G class",
              type(Q.ir()).__name__ == type(P.pure()).__name__)
        for cutoff in (1, 2, 3):
            sp, sq = P.rg_generator(cutoff), Q.rg_generator(cutoff)
            check(f"{tag}: S_RG(≤{cutoff}) identical ({len(sp)} terms)",
                  sp == sq and len(sp) > 0)
        for m, e in labs:
            a = (P.pure().fold(m, e), (0,) * nf)
            try:
                ra = dict(P.RG(a).terms)
                rb = dict(Q.RG(a).terms)
            except Exception as ex:
                check(f"{tag}: RG({m},{e})", False,
                      f"{type(ex).__name__}: {str(ex)[:90]}")
                continue
            check(f"{tag}: RG({m},{e}) identical ({len(ra)} terms)",
                  ra == rb and len(ra) > 0)


def test_keep_empty_dressing_prediction_agrees():
    """The predicted matter dressing `Z(m)` is the same object in both classes
    when nothing survives (it depends only on the dropped slots)."""
    print("predicted_dressing agrees at keep = ∅")
    P = GMatterOverPure(rd.su_2(), (1,), nf=2)
    Q = GMatterOverMatter(rd.su_2(), keep=(), drop=((1,), (1,)))
    for m in [(0,), (1,), (2,), (-1,)]:
        check(f"Z({m})", P.predicted_dressing(m) == Q.predicted_dressing(m))


# ===========================================================================
# Leg 2 + 3 — RG closes over a matter auxiliary, and the axioms hold
# ===========================================================================


def _partial_cases():
    cases = [("SU(2)+2→+1", rd.su_2(), (1,), 2, [((0,), (0,)), ((0,), (1,)),
                                                 ((0,), (2,)), ((1,), (0,))]),
             ("U(2)+2→+1", rd.u_n(2), (1, 0), 2, [((0, 0), (0, 0)),
                                                  ((0, 0), (1, 0)),
                                                  ((0, -1), (0, 0))])]
    if FULL:
        cases.append(("SU(2)+3→+2", rd.su_2(), (1,), 3,
                      [((0,), (1,)), ((1,), (0,))]))
        cases.append(("Sp(4)+2→+1", rd.sp_n(2), (1, 0), 2,
                      [((0, 0), (0, 0)), ((0, 0), (1, 0))]))
    return cases


def test_rg_closes_over_a_matter_auxiliary():
    """The partial flow's auxiliary is `(G, N_keep)`, whose charts come from the
    (★)-guarded constructor.  Does the generic graded co-solver close over it?"""
    print("RG closes over a (G, N_keep) auxiliary")
    for tag, datum, lam, nf, labs in _partial_cases():
        F = GMatterOverMatter.from_uv(datum, lam, nf=nf, drop=nf - 1)
        IR = F.ir()
        check(f"{tag}: IR carries the surviving matter",
              getattr(IR, "M", 0) == nf - 1)
        for m, e in labs:
            a = (IR.fold(m, e), (0,) * F.M)
            try:
                terms = dict(F.RG(a).terms)
            except Exception as ex:
                check(f"{tag}: RG({m},{e}) closes", False,
                      f"{type(ex).__name__}: {str(ex)[:90]}")
                continue
            # the identity must be rigid; everything else must be non-empty and
            # must start at the flow's own label (grade 0 = the IR element)
            grade0 = {lab: c for lab, c in terms.items() if sum(lab[1]) == 0}
            check(f"{tag}: RG({m},{e}) closes, {len(terms)} terms, "
                  f"grade-0 = the label itself",
                  bool(terms) and list(grade0) == [a]
                  and grade0[a] == LaurentPoly({0: 1}))


def test_rg_axioms_on_the_partial_flow():
    """`verify_rg_unital` / `_bar_invariant` / `_multiplicative`.

    Enforced-vs-emergent: `_bar_invariant` is enforced for solver-built flows
    (the graded solver peels palindromically), so it is a regression guard;
    `_multiplicative` is emergent and is the leg carrying evidence."""
    print("RG axiom battery on the partial flow")
    for tag, datum, lam, nf, labs in _partial_cases()[:2]:
        F = GMatterOverMatter.from_uv(datum, lam, nf=nf, drop=nf - 1)
        IR = F.ir()
        z = (0,) * datum.dim
        w = IR.fold(z, tuple(lam))                    # a Wilson line
        a = (w, (0,) * F.M)
        check(f"{tag}: verify_rg_unital", F.verify_rg_unital())
        check(f"{tag}: verify_rg_bar_invariant (enforced)",
              F.verify_rg_bar_invariant(a))
        check(f"{tag}: verify_rg_multiplicative(W,W) (emergent)",
              F.verify_rg_multiplicative(a, a))
        # the sharp one: a NON-COMMUTING pair (Wilson × monopole)
        try:
            mono = IR.fold(tuple([1] + [0] * (datum.dim - 1)), z)
            b = (mono, (0,) * F.M)
            check(f"{tag}: verify_rg_multiplicative(W,monopole) (emergent)",
                  F.verify_rg_multiplicative(a, b))
        except Exception as ex:
            check(f"{tag}: verify_rg_multiplicative(W,monopole)", False,
                  f"{type(ex).__name__}: {str(ex)[:90]}")


# ===========================================================================
# Leg 4 — Ψ expands on the right basis: the Wilson sector is R(G)
# ===========================================================================


def test_ir_wilson_sector_is_the_character_ring():
    """Two checks that the surviving matter does not disturb the Wilson sector:
    the `(G, N_keep)` Wilson chart IS the pure-`G` Wilson chart, and Wilson
    fusion is plain Littlewood–Richardson.

    This is (★) in the author's formulation made executable — the canonical
    elements are `𝖖`-difference operators acting on `G` characters, and
    `L_{(0,e)}|N] = χ_e`, so the Wilson sector *is* `Λ = R(G)` whatever the
    matter is."""
    print("the (G, N_keep) Wilson sector is R(G), undisturbed by the matter")
    from pure_g_abe_kalgebra import PureGAbeKAlgebra
    for tag, datum, lam, es in [
            ("SU(2)+1", rd.su_2(), (1,), [(0,), (1,), (2,)]),
            ("U(2)+1", rd.u_n(2), (1, 0), [(0, 0), (1, 0), (1, 1)])]:
        K = GNAbeKAlgebra(datum, (lam,))
        P = PureGAbeKAlgebra(datum)
        z = (0,) * datum.dim
        for e in es:
            cm = K.chart(K.fold(z, e))
            cp = P.chart(P.fold(z, e))
            fam = cm.to_family()
            ok = (list(fam) == [(0,) * K.M]
                  and fam[(0,) * K.M].residuals() == cp.residuals())
            check(f"{tag}: chart(χ_{e}) == pure chart(χ_{e})", ok)


def test_wilson_fusion_matches_the_contract_surface():
    """`fuse_characters` (used to expand `Ψ`) vs the auxiliary's own `multiply`.

    The flow uses `fuse_characters` directly because the auxiliary route is
    blocked at non-simply-laced data by the spine bug documented on it; this
    pins the two together where the auxiliary path does work."""
    print("fuse_characters == auxiliary.multiply on Wilson pairs")
    for tag, datum, lam, es in [
            ("SU(2)+2→+1", rd.su_2(), (1,), [(0,), (1,), (2,)]),
            ("U(2)+2→+1", rd.u_n(2), (1, 0), [(0, 0), (1, 0), (1, 1)])]:
        F = GMatterOverMatter(datum, keep=(lam,), drop=(lam,))
        AUX = F.auxiliary()
        zero_k = (0,) * F.M
        bad = 0
        for e1 in es:
            for e2 in es:
                want = {(F.wilson_label(e), zero_k): int(mult)
                        for e, mult in fuse_characters(datum, e1, e2).items()}
                el = AUX.multiply((F.wilson_label(e1), zero_k),
                                  (F.wilson_label(e2), zero_k))
                got = {}
                for lab, C in el.terms.items():
                    cs = {k: v for k, v in C._coeffs.items() if v}
                    if set(cs) != {0}:
                        got[lab] = ("non-q0", cs)
                    else:
                        got[lab] = cs[0]
                if got != want:
                    bad += 1
        check(f"{tag}: {len(es) ** 2} Wilson pairs agree", bad == 0,
              f"{bad} mismatches")


# ===========================================================================
# Leg 5 — flow vs native: one algebra, two flavour conventions
# ===========================================================================


def _cartan_expand(native, flow, el):
    """Push a native `GNAbeKAlgebra` element into the flow's flavour
    convention: expand each label's `R(U(n))` irrep into Cartan weights
    (`un_to_cartan_hom`) and re-key onto the flow's `(IR label, k⃗_drop)`.

    Only valid when every matter slot carries the same irrep (so the native ring
    is a single `UNZPlusRing(n)`) — which is the case being compared."""
    n = len(native.matter)
    h = un_to_cartan_hom(n)
    IR = flow.ir()
    n_keep = len(flow.keep)
    out = {}
    for (g, w), C in el.terms.items():
        for wt, mult in h.apply_basis(w).terms.items():
            keep_w = tuple(wt[:n_keep])
            drop_k = tuple(wt[n_keep:])
            ir_lab = IR.fold(g[0], g[1], keep_w) if n_keep else IR.fold(*g)
            key = (ir_lab, drop_k)
            out[key] = out.get(key, LaurentPoly.zero()) + C * LaurentPoly(
                {0: int(mult)})
    return {k: v for k, v in out.items() if not v.is_zero()}


def test_native_embeds_in_the_flow():
    """one algebra, many presentations on this family, stated honestly: the Cartan expansion embeds
    `GNAbeKAlgebra(G, keep ⊕ drop)` into the flow's UV algebra as a
    multiplicative map, and the two agree on every product checked.

    It is an **embedding**, not an isomorphism, whenever a matter irrep repeats:
    the image is the flavour-symmetric part of the flow's finer `R(U(1)^M)`
    grading.  (`SU(2)+2` here repeats the doublet, so this is the embedding case;
    the iso case is `g_matter_roster.flow_iso()`.)"""
    print("native (G, keep ⊕ drop) embeds in the flow, via un_to_cartan_hom")
    cases = [("SU(2)+2", rd.su_2(), (1,), [((0,), (0,)), ((0,), (1,)),
                                           ((0,), (2,))])]
    if FULL:
        cases.append(("U(2)+2", rd.u_n(2), (1, 0),
                      [((0, 0), (0, 0)), ((0, 0), (1, 0))]))
    for tag, datum, lam, gs in cases:
        F = GMatterOverMatter(datum, keep=(lam,), drop=(lam,))
        NAT = F.uv_native()
        check(f"{tag}: uv_matter is keep+drop",
              NAT.matter == F.uv_matter and len(NAT.matter) == 2)
        zero_k = (0,) * F.M
        triv = NAT.coefficient_ring().one_basis()
        pairs = [((m, e), (m2, e2)) for m, e in gs for m2, e2 in gs]
        # The Wilson×Wilson block above lands entirely on flavour-NEUTRAL
        # labels, so on its own it never exercises the Cartan dictionary at all.
        # These pairs do: a Wilson×monopole product carries flavour charge (the
        # matter zero-mode), in both orders (the pair does not commute).
        mono = tuple([1] + [0] * (datum.dim - 1))
        z = (0,) * datum.dim
        pairs += [((z, tuple(lam)), (mono, z)),
                  ((mono, z), (z, tuple(lam))),
                  ((mono, z), (mono, z))]
        moved = 0
        for (m, e), (m2, e2) in pairs:
            a_n, b_n = (NAT.fold(m, e, triv)), (NAT.fold(m2, e2, triv))
            a_f = (F.ir().fold(m, e), zero_k)
            b_f = (F.ir().fold(m2, e2), zero_k)
            try:
                want = _cartan_expand(NAT, F, NAT.multiply(a_n, b_n))
                got = {lab: C for lab, C in
                       F.multiply(a_f, b_f).terms.items()
                       if not C.is_zero()}
            except Exception as ex:
                check(f"{tag}: ({m},{e})·({m2},{e2})", False,
                      f"{type(ex).__name__}: {str(ex)[:90]}")
                continue
            charged = sum(1 for k in want if any(x != 0 for x in k[1]))
            moved += charged
            check(f"{tag}: ({m},{e})·({m2},{e2}) agrees ({len(want)} terms, "
                  f"{charged} flavour-charged)", want == got,
                  f"native→flow {want} vs flow {got}")
        # Guard against the comparison silently degenerating to the neutral
        # sector, where it would prove nothing about the flavour dictionary.
        check(f"{tag}: the Cartan dictionary was actually exercised "
              f"({moved} flavour-charged terms)", moved > 0)


# ===========================================================================
# The tower, and the honest failures
# ===========================================================================


def test_removal_tower_shape():
    """`matter_removal_tower` walks `(G, N)` down to pure gauge one slot at a
    time; the last rung is the `GMatterOverPure` case."""
    print("the one-slot-at-a-time removal tower")
    tower = matter_removal_tower(rd.su_2(), (1,), nf=3)
    check("SU(2)+3 tower has 3 rungs", len(tower) == 3)
    check("rungs keep 2, 1, 0 slots",
          [len(r.keep) for r in tower] == [2, 1, 0])
    check("every rung drops exactly one slot",
          all(len(r.drop) == 1 for r in tower))
    check("every rung's UV matter is the previous rung's IR + 1",
          [len(r.uv_matter) for r in tower] == [3, 2, 1])
    check("the last rung's IR is pure G",
          type(tower[-1].ir()).__name__ == "PureGAbeKAlgebra")
    check("earlier rungs' IR carries matter",
          all(getattr(r.ir(), "M", 0) == len(r.keep) for r in tower[:-1]))


def test_honest_failures():
    """An empty `drop` is not an RG flow, and a bad slot index is refused."""
    print("honest failures")
    try:
        GMatterOverMatter(rd.su_2(), keep=((1,),), drop=())
    except ValueError:
        check("empty drop raises ValueError", True)
    else:
        check("empty drop raises ValueError", False)
    try:
        GMatterOverMatter.from_uv(rd.su_2(), (1,), nf=2, drop=7)
    except ValueError:
        check("out-of-range slot index raises ValueError", True)
    else:
        check("out-of-range slot index raises ValueError", False)


if __name__ == "__main__":
    for fn in [test_keep_empty_is_gmatter_over_pure,
               test_keep_empty_dressing_prediction_agrees,
               test_rg_closes_over_a_matter_auxiliary,
               test_rg_axioms_on_the_partial_flow,
               test_ir_wilson_sector_is_the_character_ring,
               test_wilson_fusion_matches_the_contract_surface,
               test_native_embeds_in_the_flow,
               test_removal_tower_shape,
               test_honest_failures]:
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        sys.exit(1)
    print("All GMatterOverMatter tests passed."
          + ("" if FULL else "  (--full adds Sp(4) + deeper rungs)"))
