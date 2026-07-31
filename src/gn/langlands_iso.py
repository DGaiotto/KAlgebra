"""The **Langlands family of `KAlgebraIso`** — `(G, N) ≅ (G^∨, N)` under the
Kapustin charge map `S: (m, e) ↦ (e, −m)`.

User framing (2026-07-28): *"consider (G, Adj) and verify the Langlands duality to
(G^∨, Adj) m→e, e→−m"*, then *"you can formalize by a Langlands family of
KAlgebraIso"*, and *"the Langlands dual notion seems to extend cleanly to 4d gauge
group data"*.  All three are taken literally here: the duality is presented as
certified `KAlgebraIso` objects, the charge map and the dual gauge-group data live
on `global_form` (`LineLattice.langlands_dual`, `langlands_label_map`), and this
module is only the bridge from that lattice statement to the algebra statement.

**What the duality says on this tier.**  `S` exchanges the magnetic and electric
lattices, so it exchanges the two extreme global forms — `SU(N) ↔ PSU(N)`,
`SU(2) ↔ SO(3)`.  For **simply laced** `G` the root datum is Langlands self-dual
(`C^∨ = C^T = C`), so the dual theory is the *same* `RootDatum` and the *same*
matter, with only the `LineLattice` flipped.  That is what makes `(G, Adj)` the
right test case: the adjoint representation is a representation of **every** global
form of `G` (the centre acts trivially on it), so the matter transports unchanged
and the duality is a statement about the charge lattice alone.  A fundamental
hypermultiplet would not survive the trip — at `PSU(N)` it is not a
representation, which `GNAbeKAlgebra` refuses up front.

**Why this test needs ruling D10.**  The charge dictionary is `φ = C·`
(`φ(ω_i^∨) = ω_i`), so a **Wilson line of `SU(N)` maps to a fractional-coweight
monopole of `PSU(N)`** — e.g. `(0, ω_1) ↦ ((2/3, 1/3), 0)` at `SU(3)`.  Before D10
the tier refused those charges outright, so no non-trivial dual label could even be
built and this family would have been vacuous.

**WHAT IS ESTABLISHED** (measured 2026-07-28 at `SU(2)+Adj` and `SU(3)+Adj`):
`S` on the charge together with the flavour twist `κ` (`langlands_flavour_twist`) is a
**full `KAlgebra` isomorphism** — unit, round trip, multiplicativity, ρ-equivariance
**and the ρ²-twisted trace**, the last both on the nose and up to `χ_κ`.

**A truncation bug, and the lesson is the point.**  The trace legs failed until the
matter **Nahm window** was fixed.  `AbeKAlgebra.trace` / `inner_product` (and this
class's overrides) called `chart.trace(K=K)` without forwarding `W`, so the flavour
factor `∏ E(μ_i v_j)E(μ_i^{-1}v_j^{-1})` stayed expanded to its default level **4**
however large a `K` was asked for.  The first wrong `𝖖`-order is exactly `W+1`, so
everything above `𝖖⁴` was silently wrong — and it produced a *beautifully plausible*
false signal: the trace appeared to agree exactly when `κ = 0` and fail exactly when
`κ ≠ 0`, which invited the conclusion that the zero modes contribute an index factor
no monomial could carry.  That reading was wrong; there was no phenomenon.  (User
ruling: *"the trace cannot fail. You have a bug."*)  `UNNfKAlgebra` had already been
fixed this way in an earlier session — the group-general classes simply never
inherited it.

Two habits this justifies: never report a `𝖖`-series discrepancy without first
scanning the truncation parameters (`K` **and** any internal window) to see whether
the divergence order moves with them; and treat a discrepancy that switches on
cleanly with some structural label as *evidence of a cutoff correlated with that
label*, not as a discovered law.

**What bounds the certifiable set** (rewritten 2026-07-30, ruling D31 + D33).  `S` is
total on the *lattice*, and since D31 the tier builds odd-`⟨Σ⁺, m⟩` charges too — so
SO(3)'s spinorial coweight, which this module used to report as unbuildable, is no
longer the obstacle.  What remains is one presentation limit: where the magnetic
**parity** `π = ⟨Σ⁺, ·⟩ mod 2` differs across `S`, the two sides' one-sided
zero-mode frames are offset by half a unit, so the twist `κ` is half-integral and no
*integer* flavour label can carry it.  That is a limit on writing the map, not on the
duality — `δκ` stays integral, so multiplication never sees the half (user, 2026-07-30:
*"the presentation of canonicals may not respect SU(2), but the multiplication will"*).
`dual_pairs` therefore reports which labels are jointly presentable rather than
assuming any are, `verify_langlands` runs the `KAlgebraIso` battery on that set, and
`langlands_frame_coboundary` certifies the rest.

Run the battery: `python3 run_tests.py`
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from global_form import (adjoint_lines, langlands_label_map,
                         simply_connected_lines)
from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly


def langlands_dual_algebra(alg):
    """The dual theory `(G^∨, N)` of a `GNAbeKAlgebra` — same class, dual
    `LineLattice`.

    The matter is carried across **unchanged**, which is a statement and not a
    convenience: it is legitimate exactly when every matter weight is a weight of
    the dual global form too.  For the adjoint representation that is automatic
    (the centre acts trivially on it), and `GNAbeKAlgebra.__init__` re-checks it
    against `lines.elec_admits` on construction, so an illegitimate transport
    raises there rather than producing a wrong algebra here.

    **When the datum itself moves** (`B_n ↔ C_n`, `n ≥ 3`) "unchanged" is not even
    well posed — a weight of `G` is a tuple in `G`'s coordinates, and `G^∨` has its
    own.  The one matter content that transports canonically is the **adjoint**,
    which is the highest root of whichever datum it lives on, so that is what this
    carries across; anything else honest-fails rather than reinterpreting a
    coordinate tuple in a system where it means something different."""
    from g_matter_roster import highest_root
    from gn_abe_kalgebra import GNAbeKAlgebra
    dual_lines = alg.lines.langlands_dual()
    dual_datum = dual_lines.datum
    if dual_datum is alg.datum or dual_datum.name == alg.datum.name:
        matter = (alg.matter[0] if len(alg.matter) == 1 else list(alg.matter))
        nf = len(alg.matter) if len(alg.matter) == 1 else None
        return GNAbeKAlgebra(alg.datum, matter, nf=nf, allow_solve=True,
                             lines=dual_lines)
    # cross-datum: only the adjoint transports canonically
    hr = tuple(highest_root(alg.datum))
    if len(alg.matter) != 1 or tuple(alg.matter[0]) != hr:
        raise NotImplementedError(
            f"langlands_dual_algebra: {alg.datum.name} → {dual_datum.name} moves "
            f"the root datum, and only the ADJOINT transports canonically across "
            f"that (it is the highest root of each side).  Matter {alg.matter} is "
            f"not the adjoint, and reinterpreting its coordinate tuple in the "
            f"dual's convention would silently change the representation.")
    return GNAbeKAlgebra(dual_datum, highest_root(dual_datum), nf=1,
                         allow_solve=True, lines=dual_lines)


def rho_flavour_shift(alg, m, e):
    """`δ(m, e)` — how `ρ` moves the **flavour** label, measured on the algebra.

    On the matter tier `ρ` acts on the μ-level as the *affine* involution
    `k ↦ −k − D(m)`, with `D` the matter **zero-mode count** at the cell
    (`matter_wrq_torus._rungs_ED`), so `δ = −D(m)` on a label whose flavour is the
    unit.  Measured, not assumed: a monopole at `m = (1,0)` of `SU(3)+Adj` carries 4
    rungs per non-extremal cell and gives `δ = (−4,)`, the fractional-coweight
    monopole `m = (2/3, 1/3)` of `PSU(3)+Adj` carries 2 and gives `(−2,)`, and every
    Wilson line carries 0 and gives `(0,)`."""
    la = alg.fold(tuple(m), tuple(e))
    (_g_in, w_in), (_g_out, w_out) = la, alg.rho(la)
    return tuple(a - b for a, b in zip(w_out, w_in))


def langlands_frame_offset(alg, dual, m, e, image=None):
    """`κ(m, e) ∈ ½Z` — the **exact** twist, as `Fraction`s.  Never raises.

    This is the primary accessor; `langlands_flavour_twist` is the integer-label
    wrapper over it.  Split out (2026-07-30) because `κ` is half-integral on a
    definite `Z/2` of labels and that half is *real data about the two frames*, not
    an error to be raised past: the honest object is the Fraction, and only the step
    that must write an integer flavour **label** can fail.

    **What the half means — the presentation, not the physics** (user ruling,
    2026-07-30: *"the presentation of canonicals may not respect SU(2), but the
    multiplication will"*, and earlier the same day: the flavour symmetry of
    `(G, Adj)` is `SU(2)` with the adjoint hyper a **doublet**, so *"I do not see
    how the quantization could be modified"* — closing the double-cover reading).

    `_flavour_rungs` emits one rung per matter weight with `⟨m, w⟩ < 0`, each
    carrying `μ^{+1}`, so the tier's zero-mode count is **one-sided**:
    `D(m) = ⟨Σ⁺, dominant(m)⟩`, and the μ-levels at a cell run `0 … D` with centre
    `D/2`.  The tier's integer level is therefore an *offset* frame, the offset being
    `D(m)/2` — half a unit exactly when `⟨Σ⁺, m⟩` is odd.  `κ` is nothing but the
    **difference of the two sides' offsets**, so in the centred (μ-symmetric) weight
    `ŵ = w + D/2` the Langlands twist is **identically zero**: the duality does not
    move the flavour at all.

    Why that is consistent with `SU(2)` flavour rather than a violation of it: the
    half sits in the *absolute* offset, which only the labelling sees.  `D` is
    additive on the dominant cone, so `δ(D/2) = 0` (measured 9/9 and 16/16, the
    latter over half-integral coweights), and `κ mod Z` is a **character**
    (measured 225/225) — bubbling moves `m` by coroots and `π` kills `Q^∨`
    (`⟨Σ⁺, α_i^∨⟩ = 2`), dually `e` moves by roots.  Hence the coboundary
    `δκ(a, b; z) = κ(z) − κ(a) − κ(b)` is **always an integer**, even where `κ` is
    not (measured 17/17 on the real support of real products, `e = 1` Wilson squared
    included) — so **multiplication never sees the half**, which is the ruling above
    verbatim.  This is the same shape as the atom-phase resolution in ruling D31,
    where a half-integral `S̃` is legitimate because only its integral coboundary
    `δS̃` ever reaches the cocycle.

    `image` is the charge `S(m, e)` already computed by the caller.  It is a
    parameter rather than recomputed here because the **inverse** direction needs
    `S^{-1}`, and an earlier version that always applied `S` asked `ρ` about a label
    that does not exist on the target — surfacing as
    `"rho: image is not a single canonical (0 terms)"`."""
    if image is None:
        fwd, _ = langlands_label_map(alg.datum)
        image = fwd(tuple(m), tuple(e))
    m2, e2 = image
    ds = rho_flavour_shift(alg, m, e)
    dd = rho_flavour_shift(dual, m2, e2)
    return tuple(Fraction(a - b, 2) for a, b in zip(dd, ds))


def langlands_frame_coboundary(alg, dual, a, b, dual_alg=None):
    """`(ok, rows)` — `δκ(a, b; z) = κ(z) − κ(a) − κ(b)` over the **real support**
    of `a·b`, with `ok` asserting every value is an integer.

    This is the certificate that the Langlands map is an algebra isomorphism *even
    where its label shift is half-integral*, and it is the only leg that speaks to
    the user's criterion directly (2026-07-30: *"the presentation of canonicals may
    not respect SU(2), but the multiplication will"*).  Multiplicativity constrains
    `κ` only through this coboundary — the structure constants must absorb exactly
    `δκ` — so an integral `δκ` on a half-integral `κ` says the half is invisible to
    the product.

    Measured on the real support, not argued from the lattice: `δκ` is **not** zero
    (the bubbling terms of `L_{m=1}·L_{m=1}` give `−1` and `−2`), so this is a live
    condition and not a tautology.  `a` and `b` are `(m, e)` charge pairs on `alg`."""
    if dual_alg is None:
        dual_alg = dual
    off = {}

    def kap(m, e):
        key = (tuple(m), tuple(e))
        if key not in off:
            off[key] = langlands_frame_offset(alg, dual_alg, key[0], key[1])
        return off[key]

    la = alg.fold(tuple(a[0]), tuple(a[1]))
    lb = alg.fold(tuple(b[0]), tuple(b[1]))
    ka, kb = kap(la[0][0], la[0][1]), kap(lb[0][0], lb[0][1])
    rows, ok = [], True
    for (mz, ez), _wz in alg.multiply(la, lb).terms:
        kz = kap(mz, ez)
        dk = tuple(z - x - y for z, x, y in zip(kz, ka, kb))
        good = all(v.denominator == 1 for v in dk)
        ok = ok and good
        rows.append(((mz, ez), kz, dk, good))
    return ok, rows


def langlands_flavour_twist(alg, dual, m, e, image=None):
    """`κ(m, e)` as an **integer** flavour-level shift — the label-writing wrapper
    over `langlands_frame_offset`, which carries the mathematics.

    Raises when the exact `κ` is half-integral.  That is a limitation of *writing a
    label*, not a statement that the duality fails there — see
    `langlands_frame_offset` for why the half is a frame offset the product never
    sees, and `langlands_frame_coboundary` for the certificate.

    **Why there is one at all.**  `S` exchanges magnetic and electric charge, and the
    number of matter **zero modes** is a function of the *magnetic* charge only — so
    it is emphatically not `S`-invariant: `S` trades a monopole (many zero modes) for
    a Wilson line (none).  Since `ρ` moves the flavour level by exactly `−D(m)`, a
    map that fixed the flavour could not be ρ-equivariant, and the identity-on-flavour
    version measurably is not (the charge parts agree; only the flavour label differs).

    **What it has to be.**  Writing the map as `((m,e), w) ↦ (S(m,e), w + κ)`,
    ρ-equivariance reads

        κ(ρ(m,e))  +  κ(m,e)  =  δ_dual(S(m,e))  −  δ_source(m,e)

    and is solved by **half the change in the zero-mode count**,

        κ(m, e)  =  ½·( δ_dual(S(m,e))  −  δ_source(m,e) ).

    So κ is *determined* by ρ rather than fitted, which is what makes the other legs
    of the battery — multiplicativity and trace-equivariance — genuine independent
    tests of it rather than restatements.

    **Honest-fails on an odd difference — and what that does and does not mean.**  The
    flavour *label* is an integer (a μ-power), so a half-unit twist cannot be written
    as one.  It was read here until 2026-07-30 as *"the duality is projective on the
    flavour grading"*; that reading is **retracted** (ruling D33).  The half is the
    mismatch between the two sides' one-sided level frames — the offsets are
    `D_src(m)/2` and `D_dual(S m)/2`, so `κ` is half-integral exactly when the
    magnetic **parity** `π = ⟨Σ⁺, ·⟩ mod 2` differs across `S` (measured 23/23,
    equivalently ⟺ `e` odd), the same `Z/2` character as ruling D20/D23.  In the
    centred weight `ŵ = w + D/2` the twist is identically zero, and `δκ` is integral
    throughout, so the algebra isomorphism is there — only this label-level
    presentation of it is not.  (It is not a `𝖖^{1/2}` either: the grading is in μ,
    not 𝖖, so the standing ruling is not in play.)"""
    kap = langlands_frame_offset(alg, dual, m, e, image=image)
    out = []
    for k in kap:
        if k.denominator != 1:
            raise NotImplementedError(
                f"Langlands flavour twist at (m={tuple(m)}, e={tuple(e)}): the exact "
                f"κ = {k} is ODD/2, so it is not an integer flavour LEVEL and no "
                f"integer-labelled map can be written here.  This is a FRAME mismatch, "
                f"not a projective anomaly: the two sides' one-sided zero-mode frames "
                f"differ by half a unit whenever the magnetic parity π = ⟨Σ⁺,·⟩ mod 2 "
                f"changes across S.  δκ stays integral, so multiplication does not see "
                f"it — see langlands_frame_offset / langlands_frame_coboundary.")
        out.append(int(k))
    return tuple(out)


def langlands_iso(alg, dual=None, name=None) -> KAlgebraIso:
    """The Langlands `KAlgebraIso` for a `GNAbeKAlgebra`: `S` on the charge and the
    **zero-mode twist** `κ` on the flavour (`langlands_flavour_twist`).

    The flavour is *not* fixed, and that is the substantive content — see
    `langlands_flavour_twist`.  An earlier version of this function used the identity
    on flavour, on the reasoning that `S` acts on gauge charge while the flavour
    symmetry is untouched by it; the battery refuted that (ρ-, trace- and
    multiplicative-equivariance all failed while the *charge* parts agreed exactly),
    which is what located the twist.

    The two label maps are `S` and `S^{-1}`, not `S` twice: `S² = −1` is charge
    conjugation, so using `S` as its own inverse would be wrong by a sign on both
    charges."""
    if dual is None:
        dual = langlands_dual_algebra(alg)
    fwd, inv = langlands_label_map(alg.datum)
    one = LaurentPoly.one()

    def _map(source, target, f):
        def go(label):
            (m, e), w = label
            m2, e2 = f(tuple(m), tuple(e))
            k = langlands_flavour_twist(source, target, m, e, image=(m2, e2))
            w2 = target.coefficient_ring().reduce(
                tuple(a + b for a, b in zip(w, k))) if hasattr(
                    target.coefficient_ring(), "reduce") else tuple(
                        a + b for a, b in zip(w, k))
            return Element({(target.fold(m2, e2)[0], tuple(w2)): one})
        return go

    return KAlgebraIso(
        alg, dual,
        forward_label_map=_map(alg, dual, fwd),
        inverse_label_map=_map(dual, alg, inv),
        name=name or f"Langlands[{alg.theory} ↔ {dual.theory}]",
    )


def verify_trace_up_to_twist(alg, dual, labels, K: int = 8):
    """`(ok, rows)` — the trace is equivariant **up to the flavour twist**:

        Tr_source(x)  =  χ_κ · Tr_dual(S x).

    This is a *restatement* of `KAlgebraIso.verify_trace_equivariant`, not a weakening
    of it, and both now hold: the stock verifier passes because `κ` is already part of
    the iso's label map, and this one passes because it applies the same `χ_κ`
    explicitly to the untwisted dual trace.  Keeping both is deliberate — they failed
    together under the Nahm-window truncation bug and would fail together again, so
    the pair is a cheap cross-check on that specific regression.

    Measured at `SU(3)+Adj`: the monopole `m = (1,0)` has `κ = 2` and its trace's
    flavour labels are uniformly the dual's plus 2 — `{1,3}` vs `{−1,1}` at `𝖖¹`,
    `{0,4}` vs `{−2,2}` at `𝖖²`, and so on.

    Implemented as multiplication by `χ_κ` in the flavour ring rather than by
    shifting label integers, so it stays correct if the ring is not `R(U(1))` — but
    `κ` itself is currently computed as an integer tuple, which is the **abelian**
    reading, so a higher-rank flavour group (several identical hypers ⇒ `U(n)`, whose
    labels are partitions) honest-fails in `langlands_flavour_twist` rather than
    being silently mis-added.

    **Each row carries a `vacuous` flag, and it is load-bearing.**  When both traces
    are identically zero the comparison passes for *every* κ and is no evidence at
    all — measured: at `SU(3)+Adj` the Wilson↔fractional-coweight pairs have zero
    trace to `𝖖¹⁰`, and every κ from −6 to +6 "matched exactly" there.  A caller that
    counts those as confirmations will conclude the twist is verified when nothing was
    tested, so `ok` below ignores them and reports only the rows that discriminate."""
    from zplus_ring import RElement, RPowerSeries
    fwd, _ = langlands_label_map(alg.datum)
    R = alg.coefficient_ring()
    rows, ok, tested = [], True, 0
    for (m, e) in labels:
        m, e = tuple(m), tuple(e)
        img = fwd(m, e)
        k = langlands_flavour_twist(alg, dual, m, e, image=img)
        ta = alg.trace(alg.fold(m, e), K)
        tb = dual.trace(dual.fold(*img), K)
        chi = RPowerSeries(R, {0: RElement(R, {tuple(k): 1})}, K)
        tw = chi * tb
        vacuous = ta.is_zero() and tw.is_zero()
        same = (ta == tw)
        first_diff = None
        if not same:
            for qe in range(-K, K + 1):
                if ta[qe] != tw[qe]:
                    first_diff = qe
                    break
        if not vacuous:
            tested += 1
            ok = ok and same
        rows.append(((m, e), img, k, same, vacuous, first_diff))
    return (ok and tested > 0), rows


def dual_pairs(alg, labels, dual=None):
    """`(dual, jointly_buildable, skipped)` — which `(m, e)` have **both** sides
    buildable, so the iso battery has something honest to run on.

    Reported rather than assumed: `S` is total on the lattice, but a pair may fail
    to be usable for two distinct reasons, and neither is evidence for or against
    the duality:

    * one side does not **chart** (a cost or scope limit of the builder);
    * the **flavour twist** `κ` is half-integral — `langlands_flavour_twist`
      honest-fails when the zero-mode-count change is odd, because no *integer*
      flavour label can carry a half unit.  It surfaced once the odd-`⟨Σ⁺,m⟩`
      charges became buildable (ruling D31) — at `SU(2)+Adj ↔ SO(3)+Adj` the Wilson
      line `(m=0, e=1)` maps to SO(3)'s spinorial 't Hooft `(m=ω^∨, e=0)`, whose
      single matter zero mode makes `κ = −1/2`.  Before D31 the SO(3) side did not
      build, so the pair was skipped for the *other* reason and this one was
      invisible.

      **This skip is a presentation limit, not a failure of the duality** (ruling
      D33, retracting the earlier "the duality is projective on the flavour grading"
      reading).  The half is the offset between the two sides' *one-sided* zero-mode
      frames; `δκ` is integral throughout, so the product never sees it, and the
      algebra isomorphism exists at these labels even though this integer-labelled
      map cannot be written.  `langlands_frame_offset` gives the exact `κ` and
      `langlands_frame_coboundary` the certificate."""
    if dual is None:
        dual = langlands_dual_algebra(alg)
    fwd, _ = langlands_label_map(alg.datum)
    good, skipped = [], []
    for (m, e) in labels:
        m, e = tuple(m), tuple(e)
        m2, e2 = fwd(m, e)
        why = None
        # the reason is kept in FULL, not truncated: an earlier draft clipped it to
        # 60 chars and cut the words "odd ⟨Σ⁺,m⟩" off the SU(2)↔SO(3) skip, i.e. it
        # hid exactly the diagnostic the skip exists to report
        try:
            alg.chart(alg.fold(m, e))
        except Exception as ex:
            why = f"source {type(ex).__name__}: {ex}"
        if why is None:
            try:
                dual.chart(dual.fold(m2, e2))
            except Exception as ex:
                why = f"target {type(ex).__name__}: {ex}"
        if why is None:
            # both sides chart — but the map itself may still not exist here
            try:
                langlands_flavour_twist(alg, dual, m, e, image=(m2, e2))
            except NotImplementedError as ex:
                why = f"twist {type(ex).__name__}: {ex}"
        (skipped if why else good).append(
            ((m, e), (m2, e2), why) if why else ((m, e), (m2, e2)))
    return dual, good, skipped


def _multipliable(alg, dual, good, product_labels=None):
    """`(src_pairs, tgt_pairs, dropped, used, nontrivial)` — the label pairs whose
    **product** both sides can actually build.

    `product_labels` restricts which source labels are paired up.  It exists purely
    for **cost**: an unbuildable product is discovered by *attempting* it, and at
    `SU(3)+Adj` each monopole×monopole attempt spends ~27 s before the μ-divisibility
    guard refuses it.  Restricting the set is therefore a way to skip known-expensive
    *failures*, never a way to hide them — anything excluded is still listed in
    `dropped` with the reason `"not attempted (cost)"`, so a reader can tell the
    difference between "tried and refused" and "not tried".

    Needed because a product leaves the buildable region: at `SU(3)+Adj`, squaring
    `L_{(1,1),0}` asks for `L_{(2,2),0}`, whose `μ`-sector `(2,)` the divisibility
    guard refuses to pin.  That is the guard working — `solve_canonical_matter`
    raises rather than returning an unpinned candidate — but it means the
    multiplicativity leg has to *choose* its pairs and say which it dropped, rather
    than assume a full square is available.  Silently dropping them would make a
    near-vacuous multiplicativity check look like a strong one."""
    one = LaurentPoly.one()
    z = (0,) * alg.datum.dim
    allow = None if product_labels is None else {
        (tuple(m), tuple(e)) for (m, e) in product_labels}
    src_pairs, tgt_pairs, dropped, used = [], [], [], []
    for i, ((m1, e1), (dm1, de1)) in enumerate(good):
        for (m2, e2), (dm2, de2) in good[i:]:
            if allow is not None and not (
                    (m1, e1) in allow and (m2, e2) in allow):
                dropped.append(((m1, e1), (m2, e2), "not attempted (cost)"))
                continue
            la, lb = alg.fold(m1, e1), alg.fold(m2, e2)
            ta, tb = dual.fold(dm1, de1), dual.fold(dm2, de2)
            try:
                alg.multiply(la, lb)
                dual.multiply(ta, tb)
            except Exception as ex:
                dropped.append(((m1, e1), (m2, e2),
                                f"{type(ex).__name__}: {str(ex)[:70]}"))
                continue
            src_pairs.append((Element({la: one}), Element({lb: one})))
            tgt_pairs.append((Element({ta: one}), Element({tb: one})))
            used.append(((m1, e1), (m2, e2)))
    nontrivial = [(a, b) for (a, b) in used
                  if a != (z, z) and b != (z, z)]
    return src_pairs, tgt_pairs, dropped, used, nontrivial


def verify_langlands(alg, labels, dual=None, trace_K: int = 10,
                     product_labels=None) -> dict:
    """Run the full `KAlgebraIso` battery for the Langlands map on the jointly
    buildable labels, and report everything that was skipped.

    Returns `{"iso", "checks", "pairs", "skipped", "products", "dropped"}`.
    `checks` is `KAlgebraIso.verify_all`'s summary — unit, round trip,
    multiplicativity, ρ-equivariance, trace-equivariance — which is the whole
    mathematical content; this module's job is only to hand it the right label sets
    and to be honest about which ones it could not supply.

    `products` lists the pairs the multiplicativity leg actually ran on,
    `nontrivial_products` the subset where **neither** factor is the identity, and
    `dropped` says which were unavailable and why.  The distinction matters: a
    multiplicativity check over only `(x, 1)` pairs is nearly vacuous and must not be
    mistaken for a strong one, so callers should assert `nontrivial_products` is
    non-empty rather than trusting a green `multiplicative`."""
    dual, good, skipped = dual_pairs(alg, labels, dual=dual)
    iso = langlands_iso(alg, dual=dual)
    one = LaurentPoly.one()
    src = [Element({alg.fold(m, e): one}) for (m, e), _ in good]
    tgt = [Element({dual.fold(m2, e2): one}) for _, (m2, e2) in good]
    src_pairs, tgt_pairs, dropped, used, nontrivial = _multipliable(
        alg, dual, good, product_labels=product_labels)
    checks = iso.verify_all(src, tgt, src_pairs, tgt_pairs, trace_K=trace_K)
    # the stock trace leg compares UNTWISTED and so reports the twist itself as a
    # mismatch; replace it with the correct statement and keep the raw verdict
    # alongside rather than quietly dropping it
    tw_ok, tw_rows = verify_trace_up_to_twist(
        alg, dual, [p[0] for p in good], K=trace_K)
    checks["trace_equivariant_untwisted"] = checks.pop("trace_equivariant")
    checks["trace_equivariant_up_to_twist"] = tw_ok
    return {"iso": iso, "checks": checks, "pairs": good, "skipped": skipped,
            "products": used, "nontrivial_products": nontrivial,
            "dropped": dropped, "trace_rows": tw_rows}


def adjoint_theory(datum, lines=None):
    """`(G, Adj)` — `G` with one adjoint hypermultiplet, the `N = 2*` theory, at a
    given global form (default simply connected).

    The adjoint is picked out as the **highest root**, which is a weight of every
    global form of `G`, so it is the one matter content the Langlands map can carry
    across without leaving the set of representations."""
    from g_matter_roster import highest_root
    from gn_abe_kalgebra import GNAbeKAlgebra
    return GNAbeKAlgebra(datum, highest_root(datum), nf=1,
                         lines=lines or simply_connected_lines(datum))


def langlands_family(data, labels_for=None, theory_for=None, trace_K: int = 10,
                     product_labels=None):
    """The **family** of Langlands `KAlgebraIso` (user, 2026-07-28: *"you can
    formalize by a Langlands family of KAlgebraIso"*) — one entry per member, each an
    iso plus its verified battery.

    Each entry of `data` is `(tag, x)` where `x` is **either**

    * a `RootDatum` — the member theory is then `theory_for(datum)`, defaulting to
      `adjoint_theory` (i.e. `(G, Adj)` at the simply connected form); **or**
    * a ready-made `GNAbeKAlgebra` — used as is, so a family may mix matter
      contents, multiplicities and starting global forms.

    The second form is what makes this as general as `langlands_iso` itself, which
    accepts any `GNAbeKAlgebra` (any `(G, N)`, any `nf`, either global form, and it
    follows the datum across `B_n ↔ C_n`).  Before 2026-07-29 this function
    hardwired `adjoint_theory(datum)`, so the *family* was adjoint-only even though
    the underlying constructor was not.

    `labels_for(datum)` supplies the labels to certify on; the default is the
    identity, the simple monopoles, and the fundamental Wilson lines — the last of
    which are precisely what `S` sends to fractional-coweight monopoles.

    `trace_K` and `product_labels` are forwarded to `verify_langlands`.  They matter
    in practice, not just for completeness: the trace legs dominate the cost, so a
    family run at the default `trace_K = 10` is far more expensive than one at
    `trace_K = 2`, and before 2026-07-29 there was no way to ask for the cheap one.

    A member whose dual cannot be formed (no dual datum, or matter that does not
    transport across a datum change) is reported as `{"tag", "skipped"}` rather than
    raising, so one impossible member does not lose the rest of the family."""
    def _default(datum):
        d = datum.dim
        z = (0,) * d
        basis = [tuple(1 if k == i else 0 for k in range(d)) for i in range(d)]
        return [(z, z)] + [(b, z) for b in basis] + [(z, b) for b in basis]

    labels_for = labels_for or _default
    theory_for = theory_for or adjoint_theory
    out = []
    for tag, x in data:
        # a member may be given as a RootDatum (build `theory_for` on it) or as a
        # ready-made algebra (used as is) — distinguished by the algebra surface,
        # not by isinstance, so any KAlgebra carrying `lines` + `datum` qualifies
        is_alg = hasattr(x, "lines") and hasattr(x, "datum")
        datum = x.datum if is_alg else x
        try:
            alg = x if is_alg else theory_for(datum)
            res = verify_langlands(alg, labels_for(datum), trace_K=trace_K,
                                   product_labels=product_labels)
        except NotImplementedError as ex:
            out.append({"tag": tag, "skipped": str(ex)[:120]})
            continue
        res["tag"] = tag
        out.append(res)
    return out
