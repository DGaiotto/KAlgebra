"""The flavoured F-finder — `F·S = X_γ + O(\\fq)` with a non-Abelian flavour `G`.

The author's direction, 2026-09-14: *"agreed, flavoured F-finder"*, following the
assessment that a `∏_a SU(N_a)`-flavoured BPS chart is well posed
(a probe in the source repository: `σ`, `F` and the structure
constants are all equivariant for the node-permutation action realising `W(G)`,
115/115).

What it solves
--------------

On the flavoured quantum torus of `flavoured_factor_spectrum` — reduced gauge
charges `γ ∈ Γ`, flavour weights carried separately as central `\\fq`-degree-zero
characters `μ^w` — the canonical-basis element attached to the enlarged charge
`(γ, w)` is the unique `F` with

    F · S  =  μ^w X_γ  +  O(\\fq) ,        `f_{(γ,w)} = 1` ,

`S` being the flavoured spectrum generator (`FlavouredFactorSpectrum`) and every
`f` bar-invariant (palindromic in `\\fq`).  This is the ordinary defining
relation of the repo's F-solver (`kalgebra.md` "F-finding";
`bps_kalgebra_internals.solve_F_via_s_coefficient`), carried over verbatim — the
flavour changes the ring the coefficients live in, not the relation.

Why it runs on the reduced cone
-------------------------------

Because the flavour directions are central, the quantum-torus phase does not see
them: `⟨(δ,U),(ζ,V)⟩ = ⟨δ,ζ⟩`.  So the relation is a convolution over the
**reduced** cone whose coefficients happen to be graded by flavour weight, and
the recursion is the unflavoured one with that grading along for the ride —
exactly as for `S`.

The forcing is in closed form
-----------------------------

Write the relation at a cell `(ε, W)` beyond the seed.  The term with `ζ = 0`
contributes `f_{(ε,W)}·S_0 = f_{(ε,W)}` — `S_0 = 1` — and every other term
involves an `f` at a **strictly lower** reduced charge.  So with

    R  :=  Σ_{ζ > 0}  f_{(ε-ζ, W-V)} · S_{(ζ,V)} · \\fq^{⟨ε-ζ, ζ⟩}

the two conditions — `f_{(ε,W)} + R` has no `\\fq^{≤0}` part, and `f_{(ε,W)}` is
palindromic — determine `f_{(ε,W)}` outright:

    f_j = −r_j  (j ≤ 0),   f_j = f_{−j}  (j > 0) ,

i.e. `f = −r_0 − Σ_{j<0} r_j (\\fq^j + \\fq^{−j})`.  No linear system is solved
and nothing is searched: the recursion is triangular in the cone order, which is
what makes `F` unique given the seed.  (This is the same shape as the `Ω`
forcing in `bps_factor_spectrum._forced`, with `S_0 = 1` in place of that
routine's `−L·Ω`.)

Truncation
----------

Cone-truncated on the reduced cone, and only there — never in `\\fq`
(the design notes, "Exact arithmetic until the very end").  The unflavoured solver uses
the **doubly-tropical interval** `[γ₋(a), γ⁺(a)]` as an exact support window,
which needs `σ`; this one does not assume a `σ` on the flavoured side and simply
solves out to the cutoff, so the support is an *output*.  Agreement with the
unfolded chart's support is therefore evidence rather than construction.
"""

from __future__ import annotations

from typing import Sequence

from habiro import HabiroElement
from laurent_poly import LaurentPoly

from bps_factor_spectrum import cone_simplex, expansion
from flavoured_factor_spectrum import (
    FlavouredFactorSpectrum,
    FlavouredQuiver,
    Irrep,
    Vec,
    Weight,
)
from recursive_spectrum import q_pow

H0 = HabiroElement.zero()
H1 = HabiroElement.one()


def _palindromic_kill(residual: HabiroElement) -> HabiroElement:
    """The unique palindromic `f` with `f + residual` free of `\\fq^{≤0}` terms.

    `f_j = −r_j` for `j ≤ 0`, extended by `f_j = f_{−j}`.  Returns the zero
    element when the residual already has positive `\\fq`-order, which is what
    makes the recursion terminate wherever the true `F` does.
    """
    if residual.is_zero():
        return H0
    r = expansion(residual)
    coeffs: dict[int, int] = {}
    for j, c in r.items():
        if j > 0 or not c:
            continue
        coeffs[j] = coeffs.get(j, 0) - c
        if j < 0:
            coeffs[-j] = coeffs.get(-j, 0) - c
    coeffs = {k: v for k, v in coeffs.items() if v}
    if not coeffs:
        return H0
    return HabiroElement.from_laurent(LaurentPoly(coeffs))


class FlavouredFSolver:
    """`F` for the canonical labels of a flavoured BPS quiver.

    Parameters
    ----------
    quiver
        A `FlavouredQuiver`.
    cutoff
        Reduced-cone degree truncation, for both `S` and `F`.
    spectrum
        An already-built `FlavouredFactorSpectrum` to take `S` from; one is
        built at the same cutoff if omitted.
    """

    def __init__(self, quiver: FlavouredQuiver, cutoff: int, *,
                 spectrum: FlavouredFactorSpectrum | None = None):
        self.quiver = quiver
        self.rank = quiver.rank
        self.degree_cap = int(cutoff)
        if spectrum is None:
            spectrum = FlavouredFactorSpectrum(quiver, cutoff)
            spectrum.run()
        elif spectrum._S is None:
            spectrum.run()
        self.spectrum = spectrum
        self._S = spectrum.spectrum_generator()
        # `S` indexed by reduced charge, for the convolution's inner loop.
        self._S_by_charge: dict[Vec, dict[Weight, HabiroElement]] = {}
        for (k, w), h in self._S.items():
            self._S_by_charge.setdefault(k, {})[w] = h
        self._cache: dict[tuple[Vec, Weight], dict] = {}

    # ----- the solve ----------------------------------------------------
    def solve(self, charge: Vec, weight: Weight | None = None
              ) -> dict[tuple[Vec, Weight], HabiroElement]:
        """`F` for the enlarged charge `(charge, weight)`, as `{(δ,U): f}`.

        `weight` defaults to the zero weight.  The seed `f_{(γ,w)} = 1` fixes
        the normalisation, exactly as unflavoured.
        """
        F = self.quiver.flavour
        charge = tuple(int(x) for x in charge)
        weight = F.zero if weight is None else tuple(int(x) for x in weight)
        key = (charge, weight)
        hit = self._cache.get(key)
        if hit is not None:
            return dict(hit)

        base_deg = sum(charge)
        out: dict[tuple[Vec, Weight], HabiroElement] = {key: H1}
        # Reduced charges `ε = γ + (something in the cone)`, in cone order.
        for step in cone_simplex(self.rank, self.degree_cap - base_deg):
            eps = tuple(charge[i] + step[i] for i in range(self.rank))
            if sum(eps) > self.degree_cap:
                continue
            # Accumulate the residual per weight at this reduced charge.
            resid: dict[Weight, list[HabiroElement]] = {}
            for zeta, block in self._S_by_charge.items():
                if not any(zeta):
                    continue                       # the `ζ = 0` term is `f` itself
                delta = tuple(eps[i] - zeta[i] for i in range(self.rank))
                if any(x < 0 for x in (delta[i] - charge[i]
                                       for i in range(self.rank))):
                    continue                       # `F` lives on `γ + cone`
                br = self.quiver.bracket(delta, zeta)
                phase = q_pow(br) if br else None
                for V, s_h in block.items():
                    for U in list(out):
                        if U[0] != delta:
                            continue
                        f_h = out[U]
                        if f_h.is_zero():
                            continue
                        term = f_h * s_h
                        if phase is not None:
                            term = term * phase
                        resid.setdefault(F.add(U[1], V), []).append(term)
            for W, terms in resid.items():
                if (eps, W) == key:
                    continue                       # the seed is not re-solved
                f_new = _palindromic_kill(HabiroElement.sum(terms))
                if not f_new.is_zero():
                    out[(eps, W)] = f_new
        out = {k: v for k, v in out.items() if not v.is_zero()}
        self._cache[key] = dict(out)
        return out

    def solve_irrep(self, charge: Vec, rep: Irrep
                    ) -> dict[tuple[Vec, Weight], HabiroElement]:
        """`Σ_{w ∈ r} F_{(γ,w)}`, which satisfies `F·S = χ_r(μ) X_γ + O(\\fq)`.

        The `R`-form readout: a sum of canonical-basis elements, one per weight
        of `r` with multiplicity — NOT itself a canonical-basis element.  The
        canonical basis stays indexed by enlarged charges (`kalgebra.md`,
        "Freeness over `R` — a convention, not a contract"); this is the
        rep-grouped combination the flavoured bookkeeping makes natural.
        """
        F = self.quiver.flavour
        out: dict[tuple[Vec, Weight], HabiroElement] = {}
        for w, mult in F.irrep_weights(rep).items():
            for k, h in self.solve(charge, w).items():
                term = h if mult == 1 else h * mult
                prev = out.get(k)
                out[k] = term if prev is None else prev + term
        return {k: v for k, v in out.items() if not v.is_zero()}

    # ----- verification -------------------------------------------------
    def verify_defining_relation(self, charge: Vec, weight: Weight | None = None
                                 ) -> list[tuple[Vec, Weight, str]]:
        """`F·S = μ^w X_γ + O(\\fq)`, checked on the truncated cone.

        **ENFORCED, not emergent**: the recursion forces exactly this, so a pass
        is a regression guard on the arithmetic and not evidence about anything
        (the audit).  It is here because the arithmetic is easy to get
        wrong, and labelled so that it is not mistaken for evidence.
        """
        F = self.quiver.flavour
        weight = F.zero if weight is None else tuple(weight)
        charge = tuple(charge)
        prod = self.product_with_S(self.solve(charge, weight))
        bad: list[tuple[Vec, Weight, str]] = []
        for (k, w), h in prod.items():
            if h.is_zero():
                continue
            exp = expansion(h)
            want_one = (k, w) == (charge, weight)
            nonpos = {j: c for j, c in exp.items() if j <= 0 and c}
            if want_one:
                if nonpos != {0: 1}:
                    bad.append((k, w, f"leading term {nonpos}, want {{0: 1}}"))
            elif nonpos:
                bad.append((k, w, f"non-positive \\fq powers {nonpos}"))
        return bad

    def product_with_S(self, F_dict) -> dict[tuple[Vec, Weight], HabiroElement]:
        """`F · S` on the truncated reduced cone."""
        Fl = self.quiver.flavour
        out: dict[tuple[Vec, Weight], HabiroElement] = {}
        for (delta, U), f_h in F_dict.items():
            if f_h.is_zero():
                continue
            for zeta, block in self._S_by_charge.items():
                eps = tuple(delta[i] + zeta[i] for i in range(self.rank))
                if sum(eps) > self.degree_cap:
                    continue
                br = self.quiver.bracket(delta, zeta)
                phase = q_pow(br) if br else None
                for V, s_h in block.items():
                    term = f_h * s_h
                    if phase is not None:
                        term = term * phase
                    key = (eps, Fl.add(U, V))
                    prev = out.get(key)
                    out[key] = term if prev is None else prev + term
        return {k: v for k, v in out.items() if not v.is_zero()}


def solve_flavoured_F(quiver: FlavouredQuiver, charge: Vec, cutoff: int, *,
                      weight: Weight | None = None):
    """`F` for one enlarged charge of a flavoured BPS quiver."""
    return FlavouredFSolver(quiver, cutoff).solve(charge, weight)


def enlarged_charge(quiver: FlavouredQuiver, charge: Vec,
                    weight: Weight | None = None) -> Vec:
    """Invert `(enlarged charge) ↦ (reduced charge, flavour weight)`.

    The projection is a bijection onto its image, because the dimensions match:
    the enlarged lattice has rank `Σ_a dim r_a`, and the target has rank
    `g + Σ_a (N_a − 1)`, the same number when each node carries the fundamental
    of its own factor.  So a canonical label of the unfolded chart is recovered
    from its gauge charge and flavour weight, and the two descriptions are
    interchangeable.

    For `SU(N)` with the fundamental, in the `to_abelian` coordinates
    `w_i = e_i` (`i < N`) and `w_N = (−1,…,−1)`, the inverse is explicit:
    `k_i − k_N = w_i` and `Σ k_i = γ` give `k_N = (γ − Σ_i w_i) / N`.  Integrality
    of `k_N` is the **N-ality condition** — `γ ≡ Σ_i w_i (mod N)` — and a
    `(γ, w)` failing it is not the image of any enlarged charge; this raises
    rather than rounding.
    """
    F = quiver.flavour
    weight = F.zero if weight is None else tuple(int(x) for x in weight)
    _, _, labels = quiver.unfold()
    index = {(a, w): i for i, (a, w) in enumerate(labels)}
    out = [0] * len(labels)
    for a in range(quiver.rank):
        fac = None
        for i, lam in enumerate(quiver.node_reps[a]):
            if lam:
                fac = i
                break
        if fac is None:                      # unflavoured node
            key = (a, F.zero)
            out[index[key]] = charge[a]
            continue
        N = F.factors[fac]
        lo, wd = F.offsets[fac], F.widths[fac]
        head = list(weight[lo:lo + wd])
        total = charge[a] - sum(head)
        if total % N:
            raise ValueError(
                f"enlarged_charge: node {a} has gauge charge {charge[a]} and "
                f"weight {head}, which fails the N-ality condition mod {N} — "
                f"no enlarged charge projects to it")
        k_last = total // N
        for i in range(N):
            w = [0] * F.dim
            if i < N - 1:
                w[lo + i] = 1
            else:
                for j in range(wd):
                    w[lo + j] = -1
            out[index[(a, tuple(w))]] = (head[i] + k_last if i < N - 1
                                         else k_last)
    return tuple(out)


def unfolded_F(quiver: FlavouredQuiver, charge: Vec, weight: Weight | None = None,
               *, cutoff: int | None = None):
    """The same `F`, from the UNFOLDED chart's own solver — ground truth.

    Runs `BPSKAlgebra`'s F-solver on the ordinary BPS quiver whose nodes are the
    weights of the node reps, then pushes the enlarged charges forward to
    `(reduced charge, flavour weight)`.  The two routes share the defining
    relation and nothing else: this one uses the doubly-tropical support window
    and the `[n]_q` peeling solver, the flavoured one a weight-graded recursion
    on the reduced cone with a closed-form forcing.
    """
    from bps_kalgebra import BPSKAlgebra

    F = quiver.flavour
    B, charges, labels = quiver.unfold()
    weight = F.zero if weight is None else tuple(weight)
    target = enlarged_charge(quiver, charge, weight)
    A = BPSKAlgebra(pairing=B, node_charges=charges)
    raw = A.F_qn(target)
    out: dict[tuple[Vec, Weight], HabiroElement] = {}
    for k, qn in raw.items():
        gauge = [0] * quiver.rank
        wt = F.zero
        for i, n in enumerate(k):
            if not n:
                continue
            a, w = labels[i]
            gauge[a] += n
            wt = F.add(wt, F.scale(w, n))
        out[(tuple(gauge), wt)] = HabiroElement.from_laurent(qn.to_laurent())
    return out
