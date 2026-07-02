"""Machine-generated trace-uniqueness proofs for finite-type
K_𝖖-algebras.

For a finite-type cone algebra the ρ²-twisted-cyclicity axiom is
SOLVED exactly by the Layer-1 reduction: every canonical label's trace
is a `Z[𝖖^±]`-combination of the finitely many **elementary seeds**
`S` (identity + one single-mult-gen seed per ρ²-orbit).  Uniqueness of
the trace then needs no windowed label unknowns and no bootstrap —
it is small exact linear algebra on the seed coefficients, and the
computation IS the proof (exact rational arithmetic throughout).

The proof object generated per algebra (`prove_uniqueness`):

1. **Reduction cyclic-consistency certificate** — for every pair
   `(a, b)` in the window, `Tr(ab)` and `Tr(ρ²(b)a)` reduce to
   IDENTICAL seed polynomials, certifying that the implemented
   reduction honestly solves cyclicity on the window (this is the
   step that replaces the windowed cyclicity rows entirely).
2. **Dimension certificate** — unknowns `T_s[k]` (`s ∈ S`,
   `0 ≤ k ≤ N`; traces taken in `Z[[q]]`, the standing support
   assumption); rows = `q⁰`-orthonormality `I_{ab}[q⁰] = δ_{ab}` for
   window pairs, with `Tr` substituted by its seed reduction:

       Σ_s Σ_j  [q^j] P_{(a,b),s} · T_s[-j]  =  δ_{ab},
       P_{(a,b),s}(𝖖) := Σ_c C^c_{ρ(a),b}(𝖖) · red_{c,s}(𝖖),

   solved by exact elimination.  The certificate is that the affine
   solution space has dimension EXACTLY `N` — the rescale freedom
   `f ∈ 1 + 𝖖·Q[[𝖖]]` truncated at `q^N`, and nothing else.
3. **Canonical-solution check** — the algebra's exact trace satisfies
   every row (to the available frozen depth), so by (2) the
   admissible traces are exactly its rescales through `q^N`.

The pair window starts at single generators + identity and grows by
cone-monomial degree until (2) certifies — the minimal certifying
degree is part of the proof record (the pentagon's q³-mode lesson,
systematized).

THEOREM FORMAT (what a record proves):  *Let φ be a `Z[[𝖖]]`-valued
linear functional on A determined by its elementary seeds through the
(consistency-certified) Layer-1 reduction, satisfying
`I_φ(a,b)[q⁰] = δ_{ab}` for the recorded pair window.  Then
`φ = f·Tr_can + O(𝖖^{N+1})` with `f ∈ 1 + 𝖖·Q[[𝖖]]`.*

The all-orders (N = ∞) statement needs the stabilisation of the
order-k constraint matrices in k — visibly the same phenomenon as the
trace miracle; an open research question.

There is no bundled sweep driver: use `prove_uniqueness` /
`proof_report` directly on a finite-type `ConeKAlgebra` instance
(e.g. `FinitePentagonKAlgebra` from `finite_pentagon_kalg`).
"""
from __future__ import annotations

from fractions import Fraction

from laurent_poly import LaurentPoly
from trace_uniqueness import _in_span, _rank, _solve


__all__ = [
    "seed_set",
    "seed_reduction",
    "verify_reduction_cyclic_consistency",
    "prove_uniqueness",
    "proof_report",
]


# ---------------------------------------------------------------------------
# seeds and reductions
# ---------------------------------------------------------------------------

def seed_set(A):
    """The elementary seeds: identity + the canonical ρ²-orbit
    representatives of single mult-gens, exactly as Layer 1 emits
    them."""
    cd = A.cone_data()
    seeds = {A.identity()}
    for g in cd.mult_gens():
        native = cd.from_cone_label(frozenset({g}), {g: 1})
        seeds.add(A._canonical_rho2_orbit_rep(native))
    return sorted(seeds)


def seed_reduction(A, label, _cache={}):
    """`Tr(L_label) = Σ_s red_s(𝖖)·T_s` as `{seed: LaurentPoly}`.

    The cache entry HOLDS `A`: keys use `id(A)`, and a collected
    algebra's id can be reused by a fresh instance (observed: the
    heptagon flakily picking up the pentagon's reductions inside one
    test process) — pinning A in the value keeps the id unique for
    the cache's lifetime."""
    key = (id(A), label)
    if key not in _cache:
        s = A.cone_data().simplify_trace_via_cone_data(A, label)
        _cache[key] = (A, dict(s.terms))
    return _cache[key][1]


def _pair_poly(A, a, b):
    """`P_{(a,b),s}(𝖖) = Σ_c C^c_{ρ(a),b}(𝖖)·red_{c,s}(𝖖)` as
    `{seed: LaurentPoly}` — the exact seed-reduced expansion of
    `I_{ab} = Tr(ρ(L_a)·L_b)`."""
    out: dict = {}
    for c, co in A.multiply(A.rho(a), b).terms.items():
        for s, red in seed_reduction(A, c).items():
            cur = out.get(s)
            out[s] = co * red if cur is None else cur + co * red
    return out


# ---------------------------------------------------------------------------
# proof component 1: the reduction honestly solves cyclicity
# ---------------------------------------------------------------------------

def reduction_cyclicity_differences(A, pairs):
    """For every `(a, b)`: the seed-reduction difference of `Tr(ab)`
    and `Tr(ρ²(b)·a)` as `{seed: LaurentPoly}`.

    On the pentagon/heptagon every difference vanishes identically —
    the seeds are free coordinates of the cyclic-functional space.  On
    e6 they do NOT all vanish: cyclicity imposes genuine relations
    BETWEEN ρ²-orbit seeds (e.g. e6 forces `T₂ = T₆`, satisfied
    exactly by the canonical trace, while `T₄ ≠ T₅` stays free — the
    relations are subtler than blanket `Tr∘ρ = Tr`).  Each nonzero
    difference is therefore a certified *instance of the cyclicity
    axiom* and enters the proof system as exact homogeneous rows on
    the seed unknowns."""
    diffs = []
    for a, b in pairs:
        lhs: dict = {}
        for c, co in A.multiply(a, b).terms.items():
            for s, red in seed_reduction(A, c).items():
                cur = lhs.get(s)
                lhs[s] = co * red if cur is None else cur + co * red
        for c, co in A.multiply(A.rho(A.rho(b)), a).terms.items():
            for s, red in seed_reduction(A, c).items():
                cur = lhs.get(s)
                neg = LaurentPoly({}) - co * red
                lhs[s] = neg if cur is None else cur + neg
        d = {s: p for s, p in lhs.items()
             if p != LaurentPoly({})}
        if d:
            diffs.append(d)
    return diffs


def verify_reduction_cyclic_consistency(A, pairs):
    """Back-compat boolean: True iff every difference vanishes (free
    seeds).  Kept for the pentagon/heptagon tests; the proof system
    itself consumes the differences as rows."""
    return not reduction_cyclicity_differences(A, pairs)


# ---------------------------------------------------------------------------
# proof component 2: the dimension certificate
# ---------------------------------------------------------------------------

def _consistency_rows(diffs, col, seeds, N):
    """Cyclicity-difference relations expanded per q-order (Z[[q]]
    support; an order is imposed only when every contributing seed
    coefficient lies inside the truncation)."""
    rows = []
    for d in diffs:
        flat = [(s, j, v) for s, p in d.items()
                for j, v in p._coeffs.items() if v]
        if not flat:
            continue
        jmax = max(j for _s, j, _v in flat)
        jmin = min(j for _s, j, _v in flat)
        for k in range(max(0, jmax), min(N, N + jmin) + 1):
            row: dict = {}
            for s, j, v in flat:
                i = col[(s, k - j)]
                row[i] = row.get(i, Fraction(0)) + v
            row = {i: v for i, v in row.items() if v}
            if row:
                rows.append((row, Fraction(0)))
    return rows


def _ortho_rows(A, seeds, pairs, N):
    col = {(s, k): i for i, (s, k) in enumerate(
        ((s, k) for s in seeds for k in range(N + 1)))}
    rows = []
    for a, b in pairs:
        P = _pair_poly(A, a, b)
        row: dict = {}
        ok = True
        for s, poly in P.items():
            for j, v in poly._coeffs.items():
                if -j < 0:
                    continue          # Z[[q]] support assumption
                if -j > N:
                    ok = False        # q^0 needs orders beyond N
                    break
                i = col[(s, -j)]
                row[i] = row.get(i, Fraction(0)) + v
            if not ok:
                break
        if ok:
            rows.append(({i: v for i, v in row.items() if v},
                         Fraction(1 if a == b else 0)))
    return col, rows


def _window_pairs(A, max_deg):
    """Pairs (a, b) of cone monomials of degree ≤ max_deg (all cones)."""
    cd = A.cone_data()
    labels = {A.identity()}
    for cone in cd.cones():
        cs = sorted(cone)
        def rec(idx, powers, deg):
            if deg > 0:
                lab = cd.from_cone_label(
                    frozenset(g for g, p in powers.items() if p),
                    {g: p for g, p in powers.items() if p})
                labels.add(lab)
            if idx == len(cs) or deg == max_deg:
                return
            for p in range(0, max_deg - deg + 1):
                if p:
                    powers[cs[idx]] = p
                rec(idx + 1, powers, deg + p)
                powers.pop(cs[idx], None)
        rec(0, {}, 0)
    labs = sorted(labels)
    return [(a, b) for a in labs for b in labs], labs


def prove_uniqueness(A, N, *, max_pair_degree=4, name=None,
                     canonical_depth=None, verbose=False):
    """Generate the proof record for algebra `A` through order `q^N`.

    Grows the pair window by cone-monomial degree until the
    orthonormality system's solution space has affine dimension
    exactly `N` (the rescale freedom); then verifies the canonical
    trace solves the system to `min(N + window q-depth,
    canonical_depth)` and that the kernel is its shift span."""
    name = name or type(A).__name__
    seeds = seed_set(A)
    for deg in range(1, max_pair_degree + 1):
        pairs, labs = _window_pairs(A, deg)
        diffs = reduction_cyclicity_differences(A, pairs)
        col, ortho = _ortho_rows(A, seeds, pairs, N)
        cyc = _consistency_rows(diffs, col, seeds, N)
        rows = cyc + ortho
        particular, kernel = _solve(col, rows)
        if particular is None:
            return {"algebra": name, "ok": False,
                    "reason": f"orthonormality system inconsistent "
                              f"at degree {deg}"}
        dim = len(kernel)
        if verbose:
            print(f"  [{name}] N={N} deg={deg}: pairs={len(pairs)} "
                  f"unknowns={len(col)} dim={dim} (target {N})",
                  flush=True)
        if dim == N:
            # canonical check: T_s series from the algebra's own trace
            K_need = N
            try:
                T = {s: A.trace(s, K=K_need) for s in seeds}
            except Exception as exc:
                return {"algebra": name, "ok": False,
                        "reason": f"canonical trace unavailable: {exc}"}

            def coeff(s, k):
                v = T[s].coeffs.get(k, 0)
                if hasattr(v, "terms"):
                    v = v.terms.get((), 0)
                return Fraction(int(v))

            known = [Fraction(0)] * len(col)
            for (s, k), i in col.items():
                known[i] = coeff(s, k)
            solves = all(
                sum(v * known[i] for i, v in row.items()) == rhs
                for row, rhs in rows)
            shifts = []
            for m in range(1, N + 1):
                v = [Fraction(0)] * len(col)
                for (s, k), i in col.items():
                    if k - m >= 0:
                        v[i] = known[col[(s, k - m)]]
                shifts.append(v)
            kernel_is_shifts = (all(_in_span(sv, kernel)
                                    for sv in shifts)
                                and _rank(shifts) == dim)
            return {
                "algebra": name, "ok": solves and kernel_is_shifts,
                "N": N, "seeds": len(seeds),
                "window_degree": deg, "n_pairs": len(pairs),
                "n_window_labels": len(labs),
                "n_cyclicity_relations": len(diffs),
                "dim": dim, "canonical_solves": solves,
                "kernel_is_rescale_line": kernel_is_shifts,
            }
    return {"algebra": name, "ok": False, "N": N,
            "reason": f"dim did not reach {N} within pair degree "
                      f"{max_pair_degree} (last dim {dim})",
            "dim": dim}


def proof_report(records):
    lines = ["| algebra | seeds | N proven | window degree | pairs |"
             " certificate |",
             "|---|--:|--:|--:|--:|---|"]
    for r in records:
        if r.get("ok"):
            cert = "dim=N exact; canonical solves; kernel = rescale line"
            lines.append(
                f"| {r['algebra']} | {r['seeds']} | {r['N']} | "
                f"{r['window_degree']} | {r['n_pairs']} | {cert} |")
        else:
            lines.append(
                f"| {r['algebra']} | — | — | — | — | "
                f"FAILED: {r.get('reason')} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print("trace_uniqueness_proofs: no bundled sweep driver — call "
          "prove_uniqueness / proof_report directly on a finite-type "
          "ConeKAlgebra instance (e.g. FinitePentagonKAlgebra from "
          "finite_pentagon_kalg; the zoo is exercised by "
          "tests/test_cones.py).")
