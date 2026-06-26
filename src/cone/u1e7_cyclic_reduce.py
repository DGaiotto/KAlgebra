"""Multiply-based cyclic trace reduction for `U1E7ConeKAlgebra` (the QTCone where
the generic tagged-cyclicity `seed_reduction` is INCORRECT — it returns
cyclically-inconsistent reductions).

⚠ **SUPERSEDED** — the recursive `cyclic_reduce` below is itself *peel-order
dependent*: its termination heuristic "recurse on FEWER-atom daughters, keep
same-or-more-atom daughters as seeds" declares peel-order-dependent seeds, so
peeling different atoms yields different (individually invalid, mutually
contradictory) reductions.  `u1e7_trace_bootstrap` no longer uses it; it instead
takes the trace **flat** (each ρ²-canonical cone-word is its own symbol — no
peeling, trivially peel-order independent) and supplies the one-step cyclicity
relations explicitly as forward-sweep constraints.  Only the helpers
(`_nz`/`_lp`/`_natoms`) here are still imported.  The top-level peel identity
(`Tr(L_c)=Tr(L_rest·L_{ρ⁻²g1})`, ρ²-twisted cyclicity) is correct — it is the
*recursion's* seed-classification that breaks consistency.

`cyclic_reduce(A, c)` expresses `Tr(L_c) = Σ_seed red·Tr(L_seed)` using ONLY the
validated cone `multiply` and `ρ`.  The recipe (Form-B ρ²-cyclicity):

    L_{g1}·L_rest = q^e·L_c        (g1 = one cone-atom of c, rest = the others)
    Tr(L_c) = q^{-e}·Tr(L_rest·L_{ρ⁻²g1}) = q^{-e}·Σ_d C^d_{rest,ρ⁻²g1}·Tr(L_d)

Daughters `d` with FEWER atoms than `c` are recursed (the cross-product /
Plücker reductions); same-or-more-atom daughters are kept as **seeds** (the
rotation-stable cone-words the QTCone trace cannot reduce further — the analogue
of the `Tr(v²)=qTr(1)+q(q-1)Tr(v)` base).  Magnetic (`c0≠0`) seeds are 0.  All
seeds are ρ²-folded (`_canonical_rho2_orbit_rep`).

The seeds are pinned downstream by the orthonormality bootstrap (which also adds
the cyclicity relations `Tr(seed)=q^{-e}Σ C^d Tr(d)` — homogeneous rows that
relate the rotation-stable seeds, the prove_uniqueness step).
"""
from __future__ import annotations

from collections import Counter

from laurent_poly import LaurentPoly


def _nz(elt):
    return {l: c for l, c in elt.terms.items()
            if not (c.is_zero() if hasattr(c, "is_zero") else c == 0)}


def _lp(c):
    return c if isinstance(c, LaurentPoly) else LaurentPoly(dict(c._coeffs))


def _natoms(A, label):
    gens, powers = A.cone_data().to_cone_label(label)
    return sum(powers.values()), gens, powers


def cyclic_reduce(A, c, memo, _stack=None):
    """`{seed_rep: LaurentPoly}` with `Tr(L_c) = Σ red·Tr(L_seed)` via the
    validated multiply (cyclically consistent).  Recurses on atom-reducing
    daughters; rotation-stable (same/more atoms) daughters are seeds."""
    cd = A.cone_data()
    rep = A._canonical_rho2_orbit_rep(c)
    if A._mag_charge(rep) != 0:
        return {}
    if rep in memo:
        return memo[rep]
    nlet, gens, powers = _natoms(A, rep)
    if nlet <= 1:
        memo[rep] = {rep: LaurentPoly({0: 1})}
        return memo[rep]
    _stack = _stack or set()
    if rep in _stack:                       # self-loop ⇒ rotation-stable seed
        return {rep: LaurentPoly({0: 1})}

    # peel one cone-atom g1; rest = the others
    g1 = min((g for g in gens if powers[g] > 0), key=repr)
    rp = dict(powers)
    rp[g1] -= 1
    rgens = frozenset(g for g in gens if rp.get(g, 0) > 0)
    rest = cd.from_cone_label(rgens, {g: rp[g] for g in rgens})
    m1 = _nz(A.multiply(g1, rest))
    if rep not in m1 or len(m1) != 1:       # not a clean single-term peel ⇒ seed
        memo[rep] = {rep: LaurentPoly({0: 1})}
        return memo[rep]
    e_lp = _lp(m1[rep])
    (e_exp,), = [tuple(e_lp._coeffs)]        # monomial q^e
    inv = LaurentPoly({-e_exp: 1})

    g1i = A.rho_inverse(A.rho_inverse(g1))
    daughters = _nz(A.multiply(rest, g1i))

    out: dict = {}

    def acc(seed, poly):
        cur = out.get(seed)
        out[seed] = poly if cur is None else cur + poly

    for d, Cd in daughters.items():
        drep = A._canonical_rho2_orbit_rep(d)
        if A._mag_charge(drep) != 0:
            continue
        coeff = inv * _lp(Cd)
        dnlet, _, _ = _natoms(A, drep)
        if dnlet < nlet and drep != rep:
            for s, rs in cyclic_reduce(A, d, memo, _stack | {rep}).items():
                acc(s, coeff * rs)
        else:                                # rotation-stable ⇒ seed
            acc(drep, coeff)

    # self-loop: (1 - α)·Tr(rep) = Σ_{other}; treat rep as its own seed if α≠0
    # would need a non-monomial pivot — instead keep rep as a seed and let the
    # bootstrap's cyclicity row carry the relation.
    if rep in out:
        memo[rep] = {rep: LaurentPoly({0: 1})}
        return memo[rep]
    out = {s: p for s, p in out.items() if not p.is_zero()}
    memo[rep] = out
    return out
