"""Spine-free vacuum trace ``Tr(1)`` via the exact Nahm sum on the BPS spec.

``Tr(1) = (q²;q²)_∞^{rk Γ_g} · ⟨S|S⟩``, the vacuum face of the Schur pairing.
For the identity there is no ``F`` to solve, so ``⟨S|S⟩`` is a pure sum of the
S-state Nahm coefficients ``s_γ`` over the charge lattice, grouped by gauge
class with the flavour exponents carried as coefficient-ring characters:

    Tr(1) = (q²;q²)_∞^g · Σ_{[η] ∈ Γ_g}  c([η]) · c([η]),
    c([η]) = Σ_{η' : sec(η') = sec(η)} s_{η'} · χ^{flav(η')}.

Every ingredient is spine-free — ``nahm_local`` (the exact Habiro ``s_γ``),
``snf_kernel`` (the gauge/flavour split of the antisymmetric form), ``qpoch``
(the prefactor) and ``lattice`` — so this reproduces the BPS-engine vacuum
trace **without any BPS/RG machinery**, and is exact to arbitrary q-order
(``K``).  The only theory-specific datum is the ordered BPS **spec** (the
shortened spectrum), tabulated below; it is computed once from the embedded BPS
quiver and frozen here as a literal (a *seed for the bootstrap*, not a truncated
answer — the Nahm sum itself runs to any K).

This is the single input the spine-free orthonormality bootstrap
(``trace_uniqueness_proofs`` + the per-flavour drivers) needs: every other
elementary-trace seed is pinned from ``Tr(1)`` by ``I_{a,b}=δ_{a,b}+O(q)``.
"""
from __future__ import annotations

from typing import Sequence

from lattice import Lattice
from snf_kernel import integer_kernel_and_section
from nahm_local import s_gamma_habiro, gammas_to_q_order
from qpoch import qpoch_infty
from habiro import HabiroElement
from zplus_ring import RPowerSeries


# Ordered BPS spec (shortened spectrum) per finite-zoo short id.  Computed once
# from the embedded BPS quiver; the antisymmetric pairing is read separately
# from each standalone's <PREFIX>_BPS_PAIRING.
SPECS: dict[str, list[tuple[int, ...]]] = {
    "pentagon": [(1, 0), (0, 1)],
    "heptagon": [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
    "a3": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    "a5": [(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0),
           (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)],
    "a7": [(1, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0),
           (0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 1, 0),
           (0, 0, 0, 0, 0, 0, 1)],
    "a1d3": [(1, 0, 0), (0, 1, 1), (0, 1, -1)],
    "a1d5": [(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0),
             (0, 0, 0, 1, -1), (0, 0, 0, 1, 1)],
    "a1d7": [(1, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0),
             (0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 1, -1),
             (0, 0, 0, 0, 0, 1, 1)],
    "a1d4": [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, -1), (0, 0, 1, 1)],
    "a1d6": [(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0),
             (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, -1), (0, 0, 0, 0, 1, 1)],
    "a1d8": [(1, 0, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0),
             (0, 0, 1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0, 0),
             (0, 0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0, 0),
             (0, 0, 0, 0, 0, 0, 1, -1), (0, 0, 0, 0, 0, 0, 1, 1)],
    "e6": [(1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1),
           (0, 0, 0, 0, 1, 0), (0, 0, 0, 1, 0, 0), (0, 1, 0, 0, 0, 0)],
    "e7": [(1, 0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0),
           (0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 1, 0),
           (0, 0, 0, 1, 0, 0, 0)],
    "e8": [(1, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0, 0),
           (0, 0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 0, 1),
           (0, 0, 0, 0, 0, 1, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0),
           (0, 0, 0, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0)],
}


def has_spec(short_id: str) -> bool:
    return short_id in SPECS


def vacuum_trace_rps(spec_t: Sequence[Sequence[int]],
                     pairing: Sequence[Sequence[int]],
                     R, K: int) -> RPowerSeries:
    """Exact vacuum trace ``Tr(1)`` over the coefficient ring ``R``, to
    q-order ``K``, computed spine-free from the spec and the antisymmetric
    form ``pairing``."""
    spec_t = [tuple(int(x) for x in g) for g in spec_t]
    N = len(spec_t)
    lat = Lattice(pairing)
    rank = lat.rank
    kmat = [[lat.bracket(spec_t[i], spec_t[j]) for j in range(N)]
            for i in range(N)]
    ker, sec = integer_kernel_and_section(pairing)
    g = len(sec)
    f = len(ker)

    # stabilize=True: Tr(1) is the universal trace anchor, so its Nahm γ-set
    # must be CERTIFIED complete -- for a mixed-sign pairing the default DFS
    # bound is not provably sound and could silently drop a shift-≤-K tuple.
    # The guard auto-widens until the γ-set is stable (or raises), so Tr(1) is
    # exact rather than silently truncated.
    gammas = set(gammas_to_q_order(spec_t, kmat, K, rank=rank, stabilize=True))
    gammas.add(tuple([0] * rank))

    # c([η]) grouped by gauge class: list of (flavour exponent, s_η Habiro)
    by_class: dict[tuple, list] = {}
    for eta in gammas:
        s = HabiroElement.one() if not any(eta) else s_gamma_habiro(eta, spec_t, kmat)
        if s.is_zero():
            continue
        sec_c, flav_c = _decompose(eta, sec, ker)
        by_class.setdefault(tuple(sec_c), []).append((tuple(flav_c), s))

    # ⟨S|S⟩: per gauge class, c·c with the flavour exponents differenced.
    overlap: dict[tuple, list] = {}
    for lst in by_class.values():
        for fa, ha in lst:
            for fb, hb in lst:
                mu_exp = tuple(-fa[i] + fb[i] for i in range(f))
                prod = ha * hb
                if prod.is_zero():
                    continue
                overlap.setdefault(mu_exp, []).append(prod)

    pf = qpoch_infty(K)
    for _ in range(g - 1):
        pf = pf * qpoch_infty(K)

    final: dict[int, object] = {}
    for mu_exp, lst in overlap.items():
        scaled = pf * (HabiroElement.sum(lst)).expand_to_power_series(K)
        basis = R.basis_element(mu_exp)
        for q_exp, q_coeff in scaled._c.items():
            if not q_coeff:
                continue
            term = q_coeff * basis
            final[q_exp] = term if q_exp not in final else final[q_exp] + term

    return RPowerSeries(R, {q: c for q, c in final.items() if not c.is_zero()}, K)


def _decompose(eta, sec_basis, ker_basis):
    from snf_kernel import decompose_in_basis
    return decompose_in_basis(list(eta), sec_basis, ker_basis)
