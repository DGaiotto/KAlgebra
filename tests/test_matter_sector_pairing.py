"""The Schur pairing as a sum over magnetic sectors of contour integrals on the
MATTER substrate — `MatterWRQTorus.inner_by_sector` with the per-sector measure
`matter_sector_weight`.

Five legs, positive control first:

1. control — the identity's self-pairing agrees between the sector route and
   the chart-side `Tr(ρ(a)·b)` at U(2)+1 and SU(2)+1;
2. agreement — on a basket of labels (Wilson, 't Hooft, dressed, a non-trivial
   flavour label) the two routes agree exactly per μ-level, at U(2)+1, SU(2)+1
   and U(2)+2 (two matter slots, non-zero cocycle level shifts);
3. axiom 5, ρ-equivariance of the trace — on the sector route
   `I_{b,a} = ⋆(I_{a,b})` with `⋆` the flavour duality `μ^{k⃗} ↦ μ^{−k⃗}`, and the
   contract verifier `verify_pairing_rho_star_symmetric` on the packaged
   `GNAbeKAlgebra.inner_product`;
4. routing — `GNAbeKAlgebra.inner_product` and the base `AbeKAlgebra.inner_product`
   on a matter chart actually call `inner_by_sector` (call counter), so the
   sector form is what runs, not dead code;
5. the Nahm window — a dyonic pair whose residual starts at `𝖖⁻⁴` is exact at
   its top order (the audit).

Run:  `python3 run_tests.py`
"""
from __future__ import annotations

import sys
import time
sys.path.insert(0, ".")

from g_matter_roster import roster
from matter_wrq_torus import MatterWRQTorus, matter_sector_weight

K = 5
_PASS = []


def check(msg, ok):
    assert ok, msg
    _PASS.append(msg)
    print(f"  PASS: {msg}")


def _same(d1, d2):
    keys = set(d1) | set(d2)
    return all(str(d1.get(k, 0)) == str(d2.get(k, 0)) for k in keys)


def _star(d):
    return {tuple(-x for x in k): v for k, v in d.items()}


def _basket(A, gauge):
    idn = A.identity()
    w0 = idn[1]
    labs = [(A.fold(m, e)[0], w0) for m, e in gauge]
    wf = tuple([1] + [0] * (len(w0) - 1))
    labs.append((labs[1][0], wf))            # a non-trivial flavour label
    return labs


def test_control_identity():
    for name in ("u2-nf1", "su2-nf1"):
        A = roster(name)
        c0 = A.chart(A.identity())
        cs = (c0.rho() * c0).trace(K, W=K + 2)
        se = c0.inner_by_sector(c0, K=K, W=K + 2)
        check(f"{name}: identity self-pairing, sector == chart-side ({cs})", _same(cs, se) and cs)


def _agreement(name, gauge):
    A = roster(name)
    labs = _basket(A, gauge)
    charts = {l: A.chart(l) for l in labs}
    n = ok = sym = 0
    for a in labs:
        for b in labs:
            cs = (charts[a].rho() * charts[b]).trace(K, W=K + 2)
            se = charts[a].inner_by_sector(charts[b], K=K, W=K + 2)
            n += 1
            ok += _same(cs, se)
            sym += _same(se, _star(charts[b].inner_by_sector(charts[a], K=K, W=K + 2)))
    check(f"{name}: sector == chart-side on {n} ordered pairs", ok == n)
    check(f"{name}: I(b,a) == ⋆I(a,b) on the sector route, {n} pairs", sym == n)


def test_agreement_u2_nf1():
    _agreement("u2-nf1", [((0, 0), (0, 0)), ((0, 0), (1, 0)), ((0, -1), (0, 0)), ((-1, -1), (1, 0))])


def test_agreement_su2_nf1():
    _agreement("su2-nf1", [((0,), (0,)), ((0,), (1,)), ((1,), (0,)), ((1,), (1,))])


def test_agreement_u2_nf2_two_slots():
    _agreement("u2-nf2", [((0, 0), (0, 0)), ((0, 0), (1, 0)), ((0, -1), (0, 0))])


def test_nahm_window_covers_deep_residuals():
    """Regression (2026-09-23, the audit).  The dyonic pair
    `L_{(1),(−2)}`, `L_{(1),(2)}` at SU(2)+1 has a pairing residual whose
    𝖖-expansion starts at `𝖖⁻⁴`, so the callers' Nahm window `W = K + 2` was
    short: `I(a,b)` at `K = 4` carried a spurious `(μ⁶ + μ⁻⁶)𝖖⁴` and axiom 5,
    `I_{b,a} = ⋆I_{a,b}`, failed there — invisible to `_agreement` above, whose
    two routes share the residual and the window, and whose labels start at
    `𝖖^{≥ −2}`.  `_mu_trace_rows` now raises the window to `K + pad`.  The value
    through `𝖖⁴` is `𝖖² − (μ² + μ⁻²)𝖖⁴`: the printed sum over `m` of the
    paper's eq:Iexplicit, which has no Nahm window, gives `|W| = 2` times it
    (the design record's `checks_coulomb`)."""
    A = roster("su2-nf1")
    w0 = A.identity()[1]
    a = (A.fold((1,), (-2,))[0], w0)
    b = (A.fold((1,), (2,))[0], w0)
    ca, cb = A.chart(a), A.chart(b)

    def through(rows, k):
        out = {}
        for mu, lp in rows.items():
            c = {e: v for e, v in lp._coeffs.items() if e <= k and v}
            if c:
                out[mu] = c
        return out
    at4 = through(ca.inner_by_sector(cb, K=4, W=6), 4)
    at6 = through(ca.inner_by_sector(cb, K=6, W=8), 4)
    check("su2-nf1 dyonic pair: the K = 4 pairing equals the K = 6 one through q^4", at4 == at6)
    check("su2-nf1 dyonic pair: I(a,b) = q^2 - (mu^2 + mu^-2) q^4 + O(q^5)",
          at4 == {(0,): {2: 1}, (2,): {4: -1}, (-2,): {4: -1}})
    check("su2-nf1 dyonic pair: verify_pairing_rho_star_symmetric at K = 4",
          A.verify_pairing_rho_star_symmetric(a, b, K=4))


def test_sector_weight_reduces_to_pure_at_no_matter_shift():
    """At `m = 0` the matter cocycle carries no level shift and the weight is
    the pure `sector_weight(datum, 0)` at `r⃗ = 0`."""
    from wrq_torus import sector_weight
    A = roster("u2-nf1")
    w, D = matter_sector_weight(A.datum, A.chart(A.identity()).slots, (0, 0))
    check("m=0: single level shift r=0 and D=0", list(w) == [(0,)] and D == (0,))
    check("m=0: matter weight == pure sector_weight",
          (w[(0,)] - sector_weight(A.datum, (0, 0))).simplify().is_zero())


def test_contract_verifier_and_routing():
    counts = {"n": 0}
    orig = MatterWRQTorus.inner_by_sector

    def _counted(self, *a, **k):
        counts["n"] += 1
        return orig(self, *a, **k)

    MatterWRQTorus.inner_by_sector = _counted
    try:
        A = roster("u2-nf1")
        idn = A.identity()
        E = (A.fold((1, 0), (0, 0))[0], idn[1])
        W1 = (A.fold((0, 0), (1, 0))[0], idn[1])
        ok = A.verify_pairing_rho_star_symmetric(E, W1, K=4)
        check("GNAbeKAlgebra u2-nf1: verify_pairing_rho_star_symmetric(E, W1)", ok)
        check("GNAbeKAlgebra.inner_product routes through inner_by_sector", counts["n"] >= 2)
    finally:
        MatterWRQTorus.inner_by_sector = orig


if __name__ == "__main__":
    t = time.time()
    for fn in (test_control_identity, test_sector_weight_reduces_to_pure_at_no_matter_shift,
               test_agreement_u2_nf1, test_agreement_su2_nf1, test_agreement_u2_nf2_two_slots,
               test_nahm_window_covers_deep_residuals, test_contract_verifier_and_routing):
        fn()
    print(f"\nAll matter sector-pairing tests passed ({len(_PASS)} checks).   [{time.time() - t:.1f}s]")
