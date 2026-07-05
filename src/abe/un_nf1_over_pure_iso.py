"""KAlgebraIso: `UNNf1OverPure(2)` ↔ `BPSKAlgebra(U(2)+N_f=1)` — Plan 22 T7.

The Goal-1.3 certificate that the Route-A flow (pure U(2) → `add_flavour` →
generic `RGKAlgebra` with `S_RG = Ψ`, chart-evaluated pairing) and the BPS
realisation over `pure_ade.UN_Nf(2,1)` present the **same** abstract
K_𝖖-algebra.

Label correspondence (flow label `((m,e), k)` ↔ BPS charge
`γ = (m_0, e_0, m_1, e_1, f)`, `pure_un_iso` conventions + the flavour
node):

* the flavour direction is a torsor on both sides:
  `((m,e), k) ↦ γ((m,e)) + k·e_f`;
* Wilson lines (central `m`): `e_i = λ_{N−1−i}` (partition reversed) —
  `χ_(1,0) ↦ (0,0,0,1,0)`;
* principal monopoles positionally: `E ↦ (1,0,0,0,0)`, `F ↦ (0,0,−1,0,0)`,
  `det^{±1} ↦ ±(1,0,1,0,0)`, `EF ↦ (1,0,−1,0,0)`;
* dressings on **unexcited** Cartan slots map positionally
  (`((1,0),(0,1)) ↦ (1,0,0,1,0)`, `((0,−1),(−1,0)) ↦ (0,−1,−1,0,0)`);
* dressings on the **excited** slot are tropically frame-shifted —
  `((1,0),(1,0)) ↦ (1,−2,0,2,−1)` (fingerprint-unique in a ±2 box; the
  closed form of this shift is an open follow-up);
* several entries are σ-derived (`γ(ρ_flow(a)) = σ_BPS(γ(a))` — e.g.
  `((1,0),(−1,1)) = γ(ρ(F)) ↦ σ(γ_F)`), so ρ-equivariance is *enforced*
  for them and *emergent* for the fingerprint-derived ones (the
  Level-A discipline: only the latter count as evidence; the battery's
  multiplicativity + trace equivariance are emergent for all).

`build_iso()` returns `(iso, A_flow, B_bps)`; `certify(iso)` runs the
full `KAlgebraIso` battery on the dictionary.  Run as a script to certify:
`PYTHONPATH=. python implementations/un_nf1_over_pure_iso.py`.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly
from bps_kalgebra import BPSKAlgebra
import pure_ade as pa

from un_nf_over_pure_rgflow import UNNf1OverPure


# Flow pure-label ↦ BPS γ at flavour level 0 (γ = (m0, e0, m1, e1, f)).
_DICT_U2 = {
    ((0, 0), (0, 0)): (0, 0, 0, 0, 0),       # 1
    ((0, 0), (1, 0)): (0, 0, 0, 1, 0),       # χ_(1,0)
    ((0, 0), (0, -1)): (0, -1, 0, 0, 0),     # χ_(0,-1) = ρ(χ_(1,0))
    ((0, 0), (1, 1)): (0, 1, 0, 1, 0),       # χ_(1,1)
    ((0, 0), (-1, -1)): (0, -1, 0, -1, 0),   # χ_(-1,-1)
    ((0, 0), (2, 0)): (0, 0, 0, 2, 0),       # χ_(2,0)
    ((1, 0), (0, 0)): (1, 0, 0, 0, 0),       # E
    ((0, -1), (0, 0)): (0, 0, -1, 0, 0),     # F
    ((1, 1), (0, 0)): (1, 0, 1, 0, 0),       # det
    ((-1, -1), (0, 0)): (-1, 0, -1, 0, 0),   # det⁻¹
    ((1, -1), (0, 0)): (1, 0, -1, 0, 0),     # EF
    ((1, 0), (0, 1)): (1, 0, 0, 1, 0),       # E dressed (unexcited slot)
    ((0, -1), (-1, 0)): (0, -1, -1, 0, 0),   # F dressed (unexcited slot)
    ((0, -1), (0, 1)): (0, 0, -1, 1, 0),     # F dressed (excited slot, −m side)
    ((1, 0), (1, 0)): (1, -2, 0, 2, -1),     # E dressed (excited slot) — frame-shifted
    ((1, 0), (-1, 1)): (1, -1, 0, 1, 0),     # = ρ_flow(F)  [σ-derived]
    ((-1, -1), (-1, -1)): (-1, -1, -1, -1, 0),  # = ρ_flow(det)·μ² [σ-derived]
    ((0, -1), (-1, -1)): (0, -3, -1, 2, 1),  # = ρ_flow(Ed)·μ [σ-derived]
}
_E_F = (0, 0, 0, 0, 1)                       # the flavour direction in Γ


def build_iso(verify_bps: str = "off"):
    """Construct the iso (dictionary + flavour-torsor extension) and the two
    presentations.

    The base dictionary is closed under **one ρ-shell** at build time:
    for each base label `l`, the image of `ρ_flow(l)` is set to
    `σ_BPS(γ(l))` (the intertwining).  Where the entry already exists the
    closure *asserts* consistency instead — an emergent cross-check of the
    fingerprint-derived entries (e.g. `ρ(E) = F-dressed·μ⁻¹` must match
    `σ(γ_E)`)."""
    A = UNNf1OverPure(2)
    t = pa.UN_Nf(2, 1)
    B = BPSKAlgebra(pairing=t.B, node_charges=t.nodes, spec=t.spec,
                    cone_witness=getattr(t, "cone_witness", None)).shorten_spec()

    table = dict(_DICT_U2)
    for lab, g in list(table.items()):
        lr, kr = A.rho((lab, (0,)))
        gr = B.rho(g)
        g0r = tuple(x - int(kr[0]) * y for x, y in zip(gr, _E_F))
        if lr in table:
            if table[lr] != g0r:
                raise ValueError(
                    f"ρ-shell inconsistency at {lab}: ρ_flow → {lr} with "
                    f"γ {table[lr]}, but σ_BPS(γ) → {g0r}")
        else:
            table[lr] = g0r

    inv = {}
    for lab, g in table.items():
        if g[:4] in inv:
            raise ValueError(f"inverse collision on {g[:4]}")
        inv[g[:4]] = (lab, g[4])

    def forward(label):
        (m, e), k = label
        g0 = table.get((tuple(m), tuple(e)))
        if g0 is None:
            raise KeyError(f"flow label {label} not in the iso dictionary")
        g = tuple(x + int(k[0]) * y for x, y in zip(g0, _E_F))
        return Element({g: LaurentPoly.one()})

    def inverse(g):
        hit = inv.get(tuple(g[:4]))
        if hit is None:
            raise KeyError(f"BPS charge {g} not in the iso dictionary")
        lab, f0 = hit
        return Element({(lab, (int(g[4]) - f0,)): LaurentPoly.one()})

    iso = KAlgebraIso(A, B, forward, inverse,
                      name="U(2)+Nf=1: RG-flow-over-pure ≅ BPS(UN_Nf(2,1))")
    return iso, A, B


def certify(iso, A, B, trace_K: int = 4, mult_pairs=None, verbose=True):
    """Full `KAlgebraIso` battery over the dictionary (both directions).

    `mult_pairs` (flow-label pairs) must have in-dictionary products; the
    default set was chosen to close: torsor shifts, Wilson LR, det·det⁻¹,
    and a monopole–Wilson pair.
    """
    one = LaurentPoly.one()
    flow_labels = [(lab, (0,)) for lab in _DICT_U2]
    flow_labels += [(((1, 0), (0, 0)), (1,)), (((0, 0), (0, 0)), (-1,))]
    src_samples = [Element({l: one}) for l in flow_labels]
    tgt_samples = [iso.map(s) for s in src_samples]

    if mult_pairs is None:
        mult_pairs = [
            ((((0, 0), (0, 0)), (1,)), (((1, 0), (0, 0)), (0,))),   # μ · E
            ((((0, 0), (1, 0)), (0,)), (((0, 0), (1, 0)), (0,))),   # χ□ · χ□
            ((((1, 1), (0, 0)), (0,)), (((-1, -1), (0, 0)), (0,))),  # det · det⁻¹
            ((((1, 0), (0, 0)), (0,)), (((0, 0), (0, 0)), (-1,))),  # E · μ⁻¹
        ]
    src_pairs = [(Element({a: one}), Element({b: one})) for a, b in mult_pairs]
    tgt_pairs = [(iso.map(x), iso.map(y)) for x, y in src_pairs]

    out = iso.verify_all(src_samples, tgt_samples, src_pairs, tgt_pairs,
                         trace_K=trace_K)
    if verbose:
        for k, v in out.items():
            print(f"  {k}: {v}")
    return out


if __name__ == "__main__":
    iso, A, B = build_iso()
    print(f"== {iso.name} ==")
    res = certify(iso, A, B)
    ok = all(res.values())
    print("CERTIFIED" if ok else "FAILED")
    sys.exit(0 if ok else 1)
