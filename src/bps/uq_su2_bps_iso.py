"""uq_su2_bps_iso — the `KAlgebraIso` between

    UqSU2KAlgebra                (the `U_𝖖(sl_2)` `K_𝖖`-algebra, implemented
                                  directly from the draft's definition;
                                  `src/samples/uq_su2_kalgebra.py`)
and
    gf_bps_sqed_nf(2)            (the SU(2)-flavoured BPS chart of SQED_2:
                                  `GfBPSKAlgebra` over `SUNZPlusRing(2)` on
                                  the draft's SQED_2 quiver — the doublet's
                                  two weight nodes at `(m, e) = (0, 1)`, the
                                  magnetic direction the dashed, frozen node;
                                  `src/bps/gf_bps_sqed_nf.py`)

— connections to other algebraic frameworks / the design record: the quantum group and the gauge theory are ONE
`K_𝖖`-algebra in two presentations, certified by the full `KAlgebraIso`
battery in both directions (`tests/test_uq_su2_bps_iso.py`).

The chart's labels are `(m, e, k)` — magnetic charge first, electric
charge, and the highest weight `k` of the SU(2) character `χ_k` carried (the
dominant Weyl-orbit representative); its pairing is the draft's
`⟨(1,0,0),(0,1,0)⟩ = 1`.  The generator dictionary is therefore the draft's
own (§"The central quotient of `U_𝖖(sl_2)` and SQED_2"), read off the chart's
products on 2026-09-21:

    K  =  v   =  D_{0,1}    ↔  L_{(0, 1, 0)}       (the Wilson line)
    E  =  u₊  =  D_{1,0}    ↔  L_{(1, 0, 0)}       (the undressed monopole)
    F  =  u₋  =  D_{−1,−1}  ↔  L_{(−1, −1, 0)}     (the dressed anti-monopole)
    χ_k                     ↔  L_{(0, 0, k)}

so that `EF = χ₁ + 𝖖K + 𝖖⁻¹K⁻¹` is the draft's `u₊u₋ = 𝖖⁻¹v⁻¹ + μ + μ⁻¹ + 𝖖v`,
`KE = 𝖖⁻²EK`, and `ρ(E) = 𝖖⁻¹FK⁻¹ = F_{1,−1}` is the draft's
`ρ(u₊) = 𝖖v⁻¹u₋` (lower charge `(−1, −2)`).  Linearly extended, since every
canonical basis element on both sides is a `𝖖`-normalised monomial in the
generators:

    χ_k K^n     ↔  (0, n, k)
    χ_k E_{a,b} ↔  (a, b, k)               E_{a,b} = 𝖖^{−ab} E^a K^b  (= D_{a,b})
    χ_k F_{a,b} ↔  (−a, b − a, k)          F_{a,b} = 𝖖^{ab} F^a K^b

The inverse reads the magnetic charge: `m = 0` is the Cartan sector, `m > 0`
the `E` sector (`a = m`, `b = e`), `m < 0` the `F` sector (`a = −m`,
`b = e − m`).

Another consistent dictionary exists — `E ↔ (1, 1, 0)`, `F ↔ (−1, 0, 0)` —
differing from this one by the canonical-basis automorphism
`E_{a,b} ↦ E_{a,b+a}`, `F_{a,b} ↦ F_{a,b−a}` (`E ↦ 𝖖⁻¹EK`, `F ↦ 𝖖⁻¹K⁻¹F`),
which commutes with `ρ`; the draft's labelling is the one used.

Run:  `PYTHONPATH=$(ls -d src/* | paste -sd:) python3 src/bps/uq_su2_bps_iso.py`
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
from zplus_ring import SUNZPlusRing

from gf_bps_sqed_nf import gf_bps_sqed_nf
from uq_su2_kalgebra import UqSU2KAlgebra

_ONE = LaurentPoly.one()


def uq_label_to_chart(A: UqSU2KAlgebra, label) -> tuple:
    """`(sector, χ_k)` ↦ the chart label `(m, e, k)`."""
    (sector, w) = label
    k = A.spin2(w)
    kind = sector[0]
    if kind == 'K':
        return (0, sector[1], k)
    _, a, b = sector
    if kind == 'E':
        return (a, b, k)
    return (-a, b - a, k)


def chart_label_to_uq(A: UqSU2KAlgebra, label) -> tuple:
    """The chart label `(m, e, k)` ↦ `(sector, χ_k)`."""
    m, e, k = (int(x) for x in label)
    if k < 0:
        raise ValueError(f"chart label {label!r}: the weight slot must be dominant")
    w = A.chi(k)
    if m == 0:
        return (('K', e), w)
    if m > 0:
        return (('E', m, e), w)
    return (('F', -m, e - m), w)


def uq_su2_bps_iso(*, chart=None) -> KAlgebraIso:
    """The certified `UqSU2KAlgebra ≅ gf_bps_sqed_nf(2)` iso (both over
    `SUNZPlusRing(2)`).  `chart` may be passed to reuse a built chart."""
    A = UqSU2KAlgebra(ring=SUNZPlusRing(2))
    G = chart if chart is not None else gf_bps_sqed_nf(2)

    def forward(label):
        return Element({uq_label_to_chart(A, label): _ONE})

    def inverse(label):
        return Element({chart_label_to_uq(A, label): _ONE})

    return KAlgebraIso(
        A, G, forward_label_map=forward, inverse_label_map=inverse,
        name="UqSU2KAlgebra ≅ gf_bps_sqed_nf(2)  (U_𝖖(sl_2) ≅ SQED_2)",
    )


def iso_samples(A: UqSU2KAlgebra, *, max_a: int = 2, max_b: int = 1,
                max_k: int = 1) -> list:
    """The label basket the battery runs on."""
    return list(A.basis_iter(max_a=max_a, max_b=max_b, max_k=max_k))


if __name__ == "__main__":
    import time
    iso = uq_su2_bps_iso()
    A, G = iso.source, iso.target
    print(iso)
    print("  K ↔", uq_label_to_chart(A, A.K(1)), "  E ↔", uq_label_to_chart(A, A.E()),
          "  F ↔", uq_label_to_chart(A, A.F()), "  χ₁ ↔", uq_label_to_chart(A, A.K(0, 1)))
    labs = iso_samples(A)
    src = [Element({l: _ONE}) for l in labs]
    tgt = [iso.map(s) for s in src]
    gens = [A.K(1), A.K(-1), A.K(0, 1), A.E(), A.F(), A.E(1, 1), A.F(1, -1), A.E(2, 0), A.F(2, 0)]
    pairs = [(Element({a: _ONE}), Element({b: _ONE})) for a in gens for b in gens]
    tpairs = [(iso.map(a), iso.map(b)) for a, b in pairs]
    t0 = time.time()
    res = iso.verify_all(src, tgt, pairs, tpairs, trace_K=8)
    print(f"  verify_all on {len(labs)} labels / {len(pairs)} pairs ({time.time()-t0:.1f}s):", res)
    secs = [l for l in labs if l[1] == A.coefficient_ring().one_basis()]
    print("  sections ↦ 1-dim-rep dressed canonicals:",
          iso.verify_maps_section_to_section_1drep(secs, [uq_label_to_chart(A, l) for l in secs]))
