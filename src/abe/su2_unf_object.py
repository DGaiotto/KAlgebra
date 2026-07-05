"""`KAlgebraObject` for **SU(2) + N_f** (user direction, 2026-06-13 — the
sibling of `su2_nf1_object`, starting at N_f=2).

The abstract algebra `A_𝖖[SU(2)+N_f]` presented at **two manifest-flavour
levels**, bridged by the `SO(2N_f)` enhancement built in
`su2_unf_abe_kalgebra.so2nf_index`:

* **Spin(2N_f)-manifest** (N_f=2): the canonical-surface pair
  `cone` (`SU2Nf2KAlgebra`, standalone Spin(4) = SU(2)_L×SU(2)_R) ↔ `bps`
  (`build_bps_su2_nf2`).  Both carry tropical `Z⁴` labels, so the witness
  is the identity-on-labels `su2_nf2_h_iso`; unit / round-trip / multiply /
  ρ certify **raw**, and the trace certifies in the **irrep↔Cartan folded
  frame** (cone carries flavour as Spin(4) irreps `SU(2)_L×SU(2)_R`, bps as
  Cartan weights `AbelianZPlusRing(2)`; the same index in two ring
  conventions — `tests/test_su2_unf_object.py::test_cone_bps_trace_folded`).
* **U(N_f)-manifest** (all N_f): `abe` (`SU2UNf{N_f}AbeKAlgebra`), the
  **ungauged** realization on the rational quantum torus (ungauge the
  diagonal U(1) of `UNNfKAlgebra(2,N_f)`, #478).  Its manifest flavour is
  `U(N_f) ⊂ SO(2N_f)`; the full `SO(2N_f)` is recovered on the index by
  `abe.so2nf_index` (the recognize-after wrapper, #480).

For **N_f=3** the Spin(6)-manifest `bps` is `build_bps_su2_nf3` (manifest
`SU(4) ≅ SO(6)` flavour, rank-3 Dynkin Cartan); the `abe — bps` bridge is
the convention-free q-graded **dimension** match `certify_abe_bps_dim`
(abe's `SO(6)` index — `q⁰` trivial, `q²` adjoint `(1,1,0)` = 15 currents,
… — has the same dimensions as bps's `SU(4)` index).  For **N_f=4** there
is no BPS generator, so the object is `abe`-only.

**The abe ↔ bps bridge is the `SO(2N_f)` index, not a stock iso.**  abe and
bps have *different coefficient rings* (`U(N_f)` vs the `Spin(2N_f)` Cartan)
and *different canonical bases* (gauge 't Hooft–Wilson lines over the torus
vs matter flavour-doublets in a mutated chart), so a stock
`verify_trace_equivariant` cannot compare them directly.  The physically
correct equivalence — that both present the **same** `A_𝖖[SU(2)+N_f]` — is
certified at the `Spin(2N_f)` level: `abe`'s `SO(2N_f)`-lifted index
(`so2nf_index`, reconstructed to the torus) equals `bps`'s index exactly
(`certify_abe_bps_so2nf`; the N_f=2 case is pinned through q⁴ in
`tests/test_su2_unf_object.py`, and is the same fact as
`test_su2_unf_abe_kalgebra.test_nf2_so4_reconstructs_bps_index`).  This is
the cross-flavour analogue of how `su2_nf1_object` certifies its
flavour-carrying edges in the folded frame rather than raw.

So the groupoid has the certified `cone ↔ bps` edge; `abe` sits as the
`U(N_f)`-manifest / `f-presentation` realization carrying the same algebra,
bridged to the Spin(2N_f) core by the `so2nf` capability + the index
certificate (a full element-level `abe ↔ bps` label iso — the gauge-frame
map composed with the per-element `SO(2N_f)` flavour lift — is the recorded
deeper item).

Realizations (N_f=2):

* ``'cone'`` — `SU2Nf2KAlgebra`: standalone Spin(4)-manifest KAlgebra
  (tropical `Z⁴`, Spin(4) characters in `R`).
* ``'bps'``  — `build_bps_su2_nf2()`: the canonical `BPSKAlgebra` on the
  Spin(4)-manifest mutated chart.
* ``'abe'``  — `SU2UNf2AbeKAlgebra`: the ungauged U(2)-manifest abelianized
  realization (rational quantum torus; `SO(4)` via `so2nf_index`).

Witnesses:

* ``cone ↔ bps`` — `su2_nf2_h_iso` (identity on `Z⁴`; unit/round-trip/
  multiply/ρ raw, trace in the irrep↔Cartan folded frame).
* ``abe — bps`` — the `SO(2N_f)` index certificate `certify_abe_bps_so2nf`
  (not a stored groupoid edge; the cross-flavour bridge).
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra_object import KAlgebraObject


__all__ = ["su2_unf_object", "certify_abe_bps_so2nf", "certify_abe_bps_dim"]


_ABE_CLS = {
    2: "SU2UNf2AbeKAlgebra",
    3: "SU2UNf3AbeKAlgebra",
    4: "SU2UNf4AbeKAlgebra",
}


def _abe_for(Nf: int):
    import su2_unf_abe_kalgebra as M
    return getattr(M, _ABE_CLS[Nf])()


def certify_abe_bps_so2nf(abe, bps, K: int = 4) -> bool:
    """Certify the cross-flavour `abe — bps` bridge at the `Spin(4)` level
    (N_f=2): the `SO(4)`-lifted `abe` vacuum index reconstructs (via Weyl
    characters, `(μ_L,μ_R) = (x₁+x₂, x₁−x₂)`) to the `bps` Spin(4) index,
    order by order through `q^K`.

    This is the physically-correct statement that `abe` and `bps` present
    the same `A_𝖖[SU(2)+N_f=2]`: they agree on the full Spin(4) index once
    `abe`'s manifest `U(2)` flavour is enhanced to `SO(4)` by
    `so2nf_index`."""
    from so2nf_characters import reconstruct

    idx = abe.so2nf_index(abe.identity(), K)
    bps_idx = {e: {k: v for k, v in c.terms.items() if v}
               for e, c in bps.trace((0, 0, 0, 0), K=K).coeffs.items()
               if not c.is_zero()}
    for e in range(K + 1):
        mine = {}
        for x, c in reconstruct(2, idx.get(e, {})).items():
            key = (x[0] + x[1], x[0] - x[1])      # (x1,x2) -> (μ_L, μ_R)
            mine[key] = mine.get(key, 0) + c
        mine = {k: v for k, v in mine.items() if v}
        if mine != bps_idx.get(e, {}):
            return False
    return True


def certify_abe_bps_dim(abe, bps, K: int = 4) -> bool:
    """Convention-free `abe — bps` bridge for any N_f with a BPS generator:
    the **unflavoured q-graded dimension** of `abe`'s `SO(2N_f)` index
    (`Σ_λ dim(λ)·c_λ`) equals `bps`'s index dimension (all fugacities → 1),
    order by order through `q^K`.

    This certifies that `abe` and `bps` present the same `A_𝖖[SU(2)+N_f]`
    *without* committing to a flavour-Cartan convention.  The full flavoured
    character bridge (e.g. `SO(6) ≅ SU(4)` at N_f=3 — the orthogonal-`e_i`
    ↔ SU(4) Dynkin Cartan iso) is the recorded deeper item; N_f=2's full
    character bridge is `certify_abe_bps_so2nf`."""
    from so2nf_characters import dim

    Nf = abe.Nf
    idx = abe.so2nf_index(abe.identity(), K)
    abe_dim = {e: sum(dim(Nf, lam) * m for lam, m in c.items())
               for e, c in idx.items()}
    zero = bps.identity()
    tr = bps.trace(zero, K=K)
    bps_dim = {e: sum(rc.terms.values())
               for e, rc in tr.coeffs.items() if not rc.is_zero()}
    return all(abe_dim.get(e, 0) == bps_dim.get(e, 0) for e in range(K + 1))


def su2_unf_object(Nf: int = 2, with_abe: bool = True) -> KAlgebraObject:
    """The abstract `A_𝖖[SU(2)+N_f]` as a `KAlgebraObject`.

    N_f=2 gives the full triad: the certified Spin(4) pair (`cone ↔ bps`)
    plus the ungauged `abe` leg (U(2)-manifest; `SO(4)` via `so2nf_index`),
    bridged by `certify_abe_bps_so2nf`.  For N_f=3 the Spin(6) `bps`
    (`build_bps_su2_nf3`) joins `abe`; for N_f=4 only `abe` (no BPS
    spectrum generator).  `with_abe=False` builds only the Spin(2N_f) core.
    """
    obj = KAlgebraObject(f"A_q[SU(2)+Nf={Nf}]")

    if Nf == 2:
        from su2_nf2_h_iso import su2_nf2_h_iso
        iso_cb = su2_nf2_h_iso()
        cone = iso_cb.source
        bps = iso_cb.target
        obj.add_realization("cone", cone,
                            {"closed-form", "trace-exact", "spin4-manifest"})
        obj.add_realization("bps", bps,
                            {"chart", "rg", "trace-exact", "spin4-manifest"})
        obj.add_iso("cone", "bps", iso_cb)
    elif Nf == 3:
        from bps_su2_nf3 import build_bps_su2_nf3
        obj.add_realization("bps", build_bps_su2_nf3(),
                            {"chart", "rg", "trace-exact", "spin6-manifest"})

    if with_abe:
        abe = _abe_for(Nf)
        obj.add_realization(
            "abe", abe,
            {"chart", "f-presentation", "trace-exact", "ungauged",
             f"so{2 * Nf}-enhanced"})

    return obj


if __name__ == "__main__":
    obj = su2_unf_object(2)
    print(obj)
    print("  keys:", obj.keys())
    for k in obj.keys():
        print(f"    {k}: {sorted(obj.capabilities(k))}")
    bps = obj.realization("bps")
    abe = obj.realization("abe")
    print("  abe—bps SO(4) index bridge:", certify_abe_bps_so2nf(abe, bps, 4))
    # cone ↔ bps transport (identity on Z⁴)
    print("  cone→bps transport (0,0,0,0):",
          dict(obj.transport((0, 0, 0, 0), "cone", "bps").terms))
