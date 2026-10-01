"""`BPSAtlas` cone test over the **A1D-type** [A₁,Dₙ] theories,
via flavour rebase onto the abelian Cartan a `BPSKAlgebra` chart sees.

**The key:** a `BPSKAlgebra` does **not see flavour promotion** — its
chart sees only the *abelian* flavour `ker(B)`, never the non-abelian flavour
group the cone/RG realizations carry.  So to compare a flavoured cone against
the BPS chart, you must first **flavour-rebase the cone onto that abelian view**
(the canonical Cartan restriction), *then* establish the iso.

  * **D-odd** (a1d3/5/7) are **pure SU(2)** — rebased by `su2_to_u1_hom`
    (doublet ↦ z^{±1}), abelian Cartan `U(1) = ker B`.  `Su2ToU1Rebase`.
  * **D-even** (a1d4/6/8) are **SU(2)×U(1)** — rebased by
    `_su2u1_to_u1u1_hom` ((k,m) ↦ Σ_j z?^{k-2j}·z?^m), abelian Cartan
    `U(1)² = ker B`.  `CoeffRebase` + a **rank-2 section change**: the two
    `U(1)²` slots must be ordered to match the bps's `ker(B)` basis
    (a1d4: `u1su2`, U(1) first); `certify_a1d` auto-detects the GL(2,Z)
    ordering by keeping whichever ray-multiply passes.

`base_change(...)` is **trace-only** — the A1D cones are R-form-native (flavour
in the `RLaurent` cross-product *coefficients*), so the *multiply* coeffs must be
pushed too.  `Su2ToU1Rebase`/`CoeffRebase` do exactly that: push multiply AND
trace coeffs through the hom, keep the labels.  The direct `_bps_oracle(short)`
chart is already in the fundamental (`z`) normalization that `su2_to_u1_hom`
produces — so no μ=z² re-refinement is needed (that index-2 shift is only the
hexagon object's `a1d3` z-face convention).

Then `cone_to_bps_iso(short, rebased, bps)` is the cone↔BPS witness and
`bps_atlas.verify_cone_presentation` checks the cone's ray-multiplication and ρ;
`BPSAtlas(bps).complete()` completes the (finite-type) chart graph alongside.

Run:  PYTHONPATH=. python scripts/atlas_catalogue_a1d.py [--full]
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "implementations"))

from kalgebra import Element
from zplus_ring import su2_to_u1_hom, AbelianZPlusRing, RLaurent
import finite_kalgebras as fk
from elem_traces import _bps_oracle
from finite_kalgebra_objects import cone_to_bps_iso
from bps_atlas import verify_cone_presentation, BPSAtlas


class Su2ToU1Rebase:
    """Label-preserving SU(2)→U(1) rebase of an SU(2)-flavoured cone: push the
    native multiply AND trace `RLaurent` coeffs through `su2_to_u1_hom`
    (`base_change` is trace-only).  Brings the cone onto the BPS chart's abelian
    flavour view so cone↔BPS isos and the cone test become well-posed."""

    def __init__(self, native):
        self._n = native
        self._Z = AbelianZPlusRing(rank=1)
        self._phi = su2_to_u1_hom(native.coefficient_ring(), self._Z)

    def coefficient_ring(self):
        return self._Z

    def identity(self):
        return self._n.identity()

    def rho(self, a):
        return self._n.rho(a)

    def rho_inverse(self, a):
        return self._n.rho_inverse(a)

    def _label_section_decompose(self, label):
        return label, self._Z.one()

    def cone_data(self):
        return self._n.cone_data()

    def _push(self, c):
        return self._phi.apply_RLaurent(c) if isinstance(c, RLaurent) else c

    def multiply(self, a, b):
        return Element({l: self._push(c)
                        for l, c in self._n.multiply(a, b).terms.items()})

    def trace(self, a, K: int = 20):
        return self._phi.apply_RPowerSeries(self._n.trace(a, K))


class CoeffRebase:
    """Label-preserving coefficient rebase by an arbitrary ring hom `phi` — push
    the native multiply AND trace coeffs through `phi`, keep labels.  The
    SU(2)×U(1)→U(1)² generalization of `Su2ToU1Rebase` for the *su2u1* D-even
    cones (a1d4/6/8); `phi` Cartan-restricts the SU(2) factor and keeps the U(1)."""

    def __init__(self, native, phi):
        self._n = native
        self._phi = phi
        self._Z = phi.target

    def coefficient_ring(self):
        return self._Z

    def identity(self):
        return self._n.identity()

    def rho(self, a):
        return self._n.rho(a)

    def rho_inverse(self, a):
        return self._n.rho_inverse(a)

    def _label_section_decompose(self, label):
        return label, self._Z.one()

    def cone_data(self):
        return self._n.cone_data()

    def _push(self, c):
        return self._phi.apply_RLaurent(c) if isinstance(c, RLaurent) else c

    def multiply(self, a, b):
        return Element({l: self._push(c)
                        for l, c in self._n.multiply(a, b).terms.items()})

    def trace(self, a, K: int = 20):
        return self._phi.apply_RPowerSeries(self._n.trace(a, K))


def _su2u1_to_u1u1_hom(order):
    """SU(2)×U(1) → U(1)² Cartan restriction: basis `(k, m)` ↦ `Σ_j z?^{k-2j}·z?^m`
    (doublet ↦ z^{±1}).  `order` places the SU(2)-Cartan weight and the U(1)
    charge in the two `AbelianZPlusRing(2)` slots — picked to match the bps's
    `ker(B)` basis ordering (the section change).  `'u1su2'` = U(1) first (the
    a1d4 case: `ker B = {(1,0,1,0)=U(1), (0,0,0,1)=SU(2)}`)."""
    from zplus_ring import SU2xU1ZPlusRing, RElement, RingHom
    src, tgt = SU2xU1ZPlusRing(), AbelianZPlusRing(rank=2)

    def on_basis(b):
        k, m = b
        if order == "su2u1":
            terms = {(k - 2 * j, m): 1 for j in range(k + 1)}
        else:                              # 'u1su2'
            terms = {(m, k - 2 * j): 1 for j in range(k + 1)}
        return RElement(tgt, terms)
    return RingHom(src, tgt, on_basis)


def _a1d_bps(short):
    """The A1D BPS chart.  a1d3/5/7 (pure SU(2)) are in `_bps_oracle`; the su2u1
    a1d4/6/8 are built from their embedded quiver literals (`ker B` = U(1)²)."""
    if short in ("a1d3", "a1d5", "a1d7"):
        return _bps_oracle(short)
    from bps_kalgebra import BPSKAlgebra
    mod = __import__(f"finite_{short}_kalg")
    P = getattr(mod, f"{short.upper()}_BPS_PAIRING")
    N = getattr(mod, f"{short.upper()}_BPS_NODE_CHARGES")
    return BPSKAlgebra(pairing=[list(r) for r in P],
                       node_charges=[tuple(g) for g in N], verify="off")


# (short_id, rank, ray_subset, flavour) — D-odd = pure SU(2); D-even = SU(2)×U(1).
#
# The A1Dₙ feasibility frontier, MEASURED 2026-06-28 (not speculated):
#   a1d3 (rank 3)  full cone test           ~0.1 s   ✓ certified
#   a1d4 (rank 4)  su2u1, 4 rays            ~2 s     ✓ certified
#   a1d5 (rank 5)  su2,   3 rays            ~min     ✓ certified
#   a1d6 (rank 6)  su2u1, 4 rays / 16 prods  864 s   ✓ certified (strict ρ; just
#                                                     under a 900 s budget)
#   a1d7 (rank 7)  su2,   3 rays            >900 s   ✗ WALL (timed out)
#   a1d8 (rank 8)  su2u1                    beyond the wall
#
# The build is cheap at every rank (imports/cone/bps/iso ~0 s), but the cone
# test's BPS **ground truth** — the per-ray-pair F-solve `multiply` — is NOT
# rank-agnostic: it blows up at rank ≥7 (~54 s/product already at rank 6).  The
# D-type fork makes the high-rank quiver dense (like the exceptional E-types),
# unlike the cheap linear-A polygons that reached the dodecagon.  So the earlier
# "A1Dₙ is not harder / purely environmental" read was only half right: the
# container restarts were real, but a1d7+ is a genuine **compute** wall.  The
# specs are the correctly-oriented same-arrow charts (all of `vacuum_nahm`,
# embedded `_BPS_*` literals) — the spec is never the blocker; the F-solve is.
PLAN = [
    ("a1d3", 3, None, "su2"),     # rank 3 — full cone test, ~0.1 s
    ("a1d4", 4, 4, "su2u1"),      # rank 4 — SU(2)×U(1), ~2 s
    ("a1d5", 5, 3, "su2"),        # rank 5 — 3-ray sample, ~min
    ("a1d6", 6, 4, "su2u1"),      # rank 6 — 4 rays, ~864 s (the last feasible)
]
# Beyond the feasibility wall (BPS F-solve >900 s): a1d7 (rank 7), a1d8 (rank 8).


def certify_a1d(short, nray, flavour="su2"):
    t0 = time.time()
    rec = {"name": short}
    native = fk.FINITE_KALGEBRAS[short]()
    bps = _a1d_bps(short)
    rec["bps_ring"] = str(bps.coefficient_ring())
    comp = BPSAtlas(bps).complete()                     # complete the atlas
    rec["complete"] = {k: comp[k] for k in
                       ("n_charts", "n_classes", "classified", "closed")}
    cd = native.cone_data()
    allg = [cd.from_cone_label(frozenset({g}), {g: 1}) for g in sorted(cd.mult_gens())]
    if flavour == "su2":
        cone = Su2ToU1Rebase(native)                    # SU(2)→U(1)
        iso = cone_to_bps_iso(short, cone, bps)
        gens = allg if nray is None else allg[:nray]
        res = verify_cone_presentation(bps, cone, iso, gens=gens)
        rec["flavour_order"] = "su2→u1"
    else:                                               # su2u1: auto-detect the
        gens = allg if nray is None else allg[:nray]    # ker(B) basis ordering
        res = None
        for order in ("u1su2", "su2u1"):                # the GL(2,Z) section change
            cone = CoeffRebase(native, _su2u1_to_u1u1_hom(order))
            iso = cone_to_bps_iso(short, cone, bps)
            r = verify_cone_presentation(bps, cone, iso, gens=gens)
            if r["ray_multiply_ok"]:
                res, rec["flavour_order"] = r, order
                break
            res = res or r                              # keep first (failing) for report
        rec.setdefault("flavour_order", "none-matched")
    rec.update(rays=len(gens), total=len(allg), **res, t=time.time() - t0)
    return rec


def fmt(rec):
    sub = f"/{rec['total']}" if rec["rays"] < rec["total"] else ""
    cp = rec.get("complete")
    cstr = ("" if not cp else
            f"complete(n_charts={cp['n_charts']},closed={cp['closed']}) ")
    return (f"  · {rec['name']:5s} bps={rec['bps_ring']:22s} {cstr}"
            f"order={rec.get('flavour_order','?'):8s} rays={rec['rays']}{sub} "
            f"prods={rec['n_products']:4d} ray_mult_ok={rec['ray_multiply_ok']!s:5s} "
            f"rho_ok={rec['rho_ok']!s:5s} rho_modflav={rec['rho_ok_mod_flavour']!s:5s} "
            f"[{rec['t']:.1f}s]"
            + ("" if not rec["mismatches"] else f"  mism={rec['mismatches'][:2]}"))


def main():
    full = "--full" in sys.argv
    plan = PLAN if full else PLAN[:2]      # default: a1d3 (su2) + a1d4 (su2u1)
    print("=== BPSAtlas A1D-type cone test (SU(2)→U(1) / SU(2)×U(1)→U(1)² rebase) ===",
          flush=True)
    recs = []
    for short, rank, nray, flavour in plan:
        try:
            rec = certify_a1d(short, nray, flavour)
            recs.append(rec)
            print(fmt(rec), flush=True)
        except Exception as e:
            import traceback
            print(f"  · {short}: ERROR {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    ok = sum(1 for r in recs if r["ray_multiply_ok"])
    print(f"\n  ray-multiply OK: {ok}/{len(recs)} ; "
          f"ρ strict OK: {sum(1 for r in recs if r['rho_ok'])}/{len(recs)}", flush=True)
    return recs


if __name__ == "__main__":
    main()
