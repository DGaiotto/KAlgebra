"""`BPSAtlas` cone test over the **U(1)-gauged AD** family, via
**flow composition**.

These are gauge theories with **no native gauged-AD BPS quiver** — but each is
realized as an `RGKAlgebra` whose IR auxiliary *does* have a BPS chart, so
"**an RG flow to something which has a BPS chart fixes immediately a BPS chart
via flow composition**".  The RG realization is therefore a legitimate
BPS-backed `root`, and `bps_atlas.verify_cone_presentation(root, cone, iso)`
validates the cone's ray-multiplication and ρ against it.

  * **u1a1d4 / u1a1d6 / u1a1d8** — `U1A1Deven(k)` = u(1)-gauged `[A₁,D_{2k+2}]`,
      k = 1, 2, 3.  root = `U1A1DevenViaDoddRG(k)` (IR = `A1DoddConeKAlg(k−1) ⊗
      QT(Z²)`); cone = `U1A1DevenConeKAlgebra(k)`, the curve frame since
      2026-09-24 (labels `(curves, e, κ)`, derived from that flow; the design record);
      iso = `u1a1deven_cone_dodd_section_iso(k)` (the closed-form bijection of
      labels).  (Until 2026-09-24 the cone was the ray-keyed table presentation,
      now the archived tree; until 2026-09-23 k = 1
      used the legacy flow `U1A1DevenRGKAlgebra(1)` and the section map
      `u1a1deven_cone_legacy_iso`, also in the source repository's archive.)
  * **u1e7** — u(1)-gauged E₇.
      root = `U1A1E7RGKAlgebra` (IR = `A1A2k(3)` = the nonagon ⊗ QT(Z²));
      cone = `U1E7ConeKAlgebra`; iso = identity on labels.

**No flavour rebase here** (contrast the A1D BPS-chart catalogue): the RG root
carries the *full* non-abelian flavour (`SU2ZPlusRing` for u1a1d4), the same as
the cone, so the comparison is apples-to-apples in canonical R-form and **strict
ρ** holds (not just ρ-mod-flavour).  The rebase is only needed when the ground
truth is a `BPSKAlgebra` chart, which cannot see flavour promotion.

Run:  PYTHONPATH=. python scripts/atlas_catalogue_gauged_ad.py [--full]
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "implementations"))

from kalgebra import Element
from bps_atlas import verify_cone_presentation


def _u1a1deven_bundle(k):
    """U(1)-gauged [A₁,D_{2k+2}] = U1A1Deven(k): the flow `U1A1DevenViaDoddRG(k)`
    as root + the curve-frame cone derived from it, tied by the label-bijection
    iso (every k)."""
    from u1a1deven_cone_kalgebra import u1a1deven_cone_dodd_section_iso
    iso = u1a1deven_cone_dodd_section_iso(k)
    return (iso.target, iso.source, iso,
            f"U1A1DevenConeKAlgebra({k})", f"U1A1DevenViaDoddRG({k})")


def _u1e7_bundle():
    """u(1)-gauged E₇: RG root (IR = nonagon ⊗ QT(Z²)) + cone, identity iso."""
    from kalgebra_iso import KAlgebraIso
    from laurent_poly import LaurentPoly
    from u1e7_cone_kalgebra import U1E7ConeKAlgebra
    from u1a1e7_rgkalgebra import U1A1E7RGKAlgebra
    one = LaurentPoly.one()
    cone = U1E7ConeKAlgebra(use_frozen=True)
    root = U1A1E7RGKAlgebra()
    iso = KAlgebraIso(cone, root,
                      lambda l: Element({l: one}),
                      lambda l: Element({l: one}), name="u1e7[cone→rg]")
    return root, cone, iso, "U1E7ConeKAlgebra", "U1A1E7RGKAlgebra"


def bundle(name):
    if name == "u1a1d4":
        return _u1a1deven_bundle(1)
    if name == "u1a1d6":
        return _u1a1deven_bundle(2)
    if name == "u1a1d8":
        return _u1a1deven_bundle(3)
    if name == "u1e7":
        return _u1e7_bundle()
    raise KeyError(name)


def _gens(cone, mode):
    """The ray generators to cone-test.

      * ``"conedata"`` — the cone's *full* `mult_gens` (for U1A1Deven the
        curves of the once-punctured polygon and the X_{0,1} torus
        directions).  This is the complete generating set.  The U1A1Deven
        cone data is `χ`-stripped (native labels `(curves, e)`), so its
        letters are lifted to the class's labels `(curves, e, 0)`.
    """
    if mode != "conedata":
        raise ValueError(f"_gens: unknown mode {mode!r} (the 'seed' mode read "
                         f"the rays of the retired U1A1Deven tables)")
    from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra
    cd = cone.cone_data()
    labs = [cd.from_cone_label(frozenset({g}), {g: 1})
            for g in sorted(cd.mult_gens())]
    if isinstance(cone, U1A1DevenConeKAlgebra):
        labs = [(l[0], l[1], 0) for l in labs]
    return labs


# (name, ray_subset, gen_mode)  — u1a1d4 / u1e7 fast; the D-even chain heavier.
PLAN = [
    ("u1a1d4", None, "conedata"),   # full 14 mult_gens (12 curves + X_{0,1}^±)
    ("u1e7", 4, "conedata"),        # cone-vs-flow sample, ~0 s
]
# D₆ (k=2) and D₈ (k=3) certify on the FULL generator set via the
# label-bijection iso onto the flow the curve frame is derived from (D₆: 32
# generators / 1024 products; D₈: 58 generators / 3364 products).
FULL = [
    ("u1a1d6", None, "conedata"),   # D₆ = k=2 — FULL mult_gens
    ("u1a1d8", None, "conedata"),   # D₈ = k=3 — FULL mult_gens
]


def certify(name, nray, gen_mode="conedata"):
    t0 = time.time()
    rec = {"name": name, "gen_mode": gen_mode}
    root, cone, iso, clabel, rlabel = bundle(name)
    rec["cone"], rec["root"] = clabel, rlabel
    rec["root_ring"] = str(root.coefficient_ring())
    allg = _gens(cone, gen_mode)
    gens = allg if nray is None else allg[:nray]
    res = verify_cone_presentation(root, cone, iso, gens=gens)
    rec.update(rays=len(gens), total=len(allg), **res, t=time.time() - t0)
    return rec


def fmt(rec):
    sub = f"/{rec['total']}" if rec["n_rays"] < rec["total"] else ""
    return (f"  · {rec['name']:7s} root={rec['root']:24s} ring={rec['root_ring']:14s} "
            f"mode={rec['gen_mode']:8s} rays={rec['n_rays']}{sub} prods={rec['n_products']:4d} "
            f"ray_mult_ok={rec['ray_multiply_ok']!s:5s} rho_ok={rec['rho_ok']!s:5s} "
            f"rho_modflav={rec['rho_ok_mod_flavour']!s:5s} [{rec['t']:.1f}s]"
            + ("" if not rec["mismatches"] else f"  mism={rec['mismatches'][:2]}"))


def main():
    full = "--full" in sys.argv
    plan = PLAN + (FULL if full else [])
    print("=== BPSAtlas U(1)-gauged AD cone test (flow composition) ===", flush=True)
    recs = []
    for name, nray, gen_mode in plan:
        try:
            rec = certify(name, nray, gen_mode)
            recs.append(rec)
            print(fmt(rec), flush=True)
        except Exception as e:
            import traceback
            print(f"  · {name}: ERROR {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    ok = sum(1 for r in recs if r["ray_multiply_ok"])
    print(f"\n  ray-multiply OK: {ok}/{len(recs)} ; "
          f"ρ strict OK: {sum(1 for r in recs if r['rho_ok'])}/{len(recs)}", flush=True)
    return recs


if __name__ == "__main__":
    main()
