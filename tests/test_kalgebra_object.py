"""Tests for `KAlgebraObject` + the finite-zoo populations.

Covers: registry/capability routing; composed transport along witness
paths; the full per-witness `KAlgebraIso` battery on the pentagon
(7 realizations, including the stated-skein one — its witness battery
exercises the genuinely skein-side multiply on all generator pairs —
and the pinned-torus 'skein-pinned' sibling, the design record T-pin2)
and heptagon (3); coherence — both the positive case (a witness cycle
whose paths agree) and the negative control (a deliberately ρ-twisted
edge is detected as incoherent); and the flavoured A / D
entries' second component — the zoo's Z-form wrapper of the 'cone-frozen'
instance and the family class, joined by the seeds map as a `KAlgebraIso`
and not joined to {'cone-frozen', 'bps'} — on one entry per family.

Run:  `python3 run_tests.py`
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly

from finite_kalgebra_objects import kalgebra_object


PASS = []
FAIL = []
ONE = LaurentPoly.one()


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")


def _gen_labels(alg):
    """Identity + single-mult-gen native labels of a cone realization."""
    cd = alg.cone_data()
    return [alg.identity()] + [
        cd.from_cone_label(frozenset({g}), {g: 1})
        for g in cd.mult_gens()
    ]


def _samples_by_key(obj):
    out = {}
    for key in obj.keys():
        alg = obj.realization(key)
        cd = getattr(alg, "cone_data", lambda: None)()
        if cd is not None:
            out[key] = _gen_labels(alg)
        else:
            # BPS realization: identity charge + the transported gens
            cone = obj.realization("cone-frozen")
            out[key] = [alg.identity()] + [
                next(iter(obj.transport(l, "cone-frozen", key).terms))
                for l in _gen_labels(cone)[1:]
            ]
    return out


def test_pentagon_object():
    O = kalgebra_object("pentagon")
    check("pentagon: 8 realizations",
          set(O.keys()) == {"cone-frozen", "bps", "closed-form", "a1a2k",
                            "rg-sqed1", "skein", "skein-pinned",
                            "skein-cone"})
    check("pentagon: preferred('chart') is the BPS realization",
          type(O.preferred("chart")).__name__ == "BPSKAlgebra")
    check("pentagon: preferred('trace-closed-form') exists",
          O.preferred("trace-closed-form") is O.realization("closed-form"))
    check("pentagon: rg-sqed1 realization carries 'rg'",
          "rg" in O.capabilities("rg-sqed1"))
    check("pentagon: preferred('geometric') is the skein realization",
          O.preferred("geometric") is O.realization("skein"))

    samples = _samples_by_key(O)
    res = O.verify_pairwise(samples, pairs=True, trace_K=8)
    bad = {edge: {k: v for k, v in r.items() if not v}
           for edge, r in res.items() if not all(r.values())}
    check(f"pentagon: pairwise witness battery ({len(res)} edges)"
          + ("" if not bad else f" - bad: {bad}"), not bad)
    # the direct standalone(closed-form)<->bps witness is stored
    # explicitly (not only implicit through the cone-frozen hub)
    check("pentagon: direct closed-form<->bps edge stored",
          ("closed-form", "bps") in res)

    # composed transport round-trips through two hops
    lbl = ((0, 1), (1, 1)) if False else ((0, 1),)
    img = O.transport(lbl, "cone-frozen", "a1a2k")
    back = O.iso("a1a2k", "cone-frozen").map(img)
    check("pentagon: two-hop transport round-trip",
          back == Element({lbl: ONE}))

    # degree-2 label through the gamma-sum/ray-decomposition pair
    cone = O.realization("cone-frozen")
    deg2 = [l for l in _gen_labels(cone) if l != ()][:1]
    sq = ((0, 2),)
    img = O.transport(sq, "cone-frozen", "bps")
    back = O.iso("bps", "cone-frozen").map(img)
    check("pentagon: square label survives bps round-trip",
          back == Element({sq: ONE}))


def test_pentagon_coherence_positive_and_negative():
    O = kalgebra_object("pentagon")
    samples = _samples_by_key(O)

    # close a genuine cycle: the direct closed-form->a1a2k edge taken as
    # the composition of the two star edges; coherence must hold
    direct = O.iso("closed-form", "a1a2k")
    O.add_iso("closed-form", "a1a2k", direct)
    check("pentagon: coherence with closed cycle",
          O.verify_coherence(samples))

    # negative control: a rho-twisted direct edge is NOT coherent with
    # the star (it is a perfectly good iso, but a DIFFERENT one)
    O2 = kalgebra_object("pentagon")
    par = O2.realization("a1a2k")

    def rho_label_map(l):
        return Element({par.rho(l): ONE})

    def rho_inv_label_map(l):
        return Element({par.rho_inverse(l): ONE})

    rho_iso = KAlgebraIso(par, par, rho_label_map, rho_inv_label_map,
                          name="rho-twist")
    twisted = O2.iso("closed-form", "a1a2k").compose(rho_iso)
    O2.add_iso("closed-form", "a1a2k", twisted)
    check("pentagon: rho-twisted edge detected as incoherent",
          not O2.verify_coherence(_samples_by_key(O2)))


def test_heptagon_object():
    O = kalgebra_object("heptagon")
    check("heptagon: 8 realizations (closed-form + two RG flows + the "
          "stated-skein 2026-06-13 + the unified skein-cone 2026-07-10)",
          set(O.keys()) == {"cone-frozen", "bps", "a1a2k", "closed-form",
                            "rg-u1hexagon", "rg-pentagon", "skein",
                            "skein-cone"})
    check("heptagon: preferred('trace-closed-form') exists",
          O.preferred("trace-closed-form") is not None)
    check("heptagon: preferred('geometric') is the skein realization",
          O.preferred("geometric") is O.realization("skein"))
    # full pairwise battery; the RG flows' derived traces are exact
    # but slow, so their sample lists are trimmed to identity + 4 gens
    # (identity-on-labels witnesses: the bps Z^4 labels verbatim).  The
    # skein edge shares the a1a2k labels; its genuinely-skein-side
    # multiply is heavy, so it is trimmed to identity + 2 short + 2 long
    # here (the full 225-product certification is in
    # skein_sphere/the suite in the source repository).
    samples = _samples_by_key(O)
    samples["rg-u1hexagon"] = samples["bps"][:5]
    samples["rg-pentagon"] = samples["bps"][:5]
    samples["skein"] = [(), ((1, 0, 1),), ((1, 1, 1),),
                        ((2, 0, 1),), ((2, 1, 1),)]
    # The RG flows' derived TRACE is intractable in time since
    # retired the linear K_joint prune for the sound two-cutoff adaptive
    # shell (rg-u1hexagon Tr ~60 s/call on Z^4 → ~10 min for this edge
    # alone in the pairwise battery).  So the two rg edges are excluded
    # from the pairwise TRACE (empty edge_samples) and their multiply/ρ
    # are certified separately below; the heptagon's trace is certified
    # through the closed-form cone-frozen / closed-form / a1a2k edges
    # (the M(2,7) characters).  Per-instance rg-trace-equivariance is
    # machinery-certified (`tests/test_directional_subquiver_rg.py`).
    res = O.verify_pairwise(
        samples, pairs=True, trace_K=6,
        edge_samples={("rg-u1hexagon", "bps"): ([], []),
                      ("rg-pentagon", "bps"): ([], [])})
    bad = {edge: {k: v for k, v in r.items() if not v}
           for edge, r in res.items() if not all(r.values())}
    check(f"heptagon: pairwise witness battery ({len(res)} edges, "
          f"two-orbit dictionary)" + ("" if not bad else f" - {bad}"),
          not bad)
    # rg edges' multiply/ρ on real labels (identity-on-labels witnesses;
    # cheap — only the directional trace is intractable):
    rg_cheap = [Element({l: ONE}) for l in samples["bps"][:5]]
    rg_pairs = [(a, b) for a in rg_cheap for b in rg_cheap]
    for key in ("rg-u1hexagon", "rg-pentagon"):
        it = O.iso(key, "bps")
        check(f"heptagon: {key} multiply/ρ battery (no directional "
              f"trace;)",
              it.verify_unit()
              and it.verify_round_trip(rg_cheap, rg_cheap)
              and it.verify_multiplicative(rg_pairs, rg_pairs)
              and it.verify_rho_equivariant(rg_cheap, rg_cheap))
    check("heptagon: direct closed-form<->bps edge stored",
          ("closed-form", "bps") in res or ("bps", "closed-form") in res)
    check("heptagon: both RG-flow edges stored",
          any("rg-u1hexagon" in e for e in res)
          and any("rg-pentagon" in e for e in res))
    # bps -> a1a2k transport is the composite through cone-frozen
    g0 = samples["bps"][1]
    img = O.transport(g0, "bps", "a1a2k")
    check("heptagon: composed bps→a1a2k transport is a single canonical",
          len(img.terms) == 1 and str(next(iter(img.terms.values()))) == "1")
    check("heptagon: coherence (path independence) certificate",
          O.verify_coherence(samples))


def _zform_samples(O, family_key):
    """Identity, every generator and the flavour characters on both sides of
    the z-form witness."""
    fam = O.realization(family_key)
    n = len(O.realization("cone-frozen").cone_data().mult_gens())
    if family_key == "ungauged-u1a1aodd":
        z = [((), ((), (0,)))] + [((), (((g, 1),), (0,))) for g in range(n)] \
            + [((), ((), (m,))) for m in (1, -1)]
        f = [fam.identity()] + list(fam.mult_generators()) + [((), 1), ((), -1)]
    elif family_key == "a1dodd":
        z = [(0, ())] + [(0, ((g, 1),)) for g in range(n)] + [(1, ())]
        f = ([fam.identity()]
             + [(((g, 1),), 0) for g in sorted(fam.cone_data().mult_gens())]
             + [((), 1)])
    else:
        z = [(0, ((), 0))] + [(0, (((g, 1),), 0)) for g in range(n)] \
            + [(1, ((), 0)), (0, ((), 1)), (0, ((), -1))]
        f = [fam.identity()] + list(fam.mult_generators()) \
            + [((), 0, 1), ((), -1, 0), ((), 1, 0)]
    return z, f


def test_flavoured_zform_components():
    """the design record: 'z-form' and the family class, with the seeds map as the
    witness, in a component of their own."""
    for sid, fam in (("a3", "ungauged-u1a1aodd"),
                     ("hexagon", "ungauged-u1a1aodd"),
                     ("a1d3", "a1dodd"), ("a1d4", "a1deven")):
        O = kalgebra_object(sid)
        check(f"{sid}: realizations cone-frozen, bps, z-form, {fam}",
              O.keys() == ["cone-frozen", "bps", "z-form", fam])
        zf = O.realization("z-form")
        native = getattr(zf, "_native", None)
        if native is None:                    # the u1 entries' base change
            native = zf._source_alg._native
        check(f"{sid}: z-form wraps the registered cone-frozen instance",
              native is O.realization("cone-frozen"))
        try:
            O.iso("cone-frozen", "z-form")
            joined = True
        except KeyError:
            joined = False
        check(f"{sid}: no witness joins the two components "
              f"(iso cone-frozen → z-form raises)", not joined)
        check(f"{sid}: preferred('trace-exact') is still cone-frozen, "
              f"preferred('geometric') the family class",
              O.preferred("trace-exact") is O.realization("cone-frozen")
              and O.preferred("geometric") is O.realization(fam))
        z, f = _zform_samples(O, fam)
        res = O.verify_pairwise(
            {"z-form": z, fam: f}, pairs=True, trace_K=12,
            edge_samples={("cone-frozen", "bps"): ([], [])})
        r = res[("z-form", fam)]
        check(f"{sid}: witness z-form → {fam} passes the battery "
              f"({len(z) - 1}² pairs each way, traces through q^12): "
              f"{sorted(c for c, v in r.items() if not v) or 'all'}",
              all(r.values()) and len(r) == 5)


def test_discover_delegates_to_searcher():
    """discover(src, dst, searcher) finds-and-stores a witness via an
    external searcher (D4) — here match_generators on a fresh pentagon
    pair added without a witness."""
    from kalgebra_object import KAlgebraObject
    from finite_kalgebra_objects import match_generators
    import finite_kalgebras as fk
    from kalgebra_samples import PentagonKAlg
    O = KAlgebraObject("pentagon-discover")
    O.add_realization("cone", fk.FinitePentagonKAlgebra(),
                      {"multiply-fast"})
    O.add_realization("closed", PentagonKAlg(), {"trace-closed-form"})
    try:
        O.iso("cone", "closed")
        check("no witness before discover", False)
    except KeyError:
        check("no witness before discover", True)
    O.discover("cone", "closed", match_generators)
    img = O.transport(((0, 1),), "cone", "closed")
    check("discover stored a working witness",
          len(img.terms) == 1)


def test_disconnected_raises():
    from kalgebra_object import KAlgebraObject
    from kalgebra_samples import PentagonKAlg, TrivialKAlg
    O = KAlgebraObject("broken")
    O.add_realization("a", PentagonKAlg())
    O.add_realization("b", TrivialKAlg())
    try:
        O.iso("a", "b")
        check("disconnected groupoid raises", False)
    except KeyError:
        check("disconnected groupoid raises", True)


if __name__ == "__main__":
    test_pentagon_object()
    test_pentagon_coherence_positive_and_negative()
    test_heptagon_object()
    test_flavoured_zform_components()
    test_discover_delegates_to_searcher()
    test_disconnected_raises()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED, {len(PASS)} passed")
        sys.exit(1)
    print(f"All {len(PASS)} KAlgebraObject tests passed.")
