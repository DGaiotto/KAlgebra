"""`KAlgebraObject` populations for the finite-type zoo.

`kalgebra_object(short_id)` returns the abstract finite-type algebra
as a `KAlgebraObject` holding:

* ``'cone-frozen'`` — the frozen standalone (`FINITE_KALGEBRAS[sid]`);
* ``'bps'``         — `BPSKAlgebra` on the embedded quiver literals;
* for the pentagon: ``'closed-form'`` (`PentagonKAlg`), ``'a1a2k'``
  (`A1A2kKAlg(1)`), ``'rg-sqed1'`` (`PentagonSquareKAlg` — the
  pentagon as an RG flow over Sqed1 = SQED1, the U(1)-gauged square),
  and ``'skein'`` (`SkeinPentagonKAlg` — the stated-skein realization
  via the UNPIN construction); for the heptagon: ``'closed-form'``
  (`kalgebra_samples.HeptagonKAlg` — products served by `A1A2kKAlg(2)`
  under the orbit-2 relabelling `(2, i) ↦ (2, i + 4)`, M(2,7)-seeded
  two-layer trace) and ``'a1a2k'`` (`A1A2kKAlg(2)`),
  plus the direct closed-form↔bps edge — the pentagon-parallel
  structure;
* for the flavoured `A` and `D` entries, a second
  component: ``'z-form'`` — the zoo's Z-form wrapper of the ``'cone-frozen'``
  instance itself, flavour in the labels (`FiniteU1ZKAlgebra` for
  a3 / a5 / a7 and their aliases, base-changed along `μ ↦ z⁻¹`;
  `FiniteSU2ZKAlgebra` for a1d3 / a1d5 / a1d7; `FiniteSU2U1ZKAlgebra` for
  a1d4 / a1d6 / a1d8) — and the family class the zoo's seeds map lands on:
  ``'ungauged-u1a1aodd'`` (`ungauge_u1a1aodd(k)`), ``'a1dodd'``
  (`A1DoddConeKAlg(k)`), ``'a1deven'`` (`A1DevenKAlg(k)`; at a1d4 the
  method's positive control, `SU3ADKAlg` serving that entry's traces).  The
  family instance is the one the seeds module serves traces from.

Witnesses:

* cone-frozen ↔ bps — canonical: a zoo label IS the canonical basis
  element at `γ = Σ p·γ_i` (`_ray_idx_to_gamma`), so forward is the
  γ-sum and the inverse is an exact non-negative ray decomposition
  over the frozen cones (Fraction Gaussian elimination per cone).
* cone-frozen ↔ {closed-form, a1a2k} — a generator dictionary found by
  `match_generators`: ρ-orbit-respecting assignments of mult-gens are
  enumerated (orbit pairing × cyclic shift) and the first candidate
  passing the `KAlgebraIso` battery on generators + pairs wins.
  Canonical basis maps to canonical basis label-for-label (coefficient
  1) through each side's `cone_data` ↔ label bijection; the battery is
  the certificate.
* (pentagon) closed-form ↔ bps — a direct edge stored explicitly (the
  composite through the cone-frozen hub); coherent by construction.
* (pentagon) closed-form ↔ rg-sqed1 — the identity on labels
  (`PentagonSquareKAlg` IS `PentagonKAlg` with RG-mediated
  multiply/trace), so the battery is the certified theorem that the
  RG-derived multiply and Schur-transport trace reproduce the closed
  forms exactly.
* (pentagon) skein ↔ bps — the identity on labels
  (`SkeinPentagonKAlg` lives on the chart labels); its battery is the
  certified theorem that the genuinely skein-side multiply (localized
  engine products of the unpinned dressed chords) reproduces the BPS
  chart structure constants.
* (flavoured `A` / `D`) z-form ↔ family — the seeds module's certified
  generator map as a `KAlgebraIso` (`aodd_seeds.kalgebra_iso`,
  `a1dodd_seeds.kalgebra_iso`, `a1deven_seeds.kalgebra_iso`): a zoo word
  goes to the family label of its letters' images with the zoo's flavour
  charge moved into the label's flavour slot, and back through the
  generators' images.  The battery on every generator, the flavour
  characters, every ordered pair of them (at a1d8 a documented sample of the
  pairs by default, all of them with `--slow`), ρ, the traces at `K = 12` and
  the section check `verify_maps_section_to_section_1drep` is run by
  tests/test_aodd_seeds.py, tests/test_a1dodd_seeds.py and
  tests/test_a1deven_seeds.py (the numbers:
  the design record
  section N).

The two components are not joined by a witness: 'z-form' is 'cone-frozen'
with the flavour moved from the coefficients into the labels (the regrouping
`to_R_form`, certified in tests/test_finite_{u1,su2,su2u1,a1d3}_zform.py),
which a `KAlgebraIso` cannot carry — it extends a label map
coefficient-linearly and checks ρ on bare labels (measured on a3 at the
'trace-exact' comment in `kalgebra_object`).  The link is structural: the
wrapper holds the registered 'cone-frozen' instance as its native
(`realization('z-form')._native`, or `._source_alg._native` through the u1
entries' base change), and `iso` between the components raises `KeyError`.
"""
from __future__ import annotations

from fractions import Fraction

from kalgebra import Element, KAlgebra
from kalgebra_iso import KAlgebraIso
from kalgebra_object import KAlgebraObject
from laurent_poly import LaurentPoly

import finite_kalgebras as fk
from elem_traces import _bps_oracle
from regen import REGEN_SPECS, _load_standalone


__all__ = [
    "kalgebra_object",
    "cone_to_bps_iso",
    "cone_dictionary_iso",
    "match_generators",
]

_ONE = LaurentPoly.one()


# ---------------------------------------------------------------------------
# cone-frozen ↔ bps (canonical: γ-sums)
# ---------------------------------------------------------------------------

def _solve_cone(rays: list[tuple[int, ...]], gamma: tuple[int, ...]):
    """Solve `γ = Σ p_i · rays[i]` with `p_i ∈ ℤ_{≥0}` exactly, for a
    simplicial cone (len(rays) == rank).  Returns the powers list or
    None."""
    n = len(rays)
    rank = len(gamma)
    if n != rank:
        return None
    A = [[Fraction(rays[j][i]) for j in range(n)] for i in range(rank)]
    b = [Fraction(g) for g in gamma]
    # Gaussian elimination with partial pivoting (exact)
    for col in range(n):
        piv = next((r for r in range(col, rank) if A[r][col] != 0), None)
        if piv is None:
            return None
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        inv = 1 / A[col][col]
        A[col] = [x * inv for x in A[col]]
        b[col] = b[col] * inv
        for r in range(rank):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [x - f * y for x, y in zip(A[r], A[col])]
                b[r] = b[r] - f * b[col]
    p = b[:n]
    if any(x.denominator != 1 or x < 0 for x in p):
        return None
    return [int(x) for x in p]


def cone_to_bps_iso(short_id: str, cone_alg: KAlgebra,
                    bps_alg: KAlgebra) -> KAlgebraIso:
    """The canonical witness `cone-frozen → bps` for a frozen entry."""
    mod, prefix = _load_standalone(short_id)
    gens = getattr(mod, f"{prefix}_MULT_GENS_LATTICE")
    cones = getattr(mod, f"{prefix}_CONES")
    rank = len(gens[0])
    zero = (0,) * rank

    def forward(label) -> Element:
        gamma = tuple(
            sum(gens[i][k] * p for i, p in label) for k in range(rank))
        return Element({gamma: _ONE})

    def inverse(gamma) -> Element:
        if tuple(gamma) == zero:
            return Element({(): _ONE})
        for cone in cones:
            idx = sorted(cone)
            p = _solve_cone([gens[i] for i in idx], tuple(gamma))
            if p is not None:
                label = tuple(
                    (i, q) for i, q in zip(idx, p) if q > 0)
                return Element({label: _ONE})
        raise ValueError(
            f"{short_id}: BPS label {gamma} is not a non-negative ray "
            f"combination of any frozen cone")

    return KAlgebraIso(cone_alg, bps_alg, forward, inverse,
                       name=f"{short_id}[cone-frozen→bps]")


# ---------------------------------------------------------------------------
# cone ↔ cone (generator dictionaries)
# ---------------------------------------------------------------------------

def cone_dictionary_iso(src_alg: KAlgebra, tgt_alg: KAlgebra,
                        gen_map: dict, name: str | None = None
                        ) -> KAlgebraIso:
    """Witness between two `ConeKAlgebra` presentations from a mult-gen
    dictionary: each side's `cone_data` label ↔ (gens, powers)
    bijection translates canonical labels gen-by-gen (coefficient 1)."""
    src_cd = src_alg.cone_data()
    tgt_cd = tgt_alg.cone_data()
    inv_map = {v: k for k, v in gen_map.items()}
    if len(inv_map) != len(gen_map):
        raise ValueError("cone_dictionary_iso: gen_map not injective")

    def _translate(cd_from, cd_to, mapping, label) -> Element:
        gens, powers = cd_from.to_cone_label(label)
        mapped = {mapping[g]: p for g, p in powers.items()}
        out = cd_to.from_cone_label(frozenset(mapped), mapped)
        return Element({out: _ONE})

    return KAlgebraIso(
        src_alg, tgt_alg,
        lambda l: _translate(src_cd, tgt_cd, gen_map, l),
        lambda l: _translate(tgt_cd, src_cd, inv_map, l),
        name=name,
    )


def _label_orbits(alg: KAlgebra, native_gens: list) -> list[list]:
    """ρ-orbits on single-generator native labels, each in ρ-cycle
    order."""
    orbits, seen = [], set()
    for g in native_gens:
        if g in seen:
            continue
        orbit = [g]
        seen.add(g)
        nxt = alg.rho(g)
        while nxt != g and nxt not in seen:
            orbit.append(nxt)
            seen.add(nxt)
            nxt = alg.rho(nxt)
        orbits.append(orbit)
    return orbits


def match_generators(src_alg: KAlgebra, tgt_alg: KAlgebra,
                     *, name: str | None = None,
                     trace_K: int = 6) -> KAlgebraIso:
    """Find the mult-gen dictionary between two cone presentations of
    the same abstract algebra by enumerating ρ-orbit-respecting
    assignments (orbit pairing × cyclic shift) and certifying each
    candidate with the `KAlgebraIso` battery on generators + all
    generator pairs (unit, round-trip, multiplicativity,
    ρ-equivariance, trace-equivariance).  Returns the first certified
    witness; raises if none passes."""
    from itertools import permutations, product

    src_cd, tgt_cd = src_alg.cone_data(), tgt_alg.cone_data()
    src_native = [src_cd.from_cone_label(frozenset({g}), {g: 1})
                  for g in src_cd.mult_gens()]
    tgt_native = [tgt_cd.from_cone_label(frozenset({g}), {g: 1})
                  for g in tgt_cd.mult_gens()]
    if len(src_native) != len(tgt_native):
        raise ValueError("match_generators: generator counts differ")

    src_orbits = _label_orbits(src_alg, src_native)
    tgt_orbits = _label_orbits(tgt_alg, tgt_native)
    by_len: dict[int, list] = {}
    for o in tgt_orbits:
        by_len.setdefault(len(o), []).append(o)
    if sorted(len(o) for o in src_orbits) != sorted(
            len(o) for o in tgt_orbits):
        raise ValueError("match_generators: ρ-orbit profiles differ")

    # native single-gen label -> cone-data mult-gen key (for gen_map)
    def native_to_mg(cd, native_labels):
        out = {}
        for g in cd.mult_gens():
            out[cd.from_cone_label(frozenset({g}), {g: 1})] = g
        return out

    src_n2mg = native_to_mg(src_cd, src_native)
    tgt_n2mg = native_to_mg(tgt_cd, tgt_native)

    src_e = [Element({l: _ONE}) for l in
             [src_alg.identity()] + src_native]
    tgt_e = [Element({l: _ONE}) for l in
             [tgt_alg.identity()] + tgt_native]
    gen_src_e = src_e[1:]
    gen_tgt_e = tgt_e[1:]

    same_len_groups = [
        (o, by_len[len(o)]) for o in src_orbits
    ]

    def assignments():
        # which tgt orbit each src orbit maps to (within equal length),
        # then a cyclic shift per orbit
        group_perms = []
        for length in sorted({len(o) for o in src_orbits}):
            srcs = [o for o in src_orbits if len(o) == length]
            tgts = by_len[length]
            group_perms.append((srcs, list(permutations(tgts,
                                                        len(srcs)))))
        for combo in product(*[perms for _s, perms in group_perms]):
            shift_spaces = []
            pairing = []
            for (srcs, _), chosen in zip(group_perms, combo):
                for s_orb, t_orb in zip(srcs, chosen):
                    pairing.append((s_orb, t_orb))
                    shift_spaces.append(range(len(s_orb)))
            for shifts in product(*shift_spaces):
                gm = {}
                for (s_orb, t_orb), sh in zip(pairing, shifts):
                    L = len(s_orb)
                    for k, s_lbl in enumerate(s_orb):
                        gm[src_n2mg[s_lbl]] = tgt_n2mg[
                            t_orb[(k + sh) % L]]
                yield gm

    for gm in assignments():
        iso = cone_dictionary_iso(src_alg, tgt_alg, gm, name=name)
        try:
            res = iso.verify_all(
                src_e, tgt_e,
                [(a, b) for a in gen_src_e for b in gen_src_e],
                [(a, b) for a in gen_tgt_e for b in gen_tgt_e],
                trace_K=trace_K,
            )
        except Exception:
            continue
        if all(res.values()):
            return iso
    raise ValueError(
        f"match_generators: no ρ-orbit-respecting dictionary certified "
        f"between {type(src_alg).__name__} and {type(tgt_alg).__name__}")


# ---------------------------------------------------------------------------
# the populations
# ---------------------------------------------------------------------------

def _zform_component(short_id: str, cone: KAlgebra):
    """`(family key, iso)` for a flavoured A / D entry — the seeds module's
    `kalgebra_iso` wrapping the registered 'cone-frozen' instance `cone` —
    or `None` for the other entries (module docstring).  Building the iso
    computes nothing; its label maps build the generator map at first use."""
    import a1deven_seeds
    import a1dodd_seeds
    import aodd_seeds
    if short_id in aodd_seeds.K_OF:
        return ("ungauged-u1a1aodd",
                aodd_seeds.kalgebra_iso(short_id, native=cone))
    if short_id in a1dodd_seeds.K_OF:
        return "a1dodd", a1dodd_seeds.kalgebra_iso(short_id, native=cone)
    if short_id in a1deven_seeds.K_OF:
        return "a1deven", a1deven_seeds.kalgebra_iso(short_id, native=cone)
    return None


def kalgebra_object(short_id: str) -> KAlgebraObject:
    """The abstract finite-type algebra for `short_id`, populated with
    its certified realizations (see module docstring)."""
    if short_id not in REGEN_SPECS:
        raise KeyError(f"kalgebra_object: unknown short_id {short_id!r}")
    obj = KAlgebraObject(f"A_q[{short_id}]")

    cone = fk.FINITE_KALGEBRAS[short_id]()
    cone_caps = {"multiply-fast"}
    # `trace-exact`: the realization's trace is exact where it is served, in
    # practice.  Every zoo entry qualifies (closed-form characters, the
    # Nahm-sum vacuum plus the orthonormality bootstrap; e6/e8 from their W3
    # character recipes, the design record; no frozen table and no BPS fallback).
    # Until 2026-09-24 the su2u1 entries a1d6 / a1d8 were the
    # exception: they served no trace at all and raised NotImplementedError
    # naming the class that computes it (their route peeled the flavour slots
    # below was their exact route and `preferred("trace-exact")` picked it.
    # e7 was excluded as well until 2026-09-23, because its u(1) seed
    # bootstrap did not reach q^4 in 20 minutes.  The corrected bootstrap
    # serves every e7 seed through the same dispatch as
    # a5/a7, under the same recorded μ-support hypothesis (measured through
    # the zoo: Tr(1) and all 90 seeds at K = 8 in about 70 s, equal to
    # `u1_bootstrap.generate_u1`).
    # Since the same day the zoo serves e7 from its closed forms
    # (`e7_seeds`: Tr(1) and every seed to any
    # order), with the bootstrap as the witness.
    # Until 2026-09-23 the tag tracked whether a frozen table existed
    #, which tagged a5 while it served wrong values and left
    # its alias octagon untagged.
    # Since 2026-09-23 a3 / a5 / a7 serve their traces from the ungauged
    # polygon `ungauge_u1a1aodd(k)` and a1d3 from `A1DoddConeKAlg(0)`, through
    # generator maps built at runtime (`aodd_seeds`,
    # `a1d3_seeds`).  The ungauged polygon is not joined to
    # this cone class by a witness (since 2026-09-26 it is registered in the
    # Z-form component below, the design record): a stock `KAlgebraIso` from this
    # cone class to
    # `ungauge_u1a1aodd(k).base_change(μ ↦ μ⁻¹)` fails its own battery for
    # encoding reasons, not mathematical ones.  Measured on a3: unit, round
    # trip and trace equivariance pass, but forward multiplicativity holds on
    # 22 of the 36 generator pairs — exactly the 14 whose product carries μ in
    # a coefficient fail, because `KAlgebraIso` extends a label map
    # coefficient-linearly and cannot turn μ^f into the E^f shift the ungauged
    # labels carry — and ρ-equivariance fails on exactly the three generators
    # with a nonzero `_rho_delta`, because the battery's ρ is the bare label
    # 'bps' edge below fails the same two checks on a3 the same way (22/36,
    # coefficient and label, and whose ρ check uses `rho_element`, would
    # certify the pair; `aodd_seeds.to_ungauged` is that map.  The Z-form
    # component below certifies it the other way round: the zoo's Z-form
    # wrapper moves μ into the labels first, so a stock `KAlgebraIso`
    # (`aodd_seeds.kalgebra_iso`) passes its whole battery.
    # a1d4 joined them the same day, served from `SU3ADKAlg` restricted to
    # SU(2)×U(1) (`a1d4_seeds`).
    # a1d6 / a1d8 joined on 2026-09-24, served from `A1DevenKAlg(2)` /
    # `A1DevenKAlg(3)` (`a1deven_seeds`), so every entry now
    # carries the tag.  Their traces are exact arithmetic within the depth
    # `A1DevenKAlg`'s trace transport reaches (every generator through q^12 at
    # k = 2 and q^8 at k = 3), and past it they raise, naming the limit.  The
    # one statement they rest on that is not a theorem is the transport's
    # stopping rule, a measured hypothesis with a guard
    # (`u1a1deven_trace_transport`) — the standard e7 / a5 / a7 were tagged
    # under, with their μ-support hypothesis.
    cone_caps.add("trace-exact")
    obj.add_realization("cone-frozen", cone, cone_caps)

    bps = _bps_oracle(short_id)
    obj.add_realization("bps", bps, {"chart", "trace-exact", "rg"})
    obj.add_iso("cone-frozen", "bps", cone_to_bps_iso(short_id, cone, bps))

    # The Z-form component: the flavoured A / D
    # entries' seeds maps as `KAlgebraIso`s, from the zoo's Z-form wrapper of
    # the 'cone-frozen' instance itself onto the family class.  No witness
    # joins this component to {'cone-frozen', 'bps'}: the wrapper is the
    # standalone with its flavour moved from the coefficients into the labels
    # (the regrouping `to_R_form`, certified in tests/test_finite_*_zform.py),
    # which a `KAlgebraIso` cannot carry — it extends a label map
    # coefficient-linearly, so μ^f in a coefficient cannot become a label's
    # charge, and its ρ check reads the bare label `rho`, not `rho_element`
    # (measured on a3 in the comment at the 'trace-exact' tag above: the
    # products carrying μ in a coefficient and ρ on the generators with a
    # nonzero `_rho_delta` fail; the 'bps' edge above fails the same way).
    # The link is structural and documented instead: the wrapper holds the
    # 'cone-frozen' instance as its native (module docstring).  Inserted after
    # the hub's realizations, so `preferred` keeps its answers for the
    # existing tags.
    zform = _zform_component(short_id, cone)
    if zform is not None:
        family_key, iso = zform
        obj.add_realization("z-form", iso.source,
                            {"multiply-fast", "trace-exact"})
        obj.add_realization(family_key, iso.target,
                            {"trace-exact", "geometric"})
        obj.add_iso("z-form", family_key, iso)

    if short_id == "pentagon":
        from kalgebra_samples import PentagonKAlg
        from a1a2k_kalg import A1A2kKAlg
        cf = PentagonKAlg()
        obj.add_realization(
            "closed-form", cf, {"trace-exact", "trace-closed-form"})
        obj.add_iso("cone-frozen", "closed-form",
                    match_generators(cone, cf,
                                     name="pentagon[cone→closed-form]"))
        # Since 2026-09-24 'cone-frozen' serves its traces from this class
        # (`aeven_seeds`, a certified generator map; the
        # dictionary `match_generators` finds here is the served map itself,
        # ρ⁰ — measured 2026-09-26, the suite in the source repository).
        par = A1A2kKAlg(1)
        obj.add_realization(
            "a1a2k", par,
            {"multiply-fast", "trace-exact", "trace-closed-form"})
        obj.add_iso("cone-frozen", "a1a2k",
                    match_generators(cone, par,
                                     name="pentagon[cone→a1a2k]"))
        # Direct closed-form ↔ bps edge between the two presentations the
        # user blessed as definitive: the composite through the
        # cone-frozen hub, stored explicitly so the standalone↔BPS
        # witness carries its own battery line (rather than living only
        # implicitly in the groupoid).  Coherent by construction — it
        # *is* the hub path — so the coherence certificate still holds.
        obj.add_iso("closed-form", "bps",
                    obj.iso("closed-form", "bps"))
        # RG realization: the pentagon as an RG flow over Sqed1
        # (= SQED1, the U(1)-gauged square).  `PentagonSquareKAlg` is
        # `PentagonKAlg` with multiply/trace replaced by RG-mediated
        # versions through `Sqed1KAlg` (5 hard-coded RG generators +
        # the E_q(u_-) tower for S_RG).  It shares the closed-form's
        # label structure verbatim, so the witness is the identity on
        # labels — its battery is exactly the certified theorem that the
        # RG-derived multiply and Schur-transport trace reproduce the
        # closed forms (the suite in the source repository).
        from pentagon_square_rgkalg import PentagonSquareKAlg
        rg_sqed1 = PentagonSquareKAlg()
        obj.add_realization("rg-sqed1", rg_sqed1, {"rg", "trace-exact"})
        obj.add_iso(
            "closed-form", "rg-sqed1",
            KAlgebraIso(cf, rg_sqed1,
                        forward_label_map=lambda l: Element({l: _ONE}),
                        inverse_label_map=lambda l: Element({l: _ONE}),
                        name="pentagon[closed-form→rg-sqed1]"))
        # the stated-SKEIN realization (Le's stated algebra of the
        # bordered pentagon), registrable since the UNPIN construction
        # (2026-06-12): the bare stated algebra is the PINNED pentagon
        # (the A_2 quiver over Gamma~ = Gamma_A2 (+) Z^5 of boundary
        # endpoint charges, NOT iso to the realizations above), but in
        # the localization at the side arcs it factorizes as
        # Pentagon (x) T(side torus), and the dressed-chord subalgebra
        # IS the pentagon (verify_unpin_pentagon.py).
        # `SkeinPentagonKAlg` packages that subalgebra on chart labels
        # with genuinely skein-side multiply; rho/trace are transported
        # from the BPS realization (the same instance as 'bps', so the
        # witness endpoints match).
        from skein_pentagon_kalg import SkeinPentagonKAlg
        from skein_bps_iso import skein_bps_iso
        sk = SkeinPentagonKAlg(bps=bps)
        obj.add_realization("skein", sk, {"geometric"})
        # the skein->bps iso through the generator (single source; the
        # generator returns sk.build_iso() for n=5, target == this `bps`)
        obj.add_iso("skein", "bps", skein_bps_iso(5, K=sk))
        # the PINNED-torus skein realization (the design record T-pin2): the same
        # dressed-chord subalgebra computed in the pinned quantum torus
        # over Gamma~ (`PinnedPolygon(5)`), where the side arcs are
        # monomial UNITS — the localization is free, T~_p has boundary
        # block 0, and multiply is plain torus arithmetic (~17x faster
        # than 'skein', identical structure constants, asserted against
        # the chart on every product).
        from pinned_pentagon_kalg import PinnedPentagonKAlg
        skp = PinnedPentagonKAlg(bps=bps)
        obj.add_realization("skein-pinned", skp,
                            {"geometric", "multiply-fast"})
        obj.add_iso("skein-pinned", "bps", skp.build_iso())
        # the unified SkeinKAlgebra instance: derived cone multiply
        # over the pinned engine's cross-products, on the same chart
        # labels — anchored on the registered `bps` instance so the
        # witness endpoints match.
        from skein_kalgebra import PentagonSkeinKAlgebra
        skc = PentagonSkeinKAlgebra(bps=bps)
        obj.add_realization("skein-cone", skc, {"geometric", "cone"})
        obj.add_iso("skein-cone", "bps", skc.build_iso(bps))
    elif short_id == "heptagon":
        from kalgebra_samples import HeptagonKAlg
        from a1a2k_kalg import A1A2kKAlg
        # the closed-form standalone on its own labels (pentagon-parallel
        # tier): products served by A1A2kKAlg(2) under the orbit-2
        # relabelling (2, i) -> (2, i + 4), two-layer trace with the
        # three M(2,7) seeds supplied by A1A2kKAlg(2)'s Andrews–Gordon
        # character series
        cf = HeptagonKAlg()
        obj.add_realization(
            "closed-form", cf, {"trace-exact", "trace-closed-form"})
        obj.add_iso("cone-frozen", "closed-form",
                    match_generators(cone, cf,
                                     name="heptagon[cone→closed-form]"))
        # Since 2026-09-24 'cone-frozen' serves its traces from this class
        # (`aeven_seeds`, a certified generator map; the
        # dictionary `match_generators` finds here is the served map itself,
        # ρ⁰ — measured 2026-09-26, the suite in the source repository).
        par = A1A2kKAlg(2)
        obj.add_realization(
            "a1a2k", par,
            {"multiply-fast", "trace-exact", "trace-closed-form"})
        obj.add_iso("cone-frozen", "a1a2k",
                    match_generators(cone, par,
                                     name="heptagon[cone→a1a2k]"))
        # Direct closed-form ↔ bps edge (the pentagon pattern): the
        # composite through the cone-frozen hub, stored explicitly so
        # the standalone↔BPS witness carries its own battery line.
        # Coherent by construction — it IS the hub path.
        obj.add_iso("closed-form", "bps",
                    obj.iso("closed-form", "bps"))
        # RG realizations: the heptagon as a
        # directional node-deletion flow — complete RGKAlgebras (the
        # whole multiply/trace API derived through the flow), generic
        # machinery, no hand tables.  Both live on the heptagon's own
        # Z^4 BPS-charge labels (the same chamber as 'bps': identity
        # witnesses, batteries = the certified statement that the
        # RG-derived multiply/trace reproduce the BPS ones).
        from directional_subquiver_rg import (
            DirectionalSingleNodeRG, DirectionalSubquiverRG)
        _A4_P = [[0, 1, 0, 0], [-1, 0, 1, 0],
                 [0, -1, 0, 1], [0, 0, -1, 0]]
        _A4_N = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
        # Heptagon → U(1)Hexagon: drop one end node of the A_4 chain.
        # The IR auxiliary is the A_3 chamber on the surviving nodes
        # over the full Z^4 lattice with the dropped direction kept —
        # the u(1)-gauged hexagon (= `u1hexagon_object`'s bps chart,
        # up to the A_4 chain flip; the conventioned composition to
        # `U1HexagonKAlg` literally is `HeptagonHexagonRGKAlg`, whose
        # historical inner class still lacks `trace` and so cannot
        # carry the battery yet).
        rg_hex = DirectionalSingleNodeRG(
            _A4_P, _A4_N, list(_A4_N), gamma_drop=0, rg_window=6)
        obj.add_realization("rg-u1hexagon", rg_hex,
                            {"rg", "trace-exact"})
        obj.add_iso(
            "rg-u1hexagon", "bps",
            KAlgebraIso(rg_hex, bps,
                        forward_label_map=lambda l: Element({l: _ONE}),
                        inverse_label_map=lambda l: Element({l: _ONE}),
                        name="heptagon[rg-u1hexagon→bps]"))
        # Heptagon → Pentagon: drop two adjacent nodes; the IR
        # auxiliary is the A_2 chamber on the surviving pair over Z^4
        # with two decoupled directions — Pentagon ⊗ QT_2 (the flow
        # certified in tests/test_a1a2k_honest_chamber.py,
        # `test_two_node_drop_standard_spec`).
        rg_pent = DirectionalSubquiverRG(
            _A4_P, _A4_N, list(_A4_N), drop=[0, 1], rg_window=6)
        obj.add_realization("rg-pentagon", rg_pent,
                            {"rg", "trace-exact"})
        obj.add_iso(
            "rg-pentagon", "bps",
            KAlgebraIso(rg_pent, bps,
                        forward_label_map=lambda l: Element({l: _ONE}),
                        inverse_label_map=lambda l: Element({l: _ONE}),
                        name="heptagon[rg-pentagon→bps]"))
        # the stated-SKEIN realization (Le's stated algebra of the
        # bordered heptagon = the [A_1,A_4] irregular puncture, RANK 4,
        # two chord families).  Odd-marked -> no flavour -> the CLEAN
        # pentagon story: localizing the pinned 7-gon at the side arcs
        # factorizes it as Heptagon (x) T(sides), and the dressed-chord
        # subalgebra IS the A_4 heptagon (verify_unpin_heptagon.py).
        # `SkeinHeptagonKAlg` packages it on the A1A2kKAlg(2) labels with
        # genuinely skein-side multiply (D2 shorts <-> (1,i), D3 longs
        # <-> (2,i); the centered dressing keeps the longs small and the
        # exchange T(D2_0)T(D2_1) = unit + T(D3_0) clean); rho/trace
        # transported from a1a2k (the registered instance, so the
        # witness endpoints match).
        from skein_heptagon_kalg import SkeinHeptagonKAlg
        from skein_bps_iso import skein_bps_iso
        sk = SkeinHeptagonKAlg(intrinsic=par)
        obj.add_realization("skein", sk, {"geometric"})
        # the skein->a1a2k iso through the generator (single source)
        obj.add_iso("skein", "a1a2k", skein_bps_iso(7, K=sk))
        # the unified SkeinKAlgebra instance: derived cone multiply over the
        # localized engine's cross-products, on the a1a2k labels —
        # sharing the registered intrinsic AND engine instances (so the
        # witness endpoints match and the one-time δ-calibration is
        # shared).
        from skein_kalgebra import HeptagonSkeinKAlgebra
        skc = HeptagonSkeinKAlgebra(intrinsic=par, engine=sk)
        obj.add_realization("skein-cone", skc, {"geometric", "cone"})
        obj.add_iso("skein-cone", "a1a2k", skc.build_iso(par))
    return obj
