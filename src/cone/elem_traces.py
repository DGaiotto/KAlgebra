"""Exact elementary (Layer-2) traces for the finite zoo.

Every finite `ConeKAlgebra` standalone reduces an arbitrary trace via
Layer 1 (the tagged-cycle + ρ²-orbit-canonicalisation reducer in
`cone_data.simplify_trace_via_cone_data`) to a `Z[q^±]`-combination of
**elementary seeds**: the identity label `()` plus one single-mult-gen
label per ρ²-orbit.  The seed traces are the *chiral-algebra characters*
of the AD theory (e.g. the two Rogers–Ramanujan functions for the
pentagon = M(2,5), the M(2,2k+3) Andrews–Gordon characters for the odd
polygons, flavoured characters for the D-series).

:func:`trace_residual` serves those seed traces to the standalones'
`_trace_residual`, exactly, from self-contained routes only — no frozen
trace data and no runtime oracle on the serving path:

* closed forms — `a1d5` / `a1d7` from `a1d5_layer2` / `a1d7_layer2`;
  `e6` / `e8` from their W₃(3,7) / W₃(3,8) character recipes
  (`w3_seeds`); `e7` from its theta-product recipes and Bershadsky–Polyakov
  vacuum (`e7_seeds`);
* geometric classes, through a generator map built at runtime from the two
  classes' own data — `pentagon` / `heptagon` from `A1A2kKAlg(1)` /
  `A1A2kKAlg(2)` (`aeven_seeds`); `a3` / `a5` / `a7` (= `hexagon` /
  `octagon` / `decagon`) from the ungauged gauged polygon
  `ungauge_u1a1aodd(k)` (`aodd_seeds`); `a1d3` from `A1DoddConeKAlg(0)`
  (`a1d3_seeds`); `a1d4` from `SU3ADKAlg` restricted to SU(2)×U(1)
  (`a1d4_seeds`, the zoo's U(1) fugacity being the cube of the branching
  one); `a1d6` / `a1d8` from the ungauged D-even algebra `A1DevenKAlg(2)` /
  `A1DevenKAlg(3)` (`a1deven_seeds`, the zoo's U(1) fugacity being the
  ungauged gauge fugacity itself).  Closed-form products; traces exact to
  any order (the `SU3ADKAlg` `Tr(1)` and `T` / `D` seeds from the even-D
  k = 1 closed forms through its curve map — before 2026-09-24 the
  Kac–Wakimoto vacuum and its forward orthonormality pass, now witnesses;
  the `a1d6` / `a1d8` seeds from the closed forms the gauged D-even class
  serves its seeds from, `u1a1deven_seed_characters`).
  If those closed forms are switched off, the gauged class falls back to its
  exact transport, whose limit on the length of an A1Dodd word raises — here
  as a `NotImplementedError` naming the entry, the seed, the order, the class
  and the limit.

Where no route serves a seed — or Layer 1 hands over a seed shape it is not
meant to emit — the call raises `NotImplementedError` naming the entry, the
seed and the order.  Nothing falls back to the BPS engine and nothing is
read from a frozen table (`elem_trace_data.ELEM_TRACE_DATA` is empty).
Earlier releases served several entries from frozen tables, and the
`a5` / `a7` tables were wrong near the top of their windows (flavour tails
clipped from 𝖖¹⁵ on `a5`); their su2u1 route also peeled the flavour slots
the wrong way round, so `a1d4`'s 𝖖² coefficient lacked the SU(2) triplet of
the flavour adjoint.

The orthonormality bootstraps stay as witnesses, run by `generate`: the
trivial-R bootstrap `_generate_bootstrap` (`Tr(1)` from a closed-form
vacuum character, `_VACUUM_CHAR`, or the exact Nahm sum on the embedded BPS
spectrum, `vacuum_nahm`), the u(1) and su(2) bootstraps
(`u1_bootstrap.generate_u1`, `su2_bootstrap.generate_su2`) and the a3
closed-form characters (`ad_characters.a3_elem_entry`).  The trivial-R
bootstrap: the identity-pairings `Tr(((i,a),)) = O(q)` of the deep
single-mult-gen labels pin the leading seeds, and the general orthonormality
pairs `I_{La,Lb} = δ + O(q)` complete the rest.  The pairs are generated
cheapest total degree first, re-solving after each degree and stopping as
soon as no seed is free, with the pair's first factor ranging over the full
seed set (a free seed can be pinned by a pair whose two factors are OTHER
seeds — its trace appears in the Layer-1 reduction of L_idx^a·L_jj^b even
when neither factor is it).  A seed is pinned at q-order k by a pair
reaching emin ≤ −k, so the depth grows with k; stopping early keeps the deep
mixed-monomial reductions below the Layer-1 step cap.

The BPS engine is a witness on explicit request only, never a fallback:
`_bps_oracle(short_id)` builds `BPSKAlgebra` (the Step-4 layer) on the
embedded quiver literals (`<PREFIX>_BPS_PAIRING` /
`<PREFIX>_BPS_NODE_CHARGES`), and `generate(short_id, K, method="bps")`
computes an elementary-trace record seed by seed through it, un-branching
the Cartan series to the zoo's ring (`_series_to_data`; SU(2) by top-weight
peeling, certified by an exact `to_abelian` round-trip).  A flavoured BPS
trace at q-window K is exact only on a wedge `|μ| ≲ (2/3)(K−k)` at 𝖖-order
k, so generate at K′ = K + margin and keep the wedge.

Validation: the pentagon seed traces match the pentagon algebra's
Rogers–Ramanujan closed forms, the heptagon's match `A1A2kKAlg(2)`'s
Andrews–Gordon characters, the a1d3 identity trace matches the hand-written
`A1D3KAlg`, the trace-enabled entries pass `verify_orthonormality` and
`verify_rho_twisted_trace` through the contract surface, and each generator
map is certified by the generator products and ρ when it is built.
"""
from __future__ import annotations

from zplus_ring import (
    RElement,
    RPowerSeries,
    SU2ZPlusRing,
    TrivialZPlusRing,
)

from regen import REGEN_SPECS, _load_standalone


__all__ = [
    "rho2_orbit_map",
    "elementary_seed_indices",
    "generate",
    "trace_residual",
]


# ---------------------------------------------------------------------------
# ρ²-orbit folding on mg indices
# ---------------------------------------------------------------------------

def rho2_orbit_map(short_id: str) -> dict[int, int]:
    """Map every mg index to the representative (minimum index) of its
    ρ²-orbit.  Missing `RHO_PERM` keys are fixed points, matching the
    standalones' `rho` implementation."""
    mod, prefix = _load_standalone(short_id)
    perm = {int(k): int(v)
            for k, v in getattr(mod, f"{prefix}_RHO_PERM").items()}
    n = len(getattr(mod, f"{prefix}_MULT_GENS_LATTICE"))
    rho2 = {i: perm.get(perm.get(i, i), perm.get(i, i)) for i in range(n)}
    rep: dict[int, int] = {}
    for start in range(n):
        if start in rep:
            continue
        orbit = [start]
        nxt = rho2[start]
        while nxt != start:
            orbit.append(nxt)
            nxt = rho2[nxt]
        m = min(orbit)
        for i in orbit:
            rep[i] = m
    return rep


def fold_policy(short_id: str) -> str:
    """Whether single-mult-gen seeds may be folded along ρ²-orbits.

    `'rho2'` — valid for trivial R and for purely-semisimple flavour
    (su2): there are no unit characters, sections are canonical, so
    label-level ρ² IS element-level ρ² and orbit traces coincide.

    `'none'` — required whenever R has unit characters (u1, su2u1):
    label-level ρ² differs from element-level ρ² by μ-shifts, so
    ρ²-orbit members have traces differing by unit characters and every
    mg index keeps its own seed trace.  (Verified empirically: a3's
    orbit {0,3,4} carries Tr = −q, −q, −μ·q.)"""
    flavor = REGEN_SPECS[short_id][2]
    return "rho2" if flavor in ("trivial", "su2") else "none"


def elementary_seed_indices(short_id: str) -> list[int]:
    """The single-mult-gen elementary seed indices (the identity seed
    `()` is implicit): ρ²-orbit representatives under the `'rho2'` fold
    policy, every mg index under `'none'`."""
    if fold_policy(short_id) == "rho2":
        return sorted(set(rho2_orbit_map(short_id).values()))
    mod, prefix = _load_standalone(short_id)
    return list(range(len(getattr(mod, f"{prefix}_MULT_GENS_LATTICE"))))


# ---------------------------------------------------------------------------
# BPS oracle (exact Habiro/Nahm Schur engine on the embedded quiver)
# ---------------------------------------------------------------------------

_ORACLES: dict[str, object] = {}


def _bps_oracle(short_id: str):
    """Memoised `BPSKAlgebra` built from the standalone's embedded BPS
    quiver literals.  A witness only: `generate(..., method="bps")`, the
    object layer's `'bps'` realization and tests use it; nothing on the
    serving path (`trace_residual`) or in the zoo's `multiply`/`rho` touches
    it."""
    if short_id not in _ORACLES:
        from bps_kalgebra import BPSKAlgebra
        mod, prefix = _load_standalone(short_id)
        _ORACLES[short_id] = BPSKAlgebra(
            pairing=getattr(mod, f"{prefix}_BPS_PAIRING"),
            node_charges=getattr(mod, f"{prefix}_BPS_NODE_CHARGES"),
        )
    return _ORACLES[short_id]


# ---------------------------------------------------------------------------
# Flavour un-branching: BPS Cartan series → the zoo's flavour ring
# ---------------------------------------------------------------------------

def _su2_decompose(abelian_terms: dict, R2: SU2ZPlusRing) -> dict[int, int]:
    """Decompose a (virtual) SU(2) Cartan character — a μ-Laurent given
    as `{(f,): c}` over `AbelianZPlusRing(rank=1)`, weights = charges —
    into `{w: c}` over `SU2ZPlusRing` by top-weight peeling.

    Exactness is certified by an exact `to_abelian` round-trip; a
    mismatch raises (it would mean the charge↔weight normalisation of
    this entry differs and must be handled explicitly, not silently)."""
    rem = {int(f): int(c) for (f,), c in abelian_terms.items() if c}
    out: dict[int, int] = {}
    while rem:
        w = max(abs(f) for f in rem)
        c = rem[w] if w in rem else rem[-w]
        out[w] = out.get(w, 0) + c
        for f in range(-w, w + 1, 2):
            nc = rem.get(f, 0) - c
            if nc:
                rem[f] = nc
            else:
                rem.pop(f, None)
    # exact round-trip certificate
    back: dict = {}
    for w, c in out.items():
        ab = R2.to_abelian(RElement(R2, {w: c}))
        for key, cc in ab.terms.items():
            back[key] = back.get(key, 0) + cc
    back = {k: v for k, v in back.items() if v}
    inp = {k: v for k, v in abelian_terms.items() if v}
    if back != inp:
        raise ValueError(
            f"_su2_decompose: to_abelian round-trip mismatch "
            f"(got {back}, want {inp}) — charge/weight normalisation "
            f"of this entry needs explicit handling"
        )
    return out


def _series_to_data(short_id: str, series: RPowerSeries) -> dict:
    """Serialise one exact trace series to the elementary-trace record form
    over the zoo's ring: `{q_exp: int}` (trivial), `{q_exp: {(f,): c}}` (u1) or
    `{q_exp: {w: c}}` (su2, un-branched)."""
    flavor = REGEN_SPECS[short_id][2]
    out: dict = {}
    for e, c in sorted(series.coeffs.items()):
        if isinstance(c, RElement):
            terms = {k: v for k, v in c.terms.items() if v}
        else:
            terms = {(0,) * 0: int(c)} if c else {}
        if not terms:
            continue
        if flavor == "trivial":
            # only the unit character may appear
            bad = [k for k in terms if k not in ((), (0,))]
            if bad:
                raise ValueError(
                    f"{short_id}: non-trivial flavour content {terms} "
                    f"in a trivial-R entry"
                )
            out[e] = sum(terms.values())
        elif flavor == "u1":
            out[e] = {tuple(k): int(v) for k, v in terms.items()}
        elif flavor == "su2":
            out[e] = _su2_decompose(terms, SU2ZPlusRing())
        else:  # pragma: no cover — su2u1 entries are refused before any record is built
            raise NotImplementedError(flavor)
    return out


def _data_to_relement(short_id: str, ring, entry) -> RElement:
    """Inverse of `_series_to_data` at a single q-order."""
    if isinstance(entry, int):
        return RElement(ring, {ring.one_basis(): entry})
    return RElement(ring, {k: v for k, v in entry.items()})


# ---------------------------------------------------------------------------
# Generation + freezing
# ---------------------------------------------------------------------------

# u1 entries with exact closed-form chiral characters (`ad_characters`): their
# `generate` record is the characters, not the bootstrap.  a3 and hexagon are
# the same algebra (A3).  Since 2026-09-23 the served route for both is the
# ungauged polygon (`_AODD_SEEDS`); this record is its witness.
_U1_EXACT_CHARS = {"a3", "hexagon"}


# Closed-form / Nahm-sum vacuum trace Tr(1) — the ONLY otherwise-BPS input to
# the orthonormality bootstrap.  Supplied spine-free by vacuum_nahm (the exact
# Nahm sum on the embedded BPS spec, all flavours), making the bootstrap fully
# spine-free and arbitrarily q-improvable.


# Known closed-form vacuum characters Tr(1) — PREFERRED over the Nahm sum where
# a recognised character exists (per the rule "Nahm sum unless you have a known
# character").  M(2,2n+3) (Lee–Yang-type) vacua for the (A_1,A_{2n}) trivial-
# flavour AD theories: pentagon=M(2,5), heptagon=M(2,7); rbar=0 is the vacuum.
# Since 2026-09-24 the zoo serves pentagon / heptagon from `A1A2kKAlg(k)`
# (`_AEVEN_SEEDS` below); this vacuum and the spine-free bootstrap over it
# (kept as the worked bootstrap example) are the witness `generate` runs, and
# the Nahm-sum path remains the universal fallback for every other theory.
_VACUUM_CHAR = {"pentagon": (1, 0), "heptagon": (2, 0)}


# A1D_odd explicit closed-form Layer-2 characters (served ahead of any bootstrap).
# D₃ has its own standalone (`a1d3_kalg`); D₅/D₇ are served here.
def _load_a1d5_layer2():
    import a1d5_layer2
    return a1d5_layer2


def _load_a1d7_layer2():
    import a1d7_layer2
    return a1d7_layer2


_A1DODD_LAYER2 = {"a1d5": _load_a1d5_layer2, "a1d7": _load_a1d7_layer2}


# [A₁,E₆] / [A₁,E₈]: every seed a finite Z[𝖖^±]-combination of the W₃(3,7) /
# W₃(3,8) characters (`w3_seeds`) — exact to any order; they replaced the
# tables and the bootstrap, which is capped at 𝖖⁶ on E₈ by its memory wall.
# E₆'s Nahm-sum + orthonormality-bootstrap route stays as the witness.
def _load_w3(name):
    def load():
        import w3_seeds
        return getattr(w3_seeds, name)
    return load


_W3_SEEDS = {"e6": _load_w3("E6"), "e8": _load_w3("E8")}


# [A₁,E₇]: every seed a theta-product combination over `(q;q)_∞ θ(μ)` and `Tr 1`
# the Bershadsky–Polyakov vacuum product (`e7_seeds`) — exact to any order.
# The corrected u(1) bootstrap (`u1_bootstrap.generate_u1`) stays as the
# witness.
def _load_e7():
    import e7_seeds
    return e7_seeds


_E7_SEEDS = {"e7": _load_e7}


# [A₁,A₂ₖ₊₁] (a3 / a5 / a7 and their aliases hexagon / octagon / decagon):
# every seed and Tr(1) from the ungauged gauged polygon `ungauge_u1a1aodd(k)`,
# k = 1, 2, 3, through a generator map built at runtime from the gauged charges
# (`aodd_seeds`, 2026-09-23) — closed-form products and
# traces, exact to any order.  The a3 closed-form characters
# (`ad_characters.a3_elem_entry`) and the u(1) bootstrap
# (`u1_bootstrap.generate_u1`) stay as the witnesses.
def _load_aodd(short_id):
    def load():
        import aodd_seeds
        return aodd_seeds.seeds(short_id)
    return load


_AODD_SEEDS = {sid: _load_aodd(sid)
               for sid in ("a3", "hexagon", "a5", "octagon", "a7", "decagon")}


# [A₁,D₃] (a1d3): Tr(1) and both seeds from the closed-form cone class
# `A1DoddConeKAlg(0)` through a generator map found at runtime from the two
# classes' ρ-orbits and generator products (`a1d3_seeds`,
# 2026-09-23).  The su(2) bootstrap (`su2_bootstrap.generate_su2`) stays as
# the witness.
def _load_a1d3():
    import a1d3_seeds
    return a1d3_seeds


_A1D3_SEEDS = {"a1d3": _load_a1d3}


# [A₁,D₄] (a1d4): Tr(1) and every seed from `SU3ADKAlg` restricted to
# SU(2)×U(1), through a generator map and U(1) offsets found at runtime from
# the generator products, with the zoo's U(1) fugacity the cube of the
# branching one (`a1d4_seeds`, 2026-09-23).
def _load_a1d4():
    import a1d4_seeds
    return a1d4_seeds


_A1D4_SEEDS = {"a1d4": _load_a1d4}


# [A₁,D₆] / [A₁,D₈] (a1d6 / a1d8): Tr(1) and every seed from the ungauged
# D-even algebra `A1DevenKAlg(2)` / `A1DevenKAlg(3)`, through a generator map,
# a U(1) normalisation and U(1) offsets found at runtime from the generator products
# and ρ (`a1deven_seeds`, 2026-09-24); the zoo's U(1)
# fugacity is the ungauged gauge fugacity itself.  Exact up to the depth
# limit of `A1DevenKAlg`'s trace transport, past which a seed raises
# `NotImplementedError` naming the class and the limit.
def _load_a1deven(short_id):
    def load():
        import a1deven_seeds
        return a1deven_seeds.seeds(short_id)
    return load


_A1DEVEN_SEEDS = {sid: _load_a1deven(sid) for sid in ("a1d6", "a1d8")}


# [A₁,A₂ₖ] (pentagon / heptagon): Tr(1) and every seed from the closed-form
# geometric class `A1A2kKAlg(1)` / `A1A2kKAlg(2)` (the diagonals of the
# (2k+3)-gon; Layer 1 plus the M(2,2k+3) Andrews–Gordon characters), through a
# generator map found at runtime from the two classes' ρ-orbits, generator
# products and ρ (`aeven_seeds`, 2026-09-24) — exact to any
# order.  The trivial-R orthonormality bootstrap (`_generate_bootstrap`, Tr(1)
# from `_VACUUM_CHAR`) stays as the witness `generate` runs.
def _load_aeven(short_id):
    def load():
        import aeven_seeds
        return aeven_seeds.seeds(short_id)
    return load


_AEVEN_SEEDS = {sid: _load_aeven(sid) for sid in ("pentagon", "heptagon")}


def _nahm_spec_id(short_id: str):
    """The id whose Nahm spectrum (`vacuum_nahm.SPECS`) serves `short_id`: its
    own, else that of an id sharing its standalone module (the aliases
    `hexagon` / `octagon` / `decagon` share `a3` / `a5` / `a7`'s); `None` if
    there is none."""
    from vacuum_nahm import has_spec
    if has_spec(short_id):
        return short_id
    module = REGEN_SPECS[short_id][1]
    return next((sid for sid, spec in REGEN_SPECS.items()
                 if spec[1] == module and has_spec(sid)), None)


def _vacuum_rps(short_id: str, K: int):
    """`Tr(1)` (the vacuum trace) as an `RPowerSeries`, from self-contained
    routes only: a known closed-form character where one is recognised
    (`_VACUUM_CHAR`), else the exact Nahm sum on the embedded BPS spectrum
    (`vacuum_nahm`).  Every `REGEN_SPECS` entry has one of the two; a new
    entry with neither raises `NotImplementedError` naming it and the order —
    the BPS engine is not a fallback."""
    if short_id in _VACUUM_CHAR:
        from ad_characters import m2_2np3_character
        n, rbar = _VACUUM_CHAR[short_id]
        char = m2_2np3_character(n, rbar, K)
        R = TrivialZPlusRing()
        coeffs = {q: RElement(R, {R.one_basis(): int(c)})
                  for q, c in char.items() if c and q <= K}
        return RPowerSeries(R, coeffs, K)
    from vacuum_nahm import vacuum_trace_rps, SPECS
    spec_id = _nahm_spec_id(short_id)
    if spec_id is None:
        raise NotImplementedError(
            f"{short_id}: neither a closed-form vacuum character nor a Nahm "
            f"spectrum serves Tr(1) (seed 'identity') through q^{K}; the BPS "
            f"engine is a witness here, not a fallback")
    mod, prefix = _load_standalone(short_id)
    pairing = getattr(mod, f"{prefix}_BPS_PAIRING")
    R = _standalone_algebra(short_id).coefficient_ring()
    return vacuum_trace_rps(SPECS[spec_id], pairing, R, K)


# su2u1 entries: the route the zoo serves each one's traces through, named in
# the error `generate` raises (there is no elementary-trace record for su2u1).
_SU2U1_ROUTE = {
    "a1d4": "SU3ADKAlg through a1d4_seeds",
    "a1d6": "A1DevenKAlg(2) through a1deven_seeds",
    "a1d8": "A1DevenKAlg(3) through a1deven_seeds",
}


def generate(short_id: str, K: int, *, verbose: bool = False,
             method: str = "auto") -> dict:
    """Compute the elementary-trace record for `short_id` exactly, to q-order
    `K`: `{"K", "flavor", "fold", "identity": {q: entry}, "orbits": {seed
    index: {q: entry}}}`, with `entry` an `int` (trivial R), `{(f,): c}` (u1)
    or `{w: c}` (su2).

    `method`:
      * `"auto"` (default) — the orthonormality bootstrap (`_bootstrap_record`:
        `_generate_bootstrap` for trivial R, `u1_bootstrap.generate_u1`,
        `su2_bootstrap.generate_su2`; `a3` / `hexagon` from their closed-form
        characters).  If it cannot pin every seed through `K` the call raises
        `NotImplementedError` naming the entry and the order — there is no
        BPS fallback.  For the entries served from closed forms or geometric
        classes (`_seed_series`) this record is a witness, not the served
        route;
      * `"bootstrap"` — the same, raising `_BootstrapUnavailable` instead;
      * `"bps"` — the per-seed BPS engine, on explicit request only (a
        witness, not a serving route).
    The su2u1 entries raise `NotImplementedError` naming the route the zoo
    serves their traces through (`a1d4_seeds` for a1d4, `a1deven_seeds` for
    a1d6 / a1d8)."""
    if method not in ("auto", "bootstrap", "bps"):
        raise ValueError(
            f"generate: method must be 'auto', 'bootstrap' or 'bps' "
            f"(got {method!r})")
    if REGEN_SPECS[short_id][2] == "su2u1":
        raise NotImplementedError(
            f"{short_id}: no elementary-trace record through q^{K}: there is "
            f"no bootstrap for su2u1 and the BPS witness cannot un-branch it "
            f"here; the zoo serves its traces from "
            f"{_SU2U1_ROUTE.get(short_id, 'no route')}")
    if method == "bps":
        return _generate_bps(short_id, K, verbose=verbose)
    try:
        return _bootstrap_record(short_id, K, verbose=verbose)
    except _BootstrapUnavailable as e:
        if method == "bootstrap":
            raise
        raise NotImplementedError(
            f"{short_id}: the orthonormality bootstrap cannot generate the "
            f"elementary-trace record (every seed) through q^{K}: {e}; the "
            f"BPS engine is not a fallback — pass method='bps' to run it "
            f"explicitly") from e


def _generate_bps(short_id: str, K: int, *, verbose: bool = False) -> dict:
    """The per-seed BPS engine (Habiro/Nahm Schur on the embedded quiver):
    Tr(1) + one trace per ρ²-orbit seed.  Exact but heavy on the
    E-series (~200 s per e6 seed at K=8).  Reached only through
    `generate(..., method="bps")`: a witness, never a serving route."""
    mod, prefix = _load_standalone(short_id)
    gens = getattr(mod, f"{prefix}_MULT_GENS_LATTICE")
    rank = len(gens[0])
    B = _bps_oracle(short_id)
    if verbose:
        print(f"[{short_id}] Tr(1) at K={K} ...", flush=True)
    ident = B.trace((0,) * rank, K=K)
    orbits: dict[int, dict] = {}
    for rep in elementary_seed_indices(short_id):
        if verbose:
            print(f"[{short_id}] Tr(mg{rep}={gens[rep]}) at K={K} ...",
                  flush=True)
        orbits[rep] = _series_to_data(short_id, B.trace(gens[rep], K=K))
    return {
        "K": K,
        "flavor": REGEN_SPECS[short_id][2],
        "fold": fold_policy(short_id),
        "identity": _series_to_data(short_id, ident),
        "orbits": orbits,
    }


# ---------------------------------------------------------------------------
# BPS-free elementary-trace generation: the orthonormality bootstrap
# ---------------------------------------------------------------------------
#
# For a trivial-R finite ConeKAlgebra the seed traces (the chiral-algebra
# characters) satisfy orthonormality `Tr(L)=δ_{L,1}+O(q)` on every canonical
# basis element L.  The cone-data Layer-1 reducer expresses each deep
# single-mult-gen label `((i,a),)` as `Σ_s P_{i,a,s}(q)·Tr(s)` over the seeds
# + identity (cheaply); the vanishing of every q^{≤0} coefficient that stays
# closed on the seed unknowns is one exact linear equation.  Given only Tr(1)
# (one BPS call) these pin the seeds — the identity-pairings reach the
# "leading" seeds, and the general pairs `I_{La,Lb}=δ+O(q)` complete the few
# non-leading ones.  No per-seed BPS.  (SU3AD deconvolution, finite corner.)


class _BootstrapUnavailable(Exception):
    """The orthonormality bootstrap cannot generate this entry (wrong flavour
    for this bootstrap, an inconsistent system, or a seed left unpinned).  The
    serving path turns it into a `NotImplementedError` naming the entry, the
    seed and the order; nothing falls back to BPS."""


def _standalone_algebra(short_id: str):
    """Instantiate the standalone `ConeKAlgebra` for `short_id` (the cone-data
    presentation the Layer-1 reducer runs on)."""
    import inspect
    from cone_kalgebra import ConeKAlgebra
    mod, _ = _load_standalone(short_id)
    for _name, obj in vars(mod).items():
        if (inspect.isclass(obj) and issubclass(obj, ConeKAlgebra)
                and obj is not ConeKAlgebra
                and obj.__module__ == mod.__name__):
            return obj()
    raise _BootstrapUnavailable(f"no ConeKAlgebra standalone in {mod.__name__}")


def _solve_full(equations, unknowns):
    """Exact Gaussian elimination of an over-determined linear system.
    `equations` = list of (coeffs:{unk:int}, rhs).  Returns
    (solution dict, free_unknowns list, consistent bool)."""
    from fractions import Fraction as Fr
    order = list(unknowns)
    pivots: dict = {}
    for co, r in equations:
        co = {u: Fr(c) for u, c in co.items() if c}
        r = Fr(r)
        changed = True
        while changed:
            changed = False
            for u in list(co):
                if u in pivots:
                    f = co.pop(u)
                    pco, pr = pivots[u]
                    for k, v in pco.items():
                        co[k] = co.get(k, Fr(0)) + f * v
                    r -= f * pr
                    co = {k: v for k, v in co.items() if v}
                    changed = True
                    break
        if not co:
            if r != 0:
                return None, [], False
            continue
        u = next(uu for uu in order if uu in co)
        cu = co.pop(u)
        pivots[u] = ({k: -v / cu for k, v in co.items()}, r / cu)
    sol: dict = {}

    def resolve(u, seen):
        if u in sol:
            return sol[u]
        if u not in pivots or u in seen:
            return None
        co, r = pivots[u]
        val = r
        for k, v in co.items():
            kv = resolve(k, seen | {u})
            if kv is None:
                return None
            val += v * kv
        sol[u] = val
        return val
    for u in order:
        if u in pivots:
            resolve(u, set())
    free = [u for u in unknowns if u not in sol]
    return sol, free, True


def _generate_bootstrap(short_id: str, K: int, *, verbose: bool = False,
                        margin: int = 2) -> dict:
    flavor = REGEN_SPECS[short_id][2]
    if flavor != "trivial":
        raise _BootstrapUnavailable(f"flavour {flavor!r} is not trivial-R")
    from trace_uniqueness_proofs import seed_set, seed_reduction, _pair_poly
    A = _standalone_algebra(short_id)
    ident = A.identity()
    seedlabs = [s for s in seed_set(A) if s != ident]
    pos = {sl: p for p, sl in enumerate(seedlabs)}
    idxs = [sl[0][0] for sl in seedlabs]
    n = len(seedlabs)
    Ki = K + margin

    if verbose:
        src = ("closed-form character" if short_id in _VACUUM_CHAR
               else "Nahm sum")
        print(f"[{short_id}] bootstrap: Tr(1) via {src} at K={Ki} ...",
              flush=True)
    Tr1 = _series_to_data(short_id, _vacuum_rps(short_id, Ki))   # {q:int}

    equations: list = []

    def add(reduction, delta: bool):
        """Add the closed q^{≤0} equations of `Σ_s P·Tr(s) = δ·[q^0] + O(q)`."""
        P, emin = {}, 0
        for sl, poly in reduction.items():
            key = "id" if sl == ident else pos.get(sl)
            if key is None:
                continue
            P[key] = dict(poly._coeffs)
            if poly._coeffs:
                emin = min(emin, min(poly._coeffs))
        for m in range(emin, min(0, Ki + emin) + 1):
            co: dict = {}
            rhs = 1 if (delta and m == 0) else 0
            for key, poly in P.items():
                for e, c in poly.items():
                    ix = m - e
                    if key == "id":
                        if ix >= 0:
                            rhs -= c * Tr1.get(ix, 0)
                    elif 1 <= ix <= Ki:
                        co[(key, ix)] = co.get((key, ix), 0) + c
            if co or rhs:
                equations.append((co, rhs))

    # identity-pairings Tr(((i,a),))=O(q) (deep single-mult-gen labels)
    for idx in idxs:
        for a in range(2, 7):                      # reducer caps near degree 6
            try:
                add(seed_reduction(A, ((idx, a),)), False)
            except Exception:
                break
    unknowns = [(j, k) for k in range(1, Ki + 1) for j in range(n)]
    sol, free, consistent = _solve_full(equations, unknowns)
    if not consistent:
        raise _BootstrapUnavailable("identity-pairing system inconsistent")

    # general orthonormality pairs I_{La,Lb}=δ+O(q), generated
    # CHEAPEST-DEGREE-FIRST and re-solving after each total degree d=a+b: a free
    # seed is pinned at q-order k by a pair reaching emin≤−k, so the depth grows
    # with k.  Generating every pair up front (the original behaviour) wastes
    # minutes grinding the deep mixed-monomial reductions to the Layer-1 step
    # cap (the e8 "wall" at degree 4) even when a shallower degree already
    # closes the system — fatal at E8 (>900 s).  Adding only degree d and
    # stopping as soon as no seed is free keeps E8 wall-free (it closes at d=3
    # through K=6).  The pair's FIRST factor ranges over the FULL seed set, not
    # just the free seeds: a free seed's trace can be pinned by a pair whose two
    # factors are OTHER seeds (the Layer-1 reduction of L_idx^a·L_jj^b produces
    # the free seed even when neither factor is it) — restricting to free-first
    # under-constrains the deeper windows (e8 closes K=4 either way but needs
    # the full grid for K=5/6).
    free_seeds = sorted({j for (j, k) in free if k <= K})
    deg = 2
    while free_seeds and deg <= 2 * K + 2:
        for idx in idxs:
            for a in range(1, deg):
                b = deg - a
                for jj in idxs:
                    try:
                        add(_pair_poly(A, ((idx, a),), ((jj, b),)),
                            (idx, a) == (jj, b))
                    except Exception:
                        pass
        sol, free, consistent = _solve_full(equations, unknowns)
        if not consistent:
            raise _BootstrapUnavailable(
                f"pair-augmented system inconsistent (degree {deg})")
        free_seeds = sorted({j for (j, k) in free if k <= K})
        if verbose:
            print(f"[{short_id}] after degree-{deg} pairs: "
                  f"{len(free_seeds)} seeds still free", flush=True)
        deg += 1

    # assemble; a seed still unpinned in [1,K] makes the record unavailable
    # (the caller raises; there is no per-seed BPS fallback)
    free_in_K = sorted({j for (j, k) in free if k <= K})
    if free_in_K:
        raise _BootstrapUnavailable(
            f"{short_id}: seeds {[idxs[j] for j in free_in_K]} not pinned "
            f"through q^{K}")
    Trj = [dict() for _ in range(n)]
    if sol:
        for (j, k), v in sol.items():
            if k <= K and v != 0:
                if v.denominator != 1:
                    raise _BootstrapUnavailable(f"non-integer seed value {v}")
                Trj[j][k] = int(v)
    orbits: dict[int, dict] = {
        idx: {k: v for k, v in Trj[j].items() if v and k <= K}
        for j, idx in enumerate(idxs)}
    if verbose:
        print(f"[{short_id}] bootstrap pinned all {n} seeds "
              f"(certificate: {len(equations)} eqns, consistent)", flush=True)
    return {
        "K": K,
        "flavor": flavor,
        "fold": fold_policy(short_id),
        "identity": {e: v for e, v in Tr1.items() if e <= K},
        "orbits": orbits,
    }


def _bootstrap_record(short_id: str, K: int, *, verbose: bool = False) -> dict:
    """The elementary-trace record through q^K from the self-contained routes:
    the closed-form characters for `a3` / `hexagon`, else the orthonormality
    bootstrap for the entry's flavour (`_generate_bootstrap` for trivial R,
    `u1_bootstrap.generate_u1`, `su2_bootstrap.generate_su2`).  Raises
    `_BootstrapUnavailable` if it cannot pin every seed, or if the flavour has
    no bootstrap.  `_seed_series` reaches it only for the entries no closed
    form or geometric class serves — none since 2026-09-24, when the pentagon
    and heptagon seeds moved to `A1A2kKAlg` (`aeven_seeds`); for every entry
    it is the witness `generate` runs."""
    flavor = REGEN_SPECS[short_id][2]
    if flavor == "u1":
        if short_id in _U1_EXACT_CHARS:               # exact closed-form chars
            from ad_characters import a3_elem_entry
            return a3_elem_entry(K)
        from u1_bootstrap import generate_u1
        return generate_u1(short_id, K, verbose=verbose)
    if flavor == "su2":                               # SU(2)-irrep bootstrap
        from su2_bootstrap import generate_su2
        return generate_su2(short_id, K, verbose=verbose)
    if flavor == "trivial":
        return _generate_bootstrap(short_id, K, verbose=verbose)
    raise _BootstrapUnavailable(
        f"{short_id}: flavour {flavor!r} has no orthonormality bootstrap")


# ---------------------------------------------------------------------------
# Runtime: the shared `_trace_residual` implementation
# ---------------------------------------------------------------------------

# served series per (short_id, kind), valid through the recorded "K"
_EXT: dict = {}
# elementary-trace records per flavour (all seeds generated together)
_U1_REC: dict = {}
_TRIVIAL_REC: dict = {}
_SU2_REC: dict = {}
_REC_CACHE = {"u1": _U1_REC, "trivial": _TRIVIAL_REC, "su2": _SU2_REC}


def _record_entry(short_id: str, rec: dict, kind, K: int) -> dict:
    """One seed's series in an elementary-trace record.  A seed the record
    lacks raises, rather than reading as a zero trace."""
    if kind == "identity":
        return rec["identity"]
    if kind not in rec["orbits"]:
        raise NotImplementedError(
            f"{short_id}: seed {kind!r} is not in the elementary-trace record "
            f"through q^{K} (its seeds are {sorted(rec['orbits'])})")
    return rec["orbits"][kind]


def _seed_series(short_id: str, kind, K: int) -> dict:
    """Exact coefficient data `{q_exp: entry}` for one seed, valid through
    q-order `K`; `kind` is `"identity"`, a seed (mult-gen) index, or the
    charge of a multi-generator cone-monomial seed.  Served from a closed form
    or an orthonormality-bootstrap record (generated once and cached per
    process).  Where no route serves the seed this raises
    `NotImplementedError` naming the entry, the seed and the order: never a
    BPS fallback, never a frozen table."""
    flavor = REGEN_SPECS[short_id][2]
    single = kind == "identity" or isinstance(kind, int)
    # A1D_odd (D₅/D₇): the explicit closed-form sl(2)₋₂₊₂/v admissible-character
    # traces (`a1d5_layer2` / `a1d7_layer2`), exact to any q-order (see
    # the design notes).  E₆ / E₈ the same way, from their W₃ recipes;
    # E₇ from its theta-product recipes and Bershadsky–Polyakov vacuum (`e7_seeds`);
    # A₂ / A₄ (pentagon / heptagon) from `A1A2kKAlg(1)` / `A1A2kKAlg(2)`
    # (`aeven_seeds`);
    # A₃ / A₅ / A₇ from the ungauged gauged polygon (`aodd_seeds`); D₃ from
    # `A1DoddConeKAlg(0)` (`a1d3_seeds`); D₄ from `SU3ADKAlg` (`a1d4_seeds`);
    # D₆ / D₈ from `A1DevenKAlg(2)` / `A1DevenKAlg(3)` (`a1deven_seeds`; the
    # seeds' closed forms, to any order).
    closed = (_A1DODD_LAYER2.get(short_id) or _W3_SEEDS.get(short_id)
              or _AEVEN_SEEDS.get(short_id)
              or _E7_SEEDS.get(short_id) or _AODD_SEEDS.get(short_id)
              or _A1D3_SEEDS.get(short_id) or _A1D4_SEEDS.get(short_id)
              or _A1DEVEN_SEEDS.get(short_id))
    if closed is not None and single:
        mod = closed()
        series = (mod.vacuum_trace(K) if kind == "identity"
                  else mod.seed_trace(kind, K))
        return {e: c for e, c in series.items() if e <= K}
    cached = _EXT.get((short_id, kind))
    if cached is not None and cached["K"] >= K:
        return cached["data"]
    unavailable = None
    if single:
        # a3/hexagon from their closed-form characters; every other trivial-R,
        # u1 or su2 entry from its orthonormality-bootstrap record.
        cache = _REC_CACHE.get(flavor, {})
        rec = cache.get(short_id)
        try:
            if rec is None or rec["K"] < K:
                rec = _bootstrap_record(short_id, K)
                cache[short_id] = rec
        except _BootstrapUnavailable as e:
            unavailable = e
        else:
            data = _record_entry(short_id, rec, kind, K)
            _EXT[(short_id, kind)] = {"K": K, "data": data}
            return data
    why = (f"the orthonormality bootstrap stops short ({unavailable})"
           if unavailable is not None else
           "it is a multi-generator seed, and Layer 1 is meant to emit only "
           "the identity and single-generator seeds")
    raise NotImplementedError(
        f"{short_id}: no exact route serves the trace of seed {kind!r} "
        f"through q^{K}: {why}.  The BPS engine is a witness here, not a "
        f"fallback (generate(..., method='bps') runs it on request)."
    ) from unavailable


def trace_residual(short_id: str, algebra, seed_label, K: int
                   ) -> RPowerSeries:
    """The shared Layer-2 plug-in for the finite zoo.

    Layer 1 contracts to emit only the identity `()` and canonical
    single-mult-gen ρ²-orbit representatives `((i, 1),)`; those are
    served by `_seed_series` from closed forms or the orthonormality
    bootstrap.  A general cone-monomial seed (should Layer 1 ever emit
    one) is keyed by its charge `γ = Σ p·γ_i`; no self-contained route
    serves it, so it raises `NotImplementedError` (its only route was the
    BPS engine, withdrawn from this path on 2026-09-23).  Any seed no
    route serves raises the same way, naming the entry, the seed and the
    order; with the even-D closed forms switched off
    (`seed_closed_forms=False` on the gauged class), an a1d6 / a1d8 seed past
    the trace transport's word limit raises naming the class and the limit."""
    R = algebra.coefficient_ring()
    if short_id not in REGEN_SPECS:
        raise KeyError(f"trace_residual: unknown short_id {short_id!r}")
    if seed_label == ():
        kind = "identity"
    elif (isinstance(seed_label, tuple) and len(seed_label) == 1
            and isinstance(seed_label[0], tuple)
            and len(seed_label[0]) == 2 and seed_label[0][1] == 1):
        i = seed_label[0][0]
        kind = (rho2_orbit_map(short_id)[i]
                if fold_policy(short_id) == "rho2" else i)
    elif (isinstance(seed_label, tuple)
            and all(isinstance(t, tuple) and len(t) == 2
                    for t in seed_label)):
        mod, prefix = _load_standalone(short_id)
        gens = getattr(mod, f"{prefix}_MULT_GENS_LATTICE")
        rank = len(gens[0])
        kind = tuple(
            sum(gens[i][k] * p for i, p in seed_label)
            for k in range(rank)
        )
    else:
        raise ValueError(
            f"{short_id}._trace_residual: unexpected seed "
            f"{seed_label!r}; expected the identity (), a "
            f"single-mult-gen ρ²-orbit representative ((i, 1),), or a "
            f"cone-monomial label"
        )
    data = _seed_series(short_id, kind, K)
    coeffs = {e: _data_to_relement(short_id, R, c)
              for e, c in data.items() if e <= K}
    return RPowerSeries(R, coeffs, K)


def zoo_trace(short_id: str, algebra, label, K: int) -> RPowerSeries:
    """`trace_residual` under its former name; nothing calls it.

    It was the Layer-1-free trace of the unit-character (u1 / su2u1)
    entries: the identity and single generators from their seed records,
    composite labels from the per-seed BPS engine, on the premise that the
    base Layer-1 reducer is flavour-unsafe on those entries.  That premise
    was withdrawn (every u1 / su2u1 entry has `fold_policy`
    `'none'`, so the μ-leaking ρ²-orbit collapse never runs on them), and
    the standalones trace composite labels through
    `ConeKAlgebra.trace` (Layer 1, then `trace_residual`).  Measured
    2026-09-23: the six squares `L_i²` of `a3` traced that way equal the BPS
    traces of their charges, `L₂²` included (`(1+μ+μ²)q²`, the value this
    docstring once said the reducer missed).  Since the BPS engine left the
    serving path (2026-09-23), a composite label passed here raises
    `NotImplementedError`, as it does in `trace_residual`."""
    return trace_residual(short_id, algebra, label, K)
