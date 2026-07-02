"""Exact elementary (Layer-2) traces for the frozen finite zoo.

Every finite `ConeKAlgebra` standalone reduces an arbitrary trace via
Layer 1 (the tagged-cycle + ρ²-orbit-canonicalisation reducer in
`cone_data.simplify_trace_via_cone_data`) to a `Z[q^±]`-combination of
**elementary seeds**: the identity label `()` plus one single-mult-gen
label per ρ²-orbit.  The seed traces are the *chiral-algebra characters*
of the AD theory (e.g. the two Rogers–Ramanujan functions for the
pentagon = M(2,5), the M(2,2k+3) Andrews–Gordon characters for the odd
polygons, flavoured characters for the D-series).

This module produces those seed traces **exactly** — never from
truncated approximations — via the **orthonormality bootstrap** together
with closed-form characters:

* `Tr(1)` comes from a known closed-form vacuum character where one is
  recognised (`_VACUUM_CHAR`, `ad_characters`), else from the exact
  Nahm sum on the embedded quiver spec (`vacuum_nahm`);
* the orbit seeds are pinned by exact linear algebra over cheap
  cone-data Layer-1 reductions of the deep single-mult-gen labels
  `((i,a),)`: `Tr(L)=O(q)` pins the leading seeds and the general
  orthonormality pairs `I_{La,Lb}=δ+O(q)` complete the non-leading ones
  (`_generate_bootstrap` for trivial-R entries, plus the
  `u1_bootstrap` / `su2_bootstrap` / `su2u1_bootstrap` modules for
  the flavoured ones);
* where the zoo ring is non-abelian, the Cartan series is un-branched
  to the zoo's flavour ring (SU(2)-character decomposition of the
  μ-Laurent content, certified by an exact `to_abelian` round-trip);
* seed traces are frozen as integer data in `elem_trace_data.py`
  (merged per-entry via :func:`freeze`) and served to the standalones'
  `_trace_residual` through :func:`trace_residual`, extending lazily
  and exactly past the frozen q-window on demand (memoised per
  process; the frozen windows are sized so routine use stays inside).

The frozen tables in `elem_trace_data.py` were originally produced with
a per-seed BPS-quiver derivation not included in this repository; the
bootstrap reproduces them exactly and is the generation path here.

Validation: the pentagon seed traces match the pentagon algebra's
Rogers–Ramanujan closed forms, the heptagon's match `A1A2kKAlg(2)`'s
Andrews–Gordon characters, the a1d3/a1d5 identity traces match the
hand-written SU(2)-charactered algebras, and the trace-enabled entries
pass `verify_orthonormality` + `verify_rho_twisted_trace` through the
contract surface.

Frozen windows (see `elem_trace_data.py`): pentagon K=64, heptagon
K=48, a3 K=48 (per-mg, character-generated — see below), a1d3 K=40,
a1d5 K=32.  `freeze` merges per-entry, so it is safe to run
incrementally.

TRAPEZOID CAVEAT: a flavoured trace generated at q-window K is exact
only on a wedge `|μ| ≲ (2/3)(K−k)` at 𝖖-order k — flavour tails near
the top of the window come out clipped or with spurious edge terms.
The a3 entry is frozen from the su(2)_{−4/3} closed forms
(`ad_characters.a3_elem_entry`, certified two-route; NOT via `freeze`).
When (re)generating a flavoured entry — or consuming its frozen tails
near K, or extending it lazily — generate at K′ = K + margin and keep
only the wedge, or use closed forms.  Trivial-R entries have no
flavour tails and are unaffected.

Bootstrap pair augmentation: pairs are generated **cheapest-total-degree
first, re-solving and stopping the moment no seed is free**, with the
pair's **first factor ranging over the full seed set** (a free seed can
be pinned by a pair whose two factors are OTHER seeds — its trace
appears in the Layer-1 reduction of L_idx^a·L_jj^b even when neither
factor is it).  A seed is pinned at q-order k by a pair reaching
emin≤−k, so the depth grows with k; stopping early avoids grinding the
deep mixed-monomial reductions to the Layer-1 step cap when a shallower
degree already closes the system.
Trivial-R seeds that are not frozen (or are asked past the frozen
window) are served on-demand and cached (`_TRIVIAL_REC`, mirroring the
u1 `_U1_REC`).
"""
from __future__ import annotations

from typing import Optional

from zplus_ring import (
    AbelianZPlusRing,
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
    "freeze",
    "trace_residual",
    "supported_ids",
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
# BPS oracle (exact Schur-trace engine on the embedded quiver)
# ---------------------------------------------------------------------------

_ORACLES: dict[str, object] = {}


def _bps_oracle(short_id: str):
    """Memoised `BPSKAlgebra` built from the standalone's embedded BPS
    quiver literals.  Reached only as a last-resort fallback for trace
    generation / lazy extension — the zoo's `multiply`/`rho` never touch
    it — and unconditionally unavailable in this configuration (it
    requires the BPS realisation layer): the bootstrap + closed-form
    characters cover every tabulated theory before this is reached."""
    if short_id not in _ORACLES:
        raise NotImplementedError(
            "the BPS oracle requires the BPS realisation layer, which is "
            f"not available here. Tr(1) for {short_id!r} is not supplied by "
            "a closed-form vacuum character, and the requested q-order "
            "exceeds the frozen seed window. Wire its vacuum character "
            "(ad_characters) to extend.")
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


def _zoo_ring(short_id: str):
    """The flavour ring the *zoo* class declares (not the BPS Cartan)."""
    mod, prefix = _load_standalone(short_id)
    flavor = REGEN_SPECS[short_id][2]
    if flavor == "trivial":
        return TrivialZPlusRing()
    if flavor == "u1":
        return AbelianZPlusRing(rank=1)
    if flavor == "su2":
        return SU2ZPlusRing()
    raise NotImplementedError(
        f"elem_traces: flavour {flavor!r} ({short_id}) not supported here "
        f"— su2u1 entries need a flavour-in-labels (Z-form) encoding"
    )


def _series_to_data(short_id: str, series: RPowerSeries) -> dict:
    """Serialise one exact BPS trace to frozen-table form over the
    zoo's ring: `{q_exp: int}` (trivial), `{q_exp: {(f,): c}}` (u1) or
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
        else:  # pragma: no cover — guarded by _zoo_ring
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

# u1 entries with exact closed-form chiral characters (`ad_characters`) — these
# use the characters, not the bootstrap (the bootstrap is for the char-less u1
# entries a5/a7/e7).  a3 and hexagon are the same algebra (A3).
_U1_EXACT_CHARS = {"a3", "hexagon"}


# Closed-form / Nahm-sum vacuum trace Tr(1) — the ONLY otherwise-BPS input to
# the orthonormality bootstrap.  Supplied spine-free by vacuum_nahm (the exact
# Nahm sum on the embedded BPS spec, all flavours), making the bootstrap fully
# spine-free and arbitrarily q-improvable.


# Known closed-form vacuum characters Tr(1) — PREFERRED over the Nahm sum where
# a recognised character exists (per the rule "Nahm sum unless you have a known
# character").  M(2,2n+3) (Lee–Yang-type) vacua for the (A_1,A_{2n}) trivial-
# flavour AD theories: pentagon=M(2,5), heptagon=M(2,7); rbar=0 is the vacuum.
# The orbit seeds are still pinned by the spine-free bootstrap (kept as the
# worked bootstrap example), and the Nahm-sum path remains the universal
# fallback for every other theory.
_VACUUM_CHAR = {"pentagon": (1, 0), "heptagon": (2, 0)}


# A1D_odd explicit closed-form Layer-2 characters (override the frozen tables).
# D₃ has its own standalone (`a1d3_kalg`); D₅/D₇ are served here.
def _load_a1d5_layer2():
    import a1d5_layer2
    return a1d5_layer2


def _load_a1d7_layer2():
    import a1d7_layer2
    return a1d7_layer2


_A1DODD_LAYER2 = {"a1d5": _load_a1d5_layer2, "a1d7": _load_a1d7_layer2}


def _vacuum_rps(short_id: str, K: int):
    """`Tr(1)` (the vacuum trace) as an `RPowerSeries`, spine-free: a known
    closed-form character where one is recognised (`_VACUUM_CHAR`), else the
    exact Nahm sum on the BPS spec (`vacuum_nahm`), else the BPS oracle
    (requires the BPS realisation layer, not available in this
    configuration — tabulated theories never reach it)."""
    if short_id in _VACUUM_CHAR:
        from ad_characters import m2_2np3_character
        n, rbar = _VACUUM_CHAR[short_id]
        char = m2_2np3_character(n, rbar, K)
        R = TrivialZPlusRing()
        coeffs = {q: RElement(R, {R.one_basis(): int(c)})
                  for q, c in char.items() if c and q <= K}
        return RPowerSeries(R, coeffs, K)
    from vacuum_nahm import has_spec, vacuum_trace_rps, SPECS
    mod, prefix = _load_standalone(short_id)
    if has_spec(short_id):
        pairing = getattr(mod, f"{prefix}_BPS_PAIRING")
        R = _standalone_algebra(short_id).coefficient_ring()
        return vacuum_trace_rps(SPECS[short_id], pairing, R, K)
    rank = len(getattr(mod, f"{prefix}_MULT_GENS_LATTICE")[0])
    return _bps_oracle(short_id).trace((0,) * rank, K)


def generate(short_id: str, K: int, *, verbose: bool = False,
             method: str = "auto") -> dict:
    """Compute the elementary-trace table for `short_id` exactly, to
    q-order `K`.  Returns the frozen-table record (see `elem_trace_data.py`).

    `method`:
      * `"auto"` (default) — the orthonormality bootstrap (Tr(1) from a
        closed-form character or the exact Nahm sum; the seeds from cheap
        cone-data Layer-1 reductions + exact linear algebra):
        `_generate_bootstrap` for trivial-R cone algebras,
        `u1_bootstrap.generate_u1` (ρ²-orbit-reduced μ-fugacity) for u1
        entries, and the su2 / su2u1 bootstrap modules for the flavoured
        entries;
      * `"bps"` — the per-seed BPS engine on the embedded quiver; this
        path requires the BPS realisation layer and is not available in
        this configuration (the frozen tables were originally produced
        with it);
      * `"bootstrap"` — force the bootstrap (raises if unavailable)."""
    flavor = REGEN_SPECS[short_id][2]

    def _bootstrap(sid, k, vb):
        if flavor == "u1":
            if sid in _U1_EXACT_CHARS:               # exact closed-form chars
                from ad_characters import a3_elem_entry
                return a3_elem_entry(k)
            from u1_bootstrap import generate_u1
            return generate_u1(sid, k, verbose=vb)
        if flavor == "su2":                          # SU(2)-irrep bootstrap
            from su2_bootstrap import generate_su2
            return generate_su2(sid, k, verbose=vb)
        if flavor == "su2u1":                        # rank-2 abelian bootstrap
            from su2u1_bootstrap import generate_su2u1
            return generate_su2u1(sid, k, verbose=vb)
        return _generate_bootstrap(sid, k, verbose=vb)   # trivial-R

    if method == "bootstrap":
        return _bootstrap(short_id, K, verbose)
    if method == "auto" and flavor in ("trivial", "u1", "su2", "su2u1"):
        try:
            return _bootstrap(short_id, K, verbose)
        except _BootstrapUnavailable as e:
            if verbose:
                print(f"[{short_id}] bootstrap unavailable ({e}); "
                      f"BPS fallback", flush=True)
    return _generate_bps(short_id, K, verbose=verbose)


def _generate_bps(short_id: str, K: int, *, verbose: bool = False) -> dict:
    """The per-seed BPS engine (exact Schur traces on the embedded
    quiver): Tr(1) + one trace per ρ²-orbit seed.  Requires the BPS
    realisation layer, which is not available in this configuration —
    see `_bps_oracle`."""
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
    """The orthonormality bootstrap cannot generate this entry (non-trivial
    R, reducer ceiling, or a seed left unpinned) — caller falls back to BPS."""


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

    mod, prefix = _load_standalone(short_id)
    gens = getattr(mod, f"{prefix}_MULT_GENS_LATTICE")
    rank = len(gens[0])
    from vacuum_nahm import has_spec
    if verbose:
        src = "Nahm-sum (spec, spine-free)" if has_spec(short_id) else "BPS"
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

    # assemble; per-seed BPS fallback for anything still unpinned in [1,K]
    free_in_K = {j for (j, k) in free if k <= K}
    Trj = [dict() for _ in range(n)]
    if sol:
        for (j, k), v in sol.items():
            if k <= K and v != 0:
                if v.denominator != 1:
                    raise _BootstrapUnavailable(f"non-integer seed value {v}")
                Trj[j][k] = int(v)
    orbits: dict[int, dict] = {}
    for j, idx in enumerate(idxs):
        if j in free_in_K:
            if verbose:
                print(f"[{short_id}] seed mg{idx} not pinned by bootstrap; "
                      f"BPS fallback", flush=True)
            orbits[idx] = _series_to_data(short_id, B.trace(gens[idx], K=K))
        else:
            orbits[idx] = {k: v for k, v in Trj[j].items() if v and k <= K}
    if verbose:
        print(f"[{short_id}] bootstrap pinned {n - len(free_in_K)}/{n} seeds "
              f"(certificate: {len(equations)} eqns, consistent)", flush=True)
    return {
        "K": K,
        "flavor": flavor,
        "fold": fold_policy(short_id),
        "identity": {e: v for e, v in Tr1.items() if e <= K},
        "orbits": orbits,
    }


def freeze(short_ids: list[str], K: int | dict, *,
           verbose: bool = True) -> None:
    """(Re)generate `elem_trace_data.py`, merging the given entries into
    whatever is already frozen.  `K` may be an int or a per-id dict.

    NOT safe to run concurrently: the merge base is re-read from disk
    just before writing (so slow generations pick up entries frozen by
    other processes in the meantime), but two freezes *writing* at the
    same moment can still lose one of them — run long generation jobs
    sequentially."""
    import os
    import pprint
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "elem_trace_data.py")

    def _read_disk() -> dict:
        if not os.path.exists(path):
            return {}
        ns: dict = {}
        with open(path) as f:
            exec(f.read(), ns)
        return dict(ns.get("ELEM_TRACE_DATA", {}))

    fresh = {}
    for sid in short_ids:
        k = K[sid] if isinstance(K, dict) else K
        fresh[sid] = generate(sid, k, verbose=verbose)
    # merge against the CURRENT file state (generation can take long;
    # other entries may have been frozen meanwhile)
    data = _read_disk()
    data.update(fresh)
    with open(path, "w") as f:
        f.write('"""Frozen exact elementary traces for the finite zoo.\n'
                "\n"
                "GENERATED by `elem_traces.freeze` from\n"
                "the BPS quiver literals embedded in the standalones —\n"
                "do not edit by hand.  Each entry records the identity\n"
                "trace and one trace per ρ²-orbit of mult-gens, exact to\n"
                "q-order `K` (extended lazily past `K` at runtime).\n"
                "\n"
                "EXCEPTION: the a3 entry is generated by\n"
                "`ad_characters.a3_elem_entry` from the\n"
                "su(2)_{-4/3} closed-form characters — the\n"
                "flavoured BPS route clips flavour tails near the top of\n"
                'its q-window (trapezoid defect; see ad_characters).\n"""\n'
                "\n")
        f.write("ELEM_TRACE_DATA = ")
        f.write(pprint.pformat(data, width=78, sort_dicts=True))
        f.write("\n")
    if verbose:
        print(f"froze {sorted(short_ids)} -> {path}")


# ---------------------------------------------------------------------------
# Runtime: the shared `_trace_residual` implementation
# ---------------------------------------------------------------------------

# in-process exact extensions past the frozen window: (sid, kind, rep) -> record
_EXT: dict = {}
# u1 entries: full bootstrap record cache (all seeds generated together)
_U1_REC: dict = {}
# trivial-R entries: full orthonormality-bootstrap record cache (ditto)
_TRIVIAL_REC: dict = {}
# su2 / su2u1 entries: full bootstrap record cache (ditto)
_SU2_REC: dict = {}
_SU2U1_REC: dict = {}


def supported_ids() -> set[str]:
    """Entries with a frozen elementary-trace table."""
    try:
        from elem_trace_data import ELEM_TRACE_DATA
        return set(ELEM_TRACE_DATA)
    except ImportError:
        return set()


def _frozen(short_id: str) -> Optional[dict]:
    try:
        from elem_trace_data import ELEM_TRACE_DATA
        return ELEM_TRACE_DATA.get(short_id)
    except ImportError:
        return None


def _seed_series(short_id: str, kind, K: int) -> dict:
    """Exact coefficient data `{q_exp: entry}` for one seed, valid
    through q-order `K` — from the frozen table (`kind = "identity"` or
    an orbit-rep index), the in-process extension cache, or a fresh
    exact BPS computation (`kind` may then also be a full Γ-charge for
    a multi-gen cone-monomial seed)."""
    # A1D_odd (D₅/D₇): serve the explicit closed-form sl(2)₋₂₊₂/v admissible-
    # character traces (`a1d5_layer2` / `a1d7_layer2`) — exact to arbitrary
    # q-order, overriding the under-resolved frozen tails.
    if short_id in _A1DODD_LAYER2 and (kind == "identity" or isinstance(kind, int)):
        mod = _A1DODD_LAYER2[short_id]()
        series = (mod.vacuum_trace(K) if kind == "identity"
                  else mod.seed_trace(kind, K))
        return {e: c for e, c in series.items() if e <= K}
    rec = _frozen(short_id)
    if (rec is not None and rec["K"] >= K
            and (kind == "identity" or kind in rec["orbits"])):
        return rec["identity"] if kind == "identity" else rec["orbits"][kind]
    cached = _EXT.get((short_id, kind))
    if cached is not None and cached["K"] >= K:
        return cached["data"]
    # u1 entries (single-gen seeds): serve exactly without the per-seed BPS
    # engine (infeasible on the E-series — a single e7 seed trace doesn't finish
    # at K=4).  a3/hexagon have exact closed-form characters; a5/a7/e7 use the
    # orbit-reduced bootstrap (Tr(1) the only BPS call), generated once + cached.
    if (REGEN_SPECS[short_id][2] == "u1"
            and (kind == "identity" or isinstance(kind, int))):
        recb = None
        if short_id in _U1_EXACT_CHARS:
            from ad_characters import a3_elem_entry
            recb = a3_elem_entry(K)
        else:
            recb = _U1_REC.get(short_id)
            if recb is None or recb["K"] < K:
                from u1_bootstrap import generate_u1
                try:
                    recb = generate_u1(short_id, K)
                    _U1_REC[short_id] = recb
                except _BootstrapUnavailable:
                    recb = None
        if recb is not None:
            data = (recb["identity"] if kind == "identity"
                    else recb["orbits"].get(kind, {}))
            _EXT[(short_id, kind)] = {"K": K, "data": data}
            return data
    # trivial-R entries (single-gen ρ²-orbit-rep seeds): same story — the
    # per-seed BPS engine is infeasible on E7/E8 (~200 s per e6 seed alone), so
    # serve via the orthonormality bootstrap (Tr(1) the only BPS call) when not
    # frozen or asked past the frozen window, generated once + cached.
    if (REGEN_SPECS[short_id][2] == "trivial"
            and (kind == "identity" or isinstance(kind, int))):
        recb = _TRIVIAL_REC.get(short_id)
        if recb is None or recb["K"] < K:
            try:
                recb = _generate_bootstrap(short_id, K)
                _TRIVIAL_REC[short_id] = recb
            except _BootstrapUnavailable:
                recb = None
        if recb is not None:
            data = (recb["identity"] if kind == "identity"
                    else recb["orbits"].get(kind, {}))
            _EXT[(short_id, kind)] = {"K": K, "data": data}
            return data
    # su2 / su2u1 entries (single-gen ρ²-orbit-rep seeds): serve via the
    # non-abelian orthonormality bootstrap (Tr(1) the only BPS call) — the SU(2)
    # characters fuse / the U(1) charge folds, but the seeds are still pinned by
    # cyclicity + orthonormality.  Generated once + cached; per-seed BPS fallback
    # for anything the reducer can't reach.
    _NA = {"su2": (_SU2_REC, "su2_bootstrap", "generate_su2")}
    _flav = REGEN_SPECS[short_id][2]
    if _flav in _NA and (kind == "identity" or isinstance(kind, int)):
        cache, modname, fn = _NA[_flav]
        recb = cache.get(short_id)
        if recb is None or recb["K"] < K:
            import importlib
            try:
                recb = getattr(importlib.import_module(modname), fn)(short_id, K)
                cache[short_id] = recb
            except _BootstrapUnavailable:
                recb = None
        if recb is not None:
            data = (recb["identity"] if kind == "identity"
                    else recb["orbits"].get(kind, {}))
            _EXT[(short_id, kind)] = {"K": K, "data": data}
            return data
    # su2u1 Layer-2 (a1d4 / a1d6 / a1d8).  The su2u1 standalones run on the
    # base ConeKAlgebra.trace: Layer-1 (δ-honest tagged ρ²-cyclicity,
    # `_rho2_twist_unit` supplying the μ^δ shift) reduces every label to
    # identity + single-gen SEEDS, and this Layer-2 serves those seeds from
    # the working `su2u1_trace_bootstrap` (pins them BPS-free; only Tr(1) is a
    # BPS call).  No composite ever goes through the BPS oracle on the normal
    # path (the `else` below is a guarded fallback).  This supersedes the
    # `su2u1_bootstrap` Gram/window scaffold (false q³ inconsistency on a1d4).
    if _flav == "su2u1":
        import su2u1_trace_bootstrap as _su2u1tb
        if kind == "identity" or isinstance(kind, int):
            recb = _SU2U1_REC.get(short_id)
            if recb is None or recb["K"] < K:
                recb = _su2u1tb.generate_su2u1_trace(
                    short_id, K, bps_fallback=True)
                _SU2U1_REC[short_id] = recb
            data = (recb["identity"] if kind == "identity"
                    else recb["orbits"].get(kind, {}))
        else:
            # Fallback only: with the su2u1 standalones on the base
            # ConeKAlgebra.trace, Layer-1 (tagged ρ²-cyclicity) reduces every
            # composite to single-gen seeds, so this branch is normally not
            # reached.  If Layer-1 ever emits a cone-monomial seed (a full
            # Γ-charge), it IS a canonical basis element, so its trace is the
            # BPS trace at that charge, peeled to su2u1 (spin, charge).
            B = _bps_oracle(short_id)
            data = _su2u1tb._bps_seed_nw(B, kind, K)
        _EXT[(short_id, kind)] = {"K": K, "data": data}
        return data
    # exact lazy extension (may be slow for the E-series)
    mod, prefix = _load_standalone(short_id)
    gens = getattr(mod, f"{prefix}_MULT_GENS_LATTICE")
    B = _bps_oracle(short_id)
    if kind == "identity":
        gamma = (0,) * len(gens[0])
    elif isinstance(kind, int):
        gamma = gens[kind]
    else:
        gamma = kind
    data = _series_to_data(short_id, B.trace(gamma, K=K))
    _EXT[(short_id, kind)] = {"K": K, "data": data}
    return data


def trace_residual(short_id: str, algebra, seed_label, K: int
                   ) -> RPowerSeries:
    """The shared Layer-2 plug-in for the frozen finite zoo.

    Layer 1 contracts to emit only the identity `()` and canonical
    single-mult-gen ρ²-orbit representatives `((i, 1),)`; those are
    served from the frozen table.  A general cone-monomial seed (should
    Layer 1 ever emit one) is still exact: a zoo label *is* a canonical
    basis element, so its trace is the Schur trace at the charge
    `γ = Σ p·γ_i` — served from the lazy per-charge cache (extending
    past it requires the BPS realisation layer)."""
    R = algebra.coefficient_ring()
    if short_id not in REGEN_SPECS:
        raise KeyError(f"trace_residual: unknown short_id {short_id!r}")
    if (_frozen(short_id) is None
            and REGEN_SPECS[short_id][2] not in
            ("trivial", "u1", "su2", "su2u1")):
        raise NotImplementedError(
            f"{short_id}: elementary traces not available "
            f"(flavour {REGEN_SPECS[short_id][2]!r} needs a "
            f"flavour-in-labels (Z-form) encoding)"
        )
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
    """Direct (Layer-1-free) trace for unit-character entries.

    For u1 / su2u1 flavour the base Layer-1 reducer is flavour-unsafe:
    its ρ²-cyclicity slides act by *label*-level ρ², which differs from
    element-level ρ² by μ-shifts (verified: it drops a unit character
    on e.g. the a3 square L₂² — zoo (μ⁻¹+1+μ)q² vs the exact
    (1+μ+μ²)q²).  But no reduction is needed at all: a zoo label IS a
    canonical basis element with charge `γ = Σ p·γ_i`, and its trace is
    served exactly — frozen seeds for the identity and single gens,
    exact lazily-cached values for composite labels.  (Orthonormality
    over the a3 window passes with 0 bad pairs under this route.)

    A flavour-in-labels (Z-form) encoding would make label-ρ² exact and
    the Layer-1 reducer flavour-safe, removing the need for this
    bypass; for the entries served here the direct route is used."""
    return trace_residual(short_id, algebra, label, K)
