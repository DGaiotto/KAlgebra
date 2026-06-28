"""`BPSAtlas` — an ensemble of `BPSKAlgebra` charts of one abstract `A_𝖖[T]`.

*Charts + automated, certified `KAlgebraIso` transition maps across mutation
chains.*  This is the higher structure that hosts the public chart/mutation
surface (Plan 35) — deliberately **not** methods on an individual
`BPSKAlgebra` (whose chart graph stays a private accelerator).  It rides on the Plan-25 object layer
(`KAlgebraObject`) for the certified core and promotes the building blocks in
`bps_chart_object.py` to a navigable object.

Each chart is a *presentation* of the same abstract `A_𝖖[T]`; a mutation
(cluster necklacing) is a full `KAlgebraIso` preserving `multiply`, `ρ`, **and
the Schur index** — so the atlas demonstrates the repo's axiomatics against the
cluster machinery, and the full rotation monodromy closes to `ρ²` (tested on the
pentagon, `chart_monodromy_iso`).

Design (Plan 35):

* the atlas owns **one intrinsic label space** (the root chart's labels);
* a **chart key** is the mutation path from root — a tuple of `(node_index,
  direction)` steps, `()` = root;
* each materialized chart caches its **composed `root → chart` iso**, so any
  transition `iso(src, dst)` is `(root→dst) ∘ (root→src)⁻¹`;
* `S` is *not* transported by the fragile wall-crossing conjugation — the
  transition map is the certified `μ_g` label iso (Plan 35 D3).  Direct
  (spec-free) per-chart `S` and transparent cross-chart cache transport are the
  next tasks (T3/T4), not in this MVP.

The public surface here is the Plan-35 O1 *recommendation*; trim/rename on the
user's ruling.
"""
from __future__ import annotations

from typing import Sequence

from bps_kalgebra import BPSKAlgebra
from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from kalgebra_object import KAlgebraObject
from laurent_poly import LaurentPoly
from bps_chart_object import mutation_path_iso, chart_monodromy_iso

_ONE = LaurentPoly.one()

Step = tuple        # (node_index: int, direction: 'fwd' | 'inv')
ChartKey = tuple    # tuple of Steps from the root; () = root


def _rational_inverse(M):
    """Exact inverse of a square integer matrix over the rationals, or `None`
    if singular.  Unlike `bps_chart_object._int_inverse` this does **not**
    require `M` to be unimodular — the SU(2)-type BPS charts have `det M = ±2`
    (Kronecker pairing `⟨γ_1,γ_2⟩=2`), yet a chart-iso Γ-automorphism
    `g = M'σ·M⁻¹` can still land in GL(n, Z) (e.g. `g = M·M⁻¹ = I` for
    root↔root); the integrality of `g`, not of `M`, is what matters."""
    from fractions import Fraction
    n = len(M)
    A = [[Fraction(M[i][j]) for j in range(n)]
         + [Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if A[r][col] != 0), None)
        if piv is None:
            return None                       # singular
        A[col], A[piv] = A[piv], A[col]
        inv = 1 / A[col][col]
        A[col] = [x * inv for x in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [x - f * y for x, y in zip(A[r], A[col])]
    return [[A[i][n + j] for j in range(n)] for i in range(n)]


class BPSAtlas:
    """An ensemble of `BPSKAlgebra` charts + automated certified transition isos."""

    def __init__(self, seed: BPSKAlgebra, *, max_local_moves: int = 0,
                 build_S_charts: bool = False,
                 spec_free_sigma: str = "auto", build_S_cutoff=None,
                 iso_check: bool = True):
        """`seed` is the root chart.  `max_local_moves` is forwarded to the
        chart graph (0 = strict "spec cooperates" — only a spec-head node
        necklaces; positive allows pentagon-collapse / commute rearrangements
        first, all canonical-basis-preserving).

        `build_S_charts=True` (T4) materializes each non-root chart **spec-free**
        (`build_S=True`, the Upgrade-A direct-`S` engine) instead of via the
        necklaced spec — the robust path when a chart's necklaced spec is
        long/fragile (D4).  `spec_free_sigma` (#701) selects the σ source for
        those charts: **`"principled"`** (default) installs the axiom-derived
        `σ⁻¹=−upper(F)` / `σ=−upper(F̃)` (fast, works for **no-finite-spec**
        cyclic-gauge / chamberless charts where extraction can't and the global
        tRG is intractable); `"auto"` extracts a finite spec when one exists (tRG
        fallback); `"trg"` forces tRG.  `build_S_cutoff` is the F-solve cone
        cutoff: **`None` (default) = auto-stabilize** (#714 — the build engine
        grows the cone until the node canonicals are σ-stable, so spec-free
        charts need no user knob; `multiply` truncates gracefully beyond the
        built degree).  Pin an int to fix the cutoff (e.g. for deeper products).

        **Caveat (user, 2026-06-27):** direct-`S` is a *conjecture*; a `build_S`
        chart's `S` is not ground truth, so a cross-chart index disagreement
        against it (beyond truncation) may be a **conjecture objection**, not a
        truncation error — `cross_validate` reports per-chart provenance for
        exactly this reason."""
        if not isinstance(seed, BPSKAlgebra):
            raise TypeError("BPSAtlas: seed must be a BPSKAlgebra")
        self._root = seed
        self._mlm = int(max_local_moves)
        self._build_S_charts = bool(build_S_charts)
        self._spec_free_sigma = spec_free_sigma
        self._build_S_cutoff = None if build_S_cutoff is None else int(build_S_cutoff)
        self._charts: dict[ChartKey, BPSKAlgebra] = {(): seed}
        self._iso: dict[ChartKey, KAlgebraIso] = {
            (): KAlgebraIso.identity(seed, name="atlas[root]")
        }
        # Provenance of each chart's S: 'spec' (necklaced finite spec, verified
        # construction) vs 'build_S' (conjectural direct-S).  The root inherits
        # whatever the seed is; spec-mode if it has a spec, else build_S.
        self._provenance: dict[ChartKey, str] = {
            (): "spec" if getattr(seed, "spec", None) else "build_S"}
        # ---- the memoized intrinsic-data layer (Plan 35 D5/D9) ----------
        # `multiply`, `ρ`, `ρ⁻¹`, `Tr`, `I_{a,b}` are all chart-independent *by
        # axiom*, so the atlas is their natural owner: each is computed once
        # (at the root chart) and memoized by intrinsic (root) label, reused
        # across every chart and query.  `F` is the root's quantum-torus image
        # (S-free transport to other charts, T3a); multiply transports by the
        # certified label iso; `Tr`/`I_{a,b}` are pure invariants (no fragile FS
        # transport — D3).
        self._F_root: dict[tuple, dict] = {}
        self._mult_root: dict[tuple, Element] = {}
        self._rho_cache: dict[tuple, tuple] = {}
        self._rho_inv_cache: dict[tuple, tuple] = {}
        self._trace_cache: dict[tuple, object] = {}
        self._ip_cache: dict[tuple, object] = {}
        # ---- chart-isomorphism classification (user, 2026-06-28) ----------
        # "iso-checking new bps charts should be done when the charts are added":
        # every chart is classified **at materialization** against the known
        # quiver-iso-class representatives (node-perm + Γ-automorphism).  A chart
        # recognised as an automorphism-image of an earlier representative is
        # recorded with its witness iso (its work transports, no recompute);
        # otherwise it opens a new class.  This folds the (usually infinite)
        # atlas onto its fundamental domain as it is built.  `iso_check=False`
        # disables it (e.g. for a large sweep where the per-add cost matters).
        self._iso_check = bool(iso_check)
        self._iso_class: dict[ChartKey, dict] = {
            (): {"representative": (), "witness": None, "new_class": True}}
        self._representatives: list[ChartKey] = [()]
        # ---- automorphism-expanded memoization (user, 2026-06-28) ----------
        # "automorphisms greatly expand power of memoized info" (+ "mult as well",
        # "and ρ").  A root automorphism φ relates the memoized intrinsic data
        # across its orbit of labels, EXACTLY (both sides in root's frame):
        #   Tr(L_{φa})   = Tr(L_a)              (invariant)
        #   I_{φa,φb}    = I_{a,b}              (invariant)
        #   L_{φa}·L_{φb}= φ(L_a·L_b)           (covariant — φ multiplicative)
        #   ρ(L_{φa})    = φ(ρ(L_a))            (covariant — φ ρ-equivariant)
        # ⚠ but **orbits are infinite** for gauge theories (user), so we must NOT
        # enumerate/spray them.  Instead the sharing is **LAZY**: on a cache MISS
        # we run a small bounded BFS from the query label over the automorphism
        # generators; if it lands on an already-cached label we reuse it
        # (transported back for the covariant data), else we compute and cache
        # only the query.  No closure, no orbit enumeration — bounded regardless
        # of whether the group is finite.  `_aut_gens` are the generators (+
        # inverses); empty ⇒ no sharing (unchanged behaviour until registered).
        self._aut_gens: list = []
        self._aut_depth: int = 2

    # ---- accessors ------------------------------------------------------
    @property
    def root(self) -> BPSKAlgebra:
        return self._root

    def keys(self) -> list[ChartKey]:
        """Keys of the materialized charts (`()` = root)."""
        return list(self._charts)

    def chart(self, key: ChartKey = ()) -> BPSKAlgebra:
        """The `BPSKAlgebra` at chart `key` (materializing it if needed)."""
        return self._charts[self._ensure(key)]

    # ---- key normalization ----------------------------------------------
    @staticmethod
    def _norm_step(s) -> Step:
        if isinstance(s, int):
            return (int(s), "fwd")
        k, d = s
        if d not in ("fwd", "inv"):
            raise ValueError(
                f"BPSAtlas: direction must be 'fwd' or 'inv', got {d!r}")
        return (int(k), d)

    @classmethod
    def _norm_key(cls, key) -> ChartKey:
        return tuple(cls._norm_step(s) for s in key)

    @staticmethod
    def _key_str(key: ChartKey) -> str:
        return "root" if not key else "·".join(f"{k}{d}" for k, d in key)

    # ---- materialization: the automated iso generator -------------------
    def _ensure(self, key) -> ChartKey:
        """Materialize chart `key` and its composed `root → key` iso.  This is
        the bare-minimum deliverable: an automated `KAlgebraIso` across the
        mutation chain.  Raises `ValueError` (from the chart graph) if a step
        does not cooperate within the local-move budget."""
        key = self._norm_key(key)
        if key not in self._charts:
            end, iso = mutation_path_iso(
                self._root, list(key), max_local_moves=self._mlm)
            prov = "spec"
            if self._build_S_charts and key:
                # T4: rebuild the chart spec-free (conjectural direct-S).  Same
                # nodes / Γ-labels, so the μ_g label iso retargets cleanly.  The
                # principled σ (#701, −upper(F)) gives a fast axiom-derived ρ even
                # for no-finite-spec charts.
                end_free = BPSKAlgebra(
                    pairing=[list(r) for r in self._root.lattice.pairing],
                    node_charges=end.node_charges, build_S=True,
                    spec_free_sigma=self._spec_free_sigma,
                    build_S_cutoff=self._build_S_cutoff)
                iso = KAlgebraIso(self._root, end_free,
                                  iso._forward, iso._inverse, name=iso.name)
                end = end_free
                prov = "build_S"
            self._charts[key] = end
            self._iso[key] = iso
            self._provenance[key] = prov
            if self._iso_check:
                self._classify_chart(key)      # iso-check AT ADD TIME (user)
        return key

    def _classify_chart(self, key: ChartKey) -> dict:
        """Classify a freshly-added chart against the known quiver-iso-class
        representatives (node-perm + Γ-automorphism).  Records and returns
        `{'representative', 'witness', 'new_class'}` — image of an earlier
        representative (with its witness iso) or a new class (a new
        representative)."""
        for rep in self._representatives:
            if rep == key:
                continue
            wit = self.chart_isomorphism(rep, key)
            if wit is not None:
                rec = {"representative": rep, "witness": wit, "new_class": False}
                self._iso_class[key] = rec
                return rec
        rec = {"representative": key, "witness": None, "new_class": True}
        self._iso_class[key] = rec
        self._representatives.append(key)
        return rec

    # ---- mutation API ---------------------------------------------------
    def mutate(self, key: ChartKey = (), node_index: int = 0,
               direction: str = "fwd") -> "tuple[ChartKey, KAlgebraIso]":
        """Mutate chart `key` at `node_index`; return `(new_key, iso)` with the
        composed, full-battery-certifiable `root → new_key` `KAlgebraIso`."""
        new_key = self._norm_key(key) + ((int(node_index), direction),)
        new_key = self._ensure(new_key)
        return new_key, self._iso[new_key]

    def mutate_head(self, key: ChartKey = ()) -> "tuple[ChartKey, KAlgebraIso]":
        """Necklace chart `key` forward at its spec head (the rotation step) —
        the index-free "rotate" move; always cooperates at `max_local_moves=0`.

        Requires a **spec-mode** chart (head-finding reads the spec).  On a
        spec-free chart (`spec_free_sigma="principled"`/`"trg"`, empty `.spec`)
        this raises — use `mutate(key, node_index)` with an explicit node
        instead; spec-free *navigation* is a follow-on (the chart graph
        necklacing is spec-based)."""
        A = self.chart(key)
        if not getattr(A, "spec", None):
            raise NotImplementedError(
                f"mutate_head: chart {self._key_str(key)!r} is spec-free "
                f"(spec_free_sigma={self._spec_free_sigma!r}); spec-head "
                f"navigation needs a spec. Use mutate(key, node_index) with an "
                f"explicit node, or omit build_S_charts/principled.")
        head = tuple(A.spec[0])
        k = next(i for i, n in enumerate(A.node_charges) if tuple(n) == head)
        return self.mutate(key, k, "fwd")

    def mutation_path(self, seq: Sequence) -> "tuple[ChartKey, KAlgebraIso]":
        """`(end_key, iso)` for a mutation path from root; each entry is a node
        index (forward) or a `(node_index, direction)` pair."""
        key = self._ensure(tuple(seq))
        return key, self._iso[key]

    def iso(self, src: ChartKey = (), dst: ChartKey = ()) -> KAlgebraIso:
        """The composed, certifiable transition map `src → dst` (routed through
        the root): `(src→root)` then `(root→dst)`."""
        s = self._ensure(src)
        d = self._ensure(dst)
        return self._iso[s].invert().compose(self._iso[d])

    def transport(self, label, src: ChartKey = (), dst: ChartKey = ()) -> Element:
        """Transport one canonical `label` from chart `src` to chart `dst`."""
        return self.iso(src, dst).map(Element({tuple(label): _ONE}))

    # ---- transparent cross-chart cache transport (T3a, S-free) ----------
    def _fwd_edge_charges(self, key: ChartKey) -> list:
        """The consumed charges of the forward necklace edges along `key`'s
        path from root (for quantum-torus `F` transport).  Forward-only in the
        MVP; raises on an `inv` step."""
        from bps_chart_object import _graph_of
        g = _graph_of(self._root)
        cur = g.root_id
        charges = []
        for k, d in self._norm_key(key):
            if d != "fwd":
                raise NotImplementedError(
                    "BPSAtlas.F transport: forward edges only (MVP T3a)")
            dst = g.mutate(cur, k, direction="fwd", max_local_moves=self._mlm)
            if dst is None:
                raise ValueError(f"F transport: step ({k}, fwd) does not cooperate")
            charges.append(g._edges[(cur, dst)].charge)
            cur = dst
        return charges

    def F(self, label, key: ChartKey = ()) -> dict:
        """`F(L_a)` served in chart `key`'s quantum torus, `a` an intrinsic
        (root) label.  Solved **once** at the root chart (cached), then
        transported across the forward necklace chain by `lattice_mutation`
        (a cheap algebraic transport — no re-solve).  S-free."""
        a = tuple(label)
        if a not in self._F_root:
            self._F_root[a] = self._root.F(a)
        F0 = self._F_root[a]
        key = self._norm_key(key)
        if not key:
            return dict(F0)
        from lattice import LatticeTorus
        from lattice_mutation import solve as _lm_solve
        cur = LatticeTorus(self._root.lattice, dict(F0))
        for c in self._fwd_edge_charges(key):
            cur = _lm_solve(cur, c)
        return dict(cur._terms)

    # ---- automorphism-expanded memoization (user, 2026-06-28) ------------
    def register_automorphisms(self, isos=None, *, depth: int = 2,
                               verify: bool = True, trace_K: int = 6) -> dict:
        """Register root automorphisms so the memoized intrinsic data
        (`trace` / `inner_product` / `multiply` / `ρ` / `ρ⁻¹`) is **shared across
        automorphism orbits** (user: *"automorphisms greatly expand power of
        memoized info"*, *"mult as well"*, *"and ρ"*).  With `isos=None`,
        discovers them from the **folded-graph self-loops** (the chart-graph
        automorphisms); otherwise pass explicit `root → root` `KAlgebraIso`s.

        **⚠ orbits are infinite** for gauge theories (user, 2026-06-28), so they
        are **never enumerated**.  The sharing is **lazy** (`depth`-bounded BFS on
        a cache miss — see the field comment in `__init__`); only the generators
        (and inverses) are stored, **no closure**.  `verify=True` runs the full
        battery on each generator first so a non-genuine map can never poison the
        caches.  Existing cache entries are *kept* (lazy lookup reuses them).
        Returns `{n_generators, depth, note}`."""
        if isos is None:
            fg = self.folded_graph()
            isos = [sl["automorphism"] for sl in fg["self_loop_automorphisms"]]
        gens = []
        for iso in isos:
            if verify and not all(self._run_battery(
                    iso, self._window(), trace_K).values()):
                continue                           # reject a non-genuine map
            gens.append(iso)
        self._aut_gens = []
        for g in gens:                             # generators AND inverses
            self._aut_gens.extend((g, g.invert()))
        self._aut_depth = int(depth)
        return {
            "n_generators": len(gens),
            "depth": self._aut_depth,
            "note": "lazy automorphism sharing (bounded BFS; orbits never "
                    "enumerated — safe for the infinite gauge case)",
        }

    def _aut_label(self, phi: KAlgebraIso, a) -> tuple:
        """Apply automorphism `phi` to a single label."""
        return tuple(next(iter(phi.map(Element({tuple(a): _ONE})).terms)))

    def _lazy_lookup(self, labels, has, get):
        """Bounded automorphism BFS from `labels` (a tuple of query labels) to
        find an **already-cached** equivalent — the lazy orbit-sharing core.
        `has(ys)` tests whether the transported labels `ys` are cached; `get(ys,
        psi)` returns the (covariant-transported, via `psi: query→ys`) value.
        Never enumerates the orbit: bounded by `_aut_depth` and visited set.
        Returns the value or `None` (compute it)."""
        if not self._aut_gens:
            return None
        from collections import deque
        ident = KAlgebraIso.identity(self._root)
        start = tuple(labels)
        seen = {start}
        dq = deque([(start, ident, 0)])
        while dq:
            ys, psi, d = dq.popleft()
            if ys != start and has(ys):
                return get(ys, psi)
            if d >= self._aut_depth:
                continue
            for g in self._aut_gens:
                ys2 = tuple(self._aut_label(g, y) for y in ys)
                if ys2 in seen:
                    continue
                seen.add(ys2)
                dq.append((ys2, psi.compose(g), d + 1))
        return None

    def multiply(self, a, b, key: ChartKey = ()) -> Element:
        """Structure constants `L_a · L_b` served in chart `key` (`a`, `b`
        intrinsic root labels).  Computed **once** at the root chart (cached),
        then transported by the certified label iso — `iso(L_a·L_b) =
        L_{iso a}·L_{iso b}` (`verify_multiplicative`).  S-free.  With registered
        automorphisms a miss is first answered by a **lazy** orbit lookup:
        `L_{φa}·L_{φb}=φ(L_a·L_b)` (covariant — transport the cached product)."""
        ab = (tuple(a), tuple(b))
        if ab not in self._mult_root:
            hit = self._lazy_lookup(
                ab,
                has=lambda ys: ys in self._mult_root,
                get=lambda ys, psi: psi.invert().map(self._mult_root[ys]))
            self._mult_root[ab] = hit if hit is not None else self._root.multiply(*ab)
        res = self._mult_root[ab]
        key = self._norm_key(key)
        return res if not key else self.iso((), key).map(res)

    # ---- intrinsic Schur index (T3b) ------------------------------------
    def provenance(self, key: ChartKey = ()) -> str:
        """How chart `key`'s `S` was built: `'spec'` (necklaced finite spec —
        verified construction) or `'build_S'` (conjectural direct-`S`)."""
        return self._provenance[self._ensure(key)]

    def rho(self, a) -> tuple:
        """`ρ(L_a)` — intrinsic (a canonical-basis permutation), memoized.  With
        registered automorphisms a miss is first answered by a **lazy** orbit
        lookup: `ρ(L_{φa})=φ(ρ(L_a))` (covariant — transport the cached label)."""
        a = tuple(a)
        if a not in self._rho_cache:
            hit = self._lazy_lookup(
                (a,),
                has=lambda ys: ys[0] in self._rho_cache,
                get=lambda ys, psi: self._aut_label(
                    psi.invert(), self._rho_cache[ys[0]]))
            self._rho_cache[a] = hit if hit is not None else tuple(self._root.rho(a))
        return self._rho_cache[a]

    def rho_inverse(self, a) -> tuple:
        """`ρ⁻¹(L_a)` — intrinsic, memoized.  Lazy orbit lookup on a miss:
        `ρ⁻¹(L_{φa})=φ(ρ⁻¹(L_a))`."""
        a = tuple(a)
        if a not in self._rho_inv_cache:
            hit = self._lazy_lookup(
                (a,),
                has=lambda ys: ys[0] in self._rho_inv_cache,
                get=lambda ys, psi: self._aut_label(
                    psi.invert(), self._rho_inv_cache[ys[0]]))
            self._rho_inv_cache[a] = (hit if hit is not None
                                      else tuple(self._root.rho_inverse(a)))
        return self._rho_inv_cache[a]

    def intrinsic_cache_info(self) -> dict:
        """Sizes of the memoized intrinsic-data caches (multiply / ρ / ρ⁻¹ /
        trace / inner_product / F)."""
        return {
            "multiply": len(self._mult_root), "rho": len(self._rho_cache),
            "rho_inverse": len(self._rho_inv_cache), "trace": len(self._trace_cache),
            "inner_product": len(self._ip_cache), "F": len(self._F_root),
            "automorphism_generators": len(self._aut_gens),
        }

    def clear_intrinsic_caches(self) -> None:
        """Drop the memoized intrinsic data (charts + isos are kept)."""
        for c in (self._F_root, self._mult_root, self._rho_cache,
                  self._rho_inv_cache, self._trace_cache, self._ip_cache):
            c.clear()

    def trace(self, a, K: int = 12):
        """The `ρ²`-twisted trace `Tr(L_a)` = the Schur index `I_a(𝖖)`.
        **Intrinsic** (chart-independent by axiom), so computed **once** at the
        root chart and cached by intrinsic label."""
        a = tuple(a)
        ck = (a, K)
        if ck not in self._trace_cache:
            hit = self._lazy_lookup(                # invariant: Tr(L_{φa})=Tr(L_a)
                (a,),
                has=lambda ys: (ys[0], K) in self._trace_cache,
                get=lambda ys, psi: self._trace_cache[(ys[0], K)])
            self._trace_cache[ck] = hit if hit is not None else self._root.trace(a, K)
        return self._trace_cache[ck]

    def inner_product(self, a, b, K: int = 12):
        """The Schur pairing `I_{a,b}(𝖖) = Tr(ρ(L_a)·L_b)`.  **Intrinsic**
        (chart-independent), computed **once** at root and cached.  With
        registered automorphisms a miss is first answered by a **lazy** orbit
        lookup: `I_{φa,φb}=I_{a,b}` (invariant)."""
        a = tuple(a)
        b = tuple(b)
        ck = (a, b, K)
        if ck not in self._ip_cache:
            hit = self._lazy_lookup(                # invariant
                (a, b),
                has=lambda ys: (ys[0], ys[1], K) in self._ip_cache,
                get=lambda ys, psi: self._ip_cache[(ys[0], ys[1], K)])
            self._ip_cache[ck] = (hit if hit is not None
                                  else self._root.inner_product(a, b, K))
        return self._ip_cache[ck]

    # ---- cone-presentation cross-validation (user, 2026-06-28) -----------
    def test_cone_presentation(self, cone, iso, *, gens=None,
                               check_multiply=True, check_rho=True) -> dict:
        """**Efficiently test an isomorphic `ConeKAlgebra`'s ray-multiplication
        and ρ** against the atlas's BPS root, via `iso : cone → root` (user,
        2026-06-28).  The cone presentation is the fast closed-form one; this
        validates its two error-prone operations — the cocycle **ray product**
        (`cross_product` / `derived_multiply` on the multiplicative generators,
        i.e. the cone **rays**) and **ρ** — against the certified BPS ground
        truth:

            ray multiply:  iso(L_g · L_h)_cone  ==  L_{iso g} · L_{iso h}_root
            ρ          :  iso(ρ_cone L_g)        ==  ρ_root(L_{iso g})

        for every pair of cone rays `g, h`.  It is **cheap**: only the
        generators (not the whole basis), the cone's closed-form multiply, and
        the atlas's **memoized** root multiply / ρ — so repeated calls reuse one
        BPS solve.  The `iso` is typically obtained by composing the BPS↔RG
        identity iso with an RG↔cone (or directly an object-layer cone↔bps)
        witness — *"use the iso to RGKAlgebras to get an iso to coneKAlgebras
        too"*.

        **Flavour (R-form vs Z-form, the section change).**  The cone carries
        flavour in its `RLaurent` cross-product *coefficients* (R-form) while the
        BPS root carries it in the *labels* (Z-form), and the iso's section choice
        is canonical only up to **unit characters** (the `μ`-torsor).  So the
        comparison is done in a **canonical R-form**: every label is pushed
        through the root's flavour-lift coordinate (`r_label_decompose` — the
        section change), folding flavour-in-label and flavour-in-coefficient into
        one `dict[section, RLaurent]`.  ρ on the root is `σ` on the section **plus
        the `⋆` conjugation** (`ρ(c·u)=c⋆·ρ(u)`); because the section is fixed only
        up to a `μ`-unit, ρ may match only **up to the flavour torsor**, so both
        the strict `rho_ok` and the flavour-augmented `rho_ok_mod_flavour`
        (`μ→1`) are reported.  (Unflavoured cones: the R-form is the Z-form and
        both ρ flags coincide.)

        `gens` overrides the ray set (default: the cone's `mult_gens`).  Returns
        `{ray_multiply_ok, rho_ok, rho_ok_mod_flavour, n_rays, n_products,
        mismatches}` — `mismatches` lists the first few disagreeing
        `(kind, g, h)` (empty ⟺ ray-multiply + strict ρ both pass).

        Delegates to the module-level `verify_cone_presentation`, which works
        against **any** BPS-backed root — including a gauge theory presented as an
        **RG flow to an IR that has a BPS chart** (flow composition fixes a BPS
        chart immediately; user, 2026-06-28), e.g. U1E7 over the nonagon."""
        return verify_cone_presentation(
            self._root, cone, iso, gens=gens,
            check_multiply=check_multiply, check_rho=check_rho)

    def cross_validate(self, a, b=None, *, K: int = 12, chambers=None) -> dict:
        """Compute the intrinsic index (`Tr(L_a)` if `b is None`, else
        `I_{a,b}`) **natively in several charts** and check agreement.

        `Tr` / `I_{a,b}` are chart-independent *by axiom*, so a disagreement is
        diagnostic — and its meaning **depends on provenance** (user,
        2026-06-27):

        * between **spec** (verified) charts ⟹ a **truncation error** (the
          per-chart finite-`K` cone window differs);
        * involving a **build_S** chart ⟹ truncation **or** a genuine
          **objection to the direct-`S` conjecture** (the recursion built a
          wrong `S` for that quiver).

        So this is a *test*, not a guarantee: agreement is a convergence /
        conjecture-consistency certificate; disagreement is a lead.  Returns
        `{'agree': bool, 'values': {chart: series}, 'provenance': {chart: str}}`
        — it does **not** assume which kind of failure occurred.
        """
        a = tuple(a)
        b = None if b is None else tuple(b)
        if chambers is None:
            chambers = self.rotation_chambers()
        values, prov = {}, {}
        for k in chambers:
            ch = self.chart(k)
            ks = self._key_str(k)
            a_k = next(iter(self.transport(a, (), k).terms))
            if b is None:
                values[ks] = ch.trace(a_k, K)
            else:
                b_k = next(iter(self.transport(b, (), k).terms))
                values[ks] = ch.inner_product(a_k, b_k, K)
            prov[ks] = self._provenance[k]
        vals = list(values.values())
        return {"agree": all(v == vals[0] for v in vals[1:]),
                "values": values, "provenance": prov}

    # ---- demonstrations (Plan 35 D6 / T5) -------------------------------
    def monodromy(self) -> KAlgebraIso:
        """The full rotation-loop automorphism of the root chart.  On every
        tested case this is **`ρ²`** — the same `ρ²` that twists the trace
        cyclicity axiom (the headline axiomatics-vs-cluster statement)."""
        return chart_monodromy_iso(self._root)

    # ---- loop-aware atlas / automorphisms from loops (T9) ---------------
    def _chart_id(self, key: ChartKey) -> tuple:
        """Canonical identity of a chart — its `(nodes, spec)` data.  Two
        mutation paths reaching the same `_chart_id` are the **same chart**;
        the pair is a **loop** in the chart graph."""
        ch = self.chart(key)
        return (tuple(map(tuple, ch.node_charges)), tuple(map(tuple, ch.spec)))

    def discover_automorphisms(self, *, max_depth: int = 6, window=None):
        """Discover chart-graph **loops** and the **automorphisms** they carry
        (Plan 35 T9).  BFS the chart graph from root (forward necklaces),
        deduping charts by `_chart_id`; when a second path reaches an
        already-seen chart, the two paths close a **loop** and the composed
        transition iso between them is an **automorphism** of the abstract
        algebra (its label map fixes the algebra up to relabelling).  Returns
        the **distinct non-identity** automorphisms as `root → root`
        `KAlgebraIso`s, deduped by their action on `window`.

        The rotation loop appears here as **`ρ²`** — so even the pentagon's
        atlas is non-trivial: its chart graph carries the `ρ²` loop (Goal 1.3 /
        3.6; the cluster-modular structure).  Each returned iso is a genuine
        automorphism — run the `verify_all` battery to certify it.
        """
        from collections import deque
        window = self._window() if window is None else [tuple(l) for l in window]
        ident_sig = tuple(window)
        seen: dict[tuple, ChartKey] = {self._chart_id(()): ()}
        depth: dict[ChartKey, int] = {(): 0}
        queue: deque = deque([()])
        auts: dict[tuple, KAlgebraIso] = {}
        n = len(self._root.node_charges)
        while queue:
            key = queue.popleft()
            if depth[key] >= max_depth:
                continue
            for k in range(n):
                try:
                    nk, _ = self.mutate(key, k, "fwd")
                except Exception:
                    continue
                cid = self._chart_id(nk)
                first = seen.get(cid)
                if first is not None:
                    if first != nk:                       # a loop: two paths → same chart
                        loop = self.iso(first, nk)         # automorphism (same chart data)
                        sig = tuple(next(iter(loop.map(Element({l: _ONE})).terms))
                                    for l in window)
                        if sig != ident_sig and sig not in auts:
                            auts[sig] = KAlgebraIso(
                                self._root, self._root,
                                loop._forward, loop._inverse,
                                name=f"loop-aut{list(sig)}")
                else:
                    seen[cid] = nk
                    depth[nk] = depth[key] + 1
                    queue.append(nk)
        return list(auts.values())

    def _rho_orbit_window(self, *, max_size: int = 64) -> list:
        """The label window closed under `ρ`: the union of the `ρ`-orbits of the
        node charges.  **Bounded** by `max_size` — for **finite-type** charts the
        `ρ`-orbit is small (the pentagon's is the 5-cycle), but for a **gauge
        theory** `ρ` can have an unbounded orbit, so this truncates at `max_size`
        rather than looping forever.  A convenience for the finite-type case (and
        the pentagon test); `automorphism_group` does **not** depend on it (it
        must handle the infinite-orbit gauge case)."""
        orbit: list = []
        for g in self._root.node_charges:
            start = tuple(g)
            l = start
            while len(orbit) < max_size:
                if l not in orbit:
                    orbit.append(l)
                l = self.rho(l)
                if l == start:
                    break
        return orbit

    def automorphism_group(self, *, max_depth: int = 6, max_order: int = 200,
                           max_window: int = 96):
        """Close the discovered loop automorphisms (`discover_automorphisms`)
        into the **group** they generate — the chart graph's *cluster-modular*
        action on the canonical basis (Plan 35 T9).

        Each generator is a `root → root` `KAlgebraIso`; the closure walks the
        subgroup they generate (left-multiplication BFS from the identity),
        deduplicating elements by their action on a **growing label window**
        seeded from the node charges.  Two **regimes**, and the result reports
        which (user, 2026-06-28):

        * **finite type** (Argyres–Douglas, e.g. the pentagon) — the cluster
          modular group is **finite**; the window and the group both close.
          The pentagon gives **`Z/5 = ⟨ρ²⟩`** (`order=5`, `finite=True`),
          reconciling the period-4 rotation 4-cycle with A₂'s periodicity-5
          (the 4-cycle's `ρ²` monodromy has order 5, so the loop is traversed 5
          times to return to the identity automorphism).
        * **gauge theories** (U(1)-gauged, SU(2), …) — the cluster modular group
          is **typically infinite**; the label window escapes any bound and the
          closure hits `max_order`/`max_window`.  `finite=False` here is the
          **genuine, expected** answer (the affine/wild cluster modular group),
          **not** an artifact — only the *generators* (and the fact of
          infinitude) are the meaningful output.

        Returns `{order, finite, closed, n_generators, generator_signatures,
        signatures, note}` — `order`/`signatures` are the realised (possibly
        partial) element set; `finite` distinguishes the two regimes; the
        `note` states the reading.  An infinite gauge group is reported, not
        diagnosed as an error."""
        from kalgebra import Element as _El
        gens = self.discover_automorphisms(max_depth=max_depth)

        def _img(iso, l):
            return tuple(next(iter(iso.map(_El({l: _ONE})).terms)))

        gen_sigs = [tuple((tuple(g), _img(a, tuple(g)))
                          for g in self._root.node_charges) for a in gens]

        # Growing window, seeded from the node charges; kept closed under each
        # composed element so element signatures stay faithful.  When it would
        # exceed `max_window` the action provably needs an unbounded label set
        # ⇒ the group is infinite (the gauge regime).
        window: list = []
        for g in self._root.node_charges:
            t = tuple(g)
            if t not in window:
                window.append(t)

        ident = KAlgebraIso.identity(self._root)

        def _sig(iso):
            return tuple(_img(iso, l) for l in window)

        elements = [ident]
        seen = {_sig(ident)}
        frontier = [ident]
        infinite = False
        while frontier and len(elements) < max_order and not infinite:
            nxt = []
            for a in frontier:
                for gen in gens:
                    c = a.compose(gen)               # = gen ∘ a
                    i = 0
                    grew = False
                    while i < len(window):           # close the window under c
                        img = _img(c, window[i])
                        if img not in window:
                            if len(window) >= max_window:
                                infinite = True
                                break
                            window.append(img)
                            grew = True
                        i += 1
                    if infinite:
                        break
                    if grew:                          # window changed ⇒ refresh
                        seen = {_sig(e) for e in elements}
                    s = _sig(c)
                    if s not in seen:
                        seen.add(s)
                        elements.append(c)
                        nxt.append(c)
                if infinite:
                    break
            frontier = nxt
        finite = (not infinite) and len(elements) < max_order
        sigs = sorted(
            tuple((l, _img(e, l)) for l in window) for e in elements)
        note = ("finite cluster modular group" if finite else
                "did not close within bounds — typically the INFINITE cluster "
                "modular group of a gauge theory (expected; raise max_order / "
                "max_window only for a large *finite-type* group)")
        return {
            "order": len(elements),
            "finite": finite,
            "closed": finite,
            "n_generators": len(gens),
            "generator_signatures": gen_sigs,
            "signatures": sigs,
            "note": note,
        }

    # ---- chart isomorphism / automorphism-image recognition (T9-bis) -----
    # The practical use of "automorphisms" in an atlas (user, 2026-06-28): an
    # atlas is usually *infinite*, so while building it we want to be **alert
    # that a new chart is an automorphism-image of an old one** — then its work
    # is already done (transport from the representative), and the infinite atlas
    # folds onto a fundamental domain.  Two charts are isomorphic when their BPS
    # **quivers** are isomorphic up to a node permutation `σ` and a Γ-automorphism
    # `g` (a unimodular `g` preserving the fixed Dirac pairing `P`, `gᵀPg=P`) with
    # `g(γ_i)=γ'_{σ(i)}`.  Since the whole K_𝖖 construction is functorial in
    # `(P, charges)`, `g` induces a `KAlgebraIso`; the recognition reduces to
    # matching the quiver matrices `Q[i,j]=⟨γ_i,γ_j⟩` under `σ`, with
    # `g = M'_σ·M⁻¹` (node charges form a Z-basis of Γ, so `M` is unimodular).
    @staticmethod
    def _quiver_matrix(chart) -> list:
        """`Q[i][j] = ⟨γ_i, γ_j⟩` (the chart's BPS quiver, in the Dirac pairing)."""
        P = [list(r) for r in chart.lattice.pairing]
        g = [tuple(v) for v in chart.node_charges]
        n = len(P)
        return [[sum(a * P[r][c] * b
                     for r, a in enumerate(gi) for c, b in enumerate(gj))
                 for gj in g] for gi in g]

    # ---- raw quiver-mutation surface (charge-level; no chart materialization) --
    # The chart graph's necklace `mutate` certifies a *canonical-basis* witness
    # per step, which costs an unbounded local-move budget (E-type charts need
    # ~100 moves to expose an interior node) — so a budget-bounded fold can stall.
    # For the *mutation-complete graph* we only need (a) where each node-mutation
    # lands and (b) the node-perm+Γ-auto identification — both **pure quiver data**.
    # `_charge_mutate` is the quiver mutation in charge space (always succeeds);
    # `_raw_quiver_iso` is the chart-iso test on raw charge-lists (same g = M'σ·M⁻¹
    # certificate as `chart_isomorphisms`, but no `_ensure`).  Verified identical
    # to the necklace fold on every theory whose necklace fold closes.
    @staticmethod
    def _quiver_of(P, charges) -> list:
        """`Q[i][j] = ⟨γ_i, γ_j⟩` from a raw charge-list (Dirac pairing `P`)."""
        n = len(P)
        return [[sum(gi[r] * P[r][c] * gj[c] for r in range(n) for c in range(n))
                 for gj in charges] for gi in charges]

    @staticmethod
    def _quiver_invariant(Q):
        """A permutation-invariant signature of a quiver — the sorted multiset of
        its sorted rows.  Iso quivers share it, so it buckets candidates and keeps
        the (worst-case `n!`) node-perm search off all but same-signature charts."""
        return tuple(sorted(tuple(sorted(row)) for row in Q))

    @staticmethod
    def _charge_mutate(P, charges, i):
        """Quiver mutation of `charges` at node `i` (the BPS necklace's charge
        action): `γ_i ↦ −γ_i`; `γ_j ↦ γ_j + max(⟨γ_j,γ_i⟩, 0)·γ_i` (`j ≠ i`)."""
        n = len(P)
        gi = charges[i]
        out = []
        for j, gj in enumerate(charges):
            if j == i:
                out.append(tuple(-x for x in gi))
                continue
            d = sum(gj[r] * P[r][c] * gi[c] for r in range(n) for c in range(n))
            out.append(tuple(gj[t] + d * gi[t] for t in range(n)) if d > 0
                       else tuple(gj))
        return out

    @staticmethod
    def _gen_quiver_perms(QA, QB, n):
        """Backtracking generator of every node-permutation `σ` with
        `QA[i][j] == QB[σ(i)][σ(j)]` — the rows-sorted prefilter prunes early."""
        rowA = [tuple(sorted(r)) for r in QA]
        rowB = [tuple(sorted(r)) for r in QB]
        sg = [-1] * n
        used = [False] * n

        def bt(i):
            if i == n:
                yield list(sg)
                return
            for j in range(n):
                if used[j] or rowA[i] != rowB[j]:
                    continue
                if any(QA[i][i2] != QB[j][sg[i2]] or QA[i2][i] != QB[sg[i2]][j]
                       for i2 in range(i)):
                    continue
                sg[i] = j
                used[j] = True
                yield from bt(i + 1)
                used[j] = False
                sg[i] = -1

        yield from bt(0)

    @classmethod
    def _raw_quiver_iso(cls, P, ga, gb, QA=None, QB=None) -> bool:
        """Are the BPS quivers of charge-lists `ga`, `gb` isomorphic (node-perm
        `σ` + integer Γ-automorphism `g = M'σ·M⁻¹`, `det g = ±1`, `gᵀPg = P`)?
        The raw-data twin of `chart_isomorphism` — needs a full-rank square quiver
        (`len == rank`), tries every quiver-matching `σ` for a valid integral `g`."""
        from bps_chart_object import _int_inverse
        n = len(P)
        if len(ga) != n or len(gb) != n or any(len(v) != n for v in (*ga, *gb)):
            return False
        if QA is None:
            QA = cls._quiver_of(P, ga)
        if QB is None:
            QB = cls._quiver_of(P, gb)
        M = [[ga[c][r] for c in range(n)] for r in range(n)]
        Minv = _rational_inverse(M)
        if Minv is None:
            return False
        Pl = [list(r) for r in P]
        for sg in cls._gen_quiver_perms(QA, QB, n):
            Mp = [[gb[sg[c]][r] for c in range(n)] for r in range(n)]
            gq = [[sum(Mp[r][k] * Minv[k][c] for k in range(n)) for c in range(n)]
                  for r in range(n)]
            if any(x.denominator != 1 for row in gq for x in row):
                continue
            gi = [[int(x) for x in row] for row in gq]
            try:
                _int_inverse(gi)               # enforce g ∈ GL(n, Z) (det ±1)
            except ValueError:
                continue
            gT = [[gi[c][r] for c in range(n)] for r in range(n)]
            gTP = [[sum(gT[r][k] * Pl[k][c] for k in range(n)) for c in range(n)]
                   for r in range(n)]
            gTPg = [[sum(gTP[r][k] * gi[k][c] for k in range(n)) for c in range(n)]
                    for r in range(n)]
            if gTPg == Pl:
                return True
        return False

    def chart_isomorphisms(self, src: ChartKey = (), dst: ChartKey = (),
                           *, all: bool = False) -> list:
        """All quiver isomorphisms `src → dst` as witness `KAlgebraIso`s — the
        Γ-automorphism `g` (node-permutation absorbed) relabelling the canonical
        basis.  Empty if the BPS quivers are not isomorphic.  `all=False`
        (default via `chart_isomorphism`) stops at the first witness."""
        from itertools import permutations
        from bps_chart_object import _int_inverse, _apply
        A = self.chart(src)
        Bc = self.chart(dst)
        ga = [tuple(v) for v in A.node_charges]
        gb = [tuple(v) for v in Bc.node_charges]
        n = len(A.lattice.pairing)
        if len(ga) != n or len(gb) != n or any(len(v) != n for v in ga + gb):
            return []                       # only full-rank square BPS quivers
        QA, QB = self._quiver_matrix(A), self._quiver_matrix(Bc)
        P = [list(r) for r in A.lattice.pairing]
        M = [[ga[c][r] for c in range(n)] for r in range(n)]      # cols = γ_i
        # Rational inverse (M need NOT be unimodular — SU(2) charts have det ±2;
        # what must be integral is the automorphism g = M'σ·M⁻¹, checked below).
        Minv = _rational_inverse(M)
        if Minv is None:
            return []                       # node charges rank-deficient (over Q)
        out = []
        for sg in permutations(range(n)):
            if any(QA[i][j] != QB[sg[i]][sg[j]]
                   for i in range(n) for j in range(n)):
                continue
            Mp = [[gb[sg[c]][r] for c in range(n)] for r in range(n)]
            g_q = [[sum(Mp[r][k] * Minv[k][c] for k in range(n))
                    for c in range(n)] for r in range(n)]
            # g must be an INTEGER matrix (a genuine relabelling of the Z^n basis)
            if any(x.denominator != 1 for row in g_q for x in row):
                continue
            g = [[int(x) for x in row] for row in g_q]
            # certify: g preserves the Dirac pairing and matches the charges
            gT = [[g[c][r] for c in range(n)] for r in range(n)]
            gTP = [[sum(gT[r][k] * P[k][c] for k in range(n))
                    for c in range(n)] for r in range(n)]
            gTPg = [[sum(gTP[r][k] * g[k][c] for k in range(n))
                     for c in range(n)] for r in range(n)]
            if gTPg != P or any(_apply(g, ga[i]) != gb[sg[i]] for i in range(n)):
                continue
            try:
                ginv = _int_inverse(g)      # also enforces g ∈ GL(n, Z) (det ±1)
            except ValueError:
                continue                    # g integer but not unimodular — reject
            out.append(KAlgebraIso(
                A, Bc,
                lambda l, g=g: Element({_apply(g, l): _ONE}),
                lambda l, ginv=ginv: Element({_apply(ginv, l): _ONE}),
                name=f"chart-iso[{self._key_str(self._norm_key(src))}→"
                     f"{self._key_str(self._norm_key(dst))}]"))
            if not all:
                break
        return out

    def chart_isomorphism(self, src: ChartKey = (), dst: ChartKey = ()):
        """A witness `KAlgebraIso src → dst` if the BPS quivers are isomorphic
        (node-perm + Γ-automorphism), else `None`.  Run `verify_all` on it to
        certify; `verify_spec_equivalence` additionally tests that the two charts'
        *specs* agree (the conjectural canonical-`S`)."""
        isos = self.chart_isomorphisms(src, dst)
        return isos[0] if isos else None

    def recognize(self, key: ChartKey, *, among=None):
        """**Alert that chart `key` is an automorphism-image of an already-built
        chart** (the atlas work-saver).  Searches the materialized charts (or
        `among`) for one whose BPS quiver is isomorphic to `key`'s and returns
        `(rep_key, iso)` — `iso : rep → key` — so `key`'s data transports from
        `rep` instead of being recomputed; `None` if `key` is genuinely new
        (a fresh quiver-iso class).  Skips `key` itself."""
        key = self._ensure(key)
        if among is None:                       # use the add-time classification
            rec = self._iso_class.get(key)
            if rec is not None and not rec["new_class"]:
                return rec["representative"], rec["witness"]
            if rec is not None:
                return None
        pool = list(self._charts) if among is None else [self._norm_key(k) for k in among]
        for rep in pool:
            if rep == key:
                continue
            iso = self.chart_isomorphism(rep, key)
            if iso is not None:
                return rep, iso
        return None

    def iso_class(self, key: ChartKey = ()) -> dict:
        """The quiver-iso classification of chart `key`, computed **at add time**
        (user, 2026-06-28): `{'representative', 'witness', 'new_class'}` — whether
        `key` opened a new class or is an automorphism-image of an earlier
        representative (and the witness iso `representative → key`).  If eager
        checking was off (`iso_check=False`) the chart is classified lazily here."""
        key = self._ensure(key)
        if key not in self._iso_class:
            return self._classify_chart(key)
        return self._iso_class[key]

    def representatives(self) -> list:
        """One chart key per quiver-iso class materialized so far — the atlas's
        **fundamental domain**.  Every other materialized chart is an
        automorphism-image of one of these (so its work is already done)."""
        return list(self._representatives)

    def orbit(self, key: ChartKey = ()) -> list:
        """All materialized charts in the same quiver-iso class as `key` (mutual
        automorphism-images)."""
        key = self._ensure(key)
        rep = self._iso_class[key]["representative"]
        return [k for k in self._charts
                if self._iso_class.get(k, {}).get("representative") == rep]

    def complete(self, *, max_charts: int = 4096,
                 directions=("fwd", "inv")) -> dict:
        """**Complete the atlas — materialize *every* chart** (user, 2026-06-28:
        "for finite-type KAlgebras one could in principle complete the atlas by
        including all charts").  BFS over all node mutations in both directions
        (at the atlas's `max_local_moves`), deduplicating charts by their data
        (`_chart_id`), until the reachable chart graph **closes** — every distinct
        chart materialized, each classified into its quiver-iso class at add time.

        For a **finite-type** `A_𝖖[T]` the cluster exchange graph is **finite**,
        so the atlas completes in full: `closed=True`, `n_charts` = the exchange
        graph, and `cross_validate` / chart-invariance can then be checked against
        *every* chart, not just the rotation chambers.

        Returns `{n_charts, n_classes, closed, keys}`.  `closed=True` ⟺ the BFS
        terminated before `max_charts` (the reachable chart graph is finite —
        always so for finite type, and also for any theory whose necklace graph
        closes).  `closed=False` (the cap fired) ⟺ the reachable chart graph is
        **infinite** — a chamberless / unboundedly-mutating theory — so it cannot
        be completed; use `folded_graph` (the fundamental domain) or a bounded
        window instead.

        Note: this materializes the **mutation-reachable** charts at the atlas's
        `max_local_moves`; it is not the (infinite, for a gauge theory)
        automorphism orbit — `folded_graph()` keeps one chart per quiver-iso class
        (the fundamental domain).  `n_classes` is the number of quiver-iso classes
        among the distinct charts, **or `None` when `classified=False`**: the
        chart-iso recognition needs a full-rank **square** BPS quiver, so a
        U(1)-gauged / flavoured chart (frozen flavour nodes ⇒ `node_count ≠
        rank`) cannot be folded and reports `n_classes=None` — the *completion*
        (`n_charts`, `closed`) is unaffected; only the class count is."""
        from collections import deque
        n = len(self._root.node_charges)
        self._ensure(())
        seen: dict[tuple, ChartKey] = {self._chart_id(()): ()}
        queue: deque = deque([()])
        closed = True
        while queue and closed:
            key = queue.popleft()
            for i in range(n):
                if not closed:
                    break
                for d in directions:
                    if len(seen) >= max_charts:
                        closed = False
                        break
                    try:
                        nk, _ = self.mutate(key, i, d)
                    except Exception:
                        continue
                    cid = self._chart_id(nk)
                    if cid not in seen:
                        seen[cid] = nk
                        queue.append(nk)
        # quiver-iso class count over the DISTINCT charts (not the path-key
        # duplicates the BFS materializes).  `chart_isomorphism` needs a
        # full-rank **square** BPS quiver (`node_count == lattice rank`); a
        # U(1)-gauged / flavoured chart with frozen flavour nodes is non-square,
        # so the recognition is **unavailable** — report `n_classes=None`,
        # `classified=False` rather than a meaningless count.
        distinct = list(seen.values())
        rk = len(self._root.lattice.pairing)
        square = (len(self._root.node_charges) == rk
                  and all(len(tuple(g)) == rk for g in self._root.node_charges))
        if square:
            reps: list = []
            for k in distinct:
                if not any(self.chart_isomorphism(r, k) is not None for r in reps):
                    reps.append(k)
            n_classes = len(reps)
        else:
            n_classes = None
        return {
            "n_charts": len(seen),
            "n_classes": n_classes,
            "classified": bool(square),
            "closed": closed,
            "keys": distinct,
        }

    def mutation_complete(self, *, max_charts: int = 20000, keep=None) -> dict:
        """The **mutation-complete folded atlas** (user, 2026-06-28): every chart
        reachable by mutating **any** node, folded by **chart-isomorphism**
        (node-permutation + Γ-automorphism — the atlas's defining identification).

        A mutation can produce a chart whose BPS quiver is isomorphic to one
        already present, so the atlas folds them together — and a **finite** BPS
        theory has a **finite** set of charts even though the raw node charges
        grow without bound under iterated mutation.  The **pentagon (A₂) is a
        single chart with two outgoing mutations back to itself**; A₃ is four
        charts; … [A1,E6] sixty-seven; [A1,E8] one thousand five hundred and
        seventy-four — the whole Argyres–Douglas zoo closes (user, 2026-06-28:
        "it should be closable").

        Runs on the **quiver mutation in charge space** directly — `γ_i ↦ −γ_i`,
        `γ_j ↦ γ_j + max(⟨γ_j,γ_i⟩,0)·γ_i` — not the chart graph's necklace.  The
        necklace certifies a *canonical-basis* witness per step at an unbounded
        local-move cost (E-type charts need ~100 moves to expose an interior node,
        so a budget-bounded fold stalls); the **graph** only needs where each
        mutation lands and the node-perm+Γ-auto identification, both pure quiver
        data, so charge mutation always succeeds and the fold always closes for a
        finite theory.  Verified **identical** to the necklace fold on every
        theory whose necklace fold closes (pentagon … [A1,D5]).  The per-edge
        canonical-basis `KAlgebraIso` is still available via `mutate` / `mutation_path`
        for the (cooperating) edges; here it is the graph that is the deliverable.

        Folds **as it goes** — one representative per chart-isomorphism class,
        bucketed by a quiver invariant — so the growing charges never proliferate.
        Needs a full-rank **square** BPS quiver (the chart-iso fold is only defined
        there); a non-square / frozen-flavour-node theory (e.g. `sqed1`) returns
        `classified=False`, `closed=False`.

        Pass **`keep`** — a predicate `keep(charges) -> bool` on the mutated node
        charges — to restrict the fold to a **sub-class of charts** (user,
        2026-06-28: "an SU(3) atlas including only charts where S-finding works").
        A mutation whose target fails `keep` is a **wall**: not added, not
        expanded, counted in `walls`.  The result is then the connected component
        of `keep`-charts reachable from the root through `keep`-charts — e.g. with
        `keep = "has a finite negating sequence"` the (otherwise infinite, mostly
        wild) pure-SU(3) orbit folds to a finite **2-chart** atlas with 2 wild
        walls.  (With `keep`, `rank_regular` reflects only the kept edges; the full
        accounting is `len(edges) + walls == n_charts · n_nodes`.)

        Returns `{n_charts, charts, edges, self_loops, walls, rank_regular,
        classified, closed}`: `charts` = each representative's node charges;
        `edges` = one record `{src, node, dst}` per kept (chart, node) mutation — a
        **self-loop** when `dst == src`; `walls` = node-mutations excluded by
        `keep` (0 when `keep is None`); `rank_regular` = every chart has exactly one
        mutation edge per node (`len(edges) == n_charts · n_nodes`), the finite-type
        completeness certificate (unrestricted folds only); `closed` = finite fold
        reached without hitting `max_charts`."""
        P = [list(r) for r in self._root.lattice.pairing]
        n = len(P)
        rep0 = [tuple(v) for v in self._root.node_charges]
        square = len(rep0) == n and all(len(v) == n for v in rep0)
        if not square:
            # chart-iso (node-perm + Γ-auto) is only defined on a square quiver
            return {"n_charts": None, "charts": [], "edges": [],
                    "self_loops": 0, "walls": 0, "rank_regular": False,
                    "classified": False, "closed": False}
        reps = [rep0]
        Qs = [self._quiver_of(P, rep0)]
        buckets = {self._quiver_invariant(Qs[0]): [0]}
        edges: list = []
        walls = 0
        frontier = [0]
        closed = True
        while frontier and closed:
            nxt = []
            for ci in frontier:
                for k in range(n):
                    if len(reps) >= max_charts:
                        closed = False
                        break
                    m = self._charge_mutate(P, reps[ci], k)
                    if keep is not None and not keep(m):
                        walls += 1          # excluded chart = a wall, not expanded
                        continue
                    Qm = self._quiver_of(P, m)
                    inv = self._quiver_invariant(Qm)
                    dst = next((j for j in buckets.get(inv, ())
                                if self._raw_quiver_iso(P, reps[j], m, Qs[j], Qm)),
                               None)
                    if dst is None:
                        dst = len(reps)
                        reps.append(m)
                        Qs.append(Qm)
                        buckets.setdefault(inv, []).append(dst)
                        nxt.append(dst)
                    edges.append({"src": ci, "node": k, "dst": dst})
                if not closed:
                    break
            frontier = nxt
        return {
            "n_charts": len(reps),
            "charts": [[tuple(g) for g in r] for r in reps],
            "edges": edges,
            "self_loops": sum(1 for e in edges if e["src"] == e["dst"]),
            "walls": walls,
            "rank_regular": len(edges) == len(reps) * n,
            "classified": True,
            "closed": closed,
        }

    def folded_graph(self, *, max_steps: int = 256, max_classes=None,
                     directions=("fwd", "inv")) -> dict:
        """Build the **quotient of the chart graph by quiver-isomorphism** — the
        atlas's fundamental domain (user, 2026-06-28).  BFS expands **only class
        representatives**; when a neighbour is an automorphism-image of a known
        representative it is **folded back via its witness `(g, σ)`**, with the
        mutation edge **carefully transported into the representative's frame**
        (the directive: don't drop/mislabel an adjacency on merge).

        Concretely, an edge `rep --(i,d)--> N` where `N` folds to representative
        `R` (witness `w : R → N`) becomes a quotient edge `rep --(i,d)--> R`
        carrying the iso `transition(rep→N) ∘ w⁻¹ : rep → R`; when `R == rep`
        this is an **automorphism** of `rep` (the chart-graph self-loop — on the
        pentagon it reproduces `ρ²`).  An image chart is **not re-expanded** (its
        edges are `g`-images of the representative's), which is the work-saving
        and is what makes an infinite atlas fold onto a finite domain when the
        class set is finite.

        Bounded by `max_steps` (mutations attempted) and `max_classes` (a gauge
        theory has infinitely many classes — the partial quotient + the
        class-count-so-far is itself the benchmark).  Returns
        `{representatives, edges, self_loop_automorphisms, n_charts_visited,
        n_classes, fold_ratio, closed}`."""
        from collections import deque
        n = len(self._root.node_charges)
        self._ensure(())
        queue: deque = deque([()])
        expanded: set = set()
        edges: list = []
        self_loops: list = []
        visited: set = {()}
        steps = 0
        closed = True
        while queue:
            rep = queue.popleft()
            if rep in expanded:
                continue
            expanded.add(rep)
            for i in range(n):
                for d in directions:
                    if max_classes is not None and len(self._representatives) >= max_classes:
                        closed = False
                        break
                    if steps >= max_steps:
                        closed = False
                        break
                    steps += 1
                    try:
                        nk, _ = self.mutate(rep, i, d)     # materialize + classify
                    except Exception:
                        continue
                    visited.add(nk)
                    cls = self._iso_class[nk]
                    R = cls["representative"]
                    # transport the edge into rep's frame: rep --(i,d)--> N,
                    # N folds to R via witness w:R→N, so the quotient iso is
                    # transition(rep→N) ∘ w⁻¹ : rep → R.
                    edge_iso = self.iso(rep, nk)
                    if cls["new_class"]:
                        folded = edge_iso                  # N *is* its own rep
                    else:
                        folded = edge_iso.compose(cls["witness"].invert())
                    edges.append({"src": rep, "node": i, "direction": d,
                                  "dst": R, "iso": folded})
                    if R == rep:
                        self_loops.append({"at": rep, "node": i, "direction": d,
                                           "automorphism": folded})
                    if cls["new_class"] and R not in expanded:
                        queue.append(R)
                else:
                    continue
                break                                       # propagate inner break
        n_classes = len(self._representatives)
        return {
            "representatives": list(self._representatives),
            "edges": edges,
            "self_loop_automorphisms": self_loops,
            "n_charts_visited": len(visited),
            "n_classes": n_classes,
            "fold_ratio": (len(visited) / n_classes) if n_classes else 1.0,
            "closed": closed,
        }

    @staticmethod
    def _agree_order(r, d) -> int:
        """Largest `n` with two trace series agreeing on every `q`-power `< n`
        (capped at the shorter truncation) — the D8 leading-order measure."""
        n = 0
        while n <= min(r.K, d.K):
            if r[n] != d[n]:
                return n
            n += 1
        return n

    def verify_spec_equivalence(self, src: ChartKey = (), dst: ChartKey = (),
                                *, K: int = 8, window=None,
                                vs_recursive: bool = False) -> dict:
        """**Test that two chart-isomorphic charts have equivalent specs** — i.e.
        their (conjectural canonical) `S` agree (user, 2026-06-28).  The canonical
        spec/`S` existence is itself conjectural, so this is a *verifier*, not an
        assumption.

        Given the quiver isomorphism `g : src → dst` (`chart_isomorphism`), the
        specs are equivalent iff `g` carries `src`'s Schur index to `dst`'s — i.e.
        `I_src(a) == I_dst(g·a)` for all `a`.  Because a deep chamber's large
        charge makes the finite-`K` index window lag (catalogue B / D8), this is
        measured **leading-order** (`_agree_order`): full agreement = order `K+1`;
        a partial order that **deepens with `K`** is truncation, a fixed-order
        disagreement is a genuine objection to the canonical-`S` conjecture.

        `vs_recursive=True` additionally tests that `dst`'s spec-`S` **coincides
        with the chamber's recursive (direct, `build_S`) `S`** — the sharper
        conjecture check (built on demand; skipped gracefully if the direct-`S`
        engine cannot build that quiver).

        Returns `{isomorphic, trace_agree_order, min_trace_agree_order, full,
        exact_battery, spec_equivalent, [recursive], note}`.  `spec_equivalent`
        is conservative: exact structure (multiply/ρ/round-trip) **and** full
        leading-order index agreement at this `K`."""
        iso = self.chart_isomorphism(src, dst)
        if iso is None:
            return {"isomorphic": False,
                    "note": "BPS quivers are not isomorphic (no node-perm + "
                            "Γ-automorphism) — not the same chart up to symmetry"}
        src_ch, dst_ch = self.chart(src), self.chart(dst)
        window = self._window() if window is None else [tuple(l) for l in window]

        # exact (q-independent) structure: multiply / ρ / unit / round-trip
        batt = self._run_battery(iso, window, K)
        exact_ok = (batt["unit"] and batt["round_trip"]
                    and batt["multiplicative"] and batt["rho_equivariant"])

        # leading-order Schur-index agreement = equivalent S (D8 measure)
        orders = {}
        for a in window:
            ga = next(iter(iso.map(Element({a: _ONE})).terms))
            orders[a] = self._agree_order(src_ch.trace(a, K), dst_ch.trace(ga, K))
        min_order = min(orders.values())
        full = (min_order >= K + 1)

        note = ("specs equivalent (same S): exact structure + full index "
                "agreement" if (exact_ok and full) else
                "partial index agreement — raise K; if it deepens it is "
                "truncation (D8), if it sticks at a fixed order it is an "
                "objection to the canonical-S conjecture")
        out = {
            "isomorphic": True,
            "trace_agree_order": {a: orders[a] for a in window},
            "min_trace_agree_order": min_order,
            "full": full,
            "exact_battery": batt,
            "spec_equivalent": bool(exact_ok and full),
            "note": note,
        }
        if vs_recursive:
            out["recursive"] = self._spec_vs_recursive(dst, window, K)
        return out

    def _spec_vs_recursive(self, dst: ChartKey, window, K: int) -> dict:
        """Test `dst`'s spec-`S` index against the chamber's **recursive
        (direct `build_S`) `S`** — the conjectural canonical-`S` coincidence,
        leading-order.  Built on demand; honest-fails (`built=False`) if the
        direct-`S` engine cannot build this quiver."""
        dst = self._ensure(dst)
        ch = self.chart(dst)
        try:
            rec = BPSKAlgebra(
                pairing=[list(r) for r in self._root.lattice.pairing],
                node_charges=ch.node_charges, build_S=True,
                spec_free_sigma=self._spec_free_sigma,
                build_S_cutoff=self._build_S_cutoff)
            orders = {a: self._agree_order(ch.trace(a, K), rec.trace(a, K))
                      for a in window}
        except Exception as e:
            return {"built": False, "reason": f"{type(e).__name__}: {e}"}
        min_order = min(orders.values())
        return {
            "built": True,
            "trace_agree_order": orders,
            "min_trace_agree_order": min_order,
            "coincides": min_order >= K + 1,
            "note": ("spec-S coincides with the recursive direct-S at this K"
                     if min_order >= K + 1 else
                     "spec-S vs recursive-S agree only to a finite order — "
                     "raise K (truncation) or flag a conjecture objection if it "
                     "sticks"),
        }

    def _window(self) -> list:
        """Default label window: identity + node charges (small, exercises σ)."""
        win = [self._root.identity()]
        for g in self._root.node_charges:
            t = tuple(g)
            if t not in win:
                win.append(t)
        return win

    def _run_battery(self, iso: KAlgebraIso, src_labels, trace_K: int) -> dict:
        se = [Element({l: _ONE}) for l in src_labels]
        te = [iso.map(e) for e in se]
        return iso.verify_all(
            se, te,
            [(a, b) for a in se[1:] for b in se[1:]],
            [(a, b) for a in te[1:] for b in te[1:]],
            trace_K=trace_K)

    def rotation_chambers(self, *, max_steps: int = 64) -> list[ChartKey]:
        """Chamber keys of the pure rotation walk (necklace at the spec head each
        step) until the chart data returns to root.  `keys[0]` = root;
        `keys[-1]` carries the root chart data (the loop closes there)."""
        root = self._root
        root_data = (tuple(map(tuple, root.node_charges)),
                     tuple(map(tuple, root.spec)))
        keys: list[ChartKey] = [()]
        cur: ChartKey = ()
        for _ in range(max_steps):
            nk, _ = self.mutate_head(cur)
            keys.append(nk)
            ch = self.chart(nk)
            if (tuple(map(tuple, ch.node_charges)),
                    tuple(map(tuple, ch.spec))) == root_data:
                return keys
            cur = nk
        raise ValueError(f"rotation walk did not close within {max_steps} steps")

    def certificate(self, *, labels=None, trace_K: int = 8,
                    max_steps: int = 64) -> dict:
        """The one-call **axiomatics-vs-cluster certificate** over the rotation
        chamber chain (Plan 35 T5).  Demonstrates that a cluster mutation
        preserves the *whole* `K_𝖖` structure and that the **Schur index is a
        wall-crossing invariant**:

        * `period`                   — necklace steps for the chart to return;
        * `edge_batteries`           — per-edge full `KAlgebraIso` battery
          (unit / round-trip / multiplicative / ρ- / trace-equivariant);
        * `multiply_chart_invariant` — `L_a·L_b` computed natively in every
          chamber transports back to the same root element;
        * `trace_chart_invariant`    — the Schur index `I_a(𝖖)` is **identical**
          computed natively in every chamber (the headline demonstration);
        * `monodromy`                — `{'battery': …, 'is_rho2': bool}`: the
          full rotation loop composes to `ρ²`;
        * `all_ok`                   — the conjunction.
        """
        root = self._root
        labels = self._window() if labels is None else [tuple(l) for l in labels]
        keys = self.rotation_chambers(max_steps=max_steps)

        edge_batteries = []
        for i in range(1, len(keys)):
            iso = self.iso(keys[i - 1], keys[i])
            src_labels = [next(iter(self.transport(l, (), keys[i - 1]).terms))
                          for l in labels]
            edge_batteries.append(self._run_battery(iso, src_labels, trace_K))
        edges_ok = all(all(r.values()) for r in edge_batteries)

        # chart-invariance: multiply + Schur index, native per chamber vs root.
        a, b = labels[1], labels[-1]
        root_mult = root.multiply(a, b)
        root_tr = root.trace(a, trace_K)
        mult_inv = trace_inv = True
        for ck in keys[1:-1]:               # the closing chamber is root-data
            ch = self.chart(ck)
            a_c = next(iter(self.transport(a, (), ck).terms))
            b_c = next(iter(self.transport(b, (), ck).terms))
            if self.iso(ck, ()).map(ch.multiply(a_c, b_c)) != root_mult:
                mult_inv = False
            if ch.trace(a_c, trace_K) != root_tr:
                trace_inv = False

        mono = self.monodromy()
        mono_batt = self._run_battery(mono, labels, trace_K)
        mono_rho2 = all(
            next(iter(mono.map(Element({l: _ONE})).terms)) == root.rho(root.rho(l))
            for l in labels)
        mono_ok = all(mono_batt.values()) and mono_rho2

        return {
            "period": len(keys) - 1,
            "edge_batteries": edge_batteries,
            "multiply_chart_invariant": mult_inv,
            "trace_chart_invariant": trace_inv,
            "monodromy": {"battery": mono_batt, "is_rho2": mono_rho2},
            "all_ok": edges_ok and mult_inv and trace_inv and mono_ok,
        }

    def summary(self, *, trace_K: int = 8, max_depth: int = 6) -> dict:
        """One-call **catalogue record** for this atlas (T13 helper): the
        rotation certificate (`period`, per-edge battery, `multiply` /
        Schur-index chart-invariance, monodromy `= ρ²`) plus the **loop
        automorphisms** discovered from the chart graph.  Returns a flat dict
        suitable for tabulation:

            {period, edges_ok, multiply_chart_invariant, trace_chart_invariant,
             monodromy_is_rho2, all_ok, n_loop_automorphisms,
             automorphism_signatures}

        A `*_chart_invariant=False` is **diagnostic** (D8): for a verified
        (`spec`) chart it is a truncation — re-run at higher `trace_K`; for a
        `build_S` chart it may be a direct-`S` conjecture objection (check
        `cross_validate`).  Honest-fails (raises) when the rotation does not
        close (chamberless theories — use a bounded mutation window instead).
        """
        cert = self.certificate(trace_K=trace_K)
        # Ensure the BFS is deep enough to close the rotation loop (period),
        # so the ρ² automorphism is always among the discovered set.
        auts = self.discover_automorphisms(
            max_depth=max(max_depth, cert["period"] + 1))
        window = self._window()
        sigs = [tuple(next(iter(a.map(Element({l: _ONE})).terms)) for l in window)
                for a in auts]
        return {
            "period": cert["period"],
            "edges_ok": all(all(r.values()) for r in cert["edge_batteries"]),
            "multiply_chart_invariant": cert["multiply_chart_invariant"],
            "trace_chart_invariant": cert["trace_chart_invariant"],
            "monodromy_is_rho2": cert["monodromy"]["is_rho2"],
            "all_ok": cert["all_ok"],
            "n_loop_automorphisms": len(auts),
            "automorphism_signatures": sigs,
        }

    def as_object(self, name: str = "bps-atlas") -> KAlgebraObject:
        """A `KAlgebraObject` of the materialized charts (a star of certified
        `root → chart` witnesses), for the Plan-25 battery + coherence."""
        obj = KAlgebraObject(name)
        for key, alg in self._charts.items():
            obj.add_realization(self._key_str(key), alg,
                                {"chart", "trace-exact", "rg"})
        root_s = self._key_str(())
        for key in self._charts:
            if key == ():
                continue
            obj.add_iso(root_s, self._key_str(key), self._iso[key])
        return obj

    # ---- visualization seam (Plan 35 T10; ClusterApplet tie-in = Stage 5) --
    def applet_url(self, key: ChartKey = (), name: str | None = None) -> str:
        """A ClusterApplet share-URL for chart `key` (a thin seam over
        `clusterapplet_url`; the interactive round-trip is a separate stage)."""
        from clusterapplet_url import bpskalgebra_applet_url
        return bpskalgebra_applet_url(
            self.chart(key), name=name or f"atlas[{self._key_str(key)}]")

    def __repr__(self) -> str:
        return f"BPSAtlas(root={self._root!r}, charts={len(self._charts)})"


# ---------------------------------------------------------------------------
# cone-presentation cross-validation against any BPS-backed root (user, 2026-06-28)
# ---------------------------------------------------------------------------
def verify_cone_presentation(root, cone, iso, *, gens=None,
                             check_multiply=True, check_rho=True) -> dict:
    """Test an isomorphic `ConeKAlgebra`'s **ray-multiplication** and **ρ**
    against a BPS-backed `root` `KAlgebra`, via `iso : cone → root`.

    `root` need not be a `BPSKAlgebra`: any presentation whose structure
    constants are BPS-backed works — in particular a **gauge theory presented as
    an RG flow to an IR that has a BPS chart** (flow composition fixes a BPS
    chart immediately; user, 2026-06-28), e.g. `U1A1E7RGKAlgebra` (u(1)-gauged
    E₇) flowing to `A1A2kKAlg(3)` = the nonagon BPS chart.  This is why the
    routine takes a bare `root` rather than `self`/an atlas.

    The comparison is flavour-aware (canonical R-form / the section change): every
    label is pushed through `root.r_label_decompose` (the #555 flavour-lift
    primitive), folding flavour-in-label (Z-form) and flavour-in-coefficient
    (R-form) into one `dict[section, RLaurent]`; ρ on the root is `σ` on the
    section plus the `⋆` conjugation, so for flavoured cones ρ may match only up
    to the `μ`-unit section torsor — both strict `rho_ok` and `rho_ok_mod_flavour`
    (`μ→1`) are reported.  Returns `{ray_multiply_ok, rho_ok, rho_ok_mod_flavour,
    n_rays, n_products, mismatches}`."""
    from zplus_ring import RLaurent
    from kalgebra import _laurentpoly_to_rlaurent
    R = root.coefficient_ring()

    def _sec(label):
        # (section, R-coefficient) via the current-contract flavour-lift primitive
        # r_label_decompose (#555); the RElement is rebuilt by the canonical
        # BasisElement -> RElement lift (what the legacy bridge does internally).
        s, fk = root.r_label_decompose(tuple(label))
        return s, R.basis_element(fk)

    def canon(elem):
        out = {}
        for label, coeff in elem.terms.items():
            sec, rco = _sec(label)
            rl = coeff if isinstance(coeff, RLaurent) \
                else _laurentpoly_to_rlaurent(coeff, R)
            rl = rl * RLaurent(R, {0: rco})
            out[sec] = out[sec] + rl if sec in out else rl
        return {s: c for s, c in out.items() if not c.is_zero()}

    def augment(cR):
        out = {}
        for sec, rl in cR.items():
            d = {q: sum(rel.terms.values()) for q, rel in rl.coeffs.items()}
            d = {q: v for q, v in d.items() if v}
            if d:
                out[sec] = d
        return out

    cd = cone.cone_data()
    if gens is None:
        gens = [cd.from_cone_label(frozenset({g}), {g: 1})
                for g in sorted(cd.mult_gens())]

    def img(label):
        return next(iter(iso.map(Element({tuple(label): _ONE})).terms))

    mismatches = []
    rho_ok = rho_ok_mod_flavour = True
    if check_rho:
        for g in gens:
            lhs = canon(iso.map(Element({tuple(cone.rho(g)): _ONE})))
            rhs = {}
            for sec, rl in canon(iso.map(Element({g: _ONE}))).items():
                s2, rco = _sec(root.rho(sec))
                c = rl.star() * RLaurent(R, {0: rco})
                rhs[s2] = rhs[s2] + c if s2 in rhs else c
            rhs = {s: c for s, c in rhs.items() if not c.is_zero()}
            if lhs != rhs:
                rho_ok = False
                if augment(lhs) != augment(rhs):
                    rho_ok_mod_flavour = False
                if len(mismatches) < 8:
                    mismatches.append(("rho", tuple(g), None))
    mult_ok = True
    n_products = 0
    if check_multiply:
        for g in gens:
            for h in gens:
                n_products += 1
                lhs = canon(iso.map(cone.multiply(g, h)))
                rhs = canon(root.multiply(img(g), img(h)))
                if lhs != rhs:
                    mult_ok = False
                    if len(mismatches) < 8:
                        mismatches.append(("multiply", tuple(g), tuple(h)))
    return {"ray_multiply_ok": mult_ok, "rho_ok": rho_ok,
            "rho_ok_mod_flavour": rho_ok_mod_flavour,
            "n_rays": len(gens), "n_products": n_products,
            "mismatches": mismatches}
