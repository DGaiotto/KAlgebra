"""Gauge-theory `BPSAtlas` examples — mutation-finiteness, Catalan, and wild S's.

A **self-contained** companion to `bps_atlas_examples` (the Argyres–Douglas zoo):
the *gauge* atlases, as hard-coded BPS-chart / BPS-quiver literals — **no
`implementations/` dependency**, only the shipped spine (`bps_kalgebra`,
`bps_atlas`, `bps_quiver_tools`).

`BPSAtlas.mutation_complete` folds the quiver mutation orbit by chart-iso — it
counts **quiver-iso classes**, not a priori physical charts (a quiver is a
genuine chart only if it is a *finite-spec chamber* with a well-defined `S`).
Over the gauge atlases that distinction draws a sharp line:

* **SU(2) / A₁ class-S theories are mutation-finite** — the fold closes and every
  chart is finite-spec with a reasonable S, so `mutation_complete` *is* the
  genuine atlas.  The **SU(2)-gauged [A₁,Dₙ]** chain realises the Catalan numbers.

* **SU(3) theories are mutation-infinite** — the orbit never closes (the arrow
  multiplicities blow up doubly-exponentially), and the chambers are mostly
  **wild**: no finite negating sequence ⇒ no finite spectrum, and the recursive
  direct-`S` finder itself is **cutoff-dependent** there (no reasonable S).  So
  for SU(3) `mutation_complete` enumerates wild combinatorics, not charts.

    python3 run_tests.py
"""
from __future__ import annotations

from bps_kalgebra import BPSKAlgebra
from bps_atlas import BPSAtlas
from bps_quiver_tools import BPSQuiver


# --- SU(2) / A₁ class-S charts (mutation-finite; every chart has a reasonable S) -
# name -> (pairing, node_charges, spec).  BPS-chart literals (no implementations/).
_SU2 = {
    "pure_SU2": ([[0, 1], [-1, 0]], [(1, 0), (-1, 2)], [(1, 0), (-1, 2)]),
    "SU2_Nf1": ([[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
                [(1, 0, 0), (-1, 2, 0), (0, -1, 1)],
                [(0, -1, 1), (-1, 1, 1), (0, 1, 1), (1, 0, 0), (-1, 2, 0)]),
    "SU2_Nf2": ([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [(1, 0, 0, 0), (-1, 2, 0, 0), (0, -1, 1, 0), (0, -1, 0, 1)],
                [(0, -1, 1, 0), (-1, 1, 1, 0), (0, 1, 1, 0), (0, -1, 0, 1),
                 (-1, 1, 0, 1), (0, 1, 0, 1), (1, 0, 0, 0), (-1, 2, 0, 0)]),
    "SU2_Nf3": ([[0, 1, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                [(1, 0, 0, 0, 0), (-1, 2, 0, 0, 0), (0, -1, 1, 0, 0),
                 (0, -1, 0, 1, 0), (0, -1, 0, 0, 1)],
                [(0, -1, 1, 0, 0), (-1, 1, 1, 0, 0), (0, 1, 1, 0, 0),
                 (0, -1, 0, 1, 0), (-1, 1, 0, 1, 0), (0, 1, 0, 1, 0),
                 (0, -1, 0, 0, 1), (-1, 1, 0, 0, 1), (0, 1, 0, 0, 1),
                 (1, 0, 0, 0, 0), (-1, 2, 0, 0, 0)]),
    "SU2_Nf4": ([[0, 1, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [(1, 0, 0, 0, 0, 0), (-1, 2, 0, 0, 0, 0), (0, -1, 1, 0, 0, 0),
                 (0, -1, 0, 1, 0, 0), (0, -1, 0, 0, 1, 0), (0, -1, 0, 0, 0, 1)],
                [(0, -1, 1, 0, 0, 0), (-1, 1, 1, 0, 0, 0), (0, 1, 1, 0, 0, 0),
                 (0, -1, 0, 1, 0, 0), (-1, 1, 0, 1, 0, 0), (0, 1, 0, 1, 0, 0),
                 (0, -1, 0, 0, 1, 0), (-1, 1, 0, 0, 1, 0), (0, 1, 0, 0, 1, 0),
                 (0, -1, 0, 0, 0, 1), (-1, 1, 0, 0, 0, 1), (0, 1, 0, 0, 0, 1),
                 (1, 0, 0, 0, 0, 0), (-1, 2, 0, 0, 0, 0)]),
    "SU2_SU2": ([[0, 1, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 0, 0, 1, 0],
                 [0, 0, -1, 0, 0], [0, 0, 0, 0, 0]],
                [(1, 0, 0, 0, 0), (-1, 2, 0, 0, 0), (0, 0, 1, 0, 0),
                 (0, 0, -1, 2, 0), (0, -1, 0, -1, 1)],
                [(0, -1, 0, -1, 1), (0, -1, -1, 1, 1), (-1, 1, 0, -1, 1),
                 (0, -1, 0, 1, 1), (0, 1, 0, -1, 1), (-1, 1, -1, 1, 1),
                 (0, 1, -1, 1, 1), (-1, 1, 0, 1, 1), (0, 1, 0, 1, 1),
                 (1, 0, 0, 0, 0), (-1, 2, 0, 0, 0), (0, 0, 1, 0, 0),
                 (0, 0, -1, 2, 0)]),
    # SU(2)-gauged [A₁,Dₙ] — the Catalan chain (D3…D6 here; D7→429, D8→1430 close
    # too but are slower).  Linear-spec charts on the standard cocharacter basis.
    "SU2gA1D3": ([[0, 2, -1, 0], [-2, 0, 1, 0], [1, -1, 0, 1], [0, 0, -1, 0]],
                 [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
                 [(1, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 0), (0, 0, 1, 0),
                  (0, 0, 0, 1)]),
    "SU2gA1D4": ([[0, 2, -1, 0, 0], [-2, 0, 1, 0, 0], [1, -1, 0, 1, 0],
                  [0, 0, -1, 0, 1], [0, 0, 0, -1, 0]],
                 [(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0),
                  (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)],
                 [(1, 0, 0, 0, 0), (1, 0, 1, 0, 0), (0, 1, 0, 0, 0),
                  (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)]),
    "SU2gA1D5": ([[0, 2, -1, 0, 0, 0], [-2, 0, 1, 0, 0, 0], [1, -1, 0, 1, 0, 0],
                  [0, 0, -1, 0, 1, 0], [0, 0, 0, -1, 0, 1], [0, 0, 0, 0, -1, 0]],
                 [(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0),
                  (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1)],
                 [(1, 0, 0, 0, 0, 0), (1, 0, 1, 0, 0, 0), (0, 1, 0, 0, 0, 0),
                  (0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0),
                  (0, 0, 0, 0, 0, 1)]),
    "SU2gA1D6": ([[0, 2, -1, 0, 0, 0, 0], [-2, 0, 1, 0, 0, 0, 0],
                  [1, -1, 0, 1, 0, 0, 0], [0, 0, -1, 0, 1, 0, 0],
                  [0, 0, 0, -1, 0, 1, 0], [0, 0, 0, 0, -1, 0, 1],
                  [0, 0, 0, 0, 0, -1, 0]],
                 [(1, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0),
                  (0, 0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0),
                  (0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 1, 0),
                  (0, 0, 0, 0, 0, 0, 1)],
                 [(1, 0, 0, 0, 0, 0, 0), (1, 0, 1, 0, 0, 0, 0),
                  (0, 1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0),
                  (0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0),
                  (0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 1)]),
    # SU(2)³ linear quiver (rank 8) — closes to 138 charts.
    "SU2_cubed": ([[0, 2, 0, 0, 0, 0, -1, 0], [-2, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 2, 0, 0, -1, -1], [0, 0, -2, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 2, 0, -1], [0, 0, 0, 0, -2, 0, 0, 1],
                   [1, -1, 1, -1, 0, 0, 0, 0], [0, 0, 1, -1, 1, -1, 0, 0]],
                  [(1, 0, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0),
                   (0, 0, 1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0, 0),
                   (0, 0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0, 0),
                   (0, 0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 0, 1)],
                  None),
}

# The SU(2)-gauged [A₁,Dₙ] chart counts are the Catalan numbers Cₙ.
CATALAN_A1DN = {3: 5, 4: 14, 5: 42, 6: 132, 7: 429, 8: 1430}

SU2_NAMES = list(_SU2)


def su2_chart(name: str) -> BPSKAlgebra:
    """The BPS chart (`BPSKAlgebra`) of SU(2)-family example `name`."""
    P, N, S = _SU2[name]
    kw = {} if S is None else {"spec": [tuple(s) for s in S]}
    return BPSKAlgebra(pairing=[list(r) for r in P],
                       node_charges=[tuple(g) for g in N], **kw)


def su2_mutation_finite_counts(*, max_charts: int = 4000) -> list[dict]:
    """`mutation_complete` counts for every SU(2)-family example — all close
    (mutation-finite), and the SU(2)-gauged [A₁,Dₙ] members hit Catalan(n)."""
    out = []
    for name in SU2_NAMES:
        ch = su2_chart(name)
        mc = BPSAtlas(ch).mutation_complete(max_charts=max_charts)
        out.append({"theory": name, "rank": len(ch.lattice.pairing),
                    "n_charts": mc["n_charts"], "closed": mc["closed"],
                    "rank_regular": mc["rank_regular"]})
    return out


# --- SU(3): mutation-infinite, charts mostly WILD (no reasonable S) ------------
# Pure-SU(3) BPS quiver (Γ_sc coords): nodes·B·nodesᵀ is the Dirac pairing.
SU3_NODES = [(1, 0, 0, 0), (-1, 2, 0, -1), (0, 0, 1, 0), (0, -1, -1, 2)]
SU3_B = [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]
SU3_SPEC = [(1, 0, 0, 0), (0, 0, 1, 0), (-1, 2, 1, -1),
            (1, -1, -1, 2), (0, -1, -1, 2), (-1, 2, 0, -1)]


def su3_quiver() -> BPSQuiver:
    """The pure-SU(3) `BPSQuiver` (strong-coupling representative)."""
    return BPSQuiver.from_pairing([tuple(n) for n in SU3_NODES],
                                  [list(r) for r in SU3_B])


def su3_chart() -> BPSKAlgebra:
    """A pure-SU(3) `BPSKAlgebra` (for the `mutation_complete` runaway demo)."""
    return BPSKAlgebra(pairing=[list(r) for r in SU3_B],
                       node_charges=[tuple(n) for n in SU3_NODES],
                       spec=[tuple(s) for s in SU3_SPEC])


def su3_mutation_runaway(max_charts: int = 1000) -> dict:
    """`mutation_complete` on pure SU(3): hits the cap, `closed=False` — the
    quiver mutation orbit is infinite (the charts are not finite-spec)."""
    return BPSAtlas(su3_chart()).mutation_complete(max_charts=max_charts)


def su3_max_arrow(quiver: BPSQuiver) -> int:
    """Largest arrow multiplicity |⟨γ_i,γ_j⟩| in the quiver's exchange matrix."""
    ex = quiver.exchange
    n = len(ex)
    return max(abs(ex[i][j]) for i in range(n) for j in range(n))


def su3_arrow_growth(steps: int = 8) -> list[int]:
    """Greedy walk maximising the max arrow multiplicity — the doubly-exponential
    blow-up that certifies a non-finite (infinite) mutation type."""
    Q = su3_quiver()
    seq = [su3_max_arrow(Q)]
    for _ in range(steps):
        best = None
        for k in range(Q.n_nodes):
            Qk = Q.mutate(k)
            m = su3_max_arrow(Qk)
            if best is None or m > best[0]:
                best = (m, Qk)
        _, Q = best
        seq.append(su3_max_arrow(Q))
    return seq


def su3_chamber_census(max_depth: int = 2, spec_depth: int = 16) -> dict:
    """Census the chambers within `max_depth` raw quiver mutations, classifying
    each **finite-spec** (a finite negating sequence) or **wild** (none within
    `spec_depth`).  Returns `{n_chambers, n_finite, n_wild}` — wild chambers (no
    reasonable S) already appear at depth 2 for pure SU(3)."""
    root = su3_quiver()

    def key(Q):
        return tuple(sorted(tuple(c) for c in Q.charges))

    seen = {key(root): (root, 0)}
    frontier = [(root, 0)]
    while frontier:
        Q, d = frontier.pop()
        if d >= max_depth:
            continue
        for k in range(Q.n_nodes):
            Q2 = Q.mutate(k)
            kk = key(Q2)
            if kk not in seen:
                seen[kk] = (Q2, d + 1)
                frontier.append((Q2, d + 1))
    n_finite = n_wild = 0
    for Q, _ in seen.values():
        if Q.find_negating_sequence(max_depth=spec_depth) is not None:
            n_finite += 1
        else:
            n_wild += 1
    return {"n_chambers": len(seen), "n_finite": n_finite, "n_wild": n_wild}


def su3_finite_spec_atlas(spec_depth: int = 18, max_charts: int = 200) -> dict:
    """The SU(3) atlas restricted to **charts where S-finding works** — the
    finite-spec chambers (a finite negating sequence exists), folded by chart-iso,
    with the wild chambers as **walls** (user, 2026-06-28: "an SU(3) atlas
    including only charts where S-finding works").  The infinite, mostly-wild
    quiver mutation orbit collapses to a finite **2-chart** atlas (2 wild walls):
    the strong-coupling chamber (chart 0, all four mutations finite) and its image
    (chart 1, two finite mutations + two wild walls)."""
    B = [list(r) for r in SU3_B]

    def has_finite_S(charges):
        Q = BPSQuiver.from_pairing([tuple(c) for c in charges], B)
        return Q.find_negating_sequence(max_depth=spec_depth) is not None

    return BPSAtlas(su3_chart()).mutation_complete(max_charts=max_charts,
                                                   keep=has_finite_S)


def wild_chamber_nodes(path=(0, 1)) -> list:
    """The node charges of the wild chamber reached by the raw quiver mutation
    `path` from the strong-coupling representative — `find_negating_sequence`
    returns `None` there (no finite spec ⇒ no reasonable S)."""
    Q = su3_quiver()
    for k in path:
        Q = Q.mutate(int(k))
    return [tuple(c) for c in Q.charges]


def su3_recursive_S_drift(cutoffs=(2, 4)) -> list:
    """Run the **recursive direct-S finder** itself in the wild SU(3) chamber
    `(0,1)`: build the spec-free chart (`build_S`, principled σ) at increasing cone
    `cutoffs` and return `[(cutoff, multiply(n0,n1))]`.  The result **drifts
    without converging** (each cutoff injects new lower-order terms), so the
    recursive S is cutoff-dependent — **no reasonable S** in a wild chamber, the
    same verdict as the negating-sequence census but read off the S finder itself."""
    wild = wild_chamber_nodes((0, 1))
    B = [list(r) for r in SU3_B]
    out = []
    for cut in cutoffs:
        A = BPSKAlgebra(pairing=B, node_charges=wild, build_S=True,
                        spec_free_sigma="principled", build_S_cutoff=int(cut))
        out.append((cut, A.multiply(wild[0], wild[1])))
    return out


def demo() -> None:
    print("=== SU(2)/A₁ class-S: mutation-finite — every chart has a reasonable S ===")
    print(f"  {'theory':12s} {'rank':>4s} {'charts':>7s} {'closed':>6s} {'rr':>5s}")
    for r in su2_mutation_finite_counts():
        print(f"  {r['theory']:12s} {r['rank']:>4d} {r['n_charts']:>7d} "
              f"{str(r['closed']):>6s} {str(r['rank_regular']):>5s}")
    print(f"  → SU(2)-gauged [A₁,Dₙ] = Catalan(n): {CATALAN_A1DN}")

    print("\n=== SU(3): mutation-infinite — do the many charts have reasonable S's? ===")
    mc = su3_mutation_runaway(1000)
    print(f"  pure SU(3) mutation_complete(cap 1000): n_charts={mc['n_charts']} "
          f"closed={mc['closed']}  (runs away → infinite orbit)")
    print(f"  arrow multiplicity along a wild walk: {su3_arrow_growth(8)}")
    cen = su3_chamber_census(max_depth=2)
    print(f"  chamber census (depth≤2): {cen['n_chambers']} = "
          f"{cen['n_finite']} finite-spec + {cen['n_wild']} WILD (no S)")
    print(f"  a wild chamber's nodes (no finite spec): {wild_chamber_nodes((0, 1))}")
    print("  recursive direct-S finder in that wild chamber (does it find an S?):")
    drift = su3_recursive_S_drift((2, 4))
    for cut, m in drift:
        print(f"    cutoff {cut}: multiply(n0,n1) = {m}")
    print(f"    → cutoff-dependent ({'DRIFTS' if drift[0][1] != drift[1][1] else 'stable'}):"
          " the recursive S does NOT converge → no reasonable S.")
    print("  → the infinite SU(3) charts are mostly WILD: no finite spec, no reasonable")
    print("    S.  mutation_complete counts quiver-iso classes; for SU(3) that is wild")
    print("    combinatorics, not physical charts.")
    fa = su3_finite_spec_atlas()
    print(f"\n  restricting to charts where S-finding works (the finite-spec atlas):")
    print(f"    n_charts={fa['n_charts']} edges={len(fa['edges'])} walls={fa['walls']} "
          f"closed={fa['closed']}  → a finite {fa['n_charts']}-chart atlas (the tame core)")


if __name__ == "__main__":
    demo()
