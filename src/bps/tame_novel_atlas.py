"""`tame_novel_atlas` — `BPSKAlgebra` charts + `BPSAtlas` builders for the
**candidate-novel tame BPS quivers** found by the source repository's
S/spec-finder landscape programme (whose research notes catalogue them).

**What "tame" means here.**  A BPS quiver is *tame* when the tropical
half-monodromy σ (computed through a finite spec in the quiver's own frame)
has **zero algebraic entropy** on every node charge — σ² growth is finite
order or exact polynomial drift, never exponential.  Measured at K = 300 in
exact integer arithmetic, the tame/wild gap is cleanly bimodal (smallest
measured wild entropy 0.733, tames exactly 0), and every recognized physical
theory in the small-rank landscape is tame — so tameness is an extremely
selective *sieve* for candidate physical theories, while **not** a proof of
UV-completeness (the author's standing caveat).  Two further held-not-proven
assumptions transport verdicts across frames: tameness as a mutation-class
invariant (entropy invariance is proved; the *tame verdict* transport rests
on spec-choice independence, measured 57 chambers / 12 quivers all-agree on
the source branch but not proven) — see the design notes.

**What this module holds.**  The eleven verified-tame quivers that match
**no** known constructible theory (verification per entry: K=300 exact
σ-classification for the rank-5+ landscape/harvest members, the exact
quasi-polynomial / mutation-invariance certificates for r4-0121, the
σ²-profile measurements of for the flavour-web pair — see each
entry's `provenance`) (the landscape recognizer set + pure_ade
constructibles + the AD zoo + su2-gauged [A₁,Dₙ] + affine cycles + the
finite A/D/E shapes; independently re-checked against the repo's example
inventory at build time of this module — depth-2 bounded orbit
intersection, no hit).  They are *candidate new theories*; identification
(genuinely new vs a known theory in an exotic frame) is the author's call.
Each entry carries its BPS chart data as literals — pairing + a **verified
negating sequence** found by `BPSQuiver.find_negating_sequence` (each
re-verified at import via the `BPSKAlgebra` constructor, which replays the
sequence) — plus the structural metadata measured on the source branch.

The headline structure is the **node-addition tower** (`family='tower'`):
r4-0121 → r5-tame-06/07 → new0/new1 → r7-new0-ext, each rank-n member
deleting to a rank-(n−1) member, with the shared SNF signature
[1,…,1,2,2] (elementary divisors 2,2) and drift-dominated σ².  Beside it:
the all-drift/two-speed r5-tame-00, the rank-7 harvest tame r7-h1 and its
node-deletion r6-h1 (**all-finite** σ² — "finite-type-like" yet outside the
AD list), and the two SU(2)+N_f=4 flavour-edge objects (`family='su2nf4-web'`,
tentative physical reading: gauging relative U(1)'s inside the SO(8)
flavour symmetry — user identification pending).

Public API (mirrors `a1dn_atlas` / `bps_atlas_examples`):

    from tame_novel_atlas import (TAME_NAMES, tame_entry, tame_chart,
                                  tame_atlas)
    A  = tame_chart('r4_0121')         # BPSKAlgebra (spec replayed+verified)
    At = tame_atlas('r4_0121')         # BPSAtlas seeded at that chart
    e  = tame_entry('r4_0121')         # metadata (snf, corank, sigma, ...)

Charts are cached (`lru_cache`); building one is fast (the tames are the
"fast jobs": spec verified in well under a second, F supports small).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bps_kalgebra import BPSKAlgebra
from bps_atlas import BPSAtlas


__all__ = [
    "TAME_NAMES",
    "TameEntry",
    "tame_names",
    "tame_entry",
    "tame_chart",
    "tame_atlas",
]


# ---------------------------------------------------------------------------
# The catalogue: name -> chart literals + measured structural metadata.
#
# `pairing` is the Dirac matrix B in the frame carrying the verified finite
# chamber; `negseq` is the negating sequence (spec = its charge replay, done
# by the BPSKAlgebra constructor).  Node charges are always the standard
# basis.  Metadata fields are *measured on the source branch* (K=300 exact
# σ classification, bounded mutation-orbit BFS) and quoted here for
# orientation — the authoritative record is the experiment JSONs on
# a development branch / in the probes in the source repository.
# ---------------------------------------------------------------------------
_DATA = {
    "r4_0121": dict(
        label="r4-0121 (tower root, rank 4, mutation-invariant)",
        family="tower",
        pairing=[[0, -2, 1, 1], [2, 0, -1, -1], [-1, 1, 0, -1], [-1, 1, 1, 0]],
        negseq=[3, 1, 0, 2, 3, 1],
        snf=[1, 1, 2, 2], corank=0,
        sigma="uniform linear drift (-4,-4,-4,-4) on all 4 nodes",
        orbit="1 (COMPLETE — every mutation returns it up to perm x sign)",
        provenance="small_rank_landscape.py exhaustive rank-4 |b|<=2; "
                   "survived adjudication (r4-0119 = su2g[A1,D3]; this one did not match)",
        notes="deleting either non-Kronecker node leaves the SU(2)+Nf=1 cycle "
              "quiver — SU(2)+Nf=1 completed by a second dyonically-attached "
              "node into a mutation-invariant configuration",
    ),
    "r5_tame00": dict(
        label="r5-tame-00 (rank 5, all-drift two-speed, irreducible)",
        family="isolated",
        pairing=[[0, -2, -2, 1, 2], [2, 0, 0, -1, -1], [2, 0, 0, -1, 0],
                 [-1, 1, 1, 0, -1], [-2, 1, 0, 1, 0]],
        negseq=[2, 3, 4, 0, 1, 3, 4, 2, 1, 4, 3],
        snf=[1, 1, 1, 1], corank=1,
        sigma="two-speed drift: nodes {1,3,4} at -10*(1,1,1,1,1), nodes {0,2} "
              "at -5*(1,2,1,2,2)",
        orbit=">=1312 @ depth 6 (open)",
        provenance="rank5_landscape.py exhaustive rank-5 |b|<=2; workup "
                   "tame00_{structure,orbit,index,symmetry,nicest_frame} 2026-07-16",
        notes="SU(3)+Nf=1 excluded by disjoint depth-7 orbits AND differing "
              "exact vacuum indices at q^4; irreducible (no decoupled frame in "
              "1312 orbit members); zero physical deletions across 90 "
              "frame-node pairs",
    ),
    "r5_tame06": dict(
        label="r5-tame-06 (tower, rank 5, complete orbit 5)",
        family="tower",
        pairing=[[0, -2, 0, 1, 1], [2, 0, 0, -1, -1], [0, 0, 0, -1, -1],
                 [-1, 1, 1, 0, 0], [-1, 1, 1, 0, 0]],
        negseq=[3, 0, 1, 0, 3, 4, 3, 0, 2],
        snf=[1, 1, 2, 2], corank=1,
        sigma="uniform drift: nodes 0-2 at -4, nodes 3-4 at -2",
        orbit="5 (COMPLETE)",
        provenance="rank5_landscape.py; su2-gauged [A1,D4] definitively "
                   "excluded (orbit disjoint from the complete 5-key orbit)",
        notes="the r4-0121 pattern one rank up: same SNF [1,1,2,2], one U(1) "
              "flavour; deletions -> SU(2)+Nf=2 (x3) and su2g[A1,D3] (x2); "
              "U(1)-flavoured vacuum index with GROWING weights (+-1..+-5 "
              "through q^10, tame06_index.json)",
    ),
    "r5_tame07": dict(
        label="r5-tame-07 (rank 5, frame-recovered, one finite direction)",
        family="tower",
        # the spec-bearing frame = mu_3 of the recorded base B (the base frame
        # walls the auto finder — tame07_frame_recovered.json; the mu_3 recovery
        # reproduces the source branch's frame-retry finding)
        pairing=[[0, 0, -1, -1, 1], [0, 0, -1, 2, -1], [1, 1, 0, 1, -2],
                 [1, -2, -1, 0, 1], [-1, 1, 2, -1, 0]],
        negseq=[4, 3, 2, 0, 1, 4, 3, 2, 0, 3, 1],
        snf=[1, 1, 1, 1], corank=1,
        sigma="4 linear drifts + 1 FINITE ('matter-like') direction",
        orbit=">=1894 @ depth 6 (open); disjoint from r5-tame-00's",
        provenance="frame_retry.py r5: base frame walls the auto finder; "
                   "solved TAME in a mutated frame — the mutation-invariance "
                   "pipeline recovering a theory the base sweep missed",
        notes="kernel (2,2,1,3,3) in the base frame [[0,-2,-2,1,1],[2,0,-1,-2,1],"
              "[2,1,0,-1,-1],[-1,2,1,0,-1],[-1,-1,1,1,0]]: rank-2 Coulomb + 1 "
              "mass; the rank-5 member the rank-6 tower node new0 deletes to",
    ),
    "r6_new0": dict(
        label="new0 (tower, rank 6, corank 0)",
        family="tower",
        pairing=[[0, -1, 1, 1, -1, -2], [1, 0, 1, -1, 1, -1],
                 [-1, -1, 0, 1, -1, 1], [-1, 1, -1, 0, 1, 1],
                 [1, -1, 1, -1, 0, -1], [2, 1, -1, -1, 1, 0]],
        negseq=[2, 5, 1, 3, 2, 5, 0, 4, 1, 5, 2, 3, 4, 1],
        snf=[1, 1, 1, 1, 2, 2], corank=0,
        sigma="all-drift (drift-type / infinite-mutation-type)",
        orbit=">=708 @ depth 5 (open; visits |b|>=3 frames)",
        provenance="double-bond stratified sampling @ rank 6",
        notes="deletes to r5-tame-07 (node-deletion tower edge); no mass "
              "parameters (corank 0)",
    ),
    "r6_new1": dict(
        label="new1 (tower, rank 6, corank 0)",
        family="tower",
        pairing=[[0, 1, -1, 0, 1, -1], [-1, 0, 0, 2, -1, 0],
                 [1, 0, 0, -1, -1, 1], [0, -2, 1, 0, 1, -1],
                 [-1, 1, 1, -1, 0, 0], [1, 0, -1, 1, 0, 0]],
        negseq=[4, 5, 2, 3, 0, 1, 4, 3, 2, 5, 0, 3, 4],
        snf=[1, 1, 1, 1, 2, 2], corank=0,
        sigma="all-drift (drift-type / infinite-mutation-type)",
        orbit=">=633 @ depth 5 (open)",
        provenance="double-bond stratified sampling @ rank 6",
        notes="deletes to r5-tame-06",
    ),
    "r6_h1": dict(
        label="r6-h1 (rank 6, ALL-FINITE sigma — finite-type-like, corank 2)",
        family="harvest",
        pairing=[[0, -1, 0, -1, 1, 1], [1, 0, -1, 1, 0, -1],
                 [0, 1, 0, -1, 1, -1], [1, -1, 1, 0, -1, 0],
                 [-1, 0, -1, 1, 0, 1], [-1, 1, 1, 0, -1, 0]],
        negseq=[5, 0, 1, 5, 4, 2, 3, 4, 5, 1, 0, 5, 4, 3, 2, 4],
        snf=[1, 1, 2, 2], corank=2,
        sigma="FINITE sigma^2 order in ALL 6 directions (finite-type-like!)",
        orbit="~5 shallow (very rigid)",
        provenance="node-deletion from r7-h1 (novel_tame_harvest.py, branch "
                   "a development branch): deleting r7-h1's node 1 "
                   "kills all drift",
        notes="all-finite sigma with corank 2 and divisors (2,2) is NOT the "
              "finite-AD signature (those are corank-0) — if genuinely new, a "
              "'finite-type-like' theory outside the AD list; matches none of "
              "the 10 rank-6 knowns",
    ),
    "r6_flavA": dict(
        label="SU(2)+Nf=4 + one flavour-pair edge (rank 6, corank 2)",
        family="su2nf4-web",
        pairing=[[0, 2, -1, -1, -1, -1], [-2, 0, 1, 1, 1, 1],
                 [1, -1, 0, -1, 0, 0], [1, -1, 1, 0, 0, 0],
                 [1, -1, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0]],
        negseq=[3, 0, 1, 2, 3, 0, 4, 3, 2, 5, 4, 3],
        snf=[1, 1, 1, 1], corank=2,
        sigma="4 drift + 2 finite",
        orbit=">=52 @ bounded (open)",
        provenance="seeded modification hunt rank6_hunt.py: connect one "
                   "flavour pair of conformal SU(2)+Nf=4 by a single arrow",
        notes="matching selection rule: tame IFF the added flavour-flavour "
              "edges form a matching of the 4 flavour nodes; corank = 4 - "
              "2*(edges).  Tentative physical reading (unconfirmed): gauging a "
              "relative U(1) inside SO(8); deletes to SU(2)+Nf=3",
    ),
    "r6_flavB": dict(
        label="SU(2)+Nf=4 + two flavour-pair edges (rank 6, corank 0)",
        family="su2nf4-web",
        pairing=[[0, 2, -1, -1, -1, -1], [-2, 0, 1, 1, 1, 1],
                 [1, -1, 0, -1, 0, 0], [1, -1, 1, 0, 0, 0],
                 [1, -1, 0, 0, 0, -1], [1, -1, 0, 0, 1, 0]],
        negseq=[3, 1, 0, 5, 1, 2, 3, 4, 1, 5, 2, 4],
        snf=[1, 1, 1, 1, 2, 2], corank=0,
        sigma="all drift",
        orbit=">=61 @ bounded (open)",
        provenance="rank6_hunt.py: perfect matching (2 edges) on the "
                   "flavour nodes of SU(2)+Nf=4",
        notes="unrecognized deletions; the stage-2 member of the matching "
              "chain (corank 4 -> 2 -> 0); SNF picks up the (2,2) divisors",
    ),
    "r7_new0_ext": dict(
        label="r7-new0-ext (tower, rank 7, one finite spectator direction)",
        family="tower",
        pairing=[[0, -1, 1, 1, -1, -2, 0], [1, 0, 1, -1, 1, -1, -1],
                 [-1, -1, 0, 1, -1, 1, 0], [-1, 1, -1, 0, 1, 1, 0],
                 [1, -1, 1, -1, 0, -1, 1], [2, 1, -1, -1, 1, 0, 0],
                 [0, 1, 0, 0, -1, 0, 0]],
        negseq=[2, 5, 3, 6, 1, 2, 5, 0, 4, 1, 6, 5, 2, 4, 1, 3, 6, 4],
        snf=[1, 1, 1, 1, 2, 2], corank=1,
        sigma="6 linear drifts + 1 FINITE (node 6, the light added node)",
        orbit=">=1230 (open)",
        provenance="rank7_family_extend.py: the ONLY distinct tame in "
                   "a 26% sample of all 31248 |b|<=2 node-additions to "
                   "new0/new1 — family members are RARE at rank 7",
        notes="deletes to new0 by removing node 6; K=300-verified; rank 7 is "
              "odd so corank >= 1 is automatic",
    ),
    "r7_h1": dict(
        label="r7-h1 (rank 7, all-drift harvest tame)",
        family="harvest",
        pairing=[[0, 2, -1, 0, -1, 1, 1], [-2, 0, 1, -1, 0, 2, 0],
                 [1, -1, 0, -1, 1, 0, -1], [0, 1, 1, 0, -1, 1, -1],
                 [1, 0, -1, 1, 0, -1, 0], [-1, -2, 0, -1, 1, 0, 1],
                 [-1, 0, 1, 1, 0, -1, 0]],
        negseq=[6, 0, 2, 4, 3, 5, 6, 0, 2, 3, 1, 5, 2, 6, 1, 4, 0, 6, 5, 2, 3],
        snf=[1, 1, 1, 1, 2, 2], corank=1,
        sigma="linear drift in ALL 7 directions",
        orbit=">=207 (open)",
        provenance="novel_tame_harvest.py d=2 stratified census (1 tame in "
                   "12000 classes; branch a development branch)",
        notes="NOT a rung of the tower (no node-deletion lands in new0/new1 "
              "shallow neighborhoods — all 7 deletions checked); shares the "
              "family SNF but all-drift; its node-1 deletion IS the new tame "
              "r6-h1",
    ),
    "r6_970": dict(
        label=" (rank 6, corank 2, order-2 ρ; double-bond find)",
        family="isolated",
        pairing=[[0, 0, -1, -1, 0, 1], [0, 0, 1, -1, 1, -1],
                 [1, -1, 0, 0, -2, 1], [1, 1, 0, 0, 0, -1],
                 [0, -1, 2, 0, 0, -1], [-1, 1, -1, 1, 1, 0]],
        negseq=[5, 2, 3, 4, 1, 0, 5, 3, 2, 0, 1, 2, 3],
        snf=[1, 1, 1, 1], corank=2,
        sigma="σ² = 5 linear drifts + 1 FINITE; ρ carries an order-2 elliptic "
              "part (odd powers wobble, ρ² stabilizes to 5 drift directions)",
        orbit="open (mutation-infinite; >12000 with no closure, pm-relabel key)",
        provenance="double-bond stratified sampling @ rank 6, d=1 stratum "
                   "(the design notes; issue); own-frame spec "
                   "(len 13) found in <1s; studied 2026-07-18",
        notes="NOT in the SNF[1,1,..,2,2] tower (this is SNF[1,1,1,1], corank 2 "
              "= 2 masses) and NOT r6-h1 (different Dirac form); parabolic ⇒ "
              "not an SCFT, and no finite puncture-rotation order (ρ² keeps 5 "
              "drift directions) ⇒ not a single-irregular-puncture surface — a "
              "flavoured mystery cousin of new0/new1 with an extra order-2 ρ "
              "symmetry; vacuum index q² = 4 (refined 2 + μ₂^{±1}; only 1 of the "
              "2 masses visible at q²), much smaller flavour than r6-h1's 10",
    ),
}

@dataclass
class TameEntry:
    """One candidate-novel tame BPS quiver: chart literals + measured metadata."""
    name: str
    label: str
    family: str            # 'tower' | 'isolated' | 'harvest' | 'su2nf4-web'
    pairing: list
    negseq: list
    snf: list
    corank: int
    sigma: str
    orbit: str
    provenance: str
    notes: str = ""

    @property
    def rank(self) -> int:
        return len(self.pairing)


TAME_NAMES = tuple(_DATA)


def tame_names() -> list[str]:
    return list(TAME_NAMES)


def tame_entry(name: str) -> TameEntry:
    if name not in _DATA:
        raise KeyError(f"unknown tame candidate {name!r}; known: {TAME_NAMES}")
    d = _DATA[name]
    if d["pairing"] is None or d["negseq"] is None:
        raise ValueError(
            f"{name}: spec-bearing frame not yet recorded (base frame walls "
            "the auto finder; see module notes)")
    return TameEntry(name=name, **d)


@lru_cache(maxsize=None)
def tame_chart(name: str) -> BPSKAlgebra:
    """The `BPSKAlgebra` chart (std node charges; the negating sequence is
    replayed and verified by the constructor)."""
    e = tame_entry(name)
    n = e.rank
    std = [tuple(1 if j == i else 0 for j in range(n)) for i in range(n)]
    return BPSKAlgebra(pairing=e.pairing, node_charges=std,
                       negating_sequence=list(e.negseq))


def tame_atlas(name: str, **kw) -> BPSAtlas:
    """A `BPSAtlas` seeded at the tame chart.  For the closed-orbit members
    (r4_0121, r5_tame06) `mutation_complete()` closes; the drift-type
    rank-6/7 members are infinite-mutation-type — use bounded caps."""
    return BPSAtlas(tame_chart(name), **kw)


# ---------------------------------------------------------------------------
# RGKAlgebra presentations over recognized sub-quivers.
#
# Every tower rung is a node-drop flow, so the tames with a recognized (or
# chained-to-recognized) deletion admit an `RGKAlgebra` presentation whose
# IR auxiliary is the sub-quiver's algebra — bypassing the UV BPS F-solve
# entirely (the `DirectionalSubquiverRG` co-solver works IR-side).  Measured
# this session: RG images have support 1–2 where the UV BPS F-supports run
# 10–230 (or exceed the 9 GB memory guard outright), RG-derived multiply
# equals the BPS multiply wherever both exist, and the full RG axiom checks
# pass — including on r6_new1, whose BPS route is memory-walled.
#
# The recorded drop map (deletion targets verified by depth-≤3 orbit
# intersection this session / recorded on the source branch):
#
#   r4_0121      − node 2 → SU(2)+N_f=1   (cone-backed in su2_family)
#   r5_tame06    − node 2 → SU(2)+N_f=2   (full cone↔BPS iso exists)
#   r6_flavA     − node 2 → SU(2)+N_f=3   (cone-backed)
#   r6_new1      − node 1 → r5_tame06     (tower edge — chain to N_f=2)
#   r6_new0      − node 1 → r5_tame07     (tower edge; node 4 also works)
#   r7_new0_ext  − node 6 → r6_new0       (tower edge, PR)
#   r7_h1        − node 1 → r6_h1         (harvest edge)
#
# r5_tame00 has no physical deletion (all 90 frame-node pairs wild — its
# recorded distinguishing feature), r5_tame07's and r6_h1's own deletions
# are unrecognized, and r6_flavB's deletions are recorded unrecognized —
# no flow entry for those.
# ---------------------------------------------------------------------------
_FLOW_DROP = {
    "r4_0121": (2, "SU(2)+Nf=1"),
    "r5_tame06": (2, "SU(2)+Nf=2"),
    "r6_flavA": (2, "SU(2)+Nf=3"),
    "r6_new1": (1, "r5_tame06"),
    "r6_new0": (1, "r5_tame07"),
    "r7_new0_ext": (6, "r6_new0"),
    "r7_h1": (1, "r6_h1"),
}

# UVs whose BPS F-solve is memory-walled: use the pure co-solver (no UV
# delegation, no F-oracle) — the only working access route to their algebra.
_FLOW_DIRECTIONAL = {"r6_new0", "r6_new1", "r7_new0_ext", "r7_h1"}


def tame_flow_targets() -> dict:
    """name -> (dropped node index, IR description) for the entries with a
    recognized sub-quiver flow."""
    return dict(_FLOW_DROP)


@lru_cache(maxsize=None)
def tame_flow(name: str):
    """The node-drop `RGKAlgebra` presentation of a tame candidate over its
    recognized sub-quiver (see the drop map above).

    Light entries return `rg_flow.SubquiverRG` (UV kept as
    `starting_algebra()` + F-oracle cross-checks); the memory-walled
    drift entries return a bare `DirectionalSubquiverRG` (version-(c)
    ket-peel, certify before deep trust — the standing caveat of
    `require_stuff_first=False`)."""
    if name not in _FLOW_DROP:
        raise KeyError(
            f"{name} has no recognized sub-quiver flow; available: "
            f"{sorted(_FLOW_DROP)}")
    drop, _ir = _FLOW_DROP[name]
    if name in _FLOW_DIRECTIONAL:
        from directional_subquiver_rg import DirectionalSubquiverRG
        from bps_quiver_tools import BPSQuiver
        e = tame_entry(name)
        n = e.rank
        std = [tuple(1 if j == i else 0 for j in range(n)) for i in range(n)]
        spec = [tuple(g) for g in
                BPSQuiver.from_pairing(std, e.pairing)
                .build_spectrum_generator(list(e.negseq))]
        return DirectionalSubquiverRG(
            e.pairing, std, spec, [drop],
            require_stuff_first=False, ir_verify="off")
    from rg_flow import SubquiverRG
    return SubquiverRG(tame_chart(name), [drop])
