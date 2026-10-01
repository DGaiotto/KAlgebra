"""The mutation relation between spectrum generators.

**The conjecture**.
- Every quiver `Q` has a spectrum generator `S_Q`: the element of `𝓔` fixed by
  the BPS quiver axioms.  `spec_acceptance.crystalline_spectrum` builds it
  order by order on the cone, whether or not `Q` has a finite spec.
- The generators of two quivers related by a mutation at node `k` are related
  by moving the node-`k` factor across:

      S_{μ_k Q} = E_𝖖(X_{γ_k})⁻¹ · S_Q · E_𝖖(X_{−γ_k}),

  in the lattice identification of the green mutation (`BPSQuiver.mutate`:
  `γ_k ↦ −γ_k`, `γ_j ↦ γ_j + [b_jk]₊ γ_k`).  In words: strip an `E_𝖖` factor
  from one side, and add an oppositely charged one on the other.
- For a quiver with a negating sequence that starts at `k`, this is the
  sequence's rotation.  `S_Q = E(X_{γ_k}) E(X_{c_2}) ⋯ E(X_{c_L})` goes to
  `S_{μ_k Q} = E(X_{c_2}) ⋯ E(X_{c_L}) E(X_{−γ_k})`, the rotated sequence
  negating `μ_k Q`.  In general it is a conjecture, and it is what this module
  tests, above all on the quivers with no negating sequence
  (`crystalline_only`).

**Frames.**  Everything is computed in each quiver's OWN node coordinates, with
its exchange matrix as the pairing (`⟨e_i, e_j⟩ = B[i][j]`, the convention of
`recursive_spectrum.Theory`).  The green basis's pairing is exactly the
Fomin–Zelevinsky matrix `quiver_enumeration.mutate(B, k)`, so the mutated
quiver's own frame IS the green basis.  A charge's coordinates change by

    n'_j = n_j  (j ≠ k),     n'_k = Σ_{j≠k} [b_jk]₊ n_j − n_k,

an involution.  No relabelling or canonical form enters: `μ_k Q`'s `S` is
computed on `mutate(B, k)` as it stands.

**Exactness.**
- `S_Q` is known exactly on its cone to a depth `D`.  The relation fixes
  `S_{μ_k Q}` exactly on the charges whose dependencies all lie there.
  Nothing else is compared, and the verdict counts what was compared, depth by
  depth.  A depth is a truncation, never a certificate (as in
  `spec_acceptance`).
- The relation also says that `E_𝖖(X_{γ_k})⁻¹ S_Q` is supported in `μ_k Q`'s
  cone: no charge with a negative `k`-coordinate after the change.  A nonzero
  coefficient there is recorded as a SUPPORT violation.

**Paths.**  The author's extension: follow `Q_0 → Q_1 → … → Q_m`, whose
intermediate quivers may lie above the dictionary's weight cutoff.
- `mode="ends"` transports `S_{Q_0}` step by step and compares the end with
  `S_{Q_m}`.  The intermediate quivers need no `S` of their own.
- `mode="steps"` computes every quiver's `S` and checks each edge.  It is
  stronger, and costs one `S` per step.

**What a compared charge tests**.
- *Locality.*  The coefficient of `S_Q` at a charge `γ` depends only on the
  full subquiver on the nodes where `γ` is nonzero (`S_Q` restricted to a face
  is the subquiver's `S`).  The mutation at `k` commutes with that
  restriction.  So a compared charge `w` tests the relation on the subquiver
  on `supp(w) ∪ {k}`.
- *Reaching the whole quiver.*  It is reached only by charges nonzero on every
  node.  On the edge `Q → μ_k Q` that needs depth
  `(n − 1) + Σ_{j≠k} [b_jk]₊` (green; `[b_kj]₊` reverse):
  `depth_for_every_node`.
- *Automatic agreements.*  A compared charge agrees whatever `S` is when
  `k` is a source or a sink of the subquiver on `F = supp(w) ∪ {k}`, i.e. when
  `supp(w)` avoids `informative_nodes(B, k, "green")` (`[b_jk]₊ > 0`) or
  `informative_nodes(B, k, "reverse")` (`[b_kj]₊ > 0`).
  - One side is the peel of that source or sink.
  - The other is the identity `Ad_{(E(X_k) E(X_{−k}))⁻¹} = M` on the part of
    the torus pairing with `γ_k` of one sign (`M` the monomial shift of the
    change of frame).  This holds for any `S_{F∖k}`.
  - So only a charge meeting BOTH sets is informative (`is_informative`), in
    either direction.  Every multiple of `k`'s own charge also agrees, since
    its coefficient is `E_𝖖`'s.
  - The first version of this module counted only one side: it labelled the 3-Kronecker quiver at its sink and `A_2`
    `whole_quiver`, where every comparison is an identity.  Verdicts were
    never affected, only the counts.
- *Counts.*  `Comparison` therefore reports `n_informative`, `n_every_node`,
  `max_support` and `reach`:
  - `"whole_quiver"` when an informative charge is nonzero on every node;
  - `"smaller_quiver"` when informative charges exist but none reaches every
    node;
  - `"nothing_informative"` when none exists.
  For paths, where the automatic agreements are not analysed, a charge counts
  as reaching every node when `supp(w) ∪ {path nodes}` is all of them.

**Variants** (`VARIANTS`).
- `"green"` is the conjecture on the edge `Q → μ_k Q`.
- `"reverse"` is the same statement read from `μ_k Q`.  It identifies the
  lattices by `γ_j ↦ γ_j + [b_kj]₊ γ_k`, strips on the right and adds on the
  left.  It equals `"green"` on the edge `μ_k Q → Q`.
- The two `"mixed_*"` variants pair a side with the wrong identification.  They
  are NEGATIVE CONTROLS: on a quiver where the relation has content they must
  fail, which shows that the comparison has power.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from bps_factor_spectrum import cone_simplex
from habiro import HabiroElement
from quiver_enumeration import mutate as mutate_exchange
from recursive_spectrum import c_n, q_pow

H0 = HabiroElement.zero()
H1 = HabiroElement.one()

Vec = tuple[int, ...]
Matrix = tuple[tuple[int, ...], ...]
Spectrum = dict[Vec, HabiroElement]

#: name -> (the side the node-k factor is stripped from, the side its
#: opposite is added on, the lattice identification)
VARIANTS: dict[str, tuple[str, str, str]] = {
    "green": ("left", "right", "green"),
    "reverse": ("right", "left", "reverse"),
    "mixed_green": ("right", "left", "green"),        # negative control
    "mixed_reverse": ("left", "right", "reverse"),    # negative control
}


class NeedTooLarge(RuntimeError):
    """The charges a comparison needs outgrow `max_need`.  Long paths through
    heavy quivers do this; the caller records the item as undetermined."""


# ---------------------------------------------------------------------------
# E_𝖖(X) = Σ_m c_m X^m and its inverse, along one ray (⟨γ, γ⟩ = 0, so the
# powers of X_γ commute and the inverse is the plain power-series inverse)
# ---------------------------------------------------------------------------

_E_INV: list[HabiroElement] = [H1]


def e_coeff(m: int) -> HabiroElement:
    """The coefficient of `X^m` in `E_𝖖(X)` (`recursive_spectrum.c_n`)."""
    return c_n(m)


def e_inv_coeff(m: int) -> HabiroElement:
    """The coefficient of `X^m` in `E_𝖖(X)⁻¹`."""
    while len(_E_INV) <= m:
        j = len(_E_INV)
        _E_INV.append(-HabiroElement.sum([c_n(i) * _E_INV[j - i] for i in range(1, j + 1)]))
    return _E_INV[m]


def as_matrix(B: Sequence[Sequence[int]]) -> Matrix:
    return tuple(tuple(int(x) for x in row) for row in B)


def coordinate_change(B: Sequence[Sequence[int]], k: int, kind: str = "green"):
    """The map `n ↦ n'` of charge coordinates for the mutation of `B` at `k`
    (module doc), under the `"green"` or `"reverse"` identification.  An
    involution: the same map takes `n'` back to `n`."""
    n = len(B)
    if kind == "green":
        w = [max(0, int(B[j][k])) for j in range(n)]
    elif kind == "reverse":
        w = [max(0, int(B[k][j])) for j in range(n)]
    else:
        raise ValueError(f"unknown identification {kind!r}")
    w[k] = 0

    def phi(v: Vec) -> Vec:
        s = 0
        for wj, vj in zip(w, v):
            if wj:
                s += wj * vj
        return v[:k] + (s - v[k],) + v[k + 1:]

    return phi


def informative_nodes(B: Sequence[Sequence[int]], k: int, kind: str = "green") -> set[int]:
    """The nodes `j ≠ k` whose coordinate moves `k`'s under the change of
    frame: `[b_jk]₊ > 0` (green) or `[b_kj]₊ > 0` (reverse).  A compared charge
    whose support avoids all of them agrees automatically (module doc)."""
    n = len(B)
    if kind == "green":
        return {j for j in range(n) if j != k and int(B[j][k]) > 0}
    if kind == "reverse":
        return {j for j in range(n) if j != k and int(B[k][j]) > 0}
    raise ValueError(f"unknown identification {kind!r}")


def is_informative(B: Sequence[Sequence[int]], k: int, support) -> bool:
    """Whether a compared charge with this support can disagree at all: it
    must meet the nodes with `[b_jk]₊ > 0` AND those with `[b_kj]₊ > 0`, so that
    `k` is neither a source nor a sink of the subquiver on `support ∪ {k}`
    (module doc).  The same in both directions."""
    s = set(support)
    return bool(s & informative_nodes(B, k, "green")) and bool(s & informative_nodes(B, k, "reverse"))


def can_inform(B: Sequence[Sequence[int]], k: int) -> bool:
    """Whether any comparison on the edge `B → μ_k B` can be informative: `k`
    has arrows both ways in `B`.  False at a source or a sink, where the
    relation is the reflection identity."""
    return bool(informative_nodes(B, k, "green")) and bool(informative_nodes(B, k, "reverse"))


def compared_charges_edge(B: Sequence[Sequence[int]], k: int, depth: int, kind: str = "green") -> list[Vec]:
    """The charges (in `μ_k B`'s frame) that `compare_edge(B, k, depth)`
    compares, in closed form: `|w| ≤ depth` and
    `Σ_{j≠k} w_j (1 + c_j) ≤ depth`, with `c_j = [b_jk]₊` (green) or `[b_kj]₊`
    (reverse).  The deepest dependency of `w` in `B`'s frame has that depth.
    It depends on `(B, k, depth)` only, so the counts of a finished run can be
    recomputed without any `S` (`reach_counts`)."""
    n = len(B)
    if kind == "green":
        c = [max(0, int(B[j][k])) for j in range(n)]
    elif kind == "reverse":
        c = [max(0, int(B[k][j])) for j in range(n)]
    else:
        raise ValueError(f"unknown identification {kind!r}")
    c[k] = 0
    out = []
    for w in _simplex(n, depth):
        if sum(w[j] * (1 + c[j]) for j in range(n) if j != k) <= depth:
            out.append(w)
    return out


def reach_counts(B: Sequence[Sequence[int]], path: Sequence[int], compared) -> dict:
    """`n_informative`, `n_every_node`, `max_support` and `reach` over the
    compared charges of depth `≥ 2`.  For an edge, a charge counts only if it
    is informative (`is_informative`).  For a path the automatic agreements
    are not analysed (`n_informative` is None), and a charge reaches every node
    when `supp(w) ∪ {path nodes}` is all of them."""
    n = len(B)
    everything = set(range(n))
    on_path = set(path)
    edge = len(path) == 1
    n_inf = n_every = max_supp = 0
    for w in compared:
        if sum(w) < 2:
            continue
        supp = {j for j in range(n) if w[j]}
        if edge and not is_informative(B, path[0], supp):
            continue
        n_inf += 1
        reach_set = supp | on_path
        max_supp = max(max_supp, len(reach_set))
        if reach_set == everything:
            n_every += 1
    return {"n_informative": n_inf if edge else None, "n_every_node": n_every, "max_support": max_supp,
            "reach": "whole_quiver" if n_every else "smaller_quiver" if n_inf else "nothing_informative"}


def depth_for_every_node(B: Sequence[Sequence[int]], k: int, kind: str = "green") -> int:
    """The smallest depth at which the comparison on the edge `B → μ_k B`
    reaches a charge nonzero on every node:
    `(n − 1) + Σ_{j≠k} [b_jk]₊` (green; `[b_kj]₊` reverse).  The charge is
    `(1, …, 1)` off `k`, whose dependencies reach `k`-coordinate
    `Σ [b_jk]₊` in `B`'s frame."""
    n = len(B)
    if kind == "green":
        return (n - 1) + sum(max(0, int(B[j][k])) for j in range(n) if j != k)
    if kind == "reverse":
        return (n - 1) + sum(max(0, int(B[k][j])) for j in range(n) if j != k)
    raise ValueError(f"unknown identification {kind!r}")


def _with(v: Vec, k: int, x: int) -> Vec:
    return v[:k] + (x,) + v[k + 1:]


def _pairing_row(B: Matrix, k: int, v: Vec) -> int:
    """`⟨e_k, v⟩ = Σ_j B[k][j] v_j`."""
    s = 0
    for b, x in zip(B[k], v):
        if b and x:
            s += b * x
    return s


# ---------------------------------------------------------------------------
# One step: strip, change coordinates, add
# ---------------------------------------------------------------------------

def _strip(P: Spectrum, known: set, B: Matrix, k: int, side: str, targets) -> tuple[Spectrum, set]:
    """`R = E(X_{e_k})⁻¹ · P` (`side="left"`) or `P · E(X_{e_k})⁻¹` (`"right"`)
    at the `targets` whose dependencies `v − m e_k` (`0 ≤ m ≤ v_k`) are all
    `known`.  Returns `(R, the charges where R is known)`."""
    R: Spectrum = {}
    R_known: set = set()
    sign = 1 if side == "left" else -1
    for v in targets:
        vk = v[k]
        deps = [_with(v, k, vk - m) for m in range(vk + 1)]
        if any(d not in known for d in deps):
            continue
        pk = _pairing_row(B, k, v)
        terms = []
        for m, d in enumerate(deps):
            c = P.get(d)
            if c is not None:
                terms.append(e_inv_coeff(m) * q_pow(sign * m * pk) * c)
        R_known.add(v)
        s = HabiroElement.sum(terms) if terms else H0
        if not s.is_zero():
            R[v] = s
    return R, R_known


def _add(R: Spectrum, R_known: set, B_new: Matrix, k: int, side: str, phi, targets
         ) -> tuple[Spectrum, set]:
    """`P' = R · E(X_{e_k})` (`side="right"`) or `E(X_{e_k}) · R` (`"left"`) in
    the new frame, at the `targets` (new coordinates, in the new cone) whose
    dependencies are known.  `R` is keyed in the OLD frame; a dependency with
    a negative old `k`-coordinate lies outside the old cone, where `R` is
    exactly zero."""
    P: Spectrum = {}
    known: set = set()
    sign = 1 if side == "left" else -1
    for w in targets:
        wk = w[k]
        pk = _pairing_row(B_new, k, w)
        terms = []
        ok = True
        for m in range(wk + 1):
            u = phi(_with(w, k, wk - m))
            if u[k] < 0:
                continue
            if u not in R_known:
                ok = False
                break
            c = R.get(u)
            if c is not None:
                terms.append(c * e_coeff(m) * q_pow(sign * m * pk))
        if not ok:
            continue
        known.add(w)
        s = HabiroElement.sum(terms) if terms else H0
        if not s.is_zero():
            P[w] = s
    return P, known


def _simplex(n: int, depth: int) -> list[Vec]:
    return [tuple([0] * n)] + list(cone_simplex(n, depth))


def _need_sets(Bs: list[Matrix], path: Sequence[int], phis, depth_in: int, depth_out: int,
               max_need: int) -> tuple[list[set], list[set]]:
    """Backward pass: the charges of each frame that the final comparison
    region (the cone to `depth_out`) can depend on.  The first frame's set is
    cut to `depth_in`, since nothing deeper is known there."""
    n = len(Bs[0])
    m = len(path)
    need: list = [None] * (m + 1)
    need_r: list = [None] * m
    need[m] = set(_simplex(n, depth_out))
    for t in range(m - 1, -1, -1):
        k = path[t]
        phi = phis[t]
        nr = set()
        for w in need[t + 1]:
            for i in range(w[k] + 1):
                u = phi(_with(w, k, w[k] - i))
                if u[k] >= 0:
                    nr.add(u)
        nd = set()
        for v in nr:
            for j in range(v[k] + 1):
                nd.add(_with(v, k, v[k] - j))
        if t == 0:
            nd = {v for v in nd if sum(v) <= depth_in}
        if len(nr) > max_need or len(nd) > max_need:
            raise NeedTooLarge(f"step {t}: {len(nr)} / {len(nd)} charges needed (max {max_need})")
        need_r[t] = nr
        need[t] = nd
    return need, need_r


@dataclass
class Transported:
    """`S_{Q_0}` carried along a path: `B` is the end quiver (in the frame the
    path's mutations produce), `S` the transported coefficients, `known` the
    charges where they are exact (a charge in `known` and absent from `S` has
    coefficient zero)."""
    B: Matrix
    S: Spectrum
    known: set
    support_violations: list = field(default_factory=list)


def transport(B: Sequence[Sequence[int]], S: Spectrum, path: Sequence[int], depth_in: int,
              depth_out: int, *, variant: str = "green", check_support: bool = True,
              max_need: int = 400_000) -> Transported:
    """Carry `S` (the generator of `B` on its cone to `depth_in`) along
    `path`, one relation per step (`VARIANTS[variant]`), to the end quiver's
    cone to `depth_out`.

    `check_support` computes the stripped element of the FIRST step on every
    charge of `B`'s cone to `depth_in` that leaves the next cone, and records
    each nonzero coefficient as `(step, charge, coefficient)`.  Later steps
    work only on the charges the end needs, so their support is not
    swept."""
    strip_side, add_side, kind = VARIANTS[variant]
    Bs = [as_matrix(B)]
    for k in path:
        Bs.append(mutate_exchange(Bs[-1], k))
    phis = [coordinate_change(Bs[t], path[t], kind) for t in range(len(path))]
    need, need_r = _need_sets(Bs, path, phis, depth_in, depth_out, max_need)
    n = len(Bs[0])
    known = {v for v in need[0]}
    P = {v: S[v] for v in known if v in S and not S[v].is_zero()}
    if tuple([0] * n) in known and tuple([0] * n) not in P:
        P[tuple([0] * n)] = H1
    violations = []
    if check_support and path:
        k = path[0]
        sweep = [v for v in _simplex(n, depth_in) if phis[0](v)[k] < 0]
        full_known = set(_simplex(n, depth_in))
        full_P = {v: S[v] for v in full_known if v in S and not S[v].is_zero()}
        full_P.setdefault(tuple([0] * n), H1)
        R_out, _ = _strip(full_P, full_known, Bs[0], k, strip_side, sweep)
        violations = [(0, v, c) for v, c in sorted(R_out.items())]
    for t, k in enumerate(path):
        R, R_known = _strip(P, known, Bs[t], k, strip_side, need_r[t])
        P, known = _add(R, R_known, Bs[t + 1], k, add_side, phis[t], need[t + 1])
    return Transported(Bs[-1], P, known, violations)


# ---------------------------------------------------------------------------
# The generator of one quiver, and the comparison
# ---------------------------------------------------------------------------

def generator(B: Sequence[Sequence[int]], depth: int, *, spec=None) -> Spectrum:
    """`S` on `B`'s cone to `depth`: the product of `spec` when one is given
    (a certificate, e.g. from the dictionary, in `B`'s own node order), else
    the crystalline generator fixed by the axioms."""
    from spec_acceptance import crystalline_spectrum, spec_product
    if spec is not None:
        return spec_product(B, spec, depth)
    return crystalline_spectrum(B, depth)


@dataclass
class Comparison:
    """The verdict on one edge or path.  `ok` means every compared charge
    agreed and no support violation was seen; `compared_by_depth[d]` counts
    the charges compared at cone depth `d` of the END frame (depth 0 and 1
    are fixed by the axioms' leading data; the content is at `d ≥ 2`)."""
    ok: bool
    variant: str
    path: list
    depth_in: int
    depth_out: int
    compared_by_depth: dict
    mismatches: list
    support_violations: list
    seconds: float
    n_informative: int | None = None
    n_every_node: int = 0
    max_support: int = 0
    reach: str = "nothing_informative"

    @property
    def n_compared(self) -> int:
        return sum(self.compared_by_depth.values())

    @property
    def n_content(self) -> int:
        """Compared charges beyond the leading data (cone depth ≥ 2)."""
        return sum(c for d, c in self.compared_by_depth.items() if d >= 2)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "variant": self.variant, "path": list(self.path),
            "depth_in": self.depth_in, "depth_out": self.depth_out,
            "compared_by_depth": {str(d): c for d, c in sorted(self.compared_by_depth.items())},
            "n_compared": self.n_compared, "n_content": self.n_content,
            "mismatches": [[list(g), str(a.expand(12)), str(b.expand(12))]
                           for g, a, b in self.mismatches[:5]],
            "n_mismatches": len(self.mismatches),
            "support_violations": [[s, list(g), str(c.expand(12))]
                                   for s, g, c in self.support_violations[:5]],
            "n_support_violations": len(self.support_violations),
            "n_informative": self.n_informative, "n_every_node": self.n_every_node,
            "max_support": self.max_support, "reach": self.reach,
            "seconds": round(self.seconds, 3),
        }


def compare_ends(B: Sequence[Sequence[int]], path: Sequence[int], depth_in: int,
                 depth_out: int, *, variant: str = "green", S_start: Spectrum | None = None,
                 S_end: Spectrum | None = None, spec_start=None, spec_end=None,
                 check_support: bool = True, max_need: int = 400_000) -> Comparison:
    """The relation along `path` from `B`, comparing only the two ends.

    `S_start` / `S_end` may be passed (already computed, in the frames of `B`
    and of the path's end quiver `mutate(…mutate(B, k_1)…, k_m)`); otherwise
    they are built by `generator` (from `spec_start` / `spec_end` when given,
    each in its own quiver's node order).  An edge is a path of length 1."""
    t0 = time.time()
    B = as_matrix(B)
    if S_start is None:
        S_start = generator(B, depth_in, spec=spec_start)
    tr = transport(B, S_start, path, depth_in, depth_out, variant=variant,
                   check_support=check_support, max_need=max_need)
    if S_end is None:
        S_end = generator(tr.B, depth_out, spec=spec_end)
    by_depth: dict[int, int] = {}
    mism = []
    compared = []
    for w in sorted(tr.known, key=lambda v: (sum(v), v)):
        d = sum(w)
        if d > depth_out:
            continue
        compared.append(w)
        want = S_end.get(w, H1 if d == 0 else H0)
        got = tr.S.get(w, H0)
        by_depth[d] = by_depth.get(d, 0) + 1
        if got != want:
            mism.append((w, got, want))
    rc = reach_counts(B, path, compared)
    ok = not mism and not tr.support_violations
    return Comparison(ok, variant, list(path), depth_in, depth_out, by_depth, mism,
                      tr.support_violations, time.time() - t0,
                      n_informative=rc["n_informative"], n_every_node=rc["n_every_node"],
                      max_support=rc["max_support"], reach=rc["reach"])


def compare_edge(B: Sequence[Sequence[int]], k: int, depth: int, **kw) -> Comparison:
    """The relation on the edge `B → mutate(B, k)`, both ends to `depth`."""
    return compare_ends(B, [k], depth, depth, **kw)


def compare_steps(B: Sequence[Sequence[int]], path: Sequence[int], depth: int, *,
                  variant: str = "green", check_support: bool = True) -> list[Comparison]:
    """`mode="steps"`: every quiver on the path gets its own generator (to
    `depth`, built once and reused by the two edges that meet there) and each
    edge is compared."""
    B = as_matrix(B)
    Bs = [B]
    for k in path:
        Bs.append(mutate_exchange(Bs[-1], k))
    gens = [generator(b, depth) for b in Bs]
    return [compare_ends(Bs[t], [k], depth, depth, variant=variant, S_start=gens[t],
                         S_end=gens[t + 1], check_support=check_support)
            for t, k in enumerate(path)]


def path_matrices(B: Sequence[Sequence[int]], path: Sequence[int]) -> list[Matrix]:
    """`[B, μ_{k_1} B, μ_{k_2} μ_{k_1} B, …]`, in the frames the path produces."""
    out = [as_matrix(B)]
    for k in path:
        out.append(mutate_exchange(out[-1], k))
    return out
