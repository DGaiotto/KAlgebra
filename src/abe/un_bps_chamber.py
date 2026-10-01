"""Shared N=2 chamber map between lower-Kapustin `(m, λ)` labels and BPS
charges `γ`, used to wire the **abe ↔ bps** `KAlgebraIso` of both
`pure_u2_object` and `u2_nf1_object`.

A `BPSKAlgebra` is defined from a pairing + node charges, so the
**flavour-free** U(2) / U(2)+N_f=1 BPS realisations are built by *removing
the flavour (ker B) direction from the lattice and node charges* — for
these theories that direction is the gauge-centre U(1) (the genuine
flavour is `SU(N_f)`), not a real flavour, so dropping it gives a
non-degenerate pairing and `coefficient_ring = TrivialZPlusRing` directly.
The gauge pairing is the same rank-4 lattice in both cases, hence one shared
chamber map.
"""
from __future__ import annotations


def gamma_u2(m, e):
    """Lower-Kapustin label `(m, λ)` → BPS canonical-chamber charge `γ`
    (interleaved `(m1,e1,m2,e2)`): sort the two `(m_i, e_i)` pairs by `m`
    decreasing, `e` increasing (`UN_Nf.gamma_of_label`'s rule for N=2)."""
    p = sorted([(m[0], e[0]), (m[1], e[1])], key=lambda x: (-x[0], x[1]))
    return (p[0][0], p[0][1], p[1][0], p[1][1])


def label_u2(g):
    """BPS charge `γ` → lower-Kapustin label `(m, λ)`, Levi-dominance folded
    (at central magnetic `m1 = m2`, take `λ1 ≥ λ2`)."""
    p = sorted([(g[0], g[1]), (g[2], g[3])], key=lambda x: (-x[0], x[1]))
    m, e = (p[0][0], p[1][0]), (p[0][1], p[1][1])
    if m[0] == m[1] and e[0] < e[1]:
        e = (e[1], e[0])
    return (m, e)


def _u2_spread_guard(m, e):
    """Scope of the measured crossed-dyon fold (2026-07-02 session): with
    `d = λ(max-m slot) − λ(min-m slot)`, the fold below is CERTIFIED for
    `d ≤ 1` at m-spread ≤ 1, and `d ≤ 0` at m-spread ≥ 2 (m1 == m2 always
    fine).  Beyond it the doubly-tropical map redistributes BOTH e and m
    (measured: `((1,0),(2,0)) ↦ (0,1|1,1)`, `((2,0),(1,0)) ↦ (1,1|1,0)`) —
    the general closed form is the recorded follow-up."""
    if m[0] == m[1]:
        return True
    hi, lo = (0, 1) if m[0] > m[1] else (1, 0)
    d = e[hi] - e[lo]
    return d <= (1 if abs(m[0] - m[1]) <= 1 else 0)


def gamma_u2_crossed(m, e):
    """Lower-Kapustin `(m, λ)` → BPS tropical charge — the measured
    **crossed-dyon fold** (extends `gamma_u2` off the clean sector): order
    the slots by `λ` ascending (ties by slot), assign the descending-sorted
    `m`'s onto that order with `λ` kept in place, then sort `λ` ascending
    across equal-assigned-m slots.  Certified by product-coefficient
    matching against the abe keystone (the suite in the source repository);
    honest-fails outside `_u2_spread_guard`."""
    if not _u2_spread_guard(m, e):
        raise NotImplementedError(
            f"gamma_u2_crossed: ({m},{e}) beyond the certified fold scope "
            f"(doubly-tropical m/e redistribution — see _u2_spread_guard)")
    order = sorted((0, 1), key=lambda i: (e[i], i))
    ms = sorted(m, reverse=True)
    g = [0, 0, 0, 0]
    for rank, slot in enumerate(order):
        g[2 * slot] = ms[rank]
        g[2 * slot + 1] = e[slot]
    if g[0] == g[2] and g[1] > g[3]:
        g[1], g[3] = g[3], g[1]
    return tuple(g)


class UNTropicalMap:
    """The **N-general** anchored + rigidified U(N) chamber map (extends the
    U(2) original "tropical maps are not
    linear … mult and rho rigidify the lot").  BPS coordinates: interleaved
    `(m_1, e_1, …, m_N, e_N)`.

    Anchors: Wilson `L_{0,λ} ↦` zero magnetic + `λ` sorted ASCENDING in the
    e-slots (antidominant electric / lowest weight); bare monopole
    `L_{m,0} ↦` `m` sorted DESCENDING in the m-slots + zero electric
    (dominant magnetic); central `m` (all equal) keeps `m` with `λ`
    ascending.  Everything else derived: σ-transport first (ρ-images of
    anchored/cached labels — the SU(3) Witten-drift lesson), else exact
    structure-constant matching on `L_{m,0}·L_{0,λ↓}`; cached; honest-fail
    on coefficient collision."""

    def __init__(self, abe, bps, N=2):
        self._A, self._B = abe, bps
        self._N = int(N)
        self._g: dict = {}
        self._inv: dict = {}

    def _learn(self, label, g):
        self._g[label] = g
        self._inv[g] = label

    def _interleave(self, ms, es):
        g = []
        for i in range(self._N):
            g.extend((ms[i], es[i]))
        return tuple(g)

    def gamma(self, label):
        N = self._N
        m, e = tuple(label[0]), tuple(label[1])
        hit = self._g.get((m, e))
        if hit is not None:
            return hit
        zero = (0,) * N
        if e == zero or len(set(m)) == 1:        # bare monopole / central+Wilson
            if e == zero:
                g = self._interleave(sorted(m, reverse=True), zero)
            else:
                g = self._interleave(m, sorted(e))
            self._learn((m, e), g)
            return g
        if m == zero:
            g = self._interleave(zero, sorted(e))
            self._learn((m, e), g)
            return g

        # σ-transport first (ρ-images of anchored/cached labels)
        def _cheap(s):
            return (s in self._g or s[0] == zero or s[1] == zero
                    or len(set(s[0])) == 1)
        for step, b_step in ((self._A.rho_inverse, self._B.rho),
                             (self._A.rho, self._B.rho_inverse)):
            try:
                src = step((m, e))
            except NotImplementedError:
                continue
            src = (tuple(src[0]), tuple(src[1]))
            if _cheap(src):
                g = b_step(self.gamma(src))
                self._learn((m, e), g)
                return g

        # derived: match L_{m,0}·L_{0,e↓} against the BPS anchor product
        a = (m, zero)
        b = (zero, tuple(sorted(e, reverse=True)))
        ma = dict(self._A.multiply(a, b).terms)
        gb = dict(self._B.multiply(self.gamma(a), self.gamma(b)).terms)
        by_c = {}
        for g, c in gb.items():
            by_c.setdefault(str(c), []).append(g)
        for lab, c in sorted(ma.items()):
            lab = (tuple(lab[0]), tuple(lab[1]))
            bucket = by_c.get(str(c), [])
            known = self._g.get(lab)
            if known is not None:
                if known in bucket:
                    bucket.remove(known)
                continue
            if len(bucket) != 1:
                raise NotImplementedError(
                    f"UNTropicalMap: coefficient collision deriving {lab} "
                    f"from {a}·{b} (bucket {bucket})")
            self._learn(lab, bucket.pop(0))
        hit = self._g.get((m, e))
        if hit is None:
            raise NotImplementedError(
                f"UNTropicalMap: {label} absent from its anchor product")
        return hit

    def label(self, g):
        g = tuple(g)
        hit = self._inv.get(g)
        if hit is not None:
            return hit
        N = self._N
        from itertools import permutations
        ms = tuple(sorted((g[2 * i] for i in range(N)), reverse=True))
        es = [g[2 * i + 1] for i in range(N)]
        seen = set()
        for perm in permutations(es):
            e = tuple(perm)
            # Levi fold: λ non-increasing within equal-m blocks
            e = _levi_fold(ms, e)
            if (ms, e) in seen:
                continue
            seen.add((ms, e))
            try:
                if self.gamma((ms, e)) == g:
                    return (ms, e)
            except NotImplementedError:
                continue
        raise NotImplementedError(f"UNTropicalMap: no label derived for {g}")


def _levi_fold(m, e):
    """Sort `e` non-increasing within each equal-`m` block."""
    e = list(e)
    i = 0
    n = len(m)
    while i < n:
        j = i
        while j + 1 < n and m[j + 1] == m[i]:
            j += 1
        e[i:j + 1] = sorted(e[i:j + 1], reverse=True)
        i = j + 1
    return tuple(e)


class U2TropicalMap(UNTropicalMap):
    """The rigidified U(2) chamber map — the N=2 instance of
    `UNTropicalMap` (kept for the certified callers; see the general class
    for the anchor + mult/ρ rigidification architecture).  Subsumes the
    guarded `gamma_u2_crossed` fold — the doubly-tropical cases derive
    automatically."""

    def __init__(self, abe, bps):
        super().__init__(abe, bps, N=2)



def label_u2_crossed(g):
    """Inverse of `gamma_u2_crossed`: `m` = the γ magnetics sorted
    descending; `λ` = the γ electrics arranged so the forward map
    reproduces `γ` (≤ 2 candidates), Levi-folded at central `m`."""
    ms = tuple(sorted((g[0], g[2]), reverse=True))
    for e in {(g[1], g[3]), (g[3], g[1])}:
        if ms[0] == ms[1] and e[0] < e[1]:
            e = (e[1], e[0])
        try:
            if gamma_u2_crossed(ms, e) == tuple(g):
                return (ms, e)
        except NotImplementedError:
            continue
    raise NotImplementedError(
        f"label_u2_crossed: {g} not in the certified fold scope")
