"""
su2_nf1_cone_data.py
====================

`SU2Nf1ConeData(ConeData)` — H-tower cone data for `A_𝖖[SU(2) + N_f = 1]`,
mirroring `pure_su2_h_cone_data.PureSU2HConeData` but with **U(1)_F
flavour decoration**.

Mult-gens, cones, ρ-action
--------------------------
The gauge skeleton: mult-gens are the H-tower `H_n` for `n ∈ ℤ` plus
the Wilson fundamental `w_1`, with rank-3 cones tiled at **period 3**
(reflecting the ρ-shift of −3 in this theory):

    C_k = {H_{3k}, H_{3k+1}, H_{3k+2}}     (k ∈ ℤ)

and the Wilson cone (rank-1) `{w_1}`.  ρ shifts the H index by **−3**
(period-3 cone tiling: `ρ(C_k) = C_{k−1}`); ρ is trivial on the
Wilson cone.

Compare pure SU(2): rank-3 cones at period 2, `ρ(H_n) = H_{n−4}`,
`ρ(C_k) = C_{k−2}`.

The U(1)_F flavor sector enters at the **cone-monomial level**: each
canonical-basis cone monomial picks up a `μ^k` prefactor depending on
its H-content (and possibly on Wilson content too).  Concretely, the
canonical basis label for an H-monomial in cone `C_k` is

    L_native = μ^{ψ(native)} · q^{phase(native)} · (literal H-product)

with ψ a linear/bilinear functional on the (n, exp) content.  The
μ-factor is hidden in the cone-data primitives via the
`coefficient_ring = AbelianZPlusRing(rank=1)` widening:

* `q_commute(g, h)` and `cocycle(g, h)`: same integer values as pure
  SU(2) (gauge DSZ pairing is unchanged; flavour is spectator on the
  gauge sector).
* `cross_product(g, h)` returns `RLaurent[R]`-coefficient × literal-word
  pairs, with R = `AbelianZPlusRing(rank=1)`.  The μ-decorations sit in
  the RLaurent coefficients.

This module sets up the gauge skeleton and leaves the μ-decorated
ray-product table to `su2_nf1_h_gap_k.h_mul_h` (the W_1-walk cyclicity
recursion) bootstrapped from a few BPS-determined base cases.

References
----------
* `pure_su2_h_cone_data` — the unflavoured template.
* `bps_su2_nf1.build_bps_su2_nf1` — auxiliary BPSKAlgebra for one-time
  base-case extraction.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from typing import Sequence

from cone_data import ConeData, CrossProductTerm
from laurent_poly import LaurentPoly
from zplus_ring import AbelianZPlusRing, RLaurent
from pure_su2_h_wilson import is_wilson_label, WILSON_FUND


# ---------------------------------------------------------------------------
# Cone identification (verbatim from PureSU2HConeData — gauge skeleton)
# ---------------------------------------------------------------------------

def cone_index_for(n_lo: int, n_hi: int):
    """Cone index for the Nf=1 rank-2 cone family `C_a = {H_a, H_{a+1}}`.

    Returns `n_lo` (= a) if `n_hi − n_lo ≤ 1`, else None.

    Empirical fact from BPS probes: H_a and H_b q-commute (single
    q-monomial product) iff |b−a| ≤ 1.  Pair-cones tile the H-tower
    with period 1 (overlapping by 1 letter).  ρ shifts cone index by
    −3 (matching `ρ(H_n) = μ^{−1} · H_{n−3}`).
    """
    if n_hi - n_lo > 1:
        return None
    return n_lo


# ---------------------------------------------------------------------------
# H-tower cone data (gauge skeleton + μ-flavour R-coefficient ring)
# ---------------------------------------------------------------------------

class SU2Nf1ConeData(ConeData):
    """SU(2)+Nf=1 cone data: pure-SU(2) H-tower skeleton with U(1)_F
    flavour widening via `AbelianZPlusRing(rank=1)` coefficient ring.

    Mult-gens: `H_n = (n,)` for n ∈ ℤ, plus `w_1 = ('W', 1)`.
    Cones: `C_k = {H_{2k}, H_{2k+1}, H_{2k+2}}` (rank 3) + Wilson cone
    `{w_1}` (rank 1).  ρ acts as `H_n ↦ H_{n−4}`, w_1 ↦ w_1.

    The μ-flavour decoration is invisible to `q_commute` / `cocycle`
    (those are integer-valued from gauge DSZ alone) but lives in the
    `RLaurent[R]` coefficients of `cross_product` outputs and in the
    canonical-basis label encoding (`_label_section_decompose` peels
    the μ-power).
    """

    def __init__(self) -> None:
        self._R = AbelianZPlusRing(rank=1)
        self._gens: dict[int, tuple] = {}
        self._cones: dict[int, frozenset] = {}
        self._wilson_cone: frozenset = frozenset({WILSON_FUND})

    # -- R-widening hook -----------------------------------------------

    def coefficient_ring(self):
        return self._R

    # -- on-demand materialisation -------------------------------------

    def _gen(self, n: int) -> tuple:
        if n not in self._gens:
            self._gens[n] = (n,)
        return self._gens[n]

    def _cone(self, k: int) -> frozenset:
        """Cone `C_k = {H_{3k}, H_{3k+1}, H_{3k+2}}`."""
        if k not in self._cones:
            self._cones[k] = frozenset({
                self._gen(3 * k), self._gen(3 * k + 1), self._gen(3 * k + 2),
            })
        return self._cones[k]

    def seen_gens(self) -> Sequence[tuple]:
        return tuple(sorted(self._gens.values()))

    def seen_cones(self) -> Sequence[frozenset]:
        return tuple(self._cones[k] for k in sorted(self._cones))

    # -- iter_cones (lazy unbounded) -----------------------------------

    def iter_cones(self):
        """Bilateral enumeration C_0, C_1, C_{-1}, C_2, …, plus Wilson cone."""
        yield self._wilson_cone
        yield self._cone(0)
        k = 1
        while True:
            yield self._cone(k)
            yield self._cone(-k)
            k += 1

    # -- q_commute / cocycle (gauge skeleton, μ-blind) -----------------

    def q_commute(self, g, h) -> bool:
        if g == h:
            return True
        g_is_w = is_wilson_label(g)
        h_is_w = is_wilson_label(h)
        if g_is_w and h_is_w:
            return True                          # Wilson-Wilson: cocycle 0.
        if g_is_w or h_is_w:
            return False                         # Wilson-H: never commute.
        a, b = g[0], h[0]
        n_lo, n_hi = (a, b) if a < b else (b, a)
        return cone_index_for(n_lo, n_hi) is not None

    def cocycle(self, g, h) -> int:
        if g == h:
            return 0
        if is_wilson_label(g) and is_wilson_label(h):
            return 0
        if not self.q_commute(g, h):
            raise ValueError(f"cocycle: ({g}, {h}) not q-commuting")
        return h[0] - g[0]

    # -- cross_product (delegates to su2_nf1 ray-product table) ---------

    def cross_product(self, g, h) -> Sequence[CrossProductTerm]:
        """Plücker substitution for non-q-commuting pairs.

        Delegates to `su2_nf1_h_multiply._ray_product` (when wired) —
        the U(1)_F-decorated counterpart of pure SU(2)'s ray-product
        table.  Returns `[(RLaurent[R] coef, literal word), …]`.
        """
        from su2_nf1_h_multiply import _ray_product, WILSON_FUND as _WF

        def _to_letter(x):
            if isinstance(x, tuple) and x and x[0] == 'W':
                return _WF
            return ('H', x[0])

        def _from_word(word):
            out = []
            for letter in word:
                if letter[0] == 'H':
                    out.append(self._gen(letter[1]))
                else:
                    out.append(_WF)
            return tuple(out)

        g_l = _to_letter(g)
        h_l = _to_letter(h)
        substitutions = _ray_product(g_l, h_l)
        return [(coef, _from_word(word)) for coef, word in substitutions]

    # -- canonical cone order ------------------------------------------

    def canonical_cone_order(self, gens):
        """Sort by H-index (ascending integer n); Wilson last."""
        return tuple(sorted(gens, key=lambda g: (1 if is_wilson_label(g) else 0,
                                                  g if not is_wilson_label(g) else (0,))))

    # -- cone-label bijection (max-diagonal convention, μ-power tag) ----

    def to_cone_label(self, native_label):
        """Native = `(h_factors, μ_power)` where `h_factors` is a sorted
        tuple of `(n, exp)` (H-monomial) or `((W, e), 1)` (Wilson χ_e)."""
        h_factors, mu_p = native_label
        if not h_factors:
            cone = self._cone(0)
            return cone, {g: 0 for g in cone}
        first = h_factors[0]
        if isinstance(first[0], tuple) and first[0][0] == 'W':
            cone = self._wilson_cone
            return cone, {WILSON_FUND: 1}
        ns = sorted({n for (n, exp) in h_factors if exp > 0})
        n_lo, n_hi = ns[0], ns[-1]
        k = cone_index_for(n_lo, n_hi)
        if k is None:
            raise ValueError(
                f"to_cone_label: H-indices {ns} don't fit a single cone"
            )
        cone = self._cone(k)
        powers = {g: 0 for g in cone}
        for (n, exp) in h_factors:
            powers[(n,)] = powers.get((n,), 0) + exp
        return cone, powers

    def from_cone_label(self, gens, powers):
        """Inverse: (gens, powers) → `(h_factors, 0)` native label.

        μ-power is computed elsewhere (in the algebra `_label_section_
        decompose` or in the cross_product RLaurent coefficient); this
        function returns the pure cone-monomial gauge content.
        """
        factors = sorted(((g[0] if not is_wilson_label(g) else g), p)
                         for g, p in powers.items() if p > 0)
        return (tuple(factors), 0)


# ---------------------------------------------------------------------------
# Convenience: native <-> pSU2-style (m, e, mu) label conversion
# ---------------------------------------------------------------------------

def _native_to_psu2nf1(native: tuple) -> tuple:
    """Native label → pSU2nf1-style `(m, e, μ_power)` triple.

    H-monomial native: `m = Σ exp`, `e = Σ n·exp`, μ_power = explicit.
    Wilson native: `m = 0`, `e = wilson_e`, μ_power = explicit.
    """
    h_factors, mu_p = native
    m, e = 0, 0
    for entry in h_factors:
        gen, exp = entry
        if isinstance(gen, int):
            m += exp
            e += gen * exp
        elif isinstance(gen, tuple) and gen[0] == 'W':
            assert exp == 1, f"Wilson with exp != 1: {entry}"
            e += gen[1]
        else:
            raise ValueError(f"_native_to_psu2nf1: unrecognised entry {entry}")
    return (m, e, int(mu_p))


def _psu2nf1_to_native(m: int, e: int, mu_power: int = 0) -> tuple:
    """Inverse: pSU2nf1 `(m, e, μ_power)` → native, the **balanced
    H-monomial** (total over **all** (m, e), not just the period-3
    fundamental domain).

    For `m ≥ 1` the canonical native is `H_base^{m−r}·H_{base+1}^{r}` with
    `base, r = divmod(e, m)` (Python floor/mod — correct for negative `e`):
    `e` split into `m` as-equal-as-possible **consecutive** H-indices.  This
    is bubbling-free (an adjacent block of H's is already canonical), and it
    coincides exactly with the former period-3 max-diagonal form on its
    domain while also covering the period-3 **gaps** (`e_rel ∈ (2m, 3m)`)
    the max-diagonal missed — e.g. `(2,−1) = H₋₁·H₀`, `(2,5) = H₂·H₃`,
    `(3,7) = H₂²·H₃` — which ARE the cone's genuine canonicals there
    (verified `== cone.multiply` of the same adjacent block, pure `q`-power,
    no lower terms).  `m = 0` is the Wilson line.

    Totality matters: abe (Weyl-folded labels) and the flows produce these
    gap labels as ordinary canonicals (`H₋₁·H₀ = q·L_{(2,−1)}`), so the
    abe→cone leg of any witness needs the inverse defined everywhere."""
    if m == 0:
        return (() if e == 0 else ((('W', e), 1),), mu_power)
    base, rem = divmod(e, m)
    entries = []
    if m - rem > 0:
        entries.append((base, m - rem))
    if rem > 0:
        entries.append((base + 1, rem))
    return (tuple(entries), mu_power)
