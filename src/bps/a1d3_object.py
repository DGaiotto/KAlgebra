"""`KAlgebraObject` for **A1D3** — `A_𝖖([A₁, D₃])`, the SU(2)-flavoured
Argyres–Douglas theory (chiral algebra affine `sl(2)_{−4/3}`).

A1D3 is the smallest odd-`D` AD theory.  Unlike the unflavoured polygon
zoo (pentagon / heptagon / …), its flavour symmetry is the full
**SU(2)** (the `D₃` leaf-exchange `Z₂` enhances the gauge `U(1)` to the
half-integer-spin doublet `χ₁`), so it lives over `SU2ZPlusRing` rather
than `Z` / `R(U(1))`.  That is exactly why the generic finite-zoo
factory (`finite_kalgebra_objects.kalgebra_object`) cannot build it:
its generic `bps` oracle is the U(1) `BPSKAlgebra` and its
`cone_to_bps_iso` assumes full-rank simplicial cones, neither of which
fits the SU(2)-flavoured, degenerate-pairing `D₃` chart.  This module
therefore builds the object standalone (the nonagon precedent), welding
two **ring-consistent** SU(2) presentations:

Realizations
------------
* ``'standalone'`` — `A1D3KAlg` (`a1d3_kalg.py`): the fully autonomous
  `Z₃`-symmetric cone presentation on 4-tuple labels `(tile, a, b, k)`
  (6 multiplicative generators `T_i`/`D_i`, the 8 `Z₃`-symmetric Plücker
  relations, χ-content in the `k` slot).  Carries the **closed-form
  two-layer trace**: Layer 1 reduces `Tr(label)` to the elementary basis
  `{Tr(1), Tr_T, Tr_D}` over `R(SU(2))[q^±]`, and Layer 2 substitutes the
  three `sl(2)_{−4/3}` admissible characters
  `Tr_1 = κ₀`, `Tr_D = −q⁻¹·κ₁^anti`,
  `Tr_T = +q⁻¹·(κ₀ − κ₁^sym − χ₁·κ₁^anti)`.
* ``'bps'`` — `SU2BPSKAlgebra` on the `w`-diagonal `[A₁, D₃]` chart
  (pairing `[[0,1,0],[-1,0,0],[0,0,0]]`, nodes `(1,0,0)`, `(0,1,±1)`,
  `w = diag(1,1,-1)`): the BPS-quiver realization, canonical basis the
  `w`-orbit chains `Ł_γ`, trace lifted from the underlying `BPSKAlgebra`
  and re-expressed in SU(2) characters.
* ``'rg-sqed1'`` — `A1D3SqedRGKAlgebra`: A1D3 as the **SU(2)-native RG
  flow to SQED1** (the `k = 0` base case of the A1Dodd family),
  `aux = U1SquareKAlg().add_flavour(SU2)`, `S_RG = E_𝖖(μL)E_𝖖(μ⁻¹L)`.
  Canonical labels are the SQED1 ⊗ SU(2) cone monomials `((m, n), κ)`.
* ``'skein-pinned'`` — `SkeinA1D3KAlg`
  (`skein_sphere/skein_a1d3_kalg.py`): the stated-skein realization on
  the ONCE-PUNCTURED TRIANGLE — the **first flavoured skein
  leg**.  The flavour character χ₁ IS the Kauffman peripheral loop
  around the puncture (measured exactly central, image
  `lq^{1/2}(y^f + y^{-f})` with `f = r₀+r₁+r₂ ∈ ker σ` the puncture
  flavour charge); the canonical generators D_i / T_i are the fully
  UNPINNED dressed corner arcs, satisfying the standalone relation set
  exactly at `q_chart = lq^{-2}`, and `multiply` is genuinely
  skein-side with the whole-element δ ≡ 0 standing guard asserted on
  every product.  Same `(tile, a, b, k)` labels; ρ/trace transported
  from the standalone (ρ = the Z₃ chart rotation, geometrically).

Witnesses
---------
* ``'standalone' ↔ 'bps'`` — the tile↔lattice correspondence
  `(tile, a, b, k) ↔ γ = (γ₀, γ₁, −k)` with
  `(γ₀, γ₁) = a·T_vec(tile) + b·D_vec(tile)`.  Forward is the letter
  γ-sum (the same map pinned in `tests/test_a1d3_kalg.py`); the inverse
  is the 2-D gauge-plane cone decomposition (the 6 tiles fan-triangulate
  `Z²`) plus `k = −γ₂`.
* ``'rg-sqed1' ↔ 'bps'`` — the piecewise-linear cone map
  `((m, n), κ) ↦ γ = (n, −m + 2·min(n, 0), −κ)` (see
  `a1d3_rg_to_bps_iso`).
* ``'standalone' ↔ 'rg-sqed1'`` — the composite through the bps hub
  (stored explicitly, so coherence is a genuine cycle).
* ``'skein-pinned' ↔ 'standalone'`` — the identity on labels
  (`SkeinA1D3KAlg.build_iso()`; verifying it tests the SKEIN structure
  constants against the standalone's — non-circular).

All four realizations are over `SU2ZPlusRing`, so every battery —
unit, round-trip, multiplicativity, ρ-equivariance,
**trace-equivariance** — and the coherence certificate pass on the
canonical basis (the standalone's Layer-2 trace, the BPS trace, and the
RG-transport trace all agree).

See the design notes for the finalized-object summary.
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from kalgebra_object import KAlgebraObject
from laurent_poly import LaurentPoly

from a1d3_kalg import A1D3KAlg, _TILE_LETTERS
from su2_bps_kalgebra import SU2BPSKAlgebra
from a1d3_sqed_rgkalgebra import A1D3SqedRGKAlgebra

__all__ = [
    "a1d3_object", "a1d3_bps",
    "a1d3_standalone_to_bps_iso", "a1d3_rg_to_bps_iso",
]

_ONE = LaurentPoly.one()

# Gauge-plane (2-D) lattice vectors of the 6 letters, indexed by letter
# index i ∈ {0, 1, 2}.  These are the first two coordinates of the BPS
# lattice vectors (the third coordinate is the SU(2)-Cartan / flavour
# direction, carried separately as −k).
_T_GAUGE = [(1, 0), (-1, -2), (-1, 0)]
_D_GAUGE = [(0, -1), (-1, -1), (0, 1)]


def a1d3_bps() -> SU2BPSKAlgebra:
    """The `[A₁, D₃]` BPS realization in the `w`-diagonal frame."""
    return SU2BPSKAlgebra(
        pairing=[[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
        node_charges=[(1, 0, 0), (0, 1, 1), (0, 1, -1)],
        w_su2=[[1, 0, 0], [0, 1, 0], [0, 0, -1]],
    )


def _solve_2d(u, w, g):
    """Solve `a·u + b·w = g` for non-negative integers `a, b` (exact,
    `u, w, g ∈ Z²`).  Returns `(a, b)` or `None`."""
    det = u[0] * w[1] - u[1] * w[0]
    if det == 0:
        return None
    na = g[0] * w[1] - g[1] * w[0]
    nb = u[0] * g[1] - u[1] * g[0]
    if na % det or nb % det:
        return None
    a, b = na // det, nb // det
    if a < 0 or b < 0:
        return None
    return a, b


def a1d3_standalone_to_bps_iso(standalone: A1D3KAlg,
                               bps: SU2BPSKAlgebra) -> KAlgebraIso:
    """The certified witness `standalone → bps` (tile ↔ lattice)."""

    def forward(label) -> Element:
        tile, a, b, k = standalone.canonicalise(label)
        (_kt, it), (_kd, idd) = _TILE_LETTERS[tile]
        tg, dg = _T_GAUGE[it], _D_GAUGE[idd]
        g0 = a * tg[0] + b * dg[0]
        g1 = a * tg[1] + b * dg[1]
        return Element({bps.canonicalise((g0, g1, -k)): _ONE})

    def inverse(gamma) -> Element:
        g = bps.canonicalise(gamma)
        k = -g[2]                      # canonical bps rep has flavour_diff ≤ 0
        gauge = (g[0], g[1])
        if gauge == (0, 0):
            return Element({standalone.canonicalise((0, 0, 0, k)): _ONE})
        # The 6 tiles fan-triangulate the gauge plane; the first tile whose
        # 2-D cone contains `gauge` gives (a, b), and `canonicalise` fixes
        # the boundary/single-letter tie-breaks exactly as the standalone
        # does on the forward image.
        for tile in range(6):
            (_kt, it), (_kd, idd) = _TILE_LETTERS[tile]
            sol = _solve_2d(_T_GAUGE[it], _D_GAUGE[idd], gauge)
            if sol is not None:
                a, b = sol
                return Element(
                    {standalone.canonicalise((tile, a, b, k)): _ONE})
        raise ValueError(
            f"a1d3 standalone↔bps: gauge charge {gauge} (from γ={gamma}) "
            f"is not a non-negative combination of any tile's two letters")

    return KAlgebraIso(
        standalone, bps,
        forward_label_map=forward,
        inverse_label_map=inverse,
        name="a1d3[standalone→bps]",
    )


def a1d3_rg_to_bps_iso(rg: A1D3SqedRGKAlgebra,
                       bps: SU2BPSKAlgebra) -> KAlgebraIso:
    """The certified witness `rg-sqed1 → bps`.

    The SU(2)-native RG realization's canonical labels are the SQED1 ⊗
    SU(2) cone monomials `((m, n), κ)` (`m` = magnetic / `L`-power, `n` =
    gauge / `v`-power, `κ` = SU(2) highest weight).  They map to the `D₃`
    BPS chart by the **piecewise-linear** cone correspondence

        ((m, n), κ)  ↦  γ = (n, −m + 2·min(n, 0), −κ),

    i.e. `γ₀ = n`, `γ₁ = −m + 2·min(n, 0)` (piecewise-linear, as ρ itself
    is), `γ₂ = −κ`.  Pinned by the generator dictionary
    (`((0,1),0) ↦ T₀ = (1,0,0)`, `((1,0),0) ↦ D₀ = (0,−1,0)`,
    `((0,0),κ) ↦ χ_κ = (0,0,−κ)`) and certified on products by the full
    battery."""

    def forward(rg_label) -> Element:
        (m, n), kappa = rg_label
        return Element(
            {bps.canonicalise((n, -m + 2 * min(n, 0), -kappa)): _ONE})

    def inverse(gamma) -> Element:
        g = bps.canonicalise(gamma)
        n = g[0]
        kappa = -g[2]                          # bps canonical rep: γ₂ ≤ 0
        m = -g[1] + 2 * min(n, 0)
        return Element({((m, n), kappa): _ONE})

    return KAlgebraIso(
        rg, bps,
        forward_label_map=forward,
        inverse_label_map=inverse,
        name="a1d3[rg-sqed1→bps]",
    )


def a1d3_object() -> KAlgebraObject:
    """The abstract `A_𝖖([A₁, D₃])` (SU(2)-flavoured) as a
    `KAlgebraObject` holding its four certified realizations —
    `standalone`, `bps`, the SU(2)-native RG flow to SQED1
    (`rg-sqed1`), and the once-punctured-triangle stated-skein
    realization (`skein-pinned`)."""
    obj = KAlgebraObject("A_q[[A1,D3]] (A1D3, SU(2)-flavoured AD)")

    standalone = A1D3KAlg()
    bps = a1d3_bps()

    obj.add_realization(
        "standalone", standalone,
        {"multiply-fast", "trace-exact", "trace-closed-form"})
    obj.add_realization("bps", bps, {"chart", "trace-exact"})
    obj.add_iso("standalone", "bps",
                a1d3_standalone_to_bps_iso(standalone, bps))

    # The SU(2)-flavoured-native RG flow to SQED1 (the k=0 base case of
    # the A1Dodd family).  Witness rg-sqed1 ↔ bps is the piecewise-linear
    # cone map above; standalone ↔ rg-sqed1 is stored as the composite
    # through the bps hub, so it carries its own battery line and the
    # coherence certificate is a genuine cycle (not a tree).
    rg = A1D3SqedRGKAlgebra()
    obj.add_realization("rg-sqed1", rg, {"rg", "trace-exact"})
    obj.add_iso("rg-sqed1", "bps", a1d3_rg_to_bps_iso(rg, bps))
    obj.add_iso("standalone", "rg-sqed1", obj.iso("standalone", "rg-sqed1"))

    # The once-punctured-triangle stated-skein realization —
    # the first FLAVOURED skein leg: χ₁ = the Kauffman peripheral loop
    # around the puncture (exactly central), the D_i/T_i the fully
    # unpinned dressed corner arcs, multiply genuinely skein-side with
    # the whole-element δ ≡ 0 standing guard.  Built ON the registered
    # standalone instance, so the witness endpoints match.
    from skein_a1d3_kalg import SkeinA1D3KAlg
    sk = SkeinA1D3KAlg(intrinsic=standalone)
    obj.add_realization("skein-pinned", sk, {"geometric"})
    obj.add_iso("skein-pinned", "standalone", sk.build_iso())

    return obj
