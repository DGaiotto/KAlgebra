"""`skein_bps_iso` — a `KAlgebraIso` **generator** for the skein ↔ BPS
parallel on triangulated-polygon (A_{n−3} Argyres–Douglas) charts.

This packages the structural dictionary into a single generator:
given the disk with `n` marked points (an ideal
*polygon* triangulation Δ), it returns the certified `KAlgebraIso`
between the stated-skein realization and the BPS-quiver realization of
that triangulation — the executable form of "the skein algebra and
`A_𝖖[T]` are the same K_𝖖-algebra".

The dictionary it encodes
-------------------------
| skein side (triangulation Δ)          | BPS / RGKAlgebra side          |
|---------------------------------------|--------------------------------|
| underlying quantum torus `Y_Δ`        | auxiliary `QuantumTorusKAlg(B)`|
|   `y_e y_f = lq^{σ_Δ(e,f)} y_f y_e`   |   `X_γX_γ' = q^{⟨γ,γ'⟩}X_{γ+γ'}`|
| **σ_Δ (triangulation exchange mat.)** | **B (BPS Dirac pairing)** — *equal* |
| quantum trace `Tr^q : Sk → Y_Δ`       | `F = RG : A_𝖖[T] → Q_𝖖(Γ)`     |
|   ("the geometric F", `geometric_f`)  |   `F_γ·S = X_γ + O(𝖖)`          |
| `Tr^q(γ)` = **bare part of** `F(−γ)`   | `F(−γ) = Tr^q(γ) + bubbling`    |
| canonical basis = bare + dressing     | the dressed `L_a`              |

`bps_quiver_of_polygon(n)` is the explicit *triangulation → BPS quiver*
step (`σ_Δ = B`). The generator itself reuses each stated-skein class's
own `build_iso()` so the label conventions match: the iso's structure-
constant content is exactly the quantum-trace = F equality, computed
*genuinely on the skein side* (localized stated engine, peeled).

Scope
-----
The bordered polygons with a full stated-skein `KAlgebra` realization:

    n = 5  pentagon   = [A₁,A₂] = A₂ AD   → BPSKAlgebra([[0,1],[-1,0]])
    n = 6  u1hexagon  = U(1)-gauged A₃    → intrinsic U1A1AoddKAlg(1)
    n = 7  heptagon   = [A₁,A₄] = A₄ AD   → intrinsic A1A2kKAlg(2)

For `n = 5` the iso lands *literally* on the BPS quiver of the
triangulation. For `n = 6, 7` it lands on the intrinsic finite-type
presentation, which is itself a certified `KAlgebraIso`-twin of the BPS
chart in the `KAlgebraObject` (so "same as the BPS quiver" still holds,
one composition away). The *punctured-sphere* quantum-trace bridge
(`geometric_f`, the tetrahedron = SU(2) N_f=4) is the **partial**
parallel — the intrinsic dressing and trace are still open — so it
is NOT a full `KAlgebraIso` and is reached
through `geometric_f.GeometricF`, not this generator.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly

from bordered_triangulation import BorderedTriangulation, ROLE_A, ROLE_B, ROLE_C

# n -> short human description (the supported polygon family)
SUPPORTED = {
    5: "pentagon = [A1,A2] = A2 AD",
    6: "u1hexagon = U(1)-gauged A3",
    7: "heptagon = [A1,A4] = A4 AD",
}

_UNIT = LaurentPoly({0: 1})


# --------------------------------------------------------------------------
# triangulation -> BPS quiver  (the explicit sigma_Delta = B step)
# --------------------------------------------------------------------------

def bps_quiver_of_polygon(n: int):
    """The A_{n−3} BPS exchange matrix `B = σ_Δ` of the fan-triangulated
    `n`-gon (disk with `n` marked points), with unit-vector node charges.

    Internal diagonals `d_i = (0, i)` (i = 2..n−2) are the quiver nodes;
    two diagonals that are the two interior sides of a common triangle
    carry a ccw arrow (Fomin–Shapiro–Thurston / Labardini-Fragoso rule).
    Returns `(B, node_charges)`.

    This is the BPS quiver of the A_{n−3} Argyres–Douglas theory up to an
    overall orientation; the stated-skein class's *own* target chart
    (`SkeinPentagonKAlg._bps`, …) is the label-matched representative and
    is what the generated iso uses.  This function exhibits the structural
    identity `σ_Δ = B`.
    """
    if n < 4:
        raise ValueError("need n >= 4 (A_{n-3} with at least one node)")
    bt = BorderedTriangulation.fan_polygon(n)
    internal = list(bt.gluings)               # the interior diagonals
    m = len(internal)                         # = n - 3 quiver nodes
    slot2node: dict = {}
    for idx, (x, y) in enumerate(internal):
        slot2node[x] = idx
        slot2node[y] = idx
    B = [[0] * m for _ in range(m)]
    ccw = [(ROLE_A, ROLE_B), (ROLE_B, ROLE_C), (ROLE_C, ROLE_A)]
    for tri in range(bt.n_triangles):
        for rf, rt in ccw:
            sf, st = (tri, rf), (tri, rt)
            if sf in slot2node and st in slot2node:
                i, j = slot2node[sf], slot2node[st]
                if i != j:
                    B[i][j] += 1
                    B[j][i] -= 1
    node_charges = [tuple(1 if k == i else 0 for k in range(m)) for i in range(m)]
    return B, node_charges


# --------------------------------------------------------------------------
# the stated-skein realizations (the polygon family)
# --------------------------------------------------------------------------

def skein_polygon_kalgebra(n: int, **kwargs):
    """The stated-skein `KAlgebra` of the triangulated `n`-gon."""
    if n == 5:
        from skein_pentagon_kalg import SkeinPentagonKAlg
        return SkeinPentagonKAlg(**kwargs)
    if n == 6:
        from skein_u1hexagon_kalg import SkeinU1HexagonKAlg
        return SkeinU1HexagonKAlg(**kwargs)
    if n == 7:
        from skein_heptagon_kalg import SkeinHeptagonKAlg
        return SkeinHeptagonKAlg(**kwargs)
    raise ValueError(
        f"no stated-skein KAlgebra for the {n}-gon yet "
        f"(have n in {sorted(SUPPORTED)} = A2/A3/A4); the general bordered-"
        f"polygon engine is skein_sphere.stated_polygon.StatedPolygon"
    )


def generator_label_samples(K, n: int):
    """The chord-generator + identity labels of the `n`-gon skein
    realization, as the iso-battery sample set (the same generators each
    class's dedicated test uses)."""
    if n == 5:
        from skein_pentagon_kalg import ORBIT
        return list(ORBIT) + [(0, 0)]
    if n == 6:
        A = K._intr
        return (
            [A.identity(), ((), 1), ((), -1)]
            + [(((1, p, 1),), 0) for p in range(6)]
            + [(((2, i, 1),), 0) for i in range(3)]
        )
    if n == 7:
        return (
            [()]
            + [((1, i, 1),) for i in range(7)]
            + [((2, i, 1),) for i in range(7)]
        )
    raise ValueError(f"no sample generators registered for n={n}")


# --------------------------------------------------------------------------
# THE GENERATOR
# --------------------------------------------------------------------------

def skein_bps_iso(n: int, *, verify: bool = False, trace_K: int = 6, K=None):
    """Generate the skein ↔ BPS-quiver-of-triangulation `KAlgebraIso` for
    the triangulated `n`-gon (A_{n−3} AD).

    Parameters
    ----------
    n        : polygon size (5, 6 or 7 currently supported).
    verify   : if True, run the full `KAlgebraIso.verify_all` battery
               (unit / round-trip / multiplicative / ρ- / trace-
               equivariance) on the chord generators and raise on failure.
    trace_K  : q-order window for the trace-equivariance leg.
    K        : an already-built skein `KAlgebra` to reuse (else one is
               constructed).

    Returns
    -------
    `KAlgebraIso` : skein → (BPS for n=5; intrinsic-BPS-twin for n=6,7).

    Notes
    -----
    The independent content of this iso is its `multiplicative` leg: the
    skein structure constants (localized stated engine, peeled) equal the
    BPS chart's — the quantum-trace = F equality at the structure-constant
    level.  `ρ`/`trace` are transported from the BPS side on the skein
    realizations, so the `rho_equivariant` / `trace_equivariant` legs are
    enforced-by-transport (a consistency guard, not independent evidence)
    until an intrinsic skein trace exists.
    """
    if n not in SUPPORTED:
        raise ValueError(
            f"unsupported n={n}; supported polygons: {SUPPORTED}"
        )
    if K is None:
        K = skein_polygon_kalgebra(n)
    iso = K.build_iso()
    if verify:
        labels = generator_label_samples(K, n)
        samples = [Element({lab: _UNIT}) for lab in labels]
        pairs = [
            (Element({a: _UNIT}), Element({b: _UNIT}))
            for a in labels
            for b in labels
        ]
        res = iso.verify_all(samples, samples, pairs, pairs, trace_K=trace_K)
        if not all(res.values()):
            failed = {k: v for k, v in res.items() if not v}
            raise AssertionError(
                f"skein_bps_iso(n={n}) failed verification: {failed}"
            )
        iso._verify_report = dict(res)  # diagnostic stash on the returned iso
    return iso


def verify_sigma_equals_B(n: int, K=None) -> bool:
    """Exhibit `σ_Δ = B`: the fan-triangulation A_{n−3} exchange matrix
    equals the skein class's BPS-chart pairing, up to overall orientation
    (a relabeling of the same BPS quiver).  Only meaningful where the
    skein target *is* a `BPSKAlgebra` (n = 5)."""
    if K is None:
        K = skein_polygon_kalgebra(n)
    bps = getattr(K, "_bps", None)
    if bps is None:
        raise ValueError(
            f"n={n} skein target is the intrinsic chart, not a BPSKAlgebra; "
            f"σ_Δ = B holds via the object-layer intrinsic↔bps witness"
        )
    B_chart = bps._qt_aux.pairing
    B_tri, _ = bps_quiver_of_polygon(n)
    neg = [[-x for x in row] for row in B_tri]
    return B_tri == B_chart or neg == B_chart
