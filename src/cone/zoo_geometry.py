"""Geometric labels of the zoo's `A` and `D` entries: for the `A` and `D`
families, a geometric labelling of the canonical basis.

A zoo entry's canonical-basis labels are cone monomials in the indices of
machine-generated generators.  Each `A` / `D` entry is identified with a
family class by a certified generator map, the one its seed traces are served
through (or, for `a1d5` / `a1d7`, found the same way).  Through that map a zoo
canonical element is, up to a unit of the flavour ring, the family element
that the family class names by a multiset of curves.  `geometric_label(short_id,
label)` returns that multiset: a sorted tuple of pairs `(curve, m)`, `m ≥ 1`,
the unit being `()`.  Since the map carries the zoo's canonical basis
(over its coefficient ring) onto the family's, the multiset determines the zoo
element up to that unit.

| entries | map | curves |
|---|---|---|
| `pentagon`, `heptagon` | `aeven_seeds` onto `A1A2kKAlg(1)`, `A1A2kKAlg(2)` | diagonals `(v1, v2)` of the pentagon, heptagon |
| `a3` / `hexagon`, `a5` / `octagon`, `a7` / `decagon` | `aodd_seeds` onto `ungauge_u1a1aodd(k)` | balanced multisets of diagonals `(v1, v2)` of the `(2k+4)`-gon |
| `a1d3` | `a1d3_seeds` onto `A1DoddConeKAlg(0)` | curves `(x, ℓ)` of the once-punctured triangle |
| `a1d5`, `a1d7` | `a1dodd_seeds` onto `A1DoddConeKAlg(1)`, `A1DoddConeKAlg(2)` | curves `(x, ℓ)` of the once-punctured pentagon, heptagon |
| `a1d4` | `a1d4_seeds` onto `SU3ADKAlg` | balanced multisets of curves `(x, ℓ)` of the once-punctured square (`A1DevenKAlg(1)`'s curves) |
| `a1d6`, `a1d8` | `a1deven_seeds` onto `A1DevenKAlg(2)`, `A1DevenKAlg(3)` | balanced multisets of curves `(x, ℓ)` of the once-punctured hexagon, octagon |

A generator map is fixed only up to the family's rotation ρ, an automorphism
of the algebra.  The labels are therefore those of the served map; another
certified map gives the same labels rotated.

The generated zoo classes expose this as their `geometric_label(label)`
(emitted by the zoo's generator, `generate_finite_kalg`).  Pure Python; no BPS engine and no bootstrap is imported.
"""
from __future__ import annotations


__all__ = ["GEOMETRIC_IDS", "geometric_label"]

_A_EVEN = ("pentagon", "heptagon")
_A_ODD = ("a3", "hexagon", "a5", "octagon", "a7", "decagon")
_D_ODD = ("a1d3", "a1d5", "a1d7")
_D_EVEN = ("a1d4", "a1d6", "a1d8")

GEOMETRIC_IDS = _A_EVEN + _A_ODD + _D_ODD + _D_EVEN


def geometric_label(short_id: str, label) -> tuple:
    """The multiset of curves naming the zoo entry `short_id`'s canonical
    element `label` (module docstring); `KeyError` for an entry outside the
    `A` and `D` families."""
    word = tuple(label)
    if short_id in _A_EVEN:
        import aeven_seeds
        A = aeven_seeds.a1a2k_algebra(short_id)
        return A.geometric_label(aeven_seeds.to_a1a2k_label(short_id, word))
    if short_id in _A_ODD:
        import aodd_seeds
        A = aodd_seeds.ungauged_algebra(short_id)
        curves, _e = A.geometric_label(aodd_seeds.to_ungauged_label(short_id, word))
        return curves
    if short_id == "a1d3":
        import a1d3_seeds
        curves, _kappa = a1d3_seeds.cone_algebra().geometric_label(
            a1d3_seeds.to_a1dodd_label(word))
        return curves
    if short_id in _D_ODD:
        import a1dodd_seeds
        curves, _kappa = a1dodd_seeds.cone_algebra(short_id).geometric_label(
            a1dodd_seeds.to_a1dodd_label(short_id, word))
        return curves
    if short_id == "a1d4":
        import a1d4_seeds
        _charge, lab = a1d4_seeds.to_su3ad(word)
        curves, _weight = a1d4_seeds.cone_algebra().geometric_label(lab)
        return curves
    if short_id in _D_EVEN:
        import a1deven_seeds
        A = a1deven_seeds.ungauged_algebra(short_id)
        return A.geometric_label(a1deven_seeds.to_ungauged_label(short_id, word))[0]
    raise KeyError(f"zoo_geometry: {short_id!r} is not an A or D zoo entry "
                   f"({', '.join(GEOMETRIC_IDS)})")
