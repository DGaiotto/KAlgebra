"""`g_matter_roster` — named `(G, N)` **theory presets** for `GNAbeKAlgebra`.

Presets, **not classes**.  The author's ruling, 2026-07-28: *"use special names only if
you have algorithms specifically optimized for a G and/or N."*  A named class has
to earn its name with an algorithm, not by naming a nameable theory.  On
2026-09-19 the type-A classes that had claimed that licence were measured
against `GNAbeKAlgebra` / `PureGAbeKAlgebra` (adversarially reviewed) and
retired to the source repository's archive: `PureUNKAlgebra`, `PureSUNKAlgebra` (A75: wrong products),
`UNNfKAlgebra` and `UNQuiverKAlgebra` were at parity, slower, or defective.
The one that earned it is pure gauge:

| optimized class | what it optimizes |
|---|---|
| `PureSU2KAlgebra` | pure SU(2): the native adjoint-monopole fiber (the only non-solving construction of the odd-`e` fiber; ~450x on deep dressed labels) |

Everything else is `GNAbeKAlgebra(datum, matter, lines=…)` — one class taking the
**4d gauge group data** and the matter as arguments.  So this module holds no
subclasses at all: just a dict of `(group, matter)` parameter tuples with short
names, so the theories the `(G, N)` algorithm reaches are *addressable* (for
batteries, sweeps and discussion) without minting a class per theory.
`GNAbeKAlgebra.faster_equivalent()` returns `None` everywhere since the 2026-09-19
retirements (see its docstring).

An earlier version of this module *did* define per-theory subclasses
(`SU2FundAbeKAlgebra`, `SpinSpinorAbeKAlgebra`, …).  They were pure delegation —
no optimized algorithm anywhere — so the ruling retired them, and the honest
replacement is the preset table below.  Kept from that version:
`highest_root`, which is a genuine utility (and is how the adjoint presets are
declared).

Rank conventions — the repo's factories are indexed by RANK
----------------------------------------------------------
`sp_n(n).name == "Sp(n)"` while the physics group is `Sp(2n)`;
`b_n_simply_connected(n)` is `Spin(2n+1)` (vector `2n+1`, spinor `2^n`).  The
preset *names* below use the physics names, and each preset's algebra reports its
own `theory` string, so the call site is unambiguous either way.

Measured reach (2026-07-28)
--------------------------
the suite in the source repository builds every preset and records what it reaches:
all 13 certify at identity + Wilson + minimal monopole under W1 +
`certify_canonical` + ρ round-trip.  Twelve are sub-second; `g2-nf1`'s monopole
takes **~145 s** (13 atoms) — a *cost* observation, not a wall, so it is asserted
under the suite's `--full` flag.  Cost tracks bubbled cells carrying free
parameters, not rank, not `|W|`, not exceptionality.

`su2-adjoint` (`N = 2*` SU(2)) reaches all three labels instantly, so the presets
touch a corner whose specialised type-A machinery is the design record's.

Run `PYTHONPATH=$(ls -d src/* | paste -sd:) python3 src/gn/g_matter_roster.py` for a tour.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import root_datum as rd
from matter_wrq_torus import defining_weight

from gn_abe_kalgebra import GNAbeKAlgebra


__all__ = ["ROSTER", "roster", "roster_names", "roster_spec", "highest_root"]


def highest_root(datum) -> tuple:
    """The highest root of `datum` in the repo's weight coordinates — the
    adjoint representation's highest weight.

    Selected by the dominance order on the positive roots, using the same
    strictly-monotone functional the character peel uses
    (`g_matter_over_pure._height`), which is correct at **any** datum —
    including non-simply-laced, where `Σ_{α>0} α` is *not* monotone and an
    ordering by it can silently pick a non-highest weight.

    Verified rather than trusted: the maximum must be *strict* (no other positive
    root at maximal height), which is what makes "the" highest root well defined.

    **Reductive caveat.**  At `U(N)` this gives the `su(N)` adjoint (dim `N²−1`),
    not `u(N)` (dim `N²`) — the central `u(1)` is a separate *neutral* summand and
    must be declared as its own zero-weight slot if wanted.  At semisimple data
    the two agree (SU(2) 3, SU(3) 8, SU(4) 15, Sp(4) 10, Spin(5) 10, G₂ 14).
    """
    from g_matter_over_pure import _height
    pos = [tuple(a) for a in datum.positive_roots()]
    if not pos:
        raise ValueError(
            f"highest_root: {datum.name} has no positive roots — an abelian "
            "gauge factor has no adjoint matter to speak of")
    heights = {a: _height(datum, a) for a in pos}
    top = max(heights.values())
    winners = [a for a, h in heights.items() if h == top]
    if len(winners) != 1:
        raise ValueError(
            f"highest_root: {datum.name} has {len(winners)} positive roots at "
            f"maximal height {top} ({winners}) — not simple in this datum, so "
            "'the' highest root is ambiguous; declare the adjoint explicitly")
    return winners[0]


def _fund(factory, *args):
    """`(datum, defining weight)` — the fundamental/defining representation."""
    d = factory(*args)
    return d, tuple(defining_weight(d))


def _spinor(n):
    """`(datum, spinor weight)` for `Spin(2n+1)` — the last fundamental weight."""
    d = rd.b_n_simply_connected(n)
    return d, tuple([0] * (d.dim - 1) + [1])


def _adjoint(factory, *args):
    d = factory(*args)
    return d, tuple(highest_root(d))


#: Named `(G, N)` theory presets: `name -> (datum, matter highest weight, nf)`.
#: Values are lazy thunks so importing this module constructs nothing — the
#: roster is a catalogue of parameters, not a fleet of algebras.
ROSTER = {
    "su2-nf1":          lambda: (*_fund(rd.su_2), 1),
    "su2-nf2":          lambda: (*_fund(rd.su_2), 2),
    "su3-nf1":          lambda: (*_fund(rd.su_n, 3), 1),
    "u2-nf1":           lambda: (*_fund(rd.u_n, 2), 1),
    "u2-nf2":           lambda: (*_fund(rd.u_n, 2), 2),
    "u3-nf1":           lambda: (*_fund(rd.u_n, 3), 1),
    "sp4-nf1":          lambda: (*_fund(rd.sp_n, 2), 1),      # Sp(4) = sp_n(2)
    "sp4-nf2":          lambda: (*_fund(rd.sp_n, 2), 2),
    "spin5-vector-nf1": lambda: (*_fund(rd.b_n_simply_connected, 2), 1),
    "spin5-spinor-nf1": lambda: (*_spinor(2), 1),
    "spin5-spinor-nf2": lambda: (*_spinor(2), 2),
    "g2-nf1":           lambda: (*_fund(rd.g_2), 1),
    "su2-adjoint":      lambda: (*_adjoint(rd.su_2), 1),      # N = 2* SU(2)
}


def roster_names() -> tuple:
    """The preset names, in catalogue order."""
    return tuple(ROSTER)


def roster_spec(name: str) -> tuple:
    """The preset's `(datum, matter highest weight, nf)` without building the
    algebra — useful for parametrising a sweep."""
    try:
        thunk = ROSTER[name]
    except KeyError:
        raise KeyError(
            f"g_matter_roster: unknown theory {name!r}.  Available: "
            + ", ".join(sorted(ROSTER))) from None
    return thunk()


def roster(name: str, **kw) -> GNAbeKAlgebra:
    """Build a named preset: ``roster("spin5-spinor-nf2")``.

    Extra keywords go to `GNAbeKAlgebra` — notably `lines=` (the 4d gauge group
    data; default the simply connected form) and `allow_solve=False` (to isolate
    what the constructive routes reach)."""
    datum, lam, nf = roster_spec(name)
    return GNAbeKAlgebra(datum, lam, nf=nf, **kw)


if __name__ == "__main__":
    print("=" * 74)
    print("named (G, N) presets — GNAbeKAlgebra(datum, matter, lines=…)")
    print("=" * 74)
    for name in roster_names():
        A = roster(name)
        print(f"  {name:20s} {A!r}")
        print(f"  {'':20s}   theory  = {A.theory}")
        print(f"  {'':20s}   4d group= {A.lines}")
        print(f"  {'':20s}   flavour = {A.flavour_group()}  "
              f"(ring {A.coefficient_ring()})")
        fast = A.faster_equivalent()
        if fast is not None:
            print(f"  {'':20s}   faster  = {type(fast).__name__} "
                  f"(an OPTIMIZED class — this is what earns a name)")

    print()
    print("=" * 74)
    print("the removal tower of SU(2)+2 (presets ↔ matter-removal flows)")
    print("=" * 74)
    for rung in roster("su2-nf2").removal_tower():
        print(f"  {rung!r}")
