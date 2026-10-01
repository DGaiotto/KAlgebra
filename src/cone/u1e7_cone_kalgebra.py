"""
u1e7_cone_kalgebra.py
=====================

`U1E7ConeKAlgebra` — the u(1)-gauged E7 algebra as a **standalone QT-cone
ConeKAlgebra**, built once from an RG-flow oracle (`U1E7GaugedRG`, a
derivation **not included in this repository**); no engine at runtime — the
frozen tables (`u1e7_cone_tables.pkl`, `u1e7_rho_tables.pkl`) are loaded
instead, following the spine-free QTCone-from-RG recipe.

The oracle is `U1A1E7RGKAlgebra`, whose dressing chord is the central chord
`(3, 0)`; the shipped tables were rebuilt from it on 2026-09-23.  Tables built
from its earlier `(2, 2)` dressing — the flow of `[A₁,D₇]` with the Cartan
`U(1)` of its `SU(2)` gauged — are refused on load (`_STALE_SHA256`).

Structure (= the recipe):
  * **rank-1 torus** = the gauge leg `E = X_{(0,1)}` (Laurent).  The magnetic
    leg `X_{(1,0)}` is **not** a torus direction; magnetic charge is carried by
    the **dyonic chords** (200 chords, selected by the table builder,
    `c0 ∈ {-3..3}`), stored as atoms at gauge charge `c1 = 0` — with `E^{±1}`,
    202 atoms.
  * **cones** = the q-commuting cliques of the atoms (4160; `E^{±1}` lie in
    every cone).
  * **factoring** (`to_cone_label`): chord-atoms cover the A6 letters (magnetic
    `c0` matched exactly), the residual `c0` is supplied by **monopole atoms**
    `X_{(±1,0)}` (chord-atoms with empty A6 part, NOT torus), the residual
    gauge `c1` by `E`.  The build factors every product canonical of every
    ordered atom pair (it raises otherwise): 30110 cross products, 120490
    terms.
  * **ρ**: tabulated on the 201 generator keys, the rest by the automorphism
    rule (`rho`, `freeze_rho`).
  * **trace** (Layer 2): magnetic `c0 ≠ 0 ⇒ Tr = 0` exactly; every
    magnetically neutral label (`c0 = 0`: the gauge v-tower `E^n`, incl.
    `Tr(1)`, and every neutral chord label) through the ungauged `[A₁,E₇]`
    algebra `FiniteE7KAlgebra` (its closed-form traces, `e7_seeds`):

        Tr(L_ℓ) = [μ^{−r}] ( (𝖖²;𝖖²)_∞² · Tr_{[A₁,E₇]}(L_z; μ) ),   φ(L_ℓ) = μ^r·L_z,

    with `φ` the label map of the neutral sector onto `FiniteE7KAlgebra`
    (`E ↦ μ`, each zoo generator `g` the image of the label
    `_E7_GENERATOR_PREIMAGES[γ_g]`; `_e7_image` inverts it by an exact
    chord-multiset cover over the zoo's cones; `E^n` is `z = ()`, `r = n`).
    The identity is the gauging of the U(1) flavour: `(𝖖²;𝖖²)_∞²` is the
    vector-multiplet measure and the constant term in `μ` the gauge
    projection.  The E-tower's vacuum is the zoo's closed form (equal to the
    Nahm e7 sum, `_vacuum_mu`, kept as the witness).  `φ` is certified by
    products (the suite in the source repository):
    all 8100 ordered generator products and all 2600 zoo 6-cones agree, and a
    perturbed map fails.  No RG flow, no BPS engine and no bootstrap is on this
    path; the former orthonormality bootstrap `u1e7_trace_bootstrap` is not
    used (on these tables it raised `MemoryError` at `K = 4`, and it served
    every word missing from its solution as exact 0).
"""
from __future__ import annotations

import hashlib
import os
import sys
import itertools
import pickle
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing, ZPlusRing, RPowerSeries, RElement
from cone_data import FiniteConeData, Cone
from cone_kalgebra import ConeKAlgebra
# NOTE: `u1e7_gauged_rg` (the RG-flow oracle) and `u1e7_cone_derivation` are
# derivation modules; they are imported LAZILY (inside the build / oracle
# paths only) so the frozen, spine-free runtime never touches a realisation
# engine.


def _lp(c) -> LaurentPoly:
    return c if isinstance(c, LaurentPoly) else LaurentPoly(c._coeffs)


def _frozen_path() -> str:
    return os.path.join(_HERE, "u1e7_cone_tables.pkl")


def _rho_frozen_path() -> str:
    return os.path.join(_HERE, "u1e7_rho_tables.pkl")


# sha256 of the two tables built from the flow with dressing chord (2, 2) — the
# flow of [A1,D7] with the Cartan U(1) of its SU(2) gauged, not gauged [A1,E7]
# (see `u1a1e7_rgkalgebra.py`).  They are refused on load.
_STALE_SHA256 = {
    "21f311e12598d4a09da431d161e18f0eeebfb199dce1b0d12acbdf5753d03e77":
        "u1e7_cone_tables.pkl",
    "1526b1dd1b2443dd638af2b68ae5e0f5a1a732653a9712f2efcd82a339fc107d":
        "u1e7_rho_tables.pkl",
}


def _refuse_stale_tables(path: str) -> None:
    """Raise if `path` is one of the tables built from the `(2, 2)`-dressed flow."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() in _STALE_SHA256:
        raise RuntimeError(
            f"{path} was built from the U1A1E7RGKAlgebra flow with dressing chord "
            "(2, 2), which is [A1,D7] with the Cartan U(1) of its SU(2) gauged, "
            "not gauged [A1,E7]; U1E7ConeKAlgebra does not serve it.  Rebuild "
            "both tables on the corrected flow (dressing chord (3, 0)) with "
            "`PYTHONPATH=$(ls -d src/* | paste -sd:) python3 src/cone/u1e7_cone_kalgebra.py` (a long "
            "background job); see `src/rg/u1a1e7_rgkalgebra.py`.")


# ---- the neutral sector as the ungauged [A1,E7] algebra ---------------------
#
# The magnetically neutral labels `(chord, (0, c1))` of this algebra span the
# ungauged [A1,E7] algebra with its U(1) fugacity promoted to the gauge leg:
# the algebra isomorphism phi onto `FiniteE7KAlgebra` (with E |-> mu) sends
# L_{y_g} to L_g for the label y_g = (chord, (0, c1)) listed against the zoo
# generator g's E7 lattice vector (`finite_e7_kalg.E7_MULT_GENS_LATTICE[g]`).
# 70 entries are this algebra's magnetically neutral chord atoms (up to a power
# of E); each of the 20 marked ones is the product of a c0 = +1 and a c0 = -1
# chord atom of one of this presentation's cones: a generator of the neutral
# sector, though not of this cone presentation.  Found 2026-09-23 by matching
# the 70 neutral atoms' flavour-refined traces (`E7RGKAlgebra`, finder only)
# against the zoo's closed-form generator traces, then fixing the alignment by
# rho-equivariance and the q-commutation graph (unique up to rho^2, which
# moves no trace) and reading the 20 products off generator products.
# Certified by products in the suite in the source repository: all 8100
# ordered generator products agree under phi, every zoo 6-cone's image
# product is one term with the zoo's phase, and a perturbed map fails.
_E7_GENERATOR_PREIMAGES = {
    (-2, -1, 0, -1, -1, 0, 0): (((1, 5, 1),), -1),
    (-2, -1, 0, 0, -1, -1, 0): (((1, 3, 1), (2, 8, 1), (3, 7, 1)), -1),
    (-2, -1, 0, 0, 0, 0, 0): (((1, 5, 1), (2, 8, 1)), -1),  # c0 = 0 product of magnetic atoms
    (-1, -1, -1, -2, -2, -1, 0): (((3, 5, 1),), 0),
    (-1, -1, -1, -1, -2, -1, 0): (((1, 3, 1), (1, 8, 1), (2, 3, 1), (3, 7, 2)), -1),
    (-1, -1, -1, -1, -1, -1, 0): (((2, 2, 1),), 0),
    (-1, -1, -1, -1, -1, 0, 0): (((1, 3, 1), (1, 8, 1), (3, 7, 1)), -1),
    (-1, -1, -1, -1, 0, 0, 0): (((1, 5, 1), (3, 4, 1)), 0),  # c0 = 0 product of magnetic atoms
    (-1, -1, -1, -1, 0, 1, 0): (((1, 5, 1), (1, 8, 1)), -1),  # c0 = 0 product of magnetic atoms
    (-1, -1, -1, 0, 0, -1, 0): (((2, 2, 1), (2, 8, 1)), 0),  # c0 = 0 product of magnetic atoms
    (-1, -1, -1, 0, 0, 0, 0): (((1, 3, 1), (1, 8, 1), (2, 8, 1), (3, 7, 1)), -1),  # c0 = 0 product of magnetic atoms
    (-1, -1, 0, -1, -2, -1, 0): (((1, 7, 1),), 0),
    (-1, -1, 0, 0, -1, -1, 0): (((3, 7, 1),), 0),
    (-1, -1, 0, 0, 0, 0, 0): (((2, 4, 1),), 0),
    (-1, -1, 0, 1, 0, -1, 0): (((2, 8, 1), (3, 7, 1)), 0),
    (-1, 0, 0, -1, -2, -1, 0): (((1, 3, 1), (2, 3, 1), (3, 7, 1)), -1),
    (-1, 0, 0, -1, -1, 0, 0): (((1, 3, 1),), -1),
    (-1, 0, 0, -1, 0, 1, 0): (((1, 5, 1), (3, 8, 1)), -1),
    (-1, 0, 0, 0, 0, 0, 0): (((1, 3, 1), (2, 8, 1)), -1),
    (-1, 0, 1, 0, -1, -1, 0): (((1, 0, 1),), 0),
    (-1, 0, 1, 0, -1, 0, 0): (((3, 3, 1),), -1),
    (-1, 0, 1, 0, 0, 0, 0): (((2, 8, 1), (3, 3, 1)), -1),
    (0, -2, -3, -2, -2, -2, 0): (((2, 2, 1), (2, 7, 1), (3, 2, 1)), 1),
    (0, -2, -3, -2, -1, 0, 0): (((1, 8, 1), (2, 2, 1), (2, 7, 1)), 0),  # c0 = 0 product of magnetic atoms
    (0, -1, -3, -3, -2, 0, 0): (((1, 3, 2), (1, 8, 1), (2, 7, 1), (3, 2, 1)), -1),
    (0, -1, -3, -2, -1, -1, 0): (((1, 3, 1), (1, 8, 1), (2, 2, 1), (3, 2, 1)), 0),  # c0 = 0 product of magnetic atoms
    (0, -1, -2, -3, -1, 0, -2): (((3, 1, 1), (3, 6, 1)), -1),
    (0, -1, -2, -2, -2, -1, 0): (((1, 3, 1), (2, 7, 1), (3, 2, 1)), 0),
    (0, -1, -2, -2, -1, -1, -2): (((1, 3, 1), (1, 8, 1), (2, 7, 1), (3, 2, 2)), -1),
    (0, -1, -2, -2, -1, -1, -1): (((2, 2, 1), (3, 6, 1)), 0),
    (0, -1, -2, -2, -1, 0, -1): (((1, 3, 1), (1, 8, 1), (2, 7, 1), (3, 2, 1)), -1),
    (0, -1, -2, -2, -1, 0, 0): (((3, 1, 1),), 0),
    (0, -1, -2, -2, -1, 1, 0): (((1, 3, 1), (1, 8, 1), (2, 7, 1)), -1),
    (0, -1, -2, -1, -1, -2, 0): (((2, 2, 1), (3, 2, 1)), 1),
    (0, -1, -2, -1, -1, -1, 0): (((1, 3, 1), (1, 8, 1), (3, 2, 1), (3, 7, 1)), 0),
    (0, -1, -2, -1, 0, -1, -1): (((1, 8, 1), (2, 2, 1), (3, 2, 1)), 0),
    (0, -1, -2, -1, 0, 0, 0): (((1, 8, 1), (2, 2, 1)), 0),
    (0, -1, -1, -2, -2, -1, -1): (((1, 7, 1), (3, 6, 1)), 0),  # c0 = 0 product of magnetic atoms
    (0, -1, -1, -1, -2, -2, 0): (((1, 7, 1), (3, 2, 1)), 1),
    (0, -1, -1, -1, -1, -1, -1): (((2, 7, 1), (3, 2, 1)), 0),
    (0, -1, -1, -1, -1, -1, 0): (((3, 0, 1),), 1),
    (0, -1, -1, -1, -1, 0, -1): (((1, 8, 1), (2, 3, 1), (2, 7, 1), (3, 7, 1)), -1),  # c0 = 0 product of magnetic atoms
    (0, -1, -1, -1, -1, 0, 0): (((2, 7, 1),), 0),
    (0, -1, -1, -1, 0, 0, -1): (((2, 1, 1),), 0),
    (0, -1, -1, 0, 0, -1, 0): (((1, 2, 1),), 1),
    (0, -1, -1, 0, 0, 0, 0): (((1, 8, 1), (3, 7, 1)), 0),
    (0, -1, 0, 0, -1, -1, 0): (((1, 4, 1), (1, 7, 1)), 1),  # c0 = 0 product of magnetic atoms
    (0, -1, 0, 0, 0, 0, 0): (((1, 4, 1), (3, 3, 1)), 0),  # c0 = 0 product of magnetic atoms
    (0, -1, 0, 1, 0, -1, 0): (((1, 4, 1), (3, 7, 1)), 1),
    (0, 0, -1, -2, -2, -1, -1): (((1, 3, 1), (2, 3, 1), (2, 7, 1), (3, 2, 1)), -1),  # c0 = 0 product of magnetic atoms
    (0, 0, -1, -2, -2, 0, 0): (((1, 3, 1), (2, 3, 1), (2, 7, 1)), -1),
    (0, 0, -1, -2, -1, -1, -1): (((1, 6, 1), (3, 5, 1)), 0),  # c0 = 0 product of magnetic atoms
    (0, 0, -1, -2, -1, 0, -1): (((1, 3, 1), (3, 6, 1)), -1),
    (0, 0, -1, -1, -1, -1, 0): (((1, 3, 1), (3, 2, 1)), 0),
    (0, 0, -1, -1, -1, 0, 0): (((1, 3, 1), (1, 8, 1), (2, 3, 1), (3, 7, 1)), -1),
    (0, 0, -1, -1, 0, 0, -1): (((1, 3, 1), (1, 8, 1), (3, 2, 1)), -1),
    (0, 0, -1, -1, 0, 0, 0): (((2, 5, 1),), 0),
    (0, 0, -1, -1, 0, 1, 0): (((1, 3, 1), (1, 8, 1)), -1),
    (0, 0, 0, -1, -2, -1, 0): (((1, 7, 1), (2, 3, 1)), 0),  # c0 = 0 product of magnetic atoms
    (0, 0, 0, -1, -1, -1, -1): (((2, 6, 1),), 0),
    (0, 0, 0, -1, -1, 0, -1): (((2, 3, 1), (2, 7, 1)), -1),
    (0, 0, 0, -1, -1, 0, 0): (((2, 0, 1),), 0),
    (0, 0, 0, -1, 0, 1, 0): (((3, 3, 1), (3, 8, 1)), -1),
    (0, 0, 0, 0, -1, -1, 0): (((2, 3, 1), (3, 7, 1)), 0),
    (0, 0, 0, 0, 0, -1, 0): (((2, 8, 1),), 0),
    (0, 0, 0, 0, 0, 1, 0): (((1, 8, 1), (3, 3, 1)), -1),
    (0, 0, 0, 1, 0, -1, 0): (((2, 3, 1), (2, 8, 1), (3, 7, 1)), 0),
    (0, 1, 0, -1, -1, 0, 0): (((1, 3, 1), (2, 3, 1)), -1),  # c0 = 0 product of magnetic atoms
    (0, 1, 0, -1, 0, 1, 0): (((1, 3, 1), (3, 8, 1)), -1),
    (0, 1, 0, 0, 0, 0, 0): (((1, 3, 1), (2, 3, 1), (2, 8, 1)), -1),  # c0 = 0 product of magnetic atoms
    (1, -1, -3, -2, 0, 1, 0): (((1, 8, 1), (3, 1, 1)), 0),
    (1, -1, -2, -1, 0, -1, 0): (((3, 4, 1),), 1),
    (1, -1, -2, -1, 0, 0, -1): (((1, 8, 1), (2, 7, 1), (3, 2, 1)), 0),
    (1, -1, -2, -1, 0, 1, 0): (((1, 8, 1), (2, 7, 1)), 0),  # c0 = 0 product of magnetic atoms
    (1, -1, -1, 0, 0, -1, 0): (((1, 4, 1),), 1),
    (1, -1, -1, 0, 0, 0, 0): (((1, 4, 1), (2, 7, 1)), 1),  # c0 = 0 product of magnetic atoms
    (1, 0, -2, -2, 0, 0, -1): (((1, 6, 1), (3, 1, 1)), 0),
    (1, 0, -2, -1, 0, -1, 0): (((1, 6, 1), (2, 2, 1)), 1),  # c0 = 0 product of magnetic atoms
    (1, 0, -2, -1, 0, 1, 0): (((1, 3, 1), (1, 8, 2), (2, 3, 1), (3, 7, 1)), -1),
    (1, 0, -1, -1, 0, 0, -1): (((3, 6, 1),), 0),
    (1, 0, -1, -1, 0, 0, 0): (((3, 8, 1),), 0),
    (1, 0, -1, -1, 0, 1, 0): (((1, 1, 1),), 0),
    (1, 0, -1, 0, 0, -1, 0): (((3, 2, 1),), 1),
    (1, 0, -1, 0, 0, 0, 0): (((1, 8, 1), (2, 3, 1), (3, 7, 1)), 0),
    (1, 0, -1, 0, 1, 0, 0): (((1, 8, 1),), 0),
    (1, 0, 0, 0, 0, 0, 0): (((2, 3, 1),), 0),
    (1, 1, -1, -1, 0, 0, 0): (((1, 3, 1), (1, 6, 1)), 0),  # c0 = 0 product of magnetic atoms
    (1, 1, -1, -1, 0, 1, 0): (((1, 3, 1), (1, 8, 1), (2, 3, 1)), -1),
    (2, 0, -2, -1, 0, -1, 0): (((1, 6, 1),), 1),
    (2, 0, -2, -1, 0, 1, 0): (((1, 8, 1), (2, 3, 1), (2, 7, 1)), 0),
}


class U1E7ConeData(FiniteConeData):
    """QT-cone data for u(1)-gauged E7 over the trivial ring; rank-1 torus on the
    gauge leg `E = X_{(0,1)}`, magnetic carried by dyonic chords."""

    def __init__(self, oracle, use_frozen: bool = True) -> None:
        self._R = TrivialZPlusRing()
        if use_frozen and self._load_frozen():
            return
        T = oracle
        self._build(T)

    # ---- build from the oracle ------------------------------------------

    def _build(self, T) -> None:
        from u1e7_cone_derivation import (   # lazy: oracle-side only
            select_chords, _trim_oracle_caches)
        chords = select_chords(T)
        # one atom per (chord_part, c0) key, at gauge charge c1 = 0 (E carries
        # the gauge charge; c1 = 0 keeps every oracle product at the lowest
        # gauge degree)
        bykey: dict = {}
        for ch in chords:
            bykey[(ch[0], ch[1][0])] = (ch[0], (ch[1][0], 0))
        chordatoms = sorted(bykey.values(), key=repr)
        E, Ei = ((), (0, 1)), ((), (0, -1))
        self._atoms = tuple(chordatoms + [E, Ei])
        self._sig = {a: self._atom_sig(a) for a in self._atoms}
        # monopole atoms (empty A6 chord, c0 != 0): the magnetic generators
        self._mono = {}
        for a in chordatoms:
            if a[0] == ():
                self._mono[1 if a[1][0] > 0 else -1] = a
        # gauge torus E
        self._E = {self._sig[a][1]: a
                   for a in self._atoms if self._sig[a][0] is None}
        self._torus_gens = frozenset(
            g for g in (self._E.get((0, 1)), self._E.get((0, -1))) if g)
        # chord atoms indexed by (a, i)
        self._ai_atoms = defaultdict(list)
        for a in chordatoms:
            if a[0]:
                self._ai_atoms[self._sig[a][0]].append(a)
        # each unordered atom pair is multiplied by the oracle once, memoised
        # here for the q-commute graph, the cocycles and the cross products;
        # the other order is its bar image (`L_h·L_g = bar(L_g·L_h)`: the bar
        # involution is antimultiplicative and fixes the canonical basis),
        # checked against the oracle on every 25th pair.
        keep = frozenset(self._atoms)
        prods: dict = {}
        nbar = [0]

        def mul(g, h):
            v = prods.get((g, h))
            if v is None:
                w = prods.get((h, g))
                if w is not None:
                    v = w.bar()
                    nbar[0] += 1
                    if nbar[0] % 25 == 1:
                        _trim_oracle_caches(T, keep=keep)
                        if self._terms(T.multiply(g, h)) != self._terms(v):
                            raise AssertionError(
                                f"bar image of {h}·{g} differs from {g}·{h}")
                else:
                    _trim_oracle_caches(T, keep=keep)
                    v = T.multiply(g, h)
                prods[(g, h)] = v
            return v

        # q-commute graph
        self._nb = {v: set() for v in self._atoms}
        for g, h in itertools.combinations(self._atoms, 2):
            if len(self._terms(mul(g, h))) == 1:
                self._nb[g].add(h)
                self._nb[h].add(g)
        self._cones = self._maximal_cliques()
        # cone index by EVERY chord type appearing in any atom (incl multi-letter
        # atoms — indexing by the first letter only over-restricts the cover)
        self._cone_by_type = defaultdict(list)
        for ci, cone in enumerate(self._cones):
            types = set()
            for g in cone:
                for (k, i, e) in g[0]:
                    types.add((k, i))
            for tt in types:
                self._cone_by_type[tt].append(ci)
        self._post_init()           # factoring indices (cross extraction needs them)
        # cocycle + cross-product tables (from the oracle).  Two passes: the
        # cocycle table must be complete before cross extraction, which calls
        # cone_label_phase → cocycle on the daughter cone monomials.
        self._qpow: dict = {}
        self._xprod: dict = {}
        for g, h in itertools.permutations(self._atoms, 2):
            if h in self._nb[g]:
                self._qpow[(g, h)] = self._extract_cocycle(mul, g, h)
        for g, h in itertools.permutations(self._atoms, 2):
            if h not in self._nb[g]:
                self._xprod[(g, h)] = self._extract_cross(mul, g, h)

    def _post_init(self) -> None:
        """Derived indices for factoring (recomputed on build and load)."""
        self._tcl_cache: dict = {}
        self._atom_ct: dict = {}
        self._by_chordc0: dict = {}
        for a in self._atoms:
            chord, (c0, c1) = a
            ct = Counter()
            for (k, i, e) in chord:
                ct[(k, i)] += e
            self._atom_ct[a] = ct
            if chord:
                self._by_chordc0[(chord, c0)] = a
        self._cone_chord_atoms: dict = {
            cone: [a for a in cone if a[0]] for cone in self._cones}

    # ---- freeze / load (spine-free runtime) ------------------------------

    _FROZEN_KEYS = ("_atoms", "_sig", "_mono", "_E", "_torus_gens",
                    "_ai_atoms", "_nb", "_cones", "_cone_by_type",
                    "_qpow", "_xprod")

    def _load_frozen(self) -> bool:
        path = _frozen_path()
        if not os.path.exists(path):
            return False
        _refuse_stale_tables(path)
        with open(path, "rb") as f:
            d = pickle.load(f)
        for k in self._FROZEN_KEYS:
            setattr(self, k, d[k])
        self._ai_atoms = defaultdict(list, self._ai_atoms)
        self._cone_by_type = defaultdict(list, self._cone_by_type)
        self._post_init()
        return True

    def freeze(self) -> str:
        path = _frozen_path()
        d = {k: getattr(self, k) for k in self._FROZEN_KEYS}
        d["_ai_atoms"] = dict(self._ai_atoms)
        d["_cone_by_type"] = dict(self._cone_by_type)
        with open(path, "wb") as f:
            pickle.dump(d, f)
        return path

    # ---- helpers --------------------------------------------------------

    def _atom_sig(self, a):
        chord, (c0, c1) = a
        return (chord[0][:2] if chord else None), (c0, c1)

    def _terms(self, elt) -> dict:
        return {l: _lp(c) for l, c in elt.terms.items() if not c.is_zero()}

    def _maximal_cliques(self):
        cliques, nb = [], self._nb

        def bk(Rs, P, X):
            if not P and not X:
                cliques.append(frozenset(Rs))
                return
            piv = max(P | X, key=lambda u: len(P & nb[u]))
            for v in list(P - nb[piv]):
                bk(Rs | {v}, P & nb[v], X & nb[v])
                P = P - {v}
                X = X | {v}

        bk(set(), set(self._atoms), set())
        return tuple(cliques)

    def _extract_cocycle(self, mul, g, h) -> int:
        """`mul(g, h)` is the oracle product `L_g·L_h`."""
        gh = self._terms(mul(g, h))
        hg = self._terms(mul(h, g))
        (s, cgh), = gh.items()
        chg = hg[s]
        a = min(cgh._coeffs)
        b = min(chg._coeffs)
        assert (a - b) % 2 == 0, f"odd cocycle {g},{h}: {a},{b}"
        return (a - b) // 2

    def _extract_cross(self, mul, g, h):
        """`mul(g, h)` is the oracle product `L_g·L_h`."""
        terms = []
        for s, lp_can in self._terms(mul(g, h)).items():
            gens, powers = self.to_cone_label(s)
            phase = self.cone_label_phase(gens, powers)
            lp = LaurentPoly({e + phase: co for e, co in lp_can._coeffs.items()})
            terms.append((lp, self._word(gens, powers)))
        return tuple(terms)

    def _word(self, gens, powers):
        w = []
        for g in self.canonical_cone_order(gens):
            w.extend([g] * powers[g])
        return tuple(w)

    # ---- ConeData primitives --------------------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def mult_gens(self):
        return self._atoms

    def cones(self):
        return self._cones

    def q_commute(self, g, h) -> bool:
        return g == h or (h in self._nb[g])

    def cocycle(self, g, h) -> int:
        return 0 if g == h else self._qpow[(g, h)]

    def cross_product(self, g, h):
        return self._xprod[(g, h)]

    # ---- cone-label bijection -------------------------------------------
    #
    # A native label `(chord, (c0, c1))` factors as: chord-atoms covering the A6
    # letters (each carrying its magnetic `c0` share), the residual magnetic
    # `c0` supplied by monopole atoms `X_{(1,0)}^{±}`, the residual gauge `c1`
    # by the torus `E`.  (The chord-atoms' own `c1` is part of the residual.)

    def to_cone_label(self, native):
        cached = self._tcl_cache.get(native)
        if cached is not None:
            return cached
        chord, (c0, c1) = native
        ct = Counter()
        for (k, i, e) in chord:
            ct[(k, i)] += e
        # fast path: native is a single (possibly multi-letter) atom × E^k
        atom = self._by_chordc0.get((chord, c0))
        if atom is not None:
            powers = {atom: 1}
            k = c1 - self._sig[atom][1][1]
            if k:
                eg = self._E.get((0, 1) if k > 0 else (0, -1))
                if eg is not None:
                    powers[eg] = abs(k)
                    gens = frozenset(powers)
                    out = (gens, powers)
                    self._tcl_cache[native] = out
                    return out
            else:
                gens = frozenset(powers)
                out = (gens, powers)
                self._tcl_cache[native] = out
                return out
        # general exact charge-cover over the cones with all letter types
        if ct:
            types = list(ct)
            cand_idx = set(self._cone_by_type[types[0]])
            for L in types[1:]:
                cand_idx &= set(self._cone_by_type[L])
            cand_cones = [self._cones[ci] for ci in cand_idx]
        else:
            cand_cones = self._cones
        key_ct = frozenset(ct.items())
        for cone in cand_cones:
            powers = self._cover(key_ct, c0, c1, cone, {})
            if powers is not None:
                gens = frozenset(g for g, p in powers.items() if p > 0)
                out = (gens, {g: p for g, p in powers.items() if p > 0})
                self._tcl_cache[native] = out
                return out
        raise ValueError(f"to_cone_label: cannot factor {native!r}")

    def _cover(self, letters_ct, c0, c1, cone, memo):
        """Exact charge-cover: a dict `{atom: power}` of cone atoms whose
        `(chord-multiset, c0, c1)` sum to `(letters_ct, c0, c1)`, or `None`.
        Chord-bearing atoms (single or multi-letter) cover the letters; the
        residual magnetic `c0` is supplied by monopole atoms, the gauge `c1` by
        `E`."""
        key = (letters_ct, c0, c1)
        if key in memo:
            return memo[key]
        ct = Counter(dict(letters_ct))
        if not ct:                                  # letters covered
            powers: dict = {}
            r0, r1 = c0, c1
            if r0 != 0:
                mg = self._mono.get(1 if r0 > 0 else -1)
                if mg is None or mg not in cone:
                    memo[key] = None
                    return None
                (_, (m0, m1)) = self._sig[mg]
                reps = abs(r0)                      # |m0| == 1
                powers[mg] = reps
                r1 -= m1 * reps
            if r1 != 0:
                eg = self._E.get((0, 1) if r1 > 0 else (0, -1))
                if eg is None or eg not in cone:
                    memo[key] = None
                    return None
                powers[eg] = abs(r1)
            memo[key] = powers
            return powers
        L = next(iter(ct))
        for atom in self._cone_chord_atoms.get(cone, ()):
            act = self._atom_ct[atom]
            if act.get(L, 0) == 0:
                continue
            if any(ct.get(k, 0) < v for k, v in act.items()):
                continue
            rem = Counter(ct)
            rem.subtract(act)
            rem += Counter()                        # drop zero/neg
            (_, (a0, a1)) = self._sig[atom]
            sub = self._cover(frozenset(rem.items()), c0 - a0, c1 - a1, cone, memo)
            if sub is not None:
                out = dict(sub)
                out[atom] = out.get(atom, 0) + 1
                memo[key] = out
                return out
        memo[key] = None
        return None

    def from_cone_label(self, gens, powers):
        c0 = c1 = 0
        cd: dict = {}
        for g, p in powers.items():
            chord, (q0, q1) = g
            c0 += q0 * p
            c1 += q1 * p
            for (k, i, e) in chord:
                cd[(k, i)] = cd.get((k, i), 0) + e * p
        chord = tuple(sorted((k, i, e) for (k, i), e in cd.items() if e))
        return (chord, (c0, c1))

    # ---- QTCone wiring (torus on E only) --------------------------------

    def _torus_inverse_letter(self, g):
        sig = self._sig.get(g)
        if sig is None or sig[0] is not None:
            return None
        inv = {(0, 1): (0, -1), (0, -1): (0, 1)}.get(sig[1])
        return self._E.get(inv) if inv is not None else None

    def iter_cones(self):
        for cone in self._cones:
            yield Cone(self, cone, torus_gens=self._torus_gens & cone)


class U1E7ConeKAlgebra(ConeKAlgebra):
    """u(1)-gauged E7 as a standalone QT-cone ConeKAlgebra (rank-1 gauge torus),
    built from an RG-flow oracle not included in this repository (frozen
    tables → spine-free)."""

    def __init__(self, use_frozen: bool = True):
        self._R = TrivialZPlusRing()
        # spine-free ρ: frozen {(ray_word, c0): (π, c0', δ)} (at c1=0) + ρ²-orbit
        # bound H; any gauge charge c1 follows from the reflection below.  Loaded
        # first, so a table from the (2, 2)-dressed flow is refused before any
        # build work.
        self._rhotab = self._rhoitab = self._Hval = None
        self._load_rho_frozen()
        # build cone-data (frozen tables if available → spine-free, else oracle)
        if use_frozen and os.path.exists(_frozen_path()):
            self._oracle = None
            self._cone_data = U1E7ConeData(None, use_frozen=True)
        else:
            from u1e7_gauged_rg import U1E7GaugedRG   # lazy: oracle-side only
            self._oracle = U1E7GaugedRG()
            self._cone_data = U1E7ConeData(self._oracle, use_frozen=False)
        self._vac_cache: dict = {}
        self._rho_cache: dict = {}
        self._rhoi_cache: dict = {}
        # the neutral-sector route through FiniteE7KAlgebra (built lazily)
        self._e7 = None                    # the FiniteE7KAlgebra instance
        self._e7_cover = None              # per-generator data for `_e7_image`
        self._e7_image_cache: dict = {}    # neutral label -> (zoo label, r)
        self._e7_trace_cache: dict = {}    # (zoo label, K) -> {q: {n: int}}

    # ---- KAlgebra contract ----------------------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return ((), (0, 0))

    def cone_data(self):
        return self._cone_data

    def _oracle_alg(self):
        if self._oracle is None:
            from u1e7_gauged_rg import U1E7GaugedRG   # lazy: oracle-side only
            self._oracle = U1E7GaugedRG()
        return self._oracle

    def _H_value(self):
        return self._Hval if self._Hval is not None else self._oracle_alg()._H

    def rho(self, a):
        """ρ on the contract.  Spine-free via the frozen single-ray `(ray_word,
        c0)` table + the gauge-reflection formula `ρ((w,(c0,c1)))=(π,(c0',δ-c1))`
        for tabulated keys; for a multi-ray *product* canonical (key absent from
        the table) ρ is computed oracle-free by the automorphism rule
        `_rho_via_cone`.  The oracle is consulted only by an instance built from
        it before any ρ table exists (`use_frozen=False`)."""
        v = self._rho_cache.get(a)
        if v is None:
            chord, (c0, c1) = a
            ent = self._rhotab.get((chord, c0)) if self._rhotab is not None else None
            if ent is not None:
                pi, pc0, delta = ent
                v = (pi, (pc0, delta - c1))
            elif self._rhotab is None and self._oracle is not None:
                v = self._oracle.rho(a)
            else:
                v = self._rho_via_cone(a, inverse=False)
            self._rho_cache[a] = v
        return v

    def rho_inverse(self, a):
        v = self._rhoi_cache.get(a)
        if v is None:
            chord, (c0, c1) = a
            ent = self._rhoitab.get((chord, c0)) if self._rhoitab is not None else None
            if ent is not None:
                pi, pc0, delta = ent
                v = (pi, (pc0, delta - c1))
            elif self._rhoitab is None and self._oracle is not None:
                v = self._oracle.rho_inverse(a)
            else:
                v = self._rho_via_cone(a, inverse=True)
            self._rhoi_cache[a] = v
        return v

    def _rho_via_cone(self, a, inverse: bool):
        """ρ (or ρ⁻¹) on a label whose `(ray_word, c0)` is absent from the frozen
        single-ray table -- i.e. a multi-ray *product* canonical.  Oracle-free,
        via the automorphism property: ρ is an algebra automorphism, so

            ρ(L_a) = ρ(∏_g L_g^{p_g}) = ∏_g ρ(L_g)^{p_g},

        where `(g, p_g) = to_cone_label(a)` are the cone mult-gen atoms -- each a
        single ray whose ρ IS tabulated (incl. the torus `E`, `ρ(E)=E⁻¹`).
        Recombine by the (spine-free) cone multiply; ρ permutes the canonical
        basis, so the product is a single basis element `q^k·L_{ρ(a)}` -- return
        its label.  Verified == the oracle on every multi-ray product canonical."""
        from kalgebra import Element
        from laurent_poly import LaurentPoly
        cd = self._cone_data
        _, powers = cd.to_cone_label(a)
        one = LaurentPoly({0: 1})
        acc = Element({self.identity(): one})
        rfn = self.rho_inverse if inverse else self.rho
        for g, p in powers.items():
            if not p:
                continue
            r_img = Element({rfn(cd.from_cone_label(frozenset([g]), {g: 1})): one})
            for _ in range(p):
                acc = self.multiply_elements(acc, r_img)
        labels = [lab for lab, co in acc.terms.items() if not co.is_zero()]
        if len(labels) != 1:
            raise AssertionError(
                f"_rho_via_cone: expected a single basis image for {a!r}, "
                f"got {labels!r}")
        return labels[0]

    def _canonical_rho2_orbit_rep(self, label):
        """Drift-quotient canonicalisation.  Magnetic (`c0 ≠ 0`) seeds have
        infinite ρ²-orbits (ρ² drifts the gauge leg `c1`) but trace to 0, so no
        merge is needed — return as-is.  Every `c0 = 0` seed (ray-generator
        word, multi-ray cone-word, or gauge v-tower) has a FINITE ρ²-orbit (no
        c1-drift) → merge to the orbit minimum, so ρ²-cyclicity is enforced in
        code (the multi-ray seeds the Layer-1 reducer emits must be folded too,
        else cyclicity relations between them are lost)."""
        chord, (c0, c1) = label
        if c0 != 0:
            return label
        members, cur = [], label
        for _ in range(8 * self._H_value() + 12):
            members.append(cur)
            cur = self.rho(self.rho(cur))
            if cur == label:
                return min(members, key=repr)
        return label

    # ---- freeze / load spine-free ρ (ray-word table + reflection) -------

    def _load_rho_frozen(self) -> bool:
        path = _rho_frozen_path()
        if not os.path.exists(path):
            return False
        _refuse_stale_tables(path)
        with open(path, "rb") as f:
            d = pickle.load(f)
        if not {"_rho", "_rhoi", "_H"} <= d.keys():     # stale / wrong format
            return False
        self._rhotab, self._rhoitab, self._Hval = d["_rho"], d["_rhoi"], d["_H"]
        return True

    def freeze_rho(self) -> str:
        """Extract ρ / ρ⁻¹ keyed by `(ray_word, c0)` at `c1 = 0` and the ρ²-orbit
        bound `H` from the oracle, writing `u1e7_rho_tables.pkl` so ρ is
        spine-free.  Only `(π, c0', δ)` per key is stored — any gauge charge `c1`
        follows from `ρ((w,(c0,c1))) = (π,(c0',δ-c1))` (ρ(E)=E⁻¹, ρ an
        automorphism, so the gauge leg always reflects).

        Keys: every cone generator (atom), closed under ρ / ρ⁻¹.  ρ on any other
        label — a product of generators — is the oracle-free automorphism rule
        `_rho_via_cone`, so the table serves every label at any q-order.  (The
        tables built from the earlier `(2, 2)` flow also stored the keys that a
        freeze-time run of the trace bootstrap queried; that tied the ρ table to
        the trace path and is no longer done.)"""
        from u1e7_gauged_rg import U1E7GaugedRG     # lazy: oracle-side only
        from u1e7_cone_derivation import _trim_oracle_caches
        if self._oracle is None:
            self._oracle = U1E7GaugedRG()
        orc = self._oracle
        keys = {((), 0)} | {(a[0], a[1][0]) for a in self._cone_data.mult_gens()}
        rho: dict = {}
        rhoi: dict = {}
        queue = sorted(keys, key=repr)
        while queue:
            ch, c0 = queue.pop()
            if (ch, c0) in rho:
                continue
            _trim_oracle_caches(orc)
            r = orc.rho((ch, (c0, 0)))
            rho[(ch, c0)] = (r[0], r[1][0], r[1][1])
            ri = orc.rho_inverse((ch, (c0, 0)))
            rhoi[(ch, c0)] = (ri[0], ri[1][0], ri[1][1])
            for nb in ((r[0], r[1][0]), (ri[0], ri[1][0])):
                if nb not in rho:
                    queue.append(nb)
        with open(_rho_frozen_path(), "wb") as f:
            pickle.dump({"_rho": rho, "_rhoi": rhoi, "_H": orc._H}, f)
        self._rhotab, self._rhoitab, self._Hval = rho, rhoi, orc._H
        self._rho_cache.clear()
        self._rhoi_cache.clear()
        return _rho_frozen_path()

    # ---- trace (Layer 2) ------------------------------------------------

    def _int_rps(self, coeffs, K):
        """Build an `RPowerSeries` over the trivial ring from `{q: int}`."""
        ub = self._R.one_basis()
        return RPowerSeries(
            self._R,
            {q: RElement(self._R, {ub: int(c)})
             for q, c in coeffs.items() if c and 0 <= q <= K},
            K)

    @staticmethod
    def _mag_charge(label):
        """The magnetic (`X_{(1,0)}`) charge `c0` — ρ²-invariant; the QT trace
        annihilates every `c0 ≠ 0` sector exactly."""
        _chord, (c0, _c1) = label
        return c0

    def _trace_residual(self, seed_label, K):
        """Layer-2 seed value.  `c0 ≠ 0 ⇒ 0` (magnetic, exact); the gauge
        v-tower `E^n` (incl. the identity `Tr(1)`) via the lazy vacuum recipe;
        every other `c0 = 0` label (a neutral chord label) through
        `FiniteE7KAlgebra` (`_neutral_chord_trace`)."""
        if self._mag_charge(seed_label) != 0:
            return RPowerSeries(self._R, {}, K)
        chord, (_c0, c1) = seed_label
        if chord == ():                       # gauge v-tower E^{c1} (incl. Tr(1))
            return self._v_tower_trace(c1, K)
        return self._neutral_chord_trace(seed_label, K)

    # -- the neutral sector through FiniteE7KAlgebra ------------------------

    def _e7_zoo(self):
        """The ungauged `[A₁,E₇]` algebra `FiniteE7KAlgebra` (closed-form traces;
        imported lazily) and the per-generator cover data for `_e7_image`."""
        if self._e7 is None:
            from finite_e7_kalg import FiniteE7KAlgebra, E7_MULT_GENS_LATTICE
            zoo = FiniteE7KAlgebra()
            gens = {}
            for g, gamma in enumerate(E7_MULT_GENS_LATTICE):
                chord, c1 = _E7_GENERATOR_PREIMAGES[tuple(gamma)]
                ct = Counter()
                for (a, i, e) in chord:
                    ct[(a, i)] += e
                gens[g] = (ct, c1)
            if len(gens) != len(_E7_GENERATOR_PREIMAGES):
                raise AssertionError("_E7_GENERATOR_PREIMAGES does not match "
                                     "FiniteE7KAlgebra's generators")
            by_letter: dict = defaultdict(list)
            for g, (ct, _c1) in gens.items():
                for letter in ct:
                    by_letter[letter].append(g)
            self._e7_cover = (gens, dict(by_letter))
            self._e7 = zoo
        return self._e7

    def _e7_image(self, label):
        """`φ(L_ℓ) = μ^r·L_z` for a magnetically neutral chord label
        `ℓ = (chord, (0, c1))`: returns `(z, r)`, `z` a `FiniteE7KAlgebra` label.

        `φ` is linear on each zoo cone — the zoo label `z = ((g, p), …)` is the
        image of the label whose chord multiset is `Σ p·chord(y_g)` and whose
        gauge charge is `Σ p·c1(y_g) + r` — so `z` is found by an exact cover of
        `ℓ`'s chord multiset by the chord multisets of pairwise q-commuting zoo
        generators (`_E7_GENERATOR_PREIMAGES`); the cover is unique because `φ`
        is a bijection of canonical bases (checked on samples in the tests).
        Raises `ValueError` when no cover exists (not a label of this
        algebra)."""
        v = self._e7_image_cache.get(label)
        if v is not None:
            return v
        chord, (c0, c1) = label
        if c0 != 0:
            raise ValueError(f"_e7_image: {label!r} is not magnetically neutral")
        zoo = self._e7_zoo()
        gens, by_letter = self._e7_cover
        qc = zoo.cone_data().q_commute
        target = Counter()
        for (a, i, e) in chord:
            target[(a, i)] += e
        failed: set = set()

        def cover(rem, chosen):
            if not rem:
                return chosen
            key = (frozenset(rem.items()), frozenset(chosen))
            if key in failed:
                return None
            letter = min(rem)
            for g in by_letter.get(letter, ()):
                ct = gens[g][0]
                if any(rem.get(x, 0) < n for x, n in ct.items()):
                    continue
                if not all(qc(g, h) for h in chosen):
                    continue
                nxt = Counter(rem)
                nxt.subtract(ct)
                nxt = +nxt
                got = cover(nxt, chosen | {g})
                if got is not None:
                    powers[g] = powers.get(g, 0) + 1
                    return got
            failed.add(key)
            return None

        powers: dict = {}
        if cover(+target, frozenset()) is None:
            raise ValueError(
                f"_e7_image: no cover of {label!r} by the images of "
                "FiniteE7KAlgebra's generators; not a label of this algebra")
        z = tuple(sorted(powers.items()))
        r = c1 - sum(p * gens[g][1] for g, p in powers.items())
        self._e7_image_cache[label] = (z, r)
        return z, r

    def _neutral_chord_trace(self, label, K):
        """`Tr(L_ℓ) = [μ^{−r}]((𝖖²;𝖖²)_∞²·Tr_{[A₁,E₇]}(L_z; μ))` for a
        magnetically neutral chord label, `φ(L_ℓ) = μ^r·L_z` (`_e7_image`):
        the U(1) gauging of the flavoured `[A₁,E₇]` trace, which
        `FiniteE7KAlgebra` serves from its closed forms (`e7_seeds`)."""
        z, r = self._e7_image(label)
        return self._gauged_zoo_trace(z, r, K)

    def _gauged_zoo_trace(self, z, r, K):
        """`[μ^{−r}]((𝖖²;𝖖²)_∞²·Tr_{[A₁,E₇]}(L_z; μ))` through `𝖖^K`, with the
        `FiniteE7KAlgebra` trace of its label `z` (`()` is the identity, whose
        trace is the closed-form vacuum)."""
        key = (z, K)
        tz = self._e7_trace_cache.get(key)
        if tz is None:
            tr = self._e7_zoo().trace(z, K)
            tz = {}
            for q, v in tr.coeffs.items():
                for kb, c in v.terms.items():
                    if int(c):
                        n = kb[0] if kb else 0
                        tz.setdefault(q, {})[n] = int(c)
            self._e7_trace_cache[key] = tz
        from qpoch import qpoch_infty
        m = qpoch_infty(K)
        m = m * m
        meas = {e: int(c) for e, c in m._c.items() if int(c) and e <= K}
        coeffs: dict = {}
        for q1, md in tz.items():
            c = md.get(-r, 0)
            if not c:
                continue
            if q1 < 0:
                # a trace is a power series; never truncate a negative power away
                raise ValueError(
                    f"_gauged_zoo_trace: the FiniteE7KAlgebra trace of {z!r} has a "
                    f"q^{q1} term; a trace must start at q^0")
            for qe, mc in meas.items():
                q = q1 + qe
                if q <= K:
                    coeffs[q] = coeffs.get(q, 0) + c * mc
        return self._int_rps(coeffs, K)

    # -- lazy vacuum recipe: Tr(E^n) = [μ^{-n}](Tr_E7(1;μ)·(q²;q²)_∞²) ----

    def _vacuum_mu(self, K):
        """`Tr_E7(1; μ)·(q²;q²)_∞²` as `{q_exp: {mu_exp: int}}` to order `K`, with
        `Tr_E7(1; μ)` the Nahm sum on the E7 BPS spectrum.  The witness of the
        E-tower, not its serving path (`_v_tower_trace`)."""
        if K in self._vac_cache:
            return self._vac_cache[K]
        from vacuum_nahm import vacuum_trace_rps, SPECS
        from finite_e7_kalg import E7_BPS_PAIRING
        from zplus_ring import AbelianZPlusRing
        from qpoch import qpoch_infty
        R = AbelianZPlusRing(rank=1)
        vac = vacuum_trace_rps(SPECS["e7"], E7_BPS_PAIRING, R, K)
        vacd: dict = {}
        for q, c in vac.coeffs.items():
            src = getattr(c, "_coeffs", None) or getattr(c, "terms", {})
            for kb, v in src.items():
                n = kb[0] if isinstance(kb, tuple) and kb else 0
                vacd.setdefault(q, {})[n] = vacd.get(q, {}).get(n, 0) + int(v)
        m = qpoch_infty(K)
        m = m * m
        meas = {e: int(c) for e, c in m._c.items()}
        P: dict = {}
        for q1, md in vacd.items():
            for qe, mc in meas.items():
                q = q1 + qe
                if q > K:
                    continue
                for mu, co in md.items():
                    P.setdefault(q, {})[mu] = P.get(q, {}).get(mu, 0) + co * mc
        self._vac_cache[K] = P
        return P

    def _v_tower_trace(self, n, K):
        """`Tr(E^n) = [μ^{-n}](Tr_E7(1;μ)·(q²;q²)_∞²)`, with `Tr_E7(1; μ)` the
        closed-form vacuum `FiniteE7KAlgebra` serves (the Bershadsky–Polyakov
        vacuum product of `e7_seeds`): the identity case `z = ()` of
        the neutral route.  The Nahm sum (`_vacuum_mu`) gives the same series
        (tested through `𝖖¹⁰`) but grows about threefold per two orders
        (30 s at `𝖖¹⁴`), and `trace_element` widens `K` by the negative powers
        of its coefficients, so it is kept as the witness only."""
        return self._gauged_zoo_trace((), n, K)

    def trace(self, a, K=20):
        """`Tr(L_a)` for a single basis cone-word `a`.

        Served: a magnetic label (`c0 ≠ 0`) traces to 0 exactly; the gauge
        v-tower `E^n` (incl. `Tr(1)`) comes from `FiniteE7KAlgebra`'s
        closed-form vacuum; a magnetically neutral chord label (`c0 = 0`,
        non-empty chord) from `FiniteE7KAlgebra` through the label map `φ`
        (`_neutral_chord_trace`; see the module docstring).  Its cost is the zoo's: a single generator
        is a closed form, a composite zoo label goes through the zoo's own
        Layer-1 reduction, which grows fast with the label's depth.

        Layer-1 for this QTCone is pure **ρ²-canonicalisation** (`Tr(ρ²x)=Tr(x)`):
        a basis label is one cone-word ⇒ one seed, so no expansion is needed.
        The generic tagged-cycle reduction `ConeData.simplify_trace_via_cone_data`
        (used by `ConeKAlgebra.trace`) is cyclically INCONSISTENT on this QTCone
        and is therefore bypassed.  A neutral chord label needs no
        canonicalisation: the zoo trace is already ρ²-invariant.  Layer-2 is
        `_trace_residual`.  Multi-term traces go through the inherited
        `trace_element`, which sums this per term.
        """
        chord, (c0, _c1) = a
        if c0 == 0 and chord:
            return self._neutral_chord_trace(a, K)
        return self._trace_residual(self._canonical_rho2_orbit_rep(a), K)


if __name__ == "__main__":
    import time
    from u1e7_gauged_rg import U1E7GaugedRG
    t0 = time.time()
    cd = U1E7ConeData(U1E7GaugedRG(), use_frozen=False)
    print(f"U1E7ConeData built in {time.time()-t0:.1f}s: "
          f"{len(cd.mult_gens())} atoms, {len(cd.cones())} cones")
    cd.freeze()
    print("frozen ->", _frozen_path())
    # spine-free ρ tables (ray-word reflection): exact at any q-order.  The ρ
    # table is rebuilt from scratch against the new cone table; an older one is
    # removed first (it is never read back).
    if os.path.exists(_rho_frozen_path()):
        os.remove(_rho_frozen_path())
    t1 = time.time()
    K = U1E7ConeKAlgebra(use_frozen=True)
    K.freeze_rho()
    print(f"ρ tables frozen in {time.time()-t1:.1f}s -> {_rho_frozen_path()} "
          f"({len(K._rhotab)} (ray-word, c0) keys, H={K._Hval})")
