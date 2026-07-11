"""`SkeinAtlas` — the unified-class atlas: an ensemble of
**`SkeinKAlgebra` charts** of one `Sk(Σ)`, indexed by ideal
triangulations and related by diagonal FLIPS.

Relation to the existing atlases
--------------------------------
`SkeinSphereAtlas` / `BorderedSkeinAtlas` are the
BPS-chart-level atlases: nodes = triangulations, transitions certified
flip == mutation (FST) per edge, spec transported along flips.
`SkeinAtlas` WRAPS that machinery (composition, not inheritance) and
lifts every chart to a first-class **`SkeinKAlgebra`**
(`chart_skein(key)`), with certified `KAlgebraIso` transitions between
the lifted charts (`iso_skein(src, dst)`).

The per-chart `SkeinKAlgebra` (closed surfaces, v1)
---------------------------------------------------
Each chart carries the **multicurve-component cone data**
(`MulticurveChartSkeinConeData`): on the chart's canonical charge
labels `γ ∈ Z^E`,

  * rays = the connected components of the multicurve `−γ` — core
    simple closed curves (**monomial** rays: the multicurve basis is
    coordinate-additive on parallel copies) and the peripheral
    (flavour) loops as **torus** ray pairs `±f_p` (ker-σ central
    units, `X_{f}·X_{−f} = 1` exactly);
  * `q_commute` = geometric disjointness, decided EXACTLY by component
    arithmetic: `m_g + m_h` is the disjoint union iff its component
    multiset is the union of the two component multisets;
  * `cocycle` = the chart Dirac pairing `⟨g, h⟩ = Σ g_i σ_ij h_j`;
  * the cone family is infinite and lazy — `cone_of_label` constructs
    the label's own cone directly (never scans);
  * label-bijection certified scope: pure multicurve classes (`−γ`
    admissible; includes negative-flavour dressing) and pure flavour
    monomials (`γ ∈ span_Z{f_p}`, either sign).  Labels mixing a core
    curve with POSITIVE flavour content honest-fail (`ValueError`) —
    off the component dictionary, recorded future work.

**The measured tower law, and why `multiply` is
engine-anchored:** on the tetrahedron root chart the exact canonical
product of a channel curve with itself is

    L_γ · L_γ  =  L_{2γ} + L_γ + 1        (unit coefficients)

— the 3-term GNO fusion, NOT the 2-term
Chebyshev `χ₁² = χ₂ + 1` of the character-normalized sample class.
On this Z-form chart presentation same-ray towers are therefore
neither Chebyshev ("char") nor word-monomial foldable, so the derived
cone reducer does not serve them; per the blessed engine-override
precedent (`PureSU2AnnulusSkeinKAlgebra`) the chart instances anchor
`multiply` on the canonical chart engine (the transported-spec
`SkeinSphereKAlg` = exact BPS canonical multiply), and the cone data
serves the presentation (bijection, disjointness, phases).

Charts carry the full two-parent substrate: `chart=` is the
chart's own `σ_Δ` + flip-transported spec (no auto-find off the
root), `to_bpskalgebra()` returns the wrapped native twin, and the
flip transitions certify against the wrapped atlas's flip == mutation
guard on every edge.

Bordered charts (the polygon A1An roster)
------------------------------------------
`SkeinAtlas` accepts bordered charts too (wrapping
`BorderedSkeinAtlas` — flips at internal edges, both FST
certifications per transition, spec transported).  The lift uses
**`TransportedSkeinConeData`**: a chart's ray dictionary is the ROOT
ray dictionary pushed through the certified atlas transition (label
maps asserted single-label / unit-coefficient — the canonical basis is
transition-invariant), so rays stay "the chords of the polygon" in
every chart's own frame while `q_commute` (geometric disjointness) and
the cone family ride along unchanged.  `SkeinAtlas.polygon(5)` wires
the pentagon: the root dictionary is the roster
`PentagonSkeinConeData` under the node-swap frame map (the bordered
fan chart's mutable block is the roster chart with nodes swapped —
measured, structure constants match exactly under `(a,b) ↦ (b,a)`),
with the PINNED engine behind the root cross-products; flipped charts
anchor cross-products on their transported-spec twins.  The
associahedron 5-flip loop is certified at the LIFTED level: the
composed `iso_skein` transition equals ρ² (read through the loop
chart's node basis), `ord(ρ) = 5 = h+2` — the fractional quantum
monodromy on `SkeinKAlgebra` charts.  Other polygons: flips and
BPS-level isos work; `chart_skein` honest-fails until their root ray
dictionaries are wired.
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

from skein_cone_data import SkeinConeData, fraction_solve
from skein_kalgebra import SkeinKAlgebra
from skein_sphere_atlas import SkeinSphereAtlas
from triangulation import Triangulation

_ONE = LaurentPoly({0: 1})

__all__ = ["SkeinAtlas", "MulticurveChartSkeinConeData",
           "TransportedSkeinConeData"]


class TransportedSkeinConeData(SkeinConeData):
    """A chart's cone dictionary TRANSPORTED from a base ray dictionary
    through bijective NATIVE-LABEL maps (`to_base` / `from_base` — e.g.
    the certified atlas transition read as a label bijection, or a
    frame swap).  The base's cone LETTERS may be abstract (the finite
    zoo's mg-indices): the chart-side letters are always the rays'
    CHART CHARGES, obtained by pushing each base letter's
    single-generator label through `from_base` (memoized both ways).
    Rays, cones, the bijection, disjointness and cocycles all ride
    through the maps; `engine_product` is supplied in the CHART's
    native labels (the standing guard lives in that engine)."""

    def __init__(self, base_cd, to_base, from_base, engine_product_fn):
        self._base = base_cd
        self._to = to_base
        self._from = from_base
        self._engine_fn = engine_product_fn
        self._letter_fwd: dict = {}     # base letter -> chart ray charge
        self._letter_inv: dict = {}     # chart ray charge -> base letter

    # -- the letter dictionary -------------------------------------------------

    def _chart_ray(self, letter):
        if letter not in self._letter_fwd:
            lbl = self._base.from_cone_label(frozenset({letter}),
                                             {letter: 1})
            r = tuple(self._from(lbl))
            # injectivity guard: two base
            # letters pushing to the same chart charge would silently
            # overwrite the inverse dictionary and redirect every
            # cocycle/q_commute lookup through the wrong letter
            prev = self._letter_inv.get(r)
            assert prev is None or prev == letter, (
                f"transported letter dictionary is not injective: base "
                f"letters {prev} and {letter} both push to chart "
                f"charge {r}")
            self._letter_fwd[letter] = r
            self._letter_inv[r] = letter
        return self._letter_fwd[letter]

    def _base_letter(self, ray):
        ray = tuple(ray)
        if ray not in self._letter_inv:
            for g in self._base.mult_gens():
                self._chart_ray(g)
            if ray not in self._letter_inv:
                raise ValueError(
                    f"transported dictionary: {ray} is not a ray charge"
                )
        return self._letter_inv[ray]

    # -- SkeinConeData surface ------------------------------------------------

    def coefficient_ring(self):
        """DELEGATED (the inherited Trivial
        default would hand a flavoured base's `RLaurent` cross-products
        to bare-`LaurentPoly` `_q_one` arithmetic — all current bases
        are Trivial, but the default path must be right for the next
        base wired in)."""
        return self._base.coefficient_ring()

    def ray_kind(self, g) -> str:
        if hasattr(self._base, "ray_kind"):
            return self._base.ray_kind(self._base_letter(g))
        # plain-ConeData base (e.g. the finite zoo's chord dictionaries):
        # guard the char direction FIRST (a χ
        # ray classified "monomial" would make the char-aware reducer
        # fold χ towers as literal monomial words: well-formed wrong
        # products, no error), then: torus-inverse pairs are torus
        # rays, everything else is an open chord (monomial).
        letter = self._base_letter(g)
        if not hasattr(self, "_base_char_letters"):
            # iter_cones() yields Cone OBJECTS (cones() yields the raw
            # letter frozensets); the fallback fires only for finite
            # plain-ConeData bases, so the full sweep terminates
            self._base_char_letters = (
                frozenset().union(
                    *(c.char_gens() for c in self._base.iter_cones()))
                if hasattr(self._base, "iter_cones") else frozenset())
        if letter in self._base_char_letters:
            raise NotImplementedError(
                f"transported dictionary: base letter {letter} is a "
                f"CHARACTER generator — χ transport through the letter "
                f"dictionary is not implemented; "
                f"give the base a ray_kind or wire a char-aware "
                f"dictionary")
        if self._torus_inverse_letter(g) is not None:
            return "torus"
        return "monomial"

    def cone_sets(self):
        src = getattr(self._base, "cone_sets", None)
        if src is None:
            src = self._base.cones          # FiniteConeData surface
        for c in src():
            yield frozenset(self._chart_ray(g) for g in c)

    def mult_gens(self):
        return tuple(self._chart_ray(g) for g in self._base.mult_gens())

    def engine_product(self, g, h) -> Element:
        return self._engine_fn(tuple(g), tuple(h))

    def q_commute(self, g, h) -> bool:
        return self._base.q_commute(self._base_letter(g),
                                    self._base_letter(h))

    def cocycle(self, g, h) -> int:
        return self._base.cocycle(self._base_letter(g),
                                  self._base_letter(h))

    def canonical_cone_order(self, gens):
        order = self._base.canonical_cone_order(
            frozenset(self._base_letter(g) for g in gens))
        return tuple(self._chart_ray(g) for g in order)

    def canonicalize_cone_label(self, cone, gens, powers):
        """Delegated through the letter dictionary — the base overrides
        carry the non-simplicial / torus-direction identifications (the
        finite zoo, the QTCone families)."""
        base_cone = frozenset(self._base_letter(g) for g in cone)
        g2, p2, phase = self._base.canonicalize_cone_label(
            base_cone,
            frozenset(self._base_letter(g) for g in gens),
            {self._base_letter(g): p for g, p in powers.items()})
        return (frozenset(self._chart_ray(g) for g in g2),
                {self._chart_ray(g): p for g, p in p2.items()},
                phase)

    def to_cone_label(self, native_label):
        gens, powers = self._base.to_cone_label(self._to(tuple(native_label)))
        # normalize: drop zero powers (some bases emit the identity as
        # an all-zero-power cone) — gens must be the keys of powers
        out_p = {self._chart_ray(g): p for g, p in powers.items() if p}
        return frozenset(out_p), out_p

    def from_cone_label(self, gens, powers):
        return tuple(self._from(self._base.from_cone_label(
            frozenset(self._base_letter(g) for g in gens),
            {self._base_letter(g): p for g, p in powers.items()})))

    def _torus_inverse_letter(self, g):
        r = self._base._torus_inverse_letter(self._base_letter(g))
        return None if r is None else self._chart_ray(r)


class MulticurveChartSkeinConeData(SkeinConeData):
    """The multicurve-component cone data of one triangulation chart
    (see the module docstring for the dictionary and certified scope).

    `engine_product` is the chart's canonical multiply (the
    transported-spec `SkeinSphereKAlg`) — called for crossing core
    pairs and the flavour cancellations."""

    def __init__(self, chart_kalg):
        self._A = chart_kalg
        self.tri = chart_kalg.triangulation
        self._sigma = self.tri.sigma()
        self._n = self.tri.n_edges
        self._fps = [tuple(self.tri.puncture_flavour_charge(p))
                     for p in range(self.tri.n_punctures)]

    # -- multicurve arithmetic (exact, cached) --------------------------------

    def _components(self, coords):
        if not hasattr(self, "_comp_cache"):
            self._comp_cache = {}
        coords = tuple(coords)
        if coords not in self._comp_cache:
            from multicurve import components_of_coords
            self._comp_cache[coords] = sorted(
                components_of_coords(self.tri, coords))
        return self._comp_cache[coords]

    def _flavour_combo(self, gamma):
        """`gamma` as an integer combination of the peripheral classes
        `f_p`, or None.  Shared exact solver (`fraction_solve`);
        integrality + verify-by-substitution kept local."""
        cols = self._fps
        if not cols:
            return None
        got = fraction_solve(cols, list(gamma), dim=self._n)
        if got is None:
            return None
        sol, _ = got
        if any(s.denominator != 1 for s in sol):
            return None
        combo = tuple(int(s) for s in sol)
        # verify (free variables were zeroed; the combo must reproduce γ)
        if tuple(sum(combo[j] * cols[j][i] for j in range(len(cols)))
                 for i in range(self._n)) != tuple(gamma):
            return None
        return combo

    def _is_peripheral_class(self, coords) -> bool:
        return tuple(coords) in self._fps

    # -- SkeinConeData surface ------------------------------------------------

    def ray_kind(self, g) -> str:
        g = tuple(g)
        pos = tuple(x for x in g)
        if self._flavour_combo(pos) is not None:
            return "torus"
        return "monomial"

    def _torus_inverse_letter(self, g):
        g = tuple(g)
        if self._flavour_combo(g) is not None:
            return tuple(-x for x in g)
        return None

    def cone_sets(self):
        # the flavour torus pairs; core cones are constructed directly
        # from the label (`cone_of_label`) — the family is infinite.
        for f in self._fps:
            yield frozenset({f, tuple(-x for x in f)})

    def engine_product(self, g, h) -> Element:
        return self._A.multiply(tuple(g), tuple(h))

    def _cone_for_gens(self, gens_fs):
        gens_fs = frozenset(tuple(g) for g in gens_fs)
        return self._make_cone(gens_fs)

    def cone_of_label(self, native_label):
        gens_fs, _ = self.to_cone_label(native_label)
        if not gens_fs:
            gens_fs = frozenset(
                {self._fps[0], tuple(-x for x in self._fps[0])})
        return self._make_cone(gens_fs)

    # -- ConeData surface -------------------------------------------------------

    def q_commute(self, g, h) -> bool:
        g, h = tuple(g), tuple(h)
        if g == h:
            return True
        if (self._flavour_combo(g) is not None
                or self._flavour_combo(h) is not None):
            return True                    # flavour is ker-σ central
        mg = tuple(-x for x in g)
        mh = tuple(-x for x in h)
        union = tuple(a + b for a, b in zip(mg, mh))
        return (self._components(union)
                == sorted(self._components(mg) + self._components(mh)))

    def cocycle(self, g, h) -> int:
        if not self.q_commute(g, h):
            raise ValueError(f"cocycle: ({g}, {h}) not q-commuting")
        g, h = tuple(g), tuple(h)
        return sum(g[i] * self._sigma[i][j] * h[j]
                   for i in range(self._n) for j in range(self._n))

    def canonical_cone_order(self, gens):
        return tuple(sorted(tuple(g) for g in gens))

    def to_cone_label(self, native_label):
        gamma = tuple(native_label)
        if not any(gamma):
            return frozenset(), {}
        combo = self._flavour_combo(gamma)
        if combo is not None:
            powers = {}
            for p, c in enumerate(combo):
                if c > 0:
                    powers[self._fps[p]] = c
                elif c < 0:
                    powers[tuple(-x for x in self._fps[p])] = -c
            return frozenset(powers), powers
        m = tuple(-x for x in gamma)
        if any(x < 0 for x in m):
            raise ValueError(
                f"chart cone dictionary: label {gamma} mixes a core curve "
                f"with positive flavour content — off the component "
                f"dictionary's certified scope (pure multicurve classes "
                f"and pure flavour monomials); recorded future work"
            )
        powers: dict = {}
        for comp in self._components(m):
            if self._is_peripheral_class(comp):
                ray = tuple(-x for x in comp)      # canonical X_{-f}
            else:
                ray = tuple(-x for x in comp)      # core class label
            powers[ray] = powers.get(ray, 0) + 1
        return frozenset(powers), powers

    def from_cone_label(self, gens, powers):
        out = [0] * self._n
        for g, p in powers.items():
            if not p:
                continue
            for i, x in enumerate(tuple(g)):
                out[i] += p * x
        return tuple(out)


def _a1a2k_root_cd(k: int, tri: Triangulation,
                   twin_multiply_fn) -> TransportedSkeinConeData:
    """The uniform odd-polygon root dictionary (the `(2k+3)`-gon,
    `[A₁, A_{2k}]`): the `A1A2kKAlg(k)` chord dictionary on the
    bordered chart frame, with

      * charges from the closed-form BPS-free ρ-orbit seeds
        (`a1a2k_bps_iso._compute_chord_charges` — candidate orbit
        alignments: the natural shift-0, plus the recorded empirical
        `{2: 3}` at k = 2);
      * `from_base` LINEAR on cone monomials, `to_base` the fan solve
        over the `A1A2kConeData` cone family (chord powers ≥ 0, no
        Laurent direction), exactness asserted;
      * the frame identification (σ-matching node permutation × the
        global tropical flip) AND the orbit alignment selected
        together by a LIVE product check — the family meson through
        the candidate maps + the PINNED engine must equal the bordered
        twin — with the full ray-grid battery re-certifying
        downstream;
      * cross-products from the generic-n PINNED engine
        (`SkeinOddPolygonKAlg`, mapped through the label dictionary).
    """
    from itertools import permutations

    from a1a2k_bps_iso import _compute_chord_charges, _a4_pairing
    from a1a2k_kalg import A1A2kKAlg

    from skein_oddgon_kalg import SkeinOddPolygonKAlg

    intr = A1A2kKAlg(k)
    engine = SkeinOddPolygonKAlg(intrinsic=intr)
    cd = intr.cone_data()
    cones = [frozenset(c) for c in cd.cones()]
    n = 2 * k
    mb = tri.mutable_block()
    sig_std = _a4_pairing(n)

    def _maps_for(CH, perm, eps):
        inv_perm = [0] * n
        for i, p_ in enumerate(perm):
            inv_perm[p_] = i
        # chord letter -> bordered-frame charge
        CHB = {g: tuple(eps * c[inv_perm[j]] for j in range(n))
               for g, c in CH.items()}
        cache: dict = {}

        def from_base(lbl):
            out = [0] * n
            for (a, i, e) in lbl:
                c = CHB[(a, i)]
                for j in range(n):
                    out[j] += e * c[j]
            return tuple(out)

        def _solve_in_cone(cone, gamma):
            chords = sorted(cone)
            cols = [CHB[g] for g in chords]
            got = fraction_solve(cols, list(gamma), dim=n)
            if got is None:
                return None
            sol, _ = got
            if any(x.denominator != 1 or x < 0 for x in sol):
                return None
            return {g: int(x) for g, x in zip(chords, sol) if x}

        def to_base(gamma):
            gamma = tuple(gamma)
            if gamma in cache:
                return cache[gamma]
            if not any(gamma):
                cache[gamma] = intr.identity()
                return cache[gamma]
            for cone in cones:
                powers = _solve_in_cone(cone, gamma)
                if powers is None:
                    continue
                gens = frozenset(powers)
                g2, p2, _ph = cd.canonicalize_cone_label(cone, gens,
                                                         dict(powers))
                # the canonicalization phase is part of the label
                # identification (L = q^ph · L'); a non-zero phase here
                # would silently equate elements differing by q^ph
                # — no live base carries one,
                # and the day one does this map must propagate it
                assert _ph == 0, (
                    f"a1a2k chord dictionary: canonicalization of "
                    f"{dict(powers)} in cone {sorted(cone)} carries a "
                    f"non-zero q-phase {_ph} — propagate it before "
                    f"using this base")
                lbl = cd.from_cone_label(frozenset(p2), dict(p2))
                if from_base(lbl) == gamma:
                    cache[gamma] = lbl
                    return lbl
            raise ValueError(
                f"a1a2k chord dictionary (k={k}): {gamma} is not a "
                f"non-negative chord combination in any cone — off the "
                f"certified fan"
            )

        return to_base, from_base

    def _engine_via(to_base, from_base):
        def _engine(g, h):
            el = engine.multiply(to_base(g), to_base(h))
            out: dict = {}
            for l, c in el.terms.items():
                l2 = from_base(l)
                out[l2] = (out[l2] + c) if l2 in out else c
            return Element(out)
        return _engine

    shift_candidates = [dict()] + ([{2: 3}] if k == 2 else [])
    meson = (((1, 0, 1),), ((1, 1, 1),))     # adjacent family-1 chords
    last_err = None
    for shifts in shift_candidates:
        CH = {g: tuple(c)
              for g, c in _compute_chord_charges(k, shifts).items()}
        for eps in (1, -1):
            for perm in permutations(range(n)):
                if not all(sig_std[perm[i]][perm[j]] == mb[i][j]
                           for i in range(n) for j in range(n)):
                    continue
                to_base, from_base = _maps_for(CH, perm, eps)
                try:
                    ga = from_base(meson[0])
                    gb = from_base(meson[1])
                    got = _engine_via(to_base, from_base)(ga, gb)
                    want = twin_multiply_fn(ga, gb)
                    if dict(got.terms) != dict(want.terms):
                        last_err = f"meson mismatch at {shifts}/{perm}/{eps}"
                        continue
                except (ValueError, AssertionError, KeyError) as exc:
                    last_err = exc
                    continue
                return TransportedSkeinConeData(
                    cd, to_base=to_base, from_base=from_base,
                    engine_product_fn=_engine_via(to_base, from_base))
    raise RuntimeError(
        f"_a1a2k_root_cd(k={k}): no orbit-alignment × frame candidate "
        f"passes the live meson check (last: {last_err})"
    )


def _linear_charge_maps(intr):
    """The charge dictionary of a gauged even-polygon intrinsic
    (`u1a1aodd_general.U1A1AoddKAlg` — bootstrap `chord_charges` /
    `E_charge` — or `u1a1aodd_kalg.U1A1AoddKAlg` — the oracle-extracted
    `_chg` / `_MU` on its cone data; same gauged-quiver frame, same
    alternating E):  `from_base` is LINEAR on cone monomials
    (`Σ e·charge[(a,i)] + e_E·E` — cluster charges add within a cone),
    `to_base` is the fan solve over the cone family (chord powers ≥ 0,
    the E direction Laurent), canonicalized through the base's
    `canonicalize_cone_label` and verified exactly (`from_base(label)
    == γ` asserted — honest-fail off the fan)."""

    cd = intr.cone_data()
    if hasattr(intr, "chord_charges"):          # u1a1aodd_general
        E = tuple(intr.E_charge)
        CH = {g: tuple(c) for g, c in intr.chord_charges.items()}
        E_GEN, E_INV = cd.E_GEN, cd.E_INV
    else:                                        # u1a1aodd_kalg (oracle)
        E = tuple(cd._MU)
        CH = {g: tuple(c) for g, c in cd._chg.items()}
        from u1a1aodd_kalg import E_GEN, E_INV
    dim = len(E)
    cones = [frozenset(c) for c in cd.cones()]
    cache: dict = {}

    def from_base(lbl):
        factors, e_E = lbl
        out = [e_E * E[j] for j in range(dim)]
        for (a, i, e) in factors:
            c = CH[(a, i)]
            for j in range(dim):
                out[j] += e * c[j]
        return tuple(out)

    def _solve_in_cone(cone, gamma):
        chords = sorted(g for g in cone if g not in (E_GEN, E_INV))
        cols = [CH[g] for g in chords] + [E]
        got = fraction_solve(cols, list(gamma), dim=dim)
        if got is None:
            return None
        sol, _ = got
        if any(s.denominator != 1 for s in sol):
            return None
        xs = [int(s) for s in sol]
        if any(x < 0 for x in xs[:-1]):        # chords >= 0; E Laurent
            return None
        powers = {g: x for g, x in zip(chords, xs[:-1]) if x}
        e_E = xs[-1]
        return powers, e_E

    def to_base(gamma):
        gamma = tuple(gamma)
        if gamma in cache:
            return cache[gamma]
        if not any(gamma):
            cache[gamma] = intr.identity()
            return cache[gamma]
        for cone in cones:
            got = _solve_in_cone(cone, gamma)
            if got is None:
                continue
            powers, e_E = got
            gens = frozenset(powers)
            g2, p2, _ph = cd.canonicalize_cone_label(cone, gens,
                                                     dict(powers))
            # see the a1a2k twin above: a dropped canonicalization
            # phase is a silent q^ph identification
            assert _ph == 0, (
                f"u1a1aodd charge dictionary: canonicalization of "
                f"{dict(powers)} carries a non-zero q-phase {_ph} — "
                f"propagate it before using this base")
            full_p = dict(p2)
            if e_E > 0:
                full_p[E_GEN] = full_p.get(E_GEN, 0) + e_E
            elif e_E < 0:
                full_p[E_INV] = full_p.get(E_INV, 0) - e_E
            lbl = cd.from_cone_label(frozenset(full_p), full_p)
            if from_base(lbl) == gamma:
                cache[gamma] = lbl
                return lbl
        raise ValueError(
            f"u1a1aodd charge dictionary: {gamma} is not a non-negative "
            f"chord combination (+ Laurent E) in any cone — off the "
            f"certified fan"
        )

    return to_base, from_base


class A1DoddChartSkeinConeData(SkeinConeData):
    """The `[A₁, D_{2k+3}]` chart dictionary on the fork-quiver charge
    labels (the FIRST FLAVOURED atlas dictionary): rays = the `(a,p,i)`
    arc charges (monomial) + the SU(2)-flavour kernel direction as a
    torus pair `±v_f` (the Z-form: flavour weights ride as label
    coordinates); word combinatorics delegate to the intrinsic
    `a1dodd_cone_data` through the charge↔letter dictionary; the
    engine is the roster `A1DoddSkeinKAlgebra` multiply with each
    `χ_κ`-dressed output term expanded by the EXACT SU(2) weight
    dictionary `χ_κ·L_γ = Σ_{w=−κ,−κ+2,…,κ} L_{γ + w·v_f}` (the
    flavour-in-labels Z-form; unit weights — certified against the
    fork-quiver twin by the atlas battery, never assumed silently)."""

    def __init__(self, intrinsic_kalg, arc_charges: dict, v_f: tuple,
                 sigma):
        self._alg = intrinsic_kalg            # roster A1DoddSkeinKAlgebra
        self._int = intrinsic_kalg.intrinsic.cone_data()
        self._CH = {g: tuple(c) for g, c in arc_charges.items()}
        self._vf = tuple(v_f)
        self._sigma = sigma
        self._n = len(v_f)
        self._letter_of = {c: g for g, c in self._CH.items()}
        self._cones = [frozenset(c) for c in self._int.cones()]
        self._cache: dict = {}

    # -- charge arithmetic ------------------------------------------------------

    def _word_charge(self, word):
        out = [0] * self._n
        for (a, p, i, e) in ((*g, e) for g, e in word):
            c = self._CH[(a, p, i)]
            for j in range(self._n):
                out[j] += e * c[j]
        return out

    def _flavour_mult(self, gamma):
        """`gamma` as `w·v_f`, or None."""
        nz = [j for j, x in enumerate(self._vf) if x]
        if not nz:
            return None
        j0 = nz[0]
        if gamma[j0] % self._vf[j0]:
            return None
        w = gamma[j0] // self._vf[j0]
        if tuple(w * x for x in self._vf) == tuple(gamma):
            return w
        return None

    # -- SkeinConeData surface ------------------------------------------------

    def ray_kind(self, g) -> str:
        if self._flavour_mult(tuple(g)) is not None:
            return "torus"
        return "monomial"

    def _torus_inverse_letter(self, g):
        w = self._flavour_mult(tuple(g))
        if w is not None:
            return tuple(-x for x in g)
        return None

    def mult_gens(self):
        return tuple(sorted(self._letter_of)) + (
            self._vf, tuple(-x for x in self._vf))

    def cone_sets(self):
        fpair = {self._vf, tuple(-x for x in self._vf)}
        for c in self._int.cones():
            yield frozenset(self._CH[g] for g in c) | fpair

    def engine_product(self, g, h) -> Element:
        """Arc-pair product from the roster engine, each output term
        `(word, κ)` expanded by the exact SU(2) weight dictionary."""
        ga = self._letter_of[tuple(g)]
        gb = self._letter_of[tuple(h)]
        wa = self._int.from_cone_label(frozenset({ga}), {ga: 1})
        wb = self._int.from_cone_label(frozenset({gb}), {gb: 1})
        el = self._alg.multiply((wa, 0), (wb, 0))
        out: dict = {}
        for (word, kappa), c in el.terms.items():
            base = self._word_charge(word)
            for w in range(-kappa, kappa + 1, 2):
                lbl = tuple(base[j] + w * self._vf[j]
                            for j in range(self._n))
                out[lbl] = (out[lbl] + c) if lbl in out else c
        return Element(out)

    # -- ConeData surface -------------------------------------------------------

    def q_commute(self, g, h) -> bool:
        g, h = tuple(g), tuple(h)
        if g == h:
            return True
        if (self._flavour_mult(g) is not None
                or self._flavour_mult(h) is not None):
            return True                    # flavour is central (ker σ)
        return self._int.q_commute(self._letter_of[g], self._letter_of[h])

    def cocycle(self, g, h) -> int:
        if not self.q_commute(g, h):
            raise ValueError(f"cocycle: ({g}, {h}) not q-commuting")
        g, h = tuple(g), tuple(h)
        return sum(g[i] * self._sigma[i][j] * h[j]
                   for i in range(self._n) for j in range(self._n))

    def canonical_cone_order(self, gens):
        arcs = [g for g in gens if self._flavour_mult(tuple(g)) is None]
        flav = [g for g in gens if self._flavour_mult(tuple(g)) is not None]
        order = self._int.canonical_cone_order(
            frozenset(self._letter_of[tuple(g)] for g in arcs))
        return tuple(self._CH[l] for l in order) + tuple(sorted(flav))

    def to_cone_label(self, gamma):
        """Fan solve: arcs (powers ≥ 0) + the flavour weight (Laurent)."""
        gamma = tuple(gamma)
        if gamma in self._cache:
            return self._cache[gamma]
        if not any(gamma):
            self._cache[gamma] = (frozenset(), {})
            return self._cache[gamma]
        w0 = self._flavour_mult(gamma)
        if w0 is not None:                 # pure flavour monomial
            ray = self._vf if w0 > 0 else tuple(-x for x in self._vf)
            powers = {ray: abs(w0)}
            self._cache[gamma] = (frozenset(powers), powers)
            return self._cache[gamma]
        for cone in self._cones:
            chords = sorted(cone)
            cols = [self._CH[g] for g in chords] + [self._vf]
            got = fraction_solve(cols, list(gamma), dim=self._n)
            if got is None:
                continue
            sol, _ = got
            if any(x.denominator != 1 for x in sol):
                continue
            xs = [int(x) for x in sol]
            if any(x < 0 for x in xs[:-1]):    # arcs >= 0; v_f Laurent
                continue
            powers = {self._CH[g]: x
                      for g, x in zip(chords, xs[:-1]) if x}
            w = xs[-1]
            if w > 0:
                powers[self._vf] = w
            elif w < 0:
                powers[tuple(-x for x in self._vf)] = -w
            # exactness guard
            back = self.from_cone_label(frozenset(powers), powers)
            if tuple(back) == gamma:
                self._cache[gamma] = (frozenset(powers), powers)
                return self._cache[gamma]
        raise ValueError(
            f"a1dodd chart dictionary: {gamma} is not an arc+flavour "
            f"combination in any cone — off the certified fan"
        )

    def from_cone_label(self, gens, powers):
        out = [0] * self._n
        for g, p_ in powers.items():
            if not p_:
                continue
            for j, x in enumerate(tuple(g)):
                out[j] += p_ * x
        return tuple(out)


class _GaugedPolygonAtlasCore:
    """Internal wrapped-atlas core for the U1A1odd (gauged even-polygon)
    family: a `BPSAtlas` over the GAUGED quiver chart (mutable path
    nodes inside the ambient lattice carrying the frozen flavour / E
    direction), with fan-triangulation flip bookkeeping.  Charts and
    transitions stay in the ROOT frame (no node-basis
    recoordinatization — the ambient lattice is shared), so
    `chart_iso(key)` is just the atlas transition.

    Every flip certifies flip == mutation against the flipped fan
    triangulation's mutable block under the measured orientation
    dictionary (gauged node pairing == −(fan mutable block), constant
    across charts)."""

    def __init__(self, root_bps, tri=None, max_local_moves: int = 8):
        from bps_atlas import BPSAtlas
        self.atlas = BPSAtlas(root_bps, max_local_moves=max_local_moves)
        self.root_kalg = root_bps
        self._tri: dict = {(): tri}
        if tri is None:
            # QUIVER-LEVEL atlas (no triangulation bookkeeping): every
            # node mutation is a transition — for tagged-arc surfaces
            # (regular interior punctures) this covers the FULL
            # exchange graph, with the chart ↔ tagged-triangulation
            # dictionary the recorded follow-up.
            self._eps = None
            return
        sig = self._node_pairing(())
        mb = tri.mutable_block()
        if sig == [[-x for x in row] for row in mb]:
            self._eps = -1
        elif sig == mb:
            self._eps = 1
        else:
            raise RuntimeError(
                "gauged atlas: root node pairing matches neither ±(fan "
                "mutable block) under the identity node↔edge map"
            )

    def _node_pairing(self, key):
        A = self.atlas.chart(key)
        nodes = [tuple(g) for g in A.node_charges]
        br = A.lattice.bracket
        n = len(nodes)
        return [[br(nodes[i], nodes[j]) for j in range(n)]
                for i in range(n)]

    def keys(self):
        return list(self._tri)

    def triangulation(self, key=()):
        from bps_atlas import BPSAtlas
        return self._tri[BPSAtlas._norm_key(key)]

    def chart(self, key=()):
        return self.atlas.chart(key)

    def chart_kalg(self, key=()):
        return self.atlas.chart(key)

    def iso(self, src=(), dst=()):
        return self.atlas.iso(src, dst)

    def chart_iso(self, key=()):
        from bps_atlas import BPSAtlas
        key = BPSAtlas._norm_key(key)
        if not key:
            return KAlgebraIso.identity(self.root_kalg,
                                        name="gauged-atlas[root]")
        return self.atlas.iso((), key)

    def flip(self, edge: int, key=()):
        """Flip internal fan edge `edge` = mutate the gauged node at the
        same index (the identity node↔edge map, certified per flip
        against the flipped triangulation; quiver-level atlases —
        `tri=None` — mutate without triangulation certification)."""
        from bps_atlas import BPSAtlas
        key = BPSAtlas._norm_key(key)
        if self._tri[key] is None:
            last = None
            for d in ("fwd", "inv"):
                try:
                    new_key, iso = self.atlas.mutate(key, edge, d)
                    break
                except ValueError as exc:
                    last = exc
            else:
                raise last
            self._tri[new_key] = None
            return new_key, iso
        tri2 = self._tri[key].flip(edge)
        last = None
        for d in ("fwd", "inv"):
            try:
                new_key, iso = self.atlas.mutate(key, edge, d)
                break
            except ValueError as exc:
                last = exc
        else:
            raise last
        want = [[self._eps * x for x in row] for row in tri2.mutable_block()]
        if self._node_pairing(new_key) != want:
            raise RuntimeError(
                f"gauged flip at edge {edge} of chart {key}: mutated node "
                f"pairing != {self._eps:+d}·(flipped fan mutable block)"
            )
        self._tri[new_key] = tri2
        return new_key, iso

    def flip_path(self, edges, key=()):
        from bps_atlas import BPSAtlas
        key = BPSAtlas._norm_key(key)
        for e in edges:
            key, _ = self.flip(e, key)
        return key


class SkeinAtlas:
    """Ensemble of `SkeinKAlgebra` charts of one `Sk(Σ)` related by
    triangulation flips (see the module docstring)."""

    def __init__(self, triangulation: Triangulation, *, spec=None,
                 max_local_moves: int = 8, root_cone_data=None,
                 root_engine_provenance: str = "bps-chart-anchored",
                 **atlas_kwargs):
        self.is_bordered = bool(getattr(triangulation, "is_bordered", False))
        if self.is_bordered:
            from skein_sphere_atlas import BorderedSkeinAtlas
            self._w = BorderedSkeinAtlas(triangulation, spec=spec,
                                         max_local_moves=max_local_moves,
                                         **atlas_kwargs)
        else:
            self._w = SkeinSphereAtlas(triangulation, spec=spec,
                                       max_local_moves=max_local_moves,
                                       **atlas_kwargs)
        self._mode = "bordered" if self.is_bordered else "closed"
        # optional ROOT ray dictionary for bordered charts (rays = the
        # chords in the root frame, engine included); flipped charts
        # transport it through the certified transitions.
        self._root_cd = root_cone_data
        self._root_prov = root_engine_provenance
        self._skein: dict = {}
        self._label_maps: dict = {}

    # -- factories -----------------------------------------------------------

    @classmethod
    def tetrahedron(cls, **kw) -> "SkeinAtlas":
        """`S²₀,₄` = SU(2) N_f=4 (6 edges, gauge rank 1)."""
        return cls(Triangulation.tetrahedron_S2_4(), **kw)

    @classmethod
    def bipyramid(cls, *, spec=None, **kw) -> "SkeinAtlas":
        """`S²₀,₅` (spec strongly recommended — auto-find is
        prohibitive at rank 9)."""
        return cls(Triangulation.bipyramid_S2_5(), spec=spec, **kw)

    @classmethod
    def polygon(cls, n: int, **kw) -> "SkeinAtlas":
        """The bordered fan n-gon (disk, n marks = the [A₁,A_{n-3}] AD
        theory) as a flip atlas of `SkeinKAlgebra` charts.

        n=5 wires the pentagon root ray dictionary (the roster
        `PentagonSkeinConeData` under the node-swap frame map, PINNED
        engine behind the cross-products).  Odd n ≥ 7 wire the uniform
        `A1A2kKAlg(k)` chord dictionary (k = (n−3)/2): charges from the
        closed-form BPS-free ρ-orbit seeds (`a1a2k_bps_iso`), the
        inverse by the fan solve, the frame identification onto the
        bordered mutable block selected by a live product check against
        the twin, and cross-products from the generic-n PINNED engine
        (`SkeinOddPolygonKAlg`).  Even n (the ungauged even
        polygons) keep the certified flip/iso machinery with
        `chart_skein` honest-failing (the flavoured zoo entries'
        iso-inverse asymmetry; the GAUGED even
        family is `SkeinAtlas.u1a1aodd(k)`)."""
        from bordered_triangulation import BorderedTriangulation
        tri = Triangulation.from_bordered_slots(
            BorderedTriangulation.fan_polygon(n))
        if n == 5:
            from skein_kalgebra import PentagonSkeinConeData

            def _swap(l):
                a, b = tuple(l)
                return (b, a)

            roster_cd = PentagonSkeinConeData()

            def _engine(g, h):
                # the pinned engine on roster labels, read back through
                # the node swap (measured: the swap is an honest algebra
                # iso — coefficients carry over verbatim)
                el = roster_cd.engine_product(_swap(g), _swap(h))
                return Element({_swap(l): c for l, c in el.terms.items()})

            root_cd = TransportedSkeinConeData(
                roster_cd, to_base=_swap, from_base=_swap,
                engine_product_fn=_engine)
            return cls(tri, root_cone_data=root_cd,
                       root_engine_provenance="stated-skein-pinned", **kw)
        at = cls(tri, **kw)
        if n % 2 == 1 and n >= 7:
            at._root_cd = _a1a2k_root_cd(
                (n - 3) // 2, tri, at.chart_kalg_multiply_root())
            at._root_prov = "stated-skein-pinned"
        return at

    def chart_kalg_multiply_root(self):
        """The root twin's multiply, late-bound (the zoo root dictionary
        is built after the atlas)."""
        def _mul(a, b):
            return self._w.chart_kalg(()).multiply(a, b)
        return _mul

    @classmethod
    def a1dodd(cls, k: int = 0) -> "SkeinAtlas":
        """The `[A₁, D_{2k+3}]` family — the once-punctured
        `(2k+3)`-gon (regular interior puncture, SU(2) flavour) as a
        QUIVER-LEVEL flip atlas: transitions = mutations of the
        D_{2k+3} fork quiver (`build_dodd_bps_quiver`), which cover the
        FULL exchange graph including the tagged-arc flips at the
        puncture (the chart ↔ tagged-triangulation dictionary is the
        recorded follow-up — quiver mutations don't need tags).

        The FIRST FLAVOURED atlas — **LIVE at k = 0, 1, 2
        (D₃/D₅/D₇)**: the root dictionary
        (`A1DoddChartSkeinConeData`) puts the `(a,p,i)` arc charges as
        monomial rays and the SU(2) weight direction `e_{2k+3}` as the
        torus pair `±v_f`; the roster `A1DoddSkeinKAlgebra(k)`
        engine's `χ_κ`-dressed outputs are expanded by the EXACT SU(2)
        weight dictionary into Z-form flavour monomials.  **The
        arc↔charge frames come uniformly from the
        `finite_a1d{3,5,7}_kalg` ground-truth modules**, which inline
        every mult-gen as a lattice charge in exactly this fork frame
        (node charges verified equal) plus the ρ-permutation:
        `CH[(a,p,i)] = LATTICE[orbit[(s + d·i) mod H]]` with the small
        per-k gauge table below (k=1's frame equals the `a1d5_kalg`
        orbit data verbatim).  MEASURED: k=0 the
        FULL 6-arc crossing grid == the fork twin; k=1 380/380
        q-commute + 200/200 crossing products; k=2 123/123 q-commute +
        60/60 crossing products on the shallow-pair sample (deep fork
        products are hours-expensive — the SIGALRM-timed method).
        NOT BPS-ρ orbits of the seeds (the
        fork-chart BPS ρ is not the `Z_{2k+3}` rotation), and NOT
        `a1d7_kalg` (measured: its 28-atom q-commute graph is not
        isomorphic to any 4-family subgraph of the intrinsic).
        k ≥ 3 honest-fails
        (generate `finite_a1d{2k+3}` or crack the general-k frame)."""

        import a1dodd_fork_quiver as sc
        from bps_kalgebra import BPSKAlgebra

        from skein_kalgebra import A1DoddSkeinKAlgebra

        # per-k gauge: finite-module orbit rep -> ((a, p), shift, dir)
        frame_gauges = {
            0: {0: ((1, 0), 0, 1), 1: ((1, 1), 0, 1)},
            1: {0: ((2, 0), 2, 1), 1: ((1, 0), 4, 1),
                2: ((1, 1), 2, 1), 3: ((2, 1), 1, 1)},
            2: {0: ((3, 0), 0, 1), 1: ((2, 0), 0, 1), 2: ((1, 0), 0, 1),
                3: ((1, 1), 0, 1), 4: ((2, 1), 0, 1), 5: ((3, 1), 0, 1)},
        }
        if k not in frame_gauges:
            raise NotImplementedError(
                f"SkeinAtlas.a1dodd(k={k}): frames are sourced from the"
                f" finite_a1d{{3,5,7}} ground-truth lattices (k <= 2);"
                f" k >= 3 needs finite_a1d{2 * k + 3} generated"
                f" (generate_finite_kalg) or the general-k gauge frame"
                f" cracked (a1dodd_cone_data)"
            )
        pairing, node_charges = sc.build_dodd_bps_quiver(k)
        bps = BPSKAlgebra(pairing=pairing, node_charges=node_charges,
                          verify="off")
        H = 2 * k + 3
        # the SU(2) weight direction (the frame's LAST coordinate;
        # doublet steps = ±1 unit)
        ker = tuple([0] * (2 * k + 2)) + (1,)
        roster = A1DoddSkeinKAlgebra(k)
        import importlib
        M = importlib.import_module(f"finite_a1d{2 * k + 3}_kalg")
        LAT = getattr(M, f"A1D{2 * k + 3}_MULT_GENS_LATTICE")
        RHO = getattr(M, f"A1D{2 * k + 3}_RHO_PERM")
        seen: set = set()
        CH = {}
        for s0 in RHO:
            if s0 in seen:
                continue
            o, c = [s0], RHO[s0]
            seen.add(s0)
            while c != s0:
                o.append(c)
                seen.add(c)
                c = RHO[c]
            (a_, p_), sh, d = frame_gauges[k][o[0]]
            for i in range(H):
                CH[(a_, p_, i)] = tuple(LAT[o[(sh + d * i) % H]])
        cd = A1DoddChartSkeinConeData(roster, CH, ker, pairing)
        # Construction-time certification (the live meson gate): a
        # crossing arc pair through the dictionary + engine + weight
        # expansion must equal the fork-quiver twin.  With the finite
        # ground-truth frames this PASSES at k = 0, 1, 2 (full grid /
        # 200/200 / 60-of-60 shallow — the gate stays as the standing
        # per-construction guard, never a fit).  The pair is the
        # SHALLOWEST crossing one — deep fork products can take hours.
        arcs = sorted(cd._letter_of,
                      key=lambda c: (sum(abs(v) for v in c), c))
        crossing = None
        for a in arcs:
            for b in arcs:
                la, lb = cd._letter_of[a], cd._letter_of[b]
                if a != b and not cd._int.q_commute(la, lb):
                    crossing = (a, b)
                    break
            if crossing:
                break
        if crossing is not None:
            got = {l: str(c) for l, c in
                   cd.engine_product(*crossing).terms.items()}
            want = {tuple(l): str(c) for l, c in
                    bps.multiply(*crossing).terms.items()}
            if got != want:
                raise NotImplementedError(
                    f"SkeinAtlas.a1dodd(k={k}): the arc-charge dictionary "
                    f"is not yet calibrated to the roster engine on the "
                    f"cross-product sector (q-commuting sector matches; "
                    f"measured mismatch at {crossing}: derived {got} vs "
                    f"twin {want}) — the sign × rotation × q-convention "
                    f"calibration is the recorded pickup (plan notes)"
                )
        core = _GaugedPolygonAtlasCore(bps, tri=None)
        at = cls.__new__(cls)
        at.is_bordered = True
        at._mode = "gauged"
        at._w = core
        at._root_cd = cd
        at._root_prov = "cone-anchored"     # the roster engine's honest
                                            # exception rides through
        at._skein = {}
        at._label_maps = {}
        return at

    @classmethod
    def u1a1aodd(cls, k: int) -> "SkeinAtlas":
        """The U(1)-gauged `[A₁, A_{2k+1}]` family (the U1A1odd cases) —
        the HONEST stated algebras of the even fan polygons (the
        even-marked obstruction bars full unpinning; the stated
        `(2k+4)`-gon presents the gauged algebra).  The atlas runs over
        the GAUGED quiver chart (mutable path nodes inside the ambient
        lattice carrying the frozen flavour / E direction), transitions
        = mutations ≡ fan-triangulation flips (certified per flip under
        the measured orientation: gauged node pairing == −(fan mutable
        block)), and the lifted charts carry the certified gauged
        dictionaries:

          * k = 0 — SQED1 / the square: `U1SquareKAlg`'s QTCone
            dictionary through the total `sqed1_object` iso; PINNED
            engine (`SkeinSquareKAlg`) behind the root cross-products.
          * k = 1 — the gauged hexagon: `U1A1AoddKAlg(1)`'s dictionary
            through the total `u1hexagon_object` iso (measured: every
            generator inverts cleanly — unlike the ungauged even zoo
            entries); the generic-n PINNED engine
            (`SkeinEvenPolygonKAlg` — supersedes the localized
            route, ~30 s/battery → ~free).
          * k = 2..4 — the gauged octagon / decagon / 12-gon: the
            oracle-extracted `u1a1aodd_kalg.U1A1AoddKAlg(k)` intrinsic
            SHARED with the PINNED engine, and the LINEAR charge
            dictionary read off its cone data (`_chg` / `_MU`; inverse
            = the fan solve over the cone family, E Laurent —
            `_linear_charge_maps`); the battery against the gauged
            quiver twin certifies frame + engine together.
          * k ≥ 5 — honest-fail (the bootstrap's hand-discovered seeds
            end at k = 4; the RGKAlgebra-presented route is the
            recorded path)."""
        from bordered_triangulation import BorderedTriangulation
        n_marks = 2 * k + 4
        tri = Triangulation.from_bordered_slots(
            BorderedTriangulation.fan_polygon(n_marks))
        if k == 0:
            from sqed1_object import sqed1_object
            obj = sqed1_object()
            intr = obj.realization("cone")          # U1SquareKAlg
            iso = obj.iso("cone", "bps")
            engine = obj.realization("skein-pinned")
            prov = "stated-skein-pinned"
        elif k == 1:
            from hexagon_objects import u1hexagon_object
            from skein_evengon_kalg import SkeinEvenPolygonKAlg
            obj = u1hexagon_object()
            intr = obj.realization("intrinsic")     # U1A1AoddKAlg(1)
            iso = obj.iso("intrinsic", "bps")
            # the generic-n PINNED engine on the
            # registered intrinsic — supersedes the localized route
            # (measured: ~30 s/battery → ~free)
            engine = SkeinEvenPolygonKAlg(intrinsic=intr)
            prov = "stated-skein-pinned"
        elif k <= 4:
            # The gauged octagon / decagon / 12-gon: the
            # oracle-extracted intrinsic (`u1a1aodd_kalg` — the
            # roster's frame) SHARED with the generic-n PINNED engine
            # (`SkeinEvenPolygonKAlg`), the charge dictionary read off
            # ITS cone data (`_chg` / `_MU` — same gauged-quiver frame
            # as `_build_bps`; inverse = the fan solve, E Laurent).
            # The atlas battery against the gauged quiver twin
            # certifies frame + engine together.
            from u1a1aodd_kalg import U1A1AoddKAlg as _Kalg
            from u1a1aodd_general import _build_bps
            from skein_evengon_kalg import SkeinEvenPolygonKAlg
            intr = _Kalg(k)
            iso = None
            engine = SkeinEvenPolygonKAlg(intrinsic=intr)
            prov = "stated-skein-pinned"
            obj = None
            bps = _build_bps(k)
        else:
            raise NotImplementedError(
                f"u1a1aodd(k={k}): the u1a1aodd_general bootstrap knows "
                f"k ≤ 4 only (hand-discovered seeds; higher k needs the "
                f"RGKAlgebra-presented route — see the module docstring)"
            )
        if obj is not None:
            bps = obj.realization("bps")
        core = _GaugedPolygonAtlasCore(bps, tri)

        def _single(el, what):
            ts = dict(el.terms)
            assert len(ts) == 1, (what, ts)
            ((l, c),) = ts.items()
            assert dict(c._coeffs) == {0: 1}, (what, str(c))
            return tuple(l)

        if iso is not None:
            def to_base(gamma):
                return _single(iso.inverse(Element({tuple(gamma): _ONE})),
                               f"u1a1aodd[{k}]:to_base({gamma})")

            def from_base(lbl):
                return _single(iso.map(Element({tuple(lbl): _ONE})),
                               f"u1a1aodd[{k}]:from_base({lbl})")
        else:
            to_base, from_base = _linear_charge_maps(intr)

        def _engine(g, h):
            el = engine.multiply(to_base(g), to_base(h))
            out: dict = {}
            for l, c in el.terms.items():
                l2 = from_base(l)
                out[l2] = (out[l2] + c) if l2 in out else c
            return Element(out)

        # Construction-time certification (the live meson gate, mirroring
        # a1dodd — the k >= 2 dictionaries had NO
        # construction-time check and k = 4 had zero coverage anywhere):
        # the first CROSSING node pair, routed dictionary -> engine ->
        # dictionary, must equal the gauged quiver twin exactly.
        # Measured ~free (<= 0.3 s at k = 4) on top of construction;
        # PASSES at k = 0..4.
        nodes = [tuple(c) for c in bps.node_charges]
        gate = None
        for _i, _a in enumerate(nodes):
            for _b in nodes[_i + 1:]:
                if len(dict(bps.multiply(_a, _b).terms)) > 1:
                    gate = (_a, _b)
                    break
            if gate:
                break
        if gate is not None:
            got = dict(_engine(*gate).terms)
            want = dict(bps.multiply(*gate).terms)
            if got != want:
                raise NotImplementedError(
                    f"u1a1aodd(k={k}): the charge dictionary is not "
                    f"calibrated to the gauged twin on the cross-product "
                    f"sector (measured mismatch at {gate}: derived "
                    f"{ {l: str(c) for l, c in got.items()} } vs twin "
                    f"{ {l: str(c) for l, c in want.items()} })"
                )

        root_cd = TransportedSkeinConeData(
            intr.cone_data(), to_base=to_base, from_base=from_base,
            engine_product_fn=_engine)
        at = cls.__new__(cls)
        at.is_bordered = True
        at._mode = "gauged"
        at._w = core
        at._root_cd = root_cd
        at._root_prov = prov
        at._skein = {}
        at._label_maps = {}
        return at

    # -- passthrough (the certified BPS-level atlas) ---------------------------

    @property
    def atlas(self):
        """The wrapped `BPSAtlas` (folding, recognition, monodromy)."""
        return self._w.atlas

    def keys(self):
        return self._w.keys()

    def triangulation(self, key=()) -> Triangulation:
        return self._w.triangulation(key)

    def chart(self, key=()):
        """The root-frame `BPSKAlgebra` chart at `key`."""
        return self._w.chart(key)

    def chart_kalg(self, key=()):
        """The native `SkeinSphereKAlg` twin at `key` (flip-transported
        spec — no auto-find off the root)."""
        return self._w.chart_kalg(key)

    def iso(self, src=(), dst=()) -> KAlgebraIso:
        """Root-frame transition (the wrapped atlas's certified iso)."""
        return self._w.iso(src, dst)

    def flip(self, edge: int, key=()):
        """Flip `edge` (certifies flip == mutation exactly, per the
        wrapped atlas); returns `(new_key, root_frame_iso)`."""
        return self._w.flip(edge, key)

    def flip_path(self, edges, key=()):
        return self._w.flip_path(edges, key)

    def verify_bare_part(self, curve_coords, key=()) -> bool:
        if self.is_bordered:
            raise NotImplementedError(
                "verify_bare_part is the closed-surface dictionary"
            )
        return self._w.verify_bare_part(curve_coords, key)

    # -- the lifted charts -----------------------------------------------------

    def _chart_label_maps(self, key):
        """The certified atlas transition ROOT ↔ chart, read as a label
        bijection (asserted single-label, unit-coefficient — canonical
        basis elements are transition-invariant).  Memoized per key."""
        from bps_atlas import BPSAtlas
        key = BPSAtlas._norm_key(key)
        if key not in self._label_maps:
            iso = self._w.chart_iso(key)

            def _single(el, what):
                ts = dict(el.terms)
                assert len(ts) == 1, (what, ts)
                ((l, c),) = ts.items()
                assert dict(c._coeffs) == {0: 1}, (
                    f"{what}: non-unit coefficient {c} — the transition "
                    f"does not read as a label bijection here"
                )
                return tuple(l)

            def to_chart(root_l, _iso=iso):
                return _single(_iso.map(Element({tuple(root_l): _ONE})),
                               f"to_chart[{key}]({root_l})")

            def to_root(chart_l, _iso=iso):
                return _single(_iso.inverse(Element({tuple(chart_l): _ONE})),
                               f"to_root[{key}]({chart_l})")

            self._label_maps[key] = (to_chart, to_root)
        return self._label_maps[key]

    def chart_skein(self, key=()) -> SkeinKAlgebra:
        """The chart at `key` as a first-class `SkeinKAlgebra`
        (memoized).

        Closed surfaces: multicurve-component cone data, canonical
        multiply anchored on the native chart twin (the measured 3-term
        tower — module docstring), ρ/trace transported from the twin,
        the chart's own `σ_Δ` + flip-transported spec as the live BPS
        parent substrate.

        Bordered charts: the ROOT ray dictionary (`root_cone_data`,
        e.g. the pentagon's) transported through the certified
        transition; `multiply` = the derived cone reducer over the
        chart twin's cross-products (finite chord cones — the roster
        pattern); honest-fail when no root dictionary is wired."""
        from bps_atlas import BPSAtlas
        key = BPSAtlas._norm_key(key)
        if key in self._skein:
            return self._skein[key]
        twin = self._w.chart_kalg(key)
        if self._mode == "gauged":
            # root frame throughout (the ambient lattice is shared):
            # the ambient pairing off the chart's lattice, the chart's
            # own node charges + necklaced spec as the BPS substrate.
            if self._root_cd is None:
                raise NotImplementedError(
                    "SkeinAtlas.chart_skein: no root ray dictionary "
                    "(gauged mode always wires one — internal error)"
                )
            if not key:
                cd = self._root_cd
                prov = self._root_prov
            else:
                to_chart, to_root = self._chart_label_maps(key)
                cd = TransportedSkeinConeData(
                    self._root_cd, to_base=to_root, from_base=to_chart,
                    engine_product_fn=twin.multiply)
                prov = "bps-chart-anchored"
            dim = len(tuple(twin.node_charges[0]))
            amb = [tuple(1 if j == i else 0 for j in range(dim))
                   for i in range(dim)]
            pairing = [[twin.lattice.bracket(amb[i], amb[j])
                        for j in range(dim)] for i in range(dim)]
            self._skein[key] = SkeinKAlgebra(
                cone_data=cd,
                identity_label=(0,) * dim,
                rho_fn=twin.rho,
                rho_inverse_fn=twin.rho_inverse,
                trace_fn=lambda a, K, _t=twin: _t.trace(a, K),
                bps_factory=lambda _t=twin: _t,
                engine_multiply=None,          # derived cone reducer over
                                               # the engine cross-products
                name=f"skein-atlas-chart[{BPSAtlas._key_str(key)}]",
                engine_provenance=prov,
                trace_provenance="transported-bps",
                chart=dict(pairing=pairing,
                           node_charges=[tuple(g)
                                         for g in twin.node_charges],
                           spec=[tuple(g) for g in twin.spec]),
            )
            return self._skein[key]
        if self.is_bordered:
            if self._root_cd is None:
                raise NotImplementedError(
                    "SkeinAtlas.chart_skein: no root ray dictionary is "
                    "wired for this bordered surface (polygon(5) wires "
                    "the pentagon; others are recorded in the "
                    "absorption handoff)"
                )
            if not key:
                cd = self._root_cd
                prov = self._root_prov
            else:
                to_chart, to_root = self._chart_label_maps(key)
                cd = TransportedSkeinConeData(
                    self._root_cd, to_base=to_root, from_base=to_chart,
                    engine_product_fn=twin.multiply)
                prov = "bps-chart-anchored"
            n = len(twin.node_charges)
            nodes = [tuple(1 if j == i else 0 for j in range(n))
                     for i in range(n)]
            self._skein[key] = SkeinKAlgebra(
                cone_data=cd,
                identity_label=(0,) * n,
                rho_fn=twin.rho,
                rho_inverse_fn=twin.rho_inverse,
                trace_fn=lambda a, K, _t=twin: _t.trace(a, K),
                bps_factory=lambda _t=twin: _t,
                engine_multiply=None,          # derived cone reducer over
                                               # the engine cross-products
                                               # (finite chord cones)
                name=f"skein-atlas-chart[{BPSAtlas._key_str(key)}]",
                engine_provenance=prov,
                trace_provenance="transported-bps",
                chart=dict(pairing=[list(r) for r in
                                    self._w.triangulation(key)
                                    .mutable_block()],
                           node_charges=nodes,
                           spec=[tuple(g) for g in twin.spec]),
            )
            return self._skein[key]
        cd = MulticurveChartSkeinConeData(twin)
        n = twin.triangulation.n_edges
        nodes = [tuple(1 if j == i else 0 for j in range(n))
                 for i in range(n)]
        self._skein[key] = SkeinKAlgebra(
            cone_data=cd,
            identity_label=(0,) * n,
            rho_fn=twin.rho,
            rho_inverse_fn=twin.rho_inverse,
            trace_fn=lambda a, K, _t=twin: _t.trace(a, K),
            bps_factory=lambda _t=twin: _t,
            engine_multiply=twin.multiply,   # the measured 3-term
                                             # tower: the reducer
                                             # does not serve this
                                             # presentation
            name=f"skein-atlas-chart[{BPSAtlas._key_str(key)}]",
            engine_provenance="bps-chart-anchored",
            trace_provenance="transported-bps",
            chart=dict(pairing=twin.triangulation.sigma(),
                       node_charges=nodes,
                       spec=[tuple(g) for g in twin.spec]),
        )
        return self._skein[key]

    def iso_skein(self, src=(), dst=()) -> KAlgebraIso:
        """Certified `KAlgebraIso` between the LIFTED charts
        `chart_skein(src) → chart_skein(dst)`: the wrapped atlas's
        chart isos composed through the root frame."""
        from bps_atlas import BPSAtlas
        src = BPSAtlas._norm_key(src)
        dst = BPSAtlas._norm_key(dst)
        A = self.chart_skein(src)
        B = self.chart_skein(dst)
        iso_s = self._w.chart_iso(src)
        iso_d = self._w.chart_iso(dst)

        def fwd(lbl):
            root_el = iso_s.inverse(Element({tuple(lbl): _ONE}))
            out: dict = {}
            for l, c in root_el.terms.items():
                for l2, c2 in iso_d.map(Element({tuple(l): _ONE})
                                        ).terms.items():
                    out[l2] = out.get(l2, LaurentPoly.zero()) + c * c2
            return Element({l: c for l, c in out.items()
                            if dict(c._coeffs)})

        def inv(lbl):
            root_el = iso_d.inverse(Element({tuple(lbl): _ONE}))
            out: dict = {}
            for l, c in root_el.terms.items():
                for l2, c2 in iso_s.map(Element({tuple(l): _ONE})
                                        ).terms.items():
                    out[l2] = out.get(l2, LaurentPoly.zero()) + c * c2
            return Element({l: c for l, c in out.items()
                            if dict(c._coeffs)})

        return KAlgebraIso(
            A, B, fwd, inv,
            name=(f"skein-atlas[{BPSAtlas._key_str(src)}→"
                  f"{BPSAtlas._key_str(dst)}]"))
