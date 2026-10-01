"""[A₁,A₂ₖ₊₁]: the zoo entries `a3` / `a5` / `a7` (aliases `hexagon` / `octagon`
/ `decagon`) through the ungauged gauged polygon, k = 1, 2, 3.

The ADE finite K-algebras are served here by
self-contained realizations, with no frozen trace data and no runtime oracle on
the serving path, fully functional (multiply and trace on every label to any
order).  For the u(1)-flavoured `[A₁,A₂ₖ₊₁]` that realization is
`ungauge_kalgebra.ungauge_u1a1aodd(k)`: the centralizer of the gauge letter `E`
in the closed-form `U1A1AoddKAlg(k)`, with `E` promoted to the U(1) flavour
fugacity.  Its products are the gauged class's closed-form arc rules; its trace
is the gauged Layer-1 reduction over closed-form seeds
(`u1_pgon_layer2.singlet_chord_trace`), summed over the gauge charge with the
restored `(𝖖²;𝖖²)²_∞` vector-multiplet measure.  This module identifies the zoo
standalone's canonical basis with it and serves the zoo's seed traces from it
(`elem_traces._seed_series`).  Until 2026-09-23 a5 / a7 were served by the u(1)
orthonormality bootstrap `u1_bootstrap.generate_u1`, which rests on a recorded
μ-support hypothesis and is slow (a5 at 𝖖²⁰ in 100 s, a7 at 𝖖¹⁴ in 150 s);
it stays as a witness, as do the a3 closed-form characters
(`ad_characters.a3_elem_entry`), which served a3 / hexagon until the same day.

The label map, built at runtime from the two classes' own data (nothing is
stored):

  * each multiplicative generator `(F, 0)` of the ungauged algebra — a
    mixed-parity diagonal, or a non-crossing (even–even, odd–odd) pair of
    diagonals of the `(2k+4)`-gon (`U1A1AoddKAlg._centralizer_generators`) —
    has the gauged charge `v = Σ m·charge_formula(t, i)` in the `A_{2k+2}`
    chain coordinates of `U1A1AoddKAlg`.  Its last coordinate, the magnetic
    one, is 0 (the multiset is balanced); dropping it leaves `w ∈ Z^{2k+1}`.
    With `f` the last entry of `w` and `μ'` the charge of `E` with its magnetic
    coordinate dropped (`(1, 0, 1, …, 0, 1)`, last entry 1), the vector
    `w − f·μ'` has last entry 0 and is matched against the zoo's
    `<PRE>_MULT_GENS_LATTICE`, whose vectors all end in 0.  The match is a
    bijection (6/6, 24/24, 65/65); anything else raises;
  * a zoo word `((mg, p), …)` maps to `(the union of the multisets p·F_mg,
    −Σ p·f_mg)`, and a zoo coefficient `μ^f` to an `E^{+f}` shift of the
    label's E-power (`to_ungauged`);
  * a trace is read back with `z ↦ z⁻¹`: the zoo's `μ^g` coefficient is the
    ungauged trace's `z^{−g}` one.  (The ungauged trace carries a label's
    E-power as `z^{−e}`, `Tr((F, e)) = z^{−e}·Tr((F, 0))`, while `μ^f`
    multiplies the zoo trace; so `μ = z⁻¹`.)

Each of the three signs was fixed by a positive control, and the alternatives
fail it (the suite in the source repository): the three other choices of the word sign
and the coefficient shift reproduce only 18, 18 and 22 of the 36 a3 generator
products; read back with `z ↦ z`, 4 of the 6 a3 seeds disagree with the
closed-form characters.  Under the map as stated the generator products agree
(36/36, 576/576, 4225/4225), `rho_element` agrees on every generator (6, 24,
65), and the seed traces equal the a3 closed form (through 𝖖⁴⁸) and the u(1)
bootstrap wherever it finishes (a5 through 𝖖²⁰, a7 through 𝖖¹⁴).

The map is not the trace-level dictionary `_A5_UNGAUGED` of
the suite in the source repository: that one was matched on traces alone, is not
injective (eight a5 seeds share one label there, since equal series match the
same label) and reads the ungauged trace without `z ↦ z⁻¹`.  A match on traces
alone does not determine the label; the generator products do.

The KAlgebraIso
---------------
Whether the seeds maps are `KAlgebraIso`s was an open question; they were
not, and `kalgebra_iso(short_id, native=None)` makes
this one a `KAlgebraIso`.  A `KAlgebraIso` sends labels to
`Element`s and multiplies coefficients through, so both sides must keep the
flavour in the same place.  The zoo keeps it in its coefficients, and the
ungauged class in its labels.  The source is therefore the zoo's Z-form wrapper
`finite_u1_zform.FiniteU1ZKAlgebra(native)`, whose label `((), (word, (m,)))`
is `μ^m·L_word`.  The target is `ungauged_algebra(short_id)`, whose label
`(F, e)` is `z^{−e}·L_{(F, 0)}`:

  * forward: `((), (word, (m,))) ↦ (F, e + m)` with `(F, e) =
    to_ungauged_label(word)` — the coefficient shift `μ^m ↦ E^{+m}` of
    `to_ungauged`, moved into the label;
  * inverse: `(F, e) ↦ ((), (word, (e − e_word,)))`, where `word` is the zoo
    word whose generators' images make up `F`.  The mixed-parity diagonals
    come one generator each; the even–even and odd–odd diagonals are paired
    across the sign of their magnetic charge (`UngaugedKAlgebra.mag`).  The
    word is then made canonical in the zoo's cone: from k = 3 two pairings
    can name one element (the a7 cone relations).

The battery also compares traces as elements of one ring.  The two rings are
one `R(U(1))` in two coordinates: the wrapper's `TensorZPlusRing(Trivial,
R(U(1)))` in the zoo's `μ`, and the ungauged class's `R(U(1))` in `z = μ⁻¹`
(the readback `z ↦ z⁻¹` above).  So the source is the wrapper base-changed
along that ring isomorphism, `μ^m ↦ z^{−m}` (`KAlgebra.base_change` keeps the
labels, products and ρ, and pushes the trace through the isomorphism); the
target is the ungauged class itself.  To carry the section check the wrapper
gained the flavour-lift coordinate `r_label_decompose` / `r_label_compose` the
same day.

Certified by the suite in the source repository, measured 2026-09-26 (shared machine).
`verify_all` passes all five checks — unit, round trip, multiplicative,
ρ-equivariant, trace-equivariant through `𝖖¹²` — on the identity, every
generator and `μ^{±1}`, and on every ordered pair of them each way:

| entry | pairs each way | time | round trip on product labels |
|---|---|---|---|
| a3 | 8² = 64 | 0.2 s | 36 + 40 |
| a5 | 26² = 676 | 1.0 s | 246 + 301 |
| a7 | 67² = 4,489 | 8.9 s | 1,343 + 1,954 |

`verify_maps_section_to_section_1drep` holds both ways on the identity and the
generators.  Negative controls at a3 and a5: with the flavour normalisation
flipped (`μ ↦ z`) the battery fails the multiplicative, ρ and trace checks,
and so it does with two zoo generators transposed.  The aliases share the
construction.  `finite_kalgebra_objects.kalgebra_object` registers the pair as
'z-form' → 'ungauged-u1a1aodd'.

Pure Python; no BPS engine and no bootstrap is imported.
"""
from __future__ import annotations

from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import RElement, RingHom, RLaurent

from regen import (
    _canonical_zoo_word, _load_standalone, _zoo_word_of)


__all__ = [
    "K_OF",
    "ungauged_algebra",
    "generator_map",
    "to_ungauged_label",
    "to_ungauged",
    "seeds",
    "kalgebra_iso",
]

#: zoo id -> `k` of `ungauge_u1a1aodd(k)` (the `(2k+4)`-gon)
K_OF = {"a3": 1, "hexagon": 1, "a5": 2, "octagon": 2, "a7": 3, "decagon": 3}

# the canonical zoo id per k (its standalone holds the generator lattice)
_ZOO_ID = {1: "a3", 2: "a5", 3: "a7"}

_UNGAUGED: dict = {}      # k -> UngaugedKAlgebra
_GEN_MAPS: dict = {}      # k -> {mg: (F, f)}
_SEEDS: dict = {}         # k -> _AoddSeeds


def _k(short_id: str) -> int:
    if short_id not in K_OF:
        raise KeyError(
            f"aodd_seeds: {short_id!r} is not an [A1,A_(2k+1)] zoo entry "
            f"(expected one of {sorted(K_OF)})")
    return K_OF[short_id]


def ungauged_algebra(short_id: str):
    """The (shared, per-process) `ungauge_u1a1aodd(k)` for this entry."""
    k = _k(short_id)
    if k not in _UNGAUGED:
        from ungauge_kalgebra import ungauge_u1a1aodd
        _UNGAUGED[k] = ungauge_u1a1aodd(k)
    return _UNGAUGED[k]


def generator_map(short_id: str) -> dict:
    """`{mg: (F, f)}` over the zoo's multiplicative generators: zoo generator
    `mg` is the ungauged label `(F, −f)` (module docstring).  Built from the
    gauged charges and the zoo's `<PRE>_MULT_GENS_LATTICE` at first use and
    cached; raises `AssertionError` unless the match is a bijection of
    generators with vanishing magnetic charge."""
    k = _k(short_id)
    if k in _GEN_MAPS:
        return _GEN_MAPS[k]
    U = ungauged_algebra(short_id)
    cd = U._G.cone_data()
    n, mu = cd._n, cd._MU
    mod, prefix = _load_standalone(_ZOO_ID[k])
    lattice = [tuple(v) for v in getattr(mod, f"{prefix}_MULT_GENS_LATTICE")]
    if any(len(v) != n - 1 or v[-1] != 0 for v in lattice):
        raise AssertionError(
            f"aodd_seeds(k={k}): the zoo lattice is not Z^{n - 1} with last "
            f"entry 0")
    by_section: dict = {}
    for (F, e) in U.mult_generators():
        if e != 0:
            raise AssertionError(f"aodd_seeds(k={k}): generator {F!r} has "
                                 f"E-power {e}")
        v = [0] * n
        for (t, i, m) in F:
            c = cd.charge_formula(t, i)
            for j in range(n):
                v[j] += m * c[j]
        if v[n - 1] != 0:
            raise AssertionError(
                f"aodd_seeds(k={k}): generator {F!r} has magnetic charge "
                f"{v[n - 1]}")
        w = v[:n - 1]
        f = w[-1]
        section = tuple(w[j] - f * mu[j] for j in range(n - 1))
        if section in by_section:
            raise AssertionError(
                f"aodd_seeds(k={k}): generators {by_section[section][0]!r} and "
                f"{F!r} have one section {section}")
        by_section[section] = (F, f)
    missing = [j for j, s in enumerate(lattice) if s not in by_section]
    if missing or len(by_section) != len(lattice):
        raise AssertionError(
            f"aodd_seeds(k={k}): the section match is not a bijection "
            f"({len(by_section)} generators, {len(lattice)} zoo generators, "
            f"unmatched zoo generators {missing})")
    _GEN_MAPS[k] = {j: by_section[s] for j, s in enumerate(lattice)}
    return _GEN_MAPS[k]


def to_ungauged_label(short_id: str, word) -> tuple:
    """The ungauged label `(F, e)` of the zoo label `word = ((mg, p), …)`:
    `F` the union of the multisets `p·F_mg`, `e = −Σ p·f_mg`."""
    gm = generator_map(short_id)
    letters: dict = {}
    e = 0
    for (mg, p) in word:
        F, f = gm[mg]
        for (t, i, m) in F:
            letters[(t, i)] = letters.get((t, i), 0) + p * m
        e -= p * f
    return (tuple(sorted((t, i, m) for (t, i), m in letters.items() if m)), e)


def to_ungauged(short_id: str, x: Element) -> Element:
    """A zoo `Element` (coefficients `LaurentPoly`, or `RLaurent` over the
    zoo's `AbelianZPlusRing(1)`) as an ungauged `Element`: each label through
    `to_ungauged_label`, each coefficient `𝖖^j·μ^f` moved into the label as an
    `E^{+f}` shift, leaving the `LaurentPoly` coefficient `𝖖^j`."""
    out: dict = {}
    for word, c in x.terms.items():
        F, e = to_ungauged_label(short_id, word)
        if isinstance(c, RLaurent):
            items = [(q, key, v) for q, r in c.coeffs.items()
                     for key, v in r.terms.items() if v]
        else:
            items = [(q, (0,), v) for q, v in c._coeffs.items() if v]
        for q, key, v in items:
            (f,) = key
            d = out.setdefault((F, e + f), {})
            d[q] = d.get(q, 0) + v
    return Element({lab: LaurentPoly({q: v for q, v in d.items() if v})
                    for lab, d in out.items() if any(d.values())})


class _AoddSeeds:
    """The zoo's `Tr(1)` and seed traces for one `k`, read off the ungauged
    trace with `z ↦ z⁻¹`, in the zoo's data format `{𝖖-power: {(μ-power,):
    int}}`.  Each series is cached at the largest order asked and truncated on
    smaller requests (a truncation of an exact series is exact)."""

    def __init__(self, short_id: str):
        self.short_id = _ZOO_ID[_k(short_id)]
        self._cache: dict = {}

    def _series(self, key, label, K: int) -> dict:
        hit = self._cache.get(key)
        if hit is None or hit[0] < K:
            tr = ungauged_algebra(self.short_id).trace(label, K)
            data: dict = {}
            for q, r in tr.coeffs.items():
                row = {(-z,): int(v) for (z,), v in r.terms.items() if v}
                if row and q <= K:
                    data[q] = row
            hit = (K, data)
            self._cache[key] = hit
        return {q: dict(row) for q, row in hit[1].items() if q <= K}

    def vacuum_trace(self, K: int) -> dict:
        """`Tr(1)` through `𝖖^K`."""
        return self._series("identity", ungauged_algebra(self.short_id)
                            .identity(), K)

    def seed_trace(self, i: int, K: int) -> dict:
        """`Tr(L_{((i, 1),)})` of the zoo generator `i` through `𝖖^K`."""
        gm = generator_map(self.short_id)
        if i not in gm:
            raise KeyError(f"{self.short_id}: no multiplicative generator {i!r}")
        return self._series(i, to_ungauged_label(self.short_id, ((i, 1),)), K)


def seeds(short_id: str) -> _AoddSeeds:
    """The seed server for this entry (shared by an id and its alias):
    `.vacuum_trace(K)` and `.seed_trace(i, K)`, as `elem_traces._seed_series`
    reads them."""
    k = _k(short_id)
    if k not in _SEEDS:
        _SEEDS[k] = _AoddSeeds(short_id)
    return _SEEDS[k]


# ---------------------------------------------------------------------------
# the KAlgebraIso
# ---------------------------------------------------------------------------

_ONE = LaurentPoly.one()
_WORDS: dict = {}         # k -> (singles, pairs, side), the inverse's tables


def _word_tables(short_id: str) -> tuple:
    """`(singles, pairs, side)` for `_zoo_word_of`: the zoo generators whose
    image is one diagonal, keyed by it; those whose image is a pair of
    diagonals, keyed by the pair; and each paired diagonal's magnetic charge
    `mag` in the ungauged class (nonzero, of opposite signs within a pair —
    checked)."""
    k = _k(short_id)
    if k not in _WORDS:
        U = ungauged_algebra(short_id)
        singles: dict = {}
        pairs: dict = {}
        for mg, (F, _f) in generator_map(short_id).items():
            letters = tuple((t, i) for (t, i, m) in F for _ in range(m))
            if len(letters) == 1:
                singles[letters[0]] = mg
            elif len(letters) == 2:
                pairs[letters] = mg
            else:
                raise AssertionError(f"aodd_seeds(k={k}): generator {mg} has "
                                     f"{len(letters)} diagonals")
        side = {d: U.mag((((d[0], d[1], 1),), 0)) for pr in pairs for d in pr}
        if any(side[a] * side[b] >= 0 for a, b in pairs):
            raise AssertionError(f"aodd_seeds(k={k}): a paired generator whose "
                                 f"diagonals do not carry opposite magnetic "
                                 f"charges")
        _WORDS[k] = (singles, pairs, side)
    return _WORDS[k]


def _mu_to_z_inverse(R_zform, R_ungauged) -> RingHom:
    """The zoo's flavour normalisation as a ring isomorphism: the Z-form
    wrapper's `TensorZPlusRing(Trivial, R(U(1)))` (keys `((), (m,))`, the
    zoo's `μ^m`) onto the ungauged class's `R(U(1))` (keys `(f,)`, `z^f`),
    `μ^m ↦ z^{−m}` — the readback `z ↦ z⁻¹` of the module docstring."""
    return RingHom(R_zform, R_ungauged,
                   lambda b: RElement(R_ungauged, {(-b[1][0],): 1}))


def kalgebra_iso(short_id: str, native=None):
    """The generator map as a `KAlgebraIso` from the zoo's Z-form wrapper onto
    `ungauge_u1a1aodd(k)` (module docstring, "The KAlgebraIso").

    `native` is the zoo standalone to wrap, an instance of
    `FINITE_KALGEBRAS[short_id]` (default: a fresh one;
    `finite_kalgebra_objects.kalgebra_object` passes its `'cone-frozen'`
    realization).  The source is `FiniteU1ZKAlgebra(native)` base-changed
    along `μ ↦ z⁻¹`; the target is the shared `ungauged_algebra(short_id)`.
    Nothing is computed here: the label maps build the generator map and the
    inverse's tables at first use."""
    import finite_kalgebras as fk
    from finite_u1_zform import FiniteU1ZKAlgebra
    from kalgebra_iso import KAlgebraIso
    _k(short_id)
    cls = fk.FINITE_KALGEBRAS[short_id]
    if native is None:
        native = cls()
    elif not isinstance(native, cls):
        raise TypeError(f"aodd_seeds.kalgebra_iso({short_id!r}): native is a "
                        f"{type(native).__name__}, not a {cls.__name__}")
    Z = FiniteU1ZKAlgebra(native)
    U = ungauged_algebra(short_id)
    source = Z.base_change(_mu_to_z_inverse(Z.coefficient_ring(),
                                            U.coefficient_ring()))
    free_one = Z.free_ring().one_basis()

    def forward(label) -> Element:
        _w, (word, (m,)) = label
        F, e = to_ungauged_label(short_id, word)
        return Element({(F, e + m): _ONE})

    def inverse(label) -> Element:
        F, e = label
        counts: dict = {}
        for (t, i, m) in F:
            counts[(t, i)] = counts.get((t, i), 0) + m
        word = _canonical_zoo_word(native,
                                   _zoo_word_of(counts, *_word_tables(short_id)))
        F2, e2 = to_ungauged_label(short_id, word)
        if F2 != tuple(F):
            raise ValueError(f"{short_id}: the zoo word {word!r} of {label!r} "
                             f"maps to {F2!r}")
        return Element({(free_one, (word, (e - e2,))): _ONE})

    return KAlgebraIso(source, U, forward, inverse,
                       name=f"{short_id}[z-form→ungauged-u1a1aodd]")
