"""Finite K-algebra zoo.

Each module under this package is a self-contained
:class:`~cone_kalgebra.ConeKAlgebra` subclass with a finite number of
cones — the cluster cone graph of a finite ADE Argyres–Douglas
quiver, plus its multiplication table, frozen as Python data.

The classes have two parallel naming conventions:

* an **AD-pair** name reflecting the Argyres–Douglas theory
  ``A_1[X_n]`` — ``FiniteA1A2KAlgebra``, ``FiniteA1A3KAlgebra``,
  ``FiniteA1A4KAlgebra``, …, ``FiniteA1D3KAlgebra``, …
* a **physical / surface-theory** name reflecting the cluster
  cyclic symmetry:

  - ``A_1[A_{n}]`` cluster algebras are :math:`(n+3)`-gons:
    ``Pentagon`` (A_2), ``Hexagon`` (A_3), ``Heptagon`` (A_4),
    ``Octagon`` (A_5), ``Decagon`` (A_7), …
  - ``A_1[D_n]`` cluster algebras are **once-punctured** n-gons:
    ``PuncturedTriangle`` (D_3), ``PuncturedSquare`` (D_4), …,
    ``PuncturedOctagon`` (D_8).
  - ``A_1[E_n]`` cluster algebras keep their Dynkin labels
    (E_6, E_7, E_8).

Both names are exported as aliases; pick whichever fits the context.

Public API
==========

Registry / enumeration:

* :data:`FINITE_KALGEBRAS` — short-id → class.
* :data:`FINITE_KALGEBRA_IDS` — class → short-id (reverse).

Exact elementary traces (Layer 2 — the chiral-algebra characters;
see :mod:`elem_traces`):

* :func:`elem_traces.trace_residual` — serves every standalone's
  ``_trace_residual`` live (closed-form characters — ``e6`` / ``e8`` from
  their W₃ character recipes, ``w3_seeds``, the design record;
  ``e7`` from ``e7_seeds`` — the geometric classes
  ``A1A2kKAlg(1)`` / ``A1A2kKAlg(2)`` for ``pentagon`` / ``heptagon``,
  ``ungauge_u1a1aodd(k)`` for ``a3`` / ``a5`` / ``a7``,
  ``A1DoddConeKAlg(0)`` for ``a1d3``, ``SU3ADKAlg`` for ``a1d4`` and
  ``A1DevenKAlg(2)`` / ``A1DevenKAlg(3)`` for ``a1d6`` / ``a1d8``, through
  generator maps built at runtime (``aeven_seeds``,
  ``aodd_seeds``, ``a1d3_seeds``,
  ``a1d4_seeds``, ``a1deven_seeds``), and
  the ``a1d5_layer2`` / ``a1d7_layer2`` closed forms for ``a1d5`` /
  ``a1d7``; the Nahm-sum vacuum plus the orthonormality bootstrap stays as a
  witness, :func:`elem_traces.generate`).  No frozen table remains
  (``elem_trace_data.py`` is empty) and nothing falls back to BPS: a seed no
  route serves raises ``NotImplementedError`` naming the entry, the seed and
  the order (for a1d6 / a1d8 past the depth ``A1DevenKAlg``'s trace transport
  reaches, also the class and the limit).
* :func:`elem_traces.generate` — an elementary-trace record through q^K
  (the bootstrap; ``method="bps"`` runs the per-seed BPS engine, as a
  witness, on explicit request only).
* :func:`elem_traces.zoo_trace` — ``trace_residual`` under its former
  name (it was the Layer-1-free trace of the u1 / su2u1 entries); nothing
  calls it.

Geometric labels (the design record; see :mod:`zoo_geometry`):

* :func:`zoo_geometry.geometric_label` — an ``A`` / ``D`` entry's canonical
  label as the multiset of curves (diagonals of the polygon, or curves of the
  once-punctured polygon) naming it in the family class, through the
  certified generator map that serves the entry's traces (``a1d5`` /
  ``a1d7``: ``a1dodd_seeds`` onto ``A1DoddConeKAlg(1)`` /
  ``A1DoddConeKAlg(2)``).  Every ``A`` / ``D`` standalone exposes it as its
  ``geometric_label(label)``.  (The chord-type classifiers below are older
  structural fingerprints of the rays — strings, not these labels.)

Abstract-object view (the design record; see :mod:`finite_kalgebra_objects`):

* :func:`kalgebra_object` — the finite-type algebra as a
  `KAlgebraObject` of certified realizations (cone-frozen + BPS for
  every entry; pentagon/heptagon carry the closed-form and parametric
  presentations too).

Ray labels (cyclic symmetry):

* :func:`rho_orbits(cls)` — ρ-orbits on mg-indices.
* :func:`ray_labels(cls)` — ``{mg: (orbit_letter, cyclic_index)}``.
* :func:`pretty_ray_name(cls, i)` — string ``"L_A_3"``.

Chord-type classifiers (polygon-cluster-algebra geometry):

* :func:`short_chord_orbit(cls)` — orbit with max q-commutation
  density (= the short chord class).
* :func:`chord_qc_signature(cls)` — per-ray q-commutation
  fingerprint against the short orbit.
* :func:`signature_period(s)` — smallest p dividing ``len(s)`` for
  which ``s`` is p-periodic.
* :func:`chord_parity_label(cls)` — ``{mg: "mixed"|"doubled"}``.
* :func:`chord_intersection_graph(cls)` — chord-intersection
  adjacency (q_commute negated).
* :func:`predict_cross_product_term_count(cls, i, j)` — look up
  cross_product term count.
* :func:`identify_skip_classes(cls)` — geometric chord-type ID for
  A_1[A_odd]: ``"skip3_single_3"``, ``"doubled_skip2_skip4_5"``, …
* :func:`identify_a1dn_chord_classes(cls)` — geometric chord-type
  ID for A_1[D_n] n odd: ``"a1dn_skip2_even_0"``, …
* :func:`chord_labels(cls)` — top-level dispatcher; picks the
  appropriate classifier per algebra.
"""

from finite_pentagon_kalg import FinitePentagonKAlgebra
from finite_heptagon_kalg import FiniteHeptagonKAlgebra
from finite_a3_kalg import FiniteA3KAlgebra
from finite_a5_kalg import FiniteA5KAlgebra
from finite_a7_kalg import FiniteA7KAlgebra
from finite_a1d3_kalg import FiniteA1D3KAlgebra
from finite_a1d4_kalg import FiniteA1D4KAlgebra
from finite_a1d5_kalg import FiniteA1D5KAlgebra
from finite_a1d6_kalg import FiniteA1D6KAlgebra
from finite_a1d7_kalg import FiniteA1D7KAlgebra
from finite_a1d8_kalg import FiniteA1D8KAlgebra
from finite_e6_kalg import FiniteE6KAlgebra
from finite_e7_kalg import FiniteE7KAlgebra

# Parametric hard-coded closed-form families.
#
# ``A1A2kKAlg(k)`` is A_1[A_{2k}] = the trivial-flavor odd-polygon
# family (pentagon for k=1, heptagon for k=2, nonagon for k=3, …),
# implemented via the Plücker closed-form base table from
# ``A1A2k_plucker_closed_form.py``.  No BPSKAlgebra runtime dependency.
#
# Sibling parametric families are open work items:
#   * FiniteA1AOddKAlgebra(n=2k+1) -- A_1[A_{2k+1}] = U(1)-flavored
#     even-polygon family (hexagon, octagon, decagon, dodecagon, …).
#   * FiniteA1DOddKAlgebra(n)  -- A_1[D_n] for n odd (SU(2) flavor).
#   * FiniteA1DEvenKAlgebra(n) -- A_1[D_n] for n even (SU(2)×U(1)
#     flavor, plus D_4 triality at n=4).
# Each of these requires a closed-form Plücker-style derivation
# (cocycles get progressively more intricate); pending that work,
# only the specific-n standalones (finite_a3, finite_a5, ...,
# finite_a1d8) are available.
from a1a2k_kalg import A1A2kKAlg
from finite_e8_kalg import FiniteE8KAlgebra

# AD-pair aliases.
FiniteA1A2KAlgebra = FinitePentagonKAlgebra   # A_1[A_2] = pentagon (Z_5)
FiniteA1A3KAlgebra = FiniteA3KAlgebra         # A_1[A_3] = hexagon (Z_6)
FiniteA1A4KAlgebra = FiniteHeptagonKAlgebra   # A_1[A_4] = heptagon (Z_7)
FiniteA1A5KAlgebra = FiniteA5KAlgebra         # A_1[A_5] = octagon (Z_8)

# Physical / polygon aliases for the odd-A series.
FiniteHexagonKAlgebra = FiniteA3KAlgebra       # 6-gon
FiniteOctagonKAlgebra = FiniteA5KAlgebra       # 8-gon
FiniteDecagonKAlgebra = FiniteA7KAlgebra       # 10-gon

# AD-pair alias.
FiniteA1A7KAlgebra = FiniteA7KAlgebra          # A_1[A_7] = decagon

# Punctured-polygon names for the A_1[D_n] series.
# A_1[D_n] = once-punctured n-gon (n nodes around the perimeter +
# one puncture node).
FinitePuncturedTriangleKAlgebra = FiniteA1D3KAlgebra   # punctured 3-gon
FinitePuncturedSquareKAlgebra   = FiniteA1D4KAlgebra   # punctured 4-gon
FinitePuncturedPentagonKAlgebra = FiniteA1D5KAlgebra   # punctured 5-gon
FinitePuncturedHexagonKAlgebra  = FiniteA1D6KAlgebra   # punctured 6-gon
FinitePuncturedHeptagonKAlgebra = FiniteA1D7KAlgebra   # punctured 7-gon
FinitePuncturedOctagonKAlgebra  = FiniteA1D8KAlgebra   # punctured 8-gon

# AD pair aliases for E-series.
FiniteA1E6KAlgebra = FiniteE6KAlgebra
FiniteA1E7KAlgebra = FiniteE7KAlgebra
FiniteA1E8KAlgebra = FiniteE8KAlgebra

# Registry by short id.  Each value is the canonical (physical) class.
FINITE_KALGEBRAS = {
    "pentagon": FinitePentagonKAlgebra,     # A_1[A_2]   — Z_5
    "hexagon":  FiniteHexagonKAlgebra,       # A_1[A_3]   — Z_6
    "heptagon": FiniteHeptagonKAlgebra,     # A_1[A_4]   — Z_7
    "octagon":  FiniteOctagonKAlgebra,       # A_1[A_5]   — Z_8
    "decagon":  FiniteDecagonKAlgebra,       # A_1[A_7]   — Z_10  (partial)
    "a3":       FiniteA3KAlgebra,           # alias of hexagon
    "a5":       FiniteA5KAlgebra,           # alias of octagon
    "a7":       FiniteA7KAlgebra,           # alias of decagon
    "a1d3":     FiniteA1D3KAlgebra,         # A_1[D_3]
    "a1d4":     FiniteA1D4KAlgebra,         # A_1[D_4]
    "a1d5":     FiniteA1D5KAlgebra,         # A_1[D_5]
    "a1d6":     FiniteA1D6KAlgebra,         # A_1[D_6]
    "a1d7":     FiniteA1D7KAlgebra,         # A_1[D_7]
    "a1d8":     FiniteA1D8KAlgebra,         # A_1[D_8] (partial, see file note)
    "e6":       FiniteE6KAlgebra,           # A_1[E_6]
    "e7":       FiniteE7KAlgebra,           # A_1[E_7]
    "e8":       FiniteE8KAlgebra,           # A_1[E_8]
}

# Reverse: ConeKAlgebra subclass → short id (useful for tests).
FINITE_KALGEBRA_IDS = {
    cls: short_id for short_id, cls in FINITE_KALGEBRAS.items()
}


# --------------------------------------------------------------------
# Cyclic-symmetry ray labels.
#
# Each finite cone algebra has a ρ-permutation acting on its mult-gen
# indices.  This splits the mg's into ρ-orbits whose length is the
# cyclic order at that orbit.  Cluster theorists usually want to name
# rays by ``(orbit-rep, cyclic-shift)`` rather than by an arbitrary
# integer index — e.g. for the pentagon, the 5 rays are
# ``L_A_0, L_A_1, L_A_2, L_A_3, L_A_4`` (one ρ-orbit of length 5).
# For the heptagon (14 rays, 2 orbits of length 7), they are
# ``L_A_0..L_A_6, L_B_0..L_B_6``.
# --------------------------------------------------------------------

def _rho_perm_of(cls):
    """Recover the ρ-permutation as a dict {i: ρ(i)} on mg indices."""
    A = cls()
    n = len(A.cone_data().mult_gens())
    # Most generated classes expose _rho_perm directly.
    perm = getattr(cls, "_rho_perm", None) or getattr(A, "_rho_perm", None)
    if perm is not None:
        return {int(k): int(v) for k, v in perm.items()}
    # Older E6 / E8 standalones embed it as a module-level CONST.
    import importlib
    mod = importlib.import_module(cls.__module__)
    for name in dir(mod):
        if name.endswith("_RHO_PERM"):
            return {int(k): int(v) for k, v in getattr(mod, name).items()}
    # Fallback: derive from rho() applied to single-mg labels.
    out = {}
    for i in range(n):
        img = A.rho(((i, 1),))
        if len(img) == 1 and img[0][1] == 1:
            out[i] = int(img[0][0])
    return out


def rho_orbits(cls):
    """Return the list of ρ-orbits on the mg-indices of ``cls``.

    Each orbit is a tuple ``(i_0, ρ(i_0), ρ²(i_0), …)`` listing the
    mg indices in their natural cyclic order.  Orbits are sorted by
    length (descending), then by the minimum index they contain.
    """
    A = cls()
    n = len(A.cone_data().mult_gens())
    perm = _rho_perm_of(cls)
    seen: set[int] = set()
    orbits: list = []
    for start in range(n):
        if start in seen:
            continue
        orbit = [start]
        seen.add(start)
        nxt = perm.get(start, start)
        while nxt != start and nxt not in seen:
            orbit.append(nxt)
            seen.add(nxt)
            nxt = perm.get(nxt, nxt)
        orbits.append(tuple(orbit))
    orbits.sort(key=lambda o: (-len(o), min(o)))
    return orbits


def ray_labels(cls):
    """Return a dict ``{mg_index: (orbit_letter, cyclic_index)}``.

    Letters are assigned by walking the **reference cone** (the lex-min
    cluster cone, which is the root chart's cone for the usual build).
    For each ray in the reference cone, the orbit it belongs to gets
    the next free letter (A, B, C, …).  Within an orbit the
    cyclic_index counts ρ-powers starting from the smallest in-cone
    member of that orbit (or from the smallest mg index, if the orbit
    has no in-cone member).

    This matches the cluster-theoretic convention where the orbit
    label is identified by a reference-cluster ray and ``i`` simply
    shifts along the cluster cyclic symmetry.
    """
    A = cls()
    cd = A.cone_data()
    cones = list(cd.cones())
    cones.sort(key=sorted)
    ref_cone_set = set(cones[0]) if cones else set()
    orbits = rho_orbits(cls)

    orbit_letter: dict = {}
    next_idx = 0
    # Reference-cone rays in their natural (sorted) order.
    for r in sorted(ref_cone_set):
        for orbit_no, orbit in enumerate(orbits):
            if r in orbit and orbit_no not in orbit_letter:
                letter = (chr(ord("A") + next_idx)
                          if next_idx < 26 else f"O{next_idx}")
                orbit_letter[orbit_no] = letter
                next_idx += 1
                break
    # Orphan orbits (no reference-cone member) get trailing letters.
    for orbit_no in range(len(orbits)):
        if orbit_no not in orbit_letter:
            letter = (chr(ord("A") + next_idx)
                      if next_idx < 26 else f"O{next_idx}")
            orbit_letter[orbit_no] = letter
            next_idx += 1

    out: dict = {}
    for orbit_no, orbit in enumerate(orbits):
        letter = orbit_letter[orbit_no]
        in_cone = [i for i in orbit if i in ref_cone_set]
        root = min(in_cone) if in_cone else min(orbit)
        start = orbit.index(root)
        rotated = orbit[start:] + orbit[:start]
        for k, i in enumerate(rotated):
            out[i] = (letter, k)
    return out


def pretty_ray_name(cls, i: int) -> str:
    """Return a string like ``"L_A_3"`` for mg index ``i`` in ``cls``."""
    letter, k = ray_labels(cls)[i]
    return f"L_{letter}_{k}"


def a1aodd_chord_types(cls):
    """Chord-type classification for A_1[A_n] with n odd (even-gon
    polygon cluster algebras: Hexagon, Octagon, Decagon, …).

    Polygon = (n+3)-gon (even).  Cluster rays come in two families:

    * **Single chords** — full ρ-orbit of length n+3, characterized by
      their q-commutation density (the "shortest" chord has the
      highest density; longer chords have lower).
    * **Doubled chords** ((e,e)(o,o) atomic pairs) and diameters —
      ρ-orbit of half-length (n+3)/2, fixed by ρ^{(n+3)/2}.

    Returns ``{mg_idx: kind}`` where ``kind`` is one of
    ``"short"``, ``"medium-1"``, ``"medium-2"``, …, ``"long"`` for
    single-chord orbits, or ``"doubled-1"``, ``"doubled-2"``, … for
    half-length orbits.  Orbits are ordered first by length
    (full first), then by qc-density.
    """
    A = cls()
    cd = A.cone_data()
    orbits = rho_orbits(cls)
    if not orbits:
        return {}
    full_len = max(len(o) for o in orbits)
    # Group orbits by length tier
    fulls = [(i, o) for i, o in enumerate(orbits) if len(o) == full_len]
    halves = [(i, o) for i, o in enumerate(orbits) if len(o) < full_len]
    # Rank full-length orbits by qc-density descending.
    def qc(o):
        rep = o[0]
        return sum(1 for x in o[1:] if cd.q_commute(rep, x))
    fulls.sort(key=lambda t: (-qc(t[1]) / max(1, len(t[1]) - 1), t[1][0]))
    halves.sort(key=lambda t: (-qc(t[1]) / max(1, len(t[1]) - 1), t[1][0]))
    out: dict = {}
    rank_to_name = ["short", "medium-1", "medium-2", "medium-3",
                    "medium-4", "medium-5", "long"]
    for rank, (_idx, orbit) in enumerate(fulls):
        if rank == 0:
            name = "short"
        elif rank == len(fulls) - 1:
            name = "long"
        elif rank < len(rank_to_name):
            name = rank_to_name[rank]
        else:
            name = f"single-{rank}"
        for mg in orbit:
            out[mg] = name
    for rank, (_idx, orbit) in enumerate(halves):
        name = "doubled" if len(halves) == 1 else f"doubled-{rank + 1}"
        for mg in orbit:
            out[mg] = name
    return out


def short_chord_orbit(cls):
    """Return the ρ-orbit identified as the 'short chord' class
    (max q-commutation density among full-length orbits).
    """
    orbits = rho_orbits(cls)
    if not orbits:
        return ()
    full_len = max(len(o) for o in orbits)
    A = cls()
    cd = A.cone_data()
    def qc(o):
        return sum(1 for x in o[1:] if cd.q_commute(o[0], x))
    fulls = [o for o in orbits if len(o) == full_len]
    fulls.sort(key=lambda o: (-qc(o) / max(1, len(o) - 1), o[0]))
    return fulls[0]


def chord_qc_signature(cls):
    """Return ``{mg_idx: signature}`` where the signature is a tuple
    of bits ``(q_commute(mg, short[0]), q_commute(mg, short[1]), …)``
    indicating how the ray q-commutes with each cyclic shift of the
    short-chord orbit.

    Within a single ρ-orbit, signatures are cyclic shifts of each
    other; different orbits give different fundamental patterns, so
    the multiset of distinct cyclic-shift-classes of signatures
    refines the chord-type identification.
    """
    A = cls()
    cd = A.cone_data()
    short = short_chord_orbit(cls)
    if not short:
        return {}
    n_mg = len(A.cone_data().mult_gens())
    sig: dict = {}
    for i in range(n_mg):
        sig[i] = tuple(
            1 if cd.q_commute(i, s) else 0
            for s in short
        )
    return sig


def signature_period(s):
    """Smallest p dividing len(s) such that s = (s[:p]) * (len(s)//p).

    Periods smaller than len(s) indicate the orbit has additional
    cyclic symmetry — the hallmark of a doubled (e,e)(o,o) chord
    pair (which is invariant under a sub-cyclic rotation).
    """
    n = len(s)
    for p in range(1, n + 1):
        if n % p == 0 and s == (s[:p] * (n // p)):
            return p
    return n


def chord_parity_label(cls):
    """Per-ray parity label for A_1[A_n], n odd.

    Combines two structural signals:

    * Signature **period** < length → ρ-orbit has additional cyclic
      symmetry (the hallmark of doubled rays in larger polygons or
      diameter rays).
    * For orbits where the period doesn't fire (small polygons),
      fall back to **qc-bit count against the short orbit**.
      Doubled rays require both component sub-chords to be non-
      crossing → strictly fewer q-commutations than single mixed
      chords.  The orbit with maximum qc-bit count is single mixed;
      the rest are doubled.

    Returns ``{mg: 'mixed'|'doubled'}``.
    """
    sigs = chord_qc_signature(cls)
    if not sigs:
        return {}
    orbits = rho_orbits(cls)
    full_len = max(len(o) for o in orbits)

    # Group orbits by qc-bit count against the short orbit.
    orbit_qc: dict = {}  # orbit_idx -> qc_bits
    orbit_period: dict = {}
    orbit_full_member: dict = {}  # idx -> True if length == full_len
    for orb_idx, orbit in enumerate(orbits):
        rep = orbit[0]
        orbit_qc[orb_idx] = sum(sigs[rep])
        orbit_period[orb_idx] = signature_period(sigs[rep])
        orbit_full_member[orb_idx] = (len(orbit) == full_len)

    # Identify "single mixed" orbits: full-length AND period == full
    # AND maximum qc-bits among such orbits.
    full_period_full_length = [
        i for i in range(len(orbits))
        if orbit_full_member[i] and orbit_period[i] == len(sigs[orbits[i][0]])
    ]
    # Among these, single mixed has max qc-bits.
    if full_period_full_length:
        max_qc = max(orbit_qc[i] for i in full_period_full_length)
        single_mixed_set = {
            i for i in full_period_full_length if orbit_qc[i] == max_qc
        }
    else:
        # Degenerate small case (hexagon): fall back to qc-bit ranking
        # over ALL orbits.
        max_qc = max(orbit_qc.values())
        single_mixed_set = {
            i for i in range(len(orbits)) if orbit_qc[i] == max_qc
        }

    out: dict = {}
    for orb_idx, orbit in enumerate(orbits):
        label = "mixed" if orb_idx in single_mixed_set else "doubled"
        for mg in orbit:
            out[mg] = label
    return out


def _enumerate_a1dn_chords(n):
    """Enumerate A_1[D_n] chords on the 2n-cover.

    For n odd: chords are oriented (a, a+k) on Z_{2n} with skip
    k ∈ {2, 4, …, n-1}.  (a, b) is distinct from (b, a+n) —
    winding around the puncture matters.

    For n even: includes skip-n "diameter" chords; structure
    differs slightly.
    """
    cover = 2 * n
    chords = []
    max_k = n - 1 if n % 2 == 1 else n
    for a in range(cover):
        for k in range(2, max_k + 1, 2):
            b = (a + k) % cover
            chords.append({
                "chord": (a, b),
                "skip": k,
            })
    return chords


def chord_labels(cls):
    """Top-level dispatcher: return per-ray geometric chord labels.

    * For A_1[A_n] with n odd (Hexagon, Octagon, Decagon, …):
      use :func:`identify_skip_classes` — labels like
      ``"skip3_single_3"``, ``"doubled_skip2_skip4_5"``.
    * For A_1[D_n] with n odd (A_1D_3, A_1D_5, A_1D_7, …):
      use :func:`identify_a1dn_chord_classes` — labels like
      ``"a1dn_skip2_even_0"``, ``"a1dn_skip4_odd_3"``.
    * For all other algebras (Pentagon, Heptagon, E-series,
      A_1D_n n even): fall back to :func:`pretty_ray_name` —
      labels like ``"L_A_0"``.

    Returns ``{mg_idx: label_string}`` for the given class.
    """
    # Resolve to canonical short-id (handles aliases like a3↔hexagon).
    polygon_ids = {"hexagon", "octagon", "decagon", "a3", "a5", "a7"}
    a1dn_odd_ids = {"a1d3", "a1d5", "a1d7"}
    matched = {sid for sid, c in FINITE_KALGEBRAS.items() if c is cls}
    if matched & polygon_ids:
        return identify_skip_classes(cls)
    if matched & a1dn_odd_ids:
        return identify_a1dn_chord_classes(cls)
    A = cls()
    n_mg = len(A.cone_data().mult_gens())
    return {i: pretty_ray_name(cls, i) for i in range(n_mg)}


def identify_a1dn_chord_classes(cls):
    """For A_1[D_n] with n odd: identify each mg orbit's chord-type.

    On the 2n-cover with algebra ρ acting as rotation-by-2, each
    skip class {2, 4, …, n-1} gives 2 orbits of length n
    (even-start and odd-start).  We compute each chord type's
    canonical q-commutation signature against the algebra's short
    chord orbit and match.

    Returns ``{mg_idx: chord_type_label}`` such as
    ``"a1dn_skip2_even_3"`` or ``"a1dn_skip4_odd_1"``.
    """
    orbits = rho_orbits(cls)
    if not orbits:
        return {}
    n_mg = sum(len(o) for o in orbits)
    # Infer n from chord count: n_mg = n(n-1) for n odd.
    n = None
    for trial in range(3, 20, 2):
        if trial * (trial - 1) == n_mg:
            n = trial
            break
    if n is None:
        return chord_geometric_labels(cls)
    cover = 2 * n

    # Algebra ρ-orbit length should be n.
    if not all(len(o) == n for o in orbits):
        return chord_geometric_labels(cls)

    # Enumerate geometric chord-orbits on the cover under ρ-by-2.
    # For each (skip, start_parity), the orbit is
    #   (start, start + skip), (start+2, start+skip+2), ...
    geometric_orbits = []
    for skip in range(2, n, 2):  # skip ∈ {2, 4, …, n-1}
        for start_par in (0, 1):
            orb = [((start_par + 2 * i) % cover,
                    (start_par + 2 * i + skip) % cover)
                   for i in range(n)]
            geometric_orbits.append({
                "skip": skip,
                "start_parity": "even" if start_par == 0 else "odd",
                "orbit": orb,
            })

    # Reference short chords on the cover: skip-2 even-start
    # orbit = (0,2), (2,4), …, (2n-2, 0).
    ref_orbit = [(2 * i, (2 * i + 2) % cover) for i in range(n)]

    def chord_sig(chord):
        return tuple(
            0 if _chord_intersects(chord, ref, cover) else 1
            for ref in ref_orbit
        )

    # Signature per geometric orbit (canonical cyclic).
    geo_sig_to_label: dict = {}
    for gorb in geometric_orbits:
        rep_chord = gorb["orbit"][0]
        sig = chord_sig(rep_chord)
        canon = _canonical_cyclic(sig)
        label = f"a1dn_skip{gorb['skip']}_{gorb['start_parity']}"
        # Conflicts: if two geometric orbits give the same canonical
        # signature, append _A, _B, ...
        if canon in geo_sig_to_label:
            existing = geo_sig_to_label[canon]
            if isinstance(existing, list):
                existing.append(label)
            else:
                geo_sig_to_label[canon] = [existing, label]
        else:
            geo_sig_to_label[canon] = label

    # Compute mg orbit signatures and match.
    A = cls()
    cd = A.cone_data()
    short_alg = short_chord_orbit(cls)
    if len(short_alg) != n:
        return chord_geometric_labels(cls)
    mg_sigs = chord_qc_signature(cls)

    ray_lab = ray_labels(cls)
    out: dict = {}
    seen_label_counts: dict = {}
    for orbit in orbits:
        rep = orbit[0]
        canon = _canonical_cyclic(mg_sigs[rep])
        label = geo_sig_to_label.get(canon)
        if label is None:
            # The algebra's short orbit itself has all-1 self-signature
            # (it doesn't intersect any of its own cyclic shifts), and
            # the geometric reference signature is the same.  Assign
            # "skip2_even" (= the reference class).
            base = "a1dn_skip2_even"
        elif isinstance(label, list):
            base = label[seen_label_counts.get(tuple(label), 0)]
            seen_label_counts[tuple(label)] = (
                seen_label_counts.get(tuple(label), 0) + 1
            )
        else:
            base = label
        for mg in orbit:
            cyclic_i = ray_lab[mg][1]
            out[mg] = f"{base}_{cyclic_i}"
    return out


def _chord_intersects(c1, c2, n):
    """Two chords c1=(a, b), c2=(c, d) in n-gon: intersect iff
    endpoints interleave cyclically.
    """
    a, b = c1
    c, d = c2
    if {a, b} & {c, d}:
        return False  # share endpoint
    # Walk cyclically from a to b (exclusive). Count how many of {c, d}
    # fall in the open arc (a, b).
    arc1 = set()
    x = (a + 1) % n
    while x != b:
        arc1.add(x); x = (x + 1) % n
    in1 = sum(1 for v in (c, d) if v in arc1)
    return in1 == 1


def _enumerate_polygon_chords(n):
    """Return all chords (a, b) in n-gon as canonical pairs (a < b),
    classified by parity and skip-class.
    """
    chords = []
    for a in range(n):
        for k in range(2, n - 1):
            b = (a + k) % n
            if a < b:
                pair = (a, b)
            else:
                pair = (b, a)
            chords.append({
                "chord": pair,
                "skip": min(k, n - k),
                "mixed": (k % 2) == 1,  # mixed parity iff skip odd
            })
    # Deduplicate
    seen = set()
    out = []
    for c in chords:
        if c["chord"] not in seen:
            seen.add(c["chord"])
            out.append(c)
    return out


def _canonical_cyclic(s):
    return min(s[i:] + s[:i] for i in range(len(s)))


def identify_skip_classes(cls):
    """For A_1[A_n], n odd: identify each mg orbit's geometric
    chord-type by matching its q-commutation signature against
    polygon chords (and doubled pairs).

    Returns ``{mg_idx: chord_type_name}`` with names like
    ``"skip3_single"`` (single mixed chord, skip-3) or
    ``"skip5_diameter"`` (mixed-parity diameter) or
    ``"doubled_ee2_oo2"`` ((e,e)-skip-2 + (o,o)-skip-2 pair), etc.

    Falls back to generic ``"orbit_X"`` if no clean geometric match.
    """
    orbits = rho_orbits(cls)
    if not orbits:
        return {}
    # Infer polygon size from full ρ-orbit length.  For A_1[A_n] of
    # n+3 odd polygons, full_len = n+3.  For even polygons (n odd),
    # ρ has order (n+3)/2 in some cases (hexagon), full polygon = 2 * full_len.
    full_len = max(len(o) for o in orbits)
    polygon_n = full_len
    # Heuristic adjustment: if polygon_n < 6, double (hexagon case).
    if polygon_n == 3:
        polygon_n = 6
    elif polygon_n < 6:
        polygon_n *= 2

    sigs = chord_qc_signature(cls)
    parity = chord_parity_label(cls)
    short = short_chord_orbit(cls)

    # Reference skip-3 chords in the polygon.  Cycling skip-3 by 1
    # gives polygon_n distinct chord-pairs unless polygon_n is divisible
    # by something; dedupe.
    ref_chords_raw = [(i, (i + 3) % polygon_n) for i in range(polygon_n)]
    ref_chords_set = set()
    ref_chords = []
    for a, b in ref_chords_raw:
        pair = (min(a, b), max(a, b))
        if pair not in ref_chords_set:
            ref_chords_set.add(pair)
            ref_chords.append(pair)
    # Ensure same length as our short orbit signature.
    sig_len = len(sigs[next(iter(sigs))])
    if len(ref_chords) != sig_len:
        # Polygon inference uncertain; fall back to chord_geometric_labels.
        return chord_geometric_labels(cls)

    def chord_signature(chord):
        """0/1 for non-intersecting with each ref_chord."""
        return tuple(
            0 if _chord_intersects(chord, ref, polygon_n) else 1
            for ref in ref_chords
        )

    def doubled_signature(c1, c2):
        """(e,e)(o,o) pair: doesn't q-commute with ref iff EITHER
        component intersects (= q-commutes iff BOTH non-intersect).
        """
        return tuple(
            0 if (_chord_intersects(c1, ref, polygon_n)
                  or _chord_intersects(c2, ref, polygon_n)) else 1
            for ref in ref_chords
        )

    # Build sig → chord_type_name map from polygon enumeration.
    chord_db = {}  # canonical_sig -> name
    all_chords = _enumerate_polygon_chords(polygon_n)
    for c in all_chords:
        s = _canonical_cyclic(chord_signature(c["chord"]))
        k = c["skip"]
        if c["mixed"]:
            name = f"skip{k}_single"
            if k == polygon_n // 2:
                name = f"skip{k}_diameter"
        else:
            # Same-parity single chord — these don't exist as single
            # rays in the algebra (only doubled).
            continue
        chord_db.setdefault(s, name)

    # Doubled chord signatures: enumerate non-crossing same-parity pairs.
    same_chords = [c for c in all_chords if not c["mixed"]]
    seen_doubled = set()
    for i, c1 in enumerate(same_chords):
        for c2 in same_chords[i + 1:]:
            # Must be one ee and one oo.
            p1 = c1["chord"][0] % 2
            p2 = c2["chord"][0] % 2
            if p1 == p2:
                continue
            # Must be non-crossing.
            if _chord_intersects(c1["chord"], c2["chord"], polygon_n):
                continue
            s = _canonical_cyclic(doubled_signature(c1["chord"], c2["chord"]))
            k1 = c1["skip"]; k2 = c2["skip"]
            ks = tuple(sorted([k1, k2]))
            name = f"doubled_skip{ks[0]}_skip{ks[1]}"
            if s not in seen_doubled:
                chord_db.setdefault(s, name)
                seen_doubled.add(s)

    # Now match each mg orbit's canonical signature to the database.
    out: dict = {}
    ray_lab = ray_labels(cls)
    unmatched_counter = 0
    type_seen = {}  # type_name -> orbit_letter for disambiguation
    for orbit in orbits:
        rep = orbit[0]
        cs = _canonical_cyclic(sigs[rep])
        name = chord_db.get(cs)
        if name is None:
            unmatched_counter += 1
            name = f"orbit_unk{unmatched_counter}"
        # If same type-name appears for multiple orbits, distinguish by orbit letter.
        if name in type_seen:
            type_seen[name] += 1
            disp_name = f"{name}_{chr(ord('A') + type_seen[name] - 1)}"
        else:
            type_seen[name] = 1
            disp_name = name
        for mg in orbit:
            cyclic_i = ray_lab[mg][1]
            out[mg] = f"{disp_name}_{cyclic_i}"
    return out


def chord_intersection_graph(cls):
    """Return ``{mg_idx: frozenset of mg's that intersect this chord}``.

    For polygon-cluster-algebra interpretation, q_commute iff non-
    intersecting; so this returns the chord-intersection graph
    on rays.
    """
    A = cls()
    cd = A.cone_data()
    n_mg = len(cd.mult_gens())
    out: dict = {}
    for i in range(n_mg):
        out[i] = frozenset(
            j for j in range(n_mg)
            if i != j and not cd.q_commute(i, j)
        )
    return out


def predict_cross_product_term_count(cls, i: int, j: int) -> int:
    """Look up the cross-product term count for the pair (i, j).

    For a polygon cluster algebra, the number of terms in
    L_i · L_j (when i, j are NOT in a shared cone, i.e. their
    chords intersect) is controlled by the intersection count and
    chord types.  This returns the actual stored term count from
    ``cross_product(i, j)``.
    """
    A = cls()
    cd = A.cone_data()
    return len(cd.cross_product(i, j))


def chord_geometric_labels(cls):
    """Assign per-ray labels with chord type + structural identifier.

    Combines :func:`chord_parity_label` (mixed vs doubled) with the
    chord's q-commutation signature (a structural fingerprint) and
    the cyclic-orbit position.

    Returns ``{mg_idx: label_string}`` where label is e.g.
    ``"mixed_short_3"`` (mixed-parity single chord, short class,
    cyclic position 3) or ``"doubled_A_0"`` (doubled atomic pair,
    family A, cyclic position 0).
    """
    parity = chord_parity_label(cls)
    sigs = chord_qc_signature(cls)
    orbits = rho_orbits(cls)
    ray_lab = ray_labels(cls)
    # Group orbits by parity → assign sub-labels within each group
    mixed_orbits = []
    doubled_orbits = []
    for orbit in orbits:
        if parity.get(orbit[0]) == "mixed":
            mixed_orbits.append(orbit)
        else:
            doubled_orbits.append(orbit)

    def cycle_canon(s):
        return min(s[k:] + s[:k] for k in range(len(s)))

    # Sort orbits by canonical signature (deterministic naming).
    mixed_orbits.sort(key=lambda o: cycle_canon(sigs[o[0]]))
    doubled_orbits.sort(key=lambda o: cycle_canon(sigs[o[0]]))

    out: dict = {}
    for grp_idx, orbit in enumerate(mixed_orbits):
        letter = chr(ord("A") + grp_idx)
        for mg in orbit:
            cyclic_i = ray_lab[mg][1]
            out[mg] = f"mixed_{letter}_{cyclic_i}"
    for grp_idx, orbit in enumerate(doubled_orbits):
        letter = chr(ord("A") + grp_idx)
        for mg in orbit:
            cyclic_i = ray_lab[mg][1]
            out[mg] = f"doubled_{letter}_{cyclic_i}"
    return out


def orbit_chord_classification(cls):
    """Classify ρ-orbits by their q-commutation density.

    A "short chord" in the polygon-cluster-algebra interpretation
    q-commutes with the maximum number of its cyclic shifts; longer
    chords (and doubled-chord rays) q-commute with fewer.

    Returns a list of dicts ``{rep, length, qc_count, qc_fraction,
    rank}`` sorted by qc_fraction descending.  The top entry is the
    "short" chord orbit; subsequent entries are progressively longer
    or doubled.
    """
    A = cls()
    cd = A.cone_data()
    orbits = rho_orbits(cls)
    out = []
    for orbit in orbits:
        rep = orbit[0]
        h = len(orbit)
        qc = sum(1 for o in orbit[1:] if cd.q_commute(rep, o))
        out.append({
            "rep": rep,
            "length": h,
            "qc_count": qc,
            "qc_fraction": qc / (h - 1) if h > 1 else 0.0,
        })
    out.sort(key=lambda x: (-x["qc_fraction"], -x["length"], x["rep"]))
    for rank, entry in enumerate(out):
        entry["rank"] = rank
    return out

__all__ = [
    "FINITE_KALGEBRAS", "FINITE_KALGEBRA_IDS",
    "rho_orbits", "ray_labels", "pretty_ray_name",
    "orbit_chord_classification", "a1aodd_chord_types",
    "short_chord_orbit", "chord_qc_signature",
    "signature_period", "chord_parity_label",
    "chord_intersection_graph", "predict_cross_product_term_count",
    "chord_geometric_labels", "identify_skip_classes",
    "identify_a1dn_chord_classes", "chord_labels",
    # Physical / polygon names.
    "FinitePentagonKAlgebra", "FiniteHexagonKAlgebra",
    "FiniteHeptagonKAlgebra", "FiniteOctagonKAlgebra",
    "FiniteDecagonKAlgebra",
    # A-series ADE.
    "FiniteA3KAlgebra", "FiniteA5KAlgebra", "FiniteA7KAlgebra",
    # AD-pair aliases.
    "FiniteA1A2KAlgebra", "FiniteA1A3KAlgebra",
    "FiniteA1A4KAlgebra", "FiniteA1A5KAlgebra",
    "FiniteA1A7KAlgebra",
    # D-series (= punctured polygons).
    "FiniteA1D3KAlgebra", "FiniteA1D4KAlgebra",
    "FiniteA1D5KAlgebra", "FiniteA1D6KAlgebra",
    "FiniteA1D7KAlgebra", "FiniteA1D8KAlgebra",
    "FinitePuncturedTriangleKAlgebra", "FinitePuncturedSquareKAlgebra",
    "FinitePuncturedPentagonKAlgebra", "FinitePuncturedHexagonKAlgebra",
    "FinitePuncturedHeptagonKAlgebra", "FinitePuncturedOctagonKAlgebra",
    # E-series.
    "FiniteE6KAlgebra", "FiniteE7KAlgebra", "FiniteE8KAlgebra",
    "FiniteA1E6KAlgebra", "FiniteA1E7KAlgebra", "FiniteA1E8KAlgebra",
    # Parametric hard-coded closed-form families.
    "A1A2kKAlg",
]


def kalgebra_object(short_id):
    """Lazy convenience for :func:`finite_kalgebra_objects.kalgebra_object`
    (kept lazy so importing the registry does not pull in the BPS
    machinery)."""
    from finite_kalgebra_objects import kalgebra_object as _ko
    return _ko(short_id)
