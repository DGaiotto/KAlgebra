"""`KAlgebraObject`s for the **hexagon pair**:

* `u1hexagon_object()` — the **U(1)-gauged hexagon** = u(1)-gauged
  `[A₁, A₃]` (k = 1 of the `U1A1Aodd` family on the (2k+4)-gon).
* `hexagon_object()`   — the **hexagon** = μ-flavoured `[A₁, A₃]`,
  realized from the gauged algebra by the general U(1) **ungauger**
  (`ungauge_kalgebra.UngaugedKAlgebra`: centralizer of the electric
  generator `E`, `E` promoted to the flavour fugacity μ, trace with the
  restored `(𝖖²;𝖖²)²_∞` vector-multiplet measure).

The pair is kept together deliberately: the ungauging map *is* the
instructive content — `hexagon_object`'s `'ungauged'` realization is
literally `UngaugedKAlgebra` applied to `u1hexagon_object`'s
`'intrinsic'` realization (`hexagon_ungauging()` exposes the map).

Presentations and witnesses
---------------------------

Gauged (`u1hexagon_object`):

* ``'intrinsic'`` — `U1A1AoddKAlg(1)`: closed-form cone presentation;
  the contract-correct gauged hexagon (verifier battery clean,
  including E-shifted labels).  NOTE: the older `U1HexagonKAlg`
  (frozen Plücker table) presents the same algebra up to the relabel
  `E ↦ E⁻¹`, but carries a known ρ/trace inconsistency on long chords,
  so it is *not* registered here.
* ``'bps'`` — `BPSKAlgebra` on the gauged-hexagon quiver
  `O₁ → O₂ → O₃ → F` (`B_GAUGED` of `u1_hexagon_kalg`).
* Witness: per-letter charge map `γ(L_{a,i})` (the
  `u1_hexagon_bps_iso` table) with the **E-sign flip**
  (`E_intrinsic = E⁻¹_U1Hex`, so `e_E` enters the charge as
  `−e·(1,0,1,0)`); inverse = lex-min cone decomposition
  (`_cone_monomial_for_charge`) with `e_E ↦ −e_E`.
* ``'plucker'`` / ``'rg-pentagon'`` — the frozen-Plücker stand-alone
  and the pentagon-landing RG presentation (E-flip / identity
  witnesses).
* ``'skein'`` — `SkeinU1HexagonKAlg`: the stated algebra of the
  bordered hexagon, unpinned at the side arcs onto exactly this
  algebra (the
  dictionary lands on the intrinsic, with two charged shorts on
  E-shifted canonicals, resolving a half-E-unit obstruction that
  rules out any bare-label dictionary).  Genuinely skein-side multiply
  (localized engine products + a consistency-certified per-label
  normalization); identity-label witness to the intrinsic.  The first
  skein realization.

Ungauged (`hexagon_object`):

* ``'ungauged'`` — `ungauge_u1a1aodd(1)`: the centralizer `Z(E)` with
  μ-fugacity coefficient ring; trace certified against the independent
  `A1AoddToEvenRGKAlgebra` μ-flavoured `[A₁,A₃]` index
  (`tests/test_ungauge_kalgebra.py`).
* ``'bps'`` — `BPSKAlgebra` on the **linear A₃ quiver** (no frozen
  node), μ-flavour from the rank-1 kernel of `B_UNGAUGED`
  (the `hexagon_bps_iso_v2` parallel-ungauging pattern).
* ``'bps-mut'`` / ``'bps-cyclic'`` — adjacent-chamber BPS
  presentations related to ``'bps'`` by **mutation** (the A₃ quiver is
  the classic example): the strict head necklace at γ₀, and the
  middle-node mutation whose node-basis quiver is the **3-cycle**
  (linear-A₃ ↦ cyclic-A₃).  Witnesses = the μ_g tropical label maps
  of `bps_chart_object.mutate_bpskalgebra`.
* Witness: same charge map restricted to mag-zero, projected to the
  three dynamical coordinates; inverse lifts by appending 0.

The abstract ungauged algebra is `A_𝖖[[A₁,A₃]]` = the finite zoo's
`a3`; welding this object to `kalgebra_object("a3")` (cone-frozen /
zform realizations) needs an ungauged↔a3 generator dictionary — a
follow-up, not wired here.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from kalgebra_object import KAlgebraObject
from laurent_poly import LaurentPoly
from bps_kalgebra import BPSKAlgebra
from u1a1aodd_kalg import U1A1AoddKAlg
from u1_hexagon_kalg import B_GAUGED
from u1_hexagon_kalg import charge as _charge_of_letter
from u1_hexagon_bps_iso import _cone_monomial_for_charge
from hexagon_bps_iso_v2 import B_UNGAUGED, NODE_CHARGES_UNGAUGED
from ungauge_kalgebra import UngaugedKAlgebra, ungauge_u1a1aodd


__all__ = ["u1hexagon_object", "hexagon_object", "hexagon_ungauging"]

_ONE = LaurentPoly.one()
_NODE_CHARGES_GAUGED = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]


def _label_charge_4d(lbl) -> tuple:
    """`U1A1AoddKAlg(1)` label `((a,i,p)…, e)` → gauged-quiver charge in
    Z⁴.  Letter charges from the `u1_hexagon_kalg` table; the E-power
    enters with the **minus** sign (`E_intrinsic = E⁻¹_U1Hex`)."""
    factors, e = lbl
    chg = [0, 0, 0, 0]
    for (a, i, p) in factors:
        c = _charge_of_letter((a, i))
        for k in range(4):
            chg[k] += c[k] * p
    chg[0] -= e
    chg[2] -= e
    return tuple(chg)


def _charge_to_label(gamma4) -> tuple:
    """Inverse: Z⁴ charge → intrinsic label, via the lex-min cone
    decomposition (U1Hex convention) with the E-sign flipped back."""
    d = _cone_monomial_for_charge(tuple(gamma4))
    if d is None:
        raise ValueError(
            f"hexagon_objects: charge {gamma4} admits no cone-monomial "
            f"decomposition")
    return (d[0], -d[1])


def u1hexagon_object() -> KAlgebraObject:
    """The abstract u(1)-gauged hexagon as a finalized `KAlgebraObject`."""
    obj = KAlgebraObject("A_q[u(1)-gauged [A1,A3]]")

    intrinsic = U1A1AoddKAlg(1)
    obj.add_realization("intrinsic", intrinsic,
                        {"multiply-fast", "trace-exact"})

    bps = BPSKAlgebra(pairing=B_GAUGED, node_charges=_NODE_CHARGES_GAUGED,
                      verify="off")
    obj.add_realization("bps", bps, {"chart", "trace-exact", "rg"})
    obj.add_iso(
        "intrinsic", "bps",
        KAlgebraIso(
            intrinsic, bps,
            forward_label_map=lambda l: Element({_label_charge_4d(l): _ONE}),
            inverse_label_map=lambda c: Element({_charge_to_label(c): _ONE}),
            name="u1hexagon[intrinsic→bps]"))

    # The stand-alone frozen-Plücker presentation.  Same labels as the
    # intrinsic up to the E-relabel (`E_plucker = E⁻¹_intrinsic`), so the
    # witness is the e-flip.  The ρ convention does not negate the wrap
    # μ-pickup; certified against the intrinsic through this dictionary.
    from u1_hexagon_kalg import U1HexagonKAlg
    plucker = U1HexagonKAlg()
    obj.add_realization("plucker", plucker, {"multiply-fast", "trace-exact"})
    obj.add_iso(
        "intrinsic", "plucker",
        KAlgebraIso(
            intrinsic, plucker,
            forward_label_map=lambda l: Element({(l[0], -l[1]): _ONE}),
            inverse_label_map=lambda l: Element({(l[0], -l[1]): _ONE}),
            name="u1hexagon[intrinsic→plucker]  (E ↦ E⁻¹)"))

    # The RG presentation landing on the pentagon: UV = this algebra,
    # IR = PentagonKAlg ⊗ QT₂ (drop the last O of the gauged quiver),
    # hard-coded RG generators + closed-form S_RG = E_q(A)·E_q(B).
    # Subclass of the Plücker class (same labels) ⇒ identity witness.
    from u1_hexagon_pentagon_kalg import U1HexagonPentagonRGKAlg
    rg_pent = U1HexagonPentagonRGKAlg()
    obj.add_realization("rg-pentagon", rg_pent,
                        {"rg", "multiply-fast", "trace-exact"})
    obj.add_iso(
        "plucker", "rg-pentagon",
        KAlgebraIso(
            plucker, rg_pent,
            forward_label_map=lambda l: Element({l: _ONE}),
            inverse_label_map=lambda l: Element({l: _ONE}),
            name="u1hexagon[plucker→rg-pentagon]"))

    # The stated-SKEIN realization (the stated algebra of the bordered
    # hexagon).  The bare stated algebra is the PINNED hexagon (the
    # A₃/[A₁,A₃] quiver over the enlarged endpoint-charge lattice), but
    # localizing at the side arcs and dressing the chords unpins it to
    # exactly THIS algebra: the dictionary certifies onto the intrinsic
    # on the nose (two charged shorts land
    # on E-shifted canonicals — the resolution of a half-E-unit
    # obstruction that rules out any bare-label dictionary).
    # `SkeinU1HexagonKAlg` packages the localized dressed-chord
    # subalgebra on the intrinsic labels with genuinely skein-side
    # multiply (localized engine products + a consistency-certified
    # per-label normalization); ρ/trace transported from the intrinsic
    # (the registered instance, so the witness endpoints match) pending
    # the intrinsic stated-side trace.  The first skein realization.
    from skein_u1hexagon_kalg import SkeinU1HexagonKAlg
    sk = SkeinU1HexagonKAlg(intrinsic=intrinsic)
    obj.add_realization("skein", sk, {"geometric"})
    obj.add_iso("skein", "intrinsic", sk.build_iso())
    # The unified SkeinKAlgebra instance (per-theory subclass):
    # derived cone multiply over the
    # localized engine's cross-products, on the intrinsic labels —
    # sharing the registered intrinsic AND engine instances, so the
    # witness is the identity on labels.
    from skein_kalgebra import U1HexagonSkeinKAlgebra
    skc = U1HexagonSkeinKAlgebra(intrinsic=intrinsic, engine=sk)
    obj.add_realization("skein-cone", skc, {"geometric", "cone"})
    obj.add_iso("skein-cone", "intrinsic", skc.build_iso(intrinsic))
    return obj


def hexagon_ungauging() -> UngaugedKAlgebra:
    """The ungauging map applied to the gauged hexagon: the centralizer
    `Z(E)` of the electric generator with `E`-power promoted to the
    flavour fugacity μ and the `(𝖖²;𝖖²)²_∞`-measure-corrected trace."""
    return ungauge_u1a1aodd(1)


class _A1D3Restricted:
    """The finite zoo's `[A₁,D₃]` with its `R(SU(2))` flavour restricted
    to the Cartan `Z[z^±]` (fundamental normalisation; the hexagon's
    μ = z²).

    A faithful coefficient-hom'd Z-form: `KAlgebra.base_change` pushes
    only the *trace/inner_product* coefficients through the hom — but
    a1d3's structure constants are themselves `RLaurent` over R(SU(2))
    (doublet-valued, e.g. `L0·L2 ∋ [1]·q⁻¹`), so the multiply
    coefficients must be pushed too.  This wrapper does both.  Labels,
    ρ, and the section decomposition are untouched."""

    def __init__(self) -> None:
        from finite_a1d3_kalg import FiniteA1D3KAlgebra
        from zplus_ring import su2_to_u1_hom, AbelianZPlusRing
        self._native = FiniteA1D3KAlgebra()
        self._Z = AbelianZPlusRing(rank=1)
        self._phi = su2_to_u1_hom(self._native.coefficient_ring(), self._Z)

    def coefficient_ring(self):
        return self._Z

    def identity(self):
        return self._native.identity()

    def rho(self, a):
        return self._native.rho(a)

    def rho_inverse(self, a):
        return self._native.rho_inverse(a)

    def _label_section_decompose(self, label):
        return label, self._Z.one()

    def _push_coeff(self, c):
        from zplus_ring import RLaurent
        if isinstance(c, RLaurent):
            return self._phi.apply_RLaurent(c)
        return c                      # plain LaurentPoly: SU(2)-trivial

    def multiply(self, a, b) -> Element:
        base = self._native.multiply(a, b)
        return Element({l: self._push_coeff(c)
                        for l, c in base.terms.items()})

    def trace(self, a, K: int = 20):
        return self._phi.apply_RPowerSeries(self._native.trace(a, K))

    def cone_data(self):
        return self._native.cone_data()


def _a1d3_letters(a1d3) -> list:
    cd = a1d3.cone_data()
    return [cd.from_cone_label(frozenset({g}), {g: 1})
            for g in sorted(cd.mult_gens())]


def _zmono(zring, t: int):
    """The unit z^t as an RLaurent coefficient."""
    from zplus_ring import RLaurent, RElement
    return RLaurent(zring, {0: RElement(zring, {(t,): 1})})


_KAPPA = (1, 0, 1)   # kernel of B_UNGAUGED: the flavour direction, worth z²


def _kernel_split(g):
    """γ ↦ (γ_perp, k) with γ = γ_perp + k·κ and γ_perp[2] = 0."""
    k = g[2]
    return (g[0] - k, g[1], 0), k


def _bps_z_normalize(elem, Z):
    """Kernel-normalize a bps-z Element: every label's κ-component is
    moved into the z-coefficient (κ ≡ z²; μ = z²).  Two representations
    of the same algebra element normalize identically, making literal
    Element comparison sound for this edge."""
    out = {}
    for g, c in elem.terms.items():
        gp, k = _kernel_split(g)
        cc = c * _zmono(Z, 2 * k) if k else c
        out[gp] = out[gp] + cc if gp in out else cc
    return Element({g: c for g, c in out.items()
                    if not (hasattr(c, "is_zero") and c.is_zero())})


def _a1d3_bps_dictionary(a1d3, bps):
    """Derive the witness `a1d3-restricted ↔ bps-z` by ρ-propagation.

    Image shape: `letter ↦ z^t · L_γ` with γ a mag-zero 3D charge (long
    chord or diameter pair) and t the **odd** z-prefactor carrying the
    half μ-unit (the period-3 vs period-6-with-μ resolution; μ = z²).
    Seeds: letter 0 over the 3 long-chord charges × odd t; letter 1 over
    the 3 diameter-pair charges × t from the ρ³-closure; the rest of
    each ρ-orbit is propagated by `img(ρx) = ρ_bps(img(x))` (ρ inverts
    the z-prefactor).  Candidates are pre-filtered by exact ρ³-closure,
    then certified by the full `KAlgebraIso` battery; the first
    certified dictionary wins."""
    from u1a1aodd_kalg import U1A1AoddKAlg
    A = U1A1AoddKAlg(1)   # charge bookkeeping only (same letters)

    def chg3(letter_label):
        return _label_charge_4d(letter_label)[:3]

    chords = [chg3((((2, i, 1),), 0)) for i in range(3)]
    diams = [chg3((((1, i, 1), (1, (i + 3) % 6, 1)), 0)) for i in range(3)]

    letters = _a1d3_letters(a1d3)     # L0..L5
    orbit1 = [0, 2, 5]                # rho-orbit of L0 (period 3)
    orbit2 = [1, 3, 4]                # rho-orbit of L1
    # order each orbit by rho-propagation from its seed
    def rho_chain(seed_idx):
        out, lbl = [], letters[seed_idx]
        for _ in range(3):
            out.append(letters.index(lbl))
            lbl = a1d3.rho(lbl)
        return out

    o1, o2 = rho_chain(0), rho_chain(1)

    def propagate3(seed_gamma, seed_t):
        """[(γ, t)] along three ρ steps; None unless the chain closes
        as algebra elements — ρ³ = id modulo kernel shifts absorbed
        into the z-prefactor (κ ≡ z²)."""
        chain = [(seed_gamma, seed_t)]
        g, t = seed_gamma, seed_t
        for _ in range(3):
            g, t = bps.rho(g), -t
            chain.append((g, t))
        (g0, t0), (g3, t3) = chain[0], chain[3]
        p0, k0 = _kernel_split(g0)
        p3, k3 = _kernel_split(g3)
        if p0 == p3 and t0 + 2 * k0 == t3 + 2 * k3:
            return chain[:3]
        return None

    def closures(family):
        return [c for g in family for tt in range(-4, 5)
                if (c := propagate3(g, tt))]

    # Orbit→family pairing is part of the search: empirically orbit1
    # {L0,L2,L5} lands on the diameter pairs (even prefactor, e.g.
    # Tr L0 = z⁻²·Tr D0) and orbit2 on the chords (odd prefactor,
    # Tr L1 = z⁻¹·Tr C0), but both pairings are tried and the battery
    # decides.
    pairings = [(closures(diams), closures(chords)),
                (closures(chords), closures(diams))]

    one = LaurentPoly.one()
    Z = a1d3.coefficient_ring()

    def _norm(elem):
        return _bps_z_normalize(elem, Z)

    def _z_conj_coeff(c):
        """Flavour conjugation z ↦ z⁻¹ on an RLaurent coefficient (the
        contract's ρ(r·x) = r̄·ρ(x); plain LaurentPoly is flavour-blind)."""
        from zplus_ring import RLaurent, RElement
        if not isinstance(c, RLaurent):
            return c
        return RLaurent(Z, {q: RElement(Z, {(-b[0],): m
                                            for b, m in r.terms.items()})
                            for q, r in c.coeffs.items()})

    def _rho_el(elem):
        """Contract-correct flavoured ρ on a bps-z Element: permute the
        labels AND conjugate the ring coefficients (the generic
        `rho_element` maps labels only — insufficient once flavour
        rides in coefficients)."""
        return Element({bps.rho(g): _z_conj_coeff(c)
                        for g, c in elem.terms.items()})

    def _certify(iso, src_labels):
        """Custom battery with kernel-normalized comparison on the bps
        side (the stock literal comparison cannot see κ ≡ z²)."""
        src = [Element({l: one}) for l in src_labels]
        # round-trip
        for e in src:
            back = iso.invert().map(iso.map(e))
            if back != e:
                return False
        # multiplicativity (normalized)
        for x in src_labels:
            for y in src_labels:
                lhs = _norm(iso.map(a1d3.multiply(x, y)))
                rhs = _norm(bps.multiply_elements(
                    iso.map(Element({x: one})), iso.map(Element({y: one}))))
                if lhs != rhs:
                    return False
        # ρ-equivariance (normalized; flavour-conjugating ρ on bps-z)
        for x in src_labels:
            lhs = _norm(iso.map(Element({a1d3.rho(x): one})))
            rhs = _norm(_rho_el(iso.map(Element({x: one}))))
            if lhs != rhs:
                return False
        # trace equivariance
        for x in src_labels:
            ts = a1d3.trace(x, 6)
            tt = bps.trace_element(iso.map(Element({x: one})), 6)
            if ts != tt:
                return False
        return True

    for candidates1, candidates2 in pairings:
      for c1 in candidates1:
        for c2 in candidates2:
            fwd_table = {}
            for idx, (g, t) in zip(o1, c1):
                fwd_table[letters[idx]] = (g, t)
            for idx, (g, t) in zip(o2, c2):
                fwd_table[letters[idx]] = (g, t)
            # inverse keyed by kernel-normalized charge:
            #   z^t·L_γ = z^{t+2k}·L_{γ_perp}  ⇒  L_γ ↦ z^{2k-t-2k+t}=…
            inv_table = {}
            for lbl, (g, t) in fwd_table.items():
                gp, k = _kernel_split(g)
                inv_table[gp] = (lbl, t + 2 * k)

            # letter index i -> (γ_i, t_i);  fwd_table keys are ((i,1),)
            letter_data = {lbl[0][0]: gt for lbl, gt in fwd_table.items()}

            def fwd(lbl, _L=letter_data):
                # cone words map additively: charge Σ p·γ_i, prefactor
                # z^{Σ p·t_i}  (q-commuting cone families)
                if lbl == a1d3.identity():
                    return Element({(0, 0, 0): one})
                g_tot, t_tot = (0, 0, 0), 0
                for (i, p) in lbl:
                    if i not in _L:
                        raise ValueError(f"a1d3 dictionary: letter {i}")
                    g, t = _L[i]
                    g_tot = tuple(x + p * y for x, y in zip(g_tot, g))
                    t_tot += p * t
                return Element({g_tot: _zmono(Z, t_tot)})

            def inv(g, _T=inv_table):
                if tuple(g) == (0, 0, 0):
                    return Element({a1d3.identity(): one})
                gp, k = _kernel_split(tuple(g))
                if gp in _T:
                    lbl, t_eff = _T[gp]
                    return Element({lbl: _zmono(Z, 2 * k - t_eff)})
                raise ValueError(f"a1d3 dictionary: charge {g} not a "
                                 f"generator image")

            iso = KAlgebraIso(a1d3, bps,
                              forward_label_map=fwd,
                              inverse_label_map=inv,
                              name="hexagon[a1d3→bps]  (μ = z²)")
            try:
                ok = _certify(iso, [a1d3.identity()] + letters)
            except Exception:
                continue
            if ok:
                iso._kernel_normalized_certified = True
                return iso
    raise ValueError("a1d3 dictionary: no certified assignment found")


def hexagon_object() -> KAlgebraObject:
    """The abstract hexagon (flavoured `[A₁,A₃]`) as a finalized
    `KAlgebraObject`, realized through the ungauging map.

    Coefficient ring: `Z[z^±]` in the **fundamental (SU(2)-Cartan)
    normalisation** — the ungauging fugacity is μ = z², so the natively
    μ-flavoured realizations are base-changed along
    `restriction_hom([[2]])`.  This index-2 refinement is what lets the
    `[A₁,D₃]` presentation (whose flavour is the *enhanced* SU(2)) join
    the object: its generators match `z^{±1}·(chord / diameter pair)`,
    with the odd prefactor carrying the half μ-unit."""
    from zplus_ring import AbelianZPlusRing, restriction_hom
    obj = KAlgebraObject("A_q[[A1,A3]] (flavoured hexagon, mu=z^2)")
    Z = AbelianZPlusRing(rank=1)

    ung = hexagon_ungauging().base_change(
        restriction_hom(AbelianZPlusRing(rank=1), Z, [[2]]))
    obj.add_realization("ungauged", ung,
                        {"trace-exact", "flavoured", "ungauging"})

    bps_nat = BPSKAlgebra(pairing=B_UNGAUGED,
                          node_charges=NODE_CHARGES_UNGAUGED,
                          verify="off")
    bps = bps_nat.base_change(
        restriction_hom(AbelianZPlusRing(rank=1), Z, [[2]]))
    obj.add_realization("bps", bps, {"chart", "trace-exact", "flavoured"})

    def fwd(lbl):
        return Element({_label_charge_4d(lbl)[:3]: _ONE})

    def inv(c3):
        return Element({_charge_to_label(tuple(c3) + (0,)): _ONE})

    obj.add_iso(
        "ungauged", "bps",
        KAlgebraIso(ung, bps,
                    forward_label_map=fwd, inverse_label_map=inv,
                    name="hexagon[ungauged→bps]"))

    a1d3 = _A1D3Restricted()
    obj.add_realization("a1d3", a1d3,
                        {"trace-exact", "flavoured", "su2-enhanced"})
    obj.add_iso("a1d3", "bps", _a1d3_bps_dictionary(a1d3, bps))

    # Mutation charts: the A₃ quiver is
    # the classic mutation example, so the object carries two
    # adjacent-chamber BPS presentations with the μ_g tropical label
    # maps as witnesses (`bps_chart_object.mutate_bpskalgebra` — same
    # algebra, different chamber):
    #   * 'bps-mut'    — strict head necklace at γ₀ (spec cooperates
    #     at budget 0);
    #   * 'bps-cyclic' — mutation at the MIDDLE node (local-move budget
    #     for the pentagon expansion; spec length 3 → 4): the
    #     node-basis quiver becomes the 3-CYCLE — the textbook
    #     linear-A₃ ↦ cyclic-A₃ mutation.
    # Charts are built on the native μ-ring and base-changed to the
    # object's z-ring; the μ_g label maps are ring-independent, so the
    # witnesses rewrap verbatim onto the base-changed endpoints.
    from bps_chart_object import mutate_bpskalgebra
    for key, (mut_nat, w) in (
        ("bps-mut", mutate_bpskalgebra(bps_nat, 0)),
        ("bps-cyclic", mutate_bpskalgebra(bps_nat, 1, max_local_moves=6)),
    ):
        mut = mut_nat.base_change(
            restriction_hom(AbelianZPlusRing(rank=1), Z, [[2]]))
        obj.add_realization(key, mut,
                            {"chart", "trace-exact", "flavoured"})
        obj.add_iso("bps", key,
                    KAlgebraIso(bps, mut, w._forward, w._inverse,
                                name=f"hexagon[bps→{key}]  ({w.name})"))
    return obj
