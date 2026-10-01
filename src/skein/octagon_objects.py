"""`KAlgebraObject`s for the **octagon pair** (k = 2 of the `U1A1Aodd`
family; the (2k+4)-gon — see the design notes and the hexagon pair in
`hexagon_objects.py`, whose structure this mirrors one k up):

* `u1octagon_object()` — the **U(1)-gauged octagon** = u(1)-gauged
  `[A₁, A₅]`: gauged-A₅ quiver `O₁→O₂→O₃→O₄→O₅→F` (lattice dim 6).
* `octagon_object()`   — the **octagon** = μ-flavoured `[A₁, A₅]`,
  realized from the gauged algebra by the general U(1) **ungauger**
  (`ungauge_kalgebra.UngaugedKAlgebra`), exactly as the hexagon.

Presentations and witnesses
---------------------------

Gauged (`u1octagon_object`):

* ``'intrinsic'`` — `U1A1AoddKAlg(2)`: the parametric closed-form cone
  presentation (22 mult-gens: three chord families of period 8 + E^±).
* ``'bps'`` — `BPSKAlgebra` on the gauged-A₅ quiver
  (`u1a1aodd_kalg.gauged_quiver_bps(2)`, the chart in whose coordinates
  the intrinsic's chord charges are written).
* ``'rg-heptagon'`` — the heptagon-IR RG presentation through the
  MODERN directional machinery: `DirectionalSingleNodeRG` dropping one
  end O of the gauged chain; the IR auxiliary is the A₄ chamber over
  the full Z⁶ lattice (= heptagon ⊗ the dropped + frozen directions).
  Lives on the gauged BPS charges verbatim → identity witness to
  ``'bps'``.  (The hand-coded `U1OctaHeptaRGKAlg` — Plücker labels,
  hard-coded RG generators — and the stand-alone `U1OctagonKAlg` it sat on
  were retired to the source repository's archive on 2026-09-23; the same flow with `RG` solved
  from `S_RG` is `u1aodd_to_even_qt_rgkalgebra.
  U1A1AoddToEvenQTRGKAlgebra(2)`.)
* Witness intrinsic↔bps: per-letter charge map
  `label = ((a,i,p)…, e) ↦ Σ p·chord_charge(a,i) + s·e·E` keyed on the
  intrinsic's OWN charge table (`cone_data()._chg` / `_MU`, the closed
  form), with the E-sign `s` certified at construction by a round-trip +
  ρ-equivariance + family-2 product battery; inverse = `_decompose`
  (charge → e_E + q-commuting chord factorisation, below).

Ungauged (`octagon_object`):

* ``'ungauged'`` — the named `octagon_kalg.OctagonKAlg()`
  (`UngaugedPolygonKAlg(2)`): the centralizer `Z(E)` of the intrinsic
  `U1A1AoddKAlg(2)` with the μ-fugacity coefficient ring and the
  measure-restored trace, in the μ-orientation this object needs (see
  `octagon_object`).  `octagon_ungauging()` returns the ungauging map
  itself, `ungauge_u1a1aodd(2)`.
* ``'bps'`` — `BPSKAlgebra` on the **linear A₅ quiver** (dim 5, no
  frozen node); μ-flavour from the rank-1 pairing kernel
  `(1,0,1,0,1)`.
* Witness: the same charge map restricted to the centralizer,
  projected to the five dynamical coordinates; inverse pads with 0
  and reuses `_decompose`.

Unlike the hexagon there is no su2-enhanced D-series partner at k = 2
(A₃ ≅ D₃ is special), so the octagon object stays in its native
`Z[μ^±]` ring with two realizations; welding to the finite zoo's
``a5`` / ``octagon`` entries (the same abstract algebra, cone-frozen,
u1-flavoured) is the same tracked follow-up as hexagon ↔ a3.
"""
from __future__ import annotations

import sys, os
from itertools import product as iproduct
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from kalgebra_object import KAlgebraObject
from laurent_poly import LaurentPoly
from bps_kalgebra import BPSKAlgebra
from u1a1aodd_kalg import U1A1AoddKAlg, gauged_quiver_bps
from ungauge_kalgebra import ungauge_u1a1aodd


__all__ = ["u1octagon_object", "octagon_object", "octagon_ungauging"]

_ONE = LaurentPoly.one()

B_UNGAUGED_K2 = [
    [0, 1, 0, 0, 0],
    [-1, 0, 1, 0, 0],
    [0, -1, 0, 1, 0],
    [0, 0, -1, 0, 1],
    [0, 0, 0, -1, 0],
]
NODE_CHARGES_UNGAUGED_K2 = [
    (1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0),
    (0, 0, 0, 1, 0), (0, 0, 0, 0, 1),
]


# ---------------------------------------------------------------------------
# charge -> label: a q-commuting chord factorisation (moved here verbatim from
# the retired `u1a1aodd_general._decompose`, 2026-09-23)
# ---------------------------------------------------------------------------

def _qcommute(bps, c1, c2):
    return len(bps.multiply(c1, c2).terms) == 1


def _all_qcommute(bps, gens, facs):
    for i in range(len(facs)):
        for j in range(i + 1, len(facs)):
            if facs[i] != facs[j] and not _qcommute(
                    bps, gens[facs[i]], gens[facs[j]]):
                return False
    return True


def _is_E_multiple(diff, E_charge):
    """`c` with `diff = c·E`, or None."""
    n = len(diff)
    nonzero_E_positions = [i for i in range(n) if E_charge[i] != 0]
    zero_E_positions = [i for i in range(n) if E_charge[i] == 0]
    for i in zero_E_positions:
        if diff[i] != 0:
            return None
    if not nonzero_E_positions:
        return None
    c = diff[nonzero_E_positions[0]]
    for i in nonzero_E_positions:
        if diff[i] != c:
            return None
    return c


def _decompose(gamma, chord_charges, primitives, E_charge, bps, max_factors=4):
    """Find `γ = Σ charge(L_a(i)) + e_E·E` with the factors pairwise
    q-commuting in `bps`.  Greedy: tries `n_facs = 0, 1, …, max_factors` and
    returns the first `(e_E, factors)` that reproduces `γ` mod `E`."""
    n = len(gamma)
    for n_facs in range(0, max_factors + 1):
        for facs in iproduct(primitives, repeat=n_facs):
            if list(facs) != sorted(facs):
                continue
            if not _all_qcommute(bps, chord_charges, facs):
                continue
            fc = [0] * n
            for f in facs:
                fci = chord_charges[f]
                for j in range(n):
                    fc[j] += fci[j]
            diff = tuple(gamma[j] - fc[j] for j in range(n))
            e_E = _is_E_multiple(diff, E_charge)
            if e_E is not None:
                return (e_E, list(facs))
    return None


def _charge_maps(intrinsic, bps, e_sign: int):
    """(forward, inverse) label↔charge maps for the given E-sign.

    Keyed on the INTRINSIC class's own letter→charge table
    (`cone_data()._chg`, the closed form, with `E ≡ _MU`), NOT the
    independently-seeded table of the retired `u1a1aodd_general`
    bootstrap: the two agree per ρ-orbit only up to a rotation of the
    index `i`.  (Learned
    the hard way: round-trips and ρ-equivariance are blind to a uniform
    per-family rotation — ρ acts as `i ↦ i+1` on both indexings — but
    multiplication and trace flavour are not.)"""
    cd = intrinsic.cone_data()
    chg = {ai: tuple(c) for ai, c in cd._chg.items()}
    E = tuple(cd._MU)
    primitives = sorted(chg.keys())
    n = len(E)

    def fwd_charge(lbl):
        factors, e = lbl
        v = [0] * n
        for (a, i, p) in factors:
            c = chg[(a, i)]
            for j in range(n):
                v[j] += c[j] * p
        for j in range(n):
            v[j] += e_sign * e * E[j]
        return tuple(v)

    def inv_label(gamma):
        dec = _decompose(tuple(gamma), chg, primitives, E, bps)
        if dec is None:
            raise ValueError(
                f"octagon_objects: charge {gamma} admits no chord "
                f"decomposition")
        e_E, facs = dec
        powers: dict = {}
        for f in facs:
            powers[f] = powers.get(f, 0) + 1
        word = tuple(sorted((a, i, p) for (a, i), p in powers.items()))
        return (word, e_sign * e_E)

    return fwd_charge, inv_label


def _pick_e_sign(intrinsic, bps):
    """Decide the E-orientation of the intrinsic ↔ bps dictionary by
    the battery: the sign under which the labels round-trip, the charge
    map is ρ-equivariant, AND the map is multiplicative on the
    centralizer-chord pairs (whose products pick up `E^{±1}` terms —
    the check that pins the orientation and would catch any residual
    letter misalignment).  (At k = 1 the analogous sign was baked into
    `hexagon_objects._label_charge_4d` by hand; here it is certified
    at construction.)"""
    cd = intrinsic.cone_data()
    letters = sorted(cd._chg.keys())
    fam2 = [(a, i) for (a, i) in letters if a == 2]
    for s in (+1, -1):
        fwd, inv = _charge_maps(intrinsic, bps, s)
        try:
            ok = True
            for lbl in ([((), 1), ((), -1)]
                        + [(((a, i, 1),), e) for (a, i) in letters[:3]
                           for e in (0, 1)]):
                if inv(fwd(lbl)) != lbl:
                    ok = False
                    break
                # ρ-equivariance of the charge map
                img_rho = fwd(intrinsic.rho(lbl))
                rho_img = tuple(bps.rho(fwd(lbl)))
                if img_rho != rho_img:
                    ok = False
                    break
            # multiplicativity on all family-2 pairs (label-level: both
            # sides have trivial flavour, so plain dict comparison)
            if ok:
                for x in fam2:
                    for y in fam2:
                        lx, ly = (((x + (1,)),), 0), (((y + (1,)),), 0)
                        lhs = {fwd(l): c for l, c in
                               intrinsic.multiply(lx, ly).terms.items()}
                        rhs = dict(bps.multiply(fwd(lx), fwd(ly)).terms)
                        if lhs != rhs:
                            ok = False
                            break
                    if not ok:
                        break
            if ok:
                return s
        except Exception:
            continue
    raise ValueError("octagon_objects: neither E-orientation certifies")


def u1octagon_object() -> KAlgebraObject:
    """The abstract u(1)-gauged octagon as a `KAlgebraObject`."""
    obj = KAlgebraObject("A_q[u(1)-gauged [A1,A5]]")

    intrinsic = U1A1AoddKAlg(2)
    obj.add_realization("intrinsic", intrinsic,
                        {"multiply-fast", "trace-exact"})

    bps = gauged_quiver_bps(2)
    obj.add_realization("bps", bps, {"chart", "trace-exact", "rg"})

    s = _pick_e_sign(intrinsic, bps)
    fwd_charge, inv_label = _charge_maps(intrinsic, bps, s)
    obj.add_iso(
        "intrinsic", "bps",
        KAlgebraIso(
            intrinsic, bps,
            forward_label_map=lambda l: Element({fwd_charge(l): _ONE}),
            inverse_label_map=lambda c: Element({inv_label(c): _ONE}),
            name=f"u1octagon[intrinsic→bps]  (E-sign {s:+d})"))

    # The heptagon-IR RG presentation through the modern directional
    # machinery: drop one end O of the gauged chain; the IR auxiliary
    # is the A₄ chamber over the full Z⁶ lattice = heptagon ⊗ QT₂
    # (dropped + frozen directions).  Same chamber as 'bps' →
    # identity witness.
    from directional_subquiver_rg import DirectionalSingleNodeRG
    rg_hept = DirectionalSingleNodeRG(
        [list(r) for r in bps.lattice.pairing],
        list(bps.node_charges), list(bps.spec),
        gamma_drop=0, rg_window=6)
    obj.add_realization("rg-heptagon", rg_hept, {"rg", "trace-exact"})
    obj.add_iso(
        "rg-heptagon", "bps",
        KAlgebraIso(rg_hept, bps,
                    forward_label_map=lambda l: Element({l: _ONE}),
                    inverse_label_map=lambda l: Element({l: _ONE}),
                    name="u1octagon[rg-heptagon→bps]"))
    return obj


def octagon_ungauging():
    """The ungauging map applied to the gauged octagon: `Z(E)` with the
    E-power promoted to μ and the measure-corrected trace."""
    return ungauge_u1a1aodd(2)


def octagon_object() -> KAlgebraObject:
    """The abstract octagon (flavoured `[A₁,A₅]`) as a
    `KAlgebraObject`, realized through the ungauging map.

    μ-orientation: the ungauging map `ungauge_u1a1aodd(2)` grades
    `Tr((F, e)) = μ^{−e}·Tr((F, 0))`, while the A₅ bps kernel convention
    shifts with `μ^{+e}`, and the base chord traces are flavour-conjugate
    too (at k = 1 the two conventions happened to align).  The named
    `OctagonKAlg` is that ungauging read with the charge conjugation
    `μ ↦ μ⁻¹` (a ring automorphism on the trace coefficients; the
    multiply carries μ in labels), i.e. `Tr((F, e)) = μ^{e}·Tr((F, 0))`,
    which is also the section convention of its lift — so it is the
    `'ungauged'` realization as it stands.  (Until 2026-09-23 this
    function applied the conjugation here, to `octagon_ungauging()`; the
    named class then wrapped the retired stand-alone `U1OctagonKAlg` in
    other letters.)  The battery is the arbiter."""
    from octagon_kalg import OctagonKAlg
    obj = KAlgebraObject("A_q[[A1,A5]] (flavoured octagon)")

    ung = OctagonKAlg()
    obj.add_realization("ungauged", ung,
                        {"trace-exact", "flavoured", "ungauging"})

    bps = BPSKAlgebra(pairing=B_UNGAUGED_K2,
                      node_charges=NODE_CHARGES_UNGAUGED_K2,
                      verify="off")
    obj.add_realization("bps", bps, {"chart", "trace-exact", "flavoured"})

    intrinsic = U1A1AoddKAlg(2)
    gauged_bps = gauged_quiver_bps(2)
    s = _pick_e_sign(intrinsic, gauged_bps)
    fwd_charge, inv_label = _charge_maps(intrinsic, gauged_bps, s)

    def fwd(lbl):
        return Element({fwd_charge(lbl)[:5]: _ONE})

    def inv(c5):
        return Element({inv_label(tuple(c5) + (0,)): _ONE})

    obj.add_iso(
        "ungauged", "bps",
        KAlgebraIso(ung, bps,
                    forward_label_map=fwd, inverse_label_map=inv,
                    name="octagon[ungauged→bps]"))
    return obj
