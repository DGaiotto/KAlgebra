"""`su2_family_atlas` — the SU(2)-family `BPSAtlas` builders (the design record catalogue).

A uniform constructor family for the rank-1 SU(2) gauge theories, each presented
as a canonical-surface `BPSKAlgebra` from the **known BPS quiver + (matter)(pure
gauge) spec recipe**, ready to seed a `BPSAtlas`:

    pure SU(2)          Kronecker-2 chart  (= pure_ade.PureADE([("A", 1)]))
    SU(2) + N_f=1..4    pure_ade.SUN_Nf(2, N_f)   (matter dyon factors · pure-SU(2) spec)
    SU(2) - SU(2)       pure_ade.SUN_bifund(2, 2) (one bifundamental, no extra flavours)

`pure_ade.SUN_Nf` / `SUN_bifund` assemble the (legacy `CoulombAlgebra`) BPS data;
we transcribe their `.B/.nodes/.spec/.cone_witness` into the canonical
`BPSKAlgebra` exactly as the hand-written `bps_su2_nf1/2/3` charts do (and
`SUN_Nf(2, 1)` *is* `bps_su2_nf1` term-for-term).  The spec is recipe-valid by
construction (`SUN_*.verify()` runs `verify_spectrum_generator`), so the
`BPSKAlgebra` is built with `verify="off"`.

Each entry also records:

  * `matter_indices` — the node positions of the matter (fundamental / bifund)
    dyons.  Dropping these is the **node-drop RG flow** to the pure-gauge IR
    (`rg_flow.SubquiverRG`); the identity-on-labels `KAlgebraIso` to the BPS
    chart is then *guaranteed* — see `su2_family_rg_nodedrop`.
  * `cone_factory` — a zero-arg builder of the `ConeKAlgebra` presentation when
    one exists (pure SU(2), N_f=1/2/3); `None` for the conformal N_f=4 and the
    SU(2)-SU(2) quiver (no closed-form cone presentation).

Public API:
  * `su2_family_charts()      -> dict[str, BPSKAlgebra]`
  * `su2_family_entry(name)   -> SU2FamilyEntry`  (chart + metadata)
  * `su2_family_names()       -> list[str]`
  * `su2_family_rg_nodedrop(name) -> SubquiverRG`  (the node-drop RG presentation)
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pure_ade as _pa
from bps_kalgebra import BPSKAlgebra


__all__ = [
    "SU2FamilyEntry",
    "su2_family_names",
    "su2_family_entry",
    "su2_family_charts",
    "su2_family_rg_nodedrop",
    "su2_family_rg_iso",
    "su2_family_cone",
    "su2_family_cone_rays",
    "su2_family_cone_iso",
    "su2_family_cone_bps_iso",
    "su2_family_object",
    "su2_family_objects",
    "su2_family_cone_cartan_hom",
    "su2_family_cone_reduced",
    "su2_family_cone_index_matches_bps",
]


# ---------------------------------------------------------------------------
# transcription: pure_ade BPS data  ->  canonical BPSKAlgebra
# ---------------------------------------------------------------------------
def _bps_from_pure_ade(builder) -> BPSKAlgebra:
    """Transcribe a `pure_ade.SUN_*` BPS-data object into a canonical
    `BPSKAlgebra` (same `.B/.nodes/.spec/.cone_witness`; the spec is recipe-valid,
    so `verify="off"`)."""
    if not builder.verify():
        raise RuntimeError(
            f"{builder!r}: assembled spec is not a valid negating sequence "
            f"(pure_ade recipe bug — should not happen)")
    return BPSKAlgebra(
        pairing=[list(r) for r in builder.B],
        node_charges=[tuple(g) for g in builder.nodes],
        spec=[tuple(g) for g in builder.spec],
        cone_witness=tuple(builder.cone_witness),
        verify="off",
    )


# ---------------------------------------------------------------------------
# the family entry
# ---------------------------------------------------------------------------
@dataclass
class SU2FamilyEntry:
    """One SU(2)-family theory: its canonical BPS chart + atlas metadata."""
    name: str
    chart: BPSKAlgebra
    matter_indices: tuple[int, ...]      # node positions to drop for the IR flow
    pure_ir: str                         # description of the node-drop IR
    cone_factory: Optional[Callable[[], object]] = None
    note: str = ""

    @property
    def rank(self) -> int:
        return len(self.chart.lattice.pairing)

    @property
    def has_cone(self) -> bool:
        return self.cone_factory is not None


# ---------------------------------------------------------------------------
# builders (lazy; built once and cached)
# ---------------------------------------------------------------------------
def _pure_su2_chart() -> BPSKAlgebra:
    # Kronecker-2: ⟨(1,0),(-1,2)⟩ = 2.  Identical to pure_ade.PureADE([("A",1)]).
    return BPSKAlgebra(pairing=[[0, 1], [-1, 0]],
                       node_charges=[(1, 0), (-1, 2)])


def _cone_pure_su2():
    from pure_su2_h_cone_data import PureSU2KAlg
    return PureSU2KAlg()


def _cone_nf1():
    from su2_nf1_kalgebra import SU2Nf1KAlgebra
    return SU2Nf1KAlgebra()


def _cone_nf2():
    from su2_nf2_cone_standalone import SU2Nf2ConeKAlgebra
    return SU2Nf2ConeKAlgebra()


def _cone_nf3():
    from su2_nf3_cone_standalone import SU2Nf3ConeKAlgebra
    return SU2Nf3ConeKAlgebra()


_FACTORIES: dict[str, Callable[[], SU2FamilyEntry]] = {}


def _register(name):
    def deco(fn):
        _FACTORIES[name] = fn
        return fn
    return deco


@_register("pure_SU2")
def _e_pure():
    return SU2FamilyEntry(
        name="pure_SU2", chart=_pure_su2_chart(),
        matter_indices=(), pure_ir="(no matter — already pure SU(2))",
        cone_factory=_cone_pure_su2,
        note="Kronecker-2 chart; rotates with period 4.")


def _sun_nf_entry(nf: int, cone_factory):
    s = _pa.SUN_Nf(2, nf)
    n_pure = len(s.pure.nodes)                       # 2 for A1
    matter = tuple(range(n_pure, len(s.nodes)))      # the N_f matter dyon nodes
    return SU2FamilyEntry(
        name=f"SU2_Nf{nf}", chart=_bps_from_pure_ade(s),
        matter_indices=matter, pure_ir="pure SU(2)  (drop the N_f matter dyons)",
        cone_factory=cone_factory,
        note=f"pure_ade.SUN_Nf(2,{nf}); spec = {nf} matter blocks · pure-SU(2) spec.")


@_register("SU2_Nf1")
def _e_nf1():
    return _sun_nf_entry(1, _cone_nf1)


@_register("SU2_Nf2")
def _e_nf2():
    return _sun_nf_entry(2, _cone_nf2)


@_register("SU2_Nf3")
def _e_nf3():
    return _sun_nf_entry(3, _cone_nf3)


@_register("SU2_Nf4")
def _e_nf4():
    # Conformal SU(2)+N_f=4 — no closed-form cone presentation (skein-only).
    return _sun_nf_entry(4, None)


@_register("SU2_SU2")
def _e_su2su2():
    s = _pa.SUN_bifund(2, 2)
    n1 = len(s.pure1.nodes)
    n2 = len(s.pure2.nodes)
    matter = (n1 + n2,)                              # the single bifundamental node
    return SU2FamilyEntry(
        name="SU2_SU2", chart=_bps_from_pure_ade(s),
        matter_indices=matter,
        pure_ir="pure SU(2) x pure SU(2)  (drop the bifundamental)",
        cone_factory=None,
        note="pure_ade.SUN_bifund(2,2); spec = bifund block · two pure-SU(2) specs.")


# ---------------------------------------------------------------------------
# public surface
# ---------------------------------------------------------------------------
_CACHE: dict[str, SU2FamilyEntry] = {}


def su2_family_names() -> list[str]:
    """The family in canonical order."""
    return ["pure_SU2", "SU2_Nf1", "SU2_Nf2", "SU2_Nf3", "SU2_Nf4", "SU2_SU2"]


def su2_family_entry(name: str) -> SU2FamilyEntry:
    """The `SU2FamilyEntry` for `name` (built once, cached)."""
    if name not in _FACTORIES:
        raise KeyError(f"unknown SU(2)-family theory {name!r}; "
                       f"choose from {su2_family_names()}")
    if name not in _CACHE:
        _CACHE[name] = _FACTORIES[name]()
    return _CACHE[name]


def su2_family_charts() -> dict[str, BPSKAlgebra]:
    """`{name: BPSKAlgebra}` for the whole family, in canonical order."""
    return {nm: su2_family_entry(nm).chart for nm in su2_family_names()}


def su2_family_rg_nodedrop(name: str):
    """The **node-drop RG presentation** of theory `name`: the directional
    subquiver RG flow that drops the matter nodes, flowing to the pure-gauge IR
    (`rg_flow.SubquiverRG`).  Its identity-on-labels `KAlgebraIso` to the BPS
    chart is the *guaranteed* RG iso.  Raises for pure SU(2)
    (no matter to drop)."""
    from rg_flow import SubquiverRG
    e = su2_family_entry(name)
    if not e.matter_indices:
        raise ValueError(f"{name}: no matter nodes to drop (already pure gauge)")
    return SubquiverRG(e.chart, list(e.matter_indices))


def su2_family_rg_iso(name: str):
    """The **BPS ↔ node-drop-RG** `KAlgebraIso` (`KAlgebraIso.identity_on_labels`,
    `rg_presentation → bps_chart`).  By the node-drop construction this is
    *guaranteed* to be an iso once the auxiliaries match, so it needs only the
    light structural battery (unit / round-trip / multiplicative / ρ) — not the
    expensive trace stage.  Raises for pure SU(2)."""
    from kalgebra_iso import KAlgebraIso
    e = su2_family_entry(name)
    rg = su2_family_rg_nodedrop(name)
    return KAlgebraIso.identity_on_labels(
        rg, e.chart, name=f"{name}: RG(node-drop {e.pure_ir}) ≅ BPS")


# ---------------------------------------------------------------------------
# cone presentation + ray-focused cone iso
# ---------------------------------------------------------------------------
# Ray (mult-gen) labels for the H-tower cone presentations.  The cone is
# generated by the H-tower {H_n} and the fundamental Wilson line w_1; "focus on
# rays" = certify the cone↔BPS iso on these generators.
_WILSON = ("W", 1)


def su2_family_cone(name: str):
    """The `ConeKAlgebra` presentation of theory `name`, or `None` when none
    exists (the conformal N_f=4 is skein-only; the SU(2)-SU(2) quiver has no
    closed-form cone presentation)."""
    e = su2_family_entry(name)
    return e.cone_factory() if e.cone_factory is not None else None


def su2_family_cone_rays(name: str) -> list:
    """The **flavour-neutral (magnetic / gauge) ray** generators of `name`'s cone
    presentation — the H-tower rays `H_0, H_1` — in that cone's native label form.
    These are where the label↔label cone↔BPS iso is q-exact: the gauge direction
    carries no flavour, so the BPS "flavour-in-the-label" vs cone
    "flavour-in-the-coefficient-ring" convention never bites.  `pure_SU2`, being unflavoured, additionally
    includes the Wilson ray `w_1` (also clean there).

    The Wilson / flavour-charged ray direction is **not** here: its
    multiplicativity is the R-form correspondence (cone ↔ bps-rform), which the
    object-layer (`su2_nf1_object`) certifies separately — comparing it against
    the raw flavour-in-labels BPS chart would mismatch purely by convention."""
    if name == "pure_SU2":
        return [((0, 1),), ((1, 1),), ((_WILSON, 1),)]
    if name in ("SU2_Nf1", "SU2_Nf2", "SU2_Nf3"):
        flav = {"SU2_Nf1": 0, "SU2_Nf2": (0, 0), "SU2_Nf3": (0, 0, 0)}[name]
        return [(((0, 1),), flav), (((1, 1),), flav)]
    return []


def su2_family_cone_iso(name: str):
    """The **cone ↔ BPS** `KAlgebraIso` for theory `name` when a packaged witness
    exists (pure SU(2) and N_f=1, reusing the certified object-layer isos), else
    `None`.  Certify it **focused on the rays** (`su2_family_cone_rays`) — the
    ray OPE + ρ are q-exact and ring-independent, the clean part of the
    correspondence.

    For **N_f=2** an explicit full-battery witness now exists — see
    `su2_family_cone_bps_iso` (the σ-map H-tower between the flavour-reduced cone
    and the BPS chart).  N_f=3 stays a follow-up (its gauge H-tower is
    matter-dressed); this object-layer accessor returns `None` for N_f=2/3."""
    if name == "pure_SU2":
        from pure_su2_object import pure_su2_object
        return pure_su2_object().iso("cone", "bps")
    if name == "SU2_Nf1":
        from su2_nf1_object import su2_nf1_object
        return su2_nf1_object(with_flows=False).iso("cone", "bps")
    return None


# ---------------------------------------------------------------------------
# cone↔BPS through the flavour reduction
# ---------------------------------------------------------------------------
# `BPSKAlgebra` carries only **abelian** (Cartan) flavour, while the N_f=2/3
# cone presentations carry the **non-abelian** flavour ring (Spin(4)=SU(2)×SU(2)
# for N_f=2, SU(4) for N_f=3).  Comparing them directly is a category error: the
# cone must first be **reduced to its Cartan** (the flavour-reduction method —
# `base_change` along the Cartan restriction hom), after which it matches the
# abelian BPS chart.  This is the `su2_nf3_h_iso.verify_trace_via_abelian` pattern
# applied to the cone.
def _spin4_to_cartan_hom(R):
    """`R(Spin(4)) = R(SU(2))⊗R(SU(2)) → R(U(1)²)` (the Cartan restriction):
    each SU(2) spin `n` branches to its weights `Σ_k z^{n-2k}`, one U(1) per
    factor."""
    from zplus_ring import RingHom, RElement, AbelianZPlusRing
    A2 = AbelianZPlusRing(rank=2)

    def on_basis(b):
        nL, nR = b
        terms: dict = {}
        for kL in range(nL + 1):
            for kR in range(nR + 1):
                key = (nL - 2 * kL, nR - 2 * kR)
                terms[key] = terms.get(key, 0) + 1
        return RElement(A2, terms)

    return RingHom(R, A2, on_basis)


def _sun_to_cartan_hom(R):
    """`R(SU(N)) → R(U(1)^{N-1})` via the ring's own `to_abelian` torus
    embedding (the general SU(N) Cartan restriction)."""
    from zplus_ring import RingHom, AbelianZPlusRing
    rank = (R.N - 1) if hasattr(R, "N") else 3        # SU4ZPlusRing → rank 3
    A = AbelianZPlusRing(rank=rank)
    return RingHom(R, A, lambda b: R.to_abelian(R.basis_element(b), A))


def su2_family_cone_cartan_hom(name: str):
    """The **flavour-reduction** `RingHom` that restricts theory `name`'s cone
    coefficient ring to its abelian Cartan, or `None` when the cone is already
    abelian (N_f=1) or absent (N_f=4, SU(2)-SU(2)).  Required because
    `BPSKAlgebra` is abelian-flavour only — the cone↔BPS comparison goes through
    this reduction."""
    cone = su2_family_cone(name)
    if cone is None:
        return None
    R = cone.coefficient_ring()
    cls = type(R).__name__
    if cls == "AbelianZPlusRing":
        return None                                   # already abelian (N_f=1)
    if cls == "TensorZPlusRing":
        return _spin4_to_cartan_hom(R)                # Spin(4), N_f=2
    if cls in ("SU4ZPlusRing", "SUNZPlusRing"):
        return _sun_to_cartan_hom(R)                  # SU(4), N_f=3
    raise NotImplementedError(f"{name}: no Cartan reduction for ring {cls}")


def su2_family_cone_reduced(name: str):
    """The cone presentation with its flavour **reduced to the abelian Cartan**
    (`cone.base_change(Cartan-restriction)`), so it is directly comparable to the
    abelian `BPSKAlgebra`.  For N_f=1 (already abelian) returns the cone
    unchanged; `None` when there is no cone."""
    cone = su2_family_cone(name)
    if cone is None:
        return None
    hom = su2_family_cone_cartan_hom(name)
    return cone if hom is None else cone.base_change(hom)


# ---------------------------------------------------------------------------
# explicit cone↔BPS KAlgebraIso, N_f=2
# ---------------------------------------------------------------------------
# The full label↔label witness, built exactly the way the author prescribed:
#
#   * match the two **seeds** — 't Hooft `H_0` → tropical `(1,0;0,0)` and Wilson
#     `w_1` → `(0,-1;0,0)` — and **the rest is determined** by the BPS multiply
#     (`W·H_0 ∋ H_1, H_{-1}`);
#   * the gauge tower is the **piecewise-linear σ map** (`ρ` is *not* linear on
#     tropical charges) — `H_n → (1,n)` for `n≤0`, `H_n → (-1, 2-n)` for `n≥1`
#     (so `H_1 → (-1,1)`, `H_{-1} → (1,-1)` — exactly the author's reflection);
#   * Wilson `w_e → (0,-e)`; the Cartan flavour weight rides along into the BPS
#     flavour slots **unchanged** (`(wL,wR) → slots`).
#
# This map closes the *entire* multiplication table (the cone keeps flavour as a
# Cartan **weight in the label**, q-Laurent coefficients — same shape as the BPS
# chart's flavour slots — so multiply needs **no** coefficient bridge), and is
# `ρ`-equivariant, unit- and round-trip-exact.  Trace (the Schur index) is the
# only flavour-basis-dependent leg: the cone is over Spin(4) **characters** while
# the BPS chart is over Cartan **weights**, so trace-equivariance holds against
# the **flavour-reduced** cone (`su2_family_cone_reduced`, the inverse of the
# "painful promotion" that built the non-abelian class) — the augmentation the
# user pointed to.  The construction is the RG-node-drop-to-pure-SU(2) gauge
# tower (its `ρ` shift `H_n → H_{n-2}`, `H_0²` and the seed both match pure
# SU(2)) recombined with the matter, exactly as the validating oracles were.
def _su2_nf2_h_table(n: int) -> tuple:
    """SU(2)+N_f=2 BPS gauge H-tower (the piecewise-linear σ map): `H_n →`
    `(1,n)` for `n≤0`, `(-1, 2-n)` for `n≥1`."""
    return (1, n) if n <= 0 else (-1, 2 - n)


def _su2_nf2_gauge_charge(h_factors) -> tuple:
    """Gauge tropical charge `(n1,n2)` of a cone gauge monomial: sum the
    per-H-letter `_su2_nf2_h_table` charges, with Wilson `w_e → (0,-e)`."""
    n1 = n2 = 0
    for gen, exp in h_factors:
        if isinstance(gen, int):
            h = _su2_nf2_h_table(gen)
            n1 += exp * h[0]
            n2 += exp * h[1]
        elif isinstance(gen, tuple) and gen[0] == "W":
            n2 -= gen[1]
        else:
            raise ValueError(f"unrecognised gauge generator {gen!r}")
    return (n1, n2)


def _su2_nf2_inverse_gauge():
    """Build the `(n1,n2) → h_factors` inverse from the cone's own
    `(m,e)`-monomial canonicaliser (`pure_su2_h_cone_data._psu2_to_native`,
    the shared gauge-cone structure)."""
    from pure_su2_h_cone_data import _psu2_to_native
    rev: dict = {}
    for m in range(0, 8):
        for e in range(-12, 13):
            try:
                hf = _psu2_to_native(m, e)
            except Exception:
                continue
            rev[_su2_nf2_gauge_charge(hf)] = hf
    return rev


def su2_family_cone_bps_iso(name: str):
    """The **explicit cone ↔ BPS `KAlgebraIso`** for `name`, built from the
    flavour-reduced cone (so both sides are abelian-Cartan — the comparison the
    user prescribed).  Returns `None` when no such packaged witness exists.

    * `pure_SU2` / `SU2_Nf1` defer to the object-layer isos
      (`su2_family_cone_iso`) — already full-battery certified.
    * `SU2_Nf2` is the **non-abelian-flavour keystone**: the
      `_su2_nf2_h_table` σ-map gauge tower + flavour-slot ride-along, between
      `su2_family_cone_reduced("SU2_Nf2")` (Spin(4) reduced to its U(1)² Cartan)
      and `build_bps_su2_nf2()`.  Certified `unit / round_trip / multiplicative /
      rho_equivariant` exact and `trace_equivariant` to the cone trace's stable
      window (see `tests/test_bps_atlas_su2_family.py`).
    * `SU2_Nf3` returns `None`: its BPS gauge H-tower is **matter-dressed**
      (`H_0·w_1` yields a single clean gauge term, not the `H_1, H_{-1}`
      bifurcation), so the flavour-neutral σ-table does not close — a documented
      follow-up needing the dressed-H map.
    """
    if name in ("pure_SU2", "SU2_Nf1"):
        return su2_family_cone_iso(name)
    if name != "SU2_Nf2":
        return None
    from kalgebra import Element
    from laurent_poly import LaurentPoly
    from kalgebra_iso import KAlgebraIso
    one = LaurentPoly.one()
    cone_r = su2_family_cone_reduced("SU2_Nf2")
    bps = __import__("bps_su2_nf2").build_bps_su2_nf2()
    rev = _su2_nf2_inverse_gauge()

    def forward(label):
        h_factors, (wL, wR) = label
        n1, n2 = _su2_nf2_gauge_charge(h_factors)
        return Element({(n1, n2, wL, wR): one})

    def inverse(charge):
        n1, n2, wL, wR = charge
        return Element({(rev[(n1, n2)], (wL, wR)): one})

    return KAlgebraIso(
        cone_r, bps,
        forward_label_map=forward,
        inverse_label_map=inverse,
        name="SU(2)+N_f=2 cone(Cartan) ≅ BPS  "
             "(σ-map H-tower H_0→(1,0), w_1→(0,-1); the rest determined)",
    )


def su2_family_cone_index_matches_bps(name: str, *, K: int = 6,
                                      labels=None) -> bool:
    """Certify the cone↔BPS correspondence **through the flavour reduction**: the
    flavour-reduced cone's Schur index (trace) reproduces the BPS chart's, on the
    given `labels` (default: the vacuum / identity — the flavoured index, where
    the label correspondence is canonical).  This is the correct comparison the
    user prescribed: reduce the non-abelian cone to its Cartan first, then match
    the abelian BPS index."""
    cone_r = su2_family_cone_reduced(name)
    if cone_r is None:
        return False
    cone = su2_family_cone(name)
    # the cone's reference BPS chart (the one it was distilled from)
    nf = {"SU2_Nf1": 1, "SU2_Nf2": 2, "SU2_Nf3": 3}.get(name)
    if nf is None:                                     # pure SU(2): atlas root
        bps = su2_family_entry(name).chart
    else:
        bps = __import__(f"bps_su2_nf{nf}").__dict__[f"build_bps_su2_nf{nf}"]()
    if labels is None:
        labels = [(cone.identity(), tuple(bps.identity()))]
    return all(cone_r.trace(cl, K) == bps.trace(bl, K) for cl, bl in labels)


# ---------------------------------------------------------------------------
# object-layer packaging (the design record / Stage-4 export shape)
# ---------------------------------------------------------------------------
def su2_family_object(name: str):
    """A `KAlgebraObject` for theory `name` — the Stage-4 export shape: one
    abstract algebra holding its **certified presentations** under one roof.

    * `pure_SU2` / `SU2_Nf1` reuse the dedicated, fuller objects
      (`pure_su2_object` — abe / cone / bps; `su2_nf1_object` — cone / bps / abe
      / bps-rform / node-drop RG / mutation), which already wire and certify the
      cone leg.
    * `SU2_Nf2/3/4` and `SU2_SU2` get a fresh object with the **`bps`** chart
      (atlas root) and the **`rg`** node-drop presentation, joined by the
      guaranteed identity-on-labels `KAlgebraIso`.  (No cone leg wired here:
      for **N_f=2** the explicit cone↔BPS iso *exists* — `su2_family_cone_bps_iso`
      — but it targets the hand-built `bps_su2_nf2` chart (the cone's validating
      oracle), a *different frame* from this atlas-root chart, so wiring it would
      need the frame-change leg; accessible standalone instead.  N_f=3 cone iso
      is a follow-up (matter-dressed H-tower); N_f=4 / SU(2)-SU(2) have no cone.)
    """
    if name == "pure_SU2":
        from pure_su2_object import pure_su2_object
        return pure_su2_object()
    if name == "SU2_Nf1":
        from su2_nf1_object import su2_nf1_object
        return su2_nf1_object()
    from kalgebra_object import KAlgebraObject
    e = su2_family_entry(name)
    obj = KAlgebraObject(f"A_q[{name}]")
    obj.add_realization("bps", e.chart, {"chart", "rg", "trace-exact"})
    iso = su2_family_rg_iso(name)                 # rg → bps, identity-on-labels
    obj.add_realization("rg", iso.source, {"rg", "flow", "node-drop"})
    obj.add_iso("rg", "bps", iso)
    return obj


def su2_family_objects() -> dict:
    """`{name: KAlgebraObject}` for the whole family (built on demand)."""
    return {nm: su2_family_object(nm) for nm in su2_family_names()}


if __name__ == "__main__":
    for nm in su2_family_names():
        e = su2_family_entry(nm)
        print(f"{nm:9s} rank={e.rank} |spec|={len(e.chart.spec):2d} "
              f"matter={list(e.matter_indices)} cone={'yes' if e.has_cone else 'no':3s} "
              f"coeff={e.chart.coefficient_ring()}")
