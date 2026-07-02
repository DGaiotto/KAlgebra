"""`RGKAlgebraObject` — one RG *flow*, held as parallel descriptions.

`KAlgebraObject` identifies abstract *algebras*: its witnesses are
`KAlgebraIso`s, and e.g. the chambers of a BPS chart graph are
perfectly good realizations of one algebra while being **different
flows** (different spectrum generators).  This module refines the
holder one level: an RG flow is more data than its UV algebra — an
auxiliary (IR) algebra, the `RG` embedding, the spectrum generator
`S_RG`, the grading, the apex labelling — and the *same flow* often
admits parallel descriptions (unimodular re-coordinatizations of the
IR torus, thin-constructor pairs like `rg_flow.SubquiverRG` over
`DirectionalSubquiverRG`, a BPS realisation vs the generic co-solver
build).

* `RGKAlgebraIso(KAlgebraIso)` — a witness that two `RGKAlgebra`s
  describe the same flow: the inherited part is the UV-side iso (so
  everything `KAlgebraObject` needs — composition, inversion,
  transport, the algebra battery — keeps working), plus an `aux_iso`
  between the auxiliaries, with flow-level verifiers:

      verify_rg_intertwine :  aux_iso(RG₁(a)) == RG₂(uv(a))
      verify_s_rg_match    :  aux_iso(S₁_RG)  == S₂_RG   (to a window)
      verify_apex_match    :  aux_iso(L_{apex₁(a)}) == L_{apex₂(uv a)}

  (Grading-intertwine is a documented follow-up; for same-Γ frame
  changes it is subsumed by the S-match.)

* `RGKAlgebraObject(KAlgebraObject)` — same groupoid holder with
  flow-typed realizations and witnesses; `aux_object()` projects to
  the `KAlgebraObject` of the auxiliaries, and `verify_flow_pairwise`
  runs the flow battery per stored witness.

The algebra/flow distinction is executable: a chart-mutation witness
(`bps_chart_object`) is a certified `KAlgebraIso` of algebras, but
wrapped as a flow witness its `verify_s_rg_match` FAILS — different
chamber, different flow (see `tests/test_rgkalgebra_object.py`).
"""
from __future__ import annotations

from kalgebra import Element, KAlgebra
from kalgebra_iso import KAlgebraIso
from kalgebra_object import KAlgebraObject
from laurent_poly import LaurentPoly
from rgkalgebra import RGKAlgebra


__all__ = ["RGKAlgebraIso", "RGKAlgebraObject", "subquiver_flow_object"]

_ONE = LaurentPoly.one()


class RGKAlgebraIso(KAlgebraIso):
    """A flow witness: UV-side `KAlgebraIso` (inherited) + an iso of
    the auxiliaries, verified to intertwine the RG data.

    The endpoints must be `RGKAlgebra` instances and `aux_iso` must
    connect exactly `source.auxiliary()` to `target.auxiliary()` (as
    Python objects, like every iso endpoint in this groupoid)."""

    def __init__(self, source, target, forward_label_map,
                 inverse_label_map, aux_iso: KAlgebraIso,
                 name: str | None = None):
        if not isinstance(source, RGKAlgebra) or not isinstance(
                target, RGKAlgebra):
            raise TypeError(
                "RGKAlgebraIso endpoints must be RGKAlgebra instances")
        if aux_iso.source is not source.auxiliary():
            raise ValueError(
                "RGKAlgebraIso: aux_iso.source is not source.auxiliary()")
        if aux_iso.target is not target.auxiliary():
            raise ValueError(
                "RGKAlgebraIso: aux_iso.target is not target.auxiliary()")
        super().__init__(source, target, forward_label_map,
                         inverse_label_map, name=name)
        self.aux_iso = aux_iso

    # ---- groupoid ops keep the aux part --------------------------------

    @classmethod
    def identity(cls, alg: RGKAlgebra, name: str | None = None
                 ) -> "RGKAlgebraIso":
        def _id(label):
            return Element({label: _ONE})
        return cls(alg, alg, _id, _id,
                   KAlgebraIso.identity(alg.auxiliary()),
                   name=name or f"id_{type(alg).__name__}")

    def invert(self) -> "RGKAlgebraIso":
        return RGKAlgebraIso(
            self.target, self.source, self._inverse, self._forward,
            self.aux_iso.invert(),
            name=(self.name[1:] if self.name and
                  self.name.startswith("inv:")
                  else f"inv:{self.name}" if self.name else None),
        )

    def compose(self, other: "KAlgebraIso") -> "RGKAlgebraIso":
        if not isinstance(other, RGKAlgebraIso):
            raise TypeError(
                "RGKAlgebraIso.compose: a flow witness can only compose "
                "with another flow witness (got plain KAlgebraIso)")
        uv = super().compose(other)
        return RGKAlgebraIso(
            self.source, other.target, uv._forward, uv._inverse,
            self.aux_iso.compose(other.aux_iso), name=uv.name)

    # ---- flow-level verifiers ------------------------------------------

    def verify_rg_intertwine(self, source_labels) -> bool:
        """`aux_iso(RG₁(a)) == RG₂(uv(a))` on the sample labels."""
        for a in source_labels:
            lhs = self.aux_iso.map(self.source.RG(a))
            rhs = self.target.RG_element(self.map(Element({a: _ONE})))
            if lhs != rhs:
                return False
        return True

    def verify_s_rg_match(self, cutoff: int = 4,
                          K_expand: int | None = None) -> bool:
        """`aux_iso(S₁_RG) == S₂_RG`, both windowed to `cutoff` and
        expanded to `q^K_expand` (the same window the twist verifier
        uses)."""
        if K_expand is None:
            K_expand = max(4 * cutoff, 4)
        lhs = self.aux_iso.map(
            self.source._s_rg_as_aux_element(cutoff, K_expand))
        rhs = self.target._s_rg_as_aux_element(cutoff, K_expand)
        return lhs == rhs

    def verify_apex_match(self, source_labels) -> bool:
        """`aux_iso(L_{apex₁(a)}) == L_{apex₂(a')}` for one-term UV
        images `uv(L_a) = L_{a'}` (multi-term images are skipped —
        apexes are per-label data)."""
        for a in source_labels:
            img = self.map(Element({a: _ONE}))
            if len(img.terms) != 1:
                continue
            (a2,) = img.terms
            lhs = self.aux_iso.map(
                Element({tuple(self.source.apex(a)): _ONE}))
            rhs = Element({tuple(self.target.apex(a2)): _ONE})
            if lhs != rhs:
                return False
        return True

    def verify_grading_intertwine(self, aux_labels,
                                  charge_map=None) -> bool:
        """`deg₂(aux_iso(L)) == Λ(deg₁(L))` on auxiliary labels with
        one-term images, where `Λ` is an optional lattice map between
        the two `Γ_RG`s (default: identity — same-coordinates parallel
        descriptions).  Raises `ValueError` if either side does not
        expose a `grading()` (nothing to verify — do not silently
        pass)."""
        try:
            g1 = self.source.grading()
            g2 = self.target.grading()
        except NotImplementedError as exc:
            raise ValueError(
                f"verify_grading_intertwine: {exc}") from exc
        if charge_map is None:
            if g1.rank != g2.rank:
                raise ValueError(
                    "verify_grading_intertwine: Γ_RG ranks differ "
                    f"({g1.rank} vs {g2.rank}); supply charge_map")
            charge_map = lambda p: tuple(p)  # noqa: E731
        for l in aux_labels:
            img = self.aux_iso.map(Element({tuple(l): _ONE}))
            if len(img.terms) != 1:
                continue
            (l2,) = img.terms
            if tuple(g2.deg(l2)) != tuple(charge_map(g1.deg(tuple(l)))):
                return False
        return True

    def verify_flow(self, source_labels, *, cutoff: int = 4,
                    K_expand: int | None = None,
                    aux_labels=None) -> dict:
        """The flow-level bundle (algebra-level checks live in the
        inherited `verify_all` and in `aux_iso.verify_all`).  The
        grading check is included when both sides expose a `grading()`
        and `aux_labels` are supplied (identity charge map; call
        `verify_grading_intertwine` directly for a non-trivial Λ)."""
        out = {
            "rg_intertwine": self.verify_rg_intertwine(source_labels),
            "s_rg_match": self.verify_s_rg_match(cutoff, K_expand),
            "apex_match": self.verify_apex_match(source_labels),
        }
        if aux_labels is not None:
            out["grading_intertwine"] = self.verify_grading_intertwine(
                aux_labels)
        return out


class RGKAlgebraObject(KAlgebraObject):
    """One RG flow, as its parallel descriptions + flow witnesses.

    Realizations must be `RGKAlgebra`s and witnesses `RGKAlgebraIso`s;
    everything else (composed transport, capability routing, algebra
    batteries, coherence) is inherited — composition and inversion
    dispatch virtually, so composed witnesses stay flow-typed."""

    def add_realization(self, key, algebra, capabilities=()):
        if not isinstance(algebra, RGKAlgebra):
            raise TypeError(
                f"{self.name}: RGKAlgebraObject realizations must be "
                f"RGKAlgebra instances (got {type(algebra).__name__})")
        super().add_realization(key, algebra, capabilities)

    def add_iso(self, src, dst, iso):
        if not isinstance(iso, RGKAlgebraIso):
            raise TypeError(
                f"{self.name}: RGKAlgebraObject witnesses must be "
                f"RGKAlgebraIso (flow witnesses), got "
                f"{type(iso).__name__}")
        super().add_iso(src, dst, iso)

    def iso(self, src: str, dst: str) -> KAlgebraIso:
        if src == dst:
            out = self._iso_cache.get((src, dst))
            if out is None:
                out = RGKAlgebraIso.identity(self._realizations[src],
                                             name=f"id[{src}]")
                self._iso_cache[(src, dst)] = out
            return out
        return super().iso(src, dst)

    def aux_object(self) -> KAlgebraObject:
        """The projection to the auxiliaries: a `KAlgebraObject` with
        the same keys, realization = `auxiliary()`, witnesses = the
        `aux_iso` parts."""
        obj = KAlgebraObject(f"{self.name}[aux]")
        for key in self.keys():
            obj.add_realization(key, self._realizations[key].auxiliary(),
                                self._capabilities[key])
        for (s, d), iso in self._witnesses.items():
            obj.add_iso(s, d, iso.aux_iso)
        return obj

    def verify_flow_pairwise(self, samples_by_key, *, cutoff: int = 4,
                             K_expand: int | None = None) -> dict:
        """Run each stored witness's flow bundle on the source-side
        samples.  Returns `{(src, dst): {check: bool}}`."""
        out = {}
        for (s, d), iso in self._witnesses.items():
            out[(s, d)] = iso.verify_flow(
                samples_by_key[s], cutoff=cutoff, K_expand=K_expand)
        return out


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------

def subquiver_flow_object(A_uv, drop, *, name: str | None = None
                          ) -> RGKAlgebraObject:
    """One BPS node-deletion flow in its two live descriptions:

    * ``'f-oracle'``  — `rg_flow.SubquiverRG(A_uv, drop)`: `RG(a)` from
      the UV `F`-oracle (the historical extraction);
    * ``'co-solver'`` — a bare `DirectionalSubquiverRG` on the same
      `(pairing, node_charges, spec, drop)` data: `RG(a)` from the
      generic exact solve.

    The identity-shaped flow witness between them turns the historical
    "F-oracle cross-checked against the generic solve" validation into
    a curated certificate: `verify_rg_intertwine` IS that cross-check,
    `verify_s_rg_match` certifies the shared closed-form generator, and
    the inherited algebra battery compares the two derived KAlgebra
    surfaces.  Both auxiliaries are same-data IR `BPSKAlgebra`
    instances, bridged by the identity label map.

    Raises like `DirectionalSubquiverRG` when the spec does not
    cooperate (non-stuff-first orderings: arrange upstream first)."""
    from rg_flow import SubquiverRG
    from directional_subquiver_rg import DirectionalSubquiverRG

    drop = list(drop)
    F1 = SubquiverRG(A_uv, drop)
    F2 = DirectionalSubquiverRG(
        [list(r) for r in A_uv.lattice.pairing],
        list(A_uv.node_charges),
        list(A_uv.spec),
        drop=drop,
    )

    def _id(label):
        return Element({tuple(label): _ONE})

    aux_iso = KAlgebraIso(F1.auxiliary(), F2.auxiliary(), _id, _id,
                          name="aux-id")
    w = RGKAlgebraIso(F1, F2, _id, _id, aux_iso,
                      name=f"{name or 'subquiver-flow'}"
                           f"[f-oracle→co-solver]")
    obj = RGKAlgebraObject(name or f"flow[drop={drop}]")
    obj.add_realization("f-oracle", F1, {"rg", "uv-oracle"})
    obj.add_realization("co-solver", F2, {"rg", "generic-solve"})
    obj.add_iso("f-oracle", "co-solver", w)
    return obj
