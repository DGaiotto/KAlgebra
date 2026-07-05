"""
rgkalgebra_iso_via_ir.py
========================

Formalises the theorem

    **An RGKAlgebra is determined up to isomorphism by its
    `(IR algebra, RG map, S_RG)`.**

Concretely: if `inner` is an `RGKAlgebra` (a UV reconstructed from its
auxiliary IR₁ via `multiply_UV = from_ir_image(RG(a) ·_{IR} RG(b))`)
and `φ : IR₁ → IR₂` is a `KAlgebraIso`, then transporting the RG data
across `φ` yields an `RGKAlgebra` over IR₂ that is **isomorphic to
`inner` on the (shared) UV label set** — the UV reconstruction does
not depend on which (isomorphic) presentation of the IR one uses.

The constructive transport is exactly `IsoComposedRGKAlgebra(inner, φ)`
(it post-composes the RG map and `S_RG` with `φ`).  This module adds:

  * `rgkalgebra_iso_via_ir(inner, φ)` — the UV isomorphism witness
    `inner ≅ IsoComposedRGKAlgebra(inner, φ)` (identity on UV labels);
  * `verify_ir_iso_hypotheses(inner, φ, …)` — machine-checks the
    theorem's hypotheses on samples: that `φ` is a genuine
    `KAlgebraIso` of the IR, and that the transported `S_RG`
    (`φ(S_RG^{inner})`) is label-bijective in IR₂ (so the RG flow
    really does carry across).  The theorem is then invoked; the UV
    product itself — which on the slow side (e.g. `BPS`) may be
    intractable — is *not* recomputed.

Why this is the right tool.  To prove `A1A2kKAlg(k) ≅ BPS(A_{2k})`
inductively, both sides are the single-node RG (drop γ₂) over
isomorphic IRs (`A1A2kKAlg(k-1) ≅ BPS(A_{2k-2})`, the inductive
hypothesis) with the *same* `S_RG = E_𝖖(X_{γ₁})`.  The theorem gives
the UV iso from the IR iso + `S_RG` match — no `BPS` F-finder, no UV
product recomputation.  The hypotheses (IR iso, `S_RG` transport) are
what the verifier checks, and those live on the *fast* IR side.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from typing import Iterable

from kalgebra import Element
from laurent_poly import LaurentPoly
from kalgebra_iso import KAlgebraIso
from rgkalgebra import RGKAlgebra
from iso_composed_rgkalgebra import IsoComposedRGKAlgebra


def rgkalgebra_iso_via_ir(
    inner: RGKAlgebra,
    phi: KAlgebraIso,
    *,
    transported: "IsoComposedRGKAlgebra | None" = None,
    name: str | None = None,
) -> KAlgebraIso:
    """UV isomorphism witness `inner ≅ IsoComposedRGKAlgebra(inner, φ)`.

    `φ` must run `inner.auxiliary() → IR₂`.  The two UV algebras share
    `inner`'s canonical label set, and by the reconstruction theorem
    they have the same K-algebra structure; the witness is therefore
    the identity on UV labels.  Pass an already-built `transported`
    to reuse it (avoids constructing a second `IsoComposedRGKAlgebra`).
    """
    if phi.source is not inner.auxiliary():
        raise ValueError(
            "rgkalgebra_iso_via_ir: φ.source must be (identically) "
            f"inner.auxiliary(); got {type(phi.source).__name__} vs "
            f"{type(inner.auxiliary()).__name__}."
        )
    if transported is None:
        transported = IsoComposedRGKAlgebra(inner, phi)
    one = LaurentPoly.one()

    def idmap(label):
        return Element({label: one})

    return KAlgebraIso(
        source=inner, target=transported,
        forward_label_map=idmap, inverse_label_map=idmap,
        name=name or f"{type(inner).__name__} ≅ via-IR[{phi.name}]",
    )


def verify_ir_iso_hypotheses(
    inner: RGKAlgebra,
    phi: KAlgebraIso,
    ir_samples: Iterable,
    ir_pairs: Iterable,
    *,
    s_rg_support: "Iterable | None" = None,
    cutoff: int = 4,
    rho: bool = True,
) -> dict:
    """Machine-check the theorem's hypotheses on samples.

    Returns `{check: bool}` for:

      * ``phi_unit`` / ``phi_round_trip`` / ``phi_multiplicative`` /
        (``phi_rho`` if `rho`): `φ` is a genuine `KAlgebraIso`
        ``inner.auxiliary() → IR₂`` on the supplied IR samples/pairs.
      * ``s_rg_transports``: every `S_RG^{inner}` support label maps
        under `φ` to a single IR₂ canonical-basis term with
        coefficient 1 (so the RG generator carries across
        label-bijectively — the condition
        `IsoComposedRGKAlgebra.rg_generator` needs, and the
        ``φ(S_RG)=S_RG`` half of the hypothesis).

    `ir_samples` / `ir_pairs` are `Element`s / `(Element, Element)`
    pairs over `inner.auxiliary()`.

    `s_rg_support`: the IR₁ canonical labels on which `S_RG` is
    supported.  Pass it explicitly (e.g. `{n·γ₁}` for the isolated-node
    `E_𝖖(X_{γ₁})`) to avoid the (often expensive / non-terminating)
    `inner.rg_generator` peeling.  If `None`, the support is read from
    `inner.rg_generator(cutoff)` — only do this when that call is cheap
    for `inner`.  When all checks pass, the reconstruction theorem
    yields the UV iso returned by `rgkalgebra_iso_via_ir` (no UV
    product is recomputed).
    """
    ir_samples = list(ir_samples)
    tgt_samples = [phi.map(s) for s in ir_samples]
    ir_pairs = list(ir_pairs)
    tgt_pairs = [(phi.map(a), phi.map(b)) for a, b in ir_pairs]

    out = {
        "phi_unit": phi.verify_unit(),
        "phi_round_trip": phi.verify_round_trip(ir_samples, tgt_samples),
        "phi_multiplicative": phi.verify_multiplicative(ir_pairs, tgt_pairs),
    }
    if rho:
        out["phi_rho"] = phi.verify_rho_equivariant(ir_samples, tgt_samples)

    one = LaurentPoly.one()
    if s_rg_support is None:
        s_rg_support = list(inner.rg_generator(cutoff).keys())
    ok = True
    for label in s_rg_support:
        img = phi._forward(label)
        terms = list(img.terms.items())
        if len(terms) != 1 or not (terms[0][1] == one):
            ok = False
            break
    out["s_rg_transports"] = ok
    return out
