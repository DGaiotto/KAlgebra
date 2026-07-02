"""`U1SquareRGKAlgebra` — SQED₁ / U1Square presented as an **RG flow to its
quantum torus** (the simplest non-trivial `RGKAlgebra`, the `deg = id` corner).

This is the reference example for the **RG layer**
(the RG-flow engine).  Unlike the cone layer, which carries *frozen
reductions* with no engine on the path, this exercises the **generic `RGKAlgebra`
machinery itself**: every operation — `RG`, `multiply`, `rho`/`rho_inverse`,
`trace`, `inner_product` — is *inherited* from `RGKAlgebra` and computed live
from the flow data.  Nothing is overridden to a closed form.

Defining data (the whole of it)
-------------------------------
* `auxiliary()` = the **sample** Z² quantum torus `Z2QTorusSampleKAlgebra`
  (`⟨(a,b),(c,d)⟩ = ad − bc`, `ρ(γ) = −γ`, `Tr X_γ = (𝖖²;𝖖²)_∞²·δ_{γ,0}`).  Using
  the *sample* torus — not the BPS quantum torus — is the design-critical choice:
  the dependency closure is {core + RG engine}, never the BPS spine
  (`bps_kalgebra` and the rest of `src/bps/`).
* `grading()` = `Γ_RG = Z`, `deg(X_{a,b}) = b` (the magnetic direction the
  spectrum generator lives on), height `h(b) = b` (⇒ a pointed cone, strictly
  positive on the appearing charges).
* `S_RG = E_𝖖(X_{0,1})` — a **single** quantum dilogarithm, given in *both*
  contracts: `rg_generator(cutoff)` (q-order window) and the exact per-charge
  oracle `_s_rg_component((b,)) = {(0,b): e_b}` (a singleton, since `deg = id`).
  `e_i = (−1)^i 𝖖^i / (𝖖²;𝖖²)_i` are the `E_𝖖` coefficients, and
  `X_{0,1}^i = X_{0,i}` (self-pairing 0).
* `apex` = identity (UV canonical labels *are* the QT lower-tropical charges
  `γ ∈ Z²`).

Truncation-safe trace
---------------------
The pairing and trace are computed by the generic engine's **bilinear
expansion** (one `S_RG` per leg):

    I_{a,b} = Σ_{c,d} [RG(a)·S_RG]_c · [RG(b)·S_RG]_d · I^aux_{c,d},
    Tr(a)   = Σ_{c,d} [S_RG]_c       · [RG(a)·S_RG]_d · I^aux_{c,d},

with `I^aux_{c,d} = aux.inner_product(c, d)` a *well-defined* single-basis
pairing.  The undefined opposite-cone product `ρ(S_RG)·S_RG` (a formal sum over
the negative grading cone times one over the positive cone) is never formed.
The per-charge components `[RG(·)·S_RG]_c` are computed exactly (the feeding
S-charge `γ = deg(η) − deg(δ)` is forced by grading additivity, `[S_RG]_γ`
fetched whole, expanded to `q^K` **last**), so the result is exact to any `K`.

The single optional method this needs is `_s_rg_component` (the exact graded
`[S_RG]_γ` oracle); with it and the cone-equipped `grading()`, the engine turns
on its exact per-charge path — keyed on the *grading*, not on the auxiliary —
so the sample `Z2QTorusSampleKAlgebra` auxiliary is never modified.

Correctness is certified by a `KAlgebraIso` to `SQED1SampleKAlgebra` (and,
through it, to the `U1SquareKAlg` / `Sqed1KAlg` cone presentations).
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rgkalgebra import RGKAlgebra
from grading import Grading
from habiro import HabiroElement
from samples import Z2QTorusSampleKAlgebra


__all__ = ["U1SquareRGKAlgebra"]


def _eq_coeff(i: int) -> HabiroElement:
    """The `E_𝖖` coefficient `e_i = (−1)^i 𝖖^i / (𝖖²;𝖖²)_i` as an exact
    `HabiroElement` (`e_0 = 1`).  `S_RG = E_𝖖(X_{0,1}) = Σ_i e_i X_{0,i}`."""
    return HabiroElement.q_power(i, (-1) ** i) * HabiroElement.pochhammer_inverse(i)


class U1SquareRGKAlgebra(RGKAlgebra):
    """SQED₁ = U1Square as the RG flow `S_RG = E_𝖖(X_{0,1})` to the Z² quantum
    torus.  A pure `RGKAlgebra`: all K-algebra operations are inherited.

    Canonical labels are the QT charges `γ = (m, n) ∈ Z²` (apex = identity); the
    `KAlgebraIso` to `SQED1SampleKAlgebra` relabels them to the `(m, n)`
    `u_±`/`v` presentation."""

    def __init__(self) -> None:
        self._aux = Z2QTorusSampleKAlgebra()

    # ----- RGKAlgebra defining data --------------------------------------

    def auxiliary(self) -> Z2QTorusSampleKAlgebra:
        return self._aux

    def grading(self) -> Grading:
        return Grading(
            rank=1,
            deg=lambda lbl: (lbl[1],),          # deg(X_{a,b}) = b
            height=(1,),                        # h(b) = b
            cone_gens=((1,),),                  # positive cone = Z_{≥0}
        )

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG = E_𝖖(X_{0,1})` windowed to q-order ≤ `cutoff`.

        `e_i` has leading q-order `i`, so the q-order window keeps `i ≤ cutoff`;
        `[S_RG]_0 = 1_B = X_{0,0}`."""
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        return {(0, i): _eq_coeff(i) for i in range(cutoff + 1)}

    def _s_rg_component(self, p) -> dict:
        """Exact `Γ_RG`-graded component `[S_RG]_{(b,)}`: the singleton
        `{(0, b): e_b}` for `b ≥ 0` (since `deg = id` makes each component
        one-dimensional), `{}` off the positive cone (`b < 0`).

        This is the **only** optional method the truncation-safe trace needs:
        with it (plus the cone-equipped `grading()` above) the generic
        `RGKAlgebra` turns on the exact per-η FS path (`_fs_exact_available`
        keys on the *grading*, not on the auxiliary), so `rg_times_s_rg` /
        `trace` / `inner_product` are exact-and-truncation-safe with no further
        subclass code — the sample `Z2QTorusSampleKAlgebra` auxiliary is left
        untouched."""
        (b,) = p
        if b < 0:
            return {}
        return {(0, b): _eq_coeff(b)}

    def __repr__(self) -> str:
        return "U1SquareRGKAlgebra()"
