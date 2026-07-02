"""`PentagonSquareSampleRGKAlgebra` — the pentagon `K_𝖖([A_1, A_2])` (Yang-Lee /
M(2,5)) presented as an **RG flow to SQED₁**, the first RGKAlgebra
whose auxiliary is a *non-torus* K-algebra.

This is the Step-3 example with a non-torus flow target: the flow
target is the genuine gauge theory SQED₁ (the **sample** `SQED1SampleKAlgebra`,
not its quantum-torus shadow), so the spectrum generator and the trace must
cope with SQED₁'s non-commutative relation `u₊u₋ = 1+𝖖v`.  Like
`U1SquareRGKAlgebra` it is a **pure** `RGKAlgebra`: `RG`, `multiply`,
`ρ`/`ρ⁻¹`, `trace`, `inner_product` are all *inherited* from the generic engine
and computed live from the flow data — nothing is overridden to a closed form,
and in particular the trace is the generic bilinear pairing rather than a
presentation-specific heuristic override.

Defining data (the whole of it)
-------------------------------
* `auxiliary()` = the **sample** SQED₁ `SQED1SampleKAlgebra` (`u_±/v`, labels
  `(m, n)`: `m` the magnetic / charged-hyper charge, `n` the gauge-monopole `v`
  power; `u₊u₋ = 1+𝖖v`).  The *sample* (not the cone `Sqed1KAlg` / BPS torus) is
  the design-critical choice: the dependency closure is {core + RG engine},
  never the BPS spine.
* `grading()` = `Γ_RG = Z`, `deg(m, n) = m` (the **magnetic charge** — additive
  under SQED₁'s multiply, since `u₊u₋` lands in the `m = 0` Coulomb sector),
  height `h(m) = −m`, positive cone `m ≤ 0` (`cone_gens = ((−1,),)`).  `S_RG`
  lives on the `m ≤ 0` ray, so this is a genuine pointed-cone RG grading.
  (The *RG images* need not be homogeneous — `RG(L₂) = (1,−1)+(0,−1)` is
  inhomogeneous in `m`, and that is fine: `RG(a)` is a finite element; only
  `S_RG` and the grading must live on the pointed cone.)
* `S_RG = E_𝖖(u₋)` — a **single** quantum dilogarithm on the monopole `u₋ =
  (−1, 0)`, given in both contracts: `rg_generator(cutoff)` (q-order window) and
  the exact per-charge oracle `_s_rg_component((m,)) = {(−|m|, 0): e_{|m|}}` (a
  singleton, since `deg` is injective on the `u₋` ray).  `e_i = (−1)^i 𝖖^i /
  (𝖖²;𝖖²)_i` are the `E_𝖖` coefficients.
* `apex` / `_apex_inverse` — the pentagon↔SQED₁ tropical identification: a
  pentagon label `(i, a, b)` maps to the SQED₁ charge of its leading monopole,
  and an SQED₁ charge decomposes into the five pentagon chord-cones.  Both are
  pure cone arithmetic on the five chord charges (`_CHORD_CHARGE`).

Truncation-safe trace (the point of the example)
-----------------------------------------------
The trace is the generic engine's **bilinear expansion** (one `S_RG` per leg):

    I_{a,b} = Σ_{c,d} [RG(a)·S_RG]_c · [RG(b)·S_RG]_d · I^aux_{c,d},
    Tr(a)   = Σ_{c,d} [S_RG]_c       · [RG(a)·S_RG]_d · I^aux_{c,d},

with `I^aux_{c,d} = SQED1.inner_product(c, d)` a *well-defined* single-basis
pairing.  The undefined opposite-cone product `ρ(S_RG)·S_RG` is never formed.

What makes the **non-torus** case work is the FS support heuristic: SQED₁'s
`u₊u₋ = 1+𝖖v` means `RG(a)·S_RG` spreads not only along the `S_RG` ray `(−1,0)`
but into `v`-bands (the `(−1,1)` residue direction).  The generic
`RGKAlgebra._rg_times_s_rg_exact` finds exactly these labels because it generates
its support-walk neighbours by *multiplying* each label by the `S_RG` ray-unit
labels in the auxiliary (`_fs_ray_unit_labels`) — so `u₊·u₋ = 1+𝖖v` produces the
`v`-residue automatically; the steps are height-positive by grading additivity,
so the order-guided walk still terminates.  No subclass-specific FS code is needed
here.

Correctness is certified by a `KAlgebraIso` to `PentagonSampleKAlgebra` (and,
through it, to the cone presentations); the trace matches the direct
`PentagonSampleKAlgebra` trace exactly and is truncation-stable.
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from laurent_poly import LaurentPoly
from rgkalgebra import RGKAlgebra
from grading import Grading
from habiro import HabiroElement
from samples import SQED1SampleKAlgebra


__all__ = ["PentagonSquareSampleRGKAlgebra"]


# The five pentagon chord charges (γ_i), in SQED₁-friendly BPS coordinates.
# Pure data; the pentagon↔SQED₁ label maps are cone arithmetic on these.
_CHORD_CHARGE = {0: (1, 0), 1: (0, -1), 2: (-1, -1), 3: (-1, 0), 4: (0, 1)}


def _pent_to_bps(i: int, a: int, b: int) -> tuple[int, int]:
    """Pentagon label `(i, a, b)` → BPS charge `a·γ_i + b·γ_{i+1}`."""
    gi, gi1 = _CHORD_CHARGE[i % 5], _CHORD_CHARGE[(i + 1) % 5]
    return (a * gi[0] + b * gi1[0], a * gi[1] + b * gi1[1])


def _bps_to_pent(charge: tuple[int, int]) -> tuple[int, int, int]:
    """BPS charge `(m, n)` → pentagon canonical label `(i, a, b)` (decompose into
    the five cones spanned by consecutive `(γ_i, γ_{i+1})` pairs)."""
    m, n = charge
    if m == 0 and n == 0:
        return (0, 0, 0)
    for i in range(5):
        gi, gi1 = _CHORD_CHARGE[i], _CHORD_CHARGE[(i + 1) % 5]
        det = gi[0] * gi1[1] - gi[1] * gi1[0]
        if det == 0:
            continue
        a_num = m * gi1[1] - n * gi1[0]
        b_num = -m * gi[1] + n * gi[0]
        if a_num % det or b_num % det:
            continue
        a, b = a_num // det, b_num // det
        if a < 0 or b < 0:
            continue
        if a == 0 and b == 0:
            return (0, 0, 0)
        if b == 0:
            return (i, a, 0)
        if a == 0:
            return ((i + 1) % 5, b, 0)
        return (i, a, b)
    raise ValueError(f"_bps_to_pent: charge {charge} not in any chord cone")


def _pent_to_sqed1(i: int, a: int, b: int) -> tuple[int, int]:
    """Pentagon apex label → SQED₁ charge `(m, n)`.  SQED₁ `(m,n) ↔ BPS (n, −m)`,
    so `(m, n) = (−n_b, m_b)`."""
    m_b, n_b = _pent_to_bps(i, a, b)
    return (-n_b, m_b)


def _sqed1_to_pent(m: int, n: int) -> tuple[int, int, int]:
    """SQED₁ charge `(m, n)` → pentagon label, via the BPS charge `(n, −m)`."""
    return _bps_to_pent((n, -m))


def _eq_coeff(i: int) -> HabiroElement:
    """`e_i = (−1)^i 𝖖^i / (𝖖²;𝖖²)_i` — the `E_𝖖` coefficient (`u₋` has
    self-pairing 0, so `u₋^i` is a single unit-coefficient label)."""
    return HabiroElement.q_power(i, (-1) ** i) * HabiroElement.pochhammer_inverse(i)


class PentagonSquareSampleRGKAlgebra(RGKAlgebra):
    """The pentagon `K_𝖖([A_1, A_2])` as the RG flow `S_RG = E_𝖖(u₋)` to the
    **sample** SQED₁.  A pure `RGKAlgebra`: every K-algebra operation is
    inherited from the generic engine (including the truncation-safe bilinear
    trace over the non-torus SQED₁ auxiliary).

    Canonical labels are the pentagon labels `(i, a, b)` (`L_i^a L_{i+1}^b`); the
    `KAlgebraIso` to `PentagonSampleKAlgebra` is the identity on labels."""

    def __init__(self, aux=None) -> None:
        """`aux` defaults to the **sample** SQED₁ (`SQED1SampleKAlgebra`).  Any
        label-identical SQED₁ realisation works (e.g. the cone `Sqed1KAlg`) — the
        flow data (`deg = m`, `S_RG = E_𝖖(u₋)`, the apex peel) depends only on the
        `(m, n)` `u_±/v` presentation, so swapping the IR realisation is exact.
        The object layer pairs the sample- and cone-aux instances on this basis."""
        self._aux = aux if aux is not None else SQED1SampleKAlgebra()

    # ----- RGKAlgebra defining data --------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        return Grading(
            rank=1,
            deg=lambda lbl: (lbl[0],),          # deg(m, n) = m (magnetic charge)
            height=(-1,),                       # h(m) = −m  (> 0 on the cone)
            cone_gens=((-1,),),                 # S_RG on the m ≤ 0 ray
        )

    def identity(self):
        return (0, 0, 0)

    def apex(self, a):
        i, x, y = a
        return _pent_to_sqed1(i, x, y)

    def _apex_inverse(self, ir_label):
        m, n = ir_label
        return _sqed1_to_pent(m, n)

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG = E_𝖖(u₋)` windowed to q-order ≤ `cutoff` (`e_i` has leading
        q-order `i`); `[S_RG]_0 = 1_B = (0, 0)`."""
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        return {(-i, 0): _eq_coeff(i) for i in range(cutoff + 1)}

    def _s_rg_component(self, p) -> dict:
        """Exact `Γ_RG`-graded component `[S_RG]_{(m,)}`: the singleton
        `{(m, 0): e_{−m}}` for `m ≤ 0` (the `u₋^{−m}` monopole), `{}` off the
        positive cone (`m > 0`)."""
        (m,) = p
        if m > 0:
            return {}
        return {(m, 0): _eq_coeff(-m)}

    # ----- lift coordinate (trivial: the pentagon is unflavoured) --------

    def r_label_decompose(self, label):
        """Trivial lift — the pentagon carries no continuous flavour, so the
        section is the label and the R-label is `()`.  (Overrides the generic
        delegation-through-`apex`, which would needlessly round-trip the UV label
        through the SQED₁ chart and canonicalise non-canonical inputs like
        `(0,0,1) → (1,1,0)`.)"""
        return tuple(label), ()

    def r_label_compose(self, section, r_basis_label):
        return tuple(section)

    def from_ir_image(self, x_ir: Element) -> Element:
        """Cone-minimal apex peel back to pentagon labels.  Generic
        height-ordered peel (as `RGKAlgebra.from_ir_image`), but **tolerant** of
        SQED₁ charges that fall outside the five pentagon chord-cones: such a
        label cannot be an apex of any `RG(L_c)`, so it is skipped (it must
        cancel against the contributions of the genuine apices).  This guard is
        what lets the same peel drive `multiply` over the non-torus SQED₁."""
        g = self.grading()
        x = {l: c for l, c in x_ir.terms.items() if not c.is_zero()}
        out: dict = {}
        # Off-cone labels are not silently discarded: their running residual
        # is accumulated in `skipped` and must cancel to zero by the end —
        # the "it must cancel" claim is verified, not assumed.  Guard
        # exhaustion raises (a partial decomposition is never returned).
        skipped: dict = {}
        guard = 0
        while x:
            guard += 1
            if guard > 100000:
                raise ValueError(
                    f"from_ir_image: apex peel did not terminate "
                    f"(residual labels {sorted(x)[:5]}…) — refusing to "
                    f"return a partial decomposition."
                )
            delta = min(x, key=lambda l: g.height_of(l))
            try:
                c = self._apex_inverse(delta)
            except ValueError:
                sv = x.pop(delta)
                skipped[delta] = (skipped[delta] + sv) if delta in skipped \
                    else sv
                continue
            coeff = x[delta]                       # apex coeff of RG(c) is 1
            out[c] = (out[c] + coeff) if c in out else coeff
            for d, cc in self.RG(c).terms.items():
                term = coeff * cc
                if term.is_zero():
                    continue
                if d in skipped:
                    nv = skipped[d] - term
                    skipped[d] = nv
                    continue
                nv = (x[d] - term) if d in x else -term
                if nv.is_zero():
                    x.pop(d, None)
                else:
                    x[d] = nv
        bad = {l: c for l, c in skipped.items() if not c.is_zero()}
        if bad:
            raise ValueError(
                f"from_ir_image: off-cone content did not cancel "
                f"({ {l: str(c) for l, c in list(bad.items())[:4]} }) — the "
                f"input is not in the image of the pentagon RG peel."
            )
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    def __repr__(self) -> str:
        return "PentagonSquareSampleRGKAlgebra()"
