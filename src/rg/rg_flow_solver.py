"""
rg_flow_solver.py
=================

`solve_RG` — the RG-flow analogue of the BPS F-solver.  Given an IR
`KAlgebra` (canonical basis), an RG generator `S_RG` (dict
`IR_label → HabiroElement`, leading term the IR identity), and a target
IR canonical label `a`, find the IR element `RG(a)` with

    RG(a) · S_RG  =  L_a  +  O(𝖖)      (in the IR canonical basis)

i.e. `RG(a)` is the unique canonical-basis-supported solution whose
`S_RG`-dressing leads with `L_a`.  This is the directional-RGKAlgebra
primitive that was left unimplemented in `RGKAlgebra` (the general
positive-cone analogue being unclear); here the IR is a finite AD
`ConeKAlgebra`/`BPSKAlgebra` so the cone is explicit.

Two truncation boundaries, **handled carefully**
------------------------------------------------
1. **`S_RG` cutoff `C`.**  `S_RG` is only known up to dropped-node
   multiplicity `C`.  A product `RG·S_RG` term at dropped-extent `e`
   is reliable only for `e < C`.
2. **Finiteness / dropped-node cone `M`.**  The true `RG(a)` has
   *bounded* dropped-node multiplicity.  Without a cap the `[n]_𝖖`
   peel chases truncation-boundary `q⁰` artifacts and **diverges**
   (coefficients blow up).  We peel only labels with dropped-node
   extent `≤ M`, and require `M ≪ C` so the peel never touches the
   boundary.  `M` is grown until `RG` stabilises (the canonical
   solution is finite), staying well under `C`.

Coefficients of `RG(a)` are integral `[n]_𝖖`-number combinations
(`HabiroElement`s built only from the `[n]_𝖖` peel) — never arbitrary
Laurent polynomials.  Non-negativity is not imposed; it falls out.

Validated against the BPS single-node-RG `RG` (ground truth) for the
odd AD family: hexagon `[A_1,A_3]`, octagon `[A_1,A_5]`, decagon
`[A_1,A_7]` — the last being the case where the BPS UV build itself is
too slow, so the solver (on the cheap IR + closed-form `S_RG`) is the
only tractable route.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from habiro import HabiroElement
from laurent_poly import LaurentPoly
from rgkalgebra import _multiply_habiro_dicts


def _q_number(n: int) -> LaurentPoly:
    """`[n]_𝖖 = q^{-(n-1)} + q^{-(n-3)} + … + q^{n-1}`."""
    return LaurentPoly({-(n - 1) + 2 * i: 1 for i in range(n)})


def _peel_nonpositive(nonpos: dict) -> LaurentPoly:
    """Convert a non-positive q-order tail `{q_exp ≤ 0: int}` into the
    `[n]_𝖖` correction that cancels it (the BPS F-solver peel rule):
    lowest order `k` with coeff `c` → `−c·[1−k]_𝖖`, propagating the
    q-number's own tail `nonpos[k+2i] −= c`.  Returns the correction as
    a `LaurentPoly` in the `[n]_𝖖` span."""
    nonpos = dict(nonpos)
    f_corr: dict[int, int] = {}
    while any(v for v in nonpos.values()):
        k = min(e for e, v in nonpos.items() if v != 0)
        c = nonpos[k]
        nonpos[k] = 0
        if c == 0:
            continue
        n = 1 - k
        f_corr[n] = f_corr.get(n, 0) - c
        for i in range(1, n):
            ex = k + 2 * i
            if ex > 0:
                break
            nonpos[ex] = nonpos.get(ex, 0) - c
    out = LaurentPoly.zero()
    for n, co in f_corr.items():
        if co:
            out = out + _q_number(n) * co
    return out


def solve_RG(
    IR, S_RG, a_label, dropped_index=None, *,
    grading=None, M: int = 3, K: int = 8, max_iter: int = 200,
    s_rg_side: str = "right",
):
    """Solve `RG(a)·S_RG = L_a + O(𝖖)` (or, for `s_rg_side="left"`,
    `S_RG·W(a) = L_a + O(𝖖)`) in the IR canonical basis.

    `s_rg_side`        : `"right"` (default) solves for the unknown `U`
                         in `U·S_RG = L_a + O(𝖖)` (= `RG(a)`); `"left"`
                         solves `S_RG·U = L_a + O(𝖖)` (= the RG-twist
                         partner `ρ_IR⁻¹(RG(ρ_UV(a)))`).  Same `[n]_𝖖`
                         peel; only the product order differs (so neither
                         direction ever needs `S_RG⁻¹`).

    `IR`               : the IR `KAlgebra` (canonical multiply).
    `S_RG`             : `dict[IR_label, HabiroElement]`, leading term
                         `{IR.identity(): 1}`.
    `a_label`          : target IR canonical label.
    `dropped_index`    : (lattice IR) coordinate of the dropped node —
                         the direction `S_RG` extends in.  Equivalent to
                         `grading = lambda lbl: lbl[dropped_index]`.
    `grading`          : callable `IR_label -> int`, the **finiteness
                         grading** (e.g. the μ-flavour charge for an
                         `A1A2kKAlg ⊗ QT_μ` IR).  Peeling is capped at
                         grading `≤ M`.  Provide this *or* `dropped_index`.
    `M`                : grading cap (the finiteness cone); must be `≪`
                         the `S_RG` cutoff so the peel never touches the
                         truncation boundary.
    `K`                : q-order to which residuals are peeled.

    Returns `dict[IR_label, HabiroElement]` with `[n]_𝖖` coefficients.
    """
    if grading is None:
        if dropped_index is None:
            raise ValueError("solve_RG: provide grading or dropped_index")
        di = dropped_index
        grading = lambda lbl: lbl[di]
    if s_rg_side not in ("right", "left"):
        raise ValueError(f"s_rg_side must be 'right' or 'left', got {s_rg_side!r}")
    RG = {a_label: HabiroElement.one()}
    for _ in range(max_iter):
        prod = (_multiply_habiro_dicts(RG, S_RG, IR) if s_rg_side == "right"
                else _multiply_habiro_dicts(S_RG, RG, IR))
        peeled = False
        for lbl in set(prod) | set(RG):
            if grading(lbl) > M:                # finiteness cone bound
                continue
            h = prod.get(lbl, HabiroElement.zero())
            target = HabiroElement.one() if lbl == a_label else HabiroElement.zero()
            diff = h - target
            if diff.is_zero():
                continue
            nonpos = {
                e: c for e, c in diff.expand(K)._coeffs.items()
                if e <= 0 and c != 0
            }
            if not nonpos:
                continue
            corr = _peel_nonpositive(nonpos)
            RG[lbl] = RG.get(lbl, HabiroElement.zero()) + HabiroElement.from_laurent(corr)
            peeled = True
        if not peeled:
            break
    return {l: h for l, h in RG.items() if not h.is_zero()}


def verify_RG(IR, S_RG, RG, a_label, dropped_index=None, *, grading=None, M_safe: int, K: int = 6) -> bool:
    """Check `RG·S_RG = L_a + O(𝖖)` in the truncation-safe region:
    every IR label with grading `≤ M_safe` has product coefficient
    `δ_{·,a} + O(𝖖)` (no non-positive q-order beyond the leading 1 at
    `a`)."""
    if grading is None:
        di = dropped_index
        grading = lambda lbl: lbl[di]
    prod = _multiply_habiro_dicts(RG, S_RG, IR)
    for lbl, h in prod.items():
        if grading(lbl) > M_safe:
            continue
        target = HabiroElement.one() if lbl == a_label else HabiroElement.zero()
        diff = h - target
        if diff.is_zero():
            continue
        nonpos = [c for e, c in diff.expand(K)._coeffs.items() if e <= 0 and c != 0]
        if nonpos:
            return False
    return True
