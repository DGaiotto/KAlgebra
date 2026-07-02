"""`RGKAlgebra` — a `KAlgebra` presented via an RG flow to a graded auxiliary.

`RGKAlgebra(KAlgebra)` is the canonical "RG presentation" of a K_𝖖
algebra.  A concrete `RGKAlgebra` is *defined* by wrapping an
auxiliary (IR) `KAlgebra` together with RG-flow data, and **inherits the
full `KAlgebra` API generically** from that data.

The defining data (subclass-supplied)
--------------------------------------
  * `auxiliary()` — the IR `KAlgebra` (the flow target).
  * `grading()` — a `Grading` of the auxiliary (`grading.py`): a charge
    `deg(L) ∈ Γ_RG` on each auxiliary label, additive under `multiply`,
    plus an integral height functional `h` (a linearized proxy for
    `Im Z_γ`, the physical central charge).  The
    quantum torus is *not* required — any graded auxiliary with a
    pointed cone works (`BPSKAlgebra`'s aux is the quantum torus, the
    `deg = id` corner).
  * The spectrum generator `S_RG` — known through **two independent
    contracts**, because the relation between a charge and the q-order(s)
    at which it appears is presentation-specific (knowable only to the
    subclass) and in general not monotone in the grading height:
      - `_s_rg_component(p)` — the *exact* `Γ_RG`-graded component
        `[S_RG]_p` as `{aux label: HabiroElement}`: never q-truncated,
        finite (height-positivity axiom), `{}` off the cone.  The
        cutoff-free "knowing `S_RG`" oracle; windowing is a caller concern
        (by grading height for `factor_through`).  *Soft contract* — a
        documented default that raises until a subclass provides it.
      - `rg_generator(cutoff)` — `S_RG` windowed to *q-order* ≤ cutoff,
        `{aux label: HabiroElement}`, `[S_RG]_0 = 1_B`.  `S_RG` *knowledge*
        (the windowed-`solve_rg` fallback consumes it; the generic `trace`
        assembles its object from it).  *Not* a safe trace handle on its own
        — see `trace` / `rg_times_s_rg`: the trace's object is the product
        `RG(a)·S_RG` to a q-order, across which q-order is non-additive.
  * `apex(a)` — the tropical identification of a UV label with its apex
    IR label (default: identity).  **Flow-specific** — reconcile with
    other presentations of the same abstract algebra via `KAlgebraIso`.

The generic `KAlgebra` API (all derived, all overridable)
--------------------------------------------------------
  * `RG(a)` — the RG image `RG(L_a) ∈ auxiliary`, with bar-invariant
    `Z[q^±]` coefficients (non-negative `[n]_q` combinations, as `F_γ`),
    solved from the **discovery relation** `RG(a)·S_RG = L_{apex(a)} +
    O(q)` by `graded_rg_solver.solve_rg` (cached).
  * `multiply(a, b) = from_ir_image(RG(a)·_aux RG(b))` — cone-minimal
    apex peel of the IR product back into `{RG(L_c)}`.
  * `rho_inverse(a)` — `ρ_UV⁻¹(a)`, derived from `solve_rg`'s mirror
    constraint `ρ_IR(S_RG)·RG(a) = L_upper + O(q)`.
  * `rho(a)` — forward `ρ_UV`, derived from the alternative RG map `tRG`
    (`solve_trg` = `solve_rg` in the opposite algebra; its mirror yields
    `ρ_UV`).  `tRG(a)` (`RG(a)·S_RG = S_RG·tRG(a)`) is a first-class
    method, with `RG ∘ ρ_UV = ρ_IR ∘ tRG` (verifier
    `verify_rg_trg_intertwine`).  Subclasses with a combinatorial `σ`
    (e.g. `BPSKAlgebra`) may override.
  * `trace(a, K)` / `inner_product(a, b, K)` — the trace pairing, computed
    by the **bilinear expansion** (`_inner_product_uncached`):

        I_{a,b} = Σ_{c,d} [RG(a)·S_RG]_c · [RG(b)·S_RG]_d · I^aux_{c,d},
        Tr(a)   = I(1, a)  (with `RG(1)·S_RG = S_RG`),

    where `[x]_c` is the coefficient of auxiliary basis label `c` in `x`, and
    `I^aux_{c,d} = aux.inner_product(c, d)` is a *well-defined* single-basis
    pairing.  The naive `Tr_aux(ρ(S_RG)·…·S_RG)` is **not defined** — `ρ(S_RG)`
    is a formal sum over the negative grading cone and `S_RG` over the positive
    cone, so their product is not an auxiliary element; bilinearity moves `ρ`
    and the trace onto the well-defined `I^aux_{c,d}`, leaving a finite sum at
    each `q`-order.  The per-charge components are the **FS object**
    `RG(a)·S_RG` (BPS's `F·S` shorthand: `F = RG`, `S = S_RG`), supplied to
    q-order `K` by the safe overridable primitive `rg_times_s_rg` (exact per-η
    when `_s_rg_component` + flat-int aux labels are available; else a windowed
    heuristic).  A subclass with a sharper trace overrides `_trace_uncached` /
    `_inner_product_uncached` (e.g. `BPSKAlgebra._schur_index`).
  * `coefficient_ring` / `identity` / `_label_section_decompose` — from
    the auxiliary.

(Glossary: every `_fs_*` member and the phrase "FS object" denote the
`RG(a)·S_RG` object — BPS's `F·S` shorthand carried into the generic tier.)

A subclass with a fast closed form (e.g. `BPSKAlgebra`, `RG = F`, with
its own `multiply`/`rho`/`trace`) overrides the relevant primitives; the
generic defaults are then the contract's reference implementation.  This
is the basis on which concrete RG-flow theories are adapted.

Composition.  Two `RGKAlgebra`s `UV → MS` and `MS → IR` with matching
endpoints compose into a single `RGKAlgebra UV → IR` via `ComposedRG`
(`then`): `RG^UV_IR = RG^MS_IR ∘ RG^UV_MS` and
`S^UV_IR = RG^MS_IR(S^UV_MS) · S^MS_IR`.

Factorization (the inverse of composition).  Given a known `UV → IR` flow and an
`MS → IR` flow over a lattice-compatible auxiliary, `factor_through`
recovers the `UV → MS` flow (`ExtractedRG`): `S^UV_MS` is solved from
`S^UV_IR = RG^MS_IR(S^UV_MS) · S^MS_IR` (invert `S^MS_IR`, multiply by
`S^UV_IR`, pull the result back through `RG^MS_IR`, truncate), and
`RG^UV_MS = RG^MS_IR⁻¹ ∘ RG^UV_IR` is the embedding `A^UV ↪ A^MS`.  The
grading lattices form a short exact sequence
`0 → Γ^MS_IR → Γ^UV_IR → Γ^UV_MS → 0`, so the recovered flow's grading is
the quotient `Γ^UV_MS = Γ^UV_IR / Γ^MS_IR`.  `rg_flow.factor_through_
subquiver` is the BPS driving case: a theory whose spec factors as
`(outside factors)·(subquiver spec)`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element
from zplus_ring import RLaurent, RPowerSeries

if TYPE_CHECKING:
    from habiro import HabiroElement


Label = tuple
Charge = tuple   # a Γ_RG grading charge (see grading.Grading)


# ---------------------------------------------------------------------------
# Helpers for ComposedRG.rg_generator: `HabiroElement`-coefficient arithmetic
# in an auxiliary.  `HabiroElement` (habiro.py) is exact arithmetic in the
# localization Z[q^±][(1−q^{2k})^{-1}, k ≥ 1]; the class name is historical.
# ---------------------------------------------------------------------------


def _laurentpoly_to_habiro(c) -> "HabiroElement":
    """Lift a `LaurentPoly` (a Z-form `Element` coefficient) to a
    `HabiroElement` for use inside the localized-ring arithmetic helpers.

    `Element` coefficients are `LaurentPoly` over Z[q^±], not `RLaurent`
    over `R` — hence the lift starts from `LaurentPoly`.
    """
    from habiro import HabiroElement
    from laurent_poly import LaurentPoly
    return HabiroElement.from_laurent(LaurentPoly(dict(c._coeffs)))


# Backward-compat alias for any internal code that still uses the
# old name; both helpers do the same thing now (Z-form Element).
_rlaurent_to_habiro = _laurentpoly_to_habiro


def _apply_rg_to_habiro_dict(
    rg: "RGKAlgebra",
    mid_dict: dict[Label, "HabiroElement"],
) -> dict[Label, "HabiroElement"]:
    """Apply `rg.RG` to a `HabiroElement`-coefficient dict at `rg`'s source
    side, accumulating in `rg.auxiliary()`'s labels with `HabiroElement`
    coefficients.

    `mid_dict[a] = h_a` (a `HabiroElement`), labels in `rg.starting_algebra()`.
    Returns `dict[label_aux, HabiroElement]` with each entry
        `out[b] = Σ_a h_a · _lift(c^a_b)`
    where `rg.RG(a) = Σ_b c^a_b L_b^aux`.
    """
    from habiro import HabiroElement
    out: dict[Label, "HabiroElement"] = {}
    for a, h_a in mid_dict.items():
        rg_a = rg.RG(a)
        for b, c_b in rg_a.terms.items():
            if c_b.is_zero():
                continue
            c_b_h = _rlaurent_to_habiro(c_b)
            term = h_a * c_b_h
            if term.is_zero():
                continue
            out[b] = out[b] + term if b in out else term
    return {k: v for k, v in out.items() if not v.is_zero()}


def _multiply_habiro_dicts(
    left: dict[Label, "HabiroElement"],
    right: dict[Label, "HabiroElement"],
    aux: KAlgebra,
) -> dict[Label, "HabiroElement"]:
    """Multiply two `HabiroElement`-coefficient dicts in `aux`'s K-algebra:
        left = Σ_a h_a L_a,  right = Σ_b h'_b L_b
        result[c] = Σ_{a, b} h_a · h'_b · _lift(C^c_{ab}(q))
    where `C^c_{ab}` is `aux.multiply(a, b)`'s coefficient at `L_c`."""
    from habiro import HabiroElement
    out: dict[Label, "HabiroElement"] = {}
    for a, h_a in left.items():
        for b, h_b in right.items():
            prod = aux.multiply(a, b)
            for c, c_ab in prod.terms.items():
                if c_ab.is_zero():
                    continue
                c_ab_h = _rlaurent_to_habiro(c_ab)
                term = h_a * h_b * c_ab_h
                if term.is_zero():
                    continue
                out[c] = out[c] + term if c in out else term
    return {k: v for k, v in out.items() if not v.is_zero()}


# ---------------------------------------------------------------------------
# Helpers for factor_through (extract): invert a spectrum generator, pull a
# `HabiroElement`-coefficient element back through an RG map, truncate by
# q-order.
# (Validated against the known S^UV_MS on a 4-node node-deletion chain.)
# ---------------------------------------------------------------------------


def _invert_habiro_spectrum(
    s_dict: dict[Label, "HabiroElement"],
    aux: KAlgebra,
    cutoff: int,
) -> dict[Label, "HabiroElement"]:
    """Inverse of a spectrum-generator dict `S` (`[S]_0 = 1_B`) in `aux`.

    Uses the q-adically convergent geometric series
    `S^{-1} = Σ_{n≥0} (1_B − S)^n`: every non-identity `S`-term has leading
    q-order ≥ 1, so `(1_B − S)^n` starts at q-order ≥ n and the sum to
    `n = cutoff+1` is exact through q-order ≤ cutoff.  Coefficients are exact
    `HabiroElement`s; the *final* extract product is truncated to leading
    q-order ≤ K by the caller (compute-exact-truncate-last).  Used by
    `factor_through` to form `T = S^UV_IR · (S^MS_IR)^{-1}`.
    """
    from habiro import HabiroElement
    e = aux.identity()
    one_minus_s = {l: -c for l, c in s_dict.items() if l != e}
    inv: dict[Label, "HabiroElement"] = {e: HabiroElement.one()}
    term: dict[Label, "HabiroElement"] = {e: HabiroElement.one()}
    for _ in range(cutoff + 2):
        term = _multiply_habiro_dicts(term, one_minus_s, aux)
        if not term:
            break
        for l, c in term.items():
            inv[l] = (inv[l] + c) if l in inv else c
    return {l: c for l, c in inv.items() if not c.is_zero()}


def _habiro_from_ir_image(
    x_dict: dict[Label, "HabiroElement"],
    flow: "RGKAlgebra",
    height_grading: "Grading",
) -> dict[Label, "HabiroElement"]:
    """Pull a `HabiroElement`-coefficient auxiliary element `x_dict` back
    through `flow.RG`: express it as `Σ_c h_c · RG^flow(L_c)` and return
    `{flow source label: h_c}`.

    Cone-minimal apex peel ordered by `height_grading.height_of`.  The
    height **must** be the UV→IR grading (total on every appearing charge);
    a sub-flow's own grading is only partial (it cannot score charges
    outside its integrated-out directions — the very charges this pullback
    walks).  The `HabiroElement` generalization of
    `RGKAlgebra.from_ir_image`; the step-3 pullback of `factor_through`.
    """
    x = {l: c for l, c in x_dict.items() if not c.is_zero()}
    out: dict[Label, "HabiroElement"] = {}
    while x:
        delta = min(x, key=lambda l: height_grading.height_of(l))
        c = flow._apex_inverse(delta)
        coeff = x[delta]                            # apex coeff of RG(c) is 1
        out[c] = (out[c] + coeff) if c in out else coeff
        for d, lp in flow.RG(c).terms.items():
            term = coeff * _laurentpoly_to_habiro(lp)
            if term.is_zero():
                continue
            x[d] = (x[d] - term) if d in x else -term
            if x[d].is_zero():
                del x[d]
    return {l: c for l, c in out.items() if not c.is_zero()}


def _truncate_habiro_dict_by_qorder(
    d: dict[Label, "HabiroElement"], K: int,
) -> dict[Label, "HabiroElement"]:
    """Keep only labels whose `HabiroElement` coefficient has leading q-order ≤ K
    (the `rg_generator(K)` cutoff).  Drops the beyond-cutoff residue a
    truncated inverse leaves behind: the leading q-order equals the
    numerator's lowest exponent (every denominator factor `(1−q^{2k})` has
    constant term 1)."""
    out: dict[Label, "HabiroElement"] = {}
    for l, c in d.items():
        if c.is_zero():
            continue
        if min(c.numerator._coeffs) <= K:
            out[l] = c
    return out


# ---------------------------------------------------------------------------
# RGKAlgebra
# ---------------------------------------------------------------------------


def _label_diff(x, s):
    """Nested-aware label difference `x − s`: ints subtract; same-length
    int-tuples (and pair-shaped labels — tuples of ints/int-tuples, the
    `add_flavour` auxiliary shape) subtract slot-by-slot.  Returns `None`
    when the difference is undefined (opaque slots, shape mismatch)."""
    if isinstance(x, bool) or isinstance(s, bool):
        return None
    if isinstance(x, int) and isinstance(s, int):
        return x - s
    if isinstance(x, tuple) and isinstance(s, tuple) and len(x) == len(s):
        out = tuple(_label_diff(xi, si) for xi, si in zip(x, s))
        return None if any(o is None for o in out) else out
    return None


def _canon_shift(t):
    """Canonicalize a (central) flavour shift: collapse a non-empty all-zero
    (nested) tuple to the empty tuple `()` — the **arity-neutral** "no
    translation" marker.

    A central shift's chord slot is a zero chord-translation whose arity is
    incidental (it mirrors the label it came from: `()` for a 0-chord label,
    `((0,0,0),)` for a 1-chord label, …).  Collapsing it to `()` lets the
    same shift apply to output labels of *any* chord-arity (`_label_add`
    treats `()` as the additive identity, keeping the label's own slot).
    Without this the central (e.g. `μ`) part would be silently dropped
    whenever the operand arities differed (`multiply(1, L_{(c, μ)})` would
    lose `μ`)."""
    if isinstance(t, tuple):
        if t and _label_is_zero(t):
            return ()
        return tuple(_canon_shift(e) for e in t)
    return t


def _label_add(x, y):
    """Nested-aware label sum.  Ints add; equal-length tuples add slot-by-slot.

    The empty tuple `()` is the **additive identity** at any slot
    (`_label_add(c, ()) = c`), so an arity-neutral central shift (chord slot
    canonicalised to `()` by `_canon_shift`) applied to a label keeps that
    label's own slot.  Non-empty tuples of differing length remain a shape
    error (`None`), so flat-vector arithmetic is unchanged."""
    if isinstance(x, bool) or isinstance(y, bool):
        return None
    if isinstance(x, int) and isinstance(y, int):
        return x + y
    if isinstance(x, tuple) and isinstance(y, tuple):
        if x == ():
            return y
        if y == ():
            return x
        if len(x) == len(y):
            out = tuple(_label_add(xi, yi) for xi, yi in zip(x, y))
            return None if any(o is None for o in out) else out
    return None


def _label_is_zero(t) -> bool:
    """True iff every (nested) entry of the shift is zero."""
    if isinstance(t, int):
        return t == 0
    return all(_label_is_zero(x) for x in t)


def _label_translate(lbl, total):
    """Translate a label by a (nested) flavour shift; shapes must match."""
    out = _label_add(tuple(lbl), tuple(total))
    if out is None:
        raise ValueError(
            f"flavour translation undefined: label {lbl!r} + shift {total!r}")
    return out


class RGKAlgebra(KAlgebra):
    """A `KAlgebra` presented via an **RG flow to a graded auxiliary**.

    This is the canonical "RG presentation" of a K_𝖖 algebra.
    A concrete `RGKAlgebra` is *defined* by wrapping an auxiliary (IR)
    `KAlgebra` together with RG-flow data, and **inherits the full
    `KAlgebra` API generically** from that data — it does not
    re-implement `multiply`/`rho`/`trace` per realisation.

    Data a subclass supplies
    ------------------------
    * `auxiliary()` — the IR `KAlgebra` (the target of the flow).
    * `grading()` — a `Grading` of the auxiliary: a charge `deg(L) ∈ Γ_RG`
      on each auxiliary label (additive under `multiply`) plus an integral
      height functional `h` (a linearized proxy for `Im Z_γ`, the physical
      central charge).  See `grading.py`.
    * `rg_generator(cutoff)` — the spectrum generator `S_RG` as a dict
      `{aux label: HabiroElement}`, with `[S_RG]_0 = 1_B`.
    * `apex(a)` — the tropical identification of a UV canonical label `a`
      with its apex IR label (default: identity).  **This UV labelling is
      specific to the RG flow** and must be reconciled with other
      presentations of the same abstract algebra via a `KAlgebraIso`.

    Full `KAlgebra` API derived generically (all overridable)
    --------------------------------------------------------
    * `RG(a)` — solved from the discovery relation `RG(a)·S_RG = a + O(q)`
      by `graded_rg_solver.solve_rg` (cached).
    * `multiply(a, b)` — `from_ir_image(RG(a)·_aux RG(b))`.
    * `rho_inverse(a)` — derived from the same `solve_rg` (its mirror
      constraint yields `ρ_UV⁻¹(a)`).
    * `rho(a)` — forward `ρ_UV`, derived from the **alternative RG map**
      `tRG` (`= solve_rg` in the opposite algebra; its mirror yields
      `ρ_UV`).  `tRG(a)` (defined by `RG(a)·S_RG = S_RG·tRG(a)`) is
      exposed in its own right, with `RG ∘ ρ_UV = ρ_IR ∘ tRG` as the
      intertwining (verifier `verify_rg_trg_intertwine`).  MAY be
      overridden by subclasses with a combinatorial `σ` (e.g.
      `BPSKAlgebra`).
    * `trace(a, K)` / `inner_product(a, b, K)` — the bilinear trace
      pairing (see `_inner_product_uncached`):

          I_{a,b} = Σ_{c,d} [RG(a)·S_RG]_c · [RG(b)·S_RG]_d · I^aux_{c,d},
          Tr(a)   = I(1, a)  (with `RG(1)·S_RG = S_RG`).
    * `coefficient_ring` / `identity` / `_label_section_decompose` — from
      the auxiliary.

    A subclass that has a fast closed form (e.g. `BPSKAlgebra`, whose
    `RG = F`, with its own `multiply`/`rho`/`trace`) overrides the
    relevant primitives; the generic defaults above then act as the
    contract's reference implementation.
    """

    # ----- defining data (subclass-supplied) ------------------------------

    @abstractmethod
    def auxiliary(self) -> KAlgebra:
        """The auxiliary (IR) K-algebra targeted by the RG flow."""

    def grading(self) -> "Grading":
        """The `Γ_RG`-grading of the auxiliary (charge + height) driving
        the generic `solve_rg` path.  Subclasses that use the generic
        `RG`/`rho_inverse`/`trace` defaults must supply this; subclasses
        that override those primitives (e.g. `BPSKAlgebra`) need not."""
        raise NotImplementedError(
            f"{type(self).__name__}.grading() is not implemented; supply a "
            f"Grading to use the generic RGKAlgebra API, or override the "
            f"primitives directly."
        )

    def apex(self, a: Label):
        """Tropical map: the auxiliary apex label identified with UV label
        `a`.  Default: identity (UV labels are IR apex labels).  This
        labelling is RG-flow-specific."""
        return tuple(a)

    def _rg_cutoff(self) -> int:
        """`S_RG` q-order window for the windowed-`solve_rg` fallback **and**
        the fixed window the generic `trace` uses to assemble the FS object
        `RG(a)·S_RG` (`rg_times_s_rg`).  Override to tune; default is modest.

        *Trace caveat:* a fixed window cannot certify `RG(a)·S_RG` to a given
        q-order (q-order is non-additive across the product — see `trace` /
        `rg_times_s_rg`).  It is a heuristic, sound in the tested small-`K`
        regime; a flow wanting certified traces overrides `rg_times_s_rg`
        (e.g. adaptive window + two-cutoff stability) rather than relying on
        this constant."""
        return 12

    def _rg_cutoff_delta(self) -> int:
        """Increment for the *second* (higher) `S_RG` cutoff used by
        `solve_rg`'s cutoff-stability completion certificate (the deep-edge
        truncation artifacts must move between cutoff and cutoff+Δ so they
        are filtered out).  Δ=2 suffices (validated against the oracle); a
        single-cutoff bump already shifts the deep-edge artifacts."""
        return 2

    def _rg_generator_cached(self, cutoff):
        """`rg_generator(cutoff)` memoised by cutoff.  `S_RG` does not depend
        on the queried label, so the generic `RG`/`tRG` solves share one
        copy per cutoff (the solve_rg cutoff-stability certificate needs two
        cutoffs, and a suite touches many labels — recomputing `S_RG` per
        solve was the dominant cost)."""
        cache = self.__dict__.setdefault("_rg_gen_cache", {})
        s = cache.get(cutoff)
        if s is None:
            s = self.rg_generator(cutoff)
            cache[cutoff] = s
        return s

    def _exact_window_available(self) -> bool:
        """Whether the **exact per-charge** solve path is *applicable*: the
        grading carries `cone_gens` and `_s_rg_component` is implemented (the
        per-charge oracle `[S_RG]_γ` that `solve_rg_exact` consumes)."""
        if self.grading().cone_gens is None:
            return False
        try:
            self._s_rg_component(self.grading().zero_charge())
            return True
        except NotImplementedError:
            return False

    def _use_exact_window(self) -> bool:
        """Whether `RG`/`tRG` use the **exact per-charge** solve path
        (`graded_rg_solver.solve_rg_exact`) instead of the q-order
        `rg_generator` + cutoff-stability windowed path (`solve_rg`).

        Default: **on whenever the exact path is applicable**
        (`_exact_window_available()` — the grading carries `cone_gens` and
        `_s_rg_component` is implemented).  `solve_rg_exact` uses the per-charge
        oracle `_s_rg_component(γ)` to compute every residual *exactly* (a
        finite sum over the current `RG(a)` terms — no `S_RG` window and no
        q-truncation), so beyond the true top the residual is exactly `O(q)`
        and the cone-walk terminates on its own (no mirror / negative-cone
        certificate needed).  This is both *correct* and *more robust* than the
        q-order cutoff: the q-order window can silently truncate `RG` when
        `_rg_cutoff()` is too small for the flow (e.g. A1A2k `k ≥ 3` truncates
        `RG((2,2))` to its head), whereas the exact path is cutoff-free.
        Validated: matches the genuine `SubquiverRG` F-decomposition oracle
        10/10 on A_4 roots (terms + `ρ_UV⁻¹`); `verify_rg_unital` /
        `verify_rg_multiplicative` (81/81) and associativity (64/64) on A1A2k
        at k=1..4; k-independent RG images; k=1 ≡ BPS(A_2).

        Subclasses without `cone_gens` / `_s_rg_component` (no exact oracle)
        fall back to the q-order path automatically.  A subclass MAY override
        to force the q-order path (`return False`)."""
        return self._exact_window_available()

    def starting_algebra(self) -> KAlgebra:
        """The underlying K-algebra this `RGKAlgebra` represents (= the
        source of the RG flow).  Default: `self`.  Wrappers whose KAlgebra
        ops delegate to a separate UV instance override this so that
        composition can validate endpoints by Python identity."""
        return self

    @abstractmethod
    def rg_generator(self, cutoff) -> dict[Label, "HabiroElement"]:
        """`S_RG = 1 + Σ s_a L_a^aux` windowed to **q-order ≤ cutoff**,
        keyed by auxiliary labels with `HabiroElement` coefficients
        (`[S_RG]_0 = 1_B`); the form the generic `trace` / `solve_rg`
        consume.

        One of the two `S_RG` contracts (the other is `_s_rg_component`),
        and **not derivable** from the other: the relation between a charge
        and the q-order(s) at which it appears is presentation-specific —
        knowable only to the subclass — and in general not monotone in the
        grading height, so the q-order window is neither a union of
        grading-height components nor a truncation of them.  (BPS:
        q-order = Nahm shift.)"""

    def _s_rg_component(self, p: "Charge") -> dict[Label, "HabiroElement"]:
        """The exact `Γ_RG`-graded component of the spectrum generator:
        `[S_RG]_p`, returned as the dictionary
        `{aux label (deg = p): exact HabiroElement}`.  This is the
        contractual meaning of "knowing `S_RG`" — a per-charge exact oracle.

        Contract (RGKAlgebra primitive):

        * **EXACT** — full localized-ring coefficients, never q-truncated.
        * **FINITE** — `[S_RG]_p` has finitely many labels.  This is a
          structural *axiom*: a valid RG grading is height-positive
          (`h(γ) ≥ 1` on every appearing charge), so a degree-`p` term is a
          product of `≤ h(p)` spec factors; a grading admitting an infinite
          component is not a valid RG grading.
        * **VANISHES off the cone** — returns `{}` when `p` is not in the
          positive cone.

        Cutoff-free: windowing `S_RG` is a *caller* concern — by q-order for
        `trace` / `solve_rg` (use `rg_generator`), by grading height for the
        extract (`factor_through`: union of `_s_rg_component(p)` over
        `{p : h(p) ≤ K}`).  The two `S_RG` contracts are independent (see
        `rg_generator`).  For a `deg = id` / quantum-torus auxiliary the
        component is one-dimensional, i.e. the singleton `{p: s_p}`.

        **Soft contract**: declared + documented here so subclasses know to
        provide it; the default raises rather than guessing.  (Tightenable
        to `@abstractmethod` once every subclass implements it.)
        """
        raise NotImplementedError(
            f"{type(self).__name__}._s_rg_component(p) is not implemented; "
            f"supply the exact graded component [S_RG]_p (finite Habiro dict, "
            f"empty off-cone) to use the per-component S_RG oracle."
        )

    def RG(self, a) -> Element:
        """`RG(L_a) = RG_a` ∈ auxiliary, with bar-invariant `Z[q^±]`
        coefficients (non-negative `[n]_q` combinations, as for `F_γ`).

        Generic default: solve the discovery relation
        `RG(a)·S_RG = L_{apex(a)} + O(q)` via `graded_rg_solver.solve_rg`.

        **Cached as 𝖖-numbers.**  `RG(a)`'s coefficients are palindromic,
        so each is stored canonically as a `QNumberPoly` (a `Z`-combination
        of `[n]_q`), matching `BPSKAlgebra._F_cache` — the same cache
        representation, which is what lets BPS's F-cache and this RG-cache
        unify.  The `Z[q^±]` `Element` view is rebuilt on demand.

        Subclasses with a closed form (e.g. `BPSKAlgebra`, `RG = F`)
        override this."""
        cache = self.__dict__.setdefault("_rg_cache", {})  # {label: {aux label: QNumberPoly}}
        key = tuple(a)
        if key in cache:
            return self._rg_element_from_qn(cache[key])
        from graded_rg_solver import solve_rg, solve_rg_exact
        from q_number_poly import QNumberPoly
        if self._use_exact_window():
            rg, rho_uv_inv = solve_rg_exact(
                self.auxiliary(), self.grading(),
                self._s_rg_component, self.apex(a))
        else:
            cut = self._rg_cutoff()
            rg, rho_uv_inv = solve_rg(
                self.auxiliary(), self.grading(),
                self._rg_generator_cached(cut), self.apex(a),
                s_rg_hi=self._rg_generator_cached(cut + self._rg_cutoff_delta()),
            )
        cache[key] = {
            lbl: QNumberPoly.from_palindromic_laurent(c)
            for lbl, c in rg.terms.items() if not c.is_zero()
        }
        self.__dict__.setdefault("_rho_inv_cache", {})[key] = rho_uv_inv
        return rg

    @staticmethod
    def _rg_element_from_qn(qn: dict) -> Element:
        """Build the `Z[q^±]` `Element` view from a `{aux label:
        QNumberPoly}` 𝖖-number cache entry."""
        return Element({lbl: q.to_laurent() for lbl, q in qn.items()})

    def _apex_inverse(self, ir_label: Label) -> Label:
        """Inverse tropical map: the UV label whose apex is `ir_label`
        (inverse of `apex`).  Default: identity.  Override alongside a
        non-trivial `apex`."""
        return tuple(ir_label)

    def from_ir_image(self, x_ir: Element) -> Element:
        """Recognise the UV `Element` whose `RG`-image is `x_ir`.

        Generic **cone-minimal apex peel** (generalising the BPS
        `_decompose_qt_in_ir_basis`): while `x_ir` is non-zero, take the
        lowest-`grading`-height support label `δ` (the apex); it is the
        apex of `RG(L_c)` for `c = apex⁻¹(δ)`, whose apex coefficient is
        `1`, so the coefficient of `x_ir` at `δ` is the `L_c`-component.
        Subtract `coeff · RG(L_c)` and repeat.  This inverts
        `RG_element` on its image — the generic `multiply` uses it.

        Requires `grading()`.  Subclasses with a closed-form `multiply`
        (e.g. `BPSKAlgebra`) bypass this."""
        g = self.grading()
        # Spec-free cone-degree cap.  In spec-free mode `S` is built only to
        # cone-degree ≤ cutoff, so `RG(c)` is reliable only inside that cone;
        # beyond it the canonical basis is incomplete and the apex peel can fail
        # to close — the residual marches off to ever-higher (and spuriously
        # negative) cone-degree instead of terminating (an unbounded loop on an
        # under-built `S`).  The canonical expansion is triangular in the cone,
        # so dropping out-of-cone terms truncates the result to the built degree
        # *without* corrupting the in-cone part — graceful degradation instead of
        # a hang.  Spec mode (and any realisation without `_sf_*`) sets neither
        # attribute, so `_within` is always True and this path is byte-identical.
        _cap = getattr(self, "_sf_max_degree", None)
        _capfn = getattr(self, "_sf_degree_fn", None)
        if _cap is not None and _capfn is not None:
            def _within(l, _cap=_cap, _capfn=_capfn):
                try:
                    return _capfn(tuple(l)) <= _cap
                except Exception:
                    return True
        else:
            def _within(l):
                return True
        x: dict[Label, "LaurentPoly"] = {
            l: c for l, c in x_ir.terms.items() if not c.is_zero() and _within(l)
        }
        out: dict[Label, "LaurentPoly"] = {}
        while x:
            delta = min(x, key=lambda l: g.height_of(l))
            c = self._apex_inverse(delta)
            coeff = x[delta]                       # apex coeff of RG(c) is 1
            out[c] = (out[c] + coeff) if c in out else coeff
            for d, cc in self.RG(c).terms.items():
                if not _within(d):
                    continue                       # out-of-cone: truncate
                term = coeff * cc
                if term.is_zero():
                    continue
                nv = (x[d] - term) if d in x else -term
                if nv.is_zero():
                    x.pop(d, None)
                else:
                    x[d] = nv
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    # ----- concrete K-algebra defaults via the abstracts above ------------

    def _section_split(self, label: Label) -> "tuple[Label, tuple]":
        """Split a label into `(section_rep, flavour_charge)` using the
        auxiliary's `_label_section_decompose`.  `flavour_charge =
        label − section_rep` is the central (flavour / `ker B`) shift;
        it is the zero vector when the auxiliary has no flavour
        (`section_rep == label`).  Keys the multiply cache by section so
        flavour-shifted products reuse one decomposition.

        Label arithmetic is **nested-aware** (`_label_diff` /
        `_label_translate`): flat integer vectors subtract elementwise,
        and pair-shaped labels — `(base_label, k_vec)`, the standard
        `add_flavour` matter-flow auxiliary — subtract slot-by-slot.
        Without nested awareness the pair shape would fall through to the
        non-flat branch, silently dropping central flavour shifts from
        every generic matter-flow product (`μ·L_a = L_a`).

        Returns `flav = None` **only** when the difference is genuinely
        undefined — labels with opaque (non-int, non-int-tuple) slots.
        A label that *is* its own section rep returns the **zero
        vector**, not `None`: conflating the two would make `multiply` skip
        the output flavour translation whenever either operand was a
        section rep, silently dropping the other operand's shift
        (`L_κ·L_a = L_a`)."""
        label = tuple(label)
        sec, _ = self.auxiliary()._label_section_decompose(label)
        sec = tuple(sec)
        return sec, _canon_shift(_label_diff(label, sec))

    def multiply(self, a: Label, b: Label) -> Element:
        """Default `multiply` for `RGKAlgebra`: the product of RG images
        in the IR, recognised back as a UV element via `from_ir_image`,

            multiply_UV(a, b) = from_ir_image(aux.multiply_elements(RG(a), RG(b))),

        with a **section-keyed decomposition cache** (`_multiply_cache`).
        Central flavour charges are central in the auxiliary, so
        `RG(a) = X_{flav(a)}·RG(sec(a))` and the structure constants of
        `(a, b)` are those of `(sec(a), sec(b))` with every output label
        translated by `flav(a)+flav(b)`.  We therefore cache the
        decomposition of `RG(sec a)·RG(sec b)` by `(sec a, sec b)` and
        reuse it across all flavour-shifted `(a, b)` — the generic
        analogue of `BPSKAlgebra._multiply_cache` (this is what gives the
        flavoured-multiply speedup; previously generic-only multiply
        recomputed the peel per shift).

        **Non-additive flavour.**  The section-shift cache assumes the flavour is
        a *central additive* charge (a U(1) `μ`, an `AbelianZPlusRing` weight),
        so `RG(a) = X_{flav(a)}·RG(sec a)` and the output is the section product
        translated by `flav(a)+flav(b)`.  When the flavour is **non-additive** —
        an SU(N) irrep label (a partition), which fuses by Clebsch–Gordan, not
        addition — `_section_split` returns `flav = None`; there the cache does
        not apply and we compute the **full** product `RG(a)·RG(b)`, letting the
        auxiliary's own (non-abelian) `multiply` perform the flavour fusion.
        (Before this guard the `flav is None` case silently returned the
        flavour-*stripped* section product — a latent bug, harmless only because
        every non-abelian-flavour flow overrode `multiply`; SQEDNf-type flows now
        use this generic path directly.)

        Subclasses MAY override for a closed-form shortcut."""
        sec_a, fa = self._section_split(a)
        sec_b, fb = self._section_split(b)
        if fa is None or fb is None:
            # non-additive flavour (e.g. SU(N) irreps): no section cache — do the
            # full product so aux.multiply fuses the flavour (Clebsch–Gordan).
            x_ir = self.auxiliary().multiply_elements(self.RG(a), self.RG(b))
            return self.from_ir_image(x_ir)
        cache = self.__dict__.setdefault("_multiply_cache", {})
        ckey = (sec_a, sec_b)
        cached = cache.get(ckey)
        if cached is None:
            x_ir = self.auxiliary().multiply_elements(
                self.RG(sec_a), self.RG(sec_b))
            cached = self.from_ir_image(x_ir)
            cache[ckey] = cached
        total = _label_add(fa, fb)
        if total is None or _label_is_zero(total):
            return cached
        return Element({
            _label_translate(lbl, total): c
            for lbl, c in cached.terms.items()
        })

    # ----- remaining KAlgebra primitives, derived from the RG data --------

    def coefficient_ring(self):
        """Default: the auxiliary's coefficient ring."""
        return self.auxiliary().coefficient_ring()

    def identity(self) -> Label:
        """Default: the UV label whose apex is the auxiliary's identity."""
        return self._apex_inverse(self.auxiliary().identity())

    def _label_section_decompose(self, label):
        """Default: the auxiliary's section decomposition at the apex
        label (unflavoured: `(label, R.one())`).

        `r_label_decompose` (the single-irrep lift coordinate) delegates
        the same way and is the preferred coordinate for new code; this
        method is kept because it works for *every* flow, whereas
        `r_label_decompose` is available only where the auxiliary
        implements it."""
        return self.auxiliary()._label_section_decompose(self.apex(label))

    def r_label_decompose(self, label):
        """Default: **delegate** to the auxiliary's flavour-lift coordinate at
        the apex label (the section in auxiliary coordinates; UV ≡ aux when
        `apex` is the identity).  `r_label_decompose` is *optional*, so this
        simply relies on the auxiliary — itself a `KAlgebra` — optionally
        implementing it: present ⟹ the flow has it, absent ⟹ it raises, exactly
        as an optional method should.  Mirrors `_label_section_decompose`'s
        delegation; `r_label_compose` is the inverse via `_apex_inverse`.

        With delegation in place, `_label_section_decompose` is derivable from
        this by the `KAlgebra` forward bridge wherever the auxiliary implements
        `r_label_decompose`; each flow acquires the lift coordinate exactly
        when its auxiliary does."""
        return self.auxiliary().r_label_decompose(self.apex(label))

    def r_label_compose(self, section, r_basis_label):
        """Inverse of `r_label_decompose`: rebuild the auxiliary label via the
        auxiliary's `r_label_compose` (the label-level flavour shift; the
        flavour is central, so no `embed_R` round-trip), then pull back to the
        UV labelling with `_apex_inverse`.

        This is the **label-level** lift coordinate (section ↔ flavour index),
        deliberately *not* the central ring-embedding `embed_R`: for matter
        flows the canonical basis is not the rep-ring character
        (`L_{κ=1}² = M_{κ=2}` ≠ `χ_1² = χ_0+χ_2`), so `embed_R` as a ring
        homomorphism (and `forget` as one) is a separate, subtler matter than
        this coordinate."""
        return self._apex_inverse(
            self.auxiliary().r_label_compose(section, r_basis_label))

    def verify_grading_factors_through_lift(self, label) -> bool:
        """The RG grading's `label ↦ charge ∈ Γ_RG` **factors through the
        flavour-lift**: `deg(label) = deg(section) + deg(χ_r)`, with the
        flavour character's charge that of its single (highest-weight) central
        image.  I.e. `Γ_RG` receives the flavour only through its 1d-rep /
        weight — the grading-side shadow of "the lift is canonical up to 1d
        reps", and the precondition that makes `forget` quotient `Γ_RG` by the
        flavour charges cleanly."""
        g = self.grading()
        sec, r = self.r_label_decompose(label)            # sec in aux coords
        char = self.auxiliary().embed_R(
            self.coefficient_ring().basis_element(r))
        char_charges = {tuple(g.charge(lbl)) for lbl in char.terms}
        if len(char_charges) != 1:
            return False
        char_charge = next(iter(char_charges))
        lhs = tuple(g.charge(self.apex(label)))
        rhs = tuple(g.charge_sum(tuple(g.charge(sec)), char_charge))
        return lhs == rhs

    def rho_inverse(self, a) -> Label:
        """`ρ_UV⁻¹(a)`, derived from `solve_rg`'s mirror constraint
        (`ρ_UV⁻¹(a) = ρ_IR⁻¹(upper)`); cached alongside `RG`.

        The mirror yields the upper label in **auxiliary** coordinates;
        it is mapped back to the UV labelling via `_apex_inverse` (a no-op
        when `apex` is the identity, e.g. `BPSKAlgebra`), so `rho_inverse`
        returns a UV label — consistent with `identity()`."""
        key = tuple(a)
        if key not in self.__dict__.get("_rho_inv_cache", {}):
            self.RG(a)                              # populates _rho_inv_cache
        rui = self.__dict__.get("_rho_inv_cache", {}).get(key)
        if rui is None:
            raise ValueError(
                f"rho_inverse({a!r}) could not be derived (mirror "
                f"constraint did not close — raise the S_RG cutoff, or "
                f"override rho_inverse)."
            )
        return self._apex_inverse(rui)

    def tRG(self, a) -> Element:
        """The **alternative RG map** `tRG(a)`, defined by
        `RG(a)·S_RG = S_RG·tRG(a)` (the conjugate of `RG(a)` by `S_RG`,
        `= ρ_IR⁻¹(RG(ρ_UV(a)))`).

        Generic default: `graded_rg_solver.solve_trg` — `solve_rg` in the
        opposite algebra (the left discovery `S_RG·tRG(a) = a + O(q)`),
        cached.  Its mirror yields forward `ρ_UV(a)` (see `rho`)."""
        cache = self.__dict__.setdefault("_trg_cache", {})
        key = tuple(a)
        if key in cache:
            return cache[key]
        from graded_rg_solver import solve_trg, solve_trg_exact
        if self._use_exact_window():
            trg, rho_uv = solve_trg_exact(
                self.auxiliary(), self.grading(),
                self._s_rg_component, self.apex(a))
        else:
            cut = self._rg_cutoff()
            trg, rho_uv = solve_trg(
                self.auxiliary(), self.grading(),
                self._rg_generator_cached(cut), self.apex(a),
                s_rg_hi=self._rg_generator_cached(cut + self._rg_cutoff_delta()),
            )
        cache[key] = trg
        self.__dict__.setdefault("_rho_cache", {})[key] = rho_uv
        return trg

    def rho(self, a) -> Label:
        """Forward `ρ_UV(a)`, derived from the **intertwining**
        `RG ∘ ρ_UV = ρ_IR ∘ tRG`, i.e.

            ρ_UV(a) = from_ir_image( ρ_IR( tRG(a) ) ).

        This reads `ρ_UV` off the (correct) `tRG` *element* via `from_ir_image`,
        rather than the opposite-algebra mirror's `upper` — the two agree when
        the survivor is the whole IR (the quantum-torus corner) but the mirror
        `upper` is unreliable through the opposite-algebra wrapper for flows
        with a central flavour, whereas the intertwining is always valid.
        Subclasses that know a combinatorial `σ` (e.g. `BPSKAlgebra`) may
        override for speed."""
        from laurent_poly import LaurentPoly
        img = self.auxiliary().rho_element(self.tRG(a))
        e = self.from_ir_image(img)
        terms = [(l, c) for l, c in e.terms.items() if not c.is_zero()]
        if len(terms) == 1 and terms[0][1] == LaurentPoly({0: 1}):
            return terms[0][0]
        # Fallback: the mirror-derived ρ_UV cached alongside tRG.  The
        # mirror value is unreliable through the opposite-algebra wrapper
        # for flows with a central flavour (see the docstring) — warn when
        # it is actually served, so a silent wrong ρ cannot propagate.
        ruv = self.__dict__.get("_rho_cache", {}).get(tuple(a))
        if ruv is not None:
            import warnings
            warnings.warn(
                f"RGKAlgebra.rho({a!r}): from_ir_image(ρ_IR(tRG)) was not a "
                f"single canonical label; serving the mirror-derived ρ_UV "
                f"fallback, which is unreliable for flows with a central "
                f"flavour — verify with verify_rg_trg_intertwine.",
                RuntimeWarning, stacklevel=2)
            return self._apex_inverse(ruv)
        raise ValueError(
            f"rho({a!r}) could not be derived: from_ir_image(ρ_IR(tRG)) "
            f"was not a single canonical label ({dict(e.terms)}) and no "
            f"mirror fallback was available."
        )

    def verify_rg_trg_intertwine(self, a) -> bool:
        """`ρ_UV` and `ρ_IR` intertwine `RG` and `tRG`:
        `RG(ρ_UV(a)) == ρ_IR(tRG(a))`."""
        lhs = self.RG(self.rho(a))
        rhs = self.auxiliary().rho_element(self.tRG(a))
        return lhs == rhs

    def _s_rg_element_cached(self, K_expand: int, cutoff: int = None) -> Element:
        """`S_RG` as an aux `Element` (expanded to `q^K_expand`), with `cutoff`
        matter levels, cached per `(K_expand, cutoff)`.  `cutoff=None` ⇒ the
        fixed `_rg_cutoff()`; the adaptive fallback passes a growing
        `cutoff`."""
        cut = self._rg_cutoff() if cutoff is None else cutoff
        cache = self.__dict__.setdefault("_s_rg_elt_cache", {})
        s = cache.get((K_expand, cut))
        if s is None:
            s = self._s_rg_as_aux_element(cut, K_expand)
            cache[(K_expand, cut)] = s
        return s

    def _rho_s_rg_element_cached(self, K_expand: int, cutoff: int = None) -> Element:
        """`ρ_IR(S_RG)` as an aux `Element`, cached per `(K_expand, cutoff)`."""
        cut = self._rg_cutoff() if cutoff is None else cutoff
        cache = self.__dict__.setdefault("_rho_s_rg_elt_cache", {})
        s = cache.get((K_expand, cut))
        if s is None:
            s = self.auxiliary().rho_element(
                self._s_rg_element_cached(K_expand, cut))
            cache[(K_expand, cut)] = s
        return s

    def rg_times_S(self, a, K_expand: int, cutoff: int = None) -> Element:
        """`RG(a)·S_RG` (the **FS object**) as an aux `Element` expanded to
        `q^K_expand` with `cutoff` matter levels, **cached** per
        `(a, K_expand, cutoff)`.

        Generic analogue of `BPSKAlgebra._FS_cache`: the expensive part of
        the Schur-index transport, reused across repeated traces / inner
        products (e.g. a full Gram matrix at fixed `K`)."""
        cut = self._rg_cutoff() if cutoff is None else cutoff
        cache = self.__dict__.setdefault("_rg_S_cache", {})
        key = (tuple(a), K_expand, cut)
        fs = cache.get(key)
        if fs is None:
            fs = self.auxiliary().multiply_elements(
                self.RG(a), self._s_rg_element_cached(K_expand, cut))
            cache[key] = fs
        return fs

    def _s_rg_charges_to_height(self, B: int):
        """All grading charges `p` in the positive cone with `h(p) ≤ B` —
        the certified `S_RG` window enumerator (finite by
        height-positivity).  **Soft contract**, presentation-specific
        (the generic contract cannot enumerate a cone it cannot see):
        `BPSKAlgebra` supplies the cone-witness L-shell BFS; the default
        raises, and `rg_times_s_rg` falls back to its windowed heuristic."""
        raise NotImplementedError(
            f"{type(self).__name__}._s_rg_charges_to_height(B) is not "
            f"implemented; supply the finite charge window "
            f"{{p in cone : h(p) <= B}} to enable the certified FS object."
        )

    def rg_times_s_rg(self, a, k: int) -> Element:
        """The **FS object** `RG(a)·S_RG` *to q-order `k`* (as an auxiliary
        `Element`) — the safe object a `trace` / `inner_product` to `q^k`
        consumes: these components enter the bilinear trace pairing
        `I_{a,b} = Σ_{c,d} [RG(a)·S_RG]_c·[RG(b)·S_RG]_d·I^aux_{c,d}`,
        `Tr(a) = I(1, a)` (see `_inner_product_uncached`).

        This is the real trace-side contract: **not** `S_RG` to q-order `k`,
        but the *product* `RG(a)·S_RG` to q-order `k`.  q-order is non-additive
        across the product (the pairing phase `q^{⟨γ,γ'⟩}` can be negative — the
        binding quadratic form), so windowing `S_RG` by q-order cannot certify
        the product.  The sharper point:
        windowing each `s_g` by a *fixed* q-order **before** the cocycle shift
        breaks the exact cancellations — deep output charges `η` keep a spurious
        low-order remainder where two partition-series contributions should
        cancel.  (Reproduced on A_4 drop-node-1: `[RG·S_RG]_{(0,7,1,0)}` is
        exactly `-q^14`, but the fixed-window product reports a spurious `5·q^8`
        until the q-margin exceeds the η-dependent cancellation depth — up to 34
        orders on the dictionary.)

        **Exact path (generic).**  Active when the realisation supplies
        `_s_rg_component` (the exact per-component `[S_RG]_γ` oracle) and the
        auxiliary is cone-capable (`_fs_exact_available`).  We bound the
        **output charge η**, never the S-charge `γ`: by grading additivity the
        S-charge feeding `η` through `δ ∈ supp RG(a)` is *determined*,
        `γ = deg(η) − deg(δ)`, so

            [RG(a)·S_RG]_η = Σ_δ Σ_{g ∈ [S_RG]_γ} [RG(a)]_δ · s_g · ⟨η | δ·g⟩

        is a **finite, exact** sum in the localized ring — each `[S_RG]_γ` fetched *complete*
        from `_s_rg_component(γ)` (no S-window, so no broken cancellation),
        expanded to `q^k` only at the very end (`_rg_times_s_rg_exact` /
        `_fs_eta`).  Output labels are enumerated by walking outward from
        `supp RG(a)`, multiplying each by the `S_RG` ray-unit labels in the
        auxiliary (`_fs_ray_unit_labels`; label-type-agnostic, so nested
        `add_flavour` / QT-tensor labels work too) and grown to a **slack
        stability** certificate; the assembled order is super-linear, so the true
        support is finite and a finite shell captures it.  The result is exact
        through `q^k`.

        **Fallback (windowed heuristic).**  When the oracle/aux are missing:
        `RG(a) × S_RG` with `S_RG` windowed at the fixed `_rg_cutoff()`
        (expanded to `q^{k + _rg_cutoff()}`) — sound only in the small-`k` /
        shallow-`η` regime, NOT a certified bound (see above).  A subclass with
        a sharper closed form (e.g. `BPSKAlgebra` via `c_gamma_via_s`)
        overrides this method outright."""
        cache = self.__dict__.setdefault("_fs_certified_cache", {})
        key = (tuple(a), k)
        hit = cache.get(key)
        if hit is not None:
            return hit
        try:
            fs = self._rg_times_s_rg_exact(a, k)
        except NotImplementedError:
            fs = self._rg_times_S_adaptive(a, k)
        cache[key] = fs
        return fs

    def _rg_times_S_adaptive(self, a, k: int) -> Element:
        """Fallback FS object `RG(a)·S_RG` to `q^k` with an **adaptive
        matter-level cutoff** (two-cutoff stability), replacing the old fixed
        `_rg_cutoff()` level cap that silently truncated the `S_RG` tower past
        `~q^{2·cutoff}` (matter level `N` first contributes at
        a q-order growing with `N`, so a *fixed* level count drops the tail).

        Grows the level count by 2 until the `q^{≤k}` coefficients repeat across
        two successive cutoffs — truncation artifacts move with the cutoff, so a
        repeat certifies the result.  Sound because the true FS support to `q^k`
        is finite (a finite level shell captures it — the shell-stability idiom
        on the **level** axis, the dual of the BPS q-order shell).  A strict no-op where the fixed
        cutoff was already adequate (the first two cutoffs agree immediately)."""
        def _fs_at(cut):
            return self.rg_times_S(a, k + cut, cutoff=cut)
        cut = self._rg_cutoff()
        prev = _fs_at(cut)
        cap = self._rg_cutoff() + 2 * k + 6
        while cut < cap:
            cut += 2
            cur = _fs_at(cut)
            if self._fs_trunc_eq(prev, cur, k):
                return cur
            prev = cur
        import warnings
        warnings.warn(
            f"_rg_times_S_adaptive({a!r}, k={k}): cutoff stability was NOT "
            f"reached by the cap (cutoff {cap}); returning the last iterate, "
            f"whose q^<={k} coefficients may still be truncation-corrupted.",
            RuntimeWarning, stacklevel=2)
        return prev

    @staticmethod
    def _fs_trunc_eq(fs1: Element, fs2: Element, k: int) -> bool:
        """Whether two FS `Element`s agree on all coefficients through `q^k`
        (the stability test for `_rg_times_S_adaptive`)."""
        def trunc(fs):
            out = {}
            for lab, lp in fs.terms.items():
                c = {e: v for e, v in lp._coeffs.items() if e <= k and v != 0}
                if c:
                    out[lab] = c
            return out
        return trunc(fs1) == trunc(fs2)

    def _fs_exact_available(self) -> bool:
        """Whether the exact per-η FS oracle path applies: the grading carries
        `cone_gens` and `_s_rg_component` (the exact `[S_RG]_γ` oracle) is
        implemented (`_exact_window_available()`).

        **No flat-integer-label requirement.**  The exact-FS support
        walk (`_rg_times_s_rg_exact`) now enumerates its frontier by *multiplying*
        a support label by the `S_RG` ray-unit labels in the auxiliary
        (`aux.multiply(η, g_unit)`), never by vector-adding label tuples — so it
        works for **nested** aux labels (`(chord, flavour)` / `(chord, (c0,c1))`,
        the `add_flavour` and quantum-torus-tensor auxiliaries) exactly as for the
        flat-integer quantum-torus corners.  `_fs_eta` was already nested-safe (it
        uses `Γ_RG`-charge arithmetic + label *lookup*, never label addition); the
        walk's neighbour generation was the only label-space op, and the
        multiply-by-ray form removes it.  (Earlier this gate also required flat
        integer labels because the walk vector-added generators; that restriction
        is gone.)

        A subclass MAY still override to force the windowed fallback (return
        `False`) — e.g. a test pinning the windowed path."""
        return self._exact_window_available()

    def _fs_ray_unit_labels(self) -> list:
        """The **unit `S_RG` ray labels**: the auxiliary labels of `[S_RG]_g` for
        each grading-cone generator `g` (`_s_rg_component(g)`), deduplicated.

        These drive the exact-FS support walk's neighbour generation: multiplying
        a support label `η` by each of these *in the auxiliary*
        (`aux.multiply(η, g_unit)`) yields the next FS-support labels.  This is the
        general, label-type-agnostic replacement for the earlier label-space walk
        (vector-adding cone generators, with a flat-int-label restriction):

        * **quantum torus** — `aux.multiply(η, g_unit)` is the single label
          `η + deg(g_unit)`, so the walk traces the `S_RG` ray (as the earlier
          vector-add did);
        * **non-torus flat** (SQED₁, `u₊u₋ = 1+𝖖v`) — the product spreads into the
          `v`-residue labels automatically (the residue spread is found here, with
          no separate probing step);
        * **nested labels** (`add_flavour` / QT-tensor auxiliaries) — the product
          yields valid nested labels (no tuple concatenation), so the exact path
          now covers these flows too.

        Every step raises the grading height by `h(deg(g_unit)) > 0`
        (height-positivity), so the order-guided walk still terminates."""
        labels: list = []
        seen: set = set()
        for cg in (self.grading().cone_gens or ()):
            for lbl in self._s_rg_component(tuple(cg)):
                t = tuple(lbl)
                if t not in seen:
                    seen.add(t)
                    labels.append(t)
        return labels

    def _fs_eta(self, seeds: dict, eta) -> "HabiroElement":
        """Exact `[RG(a)·S_RG]_η` as a `HabiroElement`, with
        `seeds = supp RG(a)` (or `{aux.identity(): 1}` for `a = 1`).

        **Complete per η — never windows the S-charge.**  Grading additivity
        forces, for each `δ ∈ seeds`, the *unique* S-charge `γ = deg(η) − deg(δ)`
        feeding `η`; `[S_RG]_γ` is fetched *whole* from `_s_rg_component(γ)`
        (`{}` off-cone) and summed exactly in the localized ring:

            [RG(a)·S_RG]_η = Σ_δ Σ_{g ∈ [S_RG]_γ} [RG(a)]_δ · s_g · ⟨η | δ·_aux g⟩.

        Expanding to `q^k` is deferred to the caller, so the exact
        cancellations between contributions survive (a *fixed-q-order* window
        on `s_g` before the cocycle shift destroys them — the spurious-remainder
        failure of the windowed heuristic; see `rg_times_s_rg`).  This is the
        generic analogue of `bps_kalgebra_internals.c_gamma_via_s` (the
        `deg = id` quantum-torus corner, where `γ = η − δ`)."""
        from habiro import HabiroElement
        aux = self.auxiliary()
        g = self.grading()
        # Memoise `_s_rg_component` by charge: the default implementation
        # re-filters the whole assembled dict on every call, and the support
        # walk queries the same charges across many η — so dedupe them.
        comp_cache = self.__dict__.setdefault("_fs_comp_cache", {})

        def _component(p):
            p = tuple(p)
            hit = comp_cache.get(p)
            if hit is None:
                hit = self._s_rg_component(p)
                comp_cache[p] = hit
            return hit

        deg_eta = g.charge(eta)
        parts = []
        for d_label, f_d in seeds.items():
            if f_d.is_zero():
                continue
            gamma = g.charge_sum(deg_eta, g.charge_neg(g.charge(d_label)))
            comp = _component(gamma)
            for g_label, s_g in comp.items():
                if s_g.is_zero():
                    continue
                C = aux.multiply(d_label, g_label).terms.get(tuple(eta))
                if C is None or C.is_zero():
                    continue
                scale = f_d * C
                if scale.is_zero():
                    continue
                parts.append(HabiroElement(scale * s_g.numerator,
                                           dict(s_g.denom)))
        return HabiroElement.sum(parts) if parts else HabiroElement.zero()

    # Slack-stability schedule for the order-guided walk's completeness check.
    _FS_SLACK_START = 4
    _FS_SLACK_STEP = 4
    _FS_SLACK_MAX_ITERS = 8

    def _rg_times_s_rg_exact(self, a, k: int) -> Element:
        """`RG(a)·S_RG` to q-order `k`, computed exactly per output charge η and
        enumerated by walking the **tame actual support**.

        Each `[RG(a)·S_RG]_η` is the *complete* exact localized-ring sum (`_fs_eta` — the
        S-charge is never windowed, so cancellations survive).  The support is
        enumerated by walking outward from the seeds `supp RG(a)`, generating each
        frontier node's neighbours by **multiplying** it by the `S_RG` ray-unit
        labels in the auxiliary (`aux.multiply(η, g_unit)`, `g_unit ∈
        _fs_ray_unit_labels`) — a label-type-agnostic step that works for flat
        quantum-torus labels, the non-torus residue spread (`u₊u₋=1+𝖖v`), AND
        **nested** `add_flavour` / QT-tensor labels alike (no label vector-add, so
        no flat-integer-label restriction).  A node is expanded iff its
        **computed** leading order `ω(F_η) ≤ k + slack`: the assembled order grows
        super-linearly along the FS rays (every ray step raises the grading
        height), so once `ω > k` descendants stay `> k`.  This visits only the
        tame support (+ a slack collar); the support genuinely cannot be read off
        the `S_RG`/`RG` supports because of the cancellations, so the *computed*
        order is the right guide.

        Completeness is certified by **slack stability**: the support is
        recomputed at `slack` and `slack + step` and must agree (widening the
        collar adds nothing).  Raises `NotImplementedError` when the oracle/aux
        are unavailable, so `rg_times_s_rg` falls back to the windowed heuristic."""
        if not self._fs_exact_available():
            raise NotImplementedError(
                "exact per-η FS oracle needs `_s_rg_component` and a "
                "cone-capable auxiliary")
        from laurent_poly import LaurentPoly
        from collections import deque
        aux = self.auxiliary()
        a = tuple(a)
        if a == self.identity():
            seeds = {aux.identity(): LaurentPoly.one()}
        else:
            seeds = {d: c for d, c in self.RG(a).terms.items()
                     if not c.is_zero()}
        ray_units = self._fs_ray_unit_labels()
        seed_keys = [tuple(d) for d in seeds]

        # Cross-call per-η HabiroElement cache (k-independent): the vacuum FS object is
        # reused by every trace; `RG(a)·S_RG`'s coefficients across `k` / a Gram
        # matrix.  Also shared between the two slack passes below.
        xcache = self.__dict__.setdefault("_fs_eta_xcache", {})

        def fs_eta_cached(eta):
            key = (a, tuple(eta))
            h = xcache.get(key)
            if h is None:
                h = self._fs_eta(seeds, eta)
                xcache[key] = h
            return h

        def support_at(slack):
            visited = set(seed_keys)
            frontier = deque(seed_keys)
            out: dict = {}
            while frontier:
                eta = frontier.popleft()
                h = fs_eta_cached(eta)
                if h.is_zero():
                    continue
                lp = h.expand(k)
                if not lp.is_zero():
                    out[eta] = lp
                if h.k_min() <= k + slack:
                    # Generate frontier neighbours by *multiplying* η by each
                    # S_RG ray-unit label in the auxiliary — label-type-agnostic
                    # (works for flat-int AND nested labels), and it captures the
                    # non-torus residue spread (u₊u₋=1+𝖖v) for free.
                    for g_unit in ray_units:
                        for nb, C in aux.multiply(eta, g_unit).terms.items():
                            if C.is_zero():
                                continue
                            nb = tuple(nb)
                            if nb not in visited:
                                visited.add(nb)
                                frontier.append(nb)
            return out

        slack = self._FS_SLACK_START
        prev = support_at(slack)
        for _ in range(self._FS_SLACK_MAX_ITERS):
            slack += self._FS_SLACK_STEP
            cur = support_at(slack)
            if cur == prev:
                return Element(cur)
            prev = cur
        import warnings
        warnings.warn(
            f"{type(self).__name__}.rg_times_s_rg: FS support did not reach a "
            f"slack-stability certificate (k={k}); returning the widest "
            f"computed value, which may be under-converged.",
            RuntimeWarning, stacklevel=4)
        return Element(prev)

    def _rg_times_s_rg_assembled_DO_NOT_USE(self, a, k: int) -> Element:
        from habiro import HabiroElement
        aux = self.auxiliary()
        grading = self.grading()
        rg_a = self.RG(a)
        if not rg_a.terms:
            return Element({})
        # Per-(δ, γ) order bound: lead(f_δ) + h(γ).  The complete S-window
        # for q-order ≤ k is therefore h(γ) ≤ k − min_δ lead(f_δ) —
        # independent of the (possibly negative) heights of the F-support.
        neg_a = 0
        for f_d in rg_a.terms.values():
            if not f_d.is_zero():
                lead = min(f_d._coeffs.keys())
                if lead < neg_a:
                    neg_a = lead
        B = k - neg_a
        charges = list(self._s_rg_charges_to_height(B))
        zero_p = grading.zero_charge()
        contribs: dict = {}
        seen_zero = False
        for p in charges:
            comp = self._s_rg_component(tuple(p))
            if tuple(p) == tuple(zero_p):
                seen_zero = True
                if not comp:
                    comp = {aux.identity(): HabiroElement.one()}
            for g_label, s_g in comp.items():
                if s_g.is_zero():
                    continue
                for d_label, f_d in rg_a.terms.items():
                    prod = aux.multiply(d_label, g_label)
                    for eta, C in prod.terms.items():
                        scale = f_d * C
                        if scale.is_zero():
                            continue
                        contribs.setdefault(eta, []).append(
                            HabiroElement(scale * s_g.numerator,
                                          dict(s_g.denom)))
        if not seen_zero:
            # S_RG = 1 + …: the identity component must contribute.
            for d_label, f_d in rg_a.terms.items():
                contribs.setdefault(d_label, []).append(
                    HabiroElement(f_d, {}))
        terms: dict = {}
        for eta, parts in contribs.items():
            lp = HabiroElement.sum(parts).expand(k)
            if not lp.is_zero():
                terms[eta] = lp
        return Element(terms)

    # ---- inner-product computation route (selectable, see
    # `set_inner_product_route`) -------------------------------------------
    #   "direct"         — the `_inner_product_uncached` hook: the BPS
    #                      single-HabiroElement Schur formula (no Element
    #                      multiplication), or the generic RG-transport
    #                      pairing.  Sharp for an isolated `(a, b)`.
    #   "multiply_trace" — the universal product route
    #                      `I_{a,b} = Tr(ρ(a)·b) = Σ_c N^c·Tr(L_c)`
    #                      (= `KAlgebra.inner_product`).  Costs an Element
    #                      multiply per pair but reuses the memoised
    #                      single-element traces `Tr(L_c) = I(1, c)` across
    #                      ALL pairs — a big win for dense all-pairs sweeps
    #                      (orthonormality / Schur-Gram), where the shared
    #                      `Tr(L_c)` pool is `≪ n²` and each is the cheaper
    #                      1-F primitive.
    # Default "direct" (preserves the sharp BPS/RG path; no behaviour
    # change).  The base `KAlgebra.inner_product` IS the multiply+trace
    # route, so multiply+trace is the default for plain (non-RG) KAlgebras.
    inner_product_route: str = "direct"

    def set_inner_product_route(self, route: str) -> None:
        """Select how this instance's `inner_product` computes `I_{a,b}`:
        ``"direct"`` (the `_inner_product_uncached` hook) or
        ``"multiply_trace"`` (the `Tr(ρ(a)·b)` product route, reusing the
        `Tr(L_c)` memo).

        NOTE: the two routes do **not** agree in general for `a ≠ b`.
        ``"multiply_trace"`` computes `Tr(ρ(a)·b)` *literally* (exact
        multiply + exact single-element traces) — the definitional trace
        pairing; the BPS ``"direct"`` Schur is a fast approximation with a
        documented bar-asymmetry on the q-component (limitation (E) in
        `BPSKAlgebra._inner_product_uncached`) that can differ off-diagonal
        at higher q-order (`verify_inner_product_consistent` reports the
        gap).  The off-diagonal value therefore **depends on the route**,
        so the value cache is **cleared** on a switch."""
        if route not in ("direct", "multiply_trace"):
            raise ValueError(
                f"inner_product route must be 'direct' or 'multiply_trace', "
                f"got {route!r}")
        if route != self.__dict__.get("inner_product_route",
                                      type(self).inner_product_route):
            self.__dict__.pop("_ip_cache", None)          # route-dependent
        self.__dict__["inner_product_route"] = route

    def inner_product(self, a, b, K: int = 20, **kwargs):
        """Trace pairing `I_{a,b}`, with **per-`(a, b)` value memoization**
        and a selectable computation **route** (`set_inner_product_route`).

        The pairing is computed either by `_inner_product_uncached` (the
        overridable ``"direct"`` hook — subclass accelerators like the BPS
        single-`HabiroElement` Schur formula override the *hook*, not this wrapper,
        so they inherit the cache) or by the ``"multiply_trace"`` product
        route `Tr(ρ(a)·b) = Σ_c N^c·Tr(L_c)`.  Computing `I_{a,b}` is the
        dominant cost of every trace (`trace(a) = inner_product(1, a)`)
        and of the per-IR-label fan-out of the RG-transport trace; it is
        recomputed from scratch on each call otherwise (the F/FS
        intermediate caches do not memoise the assembled pairing value).

        Caching rule: store, per `(a, b)`, the series at the **highest K**
        computed; serve a request at smaller `K'` by truncating it (the
        `q^{≤K'}` coefficients of an order-`K ≥ K'` series are exact — the
        same support monotonicity the `_FS_cache` reuse across adaptive
        shell widenings relies on).  Per-instance, lazily created;
        immutable algebra ⇒ no invalidation.  `**kwargs` (e.g. BPS
        `cone_cutoff`) are forwarded to the direct hook but are NOT part
        of the cache key (perf hints; they do not affect the value).
        Labels pass through unchanged; caching is bypassed for unhashable
        labels.

        **Identity / route / recursion.** `I(1, b) ≡ Tr(L_b)` is the
        irreducible primitive, so the identity-left case is delegated to
        `trace` (a separate memoised method); the route only governs a
        general `a ≠ 1`.  Because `trace` is its own method (NOT
        `inner_product(1, ·)`), the ``"multiply_trace"`` route's
        `Σ_c N^c·Tr(L_c)` calls `trace` directly — no recursion back
        into `inner_product`."""
        if a == self.identity():
            return self.trace(b, K, **kwargs)         # I(1,b) = Tr(L_b)
        cache = self.__dict__.setdefault("_ip_cache", {})
        try:
            key = (a, b)
            hit = cache.get(key)
        except TypeError:                       # unhashable label → no cache
            key = None
            hit = None
        if hit is not None and hit.K >= K:
            return hit if hit.K == K else RPowerSeries(hit.ring, hit.coeffs, K)
        if self.inner_product_route == "multiply_trace":
            # default relation I_{a,b} = Tr(ρ(a)·b) = Σ_c N^c·Tr(L_c),
            # reusing the memoised single-element traces (`KAlgebra`'s
            # product route).
            val = KAlgebra.inner_product(self, a, b, K)
        else:
            val = self._inner_product_uncached(a, b, K, **kwargs)
        if key is not None and (hit is None or val.K > hit.K):
            cache[key] = val
        return val

    # Internal-order stability schedule for the bilinear trace pairing.
    _IP_STABILITY_STEP = 2
    _IP_STABILITY_MAX_ITERS = 16

    @staticmethod
    def _series_eq_through(s1: RPowerSeries, s2: RPowerSeries, K: int) -> bool:
        """Whether two `RPowerSeries` agree on every coefficient through `q^K`."""
        for e in set(s1.coeffs) | set(s2.coeffs):
            if e > K:
                continue
            c1 = s1.coeffs.get(e)
            c2 = s2.coeffs.get(e)
            z1 = c1 is None or c1.is_zero()
            z2 = c2 is None or c2.is_zero()
            if z1 and z2:
                continue
            if z1 != z2 or c1 != c2:
                return False
        return True

    def _inner_product_uncached(self, a, b, K: int = 20):
        """Trace pairing `I_{a,b}` via the **bilinear expansion** — the only
        well-defined route for an `RGKAlgebra`:

            I_{a,b} = Σ_{c,d} [RG(a)·S_RG]_c · [RG(b)·S_RG]_d · I^aux_{c,d},
            I^aux_{c,d} = aux.inner_product(c, d) = Tr_aux(ρ(L_c)·L_d),

        where `[x]_c` is the coefficient of the auxiliary basis label `c` in
        `x`, and the per-charge components `[RG(·)·S_RG]_c` are the exact
        pieces from `rg_times_s_rg` (one `S_RG` each).  The trace is the
        `a = 1` face `Tr(b) = I(1, b)`, with `RG(1)·S_RG = S_RG`:
        `Tr(b) = Σ_{c,d} [S_RG]_c · [RG(b)·S_RG]_d · I^aux_{c,d}`.

        **Why bilinear — `Tr_aux(ρ(S_RG)·…·S_RG)` is NOT DEFINED.**  `ρ(S_RG)`
        is a formal sum over the **negative** grading cone and `S_RG` over the
        **positive** cone, so their product is *not a well-defined auxiliary
        element* (infinitely many contributions at a fixed output charge).
        Forming it and relying on a `q^K` truncation to make it finite is
        wrong — the result depends on the cutoff (the residual marches to the
        `q^K` boundary).  Bilinearity instead moves `ρ` and the trace onto the
        **well-defined single-basis pairings** `I^aux_{c,d}`, leaving a finite
        sum at each `q`-order.  This is the generic form of
        `BPSKAlgebra._schur_index` (the quantum-torus corner, where
        `I^aux_{c,d} = (q²;q²)_∞^r δ_{c,d}`, collapses it to the diagonal
        `(q²)_∞^r Σ_η c_a(η) c_b(η)`).

        **Truncation-safe (compute exact, truncate last).**  The components are
        taken to an internal order `q^M` (each `[·]_c` exact per `η` via
        `rg_times_s_rg`) and `M` is grown until the `q^{≤K}` pairing repeats
        across two successive windows — the internal-margin discipline of
        `_rg_times_S_adaptive` / `BPSKAlgebra._schur_index_stable` (negative-`q`
        component tails pair with the higher-order tail of the opposite leg, so
        a single window can be short).

        Overridable hook behind `inner_product` (which memoises); a subclass
        with a sharper closed form (e.g. `BPSKAlgebra`) overrides *this*."""
        from kalgebra import _laurentpoly_times_rpowerseries
        aux = self.auxiliary()
        R = self.coefficient_ring()
        iaux_cache: dict = {}

        def _iaux(c, d, M):
            v = iaux_cache.get((c, d))
            if v is None or v.K < M:
                v = aux.inner_product(c, d, M)
                iaux_cache[(c, d)] = v
            return v

        def pairing_at(M):
            X = self.rg_times_s_rg(a, M)
            Y = self.rg_times_s_rg(b, M)
            acc = RPowerSeries.zero(R, K)
            for c, xc in X.terms.items():
                if xc.is_zero():
                    continue
                lc = min(xc._coeffs)
                for d, yd in Y.terms.items():
                    if yd.is_zero():
                        continue
                    # `I^aux_{c,d} = Tr_aux(ρ(c)·d)` starts at `q^0`
                    # (orthonormality: `I = δ + O(q)`, no negative powers), so a
                    # pair whose FS coefficients already exceed `q^K` contributes
                    # 0 to `I_{a,b}` through `q^K`.  Skip it — both a perf win and
                    # the thing that keeps the deep tail of the exact-FS walk from
                    # querying `aux.inner_product` on labels where an
                    # *incomplete-trace* auxiliary raises (e.g. `U1A1AoddKAlg`'s
                    # conjectural deep-power regime) — labels which never affect
                    # the `q^K` answer.
                    if lc + min(yd._coeffs) > K:
                        continue
                    iab = _iaux(c, d, M)
                    if iab.is_zero():
                        continue
                    acc = acc + _laurentpoly_times_rpowerseries(xc * yd, iab)
            return acc

        M = max(K, 1)
        prev = pairing_at(M)
        for _ in range(self._IP_STABILITY_MAX_ITERS):
            M += self._IP_STABILITY_STEP
            cur = pairing_at(M)
            if self._series_eq_through(prev, cur, K):
                return cur
            prev = cur
        import warnings
        warnings.warn(
            f"{type(self).__name__}._inner_product_uncached: bilinear pairing "
            f"did not reach internal-order stability (K={K}); returning the "
            f"widest computed value.",
            RuntimeWarning, stacklevel=3,
        )
        return prev

    def _trace_is_grade_concentrated(self) -> bool:
        """Whether `Tr_aux` is supported on `Γ_RG`-grade 0 — i.e. `Γ_RG` is a
        **gauge** grading (the graded directions are integrated out by the
        trace), so the pairing keeps only its grade-0 block and
        `_pair_grade_blocks` is exact.

        Default **False**: use the full cross-pairing, correct for ANY
        grading — including a **flavour** `Γ_RG`, where the trace keeps the
        `μ`-refinement and a grade-0 projection would wrongly integrate `μ`
        out (the gauge-vs-flavour distinction).  A flow whose `Γ_RG`
        is a gauge grading (e.g. BPS node-deletion, which drops gauge nodes)
        overrides to True; `verify_inner_product_grade_pruned` cross-checks
        the prune against the full pairing."""
        return False

    def _pair_grade_blocks(self, rho_a_S: Element, b_S: Element, K: int):
        """`Σ_q Tr_aux([ρ(a_S)]_q · [b_S]_{−q})` — the grade-0-block pairing.

        Groups the (already ρ-applied) `ρ(a_S)` and `b_S` by `Γ_RG`-grade and
        multiplies only the blocks whose product lands in grade 0 (the only
        grade a grade-concentrated `Tr_aux` sees), skipping the off-grade
        `|FS|²` work.  Exact iff `_trace_is_grade_concentrated()`; relies only
        on grade-0-trace-support, not on how `ρ` acts on the grade."""
        from collections import defaultdict
        aux = self.auxiliary()
        g = self.grading()
        ra_by: dict = defaultdict(dict)
        b_by: dict = defaultdict(dict)
        for l, c in rho_a_S.terms.items():
            if not c.is_zero():
                ra_by[g.charge(l)][l] = c
        for l, c in b_S.terms.items():
            if not c.is_zero():
                b_by[g.charge(l)][l] = c
        total = None
        for q, ablk in ra_by.items():
            bblk = b_by.get(g.charge_neg(q))
            if not bblk:
                continue
            block = aux.multiply_elements(Element(dict(ablk)), Element(dict(bblk)))
            tv = aux.trace_element(block, K)
            total = tv if total is None else total + tv
        return total if total is not None else aux.trace_element(Element({}), K)

    def verify_inner_product_grade_pruned(self, a, b, K: int = 20) -> bool:
        """Cross-check the grade-0-block prune against the full cross-pairing
        `Tr_aux(ρ(a_S)·b_S)`.  Must hold whenever
        `_trace_is_grade_concentrated()` is True (the soundness gate for the
        gauge-grading prune; it fails on a flavour `Γ_RG`)."""
        aux = self.auxiliary()
        a_S = self.rg_times_s_rg(a, K)
        b_S = self.rg_times_s_rg(b, K)
        rho_a_S = aux.rho_element(a_S)
        pruned = self._pair_grade_blocks(rho_a_S, b_S, K)
        full = aux.trace_element(aux.multiply_elements(rho_a_S, b_S), K)
        return pruned == full

    def trace(self, a, K: int = 20, **kwargs):
        """`Tr(L_a)` — the **memoised** single-element trace primitive.

        A caching wrapper around the overridable `_trace_uncached` hook
        (parallel to `inner_product`): subclasses with a sharper trace
        (the BPS direct Schur trace) override the *hook*, inheriting the
        cache.  `Tr(L_a)` is THE expensive primitive shared across all
        traces and — via the ``"multiply_trace"`` `inner_product` route —
        across all pairings, so it is the highest-value thing to memoise.

        Cached per `a` at the highest K computed; a smaller-`K'` request
        is served by truncation.  Per-instance, lazily created; immutable
        algebra ⇒ no invalidation.  `**kwargs` (e.g. BPS `cone_cutoff`)
        forward to the hook; labels pass through unchanged and caching is
        bypassed for unhashable labels."""
        cache = self.__dict__.setdefault("_trace_cache", {})
        try:
            hit = cache.get(a)
            key = a
        except TypeError:                       # unhashable label → no cache
            key = None
            hit = None
        if hit is not None and hit.K >= K:
            return hit if hit.K == K else RPowerSeries(hit.ring, hit.coeffs, K)
        val = self._trace_uncached(a, K, **kwargs)
        if key is not None and (hit is None or val.K > hit.K):
            cache[key] = val
        return val

    def _trace_uncached(self, a, K: int = 20, **kwargs):
        """Uncached `Tr_UV(L_a) = I(1, a)` — the `a_left = 1` face of the
        bilinear trace pairing (`_inner_product_uncached`):

            Tr(a) = Σ_{c,d} [S_RG]_c · [RG(a)·S_RG]_d · I^aux_{c,d}.

        Always routes through the pairing **hook** `_inner_product_uncached`
        (NOT `inner_product`, whose identity-left `I(1,a)=Tr(L_a)` delegation
        would recurse into `trace`) — so the spectrum generator enters as the
        well-defined per-charge components `[S_RG]_c`, never as the *undefined*
        opposite-cone product `Tr_aux(ρ(S_RG)·…·S_RG)` (see
        `_inner_product_uncached`).  Overridable hook behind `trace` (which
        memoises); a subclass with a sharper trace (e.g. `BPSKAlgebra`)
        overrides *this*."""
        return self._inner_product_uncached(self.identity(), a, K)

    # ----- generic cache persistence -------------------------------------

    def save_cache(self, path: str) -> None:
        """Persist the **`q`-independent** generic caches to JSON: the
        RG-cache (`{label: {aux label: QNumberPoly}}`, stored in the
        𝖖-number basis) and the section-keyed multiply-decomposition cache
        (`{(sec_a, sec_b): Element}`).

        The RG·S_RG cache is *not* persisted (it is `q`-cutoff-dependent —
        a per-run memo).  Subclasses with a richer presentation-specific
        cache (e.g. `BPSKAlgebra`'s `_F_cache`/`_FS_cache`) override this."""
        import json
        rg = self.__dict__.get("_rg_cache", {})
        mul = self.__dict__.get("_multiply_cache", {})

        def qn_obj(q):
            return {str(n): int(c) for n, c in q._coeffs.items()}

        def el_obj(el):
            return {json.dumps(list(l)): {str(e): int(c) for e, c in lp._coeffs.items()}
                    for l, lp in el.terms.items()}

        obj = {
            "rg": {json.dumps(list(k)): {json.dumps(list(lbl)): qn_obj(q)
                                         for lbl, q in d.items()}
                   for k, d in rg.items()},
            "multiply": {json.dumps([list(sa), list(sb)]): el_obj(el)
                         for (sa, sb), el in mul.items()},
        }
        with open(path, "w") as fh:
            json.dump(obj, fh)

    def load_cache(self, path: str) -> None:
        """Inverse of `save_cache`: repopulate `_rg_cache` / `_multiply_cache`."""
        import json
        from q_number_poly import QNumberPoly
        from laurent_poly import LaurentPoly
        with open(path) as fh:
            obj = json.load(fh)
        rg = self.__dict__.setdefault("_rg_cache", {})
        for k, d in obj.get("rg", {}).items():
            rg[tuple(json.loads(k))] = {
                tuple(json.loads(lbl)): QNumberPoly({int(n): c for n, c in qo.items()})
                for lbl, qo in d.items()
            }
        mul = self.__dict__.setdefault("_multiply_cache", {})
        for k, eo in obj.get("multiply", {}).items():
            sa, sb = json.loads(k)
            mul[(tuple(sa), tuple(sb))] = Element({
                tuple(json.loads(lbl)): LaurentPoly({int(e): c for e, c in lpo.items()})
                for lbl, lpo in eo.items()
            })

    # ----- linear extension of RG ----------------------------------------

    def RG_element(self, x: Element) -> Element:
        """Linear extension of `RG`: `RG(Σ c_a · L_a) = Σ c_a · RG(L_a)`.

        `Element` is over Z[q^±] universally, so no coefficient-ring
        matching is needed."""
        out: dict[Label, "LaurentPoly"] = {}
        for a, c in x.terms.items():
            if c.is_zero():
                continue
            applied = self.RG(a)
            for b, c_b in applied.terms.items():
                term = c * c_b
                if term.is_zero():
                    continue
                out[b] = out[b] + term if b in out else term
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    # ----- truncated S_RG as auxiliary Element ----------------------------

    def _s_rg_as_aux_element(
        self, cutoff: int, K_expand: int,
    ) -> Element:
        """Truncated `S_RG` as an auxiliary `Element`: each `HabiroElement`
        coefficient expanded to `q^K_expand` (LaurentPoly)."""
        from laurent_poly import LaurentPoly
        s_rg = self.rg_generator(cutoff)
        terms: dict = {}
        for label, h in s_rg.items():
            lp = h.expand(K_expand)
            # Convert to LaurentPoly explicitly (HabiroElement.expand
            # returns the Z[q^±] LaurentPoly type already).
            if not lp.is_zero():
                terms[label] = lp
        return Element(terms)

    # ----- axiom verifiers (RG-side) -------------------------------------

    def verify_rg_unital(self) -> bool:
        """`RG(1_self) = 1_aux`."""
        e_self = self.identity()
        e_aux = self.auxiliary().identity()
        applied = self.RG(e_self)
        return applied == Element.basis(e_aux)

    def verify_rg_multiplicative(self, a, b) -> bool:
        """`RG(L_a · L_b) = RG(L_a) · RG(L_b)`."""
        ab = self.multiply(a, b)
        lhs = self.RG_element(ab)
        rhs = self.auxiliary().multiply_elements(self.RG(a), self.RG(b))
        return lhs == rhs

    def verify_rg_bar_invariant(self, a) -> bool:
        """`bar(RG(L_a)) = RG(L_a)`."""
        applied = self.RG(a)
        return applied.bar() == applied

    def verify_rg_discovery(self, a, K: int = 2) -> bool:
        """Discovery leg of the central RG axiom
        `RG(L_a)·S_RG = X_{γ(a)} + O(q)`: the `q⁰` part of the FS object
        `RG(a)·S_RG` is a **single** auxiliary basis element with coefficient
        `1` (its lower-tropical / apex charge).  This is the generic analogue
        of `BPSKAlgebra.verify_F_S_leading`.

        Leading-order **exact**: it reads the `q⁰` coefficient of
        `rg_times_s_rg(a, K)`, which is complete at any `K ≥ 0` (truncation
        only drops `q^{>K}`).  So — unlike `verify_rg_twist` /
        `verify_rg_inner_product`, whose finite-window strict `==` is
        truncation-fragile — this is a reliable boolean: the `O(q)` tail is
        irrelevant to the claim.
        """
        fs = self.rg_times_s_rg(a, max(K, 0))

        def _q0(c):
            cc = getattr(c, "_coeffs", None)
            if cc is not None:                      # LaurentPoly over Z[q^±]
                return cc.get(0, 0)
            relem = c.coeffs.get(0)                 # RLaurent over R[q^±]
            if relem is None or relem.is_zero():
                return 0
            return relem.terms.get(relem.ring.one_basis(), 0)

        leading = [(lab, _q0(c)) for lab, c in fs.terms.items() if _q0(c) != 0]
        return len(leading) == 1 and leading[0][1] == 1

    def verify_rg_twist(
        self, a, cutoff: int, K_expand: int | None = None,
    ) -> bool:
        """RG-twist axiom: `RG_a · S_RG = S_RG · ρ_aux⁻¹(RG_{ρ_self(a)})`.

        Best-effort: the truncated form fails strict-equality at the
        boundary (filtration cutoff and q-expansion both introduce
        residuals), so this does not certify the axiom at the window edge.
        """
        if K_expand is None:
            K_expand = max(4 * cutoff, 4)
        aux = self.auxiliary()
        s_rg = self._s_rg_as_aux_element(cutoff, K_expand)
        rg_a = self.RG(a)
        lhs = aux.multiply_elements(rg_a, s_rg)
        rho_self_a = self.rho(a)
        rg_rho_a = self.RG(rho_self_a)
        rho_inv_rg = aux.rho_inverse_element(rg_rho_a)
        rhs = aux.multiply_elements(s_rg, rho_inv_rg)
        return lhs == rhs

    def verify_rg_inner_product(
        self, a, b, K: int, cutoff: int,
    ) -> bool:
        """Schur-index transport:
        `I^self_{a,b}(q) = ⟨RG_a · S_RG, RG_b · S_RG⟩_aux(q)`.

        Best-effort, as for `verify_rg_twist`."""
        self_I = self.inner_product(a, b, K)
        aux = self.auxiliary()
        s_rg = self._s_rg_as_aux_element(cutoff, K + cutoff)
        a_S = aux.multiply_elements(self.RG(a), s_rg)
        b_S = aux.multiply_elements(self.RG(b), s_rg)
        rho_a_S = aux.rho_element(a_S)
        prod = aux.multiply_elements(rho_a_S, b_S)
        aux_I = aux.trace_element(prod, K)
        return self_I == aux_I

    # ----- composition ----------------------------------------------------

    def then(self, other: "RGKAlgebra") -> "RGKAlgebra":
        """Compose: `self → other.auxiliary()`.  Requires
        `self.auxiliary() is other`."""
        return ComposedRG(self, other)

    def factor_through(
        self, mid_to_ir: "RGKAlgebra", *, gamma_inclusion=None,
        uv_ms_flow=None,
    ) -> "ExtractedRG":
        """Factor this UV→IR flow through a given MS→IR flow, recovering the
        UV→MS flow (the inverse of `then`).

        `mid_to_ir` must be an `RGKAlgebra` to the IR whose `auxiliary()` is
        **lattice-compatible** with `self.auxiliary()` (same pairing).  The
        returned `ExtractedRG`:

        * `auxiliary()` = the MS algebra (`mid_to_ir.starting_algebra()`),
        * `RG(a)` = the composition-inverse `mid_to_ir⁻¹ ∘ self.RG` — i.e.
          `self.RG(a)` decomposed into `{RG^MS_IR(L_b)}`, the embedding
          `A^UV ↪ A^MS`,
        * `rg_generator(K)` = `S^UV_MS` from
          `S^UV_IR = RG^MS_IR(S^UV_MS) · S^MS_IR`.

        `uv_ms_flow` (optional) is an *authoritative direct* UV→MS flow whose
        `RG`/`rg_generator` are taken as the answer (both accelerator and
        ground truth).  The BPS subquiver convenience
        (`rg_flow.factor_through_subquiver`) supplies the one-shot
        `SubquiverRG(A, outside)`, whose `rg_generator` enumerates `S^UV_MS`
        by **dropped-node multiplicity** with exact `HabiroElement` coefficients — the
        correct truncation measure for the quotient grading `Γ^UV_MS`, so
        outside bound states survive.  When omitted, the general pullback
        extraction runs (invert `S^MS_IR`, multiply by `S^UV_IR`, pull back
        through `RG^MS_IR`, truncate to q-order ≤ K); it raises / does not
        terminate if `self.RG` doesn't factor cleanly through `mid_to_ir`
        (`S^UV_IR` not "of the correct form")."""
        return ExtractedRG(
            self, mid_to_ir, gamma_inclusion=gamma_inclusion,
            uv_ms_flow=uv_ms_flow,
        )


# ---------------------------------------------------------------------------
# ComposedRG: chain two RGKAlgebras
# ---------------------------------------------------------------------------


class ComposedRG(RGKAlgebra):
    """The composition `first.then(second)`: an `RGKAlgebra` whose
    K-algebra identity is `first` (so `multiply`, `rho`, etc. delegate
    to `first`), and whose `auxiliary()` is `second.auxiliary()`.

    Validates `first.auxiliary() is second` at construction.  `RG` is
    function-composition `RG_second ∘ RG_first`.  `rg_generator` is
    intentionally not derived here -- it depends on more than the
    individual factors and is best constructed by a flow-specific
    class for the combined flow.
    """

    def __init__(self, first: RGKAlgebra, second: RGKAlgebra):
        if first.auxiliary() is not second.starting_algebra():
            raise ValueError(
                "ComposedRG: first.auxiliary() must be (identically) "
                "second.starting_algebra() for composition"
            )
        self._first = first
        self._second = second

    # ----- KAlgebra contract delegates to `first` -------------------------

    def coefficient_ring(self):
        return self._first.coefficient_ring()

    def identity(self):
        return self._first.identity()

    def multiply(self, a, b):
        return self._first.multiply(a, b)

    def rho(self, a):
        return self._first.rho(a)

    def rho_inverse(self, a):
        return self._first.rho_inverse(a)

    def trace(self, a, K=20):
        return self._first.trace(a, K)

    def _label_section_decompose(self, label):
        return self._first._label_section_decompose(label)

    def r_label_decompose(self, label):
        return self._first.r_label_decompose(label)

    def r_label_compose(self, section, r_basis_label):
        return self._first.r_label_compose(section, r_basis_label)

    # ----- RGKAlgebra contract -------------------------------------------

    def auxiliary(self):
        return self._second.auxiliary()

    def RG(self, a) -> Element:
        intermediate = self._first.RG(a)
        return self._second.RG_element(intermediate)

    def rg_generator(self, cutoff) -> dict[Label, "HabiroElement"]:
        """Squashed `S_total = inner_RG(S_outer) · S_inner`, where
        `outer = self._first` is the source step and
        `inner = self._second` is the second step.

        Computes:
          1. `S_outer_in_mid = first.rg_generator(cutoff)` (`HabiroElement`
             coefficients in the middle algebra);
          2. `S_outer_in_aux = inner_RG(S_outer_in_mid)` -- apply
             `second.RG` label-by-label, accumulating `HabiroElement`
             coefficients in the final auxiliary;
          3. `S_inner_in_aux = second.rg_generator(cutoff)` (`HabiroElement`
             coefficients in the final aux);
          4. multiply `S_outer_in_aux · S_inner_in_aux` in the final
             auxiliary, using `aux.multiply` to get structure constants
             and lifting the coefficients into the localized ring.

        Flavour-agnostic: the structure constants are integer
        `LaurentPoly` (flavour rides in the *labels*, not the coefficients),
        so the localized-ring arithmetic stays over `Z` — no `R`-coefficient
        variant is needed.  Validated on the flavoured hexagon (matches the
        one-shot `SubquiverRG`).
        """
        aux = self._second.auxiliary()
        s_outer_mid = self._first.rg_generator(cutoff)
        s_outer_aux = _apply_rg_to_habiro_dict(self._second, s_outer_mid)
        s_inner_aux = self._second.rg_generator(cutoff)
        return _multiply_habiro_dicts(s_outer_aux, s_inner_aux, aux)

    @property
    def first(self) -> RGKAlgebra:
        return self._first

    @property
    def second(self) -> RGKAlgebra:
        return self._second


# ---------------------------------------------------------------------------
# ExtractedRG: factor a UV→IR flow through a given MS→IR flow
# ---------------------------------------------------------------------------


def _pairing_of(alg: KAlgebra):
    """The antisymmetric pairing of a (quantum-torus-like) auxiliary, or
    `None`.  Handles both `BPSKAlgebra` (`.lattice.pairing`) and
    `QuantumTorusKAlg` (`.pairing`)."""
    lat = getattr(alg, "lattice", None)
    if lat is not None and hasattr(lat, "pairing"):
        return [list(r) for r in lat.pairing]
    p = getattr(alg, "pairing", None)
    if p is not None:
        return [list(r) for r in p]
    return None


def _aux_lattice_compatible(a: KAlgebra, b: KAlgebra) -> bool:
    """Two auxiliaries are compatible for `factor_through` iff they live on
    the same lattice (same pairing).  Identity (`is`) is the fast path; two
    quantum tori / BPS auxiliaries with equal pairings are distinct
    instances but compatible."""
    if a is b:
        return True
    pa, pb = _pairing_of(a), _pairing_of(b)
    return pa is not None and pa == pb


def _pullback_laurent_image(
    x_elem: Element,
    flow: "RGKAlgebra",
    height_grading: "Grading",
    *,
    max_iter: int = 1_000_000,
) -> Element:
    """`Element` (LaurentPoly) version of `_habiro_from_ir_image`: express
    `x_elem` (in `flow.auxiliary()`) as `Σ_c c_c · RG^flow(L_c)` and return
    `Σ_c c_c · L_c` over `flow`'s source labels.  Cone-minimal apex peel
    ordered by `height_grading.height_of` (the UV→IR height — total on all
    appearing charges).  This is the `RG` of an `ExtractedRG` (the
    composition-inverse / embedding)."""
    x = {l: c for l, c in x_elem.terms.items() if not c.is_zero()}
    out: dict[Label, "LaurentPoly"] = {}
    it = 0
    while x:
        it += 1
        if it > max_iter:
            raise RuntimeError(
                "factor_through pullback did not terminate; S^UV_IR may not "
                "factor through this MS→IR flow (not 'of the correct form')."
            )
        delta = min(x, key=lambda l: height_grading.height_of(l))
        c = flow._apex_inverse(delta)
        coeff = x[delta]                          # apex coeff of RG(c) is 1
        out[c] = (out[c] + coeff) if c in out else coeff
        for d, cc in flow.RG(c).terms.items():
            term = coeff * cc
            if term.is_zero():
                continue
            nv = (x[d] - term) if d in x else -term
            if nv.is_zero():
                x.pop(d, None)
            else:
                x[d] = nv
    return Element({l: c for l, c in out.items() if not c.is_zero()})


class ExtractedRG(RGKAlgebra):
    """The UV→MS flow recovered by `uv_ir.factor_through(mid_to_ir)`.

    KAlgebra ops delegate to the **UV** algebra
    (`uv_ir.starting_algebra()`); `auxiliary()` is the **MS** algebra
    (`mid_to_ir.starting_algebra()`).  `RG` is the composition-inverse
    `mid_to_ir⁻¹ ∘ uv_ir.RG` (the embedding `A^UV ↪ A^MS`); `rg_generator`
    is the extracted `S^UV_MS`.  `grading()` (the quotient
    `Γ^UV_MS = Γ^UV_IR / Γ^MS_IR`) requires `gamma_inclusion` — the core
    `RG`/`rg_generator`/embedding do not need it.
    """

    def __init__(self, uv_ir: RGKAlgebra, mid_to_ir: RGKAlgebra,
                 *, gamma_inclusion=None, uv_ms_flow=None):
        if not _aux_lattice_compatible(
            uv_ir.auxiliary(), mid_to_ir.auxiliary()
        ):
            raise ValueError(
                "factor_through: uv_ir.auxiliary() and mid_to_ir.auxiliary() "
                "must be lattice-compatible (same pairing)."
            )
        self._uv_ir = uv_ir
        self._mid = mid_to_ir
        self._iota = gamma_inclusion
        # Optional *authoritative direct* UV→MS flow (an RGKAlgebra whose
        # `auxiliary()` is label-compatible with `mid_to_ir.starting_algebra()`).
        # When present, `RG` and `rg_generator` delegate to it: for the BPS
        # subquiver convenience this is the one-shot `SubquiverRG(A, outside)`,
        # which computes `S^UV_MS` exactly — enumerated by dropped-node
        # multiplicity, so outside bound states `γ_a+γ_b` survive regardless of
        # their q-order (no q-order truncation — compute exact, truncate last
        # by the right measure).  `None` ⟹ run the general pullback
        # extraction below.
        self._uv_ms_flow = uv_ms_flow

    def _uv(self) -> KAlgebra:
        return self._uv_ir.starting_algebra()

    # ----- KAlgebra contract delegates to the UV algebra ------------------

    def coefficient_ring(self):
        return self._uv().coefficient_ring()

    def identity(self):
        return self._uv().identity()

    def multiply(self, a, b):
        return self._uv().multiply(a, b)

    def rho(self, a):
        return self._uv().rho(a)

    def rho_inverse(self, a):
        return self._uv().rho_inverse(a)

    def trace(self, a, K=20):
        return self._uv().trace(a, K)

    def _label_section_decompose(self, label):
        return self._uv()._label_section_decompose(label)

    def r_label_decompose(self, label):
        return self._uv().r_label_decompose(label)

    def r_label_compose(self, section, r_basis_label):
        return self._uv().r_label_compose(section, r_basis_label)

    def starting_algebra(self):
        return self._uv()

    # ----- RGKAlgebra contract -------------------------------------------

    def auxiliary(self):
        return self._mid.starting_algebra()

    def RG(self, a) -> Element:
        """`RG^UV_MS(a)` = `self._uv_ir.RG(a)` decomposed into
        `{RG^MS_IR(L_b)}` (the embedding `A^UV ↪ A^MS`).

        When an authoritative direct UV→MS flow is supplied (the BPS
        subquiver convenience), delegate to it — `SubquiverRG.RG` decomposes
        the UV `F_a` straight into the MS canonical basis by cone-minimal
        peeling, the same embedding computed exactly."""
        if self._uv_ms_flow is not None:
            return self._uv_ms_flow.RG(a)
        return _pullback_laurent_image(
            self._uv_ir.RG(a), self._mid, self._uv_ir.grading(),
        )

    def rg_generator(self, cutoff) -> dict[Label, "HabiroElement"]:
        """`S^UV_MS` from `S^UV_IR = RG^MS_IR(S^UV_MS) · S^MS_IR`.

        If an authoritative direct UV→MS flow was supplied, delegate to it
        (the BPS subquiver convenience — `S^UV_MS` enumerated by dropped-node
        multiplicity, exact, so outside bound states survive).  Otherwise run
        the general extract: invert `S^MS_IR`, multiply by `S^UV_IR`, pull
        back through `RG^MS_IR`, truncate to leading q-order ≤ cutoff."""
        if self._uv_ms_flow is not None:
            return self._uv_ms_flow.rg_generator(cutoff)
        aux = self._mid.auxiliary()
        s_uv_ir = self._uv_ir.rg_generator(cutoff)
        s_ms_ir = self._mid.rg_generator(cutoff)
        inv = _invert_habiro_spectrum(s_ms_ir, aux, cutoff)
        t = _multiply_habiro_dicts(s_uv_ir, inv, aux)
        s_uv_ms = _habiro_from_ir_image(t, self._mid, self._uv_ir.grading())
        return _truncate_habiro_dict_by_qorder(s_uv_ms, cutoff)

    def _s_rg_component(self, p):
        """`[S^UV_MS]_p` — exact graded component.

        When an authoritative direct UV→MS flow is supplied (the BPS subquiver
        convenience), delegate to its oracle; that is the exact `S^UV_MS`
        enumerated by the quotient `Γ^UV_MS` grading.  The general (pullback)
        case would use the grading-height extract path, which is not
        implemented, so it falls through to the soft-contract
        `NotImplementedError`."""
        if self._uv_ms_flow is not None:
            return self._uv_ms_flow._s_rg_component(p)
        return super()._s_rg_component(p)

    def grading(self):
        """The quotient grading `Γ^UV_MS = Γ^UV_IR / Γ^MS_IR`.

        `gamma_inclusion` (passed to `factor_through`) gives `Γ^MS_IR` as a
        list of **coordinate indices** into `self._uv_ir.grading()` (the
        directions `mid_to_ir` integrates out — the subquiver coords in the
        BPS case).  `deg^UV_MS(b) = deg^UV_IR(b)` with those coordinates
        dropped (the quotient); the height restricts to the complementary
        coordinates (`h^UV_IR` on a complement of `Γ^MS_IR` — the UV→MS
        height).  Both are total and additive (a coordinate
        projection of the additive `deg^UV_IR`).  A general (non-coordinate)
        `Γ^MS_IR ↪ Γ^UV_IR` (via SNF) is not implemented.
        """
        if self._iota is None:
            raise NotImplementedError(
                "ExtractedRG.grading() needs gamma_inclusion (the Γ^MS_IR "
                "coordinate indices in Γ^UV_IR's grading); pass it to "
                "factor_through (the subquiver convenience supplies it)."
            )
        from grading import Grading
        uvg = self._uv_ir.grading()
        ms = set(self._iota)
        keep = [i for i in range(uvg.rank) if i not in ms]
        if not keep:
            raise ValueError(
                "quotient grading is rank 0 (Γ^MS_IR = Γ^UV_IR); the MS→IR "
                "flow integrates out everything the UV→IR flow does."
            )

        def _deg(label):
            full = uvg.charge(self._mid.apex(label))
            return tuple(full[i] for i in keep)

        return Grading(
            rank=len(keep), deg=_deg,
            height=tuple(uvg.height[i] for i in keep),
        )

    def verify_round_trip(self, cutoff: int) -> bool:
        """`extract ∘ combine = id`: recombining the
        recovered `UV→MS` flow with the `MS→IR` flow reproduces the original
        `UV→IR` spectrum generator to leading q-order ≤ cutoff —
        `(self.then(mid)).rg_generator ≡ uv_ir.rg_generator (mod q^{>cutoff})`.
        """
        from habiro import HabiroElement
        recombined = _truncate_habiro_dict_by_qorder(
            self.then(self._mid).rg_generator(cutoff), cutoff,
        )
        original = _truncate_habiro_dict_by_qorder(
            self._uv_ir.rg_generator(cutoff), cutoff,
        )
        return all(
            recombined.get(k, HabiroElement.zero())
            == original.get(k, HabiroElement.zero())
            for k in set(recombined) | set(original)
        )

    @property
    def uv_ir(self) -> RGKAlgebra:
        return self._uv_ir

    @property
    def mid(self) -> RGKAlgebra:
        return self._mid


# ---------------------------------------------------------------------------
# JSON save / load for the BPS-realization concrete subclasses
# ---------------------------------------------------------------------------


def _bpskalgebra_data(A) -> dict:
    """Defining data for a `BPSKAlgebra`, JSON-friendly."""
    return {
        "pairing": [list(row) for row in A.lattice.pairing],
        "node_charges": [list(g) for g in A.node_charges],
        "spec": [list(g) for g in A.spec],
    }


def _bpskalgebra_from_data(data: dict):
    raise NotImplementedError(  # requires the BPS realisation layer
        "BPS/rg_flow (de)serialization requires the BPS realisation layer; "
        "not available in this configuration")
    return BPSKAlgebra(
        pairing=data["pairing"],
        node_charges=data["node_charges"],
        spec=data["spec"],
        verify="off",
    )


def _flatten_chain(rg) -> list[dict]:
    """Flatten a `RGKAlgebra` (possibly a nested `ComposedRG`) into a
    linear list of single-step descriptors, root step first."""
    raise NotImplementedError(  # requires the BPS realisation layer
        "BPS/rg_flow (de)serialization requires the BPS realisation layer; "
        "not available in this configuration")
    if isinstance(rg, ComposedRG):
        return _flatten_chain(rg.first) + _flatten_chain(rg.second)
    # SingleNodeRG is a subclass of SubquiverRG, so check it first.
    if isinstance(rg, SingleNodeRG):
        return [{"kind": "SingleNodeRG", "node_index": int(rg._j)}]
    if isinstance(rg, SubquiverRG):
        return [{"kind": "SubquiverRG", "node_indices": list(rg.node_indices)}]
    raise TypeError(
        f"_flatten_chain: don't know how to flatten {type(rg).__name__}"
    )


def to_json(rg: RGKAlgebra) -> dict:
    """Serialize a `SingleNodeRG`, `SubquiverRG`, or `ComposedRG` chain
    to a JSON-friendly dict.

    Format: a `root_uv` (the UV `BPSKAlgebra`'s defining data) plus a
    linear list of `steps`.  Composed flows are flattened: the chain
    `first.then(second).then(third)` serialises as a 3-step list.

    The IR auxiliaries between steps are not stored -- they are
    derived on `from_json` by chaining each step's `auxiliary()`
    into the next step's UV.
    """
    raise NotImplementedError(  # requires the BPS realisation layer
        "BPS/rg_flow (de)serialization requires the BPS realisation layer; "
        "not available in this configuration")
    if isinstance(rg, (SingleNodeRG, SubquiverRG, ComposedRG)):
        # Walk to the leftmost UV.
        cur = rg
        while isinstance(cur, ComposedRG):
            cur = cur.first
        if not isinstance(cur, (SingleNodeRG, SubquiverRG)):
            raise TypeError(
                f"to_json: leftmost factor is {type(cur).__name__}, "
                f"expected SingleNodeRG or SubquiverRG"
            )
        return {
            "root_uv": _bpskalgebra_data(cur.uv_algebra),
            "steps": _flatten_chain(rg),
        }
    raise TypeError(
        f"to_json: don't know how to serialize {type(rg).__name__}"
    )


def from_json(data: dict) -> RGKAlgebra:
    """Reconstruct a `RGKAlgebra` previously saved by `to_json`.

    Linearly walks `data["steps"]`, threading each step's auxiliary
    into the next step's UV so `ComposedRG`'s endpoint identity is
    satisfied.  Returns the single-step `RGKAlgebra` for length-1
    chains, else a left-leaning `ComposedRG` for longer ones.
    """
    raise NotImplementedError(  # requires the BPS realisation layer
        "BPS/rg_flow (de)serialization requires the BPS realisation layer; "
        "not available in this configuration")
    if "root_uv" not in data or "steps" not in data:
        raise ValueError(
            "from_json: data must have 'root_uv' and 'steps' keys"
        )
    steps = data["steps"]
    if not steps:
        raise ValueError("from_json: empty steps list")
    A = _bpskalgebra_from_data(data["root_uv"])
    rg: RGKAlgebra | None = None
    for step in steps:
        kind = step.get("kind")
        if kind == "SingleNodeRG":
            new_rg = SingleNodeRG(A, step["node_index"])
        elif kind == "SubquiverRG":
            new_rg = SubquiverRG(A, step["node_indices"])
        else:
            raise ValueError(
                f"from_json: unknown step kind {kind!r}; expected "
                f"'SingleNodeRG' or 'SubquiverRG'"
            )
        rg = new_rg if rg is None else ComposedRG(rg, new_rg)
        A = new_rg.auxiliary()
    return rg
