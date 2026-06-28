"""Directional node-drop RG presentations (Plan 20 T6 / decisions A8).

`DirectionalSubquiverRG` is the **directional** counterpart of the
UV-wrapping extractors `rg_flow.SingleNodeRG` / `rg_flow.SubquiverRG`:
it *defines* the UV K-algebra of a BPS node-deletion RG flow purely from
IR + RG data, inheriting the complete generic `RGKAlgebra` API
(`multiply` via the apex peel, `rho`/`rho_inverse` via the `tRG` mirror,
`trace` via RG transport).  No UV `BPSKAlgebra` appears anywhere in the
derived path — so a `KAlgebraIso` between an independently built UV
`BPSKAlgebra` and this class is a genuine cross-presentation
certificate, not a delegation tautology (the circularity that made
"iso" checks against the wrap-UV extractors vacuous).

The spectrum generator `S_RG` — three ways (the user's a/b/c)
--------------------------------------------------------------
* **(a) Closed form (default).**  When the UV spec is arranged
  **stuff-first** — `spec = [stuff_1, …, stuff_k, ir_1, …, ir_m]`, the
  `stuff_i` carrying the dropped node(s) in their node-basis
  decomposition, the `ir_j` not — the operator factorization is
  immediate:

      S_UV  =  S_RG · S_IR,      S_RG  =  ∏_{i=1..k} E_𝖖(X_{stuff_i}),

  the literal head of the UV spec product.  Its **dict representation
  over the IR auxiliary** (Habiro coefficients on IR *canonical*
  labels, the form the `RGKAlgebra` contract consumes) is computed
  chart-free from constructor data: the Nahm kets `[S_UV|0⟩]_δ` of the
  stuff-first spec (`nahm_local.s_gamma_habiro` — a closed form in
  `(spec, pairing)`) are peeled against the IR's F-elements,
  `state[η] −= s_δ · [F^IR_δ · S_IR|0⟩]_η`, walking the dropped-charge
  cone minimally.  Everything runs on the cheap IR side; **no UV
  `BPSKAlgebra` / UV F-solver is ever constructed.**  Each stuff factor
  carries ≥ 1 dropped multiplicity, so every graded component
  `[S_RG]_p` is a finite exact sum — the `_s_rg_component` oracle the
  exact solver consumes.
* **(b) Local-move arrangement.**  When the supplied spec is not
  stuff-first, `arrange_spec_stuff_first` runs a BFS over the
  S-preserving local moves (adjacent commute swaps `⟨g, g'⟩ = 0`,
  pentagon collapses `[b, a+b, a] → [a, b]` and expansions
  `[a, b] → [b, a+b, a]` at `⟨a, b⟩ = +1`) to reach a stuff-first
  presentation of the *same* spectrum generator.  Local moves preserve
  `S` exactly, hence the chamber, the canonical basis, and the labels —
  the identity-on-labels iso survives the rearrangement (this is what
  rules out the chamber-calibration trap of mutation-based
  rearrangement).  Pass `arrange=True` to do this in the constructor.
* **(c) Oracle injection.**  An externally known `S_RG` — a hand
  factorization of `S = S_RG · S_IR` order-by-order along the positive
  cone, or the extraction the UV-wrapping classes already perform
  (`SubquiverRG._s_rg_component` peels UV Nahm states against IR
  F-elements) — is supplied as `s_rg_oracle(p) → {IR label:
  HabiroElement}`, replacing the closed form.

F-oracle
--------
`f_oracle(label) → Element` optionally supplies RG images (e.g. from a
UV `BPSKAlgebra`'s F-solver via `uv_f_oracle`) for speed; the generic
solve is then skipped for those labels.  `rho_inverse` still derives
`ρ_UV⁻¹` from the exact mirror solve — and when it does, it
cross-checks the oracle's `RG(a)` against the solved one, so a wrong
oracle fails loudly instead of poisoning the presentation.

Certification
-------------
`certify_directional_vs_bps(direct, uv_bps, …)` runs the staged
battery: flow coherence first (`verify_rg_unital` /
`verify_rg_multiplicative` / `verify_rg_bar_invariant` — catches a
mis-specified flow, e.g. a relabel masquerading as RG, *before* any
cross-presentation comparison), then the `KAlgebraIso` battery on the
identity-on-labels iso (round-trip, multiplicativity, ρ-equivariance,
optionally trace-equivariance — the latter two newly meaningful here,
since neither ρ nor trace is delegated).
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from typing import Callable, Iterable, Optional, Sequence

from kalgebra import KAlgebra, Element
from laurent_poly import LaurentPoly
from rgkalgebra import RGKAlgebra
from grading import Grading
from kalgebra_iso import KAlgebraIso

Vec = tuple[int, ...]


# ---------------------------------------------------------------------------
# Node-basis decomposition (total, integer — negatives allowed)
# ---------------------------------------------------------------------------


def _completed_row_transform(cols: Sequence[Vec]):
    """For the `R×C` matrix `M` with columns `cols` (`C < R`,
    independent, **saturated** span): a unimodular `U ∈ GL_R(Z)` with
    `U·M = [I_C; 0]`.  Then `U·label` is the integer coordinate vector
    of `label` in the completed basis (node charges ⊔ a complement of
    their span): first `C` coordinates = node coefficients.  Returns
    `None` when the columns are dependent or the span is not saturated
    (some pivot ≠ 1) — callers fall back to the Fraction solver."""
    R, C = len(cols[0]), len(cols)
    M = [[cols[j][i] for j in range(C)] for i in range(R)]
    U = [[1 if i == j else 0 for j in range(R)] for i in range(R)]

    def add_row(src: int, dst: int, k: int) -> None:
        if k:
            for j in range(C):
                M[dst][j] += k * M[src][j]
            for j in range(R):
                U[dst][j] += k * U[src][j]

    def swap_rows(i1: int, i2: int) -> None:
        M[i1], M[i2] = M[i2], M[i1]
        U[i1], U[i2] = U[i2], U[i1]

    for c in range(C):
        # Euclid column c down to a single pivot at row c.
        while True:
            piv = None
            for i in range(c, R):
                if M[i][c] != 0 and (
                        piv is None or abs(M[i][c]) < abs(M[piv][c])):
                    piv = i
            if piv is None:
                return None                     # dependent columns
            if piv != c:
                swap_rows(c, piv)
            done = True
            for i in range(c + 1, R):
                if M[i][c] != 0:
                    add_row(c, i, -(M[i][c] // M[c][c]))
                    if M[i][c] != 0:
                        done = False
            if done:
                break
        if M[c][c] < 0:
            M[c] = [-x for x in M[c]]
            U[c] = [-x for x in U[c]]
        if M[c][c] != 1:
            return None                         # span not saturated
        for i in range(c):                      # clear above the pivot
            add_row(c, i, -M[i][c])
    return U


def _node_basis_solver(node_charges: Sequence[Vec]):
    """A callable `label → integer node-basis coefficients` (negatives
    allowed), for node charges forming an **independent** family — not
    necessarily a square basis (flavoured quivers and iterated drops
    have fewer nodes than the lattice rank).

    Square + unimodular → precomputed exact integer inverse (total on
    the whole lattice).  `C < R` with a **saturated** span → the basis
    is completed with a complement of the span (one tracked unimodular
    row reduction `U·M = [I_C; 0]`; coordinates are then just
    `U·label`), total on the whole lattice — the first `C` coordinates
    are the node coefficients, the rest the complement (e.g. a gauged
    quiver's frozen flavour direction, RG-inert through the flow; the
    grading reads only the dropped-node coordinates, so off-span
    labels like `E^{±1}` score correctly).  Remaining general case
    (non-saturated span) → Fraction Gaussian elimination per label,
    total on the **integer span** only; labels off the span (or with
    non-integer coefficients) raise `ValueError`.  Totality on every
    label the generic machinery touches (ρ-mirror images, products of
    RG images) is what the directional grading needs — the old
    non-negative-only decomposition raised on all of those."""
    from fractions import Fraction
    cols = [tuple(int(x) for x in g) for g in node_charges]
    if not cols:
        raise ValueError("DirectionalSubquiverRG: empty node-charge set")
    R = len(cols[0])
    C = len(cols)
    if any(len(g) != R for g in cols):
        raise ValueError(
            f"DirectionalSubquiverRG: node charges have mixed lengths "
            f"{[len(g) for g in cols]}."
        )
    if C == R:
        from snf_kernel import int_det as _int_det
        M = [[cols[j][i] for j in range(C)] for i in range(R)]
        if _int_det([list(r) for r in M]) in (-1, 1):
            from bpskalgebra_kalgebra_iso import _invert_unimodular
            inv_rows = _invert_unimodular(M)

            def solve_sq(label: Vec) -> Vec:
                return tuple(
                    sum(row[j] * label[j] for j in range(R))
                    for row in inv_rows
                )

            return solve_sq

    if C < R:
        U = _completed_row_transform(cols)
        if U is not None:

            def solve_completed(label: Vec) -> Vec:
                label = tuple(int(x) for x in label)
                if len(label) != R:
                    raise ValueError(
                        f"label {label} has length {len(label)}, "
                        f"expected {R}"
                    )
                return tuple(
                    sum(U[i][j] * label[j] for j in range(R))
                    for i in range(R)
                )

            return solve_completed

    def solve_gen(label: Vec) -> Vec:
        label = tuple(int(x) for x in label)
        if len(label) != R:
            raise ValueError(
                f"label {label} has length {len(label)}, expected {R}"
            )
        # Augmented [M | v] over Fractions; M columns = node charges.
        aug = [
            [Fraction(cols[j][i]) for j in range(C)] + [Fraction(label[i])]
            for i in range(R)
        ]
        piv_rows: list[int] = []
        r = 0
        for c in range(C):
            p = next((i for i in range(r, R) if aug[i][c] != 0), None)
            if p is None:
                raise ValueError(
                    f"DirectionalSubquiverRG: node charges are not "
                    f"independent (no pivot for column {c})."
                )
            aug[r], aug[p] = aug[p], aug[r]
            pv = aug[r][c]
            aug[r] = [x / pv for x in aug[r]]
            for i in range(R):
                if i != r and aug[i][c] != 0:
                    f = aug[i][c]
                    aug[i] = [x - f * y for x, y in zip(aug[i], aug[r])]
            piv_rows.append(r)
            r += 1
        for i in range(r, R):
            if aug[i][C] != 0:
                raise ValueError(
                    f"label {label} is not in the integer span of the "
                    f"node charges."
                )
        coeffs = []
        for c in range(C):
            x = aug[piv_rows[c]][C]
            if x.denominator != 1:
                raise ValueError(
                    f"label {label} has non-integer node-basis "
                    f"coefficient {x} at column {c}."
                )
            coeffs.append(int(x))
        return tuple(coeffs)

    return solve_gen


def _node_basis_inverse(node_charges: Sequence[Vec]):
    """Back-compat shim: returns the solver from `_node_basis_solver`
    wrapped so existing internal callers can keep the
    `_decompose_in_node_basis(inv, label)` calling shape."""
    return _node_basis_solver(node_charges)


def _decompose_in_node_basis(solver, label: Vec) -> Vec:
    """Integer node-basis coefficients of `label` via the solver from
    `_node_basis_solver` (negatives allowed; raises off the span)."""
    return solver(tuple(label))


def _rational_node_basis_decompose(node_charges: Sequence[Vec], label: Vec):
    """**Rational** (Fraction) node-basis coefficients of `label`, for the case
    Γ ⊋ node-span: gauge-theory quivers whose node charges span a finite-index
    sublattice of Γ (e.g. pure SU(2)'s standard `(1,0),(-1,2)`, index 2, whose
    Wilson `(0,1)` and monopole are half-elements).  No integrality requirement —
    raises only if `label` is not in the *rational* span (node charges dependent).
    Used by `_deg` to read off the dropped-index RG degree even when the
    non-dropped (gauge) coefficients are half-integral."""
    from fractions import Fraction
    cols = [tuple(g) for g in node_charges]
    C = len(cols)
    R = len(cols[0]) if cols else 0
    lab = [Fraction(int(x)) for x in label]
    aug = [[Fraction(cols[j][i]) for j in range(C)] + [lab[i]] for i in range(R)]
    piv_rows: list[int] = []
    r = 0
    for c in range(C):
        p = next((i for i in range(r, R) if aug[i][c] != 0), None)
        if p is None:
            raise ValueError(
                f"node charges are not independent (no pivot for column {c}).")
        aug[r], aug[p] = aug[p], aug[r]
        pv = aug[r][c]
        aug[r] = [x / pv for x in aug[r]]
        for i in range(R):
            if i != r and aug[i][c] != 0:
                f = aug[i][c]
                aug[i] = [x - f * y for x, y in zip(aug[i], aug[r])]
        piv_rows.append(r)
        r += 1
    for i in range(r, R):
        if aug[i][C] != 0:
            raise ValueError(
                f"label {tuple(label)} is not in the rational span of the node "
                f"charges.")
    return tuple(aug[piv_rows[c]][C] for c in range(C))


# ---------------------------------------------------------------------------
# Stuff / IR split + the local-move arranger (version (b))
# ---------------------------------------------------------------------------


def classify_split(
    node_charges: Sequence[Vec],
    spec: Sequence[Vec],
    drop_indices: Sequence[int],
) -> tuple[list[Vec], list[Vec]]:
    """Order-preserving classification of spec entries into
    `(stuff, ir_spec)` by dropped-charge content, with **no positional
    requirement** — the unguarded split behind
    `require_stuff_first=False` (the version-(c) mode).  Entries with
    negative node-basis decompositions raise."""
    inv = _node_basis_inverse([tuple(g) for g in node_charges])
    stuff: list[Vec] = []
    ir_spec: list[Vec] = []
    for g in spec:
        g = tuple(g)
        dec = _decompose_in_node_basis(inv, g)
        if any(dec[j] < 0 for j in range(len(dec))):
            raise ValueError(
                f"classify_split: spec entry {g} decomposes with negative "
                f"coefficients {dec} in the node basis."
            )
        if any(dec[j] > 0 for j in drop_indices):
            stuff.append(g)
        else:
            ir_spec.append(g)
    return stuff, ir_spec


def stuff_split(
    node_charges: Sequence[Vec],
    spec: Sequence[Vec],
    drop_indices: Sequence[int],
) -> tuple[list[Vec], list[Vec]]:
    """Split a spec into `(stuff, ir_spec)`, *requiring* the stuff-first
    arrangement: every entry whose node-basis decomposition meets a
    dropped index precedes every entry that doesn't.  Raises on any
    interleaving (use `arrange_spec_stuff_first` / `arrange=True`)."""
    inv = _node_basis_inverse([tuple(g) for g in node_charges])
    stuff: list[Vec] = []
    ir_spec: list[Vec] = []
    seen_ir = False
    for g in spec:
        g = tuple(g)
        dec = _decompose_in_node_basis(inv, g)
        if any(dec[j] < 0 for j in range(len(dec))):
            raise ValueError(
                f"stuff_split: spec entry {g} decomposes with negative "
                f"coefficients {dec} in the node basis."
            )
        if any(dec[j] > 0 for j in drop_indices):
            if seen_ir:
                raise ValueError(
                    f"stuff_split: spec entry {g} (carries a dropped node) "
                    f"appears after an IR-only entry; the required "
                    f"arrangement is [stuff…, ir…].  Re-arrange via local "
                    f"moves (arrange_spec_stuff_first / arrange=True)."
                )
            stuff.append(g)
        else:
            seen_ir = True
            ir_spec.append(g)
    return stuff, ir_spec


def _is_stuff_first(
    spec: tuple[Vec, ...],
    inv_rows: Sequence[Sequence[int]],
    drop_indices: Sequence[int],
) -> bool:
    seen_ir = False
    for g in spec:
        dec = _decompose_in_node_basis(inv_rows, g)
        if any(dec[j] < 0 for j in range(len(dec))):
            return False
        if any(dec[j] > 0 for j in drop_indices):
            if seen_ir:
                return False
        else:
            seen_ir = True
    return True


def _bracket(g1: Vec, g2: Vec, B: Sequence[Sequence[int]]) -> int:
    n = len(B)
    return sum(g1[a] * B[a][b] * g2[b] for a in range(n) for b in range(n))


def _local_move_neighbours(
    spec: tuple[Vec, ...],
    B: Sequence[Sequence[int]],
    max_len: int,
) -> Iterable[tuple[Vec, ...]]:
    """All specs reachable from `spec` by ONE S-preserving local move:
    adjacent commute swap (`⟨g, g'⟩ = 0`), pentagon collapse
    (`[b, a+b, a] → [a, b]`, `⟨a, b⟩ = +1`), pentagon expansion
    (`[a, b] → [b, a+b, a]`, `⟨a, b⟩ = +1`, length-capped)."""
    L = len(spec)
    for i in range(L - 1):
        x, y = spec[i], spec[i + 1]
        br = _bracket(x, y, B)
        if br == 0 and x != y:
            yield spec[:i] + (y, x) + spec[i + 2:]
        # Expansion [a, b] → [b, a+b, a] needs ⟨a, b⟩ = +1 with a = x, b = y.
        if br == 1 and L + 1 <= max_len:
            ab = tuple(xa + ya for xa, ya in zip(x, y))
            yield spec[:i] + (y, ab, x) + spec[i + 2:]
    for i in range(L - 2):
        b, ab, a = spec[i], spec[i + 1], spec[i + 2]
        if all(ab[k] == a[k] + b[k] for k in range(len(ab))) \
                and _bracket(a, b, B) == 1:
            yield spec[:i] + (a, b) + spec[i + 3:]


def arrange_spec_stuff_first(
    pairing: Sequence[Sequence[int]],
    node_charges: Sequence[Vec],
    spec: Sequence[Vec],
    drop_indices: Sequence[int],
    *,
    max_states: int = 20000,
    max_len: int | None = None,
) -> list[Vec] | None:
    """BFS over S-preserving local moves to a **stuff-first** spec
    (version (b) of the S_RG recipe).  Returns the arranged spec, or
    `None` if none is reachable within the budget.  Because every move
    preserves `S` exactly, the returned spec presents the *same*
    chamber — canonical labels are untouched, so no label transport is
    needed downstream."""
    node_charges = [tuple(g) for g in node_charges]
    inv = _node_basis_inverse(node_charges)
    start = tuple(tuple(g) for g in spec)
    if max_len is None:
        max_len = len(start) + 6
    if _is_stuff_first(start, inv, drop_indices):
        return [tuple(g) for g in start]
    seen = {start}
    frontier = [start]
    while frontier and len(seen) < max_states:
        nxt: list[tuple[Vec, ...]] = []
        for s in frontier:
            for t in _local_move_neighbours(s, pairing, max_len):
                if t in seen:
                    continue
                if _is_stuff_first(t, inv, drop_indices):
                    return [tuple(g) for g in t]
                seen.add(t)
                nxt.append(t)
                if len(seen) >= max_states:
                    break
            if len(seen) >= max_states:
                break
        frontier = nxt
    return None


# ---------------------------------------------------------------------------
# The directional node-drop RGKAlgebra
# ---------------------------------------------------------------------------


class DirectionalSubquiverRG(RGKAlgebra):
    """Directional `RGKAlgebra` for the deletion of a set of BPS-quiver
    nodes: supplies only the flow data — `auxiliary()` (the IR
    `BPSKAlgebra` on the surviving nodes), `grading()` (dropped-node
    multiplicities, total on the lattice), and `S_RG` (closed-form
    stuff-product by default; injectable) — and inherits the complete
    generic K-algebra API.  See the module docstring.

    Constructor
    -----------
        DirectionalSubquiverRG(pairing, node_charges, spec, drop, *,
                               arrange=False, s_rg_oracle=None,
                               f_oracle=None, ir_verify="off")

    * `pairing` — antisymmetric Z-pairing on the shared lattice Γ
      (non-degenerate; flavoured theories gauge-projected upstream, as
      for `BPSKAlgebra`).
    * `node_charges` — UV BPS-quiver node charges: a unimodular basis,
      or any independent saturated set (`C < R`; the grading then
      decomposes labels in the completed basis node charges ⊔ span
      complement — e.g. a gauged quiver's frozen flavour direction,
      which is RG-inert and rides through the flow).
    * `spec` — UV spec.  Must be stuff-first unless `arrange=True`.
    * `drop` — the node(s) to drop: indices into `node_charges`, or the
      charges themselves.
    * `arrange` — run `arrange_spec_stuff_first` on the spec first.
    * `require_stuff_first` — default True (raise on interleaved
      specs).  `False` runs the same ket-peel on the spec as given
      (order-by-order factorization, version (c)): whether that always
      yields a valid flow is an open question — certify the result.
    * `s_rg_oracle` — optional `p → {IR label: HabiroElement}` exact
      graded `[S_RG]_p` (version (c) by injection); default is the
      stuff-product closed form (version (a)).
    * `f_oracle` — optional `label → Element` RG-image oracle (cf.
      `uv_f_oracle`); cross-checked whenever the mirror solve runs.
    """

    def __init__(
        self,
        pairing: Sequence[Sequence[int]],
        node_charges: Sequence[Vec],
        spec: Sequence[Vec],
        drop: Sequence,
        *,
        arrange: bool = False,
        require_stuff_first: bool = True,
        s_rg_oracle: Optional[Callable[[Vec], dict]] = None,
        f_oracle: Optional[Callable[[Vec], Element]] = None,
        ir_verify: str = "off",
        rg_window: int | None = None,
    ) -> None:
        node_charges = [tuple(g) for g in node_charges]
        spec = [tuple(g) for g in spec]
        drop_indices = self._normalize_drop(node_charges, drop)

        self._pairing = [list(row) for row in pairing]
        self._node_charges = node_charges
        self._drop_indices = drop_indices
        self._inv_rows = _node_basis_inverse(node_charges)

        if arrange:
            arranged = arrange_spec_stuff_first(
                self._pairing, node_charges, spec, drop_indices,
            )
            if arranged is None:
                raise ValueError(
                    "DirectionalSubquiverRG: no stuff-first arrangement "
                    "reachable by local moves within budget; supply a "
                    "stuff-first spec, raise the budget, or fall back to "
                    "an s_rg_oracle (version (c))."
                )
            spec = arranged
        self._spec = spec
        self._require_stuff_first = require_stuff_first
        if require_stuff_first:
            stuff, ir_spec = stuff_split(node_charges, spec, drop_indices)
        else:
            # Unguarded mode (version (c)): run the same ket-peel on the
            # spec as given, classifying entries by dropped-charge
            # content only.  The operator factorization S_UV = S_RG·S_IR
            # is *guaranteed* only for a genuinely (stuff)(ir)-ordered
            # spec; whether the order-by-order ket factorization always
            # yields a valid flow on interleaved specs is an OPEN
            # question (the machinery has so far been robust well beyond
            # physical expectation — user, 2026-06-10).  Certify the
            # result (`certify_directional_vs_bps`) before trusting it.
            stuff, ir_spec = classify_split(node_charges, spec, drop_indices)
        if not stuff:
            raise ValueError(
                "DirectionalSubquiverRG: no spec entry carries the dropped "
                "node(s) — the flow is empty (nothing is integrated out)."
            )
        if not ir_spec:
            raise ValueError(
                "DirectionalSubquiverRG: the IR spec is empty — dropping "
                f"{drop_indices} integrates out the whole spectrum."
            )
        self._stuff_spec = stuff
        self._ir_spec = ir_spec
        self._s_rg_oracle = s_rg_oracle
        self._f_oracle = f_oracle
        self._rg_window = rg_window

        from bps_kalgebra import BPSKAlgebra
        surviving = [
            g for j, g in enumerate(node_charges) if j not in drop_indices
        ]
        self._aux = BPSKAlgebra(
            pairing=pairing,
            node_charges=surviving,
            spec=ir_spec,
            verify=ir_verify,
        )

        # Closed-form S_RG product, assembled lazily by grading height.
        self._assembled: dict[Vec, "object"] = {}
        self._assembled_cap = -1

        # Grading data: dropped-multiplicity charges of the stuff entries
        # generate the Γ_RG positive cone.
        gens: list[Vec] = []
        for g in stuff:
            d = self._deg(g)
            if d not in gens:
                gens.append(d)
        self._cone_gens = tuple(gens)

    @classmethod
    def from_uv(
        cls,
        uv_bps,
        drop: Sequence,
        *,
        attach_f_oracle: bool = True,
        **kwargs,
    ) -> "DirectionalSubquiverRG":
        """Build the directional flow from an existing UV `BPSKAlgebra`
        (pairing / node charges / spec read off it).  The UV instance is
        retained as `starting_algebra()` — the composition endpoint
        `ComposedRG` validates by identity — and, by default, its
        F-decomposition is attached as the `f_oracle` (speed parity with
        the historical UV-wrapping extractors; still cross-checked by
        the ρ⁻¹ mirror).  KAlgebra ops remain the derived directional
        ones — the UV is a reference, never a delegate.

        `require_stuff_first` defaults to **False** here (the historical
        extractors accepted any spec ordering); pass it explicitly to
        re-enable the strict version-(a) guard."""
        kwargs.setdefault("require_stuff_first", False)
        inst = cls(
            [list(r) for r in uv_bps.lattice.pairing],
            [tuple(g) for g in uv_bps.node_charges],
            [tuple(g) for g in uv_bps.spec],
            drop,
            **kwargs,
        )
        inst._uv_ref = uv_bps
        if attach_f_oracle and inst._f_oracle is None:
            inst._f_oracle = uv_f_oracle(uv_bps, inst.auxiliary())
        return inst

    def starting_algebra(self):
        """The K-algebra this flow represents.  Default `self` (the
        directional object IS the presentation); when built `from_uv`,
        the UV `BPSKAlgebra` instance — certified the same algebra — so
        composition endpoints (`ComposedRG`: `first.auxiliary() is
        second.starting_algebra()`) keep validating by identity."""
        uv = getattr(self, "_uv_ref", None)
        return self if uv is None else uv

    # ----- drop normalisation ---------------------------------------------

    @staticmethod
    def _normalize_drop(
        node_charges: list[Vec], drop: Sequence,
    ) -> tuple[int, ...]:
        idxs: list[int] = []
        items = list(drop) if not isinstance(drop, (int,)) else [drop]
        for d in items:
            if isinstance(d, int):
                if not 0 <= d < len(node_charges):
                    raise ValueError(
                        f"drop index {d} out of range [0, {len(node_charges)})"
                    )
                idxs.append(d)
            else:
                dt = tuple(d)
                if dt not in node_charges:
                    raise ValueError(
                        f"drop charge {dt} not found in node_charges"
                    )
                idxs.append(node_charges.index(dt))
        if len(set(idxs)) != len(idxs):
            raise ValueError(f"drop entries must be distinct: {idxs}")
        if len(idxs) >= len(node_charges):
            raise ValueError("cannot drop every node")
        return tuple(sorted(idxs))

    # ----- grading ----------------------------------------------------------

    def _deg(self, label: Vec) -> Vec:
        try:
            dec = _decompose_in_node_basis(self._inv_rows, tuple(label))
            return tuple(dec[j] for j in self._drop_indices)
        except ValueError:
            # Γ ⊋ node-span (gauge theory: the Wilson/monopole are half-elements,
            # as for pure SU(2)'s standard (1,0),(-1,2)).  The RG degree only needs
            # the DROPPED-index coefficients; solve rationally and validate just
            # those (they must be integer — they ARE the Γ_RG grading; the
            # non-dropped gauge coefficients may legitimately be half-integral).
            dec = _rational_node_basis_decompose(self._node_charges, tuple(label))
            out: list[int] = []
            for j in self._drop_indices:
                x = dec[j]
                if x.denominator != 1:
                    raise ValueError(
                        f"_deg: dropped-index coefficient {x} (column {j}) for "
                        f"label {tuple(label)} is non-integer — RG degree "
                        f"ill-defined.")
                out.append(int(x))
            return tuple(out)

    def grading(self) -> Grading:
        """`Γ_RG = Z^{|drop|}`, `deg(label)` = the dropped-index
        coefficients of the label's **integer** node-basis decomposition
        (total — negative off-cone charges score correctly, which the
        ρ-mirror filtering needs), height `(1, …, 1)` = total dropped
        multiplicity, cone generated by the stuff entries' charges."""
        return Grading(
            rank=len(self._drop_indices),
            deg=self._deg,
            height=(1,) * len(self._drop_indices),
            cone_gens=self._cone_gens,
        )

    def _trace_is_grade_concentrated(self) -> bool:
        """`Γ_RG` here is **dropped gauge-node multiplicity** — a gauge
        grading: the dropped nodes are integrated out, so `Tr_aux` is
        supported on grade 0 and the `_pair_grade_blocks` prune of the trace
        pairing is exact (no flavour μ-refinement to lose — the #327
        gauge-vs-flavour distinction).  Cross-checked by
        `verify_inner_product_grade_pruned`."""
        return True

    def _section_split(self, label):
        """Generic split, guarded for **non-unimodular node lattices**:
        when `⟨γ_i⟩` has index > 1 in the charge lattice and `ker(B)` is
        not contained in it (e.g. the SU(2)+N_f=1 quiver — index 2,
        flavour `(0,0,1)` odd class), stripping the IR flavour section
        can move a node-integral label OFF the node lattice, where the
        directional grading/peel honestly refuses.  In that case the
        label stays its own section (zero shift): correct — the
        flavour-translate cache reuse of `RGKAlgebra.multiply` is a
        speed layer — just no sharing across that label's μ-orbit."""
        sec, flav = super()._section_split(label)
        if sec != tuple(label):
            try:
                _decompose_in_node_basis(self._inv_rows, sec)
            except ValueError:
                return tuple(label), tuple(0 for _ in label)
        return sec, flav

    # ----- S_RG: closed-form stuff product (version (a)) -------------------

    def _candidate_charges(self, cap: int) -> list[Vec]:
        """Non-negative integer combinations of the stuff entries with
        total dropped multiplicity ≤ `cap` (the support enumeration the
        validated extractor uses; finite by height-positivity)."""
        g = self.grading()
        zero = tuple([0] * len(self._node_charges[0]))
        seen: set[Vec] = {zero}
        frontier: list[tuple[Vec, int]] = [(zero, 0)]
        stuff_h = [
            (tuple(s), g.h(self._deg(s))) for s in self._stuff_spec
        ]
        while frontier:
            nxt: list[tuple[Vec, int]] = []
            for delta, h_used in frontier:
                for s, h_s in stuff_h:
                    new_h = h_used + h_s
                    if new_h > cap:
                        continue
                    new_delta = tuple(d + si for d, si in zip(delta, s))
                    if new_delta in seen:
                        continue
                    seen.add(new_delta)
                    nxt.append((new_delta, new_h))
            frontier = nxt
        return sorted(seen)

    def _assemble_closed_form(self, cap: int) -> dict[Vec, "object"]:
        """`S_RG = ∏_stuff E_𝖖(X_δ)` expressed over the IR auxiliary's
        canonical basis, every term of grading height ≤ `cap`, exact.

        Chart-free ket peel: the Nahm kets `[S_UV|0⟩]_δ` of the
        stuff-first spec are a closed form in the constructor data
        (`s_gamma_habiro(δ, spec, kmat)`); peeling them against the
        IR's F-elements (`state[η] −= s_δ · [F^IR_δ · S_IR|0⟩]_η`,
        dropped-charge-cone-minimal order) leaves exactly the `S_RG`
        coefficients on IR canonical labels.  Same peel as the
        validated `rg_flow.SubquiverRG.rg_generator`, with the UV
        chart's `_s_coefficient` replaced by the spec-level closed
        form — no UV algebra object exists here."""
        if cap <= self._assembled_cap:
            g = self.grading()
            return {
                l: c for l, c in self._assembled.items()
                if g.height_of(l) <= cap
            }
        from nahm_local import s_gamma_habiro
        from bps_kalgebra_internals import c_gamma_via_s
        from rg_flow import _cone_minimal

        spec_t = [tuple(x) for x in self._spec]
        n = len(self._pairing)
        kmat = [
            [
                sum(gi[a] * self._pairing[a][b] * gj[b]
                    for a in range(n) for b in range(n))
                for gj in spec_t
            ]
            for gi in spec_t
        ]
        state: dict[Vec, "object"] = {}
        for delta in self._candidate_charges(cap):
            h = s_gamma_habiro(tuple(delta), spec_t, kmat)
            if not h.is_zero():
                state[tuple(delta)] = h

        ir = self._aux
        ir_nodes = list(ir.node_charges)
        result: dict[Vec, "object"] = {}
        while state:
            delta = _cone_minimal(list(state.keys()), ir_nodes)
            leading = state.pop(delta)
            result[delta] = leading
            F_delta_ir = ir.F(delta)
            for eta in list(state.keys()):
                ir_state_eta = c_gamma_via_s(
                    eta, F_delta_ir, ir._s_coefficient, ir.lattice,
                )
                if ir_state_eta.is_zero():
                    continue
                state[eta] = state[eta] - leading * ir_state_eta
                if state[eta].is_zero():
                    del state[eta]
        result = {l: c for l, c in result.items() if not c.is_zero()}
        self._assembled = result
        self._assembled_cap = cap
        return dict(result)

    def _s_rg_component(self, p: Vec) -> dict:
        """Exact `[S_RG]_p` (RGKAlgebra oracle contract): closed-form
        stuff-product slice by default, or the injected `s_rg_oracle`."""
        p = tuple(int(x) for x in p)
        if self._s_rg_oracle is not None:
            return self._s_rg_oracle(p)
        if any(x < 0 for x in p):
            return {}
        g = self.grading()
        if p == g.zero_charge():
            from habiro import HabiroElement
            return {self._aux.identity(): HabiroElement.one()}
        cap = g.h(p)
        window = self._assemble_closed_form(cap)
        return {l: c for l, c in window.items() if g.charge(l) == p}

    def _cone_charges_to_height(self, cap: int) -> list[Vec]:
        """All Γ_RG cone charges of height ≤ `cap` (BFS over cone-gen
        sums; finite by height-positivity)."""
        g = self.grading()
        zero = g.zero_charge()
        seen = {zero}
        frontier = [zero]
        while frontier:
            nxt = []
            for p in frontier:
                for gen in self._cone_gens:
                    q = tuple(a + b for a, b in zip(p, gen))
                    if q in seen or g.h(q) > cap:
                        continue
                    seen.add(q)
                    nxt.append(q)
            frontier = nxt
        return sorted(seen)

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG` windowed by **grading height ≤ cutoff** (total dropped
        multiplicity — the same window the UV-wrapping `SubquiverRG`
        documents; the q-order of a term is bounded below by data the
        height controls in the tested regime).  Assembled as the union
        of exact graded components, so the closed-form and oracle paths
        share one implementation."""
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        if self._s_rg_oracle is None:
            return self._assemble_closed_form(cutoff)
        out: dict = {}
        for p in self._cone_charges_to_height(cutoff):
            for l, c in self._s_rg_component(p).items():
                out[l] = c
        return out

    # ----- RGKAlgebra contract ---------------------------------------------

    def auxiliary(self) -> KAlgebra:
        return self._aux

    def RG(self, a) -> Element:
        """Generic solve by default; `f_oracle` short-circuit when
        supplied (cached in the same 𝖖-number representation, so the
        mirror solve in `rho_inverse` can cross-check it later)."""
        if self._f_oracle is None:
            return super().RG(a)
        cache = self.__dict__.setdefault("_rg_cache", {})
        key = tuple(a)
        if key in cache:
            return self._rg_element_from_qn(cache[key])
        from q_number_poly import QNumberPoly
        img = self._f_oracle(key)
        if not isinstance(img, Element):
            img = Element(dict(img))
        cache[key] = {
            lbl: QNumberPoly.from_palindromic_laurent(c)
            for lbl, c in img.terms.items() if not c.is_zero()
        }
        return self._rg_element_from_qn(cache[key])

    def rho_inverse(self, a):
        """`ρ_UV⁻¹` from the exact mirror solve.  When `RG(a)` came from
        the F-oracle the mirror hasn't run yet, so run it here — and
        cross-check the oracle image against the solved one (a wrong
        oracle fails loudly rather than corrupting the presentation)."""
        key = tuple(a)
        rcache = self.__dict__.setdefault("_rho_inv_cache", {})
        if key not in rcache:
            from graded_rg_solver import solve_rg_exact
            from q_number_poly import QNumberPoly
            rg, rho_uv_inv = solve_rg_exact(
                self.auxiliary(), self.grading(),
                self._s_rg_component, self.apex(key))
            cached = self.__dict__.setdefault("_rg_cache", {}).get(key)
            if cached is not None:
                if self._rg_element_from_qn(cached) != rg:
                    raise ValueError(
                        f"DirectionalSubquiverRG.rho_inverse({key!r}): the "
                        f"F-oracle image disagrees with the generic exact "
                        f"solve — the oracle (or the flow data) is wrong."
                    )
            else:
                self.__dict__["_rg_cache"][key] = {
                    lbl: QNumberPoly.from_palindromic_laurent(c)
                    for lbl, c in rg.terms.items() if not c.is_zero()
                }
            rcache[key] = rho_uv_inv
        rui = rcache[key]
        if rui is None:
            raise ValueError(
                f"rho_inverse({a!r}) could not be derived (mirror did not "
                f"close)."
            )
        return self._kernel_corrected_rho_inverse(key, self._apex_inverse(rui))

    def _kernel_corrected_rho_inverse(self, a: Vec, b: Vec) -> Vec:
        """Tripwire pinning the mirror's `ρ_UV⁻¹(a)` candidate `b` to the
        right flavour coset.  The root cause — the chart-level σ applied
        to full labels without the section rectification, which let the
        mirror upper land a `ker B` shift off — was fixed at the source
        by PR #415's A8 flavour seam (`BPSKAlgebra._sec_rectified_map`:
        ρ acts on the *multiplicative* pair `(section canonical,
        R element)`, with ⋆ on the R part); after it, the flavoured
        n_003 dictionary certifies 33/33 in pure-derived mode and this
        correction never fires.  Kept as defence-in-depth: ρ negates
        central charges (`ρ(b + w) = ρ(b) − w` for `w ∈ ker B`), so a
        residual coset error is corrected in one linear step

            v = a − ρ_UV(b);   v ∈ ker B  ⇒  ρ_UV(b − v) = a,

        and any NON-kernel discrepancy — or a closure failure after the
        correction — raises loudly (a genuine axiom violation, never
        silently absorbed)."""
        rho_b = tuple(self.rho(b))
        v = tuple(x - y for x, y in zip(a, rho_b))
        if not any(v):
            return tuple(b)
        n = len(self._pairing)
        Bv = [sum(self._pairing[i][j] * v[j] for j in range(n))
              for i in range(n)]
        if any(Bv):
            raise ValueError(
                f"rho_inverse({a!r}): mirror candidate {b} fails ρ∘ρ⁻¹ by "
                f"a NON-kernel vector {v} — genuine inconsistency (not the "
                f"flavour-coset ambiguity)."
            )
        b2 = tuple(x - y for x, y in zip(b, v))
        if tuple(self.rho(b2)) != tuple(a):
            raise ValueError(
                f"rho_inverse({a!r}): kernel correction {b} → {b2} did not "
                f"close (ρ(b₂) = {self.rho(b2)})."
            )
        self.__dict__["_rho_inv_cache"][tuple(a)] = b2
        return b2

    def _rg_cutoff(self) -> int:
        """The `S_RG` window for the generic trace / windowed-solve paths
        (a heuristic, per the parent's caveat).  Tunable per instance via
        the `rg_window` constructor kwarg — higher-rank flows want a
        smaller window than the parent default (the window enters the
        transported trace product quadratically)."""
        if self._rg_window is not None:
            return self._rg_window
        return super()._rg_cutoff()

    def _trace_uncached(self, a, K: int = 20, **kwargs):
        """Parent RG-transport trace + a **window prefilter**: a product
        term whose coefficient already starts beyond `q^K` cannot reach
        the `q^K` window (per-label traces start at `q^{≥0}` in the
        repo's Schur normalization), so it is dropped *before* its IR
        trace is computed.  On rank-4 node drops this removes the large
        deep-dressing tail of the transported product — same output,
        far fewer IR Schur evaluations.

        The overridable hook behind `RGKAlgebra.trace` (memoised); the IR
        traces it fans out to (`aux.trace_element → aux.trace`) hit the IR
        auxiliary's own (BPS) trace cache."""
        aux = self.auxiliary()
        Ke = K + self._rg_cutoff()
        a_S = self.rg_times_s_rg(a, K)
        rho_S = self._rho_s_rg_element_cached(Ke)
        prod = aux.multiply_elements(rho_S, a_S)
        kept = Element({
            l: c for l, c in prod.terms.items()
            if not c.is_zero() and min(c._coeffs) <= K
        })
        return aux.trace_element(kept, K)

    # ----- introspection -----------------------------------------------------

    @property
    def node_charges(self) -> list[Vec]:
        return list(self._node_charges)

    @property
    def spec(self) -> list[Vec]:
        return list(self._spec)

    @property
    def stuff_spec(self) -> list[Vec]:
        return list(self._stuff_spec)

    @property
    def ir_spec(self) -> list[Vec]:
        return list(self._ir_spec)

    @property
    def drop_indices(self) -> tuple[int, ...]:
        return self._drop_indices


class DirectionalSingleNodeRG(DirectionalSubquiverRG):
    """`|drop| = 1` convenience: same signature shape as the historical
    `SingleNodeRGKAlgebra` (`gamma_drop` = the dropped node's charge or
    index)."""

    def __init__(
        self,
        pairing: Sequence[Sequence[int]],
        node_charges: Sequence[Vec],
        spec: Sequence[Vec],
        gamma_drop,
        **kwargs,
    ) -> None:
        super().__init__(pairing, node_charges, spec, [gamma_drop], **kwargs)

    @property
    def gamma_drop(self) -> Vec:
        return tuple(self._node_charges[self._drop_indices[0]])


# ---------------------------------------------------------------------------
# F-oracle from a UV BPSKAlgebra (the speed route, decisions A8 + user 2026-06-10)
# ---------------------------------------------------------------------------


def uv_f_oracle(uv_bps, ir: KAlgebra) -> Callable[[Vec], Element]:
    """RG-image oracle backed by a UV `BPSKAlgebra`'s F-solver: `F(a)`
    peeled into `ir`'s canonical basis (the historical extraction,
    inlined — `rg_flow._decompose_qt_in_ir_basis`).  `ir` is the
    directional flow's `auxiliary()` (any instance with the same
    lattice / surviving-node data works — labels are shared tuples).
    The directional class cross-checks oracle images against the
    generic exact solve whenever its mirror runs, so the oracle is an
    accelerator, not an authority."""
    def oracle(label: Vec) -> Element:
        from rg_flow import _decompose_qt_in_ir_basis
        F_uv = uv_bps.F(tuple(label))
        decomposed = _decompose_qt_in_ir_basis(dict(F_uv), ir)
        return Element({
            l: c for l, c in decomposed.items() if not c.is_zero()
        })
    return oracle


# ---------------------------------------------------------------------------
# Axiom battery (KAlgebra axioms + RG axioms on the directional flow)
# ---------------------------------------------------------------------------


def verify_axioms(
    direct: "DirectionalSubquiverRG",
    labels: Sequence[Vec],
    pairs: Sequence[tuple[Vec, Vec]],
    *,
    trace_K: Optional[int] = None,
    deep_rho: bool = True,
) -> dict:
    """The axiom battery on a directional flow — the executable form of
    the RG axiomatics, run on a *derived* presentation (nothing
    delegated, so every check is information).

    Always: bar involution (`C^c_{ab}(q⁻¹) = C^c_{ba}(q)`),
    ρ-automorphism, ρ∘ρ⁻¹, ρ(1)=1, RG-unital, RG-multiplicative,
    RG-bar-invariance, and the tRG intertwining
    `RG∘ρ_UV = ρ_IR∘tRG`.  With `trace_K`: ρ²-twisted trace cyclicity
    and canonical-basis orthonormality (`I_{a,a}[q⁰] = 1`,
    `I_{a,b}[q⁰] = 0`) — the expensive, Schur-side axioms.

    `deep_rho=False` skips the ρ∘ρ⁻¹ round-trip: its second leg queries
    ρ/ρ⁻¹ at the *mirror* (negative) apexes, where the oracle-less exact
    solve must assemble deep `S_RG` caps — unbounded for a mass screen
    (the bulk dictionary harness pins ρ against the UV through the iso
    battery instead, at positive labels).

    Returns `{check: bool}`; any False is a counterexample candidate.
    """
    labels = [tuple(l) for l in labels]
    pairs = [(tuple(a), tuple(b)) for a, b in pairs]
    out: dict = {}
    out["rho_fixes_identity"] = direct.verify_rho_fixes_identity()
    if deep_rho:
        out["rho_inverse"] = all(
            direct.verify_rho_inverse(a) for a in labels)
    out["bar_involution"] = all(
        direct.verify_bar_involution(a, b) for a, b in pairs)
    out["rho_automorphism"] = all(
        direct.verify_rho_is_automorphism(a, b) for a, b in pairs)
    out["rg_unital"] = direct.verify_rg_unital()
    out["rg_multiplicative"] = all(
        direct.verify_rg_multiplicative(a, b) for a, b in pairs)
    out["rg_bar_invariant"] = all(
        direct.verify_rg_bar_invariant(a) for a in labels)
    out["rg_trg_intertwine"] = all(
        direct.verify_rg_trg_intertwine(a) for a in labels)
    if trace_K is not None:
        out["rho_twisted_trace"] = all(
            direct.verify_rho_twisted_trace(a, b, K=trace_K)
            for a, b in pairs[:2])
        diag = all(
            direct.verify_orthonormality(a, a, K=trace_K) for a in labels[:2])
        off = True
        distinct = [(a, b) for a, b in pairs if a != b][:1]
        for a, b in distinct:
            off = off and direct.verify_orthonormality(a, b, K=trace_K)
        out["orthonormality"] = diag and off
    return out


# ---------------------------------------------------------------------------
# Certification harness: directional RG presentation vs UV BPSKAlgebra
# ---------------------------------------------------------------------------


def default_certification_labels(uv_bps) -> list[Vec]:
    """Identity + node charges + their ρ-images + pairwise node sums —
    generators, σ-twists, and one composite layer (mirrors
    `bpskalgebra_kalgebra_iso.default_iso_samples`, label-level)."""
    labels: list[Vec] = [tuple(uv_bps.identity())]
    for g in uv_bps.node_charges:
        gt = tuple(g)
        if gt not in labels:
            labels.append(gt)
    for g in list(labels):
        if g == tuple(uv_bps.identity()):
            continue
        rg = tuple(uv_bps.rho(g))
        if rg not in labels:
            labels.append(rg)
    nodes = [tuple(g) for g in uv_bps.node_charges]
    for i, gi in enumerate(nodes):
        for j, gj in enumerate(nodes):
            if i == j:
                continue
            s = tuple(a + b for a, b in zip(gi, gj))
            if s not in labels:
                labels.append(s)
    return labels


def certify_directional_vs_bps(
    direct: DirectionalSubquiverRG,
    uv_bps,
    *,
    labels: Optional[Sequence[Vec]] = None,
    pairs: Optional[Sequence[tuple[Vec, Vec]]] = None,
    check_rho: bool = True,
    trace_K: Optional[int] = None,
    trace_labels: Optional[Sequence[Vec]] = None,
    name: str | None = None,
) -> dict:
    """Staged certificate that the directional presentation and an
    independently built UV `BPSKAlgebra` present the same K-algebra.

    Stage 0 — *flow coherence* (on `direct` alone, no comparison):
    `rg_unital`, `rg_multiplicative` on the pairs, `rg_bar_invariant`
    on the labels.  A mis-specified flow (e.g. a tropical relabel
    masquerading as RG — the recorded `a1a2k_rgkalg` wrong turn) fails
    here, before any iso interpretation is on the table.

    Stage 1 — *iso battery* on `KAlgebraIso.identity_on_labels(direct,
    uv_bps)`: unit, round-trip, multiplicativity (the core
    cross-presentation check), ρ-equivariance (`check_rho` — genuine
    here: `direct.rho` is tRG-derived, not delegated), and
    trace-equivariance at `q^trace_K` when `trace_K` is given.  Traces
    are the expensive stage; they run on `trace_labels` (default: the
    identity, the first generator, and its ρ-image) rather than the
    full label set.

    Returns `{check: bool}`; all-True is the certificate (on the given
    samples).  The iso object itself is under key `"iso"`.
    """
    if labels is None:
        labels = default_certification_labels(uv_bps)
    labels = [tuple(l) for l in labels]
    if pairs is None:
        gens = [l for l in labels if l != tuple(uv_bps.identity())]
        pairs = [(a, b) for a in gens for b in gens]
    one = LaurentPoly.one()

    out: dict = {}
    out["rg_unital"] = direct.verify_rg_unital()
    out["rg_bar_invariant"] = all(
        direct.verify_rg_bar_invariant(l) for l in labels
    )
    out["rg_multiplicative"] = all(
        direct.verify_rg_multiplicative(a, b) for a, b in pairs
    )

    iso = KAlgebraIso.identity_on_labels(direct, uv_bps, name=name)
    out["iso"] = iso
    samples = [Element({l: one}) for l in labels]
    tgt_samples = [iso.map(e) for e in samples]
    elem_pairs = [
        (Element({a: one}), Element({b: one})) for a, b in pairs
    ]
    tgt_pairs = [(iso.map(a), iso.map(b)) for a, b in elem_pairs]

    out["unit"] = iso.verify_unit()
    out["round_trip"] = iso.verify_round_trip(samples, tgt_samples)
    out["multiplicative"] = iso.verify_multiplicative(elem_pairs, tgt_pairs)
    if check_rho:
        out["rho_equivariant"] = iso.verify_rho_equivariant(
            samples, tgt_samples)
    if trace_K is not None:
        if trace_labels is None:
            ident = tuple(uv_bps.identity())
            gens = [l for l in labels if l != ident]
            trace_labels = [ident]
            if gens:
                trace_labels.append(gens[0])
                trace_labels.append(tuple(uv_bps.rho(gens[0])))
        t_samples = [Element({tuple(l): one}) for l in trace_labels]
        out["trace_equivariant"] = iso.verify_trace_equivariant(
            t_samples, [iso.map(e) for e in t_samples], K=trace_K)
    return out
