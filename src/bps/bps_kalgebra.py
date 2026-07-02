"""`BPSKAlgebra` — a `KAlgebra` presented via a chosen RG flow to a quantum torus.

The defining structure is:

    "A KAlgebra equipped with a choice of RG flow to a quantum torus,
     used to label elements and compute structure constants and
     Schur indices."

A `BPSKAlgebra` is determined by a lattice Γ with antisymmetric integer
pairing `B` (possibly degenerate), a positive cone Γ_+ ⊂ Γ, and a way
to compute `[S|0⟩]_γ` for any γ. The latter can come from either:

    (a) an ordered finite spectrum-generator factorization
        `S = E_q(X_{γ_1}) · ⋯ · E_q(X_{γ_N})` — pass `spec=...` or
        `negating_sequence=...`. `[S|0⟩]_γ` is computed by Nahm-sum
        expansion of the finite product.

    (b) a user-supplied recipe `s_coefficient(γ) -> HabiroElement` —
        pass `s_coefficient=...`. Used for theories with no finite
        `E_q`-factorization but a closed form on the lattice (e.g.,
        SU(2) N=2*).

In both cases the canonical primitive is `_s_coefficient(γ)`, and
`F`-solving / Schur-index computation are parameterized by it.

The RG flow's σ map (= label-level ρ) is derived from spec in case (a);
in case (b) the user must supply `sigma=...` and `sigma_inverse=...`.

Charges vs labels (three roles of an integer tuple)
---------------------------------------------------

`Vec = tuple[int, ...]` is overloaded across three roles, and conflating
them is the most common bug here.  Convention: **`γ` is always a charge /
QT-index; `a, b, c` are always canonical labels.**

* **charge / QT-index `γ ∈ Γ`** — a genuine *label of the quantum torus*
  `QuantumTorusKAlg(Γ, B)` (the canonical-basis label of the generator
  `X_γ`), not merely an abstract charge.  `X_γ X_{γ'} = q^{<γ,γ'>}
  X_{γ+γ'}`, `ρ_Q(γ) = −γ`.  Lattice `+`, `−`, `<·,·>` live here (and on
  `node_charges`, `spec`, `cone_gens`).
* **canonical label `a`** — indexes `L_a` in *this* `BPSKAlgebra` (its RG
  image is `F(L_a)`).  `ρ(a) = σ(a)` is **piecewise-linear** (NOT `−a`);
  `multiply(a, b)` is the only legal binary op (NOT `a + b`, and
  `multiply(a, b) ≠ L_{a+b}`).

They are bridged by `F = RG`, the **RG-flow algebra homomorphism**
`F : A_𝖖[T] → QuantumTorusKAlg(Γ, B)` (`F(L_a · L_b) = F(L_a)·F(L_b)`,
the `verify_rg_multiplicative` axiom — this is how structure constants
are computed).  `F(L_a) -> dict{γ: c_γ(q)}`: the *argument* is a label
`a`, the *keys* are charges `γ`.  Intuitively the algebra *discovers* a
canonical element `F_γ` from a charge `γ` via `F_γ · S = X_γ + O(q)`
(`S` = spectrum generator), then treats it as the RG image
`F_{γ(a)} = F(L_a)` of the abstract element `L_a`.  So write `F_γ`
(charge subscript), `F(a)`, or `F(L_a)` — never `F_a`.  A label `a` is
its own **lower tropical charge** `γ₋(a) = a`; the **upper tropical
charge** is `γ⁺(a) = ν_S(γ₋(a)) = −σ⁻¹(γ₋(a))`, and `F(L_a)` is supported
on the doubly-tropical interval `[γ₋(a), γ⁺(a)]`.

Flavour
-------

When `B` is degenerate, `Γ_f := ker(B)` is the abelian flavour
sublattice. The constructor extracts `Γ_f` via SNF, picks a Z-linear
section `Γ_g → Γ`, and the coefficient ring becomes
`AbelianZPlusRing(rank=f)` with `f = rk(Γ_f)` (else `TrivialZPlusRing()`
when `f = 0`). Canonical-basis labels are section reps in Γ_g, with
flavour shifts absorbed into μ-monomial coefficients; `multiply` and
`rho_element` both honour this convention. Lattice rank can be smaller
than the number of BPS-quiver nodes (composite charges in the spec).

Storage
-------

* `F[γ]`           — `dict[Vec, LaurentPoly]`. F-cache is exact in
  `Z[q, q⁻¹]`; the coefficients are conjecturally non-negative
  `[n]_q`-positive (the no-exotics positivity conjecture — not a
  fact checked by the code).
* `F·S[γ, η]`      — `HabiroElement` (exact in the localized ring
  `R = Z[q^±][1/(1−q^{2k})]`). Cached by `(γ, η)`.

Layout
------

The class is organized into four sections so a future reader can use
the type system to tell which operations are intrinsic to `KAlgebra`
versus chart-specific.

    1. Defining data (constructor)
    2. KAlgebra contract (intrinsic ops, computed via the chart)
    3. The RG flow itself (`F`, `S`, `σ`)
    4. Chart utilities and accessors
"""

from __future__ import annotations

from typing import Callable, Sequence
import sys, os

# Make this directory and its parent importable by bare module name.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json as _json
import hashlib as _hashlib

from kalgebra import KAlgebra, Element
from rgkalgebra import RGKAlgebra
from quantum_torus_kalgebra import QuantumTorusKAlg
from zplus_ring import (
    ZPlusRing, RElement, RLaurent, RPowerSeries,
    TrivialZPlusRing, AbelianZPlusRing,
)
from snf_kernel import integer_kernel_and_section, decompose_in_basis
from laurent_poly import LaurentPoly as QTLaurentPoly
from lattice import Lattice
from habiro import HabiroElement
from q_number_poly import QNumberPoly
from qpoch import PowerSeries
from nahm_local import s_gamma_habiro
from spec_sigma import sigma_forward, sigma_inverse as _spec_sigma_inverse

import bps_quiver_tools as bps

from bps_kalgebra_internals import (
    solve_F_via_s_coefficient,
    solve_F_with_fs_table,
    solve_F_with_initial_guess,
    c_gamma_via_s,
    _enumerate_output_charges,
    _habiro_to_ps,
    qt_multiply,
    compute_strict_cone_witness,
    find_lowest,
)
from nahm_local import fs_dict_for_eta_set
from qpoch import qpoch_infty
from spec_shortening import shorten_spec as _shorten_spec_via_local_moves
from chart_graph import ChartGraph


Vec = tuple[int, ...]


def _bps_mat_inverse(M):
    """Exact Fraction inverse of a square integer matrix (Gauss-Jordan).

    Used in spec-free mode to turn node charges (a basis of Γ) into the cone
    coordinate map for the F-solver's degree cap.
    """
    from fractions import Fraction
    n = len(M)
    A = [[Fraction(M[i][j]) for j in range(n)] + [Fraction(i == j) for j in range(n)]
         for i in range(n)]
    for col in range(n):
        piv = next(r for r in range(col, n) if A[r][col] != 0)
        A[col], A[piv] = A[piv], A[col]
        d = A[col][col]
        A[col] = [x / d for x in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [a - f * b for a, b in zip(A[r], A[col])]
    return [[A[i][n + j] for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------------------
# BPSKAlgebra
# ---------------------------------------------------------------------------


class BPSKAlgebra(RGKAlgebra):
    """A `KAlgebra` realized via a BPS-quiver RG flow to a quantum torus.

    Inherits from `RGKAlgebra` (and thus from `KAlgebra`): the BPS
    realisation *is* an RG flow, with `auxiliary()` the
    `QuantumTorusKAlg(pairing)` and `RG(L_a) = F(L_a)` lifted into the
    auxiliary's `Element` type. The four `verify_rg_*` methods on
    `RGKAlgebra` then run on every `BPSKAlgebra` instance directly.

    See module docstring for the conceptual picture.
    """

    # ====================================================================
    # 1. Defining data: the RG flow
    # ====================================================================

    def __init__(
        self,
        pairing: Sequence[Sequence[int]],
        node_charges: Sequence[Sequence[int]] | None = None,
        *,
        spec: Sequence[Sequence[int]] | None = None,
        negating_sequence: Sequence[int] | None = None,
        s_coefficient: Callable[[Vec], HabiroElement] | None = None,
        sigma: Callable[[Vec], Vec] | None = None,
        sigma_inverse: Callable[[Vec], Vec] | None = None,
        cone_gens: Sequence[Sequence[int]] | None = None,
        cone_witness: Sequence[int] | None = None,
        shorten_spec: bool = False,
        verify: str = "lazy",
        k_joint_prune: bool = False,
        build_S: bool = False,
        build_S_cutoff: int | None = None,
        build_S_order: Sequence[Sequence[int]] | None = None,
        spec_free_sigma: str = "trg",
        extract_spec: bool = True,
    ):
        """Construct from RG-flow data.

        Two modes are supported:

        **Spec mode**: pass `spec` (or `negating_sequence`, or omit both
        to auto-find one via `BPSQuiver.find_negating_sequence` -- a
        bidirectional BFS over the mutation graph of `node_charges`,
        with an edge-multiplicity heuristic.  Auto-find is the default
        fallback but is a desperate move for large quivers (n >= 8 or
        dense pairings can take minutes or fail to terminate); supply
        `spec` or `negating_sequence` directly when you know one.
        σ is derived automatically.

        **Recipe mode**: pass `s_coefficient` (a callable
        `γ → HabiroElement` returning `[S|0⟩]_γ`), plus `sigma` and
        `sigma_inverse` callables.  Used for theories without a finite
        `E_q`-factorization (e.g., SU(2) N=2*).

        **Spec-free mode** (`build_S=True`): build the spectrum generator
        `S` by the recursion (`recursive_spectrum`) — no spec, no
        green-sequence BFS.  Precedence: a finite spec is
        the *ideal* outcome, so unless `extract_spec=False` the built `S` is
        run through the extractor first and a recovered finite chamber drops
        into fast spec mode.  Only when no finite spec is found does the
        spec-free fallback engage, with `spec_free_sigma` choosing its σ
        source: `"trg"` (default) keeps the tRG-derived ρ and no exact σ;
        `"principled"` installs the axiom-derived σ⁻¹(a)=−upper(F_a),
        σ(a)=−upper(F̃_a) (read off the built `S`) with the fast
        section-rectified ρ — the path for no-finite-spec theories;
        `"auto"` is an alias for `"trg"`.  `extract_spec=False` forces the
        spec-free path.  **`build_S_cutoff` defaults to `None` = auto-stabilize**:
        the cone cutoff is grown until the node canonicals are cone-interior
        (σ-stable), mirroring the adaptive Schur shell — so spec-free mode needs
        no user cutoff (pass an int to pin one).  σ/ρ are exact; `multiply` is
        cone-truncated to the *built* degree (`from_ir_image` drops out-of-cone
        terms — graceful, never a hang), so pin a larger cutoff for deeper
        products.

        Parameters
        ----------
        pairing
            Antisymmetric integer matrix (rank n × n).  May be degenerate;
            `Γ_f := ker(B)` is the abelian-flavour sublattice and is
            extracted automatically via SNF.  `coefficient_ring()`
            becomes `AbelianZPlusRing(rank=f)` when `f = rk(Γ_f) > 0`,
            else `TrivialZPlusRing()`.
        node_charges
            BPS-quiver nodes (each a vector in Γ).  Required in spec
            mode -- both for the auto-find fallback and for the
            positive cone.  Optional in recipe mode.
        spec, negating_sequence
            See "Spec mode" above.  When neither is given, the
            constructor invokes `BPSQuiver.find_negating_sequence`
            (bidirectional BFS) to discover one.
        s_coefficient, sigma, sigma_inverse
            See "Recipe mode" above.
        cone_gens
            **Positive-cone** generators `Γ₊` (the global pointedness
            cone) — *not* the monomial-cone filtration of `ConeData`
            (the per-chart PBW blocks); see "Two senses of cone" in
            `cone_data.py`.  Default in spec mode: the node charges.
            Required in recipe mode (no nodes to fall back on).
        cone_witness
            Optional integer dual vector strictly positive on the cone
            (chart-internal, used to disambiguate cone-minima during
            F-decomposition).
        shorten_spec
            If True (and a `spec` is supplied or auto-found), greedily
            replace sub-ranges with shorter equivalent middles via
            local-moves search (`spec_shortening.shorten_spec`).
            Default False -- opt-in.  Or use the post-construction
            method `A.shorten_spec()` to get a new instance with a
            shorter spec.
        verify
            Controls the *expensive* checks: spec verification,
            pointed-cone search.  The cheap **contract checks** --
            non-degeneracy of the pairing and pointed-cone witness
            verification (when `cone_witness` is supplied) -- always
            run regardless of `verify` mode.
            * "lazy" (default): expensive checks deferred to first use.
            * "eager": run them at init.
            * "off": skip the expensive checks (the user takes
              responsibility for spec validity).
        """
        # ----- pairing + flavour kernel decomposition -----
        # Allow degenerate pairings: ker_Z(B) = Γ_f gives the abelian
        # flavour symmetry.  An SNF-derived Z-linear section
        # Γ_g → Γ is picked once and stored privately; it's used at the
        # multiply / trace boundary to translate Γ-elements into orbit
        # labels with μ-monomial R-coefficients.  Linearity makes the
        # section antipodally symmetric (= ρ-equivariant), so ρ on
        # orbit labels is a clean permutation with no μ-shift.
        self.lattice = Lattice(pairing)
        self._ker_basis, self._sec_basis = integer_kernel_and_section(
            [list(row) for row in pairing]
        )
        self._flavour_rank = len(self._ker_basis)
        self._gauge_rank = len(self._sec_basis)
        # `k_joint_prune` (accepted for backward compatibility — some
        # callers, e.g. `pure_ade_lattice`, opted in to a former prune) is
        # SUPERSEDED: the Schur/trace support is bounded by the adaptive
        # two-cutoff-stability shell (`_schur_index_stable`), which is sound
        # in every frame AND keeps the e8-scale memory win that the opt-in
        # prune was reaching for — so the unsound linear K_joint prune is
        # retired and this flag is a documented no-op.
        self._k_joint_prune = bool(k_joint_prune)
        # Coefficient ring of the flavoured-KAlgebra contract: R(U(1)^f) =
        # AbelianZPlusRing(rank=f) where f = rk(Γ_f).  When f = 0, this
        # is TrivialZPlusRing (R = Z, the unflavoured case).
        self._R: ZPlusRing = (
            TrivialZPlusRing() if self._flavour_rank == 0
            else AbelianZPlusRing(rank=self._flavour_rank)
        )

        # ----- auxiliary quantum-torus K-algebra (the RG-flow target) -----
        # Cached per construction: `auxiliary()` must return the same object every
        # call so `then(...)` composition's Python-`is` endpoint check
        # works.  The QT and self share `R` by construction (both derive
        # from the same SNF on `pairing`); we sanity-check that here.
        self._qt_aux = QuantumTorusKAlg(pairing)
        assert self._qt_aux.coefficient_ring() == self._R, (
            "BPSKAlgebra and QuantumTorusKAlg disagree on coefficient_ring; "
            "indicates an SNF / ZPlusRing inconsistency."
        )

        # ----- mode dispatch -----
        recipe_mode = s_coefficient is not None
        self._spec_free = bool(build_S)
        if spec_free_sigma not in ("auto", "principled", "trg"):
            raise ValueError(
                "spec_free_sigma must be 'trg' (spec-free σ via tRG; the "
                "default fallback), 'principled' (spec-free σ = −upper(F), the "
                "principled relation), or 'auto' (alias for 'trg').")
        # Priority: a provided spec wins; failing that, the
        # S-finder's *ideal* outcome is to recover a finite spec (extraction A,
        # run iff `extract_spec`); only when no spec is found does the spec-free
        # fallback engage, and `spec_free_sigma` chooses it — 'trg' (C) or the
        # principled −upper(F) σ (B).  `extract_spec=False` forces the spec-free
        # path (for benchmarking/validating B/C even where a spec exists).
        self._spec_free_sigma = "trg" if spec_free_sigma == "auto" else spec_free_sigma
        # Set when the spec-free σ is supplied by the principled −upper(F)
        # relation (B): then ρ uses the fast section-rectified map, not tRG.
        self._principled_sigma = False
        # Spec-free F-solver cone-degree cap (only set in spec-free mode): bounds
        # the recipe F-solver to the degree-≤cutoff cone simplex instead of the
        # generous box — the principled, flavour-safe truncation (matter dressing
        # costs cone-degree).  None in spec/recipe mode (those keep their boxes).
        self._sf_max_degree: int | None = None
        self._sf_degree_fn = None
        spec_mode_inputs = sum(x is not None for x in [spec, negating_sequence])
        if recipe_mode and spec_mode_inputs > 0:
            raise ValueError(
                "Pass either spec/negating_sequence (spec mode) OR "
                "s_coefficient (recipe mode), not both."
            )
        if recipe_mode and (sigma is None or sigma_inverse is None):
            raise ValueError(
                "Recipe mode (s_coefficient supplied) requires both "
                "sigma=... and sigma_inverse=... callables."
            )
        if self._spec_free and (recipe_mode or spec_mode_inputs > 0):
            raise ValueError(
                "build_S (spec-free mode) is exclusive with spec / "
                "negating_sequence / s_coefficient."
            )

        # ----- node charges -----
        if node_charges is None:
            if not recipe_mode:
                raise ValueError("node_charges required in spec mode")
            self.node_charges: list[Vec] = []
        else:
            self.node_charges = [tuple(self.lattice.check(g)) for g in node_charges]
        # The algebra-level flavour structure is `Γ_f = ker(B)`, derived
        # above from the pairing — *not* from any per-node "frozen" flag.
        # The bundle stack's `bps_quiver_tools.CoulombAlgebra` exposes a
        # per-node `frozen` flag for chart-mutation-policy reasons (BFS
        # over the mutation graph skips frozen nodes), but that's a
        # private chart concern, not part of the KAlgebra contract.

        # ----- spec-free: build S, then recover a finite spec if one exists ---
        # `build_S=True` builds the spectrum generator by the recursion (no
        # green-sequence BFS).  Preferred path (finding a spec is the ideal
        # outcome of the S-finder):
        # extract a finite-chamber spec from the built S and run the *fast spec
        # mode* (combinatorial σ).  This runs whenever `extract_spec` (default),
        # regardless of `spec_free_sigma`.  Only if no finite spec exists (e.g.
        # N=2*/Markov), or extraction is explicitly disabled, does the spec-free
        # fallback below engage (B or C per `spec_free_sigma`).
        # Resolve the spec-free cone cutoff.  Default (`build_S_cutoff is None`):
        # auto-stabilize — grow the cutoff until the node canonicals are
        # cone-interior (σ-stable), mirroring the adaptive Schur shell
        # (`_schur_index_stable`), so `build_S=True` needs NO user cutoff and the
        # cutoff stops being a knob later work must reason about.  An explicit
        # int is honoured as a fixed override.  The auto-built `S` is cached and
        # reused below (no rebuild).  σ/ρ are stabilised by this; `multiply` is
        # exact only to the built degree (`from_ir_image` truncates beyond it —
        # graceful, never a hang), so pass an explicit larger cutoff for deeper
        # products.
        self._auto_S = None
        if self._spec_free and build_S_cutoff is None:
            from recursive_spectrum import build_spectrum_generator_auto
            self._auto_S, build_S_cutoff = build_spectrum_generator_auto(
                [list(r) for r in pairing],
                [tuple(g) for g in self.node_charges],
                order=build_S_order,
            )

        if self._spec_free and extract_spec:
            from recursive_spectrum import extract_spec_from_quiver
            _ex = extract_spec_from_quiver(
                [list(r) for r in pairing],
                [tuple(g) for g in self.node_charges],
                cutoff=build_S_cutoff, order=build_S_order,
            )
            if _ex is not None:
                spec = [tuple(g) for g in _ex]   # → fast spec mode below
                self._spec_free = False

        # ----- spec / recipe + σ + cone -----
        self._chart = None  # populated in spec mode
        if recipe_mode:
            self._s_coefficient_fn = s_coefficient
            self._sigma_fn = sigma
            self._sigma_inverse_fn = sigma_inverse
            self.spec: list[Vec] = []  # not applicable
            if cone_gens is None:
                raise ValueError("Recipe mode requires explicit cone_gens.")
            self.cone_gens = [tuple(self.lattice.check(g)) for g in cone_gens]
        elif self._spec_free:
            # Spec-FREE mode: build the spectrum generator S by the recursive
            # engine (recursive_spectrum.build_spectrum_generator) — no spec, no
            # green-sequence BFS.  `_s_coefficient(γ) = [S]_γ`; everything else
            # (multiply, trace, ρ via tRG) derives from it as in recipe mode.
            # σ is NOT supplied: ρ/ρ⁻¹ route through the RGKAlgebra parent's
            # tRG-derived map (see rho/rho_inverse), and the F-solver uses a
            # generous cone-covering window in place of the exact tropical σ.
            from recursive_spectrum import build_spectrum_generator
            if node_charges is None:
                raise ValueError("build_S (spec-free) requires node_charges.")
            H0 = HabiroElement.zero()
            if self._auto_S is not None:        # reuse the auto-stabilized build
                S_built = self._auto_S
            else:
                S_built = build_spectrum_generator(
                    [list(r) for r in pairing],
                    [tuple(g) for g in self.node_charges],
                    build_S_cutoff, order=build_S_order,
                )
            self._built_S = {tuple(g): c for g, c in S_built.items()}
            self._s_coefficient_fn = lambda g, _H0=H0: self._built_S.get(
                tuple(g), _H0)
            n = len(pairing)
            csum = tuple(sum(nc[i] for nc in self.node_charges) for i in range(n))
            W = max(1, int(build_S_cutoff))
            # The F-solver (solve_F_via_s_coefficient) consumes `_sigma_inverse_fn`
            # as the support-window *upper bound* `[γ, −sinv(γ)]`, so it must be a
            # GENEROUS cone-covering bound — NOT the exact σ⁻¹ (which is
            # self-referential: it solves F to bound F).  Both spec-free σ
            # variants keep this generous window for the F-solver; ρ uses a
            # separate exact map when principled.
            self._sigma_fn = None
            self._sigma_inverse_fn = (
                lambda g, _c=csum, _W=W: tuple(-(g[i] + _W * _c[i])
                                               for i in range(len(g))))
            if self._spec_free_sigma == "principled":
                # Path B — the principled spec-free σ, derived from the
                # axioms.  σ⁻¹(a)=−upper(F_a),
                # σ(a)=−upper(F̃_a), read off the canonical's support against the
                # *already-built* S (no spec, no tRG).  ρ/ρ⁻¹ use these exact maps
                # via the fast section-rectified map (see rho/rho_inverse) instead
                # of the slow tRG transport — the win for theories where no finite
                # spec exists (or extraction is costly).  The provider raises if
                # the F-solve cone is too small (upper on the cone boundary); raise
                # build_S_cutoff until σ stabilises.
                from recursive_spectrum import principled_sigma_maps
                _sig, _sigi = principled_sigma_maps(
                    [list(r) for r in pairing],
                    [tuple(g) for g in self.node_charges],
                    build_S_cutoff, order=build_S_order,
                    built_S=self._built_S,
                )
                self._rho_sigma = _sig            # exact σ   for ρ
                self._rho_sigma_inverse = _sigi   # exact σ⁻¹ for ρ⁻¹
                self._principled_sigma = True
            # Path C (else): ρ/ρ⁻¹ route through the RGKAlgebra parent's
            # tRG-derived map; the generous window above serves the F-solver.
            self.spec = []
            self.cone_gens = list(self.node_charges)
            # Cone-degree cap for the recipe F-solver: enumerate the
            # degree-≤build_S_cutoff cone simplex (not the generous box).  The
            # node charges are a basis of Γ (square, unimodular), so the cone
            # coordinate is M⁻¹·v with M = node charges as columns; the degree is
            # its coordinate sum.  Flavour-safe: matter dressing raises the
            # cone-degree, so the degree cap bounds the flavour reach the box's
            # generous window does not.
            # The cone-degree is the coordinate sum in the node basis,
            # M⁻¹·v with M = node charges as columns — *rational* in general (the
            # node lattice need not be unimodular, e.g. SU(2)+N_f=1 has det 2), so
            # we keep exact Fractions and compare ≤ cutoff (matching how
            # `build_spectrum_generator` truncates the cone).  This is the same
            # cone-degree used to build S, so the cap never over-truncates F.
            try:
                nc = self.node_charges
                D = len(nc)
                Mcols = [[nc[c][r] for c in range(D)] for r in range(D)]
                _Minv = _bps_mat_inverse(Mcols)         # exact Fraction inverse
                # column-summed inverse: deg(v) = Σ_i Σ_j Minv[i][j]·v[j]
                _colsum = [sum(_Minv[i][j] for i in range(D)) for j in range(D)]
                self._sf_degree_fn = (lambda v, _c=_colsum, _D=D: sum(
                    _c[j] * v[j] for j in range(_D)))
                self._sf_max_degree = int(build_S_cutoff)
            except Exception:
                self._sf_max_degree = None  # non-invertible: keep the box
        else:
            # Spec mode: build the chart object (bps.CoulombAlgebra) so we
            # can reuse its quiver / sigma / spec. The actual F-solving
            # and Schur indices go through our parameterized helpers,
            # though, so the inheritance is clean: BPSKAlgebra has its
            # own KAlgebra ops, with bps.CoulombAlgebra as a private
            # spec-finder + cone-witness producer.
            kwargs = {}
            if cone_witness is not None:
                kwargs["cone_witness"] = cone_witness
            if spec is not None:
                kwargs["spec"] = spec
                if verify == "off":
                    kwargs["skip_spec_cone_check"] = True
            elif negating_sequence is not None:
                kwargs["negating_sequence"] = negating_sequence
            if verify == "off":
                kwargs["skip_cone_check"] = True
            self._chart = bps.CoulombAlgebra(
                pairing=pairing,
                node_charges=node_charges,
                **kwargs,
            )
            self.spec = [tuple(g) for g in self._chart.spec]
            # Optional spec shortening via local moves on S.
            # Only run if spec was user-supplied (auto-found / cached
            # specs are already optimal for the search the BFS uses).
            if shorten_spec and spec is not None:
                shorter = _shorten_spec_via_local_moves(
                    self.spec,
                    exchange=[list(row) for row in pairing],
                )
                if len(shorter) < len(self.spec):
                    self.spec = [tuple(g) for g in shorter]
                    # Rebuild the underlying chart with the shorter spec
                    # so downstream consumers (sigma, F, schur_index)
                    # use the optimized one.
                    rebuild_kwargs = dict(kwargs)
                    rebuild_kwargs.pop("negating_sequence", None)
                    rebuild_kwargs.pop("spec", None)
                    rebuild_kwargs["spec"] = self.spec
                    if verify == "off":
                        rebuild_kwargs["skip_spec_cone_check"] = True
                    self._chart = bps.CoulombAlgebra(
                        pairing=pairing,
                        node_charges=node_charges,
                        **rebuild_kwargs,
                    )
            self.cone_gens = [tuple(self.lattice.check(g)) for g in (
                cone_gens if cone_gens is not None else self._chart.cone_gens
            )]
            spec_t = list(self.spec)
            N = len(spec_t)
            kmat = [[self.lattice.bracket(spec_t[i], spec_t[j]) for j in range(N)]
                    for i in range(N)]
            self._kmat = kmat
            self._s_coefficient_fn = lambda gamma: s_gamma_habiro(
                tuple(gamma), spec_t, kmat
            )
            self._sigma_fn = lambda gamma: tuple(
                sigma_forward(self.lattice, spec_t, tuple(gamma))
            )
            self._sigma_inverse_fn = lambda gamma: tuple(
                _spec_sigma_inverse(self.lattice, spec_t, tuple(gamma))
            )

        self._verify_mode = verify

        # Strict cone witness for _decompose_in_F_basis.
        if cone_witness is not None:
            self._cone_witness: tuple[int, ...] = tuple(int(x) for x in cone_witness)
        else:
            self._cone_witness = compute_strict_cone_witness(
                self.lattice.rank, self.cone_gens,
            )

        # ----- caches -----
        # F-coefficients are palindromic Laurent polynomials in q; stored
        # natively as QNumberPoly (integer combinations of [n]_q quantum
        # integers).  Internal helpers (`_F_internal`, F-basis decomposition,
        # FS warm-up, chart-graph shortening) consume QNumberPoly directly;
        # the public `F(gamma)` accessor converts to LaurentPoly at the
        # boundary for backward compatibility.
        self._F_cache: dict[Vec, dict[Vec, QNumberPoly]] = {}
        self._FS_cache: dict[tuple[Vec, Vec], HabiroElement] = {}
        # Multiply cache, keyed by (sec(a), sec(b)) — flavour-shifted inputs
        # reuse the orbit-pair entry with a μ-monomial twist.
        self._multiply_cache: dict[tuple[Vec, Vec], Element] = {}

        # ----- lazy chart graph (private; no public surface in phase 2) -----
        # Built only in spec mode (recipe mode has no quiver to mutate).
        # The graph is lazy: only the root chart is materialized at
        # construction; further charts appear on demand when internal
        # algorithms call `_chart_graph.mutate(...)`.
        if self._chart is not None:
            self._chart_graph: ChartGraph | None = ChartGraph(
                nodes=self.node_charges,
                spec=self.spec,
                pairing=[list(row) for row in self.lattice.pairing],
            )
        else:
            self._chart_graph = None

    # ====================================================================
    # 2. KAlgebra contract: intrinsic ops, computed via the chart
    # ====================================================================

    # Coefficient ring: derived from ker(B) at construction time.
    #   * R = TrivialZPlusRing() when ker(B) = 0 (unflavoured).
    #   * R = AbelianZPlusRing(rank=f) when ker(B) has rank f (abelian flavour).

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self) -> Vec:
        return tuple(0 for _ in range(self.lattice.rank))

    # `multiply` is **inherited from `RGKAlgebra`**.
    # The generic section-keyed cached multiply —
    # `from_ir_image(RG(sec a)·RG(sec b))` decomposed in the F-basis and
    # translated by the flavour shift `flav(a)+flav(b)` — is *identical* to
    # (and as fast as) BPS's former hand-rolled version: validated 0
    # mismatch and matching speed on the deep flavoured benchmark.  The
    # generic cone-minimal apex peel uses `grading()`'s height
    # (= `_cone_witness`), so it coincides with the old `find_lowest`
    # cone-peel.  `_multiply_cache` is the same attribute, populated by the
    # generic, so `save_cache`/`load_cache` are unaffected.

    def _sec_rectified_map(self, a, chart_fn) -> Vec:
        """Lift a chart-level σ-map to the **canonical-label** action.

        A canonical label of a sectioned (flavoured) algebra is a pair
        `(section label, R element)`: `L_a = μ^f · L_{sec(a)}`.  The
        axiom-faithful ρ acts on pairs — by the chart map on the
        section and by `⋆` (flavour inversion) on the central R-part —
        so on full-Γ labels:

            ρ(L_a) = ρ(μ^f · L_{sec(a)}) = μ^{−f} · L_{σ(sec(a))}
            ⇒  ρ_can(a) = σ(sec(a)) − f·κ        (κ = ker-basis)

        The raw chart map (piecewise-linear σ) is *not* globally
        ⋆-equivariant in the flavour direction across its linearity
        cones, so applying it to full labels directly violates the
        axioms by central flavour shifts;
        rectifying through the section restores exactness.  Unflavoured
        algebras: the chart map is used as-is."""
        a_t = tuple(self.lattice.check(a))
        if self._flavour_rank == 0:
            return tuple(chart_fn(a_t))
        n = self.lattice.rank
        sec_c, flav_c = decompose_in_basis(
            a_t, self._sec_basis, self._ker_basis,
        )
        sec_label = tuple(
            sum(sec_c[i] * self._sec_basis[i][k] for i in range(len(sec_c)))
            for k in range(n)
        )
        img = tuple(chart_fn(tuple(sec_label)))
        return tuple(
            img[k] - sum(flav_c[i] * self._ker_basis[i][k]
                         for i in range(len(flav_c)))
            for k in range(n)
        )

    def rho(self, a) -> Vec:
        """Canonical-label ρ (axiom-faithful): the section-rectified
        σ-action — see `_sec_rectified_map`.  The raw chart σ remains
        available internally (`self._sigma_fn`) for chart-level
        purposes (tropical labels, F-support, mutation bookkeeping).

        In **spec-free mode** there is no combinatorial σ; ρ is derived from
        the built `S` via the RGKAlgebra parent's tRG intertwining
        (`ρ_UV = from_ir_image(ρ_IR(tRG(a)))`) — unless the **principled** σ
        (`spec_free_sigma="principled"`) is installed, in which case ρ uses the
        same fast section-rectified map as spec mode, with the exact
        σ(a)=−upper(F̃_a) from the built S (`_rho_sigma`, kept separate from the
        generous `_sigma_fn`/`_sigma_inverse_fn` that bound the F-solver)."""
        if self._principled_sigma:
            return self._sec_rectified_map(a, self._rho_sigma)
        if getattr(self, "_spec_free", False):
            return RGKAlgebra.rho(self, a)
        return self._sec_rectified_map(a, self._sigma_fn)

    def rho_inverse(self, a) -> Vec:
        if self._principled_sigma:
            return self._sec_rectified_map(a, self._rho_sigma_inverse)
        if getattr(self, "_spec_free", False):
            return RGKAlgebra.rho_inverse(self, a)
        return self._sec_rectified_map(a, self._sigma_inverse_fn)

    def _label_section_decompose(self, label):
        """Decompose a Z-form label into (section_rep, r_coefficient).
        Used by the default `to_R_form` to produce the R-form
        presentation.

        `r_coefficient` is a single-basis-element RElement μ^{flav_c}
        in the (abelian) flavour ring; on an unflavoured BPSKAlgebra
        (`TrivialZPlusRing`), it is `R.one()`.

        `r_label_decompose` returns the same `(section, flav_c)` split
        with the flavour irrep as a bare key (not wrapped as an `RElement`)
        and is preferred for new code; this method is kept because
        `to_R_form` still routes through it."""
        a_t = tuple(self.lattice.check(label))
        n = self.lattice.rank
        sec_c, flav_c = decompose_in_basis(
            a_t, self._sec_basis, self._ker_basis,
        )
        sec_label = tuple(
            sum(sec_c[i] * self._sec_basis[i][k] for i in range(len(sec_c)))
            for k in range(n)
        )
        R = self._R
        return sec_label, R.basis_element(tuple(flav_c))

    def embed_R(self, r: RElement) -> Element:
        """Central embedding `R ↪ A_𝖖[T]` for the (abelian) flavour ring.

        Each flavour character `μ^{flav}` (a basis element of
        `AbelianZPlusRing`) maps to the central canonical basis element
        `L_{γ_f}` at the flavour charge
        `γ_f = Σ_i flav[i] · ker_basis[i] ∈ Γ_f = ker(B)`; extended
        Z-linearly.  Inverse-compatible with `_label_section_decompose`:
        the faithfulness axiom `embed_R(r_coeff(a)) · L_{section(a)} ==
        L_a` holds because `γ_f` is central, so `X_{γ_f} · X_{sec} =
        X_{γ_f + sec}` with no q-twist.  Unflavoured (`TrivialZPlusRing`)
        falls back to the base default (`1_R ↦ identity`).

        The label-level reconstruction (`section + γ_f`) is also available
        as `r_label_compose`, a direct central lattice sum with no ring
        embedding, preferred for new code; this method is kept because
        `from_R_form` still uses the R-linear ring embedding.
        """
        R = self.coefficient_ring()
        if not isinstance(r, RElement) or r.ring != R:
            raise TypeError(
                "embed_R: argument must be an RElement over coefficient_ring()"
            )
        if self._flavour_rank == 0:
            return super().embed_R(r)
        n = self.lattice.rank
        out = Element.zero()
        for flav, coeff in r.terms.items():
            if coeff == 0:
                continue
            gamma_f = tuple(
                sum(flav[i] * self._ker_basis[i][k]
                    for i in range(len(flav)))
                for k in range(n)
            )
            out = out + Element.basis(gamma_f) * coeff
        return out

    def r_label_decompose(self, label):
        """The single-irrep **flavour-lift coordinate** `(section, flavour_key)`
        — the contract's optional flavour-lift method.

        `section` is the Γ-internal lift of the gauge class via `_sec_basis`
        (a *section of the projection* `Γ → Γ/Γ_f`); `flavour_key` is the SNF
        kernel-coordinate tuple of `μ^{flav}` — a **single** flavour irrep
        (`Γ_f`-flavour is abelian, so always one `AbelianZPlusRing` basis
        element), `()` on an unflavoured BPSKAlgebra.

        Supersedes `_label_section_decompose`: the same `(section, flav_c)`
        split, but the irrep is returned directly rather than wrapped as an
        `RElement`.  Mirrors `QuantumTorusKAlg` (BPS's auxiliary), built from
        the same `integer_kernel_and_section(B)` — so this override agrees with
        the inherited `RGKAlgebra` delegation `aux.r_label_decompose(apex(·))`
        at BPS's identity apex; we keep the explicit override for consistency
        with BPS's own `_label_section_decompose`/`embed_R`."""
        a_t = tuple(self.lattice.check(label))
        n = self.lattice.rank
        sec_c, flav_c = decompose_in_basis(
            a_t, self._sec_basis, self._ker_basis,
        )
        sec_label = tuple(
            sum(sec_c[i] * self._sec_basis[i][k] for i in range(len(sec_c)))
            for k in range(n)
        )
        return sec_label, tuple(flav_c)

    def r_label_compose(self, section, flavour_key):
        """Inverse of `r_label_decompose`: lift the `section` (a Γ-tuple in
        `span(_sec_basis)`) by the central flavour charge
        `γ_f = Σ_j flavour_key[j]·ker_basis[j] ∈ Γ_f = ker(B)`.

        A direct lattice sum — `Γ_f` is central (`⟨Γ_f, ·⟩ = 0`), so there is
        no q-phase and no `embed_R`/`multiply` round-trip: this is the
        **label-level** replacement for `embed_R` (the per-label reconstruction
        the faithfulness axiom needs)."""
        section = tuple(self.lattice.check(section))
        flavour_key = tuple(int(x) for x in flavour_key)
        if len(flavour_key) != self._flavour_rank:
            raise ValueError(
                f"r_label_compose: flavour_key length {len(flavour_key)} != "
                f"flavour_rank {self._flavour_rank}"
            )
        n = self.lattice.rank
        return tuple(
            section[k] + sum(flavour_key[j] * self._ker_basis[j][k]
                             for j in range(self._flavour_rank))
            for k in range(n)
        )

    def _trace_uncached(self, a, K: int = 20, **kwargs):
        """`Tr(L_a) = (q²;q²)_∞^r · ⟨S | F(L_a) S⟩` via Nahm sums, for a
        **canonical label** `a` — the BPS direct trace primitive.

        The overridable hook behind `RGKAlgebra.trace` (which memoises);
        a separate memoized entry point that delegates to the same Schur
        pairing (`_inner_product_uncached(identity, a)`).  It is
        the `a = 1` face of the Schur pairing — exact, free of the
        off-diagonal bar-asymmetry (limitation (E)) of the general
        `_inner_product_uncached`.  kwargs (e.g. `cone_cutoff`) forward to
        the Schur core."""
        return self._inner_product_uncached(self.identity(), tuple(a), K,
                                            **kwargs)

    def _inner_product_uncached(self, a, b, K: int = 20, *,
                                cone_cutoff: int | None = None):
        """`I_{a, b}(q) = (q²;q²)_∞^r · ⟨F(L_a) S | F(L_b) S⟩` via Nahm sums.

        The overridable hook behind `RGKAlgebra.inner_product` (the BPS
        Schur accelerator); per-`(a, b)` value memoization lives in that
        inherited wrapper, so every BPS trace / inner product is cached
        without a BPS-specific cache.

        **Sound support.**  The η-region summed over is grown
        adaptively until a *two-cutoff stability* certificate holds: the
        result is recomputed on a widening cone-witness shell
        ``⟨f, γ⟩ ≤ B`` until successive shells agree to ``q^K``.  This
        replaces the former one-shot heuristic shell (`_effective_cone_cutoff`)
        and the `K_joint` linear joint-bound prune, both of which silently
        **under-included** contributing charges in mixed-sign / sheared
        frames.  The assembled leading q-order of ``[F·S]_η`` grows
        *super-linearly* with the charge but can dip below any *linear*
        witness bound (the "magical cancellations"), so a linear prune is
        not a valid lower bound — an earlier implementation that used one
        computed a frame-dependent, wrong vacuum character on
        sheared unimodular frames, e.g. the pentagon `T=[[1,1],[0,1]]`.

        Soundness of the adaptive shell: the n-driven Nahm walk over
        ``⟨f, γ⟩ ≤ B`` is *hole-free* (it reaches every charge in the shell,
        including central / flavour charges in ``ker B``, via their own Nahm
        tuples) and terminates (each cone generator adds ``≥ 1`` to
        ``⟨f, ·⟩``); the true support ``{η : k_min([F·S]_η) ≤ K}`` is finite
        (super-linear growth), so some finite ``B`` contains it; once
        ``B ⊇ support`` further widening adds only ``k_min > K`` charges that
        are invisible to ``expand(K)`` — hence the value stabilises exactly
        at the correct, *frame-independent* answer.  The shell stays small
        when the support is small, so the e8-scale memory win the former
        prune was reaching for is preserved (achieved soundly, by the
        assembled order rather than a linear proxy).

        `cone_cutoff` (optional) only raises the *starting* shell; the
        adaptive growth handles correctness regardless, so it is now a perf
        hint, not a correctness knob.  Habiro-element c-data is cached in
        `self._FS_cache` and reused across shell widenings (incremental).

        Known limitations (unchanged by the support fix):

        * **(E)** Bar on the `a` side acts on the μ-component (the
          ``mu_exp = -fa + fb`` line in ``_schur_index``) but not on the
          q-component of ``ha``.  At q⁰ invisible; c-data is bar-symmetric
          in q on every theory tested.
        * **(F)** Sentinel ``F is None`` ↔ ``a == identity()``.
        """
        a = tuple(a)
        b = tuple(b)
        zero = self.identity()
        F_a = self._F_internal(a) if a != zero else None
        F_b = self._F_internal(b) if b != zero else None
        a_label = None if a == zero else a
        b_label = None if b == zero else b

        if not self.spec:
            # Recipe mode (no BPS spec): one-shot shell.  The support bug is
            # specific to the spec-mode Nahm enumeration; recipe mode walks
            # `_s_coefficient` directly and is left unchanged.
            eff = self._effective_cone_cutoff(
                F_a, F_b, K, 12 if cone_cutoff is None else cone_cutoff)
            return self._schur_index(F_a, F_b, K, a_label, b_label, eff)

        return self._schur_index_stable(
            F_a, a_label, F_b, b_label, K, cone_cutoff)

    # Two-cutoff stability schedule for the adaptive Schur shell.
    _STABILITY_STEP = 2          # cone-witness units added per widening
    _STABILITY_WINDOW = 2        # successive equal results required to certify
    _STABILITY_MAX_ITERS = 48    # generous cap (false-negative-safe)

    def _schur_index_stable(self, F_a, a_label, F_b, b_label, K, cone_cutoff):
        """`inner_product` spec-mode core: grow the cone-witness shell until a
        two-cutoff stability certificate holds (see `inner_product` for the
        soundness argument).

        Returns the stabilised `RPowerSeries`.  On budget exhaustion without
        a certificate (the support failed to stabilise within
        ``start + STEP·MAX_ITERS``), emits a `RuntimeWarning` and returns the
        widest computed value — a real signal for the theory at hand, not a
        silent default.
        """
        # Start from the *auto* shell (un-floored by the legacy
        # cone_cutoff=12 default, which is exactly what made the e8 vacuum
        # shell explode on the 8-dim lattice); honour an explicit caller
        # floor only if larger.
        start = self._effective_cone_cutoff(F_a, F_b, K, 0)
        if cone_cutoff is not None:
            start = max(start, int(cone_cutoff))

        eff = start
        prev = None
        equal_run = 0
        last = None
        for _ in range(self._STABILITY_MAX_ITERS):
            self._warm_fs_cache_for_schur(F_a, a_label, F_b, b_label, eff, K)
            val = self._schur_index(F_a, F_b, K, a_label, b_label, eff)
            if prev is not None and val == prev:
                equal_run += 1
                if equal_run >= self._STABILITY_WINDOW:
                    return val
            else:
                equal_run = 0
            prev = val
            last = val
            eff += self._STABILITY_STEP

        import warnings
        warnings.warn(
            f"{type(self).__name__}.inner_product: Schur shell did not reach "
            f"a two-cutoff stability certificate within "
            f"{self._STABILITY_MAX_ITERS} widenings "
            f"(eff={eff - self._STABILITY_STEP}, K={K}); returning the widest "
            f"computed value, which may be under-converged.",
            RuntimeWarning, stacklevel=3,
        )
        return last

    def _schur_index(
        self,
        F_a: dict[Vec, QTLaurentPoly] | None,
        F_b: dict[Vec, QTLaurentPoly] | None,
        K: int,
        a_label: Vec | None,
        b_label: Vec | None,
        eff_cutoff: int,
    ) -> RPowerSeries:
        """`I_{a,b}(q, μ) = (q²;q²)_∞^{rk Γ_g} · Σ_{[η] ∈ Γ_g} c_a([η]; μ) c_b([η]; μ)`
        as an `RPowerSeries` over `R((q))` with `R` the algebra's
        coefficient ring (`TrivialZPlusRing` for unflavoured theories,
        `AbelianZPlusRing(rank=f)` otherwise).

        For each output charge `η ∈ Γ`, decompose `η = sec_part + flav_part`
        against the SNF section.  The "gauge-class coefficient"
        `c_a([η]; μ) := Σ_{η' : sec(η') = sec(η)} c_a(η') · μ^{flav_part(η')}`
        sums HabiroElement contributions with μ-monomial weights.  The
        product `c_a([η]; μ) · c_b([η]; μ)` is a sum of (μ-monomial,
        Habiro) pairs; we accumulate per gauge class and per μ-monomial,
        then expand each Habiro to a PowerSeries(K), multiply by
        `(q²;q²)_∞^g`, and pack into the final RPowerSeries.
        """
        R = self._R
        sec_basis = self._sec_basis
        ker_basis = self._ker_basis
        f = self._flavour_rank
        g = self._gauge_rank
        fs_cache = self._FS_cache

        output_charges = _enumerate_output_charges(
            [F_a, F_b], self.cone_gens, self.lattice.rank,
            eff_cutoff,
            cone_witness=self._cone_witness,
            # The K_joint linear joint-bound prune is disabled: it was a
            # linear lower bound on a *super-linear* assembled q-order, so it
            # under-included contributing charges on mixed-sign / sheared
            # frames.  Soundness comes instead from
            # the adaptive two-cutoff-stability shell (`_schur_index_stable`),
            # which widens `eff_cutoff` until the q^K result stabilises; here
            # `eff_cutoff` is that already-chosen shell.
            K_joint=None,
        )

        def _get_c(label, F, eta):
            key = (label, eta)
            if key not in fs_cache:
                fs_cache[key] = c_gamma_via_s(
                    eta, F, self._s_coefficient, self.lattice,
                )
            return fs_cache[key]

        # Group c_a(η) and c_b(η) by (gauge-class sec_c, flavour flav_c)
        # as *lists* of HabiroElements, summed at the end.  Bulk
        # accumulation (HabiroElement.sum once per bucket) keeps the
        # `simplify` cost down — one simplify per (sec_c, flav_c) bucket
        # instead of one per η contribution.
        c_a_terms: dict[tuple, dict[tuple, list[HabiroElement]]] = {}
        c_b_terms: dict[tuple, dict[tuple, list[HabiroElement]]] = {}

        for eta in output_charges:
            sec_c, flav_c = decompose_in_basis(eta, sec_basis, ker_basis)

            # Compute c_a(η) and c_b(η) for *every* η in the output cone
            # — including when F_a = None (a = identity), in which case
            # c_a(η) = [S|0⟩]_η is generally non-zero for η ≠ 0.
            #
            # (An earlier optimisation here restricted the η-loop to
            # η = 0 when F_a (or F_b) was None.  That dropped legitimate
            # [S|0⟩]_η contributions and broke the ρ²-twisted trace
            # cyclicity on flavoured theories.)
            c_a_h = _get_c(a_label, F_a, eta)
            if not c_a_h.is_zero():
                c_a_terms.setdefault(sec_c, {}).setdefault(flav_c, []).append(c_a_h)

            c_b_h = _get_c(b_label, F_b, eta)
            if not c_b_h.is_zero():
                c_b_terms.setdefault(sec_c, {}).setdefault(flav_c, []).append(c_b_h)

        # Bulk-sum each (sec_c, flav_c) bucket.
        c_a_per_sec: dict[tuple, dict[tuple, HabiroElement]] = {
            sec_c: {fa: HabiroElement.sum(lst) for fa, lst in d.items()}
            for sec_c, d in c_a_terms.items()
        }
        c_b_per_sec: dict[tuple, dict[tuple, HabiroElement]] = {
            sec_c: {fb: HabiroElement.sum(lst) for fb, lst in d.items()}
            for sec_c, d in c_b_terms.items()
        }

        # For each shared gauge class, multiply c_a · c_b and bucket
        # by μ-monomial.  Again accumulate as lists, sum at the end.
        # I_{a,b} = ⟨F_a S | F_b S⟩_IR is bar-anti-linear in the `a`
        # side: μ-side bar negates fa.  q-side bar at q^0 is trivial
        # (integer coefficients); higher q-orders would need a q-bar
        # on `ha` (HabiroElement.bar isn't exposed — μ-bar only here).
        overlap_terms: dict[tuple, list[HabiroElement]] = {}
        shared = c_a_per_sec.keys() & c_b_per_sec.keys()
        for sec_c in shared:
            a_dict = c_a_per_sec[sec_c]
            b_dict = c_b_per_sec[sec_c]
            for fa, ha in a_dict.items():
                for fb, hb in b_dict.items():
                    mu_exp = tuple(-fa[i] + fb[i] for i in range(f))
                    prod = ha * hb
                    if prod.is_zero():
                        continue
                    overlap_terms.setdefault(mu_exp, []).append(prod)
        overlap_per_mu: dict[tuple, HabiroElement] = {
            mu_exp: HabiroElement.sum(lst)
            for mu_exp, lst in overlap_terms.items()
        }

        # Compute (q²;q²)_∞^g as a PowerSeries (μ-independent).
        pf = qpoch_infty(K)
        pf_g = pf
        for _ in range(g - 1):
            pf_g = pf_g * pf

        # Pack everything into RPowerSeries[R].
        # final_coeffs[q_exp] is an RElement (μ-Laurent at this q-power).
        final_coeffs: dict[int, RElement] = {}
        for mu_exp, h in overlap_per_mu.items():
            ps = _habiro_to_ps(h, K)
            scaled = pf_g * ps
            mu_basis_elem = R.basis_element(mu_exp)
            for q_exp, q_coeff in scaled._c.items():
                if q_coeff == 0:
                    continue
                term = q_coeff * mu_basis_elem
                if q_exp in final_coeffs:
                    final_coeffs[q_exp] = final_coeffs[q_exp] + term
                else:
                    final_coeffs[q_exp] = term

        # Drop any q-coefficients that summed to zero.
        return RPowerSeries(
            R,
            {q: c for q, c in final_coeffs.items() if not c.is_zero()},
            K,
        )

    # ====================================================================
    # 3. The RG flow: F (the canonical-basis morphism into the QT) plus
    #    `spectrum_generator` for diagnostic inspection of `S`.
    #
    # The `auxiliary()` / `RG()` / `rg_generator()` methods below are the
    # `RGKAlgebra` contract realised by BPSKAlgebra: the "RG flow" *is*
    # the canonical-basis morphism into the quantum torus.
    # ====================================================================

    # -------- RGKAlgebra contract ----------------------------------------

    def auxiliary(self) -> KAlgebra:
        """The RG-flow target: the quantum torus K-algebra on `(Γ, B)`.

        The same `QuantumTorusKAlg` instance is returned every call; this
        is needed for `RGKAlgebra.then(...)` composition to validate
        endpoints by Python `is` identity.
        """
        return self._qt_aux

    def grading(self):
        """The `Γ_RG`-grading making `BPSKAlgebra` a *complete*
        `RGKAlgebra`: `deg(X_γ) = γ` (`Γ_RG = Γ`, the full charge
        lattice), height = a cone-positive central charge.

        `deg` is the identity on the quantum torus, so every graded piece
        `B_p` is one-dimensional (`label ↔ charge` is a bijection) — the
        structural fact behind BPS's sharp F-solve / `σ` optimizations.

        The **height is `self._cone_witness`** — the *strict cone-positive
        functional* BPS already computes (`compute_strict_cone_witness`)
        and uses in `find_lowest` to peel the cone-minimum charge.  So the
        generic `RGKAlgebra.from_ir_image`'s apex (`argmin` of the height)
        **coincides with BPS's own cone-minimal peel**, robustly on
        *every* pointed cone (including oblique ones) — not just the
        `(1,…,1)`-friendly standard dictionary.

        Provided for **completeness / the generic-RGKAlgebra path** (e.g.
        cross-validation): `BPSKAlgebra` overrides every `KAlgebra` /
        `RGKAlgebra` operation with its own faster chart-based version, so
        its own ops never call `grading()`.
        """
        from grading import Grading
        rank = self.lattice.rank
        witness = getattr(self, "_cone_witness", None)
        height = (tuple(int(x) for x in witness) if witness is not None
                  else (1,) * rank)
        # Spec-free mode derives ρ via the generic tRG path; carry `cone_gens`
        # so it takes the **exact** `solve_trg_exact` route (per-charge oracle
        # `_s_rg_component` = built-S lookup, cutoff-free) rather than the slow
        # q-order-windowed `rg_generator` fallback.  (Spec mode overrides ρ with
        # the combinatorial σ and never consults this, so it is left untouched.)
        cone_gens = (tuple(tuple(g) for g in self.cone_gens)
                     if getattr(self, "_spec_free", False) else None)
        return Grading(rank=rank, deg=lambda lbl: tuple(lbl), height=height,
                       cone_gens=cone_gens)

    def RG(self, a) -> Element:
        """`RG(L_a) = F(L_a)` lifted into the auxiliary's `Element` type.

        `BPSKAlgebra.F(a)` returns `F(L_a)` as `dict[Vec, LaurentPoly]`
        with integer q-coefficients; here we wrap each LaurentPoly as
        an `RLaurent` over the auxiliary's coefficient ring (which
        equals `self._R` by the construction-time assertion in
        `__init__`).  Integer q-coefficients embed at the identity
        basis element of `R`.
        """
        a_t = tuple(self.lattice.check(a))
        F_a = self._F_internal(a_t)
        # F_a is dict[Vec, QNumberPoly]; Element expects LaurentPoly
        # (Z[q^±]) coefficients, so we
        # convert at the boundary.
        terms: dict[Vec, "LaurentPoly"] = {}
        for label, qn in F_a.items():
            if not qn.is_zero():
                terms[label] = qn.to_laurent()
        return Element(terms)

    def _s_rg_charges_to_height(self, B: int):
        """`S_RG` charge window `{γ in cone : ⟨f, γ⟩ ≤ B}` via the
        cone-witness L-shell BFS (the grading height *is* the cone
        witness here)."""
        return _enumerate_output_charges(
            [{self.identity(): 1}], self.cone_gens, self.lattice.rank,
            B, cone_witness=self._cone_witness,
        )

    def rg_times_s_rg(self, a, k: int) -> Element:
        """Certified FS object `F(L_a)·S` to q-order `k` — the BPS
        realisation of the contract (`rgkalgebra.rg_times_s_rg`).

        Two ingredients:

        * **window**: output charges η enumerated by the cone-witness
          L-shell, grown adaptively to a **two-cutoff stability**
          certificate.  The assembled leading order of `[F·S]_η` grows
          *super-linearly* with the charge but the term-wise orders dip
          below any linear bound (the "magical cancellations"), so a
          one-shot linear shell is not a sound lower bound — it can
          under-include (the same root cause as the retired Schur prune's
          unsoundness).
          Widening B only enlarges the η-shell and the exact `expand(k)`
          filter drops η with assembled order > k, so once B covers the
          (finite) support the term-dict is stable and exact;
        * **coefficients**: each `[F·S]_η` evaluated as a **complete
          Nahm sum** (`c_gamma_via_s`, exact `HabiroElement` arithmetic —
          all cancellations internal), expanded to `q^k` only at the end.
        """
        cache = self.__dict__.setdefault("_fs_bps_cache", {})
        a_t = tuple(self.lattice.check(a))
        key = (a_t, k)
        hit = cache.get(key)
        if hit is not None:
            return hit
        zero = self.identity()
        if a_t == zero:
            F_lp = None
            seeds = {zero: 1}
            H = 0
        else:
            F_lp = {d: lp for d, lp in self.F(a_t).items()
                    if not lp.is_zero()}
            seeds = F_lp
            f = self._cone_witness
            H = max(sum(fi * di for fi, di in zip(f, d)) for d in F_lp)

        def _terms_at(B: int) -> dict:
            etas = _enumerate_output_charges(
                [seeds], self.cone_gens, self.lattice.rank, B,
                cone_witness=self._cone_witness,
            )
            terms: dict = {}
            for eta in etas:
                h = c_gamma_via_s(tuple(eta), F_lp, self._s_coefficient,
                                  self.lattice)
                if h.is_zero():
                    continue
                lp = h.expand(k)
                if not lp.is_zero():
                    terms[tuple(eta)] = lp
            return terms

        B = k + max(0, H) + 2
        prev = None
        equal_run = 0
        out_terms: dict = {}
        for _ in range(self._STABILITY_MAX_ITERS):
            terms = _terms_at(B)
            if prev is not None and terms == prev:
                equal_run += 1
                if equal_run >= self._STABILITY_WINDOW:
                    out_terms = terms
                    break
            else:
                equal_run = 0
            prev = terms
            out_terms = terms
            B += self._STABILITY_STEP

        out = Element(out_terms)
        cache[key] = out
        return out

    def rg_generator(self, K: int) -> dict[Vec, "HabiroElement"]:
        """`S = E_q(X_{γ_1}) · ⋯ · E_q(X_{γ_N}) |0⟩` as
        `dict[γ, HabiroElement]`, truncated to leading q-order ≤ K.

        Mathematical contract:
        include γ iff some Nahm tuple `n` with `Σ n_a γ_a = γ` has
        `shift(n) ≤ K`; for each included γ, ``s_γ`` is the *complete*
        Nahm sum (so ``s_γ.expand(K')`` is correct for any
        ``K' ≥ K``). The implementation may over-include γ's whose
        minimum-shift tuple has shift > K; these contribute only
        at q-orders > K and are invisible to ``expand(K)``.

        Implementation: two-stage with lazy
        ``s_γ`` lookup. The inclusion pass walks Nahm tuples
        (``nahm_local.gammas_to_q_order``); per-γ ``s_γ`` is fetched
        via ``nahm_local.s_gamma_habiro`` (already module-level
        cached and shared with the F-solver).

        Spec mode and recipe mode are both supported. Recipe
        mode falls back to a cone-BFS on `_s_coefficient` and uses
        ``HabiroElement.k_min()`` to filter by leading q-order;
        termination requires that ``_s_coefficient(γ)`` becomes
        zero or has leading q-order > K along every cone-positive
        ray (true for cone-positive spec / recipe specifications).
        """
        out: dict[Vec, HabiroElement] = {self.identity(): HabiroElement.one()}

        if self.spec:
            from nahm_local import gammas_to_q_order
            gammas = gammas_to_q_order(
                self.spec, self._kmat, K, rank=self.lattice.rank,
            )
            for gamma in gammas:
                if gamma == self.identity():
                    continue   # identity already inserted (s_0 = 1)
                h = s_gamma_habiro(gamma, self.spec, self._kmat)
                if h.is_zero():
                    continue
                out[gamma] = h
            return {g: h for g, h in out.items() if not h.is_zero()}

        # Recipe mode: cone-BFS pruned by leading q-order.
        from collections import deque
        seen: set[Vec] = set()
        queue: deque = deque([self.identity()])
        while queue:
            gamma = queue.popleft()
            if gamma in seen:
                continue
            seen.add(gamma)
            s = self._s_coefficient(gamma)
            if s.is_zero():
                continue
            k_min = s.k_min()
            if k_min is None or k_min > K:
                continue
            if gamma != self.identity():
                out[gamma] = s
            for g in self.cone_gens:
                nxt = tuple(c + gi for c, gi in zip(gamma, g))
                if nxt not in seen:
                    queue.append(nxt)
        return {g: h for g, h in out.items() if not h.is_zero()}

    # -------- catalogue-friendly verifier shorthand ----------------------

    def verify_canonical_basis(
        self,
        K: int = 4,
        *,
        label_window: "list[Vec] | None" = None,
    ) -> dict[str, bool]:
        """Run the four currently-meaningful canonical-basis axiom
        checks on a curated label window and return a structured dict.

        Result keys:
            "unital"          : `verify_rg_unital()` (RG(1) = 1).
            "multiplicative"  : `verify_rg_multiplicative(a, b)` over
                                 the window (RG is multiplicative on
                                 the canonical basis).
            "bar_invariant"   : `verify_rg_bar_invariant(a)` over the
                                 window (F_a's q-coefficients are
                                 palindromic).
            "orthonormality"  : `verify_orthonormality(a, b, K)` over
                                 the window
                                 (`I_{a,b} = δ_{a,b} + O(q)`).

        The intertwining identity `F(L_a) · S = S · ρ_QT(F(L_{σ(a)}))`
        and the Schur transport identity are *not*
        included.  Both involve subtleties around the
        truncation-window of `S` that the abstract `RGKAlgebra`
        verifiers (`verify_rg_twist`, `verify_rg_inner_product`)
        don't handle correctly: at any finite cutoff, the truncated
        `S_RG` produces boundary residuals that propagate as
        spurious low-q contributions through `Element` multiplication.
        Checking them would require an exact-arithmetic intertwining
        verifier with proper support analysis, which is not provided.

        Flavoured theories
        ------------------
        With the canonical basis over Z and flavour shifts carried in
        full-Γ labels, **all four checks are meaningful
        on flavoured theories, and a `False` result should be read as a
        real failure.**  Validated on the flavoured
        hexagon and an A3-chain (both rank-1 `ker B`): all four checks
        pass.

        The `verify_orthonormality` check IS included here because
        `BPSKAlgebra.inner_product` is computed via the single-Habiro-
        path Schur formula directly (no Element multiplication), so
        its q^0 result is exact.  For ``b == a`` we expect
        `I_{a,a}[q^0] = 1`; for ``b != a`` we expect 0.

        Default `label_window`: `[identity()] + node_charges +
        [σ(g) for g in node_charges]`, which is small but exercises
        the σ-action.

        Parameters
        ----------
        K
            q-order at which `verify_orthonormality` is checked.
        label_window
            Iterable of canonical-basis labels to use as the test
            window.  Defaults to the curated set above.

        Returns
        -------
        dict[str, bool]
            See keys above.  Each value is `True` iff every per-pair
            check inside the window passes.
        """
        if label_window is None:
            label_window = self._default_verifier_window()
        labels = [tuple(g) for g in label_window]

        out: dict[str, bool] = {}

        out["unital"] = self.verify_rg_unital()

        mult_ok = True
        for a in labels:
            for b in labels:
                if not self.verify_rg_multiplicative(a, b):
                    mult_ok = False
                    break
            if not mult_ok:
                break
        out["multiplicative"] = mult_ok

        out["bar_invariant"] = all(
            self.verify_rg_bar_invariant(a) for a in labels
        )

        ortho_ok = True
        for a in labels:
            for b in labels:
                if not self.verify_orthonormality(a, b, K=K):
                    ortho_ok = False
                    break
            if not ortho_ok:
                break
        out["orthonormality"] = ortho_ok

        return out

    def _default_verifier_window(self) -> "list[Vec]":
        """Curated label set for `verify_canonical_basis`'s default.

        Identity + node charges that lie in the SNF section image
        (i.e., have zero flavour-direction component).

        For unflavoured theories every label is automatically in the
        section image. For flavoured theories some node charges may
        carry non-trivial flavour direction; those are filtered out
        of the default window because the abstract
        `verify_orthonormality(a, b, K)[q^0] == δ_{a,b} ∈ R.one()`
        check expects the q^0 result to be `R.one()` (= μ^0). For
        non-section-image labels `a` with flavour exponent
        `f = flav(a) ≠ 0`, the q^0 result is `μ^{2 f}` — a real
        flavour quantity, not a verifier failure but also not
        captured by the strict `R.one()` comparison. Callers who
        want a wider window can pass `label_window=` explicitly and
        use the R-form view (`to_R_form`) for the proper flavoured
        orthonormality check.
        """
        window: list[Vec] = [self.identity()]
        for g in self.node_charges:
            g_t = tuple(self.lattice.check(g))
            if g_t in window:
                continue
            # Filter to section-image labels (r_coef = identity).
            _, r_coef = self._label_section_decompose(g_t)
            if not r_coef.is_one():
                continue
            window.append(g_t)
        return window

    # -------- BPS-realisation accessors ----------------------------------

    def gamma_lower(self, a) -> Vec:
        """The **lower tropical charge** `γ₋(a)` of the canonical label
        `a` — equal to `a` by construction (the BPS canonical basis is
        indexed by its lower tropical charges).  Returned validated as a
        full Γ-tuple."""
        return tuple(self.lattice.check(a))

    def gamma_upper(self, a) -> Vec:
        """The **upper tropical charge** `γ⁺(a) = ν_S(γ₋(a)) =
        −σ⁻¹(γ₋(a))`.  `F(L_a)` is supported on the doubly-tropical
        interval `[γ₋(a), γ⁺(a)]` (in cone order)."""
        a_t = tuple(self.lattice.check(a))
        return tuple(-x for x in self._sigma_inverse_fn(a_t))

    def tropical_interval(self, a) -> "tuple[Vec, Vec]":
        """The endpoints `(γ₋(a), γ⁺(a))` of the doubly-tropical interval
        on which `F(L_a)` is supported (`F(L_a)`'s charge keys all lie in
        `[γ₋(a), γ⁺(a)] ∩ (γ₋(a) + cone)`)."""
        return self.gamma_lower(a), self.gamma_upper(a)

    def verify_F_S_leading(self, a, K: int = 4) -> bool:
        """Discovery relation `F_γ · S|0⟩ = X_γ + O(q)` at `γ = γ₋(a)`:
        the `q⁰` part of `[F(L_a)·S|0⟩]_η` is `δ_{η, γ₋(a)}` over `F`'s
        support charges `η`.  This is the defining relation by which the
        `BPSKAlgebra` discovers `F_γ` from a charge `γ` (note `F(L_a)`
        itself may carry several `q⁰` charges — they cancel against
        `S|0⟩` to leave the single `X_{γ₋(a)}`)."""
        a_t = tuple(self.lattice.check(a))
        F = self._F_internal(a_t)
        for eta in set(F.keys()) | {a_t}:
            h = c_gamma_via_s(eta, F, self._s_coefficient, self.lattice)
            ps = _habiro_to_ps(h, K)
            if ps[0] != (1 if eta == a_t else 0):
                return False
        return True

    def F(self, a) -> dict[Vec, QTLaurentPoly]:
        """`F(L_a)` for a **canonical label** `a`, as `dict{γ: LaurentPoly}`
        keyed by **QT-index charges** `γ` — the RG image of `L_a` in the
        quantum torus.  `F_{γ₋(a)} · S = X_{γ₋(a)} + O(q)` (`γ₋(a) = a`
        the lower tropical charge), supported on the doubly-tropical
        interval `[γ₋(a), γ⁺(a)]` with `γ⁺(a) = −σ⁻¹(γ₋(a))`.

        Coefficients are palindromic Laurent polynomials in `q` with
        non-negative integer coefficients in the `[n]_q` basis.  Stored
        internally as :class:`QNumberPoly`; converted here to
        :class:`LaurentPoly` for backward-compatible inspection.  Use
        :meth:`F_qn` to retrieve the native q-number representation.
        """
        return {d: qn.to_laurent() for d, qn in self._F_internal(a).items()}

    def F_qn(self, a) -> dict[Vec, QNumberPoly]:
        """`F(L_a)` for a **canonical label** `a`, as `dict{γ: QNumberPoly}`
        keyed by **QT-index charges** `γ`.

        Native form: each `f_γ` is stored as an integer combination of
        quantum integers `[n]_q`.  Mirrors :meth:`F` but skips the
        conversion to LaurentPoly --- preferred for code that does
        further palindromic-poly arithmetic on the result.
        """
        return dict(self._F_internal(a))

    def F_qn_with_guess(
        self, a, guess: dict[Vec, QNumberPoly],
    ) -> tuple[dict[Vec, QNumberPoly], int]:
        """Solve ``F(L_a)`` for a **canonical label** `a`, starting from a
        candidate ``guess`` instead of from scratch.

        Returns ``(F_dict, n_corrections)``: ``F_dict`` is the certified
        true ``F(L_a)`` (built by applying any necessary peels on top of
        the guess); ``n_corrections`` is the total number of ``[n]_q``
        peel corrections performed (``0`` iff the guess was already
        exact at every ``δ``).

        Hook for warm-started F-solves --- e.g. an ML F-predictor
        emits a candidate and this routine certifies / repairs it, or
        a product ``F_{γ'} · F_{γ−γ'}`` palindromised into a
        ``QNumberPoly`` dict serves as the guess.  The result is
        cached in ``_F_cache`` so subsequent ``A.F(a)`` / ``A.F_qn(a)``
        calls reuse it.

        **Cost note:** an exact guess saves ~15-18% wall time
        (peel-decision + correction-propagation drop out), but the
        common-denominator materialisation of ``F_candidate · S``
        still happens at every ``δ`` so the verification is real.
        See :func:`bps_kalgebra_internals.solve_F_with_initial_guess`
        for the full breakdown.  For larger savings on a trusted
        guess, the caller can combine this entry point with an
        external certificate (e.g. zero ``σ``-defect for a product
        reconstruction) and skip the verification pass entirely.
        """
        gamma_t = tuple(self.lattice.check(a))
        if self.spec:
            F_dict, n_corr = solve_F_with_initial_guess(
                self.lattice, self.cone_gens, gamma_t,
                self._s_coefficient, self._sigma_inverse_fn,
                initial_guess=guess,
            )
        else:
            F_dict, n_corr = solve_F_with_initial_guess(
                self.lattice, self.cone_gens, gamma_t,
                self._s_coefficient, self._sigma_inverse_fn,
                initial_guess=guess,
            )
        self._F_cache[gamma_t] = F_dict
        return F_dict, n_corr

    def spectrum_generator(self, K: int) -> dict[Vec, QTLaurentPoly]:
        """`S = E_q(X_{γ_1}) · ⋯ · E_q(X_{γ_N})` in the QT, truncated to q^K.

        Spec mode: one n-driven Nahm-tuple walk produces every
        ``[S|0⟩]_γ`` whose support reaches order ``q^K``.  This is
        equivalent to multiplying out the ``E_q`` factors term by term
        -- the natural form of the spectrum generator -- and avoids
        the per-γ Gaussian-elimination-over-``Fraction`` cost of
        looking up Nahm tuples one γ at a time.

        Recipe mode falls back to the pointwise s-coefficient call.
        """
        from laurent_poly import LaurentPoly as QTLP
        out: dict[Vec, QTLP] = {}

        if self.spec:
            from nahm_local import enumerate_nahm_buckets, _nahm_shift
            from habiro import HabiroElement
            spec_t = self.spec
            kmat = self._kmat

            # The walk's predicate uses `sum(γ) ≤ K + 5` -- the same
            # cone-walk cutoff used previously by the pointwise driver,
            # mirrored here as an n-walk pruning predicate.  Sound for
            # cone-positive spec charges (each γ_i has ``sum(γ_i) > 0``,
            # so partial-γ sums grow monotonically along the recursion).
            cap = K + 5

            def accept(g):
                return sum(g) <= cap

            buckets = enumerate_nahm_buckets(
                spec_t, accept, rank=self.lattice.rank,
            )
            for gamma, tuples in buckets.items():
                terms = []
                for ns in tuples:
                    sign = 1 if sum(ns) % 2 == 0 else -1
                    shift = _nahm_shift(ns, kmat)
                    if shift > K:
                        # Cannot contribute to q^K truncation.
                        continue
                    terms.append(HabiroElement.nahm_term(sign, shift, list(ns)))
                if not terms:
                    continue
                h = HabiroElement.sum(terms)
                if h.is_zero():
                    continue
                lp = h.expand(K)
                if not lp.is_zero():
                    out[gamma] = QTLP(dict(lp._coeffs))
            return out

        # Recipe mode: pointwise fall-back.
        seen: set[Vec] = set()
        frontier = [self.identity()]
        while frontier:
            new_front: list[Vec] = []
            for gamma in frontier:
                if gamma in seen:
                    continue
                seen.add(gamma)
                s = self._s_coefficient(gamma)
                if s.is_zero():
                    continue
                lp = s.expand(K)
                if not lp.is_zero():
                    out[gamma] = QTLP(dict(lp._coeffs))
                for g in self.cone_gens:
                    nxt = tuple(c + gi for c, gi in zip(gamma, g))
                    if sum(nxt) > K + 5:
                        continue
                    new_front.append(nxt)
            frontier = new_front
        return out

    # ====================================================================
    # 4. Root-chart accessors and chart-level transformations
    # ====================================================================

    def root_data(self) -> dict:
        """The BPS-quiver data of the root chart used to construct this
        algebra:

            {"pairing": list[list[int]], "node_charges": list[Vec]}

        The `KAlgebra` contract (multiplication, ρ, trace, inner
        product) is intrinsic to `A_𝖖[T]`.  This method exposes the
        chart-specific data of *this* presentation -- the lattice
        pairing on Γ and the BPS-quiver node charges in Γ.
        """
        return {
            "pairing": [list(row) for row in self.lattice.pairing],
            "node_charges": [tuple(g) for g in self.node_charges],
        }

    def rg_flow(self, node_index: int) -> "BPSKAlgebra":
        """Single-node RG flow: delete the `node_index`-th BPS-quiver
        node and return the resulting `BPSKAlgebra`.

        Same Γ, same `B`; node_charges loses one entry; spec is
        filtered to drop entries containing `γ_{node_index}` as a
        non-zero summand in the old node-charge basis; cached F's are
        filtered analogously on their offsets `δ - γ_a`.

        See `rg_flow.single_node_rg_flow` for the recipe details and
        error semantics.  See `rg_flow_morphism` for the
        `RGKAlgebra`-typed view of the same flow.
        """
        from rg_flow import single_node_rg_flow
        return single_node_rg_flow(self, node_index)

    def rg_flow_morphism(self, node_index: int):
        """The single-node RG flow as an `RGKAlgebra`.

        Returns a `SingleNodeRG` whose `starting_algebra()` is `self`
        and whose `auxiliary()` is the algebra produced by
        `rg_flow(node_index)`.  Conceptually this *expands* `self`
        into an `RGKAlgebra` with a `BPSKAlgebra` auxiliary, without
        changing its intrinsic K-algebra content.
        """
        from rg_flow import SingleNodeRG
        return SingleNodeRG(self, node_index)

    def subquiver_rg_morphism(self, node_indices):
        """Multi-node analogue of `rg_flow_morphism`: return a
        `SubquiverRG` that drops the listed nodes at once.

        Like `rg_flow_morphism`, this *expands* `self` into an
        `RGKAlgebra` with a `BPSKAlgebra` auxiliary -- the IR algebra
        with all listed nodes removed.  Same intrinsic K-algebra,
        different presentation.

        Equivalent (mathematically) to composing
        `rg_flow_morphism(i)` in any order over `i ∈ node_indices`,
        but built in one shot and exposes a uniform `rg_generator(cutoff)`
        that truncates by the *total* dropped multiplicity
        `Σ_{i ∈ node_indices} n_i(δ) ≤ cutoff`.
        """
        from rg_flow import SubquiverRG
        return SubquiverRG(self, node_indices)

    def shorten_spec(self) -> "BPSKAlgebra":
        """Return a new `BPSKAlgebra` whose spec has been shortened by
        local moves (pentagon collapses + commute swaps).

        The intrinsic algebra is unchanged; only the presentation is
        more compact.  Spec mode only.

        On no-op (already shortest), returns a fresh `BPSKAlgebra`
        with the same data — semantically equivalent.
        """
        if self._chart is None:
            raise NotImplementedError(
                "shorten_spec is spec-mode-only"
            )
        shorter = _shorten_spec_via_local_moves(
            self.spec,
            exchange=[list(row) for row in self.lattice.pairing],
        )
        return BPSKAlgebra(
            pairing=[list(row) for row in self.lattice.pairing],
            node_charges=self.node_charges,
            spec=[tuple(g) for g in shorter],
            cone_gens=self.cone_gens,
            verify=self._verify_mode,
            k_joint_prune=self._k_joint_prune,
        )

    # ====================================================================
    # Private: algorithm (c) F-finding via the chart graph,
    # with strategy-(b) used when the chain preserves the boundary.
    # ====================================================================
    #
    # The chart graph tracks the `(S_2, -S_1)` boundary at each chart.
    # When the descent reaches a chart with the boundary still
    # preserved AND a non-empty `s1_chain`, we have the *option* to
    # use the (b) solver: solve `S_1 · F' · S_2 = X_γ + O(q)` and
    # inverse-transport `F'` across `S_1` to get F directly --
    # bypassing the chart-end's standalone canonical-basis solve and
    # transport-back.  When boundary is broken or the (b) solve
    # fails its round-trip check, we fall through to the standard (c)
    # path.

    def _solve_F_via_chain(
        self,
        gamma,
        *,
        max_chain_length: int = 12,
        max_local_moves: int = 64,
    ) -> dict[Vec, QNumberPoly]:
        """Find `F_γ` by greedily walking the chart graph forward, solving
        at the chart where `(u − l)` is smallest in cone-distance, then
        inverse-transporting back to the root.

        Strategy (c): each step picks the forward mutation whose
        destination chart minimizes `L1_in_cone(u' − l', dst.cone)`.
        Stops when no forward mutation strictly shrinks the cost or
        the budget is exhausted.

        Falls back to the standard solver if the chain is empty (no
        forward mutation helps) or recipe mode (no chart graph).

        Returns `dict[Vec, LaurentPoly]` in the ROOT chart's label
        convention, identical to `_F_internal(γ)` modulo numerical
        equivalence.
        """
        if self._chart_graph is None:
            return self._F_internal(gamma)
        from chart_graph import _mu_g, _nu_g
        from spec_shortening import _nu_forward, doubly_tropical_l1_in_cone
        from lattice import LatticeTorus
        from lattice_mutation import solve_inverse as _lm_solve_inverse

        gamma_t = tuple(self.lattice.check(gamma))
        pairing_l = [list(row) for row in self.lattice.pairing]
        graph = self._chart_graph

        # Bounds (l, u) for F at the same intrinsic L_a transform under
        # cluster mutation by their *own* tropical rules:
        #   l → μ_g(l)    (lower tropical: forward μ)
        #   u → ν_g(u)    (upper tropical: ν, the dual map)
        # Node charges and spec mutate by FZ and rotation respectively
        # (in `chart_graph._necklace_forward`).
        l_cur = gamma_t
        u_cur = _nu_forward(self.spec, gamma_t, pairing_l)
        cur_id = graph.root_id
        chain: list = []   # list of Mutation edges traversed

        for _ in range(max_chain_length):
            cur_chart = graph.chart(cur_id)
            if l_cur == u_cur:
                break  # F is already monomial at this chart
            cur_cost = doubly_tropical_l1_in_cone(
                l_cur, u_cur, cur_chart.nodes,
            )
            best = None
            best_cost = cur_cost
            for k in range(len(cur_chart.nodes)):
                dst_id = graph.mutate(
                    cur_id, k, direction="fwd",
                    max_local_moves=max_local_moves,
                )
                if dst_id is None:
                    continue
                edge = graph._edges[(cur_id, dst_id)]
                new_l = _mu_g(l_cur, edge.charge, pairing_l)
                new_u = _nu_g(u_cur, edge.charge, pairing_l)
                dst = graph.chart(dst_id)
                cost = doubly_tropical_l1_in_cone(new_l, new_u, dst.nodes)
                if cost == float("inf"):
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best = (dst_id, edge, new_l, new_u)
            if best is None:
                break
            cur_id, edge, l_cur, u_cur = best
            chain.append(edge)

        # If no chain was found, fall back to root-only solve.
        if not chain:
            return self._F_internal(gamma_t)

        end_chart = graph.chart(cur_id)

        # Strategy-(b) shortcut: if the chain preserved the
        # `(S_2, -S_1)` boundary, we can solve `S_1 · F' · S_2 = X_γ
        # + O(q)` directly for F' (typically small), then inverse-
        # transport through `S_1` to get root-F.
        if end_chart.boundary_preserved and end_chart.s1_chain:
            F_via_b = self._try_solve_F_via_b(
                gamma_t, end_chart, l_cur, u_cur,
            )
            if F_via_b is not None:
                return F_via_b

        # Standard (c): solve F at the end-of-chain chart with
        # (l_cur, u_cur), then inverse-transport back through the chain.
        # Transport runs in LaurentPoly (LatticeTorus' native ring); the
        # result is then re-encoded as palindromic-native QNumberPoly.
        F_end = self._solve_F_at_chart(end_chart, l_cur)
        F_end_lp = {d: qn.to_laurent() for d, qn in F_end.items()}
        cur = LatticeTorus(self.lattice, F_end_lp)
        for edge in reversed(chain):
            cur = _lm_solve_inverse(cur, edge.charge)
        return {
            tuple(d): QNumberPoly.from_palindromic_laurent(c)
            for d, c in cur._terms.items()
        }

    def _try_solve_F_via_b(
        self, gamma, end_chart, l_cur, u_cur,
    ) -> dict[Vec, QNumberPoly] | None:
        """At a chart-end with the (S_2, -S_1) boundary preserved,
        solve the modified equation `S_1 · F' · S_2 = X_γ + O(q)` for
        F', then inverse-transport across S_1 to get F.  Returns the
        dict on success, or `None` if the modified solver fails or
        the round-trip verification rejects the result.
        """
        from bps_kalgebra_internals import solve_F_modified
        from spec_shortening import doubly_tropical_interval_set
        from lattice import LatticeTorus
        from lattice_mutation import (
            solve as _lm_solve, solve_inverse as _lm_solve_inverse,
        )

        S_1 = [tuple(g) for g in end_chart.s1_chain]
        S_2 = [tuple(g) for g in end_chart.spec[: end_chart.s2_length]]

        # F'-support: doubly-tropical interval [l_cur, u_cur] in the
        # chart-end's positive cone.
        cone_end = [tuple(n) for n in end_chart.nodes]
        F_prime_support = doubly_tropical_interval_set(
            l_cur, u_cur, cone_end,
        )
        try:
            F_prime_dict = solve_F_modified(
                self.lattice, S_1, S_2, self.cone_gens,
                gamma, F_prime_support,
            )
        except Exception:
            return None

        # Transport runs in LaurentPoly; convert F' for input and
        # encode the result back as palindromic-native QNumberPoly.
        F_prime_lp = {d: qn.to_laurent() for d, qn in F_prime_dict.items()}
        cur = LatticeTorus(self.lattice, F_prime_lp)
        try:
            for g in reversed(S_1):
                cur = _lm_solve_inverse(cur, g)
        except ValueError:
            return None
        F_recovered_lp = {tuple(d): c for d, c in cur._terms.items()}

        # Round-trip check: forward-transport must reproduce F'.
        try:
            chk = LatticeTorus(self.lattice, F_recovered_lp)
            for g in S_1:
                chk = _lm_solve(chk, g)
        except ValueError:
            return None
        if dict(chk._terms) != F_prime_lp:
            return None
        return {
            d: QNumberPoly.from_palindromic_laurent(c)
            for d, c in F_recovered_lp.items()
        }

    def _multiply_via_chart_search(
        self,
        a, b,
        *,
        max_chain_length: int = 12,
        max_local_moves: int = 64,
    ) -> Element:
        """Find structure constants `F_a · F_b = Σ_c C^c_{ab}(q) F_c`
        by descending to a chart where both factors are short.

        Greedy: at each step, pick the forward mutation that minimizes
        `|F_a in chart| × |F_b in chart|` after transport.  When no
        mutation improves, multiply at the chart-end and decompose
        into the chart-end's canonical basis (via a fresh `BPSKAlgebra`
        on chart-end's `(nodes, spec)`); translate the decomposition
        labels back to root via the inverse μ chain.

        Falls back to root `multiply(a, b)` if the chain is empty.
        """
        if self._chart_graph is None:
            return self.multiply(a, b)
        from chart_graph import _mu_g, _mu_inv_g
        from lattice import LatticeTorus
        from lattice_mutation import solve as _lm_solve

        a_t = tuple(self.lattice.check(a))
        b_t = tuple(self.lattice.check(b))
        pairing_l = [list(row) for row in self.lattice.pairing]
        graph = self._chart_graph

        # Chart-search transport runs in LaurentPoly (LatticeTorus' ring);
        # convert the cached QNumberPoly coefficients up front.
        F_a_root = {d: qn.to_laurent() for d, qn in self._F_internal(a_t).items()}
        F_b_root = {d: qn.to_laurent() for d, qn in self._F_internal(b_t).items()}

        cur_id = graph.root_id
        F_a_cur = dict(F_a_root)
        F_b_cur = dict(F_b_root)
        cur_cost = len(F_a_cur) * len(F_b_cur)
        chain: list = []

        for _ in range(max_chain_length):
            if min(len(F_a_cur), len(F_b_cur)) <= 1:
                break
            cur_chart = graph.chart(cur_id)
            best = None
            best_cost = cur_cost
            for k in range(len(cur_chart.nodes)):
                dst_id = graph.mutate(
                    cur_id, k, direction="fwd",
                    max_local_moves=max_local_moves,
                )
                if dst_id is None:
                    continue
                edge = graph._edges[(cur_id, dst_id)]
                try:
                    F_a_lt = _lm_solve(
                        LatticeTorus(self.lattice, F_a_cur), edge.charge,
                    )
                    F_b_lt = _lm_solve(
                        LatticeTorus(self.lattice, F_b_cur), edge.charge,
                    )
                except ValueError:
                    continue
                F_a_new = dict(F_a_lt._terms)
                F_b_new = dict(F_b_lt._terms)
                cost = len(F_a_new) * len(F_b_new)
                if cost < best_cost:
                    best_cost = cost
                    best = (dst_id, edge, F_a_new, F_b_new)
            if best is None:
                break
            cur_id, edge, F_a_cur, F_b_cur = best
            cur_cost = best_cost
            chain.append(edge)

        if not chain:
            return self.multiply(a_t, b_t)

        # Multiply at chart-end via a fresh BPSKAlgebra.  Tighten the
        # chart-end's spec via local moves before constructing -- spec
        # growth during descent (added by pentagon expansions in the
        # mutation edges' `local_moves`) is fine for getting to a chart
        # where F's are short, but we want the spec compact before
        # multiplying so the chart-c canonical-basis decomposer doesn't
        # pay for redundant factors.
        end_chart = graph.chart(cur_id)
        A_c = BPSKAlgebra(
            pairing=pairing_l,
            node_charges=end_chart.nodes,
            spec=end_chart.spec,
            shorten_spec=True,
            verify="off",
        )
        # Chart-end labels of a, b: μ chain.
        a_c = a_t
        b_c = b_t
        for edge in chain:
            a_c = _mu_g(a_c, edge.charge, pairing_l)
            b_c = _mu_g(b_c, edge.charge, pairing_l)
        decomp = A_c.multiply(a_c, b_c)

        # Translate chart-c labels → root labels via reverse μ chain.
        # `decomp` carries Z[q^±] LaurentPoly coefficients (the
        # Z-form); we keep them as-is during relabeling.
        out: dict[Vec, "LaurentPoly"] = {}
        for label_c, coeff in decomp.terms.items():
            label_root = label_c
            for edge in reversed(chain):
                label_root = _mu_inv_g(
                    label_root, edge.charge, pairing_l,
                )
            out[label_root] = coeff
        return Element(out)

    def _solve_F_at_chart(self, chart, gamma) -> dict[Vec, QNumberPoly]:
        """Solve F at a non-root chart with chart-local label `gamma`.

        Before solving, the chart's spec is **tightened by local moves**
        (pentagon collapses + commute swaps).  Local moves preserve `S`
        as an algebra element, so the tighter spec gives the same
        canonical-basis F -- but the Nahm sums on a shorter spec are
        cheaper.  Spec growth during chart-graph descent (e.g. via
        pentagon expansions in the `local_moves` of mutation edges) is
        absorbed here.
        """
        spec_t = list(chart.spec)
        # Tighten spec via local moves before solving -- this is a
        # gauge-equivalent rearrangement that may shrink the spec
        # without affecting F.
        shorter = _shorten_spec_via_local_moves(
            spec_t,
            exchange=[list(row) for row in self.lattice.pairing],
        )
        if len(shorter) < len(spec_t):
            spec_t = [tuple(g) for g in shorter]
        N = len(spec_t)
        kmat = [[self.lattice.bracket(spec_t[i], spec_t[j])
                 for j in range(N)] for i in range(N)]

        def _s_chart(g):
            return s_gamma_habiro(tuple(g), spec_t, kmat)

        def _sinv_chart(g):
            return tuple(_spec_sigma_inverse(self.lattice, spec_t, tuple(g)))

        cone_chart = [tuple(n) for n in chart.nodes]
        return solve_F_via_s_coefficient(
            self.lattice, cone_chart, gamma, _s_chart, _sinv_chart,
        )

    # ====================================================================
    # Private helpers
    # ====================================================================

    def _decompose_in_F_basis(
        self,
        F_a: dict[Vec, QNumberPoly],
        F_b: dict[Vec, QNumberPoly],
    ) -> dict[Vec, QTLaurentPoly]:
        """Decompose `F_a · F_b` in the F-basis via a cone-order subtraction loop.

        Returns `{c: coeff_c}` with `F_a · F_b = Σ_c coeff_c · F_c`.

        The algorithm: compute the QT product, repeatedly peel off the
        cone-minimum surviving charge (using `find_lowest` with
        `self._cone_witness`), record its coefficient, subtract
        `coeff · F_c` from the remainder.  Terminates when the
        remainder is zero.  Uses `self._F_internal` so every F solved
        here is cached and reused by subsequent calls.

        F-coefficients enter as :class:`QNumberPoly` (palindromic);
        the QT-product introduces shifts ``q^{<γ_1,γ_2>}`` per pair so
        the intermediate ``prod`` is no longer palindromic, hence
        operates in :class:`LaurentPoly`.  The structure-constant
        outputs ``coeff_c`` are also LaurentPoly (they satisfy
        ``m_{ab}^c(q) = m_{ba}^c(q^{-1})``, i.e. bar-skew under
        ``(a,b) ↔ (b,a)``, but are not individually palindromic).
        """
        # Convert F_a, F_b to LaurentPoly form once up front: the
        # F-basis decomp loop performs many qt_multiply calls, and the
        # per-call QNumberPoly -> LaurentPoly conversion would otherwise
        # repeat for every iteration.  QNumberPoly.to_laurent() is
        # memoised so the cost is paid once.
        F_a_lp = {d: (qn.to_laurent() if hasattr(qn, "to_laurent") else qn)
                  for d, qn in F_a.items()}
        F_b_lp = {d: (qn.to_laurent() if hasattr(qn, "to_laurent") else qn)
                  for d, qn in F_b.items()}
        prod = qt_multiply(F_a_lp, F_b_lp, self.lattice)
        zero = self.identity()
        result: dict[Vec, QTLaurentPoly] = {}
        # Cache F_low LaurentPoly conversions across loop iterations:
        # the same c_low may be encountered multiple times in
        # principle, and per-iteration to_laurent dispatch is a
        # measurable warm-path overhead on small structure constants.
        F_low_lp_cache: dict[Vec, dict[Vec, QTLaurentPoly]] = {}
        for _ in range(1000):
            prod = {g: c for g, c in prod.items() if not c.is_zero()}
            if not prod:
                break
            c_low = find_lowest(list(prod.keys()), self.cone_gens, self._cone_witness)
            coeff = prod[c_low]
            prev = result.get(c_low)
            new_c = coeff if prev is None else (prev + coeff)
            if new_c.is_zero():
                result.pop(c_low, None)
            else:
                result[c_low] = new_c
            F_low_lp = F_low_lp_cache.get(c_low)
            if F_low_lp is None:
                F_low = self._F_internal(c_low)
                F_low_lp = {d: (qn.to_laurent() if hasattr(qn, "to_laurent") else qn)
                            for d, qn in F_low.items()}
                F_low_lp_cache[c_low] = F_low_lp
            neg = QTLaurentPoly({e: -v for e, v in coeff._coeffs.items()})
            sub = qt_multiply({zero: neg}, F_low_lp, self.lattice)
            for g, nc in sub.items():
                cur = prod.get(g)
                if cur is None:
                    prod[g] = nc
                else:
                    s = cur + nc
                    if s.is_zero():
                        prod.pop(g, None)
                    else:
                        prod[g] = s
        prod = {g: c for g, c in prod.items() if not c.is_zero()}
        if prod:
            raise RuntimeError(
                "_decompose_in_F_basis: peel cap (1000 iterations) "
                f"exhausted with {len(prod)} residual charges; the "
                "decomposition is incomplete and the structure constants "
                "would be silently truncated -- raise the cap."
            )
        return {g: c for g, c in result.items() if not c.is_zero()}

    def _effective_cone_cutoff(
        self,
        F_a: dict[Vec, QNumberPoly] | None,
        F_b: dict[Vec, QNumberPoly] | None,
        K: int,
        user_cone_cutoff: int,
    ) -> int:
        """Auto-derive the cone-witness L-shell bound for K-precision.

        ``c(γ)``'s leading q-power is bounded below by ``⟨f, γ−δ⟩`` for
        each δ in the F-support (cone-witness lower bound on Nahm-sum
        exponents).  For the product ``c_a(γ)·c_b(γ)`` to contribute to
        q^{≤K} we need

            ⟨f, γ−δ_a⟩ + ⟨f, γ−δ_b⟩ ≤ K
            ⇒  ⟨f, γ⟩ ≤ (K + ⟨f, δ_a⟩ + ⟨f, δ_b⟩) / 2

        Take the maxima over each F's support (worst case) and add a
        slack of 2 for safety (Habiro expansions, (q²;q²)_∞ prefactor).
        The user-supplied ``user_cone_cutoff`` is used as a floor to
        preserve backward compatibility.
        """
        cw = self._cone_witness
        def _wval(g):
            return sum(int(fi) * int(gi) for fi, gi in zip(cw, g))
        max_delta_sum = 0
        for F in (F_a, F_b):
            if F is None:
                continue
            max_delta_sum += max(_wval(d) for d in F)
        # Slack of 2 accounts for the (q²;q²)_∞^g prefactor terms which
        # can shift contributions by O(g) at the K boundary.  Empirically
        # sufficient for ρ²-twisted trace cyclicity across pentagon,
        # hexagon, [A_1, A_4], SU(3) at K up to 30 (provided the caller
        # bumps K when post-multiplying by q-coefficients with negative
        # powers -- otherwise q^{-n}·Tr_K loses q^{K-n..K} information).
        auto = (K + max_delta_sum) // 2 + 2
        return max(user_cone_cutoff, auto)

    def _warm_fs_cache_for_schur(
        self,
        F_a: dict[Vec, QNumberPoly] | None,
        a_label: Vec | None,
        F_b: dict[Vec, QNumberPoly] | None,
        b_label: Vec | None,
        eff_cutoff: int,
        K: int,
    ) -> None:
        """Pre-populate `_FS_cache` for both F's over the Schur η-region.

        The Schur-index path needs ``[F·S|0⟩]_η`` for every η in the
        output-charge BFS (``_enumerate_output_charges``).  Without
        this, each cache miss falls through to a pointwise
        ``c_gamma_via_s`` -> ``s_gamma_habiro`` -> ``_solve_nahm_indices``
        chain (Gaussian elimination over ``Fraction``, per η).

        With this, one n-driven Nahm-tuple walk covers every ``μ = η − δ``
        the path will query, and the FS dicts for both labels are
        assembled by direct lookup against that single table.  The walk
        is bounded by an L-shell predicate driven by the strict
        cone-pointedness witness ``self._cone_witness``: ``L_max =
        max ⟨f, η − δ⟩``, accept ``γ`` iff ``⟨f, γ⟩ ≤ L_max``.  This
        is cone-direction-agnostic (works for any pointed cone, not
        just non-negative) and naturally tight.

        Spec-mode only.  Called from ``inner_product`` / Schur paths.
        """
        output_charges = _enumerate_output_charges(
            [F_a, F_b], self.cone_gens, self.lattice.rank,
            eff_cutoff,
            cone_witness=self._cone_witness,
            # The K_joint linear joint-bound prune is disabled: it was a
            # linear lower bound on a *super-linear* assembled q-order, so it
            # under-included contributing charges on mixed-sign / sheared
            # frames.  Soundness comes instead from
            # the adaptive two-cutoff-stability shell (`_schur_index_stable`),
            # which widens `eff_cutoff` until the q^K result stabilises; here
            # `eff_cutoff` is that already-chosen shell.
            K_joint=None,
        )
        if not output_charges:
            return

        for label, F in [(a_label, F_a), (b_label, F_b)]:
            # Skip the F=None (vacuum) entries: vacuum c-data is just
            # ``[S|0>]_η``, and ``c_gamma_via_s(eta, None, ...)``
            # returns exactly that with no additional cost.
            if F is None:
                continue
            # If the cache already covers every requested η for this
            # label, skip the rebuild -- repeated `inner_product`
            # calls on the same F-pair shouldn't redo the walk.
            if all((label, eta) in self._FS_cache for eta in output_charges):
                continue
            _s_tbl, fs_dict = fs_dict_for_eta_set(
                self.spec, self._kmat,
                F, output_charges, self.lattice, self._cone_witness,
            )
            for eta, fs_h in fs_dict.items():
                self._FS_cache[(label, eta)] = fs_h
            # η's that didn't appear in fs_dict are genuinely zero
            # within the L-shell of this F; do NOT poison the cache
            # with `HabiroElement.zero()` -- leave the key absent so
            # the lazy `c_gamma_via_s` fallback in `_get_c` can fill
            # them on demand (cheap; ``s_gamma_habiro`` is per-γ
            # cached at the module level).

    def _s_coefficient(self, gamma) -> HabiroElement:
        """`[S|0⟩]_γ` as a HabiroElement.  In spec mode, derived from
        the spec via Nahm-sum expansion; in recipe mode, the user-
        supplied function."""
        return self._s_coefficient_fn(tuple(gamma))

    def _s_rg_component(self, p):
        """Exact `Γ_RG`-graded component `[S_RG]_p` (RGKAlgebra contract).

        BPS grading is `deg = id` (`Γ_RG = Γ`), so the degree-`p` piece of the
        quantum torus is one-dimensional: the component is the singleton
        `{p: [S|0⟩]_p}` via the exact Nahm-sum oracle `_s_coefficient`, and
        `{}` off the cone (where `_s_coefficient` vanishes)."""
        p = tuple(p)
        c = self._s_coefficient(p)
        return {} if c.is_zero() else {p: c}

    def _F_internal(self, gamma) -> dict[Vec, QNumberPoly]:
        """Internal F lookup, returning the cached dict-of-QNumberPoly form.

        Flavour optimization: for `γ_f ∈ Γ_f = ker(B)`, `X_{γ_f}` is central
        in the quantum torus, and `F_{γ + γ_f} = X_{γ_f} · F_γ`.  Because
        `⟨γ_f, δ⟩ = 0` for all `δ ∈ Γ`, the QT product is a clean lattice
        translation: `F_{γ + γ_f} = {δ + γ_f : c_δ}` (no q-twist).

        We exploit this by reducing every requested `γ` to its section
        representative `sec(γ)` first.  If `F_{sec(γ)}` is cached, the
        flavour-shifted result comes for free; only one solve_F per
        Γ_f-orbit is ever performed.
        """
        gamma = tuple(self.lattice.check(gamma))
        if gamma in self._F_cache:
            return self._F_cache[gamma]

        # Flavoured fast path: reduce γ to its section rep, fetch (or
        # compute) F at the section rep, translate by the flavour part.
        if self._flavour_rank > 0:
            sec_c, flav_c = decompose_in_basis(
                gamma, self._sec_basis, self._ker_basis,
            )
            if any(fc != 0 for fc in flav_c):
                n = self.lattice.rank
                # Reconstruct sec_gamma and flav_gamma in standard Γ-coords.
                sec_gamma = tuple(
                    sum(sec_c[i] * self._sec_basis[i][k] for i in range(len(sec_c)))
                    for k in range(n)
                )
                flav_gamma = tuple(
                    sum(flav_c[i] * self._ker_basis[i][k] for i in range(len(flav_c)))
                    for k in range(n)
                )
                # Recurse into _F_internal on sec_gamma (which has zero
                # flavour part, so the recursion lands in the actual-solve
                # branch below).  Translate by flav_gamma.
                sec_F = self._F_internal(sec_gamma)
                shifted = {
                    tuple(d + f for d, f in zip(delta, flav_gamma)): coeff
                    for delta, coeff in sec_F.items()
                }
                self._F_cache[gamma] = shifted
                return shifted

        # Either unflavoured or γ already in the section image: solve fresh.
        if self.spec:
            # Spec mode: one n-driven Nahm-tuple walk drives both the
            # F-solver inner loop and the [F·S|0⟩]_δ byproduct that the
            # Schur-index path would otherwise recompute pointwise.
            F_dict, FS_dict, _s_tbl = solve_F_with_fs_table(
                self.lattice,
                self.cone_gens,
                gamma,
                self.spec,
                self._kmat,
                self._sigma_inverse_fn,
            )
            for eta, fs_h in FS_dict.items():
                self._FS_cache[(gamma, eta)] = fs_h
        else:
            # Recipe mode: no Nahm-tuple structure available; fall back
            # to the pointwise solver against the user-supplied callable.
            # In spec-free mode (`_sf_degree_fn` set) the F-support is bounded by
            # the degree-≤cutoff cone simplex — the flavour-safe truncation that
            # the generous box's window does not give (matter dressing costs
            # cone-degree).  Genuine recipe mode passes None and keeps the box.
            F_dict = solve_F_via_s_coefficient(
                self.lattice,
                self.cone_gens,
                gamma,
                self._s_coefficient,
                self._sigma_inverse_fn,
                max_degree=self._sf_max_degree,
                degree_fn=self._sf_degree_fn,
            )
        self._F_cache[gamma] = F_dict
        return F_dict

    # ====================================================================
    # Cache persistence
    # ====================================================================

    def _cache_signature(self) -> str:
        """SHA-256 of (pairing, node_charges) for cache identity checks."""
        key = _json.dumps(
            [[list(row) for row in self.lattice.pairing], [list(g) for g in self.node_charges]],
            sort_keys=True,
        )
        return _hashlib.sha256(key.encode()).hexdigest()[:16]

    def save_cache(self, path: str) -> None:
        """Persist ``_F_cache`` and ``_multiply_cache`` to *path* (JSON).

        Only works for unflavoured (TrivialZPlusRing) algebras.  Raises
        ``TypeError`` for flavoured theories.
        """
        if not isinstance(self._R, TrivialZPlusRing):
            raise TypeError("save_cache only supports TrivialZPlusRing algebras")

        # Element coefficients are LaurentPoly (Z[q^±]).
        def lp_to_dict(lp) -> dict:
            return {str(q): int(c) for q, c in lp._coeffs.items()}

        def element_to_obj(el: Element) -> dict:
            return {
                _json.dumps(list(label)): lp_to_dict(lp)
                for label, lp in el.terms.items()
            }

        # F-cache stores QNumberPoly coefficients (integer combinations
        # of [n]_q quantum integers).  Serialize each as {str(n): c_n};
        # the dict-of-dicts shape mirrors the dict[Vec, QNumberPoly]
        # structure.
        f_cache_obj = {
            _json.dumps(list(gamma)): {
                _json.dumps(list(delta)): {str(n): c for n, c in qn._coeffs.items()}
                for delta, qn in F_dict.items()
            }
            for gamma, F_dict in self._F_cache.items()
        }

        mul_cache_obj = {}
        for (a, b), el in self._multiply_cache.items():
            k = _json.dumps([list(a), list(b)])
            mul_cache_obj[k] = element_to_obj(el)

        data = {
            "sig": self._cache_signature(),
            # Format marker: F_cache values are QNumberPoly (n -> c_n)
            # dicts.  load_cache rejects files without this marker so
            # legacy LaurentPoly caches don't load silently into the
            # QNumberPoly-typed cache.
            "F_cache_format": "qnumberpoly_v1",
            "F_cache": f_cache_obj,
            "multiply_cache": mul_cache_obj,
        }
        with open(path, "w") as fh:
            _json.dump(data, fh)

    def load_cache(self, path: str) -> int:
        """Load ``_F_cache`` and ``_multiply_cache`` from *path*.

        Returns the number of F-cache entries loaded.  Raises
        ``ValueError`` if the file was written for a different algebra.
        """
        if not isinstance(self._R, TrivialZPlusRing):
            raise TypeError("load_cache only supports TrivialZPlusRing algebras")

        with open(path) as fh:
            data = _json.load(fh)

        if data.get("sig") != self._cache_signature():
            raise ValueError(
                f"Cache file {path!r} was written for a different algebra "
                f"(sig {data.get('sig')!r} != {self._cache_signature()!r})"
            )

        # Element is over Z[q^±] (LaurentPoly) — load
        # cached multiply results as plain LaurentPoly coefficients.

        def dict_to_lp(d: dict) -> QTLaurentPoly:
            return QTLaurentPoly({int(q): int(c) for q, c in d.items()})

        def obj_to_element(obj: dict) -> Element:
            terms = {
                tuple(map(int, _json.loads(k))): dict_to_lp(v)
                for k, v in obj.items()
            }
            return Element(terms)

        fmt = data.get("F_cache_format")
        if fmt != "qnumberpoly_v1":
            raise ValueError(
                f"Cache file {path!r} has F_cache_format={fmt!r}; this "
                f"build expects 'qnumberpoly_v1'.  Rebuild the cache."
            )
        for gamma_s, F_obj in data["F_cache"].items():
            gamma = tuple(map(int, _json.loads(gamma_s)))
            if gamma not in self._F_cache:
                self._F_cache[gamma] = {
                    tuple(map(int, _json.loads(delta_s))): QNumberPoly(
                        {int(n): int(c) for n, c in qn_obj.items()}
                    )
                    for delta_s, qn_obj in F_obj.items()
                }

        n_mul = 0
        for k_s, el_obj in data["multiply_cache"].items():
            ab = _json.loads(k_s)
            key = (tuple(ab[0]), tuple(ab[1]))
            if key not in self._multiply_cache:
                self._multiply_cache[key] = obj_to_element(el_obj)
                n_mul += 1

        return len(data["F_cache"])

    # ====================================================================
    # Display
    # ====================================================================

    def __repr__(self):
        mode = "recipe" if self._chart is None else "spec"
        return (
            f"BPSKAlgebra(rank={self.lattice.rank}, "
            f"|spec|={len(self.spec)}, "
            f"mode={mode})"
        )
