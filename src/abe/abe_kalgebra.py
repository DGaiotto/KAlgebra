"""`AbeKAlgebra` — `KAlgebra` presented on the abelianized (enriched-torus)
chart: the **completed contract** (the design record; rulings D1–D4, D7, D8).

`AbeKAlgebra(KAlgebra)` is the chart/torus-presentation tier, beside `RGKAlgebra` (flow presentation):

    KAlgebra
    ├── RGKAlgebra        presented by an RG flow over a graded auxiliary
    └── AbeKAlgebra       presented faithfully on an enriched rational
          │               quantum torus (THIS TIER)
          ├── PureUNKAlgebra        (retrofit: ruling T3)
          ├── UNNfKAlgebra          (born from this contract: T4)
          └── UNQuiverKAlgebra      (born from this contract: T4)

The presentation (repo-pinned definitions — read this, not the literature)
--------------------------------------------------------------------------
Elements live in the **f-presentation** on the product cocharacter lattice

    x  =  Σ_{m⃗}  f_{m⃗}(𝔮^{m⃗} v) · U_{m⃗},      f_{m⃗} = Σ_{k⃗} f^{(k⃗)}_{m⃗}·μ^{k⃗},

with `U_{m⃗}` the enriched atoms of `quiver_urq_torus.QuiverURQTorus`
(per-node pure normalization × the per-cell matter rung products), labels
in **lower Kapustin** convention per node, ρ the √(full measure)
G-cocycle, the trace the (product) Schur-measure residue with the matter
M-factors, bar the componentwise `𝔮 ↦ 𝔮⁻¹` (so bar-invariance ⟺ W1
palindromicity), and the canonical basis characterized executably by
`well_formed` = W1 + the single-leading-orbit q-extreme (W2) — the
Kazhdan–Lusztig read.  The literature's abelianization (BFN et al.) is a
*related* construction and is **non-load-bearing here**: no contract
semantics may be imported from it (the design notes "false friends"; ruling D2).

**No DOp on this tier** (ruling D4): the f-presentation is the single
public language; chart-operator machinery (`abelianized_torus.DOp`) is
engine-internal only.  The pioneer DOp-speaking tier is frozen at
the archived tree.

The contract (ruling D3 — the minimal triple)
---------------------------------------------
Concrete realisations supply exactly three primitives:

  * `torus_shape()` — the `TorusShape` of the enriched torus (per-node
    `RootDatum`s, matter multiplicities, links; type-A chains via
    `TorusShape.from_ranks_nf`).  This is the D6 shape surface (re-ruled
    2026-07-02 to the root-datum form).
  * `chart(label)` — the faithful f-presentation image of the canonical
    `L_label` (a `QuiverURQTorus` element).
  * `decompose(x)` — a torus element as `Element({label: C(q)})`: the
    **no-target**, level-ascending canonical read.  It must honest-fail
    (raise) off its certified scope — never guess.

Everything else is derived here: `multiply`, `rho`, `rho_inverse`,
`trace`, `inner_product`, the chart-level bar verifier, and the
`certify_canonical` acceptance.

constructive-build rule (contract-enforced)
---------------------------------
A canonical may be constructed **only** as a polynomial in basis
generators minus already-built lower canonicals; acceptance is
`well_formed` equality with the intended label.  This tier exposes **no
solve entry point**, and `decompose` is a read (it has no target to
fabricate toward).  Where a build does not reach, the realisation must
honest-fail — never fit, never solve.  (the design notes, the constructive-build rule.)

Flavour rings (ruling D8)
-------------------------
The faithful flavour symmetry of a U(N) node with `M` fundamentals is
**SU(M)** (the central U(1) is absorbed by the gauge centre); link μ's
are formal solving gradings, not flavour.  The torus levels `k⃗` are the
internal torus-refined bookkeeping; the contract packages them through
`_flavour_element` — abelian identification by default, with the
`R(SU(M))` packaging supplied by the T4 shells (which also certify the
central-direction identification against gauge det-Wilson shifts).
"""
from __future__ import annotations

import os
import sys
from abc import abstractmethod

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element, Label
from laurent_poly import LaurentPoly
from root_datum import u_n
from zplus_ring import RElement, RPowerSeries


__all__ = ["AbeKAlgebra", "TorusShape"]


def _defining_weight(datum):
    """The node's defining representation's highest weight — what an `int`
    matter entry is shorthand for.  Shared with the substrate, so the shape and
    the torus agree on what `int` means."""
    from matter_wrq_torus import defining_weight
    return defining_weight(datum)


class TorusShape:
    """The declared shape of the enriched torus — the tier's "which theory am
    I" surface (ruling D6, re-ruled 2026-07-02 to the root-datum form, option
    (b): one honest object).

    Fields: `data` — the per-node `RootDatum`s (gauge factors); `matter` —
    per-node hypermultiplet **counts**; `matter_reps` — per-node tuples of
    matter **highest weights**; `links` — bifundamental edges as node-index
    pairs.  Type-A chains construct via `from_ranks_nf` (u_n data, consecutive
    links — the legacy `(ranks, Nf)` reading); group-general realisations via
    `from_root_data`.  Equality compares datum names + `matter_reps` + links
    (factory data are name-canonical).

    **Matter is a representation, not a count** (user ruling, 2026-07-28: *"fix
    the torusshape so `GMatterAbeKAlgebra` can name itself"*).  The field was
    originally one `int` per node — per-node *fundamental* multiplicities, the
    type-A reading "`N_f` fundamentals".  For a general `(G, N)` the matter is a
    representation `N = ⊕N_i`, so `Sp(4)`+**4**, `Spin(5)`+**5** and
    `Spin(5)`+**4**(spinor) all declared `1` and the tier could not say which
    theory it was.  A node's matter may now be given either as

      * an `int` — shorthand for that many copies of the node's **defining**
        representation (the type-A reading, preserved exactly), or
      * a sequence of dominant weights — one per hypermultiplet slot.

    `.matter` still returns per-node **counts**, so every existing consumer is
    unaffected; `.matter_reps` carries the weights and is what `_key()` compares,
    which is what makes the shape name its theory."""

    __slots__ = ("data", "matter", "matter_reps", "links")

    def __init__(self, data, matter, links=()):
        self.data = tuple(data)
        self.links = tuple(tuple(l) for l in links)
        matter = tuple(matter)
        if len(matter) != len(self.data):
            raise ValueError("TorusShape: one matter entry per node")
        counts, reps = [], []
        for datum, entry in zip(self.data, matter):
            if isinstance(entry, int):
                # `matter=0` (a pure node) must not need a defining weight at
                # all — it has no matter to name.
                node = () if entry == 0 else (_defining_weight(datum),) * entry
            else:
                node = tuple(tuple(x) for x in entry)
                for w in node:
                    if len(w) != datum.dim:
                        raise ValueError(
                            f"TorusShape: matter weight {w} has length "
                            f"{len(w)}, expected {datum.dim} for {datum.name}")
            counts.append(len(node))
            reps.append(node)
        self.matter = tuple(counts)
        self.matter_reps = tuple(reps)

    @classmethod
    def from_ranks_nf(cls, ranks, nf, links=None):
        """The type-A chain shape — per-node `u_n` data, consecutive links."""
        ranks = tuple(int(r) for r in ranks)
        if links is None:
            links = tuple((i, i + 1) for i in range(len(ranks) - 1))
        return cls(tuple(u_n(r) for r in ranks), tuple(nf), links)

    @classmethod
    def from_root_data(cls, data, matter=None, links=()):
        data = tuple(data)
        if matter is None:
            matter = (0,) * len(data)
        return cls(data, matter, links)

    @property
    def ranks(self):
        """Per-node unitary ranks — honest-fails on non-U(N) nodes."""
        out = []
        for d in self.data:
            if not d.name.startswith("U("):
                raise NotImplementedError(
                    f"TorusShape.ranks: node {d.name} is not a U(N) node")
            out.append(d.dim)
        return tuple(out)

    @property
    def nf(self):
        return self.matter

    def _key(self):
        return (tuple(d.name for d in self.data), self.matter_reps, self.links)

    def __eq__(self, other):
        if not isinstance(other, TorusShape):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        nodes = " × ".join(d.name for d in self.data)
        plain = all(not n or reps == (_defining_weight(d),) * n
                    for d, reps, n in zip(self.data, self.matter_reps,
                                          self.matter))
        mat = self.matter if plain else self.matter_reps
        return f"TorusShape({nodes}; matter={mat}; links={self.links})"


def _datum_fingerprint(d) -> dict:
    """A structural + BEHAVIOURAL digest of a `RootDatum`, for cache provenance.

    Structure alone is not enough.  A datum's atom phase and ρ-sign are
    *callables* (`atom_phase=` / `rho_sign=` constructor overrides), and ruling
    D31 turns exactly on their values at odd `⟨Σ⁺, m⟩` — two data with identical
    roots and a different phase convention produce different charts, both
    well-formed.  So the phase is PROBED: evaluated on the simple coroots and
    their pairwise sums, which is a canonical finite set determined by the datum
    itself.  A probe that raises is recorded as its exception type, so a datum
    that refuses on some charge still fingerprints deterministically."""
    def probe(fn, k):
        try:
            return int(fn(k))
        except Exception as ex:                       # phase overrides may refuse
            return f"!{type(ex).__name__}"
    cor = [tuple(c) for c in d.simple_coroots]
    ks = list(cor) + [tuple(a + b for a, b in zip(x, y))
                      for i, x in enumerate(cor) for y in cor[i:]]
    return {
        "name": d.name,
        "dim": int(d.dim),
        "simple_roots": [list(a) for a in d.simple_roots],
        "simple_coroots": [list(a) for a in d.simple_coroots],
        "positive_roots": sorted(list(a) for a in d.positive_roots()),
        "shift_pairing": [[int(d.shift_pairing(c, a)) for a in d.simple_roots]
                          for c in d.simple_coroots],
        "atom_phase_doubled": [probe(d.atom_phase_doubled, k) for k in ks],
        "rho_sign_exp": [probe(d.rho_sign_exp, k) for k in ks],
    }


class AbeTorus:
    """The enriched **WRQTorus** substrate of an `AbeKAlgebra`, SELECTED by its
    `TorusShape` (D9/D10: the group-general WRQTorus is the *universal*
    substrate — the pure `WRQTorus` / matter `MatterWRQTorus` / quiver
    `QuiverWRQTorus` element type falls out of the shape, so there is no
    per-tier substrate/algebra class).  A realisation picks its substrate purely
    by what `torus_shape()` returns; `AbeKAlgebra.torus()` builds this from it."""

    __slots__ = ("shape", "n", "nf", "kind", "datum")

    def __init__(self, shape: "TorusShape"):
        self.shape = shape
        self.n = len(shape.data)
        self.nf = shape.matter
        if self.n == 1 and shape.matter[0] == 0:
            self.kind = "pure"
            self.datum = shape.data[0]
        elif self.n == 1:
            self.kind = "matter"
            self.datum = shape.data[0]
        else:
            from quiver_wrq_torus import quiver_datum
            self.kind = "quiver"
            self.datum = quiver_datum(shape.ranks)

    @property
    def ranks(self):
        return self.shape.ranks

    def element(self, residuals):
        """Build a substrate element (the WRQTorus type selected by the shape)
        from `{atom: …}` residuals."""
        if self.kind == "pure":
            from wrq_torus import WRQTorus
            return WRQTorus(self.datum, residuals)
        if self.kind == "matter":
            from matter_wrq_torus import MatterWRQTorus
            # Pass the matter **representations**, not the count: a `(G, N)`
            # node's rungs are indexed by the weights of `N_i`, and at U(N)
            # fundamentals the two declarations coincide.
            return MatterWRQTorus(self.datum, self.shape.matter_reps[0],
                                  residuals)
        from quiver_wrq_torus import QuiverWRQTorus
        return QuiverWRQTorus(self.shape.ranks, self.nf, residuals,
                              datum=self.datum)

    def __repr__(self):
        return f"AbeTorus({self.kind}, {self.datum.name}, matter={self.nf})"


class AbeKAlgebra(KAlgebra):
    """`KAlgebra` presented faithfully on the enriched rational quantum
    torus — supply `torus_shape` / `chart` / `decompose`, inherit the
    algebra.  See the module docstring for the pinned conventions.

    The substrate is the **group-general WRQTorus**, selected by `torus_shape()`
    and exposed as the `torus()` property (D9/D10, user 2026-07-04: use WRQ
    universally, no extra WRQ-named classes — the substrate is a property of
    `AbeKAlgebra` picked by the subclass's shape)."""

    def torus(self) -> "AbeTorus":
        """The WRQTorus substrate selected by `torus_shape()` (cached)."""
        t = getattr(self, "_abe_torus", None)
        if t is None:
            t = AbeTorus(self.torus_shape())
            self._abe_torus = t
        return t

    # ------------------------------------------------------------------
    # The contract: three abstract primitives (D3).
    # ------------------------------------------------------------------

    @abstractmethod
    def torus_shape(self) -> "TorusShape":
        """The `TorusShape` of the enriched torus — per-node `RootDatum`s,
        matter multiplicities, and links.  The D6 shape surface (re-ruled
        2026-07-02 to the root-datum form); everything geometric (lattice
        rank, Weyl blocks, dressing cells, measure) derives from it.
        Type-A chains: `TorusShape.from_ranks_nf(ranks, Nf)`."""

    @abstractmethod
    def chart(self, label: Label):
        """`L_label` as a `QuiverURQTorus` element (the f-presentation
        image; faithful for all derived operations)."""

    @abstractmethod
    def decompose(self, x) -> Element:
        """A torus element in the canonical basis: `Element({label: C(q)})`.
        The no-target, level-ascending read; **honest-fail off-scope**
        (raise `NotImplementedError`) — never guess, never solve."""

    # ------------------------------------------------------------------
    # Derived KAlgebra primitives (the whole algebra).
    # ------------------------------------------------------------------

    def multiply(self, a: Label, b: Label) -> Element:
        """`L_a · L_b = Σ_c C^c_{ab}(q)·L_c` — chart-multiply (the exact
        torus cocycle product) then decompose."""
        return self.decompose(self.chart(a) * self.chart(b))

    def _single_label(self, x, what: str) -> Label:
        d = self.decompose(x)
        terms = {lab: c for lab, c in d.terms.items()
                 if not (hasattr(c, "is_zero") and c.is_zero())}
        if len(terms) != 1:
            raise NotImplementedError(
                f"{type(self).__name__}.{what}: image is not a single "
                f"canonical ({len(terms)} terms)")
        (lab, c), = terms.items()
        cl = c if isinstance(c, LaurentPoly) else LaurentPoly({0: int(c)})
        if cl != LaurentPoly({0: 1}):
            raise NotImplementedError(
                f"{type(self).__name__}.{what}: image coefficient {c} ≠ 1 "
                f"(the basis map must be coefficient-free)")
        return lab

    def rho(self, a: Label) -> Label:
        """ρ on labels.  The tier's PRIMARY ρ is the explicit label-level
        closed form `wrq_torus.rho_label` (+ `rho_level_star` on flavour;
        promoted by user ruling 2026-08-23), wired per realisation since each
        realisation owns its label frame — the U(N) keystone's
        `pure_un_kalgebra.rho_label` Witten-shift maps are its type-A special
        case.  This default — the torus √measure conjugation read back through
        `decompose` (a sign-free basis permutation, coefficient 1) — is the
        DEMOTED route: it remains only as (i) the fallback for a realisation
        that has not wired its frame yet (today: the quiver chains) and
        (ii) the verification route, `verify_rho_via_twist`."""
        return self._single_label(self.chart(a).rho(), "rho")

    def rho_inverse(self, a: Label) -> Label:
        return self._single_label(self.chart(a).rho_inverse(), "rho_inverse")

    def verify_rho_via_twist(self, a: Label) -> bool:
        """The demoted chart→twist→`decompose` route as a TEST (user ruling
        2026-08-23: "chart/twist/decompose then becomes a test"): certify the
        realisation's label-level ρ and ρ⁻¹ against the torus √measure
        conjugation read back through `decompose`.  Emergent for every
        realisation that overrides `rho` with the closed form; tautological
        only on a realisation still using the tier fallback."""
        ch = self.chart(a)
        if self.rho(a) != self._single_label(ch.rho(), "verify_rho_via_twist"):
            return False
        return self.rho_inverse(a) == self._single_label(
            ch.rho_inverse(), "verify_rho_via_twist")

    def r_label_decompose(self, label):
        """The flavour-lift coordinate, **defaulted for the
        abelianized contract**: this tier packages flavour into the
        *coefficient ring* (`_flavour_element` — the torus levels become
        `R(SU(M))` characters, "free over `R(G_f)`"), so the canonical basis is
        flavour-neutral and the lift is trivial — section = `label`, single
        irrep = χ₀ (`coefficient_ring().one_basis()`).

        Implemented **directly** (independent of `_label_section_decompose`,
        which now *derives* from this by the KAlgebra forward bridge) so that
        method is obsoletable, and `forget()` / ring-hom flavour reduction
        (`base_change(restriction)`) + promotion (`base_change(unit_hom)`) read
        this coordinate cleanly.  A realisation that instead carries the flavour
        irrep in a **label slot** (e.g. `UNNfKAlgebra`'s `((m, λ), w)`) overrides
        this with its gauge/weight split."""
        return label, self.coefficient_ring().one_basis()

    def r_label_compose(self, section, r_basis_label):
        """Inverse of the trivial default `r_label_decompose`: the label is the
        section (the flavour is central in the coefficient ring, so there is no
        `embed_R`/`multiply` round-trip).  Overridden alongside
        `r_label_decompose` by flavour-in-a-label-slot realisations."""
        return section

    def _flavour_element(self, ring, lev: tuple, c):
        """Package one torus μ-level coefficient as a coefficient-ring
        element.  Default: the abelian identification (level vector =
        ring basis key); rank-0 shapes use the bare integer convention.
        The D8 SU(M) packaging overrides this in the flavoured shells."""
        if not lev:
            return c
        return RElement(ring, {tuple(lev): c})

    @staticmethod
    def _trace_levels(tr):
        """Normalize a chart trace to the per-μ-level form `{lev: LaurentPoly}`.
        Matter / quiver charts return that dict directly; a **pure-gauge**
        `WRQTorus` trace is a bare `LaurentPoly` (no matter levels) — treat it as
        the single trivial level `()`."""
        return tr if isinstance(tr, dict) else {(): tr}

    @staticmethod
    def _chart_trace(chart, K: int):
        """`chart.trace(K=K)` with the **Nahm window `W` tied to `K`**.

        *This is a correctness fix, not a tuning knob* (user, 2026-07-28: the trace
        "cannot fail … probably a truncation effect", and "it is possible that the
        AbeKAlgebra trace tool escaped previous efforts to make trace truncation
        reliable" — it had).

        The matter-carrying charts expand the flavour factor
        `∏_{i,j} E(μ_i v_j)E(μ_i^{-1} v_j^{-1})` only to Nahm level `W`, defaulting to
        **4**, and `AbeKAlgebra.trace` / `inner_product` never forwarded a `W`.  So
        every μ-refined trace was silently reliable only to `𝖖⁴` however large a `K`
        the caller asked for, and quietly wrong above it — exactly the early-truncation
        failure mode `the design notes` warns is "the single most common and most damaging
        error here".

        **Measured** (`SU(2)+Adj` vs `SO(3)+Adj`, the Langlands pair that exposed it):
        the first wrong `𝖖`-order is exactly `W + 1` — `W=4 → 𝖖⁵`, `W=6 → 𝖖⁷`,
        `W=8 → 𝖖⁹`, `W=10 → 𝖖¹¹` — and at `W = K` the two sides agree to all of `K`.

        `W = K + 2` rather than the minimal `W = K`, to match the **existing
        precedent**: `UNNfKAlgebra.trace` already passes `W = K + 2`, i.e. an earlier
        session found and fixed this for the type-A class, and the later group-general
        classes simply did not inherit the lesson.  Consistency with that convention is
        worth more than saving one Nahm level.

        Passed only where the signature accepts it: the pure-gauge `WRQTorus.trace`
        has no matter factor and therefore no `W`."""
        try:
            import inspect
            if "W" in inspect.signature(chart.trace).parameters:
                return chart.trace(K=K, W=K + 2)
        except (TypeError, ValueError):        # pragma: no cover - builtins etc.
            pass
        return chart.trace(K=K)

    def trace(self, a: Label, K: int = 20) -> RPowerSeries:
        """`Tr(L_a)` — the torus trace (product Schur-measure residue with
        the matter M-factors), packaged over the coefficient ring.

        The matter Nahm window is tied to `K` (`_chart_trace`); before 2026-07-28 it
        was pinned at 4, so results above `𝖖⁴` were silently wrong."""
        R = self.coefficient_ring()
        acc: dict = {}
        for lev, lp in self._trace_levels(self._chart_trace(self.chart(a), K)).items():
            for e, c in lp._coeffs.items():
                if not (0 <= e <= K) or not c:
                    continue
                cur = acc.get(e)
                fe = self._flavour_element(R, tuple(lev), c)
                acc[e] = fe if cur is None else (cur + fe)
        return RPowerSeries(R, acc, K)

    def inner_product(self, a: Label, b: Label, K: int = 20) -> RPowerSeries:
        """`I_{a,b} = Tr(ρ(L_a)·L_b)` — evaluated entirely chart-side
        (torus ρ, torus product, torus trace): exact, no decompose.

        Same Nahm-window fix as `trace` (`_chart_trace`): the matter factor is now
        expanded to level `K` rather than the default 4, so `I_{a,b}` is trustworthy
        to the requested order instead of only to `𝖖⁴`."""
        R = self.coefficient_ring()
        prod = self.chart(a).rho() * self.chart(b)
        acc: dict = {}
        for lev, lp in self._trace_levels(self._chart_trace(prod, K)).items():
            for e, c in lp._coeffs.items():
                if not (0 <= e <= K) or not c:
                    continue
                cur = acc.get(e)
                fe = self._flavour_element(R, tuple(lev), c)
                acc[e] = fe if cur is None else (cur + fe)
        return RPowerSeries(R, acc, K)

    # ------------------------------------------------------------------
    # Chart-level certificates / verifiers.
    # ------------------------------------------------------------------

    def verify_chart_bar(self, a: Label) -> bool:
        """Bar-invariance of `L_a` in chart form — W1: every residual
        component q-palindromic (`bar(chart) == chart`)."""
        return bool(self.chart(a).well_formed_w1())

    def certify_canonical(self, a: Label):
        """The executable KL acceptance of `L_a`'s chart: `well_formed()`
        = W1 + single-leading-orbit q-extreme (W2).  Returns the torus's
        `(joint lower-Kapustin label, base level)` read — the realisation's
        tests assert it against their own label map — or ``False``."""
        return self.chart(a).well_formed()

    # ------------------------------------------------------------------
    # Chart memoization and its persistence.
    # ------------------------------------------------------------------
    # The `L_{m,e}` are expensive — the one production route is a guarded
    # solve, and at the hard end a single label costs seconds (SU(3) `m=(2,2)`
    # 5.4 s, G₂ `m=(2,3)` 6.3 s).  Every realisation already memoizes its
    # charts in memory; what this section adds is (a) a UNIFORM way to reach
    # that memo and (b) persistence, so the cost is paid once per label ever
    # rather than once per process.
    #
    # The discipline point.  A cache file is untrusted input: it is a plain
    # JSON file that anyone (including a stale copy of this repo) may have
    # written, and a wrong `L_{m,e}` that passes silently is this tier's worst
    # failure mode — the whole reason the constructive-build rule exists.  So
    # `load_cache` does BOTH: it refuses a file whose header does not fingerprint
    # this exact presentation, and it re-runs the axioms on every element it
    # admits.  That re-verification is affordable precisely because verifying is
    # not solving: measured at ~5% of a rebuild (SU(3) `m=(2,2)` 0.28 s against
    # 5.4 s; G₂ `m=(2,3)` 0.41 s against 6.3 s), so trusting the file buys ~5%
    # and risks everything.

    CACHE_FORMAT = 1

    # ----- memoization: the first universal optimization -----------------
    # Stage 2 closes with two optimizations on the GENERAL tier and everything
    # else devolved to specializations (user, 2026-08-25): *"the θ-twist applied
    # to general `(m,e)` is a useful optimization … and memoization of course"*,
    # with *"other obsolete optimizations … left maybe to specific
    # specializations of `AbeKAlgebra` which make assumptions on `G` and `N`"*.
    # The two that stay here are exactly the two that assume nothing about `G` or
    # `N`, and they COMPOSE: the memo holds what has been built, and the θ-twist
    # turns each entry into its whole `m̄`-line.  Persistence carries both across
    # processes, so the composition accumulates instead of restarting.
    #
    # Memoization is NOT in the `optimizations` register, and that is not an
    # oversight.  Every entry there is a *route* — a different way to compute the
    # same element — and switching one off leaves a working algebra that is
    # merely slower.  The memo is structural: `_build`'s cycle detection, the
    # general θ-twist's search for a `T`-related source, and `decompose`'s
    # repeated `chart` calls all read it, so an off-switch would change behaviour
    # and not only cost.  What a caller legitimately wants is to RECLAIM it,
    # which is what `clear_cache` is for.

    def clear_cache(self) -> int:
        """Drop the memoized charts (and their route provenance); returns how
        many were dropped.

        For memory pressure on a long label sweep, and for cold-build timing in
        the probes in the source repository.  Not an off-switch — the memo refills as soon as
        `chart` is called again — because the tier's build structure reads it
        (see the note above)."""
        cache = self.chart_cache()
        n = 0
        if cache is not None:
            n = len(cache)
            cache.clear()
        routes = self.route_cache()
        if routes is not None:
            routes.clear()
        for sub in self.nested_cache_algebras().values():
            n += sub.clear_cache()
        return n

    def chart_cache(self) -> dict | None:
        """The realisation's `{label: substrate element}` chart memo, or `None`
        if it does not memoize.

        The hook `save_cache` / `load_cache` are written against.  A realisation
        returns its LIVE dict (not a copy), so `load_cache` populates the same
        memo `chart` reads.  Default `None` — a realisation that has no memo is
        not broken, it simply has nothing to persist, and the two methods say so
        rather than writing an empty file."""
        return None

    def cache_identity(self) -> str:
        """Tag the STRUCTURE-CONSTANT cache with the same presentation digest the
        chart cache uses, so the two halves of persistence are guarded alike.

        `KAlgebra.cache_identity` guards `save_structure_constants` /
        `load_structure_constants`, and its docstring warns that parametric
        subclasses MUST refine it "otherwise two distinct instances of one class
        would share a cache file".  On this tier every subclass is parametric and
        six never did, so the products half was measurably unguarded: at SO(3),
        SU(3), G₂ and BOTH su(2) global forms `PureGAbeKAlgebra` returned the one
        string `'PureGAbeKAlgebra'`, and `load_structure_constants` accepted
        SO(3)'s table into G₂ and returned `True`.  Meanwhile the CHART half was
        strongly guarded by `cache_fingerprint`, which separates all five — an
        asymmetry on one tier, with the weaker guard on the half that has no
        `verify=` re-check to fall back on.

        Derived from `cache_fingerprint()` rather than reimplemented, so there is
        ONE source of truth: anything that makes two presentations distinct
        enough to need separate chart files makes them distinct enough to need
        separate structure-constant files, and a realisation that extends the
        fingerprint gets the sharper identity for free.  Serialized with sorted
        keys so the tag is stable across processes."""
        import json
        return (f"{type(self).__name__}("
                + json.dumps(self.cache_fingerprint(), sort_keys=True,
                             separators=(",", ":")) + ")")

    def route_cache(self) -> dict | None:
        """The realisation's `{label: route name}` provenance memo, or `None`.

        Persisted alongside the charts so a reloaded instance can still answer
        `route(label)` — which is the single most informative number about a
        build (see `gn_abe_kalgebra`'s benchmark docstring) and
        would otherwise be silently lost across a save/load."""
        return None

    def nested_cache_algebras(self) -> dict:
        """`{section name: AbeKAlgebra}` whose chart memos are persisted INSIDE
        this one's file.

        A realisation that delegates part of its build to another `AbeKAlgebra`
        declares it here, so one file holds everything one instance needs.  The
        `(G, N)` tier does: `decompose` reads its lowest μ-level slice with the
        inner pure-`G` engine (ruling D3, `Q[0⃗] = pure`), so a matter cache
        without the pure charts still pays for them on the first read — which is
        exactly the cost the file exists to avoid.  Each section carries its own
        fingerprint and is verified on its own terms."""
        return {}

    def cache_fingerprint(self) -> dict:
        """A structural digest of the presentation this instance realises.

        `load_cache` refuses a file whose fingerprint differs, because a chart is
        a vector of residuals in RAW coordinates of a particular root datum, line
        lattice and phase convention — nothing in the numbers themselves says
        which.  Loading SU(3)'s `m=(1,1)` into `Sp(2)` would not raise anywhere;
        it would just be wrong.

        What goes in, and why each is load-bearing:

        * the **torus shape** — the substrate type and its matter/link data;
        * the **root datum's structure** — name, rank, simple roots and coroots,
          the full positive-root set, and the shift pairing on the simple roots
          (so a non-standard `pairing=` cannot pass as the dot product);
        * the **phase convention**, probed rather than described: `atom_phase_doubled`
          and `rho_sign_exp` evaluated on the simple coroots and their pairwise
          sums.  A datum's phase is a CALLABLE and cannot be compared any other
          way, and the phase is exactly what ruling D31's odd-`⟨Σ⁺,m⟩` correction
          moves — so two data agreeing on structure and disagreeing here produce
          different, both-well-formed, charts;
        * the **line lattice** — its name and its centre-class pairs, since the
          4d gauge group data decides which `(m, e)` are labels at all.

        A realisation with more presentation data than this should extend it
        (call `super().cache_fingerprint()` and add), not replace it."""
        shape = self.torus_shape()
        fp = {"class": type(self).__name__,
              "shape": repr(shape),
              "kind": self.torus().kind,
              "data": [_datum_fingerprint(d) for d in shape.data]}
        lines = getattr(self, "lines", None)
        if lines is not None:
            fp["lines"] = {
                "name": getattr(lines, "name", None),
                "pairs": sorted(repr(x) for x in lines.class_pairs()),
            }
        return fp

    def _verify_loaded_chart(self, label: Label, x) -> dict:
        """`{condition: (ok, detail)}` — the axioms run against a chart that was
        LOADED rather than built.  Realisation hook.

        The tier default is what is universally available on the substrate: W1
        (every residual `𝖖`-palindromic) and W2 (`well_formed()` returns a label
        rather than `False`).  It deliberately does NOT try to compare that label
        to `label` — the label frame belongs to the realisation, not to the tier
        (`certify_canonical` says the same), so a realisation that knows its
        frame must override and check the identity.  `PureGAbeKAlgebra` does,
        through `star_bubbling.verify_axioms`, which is the full five-condition
        battery including the seed identity and (★)."""
        out = {"bar (W1)": (bool(x.well_formed_w1()), "")}
        wf = x.well_formed()
        out["seed (W2)"] = (wf is not False, f"well_formed() = {wf}")
        return out

    # ----- the label codec ---------------------------------------------
    # Labels are opaque to the contract, so the tier serializes only what it can
    # do EXACTLY: nested tuples of ints (which is what every shipped realisation
    # uses — `(m, e)` for pure `G`, `((m, e), w)` with matter).  Anything else
    # honest-fails on save, naming the label, rather than being coerced into
    # something that will not round-trip.

    @staticmethod
    def _label_to_json(label):
        def enc(v):
            if isinstance(v, bool):
                raise TypeError(f"bool in label: {label!r}")
            if isinstance(v, int):
                return v
            if isinstance(v, str):
                return {"s": v}
            if isinstance(v, (tuple, list)):
                return [enc(u) for u in v]
            raise TypeError(
                f"AbeKAlgebra.save_cache: label {label!r} contains a "
                f"{type(v).__name__}; the tier serializes nested tuples of ints "
                f"and strings only.  Override `_label_to_json`/`_label_from_json` "
                f"in the realisation rather than widening this.")
        return enc(label)

    @staticmethod
    def _label_from_json(obj):
        def dec(v):
            if isinstance(v, dict):
                return v["s"]
            if isinstance(v, list):
                return tuple(dec(u) for u in v)
            return int(v)
        return dec(obj)

    # ----- element codec -------------------------------------------------

    def _residuals_to_json(self, x):
        """`{atom: TorusRational}` (pure) or `{atom: {μ-level: TorusRational}}`
        (matter / quiver) → JSON.  The two shapes are distinguished on the value,
        which is what `AbeTorus.element` reads back."""
        out = []
        for m, val in sorted(x.residuals().items()):
            if isinstance(val, dict):
                out.append([list(m), [[list(k), fr.to_json()]
                                      for k, fr in sorted(val.items())]])
            else:
                out.append([list(m), val.to_json()])
        return out

    def _residuals_from_json(self, obj):
        from weyl_torus_ring import TorusRational
        dat = self.torus().datum
        res = {}
        for m, val in obj:
            # A pure residual serializes as the `TorusRational` dict
            # (`{"num": …, "den": …}`); a matter / quiver one as the LIST of its
            # per-μ-level entries.  The two shapes are what `AbeTorus.element`
            # reads back, so dispatching on the container is exact.
            if isinstance(val, dict):
                res[tuple(m)] = TorusRational.from_json(dat, val)
            else:
                res[tuple(m)] = {tuple(k): TorusRational.from_json(dat, fo)
                                 for k, fo in val}
        return res

    # ----- save / load ---------------------------------------------------

    def save_cache(self, path: str) -> int:
        """Persist the built `L_{m,e}` charts to JSON at `path`; returns how many
        were written.

        Mirrors `RGKAlgebra.save_cache` / `load_cache` in name and in shape — the
        flow tier's caches are persisted the same way, and this is the chart
        tier's counterpart.  What is written is the chart memo plus the route
        provenance plus `cache_fingerprint()`; nothing `𝖖`-cutoff-dependent is
        written (traces and inner products are cutoff-dependent and stay session
        memos, exactly as `RGKAlgebra` leaves `RG·S_RG` unwritten).

        The file is EXACT — residuals are integer numerator coefficients over an
        explicit denominator multiset, so a round trip is an identity, not a
        re-derivation."""
        import json
        if self.chart_cache() is None:
            raise NotImplementedError(
                f"{type(self).__name__}.save_cache: this realisation does not "
                f"expose a chart memo (`chart_cache()` returned None), so there "
                f"is nothing to persist.  Implement `chart_cache()` to opt in.")
        payload, n = self._cache_payload()
        payload["format"] = self.CACHE_FORMAT
        with open(path, "w") as fh:
            json.dump(payload, fh)
        return n

    def _cache_payload(self):
        """`(payload, count)` — this instance's cache section, without the file
        wrapper.  Shared by `save_cache` and by a parent's nested write."""
        cache = self.chart_cache()
        if cache is None:
            return ({"fingerprint": self.cache_fingerprint(), "charts": [],
                     "nested": {}}, 0)
        routes = self.route_cache() or {}
        charts = [{"label": self._label_to_json(label),
                   "route": routes.get(label),
                   "residuals": self._residuals_to_json(x)}
                  for label, x in cache.items()]
        obj = {"fingerprint": self.cache_fingerprint(), "charts": charts,
               "nested": {}}
        n = len(charts)
        for name, sub in self.nested_cache_algebras().items():
            sub_obj, sub_n = sub._cache_payload()
            obj["nested"][name] = sub_obj
            n += sub_n
        return obj, n

    def load_cache(self, path: str, verify: str = "axioms",
                   overwrite: bool = False) -> int:
        """Repopulate the chart memo from a file written by `save_cache`;
        returns how many charts were admitted.

        `verify` — what is run on each loaded chart before it is admitted:

        | value | what runs | when to use |
        |---|---|---|
        | `"axioms"` (default) | `_verify_loaded_chart` — on `PureGAbeKAlgebra` the full five-condition battery (W1, the W2 seed identity, support, `O(𝖖)`, (★)) | always, unless you have a measured reason not to |
        | `"none"` | nothing | only for a file this same process just wrote |

        The default is the strict one on purpose.  Verifying is ~5% of the cost
        of solving (measured above), so `"none"` buys almost nothing and gives up
        the one guard that distinguishes a canonical from a plausible impostor —
        and (★) is precisely the condition that caught the
        `pure_un_joint_fiber_defect` failure mode 23/23 where the `𝖖⁰` self-norm
        did not.

        A chart that fails verification raises rather than being skipped: a cache
        file that disagrees with the axioms is evidence of a real problem (a
        stale format, a changed convention, a corrupted write), and silently
        dropping the bad entries would hide it while the good ones still loaded.

        `overwrite=False` (the default) leaves an already-built label alone —
        the in-memory element was built by this process and is the one
        `decompose` may already have consumed."""
        import json
        if self.chart_cache() is None:
            raise NotImplementedError(
                f"{type(self).__name__}.load_cache: this realisation does not "
                f"expose a chart memo (`chart_cache()` returned None).")
        if verify not in ("axioms", "none"):
            raise ValueError(
                f"load_cache: verify={verify!r} — expected 'axioms' or 'none'")
        with open(path) as fh:
            obj = json.load(fh)
        fmt = obj.get("format")
        if fmt != self.CACHE_FORMAT:
            raise ValueError(
                f"{type(self).__name__}.load_cache: {path} is format {fmt!r}, "
                f"this build reads format {self.CACHE_FORMAT}.  Rebuild the "
                f"cache rather than reading it — a format bump means the "
                f"residual encoding changed, so the numbers would be misread, "
                f"not merely rejected.")
        return self._load_payload(obj, path, verify=verify,
                                  overwrite=overwrite)

    def _load_payload(self, obj, path, verify="axioms", overwrite=False,
                      section=None) -> int:
        """Admit one cache section (this instance's, or a nested one).  Same
        fingerprint refusal and same per-chart verification as `load_cache`;
        `section` only sharpens the error message."""
        where = f"{path}" + (f" [nested section {section!r}]" if section else "")
        cache = self.chart_cache()
        if cache is None:
            raise NotImplementedError(
                f"{type(self).__name__}: {where} carries charts but this "
                f"realisation exposes no chart memo (`chart_cache()` → None).")
        want, got = self.cache_fingerprint(), obj.get("fingerprint")
        if got != want:
            diff = sorted(k for k in set(want) | set(got or {})
                          if (got or {}).get(k) != want.get(k))
            raise ValueError(
                f"{type(self).__name__}: {where} was written for a DIFFERENT "
                f"presentation — {diff} differ.")
        routes = self.route_cache()
        n = 0
        for rec in obj.get("charts", []):
            label = self._label_from_json(rec["label"])
            if label in cache and not overwrite:
                continue
            x = self.torus().element(self._residuals_from_json(rec["residuals"]))
            if verify == "axioms":
                report = self._verify_loaded_chart(label, x)
                bad = {k: d for k, (ok, d) in report.items() if not ok}
                if bad:
                    raise ValueError(
                        f"{type(self).__name__}: the chart loaded for L_{label} "
                        f"from {where} FAILS the axioms it claims to satisfy: "
                        f"{bad}.")
            cache[label] = x
            if routes is not None and rec.get("route") is not None:
                routes[label] = rec["route"]
            n += 1
        for name, sub in self.nested_cache_algebras().items():
            sub_obj = (obj.get("nested") or {}).get(name)
            if sub_obj is not None:
                n += sub._load_payload(sub_obj, path, verify=verify,
                                       overwrite=overwrite, section=name)
        return n
