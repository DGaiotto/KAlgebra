"""
a1dodd_kalg.py
==============

`A1DoddKAlg(k)` — the **odd** D-type Argyres–Douglas family
`A_𝖖([A_1, D_{2k+3}])` with **SU(2) flavour symmetry**, as a self-contained
K-algebra over `R(SU(2))`, in the **RG-flow frame** of the fast, BPS-free
`A1DoddRGKAlgebra(k)` (whose auxiliary is the stand-alone
`U1A1AoddKAlg(k).add_flavour(SU2ZPlusRing())`, `S_RG = E_𝖖(μL)·E_𝖖(μ⁻¹L)` an
SU(2) doublet).

The cracked structure (RG-flow frame == u1a1aodd ⊗ SU(2)-CG)
-----------------------------------------------------------
Cracking the cones/Plückers with the RG engine (BPS would choke) gives a
strikingly clean answer: in the **clean E-frame** of each chord ray — the
E-power at which its monomial `RG` is a single term, the un-dressed frame; a
label `(factors, e_E)` is clean iff `e_E ≥ Σ exp·eE*(a,i)` — the χ-stripped
cone skeleton (q-commute graph, cones, cocycle = the `A_{2k+2}` chain pairing,
ρ) is **identical** to `U1A1AoddKAlg(k)`'s, and

    multiply( (m_a, κ_a), (m_b, κ_b) )
        =  U1A1AoddKAlg(k).multiply(m_a, m_b)  ⊗  CG(χ_{κ_a} · χ_{κ_b})

— the u1a1aodd cone product on the monomial part, tensored with the **SU(2)
Clebsch–Gordan** of the flavour characters on the κ-label.  Verified against
the engine (k=1): chords 729/729, cone monomials 900/900, daughter-closure
1908/1908 (the clean canonical basis is closed under the product).

Why this is faithful (not a trivial spectator flavour)
------------------------------------------------------
The RG flow is genuinely non-trivial (`engine.multiply ≠ aux.multiply` on the
RG-dressed, sub-clean labels), so this is *not* a bare `add_flavour`: the
canonical basis is built from the **chord rays** (the un-dressed, RG-monomial
generators) exactly as the ABSOLUTE RULE demands, and the SU(2) enters *only*
through products of those chord rays and their ρ-images (the κ-CG).  The hard
cross-cone Plückers ("products between cones and ρ-images of cones") are
inherited **already-cracked** from `U1A1AoddKAlg`; A1Dodd adds only the κ-CG.

Encoding (cone-data Pattern III, like the hand-written `A1D5KAlg`)
-----------------------------------------------------------------
A canonical label is `((factors, e_E), κ)` — a `U1A1AoddKAlg` cone monomial
(`factors` a sorted tuple of `(a, i, exp)` chord powers, `e_E ∈ Z` the gauge
`E`-power) together with the SU(2) χ-index `κ` (highest weight κ, spin κ/2).
The SU(2) flavour lives in the **label** at the K-algebra surface (Z-form, the
form the RG finder consumes) and folds onto an `RLaurent[SU2]` **coefficient**
only via `r_label_decompose` — the single-irrep lift coordinate that *replaces*
the retired `_label_section_decompose`.

trace (SU(2)-refined Schur index)
---------------------------------
`Tr_UV(L_a)` is the UV Schur index — an **RG-flow quantity** (`⟨1, a⟩_RG`, the
vacuum pairing of `RG(a)·S_RG`), computed by the engine.  It does **not** factor
as `χ_κ · Tr_{u1a1aodd}(m)`: the RG dressing changes it (e.g. the identity trace
is the `D_5` vacuum character `1 + χ₂ q² + …` — the `q²` term the SU(2) adjoint
flavour current — not u1a1aodd's `1 − q²`).  k = 1 → D_5.

The ρ frame (5-fold, the genuine D-type cluster rotation)
--------------------------------------------------------
A subtlety worth stating: A1Dodd's ρ is **not** u1a1aodd's.  u1a1aodd's ρ is
6-fold (`(2k+4)`-gon, drifting in `E`); A1Dodd's is the **finite 5-fold** (for
k=1) `Z_{2k+3}` rotation of the once-punctured `(2k+3)`-gon — the genuine
D_{2k+3} cluster symmetry, exactly the frame of the hand-coded
`a1d5_cone_data`.  (They differ because u1a1aodd is the RG *auxiliary*, not the
UV theory.)  The *algebra* (multiply) still factors through u1a1aodd on the
clean sector, but ρ is the UV 5-fold rotation, supplied here by the engine.

Self-containment
----------------
Fully self-contained and **BPS-free**.  The complete engine is
`A1DoddRGKAlgebra(k)` — a standard `RGKAlgebra` over the frozen-table, BPS-free
`U1A1AoddKAlg(k).add_flavour(SU2)` with `S_RG = E_𝖖(μL)E_𝖖(μ⁻¹L)` — which derives
*everything* (multiply, ρ, trace) from the RG data.  `A1DoddKAlg` is a
**closed-form fast path** layered over that engine: it short-circuits the clean
sector (the bulk — products of chord rays and their cones, `U1A1AoddKAlg ⊗ CG`)
and defers the rest (the dressed sector = cone × ρ-image-of-cone, ρ, and the
trace) to the engine.  "Closed-form on the clean sector" is a *speed* property,
**not** a self-containment boundary — the engine is self-contained on all of it.
A closed form for the dressed sector would need the genuine D-type arc frame
(`a1d5_cone_data`); see `a1dodd_cone_cracking_notes.md`.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element
from cone_kalgebra import ConeKAlgebra
from laurent_poly import LaurentPoly
from zplus_ring import SU2ZPlusRing, RPowerSeries, RLaurent
# `u1a1aodd_kalg` is imported lazily inside `A1DoddEngineKAlg.__init__` (the only
# user), so importing this module for the self-contained `A1DoddConeKAlg` /
# `A1D{3,5,7}ConeKAlg` does NOT pull in the engine auxiliary.


class A1DoddEngineKAlg(KAlgebra):
    """`[A_1, D_{2k+3}]` (odd-D AD, SU(2) flavour) as a self-contained K-algebra
    over `R(SU(2))` — the **RG-flow / u1a1aodd-auxiliary frame** `u1a1aodd ⊗
    SU(2)-CG`.  See the module docstring.

    This is the *engine* presentation (the original `A1DoddKAlg`), kept as the
    BPS-free oracle/fallback.  The genuine **closed-form D-type cone
    presentation** (the `(a,p,i)` cluster frame, a `ConeKAlgebra`) is
    `A1DoddConeKAlg`; the public factory `A1DoddKAlg(k, presentation=...)`
    dispatches between them (default `"engine"` so existing callers/tests are
    byte-unchanged; pass `presentation="cone"` for the closed-form Ptolemy).

    Labels: `((factors, e_E), κ)` (a `U1A1AoddKAlg` cone monomial × SU(2) index
    `κ`).  `k = 1` is `[A_1, D_5]`."""

    def __init__(self, k: int = 1):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        from u1a1aodd_kalg import U1A1AoddKAlg
        self._U = U1A1AoddKAlg(k)
        self._R = SU2ZPlusRing()
        self._engine_cache = None
        self._clean_cache = None
        self._thr_cache = None

    # ----- KAlgebra primitives -------------------------------------------

    def coefficient_ring(self):
        return self._R

    def identity(self):
        # u1a1aodd identity cone monomial, SU(2) singlet.
        return (self._U.identity(), 0)

    def multiply(self, a, b) -> Element:
        """`(m_a, κ_a)·(m_b, κ_b)`.

        **Clean sector — closed-form, BPS-free, fast.**  When both inputs are
        *clean* (RG-monomial, `e_E ≥ Σ exp·thr`; chord rays and their cone
        products) the structure constants factor:

            multiply = U1A1AoddKAlg.multiply(m_a, m_b) ⊗ CG(χ_{κ_a}·χ_{κ_b})

        — the u1a1aodd cone product on the monomial part, tensored with the
        SU(2) Clebsch–Gordan on the flavour index (verified vs the engine:
        chords 729/729, cone monomials 900/900, daughter-closure 1908/1908).

        **Dressed sector — deferred to the self-contained RG engine.**  When an
        input is *below threshold* (the genuinely RG-dressed elements — chiefly
        products of cones with **ρ-images of cones**, where the 5-fold ρ pushes a
        chord below its clean `E`-frame) the factorisation fails: the dressing is
        bound to the low-`E` frame (neither ρ nor the central-looking `E` shift
        recovers it; verified — `(x·y_clean)·E⁻¹` drops a daughter the engine
        keeps), so the product is computed by the BPS-free engine
        `A1DoddRGKAlgebra`.  This is a *speed* fallback, not a self-containment
        gap; a closed form here needs the genuine arc frame (module docstring)."""
        (m_a, ka), (m_b, kb) = a, b
        if self._is_clean(m_a) and self._is_clean(m_b):
            mono = self._U.multiply(m_a, m_b)        # clean ⇒ factorisation exact
            cg = self._R.multiply_basis(ka, kb)      # {κ_c: multiplicity}
            out: dict = {}
            for d, lp in mono.terms.items():
                if lp.is_zero():
                    continue
                for kc, mult in cg.items():
                    key = (d, kc)
                    add = lp * mult
                    out[key] = out[key] + add if key in out else add
            return Element({l: c for l, c in out.items() if not c.is_zero()})
        # dressed sector: the RG engine (correct; the self-contained
        # closed form is the open dressing-relation extraction).
        return self._engine.multiply(a, b)

    # ----- clean-sector test (closed-form after one threshold scan) -------

    def thresholds(self):
        """`{(a,i): thr}` clean thresholds (min RG-monomial E-power); computed
        once from the engine, after which the clean test is closed-form."""
        if self._thr_cache is None:
            from a1dodd_mult_table import clean_thresholds
            self._thr_cache = clean_thresholds(self._engine, self._U.cone_data())
        return self._thr_cache

    def _is_clean(self, mono) -> bool:
        """A u1a1aodd cone monomial `(factors, e_E)` is clean (RG-monomial,
        un-dressed) iff `e_E ≥ Σ exp·thr(a,i)` — the half-line threshold
        (verified exact vs the engine's RG, k=1: 300/300)."""
        factors, e_E = mono
        thr = self.thresholds()
        return e_E >= sum(ex * thr[(a, i)] for (a, i, ex) in factors)

    def rho(self, a):
        """ρ is the genuine **Z_{2k+3} once-punctured-`(2k+3)`-gon cluster
        rotation** of `[A_1, D_{2k+3}]` (k=1 → the 5-fold rotation of
        `a1d5_cone_data`), a *finite-order* automorphism — **not**
        u1a1aodd's (6-fold, drifting) ρ.  The two differ because u1a1aodd is the
        RG *auxiliary*, not the UV theory.  Supplied by the engine's UV ρ
        (a quick label map); SU(2) flavour is ρ-fixed (self-dual irreps)."""
        return self._engine.rho(a)

    def rho_inverse(self, a):
        return self._engine.rho_inverse(a)

    def rho_squared_is_identity(self) -> bool:
        return self._engine.rho_squared_is_identity()

    # ----- flavour-lift coordinate (replaces _label_section_decompose) ----

    def r_label_decompose(self, label):
        """The single-irrep flavour-lift coordinate `(section, r_basis_label)`,
        `L_label = χ_κ · L_section`: peel the SU(2) index κ off the label onto
        the coefficient ring, leaving the flavour-singlet section `(m, 0)`.
        This is *the* lift primitive — `_label_section_decompose` is not
        implemented (it derives from this by the `KAlgebra` bridge)."""
        (m, k) = label
        return (m, 0), k

    def r_label_compose(self, section, r_basis_label):
        """Inverse of `r_label_decompose`: write the SU(2) index back into the
        label.  A direct slot write (the flavour is central)."""
        (m, _zero) = section
        return (m, r_basis_label)

    # ----- trace (SU(2)-refined Schur index) -----------------------------

    def trace(self, a, K: int = 20) -> RPowerSeries:
        """`Tr_UV(L_a)` — the **SU(2)-refined Schur index** of `[A_1, D_{2k+3}]`.

        The trace is an **RG-flow quantity** (`Tr_UV(L_a) = ⟨1, a⟩_RG`, the pairing
        of `RG(a)·S_RG` against the vacuum), *not* the auxiliary's trace: it does
        **not** factor as `χ_κ · Tr_{u1a1aodd}(m)` (the RG dressing changes it —
        e.g. the identity trace is the `D_5` vacuum character `1 + χ₂q² + …`, the
        `q²` term the SU(2) adjoint flavour current, not u1a1aodd's `1 − q²`).  So
        it is computed by the self-contained RG engine `A1DoddRGKAlgebra`."""
        return self._engine.trace(tuple(a), K)

    # ----- convenience builders ------------------------------------------

    def L(self, a, i, kappa: int = 0):
        """Canonical chord ray `(a, i)` (a `U1A1AoddKAlg` chord) at SU(2) index
        `kappa` (default the flavour singlet)."""
        return ((((a, i, 1),), 0), kappa)

    def chi(self, kappa: int):
        """The central SU(2) character `χ_κ` as a canonical label (the identity
        cone monomial at flavour `κ`)."""
        return (self._U.identity(), kappa)

    # ----- the self-contained RG engine (clean frames / geometry / ρ) -----

    @property
    def _engine(self):
        """The complete, self-contained, BPS-free RG realisation
        `A1DoddRGKAlgebra(k)` (auxiliary = `U1A1AoddKAlg(k).add_flavour(SU2)`,
        `S_RG = E_𝖖(μL)E_𝖖(μ⁻¹L)`).  This *is* the algebra; `A1DoddKAlg` is a fast
        closed-form path over it on the clean sector."""
        if self._engine_cache is None:
            from a1dodd_rgkalgebra import A1DoddRGKAlgebra
            self._engine_cache = A1DoddRGKAlgebra(self.k)
        return self._engine_cache

    def clean_frames(self):
        """`{(a,i): eE*}` — the clean E-frame of each chord ray (the E-power at
        which its monomial `RG` is single-term).  Lazily computed from the
        engine; used for the chord-ray geometry and validation."""
        if self._clean_cache is None:
            from a1dodd_mult_table import clean_frames
            self._clean_cache = clean_frames(self._engine, self._U.cone_data())
        return self._clean_cache

    def clean_chord(self, a, i, kappa: int = 0):
        """Chord ray `(a, i)` at its clean (un-dressed, RG-monomial) E-frame,
        SU(2) index `kappa` — the canonical generator."""
        return ((((a, i, 1),), self.clean_frames()[(a, i)]), kappa)

    def geometric_label(self, label):
        """A single-chord ray as a once-punctured `(2k+3)`-gon arc
        `(endpoints, winding)` — delegated to the engine's geometric label
        (collapse a `(2k+4)`-gon vertex onto the puncture).  `None` for
        non-single-chord / cluster-monomial labels."""
        return self._engine.geometric_label(label)

    # ----- verification ---------------------------------------------------

    def verify_clean_sector(self, max_kappa: int = 2):
        """Cross-check the **closed-form clean-sector** `multiply`
        (`U1A1AoddKAlg ⊗ CG`) against the RG engine on the chord rays at
        their clean frames and their q-commuting cone-monomial products (both
        inputs clean).  This is the genuinely-cracked content; the dressed
        sector is the engine by construction.  Returns `(n_ok, n_total)`."""
        T = self._engine
        clean = self.clean_frames()
        chords = self._U.cone_data()._chords
        ucd = self._U.cone_data()

        def clab(a, i, kp):
            return ((((a, i, 1),), clean[(a, i)]), kp)

        gens = [clab(a, i, 0) for (a, i) in chords]
        for idx, (a, i) in enumerate(chords):
            for (b, j) in chords[idx:]:
                if (a, i) != (b, j) and ucd.q_commute((a, i), (b, j)):
                    m = (tuple(sorted([(a, i, 1), (b, j, 1)])),
                         clean[(a, i)] + clean[(b, j)])
                    gens.append((m, 0))

        n_ok = n_tot = 0
        for ga in gens:
            for gb in gens:
                for ka in range(max_kappa + 1):
                    for kb in range(max_kappa + 1):
                        la, lb = (ga[0], ka), (gb[0], kb)
                        if not (self._is_clean(la[0]) and self._is_clean(lb[0])):
                            continue
                        n_tot += 1
                        if self.multiply(la, lb).terms == T.multiply(la, lb).terms:
                            n_ok += 1
        return n_ok, n_tot


# ===========================================================================
# A1DoddConeKAlg — the genuine closed-form D-type cone presentation
# ===========================================================================


class _TraceAdapter:
    """Word-frame view of the cone-data for the Layer-1 trace reducer.

    `ConeData.simplify_trace_via_cone_data(alg, native_word)` needs `alg.rho`,
    `alg.rho_inverse`, `alg._canonical_rho2_orbit_rep`, `alg.rho_squared_is_
    identity` acting on *cone-data native labels* (bare `(a,p,i)`-words).  The
    K-algebra's own `rho` acts on `(word, κ)` labels, so we wrap the cone-data's
    word-frame ρ (`rho_native`, the clean `i -> i+1` rotation) here.  ρ² on the
    finite once-punctured-`(2k+3)`-gon has finite order (period | 2k+3), so the
    default orbit-walk canonicalisation terminates."""

    def __init__(self, cone_data):
        self._cd = cone_data

    def rho(self, word):
        return self._cd.rho_native(word)

    def rho_inverse(self, word):
        return self._cd.rho_inv_native(word)

    def rho_squared_is_identity(self) -> bool:
        # i -> i+2 mod (2k+3); identity iff 2 ≡ 0 mod (2k+3), i.e. never (H>=3).
        return False

    def _canonical_rho2_orbit_rep(self, word):
        """Min-by-sort representative of the ρ²-orbit (orbit walk)."""
        orbit = [word]
        seen = {word}
        cur = self.rho(self.rho(word))
        steps = 0
        while cur not in seen:
            steps += 1
            if steps > 4 * self._cd.H + 4:
                raise RuntimeError(
                    f"_TraceAdapter: ρ²-orbit of {word!r} did not close"
                )
            seen.add(cur)
            orbit.append(cur)
            cur = self.rho(self.rho(cur))
        return min(orbit)


class A1DoddConeKAlg(ConeKAlgebra):
    """`[A_1, D_{2k+3}]` (odd-D AD, SU(2) flavour) as a **standalone
    `ConeKAlgebra`** in the genuine D-type cluster frame — the once-punctured
    `(2k+3)`-gon `(a, p, i)` labeling (`k=0 → D_3 = a1d3`, `k=1 → D_5 = a1d5`),
    with the closed-form Ptolemy multiply via `a1dodd_cone_data(k)`.

    This is the closed-form analog of the A-type `A1A2k_plucker_closed_form` —
    a genuine cluster presentation (NOT engine extraction): cocycle = the
    A_{2k+2} chain pairing, ρ = the clean `Z_{2k+3}` rotation `i -> i+1`, and the
    multiply ρ-lifts from the `i=0` Plücker row (sourced from the decoded
    ground truth a1d5 / a1d3; see `a1dodd_cone_data` for the closed-form status
    and the open puncture-crossing rule).

    Native label (Pattern III, like `A1D5KAlg`): `(word, κ)` where

      * `word`  = the cone-monomial — a sorted tuple of `((a,p,i), power)` pairs
                  (the χ-stripped canonical-basis g-vector of the genuine frame),
      * `κ ≥ 0` = the SU(2) flavour index (highest weight κ, spin κ/2).

    `multiply` splits κ at the boundary, runs the cone-data `derived_multiply`
    on the χ-stripped `word`, fuses `χ_{κ_a}·χ_{κ_b}` by SU(2) Clebsch–Gordan,
    and re-expands.  `trace` (inherited from `ConeKAlgebra`) runs the ρ²-cycle
    Layer-1 reducer then `_trace_residual` = `a1dodd_layer2.seed_trace_ap`.
    """

    def __init__(self, k: int = 1):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        self.k = k
        self._R = SU2ZPlusRing()
        self._cone_data_cache = None

    # ----- KAlgebra primitives -------------------------------------------

    def coefficient_ring(self):
        return self._R

    def identity(self):
        # empty cone-monomial word, SU(2) singlet.
        return ((), 0)

    def cone_data(self):
        if self._cone_data_cache is None:
            from a1dodd_cone_data import a1dodd_cone_data
            self._cone_data_cache = a1dodd_cone_data(self.k)
        return self._cone_data_cache

    def canonicalise(self, label):
        """Canonical native label: word sorted with positive powers, κ ≥ 0."""
        word, kappa = label
        word = tuple(sorted((tuple(g), int(p)) for (g, p) in word if int(p) > 0))
        return (word, int(kappa))

    def multiply(self, a, b) -> Element:
        """`(word_a, κ_a)·(word_b, κ_b)` — cone-data product ⊗ SU(2) CG.

        Mirrors `A1D5KAlg.multiply`: strip κ at the boundary, run the cone-data
        `derived_multiply` on the χ-stripped words, fuse `χ_{κ_a}·χ_{κ_b}` by
        SU(2) Clebsch–Gordan, and re-expand into `(word, κ_out)` labels with
        integer q-coefficients."""
        a = self.canonicalise(a)
        b = self.canonicalise(b)
        (word_a, k_a), (word_b, k_b) = a, b
        cone_result = self.cone_data().derived_multiply(word_a, word_b)
        chi_prod = self._R.basis_element(k_a) * self._R.basis_element(k_b)
        out: dict = {}
        for word_out, coef in cone_result.terms.items():
            if isinstance(coef, LaurentPoly):
                rl = RLaurent(self._R, dict(coef._coeffs))
            else:
                rl = coef
            for q_exp, r_elt in rl.coeffs.items():
                scaled = r_elt * chi_prod
                if scaled.is_zero():
                    continue
                for k_out, c_int in scaled.terms.items():
                    if c_int == 0:
                        continue
                    key = (word_out, k_out)
                    lp_add = LaurentPoly({q_exp: int(c_int)})
                    out[key] = out.get(key, LaurentPoly({})) + lp_add
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    def rho(self, a):
        """ρ = the clean `Z_{2k+3}` once-punctured-`(2k+3)`-gon rotation
        `(a,p,i) -> (a,p,i+1)` on every factor; SU(2) flavour ρ-fixed."""
        word, kappa = self.canonicalise(a)
        return (self.cone_data().rho_native(word), kappa)

    def rho_inverse(self, a):
        word, kappa = self.canonicalise(a)
        return (self.cone_data().rho_inv_native(word), kappa)

    # ----- trace (Layer-1 ρ²-cycle on words + Layer-2 character) ----------
    #
    # The cone-data Layer-1 reducer (`simplify_trace_via_cone_data`) operates
    # on the *cone-data* native labels (bare `(a,p,i)`-words, χ stripped), and
    # needs `rho`/`rho_inverse`/`_canonical_rho2_orbit_rep` acting on *those*
    # words — not on the K-algebra `(word, κ)` labels.  We give it a thin
    # word-frame adapter (`_TraceAdapter`) and plug Layer-2 ourselves, then
    # fold the κ-character back in.  (Mirrors a1d5/a1d3 carrying χ in the
    # coefficient at the trace boundary.)

    def _trace_residual(self, seed_word, K: int) -> RPowerSeries:
        """Closed-form trace of a Layer-1 seed (χ-stripped cone *word*), via
        `a1dodd_layer2`:

          * `()`               → `vacuum_trace(k, K)` (the sl(2) vacuum char);
          * `(((a,p,i),1),)`   → `seed_trace_ap(k, a, p, K)` (ρ²-invariant, so
                                  position-independent)."""
        from a1dodd_layer2 import vacuum_trace, seed_trace_ap
        if seed_word == ():
            qmu = vacuum_trace(self.k, K)
        else:
            if len(seed_word) != 1 or seed_word[0][1] != 1:
                raise ValueError(
                    f"_trace_residual: expected a single-mult-gen seed word, "
                    f"got {seed_word!r}"
                )
            (a, p, _i) = seed_word[0][0]
            qmu = seed_trace_ap(self.k, a, p, K)
        coeffs = {}
        for q, irreps in qmu.items():
            if q > K:
                continue
            re = self._R.zero()
            for n, c in irreps.items():
                if c:
                    re = re + self._R.basis_element(n) * c
            if not re.is_zero():
                coeffs[q] = re
        return RPowerSeries(self._R, coeffs, K)

    def trace(self, a, K: int = 20) -> RPowerSeries:
        """SU(2)-refined Schur index `Tr(L_{(word, κ)})`.

        Layer 1 = the cone-data ρ²-cyclicity reducer on the χ-stripped `word`
        (the seeds = identity + single mult-gens, ρ²-canonicalised), Layer 2 =
        `_trace_residual` (= `a1dodd_layer2`).  The κ-character `χ_κ` is
        central, folded onto the coefficient at the end."""
        word, kappa = self.canonicalise(a)
        cd = self.cone_data()
        adapter = _TraceAdapter(cd)
        simplified: Element = cd.simplify_trace_via_cone_data(adapter, word)
        # Accumulate Σ_seed c_seed(q) · Tr(seed)(q).
        emin = 0
        for c_poly in simplified.terms.values():
            coeffs = (c_poly._coeffs if isinstance(c_poly, LaurentPoly)
                      else c_poly.coeffs)
            for e in coeffs:
                emin = min(emin, e)
        inner_K = K - emin
        result = RPowerSeries.zero(self._R, K)
        for seed_word, c_poly in simplified.terms.items():
            seed_trace = self._trace_residual(seed_word, inner_K)
            if isinstance(c_poly, LaurentPoly):
                c_rl = RLaurent(self._R, dict(c_poly._coeffs))
            else:
                c_rl = c_poly
            product = seed_trace * c_rl
            trimmed = RPowerSeries(
                self._R, {e: c for e, c in product.coeffs.items() if e <= K}, K,
            )
            result = result + trimmed
        if kappa:
            chi = RLaurent(self._R, {0: self._R.basis_element(kappa)})
            result = result * chi
            result = RPowerSeries(
                self._R, {q: c for q, c in result.coeffs.items() if q <= K}, K,
            )
        return result

    # ----- flavour-lift coordinate ---------------------------------------

    def r_label_decompose(self, label):
        """The single-irrep flavour-lift coordinate `(section, r_basis_label)`,
        `L_label = χ_κ · L_section`: peel κ off onto the coefficient ring,
        leaving the flavour-singlet section `(word, 0)`."""
        word, kappa = self.canonicalise(label)
        return (word, 0), kappa

    def r_label_compose(self, section, r_basis_label):
        """Inverse of `r_label_decompose`: write the SU(2) index into the
        label.  A direct slot write (the flavour is central)."""
        word, _zero = section
        return self.canonicalise((word, r_basis_label))

    # ----- convenience builders ------------------------------------------

    def gen(self, a, p, i, kappa: int = 0):
        """The atomic mult-gen `(a,p,i)` at SU(2) index `kappa`
        (a=level, p=parity, i=position): native label `((((a,p,i), 1),), κ)`."""
        return ((((a, p, i), 1),), kappa)

    def chi(self, kappa: int):
        """The central SU(2) character `χ_κ` (the identity word at flavour κ)."""
        return ((), kappa)

    def __repr__(self):
        return f"A1DoddConeKAlg(k={self.k})  # [A_1, D_{2*self.k+3}], cone frame"


# ===========================================================================
# Public factory
# ===========================================================================


def A1DoddKAlg(k: int = 1, presentation: str = "engine"):
    """`[A_1, D_{2k+3}]` (odd-D AD, SU(2) flavour) over `R(SU(2))`.

    `presentation`:
      * ``"engine"`` (default) — the BPS-free RG-flow / u1a1aodd-auxiliary
        realisation `A1DoddEngineKAlg` (the original `A1DoddKAlg`; labels
        `((factors, e_E), κ)`).  Self-contained on all sectors; multiply is
        closed-form on the clean sector and engine-backed on the dressed
        sector.  Kept as the oracle/fallback — existing callers/tests are
        byte-unchanged.
      * ``"cone"`` — the genuine closed-form **D-type cone presentation**
        `A1DoddConeKAlg` (labels `(word, κ)` in the `(a,p,i)` cluster frame;
        closed-form Ptolemy multiply via `a1dodd_cone_data(k)`).  `k=0 → D_3`,
        `k=1 → D_5`.

    Default is ``"engine"`` purely for backward compatibility; the cone
    presentation is the new genuine-cluster deliverable (verified to reproduce
    a1d5 at k=1 and a1d3 at k=0)."""
    if presentation == "engine":
        return A1DoddEngineKAlg(k)
    if presentation == "cone":
        return A1DoddConeKAlg(k)
    raise ValueError(
        f"A1DoddKAlg: presentation must be 'engine' or 'cone', got "
        f"{presentation!r}"
    )


# ===========================================================================
# Explicitly-named, truly self-contained, exportable D-type cone algebras
# ===========================================================================
#
# `A1D{3,5,7}ConeKAlg` are the genuine closed-form D-type cone presentations of
# `[A_1, D_n]` for n = 2k+3 at fixed k = 0, 1, 2 — the **first complete standalone
# `[A_1, D_n]` with BOTH closed-form multiply AND closed-form trace** (the older
# `A1D5KAlg`/`A1D7KAlg` lack the trace).  They are *truly self-contained*: the
# multiply (`a1dodd_cone_data`, frozen i=0 table in `a1dodd_cone_tables`) needs no
# a1d5_kalg / a1d5_decomposer / finite_a1d7_kalg, and the engine auxiliary
# (u1a1aodd) is never imported on the cone path.  Runtime deps are only the cone
# framework (`cone_kalgebra`, `cone_data`, `kalgebra`, `zplus_ring`,
# `laurent_poly`) and the closed-form SU(2) trace (`a1dodd_layer2`, with the
# shared `a1d3_kalg` Verma/Laurent helpers) — all in `export/ConeKAlgebra/`.
#
# multiply reproduces the references exactly (a1d3 k=0, a1d5 k=1, FiniteA1D7 k=2);
# bar-involution + ρ-automorphism pass; `Tr(1)` = the sl(2) vacuum character.


class A1D3ConeKAlg(A1DoddConeKAlg):
    """`[A_1, D_3]` (= k=0) genuine D-type cone algebra over R(SU(2)).
    Self-contained, exportable.  See the module header."""

    def __init__(self):
        super().__init__(0)

    def __repr__(self):
        return "A1D3ConeKAlg()  # [A_1, D_3], self-contained cone frame"


class A1D5ConeKAlg(A1DoddConeKAlg):
    """`[A_1, D_5]` (= k=1) genuine D-type cone algebra over R(SU(2)) — the first
    complete standalone [A_1,D_5] with both closed-form multiply and trace.
    Self-contained, exportable.  See the module header."""

    def __init__(self):
        super().__init__(1)

    def __repr__(self):
        return "A1D5ConeKAlg()  # [A_1, D_5], self-contained cone frame"


class A1D7ConeKAlg(A1DoddConeKAlg):
    """`[A_1, D_7]` (= k=2) genuine D-type cone algebra over R(SU(2)) — the first
    complete standalone [A_1,D_7] with both closed-form multiply and trace.
    Self-contained, exportable.  See the module header."""

    def __init__(self):
        super().__init__(2)

    def __repr__(self):
        return "A1D7ConeKAlg()  # [A_1, D_7], self-contained cone frame"


if __name__ == "__main__":
    # Fast: the genuine closed-form D-type cone presentation (no engine build).
    C = A1DoddKAlg(1, presentation="cone")
    print("A1DoddKAlg(1, 'cone') = [A_1, D_5]  (genuine D-type cone frame)")
    print("  ring =", C.coefficient_ring())
    g = C.gen(1, 0, 0, 0)         # mult-gen (a=1,p=0)=T at position 0
    g1 = C.gen(1, 0, 1, 0)        # T at position 1
    print(f"  L(T_0)·L(T_1) = {C.multiply(g, g1)}")
    d = C.gen(1, 0, 0, 1)         # T_0 as an SU(2) doublet (κ=1)
    print("  L(T_0)_doublet · L(T_0)_doublet =")
    for l, c in sorted(C.multiply(d, d).terms.items(), key=str):
        print(f"     {l}:  {c}")
    print(f"  Tr(1)  = {C.trace(C.identity(), K=6)}   # D_5 vacuum character")
    print(f"  Tr(χ₁) = {C.trace(C.chi(1), K=4)}")
    print()
    # Slow: the BPS-free RG-flow / u1a1aodd-auxiliary presentation (the oracle).
    A = A1DoddKAlg(1, presentation="engine")
    print("A1DoddKAlg(1, 'engine') = [A_1, D_5]  (SU(2) flavour, RG frame)")
    print("  Tr(χ_1 · 1) =", A.trace(A.chi(1), K=6))
