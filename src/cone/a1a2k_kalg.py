"""
a1a2k_kalg.py
=============

The standalone K-algebra subclass for the [A_1, A_{2k}] Argyres-Douglas
family, analogous to `kalgebra_samples.PentagonKAlg` and the upcoming
`HeptagonKAlg` (claude/heptagon-wrapper-continue branch).

Generators `L((k_letter, i))` with `k_letter ∈ {1, ..., k}` (orbit
index) and `i ∈ Z/(2k+3)` (within-orbit ρ-index).  Canonical basis
labels are sorted tuples

    ((k_1, i_1, e_1), ..., (k_m, i_m, e_m))

of (letter, positive exponent) entries with pairwise q-commuting
letters; the empty tuple is the identity.

Multiplication is driven by a per-k base table extracted ONCE at
construction time via the `A1A2k` wrapper.  After construction the
class has no `BPSKAlgebra` runtime dependency; the wrapper is the
helper that supplies the per-k structural data.

Scope:
  * `coefficient_ring`, `identity`, `multiply`, `rho`, `rho_inverse`,
    `_label_section_decompose`, plus the `L((k, i))` accessor.
  * `trace()` is a full two-layer trace:
      Layer 1 reduces  Tr(X-label)  to  Σ c_i T_i  via repeated ρ²-
        twisted cyclicity, with the X-basis ↔ L-product correction
        (`q^{-bps_canon + bps_left}` applied at each peel).  Cycles
        in the recursion are resolved by a Q(q) linear-equation
        solver across reachable labels.
      Layer 2 substitutes  T_0 = χ_1(fq²)  and
        T_i = (-1)^{m+1} fq^{-m} (χ_m - χ_{m+1})(fq²),  m = k-i+1
        with χ_s the M(2, 2k+3) Andrews-Gordon character
        (`minimal_model_characters.char_product`).

Verified against `A1A2k(k).A.trace(...)` for all canonical labels with
total exponent ≤ 3 at k = 2 (heptagon, 43 labels) and k = 3
(nonagon).  Multiplication is closed under canonicalisation;
ρ-equivariance verified.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laurent_poly import LaurentPoly
from zplus_ring import ZPlusRing, RPowerSeries, TrivialZPlusRing
from kalgebra import KAlgebra, Element
from cone_kalgebra import ConeKAlgebra


# ---------------------------------------------------------------------------
# A1A2kKAlg(KAlgebra)
# ---------------------------------------------------------------------------


class A1A2kKAlg(ConeKAlgebra):
    """`A_𝖖([A_1, A_{2k}])` as a `KAlgebra` subclass, parameterised by k.

    Generators `L((k_letter, i))` with `k_letter ∈ {1, ..., k}` and
    `i ∈ Z/(2k+3)`.  ρ-orbit indexing matches
    `heptagon_kalg.HeptagonKAlg` at k = 2 and the verified
    `A1A2k_naming_audit.predicted_lengths_and_shifts(k)` table for
    k ≥ 3.  Multiplication is driven by a per-k base product table
    `self._base_table` extracted at construction from the
    `A1A2k` wrapper.

    Canonical-basis labels are sorted tuples
    `((k_1, i_1, e_1), ..., (k_m, i_m, e_m))` with `e_r ≥ 1` and the
    letter pairs `(k_r, i_r)` pairwise q-commuting; empty tuple is
    the identity."""

    _R = TrivialZPlusRing()

    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.H = 2 * k + 3   # cyclic order = (2k+3)-gon vertex count
        # Natural labeling: orbit a has length a+1 and shift 0, so
        # L_{a, j} has chord endpoints (j, j+a+1) on the (2k+3)-gon.
        self.lengths = {a: a + 1 for a in range(1, k + 1)}
        self.shifts = {a: 0 for a in range(1, k + 1)}
        # Closed-form base table from chord geometry — no BPSKAlgebra call.
        # (Was: A1A2k(k).base_table() which built a BPSKAlgebra at construction;
        # at k=4 that took ~90s.  The closed-form is < 1s at any k.)
        from A1A2k_plucker_closed_form import base_table_predict
        self._base_table = base_table_predict(k)
        # Cache q-commute factors among all (k_letter, i) pairs for fast lookup.
        self._qc_cache: dict[
            tuple[tuple[int, int], tuple[int, int]], int | None
        ] = {}
        self._fwd_q_cache: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        for ka in range(1, k + 1):
            for kb in range(1, k + 1):
                for d in range(self.H):
                    la = (ka, 0)
                    lb = (kb, d)
                    entries = self._base_table[(ka, kb, d)]
                    if len(entries) == 1:
                        # Monomial: q-commute factor exists.
                        kind, c_fwd = entries[0]
                        # Compute backward q-coefficient via reverse table entry.
                        rev = self._base_table[(kb, ka, (-d) % self.H)]
                        if len(rev) == 1:
                            _, c_bwd = rev[0]
                            self._qc_cache[(la, lb)] = c_fwd - c_bwd
                            self._fwd_q_cache[(la, lb)] = c_fwd
                        else:
                            self._qc_cache[(la, lb)] = None
                    else:
                        self._qc_cache[(la, lb)] = None
        # We don't need the wrapper anymore; drop the reference.
        # (Garbage collection will reclaim the BPSKAlgebra instance.)

    # -- KAlgebra primitives -------------------------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return ()

    def L(self, label: tuple[int, int]):
        """Canonical-form label for the single named generator
        `L((k_letter, i))`.  `k_letter ∈ {1, ..., k}`, `i` mod (2k+3)."""
        k_letter, i = label
        if not (1 <= k_letter <= self.k):
            raise ValueError(
                f"k_letter must be in [1, {self.k}], got {k_letter}"
            )
        return ((k_letter, i % self.H, 1),)

    def rho(self, a):
        if not a:
            return ()
        shifted = [(k_l, (i + 1) % self.H, e) for (k_l, i, e) in a]
        shifted.sort()
        return tuple(shifted)

    def rho_inverse(self, a):
        if not a:
            return ()
        shifted = [(k_l, (i - 1) % self.H, e) for (k_l, i, e) in a]
        shifted.sort()
        return tuple(shifted)

    def cone_data(self):
        """Per-instance `A1A2kConeData` sidecar."""
        cached = getattr(self, "_cone_data_instance", None)
        if cached is None:
            from a1a2k_cone_data import A1A2kConeData
            cached = A1A2kConeData(self)
            self._cone_data_instance = cached
        return cached

    def multiply(self, a, b):
        return self._multiply_via_cone_data(a, b)

    def _legacy_multiply(self, a, b):
        """Legacy multiply via `_label_canon_twist` + `_reduce` (the
        pre-cone-data implementation).  Kept as a test oracle; not on
        the live `multiply` path.

        Labels represent X-basis canonical-basis elements:
            label  =  q^{-bps_twist(label)}  ·  L-product(label sorted).
        So concatenating two label-words as L-products represents
            X_a · X_b · q^{-bps_a - bps_b}^{-1}  =  q^{bps_a + bps_b} X_a · X_b ?
        Actually: X_a = q^{-bps_a} L_a-sorted, so
            L_a-sorted · L_b-sorted  =  q^{bps_a + bps_b} · X_a · X_b.
        We want  X_a · X_b, so divide by q^{bps_a + bps_b}.
        Equivalently, multiply the reduced output by q^{-bps_a - bps_b}."""
        bps_a = self._label_canon_twist(a)
        bps_b = self._label_canon_twist(b)
        word = list(a) + list(b)
        out = self._reduce(word)
        shift = -bps_a - bps_b
        if shift == 0:
            return Element(out)
        factor = LaurentPoly.q(shift)
        return Element({lbl: coef * factor for lbl, coef in out.items()})

    def _label_canon_twist(self, label) -> int:
        """The q-exponent c such that  X_label  =  q^{-c}  L-product(label).
        Computed as  Σ_{i<j} c_fwd(letter_i, letter_j) * e_i * e_j  over
        pairs in canonical (sorted) order."""
        twist = 0
        for ii in range(len(label)):
            for jj in range(ii + 1, len(label)):
                la = (label[ii][0], label[ii][1])
                lb = (label[jj][0], label[jj][1])
                ea = label[ii][2]
                eb = label[jj][2]
                twist += self._forward_q_coeff(la, lb) * ea * eb
        return twist

    # ----- Layer-2 trace residual ---------------------------------------
    #
    # Canonical ρ²-orbit seeds produced by Layer 1 (tagged-cycle +
    # ρ²-orbit canonicalisation in `simplify_trace_via_cone_data`):
    #   * `()`                  -- identity              → T_0 = χ_1(q²)
    #   * `((a, 0, 1),)` for    -- canonical orbit-a     → T_a
    #     a ∈ {1, ..., k}          seed (min of ρ²-orbit
    #                              {((a, j, 1),) : j ∈ Z/(2k+3)})
    #
    # Layer-2 plug-in: M(2, 2k+3) Andrews-Gordon characters via
    # `_compute_T_series(K)`:
    #     T_0  =  χ_1(q²)
    #     T_a  =  (-1)^{m+1} q^{-m} (χ_m − χ_{m+1})(q²),  m = k − a + 1
    # Verified for k = 1..6 against BPS reference (PR #188).
    #
    # `multiply` and `trace` are inherited from `ConeKAlgebra`.  ρ²-
    # invariance on the (2k+3) single-mult-gen seeds per orbit is
    # enforced by Layer 1, not by this method.

    def _trace_residual(self, seed_label, K):
        T_series = self._compute_T_series(K)
        if seed_label == ():
            return T_series[0]
        # Single-letter seeds: ((a, 0, 1),) for a in {1, ..., k}.
        if (
            isinstance(seed_label, tuple)
            and len(seed_label) == 1
            and isinstance(seed_label[0], tuple)
            and len(seed_label[0]) == 3
        ):
            a, i, e = seed_label[0]
            if i == 0 and e == 1 and 1 <= a <= self.k:
                return T_series[a]
        raise ValueError(
            f"A1A2kKAlg._trace_residual: unexpected seed {seed_label!r}; "
            f"expected canonical ρ²-orbit representative "
            f"() or ((a, 0, 1),) for a in 1..{self.k}"
        )

    def _compute_T_series(self, K: int) -> list[RPowerSeries]:
        """Compute T_0, T_1, ..., T_k as RPowerSeries in fq.

        T_0 = chi_1(fq²)
        T_a = (-1)^{m+1} fq^{-m} (chi_m - chi_{m+1})(fq²)  for a ≥ 1,
              where m = m(a) is the alternating-pattern "number of ones"
              corresponding to the length-(a+1) chord in the natural
              labeling.  In the old (alternating-pattern) labeling this
              was m = k - i + 1 with i the old orbit index; in the new
              labeling we look up the old orbit index via
              `predicted_lengths_and_shifts` so that orbit a (length a+1)
              receives the correct M(2, 2k+3) character difference.

        The fq² substitution doubles all q-exponents in chi_s(q).
        The fq^{-m} prefactor shifts by -m."""
        from minimal_model_characters import char_product
        from A1A2k_naming_audit import predicted_lengths_and_shifts
        T: list[RPowerSeries] = []
        # T_0 = chi_1(fq²)
        # Need terms up to fq^K, so q-power up to K // 2.
        K_q = (K + 1) // 2 + 1  # a touch extra for safety
        chi_1 = char_product(self.k, 1, K_q)
        t0_coeffs = {2 * n: c for n, c in enumerate(chi_1)
                     if c != 0 and 2 * n <= K}
        T.append(RPowerSeries(self._R, t0_coeffs, K))
        # Map: natural orbit a → m(a) via the old length-i ↔ orbit-i table.
        old_lengths, _ = predicted_lengths_and_shifts(self.k)
        old_idx_of_length = {L: i for i, L in old_lengths.items()}
        # T_a for a = 1, ..., k
        for a in range(1, self.k + 1):
            length = a + 1
            i_old = old_idx_of_length[length]
            m = self.k - i_old + 1
            sign = 1 if m % 2 == 1 else -1
            # We want terms up to fq^K after the fq^{-m} shift, so q-power up
            # to (K + m) // 2 + 1.
            K_q_i = (K + m) // 2 + 1
            chi_a = char_product(self.k, m, K_q_i)
            if m + 1 <= self.k + 1:
                chi_b = char_product(self.k, m + 1, K_q_i)
            else:
                chi_b = [0] * len(chi_a)
            diff = [a - b for a, b in zip(chi_a, chi_b)]
            coeffs = {}
            for n, c in enumerate(diff):
                if c == 0:
                    continue
                fq_exp = 2 * n - m
                if fq_exp <= K:
                    coeffs[fq_exp] = sign * c
            T.append(RPowerSeries(self._R, coeffs, K))
        return T

    # ---- Trace Layer 1 ----

    def trace_layer1(self, label):
        """Reduce  Tr(label)  to a tuple  (c_0, c_1, ..., c_k)  of
        LaurentPoly's such that  Tr(label) = Σ_i c_i · T_i,  where
        T_0 = Tr(1)  and  T_i = Tr(L((i, *)))  for i = 1, ..., k.

        Routes through `cone_data().simplify_trace_via_cone_data`: the
        generic tagged-cyclicity engine reduces `L_label` to an Element
        supported on `A1A2kKAlg`'s trace seeds (identity + single
        mult-gens), and we read the seed coefficients off — `c_0` from
        `()`, `c_i` from `((i, *, 1),)` summed over `*` (ρ-invariance).
        """
        cd = self.cone_data()
        simplified = cd.simplify_trace_via_cone_data(self, label)
        zero = LaurentPoly.zero()
        cs = [zero] * (self.k + 1)
        cs[0] = simplified.terms.get((), zero)
        for k_l in range(1, self.k + 1):
            acc = zero
            for i in range(self.H):
                acc = acc + simplified.terms.get(((k_l, i, 1),), zero)
            cs[k_l] = acc
        return tuple(cs)

    def _legacy_trace_layer1(self, label):
        """Legacy `trace_layer1` via the in-class
        `_tagged_cyclicity_loop` (which already implements the right
        algorithm — kept as an independent test oracle).  Not on the
        live `trace_layer1` path.

        Algorithm: tagged cyclicity.
          1. Convert the canonical-basis label to its L-product `word`.
          2. Tag the front letter `τ` of the word and iterate:
               (a) **cycle** — Form-B cyclicity moves the tagged factor
                   from the front to the back with `ρ^{-2}` applied to
                   it (the other letters stay put);
               (b) **q-commute** — slide the tagged factor past each
                   non-crossing partner from right to left, accumulating
                   q-factors from `L_partner · L_τ = q^c · L_τ · L_partner`;
               (c) **Plücker** — when the tagged factor meets a crossing
                   partner, expand `L_partner · L_τ` via the chord-pair
                   Plücker rule, splice each daughter back into the word,
                   and recurse.
          3. Base cases: empty word → T_0; single letter L((a, *)) → T_a.

        Termination.  Each cycle applies `ρ^{-2}` to the tagged letter,
        so its position cycles through `Z/(2k+3)`; for any rest letter
        in the word, some `ρ^{-2n}`-image of the tag is a crossing
        partner.  Plücker fires after at most `H = 2k+3` cycles.  No
        closed clusters; no insertion trick."""
        K = self.k + 1
        zero = LaurentPoly.zero()
        one = LaurentPoly.one()
        canon = self._rho_canonical_label(label)

        # Base cases on canonical-basis label.
        if canon == ():
            c = [zero] * K
            c[0] = one
            return tuple(c)
        if len(canon) == 1 and canon[0][2] == 1:
            c = [zero] * K
            c[canon[0][0]] = one
            return tuple(c)

        # Convert canon (canonical-basis label) to L-product word.
        #   L_{canon} = 𝖖^{-canon_twist(canon)} · L-product(canon-sorted)
        # so  Tr(L_{canon}) = 𝖖^{-canon_twist} · Tr(L-product(canon-sorted)).
        # Enter the L-product algorithm with that prefactor baked in.
        canon_twist = self._label_canon_twist(canon)
        word: list[tuple[int, int]] = []
        for (k_l, i, e) in canon:
            for _ in range(e):
                word.append((k_l, i))
        return self._tagged_cyclicity_loop(word, -canon_twist, depth=0)

    def _tagged_cyclicity_loop(self, word, q_accum, depth):
        """Run tagged cyclicity on an L-product `word` (list of (k, i)
        letters in any order), with accumulated q-exponent `q_accum`.

        Returns a tuple `(c_0, ..., c_k)` of LaurentPoly coefficients
        such that the trace of `𝖖^{q_accum} · L-product(word)` equals
        `Σ c_i · T_i`."""
        K = self.k + 1
        zero = LaurentPoly.zero()
        # Safety depth limit -- shouldn't trigger if algorithm is correct.
        if depth > 50 + 4 * self.H:
            raise RuntimeError(
                f"_tagged_cyclicity_loop: recursion depth {depth} exceeded "
                f"on word {word}; possible non-terminating Plücker chain."
            )
        # Base cases on word length.
        if not word:
            c = [zero] * K
            c[0] = LaurentPoly.q(q_accum)
            return tuple(c)
        if len(word) == 1:
            k_l = word[0][0]
            c = [zero] * K
            c[k_l] = LaurentPoly.q(q_accum)
            return tuple(c)

        # Cycle + q-commute loop.  word is mutated in place.
        word = list(word)
        n = len(word)
        max_rounds = self.H + 1  # ρ^{-2} has order H since gcd(2, H) = 1.
        for _round in range(max_rounds):
            # CYCLE: pop front (the tag), apply ρ^{-2}, push to back.
            tag = word.pop(0)
            tag_shifted = (tag[0], (tag[1] - 2) % self.H)
            word.append(tag_shifted)

            # Q-COMMUTE: slide tag from position n-1 back toward position 0,
            # checking each adjacent left-partner for crossing.
            for pos in range(n - 1, 0, -1):
                partner = word[pos - 1]
                tagged = word[pos]
                c = self._qcommute_factor(partner, tagged)
                if c is None:
                    # Crossing pair -- PLÜCKER fires here.
                    return self._apply_plucker_in_word(
                        word, pos - 1, pos, q_accum, depth + 1,
                    )
                # q-commute: L_partner · L_tagged = 𝖖^c · L_tagged · L_partner,
                # so Tr(... partner · tagged ...) = 𝖖^c · Tr(... tagged · partner ...).
                q_accum += c
                word[pos - 1], word[pos] = word[pos], word[pos - 1]
            # Tag is back at position 0 with one more ρ^{-2} accumulated;
            # loop and cycle again.
        raise RuntimeError(
            f"_tagged_cyclicity_loop: no Plücker found after H={self.H} "
            f"cycles on word {word}; this shouldn't happen in [A_1, A_{{2k}}]."
        )

    def _apply_plucker_in_word(self, word, idx_a, idx_b, q_accum, depth):
        """Apply Plücker on the crossing pair `(word[idx_a], word[idx_b])`.
        Each daughter canonical-basis term is spliced back into the
        L-product (replacing the crossing pair) and recursed."""
        K = self.k + 1
        zero = LaurentPoly.zero()
        L_a = word[idx_a]
        L_b = word[idx_b]
        # multiply returns a sum of canonical-basis (X-basis) Elements.
        plucker = self.multiply(((L_a[0], L_a[1], 1),),
                                ((L_b[0], L_b[1], 1),))
        prefix = word[:idx_a]
        suffix = word[idx_b + 1:]
        result = [zero] * K
        for daughter_label, daughter_coef in plucker.terms.items():
            # Convert each X-basis daughter to L-product form:
            #   X[daughter] = 𝖖^{-canon_twist(daughter)} · L-product(daughter).
            daughter_twist = self._label_canon_twist(daughter_label)
            daughter_letters: list[tuple[int, int]] = []
            for (dk, di, de) in daughter_label:
                for _ in range(de):
                    daughter_letters.append((dk, di))
            spliced = prefix + daughter_letters + suffix
            # daughter_coef may be a multi-monomial LaurentPoly; iterate.
            for exp, coef_val in daughter_coef._coeffs.items():
                sub_q = q_accum + exp - daughter_twist
                sub_trace = self._tagged_cyclicity_loop(spliced, sub_q, depth)
                for i in range(K):
                    result[i] = result[i] + coef_val * sub_trace[i]
        return tuple(result)

    def _canonicalise_letter_tuple(self, letters):
        """Convenience wrapper around `_canonicalise_letter_tuple_with_twist`
        that returns only the canonical label (drops the twist).  Use the
        `_with_twist` variant when the L-product-to-X-basis correction is
        needed."""
        cf = self._canonicalise_letter_tuple_with_twist(letters)
        if cf is None:
            return None
        return cf[0]

    def _canonicalise_letter_tuple_with_twist(self, letters):
        """Take a tuple of (k, i, e) entries that may be in any order /
        with duplicates and reduce to canonical X-basis form, returning
        (canonical_label, total_twist) where

            L-product(letters in given order) = q^{total_twist} · X[label]

        total_twist combines: (a) the bubble-sort q-twist from re-ordering
        the input word to canonical (k, j) order, and (b) the X-basis
        forward_q_coefficient corrections at the canonical letter pairs.
        Returns None if the letters don't all q-commute."""
        letters = [t for t in letters if t[2] != 0]
        if not letters:
            return ((), 0)
        letter_pairs = list({(t[0], t[1]) for t in letters})
        if not self._letters_qcommute(letter_pairs):
            return None
        word = []
        for (k_l, i, e) in letters:
            for _ in range(e):
                word.append((k_l, i, 1))
        cf = self._canonical_form(word)
        if cf is None:
            return None
        return cf  # (label, twist)

    def _rho_canonical_label(self, label) -> tuple:
        """Return the ρ-orbit representative of `label`: the
        lex-smallest ρ-shift  ρ^k(label)  for k = 0, ..., H-1.  Since Tr
        is ρ-invariant, all ρ-shifts have the same trace."""
        if not label:
            return ()
        best = label
        cur = label
        for _ in range(self.H - 1):
            cur = self.rho(cur)
            if cur < best:
                best = cur
        return best

    # -- Internal: q-commute and X-basis conventions -------------------------

    def _pair_class(self, la: tuple[int, int], lb: tuple[int, int]
                    ) -> tuple[int, int, int, int]:
        """Return (k_a, k_b, d, a_shift) where d = (i_b - i_a) mod H
        and a_shift = i_a (the ρ-shift to apply to base-table outputs
        to lift them back to absolute indices)."""
        ka, ia = la
        kb, ib = lb
        return (ka, kb, (ib - ia) % self.H, ia)

    def _lift_term(self, kind, a_shift: int):
        """Lift a single base-table term `kind` from base ρ-index 0
        to starting ρ-index `a_shift`.  Returns the abstract-letter form:
           ('I',)                                                  -- identity
           ('letter', (k, i))                                      -- single
           ('pair', ((k1, i1), (k2, i2)))                          -- pair
        with absolute indices."""
        if kind == ('I',):
            return ('I',)
        if kind[0] == 'L':
            kx, ix = kind[1]
            return ('letter', (kx, (ix + a_shift) % self.H))
        # 'X'
        ((k1, i1), (k2, i2)) = kind[1]
        return ('pair', ((k1, (i1 + a_shift) % self.H),
                         (k2, (i2 + a_shift) % self.H)))

    def _pair_product(self, la: tuple[int, int], lb: tuple[int, int]):
        """Return the canonical expansion of `L_la · L_lb` as a list of
        `(lifted_term, q_exp)` entries."""
        ka, kb, d, a_shift = self._pair_class(la, lb)
        base = self._base_table[(ka, kb, d)]
        return [(self._lift_term(kind, a_shift), c) for (kind, c) in base]

    def _qcommute_factor(self, la: tuple[int, int],
                         lb: tuple[int, int]) -> int | None:
        """Integer `c` such that `L_la · L_lb = q^c · L_lb · L_la`, or
        `None` if (la, lb) is a Plücker pair (multi-term product)."""
        if la == lb:
            return 0
        ka, kb, d, _ = self._pair_class(la, lb)
        # Normalise to base position
        c = self._qc_cache.get(((ka, 0), (kb, d)))
        return c

    def _forward_q_coeff(self, la: tuple[int, int],
                         lb: tuple[int, int]) -> int:
        """For q-commuting `(la, lb)`, integer `c` such that
        `L_la · L_lb = q^c · X[γ_la + γ_lb]`  (BPS X-basis convention)."""
        if la == lb:
            return 0
        ka, kb, d, _ = self._pair_class(la, lb)
        c = self._fwd_q_cache.get(((ka, 0), (kb, d)))
        if c is None:
            raise ValueError(
                f"_forward_q_coeff: {la} and {lb} are not q-commuting"
            )
        return c

    def _letters_qcommute(self, letters: list[tuple[int, int]]) -> bool:
        """True iff every pair of distinct letters in `letters` q-commutes."""
        for i in range(len(letters)):
            for j in range(i + 1, len(letters)):
                if self._qcommute_factor(letters[i], letters[j]) is None:
                    return False
        return True

    # -- Internal: canonical form + reduction --------------------------------

    def _canonical_form(self, word):
        """Given a word `[(k1, i1, e1), ..., (km, im, em)]` of letters
        with positive exponents, attempt to put it in canonical basis
        form.  Returns `(label, q_exponent)` if successful, else `None`.

        IMPORTANT: must bubble-sort the WORD (not the bucket), because
        same-letter entries may be non-contiguous and the q-twists from
        moving bystanders past them need to be tracked.  Only after the
        word is in canonical lex order do we merge same-letter entries."""
        # First check all letter pairs q-commute (a necessary condition).
        letters_set = list({(k_l, i) for (k_l, i, _e) in word if _e != 0})
        if not self._letters_qcommute(letters_set):
            return None
        # Bubble-sort the WORD (not the bucket) by (k, i) lex order,
        # accumulating q-twist from each adjacent swap.
        arr = [t for t in word if t[2] != 0]
        twist = 0
        n = len(arr)
        # Bubble sort; the q-twist for swapping (k_a, i_a, e_a) past
        # (k_b, i_b, e_b) is q_commute((k_a, i_a), (k_b, i_b)) * e_a * e_b.
        for ii in range(n):
            for jj in range(n - 1 - ii):
                la = (arr[jj][0], arr[jj][1])
                lb = (arr[jj + 1][0], arr[jj + 1][1])
                if la > lb:
                    ea = arr[jj][2]
                    eb = arr[jj + 1][2]
                    c = self._qcommute_factor(la, lb)
                    twist += c * ea * eb
                    arr[jj], arr[jj + 1] = arr[jj + 1], arr[jj]
        # Now merge adjacent same-letter entries.
        merged: list[tuple[int, int, int]] = []
        for (k_l, i, e) in arr:
            if merged and merged[-1][0] == k_l and merged[-1][1] == i:
                merged[-1] = (k_l, i, merged[-1][2] + e)
            else:
                merged.append((k_l, i, e))
        # BPS X-basis convention.
        bps_twist = 0
        for ii in range(len(merged)):
            for jj in range(ii + 1, len(merged)):
                la = (merged[ii][0], merged[ii][1])
                lb = (merged[jj][0], merged[jj][1])
                ea = merged[ii][2]
                eb = merged[jj][2]
                bps_twist += self._forward_q_coeff(la, lb) * ea * eb
        return tuple(merged), twist + bps_twist

    def _term_to_word(self, term):
        """Convert a lifted-term to a list of `(k, i, e)` entries."""
        if term == ('I',):
            return []
        if term[0] == 'letter':
            (k_l, i) = term[1]
            return [(k_l, i, 1)]
        # 'pair'
        ((k1, i1), (k2, i2)) = term[1]
        if (k1, i1) == (k2, i2):
            return [(k1, i1, 2)]
        return [(k1, i1, 1), (k2, i2, 1)]

    def _reduce(self, word):
        """Reduce a word to a canonical-basis sum
        `{label: LaurentPoly}`.  Iterative DFS with Plücker branching.

        Strategy:
          1. Try canonical-form (succeeds iff every letter pair in the
             bucket q-commutes).
          2. Otherwise pick a *target* Plücker pair  (t_a, t_b)  from
             the sorted bucket -- this is DETERMINISTIC across iterations
             so we don't cycle.
          3. Locate t_a, t_b's leftmost positions in the word.  Walk
             t_a rightward toward t_b:
               - If the next bystander q-commutes with t_a, swap it
                 past, picking up the q-twist
                   q^{q_commute_factor(t_a, bystander) * e_a * e_bystander}.
               - If the next bystander is itself a Plücker partner of
                 t_a, apply *that* Plücker first (sub-Plücker recursion).
               - If t_a and t_b become adjacent, apply the target
                 Plücker.

        This makes progress on each iteration (gap-to-target-pair
        decreases by 1 per swap, or a Plücker branches the word into
        smaller pieces)."""
        out: dict[tuple, LaurentPoly] = {}

        stack: list[tuple[LaurentPoly, list[tuple[int, int, int]]]] = [
            (LaurentPoly.one(), [t for t in word if t[2] != 0])
        ]

        while stack:
            coeff, w = stack.pop()
            if coeff.is_zero():
                continue

            cf = self._canonical_form(w)
            if cf is not None:
                label, twist = cf
                adj = coeff * LaurentPoly.q(twist)
                cur = out.get(label, LaurentPoly.zero())
                s = cur + adj
                if s.is_zero():
                    out.pop(label, None)
                else:
                    out[label] = s
                continue

            # Pick a DETERMINISTIC target Plücker pair from the sorted
            # bucket -- consistent across iterations so we don't cycle.
            bucket_letters = sorted({(t[0], t[1]) for t in w})
            target = None
            for i in range(len(bucket_letters)):
                for j in range(i + 1, len(bucket_letters)):
                    if self._qcommute_factor(
                        bucket_letters[i], bucket_letters[j]
                    ) is None:
                        target = (bucket_letters[i], bucket_letters[j])
                        break
                if target:
                    break
            if target is None:
                raise RuntimeError(
                    f"_reduce: bucket has no Plücker pair but "
                    f"canonical_form failed: {w}"
                )
            ta, tb = target
            # Locate t_a, t_b leftmost positions
            pos_ta = next(idx for idx, t in enumerate(w) if (t[0], t[1]) == ta)
            pos_tb = next(idx for idx, t in enumerate(w) if (t[0], t[1]) == tb)
            if pos_ta > pos_tb:
                pos_ta, pos_tb = pos_tb, pos_ta

            if pos_tb == pos_ta + 1:
                # Adjacent target Plücker -- apply.
                pos = pos_ta
                apply_plucker = True
            else:
                # Walk t_a (= w[pos_ta]) rightward by one toward t_b.
                next_letter = (w[pos_ta + 1][0], w[pos_ta + 1][1])
                qc = self._qcommute_factor(
                    (w[pos_ta][0], w[pos_ta][1]), next_letter
                )
                if qc is None:
                    # Sub-Plücker between pos_ta and pos_ta+1 -- apply.
                    pos = pos_ta
                    apply_plucker = True
                else:
                    # Swap w[pos_ta] past w[pos_ta+1]; pick up q-twist.
                    ea = w[pos_ta][2]
                    en = w[pos_ta + 1][2]
                    new_word = (list(w[:pos_ta])
                                + [w[pos_ta + 1], w[pos_ta]]
                                + list(w[pos_ta + 2:]))
                    new_coeff = coeff * LaurentPoly.q(qc * ea * en)
                    stack.append((new_coeff, new_word))
                    continue

            # Apply Plücker on adjacent letters at position `pos`.
            (ka, ia, ea) = w[pos]
            (kb, ib, eb) = w[pos + 1]
            la = (ka, ia); lb = (kb, ib)
            la_pre = list(w[:pos])
            lb_post = list(w[pos + 2:])
            la_extra = [(ka, ia, ea - 1)] if ea > 1 else []
            lb_extra = [(kb, ib, eb - 1)] if eb > 1 else []
            for term, c in self._pair_product(la, lb):
                term_word = self._term_to_word(term)
                c_eff = c
                if term[0] == 'pair':
                    ((k1, i1), (k2, i2)) = term[1]
                    if (k1, i1) != (k2, i2):
                        c_eff -= self._forward_q_coeff((k1, i1), (k2, i2))
                new_word = (la_pre + la_extra + term_word
                            + lb_extra + lb_post)
                new_coeff = coeff * LaurentPoly.q(c_eff)
                stack.append((new_coeff, new_word))

        return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for k in (2, 3):
        print(f"\n{'='*68}\n  A1A2kKAlg(k={k})  ((2k+3) = {2*k+3}-gon)\n{'='*68}")
        A = A1A2kKAlg(k)
        # Identity
        e = A.identity()
        print(f"  identity = {e!r}")
        # L((1, 0)), L((2, 0))
        L1_0 = A.L((1, 0))
        L2_0 = A.L((2, 0))
        print(f"  L((1, 0)) = {L1_0}")
        print(f"  L((2, 0)) = {L2_0}")
        # Products
        prod = A.multiply(L1_0, L1_0)
        print(f"  L((1, 0)) · L((1, 0))  ⇒  {dict(prod.terms)}")
        prod = A.multiply(L1_0, A.L((1, 1)))
        print(f"  L((1, 0)) · L((1, 1))  ⇒  {dict(prod.terms)}")
        prod = A.multiply(L2_0, A.L((2, 1)))
        print(f"  L((2, 0)) · L((2, 1))  ⇒  {dict(prod.terms)}")
        # ρ
        rL1_0 = A.rho(L1_0)
        print(f"  ρ(L((1, 0)))  =  {rL1_0}  (expected L((1, 1)) form)")
        # Identity multiplication
        prod = A.multiply((), L1_0)
        print(f"  1 · L((1, 0))  ⇒  {dict(prod.terms)}")
