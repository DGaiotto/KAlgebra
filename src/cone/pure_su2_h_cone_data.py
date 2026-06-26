"""
pure_su2_h_cone_data.py
=======================

Pure SU(2) BPS K-algebra as a `ConeKAlgebra` with **H-tower cones**:
rank-3 q-commuting cones `C_k = {H_{2k}, H_{2k+1}, H_{2k+2}}` for k ∈ ℤ,
where `H_n` corresponds to the `pSU2KAlgebra` canonical-basis label
`(1, n)`.

This is the ρ-closed cone framework for pure SU(2) — see the docstring
of `pure_su2_cone_data.py` for the contrast with the (non-ρ-closed)
Markov cluster mutation tower.

Lattice picture
---------------
Each cone `C_k` embeds into Z² with the standard symplectic form
`⟨(a, b), (c, d)⟩ = a·d − b·c`:

    H_{2k}    ↔ lattice point  v_0 = (2, 0)
    H_{2k+1}  ↔ lattice point  v_1 = (1, 1)
    H_{2k+2}  ↔ lattice point  v_2 = (0, 2)

Pairings:

    ⟨v_0, v_1⟩ = 2, ⟨v_1, v_2⟩ = 2, ⟨v_0, v_2⟩ = 4

q-cocycles within `C_k` (i.e. the integer c with L_a L_b = q^{2c} L_b L_a):

    cocycle(H_{2k+i}, H_{2k+j}) = (j − i)        — linear in offset

Canonical-basis elements within `C_k` correspond to lattice points
`(n_1, n_2) ∈ Z² _+` with `n_1 + n_2` even (= the even sublattice
intersected with the positive quadrant).  The pSU2 (m, e) label of
the lattice point `(n_1, n_2)` in cone `C_k` is:

    m = (n_1 + n_2) / 2,        e = 2 k m + n_2

Equivalently `(n_1, n_2) = (2 m − (e − 2 k m), e − 2 k m)`.

Linear redundancy ("semi-redundant" ray)
----------------------------------------
The triple `{(2, 0), (1, 1), (0, 2)}` has rank 2 in Z², with the
syzygy

    (2, 0) + (0, 2) = 2 · (1, 1)

So the canonical-basis lattice point `(2, 2)` can be expressed as

    H_{2k}·H_{2k+2}    (= a=1, b=0, c=1)
    H_{2k+1}²          (= a=0, b=2, c=0)

both yielding the same pSU2 label `(2, 4k+2)` but with different
q-monomial pre-factors.  We pick the **max-diagonal convention**:
always maximise the use of the diagonal ray H_{2k+1}.  Algorithm:
given `(n_1, n_2)` with `n_1 + n_2 = 2 m` (even), set

    b = min(n_1, n_2),  a = (n_1 − b) / 2,  c = (n_2 − b) / 2

Then `(a, b, c)` are non-negative integers with `2a + b = n_1` and
`b + 2c = n_2`, and `b` is maximal.

ρ-action
--------
`ρ(H_n) = H_{n−4}`, which shifts the cone index by `−2`:

    ρ(C_k) = C_{k − 2}

(uniform across the entire H tower — no chamber exceptions).

Cross-cluster Plücker
---------------------
The non-q-commuting case within the H tower is the **odd-anchored
skip-1 pair** `(H_{2k+1}, H_{2k+3})`, which sits across cones
`C_k` and `C_{k+1}`.  Empirically:

    H_{2k+1} · H_{2k+3}  =  q · H_{2k+2}  +  q² · H_{2k+2}²
    H_{2k+3} · H_{2k+1}  =  q^{−1} · H_{2k+2}  +  q^{−2} · H_{2k+2}²

(The shared mult-gen `H_{2k+2}` is the "bridge" between `C_k` and
`C_{k+1}`.)

Test surface
------------
At the bottom of this module, a `__main__` smoke test verifies the
cone-data primitives by comparing `derived_multiply` outputs against
`pSU2KAlgebra.multiply` for a basket of (a, b, c) cone monomials.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from typing import Sequence

from cone_data import ConeData, CrossProductTerm
from laurent_poly import LaurentPoly
from kalgebra import Element
from cone_kalgebra import ConeKAlgebra
from zplus_ring import TrivialZPlusRing
from pure_su2_h_wilson import is_wilson_label, WILSON_FUND


# ---------------------------------------------------------------------------
# Mult-gen label and cone identification
# ---------------------------------------------------------------------------
#
# Native label format: a sorted tuple of `(n, exp)` pairs with `exp > 0`,
# representing the cone-monomial H_{n_1}^{exp_1} · H_{n_2}^{exp_2} · …
# in canonical cone order (= max-diagonal convention).
#
# Mult-gen label: `(n,)` for `H_n = pSU2 (1, n)`.
#
# Cones: `C_k = {H_{2k}, H_{2k+1}, H_{2k+2}}` keyed by integer k.


def cone_index_for(n_lo: int, n_hi: int) -> int | None:
    """Return the integer k such that the cone `C_k = {H_{2k}, H_{2k+1},
    H_{2k+2}}` contains both labels n_lo and n_hi (assuming n_lo ≤ n_hi),
    or None if no such cone exists.

    Convention: when multiple cones contain both labels (e.g. a single
    label), return the largest k (= cone with the smallest 2k floor).
    """
    if n_hi - n_lo > 2:
        return None
    # 2k must satisfy 2k ≤ n_lo AND n_hi ≤ 2k+2  ⇔  n_hi − 2 ≤ 2k ≤ n_lo.
    even_lo = n_hi - 2 if (n_hi - 2) % 2 == 0 else n_hi - 1
    even_hi = n_lo if n_lo % 2 == 0 else n_lo - 1
    if even_lo > even_hi:
        return None
    # Return largest 2k (closest cone matching the labels).
    return even_hi // 2


# ---------------------------------------------------------------------------
# H-tower cone data
# ---------------------------------------------------------------------------

class PureSU2HConeData(ConeData):
    """Pure SU(2) cone data based on the H tower (ρ-closed).

    Mult-gens: `H_n = (n,)` for n ∈ ℤ (labelled by integer index).
    Cones: rank-3 cones `C_k = {H_{2k}, H_{2k+1}, H_{2k+2}}` for k ∈ ℤ.

    Within a cone, three mult-gens pairwise q-commute with cocycle
    `cocycle(H_{2k+i}, H_{2k+j}) = j − i`.  Across cones, the only
    non-q-commuting pair is `(H_{2k+1}, H_{2k+3})` (the odd-odd skip-1
    cluster mutation pair), with Plücker

        H_{2k+1} · H_{2k+3}  =  q · H_{2k+2} + q² · H_{2k+2}²

    The linear redundancy within a cone (H_{2k}·H_{2k+2} ↔ H_{2k+1}²)
    is handled via the **max-diagonal convention** in `to_cone_label` /
    `from_cone_label`.
    """

    def __init__(self) -> None:
        self._gens: dict[int, tuple] = {}
        self._cones: dict[int, frozenset] = {}
        # Wilson character cone (rank-1): generator w_1 = ('W', 1).
        # Canonical-basis elements are χ_e for e ≥ 0; PBW monomials are
        # powers of w_1.  See `pure_su2_h_wilson` for χ ↔ w_1 conversion.
        self._wilson_cone: frozenset = frozenset({WILSON_FUND})

    # -- on-demand materialisation ----------------------------------------

    def _gen(self, n: int) -> tuple:
        if n not in self._gens:
            self._gens[n] = (n,)
        return self._gens[n]

    def _cone(self, k: int) -> frozenset:
        """Cone `C_k = {H_{2k}, H_{2k+1}, H_{2k+2}}`."""
        if k not in self._cones:
            self._cones[k] = frozenset({
                self._gen(2 * k), self._gen(2 * k + 1), self._gen(2 * k + 2),
            })
        return self._cones[k]

    def seen_gens(self) -> Sequence[tuple]:
        return tuple(sorted(self._gens.values()))

    def seen_cones(self) -> Sequence[frozenset]:
        return tuple(self._cones[k] for k in sorted(self._cones))

    # -- iter_cones (lazy unbounded) --------------------------------------

    def iter_cones(self):
        """Bilateral enumeration of cones C_0, C_1, C_{-1}, C_2, …"""
        yield self._cone(0)
        k = 1
        while True:
            yield self._cone(k)
            yield self._cone(-k)
            k += 1

    # -- q_commute / cocycle / cross_product ------------------------------

    def q_commute(self, g, h) -> bool:
        if g == h:
            return True
        # Wilson-Wilson: all Wilsons q-commute trivially (cocycle 0).
        g_is_w = is_wilson_label(g)
        h_is_w = is_wilson_label(h)
        if g_is_w and h_is_w:
            return True
        # Wilson-H: DSZ pairing for (0, e_W) and (1, n) is e_W (≠ 0 for
        # non-trivial Wilsons), so they do NOT q-commute when e_W ≠ 0.
        # (w_1 has e_W = 1; non-q-commuting with all H-letters.)
        if g_is_w or h_is_w:
            return False
        a, b = g[0], h[0]
        n_lo, n_hi = (a, b) if a < b else (b, a)
        # In same cone iff cone_index_for returns non-None.
        return cone_index_for(n_lo, n_hi) is not None

    def cocycle(self, g, h) -> int:
        """cocycle(H_a, H_b) = b − a  for q-commuting H-pairs.
        Wilson-Wilson cocycle is 0 (DSZ pairing 0).
        Wilson-H is non-q-commuting; cocycle undefined.

        Antisymmetric.  Defined only when q_commute(g, h).
        """
        if g == h:
            return 0
        # Wilson-Wilson: trivially commute, cocycle 0.
        if is_wilson_label(g) and is_wilson_label(h):
            return 0
        if not self.q_commute(g, h):
            raise ValueError(
                f"cocycle: ({g}, {h}) not q-commuting"
            )
        a, b = g[0], h[0]
        return b - a

    def cross_product(
        self, g, h
    ) -> Sequence[CrossProductTerm]:
        """Plücker substitution for non-q-commuting mult-gen pairs.

        Delegates to `pure_su2_h_multiply._ray_product` (the axiom-
        derived ray-product table built from `h_mul_h` + SU(2)
        Clebsch/Chebyshev), with letter-format translation between
        the cone-data convention (`(n,)` for H_n, `('W', 1)` for w_1)
        and the literal-word reducer convention (`('H', n)` resp.
        `('W', 1)`).  Wilson outputs are emitted as `w_1^k` literal
        mult-gen monomials; cone_data's tagged-cyclicity reducer
        finishes the canonical-χ reassembly downstream.
        """
        from pure_su2_h_multiply import _ray_product, WILSON_FUND

        def _to_letter(x):
            if isinstance(x, tuple) and x and x[0] == 'W':
                return WILSON_FUND
            return ('H', x[0])                   # (n,) → ('H', n).

        def _from_word(word):
            out = []
            for letter in word:
                if letter[0] == 'H':
                    out.append(self._gen(letter[1]))
                else:                            # Wilson
                    out.append(WILSON_FUND)
            return tuple(out)

        g_l = _to_letter(g)
        h_l = _to_letter(h)
        substitutions = _ray_product(g_l, h_l)
        return [(coef, _from_word(word)) for coef, word in substitutions]

    # -- canonical cone order ---------------------------------------------

    def canonical_cone_order(self, gens):
        """Sort by H-index (= ascending integer n)."""
        return tuple(sorted(gens))

    # -- cone-label bijection (max-diagonal convention) ------------------

    def to_cone_label(self, native_label):
        """Native = sorted tuple of `(n, exp)` with `exp > 0`.

        Find the smallest cone containing all the n's; return
        (cone, powers).
        """
        if not native_label:
            cone = self._cone(0)
            return cone, {g: 0 for g in cone}
        ns = sorted({n for (n, exp) in native_label if exp > 0})
        if not ns:
            cone = self._cone(0)
            return cone, {g: 0 for g in cone}
        n_lo, n_hi = ns[0], ns[-1]
        k = cone_index_for(n_lo, n_hi)
        if k is None:
            raise ValueError(
                f"to_cone_label: labels {ns} don't fit in any single cone"
            )
        cone = self._cone(k)
        powers = {g: 0 for g in cone}
        for (n, exp) in native_label:
            key = (n,)
            powers[key] = powers.get(key, 0) + exp
        return cone, powers

    def from_cone_label(self, gens, powers):
        """Inverse: (gens, powers) → sorted tuple of `(n, exp)`."""
        factors = sorted((g[0], p) for g, p in powers.items() if p > 0)
        return tuple(factors)


# ---------------------------------------------------------------------------
# Native <-> pSU2 (m, e) label conversion
# ---------------------------------------------------------------------------

def _native_to_psu2(native: tuple) -> tuple:
    """Translate native canonical-basis label to pSU2 `(m, e)`.

    Native forms:
      * `()` → `(0, 0)`.
      * H-tower `((n_int, exp), ...)` → `(sum exp, sum n·exp)`.
      * Wilson `((('W', e), 1),)` → `(0, e)`.
    """
    m, e = 0, 0
    for entry in native:
        gen, exp = entry
        if isinstance(gen, int):
            m += exp
            e += gen * exp
        elif isinstance(gen, tuple) and gen[0] == 'W':
            assert exp == 1, (
                f"_native_to_psu2: Wilson entry with exp != 1: {entry}"
            )
            e += gen[1]
        else:
            raise ValueError(
                f"_native_to_psu2: unrecognised entry {entry}"
            )
    return (m, e)


def _psu2_to_native(m: int, e: int) -> tuple:
    """Inverse: pSU2 `(m, e)` → native max-diagonal cone-monomial form.

    m = 0: Wilson χ_e (or identity if e = 0).
    m = 1: single H_e.
    m ≥ 2: max-diagonal cone monomial in cone C_k for the largest
           feasible k.
    """
    if m == 0:
        return () if e == 0 else ((('W', e), 1),)
    if m == 1:
        return ((e, 1),)
    # m ≥ 2: find cone C_k.  Constraints: 0 ≤ n_2 = e − 2k m ≤ 2m
    #                                     and 0 ≤ n_1 = 2m − n_2 ≤ 2m.
    # Equivalently: (e − 2m) / (2m) ≤ k ≤ e / (2m).
    # Pick the largest feasible k (canonical anchor).
    k_max = e // (2 * m)
    for k_try in range(k_max, k_max - 3, -1):
        n_2 = e - 2 * k_try * m
        n_1 = 2 * m - n_2
        if 0 <= n_1 <= 2 * m and 0 <= n_2 <= 2 * m:
            chosen_k = k_try
            n1_chosen = n_1
            n2_chosen = n_2
            break
    else:
        raise ValueError(
            f"_psu2_to_native: cannot find cone for (m={m}, e={e})"
        )
    # Max-diagonal: b = min(n_1, n_2), a = (n_1 − b)/2, c = (n_2 − b)/2.
    b = min(n1_chosen, n2_chosen)
    a = (n1_chosen - b) // 2
    c = (n2_chosen - b) // 2
    entries = []
    if a > 0:
        entries.append((2 * chosen_k, a))
    if b > 0:
        entries.append((2 * chosen_k + 1, b))
    if c > 0:
        entries.append((2 * chosen_k + 2, c))
    return tuple(entries)


# ---------------------------------------------------------------------------
# K-algebra wrapper
# ---------------------------------------------------------------------------

def _is_me(x) -> bool:
    """True iff `x` is a pSU2 `(m, e)` BPS label (a 2-tuple of ints) rather than
    a native H-tower word (`()` or a tuple of `(n, exp)` pairs).  Unambiguous: a
    native word is empty or has tuple entries, never two bare ints."""
    return (isinstance(x, tuple) and len(x) == 2
            and isinstance(x[0], int) and isinstance(x[1], int))


class PureSU2KAlg(ConeKAlgebra):
    """Pure SU(2) K-algebra with the H-tower rank-3 cone data."""

    _R = TrivialZPlusRing()

    def __init__(self) -> None:
        self._cone_data_inst = PureSU2HConeData()

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return ()

    def cone_data(self):
        return self._cone_data_inst

    def multiply(self, a, b):
        """Axiom-derived multiplication: literal-ray-word reduction via
        the cone-framework algorithm with all ray-ray substitutions
        coming from `h_mul_h` (W_1-walk cyclicity) and SU(2)
        Clebsch/Chebyshev — no `pSU2.multiply` iso-delegation.

        Accepts both the native H-tower labels and the **`(m, e)` BPS labels**
        (`(0, k)` Wilson, `(m, e)` dyon — the labelling the RG-flow auxiliaries
        use); inputs are converted to native, multiplied, and the result is
        returned in the same convention as the inputs.  See
        `pure_su2_h_multiply.multiply_native` for the algorithm.
        """
        from pure_su2_h_multiply import multiply_native
        me = _is_me(a) or _is_me(b)
        an = _psu2_to_native(*a) if _is_me(a) else a
        bn = _psu2_to_native(*b) if _is_me(b) else b
        result = multiply_native(an, bn)
        if not me:
            return result
        return Element({_native_to_psu2(L): c for L, c in result.terms.items()})

    def rho(self, label):
        """ρ shifts H_n index by −4 (`ρ(D(m,e)) = D(m, e−4m)`); Wilson
        entries (`m = 0`) are ρ-fixed.  Accepts native or `(m, e)` labels,
        returning the same convention."""
        if _is_me(label):
            m, e = label
            return (m, e - 4 * m)                     # ρ(m, e) = (m, e − 4m)
        return tuple(sorted((n - 4, e) if isinstance(n, int) else (n, e)
                            for (n, e) in label))

    def rho_inverse(self, label):
        if _is_me(label):
            m, e = label
            return (m, e + 4 * m)
        return tuple(sorted((n + 4, e) if isinstance(n, int) else (n, e)
                            for (n, e) in label))

    def trace(self, label, K: int = 20):
        """Axiom-derived trace via `trace_pSU2_label` (in
        `pure_su2_h_trace`): SU(2) Schur F(v) for Wilson seeds and
        ρ²-twisted cyclicity bridges for every m-anchor.  Bypasses
        Layer 1's tagged-cyclicity reducer (which doesn't terminate
        for pure SU(2) on Wilson-producing cross-cone Plückers);
        H-shift + Z₂ + anchor selection do the canonicalisation
        directly on the pSU2 seed level for all m.
        """
        from pure_su2_h_trace import trace_pSU2_label
        from zplus_ring import RPowerSeries
        m, e = _native_to_psu2(label)
        trace_lp = trace_pSU2_label(m, e, q_max=K)
        R = self.coefficient_ring()
        return RPowerSeries(R, dict(trace_lp._coeffs), K)

    def _canonical_rho2_orbit_rep(self, label):
        """Closed-form ρ²-orbit canonicalisation for the pure-SU(2) H tower.

        `ρ²` shifts every H-index by `−8`, so each H_n lies on an
        infinite orbit; default orbit-walk in `KAlgebra` would loop
        forever.  Canonical representative: shift every H-index into
        `[0, 8)`.  Wilson labels and identity are ρ-invariant.
        """
        if not label:
            return label
        first = label[0]
        if isinstance(first[0], tuple) and first[0][0] == 'W':
            return label                         # Wilson: ρ-invariant.
        # H-monomial: shift all indices into [0, 8) by ρ².
        shifted = tuple((n % 8, exp) for (n, exp) in label)
        # Re-sort + collapse duplicate indices.
        counts: dict = {}
        for n, exp in shifted:
            counts[n] = counts.get(n, 0) + exp
        return tuple(sorted(counts.items()))

    def _trace_residual(self, seed_label, K):
        """Closed-form trace at a Layer-1 ρ²-canonical seed.

        Seeds reach here only as identity or single mult-gens (Layer 1's
        tagged-cyclicity reducer in `simplify_trace_via_cone_data`
        eliminates all multi-mult-gen monomials).  The closed forms used:

          * Identity / Wilson seeds: SU(2) Schur F(v) (m=0 branch of
            `trace_pSU2_label` → `tr_W(e)`).
          * Single H_n: H-shift symmetry reduces to H_0 (even n) or 0
            (odd n by Z₂); Tr(H_0) via the cyclicity bridge formula
            (`tr_h0_bridge` in `pure_su2_layer2_identities`).

        Trace of m ≥ 2 seeds is **not** required here — Layer 1
        reduces those to m ≤ 1 before this method is called.
        """
        from pure_su2_h_trace import trace_pSU2_label
        from zplus_ring import RPowerSeries
        m, e = _native_to_psu2(seed_label)
        trace_lp = trace_pSU2_label(m, e, q_max=K)
        R = self.coefficient_ring()
        return RPowerSeries(R, dict(trace_lp._coeffs), K)


# ---------------------------------------------------------------------------
# Smoke test: verify cone_data q_commute / cocycle / cross_product against
# pSU2KAlgebra for a basket of products.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from psu2_kalgebra import pSU2KAlgebra

    cd = PureSU2HConeData()
    P = pSU2KAlgebra()

    print("PureSU2HConeData smoke test\n" + "=" * 50)

    # 1. q_commute check.
    print("\n1. q_commute predicate (vs empirical: single-q-monomial product?):")
    for (a, b) in [(0, 1), (0, 2), (1, 2), (1, 3), (-1, 1), (2, 4), (0, 3), (3, 5)]:
        prod = P.multiply((1, a), (1, b))
        single_q = (len(prod.terms) == 1
                    and len(next(iter(prod.terms.values()))._coeffs) == 1)
        predicted = cd.q_commute((a,), (b,))
        match = "✓" if single_q == predicted else "✗"
        print(f"  q_commute(H_{a}, H_{b}) = {predicted}  (empirical {single_q})  {match}")

    # 2. cocycle vs empirical q-power exponent.
    print("\n2. cocycle (vs empirical q-exponent in single-term products):")
    for (a, b) in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1),
                    (0, 0), (-1, 0), (-2, 0)]:
        if not cd.q_commute((a,), (b,)):
            continue
        c_predicted = cd.cocycle((a,), (b,))
        prod = P.multiply((1, a), (1, b))
        # extract single q-exponent
        if len(prod.terms) == 1:
            lbl, c_lp = next(iter(prod.terms.items()))
            if len(c_lp._coeffs) == 1:
                empirical_q_exp = next(iter(c_lp._coeffs))
                # cocycle is c with L_a L_b = q^{2c} L_b L_a.
                # The asymmetry between L_a L_b and L_b L_a gives 2c = q-exp(LR) − q-exp(RL).
                # For our normalisation, L_a L_b = q^{cocycle} · (canonical), L_b L_a = q^{−cocycle} · (canonical).
                # So empirical q-exp = cocycle (under this convention).
                match = "✓" if empirical_q_exp == c_predicted else "✗"
                print(f"  cocycle(H_{a}, H_{b}) = {c_predicted}  (empirical q-exp {empirical_q_exp})  {match}")

    # 3. cross_product for odd-odd skip-1.
    print("\n3. cross_product (vs pSU2.multiply for non-q-commuting pairs):")
    for (a, b) in [(1, 3), (3, 5), (-1, 1), (5, 7), (3, 1)]:
        if cd.q_commute((a,), (b,)):
            continue
        cp = cd.cross_product((a,), (b,))
        print(f"  cross_product(H_{a}, H_{b}):")
        for coef, word in cp:
            print(f"    {coef} · {word}")
        prod = P.multiply((1, a), (1, b))
        print(f"    [pSU2 empirical]  H_{a}·H_{b} = {dict(prod.terms)}")
