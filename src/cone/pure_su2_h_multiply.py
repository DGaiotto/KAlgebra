"""Axiom-derived multiplication for `PureSU2KAlg`.

**Literal-word reduction**: products are computed by maintaining the
state as a polynomial in `(q-coef, literal-ray-word)` pairs and
reducing until each word is a canonical sorted-cone monomial.  All
ray-ray products come from axiom-derived kernels:

  * `pure_su2_h_gap_k.h_mul_h(a, b)` (a ≤ b) — H_a · H_b via the
    `w_1 · H_a · H_b` cyclicity recursion (user's hint:
    *"w_1 H_a H_b associativity reconstructs all gaps recursively"*).
  * Direct SU(2) Clebsch for w_1 ↔ H_n and w_1 × w_1.
  * Chebyshev (`chi_to_w1_powers`, `w1_power_to_chi`) for converting
    between Wilson canonical χ_e and literal w_1^k.

No iso-delegation to `pSU2KAlgebra.multiply`.  All structure constants
come from the cyclicity / Clebsch / Chebyshev primitives, composed
purely by associativity using the cone framework's sort-and-substitute
algorithm.

Letter convention
-----------------
A "letter" is one of

  ``('H', n)``  — the H-cone mult-gen H_n  (n ∈ ℤ)
  ``WILSON_FUND = ('W', 1)``  — the Wilson fundamental ray w_1

A "literal word" is a tuple of letters; it represents the literal PBW
monomial in mult-gens.  Canonical-basis ↔ literal conversion happens
only at the entry (`_expand_native_to_literal`) and exit
(`_word_to_native_poly`) boundaries.
"""
from __future__ import annotations
from laurent_poly import LaurentPoly
from kalgebra import Element
from pure_su2_h_gap_k import h_mul_h, _seed_to_h_pair
from pure_su2_h_wilson import (
    chi_to_w1_powers, w1_power_to_chi, WILSON_FUND, is_wilson_label,
)


LP = LaurentPoly


# -------------------------------------------------------------------
# Cone identification (duplicated from cone_data to keep this module
# self-contained on its hot path).
# -------------------------------------------------------------------

def _cone_idx_for(a: int, b: int):
    """Integer k such that C_k = {2k, 2k+1, 2k+2} contains both H-indices
    a ≤ b, or None if no single cone contains both.

    Need `2k ≤ a` AND `2k + 2 ≥ b`.  Returns the largest feasible k."""
    if b - a > 2:
        return None
    even_hi = a if a % 2 == 0 else a - 1
    if even_hi < b - 2:
        return None
    return even_hi // 2


def _letter_kind(l: tuple) -> str:
    """'H' or 'W' for a letter."""
    return l[0]


def _is_H(l: tuple) -> bool:
    return l[0] == 'H'


def _is_W(l: tuple) -> bool:
    return l[0] == 'W'


def _q_commute(g: tuple, h: tuple) -> bool:
    """Two letters q-commute iff in the same cone."""
    if g == h:
        return True
    if _is_W(g) and _is_W(h):
        return True                              # Wilson-Wilson: cocycle 0.
    if _is_W(g) or _is_W(h):
        return False                             # Wilson-H: don't q-commute.
    a, b = g[1], h[1]
    lo, hi = (a, b) if a <= b else (b, a)
    return _cone_idx_for(lo, hi) is not None


def _cocycle(g: tuple, h: tuple) -> int:
    """Cocycle c with `g · h = q^{2c} · h · g` (= b − a for H_a, H_b)."""
    if g == h:
        return 0
    if _is_W(g) and _is_W(h):
        return 0
    if _is_W(g) or _is_W(h):
        raise ValueError(f"_cocycle: ({g}, {h}) non-q-commuting")
    return h[1] - g[1]


# -------------------------------------------------------------------
# Ray-product substitutions (the "Plücker table"):
#   substitution(g, h) → list of (LP coef, replacement_word).
# -------------------------------------------------------------------

_RAY_PRODUCT_CACHE: dict = {}


def _ray_product(g: tuple, h: tuple) -> list:
    """Substitute the literal pair `g · h` (when non-q-commuting) by a
    polynomial of literal-word replacements.

    Cases:
      * H × H: h_mul_h, then expand canonical seeds to literal words.
      * H × w_1: q^{-1}·H_{n−1} + q·H_{n+1} + [n odd: identity].
      * w_1 × H: q·H_{n−1} + q^{-1}·H_{n+1} + [n odd: identity].
      * w_1 × w_1: q-commutes; this function is not called for it.
    """
    key = (g, h)
    if key in _RAY_PRODUCT_CACHE:
        return _RAY_PRODUCT_CACHE[key]
    result = _ray_product_impl(g, h)
    _RAY_PRODUCT_CACHE[key] = result
    return result


def _ray_product_impl(g: tuple, h: tuple) -> list:
    if _is_H(g) and _is_H(h):
        return _hh_product(g[1], h[1])
    if _is_H(g) and _is_W(h):
        return _Hw_product(g[1])
    if _is_W(g) and _is_H(h):
        return _wH_product(h[1])
    raise ValueError(f"_ray_product: unexpected ({g}, {h})")


def _hh_product(a: int, b: int) -> list:
    """Literal `H_a · H_b` substitution from `h_mul_h`, expanded into
    literal-word replacements (no canonical-seed intermediates).

    `h_mul_h` is total in (a, b): a <= b runs the gap recursion, a > b is
    the bar-conjugate of (b, a) -- handled inside `h_mul_h`, so this wrapper
    needs no ordering case."""
    return _canonical_dict_to_literal_replacements(h_mul_h(a, b))


def _canonical_dict_to_literal_replacements(canonical: dict) -> list:
    """Convert `{(m, e): LP coef}` (canonical-basis polynomial) into a
    list of `(LP coef, literal word)` pairs.

    `(0, 0)` → empty word.
    `(0, e ≥ 1)` → χ_e expanded via `chi_to_w1_powers` (Chebyshev) to
                   multiple w_1^j literal monomials.
    `(1, n)`  → single `('H', n)` letter.
    `(2, e)`  → literal pair `(('H', α), ('H', β))` with phase factor.
    """
    out: list = []
    for seed, coef_lp in canonical.items():
        m, e = seed
        if coef_lp.is_zero():
            continue
        if m == 0 and e == 0:
            out.append((coef_lp, ()))
        elif m == 0:
            # χ_e canonical = Σ_j chi_to_w1_powers[j] · w_1^j literal.
            chi_w1 = chi_to_w1_powers(e)
            for j, scalar in chi_w1.items():
                if scalar == 0:
                    continue
                word = (WILSON_FUND,) * j
                term_coef = coef_lp * LP({0: scalar})
                if not term_coef.is_zero():
                    out.append((term_coef, word))
        elif m == 1:
            out.append((coef_lp, (('H', e),)))
        elif m == 2:
            alpha, beta = _seed_to_h_pair(seed)
            # L_{(2, e)} = q^{-(β − α)} · literal pair, so literal pair
            # = q^{β − α} · L_{(2, e)}.  Substituting canonical seed
            # by literal pair: coef × q^{−(β − α)} multiplier.
            phase_shift = -(beta - alpha) if alpha != beta else 0
            word = (('H', alpha), ('H', beta))
            term_coef = coef_lp * LP({phase_shift: 1})
            if not term_coef.is_zero():
                out.append((term_coef, word))
        else:
            # Invariant: h_mul_h outputs canonical seeds with m <= 2
            # (H*H has magnetic charge = #H-factors <= 2); m>2 cannot occur.
            raise AssertionError(
                f"_canonical_dict_to_literal_replacements: invariant m<=2 "
                f"violated (m={m}); h_mul_h seeds carry m = #H-factors <= 2")
    return out


def _Hw_product(n: int) -> list:
    """`H_n · w_1 = q^{-1}·H_{n−1} + q·H_{n+1} + [n odd: identity]`."""
    out = [
        (LP({-1: 1}), (('H', n - 1),)),
        (LP({1: 1}),  (('H', n + 1),)),
    ]
    if n % 2 == 1:
        out.append((LP({0: 1}), ()))
    return out


def _wH_product(n: int) -> list:
    """`w_1 · H_n = q·H_{n−1} + q^{-1}·H_{n+1} + [n odd: identity]`."""
    out = [
        (LP({1: 1}),  (('H', n - 1),)),
        (LP({-1: 1}), (('H', n + 1),)),
    ]
    if n % 2 == 1:
        out.append((LP({0: 1}), ()))
    return out


# -------------------------------------------------------------------
# Native ↔ literal conversion
# -------------------------------------------------------------------

def _native_to_psu2(native: tuple) -> tuple:
    m, e = 0, 0
    for entry in native:
        gen, exp = entry
        if isinstance(gen, int):
            m += exp
            e += gen * exp
        elif isinstance(gen, tuple) and gen[0] == 'W':
            assert exp == 1
            e += gen[1]
        else:
            raise ValueError(f"_native_to_psu2: unrecognised entry {entry}")
    return (m, e)


def _psu2_to_native(m: int, e: int) -> tuple:
    if m == 0:
        return () if e == 0 else ((('W', e), 1),)
    if m == 1:
        return ((e, 1),)
    # m ≥ 2: max-diagonal H-monomial.
    k_max = e // (2 * m)
    chosen_k = None
    for k_try in range(k_max, k_max - 3, -1):
        n2 = e - 2 * k_try * m
        n1 = 2 * m - n2
        if 0 <= n1 <= 2 * m and 0 <= n2 <= 2 * m:
            chosen_k = k_try
            break
    if chosen_k is None:
        raise ValueError(f"_psu2_to_native: cannot find cone for ({m}, {e})")
    n2 = e - 2 * chosen_k * m
    n1 = 2 * m - n2
    b = min(n1, n2)
    a = (n1 - b) // 2
    c = (n2 - b) // 2
    entries = []
    if a > 0:
        entries.append((2 * chosen_k, a))
    if b > 0:
        entries.append((2 * chosen_k + 1, b))
    if c > 0:
        entries.append((2 * chosen_k + 2, c))
    return tuple(entries)


def _expand_native_to_literal(native: tuple) -> list:
    """Native canonical-basis label → list of `(LP coef, literal word)`.

    H-monomial native: a single literal term with the H-letters in
    ascending order; the q-coef is `q^{phase}` where `phase = Σ_{i<j}
    (c_j − c_i)` is the literal-to-canonical conversion phase (since
    canonical = q^{−phase} · literal, expanding canonical = q^{−phase}
    · literal as `q^{−phase}` times the literal word reproduces the
    canonical native at the literal-word level).

    Wilson native `((('W', e), 1),)`: χ_e expanded via Chebyshev
    (`chi_to_w1_powers`) into multiple literal-word terms `w_1^j`.

    Empty native: identity = single empty literal word.
    """
    if not native:
        return [(LP({0: 1}), ())]
    first = native[0]
    if isinstance(first[0], tuple) and first[0][0] == 'W':
        e = first[0][1]
        if e == 0:
            return [(LP({0: 1}), ())]
        chi_w1 = chi_to_w1_powers(e)
        return [
            (LP({0: scalar}), (WILSON_FUND,) * j)
            for j, scalar in chi_w1.items() if scalar != 0
        ]
    # H-monomial.
    chain: list = []
    for entry in native:
        n, exp = entry
        chain += [n] * exp
    # Sort ascending (should already be sorted).
    chain.sort()
    phase = 0
    for i in range(len(chain)):
        for j in range(i + 1, len(chain)):
            phase += chain[j] - chain[i]
    word = tuple(('H', n) for n in chain)
    return [(LP({-phase: 1}), word)]


def _word_to_native_poly(word: tuple) -> dict:
    """Sorted canonical-cone literal word → `{native: LP coef}` (polynomial).

    Word must be a single-cone literal monomial in canonical sorted order:
      * pure H-letters in ascending order, all in one H-cone, OR
      * pure w_1 letters (the Wilson cone has rank 1).

    Empty word → identity native.
    Pure H-monomial → single native with `q^{phase}` (canonical =
                       q^{−phase} · literal ⇒ literal = q^{phase} ·
                       canonical, so the native gets `q^{phase}`).
    Pure w_1^k → canonical χ_j polynomial via `w1_power_to_chi`
                  (multiple native terms).
    """
    out: dict = {}
    if not word:
        out[()] = LP({0: 1})
        return out
    if all(_is_W(l) for l in word):
        k = len(word)
        chi_decomp = w1_power_to_chi(k)
        for e, mult in chi_decomp.items():
            if mult == 0:
                continue
            native = () if e == 0 else ((('W', e), 1),)
            existing = out.get(native, LP.zero())
            out[native] = existing + LP({0: mult})
        return {k: v for k, v in out.items() if not v.is_zero()}
    if all(_is_H(l) for l in word):
        chain = [l[1] for l in word]
        if any(chain[i] > chain[i + 1] for i in range(len(chain) - 1)):
            raise ValueError(
                f"_word_to_native_poly: H-word not sorted ascending: {word}"
            )
        # Canonical-vs-literal phase: literal_chain = q^{phase} · L_{(m, e)}.
        phase = 0
        for i in range(len(chain)):
            for j in range(i + 1, len(chain)):
                phase += chain[j] - chain[i]
        # Canonicalise to max-diagonal native via pSU2 (m, e).  The
        # literal sorted-cone chain may not be max-diagonal (e.g.
        # `H_4·H_6` for `(2, 10)` whose canonical is `H_5²`); we emit
        # the max-diagonal native unambiguously.
        m = len(chain)
        e = sum(chain)
        native = _psu2_to_native(m, e)
        out[native] = LP({phase: 1})
        return out
    raise ValueError(
        f"_word_to_native_poly: mixed H+Wilson word should have been "
        f"reduced before reaching here: {word}"
    )


# -------------------------------------------------------------------
# Word reducer
# -------------------------------------------------------------------

def _word_is_canonical(word: tuple) -> bool:
    """True iff `word` is a sorted single-cone monomial (and can be
    converted directly to canonical native)."""
    if not word:
        return True
    if all(_is_W(l) for l in word):
        return True
    if all(_is_H(l) for l in word):
        chain = [l[1] for l in word]
        if any(chain[i] > chain[i + 1] for i in range(len(chain) - 1)):
            return False
        # All in single cone?
        return _cone_idx_for(min(chain), max(chain)) is not None
    return False


def _find_action(word: tuple):
    """Return an `(idx, kind, *args)` action to apply, or None if `word`
    is already canonical.

    Strategy: prefer non-q-commuting adjacency (=> cross_product).
    Otherwise, find an out-of-order q-commuting adjacency (=> swap).
    Otherwise, the word may still have non-adjacent non-q-commuting
    pairs separated by q-commuting partners — bubble to expose.
    """
    if _word_is_canonical(word):
        return None
    # 1. Adjacent non-q-commuting?
    for i in range(len(word) - 1):
        g, h = word[i], word[i + 1]
        if not _q_commute(g, h):
            return (i, 'cross')
    # 2. Adjacent q-commuting but out of order?
    for i in range(len(word) - 1):
        g, h = word[i], word[i + 1]
        if g == h:
            continue
        if _q_commute(g, h):
            # Define a strict canonical letter order: H-letters by index,
            # Wilson letters AFTER all H-letters (Wilson cone is "after" H).
            if _letter_gt(g, h):
                return (i, 'swap')
    # 3. Non-adjacent non-q-commuting: bubble.
    # Find first such pair.
    for i in range(len(word)):
        for j in range(i + 1, len(word)):
            if not _q_commute(word[i], word[j]):
                # Bubble word[j] leftward until adjacent to word[i].
                return (i, 'bubble', j)
    return None


def _letter_gt(g: tuple, h: tuple) -> bool:
    """Strict order: H-letters precede Wilson, H-letters by index."""
    if _is_H(g) and _is_H(h):
        return g[1] > h[1]
    if _is_H(g) and _is_W(h):
        return False                             # H precedes W.
    if _is_W(g) and _is_H(h):
        return True                              # W after H.
    return False                                 # W == W: equal.


def _reduce(initial_terms: list) -> list:
    """Reduce `[(LP coef, word), ...]` to canonical sorted-cone words."""
    completed: list = []
    work = list(initial_terms)
    max_steps = 200_000
    steps = 0
    while work:
        steps += 1
        if steps > max_steps:
            raise RuntimeError(
                f"_reduce: too many steps ({max_steps}); reduction "
                f"may not terminate"
            )
        c, w = work.pop()
        if c.is_zero():
            continue
        action = _find_action(w)
        if action is None:
            completed.append((c, w))
            continue
        kind = action[1]
        if kind == 'swap':
            i = action[0]
            g, h = w[i], w[i + 1]
            new_c = c * LP({2 * _cocycle(g, h): 1})
            new_w = w[:i] + (h, g) + w[i + 2:]
            work.append((new_c, new_w))
        elif kind == 'cross':
            i = action[0]
            g, h = w[i], w[i + 1]
            for sub_c, sub_w in _ray_product(g, h):
                work.append((c * sub_c, w[:i] + sub_w + w[i + 2:]))
        elif kind == 'bubble':
            i, j = action[0], action[2]
            # Bubble w[j] leftward through w[j-1], w[j-2], ..., w[i+1]
            # (all of which q-commute with w[j] by construction of j),
            # accumulating cocycle.  After bubbling, w[j] sits at index
            # i+1, adjacent to w[i] which it does NOT q-commute with —
            # cross_product will fire next iteration.
            new_w = list(w)
            phase_exp = 0
            target = new_w[j]
            for k in range(j, i + 1, -1):
                partner = new_w[k - 1]
                if not _q_commute(partner, target):
                    # Earlier collision exposed: apply cross at (k-1, k).
                    cur_w = tuple(new_w)
                    cur_c = c * LP({phase_exp: 1})
                    for sub_c, sub_w in _ray_product(partner, target):
                        new_w_split = cur_w[:k - 1] + sub_w + cur_w[k + 1:]
                        work.append((cur_c * sub_c, new_w_split))
                    break
                phase_exp += 2 * _cocycle(partner, target)
                new_w[k - 1], new_w[k] = new_w[k], new_w[k - 1]
            else:
                # No earlier collision; target is now at i+1, adjacent
                # to w[i]; re-queue for the outer scan to fire cross.
                work.append((c * LP({phase_exp: 1}), tuple(new_w)))
    return completed


# -------------------------------------------------------------------
# Top-level entry
# -------------------------------------------------------------------

def multiply_native(a: tuple, b: tuple) -> Element:
    """Native_a × native_b → Element.  Pure axiom-derived: no pSU2 iso.

    Inputs are canonicalised to max-diagonal form on entry to absorb
    any non-canonical native that may arise from upstream code paths.
    """
    a = _psu2_to_native(*_native_to_psu2(a))
    b = _psu2_to_native(*_native_to_psu2(b))
    terms_a = _expand_native_to_literal(a)
    terms_b = _expand_native_to_literal(b)
    initial: list = []
    for ca, wa in terms_a:
        for cb, wb in terms_b:
            initial.append((ca * cb, wa + wb))
    completed = _reduce(initial)
    out: dict = {}
    for c, w in completed:
        natives = _word_to_native_poly(w)
        for native, factor in natives.items():
            term = c * factor
            if term.is_zero():
                continue
            existing = out.get(native, LP.zero())
            out[native] = existing + term
    out = {k: v for k, v in out.items() if not v.is_zero()}
    return Element(out)


# -------------------------------------------------------------------
# Smoke test (verifies against pSU2KAlgebra ONLY — pSU2 is NOT used
# on the production path).
# -------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from psu2_kalgebra import pSU2KAlgebra

    P = pSU2KAlgebra()

    print("pure_su2_h_multiply smoke test (literal-word reduction)\n" + "=" * 60)

    tests = [
        # (a_native, b_native)
        ((), ()),
        ((), ((1, 1),)),
        (((1, 1),), ((2, 1),)),                                   # H_1 · H_2
        (((0, 1),), ((4, 1),)),                                   # gap 4
        (((0, 1),), ((7, 1),)),                                   # gap 7
        (((('W', 1), 1),), ((3, 1),)),                            # w_1 · H_3
        (((3, 1),), ((('W', 1), 1),)),                            # H_3 · w_1
        (((('W', 2), 1),), ((1, 1),)),                            # χ_2 · H_1
        (((('W', 3), 1),), ((('W', 2), 1),)),                     # χ_3 · χ_2
        (((0, 1), (1, 1)), ((2, 1),)),                            # m=2 · H_2
        (((0, 1), (4, 1)), ((7, 1),)),                            # m=2 · H_7
        (((0, 1), (1, 1)), ((1, 1), (2, 1))),                     # m=2 · m=2
        (((1, 2),), ((1, 1),)),                                   # H_1² · H_1
        (((1, 2),), ((('W', 1), 1),)),                            # H_1² · w_1
        (((1, 1), (2, 1)), ((3, 1),)),                            # m=2 · H_3
        (((('W', 4), 1),), ((('W', 1), 1),)),                     # χ_4 · w_1
        (((0, 2),), ((4, 1),)),                                   # H_0² · H_4
    ]
    all_ok = True
    for a, b in tests:
        s_a = _native_to_psu2(a)
        s_b = _native_to_psu2(b)
        mine = multiply_native(a, b)
        mine_psu2: dict = {}
        for lbl, c in mine.terms.items():
            m_, e_ = _native_to_psu2(lbl)
            mine_psu2[(m_, e_)] = mine_psu2.get((m_, e_), LP.zero()) + c
        mine_psu2 = {k: v for k, v in mine_psu2.items() if not v.is_zero()}
        ref = P.multiply(s_a, s_b)
        ref_dict = {k: v for k, v in ref.terms.items() if not v.is_zero()}
        ok = mine_psu2 == ref_dict
        all_ok = all_ok and ok
        print(f"  {a} · {b}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"    mine: {mine_psu2}")
            print(f"    pSU2: {ref_dict}")
    print(f"\n  → {'ALL PASS' if all_ok else 'SOME FAILED'}")
