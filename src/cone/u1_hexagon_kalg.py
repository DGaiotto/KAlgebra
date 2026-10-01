"""
u1_hexagon_kalg.py
==================

`U1HexagonKAlg` — K-algebra of the u(1)-gauged hexagon = [A_1, A_3]
(k=1 of the odd family [A_1, A_{2k+1}]) with one frozen node attached
at the end of the linear A_3 quiver:  O_1 → O_2 → O_3 → F.

Generators
----------
* μ := L_{(1, 0, 1, 0)}, the central auxiliary-torus generator.
* 6 short-diagonal references L_{1, i} for i ∈ Z_6, periodicity
        L_{1, i+6} = μ^{-2} · L_{1, i}.
* 3 long-diagonal references L_{2, i} for i ∈ Z_3, with antipodal
        L_{2, i+3} = μ^{-1} · L_{2, i}  (ρ-twist).

ρ symmetry
----------
ρ acts as (i → i+1, μ → μ^{-1}) jointly.

Serving path (2026-09-23)
-------------------------
`multiply`, `rho`, `rho_inverse` and `trace` are served by the closed-form
`u1a1aodd_kalg.U1A1AoddKAlg(1)` through the label dictionary

    (F, e)  ↦  (F, −e)            (the same letters; `E ↦ E⁻¹`),

certified on all 121 ordered products of the 11 multiplicative generators
(the 9 chords and `E^{±1}`) and on `ρ`, `ρ⁻¹` of all 11, against this class's
own frozen tables.  So neither a frozen table nor the orthonormality bootstrap
(`u1aodd_trace_bootstrap.solve_intermediate`, which has known
certification holes) is on the serving path; the chord seeds of
`_trace_residual` come from `U1A1AoddKAlg(1)`'s closed-form trace too.  The
labels are unchanged.

Frozen witnesses (off the serving path)
---------------------------------------
The 81 = 9 × 9 reference pairwise products `FULL_PLUCKER_TABLE` — derived once
via BPSKAlgebra on the gauged quiver and frozen — live in `cone_algebra.py`
beside their cone-monomial twin `MULT_TABLE_LL`, and are re-exported here
lazily (`from u1_hexagon_kalg import FULL_PLUCKER_TABLE` still works), so that
importing this class loads no frozen table.  They serve `multiply_single`,
`_legacy_multiply`, `cone_data()` and `trace_layer1`, which are kept as the
independent cross-checks.

Coefficient ring
----------------
Z[q^±] (μ tracked as a label-level integer power, not as a coefficient).
This keeps the implementation simple; μ could equally be promoted to a
proper torus generator in `AbelianZPlusRing(rank=1)`.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element
from cone_kalgebra import ConeKAlgebra
from laurent_poly import LaurentPoly
from zplus_ring import AbelianZPlusRing, RPowerSeries, TrivialZPlusRing


def _plucker_table() -> dict:
    """The frozen 81-entry `FULL_PLUCKER_TABLE` (in `cone_algebra`), loaded on
    first use — it serves only the cross-check surfaces."""
    from cone_algebra import FULL_PLUCKER_TABLE
    return FULL_PLUCKER_TABLE


def __getattr__(name):
    """Lazy re-exports of the frozen k = 1 witnesses that used to be defined
    or imported at the top of this module (PEP 562), so that their importers
    keep working while `import u1_hexagon_kalg` loads no frozen table."""
    if name == "FULL_PLUCKER_TABLE":
        return _plucker_table()
    if name == "_cone_multiply":
        from cone_algebra import multiply_cone_monomials
        return multiply_cone_monomials
    if name == "_cone_charge_of_label":
        from cone_algebra import charge_of_label
        return charge_of_label
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ----------------------------------------------------------------------
# Static data: tropical charges and tables.
# ----------------------------------------------------------------------

MU_CHARGE: tuple[int, int, int, int] = (1, 0, 1, 0)

# Flavour coords = the directions where μ (the v generator) has no charge;
# a seed's trace vanishes unless its whole ρ²-orbit can be made neutral there.
_FLAV_COORDS = tuple(j for j in range(len(MU_CHARGE)) if MU_CHARGE[j] == 0)

L1_CHARGES: dict[int, tuple[int, int, int, int]] = {
    0: (0,  1,  0,  1),
    1: (0, -1,  0, -1),
    2: (-1, 0, -1,  1),
    3: (1,  0,  1, -1),
    4: (-1, 0, -2,  1),
    5: (1, -1,  2, -1),
}

L2_CHARGES: dict[int, tuple[int, int, int, int]] = {
    0: (1,  0,  0,  0),
    1: (-1,-1, -1,  0),
    2: (0,  0,  1,  0),
    3: (0,  0, -1,  0),
    4: (0, -1,  0,  0),
    5: (-1, 0,  0,  0),
}

# BPS antisymmetric pairing for the gauged quiver O_1 → O_2 → O_3 → F.
# Kept for tests only; the closed-form FULL_PLUCKER_TABLE doesn't need it.
B_GAUGED = [
    [ 0,  1,  0,  0],
    [-1,  0,  1,  0],
    [ 0, -1,  0,  1],
    [ 0,  0, -1,  0],
]


# B(letter, μ) values — all letter·μ products are SINGLE-TERM q-commute
# in BPS, so these are well-defined integer q-powers.  Derived once via
# BPS, frozen here.
MU_LETTER_QPOWER: dict = {
    (1, 0): -1, (1, 1): +1, (1, 2): -1,
    (1, 3): +1, (1, 4): -1, (1, 5): +1,
    (2, 0): 0,  (2, 1): 0,  (2, 2): 0,
}


# ----------------------------------------------------------------------
# FULL_PLUCKER_TABLE (the 81 frozen reference products, a cross-check only)
# lives in `cone_algebra.py` since 2026-09-23 and is re-exported lazily by
# this module's `__getattr__`; `_plucker_table()` reads it.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def charge(letter: tuple[int, int]) -> tuple[int, int, int, int]:
    """Tropical charge of a single-letter label `(a, i)` with i ∈ canonical range."""
    a, i = letter
    if a == 1:
        return L1_CHARGES[i % 6]
    elif a == 2:
        return L2_CHARGES[i % 6]
    raise ValueError(f"a must be 1 or 2, got {a}")


def _decompose_charge_to_label(chg):
    """Decompose a tropical charge `chg` ∈ Z^4 as a (factors, mu_p) label,
    preferring the simplest canonical form: fewest factors, then smallest
    |mu_p|, then lex-min sorted-factors-tuple.

    Returns (factors_tuple, μ_p) or None if no decomposition found."""
    refs = [(1, i) for i in range(6)] + [(2, i) for i in range(3)]
    candidates = []  # (priority, factors, mu_p)
    for mu_p in range(-8, 9):
        residual = tuple(chg[k] - mu_p * MU_CHARGE[k] for k in range(4))
        # 0-factor
        if residual == (0, 0, 0, 0):
            candidates.append(((0, abs(mu_p)), (), mu_p))
            continue
        # 1-factor
        for r in refs:
            if charge(r) == residual:
                candidates.append(((1, abs(mu_p)), ((r[0], r[1], 1),), mu_p))
        # 1-factor with exp >= 2
        for r in refs:
            c = charge(r)
            for e in range(2, 4):
                if tuple(e*c[k] for k in range(4)) == residual:
                    candidates.append(((1, abs(mu_p) + e), ((r[0], r[1], e),), mu_p))
        # 2-factor distinct
        for r1 in refs:
            for r2 in refs:
                if r1 >= r2: continue
                c1 = charge(r1); c2 = charge(r2)
                if tuple(c1[k]+c2[k] for k in range(4)) == residual:
                    candidates.append((
                        (2, abs(mu_p)),
                        ((r1[0], r1[1], 1), (r2[0], r2[1], 1)),
                        mu_p,
                    ))
    # 3+ factor as fallback (only if no 0/1/2-factor found)
    for mu_p in range(-8, 9):
        residual = tuple(chg[k] - mu_p * MU_CHARGE[k] for k in range(4))
        for r1 in refs:
            for r2 in refs:
                if r1 == r2: continue
                c1 = charge(r1); c2 = charge(r2)
                if tuple(2*c1[k]+c2[k] for k in range(4)) == residual:
                    candidates.append((
                        (3, abs(mu_p)),
                        tuple(sorted([(r1[0], r1[1], 2), (r2[0], r2[1], 1)])),
                        mu_p,
                    ))
        for r1 in refs:
            for r2 in refs:
                if r1 >= r2: continue
                for r3 in refs:
                    if r3 <= r2: continue
                    c1, c2, c3 = charge(r1), charge(r2), charge(r3)
                    if tuple(c1[k]+c2[k]+c3[k] for k in range(4)) == residual:
                        candidates.append((
                            (3, abs(mu_p)),
                            tuple(sorted([(r1[0], r1[1], 1), (r2[0], r2[1], 1), (r3[0], r3[1], 1)])),
                            mu_p,
                        ))
    if not candidates:
        return None
    # Pick smallest priority (fewest factors, then smallest |mu_p|, then lex)
    candidates.sort(key=lambda c: (c[0], c[1]))
    _, factors, mu_p = candidates[0]
    return (factors, mu_p)


def _normalize_letter(letter: tuple[int, int]) -> tuple[tuple[int, int], int]:
    """Reduce (a, i) to canonical-form (a, i') with i' in PLUCKER_TABLE range,
    returning the μ-power picked up.

    For longs: antipodal rule L_{2, i+3} = μ^{ε_i} L_{2, i} with
    sign ε_i = -1 if i even, +1 if i odd."""
    a, i = letter
    if a == 1:
        return ((1, i % 6), -2 * (i // 6))
    elif a == 2:
        i_mod6 = i % 6
        wraps = i // 6  # number of full 6-cycles
        # Each full 6-cycle: L_{2, i+6} = L_{2, i+3+3} = μ^{ε_{i+3}} μ^{ε_i} L_{2, i}
        # = product of two ε's = (-1)(+1) or (+1)(-1) depending on parity.
        # For i base 0,1,2: ε_0+ε_1+ε_2 contributions over 2 hops (i+3 then i+6)
        # = (-1) + (+1) for i=0, etc.  Net per 6: -1+1=0 for i=0/2, +1-1=0 for i=1.
        # So 6-cycle contributes 0 to mu_p for longs.
        mu_p = 0
        if i_mod6 < 3:
            return ((2, i_mod6), mu_p)
        base_i = i_mod6 - 3
        sign = -1 if base_i % 2 == 0 else 1
        return ((2, base_i), mu_p + sign)
    raise ValueError(f"a must be 1 or 2, got {a}")


# ----------------------------------------------------------------------
# Term representation: (q_power: int, mu_power: int, factors: tuple[(a, i), ...])
# Element: dict from sorted factors tuple to (mu_power_total, LaurentPoly)
# Simplified representation: a list of (q, mu, factors).
# ----------------------------------------------------------------------


def _multiply_letters(letter1, letter2) -> list[tuple[int, int, list]]:
    """Closed-form single-letter multiply via FULL_PLUCKER_TABLE.  Handles
    arbitrary i by normalizing letters first."""
    l1n, mu1 = _normalize_letter(letter1)
    l2n, mu2 = _normalize_letter(letter2)
    pre_mu = mu1 + mu2
    table = _plucker_table()
    if (l1n, l2n) not in table:
        raise KeyError(f"No PLUCKER_TABLE entry for ({l1n}, {l2n})")
    entries = table[(l1n, l2n)]
    return [(q, mu + pre_mu, list(factors)) for (q, mu, factors) in entries]


# ----------------------------------------------------------------------
# U1HexagonKAlg class
# ----------------------------------------------------------------------


class U1HexagonKAlg(ConeKAlgebra):
    """K-algebra of u(1)-gauged [A_1, A_3] hexagon.

    Cone-monomial canonical basis: each label is `(factors, e_E)` where
    `factors` is a sorted tuple of `(a, i, exp)` triples all drawn from
    one of the 14 maximal q-commuting subsets (cones), and `e_E ∈ Z` is
    the E-power.  Each cone monomial carries an implicit `fq^{T_half}`
    prefactor (with `T_half = cone_T_half(factors, e_E)`) making it
    bar-invariant under `q → 1/q, factor-reverse`.

    Public surface:
      * `L((a, i))` accessor — single-letter cone monomial
      * `identity`, `mu` — distinguished labels
      * `rho`, `rho_inverse` — Z_6 cyclic symmetry (inverts E)
      * `multiply(label1, label2)` — the closed-form product of
        `U1A1AoddKAlg(1)` read through `(F, e) ↦ (F, −e)` (see the module
        docstring); the frozen single-letter table survives as
        `multiply_single` and the frozen cone reducer as `_legacy_multiply`,
        both cross-checks
      * `as_generator_monomial(label)` — expresses a label as
        `fq^c · sorted L-product · μ^m`
      * `trace_layer1(label)` — closed-form Layer 1 ρ²-tagged cyclicity
        reduction to a dict of irreducible canonical labels with
        LaurentPoly coefficients (over the frozen cone data; a cross-check)
      * `trace(label, K)` — `U1A1AoddKAlg(1)`'s closed-form trace of the
        relabelled label (Layer-1 cyclicity + the `M(1,3)` singlet
        closed forms), any label to any order; `_bps_trace` is the
        off-path verification twin
      * `elementary_traces()` — provisional list of elementary trace
        symbols (T_0, T_long)
    """

    H = 6  # Z_6 cyclic order on label indices

    def __init__(self):
        self._R = TrivialZPlusRing()
        self.k = 1                  # k of the U1A1AoddKAlg family (= [A_1, A_3])
        self._gen = None            # U1A1AoddKAlg(1), the serving presentation
        self._boot_cache = {}       # cross-check only: the bootstrap solve (max K)
        self._rep_cache = {}        # ρ²-rep memo for the bootstrap lookup

    # -- the serving presentation -------------------------------------------

    def _general(self):
        """`U1A1AoddKAlg(1)`, which serves `multiply`, `rho`, `rho_inverse`
        and `trace` through `_to_general` / `_from_general`.  Built on first
        use and cached."""
        if self._gen is None:
            from u1a1aodd_kalg import U1A1AoddKAlg
            self._gen = U1A1AoddKAlg(1)
        return self._gen

    @staticmethod
    def _to_general(label):
        """This class's label `(F, e)` as the `U1A1AoddKAlg(1)` label
        `(F, −e)`: the letters `(a, i)` are the same diagonals, and this
        class's `E` is `U1A1AoddKAlg(1)`'s `E⁻¹`.  Certified on all 121
        ordered products of the 11 multiplicative generators and on `ρ`,
        `ρ⁻¹` of each (the suite in the source repository compares the
        served product with the frozen `_legacy_multiply`; the dictionary
        is its own inverse)."""
        factors, e = label
        return (factors, -e)

    _from_general = _to_general

    # -- KAlgebra contract --

    def coefficient_ring(self):
        """U(1)Hex is unflavoured (E is a dynamical algebra generator,
        not a flavour fugacity).  Returns `TrivialZPlusRing()`."""
        return self._R

    # -- Generators / labels --

    def L(self, label: tuple[int, int]):
        """Canonical-form label `((a, i_mod_period, 1),)` with μ-power 0.

        Returns a 2-tuple (factors_tuple, mu_power)."""
        a, i = label
        if a == 1:
            return (((1, i % 6, 1),), 0)
        elif a == 2:
            return (((2, i % 3, 1),), 0)
        raise ValueError(f"a must be 1 or 2, got {a}")

    def identity(self):
        """Identity = empty factor tuple, μ-power 0."""
        return ((), 0)

    @property
    def mu(self):
        """The μ generator = (empty factors, μ-power 1)."""
        return ((), 1)

    # -- Geometry: the diagonals of the hexagon --

    def geometric_label(self, g):
        """A letter `g = (a, i)` — the letter of a factor `(a, i, exp)`, a
        multiplicative generator of `cone_data()` — as a sorted diagonal
        `(v1, v2)` of the hexagon with marked points `0, …, 5`: `(1, i)` is
        `{i, i + 2}`, `(2, i)` the diameter `{i, i + 3}`.  Read from
        `U1A1AoddKAlg(1).geometric_label`, whose letters these are (see
        `_to_general`).  `None` for the gauge letters `E = (3, 0)` and
        `E⁻¹ = (3, 1)` (`u1_hexagon_cone_data`'s `E_GEN`, `E_INV`, named here
        rather than imported: that module loads the frozen tables of
        `cone_algebra`, which no serving path reads)."""
        g = tuple(g)
        if g in ((3, 0), (3, 1)):
            return None
        return self._general().geometric_label(g)

    # -- ρ symmetry --

    def rho(self, lbl):
        """ρ, served by `U1A1AoddKAlg(1)` through `(F, e) ↦ (F, −e)`
        (`_legacy_rho` is this class's own closed form, kept as the
        cross-check; the two agree on every label)."""
        return self._from_general(self._general().rho(self._to_general(lbl)))

    def rho_inverse(self, lbl):
        """ρ⁻¹, served by `U1A1AoddKAlg(1)` through `(F, e) ↦ (F, −e)`
        (`_legacy_rho_inverse` is the cross-check)."""
        return self._from_general(
            self._general().rho_inverse(self._to_general(lbl)))

    def _legacy_rho(self, lbl):
        """ρ acts as (i → i+1, μ → μ^{-1}) jointly.

        For each (a, i, e) factor: shift i → i+1 (with period 6 for a=1,
        period 3 for a=2 — kept in canonical range via _normalize_letter).
        μ-power: invert the *input* power, then ADD the wrap pickups —
        `ρ(μ^m·L_w) = μ^{-m}·ρ(L_w) = μ^{-m+δ(w)}·L_{Pw}`: the pickup
        δ(w) is a property of the shifted word and is not subject to
        the inversion.  Rule derived from and certified against the
        contract-correct `U1A1AoddKAlg(1)` through the `E ↦ E⁻¹`
        dictionary; the prior version negated the pickup together with
        the input power (the long-standing long-chord ρ/trace
        inconsistency, now fixed)."""
        factors, mu_p = lbl
        new_factors = []
        pickup = 0
        for (a, i, e) in factors:
            (a_n, i_n), mu_add = _normalize_letter((a, i + 1))
            new_factors.append((a_n, i_n, e))
            pickup += e * mu_add  # μ pickup from wrap-around
        new_factors.sort()
        return (tuple(new_factors), -mu_p + pickup)

    def _legacy_rho_inverse(self, lbl):
        """Inverse of `_legacy_rho`: from `ρ(w, m) = (Pw, −m + δ(w))` solve
        `m = δ(w) − n` with `w = P⁻¹·input`.  δ(w) is recomputed with
        the SAME up-shift normalization `rho` uses (`_normalize_letter`'s
        μ-pickup conventions differ between the a=1 and a=2 families, so
        re-deriving the pickup from the down-shift is not reliable);
        with δ shared, both round-trips hold identically."""
        factors, mu_p = lbl
        new_factors = []
        for (a, i, e) in factors:
            (a_n, i_n), _mu_add = _normalize_letter((a, i - 1))
            new_factors.append((a_n, i_n, e))
        new_factors.sort()
        delta_w = 0
        for (a, i, e) in new_factors:
            _l, mu_add = _normalize_letter((a, i + 1))
            delta_w += e * mu_add
        return (tuple(new_factors), delta_w - mu_p)

    # -- Multiplication (single-letter) --

    def multiply_single(self, letter1, letter2):
        """Multiply two single-letter labels `(a, i)`.  Returns a dict
        {label: LaurentPoly_coeff} with canonical labels via
        `_decompose_charge_to_label`."""
        out_terms = _multiply_letters(letter1, letter2)
        result: dict = {}
        for (q_p, mu_p, factors) in out_terms:
            # Compute the X-monomial charge of this output term.
            ch = list(mu_p * c for c in MU_CHARGE)
            for (a, i) in factors:
                fc = charge((a, i))
                ch = [ch[k] + fc[k] for k in range(4)]
            chg = tuple(ch)
            # Canonicalize via decompose
            decomp = _decompose_charge_to_label(chg)
            if decomp is None:
                continue
            key = decomp
            coef = LaurentPoly.q(q_p)
            if key in result:
                result[key] = result[key] + coef
            else:
                result[key] = coef
        return {k: v for k, v in result.items() if not v.is_zero()}

    # -- Multi-letter word multiplication --

    def _pair_q_power(self, la, lb):
        """X-basis q-power for single-term pair (la, lb) — = B(γ_la, γ_lb)
        derived from the table.  Returns None if (la, lb) Plückers."""
        la_n, _ = _normalize_letter(la)
        lb_n, _ = _normalize_letter(lb)
        entries = _plucker_table().get((la_n, lb_n))
        if entries is None or len(entries) != 1:
            return None
        return entries[0][0]

    def _pair_mu_power(self, la, lb):
        """μ-power contribution from single-term pair (la, lb)."""
        la_n, _ = _normalize_letter(la)
        lb_n, _ = _normalize_letter(lb)
        entries = _plucker_table().get((la_n, lb_n))
        if entries is None or len(entries) != 1:
            return None
        return entries[0][1]

    def cone_data(self):
        """`U1HexagonConeData` singleton (stateless)."""
        from u1_hexagon_cone_data import U1HEXAGON_CONE_DATA
        return U1HEXAGON_CONE_DATA

    def multiply(self, label1, label2):
        """The product of `U1A1AoddKAlg(1)` (closed-form analytic peel, no
        frozen table) read through `(F, e) ↦ (F, −e)`.  Until 2026-09-23 this
        routed through `cone_data().derived_multiply`, the reducer over the
        frozen `MULT_TABLE_LL`; that route (`_multiply_via_cone_data`) and
        `_legacy_multiply` are kept as cross-checks."""
        prod = self._general().multiply(self._to_general(label1),
                                        self._to_general(label2))
        return Element({self._from_general(l): c for l, c in prod.terms.items()})

    def _legacy_multiply(self, label1, label2):
        """Legacy multiply via `cone_algebra.multiply_cone_monomials`
        (frozen `MULT_TABLE_LL` + closed-form bubble reducer).  Kept
        as an independent test oracle; not on the live `multiply` path.

        Labels are cone monomials `(factors, e_E)` where `factors` is a
        sorted tuple of `(a, i, exp)` all drawn from one of the 14 maximal
        q-commuting subsets, and `e_E ∈ Z` is the E-power.
        """
        from cone_algebra import multiply_cone_monomials as _cone_multiply
        return Element(_cone_multiply(label1, label2))

    @classmethod
    def _charge_of_label(cls, label):
        """Tropical charge γ ∈ Z⁴ of a cone-monomial label `(factors, e_E)`."""
        factors, e_E = label
        ch = list(e_E * c for c in MU_CHARGE)
        for (a, i, e) in factors:
            fc = charge((a, i))
            for _ in range(e):
                ch = [ch[k] + fc[k] for k in range(4)]
        return tuple(ch)

    # -- Canonical basis ↔ generator monomial conversion --

    def as_generator_monomial(self, label):
        """Express the canonical basis element labeled by `label` as
        `q^c · (L-product in sorted generator monomial form) · μ^m`.

        Returns: (q_power_c, factors_tuple, mu_power).

        The canonical basis element X^{charge(label)} equals
            q^{-bps_twist(label)} · L_{a_1, i_1}^{e_1} · ... · μ^{mu_power}
        where bps_twist = Σ pairwise forward-q-powers from FULL_PLUCKER_TABLE
        across all single-letter pairs (i, j) in the factor multiset (i ≤ j).
        """
        factors, mu_p = label
        bps_twist = 0
        # Pair q-powers for all (a, i, e) pair combinations including self-pairs
        # with multiplicity n(n-1)/2 for e copies of same letter.
        for idx_a in range(len(factors)):
            (ka, ia, ea) = factors[idx_a]
            for idx_b in range(idx_a, len(factors)):
                (kb, ib, eb) = factors[idx_b]
                if idx_a == idx_b:
                    # Same letter — n choose 2 self-pairs.  Same-letter pair
                    # is single-term in table with q-power 0 (table check).
                    qp = self._pair_q_power((ka, ia), (ka, ia))
                    if qp is None: continue
                    bps_twist += qp * ea * (ea - 1) // 2
                else:
                    qp = self._pair_q_power((ka, ia), (kb, ib))
                    if qp is None: continue
                    bps_twist += qp * ea * eb
        # X^{charge(label)} = q^{-bps_twist} · L-product · μ^{mu_p}
        return (-bps_twist, factors, mu_p)

    # -- Trace via ρ²-tagged-cyclicity (Layer 1) --

    def _rho2_label(self, label):
        """Apply ρ² to a cone monomial label, as the contract `rho∘rho`.

        The previous standalone formula (a=1: i→(i+2)%6 with no pickup;
        antipodal sign into e_E) DISAGREED with `rho∘rho` on the a=1 wrap and
        was *wrong* vs the BPS ground truth (`bps.rho²(charge) ==
        charge(rho∘rho(label))`, verified on the label sweep) — it broke
        `Tr∘ρ²=Tr` on multi-letter labels.  Delegating to the BPS-correct
        `rho` keeps `_rho2_label`, `_rho2_canonical_label`,
        `_canonical_rho2_orbit_rep`, and the trace pipeline mutually
        consistent."""
        return self.rho(self.rho(label))

    def _rho2_canonical_label(self, label):
        """Lex-min ρ²-orbit representative (Tr is ρ²-invariant, so all orbit
        elements share the same trace).  Walks the FULL ρ²-orbit until it
        closes (bounded by `self.H * 2`) — the old 2-step walk assumed every
        orbit had period 3 and returned an unstable rep on the longer ones."""
        best = label
        cur = label
        for _ in range(self.H * 2):
            cur = self._rho2_label(cur)
            if cur == label:
                break
            if cur < best:
                best = cur
        return best

    # ----- KAlgebra._canonical_rho2_orbit_rep override -------------------

    def _canonical_rho2_orbit_rep(self, label):
        """Delegate to `_rho2_canonical_label` (closed-form period-3
        walk).  Used by `ConeData._collapse_rho2_orbits_in_element` for
        the trace pipeline canonicalisation."""
        return self._rho2_canonical_label(label)

    # ----- Layer-2 trace residual (BPS-free: exact v-tower + bootstrap) ---

    def _seed_charge(self, seed):
        """Tropical charge `γ ∈ Z⁴` of a cone-monomial seed `(factors, μ_p)`
        (= `_charge_of_label`, as a list)."""
        return list(self._charge_of_label(seed))

    def _orbit_has_physical(self, seed):
        """`True` iff the ρ²-orbit of `seed` contains a flavour-neutral
        member (zero charge on `_FLAV_COORDS`) — i.e. `Tr ≠ 0`.  The short
        chords (type 1) are flavour-charged throughout their orbit → `Tr = 0`,
        reproducing the U1Hex-specific u(1)-mag-charge vanishing rule."""
        factors = seed[0]
        if not factors:
            return True
        P = 6 if factors[0][0] == 1 else 3
        cur = seed
        for _ in range(2 * P + 2):
            g = self._seed_charge(cur)
            if not any(g[j] for j in _FLAV_COORDS):
                return True
            cur = self.rho(self.rho(cur))
        return False

    def _lp_to_rps(self, lp, K):
        return RPowerSeries(
            self._R, {e: c for e, c in lp._coeffs.items() if 0 <= e <= K}, K)

    def _boot(self, K, nmax=3):
        """CROSS-CHECK ONLY since 2026-09-23 — not on any serving path: the
        chord seeds of `_trace_residual` now come from `U1A1AoddKAlg(1)`'s
        closed-form trace, and `u1aodd_trace_bootstrap.solve_intermediate`
        has known certification holes.
        Measured equal to the new seeds at `k = 1` through `K = 40`.

        Orthonormality-bootstrap solve, cached at the largest
        (K, nmax) requested (smaller reuse it).  Trusts ONLY the exact M(1,3)
        singlet v-tower and solves the long chord (type 2 = the k=1 diameter)
        from orthonormality — BPS-free.

        `nmax` is the gauge half-width of the pool: a single chord at
        |gauge| ≤ nmax is solved.  The default 3 suffices for ordinary traces
        (Layer-1 leaves shallow gauges); ungauging widens it on demand (it
        needs `Tr(L·E^n)` up to |n| ~ K)."""
        from u1aodd_trace_bootstrap import solve_intermediate
        curK = self._boot_cache.get("K", -1)
        curN = self._boot_cache.get("nmax", -1)
        if K > curK or nmax > curN:
            useK, useN = max(K, curK), max(nmax, curN, 3)
            self._boot_cache["Tr"] = solve_intermediate(
                self, useK, unknown_types={2}, nmax=useN)
            self._boot_cache["K"] = useK
            self._boot_cache["nmax"] = useN
        return self._boot_cache["Tr"]

    def _trace_residual(self, seed_label, K):
        """Layer-2 trace of one Layer-1 seed — closed forms only.

          * vanishing (short chords, flavour-charged orbit) → 0;
          * v-tower `μ^m` → the exact M(1,3) singlet character `tr_v_n(m)`;
          * long chord (type 2) → `U1A1AoddKAlg(1)`'s closed-form trace of
            the relabelled seed `(F, −m)` (`u1_pgon_layer2.
            singlet_chord_trace`, every `m`).  Until 2026-09-23 this branch
            called `_boot` (`u1aodd_trace_bootstrap.solve_intermediate`),
            which is now a cross-check only; the two agree through `K = 40`.

        `trace` itself no longer comes through here (it is served by
        `U1A1AoddKAlg(1)` directly); this reduction stays for the class's
        own Layer-1 route (`ConeKAlgebra.trace` over `cone_data()`)."""
        if not self._orbit_has_physical(seed_label):
            return RPowerSeries(self._R, {}, K)
        factors, m = seed_label
        g0 = self._seed_charge(seed_label)[0]
        if not factors:
            import u1_hexagon_singlet as _sing   # exact M(1,3) singlet v-tower
            return self._lp_to_rps(_sing.tr_v_n(g0, K), K)      # v-tower: n = g0
        if len(factors) == 1 and factors[0][2] == 1:
            return self._general().trace(self._to_general(seed_label), K)
        raise NotImplementedError(
            f"U1HexagonKAlg._trace_residual: physical seed {seed_label!r} "
            f"(g0={g0}) is neither a v-tower nor a single chord — Layer-1 did "
            f"not reduce it.  Honest-fail rather than return silently wrong.")

    def _is_cone_monomial(self, label):
        """True iff every chord letter of `label` lies in a common cone — i.e.
        `label` is a genuine canonical basis element (single cone monomial).
        The cones are the maximal sets of pairwise q-commuting letters, so this
        is the pairwise test, read off `U1A1AoddKAlg(1)`'s closed-form
        q-commutation (non-crossing diagonals; the same letters here)."""
        factors, _ = label
        letters = sorted(set((a, i) for (a, i, exp) in factors))
        cd = self._general().cone_data()
        return all(cd.q_commute(letters[x], letters[y])
                   for x in range(len(letters))
                   for y in range(x + 1, len(letters)))

    def _mag_charge(self, label):
        """Σ_l MU_LETTER_QPOWER[l] · e_l for the L-factors.  Non-zero
        ⇒ Tr(label) = 0 by E-insertion-cycling argument."""
        factors, _ = label
        return sum(MU_LETTER_QPOWER[(a, i)] * e for (a, i, e) in factors)

    def trace_layer1(self, label):
        """Reduce Tr(label) to a dict {irreducible_canonical_label: LaurentPoly}.

        Routes through `cone_data().simplify_trace_via_cone_data`: the
        generic tagged-cyclicity engine reduces `L_label` to an
        Element supported on U1Hex's trace seeds (identity, pure
        `E^n` for `n ∈ ℤ`, and single L-letter cone-monomials), and
        we return that as a dict.

        Preflight: if `_mag_charge(label) ≠ 0` (= Σ_l MU_LETTER_QPOWER[l]·e_l
        non-zero), `Tr = 0` by the u(1)-gauging E-insertion-and-cycling
        argument; return `{}`.  This vanishing rule is U1Hex-specific
        (= the u(1) projection) and not implemented at the cone_data level.
        """
        if self._mag_charge(label) != 0:
            return {}
        cd = self.cone_data()
        # ρ²-canonicalize the input — matches legacy convention so the
        # output dict keys are the lex-min ρ²-orbit representatives.
        canon = self._rho2_canonical_label(label)
        simplified = cd.simplify_trace_via_cone_data(self, canon)
        # Each output label is itself ρ²-canonicalized; coefficients
        # collected across labels that collapse to the same orbit-rep.
        out: dict = {}
        for lbl, coef in simplified.terms.items():
            key = self._rho2_canonical_label(lbl)
            if key in out:
                out[key] = out[key] + coef
            else:
                out[key] = coef
        return {k: v for k, v in out.items() if not v.is_zero()}

    def _legacy_trace_layer1(self, label):
        """Legacy `trace_layer1` via the in-class bubble-cycle reducer
        with γ_dict closure.  Kept as an independent test oracle; not
        on the live `trace_layer1` path."""
        return self._trace_layer1(label, {})

    @staticmethod
    def _is_atomic(label):
        """True if `label` is a trivially-base case: identity, pure E^n, or a
        single L-letter with exp 1 and e_E = 0."""
        factors, e_E = label
        if not factors: return True  # identity or pure E^n
        return len(factors) == 1 and factors[0][2] == 1 and e_E == 0

    def _trace_layer1(self, label, in_progress):
        from cone_algebra import cone_T_half
        zero = LaurentPoly.zero()
        one = LaurentPoly.one()
        # 1. Vanishing
        if self._mag_charge(label) != 0:
            return {}
        # 2. ρ²-canonicalize
        canon = self._rho2_canonical_label(label)
        # 3. Atomic base cases (identity, pure E^n, single L letter)
        if self._is_atomic(canon):
            return {canon: one}
        # 4. Cycle detection — treat the irreducible canonical label as its
        # own elementary symbol.  Layer 2 enumerates the finite set of
        # irreducible canonical labels and computes their power series.
        if canon in in_progress:
            return {canon: one}
        new_in_progress = dict(in_progress); new_in_progress[canon] = True
        # 5. Peel.  Use the ρ²-twisted cyclicity axiom:
        #   Tr(L_first · algebra(rest)) = Tr(ρ²(algebra(rest)) · L_first)
        # Take L_first = smallest factor of canon; rest = canon minus one copy.
        factors, e_E = canon
        first_a, first_i, first_e = factors[0]
        rest_factors = list(factors)
        if first_e == 1:
            rest_factors.pop(0)
        else:
            rest_factors[0] = (first_a, first_i, first_e - 1)
        rest_label = (tuple(rest_factors), e_E)
        rest_rho2 = self._rho2_label(rest_label)
        L_first_label = (((first_a, first_i, 1),), 0)
        # Compute ρ²(algebra(rest)) · L_first  via cone_multiply on cone monomials.
        from cone_algebra import multiply_cone_monomials as _cone_multiply
        prod = _cone_multiply(rest_rho2, L_first_label)
        # Prefactor: M_canon = fq^{T_canon} · algebra(canon),  T_canon = cone_T_half(canon).
        # Tr(M_canon) = fq^{T_canon} · Tr(algebra(canon))
        #             = fq^{T_canon} · Tr(ρ²(alg(rest)) · L_first)
        #             = fq^{T_canon} · fq^{-T_rho2_rest} · Σ c_k · Tr(M_k)
        # (T(L_first) = 0 for a single L-letter with e_E = 0.)
        T_canon     = cone_T_half(canon[0],       canon[1])
        T_rho2_rest = cone_T_half(rest_rho2[0],   rest_rho2[1])
        prefactor = LaurentPoly.q(T_canon - T_rho2_rest)
        result: dict = {}
        for M_out, coef in prod.items():
            sub = self._trace_layer1(M_out, new_in_progress)
            adj = coef * prefactor
            for s, v in sub.items():
                result[s] = (result.get(s, zero) + adj * v)
        # Drop zero coefficients
        return {s: v for s, v in result.items() if not v.is_zero()}

    def trace(self, label, K=20):
        """Schur-index trace as an `RPowerSeries` in fq — closed forms only.

        Served by `U1A1AoddKAlg(1).trace` of the relabelled label
        `(F, −e)`: its Layer-1 cyclicity reduction with every seed a closed
        form (the `M(1,3)` singlet characters of `u1_pgon_layer2`), defined
        on every label to any order.  Until 2026-09-23 this was the class's
        own Layer-1 over the frozen cone data with chord seeds from the
        orthonormality bootstrap; that route (`ConeKAlgebra.trace(self, …)`,
        whose chord seeds now also come from the closed form) and the BPS
        twin (`_bps_trace`, the suite in the source repository) are the
        cross-checks.

        `label` must be a single cone monomial (canonical basis element); a
        cross-cone product is not a basis element — `multiply` it first.  For
        the raw Layer-1 dict, call `trace_layer1`."""
        if not self._is_cone_monomial(label):
            raise ValueError(
                f"U1HexagonKAlg.trace: {label!r} is not a single cone monomial "
                f"(its chord letters span multiple cones), so it is a product "
                f"of basis elements rather than one basis element.  Call "
                f"`multiply` first, then trace each summand.")
        return self._general().trace(self._to_general(label), K)

    def _bps_trace(self, label, K=20):
        """Cross-check oracle: trace via BPSKAlgebra(gauged-hexagon) on the
        label's tropical charge `_charge_of_label(label)` (cached).  The BPS
        trace is automatically ρ²-invariant.  Retained as the independent
        ground truth the closed-form `trace` is verified against; not on the
        live path."""
        chg = self._charge_of_label(label)
        cache_key = (chg, K)
        if not hasattr(self, '_trace_cache'):
            self._trace_cache: dict = {}
        if cache_key in self._trace_cache:
            return self._trace_cache[cache_key]
        if not hasattr(self, '_bps'):
            from bps_kalgebra import BPSKAlgebra
            self._bps = BPSKAlgebra(
                pairing=B_GAUGED,
                node_charges=[(1,0,0,0),(0,1,0,0),(0,0,1,0)],
            )
        result = self._bps.trace(chg, K=K)
        self._trace_cache[cache_key] = result
        return result

    # -- Trace cyclicity helper --

    def rho_canonical_label(self, label):
        """Return the lex-minimum label in the ρ²-orbit of `label`.

        ρ² is an algebra automorphism (i → i+2, μ unchanged).  Tr is
        invariant under ρ²: Tr(X) = Tr(ρ²(X)).  So all labels in a
        ρ²-orbit have equal trace; we represent the orbit by its
        lex-min representative."""
        orbit = [label]
        cur = label
        for _ in range(self.H * 2):
            cur = self.rho(self.rho(cur))
            if cur == label:
                break
            orbit.append(cur)
        return min(orbit)

    def elementary_traces(self):
        """Elementary trace labels (partial).

        For U1HexagonKAlg (gauged hexagon, NO flavour μ):
          T_0 = Tr(1)
          T_long = Tr(L_{2, 0})  (and ρ²-images Tr(L_{2, j}))
          (L_{1, j} traces are constrained by ρ²-drift; their reduction
          is not derived here)
        """
        return [
            ('T_0', ((), 0)),
            ('T_long', self.L((2, 0))),
        ]

    # -- Pretty printing --

    @staticmethod
    def label_str(lbl) -> str:
        factors, mu_p = lbl
        if not factors and mu_p == 0:
            return "1"
        parts = []
        if mu_p != 0:
            parts.append(f"μ^{mu_p}" if mu_p != 1 else "μ")
        for (a, i, e) in factors:
            base = f"L_{{{a},{i}}}"
            parts.append(base if e == 1 else f"{base}^{e}")
        return " · ".join(parts) if parts else "1"


# ----------------------------------------------------------------------
# Verification (against BPSKAlgebra ground truth) — only used in tests
# ----------------------------------------------------------------------


def verify_all_reference_pairs():
    """Verify all 81 reference pairwise products against BPS."""
    from bps_kalgebra import BPSKAlgebra
    A_bps = BPSKAlgebra(pairing=B_GAUGED,
                        node_charges=[(1,0,0,0),(0,1,0,0),(0,0,1,0)])
    A_my = U1HexagonKAlg()
    refs = [(1, i) for i in range(6)] + [(2, i) for i in range(3)]
    pass_count = 0
    fail_count = 0
    for l1 in refs:
        for l2 in refs:
            mine = A_my.multiply_single(l1, l2)
            # Reconstruct charges
            mine_by_charge: dict = {}
            for (factors, mu_p), coef in mine.items():
                ch = list(mu_p * c for c in MU_CHARGE)
                for (a, i, e) in factors:
                    fc = charge((a, i))
                    for _ in range(e):
                        ch = [ch[k] + fc[k] for k in range(4)]
                mine_by_charge[tuple(ch)] = mine_by_charge.get(tuple(ch), LaurentPoly.zero()) + coef
            actual = {chg: c for chg, c in A_bps.multiply(charge(l1), charge(l2)).terms.items()}
            keys = set(mine_by_charge) | set(actual)
            ok = all(str(mine_by_charge.get(k, LaurentPoly.zero())) == str(actual.get(k, LaurentPoly.zero())) for k in keys)
            if ok:
                pass_count += 1
            else:
                fail_count += 1
                if fail_count <= 3:
                    print(f"  FAIL: L_{l1} · L_{l2}")
                    print(f"    mine:   {dict((k, str(v)) for k, v in mine_by_charge.items())}")
                    print(f"    actual: {dict((k, str(v)) for k, v in actual.items())}")
    return pass_count, fail_count


def verify_rho_equivariance():
    """Verify ρ(L_{a,i_1} · L_{a',i_2}) = ρ(L_{a,i_1}) · ρ(L_{a',i_2})
    for all reference pairs."""
    A = U1HexagonKAlg()
    refs = [(1, i) for i in range(6)] + [(2, i) for i in range(3)]
    pass_count = 0
    fail_count = 0
    for l1 in refs:
        for l2 in refs:
            # LHS: ρ(L_l1 · L_l2)
            lhs_prod = A.multiply_single(l1, l2)
            lhs_rho: dict = {}
            for lbl, coef in lhs_prod.items():
                new_lbl = A.rho(lbl)
                lhs_rho[new_lbl] = lhs_rho.get(new_lbl, LaurentPoly.zero()) + coef
            # RHS: ρ(L_l1) · ρ(L_l2) = L_{l1+1} · L_{l2+1}
            a1, i1 = l1; a2, i2 = l2
            rhs_prod = A.multiply_single((a1, i1 + 1), (a2, i2 + 1))
            # Note: the result lives in the shifted labelling, but with μ-twist
            # applied (ρ inverts μ), so we need to invert μ-powers of LHS for comparison.
            keys = set(lhs_rho) | set(rhs_prod)
            ok = all(str(lhs_rho.get(k, LaurentPoly.zero())) == str(rhs_prod.get(k, LaurentPoly.zero())) for k in keys)
            if ok:
                pass_count += 1
            else:
                fail_count += 1
                if fail_count <= 3:
                    print(f"  FAIL ρ-equivariance: L_{l1} · L_{l2}")
                    print(f"    LHS = ρ(prod): {dict((str(k), str(v)) for k, v in lhs_rho.items())}")
                    print(f"    RHS = ρL·ρL:   {dict((str(k), str(v)) for k, v in rhs_prod.items())}")
    return pass_count, fail_count


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == '__main__':
    A = U1HexagonKAlg()
    print("U1HexagonKAlg (closed-form, no BPSKAlgebra runtime dependency)")
    print(f"  μ charge: {MU_CHARGE}")
    print(f"  9 references: 6 short L_{{1, 0..5}} + 3 long L_{{2, 0..2}}")
    print(f"  {len(_plucker_table())} reference pairwise products tabulated")
    print()
    print("Sample products:")
    for l1, l2 in [((2, 0), (2, 2)), ((1, 0), (1, 1)), ((1, 0), (2, 1))]:
        result = A.multiply_single(l1, l2)
        parts = []
        for lbl, coef in result.items():
            parts.append(f"({coef}) · {A.label_str(lbl)}")
        print(f"  L_{{{l1[0]},{l1[1]}}} · L_{{{l2[0]},{l2[1]}}} = {' + '.join(parts)}")
    print()
    print("Verification:")
    p, f = verify_all_reference_pairs()
    print(f"  All 81 reference pairs vs BPS: {p} pass, {f} fail")
