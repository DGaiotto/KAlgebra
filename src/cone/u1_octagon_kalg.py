"""
u1_octagon_kalg.py
==================

U1OctaKAlg — u(1)-gauged [A_1, A_5] AD theory, k=2 of the U1A1AoddKAlg
family.  Closed-form chord-pair multiply via static `MULT_TABLE`
(no BPS dependency at runtime for chord-letter products).

Primitive chord generators (= the multiplicative basis for chord-pair
products):
  L_1 (short,  distance 2): 8 generators, indexed by i ∈ Z_8
  L_2 (medium, distance 3): 8 generators, indexed by i ∈ Z_8
  E (central flavour)
Total: 16 chord + E^± = 18 mult generators.

L_3 (diameters) are COMPOSITE (= E^±·L_2·L_2); see
`u1a1aodd_k2_chord_charges.L_3_FACTORISATION` and `l3_as_l2_product`.

This SKELETON implements:
  - Single chord-letter products via MULT_TABLE (closed-form, no BPS).
  - Round-trip verification against `BPSKAlgebra` for all 256
    chord-pair products.

Future work (next session): full cone-monomial multiply (handling
compound labels like ((1, 0, 2), (2, 0, 1)) = L_1(0)²·L_2(0)),
trace via Layer 1 (ρ²-tagged cyclicity), KAlgebraIso to U1A1Aodd_k2.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element
from cone_kalgebra import ConeKAlgebra
from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing, RPowerSeries
from u1a1aodd_k2_chord_charges import (
    chord_charge, E_CHARGE, PRIMITIVE_FAMILY_PERIOD,
)
from u1_octagon_mult_table import MULT_TABLE
from u1_octagon_cone_data import U1OctaConeData
import u1_octagon_singlet as _sing   # exact M(1,4) singlet v-tower character

# Flavour coords = the B_GAUGED directions where E (the v generator) has no
# charge; a seed's trace vanishes unless its whole ρ²-orbit can be made
# neutral on these.
_FLAV_COORDS = tuple(j for j in range(len(E_CHARGE)) if E_CHARGE[j] == 0)


class U1OctagonKAlg(ConeKAlgebra):
    """Closed-form u(1)-gauged octagon K-algebra (k=2 of U1A1AoddKAlg).

    Inherits from ConeKAlgebra: multiply via inherited
    `cone_data().derived_multiply`; trace via the inherited Layer-1
    pipeline (ρ²-canonicalisation + universal vanishing check) + the
    `_trace_residual` stub below.

    Standalone — no BPS dependency at runtime.  The MULT_TABLE was
    one-time BPS-extracted (see `u1_octagon_table_gen.py`) and is
    frozen as Python data; the live multiply / trace paths reference
    only the static table + cone_data primitives.
    """

    H = 8  # cyclic order on chord positions

    def __init__(self):
        self._R = TrivialZPlusRing()
        self._cone_data = U1OctaConeData()
        self.k = 2                  # k of the U1A1AoddKAlg family (= [A_1, A_5])
        self._boot_cache = {}       # single certified bootstrap solve (max K)
        self._rep_cache = {}        # ρ²-rep memo for the bootstrap lookup

    # -- KAlgebra contract --

    def coefficient_ring(self):
        return self._R

    def identity(self):
        """Empty cone monomial: no factors, no E."""
        return ((), 0)

    def cone_data(self):
        return self._cone_data

    # `multiply` inherited from ConeKAlgebra (delegates to
    # `self.cone_data().derived_multiply`).

    # Per-family wrap-around E-drift: ρ(L_{a, P_a-1}) = E^{EPS_WRAP[a]}·L_{a, 0}
    # (with ρ(E)=E⁻¹).  Frozen data — derived once from BPS (the gauged-A_5
    # quiver) and validated 0-mismatch vs BPS over single chords incl. e_E≠0
    # and exp=2, consistent with this module's own `chord_charge` /
    # `PRIMITIVE_FAMILY_PERIOD` (which differ from the bootstrap on family 3).
    EPS_WRAP = {1: -2, 2: 0, 3: -2}

    def rho(self, label):
        """ρ on a cone label `(factors, e_E)`: shift each chord
        `(a, i) → (a, i+1 mod P_a)`, accumulate the wrap E-drift at the last
        orbit element, and negate the E-power (`ρ(E)=E⁻¹`).

        The wrap drift + E-inversion are essential: the previous version
        (`(i+1)%P`, `e_E` unchanged) dropped both, so ρ disagreed with BPS on
        the wrap letters and every `e_E≠0` label — failing
        `verify_rho_is_automorphism` and corrupting the multi-letter trace
        (the same long-chord ρ/trace defect fixed in `U1HexagonKAlg`)."""
        factors, e_E = label
        new_factors = []
        drift = 0
        for (a, i, e) in factors:
            if i == PRIMITIVE_FAMILY_PERIOD[a] - 1:
                drift += e * self.EPS_WRAP[a]
            new_factors.append((a, (i + 1) % PRIMITIVE_FAMILY_PERIOD[a], e))
        return (tuple(sorted(new_factors)), -e_E + drift)

    def rho_inverse(self, label):
        """Inverse of `rho`: shift `i → i-1 mod P_a`, accumulate the wrap drift
        when leaving the first orbit element, and invert the E-power."""
        factors, e_E = label
        new_factors = []
        drift = 0
        for (a, i, e) in factors:
            if i == 0:
                drift += e * self.EPS_WRAP[a]
            new_factors.append((a, (i - 1) % PRIMITIVE_FAMILY_PERIOD[a], e))
        return (tuple(sorted(new_factors)), -(e_E - drift))

    def _mag_charge(self, label):
        """Total magnetic charge: Σ MU_LETTER_QPOWER[(a, i)] · exp."""
        from u1_octagon_cone_data import MU_LETTER_QPOWER
        factors, e_E = label
        return sum(MU_LETTER_QPOWER[(a, i)] * e for (a, i, e) in factors)

    def trace_layer1(self, label):
        """Reduce Tr(L_label) to a dict {trace_seed_label: LaurentPoly}.

        Preflight: if magnetic charge ≠ 0, Tr = 0 by u(1)-gauging
        E-insertion argument.  Otherwise routes through
        cone_data.simplify_trace_via_cone_data (tagged ρ⁻² cyclicity).

        If `label` is a cross-cone product (factors not in a single cone),
        first reduces it via derived_multiply to a sum of cone monomials,
        then traces each.
        """
        if self._mag_charge(label) != 0:
            return {}
        # Check if label is a valid cone monomial (all factors in one cone)
        factors, e_E = label
        letters = [(a, i) for (a, i, exp) in factors for _ in range(exp)]
        gens_set = set(letters)
        cones = self._cone_data.cones()
        is_cone_mono = any(gens_set <= set(c) for c in cones) if gens_set else True
        if not is_cone_mono:
            # Reduce via multiply, then trace each summand linearly.
            # Split label into two halves and multiply.
            half = len(letters) // 2
            left_letters = letters[:half]
            right_letters = letters[half:]
            def letters_to_label(lst, e_E_local):
                from collections import Counter
                ctr = Counter(lst)
                f = tuple(sorted((a, i, e) for (a, i), e in ctr.items()))
                return (f, e_E_local)
            left = letters_to_label(left_letters, e_E)
            right = letters_to_label(right_letters, 0)
            prod = self._cone_data.derived_multiply(left, right)
            out = {}
            for sub_lbl, sub_coef in prod.terms.items():
                sub_trace = self.trace_layer1(sub_lbl)
                for k, v in sub_trace.items():
                    contribution = sub_coef * v
                    if k in out:
                        out[k] = out[k] + contribution
                    else:
                        out[k] = contribution
            return {k: v for k, v in out.items() if not v.is_zero()}
        simplified = self._cone_data.simplify_trace_via_cone_data(self, label)
        return {k: v for k, v in simplified.terms.items() if not v.is_zero()}

    def trace(self, label, K=20):
        """Schur-index trace as an `RPowerSeries` in fq — **BPS-free**.

        Layer-1 cone-data ρ²-cyclicity (`ConeKAlgebra.trace`) reduces every
        trace to v-tower / long-chord seeds (the composite diameter reduces
        onto these), and Layer-2 (`_trace_residual`) evaluates them via the
        exact M(1,4) singlet character + the certified orthonormality
        bootstrap.  Verified seed-by-seed against gauged-A_6 `BPSKAlgebra`
        across the v-tower, the long chord and its powers, and the diameter
        and its powers.

        For the raw Layer-1 cone-monomial decomposition (a dict), call
        `trace_layer1` directly.

        `label` must be a genuine canonical basis element — a single cone
        monomial (all chord letters in one cone).  A cross-cone "label" is a
        *product* of basis elements, not a basis element; `multiply` it first,
        then trace each resulting cone-monomial summand."""
        if not self._is_cone_monomial(label):
            raise ValueError(
                f"U1OctagonKAlg.trace: {label!r} is not a single cone monomial "
                f"(its chord letters span multiple cones), so it is a product "
                f"of basis elements rather than one basis element.  Call "
                f"`multiply` first, then trace each summand.")
        return ConeKAlgebra.trace(self, label, K)

    def _is_cone_monomial(self, label):
        """True iff every chord letter of `label` lies in a common cone — i.e.
        `label` is a genuine canonical basis element (single cone monomial),
        not a cross-cone product."""
        factors, _ = label
        letters = set((a, i) for (a, i, exp) in factors)
        if not letters:
            return True
        return any(letters <= set(c) for c in self._cone_data.cones())

    # ----- KAlgebra._canonical_rho2_orbit_rep override -------------------

    def _canonical_rho2_orbit_rep(self, label):
        """Closed-form ρ²-orbit representative (drift quotient).

        With the wrap-pickup-correct ρ (`ρ(E)=E⁻¹` + wrap E-drift), ρ² is not
        finite-order on a single chord: it shifts `i → i+2 (mod P)` AND drifts
        the E-power, so the orbit runs off to `e = ±∞` (the default orbit walk
        would not terminate).  E-powers are ρ²-fixed; a single chord reduces by
        the drift quotient; multi-gen seeds are left as-is (traced by charge,
        which is ρ²-invariant).  Mirrors `U1SquareKAlg` / the good-branch
        `u1a1aodd_kalg`."""
        factors, e = label
        if len(factors) == 0:
            return ((), e)
        if len(factors) == 1 and factors[0][2] == 1:
            from math import gcd
            a = factors[0][0]
            s = PRIMITIVE_FAMILY_PERIOD[a]
            P = s // gcd(s, 2)
            members, cur = [], label
            for _ in range(P):
                members.append(cur)
                cur = self.rho(self.rho(cur))
            D = cur[1] - e
            if D == 0:
                return min(members, key=lambda m: (m[0][0][1], m[1]))
            istar = min(m[0][0][1] for m in members)
            estar = next(m[1] for m in members if m[0][0][1] == istar)
            return (((a, istar, 1),), estar % abs(D))
        return label

    # ----- Layer-2 trace residual (BPS-free: exact v-tower + bootstrap) ---

    def _seed_charge(self, seed):
        """B_GAUGED charge `γ` of a cone-monomial seed `(factors, e_E)`:
        `e_E·E_CHARGE + Σ exp·chord_charge(a,i)` (length-6 list)."""
        factors, e = seed
        g = [e * c for c in E_CHARGE]
        for (a, i, ex) in factors:
            c = chord_charge(a, i)
            for j in range(len(E_CHARGE)):
                g[j] += ex * c[j]
        return g

    def _orbit_has_physical(self, seed):
        """`True` iff the ρ²-orbit of `seed` contains a flavour-neutral
        member (zero charge on `_FLAV_COORDS`) — i.e. `Tr ≠ 0`.  ρ²
        rotates the quiver, so reading the flavour charge off a single rep
        is not ρ²-invariant; the whole-orbit test is."""
        factors = seed[0]
        if not factors:
            return True
        P = PRIMITIVE_FAMILY_PERIOD[factors[0][0]]
        cur = seed
        for _ in range(2 * P + 2):
            g = self._seed_charge(cur)
            if not any(g[j] for j in _FLAV_COORDS):
                return True
            cur = self.rho(self.rho(cur))
        return False

    def _lp_to_rps(self, lp, K):
        """LaurentPoly (integer fq-series) → RPowerSeries, truncated to [0, K]."""
        return RPowerSeries(
            self._R, {e: c for e, c in lp._coeffs.items() if 0 <= e <= K}, K)

    def _boot(self, K, nmax=3):
        """Certified orthonormality-bootstrap solve, cached at the largest
        (K, nmax) requested (smaller reuse it — the caller truncates).

        Trusts ONLY the exact M(1,4) singlet v-tower as anchor and solves the
        long chord (type 2) from orthonormality — BPS-free, no conjectural
        closed form.  `solve_intermediate` raises if under-determined or
        non-integer, so the result is certified exact (never silently wrong).
        Type 3 (the composite diameter) auto-filters as orbit-vanishing.

        `nmax` is the gauge half-width of the pool (default 3 for ordinary
        traces; ungauging widens it on demand to reach `Tr(L·E^n)` for deep n)."""
        from u1aodd_trace_bootstrap import solve_intermediate
        curK = self._boot_cache.get("K", -1)
        curN = self._boot_cache.get("nmax", -1)
        if K > curK or nmax > curN:
            useK, useN = max(K, curK), max(nmax, curN, 3)
            self._boot_cache["Tr"] = solve_intermediate(
                self, useK, unknown_types={2, 3}, nmax=useN)
            self._boot_cache["K"] = useK
            self._boot_cache["nmax"] = useN
        return self._boot_cache["Tr"]

    def _trace_residual(self, seed_label, K):
        """Layer-2 trace of one Layer-1 seed — BPS-free, certified.

          * vanishing (flavour-charged whole orbit) → 0;
          * v-tower `E^e` → the exact M(1,4) singlet character `tr_v_n(e)`;
          * long chord (type 2) → the certified orthonormality bootstrap.

        The chord branch widens the bootstrap's gauge half-width to cover the
        seed's gauge, so a chord absent from the (certified) result is exactly
        "trace 0 through q^K" → returns 0; only a non-reduced multi-gen physical
        seed honest-fails."""
        from u1aodd_trace_bootstrap import _rho2_rep
        if not self._orbit_has_physical(seed_label):
            return RPowerSeries(self._R, {}, K)
        factors, e = seed_label
        g0 = self._seed_charge(seed_label)[0]
        if not factors:
            return self._lp_to_rps(_sing.tr_v_n(g0, K), K)      # v-tower: n = g0
        if len(factors) == 1 and factors[0][2] == 1:
            Tr = self._boot(K, nmax=abs(e) + 1)                 # cover this gauge
            rep = _rho2_rep(self, seed_label, self._rep_cache)
            ser = Tr.get(rep, {})        # in-pool (|gauge|<=nmax); absent ⇒ 0 through K
            return self._lp_to_rps(
                LaurentPoly({q: c for q, c in ser.items() if 0 <= q <= K}), K)
        raise NotImplementedError(
            f"U1OctagonKAlg._trace_residual: physical seed {seed_label!r} "
            f"(g0={g0}) is neither a v-tower nor a single chord — Layer-1 "
            f"did not reduce it.  Honest-fail rather than return silently wrong."
        )

    def verify_associativity(self, sample_labels=None):
        """Self-consistency: check (a·b)·c = a·(b·c) on sample triples.
        Returns (n_ok, n_bad, list_of_failures)."""
        if sample_labels is None:
            sample_labels = [(((a, 0, 1),), 0) for a in (1, 2, 3)] + [
                (((1, 1, 1),), 0), (((2, 1, 1),), 0),
            ]
        n_ok = n_bad = 0
        fails = []
        for a in sample_labels:
            for b in sample_labels:
                for c in sample_labels:
                    ab = self.multiply(a, b)
                    left = Element({})
                    for lab, coef in ab.terms.items():
                        sub = self.multiply(lab, c)
                        for lo, co in sub.terms.items():
                            left.terms[lo] = left.terms.get(lo, LaurentPoly({})) + coef * co
                    bc = self.multiply(b, c)
                    right = Element({})
                    for lab, coef in bc.terms.items():
                        sub = self.multiply(a, lab)
                        for lo, co in sub.terms.items():
                            right.terms[lo] = right.terms.get(lo, LaurentPoly({})) + coef * co
                    L = {k: v for k, v in left.terms.items() if not v.is_zero()}
                    R = {k: v for k, v in right.terms.items() if not v.is_zero()}
                    if L == R:
                        n_ok += 1
                    else:
                        n_bad += 1
                        if len(fails) < 2:
                            fails.append((a, b, c, L, R))
        return n_ok, n_bad, fails


# -----------------------------------------------------------------------------
# Verification: MULT_TABLE matches BPS multiply on all 256 chord pairs
# -----------------------------------------------------------------------------

# Backward-compat alias (class renamed U1OctaKAlg → U1OctagonKAlg
# for consistency with U1HexagonKAlg / U1DecagonKAlg naming).
U1OctaKAlg = U1OctagonKAlg


def verify_against_bps(verbose=True):
    """Confirm the closed-form MULT_TABLE matches BPSKAlgebra on every
    chord-pair product."""
    from u1a1aodd_k2 import U1A1Aodd_k2
    A_bps = U1A1Aodd_k2()
    bps = A_bps._bps
    A_oct = U1OctaKAlg()
    one = LaurentPoly.one()

    primitive_gens = [(a, i) for a in (1, 2, 3)
                      for i in range(PRIMITIVE_FAMILY_PERIOD[a])]
    n_ok = n_bad = 0
    for g1 in primitive_gens:
        for g2 in primitive_gens:
            # U1Octa side: single-letter labels
            lbl_a = (((g1[0], g1[1], 1),), 0)
            lbl_b = (((g2[0], g2[1], 1),), 0)
            oct_result = A_oct.multiply(lbl_a, lbl_b)
            # Convert oct_result labels → BPS charges and compare with bps.multiply
            oct_as_bps = {}
            for label, qcoef in oct_result.terms.items():
                factors, e_E = label
                chg = [e_E * c for c in E_CHARGE]
                for (a, i, e) in factors:
                    fc = chord_charge(a, i)
                    for k in range(6): chg[k] += e * fc[k]
                chg = tuple(chg)
                if chg in oct_as_bps:
                    oct_as_bps[chg] = oct_as_bps[chg] + qcoef
                else:
                    oct_as_bps[chg] = qcoef
            bps_prod = bps.multiply(chord_charge(*g1), chord_charge(*g2))
            bps_dict = {chg: qcoef for chg, qcoef in bps_prod.terms.items()}
            if oct_as_bps == bps_dict:
                n_ok += 1
            else:
                n_bad += 1
                if verbose and n_bad <= 3:
                    print(f"  MISMATCH L_{g1}·L_{g2}:")
                    print(f"    octa: {oct_as_bps}")
                    print(f"    bps:  {bps_dict}")
    print(f"Chord-pair multiply via MULT_TABLE: {n_ok} match, {n_bad} mismatch (out of {len(primitive_gens)**2})")
    return n_bad == 0


if __name__ == "__main__":
    A = U1OctaKAlg()
    print(f"U1OctaKAlg (k=2): coefficient_ring = {A.coefficient_ring()}, H = {A.H}")
    print(f"  identity = {A.identity()}")
    print()
    print("Sample multiplies (compound labels exercised):")
    samples = [
        ((((1, 0, 1),), 0), (((1, 0, 1),), 0)),  # L_1(0) · L_1(0) (cone-mate)
        ((((1, 0, 1),), 0), (((1, 1, 1),), 0)),  # L_1(0) · L_1(1) (Plücker)
        ((((2, 0, 1),), 0), (((1, 3, 1),), 0)),  # L_2(0) · L_1(3)
        ((((1, 0, 1), (2, 0, 1)), 0), (((1, 1, 1),), 0)),  # (L_1(0)·L_2(0)) · L_1(1) — COMPOUND
    ]
    for a, b in samples:
        result = A.multiply(a, b)
        print(f"  {a} · {b}:")
        for label, coef in result.terms.items():
            print(f"    {label}: {coef}")
    print()
    verify_against_bps()
    print()
    print("Trace tests:")
    for label, name in [
        ((((1, 0, 1),), 0), 'L_1(0) — mag=-1 → 0'),
        ((((2, 0, 1),), 0), 'L_2(0) — trace seed'),
        ((((1, 0, 1), (1, 1, 1)), 0), 'L_1(0)·L_1(1) — cross-cone'),
        ((((2, 0, 1), (2, 4, 1)), 0), 'L_2(0)·L_2(4) — q-commute'),
    ]:
        tr = A.trace_layer1(label)
        print(f"  Tr({name}): n_terms={len(tr)}")
        for k, v in list(tr.items())[:3]:
            print(f"    {k}: {v}")
    print()
    n_ok, n_bad, fails = A.verify_associativity()
    print(f"Associativity self-check: {n_ok} pass, {n_bad} fail")
