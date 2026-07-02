"""
u1_decagon_kalg.py
==================

U1DecagonKAlg — u(1)-gauged [A_1, A_7] AD theory, k=3 of the
U1A1AoddKAlg family.  Closed-form via FiniteConeData on the
30-chord-generator + E^± primitive set (32 mult-gens total).

Mirrors U1OctaKAlg structure.  Multiply via derived_multiply,
trace_layer1 via tagged ρ⁻² cyclicity.
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
from u1a1aodd_k3_chord_charges import (
    chord_charge, E_CHARGE, PRIMITIVE_FAMILY_PERIOD,
)
from u1_decagon_cone_data import U1DecagonConeData, MU_LETTER_QPOWER
import u1_decagon_singlet as _sing   # exact M(1,5) singlet v-tower character

# Flavour coords = the B_GAUGED directions where E (the v generator) has no
# charge; a seed's trace vanishes unless its whole ρ²-orbit can be made
# neutral on these.
_FLAV_COORDS = tuple(j for j in range(len(E_CHARGE)) if E_CHARGE[j] == 0)


class U1DecagonKAlg(ConeKAlgebra):
    """u(1)-gauged decagon K-algebra (k=3 of U1A1AoddKAlg).

    Standalone — no BPS dependency at runtime.  Closed-form via
    FiniteConeData + ConeKAlgebra inheritance: multiply inherited
    from `cone_data().derived_multiply`; trace via Layer-1 reduction.
    """

    H = 10

    def __init__(self):
        self._R = TrivialZPlusRing()
        self._cone_data = U1DecagonConeData()
        self.k = 3                  # k of the U1A1AoddKAlg family (= [A_1, A_7])
        self._boot_cache = {}       # single certified bootstrap solve (max K)
        self._rep_cache = {}        # ρ²-rep memo for the bootstrap lookup

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return ((), 0)

    def cone_data(self):
        return self._cone_data

    # `multiply` inherited from ConeKAlgebra.

    # Per-family wrap-around E-drift: ρ(L_{a, P_a-1}) = E^{EPS_WRAP[a]}·L_{a, 0}
    # (with ρ(E)=E⁻¹).  Frozen data — derived once from BPS (the gauged-A_7
    # quiver) and validated 0-mismatch vs BPS over single chords incl. e_E≠0
    # and exp=2, consistent with this module's `chord_charge` /
    # `PRIMITIVE_FAMILY_PERIOD`.
    EPS_WRAP = {1: -2, 2: 0, 3: -2, 4: -1}

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
        factors, e_E = label
        return sum(MU_LETTER_QPOWER[(a, i)] * e for (a, i, e) in factors)

    def trace_layer1(self, label):
        """Reduce Tr(L_label) to a dict {trace_seed: q-coeff}.
        Preflight: mag_charge ≠ 0 → Tr = 0.  Cross-cone labels
        get multiplied out first then traced linearly."""
        if self._mag_charge(label) != 0:
            return {}
        factors, e_E = label
        letters = [(a, i) for (a, i, exp) in factors for _ in range(exp)]
        gens_set = set(letters)
        cones = self._cone_data.cones()
        is_cone_mono = any(gens_set <= set(c) for c in cones) if gens_set else True
        if not is_cone_mono:
            half = len(letters) // 2
            left_letters = letters[:half]
            right_letters = letters[half:]
            def to_label(lst, e_E_local):
                from collections import Counter
                ctr = Counter(lst)
                f = tuple(sorted((a, i, e) for (a, i), e in ctr.items()))
                return (f, e_E_local)
            prod = self._cone_data.derived_multiply(
                to_label(left_letters, e_E), to_label(right_letters, 0))
            out = {}
            for sub_lbl, sub_coef in prod.terms.items():
                sub_trace = self.trace_layer1(sub_lbl)
                for k, v in sub_trace.items():
                    contribution = sub_coef * v
                    if k in out: out[k] = out[k] + contribution
                    else: out[k] = contribution
            return {k: v for k, v in out.items() if not v.is_zero()}
        simplified = self._cone_data.simplify_trace_via_cone_data(self, label)
        return {k: v for k, v in simplified.terms.items() if not v.is_zero()}

    def trace(self, label, K=20):
        """Schur-index trace as an `RPowerSeries` in fq — **BPS-free**.

        Layer-1 cone-data ρ²-cyclicity (`ConeKAlgebra.trace`) reduces every
        trace to v-tower / chord seeds, and Layer-2 (`_trace_residual`)
        evaluates them via the exact M(1,5) singlet character + the certified
        orthonormality bootstrap.  Verified seed-by-seed against gauged-A_8
        `BPSKAlgebra`.

        `label` must be a single cone monomial (canonical basis element); a
        cross-cone product is not a basis element — `multiply` it first, then
        trace each summand.  For the raw Layer-1 dict, call `trace_layer1`."""
        if not self._is_cone_monomial(label):
            raise ValueError(
                f"U1DecagonKAlg.trace: {label!r} is not a single cone monomial "
                f"(its chord letters span multiple cones), so it is a product "
                f"of basis elements rather than one basis element.  Call "
                f"`multiply` first, then trace each summand.")
        return ConeKAlgebra.trace(self, label, K)

    def _is_cone_monomial(self, label):
        """True iff every chord letter of `label` lies in a common cone — i.e.
        `label` is a genuine canonical basis element (single cone monomial)."""
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
        the E-power, so the orbit runs off to `e = ±∞`.  E-powers are ρ²-fixed;
        a single chord reduces by the drift quotient; multi-gen seeds are left
        as-is (traced by charge, ρ²-invariant).  Mirrors `U1SquareKAlg`."""
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
        `e_E·E_CHARGE + Σ exp·chord_charge(a,i)`."""
        factors, e = seed
        g = [e * c for c in E_CHARGE]
        for (a, i, ex) in factors:
            c = chord_charge(a, i)
            for j in range(len(E_CHARGE)):
                g[j] += ex * c[j]
        return g

    def _orbit_has_physical(self, seed):
        """`True` iff the ρ²-orbit of `seed` contains a flavour-neutral member
        (zero charge on `_FLAV_COORDS`) — i.e. `Tr ≠ 0`."""
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
        return RPowerSeries(
            self._R, {e: c for e, c in lp._coeffs.items() if 0 <= e <= K}, K)

    def _boot(self, K, nmax=3):
        """Certified orthonormality-bootstrap solve, cached at the largest
        (K, nmax) requested.  Trusts ONLY the exact M(1,5) singlet v-tower and
        solves the long (type 2), intermediate (type 3), and diameter (type 4)
        chords from orthonormality — BPS-free, no conjectural closed form.
        Raises if under-determined / non-integer.

        `nmax` is the gauge half-width of the pool (default 3 for ordinary
        traces; ungauging widens it on demand to reach deep `Tr(L·E^n)`)."""
        from u1aodd_trace_bootstrap import solve_intermediate
        curK = self._boot_cache.get("K", -1)
        curN = self._boot_cache.get("nmax", -1)
        if K > curK or nmax > curN:
            useK, useN = max(K, curK), max(nmax, curN, 3)
            self._boot_cache["Tr"] = solve_intermediate(
                self, useK, unknown_types={2, 3, 4}, nmax=useN)
            self._boot_cache["K"] = useK
            self._boot_cache["nmax"] = useN
        return self._boot_cache["Tr"]

    def _trace_residual(self, seed_label, K):
        """Layer-2 trace of one Layer-1 seed — BPS-free, certified.

          * vanishing (flavour-charged whole orbit) → 0;
          * v-tower `E^e` → the exact M(1,5) singlet character `tr_v_n(e)`;
          * single physical chord → the certified orthonormality bootstrap.

        The chord branch widens the bootstrap's gauge half-width to cover the
        seed's gauge, so a chord absent from the (certified) result is "trace 0
        through q^K" → returns 0; only a non-reduced multi-gen physical seed
        raises (rather than silently degrading)."""
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
            f"U1DecagonKAlg._trace_residual: physical seed {seed_label!r} "
            f"(g0={g0}) is neither a v-tower nor a single chord — Layer-1 did "
            f"not reduce it.  Honest-fail rather than return silently wrong.")


# -----------------------------------------------------------------------------
# Verification: derived_multiply matches BPS on all 900 chord pairs
# -----------------------------------------------------------------------------

def verify_against_bps():
    from u1a1aodd_k3_chord_charges import _BPS, FAMILY_PERIOD
    A = U1DecagonKAlg()
    bps = _BPS

    gens = [(a, i) for a in (1, 2, 3, 4) for i in range(FAMILY_PERIOD[a])]
    n_ok = n_bad = 0
    for g1 in gens:
        for g2 in gens:
            lbl_a = (((g1[0], g1[1], 1),), 0)
            lbl_b = (((g2[0], g2[1], 1),), 0)
            oct_result = A._cone_data.derived_multiply(lbl_a, lbl_b)
            # Sum contributions per BPS charge
            oct_chgs = {}
            for label, qcoef in oct_result.terms.items():
                factors, e_E = label
                chg = [e_E * c for c in E_CHARGE]
                for (a, i, e) in factors:
                    fc = chord_charge(a, i)
                    for k in range(8): chg[k] += e * fc[k]
                chg = tuple(chg)
                oct_chgs[chg] = oct_chgs.get(chg, LaurentPoly({})) + qcoef
            bps_dict = dict(bps.multiply(chord_charge(*g1), chord_charge(*g2)).terms)
            if oct_chgs == bps_dict:
                n_ok += 1
            else:
                n_bad += 1
    print(f"derived_multiply: {n_ok}/{len(gens)**2} match BPS ({n_bad} mismatch)")
    return n_bad == 0


if __name__ == "__main__":
    A = U1DecagonKAlg()
    print(f"U1DecagonKAlg: 37 mult-gens (L_1×10 + L_2×10 + L_3×10 + L_4×5 + E^±), H = {A.H}")
    print()
    print("Sample multiplies:")
    samples = [
        ((((1, 0, 1),), 0), (((1, 0, 1),), 0)),
        ((((1, 0, 1),), 0), (((1, 1, 1),), 0)),
        ((((1, 0, 1),), 0), (((2, 1, 1),), 0)),
    ]
    for a, b in samples:
        result = A.multiply(a, b)
        print(f"  {a} · {b}:")
        for lab, coef in result.terms.items():
            print(f"    {lab}: {coef}")
    print()
    verify_against_bps()
    print()
    print("Trace tests:")
    for label, name in [
        ((((1, 0, 1),), 0), 'L_1(0) — mag=-1'),
        ((((2, 0, 1),), 0), 'L_2(0) — mag=0'),
        ((((1, 0, 1), (1, 1, 1)), 0), 'L_1(0)·L_1(1) — mag=0'),
        ((((2, 0, 1), (2, 5, 1)), 0), 'L_2(0)·L_2(5)'),
    ]:
        try:
            tr = A.trace_layer1(label)
            print(f"  Tr({name}): n_terms={len(tr)}")
            for k, v in list(tr.items())[:3]:
                print(f"    {k}: {v}")
        except Exception as e:
            print(f"  Tr({name}): ERROR {type(e).__name__}: {str(e)[:100]}")
