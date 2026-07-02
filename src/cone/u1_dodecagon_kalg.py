"""
u1_dodecagon_kalg.py
====================

`U1DodecagonKAlg` — u(1)-gauged [A_1, A_9] AD theory, k=4 of the even-polygon
family.  **Standalone** (mirrors `U1OctagonKAlg` / `U1DecagonKAlg`): a
self-contained `ConeKAlgebra` over its OWN frozen data files
(`u1a1aodd_k4_chord_charges`, `u1_dodecagon_mult_table`,
`u1_dodecagon_cone_data`) — NO BPS and NO canonical/RG-oracle import at
runtime.

Trace (BPS-free, certified): Layer-1 cone-data ρ²-cyclicity reduces every
trace to v-tower / chord seeds; Layer-2 evaluates the v-tower via the exact
M(1,6) singlet character and every physical chord (types 2 and 4) via the
certified orthonormality bootstrap (trusting ONLY the v-tower).  At k=4 the
physical chord types are 2 and 4; types 1/3/5 — incl. the diameter type 5 —
are orbit-vanishing, so no conjectural cf_diameter is needed.
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
from u1a1aodd_k4_chord_charges import (
    chord_charge, E_CHARGE, PRIMITIVE_FAMILY_PERIOD, EPS_WRAP,
)
from u1_dodecagon_cone_data import U1DodecagonConeData, MU_LETTER_QPOWER
import u1_pgon_layer2 as _gp        # general-p (p=6) exact singlet v-tower character

# Flavour coords = the directions where E (the v generator) has no charge.
_FLAV_COORDS = tuple(j for j in range(len(E_CHARGE)) if E_CHARGE[j] == 0)
_P = 6                              # p = k + 2 = 6  (M(1, 6) singlet)


class U1DodecagonKAlg(ConeKAlgebra):
    """Standalone u(1)-gauged dodecagon K-algebra (k=4 of the even family)."""

    H = 12

    # Physical (trace-nonzero) chord types at k=4: the long chord (type 2) and
    # the surviving intermediate (type 4).  Types 1/3/5 are orbit-vanishing.
    _UNKNOWN_TYPES = frozenset({2, 4})

    def __init__(self):
        self._R = TrivialZPlusRing()
        self._cone_data = U1DodecagonConeData()
        self.k = 4                  # k of the even-polygon family ([A_1, A_9])
        self._boot_cache = {}       # single certified bootstrap solve (max K)
        self._rep_cache = {}        # ρ²-rep memo for the bootstrap lookup

    # -- KAlgebra contract --

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return ((), 0)

    def cone_data(self):
        return self._cone_data

    # `multiply` inherited from ConeKAlgebra (delegates to derived_multiply).

    def rho(self, label):
        """ρ on a cone label `(factors, e_E)`: shift each chord
        `(a, i) → (a, i+1 mod P_a)`, accumulate the wrap E-drift at the last
        orbit element, and negate the E-power (`ρ(E)=E⁻¹`)."""
        factors, e_E = label
        new_factors = []
        drift = 0
        for (a, i, e) in factors:
            if i == PRIMITIVE_FAMILY_PERIOD[a] - 1:
                drift += e * EPS_WRAP[a]
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
                drift += e * EPS_WRAP[a]
            new_factors.append((a, (i - 1) % PRIMITIVE_FAMILY_PERIOD[a], e))
        return (tuple(sorted(new_factors)), -(e_E - drift))

    def _mag_charge(self, label):
        factors, e_E = label
        return sum(MU_LETTER_QPOWER[(a, i)] * e for (a, i, e) in factors)

    def trace_layer1(self, label):
        """Reduce Tr(L_label) to a dict {trace_seed: q-coeff}.
        Preflight: mag_charge ≠ 0 → Tr = 0.  Cross-cone labels are multiplied
        out first then traced linearly."""
        if self._mag_charge(label) != 0:
            return {}
        factors, e_E = label
        letters = [(a, i) for (a, i, exp) in factors for _ in range(exp)]
        is_cone_mono = self._is_cone_monomial(label)
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
                for kk, v in sub_trace.items():
                    contribution = sub_coef * v
                    out[kk] = out[kk] + contribution if kk in out else contribution
            return {kk: v for kk, v in out.items() if not v.is_zero()}
        simplified = self._cone_data.simplify_trace_via_cone_data(self, label)
        return {kk: v for kk, v in simplified.terms.items() if not v.is_zero()}

    def trace(self, label, K=20):
        """Schur-index trace as an `RPowerSeries` in fq — **BPS-free**.

        Layer-1 cone ρ²-cyclicity (`ConeKAlgebra.trace`) → v-tower / chord
        seeds; Layer-2 (`_trace_residual`) → exact M(1,6) v-tower + certified
        bootstrap.  `label` must be a single cone monomial (basis element); a
        cross-cone product raises (multiply it first).  Raw Layer-1 dict:
        `trace_layer1`."""
        if not self._is_cone_monomial(label):
            raise ValueError(
                f"U1DodecagonKAlg.trace: {label!r} is not a single cone monomial "
                f"(its chord letters span multiple cones).  Call `multiply` "
                f"first, then trace each summand.")
        return ConeKAlgebra.trace(self, label, K)

    def _is_cone_monomial(self, label):
        """True iff the label's chord letters pairwise q-commute (so they form
        a single cone monomial / basis element).  O(letters²) via q_commute —
        avoids scanning the 16796 maximal cones."""
        factors, _ = label
        letters = sorted(set((a, i) for (a, i, exp) in factors))
        cd = self._cone_data
        for x in range(len(letters)):
            for y in range(x + 1, len(letters)):
                if not cd.q_commute(letters[x], letters[y]):
                    return False
        return True

    # ----- ρ²-orbit canonicalisation (closed-form drift quotient) --------

    def _canonical_rho2_orbit_rep(self, label):
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
        """B_GAUGED charge of a cone-monomial seed `(factors, e_E)`:
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
        Pp = PRIMITIVE_FAMILY_PERIOD[factors[0][0]]
        cur = seed
        for _ in range(2 * Pp + 2):
            g = self._seed_charge(cur)
            if not any(g[j] for j in _FLAV_COORDS):
                return True
            cur = self.rho(self.rho(cur))
        return False

    def _lp_to_rps(self, lp, K):
        return RPowerSeries(
            self._R, {e: c for e, c in lp._coeffs.items() if 0 <= e <= K}, K)

    def _cf_v(self, m, K):
        """Tr(v^m) — exact M(1,6) singlet character (general p = k+2 = 6)."""
        return self._lp_to_rps(_gp.tr_v_n(_P, m, K), K)

    def _boot(self, K, nmax=3):
        """Certified orthonormality-bootstrap solve over both physical chord
        types (2 and 4), cached at the largest (K, nmax) requested; trusts ONLY
        the exact M(1,6) v-tower.  Raises if under-determined / non-integer.

        `nmax` is the gauge half-width of the pool (default 3 for ordinary
        traces; ungauging widens it on demand to reach deep `Tr(L·E^n)`)."""
        from u1aodd_trace_bootstrap import solve_intermediate
        curK = self._boot_cache.get("K", -1)
        curN = self._boot_cache.get("nmax", -1)
        if K > curK or nmax > curN:
            useK, useN = max(K, curK), max(nmax, curN, 3)
            self._boot_cache["Tr"] = solve_intermediate(
                self, useK, unknown_types=self._UNKNOWN_TYPES, nmax=useN)
            self._boot_cache["K"] = useK
            self._boot_cache["nmax"] = useN
        return self._boot_cache["Tr"]

    def _trace_residual(self, seed_label, K):
        """Layer-2 trace of one Layer-1 seed — BPS-free, certified.

          * vanishing → 0;
          * v-tower E^e → exact M(1,6) singlet character;
          * single physical chord (type 2 / 4) → certified bootstrap.

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
            return self._cf_v(g0, K)                            # v-tower: n = g0
        if len(factors) == 1 and factors[0][2] == 1:
            Tr = self._boot(K, nmax=abs(e) + 1)                 # cover this gauge
            rep = _rho2_rep(self, seed_label, self._rep_cache)
            ser = Tr.get(rep, {})        # in-pool (|gauge|<=nmax); absent ⇒ 0 through K
            return self._lp_to_rps(
                LaurentPoly({q: c for q, c in ser.items() if 0 <= q <= K}), K)
        raise NotImplementedError(
            f"U1DodecagonKAlg._trace_residual: physical seed {seed_label!r} "
            f"(g0={g0}) is neither a v-tower nor a single chord — Layer-1 did "
            f"not reduce it.")


# Backward-compat: the previous (canonical-subclass) U1DodecagonKAlg is
# superseded by this standalone class.


def verify_against_canonical(K=6, nmax_chords=10):
    """Confirm derived_multiply matches the canonical U1A1AoddKAlg(4) by charge
    on a sample of chord pairs (uses the canonical only as a test oracle)."""
    from u1a1aodd_kalg import U1A1AoddKAlg
    A = U1DodecagonKAlg()
    C = U1A1AoddKAlg(4)
    cd = C.cone_data()
    MU = tuple(cd._MU)

    def can_charge(label):
        factors, e = label
        g = [e * c for c in MU]
        for (a, i, ex) in factors:
            c = cd._chg[(a, i)]
            for j in range(len(MU)):
                g[j] += ex * c[j]
        return tuple(g)

    gens = A.cone_data()._chords[:nmax_chords]
    n_ok = n_bad = 0
    for g1 in gens:
        for g2 in gens:
            mine = {}
            for lbl, coef in A.multiply((((g1[0], g1[1], 1),), 0),
                                        (((g2[0], g2[1], 1),), 0)).terms.items():
                ch = tuple(A._seed_charge(lbl))
                mine[ch] = mine.get(ch, LaurentPoly({})) + coef
            mine = {k: v for k, v in mine.items() if not v.is_zero()}
            canon = {}
            for lbl, coef in C.multiply(C.L(*g1), C.L(*g2)).terms.items():
                ch = can_charge(lbl)
                canon[ch] = canon.get(ch, LaurentPoly({})) + coef
            canon = {k: v for k, v in canon.items() if not v.is_zero()}
            keys = set(mine) | set(canon)
            if all(str(mine.get(k, LaurentPoly({}))) == str(canon.get(k, LaurentPoly({})))
                   for k in keys):
                n_ok += 1
            else:
                n_bad += 1
    print(f"derived_multiply vs canonical (by charge): {n_ok} match, {n_bad} differ")
    return n_bad == 0


if __name__ == "__main__":
    A = U1DodecagonKAlg()
    cd = A.cone_data()
    from collections import Counter
    phys = [(a, i) for (a, i) in cd._chords
            if A._orbit_has_physical((((a, i, 1),), 0))]
    print(f"U1DodecagonKAlg (k=4, [A_1, A_9]): {len(cd._chords)} chords, "
          f"{len(cd.cones())} cones, {len(phys)} physical "
          f"(types {sorted(set(a for a, i in phys))})")
    print("  Tr(v^1) (exact M(1,6)) =", dict(A.trace(((), 1), K=12).coeffs))
    verify_against_canonical()
