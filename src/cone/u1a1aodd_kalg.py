"""
u1a1aodd_kalg.py
================

`U1A1AoddKAlg(k)` — the gauged `[A_1, A_{2k+1}]` family as a closed-form
`ConeKAlgebra`.  Its cone tables were extracted once from a validated
RG-flow oracle (`U1A1AoddGaugedRG(k)`, a derivation **not included in
this repository**) and are stored frozen in `u1a1aodd_tables_k{k}.pkl`
(the analogue of `A1A2kKAlg` computing its base table at construction).
At runtime there is no per-multiply RG solve — `multiply` is the
generic cone-monomial reducer over:

**Cyclicity-reduced build.**  The q-commute / cross_product table is
ρ_UV-invariant, so the oracle is consulted only on the **L-shape
representatives** (`min(i,j)=0`) of each chord-type pair — every other
position is lifted by the **hard-coded ρ_UV** (`_rho_label_cone`: chord
position `i→i+1` with the single wrap E-drift, `E→E⁻¹`; a closed-form
label map verified against the oracle).  This is the cyclicity-driven
speedup: oracle `multiply` calls drop from `N²` to ≈`Σ(size_A+size_B−1)`
(k=1,2,3: 1.7×/2.5×/3.4× fewer, growing with k).  `multiply` is the
generic cone-monomial reducer over:

  * **mult-generators**: the chord rays (the irreducible ρ_UV-orbits)
    + the gauge generator `E` as a `QTCone` Laurent direction;
  * **cocycle** = the `A_{2k+2}` **chain pairing** of charges (geometric
    chord-crossing form), verified on every q-commuting pair;
  * **cross_product** = the 2-term **Plücker** (cluster exchange),
    ρ-reduced, read off the oracle.

The chords (= the `(2k+4)`-gon diagonals / cluster variables) are
selected from the oracle by `u1a1aodd_mult_table.select_chords` (the
irreducible ρ_UV-orbits, via charge-guided factoring).

Structure and validation (general k):
  * Builds: `(k+2)(2k+1)` chords + `E^±`; cones = Catalan(2k+2)
    triangulations of the `(2k+4)`-gon (k=1: 14, k=2: 132).
  * **Cocycle = chain pairing** (closed form); cross_product extracted
    from the oracle (`c_lit = c_can + cone_label_phase`, with the
    daughter word sorted into canonical cone order — an unsorted word
    would double-count `_sort_within_cone`'s cocycle swaps and break
    the Plückers).
  * **Bar-invariant**, **associative**, unital, `E·E⁻¹ = 𝟙`, and
    `multiply` matches the oracle **exactly** on all chord pairs —
    verified k=1 (81/81) and k=2 (400/400).  k=1 = `U1HexagonKAlg`'s
    L1/L2 mod E.
  * Build is fast (oracle-extracted once at construction, ρ-reduced:
    ~0.2s k=1, ~0.7s k=2, ~2.3s k=3).
  * **ρ_UV is hard-coded** (`rho`/`rho_inverse`): closed-form label maps,
    matched to the oracle on chords + q-commuting monomials (k=1,2,3).
  * **trace** (Schur index): Layer-1 ρ²-cyclicity over the cones reduces
    any trace to **low-charge seeds** (v-tower / single chords), whose
    values are the **closed-form M(1, p) singlet characters** of
    `u1_pgon_layer2` (`_trace_residual` → `_cf_v` / `_cf_Llong` /
    `_cf_diameter`; an unreduced seed raises rather than silently
    falling back).  Verified seed-by-seed against the oracle's
    **adaptive-window** trace (the generic fixed-`_rg_cutoff` trace is
    unreliable on high-charge — its FS object `RG(a)·S_RG` is
    under-resolved; the oracle's trace overrides it with an
    adaptive `S_RG` window + two-cutoff stability) with
    `oracle_residual_calls=0` in the k=1..3 sweeps.  ρ²-invariant,
    K-stable; gets the squared **wrap** chord right where the direct
    oracle trace cannot.
  * **geometric chord labels**: chord `(a,i)` = a `(2k+4)`-gon diagonal of
    length `a+1` (ρ_UV = rotate-by-one); the cocycle is the crossing form
    — `q_commute ⟺ non-crossing` (`geometric_label`,
    `verify_qcommute_is_noncrossing`).
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from typing import Sequence

from kalgebra import KAlgebra, Element
from cone_kalgebra import ConeKAlgebra
from cone_data import CrossProductTerm, FiniteConeData, Cone
from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing, RPowerSeries

# NOTE: construction loads the pre-extracted cone tables
# (`u1a1aodd_tables_k{k}.pkl`) whenever present, so no oracle is built —
# construction is then spine-free, and `_trace_residual` is closed-form
# (`u1_pgon_layer2`).  The tables were produced by an RG-flow derivation
# (`u1a1aodd_gauged_rg`) that is not included in this repository; `_extract`
# → `select_chords` imports it lazily and runs only where it is available.
# `chain_pairing` imports spine-free (its module's spine imports are lazy too).
from u1a1aodd_mult_table import bijection_charge, chain_pairing, k1_true_rays
import u1_pgon_layer2 as _gp   # general-p (p=k+2) closed-form Layer-2 characters


def _frozen_tables_path(k):
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"u1a1aodd_tables_k{k}.pkl")


def _load_frozen_into(cd, k):
    """Load the pre-extracted cone tables for `k` into `cd` (spine-free); the
    oracle-holding attributes are intentionally absent.  Returns True on hit."""
    import os, pickle
    path = _frozen_tables_path(k)
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        cd.__dict__.update(pickle.load(f))
    return True


def freeze_tables(k):
    """(Re)generate `u1a1aodd_tables_k{k}.pkl` from the oracle (runs only
    where the RG-flow derivation, not included in this repository, is
    available).  Pickles every cone-data attribute except the oracle-holders,
    so later constructions load the freeze and never build the oracle."""
    import pickle
    alg = U1A1AoddKAlg(k)
    cd = alg._cd
    list(cd.cones())                       # populate the lazy cone structure
    drop = {"_alg", "_oracle", "_elt", "_phi", "_aux"}
    frozen = {key: v for key, v in cd.__dict__.items() if key not in drop}
    with open(_frozen_tables_path(k), "wb") as f:
        pickle.dump(frozen, f)
    return _frozen_tables_path(k)


# Mult-gen ids: chord (a, i) with a >= 1; gauge E = (0, 0), E^{-1} = (0, 1).
E_GEN = (0, 0)
E_INV = (0, 1)


def _extract(k):
    """Extract from the oracle, for any k: chord `(a, i)` -> (oracle
    element, `B_GAUGED` charge), via the general-k `select_chords`."""
    from u1a1aodd_mult_table import select_chords
    T, phi, chords = select_chords(k)
    n = 2 * k + 2
    MU = tuple(1 if (j % 2 == 0 and j <= 2 * k) else 0 for j in range(n))
    elt = dict(chords)
    chg = {ai: phi(e) for ai, e in chords.items()}
    return T, elt, chg, phi, MU, n


class U1A1AoddConeData(FiniteConeData):
    """Oracle-extracted `ConeData` for `U1A1AoddKAlg(k)` (k=1 here).

    Native labels are cone monomials `(factors, e_E)` with `factors` a
    sorted tuple of `(a, i, exp)` chord powers and `e_E in Z` the
    `E`-power; `E` is the `QTCone` Laurent direction."""

    def __init__(self, alg: "U1A1AoddKAlg", k: int) -> None:
        self._alg = alg
        self._k = k
        if _load_frozen_into(self, k):    # spine-free: pre-extracted tables
            self._alg = alg
            self._k = k
            return
        T, elt, chg, phi, MU, n = _extract(k)
        self._oracle = T
        self._elt = elt          # (a,i) -> oracle element
        self._chg = chg          # (a,i) -> charge
        self._phi = phi
        self._MU = MU
        self._n = n
        self._chords = sorted(elt)
        self._aux = T.auxiliary()                # hard-coded IR: A1A2k(k) base_table (x) QT
        _bt = {}
        for (a, i) in self._chords:
            _bt.setdefault(a, []).append(i)
        self._types = {a: sorted(iis) for a, iis in _bt.items()}
        self._size = {a: len(iis) for a, iis in _bt.items()}
        self._mult_gens = tuple(self._chords) + (E_GEN, E_INV)
        # charge -> (a,i) for decoding products
        self._by_chg = {tuple(c): ai for ai, c in chg.items()}
        # --- hard-coded ρ_UV data (extracted once) ------------------------
        # Within an orbit `ρ(elt[a,i]) = elt[a,i+1]` exactly (`select_chords`
        # stores the ρ-image directly); the only E-drift is at the single
        # wrap `ρ(elt[a,size-1]) = E^{eps_wrap[a]}·elt[a,0]`, and `ρ(E)=E⁻¹`
        # uniformly.  These constants make ρ_UV a closed-form label map
        # (`_rho_label_cone`) — no oracle call per ρ.
        self._eps_wrap = self._extract_wrap_drift()
        # extract q-commute + cross_product once, ρ-reduced (cyclicity)
        self._qcom = {}      # ((A,i),(B,j)) -> bool        (full table)
        self._rep_q = {}     # (A,B,i0,j0) -> bool          (L-shape reps)
        self._rep_cross = {} # (A,B,i0,j0) -> ((coeff, daughter_cone_label), …)
        self._cross = {}     # ((A,i),(B,j)) -> tuple[CrossProductTerm]
        self._build_tables()
        self._cones = None
        self._mono_cache = None

    # -- hard-coded ρ_UV --------------------------------------------------
    def _extract_wrap_drift(self):
        """Wrap-around E-drift `eps_wrap[a]` and the `ρ(E)=E^{-1}` check —
        the only data ρ_UV needs beyond the orbit sizes."""
        T, n, MU = self._oracle, self._n, self._MU
        eps = {}
        for a, iis in self._types.items():
            r = T.rho(self._elt[(a, iis[-1])])          # ρ of the last orbit elt
            base = self._chg[(a, iis[0])]
            diff = tuple(self._phi(r)[j] - base[j] for j in range(n))
            ms = {diff[j] // MU[j] for j in range(n) if MU[j] != 0}
            if len(ms) != 1 or tuple(next(iter(ms)) * MU[j] for j in range(n)) != diff:
                raise AssertionError(f"wrap drift not a pure E-power: type {a}, {diff}")
            eps[a] = next(iter(ms))
        rE = T.rho(((), (0, 1)))                          # ρ(E_GEN)
        if self._phi(rE) != tuple(-x for x in MU):
            raise AssertionError(f"ρ(E) is not E^-1: {rE}")
        return eps

    @property
    def _mono(self):
        """Monomial-RG positions per chord type (lazy: only `_ir_multiply`,
        the off-path documentation, needs the `T.RG` probes)."""
        if self._mono_cache is None:
            T = self._oracle
            self._mono_cache = {
                a: set(i for i in iis if len(T.RG(self._elt[(a, i)]).terms) == 1)
                for a, iis in self._types.items()}
        return self._mono_cache

    # -- decode an oracle product into cone-monomial daughters ------------
    def _decode(self, charge):
        """charge -> (factors_sorted, e_E) cone monomial: fewest q-commuting
        chord factors + an E-power.  E = MU; q-commuting = compatible."""
        n, MU = self._n, self._MU
        for eE in range(-6, 7):
            gp = tuple(charge[j] - eE * MU[j] for j in range(n))
            if all(x == 0 for x in gp):
                return ((), eE)
            for ai in self._chords:
                if tuple(gp[j] - self._chg[ai][j] for j in range(n)) == (0,) * n:
                    return (((ai[0], ai[1], 1),), eE)
            for a in self._chords:
                for b in self._chords:
                    if a <= b and self._is_qcom(a, b):
                        if tuple(gp[j] - self._chg[a][j] - self._chg[b][j]
                                 for j in range(n)) == (0,) * n:
                            if a == b:
                                return (((a[0], a[1], 2),), eE)
                            return (tuple(sorted([(a[0], a[1], 1), (b[0], b[1], 1)])), eE)
        return None

    def _ir_multiply(self, ai, bj):
        """`L_ai · L_bj` via the HARD-CODED IR multiply (`aux` =
        `A1A2k(k).base_table_predict` ⊗ quantum torus), ρ-shifted so both
        chords sit at monomial-RG positions — no oracle RG-solving:
        `multiply = ρ⁻ᵐ( aux.multiply(ρᵐ a, ρᵐ b) )` (verified == oracle
        on every chord pair, k=1 81/81, k=2 400/400).

        NOTE: this documents that the ray products ARE the hard-coded IR
        mult (copyable) — it is NOT on the build path.  `_build_tables`
        instead reads the oracle on the ρ-reduced L-shape reps and lifts by
        the hard-coded `_rho_label_cone` (cyclicity); this `aux`-at-monomial
        route is kept only as the explicit IR-product derivation."""
        a, i = ai
        b, j = bj
        sa, sb = self._size[a], self._size[b]
        m = next(mm for mm in range(2 * max(sa, sb) + 2)
                 if (i + mm) % sa in self._mono[a] and (j + mm) % sb in self._mono[b])
        T = self._oracle
        x, y = self._elt[ai], self._elt[bj]
        for _ in range(m):
            x = T.rho(x); y = T.rho(y)
        out = self._aux.multiply(x, y)
        res = {}
        for l, c in out.terms.items():
            ll = l
            for _ in range(m):
                ll = T.rho_inverse(ll)
            res[ll] = res.get(ll, LaurentPoly.zero()) + c
        return Element({kk: v for kk, v in res.items() if not v.is_zero()})

    def _rho_label_cone(self, lbl):
        """Hard-coded ρ_UV on a cone label `(factors, e_E)`: shift each chord
        `(a, i) → (a, i+1 mod size_a)`, accumulate the wrap E-drift, and
        negate the E-power (`ρ(E)=E⁻¹`).  A pure label map (no oracle, no
        q-coefficient — ρ permutes the canonical basis)."""
        factors, e_E = lbl
        nf = []
        drift = 0
        for (a, i, exp) in factors:
            s = self._size[a]
            if i == s - 1:
                drift += exp * self._eps_wrap[a]
            nf.append((a, (i + 1) % s, exp))
        return tuple(sorted(nf)), -e_E + drift

    def _rho_inverse_label_cone(self, lbl):
        """Inverse of `_rho_label_cone` (`ρ⁻¹` on cone labels)."""
        factors, e_E = lbl
        nf = []
        drift = 0
        for (a, i, exp) in factors:
            s = self._size[a]
            if i == 0:
                drift += exp * self._eps_wrap[a]
            nf.append((a, (i - 1) % s, exp))
        return tuple(sorted(nf)), -(e_E - drift)

    def _lshape(self, sA, sB):
        """L-shape ρ-orbit reps for a type pair: `{(0,j)} ∪ {(i,0)}` — the
        pairs reached by reducing `min(i,j) → 0` without either input
        wrapping."""
        return sorted(set((0, j) for j in range(sB)) | set((i, 0) for i in range(sA)))

    def _reduce(self, a, b):
        """`((A,i),(B,j)) → (rep_key, t)`: reduce by `t=min(i,j)` ρ-inverse
        steps to an L-shape rep `(A,B,i-t,j-t)`; neither input wraps."""
        (A, i), (B, j) = a, b
        t = min(i, j)
        return (A, B, i - t, j - t), t

    def _qcom_pair(self, a, b):
        return self._rep_q[self._reduce(a, b)[0]]

    def _is_qcom(self, a, b):
        if a == b:
            return True
        return self._qcom_pair(a, b)

    def _cross_terms(self, daughters, t):
        """Lift decoded canonical `daughters` of an L-shape rep by `ρ^t`
        (hard-coded) into the literal `CrossProductTerm`s at the target
        pair: daughter `→ ρᵗ(daughter)`, literal coeff `= c_can + phase(ρᵗ·)`."""
        terms = []
        for (c, dec) in daughters:
            factors, eE = dec
            for _ in range(t):
                factors, eE = self._rho_label_cone((factors, eE))
            wl = [g for (ga, gi, ex) in factors for g in ((ga, gi),) * ex]
            if eE > 0:
                wl += [E_GEN] * eE
            elif eE < 0:
                wl += [E_INV] * (-eE)
            word = tuple(sorted(wl))                       # canonical cone order
            gset, pdict = self.to_cone_label((tuple(factors), eE))
            phase = self.cone_label_phase(gset, pdict)     # c_lit = c_can + phase
            for q, coef in c._coeffs.items():
                terms.append((LaurentPoly({q + phase: coef}), word))
        return tuple(terms)

    def _build_tables(self):
        """Cyclicity-reduced extraction.  The q-commute / cross_product
        table is ρ_UV-invariant, so it is read off the oracle only on the
        L-shape representatives (`min(i,j)=0`) of each type pair and lifted
        to every position by the hard-coded `ρ_UV` (`_rho_label_cone`).
        Cocycle is closed-form (chain pairing), so no backward products."""
        T = self._oracle
        # Pass 1 — oracle products on the L-shape reps; q-commute is the
        # single-term test (ρ-invariant, fills `_rep_q` for `_is_qcom`).
        rep_pr = {}
        for A, iA in self._types.items():
            for B, iB in self._types.items():
                for (i0, j0) in self._lshape(len(iA), len(iB)):
                    pr = T.multiply(self._elt[(A, i0)], self._elt[(B, j0)])
                    rep_pr[(A, B, i0, j0)] = pr
                    self._rep_q[(A, B, i0, j0)] = (len(pr.terms) == 1)
        # Pass 1b — decode the crossing reps to canonical daughters.
        for key, pr in rep_pr.items():
            if self._rep_q[key]:
                continue
            daughters = []
            for l, c in pr.terms.items():
                dec = self._decode(self._phi(l))
                if dec is None:
                    raise RuntimeError(f"cross_product decode failed at rep {key}")
                daughters.append((c, dec))
            self._rep_cross[key] = tuple(daughters)
        # Pass 2 — lift to the full table by ρ_UV-equivariance.
        for a in self._chords:
            for b in self._chords:
                key, t = self._reduce(a, b)
                self._qcom[(a, b)] = self._rep_q[key]
                if not self._rep_q[key]:
                    self._cross[(a, b)] = self._cross_terms(self._rep_cross[key], t)

    # -- FiniteConeData surface -------------------------------------------
    def mult_gens(self) -> Sequence:
        return self._mult_gens

    def cones(self) -> Sequence[frozenset]:
        if self._cones is None:
            V = list(self._mult_gens)
            adj = {v: frozenset(u for u in V if u != v and self.q_commute(v, u)) for v in V}
            out = []
            def bk(R, P, X):
                if not P and not X:
                    out.append(R); return
                piv = max(P | X, key=lambda u: len(P & adj[u]))
                for v in list(P - adj[piv]):
                    bk(R | {v}, P & adj[v], X & adj[v]); P = P - {v}; X = X | {v}
            bk(frozenset(), frozenset(V), frozenset())
            self._cones = tuple(out)
        return self._cones

    def q_commute(self, g, h) -> bool:
        if g == h:
            return True
        if {g, h} == {E_GEN, E_INV}:
            return True
        if g in (E_GEN, E_INV) or h in (E_GEN, E_INV):
            return True   # E q-commutes (Laurent torus) with everything
        return self._qcom[(g, h)]

    def cocycle(self, g, h) -> int:
        """Half-exponent `c` with `L_g L_h = q^{2c} L_h L_g`.  Uniformly the
        `A_{2k+2}` chain pairing of charges (verified `c_fwd-c_bwd =
        2·chain_pairing` on every q-commuting chord pair); `E` carries
        charge `+MU` (`E_GEN`) / `-MU` (`E_INV`)."""
        if g == h or {g, h} == {E_GEN, E_INV}:
            return 0
        return chain_pairing(self._charge(g), self._charge(h))

    def _charge(self, g):
        if g == E_GEN:
            return self._MU
        if g == E_INV:
            return tuple(-x for x in self._MU)
        return self._chg[g]

    def cross_product(self, g, h) -> Sequence[CrossProductTerm]:
        if self.q_commute(g, h):
            raise ValueError(f"cross_product on q-commuting {g},{h}")
        return self._cross[(g, h)]

    def _torus_inverse_letter(self, g):
        if g == E_GEN:
            return E_INV
        if g == E_INV:
            return E_GEN
        return None

    def iter_cones(self):
        for cone in self.cones():
            yield Cone(self, cone, torus_gens=frozenset({E_GEN, E_INV}) & cone)

    def to_cone_label(self, native_label):
        factors, e_E = native_label
        gens = set(); powers = {}
        for (a, i, ex) in factors:
            if ex <= 0:
                continue
            g = (a, i); gens.add(g); powers[g] = powers.get(g, 0) + ex
        if e_E > 0:
            gens.add(E_GEN); powers[E_GEN] = e_E
        elif e_E < 0:
            gens.add(E_INV); powers[E_INV] = -e_E
        return frozenset(gens), powers

    def from_cone_label(self, gens, powers):
        ch = sorted((g, powers[g]) for g in gens if g not in (E_GEN, E_INV))
        factors = tuple((g[0], g[1], e) for (g, e) in ch)
        e_E = powers.get(E_GEN, 0) - powers.get(E_INV, 0)
        return factors, e_E

    def cycle_period_bound(self) -> int:
        return 2 * (2 * self._k + 4)

    # -- geometric (2k+4)-gon chord labels --------------------------------
    # User's geometry (verified k=1,2,3): the chords are the diagonals of
    # the (2k+4)-gon, ρ_UV is rotation by one vertex, and the cocycle is
    # the crossing form — `q_commute ⟺ the two diagonals do not cross away
    # from their endpoints`.  Chord-type `a` has length `a+1` (type 1 =
    # length-2 diagonal, …, type k+1 = diameter), read off the ρ_UV
    # self-crossing pattern (a chord crosses its ρⁿ-translate iff |n|<ℓ).

    @staticmethod
    def _diagonals_cross(d1, d2) -> bool:
        a, b = d1
        c, d = d2
        if len({a, b, c, d}) < 4:
            return False                       # shared endpoint ⇒ no crossing
        lo, hi = (a, b) if a < b else (b, a)
        return (lo < c < hi) != (lo < d < hi)  # endpoints alternate ⇒ cross

    def chord_length(self, a: int) -> int:
        """Geometric length of chord-type `a` on the `(2k+4)`-gon: `a+1`."""
        return a + 1

    def _geom(self):
        """`(sgn, {type: base-vertex offset})` aligning chords to `(2k+4)`-gon
        diagonals with `q_commute ⟺ non-crossing`.  Cached; fit greedily
        (type 1 at vertex 0) one type at a time — O(types·N·#pairs)."""
        cached = getattr(self, "_geom_cache", None)
        if cached is not None:
            return cached
        N = 2 * self._k + 4
        types = sorted(self._size)
        for sgn in (1, -1):
            off = {types[0]: 0}

            def diag(a, i):
                v = (sgn * i + off[a]) % N
                return (v, (v + self.chord_length(a)) % N)

            ok = True
            for a in types[1:]:
                placed = [t for t in types if t in off]
                sol = None
                for o in range(N):
                    off[a] = o
                    if all(self.q_commute((a, i), (b, j))
                           == (not self._diagonals_cross(diag(a, i), diag(b, j)))
                           for b in placed
                           for i in range(self._size[a])
                           for j in range(self._size[b])):
                        sol = o
                        break
                if sol is None:
                    ok = False
                    break
                off[a] = sol
            if ok:
                self._geom_cache = (sgn, dict(off))
                return self._geom_cache
        raise RuntimeError("no (2k+4)-gon diagonal labelling found")

    def geometric_label(self, g):
        """Chord `g=(a,i)` as a sorted `(2k+4)`-gon diagonal `(v1, v2)`;
        `None` for the gauge torus direction `E`/`E⁻¹`."""
        if g in (E_GEN, E_INV):
            return None
        sgn, off = self._geom()
        N = 2 * self._k + 4
        a, i = g
        v = (sgn * i + off[a]) % N
        return tuple(sorted((v, (v + self.chord_length(a)) % N)))

    def verify_qcommute_is_noncrossing(self) -> bool:
        """The cocycle's geometric content: `q_commute(g,h)` iff the
        diagonals `geometric_label(g), geometric_label(h)` do not cross."""
        ch = self._chords
        for g in ch:
            for h in ch:
                if g == h:
                    continue
                nc = not self._diagonals_cross(self.geometric_label(g),
                                               self.geometric_label(h))
                if self.q_commute(g, h) != nc:
                    return False
        return True


def _half(c):
    assert c % 2 == 0, c
    return c // 2


class U1A1AoddKAlg(ConeKAlgebra):
    """Closed-form gauged `[A_1, A_{2k+1}]` over oracle-extracted cone
    tables.  Native label = `(factors, e_E)` cone monomial.

    The trace is **oracle-free** for k = 1, 2, 3: Layer-1 cone-data
    ρ²-cyclicity reduces every trace to v-tower / single-chord seeds, and
    `_trace_residual` evaluates those in closed form (`u1_pgon_layer2`,
    p = k + 2) — the M(1,p) singlet character `cf_v`, the long chord
    `cf_Llong` (both ρ²-orbits), and the diameter `cf_diameter` (odd k).
    See `_trace_residual` for the ρ²-invariant dispatch."""

    _R = TrivialZPlusRing()

    def __init__(self, k: int = 1):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._cd = U1A1AoddConeData(self, k)
        self._oracle = getattr(self._cd, "_oracle", None)   # absent when frozen
        self._etr_cache = {}                 # (e, K) -> Tr(E^e) RPowerSeries
        self._oracle_residual_calls = 0      # closed-form coverage monitor
        self._intermediate_cache = {}        # K -> {rep: {q:int}} (bootstrap)
        self._rep_cache = {}                 # ρ²-rep memo for the bootstrap

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return ((), 0)

    def cone_data(self):
        return self._cd

    def multiply(self, a, b):
        return self._multiply_via_cone_data(a, b)

    def rho(self, lbl):
        """Hard-coded ρ_UV: chord position `i → i+1` (mod orbit size) with the
        wrap E-drift, and `E → E⁻¹` — a closed-form label map (no oracle)."""
        return self._cd._rho_label_cone(lbl)

    def rho_inverse(self, lbl):
        return self._cd._rho_inverse_label_cone(lbl)

    # ----- ρ²-orbit canonicalisation (closed-form drift quotient) --------
    #
    # ρ² is NOT finite-order here: on a single chord it shifts the position
    # `i → i+2 (mod size)` AND drifts the E-power (the wrap E-drift), so the
    # type-1 orbit `{(i, e), (i+2, e'), …}` runs off to e = ±∞.  Per the
    # `ConeKAlgebra` contract such a subclass MUST supply a closed-form
    # canonical representative (the default orbit walk would not terminate).
    # E-powers are ρ²-fixed (`ρ(E)=E⁻¹ ⇒ ρ²(E)=E`); chords reduce by the
    # drift quotient (cf. `U1SquareKAlg`'s `n mod |m|`).

    def _canonical_rho2_orbit_rep(self, label):
        factors, e = label
        if len(factors) == 0:
            return ((), e)                                  # E^e : ρ²-fixed
        if len(factors) == 1 and factors[0][2] == 1:
            from math import gcd
            a = factors[0][0]
            s = self._cd._size[a]
            P = s // gcd(s, 2)                              # ρ²-period in position
            members, cur = [], label
            for _ in range(P):
                members.append(cur)
                cur = self.rho(self.rho(cur))               # ρ²
            D = cur[1] - e                                  # drift over one period
            if D == 0:                                      # finite orbit
                return min(members, key=lambda m: (m[0][0][1], m[1]))
            istar = min(m[0][0][1] for m in members)        # canonical position
            estar = next(m[1] for m in members if m[0][0][1] == istar)
            return (((a, istar, 1),), estar % abs(D))       # drift quotient
        # Multi-gen q-commuting seed (Layer-1 "exhausted" single-cone word):
        # leave unchanged.  `_trace_residual` reads its value from the oracle,
        # which IS ρ²-invariant, so distinct ρ²-relatives need not be merged
        # for correctness (the final Σ c_seed·Tr(seed) is linear); skipping
        # the merge only forgoes a (cached) dedup, and avoids walking the
        # possibly-infinite (drifting) ρ²-orbit of a multi-gen monomial.
        return label

    # ----- Layer-2 trace residual ----------------------------------------

    def _trace_residual(self, seed_label, K):
        """Trace of a Layer-1 seed — closed form, **oracle-free** (k=1,2,3).

        The gauged chiral algebra is the rank-1 singlet `M(1,p)`, p=k+2.
        Layer-1 cone-data ρ²-cyclicity reduces every trace to a v-tower
        `E^{e}` or single-chord `L_{(a,i)}·E^{e}` seed; this evaluates them
        via the general-p closed forms in `u1_pgon_layer2`.

        The dispatch is **ρ²-invariant** — necessary because the seed is a
        ρ²-CANONICAL rep, and ρ² rotates the quiver (permuting the B_GAUGED
        odd/even coords):

          * **vanishing** — `Tr = 0` iff the *whole* ρ²-orbit is flavour-
            charged (`_orbit_has_physical`).  Reading a non-zero odd coord
            off a single canonical rep is WRONG (not ρ²-invariant).
          * **v-tower** (`E^{e}`) → `cf_v(g0)`, the M(1,p) singlet character.
          * **single chord** `(a, istar)` → `cf_{family}(n)` with gauge charge
            (v-power) `n = (-1)^{istar}·g0 - off`.  The canonical rep can sit
            at the orbit-min ODD position; ρ sends `E -> E^{-1}` per unit
            position shift, reflecting `g0`.  `off = 1` for the long chord
            (type 2; its base chord carries `g0=1` at `n=0`), `0` for the
            v-tower / diameter.  Both ρ²-orbits of the long chord thus land
            on the single `cf_Llong` (even `n=g0-1`, odd `n=-g0-1`); the
            diameter (type k+1, odd k) → `cf_diameter(g0)`.

        Verified against `BPSKAlgebra(B_GAUGED)` by charge: k=1 (33 inputs),
        k=2 (v-tower, both long-chord orbits, multi-exponents, chord
        products — 0 fails, oracle_residual_calls=0), k=3 (v-tower, long,
        diameter at all orbit positions, products).  Multi-chord seeds are
        not expected (Layer-1 reduces them); if one reaches this point it
        RAISES rather than silently falling back to the oracle."""
        factors, estar = seed_label
        g0 = self._seed_charge(seed_label)[0]

        # Vanishing: Tr = 0 iff the seed is flavour-charged.  This is NOT the
        # B_GAUGED odd-coord condition on an arbitrary ρ²-canonical rep (ρ²
        # rotates the quiver, permuting odd<->even coords); it is the
        # ρ²-invariant statement that the whole orbit is flavour-charged.
        if not self._orbit_has_physical(seed_label):
            return RPowerSeries(self._R, {}, K)

        if not factors:
            return self._cf_v(g0, K)                        # v-tower: n = g0

        if len(factors) == 1 and factors[0][2] == 1:
            a, istar, _ = factors[0]
            # The cf families are indexed by the gauge charge (v-power) n of
            # the EVEN-orbit base chord: cf(n) = Tr(chg_base + n·μ), so on the
            # base orbit n = g0 - off(family).  The ρ²-canonical rep can land
            # at the orbit-min odd position; ρ sends E -> E^{-1} per unit
            # position shift, reflecting g0, so for the odd ρ²-orbit
            # n = (-1)^{istar}·g0 - off.  (off = 1 for the long chord — its
            # base chord carries g0 = 1 at n = 0; 0 for v-tower/diameter.)
            base = (-1) ** istar * g0
            if a == 2:                                      # long chord (type 2)
                return self._cf_Llong(base - 1, K)
            if a == self.k + 1:                             # diameter (type k+1)
                return self._cf_diameter(base, K)
            if 2 < a < self.k + 1:                          # intermediate chord
                # No closed form (its LOG b-slopes are an open (1,p)/B_p
                # log-module question); compute to arbitrary q-order by the
                # spine-free orthonormality bootstrap (certified exact).
                return self._intermediate_trace(seed_label, K)

        # Layer-1 reduces every trace to v-tower / single-chord seeds, so this
        # is a trace-reduction gap if reached — surface it rather than hide it.
        raise NotImplementedError(
            f"_trace_residual: unreduced seed {seed_label!r} (k={self.k}) — "
            f"Layer-1 cone reduction did not land on a v-tower or single "
            f"chord."
        )

    def _intermediate_trace(self, seed_label, K):
        """`Tr` of an intermediate chord (type a, 2 < a < k+1) to order `K`,
        via the BPS-free orthonormality bootstrap (`u1aodd_trace_bootstrap`).

        Arbitrary q-order, certified (the solver raises if under-determined or
        inconsistent), cached per `K`.  The gauge charge is handled by the
        bootstrap's own ρ²-rep keying, so the full `seed_label` is passed."""
        from u1aodd_trace_bootstrap import solve_intermediate, _rho2_rep
        if K not in self._intermediate_cache:
            self._intermediate_cache[K] = solve_intermediate(self, K)
        Tr = self._intermediate_cache[K]
        rep = _rho2_rep(self, seed_label, self._rep_cache)
        series = Tr.get(rep, {})
        return self._lp_to_rps(
            LaurentPoly({q: c for q, c in series.items() if 0 <= q <= K}), K)

    def _orbit_has_physical(self, seed) -> bool:
        """`True` iff the ρ²-orbit of `seed` contains a flavour-neutral
        member (zero odd charge) — i.e. `Tr != 0`.  Odd charge depends only
        on the chord positions (E adds only to μ/even coords), so one orbit
        period suffices."""
        cd = self._cd
        n, MU = cd._n, cd._MU
        factors = seed[0]
        if not factors:
            return True
        P = cd._size[factors[0][0]]
        cur = seed
        for _ in range(P + 1):
            cg = self._seed_charge(cur)
            if not any(cg[j] for j in range(n) if MU[j] == 0):
                return True
            cur = self.rho(self.rho(cur))
        return False

    # ----- k=1 closed-form Layer-2 (singlet M(3) characters) -------------
    def _seed_charge(self, seed):
        """The `B_GAUGED` charge `γ` of a cone-monomial seed
        `(factors, e_E)` — `e_E·μ + Σ exp·charge(chord)` in `B_GAUGED`
        coords (a length-`n` list)."""
        cd = self._cd
        n, MU = cd._n, cd._MU
        factors, e = seed
        g = [e * MU[j] for j in range(n)]
        for (a, i, ex) in factors:
            c = cd._chg[(a, i)]
            for j in range(n):
                g[j] += ex * c[j]
        return g

    def _seed_gauge_charged(self, seed):
        """`True` iff the seed's `B_GAUGED` charge has a non-zero odd
        (non-`μ`) coordinate — i.e. it is a gauge-charged seed whose trace
        vanishes."""
        cd = self._cd
        g = self._seed_charge(seed)
        return any(g[j] for j in range(cd._n) if cd._MU[j] == 0)

    def _lp_to_rps(self, lp, K):
        """LaurentPoly (integer `q`-series from `u1_pgon_layer2`) ->
        `RPowerSeries` over the algebra's coefficient ring."""
        return RPowerSeries(
            self._R, {e: c for e, c in lp._coeffs.items() if 0 <= e <= K}, K)

    def _cf_v(self, m, K):
        """`Tr(v^m)` — the M(1,p) singlet character at charge `m`
        (general p = k+2, all m)."""
        return self._lp_to_rps(_gp.tr_v_n(self.k + 2, m, K), K)

    def _cf_Llong(self, m, K):
        """`Tr(L_long·v^m)` — long chord (type 2), general p = k+2, all m
        (n>=0 cross-rung; n<0 the spectral-flow mirror family)."""
        return self._lp_to_rps(_gp.tr_L_long_v_n(self.k + 2, m, K), K)

    def _cf_diameter(self, m, K):
        """`Tr(L_diam·v^m)` — diameter chord (type k+1, survives for odd k),
        general p = k+2, all m."""
        return self._lp_to_rps(_gp.tr_L_diameter_v_n(self.k + 2, m, K), K)

    def _oracle_elt_of_label(self, label):
        """Oracle canonical basis element for a cone monomial `(factors, e_E)`
        (its factors q-commute, so the product is a single oracle term) —
        the iso image used to read the seed's trace off the oracle."""
        T = self._oracle
        factors, e = label
        prod = Element({((), (0, e)): LaurentPoly.one()})
        for (a, i, ex) in factors:
            f = Element({self._cd._elt[(a, i)]: LaurentPoly.one()})
            for _ in range(ex):
                prod = T.multiply_elements(prod, f)
        (oelt, _), = prod.terms.items()
        return oelt

    # convenience: native label for a single chord (a, i)
    def L(self, a, i):
        return (((a, i, 1),), 0)

    def geometric_label(self, g):
        """Chord mult-gen `g=(a,i)` as a `(2k+4)`-gon diagonal `(v1,v2)`
        (length `a+1`, ρ_UV = rotate-by-one); `None` for `E`/`E⁻¹`."""
        return self._cd.geometric_label(g)


if __name__ == "__main__":
    A = U1A1AoddKAlg(1)
    cd = A.cone_data()
    print("U1A1AoddKAlg(1): %d mult-gens (%d chords + E^±), %d cones"
          % (len(cd.mult_gens()), len(cd._chords), len(cd.cones())))
    print("L(1,0)*L(1,1) =", dict(A.multiply(A.L(1, 0), A.L(1, 1)).terms))
    print("L(2,0)*L(2,2) =", dict(A.multiply(A.L(2, 0), A.L(2, 2)).terms))
