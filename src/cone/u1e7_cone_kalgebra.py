"""
u1e7_cone_kalgebra.py
=====================

`U1E7ConeKAlgebra` — the u(1)-gauged E7 algebra as a **standalone QT-cone
ConeKAlgebra**, built once from the `U1E7GaugedRG` RG-flow oracle (no engine at
runtime once the tables are frozen).  Rebuilt per the spine-free
QTCone-from-RG recipe; the earlier explicit-`(c0,c1)`
construction (#589) is superseded.

Structure (= the recipe):
  * **rank-1 torus** = the gauge leg `E = X_{(0,1)}` (Laurent).  The magnetic
    leg `X_{(1,0)}` is **not** a torus direction; magnetic charge is carried by
    the **dyonic chords** (`u1e7_cone_derivation.select_chords`: 182 chords,
    `c0 ∈ {-3..3}`).
  * **cones** = the q-commuting cliques of the chords + `E` (built fresh; 2508).
  * **factoring** (`to_cone_label`): chord-atoms cover the A6 letters (magnetic
    `c0` matched exactly), the residual `c0` is supplied by **monopole atoms**
    `X_{(1,0)}^{±}` (chord-atoms with empty A6 part, NOT torus), the residual
    gauge `c1` by `E`.  Validated to close `multiply` 100% (78376 product
    canonicals, magnetic and c0=0 alike).
  * **trace** (Layer 2): magnetic `c0 ≠ 0 ⇒ Tr = 0` exactly; the gauge v-tower
    `E^n` via the lazy vacuum recipe; the `c0 = 0` chord seeds via the
    spine-free orthonormality bootstrap (`u1e7_trace_bootstrap`).
"""
from __future__ import annotations

import os
import sys
import itertools
import pickle
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing, ZPlusRing, RPowerSeries, RElement
from cone_data import FiniteConeData, Cone
from cone_kalgebra import ConeKAlgebra
# NOTE: `u1e7_gauged_rg` (the RG-flow oracle) and `u1e7_cone_derivation` are
# imported LAZILY (inside the build / oracle paths only) so the frozen,
# spine-free release never touches the realisation engine.


def _lp(c) -> LaurentPoly:
    return c if isinstance(c, LaurentPoly) else LaurentPoly(c._coeffs)


def _frozen_path() -> str:
    return os.path.join(_HERE, "u1e7_cone_tables.pkl")


def _rho_frozen_path() -> str:
    return os.path.join(_HERE, "u1e7_rho_tables.pkl")


class U1E7ConeData(FiniteConeData):
    """QT-cone data for u(1)-gauged E7 over the trivial ring; rank-1 torus on the
    gauge leg `E = X_{(0,1)}`, magnetic carried by dyonic chords."""

    def __init__(self, oracle: U1E7GaugedRG, use_frozen: bool = True) -> None:
        self._R = TrivialZPlusRing()
        if use_frozen and self._load_frozen():
            return
        T = oracle
        self._build(T)

    # ---- build from the oracle ------------------------------------------

    def _build(self, T) -> None:
        from u1e7_cone_derivation import select_chords  # lazy: oracle-side only
        chords = select_chords(T)
        # one (chord_part, c0) atom per key, canonical c1 = min |c1|
        bykey: dict = {}
        for ch in chords:
            k = (ch[0], ch[1][0])
            if k not in bykey or abs(ch[1][1]) < abs(bykey[k][1][1]):
                bykey[k] = ch
        chordatoms = sorted(bykey.values(), key=repr)
        E, Ei = ((), (0, 1)), ((), (0, -1))
        self._atoms = tuple(chordatoms + [E, Ei])
        self._sig = {a: self._atom_sig(a) for a in self._atoms}
        # monopole atoms (empty A6 chord, c0 != 0): the magnetic generators
        self._mono = {}
        for a in chordatoms:
            if a[0] == ():
                self._mono[1 if a[1][0] > 0 else -1] = a
        # gauge torus E
        self._E = {self._sig[a][1]: a
                   for a in self._atoms if self._sig[a][0] is None}
        self._torus_gens = frozenset(
            g for g in (self._E.get((0, 1)), self._E.get((0, -1))) if g)
        # chord atoms indexed by (a, i)
        self._ai_atoms = defaultdict(list)
        for a in chordatoms:
            if a[0]:
                self._ai_atoms[self._sig[a][0]].append(a)
        # q-commute graph
        self._nb = {v: set() for v in self._atoms}
        for g, h in itertools.combinations(self._atoms, 2):
            if len(self._terms(T.multiply(g, h))) == 1:
                self._nb[g].add(h)
                self._nb[h].add(g)
        self._cones = self._maximal_cliques()
        # cone index by EVERY chord type appearing in any atom (incl multi-letter
        # atoms — indexing by the first letter only over-restricts the cover)
        self._cone_by_type = defaultdict(list)
        for ci, cone in enumerate(self._cones):
            types = set()
            for g in cone:
                for (k, i, e) in g[0]:
                    types.add((k, i))
            for tt in types:
                self._cone_by_type[tt].append(ci)
        self._post_init()           # factoring indices (cross extraction needs them)
        # cocycle + cross-product tables (from the oracle).  Two passes: the
        # cocycle table must be complete before cross extraction, which calls
        # cone_label_phase → cocycle on the daughter cone monomials.
        self._qpow: dict = {}
        self._xprod: dict = {}
        for g, h in itertools.permutations(self._atoms, 2):
            if h in self._nb[g]:
                self._qpow[(g, h)] = self._extract_cocycle(T, g, h)
        for g, h in itertools.permutations(self._atoms, 2):
            if h not in self._nb[g]:
                self._xprod[(g, h)] = self._extract_cross(T, g, h)

    def _post_init(self) -> None:
        """Derived indices for factoring (recomputed on build and load)."""
        self._tcl_cache: dict = {}
        self._atom_ct: dict = {}
        self._by_chordc0: dict = {}
        for a in self._atoms:
            chord, (c0, c1) = a
            ct = Counter()
            for (k, i, e) in chord:
                ct[(k, i)] += e
            self._atom_ct[a] = ct
            if chord:
                self._by_chordc0[(chord, c0)] = a
        self._cone_chord_atoms: dict = {
            cone: [a for a in cone if a[0]] for cone in self._cones}

    # ---- freeze / load (spine-free release) ------------------------------

    _FROZEN_KEYS = ("_atoms", "_sig", "_mono", "_E", "_torus_gens",
                    "_ai_atoms", "_nb", "_cones", "_cone_by_type",
                    "_qpow", "_xprod")

    def _load_frozen(self) -> bool:
        path = _frozen_path()
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            d = pickle.load(f)
        for k in self._FROZEN_KEYS:
            setattr(self, k, d[k])
        self._ai_atoms = defaultdict(list, self._ai_atoms)
        self._cone_by_type = defaultdict(list, self._cone_by_type)
        self._post_init()
        return True

    def freeze(self) -> str:
        path = _frozen_path()
        d = {k: getattr(self, k) for k in self._FROZEN_KEYS}
        d["_ai_atoms"] = dict(self._ai_atoms)
        d["_cone_by_type"] = dict(self._cone_by_type)
        with open(path, "wb") as f:
            pickle.dump(d, f)
        return path

    # ---- helpers --------------------------------------------------------

    def _atom_sig(self, a):
        chord, (c0, c1) = a
        return (chord[0][:2] if chord else None), (c0, c1)

    def _terms(self, elt) -> dict:
        return {l: _lp(c) for l, c in elt.terms.items() if not c.is_zero()}

    def _maximal_cliques(self):
        cliques, nb = [], self._nb

        def bk(Rs, P, X):
            if not P and not X:
                cliques.append(frozenset(Rs))
                return
            piv = max(P | X, key=lambda u: len(P & nb[u]))
            for v in list(P - nb[piv]):
                bk(Rs | {v}, P & nb[v], X & nb[v])
                P = P - {v}
                X = X | {v}

        bk(set(), set(self._atoms), set())
        return tuple(cliques)

    def _extract_cocycle(self, T, g, h) -> int:
        gh = self._terms(T.multiply(g, h))
        hg = self._terms(T.multiply(h, g))
        (s, cgh), = gh.items()
        chg = hg[s]
        a = min(cgh._coeffs)
        b = min(chg._coeffs)
        assert (a - b) % 2 == 0, f"odd cocycle {g},{h}: {a},{b}"
        return (a - b) // 2

    def _extract_cross(self, T, g, h):
        terms = []
        for s, lp_can in self._terms(T.multiply(g, h)).items():
            gens, powers = self.to_cone_label(s)
            phase = self.cone_label_phase(gens, powers)
            lp = LaurentPoly({e + phase: co for e, co in lp_can._coeffs.items()})
            terms.append((lp, self._word(gens, powers)))
        return tuple(terms)

    def _word(self, gens, powers):
        w = []
        for g in self.canonical_cone_order(gens):
            w.extend([g] * powers[g])
        return tuple(w)

    # ---- ConeData primitives --------------------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def mult_gens(self):
        return self._atoms

    def cones(self):
        return self._cones

    def q_commute(self, g, h) -> bool:
        return g == h or (h in self._nb[g])

    def cocycle(self, g, h) -> int:
        return 0 if g == h else self._qpow[(g, h)]

    def cross_product(self, g, h):
        return self._xprod[(g, h)]

    # ---- cone-label bijection -------------------------------------------
    #
    # A native label `(chord, (c0, c1))` factors as: chord-atoms covering the A6
    # letters (each carrying its magnetic `c0` share), the residual magnetic
    # `c0` supplied by monopole atoms `X_{(1,0)}^{±}`, the residual gauge `c1`
    # by the torus `E`.  (The chord-atoms' own `c1` is part of the residual.)

    def to_cone_label(self, native):
        cached = self._tcl_cache.get(native)
        if cached is not None:
            return cached
        chord, (c0, c1) = native
        ct = Counter()
        for (k, i, e) in chord:
            ct[(k, i)] += e
        # fast path: native is a single (possibly multi-letter) atom × E^k
        atom = self._by_chordc0.get((chord, c0))
        if atom is not None:
            powers = {atom: 1}
            k = c1 - self._sig[atom][1][1]
            if k:
                eg = self._E.get((0, 1) if k > 0 else (0, -1))
                if eg is not None:
                    powers[eg] = abs(k)
                    gens = frozenset(powers)
                    out = (gens, powers)
                    self._tcl_cache[native] = out
                    return out
            else:
                gens = frozenset(powers)
                out = (gens, powers)
                self._tcl_cache[native] = out
                return out
        # general exact charge-cover over the cones with all letter types
        if ct:
            types = list(ct)
            cand_idx = set(self._cone_by_type[types[0]])
            for L in types[1:]:
                cand_idx &= set(self._cone_by_type[L])
            cand_cones = [self._cones[ci] for ci in cand_idx]
        else:
            cand_cones = self._cones
        key_ct = frozenset(ct.items())
        for cone in cand_cones:
            powers = self._cover(key_ct, c0, c1, cone, {})
            if powers is not None:
                gens = frozenset(g for g, p in powers.items() if p > 0)
                out = (gens, {g: p for g, p in powers.items() if p > 0})
                self._tcl_cache[native] = out
                return out
        raise ValueError(f"to_cone_label: cannot factor {native!r}")

    def _cover(self, letters_ct, c0, c1, cone, memo):
        """Exact charge-cover: a dict `{atom: power}` of cone atoms whose
        `(chord-multiset, c0, c1)` sum to `(letters_ct, c0, c1)`, or `None`.
        Chord-bearing atoms (single or multi-letter) cover the letters; the
        residual magnetic `c0` is supplied by monopole atoms, the gauge `c1` by
        `E`."""
        key = (letters_ct, c0, c1)
        if key in memo:
            return memo[key]
        ct = Counter(dict(letters_ct))
        if not ct:                                  # letters covered
            powers: dict = {}
            r0, r1 = c0, c1
            if r0 != 0:
                mg = self._mono.get(1 if r0 > 0 else -1)
                if mg is None or mg not in cone:
                    memo[key] = None
                    return None
                (_, (m0, m1)) = self._sig[mg]
                reps = abs(r0)                      # |m0| == 1
                powers[mg] = reps
                r1 -= m1 * reps
            if r1 != 0:
                eg = self._E.get((0, 1) if r1 > 0 else (0, -1))
                if eg is None or eg not in cone:
                    memo[key] = None
                    return None
                powers[eg] = abs(r1)
            memo[key] = powers
            return powers
        L = next(iter(ct))
        for atom in self._cone_chord_atoms.get(cone, ()):
            act = self._atom_ct[atom]
            if act.get(L, 0) == 0:
                continue
            if any(ct.get(k, 0) < v for k, v in act.items()):
                continue
            rem = Counter(ct)
            rem.subtract(act)
            rem += Counter()                        # drop zero/neg
            (_, (a0, a1)) = self._sig[atom]
            sub = self._cover(frozenset(rem.items()), c0 - a0, c1 - a1, cone, memo)
            if sub is not None:
                out = dict(sub)
                out[atom] = out.get(atom, 0) + 1
                memo[key] = out
                return out
        memo[key] = None
        return None

    def from_cone_label(self, gens, powers):
        c0 = c1 = 0
        cd: dict = {}
        for g, p in powers.items():
            chord, (q0, q1) = g
            c0 += q0 * p
            c1 += q1 * p
            for (k, i, e) in chord:
                cd[(k, i)] = cd.get((k, i), 0) + e * p
        chord = tuple(sorted((k, i, e) for (k, i), e in cd.items() if e))
        return (chord, (c0, c1))

    # ---- QTCone wiring (torus on E only) --------------------------------

    def _torus_inverse_letter(self, g):
        sig = self._sig.get(g)
        if sig is None or sig[0] is not None:
            return None
        inv = {(0, 1): (0, -1), (0, -1): (0, 1)}.get(sig[1])
        return self._E.get(inv) if inv is not None else None

    def iter_cones(self):
        for cone in self._cones:
            yield Cone(self, cone, torus_gens=self._torus_gens & cone)


class U1E7ConeKAlgebra(ConeKAlgebra):
    """u(1)-gauged E7 as a standalone QT-cone ConeKAlgebra (rank-1 gauge torus),
    built from the U1E7GaugedRG oracle (frozen tables → spine-free)."""

    def __init__(self, use_frozen: bool = True):
        self._R = TrivialZPlusRing()
        # build cone-data (frozen tables if available → spine-free, else oracle)
        if use_frozen and os.path.exists(_frozen_path()):
            self._oracle = None
            self._cone_data = U1E7ConeData(None, use_frozen=True)
        else:
            from u1e7_gauged_rg import U1E7GaugedRG   # lazy: oracle-side only
            self._oracle = U1E7GaugedRG()
            self._cone_data = U1E7ConeData(self._oracle, use_frozen=False)
        self._seed_cache: dict = {}
        self._vac_cache: dict = {}
        self._boot_cache: dict = {}
        self._rho_cache: dict = {}
        self._rhoi_cache: dict = {}
        # spine-free ρ: frozen {(ray_word, c0): (π, c0', δ)} (at c1=0) + ρ²-orbit
        # bound H; any gauge charge c1 follows from the reflection below.
        self._rhotab = self._rhoitab = self._Hval = None
        self._rho_rec = None        # freeze-time (ray_word, c0) recorder (else None)
        self._load_rho_frozen()

    # ---- KAlgebra contract ----------------------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return ((), (0, 0))

    def cone_data(self):
        return self._cone_data

    def _oracle_alg(self):
        if self._oracle is None:
            from u1e7_gauged_rg import U1E7GaugedRG   # lazy: oracle-side only
            self._oracle = U1E7GaugedRG()
        return self._oracle

    def _H_value(self):
        return self._Hval if self._Hval is not None else self._oracle_alg()._H

    def rho(self, a):
        """ρ on the contract.  Spine-free via the frozen single-ray `(ray_word,
        c0)` table + the gauge-reflection formula `ρ((w,(c0,c1)))=(π,(c0',δ-c1))`
        for tabulated keys; for a multi-ray *product* canonical (key absent from
        the table) ρ is computed oracle-free by the automorphism rule
        `_rho_via_cone`.  The oracle is consulted only at freeze/build time (when
        `_rho_rec` is recording the table)."""
        v = self._rho_cache.get(a)
        if v is None:
            chord, (c0, c1) = a
            ent = self._rhotab.get((chord, c0)) if self._rhotab is not None else None
            if ent is not None:
                pi, pc0, delta = ent
                v = (pi, (pc0, delta - c1))
            elif self._rho_rec is not None:
                # freeze/build time: record the key and tabulate from the oracle.
                self._rho_rec.add((chord, c0))
                v = self._oracle_alg().rho(a)
            else:
                v = self._rho_via_cone(a, inverse=False)
            self._rho_cache[a] = v
        return v

    def rho_inverse(self, a):
        v = self._rhoi_cache.get(a)
        if v is None:
            chord, (c0, c1) = a
            ent = self._rhoitab.get((chord, c0)) if self._rhoitab is not None else None
            if ent is not None:
                pi, pc0, delta = ent
                v = (pi, (pc0, delta - c1))
            elif self._rho_rec is not None:
                self._rho_rec.add((chord, c0))
                v = self._oracle_alg().rho_inverse(a)
            else:
                v = self._rho_via_cone(a, inverse=True)
            self._rhoi_cache[a] = v
        return v

    def _rho_via_cone(self, a, inverse: bool):
        """ρ (or ρ⁻¹) on a label whose `(ray_word, c0)` is absent from the frozen
        single-ray table -- i.e. a multi-ray *product* canonical.  Oracle-free,
        via the automorphism property: ρ is an algebra automorphism, so

            ρ(L_a) = ρ(∏_g L_g^{p_g}) = ∏_g ρ(L_g)^{p_g},

        where `(g, p_g) = to_cone_label(a)` are the cone mult-gen atoms -- each a
        single ray whose ρ IS tabulated (incl. the torus `E`, `ρ(E)=E⁻¹`).
        Recombine by the (spine-free) cone multiply; ρ permutes the canonical
        basis, so the product is a single basis element `q^k·L_{ρ(a)}` -- return
        its label.  Verified == the oracle on every multi-ray product canonical."""
        from kalgebra import Element
        from laurent_poly import LaurentPoly
        cd = self._cone_data
        _, powers = cd.to_cone_label(a)
        one = LaurentPoly({0: 1})
        acc = Element({self.identity(): one})
        rfn = self.rho_inverse if inverse else self.rho
        for g, p in powers.items():
            if not p:
                continue
            r_img = Element({rfn(cd.from_cone_label(frozenset([g]), {g: 1})): one})
            for _ in range(p):
                acc = self.multiply_elements(acc, r_img)
        labels = [lab for lab, co in acc.terms.items() if not co.is_zero()]
        if len(labels) != 1:
            raise AssertionError(
                f"_rho_via_cone: expected a single basis image for {a!r}, "
                f"got {labels!r}")
        return labels[0]

    def _canonical_rho2_orbit_rep(self, label):
        """Drift-quotient canonicalisation.  Magnetic (`c0 ≠ 0`) seeds have
        infinite ρ²-orbits (ρ² drifts the gauge leg `c1`) but trace to 0, so no
        merge is needed — return as-is.  Every `c0 = 0` seed (ray-generator
        word, multi-ray cone-word, or gauge v-tower) has a FINITE ρ²-orbit (no
        c1-drift) → merge to the orbit minimum, so ρ²-cyclicity is enforced in
        code (the multi-ray seeds the Layer-1 reducer emits must be folded too,
        else cyclicity relations between them are lost)."""
        chord, (c0, c1) = label
        if c0 != 0:
            return label
        members, cur = [], label
        for _ in range(8 * self._H_value() + 12):
            members.append(cur)
            cur = self.rho(self.rho(cur))
            if cur == label:
                return min(members, key=repr)
        return label

    # ---- freeze / load spine-free ρ (ray-word table + reflection) -------

    def _load_rho_frozen(self) -> bool:
        path = _rho_frozen_path()
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            d = pickle.load(f)
        if not {"_rho", "_rhoi", "_H"} <= d.keys():     # stale / wrong format
            return False
        self._rhotab, self._rhoitab, self._Hval = d["_rho"], d["_rhoi"], d["_H"]
        return True

    def freeze_rho(self, K: int = 6) -> str:
        """Extract ρ / ρ⁻¹ keyed by `(ray_word, c0)` at `c1 = 0` and the ρ²-orbit
        bound `H` from the oracle, writing `u1e7_rho_tables.pkl` so ρ is
        spine-free.  Only `(π, c0', δ)` per key is stored — any gauge charge `c1`
        follows from `ρ((w,(c0,c1))) = (π,(c0',δ-c1))` (ρ(E)=E⁻¹, ρ an
        automorphism, so the gauge leg always reflects; validated for every c0).
        The `(ray_word, c0)` set is bounded (independent of `K` — capped by the
        bootstrap deep-power reach `pcap`, not the q-order), so this is exact at
        arbitrary q-order (no K cap).

        Completeness: *record* every `(ray_word, c0)` the trace path actually
        queries ρ on (a freeze-time bootstrap + orthonormality battery, ρ falling
        back to the oracle), seed with the cone-structure ray-words, and close
        under ρ / ρ⁻¹ — so the ρ²-orbit walk never escapes the table."""
        from u1e7_gauged_rg import U1E7GaugedRG     # lazy: oracle-side only
        if self._oracle is None:
            self._oracle = U1E7GaugedRG()
        orc = self._oracle
        cd = self._cone_data
        # 1) record the (ray_word, c0) keys the trace/bootstrap path queries
        rec: set = set()
        self._rho_rec = rec
        self._rho_cache.clear()
        self._rhoi_cache.clear()
        try:
            from u1e7_trace_bootstrap import solve_chord_seeds
            solve_chord_seeds(self, K)
            atoms = [g for g in cd.mult_gens() if g[1][0] == 0 and g[0] != ()][:14]
            for a in atoms:
                for b in atoms:
                    self.inner_product(a, b, K)
        finally:
            self._rho_rec = None
            self._rho_cache.clear()
            self._rhoi_cache.clear()
        # 2) seed with the cone-structure ray-words (at c0=0) too
        keys = set(rec) | {((), 0)}
        for a in cd.mult_gens():
            keys.add((a[0], a[1][0]))
        for cone in cd.cones():
            for lab in cone:
                keys.add((lab[0], lab[1][0]))
        # 3) close under ρ / ρ⁻¹ and tabulate (π, c0', δ)
        rho: dict = {}
        rhoi: dict = {}
        queue = list(keys)
        while queue:
            ch, c0 = queue.pop()
            if (ch, c0) in rho:
                continue
            r = orc.rho((ch, (c0, 0)))
            rho[(ch, c0)] = (r[0], r[1][0], r[1][1])
            ri = orc.rho_inverse((ch, (c0, 0)))
            rhoi[(ch, c0)] = (ri[0], ri[1][0], ri[1][1])
            for nb in ((r[0], r[1][0]), (ri[0], ri[1][0])):
                if nb not in rho:
                    queue.append(nb)
        with open(_rho_frozen_path(), "wb") as f:
            pickle.dump({"_rho": rho, "_rhoi": rhoi, "_H": orc._H}, f)
        self._rhotab, self._rhoitab, self._Hval = rho, rhoi, orc._H
        return _rho_frozen_path()

    # ---- trace (Layer 2) ------------------------------------------------

    def _int_rps(self, coeffs, K):
        """Build an `RPowerSeries` over the trivial ring from `{q: int}`."""
        ub = self._R.one_basis()
        return RPowerSeries(
            self._R,
            {q: RElement(self._R, {ub: int(c)})
             for q, c in coeffs.items() if c and 0 <= q <= K},
            K)

    @staticmethod
    def _mag_charge(label):
        """The magnetic (`X_{(1,0)}`) charge `c0` — ρ²-invariant; the QT trace
        annihilates every `c0 ≠ 0` sector exactly."""
        _chord, (c0, _c1) = label
        return c0

    def _trace_residual(self, seed_label, K):
        """Layer-2 seed value.  `c0 ≠ 0 ⇒ 0` (magnetic, exact); the gauge
        v-tower `E^n` (incl. the identity `Tr(1)`) via the lazy vacuum recipe;
        every other `c0 = 0` cone-word via the **forward-triangular
        orthonormality bootstrap** (`u1e7_trace_bootstrap.solve_chord_seeds`).

        The bootstrap is the "orthonormality fixes traces up to Tr(1)" demo for
        this QTCone: the only supplied input is the gauge/vacuum sector
        (`Tr(1)`/`Tr(E^n)`); the deep-power, self-/cross-orthonormality, and
        one-step cyclicity constraints then pin every `c0 = 0` word by a
        certified forward sweep (no contradiction / under-determination at order
        ≤ K).  See `_chord_seed_trace` and `trace`."""
        if self._mag_charge(seed_label) != 0:
            return RPowerSeries(self._R, {}, K)
        chord, (_c0, c1) = seed_label
        if chord == ():                       # gauge v-tower E^{c1} (incl. Tr(1))
            return self._v_tower_trace(c1, K)
        return self._chord_seed_trace(seed_label, K)

    # -- lazy vacuum recipe: Tr(E^n) = [μ^{-n}](Tr_E7(1;μ)·(q²;q²)_∞²) ----

    def _vacuum_mu(self, K):
        """`Tr_E7(1; μ)·(q²;q²)_∞²` as `{q_exp: {mu_exp: int}}` to order `K`."""
        if K in self._vac_cache:
            return self._vac_cache[K]
        from vacuum_nahm import vacuum_trace_rps, SPECS
        from finite_e7_kalg import E7_BPS_PAIRING
        from zplus_ring import AbelianZPlusRing
        from qpoch import qpoch_infty
        R = AbelianZPlusRing(rank=1)
        vac = vacuum_trace_rps(SPECS["e7"], E7_BPS_PAIRING, R, K)
        vacd: dict = {}
        for q, c in vac.coeffs.items():
            src = getattr(c, "_coeffs", None) or getattr(c, "terms", {})
            for kb, v in src.items():
                n = kb[0] if isinstance(kb, tuple) and kb else 0
                vacd.setdefault(q, {})[n] = vacd.get(q, {}).get(n, 0) + int(v)
        m = qpoch_infty(K)
        m = m * m
        meas = {e: int(c) for e, c in m._c.items()}
        P: dict = {}
        for q1, md in vacd.items():
            for qe, mc in meas.items():
                q = q1 + qe
                if q > K:
                    continue
                for mu, co in md.items():
                    P.setdefault(q, {})[mu] = P.get(q, {}).get(mu, 0) + co * mc
        self._vac_cache[K] = P
        return P

    def _v_tower_trace(self, n, K):
        """`Tr(E^n) = [μ^{-n}](Tr_E7(1;μ)·(q²;q²)_∞²)`."""
        P = self._vacuum_mu(K)
        coeffs = {q: md[-n] for q, md in P.items() if md.get(-n, 0) and q <= K}
        return self._int_rps(coeffs, K)

    def _chord_seed_trace(self, seed_label, K):
        """`Tr` of a `c0 = 0` chord seed via the spine-free orthonormality
        bootstrap (`u1e7_trace_bootstrap`), cached per `K`."""
        from u1e7_trace_bootstrap import solve_chord_seeds, rho2_rep
        if K not in self._boot_cache:
            self._boot_cache[K] = solve_chord_seeds(self, K)
        Tr = self._boot_cache[K]
        rep = rho2_rep(self, seed_label)
        series = Tr.get(rep, {})
        return self._int_rps(series, K)

    def trace(self, a, K=20):
        """`Tr(L_a)` for a single basis cone-word `a`.

        Layer-1 for this QTCone is pure **ρ²-canonicalisation** (`Tr(ρ²x)=Tr(x)`):
        a basis label is one cone-word ⇒ one seed, so no expansion is needed.
        The generic tagged-cycle reduction `ConeData.simplify_trace_via_cone_data`
        (used by `ConeKAlgebra.trace`) is cyclically INCONSISTENT on this QTCone
        and is therefore bypassed.  Layer-2 is `_trace_residual` (magnetic → 0,
        gauge v-tower, and the orthonormality bootstrap for chords).  Multi-term
        traces go through the inherited `trace_element`, which sums this per
        term — so `inner_product` reproduces orthonormality through the contract.
        """
        return self._trace_residual(self._canonical_rho2_orbit_rep(a), K)


if __name__ == "__main__":
    import time
    from u1e7_gauged_rg import U1E7GaugedRG
    t0 = time.time()
    cd = U1E7ConeData(U1E7GaugedRG(), use_frozen=False)
    print(f"U1E7ConeData built in {time.time()-t0:.1f}s: "
          f"{len(cd.mult_gens())} atoms, {len(cd.cones())} cones")
    cd.freeze()
    print("frozen ->", _frozen_path())
    # spine-free ρ tables (ray-word reflection): exact at any q-order
    t1 = time.time()
    K = U1E7ConeKAlgebra(use_frozen=True)
    K.freeze_rho()
    print(f"ρ tables frozen in {time.time()-t1:.1f}s -> {_rho_frozen_path()} "
          f"({len(K._rhotab)} (ray-word, c0) keys, H={K._Hval})")
