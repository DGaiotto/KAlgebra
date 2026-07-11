"""`SkeinEvenPolygonKAlg` — the general even `(2k+4)`-gon (the
U(1)-gauged `[A₁, A_{2k+1}]`) as a `KAlgebra` realized by the
stated-SKEIN engine on the PINNED torus (the general-n step of the
pinned / unpin programme).

This generalizes `SkeinSquareKAlg` (n = 4, the k = 0 corner outside
this class's domain) and supersedes the per-polygon fit route of
`SkeinU1HexagonKAlg` (n = 6, localized) with a uniform derivation on
`PinnedPolygon(2k+4)`.  The even-marked polygon carries the boundary
obstruction grading `χ = (1,−1,…)`: the side-pin image is exactly
`ker χ`, so even-SPAN chords cannot unpin — the construction derives,
per instance:

* **Parity-covariant dressing words.**  Odd-span chords unpin fully
  (`w` = the alternating block on the span, rotated per chord);
  even-span chords are dressed to the **residual pin `±ν`**
  (`ν = e₀ + e₂`, sign = the base-vertex parity) — the general form of
  the square's `±μ` 't Hooft half-quantization, and FORCED, not
  aesthetic: matching the identity daughter of the family meson
  requires the residual pins of adjacent chords to cancel, so `pin` is
  the linear grading `P∘charge` of the intrinsic charge lattice.

* **Kernel offsets — solved.**  On an even cycle the pin-cancellation
  system has the alternating kernel, whose side-word is exactly the
  gauge unit `E = ∏_r s_r^{(−1)^r}`; a per-chord kernel choice shifts
  the dressed chord by `E`-units and must match the intrinsic's clean
  `E`-frame (`e_E` labels).  The integer offsets `c_g` are solved at
  construction from the **charge-alignment linear system** measured on
  the letter-pair pinned products (support offsets in `E`-charge
  units), with the `E` orientation `ε`; residual nullspace directions
  are scanned (|t| ascending) against the gauge stage.

* **Per-letter gauges — solved.**  The lq-gauges `x_g` (per chord
  letter) and `x_E` are solved jointly from the Δ-equations
  `Δ(d ∈ g·h) = x_g + x_h − Σ m_ℓ x_ℓ − e_E·x_E` measured on the
  generator-pair peels — the `SkeinHeptagonKAlg` δ-calibration pattern,
  lifted to a construction-time exact linear solve.  (Unlike the odd
  polygons, a single per-family meson gauge does not exist here — the
  residual pins break p-uniformity.)

After construction every determination is **asserted, never fitted**:
`multiply(a, b)` is computed GENUINELY ON THE PINNED SIDE (torus
product of forward images, peeled against the intrinsic-hinted label
set) and each peeled coefficient is ASSERTED to equal the intrinsic
coefficient VERBATIM under `q → lq^{−2}` — the standing guard, δ ≡ 0.

Labels are the intrinsic `U1A1AoddKAlg(k)` labels `(factors, e_E)`;
the chord↔diagonal dictionary is the intrinsic's own certified
`geometric_label` (`q_commute ⟺ non-crossing`).  Transported
primitives: `rho`/`rho_inverse`/`trace` via the intrinsic (the trace is
BPS-free: Layer-1 cyclicity + closed-form/bootstrap seeds).
`coefficient_ring` is Trivial (the U(1)-gauged A-odd theories are
unflavoured).  ρ² has infinite orbits (E-drift shears) — the intrinsic
closed-form drift quotient is the contract-required
`_canonical_rho2_orbit_rep`.

Validated:
k = 1 (hexagon — independent pinned route beside the localized
`SkeinU1HexagonKAlg`) and k = 2 (octagon, [A₁, A₅] gauged — NEW) full
generator sweeps + deeper batteries; k = 3 (decagon) battery.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import Element, KAlgebra
from laurent_poly import LaurentPoly

from half_laurent import HalfLaurent
from pinned_polygon import PinnedPolygon
from skein_pentagon_kalg import SkeinPentagonKAlg

Q = HalfLaurent.monomial


def _gauss(rows_aug, ncols):
    """Exact Gaussian elimination on augmented rows; returns
    (solution-with-free-cols-0, null-basis) or None if inconsistent."""
    m0 = [row[:] for row in rows_aug]
    r = 0
    piv = []
    for col in range(ncols):
        p = next((i for i in range(r, len(m0)) if m0[i][col] != 0), None)
        if p is None:
            continue
        m0[r], m0[p] = m0[p], m0[r]
        m0[r] = [x / m0[r][col] for x in m0[r]]
        for i in range(len(m0)):
            if i != r and m0[i][col] != 0:
                m0[i] = [x - m0[i][col] * y for x, y in zip(m0[i], m0[r])]
        piv.append(col)
        r += 1
    if any(all(x == 0 for x in row[:-1]) and row[-1] != 0 for row in m0):
        return None
    sol = [Fraction(0)] * ncols
    for i, col in enumerate(piv):
        sol[col] = m0[i][-1]
    nulls = []
    for fc in (c for c in range(ncols) if c not in piv):
        v = [Fraction(0)] * ncols
        v[fc] = Fraction(1)
        for i, col in enumerate(piv):
            v[col] = -m0[i][fc]
        nulls.append(v)
    return sol, nulls


class SkeinEvenPolygonKAlg(KAlgebra):
    """The pinned-torus stated-skein realization of the even
    `(2k+4)`-gon (U(1)-gauged `[A₁, A_{2k+1}]`), on the
    `U1A1AoddKAlg(k)` labels — the generic-n unpin engine for the
    even/gauged family (words parity-covariant, kernel offsets and
    gauges construction-solved, standing guard verbatim)."""

    ORI = -2

    def __init__(self, k: int = None, intrinsic=None):
        if intrinsic is None:
            if k is None:
                raise ValueError("need k or a U1A1AoddKAlg intrinsic")
            from u1a1aodd_kalg import U1A1AoddKAlg
            intrinsic = U1A1AoddKAlg(k)
        from u1a1aodd_kalg import E_GEN, E_INV
        self._E_GEN, self._E_INV = E_GEN, E_INV
        self.k = intrinsic.k
        n = 2 * self.k + 4
        self.n = n
        self._intr = intrinsic
        self._cd = intrinsic.cone_data()
        assert self._cd.verify_qcommute_is_noncrossing(), \
            "intrinsic chord geometry failed its own certificate"
        sgn, off = self._cd._geom()
        self.pp = PinnedPolygon(n)
        pp = self.pp
        self._sides = [pp.arc_F(r, (r + 1) % n, 1, 1) for r in range(n)]
        self._sides_inv = [pp.inverse_monomial(s) for s in self._sides]
        self._NU = tuple(1 if r in (0, 2) else 0 for r in range(n))
        self._ALT = tuple((-1) ** r for r in range(n))
        self._letters = [g for g in self._cd.mult_gens()
                         if g not in (E_GEN, E_INV)]
        self._base_vertex = {g: (sgn * g[1] + off[g[0]]) % n
                             for g in self._letters}
        self._words = {}
        self._resid = {}
        for g in self._letters:
            self._words[g], self._resid[g] = self._base_word(g)
        # the gauge unit E = prod s_r^{(-1)^r} (single pin-0 monomial)
        E_word = None
        for r in range(n):
            f = self._sides[r] if r % 2 == 0 else self._sides_inv[r]
            E_word = f if E_word is None else pp.multiply(E_word, f)
        assert len(E_word) == 1 and \
            pp.pinning_vector(E_word) == (0,) * n, "E word not a pin-0 unit"
        self._E_word = E_word
        ((ec, _),) = E_word.items()
        self._E_chg = tuple(ec)
        # stage 1: kernel-offset charge solve; stage 2: joint gauge solve
        self._solve_frames()
        # construction certificates on the final dressed letters
        for g in self._letters:
            assert pp.pinning_vector(self._T[g]) == self._resid[g], g
        self._fwd_cache = {}
        self._mult_cache = {}

    # ---- dressing ------------------------------------------------------------

    def _base_word(self, g):
        """Parity-covariant particular dressing word + residual pin for
        chord letter g: odd spans unpin fully, even spans to
        `(−1)^{base vertex}·ν`."""
        n = self.n
        a, i = g
        j = a + 1
        v = self._base_vertex[g]
        pinD = [0] * n
        pinD[v] += 1
        pinD[(v + j) % n] += 1
        if j % 2 == 1:
            want = (0,) * n
        else:
            s = 1 if v % 2 == 0 else -1
            want = tuple(s * x for x in self._NU)
        rhs = tuple(pinD[r] - want[r] for r in range(n))
        # anchored particular solution (kernel fixed by the c-solve)
        w = [0] * n
        w[0] = rhs[0]
        for r in range(1, n):
            w[r] = rhs[r] - w[r - 1]
        assert w[n - 1] + w[0] - rhs[0] == w[n - 1], "internal"
        return tuple(w), want

    def _dress(self, g, coff: int) -> dict:
        """T_g at kernel offset coff:
        F(D_g) · NO_v(sides^{-(w_g + coff·ALT)})."""
        n = self.n
        pp = self.pp
        a, i = g
        v = self._base_vertex[g]
        out = pp.arc_F(v, (v + a + 1) % n, 1, 1)
        w = [self._words[g][r] + coff * self._ALT[r] for r in range(n)]
        for t in range(n):
            r = (v + t) % n
            e = -w[r]
            for _ in range(abs(e)):
                out = pp.multiply(
                    out, self._sides[r] if e > 0 else self._sides_inv[r])
        return out

    # ---- construction-time solves ----------------------------------------------

    def _lbl(self, g):
        return self._cd.from_cone_label(frozenset({g}), {g: 1})

    def _word_letters(self, lbl):
        gens, powers = self._cd.to_cone_label(lbl)
        out, eE = [], 0
        for g in gens:
            if g == self._E_GEN:
                eE += powers[g]
            elif g == self._E_INV:
                eE -= powers[g]
            else:
                out.append((g, powers[g]))
        return out, eE

    def _support_offset(self, A, B):
        """o with supp(A) = supp(B) + o·E_chg; None if no such o."""
        if len(A) != len(B):
            return None
        sa, sb = sorted(A), sorted(B)
        d = tuple(x - y for x, y in zip(sa[0], sb[0]))
        E = self._E_chg
        nz = next(i for i, x in enumerate(E) if x)
        if d[nz] % E[nz]:
            return None
        o = d[nz] // E[nz]
        if any(x != o * e for x, e in zip(d, E)):
            return None
        if all(tuple(x - o * e for x, e in zip(u, E)) in B for u in A):
            return o
        return None

    def _fwd_plain(self, Timgs, Eimg, Einv, lbl):
        lets, eE = self._word_letters(lbl)
        img = self.pp.one()
        for g, m in sorted(lets):
            for _ in range(m):
                img = self.pp.multiply(img, Timgs[g])
        for _ in range(abs(eE)):
            img = self.pp.multiply(img, Eimg if eE > 0 else Einv)
        return img

    def _fwd_packaged(self, Timgs, Eimg, Einv, lbl):
        """The cone-packaged forward (canonical cone order + the
        accumulated intrinsic phase), parameterized by the letter
        images — the phase convention of `forward`, usable during the
        construction solves."""
        lbl = tuple(lbl)
        pp = self.pp
        if lbl == self._intr.identity():
            return pp.one()
        gens, powers = self._cd.to_cone_label(lbl)
        order = self._cd.canonical_cone_order(gens)
        acc = None
        img = None
        for g in order:
            glbl = self._cd.from_cone_label(frozenset({g}), {g: 1})
            if g == self._E_GEN:
                gT = Eimg
            elif g == self._E_INV:
                gT = Einv
            else:
                gT = Timgs[g]
            for _ in range(powers.get(g, 0)):
                if acc is None:
                    acc = (glbl, Fraction(0))
                else:
                    prod = self._intr.multiply(acc[0], glbl)
                    terms = dict(prod.terms)
                    assert len(terms) == 1, (lbl, g, terms)
                    nl = next(iter(terms))
                    acc = (nl, acc[1] + self._qexp1(terms[nl]))
                img = gT if img is None else pp.multiply(img, gT)
        assert acc is not None and tuple(acc[0]) == lbl, (lbl, acc)
        return pp.scale(img, Q(Fraction(-self.ORI * acc[1])))

    def _solve_frames(self):
        """Solve the kernel offsets (+E orientation) from charge
        alignment, then the per-letter gauges + x_E from the Δ-system;
        scan residual charge-nullspace directions |t|-ascending until
        the gauge system is consistent."""
        pp = self.pp
        letters = self._letters
        E_GEN, E_INV = self._E_GEN, self._E_INV
        T0 = {g: self._dress(g, 0) for g in letters}
        E_inv0 = pp.inverse_monomial(self._E_word)
        idx = {g: i for i, g in enumerate(letters)}
        # charge equations from letter-pair products
        rows = []
        for g in letters:
            for h in letters:
                prod = pp.multiply(T0[g], T0[h])
                hint = self._intr.multiply(self._lbl(g), self._lbl(h))
                terms = list(hint.terms)

                def coeff_of(d):
                    lets, eE = self._word_letters(tuple(d))
                    coeff = {}
                    for z in (g, h):
                        coeff[z] = coeff.get(z, 0) + 1
                    for (l, m) in lets:
                        coeff[l] = coeff.get(l, 0) - m
                    return coeff, eE

                if len(terms) == 1:
                    o = self._support_offset(
                        prod, self._fwd_plain(T0, self._E_word, E_inv0,
                                              tuple(terms[0])))
                    if o is not None:
                        rows.append((*coeff_of(terms[0]), o))
                elif len(terms) == 2:
                    i1 = self._fwd_plain(T0, self._E_word, E_inv0,
                                         tuple(terms[0]))
                    i2 = self._fwd_plain(T0, self._E_word, E_inv0,
                                         tuple(terms[1]))
                    sols = []
                    for o1 in range(-4, 5):
                        s1 = {tuple(x + o1 * e
                                    for x, e in zip(u, self._E_chg))
                              for u in i1}
                        if not s1 <= set(prod):
                            continue
                        rest = set(prod) - s1
                        o2 = self._support_offset(
                            dict.fromkeys(rest, 1), i2) if rest else None
                        if o2 is not None:
                            sols.append((o1, o2))
                    if len(sols) == 1:
                        for d, o in zip(terms, sols[0]):
                            rows.append((*coeff_of(d), o))
        sol_nulls = None
        for eps in (1, -1):
            aug = []
            for coeff, eE, o in rows:
                row = [Fraction(0)] * len(letters)
                for z, m in coeff.items():
                    row[idx[z]] += m
                aug.append(row + [Fraction(o - eE * (eps - 1))])
            out = _gauss(aug, len(letters))
            if out is not None:
                sol_nulls = (eps, *out)
                break
        assert sol_nulls is not None, \
            "even-gon charge alignment has no solution"
        eps, c_sol, c_nulls = sol_nulls
        self._eps = eps
        # scan nullspace points |t|-ascending; per point, joint gauge solve
        tvals = sorted(range(-2, 3), key=abs)
        chosen = None
        for tvec in itertools.product(tvals, repeat=len(c_nulls)):
            cm = {}
            ok = True
            for g in letters:
                v = c_sol[idx[g]] + sum(
                    t * nv[idx[g]] for t, nv in zip(tvec, c_nulls))
                if v.denominator != 1:
                    ok = False
                    break
                cm[g] = int(v)
            if not ok:
                continue
            T = {g: self._dress(g, cm[g]) for g in letters}
            gs = self._gauge_solve(T)
            if gs is None:
                continue
            chosen = (tvec, cm, T, gs)
            break
        assert chosen is not None, \
            "even-gon gauge system inconsistent at every scanned " \
            "charge-nullspace point"
        tvec, cm, T, (xg, xE) = chosen
        self._c_offsets = cm
        self._gauges = xg
        self._x_E = xE
        self._T = {g: pp.scale(T[g], Q(xg[g])) for g in letters}
        base = self._E_word if eps == 1 else pp.inverse_monomial(self._E_word)
        self._E_img = pp.scale(base, Q(xE))
        self._E_img_inv = pp.inverse_monomial(self._E_img)

    def _gauge_solve(self, T):
        """The Δ-system for per-letter gauges + x_E on all
        generator-pair peels at zero gauges; returns
        ({letter: x}, x_E) or None if inconsistent."""
        pp = self.pp
        letters = self._letters
        E_GEN, E_INV = self._E_GEN, self._E_INV
        unknowns = letters + ["E"]
        uidx = {u: i for i, u in enumerate(unknowns)}
        base = self._E_word if getattr(self, "_eps", 1) == 1 \
            else pp.inverse_monomial(self._E_word)
        base_inv = pp.inverse_monomial(base)
        imgs = dict(T)
        aug = []
        for g in letters + [E_GEN, E_INV]:
            for h in letters + [E_GEN, E_INV]:
                a_l, b_l = self._lbl(g), self._lbl(h)
                ga = imgs[g] if g in imgs else (base if g == E_GEN
                                                else base_inv)
                gb = imgs[h] if h in imgs else (base if h == E_GEN
                                                else base_inv)
                prod = pp.multiply(ga, gb)
                hterms = {tuple(l): c for l, c in
                          self._intr.multiply(a_l, b_l).terms.items()}
                pool = {l: self._fwd_packaged(imgs, base, base_inv, l)
                        for l in hterms}
                meas = self._peel_raw(prod, pool)
                if meas is None:
                    return None
                for d, got in meas.items():
                    want = SkeinPentagonKAlg._chart_to_hl(hterms[d])
                    wi, gi = want.items(), got.items()
                    if len(wi) != 1 or len(gi) != 1:
                        continue
                    if wi[0][1] != gi[0][1]:
                        return None
                    row = [Fraction(0)] * len(unknowns)
                    for z in (g, h):
                        if z == E_GEN:
                            row[uidx["E"]] += 1
                        elif z == E_INV:
                            row[uidx["E"]] -= 1
                        else:
                            row[uidx[z]] += 1
                    lets, eE = self._word_letters(tuple(d))
                    for (l, m) in lets:
                        row[uidx[l]] -= m
                    row[uidx["E"]] -= eE
                    aug.append(row + [Fraction(wi[0][0] - gi[0][0])])
        out = _gauss(aug, len(unknowns))
        if out is None:
            return None
        sol, _nulls = out
        return ({g: sol[uidx[g]] for g in letters}, sol[uidx["E"]])

    def _peel_raw(self, prod, pool):
        """Unique-charge anchored peel; returns {label: measured
        HalfLaurent} or None if stuck / non-monomial where a monomial is
        needed."""
        pp = self.pp
        pool = dict(pool)
        rem = dict(prod)
        meas = {}
        while rem:
            hit = None
            for u in sorted(rem):
                owners = [l for l, c in pool.items() if u in c]
                if len(owners) != 1:
                    continue
                lam = owners[0]
                anchor = pool[lam][u].items()
                if len(anchor) == 1 and anchor[0][1] in (1, -1):
                    hit = (u, lam, anchor[0], None)
                    break
                r = SkeinPentagonKAlg._mono_ratio_hl(rem[u], pool[lam][u])
                if r is not None:
                    hit = (u, lam, None, r)
                    break
            if hit is None:
                return None
            u, lam = hit[0], hit[1]
            if hit[2] is not None:
                be, bv = hit[2]
                h = rem[u] * Q(-be, bv)
            else:
                e, sg = hit[3]
                h = Q(e) * sg
            piece = pp.scale(pool.pop(lam), h)
            rem = pp.add(rem, {v: -c for v, c in piece.items()})
            meas[lam] = h
        if pool:
            return None
        return meas

    # ---- public engine surface ---------------------------------------------

    def T(self, a: int, i: int) -> dict:
        """The dressed chord for intrinsic letter (a, i)."""
        return self._T[(a, i)]

    def E(self) -> dict:
        """The gauge-unit image (the interleaved side word, oriented and
        gauged)."""
        return self._E_img

    def gauges(self) -> dict:
        """The solved per-letter lq-gauges (+ 'E')."""
        out = {g: x for g, x in self._gauges.items()}
        out["E"] = self._x_E
        return out

    def kernel_offsets(self) -> dict:
        """The solved per-letter kernel offsets c_g (E-frame alignment)."""
        return dict(self._c_offsets)

    # ---- forward map ----------------------------------------------------------

    @staticmethod
    def _qexp1(poly):
        items = list(poly._coeffs.items())
        assert len(items) == 1 and items[0][1] == 1, poly
        return items[0][0]

    def forward(self, lbl) -> dict:
        lbl = tuple(lbl)
        if lbl in self._fwd_cache:
            return self._fwd_cache[lbl]
        pp = self.pp
        if lbl == self._intr.identity():
            out = pp.one()
        else:
            gens, powers = self._cd.to_cone_label(lbl)
            order = self._cd.canonical_cone_order(gens)
            acc = None
            img = None
            for g in order:
                glbl = self._cd.from_cone_label(frozenset({g}), {g: 1})
                if g == self._E_GEN:
                    gT = self._E_img
                elif g == self._E_INV:
                    gT = self._E_img_inv
                else:
                    gT = self._T[g]
                for _ in range(powers.get(g, 0)):
                    if acc is None:
                        acc = (glbl, Fraction(0))
                    else:
                        prod = self._intr.multiply(acc[0], glbl)
                        terms = dict(prod.terms)
                        assert len(terms) == 1, (lbl, g, terms)
                        nl = next(iter(terms))
                        acc = (nl, acc[1] + self._qexp1(terms[nl]))
                    img = gT if img is None else pp.multiply(img, gT)
            assert acc is not None and tuple(acc[0]) == lbl, (lbl, acc)
            out = pp.scale(img, Q(Fraction(-self.ORI * acc[1])))
        self._fwd_cache[lbl] = out
        return out

    # ---- KAlgebra primitives ----------------------------------------------------

    def coefficient_ring(self):
        return self._intr.coefficient_ring()

    def identity(self):
        return self._intr.identity()

    def multiply(self, a, b) -> Element:
        a = tuple(a)
        b = tuple(b)
        key = (a, b)
        if key in self._mult_cache:
            return self._mult_cache[key]
        pp = self.pp
        prod = pp.multiply(self.forward(a), self.forward(b))
        hint = self._intr.multiply(a, b)
        hterms = {tuple(l): c for l, c in hint.terms.items()}
        pool = {l: self.forward(l) for l in hterms}
        meas = self._peel_raw(prod, pool)
        assert meas is not None, f"peel stuck for {a} * {b}"
        out = {}
        for lam, h in meas.items():
            # THE STANDING GUARD (δ ≡ 0): measured == intrinsic verbatim
            # under q -> lq^{-2}.  Asserted, never fitted.
            want = SkeinPentagonKAlg._chart_to_hl(hterms[lam])
            assert h == want, (
                f"pinned/intrinsic coefficient mismatch at {lam} in "
                f"{a} * {b}: measured {h}, intrinsic {hterms[lam]}")
            out[lam] = hterms[lam]
        el = Element(out)
        self._mult_cache[key] = el
        return el

    def rho(self, a):
        r = self._intr.rho(tuple(a))
        if hasattr(r, "terms"):
            ts = dict(r.terms)
            assert len(ts) == 1
            return next(iter(ts))
        return r

    def rho_inverse(self, a):
        r = self._intr.rho_inverse(tuple(a))
        if hasattr(r, "terms"):
            ts = dict(r.terms)
            assert len(ts) == 1
            return next(iter(ts))
        return r

    def _canonical_rho2_orbit_rep(self, label):
        # infinite ρ²-orbits (E-drift shear): the intrinsic closed-form
        # drift quotient is the contract-required rep.
        return self._intr._canonical_rho2_orbit_rep(tuple(label))

    def trace(self, a, K: int = 20):
        return self._intr.trace(tuple(a), K)

    def _label_section_decompose(self, label):
        return (tuple(label), self.coefficient_ring().one())

    def r_label_decompose(self, label):
        """Trivial flavour-lift coordinate: the gauged A-odd theories
        are unflavoured."""
        return tuple(label), self.coefficient_ring().one_basis()

    def r_label_compose(self, section, r_basis_label):
        return tuple(section)

    # ---- the iso witness -----------------------------------------------------------

    def build_iso(self):
        from kalgebra_iso import KAlgebraIso
        one = LaurentPoly.one()

        def _id(lbl):
            return Element({tuple(lbl): one})

        return KAlgebraIso(self, self._intr, _id, _id,
                           name=f"{self.n}-gon[skein-pinned→u1a1aodd]")
