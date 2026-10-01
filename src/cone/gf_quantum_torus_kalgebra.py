"""`GfQuantumTorusKAlg` -- the G-flavoured quantum-torus K-algebra.

The principled non-abelian-flavour generalisation of `QuantumTorusKAlg`:
the **Weyl-invariant subalgebra of an abelian quantum torus**, for a
reductive flavour group `G` of a *single simple factor* whose **root
lattice is a sublattice of the flavour lattice** `Gamma_f = ker B`.

Data
----
* `pairing`: an antisymmetric integer matrix `B` on `Gamma = Z^n`
  (degenerate; `Gamma_f = ker B`).
* `reflections`: a list of `GL(Gamma, Z)` matrices `s_i`, the simple
  reflections, each a **Poisson involution** (`s_i^2 = id`,
  `s_i^T B s_i = B`).  They generate the flavour Weyl group `W`, which
  acts on `ker B` as the reflection representation of `Weyl(Phi)`; the
  roots lie in `ker B` (the hypothesis `Q(Phi) ⊆ ker B`).
* `ring`: the coefficient ring `R = R(T)^W` of the chosen **global form**
  of `G`; its character lattice `X^*(T)` is `ker B`.  Adjoint
  (`ker B = Q`): `SO3ZPlusRing`.  Simply connected (`ker B = P`):
  `SU2ZPlusRing`, `SU3ZPlusRing`.  Intermediate forms use a weight
  sublattice `Q ⊆ ker B ⊆ P`.
* `mu_basis`: the images under the alignment `phi: X^*(T)_ring -> ker B`
  of the unit vectors of the ring's `to_abelian` weight coordinates -- a
  `Z`-basis of `ker B`.  (`SU2`/`SO3`: one vector; `SU3`: the two units
  of the fundamental-orbit basis.)

The algebra `A_q[QT(Gamma, B)]^W`
---------------------------------
Each `s_i` preserves `B` and fixes the gauge quotient `Gamma / ker B`, so
the `W`-invariants form a sub-`KAlgebra` of `QuantumTorusKAlg(Gamma, B)`.
Canonical labels are **dominant `W`-orbit representatives** `gamma`; the
basis element is the **Weyl character** at `gamma`'s gauge cell.  `multiply`
lifts both labels to `X`-monomials, multiplies in the quantum torus, and
re-decomposes by peeling the most-dominant character inward; `rho` is
`gamma -> -gamma` then canonicalise; the trace is the QT trace, non-zero
only on the central characters, valued in `R = R(T)^W`.

Scope (first cut)
-----------------
A **single simple factor**, only the **integer ring-weight sector** (the
whole algebra when simply connected; the closed integer-weight subalgebra
when adjoint).  The fractional **spinor / virtual-character** sector (e.g.
the adjoint `SO3` half-integer doublets) raises.  Ring support: rank-1
(`SU2`, `SO3`) and `SU3`.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element
from zplus_ring import ZPlusRing, RElement, RPowerSeries
from snf_kernel import integer_kernel_and_section, decompose_in_basis
from qpoch import qpoch_infty
from laurent_poly import LaurentPoly


Vec = tuple[int, ...]
QVec = tuple[Fraction, ...]
Label = Vec


# ---------------------------------------------------------------------------
# linear algebra helpers
# ---------------------------------------------------------------------------


def _matvec(M, v):
    return tuple(sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M)))


def _matmul(A, B):
    n = len(A)
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n))
                 for i in range(n))


def _bilinear(B, a, b) -> int:
    n = len(B)
    return sum(int(B[i][j]) * int(a[i]) * int(b[j])
               for i in range(n) for j in range(n))


def _brow(B, w, j) -> int:
    return sum(int(B[j][i]) * int(w[i]) for i in range(len(B)))


def _solve_in_basis(basis, target, allow_fraction=False):
    """Coeffs `c` with `sum c_i basis[i] == target` (integer, or rational)."""
    r = len(basis)
    if r == 0:
        if any(x != 0 for x in target):
            raise ValueError("empty basis, nonzero target")
        return ()
    n = len(target)
    A = [[Fraction(basis[j][i]) for j in range(r)] + [Fraction(target[i])]
         for i in range(n)]
    piv = []
    row = 0
    for col in range(r):
        p = next((rr for rr in range(row, n) if A[rr][col] != 0), None)
        if p is None:
            raise ValueError("dependent basis")
        A[row], A[p] = A[p], A[row]
        pv = A[row][col]
        A[row] = [x / pv for x in A[row]]
        for rr in range(n):
            if rr != row and A[rr][col] != 0:
                f = A[rr][col]
                A[rr] = [a - f * b for a, b in zip(A[rr], A[row])]
        piv.append(col)
        row += 1
        if row == n:
            break
    sol = [Fraction(0)] * r
    for i, col in enumerate(piv):
        sol[col] = A[i][r]
    for i in range(n):
        if sum(sol[j] * Fraction(basis[j][i]) for j in range(r)) != Fraction(target[i]):
            raise ValueError(f"target {target} not in span")
    if allow_fraction:
        return tuple(sol)
    if any(x.denominator != 1 for x in sol):
        raise ValueError("non-integer solution")
    return tuple(int(x) for x in sol)


def _mat_inverse(M):
    n = len(M)
    A = [[Fraction(M[i][j]) for j in range(n)] + [Fraction(1 if i == j else 0)
                                                  for j in range(n)]
         for i in range(n)]
    for col in range(n):
        p = next((rr for rr in range(col, n) if A[rr][col] != 0), None)
        if p is None:
            raise ValueError("singular")
        A[col], A[p] = A[p], A[col]
        pv = A[col][col]
        A[col] = [x / pv for x in A[col]]
        for rr in range(n):
            if rr != col and A[rr][col] != 0:
                f = A[rr][col]
                A[rr] = [a - f * b for a, b in zip(A[rr], A[col])]
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            x = A[i][n + j]
            if x.denominator != 1:
                raise ValueError("inverse not integral")
            row.append(int(x))
        out.append(tuple(row))
    return tuple(out)


def _ring_torus_rank(ring) -> int:
    name = type(ring).__name__
    if name == "TensorZPlusRing":
        return sum(_ring_torus_rank(f) for f in ring.factors)
    if name == "TrivialZPlusRing":
        return 0
    # `if name in …`, not `.get(name) or …`: a rank of 0 is falsy, so the `or`
    # form would discard a legitimate rank-0 entry and fall through to None.
    known = {"SU2ZPlusRing": 1, "SO3ZPlusRing": 1, "SU3ZPlusRing": 2,
             "SU4ZPlusRing": 3}
    if name in known:
        return known[name]
    if name == "SUNZPlusRing":
        return int(ring.N) - 1
    if name == "AbelianZPlusRing":
        return int(ring.rank)
    return None


# ----- the flavour chart, as a recursion over the RING ---------------------
#
# A product flavour group `∏_a G_a` has weight coordinates that are the
# CONCATENATION of the factors' own, so dominance, the orbit-ordering key and
# the weight↔label maps all act block by block.  Written as a recursion over the
# ring rather than a per-kind switch, these are **shape-preserving**: on a bare
# simple ring they return exactly what the previous per-kind switches returned
# (a bare label), and on a `TensorZPlusRing` they return that ring's own tuple
# key — so extending to a product changes nothing for a single simple factor.
# Shared by `GfBPSKAlgebra` and `GfQuantumTorusKAlg`, which carried identical
# copies of the switch.

def _ring_kind(ring) -> str:
    name = type(ring).__name__
    if name in ("SU2ZPlusRing", "SO3ZPlusRing"):
        return "rank1"
    if name == "SU3ZPlusRing":
        return "su3"
    if name == "SUNZPlusRing":
        return "sun"
    if name == "TrivialZPlusRing":
        return "trivial"
    if name == "TensorZPlusRing":
        return "tensor"
    raise NotImplementedError(f"Gf flavour chart: ring {name} not supported")


def _ring_blocks(ring) -> tuple:
    """`((kind, width), …)` — one entry per SIMPLE factor, in coordinate order."""
    if _ring_kind(ring) == "tensor":
        out = []
        for f in ring.factors:
            out.extend(_ring_blocks(f))
        return tuple(out)
    if _ring_kind(ring) == "trivial":
        return ()
    return ((_ring_kind(ring), _ring_torus_rank(ring)),)


def _ring_is_dominant(ring, wt) -> bool:
    """Dominant iff dominant in EVERY factor's own block."""
    kind = _ring_kind(ring)
    if kind == "tensor":
        off = 0
        for f in ring.factors:
            w = _ring_torus_rank(f)
            if not _ring_is_dominant(f, tuple(wt[off:off + w])):
                return False
            off += w
        return True
    if kind == "trivial":
        return True
    if kind == "rank1":
        return wt[0] >= 0
    if kind == "su3":
        a, b = wt
        return 0 <= a <= b
    # sun: projected coords (w_1-w_N, …, w_{N-1}-w_N); dominant = weakly
    # decreasing and non-negative.
    return (all(wt[i] >= wt[i + 1] for i in range(len(wt) - 1))
            and (len(wt) == 0 or wt[-1] >= 0))


def _ring_dom_key(ring, wt) -> tuple:
    """Total order used to pick the dominant `W`-orbit representative.

    Concatenated block by block, so it orders the product lexicographically by
    factor — any total order will do, provided it is the SAME one everywhere.
    """
    kind = _ring_kind(ring)
    if kind == "tensor":
        out, off = (), 0
        for f in ring.factors:
            w = _ring_torus_rank(f)
            out = out + tuple(_ring_dom_key(f, tuple(wt[off:off + w])))
            off += w
        return out
    if kind == "trivial":
        return ()
    if kind == "rank1":
        return (wt[0],)
    if kind == "su3":
        return (wt[1], -wt[0])
    # `sun`: the projected coordinates ARE the key.  NOT `int(x)` — `canonicalise`
    # hands this the raw Fraction-valued `_weight` (unlike `_lift_to_F` and
    # `_decompose_F_into_L`, which guard with `_weight_int` / a denominator test
    # first), and truncating collapses `1/2` onto `0`, silently merging two
    # distinct orbits into one ordering key.  The `rank1` and `su3` branches
    # above keep the Fraction, so truncating here would also make the product
    # path behave differently from the single-factor one.
    return tuple(wt)


def _ring_wt_to_label(ring, wt):
    """A dominant weight ↦ the ring's own basis key (tuple-of-keys if a product)."""
    kind = _ring_kind(ring)
    if kind == "tensor":
        out, off = [], 0
        for f in ring.factors:
            w = _ring_torus_rank(f)
            out.append(_ring_wt_to_label(f, tuple(wt[off:off + w])))
            off += w
        return tuple(out)
    if kind == "trivial":
        return ()
    if kind == "rank1":
        return int(wt[0])
    if kind == "su3":
        a, b = int(wt[0]), int(wt[1])
        return (b - a, a)
    # sun: a dominant projected coord IS the partition (drop trailing zeros).
    w = [int(x) for x in wt]
    while w and w[-1] == 0:
        w.pop()
    return tuple(w)


def _ring_label_to_hw(ring, label) -> tuple:
    """Inverse of `_ring_wt_to_label`: the highest weight, in coordinates."""
    kind = _ring_kind(ring)
    if kind == "tensor":
        if len(label) != len(ring.factors):
            raise ValueError(
                f"Gf flavour chart: product label {label!r} must have "
                f"{len(ring.factors)} entries")
        out = ()
        for f, part in zip(ring.factors, label):
            out = out + tuple(_ring_label_to_hw(f, part))
        return out
    if kind == "trivial":
        return ()
    if kind == "rank1":
        return (int(label),)
    if kind == "su3":
        p, q = label
        return (q, p + q)
    # sun: pad the partition to the factor's torus rank with zeros.
    lam = [int(x) for x in label]
    return tuple(lam + [0] * (_ring_torus_rank(ring) - len(lam)))


class GfQuantumTorusKAlg(KAlgebra):
    """`A_q[QT(Gamma, B)]^W`, the `G`-flavoured quantum torus (see module doc)."""

    def __init__(self, pairing, reflections, ring, mu_basis):
        n = len(pairing)
        if any(len(row) != n for row in pairing):
            raise ValueError("pairing must be square")
        for i in range(n):
            for j in range(n):
                if int(pairing[i][j]) != -int(pairing[j][i]):
                    raise ValueError("pairing must be antisymmetric")
        self._B = [[int(x) for x in row] for row in pairing]
        self._rank = n
        self._R = ring
        self._ring_kind = self._classify_ring(ring)

        self._ker_basis, self._sec_basis = integer_kernel_and_section(self._B)
        self._ker_basis = [tuple(int(x) for x in v) for v in self._ker_basis]
        self._sec_basis = [tuple(int(x) for x in v) for v in self._sec_basis]
        self._gauge_rank = len(self._sec_basis)
        self._flavour_rank = len(self._ker_basis)

        self._reflections = [tuple(tuple(int(x) for x in row) for row in s)
                             for s in reflections]
        ident = tuple(tuple(1 if i == j else 0 for j in range(n)) for i in range(n))
        for idx, s in enumerate(self._reflections):
            if len(s) != n or any(len(row) != n for row in s):
                raise ValueError(f"reflection {idx} wrong shape")
            if _matmul(s, s) != ident:
                raise ValueError(f"reflection {idx} not an involution")
            sBs = [[sum(s[k][i] * self._B[k][l] * s[l][j]
                        for k in range(n) for l in range(n))
                    for j in range(n)] for i in range(n)]
            if sBs != self._B:
                raise ValueError(f"reflection {idx} not Poisson")

        self._W = self._generate_group(self._reflections)
        self._W_inv = [_mat_inverse(w) for w in self._W]

        self._mu_basis = [tuple(int(x) for x in v) for v in mu_basis]
        tr = _ring_torus_rank(self._R)
        if tr is None:
            raise NotImplementedError(f"unsupported ring {type(self._R).__name__}")
        if len(self._mu_basis) != tr:
            raise ValueError(f"mu_basis size {len(self._mu_basis)} != torus rank {tr}")
        if tr != self._flavour_rank:
            raise ValueError(f"flavour torus rank {tr} != "
                             f"rk(ker B) {self._flavour_rank}")
        for w in self._mu_basis:
            if any(_brow(self._B, w, j) != 0 for j in range(n)):
                raise ValueError(f"mu_basis vector {w} not in ker B")
        for kv in self._ker_basis:
            _solve_in_basis(self._mu_basis, kv)   # mu_basis spans ker B
        # roots ⊂ ker B (Q ⊆ ker B)
        for idx, s in enumerate(self._reflections):
            alpha = self._neg1_eig(s, n)
            try:
                _solve_in_basis(self._ker_basis, alpha)
            except ValueError:
                raise ValueError(f"reflection {idx}: root {alpha} not in ker B")

    # ----- setup helpers -------------------------------------------------

    @staticmethod
    def _classify_ring(ring) -> str:
        """Kept as the historical name; the chart itself is `_ring_*` above."""
        return _ring_kind(ring)

    def _generate_group(self, gens, cap=200000):
        n = self._rank
        I = tuple(tuple(1 if i == j else 0 for j in range(n)) for i in range(n))
        seen, frontier = {I}, [I]
        while frontier:
            cur = frontier.pop()
            for g in gens:
                nx = _matmul(cur, g)
                if nx not in seen:
                    seen.add(nx)
                    frontier.append(nx)
                    if len(seen) > cap:
                        raise ValueError(
                            f"flavour Weyl group exceeded the cap ({cap}) — this "
                            "is a BUDGET, not a finiteness verdict: for a product "
                            "flavour group the order is ∏_a N_a!, which multiplies "
                            "(SU(5)×SU(5) is 14400 and finite).  Raise `cap` if "
                            "that is intended.")
        return list(seen)

    @staticmethod
    def _neg1_eig(s, n) -> Vec:
        cols = [tuple(s[i][j] - (1 if i == j else 0) for i in range(n)) for j in range(n)]
        nz = [c for c in cols if any(x != 0 for x in c)]
        if not nz:
            raise ValueError("reflection is identity")
        a = nz[0]
        from math import gcd
        g = 0
        for x in a:
            g = gcd(g, abs(int(x)))
        return tuple(int(x) // g for x in a) if g > 1 else tuple(int(x) for x in a)

    # ----- flavour chart: one block per simple factor --------------------
    #
    # The same recursion `GfBPSKAlgebra` uses — this class carried an identical
    # copy of the per-kind switch, restricted to rank-1 and `SU3`.  Delegating
    # both removes the duplicate and widens this class to `SUNZPlusRing(N)` at
    # any `N` and to products, since the only ring operations it performs
    # (`to_abelian`, `basis_element`) are defined there too.

    def flavour_blocks(self) -> tuple:
        """`((kind, width), …)`, one entry per simple factor of `R`."""
        return _ring_blocks(self._R)

    def _is_dominant(self, wt) -> bool:
        return _ring_is_dominant(self._R, tuple(wt))

    def _dom_key(self, wt):
        return _ring_dom_key(self._R, tuple(wt))

    def _wt_to_label(self, wt):
        return _ring_wt_to_label(self._R, tuple(wt))

    def _label_to_hw(self, label) -> Vec:
        return _ring_label_to_hw(self._R, label)

    # ----- W-equivariant flavour weight ---------------------------------

    def _proj0(self, gamma) -> QVec:
        sec_c, ker_c = decompose_in_basis(list(gamma), self._sec_basis, self._ker_basis)
        n = self._rank
        return tuple(Fraction(sum(ker_c[i] * self._ker_basis[i][k]
                                  for i in range(len(ker_c)))) for k in range(n))

    def _proj_kerB(self, gamma) -> QVec:
        n = self._rank
        acc = [Fraction(0)] * n
        for w, winv in zip(self._W, self._W_inv):
            wp = _matvec(w, self._proj0(_matvec(winv, gamma)))
            for k in range(n):
                acc[k] += wp[k]
        m = len(self._W)
        return tuple(a / m for a in acc)

    def _weight(self, gamma) -> QVec:
        return _solve_in_basis(self._mu_basis, self._proj_kerB(gamma), allow_fraction=True)

    def _weight_int(self, gamma) -> Vec:
        mu = self._weight(gamma)
        if any(x.denominator != 1 for x in mu):
            raise NotImplementedError(
                f"GfQuantumTorusKAlg: spinor sector (fractional weight "
                f"{tuple(mu)} for {tuple(gamma)}) is out of scope"
            )
        return tuple(int(x) for x in mu)

    def _phi(self, wt) -> Vec:
        n = self._rank
        return tuple(sum(int(wt[i]) * self._mu_basis[i][k] for i in range(len(wt)))
                     for k in range(n))

    # ----- canonicalisation ---------------------------------------------

    def canonicalise(self, gamma: Vec) -> Label:
        """Dominant `W`-orbit representative of `gamma` (orbit enumeration)."""
        gamma = self._check(gamma)
        best, best_key = None, None
        for w in self._W:
            g = _matvec(w, gamma)
            wt = self._weight(g)
            if not self._is_dominant(wt):
                continue
            key = self._dom_key(wt)
            if best is None or key > best_key:
                best, best_key = g, key
        if best is None:
            raise ValueError(f"canonicalise: no dominant rep for {gamma}")
        return best

    # ----- L-basis lift to the quantum torus ----------------------------

    def _lift_to_qt(self, gamma: Label) -> dict[Vec, int]:
        g = self.canonicalise(gamma)
        wt = self._weight_int(g)                       # dominant lattice weight
        label = self._wt_to_label(wt)
        ab = self._R.to_abelian(self._R.basis_element(label))
        out: dict[Vec, int] = {}
        for key, mult in ab.terms.items():
            kk = tuple(int(x) for x in key)
            shift = self._phi(tuple(kk[i] - wt[i] for i in range(len(wt))))
            pt = tuple(g[k] + shift[k] for k in range(self._rank))
            out[pt] = out.get(pt, 0) + mult
        return out

    # ----- the seven KAlgebra primitives --------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self) -> Label:
        return tuple([0] * self._rank)

    def rho(self, a: Label) -> Label:
        return self.canonicalise(tuple(-int(x) for x in self._check(a)))

    def rho_inverse(self, a: Label) -> Label:
        return self.rho(a)

    def multiply(self, a: Label, b: Label) -> Element:
        a, b = self._check(a), self._check(b)
        ta, tb = self._lift_to_qt(a), self._lift_to_qt(b)
        qt: dict[Vec, dict[int, int]] = {}
        for ga, ca in ta.items():
            for gb, cb in tb.items():
                g = tuple(ga[i] + gb[i] for i in range(self._rank))
                e = _bilinear(self._B, ga, gb)
                slot = qt.setdefault(g, {})
                slot[e] = slot.get(e, 0) + ca * cb
        qt = {g: {e: c for e, c in d.items() if c} for g, d in qt.items()}
        qt = {g: d for g, d in qt.items() if d}
        return self._decompose_qt_into_L(qt)

    def _decompose_qt_into_L(self, qt_terms) -> Element:
        remaining = {g: dict(d) for g, d in qt_terms.items()}
        out: dict[Label, dict[int, int]] = {}
        while remaining:
            best, best_key = None, None
            for g in remaining:
                wt = self._weight(g)
                if any(x.denominator != 1 for x in wt) or not self._is_dominant(wt):
                    continue
                key = self._dom_key(wt)
                if best is None or key > best_key:
                    best, best_key = g, key
            if best is None:
                raise ValueError(f"_decompose: no in-scope dominant weight in "
                                 f"{list(remaining)} (spinor sector?)")
            label = self.canonicalise(best)
            chain = self._lift_to_qt(label)
            if chain.get(best, 0) != 1:
                raise ValueError(f"_decompose: highest-weight mult != 1 at {best}")
            coef = dict(remaining[best])
            for g, m in chain.items():
                if g not in remaining:
                    raise ValueError(f"_decompose: W-invariance failure at {g}")
                slot = remaining[g]
                for e, c in coef.items():
                    nv = slot.get(e, 0) - m * c
                    if nv == 0:
                        slot.pop(e, None)
                    else:
                        slot[e] = nv
                if not slot:
                    del remaining[g]
            acc = out.setdefault(label, {})
            for e, c in coef.items():
                acc[e] = acc.get(e, 0) + c
            out[label] = {e: c for e, c in acc.items() if c}
            if not out[label]:
                del out[label]
        return Element({lab: LaurentPoly(d) for lab, d in out.items()})

    def trace(self, a: Label, K: int = 20) -> RPowerSeries:
        a = self._check(a)
        if any(_brow(self._B, a, j) != 0 for j in range(self._rank)):
            return RPowerSeries(self._R, {}, K)
        label = self._wt_to_label(self._weight_int(self.canonicalise(a)))
        chi = self._R.basis_element(label)
        pref = qpoch_infty(K)
        for _ in range(self._gauge_rank - 1):
            pref = pref * qpoch_infty(K)
        out: dict[int, RElement] = {}
        for e, c in pref._c.items():
            r = chi * c
            if not r.is_zero():
                out[e] = r
        return RPowerSeries(self._R, out, K)

    def r_label_decompose(self, label: Label):
        """The flavour-lift coordinate `(section, ring_label)` (replaces the
        retired `_label_section_decompose`): peel the Weyl character
        `χ_{wt}` (the R-basis label) off the gauge cell.  The section
        `a − φ(wt)` has zero flavour weight — a central *source* label, as the
        `embed_R` faithfulness axiom needs.  `Γ_g` is abstract only after
        `forget()`; here the section is a section of `π : Γ ↠ Γ/ker B`."""
        a = self.canonicalise(label)
        wt = self._weight_int(a)
        section = tuple(a[k] - self._phi(wt)[k] for k in range(self._rank))
        if any(x != 0 for x in self._weight(section)):
            raise NotImplementedError(
                "GfQuantumTorusKAlg.r_label_decompose: non-central section "
                "(spinor sector) not supported"
            )
        return section, self._wt_to_label(wt)

    def r_label_compose(self, section, ring_label) -> Label:
        """Inverse of `r_label_decompose`: re-attach the Weyl character by
        shifting the central section by the weight lift `φ(hw(ring_label)) ∈
        ker B`, then canonicalise.  A direct label-producer (no
        `embed_R`/`multiply` round-trip)."""
        section = self._check(section)
        hw = self._label_to_hw(ring_label)
        shifted = tuple(section[k] + self._phi(hw)[k] for k in range(self._rank))
        return self.canonicalise(shifted)

    def embed_R(self, r: RElement) -> Element:
        """Central embedding `ι : R(G) ↪ A_𝖖`: `χ_{ring_label}` maps to the
        Weyl character at the gauge origin (`character_label`).  The weights
        `φ(·) ⊆ ker B` are central, so `embed_R(χ)·L_section = L_label` — the
        faithfulness axiom.  (The base default embeds only `1_R`, so
        `from_R_form` and the faithfulness verifier failed for non-trivial
        `χ`.)"""
        if not isinstance(r, RElement) or r.ring != self._R:
            raise TypeError(
                "embed_R: argument must be an RElement over coefficient_ring()"
            )
        out = Element.zero()
        for ring_label, c in r.terms.items():
            if c == 0:
                continue
            out = out + Element.basis(self.character_label(ring_label)) * c
        return out

    def gauge_class(self, gamma: Label) -> Vec:
        """Gauge class of `γ` in `Γ_g = Γ/ker B` — the SNF section
        coordinates (length `gauge_rank`)."""
        gamma = self._check(gamma)
        sec_c, _ = decompose_in_basis(
            list(gamma), self._sec_basis, self._ker_basis,
        )
        return tuple(sec_c)

    def forget(self) -> "KAlgebra":
        """Forget the `G` flavour: the (unflavoured) quantum torus on the now
        **abstract** gauge lattice `Γ_g = Γ/ker B`, i.e. `QuantumTorusKAlg(B_g)`
        with `B_g[i][j] = ⟨sec_basis[i], sec_basis[j]⟩` the induced
        (non-degenerate) pairing.

        The forget map `L_a ↦ dim(χ_{wt(a)})·M_{gauge_class(a)}` is a
        trace-preserving homomorphism: `dim` (the rep-ring augmentation) is
        multiplicative across Weyl-character fusion, and the gauge q-twist is
        exactly `B_g` on the quotient (`⟨ker B, ·⟩ = 0`)."""
        from quantum_torus_kalgebra import QuantumTorusKAlg
        g = self._gauge_rank
        B_g = [
            [_bilinear(self._B, self._sec_basis[i], self._sec_basis[j])
             for j in range(g)]
            for i in range(g)
        ]
        return QuantumTorusKAlg(B_g)

    def forget_label(self, label: Label) -> Vec:
        """The forget image of a label: its gauge class in the abstract
        `Γ_g` (= `forget()`'s basis label)."""
        return self.gauge_class(label)

    # ----- convenience ---------------------------------------------------

    def character_label(self, ring_label) -> Label:
        """The canonical label of the central character `chi_{ring_label}`
        (gauge cell at the origin)."""
        hw = self._label_to_hw(ring_label)
        return self.canonicalise(self._phi(hw))

    def ring_label_of(self, label: Label):
        """The `ring` basis label of a *central* canonical label (its
        Weyl character); raises in the spinor sector."""
        return self._wt_to_label(self._weight_int(self.canonicalise(label)))

    def L(self, label: Label) -> Element:
        return Element.basis(self.canonicalise(label))

    def expand_to_quantum_torus(self, elt: Element):
        from quantum_torus_kalgebra import QuantumTorusKAlg
        qt = QuantumTorusKAlg(pairing=self._B)
        out: dict[Vec, LaurentPoly] = {}
        for label, lp in elt.terms.items():
            for g, mult in self._lift_to_qt(self.canonicalise(label)).items():
                out[g] = (out[g] + lp * mult) if g in out else lp * mult
        out = {g: c for g, c in out.items() if not c.is_zero()}
        return qt, Element(out)

    @property
    def pairing(self):
        return [list(r) for r in self._B]

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def gauge_rank(self) -> int:
        return self._gauge_rank

    @property
    def mu_basis(self):
        return [tuple(v) for v in self._mu_basis]

    def _check(self, a) -> Vec:
        a = tuple(int(x) for x in a)
        if len(a) != self._rank:
            raise ValueError(f"label length {len(a)} != rank {self._rank}")
        return a

    def __repr__(self) -> str:
        return (f"GfQuantumTorusKAlg(rank={self._rank}, "
                f"gauge_rank={self._gauge_rank}, ring={self._R!r})")
