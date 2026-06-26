"""Authoritative sample K-algebras — **direct** `KAlgebra` subclasses on the
lift-coordinate surface.

Each class subclasses `KAlgebra` *directly* (no `ConeKAlgebra` /
`QuantumTorusKAlg` / URQTorus engine), implements the six abstract primitives
plus the optional lift-coordinate methods `r_label_decompose` /
`r_label_compose`, and does **not** implement the to-be-retired
`_label_section_decompose` / `embed_R` (the former is supplied by the base
default, which bridges to `r_label_decompose`).

These are reference examples of "how to write a K_𝖖-algebra" against the
current contract (Ambrosino–Gaiotto, §"K_𝖖-algebras"):

  * `Z2QTorusSampleKAlgebra` — `Q_𝖖(Z²)`, the canonical example (unflavoured):
    `X_γ X_{γ'} = 𝖖^{⟨γ,γ'⟩} X_{γ+γ'}`, `ρ(γ) = −γ`,
    `Tr X_γ = (𝖖²)_∞^{rk Γ} δ_{γ,0}`.
  * `PentagonSampleKAlgebra` — `K_𝖖([A_1, A_2])`, the Yang-Lee / M(2,5) pentagon.

For unflavoured algebras the lift coordinate is trivial:
`r_label_decompose(a) = (a, ())` and `r_label_compose(a, ()) = a`.  The
flavoured samples (SQED₂ = U_𝖖(𝔰𝔩₂), [A_1,D_3]) — to follow — carry an SU(2)
flavour, where the section is the gauge monomial and the R-label is χ_k.

The pentagon's pure reduction / Nahm-sum helpers are shared (imported) with
`kalgebra_samples`; they are plain module functions (not `ConeKAlgebra`), so the
sample *class* stays a direct `KAlgebra` subclass.
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # repo root
_IMPL = os.path.join(_ROOT, "implementations")       # uq_sl2_pbw lives here
for _p in (_ROOT, _HERE, _IMPL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from laurent_poly import LaurentPoly
from zplus_ring import ZPlusRing, RPowerSeries, TrivialZPlusRing
from kalgebra import KAlgebra, Element, _laurentpoly_times_rpowerseries

# Pure (engine-free) pentagon helpers: the three-letter reduction and the
# Rogers-Ramanujan Nahm-sum traces.  These are plain functions, not part of any
# ConeKAlgebra machinery.

# U_q(sl_2) PBW pure reducer (Casimir C = χ_1 carried as a central C^j power) —
# the gauge-side engine for the SU(2)-flavoured SQED_2 sample.  These are plain
# module functions, so `SQED2SampleKAlgebra` stays a direct `KAlgebra` subclass.
from uq_sl2_pbw import (
    multiply_canonical as _uqsl2_multiply,
    rho_label as _uqsl2_rho,
    rho_inverse_label as _uqsl2_rho_inv,
)
from zplus_ring import SU2ZPlusRing, SUNZPlusRing, RElement

# SU(N_f) character-ring un-branching (type-A Weyl / Schur) for the SQED_{N_f}
# trace — pure functions; `SQEDNfSampleKAlgebra` stays a direct `KAlgebra`
# subclass (NO RGKAlgebra / spine machinery).
import sun_characters as _sun



# --- inlined pentagon helpers + caches (self-contained; no ConeKAlgebra) ---

def _pent_idx(i: int) -> int:
    return i % 5

def _pent_canon_key(i: int, a: int, b: int) -> tuple[int, int, int]:
    """Canonical (i, a, b) basis label.  Unit is uniquely (0,0,0);
    `L_{i;0,b} = L_{i+1}^b` collapses to `(i+1, b, 0)`."""
    if a == 0 and b == 0:
        return (0, 0, 0)
    if b == 0:
        return (_pent_idx(i), a, 0)
    if a == 0:
        return (_pent_idx(i + 1), b, 0)
    return (_pent_idx(i), a, b)

def _pent_normalize_mono(letters):
    """Drop zero exponents, merge consecutive equal indices, reduce mod 5."""
    out: list[tuple[int, int]] = []
    for i, e in letters:
        if e == 0:
            continue
        ii = _pent_idx(i)
        if out and out[-1][0] == ii:
            out[-1] = (ii, out[-1][1] + e)
        else:
            out.append((ii, e))
    return tuple(out)

def _pent_is_basis_form(m) -> bool:
    """Monomial is already L_i^a L_{i+1}^b shape."""
    if len(m) <= 1:
        return True
    return len(m) == 2 and _pent_idx(m[1][0] - m[0][0]) == 1

def _pent_step(coeff, m):
    """One reduction step on `coeff * m`.  Returns `None` iff `m` is
    already in basis form; otherwise a list of `(coeff, m)` summands."""
    n = len(m)
    if _pent_is_basis_form(m):
        return None
    # First pass: real moves (merge, descending swap, Plücker).
    for k in range(n - 1):
        (i, e), (j, f) = m[k], m[k + 1]
        di = _pent_idx(j - i)
        if di == 0:
            new = m[:k] + ((i, e + f),) + m[k + 2:]
            return [(coeff, _pent_normalize_mono(new))]
        if di == 4:  # descending adjacent: L_i L_{i-1} -> q² L_{i-1} L_i.
            tw = LaurentPoly.q(2 * e * f)
            new = m[:k] + ((j, f), (i, e)) + m[k + 2:]
            return [(coeff * tw, _pent_normalize_mono(new))]
        if di == 2:  # Plücker: L_i L_{i+2} = 1 + q⁻¹ L_{i+1}.
            left = m[:k] + (((i, e - 1),) if e > 1 else ())
            right = (((j, f - 1),) if f > 1 else ()) + m[k + 2:]
            mid_one = _pent_normalize_mono(left + right)
            mid_L = _pent_normalize_mono(left + ((i + 1, 1),) + right)
            return [(coeff,                       mid_one),
                    (coeff * LaurentPoly.q(-1),   mid_L)]
        if di == 3:  # Plücker: L_i L_{i-2} = 1 + q L_{i-1}.
            left = m[:k] + (((i, e - 1),) if e > 1 else ())
            right = (((j, f - 1),) if f > 1 else ()) + m[k + 2:]
            mid_one = _pent_normalize_mono(left + right)
            mid_L = _pent_normalize_mono(left + ((i - 1, 1),) + right)
            return [(coeff,                       mid_one),
                    (coeff * LaurentPoly.q(1),    mid_L)]
        # di == 1: ascending adjacent — skip in this pass.
    # All adjacencies are di=1, length >= 3.  Swap rightmost pair
    # (cost q^{-2ef}) to expose a di=2 pair on its left.
    k = n - 2
    (i, e), (j, f) = m[k], m[k + 1]
    tw = LaurentPoly.q(-2 * e * f)
    new = m[:k] + ((j, f), (i, e)) + m[k + 2:]
    return [(coeff * tw, _pent_normalize_mono(new))]

def _pent_reduce(coeff, m):
    """Reduce `coeff * m` to a basis dict `{(i,a,b): LaurentPoly}`."""
    work = [(coeff, _pent_normalize_mono(m))]
    out: dict[tuple[int, int, int], LaurentPoly] = {}
    while work:
        c, mm = work.pop()
        if c.is_zero():
            continue
        if _pent_is_basis_form(mm):
            if len(mm) == 0:
                key, a, b = (0, 0, 0), 0, 0
            elif len(mm) == 1:
                i_, a = mm[0]
                key, b = (_pent_idx(i_), a, 0), 0
            else:
                (i_, a), (_, b) = mm
                key = (_pent_idx(i_), a, b)
            # L_i^a L_{i+1}^b = q^{-ab} L_{i;a,b}.
            adj = c * LaurentPoly.q(-a * b)
            cur = out.get(key, LaurentPoly.zero())
            s = cur + adj
            if s.is_zero():
                out.pop(key, None)
            else:
                out[key] = s
            continue
        for nc, nm in _pent_step(c, mm):
            work.append((nc, nm))
    return out

_pent_trpow_cache: dict[int, tuple[LaurentPoly, LaurentPoly]] = {}

def _pent_tr_power_coeffs(n: int) -> tuple[LaurentPoly, LaurentPoly]:
    """Return `(c1, cL)` with `Tr(L^n) = c1 * Tr(1) + cL * Tr(L)`."""
    if n in _pent_trpow_cache:
        return _pent_trpow_cache[n]
    if n == 0:
        out = (LaurentPoly.one(), LaurentPoly.zero())
    elif n == 1:
        out = (LaurentPoly.zero(), LaurentPoly.one())
    else:
        a1, b1 = _pent_tr_power_coeffs(n - 1)
        a2, b2 = _pent_tr_power_coeffs(n - 2)
        q1 = LaurentPoly.q(1 - 2 * n)
        q2 = LaurentPoly.q(2 - 2 * n)
        out = (q1 * a1 + q2 * a2, q1 * b1 + q2 * b2)
    _pent_trpow_cache[n] = out
    return out

def _pent_chi_q2(use_n_squared: bool, K: int) -> dict[int, int]:
    """Coefficients of χ₀(q²) or χ₁(q²) truncated to q^K, as a sparse
    dict {q-exponent: int}.  `use_n_squared=True` selects χ₁ (shift n²),
    `False` selects χ₀ (shift n(n+1))."""
    total: dict[int, int] = {}
    inv: dict[int, int] = {0: 1}  # 1 / (q²;q²)_0 = 1
    nn = 0
    while True:
        shift = 2 * nn * nn if use_n_squared else 2 * nn * (nn + 1)
        if shift > K:
            break
        for e, c in inv.items():
            if e + shift > K:
                continue
            total[e + shift] = total.get(e + shift, 0) + c
            if total[e + shift] == 0:
                del total[e + shift]
        # Advance inv: 1/(q²;q²)_{nn+1} = 1/(q²;q²)_{nn} * 1/(1 - q^{2(nn+1)}).
        step = 2 * (nn + 1)
        for e in range(step, K + 1):
            prev = inv.get(e - step, 0)
            if prev:
                inv[e] = inv.get(e, 0) + prev
                if inv[e] == 0:
                    del inv[e]
        nn += 1
    return total

_pent_tr_cache: dict[tuple[str, int], "RPowerSeries"] = {}

def _pent_tr_1_rps(R: ZPlusRing, K: int) -> "RPowerSeries":
    key = ("Tr1", K)
    if key in _pent_tr_cache:
        return _pent_tr_cache[key]
    coeffs = _pent_chi_q2(use_n_squared=False, K=K)
    out = RPowerSeries(R, coeffs, K)
    _pent_tr_cache[key] = out
    return out

def _pent_tr_L_rps(R: ZPlusRing, K: int) -> "RPowerSeries":
    """Tr(L) = q⁻¹ (χ₀(q²) - χ₁(q²)) truncated to q^K."""
    key = ("TrL", K)
    if key in _pent_tr_cache:
        return _pent_tr_cache[key]
    inner_K = K + 1  # q⁻¹ shift consumes one order
    chi0 = _pent_chi_q2(use_n_squared=False, K=inner_K)
    chi1 = _pent_chi_q2(use_n_squared=True, K=inner_K)
    diff: dict[int, int] = {}
    for e, c in chi0.items():
        diff[e] = diff.get(e, 0) + c
    for e, c in chi1.items():
        diff[e] = diff.get(e, 0) - c
        if diff[e] == 0:
            del diff[e]
    shifted = {e - 1: c for e, c in diff.items() if e - 1 <= K}
    out = RPowerSeries(R, shifted, K)
    _pent_tr_cache[key] = out
    return out

__all__ = [
    "Z2QTorusSampleKAlgebra",
    "PentagonSampleKAlgebra",
    "SQED1SampleKAlgebra",
    "SQED2SampleKAlgebra",
    "SQEDNfSampleKAlgebra",
]


# ---------------------------------------------------------------------------
# (q²;q²)_∞^rank truncated to q^K, self-contained.
# ---------------------------------------------------------------------------


def _qpoch_q2_pow(rank: int, K: int) -> dict[int, int]:
    """`(𝖖²;𝖖²)_∞^rank = ∏_{n≥1}(1 − 𝖖^{2n})^rank` truncated to `𝖖^K`,
    as a sparse `{exponent: int}` dict."""
    base = {0: 1}
    n = 1
    while 2 * n <= K:
        nxt = dict(base)
        for e, c in base.items():
            ee = e + 2 * n
            if ee <= K:
                nxt[ee] = nxt.get(ee, 0) - c
        base = {e: c for e, c in nxt.items() if c}
        n += 1
    out = {0: 1}
    for _ in range(rank):
        prod: dict[int, int] = {}
        for e1, c1 in out.items():
            for e2, c2 in base.items():
                ee = e1 + e2
                if ee <= K:
                    prod[ee] = prod.get(ee, 0) + c1 * c2
        out = {e: c for e, c in prod.items() if c}
    return out


# ---------------------------------------------------------------------------
# Z² quantum torus  Q_𝖖(Z²)  with the symplectic pairing.
# ---------------------------------------------------------------------------


class Z2QTorusSampleKAlgebra(KAlgebra):
    """The canonical example: the quantum torus `Q_𝖖(Γ)` for `Γ = Z²` with the
    non-degenerate antisymmetric pairing `⟨(a,b),(c,d)⟩ = ad − bc`.

    Canonical basis `{X_γ : γ ∈ Z²}`, `X_γ X_{γ'} = 𝖖^{⟨γ,γ'⟩} X_{γ+γ'}`; the
    bar involution fixes each `X_γ`; `ρ(γ) = −γ` (so `ρ² = id`); and
    `Tr X_γ = (𝖖²)_∞^{rk Γ} δ_{γ,0} = (𝖖²;𝖖²)_∞² δ_{γ,0}`, giving
    `I_{γ,γ'} = (𝖖²;𝖖²)_∞² δ_{γ,γ'}`.

    Unflavoured (the pairing is non-degenerate, so `Γ_f = 0`): the lift
    coordinate is trivial."""

    _R = TrivialZPlusRing()

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return (0, 0)

    @staticmethod
    def _pairing(g, h) -> int:
        return g[0] * h[1] - g[1] * h[0]

    def multiply(self, a, b) -> Element:
        c = self._pairing(a, b)
        return Element({(a[0] + b[0], a[1] + b[1]): LaurentPoly({c: 1})})

    def rho(self, a):
        return (-a[0], -a[1])

    def rho_inverse(self, a):
        return (-a[0], -a[1])

    def rho_squared_is_identity(self) -> bool:
        return True

    def trace(self, a, K: int = 20) -> RPowerSeries:
        if a != (0, 0):
            return RPowerSeries.zero(self._R, K)
        return RPowerSeries(self._R, _qpoch_q2_pow(2, K), K)

    # ---- lift coordinate (trivial: unflavoured) ----

    def r_label_decompose(self, label):
        return label, ()                       # section = label, χ = trivial ()

    def r_label_compose(self, section, r_basis_label):
        return section

    def __repr__(self) -> str:
        return "Z2QTorusSampleKAlgebra()"


# ---------------------------------------------------------------------------
# Pentagon  K_𝖖([A_1, A_2]).
# ---------------------------------------------------------------------------


class PentagonSampleKAlgebra(KAlgebra):
    """The pentagon algebra `K_𝖖([A_1, A_2])` (A_2 Argyres-Douglas / Yang-Lee).

    Generators `L_i`, `i ∈ Z/5`, with the quantum-Ptolemy relations
    `L_{i+1} L_i = 𝖖² L_i L_{i+1}`, `L_{i+1} L_{i-1} = 1 + 𝖖 L_i`,
    `L_{i-1} L_{i+1} = 1 + 𝖖⁻¹ L_i`.  Canonical basis
    `L_{i;a,b} = 𝖖^{ab} L_i^a L_{i+1}^b`, labelled `(i, a, b) ∈ Z/5 × Z_{≥0}²`
    and canonicalised so the unit is `(0,0,0)` and `L_i^a = (i, a, 0)`.
    `ρ(L_i) = L_{i+2}` generates Z/5.

    Trace (two layers, ρ²-twisted-cyclic):
    `Tr L_{i;a,b} = 𝖖^{ab} Tr L_i^{a+b}`, with the Schur-like recursion
    `Tr L^n = 𝖖^{1−2n} Tr L^{n−1} + 𝖖^{2−2n} Tr L^{n−2}` reducing every trace
    to `Tr 1` and `Tr L`, supplied by the M(2,5) characters
    `Tr 1 = χ₀(𝖖²)`, `Tr L = 𝖖⁻¹(χ₀(𝖖²) − χ₁(𝖖²))` (Rogers-Ramanujan Nahm
    sums).  Unflavoured."""

    _R = TrivialZPlusRing()

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return (0, 0, 0)

    def multiply(self, a, b) -> Element:
        """`L_{i1;x1,y1} · L_{i2;x2,y2}` reduced to the canonical basis.

        `L_{i;a,b} = 𝖖^{ab} L_i^a L_{i+1}^b`, so the product is
        `𝖖^{x1 y1 + x2 y2}` times the four-letter word
        `L_{i1}^{x1} L_{i1+1}^{y1} L_{i2}^{x2} L_{i2+1}^{y2}`, reduced by the
        Ptolemy relations (`_pent_reduce`)."""
        i1, x1, y1 = a
        i2, x2, y2 = b
        coeff = LaurentPoly.q(x1 * y1 + x2 * y2)
        word = ((i1, x1), (i1 + 1, y1), (i2, x2), (i2 + 1, y2))
        red = _pent_reduce(coeff, word)
        return Element({k: v for k, v in red.items() if not v.is_zero()})

    def rho(self, a):
        i, x, y = a
        return _pent_canon_key(i + 2, x, y)

    def rho_inverse(self, a):
        i, x, y = a
        return _pent_canon_key(i - 2, x, y)

    def trace(self, a, K: int = 20) -> RPowerSeries:
        i, x, y = a
        n = x + y
        c1, cL = _pent_tr_power_coeffs(n)         # Tr L^n = c1·Tr1 + cL·TrL
        # c1, cL carry deep negative 𝖖-powers (~ −(n²−1)); compute Tr1/TrL to a
        # generous inner order so the combination is exact to O(𝖖^K).
        inner = K + 2 * n * n + 10
        tr1 = _pent_tr_1_rps(self._R, inner)
        trL = _pent_tr_L_rps(self._R, inner)
        comb = (_laurentpoly_times_rpowerseries(c1, tr1)
                + _laurentpoly_times_rpowerseries(cL, trL))
        # Tr L_{i;a,b} = 𝖖^{ab} Tr L^n; apply the prefactor and truncate to K.
        shifted = _laurentpoly_times_rpowerseries(LaurentPoly.q(x * y), comb)
        return RPowerSeries(
            self._R, {e: c for e, c in shifted.coeffs.items() if e <= K}, K)

    # ---- lift coordinate (trivial: unflavoured) ----

    def r_label_decompose(self, label):
        return label, ()

    def r_label_compose(self, section, r_basis_label):
        return section

    def __repr__(self) -> str:
        return "PentagonSampleKAlgebra()"


# ---------------------------------------------------------------------------
# SQED_2 = U_𝖖(𝔰𝔩₂), SU(2)-flavoured.
# ---------------------------------------------------------------------------
#
# The gauge reduction is the U_𝖖(sl_2) PBW straightening (`_uqsl2_multiply`),
# which carries the Casimir as a central power `C^j` (`C = χ_1`).  The flavoured
# sample promotes that central direction to genuine SU(2) characters: each `C^j`
# becomes `χ_1^{⊗ j}`, fused with the input characters by Clebsch-Gordan.  So
# `multiply` stays Z-valued, with the flavour irrep carried in the label.


def _su2_fuse(R, d1: dict, d2: dict) -> dict:
    """Clebsch-Gordan fuse two `{spin: int}` character combinations."""
    out: dict = {}
    for a, ca in d1.items():
        for b, cb in d2.items():
            for c, m in R.multiply_basis(a, b).items():
                out[c] = out.get(c, 0) + ca * cb * m
    return {k: v for k, v in out.items() if v}


def _eq_coeffs(K: int) -> list:
    """`e_i(𝖖) = [y^i] E_𝖖(y) = (−1)^i 𝖖^i / (𝖖²;𝖖²)_i`, as `LaurentPoly`
    truncated to `𝖖^K`, for `i = 0..K`."""
    inv = LaurentPoly({0: 1})                     # 1/(𝖖²;𝖖²)_0
    out = []
    for i in range(K + 1):
        ei = LaurentPoly({e: c for e, c in
                          (inv * LaurentPoly({i: (-1) ** i}))._coeffs.items()
                          if e <= K})
        out.append(ei)
        step = 2 * (i + 1)                         # advance inv by 1/(1−𝖖^step)
        if step <= K:
            new: dict = {}
            for e, c in inv._coeffs.items():
                t = 0
                while e + step * t <= K:
                    new[e + step * t] = new.get(e + step * t, 0) + c
                    t += 1
            inv = LaurentPoly({e: c for e, c in new.items() if c})
    return out


def _mu_weyl_to_su2(mud: dict) -> dict:
    """A Weyl-symmetric `{μ-power: int}` Laurent polynomial → `{spin: int}`
    SU(2)-character content (peel `χ_k = μ^k + μ^{k−2} + … + μ^{−k}` from the
    top)."""
    rem = {m: c for m, c in mud.items() if c}
    out: dict = {}
    while rem:
        kmax = max(rem)
        if kmax < 0:
            break
        c = rem[kmax]
        out[kmax] = out.get(kmax, 0) + c
        for e in range(-kmax, kmax + 1, 2):
            rem[e] = rem.get(e, 0) - c
            if rem[e] == 0:
                rem.pop(e, None)
    return {k: v for k, v in out.items() if v}


def _sqed2_trK(n: int, K: int) -> RPowerSeries:
    """`Tr K^n ∈ R(SU(2))((𝖖))` via the paper's `G(x,μ) = (𝖖²;𝖖²)_∞²
    ∏_{s,t=±1} E_𝖖(μ^s x^t)`, `Tr K^n = [x^n] G` — computed exactly to `𝖖^K`."""
    R = SU2ZPlusRing()
    e = _eq_coeffs(K)
    acc: dict = {}                                # {μ-power: LaurentPoly}
    for i1 in range(K + 1):
        for i2 in range(K + 1 - i1):
            for i3 in range(K + 1 - i1 - i2):
                rem = K - i1 - i2 - i3
                for i4 in range(rem + 1):
                    if i1 + i2 - i3 - i4 != n:
                        continue
                    coeff = e[i1] * e[i2] * e[i3] * e[i4]
                    coeff = LaurentPoly({ex: c for ex, c in coeff._coeffs.items()
                                         if ex <= K})
                    if coeff.is_zero():
                        continue
                    md = i1 - i2 + i3 - i4         # μ-power: +i1 −i2 +i3 −i4
                    acc[md] = acc.get(md, LaurentPoly.zero()) + coeff
    pref = LaurentPoly(_qpoch_q2_pow(2, K))       # (𝖖²;𝖖²)_∞²
    byq: dict = {}                                # 𝖖-exp → {μ-power: int}
    for md, lp in acc.items():
        prod = LaurentPoly({ex: c for ex, c in (lp * pref)._coeffs.items()
                            if ex <= K})
        for ex, c in prod._coeffs.items():
            if c:
                byq.setdefault(ex, {})[md] = byq.setdefault(ex, {}).get(md, 0) + c
    coeffs: dict = {}
    for ex, mud in byq.items():
        chi = _mu_weyl_to_su2(mud)
        if chi:
            coeffs[ex] = RElement(R, chi)
    return RPowerSeries(R, coeffs, K)


class SQED2SampleKAlgebra(KAlgebra):
    """SQED₂ = U_𝖖(𝔰𝔩₂), the SU(2)-flavoured K-algebra (U(1) gauge, two hypers).

    Canonical basis `(gauge, k)`: `gauge ∈ {('K',n), ('E',a,b), ('F',a,b)}` (the
    U_𝖖(𝔰𝔩₂) PBW monomials `K^n`, `E_{a,b}=𝖖^{−ab}E^aK^b`, `F_{a,b}=𝖖^{ab}F^aK^b`)
    dressed by the SU(2) character `χ_k` (`k ≥ 0` the spin).  The pure characters
    `(('K',0), k)` are basis elements; in particular `χ_1` is `(('K',0), 1)`.

    Relations (`χ_1` = the fundamental SU(2) character): `KE=𝖖⁻²EK`, `KF=𝖖²FK`,
    `EF=χ_1+𝖖K+𝖖⁻¹K⁻¹`, `FE=χ_1+𝖖⁻¹K+𝖖K⁻¹`.  `multiply` reuses the pure PBW
    straightener (`_uqsl2_multiply`, Casimir as `C^j`) and promotes each `C^j` to
    `χ_1^{⊗j}` fused with the input characters by Clebsch-Gordan — Z-valued, with
    flavour in the label.  `ρ` is Lusztig's braid (`K↦K⁻¹`, `E↦𝖖⁻¹FK⁻¹`,
    `F↦𝖖KE`), χ_k fixed (SU(2) self-dual).  The trace localises to the Cartan
    sector: `Tr (K^n·χ_k) = χ_k · Tr K^n` with `Tr K^n = [x^n]G(x,μ)` the paper's
    Schur index; everything with net E/F charge has zero trace.

    The lift coordinate is genuinely flavoured: `r_label_decompose((g,k)) =
    ((g,0), k)` (section = the undressed gauge monomial, R-label = the spin `k`)."""

    _R = SU2ZPlusRing()

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return (("K", 0), 0)

    @staticmethod
    def _to_pbw(g):
        return g + (0,)                           # ('K',n)->('K',n,0); etc. (j=0)

    def multiply(self, a, b) -> Element:
        (g1, k1), (g2, k2) = a, b
        prod = _uqsl2_multiply(self._to_pbw(g1), self._to_pbw(g2))
        R = self._R
        out: dict = {}
        for plab, coeff in prod.items():
            gauge, jc = plab[:-1], plab[-1]       # strip the Casimir power j
            fused = _su2_fuse(R, {k1: 1}, {k2: 1})
            for _ in range(jc):                   # C^j = χ_1^{⊗ j}
                fused = _su2_fuse(R, fused, {1: 1})
            for l, m in fused.items():
                lab = (gauge, l)
                t = coeff * m
                out[lab] = t if lab not in out else out[lab] + t
        return Element({k: v for k, v in out.items() if not v.is_zero()})

    def rho(self, a):
        g, k = a
        return (_uqsl2_rho(self._to_pbw(g))[:-1], k)

    def rho_inverse(self, a):
        g, k = a
        return (_uqsl2_rho_inv(self._to_pbw(g))[:-1], k)

    def trace(self, a, K: int = 20) -> RPowerSeries:
        g, k = a
        R = self._R
        if g[0] != "K":                            # net E/F charge ⇒ zero trace
            return RPowerSeries.zero(R, K)
        return _sqed2_trK(g[1], K) * R.basis_element(k)

    # ---- lift coordinate (genuinely flavoured: section = gauge, R-label = spin) ----

    def r_label_decompose(self, label):
        g, k = label
        return (g, 0), k                           # section (g, χ_0), char χ_k

    def r_label_compose(self, section, r_basis_label):
        g, _zero = section
        return (g, r_basis_label)

    def __repr__(self) -> str:
        return "SQED2SampleKAlgebra()"


# ---------------------------------------------------------------------------
# SQED_1 = U1Square: U(1) gauge + one charged hyper (unflavoured).
# ---------------------------------------------------------------------------
#
# Generators u_± (dressed monopoles) and v (gauge monopole / Coulomb variable),
# relations  u_+ v = 𝖖² v u_+,  u_- v = 𝖖⁻² v u_-,  u_+ u_- = 1 + 𝖖 v,
# u_- u_+ = 1 + 𝖖⁻¹ v.  Labels (m, n): `L_{m,n} = 𝖖^{−mn} u_+^m v^n` (m>0),
# `𝖖^{|m|n} u_-^{|m|} v^n` (m<0), `v^n` (m=0).  Note the uniform prefactor
# `𝖖^{−mn}` and: moving `v^{n1}` past `u_{·}^{m2}` costs `𝖖^{−2 n1 m2}`;
# `u^m v^N = 𝖖^{mN} L_{m,N}`.


def _sqed1_ureduce(c1: int, c2: int) -> dict:
    """Reduce the u-product `u_{sgn c1}^{|c1|} · u_{sgn c2}^{|c2|}` to
    `{(net_charge, v_extra): LaurentPoly}` (net charge `= c1 + c2`).  Same sign
    or a zero factor: a single block.  Opposite signs: peel via
    `u_+u_- = 1 + 𝖖 v` / `u_-u_+ = 1 + 𝖖⁻¹ v`, each peel commuting the emitted
    `v` past the surviving opposite block."""
    if c1 == 0 or c2 == 0 or (c1 > 0) == (c2 > 0):
        return {(c1 + c2, 0): LaurentPoly.one()}
    net = c1 + c2
    out: dict = {}
    if c1 > 0:                                        # u_+^{c1} u_-^{|c2|}
        stack = [(c1, -c2, 0, LaurentPoly.one())]     # (a_+, b_-, v_extra, coeff)
        while stack:
            aa, bb, ve, co = stack.pop()
            if aa == 0 or bb == 0:
                k = (net, ve)
                out[k] = co if k not in out else out[k] + co
            else:
                stack.append((aa - 1, bb - 1, ve, co))
                stack.append((aa - 1, bb - 1, ve + 1,
                              co * LaurentPoly.q(2 * bb - 1)))
    else:                                             # u_-^{|c1|} u_+^{c2}
        stack = [(-c1, c2, 0, LaurentPoly.one())]     # (a_-, b_+, v_extra, coeff)
        while stack:
            aa, bb, ve, co = stack.pop()
            if aa == 0 or bb == 0:
                k = (net, ve)
                out[k] = co if k not in out else out[k] + co
            else:
                stack.append((aa - 1, bb - 1, ve, co))
                stack.append((aa - 1, bb - 1, ve + 1,
                              co * LaurentPoly.q(1 - 2 * bb)))
    return {k: v for k, v in out.items() if not v.is_zero()}


def _sqed1_trvn(n: int, K: int) -> dict:
    """`Tr v^n = [x^n] (𝖖²;𝖖²)_∞² E_𝖖(x) E_𝖖(x⁻¹)` — the SQED_1 Schur index
    (one hyper ⇒ two `E_𝖖` factors) — as `{𝖖-exp: int}` truncated to `𝖖^K`."""
    e = _eq_coeffs(K)
    acc = LaurentPoly.zero()
    for i in range(K + 1):
        j = i - n
        if 0 <= j <= K:
            acc = acc + e[i] * e[j]
    r = acc * LaurentPoly(_qpoch_q2_pow(2, K))
    return {ex: c for ex, c in r._coeffs.items() if ex <= K and c}


class SQED1SampleKAlgebra(KAlgebra):
    """SQED₁ = U1Square: a U(1) **gauge** theory with one charged hyper —
    UNFLAVOURED (no leftover continuous flavour), so the lift coordinate is
    trivial.

    Canonical basis `(m, n)`: `m` the magnetic / charged-hyper charge, `n` the
    gauge-monopole (`v`) power.  `multiply` brute-forces the monomial
    straightening from `u_± v = 𝖖^{±2} v u_±`, `u_+u_- = 1+𝖖v`,
    `u_-u_+ = 1+𝖖⁻¹v` (`_sqed1_ureduce`).  `ρ(m,n) = (−m, −n − max(m,0))`.  The
    trace localises to the Coulomb sector: `Tr L_{m,n} = 0` for `m ≠ 0`, and
    `Tr v^n = [x^n] (𝖖²;𝖖²)_∞² E_𝖖(x) E_𝖖(x⁻¹)`.

    Isomorphic to `U1SquareKAlg` / `Sqed1KAlg` (the ConeKAlgebra presentations);
    here a direct brute-force `KAlgebra` on the new optional surface."""

    _R = TrivialZPlusRing()

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return (0, 0)

    def multiply(self, a, b) -> Element:
        m1, n1 = a
        m2, n2 = b
        base = LaurentPoly.q(-m1 * n1 - m2 * n2 - 2 * n1 * m2)
        out: dict = {}
        for (m, vext), lp_u in _sqed1_ureduce(m1, m2).items():
            N = n1 + n2 + vext
            coeff = base * lp_u * LaurentPoly.q(m * N)
            out[(m, N)] = coeff if (m, N) not in out else out[(m, N)] + coeff
        return Element({k: v for k, v in out.items() if not v.is_zero()})

    def rho(self, a):
        m, n = a
        return (-m, -n - max(m, 0))

    def rho_inverse(self, a):
        m, n = a
        mo = -m
        return (mo, -n - max(mo, 0))

    def trace(self, a, K: int = 20) -> RPowerSeries:
        m, n = a
        if m != 0:
            return RPowerSeries.zero(self._R, K)
        return RPowerSeries(self._R, _sqed1_trvn(n, K), K)

    # ---- lift coordinate (trivial: unflavoured) ----

    def r_label_decompose(self, label):
        return label, ()

    def r_label_compose(self, section, r_basis_label):
        return section

    def __repr__(self) -> str:
        return "SQED1SampleKAlgebra()"


# ---------------------------------------------------------------------------
# SQED_{N_f} = U(1) gauge + N_f charged hypers, SU(N_f) flavour.
# ---------------------------------------------------------------------------
#
# The genuine SU(N_f)-flavoured generalisation of SQED_1, in the SAME u_±/v
# presentation (NOT the U_𝖖(𝔰𝔩₂) PBW form of SQED_2): u_± the dressed monopoles,
# v the gauge monopole, with
#     u_+ v = 𝖖² v u_+,   u_- v = 𝖖⁻² v u_-,
#     u_+ u_- = Σ_{k=0}^{N_f} 𝖖^k  χ_{(1^k)} v^k ,
#     u_- u_+ = Σ_{k=0}^{N_f} 𝖖^{-k} χ_{(1^k)} v^k ,
# where `χ_{(1^k)}` is the SU(N_f) character of `Λ^k(fundamental)`.
# N_f=1 ⇒ χ trivial ⇒ `u_+u_- = 1 + 𝖖v` = SQED_1.  Direct `KAlgebra` subclass:
# helpers below are plain module functions; flavour lives in the label.


def _sqednf_antisym(R, k):
    """SU(N_f) label of `Λ^k(fundamental)` = partition `(1^k)`; the trivial irrep
    `()` for `k = 0` or `k = N_f` (the determinant ≡ 1 in SU)."""
    if k == 0 or k == R.N:
        return R.one_basis()
    return (1,) * k


def _sqednf_ureduce(R, c1, c2) -> dict:
    """Reduce `u_{sgn c1}^{|c1|} · u_{sgn c2}^{|c2|}` to
    `{(net_charge, v_extra): {irrep: LaurentPoly}}` (net charge `= c1 + c2`).
    Generalises `_sqed1_ureduce`: each `u_+u_-` annihilation emits the full sum
    `Σ_k 𝖖^k χ_{(1^k)} v^k`, the emitted `v^k` commuting past the surviving
    block of `|·|−1` opposite letters at cost `𝖖^{k(2bb−1)}` (sign-flipped for
    `u_-u_+`)."""
    if c1 == 0 or c2 == 0 or (c1 > 0) == (c2 > 0):
        return {(c1 + c2, 0): {R.one_basis(): LaurentPoly.one()}}
    net = c1 + c2
    out: dict = {}
    if c1 > 0:                                        # u_+^{c1} u_-^{|c2|}
        stack = [(c1, -c2, 0, {R.one_basis(): LaurentPoly.one()})]
        sign = +1
    else:                                             # u_-^{|c1|} u_+^{c2}
        stack = [(-c1, c2, 0, {R.one_basis(): LaurentPoly.one()})]
        sign = -1
    while stack:
        aa, bb, ve, ch = stack.pop()
        if aa == 0 or bb == 0:
            slot = out.setdefault((net, ve), {})
            for irr, co in ch.items():
                slot[irr] = co if irr not in slot else slot[irr] + co
        else:
            for k in range(R.N + 1):                  # emit v^k dressed by χ_{(1^k)}
                qshift = LaurentPoly.q(sign * k * (2 * bb - 1))
                newch = _su2_fuse(R, ch, {_sqednf_antisym(R, k): 1})
                newch = {i: c * qshift for i, c in newch.items()}
                stack.append((aa - 1, bb - 1, ve + k, newch))
    return {key: {i: c for i, c in d.items() if not c.is_zero()}
            for key, d in out.items()}


def _norm_su_weight(lam, Nf) -> tuple:
    """`sun.decompose`'s U(N_f) dominant weight → SU(N_f) partition label
    (subtract the last part, strip trailing zeros)."""
    if not lam:
        return ()
    base = lam[-1]
    part = tuple(x - base for x in lam[:-1])
    while part and part[-1] == 0:
        part = part[:-1]
    return part


def _sqednf_trvn(Nf: int, n: int, K: int) -> RPowerSeries:
    """`Tr v^n = [x^n] (𝖖²;𝖖²)_∞² ∏_{i=1}^{N_f} E_𝖖(μ_i⁻¹ x) E_𝖖(μ_i x⁻¹)` as an
    `R(SU(N_f))`-character `RPowerSeries` to `𝖖^K` (`μ_i` = the N_f weights of
    the fundamental).  The Coulomb/`v` direction couples to the **conjugate**
    flavour: the trace pairing dualizes, so the `x^{+1}` (= `v`) factor carries
    the anti-fundamental `μ_i⁻¹`.  N_f=1 ⇒ the SQED_1 index `E_𝖖(x)E_𝖖(x⁻¹)`;
    N_f=2 ⇒ the SQED_2 Schur index (the SU(2) fundamental is self-conjugate, so
    the sign is invisible there — which is exactly why a `μ_i x` mis-sign passes
    N_f≤2 and only bites at N_f≥3).

    Each factor `E_𝖖(μ_i⁻¹ x)` contributes `(a_i)`, `E_𝖖(μ_i x⁻¹)` contributes
    `(b_i)`; the net `x`-power `Σ(a_i−b_i) = n` selects the coefficient, the
    `μ`-weight is `(b_1−a_1, …, b_{N_f}−a_{N_f})` (the conjugate sign), and the
    symmetric weight content is un-branched into SU(N_f) characters by
    `sun.decompose`.  Validated against the RG-flow realisation term-by-term and
    via orthonormality."""
    R = SUNZPlusRing(Nf)
    e = _eq_coeffs(K)
    acc: dict = {}                                    # {weight Nf-tuple: LaurentPoly}

    def rec(i, weight, xnet, qco, pmin):
        if i == Nf:
            if xnet == n:
                w = tuple(weight)
                acc[w] = acc.get(w, LaurentPoly.zero()) + qco
            return
        for a in range(K - pmin + 1):
            for b in range(K - pmin - a + 1):
                prod = LaurentPoly(
                    {ex: c for ex, c in (qco * e[a] * e[b])._coeffs.items()
                     if ex <= K})
                if prod.is_zero():
                    continue
                rec(i + 1, weight + [b - a], xnet + a - b, prod, pmin + a + b)

    rec(0, [], 0, LaurentPoly.one(), 0)
    pref = LaurentPoly(_qpoch_q2_pow(2, K))           # (𝖖²;𝖖²)_∞²
    byq: dict = {}                                    # 𝖖-exp → {weight: int}
    for w, lp in acc.items():
        prod = LaurentPoly({ex: c for ex, c in (lp * pref)._coeffs.items()
                            if ex <= K})
        for ex, c in prod._coeffs.items():
            if c:
                d = byq.setdefault(ex, {})
                d[w] = d.get(w, 0) + c
    coeffs: dict = {}
    for ex, wd in byq.items():
        if Nf >= 2:
            chi: dict = {}
            for lam, mult in _sun.decompose(Nf, wd).items():
                if mult:
                    lab = R.reduce(_norm_su_weight(lam, Nf))
                    chi[lab] = chi.get(lab, 0) + mult
        else:                                         # SU(1): all content trivial
            tot = sum(wd.values())
            chi = {(): tot} if tot else {}
        chi = {k: v for k, v in chi.items() if v}
        if chi:
            coeffs[ex] = RElement(R, chi)
    return RPowerSeries(R, coeffs, K)


class SQEDNfSampleKAlgebra(KAlgebra):
    """SQED_{N_f}: a U(1) **gauge** theory with `N_f` charged hypers and an
    **SU(N_f) flavour** symmetry — the genuine flavoured generalisation of
    `SQED1SampleKAlgebra`, in the same `u_±/v` presentation.  A **direct**
    `KAlgebra` subclass (no `RGKAlgebra` / spine machinery): the RG-flow
    realisation was only the cross-validation oracle.

    Canonical basis `(m, n, w)`: `(m, n)` the SQED_1 gauge label, `w` an SU(N_f)
    irrep (a `SUNZPlusRing(N_f)` partition; `()` = trivial).  `multiply` does the
    SQED_1 monomial straightening with the flavoured monopole relation
    `u_+u_- = Σ_k 𝖖^k χ_{(1^k)} v^k` (`_sqednf_ureduce`) and fuses the SU(N_f)
    characters by Littlewood–Richardson — Z-valued, flavour in the label.

    `ρ((m,n,w)) = (−m, −n−N_f·max(m,0), w*)` (the SQED_1 gauge automorphism with
    the v-shift scaled by `N_f`, tensored with the rep-ring duality `⋆`).  The
    trace localises to the Coulomb sector: `Tr (χ_w·v^n) = χ_w · Tr v^n`, zero
    for `m ≠ 0` (`_sqednf_trvn`).  Lift coordinate:
    `r_label_decompose((m,n,w)) = ((m,n,()), w)`.

    `SQEDNfSampleKAlgebra(1)` ≡ `SQED1SampleKAlgebra`; `(2)` is the `u_±/v`
    presentation of SQED_2 = U_𝖖(𝔰𝔩₂)."""

    def __init__(self, Nf: int) -> None:
        if Nf < 1:
            raise ValueError(f"SQEDNfSampleKAlgebra: N_f >= 1 required, got {Nf}")
        self._Nf = int(Nf)
        self._R = SUNZPlusRing(self._Nf)

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return (0, 0, self._R.one_basis())

    def multiply(self, a, b) -> Element:
        R = self._R
        m1, n1, l1 = a
        m2, n2, l2 = b
        base = LaurentPoly.q(-m1 * n1 - m2 * n2 - 2 * n1 * m2)
        lam12 = _su2_fuse(R, {l1: 1}, {l2: 1})        # generic ZPlusRing fusion
        out: dict = {}
        for (m, vext), chdict in _sqednf_ureduce(R, m1, m2).items():
            N = n1 + n2 + vext
            gauge = base * LaurentPoly.q(m * N)
            for irr_c, lp in chdict.items():
                for lout, mult in _su2_fuse(R, lam12, {irr_c: 1}).items():
                    key = (m, N, lout)
                    term = gauge * lp * mult
                    out[key] = term if key not in out else out[key] + term
        return Element({k: v for k, v in out.items() if not v.is_zero()})

    def rho(self, a):
        m, n, w = a
        return (-m, -n - self._Nf * max(m, 0), self._R.star_basis(w))

    def rho_inverse(self, a):
        m, n, w = a
        mo = -m
        return (mo, -n - self._Nf * max(mo, 0), self._R.star_basis(w))

    def trace(self, a, K: int = 20) -> RPowerSeries:
        m, n, w = a
        if m != 0:                                    # net magnetic charge ⇒ zero
            return RPowerSeries.zero(self._R, K)
        return _sqednf_trvn(self._Nf, n, K) * self._R.basis_element(w)

    # ---- lift coordinate (section = undressed gauge monomial, R-label = irrep) ----

    def r_label_decompose(self, label):
        m, n, w = label
        return (m, n, self._R.one_basis()), w

    def r_label_compose(self, section, r_basis_label):
        m, n, _triv = section
        return (m, n, r_basis_label)

    def __repr__(self) -> str:
        return f"SQEDNfSampleKAlgebra({self._Nf})"
