"""Hard-coded sample K-algebras for testing the `KAlgebra` ABC.

Each sample is over `R = TrivialZPlusRing()` (= the unflavoured case);
`coefficient_ring()` returns this in all four samples, and the ρ-twist
collapses to the strict-automorphism axiom (since `⋆ = id` on Z).

Samples:

* `TrivialKAlg`: rank-0 ABC sanity (Tr(1) = 1; ρ = id).
* `QuantumTorusZ2KAlg`: closed-form everything; basis Z²; Tr(X_γ) =
  (q²;q²)_∞² δ_{γ,0}.
* `Sqed1KAlg`: U(1) Nf=1, generators u_±, v^±, four relations from the
  user's documentation. Tr(L_{m, n}) closed-form via Nahm sums for m=0,
  vanishes for m≠0.
* `PentagonKAlg`: self-contained pentagon (A_2 Argyres-Douglas) algebra.
  Multiplication via an inline three-letter reducer.  Trace in two
  layers: (1) `Tr(L_{i;a,b}) = q^{ab} Tr(L^{a+b})` plus the Schur-like
  recursion `Tr(L^n) = q^{1-2n} Tr(L^{n-1}) + q^{2-2n} Tr(L^{n-2})`
  (memoized) reduces to `Tr(1)` and `Tr(L)`; (2) Yang-Lee plug-in
  `Tr(1) = χ₀(q²)`, `Tr(L) = q⁻¹(χ₀(q²) - χ₁(q²))` with the
  Rogers-Ramanujan Nahm sums for χ₀, χ₁ computed inline.

Each sample has its own closed-form `trace`.  The shared scaffolding is
`_expand_habiro_with_prefactor`, which expands a HabiroElement against
`(q²;q²)_∞^r` and converts the result to an `RPowerSeries[Trivial]`
at the boundary.  Internally HabiroElement and qpoch_infty (from the
repo) are still used as q-series tools — these are computational
machinery, not algebra-contract concerns; eliminating those imports is
a separate cleanup.

Dependencies: `zplus_ring` (local) + `laurent_poly.LaurentPoly`.
`Sqed1KAlg` still uses `habiro.HabiroElement` and `qpoch.qpoch_infty`
for its Nahm-sum trace expansion; `PentagonKAlg` no longer depends on
the legacy `pentagon_algebra` module — its multiplication reducer and
trace (Schur recursion + Rogers-Ramanujan Nahm sums) are inlined.
"""

from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laurent_poly import LaurentPoly
from habiro import HabiroElement
from qpoch import PowerSeries, qpoch_infty

from zplus_ring import (
    ZPlusRing, RElement, RLaurent, RPowerSeries, TrivialZPlusRing,
)
from kalgebra import KAlgebra, Element
from cone_kalgebra import ConeKAlgebra
from quantum_torus_kalgebra import QuantumTorusKAlg


# ---------------------------------------------------------------------------
# Boundary conversions (LaurentPoly ↔ RLaurent[Trivial], PowerSeries ↔
# RPowerSeries[Trivial]).  These let the samples keep their internal
# computations on the legacy q-series types, converting only at the
# KAlgebra-contract boundary.
# ---------------------------------------------------------------------------


def _lp_to_rlaurent(lp: LaurentPoly, ring: ZPlusRing) -> RLaurent:
    """Convert a `LaurentPoly` (Z[q^±]) to `RLaurent[ring]`.  Only
    sensible when `ring == TrivialZPlusRing()`."""
    return RLaurent(ring, dict(lp._coeffs))


def _ps_to_rpowerseries(
    ps: PowerSeries, ring: ZPlusRing, K: int,
) -> RPowerSeries:
    """Convert a `PowerSeries` (truncated Z((q))) to `RPowerSeries[ring]`."""
    return RPowerSeries(ring, dict(ps._c), K)


# ---------------------------------------------------------------------------
# Habiro-element trace expansion: (q²;q²)_∞^r · h(q) truncated to q^K,
# returned as RPowerSeries[Trivial].
# ---------------------------------------------------------------------------


def _expand_habiro_with_prefactor(
    h: HabiroElement, prefactor_rank: int, K: int, ring: ZPlusRing,
) -> RPowerSeries:
    """`(q²;q²)_∞^r · h(q)` truncated to `q^K`, as `RPowerSeries[ring]`.

    Negative q-powers in the inner expansion are kept (they cancel against
    the prefactor's positive shifts).  Internally uses `PowerSeries` and
    `qpoch_infty` for the actual q-series arithmetic; converts at the end.
    """
    lp = h.expand(K)
    ps = PowerSeries({e: c for e, c in lp._coeffs.items() if e <= K}, K)
    if prefactor_rank == 0:
        return _ps_to_rpowerseries(ps, ring, K)
    pref = qpoch_infty(K)
    for _ in range(prefactor_rank - 1):
        pref = pref * qpoch_infty(K)
    return _ps_to_rpowerseries(pref * ps, ring, K)


# ---------------------------------------------------------------------------
# (1) TrivialKAlg -- A_q = Z[q, q^{-1}], basis {()}.
# ---------------------------------------------------------------------------


class TrivialKAlg(KAlgebra):
    """The trivial K-algebra `A_q = Z[q, q^{-1}]`."""

    _R = TrivialZPlusRing()

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return ()

    def multiply(self, a, b):
        return Element({(): LaurentPoly({0: 1})})

    def rho(self, a):
        return ()

    def rho_inverse(self, a):
        return ()

    def rho_squared_is_identity(self) -> bool:
        # ρ is the identity on the trivial algebra's single label.
        return True

    def trace(self, a, K=20):
        # Tr(1) = 1.
        return RPowerSeries(self._R, {0: 1}, K)

    # Unflavoured: r_label_decompose / r_label_compose / _label_section_decompose
    # inherited from the KAlgebra universal trivial-flavour treatment.


# ---------------------------------------------------------------------------
# (2) QuantumTorusZ2KAlg -- the quantum torus on Z² with pairing
#     ⟨(a,b),(c,d)⟩ = ad - bc.  Special case of `QuantumTorusKAlg`,
#     kept as a no-arg-constructor alias for test convenience.
# ---------------------------------------------------------------------------


class QuantumTorusZ2KAlg(QuantumTorusKAlg):
    """`QuantumTorusKAlg([[0, 1], [-1, 0]])` -- the rank-2 quantum torus
    with the symplectic pairing.  Subclass solely to allow the no-arg
    `QuantumTorusZ2KAlg()` form used by tests."""

    def __init__(self):
        super().__init__([[0, 1], [-1, 0]])

    def rho_squared_is_identity(self) -> bool:
        # ρ((m, n)) = (-m, -n) (the Z₂ negation) on the rank-2 QT with
        # symplectic pairing; ρ² = id everywhere.
        return True


# ---------------------------------------------------------------------------
# (3) Sqed1KAlg -- U(1) with one fundamental hyper.
#
# Generators u_± (= L_{±1, 0}), v, v^{-1} (= L_{0, ±1}). Relations:
#
#   u_+ v   = q²    v u_+
#   u_- v   = q^{-2} v u_-
#   u_+ u_- = 1 + q v
#   u_- u_+ = 1 + q^{-1} v
#
# Linear basis convention:
#
#   L_{a, b}  = q^{-ab} u_+^a v^b   for a > 0
#   L_{-a, b} = q^{ab}  u_-^a v^b   for a > 0
#   L_{0, n}  = v^n
#
# ρ:  ρ(v) = v^{-1},  ρ(u_+) = q v^{-1} u_-,  ρ(u_-) = u_+
# Equivalently  ρ(m, n) = (-m, -n - max(m, 0)).
# ---------------------------------------------------------------------------


class Sqed1KAlg(KAlgebra):
    """The K-algebra of SQED_1 (U(1) gauge with one charged hyper).

    Realised on the canonical **U(1) matter URQTorus** substrate: the
    monopole generators are the torus atoms `u_± = U_{±1}`, multiply is
    the rational 2-cocycle on the atoms, and the Schur trace is the
    magnetic-0 matter-measure residue.  SQED1 is the flavour-specialized
    (μ→1) face of the matter torus, so the coefficient ring is trivial.
    (Verified to reproduce the prior monomial-rewriting structure
    constants exactly: multiply on the full label grid, trace through
    q^20.)"""

    _R = TrivialZPlusRing()

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return (0, 0)

    def rho(self, a):
        m, n = a
        return (-m, -n - max(m, 0))

    def rho_inverse(self, a):
        m, n = a
        m_orig = -m
        n_orig = -n - max(m_orig, 0)
        return (m_orig, n_orig)

    # ----- URQTorus substrate: u_± = U_{±1} -----
    #
    # SQED1 lives on the U(1) matter quantum torus.  The canonical basis
    # element `L_{m,n}` is the dressed minuscule `minuscule((m,), (n,))`
    # (so the monopoles are the bare atoms `u_± = U_{±1}`); multiply is
    # the rational 2-cocycle on the atoms.  The matter torus tracks the
    # U(1)_F charge as its flavour *level*; SQED1 is the μ→1 face, so
    # `_decode` sums the levels and reads each magnetic charge's
    # polynomial residual in the Coulomb variable `v_0`.

    @staticmethod
    def _matter():
        from matter_urq_torus import MatterURQTorus
        return MatterURQTorus

    def _encode(self, m: int, n: int):
        """Canonical label `(m, n)` → the dressed minuscule on the U(1)
        matter torus (`u_± = U_{±1}`, `v = v_0`)."""
        return self._matter().minuscule((m,), (n,), 1, 1)

    def _decode(self, x) -> Element:
        """Matter-torus element → SQED1 `Element` over `(m, n)`: μ→1
        (sum the flavour levels), then read each magnetic charge's
        polynomial residual in `v_0`."""
        terms: dict = {}
        for mag, row in x.residuals().items():
            tot = None
            for _lev, vr in row.items():
                tot = vr if tot is None else (tot + vr)
            vrs = tot.simplify()
            if vrs.den:
                raise RuntimeError(
                    f"Sqed1KAlg._decode: non-polynomial residual at "
                    f"magnetic {mag}: denominator {vrs.den}")
            for ve, lp in dict(vrs.num._terms).items():
                if lp.is_zero():
                    continue
                key = (mag[0], ve[0])
                terms[key] = terms.get(key, LaurentPoly.zero()) + lp
        return Element({k: v for k, v in terms.items() if not v.is_zero()})

    def multiply(self, a, b):
        """`U_a · U_b` on the matter torus (rational 2-cocycle), decoded
        to the `(m, n)` basis."""
        return self._decode(self._encode(*a) * self._encode(*b))

    # ----- Schur trace via the matter-measure residue -----

    def trace(self, a, K=20):
        """The SQED1 Schur index: the magnetic-0 matter-measure residue
        on the U(1) torus (μ→1; vanishes for charged `m ≠ 0` labels).
        The residue window is scaled with `K` so the result is exact to
        `O(q^K)`."""
        levels = self._encode(*a).trace(K=K, W=K, Kq_margin=K + 10)
        acc = LaurentPoly.zero()
        for _lev, lp in levels.items():
            acc = acc + lp
        return RPowerSeries(
            self._R, {e: c for e, c in acc._coeffs.items() if e <= K}, K)

    # Unflavoured: r_label_decompose / r_label_compose / _label_section_decompose
    # inherited from the KAlgebra universal trivial-flavour treatment.


# ---------------------------------------------------------------------------
# (4) PentagonKAlg -- the pentagon algebra of the (A_1, A_2) AD theory.
#
# Generators L_i, i in Z/5, with relations
#     L_{i+1} L_i      = q²  L_i L_{i+1}                       (swap)
#     L_{i+1} L_{i-1}  = 1 + q   L_i                          (Plücker)
#     L_{i-1} L_{i+1}  = 1 + q⁻¹ L_i                          (Plücker).
# Basis L_{i;a,b} = q^{ab} L_i^a L_{i+1}^b, i in Z/5, a, b >= 0.
# ρ(L_i) = L_{i+2} generates the Z/5 automorphism group.
# ---------------------------------------------------------------------------


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


# ---- Trace Layer 1: Schur-like recursion on Tr(L^n). ---------------------
#
# By ρ²-twisted cyclicity (Tr(ab) = Tr(ρ²(b) a)) one shows
#     Tr(L_{i;a,b}) = q^{ab} Tr(L_i^{a+b}),
# and (e.g. Tr(L_i^a) = Tr(L_{i-1} L_i^{a-1} L_i) chased through swaps
# and a single Plücker hit)
#     Tr(L_i^n) = q^{1-2n} Tr(L_i^{n-1}) + q^{2-2n} Tr(L_i^{n-2}),
# with Tr(L_i^0) = Tr(1), Tr(L_i^1) = Tr(L).  Memoize the (c1, cL) pair.

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


# ---- Trace Layer 2: Yang-Lee characters via Rogers-Ramanujan. -----------
#
#     χ₀(q) = Σ_{n>=0} q^{n(n+1)} / (q;q)_n,
#     χ₁(q) = Σ_{n>=0} q^{n²}     / (q;q)_n.
# At q -> q²:
#     Tr(1)   = χ₀(q²),
#     Tr(L_i) = q⁻¹ (χ₀(q²) - χ₁(q²)).
# Compute 1 / (q²;q²)_n incrementally and accumulate the Nahm sum.


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


class PentagonKAlg(ConeKAlgebra):
    """The pentagon K-algebra `A_𝖖([A_1, A_2])`, defined intrinsically.

    *Definition.*  Generators `L_i` for `i ∈ Z/5`, the chord generators
    of the pentagon (the `(2k+3)`-gon for `k = 1`): `L_i` is the diagonal
    `{3i, 3i + 2}` (`curve`, `geometric_label`).  Relations are the
    Z/5 quantum Ptolemy relations (swap and two Plückers; equivalently,
    the `k = 1` case of the chord-pair rules).
    The canonical basis is `L_{i;a,b} = 𝖖^{ab} L_i^a L_{i+1}^b` with
    `(i, a, b) ∈ Z/5 × Z_{≥0} × Z_{≥0}`, canonicalised so the unit is
    uniquely `(0, 0, 0)` and pure powers `L_i^a` are `(i, a, 0)`.

    *Property.*  Satisfies the KAlgebra axioms (bar / ρ-twist /
    ρ²-cyclicity / canonical orthonormality); in code, exposed by
    subclassing `ConeKAlgebra` (the closed-form presentation tier).
    `ρ(L_i) = L_{i+2}` generates Z/5.

    *Property (separate theorem).*  There is a KAlgebra isomorphism
    `A_𝖖([A_1, A_2]) ≅ A_𝖖^BPS(A_2-quiver)`; the chord generators
    correspond to specific F-elements of the BPS realisation.  See
    `kalgebra_iso.KAlgebraIso`.  This iso is *not* used at runtime
    here; the class is a self-contained, intrinsic implementation.

    Trace is two-layered.  Layer 1 (the tagged-cycle ρ²-cyclicity
    reducer in `cone_data.simplify_trace_via_cone_data`) is inherited
    universally from `ConeKAlgebra.trace`: each basis label is reduced
    to a `Z[𝖖^±]`-linear combination of trace seeds — the identity
    `(0, 0, 0)` (→ `Tr(1)`) and the single-mult-gen labels `(i, 1, 0)`
    for `i ∈ ℤ/5` (→ `Tr(L)`, identical across the ρ-orbit).  Layer 2
    is supplied here via `_trace_residual`: the Yang-Lee CFT characters
    `Tr(1) = χ₀(𝖖²)` and `Tr(L) = 𝖖⁻¹ (χ₀(𝖖²) - χ₁(𝖖²))` with
    `χ_{0,1}` the two Rogers-Ramanujan functions, computed inline as
    Nahm sums."""

    _R = TrivialZPlusRing()

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return (0, 0, 0)

    def cone_data(self):
        from pentagon_cone_data import PENTAGON_CONE_DATA
        return PENTAGON_CONE_DATA

    def rho(self, a):
        i, x, y = a
        return _pent_canon_key(i + 2, x, y)

    def rho_inverse(self, a):
        i, x, y = a
        return _pent_canon_key(i - 2, x, y)

    # ----- geometry: the diagonals of the pentagon ----------------------
    #
    # `L_i` is the diagonal `{3i, 3i + 2}` of the pentagon with marked points
    # `0, …, 4`.  ρ-equivariance alone fixes this up to the basepoint:
    # `ρ(L_i) = L_{i+2}` is the rotation `x ↦ x + 1` exactly when `L_i` starts
    # at `3i + t`, and `t = 0` (`L_0 = {0, 2}`) is a convention — the
    # rotations are automorphisms, so every `t` presents the algebra
    # equally.  Measured against `A1A2kKAlg(1)`, whose letter `(1, j)` is
    # `{j, j + 2}`, under `L_i ↦ (1, 3i)`: all 961 products of the 31
    # canonical labels with `a, b ≤ 2` agree, and so does ρ on each; the
    # reflected orientation `L_i ↦ (1, −3i)` agrees on 111 of the 961 and is
    # not ρ-equivariant (the suite in the source repository).

    H = 5

    @staticmethod
    def _diagonal(i: int) -> tuple[int, int]:
        """The diagonal of `L_i`, a sorted pair of marked points."""
        x = (3 * i) % 5
        return tuple(sorted((x, (x + 2) % 5)))

    def curve(self, x: int, ell: int):
        """The label of the diagonal from marked point `x` to `x + ell`,
        `2 ≤ ell ≤ 3`, as `A1A2kKAlg(1).curve`: the pentagon has no puncture,
        so `curve(x, ell) == curve(x + ell, 5 − ell)`.  A boundary edge
        (`ell = 1` or `4`) is the identity and not a letter, and raises, as
        does any other `ell`."""
        x, ell = int(x), int(ell)
        if not 2 <= ell <= 3:
            raise ValueError(
                f"PentagonKAlg.curve({x}, {ell}): need 2 <= ell <= 3 (ell = 1 "
                f"or 4 is a boundary edge of the pentagon)")
        start = x if ell == 2 else x + 3        # {x, x + 3} = {x + 3, x + 5}
        return _pent_canon_key(2 * start, 1, 0)  # 3i ≡ start  ⇔  i ≡ 2·start

    def geometric_label(self, label):
        """The label `(i, a, b)` — the element `𝖖^{ab} L_i^a L_{i+1}^b` — as
        the multiset of its diagonals: a sorted tuple of pairs `((v1, v2), m)`,
        `(v1, v2)` a diagonal (sorted marked points) and `m ≥ 1` its power;
        the unit is `()`.  `L_i` and `L_{i+1}` share the marked point `3i`, so
        the two diagonals of a label never cross.  The layout of the `curves`
        of `A1DnKAlg`'s `(curves, κ)`, on this unflavoured polygon without a
        puncture."""
        i, a, b = _pent_canon_key(*label)
        out = []
        if a:
            out.append((self._diagonal(i), a))
        if b:
            out.append((self._diagonal(i + 1), b))
        return tuple(sorted(out))

    # ----- Layer-2 trace residual ---------------------------------------
    #
    # Canonical ρ²-orbit seeds produced by Layer 1 (tagged-cycle +
    # ρ²-orbit canonicalisation in `simplify_trace_via_cone_data`):
    #   * `(0, 0, 0)` -- the identity              → `Tr(1)` = χ₀(q²)
    #   * `(0, 1, 0)` -- canonical mult-gen seed   → `Tr(L)`
    #     (min of the single ρ²-orbit                 = q⁻¹(χ₀(q²) - χ₁(q²))
    #     {(0,1,0), (1,1,0), (2,1,0), (3,1,0), (4,1,0)})
    #
    # `multiply` and `trace` are inherited from `ConeKAlgebra`.  ρ²-
    # invariance of the trace on the 5 single-mult-gen seeds is enforced
    # by Layer 1 (not by this method), so this method only sees the
    # canonical `(0, 1, 0)`.

    def _trace_residual(self, seed_label, K):
        if seed_label == (0, 0, 0):
            return _pent_tr_1_rps(self._R, K)
        if seed_label == (0, 1, 0):
            return _pent_tr_L_rps(self._R, K)
        # Defensive: any other label means Layer 1 broke its contract.
        raise ValueError(
            f"PentagonKAlg._trace_residual: unexpected seed {seed_label!r}; "
            f"expected canonical ρ²-orbit representative (0, 0, 0) or "
            f"(0, 1, 0)"
        )


# ===========================================================================
# Heptagon  K_𝖖([A_1, A_4])
# ===========================================================================
#
# Pentagon-style direct realisation as a `KAlgebra` subclass.  Fourteen
# named generators `L((k, i))` with `k ∈ {1, 2}` (orbit index) and
# `i ∈ Z/7` (within-orbit ρ-index), the diagonals of the heptagon with
# marked points `0, …, 6`:
#
#   * `L((1, i))` -- the short diagonal `{i, i+2}`;
#   * `L((2, i))` -- the long diagonal `{i, i+4}`.
#
# These are the letters of `A1A2kKAlg(2)`, whose `(a, j)` is the diagonal
# `{j, j+a+1}`, under the relabelling of the orbit-2 index
#
#     (1, i) ↦ (1, i),        (2, i) ↦ (2, i + 4),
#
# which carries all 196 generator products of this class onto
# `A1A2kKAlg(2)`'s (under the identity map on labels, 140 of them differ).
# The products are served through it: `HeptagonKAlg.multiply` relabels,
# multiplies in `A1A2kKAlg(2)` and relabels back, and the per-pair
# expansions `_hept_pair_product` read by `HEPTAGON_CONE_DATA` (the cone
# data behind the Layer-1 trace) and by the reducers `_hept_reduce` /
# `_hept_trace_layer1` are `A1A2kKAlg(2)`'s generator products,
# relabelled.  No product table is stored in this module.
#
# On the 182 ordered pairs of distinct letters, crossing diagonals give a
# two-term (Plücker) product and non-crossing ones fq-commute.  Whether a
# non-crossing pair commutes exactly is decided by the arc-parity rule of
# the chord algebra, not by sharing a vertex: 42 of the 84 vertex-sharing
# ordered pairs commute exactly, and 14 of the 28 disjoint ones do not
# (measured 2026-09-23), so the finer reading "share ↔ fq-commute,
# disjoint ↔ commute" does not hold.
#
# Canonical basis labels are sorted tuples  `((k_1, i_1, e_1), ..., (k_m, i_m, e_m))`
# of (letter, positive exponent) entries, where the letters
# `(k_1, i_1), ..., (k_m, i_m)`  are pairwise fq-commuting (so the
# underlying element  `∏ L((k_r, i_r))^{e_r}`  is well-defined up to
# the fq-twist read off from the fq-commutation factors).  The empty
# tuple `()` is the identity.
#
# Scope:  `multiply`, `rho`, `rho_inverse`, and the two-layer trace —
# Layer 1 reduces  Tr label  to  c0·T_0 + c1·T_1 + c2·T_2  with
# Z[fq, fq^{-1}] coefficients (= the M(2, 7)-elementary trace space),
# and the analytic Layer 2 plug-in (T_0, T_1, T_2 → the three M(2, 7)
# Andrews-Gordon characters via `A1A2kKAlg(2)._compute_T_series`) —
# see the "Layer-2 trace residual" section in the class body below.
#
# The cross-check against `heptagon_kalg.HeptagonKAlg` (BPSKAlgebra
# wrapper) lives in the suite of the source repository.

_HEPT_H = 7
_HEPT_ORBIT2_SHIFT = 4        # (2, i) here is A1A2kKAlg(2)'s (2, i + 4)

_HEPT_A1A2K = None


def _hept_a1a2k():
    """The `A1A2kKAlg(2)` instance that serves `HeptagonKAlg` (built on
    first use, then shared)."""
    global _HEPT_A1A2K
    if _HEPT_A1A2K is None:
        from a1a2k_kalg import A1A2kKAlg
        _HEPT_A1A2K = A1A2kKAlg(2)
    return _HEPT_A1A2K


def _hept_letter_to_a1a2k(letter: tuple[int, int]) -> tuple[int, int]:
    """`(1, i) ↦ (1, i)`, `(2, i) ↦ (2, i + 4)` (indices mod 7)."""
    k, i = letter
    return (k, (i + _HEPT_ORBIT2_SHIFT) % _HEPT_H if k == 2 else i % _HEPT_H)


def _hept_letter_from_a1a2k(letter: tuple[int, int]) -> tuple[int, int]:
    """Inverse of `_hept_letter_to_a1a2k`."""
    k, i = letter
    return (k, (i - _HEPT_ORBIT2_SHIFT) % _HEPT_H if k == 2 else i % _HEPT_H)


def _hept_label_to_a1a2k(label) -> tuple:
    """A `HeptagonKAlg` label as the `A1A2kKAlg(2)` label of the same
    element (letters relabelled, entries re-sorted)."""
    return tuple(sorted(_hept_letter_to_a1a2k((k, i)) + (e,)
                        for (k, i, e) in label))


def _hept_label_from_a1a2k(label) -> tuple:
    """Inverse of `_hept_label_to_a1a2k`."""
    return tuple(sorted(_hept_letter_from_a1a2k((k, i)) + (e,)
                        for (k, i, e) in label))


# Per-pair caches.  `_hept_pair_product`, `_hept_qcommute_factor`, and
# `_hept_forward_q_coeff` are pure functions of `(la, lb)` over the
# 14 × 14 = 196 ordered letter pairs, hot-pathed millions of times by the
# trace reducer; each is filled on first use.
_HEPT_PAIR_PRODUCT_CACHE: dict[
    tuple[tuple[int, int], tuple[int, int]], tuple
] = {}
_HEPT_QCOMMUTE_CACHE: dict[
    tuple[tuple[int, int], tuple[int, int]], int | None
] = {}
_HEPT_FWD_Q_CACHE: dict[
    tuple[tuple[int, int], tuple[int, int]], int
] = {}


def _hept_pair_product(la: tuple[int, int], lb: tuple[int, int]):
    """Return the canonical expansion of `L(la) · L(lb)` as a tuple of
    `(lifted_term, c)` entries.  Each `lifted_term` is one of
    `('I',)`, `('letter', (k, i))`, `('pair', ((k1, i1), (k2, i2)))`
    (a sorted pair; a repeated letter for a square), and `c` is the
    q-exponent of its coefficient.  Read off `A1A2kKAlg(2)`'s product of
    the relabelled letters."""
    cached = _HEPT_PAIR_PRODUCT_CACHE.get((la, lb))
    if cached is not None:
        return cached
    A = _hept_a1a2k()
    prod = A.multiply(A.L(_hept_letter_to_a1a2k(la)),
                      A.L(_hept_letter_to_a1a2k(lb)))
    result = []
    for label, coeff in prod.terms.items():
        if len(coeff._coeffs) != 1 or next(iter(coeff._coeffs.values())) != 1:
            raise ValueError(
                f"_hept_pair_product({la}, {lb}): coefficient {coeff} of "
                f"{label} is not a single power of q")
        (c, _one), = coeff._coeffs.items()
        letters = [(_hept_letter_from_a1a2k((k, i)), e) for (k, i, e) in label]
        if not letters:
            term = ('I',)
        elif len(letters) == 1 and letters[0][1] == 1:
            term = ('letter', letters[0][0])
        elif len(letters) == 1 and letters[0][1] == 2:
            term = ('pair', (letters[0][0], letters[0][0]))
        elif len(letters) == 2 and letters[0][1] == letters[1][1] == 1:
            term = ('pair', tuple(sorted(l for l, _e in letters)))
        else:
            raise ValueError(
                f"_hept_pair_product({la}, {lb}): unexpected term {label}")
        result.append((term, c))
    result = tuple(result)
    _HEPT_PAIR_PRODUCT_CACHE[(la, lb)] = result
    return result


def _hept_qcommute_factor(la: tuple[int, int],
                          lb: tuple[int, int]) -> int | None:
    """Return integer `c` such that `L(la) · L(lb) = fq^c · L(lb) · L(la)`
    (so `L_a L_b` and `L_b L_a` differ by `fq^c`), or `None` if (la, lb)
    is a Plücker pair (not fq-commuting)."""
    cached = _HEPT_QCOMMUTE_CACHE.get((la, lb), _HEPT_PAIR_PRODUCT_CACHE)  # sentinel
    if cached is not _HEPT_PAIR_PRODUCT_CACHE:
        return cached
    if la == lb:
        out: int | None = 0
    else:
        fwd = _hept_pair_product(la, lb)
        bwd = _hept_pair_product(lb, la)
        if len(fwd) != 1 or len(bwd) != 1:
            out = None
        else:
            (_, cf), = fwd
            (_, cb), = bwd
            out = cf - cb
    _HEPT_QCOMMUTE_CACHE[(la, lb)] = out
    return out


def _hept_forward_q_coeff(la: tuple[int, int], lb: tuple[int, int]) -> int:
    """For fq-commuting `(la, lb)`, return the integer `c` such that
    `L(la) · L(lb) = fq^c · X[γ_la + γ_lb]` in the BPS X-basis
    convention.  This is the `c`-value of `_hept_pair_product(la, lb)`'s
    single term.  Returns 0 when `la == lb` (L_a² = X[2γ_a] without
    twist)."""
    cached = _HEPT_FWD_Q_CACHE.get((la, lb))
    if cached is not None:
        return cached
    if la == lb:
        out = 0
    else:
        fwd = _hept_pair_product(la, lb)
        assert len(fwd) == 1, f"_hept_forward_q_coeff: {la}, {lb} are not fq-commuting"
        out = fwd[0][1]
    _HEPT_FWD_Q_CACHE[(la, lb)] = out
    return out


# ---- LaurentPoly fast-path helpers ----
#
# The bubble-cycle trace reducer multiplies LaurentPolys millions of
# times for large-`a` powers like Tr(L^a) and Tr(N^a).  Almost all of
# these multiplications have one operand as a single monomial fq^k, in
# which case the result is just a shift of the other's exponents.
# These helpers bypass `LaurentPoly.__init__` / `__mul__` for the
# monomial case.

def _hept_lp_mul(a: LaurentPoly, b: LaurentPoly) -> LaurentPoly:
    """Multiply two LaurentPolys.  Fast-paths when either operand is a
    single-monomial (uses pure exponent shift)."""
    ac = a._coeffs
    bc = b._coeffs
    if not ac or not bc:
        return LaurentPoly.zero()
    if len(ac) == 1:
        ((ea, ca),) = ac.items()
        if ca == 1 and ea == 0:
            return b
        new_lp = LaurentPoly.__new__(LaurentPoly)
        if ca == 1:
            new_lp._coeffs = {e + ea: c for e, c in bc.items()}
        else:
            new_lp._coeffs = {e + ea: c * ca for e, c in bc.items() if c}
        return new_lp
    if len(bc) == 1:
        ((eb, cb),) = bc.items()
        if cb == 1 and eb == 0:
            return a
        new_lp = LaurentPoly.__new__(LaurentPoly)
        if cb == 1:
            new_lp._coeffs = {e + eb: c for e, c in ac.items()}
        else:
            new_lp._coeffs = {e + eb: c * cb for e, c in ac.items() if c}
        return new_lp
    return a * b


def _hept_lp_add(a: LaurentPoly, b: LaurentPoly) -> LaurentPoly:
    """Fast LaurentPoly addition (bypasses `LaurentPoly.__init__`'s
    per-coefficient int-conversion)."""
    ac = a._coeffs
    bc = b._coeffs
    if not ac:
        return b
    if not bc:
        return a
    result: dict[int, int] = dict(ac)
    for e, c in bc.items():
        s = result.get(e, 0) + c
        if s:
            result[e] = s
        else:
            result.pop(e, None)
    new_lp = LaurentPoly.__new__(LaurentPoly)
    new_lp._coeffs = result
    return new_lp


def _hept_letters_qcommute(letters: list[tuple[int, int]]) -> bool:
    """True iff every pair of distinct letters in `letters` fq-commutes."""
    for i in range(len(letters)):
        for j in range(i + 1, len(letters)):
            if _hept_qcommute_factor(letters[i], letters[j]) is None:
                return False
    return True


def _hept_step(word: list[tuple[int, int, int]]):
    """One left-to-right scan over `word`, applying the FIRST adjacent
    pair needing action (merge / Plücker / descending q-commute swap),
    returning a list of `(coeff_factor, new_word)` summands.  Returns
    `None` iff every adjacent pair is in canonical order with distinct
    fq-commuting letters (i.e., `word` is in canonical basis form
    modulo the BPS X-twist)."""
    n = len(word)
    for idx in range(n - 1):
        ka, ia, ea = word[idx]
        kb, ib, eb = word[idx + 1]
        la = (ka, ia)
        lb = (kb, ib)
        if la == lb:
            # Merge adjacent same-letter into a single block.
            merged_entry = (ka, ia, ea + eb)
            new_word = word[:idx] + [merged_entry] + word[idx + 2:]
            return [(LaurentPoly.one(), new_word)]
        c_sw = _hept_qcommute_factor(la, lb)
        if c_sw is None:
            # Plücker.  Apply forward (la, lb) Plücker on one copy of each:
            #   L_la^{ea} · L_lb^{eb}  =  L_la^{ea-1} · (L_la · L_lb) · L_lb^{eb-1}.
            la_pre = [(ka, ia, ea - 1)] if ea > 1 else []
            lb_post = [(kb, ib, eb - 1)] if eb > 1 else []
            results = []
            for term, c in _hept_pair_product(la, lb):
                term_word = _hept_term_to_word(term)
                c_eff = c
                if term[0] == 'pair':
                    ((k1, i1), (k2, i2)) = term[1]
                    if (k1, i1) != (k2, i2):
                        c_eff -= _hept_forward_q_coeff((k1, i1), (k2, i2))
                new_word = word[:idx] + la_pre + term_word + lb_post + word[idx + 2:]
                new_factor = LaurentPoly.q(c_eff) if c_eff else LaurentPoly.one()
                results.append((new_factor, new_word))
            return results
        if la > lb:
            # Descending q-commute -- swap with twist `c_sw · ea · eb`.
            new_word = word[:idx] + [(kb, ib, eb), (ka, ia, ea)] + word[idx + 2:]
            factor = LaurentPoly.q(c_sw * ea * eb) if c_sw else LaurentPoly.one()
            return [(factor, new_word)]
        # la < lb in canonical order: continue to next adjacent pair.

    # All adjacent pairs are merged-sorted-fq-commuting.  Now check
    # whether any *non-adjacent* pair is a Plücker -- a sorted word like
    # `[(1, 0, 1), (1, 2, 1), (1, 6, 1)]` has all adjacent pairs OK but
    # the outer (1, 0)/(1, 6) pair is a Plücker (d=6 in orbit 1).
    #
    # Scan by increasing distance `j - i`: the closest Plücker pair has
    # the shortest bubble path, and the path won't cross any other
    # Plücker (otherwise *that* pair would have been picked first at
    # smaller distance).
    pairs_by_distance = sorted(
        ((j - i, i, j) for i in range(n - 1) for j in range(i + 1, n)),
        key=lambda t: t[0],
    )
    for _dist, i, j in pairs_by_distance:
        ka, ia, ea = word[i]
        la = (ka, ia)
        kb, ib, eb = word[j]
        lb = (kb, ib)
        if _hept_qcommute_factor(la, lb) is not None:
            continue
        # Found a non-adjacent Plücker pair (la at i, lb at j).
        # Bubble one copy of lb left past intermediate blocks (i+1,
        # ..., j-1) paying swap twist for each.  Distance scan order
        # guarantees no intermediate block is itself a Plücker partner
        # of lb (else *that* pair would have been picked at a shorter
        # distance).
        bubble_twist = 0
        for k in range(i + 1, j):
            km, im, em = word[k]
            lm = (km, im)
            c_sw_mid = _hept_qcommute_factor(lb, lm)
            assert c_sw_mid is not None, (
                f"_hept_step: bubble path has Plücker mid-block "
                f"{lm} between Plücker pair {la}/{lb} (word={word})"
            )
            # Moving one lb LEFT past L_lm^{em}: original is
            # L_lm^{em} · L_lb, rewrite as fq^{-c_sw(lb, lm) · em} · L_lb · L_lm^{em}.
            bubble_twist += -c_sw_mid * em
        la_pre = [(ka, ia, ea - 1)] if ea > 1 else []
        lb_post = [(kb, ib, eb - 1)] if eb > 1 else []
        intermediate = list(word[i + 1:j])
        results = []
        for term, c in _hept_pair_product(la, lb):
            term_word = _hept_term_to_word(term)
            c_eff = c
            if term[0] == 'pair':
                ((k1, i1), (k2, i2)) = term[1]
                if (k1, i1) != (k2, i2):
                    c_eff -= _hept_forward_q_coeff((k1, i1), (k2, i2))
            new_word = (word[:i] + la_pre + term_word + intermediate
                        + lb_post + word[j + 1:])
            total_pow = c_eff + bubble_twist
            factor = LaurentPoly.q(total_pow) if total_pow else LaurentPoly.one()
            results.append((factor, new_word))
        return results
    return None


def _hept_canonical_bps_twist(word: list[tuple[int, int, int]]) -> int:
    """Given a word already in canonical basis form (sorted by (k, i),
    distinct letters, all pairs fq-commuting), compute the BPS X-basis
    offset
        T_bps  =  Σ_{i<j}  c_fwd(l_i, l_j) · e_i · e_j
    so that  `L-product(word)  =  fq^{T_bps} · X[word]`."""
    T = 0
    n = len(word)
    for ii in range(n):
        la = (word[ii][0], word[ii][1])
        ea = word[ii][2]
        for jj in range(ii + 1, n):
            lb = (word[jj][0], word[jj][1])
            eb = word[jj][2]
            T += _hept_forward_q_coeff(la, lb) * ea * eb
    return T


def _hept_T_bps(label) -> int:
    """The BPS X-vs-L-product offset for a canonical basis label:
       T_bps  =  Σ_{i<j}  c_fwd(l_i, l_j) · e_i · e_j.
    The basis element  `X[label] = fq^{-T_bps} · L_1^{e_1} ... L_m^{e_m}`
    (with letters in canonical (k, i) sort order).  Single letters
    contribute 0; the identity contributes 0."""
    T = 0
    n = len(label)
    for ii in range(n):
        la = (label[ii][0], label[ii][1])
        ea = label[ii][2]
        for jj in range(ii + 1, n):
            lb = (label[jj][0], label[jj][1])
            eb = label[jj][2]
            T += _hept_forward_q_coeff(la, lb) * ea * eb
    return T


def _hept_term_to_word(term) -> list[tuple[int, int, int]]:
    """Convert a lifted-term (`('I',)`, `('letter', (k, i))`, or
    `('pair', ((k1, i1), (k2, i2)))`) to a list of `(k, i, e)` entries."""
    if term == ('I',):
        return []
    if term[0] == 'letter':
        (k, i) = term[1]
        return [(k, i, 1)]
    # pair
    ((k1, i1), (k2, i2)) = term[1]
    if (k1, i1) == (k2, i2):
        return [(k1, i1, 2)]
    return [(k1, i1, 1), (k2, i2, 1)]


_HEPT_REDUCE_CACHE: dict[tuple, dict] = {}


def _hept_reduce(word: list[tuple[int, int, int]]
                 ) -> dict[tuple[tuple[int, int, int], ...], LaurentPoly]:
    """Reduce a word to a canonical-basis sum.  Returns
    `{basis_label: LaurentPoly}`.  Uses iterative DFS with Plücker
    branching.  Cached by a normalised (sorted, e-merged) word key."""
    # Build a canonical cache key: group by (k, i), sum exponents, sort.
    bucket: dict[tuple[int, int], int] = {}
    for t in word:
        if t[2]:
            bucket[(t[0], t[1])] = bucket.get((t[0], t[1]), 0) + t[2]
    key_word_canonical = tuple(sorted((k, i, e) for (k, i), e in bucket.items() if e))
    # WORD ORDER MATTERS for reduce -- a non-canonical input order may
    # Plücker-expand differently than the sorted form because the
    # reducer's pair-selection picks the first non-q-commuting pair
    # iterating items() in insertion order.  Cache by the raw word
    # tuple (after dropping e=0 entries) to be safe.
    raw_key = tuple(t for t in word if t[2])
    cached = _HEPT_REDUCE_CACHE.get(raw_key)
    if cached is not None:
        return cached
    out: dict[tuple[tuple[int, int, int], ...], LaurentPoly] = {}

    stack: list[tuple[LaurentPoly, list[tuple[int, int, int]]]] = [
        (LaurentPoly.one(), [t for t in word if t[2] != 0])
    ]

    while stack:
        coeff, w = stack.pop()
        if coeff.is_zero():
            continue
        # Reduce one step.  Returns None iff `w` is already in canonical
        # basis form (sorted, distinct letters, all pairs fq-commute).
        step = _hept_step(w)
        if step is None:
            # Convert L-product to BPS X-basis via the X-offset twist.
            bps_twist = _hept_canonical_bps_twist(w)
            label = tuple(w)
            adj = _hept_lp_mul(coeff, LaurentPoly.q(bps_twist)) if bps_twist else coeff
            cur = out.get(label)
            if cur is None:
                out[label] = adj
            else:
                s = _hept_lp_add(cur, adj)
                if s._coeffs:
                    out[label] = s
                else:
                    out.pop(label, None)
            continue
        for (factor, new_word) in step:
            new_coeff = _hept_lp_mul(coeff, factor) if factor._coeffs else coeff
            stack.append((new_coeff, new_word))

    _HEPT_REDUCE_CACHE[raw_key] = out
    return out


def _hept_single(k: int, i: int) -> tuple[tuple[int, int, int], ...]:
    """Canonical-form label for the single named generator `L((k, i))`."""
    return ((k, i % _HEPT_H, 1),)


_HEPT_RHO_KEY_CACHE: dict[tuple, tuple] = {}


def _hept_label_rho_canonical_key(label) -> tuple:
    """ρ-canonical key for a label: lex-smallest of the 7 ρ-rotations.
    Used for memoising trace values across the Z/7 orbit of a label."""
    if not label:
        return ()
    cached = _HEPT_RHO_KEY_CACHE.get(label)
    if cached is not None:
        return cached
    best = None
    for shift in range(_HEPT_H):
        rotated = tuple(sorted((k, (i + shift) % _HEPT_H, e) for (k, i, e) in label))
        if best is None or rotated < best:
            best = rotated
    _HEPT_RHO_KEY_CACHE[label] = best
    return best


def _laurent_div(num: LaurentPoly, denom: LaurentPoly) -> LaurentPoly:
    """Divide `num` by `denom` in `Z[fq^±]`.  Raises `ValueError` if the
    quotient is not in `Z[fq^±]` (i.e., division has a remainder or a
    non-integer coefficient).  Used by the trace closure solver to invert
    factors like `1 - fq^c`."""
    if denom.is_zero():
        raise ZeroDivisionError("_laurent_div: zero denominator")
    if num.is_zero():
        return LaurentPoly.zero()
    # Shift both polynomials so denom starts at fq^0.
    denom_min = min(denom._coeffs)
    p_denom: dict[int, int] = {e - denom_min: c for e, c in denom._coeffs.items()}
    p_denom_max = max(p_denom)
    # Shift num so its min degree is 0; track the shift adjustment for the
    # quotient's final exponent.  shift_adj = num_min - denom_min.
    num_min = min(num._coeffs)
    p_num: dict[int, int] = {e - num_min: c for e, c in num._coeffs.items()}
    shift_adj = num_min - denom_min
    lead = p_denom[p_denom_max]
    if lead not in (1, -1):
        raise ValueError(f"_laurent_div: leading denom coeff {lead} not ±1")
    quot: dict[int, int] = {}
    rem = dict(p_num)
    while rem and max(rem) >= p_denom_max:
        top = max(rem)
        c_top = rem[top]
        shift = top - p_denom_max
        q_coef = c_top // lead
        if q_coef * lead != c_top:
            raise ValueError(
                f"_laurent_div: non-integer quotient coefficient at degree {top}"
            )
        quot[shift] = quot.get(shift, 0) + q_coef
        if quot[shift] == 0:
            del quot[shift]
        for e, c in p_denom.items():
            ne = e + shift
            rem[ne] = rem.get(ne, 0) - q_coef * c
            if rem[ne] == 0:
                del rem[ne]
    if rem:
        raise ValueError(f"_laurent_div: non-zero remainder {rem}")
    out = {e + shift_adj: c for e, c in quot.items() if c != 0}
    return LaurentPoly(out)


def _hept_trace_layer1(label, memo: dict,
                       in_progress: dict | None = None
                       ) -> tuple[LaurentPoly, LaurentPoly, LaurentPoly,
                                  dict]:
    """Bubble-cycle trace-Layer-1 reducer.

    Returns `(c_0, c_L, c_N, γ_dict)` representing the equation

        Tr(label)  =  c_0 · T_0 + c_L · T_L + c_N · T_N
                       + Σ_K  γ_dict[K] · Tr(label_at_K)

    where each `K ∈ γ_dict` is the ρ-canonical key of an ancestor
    label *still in progress* on the call stack (the sub-call has not
    yet closed its own self-reference).  At the top-level call (where
    `in_progress` is empty on entry), `γ_dict` returns empty.

    Algorithm:
      1.  Base cases: `()` → T_0,  `((2, i, 1),)` → T_L,  `((1, i, 1),)` → T_N.
      2.  Add the label's ρ-canonical key to `in_progress`.
      3.  For each candidate tag-position (short-diagonal N's preferred
          over long-diagonal L's), apply the bubble-cycle step at that
          tag:
            (a) Bubble one copy of `L_tag` to the right end of the
                L-product, accumulating  `bubble_twist`  in
                q-commutation factors.
            (b) Apply ρ²-twisted cyclicity `Tr(A·L_tag) =
                Tr(ρ²(L_tag)·A)` to cycle the tag back to the left.
            (c) Reduce the cycled word in the BPS X-basis via
                `_hept_reduce`, getting `{term_label: α_term}`.
            (d) For each term, recurse to get its
                `(c_0, c_L, c_N, γ_dict)`; combine.  The cycle factor
                is  `fq^{-T_bps + bubble_twist}`.
            (e) If the term's ρ-canonical key is already in `in_progress`,
                the recursion returns the unit γ-marker
                `{term_key: 1}` instead of recursing further (avoids
                cycling).
      4.  Close on our own ρ-key:  `γ_self = γ_dict.pop(rho_key)`,
          `closure = 1 − γ_self`, then divide all `(c_0, c_L, c_N,
          γ_dict_others)` by `closure` (integer Laurent division;
          raises on non-clean division).
      5.  Memoise the result if `γ_dict_others` is empty (fully
          resolved); otherwise return as-is for the ancestor to
          resolve.

    The cycle factor for the single ρ²-shift on a tag at position
    `tag_idx` in the canonical label is

        fq_cycle  =  fq^{−T_bps  +  bubble_twist},

    where  `T_bps = Σ_{i<j} c_fwd(l_i, l_j) e_i e_j`  (the X-vs-L
    offset) and  `bubble_twist = Σ_{r > tag_idx} c_sw(tag, l_r) e_r`.
    Multi-cycle (`ρ^{2k}` for `k > 1`) is *not* used here: each ρ²-step
    transforms the cycled algebra element, and the per-step
    bubble_twist generally changes after the transformation, so the
    closed-form `k · bubble_twist` is wrong in general.  When a
    single-cycle equation doesn't pin Tr, the recursion proceeds via
    sub-labels through `γ_dict` propagation."""
    if not isinstance(memo, dict):
        raise TypeError("_hept_trace_layer1: memo must be a dict")
    if in_progress is None:
        in_progress = set()

    zero = LaurentPoly.zero()
    one = LaurentPoly.one()
    rho_key = _hept_label_rho_canonical_key(label)
    empty_gamma: dict = {}

    # Base cases (no self-ref possible).
    if label == ():
        return (one, zero, zero, empty_gamma)
    if len(label) == 1 and label[0][2] == 1:
        k = label[0][0]
        if k == 2:
            return (zero, one, zero, empty_gamma)
        if k == 1:
            return (zero, zero, one, empty_gamma)
        raise ValueError(f"unknown orbit index k={k} in label {label}")

    # Sub-cycle: this label is currently in progress at an ancestor
    # level.  Return the unit γ-marker for it; the ancestor will close
    # this reference.
    if rho_key in in_progress:
        return (zero, zero, zero, {rho_key: one})

    # Memoised (already fully resolved).
    if rho_key in memo:
        c0, cL, cN = memo[rho_key]
        return (c0, cL, cN, empty_gamma)

    in_progress.add(rho_key)
    try:
        tag_candidates = _hept_tag_candidates(label)
        candidates_results: list = []  # list of (c0, cL, cN, gamma_dict) per tag
        for tag_idx in tag_candidates:
            result = _hept_try_cycles_at_tag(
                label, tag_idx, memo, in_progress
            )
            if result is None:
                continue
            candidates_results.append(result)

        if not candidates_results:
            raise NotImplementedError(
                f"trace_layer1: no productive cycle found for label {label}.  "
                f"This label is outside the bubble-cycle algorithm's reach in "
                f"the current implementation."
            )

        # Process each candidate: close on γ_self (own rho_key) via
        # `(1 − γ_self)` division.  Keep only the candidates with a
        # cleanly invertible closure in Z[fq^±].  Among those, prefer
        # those with empty γ_others (no further self-references to
        # propagate), then those with concrete c_*.
        processed: list = []
        for result in candidates_results:
            c0, cL, cN, gamma_dict = result
            gamma_self = gamma_dict.pop(rho_key, zero)
            closure = one + (gamma_self * LaurentPoly({0: -1}))  # 1 − γ_self
            if closure.is_zero():
                continue
            if closure == one:
                processed.append((c0, cL, cN, gamma_dict))
                continue
            try:
                c0_d = _laurent_div(c0, closure)
                cL_d = _laurent_div(cL, closure)
                cN_d = _laurent_div(cN, closure)
                gamma_d = {k: _laurent_div(v, closure)
                           for k, v in gamma_dict.items()}
            except ValueError:
                continue
            processed.append((c0_d, cL_d, cN_d, gamma_d))

        def _score(r):
            c0_, cL_, cN_, g_ = r
            empty_g = 0 if g_ else 1
            concrete = 0 if (c0_.is_zero() and cL_.is_zero()
                             and cN_.is_zero()) else 1
            return (empty_g, concrete)

        chosen = max(processed, key=_score, default=None)

        if chosen is None:
            raise NotImplementedError(
                f"trace_layer1: no candidate cycle has an invertible closure "
                f"for label {label}.  Linear system over Q(fq) would be "
                f"needed."
            )

        c0_out, cL_out, cN_out, gamma_out = chosen
        if not gamma_out:
            # Fully resolved.  Memoise.
            memo[rho_key] = (c0_out, cL_out, cN_out)
        return (c0_out, cL_out, cN_out, gamma_out)
    finally:
        in_progress.discard(rho_key)


def _hept_tag_candidates(label) -> list[int]:
    """Return tag-index candidates for the bubble-cycle reducer, ordered
    by preference: short-diagonal letters (k=1) first, in canonical
    order; then long-diagonal letters (k=2)."""
    short = [ii for ii, (k, _i, _e) in enumerate(label) if k == 1]
    longs = [ii for ii, (k, _i, _e) in enumerate(label) if k == 2]
    return short + longs


def _hept_try_cycles_at_tag(label, tag_idx, memo, in_progress
                            ) -> tuple[LaurentPoly, LaurentPoly,
                                       LaurentPoly, dict] | None:
    """For a fixed tag position, apply one ρ²-cycle and return
    `(c_0, c_L, c_N, γ_dict)` where γ_dict tracks self-references to
    ancestors still in progress.  Returns `None` if the cycle yields
    no useful info (e.g., reduce result is empty)."""
    zero = LaurentPoly.zero()
    label_rho_key = _hept_label_rho_canonical_key(label)
    k_tag, i_tag, e_tag = label[tag_idx]

    # T_bps for the label (the BPS X-vs-L-product offset).
    T_bps = 0
    for ii in range(len(label)):
        for jj in range(ii + 1, len(label)):
            la = (label[ii][0], label[ii][1])
            lb = (label[jj][0], label[jj][1])
            T_bps += _hept_forward_q_coeff(la, lb) * label[ii][2] * label[jj][2]

    # Bubble twist for one copy of L_tag from `tag_idx` to the right end.
    bubble_twist = 0
    for r in range(tag_idx + 1, len(label)):
        l_r = (label[r][0], label[r][1])
        e_r = label[r][2]
        c_sw = _hept_qcommute_factor((k_tag, i_tag), l_r)
        # Sanity: all letters in a canonical label pairwise q-commute, so
        # c_sw must not be None.
        if c_sw is None:
            raise AssertionError(
                f"_hept_try_cycles_at_tag: canonical label {label} has a "
                f"Plücker pair at tag={tag_idx}, l_r={l_r}"
            )
        bubble_twist += c_sw * e_r

    cycle_pow = -T_bps + bubble_twist
    cycle_factor = LaurentPoly.q(cycle_pow)

    # Sub-label = label with one copy of L_tag removed.
    rest_letters: list[tuple[int, int, int]] = []
    for jj, (kj, ij, ej) in enumerate(label):
        if jj == tag_idx:
            if ej > 1:
                rest_letters.append((kj, ij, ej - 1))
        else:
            rest_letters.append((kj, ij, ej))

    # Single ρ²-cycle from the chosen tag position.  Multi-cycle
    # (`ρ^{2k}` for `k > 1`) is *not* used: each ρ²-step transforms the
    # cycled algebra element, so the per-step bubble_twist generally
    # changes and the closed-form `k · bubble_twist` is wrong in general.
    i_new = (i_tag + 2) % _HEPT_H
    cycled_word = [(k_tag, i_new, 1)] + list(rest_letters)
    reduced = _hept_reduce(cycled_word)

    rhs_c0 = LaurentPoly.zero()
    rhs_cL = LaurentPoly.zero()
    rhs_cN = LaurentPoly.zero()
    rhs_gamma: dict = {}

    for term_label, term_coeff in reduced.items():
        t0, tL, tN, tg_dict = _hept_trace_layer1(
            term_label, memo, in_progress
        )
        rhs_c0 = _hept_lp_add(rhs_c0, _hept_lp_mul(term_coeff, t0))
        rhs_cL = _hept_lp_add(rhs_cL, _hept_lp_mul(term_coeff, tL))
        rhs_cN = _hept_lp_add(rhs_cN, _hept_lp_mul(term_coeff, tN))
        for k, v in tg_dict.items():
            prev = rhs_gamma.get(k)
            if prev is None:
                rhs_gamma[k] = _hept_lp_mul(term_coeff, v)
            else:
                rhs_gamma[k] = _hept_lp_add(prev, _hept_lp_mul(term_coeff, v))

    # Apply the cycle factor.
    rhs_c0 = _hept_lp_mul(cycle_factor, rhs_c0)
    rhs_cL = _hept_lp_mul(cycle_factor, rhs_cL)
    rhs_cN = _hept_lp_mul(cycle_factor, rhs_cN)
    rhs_gamma = {k: _hept_lp_mul(cycle_factor, v) for k, v in rhs_gamma.items()}
    # Drop zero gamma entries.
    rhs_gamma = {k: v for k, v in rhs_gamma.items() if v._coeffs}

    return (rhs_c0, rhs_cL, rhs_cN, rhs_gamma)



class HeptagonKAlg(ConeKAlgebra):
    """The heptagon K-algebra `A_𝖖([A_1, A_4])`, on its own labels.

    *Definition.*  Generators `L((k, i))` for `k ∈ {1, 2}` and
    `i ∈ Z/7`, the diagonals of the heptagon (the `(2k+3)`-gon for
    `k = 2`, marked points `0, …, 6`): `L((1, i))` is the diagonal
    `{i, i + 2}` and `L((2, i))` the diagonal `{i, i + 4}`; 14 letters in
    total.  Relations are the Z/7 quantum Ptolemy relations at
    `k = 2`.  Canonical-basis labels are sorted
    tuples `((k₁, i₁, e₁), ..., (k_m, i_m, e_m))` with `e_r ≥ 1` and the
    letter pairs `(k_r, i_r)` pairwise q-commuting; the empty tuple
    is the identity.

    *Labels versus `A1A2kKAlg(2)`.*  The tuple format is
    `A1A2kKAlg(2)`'s, but the orbit-2 index is not: this class's `(2, i)`
    is `A1A2kKAlg(2)`'s `(2, i + 4)` (there `(a, j)` is the diagonal
    `{j, j + a + 1}`), while `(1, i)` is `(1, i)`.  The same tuple in the
    two classes is in general a different element — under the identity
    map on labels 140 of the 196 generator products differ.  Products are
    served through `A1A2kKAlg(2)` under this relabelling: `multiply`
    relabels, multiplies there and relabels back, and the per-pair
    expansions behind `cone_data()` (`HEPTAGON_CONE_DATA`, which drives
    the Layer-1 trace) are `A1A2kKAlg(2)`'s generator products,
    relabelled.  No product table is stored in this module.

    *Property.*  Satisfies the KAlgebra axioms; in code, exposed by
    subclassing `cone_kalgebra.ConeKAlgebra` (the closed-form
    presentation tier).  `ρ(L((k, i))) = L((k, i+1))` generates `Z/7`
    and extends to basis labels by shifting every letter index (the
    rotation of the heptagon, which commutes with the relabelling).

    *Property (separate theorem).*  There is a KAlgebra isomorphism
    `A_𝖖([A_1, A_4]) ≅ A_𝖖^BPS(A_4-quiver)`; the chord generators
    correspond to specific F-elements of the BPS realisation.  See
    `kalgebra_iso.KAlgebraIso` and the BPSKAlgebra-wrapper variant
    `heptagon_kalg.HeptagonKAlg` (which exposes the same generator
    indexing).  This iso is *not* used at runtime here.

    Trace is **two-layered** as in pentagon: `trace_layer1(label) →
    (c_0, c_L, c_N)` expresses

        Tr(label) = c_0 · T_0 + c_L · T_L + c_N · T_N

    with `Z[𝖖, 𝖖^{-1}]` coefficients, where  T_0 = Tr(𝟙),
    T_L = Tr(L((2, *))),  T_N = Tr(L((1, *)))  are the three
    elementary trace values matching the three primaries of the
    M(2, 7) Virasoro minimal model.  Layer 2 (`_trace_residual`
    below) supplies the three seeds from `A1A2kKAlg(2)`'s
    Andrews-Gordon M(2,7) character series, so `trace()` is the full
    exact two-layer trace (certified against the vortex-residue theta
    forms over the frozen K=48 window, in the suite of the source
    repository)."""

    _R = TrivialZPlusRing()
    H = _HEPT_H

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return ()

    def L(self, label: tuple[int, int]):
        """Canonical-form label for the single generator `L((k, i))`."""
        k, i = label
        return _hept_single(k, i)

    # ----- geometry: the diagonals of the heptagon ----------------------

    def curve(self, x: int, ell: int):
        """The label of the diagonal from marked point `x` to `x + ell`,
        `2 ≤ ell ≤ 5`: `A1A2kKAlg(2).curve(x, ell)` read through this class's
        relabelling.  The heptagon has no puncture, so
        `curve(x, ell) == curve(x + ell, 7 − ell)`; a boundary edge (`ell = 1`
        or `6`) is the identity and not a letter, and raises, as does any
        other `ell`."""
        return _hept_label_from_a1a2k(_hept_a1a2k().curve(x, ell))

    def geometric_label(self, label):
        """The label — letters `(k, i, e)` — as the multiset of its diagonals:
        a sorted tuple of pairs `((v1, v2), e)`, `(v1, v2)` the letter's
        diagonal (sorted marked points: `{i, i + 2}` for `k = 1`,
        `{i, i + 4}` for `k = 2`, the class docstring) and `e ≥ 1` its power;
        the unit is `()`.  The letters of a label pairwise q-commute, so its
        diagonals do not cross.  The layout of the `curves` of `A1DnKAlg`'s
        `(curves, κ)`, on this unflavoured polygon without a puncture."""
        span = {1: 2, 2: 4}
        return tuple(sorted(
            (tuple(sorted((i % _HEPT_H, (i + span[k]) % _HEPT_H))), e)
            for (k, i, e) in label))

    def cone_data(self):
        from heptagon_cone_data import HEPTAGON_CONE_DATA
        return HEPTAGON_CONE_DATA

    def multiply(self, a, b):
        """`L_a · L_b`, served by `A1A2kKAlg(2)` through the orbit-2
        relabelling `(2, i) ↦ (2, i + 4)` (class docstring); structure
        constants are cached on the instance."""
        cache = getattr(self, "_multiply_cache", None)
        if cache is None:
            cache = {}
            self._multiply_cache = cache
        key = (a, b)
        hit = cache.get(key)
        if hit is not None:
            return hit
        prod = _hept_a1a2k().multiply(_hept_label_to_a1a2k(a),
                                      _hept_label_to_a1a2k(b))
        result = Element({_hept_label_from_a1a2k(lbl): c
                          for lbl, c in prod.terms.items()})
        cache[key] = result
        return result

    def _legacy_multiply(self, a, b):
        """Legacy multiplication via `_hept_reduce` + BPS X-vs-L-product
        offset twist.  Kept as a reference for testing `multiply`: a
        separate reducer over the same per-pair expansions
        (`_hept_pair_product`).  Not on the live `multiply` path."""
        # Basis labels `a`, `b` are in the BPS X-basis; their L-product
        # representations are  fq^{-T_bps_*} · sorted-letters-of-*.
        # Concatenating the WORDS  list(a) + list(b)  and handing to
        # `_hept_reduce` gives the structure constants for
        # `L-product(a) · L-product(b) = Σ α_c · X[c]`.  Bridging to
        # the X-basis on the *inputs* requires the prefactor
        # `fq^{-T_bps_a - T_bps_b}`, so
        #   X[a] · X[b]  =  fq^{-T_bps_a - T_bps_b} · Σ α_c · X[c].
        T_a = _hept_T_bps(a)
        T_b = _hept_T_bps(b)
        word = list(a) + list(b)
        raw = _hept_reduce(word)
        if T_a == 0 and T_b == 0:
            terms = dict(raw)
        else:
            prefactor = LaurentPoly.q(-T_a - T_b)
            terms = {lbl: _hept_lp_mul(prefactor, c) for lbl, c in raw.items()}
        return Element(terms)

    def rho(self, a):
        if not a:
            return ()
        # Shift every letter's i by +1; canonicalise (preserves fq-commute
        # structure since ρ is an algebra automorphism, so no twist).
        shifted = [(k, (i + 1) % self.H, e) for (k, i, e) in a]
        # Already sorted iff original was sorted (ρ preserves the order
        # on (k, *), and the shift by 1 mod 7 may rotate the sort);
        # re-sort to be safe.
        shifted.sort()
        return tuple(shifted)

    def rho_inverse(self, a):
        if not a:
            return ()
        shifted = [(k, (i - 1) % self.H, e) for (k, i, e) in a]
        shifted.sort()
        return tuple(shifted)

    # ---- Trace Layer 1 ----
    #
    # `trace_layer1(label)` returns `(c_0, c_L, c_N)` ∈ Z[fq^±]³ with
    #   Tr(label) = c_0 · T_0 + c_L · T_L + c_N · T_N.
    #
    # By ρ-invariance and cyclicity, Tr depends only on the
    # ρ-equivalence class of the label.  For the named-letter window
    # (basis labels with total weight ≤ 1) the reduction is direct:
    #   Tr(())                       = T_0,
    #   Tr(((2, i, 1),))             = T_L,
    #   Tr(((1, i, 1),))             = T_N.
    # Every other label is reduced to these by the cone data's tagged
    # ρ²-cyclicity (`trace_layer1` below).

    def trace_layer1(self, label) -> tuple[LaurentPoly, LaurentPoly, LaurentPoly]:
        """Layer 1 trace reduction via the generic cone-data engine.

        Returns `(c_0, c_L, c_N)` ∈ Z[q^±]³ with
            `Tr(label) = c_0 · T_0 + c_L · T_L + c_N · T_N`,
        where `T_0 = Tr(𝟙)`, `T_L = Tr(L((2, *)))`, `T_N = Tr(L((1, *)))`.

        Implementation: route through `cone_data().simplify_trace_via_cone_data`
        which reduces the label to an Element supported on heptagon's
        trace seeds (`()`, `((1, i, 1),)`, `((2, i, 1),)`) by iterated
        tagged cyclicity, then read off the seed coefficients.  By
        ρ-invariance `Tr(((k, i, 1),)) = Tr(((k, 0, 1),))` for any `i`,
        so we sum the `(1, *, 1)` entries into `c_N` and the
        `(2, *, 1)` entries into `c_L`.
        """
        memo = getattr(self, "_trace_layer1_memo", None)
        if memo is None:
            memo = {}
            self._trace_layer1_memo = memo
        cached = memo.get(_hept_label_rho_canonical_key(label))
        if cached is not None:
            return cached
        cd = self.cone_data()
        simplified = cd.simplify_trace_via_cone_data(self, label)
        zero = LaurentPoly.zero()
        c0 = simplified.terms.get((), zero)
        cN = zero
        cL = zero
        for i in range(self.H):
            cN = cN + simplified.terms.get(((1, i, 1),), zero)
            cL = cL + simplified.terms.get(((2, i, 1),), zero)
        result = (c0, cL, cN)
        memo[_hept_label_rho_canonical_key(label)] = result
        return result

    def _legacy_trace_layer1(self, label) -> tuple[LaurentPoly, LaurentPoly, LaurentPoly]:
        """Legacy Layer 1 via the bubble-cycle / γ_dict closure
        algorithm (`_hept_trace_layer1`).  Kept as an independent
        reference for testing `trace_layer1` (which now routes through
        cone_data).  Not on the live `trace_layer1` path."""
        legacy_memo: dict = {}
        c0, cL, cN, gamma = _hept_trace_layer1(label, legacy_memo, in_progress=None)
        assert not gamma, (
            f"_legacy_trace_layer1: residual γ-dict={gamma} for {label}"
        )
        return (c0, cL, cN)

    # ----- Layer-2 trace residual ---------------------------------------
    #
    # Canonical ρ²-orbit seeds produced by Layer 1 (tagged-cycle +
    # ρ²-orbit canonicalisation in `simplify_trace_via_cone_data`):
    #   * `()`            -- identity                  → T_0 = Tr(𝟙)
    #   * `((1, 0, 1),)`  -- canonical orbit-1 seed    → T_1 = Tr(L((1, *)))
    #   * `((2, 0, 1),)`  -- canonical orbit-2 seed    → T_2 = Tr(L((2, *)))
    #
    # Layer-2 plug-in: the three M(2, 7) Andrews-Gordon characters
    # (Virasoro minimal-model primaries), computed via
    # `a1a2k_kalg.A1A2kKAlg(k=2)._compute_T_series(K)`.  Heptagon is the
    # k=2 case of the [A_1, A_{2k}] family; the elementary trace values
    # T_0, T_1, T_2 are given by the same Z[q^±]-combination of
    # M(2, 2k+3=7) characters that A1A2k uses at k=2.  Concretely:
    #
    #     T_0 = chi_1(q^2)
    #     T_a = (-1)^{m+1} q^{-m} (chi_m - chi_{m+1})(q^2)   (a = 1, 2)
    #
    # with `m` looked up via the natural-orbit ↔ alternating-pattern
    # length table in `A1A2k_naming_audit.predicted_lengths_and_shifts`.
    # See `a1a2k_kalg.A1A2kKAlg._compute_T_series` for the verified
    # formula (PR, k = 1..6).
    #
    # The Heptagon trace seeds T_0, T_1, T_2 correspond to A1A2k(k=2)'s
    # T_0, T_1, T_2 respectively (same indexing: orbit-a maps to T_a; the
    # orbit-2 relabelling only moves a seed within its ρ-orbit).
    #
    # `trace` is inherited from `ConeKAlgebra` (`multiply` is served by
    # `A1A2kKAlg(2)`, above).  ρ²-invariance on the 7 + 7 single-mult-gen
    # seeds is enforced by Layer 1, not by this method.

    def _trace_residual(self, seed_label, K):
        # The shared A1A2kKAlg(2) instance (the one serving `multiply`).
        a1a2k_k2 = _hept_a1a2k()
        # Compute T_0, T_1, T_2 as RPowerSeries in q (each K request
        # rebuilds).
        T_series = a1a2k_k2._compute_T_series(K)
        # T_series is a list [T_0, T_1, T_2] of RPowerSeries.
        if seed_label == ():
            return T_series[0]
        if seed_label == ((1, 0, 1),):
            return T_series[1]
        if seed_label == ((2, 0, 1),):
            return T_series[2]
        raise ValueError(
            f"HeptagonKAlg._trace_residual: unexpected seed {seed_label!r}; "
            f"expected canonical ρ²-orbit representative "
            f"(), ((1, 0, 1),), or ((2, 0, 1),)"
        )

    # ---- Cache persistence ----
    #
    # `_multiply_cache` and `_trace_layer1_memo` accumulate as the
    # algebra is exercised.  For larger-scale computations (Tr L^a at
    # large `a`, etc.) it's worth saving the cache to disk so a fresh
    # session can resume warm.

    _CACHE_FORMAT_VERSION = "heptagon_v1"

    def save_cache(self, path: str) -> None:
        """Persist `_multiply_cache` and `_trace_layer1_memo` to `path`
        (JSON).  Format-tagged with `_CACHE_FORMAT_VERSION` so reloads
        can reject mismatched formats."""
        import json as _json

        def _lp_to_obj(lp: LaurentPoly) -> dict[str, int]:
            return {str(e): int(c) for e, c in lp._coeffs.items()}

        def _label_key(label) -> str:
            return _json.dumps([list(t) for t in label])

        def _multiply_pair_key(pair) -> str:
            return _json.dumps([list(t) for t in pair[0]]
                                + [None]
                                + [list(t) for t in pair[1]])

        mul_obj: dict[str, dict[str, dict[str, int]]] = {}
        for (a, b), elem in getattr(self, "_multiply_cache", {}).items():
            mul_obj[_multiply_pair_key((a, b))] = {
                _label_key(lbl): _lp_to_obj(coeff)
                for lbl, coeff in elem.terms.items()
            }

        trace_obj: dict[str, list[dict[str, int]]] = {}
        for rho_key, (c0, cL, cN) in getattr(self, "_trace_layer1_memo", {}).items():
            trace_obj[_label_key(rho_key)] = [
                _lp_to_obj(c0), _lp_to_obj(cL), _lp_to_obj(cN),
            ]

        data = {
            "format": self._CACHE_FORMAT_VERSION,
            "multiply_cache": mul_obj,
            "trace_layer1_memo": trace_obj,
        }
        with open(path, "w") as fh:
            _json.dump(data, fh)

    def load_cache(self, path: str) -> tuple[int, int]:
        """Load caches from `path` (JSON).  Returns
        `(n_multiply, n_trace)` -- the counts of entries loaded.
        Existing in-memory caches are extended (not replaced)."""
        import json as _json
        with open(path) as fh:
            data = _json.load(fh)
        fmt = data.get("format")
        if fmt != self._CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Cache file {path!r} has format {fmt!r}, expected "
                f"{self._CACHE_FORMAT_VERSION!r}"
            )

        def _obj_to_lp(obj: dict) -> LaurentPoly:
            lp = LaurentPoly.__new__(LaurentPoly)
            lp._coeffs = {int(e): int(c) for e, c in obj.items() if int(c) != 0}
            return lp

        def _label_from_key(s: str) -> tuple:
            return tuple(tuple(t) for t in _json.loads(s))

        def _multiply_pair_from_key(s: str) -> tuple:
            parts = _json.loads(s)
            sep = parts.index(None)
            a = tuple(tuple(t) for t in parts[:sep])
            b = tuple(tuple(t) for t in parts[sep + 1:])
            return (a, b)

        mul_cache = getattr(self, "_multiply_cache", None)
        if mul_cache is None:
            mul_cache = {}
            self._multiply_cache = mul_cache
        n_mul = 0
        for pair_key, term_obj in data.get("multiply_cache", {}).items():
            pair = _multiply_pair_from_key(pair_key)
            terms = {
                _label_from_key(lbl_key): _obj_to_lp(coeff_obj)
                for lbl_key, coeff_obj in term_obj.items()
            }
            mul_cache[pair] = Element(terms)
            n_mul += 1

        trace_memo = getattr(self, "_trace_layer1_memo", None)
        if trace_memo is None:
            trace_memo = {}
            self._trace_layer1_memo = trace_memo
        n_trace = 0
        for key_str, triple in data.get("trace_layer1_memo", {}).items():
            rho_key = _label_from_key(key_str)
            c0, cL, cN = (_obj_to_lp(t) for t in triple)
            trace_memo[rho_key] = (c0, cL, cN)
            n_trace += 1

        return (n_mul, n_trace)
