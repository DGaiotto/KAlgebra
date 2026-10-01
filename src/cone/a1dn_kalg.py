"""`A1DnKAlg(n)` -- the [A_1, D_n] Argyres-Douglas K-algebra for **odd**
n ≥ 3, over `R(SU(2))`, with canonical-basis labels that are simple curves
on the n-gon with one interior puncture.

Rebuilt 2026-09-23.  Until then this module held a different class — the
SU(2)-symmetrised quantum torus of the D_n BPS-quiver chamber, whose trace was
the flat-torus trace (not the Schur index: `Tr 1 = 1 − 2𝖖² − 𝖖⁴` at n = 3,
where `[A_1, D_3]` has `1 + χ₂𝖖² + …`) and whose ρ was the antipode.  That
class is retired (it lived in `src/abe/` in earlier releases of this
repository; this module now lives in `src/cone/`).

Surface
-------
The n-gon with marked points `0, …, n−1` (counter-clockwise on the boundary)
and one regular interior puncture P, which carries the SU(2) flavour (the
family map, user 2026-07-10: "A1Dn is a polygon with an extra regular puncture
in the middle").  No curve ends at P.

Curves and labels
-----------------
A curve `(x, ℓ)`, `x ∈ Z/n`, `2 ≤ ℓ ≤ n`, is the simple curve from marked point
`x` to `x + ℓ` with `ℓ` boundary edges on the side not containing P; `ℓ = n` is
the loop at `x` around P.  There are `n(n−1)` curves.  A canonical-basis label
is `(curves, κ)`:

  * `curves` — a sorted tuple of `((x, ℓ), m)`, `m ≥ 1`, the curves pairwise
    non-crossing (a cone monomial; a label with crossing curves is refused);
  * `κ ≥ 0` — the SU(2) highest weight: `L_{(curves, κ)} = χ_κ · L_{(curves, 0)}`,
    `χ_κ` the character of the irrep of highest weight κ.  In the skein
    tier's reading the peripheral loop around P is `χ₁`
    (`A1DoddSkeinConeData`), and `χ_κ` is the polynomial in the loop that
    Clebsch–Gordan fixes, not its κ-th power: `χ₁² = χ₀ + χ₂`.

The identity is `((), 0)`.  Two curves cross `c` times, `c ∈ {0, 1, 2}`,
counted in the universal cover around P (`_arc_crossings`): they `𝖖`-commute
iff `c = 0`, and their product carries `χ₁` iff `c = 2`.  The parity
`κ + Σ m·p(ℓ) (mod 2)`, `p(ℓ) = (ℓ + 1) mod 2`, is conserved by products.

Arithmetic (exact delegation)
-----------------------------
`multiply`, `rho` / `rho_inverse` and `trace` are computed by
`a1dodd_kalg.A1DoddConeKAlg(k)`, `k = (n − 3)/2`, through the bijection of
canonical labels

    (a, p, i)  ↦  (x, ℓ):   ℓ = 2a + 1 (p = 0),  ℓ = 2(k + 2 − a) (p = 1),
                            x = (_ap_base(a, p, k) + i) mod n

(`a1dodd_cone_data`'s generators; `_ap_base` is that module's per-orbit
origin).  Verified at k = 0..4 (the suite in the source repository): a bijection onto the
curves, `𝖖`-commuting ⟺ no crossing, the `χ₁`-carrying relations ⟺ two
crossings, and ρ = `x ↦ x + 1` (κ fixed, order n).  Because the dictionary is a
bijection of canonical labels, the delegated structure constants are exact.
The two presentations are related by the certified iso
`a1dn_a1dodd_iso.a1dn_a1dodd_iso(n)`.

Trace: the SU(2)-refined Schur index (`A1DoddConeKAlg.trace`: the
ρ²-cyclicity reduction, then `a1dodd_layer2`): `Tr(1)` is
`a1dodd_layer2.vacuum_trace(k)`, and the trace of a single curve depends only
on ℓ, `seed_trace_ap(k, a(ℓ), p(ℓ))`.

`cone_data()` is the same cone data in curve letters (`_CurveConeData`):
`q_commute` is the crossing count, `cocycle` and `cross_product` are
translated from `a1dodd_cone_data(k)`; its generic `derived_multiply` equals
the delegated `multiply` (the suite in the source repository).  It is what
`A1DoddSkeinKAlgebra(k, intrinsic=A1DnKAlg(2k + 3))` reads.

Even n
------
`[A_1, D_n]` with n even has flavour SU(2)×U(1) (SU(3) at n = 4); this class
refuses it at construction, naming the classes that exist.

Self-contained: no BPS / RG machinery at runtime (checked by the test-suite on
`sys.modules`).  `_build_d_n_pairing` (the D_n BPS-quiver exchange matrix) is
kept as a module-level helper: the suite in the source repository checks that it is
the unfolded chart of the `GBPSKAlgebra` anchor at n = 3.
"""

from __future__ import annotations

import sys
import os
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from cone_kalgebra import ConeKAlgebra
from cone_data import FiniteConeData
from zplus_ring import SU2ZPlusRing, RLaurent, RPowerSeries


# ---------------------------------------------------------------------------
# The D_n BPS-quiver exchange matrix (kept: the GBPS anchor test reads it)
# ---------------------------------------------------------------------------


def _build_d_n_pairing(n: int) -> list[list[int]]:
    """D_n antisymmetric pairing (Bourbaki, 0-indexed)."""
    if n < 3:
        raise ValueError(f"n must be ≥ 3, got {n}")
    B = [[0] * n for _ in range(n)]
    # Stem edges: 0—1, 1—2, …, (n−4)—(n−3).
    for i in range(n - 3):
        B[i][i + 1] = 1
        B[i + 1][i] = -1
    # Trivalent-to-leaves: (n−3)—(n−2) and (n−3)—(n−1).
    B[n - 3][n - 2] = 1
    B[n - 2][n - 3] = -1
    B[n - 3][n - 1] = 1
    B[n - 1][n - 3] = -1
    return B


# ---------------------------------------------------------------------------
# The curve dictionary (private)
# ---------------------------------------------------------------------------


def _ap_to_arc(g, k: int):
    """`a1dodd_cone_data(k)` generator `(a, p, i)` ↦ curve `(x, ℓ)`."""
    from a1dodd_cone_data import _ap_gap, _ap_base
    a, p, i = g
    n = 2 * k + 3
    return ((_ap_base(a, p, k) + i) % n, _ap_gap(a, p, k))


def _arc_to_ap(c, k: int):
    """Curve `(x, ℓ)` ↦ `a1dodd_cone_data(k)` generator `(a, p, i)`."""
    from a1dodd_cone_data import _ap_base
    x, ell = c
    n = 2 * k + 3
    if not (2 <= ell <= n):
        raise ValueError(f"curve {c!r}: need 2 ≤ ℓ ≤ {n}")
    if ell % 2:
        p, a = 0, (ell - 1) // 2
    else:
        p, a = 1, k + 2 - ell // 2
    return (a, p, (x - _ap_base(a, p, k)) % n)


def _arc_crossings(c1, c2, n: int) -> int:
    """Number of crossings of the curves `c1 = (x, ℓ)`, `c2 = (y, m)` of the
    n-gon with one interior puncture, counted in the universal cover around
    the puncture: `c1` lifts to the interval `(x, x + ℓ)` of the boundary line,
    `c2` to `(y + t·n, y + t·n + m)`, `t ∈ Z`, and each strictly interleaving
    pair of lifts is one crossing (a shared endpoint is not a crossing)."""
    (x, l), (y, m) = c1, c2
    x %= n
    y %= n
    count = 0
    for t in range(-2, 3):
        yy = y + t * n
        if x < yy < x + l < yy + m or yy < x < yy + m < x + l:
            count += 1
    return count


# ---------------------------------------------------------------------------
# The cone data in curve letters
# ---------------------------------------------------------------------------


class _CurveConeData(FiniteConeData):
    """`a1dodd_cone_data(k)` with the generators renamed to curves `(x, ℓ)`.

    `q_commute` is the crossing count (`_arc_crossings == 0`); `cocycle` is
    translated; `cross_product` translates the CANONICAL-basis q-powers
    (`literal − cone_label_phase` in the `(a, p, i)` order) and re-adds the
    phase of this data's own order (sorted curves), since the literal phase
    depends on the order.  ρ is `x ↦ x + 1`."""

    def __init__(self, k: int, ap=None):
        """`ap`: the `a1dodd_cone_data(k)` instance to rename — `A1DnKAlg`
        passes its delegate's, so the two share one build; built here when
        omitted."""
        if ap is None:
            from a1dodd_cone_data import a1dodd_cone_data
            ap = a1dodd_cone_data(k)
        self.k = k
        self._n = 2 * k + 3
        self._ap = ap
        self._to_arc = {g: _ap_to_arc(g, k) for g in self._ap.mult_gens()}
        self._to_ap = {c: g for g, c in self._to_arc.items()}
        n = self._n
        if (len(self._to_ap) != len(self._to_arc)
                or set(self._to_ap) != {(x, l) for x in range(n)
                                        for l in range(2, n + 1)}):
            raise AssertionError(
                f"_CurveConeData(k={k}): the (a,p,i) ↦ (x,ℓ) dictionary is "
                f"not a bijection onto the curves")
        self._mult_gens = tuple(sorted(self._to_ap))
        self._cones = None
        self._xp_cache = {}

    def coefficient_ring(self):
        return self._ap.coefficient_ring()

    def mult_gens(self):
        return self._mult_gens

    def cones(self):
        if self._cones is None:
            self._cones = tuple(
                frozenset(self._to_arc[g] for g in C) for C in self._ap.cones())
        return self._cones

    def q_commute(self, g, h):
        return g == h or _arc_crossings(g, h, self._n) == 0

    def cocycle(self, g, h):
        if g == h:
            return 0
        return self._ap.cocycle(self._to_ap[g], self._to_ap[h])

    def cross_product(self, g, h):
        key = (g, h)
        if key in self._xp_cache:
            return self._xp_cache[key]
        R = self.coefficient_ring()
        out = []
        for coef, word_ap in self._ap.cross_product(self._to_ap[g],
                                                    self._to_ap[h]):
            if word_ap:
                ctr_ap = Counter(word_ap)
                ph_ap = self._ap.cone_label_phase(frozenset(ctr_ap), dict(ctr_ap))
                word = tuple(sorted(self._to_arc[f] for f in word_ap))
                ctr = Counter(word)
                ph = self.cone_label_phase(frozenset(ctr), dict(ctr))
            else:
                word, ph_ap, ph = (), 0, 0
            shift = ph - ph_ap
            out.append((RLaurent(R, {e + shift: r for e, r in coef.coeffs.items()}),
                        word))
        self._xp_cache[key] = out
        return out

    def to_cone_label(self, native_label):
        if not native_label:
            return (frozenset(), {})
        powers = {c: m for (c, m) in native_label}
        return (frozenset(powers), powers)

    def from_cone_label(self, gens, powers):
        return tuple(sorted(
            (c, powers[c]) for c in gens if powers.get(c, 0) > 0))

    def rho_label(self, c):
        x, ell = c
        return ((x + 1) % self._n, ell)

    def rho_inv_label(self, c):
        x, ell = c
        return ((x - 1) % self._n, ell)

    def rho_native(self, native_label):
        return tuple(sorted((self.rho_label(c), m) for (c, m) in native_label))

    def rho_inv_native(self, native_label):
        return tuple(sorted((self.rho_inv_label(c), m)
                            for (c, m) in native_label))

    def cycle_period_bound(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# A1DnKAlg
# ---------------------------------------------------------------------------


def _even_n_message(n: int) -> str:
    k = (n - 2) // 2
    msg = (
        f"A1DnKAlg({n}): [A_1, D_{n}] with n even has flavour SU(2)×U(1) "
        f"(SU(3) at n = 4); A1DnKAlg is the odd-n class.  Use "
        f"a1deven_kalg.A1DevenKAlg({k}) (SU(2)×U(1), obtained by ungauging "
        f"U1A1DevenConeKAlgebra({k}), the U(1)-gauged algebra on the curves "
        f"of the once-punctured {n}-gon)"
    )
    if n == 4:
        msg += (", finite_a1d4_kalg.FiniteA1D4KAlgebra or "
                "su3_ad_kalg.SU3ADKAlg (the SU(3)-flavoured presentation)")
    msg += (f"; the U(1)-gauged algebra is "
            f"u1a1deven_cone_kalgebra.U1A1DevenConeKAlgebra({k}).")
    return msg


class A1DnKAlg(ConeKAlgebra):
    """`[A_1, D_n]` (n odd ≥ 3, SU(2) flavour) on the curves of the n-gon
    with one interior puncture; see the module docstring.

    Labels `(curves, κ)`; builders `curve(x, ell, kappa=0)` and
    `chi(kappa=…)`.  Products, ρ and trace are those of
    `A1DoddConeKAlg((n − 3)/2)`, computed through the curve dictionary."""

    def __init__(self, n: int):
        if n < 3:
            raise ValueError(f"A1DnKAlg(n): n must be ≥ 3, got {n}")
        if n % 2 == 0:
            raise NotImplementedError(_even_n_message(n))
        from a1dodd_kalg import A1DoddConeKAlg
        self._n = n
        self.k = (n - 3) // 2
        self._R = SU2ZPlusRing()
        self._odd = A1DoddConeKAlg(self.k)
        gens = [(a, p, i) for a in range(1, self.k + 2) for p in (0, 1)
                for i in range(n)]
        self._to_arc = {g: _ap_to_arc(g, self.k) for g in gens}
        self._to_ap = {c: g for g, c in self._to_arc.items()}
        if len(self._to_ap) != len(gens) or any(
                _arc_to_ap(c, self.k) != g for g, c in self._to_arc.items()):
            raise AssertionError(
                f"A1DnKAlg({n}): the curve dictionary is not a bijection")
        self._cone_data_cache = None

    # ----- accessors ------------------------------------------------------

    @property
    def n(self) -> int:
        return self._n

    # ----- KAlgebra primitives --------------------------------------------

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return ((), 0)

    def cone_data(self):
        if self._cone_data_cache is None:
            self._cone_data_cache = _CurveConeData(self.k,
                                                   self._odd.cone_data())
        return self._cone_data_cache

    def canonicalise(self, label):
        """Canonical label: curves reduced (`x mod n`), merged and sorted,
        powers ≥ 1, κ ≥ 0.  Raises `ValueError` on a curve outside
        `2 ≤ ℓ ≤ n`, a negative power or κ, or two crossing curves."""
        curves, kappa = label
        n = self._n
        agg: dict = {}
        for c, m in curves:
            x, ell = c
            m = int(m)
            if not (2 <= int(ell) <= n):
                raise ValueError(f"A1DnKAlg({n}): curve {c!r} needs 2 ≤ ℓ ≤ {n}")
            if m < 0:
                raise ValueError(f"A1DnKAlg({n}): negative power in {label!r}")
            if m == 0:
                continue
            key = (int(x) % n, int(ell))
            agg[key] = agg.get(key, 0) + m
        kappa = int(kappa)
        if kappa < 0:
            raise ValueError(f"A1DnKAlg({n}): κ must be ≥ 0, got {kappa}")
        keys = sorted(agg)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if _arc_crossings(keys[i], keys[j], n):
                    raise ValueError(
                        f"A1DnKAlg({n}): curves {keys[i]} and {keys[j]} "
                        f"cross — not a canonical-basis label")
        return (tuple((c, agg[c]) for c in keys), kappa)

    def _to_odd(self, label):
        curves, kappa = self.canonicalise(label)
        return (tuple(sorted((self._to_ap[c], m) for c, m in curves)), kappa)

    def _from_odd(self, label):
        word, kappa = label
        return (tuple(sorted((self._to_arc[g], m) for g, m in word)), kappa)

    def multiply(self, a, b) -> Element:
        prod = self._odd.multiply(self._to_odd(a), self._to_odd(b))
        return Element({self._from_odd(l): c for l, c in prod.terms.items()})

    def rho(self, a):
        """ρ = the rotation `x ↦ x + 1` of every curve, κ fixed (order n)."""
        return self._from_odd(self._odd.rho(self._to_odd(a)))

    def rho_inverse(self, a):
        return self._from_odd(self._odd.rho_inverse(self._to_odd(a)))

    def trace(self, a, K: int = 20) -> RPowerSeries:
        """SU(2)-refined Schur index `Tr(L_a)` (`A1DoddConeKAlg.trace`)."""
        return self._odd.trace(self._to_odd(a), K)

    def _trace_residual(self, seed_label, K: int) -> RPowerSeries:
        """Layer-2 trace of a χ-stripped seed word in curve letters (the
        identity, or one curve): `A1DoddConeKAlg._trace_residual`."""
        word = tuple(sorted((self._to_ap[c], m) for c, m in seed_label))
        return self._odd._trace_residual(word, K)

    # ----- flavour-lift coordinate ----------------------------------------

    def r_label_decompose(self, label):
        """`L_{(curves, κ)} = χ_κ · L_{(curves, 0)}`: the single-irrep lift
        coordinate `((curves, 0), κ)`."""
        curves, kappa = self.canonicalise(label)
        return (curves, 0), kappa

    def r_label_compose(self, section, r_basis_label):
        curves, _zero = section
        return self.canonicalise((curves, r_basis_label))

    # ----- builders --------------------------------------------------------

    def curve(self, x, ell, kappa: int = 0):
        """The label of the single curve `(x, ℓ)` (from marked point `x` to
        `x + ℓ`, ℓ boundary edges on the side away from the puncture; ℓ = n
        the loop around the puncture) at SU(2) highest weight `kappa`."""
        return self.canonicalise(((((x, ell), 1),), kappa))

    def chi(self, *, kappa: int):
        """The central SU(2) character `χ_κ` (highest weight κ), label
        `((), κ)`.  `χ₁` is the peripheral loop around the puncture; `χ_κ` is
        not the κ-th power of the loop (`χ₁² = χ₀ + χ₂`).  Keyword-only: the
        pre-2026-09-23 `chi(j)` took the spin j."""
        if int(kappa) < 0:
            raise ValueError(f"chi: kappa must be ≥ 0, got {kappa}")
        return ((), int(kappa))

    def L(self, label) -> Element:
        return Element.basis(self.canonicalise(label))

    def __repr__(self) -> str:
        return (f"A1DnKAlg(n={self._n})  # [A_1, D_{self._n}], curves of the "
                f"{self._n}-gon with one interior puncture")
