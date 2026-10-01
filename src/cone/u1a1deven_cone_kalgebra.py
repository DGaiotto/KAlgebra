"""
u1a1deven_cone_kalgebra.py
==========================

`U1A1DevenConeKAlgebra(k)` — the **U(1)-gauged** `[A_1, D_{2k+2}]` as a
`ConeKAlgebra` over `R(SU(2))`, with canonical-basis labels that are curves on
the `(2k+2)`-gon with one interior puncture, plus a power of the gauge letter
`E = X_{0,1}` and an `SU(2)` weight.  The D-even analogue of the gauged-A
`U1A1AoddKAlg`, and the U(1)-gauged partner of `A1DnKAlg` (odd `n`).

Since 2026-09-24 this class is the curve frame, in the curve convention of
`A1DnKAlg`.  Until then the class of this name was a ray-keyed table
presentation built on an RG flow not included in this repository (earlier
releases shipped its `k = 1` tables, `u1a1deven_tables_k1.pkl` /
`u1a1deven_traces_k1.pkl`); it is retired, and stays the regression
baseline of this class in the source repository.  The
combinatorics, the closed forms and the product rule live in the
implementation module `u1a1deven_geometric_frame.py` (sections A, B, D of its
docstring); this module is its section C.

Labels
------
A label is `(curves, e, κ)`:

  * `curves` — a sorted tuple of `((x, ℓ), m)`, `m ≥ 1`, the curves pairwise
    non-crossing.  A curve `(x, ℓ)`, `x ∈ Z/(2k+2)`, `2 ≤ ℓ ≤ 2k+2`, is the
    simple curve from marked point `x` to `x + ℓ` with `ℓ` boundary edges on
    the side not containing the puncture; `ℓ = 2k + 2` is the loop at `x`
    around the puncture (`A1DnKAlg`'s curves);
  * `e ∈ Z` — the power of `E = X_{0,1}`, which `𝖖`-commutes with every
    curve (a quantum-torus direction; ungauging it gives `A1DevenKAlg`);
  * `κ ≥ 0` — the `SU(2)` highest weight: `L_{(curves, e, κ)} = χ_κ ·
    L_{(curves, e, 0)}`, as in `A1DnKAlg`'s `(curves, κ)`.

The identity is `((), 0, 0)`; `curve(x, ell, e=0, kappa=0)` is the label of
one curve.  The flavour is in the label: `multiply` returns an `Element` with
integral `LaurentPoly` coefficients on full labels — the Z-form of
the `KAlgebra` contract (`kalgebra.py`) — and `to_R_form` / `from_R_form` regroup it
through the lift coordinate `r_label_decompose((curves, e, κ)) = ((curves, e,
0), κ)`.  (Until 2026-09-24 this class and `A1DevenKAlg` returned `RLaurent`
coefficients inside a Z-form `Element`, so `to_R_form` raised.)

Every curve has a magnetic charge `c0 ∈ {−1, 0, +1}` (`_charge`: 0 on odd `ℓ`;
on even `ℓ`, `−1` if its endpoints have the parity of marked point 2, `+1` if
not); a label's charge is `Σ m·c0`, and the trace vanishes unless it is 0.
`E` has charge 0 (it is the electric direction).

The position `e` is measured from (the public convention)
--------------------------------------------------------
`e` needs a reference element per curve: which element of the line
`{E^j·L_{(x,ℓ)}}` is called `e = 0`.  This class takes it from the RG flow it
was built from, `U1A1DevenViaDoddRG(k)`, whose image of a curve (`Φ`, frame
docstring B) deletes marked point 2 of the `(2k+2)`-gon and joins its two
boundary edges into one; `e = 0` is the element whose image has lowest term at
`E`-power 0 with coefficient 1.  So the boundary edge from marked point 1 to
marked point 2 is special, in three rules:

  (a) multiplication: in a skein resolution a piece of the boundary is 1,
      except along the edge {1, 2}, where it is `E`;
  (b) ρ rotates `(x, ℓ) ↦ (x + 1, ℓ)` and `E ↦ E⁻¹`, with one extra `E⁻¹`
      per endpoint of the curve at marked point 1:
      `ρ((x, ℓ)) = E^d·(x + 1, ℓ)`, `d = −#{endpoints at 1}`;
  (c) a curve with even `ℓ` has magnetic charge `−1` if its endpoints share
      marked point 2's parity, `+1` if not.

No labelling avoids naming a position.  A full turn returns every curve to
itself up to a power of `E`: `ρ^{2k+2}(L_{(x,ℓ)}) = E^{2c0}·L_{(x,ℓ)}` (`E^{±2}`
on even-`ℓ` curves, `E^0` on odd-`ℓ` ones), and that shift does not depend on
where `e = 0` is put (re-normalising `L_c ↦ E^{f(c)}·L_c` conjugates ρ, and
`ρ^{2k+2}` is unchanged because `2k + 2` is even), so `e` must jump somewhere
on the way round.  A rule treating every position alike — `ρ((x, ℓ)) =
E^{g(c0)}·(x + 1, ℓ)` with `g` depending only on the charge, which alternates
along the turn on even `ℓ` — gives `(k + 1)·(g(−1) − g(+1))` for the full turn
of a `+1` curve, so it would need `g(+1) − g(−1) = −2/(k + 1)`: an integer
only at k = 1.  (Measured 2026-09-24 at k = 1..5, with `ρ∘ρ⁻¹ = id` as the
control; the source repository's test `test_full_turn_shift` pins
the full turn, its independence of the normalisation and the failure of every
position-blind rule at k = 1..4.)  Precedent: `U1A1AoddKAlg`, whose flow
collapses the edge `(H − 1, 0)` and whose ρ concentrates the `E`-drift at one
position per curve type.  Status: accepted tentatively; the convention
may be revised in a later release.

Products and ρ
--------------
`multiply` strips `κ`, runs the generic cone-monomial reducer over the
frame's `_CurveConeData` on `(curves, e)` (cocycle in closed form; cross
products of two crossing curves by the analytic skein rule of frame docstring
D, the derived peel of `Φ(g)·Φ(h)` wherever the analytic rule returns `None`
— on no crossing pair at k = 1..5), fuses the `SU(2)` weights by
Clebsch–Gordan and writes them into the labels (the `A1DoddConeKAlg`
pattern: `cone_data()` is the `χ`-stripped data).  `route="derived"` takes
every cross product from the derived peel (the independent route the analytic
rule was certified against).  `ρ` is the closed form (b), `κ` fixed; its
inverse likewise.  Neither needs the flow.

Trace and pairing
-----------------
`trace(a, K)`, in order:

  1. exactly 0 if the magnetic charge is nonzero;
  2. `Tr(E^e)` from Creutzig's closed form (`exact_characters.
     deven_gauged_xn_qn`, the `xᵉ` slice of the `(A_1, D_{2p})` index,
     `p = k + 1`; arbitrary `𝖖`);
  3. a SEED — one curve of odd length, or a non-crossing pair of a charge +1
     and a charge -1 curve, times `E^e` — from its closed form
     (`u1a1deven_seed_characters.seed_trace`, any order).  These are the
     traces every `A1DevenKAlg(k)` generator (so every zoo a1d6 / a1d8 seed)
     reduces to.  The closed forms are MEASURED, not derived, against the
     exact transport (the evidence is in `u1a1deven_seed_characters`'
     docstring);
  4. every other label (since 2026-09-24): the generic Layer-1 reduction of
     the cone data (`ConeData.simplify_trace_via_cone_data` — `ρ²`-twisted
     cyclicity and the product rule, run on the `χ`-stripped labels through
     `_ChiStrippedView`) onto trace leaves, each traced by steps 1–3.  The
     reduction depends only on the label and is memoised; its coefficients
     reach negative `𝖖`-powers, so the leaves are traced correspondingly
     deeper.  In every measurement the leaves are single odd curves and `E^e`
     — it takes the pairs apart too — and a leaf that were not a seed would be
     traced by the transport (counted in `_layer1_stats`).

`χ_κ` is folded in last.  `inner_product(a, b, K)` is multiply-then-trace:
this class's product of `ρ(a)` and `b`, every term traced as above.

The transport route — `seed_closed_forms=False` in the constructor — is the
independent witness: every trace of a label with curves is the exact
transport `u1a1deven_trace_transport.DevenTraceTransport.trace_aux(Φ(a), K)`
of the closed-form RG image `Φ(a)` (frame docstring B; it never calls the
flow's `RG`), memoised per process and k, and `inner_product` is the same
transport on `Φ(ρ(a))·Φ(b)`, a product in the auxiliary algebra, so it uses
no product of this class.  Its stopping rule is a measured hypothesis with a
guard, and its limit on the length of an `A1Dodd` word (`_MAX_WORD_DEGREE`,
per k) raises `ValueError` rather than truncate (that module's docstring).
Measured 2026-09-24 against it: composite labels (20 / 15 / 15 at k = 1 / 2 /
3 through `𝖖¹⁶` / `𝖖¹⁰` / `𝖖⁶`), pairings, a ρ without its `E`-drift handed
to the reducer as the negative control (it disagrees); and the pair seeds by
Layer 1 against their own closed forms through `𝖖⁴⁰`.

Bridges to the flow
-------------------
`Φ` is the RG image of `U1A1DevenViaDoddRG(k)`, so a label names the same
canonical element as the flow label of its lowest term,
`((word, κ), (c0, e))` (closed form).  `oracle_section_of(label)` returns it
in the section layout of the retired tables, `((word, (c0, e)), κ)`, and
`native_of_oracle_section(section)` inverts it (a search over the at most two
curves with a given lowest term);
`u1a1deven_cone_dodd_section_iso(k)` is the `KAlgebraIso` onto the flow built
on that bijection of labels.

Runtime dependencies
--------------------
Products and ρ: the frame module and the `A1Dodd` arc rules
(`a1dodd_cone_data`, `a1dodd_skein`) only.  Traces and the pairing: the
closed forms (`u1a1deven_seed_characters`, `exact_characters`) and the cone
data's Layer-1 reducer (`cone_data`), pure Python — neither the transport nor
any RG module is imported (the source repository's test
`test_no_rg_flow_on_the_serving_path`, in a fresh process).  The transport
route (the witness) builds the flow's auxiliary algebra
`A1DoddConeKAlg(k − 1) ⊗ QT(Z²)` and the closed-form coefficients of its
`S_RG` itself, and its `A1Dodd` traces are `A1DoddConeKAlg.trace` (closed
forms): no RG module on it either — not the flow `U1A1DevenViaDoddRG`, not
`rgkalgebra` / `graded_rg_solver` (the same test with them blocked, and the
source repository's test `test_trace_calls_no_rg`, with the flow's `RG`
disabled).  The flow stays the independent witness of the tests; the older
flow `U1A1DevenRGKAlgebra` is on no runtime path.
"""
from __future__ import annotations

import itertools
import os
import sys

# APPEND, never insert(0) (the hazard recorded in u1a1aodd_kalg.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

from kalgebra import Element
from laurent_poly import LaurentPoly
from zplus_ring import ZPlusRing, SU2ZPlusRing, RLaurent, RPowerSeries
from cone_kalgebra import ConeKAlgebra
from a1dn_kalg import _arc_to_ap, _ap_to_arc
import u1a1deven_geometric_frame as _frame
from u1a1deven_geometric_frame import (
    _curves, _curves_cross, _rotate_curve, _charge, _phi_curve, _lowest_term,
    _rho_curve, _compositions, _su2_fuse, _CurveConeData, _ONE)
# `_resolutions` and `_analytic_product` are called through the module object
# (`_frame._resolutions`, …), so that negative controls which replace them
# there reach this class too.


# The transport's output `{𝖖-power: {SU(2) weight: int}}` per `(k, (curves,
# e))`, with the depth it was computed to: shared by every instance in the
# process (`U1A1DevenConeKAlgebra._transport_trace`).
_TRACE_MEMO: dict = {}

# At most this many labels keep their Layer-1 reduction per instance
# (`U1A1DevenConeKAlgebra._layer1_reduction`); the memo restarts beyond it.  The
# cone data keeps its own word memo, capped likewise (`cone_data.py`).
_L1_MEMO_CAP = 200_000


class _ChiStrippedView:
    """`U1A1DevenConeKAlgebra` on the `χ`-stripped labels `(curves, e)` of its
    cone data: what the generic Layer-1 reducer
    (`ConeData.simplify_trace_via_cone_data`) asks of the algebra it is handed
    — ρ, ρ⁻¹ (the class's closed forms, `κ` set to 0 and dropped) and the
    `ρ²`-orbit representative — and the object that holds its word memo."""

    def __init__(self, alg):
        self._alg = alg

    def rho(self, label2):
        curves, e, _ = self._alg.rho((label2[0], label2[1], 0))
        return (curves, e)

    def rho_inverse(self, label2):
        curves, e, _ = self._alg.rho_inverse((label2[0], label2[1], 0))
        return (curves, e)

    def rho_squared_is_identity(self) -> bool:
        return False

    def _canonical_rho2_orbit_rep(self, label2):
        curves, e, _ = self._alg._canonical_rho2_orbit_rep((label2[0], label2[1], 0))
        return (curves, e)


class U1A1DevenConeKAlgebra(ConeKAlgebra):
    """The U(1)-gauged `[A_1, D_{2k+2}]` on the curves of the once-punctured
    `(2k+2)`-gon: labels `(curves, e, κ)` (`e` the power of `E = X_{0,1}`,
    measured from the edge {1, 2} convention; `κ` the `SU(2)` weight), Z-form
    products, closed-form ρ, flow-free traces.  See the module docstring.

    `route`: `"analytic"` (default) — cross products from the analytic skein
    rule, the derived peel where it does not apply; `"derived"` — from the
    derived peel throughout.  `seed_closed_forms`: True (default) — the
    closed-form route: the seed traces from `u1a1deven_seed_characters`, every
    other label by the Layer-1 reduction onto them, the pairing by
    multiply-then-trace; False — the transport route for every trace with
    curves and for the pairing (the tests' witness)."""

    def __init__(self, k: int, route: str = "analytic",
                 seed_closed_forms: bool = True):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if route not in ("analytic", "derived"):
            raise ValueError(f"route must be 'analytic' or 'derived', got {route!r}")
        self.k = k
        # steps 3 and 4 of `trace` and the pairing (module docstring): the
        # closed-form route; False — the transport route (the tests' witness)
        self._seed_closed_forms = bool(seed_closed_forms)
        self._view_ = None                    # _ChiStrippedView, built lazily
        self._l1_memo: dict = {}              # (curves, e) -> Layer-1 Element
        self._leaf_cache: dict = {}           # (curves, e) -> (K, RPowerSeries)
        # leaves of the Layer-1 reduction that were not seeds (traced by the
        # transport instead); none in any measurement so far
        self._layer1_stats = {"transport_leaves": 0}
        self.n = 2 * k + 2
        self._route = route
        self._R = SU2ZPlusRing()
        self._curve_set = frozenset(_curves(self.n))
        # the terms of lowest E-power of the curves' images (frame docstring B)
        self._low = {c: _lowest_term(c, k) for c in self._curve_set}
        self._phi_cache: dict = {}
        self._gauge_cache: dict = {}          # e -> (K, RPowerSeries)
        self._transport_ = None
        self._cone_data_ = None
        # crossing pairs the analytic rule did not cover (served by the
        # derived peel instead); empty at k = 1..5 (frame docstring D)
        self._analytic_not_applied: set = set()

    # ----- contract primitives -----------------------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return ((), 0, 0)

    def cone_data(self):
        """The `χ`-stripped cone data (`u1a1deven_geometric_frame.
        _CurveConeData`): native labels `(curves, e)`, `χ_κ` in `RLaurent`
        coefficients (the `A1DoddConeKAlg` pattern; `multiply` moves `χ_κ`
        into the label)."""
        if self._cone_data_ is None:
            self._cone_data_ = _CurveConeData(self)
        return self._cone_data_

    def canonicalise(self, label):
        """The canonical label `(curves, e, κ)`: curves reduced (`x mod n`),
        merged and sorted, powers ≥ 1, `e` an int, `κ ≥ 0`.  `ValueError` on a
        label that is not a triple, a curve outside `2 ≤ ℓ ≤ n`, a negative
        power or `κ`, or two crossing curves."""
        try:
            curves, e, kappa = label
        except (TypeError, ValueError):
            raise ValueError(
                f"U1A1DevenConeKAlgebra({self.k}): a label is (curves, e, kappa), "
                f"got {label!r}") from None
        curves, e = self._canon2((curves, e))
        kappa = int(kappa)
        if kappa < 0:
            raise ValueError(f"U1A1DevenConeKAlgebra({self.k}): κ must be ≥ 0, "
                             f"got {kappa}")
        return (curves, e, kappa)

    def _canon2(self, label2):
        """`canonicalise` on the `χ`-stripped `(curves, e)` (the cone data's
        native labels)."""
        curves, e = label2
        n = self.n
        agg: dict = {}
        for c, m in curves:
            x, l = c
            if not (2 <= int(l) <= n):
                raise ValueError(f"U1A1DevenConeKAlgebra({self.k}): curve {c!r} "
                                 f"needs 2 ≤ ℓ ≤ {n}")
            m = int(m)
            if m < 0:
                raise ValueError(f"U1A1DevenConeKAlgebra({self.k}): negative "
                                 f"power in {label2!r}")
            if m:
                key = (int(x) % n, int(l))
                agg[key] = agg.get(key, 0) + m
        keys = sorted(agg)
        for a, b in itertools.combinations(keys, 2):
            if _curves_cross(a, b, n):
                raise ValueError(f"U1A1DevenConeKAlgebra({self.k}): curves {a} "
                                 f"and {b} cross — not a canonical-basis label")
        return (tuple((c, agg[c]) for c in keys), int(e))

    def multiply(self, a, b) -> Element:
        """`L_a·L_b` (Z-form): the cone-monomial reducer over the
        `χ`-stripped `_CurveConeData`, then the `SU(2)` weights of `a`, `b`
        and of each term fused by Clebsch–Gordan into the labels."""
        a, b = self.canonicalise(a), self.canonicalise(b)
        prod = self.cone_data().derived_multiply(a[:2], b[:2])
        return self._fuse_chi(prod, a[2], b[2])

    def _fuse_chi(self, prod, ka: int, kb: int) -> Element:
        """An element over `(curves, e)` with `RLaurent` (or `LaurentPoly`)
        coefficients, times `χ_ka·χ_kb`, as a Z-form element over
        `(curves, e, κ)`."""
        R = self._R
        chi = R.basis_element(ka) * R.basis_element(kb)
        out: dict = {}
        for (curves, e), co in prod.terms.items():
            rl = co if isinstance(co, RLaurent) else RLaurent(R, dict(co._coeffs))
            for q, r in rl.coeffs.items():
                for kap, c in (r * chi).terms.items():
                    if c:
                        key = (curves, e, kap)
                        out[key] = out.get(key, LaurentPoly({})) + LaurentPoly({q: int(c)})
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    def rho(self, a):
        """`ρ((curves, e, κ)) = (rotated curves, −e + Σ m·d, κ)`, `d` the
        drift of rule (b) (module docstring)."""
        curves, e, kappa = self.canonicalise(a)
        new, drift = [], 0
        for c, m in curves:
            c2, d = _rho_curve(c, self.k)
            new.append((c2, m))
            drift += m * d
        return (tuple(sorted(new)), -e + drift, kappa)

    def rho_inverse(self, a):
        curves, e, kappa = self.canonicalise(a)
        new, drift = [], 0
        for c, m in curves:
            c1 = _rotate_curve(c, self.n, -1)
            _, d = _rho_curve(c1, self.k)
            new.append((c1, m))
            drift += m * d
        return (tuple(sorted(new)), -e + drift, kappa)

    def _canonical_rho2_orbit_rep(self, label):
        """Closed-form representative of the ρ²-orbit (ρ² shears the `E`
        power, so orbits are infinite): walk ρ² until the curve multiset
        recurs (at most `k + 1` steps), take the smallest multiset of the
        period and its `E`-power, reduced mod the drift `D` of one period
        when `D ≠ 0`.  `E^e` is ρ²-fixed.  Used by `U1A1DevenSkeinKAlgebra`
        and, through `_ChiStrippedView`, by the Layer-1 reduction of this
        class's traces (to fold the leaves)."""
        curves, e, kappa = self.canonicalise(label)
        if not curves:
            return ((), e, kappa)
        seq = [(curves, e)]
        cur = (curves, e, kappa)
        while True:
            cur = self.rho(self.rho(cur))
            if cur[0] == curves:
                break
            seq.append(cur[:2])
            if len(seq) > self.n:
                raise AssertionError(f"_canonical_rho2_orbit_rep({label!r}): "
                                     f"no recurrence within {self.n} steps")
        D = cur[1] - e
        kmin = min(s[0] for s in seq)
        (emin,) = [s[1] for s in seq if s[0] == kmin]
        return (kmin, emin if D == 0 else emin % abs(D), kappa)

    # ----- flavour-lift coordinate -------------------------------------------

    def r_label_decompose(self, label):
        """`L_{(curves, e, κ)} = χ_κ · L_{(curves, e, 0)}`: the single-irrep
        lift coordinate `((curves, e, 0), κ)`."""
        curves, e, kappa = self.canonicalise(label)
        return (curves, e, 0), kappa

    def r_label_compose(self, section, r_basis_label):
        curves, e, _zero = section
        return self.canonicalise((curves, e, r_basis_label))

    # ----- builder ------------------------------------------------------------

    def curve(self, x, ell, e: int = 0, kappa: int = 0):
        """The label of the single curve `(x, ℓ)` (from marked point `x` to
        `x + ℓ`, `ℓ` boundary edges on the side away from the puncture;
        `ℓ = 2k + 2` the loop around the puncture) times `E^e`, at `SU(2)`
        highest weight `kappa` — `A1DnKAlg.curve(x, ell, kappa)` with the
        power of `E` beside it."""
        return self.canonicalise(((((x, ell), 1),), e, kappa))

    def geometric_label(self, g):
        """The geometric label of a letter of the cone data (the hook
        `UngaugedKAlgebra.geometric_label` reads, as for `U1A1AoddKAlg`): a
        curve letter `(x, ℓ)` is its own geometric label, the curve of the
        once-punctured `(2k+2)`-gon; `None` for the gauge letters `E^{±1}`.
        `ValueError` on anything else."""
        g = tuple(g)
        if g in _frame._E_LETTERS:
            return None
        if len(g) == 2 and 2 <= int(g[1]) <= self.n and 0 <= int(g[0]) < self.n:
            return (int(g[0]), int(g[1]))
        raise ValueError(f"U1A1DevenConeKAlgebra({self.k}).geometric_label: "
                         f"{g!r} is not a letter (a curve (x, ℓ) or E^±1)")

    # ----- the magnetic charge, and the hook UngaugedKAlgebra reads ------------

    def _magnetic_charge(self, label) -> int:
        """`Σ m·c0` over the curves of a label (2- or 3-slot)."""
        return sum(m * _charge(c, self.k) for c, m in label[0])

    def _label_mag(self, x, E):
        """For `UngaugedKAlgebra.in_centralizer`: the charge `mag(x)` with
        `E·L_x = 𝖖^{mag(x)}·L_x·E` read off the curves, `−2·Σ m·c0`
        (checked against the `E`-commutator in the tests), when `E` is
        `((), 1, 0)`; `None` otherwise."""
        try:
            if self.canonicalise(E) != ((), 1, 0):
                return None
            x = self.canonicalise(x)
        except ValueError:
            return None
        return -2 * self._magnetic_charge(x)

    # ----- products of two crossing curves, and the peel -----------------------

    def _cross(self, g, h):
        """`L_g·L_h` for two crossing curves as `{(curves, e): {κ:
        LaurentPoly}}` (canonical-basis coefficients, `χ_κ` by κ): the
        analytic rule where it applies (route `"analytic"`), else the derived
        peel."""
        if self._route == "analytic":
            got = _frame._analytic_product(g, h, self.k)
            if got is not None:
                return got
            self._analytic_not_applied.add((g, h))
        return self._cross_derived(g, h)

    def _cross_derived(self, g, h):
        """The derived route: `Φ(g)·Φ(h)` in the auxiliary algebra, peeled with
        the skein resolutions of `g, h` (`_resolutions`) as the candidates.
        `ValueError` if the resolution does not fold into labels or the peel
        fails (frame docstring D)."""
        kind, states = _frame._resolutions(g, h, self.k)
        if kind is None:
            raise ValueError(f"U1A1DevenConeKAlgebra({self.k})._cross_derived({g}, "
                             f"{h}): a skein resolution does not fold into a "
                             f"label, so there are no candidates")
        X = self._aux().multiply_elements(self._phi2((((g, 1),), 0)),
                                          self._phi2((((h, 1),), 0)))
        return self._peel(X, {curves for curves, _e, _chi, _nb in states})

    def _peel(self, X, candidates=None):
        """Peel the auxiliary element `X` into this algebra's canonical
        elements: `{(curves, e): {κ: LaurentPoly}}` with `X = Σ c·χ_κ·Φ(label)`.

        Repeatedly: a term of lowest `E`-power `c1` of the residual,
        `((word, κ), (c0, c1))` with coefficient `c`, is the lowest term of
        `c·χ_κ·Φ(label)` for exactly one label — `(curves, c1)` with `curves`
        the candidate whose lowest terms are `(word, c0)` (`candidates`: a set
        of curve tuples, the skein resolutions), or, with `candidates=None`,
        the label the inverse label map names; that multiple of `Φ(label)` is
        subtracted.  Exact: the result is returned only when the residual is
        zero.  `ValueError` if a lowest term names no candidate (or no label),
        or if the residual reaches above the top `E`-power of `X` (which a
        decomposition with positive coefficients cannot need)."""
        zero = LaurentPoly({})
        res = {l: c for l, c in X.terms.items() if not c.is_zero()}
        if not res:
            return {}
        lookup = None
        if candidates is not None:
            lookup = {}
            for curves in candidates:
                (word, _kap), (c0, _e) = self._flow_label((curves, 0, 0))
                if lookup.get((word, c0), curves) != curves:
                    raise ValueError(f"U1A1DevenConeKAlgebra({self.k})._peel: the "
                                     f"candidates {curves} and {lookup[(word, c0)]} "
                                     f"share their lowest term")
                lookup[(word, c0)] = curves
        top = max(l[1][1] for l in res)
        out: dict = {}
        while res:
            cmin = min(l[1][1] for l in res)
            if cmin > top:
                raise ValueError(f"U1A1DevenConeKAlgebra({self.k})._peel: the "
                                 f"residual reaches E-power {cmin} above the top "
                                 f"{top} of the product — no exact decomposition")
            for l in [l for l in res if l[1][1] == cmin]:
                c = res[l]
                (word, kap), (c0, c1) = l
                if lookup is not None:
                    curves = lookup.get((word, c0))
                    if curves is None:
                        raise ValueError(
                            f"U1A1DevenConeKAlgebra({self.k})._peel: the term "
                            f"{l!r} is not the lowest term of a skein resolution "
                            f"— no exact decomposition over the candidates")
                    lab = (curves, c1)
                else:
                    lab = self._label_of_flow_label(((word, 0), (c0, c1)))[:2]
                for l2, c2 in self._phi2(lab).terms.items():
                    (w2, k2), qt2 = l2
                    for kk in _su2_fuse(kap, k2):
                        key = ((w2, kk), qt2)
                        v = res.get(key, zero) - c2 * c
                        if v.is_zero():
                            res.pop(key, None)
                        else:
                            res[key] = v
                byk = out.setdefault(lab, {})
                byk[kap] = byk.get(kap, zero) + c
        return {lab: {kap: c for kap, c in byk.items() if not c.is_zero()}
                for lab, byk in out.items()
                if any(not c.is_zero() for c in byk.values())}

    def _z_form(self, dec, ka: int = 0, kb: int = 0) -> Element:
        """`{(curves, e): {κ: LaurentPoly}}` times `χ_ka·χ_kb`, as a Z-form
        element over `(curves, e, κ)`."""
        out: dict = {}
        for (curves, e), byk in dec.items():
            for kap, lp in byk.items():
                for k1 in _su2_fuse(ka, kb):
                    for k2 in _su2_fuse(k1, kap):
                        key = (curves, e, k2)
                        out[key] = out.get(key, LaurentPoly({})) + lp
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    def _multiply_via_phi(self, a, b) -> Element:
        """`L_a·L_b` by the derived peel of `Φ(a)·Φ(b)`, with no reducer — the
        independent cross-check of `multiply` on composite labels (Z-form).
        For two single crossing curves (any `E`-powers) the candidates are
        their skein resolutions; otherwise the lowest terms name their
        labels."""
        a, b = self.canonicalise(a), self.canonicalise(b)
        X = self._aux().multiply_elements(self._phi2(a[:2]), self._phi2(b[:2]))
        cands = None
        ca, cb = a[0], b[0]
        if (len(ca) == 1 and len(cb) == 1 and ca[0][1] == 1 and cb[0][1] == 1
                and _curves_cross(ca[0][0], cb[0][0], self.n)):
            kind, states = _frame._resolutions(ca[0][0], cb[0][0], self.k)
            if kind is None:
                raise ValueError(f"U1A1DevenConeKAlgebra({self.k})._multiply_via_phi: "
                                 f"a skein resolution of {ca[0][0]}, {cb[0][0]} "
                                 f"does not fold into a label")
            cands = {curves for curves, _e, _chi, _nb in states}
        return self._z_form(self._peel(X, cands), a[2], b[2])

    # ----- the closed-form image and the label map to the flow -------------------

    @property
    def _transport(self):
        if self._transport_ is None:
            from u1a1deven_trace_transport import _shared_transport
            self._transport_ = _shared_transport(self.k)
        return self._transport_

    def _aux(self):
        """The flow's auxiliary algebra `A1DoddConeKAlg(k − 1) ⊗ QT(Z²)`
        (Z-form labels `((word, κ), (c0, c1))`), from the transport."""
        return self._transport.aux

    def _phi2(self, label2) -> Element:
        """`Φ(L_{(curves, e, 0)})` in the auxiliary algebra: the product of the
        `Φ(curve)^m` (sorted order) and `X_{(0, e)}`, times `𝖖^{−T}`, `T` the
        `𝖖`-exponent of the product's lowest-`E` term (so that term is the flow
        label with coefficient 1).  `ValueError` if that term is not a single
        label with coefficient a power of `𝖖` (it always is on a canonical
        label)."""
        label2 = self._canon2(label2)
        hit = self._phi_cache.get(label2)
        if hit is not None:
            return hit
        curves, e = label2
        aux = self._aux()
        out = Element({(((), 0), (0, e)): _ONE})
        for c, m in reversed(curves):
            pc = Element(dict(_phi_curve(c, self.k)))
            for _ in range(m):
                out = aux.multiply_elements(pc, out)
        terms = {l: co for l, co in out.terms.items() if not co.is_zero()}
        cmin = min(l[1][1] for l in terms)
        low = [(l, co) for l, co in terms.items() if l[1][1] == cmin]
        co = low[0][1] if len(low) == 1 else None
        cd = dict(getattr(co, "_coeffs", {})) if co is not None else {}
        if len(low) != 1 or len(cd) != 1 or list(cd.values())[0] != 1:
            raise ValueError(f"U1A1DevenConeKAlgebra({self.k})._phi2({label2!r}): "
                             f"the lowest-E part {low!r} is not one label with "
                             f"coefficient a power of q")
        T = next(iter(cd))
        res = Element({l: co * LaurentPoly({-T: 1}) for l, co in terms.items()})
        self._phi_cache[label2] = res
        return res

    def _phi(self, label) -> Element:
        """`Φ(L_label)` for a label `(curves, e, κ)`: `χ_κ·Φ(L_{(curves, e,
        0)})` — the flow's `RG` image of the label's flow label."""
        curves, e, kappa = self.canonicalise(label)
        base = self._phi2((curves, e))
        if not kappa:
            return base
        return self._aux().multiply_elements(
            Element({(((), kappa), (0, 0)): _ONE}), base)

    def _flow_label(self, label):
        """The flow label of `L_label` (closed form): the lowest-`E` term of
        `Φ(label)`, `((word, κ), (Σ m·c0, e))`, `word` the `A1Dodd` word of the
        curves' lowest terms.  Accepts `(curves, e, κ)` or `(curves, e)`."""
        if len(label) == 2:
            (curves, e), kappa = self._canon2(label), 0
        else:
            curves, e, kappa = self.canonicalise(label)
        word: dict = {}
        c0 = 0
        for c, m in curves:
            d, q0 = self._low[c]
            c0 += m * q0
            if d is not None:
                g = _arc_to_ap(d, self.k - 1)
                word[g] = word.get(g, 0) + m
        return ((tuple(sorted(word.items())), kappa), (c0, e))

    def _label_of_flow_label(self, flow_label):
        """Inverse of `_flow_label` (a search, not a closed form: a curve of the
        `(2k+1)`-gon is the lowest term of up to two curves, and the two curves
        with an empty lowest term carry charges `±1`).  `ValueError` unless
        exactly one canonical label matches."""
        (word, kappa), (c0, e) = flow_label
        k = self.k
        need = {}
        for g, m in word:
            need[_ap_to_arc(g, k - 1)] = need.get(_ap_to_arc(g, k - 1), 0) + m
        pre: dict = {}
        for c, (d, q0) in self._low.items():
            pre.setdefault(d, []).append(c)
        slots = sorted(need)
        sols = []

        def rec(i, chosen):
            if i == len(slots):
                have = sum(m * _charge(c, k) for c, m in chosen)
                rest = c0 - have
                empties = sorted(pre.get(None, ()))
                opts = [()] if rest == 0 else [((c, abs(rest)),) for c in empties
                                                if _charge(c, k) * rest > 0]
                for extra in opts:
                    lab = tuple(sorted(chosen + list(extra)))
                    cs = [c for c, _ in lab]
                    if any(_curves_cross(a, b, self.n)
                           for a, b in itertools.combinations(cs, 2)):
                        continue
                    sols.append((lab, e, int(kappa)))
                return
            d = slots[i]
            cands = sorted(pre.get(d, ()))
            m = need[d]
            for split in _compositions(m, len(cands)):
                part = [(c, s) for c, s in zip(cands, split) if s]
                rec(i + 1, chosen + part)

        rec(0, [])
        if len(sols) != 1:
            raise ValueError(f"U1A1DevenConeKAlgebra({k}): {len(sols)} canonical "
                             f"labels have the flow label {flow_label!r}")
        return sols[0]

    def oracle_section_of(self, label):
        """The flow's label of `L_label` in the section layout of the retired
        tables, `((word, (c0, e)), κ)` — i.e. the flow label
        `((word, κ), (c0, e))` of `U1A1DevenViaDoddRG(k)` (module docstring,
        "Bridges to the flow")."""
        (word, kappa), (c0, e) = self._flow_label(label)
        return ((word, (c0, e)), kappa)

    def native_of_oracle_section(self, section):
        """Inverse of `oracle_section_of`: the label `(curves, e, κ)` of the
        section `((word, (c0, c1)), κ)`."""
        (word, (c0, c1)), kappa = section
        return self._label_of_flow_label(((word, kappa), (c0, c1)))

    # ----- trace and pairing (no RG call) ------------------------------------------

    def _to_rps(self, qn, K):
        """`{𝖖: {irrep: int}}` → `RPowerSeries` over `R(SU(2))`."""
        coeffs = {}
        for q, d in qn.items():
            if q > K:
                continue
            re = None
            for irrep, v in d.items():
                t = v * self._R.basis_element(irrep)
                re = t if re is None else re + t
            if re is not None:
                coeffs[q] = re
        return RPowerSeries(self._R, coeffs, K)

    def _times_chi(self, series, kappa: int, K: int):
        if not kappa:
            return series
        R = self._R
        res = series * RLaurent(R, {0: R.basis_element(kappa)})
        return RPowerSeries(R, {q: c for q, c in res.coeffs.items() if q <= K}, K)

    def trace(self, a, K: int = 20) -> RPowerSeries:
        """`Tr(L_a)` through `𝖖^K` (the `SU(2)`-refined Schur index): 0 if the
        magnetic charge is nonzero; Creutzig's closed form on `E^e`; the seed
        closed forms (`u1a1deven_seed_characters.seed_trace`) on a seed;
        otherwise the Layer-1 reduction onto those (`_layer1_trace`).  With
        `seed_closed_forms=False`, every label with curves by
        `DevenTraceTransport.trace_aux(Φ(a), K)` instead (the witness).  `χ_κ`
        folded in last."""
        curves, e, kappa = self.canonicalise(a)
        if self._magnetic_charge((curves, e)) != 0:
            return RPowerSeries(self._R, {}, K)
        if not curves:
            base = self._gauge_trace(e, K)
        elif self._seed_closed_forms:
            base = self._seed_trace((curves, e), K)
            if base is None:
                base = self._layer1_trace((curves, e), K)
        else:
            base = self._transport_trace((curves, e), K)
        return self._times_chi(base, kappa, K)

    def _seed_trace(self, label2, K: int):
        """The closed form of a seed `(curves, e)` through `𝖖^K`
        (`u1a1deven_seed_characters.seed_trace`), or None if `label2` is not a
        seed.  Its labels are this class's: the same curves, `e` measured from
        the same edge-{1, 2} convention (checked against the transport route on
        every seed orbit at k = 1..3, the suite in the source repository,
        and ρ on 1,200 random labels at k = 1..4)."""
        from u1a1deven_seed_characters import seed_trace
        curves, e = label2
        qn = seed_trace(self.k, curves, e, K)
        return None if qn is None else self._to_rps(qn, K)

    def _layer1_reduction(self, label2) -> Element:
        """`L_{(curves, e)}` as a combination of trace leaves with the same
        trace: the generic Layer-1 reduction of the cone data
        (`ConeData.simplify_trace_via_cone_data`: `ρ²`-twisted cyclicity and
        the product rule, on the `χ`-stripped labels through
        `_ChiStrippedView`), leaves on their `ρ²`-orbit representatives.
        Memoised per label (the reduction does not depend on the depth)."""
        hit = self._l1_memo.get(label2)
        if hit is None:
            if self._view_ is None:
                self._view_ = _ChiStrippedView(self)
            hit = self.cone_data().simplify_trace_via_cone_data(self._view_, label2)
            if len(self._l1_memo) >= _L1_MEMO_CAP:
                self._l1_memo = {}
            self._l1_memo[label2] = hit
        return hit

    def _layer1_trace(self, label2, K: int) -> RPowerSeries:
        """`Tr` of a label that is not a seed: its Layer-1 reduction
        (`_layer1_reduction`), each leaf traced by `_leaf_trace` through
        `𝖖^{K − v}`, `v ≤ 0` the lowest `𝖖`-power among the reduction's
        coefficients (they reach negative powers), then truncated to `𝖖^K`."""
        red = self._layer1_reduction(label2)
        R = self._R
        terms = [(leaf, co if isinstance(co, RLaurent) else RLaurent(R, dict(co._coeffs)))
                 for leaf, co in red.terms.items()]
        low = 0
        for _leaf, co in terms:
            for q in co.coeffs:
                low = min(low, q)
        acc: dict = {}
        for leaf, co in terms:
            t = self._leaf_trace(leaf, K - low)
            for q, c in (t * co).coeffs.items():
                if q <= K:
                    acc[q] = acc[q] + c if q in acc else c
        return RPowerSeries(self._R, {q: c for q, c in acc.items()
                                      if not c.is_zero()}, K)

    def _leaf_trace(self, leaf, K: int) -> RPowerSeries:
        """`Tr` of a leaf `(curves, e)` of the Layer-1 reduction: 0 if its
        magnetic charge is nonzero, Creutzig's closed form on `E^e`, the
        seed's closed form otherwise — in every measurement the leaves are
        single odd curves times `E^e` (the reduction takes the pairs of a +1
        and a −1 curve apart too).  A leaf that is not a seed would be traced
        by the transport and counted in `_layer1_stats`."""
        if self._magnetic_charge(leaf) != 0:
            return RPowerSeries(self._R, {}, K)
        if not leaf[0]:
            return self._gauge_trace(leaf[1], K)
        hit = self._leaf_cache.get(leaf)
        if hit is not None and hit[0] >= K:
            return RPowerSeries(self._R, {q: c for q, c in hit[1].coeffs.items()
                                          if q <= K}, K)
        t = self._seed_trace(leaf, K)
        if t is None:
            self._layer1_stats["transport_leaves"] += 1
            t = self._transport_trace(leaf, K)
        self._leaf_cache[leaf] = (K, t)
        return t

    def _transport_trace(self, label2, K: int) -> RPowerSeries:
        """`trace_aux(Φ(label2), K)` — the transport route (the witness, with
        `seed_closed_forms=False`; and for `E^e` the route Creutzig's closed
        form is checked against) — with a depth-extending memo shared by every
        instance of the same k in the process (`_TRACE_MEMO`, keyed by
        `(k, (curves, e))`, as the transport itself is shared:
        `_shared_transport(k)`), so a new instance — e.g. the one inside each
        `A1DevenKAlg(k)` — reuses the traces already computed."""
        label2 = self._canon2(label2)
        key = (self.k, label2)
        hit = _TRACE_MEMO.get(key)
        if hit is None or hit[0] < K:
            hit = (K, self._transport.trace_aux(self._phi2(label2), K))
            _TRACE_MEMO[key] = hit
        return self._to_rps({q: d for q, d in hit[1].items() if q <= K}, K)

    def _gauge_trace(self, e, K):
        """`Tr(E^e)` from the closed form, with a depth-extending memo: the
        `xᵉ` slice of Creutzig's `(A_1, D_{2p})` index, `p = k + 1`
        (`exact_characters.deven_gauged_xn_qn`; arbitrary `𝖖`).
        Cross-verified against the repo's independent `sl(3)₋₃/₂`
        Kac–Wakimoto vacuum character for `e = 0…4` to `q_d²⁸`
        (the design notes §2.1, the audit)."""
        hit = self._gauge_cache.get(e)
        if hit is not None and hit[0] >= K:
            return RPowerSeries(self._R, {q: c for q, c in hit[1].coeffs.items()
                                          if q <= K}, K)
        from exact_characters import deven_gauged_xn_qn
        series = self._to_rps(deven_gauged_xn_qn(self.k, e, K), K)
        self._gauge_cache[e] = (K, series)
        return series

    def _trace_residual(self, seed_label, K):
        """Contract hook (a seed of the generic Layer-1 reduction, either a
        label or a `χ`-stripped `(curves, e)`); the entry point is `trace`."""
        if len(seed_label) == 2:
            seed_label = (seed_label[0], seed_label[1], 0)
        return self.trace(seed_label, K)

    def _trace_of_product(self, a, b, K: int = 5) -> RPowerSeries:
        """`Tr(L_a·L_b)`, the product taken in the auxiliary algebra:
        `χ_{κa}·χ_{κb}·trace_aux(Φ(a)·Φ(b))`."""
        a, b = self.canonicalise(a), self.canonicalise(b)
        if self._magnetic_charge(a) + self._magnetic_charge(b):
            return RPowerSeries(self._R, {}, K)
        X = self._aux().multiply_elements(self._phi2(a[:2]), self._phi2(b[:2]))
        res = self._to_rps(self._transport.trace_aux(X, K), K)
        R = self._R
        chi = R.basis_element(a[2]) * R.basis_element(b[2])
        if chi == R.one():
            return res
        res = res * RLaurent(R, {0: chi})
        return RPowerSeries(R, {q: c for q, c in res.coeffs.items() if q <= K}, K)

    def inner_product(self, a, b, K: int = 20) -> RPowerSeries:
        """`I_{a,b} = Tr(L_{ρ(a)}·L_b)`.  The closed-form route (default):
        multiply then trace — this class's product, every term traced by
        `trace` (seeds from their closed forms, the rest by Layer 1 onto them).
        The transport route (`seed_closed_forms=False`, the witness): the
        product taken in the auxiliary algebra (`Φ` is the RG image, which is
        multiplicative), so no product of this class is used
        (`_trace_of_product`)."""
        if self._seed_closed_forms:
            return self.trace_element(self.multiply(self.rho(a), b), K)
        return self._trace_of_product(self.rho(a), b, K)

    def __repr__(self):
        return (f"U1A1DevenConeKAlgebra(k={self.k})  # U(1)-gauged "
                f"[A_1, D_{2 * self.k + 2}], curves of the {self.n}-gon with one "
                f"interior puncture, E = X_(0,1), SU(2) weight")


def u1a1deven_cone_dodd_section_iso(k: int):
    """The `KAlgebraIso  U1A1DevenConeKAlgebra(k) ≅ U1A1DevenViaDoddRG(k)`
    given by the bijection of labels `(curves, e, κ) ↦ ((word, κ), (c0, e))`
    (the flow label of the lowest term of `Φ`; module docstring, "Bridges to
    the flow"), coefficient 1 both ways.  Both sides are Z-form over
    `R(SU(2))`, so `bps_atlas.verify_cone_presentation` compares them
    directly.  Until 2026-09-24 the iso of this name was the section map of
    the retired ray-keyed tables."""
    from kalgebra_iso import KAlgebraIso
    from u1a1deven_via_dodd_rg import U1A1DevenViaDoddRG
    one = LaurentPoly.one()
    C = U1A1DevenConeKAlgebra(k)
    T = U1A1DevenViaDoddRG(k)
    return KAlgebraIso(
        C, T,
        lambda l: Element({C._flow_label(l): one}),
        lambda l: Element({C._label_of_flow_label(l): one}),
        name=f"U1A1DevenConeKAlgebra({k}) ≅(labels) U1A1DevenViaDoddRG({k})")


if __name__ == "__main__":
    import time
    t0 = time.time()
    A = U1A1DevenConeKAlgebra(1)
    cd = A.cone_data()
    print(f"U1A1DevenConeKAlgebra(k=1): {len(A._curve_set)} curves + E^±, "
          f"{len(cd.cones())} cones, R = {A.coefficient_ring()}  "
          f"[built in {time.time() - t0:.2f}s]")
    print("  (0,2)·(1,2) =", A.multiply(A.curve(0, 2), A.curve(1, 2)))
