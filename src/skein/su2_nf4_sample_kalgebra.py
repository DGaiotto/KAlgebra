"""`SU2Nf4SampleKAlgebra` — SU(2)+N_f=4 (conformal) as a **clean, self-contained,
spine-free** `KAlgebra` on the **SU(2)-character-cone (bracelet) basis** over
**R(Spin(8))**.

This is the orthonormal rebuild: *every elementary skein / simple closed
curve is a RAY, and
each ray generates an SU(2) character cone* — the fundamental Wilson line (the
2-dim SU(2) rep) is the generating ray `χ₁ = γ`, and its powers decompose into the
SU(2) characters `χ_k` (Chebyshev 2nd kind).  By contrast, a GNO/SO(3) basis
with `χ₁ = γ + 𝟙` carries a `+𝟙` identity-screening piece that breaks
cross-ray orthonormality (`I(a,b)|q⁰ = 8s − 1`, χ₀-component `−1 ≠ 0`).

NO BPS / RG / quantum-torus engine on any path: multiply is the intrinsic
Kauffman skein (`skein_algebra`), peeled to the bracelet basis,
with the **A² = −𝖖** sign (the skein↔SU(2)/quantum-group dictionary — the unknot
`−A²−A⁻²` is the positive quantum dimension `𝖖+𝖖⁻¹`); the trace is the exact
Schur–Askey–Wilson contour functional (`schur_measure`),
**transformed by triality under SL(2,ℤ)** from one channel frame to another; the
flavour is **Spin(8) = D₄** (triality); `ρ = id` (conformal N_f=4: the SU(2)
curves are self-dual, so `ρ² = id` and the bar involution fixes the basis).

Canonical basis (the per-ray **SU(2)-character cone**)
=====================================================
  * the identity `𝟙` ........................................ label `("I",)`
  * `χ_w · χ_k(s)`: the SU(2) character `χ_k` of a primitive curve-slope `s`
    (`χ₀ = 𝟙`, `χ₁ = γ_s`, `χ_{k+1} = γ_s·χ_k − χ_{k-1}`; same-ray SU(2) Clebsch
    `χ_jχ_k = Σ_{l=|j-k|, step 2}^{j+k} χ_l`), dressed by a **single Spin(8)
    irrep** `χ_w` (central flavour) .... label `("C", slope, k, dynkin)`
  * central `χ_w · 𝟙` (a dressed identity) ................. label `("F", w)`
with `slope` a primitive Z⁶ normal-coordinate vector (an SL(2,ℤ) slope on
S²₀,₄), `k ≥ 1`, and `dynkin ∈ ℕ⁴` a Spin(8) = D₄ highest weight in the
**triality-manifest** convention: `(0,0,0,0)=𝟙`, `(1,0,0,0)=8v`, `(0,0,1,0)=8s`,
`(0,0,0,1)=8c`, `(0,1,0,0)=28`.  `("C", slope, k, (0,0,0,0))` is the **section**
(undressed curve); the irrep is the flavour dressing.  This is the contract's
flavour-in-labels **Z-form**: `multiply` returns integer (`LaurentPoly`)
structure constants, flavour carried in the labels.

The quantum parameter — `𝖖 = A²` (the A↔𝖖 relation)
==================================================
The skein lives over `Z[A^±]` (Kauffman); the Coulomb-branch quantum parameter is
`𝖖 = A²`.  **Only integer powers of 𝖖 occur** in every physical (closed-correlator
/ trace) quantity: closed-curve products have even A-powers (→ integer 𝖖), the
Schur–measure trace is graded by integer 𝖖-powers, and the matter `8v` sits at
`𝖖¹` (`Tr(γ) = 8v·𝖖 + 160v·𝖖³ + …`).  (A single *open* Wilson strand is `A^{odd}`,
but it is never a closed correlator, so its "𝖖^{1/2}" never enters multiply or
trace — the half-integer language elsewhere in the scaffolding conflated an
A-power with a 𝖖-power.)

Orthonormality — HOLDS, every pairing on 𝖖 ≥ 0
=========================================================
The canonical basis is orthonormal to leading order with **no negative 𝖖-power in
front of any Spin(8) character** (the user's hard requirement): `I_{a,b} =
δ_{a,b} + O(𝖖)`, read on the χ₀ (trivial-Spin(8)-irrep) component of the 𝖖⁰
coefficient (the flavoured orthonormality).  Verified
across the level-1,2 basis (`test_all_pairings_q_nonneg_and_orthonormal`):
  * diagonal `I(χ_k(X), χ_k(X))|𝖖⁰ = 𝟙` (χ₀-component 1), support 𝖖 ≥ 0;
  * cross-ray `I(χ₁(a), χ₁(b))` supported on **𝖖 ∈ {2,4}** — q⁰ empty, fully
    orthogonal — matching the U(2)+N_f=4 oracle `UNNfKAlgebra(2,4)` (U(1)-ungauged):
    `I(H,W) = [2,4]`;
  * higher-k cross-ray `I(χ₂(a), χ₁(b))` supported on **𝖖 ≥ 3** — the spurious
    `𝖖⁻¹·(8c+8v)` of the sign-blind trace cancels exactly under A²=−𝖖 + triality.
The `8s − 1` defect of the GNO basis is gone: on `χ₁ = γ` there is no `+𝟙`
screening, so no spurious `−χ₀`.

The two ingredients that secure 𝖖 ≥ 0 (both physical, not fits):
  * **A² = −𝖖** in `multiply` (the unknot `−A²−A⁻²` becomes the positive quantum
    dimension `𝖖+𝖖⁻¹`; "𝖖 factors only appear in mult"), and
  * the trace **transforms by triality under SL(2,ℤ)** between channel frames
    (`trace`, `_CHANNEL_TRIALITY`): channel `a` is the electric/Wilson frame
    (matter 8v), `b`/`c` are its triality images (8c, 8s).
`ρ = id` is correct (it is NOT the source of the old 𝖖⁻¹): for conformal N_f=4 the
SU(2) curves are self-dual, and the ungauged U(2) charge-conjugation ρ collapses
to the identity on the SU(2) content (confirmed against the oracle).

`Tr(𝟙) = 𝟙 + 28·𝖖² + (𝟙 + 28 + 300)·𝖖⁴ + …` is the SU(2)+N_f=4 Schur index
(28 = the SO(8) conserved-current adjoint).

Coefficient ring
================
`coefficient_ring()` returns `spin8_characters.Spin8ZPlusRing` = R(Spin(8)) =
R(D₄) in the triality-manifest Dynkin basis — the ring the verified trace /
multiply / un-branching machinery is built on (its object-identity equality is
load-bearing).  Mathematically this is `SO2NfZPlusRing(4)` in a different
label convention.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import KAlgebra, Element                            # noqa: E402
from laurent_poly import LaurentPoly                              # noqa: E402
from zplus_ring import RElement, RPowerSeries, AbelianZPlusRing   # noqa: E402

from su2_nf4_bracelet import bracelet, _peel_to_bracelet          # noqa: E402
from su2_nf4_so8 import periph_x_weights as _periph_x_weights     # noqa: E402
from su2_nf4_skein_chars import _S, GA, GB, GC, GAT, GBT, GCT     # noqa: E402
from su2_nf4_symmetry import GEN as _GEN                          # noqa: E402
import su2_nf4_trace as _trace                                    # noqa: E402


_IDENTITY = ("I",)

# The three simple-closed-curve channels split by the mod-2 reduction of the
# Z⁶ normal coordinates (a known Z₂×Z₂ structure on S²₀,₄: distinct slopes in
# the same class are SL(2,ℤ)/Dehn-twist partners).
_MOD2_CHANNEL = {tuple(x % 2 for x in g): nm for nm, g in _GEN.items()}

# The Schur trace transforms by **triality under SL(2,ℤ)**: channel `a` is the
# Wilson (electric) reference frame — matter in **8v** (`σ = id`); the duality to
# the other two frames is the triality that carries 8v to that channel's matter
# 8 (channel↔irrep dictionary `a↔8v, b↔8c, c↔8s`).  `triality(·, perm)` permutes
# the three legs (8v, 8s, 8c) = Dynkin nodes (0, 2, 3); perm `(2,1,0)` swaps
# 8v↔8c and `(1,0,2)` swaps 8v↔8s.
_CHANNEL_TRIALITY = {
    "a": (0, 1, 2),     # 8v matter  (Wilson reference frame)
    "b": (2, 1, 0),     # 8v ↔ 8c    (8c matter)
    "c": (1, 0, 2),     # 8v ↔ 8s    (8s matter)
}

# The canonical R(Spin(8)) instance.  `Spin8ZPlusRing` uses object-identity
# equality, so EVERY Spin(8) element this class builds (un-branching here, the
# central dressings, the trace) must go through this one instance — the one the
# trace (`su2_nf4_trace._R8`) returns its RElements over — or arithmetic hits
# "RElement: ring mismatch".
_R8 = _trace._R8
# the SO(8) orthogonal-frame torus (doubled-e weights) used to un-branch
# accumulated peripheral x-weights to genuine Spin(8) characters.
_AB = AbelianZPlusRing(rank=4)


def _unbranch_xweights(xw: dict) -> RElement:
    """A doubled-e SO(8) orthogonal-frame x-weight multiset `{x-tuple: mult}` →
    a genuine Spin(8) `RElement` (over the canonical `_R8`).  Accumulate the
    WHOLE flavour content of a (curve, 𝖖-power) before calling: a single
    peripheral label is generally only part of a Spin(8) multiplet."""
    xw = {x: m for x, m in xw.items() if m}
    if not xw:
        return _R8.zero()
    return _R8.from_abelian(RElement(_AB, xw))


class SU2Nf4SampleKAlgebra(KAlgebra):
    """SU(2)+N_f=4 (conformal) on the SU(2)-character-cone (bracelet) basis over
    R(Spin(8)) — clean, self-contained, spine-free, and orthonormal.  See the
    module docstring for the label scheme, multiply, ρ, trace, and the 𝖖 = A²
    convention."""

    #: the three channel curve-slopes (puncture pairings) and their first
    #: Dehn-twist partners on the tetrahedral S²₀,₄.
    SLOPES = {"a": GA, "b": GB, "c": GC, "a~": GAT, "b~": GBT, "c~": GCT}

    def __init__(self, channel: str = "a") -> None:
        # `channel` selects the W_ch frame used to un-branch peripheral flavour
        # to Spin(8).  The Spin(8) flavour identity is channel-independent
        # (triality), so the choice is a convention.
        self.skein = _S
        self._channel = channel
        self._mul_cache: dict = {}

    # ==================== the six abstract primitives ====================

    def coefficient_ring(self):
        """R(Spin(8)) = R(D₄), the SU(2)+N_f=4 flavour ring (triality-manifest
        Dynkin basis)."""
        return _R8

    def identity(self):
        return _IDENTITY

    # ---- label <-> (curve, flavour) plumbing ----

    @staticmethod
    def _chan_of(slope) -> str:
        """The triality channel ('a' | 'b' | 'c') of a curve-slope, by the mod-2
        reduction of its normal coordinates."""
        m2 = tuple(x % 2 for x in slope)
        ch = _MOD2_CHANNEL.get(m2)
        if ch is None:
            raise ValueError(
                f"slope {slope!r} (mod 2 = {m2}) is not in a known triality "
                f"channel {{a, b, c}} — not a simple-closed-curve slope on S²₀,₄?")
        return ch

    @staticmethod
    def _split_label(label):
        """Split any canonical label into `(curve_label, flavour_dynkin)`, where
        `curve_label` is `("I",)` (the identity / a k=0 curve) or
        `("C", slope, k, (0,0,0,0))` (a bare curve-ray), and `flavour_dynkin` is
        the Spin(8) irrep dressing it."""
        if label == _IDENTITY:
            return _IDENTITY, (0, 0, 0, 0)
        if label[0] == "F":                        # central χ_w · 𝟙
            return _IDENTITY, label[1]
        _tag, slope, k, dynkin = label             # ("C", slope, k, dynkin)
        return ("C", slope, k, (0, 0, 0, 0)), dynkin

    def _curve_el(self, curve_label):
        """The skein element of a bare curve label (flavour stripped): `𝟙` or the
        bracelet SU(2)-character `χ_k(slope)`."""
        if curve_label == _IDENTITY:
            return self.skein.one()
        _tag, slope, k, _zero = curve_label
        return bracelet(slope, k)

    def _curve_product_spin8(self, curve_a, curve_b) -> dict:
        """The bare-curve bracelet product `χ_{k_a}(s_a) · χ_{k_b}(s_b)` (flavour
        dropped) as `{('I',) | (slope, k): {𝖖-power: Spin(8) RElement}}` — the
        peripheral content grouped per (curve, 𝖖-power) and un-branched once."""
        key = (curve_a, curve_b)
        cached = self._mul_cache.get(key)
        if cached is not None:
            return cached
        prod = _peel_to_bracelet(self._curve_el(curve_a) * self._curve_el(curve_b))
        # accumulate peripheral x-weights per (curve-key, 𝖖-power)
        acc: dict = {}
        for lbl, co in prod.items():
            if isinstance(lbl[0], str) and lbl[0] == "P":   # identity-flavour
                ckey, periph = _IDENTITY, lbl[1]
            elif len(lbl) == 3:                             # curve + peripheral
                ckey, periph = (lbl[0], lbl[1]), lbl[2]
            else:                                           # bare curve (slope,k)
                ckey, periph = (lbl[0], lbl[1]), (0, 0, 0, 0)
            xw = _periph_x_weights(periph, self._channel)
            for e, c in co._c.items():
                if e % 2:
                    raise ValueError(
                        f"odd Kauffman A-power A^{e} on curve term {lbl!r} — "
                        f"closed-curve products must have even A-powers (𝖖 = A²)")
                q = e // 2
                # A² = −𝖖 (the skein↔SU(2)/quantum-group dictionary): A^{2q} =
                # (−1)^q 𝖖^q, so the unknot −A²−A⁻² becomes the *positive* SU(2)
                # quantum dimension 𝖖+𝖖⁻¹.  The sign is carried by `multiply`
                # ("𝖖 factors only appear in mult"); it is +1 on the 𝖖⁰
                # (same-ray Clebsch) terms and flips only the crossing terms.
                sgn = -1 if (q & 1) else 1
                d = acc.setdefault(ckey, {}).setdefault(q, {})
                for x, m in xw.items():
                    d[x] = d.get(x, 0) + sgn * c * m
        out: dict = {}
        for ckey, qd in acc.items():
            for q, wd in qd.items():
                r = _unbranch_xweights(wd)
                if not r.is_zero():
                    out.setdefault(ckey, {})[q] = r
        self._mul_cache[key] = out
        return out

    def multiply(self, a, b) -> Element:
        """`L_a · L_b` as an `Element` over `Z[𝖖^±]` (flavour in the labels).

        Spin(8) flavour is central, so the product is
        `(χ_{r_a} · χ_{r_b}) ⊗ [χ_{k_a}(s_a) · χ_{k_b}(s_b)]`: the two explicit
        irrep dressings fuse by the R(Spin(8)) Clebsch–Gordan tensor and combine
        with the (Spin(8)-valued) bare-curve bracelet product."""
        curve_a, dyn_a = self._split_label(a)
        curve_b, dyn_b = self._split_label(b)
        dress = _R8.basis_element(dyn_a) * _R8.basis_element(dyn_b)  # CG fuse
        out: dict = {}
        for ckey, qd in self._curve_product_spin8(curve_a, curve_b).items():
            for q, r_curve in qd.items():
                r_tot = dress * r_curve            # central: fuse all flavour
                for w, n in r_tot.terms.items():
                    if n == 0:
                        continue
                    if ckey == _IDENTITY:
                        lbl = _IDENTITY if w == (0, 0, 0, 0) else ("F", w)
                    else:
                        slope, k = ckey
                        lbl = ("C", slope, k, w)
                    poly = out.get(lbl)
                    add = LaurentPoly({q: n})
                    out[lbl] = (poly + add) if poly is not None else add
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    def rho(self, a):
        """ρ = the identity — correct for conformal N_f=4.  On the central
        flavour ρ is rep-ring duality ⋆ (every D₄ irrep is self-dual ⇒ id); on the
        curves the SU(2) holonomies are self-dual, and the ungauged-U(2)
        charge-conjugation collapses to the identity on the SU(2) content
        (confirmed against the `UNNfKAlgebra(2,4)` oracle).  So `ρ² = id` and the
        bar involution fixes the basis.  (The old 𝖖⁻¹ in higher-k cross-ray
        pairings was NOT a ρ defect — it was the sign-blind / slope-blind trace;
        fixed by A²=−𝖖 in multiply + the SL(2,ℤ)-triality trace.)"""
        return a

    rho_inverse = rho

    def rho_squared_is_identity(self) -> bool:
        return True

    def trace(self, a, K: int = 20) -> RPowerSeries:
        """The `ρ²`-twisted trace `Tr(L_a) ∈ R(Spin(8))((𝖖))`, the exact
        Schur–Askey–Wilson contour functional, in integer 𝖖 = A² powers.

        Flavour is central, so `Tr(χ_w · χ_k(s)) = χ_w · Tr(χ_k(s))`.  The
        reference (channel `a`, electric/Wilson) trace is the SU(2)-character
        moment `tr_wilson(k+1)` (the bracelet `χ_k` is the (k+1)-dim SU(2)
        character; `tr_wilson(1) = Tr(𝟙)`), matter in **8v**.  The trace
        **transforms by triality under SL(2,ℤ)**: a curve in channel `b`/`c` is a
        different duality frame, and its trace is the triality image
        `σ_channel(tr_wilson(k+1))` (`_CHANNEL_TRIALITY`).  `tr_wilson(k+1)`
        starts at `𝖖^k` in every frame (so `Tr(χ_k) = O(𝖖^k)`, the orthonormality
        grading); triality is a pure flavour permutation — no extra 𝖖-shift."""
        curve, dynkin = self._split_label(a)
        k = 0 if curve == _IDENTITY else curve[2]
        qmax = max(K, 0)
        base = _trace.tr_wilson(k + 1, qmax)       # {𝖖-power: Spin(8) RElement}
        if curve != _IDENTITY:                     # identity sector is triality-fixed
            perm = _CHANNEL_TRIALITY[self._chan_of(curve[1])]
            if perm != (0, 1, 2):
                base = {q: _R8.triality(r, perm) for q, r in base.items()}
        r_w = _R8.basis_element(dynkin)
        out: dict = {}
        for q, r in base.items():
            if q > K:
                continue
            c = r_w * r
            if not c.is_zero():
                out[q] = c
        return RPowerSeries(_R8, out, K)

    # ============ flavour-lift coordinate (the seventh primitive) ============

    def r_label_decompose(self, label):
        """`(section_label, irrep)` with `L_label = χ_irrep · L_section`: the
        section is the undressed curve / identity, the irrep its single Spin(8)
        flavour dressing (a D₄ Dynkin tuple)."""
        return self._split_label(label)

    def r_label_compose(self, section, r_basis_label):
        """Inverse of `r_label_decompose`: dress a section curve / identity by a
        single Spin(8) irrep."""
        if section == _IDENTITY:
            return _IDENTITY if tuple(r_basis_label) == (0, 0, 0, 0) \
                else ("F", tuple(r_basis_label))
        _tag, slope, k, _zero = section
        return ("C", slope, k, tuple(r_basis_label))

    def embed_R(self, r: RElement) -> Element:
        """Central embedding R(Spin(8)) ↪ A: each irrep χ_w ↦ the central
        canonical `χ_w · 𝟙` (label `("F", w)`, or the identity for χ₀)."""
        if not isinstance(r, RElement) or r.ring is not _R8:
            raise TypeError("embed_R: argument must be an RElement over R(Spin(8))")
        out: dict = {}
        for w, n in r.terms.items():
            if n == 0:
                continue
            lbl = _IDENTITY if w == (0, 0, 0, 0) else ("F", w)
            poly = out.get(lbl)
            add = LaurentPoly({0: n})
            out[lbl] = (poly + add) if poly is not None else add
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    # ==================== convenience constructors ====================

    def gen(self, name: str):
        """The Wilson-line / character-ray generator `χ₁(channel)` (label)."""
        return ("C", self.SLOPES[name], 1, (0, 0, 0, 0))

    def character(self, name: str, k: int):
        """The SU(2) spin-k/2 character ray `χ_k(channel)` (label); k=0 → 𝟙."""
        if k == 0:
            return _IDENTITY
        return ("C", self.SLOPES[name], k, (0, 0, 0, 0))


__all__ = ["SU2Nf4SampleKAlgebra"]


if __name__ == "__main__":
    NAME = {(0, 0, 0, 0): "1", (1, 0, 0, 0): "8v", (0, 0, 1, 0): "8s",
            (0, 0, 0, 1): "8c", (0, 1, 0, 0): "28"}

    def fmt(r):
        return " + ".join((f"{c}·" if c != 1 else "") + NAME.get(b, str(b))
                          for b, c in sorted(r.terms.items())) or "0"

    def kfmt(el):
        parts = []
        for lbl in sorted(el.terms, key=str):
            co = el.terms[lbl]
            if lbl == ("I",):
                nm = "𝟙"
            elif lbl[0] == "F":
                nm = NAME.get(lbl[1], str(lbl[1])) + "·𝟙"
            else:
                fl = "" if lbl[3] == (0, 0, 0, 0) else NAME.get(lbl[3], str(lbl[3])) + "·"
                nm = f"{fl}χ{lbl[2]}({'a' if lbl[1]==GA else 'b' if lbl[1]==GB else 'c' if lbl[1]==GC else lbl[1]})"
            parts.append(f"({co})·{nm}")
        return " + ".join(parts) or "0"

    def supp(ip):
        return sorted(q for q, r in ip.coeffs.items() if not r.is_zero())

    A = SU2Nf4SampleKAlgebra()
    a, b, c = A.gen("a"), A.gen("b"), A.gen("c")
    chi2a = A.character("a", 2)
    print(f"mro: {[t.__name__ for t in type(A).__mro__]}")
    print(f"coefficient_ring(): {type(A.coefficient_ring()).__name__}  (= R(Spin(8)))")
    print(f"\nsame-ray SU(2) Clebsch   χ₁(a)·χ₁(a) = {kfmt(A.multiply(a, a))}")
    print(f"cross-ray (A²=−𝖖 signs)  χ₁(a)·χ₁(b) = {kfmt(A.multiply(a, b))}")
    print(f"\ntrace by triality (SL(2,ℤ)):  Tr(χ₁(a))|𝖖¹ = {fmt(A.trace(a, 3)[1])}"
          f"   Tr(χ₁(b))|𝖖¹ = {fmt(A.trace(b, 3)[1])}"
          f"   Tr(χ₁(c))|𝖖¹ = {fmt(A.trace(c, 3)[1])}")
    print(f"Schur index  Tr(𝟙)|𝖖² = {fmt(A.trace(_IDENTITY, 4)[2])} (28 currents)")
    print("\northonormal, every pairing 𝖖 ≥ 0:")
    print(f"  I(a,a) 𝖖-support {supp(A.inner_product(a, a, 4))}  (diagonal: 𝖖⁰ → 1)")
    print(f"  I(a,b) 𝖖-support {supp(A.inner_product(a, b, 4))}  (cross-ray; 𝖖⁰ empty"
          f" = oracle I(H,W)=[2,4])")
    print(f"  I(χ₂(a),χ₁(b)) 𝖖-support {supp(A.inner_product(chi2a, b, 5))}  "
          f"(was a spurious 𝖖⁻¹; now 𝖖 ≥ 3)")
    print("done")
