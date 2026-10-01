"""
a1dodd_rgkalgebra.py
====================

`A1DoddRGKAlgebra(k)` — the **odd** D-type Argyres–Douglas family
`A_𝖖([A_1, D_{2k+3}])` with **SU(2) flavour symmetry**, as a new-contract
`RGKAlgebra`.  The SU(2)-flavoured, gauged-IR sibling of the
even-D `A1DevenRGKAlgebra` (U(2), over the *ungauged* even algebra).

The flow (two collinear fork hypers, μ ↔ μ⁻¹)
---------------------------------------------
Same short chord `L` (magnetic charge 1) of the gauged-odd standalone
`U1A1AoddKAlg(k)` as the warm-up `A1AevenToU1AoddRGKAlgebra`, but now
dressed by an SU(2)-Cartan fugacity `μ` in **two opposite-sign** quantum
dilogs:

    S_RG  =  E_𝖖(μ · L) · E_𝖖(μ⁻¹ · L)

— the two fork hypers form an SU(2) **doublet** (weights `μ^{±1}`).  Over
the *gauged*-odd survivor (the U(1) is already gauged away), the two
collinear `E_𝖖`'s combine into **SU(2)** characters (vs the **U(2)** of the
even-D case `E_𝖖(μ_1 L) E_𝖖(μ_2 L)` over the ungauged even algebra).

    auxiliary  =  U1A1AoddKAlg(k).add_flavour(SU2ZPlusRing())

— labels `((factors, e_E), κ)`: a `U1A1AoddKAlg(k)` cone monomial together
with the SU(2) χ-index `κ` (highest weight κ, spin κ/2).

Spectrum generator (SU(2)-character peel)
-----------------------------------------
`E_𝖖(μL) E_𝖖(μ⁻¹L) = Σ_{m,n} c_m c_n μ^{m−n} L^{m+n}`, `c_m = (−q)^m/(q²;q²)_m`.
At level `N = m+n` (the L-power, = magnetic charge of `L^N`) the μ-content
`Σ_{m+n=N} c_m c_n μ^{m−n}` peels into SU(2) irreps of highest weight
`κ = a−b` (`a = (N+κ)/2 ≥ b = (N−κ)/2 ≥ 0`):

    [S_RG]_{(N,)}  =  Σ_{κ ∈ {N, N−2, …}}  (c_a c_b − c_{a+1} c_{b−1})
                          · ( L^N , κ ),

the inverse-Kostka / Weyl peel `c_a c_b − c_{a+1} c_{b−1}` — *identical* to
the even-D U(2) coefficient (a U(2) irrep `(a,b)` restricts to the SU(2)
irrep of highest weight `a−b`).  Degree 0 is the SU(2) singlet (identity).

Grading (Γ_RG = Z = magnetic charge / L-power)
----------------------------------------------
`Γ_RG = Z` is the magnetic charge `N` of the cone monomial (= the L-power;
`E` is mag-neutral), positive cone `Z_{≥0}`, height 1 — the same grading
as the warm-up `A1AevenToU1AoddRGKAlgebra`.  `apex` is the identity (UV
labels = auxiliary cone labels); the SU(2) χ-content lives *within* each
grade as the flavour character.

Status
------
Odd-D companion to `A1DevenRGKAlgebra`.  UV `[A_1, D_{2k+3}]` (k=1 → D₅,
k=2 → D₇), SU(2) flavour.  Verified by the intrinsic K-algebra axioms and
the SU(2)-refined Schur index (the q² flavour current is the SU(2) adjoint
χ₂).
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from laurent_poly import LaurentPoly
from habiro import HabiroElement
from rgkalgebra import RGKAlgebra
from grading import Grading
from zplus_ring import SU2ZPlusRing
from u1a1aodd_kalg import U1A1AoddKAlg
from su2_doublet_dilog import (                        # the general-purpose
    su2_char_coeff as _su2_char_coeff,                 # E_𝖖(μL)·E_𝖖(μ⁻¹L) → χ
    su2_doublet_components as _su2_doublet_components,  # tool (su2_doublet_dilog)
)


class A1DoddRGKAlgebra(RGKAlgebra):
    """`[A_1, D_{2k+3}]` (odd-D AD, SU(2) flavour) as a directional new-contract
    `RGKAlgebra` wrapping `U1A1AoddKAlg(k).add_flavour(SU2ZPlusRing())`, with
    `S_RG = E_𝖖(μL) E_𝖖(μ⁻¹L)`.  See the module docstring."""

    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._base = U1A1AoddKAlg(k)
        self._aux = self._base.add_flavour(SU2ZPlusRing())     # SU(2) flavour
        self._cd = self._base.cone_data()
        self._n = self._cd._n                                  # B_GAUGED rank = 2k+2
        self._mag_index = self._n - 1                          # F coord = magnetic charge
        chg = self._cd._chg
        shorts_mag1 = sorted(
            i for i in self._cd._types[1] if chg[(1, i)][self._mag_index] == 1)
        if not shorts_mag1:
            raise RuntimeError(f"no magnetic-charge-1 short chord at k={k}")
        self._i0 = shorts_mag1[0]

    # ----- magnetic grading helper ---------------------------------------

    def _mag(self, cone_label) -> int:
        """Magnetic charge of a `U1A1AoddKAlg` cone label `(factors, e_E)`
        (`Σ exp·mag(chord)`; `E` is mag-neutral)."""
        chg, mi = self._cd._chg, self._mag_index
        factors, _e_E = cone_label
        return sum(exp * chg[(a, i)][mi] for (a, i, exp) in factors)

    # ----- RGKAlgebra contract -------------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z = magnetic charge of the cone monomial (= L-power = level N);
        # positive cone Z_{>=0}, height 1.
        return Grading(rank=1, deg=lambda lbl: (self._mag(lbl[0]),),
                       height=(1,), cone_gens=((1,),))

    def apex(self, a):
        """Identity apex: UV labels coincide with auxiliary cone labels."""
        a = tuple(a)
        return (a[0], a[1])

    def _short_chord_power(self, N: int):
        """`L^N` as a `U1A1AoddKAlg` cone label: the mag-1 short chord raised to
        `N` (the identity cone label `((), 0)` for `N = 0`)."""
        if N == 0:
            return ((), 0)
        return (((1, self._i0, N),), 0)

    def _s_rg_component(self, p):
        """`[S_RG]_{(N,)}` — exact, finite, vanishing off the cone.

        `S_RG = E_𝖖(μL) E_𝖖(μ⁻¹L)` ⇒ the level-`N` part (magnetic charge `N`) is
        the sum over SU(2) irreps of highest weight `κ = a−b` (`a+b=N`,
        `a≥b≥0`) of the chord `L^N` carried at SU(2) flavour `κ` with peel
        coefficient `c_a c_b − c_{a+1} c_{b-1}`; degree 0 is the SU(2) singlet
        (identity)."""
        (N,) = p
        if N < 0:
            return {}
        chord = self._short_chord_power(N)
        # The level-N SU(2)-character content of E_𝖖(μL)·E_𝖖(μ⁻¹L) is the
        # general-purpose `su2_doublet_components`; wrap each χ_κ on the
        # carrier chord `L^N`.
        return {(chord, kappa): coeff
                for kappa, coeff in _su2_doublet_components(N).items()}

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG` as the level ≤ `cutoff` window: the SU(2)-character tower
        `{(L^N, κ): coeff}` for `N < cutoff`."""
        out: dict = {}
        for N in range(cutoff):
            out.update(self._s_rg_component((N,)))
        return out

    # ----- trace (SU(2)-refined Schur index) -----------------------------

    def rg_s_graded(self, a, K: int) -> dict:
        """`RG(a)·S_RG` up to order `q^K`, decomposed by **Γ_RG (magnetic /
        level) charge** — `{γ : aux Element}`.  Each `S_RG` coefficient kept
        exact (Habiro), expanded to `q^{K+margin}` only here; the SU(2) flavour
        combines by Clebsch–Gordan (`multiply_basis`).  γ-graded so each chord
        product stays bounded."""
        base = self._base
        flav_ring = self._aux._flav
        rg = self.RG(a)
        if not rg.terms:
            return {}
        m_min = min(self._mag(lbl[0]) for lbl in rg.terms)
        m_max = max(self._mag(lbl[0]) for lbl in rg.terms)
        margin = (self.k + 1) * K + 8
        pieces: dict[int, Element] = {}
        for gamma in range(m_min, m_max + K + 1):
            piece: dict = {}
            for (cone_c, kappa_c), f_c in rg.terms.items():
                j = gamma - self._mag(cone_c)
                if j < 0 or f_c.is_zero():
                    continue
                comps = self._s_rg_component((j,))
                if not comps:
                    continue
                base_terms = ({cone_c: LaurentPoly.one()} if j == 0
                              else base.multiply(cone_c,
                                                 self._short_chord_power(j)).terms)
                for (_chord_s, kappa_s), coeff_s in comps.items():
                    cs = coeff_s.expand(K + margin)
                    if cs.is_zero():
                        continue
                    flav = flav_ring.multiply_basis(kappa_c, kappa_s)
                    for cone_out, sc in base_terms.items():
                        if sc.is_zero():
                            continue
                        contrib = f_c * sc * cs
                        if contrib.is_zero():
                            continue
                        for fout, mult in flav.items():
                            key = (cone_out, fout)
                            add = contrib * mult
                            piece[key] = piece[key] + add if key in piece else add
            piece = {kk: lp for kk, lp in piece.items() if not lp.is_zero()}
            if piece:
                pieces[gamma] = Element(piece)
        return pieces

    def inner_product(self, a, b, K: int = 20):
        """`I_{a,b}(q)` — the **SU(2)-refined** Schur inner product
        `⟨RG_a·S_RG, RG_b·S_RG⟩_aux = Σ_{γ',γ} ⟨(RG_a S)_{γ'}, (RG_b S)_γ⟩_aux`.
        Pairs **all** `(γ',γ)` so the SU(2) flavour character (carried by the
        net charge) is kept (pairing only `γ'=γ` would drop the charged SU(2)
        currents)."""
        from zplus_ring import RPowerSeries
        aux = self.auxiliary()
        R = self.coefficient_ring()
        Kc = K + 2
        ga = self.rg_s_graded(a, Kc)
        gb = self.rg_s_graded(b, Kc)
        total = RPowerSeries(R, {}, Kc)
        for gp in ga:
            for g in gb:
                total = total + aux.trace_element(
                    aux.multiply_elements(aux.rho_element(ga[gp]), gb[g]), Kc)
        return RPowerSeries(R, {e: c for e, c in total.coeffs.items() if e <= K}, K)

    def trace(self, a, K: int = 20):
        """`Tr_UV(L_a) = I_{1,a}` — the SU(2)-refined Schur index of
        `[A_1, D_{2k+3}]` (see `inner_product`)."""
        return self.inner_product(self.identity(), tuple(a), K)

    # ----- geometric labels: once-punctured (2k+3)-gon arcs --------------

    def _hexchord(self, ray):
        """The underlying `U1A1AoddKAlg` chord `(a, i)` of a single-chord ray
        `((factors, e_E), κ)`, or `None` if not a single chord (product /
        E-power / identity)."""
        (factors, _e_E), _kappa = ray
        if len(factors) == 1 and factors[0][2] == 1:
            return (factors[0][0], factors[0][1])
        return None

    def geometric_label(self, ray):
        """A single-chord ray as a **once-punctured `(2k+3)`-gon arc**:
        `(endpoints, tag)`.  The underlying `U1A1AoddKAlg` `(2k+4)`-gon diagonal
        collapses one vertex (taken to be the last, `2k+3 → P`, the puncture —
        a free, crossing-transparent choice) onto the puncture; `tag = κ` is the
        winding (0 = plain / non-winding, 1 = notched / winds around `P`).
        Endpoints are boundary vertices `0..2k+2`, or `'P'` for the puncture.
        Returns `None` for non-single-chord rays (cluster monomials = products
        of arcs)."""
        hc = self._hexchord(ray)
        if hc is None:
            return None
        (_factors, _e_E), kappa = ray
        v1, v2 = self._base.cone_data().geometric_label(hc)   # (2k+4)-gon diag
        c = 2 * self.k + 3                                     # collapsed vertex → P
        ends = []
        for v in (v1, v2):
            ends.append('P' if v == c else (v if v < c else v))
        return (tuple(ends), kappa)

    def intersection_number(self, ray_a, ray_b):
        """Geometric intersection number of two single-chord ray arcs on the
        once-punctured `(2k+3)`-gon: the boundary crossing of the underlying
        `(2k+4)`-gon diagonals **plus** one around-the-puncture crossing when
        **both** arcs are notched (`κ=1`) and distinct.  Equals
        `log2(#terms in multiply)` (the Plücker exchange doubles per crossing) —
        verified by `verify_plucker_is_intersection_number`."""
        ha, hb = self._hexchord(ray_a), self._hexchord(ray_b)
        if ha is None or hb is None:
            raise ValueError("intersection_number is defined on single-chord rays")
        cd = self._base.cone_data()
        boundary = 0 if ha == hb else (0 if cd.q_commute(ha, hb) else 1)
        ka, kb = ray_a[1], ray_b[1]
        winding = 1 if (ka == 1 and kb == 1 and ha != hb) else 0
        return boundary + winding

    def verify_plucker_is_intersection_number(self, rays) -> bool:
        """The geometric content of the multiply: for single-chord ray
        generators, `#terms(multiply(a, b)) == 2 ** intersection_number(a, b)` —
        Plücker exchange doubles once per arc crossing (boundary + the
        around-puncture crossing of two notched arcs).  Verified 153/153 on the
        k=1 single-chord generators."""
        rays = list(rays)
        for i, a in enumerate(rays):
            for b in rays[i + 1:]:
                if self._hexchord(a) is None or self._hexchord(b) is None:
                    continue
                nt = len(self.multiply(a, b).terms)
                if nt != 2 ** self.intersection_number(a, b):
                    return False
        return True

    # ----- flavour-aware section split -----------------------------------

    def _section_split(self, label):
        """The auxiliary labels are `((factors, e_E), κ)` — the SU(2) flavour is
        a basis index, not a flat-vector charge — so disable the flavour-shift
        multiply cache (`flav = None`): `multiply` falls back to the direct
        `from_ir_image(RG(a)·RG(b))` per pair (the flavour is central)."""
        return tuple(label), None


if __name__ == "__main__":
    for k in (1,):
        T = A1DoddRGKAlgebra(k)
        print(f"A1DoddRGKAlgebra(k={k}) = [A_1, D_{2*k+3}]  (SU(2) flavour)")
        print("  aux =", type(T.auxiliary()).__name__, "  dressing short-chord i0 =", T._i0)
        print("  S_RG = E_q(muL)E_q(mu^-1 L), SU(2)-char peel:")
        for N in range(3):
            comp = T._s_rg_component((N,))
            print(f"    level N={N}:")
            for (chord, kappa), c in sorted(comp.items(), key=lambda t: -t[0][1]):
                print(f"      chi_{kappa} (spin {kappa}/2)  chord={chord}:  {c}")
        # RG of a survivor generator (a long mag-0 chord), flavour-singlet.
        g = ((((2, 0, 1),), 0), 0)
        rg = T.RG(g)
        print(f"  RG(L(2,0)) has {len(rg.terms)} terms:",
              {l: str(c) for l, c in list(rg.terms.items())[:4]})
