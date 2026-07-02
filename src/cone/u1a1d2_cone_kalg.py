"""`U1A1D2ConeKAlgebra` — SQED₂ = [A₁, D₂] = U(1) gauge + two charged
hypers = `U_𝖖(𝔰𝔩₂)`, the **SU(2)-flavoured cone standalone**.

Algebraically identical to `SQED2SampleKAlgebra` (the Step-1 direct
sample); this class exposes the **cone-monomial presentation** with the
`U_𝖖(𝔰𝔩₂)` Laurent-cone structure (`U1A1D2ConeData`), fitting SQED₂ into
the `ConeKAlgebra` tier alongside Pentagon, A1A2k, U1Square, A1D3.  The
certified Step-1↔Step-2 correspondence is
`u1a1d2_sqed2_sample_cone_iso.py`
(`U1A1D2ConeKAlgebra (cone) ↔ SQED2SampleKAlgebra (sample)`).

Encoding (the crux — Z-representation / flavour-in-LABEL)
--------------------------------------------------------
To match `SQED2SampleKAlgebra` exactly:
  * `coefficient_ring()` → `SU2ZPlusRing()` (basis = int spin `k ≥ 0`;
    `multiply_basis` = SU(2) Clebsch-Gordan).
  * the canonical basis **label** carries the spin `k` as a coordinate:
    `(m, n, k)` (3-tuple of ints), with `(m, n)` the gauge part and
    `k ≥ 0` the SU(2) spin (the `χ_k` flavour irrep).
  * `Element` coefficients are plain `LaurentPoly` (ℤ[𝖖^±]) — NOT
    `RLaurent` (so `KAlgebraIso._extend`'s `coef * sub_coef` stays
    well-defined against the sample's `LaurentPoly` coeffs).
  * `trace` returns an `RPowerSeries` over `SU2ZPlusRing` (R-valued).

Native label structure
----------------------
Public label `(m, n, k)`:
  * `m` — the magnetic / charged-hyper direction; signs the `E`(+) /
    `F`(−) power.  `m > 0` ↔ `E_{m,n}`, `m < 0` ↔ `F_{−m,n}`,
    `m = 0` ↔ `Kⁿ` (the `U_𝖖(𝔰𝔩₂)` PBW basis with χ-index `k`).
  * `n` — the Cartan / `K` power.
  * `k ≥ 0` — the SU(2) spin (`χ_k`).

An INTERNAL `U1A1D2ConeData` works on the gauge part `(m, n)` (`k`
stripped) over `SU2ZPlusRing`, with the `χ₁`-content of the `E·F`
cross-products living in the `RLaurent[SU(2)]` coefficient.

`multiply` (A1D3-style Pattern-III recipe)
------------------------------------------
  1. split each label `(m, n, k)` into a gauge part `(m, n)` and a spin
     `k`;
  2. `cone_data().derived_multiply((m_a,n_a), (m_b,n_b))` → `Element`
     over `(m, n)` with `RLaurent[SU(2)]` coeffs (the generic cone
     reducer; **de-risked to reproduce SQED₂'s U_𝖖(𝔰𝔩₂) straightener
     exactly**, including E²F² and mixed K-powers — no `_uqsl2_multiply`
     fallback needed);
  3. fuse `χ_{k_a}·χ_{k_b}` (SU(2) Clebsch-Gordan, via `RElement.__mul__`)
     into the gauge result's coefficient;
  4. re-expand each `((m,n), RLaurent)` term into `(m, n, k_out)` labels
     with `LaurentPoly` coeffs (iterate the RLaurent 𝖖-powers and the
     RElement spin components).

`ρ`
---
Lusztig's braid (matching `SQED2SampleKAlgebra` / `uq_sl2_pbw`):
`ρ(K^n) = K^{−n}`, `ρ(E_{a,b}) = F_{a,−a−b}`, `ρ(F_{a,b}) = E_{a,a−b}`,
`χ_k` fixed (SU(2) self-dual).  On the `(m, n, k)` label this is the
**uniform** permutation

    ρ(m, n, k) = (−m, −m − n, k)

(verified term-by-term against the oracle — for `m = 0`:
`(0, −n, k) = ρ(Kⁿ·χ_k)`; for `m ≠ 0`: the E↔F braid).

trace (spine-free)
------------------
The Schur index localises to the Cartan sector: `Tr L_{m,n,k} = 0` for
`m ≠ 0`, and `Tr (Kⁿ·χ_k) = χ_k · [x^n] G(x, μ)` (the paper's SQED₂
index).  Delegated to the Step-1 `SQED2SampleKAlgebra.trace` (the same
algebra, the same gauge labels), keeping a single spine-free source of
truth for the closed-form Schur index.  (The cone's universal
`ConeKAlgebra.trace` Layer-1/Layer-2 pipeline is **not** used — the
flavour `χ_k` sits in the public label, not in the coefficient ring, so
SQED₂ supplies its own `trace` like `A1D3KAlg` does; `_trace_residual`
is a raising stub.)
"""

from __future__ import annotations

import sys
import os

# Put every src/<layer>/ directory on sys.path (the project's bare-name import
# convention) so this module also imports standalone from a checkout;
# run_tests.py / conftest.py do the same for the full gate.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _root, _dirs, _ in os.walk(_SRC):
    _dirs[:] = [_d for _d in _dirs if _d != "__pycache__"]
    if _root not in sys.path:
        sys.path.insert(0, _root)

from kalgebra import KAlgebra, Element
from cone_kalgebra import ConeKAlgebra
from zplus_ring import ZPlusRing, RElement, RLaurent, RPowerSeries, SU2ZPlusRing
from laurent_poly import LaurentPoly
from u1a1d2_cone_data import U1A1D2_CONE_DATA


__all__ = ["U1A1D2ConeKAlgebra"]


Label = tuple   # (m, n, k) 3-tuple of ints


class U1A1D2ConeKAlgebra(ConeKAlgebra):
    """SQED₂ = [A₁, D₂] = `U_𝖖(𝔰𝔩₂)` K-algebra, cone-monomial-presented
    over `SU2ZPlusRing` with the spin `k` in the label (the SU(2)-flavoured
    cone standalone).

    See module docstring for the `(m, n, k)` label structure, the
    A1D3-style `multiply`, the uniform braid `ρ(m,n,k) = (−m,−m−n,k)`, and
    the spine-free `trace` (m≠0 ⇒ 0; m=0 ⇒ the SQED₂ index via the Step-1
    sample)."""

    def __init__(self):
        self._R = SU2ZPlusRing()

    # -- KAlgebra primitives ----------------------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self) -> Label:
        return (0, 0, 0)

    def canonicalise(self, x) -> Label:
        x = tuple(int(c) for c in x)
        if len(x) != 3:
            raise ValueError(f"U1A1D2 label must be 3-tuple, got {x}")
        m, n, k = x
        if k < 0:
            raise ValueError(f"U1A1D2 label spin k must be ≥ 0, got {x}")
        return (m, n, k)

    def cone_data(self):
        return U1A1D2_CONE_DATA

    def _trace_residual(self, seed_label, K):
        """Required abstract on `ConeKAlgebra`.  Unused: SQED₂'s public
        `trace` carries the spin `k` in the label (not the coefficient
        ring), so it supplies its own closed-form trace (delegated to the
        Step-1 sample) rather than routing cone-data trace seeds here —
        mirroring `A1D3KAlg._trace_residual`.  Raising keeps the two
        pipelines clearly separated."""
        raise NotImplementedError(
            "U1A1D2ConeKAlgebra._trace_residual: SQED₂ uses its own "
            "closed-form `trace` (spine-free, delegated to the Step-1 "
            "SQED2SampleKAlgebra); cone-data trace seeds are not routed here."
        )

    # -- multiply (A1D3-style flavour-in-label) ----------------------------

    def multiply(self, a: Label, b: Label) -> Element:
        """Cone-data multiplication with χ-index folding at the
        3-tuple ↔ 2-tuple boundary.  See module docstring."""
        m_a, n_a, k_a = self.canonicalise(a)
        m_b, n_b, k_b = self.canonicalise(b)

        # Cone-data multiply on the gauge parts (k stripped).
        result_cone = self.cone_data().derived_multiply(
            (m_a, n_a), (m_b, n_b),
        )

        # χ_{k_a} · χ_{k_b} via SU(2) CG (handled by RElement.__mul__).
        chi_a = self._R.basis_element(k_a)
        chi_b = self._R.basis_element(k_b)
        chi_prod = chi_a * chi_b   # RElement: Σ χ_j over the CG decomposition

        # Re-expand to 3-tuple labels with LaurentPoly coeffs.
        out: dict[Label, LaurentPoly] = {}
        for gauge, coef in result_cone.terms.items():
            m, n = gauge
            # Coerce to RLaurent for uniform handling.
            if isinstance(coef, LaurentPoly):
                rl = RLaurent(
                    self._R,
                    {e: RElement(self._R, {0: c})
                     for e, c in coef._coeffs.items()},
                )
            else:
                rl = coef
            # Scale by chi_prod on the R-side and expand into SU(2) components.
            for q_exp, r_elt in rl.coeffs.items():
                scaled = r_elt * chi_prod
                if scaled.is_zero():
                    continue
                for k_out, coef_int in scaled.terms.items():
                    if coef_int == 0:
                        continue
                    lab = (m, n, k_out)
                    lp_add = LaurentPoly({q_exp: int(coef_int)})
                    out[lab] = out.get(lab, LaurentPoly({})) + lp_add
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    # -- ρ (Lusztig braid; uniform on (m, n, k)) --------------------------

    def rho(self, a: Label) -> Label:
        m, n, k = self.canonicalise(a)
        # ρ(K^n)=K^{-n}; ρ(E_{a,b})=F_{a,-a-b}; ρ(F_{a,b})=E_{a,a-b}.
        # On (m, n): uniformly (m, n) ↦ (-m, -m - n).
        return (-m, -m - n, k)

    def rho_inverse(self, a: Label) -> Label:
        M, N, k = self.canonicalise(a)
        # ρ has the braid drift ρ²(m,n) = (m, n + 2m) (NOT an involution on
        # the Cartan power n — the infinite-order Lusztig braid covering
        # K ↦ K⁻¹).  Invert ρ(m,n) = (-m, -m-n) explicitly: from
        # (M, N) = (-m, -m-n) ⇒ m = -M and n = -m - N = M - N.
        return (-M, M - N, k)

    # -- trace (spine-free; delegated to the Step-1 sample) ---------------

    def trace(self, a: Label, K: int = 20) -> RPowerSeries:
        """Schur index, localised to the Cartan sector.

        `Tr L_{m,n,k} = 0` for `m ≠ 0`; `Tr (Kⁿ·χ_k) = χ_k · [x^n] G(x,μ)`
        for `m = 0` — delegated to the Step-1 `SQED2SampleKAlgebra`
        (identical algebra, same gauge labels), the spine-free single
        source of truth for the closed-form SQED₂ index."""
        m, n, k = self.canonicalise(a)
        if m != 0:
            return RPowerSeries.zero(self._R, K)
        # m = 0: Tr(Kⁿ·χ_k).  Delegate to the Step-1 sample's closed form.
        return self._sqed2_helper().trace((("K", n), k), K=K)

    def _sqed2_helper(self):
        """Cached `SQED2SampleKAlgebra` (Step 1) for the m=0 Cartan-sector
        trace — the spine-free closed-form SQED₂ Schur index."""
        cached = getattr(self, "_sqed2_cache", None)
        if cached is None:
            from samples import SQED2SampleKAlgebra
            cached = SQED2SampleKAlgebra()
            self._sqed2_cache = cached
        return cached

    # -- flavour-lift coordinate (genuinely flavoured) --------------------

    def r_label_decompose(self, label):
        """The single-irrep flavour-lift coordinate: peel the SU(2) spin
        `k` (the R-basis-label) off the gauge part `(m, n)`."""
        m, n, k = self.canonicalise(label)
        return (m, n, 0), k

    def r_label_compose(self, section, r_basis_label):
        """Inverse of `r_label_decompose`: insert the spin `r_basis_label`
        into the (flavour-trivial) gauge section's spin slot."""
        m, n, _ = self.canonicalise(section)
        return self.canonicalise((m, n, r_basis_label))

    def embed_R(self, r: RElement) -> Element:
        """Central embedding `R(SU(2)) ↪ A_𝖖`: each character `χ_k` maps to
        the central canonical basis element `L_{(0,0,k)}` (the pure spin-k/2
        character), extended ℤ-linearly.  Backs `from_R_form`."""
        R = self.coefficient_ring()
        if not isinstance(r, RElement) or r.ring != R:
            raise TypeError(
                "embed_R: argument must be an RElement over coefficient_ring()"
            )
        out = Element.zero()
        for k, coeff in r.terms.items():
            if coeff == 0:
                continue
            out = out + Element.basis(self.canonicalise((0, 0, k))) * coeff
        return out

    # -- convenience ------------------------------------------------------

    def L(self, label) -> Element:
        return Element.basis(self.canonicalise(label))

    def __repr__(self) -> str:
        return "U1A1D2ConeKAlgebra(SQED₂ = [A₁, D₂] = U_𝖖(𝔰𝔩₂), SU(2)-flavoured cone)"


if __name__ == "__main__":
    A = U1A1D2ConeKAlgebra()
    print(repr(A))
    print("  E·F =", {l: str(c) for l, c in A.multiply((1, 0, 0), (-1, 0, 0)).terms.items()})
    print("  F·E =", {l: str(c) for l, c in A.multiply((-1, 0, 0), (1, 0, 0)).terms.items()})
    print("  ρ(E) =", A.rho((1, 0, 0)), " ρ(F) =", A.rho((-1, 0, 0)), " ρ(K) =", A.rho((0, 1, 0)))
    print("  Tr(1) =", A.trace((0, 0, 0), K=8))
