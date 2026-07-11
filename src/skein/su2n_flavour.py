"""`su2n_flavour` -- promote a `U(1)^n`-Cartan-flavoured KAlgebra to its
manifest **SU(2)^n** flavour symmetry.

The closed n-punctured sphere `Sk(S^2_{0,n})` has a per-puncture SU(2)
flavour (guaranteed chart-by-chart by the self-folded-triangle Z_2
criterion), but the BPS realisation exposes only the
abelian Cartan `U(1)^n` (`coefficient_ring() == AbelianZPlusRing(n)`).
`SU2nFlavouredKAlgebra(base, n)` re-presents `base` over
`SU(2)^n = TensorZPlusRing([SU2ZPlusRing()] * n)`.

This is the SU(2)^n analogue of `flavour_enhancement.FlavourEnhancementKAlgebra`
(which does `U(1)^{Nf}` -> `Spin(2Nf)`): the **inverse of `base_change`
restriction** `SU(2)^n -> U(1)^n`.  The Weyl group is `(Z_2)^n` -- each
puncture's `mu_p <-> mu_p^{-1}` -- and the un-branch is factor-wise:
`chi_k <- mu^k + mu^{k-2} + ... + mu^{-k}` per puncture (the inverse of
`SU2ZPlusRing.to_abelian`).  The Z-form `multiply` is unchanged (flavour
lives in the coefficient ring); the SU(2)^n multiplets surface in
`trace` / `inner_product` / `to_R_form`.
"""

from __future__ import annotations

import os
import sys
import warnings
from itertools import product as _iproduct

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import KAlgebra, Element
from zplus_ring import (
    AbelianZPlusRing, SU2ZPlusRing, TensorZPlusRing, RElement, RPowerSeries,
)


class SU2nFlavourError(ValueError):
    """Raised when a quantity that should be SU(2)^n-covariant is not
    (Z_2)^n-invariant (the per-puncture SU(2) symmetry is absent/broken)."""


def _su2n_char_to_abelian(k: tuple[int, ...]) -> dict[tuple[int, ...], int]:
    """Abelian (U(1)^n) content of the SU(2)^n character
    chi_{k_1} (x) ... (x) chi_{k_n} = prod_i (mu_i^{k_i} + ... + mu_i^{-k_i}).
    Every resulting weight has coefficient 1."""
    ranges = [range(-ki, ki + 1, 2) for ki in k]
    return {combo: 1 for combo in _iproduct(*ranges)}


def unbranch_su2n(r: RElement):
    """Un-branch a `U(1)^n` Cartan character `r` (RElement over
    `AbelianZPlusRing(n)`) into `SU(2)^n` content.

    Returns `(content, invariant, remainder)`: `content` is the RElement
    over `TensorZPlusRing([SU2]*n)`; `invariant` is True iff `r` is fully
    `(Z_2)^n`-symmetric (remainder empty); `remainder` is the leftover
    abelian content (non-empty iff not invariant).

    Peel highest-dominant-weight: among weights with all coords >= 0 take
    the maximal (by total then lex; same-total dominant weights are
    incomparable so the order is immaterial), record that SU(2)^n
    character, subtract its abelian expansion, repeat -- triangular."""
    if not isinstance(r.ring, AbelianZPlusRing):
        raise TypeError("unbranch_su2n: r must be over AbelianZPlusRing(n)")
    n = r.ring.rank
    R = TensorZPlusRing([SU2ZPlusRing() for _ in range(n)])
    rem: dict[tuple[int, ...], int] = {w: c for w, c in r.terms.items() if c}
    out: dict[tuple[int, ...], int] = {}
    while True:
        dominant = [w for w in rem if all(x >= 0 for x in w)]
        if not dominant:
            break
        w = max(dominant, key=lambda v: (sum(v), v))
        c = rem[w]
        out[w] = out.get(w, 0) + c
        for ab, abc in _su2n_char_to_abelian(w).items():
            rem[ab] = rem.get(ab, 0) - c * abc
            if rem[ab] == 0:
                del rem[ab]
    content = RElement(R, {k: v for k, v in out.items() if v})
    return content, (not rem), dict(rem)


class SU2nFlavouredKAlgebra(KAlgebra):
    """`base` (a `U(1)^n`-Cartan-flavoured KAlgebra) re-presented over
    `SU(2)^n`.  See the module docstring.

    **Trace-level RECOGNITION DIAGNOSTIC, not a faithful
    ``free``-over-``SU(2)^n`` KAlgebra.**  It keeps the abelian base's
    ``U(1)^n`` labels and only un-branches the *trace* into SU(2)^n
    multiplets — the canonical basis is never reorganized — so it has no
    per-label enhanced irrep (``r_label_decompose`` raises;
    ``_label_section_decompose`` returns the base ``U(1)^n`` Cartan
    coordinate).  A *faithful* SU(2)^n-flavoured closed-sphere realisation is
    not provided; this class is the recognition
    diagnostic ``verify_su2n_enhancement`` / trace un-brancher.
    """

    def __init__(self, base: KAlgebra, n: int | None = None,
                 *, strict: bool = True):
        warnings.warn(
            "SU2nFlavouredKAlgebra is a deprecated trace-level recognition "
            "diagnostic, not a faithful free-over-SU(2)^n KAlgebra (a faithful "
            "closed-sphere realisation is future Plan-31 work).",
            DeprecationWarning, stacklevel=2,
        )
        Rb = base.coefficient_ring()
        if not isinstance(Rb, AbelianZPlusRing):
            raise TypeError(
                "SU2nFlavouredKAlgebra: base.coefficient_ring() must be "
                f"AbelianZPlusRing(n); got {Rb!r}"
            )
        if n is None:
            n = Rb.rank
        if Rb.rank != n:
            raise TypeError(
                f"base Cartan rank {Rb.rank} != n={n}"
            )
        self._base = base
        self._n = n
        self._R = TensorZPlusRing([SU2ZPlusRing() for _ in range(n)])
        self._strict = strict

    # ---- contract: structure delegates, flavour re-presented ----
    def coefficient_ring(self):
        return self._R

    def identity(self):
        return self._base.identity()

    def multiply(self, a, b) -> Element:
        return self._base.multiply(a, b)   # Z-form is flavour-agnostic

    def rho(self, a):
        return self._base.rho(a)

    def rho_inverse(self, a):
        return self._base.rho_inverse(a)

    def _label_section_decompose(self, label):
        """Delegates to the base: the per-label flavour coordinate lives at the
        U(1)^n **Cartan** level (the base's), so the returned `RElement` is over
        the base's `AbelianZPlusRing(n)` — *not* `coefficient_ring()` = SU(2)^n.
        The SU(2)^n multiplet structure is **emergent in the trace**
        (`_lift_rps`), not a per-basis-element grading — see `r_label_decompose`
        for why no per-label SU(2)^n lift exists."""
        return self._base._label_section_decompose(label)

    def r_label_decompose(self, label):
        """**No per-label SU(2)^n lift coordinate** (deliberately raises).

        Each base canonical `L_a` carries a single U(1)^n Cartan weight `μ^{w_a}`
        (`base.r_label_decompose`), which is one *weight* of an SU(2)^n
        multiplet — not a whole irrep, since `χ_k` spans several base labels.
        So `L_a ≠ χ · L_{section}` for any single SU(2)^n irrep: the SU(2)^n
        symmetry is **emergent in the trace** (`_lift_rps` un-branches `Tr` to
        SU(2)^n), not a per-basis-element grading.  The genuine per-label lift
        is the base's, at the U(1)^n Cartan level (`self._base`); flavour
        reduction back to U(1)^n (`base_change(restriction)`) and `forget()`
        (`base_change(augmentation)`) need only the coefficient pushforward, not
        this coordinate."""
        raise NotImplementedError(
            "SU2nFlavouredKAlgebra.r_label_decompose: the SU(2)^n flavour is "
            "emergent in the trace, not a per-label grading; the per-label lift "
            "coordinate is the base's (U(1)^n Cartan) — use self._base."
        )

    # ---- the enhancing lift  R(U(1)^n) -> R(SU(2)^n) ----
    def _lift_relement(self, r: RElement) -> RElement:
        content, invariant, rem = unbranch_su2n(r)
        if not invariant and self._strict:
            raise SU2nFlavourError(
                f"flavour content {dict(r.terms)} is not (Z_2)^{self._n}-"
                f"invariant — the SU(2)^{self._n} symmetry is absent "
                f"(remainder {rem})."
            )
        return content

    def _lift_rps(self, rps: RPowerSeries) -> RPowerSeries:
        out = {}
        for qexp, rc in rps.coeffs.items():
            lifted = self._lift_relement(rc)
            if not lifted.is_zero():
                out[qexp] = lifted
        K = rps.K if hasattr(rps, "K") else 20
        return RPowerSeries(self._R, out, K)

    def trace(self, a, K: int = 20) -> RPowerSeries:
        return self._lift_rps(self._base.trace(a, K))

    def inner_product(self, a, b, K: int = 20) -> RPowerSeries:
        return self._lift_rps(self._base.inner_product(a, b, K))

    def verify_su2n_enhancement(self, labels, K: int = 6) -> bool:
        """Whether `base` is genuinely SU(2)^n-enhanced on `labels`: every
        `Tr(L_a)` Cartan-coefficient is `(Z_2)^n`-invariant (un-branches)."""
        for a in labels:
            for rc in self._base.trace(a, K).coeffs.values():
                _, invariant, _ = unbranch_su2n(rc)
                if not invariant:
                    return False
        return True
