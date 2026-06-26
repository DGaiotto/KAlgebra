"""``TensorZPlusRing`` — re-export shim.

The tensor-product Z₊-ring `R₁ ⊗ ⋯ ⊗ R_k` now lives canonically in
`zplus_ring.py` (the n-ary `TensorZPlusRing`), whose constructor accepts both
the list form `TensorZPlusRing([R1, R2, ...])` and the binary positional form
`TensorZPlusRing(R_a, R_b)`.  This module is kept only so existing
`from tensor_zplus_ring import TensorZPlusRing` imports keep resolving.

History: a separate *binary* `TensorZPlusRing` (`factor_a`/`factor_b`, itself
the earlier unification of `product_zplus_ring.ProductZPlusRing`) once lived
here.  It duplicated the n-ary class in `zplus_ring.py`, and its distinctive
accessors (`factor_a`/`factor_b`/`split_basis`/`factors()`) had **no external
callers**, so the two classes were **unified into the n-ary one** (Plan 32
streamline, 2026-06-14).  Every consumer uses the back-compatible constructor.
"""
from __future__ import annotations

from zplus_ring import TensorZPlusRing

__all__ = ["TensorZPlusRing"]
