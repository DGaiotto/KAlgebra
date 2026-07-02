"""``TensorZPlusRing`` — re-export shim.

The tensor-product Z₊-ring `R₁ ⊗ ⋯ ⊗ R_k` lives canonically in
`zplus_ring.py` (the n-ary `TensorZPlusRing`), whose constructor accepts both
the list form `TensorZPlusRing([R1, R2, ...])` and the binary positional form
`TensorZPlusRing(R_a, R_b)`.  This module is kept only so existing
`from tensor_zplus_ring import TensorZPlusRing` imports keep resolving.
"""
from __future__ import annotations

from zplus_ring import TensorZPlusRing

__all__ = ["TensorZPlusRing"]
