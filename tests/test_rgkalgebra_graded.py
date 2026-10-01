"""Validate the generic `RGKAlgebra` API via a concrete graded subclass.

A minimal `RGKAlgebra` that supplies only the RG-flow data — `auxiliary`
(a quantum torus), `grading`, and `rg_generator` (`= S`) — and inherits
the full `KAlgebra` API.  Against the BPS anchor (`RG = F`), the inherited
generic `RG` / `multiply` / `rho_inverse` / `trace` must reproduce
`BPSKAlgebra`.  (Forward `rho` is overridden in subclasses; not tested
generically here — see A5.)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bps_kalgebra import BPSKAlgebra
from rgkalgebra import RGKAlgebra
from grading import Grading


class _PentagonViaQT(RGKAlgebra):
    """Graded `RGKAlgebra` over the pentagon's quantum torus, with
    `S_RG = S`.  Supplies only `(auxiliary, grading, rg_generator)`;
    everything else is the generic RGKAlgebra default."""

    def __init__(self, bps: BPSKAlgebra):
        self._bps = bps
        self._aux = bps.auxiliary()
        self._rank = len(bps.lattice.pairing)

    def auxiliary(self):
        return self._aux

    def grading(self):
        return Grading(rank=self._rank, deg=lambda lbl: tuple(lbl),
                       height=(1,) * self._rank)

    def rg_generator(self, cutoff):
        return self._bps.rg_generator(cutoff)


def _pentagon():
    A = BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=[(1, 0), (0, 1)])
    return A, _PentagonViaQT(A)


def test_generic_RG_reproduces_F():
    A, M = _pentagon()
    for a in [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)]:
        F = {k: v for k, v in A.F(a).items() if not v.is_zero()}
        RG = {k: v for k, v in M.RG(a).terms.items() if not v.is_zero()}
        assert RG == F, (a, RG, F)
    print("  PASS: test_generic_RG_reproduces_F")


def test_generic_multiply_reproduces_bps():
    A, M = _pentagon()
    pairs = [((1, 0), (0, 1)), ((1, 0), (1, 0)), ((0, 1), (1, 0)),
             ((1, 1), (0, 1))]
    for a, b in pairs:
        bps = {k: v for k, v in A.multiply(a, b).terms.items() if not v.is_zero()}
        gen = {k: v for k, v in M.multiply(a, b).terms.items() if not v.is_zero()}
        assert gen == bps, (a, b, gen, bps)
    print("  PASS: test_generic_multiply_reproduces_bps")


def test_generic_rho_inverse_reproduces_bps():
    A, M = _pentagon()
    for a in [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)]:
        assert M.rho_inverse(a) == A.rho_inverse(a), (a, M.rho_inverse(a), A.rho_inverse(a))
    print("  PASS: test_generic_rho_inverse_reproduces_bps")


def test_generic_rho_reproduces_bps():
    # Forward ρ_UV via the alternative RG map tRG (solve in the opposite algebra).
    A, M = _pentagon()
    for a in [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)]:
        assert M.rho(a) == A.rho(a), (a, M.rho(a), A.rho(a))
    print("  PASS: test_generic_rho_reproduces_bps")


def test_rg_trg_intertwine():
    # RG ∘ ρ_UV = ρ_IR ∘ tRG.
    A, M = _pentagon()
    for a in [(1, 0), (0, 1), (1, 1), (2, 1)]:
        assert M.verify_rg_trg_intertwine(a), a
    print("  PASS: test_rg_trg_intertwine")


def test_generic_trace_reproduces_bps():
    A, M = _pentagon()
    K = 8
    for a in [(1, 0), (0, 1), (1, 1)]:
        assert str(M.trace(a, K)) == str(A.trace(a, K)), (a, M.trace(a, K), A.trace(a, K))
    print("  PASS: test_generic_trace_reproduces_bps")


def test_coefficient_ring_and_identity():
    A, M = _pentagon()
    assert M.coefficient_ring() == A.coefficient_ring()
    assert M.identity() == A.auxiliary().identity()
    print("  PASS: test_coefficient_ring_and_identity")


def main():
    test_generic_RG_reproduces_F()
    test_generic_multiply_reproduces_bps()
    test_generic_rho_inverse_reproduces_bps()
    test_generic_rho_reproduces_bps()
    test_rg_trg_intertwine()
    test_generic_trace_reproduces_bps()
    test_coefficient_ring_and_identity()
    print("\nAll generic RGKAlgebra API tests passed.")


if __name__ == "__main__":
    main()
