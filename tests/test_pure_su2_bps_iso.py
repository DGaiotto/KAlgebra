"""Full verifier battery for `pure_su2_bps_iso` — the modern
`PureSU2KAlgebra ≅ BPS pure SU(2)` witness, INCLUDING trace-equivariance
(the documented gap of `abelianized_su2_bps_iso`, whose abe side carried the
decoupled U(2) photon trace)."""
import os
import sys
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kalgebra import Element
from laurent_poly import LaurentPoly
from pure_su2_bps_iso import pure_su2_bps_iso

LABELS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, -1), (2, 0), (2, 1)]


def _el(lab):
    return Element({lab: LaurentPoly.one()})


def main():
    print("PureSU2KAlgebra ≅ BPS pure SU(2) — full battery")
    iso = pure_su2_bps_iso()
    src = [_el(a) for a in LABELS]
    tgt = [iso.map(x) for x in src]
    assert iso.verify_unit(), "unit"
    assert iso.verify_round_trip(src, tgt), "round trip"
    print("  PASS: unit + round-trip (8 labels)")
    pairs = [(_el(a), _el(b)) for a, b in
             itertools.product(LABELS[:6], repeat=2)]
    tpairs = [(iso.map(x), iso.map(y)) for x, y in pairs]
    assert iso.verify_multiplicative(pairs, tpairs), "multiplicative"
    print("  PASS: multiplicativity (36 pairs, both directions)")
    assert iso.verify_rho_equivariant(src, tgt), "rho"
    print("  PASS: ρ-equivariance")
    assert iso.verify_trace_equivariant(src, tgt, K=10), "trace"
    print("  PASS: trace-equivariance (K=10) — the upgraded claim")
    print("All PureSU2KAlgebra↔BPS iso tests passed.")


if __name__ == "__main__":
    main()
