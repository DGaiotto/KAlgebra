"""Tests for `rg_flow_solver.solve_RG` — the RG-flow F-solver analogue.

Validated against the BPS single-node-RG `RG` (ground truth) for the
odd AD family: hexagon `[A_1,A_3]` and octagon `[A_1,A_5]` on every IR
generator; decagon `[A_1,A_7]` by truncation-safe self-consistency
(`RG·S_RG = L_a + O(𝖖)`), the case where the BPS UV build is too slow.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bps_kalgebra import BPSKAlgebra
from rg_flow import SingleNodeRG
from rg_flow_solver import solve_RG, verify_RG
from laurent_poly import LaurentPoly


def _alin(n):
    B = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        B[i][i + 1] = 1
        B[i + 1][i] = -1
    return B, [tuple(1 if p == a else 0 for p in range(n)) for a in range(n)]


def _setup(k):
    n = 2 * k + 1
    B, N = _alin(n)
    uv = BPSKAlgebra(pairing=B, node_charges=N, spec=N, verify="off")
    rg = SingleNodeRG(uv, node_index=0)   # drop γ₁ (terminal)
    return rg, rg.ir_algebra, rg.rg_generator(10)


def _match(got, truth):
    gl = {l: h.expand(6) for l, h in got.items()}
    tr = dict(truth.terms)
    keys = set(gl) | set(tr)
    return all(gl.get(l, LaurentPoly.zero()) == tr.get(l, LaurentPoly.zero())
               for l in keys)


def test_solve_RG_matches_bps_hexagon():
    """solve_RG reproduces the true BPS RG on every hexagon IR generator
    and a product (γ₂·γ₃)."""
    rg, ir, S_RG = _setup(1)
    targets = list(ir.node_charges)
    # add a product label
    nn = list(ir.node_charges)
    for t in ir.multiply(nn[0], nn[1]).terms:
        targets.append(t)
    for a in targets:
        got = solve_RG(ir, S_RG, a, dropped_index=0, M=3)
        assert _match(got, rg.RG(a)), (a, {l: str(h.expand(6)) for l, h in got.items()})


def test_solve_RG_matches_bps_octagon():
    """solve_RG reproduces the true BPS RG on every octagon IR generator."""
    rg, ir, S_RG = _setup(2)
    for a in ir.node_charges:
        got = solve_RG(ir, S_RG, a, dropped_index=0, M=3)
        assert _match(got, rg.RG(a)), a


def test_solve_RG_decagon_self_consistent():
    """Decagon (the BPS-obstructed case): solve_RG yields RG satisfying
    `RG·S_RG = L_a + O(𝖖)` in the truncation-safe region — no slow A₇
    F-finder needed."""
    rg, ir, S_RG = _setup(3)
    for a in ir.node_charges:
        got = solve_RG(ir, S_RG, a, dropped_index=0, M=3)
        assert verify_RG(ir, S_RG, got, a, dropped_index=0, M_safe=2), a


if __name__ == "__main__":
    test_solve_RG_matches_bps_hexagon()
    test_solve_RG_matches_bps_octagon()
    test_solve_RG_decagon_self_consistent()
    print("\nAll rg_flow_solver tests passed.")
