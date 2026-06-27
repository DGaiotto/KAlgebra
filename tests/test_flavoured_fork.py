"""Self-test for the **ungauged `add_flavour` flows** (Step 3) — flows
realised over `A1A2kKAlg(k).add_flavour(flavour)` (the dropped matter kept as a
*spectator* flavour, not gauged), the ungauged companions to the gauged forks
(`test_e_type.py`, `test_a1an_chain.py`):

    A1Aodd:  [A₁, A_{2k+1}]  →  [A₁, A_{2k}] ⊕ U(1)    (A1AoddToEvenRGKAlgebra)
    E₇:      [A₁, E₇]        →  [A₁, A₆]    ⊕ U(1)    (E7RGKAlgebra, ungauged)
    A1Deven: [A₁, D_{2k+2}]  →  [A₁, A_{2k}] ⊕ U(2)    (A1DevenRGKAlgebra)

The U(1)-flavoured pair (A1Aodd, E₇) drops a *single* spectator hyper
(`S_RG = E_𝖖(μ·L)`); the U(2)-flavoured A1Deven drops the *two-fork doublet*
(`S_RG = E_𝖖(μ₁L)E_𝖖(μ₂L)` → U(2) characters).  All are pure exact-FS.

Both are pure exact-FS `RGKAlgebra`s over `A1A2kKAlg(k).add_flavour(1)` with
`S_RG = E_𝖖(μ·L)`: a single chord tower dressed by the spectator U(1) fugacity
μ (the dropped terminal's central charge).  *Which* chord `L` is dressed is the
whole distinction — the **short end** chord gives A1Aodd, the **central** chord
gives E₇ (the A/E fork, now in the U(1)-flavoured world; the gauged world's fork
is E₆/E₈ in `test_e_type.py`).

The trace is the **generic exact-FS bilinear pairing** (no override): it pairs
all components through the `add_flavour` auxiliary's `I^aux`, keeping the μ
flavour character, so the index is flavour-valued in `R(U(1))` — e.g. for
`[A₁, A₃]` the q² flavour current is the SU(2) adjoint `μ + 1 + μ⁻¹` (the U(1) ⊂
the enhanced SU(2)).

Depends on Step 1 (KAlgebra: `flavoured_kalgebra`, `zplus_ring`) and Step 2
(ConeKAlgebra: `a1a2k_kalg`).  Run (from the repository root):

    python3 run_tests.py
"""
import os
import sys
import warnings

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
for _root, _dirs, _ in os.walk(_SRC):
    _dirs[:] = [_d for _d in _dirs if _d != "__pycache__"]
    if _root not in sys.path:
        sys.path.insert(0, _root)

from a1aodd_to_even_rgkalgebra import A1AoddToEvenRGKAlgebra
from e7_rgkalgebra import E7RGKAlgebra
from a1deven_rgkalgebra import A1DevenRGKAlgebra

_SPINE = ("bps_kalgebra", "rg_flow", "lattice_torus", "nahm_data", "chart_graph",
          "bps_quiver_tools", "f_solver", "directional_subquiver_rg")


def _ser(rps, K):
    return {e: str(r) for e, r in sorted(rps.coeffs.items())
            if e <= K and str(r) not in ("0", "")}


def _axioms(A, labs, ortho_K=4):
    assert A.verify_rg_unital()
    for a in labs:
        assert A.verify_rg_bar_invariant(a), ("bar", a)
        for b in labs:
            assert A.verify_rg_multiplicative(a, b), ("mult", a, b)
            assert A.verify_orthonormality(a, b, ortho_K), ("ortho", a, b)


def _survivor_labs(A, n=3):
    labs = [A.identity()]
    H = A._A.H
    for i in range(min(n, H)):
        labs.append((((1, i, 1),), (0,)))
    return labs


def test_a1aodd_short_chord():
    """A1Aodd `[A₁, A_{2k+1}]` (end chord): exact-FS, μ-refined vacuum
    truncation-stable; k=1 q² flavour current = the SU(2) adjoint μ+1+μ⁻¹."""
    for k in (1, 2):
        A = A1AoddToEvenRGKAlgebra(k)
        assert A._fs_exact_available()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v6 = _ser(A.trace(A.identity(), 6), 6)
            nw = len(w)
        assert nw == 0, ("warns", k, nw)
        assert v6 == _ser(A.trace(A.identity(), 8), 6), ("not truncation-stable", k)
        _axioms(A, _survivor_labs(A))
    # k=1: the SU(2) adjoint at q²
    A1 = A1AoddToEvenRGKAlgebra(1)
    q2 = A1.trace(A1.identity(), 4).coeffs[2]
    assert dict(q2.terms) == {(-1,): 1, (0,): 1, (1,): 1}, ("q2 char", dict(q2.terms))
    print("  PASS: test_a1aodd_short_chord ([A1,A3] q² = μ+1+μ⁻¹; [A1,A5])")


def test_e7_central_chord():
    """Ungauged E₇ `[A₁, E₇]` (central chord): exact-FS, μ-refined vacuum (q²=1,
    the U(1) flavour current), truncation-stable, axioms; and the UV BPS quiver is
    E₇ (Cartan det 2 — distinguishes the central chord from the end chord's A₇=8)."""
    A = E7RGKAlgebra()
    assert A._fs_exact_available()
    assert A.uv_cartan_determinant() == 2, "UV quiver should be E7 (Cartan det 2)"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v6 = _ser(A.trace(A.identity(), 6), 6)
        nw = len(w)
    assert nw == 0, ("E7 warns", nw)
    assert str(A.trace(A.identity(), 2).coeffs.get(2)) == "1", "E7 q² = U(1) current"
    assert v6 == _ser(A.trace(A.identity(), 8), 6), "E7 not truncation-stable"
    _axioms(A, _survivor_labs(A))
    print("  PASS: test_e7_central_chord (UV = E7, Cartan det 2; q² = 1)")


def test_a1deven_u2():
    """A1Deven `[A₁, D_{2k+2}]` (two-fork doublet → U(2) flavour): exact-FS,
    U(2)-refined vacuum truncation-stable; k=1 ([A₁,D₄]) q² carries the U(2)
    flavour currents `1 + χ_{(1,−1)} + χ_{(1,1)} + χ_{(2,0)}` (the U(2) ⊂ the full
    D₄ flavour visible in this presentation)."""
    for k in (1, 2):
        A = A1DevenRGKAlgebra(k)
        assert A._fs_exact_available()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v4 = _ser(A.trace(A.identity(), 4), 4)
            nw = len(w)
        assert nw == 0, ("A1Deven warns", k, nw)
        assert v4 == _ser(A.trace(A.identity(), 6), 4), ("A1Deven not truncation-stable", k)
        # U(2) flavour ⇒ labels carry a 2-tuple (κ, m); survivor gens are (0,0).
        labs = [A.identity()] + [(((1, i, 1),), (0, 0)) for i in range(min(3, A._A.H))]
        _axioms(A, labs)
    A1 = A1DevenRGKAlgebra(1)
    q2 = A1.trace(A1.identity(), 4).coeffs[2]
    assert dict(q2.terms) == {(0, 0): 1, (1, -1): 1, (1, 1): 1, (2, 0): 1}, \
        ("A1Deven q2 U(2) currents", dict(q2.terms))
    print("  PASS: test_a1deven_u2 ([A1,D4] q² = U(2) adjoint; [A1,D6])")


def test_fork_distinct():
    """The end-chord (A1Aodd k=2 = [A₁,A₅]) and central-chord (E₇) flows are
    distinct theories: same μ-refined index through q⁵, diverging at q⁶
    (μ⁻²+5+μ² vs μ⁻²+6+μ²)."""
    A5 = A1AoddToEvenRGKAlgebra(2)
    E7 = E7RGKAlgebra()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        v5 = _ser(A5.trace(A5.identity(), 6), 6)
        v7 = _ser(E7.trace(E7.identity(), 6), 6)
    assert {e: c for e, c in v5.items() if e < 6} == {e: c for e, c in v7.items() if e < 6}, \
        "should agree < q6"
    assert v5[6] != v7[6], ("should diverge at q6", v5[6], v7[6])
    print("  PASS: test_fork_distinct ([A1,A5] vs E7 diverge at q⁶: %s vs %s)" % (v5[6], v7[6]))


def test_spine_free():
    hit = sorted(m for m in sys.modules
                 if any(m == s or m.startswith(s + ".") for s in _SPINE))
    assert not hit, ("spine modules leaked into the flavoured-fork release", hit)
    print("  PASS: test_spine_free (no spine modules imported)")


def main():
    test_a1aodd_short_chord()
    test_e7_central_chord()
    test_a1deven_u2()
    test_fork_distinct()
    test_spine_free()
    print("\nAll ungauged add_flavour-fork (A1Aodd + E7 + A1Deven) Step 3 tests passed.")


if __name__ == "__main__":
    main()
