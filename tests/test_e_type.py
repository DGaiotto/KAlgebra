"""Self-test for the **exceptional E-type Argyres–Douglas RG flows** (Step 3)
— the flavourless, exact-FS entries `[A₁, E₆]` and `[A₁, E₈]`:

    [A₁, E₆]  ──drop the central node──▶  u(1)-gauged [A₁, A₅]  (= U1A1AoddKAlg(2))
    [A₁, E₈]  ──drop the central node──▶  u(1)-gauged [A₁, A₇]  (= U1A1AoddKAlg(3))

Both are the E-series analogues of the A-type chain's leg 1
(`A1AevenToU1AoddRGKAlgebra`): a single-node drop with `S_RG = E_𝖖(L)`, the chord
`L` being the **central diameter** (vs the A-type's end short chord) — which chord
is dropped is the whole A/E distinction.  Flavourless (`[A₁, E₆]`/`[A₁, E₈]` have
no flavour symmetry; rank 6/8 = `U1A1AoddKAlg(2/3)`'s rank `2k+2`), so the
auxiliary is the gauged-odd cone directly — no `add_flavour`, no quantum torus.

This suite depends on Step 1 (KAlgebra) + Step 2 (ConeKAlgebra: the cone
auxiliary `U1A1AoddKAlg`, whose frozen `.pkl` tables are what make these flows
**spine-free** at runtime — the tables come from a derivation that is not run,
or needed, here).

For each: exact-FS, truncation-stable vacuum Schur index, the KAlgebra axioms +
orthonormality, and a no-spine-imported assertion.

`[A₁, E₇]` enters this suite via its **u(1)-gauged** form `U1A1E7RGKAlgebra` (aux
`A1A2kKAlg(3) ⊗ QT(Z²)`, `S_RG = E_𝖖(X_{(0,1)}·L_{(2,2)})`): gauging the U(1)
turns the ungauged E₇'s slow `add_flavour(1)` spectator into a clean
quantum-torus leg, putting it on the same pure exact-FS engine as E₆/E₈.

Run (from the repository root):

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

from e6_rgkalgebra import E6RGKAlgebra
from e8_rgkalgebra import E8RGKAlgebra
from u1a1e7_rgkalgebra import U1A1E7RGKAlgebra

# Spine-freeness: the module list is shared and filesystem-derived
# (tests/_spine.py) — the previous hand-copied tuple named modules that no
# longer exist and omitted real spine modules.
from _spine import SPINE as _SPINE


def _ser(rps, K):
    return {e: str(r) for e, r in sorted(rps.coeffs.items())
            if e <= K and str(r) not in ("0", "")}


def _check(A, name):
    assert A._fs_exact_available(), (name, "exact-FS unavailable")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v6 = _ser(A.trace(A.identity(), 6), 6)
        v8 = _ser(A.trace(A.identity(), 8), 6)
        nw = len(w)
    assert v6 == v8 and nw == 0, (name, "vacuum not truncation-stable / warns", v6, v8, nw)
    # small label battery from the survivor cone
    e = A.identity()
    labs = [e]
    for d in list(A.rg_generator(2)):
        if d != e and d not in labs:
            labs.append(d)
        if len(labs) >= 4:
            break
    assert A.verify_rg_unital()
    for a in labs:
        assert A.verify_rg_bar_invariant(a), (name, "bar", a)
        for b in labs:
            assert A.verify_rg_multiplicative(a, b), (name, "mult", a, b)
            assert A.verify_orthonormality(a, b, 4), (name, "ortho", a, b)
    return v6


def test_e6():
    A = E6RGKAlgebra()
    v = _check(A, "E6")
    print("  PASS: test_e6 ([A1,E6] → u(1)-gauged [A1,A5])  vac =", v)


def test_e8():
    A = E8RGKAlgebra()
    v = _check(A, "E8")
    print("  PASS: test_e8 ([A1,E8] → u(1)-gauged [A1,A7])  vac =", v)


def test_u1a1e7():
    """The u(1)-gauged `[A₁, E₇]` — the E₇ representative (the ungauged E₇ has a
    slow `add_flavour(1)` refined trace; gauging the U(1) makes it pure exact-FS).
    aux = `A1A2kKAlg(3) ⊗ QT(Z²)`, `S_RG = E_𝖖(X_{(0,1)}·L_{(2,2)})`."""
    A = U1A1E7RGKAlgebra()
    assert A._fs_exact_available()
    assert (A.DRESS_TYPE, A._i0) == (2, 2)             # node-4 interior dressing
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v10 = _ser(A.trace(A.identity(), 10), 10)
        nw = len(w)
    assert nw == 0 and v10 == {0: "1", 2: "-1", 6: "1", 8: "1", 10: "1"}, ("U1A1E7 vac", v10, nw)
    assert _ser(A.trace(A.identity(), 6), 6) == _ser(A.trace(A.identity(), 10), 6)
    labs = [A.identity(), (((1, 0, 1),), (0, 0)), (((1, 1, 1),), (0, 0)),
            ((), (0, 1)), ((), (1, 0))]
    assert A.verify_rg_unital()
    for a in labs:
        assert A.verify_rg_bar_invariant(a), ("U1A1E7 bar", a)
        for b in labs:
            assert A.verify_rg_multiplicative(a, b), ("U1A1E7 mult", a, b)
            assert A.verify_orthonormality(a, b, 4), ("U1A1E7 ortho", a, b)
    print("  PASS: test_u1a1e7 (u(1)-gauged [A1,E7])  vac =", v10)


def test_spine_free():
    hit = sorted(m for m in sys.modules
                 if any(m == s or m.startswith(s + ".") for s in _SPINE))
    assert not hit, ("spine modules leaked into the E-type release", hit)
    print("  PASS: test_spine_free (no spine modules imported)")


def main():
    test_e6()
    test_e8()
    test_u1a1e7()
    test_spine_free()
    print("\nAll E-type (E6, E8, u(1)-gauged E7) RG-flow (Step 3) tests passed.")


if __name__ == "__main__":
    main()
