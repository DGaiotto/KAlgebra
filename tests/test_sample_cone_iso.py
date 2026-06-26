"""Runnable self-test for the Step-1 (sample) ↔ Step-2 (cone)
``KAlgebraIso`` correspondences.

Pure Python 3, no third-party dependencies, and **engine-free**: no
realisation-spine modules (no BPS / RG / quantum-torus backend) are
imported, either here or anywhere on the import chain of the iso builders
and the algebras they bridge.

It builds and fully certifies, via ``KAlgebraIso.verify_all`` in BOTH
directions, the following Sample↔Cone isomorphisms:

  * pentagon : ``FinitePentagonKAlgebra`` (cone) ↔ ``PentagonSampleKAlgebra``
               (``K_𝖖([A_1, A_2])`` — A_2 Argyres-Douglas / Yang-Lee)
  * A1A1     : ``U1SquareKAlg`` (cone) ↔ ``SQED1SampleKAlgebra``
               (SQED N_f=1 = U(1)-gauged ``[A_1, A_1]``)
  * A1D2     : ``U1A1D2ConeKAlgebra`` (cone) ↔ ``SQED2SampleKAlgebra``
               (SQED N_f=2 = U(1)-gauged ``[A_1, D_2]`` = ``U_𝖖(𝔰𝔩₂)``,
               SU(2)-flavoured)

``verify_all`` checks, in both directions:

  * ``unit``               : map(1) = 1, inverse(1) = 1
  * ``round_trip``         : inverse(map(a)) = a, map(inverse(b)) = b
  * ``multiplicative``     : map(a·b) = map(a)·map(b)
  * ``rho_equivariant``    : map(ρ a) = ρ(map a)
  * ``trace_equivariant``  : Tr(a) = Tr(map a)  (truncated)

Run (BOTH packages on the path)::

    PYTHONPATH=/abs/.../export/KAlgebra:/abs/.../export/ConeKAlgebra \\
        python test_sample_cone_iso.py
"""
from __future__ import annotations

import os
import sys
import traceback

# Make both sibling export packages importable (the documented contract is
# to set PYTHONPATH to both dirs; this is belt-and-braces for a checkout).
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPORT_ROOT = os.path.dirname(_HERE)
for _sib in ("KAlgebra", "ConeKAlgebra"):
    _p = os.path.join(_EXPORT_ROOT, _sib)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from laurent_poly import LaurentPoly
from kalgebra import Element
from kalgebra_iso import _mul
from pentagon_sample_cone_iso import (
    build_pentagon_sample_cone_iso,
    PENTAGON_SIGMA,
)
from u1square_sample_cone_iso import build_u1square_sqed1_sample_cone_iso
from u1a1d2_sqed2_sample_cone_iso import build_u1a1d2_sqed2_sample_cone_iso

_ONE = LaurentPoly.one()

# q-order window for the trace comparison.
TRACE_K = 12

_ORDER = ["unit", "round_trip", "multiplicative",
          "rho_equivariant", "trace_equivariant"]


def _report(title: str, res: dict) -> list:
    """Print a verify_all summary and return the list of failed checks."""
    print(f"\n  [{title}] verify_all (both directions):")
    for k in _ORDER:
        print(f"     {k:18s}: {'PASS' if res[k] else 'FAIL'}")
    return [k for k in _ORDER if not res[k]]


# --------------------------------------------------------------------------
# pentagon : FinitePentagonKAlgebra (cone) ↔ PentagonSampleKAlgebra
# --------------------------------------------------------------------------


def _pentagon_samples(iso):
    """Source/target sample sets and generator pairs for the pentagon iso."""
    cone = iso.source
    cd = cone.cone_data()
    mult_gens = list(cd.mult_gens())

    src_gen = [
        Element({cd.from_cone_label(frozenset({g}), {g: 1}): _ONE})
        for g in mult_gens
    ]
    src_samples = [Element({cone.identity(): _ONE})] + list(src_gen)
    for c in cd.cones():
        cl = sorted(c)
        native = cd.from_cone_label(frozenset(cl), {g: 1 for g in cl})
        src_samples.append(Element({native: _ONE}))
    for g in mult_gens[:2]:
        native = cd.from_cone_label(frozenset({g}), {g: 2})
        src_samples.append(Element({native: _ONE}))

    tgt_gen = [Element({(i, 1, 0): _ONE}) for i in range(5)]
    tgt_samples = [Element({iso.target.identity(): _ONE})] + list(tgt_gen)
    for lbl in [(0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 1, 1),
                (0, 2, 0), (0, 2, 1)]:
        tgt_samples.append(Element({lbl: _ONE}))

    src_pairs = [(a, b) for a in src_gen for b in src_gen]
    tgt_pairs = [(a, b) for a in tgt_gen for b in tgt_gen]
    return src_samples, tgt_samples, src_pairs, tgt_pairs


def run_pentagon() -> bool:
    print("Pentagon cone ↔ sample KAlgebraIso")
    print("-" * 68)
    iso = build_pentagon_sample_cone_iso()
    print(f"  iso         : {iso!r}")
    print(f"  source(cone): {type(iso.source).__name__}")
    print(f"  target      : {type(iso.target).__name__}")
    print(f"  mult-gen map: cone g -> sample L_i  =  {PENTAGON_SIGMA}")

    src_samples, tgt_samples, src_pairs, tgt_pairs = _pentagon_samples(iso)
    print(f"  samples: {len(src_samples)} source, {len(tgt_samples)} target;"
          f"  pairs: {len(src_pairs)} source, {len(tgt_pairs)} target")
    res = iso.verify_all(
        src_samples, tgt_samples, src_pairs, tgt_pairs, trace_K=TRACE_K)
    failed = _report("pentagon", res)

    # Belt-and-braces cross-product spot checks in both directions.
    cone, cd = iso.source, iso.source.cone_data()
    g0 = Element({cd.from_cone_label(frozenset({0}), {0: 1}): _ONE})
    g1 = Element({cd.from_cone_label(frozenset({1}), {1: 1}): _ONE})
    assert iso.map(_mul(cone, g0, g1)) == _mul(iso.target, iso.map(g0), iso.map(g1))
    L0 = Element({(0, 1, 0): _ONE})
    L4 = Element({(4, 1, 0): _ONE})   # = L_{-1}; L_4·L_0 hits a Plücker move
    assert (iso.inverse(_mul(iso.target, L4, L0))
            == _mul(cone, iso.inverse(L4), iso.inverse(L0)))
    return not failed


# --------------------------------------------------------------------------
# A1A1 : U1SquareKAlg (cone) ↔ SQED1SampleKAlgebra
# --------------------------------------------------------------------------


def _u1square_samples(iso):
    """Source/target sample sets and pairs for the A1A1 iso.

    Identity-on-labels: both sides use the same ``(m, n)`` charge labels,
    so source and target sample sets are the *same* Elements.  We use a
    grid spanning all sign sectors (both cones, the v-Laurent boundary, the
    u^± boundary, and mixed two-gen words) plus squared/higher powers.
    """
    labs = [(0, 0),
            (1, 0), (-1, 0), (0, 1), (0, -1),          # single-gen rays
            (1, 1), (1, -1), (-1, 1), (-1, -1),        # both cones, 2-gen
            (2, 0), (-2, 0), (0, 2), (0, -2),          # squared rays
            (2, 1), (2, -1), (-2, 3), (3, -2), (-3, -1)]  # higher words
    elems = [Element({l: _ONE}) for l in labs]
    # pairs: every ordered pair among the small ray set (exercises the
    # u_+·u_- cross-product / Plücker in both presentations).
    ray = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
    rel = [Element({l: _ONE}) for l in ray]
    pairs = [(a, b) for a in rel for b in rel]
    return elems, elems, pairs, pairs


def run_u1square() -> bool:
    print("\nA1A1 cone ↔ sample KAlgebraIso")
    print("-" * 68)
    iso = build_u1square_sqed1_sample_cone_iso()
    print(f"  iso         : {iso!r}")
    print(f"  source(cone): {type(iso.source).__name__}")
    print(f"  target      : {type(iso.target).__name__}")
    print("  correspondence: identity on (m, n) labels")

    src_samples, tgt_samples, src_pairs, tgt_pairs = _u1square_samples(iso)
    print(f"  samples: {len(src_samples)} each side;"
          f"  pairs: {len(src_pairs)} each side")
    res = iso.verify_all(
        src_samples, tgt_samples, src_pairs, tgt_pairs, trace_K=TRACE_K)
    failed = _report("A1A1", res)

    # Belt-and-braces: the u_+·u_- cross-product = 1 + 𝖖 v, both directions.
    up = Element({(1, 0): _ONE})
    un = Element({(-1, 0): _ONE})
    assert iso.map(_mul(iso.source, up, un)) == _mul(iso.target, iso.map(up), iso.map(un))
    assert iso.inverse(_mul(iso.target, up, un)) == _mul(iso.source, iso.inverse(up), iso.inverse(un))
    return not failed


# --------------------------------------------------------------------------
# A1D2 : U1A1D2ConeKAlgebra (cone) ↔ SQED2SampleKAlgebra
# --------------------------------------------------------------------------


def _u1a1d2_samples(iso):
    """Source/target sample sets and pairs for the A1D2 (SU(2)-flavoured
    SQED₂ = U_𝖖(𝔰𝔩₂)) iso.

    The cone side uses ``(m, n, k)`` labels (gauge ``(m, n)`` + SU(2) spin
    ``k``); the target ``(gauge, k)`` samples are the forward images, so the
    two sides are the *same* algebra elements under the relabeling
    bijection.  The grid spans the E/F charge sectors, the K Laurent
    boundary, χ-dressed generators, and squared / mixed-K powers (exercising
    the ``E·F`` cross-product / ``U_𝖖(𝔰𝔩₂)`` straightening + the SU(2)
    Clebsch-Gordan spin fusion).
    """
    cone_labels = [
        (0, 0, 0),          # identity
        (1, 0, 0),          # E
        (-1, 0, 0),         # F
        (0, 1, 0),          # K
        (0, -1, 0),         # K^{-1}
        (0, 0, 1),          # χ_1
        (1, 1, 0),          # E·K  (E_{1,1})
        (-1, 1, 0),         # F·K  (F_{1,1})
        (1, -1, 0),         # E_{1,-1}
        (2, 0, 0),          # E²
        (-2, 0, 0),         # F²
        (1, 0, 1),          # E·χ_1
        (-1, 0, 1),         # F·χ_1
        (0, 2, 0),          # K²
        (0, 0, 2),          # χ_2
        (2, 1, 1),          # E²·K·χ_1
    ]
    src_samples = [Element({l: _ONE}) for l in cone_labels]
    tgt_samples = [iso.map(e) for e in src_samples]

    # length-2 products among the generators E, F, K, K^{-1}, χ_1 ...
    gen_labels = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
    src_gen = [Element({l: _ONE}) for l in gen_labels]
    tgt_gen = [iso.map(e) for e in src_gen]
    src_pairs = [(a, b) for a in src_gen for b in src_gen]
    tgt_pairs = [(a, b) for a in tgt_gen for b in tgt_gen]
    # ... plus higher pairs to stress E²F²/E³F³ + spin fusion.
    for a, b in [((2, 0, 0), (-2, 0, 0)), ((3, 0, 1), (-3, 0, 2)),
                 ((1, 1, 0), (-1, -1, 1)), ((2, -1, 2), (-1, 0, 1))]:
        ea, eb = Element({a: _ONE}), Element({b: _ONE})
        src_pairs.append((ea, eb))
        tgt_pairs.append((iso.map(ea), iso.map(eb)))
    return src_samples, tgt_samples, src_pairs, tgt_pairs


def run_u1a1d2() -> bool:
    print("\nA1D2 cone ↔ sample KAlgebraIso")
    print("-" * 68)
    iso = build_u1a1d2_sqed2_sample_cone_iso()
    print(f"  iso         : {iso!r}")
    print(f"  source(cone): {type(iso.source).__name__}")
    print(f"  target      : {type(iso.target).__name__}")
    print("  correspondence: (m, n, k) ↔ (gauge, k)  [E_{m,n}/F_{-m,n}/Kⁿ · χ_k]")

    src_samples, tgt_samples, src_pairs, tgt_pairs = _u1a1d2_samples(iso)
    print(f"  samples: {len(src_samples)} each side;"
          f"  pairs: {len(src_pairs)} each side")
    res = iso.verify_all(
        src_samples, tgt_samples, src_pairs, tgt_pairs, trace_K=TRACE_K)
    failed = _report("A1D2", res)

    # Belt-and-braces: the E·F cross-product (χ_1 + 𝖖K + 𝖖⁻¹K⁻¹), both
    # directions — exercises the χ_1 daughter + SU(2) fusion through the iso.
    Ec = Element({(1, 0, 0): _ONE})
    Fc = Element({(-1, 0, 0): _ONE})
    assert iso.map(_mul(iso.source, Ec, Fc)) == _mul(iso.target, iso.map(Ec), iso.map(Fc))
    assert (iso.inverse(_mul(iso.target, iso.map(Ec), iso.map(Fc)))
            == _mul(iso.source, Ec, Fc))
    return not failed


def main() -> None:
    print("Sample ↔ Cone KAlgebraIso self-test")
    print("=" * 68)
    ok_pent = run_pentagon()
    ok_a1a1 = run_u1square()
    ok_a1d2 = run_u1a1d2()

    print("\n" + "=" * 68)
    assert ok_pent, "pentagon verify_all FAILED"
    assert ok_a1a1, "A1A1 verify_all FAILED"
    assert ok_a1d2, "A1D2 verify_all FAILED"
    print(f"PASS  pentagon cone ↔ sample  +  A1A1 cone ↔ sample  +  A1D2 cone "
          f"↔ sample  KAlgebraIso certified: unit / round-trip / "
          f"multiplicative (generators + all pairs) / ρ-equivariant / "
          f"trace-equivariant (to q^{TRACE_K}), both directions.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nFAIL")
        traceback.print_exc()
        raise SystemExit(1)
