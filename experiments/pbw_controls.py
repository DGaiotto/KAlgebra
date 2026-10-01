"""Controls for a probe in the source repository — run these before any sweep.

the design notes "Positive controls before trusting any scan": assert the known answer
passes first, and abort rather than report if a control fails.  Equally
important here are the NEGATIVE controls: a test of the PBW factorization
conjecture that accepts everything measures nothing, and the paper hands us the
sharpest negative control there is --

    "It is easy to argue that every element g in E satisfies g^{-1}(q) = g(q^{-1}),
     but E is smaller than the group of elements which satisfy that constraint.
     For example, it does not include E_{q^2}(x)."

So `E_{q^2}(X_gamma)` must FAIL to factorize, while satisfying bar-unitarity.

Run:  PYTHONPATH=. python a probe in the source repository
"""

from __future__ import annotations

import random
import sys

sys.path.insert(0, ".")

from experiments.pbw_e_group import (  # noqa: E402
    CF_ONE, CF_ZERO, Charge, PBWFailure,
    E_spin_element, E_spin_linear_coeff, E_spin_series,
    _cf_simplify, cf_add, cf_bar, cf_is_zero, cf_mul, cf_to_lp,
    chi, cone_points, deg, decompose_into_chi,
    elt_bar, elt_eq, elt_inv, elt_mul, elt_one, elt_truncate,
    key_anti_degree, key_degree_lex, key_lex_reversed, key_spin_first,
    make_key_phase, make_key_shuffled,
    lp_mul, ordered_product, pbw_factorize,
)

PENTAGON = [[0, 1], [-1, 0]]
KRONECKER2 = [[0, 2], [-2, 0]]
KRONECKER3 = [[0, 3], [-3, 0]]
THREE_CYCLE = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, note: str = "") -> bool:
    _results.append((name, bool(ok), note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {note}" if note else ""))
    return bool(ok)


# --------------------------------------------------------------------------
# C1.  The two routes to the x-linear coefficient of E^{(s)} agree.
# --------------------------------------------------------------------------


def control_linear_coefficient() -> bool:
    ok = True
    for two_s in range(0, 9):
        ser = E_spin_series(two_s, 2)
        closed = E_spin_linear_coeff(two_s)
        a = _cf_simplify(dict(ser[1][0]), dict(ser[1][1]))
        b = _cf_simplify(dict(closed[0]), dict(closed[1]))
        # compare as rational functions: a.num * b.den == b.num * a.den
        from experiments.pbw_e_group import _den_expand
        lhs = lp_mul(a[0], _den_expand(b[1]))
        rhs = lp_mul(b[0], _den_expand(a[1]))
        ok &= check(f"C1  E^(s) linear coeff == -q*chi_s/(1-q^2)  [2s={two_s}]",
                    lhs == rhs)
    return ok


# --------------------------------------------------------------------------
# C2.  A single generator factorizes to itself.
# --------------------------------------------------------------------------


def control_single_generator() -> bool:
    ok = True
    B, D = PENTAGON, 6
    for gamma in [(1, 0), (0, 1), (1, 1), (2, 1)]:
        for two_s in (0, 1, 2, 3):
            for power in (1, -1, 2, -3):
                g = E_spin_element(gamma, two_s, power, D)
                try:
                    factors, om = pbw_factorize(g, B, D, key_degree_lex)
                except PBWFailure as e:
                    ok &= check(f"C2  single generator {gamma} 2s={two_s} "
                                f"^{power}", False, str(e))
                    continue
                expect = {(gamma, two_s): power}
                ok &= check(f"C2  single generator {gamma} 2s={two_s} ^{power}",
                            om == expect, f"got {om}" if om != expect else "")
    return ok


# --------------------------------------------------------------------------
# C3.  Bar-unitarity: every element of E satisfies bar(g) = g^{-1}.
#      (Proved in the repo for any ordered product; here it validates that this
#      module's E^{(s)} really are the paper's generators.)
# --------------------------------------------------------------------------


def random_word(B, D, rng, *, n_factors=4, max_two_s=3, max_deg=2,
                allow_negative=True):
    """A random word in the generators E^{(s)}(X_gamma)^{+-1}."""
    rank = len(B)
    charges = [c for d in range(1, max_deg + 1) for c in cone_points(rank, d)]
    word = []
    for _ in range(n_factors):
        gamma = rng.choice(charges)
        two_s = rng.randrange(0, max_two_s + 1)
        p = rng.choice([1, -1, 2, -2] if allow_negative else [1, 2])
        word.append((gamma, two_s, p))
    g = elt_one(rank)
    for (gamma, two_s, p) in word:
        g = elt_mul(g, E_spin_element(gamma, two_s, p, D), B, D)
    return g, word


def control_bar_unitary() -> bool:
    ok = True
    rng = random.Random(20260904)
    for B, D, label in ((PENTAGON, 6, "pentagon"), (KRONECKER2, 5, "Kronecker-2"),
                        (THREE_CYCLE, 4, "3-cycle")):
        rank = len(B)
        good = 0
        for _ in range(6):
            g, _w = random_word(B, D, rng)
            ginv = elt_inv(g, B, D, rank)
            good += elt_eq(elt_bar(g), ginv, D)
        ok &= check(f"C3  bar(g) == g^-1 on random words  [{label}]", good == 6,
                    f"{good}/6")
    return ok


# --------------------------------------------------------------------------
# C4.  Round trip: factorize a random word, rebuild, compare exactly.
# --------------------------------------------------------------------------


def control_round_trip() -> bool:
    ok = True
    rng = random.Random(9091)
    for B, D, label in ((PENTAGON, 6, "pentagon"), (KRONECKER2, 5, "Kronecker-2"),
                        (KRONECKER3, 4, "Kronecker-3"),
                        (THREE_CYCLE, 4, "3-cycle")):
        rank = len(B)
        good = 0
        trials = 5
        for _ in range(trials):
            g, _w = random_word(B, D, rng)
            try:
                factors, om = pbw_factorize(g, B, D, key_degree_lex)
            except PBWFailure:
                continue
            rebuilt = ordered_product(factors, B, D, rank)
            good += elt_eq(rebuilt, g, D)
        ok &= check(f"C4  factorize->rebuild is exact  [{label}]", good == trials,
                    f"{good}/{trials}")
    return ok


# --------------------------------------------------------------------------
# C5.  NEGATIVE control — E_{q^2}(x) is bar-unitary but NOT in E.
# --------------------------------------------------------------------------


def E_q2_element(gamma: Charge, D: int) -> dict:
    """`E_{q^2}(X_gamma) = sum_n (-q^2)^n / (q^4; q^4)_n X_{n gamma}`.

    The paper's own example of a bar-unitary element OUTSIDE `E`.
    """
    dg = deg(gamma)
    N = D // dg
    rank = len(gamma)
    out = {(0,) * rank: CF_ONE}
    for n in range(1, N + 1):
        num = {2 * n: (-1) ** n}
        den = {2 * k: 1 for k in range(1, n + 1)}   # (1-q^{4k}) = (1-q^{2*(2k)})
        out[tuple(n * x for x in gamma)] = _cf_simplify(num, den)
    return out


def control_negative_Eq2() -> bool:
    """`E_{q^2}` must be bar-unitary AND must fail to factorize."""
    ok = True
    B, D = PENTAGON, 6
    rank = 2
    g = E_q2_element((1, 0), D)
    unitary = elt_eq(elt_bar(g), elt_inv(g, B, D, rank), D)
    ok &= check("C5a E_{q^2} IS bar-unitary (so bar-unitarity is not a "
                "membership test)", unitary)
    try:
        pbw_factorize(g, B, D, key_degree_lex)
        ok &= check("C5b E_{q^2} FAILS to factorize (it is not in E)", False,
                    "it factorized — the test would accept non-members")
    except PBWFailure as e:
        ok &= check("C5b E_{q^2} FAILS to factorize (it is not in E)", True,
                    f"{e.reason} at {e.gamma}")
    return ok


# --------------------------------------------------------------------------
# C6.  NEGATIVE control — perturb one coefficient off the L*P lattice.
# --------------------------------------------------------------------------


def control_negative_perturbation() -> bool:
    ok = True
    rng = random.Random(31337)
    B, D = PENTAGON, 6
    g, _w = random_word(B, D, rng)
    try:
        pbw_factorize(g, B, D, key_degree_lex)
        base_ok = True
    except PBWFailure:
        base_ok = False
    ok &= check("C6a the unperturbed word factorizes (else the control is "
                "vacuous)", base_ok)

    from experiments.pbw_e_group import cf_from_lp
    caught = 0
    for bump_charge in [(1, 0), (0, 1), (1, 1)]:
        h = dict(g)
        h[bump_charge] = cf_add(h.get(bump_charge, CF_ZERO), cf_from_lp({0: 1}))
        try:
            pbw_factorize(h, B, D, key_degree_lex)
        except PBWFailure:
            caught += 1
    ok &= check("C6b a q^0 bump off the lattice FAILS", caught == 3,
                f"{caught}/3 caught")
    return ok


# --------------------------------------------------------------------------
# C7.  Cross-check against the repo engine: this module's arithmetic must
#      reproduce `bps_factor_spectrum`'s S exactly, coefficient by coefficient.
# --------------------------------------------------------------------------


def _mine_to_habiro(c):
    from habiro import HabiroElement
    from laurent_poly import LaurentPoly
    num, den = _cf_simplify(dict(c[0]), dict(c[1]))
    return HabiroElement(LaurentPoly(dict(num)), dict(den)).simplify()


def control_cross_check_repo() -> bool:
    """Cross-check this module's arithmetic against `bps_factor_spectrum`.

    The repo engine is asked for `S` AND for its multiplicities in ONE named
    total order (`piece_key`, the contract's own most general form).  Those
    multiplicities are then composed HERE, with this module's independent
    `E^{(s)}` and quantum-torus arithmetic, in that SAME order, and the two
    elements are compared inside the repo's ring (`HabiroElement.__eq__`, exact
    algebraic equality).

    Order matters: `S` is order-independent but `Omega` is not, so composing
    one order's multiplicities in another order is simply a different element.

    Finding to record (not a workaround): `LaurentPoly` exposes no public
    accessor for its coefficient dict, so the bridge can only run
    mine -> repo.  Hence this control compares elements rather than
    re-factorizing the repo's `S` here.
    """
    from bps_factor_spectrum import (build_spectrum_generator_from_factors,
                                     spin_decompose)
    from habiro import HabiroElement

    ok = True
    cases = [("pentagon", PENTAGON, 6),
             ("Kronecker-2", KRONECKER2, 5),
             ("Kronecker-3", KRONECKER3, 4),
             ("3-cycle(1,1,1)", THREE_CYCLE, 4)]
    for label, B, D in cases:
        rank = len(B)
        nodes = [tuple(1 if i == j else 0 for i in range(rank))
                 for j in range(rank)]
        S, mult = build_spectrum_generator_from_factors(
            [list(r) for r in B], nodes, D,
            piece_key=key_degree_lex, with_multiplicities=True)

        pieces = []
        for gam, om in mult.items():
            gam = tuple(int(x) for x in gam)
            for two_s, a in spin_decompose(om).items():
                if a:
                    pieces.append((gam, two_s, a))
        pieces.sort(key=lambda p: key_degree_lex(p[0], p[1]))
        built = ordered_product(pieces, B, D, rank)

        Srepo = {tuple(int(x) for x in k): v for k, v in S.items()}
        agree, first_bad = True, None
        for gam in set(built) | set(Srepo):
            if deg(gam) > D or any(x < 0 for x in gam):
                continue
            a = _mine_to_habiro(built.get(gam, CF_ZERO))
            b = Srepo.get(gam, HabiroElement.zero()).simplify()
            if a != b:
                agree, first_bad = False, gam
                break
        ok &= check(f"C7  reproduces the repo engine's S  [{label}]", agree,
                    "" if agree else f"first mismatch at {first_bad}")

        if label == "pentagon":
            # The two node factors are the answer in the ENGINE'S DEFAULT
            # (strip / phase) order only.  In degree-lex the same element needs
            # a longer, sign-alternating content -- which is the headline
            # property, so assert BOTH halves rather than the tidy one.
            S2, mult2 = build_spectrum_generator_from_factors(
                [list(r) for r in B], nodes, D, with_multiplicities=True)
            strip = {(tuple(int(x) for x in gam), ts): a
                     for gam, om in mult2.items()
                     for ts, a in spin_decompose(om).items() if a}
            ok &= check("C7b pentagon S is exactly the two node factors in the "
                        "engine's default (strip) order",
                        strip == {((1, 0), 0): 1, ((0, 1), 0): 1}, f"got {strip}")
            lex = {(g_, s_): a_ for (g_, s_, a_) in pieces}
            ok &= check("C7c ... and a DIFFERENT content in degree-lex, for the "
                        "same element (order-independence of S)",
                        lex != strip and len(lex) > len(strip),
                        f"{len(lex)} factors vs {len(strip)}")
    return ok


# --------------------------------------------------------------------------
# C8.  Order-independence of the ELEMENT (the factorisation may differ).
# --------------------------------------------------------------------------


def control_order_independence() -> bool:
    ok = True
    rng = random.Random(5150)
    B, D = PENTAGON, 6
    rank = 2
    g, _w = random_word(B, D, rng)
    keys = {"degree-lex": key_degree_lex, "anti-degree": key_anti_degree,
            "lex-reversed": key_lex_reversed, "spin-first": key_spin_first,
            "shuffled": make_key_shuffled(rank, D, random.Random(7), 3)}
    counts = {}
    for name, k in keys.items():
        try:
            factors, om = pbw_factorize(g, B, D, k)
        except PBWFailure as e:
            ok &= check(f"C8  refactorizes in order '{name}'", False, str(e))
            continue
        rebuilt = ordered_product(factors, B, D, rank)
        good = elt_eq(rebuilt, g, D)
        counts[name] = len(factors)
        ok &= check(f"C8  refactorizes in order '{name}'", good,
                    f"{len(factors)} factors")
    moved = len(set(counts.values())) > 1
    ok &= check("C8b the factorisation MOVES with the order (else the order "
                "axis is vacuous)", moved, f"factor counts {counts}")
    return ok


def main() -> int:
    print("PBW / E-group CONTROLS")
    print("=" * 72)
    allok = True
    for fn in (control_linear_coefficient, control_single_generator,
               control_bar_unitary, control_round_trip,
               control_negative_Eq2, control_negative_perturbation,
               control_order_independence, control_cross_check_repo):
        print(f"\n--- {fn.__name__}")
        try:
            allok &= fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            allok &= check(fn.__name__, False, f"EXCEPTION {e}")
    print("\n" + "=" * 72)
    npass = sum(1 for _n, o, _d in _results if o)
    print(f"{npass}/{len(_results)} controls passed")
    if not allok:
        print("\nCONTROLS FAILED — do not run or report the sweep.")
        return 1
    print("controls green — the sweep may be trusted to the extent these cover it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
