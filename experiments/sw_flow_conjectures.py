"""The remaining Seiberg-Witten RG flow conjectures, on the BPS-quiver dictionary.

The paper's "Seiberg-Witten RG flows" section states, besides the PBW
factorization conjecture (tested in a probe in the source repository and
recorded as `pronilpotent_group_conjecture.md` section 6b):

  * **Uniqueness of S** -- there is a unique `S` in `E` satisfying the BPS
    quiver constraint `S = 1 - q*sum_i X_{gamma_i} + O(q^2)`.
  * **S_cluster = S** -- when a maximal green sequence exists, the ordered
    product `S_cluster = prod_a E_q(X_{gamma(a)})` over its charges equals `S`.
    The paper flags the non-obvious half explicitly: *"Although clearly
    `S_cluster` is in `E`, it is far from obvious that it satisfies
    (eq:quiver)."*
  * **Upper cluster algebra**, **S_DT = S**, **S_CoHA = S** -- the last two need
    geometric / CoHA input that is not computable here and are not touched.

This harness tests the first two with the independent arithmetic of
a probe in the source repository -- no shared code path with `bps_factor_spectrum`,
`recursive_spectrum` or `habiro`.

What each measurement is worth
------------------------------
*Uniqueness* is approached through order-independence: `S` is built by the
forced recursion in several total orders, including orders no central charge
can produce, and the resulting ELEMENTS are compared.  The recursion itself is
not independent evidence -- it imposes the leading data by construction -- but
that the same element comes out of every order is a genuine measurement, and it
is what uniqueness needs.

*S_cluster = S* splits into two, and only the first is far from obvious:

  (a) `S_cluster` satisfies the BPS quiver constraint.  It is an ordered
      product of plain `E_q` factors, so nothing forces its `q`-expansion to
      have empty non-positive part and `q^1`-coefficient `-1` exactly on the
      nodes.  This is measured, not imposed.
  (b) given (a) and uniqueness, `S_cluster = S`.  Checked directly anyway, by
      comparing elements.

Run:  PYTHONPATH=. python a probe in the source repository [--per-file N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

sys.path.insert(0, ".")

from dictionary_loader import default_dictionary_dir  # noqa: E402

from experiments.pbw_e_group import (  # noqa: E402
    CF_ONE, CF_ZERO, PBWFailure,
    E_spin_element_cached, build_S_forced, cf_expand, cf_from_lp, cf_mul,
    deg, elt_eq, elt_mul, elt_one, forced_omega,
    key_anti_degree, key_degree_lex, key_lex_reversed, key_spin_first,
    leading_data_violations, make_key_shuffled, spec_product,
)

DEGREE_FOR_RANK = {1: 8, 2: 6, 3: 6, 4: 5, 5: 5, 6: 4, 7: 4, 8: 4}

PENTAGON = [[0, 1], [-1, 0]]
KRONECKER2 = [[0, 2], [-2, 0]]
THREE_CYCLE = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, note: str = "") -> bool:
    _results.append((name, bool(ok), note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {note}" if note else ""))
    return bool(ok)


def nodes_of(rank):
    return [tuple(1 if i == j else 0 for i in range(rank)) for j in range(rank)]


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------


def control_expansion() -> bool:
    ok = True
    # 1/(1-q^2) = 1 + q^2 + q^4 + ...
    e = cf_expand(({0: 1}, {1: 1}), 6)
    ok &= check("D1  expand 1/(1-q^2)", e == {0: 1, 2: 1, 4: 1, 6: 1}, str(e))
    # -q/(1-q^2) = -q - q^3 - q^5 ...   (the L of the forced recursion)
    e = cf_expand(({1: -1}, {1: 1}), 5)
    ok &= check("D1b expand -q/(1-q^2) = L", e == {1: -1, 3: -1, 5: -1}, str(e))
    # expansion is multiplicative: expand(a*b) agrees with expand(a)*expand(b)
    a = ({1: -1}, {1: 1})
    b = ({0: 1, 2: 3}, {2: 1})
    top = 8
    ea, eb = cf_expand(a, top), cf_expand(b, top)
    conv = {}
    for i, u in ea.items():
        for j, v in eb.items():
            if i + j <= top:
                conv[i + j] = conv.get(i + j, 0) + u * v
    conv = {k: v for k, v in conv.items() if v}
    prod = cf_expand(cf_mul(a, b), top)
    # the convolution is only correct where both factors are fully expanded;
    # compare on the common lower range
    lo = min(prod) if prod else 0
    same = all(conv.get(k, 0) == prod.get(k, 0) for k in range(lo, top + 1))
    ok &= check("D1c expansion is multiplicative", same)
    return ok


def control_forced_omega() -> bool:
    """`forced_omega` must ACHIEVE the constraint, not merely be derived for it."""
    ok = True
    L = ({1: -1}, {1: 1})            # -L = -q/(1-q^2)
    rng = random.Random(4242)
    good = 0
    trials = 40
    for _ in range(trials):
        # a random accumulated coefficient, with a genuine denominator
        num = {rng.randrange(-3, 4): rng.randrange(-4, 5) for _ in range(4)}
        num = {k: v for k, v in num.items() if v}
        if not num:
            continue
        den = {rng.choice([1, 1, 2, 3]): 1}
        f = (num, den)
        target = rng.choice([0, -1])
        f_exp = cf_expand(f, 1)
        om = forced_omega(f_exp, target)
        # total = f + (-L)*Omega
        tot = f
        from experiments.pbw_e_group import cf_add
        tot = cf_add(f, cf_mul(L, cf_from_lp(om)))
        te = cf_expand(tot, 1)
        ok_here = all(te.get(e, 0) == 0 for e in range(min(te) if te else 0, 1)) \
            and te.get(1, 0) == target
        good += ok_here
    ok &= check("D2  forced_omega achieves the constraint it was derived for",
                good == trials, f"{good}/{trials}")
    return ok


def control_pentagon_S() -> bool:
    ok = True
    B, D = PENTAGON, 6
    nodes = nodes_of(2)
    S, factors = build_S_forced(B, nodes, D, key_degree_lex)
    viol = leading_data_violations(S, nodes, D=D)
    ok &= check("D3  built S satisfies the quiver constraint (a construction "
                "guard, not evidence)", not viol, str(viol[:3]))
    # The two node factors are the answer only in the order that puts gamma_1
    # first; degree-lex puts (0,1) first and needs 7 sign-alternating factors
    # for the SAME element.  Assert both halves -- that is the order-dependence
    # of the factorisation, which is the property, not a defect.
    def key_node1_first(gamma, two_s):
        return (deg(gamma), tuple(-x for x in gamma), two_s)

    S2, f2 = build_S_forced(B, nodes, D, key_node1_first)
    ok &= check("D3b pentagon S is the two node factors in the good order",
                sorted(f2) == sorted([((0, 1), 0, 1), ((1, 0), 0, 1)]), str(f2))
    ok &= check("D3c ... and the SAME element in degree-lex, with more factors",
                elt_eq(S, S2, D) and len(factors) > len(f2),
                f"{len(factors)} vs {len(f2)} factors")
    return ok


def control_cross_check_repo() -> bool:
    """`build_S_forced` must reproduce the repo engine's `S`, same order."""
    from bps_factor_spectrum import build_spectrum_generator_from_factors
    from habiro import HabiroElement
    from laurent_poly import LaurentPoly
    from experiments.pbw_e_group import _cf_simplify

    def to_habiro(c):
        num, den = _cf_simplify(dict(c[0]), dict(c[1]))
        return HabiroElement(LaurentPoly(dict(num)), dict(den)).simplify()

    ok = True
    for label, B, D in (("pentagon", PENTAGON, 6), ("Kronecker-2", KRONECKER2, 5),
                        ("3-cycle(1,1,1)", THREE_CYCLE, 4)):
        rank = len(B)
        nodes = nodes_of(rank)
        Srepo = build_spectrum_generator_from_factors(
            [list(r) for r in B], nodes, D, piece_key=key_degree_lex)
        Srepo = {tuple(int(x) for x in k): v for k, v in Srepo.items()}
        Smine, _f = build_S_forced(B, nodes, D, key_degree_lex)
        agree, bad = True, None
        for gam in set(Smine) | set(Srepo):
            if deg(gam) > D or any(x < 0 for x in gam):
                continue
            if to_habiro(Smine.get(gam, CF_ZERO)) != \
                    Srepo.get(gam, HabiroElement.zero()).simplify():
                agree, bad = False, gam
                break
        ok &= check(f"D4  build_S_forced == the repo engine's S  [{label}]",
                    agree, "" if agree else f"first mismatch at {bad}")
    return ok


def control_negative_wrong_spec() -> bool:
    """A WRONG spec must violate the quiver constraint.

    Without this the S_cluster test is vacuous: if every ordered product of
    `E_q` factors passed, the far-from-obvious half would be measuring nothing.
    """
    ok = True
    B, D = PENTAGON, 6
    nodes = nodes_of(2)

    # POSITIVE: the pentagon's two genuine chambers -- the 2-state one and the
    # 3-state one -- must BOTH satisfy the constraint.  That is the pentagon
    # identity E(X_1)E(X_2) = E(X_2)E(X_{1+2})E(X_1), and it is the reason a
    # "non-node charge appears in the spec" is NOT by itself wrong.
    for label, spec in (("2-state chamber", [(1, 0), (0, 1)]),
                        ("3-state chamber", [(0, 1), (1, 1), (1, 0)])):
        v = leading_data_violations(spec_product(spec, B, D), nodes, D=D)
        ok &= check(f"D5a both genuine pentagon chambers are specs [{label}]",
                    not v, str(v[:2]))

    bad_specs = {
        "the reversed order": [(0, 1), (1, 0)],
        "a node repeated": [(1, 0), (1, 0), (0, 1)],
        "a node missing": [(1, 0)],
        "a spurious extra charge": [(1, 0), (1, 1), (0, 1), (2, 1)],
        "the 3-state chamber reversed": [(1, 0), (1, 1), (0, 1)],
    }
    missed = [n for n, spec in bad_specs.items()
              if not leading_data_violations(spec_product(spec, B, D), nodes, D=D)]
    ok &= check("D5b wrong specs violate the quiver constraint",
                not missed, f"missed: {missed}" if missed else
                f"{len(bad_specs)}/{len(bad_specs)} caught")

    # and a doubled factor at a node: coefficient is -2q, must be caught
    g = elt_mul(E_spin_element_cached((1, 0), 0, 2, D),
                E_spin_element_cached((0, 1), 0, 1, D), B, D)
    ok &= check("D5c a doubled node factor is caught",
                bool(leading_data_violations(g, nodes, D=D)))
    return ok


def run_controls() -> bool:
    print("CONTROLS")
    allok = True
    for fn in (control_expansion, control_forced_omega, control_pentagon_S,
               control_negative_wrong_spec, control_cross_check_repo):
        allok &= fn()
    return allok


# --------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------


def orders_for(rank, D, rng):
    return [("degree-lex", key_degree_lex),
            ("anti-degree", key_anti_degree),
            ("lex-reversed", key_lex_reversed),
            ("spin-first", key_spin_first),
            ("shuffled", make_key_shuffled(rank, D, rng, 4))]


def sweep_entry(name, B, specs, rng):
    rank = len(B)
    D = DEGREE_FOR_RANK.get(rank, 4)
    nodes = nodes_of(rank)
    nodeset = set(nodes)
    out = {"name": name, "rank": rank, "D": D, "orders": 0, "orders_agree": 0,
           "unique_ok": None, "n_factors": [], "note": "",
           "specs_tested": 0, "specs_leading_ok": 0, "specs_equal_S_ok": 0,
           "specs_nontrivial": 0}

    elements = []
    for oname, key in orders_for(rank, D, rng):
        try:
            S, factors = build_S_forced(B, nodes, D, key)
        except PBWFailure as e:
            out["note"] = f"build failed in {oname}: {e}"
            return out
        elements.append((oname, S))
        out["n_factors"].append(len(factors))
        out["orders"] += 1

    S0 = elements[0][1]
    out["orders_agree"] = sum(1 for _n, S in elements if elt_eq(S, S0, D))
    out["unique_ok"] = out["orders_agree"] == len(elements)

    # Every KNOWN chamber, not just one: the conjecture quantifies over maximal
    # green sequences, so distinct chambers of the same quiver must all give
    # the same S.
    seen = set()
    for spec in specs:
        if not spec:
            continue
        spec_t = tuple(tuple(int(x) for x in g) for g in spec)
        if spec_t in seen:
            continue
        seen.add(spec_t)
        if any(any(x < 0 for x in g) for g in spec_t):
            out["note"] = "a spec has a negative charge; skipped"
            continue
        Sc = spec_product(list(spec_t), B, D)
        out["specs_tested"] += 1
        # "far from obvious" bites where the spec is not just the node charges
        if any(g not in nodeset for g in spec_t):
            out["specs_nontrivial"] += 1
        out["specs_leading_ok"] += not leading_data_violations(Sc, nodes, D=D)
        out["specs_equal_S_ok"] += elt_eq(Sc, S0, D)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-file", type=int, default=0, help="0 = every entry")
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("SEIBERG-WITTEN RG FLOW CONJECTURES — uniqueness of S, and "
          "S_cluster = S")
    print("=" * 78)
    print(f"cone degree by rank: {DEGREE_FOR_RANK}")
    print("orders per quiver: degree-lex, anti-degree, lex-reversed, "
          "spin-first, shuffled")
    print()
    if not run_controls():
        print("\ncontrols failed — sweep not run")
        return 1
    print()

    root = default_dictionary_dir()
    tot = {"entries": 0, "uniq": 0, "uniq_n": 0,
           "clead": 0, "clead_n": 0, "ceq": 0, "ceq_n": 0}
    failures = []
    t0 = time.time()
    for r in range(1, 9):
        path = root / f"n_{r:03d}.json"
        if not path.exists():
            continue
        entries = json.load(open(path))
        if args.per_file and len(entries) > args.per_file:
            entries = rng.sample(entries, args.per_file)
        rs = {"n": 0, "uniq": 0, "uniq_n": 0, "clead": 0, "clead_n": 0,
              "ceq": 0, "ceq_n": 0, "skipped": 0, "nontriv": 0}
        tr = time.time()
        for e in entries:
            B = e["exchange"]
            if len(B) != r:
                continue
            specs = [e.get("spec")] + list(e.get("known_specs") or [])
            s = sweep_entry(e.get("name", "?"), B, specs, rng)
            if s["unique_ok"] is None:
                rs["skipped"] += 1
                failures.append({"entry": s["name"], "reason": s["note"],
                                 "lattice": B})
                continue
            rs["n"] += 1
            rs["uniq_n"] += 1
            rs["uniq"] += bool(s["unique_ok"])
            if not s["unique_ok"]:
                failures.append({"entry": s["name"], "lattice": B,
                                 "reason": "S depends on the order",
                                 "detail": s["orders_agree"]})
            rs["clead_n"] += s["specs_tested"]
            rs["clead"] += s["specs_leading_ok"]
            rs["ceq_n"] += s["specs_tested"]
            rs["ceq"] += s["specs_equal_S_ok"]
            rs["nontriv"] += s["specs_nontrivial"]
            if s["specs_leading_ok"] != s["specs_tested"]:
                failures.append({"entry": s["name"], "lattice": B,
                                 "reason": "S_cluster violates the quiver "
                                           "constraint"})
            if s["specs_equal_S_ok"] != s["specs_tested"]:
                failures.append({"entry": s["name"], "lattice": B,
                                 "reason": "S_cluster != S"})
        for k in ("uniq", "uniq_n", "clead", "clead_n", "ceq", "ceq_n"):
            tot[k] += rs[k]
        tot["entries"] += rs["n"]
        tot["nontriv"] = tot.get("nontriv", 0) + rs["nontriv"]
        print(f"rank {r}: uniqueness {rs['uniq']:4d}/{rs['uniq_n']:4d}   "
              f"S_cluster satisfies the constraint {rs['clead']:5d}/"
              f"{rs['clead_n']:5d}   S_cluster == S {rs['ceq']:5d}/"
              f"{rs['ceq_n']:5d}   ({rs['n']} entries, "
              f"{rs['nontriv']} specs beyond the nodes, D="
              f"{DEGREE_FOR_RANK.get(r)}, {time.time()-tr:.0f}s"
              + (f", {rs['skipped']} skipped" if rs["skipped"] else "") + ")")
        sys.stdout.flush()

    print()
    print("=" * 78)
    print(f"uniqueness of S (same element in every order): "
          f"{tot['uniq']}/{tot['uniq_n']}")
    print(f"S_cluster satisfies the BPS quiver constraint:  "
          f"{tot['clead']}/{tot['clead_n']}   <-- the far-from-obvious half")
    print(f"S_cluster == S:                                 "
          f"{tot['ceq']}/{tot['ceq_n']}")
    print(f"({tot['entries']} dictionary lattices, "
          f"{tot.get('nontriv', 0)} of the specs tested contain a charge that "
          f"is NOT a node — where the claim has content, "
          f"{time.time()-t0:.0f}s)")
    if failures:
        print(f"\nFAILURES / SKIPS: {len(failures)}")
        for f in failures[:20]:
            print(f"  {f['entry'][:50]}: {f['reason']}")
        if args.out:
            json.dump(failures, open(args.out, "w"), indent=2)
            print(f"  (full detail in {args.out})")
        hard = [f for f in failures if "skipped" not in f["reason"]]
        return 1 if hard else 0
    print("\nNO FAILURES.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
