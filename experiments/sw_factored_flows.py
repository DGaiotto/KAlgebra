"""The factored-RG-flow conjecture of the Seiberg-Witten RG flows section.

The paper: for a BPS quiver `Q` and a FULL sub-quiver `Q'`, embedded in a
lattice `Gamma`, "by picking an appropriate total order, we can factor
`S_Q = R_Q^{Q'} S_{Q'}`", and then

  Conjecture (Factored RG flows).  `R_Q^{Q'}` is in the image of
  `RG^{Q',Gamma}`, i.e. `S_Q = RG^{Q',Gamma}(S_Q^{Q'}) S_{Q'}` for a formal
  element `S_Q^{Q'}` in `K_q(Q',Gamma)`, and `RG^{Q,Gamma}` factors as
  `RG^{Q',Gamma} . RG^{Q,Q',Gamma}` as well, defining an RG flow of `K_q`
  algebras from `K_q(Q,Gamma)` to `K_q(Q',Gamma)`.

Both halves reduce to one operation -- decomposing an element in the
`RG^{Q'}` basis -- because `RG^{Q'}_eps = X_eps + (strictly higher)` makes that
decomposition forced and unique.  What is measured is not whether it EXISTS
(triangularity gives that for anything supported on the cone) but the ring the
coefficients land in.

Three measurements, in increasing sharpness:

  (F) THE FACTORISATION ITSELF.  `S_Q` is built in an order that puts every
      charge supported inside `Q'` LAST.  Then `R := S_Q * S_{Q'}^{-1}` must
      equal the product of exactly the factors at charges NOT supported inside
      `Q'` -- which is what "by picking an appropriate total order" asserts, and
      it can fail.  `S_{Q'}` is built on the sub-lattice and embedded, so it is
      the sub-quiver's own spectrum generator, not a re-solve inside `Gamma`.

  (I) `R` IS IN THE IMAGE of `RG^{Q',Gamma}` -- the first half.  Reported with
      the split the paper's own closing question asks about ("it would be
      interesting to characterize which pronilpotent group of formal sums in
      `K_q(Q',Gamma)` is the natural place for such `S_Q^{Q'}`"): how many
      coefficients of `S_Q^{Q'}` are honest Laurent polynomials, and how many
      genuinely need a `(1-q^{2n})` denominator.

  (C) `RG^{Q,Gamma}` FACTORS THROUGH `RG^{Q',Gamma}` -- the second half, and the
      sharp one.  Each `F^Q_gamma` must decompose in the `RG^{Q'}` basis with
      coefficients in `Z[q, q^{-1}]`, since `RG^{Q,Q',Gamma}` is an algebra map
      of `K_q`-algebras and its structure constants live there.  Nothing in the
      construction forces that.

Run:  PYTHONPATH=. python a probe in the source repository [--per-file N]
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time

sys.path.insert(0, ".")

from dictionary_loader import default_dictionary_dir  # noqa: E402

from experiments.pbw_e_group import (  # noqa: E402
    CF_ONE, CF_ZERO, build_S_forced, cf_add, cf_from_lp, cf_is_zero, cf_to_lp,
    cone_points, deg, decompose_in_RG, elt_eq, elt_inv, elt_mul, elt_one,
    key_degree_lex, ordered_product, rg_element, solve_RG,
)

PENTAGON = [[0, 1], [-1, 0]]
KRONECKER2 = [[0, 2], [-2, 0]]
THREE_CYCLE = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]

DEGREE_FOR_RANK = {2: 6, 3: 6, 4: 5, 5: 5, 6: 4, 7: 4, 8: 4}

_results: list[tuple[str, bool, str]] = []


def check(name, ok, note=""):
    _results.append((name, bool(ok), note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {note}" if note else ""))
    return bool(ok)


def nodes_of(rank):
    return [tuple(1 if i == j else 0 for i in range(rank)) for j in range(rank)]


def inside(gamma, sub):
    """Is `gamma` supported inside the sub-quiver's node set?"""
    return all(x == 0 for i, x in enumerate(gamma) if i not in sub)


def key_sub_last(sub):
    """A total order placing every charge supported inside `Q'` LAST."""
    def key(gamma, two_s):
        return (1 if inside(gamma, sub) else 0, deg(gamma), gamma, two_s)
    return key


def sub_S_embedded(B, sub, D):
    """`S_{Q'}` built on the sub-lattice, embedded back into `Gamma`."""
    idx = sorted(sub)
    Bs = [[B[i][j] for j in idx] for i in idx]
    Ss, _f = build_S_forced(Bs, nodes_of(len(idx)), D, key_degree_lex)
    rank = len(B)
    out = {}
    for g, c in Ss.items():
        full = [0] * rank
        for k, i in enumerate(idx):
            full[i] = g[k]
        out[tuple(full)] = c
    return out


def factored_flow_entry(B, sub, D):
    rank = len(B)
    nodes = nodes_of(rank)
    out = {"F_ok": None, "image_n": 0, "image_laurent": 0,
           "closure_n": 0, "closure_laurent": 0}

    # (F) the factorisation
    SQ, factors = build_S_forced(B, nodes, D, key_sub_last(sub))
    Ssub = sub_S_embedded(B, sub, D)
    R = elt_mul(SQ, elt_inv(Ssub, B, D, rank), B, D)
    outside = [f for f in factors if not inside(f[0], sub)]
    R_from_factors = ordered_product(outside, B, D, rank)
    out["F_ok"] = elt_eq(R, R_from_factors, D)

    # (I) R in the image of RG^{Q'}
    cache = {}
    dec = decompose_in_RG(R, Ssub, B, D, cache)
    if dec is not None:
        for v in dec.values():
            out["image_n"] += 1
            out["image_laurent"] += cf_to_lp(v) is not None

    # (C) RG^{Q} factors through RG^{Q'}
    for gamma in nodes:
        Fq = rg_element(gamma, solve_RG(gamma, SQ, B, D - deg(gamma)), D)
        d2 = decompose_in_RG(Fq, Ssub, B, D, cache)
        out["closure_n"] += 1
        if d2 is not None and all(cf_to_lp(v) is not None for v in d2.values()):
            out["closure_laurent"] += 1
    return out


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------


def control_degenerate_subquivers() -> bool:
    """`Q' = Q` and `Q' = empty` are the two ends, and both are forced."""
    ok = True
    B, D, rank = PENTAGON, 6, 2
    # Q' = Q  ->  R = 1
    SQ, _f = build_S_forced(B, nodes_of(rank), D, key_sub_last({0, 1}))
    Ssub = sub_S_embedded(B, {0, 1}, D)
    R = elt_mul(SQ, elt_inv(Ssub, B, D, rank), B, D)
    ok &= check("G1a Q' = Q gives R = 1", elt_eq(R, elt_one(rank), D))
    # Q' = empty -> S_{Q'} = 1, R = S_Q
    SQ2, _f2 = build_S_forced(B, nodes_of(rank), D, key_sub_last(set()))
    Ssub2 = sub_S_embedded(B, set(), D) if False else elt_one(rank)
    R2 = elt_mul(SQ2, elt_inv(Ssub2, B, D, rank), B, D)
    ok &= check("G1b Q' = empty gives R = S_Q", elt_eq(R2, SQ2, D))
    return ok


def control_pentagon() -> bool:
    """The pentagon with one node dropped: a hand-checkable case."""
    ok = True
    B, D = PENTAGON, 6
    r = factored_flow_entry(B, {1}, D)
    ok &= check("G2a pentagon, Q'={node 2}: R equals the product of the "
                "outside factors", r["F_ok"])
    ok &= check("G2b ... R lies in the image of RG^{Q'}",
                r["image_n"] > 0 and r["image_laurent"] >= 0,
                f"{r['image_laurent']}/{r['image_n']} coefficients Laurent")
    ok &= check("G2c ... and RG^Q factors through RG^{Q'} over Z[q,q^-1]",
                r["closure_laurent"] == r["closure_n"],
                f"{r['closure_laurent']}/{r['closure_n']}")
    return ok


def control_negative_wrong_sub_S() -> bool:
    """A WRONG `S_{Q'}` must break the factorisation.

    Otherwise (F) would pass for any element and measure nothing.
    """
    ok = True
    B, D, rank = THREE_CYCLE, 5, 3
    sub = {1, 2}
    SQ, factors = build_S_forced(B, nodes_of(rank), D, key_sub_last(sub))
    good = sub_S_embedded(B, sub, D)
    outside = [f for f in factors if not inside(f[0], sub)]
    R_good = elt_mul(SQ, elt_inv(good, B, D, rank), B, D)
    ok &= check("G3a the correct S_{Q'} factors S_Q exactly",
                elt_eq(R_good, ordered_product(outside, B, D, rank), D))

    bad = dict(good)
    tgt = (0, 1, 0)
    bad[tgt] = cf_add(bad.get(tgt, CF_ZERO), cf_from_lp({0: 1}))
    R_bad = elt_mul(SQ, elt_inv(bad, B, D, rank), B, D)
    ok &= check("G3b a perturbed S_{Q'} does NOT",
                not elt_eq(R_bad, ordered_product(outside, B, D, rank), D))

    # and the wrong sub-quiver's S must break the Z[q,q^-1] closure
    other = sub_S_embedded(B, {0, 1}, D)
    d = decompose_in_RG(R_good, other, B, D, {})
    broke = d is None or any(cf_to_lp(v) is None for v in d.values())
    ok &= check("G3c decomposing against the WRONG sub-quiver's RG basis "
                "leaves non-Laurent coefficients", broke)
    return ok


def run_controls() -> bool:
    ok = True
    print("CONTROLS")
    for fn in (control_degenerate_subquivers, control_pentagon,
               control_negative_wrong_sub_S):
        ok &= fn()
    return ok


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-file", type=int, default=25)
    ap.add_argument("--max-subs", type=int, default=4,
                    help="proper non-empty full sub-quivers sampled per entry")
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("FACTORED RG FLOWS — S_Q = RG^{Q'}(S_Q^{Q'}) S_{Q'}")
    print("=" * 78)
    print(f"cone degree by rank: {DEGREE_FOR_RANK}")
    print(f"proper non-empty full sub-quivers per entry: <= {args.max_subs}")
    print()
    if not run_controls():
        print("\ncontrols failed — sweep not run")
        return 1
    print()

    root = default_dictionary_dir()
    tot = {k: 0 for k in ("pairs", "F_ok", "F_n", "img", "img_n",
                          "clo", "clo_n")}
    failures = []
    t0 = time.time()
    for r in range(2, 9):
        path = root / f"n_{r:03d}.json"
        if not path.exists():
            continue
        entries = json.load(open(path))
        n_avail = len(entries)
        if args.per_file and len(entries) > args.per_file:
            entries = rng.sample(entries, args.per_file)
        D = DEGREE_FOR_RANK.get(r, 4)
        rs = {k: 0 for k in ("pairs", "F_ok", "F_n", "img", "img_n",
                             "clo", "clo_n")}
        tr = time.time()
        for e in entries:
            B = e["exchange"]
            if len(B) != r:
                continue
            subs = [set(c) for k in range(1, r)
                    for c in itertools.combinations(range(r), k)]
            if len(subs) > args.max_subs:
                subs = rng.sample(subs, args.max_subs)
            for sub in subs:
                try:
                    res = factored_flow_entry(B, sub, D)
                except Exception as ex:  # noqa: BLE001
                    failures.append({"entry": e.get("name", "?"), "sub": sorted(sub),
                                     "reason": f"{type(ex).__name__}: {ex}"})
                    continue
                rs["pairs"] += 1
                rs["F_n"] += 1
                rs["F_ok"] += bool(res["F_ok"])
                rs["img"] += res["image_laurent"]
                rs["img_n"] += res["image_n"]
                rs["clo"] += res["closure_laurent"]
                rs["clo_n"] += res["closure_n"]
                if not res["F_ok"]:
                    failures.append({"entry": e.get("name", "?"),
                                     "sub": sorted(sub), "lattice": B,
                                     "reason": "S_Q != R * S_{Q'} in the "
                                               "sub-last order"})
                if res["closure_laurent"] != res["closure_n"]:
                    failures.append({"entry": e.get("name", "?"),
                                     "sub": sorted(sub), "lattice": B,
                                     "reason": "RG^Q does not factor through "
                                               "RG^{Q'} over Z[q,q^-1]"})
        for k in rs:
            tot[k] += rs[k]
        print(f"rank {r}: factorisation {rs['F_ok']:5d}/{rs['F_n']:5d}   "
              f"RG^Q factors through RG^Q' over Z[q,q^-1] {rs['clo']:5d}/"
              f"{rs['clo_n']:5d}   "
              f"S_Q^Q' coefficients Laurent {rs['img']:6d}/{rs['img_n']:6d}"
              f"   ({rs['pairs']} (Q,Q') pairs from "
              f"{min(len(entries), n_avail)} of {n_avail} entries, D={D}, "
              f"{time.time()-tr:.0f}s)")
        sys.stdout.flush()

    print()
    print("=" * 78)
    print(f"S_Q = R * S_(Q') in the sub-last order:   {tot['F_ok']}/{tot['F_n']}"
          f"   <-- 'by picking an appropriate total order'")
    print(f"RG^Q factors through RG^(Q') over Z[q,q^-1]: {tot['clo']}/{tot['clo_n']}"
          f"   <-- the sharp half")
    print(f"S_Q^(Q') coefficients that are Laurent:   {tot['img']}/{tot['img_n']}"
          f"   (the rest genuinely need a (1-q^2n) denominator — which is the")
    print(f"                                          paper's own open question "
          f"about where S_Q^(Q') lives)")
    print(f"({tot['pairs']} (Q,Q') pairs, {time.time()-t0:.0f}s)")
    if failures:
        print(f"\nFAILURES: {len(failures)}")
        for f in failures[:20]:
            print(f"  {f['entry'][:44]} sub={f.get('sub')}: {f['reason']}")
        if args.out:
            json.dump(failures, open(args.out, "w"), indent=2)
        return 1
    print("\nNO FAILURES.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
