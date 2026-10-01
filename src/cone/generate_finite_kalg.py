"""Generic generator: build a cluster cone graph for an arbitrary
finite-type BPS quiver and emit a self-contained ``ConeKAlgebra``
subclass with all data (mult_gens, cones, cocycle, cross_product,
ρ-permutation) inlined as Python literals.

Usage as a library::

    from generate_finite_kalg import generate

    # E_7 (rank 7, 1d ker B = U(1) flavor)
    generate(
        out_path="finite_e7_kalg.py",
        class_name="FiniteE7KAlgebra",
        prefix="E7",
        pairing=B_e7,           # 7×7 antisymmetric integer matrix
        node_charges=N_e7,      # list of 7 length-7 integer tuples
        flavor="u1",            # "trivial" | "u1" | "su2" | "su2u1"
        max_charts=20000,
    )

For SU(2) and SU(2)×U(1) flavor, the BPS quiver must be provided in
the w-diagonal frame (the σ-anti-fixed direction = ker B aligned with
a single basis vector).

The generator writes a single .py file with:
  - <PREFIX>_BPS_PAIRING, <PREFIX>_BPS_NODE_CHARGES literals
  - <PREFIX>_MULT_GENS_LATTICE, <PREFIX>_CONES, <PREFIX>_COCYCLE_TABLE
  - <PREFIX>_RHO_PERM
  - <PREFIX>_CROSS_TABLE (full or ρ-orbit-reduced)
  - <ClassName>(ConeKAlgebra) and _Exported<ClassName>ConeData
"""
from __future__ import annotations
import sys
sys.path.insert(0, ".")


def build_finite_kalg_data(
    pairing: list,
    node_charges: list,
    flavor: str = "auto",
    su2_axis: int = -1,
    max_charts: int = 20000,
    *,
    prefix: str = "FINITE",
    verbose: bool = False,
    w_pairs: "list | None" = None,
) -> dict:
    """Run the FiniteBPSKAlgebra build pipeline and return the cone-algebra
    data as a dict — without emitting a standalone file.

    Returns ``{flavor, mult_gens_lattice, cones, cocycle_table, rho_perm,
    cross_table, cone_canon, rank_R}``.

    This is the data side of :func:`generate`; the latter is now a thin
    wrapper that calls this and then writes the standalone file.
    """
    from cluster_cone_builder import (
        FiniteBPSKAlgebra,
        recognize_su2_flavor, recognize_su2_u1_flavor,
    )
    import time, os

    def _log(msg):
        if verbose:
            print(msg)

    ckpt = f"/tmp/{prefix.lower()}_cross_cache_ckpt.pkl"
    os.environ["CROSS_CACHE_CKPT"] = ckpt
    _log(f"  checkpoint: {ckpt}")
    t0 = time.time()
    A = FiniteBPSKAlgebra(
        pairing=pairing, node_charges=node_charges,
        verify="off", max_charts=max_charts, w_pairs=w_pairs,
    )
    _log(f"  cluster graph: {time.time()-t0:.1f}s "
         f"({len(A._fbk_clusters)} clusters, {len(A._fbk_edges)} edges)")
    _log(f"  L_basis: {A._fbk_L_basis}")
    _log(f"  w-pairs: {A._fbk_w_pairs}")

    t0 = time.time()
    X_u1 = A.to_cone_kalgebra()
    _log(f"  to_cone_kalgebra: {time.time()-t0:.1f}s "
         f"(mult-gens={len(X_u1._ray_idx_to_gamma)})")

    # Detect or apply flavor recognition.
    if flavor == "auto":
        rank_L = len(A._fbk_L_basis)
        if rank_L == 0:
            flavor = "trivial"
        elif rank_L == 1 and len(A._fbk_w_pairs) == 1:
            flavor = "su2"
        elif rank_L == 1:
            flavor = "u1"
        elif rank_L == 2 and len(A._fbk_w_pairs) >= 1:
            flavor = "su2u1"
        else:
            raise ValueError(
                f"Cannot auto-detect flavor: rank(L)={rank_L}, "
                f"w-pairs={A._fbk_w_pairs}"
            )
        _log(f"  detected flavor: {flavor}")

    if flavor == "trivial" or flavor == "u1":
        X = X_u1
    elif flavor == "su2":
        X = recognize_su2_flavor(X_u1)
    elif flavor == "su2u1":
        X = recognize_su2_u1_flavor(X_u1, su2_axis=su2_axis)
    else:
        raise ValueError(f"Unknown flavor: {flavor!r}")

    cd = X.cone_data()
    mult_gens_lattice = tuple(X._ray_idx_to_gamma)
    cones = tuple(tuple(sorted(c)) for c in cd.cones())

    # Cocycle table
    cocycle_table: dict = {}
    for cone in cd.cones():
        cone_ids = sorted(cone)
        for i in cone_ids:
            for j in cone_ids:
                if i != j:
                    cocycle_table[(i, j)] = cd.cocycle(i, j)

    # Cross-product table.  Materialize as
    #   (i, j) -> [(coeff_repr, word), ...]
    # where coeff_repr is a dict suitable for reconstructing the
    # appropriate poly type at load time.
    n = len(mult_gens_lattice)
    cross_table: dict = {}
    for i in range(n):
        for j in range(n):
            terms = cd.cross_product(i, j)
            entry = []
            for coeff, word in terms:
                if hasattr(coeff, "_coeffs"):    # LaurentPoly
                    entry.append(("LP", dict(coeff._coeffs), tuple(word)))
                elif hasattr(coeff, "coeffs"):   # RLaurent
                    rl: dict = {}
                    for q_exp, r_el in coeff.coeffs.items():
                        rl[q_exp] = dict(r_el.terms)
                    entry.append(("RL", rl, tuple(word)))
                else:
                    raise TypeError(
                        f"Unsupported coeff type {type(coeff).__name__} "
                        f"at ({i},{j})"
                    )
            cross_table[(i, j)] = entry

    rho_perm = dict(X._rho_perm)

    # Per-cone lattice canonicalization for non-simplicial cones.
    #
    # Tropical labels (lattice points) uniquely identify canonical
    # basis elements.  Two cone-monomial words (gens, powers) and
    # (gens', powers') with the same lattice sum γ represent the SAME
    # canonical basis element L_γ.  Identify them with the appropriate
    # q-phase:
    #
    #   coeff_on_L_native_new = coeff_on_L_native_old
    #     ⇔  canon_phase = phase_out_new - phase_out_old
    #
    # where phase_out_X = cone_label_phase(X) = -Σ cocycle(g_a, g_b) · pa · pb.
    from itertools import combinations

    def _cone_label_phase(face_ids, powers):
        """Same formula derived_multiply uses (cone_label_phase)."""
        phase = 0
        for a in range(len(face_ids)):
            for b in range(a + 1, len(face_ids)):
                pa, pb = powers[a], powers[b]
                if pa and pb:
                    phase -= cocycle_table.get(
                        (face_ids[a], face_ids[b]), 0,
                    ) * pa * pb
        return phase

    cone_canon: dict = {}  # word_non_canon -> (word_canon, canon_phase)
    for cone in cd.cones():
        cone_ids = sorted(cone)
        if len(cone_ids) < 2:
            continue
        sums_to_pairs: dict = {}
        for a, b in combinations(cone_ids, 2):
            s = tuple(mult_gens_lattice[a][k] + mult_gens_lattice[b][k]
                      for k in range(len(mult_gens_lattice[a])))
            sums_to_pairs.setdefault(s, []).append((a, b))
        for s, plist in sums_to_pairs.items():
            if len(plist) <= 1:
                continue
            plist.sort()
            canon_pair = plist[0]
            canon_word = ((canon_pair[0], 1), (canon_pair[1], 1))
            canon_phase_new = _cone_label_phase(
                [canon_pair[0], canon_pair[1]], [1, 1],
            )
            for other in plist[1:]:
                other_word = ((other[0], 1), (other[1], 1))
                other_phase = _cone_label_phase(
                    [other[0], other[1]], [1, 1],
                )
                # canon_phase = phase_out_new - phase_out_old
                q_phase = canon_phase_new - other_phase
                cone_canon[other_word] = (canon_word, q_phase)

        # Length-3 canonicalization too (for non-simplicial cones whose
        # pair-substitution system isn't confluent for triples).
        if len(cone_ids) >= 3:
            from itertools import combinations as _combos
            sums_to_triples: dict = {}
            for a, b, c in _combos(cone_ids, 3):
                s = tuple(
                    mult_gens_lattice[a][k] + mult_gens_lattice[b][k]
                    + mult_gens_lattice[c][k]
                    for k in range(len(mult_gens_lattice[a]))
                )
                sums_to_triples.setdefault(s, []).append((a, b, c))
            for s, tlist in sums_to_triples.items():
                if len(tlist) <= 1:
                    continue
                tlist.sort()
                canon_triple = tlist[0]
                canon_word3 = tuple((g, 1) for g in canon_triple)
                canon_phase3 = _cone_label_phase(
                    list(canon_triple), [1, 1, 1],
                )
                for other in tlist[1:]:
                    other_word3 = tuple((g, 1) for g in other)
                    other_phase3 = _cone_label_phase(list(other), [1, 1, 1])
                    q_phase = canon_phase3 - other_phase3
                    # Only add if not already covered by pair-subs reaching same form
                    cone_canon.setdefault(other_word3, (canon_word3, q_phase))

    # Apply canonicalization to cross_table words: replace word with
    # canon_word and multiply coefficient by q^{q_phase}.
    def _shift_q(kind, data, q_phase):
        if q_phase == 0:
            return data
        if kind == "LP":
            return {e + q_phase: c for e, c in data.items()}
        elif kind == "RL":
            return {q + q_phase: dict(r) for q, r in data.items()}
        raise TypeError(kind)

    if cone_canon:
        _log(f"  cone canonicalizations (length-2): {len(cone_canon)}")
        for (i, j), entry in list(cross_table.items()):
            by_word: dict = {}
            for kind, data, word in entry:
                tup = cone_canon.get(word)
                if tup is not None:
                    canon_word, q_phase = tup
                    data = _shift_q(kind, data, q_phase)
                else:
                    canon_word = word
                key = (kind, canon_word)
                if key in by_word:
                    if kind == "LP":
                        d = by_word[key]
                        for e, c in data.items():
                            d[e] = d.get(e, 0) + c
                    elif kind == "RL":
                        d = by_word[key]
                        for q_exp, r_terms in data.items():
                            if q_exp in d:
                                d2 = d[q_exp]
                                for char, c in r_terms.items():
                                    d2[char] = d2.get(char, 0) + c
                            else:
                                d[q_exp] = dict(r_terms)
                else:
                    by_word[key] = dict(data) if kind == "LP" else {
                        q: dict(r) for q, r in data.items()
                    }
            new_entry = []
            for (kind, canon_word), data in by_word.items():
                if kind == "LP":
                    data = {e: c for e, c in data.items() if c != 0}
                    if not data:
                        continue
                elif kind == "RL":
                    cleaned = {}
                    for q_exp, rt in data.items():
                        rt2 = {ch: c for ch, c in rt.items() if c != 0}
                        if rt2:
                            cleaned[q_exp] = rt2
                    data = cleaned
                    if not data:
                        continue
                new_entry.append((kind, data, canon_word))
            cross_table[(i, j)] = new_entry

    _log(f"  mult-gens: {n}")
    _log(f"  cones:     {len(cones)}")
    _log(f"  cocycles:  {len(cocycle_table)}")
    _log(f"  cross-table entries: {len(cross_table)}")

    rank_R = getattr(X.coefficient_ring(), "rank", None) if flavor == "u1" else None
    rho_delta = dict(getattr(X, "_rho_delta", None) or {})
    return {
        "flavor": flavor,
        "mult_gens_lattice": mult_gens_lattice,
        "cones": cones,
        "cocycle_table": cocycle_table,
        "rho_perm": rho_perm,
        "rho_delta": rho_delta,
        "cross_table": cross_table,
        "cone_canon": cone_canon,
        "rank_R": rank_R,
    }


def _emit_standalone(
    out_path: str,
    class_name: str,
    prefix: str,
    pairing: list,
    node_charges: list,
    data: dict,
    layer1_on_labels: bool = False,
) -> None:
    """Write the standalone .py file from a build_finite_kalg_data() result.
    `layer1_on_labels` sets the cone data's flag of that name (Layer 1 on
    canonical labels, `cone_data.ConeData.layer1_on_labels`)."""
    flavor = data["flavor"]
    mult_gens_lattice = data["mult_gens_lattice"]
    cones = data["cones"]
    cocycle_table = data["cocycle_table"]
    rho_perm = data["rho_perm"]
    rho_delta = data.get("rho_delta", {})
    cross_table = data["cross_table"]
    cone_canon = data["cone_canon"]
    rank_R = data["rank_R"]
    n = len(mult_gens_lattice)

    print(f"  writing {out_path} ...")
    pairing_lit = [list(row) for row in pairing]
    N_lit = [tuple(int(x) for x in g) for g in node_charges]

    with open(out_path, "w") as f:
        f.write(f'"""``{class_name}`` -- standalone {flavor}-flavored '
                f'ConeKAlgebra.\n\n')
        f.write(f"Generated by ``generate_finite_kalg.py`` from a finite\n")
        f.write(f"BPS quiver.  All data (mult_gens, cones, cocycle,\n")
        f.write(f"cross_product, ρ-permutation) inlined as Python literals.\n")
        f.write(f"No runtime dependency on ``FiniteBPSKAlgebra``.\n")
        f.write(f'"""\n')
        f.write("from __future__ import annotations\n\n")
        f.write("from cone_data import FiniteConeData\n")
        f.write("from cone_kalgebra import ConeKAlgebra\n")
        f.write("from laurent_poly import LaurentPoly\n")
        if flavor == "su2":
            f.write("from zplus_ring import SU2ZPlusRing, RElement, RLaurent\n")
            f.write("_R = SU2ZPlusRing()\n")
        elif flavor == "su2u1":
            f.write("from zplus_ring import SU2xU1ZPlusRing, RElement, RLaurent\n")
            f.write("_R = SU2xU1ZPlusRing()\n")
        elif flavor == "u1":
            f.write("from zplus_ring import AbelianZPlusRing, RElement, RLaurent\n")
            f.write(f"_R = AbelianZPlusRing(rank={rank_R})\n")
        else:
            # `RElement` / `RLaurent` are referenced by the emitted
            # `_make_coeff` helper unconditionally, so import them here too.
            # The R-branch is dead code for a genuinely unflavoured algebra
            # (no cross-table entry carries an R-kind), but it is NOT dead for
            # every build that asks for `flavor="trivial"` — a quiver whose
            # nodes do not span the ambient lattice produces R-kind entries
            # and used to emit a file that raised `NameError` on first use.
            f.write("from zplus_ring import TrivialZPlusRing, RElement, RLaurent\n")
            f.write("_R = TrivialZPlusRing()\n")
        f.write("\n")
        f.write(f"{prefix}_BPS_PAIRING = {pairing_lit!r}\n")
        f.write(f"{prefix}_BPS_NODE_CHARGES = {N_lit!r}\n\n")
        f.write(f"{prefix}_MULT_GENS_LATTICE = {mult_gens_lattice!r}\n\n")
        f.write(f"{prefix}_CONES = {cones!r}\n\n")
        f.write(f"{prefix}_COCYCLE_TABLE = {cocycle_table!r}\n\n")
        f.write(f"{prefix}_RHO_PERM = {rho_perm!r}\n\n")
        f.write(f"{prefix}_RHO_DELTA = {rho_delta!r}\n\n")
        f.write(f"{prefix}_CONE_CANON = {cone_canon!r}\n\n")
        f.write(f"{prefix}_CROSS_TABLE = {{\n")
        for (i, j), terms in cross_table.items():
            f.write(f"    ({i}, {j}): {terms!r},\n")
        f.write("}\n\n")

        # Coefficient-reconstruction helper.
        f.write("def _make_coeff(kind, data):\n")
        f.write("    if kind == 'LP':\n")
        f.write("        return LaurentPoly(data)\n")
        f.write("    if kind == 'RL':\n")
        f.write("        d = {q: RElement(_R, terms) for q, terms in data.items()}\n")
        f.write("        return RLaurent(_R, d)\n")
        f.write("    raise ValueError(f'Unknown coeff kind: {kind!r}')\n\n")

        # ConeData class.
        cd_name = f"_Exported{class_name}ConeData"
        f.write(f"class {cd_name}(FiniteConeData):\n")
        if layer1_on_labels:
            f.write("    # Layer 1 on canonical labels (`ConeData.layer1_on_labels`): the word\n")
            f.write("    # route's memo ran out of memory on deep labels.\n")
            f.write("    layer1_on_labels = True\n\n")
        f.write("    def coefficient_ring(self):\n        return _R\n\n")
        f.write(f"    def mult_gens(self):\n        return tuple(range(len({prefix}_MULT_GENS_LATTICE)))\n\n")
        f.write(f"    def cones(self):\n        return tuple(frozenset(c) for c in {prefix}_CONES)\n\n")
        f.write(f"    def q_commute(self, g, h):\n        return g == h or (g, h) in {prefix}_COCYCLE_TABLE\n\n")
        f.write(f"    def cocycle(self, g, h):\n        if g == h: return 0\n        return {prefix}_COCYCLE_TABLE[(g, h)]\n\n")
        f.write(f"    def cross_product(self, g, h):\n")
        f.write(f"        terms = {prefix}_CROSS_TABLE.get((g, h), [])\n")
        f.write(f"        return tuple((_make_coeff(kind, data), word) for kind, data, word in terms)\n\n")
        f.write("    def to_cone_label(self, native_label):\n")
        f.write("        gens = frozenset(i for i, p in native_label if p != 0)\n")
        f.write("        powers = {i: p for i, p in native_label if p != 0}\n")
        f.write("        return (gens, powers)\n\n")
        f.write("    def from_cone_label(self, gens, powers):\n")
        f.write("        return tuple(sorted((i, powers[i]) for i in gens))\n\n")
        f.write("    def canonicalize_cone_label(self, cone, gens, powers):\n")
        f.write(f"        # Apply subword substitutions; q-phase = full cone_label_phase\n")
        f.write(f"        # difference between before/after each step (principled — uses\n")
        f.write(f"        # the same formula as cone_label_phase, so signs are consistent).\n")
        f.write(f"        from itertools import combinations as _C\n")
        f.write(f"        def _full_phase(d):\n")
        f.write(f"            # cone_label_phase = -Σ_{{i<j in sorted flat word}} cocycle(i, j).\n")
        f.write(f"            ks = sorted(g for g, p in d.items() if p > 0)\n")
        f.write(f"            total = 0\n")
        f.write(f"            for ii in range(len(ks)):\n")
        f.write(f"                gi, pi = ks[ii], d[ks[ii]]\n")
        f.write(f"                # pairs (gi, gi) — same gen, count C(pi, 2) pairs\n")
        f.write(f"                # cocycle(gi, gi) is 0 so skip.\n")
        f.write(f"                for jj in range(ii + 1, len(ks)):\n")
        f.write(f"                    gj, pj = ks[jj], d[ks[jj]]\n")
        f.write(f"                    total += {prefix}_COCYCLE_TABLE.get((gi, gj), 0) * pi * pj\n")
        f.write(f"            return -total\n")
        f.write(f"        cur = {{g: p for g, p in powers.items() if p != 0}}\n")
        f.write(f"        accum_phase = 0\n")
        f.write(f"        changed = True\n")
        f.write(f"        while changed:\n")
        f.write(f"            changed = False\n")
        f.write(f"            in_cone = sorted(cur.keys())\n")
        f.write(f"            # Try pair subs, then triple subs.\n")
        f.write(f"            for size in (2, 3):\n")
        f.write(f"                if changed: break\n")
        f.write(f"                for combo in _C(in_cone, size):\n")
        f.write(f"                    if any(cur.get(g, 0) <= 0 for g in combo): continue\n")
        f.write(f"                    key = tuple((g, 1) for g in combo)\n")
        f.write(f"                    entry = {prefix}_CONE_CANON.get(key)\n")
        f.write(f"                    if entry is None: continue\n")
        f.write(f"                    canon, _stored_phase = entry  # recompute properly\n")
        f.write(f"                    new_cur = dict(cur)\n")
        f.write(f"                    for g in combo: new_cur[g] -= 1\n")
        f.write(f"                    for g, p in canon: new_cur[g] = new_cur.get(g, 0) + p\n")
        f.write(f"                    new_cur = {{g: p for g, p in new_cur.items() if p > 0}}\n")
        f.write(f"                    accum_phase += _full_phase(new_cur) - _full_phase(cur)\n")
        f.write(f"                    cur = new_cur\n")
        f.write(f"                    changed = True\n")
        f.write(f"                    break\n")
        f.write(f"        new_powers = dict(cur)\n")
        f.write(f"        new_gens = frozenset(new_powers)\n")
        f.write(f"        return new_gens, new_powers, accum_phase\n\n")
        f.write(f"    def cycle_period_bound(self):\n        return len({prefix}_MULT_GENS_LATTICE)\n\n")

        # ConeKAlgebra class.
        f.write(f"class {class_name}(ConeKAlgebra):\n")
        f.write(f'    """Self-contained {flavor}-flavored ConeKAlgebra.\n\n')
        f.write(f"    Generated by ``generate_finite_kalg.generate`` from\n")
        f.write(f"    a finite ADE BPS quiver.\n")
        f.write(f'    """\n\n')
        f.write(f"    _R = _R\n")
        f.write(f"    _rho_perm = {prefix}_RHO_PERM\n")
        f.write(f"    _rho_inv_perm = {{v: k for k, v in {prefix}_RHO_PERM.items()}}\n")
        f.write(f"    _ray_idx_to_gamma = {prefix}_MULT_GENS_LATTICE\n\n")
        emit_rho_R_su2 = flavor == "su2"
        # Flavour bookkeeping consumed by the contract-compliant Z-form
        # wrappers (finite_u1_zform / finite_su2u1_zform).  Emit the rho-delta
        # shift table AND the U(1)-axis rank for every u1/su2u1 entry, even
        # when the delta is currently all-zero: the wrapper requires both
        # attributes to exist, and a missing _L_basis_rank is exactly the
        # e7 / a1d8 defect.  (For SU(2)×U(1) the delta is projected to the
        # U(1) axis; an empty delta makes the wrapper honest-fail at
        # construction, which is correct — not a silent wrong rho.)
        if flavor == "u1":
            f.write(f"    _rho_delta = {prefix}_RHO_DELTA\n")
            f.write(f"    _L_basis_rank = {rank_R}\n\n")
        elif flavor == "su2u1":
            rho_delta_u1_only = {i: (d[0],) for i, d in rho_delta.items()}
            f.write(f"    _rho_delta = {rho_delta_u1_only!r}\n")
            f.write(f"    _L_basis_rank = 1\n\n")
        f.write(f"    def __init__(self):\n")
        f.write(f"        self._cone_data = {cd_name}()\n\n")
        f.write("    def coefficient_ring(self): return self._R\n")
        f.write("    def identity(self): return ()\n")
        f.write("    def cone_data(self): return self._cone_data\n\n")
        f.write("    def rho(self, label):\n")
        f.write("        return tuple(sorted(\n")
        f.write("            (self._rho_perm.get(i, i), p) for (i, p) in label\n")
        f.write("        ))\n\n")
        f.write("    def rho_inverse(self, label):\n")
        f.write("        return tuple(sorted(\n")
        f.write("            (self._rho_inv_perm.get(i, i), p) for (i, p) in label\n")
        f.write("        ))\n\n")
        if emit_rho_R_su2:
            # SU(2): χ-characters are Weyl-invariant (⋆ = id, δ = 0), so the
            # element-level ρ is just the label permutation carrying coeffs.
            f.write("    def rho_R_element(self, elem):\n")
            f.write("        return self.rho_element(elem)\n\n")
        # NOTE: u1 / su2u1 deliberately do NOT emit a bespoke rho_R_element.
        # The old codegen'd version twisted μ on the coefficients but the
        # contract surface (rho / verify_rho_is_automorphism) used the plain
        # label permutation, so ρ dropped the μ-twist and the universal
        # verifiers failed (and it crashed on plain LaurentPoly coeffs).  The
        # contract-compliant ρ now lives in the flavour-in-labels Z-form
        # subclass emitted at the end of this file (FiniteU1ZKAlgebra /
        # FiniteSU2U1ZKAlgebra), where ρ is an honest label permutation.
        # No `_label_section_decompose`: the finite zoo is gauge-only /
        # flavour-in-coefficients, so `ConeKAlgebra.r_label_decompose`'s
        # trivial-lift default `(label, R.one_basis())` covers it (the base
        # forward bridge reconstructs `(label, R.one())` for `to_R_form`).
        if flavor in ("u1", "su2u1"):
            # BOTH u1 and su2u1 use the base ConeKAlgebra.trace (no override):
            # Layer-1 reduces a composite to single-gen seeds via the δ-honest
            # tagged-cyclicity (`_rho2_twist_unit` supplying the μ^δ shift),
            # Layer-2 serves the seeds from the elementary-trace bootstrap — no
            # BPS per composite.  This includes e7 (90 mg): the per-seed BPS
            # bypass is *infeasible* for e7 composites, so the reducer is the
            # only viable path, not merely the preferred one.  The no-fold
            # `_canonical_rho2_orbit_rep` is REQUIRED: ρ²-orbits carry μ-shifts
            # (su2u1) or differ from the element-level ρ² by μ-shifts (u1), so
            # orbit members do NOT share a trace and must not be collapsed (the
            # collapse is a no-op for both u1 and su2u1).
            f.write("    def _canonical_rho2_orbit_rep(self, label):\n")
            f.write("        # NO rho^2-orbit folding: R has unit characters, so the\n")
            f.write("        # label-level rho^2 differs from the element-level rho^2 by\n")
            f.write("        # mu-shifts and orbit members do NOT share a trace (see\n")
            f.write("        # finite_kalgebras.elem_traces.fold_policy).  Seeds are\n")
            f.write("        # served per-mg by the elementary-trace table.\n")
            f.write("        return label\n\n")
        f.write("    def _trace_residual(self, seed_label, K):\n")
        f.write("        from finite_kalgebras.elem_traces import trace_residual\n")
        f.write(f"        return trace_residual({prefix.lower()!r}, self, seed_label, K)\n")
        # The A and D entries name their canonical basis by curves through the
        # certified map onto a family class (zoo_geometry,
        # the design record); the E entries have no geometric labelling.
        if prefix.lower() in ("pentagon", "heptagon", "a3", "a5", "a7", "a1d3",
                              "a1d4", "a1d5", "a1d6", "a1d7", "a1d8"):
            f.write("\n    def geometric_label(self, label):\n")
            f.write("        from finite_kalgebras.zoo_geometry import geometric_label\n")
            f.write(f"        return geometric_label({prefix.lower()!r}, label)\n")

    print(f"  done.  File size: {__import__('os').path.getsize(out_path)/1024:.1f} KB")


def generate(
    out_path: str,
    class_name: str,
    prefix: str,
    pairing: list,
    node_charges: list,
    flavor: str = "auto",
    su2_axis: int = -1,
    max_charts: int = 20000,
    rho_orbit_reduced: bool = False,
    layer1_on_labels: bool = False,
) -> None:
    """Build cluster cone graph + cross-product table, emit standalone.

    Thin wrapper: build the data with
    :func:`build_finite_kalg_data`, then write the standalone file with
    :func:`_emit_standalone`.
    """
    print(f"=== Generating {class_name} → {out_path} ===")
    data = build_finite_kalg_data(
        pairing=pairing, node_charges=node_charges, flavor=flavor,
        su2_axis=su2_axis, max_charts=max_charts, prefix=prefix,
        verbose=True,
    )
    _emit_standalone(
        out_path=out_path, class_name=class_name, prefix=prefix,
        pairing=pairing, node_charges=node_charges, data=data,
        layer1_on_labels=layer1_on_labels,
    )


# ---------------------------------------------------------------------------
# Sample driver -- regenerate finite_e7_kalg.py and small A_1D family
# ---------------------------------------------------------------------------

# The entries whose cone data runs Layer 1 on canonical labels
# (`cone_data.ConeData.layer1_on_labels`).
_LAYER1_ON_LABELS = {"a5", "a7", "a1d4", "a1d6", "a1d8", "e7"}


def _build_a1d_odd(n):
    """A_1D_n quiver (n odd), w-diagonal frame."""
    dim = n
    B = [[0]*dim for _ in range(dim)]
    chain_len = n - 2
    for i in range(chain_len - 1):
        B[i][i+1] = 1
        B[i+1][i] = -1
    last_chain = chain_len - 1
    sum_pos = chain_len
    B[last_chain][sum_pos] = 1
    B[sum_pos][last_chain] = -1
    N = [tuple(1 if p == k else 0 for p in range(dim)) for k in range(chain_len)]
    N.append(tuple(1 if p in (sum_pos, dim-1) else 0 for p in range(dim)))
    N.append(tuple((1 if p == sum_pos else (-1 if p == dim-1 else 0)) for p in range(dim)))
    return B, N


def _build_a1d_even(n):
    """A_1D_n quiver (n even), w-diagonal frame."""
    return _build_a1d_odd(n)  # same construction; flavor differs


def _build_e7():
    """E_7 BPS quiver (bipartite orientation), 1-d ker B → U(1) flavor."""
    B = [[0]*7 for _ in range(7)]
    def arr(a, b):
        B[a][b] = 1; B[b][a] = -1
    arr(0, 1); arr(2, 1); arr(2, 3); arr(4, 3); arr(4, 5); arr(6, 3)
    N = [tuple(1 if p == i else 0 for p in range(7)) for i in range(7)]
    return B, N


def _build_a_linear(n):
    """A_n linear quiver: 0→1→2→...→(n-1).  Trivial flavor for even n,
    U(1) for odd n."""
    B = [[0]*n for _ in range(n)]
    for i in range(n - 1):
        B[i][i+1] = 1
        B[i+1][i] = -1
    N = [tuple(1 if p == i else 0 for p in range(n)) for i in range(n)]
    return B, N


if __name__ == "__main__":
    import sys as _sys
    args = set(_sys.argv[1:])
    do_all = not args
    targets = {
        "pentagon": ("FinitePentagonKAlgebra", "PENTAGON", _build_a_linear, (2,), "trivial", 2000),
        "heptagon": ("FiniteHeptagonKAlgebra", "HEPTAGON", _build_a_linear, (4,), "trivial", 2000),
        "a3":       ("FiniteA3KAlgebra",       "A3",       _build_a_linear, (3,), "u1", 2000),
        "a5":       ("FiniteA5KAlgebra",       "A5",       _build_a_linear, (5,), "u1", 5000),
        "a7":       ("FiniteA7KAlgebra",       "A7",       _build_a_linear, (7,), "u1", 10000),
        "a1d3":     ("FiniteA1D3KAlgebra",     "A1D3",     _build_a1d_odd,  (3,), "su2", 2000),
        "a1d4":     ("FiniteA1D4KAlgebra",     "A1D4",     _build_a1d_even, (4,), "su2u1", 2000),
        "a1d5":     ("FiniteA1D5KAlgebra",     "A1D5",     _build_a1d_odd,  (5,), "su2", 2000),
        "a1d6":     ("FiniteA1D6KAlgebra",     "A1D6",     _build_a1d_even, (6,), "su2u1", 5000),
        "a1d7":     ("FiniteA1D7KAlgebra",     "A1D7",     _build_a1d_odd,  (7,), "su2", 10000),
        "a1d8":     ("FiniteA1D8KAlgebra",     "A1D8",     _build_a1d_even, (8,), "su2u1", 20000),
        "a1d9":     ("FiniteA1D9KAlgebra",     "A1D9",     _build_a1d_odd,  (9,), "su2", 50000),
        "e7":       ("FiniteE7KAlgebra",       "E7",       lambda *_: _build_e7(), (), "u1", 20000),
    }
    for name, (cls, prefix, builder, bargs, flavor, max_charts) in targets.items():
        if not do_all and name not in args:
            continue
        out_path = f"finite_{name}_kalg.py"
        B, N = builder(*bargs)
        try:
            generate(out_path=out_path, class_name=cls, prefix=prefix,
                     pairing=B, node_charges=N, flavor=flavor,
                     max_charts=max_charts,
                     layer1_on_labels=name in _LAYER1_ON_LABELS)
        except Exception as e:
            print(f"  FAILED ({name}): {type(e).__name__}: {e}")
