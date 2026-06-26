"""Systematic trace-uniqueness testing on finite-type KAlgebras.

The `K_𝖖`-contract treats the ρ²-twisted trace as *data*: a linear
functional with `Tr(ab) = Tr(ρ²(b)a)` and `Tr(ρ(a)b)[q⁰] = δ_{ab}`,
"defined up to an overall rescaling by `1 + O(𝖖)`" (README).  Finite-type
KAlgebras are the corner of the catalogue where that uniqueness claim
is *exhaustively testable*: multiplication is table-driven and exact,
so on any finite label window the space of admissible traces is the
solution set of an explicit linear system over Q, order by order in q.

Setup.  Unknowns `t_c[k]` (`c` a canonical label in the window closure,
`0 ≤ k ≤ K`; traces of every known finite-type algebra live in
`Z[[q]]`, so non-negative orders only — assumption surfaced here).
Constraints:

* **cyclicity** (homogeneous): for window pairs `(a, b)`,
  `Σ_c C^c_{ab}·t_c − Σ_c C^c_{ρ²(b),a}·t_c = 0`, expanded per q-order
  (an equation at order `k` is imposed only when every contributing
  `t_c[k−j]` lies inside the truncation — no silently-dropped terms);
* **q⁰-orthonormality** (inhomogeneous, optional): for window pairs,
  `[Σ_c C^c_{ρ(a),b}·t_c]_{q⁰} = δ_{ab}`;
* **unit normalization** (optional): `t_identity[0] = 1`.

Outputs (`trace_solution_space`): the affine solution dimension, a
particular solution, and a kernel basis — all exact (Fraction Gaussian
elimination).  `verify_unique_up_to_rescale` then tests the sharp form
of the contract's claim on an algebra whose canonical trace is known:

    the admissible traces are EXACTLY  { f · Tr_canonical :
                                          f ∈ 1 + q·Q[[q]] }

i.e. (i) the known trace solves the system, (ii) the kernel has
dimension exactly `K` (the free coefficients `f[1..K]`), and (iii) the
kernel coincides with `span{ q^m · Tr_canonical : 1 ≤ m ≤ K }`.

Caveats (windowed truth): a finite window only *under*-constrains —
missing pairs mean missing equations — so a PASS at growing windows is
accumulating evidence, not a proof; a kernel SMALLER than the rescale
line, or the known trace failing the system, would be a genuine
contradiction.  Dimensions are reported per window so growth can be
watched.  First systematic results live in
`tests/test_trace_uniqueness.py` (pentagon, heptagon) and are discussed
in `finite_type_kalgebras.md`.
"""
from __future__ import annotations

from fractions import Fraction

from laurent_poly import LaurentPoly


__all__ = [
    "window_closure",
    "trace_solution_space",
    "verify_unique_up_to_rescale",
    "known_trace_vector",
    "quartic_bootstrap_rows",
]


def _coeffs(c) -> dict[int, int]:
    """Exponent dict of a LaurentPoly coefficient (trivial-R only)."""
    if isinstance(c, LaurentPoly):
        return dict(c._coeffs)
    raise TypeError(
        f"trace_uniqueness: trivial-R (LaurentPoly) coefficients only "
        f"for now; got {type(c).__name__} — flavoured entries follow "
        f"the Plan-18 Z-form regeneration")


def window_closure(A, labels):
    """The window plus every label appearing in the products the
    cyclicity/orthonormality equations touch."""
    labels = list(dict.fromkeys(labels))
    closure = dict.fromkeys(labels)
    for a in labels:
        for b in labels:
            for x, y in ((a, b), (A.rho(A.rho(b)), a),
                         (A.rho(a), b)):
                for c in A.multiply(x, y).terms:
                    closure.setdefault(c)
    return list(closure)


def _mul_elements(A, x, y):
    """Bilinear extension of `A.multiply` (trivial-R Elements)."""
    out: dict = {}
    for la, ca in x.terms.items():
        for lb, cb in y.terms.items():
            for lc, cc in A.multiply(la, lb).terms.items():
                v = ca * cb * cc
                out[lc] = out[lc] + v if lc in out else v
    from kalgebra import Element
    return Element({l: c for l, c in out.items() if not c.is_zero()})


def _element_diff_terms(e1, e2):
    """`e1 − e2` as `{label: {q_exp: int}}`."""
    diff: dict[tuple, dict[int, int]] = {}
    for sign, e in ((1, e1), (-1, e2)):
        for c, co in e.terms.items():
            for j, v in _coeffs(co).items():
                diff.setdefault(c, {})[j] = (
                    diff.get(c, {}).get(j, 0) + sign * v)
    return {c: {j: v for j, v in js.items() if v}
            for c, js in diff.items()
            if any(js.values())}


def quartic_bootstrap_rows(A, quadruples, col, K):
    """The trace-bootstrap channel equations (user, 2026-06-10): for
    each quadruple `(a, b, c, d)`,

        Tr( (L_a·L_b) · (L_c·L_d) )  =  Tr( (ρ²(L_d)·L_a) · (L_b·L_c) )

    — 4-point ρ²-twisted cyclicity, factorized in the (ab|cd) vs
    (ρ²(d)a|bc) channels.  Expanded onto canonical labels these are
    linear constraints on the 1-point unknowns `t_c[k]` that reach
    relations a pair window cannot see (the pair system only imposes
    cyclicity for window pairs; the 4-point crossing imports the
    deeper labels' relations while keeping every *external* leg in
    the window).  Unknown labels beyond `col` are added by the
    caller via `window_closure`-style extension — here rows touching
    unknowns outside `col` or beyond order `K` are dropped whole (no
    silent truncation)."""
    from kalgebra import Element
    one = LaurentPoly.one()
    rows: list[tuple[dict, Fraction]] = []
    for (a, b, c, d) in quadruples:
        ab = A.multiply(a, b)
        cd = A.multiply(c, d)
        lhs = _mul_elements(A, ab, cd)
        da = A.multiply(A.rho(A.rho(d)), a)
        bc = A.multiply(b, c)
        rhs = _mul_elements(A, da, bc)
        terms = [(lab, j, v)
                 for lab, js in _element_diff_terms(lhs, rhs).items()
                 for j, v in js.items()]
        if not terms:
            continue
        if any((lab, 0) not in col for lab, _j, _v in terms):
            continue                  # outside the unknown closure
        jmax = max(j for _l, j, _v in terms)
        jmin = min(j for _l, j, _v in terms)
        for k in range(max(0, jmax), min(K, K + jmin) + 1):
            row: dict[int, Fraction] = {}
            for lab, j, v in terms:
                i = col[(lab, k - j)]
                row[i] = row.get(i, Fraction(0)) + v
            row = {i: v for i, v in row.items() if v}
            if row:
                rows.append((row, Fraction(0)))
    return rows


def _quadruple_closure(A, quadruples):
    """Labels appearing in the bootstrap products."""
    out: dict = {}
    for (a, b, c, d) in quadruples:
        ab = A.multiply(a, b)
        cd = A.multiply(c, d)
        for lab in _mul_elements(A, ab, cd).terms:
            out.setdefault(lab)
        da = A.multiply(A.rho(A.rho(d)), a)
        bc = A.multiply(b, c)
        for lab in _mul_elements(A, da, bc).terms:
            out.setdefault(lab)
    return list(out)


def _equations(A, labels, K, *, orthonormal_q0, unit_norm,
               quadruples=None):
    """Build the sparse linear system.  Returns (unknown_index, rows)
    with rows as (dict[col, Fraction], rhs)."""
    closure = window_closure(A, labels)
    if quadruples:
        for lab in _quadruple_closure(A, quadruples):
            if lab not in closure:
                closure.append(lab)
    col = {(c, k): i for i, (c, k) in enumerate(
        ((c, k) for c in closure for k in range(K + 1)))}
    rows: list[tuple[dict, Fraction]] = []

    def _add_cyclicity(a, b):
        diff: dict[tuple, dict[int, int]] = {}
        for c, co in A.multiply(a, b).terms.items():
            for j, v in _coeffs(co).items():
                diff.setdefault(c, {})[j] = diff.get(c, {}).get(j, 0) + v
        for c, co in A.multiply(A.rho(A.rho(b)), a).terms.items():
            for j, v in _coeffs(co).items():
                diff.setdefault(c, {})[j] = diff.get(c, {}).get(j, 0) - v
        terms = [(c, j, v) for c, js in diff.items()
                 for j, v in js.items() if v]
        if not terms:
            return
        jmax = max(j for _c, j, _v in terms)
        jmin = min(j for _c, j, _v in terms)
        # equation at order k uses t_c[k-j]: valid iff 0 <= k-j <= K
        # for every term  =>  jmax <= k <= K + jmin  (so no term is
        # silently truncated away)
        for k in range(max(0, jmax), min(K, K + jmin) + 1):
            row: dict[int, Fraction] = {}
            for c, j, v in terms:
                i = col[(c, k - j)]
                row[i] = row.get(i, Fraction(0)) + v
            row = {i: v for i, v in row.items() if v}
            if row:
                rows.append((row, Fraction(0)))

    def _add_orthonormal(a, b):
        row: dict[int, Fraction] = {}
        for c, co in A.multiply(A.rho(a), b).terms.items():
            for j, v in _coeffs(co).items():
                if -j < 0:
                    continue          # t_c is supported in q >= 0
                if -j > K:
                    return            # q^0 needs t-orders beyond K: skip
                i = col[(c, -j)]
                row[i] = row.get(i, Fraction(0)) + v
        rows.append(({i: v for i, v in row.items() if v},
                     Fraction(1 if a == b else 0)))

    for a in labels:
        for b in labels:
            _add_cyclicity(a, b)
    if orthonormal_q0:
        for a in labels:
            for b in labels:
                _add_orthonormal(a, b)
    if unit_norm:
        rows.append(({col[(A.identity(), 0)]: Fraction(1)}, Fraction(1)))
    if quadruples:
        rows.extend(quartic_bootstrap_rows(A, quadruples, col, K))
    return col, rows


def _solve(col, rows):
    """Exact SPARSE Gaussian elimination (dict rows + live column
    index).  Returns (particular | None, kernel_basis) as dense
    Fraction vectors over the columns — kernels are assembled densely
    only at the end (each kernel vector's support is one free column
    plus the sparse pivot couplings)."""
    n = len(col)
    RHS = n                            # sentinel key for the rhs
    work: list[dict] = []
    for row, rhs in rows:
        r = {i: v for i, v in row.items() if v}
        if rhs:
            r[RHS] = rhs
        if r:
            work.append(r)
    # column -> set of live row ids containing it
    where: dict[int, set] = {}
    for ri, r in enumerate(work):
        for c in r:
            if c != RHS:
                where.setdefault(c, set()).add(ri)
    pivot_of_col: dict[int, int] = {}
    alive = [True] * len(work)
    for c in range(n):
        # candidates: live rows containing c that are not already
        # pivot rows; prefer the sparsest
        cand = [ri for ri in where.get(c, ()) if alive[ri]
                and "_piv" not in work[ri]]
        if not cand:
            continue
        ri = min(cand, key=lambda j: len(work[j]))
        row = work[ri]
        inv = 1 / row[c]
        for k in list(row):
            if k != "_piv":
                row[k] = row[k] * inv
        row["_piv"] = c
        pivot_of_col[c] = ri
        for rj in list(where.get(c, ())):
            if rj == ri or not alive[rj]:
                continue
            other = work[rj]
            f = other.get(c)
            if not f:
                continue
            for k, v in row.items():
                if k == "_piv":
                    continue
                nv = other.get(k, Fraction(0)) - f * v
                if nv:
                    other[k] = nv
                    if k != RHS:
                        where.setdefault(k, set()).add(rj)
                else:
                    other.pop(k, None)
                    if k != RHS and k in where:
                        where[k].discard(rj)
            if not any(k != "_piv" for k in other):
                alive[rj] = False
    # consistency: any live row with ONLY a rhs entry
    for ri, r in enumerate(work):
        if not alive[ri]:
            continue
        keys = [k for k in r if k not in ("_piv", RHS)]
        if not keys and r.get(RHS):
            return None, []
    particular = [Fraction(0)] * n
    for c, ri in pivot_of_col.items():
        particular[c] = work[ri].get(RHS, Fraction(0))
    free = [c for c in range(n) if c not in pivot_of_col]
    kernel = []
    for fc in free:
        v = [Fraction(0)] * n
        v[fc] = Fraction(1)
        for ri in where.get(fc, ()):
            if not alive[ri]:
                continue
            row = work[ri]
            pc = row.get("_piv")
            if pc is not None and fc in row:
                v[pc] = -row[fc]
        kernel.append(v)
    return particular, kernel


def trace_solution_space(A, labels, K, *, orthonormal_q0=True,
                         unit_norm=False, quadruples=None):
    """Solve the admissible-trace system on the window.  Returns a dict
    with `columns` (the (label, order) index), `particular` (None if
    inconsistent), `kernel` (basis vectors), and `dim`."""
    col, rows = _equations(A, labels, K,
                           orthonormal_q0=orthonormal_q0,
                           unit_norm=unit_norm,
                           quadruples=quadruples)
    particular, kernel = _solve(col, rows)
    return {
        "columns": col,
        "particular": particular,
        "kernel": kernel,
        "dim": None if particular is None else len(kernel),
        "n_unknowns": len(col),
        "n_equations": len(rows),
    }


def known_trace_vector(A, columns, K):
    """The algebra's own exact trace as a dense vector over `columns`
    (computed through `A.trace`, i.e. the frozen elementary tables for
    the zoo)."""
    vals: dict[tuple, Fraction] = {}
    by_label: dict = {}
    for (c, k) in columns:
        by_label.setdefault(c, []).append(k)
    for c in by_label:
        tr = A.trace(c, K=K)
        for k in by_label[c]:
            v = tr.coeffs.get(k, 0)
            if hasattr(v, "terms"):
                terms = {key: x for key, x in v.terms.items() if x}
                bad = [key for key in terms if key != ()]
                if bad:
                    raise ValueError(
                        f"known_trace_vector: non-trivial flavour "
                        f"content {terms} in Tr({c}) — trivial-R only")
                v = terms.get((), 0)
            vals[(c, k)] = Fraction(int(v))
    vec = [Fraction(0)] * len(columns)
    for key, i in columns.items():
        vec[i] = vals[key]
    return vec


def _in_span(vec, basis):
    """Exact membership of `vec` in span(basis)."""
    if not basis:
        return all(x == 0 for x in vec)
    n = len(vec)
    cols = len(basis)
    aug = [[basis[j][i] for j in range(cols)] + [vec[i]]
           for i in range(n)]
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, n) if aug[i][c] != 0), None)
        if piv is None:
            continue
        aug[r], aug[piv] = aug[piv], aug[r]
        inv = 1 / aug[r][c]
        aug[r] = [x * inv for x in aug[r]]
        for i in range(n):
            if i != r and aug[i][c] != 0:
                f = aug[i][c]
                aug[i] = [x - f * y for x, y in zip(aug[i], aug[r])]
        r += 1
    return all(aug[i][cols] == 0 for i in range(r, n))


def _project(vectors, columns, label_set, K_eq):
    """Restrict dense vectors to the (window label, order ≤ K_eq)
    coordinates."""
    wcols = [i for (c, k), i in columns.items()
             if c in label_set and k <= K_eq]
    return [[v[i] for i in wcols] for v in vectors], wcols


def _rank(rows):
    rows = [r[:] for r in rows]
    if not rows:
        return 0
    n = len(rows[0])
    r = 0
    for c in range(n):
        piv = next((j for j in range(r, len(rows)) if rows[j][c] != 0),
                   None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        inv = 1 / rows[r][c]
        rows[r] = [x * inv for x in rows[r]]
        for j in range(len(rows)):
            if j != r and rows[j][c] != 0:
                f = rows[j][c]
                rows[j] = [x - f * y for x, y in zip(rows[j], rows[r])]
        r += 1
    return r


def verify_unique_up_to_rescale(A, labels, K_eq, *, K_unknown=None,
                                quadruples=None, verbose=False):
    """The sharp uniqueness statement, measured on the window's
    low-order coordinates: with cyclicity + q⁰-orthonormality (+ the
    optional 4-point bootstrap channels), the admissible traces —
    *projected to (window labels, orders ≤ K_eq)* — are exactly the
    `1 + q·Q[[q]]` rescales of the algebra's canonical trace.

    `K_unknown` (default `2·K_eq + 4`) carries the unknowns deep
    enough that no equation is dropped for truncation reasons near the
    measured orders.  The projection is the honest windowed statement:
    closure-only labels are nuisance unknowns whose own equations lie
    outside the window, so the full kernel dimension is
    window-inflated; what the window genuinely measures is the
    solution set's shadow on its own coordinates.

    Checks: (i) the canonical trace solves every equation;
    (ii) projected kernel dimension == K_eq; (iii) projected kernel ==
    projected span{q^m·Tr : 1 ≤ m ≤ K_eq}."""
    if K_unknown is None:
        K_unknown = 2 * K_eq + 4
    col, rows = _equations(A, labels, K_unknown, orthonormal_q0=True,
                           unit_norm=False, quadruples=quadruples)
    particular, kernel = _solve(col, rows)
    known = known_trace_vector(A, col, K_unknown)
    ok_known = particular is not None and all(
        sum(v * known[i] for i, v in row.items()) == rhs
        for row, rhs in rows
    )

    shifts = []
    for m in range(1, K_eq + 1):
        v = [Fraction(0)] * len(col)
        for (c, k), i in col.items():
            if k - m >= 0:
                v[i] = known[col[(c, k - m)]]
        shifts.append(v)

    label_set = set(labels)
    pk, _w = _project(kernel, col, label_set, K_eq)
    ps, _w = _project(shifts, col, label_set, K_eq)
    dim_proj = _rank(pk)
    ok_dim = dim_proj == K_eq
    ok_span = (all(_in_span(s, pk) for s in ps)
               and _rank(ps) == dim_proj) if ok_dim else False
    out = {
        "known_trace_solves": ok_known,
        "projected_dim_equals_K": ok_dim,
        "projected_kernel_is_rescale_line": ok_span,
        "projected_dim": dim_proj,
        "full_dim": None if particular is None else len(kernel),
        "n_unknowns": len(col),
        "n_equations": len(rows),
    }
    if verbose:
        print(f"  [{type(A).__name__}] K_eq={K_eq} "
              f"K_unknown={K_unknown} window={len(labels)} "
              f"bootstrap={'on' if quadruples else 'off'}: {out}")
    return out


# ---------------------------------------------------------------------------
# v2 — the Gram (2-point) bootstrap
# ---------------------------------------------------------------------------
#
# The v1 system above keeps 1-POINT data `t_c = Tr(L_c)` primary — the
# recursion-style coordinates (Layer-1 cyclicity iterates traces at
# q^k to q^{k+1}).  Measured on the pentagon, 4-point rows bolted onto
# that system are weaker than window-widening: orthonormality enters
# only through deep C-mixing, and the channels' content dissipates
# into closure-label unknowns.
#
# The bootstrap framing (user, 2026-06-10: "a procedure based on
# 4pt-functions as in other bootstrap problems may win in generality")
# makes the 2-POINT data primary instead:
#
#     G_{ab}(q) = Tr(L_a · L_b)            (the un-normalized Gram),
#
# with axioms imposed DIRECTLY on G:
#
#   (2pt cyclicity)   G_{ab} = G_{ρ²(b),a}
#   (orthonormality)  G_{ρ(a),b}[q⁰] = δ_{ab}        — boundary data!
#   (4pt crossing)    Σ_{ef} C^e_{ab} C^f_{cd} G_{ef}
#                       = Σ_{ef} C^e_{ρ²(d)a} C^f_{bc} G_{ef}
#                     — channel (ab|cd) vs (ρ²(d)a|bc) of Tr(abcd).
#
# No Layer-1, no cone structure, no 1-point closure: the system needs
# only structure constants on a window — which is why it generalises
# beyond finite type (the same equations run on any KAlgebra whose
# multiply is computable on a window, e.g. pure U(N)).
#
# MEASURED (pentagon q³-mode; tests::test_gram_bootstrap_ledger):
# crossing alone never pins the trace at any tried leg degree; the OPE
# half (`ope_reduction=True`) is independent load-bearing data, and
# with it the Gram system reproduces the v1 verdicts.  The reach of
# the q⁰ boundary data is the resolution knob (deg-2 leaves the mode,
# deg-3 kills it).

def _gram_pairs(A, labels, quadruples):
    """The 2-point unknown index: window pairs, their ρ²-cyclic
    partners, identity pairs, and every channel pairing the quadruples
    touch."""
    pairs: dict = {}

    def add(e, f):
        pairs.setdefault((e, f))

    for a in labels:
        add(a, A.identity())
        for b in labels:
            add(a, b)
            add(A.rho(A.rho(b)), a)
            add(A.rho(a), b)
    for (a, b, c, d) in quadruples:
        ab = A.multiply(a, b)
        cd = A.multiply(c, d)
        for e in ab.terms:
            for f in cd.terms:
                add(e, f)
                add(A.rho(A.rho(f)), e)
        da = A.multiply(A.rho(A.rho(d)), a)
        bc = A.multiply(b, c)
        for e in da.terms:
            for f in bc.terms:
                add(e, f)
                add(A.rho(A.rho(f)), e)
    return list(pairs)


def gram_bootstrap_space(A, labels, K_eq, *, quadruples=(),
                         K_unknown=None, orthonormal_q0=True,
                         ope_reduction=False):
    """Solve the 2-point bootstrap system.  Returns the same shape as
    `trace_solution_space`, with columns indexed by ((e, f), k).

    `ope_reduction=True` adds the factorization-through-the-identity-
    channel axiom `G_{ef} = Σ_c C^c_{ef}·G_{c,()}` for every pair in
    the pair set (extending the unknowns by the identity-leg pairs of
    the product supports) — the OPE half of the bootstrap, which pure
    crossing does not imply."""
    if K_unknown is None:
        K_unknown = 2 * K_eq + 4
    K = K_unknown
    pair_list = _gram_pairs(A, labels, quadruples)
    if ope_reduction:
        extra: dict = {}
        for (e, f) in pair_list:
            for c in A.multiply(e, f).terms:
                if (c, A.identity()) not in extra:
                    extra[(c, A.identity())] = None
        for p in extra:
            if p not in pair_list:
                pair_list.append(p)
    pset = set(pair_list)
    col = {(p, k): i for i, (p, k) in enumerate(
        ((p, k) for p in pair_list for k in range(K + 1)))}
    rows: list[tuple[dict, Fraction]] = []

    # (2pt cyclicity)
    for (e, f) in pair_list:
        partner = (A.rho(A.rho(f)), e)
        if partner in pset and partner != (e, f):
            for k in range(K + 1):
                rows.append((
                    {col[((e, f), k)]: Fraction(1),
                     col[(partner, k)]: Fraction(-1)},
                    Fraction(0)))

    # (orthonormality at q⁰, directly on G)
    if orthonormal_q0:
        for a in labels:
            for b in labels:
                p = (A.rho(a), b)
                rows.append(({col[(p, 0)]: Fraction(1)},
                             Fraction(1 if a == b else 0)))

    # (OPE reduction: G_{ef} = Σ_c C^c_{ef} G_{c, id})
    if ope_reduction:
        ident = A.identity()
        for (e, f) in list(pset):
            terms: dict = {}
            for cc, co in A.multiply(e, f).terms.items():
                for j, v in _coeffs(co).items():
                    terms.setdefault((cc, ident), {})[j] = (
                        terms.get((cc, ident), {}).get(j, 0) + v)
            flat = [(p, j, v) for p, js in terms.items()
                    for j, v in js.items() if v]
            jmax = max((j for _p, j, _v in flat), default=0)
            jmin = min((j for _p, j, _v in flat), default=0)
            for k in range(max(0, jmax), min(K, K + jmin) + 1):
                row: dict[int, Fraction] = {
                    col[((e, f), k)]: Fraction(1)}
                for p, j, v in flat:
                    i = col[(p, k - j)]
                    row[i] = row.get(i, Fraction(0)) - v
                row = {i: v for i, v in row.items() if v}
                if row:
                    rows.append((row, Fraction(0)))

    # (4pt crossing)
    for (a, b, c, d) in quadruples:
        terms: dict = {}
        ab = A.multiply(a, b)
        cd = A.multiply(c, d)
        for e, ce in ab.terms.items():
            for f, cf in cd.terms.items():
                prod = ce * cf
                for j, v in _coeffs(prod).items():
                    key = (e, f)
                    terms.setdefault(key, {})[j] = (
                        terms.get(key, {}).get(j, 0) + v)
        da = A.multiply(A.rho(A.rho(d)), a)
        bc = A.multiply(b, c)
        for e, ce in da.terms.items():
            for f, cf in bc.terms.items():
                prod = ce * cf
                for j, v in _coeffs(prod).items():
                    key = (e, f)
                    terms.setdefault(key, {})[j] = (
                        terms.get(key, {}).get(j, 0) - v)
        flat = [(p, j, v) for p, js in terms.items()
                for j, v in js.items() if v]
        if not flat:
            continue
        jmax = max(j for _p, j, _v in flat)
        jmin = min(j for _p, j, _v in flat)
        for k in range(max(0, jmax), min(K, K + jmin) + 1):
            row: dict[int, Fraction] = {}
            for p, j, v in flat:
                i = col[(p, k - j)]
                row[i] = row.get(i, Fraction(0)) + v
            row = {i: v for i, v in row.items() if v}
            if row:
                rows.append((row, Fraction(0)))

    particular, kernel = _solve(col, rows)
    return {
        "columns": col,
        "particular": particular,
        "kernel": kernel,
        "dim": None if particular is None else len(kernel),
        "n_unknowns": len(col),
        "n_equations": len(rows),
    }


def known_gram_vector(A, columns, K):
    """The exact Gram data `G_{ef} = Tr(L_e·L_f)` over the columns."""
    by_pair: dict = {}
    for ((e, f), k) in columns:
        by_pair.setdefault((e, f), []).append(k)
    vec = [Fraction(0)] * len(columns)
    for (e, f), ks in by_pair.items():
        prod = A.multiply(e, f)
        need = max(ks)
        acc: dict[int, int] = {}
        for lab, co in prod.terms.items():
            cs = _coeffs(co)
            emin = min(cs)
            tr = A.trace(lab, K=need - min(emin, 0))
            for j, v in cs.items():
                for kk, tv in tr.coeffs.items():
                    if hasattr(tv, "terms"):
                        tv = tv.terms.get((), 0)
                    if tv and j + kk <= need:
                        acc[j + kk] = acc.get(j + kk, 0) + v * int(tv)
        for k in ks:
            vec[columns[((e, f), k)]] = Fraction(acc.get(k, 0))
    return vec


def verify_gram_unique_up_to_rescale(A, labels, K_eq, *,
                                     quadruples=(), K_unknown=None,
                                     verbose=False):
    """The v2 uniqueness statement: the admissible Gram data —
    projected to (window pairs, orders ≤ K_eq) — is exactly the
    `1 + q·Q[[q]]` rescale line of the canonical Gram."""
    if K_unknown is None:
        K_unknown = 2 * K_eq + 4
    sol = gram_bootstrap_space(A, labels, K_eq, quadruples=quadruples,
                               K_unknown=K_unknown)
    col = sol["columns"]
    known = known_gram_vector(A, col, K_unknown)
    # known solves: rebuild rows is expensive; check via residual on a
    # fresh system build
    sol2 = gram_bootstrap_space(A, labels, K_eq, quadruples=quadruples,
                                K_unknown=K_unknown)
    ok_known = True
    # reconstruct rows once more for the residual check
    # (gram_bootstrap_space does not return rows; recompute minimal)
    # -- use the solution test instead: known must equal
    #    particular + kernel combination; equivalently known - particular
    #    must lie in span(kernel).
    if sol["particular"] is None:
        ok_known = False
    else:
        diff = [a - b for a, b in zip(known, sol["particular"])]
        ok_known = _in_span(diff, sol["kernel"])

    window_pairs = {(a, b) for a in labels for b in labels}
    shifts = []
    for m in range(1, K_eq + 1):
        v = [Fraction(0)] * len(col)
        for (p, k), i in col.items():
            if k - m >= 0:
                v[i] = known[col[(p, k - m)]]
        shifts.append(v)
    wcols = [i for (p, k), i in col.items()
             if p in window_pairs and k <= K_eq]
    pk = [[v[i] for i in wcols] for v in sol["kernel"]]
    ps = [[v[i] for i in wcols] for v in shifts]
    dim_proj = _rank(pk)
    ok_dim = dim_proj == K_eq
    ok_span = (all(_in_span(s, pk) for s in ps)
               and _rank(ps) == dim_proj) if ok_dim else False
    out = {
        "known_gram_in_solution_set": ok_known,
        "projected_dim_equals_K": ok_dim,
        "projected_kernel_is_rescale_line": ok_span,
        "projected_dim": dim_proj,
        "full_dim": sol["dim"],
        "n_unknowns": sol["n_unknowns"],
        "n_equations": sol["n_equations"],
    }
    if verbose:
        print(f"  [GRAM {type(A).__name__}] K_eq={K_eq} "
              f"K_unknown={K_unknown} window={len(labels)} "
              f"quads={len(list(quadruples))}: {out}")
    return out
