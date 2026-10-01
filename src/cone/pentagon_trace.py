"""
pentagon_trace.py — Trace reduction for the pentagon K-algebra.

Algebra (encoded in `pentagon_algebra.PentagonAlgebra`):
    L_{i+1} L_i     = q^2 L_i L_{i+1}
    L_{i+1} L_{i-1} = 1 + q   L_i
    L_{i-1} L_{i+1} = 1 + q^{-1} L_i                       (i in Z/5)
Canonical basis     L_{i;a,b} = q^{ab} L_i^a L_{i+1}^b.
Automorphism        rho(L_i) = L_{i+2}     (PentagonAlgebra.rho).
Trace cyclicity     Tr(X Y) = Tr(rho^2(Y) X)   (KAlgebra contract).

Reduction.  Let  t_n := Tr L_i^n.  It is i-independent because the
trace is rho^2-invariant and rho^2 generates Z_5.

(R1)   Tr L_i^a L_{i+1}^b   =   Tr L_i^{a+1} L_{i+1}^{b-1}
       (cyclicity applied to the rightmost L_{i+1}: with the convention
       rho(L_i) = L_{i+2}, rho^2(L_{i+1}) = L_i; equivalently the author's
       form  Tr L_{i+1} L_i^a = Tr L_i^a L_{i+2}, which differs by an
       overall rho-shift inside the Z_5-invariant trace.)
       Iterating b times:        Tr L_i^a L_{i+1}^b  =  t_{a+b}.

(R2)   Tr L_i^a L_{i+1}  =  q^{-2a} Tr L_{i+1} L_i^a
                          =  q^{-2a} Tr L_i^a L_{i+2}
                          =  q^{-2a} ( Tr L_i^{a-1} + q^{-1} Tr L_i^{a-1} L_{i+1} )
       using  L_{i+1}L_i = q^2 L_i L_{i+1}  and  L_i L_{i+2} = 1 + q^{-1} L_{i+1}.

       Combined with R1 the LHS is t_{a+1} and the trailing trace is t_a:

           t_{a+1}  =  q^{-2a} t_{a-1}  +  q^{-(2a+1)} t_a     (a >= 1).

       Base cases  t_0 = T0 := Tr 1   and   t_1 = T1 := Tr L_1.

Hence  Tr L_{i;a,b} = q^{ab} t_{a+b}  reduces every trace value to a
Z[q, q^{-1}]-linear combination of the two unknowns T0 and T1.
"""

from __future__ import annotations

from laurent_poly import LaurentPoly
from pentagon_algebra import PentagonAlgebra, _canon_key


def t_coeffs(n: int) -> tuple[LaurentPoly, LaurentPoly]:
    """(c0, c1) with  Tr L_i^n = c0(q) * T0 + c1(q) * T1."""
    if n < 0:
        raise ValueError(f"n must be >= 0 (got {n})")
    c0p, c1p = LaurentPoly.one(),  LaurentPoly.zero()    # t_0
    if n == 0:
        return c0p, c1p
    c0c, c1c = LaurentPoly.zero(), LaurentPoly.one()     # t_1
    for k in range(1, n):
        a = LaurentPoly.q(-2 * k)
        b = LaurentPoly.q(-(2 * k + 1))
        c0n = a * c0p + b * c0c
        c1n = a * c1p + b * c1c
        c0p, c1p = c0c, c1c
        c0c, c1c = c0n, c1n
    return c0c, c1c


def trace(P: PentagonAlgebra) -> tuple[LaurentPoly, LaurentPoly]:
    """Tr(P) = c0(q) T0 + c1(q) T1, where T0 = Tr 1, T1 = Tr L_1.

    Uses  Tr L_{i;a,b} = q^{ab} t_{a+b}.
    """
    c0 = LaurentPoly.zero()
    c1 = LaurentPoly.zero()
    for (i, a, b), coeff in P.terms():
        d0, d1 = t_coeffs(a + b)
        scale = coeff * LaurentPoly.q(a * b)
        c0 = c0 + scale * d0
        c1 = c1 + scale * d1
    return c0, c1


# ----------------------------------------------------------------------
# Pretty-printing helpers and demo / self-check.
# ----------------------------------------------------------------------

def _laurent_coeff(p: LaurentPoly, n: int) -> int:
    return p._coeffs.get(n, 0)


def _fmt_lpoly(p: LaurentPoly) -> str:
    if p.is_zero():
        return "0"
    out = []
    for n in sorted(p._coeffs):
        c = p._coeffs[n]
        sgn = "+" if c > 0 else "-"
        ac = abs(c)
        if n == 0:
            tok = f"{ac}"
        elif n == 1:
            tok = "q" if ac == 1 else f"{ac}*q"
        else:
            tok = f"q^{n}" if ac == 1 else f"{ac}*q^{n}"
        if not out and sgn == "+":
            out.append(tok)
        else:
            out.append(f" {sgn} {tok}")
    return "".join(out)


def _fmt_pair(p) -> str:
    c0, c1 = p
    if c0.is_zero() and c1.is_zero():
        return "0"
    if c0.is_zero():
        return f"[{_fmt_lpoly(c1)}] T1"
    if c1.is_zero():
        return f"[{_fmt_lpoly(c0)}] T0"
    return f"[{_fmt_lpoly(c0)}] T0  +  [{_fmt_lpoly(c1)}] T1"


def _fmt_series(coeffs: list[int]) -> str:
    out = []
    for n, c in enumerate(coeffs):
        if c == 0:
            continue
        sgn = "+" if c > 0 else "-"
        ac = abs(c)
        if n == 0:
            tok = f"{ac}"
        elif n == 1:
            tok = "q" if ac == 1 else f"{ac}*q"
        else:
            tok = f"q^{n}" if ac == 1 else f"{ac}*q^{n}"
        if not out and sgn == "+":
            out.append(tok)
        else:
            out.append(f" {sgn} {tok}")
    return "".join(out) if out else "0"


def _demo() -> None:
    A = PentagonAlgebra
    one = A.one()

    print("=" * 72)
    print("(1) Pentagon relations (each entry should be True):")
    for i in range(5):
        Li, Lip, Lim = A.L(i), A.L(i + 1), A.L(i - 1)
        r1 = (Lip * Li - LaurentPoly.q(2) * Li * Lip).is_zero()
        r2 = (Lip * Lim - (one + LaurentPoly.q(1)  * Li)).is_zero()
        r3 = (Lim * Lip - (one + LaurentPoly.q(-1) * Li)).is_zero()
        print(f"   i={i}:  L_(i+1)L_i=q^2 L_i L_(i+1) : {r1}"
              f"   L_(i+1)L_(i-1)=1+q L_i : {r2}"
              f"   L_(i-1)L_(i+1)=1+q^-1 L_i : {r3}")

    print("\n(2) rho is an algebra automorphism of order 5:")
    x = A.L(0) * A.L(2) + LaurentPoly.q(3) * A.L(1)
    y = A.L(1) + A.L(3) * A.L(4)
    print(f"   rho(x y) = rho(x) rho(y):  {(x * y).rho() == x.rho() * y.rho()}")
    p = x
    for _ in range(5):
        p = p.rho()
    print(f"   rho^5 = id:                {p == x}")

    print("\n(3) Trace cyclicity  Tr(XY) - Tr(rho^2(Y) X)  (should vanish):")
    samples = [
        (A.L(0), A.L(1)),
        (A.L(0) * A.L(2), A.L(3)),
        (A.L(1) * A.L(4), A.L(0) * A.L(1)),
        (A.basis(0, 2, 1), A.basis(2, 1, 1)),
        (A.basis(1, 3, 2), A.L(0) * A.L(2)),
    ]
    for (X, Y) in samples:
        lhs = trace(X * Y)
        rhs = trace(Y.rho().rho() * X)
        d0 = lhs[0] - rhs[0]
        d1 = lhs[1] - rhs[1]
        print(f"   diff = [{_fmt_lpoly(d0)}] T0 + [{_fmt_lpoly(d1)}] T1")

    print("\n(4) Trace recursion  t_n = Tr L_i^n  (T0 = Tr 1, T1 = Tr L_1):")
    for n in range(6):
        print(f"   t_{n}  =  {_fmt_pair(t_coeffs(n))}")

    print("\n    Z_5 invariance of trace  (Tr L_{i;a,b} independent of i):")
    for (a, b) in [(2, 0), (3, 0), (1, 1), (2, 1), (2, 2)]:
        vals = [trace(A.basis(i, a, b)) for i in range(5)]
        ok = all(v == vals[0] for v in vals)
        print(f"      Tr L_{{i;{a},{b}}} same for all i in Z/5: {ok}")

    print("\n    Tr L_{i;a,b} = q^{ab} t_{a+b}  table:")
    for (a, b) in [(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (2, 2), (3, 2)]:
        v = trace(A.basis(0, a, b))
        print(f"      Tr L_{{0;{a},{b}}} = {_fmt_pair(v)}")

    print("\n(5) Pairing matrix  I[a,b] = Tr rho(a) b   on simples")
    print("    {1, L_0, L_1, L_2, L_3, L_4}:")
    labels = ["1"] + [f"L_{i}" for i in range(5)]
    elems  = [A.one()] + [A.L(i) for i in range(5)]
    pairings: dict[tuple[str, str], tuple[LaurentPoly, LaurentPoly]] = {}
    for la, a in zip(labels, elems):
        for lb, b in zip(labels, elems):
            pairings[(la, lb)] = trace(a.rho() * b)
            print(f"   I[{la:>3s}, {lb:>3s}] = {_fmt_pair(pairings[(la, lb)])}")

    # (6) Pin (T0, T1) by solving the linear system that  I_{ab} = delta_{ab}
    # + O(q)  imposes on the unknown power-series coefficients.
    #
    # Write  T0 = sum_{m>=0} alpha_m q^m,  T1 = sum_{m>=0} beta_m q^m.
    # For each ordered pair (a,b) of canonical basis elements,
    #
    #     c0(q) * T0  +  c1(q) * T1  -  delta_{ab}    in  q * Z[[q]],
    #
    # i.e. its coefficient at q^k vanishes for every  k <= 0.  Each such k
    # is a linear equation in finitely many alpha_m, beta_m (m up to the
    # minimum q-power -k_min of c0, c1).
    #
    # We enrich the basis with all canonical labels L_{i;a,b} with a+b <= D
    # (D = 1 = simples, D = 2 adds L_{i;2,0} and L_{i;1,1}, ...).
    from fractions import Fraction

    def _solve_with_basis(basis_labels):
        # Compute the pairing matrix on the chosen basis.
        pairs: dict[tuple, tuple[LaurentPoly, LaurentPoly]] = {}
        elts = {lab: A.basis(*lab) for lab in basis_labels}
        for la, ea in elts.items():
            for lb, eb in elts.items():
                pairs[(la, lb)] = trace(ea.rho() * eb)
        # Determine number of unknowns from negative q-power reach.
        min_pow = 0
        for c0, c1 in pairs.values():
            for p in (c0, c1):
                if p._coeffs:
                    min_pow = min(min_pow, min(p._coeffs))
        N = -min_pow + 1                # alpha_0..alpha_{N-1}, beta_0..beta_{N-1}
        # Build equations [q^k] (c0*T0 + c1*T1 - delta_{ab}) = 0  for k <= 0.
        # Unknown order:  alpha_0, ..., alpha_{N-1}, beta_0, ..., beta_{N-1}.
        rows = []
        rhss = []
        for (la, lb), (c0, c1) in pairs.items():
            for k in range(min_pow, 1):                 # k = min_pow .. 0
                row = [Fraction(0)] * (2 * N)
                for m in range(N):
                    row[m]     = Fraction(_laurent_coeff(c0, k - m))
                    row[N + m] = Fraction(_laurent_coeff(c1, k - m))
                target = Fraction(1 if (la == lb and k == 0) else 0)
                rows.append(row)
                rhss.append(target)
        # Gaussian elimination on the (rows | rhss) system.
        M = [r + [v] for r, v in zip(rows, rhss)]
        nrows = len(M)
        ncols = 2 * N
        pivots: list[tuple[int, int]] = []
        r = 0
        for col in range(ncols):
            piv = None
            for i in range(r, nrows):
                if M[i][col] != 0:
                    piv = i
                    break
            if piv is None:
                continue
            M[r], M[piv] = M[piv], M[r]
            inv = Fraction(1) / M[r][col]
            M[r] = [v * inv for v in M[r]]
            for i in range(nrows):
                if i == r:
                    continue
                if M[i][col] != 0:
                    f = M[i][col]
                    M[i] = [M[i][c] - f * M[r][c] for c in range(ncols + 1)]
            pivots.append((r, col))
            r += 1
        # Check consistency on remaining zero rows.
        for i in range(r, nrows):
            if M[i][ncols] != 0:
                return None, "inconsistent"
        # Read solution; non-pivot columns are free.
        pivot_col = {col: row for row, col in pivots}
        free_cols = [c for c in range(ncols) if c not in pivot_col]
        if free_cols:
            return (pivot_col, free_cols, M, N), "partial"
        sol_alpha = [int(M[pivot_col[m]][ncols])     for m in range(N)]
        sol_beta  = [int(M[pivot_col[N + m]][ncols]) for m in range(N)]
        return (sol_alpha, sol_beta), "ok"

    print("\n(6) Linear constraints from  I_{ab} = delta_{ab} + O(q)  on")
    print("    alpha_m = [q^m] T0,  beta_m = [q^m] T1.  Enrich the canonical")
    print("    basis with all L_{i;a,b}, a+b <= D.  System is under-determined")
    print("    on its own; what it gives is an infinite chain of relations.")
    for D in (1, 2):
        labels = [(0, 0, 0)]                 # the unit
        for i in range(5):
            for a in range(D + 1):
                for b in range(D + 1):
                    if a == 0 and b == 0:
                        continue
                    if a + b > D:
                        continue
                    lab = _canon_key(i, a, b)
                    if lab not in labels:
                        labels.append(lab)
        result, status = _solve_with_basis(labels)
        if status == "ok":
            alpha, beta = result
            # Trust only the coefficients pinned at orders <= some threshold;
            # higher-order alpha_m, beta_m may pick up artefacts from the
            # finite truncation.  We report all N of them but mark the
            # "fully constrained" prefix.
            print(f"\n  D = {D}  (#basis = {len(labels)},  N = {len(alpha)} orders):")
            print(f"    T0 = Tr 1    =  {_fmt_series(alpha)}")
            print(f"    T1 = Tr L_1  =  {_fmt_series(beta)}")
        elif status == "partial":
            pivot_col, free_cols, M, N = result
            print(f"\n  D = {D}  (#basis = {len(labels)},  N = {N} orders):  "
                  f"{len(free_cols)} free parameter(s)")
            # Print the pinned coefficients (those whose row has a single
            # pivot value in the alpha/beta block and no free dependencies).
            def coeff_expr(idx):
                if idx not in pivot_col:
                    name = (f"alpha_{idx}" if idx < N else f"beta_{idx - N}")
                    return name + "  (free)"
                row = M[pivot_col[idx]]
                # Constant part:
                const = row[2 * N]
                deps = []
                for c in free_cols:
                    if row[c] != 0:
                        nm = (f"alpha_{c}" if c < N else f"beta_{c - N}")
                        deps.append(f"({-row[c]})*{nm}")
                expr = f"{const}"
                if deps:
                    expr += " + " + " + ".join(deps)
                return expr
            UP_TO = min(8, N)
            for m in range(UP_TO):
                print(f"    alpha_{m} = {coeff_expr(m)}")
            for m in range(UP_TO):
                print(f"    beta_{m}  = {coeff_expr(N + m)}")
            # Also display the homogeneous-form relations  L(alpha, beta) = c.
            print("    Equivalently, the inhomogeneous linear relations:")
            for m in range(UP_TO):
                if m not in pivot_col:
                    continue
                row = M[pivot_col[m]]
                lhs_terms = [f"alpha_{m}"]
                for c in free_cols:
                    if row[c] != 0:
                        nm = (f"alpha_{c}" if c < N else f"beta_{c - N}")
                        coef = row[c]
                        sgn = "+" if coef > 0 else "-"
                        ac  = abs(coef)
                        lhs_terms.append(f"{sgn} {ac}*{nm}" if ac != 1
                                          else f"{sgn} {nm}")
                rhs = row[2 * N]
                print(f"      {' '.join(lhs_terms)} = {rhs}")
            for m in range(UP_TO):
                if (N + m) not in pivot_col:
                    continue
                row = M[pivot_col[N + m]]
                lhs_terms = [f"beta_{m}"]
                for c in free_cols:
                    if row[c] != 0:
                        nm = (f"alpha_{c}" if c < N else f"beta_{c - N}")
                        coef = row[c]
                        sgn = "+" if coef > 0 else "-"
                        ac  = abs(coef)
                        lhs_terms.append(f"{sgn} {ac}*{nm}" if ac != 1
                                          else f"{sgn} {nm}")
                rhs = row[2 * N]
                print(f"      {' '.join(lhs_terms)} = {rhs}")
        else:
            print(f"\n  D = {D}: INCONSISTENT — this would mean the proposed"
                  " orthonormality is violated.")
            break

    print("\n(7) Tr rho(L_{i;a,b}) L_{j;c,d}  for a few samples"
          " (as c0(q) T0 + c1(q) T1):")
    for (i, a, b, j, c, d) in [
        (0, 1, 0, 0, 1, 0),   # Tr rho(L_0) L_0
        (0, 1, 0, 1, 1, 0),   # Tr rho(L_0) L_1
        (0, 1, 0, 2, 1, 0),   # Tr rho(L_0) L_2  = Tr L_2 L_2 = t_2
        (0, 2, 0, 0, 0, 1),   # Tr rho(L_0^2) L_1
        (0, 1, 1, 0, 1, 1),   # Tr rho(L_{0;1,1}) L_{0;1,1}
        (1, 2, 1, 3, 1, 2),
    ]:
        X = A.basis(i, a, b).rho()
        Y = A.basis(j, c, d)
        v = trace(X * Y)
        print(f"   Tr rho(L_{{{i};{a},{b}}}) L_{{{j};{c},{d}}}  =  {_fmt_pair(v)}")

    # (8) Specialise to  T0 = Tr 1 = 1  exactly and solve for T1 = Tr L_1
    # order by order from the O(q) conditions.
    #
    # Single-equation argument (cf. Tr rho(L_{i+1}) L_i = T0 + q^{-1} T1):
    #     0 + O(q) = T0 + q^{-1} T1   ==>   T1 = - q T0 + O(q^2) = -q + O(q^2).
    # High-degree pairings have c0, c1 with much lower q-powers; setting their
    # negative-power piece to zero pins many orders of T1 at once.
    from fractions import Fraction

    def _solve_T1_given_T0_is_one(D: int, target_order: int):
        # Enumerate canonical basis with a+b <= D.
        labels = [(0, 0, 0)]
        for i in range(5):
            for a in range(D + 1):
                for b in range(D + 1):
                    if a == 0 and b == 0:
                        continue
                    if a + b > D:
                        continue
                    lab = _canon_key(i, a, b)
                    if lab not in labels:
                        labels.append(lab)
        # Pairing matrix.
        elts = {lab: A.basis(*lab) for lab in labels}
        pairs = {}
        for la, ea in elts.items():
            for lb, eb in elts.items():
                pairs[(la, lb)] = trace(ea.rho() * eb)
        # Linear system in beta_0 .. beta_{N-1}.
        N = target_order
        rows, rhss = [], []
        for (la, lb), (c0, c1) in pairs.items():
            # Minimum q-power appearing in this pair's expression.
            qpows = list(c0._coeffs) + list(c1._coeffs)
            if not qpows:
                continue
            kmin = min(qpows)
            for k in range(kmin, 1):                  # equations at q^k, k <= 0
                row = [Fraction(0)] * N
                for n in range(N):
                    row[n] = Fraction(_laurent_coeff(c1, k - n))
                target = Fraction(1 if (la == lb and k == 0) else 0)
                target -= Fraction(_laurent_coeff(c0, k))
                rows.append(row)
                rhss.append(target)
        # Gaussian elimination.
        M = [r + [v] for r, v in zip(rows, rhss)]
        nr, nc = len(M), N
        r = 0
        pivot_col: dict[int, int] = {}
        for col in range(nc):
            piv = next((i for i in range(r, nr) if M[i][col] != 0), None)
            if piv is None:
                continue
            M[r], M[piv] = M[piv], M[r]
            inv = Fraction(1) / M[r][col]
            M[r] = [v * inv for v in M[r]]
            for i in range(nr):
                if i != r and M[i][col] != 0:
                    f = M[i][col]
                    M[i] = [M[i][c] - f * M[r][c] for c in range(nc + 1)]
            pivot_col[col] = r
            r += 1
        # Check consistency.
        inconsistent = any(
            all(v == 0 for v in M[i][:nc]) and M[i][nc] != 0
            for i in range(nr)
        )
        # Read off pinned coefficients.
        beta = []
        for m in range(nc):
            if m in pivot_col:
                beta.append(M[pivot_col[m]][nc])
            else:
                beta.append(None)
        return beta, inconsistent, len(labels)

    print("\n(8) Specialise  T0 = Tr 1 = 1.  Solve  I_{ab} = delta_{ab} + O(q)")
    print("    for the q-series  T1 = Tr L_1 = sum beta_m q^m.")
    for D in (1, 2, 3):
        beta, bad, nb = _solve_T1_given_T0_is_one(D, target_order=20)
        # Determine the longest pinned prefix.
        prefix_len = 0
        for v in beta:
            if v is None:
                break
            prefix_len += 1
        # Format.
        terms = []
        for m in range(prefix_len):
            v = beta[m]
            assert v.denominator == 1, f"non-integer beta_{m} = {v} (bug?)"
            c = int(v)
            if c == 0:
                continue
            sgn = "+" if c > 0 else "-"
            ac = abs(c)
            if m == 0:
                tok = f"{ac}"
            elif m == 1:
                tok = "q" if ac == 1 else f"{ac}*q"
            else:
                tok = f"q^{m}" if ac == 1 else f"{ac}*q^{m}"
            if not terms and sgn == "+":
                terms.append(tok)
            else:
                terms.append(f" {sgn} {tok}")
        body = "".join(terms) if terms else "0"
        tag = "  [INCONSISTENT]" if bad else ""
        print(f"   D = {D}  (#basis = {nb}, pinned through q^{prefix_len - 1}):")
        print(f"     Tr L_1  =  {body}  + O(q^{prefix_len}){tag}")

    # (9) Check the author's proposed "ratio" formula:  setting the negative-q-power
    # part of  Tr rho(L_{1;2,1}) L_{3;1,2} = c0(q) T0 + c1(q) T1  to zero with
    # T0 = 1 gives T1 = -c0_neg(q) / c1_neg(q)  to all orders the single
    # equation can reach.
    c0, c1 = trace(A.basis(1, 2, 1).rho() * A.basis(3, 1, 2))
    print("\n(9) Single-equation pin from  Tr rho(L_{1;2,1}) L_{3;1,2} = 0 + O(q),"
          " T0 = 1:")
    # Negative-q-power parts.
    c0_neg = {p: v for p, v in c0._coeffs.items() if p < 0}
    c1_neg = {p: v for p, v in c1._coeffs.items() if p < 0}
    pmin = min(list(c0_neg) + list(c1_neg))
    # Solve for beta_0 ... beta_{(-pmin)-1} from [q^k] (c0 + c1 T1) = 0,  k <= 0.
    # (Top equation at k = pmin involves only beta_0; descend.)
    N = -pmin
    rows, rhss = [], []
    for k in range(pmin, 1):
        row = [Fraction(_laurent_coeff(c1, k - n)) for n in range(N)]
        rhs = -Fraction(_laurent_coeff(c0, k))
        rows.append(row)
        rhss.append(rhs)
    # Gaussian elimination.
    M = [r + [v] for r, v in zip(rows, rhss)]
    r = 0
    pivot_col = {}
    for col in range(N):
        piv = next((i for i in range(r, len(M)) if M[i][col] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = Fraction(1) / M[r][col]
        M[r] = [v * inv for v in M[r]]
        for i in range(len(M)):
            if i != r and M[i][col] != 0:
                f = M[i][col]
                M[i] = [M[i][c] - f * M[r][c] for c in range(N + 1)]
        pivot_col[col] = r
        r += 1
    beta_single = [int(M[pivot_col[m]][N]) if m in pivot_col else None
                   for m in range(N)]
    pref = 0
    for v in beta_single:
        if v is None:
            break
        pref += 1
    body = []
    for m in range(pref):
        c = beta_single[m]
        if c == 0:
            continue
        sgn = "+" if c > 0 else "-"
        ac = abs(c)
        tok = ("q" if ac == 1 else f"{ac}*q") if m == 1 else (
            f"{ac}" if m == 0 else (f"q^{m}" if ac == 1 else f"{ac}*q^{m}")
        )
        body.append(tok if not body and sgn == "+" else f" {sgn} {tok}")
    print(f"   That one equation alone pins {pref} coefficients of T1:")
    print(f"     T1  =  {''.join(body) if body else '0'}  + O(q^{pref})")


if __name__ == "__main__":
    _demo()
