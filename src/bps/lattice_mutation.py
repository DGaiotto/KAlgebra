"""
Conjugation by the quantum dilogarithm in the quantum torus algebra
Q_Gamma of a general lattice Gamma.

Given O in Q_Gamma and a primitive lattice vector gamma in Gamma with
<gamma, gamma> = 0, we want to solve

    O * E_q(X_gamma)  =  E_q(X_gamma) * O'                        (*)

for an *undetermined* element O' of Q_Gamma that must be a finite
Z[q, q^{-1}]-linear combination of basis monomials X_delta.  Formally,

    O'  =  E_q(X_gamma)^{-1} * O * E_q(X_gamma),

but the explicit series expansion of E_q is never used.  As in the
rank-2 implementation in ``mutation.py``, the problem
decouples along the gamma-lines of the support of O.  A gamma-line is an
orbit of the translation  alpha -> alpha + gamma  inside Gamma; the
"offset" k parametrises alpha = beta + k*gamma where beta is the unique
representative of the line orthogonal to an auxiliary Bezout functional u
with u . gamma = 1.  Since <gamma, gamma> = 0, the integer
m := <gamma, alpha> only depends on the line, not on k.

Along a line with pairing m, equation (*) becomes a *purely 1d* relation
in the generating polynomial in the shift variable z:

    m == 0 :  O' agrees with O on this line (pass-through).
    m <  0 :  always solvable.  The new offset-polynomial is
              C'(z) = C(z) * B_{|m|}(z)  with B_n(z) = sum_k [n,k]_q z^k,
              i.e. a single monomial expands into a q-binomial packet.
    m >  0 :  solvable iff C(z) is divisible by B_m(z); then
              C'(z) = C(z) / B_m(z)  and a q-binomial packet contracts
              back to a single monomial (or a shorter packet).

Thus O is finitely conjugable iff every m > 0 gamma-line of O is a
q-binomial packet, and this module provides the checks, the solver, the
minimal completion that turns a given O into a solvable one, and a
diagnostic decomposition.

All of the q-binomial bookkeeping is shared with ``mutation.py``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from laurent_poly import LaurentPoly
from lattice import Lattice, LatticeTorus, Vec
from mutation import _ascend, _bpoly, _conv, _divmod_desc, _xgcd


# ---------------------------------------------------------------------------
# Bezout functional for a primitive gamma
# ---------------------------------------------------------------------------


def bezout_cofactor(gamma: Sequence[int]) -> tuple[int, ...]:
    """Return an integer vector u with sum_i u_i * gamma_i = 1.

    Such a u exists iff gamma is primitive, i.e. gcd of its coordinates is 1.
    It lets us parametrise every alpha in Z^n uniquely as

        alpha = beta + k * gamma      with     k = u . alpha
                                      and     u . beta = 0.
    """
    g = 0
    n = len(gamma)
    u = [0] * n
    for i, gi in enumerate(gamma):
        if gi == 0:
            continue
        if g == 0:
            g = abs(gi)
            u[i] = 1 if gi > 0 else -1
        else:
            ng, s, t = _xgcd(g, gi)
            u = [s * x for x in u]
            u[i] += t
            g = ng
        if g == 1:
            break
    if g != 1:
        raise ValueError(f"gamma={tuple(gamma)} is not primitive (gcd={g})")
    # sanity check
    assert sum(ui * gi for ui, gi in zip(u, gamma)) == 1
    return tuple(u)


# ---------------------------------------------------------------------------
# Internal: decomposition along gamma-lines
# ---------------------------------------------------------------------------


def _decompose(element: LatticeTorus, gamma: Vec, u: Vec):
    """Split ``element`` into gamma-lines.

    Returns
    -------
    lines : dict[Vec, dict[..., LaurentPoly]]
        Map from the line representative ``beta`` (with u . beta = 0) to a
        dict ``{k: coefficient of X_{beta + k*gamma}}``.
        k is int when element charges are integral, Fraction otherwise.
    m_of_line : dict[Vec, int]
        Map from ``beta`` to the integer ``m = <gamma, beta>`` which equals
        ``<gamma, alpha>`` for any alpha on that line.
    """
    lines: dict[Vec, dict] = defaultdict(dict)
    m_of_line: dict[Vec, int] = {}
    bracket = element.lattice.bracket
    for alpha, coeff in element.terms():
        k = 0
        for ui, ai in zip(u, alpha):
            if ui:
                k += ui * ai
        # Normalise k to int when possible
        if isinstance(k, int) or k == int(k):
            k = int(k)
        beta = tuple(ai - k * gi for ai, gi in zip(alpha, gamma))
        lines[beta][k] = coeff
        if beta not in m_of_line:
            m = bracket(gamma, beta)
            m_of_line[beta] = int(m) if m == int(m) else m
    return dict(lines), m_of_line


def _recover(beta: Vec, k, gamma: Vec) -> Vec:
    return tuple(bi + k * gi for bi, gi in zip(beta, gamma))


def _require_isotropic(lattice: Lattice, gamma: Vec) -> None:
    s = lattice.bracket(gamma, gamma)
    if s != 0:
        raise ValueError(
            f"conjugation by E_q(X_gamma) requires <gamma, gamma> = 0, "
            f"got <{gamma},{gamma}> = {s}"
        )


def _prepare(element: LatticeTorus, gamma: Sequence[int]):
    lattice = element.lattice
    g = lattice.check(gamma)
    _require_isotropic(lattice, g)
    u = bezout_cofactor(g)
    return lattice, g, u


def _build_from_dict(lattice: Lattice, terms: dict[Vec, LaurentPoly]) -> LatticeTorus:
    out = LatticeTorus(lattice)
    out._terms = {k: v for k, v in terms.items() if not v.is_zero()}
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve(element: LatticeTorus, gamma: Sequence[int]) -> LatticeTorus:
    """Return the unique O' with  O * E_q(X_gamma) = E_q(X_gamma) * O'.

    Raises ``ValueError`` if no finite O' exists, i.e. some gamma-line of
    ``element`` with positive pairing fails to be a q-binomial packet.
    """
    lattice, g, u = _prepare(element, gamma)
    if element.is_zero():
        return LatticeTorus.zero(lattice)

    lines, m_of = _decompose(element, g, u)
    acc: dict[Vec, LaurentPoly] = {}

    def add(ch: Vec, c: LaurentPoly) -> None:
        if c.is_zero():
            return
        if ch in acc:
            s = acc[ch] + c
            if s.is_zero():
                del acc[ch]
            else:
                acc[ch] = s
        else:
            acc[ch] = c

    for beta, offs in lines.items():
        m = m_of[beta]
        if m == 0:
            for k, c in offs.items():
                add(_recover(beta, k, g), c)
        elif m < 0:
            img = _conv(offs, _bpoly(-m))
            for k, c in img.items():
                add(_recover(beta, k, g), c)
        else:  # m > 0 : must be divisible by B_m(z)
            Q, R = _divmod_desc(offs, _bpoly(m))
            if any(not v.is_zero() for v in R.values()):
                raise ValueError(
                    f"not finitely conjugable: gamma-line at beta={beta} has "
                    f"<gamma, beta>={m} > 0 and is not a q-binomial packet"
                )
            for k, c in Q.items():
                add(_recover(beta, k, g), c)

    return _build_from_dict(lattice, acc)


def can_solve(element: LatticeTorus, gamma: Sequence[int]) -> bool:
    """Return True iff O * E_q(X_gamma) = E_q(X_gamma) * O' has a finite
    solution O' in Q_Gamma.

    This runs the same check as :func:`solve` but only inspects the
    positive-pairing gamma-lines for divisibility and never materialises
    the full image, so it is cheap.
    """
    lattice, g, u = _prepare(element, gamma)
    if element.is_zero():
        return True
    lines, m_of = _decompose(element, g, u)
    for beta, offs in lines.items():
        m = m_of[beta]
        if m > 0:
            _, R = _divmod_desc(offs, _bpoly(m))
            if any(not v.is_zero() for v in R.values()):
                return False
    return True


def complete_to_solvable(
    element: LatticeTorus, gamma: Sequence[int]
) -> tuple[LatticeTorus, LatticeTorus]:
    """Add the minimal correction that makes O * E_q(X_gamma) = E_q(X_gamma) * O'
    solvable.

    Every gamma-line with m > 0 that does not already form a q-binomial packet
    is completed by ascending division: the correction adds monomials *above*
    the current top offset of the line so that the line becomes a multiple of
    B_m(z).  Lines with m <= 0 are left alone.

    Returns ``(completed, correction)`` with ``completed = element + correction``
    and ``can_solve(completed, gamma) == True``.
    """
    lattice, g, u = _prepare(element, gamma)
    if element.is_zero():
        return element, LatticeTorus.zero(lattice)

    lines, m_of = _decompose(element, g, u)
    corr: dict[Vec, LaurentPoly] = {}

    for beta, offs in lines.items():
        m = m_of[beta]
        if m <= 0:
            continue
        Bm = _bpoly(m)
        kmax = max(offs)
        _, G = _ascend(offs, Bm, kmax)
        for k, c in G.items():
            ch = _recover(beta, k, g)
            if ch in corr:
                s = corr[ch] + c
                if s.is_zero():
                    del corr[ch]
                else:
                    corr[ch] = s
            else:
                corr[ch] = c

    correction = _build_from_dict(lattice, corr)
    return element + correction, correction


# ---------------------------------------------------------------------------
# Inverse direction: O' with  O' * E_q(X_gamma) = E_q(X_gamma) * O,
#                   i.e. O' = E_q(X_gamma) * O * E_q(X_gamma)^{-1}
#
# This is the "unmutate" operation from mutation.py: it expands lines with
# positive pairing (m > 0) and contracts lines with negative pairing (m < 0).
# Same gamma-line decomposition as solve(), only the roles of m > 0 and
# m < 0 are swapped.
# ---------------------------------------------------------------------------


def solve_inverse(
    element: LatticeTorus, gamma: Sequence[int]
) -> LatticeTorus:
    """Return the unique O' with  O' * E_q(X_gamma) = E_q(X_gamma) * O.

    Equivalently, ``O' = E_q(X_gamma) * O * E_q(X_gamma)^{-1}``.  Raises
    ``ValueError`` if no finite O' exists.
    """
    lattice, g, u = _prepare(element, gamma)
    if element.is_zero():
        return LatticeTorus.zero(lattice)

    lines, m_of = _decompose(element, g, u)
    acc: dict[Vec, LaurentPoly] = {}

    def add(ch: Vec, c: LaurentPoly) -> None:
        if c.is_zero():
            return
        if ch in acc:
            s = acc[ch] + c
            if s.is_zero():
                del acc[ch]
            else:
                acc[ch] = s
        else:
            acc[ch] = c

    for beta, offs in lines.items():
        m = m_of[beta]
        if m == 0:
            for k, c in offs.items():
                add(_recover(beta, k, g), c)
        elif m > 0:  # expand (opposite of solve)
            img = _conv(offs, _bpoly(m))
            for k, c in img.items():
                add(_recover(beta, k, g), c)
        else:  # m < 0 : must be divisible by B_{|m|}
            Q, R = _divmod_desc(offs, _bpoly(-m))
            if any(not v.is_zero() for v in R.values()):
                raise ValueError(
                    f"not inverse-conjugable: gamma-line at beta={beta} has "
                    f"<gamma, beta>={m} < 0 and is not a q-binomial packet"
                )
            for k, c in Q.items():
                add(_recover(beta, k, g), c)

    return _build_from_dict(lattice, acc)


def can_solve_inverse(
    element: LatticeTorus, gamma: Sequence[int]
) -> bool:
    """Return True iff ``solve_inverse`` would succeed."""
    lattice, g, u = _prepare(element, gamma)
    if element.is_zero():
        return True
    lines, m_of = _decompose(element, g, u)
    for beta, offs in lines.items():
        m = m_of[beta]
        if m < 0:
            _, R = _divmod_desc(offs, _bpoly(-m))
            if any(not v.is_zero() for v in R.values()):
                return False
    return True


def complete_to_inverse_solvable(
    element: LatticeTorus, gamma: Sequence[int]
) -> tuple[LatticeTorus, LatticeTorus]:
    """Add the minimal correction making ``solve_inverse`` applicable.

    Every gamma-line with m < 0 that is not already a q-binomial packet in
    B_{|m|} is completed by ascending division.  Lines with m >= 0 are
    left alone.
    """
    lattice, g, u = _prepare(element, gamma)
    if element.is_zero():
        return element, LatticeTorus.zero(lattice)

    lines, m_of = _decompose(element, g, u)
    corr: dict[Vec, LaurentPoly] = {}

    for beta, offs in lines.items():
        m = m_of[beta]
        if m >= 0:
            continue
        Bm = _bpoly(-m)
        kmax = max(offs)
        _, G = _ascend(offs, Bm, kmax)
        for k, c in G.items():
            ch = _recover(beta, k, g)
            if ch in corr:
                s = corr[ch] + c
                if s.is_zero():
                    del corr[ch]
                else:
                    corr[ch] = s
            else:
                corr[ch] = c

    correction = _build_from_dict(lattice, corr)
    return element + correction, correction


def joint_complete(
    element: LatticeTorus,
    fwd_gamma: Sequence[int],
    back_gammas: Sequence[Sequence[int]] = (),
    max_iter: int = 50,
    max_terms: int = 0,
) -> tuple[LatticeTorus, LatticeTorus]:
    """Add minimal corrections so that ``element`` is simultaneously:

    - forward-solvable through  E_q(X_{fwd_gamma})  (``solve`` succeeds), and
    - inverse-solvable through each  E_q(X_{back_gamma})  for
      ``back_gamma in back_gammas`` (``solve_inverse`` succeeds).

    Iterates the individual completions until a fixed point is reached.

    When ``max_terms > 0``, the iteration aborts and returns the current
    state if the element exceeds that many terms, as a safeguard against
    the exponential blowup that occurs for some higher-rank theories.
    """
    original = element
    F = element
    back_list = list(back_gammas)
    for _ in range(max_iter):
        changed = False
        F, c = complete_to_solvable(F, fwd_gamma)
        if not c.is_zero():
            changed = True
        for bg in back_list:
            F, c = complete_to_inverse_solvable(F, bg)
            if not c.is_zero():
                changed = True
        if not changed:
            return F, F - original
        if max_terms > 0 and len(F._terms) > max_terms:
            return F, F - original
    raise RuntimeError("joint_complete did not converge")


# ---------------------------------------------------------------------------
# Diagnostic: how does O split into monomials and q-binomial blocks?
# ---------------------------------------------------------------------------


def packet_decomposition(
    element: LatticeTorus, gamma: Sequence[int]
) -> list[dict]:
    """Describe how ``element`` splits along gamma-lines.

    For each nonempty gamma-line in the support of ``element`` the output
    contains a dict with keys

        'line'   : Vec      -- the orthogonal representative beta
        'm'      : int      -- <gamma, beta> (the line pairing)
        'type'   : str      -- one of {'pass', 'expand', 'packet', 'obstruction'}
                               'pass'        : m == 0, passes through
                               'expand'      : m <  0, always solvable
                               'packet'      : m >  0 and divisible by B_m
                               'obstruction' : m >  0 but NOT divisible
        'offsets': dict[int, LaurentPoly]  -- raw polynomial C(z) along the line
        'quotient' (only for 'packet'): dict[int, LaurentPoly]
                               -- C(z) / B_m(z), i.e. the contracted line
        'remainder' (only for 'obstruction'): dict[int, LaurentPoly]
                               -- nonzero residue under B_m-division

    The element is finitely conjugable by E_q(X_gamma) iff no entry has type
    'obstruction'.
    """
    lattice, g, u = _prepare(element, gamma)
    if element.is_zero():
        return []
    lines, m_of = _decompose(element, g, u)

    report: list[dict] = []
    for beta, offs in lines.items():
        m = m_of[beta]
        entry: dict = {
            "line": beta,
            "m": m,
            "offsets": dict(offs),
        }
        if m == 0:
            entry["type"] = "pass"
        elif m < 0:
            entry["type"] = "expand"
        else:
            Q, R = _divmod_desc(offs, _bpoly(m))
            R = {k: v for k, v in R.items() if not v.is_zero()}
            if R:
                entry["type"] = "obstruction"
                entry["remainder"] = R
            else:
                entry["type"] = "packet"
                entry["quotient"] = Q
        report.append(entry)
    return report


# ---------------------------------------------------------------------------
# Small demonstration
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Rank 2, symplectic: reproduces the q-Weyl example from mutation.py.
    L = Lattice.symplectic(1)
    X = lambda *g: LatticeTorus.monomial(L, g)

    gamma = (0, 1)
    print("=== Solving O * E_q(X_gamma) = E_q(X_gamma) * O' ===")
    print("Lattice: symplectic rank 2;  gamma =", gamma)

    for label, O in [
        ("X(1,0)", X(1, 0)),
        ("X(-1,0)", X(-1, 0)),
        ("X(-1,0) + X(-1,1)", X(-1, 0) + X(-1, 1)),
        ("X(0,-1)", X(0, -1)),
    ]:
        ok = can_solve(O, gamma)
        if ok:
            print(f"  {label:20s} -> O' = {solve(O, gamma)}")
        else:
            print(f"  {label:20s} -> NO finite solution")

    # Completion of an obstruction.
    print("\n--- Completion ---")
    O = X(1, 0)  # m = <gamma, (1,0)> = 0*0 - 1*1 = -1  -> expand, always fine
    print(f"  solve({O}) = {solve(O, gamma)}")
    # A genuine obstruction: <gamma, (1,-1)> = 1 but coefficient is 1, not B_1
    O = X(1, -1)
    print(f"  can_solve(X(1,-1)) = {can_solve(O, gamma)}")
    comp, corr = complete_to_solvable(O, gamma)
    print(f"    completion = {comp}")
    print(f"    correction = {corr}")
    print(f"    solve(completed) = {solve(comp, gamma)}")

    # Rank 3, arbitrary antisymmetric pairing.
    print("\n--- Rank 3 example ---")
    L3 = Lattice([[0, 1, 0], [-1, 0, 2], [0, -2, 0]])
    Y = lambda *g: LatticeTorus.monomial(L3, g)
    gamma3 = (1, 0, 0)
    # <gamma3, (a,b,c)> = b. So lines are grouped by (0, b, c) and
    # m = b.  A line with b=2 must be a B_2 packet to be solvable.
    O = Y(0, 2, 0) + (LaurentPoly.q(1) + LaurentPoly.q(-1)) * Y(1, 2, 0) + Y(2, 2, 0)
    print(f"  O = {O}")
    print(f"  can_solve = {can_solve(O, gamma3)}  (expected True, it is a B_2 packet)")
    print(f"  O' = {solve(O, gamma3)}")
