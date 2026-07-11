"""The map phi : S^s(T) -> Y(T) on a single triangle arc.

In each ideal triangle T with edges (a, b, c) in counterclockwise order,
there are three arc types, each opposite to one edge:
    alpha : opposite edge a, connects b and c.   tau-image:    alpha
    beta  : opposite edge b, connects c and a.   tau-image:    tau(alpha)
    gamma : opposite edge c, connects a and b.   tau-image:    tau^2(alpha)

State convention:  in the ccw boundary
traversal a -> b -> c -> a, an arc's "first index" eps is the state at
the ccw-LATER endpoint and the "second index" eps' is at the ccw-EARLIER.
For each arc:
    alpha:  eps at c, eps' at b.
    beta :  eps at a, eps' at c.
    gamma:  eps at b, eps' at a.

Formula (with its tau-rotations):
    phi(alpha(eps, eps')) = [y_c^eps y_b^{eps'}]   in un-ordered form
                          = lq^{eps eps' / 2} y_c^eps y_b^{eps'}
    phi(beta(eps, eps'))  = [y_a^eps y_c^{eps'}]
                          = lq^{eps eps' / 2} y_a^eps y_c^{eps'}
    phi(gamma(eps, eps')) = [y_b^eps y_a^{eps'}]
                          = lq^{eps eps' / 2} y_b^eps y_a^{eps'}
in all three cases, vanishing if (eps, eps') = (-, +).

In our canonical ordered form (alphabetical a < b < c) these become:
    phi(alpha) = lq^{-eps eps' / 2} y_b^{eps'} y_c^eps      (exp (0, eps', eps))
    phi(beta)  = lq^{+eps eps' / 2} y_a^eps  y_c^{eps'}     (exp (eps, 0, eps'))
    phi(gamma) = lq^{-eps eps' / 2} y_a^{eps'} y_b^eps      (exp (eps', eps, 0))

The +/- sign of the lq exponent in ordered form depends on whether the
un-ordered product is already alphabetical:
    alpha (c, b):  swap -> extra lq^{-eps eps'}, net lq^{-eps eps'/2}.
    beta  (a, c):  no swap,                          net lq^{+eps eps'/2}.
    gamma (b, a):  swap -> extra lq^{-eps eps'}, net lq^{-eps eps'/2}.

Self-contained: depends only on half_laurent and
triangle_algebra.
"""

from __future__ import annotations

from fractions import Fraction

from half_laurent import HalfLaurent
from triangle_algebra import TriangleAlgebra


# Arc type labels.
ALPHA = "alpha"
BETA = "beta"
GAMMA = "gamma"

ARC_TYPES = (ALPHA, BETA, GAMMA)


def phi(arc_type: str, eps: int, eps_prime: int) -> TriangleAlgebra:
    """phi(arc_type(eps, eps')) in Y(T), in canonical ordered form.

    arc_type    : one of "alpha", "beta", "gamma".
    eps, eps_p  : in {+1, -1}.  Convention: eps is the state at the
                  ccw-later endpoint; eps_prime at the ccw-earlier.

    Returns the TriangleAlgebra element described in the module
    docstring.  Vanishing case (-, +) returns the zero element.
    """
    if arc_type not in ARC_TYPES:
        raise ValueError(
            f"arc_type must be one of {ARC_TYPES}, got {arc_type!r}"
        )
    if eps not in (1, -1) or eps_prime not in (1, -1):
        raise ValueError("eps, eps_prime must be in {+1, -1}")

    if eps == -1 and eps_prime == 1:
        return TriangleAlgebra.zero()

    e = eps * eps_prime
    if arc_type == ALPHA:
        coef = HalfLaurent.monomial(Fraction(-e, 2))
        return TriangleAlgebra.monomial((0, eps_prime, eps), coef)
    if arc_type == BETA:
        coef = HalfLaurent.monomial(Fraction(e, 2))
        return TriangleAlgebra.monomial((eps, 0, eps_prime), coef)
    if arc_type == GAMMA:
        coef = HalfLaurent.monomial(Fraction(-e, 2))
        return TriangleAlgebra.monomial((eps_prime, eps, 0), coef)
    raise AssertionError("unreachable")


def is_annihilating(eps: int, eps_prime: int) -> bool:
    """The (-, +) state pattern that makes phi vanish."""
    return eps == -1 and eps_prime == 1
