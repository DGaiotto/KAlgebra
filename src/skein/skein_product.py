"""Skein-algebra products and decomposition into a basis of curves.

The skein-algebra product alpha . beta of two skein elements maps under
the quantum trace to the Y_Delta product Tr^lq(alpha) * Tr^lq(beta).
This module provides convenience wrappers for:

  * `skein_product(triangulation, *charges)`: tropical-charge inputs
    produce simple closed curves; their Y_Delta product is returned.
    For multicurves, pass each component as a separate charge -- the
    product handles the Kauffman-bracket q-twists automatically via
    Y_Delta multiplication.

  * `decompose_against_basis(target, basis)`: try to express a Y_Delta
    element as a linear combination of given basis elements.  Returns
    the coefficients dict if successful, plus a residual element
    (zero on success).  Uses a 'unique-charge' heuristic that works
    when basis elements have disjoint or nearly-disjoint supports.

Usage example for S^2_{0,4} on the tetrahedral triangulation:

    import (
        Triangulation, quantum_trace, skein_product,
        decompose_against_basis,
    )

    t = Triangulation.tetrahedron_S2_4()
    ga = quantum_trace(t, (0, 1, 1, 1, 1, 0))     # gamma_{12|34}
    gb = quantum_trace(t, (1, 0, 1, 1, 0, 1))     # gamma_{13|24}
    gc = quantum_trace(t, (1, 1, 0, 0, 1, 1))     # gamma_{14|23}
    one = quantum_trace(t, (0, 0, 0, 0, 0, 0))    # vacuum

    product = ga * gb
    coefs, residual = decompose_against_basis(
        product, {"1": one, "gamma_a": ga, "gamma_b": gb, "gamma_c": gc},
    )
    # coefs is a dict of HalfLaurent values; residual is what couldn't be
    # explained by the chosen basis (may be non-zero, indicating extra
    # multicurves or higher-Chebyshev terms in the decomposition).

Self-contained: depends only on skein_sphere modules.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from half_laurent import HalfLaurent
from quantum_trace import quantum_trace
from triangulation import Triangulation
from y_delta import YDelta


def skein_product(
    triangulation: Triangulation,
    *charges: Sequence[int],
) -> YDelta:
    """Y_Delta product of Tr^lq for each input tropical charge.

    Each input charge is interpreted as a simple closed curve; the
    product corresponds to the skein-algebra product of those curves
    (which is naturally a sum over multicurve resolutions per the
    Kauffman bracket skein relation).

    With zero arguments, returns the multiplicative identity of Y_Delta.
    With one argument, returns Tr^lq of that curve.
    """
    if not charges:
        return YDelta.one(triangulation)
    result = quantum_trace(triangulation, charges[0])
    for c in charges[1:]:
        result = result * quantum_trace(triangulation, c)
    return result


def flavour_fugacity(
    triangulation: Triangulation,
    puncture: int,
    power: int = 1,
) -> YDelta:
    """The flavour fugacity mu_p^power as an element of Y_Delta.

    mu_p := X_{f_p}, where f_p is the puncture's flavour charge
    (= sum of edges incident to p, in ker(sigma)).  Since f_p is in
    ker(sigma), mu_p commutes with all of Y_Delta (it's central).

    Returned as a Weyl-form (X-monomial) at charge power * f_p, with
    coefficient 1.
    """
    f = triangulation.puncture_flavour_charge(puncture)
    charge = tuple(power * x for x in f)
    return YDelta.weyl_monomial(triangulation, charge, 1)


def peripheral_curve(
    triangulation: Triangulation,
    puncture: int,
) -> YDelta:
    """The peripheral curve around puncture p: mu_p + mu_p^{-1}.

    A small loop bounding puncture p in the SL_2-Kauffman skein
    algebra evaluates to the trace of the puncture's holonomy =
    mu_p + mu_p^{-1} where mu_p is the flavour fugacity.  This element
    is central in Y_Delta and serves as a 'puncture monodromy'
    parameter in skein-algebra structure constants.
    """
    return flavour_fugacity(triangulation, puncture, +1) + flavour_fugacity(
        triangulation, puncture, -1
    )


class BasisError(ValueError):
    """Raised when basis_element_for_charge cannot identify the basis
    element corresponding to a given tropical charge with the
    primitive curves provided."""


def basis_element_for_charge(
    triangulation: Triangulation,
    charge: Sequence[int],
    primitives: Sequence[Sequence[int]],
) -> YDelta:
    """Skein-algebra basis element labeled by tropical charge `charge`.

    The skein algebra has a linear basis labeled by isotopy classes of
    multicurves, each of which carries a tropical charge in Z^{n_edges}.
    For each realizable charge `a`, there is a unique basis element:

      * a = 0           : 1 (empty curve)
      * a = n * gamma_i : T_n(gamma_i) (Chebyshev-renormalized n parallel
                          copies of a primitive simple closed curve)
      * a = sum of non-intersecting components: product of components'
                          Tr^lq's (handled if you supply an explicit
                          decomposition; not auto-detected here).

    Args:
        triangulation: the ambient triangulation.
        charge:       the tropical charge labeling the basis element.
        primitives:   list of charges of the primitive simple closed
                      curves to consider for the n*gamma_i decomposition.
                      For S^2_{0,4} on the tetrahedral triangulation:
                      [(0,1,1,1,1,0), (1,0,1,1,0,1), (1,1,0,0,1,1)].

    Returns: the YDelta element of the labeled basis vector.
    Raises BasisError if `charge` is not zero and not a non-negative
    integer multiple of any primitive (i.e., requires an explicit
    multicurve decomposition that this helper doesn't perform).
    """
    n_edges = triangulation.n_edges
    if len(charge) != n_edges:
        raise ValueError(
            f"charge has length {len(charge)}, expected {n_edges}"
        )
    charge = tuple(charge)
    if all(x == 0 for x in charge):
        return YDelta.one(triangulation)

    # Try to express charge = n * primitive for some primitive.
    for prim in primitives:
        if len(prim) != n_edges:
            raise ValueError(
                f"primitive has length {len(prim)}, expected {n_edges}"
            )
        prim = tuple(prim)
        # Find a coordinate where prim is nonzero to extract n.
        n = None
        consistent = True
        for i in range(n_edges):
            if prim[i] == 0:
                if charge[i] != 0:
                    consistent = False
                    break
                continue
            ratio_num, ratio_den = charge[i], prim[i]
            if ratio_num % ratio_den != 0:
                consistent = False
                break
            n_i = ratio_num // ratio_den
            if n is None:
                n = n_i
            elif n != n_i:
                consistent = False
                break
        if not consistent or n is None:
            continue
        if n < 0:
            # T_n is even in n (Chebyshev parity); use |n|.
            n = -n
            # And the curve is the "reversed" curve, but Tr^lq is the
            # same for both orientations of an unoriented curve.
        return chebyshev_skein(triangulation, prim, n)

    raise BasisError(
        f"charge {charge} is not 0 and not n*primitive for any "
        f"of the {len(primitives)} primitives provided; "
        "basis_element_for_charge does not auto-decompose multicurves "
        "of multi-component disjoint unions"
    )


def multiply_in_F_basis(
    triangulation: Triangulation,
    a: Sequence[int],
    b: Sequence[int],
    primitives: Sequence[Sequence[int]],
    *,
    cone_witness: Sequence[int] | None = None,
    max_steps: int = 200,
) -> dict[tuple, "HalfLaurent"]:
    """Decompose F_a * F_b in the F-basis and return structure constants.

    Implements the same algorithm as `bps_quiver_tools.CoulombAlgebra.multiply`,
    natively over Y_Delta:

      product = F_a * F_b  in Y_Delta
      while product is non-empty:
        c = lowest-charge in product (in the positive-cone partial order)
        coeff = product.coefficient(c)
        result[c] = coeff
        product -= coeff * F_c
      return result

    `F_a` is the Y_Delta basis element labeled by tropical charge `a`,
    obtained via `basis_element_for_charge(triangulation, a, primitives)`.

    Args:
        triangulation: ambient triangulation.
        a, b:          tropical charges of the two basis elements
                       being multiplied (e.g. simple-closed-curve charges
                       or Chebyshev multiples thereof).
        primitives:    list of primitive simple-closed-curve charges,
                       used to identify F_c via Chebyshev decomposition.
        cone_witness:  a vector f such that <f, prim> > 0 for every
                       primitive (used for the lex-like cone ordering).
                       Defaults to the all-ones vector, which works
                       for S^2_{0,n} triangulations.
        max_steps:     safety cap; raises if the decomposition doesn't
                       terminate.

    Returns: dict {c: coefficient} of structure constants, where each
        coefficient is a HalfLaurent in lq.

    Raises:
        BasisError if the algorithm encounters a charge not expressible
        as n*primitive (= probably needs flavour decoration in the basis,
        not yet supported here).
        RuntimeError if it doesn't terminate within max_steps.
    """
    n = triangulation.n_edges
    if cone_witness is None:
        cone_witness = (1,) * n
    cone_witness = tuple(cone_witness)

    def cone_score(c: tuple) -> int:
        return sum(c[i] * cone_witness[i] for i in range(n))

    F_a = basis_element_for_charge(triangulation, a, primitives)
    F_b = basis_element_for_charge(triangulation, b, primitives)
    product = F_a * F_b

    result: dict[tuple, "HalfLaurent"] = {}

    for step in range(max_steps):
        nonzero = {c: v for c, v in product._terms.items() if not v.is_zero()}
        if not nonzero:
            return {c: v for c, v in result.items() if not v.is_zero()}
        # Pick the charge with smallest cone_score (lowest in positive cone).
        lowest = min(nonzero, key=cone_score)
        coeff = nonzero[lowest]

        # Add coeff to result[lowest].
        if lowest in result:
            new_v = result[lowest] + coeff
            if new_v.is_zero():
                del result[lowest]
            else:
                result[lowest] = new_v
        else:
            result[lowest] = coeff

        # Subtract coeff * F_lowest from product.
        try:
            F_low = basis_element_for_charge(triangulation, lowest, primitives)
        except BasisError as e:
            raise BasisError(
                f"step {step}: charge {lowest} doesn't decompose in the "
                f"current basis -- need richer basis elements.  Original: {e}"
            )
        product = product - coeff * F_low
    raise RuntimeError(
        f"multiply_in_F_basis did not terminate within {max_steps} steps"
    )


def lq_to_fq_substitute(coef: HalfLaurent) -> dict[int, int]:
    """Convert a HalfLaurent in `lq` to a `fq`-exponent dict, using the
    empirical relation lq = fq^{-1} (= fq^{-1}).

    The empirical relation `lq * fq = 1` is established by comparing
    pair-commutation powers in skein_sphere's Y_Delta (in lq) against
    the cluster code's BPS quiver Y_Delta (in fq) for the tetrahedral
    S^2_{0,4}: every pair (e_i, e_j) gives lq^{sigma_mine[i][j]} on my
    side and fq^{sigma_oct[pi(i)][pi(j)]} on the cluster side, with
    sigma_oct = -pi sigma_mine pi^T.  Hence lq^k = fq^{-k}.

    Returns a dict {fq_exponent: integer_coefficient}.  Raises
    ValueError if `coef` has a half-integer lq exponent (which would
    map to a half-integer fq exponent, not representable in the
    cluster's integer-power LaurentPoly).
    """
    out: dict[int, int] = {}
    for lq_exp, c in coef.items():
        if lq_exp.denominator != 1:
            raise ValueError(
                f"HalfLaurent has half-integer lq exponent {lq_exp}; "
                "lq -> fq^{-1} only converts integer lq powers cleanly. "
                "Use the y-form ordered_coef (= self.coefficient(charge)) "
                "instead of triangle_weyl_coefficient to get integer powers."
            )
        fq_exp = -int(lq_exp)
        out[fq_exp] = c
    return out


def chebyshev_skein(
    triangulation: Triangulation,
    charge: Sequence[int],
    n: int,
) -> YDelta:
    """T_n(gamma) in Y_Delta, for gamma the simple closed curve at `charge`.

    Chebyshev polynomial of the first kind, with the SL_2-trace
    normalization standard for Kauffman-bracket skein algebras:
        T_0 = 2 * 1      (twice the empty curve)
        T_1 = gamma
        T_n = gamma * T_{n-1} - T_{n-2}

    The Y_Delta element T_n(Tr^lq(gamma)) corresponds to the n-th
    "Chebyshev-renormalized" parallel copy basis element of the skein
    algebra: gamma traversed n times with the standard recursive
    smoothing that keeps it linearly independent from lower-power
    curves.

    For n = 0 returns 2 * Y_Delta.one().  For n >= 1 builds via the
    recursion using YDelta arithmetic.
    """
    if n < 0:
        raise ValueError(f"Chebyshev index must be non-negative, got n={n}")
    if n == 0:
        return YDelta.one(triangulation) + YDelta.one(triangulation)
    g = quantum_trace(triangulation, charge)
    if n == 1:
        return g
    # Iterative recursion: T_n = g * T_{n-1} - T_{n-2}
    t_prev2 = YDelta.one(triangulation) + YDelta.one(triangulation)  # T_0
    t_prev1 = g                                                       # T_1
    for _ in range(2, n + 1):
        t_curr = g * t_prev1 - t_prev2
        t_prev2 = t_prev1
        t_prev1 = t_curr
    return t_prev1


def decompose_against_basis(
    target: YDelta,
    basis: Mapping[str, YDelta],
    *,
    use_weyl: bool = True,
) -> tuple[dict[str, HalfLaurent], YDelta]:
    """Try to express `target` as sum_i c_i * basis[name_i] in Y_Delta.

    Heuristic strategy ("unique charge"):
      * For each basis element, look at the charges where it has a
        non-zero coefficient.  Among these, find a "characteristic"
        charge that no other basis element has (in its support).
      * Extract c_i = target.coef(char_i) / basis[name_i].coef(char_i).
      * Sum up the contributions and return the residual.

    On success (residual = 0), the target is fully expressed.  On
    failure (residual != 0), partial coefficients are returned and
    the residual flags what's missing -- typically additional
    multicurves or higher-Chebyshev terms not in the basis.

    Args:
        target: Y_Delta element to decompose.
        basis : dict {name: Y_Delta basis element}.
        use_weyl: if True, work with `triangle_weyl_coefficient` (the
            Z_3-equivariant X-form, where the example doc and cluster
            code's F_p's have all-1 coefficients).  If False, use the
            ordered y-form `coefficient`.

    Returns:
        (coefs_dict, residual)
            coefs_dict : {name: HalfLaurent} of extracted coefficients
                         (any name not in coefs_dict had no characteristic
                         charge, so its coefficient is undetermined and
                         set to zero).
            residual    : target - sum_i c_i * basis[name_i] in Y_Delta.
                          Zero on success.
    """
    triangulation = target.triangulation

    if use_weyl:
        get = lambda yd, c: yd.triangle_weyl_coefficient(c)
    else:
        get = lambda yd, c: yd.coefficient(c)

    # Build per-basis-element support: set of charges with nonzero coef.
    supports: dict[str, set] = {}
    for name, b in basis.items():
        if b.triangulation is not triangulation:
            raise ValueError(
                f"basis['{name}'] uses a different triangulation than target"
            )
        supports[name] = {
            c for c in b._terms.keys() if not get(b, c).is_zero()
        }

    # Find a characteristic charge for each basis element: a charge in
    # this basis element's support but not in any other's.
    coefs: dict[str, HalfLaurent] = {}
    for name in basis:
        chars = supports[name].copy()
        for other_name, other_supp in supports.items():
            if other_name == name:
                continue
            chars -= other_supp
        if not chars:
            # No characteristic charge: this basis element overlaps
            # entirely with others.  Skip extraction.
            continue
        # Pick any characteristic charge (sorted for determinism).
        char_charge = sorted(chars)[0]
        target_coef = get(target, char_charge)
        basis_coef = get(basis[name], char_charge)
        if basis_coef.is_zero():
            # Shouldn't happen by construction; skip.
            continue
        # c_i = target_coef / basis_coef.  HalfLaurent supports division
        # only by units (single +/-1 coefficient monomials).
        if len(basis_coef._c) == 1:
            coefs[name] = target_coef * (basis_coef ** -1)
        else:
            # General (non-monomial) basis_coef: unable to divide
            # cleanly without polynomial division.  Skip and let the
            # residual flag this case.
            continue

    # Compute residual
    residual = target
    for name, c in coefs.items():
        residual = residual - (c * basis[name])

    return coefs, residual
