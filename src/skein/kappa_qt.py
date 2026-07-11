"""kappa_Delta — the quantum trace on triangulated polygons, in the
split representation.

kappa = (tensor phi) o rho: on a `StatedPolygon` element
(already split: a sum of tensors of per-triangle labels), apply the
triangle phi (`phi_map.phi`) factorwise into
(x)_T Y(T) = the tensor of triangle Chekhov-Fock algebras.

CONVENTION (repo-wide):
in this representation phi is an ANTI-homomorphism — products must be
reversed when verifying algebra relations in Y.

Torus elements are dict[tuple of per-triangle exponent triples,
HalfLaurent]; `ymul` multiplies componentwise via TriangleAlgebra.
`internal_exps` reads the exponent on each internal edge of the fan,
asserting the two glued copies agree (Chekhov-Fock well-formedness of
the image — measured to HOLD on kappa images, in contrast to the
stated-basis cut-counts which are not invariant).

Used for the pentagon/A2 and hexagon/A3 dictionaries.
"""

from __future__ import annotations

from itertools import product as iproduct

from half_laurent import HalfLaurent, ONE as HL_ONE
from phi_map import phi as phi_arc
from stated_polygon import StatedPolygon
from triangle_algebra import TriangleAlgebra


def phi_label(lab):
    """phi of a single triangle basis label (identity or one corner
    arc) as a TriangleAlgebra element."""
    k, sa, sb, sc = lab
    if k == (0, 0, 0):
        return TriangleAlgebra.one()
    if sum(k) != 1:
        raise NotImplementedError(f"multi-arc label {lab}")
    if k == (1, 0, 0):     # alpha joins b, c; phi convention eps@c, eps'@b
        return phi_arc("alpha", sc[0], sb[0])
    if k == (0, 1, 0):     # beta joins c, a; eps@a, eps'@c
        return phi_arc("beta", sa[0], sc[0])
    return phi_arc("gamma", sb[0], sa[0])   # gamma joins a, b; eps@b, eps'@a


def kappa(P: StatedPolygon, el) -> dict:
    """kappa of a PolyElement whose tensor labels carry <= 1 arc per
    triangle (e.g. routed boundary arcs)."""
    out: dict = {}
    for lab, c in el.terms.items():
        factors = [phi_label(l) for l in lab]
        if any(f.is_zero() for f in factors):
            continue
        for combo in iproduct(*(f.items() for f in factors)):
            key = tuple(t for t, _ in combo)
            coeff = c
            for _, cc in combo:
                coeff = coeff * cc
            s = out.get(key, HalfLaurent.zero()) + coeff
            if s.is_zero():
                out.pop(key, None)
            else:
                out[key] = s
    return out


def ymul(P: StatedPolygon, x: dict, y: dict) -> dict:
    """Product in (x)_T Y(T) (componentwise TriangleAlgebra)."""
    out: dict = {}
    for kx, cx in x.items():
        for ky, cy in y.items():
            comps = [TriangleAlgebra({kx[i]: HL_ONE}) *
                     TriangleAlgebra({ky[i]: HL_ONE}) for i in range(P.m)]
            for combo in iproduct(*(c.items() for c in comps)):
                key = tuple(t for t, _ in combo)
                coeff = cx * cy
                for _, cc in combo:
                    coeff = coeff * cc
                s = out.get(key, HalfLaurent.zero()) + coeff
                if s.is_zero():
                    out.pop(key, None)
                else:
                    out[key] = s
    return out


def yadd(x: dict, y: dict) -> dict:
    out = dict(x)
    for k, c in y.items():
        s = out.get(k, HalfLaurent.zero()) + c
        if s.is_zero():
            out.pop(k, None)
        else:
            out[k] = s
    return out


def yscale(x: dict, c: HalfLaurent) -> dict:
    return {k: v * c for k, v in x.items()}


def fan_internal_slots(P: StatedPolygon):
    """[(left slot, right slot)] per internal edge of the fan:
    edge (0, i+2) = T_i.c ~ T_{i+1}.a."""
    return [((i, 2), (i + 1, 0)) for i in range(P.m - 1)]


def internal_exps(P: StatedPolygon, key) -> tuple:
    """Exponent on each internal edge; asserts the two glued copies
    agree."""
    out = []
    for (t1, r1), (t2, r2) in fan_internal_slots(P):
        e1 = key[t1][r1]
        e2 = key[t2][r2]
        assert e1 == e2, f"internal copies disagree: {key}"
        out.append(e1)
    return tuple(out)
