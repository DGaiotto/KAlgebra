"""Pinned polygon algebras over the enlarged lattice Gamma-tilde.

The "pinned" picture: cutting a surface into polygons leaves
each sub-theory embedded in a BIGGER lattice

    Gamma-tilde  =  Gamma_int (+) Z^{boundary}

— the full bordered exchange matrix sigma-tilde including the frozen
boundary directions, exactly `BorderedTriangulation.sigma()`.  The
pinned polygon algebra realises this concretely via the quantum trace:
kappa = (tensor phi) o rho sends the stated-skein polygon into the
tensor of triangle Chekhov-Fock algebras (x)_T Y(T); the image is
Chekhov-Fock WELL-FORMED — both copies of every internal edge carry
EQUAL exponents (asserted
here on every conversion) — so

    key  (per-triangle exponent triples)
    <->  charge  (one coordinate per EDGE of the bordered
                  triangulation, internal ids first)

is a BIJECTION, and it transports the (x)_T Y(T) product to a quantum
torus on Gamma-tilde.  Measured (test_pinned_polygon):

    U_u U_v = lq^{c(u,v)} U_{u+v},
    c(u,v) - c(v,u) = - sum_{e,f} sigma_tilde[e][f] u_e v_f,

i.e. the pinned polygon IS the quantum torus of the frozen-extended
exchange matrix — internal block = the A_{n-3} BPS quiver (dynamical),
boundary block = the pins.

THE PINNING.  phi sends each stated corner arc to a single monomial
whose exponent at an edge slot equals the STATE at that endpoint
(phi_map eq.qt), and routing only touches the two end sides; hence for
a stated boundary arc every kappa-term carries the SAME boundary
block — the endpoint-state vector (`pinning_vector`).  Boundary
coordinates are not dynamical: they pin the arc.  Gluing two polygons
along cut sides merges the two pinned coordinates into one new
internal (dynamical) direction — the gluing y_e |-> y_{e'} (x) y_{e''}
read backwards: a merged
monomial is well-formed iff the two cut coordinates AGREE, and the
common value becomes the glued-edge exponent (`glue_charge` /
`glue_state_sum`; the square + triangle -> pentagon demo is certified
in the tests).

CONVENTION.  phi is an ANTI-homomorphism in this repo's representation
(products reverse in Y — `phi_map.phi` docstring); `multiply` here
reverses the ymul order once and for all, so the pinned algebra is an
honest ALGEBRA image of the stated skein algebra:

    pinned(x . y) = multiply(pinned(x), pinned(y)).

Elements are plain dict[charge tuple, HalfLaurent], matching the
`kappa_qt` y-dict style (`add` / `scale` reuse yadd / yscale).

Scope: fan polygons through `StatedPolygon`; the algebra-level
glue where the merged direction becomes dynamical (an inverse node-drop
RG) is handled separately.
"""

from __future__ import annotations

from half_laurent import HalfLaurent, ONE as HL_ONE
from kappa_qt import kappa, yadd, ymul, yscale
from stated_polygon import PolyElement, StatedPolygon


class PinnedPolygon:
    """The pinned (Gamma-tilde) presentation of a tree-dual polygon.

    Construct with an int `n` (the fan n-gon) or any tree-dual
    `BorderedTriangulation` (see `StatedPolygon`)."""

    def __init__(self, n):
        self.P = StatedPolygon(n)
        self.n = self.P.n
        self.bt = self.P.t
        self.m = self.P.m
        (self.slot_to_edge, self.internal_ids,
         self.boundary_ids) = self.bt.edge_numbering()
        self.n_edges = self.bt.n_edges
        self.sigma_full = self.bt.sigma()

    # -- Gamma-tilde coordinates ------------------------------------------

    def charge_of_key(self, key) -> tuple:
        """Gamma-tilde charge of a well-formed y-key; raises ValueError
        if the two copies of an internal edge disagree (Chekhov-Fock
        ill-formed)."""
        chg = [None] * self.n_edges
        for tri in range(self.m):
            for role in range(3):
                e = self.slot_to_edge[(tri, role)]
                v = key[tri][role]
                if chg[e] is None:
                    chg[e] = v
                elif chg[e] != v:
                    raise ValueError(
                        f"internal copies disagree on edge {e}: {key}")
        return tuple(chg)

    def key_of_charge(self, charge) -> tuple:
        return tuple(
            tuple(charge[self.slot_to_edge[(tri, role)]] for role in range(3))
            for tri in range(self.m))

    def to_charges(self, ydict: dict) -> dict:
        return {self.charge_of_key(k): c for k, c in ydict.items()}

    def to_keys(self, chdict: dict) -> dict:
        return {self.key_of_charge(u): c for u, c in chdict.items()}

    # -- elements ----------------------------------------------------------

    def one(self) -> dict:
        return {(0,) * self.n_edges: HL_ONE}

    def monomial(self, charge, coeff: HalfLaurent = HL_ONE) -> dict:
        return {tuple(charge): coeff}

    def pinned(self, el: PolyElement) -> dict:
        """kappa of a stated-polygon element, re-keyed by charge."""
        return self.to_charges(kappa(self.P, el))

    def arc_F(self, s: int, t: int, eps_s: int, eps_t: int) -> dict:
        """The pinned F of the stated boundary arc (s, t; eps_s, eps_t)."""
        return self.pinned(self.P.arc(s, t, eps_s, eps_t))

    # -- algebra -----------------------------------------------------------

    def multiply(self, x: dict, y: dict) -> dict:
        """Product in the pinned algebra.  kappa is an anti-hom, so the
        ymul order is reversed here: pinned(a.b) = multiply(pinned(a),
        pinned(b))."""
        return self.to_charges(
            ymul(self.P, self.to_keys(y), self.to_keys(x)))

    add = staticmethod(yadd)
    scale = staticmethod(yscale)

    def inverse_monomial(self, x: dict) -> dict:
        """Inverse of a single-term element with unit coefficient —
        quantum-torus monomials are units, so localization is free on
        the pinned tier: multiply(x, inverse_monomial(x)) == one()."""
        ((u, c),) = x.items()
        neg = tuple(-a for a in u)
        z = self.multiply({u: c}, {neg: HL_ONE})
        ((_zk, zc),) = z.items()
        return {neg: zc ** -1}

    def commutation_exponent(self, u, v):
        """The lq-exponent c with U_u U_v = lq^c U_v U_u (a Fraction)."""
        xy = self.multiply(self.monomial(u), self.monomial(v))
        yx = self.multiply(self.monomial(v), self.monomial(u))
        (cu,) = xy.values()
        (cv,) = yx.values()
        ((eu, au),) = cu.items()
        ((ev, av),) = cv.items()
        if au != av:
            raise ValueError("monomial commutator is not an lq-power")
        return eu - ev

    # -- the pinning -------------------------------------------------------

    def boundary_block(self, charge) -> tuple:
        return tuple(charge[e] for e in self.boundary_ids)

    def internal_block(self, charge) -> tuple:
        return tuple(charge[e] for e in self.internal_ids)

    def pinning_vector(self, x: dict) -> tuple:
        """The boundary block shared by EVERY term of x — the pins.
        Raises ValueError if the terms do not agree (x not pinned)."""
        blocks = {self.boundary_block(u) for u in x}
        if len(blocks) != 1:
            raise ValueError(f"element is not pinned: {sorted(blocks)}")
        return next(iter(blocks))

    def state_vector(self, s: int, t: int, eps_s: int, eps_t: int) -> tuple:
        """The expected pinning vector of arc(s, t, eps_s, eps_t):
        eps_s at side s, eps_t at side t, 0 elsewhere."""
        out = [0] * len(self.boundary_ids)
        out[s] += eps_s
        out[t] += eps_t
        return tuple(out)


# -- gluing on charges ------------------------------------------------------

def glue_charge(target: PinnedPolygon, left, right,
                left_edge_map: dict, right_edge_map: dict) -> tuple:
    """Merge a (left, right) charge pair into a target charge.

    The maps send piece edge ids to target edge ids; the two CUT sides
    map to the SAME target internal id, and their coordinates must
    AGREE (y_e |-> y_{e'} (x) y_{e''} read backwards) — the common
    value becomes the glued-edge exponent.  Raises ValueError on
    disagreement or uncovered target edges."""
    out = [None] * target.n_edges
    for cmap, chg in ((left_edge_map, left), (right_edge_map, right)):
        for e_piece, e_target in cmap.items():
            v = chg[e_piece]
            if out[e_target] is None:
                out[e_target] = v
            elif out[e_target] != v:
                raise ValueError(
                    f"cut coordinates disagree at target edge {e_target}")
    if any(v is None for v in out):
        raise ValueError("target edge not covered by the gluing maps")
    return tuple(out)


def glue_state_sum(target: PinnedPolygon,
                   left_by_state: dict, right_by_state: dict,
                   left_edge_map: dict, right_edge_map: dict) -> dict:
    """The state-sum merge sum_{eps = +-1} left(eps) (x) right(eps) as a
    target charge dict: for each eps, every (left term, right term)
    pair merges through `glue_charge` with coefficient product.  This
    is how a pinned arc crossing the cut reassembles from its two
    pinned halves (certified square + triangle -> pentagon in the
    tests)."""
    out: dict = {}
    for eps in (1, -1):
        for ca, xa in left_by_state[eps].items():
            for cb, xb in right_by_state[eps].items():
                key = glue_charge(target, ca, cb,
                                  left_edge_map, right_edge_map)
                s = out.get(key, HalfLaurent.zero()) + xa * xb
                if s.is_zero():
                    out.pop(key, None)
                else:
                    out[key] = s
    return out
