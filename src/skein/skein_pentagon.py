"""SkeinPentagonAlgebra -- the Pentagon (A_2 Argyres-Douglas) algebra
realized as a stated-SKEIN algebra.

The realization is Le's stated skein algebra of the bordered pentagon
(disk with 5 marked boundary intervals = the [A_1, A_2] irregular
puncture), computed in the split representation over the fan
triangulation (`stated_polygon.StatedPolygon(5)`).  The measured
structure:

* THE PINNED PENTAGON: the algebra is the A_2/pentagon algebra over
  the enlarged lattice Gamma~ = Gamma_A2 (+) Z^5_boundary -- same BPS
  quiver, endpoint charges adjoined.  Each stated chord D_p(e1,e2)
  (p, p+2 with states) is a canonical over the fiber of
  Gamma~ -> Gamma_A2: all its quantum-trace terms share ONE boundary
  vector (the states at the endpoint slots), internal variation = the
  A_2 F-directions.

* THE UNIVERSAL STATE-TRANSPORT EXCHANGE LAW (all 16 states x Z_5):

    D_p(e1,e2) D_{p+1}(e3,e4) = q^{-1} s_p(e1,e3) s_{p+2}(e2,e4)
                              + q       D_{p+3}(e4,e1) s_{p+1}(e3,e2)

* rho = the one-unit rotation of the irregular puncture (order 5,
  the 1/(h+2) fractional quantum monodromy), acting on generators as
  arc(u,v,e,e') -> arc(u+1,v+1,e,e').

* The chord quadruple {D_p(e,e')} satisfies Le's bigon presentation
  re1/re2 (an O_{q^2}(SL_2)-type stated matrix per chord).

* Odd marks (5): NO flavour -- the parity/2-coloring obstruction; the
  only q-central torus direction is the Chekhov-Fock all-edges
  monomial.

Certification against the BPS chart (pairing [[0,1],[-1,0]]) --
including the kappa dictionary D_p <-> L_{g_{p+1}}
(rotation-equivariant, X = y^2, uniform coefficients).

Relation to the Pentagon `KAlgebraObject` (documented, not an iso)
------------------------------------------------------------------
The repo's pentagon `KAlgebraObject` (cone-frozen / bps / closed-form /
a1a2k) collects presentations of the rank-2 pentagon algebra.  This
skein realization is NOT iso to them: it presents the PINNED
pentagon -- the same A_2 BPS quiver over the enlarged lattice
Gamma~ = Gamma_A2 (+) Z^5_boundary, i.e. the pentagon with extra
quantum-torus (boundary endpoint) factors.  Measured obstruction
(robust): the bare-chord exchange flips BOTH coefficients under
order reversal (D_p D_{p+1} = q^{-1} s s + q D s vs the reversed
q s s + q^{-1} D s) while the canonical presentations flip only one
(L_p L_{p+1} = 1 + q^{-1} L_{p+3} vs 1 + q L_{p+3}); any label map
with monomial side-values would force q^2 = 1.  The certified
bridge is the kappa dictionary: kappa(D_p(+,+)) carries
the F-Newton of the chart canonical L_{g_{p+1}} up to one frame
monomial.  RESOLVED by `unpin()`: in the LOCALIZATION at the side
arcs (an Ore set), the dressed
chords T_p = q^{x_p} D_p(+,+) (s_{p+2}s_{p+4}s_{p+3}^{-1})^{-1}
satisfy exactly the pentagon relations (T T = 1 + q^2 T after
normalization; q_chart = q_skein^2) and commute with the whole side
torus:  Pinned[s^{-1}] = Pentagon (x) T(sides) -- a full tensor
factorization.  The earlier no-go is evaded precisely because the
side MONOMIALS do not commute with the chords (scalars could not
supply the order-dependent q-powers).  The localized dressed-chord
subalgebra is packaged as a `KAlgebra` contract instance in
`skein_pentagon_kalg.SkeinPentagonKAlg` (chart labels, genuinely
skein-side multiply) and registered as the ``'skein'`` realization
of the pentagon `KAlgebraObject`, with the
`KAlgebraIso` to the BPS realization as the witness.

This class itself is the certified PINNED algebra surface (the
stated generators and their exchange structure); the intrinsic
stated-side trace (half-index machinery) remains a separate work
item — `SkeinPentagonKAlg` transports trace through the BPS
realization for now.
"""

from __future__ import annotations

from half_laurent import HalfLaurent
from stated_polygon import PolyElement, StatedPolygon

Q = HalfLaurent.monomial


class SkeinPentagonAlgebra:
    """The stated-skein realization of the pentagon algebra."""

    N = 5

    def __init__(self):
        self.P = StatedPolygon(self.N)

    # -- generators ------------------------------------------------------

    def one(self) -> PolyElement:
        return self.P.one()

    def chord(self, p: int, e1: int = 1, e2: int = 1) -> PolyElement:
        """The stated chord D_p(e1, e2) joining sides p and p+2."""
        return self.P.arc(p % self.N, (p + 2) % self.N, e1, e2)

    def side(self, p: int, e1: int = 1, e2: int = 1) -> PolyElement:
        """The stated side arc s_p(e1, e2) joining sides p and p+1."""
        return self.P.arc(p % self.N, (p + 1) % self.N, e1, e2)

    def arc(self, u: int, v: int, e1: int = 1, e2: int = 1) -> PolyElement:
        return self.P.arc(u % self.N, v % self.N, e1, e2)

    # -- structure -------------------------------------------------------

    def rho_generator(self, kind: str, p: int, e1: int, e2: int):
        """rho (one-unit rotation) on a generator: the rotated
        generator.  rho has order 5; it is the fractional quantum
        monodromy of the irregular puncture."""
        if kind == "chord":
            return self.chord(p + 1, e1, e2)
        if kind == "side":
            return self.side(p + 1, e1, e2)
        raise ValueError(kind)

    def exchange_lhs_rhs(self, p: int, e1: int, e2: int, e3: int, e4: int):
        """Both sides of the universal state-transport exchange law at
        rotation p with the given states."""
        lhs = self.chord(p, e1, e2) * self.chord(p + 1, e3, e4)
        rhs = self.side(p, e1, e3) * self.side(p + 2, e2, e4) * Q(-1) \
            + self.chord(p + 3, e4, e1) * self.side(p + 1, e3, e2) * Q(1)
        return lhs, rhs

    # -- unpinning ---------------------------------------------------------

    def unpin(self):
        """The UNPIN data: the pentagon extracted from the pinned
        algebra, and the full factorization.

        In the localization at the (all-+) side arcs (an Ore set:
        sides q-commute pairwise and monomially with every chord),
        the dressed chords

            T_p = q^{x_p} D_p(+,+) W_p^{-1},
            W_p = s_{p+2} s_{p+4} s_{p+3}^{-1}

        (the unique side-word solution of the lattice system) satisfy

            T_p T_{p+1} = q^{a} + q^{b} T_{p+3},   (a, b) = (-4, 0)

        i.e. after T -> q^{-a/2} T exactly the pentagon
        T T = 1 + q^2 T at q_chart = q_skein^2, AND every T_p commutes
        with every side: Pinned[s^{-1}] = <T_p> (x) T(side torus).

        Returns the dict of construction data: the side-word exponent
        vectors `w[p]` (in Z^5, entries = exponents of s_0..s_4), the
        gauge `x[p]`, the relation constants `a`, `b`, and the
        commutation matrices `Mss(r,t)`, `Msd(r,p)` (engine-verified).
        """
        def f5(z):
            return z % 5

        def Mss(r, t):
            if t == f5(r + 1):
                return -1
            if t == f5(r - 1):
                return 1
            return 0

        def Msd(r, p):
            return [1, -1, 1, 0, -1][f5(r - p)]

        def evec(*idx):
            v = [0] * 5
            for i in idx:
                v[f5(i)] += 1
            return tuple(v)

        w = {p: tuple(
            evec(p + 2)[t] + evec(p + 4)[t] - evec(p + 3)[t]
            for t in range(5)) for p in range(5)}
        from fractions import Fraction
        x = {0: Fraction(-3), 1: Fraction(-2), 2: Fraction(-2),
             3: Fraction(-3), 4: Fraction(-3)}
        return {"w": w, "x": x, "a": Fraction(-4), "b": Fraction(0),
                "Mss": Mss, "Msd": Msd}
