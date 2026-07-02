"""
e6_rgkalgebra.py
================

`E6RGKAlgebra` — the exceptional Argyres–Douglas theory `A_𝖖([A_1, E_6])` as an
`RGKAlgebra`, realised by the RG flow

    [A_1, E_6]   ──drop the central node──▶   u(1)-gauged [A_1, A_5]

— the E-series analogue of the warm-up `A1AevenToU1AoddRGKAlgebra`
(`[A_1, A_{2k+2}] → u(1)-gauged [A_1, A_{2k+1}]`).  The auxiliary is the
gauged-odd standalone

    auxiliary  =  U1A1AoddKAlg(2)              # u(1)-gauged [A_1, A_5]

*directly* — no `add_flavour`, no quantum torus ([A_1, E_6] is flavourless;
rank 6 = `U1A1AoddKAlg(2)`'s rank 2k+2 = 6, so the dropped node integrates into
the gauge U(1) rather than adding a spectator flavour, exactly as in the warm-up).

End vs centre — the A/E fork
----------------------------
`E_6` is the `A_5` chain `1-2-3-4-5` with one extra node attached at the **centre**
(node 3), arms of length (2, 2, 1).  Compare the warm-up's `[A_1, A_6]` = `A_5`
chain + a node at the **end**.  Both are single-node drops with `S_RG = E_𝖖(L)`;
**which chord `L` is dropped selects A₆ vs E₆**:

* end attachment — `L` a **short** (length-2, type-1) mag-1 chord ⇒ `[A_1, A_6]`
  (the four short mag-1 chords `(1,1),(1,3),(1,5),(1,7)` are a single ρ²-orbit, all
  the warm-up);
* centre attachment — `L` the **central DIAMETER** (longest type, here type-3)
  mag-1 chord `(3,1)` (8-gon diagonal `(1,5)`) ⇒ `[A_1, E_6]`.

Verified by the UV BPS quiver's **Cartan determinant** (`uv_cartan_determinant`):
the A₅ chain (the mag-0 type-2 chords forming a 5-path) plus `L`'s attachment is
the E₆ Dynkin — `det = 3` (E₆) for the central diameter, vs `det = 7` (A₆) for an
end short chord, `det = 4` (D₆) for a second-from-end chord.  This structural
witness sidesteps the `U1A1AoddKAlg` trace, which is incomplete in the
deep-power regime: the generic windowed Schur-index path cannot reduce the
high-power diameter labels.

Spectrum generator
------------------
    S_RG  =  E_𝖖(X_L),   L = the central mag-1 diameter chord,

with `[S_RG]_{(m,)} = c_m · L^m`, `c_m = (−q)^m/(q²;q²)_m` — `L` q-commutes with
itself so `L^m = (((La, i0, m),), 0)` is a single cone monomial (verified).
Grading `Γ_RG = Z` by the magnetic charge; positive cone `Z_{≥0}`, height 1;
identity apex.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fractions import Fraction
from rgkalgebra import RGKAlgebra
from grading import Grading
from u1a1aodd_kalg import U1A1AoddKAlg
from a1aeven_to_u1aodd_rgkalgebra import _e_q_coeff       # c_m = (−q)^m/(q²;q²)_m


class E6RGKAlgebra(RGKAlgebra):
    """`[A_1, E_6]` as a directional `RGKAlgebra` over the gauged-odd
    standalone `U1A1AoddKAlg(2) = u(1)-gauged [A_1, A_5]`, with `S_RG = E_𝖖(L)`
    for `L` the **central diameter** mag-1 chord (the centre-of-A₅ attachment that
    distinguishes E₆ from the warm-up's A₆).  See the module docstring."""

    def __init__(self):
        self._aux = U1A1AoddKAlg(2)
        self._cd = self._aux.cone_data()
        self._n = self._cd._n                       # = 6 (B_GAUGED lattice rank)
        self._mag_index = self._n - 1               # the F coord = magnetic charge
        chg = self._cd._chg
        # central dressing chord: the longest type (diameter), magnetic charge +1.
        diam_type = max(self._cd._types)            # = 3 for k=2 (the 8-gon diameter)
        diams_mag1 = sorted(
            i for i in self._cd._types[diam_type] if chg[(diam_type, i)][self._mag_index] == 1)
        if not diams_mag1:
            raise RuntimeError("no magnetic-charge-1 diameter chord in U1A1AoddKAlg(2)")
        self._La = diam_type
        self._i0 = diams_mag1[0]

    # ----- magnetic grading helper ---------------------------------------

    def _mag(self, label) -> int:
        """Magnetic charge of a cone label `(factors, e_E)` — `Σ exp·mag(chord)`
        (the gauge generator `E` is mag-neutral, so `e_E` drops out)."""
        chg, mi = self._cd._chg, self._mag_index
        factors, _e_E = label
        return sum(exp * chg[(a, i)][mi] for (a, i, exp) in factors)

    # ----- RGKAlgebra contract -------------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z = the magnetic charge; positive cone Z_{>=0}, height 1.
        return Grading(rank=1, deg=lambda lbl: (self._mag(lbl),), height=(1,),
                       cone_gens=((1,),))

    def apex(self, a):
        """Identity apex: UV labels coincide with auxiliary cone labels."""
        a = tuple(a)
        return (a[0], a[1])

    def _diameter_power(self, m: int):
        """`L^m` as a `U1A1AoddKAlg` cone label: the mag-1 central diameter chord
        raised to `m` (a single q-commuting generator power)."""
        return (((self._La, self._i0, m),), 0)

    def _s_rg_component(self, p):
        """`[S_RG]_{(m,)}` — exact, finite, vanishing off the cone.  `S_RG = E_𝖖(X_L)`
        ⇒ degree-`m` part is the single label `L^m` with coefficient `c_m`;
        degree 0 the identity, negative degree empty."""
        (m,) = p
        if m < 0:
            return {}
        if m == 0:
            return {self._aux.identity(): _e_q_coeff(0)}
        return {self._diameter_power(m): _e_q_coeff(m)}

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG` windowed to q-order ≤ `cutoff`: the magnetic tower `{L^m : c_m}`
        for `m < cutoff` (the exact `RG` path uses `_s_rg_component`)."""
        out: dict = {}
        for m in range(cutoff):
            out.update(self._s_rg_component((m,)))
        return out

    # ----- flavour-aware section split -----------------------------------

    def _section_split(self, label):
        """The auxiliary cone label `(factors, e_E)` is not a flat charge vector,
        so disable the flavour-shift multiply cache (`flav = None`): `multiply`
        falls back to the direct `from_ir_image(RG(a)·RG(b))` per pair."""
        return tuple(label), None

    # ----- structural validation: the UV BPS quiver is E6 ----------------

    def _cocycle(self, g, h) -> int:
        """Dirac pairing `⟨g,h⟩` of two auxiliary chords (`L_g L_h = q^{2⟨⟩} L_h L_g`)."""
        return self._cd.cocycle(g, h)

    def _a5_chain(self):
        """The auxiliary A₅ chain: the five mag-0 (type-2) chords forming a 5-path
        under the Dirac pairing, returned in path order."""
        import itertools
        chg, mi = self._cd._chg, self._mag_index
        nodes = [(2, i) for i in self._cd._types[2] if chg[(2, i)][mi] == 0]

        def edge(g, h):
            return abs(self._cocycle(g, h)) == 1

        def is_path(p):
            n = len(p)
            deg = [sum(1 for y in p if y != x and edge(x, y)) for x in p]
            edges = sum(1 for i in range(n) for j in range(i + 1, n) if edge(p[i], p[j]))
            return edges == n - 1 and sorted(deg) == [1, 1, 2, 2, 2]

        for combo in itertools.combinations(nodes, 5):
            if is_path(combo):
                deg1 = [x for x in combo if sum(1 for y in combo if y != x and edge(x, y)) == 1]
                order = [deg1[0]]
                while len(order) < 5:
                    order.append(next(y for y in combo
                                      if y not in order and edge(order[-1], y)))
                return order
        raise RuntimeError("no A5 chain found among the mag-0 type-2 chords")

    def _cartan_det(self, nodes) -> int:
        """Determinant of the Cartan matrix `2I − adjacency` of the Dirac-pairing
        quiver on `nodes` (E₆ = 3, A₆ = 7, D₆ = 4)."""
        n = len(nodes)
        adj = [[1 if (i != j and self._cocycle(nodes[i], nodes[j]) != 0) else 0
                for j in range(n)] for i in range(n)]
        M = [[Fraction(2 if i == j else -adj[i][j]) for j in range(n)] for i in range(n)]
        d = Fraction(1)
        for c in range(n):
            piv = next((i for i in range(c, n) if M[i][c] != 0), None)
            if piv is None:
                return 0
            if piv != c:
                M[c], M[piv] = M[piv], M[c]
                d = -d
            d *= M[c][c]
            for i in range(c + 1, n):
                f = M[i][c] / M[c][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[c])]
        return int(d)

    def uv_cartan_determinant(self) -> int:
        """The UV BPS quiver Cartan determinant: the A₅ chain plus the dropped
        central diameter `L`.  `== 3` (E₆) — the structural witness that this flow
        realises `[A_1, E_6]` (vs 7 = A₆ for an end short chord)."""
        L = (self._La, self._i0)
        return self._cartan_det(self._a5_chain() + [L])

    def verify_is_E6(self) -> bool:
        """`uv_cartan_determinant() == 3` — the UV BPS quiver is the E₆ Dynkin."""
        return self.uv_cartan_determinant() == 3


if __name__ == "__main__":
    T = E6RGKAlgebra()
    print("E6RGKAlgebra = [A_1, E_6]  via  drop central node of E6 → u(1)-gauged [A_1, A_5]")
    print("  aux =", type(T.auxiliary()).__name__,
          "  central diameter L = (", T._La, ",", T._i0, ")  mag =",
          T._mag(T._diameter_power(1)))
    print("  A5 chain (mag-0 type-2 path):", T._a5_chain())
    print("  UV BPS quiver Cartan determinant =", T.uv_cartan_determinant(),
          "(E6 = 3)   verify_is_E6:", T.verify_is_E6())
    print("  verify_rg_unital:", T.verify_rg_unital())
    g = (((2, 0, 1),), 0)
    print("  RG(L(2,0)) =", {l: str(c) for l, c in T.RG(g).terms.items()})
