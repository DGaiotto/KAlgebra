"""
e8_rgkalgebra.py
================

`E8RGKAlgebra` — the exceptional Argyres–Douglas theory `A_𝖖([A_1, E_8])` as an
`RGKAlgebra`, realised by the RG flow

    [A_1, E_8]   ──drop the central node──▶   u(1)-gauged [A_1, A_7]

— the rank-8 E-series entry, completing ADE.  The clean (flavourless,
even-rank) case, exactly like `E6RGKAlgebra`: the auxiliary is the gauged-odd
standalone

    auxiliary  =  U1A1AoddKAlg(3)              # u(1)-gauged [A_1, A_7] (decagon)

*directly* — no `add_flavour`, no quantum torus (rank 8 = `U1A1AoddKAlg(3)`'s
rank 2k+2 = 8, so the dropped node integrates into the gauge U(1); `[A_1, E_8]`
is flavourless).  Contrast E₇ (odd rank 7), which needs `[A_1,A_6]` + U(1) flavour.

End vs centre — the A/E fork
----------------------------
`E_8` is the `A_7` chain `1-…-7` with one extra node attached **off-centre** at
position 2 (arms 2-4-1; `E_8 = T_{2,3,5}`).  Compare the warm-up's `[A_1, A_8]`
= `A_7` chain + a node at the **end**.  Both are single-node drops with
`S_RG = E_𝖖(L)`; **which chord `L` is dropped selects A₈ vs E₈**:

* end attachment — a **short** (type-1) mag-1 chord ⇒ `[A_1, A_8]`;
* off-centre attachment — the mag-1 chord `L = (3, 0)` (a type-3 chord crossing
  the A₇ chain at its position-2 node) ⇒ `[A_1, E_8]`.

(Unlike E₆ — where the central node sits at the *symmetric* centre of A₅ and `L`
is the diameter — E₈'s arms 2,4 are asymmetric, so the E₈ node is off-centre and
`L` is a type-3 chord, not the longest type-4 diameter.  The true-centre
attachment of A₇ would be the affine `T_{2,4,4}`, Cartan det 0.)

Spectrum generator
------------------
    S_RG  =  E_𝖖(X_L),   L = the central mag-1 chord (3, 0),

`[S_RG]_{(m,)} = c_m · L^m`, `c_m = (−q)^m/(q²;q²)_m` — `L` q-commutes with itself
so `L^m = (((3, 0, m),), 0)` is a single cone monomial (verified).  Grading
`Γ_RG = Z` by the magnetic charge; positive cone `Z_{≥0}`, height 1; identity apex.

Validation
----------
The UV BPS quiver is the **E₈ Dynkin**: the A₇ chain (the mag-0 type-2 chords, a
7-path) plus the central node `L` has Cartan determinant `uv_cartan_determinant()
== 1` (E₈), vs `9` (A₈) for an end chord, `4` (D₈) for a second-from-end chord.
Among trees on 8 nodes only E₈ has Cartan det 1, certifying E₈ uniquely; this
matches the standard E₈ Dynkin that `BPSKAlgebra` is built from.  (The structural
witness sidesteps the deep-power-incomplete `U1A1AoddKAlg` trace, as for E₆.)
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


class E8RGKAlgebra(RGKAlgebra):
    """`[A_1, E_8]` as a directional `RGKAlgebra` over the gauged-odd
    standalone `U1A1AoddKAlg(3) = u(1)-gauged [A_1, A_7]`, with `S_RG = E_𝖖(L)`
    for `L` the central mag-1 chord `(3, 0)` (the off-centre-of-A₇ attachment that
    distinguishes E₈ from the warm-up's A₈).  See the module docstring."""

    def __init__(self):
        self._aux = U1A1AoddKAlg(3)
        self._cd = self._aux.cone_data()
        self._n = self._cd._n                       # = 8 (B_GAUGED lattice rank)
        self._mag_index = self._n - 1               # the F coord = magnetic charge
        # central dressing chord (3, 0): mag +1, crosses the A₇ chain at its
        # position-2 node → Cartan det 1 (E₈); found by the attachment analysis
        # (verified by verify_is_E8).
        self._La = 3
        self._i0 = 0
        if self._cd._chg[(self._La, self._i0)][self._mag_index] != 1:
            raise RuntimeError("E8 dressing chord (3,0) is not magnetic charge +1")

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

    def _chord_power(self, m: int):
        """`L^m` as a `U1A1AoddKAlg` cone label: the central mag-1 chord `(3, 0)`
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
        return {self._chord_power(m): _e_q_coeff(m)}

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

    # ----- structural validation: the UV BPS quiver is E8 ----------------

    def _cocycle(self, g, h) -> int:
        """Dirac pairing `⟨g,h⟩` of two auxiliary chords (`L_g L_h = q^{2⟨⟩} L_h L_g`)."""
        return self._cd.cocycle(g, h)

    def _a7_chain(self):
        """The auxiliary A₇ chain: the seven mag-0 (type-2) chords forming a 7-path
        under the Dirac pairing, returned in path order — the `[A_1, A_7]` simple
        roots."""
        chg, mi = self._cd._chg, self._mag_index
        nodes = [(2, i) for i in self._cd._types[2] if chg[(2, i)][mi] == 0]
        edge = lambda g, h: abs(self._cocycle(g, h)) == 1
        adj = {g: [h for h in nodes if h != g and edge(g, h)] for g in nodes}

        def dfs(path, used):
            if len(path) == 7:
                return list(path)
            for nb in adj[path[-1]]:
                if nb in used or any(edge(nb, p) for p in path[:-1]):
                    continue
                r = dfs(path + [nb], used | {nb})
                if r:
                    return r
            return None
        for s in nodes:
            r = dfs([s], {s})
            if r:
                return r
        raise RuntimeError("no A7 chain (induced 7-path) found")

    def _cartan_det(self, nodes) -> int:
        """Determinant of the Cartan matrix `2I − adjacency` of the Dirac-pairing
        quiver on `nodes` (E₈ = 1, A₈ = 9, D₈ = 4)."""
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
        """UV BPS quiver Cartan determinant: the A₇ chain plus the central chord
        `L`.  `== 1` (E₈) — the structural witness that this flow realises
        `[A_1, E_8]` (vs 9 = A₈ for an end chord)."""
        L = (self._La, self._i0)
        return self._cartan_det(self._a7_chain() + [L])

    def verify_is_E8(self) -> bool:
        """`uv_cartan_determinant() == 1` — the UV BPS quiver is the E₈ Dynkin."""
        return self.uv_cartan_determinant() == 1


if __name__ == "__main__":
    T = E8RGKAlgebra()
    print("E8RGKAlgebra = [A_1, E_8]  via  drop central node → u(1)-gauged [A_1, A_7]")
    print("  aux =", type(T.auxiliary()).__name__,
          "  central chord L = (", T._La, ",", T._i0, ")  mag =",
          T._mag(T._chord_power(1)))
    print("  A7 chain (mag-0 type-2 path):", T._a7_chain())
    print("  UV BPS quiver Cartan determinant =", T.uv_cartan_determinant(),
          "(E8 = 1)   verify_is_E8:", T.verify_is_E8())
    print("  verify_rg_unital:", T.verify_rg_unital())
    g = (((2, 0, 1),), 0)
    print("  RG(L(2,0)) =", {l: str(c) for l, c in T.RG(g).terms.items()})
