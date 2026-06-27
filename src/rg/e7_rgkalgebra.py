"""
e7_rgkalgebra.py
================

`E7RGKAlgebra` — the exceptional Argyres–Douglas theory `A_𝖖([A_1, E_7])` as a
new-contract `RGKAlgebra` (Plan 20), realised by the RG flow

    [A_1, E_7]   ──drop the central node──▶   [A_1, A_6] + U(1) flavour

— the E-series sibling of `E6RGKAlgebra`.  Where E₆ (rank 6) drops onto the
*gauged*-odd `U1A1AoddKAlg(2)`, E₇ has odd rank 7, so (user's prescription) the
auxiliary is the even algebra `[A_1, A_6]` with a **spectator U(1) flavour**
adjoined — the odd rank-7 lattice the gauged-odd standalones (even rank `2k+2`)
cannot supply:

    auxiliary  =  A1A2kKAlg(3).add_flavour(1)            # [A_1, A_6] ⊕ U(1) μ

exactly the template of `A1AoddToEvenRGKAlgebra` (which realises `[A_1, A_{2k+1}]`
the same way with an **end** short chord); E₇ differs only in dropping a
**central** chord.

End vs centre — the A/E fork
----------------------------
`E_7` is the `A_6` chain `1-2-3-4-5-6` with one extra node attached at the
**centre** (position 3, arms 2-3-1).  Which chord `L` is dressed by `S_RG = E_𝖖(μL)`
selects A₇ vs E₇:

* end attachment — `L` the **short** end chord `(1, H−2)` ⇒ `[A_1, A_7]`
  (`A1AoddToEvenRGKAlgebra(3)`);
* centre attachment — `L` a **central** chord (here the longest type-3 chord
  `(3, 0)`, crossing the chain only at its centre `(1, 3)`) ⇒ `[A_1, E_7]`.

Spectrum generator
------------------
    S_RG  =  E_𝖖(μ · L),   L = the central chord (3, 0),

`[S_RG]_{(m,)} = c_m · (L^m, (m,))`, `c_m = (−q)^m/(q²;q²)_m` — `L` q-commutes
with itself so `L^m = ((3, 0, m),)` is a single cone monomial (verified).
Grading `Γ_RG = Z` by the μ (flavour) charge; positive cone `Z_{≥0}`, height 1;
identity apex.  The μ-refined Schur index computes at **low order** (e.g. the
vacuum index `1 + q² + …`, the `q²` being the adjoined U(1) flavour current) via
the survivor's closed-form trace; high order is limited by `A1A2kKAlg(3)`'s
cone trace reducer on the high central-chord powers `(3,0)^N` (the same long-chord
overflow that makes E₆'s trace WIP) — so validation is structural.

Validation
----------
The UV BPS quiver is the **E₇ Dynkin**: the A₆ chain (the type-1 chords
`(1,0)..(1,5)`, a 6-path) plus the central node `L` has Cartan determinant
`uv_cartan_determinant() == 2` (E₇), vs `8` (A₇) for an end chord, `4` (D₇) for
a second-from-end chord — the central-vs-end fork.  Among trees on 7 nodes only
E₇ has Cartan det 2, certifying E₇ uniquely; this matches the standard E₇ Dynkin
that `BPSKAlgebra` is built from.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fractions import Fraction
from kalgebra import Element
from laurent_poly import LaurentPoly
from rgkalgebra import RGKAlgebra
from grading import Grading
from a1a2k_kalg import A1A2kKAlg
from a1aodd_to_even_rgkalgebra import _e_q_coeff       # c_m = (−q)^m/(q²;q²)_m


class E7RGKAlgebra(RGKAlgebra):
    """`[A_1, E_7]` as a directional new-contract `RGKAlgebra` over
    `A1A2kKAlg(3).add_flavour(1)` (= `[A_1, A_6]` ⊕ U(1) flavour), with
    `S_RG = E_𝖖(μL)` for `L` the **central** chord `(3, 0)` (the centre-of-A₆
    attachment that distinguishes E₇ from the end-chord A₇).  See the module
    docstring."""

    def __init__(self):
        self._A = A1A2kKAlg(3)
        self._aux = self._A.add_flavour(1)          # μ = spectator U(1) flavour
        self._H = self._A.H                          # = 9
        # central dressing chord: longest type, crossing the A6 chain at its centre.
        self._La = max(a for (a, _i) in self._A.cone_data().mult_gens())   # = 3
        self._i0 = 0                                 # (3, 0): central chord (Cartan det 2)

    # ----- RGKAlgebra contract -------------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z = μ-power (label[1]); positive cone Z_{>=0}, height 1.
        return Grading(rank=1, deg=lambda lbl: tuple(lbl[1]), height=(1,),
                       cone_gens=((1,),))

    def apex(self, a):
        """Identity apex: UV labels coincide with auxiliary labels (the survivor
        sits at grading-degree 0)."""
        return (a[0], tuple(a[1]))

    def _central_chord_power(self, m: int):
        """`L^m` as an `A1A2kKAlg` label: the central chord `(3, 0)` raised to `m`
        (a single q-commuting generator power, `((3, 0, m),)`)."""
        return ((self._La, self._i0, m),)

    def _s_rg_component(self, p):
        """`[S_RG]_{(m,)}` — exact, finite, vanishing off the cone.  `S_RG = E_𝖖(μL)`
        ⇒ degree-`m` part is the single dressed label `(L^m, (m,))` with Habiro
        coefficient `c_m`; degree 0 the identity, negative degree empty."""
        (m,) = p
        if m < 0:
            return {}
        if m == 0:
            return {self._aux.identity(): _e_q_coeff(0)}
        return {(self._central_chord_power(m), (m,)): _e_q_coeff(m)}

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG` windowed to q-order ≤ `cutoff`: the μ-tower `{(L^m,(m,)): c_m}`
        for `m < cutoff` (the exact `RG` path uses `_s_rg_component`)."""
        out: dict = {}
        for m in range(cutoff):
            out.update(self._s_rg_component((m,)))
        return out

    # ----- trace: the generic exact-FS μ-refined Schur index (no override) -
    # The hand-rolled μ-graded rg_s_graded/inner_product/trace that lived here
    # (mirroring A1AoddToEvenRGKAlgebra) predated the nested-aux exact-FS engine
    # (#666); the generic engine now reproduces the μ-refined index term-for-term
    # and ~60x faster, so the override is removed and `trace`/`inner_product` are
    # inherited from RGKAlgebra (the bilinear exact-FS pairing over the
    # add_flavour aux, which keeps the μ-character).

    # ----- flavour-aware section split -----------------------------------

    def _section_split(self, label):
        """Auxiliary labels `(chord, (m,))` — μ is the second coordinate (central
        flavour, not a flat-vector charge) — so disable the flavour-shift multiply
        cache (`flav = None`): `multiply` falls back to `from_ir_image(RG(a)·RG(b))`."""
        return tuple(label), None

    # ----- structural validation: the UV BPS quiver is E7 ----------------

    def _qcommute(self, g, h) -> bool:
        return self._A.cone_data().q_commute(g, h)

    def _a6_chain(self):
        """The auxiliary A₆ chain: an induced 6-path of `A1A2kKAlg(3)` chords under
        the crossing (non-q-commuting) relation — the `[A_1, A_6]` simple roots."""
        chords = list(self._A.cone_data().mult_gens())
        edge = lambda g, h: not self._qcommute(g, h)
        type1 = [g for g in chords if g[0] == 1]

        def find(cands):
            adj = {g: [h for h in cands if h != g and edge(g, h)] for g in cands}

            def dfs(path, used):
                if len(path) == 6:
                    return list(path)
                for nb in adj[path[-1]]:
                    if nb in used or any(edge(nb, p) for p in path[:-1]):
                        continue
                    r = dfs(path + [nb], used | {nb})
                    if r:
                        return r
                return None
            for s in cands:
                r = dfs([s], {s})
                if r:
                    return r
            return None
        chain = find(type1) or find(chords)
        if chain is None:
            raise RuntimeError("no A6 chain (induced 6-path) found")
        return chain

    def _cartan_det(self, nodes) -> int:
        """Determinant of the Cartan matrix `2I − adjacency` of the crossing quiver
        on `nodes` (E₇ = 2, A₇ = 8, D₇ = 4)."""
        n = len(nodes)
        edge = lambda g, h: not self._qcommute(g, h)
        adj = [[1 if (i != j and edge(nodes[i], nodes[j])) else 0 for j in range(n)]
               for i in range(n)]
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
        """UV BPS quiver Cartan determinant: the A₆ chain plus the central chord
        `L`.  `== 2` (E₇) — the structural witness that this flow realises
        `[A_1, E_7]` (vs 8 = A₇ for an end chord)."""
        L = (self._La, self._i0)
        return self._cartan_det(self._a6_chain() + [L])

    def verify_is_E7(self) -> bool:
        """`uv_cartan_determinant() == 2` — the UV BPS quiver is the E₇ Dynkin."""
        return self.uv_cartan_determinant() == 2


if __name__ == "__main__":
    T = E7RGKAlgebra()
    print("E7RGKAlgebra = [A_1, E_7]  via  drop central node → [A_1, A_6] ⊕ U(1) flavour")
    print("  aux =", type(T.auxiliary()).__name__, "= A1A2kKAlg(3).add_flavour(1)",
          "  central chord L = (", T._La, ",", T._i0, ")")
    print("  A6 chain:", T._a6_chain())
    print("  UV BPS quiver Cartan determinant =", T.uv_cartan_determinant(),
          "(E7 = 2)   verify_is_E7:", T.verify_is_E7())
    print("  verify_rg_unital:", T.verify_rg_unital())
    print("  vacuum Schur index Tr(1) (low order; q²=U(1) current):",
          {e: str(c) for e, c in T.trace(((), (0,)), 2).coeffs.items()})
