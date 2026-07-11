"""Algebra-level gluing of pinned polygons along a seam.

Cutting the fan n-gon at internal edge k (the diagonal (0, k+2))
yields two fan polygons — left = the (k+3)-gon (triangles 0..k),
right = the (n-k-1)-gon (triangles k+1..n-3, re-fanned at vertex 0) —
each pinned over its own Gamma~.  `PinnedSeam` is the inverse
operation AT ALGEBRA LEVEL: the target's Gamma~ is covered by the two
piece lattices with the two CUT directions merged into the seam
coordinate (`glue_charge`), and

* `embed_left` / `embed_right` (merge against the other piece's
  identity) are certified algebra HOMOMORPHISMS of pinned tori — the
  glued cocycle restricts to each piece's cocycle, so the pinned
  sub-theories embed into the bigger Gamma~ exactly as the
  pinned picture wants;
* `arc` reassembles EVERY stated arc of the target from piece arcs:
  contained arcs embed, seam-crossing arcs are the state-sum merge
  sum_eps left(.., eps) (x) right(eps, ..) (`glue_state_sum`);
* the SEAM DIRECTION BECOMES DYNAMICAL: on seam-crossing arcs the
  merged cut coordinate varies term-by-term (it is a new internal
  F-direction — the new mutable node of the glued exchange matrix),
  while the surviving boundary block stays pinned.  This is the
  inverse of the certified node-drop RG (cut = drop the seam node;
  glue = adjoin it).

The pentagon demo: the A_2 dressed-chord
canonicals T~_p of `PinnedPentagonKAlg` assemble EXACTLY from
square + triangle piece data — merged chords dressed by merged side
monomials, multiplied in the glued torus — so the UV canonical
algebra (fast RG(a)'s + fast mult) is built from polygon pieces.
Even-gon pieces carry the known
even-marks U(1) subtlety in their OWN unpinning;
the seam mechanics here do not depend on it.
"""

from __future__ import annotations

from half_laurent import HalfLaurent, ONE as HL_ONE
from pinned_polygon import PinnedPolygon, glue_charge, glue_state_sum


def fan_cut_maps(n: int, k: int):
    """Edge-id maps for cutting the fan n-gon at internal edge k.

    Returns ``(n_left, n_right, left_map, right_map)`` where the maps
    send piece edge ids to target edge ids; both cut sides (left side
    k+2, right side 0) map to the target internal (seam) edge k.
    Target ids: internal 0..n-4, boundary side j = (n-3)+j; piece ids
    follow the same convention on their own fans.
    """
    if not (0 <= k <= n - 4):
        raise ValueError(f"internal edge k={k} out of range for n={n}")
    n_left, n_right = k + 3, n - k - 1
    left_map = {}
    for i in range(k):                      # left internal -> target internal
        left_map[i] = i
    for j in range(k + 2):                  # left sides 0..k+1 -> target sides
        left_map[(n_left - 3) + j] = (n - 3) + j
    left_map[(n_left - 3) + (k + 2)] = k    # left cut side -> seam
    right_map = {}
    for i in range(n_right - 3):            # right internal -> target internal
        right_map[i] = (k + 1) + i
    right_map[(n_right - 3) + 0] = k        # right cut side (0) -> seam
    for j in range(1, n_right):             # right sides 1.. -> target sides
        right_map[(n_right - 3) + j] = (n - 3) + (k + 1) + j
    return n_left, n_right, left_map, right_map


class PinnedSeam:
    """The glued presentation of the fan n-gon over its two pieces."""

    def __init__(self, n: int, k: int):
        self.n, self.k = n, k
        n_left, n_right, left_map, right_map = fan_cut_maps(n, k)
        self.target = PinnedPolygon(n)
        self.left = PinnedPolygon(n_left)
        self.right = PinnedPolygon(n_right)
        self.left_map = left_map
        self.right_map = right_map
        self.seam_edge = k
        self.left_cut_side = k + 2          # polygon-side index in the left fan
        self.right_cut_side = 0             # polygon-side index in the right fan

    # -- side bookkeeping ---------------------------------------------------

    def side_owner(self, s: int):
        """('left', piece side) or ('right', piece side) of target side s."""
        if s <= self.k + 1:
            return ("left", s)
        return ("right", s - (self.k + 1))

    # -- embeddings (merge against the other piece's identity) ---------------

    def embed_left(self, x: dict) -> dict:
        one = {1: self.right.one(), -1: {}}
        return glue_state_sum(self.target, {1: x, -1: {}}, one,
                              self.left_map, self.right_map)

    def embed_right(self, x: dict) -> dict:
        one = {1: self.left.one(), -1: {}}
        return glue_state_sum(self.target, one, {1: x, -1: {}},
                              self.left_map, self.right_map)

    # -- arcs ------------------------------------------------------------------

    def arc(self, s: int, t: int, eps_s: int, eps_t: int) -> dict:
        """The target arc (s, t; eps_s, eps_t) assembled from piece
        data: embedded if contained in one piece, state-sum merged if
        it crosses the seam."""
        (os_, ps) = self.side_owner(s)
        (ot_, pt) = self.side_owner(t)
        if os_ == ot_ == "left":
            return self.embed_left(self.left.arc_F(ps, pt, eps_s, eps_t))
        if os_ == ot_ == "right":
            return self.embed_right(self.right.arc_F(ps, pt, eps_s, eps_t))
        if os_ == "right":                  # normalize: left endpoint first
            return self.arc(t, s, eps_t, eps_s)
        lby = {e: self.left.arc_F(ps, self.left_cut_side, eps_s, e)
               for e in (1, -1)}
        rby = {e: self.right.arc_F(self.right_cut_side, pt, e, eps_t)
               for e in (1, -1)}
        return glue_state_sum(self.target, lby, rby,
                              self.left_map, self.right_map)

    # -- the dynamical seam ------------------------------------------------------

    def seam_exponents(self, x: dict) -> set:
        """The set of seam-coordinate values over the terms of x —
        more than one value == the seam direction is dynamical on x."""
        return {u[self.seam_edge] for u in x}
