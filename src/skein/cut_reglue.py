"""`CutReglue` — the certified bordered-chart pipeline for one cut
edge of a closed Triangle-based chart.  The T-pin6 universality sweep
certified it over all tetrahedron cuts x once-crossing curves, and
the chamber-free closure on the once-punctured torus is certified.

For a closed chart `t`, a cut edge `c`, and a core curve `gamma`
crossing `c` exactly once:

  1. the cut chart on the FULL frozen-extended lattice (bordered spec
     auto-found once per cut, embedded) — the extended `BPSKAlgebra`
     is `self.B`;
  2. the stated-arc images of the cut-open curve for all four
     endpoint-state sectors, built AUTOMATICALLY from
     `realize_curve(t, gamma)` (per-triangle phi state sum, free
     states at the two boundary slots, projected to the cut chart
     with Weyl coefficients) — `arc_image`;
  3. the arc dictionary by THE LEADING-TERM LAW:
     gamma~ = -min(arc image support) — X_{-gamma~} is the q->0
     leading term of F(-gamma~) and the stated arc's minimal charge
     is that leading term (measured on both charts).
     The bare inclusion is asserted per sector — `host`;
  4. the regluing:  F_closed(-gamma) = CornerPass(EdgePass({...}))
     with THE SINGLE T-pin6 convention — edge mismatch frames
     w(+,-) = q^{-1}, w(-,+) = q (eps1 = state at the lower boundary
     id), corner pass with u = the internal edge the arc joins in the
     HIGHER boundary id's triangle, v = in the lower's (teleport
     (z_u,z_v) = (-+1,+-1) -> (0,0); frame odd-unbalanced terms by
     q^{z_v - z_u}; counit unit at charge 0) — `reglue`.
"""

import os
import sys
from itertools import product as iproduct

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from laurent_poly import LaurentPoly

from curve_realization import realize_curve
from half_laurent import HalfLaurent, ONE as HL_ONE
from phi_map import phi
from skein_sphere_atlas import BorderedSkeinAtlas
from y_delta import project_triangle_monomial, _weyl_correction

__all__ = ["CutReglue"]

ONE = LaurentPoly({0: 1})


def Q(k):
    return LaurentPoly({k: 1})


class CutReglue:
    """The automated pipeline for one cut edge of a closed chart."""

    def __init__(self, t, c):
        self.t, self.c = t, c
        self.tc = t.cut(c)
        self.INT = self.tc.internal_edge_ids
        self.BND = list(self.tc.boundary_edge_ids)        # ascending
        self.sig = self.tc.sigma()
        at = BorderedSkeinAtlas(self.tc)

        def embed(v):
            out = [0] * self.tc.n_edges
            for i, e in enumerate(self.INT):
                out[e] = v[i]
            return tuple(out)

        from bps_kalgebra import BPSKAlgebra
        spec = [embed(tuple(s)) for s in at.root_kalg.spec]
        n = self.tc.n_edges
        units = [tuple(1 if k == i else 0 for k in range(n)) for i in range(n)]
        self.B = BPSKAlgebra(pairing=self.sig, node_charges=units, spec=spec)
        # boundary id per adjacent triangle (slot structure is preserved)
        self.bnd_of_tri = {}
        for ti in range(t.n_triangles):
            for pos, e in enumerate(t.triangle_edges[ti]):
                if e == c:
                    self.bnd_of_tri[ti] = self.tc.triangle_edges[ti][pos]
        assert sorted(self.bnd_of_tri.values()) == sorted(self.BND)
        self._F = {}

    def F(self, gt):
        """F(-gt), or None if the solve explodes (run under a ulimit
        memory guard: an explosive junk candidate raises MemoryError
        and is treated as not-a-host rather than OOM-killing the
        sweep)."""
        if gt not in self._F:
            try:
                self._F[gt] = dict(self.B.F(tuple(-x for x in gt)))
            except MemoryError:
                self._F[gt] = None
        return self._F[gt]

    def weylize(self, ch, oc):
        class _W:
            def sigma(s):
                return self.sig
            n_edges = self.tc.n_edges
        w = oc.shift(-_weyl_correction(_W(), ch))
        d = {}
        for e, co in w.items():
            assert e.denominator == 1
            d[int(e)] = d.get(int(e), 0) + co
        return LaurentPoly(d)

    def arc_image(self, gamma, eps_by_bnd):
        """Quantum trace of the cut-open curve with free end states
        (eps_by_bnd: boundary id -> state), as a cut-chart charge dict."""
        arcs = realize_curve(self.t, gamma)
        internal_states = sorted(
            e for e, x in enumerate(gamma) if x and e != self.c)
        out = {}
        for bits in iproduct((1, -1), repeat=len(internal_states)):
            mu = dict(zip(internal_states, bits))

            def state_at(ti, e):
                if e == self.c:
                    return eps_by_bnd[self.bnd_of_tri[ti]]
                return mu[e]

            triples, coef, dead = [], HL_ONE, False
            for ti, tlist in enumerate(arcs):
                if not tlist:
                    triples.append((0, 0, 0))
                    continue
                (a,) = tlist
                f = phi(a.arc_type, state_at(ti, a.edge_at_eps),
                        state_at(ti, a.edge_at_eps_prime))
                if f.is_zero():
                    dead = True
                    break
                ((e, co),) = f.items()
                triples.append(e)
                coef = coef * co
            if dead:
                continue
            pr = project_triangle_monomial(self.tc, triples, coef)
            if pr is None:
                continue
            ch, oc = pr
            out[ch] = out.get(ch, HalfLaurent.zero()) + oc
        return {k: self.weylize(k, v) for k, v in out.items()
                if not v.is_zero()}

    def host(self, gamma, eps_by_bnd):
        """The extended charge hosting the arc image as the exact bare
        part of F(-gamma~) — by THE LEADING-TERM LAW (measured on the
        tetrahedron and the bipyramid):

            gamma~ = - min(arc image support)

        (X_{-gamma~} is the q->0 leading term of F(-gamma~), and the
        stated arc's minimal charge IS that leading term).  The
        boundary entries come out as (-eps_1, -eps_2) automatically.
        The bare inclusion is asserted, so a violation of the law
        fails loudly rather than silently."""
        A = self.arc_image(gamma, eps_by_bnd)
        gt = tuple(-x for x in min(A))
        Ff = self.F(gt)
        if Ff is None:
            raise AssertionError(
                f"host F-solve exploded at {gt} (raise the memory guard)")
        if not all(ch in Ff and Ff[ch] == v for ch, v in A.items()):
            raise AssertionError(
                f"leading-term law violated: -min(A) = {gt} does not "
                f"host the arc image for {gamma}, eps {eps_by_bnd}")
        return A, [gt]

    def reglue(self, gamma):
        """The complete rule; returns (closed-charge dict, hosts)."""
        b_lo, b_hi = self.BND
        arcs = realize_curve(self.t, gamma)
        joins = {}                      # boundary id -> adjacent internal edge
        for ti, b in self.bnd_of_tri.items():
            (a,) = arcs[ti]
            other = (a.edge_at_eps if a.edge_at_eps != self.c
                     else a.edge_at_eps_prime)
            joins[b] = other
        u, v = joins[b_hi], joins[b_lo]
        W = {(1, 1): ONE, (-1, -1): ONE, (1, -1): Q(-1), (-1, 1): Q(1)}
        # index map: cut-chart charge -> closed charge (merge boundary)
        keep = [e for e in range(self.tc.n_edges) if e not in self.BND]
        pos_of_closed = {}              # closed edge -> cut edge (identity ids)
        tot, hosts = {}, {}
        for e1 in (1, -1):
            for e2 in (1, -1):
                eps = {b_lo: e1, b_hi: e2}
                _A, found = self.host(gamma, eps)
                if len(found) != 1:
                    raise AssertionError(
                        f"host not unique for {gamma} eps {eps}: {found}")
                hosts[(e1, e2)] = found[0]
                for ch, val in self.F(found[0]).items():
                    s = ch[b_lo] + ch[b_hi]
                    assert s % 2 == 0
                    m = [0] * self.t.n_edges
                    for e in range(self.tc.n_edges):
                        if e == b_lo:
                            m[self.c] = s // 2
                        elif e != b_hi:
                            m[e] = ch[e]
                    m = tuple(m)
                    tot[m] = tot.get(m, LaurentPoly.zero()) + W[(e1, e2)] * val
        out = {}
        for m, val in tot.items():
            if m[self.c] == 0:
                zu, zv = m[u], m[v]
                if zu * zv == -1:
                    mm = list(m)
                    mm[u] = 0
                    mm[v] = 0
                    m2 = tuple(mm)
                    out[m2] = out.get(m2, LaurentPoly.zero()) + val
                    continue
                if (zu + zv) % 2:
                    val = val * Q(zv - zu)
            out[m] = out.get(m, LaurentPoly.zero()) + val
        z = (0,) * self.t.n_edges
        out[z] = out.get(z, LaurentPoly.zero()) + ONE
        return ({k: v for k, v in out.items() if dict(v._coeffs)}, hosts)
