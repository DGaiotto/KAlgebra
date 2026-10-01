"""
A1A2k.py
========

Light wrapper around :class:`bps_kalgebra.BPSKAlgebra` for the
[A_1, A_{2k}] Argyres-Douglas family.  Designed to function as a HELPER
for the standalone `A1A2kKAlg(KAlgebra)` subclass (`a1a2k_kalg.py`,
analogous to `kalgebra_samples.PentagonKAlg` and `HeptagonKAlg`).

Wrapper-as-helper means:

  * The wrapper supplies the QUIVER DATA for any k (pairing matrix,
    node charges, ρ on the lattice, the k named ρ-orbits).
  * The wrapper supplies the CHORD GEOMETRY -- the verified (length,
    shift) assignment from `A1A2k_naming_audit.predicted_lengths_and_shifts(k)`
    and the resulting `L_chord(a, b)` unordered-endpoint API.
  * The wrapper supplies the EMPIRICAL STRUCTURE -- products of named
    generators (via the BPSKAlgebra), q-commute factors, and a base-
    table extraction routine that gives the per-k product table in the
    same `('I',) / ('L', (k, i)) / ('X', ((k1, i1), (k2, i2)))` term
    language used by `kalgebra_samples.HeptagonKAlg`.

The standalone `A1A2kKAlg` will hard-code (or lazily extract) the base
table per k and not depend on `BPSKAlgebra` at runtime.
"""
from __future__ import annotations

from typing import Iterable

from bps_kalgebra import BPSKAlgebra


class _PluckerPair(Exception):
    """Raised by `A1A2k._qc_closed_form` to signal a crossing chord pair
    (multi-term Plücker product, no single q-commute factor)."""


def A2k_pairing(k: int) -> list[list[int]]:
    """Linear  A_{2k}  pairing matrix:  B_{a, a+1} = 1,  B_{a+1, a} = -1."""
    n = 2 * k
    B = [[0] * n for _ in range(n)]
    for a in range(n - 1):
        B[a][a + 1] = 1
        B[a + 1][a] = -1
    return B


def A2k_node_charges(k: int) -> list[tuple[int, ...]]:
    n = 2 * k
    return [tuple(1 if p == a else 0 for p in range(n)) for a in range(n)]


def L_i_seed(k: int, i: int) -> tuple[int, ...]:
    """Natural-labeling lattice seed for L_{i, 0}, the length-(i+1) chord
    starting at vertex 0 on the (2k+3)-gon.  Derived from the
    alternating-pattern seed of the (old) length-(i+1) orbit by applying
    ρ^{-S_old} on the BPS lattice (see
    `A1A2k_naming_audit.natural_orbit_seeds`)."""
    if not (1 <= i <= k):
        raise ValueError(f"i must be in [1, {k}], got {i}")
    from A1A2k_naming_audit import natural_orbit_seeds
    return natural_orbit_seeds(k)[i]


class A1A2k:
    """Wrapper around BPSKAlgebra for the  [A_1, A_{2k}]  family."""

    def __init__(self, k: int, *, verify: str = "off"):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.n = 2 * k
        self.cyc = 2 * k + 3
        self.A = BPSKAlgebra(
            pairing=A2k_pairing(k),
            node_charges=A2k_node_charges(k),
            verify=verify,
        )
        # Natural labeling: orbit a has length a+1 and shift 0, so
        # L_{a, j} has chord endpoints (j, j+a+1) on the (2k+3)-gon.
        self.lengths = {a: a + 1 for a in range(1, k + 1)}
        self.shifts = {a: 0 for a in range(1, k + 1)}
        # Antisymmetric pairing matrix (Z[q^±] q-commute exponent is 2·⟨,⟩).
        self._B = A2k_pairing(k)
        self._L: dict[tuple[int, int], tuple[int, ...]] = {}
        self._L_inv: dict[tuple[int, ...], tuple[int, int]] = {}
        self._build_L_orbits()
        # charge -> list of (coef, (i1,j1), (i2,j2)) when L_{i1,j1} L_{i2,j2}
        # gives a single-term product at that charge.
        self._quad_lookup: dict[tuple[int, ...], list] = {}
        self._build_quadratic_lookup()

    # -- construction ---------------------------------------------------------

    def _build_L_orbits(self) -> None:
        """Compute the tropical charges of every L_{i, j} via ρ from the seed."""
        for i in range(1, self.k + 1):
            seed = L_i_seed(self.k, i)
            self._L[(i, 0)] = seed
            self._L_inv[seed] = (i, 0)
            cur = seed
            for j in range(1, self.cyc):
                cur = tuple(self.A.rho(cur))
                self._L[(i, j)] = cur
                self._L_inv[cur] = (i, j)
            # Sanity check: ρ-orbit must close
            closed = tuple(self.A.rho(cur))
            if closed != seed:
                raise RuntimeError(
                    f"ρ-orbit of L_{{{i}, 0}} = {seed} did not close: "
                    f"ρ^{self.cyc} = {closed}"
                )

    def _build_quadratic_lookup(self) -> None:
        """For every ordered pair of generators, record the charges of
        the canonical-basis terms in their product, so that arbitrary
        charges in the relations report can be backresolved into  L · L
        forms when possible.

        Restriction: only record (la, lb) pairs whose product is SINGLE-
        TERM (= they q-commute), so the X-name `((la, lb))` always
        refers to a q-commuting pair (= the BPS X-basis convention).
        Pairs that produce multi-term Plücker outputs don't get
        recorded -- they don't define a clean X-name for their output."""
        for i1 in range(1, self.k + 1):
            for j1 in range(self.cyc):
                g1 = self.charge(i1, j1)
                for i2 in range(1, self.k + 1):
                    for j2 in range(self.cyc):
                        g2 = self.charge(i2, j2)
                        prod = self.A.multiply(g1, g2)
                        # Only record SINGLE-TERM products
                        if len(prod.terms) != 1:
                            continue
                        for ch, coef in prod.terms.items():
                            ch = tuple(ch)
                            if self._L_inv.get(ch) is not None:
                                continue  # already a single L_{i, j}
                            if ch == (0,) * self.n:
                                continue
                            self._quad_lookup.setdefault(ch, []).append(
                                (coef, (i1, j1), (i2, j2))
                            )

    # -- public accessors -----------------------------------------------------

    def charge(self, i: int, j: int) -> tuple[int, ...]:
        """Tropical (lattice) charge of  L_{i, j}."""
        if not (1 <= i <= self.k):
            raise ValueError(f"i must be in [1, {self.k}], got {i}")
        return self._L[(i, j % self.cyc)]

    def name(self, gamma: Iterable[int]) -> str | None:
        """Inverse lookup:  charge -> 'L_{i, j}'  (or None if not on the
        length-one canonical-basis sheet)."""
        g = tuple(gamma)
        if g == (0,) * self.n:
            return "1"
        ij = self._L_inv.get(g)
        if ij is None:
            return None
        i, j = ij
        return f"L_{{{i}, {j}}}"

    def name_or_decompose(self, gamma: Iterable[int]) -> str | None:
        """Like  :meth:`name`, but if  gamma  is a higher-weight canonical
        basis element AND it arises as a UNIT-coefficient single-term
        product  L_{a,b} · L_{c,d},  return that decomposition's name.
        For all other higher-weight charges, returns None.

        We restrict to unit coefficients to avoid trivial circular
        ``F_γ = c · L_a L_b  =>  F_γ described as `c · L_a L_b`''
        renderings."""
        nm = self.name(gamma)
        if nm is not None:
            return nm
        g = tuple(gamma)
        pairs = self._quad_lookup.get(g, [])
        unit_pairs = [(coef, a, b) for coef, a, b in pairs if str(coef) == "1"]
        if not unit_pairs:
            return None
        _, (i1, j1), (i2, j2) = min(unit_pairs, key=lambda t: (t[1], t[2]))
        return f"L_{{{i1}, {j1}}} L_{{{i2}, {j2}}}"

    # -- API hooks for use as a BPSKAlgebra translator ------------------------

    def bps_algebra(self) -> BPSKAlgebra:
        """The underlying BPSKAlgebra (for callers who want lattice-level
        access to charges, F-elements, products, traces, Schur indices)."""
        return self.A

    def to_bps_charge(self, label) -> tuple[int, ...]:
        """Translate an  (i, j)  L-label  to the corresponding lattice
        charge in the BPS quiver.  Returns the input unchanged if it
        already looks like a charge tuple."""
        if isinstance(label, tuple) and len(label) == 2 and all(
                isinstance(x, int) for x in label) and (
                self.n != 2 or label in self._L_inv):
            return self.charge(*label)
        return tuple(label)

    def lattice_data(self) -> dict:
        """Quiver / lattice metadata for downstream consumers."""
        return {
            "k": self.k,
            "rank": self.n,
            "cyclic_order": self.cyc,
            "pairing": A2k_pairing(self.k),
            "node_charges": A2k_node_charges(self.k),
            "L_charges": {ij: list(g) for ij, g in self._L.items()},
        }

    def L(self, i: int, j: int) -> tuple[int, ...]:
        """Alias of :meth:`charge`."""
        return self.charge(i, j)

    def rho(self, gamma: Iterable[int]) -> tuple[int, ...]:
        return tuple(self.A.rho(tuple(gamma)))

    # -- products in canonical-basis form -------------------------------------

    def multiply(self, a, b):
        """Return the canonical-basis expansion of  L_a · L_b  as a
        dict  {charge: LaurentPoly}, where  a, b  may be either (i, j)
        index pairs or raw tropical-charge tuples."""
        ga = self.charge(*a) if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], int) and a[0] <= self.k else tuple(a)
        gb = self.charge(*b) if isinstance(b, tuple) and len(b) == 2 and isinstance(b[0], int) and b[0] <= self.k else tuple(b)
        # Defensive: if a is a 2-tuple in the rank-2 case (k = 1), the
        # length-2 charge collides with (i, j); disambiguate by checking
        # whether the lookup succeeds with that interpretation.
        if isinstance(a, tuple) and len(a) == 2 and self.n == 2:
            ga = tuple(a) if a in self._L_inv else self.charge(*a)
        if isinstance(b, tuple) and len(b) == 2 and self.n == 2:
            gb = tuple(b) if b in self._L_inv else self.charge(*b)
        prod = self.A.multiply(ga, gb)
        return dict(prod.terms)

    def pretty_term(self, charge: tuple[int, ...], coef) -> str:
        """Render a single term  coef * F_charge  using L-naming when
        possible, otherwise as  ?(γ)."""
        nm = self.name(charge)
        if nm is None:
            nm = f"?{charge}"
        cstr = str(coef)
        if cstr == "1":
            return nm
        return f"({cstr}) {nm}"

    def pretty_product(self, a, b) -> str:
        """Pretty-print  L_a · L_b  in L-notation."""
        terms = self.multiply(a, b)
        # Sort by some canonical order
        ordered = sorted(terms.items(),
                         key=lambda kv: (self.name(kv[0]) is None, self.name(kv[0]) or str(kv[0])))
        parts = [self.pretty_term(ch, c) for ch, c in ordered]
        return " + ".join(parts) if parts else "0"

    # -- diagnostics ----------------------------------------------------------

    def list_L(self) -> None:
        """Print the table of  L_{i, j}  ↔ tropical charge."""
        print(f"[A_1, A_{2*self.k}]  ((2k+3)={self.cyc}-gon)   "
              f"k = {self.k},  rank = {self.n}")
        for i in range(1, self.k + 1):
            print(f"  L_{{{i}, *}}:")
            for j in range(self.cyc):
                print(f"     L_{{{i}, {j}}} = {self._L[(i, j)]}")

    def quadratic_table(self, classes_left=None, classes_right=None,
                        j_range=None):
        """Print the canonical-basis expansion of  L_{i, 0} · L_{i', j}
        for the requested classes and j values.  Defaults: all classes,
        j in 0..cyc-1."""
        cl = list(classes_left or range(1, self.k + 1))
        cr = list(classes_right or range(1, self.k + 1))
        jr = list(j_range if j_range is not None else range(self.cyc))
        for i in cl:
            for ip in cr:
                print(f"\n  L_{{{i}, 0}} · L_{{{ip}, j}}:")
                for j in jr:
                    s = self.pretty_product((i, 0), (ip, j))
                    print(f"     j = {j}:  {s}")

    # -- geometric classifier -------------------------------------------------

    def classify(self, i1: int, j1: int, i2: int, j2: int) -> str:
        """Geometric relationship between L_{i1, j1} and L_{i2, j2} as
        diagonals  (s_{j1}, s_{j1 + i1 + 1})  and  (s_{j2}, s_{j2 + i2 + 1})
        of the (2k+3)-gon."""
        cyc = self.cyc
        e1 = (j1 % cyc, (j1 + i1 + 1) % cyc)
        e2 = (j2 % cyc, (j2 + i2 + 1) % cyc)
        if set(e1) == set(e2):
            return "same"
        if set(e1) & set(e2):
            return "share"
        # Cross iff endpoints of e2 interleave with endpoints of e1.
        a, b = e1
        # Walk forward from a to b (not through b), see if exactly one of
        # e2's endpoints is encountered.
        arc1, arc2 = set(), set()
        x = (a + 1) % cyc
        while x != b:
            arc1.add(x)
            x = (x + 1) % cyc
        x = (b + 1) % cyc
        while x != a:
            arc2.add(x)
            x = (x + 1) % cyc
        c, d = e2
        in1 = (c in arc1, d in arc1)
        in2 = (c in arc2, d in arc2)
        if (in1[0] and in2[1]) or (in1[1] and in2[0]):
            return "cross"
        return "disjoint"

    # -- relations report -----------------------------------------------------

    def relations_report(self, *, sort_by="i_ip_j"):
        """Print every quadratic product  L_{i, 0} · L_{i', j}  tagged
        by geometric class, for every (i, i') and j in Z/(2k+3).

        ρ-invariance reduces the full table to this slice without loss."""
        cyc = self.cyc
        rows = []
        for i in range(1, self.k + 1):
            for ip in range(1, self.k + 1):
                for j in range(cyc):
                    cls = self.classify(i, 0, ip, j)
                    prod = self.multiply((i, 0), (ip, j))
                    named = {self.name(ch): coef for ch, coef in prod.items()}
                    # tag terms in the product that are themselves L_{?, ?}
                    sym_terms = [(coef, name) for name, coef in named.items()
                                 if name is not None]
                    higher_terms = [(coef, ch) for ch, coef in prod.items()
                                    if self.name(ch) is None]
                    rows.append({
                        "i": i, "ip": ip, "j": j, "cls": cls,
                        "sym_terms": sym_terms,
                        "higher_terms": higher_terms,
                        "prod": dict(prod),
                    })
        # Print grouped by (i, ip), then by classification
        print(f"=== Quadratic-product report  [A_1, A_{2*self.k}]  "
              f"(k = {self.k},  (2k+3) = {self.cyc}-gon) ===")
        for i in range(1, self.k + 1):
            for ip in range(1, self.k + 1):
                print(f"\n  L_{{{i}, 0}} · L_{{{ip}, j}}  for j in Z/{cyc}:")
                slice_rows = [r for r in rows if r["i"] == i and r["ip"] == ip]
                for r in sorted(slice_rows, key=lambda r: r["j"]):
                    parts = []
                    for ch, coef in r["prod"].items():
                        nm = self.name_or_decompose(ch) or f"?{ch}"
                        cs = str(coef)
                        parts.append(nm if cs == "1" else f"({cs}) {nm}")
                    rhs = " + ".join(parts) if parts else "0"
                    flag = ""
                    if r["higher_terms"] and r["sym_terms"]:
                        flag = "  [mixed]"
                    elif r["higher_terms"]:
                        flag = "  [HIGHER]"
                    print(f"     j={r['j']}  [{r['cls']:>8}]  "
                          f"L_{{{i}, 0}} L_{{{ip}, {r['j']}}}  =  {rhs}{flag}")
        return rows

    def universal_patterns(self):
        """Try to identify universal (k-independent) patterns in the
        same-class crossing/sharing relations.  Returns a dict of
        observed pattern fingerprints."""
        cyc = self.cyc
        out = {}
        for i in range(1, self.k + 1):
            # crossings of  L_{i, 0}  with  L_{i, dj}  for dj = 1..cyc-1
            for dj in range(1, cyc):
                cls = self.classify(i, 0, i, dj)
                prod = self.multiply((i, 0), (i, dj))
                key = ("same-class", i, dj, cls)
                # Encode the RHS as a list of (name-or-charge, coef) tuples
                rhs = []
                for ch, coef in prod.items():
                    nm = self.name(ch)
                    rhs.append((nm if nm else "H", str(coef)))
                out[key] = tuple(sorted(rhs))
        return out

    # -- Geometric chord API (matches heptagon_kalg.HeptagonKAlg shape) -------

    def L_chord(self, a: int, b: int) -> tuple[int, ...]:
        """Tropical charge of the chord on the (2k+3)-gon with endpoints
        `{a, b}`.  Endpoints unordered.

        Convention (verified by `A1A2k_naming_audit.predicted_lengths_and_shifts`):
          * Cyclic distance d = 1 (edges) → algebra identity `(0, ..., 0)`.
          * Cyclic distance d = lengths[i] for some orbit i ∈ {1, ..., k} →
            L((i, (start - shifts[i]) mod (2k+3))), where `start` is the
            chord's canonical 'forward-distance' starting vertex.
        """
        cyc = self.cyc
        a_m, b_m = a % cyc, b % cyc
        if a_m == b_m:
            raise ValueError(f"L_chord: distinct endpoints required, got {a}, {b}")
        forward = (b_m - a_m) % cyc
        backward = (a_m - b_m) % cyc
        if forward <= backward:
            d, start = forward, a_m
        else:
            d, start = backward, b_m
        if d == 1:
            return (0,) * self.n   # edge -- identity
        # Find orbit i with length d.
        for i in range(1, self.k + 1):
            if self.lengths[i] == d:
                j = (start - self.shifts[i]) % cyc
                return self.charge(i, j)
        raise ValueError(
            f"L_chord: no named orbit has length {d}; "
            f"(2k+3)-gon has lengths up to k+1 = {self.k + 1}"
        )

    def L_chord_label(self, gamma: Iterable[int]) -> tuple[int, int] | None:
        """Inverse of `L_chord`.  Returns `(a, b)` with `a < b` if γ is a
        named L's lattice charge, else `None` (identity → None too)."""
        g = tuple(gamma)
        if g == (0,) * self.n:
            return None
        ij = self._L_inv.get(g)
        if ij is None:
            return None
        i, j = ij
        a = (j + self.shifts[i]) % self.cyc
        b = (a + self.lengths[i]) % self.cyc
        return (min(a, b), max(a, b))

    # -- q-commute / X-basis convention helpers ------------------------------

    def pair_antisym(self, gamma1: Iterable[int], gamma2: Iterable[int]) -> int:
        """Antisymmetric quiver pairing ⟨γ_1, γ_2⟩ via the B matrix.
        For monomial pairs, L_a · L_b = q^{2⟨γ_a, γ_b⟩} · L_b · L_a."""
        g1, g2 = tuple(gamma1), tuple(gamma2)
        return sum(g1[i] * self._B[i][j] * g2[j]
                   for i in range(self.n) for j in range(self.n))

    def q_commute_factor(self, la: tuple[int, int], lb: tuple[int, int]) -> int | None:
        """For two named L-labels  la = (k_a, j_a),  lb = (k_b, j_b),
        return integer c with  L_la · L_lb  =  q^c · L_lb · L_la, OR
        `None` if (la, lb) is a Plücker pair (multi-term product).

        Closed-form (no BPSKAlgebra runtime call): for non-crossing chord
        pairs on the (2k+3)-gon, the q-commute factor depends only on
        chord-pair geometry.  See `_qc_closed_form` for the arc-parity
        rule.  Returns None for crossing pairs (Plücker).
        """
        # Fast path: closed-form arc-parity rule.
        try:
            return self._qc_closed_form(la, lb)
        except _PluckerPair:
            return None

    def _qc_closed_form(self, la: tuple[int, int], lb: tuple[int, int]) -> int:
        """Closed-form q-commute factor via the arc-parity rule.

        For chords L_a = (j_a, j_a + a + 1) and L_b = (j_b, j_b + b + 1)
        on the (2k+3)-gon, list the distinct endpoints in cyclic CCW
        order from L_a's first endpoint; classify each arc by its
        endpoint labels (only-L_a, only-L_b, or shared); count odd
        arcs.  Since H = 2k+3 is odd, exactly 1 or 3 arcs are odd.

        Rule:
          * Crossing chords (interiors meet): raises `_PluckerPair`.
          * Same chord: returns 0.
          * Exactly 1 odd arc, with both endpoints strictly one chord's
            only (NOT shared), in L_a → L_b CCW direction: c = -2.
          * Same but L_b → L_a CCW direction: c = +2.
          * Otherwise (odd arc adjacent to shared, or interior arc, or
            3 odd arcs): c = 0.
        """
        ka, ja = la; kb, jb = lb
        H = self.cyc
        c1 = (ja % H, (ja + ka + 1) % H)
        c2 = (jb % H, (jb + kb + 1) % H)
        if set(c1) == set(c2):
            return 0  # same chord
        p, q = c1
        a_endpts = {p, q}
        b_endpts = set(c2)
        # Crossing check: alternate endpoints in cyclic CCW.
        if not (a_endpts & b_endpts):
            # Check whether L_b endpoints lie on opposite arcs of L_a.
            arc_fwd = set()
            x = (p + 1) % H
            while x != q:
                arc_fwd.add(x); x = (x + 1) % H
            r, s = c2
            if (r in arc_fwd) != (s in arc_fwd):
                raise _PluckerPair()
        # Enumerate distinct endpoints in cyclic CCW order from p.
        all_endpts = a_endpts | b_endpts
        order = []
        seen = set()
        x = p
        for _ in range(H):
            if x in all_endpts and x not in seen:
                label = set()
                if x in a_endpts: label.add('a')
                if x in b_endpts: label.add('b')
                order.append((x, frozenset(label)))
                seen.add(x)
            x = (x + 1) % H
        n = len(order)
        arcs = [((order[(i+1) % n][0] - order[i][0]) % H) for i in range(n)]
        odd_indices = [i for i in range(n) if arcs[i] % 2 == 1]
        if len(odd_indices) != 1:
            return 0
        idx = odd_indices[0]
        l0 = order[idx][1]
        l1 = order[(idx + 1) % n][1]
        only_a = frozenset({'a'})
        only_b = frozenset({'b'})
        if l0 == only_a and l1 == only_b:
            return -2
        if l0 == only_b and l1 == only_a:
            return +2
        return 0

    def forward_q_coeff(self, la: tuple[int, int], lb: tuple[int, int]) -> int:
        """For q-commuting `(la, lb)`, the integer c such that
        `L_la · L_lb = q^c · X[γ_la + γ_lb]`  (BPS X-basis convention).
        Equals the q-exponent of the single canonical-basis term in
        the forward product."""
        ga, gb = self.charge(*la), self.charge(*lb)
        if ga == gb:
            return 0
        prod = self.A.multiply(ga, gb)
        if len(prod.terms) != 1:
            raise ValueError(f"forward_q_coeff: {la} and {lb} do not q-commute")
        (_, coef), = prod.terms.items()
        if len(coef._coeffs) != 1:
            raise ValueError(f"forward_q_coeff: non-monomial coefficient {coef}")
        return next(iter(coef._coeffs))

    # -- Base-table extractor ------------------------------------------------

    def base_table(self) -> dict[tuple[int, int, int], list]:
        """Per-k base product table for the standalone `A1A2kKAlg`.

        Returns a dict `{(k_a, k_b, d): [(kind, q_exp), ...]}` for
        every `k_a, k_b ∈ {1, ..., k}` and `d ∈ Z/(2k+3)`, where each
        `kind` is one of

            ('I',)                                  -- identity
            ('L', (k, i))                           -- single named generator
            ('X', ((k1, i1), (k2, i2)))             -- q-commuting pair
                                                       (= the canonical
                                                       basis element at
                                                       lattice charge
                                                       γ_{k1, i1} + γ_{k2, i2})

        Used for hard-coding the multiplication reducer in the
        standalone class (`A1A2kKAlg` now computes it in closed form,
        `A1A2k_plucker_closed_form.base_table_predict`).
        """
        table: dict[tuple[int, int, int], list] = {}
        zero = (0,) * self.n
        for k_a in range(1, self.k + 1):
            for k_b in range(1, self.k + 1):
                for d in range(self.cyc):
                    ga = self.charge(k_a, 0)
                    gb = self.charge(k_b, d)
                    prod = self.A.multiply(ga, gb)
                    entries = []
                    for ch, coef in prod.terms.items():
                        ch_t = tuple(ch)
                        if len(coef._coeffs) != 1:
                            raise ValueError(
                                f"base_table: non-monomial coefficient "
                                f"at ({k_a}, {k_b}, {d}): {coef}"
                            )
                        q_exp = next(iter(coef._coeffs))
                        if ch_t == zero:
                            entries.append((('I',), q_exp))
                        elif ch_t in self._L_inv:
                            entries.append((('L', self._L_inv[ch_t]), q_exp))
                        else:
                            # Higher charge -- find an (la, lb) pair whose
                            # forward product gives this charge with unit coef
                            # (so the X-basis convention applies cleanly).
                            decomp = self._find_X_decomp(ch_t)
                            if decomp is None:
                                raise ValueError(
                                    f"base_table: cannot decompose higher charge "
                                    f"{ch_t} at ({k_a}, {k_b}, {d}) as L·L"
                                )
                            entries.append((('X', decomp), q_exp))
                    table[(k_a, k_b, d)] = entries
        return table

    def _find_X_decomp(self, charge: tuple[int, ...]
                       ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """For a higher-weight canonical basis charge γ, find an ordered
        pair ((k_a, j_a), (k_b, j_b)) of named L-labels whose forward
        product is single-term at γ (any q-coefficient -- the BPS X-basis
        convention absorbs it as `X[γ] = q^{-c} · L_a · L_b`).  Returns
        the lex-smallest such pair, or None.  `_quad_lookup` stores
        entries as `(coef, label_a, label_b)` where the labels are
        `(k, i)` named-L pairs (not lattice charges)."""
        pairs = self._quad_lookup.get(tuple(charge), [])
        if not pairs:
            return None
        return min((a, b) for _coef, a, b in pairs)

    # -- Geometric chord-pair classifier (using corrected chord assignment) --

    def chord_class(self, la: tuple[int, int], lb: tuple[int, int]) -> str:
        """Geometric chord-pair class for two named L-labels under the
        corrected (length, shift) assignment.  Returns
        'same' / 'share' / 'cross' / 'disjoint'."""
        if la == lb:
            return "same"
        ep_a = self._chord_endpoints(*la)
        ep_b = self._chord_endpoints(*lb)
        return self.classify(*ep_a, *ep_b) if False else self._classify_ep(ep_a, ep_b)

    def _chord_endpoints(self, i: int, j: int) -> tuple[int, int]:
        a = (j + self.shifts[i]) % self.cyc
        return (a, (a + self.lengths[i]) % self.cyc)

    def _classify_ep(self, ep1: tuple[int, int], ep2: tuple[int, int]) -> str:
        cyc = self.cyc
        if set(ep1) == set(ep2):
            return "same"
        if set(ep1) & set(ep2):
            return "share"
        a, b = ep1
        arc1 = set()
        x = (a + 1) % cyc
        while x != b:
            arc1.add(x); x = (x + 1) % cyc
        arc2 = set()
        x = (b + 1) % cyc
        while x != a:
            arc2.add(x); x = (x + 1) % cyc
        c, d = ep2
        if (c in arc1 and d in arc2) or (c in arc2 and d in arc1):
            return "cross"
        return "disjoint"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("A1A2k wrapper -- helper-for-A1A2kKAlg smoke test")
    print("=" * 70)
    for k in (2, 3):
        print(f"\n[k = {k}]  ({2*k+3}-gon, rank {2*k})")
        A = A1A2k(k)
        print(f"  lengths per orbit: {A.lengths}")
        print(f"  shifts  per orbit: {A.shifts}")
        # Demo L_chord
        print(f"  L_chord round-trip:")
        for a in range(min(3, A.cyc)):
            for b_off in range(1, A.k + 2):
                b = (a + b_off) % A.cyc
                if a == b: continue
                ch = A.L_chord(a, b)
                lbl = A.L_chord_label(ch)
                print(f"     L_chord({a}, {b})  →  charge {ch}   "
                      f"label_back = {lbl}")
        # Sample the base table
        bt = A.base_table()
        n_entries = sum(len(v) for v in bt.values())
        print(f"  base_table extracted: {len(bt)} (k_a, k_b, d) keys, "
              f"{n_entries} canonical-basis terms total")
        # Show a few sample entries
        print(f"  sample base-table entries (k_a, k_b, d) → terms:")
        for key in sorted(bt.keys())[:8]:
            print(f"     {key}: {bt[key]}")

    print()
    print("=" * 70)
    print("Heptagon Plücker check:  L_{1, j} L_{1, j+1} = 1 + q^{-1} L_{2, j}")
    print("=" * 70)
    A = A1A2k(2)
    for j in range(A.cyc):
        s = A.pretty_product((1, j), (1, (j + 1) % A.cyc))
        # expected: "1 + (q^-1) L_{2, j}"
        print(f"   j = {j}:  L_{{1, {j}}} L_{{1, {(j+1)%A.cyc}}}  =  {s}")
