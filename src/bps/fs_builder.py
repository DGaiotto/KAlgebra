"""`F_γ` and `S` built together, from the relation `F_γ·S = X_γ + O(𝖖)` alone.

A third spec-free engine on the BPS side, and the first that does not need `S`
as an input.  `bps_factor_spectrum.py` builds `S` from its leading data by placing
palindromic BPS factors; `bps_kalgebra_internals.solve_F_via_s_coefficient`
builds `F_γ` by peeling `[n]_𝖖` corrections **against an `S` it is handed**.
This module runs the two recursions as ONE, so neither object is an input to the
other: at each cone degree it places the BPS factors that `S`'s leading data
forces, then reads off the palindromic `F`-coefficients that
`F_γ·S = X_γ + O(𝖖)` forces, and moves up a degree.

THE CONSTRUCTION

    "The crystalline S builder suggests a crystalline builder for F_γ S which
     does not rely on S but rather directly solves for FS = X_γ + O(𝖖) by adding
     at each steps either E^{(s)}_𝖖(X_γ') factors to S or (palindromic)
     X_{γ+γ'} summands to F."

Both moves live in ONE alphabet.  A BPS factor is fixed by a palindromic
multiplicity `Ω = Σ_s a_s χ_s`, `χ_s = 𝖖^{−2s} + ⋯ + 𝖖^{2s}`; an `F`-coefficient
is a palindromic Laurent polynomial, which the shipped solver peels in the
`[n]_𝖖` basis.  These are the same `Z`-basis of the palindromic Laurent
polynomials `P`:

    [n]_𝖖 = 𝖖^{−(n−1)} + ⋯ + 𝖖^{n−1} = χ_{(n−1)/2} .

So the two moves differ not in what they add but in **where** it lands: a factor
factor at `γ'` enters through `S`, hence reaches `F·S` shifted by the quantum-
torus bracket and smeared by the `E_𝖖` tail `L = 𝖖/(1−𝖖²)`; an `F`-summand at
`γ+γ'` enters against `S_0 = 1`, hence bare and unshifted.

WHY THEY ARE THE SAME ALPHABET — THE ENLARGED QUIVER.  Adjoin to the BPS quiver one node at `γ + γ_0`, with `γ_0` a **pure
flavour** direction, so `X_{γ_0}` is central.  Expanding the enlarged quiver's
spectrum generator in powers of `X_{γ_0}`:

    S_new  =  S  −  𝖖/(1−𝖖²) · X_{γ_0} · F_γ S  +  O(X_{2γ_0}) .

So `F_γ S` is the `X_{γ_0}`-LINEAR SECTOR of an ordinary `S` build, and this
module's recursion is that build linearised — one construction, not two
interleaved ones.  `𝖖/(1−𝖖²) = −c_1` is the BPS factor's own tail, i.e. the
leading data `−1` at the new node, which is also why `F_γ`'s coefficient at `γ`
comes out `1` rather than being imposed.

⚠ **Hence `F_γ·S` is one PRESENTATION of that sector, not the only one.**  The
linear sector is a sum of terms with the new `X` monomials inserted at *various
positions* between the old `E_𝖖` factors — one position per BPS factor of the
enlarged quiver — and `F_γ·S`, with `F_γ` collected on the far left, is the
special case where every insertion is pushed leftmost.  Order-independence of
`S_new` is what makes those forms equal as elements.  Measured: the identity
above holds under every placement order tried, while `Ω` at `γ+γ_0+δ` equals
`F_γ`'s coefficient `c_δ`
**only** in the leftmost placement — same element, different multiplicities.
`F_from_enlarged_quiver` builds `F_γ` by that route, using the `S` engine alone.

WHY THE RECURSION CLOSES.  Write `F = Σ_{k} c_k X_{γ+k}` over the positive cone
in node coordinates (`c_0 = 1`) and `S_k` for `S`'s coefficient at `X_k`.  The
quantum torus gives

    [F·S]_{γ+k}  =  Σ_{k' ≤ k}  c_{k'} · 𝖖^{⟨γ+k', k−k'⟩} · S_{k−k'} .

At cone degree `d = deg k` exactly two unknowns of that degree appear: `c_k`, at
`k' = k` where `S_0 = 1`; and `Ω_k`, at `k' = 0` through `S_k`.  Every other term
has both factors of strictly smaller degree, hence is already fixed — the same
degree induction that makes the `S` recursion well-founded, and the reason the
two can be interleaved rather than sequenced.

WHAT PINS WHAT — and the freedom, stated plainly.  `F·S = X_γ + O(𝖖)` constrains
only the `𝖖^{≤0}` part, and a palindromic Laurent polynomial is **determined by
its non-positive part**.  So for ANY choice of `Ω_k` there is a unique `c_k`
completing the relation: the relation alone pins neither object (take `Ω ≡ 0` and
get `S = 1`, `F = X_γ`, a perfectly good solution of it).  What removes the
freedom is the one extra input the crystalline `S` builder already takes — `S`'s
leading data `t` (`−1` on the node charges, `0` elsewhere).  With it:

    S's leading data at k        forces  Ω_k       (`BPSFactorSpectrum._forced`)
    F·S = X_γ + O(𝖖) at γ+k      forces  c_k       (palindromic completion of −[𝖖^{≤0}])

and the build is deterministic.  `omega_policy=` exposes the freedom for anyone
who wants to search it instead (the BFS-over-orderings idea is one axis;
this is the other — how much of each residual to route through which side).

WHAT IS ENFORCED AND WHAT IS EVIDENCE.  `verify_fs_leading` re-checks the very
relation `c_k` was solved from, so it is a **regression guard on the arithmetic,
never evidence** — the same status established for
`BPSFactorSpectrum.verify_leading_data`, and for the same reason.  What IS evidence,
all of it emergent:

* `verify_against_solve_F` — agreement with `BPSKAlgebra.F_qn(γ)`, which solves
  against a Nahm-expanded `S` through a code path this module shares nothing with
  beyond exact `HabiroElement` arithmetic;
* **the support**: this builder never sees the doubly-tropical interval
  `[γ, −σ⁻¹(γ)]` that the shipped solver enumerates over, so `c_k = 0` outside it
  is an independent output.  `support_beyond_interval` reports it;
* **the enlarged-quiver route**: `F_from_enlarged_quiver` reaches the same `F_γ`
  through `BPSFactorSpectrum` alone, with no `F` recursion anywhere in it.

EXACTNESS.  `F`-coefficients at cone degree `≤ D` are EXACT, not truncated: they
read only `S_ε` with `deg ε ≤ D` and `c_{k'}` with `deg k' ≤ D`, all final by the
time they are read.  As in `bps_factor_spectrum.py`, the only truncation is along the
positive cone; never in `𝖖`.

Run:  PYTHONPATH=. python fs_builder.py
"""

from __future__ import annotations

from typing import Callable, Sequence

from habiro import HabiroElement
from laurent_poly import LaurentPoly
from q_number_poly import QNumberPoly
from bps_factor_spectrum import BPSFactorSpectrum, expansion, q_pow, spin_decompose

Vec = tuple[int, ...]

H1 = HabiroElement.one()


# --------------------------------------------------------------------------
# the F-side move
# --------------------------------------------------------------------------

def palindromic_completion(nonpositive: dict[int, int]) -> dict[int, int]:
    """The unique palindromic Laurent polynomial with this non-positive part.

    `d_k = d_{−k}` makes the non-positive half a complete set of coordinates, so
    there is nothing to solve: mirror the strictly negative exponents and keep
    `d_0`.  This is the `F`-side move of the joint recursion, and it is the same
    step the shipped solver takes in the `[n]_𝖖` basis — peeling the lowest
    non-positive exponent `k` with `[1−k]_𝖖` is exactly mirroring `d_k` to
    `d_{−k}`, one basis vector at a time.
    """
    out = dict(nonpositive)
    for e, c in nonpositive.items():
        if e < 0:
            out[-e] = out.get(-e, 0) + c
    return {e: c for e, c in out.items() if c}


class FSBuilder:
    """`F_γ` and `S` for one BPS quiver, one charge, one cone cutoff.

    Parameters
    ----------
    pairing, node_charges, cutoff
        As :class:`bps_factor_spectrum.BPSFactorSpectrum` — the lattice, the cone generators,
        and the cone-degree truncation `D`.  Node charges must be linearly
        independent (the recursion indexes the cone by node coordinates).
    gamma
        The charge whose canonical element is built.  Any lattice charge; it does
        **not** have to lie in the positive cone (`F`'s support is `γ` plus the
        cone, and that is how it is enumerated).
    leading_data
        `S`'s leading data, `charge -> int`.  Default `−1` on the node charges and
        `0` elsewhere — the BPS spectrum generator.  **This is the input that
        removes the freedom** described in the module docstring; without it the
        relation `F·S = X_γ + O(𝖖)` has a solution for every `Ω`.
    order, phases, order_key, seed
        Placement of the BPS factors, passed straight through.  `S` is
        order-independent, so these move the *content* and the cost; whether
        `F` is likewise order-independent is a question this class makes
        measurable rather than assumes.
    omega_policy
        Optional `(k, forced_omega, fs_residual) -> omega`, consulted at each
        cone charge (`k` in node coordinates) with the `Ω` that `S`'s leading
        data forces there and the `F·S` residual accumulated so far.  Returning
        `forced_omega` reproduces the deterministic build exactly; returning any
        other palindromic `Ω` still yields a valid `F·S = X_γ + O(𝖖)` pair, with
        `S` no longer carrying the prescribed leading data.

        The seam exists because the construction's "either/or" is a genuine
        choice — see the module docstring — and a builder that hid it would be
        claiming a uniqueness the relation does not have.  It needs a
        **degree-compatible** order (`"lex"` or `"degree-key"`), because a policy
        that reads the running residual has to see the factors placed in the
        order they are decided; the insert-based orders rebuild prefixes and do
        not offer that.  Supplying it with any other order raises.
    """

    def __init__(
        self,
        pairing: Sequence[Sequence[int]],
        node_charges: Sequence[Sequence[int]],
        gamma: Sequence[int],
        cutoff: int,
        *,
        leading_data: Callable[[Vec], int] | None = None,
        order: str | None = None,
        phases: Sequence[complex] | None = None,
        order_key: Callable[[Vec], object] | None = None,
        seed: int = 20260813,
        omega_policy: Callable[[Vec, dict, HabiroElement], dict] | None = None,
    ):
        self.factors = BPSFactorSpectrum(
            pairing, node_charges, cutoff,
            leading_data=leading_data, order=order, phases=phases,
            order_key=order_key, seed=seed,
        )
        self.gamma: Vec = tuple(int(x) for x in gamma)
        if len(self.gamma) != self.factors.dim:
            raise ValueError(
                f"gamma has {len(self.gamma)} coordinates but the lattice has "
                f"{self.factors.dim}")
        self._policy = omega_policy
        if omega_policy is not None and self.factors.order not in (
                "lex", "degree-key"):
            raise ValueError(
                f"omega_policy needs a degree-compatible order ('lex' or "
                f"'degree-key'), got '{self.factors.order}': a policy that reads "
                f"the running residual must see the BPS factors placed in the "
                f"order they are decided, and the insert-based orders rebuild "
                f"cached prefixes instead.")
        self.degree_cap = self.factors.degree_cap
        self.rank = self.factors.rank

        B = [[int(x) for x in row] for row in pairing]
        self._pairing = B
        dim = self.factors.dim
        # gb[j] = ⟨γ, γ_j⟩ — the only lattice-level bracket the F side needs.
        # Everything else is ⟨k', k−k'⟩, which is the node-coordinate bracket
        # BPSFactorSpectrum already assembled.
        self._gb = [
            sum(self.gamma[p] * B[p][r] * self.factors.nodes[j][r]
                for p in range(dim) for r in range(dim))
            for j in range(self.rank)
        ]
        self._bnode = self.factors.node_pairing

        self.zero: Vec = tuple([0] * self.rank)
        # node coords k -> palindromic {exponent: coefficient} of c_k
        self.f: dict[Vec, dict[int, int]] = {}
        # node coords k -> the FINAL [F·S]_{γ+k}, kept for the verifier and for
        # anyone who wants to look at what the relation actually left behind.
        self.fs: dict[Vec, HabiroElement] = {}
        self._S: dict[Vec, HabiroElement] | None = None
        self._by_degree: dict[int, list[Vec]] = {}
        for k in self.factors.cone:
            self._by_degree.setdefault(sum(k), []).append(k)

    # ---- the joint recursion ----------------------------------------------

    def run(self) -> dict[Vec, dict[int, int]]:
        """Build `F_γ` and `S` together.  Returns `F` keyed by node coordinates.

        One pass up the cone.  At degree `d` the `S` engine places every factor
        factor its leading data forces (so `S_ε` becomes final for all
        `deg ε ≤ d`), and only then is the `F`-coefficient at each degree-`d`
        charge read off — which is exactly the interleaving that lets neither
        object be an input to the other.
        """
        self.f = {self.zero: {0: 1}}
        self.fs = {self.zero: H1}
        self._S = {self.zero: H1}
        if self._policy is None:
            self.factors.run(after_degree=self._after_degree)
        else:
            self._run_with_policy()
        return self.f

    def _run_with_policy(self) -> None:
        """The same recursion, with the `Ω` choice handed to `omega_policy`.

        Kept separate from the delegated path rather than folded into it: the
        delegated path IS `BPSFactorSpectrum`'s own build, so `S` is bit-identical to
        what that engine produces alone, and that identity is worth not putting
        behind a branch.  Here the loop is owned locally because the policy has
        to be consulted between forcing `Ω` and placing the factor — but the
        arithmetic is still the `S` engine's (`forced_multiplicity`,
        `degree_order`, `factor_multiply`), so the two paths cannot drift.

        In this mode the `BPSFactorSpectrum` instance is used for its arithmetic and
        its `omega` record only; read `S` off `FSBuilder.spectrum_generator`.
        """
        acc: dict[Vec, HabiroElement] = {self.zero: H1}
        acc_deg: dict[Vec, int] = {self.zero: 0}
        self.factors.omega = {}
        for d in range(1, self.degree_cap + 1):
            for k in self.factors.degree_order(d):
                coeff = acc.get(k)
                forced = BPSFactorSpectrum._forced(
                    expansion(coeff) if coeff is not None else {},
                    self.factors.target(k))
                om = self._policy(k, dict(forced), self._fs_residual(k, acc))
                om = {e: c for e, c in (om or {}).items() if c}
                if any(om.get(e, 0) != om.get(-e, 0) for e in list(om)):
                    raise ValueError(
                        f"omega_policy returned a non-palindromic Ω at {k}: "
                        f"{sorted(om.items())}.  The BPS factor is defined by a "
                        f"palindromic multiplicity; a non-palindromic one names "
                        f"no factor.")
                if om:
                    self.factors.omega[k] = om
                    acc, acc_deg = self.factors.factor_multiply(acc, acc_deg, k, d, om)
            self._after_degree(d, acc)

    def _after_degree(self, degree: int, partial: dict) -> None:
        """Read every `F`-coefficient of this cone degree off the partial `S`.

        `partial[ε]` is final for `deg ε ≤ degree` (a BPS factor of degree
        `d' > degree` first contributes at `d'`), and every term of
        `[F·S]_{γ+k}` at `deg k = degree` uses `S_ε` with `deg ε ≤ degree` and
        `c_{k'}` with `deg k' ≤ degree` — the latter at `deg k' = degree` only
        for `k' = k`, against `S_0 = 1`.  So the readout is exact, not
        provisional.
        """
        for k in self._by_degree.get(degree, ()):
            resid = self._fs_residual(k, partial)
            nonpos = {e: c for e, c in expansion(resid).items() if e <= 0 and c}
            c_k = palindromic_completion({e: -v for e, v in nonpos.items()})
            if c_k:
                self.f[k] = c_k
            self.fs[k] = resid + HabiroElement(LaurentPoly(c_k), {})
        self._S = partial

    def _fs_residual(self, k: Vec, partial: dict) -> HabiroElement:
        """`[F·S]_{γ+k}` from everything fixed so far — i.e. omitting `c_k`.

        `Σ_{k' < k} c_{k'} · 𝖖^{⟨γ+k', k−k'⟩} · S_{k−k'}`, with the bracket split
        as `⟨γ, k−k'⟩ + ⟨k', k−k'⟩` so the lattice enters only through the
        precomputed `⟨γ, γ_j⟩`.
        """
        terms: list[HabiroElement] = []
        for kp, c_kp in self.f.items():
            diff = tuple(a - b for a, b in zip(k, kp))
            if any(x < 0 for x in diff):
                continue
            if not any(diff):
                continue                      # that is the c_k term itself
            s_el = partial.get(diff)
            if s_el is None or s_el.is_zero():
                continue
            twist = sum(diff[j] * self._gb[j] for j in range(self.rank))
            for i, ki in enumerate(kp):
                if ki:
                    twist += ki * sum(self._bnode[i][j] * diff[j]
                                      for j in range(self.rank))
            num = LaurentPoly(c_kp) * s_el.numerator
            if twist:
                num = num * LaurentPoly({twist: 1})
            terms.append(HabiroElement(num, dict(s_el.denom)))
        return HabiroElement.sum(terms) if terms else HabiroElement.zero()

    # ---- output ------------------------------------------------------------

    def charge(self, k: Vec) -> Vec:
        """Node coordinates `k` -> the lattice charge `γ + Σ k_i γ_i`."""
        return tuple(g + c for g, c in zip(self.gamma, self.factors.charge(k)))

    def F(self) -> dict[Vec, LaurentPoly]:
        """`F_γ` keyed by **lattice charges**, palindromic coefficients.

        The same shape as `BPSKAlgebra.F(a)`, so the two are directly
        comparable — which is the point (`verify_against_solve_F`).
        """
        if self._S is None:
            self.run()
        return {self.charge(k): LaurentPoly(c) for k, c in self.f.items()}

    def F_qn(self) -> dict[Vec, QNumberPoly]:
        """`F_γ` keyed by lattice charges, in the native `[n]_𝖖` basis.

        Mirrors `BPSKAlgebra.F_qn(a)`.  The conversion cannot fail on a correct
        build: every `c_k` is palindromic by construction, and the palindromic
        Laurent polynomials are exactly the integral span of the `[n]_𝖖`.
        """
        if self._S is None:
            self.run()
        return {self.charge(k): QNumberPoly.from_palindromic_laurent(
            LaurentPoly(c)) for k, c in self.f.items()}

    def spectrum_generator(self) -> dict[Vec, HabiroElement]:
        """`S` keyed by lattice charges — the by-product of the joint build.

        Identical to what `BPSFactorSpectrum` alone would produce from the same
        leading data and order: the `F` side reads the partial products but
        never writes to them.
        """
        if self._S is None:
            self.run()
        return {self.factors.charge(k): c for k, c in self._S.items()}

    def multiplicities(self) -> dict[Vec, dict[int, int]]:
        """`Ω` keyed by lattice charges — which BPS factors the build placed."""
        if self._S is None:
            self.run()
        return {self.factors.charge(k): dict(om)
                for k, om in self.factors.omega.items()}

    def moves(self) -> list[tuple[Vec, str, dict[int, int]]]:
        """Every move the build made, as `(lattice charge, side, palindromic)`.

        `side` is `"S"` for a BPS factor `E^{(s)}_𝖖(X_{γ'})` placed at `γ'` and
        `"F"` for a summand added at `γ+γ'`; the third entry is the palindromic
        Laurent polynomial in the `{exponent: coefficient}` form both sides
        share.  `spin_decompose` turns either into `{2s: a_s}`.

        This is the construction's own trace, in the terms the construction is
        stated in — useful for seeing at which charges the two sides actually
        carry the residual, which is the question `omega_policy` reopens.
        """
        if self._S is None:
            self.run()
        out: list[tuple[Vec, str, dict[int, int]]] = []
        for k in self.factors.cone:
            om = self.factors.omega.get(k)
            if om:
                out.append((self.factors.charge(k), "S", dict(om)))
            c = self.f.get(k)
            if c and any(k):
                out.append((self.charge(k), "F", dict(c)))
        return out

    # ---- checks ------------------------------------------------------------

    def verify_fs_leading(self) -> list[tuple[Vec, str]]:
        """`F·S == X_γ + O(𝖖)` on the built cone.  Returns violations.

        ⚠ **A REGRESSION GUARD, NOT EVIDENCE.**  `palindromic_completion` chooses
        `c_k` precisely so this holds, integrally and palindromically, so on a
        correct engine it cannot fail — the status established for
        `BPSFactorSpectrum.verify_leading_data`, for the same reason.
        Never cite it as support for the construction; use
        `verify_against_solve_F` and `support_beyond_interval`.

        **It is recomputed, not read back.**  The first version of this method
        checked the residual buffer the build had already accumulated, and was
        therefore vacuous in the strong sense: a corrupted quantum-torus bracket
        entered `c_k` and the stored residual identically and cancelled, so the
        negative control reported a pass on a broken engine.  It now
        re-multiplies the OUTPUT `F` against the OUTPUT `S` on
        **lattice charges through the raw pairing matrix**, a different code path
        from the node-coordinate brackets (`_gb`, `_bnode`) the build runs on —
        so a wrong bracket there is now caught, and the control measures that.

        What it still cannot see, and no arithmetic guard here could: whether
        `S`'s leading data was the right input at all.  Any `Ω` admits a
        completing `F` (module docstring), and every such pair passes this.
        """
        if self._S is None:
            self.run()
        B = self._pairing
        dim = self.factors.dim
        F = self.F()
        S = self.spectrum_generator()
        acc: dict[Vec, list[HabiroElement]] = {}
        for eta, lp in F.items():
            for mu, s_el in S.items():
                if s_el.is_zero():
                    continue
                twist = sum(eta[p] * B[p][r] * mu[r]
                            for p in range(dim) for r in range(dim))
                num = lp * s_el.numerator
                if twist:
                    num = num * LaurentPoly({twist: 1})
                total = tuple(a + b for a, b in zip(eta, mu))
                acc.setdefault(total, []).append(
                    HabiroElement(num, dict(s_el.denom)))
        bad: list[tuple[Vec, str]] = []
        for total, terms in sorted(acc.items()):
            k = self._node_coords(total)
            if k is None or sum(k) > self.degree_cap:
                continue                  # outside the built cone: not checkable
            f = expansion(HabiroElement.sum(terms))
            want = 1 if total == self.gamma else 0
            if any(e < 0 and c for e, c in f.items()):
                bad.append((total, "negative power of q"))
            elif f.get(0, 0) != want:
                bad.append((total, f"q^0 == {f.get(0, 0)}, want {want}"))
        return bad

    def support_beyond_interval(self, upper: Sequence[int]) -> list[Vec]:
        """Charges where this build put an `F`-coefficient outside `[γ, upper]`.

        `upper` is the upper tropical charge `γ⁺ = −σ⁻¹(γ)` —
        `BPSKAlgebra.gamma_upper(γ)`.

        EMERGENT, and the sharpest single check available here: the joint
        recursion walks the whole cone and knows nothing of the doubly-tropical
        interval, which is the window the shipped solver *enumerates over*.  So
        an empty list says the relation alone reproduced a support the other
        engine is handed.  A non-empty one is a finding, not automatically a bug —
        it is exactly where the two constructions would part company.
        """
        if self._S is None:
            self.run()
        from lattice import make_cone_predicate
        in_cone = make_cone_predicate([tuple(g) for g in self.factors.nodes])
        upper = tuple(int(x) for x in upper)
        out = []
        for k in self.f:
            eta = self.charge(k)
            if not in_cone(tuple(u - e for u, e in zip(upper, eta))):
                out.append(eta)
        return out

    def cutoff_needed(self, bps) -> int:
        """The smallest cone cutoff that covers `bps.F_qn(γ)`'s whole support.

        `verify_against_solve_F` refuses to pass below this, so a comparison can
        never be quietly decided by a cutoff that hid the disagreement — the
        "no silent caps" rule.  `-1` if some shipped charge is not `γ` plus a
        cone element at all, which would be a genuine finding rather than a
        window problem.
        """
        worst = 0
        for eta in bps.F_qn(self.gamma):
            k = self._node_coords(tuple(eta))
            if k is None:
                return -1
            worst = max(worst, sum(k))
        return worst

    def verify_against_solve_F(self, bps) -> list[tuple[Vec, str]]:
        """`F_γ` here `==` `bps.F_qn(γ)`.  Returns disagreements (empty = pass).

        EMERGENT, and the load-bearing check.  The right side solves against
        `bps._s_coefficient` — a Nahm-sum expansion of the spec in spec mode —
        and enumerates the doubly-tropical interval; the left side never forms a
        spec, never enumerates that interval, and grows `S` as it goes.  Nothing
        is shared but exact `HabiroElement` arithmetic, so agreement is evidence.

        **A cutoff too small to hold the shipped support is reported, not
        tolerated.**  Truncation and disagreement are different findings and the
        one must never be able to masquerade as the other: if some shipped charge
        lies beyond cone degree `D`, that is the first entry of the returned list
        and the comparison is not a pass.  Raise the cutoff to
        `cutoff_needed(bps)` and re-run.
        """
        if self._S is None:
            self.run()
        mine = self.F()
        theirs = {tuple(g): qn.to_laurent()
                  for g, qn in bps.F_qn(self.gamma).items()}
        bad: list[tuple[Vec, str]] = []
        for eta, lp in sorted(theirs.items()):
            k = self._node_coords(eta)
            if k is None:
                bad.append((eta, "shipped F charge is not γ + (positive cone)"))
                continue
            if sum(k) > self.degree_cap:
                bad.append((eta, f"beyond cone degree {self.degree_cap} "
                                 f"(needs {sum(k)}) — raise the cutoff"))
                continue
            if mine.get(eta, LaurentPoly({})) != lp:
                bad.append((eta,
                            f"got {mine.get(eta, LaurentPoly({}))}, want {lp}"))
        for eta in sorted(mine):
            if eta not in theirs:
                bad.append((eta, f"extra: {mine[eta]}"))
        return bad

    def _node_coords(self, eta: Vec) -> Vec | None:
        """Lattice charge -> node coordinates of `η − γ`, or `None` if outside.

        Solved by exact rational elimination against the node basis; the nodes
        are linearly independent (checked by `BPSFactorSpectrum`), so the solution is
        unique when it exists.
        """
        from fractions import Fraction
        rhs = [Fraction(e - g) for e, g in zip(eta, self.gamma)]
        cols = [[Fraction(n[d]) for n in self.factors.nodes]
                for d in range(self.factors.dim)]
        rows = [cols[d] + [rhs[d]] for d in range(self.factors.dim)]
        piv = []
        r = 0
        for c in range(self.rank):
            p = next((i for i in range(r, len(rows)) if rows[i][c]), None)
            if p is None:
                continue
            rows[r], rows[p] = rows[p], rows[r]
            lead = rows[r][c]
            rows[r] = [x / lead for x in rows[r]]
            for i in range(len(rows)):
                if i != r and rows[i][c]:
                    fac = rows[i][c]
                    rows[i] = [a - fac * b for a, b in zip(rows[i], rows[r])]
            piv.append(c)
            r += 1
        for i in range(r, len(rows)):
            if rows[i][self.rank]:
                return None                    # inconsistent: not in the span
        sol = [Fraction(0)] * self.rank
        for i, c in enumerate(piv):
            sol[c] = rows[i][self.rank]
        if any(x.denominator != 1 or x < 0 for x in sol):
            return None
        return tuple(int(x) for x in sol)


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def build_F_and_S(
    pairing: Sequence[Sequence[int]],
    node_charges: Sequence[Sequence[int]],
    gamma: Sequence[int],
    cutoff: int,
    *,
    leading_data: Callable[[Vec], int] | None = None,
    order: str | None = None,
    phases: Sequence[complex] | None = None,
):
    """`(F_γ, S)` from `F_γ·S = X_γ + O(𝖖)` plus `S`'s leading data.

    `F_γ` as `{lattice charge: LaurentPoly}` (palindromic, `1` at `γ`), `S` as
    `{lattice charge: HabiroElement}` — the same shapes `BPSKAlgebra.F` and
    `bps_factor_spectrum.build_spectrum_generator_from_factors` return.

    No spec, no green sequence, no `S` input, and no `F`-support window: the only
    inputs are the quiver, the charge, the cone cutoff, and the leading data.
    """
    b = FSBuilder(pairing, node_charges, gamma, cutoff,
              leading_data=leading_data, order=order, phases=phases)
    b.run()
    return b.F(), b.spectrum_generator()


def enlarged_quiver(pairing: Sequence[Sequence[int]],
                    node_charges: Sequence[Sequence[int]],
                    gamma: Sequence[int]):
    """The BPS quiver with one node added at `γ + γ_0`, `γ_0` pure flavour.

    Returns `(pairing', node_charges')` on `Γ' = Γ ⊕ Z·γ_0`.  `γ_0` is adjoined
    as a **pure flavour** direction — the new row and column of the pairing are
    zero — so `X_{γ_0}` is central and `X_{γ_0}X_μ = X_{γ_0+μ}` with no phase.
    The new node's own pairing against the old ones is `⟨γ+γ_0, γ_i⟩ = ⟨γ, γ_i⟩`,
    which is the same vector the joint recursion carries as `_gb`.
    """
    dim = len(pairing)
    B = [[int(x) for x in row] + [0] for row in pairing] + [[0] * (dim + 1)]
    nodes = ([tuple(int(x) for x in g) + (0,) for g in node_charges]
             + [tuple(int(x) for x in gamma) + (1,)])
    return B, nodes


def _new_node_leftmost(k):
    """Order key placing every factor that carries `γ_0` to the LEFT.

    Smaller key = further left in the product, and `k[-1]` is the new node's
    coordinate.  This is the placement in which the `X_{γ_0}`-linear sector reads
    as `F_γ·S` with `F_γ` on the far left — see `F_from_enlarged_quiver`.
    """
    return (0 if k[-1] >= 1 else 1, k)


def F_from_enlarged_quiver(
    pairing: Sequence[Sequence[int]],
    node_charges: Sequence[Sequence[int]],
    gamma: Sequence[int],
    cutoff: int,
    *,
    order_key: Callable[[Vec], object] | None = None,
) -> dict[Vec, LaurentPoly]:
    """`F_γ` read off the factor multiplicities of the enlarged quiver.

    THE CONSTRUCTION.  Adjoin a node at `γ + γ_0` with `γ_0`
    pure flavour.  Expanding the enlarged quiver's spectrum generator in powers
    of the central `X_{γ_0}`,

        S_new  =  S  −  𝖖/(1−𝖖²) · X_{γ_0} · F_γ S  +  O(X_{2γ_0}) ,

    so `F_γ S` is the `X_{γ_0}`-linear sector of an ordinary `S` build and the
    joint recursion in `FSBuilder` is that build linearised.  `𝖖/(1−𝖖²)` is `−c_1`,
    the BPS factor's own tail — i.e. exactly the leading data `−1` at the new
    node, which is also why `F_γ`'s coefficient at `γ` comes out `1`.

    **This route needs no `F` recursion at all** — only `BPSFactorSpectrum`.  It is
    therefore an independent construction of `F_γ`, and its agreement with
    `FSBuilder` and with `BPSKAlgebra.F_qn` is a three-way cross-check between
    engines that share only exact `HabiroElement` arithmetic.  It costs more
    than `FSBuilder` (a rank-`r+1` cone to degree `cutoff+1`, and it builds the
    `O(X_{2γ_0})` sectors the linearised recursion never forms), so `FSBuilder` stays
    the working route and this one is the check.

    **Placement matters here, and that is the point.**  The linear sector is a
    sum of terms with the new `X` monomials inserted at *various positions*
    between the old `E_𝖖` factors; `F_γ·S`, with `F_γ` on the left, is the
    special case where every insertion is pushed leftmost.  So `Ω` at the charges
    `γ+γ_0+δ` equals `F_γ`'s coefficient `c_δ` **only in that placement**, which
    is the default here.  Under another order the *element* is unchanged (the
    identity above still holds) while the multiplicities are different
    numbers — measured.
    """
    B, nodes = enlarged_quiver(pairing, node_charges, gamma)
    factors = BPSFactorSpectrum(B, nodes, int(cutoff) + 1, order="key",
                       order_key=order_key or _new_node_leftmost)
    factors.run()
    gamma = tuple(int(x) for x in gamma)
    out: dict[Vec, LaurentPoly] = {}
    for charge, om in factors.multiplicities().items():
        if charge[-1] != 1 or not om:
            continue                      # γ_0-degree 0 or ≥ 2: not this sector
        # charge = (γ + δ, 1) in Γ' ⇒ the F-coefficient sits at γ + δ in Γ.
        out[tuple(charge[:-1])] = LaurentPoly(dict(om))
    return out


def fs_linear_sector(
    pairing: Sequence[Sequence[int]],
    node_charges: Sequence[Sequence[int]],
    gamma: Sequence[int],
    cutoff: int,
    *,
    piece_key: Callable[[Vec, int], object] | None = None,
) -> dict[Vec, HabiroElement]:
    """`F_γ S` with the summands placed WHEREVER `piece_key` says.

    THE GENERAL PRESENTATION.  `F_γ·S` collects every summand
    on the far left; in general the object is

        Σ over pieces  ( ∏ E's before ) · ω(s,γ')·χ_s·X_{γ+γ'} · ( ∏ E's after )

    — one placement per `(s, γ')` summand, independently of the placements of the
    `E^{(s)}_𝖖(X_{γ'})^{Ω(s,γ')}` factors themselves.  Both freedoms are the SAME
    mechanism seen in the enlarged quiver of `F_from_enlarged_quiver`: there a
    summand at `γ+γ'` **is** a spin piece of the BPS factor at `γ+γ_0+γ'`, so a
    single per-`(s, charge)` placement rule covers both sides at once.  That is
    what this function uses (`BPSFactorSpectrum(piece_key=…)`), and it is why no second
    placement mechanism is needed for the `F` side.

    `piece_key(γ, 2s) -> sortable` — main's contract signature — takes the
    enlarged quiver's node coordinates, whose LAST entry is the `γ_0`-degree: `1` marks an
    `F` summand, `0` an `S` factor.  Smaller key = further left.  Default: every
    `F` summand leftmost — the placement in which the sum IS the product `F_γ·S`.

    Returns the sector keyed by charges of `Γ` (i.e. `γ + γ'`), normalised by
    dividing out the new node's own tail: the raw `X_{γ_0}`-linear coefficient is
    `c_1` times this, `c_1 = −𝖖/(1−𝖖²)`.

    **The element must not depend on `piece_key`** — that is the `F`-side
    counterpart of the `S`-side order-independence the whole construction rests
    on, and it is measured rather than assumed.  What DOES depend on it is
    the split into summands: the individual `ω(s,γ')` are placement-dependent,
    and
    only the leftmost placement makes them `F_γ`'s coefficients.
    """
    B, nodes = enlarged_quiver(pairing, node_charges, gamma)
    if piece_key is None:
        def piece_key(k, two_s):          # noqa: E306 — every F summand leftmost
            return 0 if k[-1] >= 1 else 1
    factors = BPSFactorSpectrum(B, nodes, int(cutoff) + 1, piece_key=piece_key)
    factors.run()
    gamma = tuple(int(x) for x in gamma)
    c1 = _c1()
    out: dict[Vec, HabiroElement] = {}
    for charge, coeff in factors.spectrum_generator().items():
        if charge[-1] != 1 or coeff.is_zero():
            continue
        # `coeff = c_1 · [F_γ S]_{charge[:-1]}`; divide the scalar back out.
        out[tuple(charge[:-1])] = _divide_by_c1(coeff, c1)
    return out


def _c1() -> HabiroElement:
    from recursive_spectrum import c_n
    return c_n(1)


def _divide_by_c1(x: HabiroElement, c1: HabiroElement) -> HabiroElement:
    """`x / c_1` for an `x` known to be divisible by it.

    `c_1 = −𝖖/(1−𝖖²)`, so dividing is multiplying by `−𝖖⁻¹(1−𝖖²)` — a Laurent
    polynomial, hence exact and inside `HabiroElement`'s own arithmetic; no
    series division is involved and nothing is truncated.
    """
    return x * HabiroElement(
        LaurentPoly({-1: -1, 1: 1}), {})


def fs_closed_form(
    pairing: Sequence[Sequence[int]],
    node_charges: Sequence[Sequence[int]],
    gamma: Sequence[int],
    cutoff: int,
):
    """`F_γ S = E_1 · X_γ · E_2` — no recursion, no solve.  `None` if unavailable.

    THE HYPOTHESIS THIS ANSWERS: *"letting the order of factors
    in `S` depend on the choice of `γ` may make the FS builder better than
    building `S` and finding `F` in the usual way"*.

    It does, and by more than a constant.  The joint builder never needs a
    *canonical* `S` — `S` is order-independent, so the order is free and may be
    chosen **per `γ`**, which the build-`S`-once-then-solve route cannot do.
    Choose it on the enlarged quiver of `enlarged_quiver` (the node at `γ + γ_0`),
    which is itself a function of `γ`.  Where that quiver is **acyclic** the
    repo's own mantle theorem applies: the strip order factors its `S` into one
    `E_𝖖(X_ν)` per node and nothing else.  Linearising in `X_{γ_0}` replaces the
    new node's factor by its `−L·X_{γ+γ_0}` term and leaves the others alone:

        F_γ S  =  E_1 · X_γ · E_2 ,      S = E_1 · E_2 ,

    with `E_1`, `E_2` the old nodes' factors before and after the new node's
    position in the strip order — **a position that depends on `γ`** (measured at
    the pentagon: between the two old factors at `γ = (1,1)`, last at
    `γ = (−1,1)`).  Equivalently `F_γ = E_1 X_γ E_1^{-1}`: a conjugate of a bare
    monomial.  Nothing is solved and no `Ω` is forced at any charge.

    Returns `(product, E_1, E_2)` keyed by lattice charges, cone-truncated to
    `γ + {deg ≤ cutoff}`.

    **Honest-fails (`None`) when the enlarged quiver is not acyclic**, which is
    the scope of the closed form and not of the collapse: at the 3-cycle(1,1,1)
    the searched order still collapses `F_γ S` to a single insertion even though
    no spin-0 spec exists, so acyclicity is SUFFICIENT for this route, not
    necessary for the phenomenon (measured).
    """
    from bps_factor_spectrum import acyclic_node_order

    B, np_ = enlarged_quiver(pairing, node_charges, gamma)
    order = acyclic_node_order(B, np_)
    if order is None:
        return None

    gamma = tuple(int(x) for x in gamma)
    rank = len(node_charges)
    pos = order.index(rank)                      # where the γ-factor sits

    # EVERYTHING IN NODE COORDINATES, for the reason `bps_factor_spectrum` gives: the
    # cone is then a simplex, membership is two integer tests, and the brackets
    # are integer dot products.  A first version worked in lattice charges and
    # solved a `Fraction` system per charge to test membership; it was CORRECT
    # and 2–10× SLOWER than the recursion it was meant to beat, which is the
    # whole reason this one exists.
    engine = BPSFactorSpectrum(pairing, node_charges, cutoff, order="lex")
    bnode = engine.node_pairing
    dim = engine.dim
    Braw = [[int(x) for x in row] for row in pairing]
    # gb[j] = ⟨γ, γ_j⟩ — the only lattice-level bracket needed.
    gb = [sum(gamma[p] * Braw[p][r] * engine.nodes[j][r]
              for p in range(dim) for r in range(dim)) for j in range(rank)]

    zero = tuple([0] * rank)

    def e_side(indices):
        """`∏_{i in indices} E_𝖖(X_{γ_i})`, cone-truncated, in node coords."""
        acc = {zero: H1}
        acc_deg = {zero: 0}
        for i in indices:
            k0 = tuple(1 if j == i else 0 for j in range(rank))
            acc, acc_deg = engine.factor_multiply(acc, acc_deg, k0, 1, {0: 1})
        return acc

    left = e_side(order[:pos])
    right = e_side(order[pos + 1:])

    # `[E_1 · X_γ · E_2]_{γ+a+b} = e1[a]·e2[b]·𝖖^{⟨a,b⟩ + Σ_j (b_j − a_j)·gb_j}`,
    # from ⟨a,γ⟩ = −Σ a_j gb_j and ⟨a+γ, b⟩ = ⟨a,b⟩ + Σ b_j gb_j.
    out: dict[Vec, HabiroElement] = {}
    for a, va in left.items():
        da = sum(a)
        ga = sum(a[j] * gb[j] for j in range(rank))
        for b, vb in right.items():
            if da + sum(b) > cutoff:
                continue
            k = tuple(x + y for x, y in zip(a, b))
            twist = (sum(a[i] * bnode[i][j] * b[j]
                         for i in range(rank) for j in range(rank))
                     + sum(b[j] * gb[j] for j in range(rank)) - ga)
            term = va * vb
            if twist:
                term = term * q_pow(twist)
            out[k] = out.get(k, HabiroElement.zero()) + term

    def lattice(k):
        return tuple(g + c for g, c in zip(gamma, engine.charge(k)))

    return ({lattice(k): v for k, v in out.items() if not v.is_zero()},
            {engine.charge(k): v for k, v in left.items()},
            {engine.charge(k): v for k, v in right.items()})


def _main() -> None:
    pentagon = ([[0, 1], [-1, 0]], [(1, 0), (0, 1)])

    print("PENTAGON — F_γ and S built together, no S input\n")
    for gamma in [(1, 0), (0, 1), (1, 1), (-1, 0)]:
        b = FSBuilder(*pentagon, gamma, 6)
        b.run()
        bad = b.verify_fs_leading()
        F = b.F()
        print(f"  γ = {gamma}:  F = "
              + " + ".join(f"({LaurentPoly(c)})·X_{b.charge(k)}"
                           for k, c in sorted(b.f.items()))
              + f"   [FS guard: {'ok' if not bad else bad}]")

    print("\n  the moves, at γ = (1,1) — which side carried each residual:")
    b = FSBuilder(*pentagon, (1, 1), 4)
    b.run()
    for charge, side, pal in b.moves():
        print(f"    {side}  at {charge}:  {sorted(pal.items())}"
              f"   spins {sorted(spin_decompose(pal).items())}")

    print("\n  S built as a by-product == BPSFactorSpectrum's own S: ", end="")
    from bps_factor_spectrum import build_spectrum_generator_from_factors
    S_alone = build_spectrum_generator_from_factors(*pentagon, 4, order="lex")
    b2 = FSBuilder(*pentagon, (1, 1), 4, order="lex")
    b2.run()
    print(b2.spectrum_generator() == S_alone)


if __name__ == "__main__":
    _main()
