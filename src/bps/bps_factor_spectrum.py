"""Spectrum generator `S` from its leading data, via palindromic BPS factors.

A second, independent spec-free `S`-finder, beside the peel recursion in
`recursive_spectrum.py`.  Where that one peels a node, solves `F` against the
sub-quiver's `S` and reattaches `E_𝖖(F)` — and honest-fails on character-charge
(matter) nodes, so N=2\\*/Markov is out of reach — this one never solves an `F`
and never peels.  It **prescribes the leading data and reads the factor
multiplicities off degree by degree**, which works at any BPS quiver.

THE CONSTRUCTION

On the quantum torus `X_γ X_δ = 𝖖^{⟨γ,δ⟩} X_{γ+δ}`, the spin-`s` multiplet at a
charge `γ` carries the sign in BOTH the argument and the exponent,

    m_s(γ) = ∏_{j=−2s, step 2}^{2s} E_𝖖( (−1)^{2s} 𝖖^j X_γ )^{(−1)^{2s}} ,

and the **BPS factor** at `γ` is `A_γ(Ω) = ∏_s m_s(γ)^{a_s}` for
`Ω = Σ_s a_s χ_s`, `χ_s = 𝖖^{−2s} + ⋯ + 𝖖^{2s}`.  Since the `χ_s` are a
`Z`-basis of the palindromic Laurent polynomials `P`, the BPS factor is
determined by, and determines, `Ω ∈ P`.

`S` is then pinned by its **leading data** `t : Γ⁺ → Z`: it is the element whose
coefficient at `X_γ` has no `𝖖^{≤0}` part and has `𝖖¹`-coefficient `t(γ)`, for
every `γ`.  Taking `t = −1` on the node charges and `0` elsewhere is the BPS
spectrum generator, `S = 1 − 𝖖 Σ_a X_{γ(a)} + O(𝖖²)` in the sign convention
`E_𝖖(x) = 1 − 𝖖x + ⋯` produces.  The recursion is on cone degree: every
decomposition `γ = γ₁ + γ₂` inside the cone has both parts of strictly smaller
degree, so when `γ`'s turn comes every factor that can contribute at `γ` is
already fixed, and `Ω_γ` is forced —

    killing the 𝖖^{≤0} part   fixes  d_k  for k ≤ −1
    palindromy                fixes  d_k = d_{−k}
    the 𝖖¹ condition          fixes  d_0

with `Ω_γ = Σ_k d_k 𝖖^k`.  Nothing over-determines it.

**Order-independence** is what makes this an algorithm rather than a family of
them: the resulting `S` does not depend on where the BPS factors are placed
relative to one another (measured extensively, including random quivers at
rank 8/12/16).  Surjectivity of the factorisation is conjectural, resting on
one pairwise exchange identity; `docs/conjectures-step4-bps.md` §3 is the
honest status table.  **So a build
here is a construction whose *output* is checked, not a theorem** — which is why
the consumers cross-check against an independently computed `S`
(`BPSKAlgebra.verify_spectrum_generator_from_factors` against the chart's own
`_s_coefficient`; `recursive_spectrum` wherever that engine also runs).

⚠ `verify_leading_data` is **not** part of that evidence: `_forced` picks `Ω`
precisely so the leading conditions hold, integrally and palindromically by
construction, so the check cannot fail on a correct engine.  It is a
**regression guard on the arithmetic** — a sensitive one (it catches the
superseded `−𝖖·Ω` read on Kronecker-3 and Markov) but a *blind* one on any
chamber whose multiplicities are all `χ₀`/`χ_{1/2}`, pure SU(2) and the pentagon
included, where that wrong read gives bit-identical output.  Cite it as a guard,
never as support for surjectivity.

WHAT THIS MODULE DOES DIFFERENTLY FROM THE PROTOTYPES

Two prototypes preceded it — one phase-ordered, one free-placement.  Both
rebuild the whole partial product from scratch at every cone degree, and build
every BPS factor from scratch every time they do.  This module keeps their
arithmetic **bit-for-bit** (verified) and changes only how it is organised:

* **Node coordinates.**  A cone charge is its coefficient vector
  `k ∈ Z_{≥0}^r`, `Σk ≤ D`, so the cone is a simplex, `in_cone` is two integer
  tests, and the `Fraction` coordinate matrix-vector product per candidate pair
  disappears.  Lattice charges are restored only on output.
* **Simplex enumeration.**  `C(D+r, r)` cone points instead of the prototype's
  `(D+1)^r` scan — at rank 16, degree 2 that is 153 against 43 046 721.
* **Incremental accumulation.**  `Ω` at degree `d` is forced by the
  strictly-lower-degree factors alone — the placement of `γ` among them is free
  at that moment — so any **degree-compatible** order lets each BPS factor be
  *appended* and the partial product carried forward: `O(factors)` accumulator
  products instead of `O(degree × factors)`.  That is what `"lex"` and
  `"degree-key"` do.  A key order over the whole cone is *not* degree-compatible
  (a high-degree charge can have a key between two low-degree ones), so `"key"` /
  `"strip"` keep **cached prefix products** and rebuild only the suffix past the
  earliest insertion — measured 1.0–2.7 products per factor, against the prototype's
  full rebuild at every degree.  `"random"` is prefix-cached the same way.
* **Charge-local univariate arithmetic.**  `⟨γ,γ⟩ = 0`, so a BPS factor is a
  *commutative* truncated series in one variable; it is built as such (with
  binary powering for `m_s^{a_s}`) instead of through full-lattice products.
* **A charge-specialised accumulator product.**  `⟨k, n k₀⟩ = n·⟨k, k₀⟩`, so
  multiplying the accumulator by a BPS factor costs one integer dot product per
  accumulator key, not one per pair.
* **One expansion per readout.**  A single `expand(1)` per charge instead of a
  `coefficient(e)` call per exponent, each of which re-walks from `k_min`.

WHAT THE ORDER IS, AND WHAT IT IS NOT.  The recursion needs only a **total order
on the pairs `(γ, s)`** — one position per BPS factor
`E^{(s)}_𝖖(X_γ)^{Ω(γ,s)}`, each placeable at an arbitrary position among the
pre-existing ones.  Three parametrizations, from most to least
general:

    piece_key=  (γ, 2s) -> sortable     THE CONTRACT: any total order on pairs
    order_key=  γ       -> sortable     the coarse form: all of γ's pieces at
                                        one position, in ascending spin
    phases=     one complex per node    the LINEAR form: order by `arg Z_γ`

Each is strictly narrower than the one above it.  `phases=` is narrow enough to
matter: at rank 2 the phase order is monotone in slope, so a linear `Z` reaches
exactly **two** orders against the 43 branches free insertion finds at
Kronecker-2.  So read `phases=` as "the linear parametrization", never as what the
algorithm requires; the engine consumes all three only through `_pkey`.

`S` is conjecturally independent of the choice — measured over the whole
dictionary at charge level and at piece level.  What the choice *does* move is
the content `Ω`, which is what `factor_order_search.py` searches over.

So **no central charge is used unless one is supplied**.  The
default is the phase-free `"strip"` heuristic — the source/sink strip's node order
(the repo's own mantle theorem) extended to the cone by `mean_rank_key`, which
reproduces a good central charge's minimum-factor chamber *exactly* (`rank` factors,
spin-0 only).  Where the strip leaves a core but the quiver is not strongly
connected, the default is the **component order** — each strongly connected component
built on its own and the product assembled source-first,
the design notes §1.3 — and only on a strongly connected quiver,
where the decomposition determines nothing, or under a custom `leading_data`,
does it fall back to `"random"` placement of the individual `(s, γ)` pieces.
Placing each
`(s, γ)` piece independently is licensed: `⟨γ,γ⟩ = 0` makes the pieces at one `γ`
commute, and `S` is measured unchanged over 30 independent-placement
builds.

EXACTNESS.  Arithmetic is exact throughout (`HabiroElement`).  **The only
truncation is along the positive cone** — never in `𝖖`.  A `𝖖`-truncated engine
is not a cheaper approximation here but a wrong one: negative brackets shift
high-`𝖖` terms down into the visible window, and a truncated engine reported one
pentagon factorisation where the exact engine finds two.

Run:  PYTHONPATH=. python bps_factor_spectrum.py
"""

from __future__ import annotations

import cmath
import random
from fractions import Fraction
from typing import Callable, Sequence

from habiro import HabiroElement

# `c_n` (the `E_𝖖` factor coefficients) and the memoised `q_power` are shared with
# the peel engine deliberately: one definition, one memo.  Defining `c_n` twice
# flags.
from recursive_spectrum import c_n, q_pow

H0 = HabiroElement.zero()
H1 = HabiroElement.one()

Vec = tuple[int, ...]


# --------------------------------------------------------------------------
# palindromic multiplicities
# --------------------------------------------------------------------------

def spin_decompose(omega: dict[int, int]) -> dict[int, int]:
    """Palindromic `Ω` -> `{2s: a_s}` in the basis `χ_s`.

    Unique, and peelable from the top: `χ_s` has support of parity `2s` and top
    exponent `2s`, so the top exponent of what remains names the next spin.
    """
    remaining = {k: v for k, v in omega.items() if v}
    out: dict[int, int] = {}
    while remaining:
        top = max(remaining)
        amount = remaining[top]
        out[top] = out.get(top, 0) + amount
        for j in range(-top, top + 1, 2):
            new = remaining.get(j, 0) - amount
            if new:
                remaining[j] = new
            else:
                remaining.pop(j, None)
    return out


def _rank_over_q(rows) -> int:
    """Exact rank of an integer matrix, by Fraction row reduction."""
    from fractions import Fraction
    A = [[Fraction(x) for x in row] for row in rows]
    rank = 0
    width = len(A[0]) if A else 0
    for col in range(width):
        pivot = next((r for r in range(rank, len(A)) if A[r][col]), None)
        if pivot is None:
            continue
        A[rank], A[pivot] = A[pivot], A[rank]
        lead = A[rank][col]
        A[rank] = [x / lead for x in A[rank]]
        for r in range(len(A)):
            if r != rank and A[r][col]:
                factor = A[r][col]
                A[r] = [a - factor * b for a, b in zip(A[r], A[rank])]
        rank += 1
    return rank


def omega_string(omega: dict[int, int]) -> str:
    """`Ω` as a readable Laurent polynomial in `𝖖` (`"1q^-1 + 1q^1"`)."""
    if not omega:
        return "0"
    return " + ".join(f"{v}q^{k}" if k else f"{v}"
                      for k, v in sorted(omega.items()))


# --------------------------------------------------------------------------
# univariate (single-charge) arithmetic
# --------------------------------------------------------------------------
#
# Everything at one charge commutes: `X_{iγ} X_{jγ} = 𝖖^{ij⟨γ,γ⟩} X_{(i+j)γ}` and
# `⟨γ,γ⟩ = 0`.  So a BPS factor is an ordinary truncated power series in one
# variable, and none of the lattice machinery (tuple keys, brackets, cone tests)
# is needed to build it.

def _uni_mul(u: list, v: list, N: int) -> list:
    """Truncated product of two charge series, `[w_0 … w_N]`."""
    terms: list[list] = [[] for _ in range(N + 1)]
    for i, ui in enumerate(u):
        if ui.is_zero():
            continue
        cap = N - i
        if cap < 0:
            break
        for j, vj in enumerate(v):
            if j > cap:
                break
            if vj.is_zero():
                continue
            terms[i + j].append(ui * vj)
    return [HabiroElement.sum(t) if t else H0 for t in terms]


def _uni_inv(u: list, N: int) -> list:
    """Truncated inverse of a charge series with `u_0 = 1`.

    `w_0 = 1`, `w_n = −Σ_{i=1..n} u_i w_{n−i}` — the standard recurrence, exact.
    """
    w = [H1] + [H0] * N
    for n in range(1, N + 1):
        terms = []
        for i in range(1, n + 1):
            ui = u[i]
            if ui.is_zero():
                continue
            wn = w[n - i]
            if wn.is_zero():
                continue
            terms.append(ui * wn)
        if terms:
            w[n] = -HabiroElement.sum(terms)
    return w


def _uni_pow(u: list, e: int, N: int) -> list:
    """`u^e` for `e ≥ 1`, by binary powering."""
    result = None
    base = u
    while e:
        if e & 1:
            result = base if result is None else _uni_mul(result, base, N)
        e >>= 1
        if e:
            base = _uni_mul(base, base, N)
    return result if result is not None else [H1] + [H0] * N


def _e_series(shift: int, sign: int, N: int) -> list:
    """`E_𝖖( sign · 𝖖^shift · X_γ )` at the charge, as `[u_0 … u_N]`.

    `u_n = c_n · sign^n · 𝖖^{shift·n}`, since `X_γ^n = X_{nγ}` along one charge.
    """
    out = [H1] + [H0] * N
    for n in range(1, N + 1):
        term = c_n(n)
        if shift:
            term = term * q_pow(shift * n)
        if sign < 0 and n & 1:
            term = -term
        out[n] = term
    return out


def nahm_generators(charge: Vec, omega: dict[int, int]) -> list[tuple]:
    """`A_γ(Ω)` as a flat list of `E_𝖖` factors, for the Nahm route.

    Returns one `(charge, linear, diagonal, sign)` per factor, in product order,
    reading `charge_series`'s own decomposition
    `A_γ(Ω) = ∏_s m_s(γ)^{a_s}`,
    `m_s(γ) = ∏_{j=−2s,step 2}^{2s} E_𝖖((−1)^{2s} 𝖖^j X_γ)^{(−1)^{2s}}`.

    The per-factor coefficients follow from the two series (derivation and
    pinning in `nahm_local.general_nahm_habiro`):

        exponent +1  ->  linear 1+j,  diagonal 0,  sign base −σ
        exponent −1  ->  linear j,    diagonal 1,  sign base  σ

    This is what lets `general_nahm_habiro` verify **any** content rather than
    only a spin-0 spec — including the spin-carrying output of the order search's
    sparse fallback.  It walks `spin_decompose` in the same sorted order
    `charge_series` does, so the two describe the identical product.
    """
    out: list[tuple] = []
    for two_s, amount in sorted(spin_decompose(omega).items()):
        sigma = -1 if two_s % 2 else 1          # (−1)^{2s}, the argument sign
        exponent = amount * sigma
        if exponent == 0:
            continue
        for j in range(-two_s, two_s + 1, 2):
            if exponent > 0:
                entry = (tuple(charge), 1 + j, 0, -sigma)
            else:
                entry = (tuple(charge), j, 1, sigma)
            out.extend([entry] * abs(exponent))
    return out


_RAY_CACHE: dict[tuple, list] = {}


def charge_series(omega: dict[int, int], N: int) -> list:
    """`A_γ(Ω)` restricted to the charge set `{nγ : 0 ≤ n ≤ N}`, exact.

    `A_γ(Ω) = ∏_s m_s(γ)^{a_s}` with
    `m_s(γ) = ∏_{j=−2s,step 2}^{2s} E_𝖖((−1)^{2s} 𝖖^j X_γ)^{(−1)^{2s}}`.

    Memoised on `(Ω, N)`: the series depends on nothing else (the charge's own
    self-bracket vanishes), and the same `Ω` recurs across charges and — in the
    prefix-rebuilt phase order — across rebuilds.
    """
    key = (tuple(sorted(omega.items())), N)
    hit = _RAY_CACHE.get(key)
    if hit is not None:
        return hit
    acc = [H1] + [H0] * N
    for two_s, amount in sorted(spin_decompose(omega).items()):
        sign = -1 if two_s % 2 else 1          # (−1)^{2s}
        exponent = amount * sign
        if exponent == 0:
            continue
        for j in range(-two_s, two_s + 1, 2):
            base = _e_series(j, sign, N)
            if exponent < 0:
                base = _uni_inv(base, N)
            acc = _uni_mul(acc, _uni_pow(base, abs(exponent), N)
                           if abs(exponent) > 1 else base, N)
    _RAY_CACHE[key] = acc
    return acc


# --------------------------------------------------------------------------
# the cone, in node coordinates
# --------------------------------------------------------------------------

def cone_simplex(rank: int, degree: int) -> list[Vec]:
    """Node-coordinate cone points `k` with `1 ≤ Σk ≤ degree`, in `(deg, lex)`
    order.

    `C(degree + rank, rank) − 1` of them.  The prototype scanned the
    `(degree+1)^rank` box and filtered, which is exponential in the rank where
    this is polynomial at fixed degree (rank 16, degree 2: 153 against 43M).
    """
    out: list[Vec] = []
    cur = [0] * rank

    def rec(i: int, left: int) -> None:
        if i == rank:
            if left < degree:
                out.append(tuple(cur))
            return
        for v in range(left + 1):
            cur[i] = v
            rec(i + 1, left - v)
        cur[i] = 0

    rec(0, degree)
    out = [k for k in out if any(k)]
    out.sort(key=lambda k: (sum(k), k))
    return out


# --------------------------------------------------------------------------
# choosing the order: the source/sink strip
# --------------------------------------------------------------------------

def source_sink_strip(bracket: Sequence[Sequence[int]]) -> tuple[list[int], int]:
    """`(placement order, core size)` from iterated source/sink stripping.

    Sources (`⟨γ_k, γ_j⟩ ≥ 0` on every remaining `j`) go to the front in strip
    order, sinks to the back in reversed strip order, until the remainder — the
    **core** — has neither.  `core == 0` is the acyclic case, where the returned
    order is a genuine topological one; any core is spliced in the middle in index
    order, which carries no information (see `acyclic_node_order`).

    DELIBERATE DUPLICATION of `bps_quiver_tools._strip_mantle_indices`, which is
    the same recipe for the same reason on the spec side (the **mantle theorem**:
    acyclic ⇒ topological product spec, no search).  It is fifteen lines of
    integer work, and importing that 4 500-line module into this dependency-light
    spine module for it would be much the worse trade — the repo already takes
    this trade deliberately for `spin_decompose`, and the two are asserted
    to agree so the copy cannot drift.
    """
    rem = list(range(len(bracket)))
    heads: list[int] = []
    tails: list[int] = []
    changed = True
    while changed and rem:
        changed = False
        for k in list(rem):
            others = [j for j in rem if j != k]
            if all(bracket[k][j] >= 0 for j in others):
                heads.append(k)
                rem.remove(k)
                changed = True
            elif all(bracket[k][j] <= 0 for j in others):
                tails.append(k)
                rem.remove(k)
                changed = True
    return heads + list(rem) + list(reversed(tails)), len(rem)


def acyclic_node_order(pairing: Sequence[Sequence[int]],
                       node_charges: Sequence[Sequence[int]]
                       ) -> list[int] | None:
    """The nodes' placement order from the source/sink strip — or `None`.

    Returns a permutation of the node indices, first-placed first: the honest
    object, since what the strip determines is an *order*, not a central charge.
    `None` means the quiver is **not acyclic** (the strip leaves a core), and then
    this route has nothing to say: an honest failure rather than a guess.

    A node order does **not** by itself determine an order on the cone — the
    interior charges need positions too — so to feed it to `BPSFactorSpectrum` extend it
    with `central_charge_for_node_order`, the canonical linear extension.  (That
    extension is why the strip is expressed through a central charge at all; the
    engine itself needs only a total order, `order_key=`.)

    WHY THIS IS WORTH HAVING, and why it is not the default:

    * Cost tracks the number of factors placed, i.e. the number of BPS states in the
      chamber the central charge selects.  On an acyclic quiver this order selects
      a chamber with exactly `rank` states — the minimum — measured **7/7**.
    * The penalty for a bad order is **not a constant factor** on quivers with an
      infinite chamber: the good order stays at `rank` factors for every cutoff while
      a bad one pays for the whole tower.  Measured on Kronecker-3, good against
      reversed: 2 factors against 13/20/29/40 at `D = 6/8/10/12`, a time ratio rising
      13.5× → **118.6×**.  On finite-chamber quivers it is a bounded 2–4×.
    * **`None` is the honest answer on a core, not a fallback.**  Where the strip
      leaves a core it degenerates to the identity order, which measures *worse
      than a random permutation*: 15 factors against a best of 6 at pure
      SU(3)-cyclic, 70 against 13 at random rank 6 (79th percentile).  Returning
      the degenerate order would be worse than returning nothing.
    * It is **not** the default because there is almost nothing to gain in
      production — 9 of the 11 chart shapes the repo's own tests build are already
      in a strip order (median gain 1.04×) — and because changing the default
      central charge would silently move which chamber `multiplicities()` reports,
      which is reported physics rather than an internal.  Mutated frames and
      user-supplied charts are what this is for.
    """
    nodes = [tuple(int(x) for x in g) for g in node_charges]
    dim = len(pairing)
    B = [[int(x) for x in row] for row in pairing]
    bracket = [[sum(a[p] * B[p][r] * b[r] for p in range(dim) for r in range(dim))
                for b in nodes] for a in nodes]
    order, core = source_sink_strip(bracket)
    return None if core else order


def central_charge_for_node_order(node_order: Sequence[int]) -> list[complex]:
    """Extend a node order to an order on the whole cone, linearly.

    Returns the `phases=` vector whose induced order places the nodes in
    `node_order`, first-placed first — one complex number per node, arguments
    strictly increasing across the upper half plane (the engine's own default
    shape, so this is exactly a permutation of it).

    THE EXTENSION IS THE POINT, and it is why a central charge appears here at
    all.  A permutation of the *nodes* leaves every interior cone charge
    unplaced; a linear `Z` extends it to all of `Γ⁺` at once by
    `arg Z(Σ k_i γ_i)`.  That is a genuine choice of extension, not a neutral
    encoding — and a severely restrictive one: at rank 2 the phase order is
    monotone in slope, so only **two** of the admissible orders are reachable by
    any linear `Z`, against 43 branches that free insertion finds at Kronecker-2
    (measured).  For an order outside that
    subclass, pass `order_key=` instead.
    """
    order = [int(i) for i in node_order]
    rank = len(order)
    pos = {node: i for i, node in enumerate(order)}
    return [cmath.exp(1j * cmath.pi * (pos[i] + 1) / (rank + 1))
            for i in range(rank)]


def mean_rank_key(node_order: Sequence[int]):
    """Extend a node order to the whole cone **without any central charge**.

    `key(k) = (Σ k_i r_i) / (Σ k_i)` — the mean rank of `k`'s constituent nodes,
    `r_i` being node `i`'s position in `node_order`.  Exact `Fraction`
    arithmetic, so no float tie ever decides a placement.

    WHY THIS IS THE RIGHT PHASE-FREE EXTENSION.  What a linear central charge
    contributes to the order is one property: `arg Z(a+b)` lies strictly between
    `arg Z(a)` and `arg Z(b)`, so a sum sits between its summands and the
    interior charges land where the KS wall-crossing wants them.  A mean has
    exactly that betweenness property, and needs no complex numbers at all.
    Measured to reproduce the central charge's minimum-factor chamber *exactly* —
    `rank` factors at pentagon / pure SU(2) / Kronecker-3 / Kronecker-4 / A3 / A4 /
    D4-star, identical to the `phases=` route and with the same `S`.
    """
    pos = {node: i for i, node in enumerate(node_order)}

    def key(k: Vec) -> Fraction:
        total = sum(k)
        if not total:
            return Fraction(0)
        return Fraction(sum(ki * pos[i] for i, ki in enumerate(k)), total)

    return key


#: `"phase"` / `"degree-phase"` name a *parametrization* (a linear central
#: charge), but the modes they select are key-driven and accept any
#: `order_key` — so the honest names are `"key"` / `"degree-key"` and the old
#: ones are kept as aliases rather than breaking the merged surface.
_MODE_ALIASES = {"phase": "key", "degree-phase": "degree-key"}


# --------------------------------------------------------------------------
# the builder
# --------------------------------------------------------------------------

class BPSFactorSpectrum:
    """`S` and its factor multiplicities `Ω`, for one BPS quiver and one cutoff.

    Parameters
    ----------
    pairing
        Antisymmetric integer matrix of the lattice `Γ`.
    node_charges
        The BPS-quiver nodes, generating the positive cone.  Must be linearly
        independent (they are a basis of `Γ` in every shipped chart) — the node
        coordinates are the working representation.
    cutoff
        Cone-degree truncation `D`.  The **only** truncation; never a
        `𝖖`-truncation.
    leading_data
        Optional `charge -> int`, the prescribed `𝖖¹`-coefficient of `S`.
        Default: `−1` on each node charge, `0` elsewhere — the BPS spectrum
        generator.  (Varying it asks whether order-independence needs the BPS
        leading data or holds for any.)
    order
        Where to place the BPS factors relative to one another.  `S` is
        order-independent, so this moves only the *content* and the cost.

        WHAT THE ENGINE ACTUALLY REQUIRES IS A TOTAL ORDER ON THE PAIRS `(γ, s)`,
        and nothing more — every BPS factor `E^{(s)}_𝖖(X_γ)^{Ω(γ,s)}` may be
        placed at an arbitrary position among the pre-existing ones.  A
        central charge is one
        *parametrization* of such an order, not the requirement, and a narrow one:
        at rank 2 the phase order is monotone in slope, so a linear `Z` reaches
        only **two** orders against the 43 branches free insertion finds at
        Kronecker-2.  Use `piece_key=` for the contract's own generality, or
        `order_key=` for the charge-level special case.

        * `"strip"` (DEFAULT on an acyclic quiver) — the source/sink strip's node
          order, extended to the cone by `mean_rank_key`.  Phase-free, and
          measured to place exactly `rank` factors, spin-0 only — the same chamber a
          well-chosen central charge selects.
        * `"component"` (DEFAULT on a cyclic quiver that is not strongly
          connected; user, 2026-09-23) — the component order of
          the design record §1.3: the strongly connected components
          (`quiver_enumeration.strongly_connected_components`, source-first —
          every arrow between two of them points from an earlier one to a later
          one), each component's pieces before those of every later component.
          In that order `Ω = 0` is FORCED at every charge whose support meets two
          or more components (its coefficient in the partial product is a
          product of two or more coefficients in `𝖖ℤ[[𝖖]]`), so the engine
          builds each component ALONE, on its own cone and in its own default
          order, and assembles `S` as their ordered product — one term per
          coefficient, since a charge splits among the components in exactly
          one way.  `Ω` is the disjoint union of the components'.  Needs the BPS
          leading data (the zero at mixed charges is where the forcing comes
          from), so a custom `leading_data` is refused.  On a strongly connected
          quiver it is the component's own default, i.e. `"random"`.
        * `"random"` (DEFAULT on a strongly connected quiver) — every `(s, γ)`
          piece at its own random position.  Verified to leave `S` unchanged
          (a probe in the source repository).
        * `"key"` — ordered by the key: `piece_key` if given, else `order_key`,
          else `arg Z_γ` from `phases`.  Not degree-compatible, so it is built
          with cached prefix products.  With `piece_key` the placement unit is the
          individual `(s, γ)` piece; otherwise it is the whole BPS factor.  With a
          central charge this is the classical phase order, and then the only mode
          whose `Ω` is a chamber's BPS spectrum.
        * `"lex"` — node-coordinate lexicographic within each degree.
        * `"degree-key"` — by the same key, within each degree.
        * `"phase"` / `"degree-phase"` — aliases of `"key"` / `"degree-key"`, kept
          because they name a *parametrization* rather than the mode.

        `"lex"` and `"degree-key"` are degree-compatible, so each factor is
        appended and one accumulator is carried through.  See `PLACEMENT_ORDERS`.
    phases
        Complex central charge per node — the *linear* parametrization of the
        order.  **Supplying it is what asks for a central-charge order** and the
        only way one is ever used; omitted, no phase is
        computed at all.  Falls back to a generic spread across the upper half
        plane when a key-driven mode is named without either knob.  Mutually
        exclusive with `order_key`.
    order_key
        `charge (node coords) -> sortable`, an ARBITRARY total order on the cone —
        the general form of what the two key-driven modes consume, without the
        linearity a central charge imposes.  Ties break on the node-coordinate
        vector, so a constant key degenerates to lexicographic rather than
        becoming ill-defined.  Mutually exclusive with `phases` and `piece_key`.

        **Its generality stops at the charge**, which is why `piece_key` exists:
        a key on `γ` alone sends all of `γ`'s spin pieces
        `E^{(s)}_𝖖(X_γ)^{Ω(γ,s)}` to one position, in ascending spin.  That is a
        genuine special case of the contract, not a defect — but it is a special
        case, and the general one is below.
    piece_key
        `(charge (node coords), 2s) -> sortable` — **the contract's own form**: an
        arbitrary total order on the pairs `(γ, s)`, one position per multiplet
        factor `E^{(s)}_𝖖(X_γ)^{Ω(γ,s)}`.  Ties break on
        `(charge, 2s)`, so a constant key degenerates to lexicographic rather than
        becoming ill-defined.  Mutually exclusive with `phases` and `order_key`.

        Placing the pieces at a fixed `γ` independently is licensed because
        `⟨γ,γ⟩ = 0` makes them commute, so a split can only matter against factors
        at *other* charges; `S` is measured unchanged under it.

        ⚠ It has content only where some `Ω(γ)` carries **≥ 2 distinct spins**,
        and in a central-charge order there are ZERO such charges on all eight
        quivers surveyed — each `Ω` is a single `χ_s`, so all of `γ`'s pieces
        already *are* one piece.  A battery run only on phase orders reports a
        vacuous pass; use `"lex"` or a core quiver to give it content.
    """

    def __init__(
        self,
        pairing: Sequence[Sequence[int]],
        node_charges: Sequence[Sequence[int]],
        cutoff: int,
        *,
        leading_data: Callable[[Vec], int] | None = None,
        order: str | None = None,
        phases: Sequence[complex] | None = None,
        order_key: Callable[[Vec], object] | None = None,
        piece_key: Callable[[Vec, int], object] | None = None,
        seed: int = 20260813,
    ):
        self._rng = random.Random(seed)
        self.seed = seed
        given = [n for n, v in (("phases", phases), ("order_key", order_key),
                                ("piece_key", piece_key)) if v is not None]
        if len(given) > 1:
            raise ValueError(
                f"pass at most one of phases / order_key / piece_key (got "
                f"{', '.join(given)}): they are three parametrizations of the "
                f"same thing — a total order on the pairs (γ, s) — of strictly "
                f"decreasing generality, piece_key being the contract's own form "
                f"and a central charge the narrowest.")
        self._order_key = order_key
        self._piece_key = piece_key
        self.order_requested = order
        order = _MODE_ALIASES.get(order, order)
        # Only a caller who asked for no order at all gets the component order
        # by default; an explicit order, key or central charge is honoured as is.
        default_order = order is None and not given
        if order == "component" and given:
            raise ValueError(
                "order='component' is an order of its own — the strongly "
                "connected components in source-first order, each in its own "
                "default order — so it takes no phases / order_key / piece_key.")
        # DEFAULT MODE SELECTION — no central charge unless one is supplied
        #  Supplying `phases` IS the request for a
        # central-charge order; otherwise the default is a combinatorial
        # heuristic, and only where that heuristic has nothing to say does it
        # fall back to random placement.
        self._strip_node_order: list[int] | None = None
        if order is None:
            if phases is not None or order_key is not None or piece_key is not None:
                order = "key"
            else:
                seq = acyclic_node_order(pairing, node_charges)
                if seq is None:
                    order = "random"
                else:
                    order = "strip"
        if order == "strip":
            seq = acyclic_node_order(pairing, node_charges)
            if seq is None:
                raise ValueError(
                    "order='strip' needs an ACYCLIC quiver: the source/sink "
                    "strip leaves a core here, so it determines no order.  Omit "
                    "`order` (the default falls back to random placement) or "
                    "pass an explicit order_key.")
            self._strip_node_order = seq
            self._order_key = mean_rank_key(seq)
        self.nodes: list[Vec] = [tuple(int(x) for x in g) for g in node_charges]
        self.rank = len(self.nodes)
        self.dim = len(pairing)
        self.degree_cap = int(cutoff)
        if self.degree_cap < 1:
            raise ValueError("cutoff must be at least 1")
        if order not in PLACEMENT_ORDERS:
            raise ValueError(f"order must be one of {sorted(PLACEMENT_ORDERS)}")
        self.order = order
        # The whole construction runs in node coordinates, so `k ↦ Σ k_i γ_i`
        # must be injective on the cone.  Linear independence is what makes it
        # so; without this check a dependent node set would quietly identify two
        # distinct cone points and merge their coefficients — a wrong `S` with no
        # symptom.  Every shipped BPS chart has independent nodes (they are a
        # basis of Γ), so this is a guard, not a restriction.
        if _rank_over_q(self.nodes) != self.rank:
            raise ValueError(
                f"node charges must be linearly independent (got {self.rank} "
                f"charges of rank {_rank_over_q(self.nodes)}): the BPS-factor "
                f"recursion indexes the cone by node coordinates, which a "
                f"dependent set makes ambiguous.")

        # Node-coordinate bracket: ⟨Σ k_i γ_i, Σ l_j γ_j⟩ = k·(B_node)·l.
        B = [[int(x) for x in row] for row in pairing]
        self._bnode = [
            [sum(a[p] * B[p][r] * b[r] for p in range(self.dim)
                 for r in range(self.dim)) for b in self.nodes]
            for a in self.nodes
        ]

        # The component order: the default on a cyclic
        # quiver that is not strongly connected, and available by name on any
        # quiver.  `"random"` is reached by default only on a cyclic quiver, so
        # that is where the decomposition is asked for.
        self._components: list[list[int]] | None = None
        if self.order == "component" or (default_order and self.order == "random"):
            from quiver_enumeration import strongly_connected_components
            comps = strongly_connected_components(self._bnode)
            if self.order == "component":
                if leading_data is not None:
                    raise ValueError(
                        "order='component' needs the BPS leading data: the zero "
                        "multiplicity at every charge meeting two components is "
                        "forced by the leading data vanishing there, so a custom "
                        "leading_data is built with the ordinary engine instead.")
                self._components = comps
            elif len(comps) > 1 and leading_data is None:
                self.order = "component"
                self._components = comps

        # A generic linear central charge: strictly decreasing argument across
        # the upper half plane, so no two cone charges share a phase and the
        # order is a genuine total order (all-equal phases would collapse the
        # sort onto its lexicographic tiebreak).
        self._phases = ([complex(p) for p in phases] if phases is not None
                        else [cmath.exp(1j * cmath.pi * (i + 1) / (self.rank + 1))
                             for i in range(self.rank)])
        self.cone: list[Vec] = cone_simplex(self.rank, self.degree_cap)
        self._leading = leading_data
        self.omega: dict[Vec, dict[int, int]] = {}   # node coords -> Ω
        self._S: dict[Vec, HabiroElement] | None = None  # node coords -> coeff

    # ---- coordinates ------------------------------------------------------

    @property
    def node_pairing(self) -> list[list[int]]:
        """`⟨γ_i, γ_j⟩` — the antisymmetric pairing in node coordinates.

        Public because the joint `F`/`S` builder in `fs_builder.py` needs the
        same bracket for its `F` side, and a second assembly of it from the
        lattice matrix is exactly the kind of duplicate that drifts.
        """
        return self._bnode

    def charge(self, k: Vec) -> Vec:
        """Node coordinates -> the lattice charge `Σ k_i γ_i`."""
        return tuple(sum(k[i] * self.nodes[i][d] for i in range(self.rank))
                     for d in range(self.dim))

    def target(self, k: Vec) -> int:
        """The prescribed `𝖖¹`-coefficient of `S` at this charge."""
        if self._leading is not None:
            return int(self._leading(self.charge(k)))
        return -1 if sum(k) == 1 and max(k) == 1 else 0

    def _phase(self, k: Vec) -> float:
        """`arg Z_γ` for the linear central charge in `self._phases`."""
        return cmath.phase(sum(complex(c) * p
                               for c, p in zip(k, self._phases)))

    def _key(self, k: Vec):
        """The order key at this charge — the ONLY thing the engine asks of an
        order, which is why an arbitrary `order_key` is admissible and a central
        charge is merely its linear special case.  The node-coordinate tiebreak
        keeps the order total even for a degenerate key."""
        return ((self._order_key(k) if self._order_key is not None
                 else self._phase(k)), k)

    def _pkey(self, k: Vec, two_s: int):
        """The order key at the PAIR `(γ, s)` — the contract's placement unit.

        The three parametrizations collapse to this one function, which is the
        whole point of having it: `piece_key` uses both arguments, `order_key` and
        `phases` use only `k` and then tie-break on `(k, 2s)`, so their pieces
        stay adjacent in ascending spin — which is *exactly* the whole BPS factor
        `∏_s m_s(γ)^{a_s}`, since `charge_series` multiplies its pieces in that same
        ascending order.  So the charge-level modes are recovered bit-for-bit by
        the piece-level machinery rather than approximated by it.
        """
        if self._piece_key is not None:
            return (self._piece_key(k, two_s), k, two_s)
        return (self._key(k), two_s)

    def degree_order(self, degree: int) -> list[Vec]:
        """The charges of this cone degree, in placement order."""
        here = [k for k in self.cone if sum(k) == degree]
        if self.order == "degree-key":
            here.sort(key=self._key)
        return here

    # ---- the recursion ----------------------------------------------------

    def run(self, *, after_degree: Callable[[int, dict], None] | None = None,
            ) -> dict[Vec, dict[int, int]]:
        """Build `S`.  Returns `Ω` keyed by node coordinates.

        `after_degree(d, partial)` — optional, called once per cone degree with
        the **complete** partial product after every piece of degree `≤ d` has
        been placed.  At that moment `partial[ε]` is FINAL for every `deg ε ≤ d`
        (a piece of degree `d' > d` first contributes at `d'`), so a consumer may
        read those coefficients as exact rather than provisional.  That is the
        seam `fs_builder.FSBuilder` uses to grow `F_γ` in lockstep with `S`; it
        fires at every degree, including one that places nothing.

        Three shapes, and the component order built from them:

        * `"component"` (the DEFAULT on a cyclic quiver that is not strongly
          connected) — each strongly connected component run as its own
          engine in its own default order, the product assembled source-first
          (`_run_components`); no charge meeting two components is ever read.
        * `"strip"` (the DEFAULT where it applies) — the source/sink strip's node
          order, extended to the cone by `mean_rank_key`.  **No central charge**,
          which is the point: the recursion needs only some
          total order, and the default must not smuggle in physics it does not
          need.  Measured to reproduce the central charge's minimum-factor chamber
          exactly — `rank` factors, spin-0 only — so being phase-free costs nothing.
        * `"random"` (the DEFAULT fallback, on a strongly connected quiver)
          — each `E^{(s)}_𝖖(X_γ)^{Ω(γ,s)}` goes at its own uniformly random
          position among those already placed.  Prefix-cached like `"key"`, since a
          random position is an interior insertion.
        * `"lex"` / `"degree-phase"` are degree-compatible, so every new factor
          factor is **appended** and one accumulator is carried right through —
          `O(factors)` products.
        * `"phase"` is not (a high-degree charge can have a phase between two
          low-degree ones), so the placed factors are kept in phase order with
          **cached prefix products** and only the suffix past the earliest
          insertion is rebuilt.  Costlier per factor, but with a central charge it
          places far fewer and much smaller `Ω`, and it is the only mode whose `Ω`
          is a chamber's BPS spectrum — ask for it when that is what you want.
        """
        if self.order == "component":
            return self._run_components(after_degree)
        if self.order == "random":
            return self._run_piece_insert(keyed=False, after_degree=after_degree)
        if self.order in ("strip", "key"):
            return (self._run_piece_insert(keyed=True, after_degree=after_degree)
                    if self._piece_key is not None
                    else self._run_insert(after_degree))
        return self._run_append(after_degree)

    def _run_piece_insert(self, *, keyed: bool, after_degree=None,
                          ) -> dict[Vec, dict[int, int]]:
        """Place every `(s, γ)` piece at its own position — the contract's unit.

        The placement unit is the individual `E^{(s)}_𝖖(X_γ)^{Ω(γ,s)}`, not the
        whole BPS factor `∏_s m_s(γ)^{a_s}` — a relaxation of the free-insertion
        one, licensed because `⟨γ,γ⟩ = 0` makes all factors at a fixed `γ` commute,
        so splitting them can only matter against *other* charges.  Verified to
        leave `S` unchanged over 30 independent-placement builds on five quivers
        (measured), with the faithfulness control
        reproducing the appended build exactly.

        `keyed=False` is `order="random"`: each piece at a uniformly random
        position.  `keyed=True` is `piece_key=`: each piece where its key on the
        pair `(γ, s)` puts it.  The two differ in one line, which is the honest
        shape — a random order IS a total order on the pairs, drawn rather than
        specified, so it does not deserve a build loop of its own.

        Prefix products are cached exactly as in `_run_insert`; the rebuild point is
        the minimum insertion index recorded this degree, which is a safe lower
        bound for the earliest invalidated prefix (a later insertion at a smaller
        index only shifts an earlier one further right).
        """
        import bisect

        zero = tuple([0] * self.rank)
        placed: list[tuple[Vec, int, dict[int, int]]] = []
        sort_keys: list = []
        prefix: list[dict] = [{zero: H1}]
        prefix_deg: list[dict] = [{zero: 0}]
        self.omega = {}

        for d in range(1, self.degree_cap + 1):
            fresh = self._readout(prefix[-1], d)
            if fresh:
                first = len(placed)
                for k, om in fresh:
                    self.omega[k] = om
                    for two_s, amount in sorted(spin_decompose(om).items()):
                        piece = {j: amount
                                 for j in range(-two_s, two_s + 1, 2)}
                        if keyed:
                            pkey = self._pkey(k, two_s)
                            at = bisect.bisect_left(sort_keys, pkey)
                            sort_keys.insert(at, pkey)
                        else:
                            at = self._rng.randrange(len(placed) + 1)
                        placed.insert(at, (k, d, piece))
                        first = min(first, at)
                del prefix[first + 1:]
                del prefix_deg[first + 1:]
                for i in range(first, len(placed)):
                    k0, deg0, piece = placed[i]
                    nxt, nxt_deg = self.factor_multiply(
                        prefix[i], prefix_deg[i], k0, deg0, piece)
                    prefix.append(nxt)
                    prefix_deg.append(nxt_deg)
            if after_degree is not None:
                after_degree(d, prefix[-1])

        self._S = prefix[-1]
        return self.omega

    def _run_components(self, after_degree=None) -> dict[Vec, dict[int, int]]:
        """The component order: each strongly connected component built alone,
        `S` assembled as the ordered product (the design notes
        §1.3; user, 2026-09-23).

        Each component runs as its own `BPSFactorSpectrum` — its node bracket,
        the standard basis, this cutoff and seed, and its own default order
        (`"strip"` for a single node, `"random"` for a cyclic component).  That
        IS this engine in the total order placing every component's pieces
        before the later components' — the readout at a charge of one component
        sees only that component's factors, and at a charge meeting two or more
        components `Ω` is forced to zero — so no mixed charge is ever read out.

        `after_degree` keeps its contract: the partial product after every piece
        of degree `≤ d` is the ordered product of the components' own partial
        products at degree `d`, assembled and delivered once each component has
        been built.
        """
        builds = []
        for comp in self._components:
            m = len(comp)
            sub = BPSFactorSpectrum(
                [[self._bnode[i][j] for j in comp] for i in comp],
                [tuple(1 if c == r else 0 for c in range(m)) for r in range(m)],
                self.degree_cap, seed=self.seed)
            partials: list | None = [] if after_degree is not None else None
            sub.run(after_degree=None if partials is None
                    else (lambda d, p, _into=partials: _into.append(p)))
            builds.append((comp, sub, partials))
        self.omega = {}
        for comp, sub, _ in builds:
            for k, om in sub.omega.items():
                self.omega[self._embed(k, comp)] = om
        if after_degree is not None:
            for d in range(1, self.degree_cap + 1):
                after_degree(d, self._assemble(
                    [(comp, partials[d - 1]) for comp, _, partials in builds]))
        self._S = self._assemble([(comp, sub._S) for comp, sub, _ in builds])
        return self.omega

    def _embed(self, k: Vec, comp: Sequence[int]) -> Vec:
        """A component's node coordinates, as this quiver's."""
        out = [0] * self.rank
        for a, node in enumerate(comp):
            out[node] = k[a]
        return tuple(out)

    def _assemble(self, parts) -> dict[Vec, HabiroElement]:
        """`∏_k S_k` in the order of `parts` — `(component nodes, coefficients in
        the component's node coordinates)`, source-first — truncated to this
        cone.  The supports are disjoint, so a product charge `Σ_k γ_k` arises
        from exactly one tuple of parts and its coefficient is the single term
        `𝖖^{Σ_{k<l} ⟨γ_k, γ_l⟩} ∏_k (S_k)_{γ_k}` — no sums, and the exponent is
        never negative, every arrow between two components pointing forward."""
        D, n = self.degree_cap, self.rank
        zero = tuple([0] * n)
        acc: dict[Vec, HabiroElement] = {zero: H1}
        acc_deg: dict[Vec, int] = {zero: 0}
        for comp, coeffs in parts:
            m = len(comp)
            items = []
            for ka, ca in coeffs.items():
                da = sum(ka)
                if not da or ca.is_zero():
                    continue
                # w[j] = ⟨γ_j, the embedded ka⟩, so ⟨key, ka⟩ = Σ_j key_j w_j.
                w = [sum(self._bnode[j][comp[b]] * ka[b] for b in range(m))
                     for j in range(n)]
                items.append((da, self._embed(ka, comp), ca, w))
            items.sort(key=lambda t: t[0])
            new, new_deg = dict(acc), dict(acc_deg)     # the (S_k)_0 = 1 term
            for key, c in acc.items():
                room = D - acc_deg[key]
                for da, emb, ca, w in items:
                    if da > room:
                        break
                    br = 0
                    for j, kj in enumerate(key):
                        if kj:
                            br += kj * w[j]
                    term = c * ca
                    if br:
                        term = term * q_pow(br)
                    nk = tuple(a + b for a, b in zip(key, emb))
                    new[nk] = term
                    new_deg[nk] = acc_deg[key] + da
            acc, acc_deg = new, new_deg
        return acc

    def _readout(self, partial: dict[Vec, HabiroElement],
                 degree: int) -> list[tuple[Vec, dict[int, int]]]:
        """Every forced `Ω` at this cone degree, read off one partial product.

        A same-degree factor contributes nothing at another degree-`d` charge
        (two of them reach degree `≥ 2d`; one alone reaches `nγ` at degree `nd`),
        so these readouts are independent of each other and of their relative
        placement — which is what lets a whole degree be read before any of it is
        placed.
        """
        fresh = []
        for k in self.degree_order(degree):
            coeff = partial.get(k)
            om = self._forced(expansion(coeff) if coeff is not None else {},
                              self.target(k))
            if om:
                fresh.append((k, om))
        return fresh

    def _run_append(self, after_degree=None) -> dict[Vec, dict[int, int]]:
        """The degree-compatible modes: append, carrying one accumulator.

        With a `piece_key` the appended unit becomes the individual `(s, γ)`
        piece, sorted within the degree by the key — the same relationship
        `"degree-key"` already has to `"key"`, restricted to a degree-compatible
        order rather than a global one.  Without one the whole BPS factor is
        appended, which is that product of pieces in ascending spin, so the two
        agree wherever both apply.
        """
        zero = tuple([0] * self.rank)
        acc: dict[Vec, HabiroElement] = {zero: H1}
        acc_deg: dict[Vec, int] = {zero: 0}
        self.omega = {}
        for d in range(1, self.degree_cap + 1):
            fresh = self._readout(acc, d)
            for k, om in fresh:
                self.omega[k] = om
            if self._piece_key is None:
                for k, om in fresh:
                    acc, acc_deg = self.factor_multiply(acc, acc_deg, k, d, om)
            else:
                pieces = [(self._pkey(k, two_s), k,
                           {j: amount for j in range(-two_s, two_s + 1, 2)})
                          for k, om in fresh
                          for two_s, amount in spin_decompose(om).items()]
                pieces.sort(key=lambda t: t[0])
                for _, k, piece in pieces:
                    acc, acc_deg = self.factor_multiply(acc, acc_deg, k, d, piece)
            if after_degree is not None:
                after_degree(d, acc)
        self._S = acc
        return self.omega

    def _run_insert(self, after_degree=None) -> dict[Vec, dict[int, int]]:
        import bisect

        zero = tuple([0] * self.rank)
        placed: list[Vec] = []          # charges, sorted by (phase, charge)
        sort_keys: list[tuple] = []
        prefix: list[dict] = [{zero: H1}]        # prefix[i] = ∏ placed[:i]
        prefix_deg: list[dict] = [{zero: 0}]
        self.omega = {}

        for d in range(1, self.degree_cap + 1):
            fresh = self._readout(prefix[-1], d)
            if fresh:
                first = len(placed)
                for k, om in fresh:
                    self.omega[k] = om
                    key = self._key(k)
                    at = bisect.bisect_left(sort_keys, key)
                    sort_keys.insert(at, key)
                    placed.insert(at, k)
                    first = min(first, at)
                # Only the suffix from the earliest insertion is invalid; every
                # prefix before it is still the product of the same factors in
                # the same order.
                del prefix[first + 1:]
                del prefix_deg[first + 1:]
                for i in range(first, len(placed)):
                    k = placed[i]
                    nxt, nxt_deg = self.factor_multiply(
                        prefix[i], prefix_deg[i], k, sum(k), self.omega[k])
                    prefix.append(nxt)
                    prefix_deg.append(nxt_deg)
            if after_degree is not None:
                after_degree(d, prefix[-1])

        self._S = prefix[-1]
        return self.omega

    @staticmethod
    def _forced(f: dict[int, int], target: int) -> dict[int, int]:
        """`Ω` forced at one charge by the accumulated coefficients `f`.

        A BPS factor contributes `−L·Ω` with `L = 𝖖/(1−𝖖²)`, NOT `−𝖖·Ω`:
        `c_1 = −𝖖/(1−𝖖²)` carries an infinite tail.  So the coefficient at
        `𝖖^m` is `−Σ_{k ≤ m−1, k ≡ m−1 (2)} d_k`, and that triangular system
        inverts to a plain difference.  (Using `−𝖖·Ω` is wrong from spin 1 up,
        and was a real defect.)
        """
        omega: dict[int, int] = {}
        lo = min([e for e in f if e <= 0] or [0])
        for m in range(lo, 1):
            omega[m - 1] = f.get(m, 0) - f.get(m - 2, 0)
        omega[0] = f.get(1, 0) - f.get(-1, 0) - target
        for k in [k for k in omega if k < 0]:
            omega[-k] = omega[k]
        return {k: v for k, v in omega.items() if v}

    def factor_multiply(self, acc, acc_deg, k0: Vec, deg0: int,
                      omega: dict[int, int]):
        """`acc · A_{k0}(Ω)`, cone-truncated.  Returns `(product, degrees)`.

        Specialised to a charge right operand, which is where all the cost is:

        * `⟨k, n·k0⟩ = n·⟨k, k0⟩`, so **one** integer dot product per
          accumulator key replaces one bracket per *pair*;
        * the nonzero factor coefficients are collected **once**, not re-tested per
          accumulator key (that test alone was a quarter of a million calls on a
          rank-8 degree-4 build);
        * out-of-cone multiples are skipped without ever forming the key;
        * output degrees come out arithmetically as `deg(k) + n·deg0`, so no key
          is ever re-summed;
        * `q_pow(...)` multiplication is free of a `simplify` (`__mul__` has a
          monomial fast path).

        TWO OPTIMIZATIONS TRIED HERE AND MEASURED WORSE — both recorded because
        each looks obviously right:

        * **Deferring the simplify.**  `c · u_n` is the one unavoidable
          `simplify` per term; forming the product unsimplified and letting the
          single closing `HabiroElement.sum` simplify once per key halves the
          count, and ran **5× slower** on rank 8 degree 4 (0.13 s → 0.72 s).
          Uncancelled `(1−𝖖^{2k})` factors make the numerators grow, and every
          later `sum` then scales them over a larger common denominator.  Cancel
          early.
        * **Carrying each key's support** so the bracket runs over `≤ deg`
          nonzero coordinates instead of all `rank` of them.  The integer ops it
          saves are cheaper than the tuple it must allocate per pair to maintain
          the support: ~3× slower at rank 8, ~3× at rank 16.  The full-length
          scan wins.
        """
        D = self.degree_cap
        nmax_charge = D // deg0
        series = charge_series(omega, nmax_charge)
        nonzero = [(n, series[n]) for n in range(1, nmax_charge + 1)
                   if not series[n].is_zero()]
        # w_i = ⟨γ_i, k0⟩ in node coordinates, so ⟨k, k0⟩ = Σ k_i w_i.
        w = [sum(self._bnode[i][j] * k0[j] for j in range(self.rank))
             for i in range(self.rank)]
        buckets: dict[Vec, list] = {}
        out_deg: dict[Vec, int] = {}
        for k, c in acc.items():
            deg = acc_deg[k]
            bucket = buckets.get(k)
            if bucket is None:
                buckets[k] = [c]
                out_deg[k] = deg
            else:
                bucket.append(c)
            room = (D - deg) // deg0
            if room <= 0:
                continue
            br = 0
            for i, ki in enumerate(k):
                if ki:
                    br += ki * w[i]
            for n, un in nonzero:
                if n > room:
                    break
                nk = tuple(ki + n * k0i for ki, k0i in zip(k, k0))
                term = c * un
                if br:
                    term = term * q_pow(n * br)
                bucket = buckets.get(nk)
                if bucket is None:
                    buckets[nk] = [term]
                    out_deg[nk] = deg + n * deg0
                else:
                    bucket.append(term)
        out: dict[Vec, HabiroElement] = {}
        new_deg: dict[Vec, int] = {}
        for key, terms in buckets.items():
            total = terms[0] if len(terms) == 1 else HabiroElement.sum(terms)
            if not total.is_zero():
                out[key] = total
                new_deg[key] = out_deg[key]
        return out, new_deg

    # Back-compatible private aliases.  `factor_multiply` / `degree_order` /
    # `node_pairing` were promoted so `fs_builder.py` could reuse the arithmetic
    # instead of restating it; callers written against the private names —
    # `factor_order_search.py` among them — keep working through these.
    _charge_multiply = factor_multiply
    _degree_order = degree_order

    # ---- output -----------------------------------------------------------

    def spectrum_generator(self) -> dict[Vec, HabiroElement]:
        """`S` keyed by **lattice charges**, exact, cone-truncated."""
        if self._S is None:
            self.run()
        return {self.charge(k): c for k, c in self._S.items()}

    def multiplicities(self) -> dict[Vec, dict[int, int]]:
        """`Ω` keyed by **lattice charges**.

        `Ω_γ = Σ_s a_s χ_s` records which BPS multiplets the construction placed
        at `γ`: `spin_decompose` turns it into `{2s: a_s}`.  Read it as spectrum
        content only with the order in mind — the *content* is
        order-dependent even though `S` is not (86–91 factors under a linear
        central charge against 150–180 under free insertion, same `S`).
        """
        if self._S is None:
            self.run()
        return {self.charge(k): dict(om) for k, om in self.omega.items()}

    def negative_multiplicities(self) -> list[tuple[Vec, int, int]]:
        """`(charge, 2s, a_s)` for every negative multiplicity placed.

        Not an error: negative `a_s` occur and the element is still in the
        integral palindromic lattice — the lattice condition survives what
        positivity does not.  They are a
        property of the ORDER, not of `S`.
        """
        return [(self.charge(k), two_s, a)
                for k, om in self.omega.items()
                for two_s, a in spin_decompose(om).items() if a < 0]

    def verify_leading_data(self) -> list[tuple[Vec, str]]:
        """Check the built `S` against the leading data it was built from.

        Returns the violations (empty = pass): no `𝖖^{≤0}` part off the origin,
        `1` at the origin, and `𝖖¹`-coefficient equal to the target.  Since the
        factorisation's surjectivity is conjectural this is the honest
        acceptance check on the output, not a formality.
        """
        if self._S is None:
            self.run()
        zero = tuple([0] * self.rank)
        bad: list[tuple[Vec, str]] = []
        for k, h in self._S.items():
            f = expansion(h)
            if any(e < 0 and c for e, c in f.items()):
                bad.append((self.charge(k), "negative power of q"))
            elif k == zero:
                if f.get(0, 0) != 1:
                    bad.append((self.charge(k), "q^0 != 1 at the origin"))
            elif f.get(0, 0):
                bad.append((self.charge(k), "nonzero q^0"))
            elif f.get(1, 0) != self.target(k):
                bad.append((self.charge(k), "q^1 != leading data"))
        return bad


PLACEMENT_ORDERS = ("strip", "random", "key", "degree-key",
              "lex", "phase", "degree-phase", "component")



def expansion(h: HabiroElement) -> dict[int, int]:
    """`{exponent: coefficient}` of `h` for every exponent `≤ 1`, exactly.

    One `expand(1)` per charge.  The prototype called `coefficient(e)` once per
    exponent, and each such call re-walks the series-division recurrence from
    `k_min`, so reading a window of width `n` cost `O(n²)` instead of `O(n)`.
    `expand` keeps negative exponents (`expand_to_power_series` does not — and
    negative exponents are exactly what a bracket `≤ −2` produces, so dropping
    them would make the leading condition unenforceable precisely where it
    bites).
    """
    if h.is_zero():
        return {}
    return {e: c for e, c in h.expand(1)._coeffs.items() if c}



_expansion = expansion   # back-compatible alias (see the note in the class)

# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def build_spectrum_generator_from_factors(
    pairing: Sequence[Sequence[int]],
    node_charges: Sequence[Sequence[int]],
    cutoff: int,
    *,
    leading_data: Callable[[Vec], int] | None = None,
    order: str | None = None,
    phases: Sequence[complex] | None = None,
    order_key: Callable[[Vec], object] | None = None,
    piece_key: Callable[[Vec, int], object] | None = None,
    with_multiplicities: bool = False,
):
    """`S = {γ: HabiroElement}` of the BPS quiver `(pairing, node_charges)`,
    built from its leading data by the factor recursion.

    No spec, no green-sequence BFS, no `F`-solve, and **no honest-fail mode**:
    unlike the retired peel engine (`recursive_spectrum.build_spectrum_generator`)
    there is no monomial-charge gate to trip, so quivers with no finite
    `E_𝖖`-product — N=2\\*/Markov — build here.  Truncated only along the positive
    cone (`deg ≤ cutoff`).

    The order is any total order on the pairs `(γ, s)`; pass it as `piece_key`
    (the general form), `order_key` (charge level) or `phases` (a linear central
    charge).  **`order` now defaults to `None` = the `BPSFactorSpectrum` default**, which
    is phase-free (`"strip"` on an acyclic quiver, the component order `"component"`
    on a cyclic quiver that is not strongly connected, `"random"` on a strongly
    connected one).  It used
    to default to `"phase"`, which synthesised a generic central charge — that
    contradicted the 2026-08-13 ruling that no central charge is used unless one is
    supplied, and the ruling had reached `BPSFactorSpectrum` but not this wrapper.  `S`
    is unaffected (it is order-independent); the **factor content `Ω`** returned by
    `with_multiplicities=True` moves, so a caller reading `Ω` as a chamber's BPS
    spectrum must now pass `phases=` explicitly, which is what asks for a chamber.

    With `with_multiplicities=True` returns `(S, Ω)`, where `Ω` maps a charge to
    its palindromic multiplicity `{exponent: coefficient}`; `spin_decompose`
    turns one into `{2s: a_s}`.
    """
    builder = BPSFactorSpectrum(pairing, node_charges, cutoff,
                          leading_data=leading_data, order=order,
                          phases=phases, order_key=order_key,
                          piece_key=piece_key)
    builder.run()
    S = builder.spectrum_generator()
    if with_multiplicities:
        return S, builder.multiplicities()
    return S


def _main() -> None:
    def three_cycle(a, b, c):
        return [[0, a, -c], [-a, 0, b], [c, -b, 0]]

    b2 = [(1, 0), (0, 1)]
    b3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    print("pentagon, degree 6 — two central-charge orders, one element")
    _, om = build_spectrum_generator_from_factors(
        [[0, 1], [-1, 0]], b2, 6, order="phase",
        phases=[complex(1, 1), complex(-1, 1)], with_multiplicities=True)
    print("   ", {g: omega_string(o) for g, o in om.items()})

    print("pure SU(2), degree 6 — the physical spectrum")
    _, om = build_spectrum_generator_from_factors(
        [[0, 2], [-2, 0]], b2, 6, order="phase",
        phases=[complex(-1, 1), complex(1, 1)], with_multiplicities=True)
    print("   ", {g: omega_string(o) for g, o in sorted(om.items())})

    print("Markov (2,2,2), degree 4 — no finite E_q product exists")
    builder = BPSFactorSpectrum(three_cycle(2, 2, 2), b3, 4)
    builder.run()
    print(f"    {len(builder.omega)} factors, "
          f"{len(builder.spectrum_generator())} cone charges, "
          f"leading-data violations {len(builder.verify_leading_data())}")


if __name__ == "__main__":
    _main()
