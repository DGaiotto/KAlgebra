"""Closed-form pure-U(N) canonical-basis build + multiply.

This is the **PRIMARY pure-U(N) engine**: ``CanonicalBasis.ensure`` builds via
``cf_build_engine`` (+ chart-ρ), and ``CanonicalBasis.multiply`` is
``multiply_exact``.  The M-test engine in ``pure_un_kalgebra``
(``_build_clean`` / ``_multiply_mtest`` / ``decompose`` / ``_peel_to``) is
**off the primary path** -- retained as a callable cross-validation oracle
(``test_closed_form_vs_engine`` / ``test_pure_un_engine_crosscheck``) AND as the
automatic **rare-case fallback**: the closed-form lowest-tropical-charge peel can
leave a norm>1 residue when a summand's leading orbit is hidden in the bubbling of
a larger-magnetic one (it RAISES there, never wrong), and ``ensure`` / ``multiply``
then fall back to the M-test covariant peel, which separates leading orbits
(deg_W=0) from bubbling (deg_W≥1).  The two are complementary (the closed-form
handles the dressed-monopole products the M-test HANGS on).

Why this is the engine
----------------------
Bar-invariance of a canonical cannot be certified from its rational-QT image
(the W-boson denominators are q-free, so q->q^{-1} is invisible).  The M-test peel worked
around this by *imposing* bar-covariance and gating on the bar-blind q^0
self-norm -- which silently masks wrong builds (PR #362 F-sector) and HANGS on
dressed-monopole products.

This module sidesteps the whole problem: every canonical is *assembled from the
closed-form bar-invariant generators* ``L_E`` / ``L_F`` / ``det`` / ``W``, so it
is bar-invariant **by construction** -- no image-based certificate needed.
Decomposition is closed-form **leading-Weyl-orbit recognition + subtract**, and
every step is checkable by **exact reconstruction** (``Sum C^c image(c) ==
product``), which is what catches the #362-style masking the M-test cannot.

The leading-orbit closed form::

    leading(L_{m,e}) = q^lambda * sum_{w in W/W_m} chi_e[Levi_m](w.v) * w(U_m^dom)
    U_m^dom = prod_k U_{phi_k}^{c_k}      (dominant minuscule atoms; they commute)

``build_key_from_ansatz`` reads this backwards: build the leading orbit from a
Kapustin ``(m, e)`` and read ``_lowest_charge`` -> the internal build key (the
on-demand ``(m,e)`` -> build translation used by ``PureUNKAlgebra._bk``).

Canonical building -- the constructive-build rule
-----------------------------------
A canonical may be built ONLY as a **polynomial in other basis elements**: a
PRODUCT of basis generators (dressed minuscules ``L_E``/``L_F``, ``det``, Wilson
``L_W``, cone monomials) minus already-built LOWER canonicals.  Such a build
provably stays in the linear span of the canonical basis.  A method that instead
SOLVES for the canonical (a ratio elimination, a least-squares, "impose
bar-covariance") can leave the span -- and there is NO way to guard it: the
orthonormality pairing and the M-test are defined *within* the span, so they are
meaningless on an off-span element (they pass on garbage).  By the theorem that
every canonical is a polynomial of dressed minuscules, the product-and-peel
builders below (``_cf_monopole_cone_peel``, ``_cf_wr_cone_peel``) suffice in
principle; where a particular closed form does not yet reach, ``cf_build_engine``
returns ``None`` (honest failure) rather than fabricate.  (The ``_cf_joint_fiber``
solve was removed 2026-06-09 for violating this -- it emitted non-canonical
elements and spurious NEGATIVE structure constants.)
"""

from __future__ import annotations

import itertools

from abelianized_torus import DOp, VRational, VLaurent
from laurent_poly import LaurentPoly
from pure_un_kalgebra import L_E, L_det, L_W, _lowest_charge

from abelianized_torus import simplify_dop as _simp


# ---------------------------------------------------------------------------
# Weyl action: permute v- and u-indices by sigma in S_N
# ---------------------------------------------------------------------------

def _perm_vr(vr: VRational, sig, N: int) -> VRational:
    """Permute the v-indices of a VRational by ``sig`` (sig[i] = new slot of i)."""
    num = {}
    for ve, lp in vr.num._terms.items():
        ne = [0] * N
        for i, e in enumerate(ve):
            ne[sig[i]] = e
        num[tuple(ne)] = lp
    den = {}
    for (i, j, m), mu in vr.den.items():
        den[(sig[i], sig[j], m)] = den.get((sig[i], sig[j], m), 0) + mu
    sq = {}
    for (i, m), mu in vr.sq.items():
        sq[(sig[i], m)] = sq.get((sig[i], m), 0) + mu
    return VRational(VLaurent(num, n=N), den, n=N, sq=sq)


def perm_dop(D: DOp, sig, N: int) -> DOp:
    """Apply the Weyl element ``sig`` to a whole DOp (permute u- and v-indices)."""
    out = None
    for p, coeff in D.items():
        np_ = [0] * N
        for i, e in enumerate(p):
            np_[sig[i]] = e
        t = DOp({tuple(np_): _perm_vr(coeff, sig, N)}, n=N, n_gauge=N)
        out = t if out is None else out + t
    return out if out is not None else DOp.zero(N)


# ---------------------------------------------------------------------------
# Dominant cone atom  U_mu^dom = prod_k U_{phi_k}^{c_k} (E + det sector)
# ---------------------------------------------------------------------------

def _min_atom_dom(k: int, N: int) -> DOp:
    """Dominant minuscule atom = the highest-u-power summand of ``L_E(k, 0)``
    (the dominant-chamber Weyl piece of the bare minuscule monopole)."""
    D = L_E(k, (0,) * N, N)
    p = max(D.items(), key=lambda kv: kv[0])
    return DOp({p[0]: p[1]}, n=N, n_gauge=N)


def dom_atom(mu, N: int) -> DOp:
    """Dominant cone atom for a *dominant* cocharacter ``mu`` (decreasing).

    ``mu = sum_{k<N} c_k phi_k + c_N det`` with ``c_k = mu_{k-1}-mu_k`` and
    ``c_N = mu_{N-1}``; the atom is the (q-commuting) product of the dominant
    minuscule atoms.  **E + det sector only** (``c_N >= 0``); F-sector TODO."""
    mu = list(mu)
    c = [mu[k - 1] - mu[k] for k in range(1, N)]
    cN = mu[N - 1]                       # det power (dominant min; may be < 0)
    U = DOp.one(N)
    for k in range(1, N):
        for _ in range(c[k - 1]):
            U = U * _min_atom_dom(k, N)
    if cN != 0:
        U = U * L_det(cN, N)             # det^{cN}, either sign
    return _simp(U)


def leading_orbit(mu, N: int, char_vr: VRational | None = None) -> DOp:
    """Closed-form leading Weyl orbit ``sum_{w in W/W_mu} chi(w.v) w(U_mu^dom)``.

    ``char_vr`` is the dressing character at the dominant chamber (``None`` =
    trivial, ``chi=1``).  Dedup over ``W/W_mu`` by distinct u-power (the atom is
    ``W_mu``-invariant when ``char`` is trivial).  Dressing dedup is exact only
    for ``char=None`` here; the dressed case is handled by the caller."""
    A = dom_atom(mu, N)
    seen: dict = {}
    for sig in set(itertools.permutations(range(N))):
        Aw = perm_dop(A, sig, N)
        cw = _perm_vr(char_vr, sig, N) if char_vr is not None else None
        for p, coeff in Aw.items():
            term = coeff if cw is None else (cw * coeff)
            if p not in seen:
                seen[p] = term
    return DOp({k: v for k, v in seen.items() if not v.is_zero()}, n=N, n_gauge=N)


# ---------------------------------------------------------------------------
# Dressing helpers (Levi blocks, coweight decomposition, character read)
# ---------------------------------------------------------------------------

def _coweight_decomp(m, N):
    """m anti-dominant -> (a[0..N-2], p): m = sum_k a_k phi_k^antidom + p*det."""
    m = list(m)
    p = m[0]
    mp = [x - p for x in m]
    a = [mp[N - k] - mp[N - k - 1] for k in range(1, N)]
    return a, p


def _is_minuscule(m, N):
    """Return ``(k, p)`` if anti-dominant ``m`` is ``phi_k + p*det`` (a single
    minuscule, optionally det-shifted), else ``(None, None)``."""
    a, p = _coweight_decomp(m, N)
    nz = [k for k in range(1, N) if a[k - 1] != 0]
    if len(nz) == 1 and a[nz[0] - 1] == 1:
        return nz[0], p
    return None, None


def _levi_blocks(m):
    """Index runs of equal value in anti-dominant ``m`` (the Levi block structure)."""
    m = list(m)
    blocks = []
    start = 0
    for i in range(1, len(m) + 1):
        if i == len(m) or m[i] != m[start]:
            blocks.append(list(range(start, i)))
            start = i
    return blocks


def _div_num_by_monomial(num: VLaurent, mono: VLaurent, N: int) -> VLaurent | None:
    """Divide a VLaurent by a single-term VLaurent (a v-monomial)."""
    items = list(mono._terms.items())
    if len(items) != 1:
        return None
    (mve, mlp), = items
    mc = list(mlp._coeffs.items())
    if len(mc) != 1:
        return None
    (meq, mev), = mc
    out = {}
    for ve, lp in num._terms.items():
        nve = tuple(ve[i] - mve[i] for i in range(N))
        nlp = {}
        for e, c in lp._coeffs.items():
            if c % mev != 0:
                return None
            nlp[e - meq] = c // mev
        out[nve] = LaurentPoly(nlp)
    return VLaurent(out, n=N)


def _dom_weight(num: VLaurent, blocks, N):
    """Highest (dominant) v-monomial of a VLaurent, returned as a v-exponent
    tuple with each Levi block sorted descending (the dressing ``e``)."""
    def key(ve):
        return tuple(tuple(sorted((ve[i] for i in blk), reverse=True)) for blk in blocks)
    ve = max(num._terms, key=key)
    out = [0] * N
    for blk in blocks:
        vals = sorted((ve[i] for i in blk), reverse=True)
        for i, v in zip(blk, vals):
            out[i] = v
    return tuple(out)


# ---------------------------------------------------------------------------
# Canonical build  (e=0 pure-E + det sector; minuscule dressed)
# ---------------------------------------------------------------------------

def cf_build(m, e, N: int) -> DOp | None:
    """Full canonical ``L_{m,e}`` as a closed-form generator product.

    In scope: trivial dressing (any ``m``, the cone monomial x det^p), and the
    **minuscule dressed** case (``m = phi_k``, ``L_E(k, e)``).  Returns ``None``
    for out-of-scope (det-shifted or non-minuscule dressed -- the Wilson x cone
    peel, TODO)."""
    m = tuple(m)
    e = tuple(e) if not isinstance(e, int) else (e,) * N
    if any(x != 0 for x in e):
        k, p = _is_minuscule(m, N)
        if k is not None and p == 0:
            # pure minuscule phi_k, dressed: the closed-form dressed minuscule.
            # e is read at the dominant chamber (J-block first), matching L_E's
            # electric convention electric[:k] = J-side.
            return _simp(L_E(k, e, N))
        return None                       # det-shifted / non-minuscule dressed: TODO
    # Bare cone monomial.  For ANY anti-dominant m the phi_k coefficients are
    # >= 0 (m increasing); the only sign is the det power c = m_0, so the
    # F-sector (m_0 < 0) is just E-cone x det^{-|m_0|} -- no F-minuscule atoms.
    m = list(m)
    p = m[0]                              # det power (anti-dominant min; may be < 0)
    mp = [x - p for x in m]
    c = [mp[N - k] - mp[N - k - 1] for k in range(1, N)]   # phi_k^antidom coeffs >= 0
    L = DOp.one(N)
    for k in range(1, N):
        for _ in range(c[k - 1]):
            L = L * L_E(k, (0,) * N, N)
    if p != 0:
        L = L * L_det(p, N)               # det^p, p of either sign
    return _simp(L)


# ---------------------------------------------------------------------------
# Decompose / multiply by leading-orbit recognition
# ---------------------------------------------------------------------------

def _scalar_ratio(cD: VRational, cL: VRational, N: int) -> LaurentPoly | None:
    """If ``cD == C(q) * cL`` for a q-Laurent scalar ``C``, return ``C`` else None."""
    cD = cD.simplify()
    cL = cL.simplify()
    lnum = cL.num._terms
    if not lnum:
        return None
    # pick a v-monomial present in cL's numerator; read the q-ratio there
    ve, lp_L = next(iter(lnum.items()))
    lp_D = cD.num._terms.get(ve)
    if lp_D is None:
        # denominators may differ by a v-shift; fall back to cross-multiplication test
        lp_D = LaurentPoly.zero()
    # C = lp_D / lp_L  (must be exact monomial division -> q-Laurent)
    C = _lp_divide(lp_D, lp_L)
    if C is None:
        return None
    # verify cD == C * cL exactly
    Cv = VRational.from_scalar(C, n=N)
    if (cD + (Cv * cL) * (-1)).simplify().is_zero():
        return C
    return None


def _lp_divide(a: LaurentPoly, b: LaurentPoly) -> LaurentPoly | None:
    """Exact division a/b when both are single-... general Laurent division by
    matching: returns C with a == C*b, else None.  Implemented via the leading
    terms + verification."""
    bc = b._coeffs
    if not bc:
        return None
    ac = a._coeffs
    if not ac:
        return LaurentPoly.zero()
    # shift so that the lowest exponent of b is 0, deduce C as a*b^{-1} only if
    # b is a monomial; otherwise attempt polynomial division.
    if len(bc) == 1:
        (eb, vb), = bc.items()
        if any(v % vb != 0 for v in ac.values()):
            return None
        return LaurentPoly({e - eb: v // vb for e, v in ac.items()})
    # general: try C = a // b via long division on exponents
    from fractions import Fraction
    rem = dict(ac)
    quot: dict = {}
    blo = min(bc)
    bhi = max(bc)
    guard = 0
    while rem:
        guard += 1
        if guard > 10000:
            return None
        rlo = min(rem)
        ce = rlo - blo
        # leading coeff division
        if rem[rlo] % bc[blo] != 0:
            return None
        cc = rem[rlo] // bc[blo]
        quot[ce] = quot.get(ce, 0) + cc
        for be, bv in bc.items():
            rem[ce + be] = rem.get(ce + be, 0) - cc * bv
        rem = {e: v for e, v in rem.items() if v != 0}
    return LaurentPoly(quot)


def _dominance_key(p):
    """Sort key picking the most-dominant magnetic charge (descending sort)."""
    return tuple(sorted(p, reverse=True))


def cf_decompose(D: DOp, N: int, max_terms: int = 5000) -> dict:
    """Express ``D`` in the canonical basis: ``{(m, e): C(q)}``.

    Leading-Weyl-orbit recognition: peel the most-dominant u-power, identify the
    canonical ``L_{m,e}`` (closed form), read off the exact q-Laurent structure
    constant, subtract the *full* canonical, recurse.  Raises on out-of-scope
    targets (so coverage gaps fail loud, never silently)."""
    out: dict = {}
    D = _simp(D)
    for _ in range(max_terms):
        if D.is_zero():
            return out
        p_lead = max(D._terms, key=_dominance_key)
        m_dom = tuple(sorted(p_lead, reverse=True))   # dominant chamber
        m = tuple(sorted(p_lead))                      # anti-dominant label
        # --- dressing read at the dominant chamber ---
        cD = D._terms[m_dom].simplify()
        cA = dom_atom(m_dom, N)._terms[m_dom].simplify()
        if cD.den != cA.den or cD.sq != cA.sq:
            raise RuntimeError(f"cf_decompose: denom mismatch at {m_dom}")
        ratio = _div_num_by_monomial(cD.num, cA.num, N)   # = sum_e c_e chi_e(v)
        if ratio is None:
            raise RuntimeError(f"cf_decompose: dom_atom numerator not a monomial at {m_dom}")
        blocks = _levi_blocks(m_dom)
        e = _dom_weight(ratio, blocks, N)              # highest dressing present
        L = cf_build(m, e, N)
        if L is None:
            raise NotImplementedError(
                f"cf_decompose: L_{(m, e)} out of scope (det-shifted/non-min dressed)")
        # structure constant read at the top (u,v) monomial of this character
        cAmono = next(iter(cA.num._terms))
        top_v = tuple(e[i] + cAmono[i] for i in range(N))
        cL = L._terms[m_dom].simplify()
        lpD = cD.num._terms.get(top_v)
        lpL = cL.num._terms.get(top_v)
        if lpD is None or lpL is None:
            raise RuntimeError(f"cf_decompose: top monomial {top_v} missing at {m_dom}")
        Cq = _lp_divide(lpD, lpL)
        if Cq is None:
            raise RuntimeError("cf_decompose: structure constant is not a q-Laurent scalar")
        out[(m, e)] = out.get((m, e), LaurentPoly.zero()) + Cq
        Cdop = DOp.from_scalar(VRational.from_scalar(Cq, n=N), n=N)
        D = _simp(D + (Cdop * L) * (-1))
    raise RuntimeError("cf_decompose: did not terminate")


def cf_multiply(image_a: DOp, image_b: DOp, N: int) -> dict:
    """Structure constants ``C^c_{ab}(q)`` via closed-form decomposition."""
    return cf_decompose(_simp(image_a * image_b), N)


# ===========================================================================
# Leading-magnetic multiply: Levi-character decompose + palindromic centering
# (bar-free; gives structure constants at the leading magnetic with no build
#  and no bar certificate -- the masking failure mode cannot occur here.)
# ===========================================================================

def _schur(lam, idx, N):
    """Schur polynomial s_lam in the variables ``idx`` (a list of v-indices),
    as a VLaurent over all N variables.  Recursive GL_n -> GL_{n-1} interlacing."""
    lam = list(lam)
    if len(idx) == 1:
        ve = [0] * N
        ve[idx[0]] = lam[0] if lam else 0
        return VLaurent({tuple(ve): LaurentPoly({0: 1})}, n=N)
    out = VLaurent.zero(N)
    last = idx[-1]
    # mu interlaces lam:  lam_0 >= mu_0 >= lam_1 >= mu_1 >= ... (mu length len-1)
    def gen(i, lo):
        if i == len(lam) - 1:
            yield []
            return
        hi = lam[i]
        floor = lam[i + 1]
        for mval in range(floor, hi + 1):
            for rest in gen(i + 1, mval):
                yield [mval] + rest
    for mu in gen(0, 0):
        sub = _schur(mu, idx[:-1], N)
        power = sum(lam) - sum(mu)
        # multiply sub by x_last^power: shift v-exponent
        terms = {}
        for ve, lp in sub._terms.items():
            nve = list(ve); nve[last] += power
            terms[tuple(nve)] = lp
        out = out + VLaurent(terms, n=N)
    return out


def _levi_decompose(ratio: VLaurent, blocks, N):
    """Decompose a W_m-symmetric VLaurent into Levi Schur characters.
    Returns ``[(e_tuple, coeff_LaurentPoly)]`` (e = dominant weight per block)."""
    work = {ve: dict(lp._coeffs) for ve, lp in ratio._terms.items()}
    out = []
    guard = 0
    while any(any(v for v in lp.values()) for lp in work.values()):
        guard += 1
        if guard > 2000:
            raise RuntimeError("_levi_decompose: no termination")
        def key(ve):
            return tuple(tuple(sorted((ve[i] for i in blk), reverse=True)) for blk in blocks)
        top = max((ve for ve, lp in work.items() if any(lp.values())), key=key)
        # e = dominant weight per block
        e = [0] * N
        for blk in blocks:
            vals = sorted((top[i] for i in blk), reverse=True)
            for i, v in zip(blk, vals):
                e[i] = v
        e = tuple(e)
        coeff = dict(work[top])
        out.append((e, LaurentPoly(coeff)))
        # subtract coeff * prod_blocks s_{e|block}
        char = None
        for blk in blocks:
            lam = sorted((e[i] for i in blk), reverse=True)
            sb = _schur(lam, blk, N)
            char = sb if char is None else _vl_mul(char, sb, N)
        for ve, lp in char._terms.items():
            for eq, cc in lp._coeffs.items():
                for ce, cv in coeff.items():
                    cur = work.get(ve, {})
                    cur[eq + ce] = cur.get(eq + ce, 0) - cc * cv
                    work[ve] = cur
        work = {ve: {e2: v for e2, v in lp.items() if v} for ve, lp in work.items()}
    return out


def _vl_mul(a: VLaurent, b: VLaurent, N):
    out = {}
    for ve1, lp1 in a._terms.items():
        for ve2, lp2 in b._terms.items():
            nve = tuple(ve1[i] + ve2[i] for i in range(N))
            prod = lp1 * lp2
            cur = out.get(nve)
            out[nve] = prod if cur is None else cur + prod
    return VLaurent({k: v for k, v in out.items() if not v.is_zero()}, n=N)


def levi_character(m_dom, e, N):
    """The Levi-irrep character `χ_e[L_{m_dom}]` as a VLaurent: the product of
    the per-Levi-block Schur polynomials `s_{e|block}` (each block sorted
    descending = its dominant weight).  `m_dom` dominant (descending)."""
    char = None
    for blk in _levi_blocks(m_dom):
        lam = sorted((e[i] for i in blk), reverse=True)
        sb = _schur(lam, blk, N)
        char = sb if char is None else _vl_mul(char, sb, N)
    return char


def build_key_from_ansatz(m, e, N):
    """Translate a Kapustin ansatz label `(m, e)` (dominant m, Levi-dominant e)
    to the internal build key `_lowest_charge` -- WITHOUT a full build.

    Builds only the leading Weyl orbit `Σ_{w∈W/W_m} χ_e(w·v)·w(U_m^dom)` (closed
    form), then reads `_lowest_charge`: the anti-dominant u_bot block is the
    most-anti-dominant extreme of that orbit, so its lowest-q v-weight -- the
    build key -- is already determined by the leading orbit (no bubbling needed).
    Verified to round-trip the full N=2,3 registered bijection."""
    m_dom = tuple(sorted(m, reverse=True))
    char = levi_character(m_dom, tuple(e), N)
    O = leading_orbit(m_dom, N, char_vr=VRational.from_vlaurent(char))
    return _lowest_charge(O, N)


def _center_palindromic(lp: LaurentPoly):
    """Return (C_centered, shift) where C = q^{-shift}*lp is palindromic, or
    (None, None) if lp's support is not symmetric after integer centering."""
    es = list(lp._coeffs)
    if not es:
        return LaurentPoly.zero(), 0
    lo, hi = min(es), max(es)
    if (lo + hi) % 2 != 0:
        return None, None
    shift = (lo + hi) // 2
    cen = LaurentPoly({e - shift: v for e, v in lp._coeffs.items()})
    if all(cen._coeffs.get(e, 0) == cen._coeffs.get(-e, 0) for e in cen._coeffs):
        return cen, shift
    return None, None


def cf_multiply_leading(image_a: DOp, image_b: DOp, N: int):
    """Structure constants at the LEADING magnetic charge of image_a * image_b,
    via Levi-character decomposition + palindromic centering.  Bar-free; returns
    ``{(m, e): C(q)}`` for the top magnetic.  (Descent to lower magnetic needs
    the W_R-cone build; not done here.)"""
    D = _simp(image_a * image_b)
    if D.is_zero():
        return {}
    p_lead = max(D._terms, key=_dominance_key)
    m_dom = tuple(sorted(p_lead, reverse=True))
    m = tuple(sorted(p_lead))
    cD = D._terms[m_dom].simplify()
    cA = dom_atom(m_dom, N)._terms[m_dom].simplify()
    if cD.den != cA.den or cD.sq != cA.sq:
        raise RuntimeError("cf_multiply_leading: denom mismatch")
    ratio = _div_num_by_monomial(cD.num, cA.num, N)
    blocks = _levi_blocks(m_dom)
    out = {}
    for e, c_e in _levi_decompose(ratio, blocks, N):
        C, shift = _center_palindromic(c_e)
        if C is None:
            raise RuntimeError(f"cf_multiply_leading: non-palindromic centering for e={e}: {dict(c_e._coeffs)}")
        out[(m, e)] = C
    return out


def _levi_character(e, blocks, N) -> VRational:
    """The Levi character chi_e = prod_blocks s_{e|block}(v_block) as a VRational."""
    char = None
    for blk in blocks:
        lam = sorted((e[i] for i in blk), reverse=True)
        sb = _schur(lam, blk, N)
        char = sb if char is None else _vl_mul(char, sb, N)
    if char is None:
        return VRational.one(N)
    return VRational.from_vlaurent(char)


def cf_leading(m, e, N):
    """LEADING-MAGNETIC builder: the predicted leading Weyl orbit of L_{m,e},
    ``sum_{w in W/W_m} chi_e[Levi_m](w.v) * w(U_m^dom)`` (up to the uniform q^lambda
    centering -- which the multiply fixes by palindromicity, lambda = <omega_top,e>)."""
    m = tuple(m); e = tuple(e)
    m_dom = tuple(sorted(m, reverse=True))
    blocks = _levi_blocks(m_dom)
    char = _levi_character(e, blocks, N)
    return leading_orbit(m_dom, N, char_vr=char)


# ===========================================================================
# Full recursive build (W_R.cone) + full multiply
#   decompose: leading u^m -> recognize q^? L_{m,e} -> subtract FULL L_{m,e} -> repeat
#   build:     L_{m,e} = W_R.cone - (lower canonicals, recursively built)
# ===========================================================================

_CACHE: dict = {}
_BUILDING: set = set()


def _recognize_leading(P, N):
    """At the leading (most-dominant) u-power of P, return (m, e, m_dom, cAmono)
    where e is the highest Levi dressing and cAmono is dom_atom's leading v-monomial."""
    p_lead = max(P._terms, key=_dominance_key)
    m_dom = tuple(sorted(p_lead, reverse=True))
    m = tuple(sorted(p_lead))
    cD = P._terms[m_dom].simplify()
    cA = dom_atom(m_dom, N)._terms[m_dom].simplify()
    if cD.den != cA.den or cD.sq != cA.sq:
        raise RuntimeError(f"_recognize_leading: denom mismatch at {m_dom} (tail contamination?)")
    ratio = _div_num_by_monomial(cD.num, cA.num, N)
    blocks = _levi_blocks(m_dom)
    e, _c = _levi_decompose(ratio, blocks, N)[0]      # highest dressing
    cAmono = next(iter(cA.num._terms))
    return m, e, m_dom, cAmono


def _struct_const(P, Lc, m_dom, e, cAmono, N):
    """Structure constant C(q): P's coeff / Lc's coeff at the top (u,v) monomial."""
    top_v = tuple(e[i] + cAmono[i] for i in range(N))
    cP = P._terms[m_dom].simplify().num._terms.get(top_v)
    cL = Lc._terms[m_dom].simplify().num._terms.get(top_v)
    if cP is None or cL is None:
        raise RuntimeError("_struct_const: top monomial missing")
    C = _lp_divide(cP, cL)
    if C is None:
        raise RuntimeError("_struct_const: not a q-Laurent scalar")
    return C


def cf_build_full(m, e, N, _depth=0):
    """Full canonical L_{m,e} via recursive W_R.cone.  *** WIP / incomplete ***

    Two structural gaps remain (single-ordering version below returns None on them):
      (1) q-PREFACTOR.  The centering q^lambda of each peeled summand must be
          pinned by BOTH orderings L_a.L_b and L_b.L_a (bar-covariance
          C(q)=C(q^-1)); the single-ordering peel here cannot fix it.
      (2) R-SELECTION / unitriangular ordering.  A dressing that is not gl_N-
          dominant (e.g. (0,2,0) at magnetic (0,0,2): block-0 value < block-1
          value) is never the highest weight of any chi_R, so it cannot LEAD a
          W_R.cone seed -- it must be built as a lower branch, peeling the higher
          dressings at the same magnetic first (a joint unitriangular solve).
    The bar-free LEADING-magnetic multiply (cf_multiply_leading) does not need
    this and is the working deliverable."""
    m = tuple(m); e = tuple(e); key = (m, e, N)
    if key in _CACHE:
        return _CACHE[key]
    if all(x == 0 for x in e):
        L = cf_build(m, (0,) * N, N)               # bare cone x det^p
        _CACHE[key] = L; return L
    k, p = _is_minuscule(m, N)
    if k is not None and p == 0:
        L = _simp(L_E(k, e, N))                     # dressed minuscule
        _CACHE[key] = L; return L
    if _depth > 40 or key in _BUILDING:
        return None
    _BUILDING.add(key)
    try:
        # seed = W_R . (bare cone),  R = e at the dominant chamber (highest dressing)
        R = tuple(e)
        bare = cf_build(m, (0,) * N, N)
        seed = _simp(L_W(R, N) * bare)
        # strip the target's leading orbit (CENTERED, using its actual seed coeff),
        # then recursively subtract the full lower canonicals; L = seed - sum(lower).
        clead = cf_leading(m, e, N)
        m_dom0 = tuple(sorted(m, reverse=True))
        cA0 = dom_atom(m_dom0, N)._terms[m_dom0].simplify()
        cAmono0 = next(iter(cA0.num._terms))
        top0 = tuple(e[i] + cAmono0[i] for i in range(N))
        fac = _lp_divide(seed._terms[m_dom0].simplify().num._terms[top0],
                         clead._terms[m_dom0].simplify().num._terms[top0])
        facd = DOp.from_scalar(VRational.from_scalar(fac, n=N), n=N)
        P = _simp(seed + (facd * clead) * (-1))
        L = seed
        for _ in range(4000):
            if P.is_zero():
                break
            mm, ee, mdom, cAmono = _recognize_leading(P, N)
            if (mm, ee) == (m, e):
                break                                   # target tail; remaining stays in L
            Lc = cf_build_full(mm, ee, N, _depth + 1)
            if Lc is None:
                return None
            C = _struct_const(P, Lc, mdom, ee, cAmono, N)
            Cd = DOp.from_scalar(VRational.from_scalar(C, n=N), n=N)
            P = _simp(P + (Cd * Lc) * (-1))
            L = _simp(L + (Cd * Lc) * (-1))
    finally:
        _BUILDING.discard(key)
    _CACHE[key] = L
    return L


def cf_multiply(image_a, image_b, N):
    """Full structure constants of L_a . L_b via recognize-and-subtract."""
    P = _simp(image_a * image_b)
    out = {}
    for _ in range(8000):
        if P.is_zero():
            return out
        m, e, m_dom, cAmono = _recognize_leading(P, N)
        Lc = cf_build_full(m, e, N)
        if Lc is None:
            raise NotImplementedError(f"cf_multiply: cannot build L_{(m, e)}")
        C = _struct_const(P, Lc, m_dom, e, cAmono, N)
        out[(m, e)] = out.get((m, e), LaurentPoly.zero()) + C
        Cd = DOp.from_scalar(VRational.from_scalar(C, n=N), n=N)
        P = _simp(P + (Cd * Lc) * (-1))
    raise RuntimeError("cf_multiply: did not terminate")


# ===========================================================================
# Engine-backed closed-form build:  QTCone (_cone_build) + W_R.QTCone bar-peel
#   -------------------------------------------------------------------------
#   Builds EVERY canonical fast and without hanging, using the engine `B` ONLY
#   for its CORRECT primitives -- the QTCone monomial build (`_cone_build`), the
#   covariant q-graded M-test orbit (`_cov_orbit`/`_mtest_mostneg`), the affine
#   dressing/label read (`_cov_label`) and the monomial ratio (`_monomial_ratio`)
#   -- NEVER `B.ensure`/`_build_clean` (the recursive W_R-cone M-test peel, which
#   hangs on e.g. ((0,0,2),(4,0,0))).  The lower canonicals revealed in the peel
#   are themselves built HERE (closed form, recursively), so the only "M-test" left
#   is the q-grading + label read; the q-prefactor is pinned by the bar-symmetric
#   two-ordering extraction (P == q^{2p}.Q), and the build is self-validating
#   (`_lowest_charge(L) == charge`).
# ===========================================================================

def _cf_is_min_mag(m):
    """Anti-dominant single-block minuscule magnetic (E-sector {0,1})."""
    return list(m) == sorted(m) and set(m) <= {0, 1} and sum(m) >= 1


def _cf_cov_q0(B, L):
    """The deg_W=0 covariant slice {(u,v): int} of L (its leading signed orbit)."""
    return {(u, v): c for (dw, u, v), c in B._cov_orbit(L).items() if dw == 0}


_CF_BUILDING: set = set()       # cycle guard for the recursive closed-form build
_CF_BUDGET = [0]                # per-top-level-build recursive-call budget (a
#                                 backstop against any residual fallback divergence)
_CF_MONOPOLE_MAXTRY = 48        # cap on monopole.cone candidate products tried.
#                                 The e' search is O(window^N) and a large `f` ->
#                                 a large L_E.cone product (slow), so we try the
#                                 SMALLEST-dressing candidates first (the convergent
#                                 ones) and bail to None past the cap rather than
#                                 grinding through big-f products (which else hang at
#                                 spread-out lam, e.g. N=4 m=(0,1,2,3) lam=(0,-1,2,4)).


# ---------------------------------------------------------------------------
# Bar-symmetric 2-member JOINT fiber solve (the N>=4 GL_k-block fix)
# ---------------------------------------------------------------------------
#
# At N>=4 a single fundamental/antifundamental Wilson box on the cone splits
# across TWO Levi U(n) blocks, so `W_box . cone` decomposes into a COUPLED pair
# of same-magnetic canonicals {target, co-summand}.  The co-summand is not
# gl_N-dominant, so it leads no seed of its own -- the recursive single-canonical
# peel diverges trying to build it (it marches away from the cone along the roots
# e_i - e_j).  The fix solves the members TOGETHER as a small linear system from
# the two seed orderings P = W_R . base and Q = base . W_R:
#
#     P = c_A L_A + c_B L_B,   Q = c'_A L_A + c'_B L_B
#
# (L_A, L_B the fiber's bar-fixed canonicals).  A canonical's NATIVE leading
# monomial is exactly its label `(m, lam)` (anti-dominant `m`, v = lam, lowest q;
# see `_lowest_charge`), and distinct members have DISJOINT leading monomials, so
# the structure-constant RATIO at member B's leading monomial reads off directly:
#
#     r_B = P[(m, lam_B)] / Q[(m, lam_B)] = c_B / c'_B
#
# (the leading-coefficient normalization cancels).  Then `P - r_B . Q` kills the
# L_B component, leaving `(c_A - r_B c'_A) . L_A` -- a SCALAR multiple of the
# target; divide by that scalar (= the residual's own value at `(m, lam_A)`,
# which cancels it exactly) and bar-recenter.  L_E-FREE; the per-magnetic leading
# sign is aligned to the bare cone's.
#
# At a k-BLOCK magnetic (first at N=5, e.g. m=(0,1,1,2,2): blocks {0},{1,2},{3,4})
# a single box raises EACH Levi block (`W_fund|_Levi = +_i std_i`), so the fiber
# has one member per block.  Members raising a size-1 (U(1)) block -- or otherwise
# cone / dominant-Wilson reachable -- build INDEPENDENTLY; we peel those out of
# P, Q first (bottom-up), reducing the fiber to its genuinely-coupled remainder
# (the size->=2 block raises, none gl_N-dominant).  At N=5 that remainder is <=2
# (a 5-row m has at most two U(2)+ blocks), so P, Q suffice and the 2-member ratio
# elimination above closes it.  A >2-member remainder (first at N=6) needs more
# seeds and is not handled yet.

def _cf_scl(lp, N):
    return DOp.from_scalar(VRational.from_scalar(lp, n=N), n=N)


def _cf_coeff(D, u, v):
    """q-Laurent coefficient of DOp ``D`` at the native monomial ``(u, v)`` (zero
    if absent)."""
    if u not in D._terms:
        return LaurentPoly.zero()
    return D._terms[u].simplify().num._terms.get(v, LaurentPoly.zero())


def _cf_divD(D, divisor, N):
    """Exact coefficient-wise division of DOp ``D`` by the q-Laurent ``divisor``
    (None if any coefficient is not divisible)."""
    out = None
    for u, vr in D._terms.items():
        vr = vr.simplify()
        nn = {}
        for v, lp in vr.num._terms.items():
            q = _lp_divide(lp, divisor)
            if q is None:
                return None
            if q._coeffs:
                nn[v] = q
        if nn:
            t = DOp({u: VRational(VLaurent(nn, n=N), vr.den, n=N, sq=vr.sq)},
                    n=N, n_gauge=N)
            out = t if out is None else out + t
    return _simp(out) if out is not None else DOp.zero(N)


def _cf_lowest_sign(B, L, N):
    """Sign (+-1) of ``L``'s leading coefficient at its lowest tropical charge."""
    c = B._coeff_at(L, _lowest_charge(L, N))._coeffs
    return 1 if (not c or c[min(c)] >= 0) else -1


def _cf_apply_sign(B, L, sigma, N):
    """Return ``L`` flipped so its lowest-charge leading coefficient has sign
    ``sigma`` (the fixed per-magnetic canonical convention)."""
    if L is None:
        return None
    return L if _cf_lowest_sign(B, L, N) == sigma else _simp(L * (-1))


# _cf_joint_fiber / _cf_eliminate / _cf_fiber_members -- REMOVED (2026-06-09).
#
# They built a same-magnetic GL_k-block dressing by a two-seed linear RATIO
# ELIMINATION (solving for the canonical), not by a polynomial in basis elements.
# That SOLVE can land OUTSIDE the linear span of the canonical basis, and -- the
# decisive point -- there is NO way to detect it afterwards: orthonormality and the
# M-test are pairings defined *within* the span, so they say nothing about an
# element that has left it (both PASSED on the defective joint-fiber output).  It
# produced non-canonical elements and spurious NEGATIVE structure constants
# (cross-checked against the trusted BPS pure-U(3) realisation).
#
# Constructive-build rule going forward: a canonical may be built ONLY as a polynomial in
# other basis elements (product of dressed minuscules / det / Wilson / cone, minus
# already-built lower canonicals) -- such a build provably stays in the span.  By
# the theorem that every canonical is a polynomial of dressed minuscules, the
# product-and-peel builders (4b `_cf_monopole_cone_peel`, 4c `_cf_wr_cone_peel`)
# suffice in principle.


def cf_build_engine(B, charge, cache=None, depth=0):
    """Closed-form canonical ``L_charge`` (DOp), fast and non-hanging.

    Path: QTCone monomial (`_cone_build`, correct by construction) -> dressed
    single minuscule (`L_E`) -> F-sector det-factoring -> ``W_R . base``
    bar-symmetric peel (two factorization families).  ``cache`` memoizes by charge
    across the recursion; ``_CF_BUILDING`` guards against re-entrant cycles.
    Returns the DOp or None (out of scope)."""
    N = B._N
    charge = (tuple(charge[0]), tuple(charge[1]))
    # Weyl-normalize: the Weyl group acts on (m, e) JOINTLY, so a non-anti-dominant
    # m is the same canonical viewed in a rotated frame.  Sort m ascending
    # (anti-dominant) and apply the SAME permutation to e -- the canonical's proper
    # build key.  (e.g. N=2 ((0,-1),(-1,0)) -> ((-1,0),(0,-1)); the cone/coweight
    # logic below assumes anti-dominant m, so without this a descending m fails.)
    m0 = charge[0]
    if list(m0) != sorted(m0):
        sig = sorted(range(N), key=lambda i: m0[i])
        charge = (tuple(m0[i] for i in sig), tuple(charge[1][i] for i in sig))
    if cache is None:
        cache = {}
    if depth == 0:
        _CF_BUDGET[0] = 600                          # reset per top-level build
        #                                              (modest raise from 250: the
        #                                              monopole.cone (4b) recursion
        #                                              needs headroom; still bounds
        #                                              the W_R-peel / N>=4 runaway)
    if charge in cache:
        return cache[charge]
    _CF_BUDGET[0] -= 1
    if _CF_BUDGET[0] < 0:                            # WIP: bound the N>=4 divergence
        return None
    # WIP fail-fast guard: at N>=4 the peel's _cov_label can MISLABEL a slice,
    # making the recursion chase absurd dressings (e.g. (-11,-12,12,12)).  A
    # genuine dressing spread is small; bail on a clearly-bogus one so the build
    # fails fast (returns None) instead of diverging.  (Remove once the N>=4
    # _cov_label recognition is fixed.)
    if max(charge[1]) - min(charge[1]) > 4 * N:
        return None
    m, lam = charge
    # O(N) fail-fast: a dressing that INCREASES inside an equal-m block is never a
    # dominant Levi irrep, so it labels no canonical -- reject before any (peel)
    # branch, so non-canonical labels stay fast (test_pure_un_cone_membership).
    for i in range(1, N):
        if m[i] == m[i - 1] and lam[i] > lam[i - 1]:
            return None
    cb = B._cone_build(charge)                       # 1. QTCone monomial
    if cb is not None and _lowest_charge(cb[0], N) == charge:
        cache[charge] = cb[0]
        return cb[0]
    if _cf_is_min_mag(m):                            # 2. dressed minuscule E_k
        L = _simp(L_E(sum(m), lam, N))
        if _lowest_charge(L, N) == charge:
            cache[charge] = L
            return L
    if depth > 24 or charge in _CF_BUILDING:
        return None
    _CF_BUILDING.add(charge)
    try:
        p = m[0]                                        # anti-dominant: m[0] = min entry
        L = None
        if p != 0:
            # 3. det^p shift (GENERAL): det^p is central + invertible, so it shifts
            #    m by p.1 with a known q-power -- L_{(m,e)} = q^c . L_{(m-p.1, e)} . det^p
            #    -- reducing the target to the m[0]=0 core, then recentering (no peel).
            #    For p<0 (F-sector) this is the only path -- the cap=0 covariant M-test
            #    sees only q<=0 so the W_R.base peel is oriented for q>=0 (E-sector)
            #    targets and a q<0 target blocks it.  For p>0 (e.g. central m=(k,..,k),
            #    k>0) it routes the build down to the m[0]=0 / Wilson core cheaply,
            #    avoiding the deep W_R.cone recursion (and its budget/depth blow-up
            #    that otherwise starves the dependency builds higher up).
            core = cf_build_engine(B, (tuple(x - p for x in m), lam), cache, depth + 1)
            if core is not None:
                cand = B._cone_recenter(_simp(core * L_det(p, N)))
                if cand is not None and _lowest_charge(cand, N) == charge:
                    L = cand
        if L is None and p >= 0:
            # Constructive-build rule: a canonical may be built ONLY as a polynomial in other
            # basis elements -- a PRODUCT of basis generators (dressed minuscules
            # `L_E`/`L_F`, `det`, Wilson `W`, cone monomials) minus already-built
            # (in-span) lower canonicals, which provably stays in the linear span.
            # A method that instead SOLVES a linear system (the REMOVED
            # `_cf_joint_fiber` 2-seed ratio elimination) can land OUTSIDE the span,
            # and there is NO way to guard it: orthonormality and the M-test are
            # pairings defined *within* the span, so they are meaningless on an
            # element that has left it (both PASSED on the defective joint-fiber
            # output -- bar-blind q^0 self-norm 1, fake 1-4q^2 Schur index, spurious
            # negative structure constants; cross-checked vs BPS pure-U(3)).
            #
            # Hence only the product-and-peel builders below.  By the theorem that
            # every canonical is a polynomial of dressed minuscules they suffice in
            # principle; the general-N monopole.cone peel (#383) builds the N>=4
            # coupled fibers IN-SPAN (no solve needed -- e.g. ((0,0,1,1),(1,0,2,2))
            # builds == its L_E oracle).  Where a closed form genuinely does not yet
            # reach, the build HONESTLY FAILS (None) rather than fabricate an off-span
            # element.
            #
            # 4b. dressed-monopole . cone:  L_E(k,f) . cone(m',e').  The dressed
            #     minuscule carries magnetic phi_k, so the product couples the target
            #     only to STRICTLY-LOWER-magnetic (buildable) co-summands -> a
            #     well-founded, convergent peel (bounded, cheapest-dressing-first).
            L = _cf_monopole_cone_peel(B, charge, cache, depth)
            if L is None:
                # 4c. single-canonical W_R . base peel (asymmetric-Levi at non-regular
                #     m / cleanly-leading cases the monopole.cone peel does not cover).
                L = _cf_wr_cone_peel(B, charge, cache, depth)
    finally:
        _CF_BUILDING.discard(charge)
    if L is not None:
        cache[charge] = L
    return L


def _cf_monopole_cone_peel(B, charge, cache, depth):
    """Build a non-cone ``L_{(m,lam)}`` as ``L_E(k,f) . cone(m', e')`` (E-sector)
    or ``L_F(k,g) . cone(m', e')`` (F-sector), where ``m' = m - phi_k`` is the
    cone-buildable remainder and the DRESSED minuscule monopole carries the rest
    of the dressing.

    This is the gap>=2 / regular-`m` builder.  Unlike the magnetically-neutral
    ``W_R . cone`` (whose Wilson weight-diagram throws a co-summand to a
    further-from-cone sibling -> divergent peel), the dressed monopole carries
    magnetic ``phi_k``, so the product couples the target only to STRICTLY
    LOWER-MAGNETIC co-summands -- a well-founded measure, so the peel converges.

    ``f``/``e'`` are pinned by the leading-product relation
    ``leading(L_E(k,f).cone(m',e')) = leading(L_E(k,f)) + e' = lam``: we range
    ``e'`` over cone dressings at ``m'`` (O(N) ``_qtcone_recipe`` pruned) and
    ``f`` over the tight box implied by ``lam - e'``."""
    N = B._N
    m, lam = charge
    # O(N) fail-fast: a dressing that INCREASES inside an equal-m block is never a
    # dominant Levi irrep -> no canonical; reject before the e' loop (keeps the
    # non-canonical-label rejection fast; test_pure_un_cone_membership).
    for i in range(1, N):
        if m[i] == m[i - 1] and lam[i] > lam[i - 1]:
            return None
    a, p = _coweight_decomp(m, N)               # m = sum a_k phi_k^antidom + p det
    lo, hi = min(lam), max(lam)
    # Phase 1 -- CHEAP filters only (cone prune + closed-form `f`): collect every
    # (k, m', e', f) candidate, then sort by |f| so the SMALLEST dressings (cheap,
    # convergent products) are tried first.  A genuine build sits among the lightest
    # candidates; the heavy ones (large `f` from a spread-out `lam`) are what make
    # the unbounded search hang, so they end up last and fall past the cap.
    cands = []
    for k in range(1, N):
        if a[k - 1] <= 0:
            continue
        phik = tuple([0] * (N - k) + [1] * k)   # phi_k anti-dominant
        mp = tuple(sorted(m[i] - phik[i] for i in range(N)))
        for ep in itertools.product(range(lo - 2, hi + 2), repeat=N):
            if B._qtcone_recipe(mp, ep) is None:                 # O(N) cone prune
                continue
            need = tuple(lam[i] - ep[i] for i in range(N))       # = leading(L_E(k,f))
            f = _cf_min_f(k, need)                               # closed-form (or None)
            if f is None:
                continue
            cands.append((sum(abs(x) for x in f), k, mp, ep, f))
    cands.sort(key=lambda t: t[0])
    # Phase 2 -- EXPENSIVE work (cone build + L_E + product + peel), bounded.
    for _, k, mp, ep, f in cands[:_CF_MONOPOLE_MAXTRY]:
        cbb = B._cone_build((mp, ep))
        if cbb is None or _lowest_charge(cbb[0], N) != (mp, ep):
            continue
        G = _simp(L_E(k, f, N))
        P = _simp(G * cbb[0])
        if _lowest_charge(P, N) != charge:                       # verify the f-map
            continue
        Q = _simp(cbb[0] * G)
        L = _cf_peel_extract(B, charge, P, Q, cache, depth)
        if L is not None and _lowest_charge(L, N) == charge:
            return L
    return None


def _cf_min_f(k, need):
    """Closed-form dressing `f` with `leading(L_E(k, f)) == need` for the minuscule
    `phi_k` at general N (the Levi-character highest-weight inverse), or None if
    `need` is not a dominant Levi weight reachable at `k`.

    `phi_k = (0,...,0,1,...,1)` (k ones, anti-dominant); its Levi is `U(N-k)xU(k)`.
    The leading map of `L_E(k, f)` (verified empirically, all N) splits `f` as
    `f[:k]` (the U(k) factor) and `f[k:]` (the U(N-k) factor) and reads

        leading[0 : N-k] = sorted(f[k:],  desc)
        leading[N-k : N] = (N-k) + sorted(f[:k], desc)

    so with `a = N-k`, blkA = need[:a], blkB = need[a:] the inverse is
    `f = (blkB - a) ++ blkA`, valid exactly when each block of `need` is already
    non-increasing (i.e. `f` is Levi-dominant -- the only case yielding a clean
    leading point).  Reduces exactly to the former N=3 k=1,2 formulas."""
    N = len(need)
    a = N - k                                   # size of the U(N-k) (low) block
    blkA = need[:a]                             # = sorted(f[k:],  desc)
    blkB = need[a:]                             # = (N-k) + sorted(f[:k], desc)
    if list(blkA) != sorted(blkA, reverse=True):    # block-dominance (U(N-k))
        return None
    if list(blkB) != sorted(blkB, reverse=True):    # block-dominance (U(k))
        return None
    return tuple(blkB[i] - a for i in range(k)) + tuple(blkA)


def _cf_peel_extract(B, charge, P, Q, cache, depth):
    """Peel every strictly-lower canonical out of BOTH orderings of a
    ``W_R . base`` seed (each recognized by the affine label read and built
    closed-form here as a CONSISTENT sub-orbit -- the slice may superpose several
    canonicals at one deg_W, but the lead point is unique to the dominant one),
    then extract the bar-centered residual ``L = q^{-p}.P`` from ``P == q^{2p}.Q``.
    Returns the canonical DOp or None."""
    N = B._N
    for _ in range(4000):
        mp = B._mtest_mostneg(P)
        mq = B._mtest_mostneg(Q)
        pick = None
        for (mm, fwd) in ((mp, True), (mq, False)):
            if mm is None:
                continue
            nd, sl, _lab, lead = mm
            (mc, lamc), _l = B._cov_label(sl)        # affine label of the slice
            if (mc, lamc) == charge:                 # the target is never peeled
                continue
            Lc = cf_build_engine(B, (mc, lamc), cache, depth + 1)
            if Lc is None:
                continue
            orb = _cf_cov_q0(B, Lc)
            if lead not in orb or lead not in sl or orb[lead] == 0:
                continue
            s, r = divmod(sl[lead], orb[lead])
            if r != 0 or s == 0 or any(sl.get(k, 0) != s * orb[k] for k in orb):
                continue
            pick = (nd, Lc, s, fwd)
            break
        if pick is None:
            break
        nd, Lc, s, fwd = pick
        fp = DOp.from_scalar(VRational.from_scalar(LaurentPoly({nd: s}), n=N), n=N)
        fm = DOp.from_scalar(VRational.from_scalar(LaurentPoly({-nd: s}), n=N), n=N)
        hereP, hereQ = (fp, fm) if fwd else (fm, fp)   # bar pair: q^{nd} / q^{-nd}
        P = _simp(P + hereP * Lc * (-1))
        Q = _simp(Q + hereQ * Lc * (-1))
    else:
        return None
    if not P._terms or not Q._terms:
        return None
    e = B._monomial_ratio(P, Q)                       # P == q^e . Q
    if e is None or e % 2 != 0:
        return None
    fac2 = DOp.from_scalar(VRational.from_scalar(LaurentPoly({-(e // 2): 1}), n=N), n=N)
    L = _simp(fac2 * P)
    if _lowest_charge(L, N) != charge:
        return None
    return L


def _cf_wr_cone_peel(B, charge, cache, depth):
    """Build a non-cone ``L_{(m,lam)}`` as the bar-symmetric residual of
    ``W_R . base``, over the engine's two factorization families (`_wR_cone_build`):

      (1) a DOMINANT Wilson ``R`` with ``lam-R`` a QTCone dressing -- ``base`` is
          the cone monomial; and
      (2) a single Wilson BOX (fundamental ``e_i`` / antifundamental ``-e_i``) onto
          a strictly-closer-to-cone canonical ``L_{(m, lam-box)}`` -- ``base`` is
          that canonical, built recursively (the L1 distance to the bare-cone
          dressing strictly decreases, so the recursion bottoms out on cones).

    Family (2) reaches the non-det GL_k-block dressings (standard-rep weights like
    ``(1,0,..)``) that first appear at N>=4 and that family (1) alone cannot."""
    N = B._N
    m, lam = charge
    ref = B._cone_ref(tuple(m))
    lam_cone = ref[1] if ref is not None else None

    def cplx(l):
        return (sum(abs(l[i] - lam_cone[i]) for i in range(N))
                if lam_cone is not None else None)

    c_lam = cplx(lam)
    candidates = []                                   # (R_wilson, base_charge, base_dop_or_None)
    # family (1): smallest dominant R with lam-R a cone dressing
    has_recipe = hasattr(B, "_qtcone_recipe")
    rng = range(0, max(lam) - min(lam) + 2)
    for R in sorted((r for r in itertools.product(rng, repeat=N)
                     if list(r) == sorted(r, reverse=True) and any(r)),
                    key=lambda r: (sum(r), r)):
        eC = tuple(lam[i] - R[i] for i in range(N))
        if has_recipe and B._qtcone_recipe(m, eC) is None:    # O(N) cone probe (#366)
            continue
        cb = B._cone_build((m, eC))
        if cb is not None and _lowest_charge(cb[0], N) == (m, eC):
            candidates.append((R, (m, eC), cb[0]))
            break
    # family (2): single Wilson box toward the cone, closest-first
    fund_hw = tuple([1] + [0] * (N - 1))
    anti_hw = tuple([0] * (N - 1) + [-1])
    boxes = ([tuple(1 if t == i else 0 for t in range(N)) for i in range(N)]
             + [tuple(-1 if t == i else 0 for t in range(N)) for i in range(N)])
    box_facs = []
    for w in boxes:
        lam_p = tuple(lam[i] - w[i] for i in range(N))
        cp = cplx(lam_p)
        if c_lam is None or cp is None or cp >= c_lam:
            continue
        Rhw = fund_hw if sum(w) > 0 else anti_hw
        box_facs.append((cp, Rhw, (m, lam_p)))
    box_facs.sort(key=lambda t: t[0])
    candidates += [(Rhw, bch, None) for _cp, Rhw, bch in box_facs]

    for R, base_charge, base_dop in candidates:
        if base_charge == charge:
            continue
        base = (base_dop if base_dop is not None
                else cf_build_engine(B, base_charge, cache, depth + 1))
        if base is None:
            continue
        P = _simp(L_W(R, N) * base)
        Q = _simp(base * L_W(R, N))                   # Q = bar(P) (W_R, base bar-fixed)
        L = _cf_peel_extract(B, charge, P, Q, cache, depth)
        if L is not None:
            return L
    return None


def multiply_exact(B, a, b, cache=None):
    """EXACT structure constants of ``L_a.L_b`` via closed-form build +
    recognize-and-subtract by lowest tropical charge (read
    ``C(q) = [P]_charge / [L_c]_charge``; subtract the FULL canonical; repeat).

    Builds are routed through `cf_build_engine` (QTCone + W_R.cone peel), so this
    never touches the engine's hanging `_build_clean`.  Reconstructs BY
    CONSTRUCTION (only ever subtracts exact full canonicals), so it CANNOT mask:
    where a canonical is mis-built the result is non-palindromic (bar violation
    stays VISIBLE) instead of being laundered into a palindromic-but-wrong answer
    by the M-test's two-ordering union.  ``cache`` is the per-call build memo."""
    N = B._N
    if cache is None:
        cache = {}

    def img(ch):
        ch = (tuple(ch[0]), tuple(ch[1]))
        if B.has(ch):
            return B.image(ch)
        L = cf_build_engine(B, ch, cache)
        if L is None:
            raise NotImplementedError(f"multiply_exact: cannot build L_{ch}")
        return L

    P = _simp(img(a) * img(b))
    out = {}
    for _ in range(20000):
        if P.is_zero():
            return out
        charge = _lowest_charge(P, N)
        Lc = img(charge)
        cP = B._coeff_at(P, charge)
        cL = B._coeff_at(Lc, charge)
        lc = list(cL._coeffs.items())
        if len(lc) != 1:
            raise RuntimeError(f"multiply_exact: L_c leading not a q-monomial at {charge}")
        (eb, vb), = lc
        Cq = LaurentPoly({e - eb: (v if vb == 1 else v // vb) for e, v in cP._coeffs.items()})
        out[charge] = out.get(charge, LaurentPoly.zero()) + Cq
        Cd = DOp.from_scalar(VRational.from_scalar(Cq, n=N), n=N)
        P = _simp(P + (Cd * Lc) * (-1))
    raise RuntimeError("multiply_exact: did not terminate")
