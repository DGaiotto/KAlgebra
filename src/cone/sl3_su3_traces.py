"""Spine-free elementary traces and trace assembly for `SU3ADKAlg`
([A_1, D_4] with SU(3) enhancement = SU(3)_{-3/2}).

This is the production, BPS-free trace backend for `su3_ad_kalg`: it computes
the ρ²-twisted Schur trace entirely from

  * **Tr_1** — the closed-form Kac–Wakimoto vacuum character of
    `\\widehat{sl}(3)_{-3/2}` (`vacuum_character`), an exact SU(3)-character
    q-series;
  * **Tr_T, Tr_D** — the two remaining elementary traces, obtained from the
    **orthonormality bootstrap** seeded by Tr_1 (`SU3ElemTraces`): Layer-1
    forces `Tr(seed^a) = O(𝖖^a)`, which deconvolves order-by-order into Tr_T
    and ⋆Tr_D (a single forward pass in Cartan fugacities); arbitrary q-order;
  * the **Layer-1 reduction** of any canonical monomial to those three seeds
    (`seed_z_fast` / `fug_multiply`), carried out in **Cartan fugacities**
    (z-Laurent over the SU(3) weight lattice) so non-self-dual product content
    is never prematurely symmetrized — Weyl-symmetrization to SU(3) characters
    happens only on the *total* (`sym_to_char`).

No BPS / RG / quantum-torus engine is touched.  Everything below uses only
`zplus_ring.SU3ZPlusRing` (weight diagrams / characters) and the letter
machinery of `su3_ad_kalg` (cone relations; lazily, no import cycle).

Background: `su3ad_layer2_numerators.md`, `su3ad_flavour_layer1_defect.md`.
The character/fugacity bootstrap was prototyped in
`experiments/sl3_{m32_vacuum_char,fug_reduce,fugacity_solve}.py`; this module
is the consolidated, engine-free production version those validate against.
"""
from __future__ import annotations

from fractions import Fraction as Fr

from zplus_ring import SU3ZPlusRing, RElement, RPowerSeries

# Letter machinery of the cone presentation (no BPS): q-commute cocycle, the
# hardcoded interaction relations, the label<->monomial maps, ρ² on letters,
# and the Z_4 order.  Imported at top level: `su3_ad_kalg` never imports this
# module at *its* top level (only lazily inside trace methods), so there is no
# import cycle in either order.
from su3_ad_kalg import (
    _rho_n_letter, _q_commute_twist, _interaction,
    _label_to_monomial, _monomial_to_label, _N,
)

# ---------------------------------------------------------------------------
# sl(3) finite data (Dynkin-label coordinates) + Kac–Wakimoto theta sums
# ---------------------------------------------------------------------------

G = [[Fr(2, 3), Fr(1, 3)], [Fr(1, 3), Fr(2, 3)]]      # (ω_i,ω_j) = A^{-1}
RHO = (1, 1)


def ip(a, b):
    return sum(Fr(a[i]) * G[i][j] * Fr(b[j]) for i in range(2) for j in range(2))


def _s1(v):
    return (-v[0], v[1] + v[0])


def _s2(v):
    return (v[0] + v[1], -v[1])


WEYL = [(1, []), (-1, [_s1]), (-1, [_s2]),
        (1, [_s1, _s2]), (1, [_s2, _s1]), (-1, [_s1, _s2, _s1])]


def _apply(word, v):
    for f in reversed(word):
        v = f(v)
    return v


def qvec(n1, n2):                                     # n1·α1 + n2·α2 in ω-labels
    return (2 * n1 - n2, -n1 + 2 * n2)


def theta(step, factor, Kmax, base=RHO):
    """Kac–Wakimoto theta sum {q_power: {Dynkin weight: coeff}}.  `base` is the
    finite weight the β-translations centre on (ρ for the denominator/vacuum
    numerator); `step`/`factor` are the lattice step and (k+h∨)."""
    out = {}
    B = Kmax + 6
    f = Fr(factor)
    base = tuple(Fr(b) for b in base)
    for n1 in range(-B, B + 1):
        for n2 in range(-B, B + 1):
            beta = qvec(step * n1, step * n2)
            qpow = ip(base, beta) + f / 2 * ip(beta, beta)
            if qpow != int(qpow) or not (0 <= int(qpow) <= Kmax):
                continue
            qp = int(qpow)
            fin0 = tuple(base[i] + f * Fr(beta[i]) for i in range(2))
            for sign, word in WEYL:
                wf = _apply(word, fin0)
                if any(x.denominator != 1 for x in wf):
                    continue
                key = (int(wf[0]), int(wf[1]))
                d = out.setdefault(qp, {})
                d[key] = d.get(key, 0) + sign
    return {q: {w: c for w, c in d.items() if c} for q, d in out.items()}


# ---------------------------------------------------------------------------
# SU(3) weight diagrams + z-Laurent (Cartan fugacity) primitives
# ---------------------------------------------------------------------------

_R3 = SU3ZPlusRing()
_wd_cache = {}


def weight_diagram(hw):
    """{Dynkin weight: multiplicity} for the SU(3) irrep with h.w. hw=(p,q).
    Via the tested `SU3ZPlusRing._irrep_weights` (abelian basis) + the verified
    map  Dynkin(m1,m2) <-> abelian(a,b)=(m2, m1+m2),  i.e. m1=b-a, m2=a."""
    if hw in _wd_cache:
        return _wd_cache[hw]
    res = {}
    for (a, b), m in _R3._irrep_weights(hw).items():
        res[(b - a, a)] = res.get((b - a, a), 0) + m
    _wd_cache[hw] = res
    return res


# short alias used throughout the reducer
def wd(pq):
    return weight_diagram(pq)


def zmul(a, b):                                       # weight addition
    out = {}
    for wa, ca in a.items():
        for wb, cb in b.items():
            k = (wa[0] + wb[0], wa[1] + wb[1])
            out[k] = out.get(k, 0) + ca * cb
    return {w: v for w, v in out.items() if v}


def zadd(a, b, s=1):
    out = dict(a)
    for w, c in b.items():
        out[w] = out.get(w, 0) + s * c
    return {w: c for w, c in out.items() if c}


def zsub(a, b):
    return zadd(a, b, s=-1)


def zstar(a):                                         # ⋆ = weight negation
    return {(-w[0], -w[1]): c for w, c in a.items()}


def char_to_zlaurent(char):                           # {(p,q):c} -> {weight:c}
    out = {}
    for hw, c in char.items():
        for w, m in weight_diagram(hw).items():
            out[w] = out.get(w, 0) + c * m
    return {w: v for w, v in out.items() if v}


def antisym_to_char(R):
    """Divide an antisymmetric z-Laurent by the Weyl denominator A(ρ):
    returns {(p,q): coeff} (the symmetric SU(3) virtual character)."""
    R = dict(R)
    char = {}
    guard = 0
    while R:
        guard += 1
        if guard > 5000:
            raise RuntimeError("antisym_to_char did not terminate")
        dom = [w for w, c in R.items() if w[0] >= 1 and w[1] >= 1 and c]
        if not dom:
            raise RuntimeError(f"antisym residue off dominant chamber: {R}")
        nu = max(dom, key=lambda w: ip(w, RHO))
        a = R[nu]
        lam = (nu[0] - 1, nu[1] - 1)
        char[lam] = char.get(lam, 0) + a
        for sign, word in WEYL:
            wv = _apply(word, nu)
            R[wv] = R.get(wv, 0) - a * sign
        R = {w: c for w, c in R.items() if c}
    return {k: v for k, v in char.items() if v}


def sym_to_char(z):
    """Weyl-**symmetric** z-Laurent -> {(p,q): coeff} SU(3) (virtual)
    character.  Peel the highest dominant weight, subtract its full irrep
    weight diagram, repeat.  Used to symmetrize the *total* of a trace only
    at the very end (never per intermediate term)."""
    z = {w: c for w, c in z.items() if c}
    char = {}
    guard = 0
    while z:
        guard += 1
        if guard > 20000:
            raise RuntimeError("sym_to_char did not terminate")
        dom = [w for w, c in z.items() if w[0] >= 0 and w[1] >= 0 and c]
        if not dom:
            raise RuntimeError(f"sym residue off dominant chamber: {z}")
        hw = max(dom, key=lambda w: ip(w, RHO))
        c = z[hw]
        char[hw] = char.get(hw, 0) + c
        for w, m in weight_diagram(hw).items():
            z[w] = z.get(w, 0) - c * m
        z = {w: cc for w, cc in z.items() if cc}
    return {k: v for k, v in char.items() if v}


# ---------------------------------------------------------------------------
# Tr_1 — the Kac–Wakimoto vacuum character of \widehat{sl}(3)_{-3/2}
# ---------------------------------------------------------------------------
# Conventions (user, 2026-06-13): q_paper = 𝖖² (BPS index variable), no
# q^{-c/24} prefactor (vacuum starts at 1).  ch = N/D solved order-by-order;
# /D_0 (= /Weyl-denominator) turns an antisymmetric z-Laurent into a symmetric
# SU(3) character.

def vacuum_character(Kmax):
    """{q_paper grade: {(p,q): coeff}} — exact, BPS-free."""
    N = theta(2, Fr(3, 2), Kmax)                      # numerator: vQ, k+h∨=3/2
    D = theta(1, 3, Kmax)                             # denominator: Q, h∨=3
    ch, ch_z = {}, {}
    for n in range(0, Kmax + 1):
        Rn = dict(N.get(n, {}))
        for j in range(0, n):
            if j in ch_z and (n - j) in D:
                Rn = zsub(Rn, zmul(ch_z[j], D[n - j]))
        cn = antisym_to_char(Rn)
        ch[n] = cn
        ch_z[n] = char_to_zlaurent(cn)
    return ch


# ---------------------------------------------------------------------------
# Layer-1 reduction in Cartan fugacities (CG-free, memoized)
# ---------------------------------------------------------------------------
# Carry a z-Laurent (Cartan fugacity) instead of an SU(3) irrep label: each
# interaction fuses by a single `zmul` with the (tiny) weight diagram of
# chi_inter — no Clebsch–Gordan, no fan-out.  z and q_factor are passive, so
# the general reduction is the unit reduction post-multiplied; `reduce_unit`
# memoizes by word and collapses the otherwise-exponential tag-move tree.

_UNIT = {(0, 0): 1}
_unit_cache = {}


def reduce_unit(word):
    """Reduce the trace of a letter `word` (with z=unit, q=0) to
    {seed_key: {q_exp: z-Laurent}}, memoized by word.  seed_key ∈
    {('Tr_1',), ('Tr_T',), ('Tr_D',0/1)} (or ('Tr_irreducible', …))."""
    word = tuple(word)
    if word in _unit_cache:
        return _unit_cache[word]
    out = _reduce_unit_impl(word)
    _unit_cache[word] = out
    return out


def _reduce_unit_impl(word):
    if not word:
        return {('Tr_1',): {0: dict(_UNIT)}}
    if len(word) == 1:
        L = word[0]
        key = ('Tr_T',) if L[0] == 'T' else ('Tr_D', L[1] % 2)
        return {key: {0: dict(_UNIT)}}
    n = len(word)
    cur = list(word)
    qa = 0
    for _cycle in range(_N):
        cur = [_rho_n_letter(cur[-1], 2)] + cur[:-1]
        pos = 0
        while pos < n - 1:
            L_tag, L_next = cur[pos], cur[pos + 1]
            t = _q_commute_twist(L_tag, L_next)
            if t is None:
                left, right = cur[:pos], cur[pos + 2:]
                result = {}
                for q_delta, ld, chi_inter, mult in _interaction(L_tag, L_next):
                    mid = []
                    for L, e in ld.items():
                        mid += [L] * e
                    sub = reduce_unit(left + mid + right)
                    zf = wd(chi_inter)
                    for key, qd in sub.items():
                        b = result.setdefault(key, {})
                        for qe, zz in qd.items():
                            zc = zmul(zf, zz)
                            if mult != 1:
                                zc = {w: c * mult for w, c in zc.items()}
                            qk = qa + q_delta + qe
                            b[qk] = zadd(b.get(qk, {}), zc)
                for key in list(result):
                    result[key] = {qk: zz for qk, zz in result[key].items() if zz}
                    if not result[key]:
                        del result[key]
                return result
            qa += 2 * t
            cur[pos], cur[pos + 1] = L_next, L_tag
            pos += 1
    return {('Tr_irreducible', tuple(cur)): {qa: dict(_UNIT)}}


def seed_z_fast(lab):
    """Layer-1 reduction of a single canonical label (tile,a,b,p,q) to
    {seed_key: {q_exp: z-Laurent}} in fugacity space."""
    letters, chi_pq, q_factor = _label_to_monomial(lab)
    word = []
    for L in sorted([L for L in letters if L[0] == 'T']):
        word += [L] * letters[L]
    for L in sorted([L for L in letters if L[0] == 'D']):
        word += [L] * letters[L]
    unit = reduce_unit(word)
    z0 = wd(chi_pq)
    return {key: {q_factor + qe: zmul(z0, zz) for qe, zz in qd.items()}
            for key, qd in unit.items()}


# ---------------------------------------------------------------------------
# Fugacity-level multiply (carry weights, NOT characters, through a product)
# ---------------------------------------------------------------------------
# Port of `su3_ad_kalg._reduce_letter_seq` with the SU(3) Clebsch–Gordan
# `_su3_cg(pq, chi)` replaced by Cartan-weight multiplication `zmul(z, wd(chi))`.
# Result: {(tile,a,b): {q_exp: z-Laurent}} — canonical monomials whose flavour
# coefficient is a (possibly non-self-dual) Cartan weight, NOT a prematurely
# symmetrized character.  This is the fix the layer-1 defect note prescribes
# for the ρ²-twisted trace of a *product* (`su3ad_flavour_layer1_defect.md`).

def fug_multiply(letter_seq, z, q_factor, depth=0):
    if depth > 90:
        raise RecursionError(f"fug_multiply depth {depth}: {letter_seq}")
    seq = [L for L in letter_seq if L is not None]
    n = len(seq)
    distinct = list(set(seq))
    all_commute = True
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            if _q_commute_twist(distinct[i], distinct[j]) is None:
                all_commute = False
                break
        if not all_commute:
            break

    first_bad = None
    if not all_commute:
        for idx in range(n - 1):
            if _q_commute_twist(seq[idx], seq[idx + 1]) is None:
                first_bad = idx
                break
        if first_bad is None:                          # bubble offenders adjacent
            target_i = target_j = None
            for i_idx in range(n):
                for j_idx in range(i_idx + 1, n):
                    if _q_commute_twist(seq[i_idx], seq[j_idx]) is None:
                        target_i, target_j = i_idx, j_idx
                        break
                if target_i is not None:
                    break
            new_seq = list(seq)
            q_pre = 0
            pos = target_j
            while pos > target_i + 1:
                L_a, L_b = new_seq[pos - 1], new_seq[pos]
                t = _q_commute_twist(L_a, L_b)
                if t is None:
                    break
                q_pre += 2 * t
                new_seq[pos - 1], new_seq[pos] = L_b, L_a
                pos -= 1
            return fug_multiply(new_seq, z, q_factor + q_pre, depth + 1)

    if first_bad is None:                              # all commute: normal-order
        sequence = list(seq)
        target = (sorted([L for L in sequence if L[0] == 'T']) +
                  sorted([L for L in sequence if L[0] == 'D']))
        q_delta = 0
        for tp in range(len(target)):
            tgt_L = target[tp]
            sp = tp
            while sp < len(sequence) and sequence[sp] != tgt_L:
                sp += 1
            while sp > tp:
                L_a, L_b = sequence[sp - 1], sequence[sp]
                q_delta += 2 * _q_commute_twist(L_a, L_b)
                sequence[sp - 1], sequence[sp] = L_b, L_a
                sp -= 1
        out_letters = {}
        for L in sequence:
            out_letters[L] = out_letters.get(L, 0) + 1
        (tile, a, b, _p, _q), lp = _monomial_to_label(
            out_letters, (0, 0), q_factor + q_delta)
        qp = next(iter(lp._coeffs))
        return {(tile, a, b): {qp: dict(z)}}

    # interaction at first_bad
    L1, L2 = seq[first_bad], seq[first_bad + 1]
    left = seq[:first_bad]
    right = seq[first_bad + 2:]
    acc = {}
    for q_delta, new_letters_dict, chi_pq, mult in _interaction(L1, L2):
        mid = []
        for L, e in new_letters_dict.items():
            mid += [L] * e
        new_z = zmul(z, wd(chi_pq))
        if mult != 1:
            new_z = {w: c * mult for w, c in new_z.items()}
        if not new_z:
            continue
        sub = fug_multiply(left + mid + right, new_z, q_factor + q_delta,
                           depth + 1)
        for lab, qd in sub.items():
            bucket = acc.setdefault(lab, {})
            for qe, zz in qd.items():
                bucket[qe] = zadd(bucket.get(qe, {}), zz)
    for lab in list(acc):
        acc[lab] = {qe: zz for qe, zz in acc[lab].items() if zz}
        if not acc[lab]:
            del acc[lab]
    return acc


# ---------------------------------------------------------------------------
# Tr_T / Tr_D — the orthonormality bootstrap (forward pass, fugacity space)
# ---------------------------------------------------------------------------
# Layer-1 gives, for the pure seeds,
#   Tr(seed^a) = A·Tr_1 + B·Tr_T + C0·Tr_D + C1·(⋆Tr_D),   Tr_D = ⋆(SD),
# and orthonormality forces Tr(seed^a)=O(𝖖^a).  The Layer-1 supports make this
# a strictly-triangular recursion (Tr_T[k] ← SD[≤k-2], SD[k] ← Tr_T[≤k-4]), so
# a single forward pass closes the tower.  The whole tower runs on the cheap
# D-seeds (D^a, a≡0 mod3 → Tr_T scalar lead; a≡2 mod3 → ⋆Tr_D scalar lead);
# the larger D-seed re-derivation is a BPS-free over-determination certificate.


def _seed_layer1_z(lab):
    """(A, B, C0, C1) dressings of (Tr_1, Tr_T, Tr_D₀, Tr_D₁) for a seed."""
    R = seed_z_fast(lab)
    return (R.get(("Tr_1",), {}), R.get(("Tr_T",), {}),
            R.get(("Tr_D", 0), {}), R.get(("Tr_D", 1), {}))


class SU3ElemTraces:
    """Cache of the three elementary trace seeds to a given 𝖖-order, all
    BPS-free: Tr_1 (vacuum char), Tr_T, ⋆Tr_D (=SD).  Indexed by 𝖖-power
    (q_paper grade k sits at 𝖖^{2k})."""

    def __init__(self):
        self.K = 0
        self.Tr1 = {}                                  # {𝖖-power: z-Laurent}
        self.TrT = {}
        self.SD = {}                                   # = ⋆Tr_D
        self._seedcache = {}

    # -- ensure the seeds are known through 𝖖^K --------------------------
    def ensure(self, K):
        if K <= self.K:
            return self
        # Tr_1: vacuum char (q_paper grade k -> 𝖖^{2k}); need grade K//2 + 1
        self.Tr1 = {2 * k: char_to_zlaurent(c)
                    for k, c in vacuum_character(K // 2 + 2).items()}
        self.TrT, self.SD = {}, {}
        for k in range(1, K + 1):
            self.SD[k] = self._solve_SD(k)
            self.TrT[k] = self._solve_TrT(k)
        self.K = K
        return self

    def Dseed(self, a):
        key = ("D", a)
        if key not in self._seedcache:
            self._seedcache[key] = _seed_layer1_z((0, 0, a, 0, 0))
        return self._seedcache[key]

    def _eq_rest(self, A, B, C0, C1, m, drop):
        acc = {}
        for e, ze in A.items():
            t = self.Tr1.get(m - e)
            if t:
                acc = zadd(acc, zmul(ze, t))
        for e, ze in B.items():
            if drop == ("B", e):
                continue
            t = self.TrT.get(m - e)
            if t:
                acc = zadd(acc, zmul(ze, t))
        for e, ze in C0.items():
            s = self.SD.get(m - e)                      # Tr_D = ⋆SD
            if s:
                acc = zadd(acc, zmul(ze, zstar(s)))
        for e, ze in C1.items():
            if drop == ("C1", e):
                continue
            s = self.SD.get(m - e)
            if s:
                acc = zadd(acc, zmul(ze, s))
        return acc

    def _smallest_DT(self, k):                          # a≡0 mod3 isolates Tr_T
        a = 3
        while (a * a + 3 * a - 3) // 3 <= k:
            a += 3
        return a

    def _smallest_DS(self, k):                          # a≡2 mod3 isolates SD
        a = 2
        while (a * a + 3 * a - 1) // 3 <= k:
            a += 3
        return a

    def _solve_TrT(self, k):
        A, B, C0, C1 = self.Dseed(self._smallest_DT(k))
        eb = min(B)
        lead = B[eb][(0, 0)]                            # scalar ±1
        rest = self._eq_rest(A, B, C0, C1, k + eb, ("B", eb))
        return rest if lead == -1 else {w: -c for w, c in rest.items()}

    def _solve_SD(self, k):
        A, B, C0, C1 = self.Dseed(self._smallest_DS(k))
        ec = min(C1)
        lead = C1[ec][(0, 0)]
        rest = self._eq_rest(A, B, C0, C1, k + ec, ("C1", ec))
        return rest if lead == -1 else {w: -c for w, c in rest.items()}

    # -- seed series lookup ({𝖖-power: z-Laurent}) -----------------------
    def series(self, key):
        if key == ("Tr_1",):
            return self.Tr1
        if key == ("Tr_T",):
            return self.TrT
        if key == ("Tr_D", 0):                          # Tr_D = ⋆SD
            return {k: zstar(self.SD[k]) for k in self.SD}
        if key == ("Tr_D", 1):                          # ⋆Tr_D = SD
            return self.SD
        raise KeyError(key)


# one process-wide cache (the seeds are universal for SU3AD)
_PROVIDER = SU3ElemTraces()


def _provider(K):
    return _PROVIDER.ensure(K)


# ---------------------------------------------------------------------------
# Public trace API (BPS-free)
# ---------------------------------------------------------------------------

def _char_dict_to_relt(ch):
    return RElement(_R3, {hw: c for hw, c in ch.items() if c})


def _assemble(total_z, chi_z, K):
    """Symmetrize each 𝖖-order of a z-Laurent total to SU(3) characters,
    optionally pre-multiplying a spectator flavour weight chi_z, -> {q:RElement}."""
    out = {}
    for qp, z in total_z.items():
        if qp < 0 or qp > K:
            continue
        z2 = zmul(chi_z, z) if chi_z != {(0, 0): 1} else z
        if not z2:
            continue
        relt = _char_dict_to_relt(sym_to_char(z2))
        if not relt.is_zero():
            out[qp] = relt
    return out


def _single_label_trace_z(prov, tab, K):
    """{𝖖-power: z-Laurent} BPS-free trace of a flavour-trivial canonical
    monomial (tile,a,b).  Seeds must already cover the required depth."""
    tile, a, b = tab
    red = seed_z_fast((tile, a, b, 0, 0))
    out = {}
    for seedkey, qd in red.items():
        if seedkey[0] in ("Tr_irreducible", "Tr_max_depth"):
            raise NotImplementedError(
                f"SU3AD trace: Layer-1 left {seedkey} unreduced for {tab}")
        ser = prov.series(seedkey)
        for qexp, zdress in qd.items():
            for sq, sz in ser.items():
                qp = qexp + sq
                if qp > K or qp < 0:
                    continue
                out[qp] = zadd(out.get(qp, {}), zmul(zdress, sz))
    return out


def _seed_depth(*qexp_iterables):
    lo = 0
    for it in qexp_iterables:
        for qe in it:
            if qe < lo:
                lo = qe
    return -lo                                          # extra orders needed below 0


def trace(alg, label, K=20):
    """ρ²-twisted Schur trace Tr(L_label) over R(SU(3))((𝖖)), BPS-free.

    R(SU(3))-linear flavour stripping (`Tr(χ_(p,q)·M)=χ_(p,q)·Tr(M)`): reduce
    the flavour-trivial gauge monomial in fugacity to the {Tr_1,Tr_T,Tr_D}
    seeds, symmetrize the total, then multiply the spectator character back."""
    tile, a, b, p, q = alg.canonicalise(label)
    red = seed_z_fast((tile, a, b, 0, 0))
    margin = _seed_depth(*(qd.keys() for qd in red.values()))
    prov = _provider(K + margin + 2)
    total = {}
    for seedkey, qd in red.items():
        if seedkey[0] in ("Tr_irreducible", "Tr_max_depth"):
            raise NotImplementedError(
                f"SU3AD trace: Layer-1 left {seedkey} unreduced for {label}")
        ser = prov.series(seedkey)
        for qexp, zdress in qd.items():
            for sq, sz in ser.items():
                qp = qexp + sq
                if qp > K or qp < 0:
                    continue
                total[qp] = zadd(total.get(qp, {}), zmul(zdress, sz))
    chi_z = wd((p, q)) if (p, q) != (0, 0) else {(0, 0): 1}
    return RPowerSeries(_R3, _assemble(total, chi_z, K), K)


def product_trace(alg, factors, K=8):
    """ρ²-twisted Schur trace Tr(L_{f₁}·…·L_{fₙ}) of a product of canonical
    generators, BPS-free.  Fugacity-multiply the whole product (weights, not
    characters), single-label-trace each resulting canonical monomial, and
    Weyl-symmetrize only the *total* — so non-self-dual product content is
    handled correctly (the `su3ad_flavour_layer1_defect` fix)."""
    letters_all = []
    chi_z = {(0, 0): 1}
    qf = 0
    for lab in factors:
        tile, a, b, p, q = alg.canonicalise(lab)
        letters, _chi, q_factor = _label_to_monomial((tile, a, b, 0, 0))
        w = []
        for L in sorted([L for L in letters if L[0] == 'T']):
            w += [L] * letters[L]
        for L in sorted([L for L in letters if L[0] == 'D']):
            w += [L] * letters[L]
        letters_all += w
        qf += q_factor
        if (p, q) != (0, 0):
            chi_z = zmul(chi_z, wd((p, q)))
    prod = fug_multiply(tuple(letters_all), {(0, 0): 1}, qf)
    # depth needed: most-negative q across product monomials + their traces
    margin = _seed_depth(*(qd.keys() for qd in prod.values()))
    # plus the single-label trace depth of each monomial; pad generously
    prov = _provider(K + margin + max((b for (_, _, b) in prod), default=0) ** 2
                     + max((a for (_, a, _) in prod), default=0) ** 2 + 4)
    total = {}
    for tab, qd in prod.items():
        minq = min(qd) if qd else 0
        base = _single_label_trace_z(prov, tab, K - minq)
        for q1, z1 in qd.items():
            for q2, z2 in base.items():
                qp = q1 + q2
                if qp > K or qp < 0:
                    continue
                total[qp] = zadd(total.get(qp, {}), zmul(z1, z2))
    return RPowerSeries(_R3, _assemble(total, chi_z, K), K)


def inner_product(alg, a, b, K=20):
    """Schur pairing I_{a,b} = Tr(ρ(L_a)·L_b), BPS-free."""
    return product_trace(alg, [alg.rho(alg.canonicalise(a)), b], K)
