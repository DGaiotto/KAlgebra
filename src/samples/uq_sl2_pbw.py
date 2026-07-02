"""uq_sl2_pbw — the central quotient of `U_𝖖(sl_2)` on its PBW basis, as a
`KAlgebra`.

This is the **algebra-side** realisation (no RG flow, no chart): the canonical
basis is the PBW basis of the central quotient,

    K^n        (n in Z)                         -- Cartan sector
    E_{a,b} = q^{-ab} E^a K^b   (a>=1, b in Z)  -- raising sector
    F_{a,b} = q^{ab}  F^a K^b   (a>=1, b in Z)  -- lowering sector

each **bar-invariant by construction** (the q^{∓ab} normalisation; `bar` fixes
`{E,F,K}` and sends `q->q^{-1}`, antimultiplicative).  The Casimir `C` is the
central element identified with the flavour character `χ_1`; here it is carried
as a non-negative power `j` in the label (`C^j` central), so multiply is
`Z[q^±]`-valued.

Defining relations (central quotient):

    K E = q^{-2} E K ,   K F = q^{2} F K ,
    E F = C + q K + q^{-1} K^{-1} ,   F E = C + q^{-1} K + q K^{-1} ,
    [E,F] = (q - q^{-1})(K - K^{-1}) .

`ρ` is Lusztig's braid (infinite order): `ρ(K)=K^{-1}`, `ρ(E)=q^{-1}FK^{-1}`,
`ρ(F)=qKE`, fixing `C`.  The trace lives on the Cartan sector (Tr vanishes on
any net E/F charge); `Tr K^n` is supplied by `trace`.

Internally elements are dicts over **un-normalised** monomials
`(side, p, k, j)` — `K^k` (side='K'), `K^k E^p` (side='E', p>=1), `F^p K^k`
(side='F', p>=1), times `C^j` — with `LaurentPoly(q)` coefficients; the public
canonical labels carry the `q^{∓ab}` normalisation.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from laurent_poly import LaurentPoly


# --- internal monomial algebra (un-normalised; C^j central) -----------------
# monomial key: (side, p, k, j)
#   side 'K': K^k                  (p ignored, store 0)
#   side 'E': K^k E^p   (p>=1)
#   side 'F': F^p K^k   (p>=1)
# value: LaurentPoly in q.

def _q(n: int) -> LaurentPoly:
    return LaurentPoly({n: 1})


def _add(out: dict, key, coeff: LaurentPoly) -> None:
    if coeff.is_zero():
        return
    cur = out.get(key)
    s = coeff if cur is None else cur + coeff
    if s.is_zero():
        out.pop(key, None)
    else:
        out[key] = s


def _lm_K(sign: int, mono, coeff: LaurentPoly) -> dict:
    """Left-multiply K^{sign} (sign=±1) onto a monomial."""
    side, p, k, j = mono
    out: dict = {}
    if side == 'K':
        _add(out, ('K', 0, k + sign, j), coeff)
    elif side == 'E':                       # K^{±1}·K^k E^p = K^{k±1} E^p
        _add(out, ('E', p, k + sign, j), coeff)
    else:                                   # K^{±1}·F^p K^k = q^{±2p} F^p K^{k±1}
        _add(out, ('F', p, k + sign, j), _q(sign * 2 * p) * coeff)
    return out


def _lm_E(mono, coeff: LaurentPoly) -> dict:
    """Left-multiply E onto a monomial (reduces the F-sector via EF=C+qK+q^{-1}K^{-1})."""
    side, p, k, j = mono
    out: dict = {}
    if side == 'K':                         # E·K^k = q^{2k} K^k E
        _add(out, ('E', 1, k, j), _q(2 * k) * coeff)
    elif side == 'E':                       # E·K^k E^p = q^{2k} K^k E^{p+1}
        _add(out, ('E', p + 1, k, j), _q(2 * k) * coeff)
    else:                                   # E·F^p K^k
        # E·F^p = C F^{p-1} + q^{2p-1} F^{p-1} K + q^{1-2p} F^{p-1} K^{-1}
        # then ·K^k.  F^{p-1}=Cartan if p==1.
        if p == 1:
            _add(out, ('K', 0, k, j + 1), coeff)                       # C·K^k
            _add(out, ('K', 0, k + 1, j), _q(1) * coeff)               # q·K^{k+1}
            _add(out, ('K', 0, k - 1, j), _q(-1) * coeff)              # q^{-1}·K^{k-1}
        else:
            _add(out, ('F', p - 1, k, j + 1), coeff)                   # C·F^{p-1}K^k
            _add(out, ('F', p - 1, k + 1, j), _q(2 * p - 1) * coeff)   # q^{2p-1}F^{p-1}K^{k+1}
            _add(out, ('F', p - 1, k - 1, j), _q(1 - 2 * p) * coeff)   # q^{1-2p}F^{p-1}K^{k-1}
    return out


def _lm_F(mono, coeff: LaurentPoly) -> dict:
    """Left-multiply F onto a monomial (reduces the E-sector via FE=C+q^{-1}K+qK^{-1})."""
    side, p, k, j = mono
    out: dict = {}
    if side == 'K':                         # F·K^k = F K^k
        _add(out, ('F', 1, k, j), coeff)
    elif side == 'F':                       # F·F^p K^k = F^{p+1} K^k
        _add(out, ('F', p + 1, k, j), coeff)
    else:                                   # F·K^k E^p = q^{-2k} K^k (F·E^p)
        # F·E^p = C E^{p-1} + q^{1-2p} E^{p-1} K + q^{2p-1} E^{p-1} K^{-1}
        base = _q(-2 * k) * coeff
        if p == 1:
            _add(out, ('K', 0, k, j + 1), base)                        # C·K^k
            _add(out, ('K', 0, k + 1, j), _q(-1) * base)               # q^{-1}K^{k+1}
            _add(out, ('K', 0, k - 1, j), _q(1) * base)                # q K^{k-1}
        else:
            # K^k E^{p-1}, K^k E^{p-1}K = q^{2(p-1)}K^{k+1}E^{p-1}, ...K^{-1}=q^{-2(p-1)}K^{k-1}E^{p-1}
            _add(out, ('E', p - 1, k, j + 1), base)
            _add(out, ('E', p - 1, k + 1, j), _q(1 - 2 * p) * _q(2 * (p - 1)) * base)
            _add(out, ('E', p - 1, k - 1, j), _q(2 * p - 1) * _q(-2 * (p - 1)) * base)
    return out


def _left_mult_gen(g, X: dict) -> dict:
    """Left-multiply generator g in {'E','F','K','Ki'} onto element X (dict)."""
    out: dict = {}
    for mono, coeff in X.items():
        if g == 'K':
            part = _lm_K(+1, mono, coeff)
        elif g == 'Ki':
            part = _lm_K(-1, mono, coeff)
        elif g == 'E':
            part = _lm_E(mono, coeff)
        else:
            part = _lm_F(mono, coeff)
        for kk, cc in part.items():
            _add(out, kk, cc)
    return out


def _mono_word(mono) -> list:
    """Generator word (left-to-right) for a monomial C^j·M, EXCLUDING C^j
    (the central power is added to the result's j).  K^k -> k×K (or |k|×Ki);
    K^k E^p -> word(K^k)+p×E ; F^p K^k -> p×F + word(K^k)."""
    side, p, k, j = mono
    kw = (['K'] * k) if k >= 0 else (['Ki'] * (-k))
    if side == 'K':
        return kw
    if side == 'E':
        return kw + ['E'] * p
    return ['F'] * p + kw


def _mono_mult(m1, c1: LaurentPoly, m2, c2: LaurentPoly) -> dict:
    """Multiply two monomials: (C^{j1} M1)·(C^{j2} M2).  Apply M1's generator
    word (left) onto the element {M2: c1 c2}, then add j1 to the j of every
    output term (C central)."""
    j1 = m1[3]
    word = _mono_word(m1)
    X = {(m2[0], m2[1], m2[2], m2[3]): c1 * c2}
    for g in reversed(word):                 # apply rightmost generator first
        X = _left_mult_gen(g, X)
    if j1:
        X = {(s, p, k, j + j1): c for (s, p, k, j), c in X.items()}
    return X


# --- canonical (bar-invariant) basis layer ----------------------------------
# Public canonical label: ('K', n, j) | ('E', a, b, j) | ('F', a, b, j), with
# j>=0 the Casimir power C^j.  Bar-invariant normalisation:
#   K^n          (= internal K^n)
#   E_{a,b} = q^{-ab} E^a K^b = q^{ab} (K^b E^a)   [internal ('E',a,b)]
#   F_{a,b} = q^{ab} F^a K^b                       [internal ('F',a,b)]
# so canonical coeff = q^{-a b} * internal coeff for E/F sides (K^b E^a = q^{-ab}E_{a,b}).

def _internal_to_canonical(X: dict) -> dict:
    """internal monomials {(side,p,k,j): LaurentPoly} -> canonical {label: LaurentPoly}."""
    out: dict = {}
    for (side, p, k, j), c in X.items():
        if side == 'K':
            lab = ('K', k, j)
            coeff = c
        elif side == 'E':                    # internal K^k E^p ; E_{p,k}=q^{pk}K^kE^p
            lab = ('E', p, k, j)
            coeff = _q(-p * k) * c
        else:                                # internal F^p K^k ; F_{p,k}=q^{pk}F^pK^k
            lab = ('F', p, k, j)
            coeff = _q(-p * k) * c
        _add(out, lab, coeff)
    return out


def _canonical_to_internal(lab, coeff: LaurentPoly):
    """canonical label -> (internal monomial key, internal coeff)."""
    kind = lab[0]
    if kind == 'K':
        _, n, j = lab
        return ('K', 0, n, j), coeff
    if kind == 'E':
        _, a, b, j = lab
        return ('E', a, b, j), _q(a * b) * coeff      # E_{a,b}=q^{ab}K^bE^a
    _, a, b, j = lab
    return ('F', a, b, j), _q(a * b) * coeff          # F_{a,b}=q^{ab}F^aK^b


def multiply_canonical(a, b) -> dict:
    """Multiply two canonical basis labels -> {canonical label: LaurentPoly}."""
    ma, ca = _canonical_to_internal(a, LaurentPoly.one())
    mb, cb = _canonical_to_internal(b, LaurentPoly.one())
    return _internal_to_canonical(_mono_mult(ma, ca, mb, cb))


def bar_element(elt: dict) -> dict:
    """Bar on a canonical Element: every canonical basis label is bar-invariant
    (the q^{∓ab} normalisation; C, K, E, F all bar-fixed), so bar conjugates the
    Laurent coefficients q -> q^{-1}."""
    return {lab: c.bar() if hasattr(c, "bar") else
            LaurentPoly({-e: v for e, v in c._coeffs.items()})
            for lab, c in elt.items()}


# canonical generator labels (j=0)
E_LAB = ('E', 1, 0, 0)        # E_{1,0} = E
F_LAB = ('F', 1, 0, 0)        # F_{1,0} = F
def K_LAB(n=1): return ('K', n, 0)
def C_LAB(j=1): return ('K', 0, j)        # C^j (Casimir^j · 1)


# --- rho: Lusztig's braid symmetry (a sign-free permutation of labels) -------
# ρ(K)=K^{-1}, ρ(E)=q^{-1}FK^{-1}=F_{1,-1}, ρ(F)=qKE=E_{1,1}; ρ(C)=C.
# As an algebra automorphism on the canonical basis (derived):
#   ρ(K^n)     = K^{-n}
#   ρ(E_{a,b}) = F_{a,-a-b}
#   ρ(F_{a,b}) = E_{a, a-b}
# ρ² (E_{a,b}) = E_{a, b+2a}  — infinite order (the braid Z, covering K↦K^{-1}).

def rho_label(lab):
    kind = lab[0]
    if kind == 'K':
        _, n, j = lab
        return ('K', -n, j)
    if kind == 'E':
        _, a, b, j = lab
        return ('F', a, -a - b, j)
    _, a, b, j = lab
    return ('E', a, a - b, j)


def rho_inverse_label(lab):
    kind = lab[0]
    if kind == 'K':
        _, n, j = lab
        return ('K', -n, j)
    if kind == 'F':                          # ρ(E_{a,-a-c}) = F_{a,c}  -> ρ^{-1}(F_{a,c})=E_{a,-a-c}
        _, a, c, j = lab
        return ('E', a, -a - c, j)
    _, a, c, j = lab                         # ρ(F_{a,a-c}) = E_{a,c}   -> ρ^{-1}(E_{a,c})=F_{a,a-c}
    return ('F', a, a - c, j)


# --- KAlgebra wrapper --------------------------------------------------------

from kalgebra import KAlgebra, Element       # noqa: E402
from zplus_ring import TrivialZPlusRing       # noqa: E402


class UqSL2PBW(KAlgebra):
    """Standalone `U_𝖖(sl_2)` (central quotient) as a `KAlgebra` on the PBW basis.

    Canonical basis labels (all bar-invariant): ``('K', n, j)`` = `C^j K^n`,
    ``('E', a, b, j)`` = `C^j E_{a,b}` (a≥1), ``('F', a, b, j)`` = `C^j F_{a,b}`
    (a≥1), with the central Casimir power `j≥0` (`C = χ_1`).  `multiply` is the
    PBW straightening (central-quotient relations); it is `Z[𝖖^±]`-valued since
    `C` sits in the labels.  `ρ` is Lusztig's braid.

    `coefficient_ring` is `Z` here (the Casimir is a label coordinate); the
    SU(2)-flavoured refinement `C = χ_1 = μ+μ^{-1}` (and the Cartan-sector Schur
    trace `Tr K^n = [x^n]G(x,μ)`) is supplied by `SQED2SampleKAlgebra` in
    `samples.py`, which reuses this PBW straightener.
    """

    def coefficient_ring(self):
        return TrivialZPlusRing()

    def identity(self):
        return ('K', 0, 0)

    def multiply(self, a, b) -> Element:
        return Element(dict(multiply_canonical(a, b)))

    def rho(self, a):
        return rho_label(a)

    def rho_inverse(self, a):
        return rho_inverse_label(a)

    # Unflavoured (Trivial coefficient ring): the flavour-lift coordinate and
    # _label_section_decompose are inherited from the KAlgebra universal
    # trivial-flavour treatment.

    def trace(self, a, K: int = 20):
        # Trace localises to the Cartan sector (Tr vanishes on net E/F charge);
        # the surviving Tr(K^n) is the Schur index [x^n]G(x,μ), which requires
        # the SU(2) flavour layer (C=χ_1).  Supplied by SQED2SampleKAlgebra in
        # samples.py; the bare (Z-coefficient) algebra defers it.
        raise NotImplementedError(
            "UqSL2PBW.trace: the Cartan-sector Schur trace Tr(K^n)=[x^n]G(x,μ) "
            "needs the SU(2) flavour layer (C=χ_1); see SQED2SampleKAlgebra "
            "in samples.py.")


if __name__ == "__main__":
    A = UqSL2PBW()
    E, F, K = E_LAB, F_LAB, K_LAB(1)
    print("UqSL2PBW — standalone U_𝖖(sl_2) on the PBW basis")
    print("  E*F =", {l: str(c) for l, c in A.multiply(E, F).terms.items()})
    print("  F*E =", {l: str(c) for l, c in A.multiply(F, E).terms.items()})
    print("  rho(E) =", A.rho(E), "  rho(F) =", A.rho(F), "  rho(K) =", A.rho(K))
