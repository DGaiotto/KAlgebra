"""checks_kq.py — the design record adapters for the first claims of the draft.

The claims: (a) "does the
KAlgebra contract express correctly the K_q algebra axioms?"; (b) the quantum
torus is a K_q-algebra analytically, so "the real check is that the quantum
torus KAlgebra implements correctly the quantum torus K_q algebra"; (c) the
pentagon: "is the stand-alone Pentagon class implementing correctly the
definition, and is it actually a KAlgebra? ... all formulae in the pentagon
example should be checked".

Every closed form the paper prints is recomputed here INDEPENDENTLY of the
repository's series types, with a small exact truncated-series class (integer
coefficients, Laurent exponents), and compared coefficient by coefficient with
what the classes return.  Positive controls (the paper's printed expansions,
the pentagonal-number theorem for the Euler product) run before any
comparison is trusted.

Adapters take the environment name and return
    {"checks": [{"name", "ok", "note"}...], "population": {...},
     "controls": {"positive": ..., "negative": ...}, "notes": str, "inputs": []}
and are registered in ADAPTERS at the bottom.
"""
from __future__ import annotations

import itertools
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the release root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kalgebra import Element  # noqa: E402
from laurent_poly import LaurentPoly  # noqa: E402
from zplus_ring import RPowerSeries  # noqa: E402
from quantum_torus_kalgebra import QuantumTorusKAlg  # noqa: E402
from kalgebra_samples import PentagonKAlg, _pent_canon_key  # noqa: E402


# --------------------------------------------------------------------------
# An independent exact truncated series  Σ c_e q^e  (c_e ∈ Z, e ∈ Z, e <= K)
# --------------------------------------------------------------------------
class S:
    __slots__ = ("c", "K")

    def __init__(self, c: dict, K: int):
        self.K = K
        self.c = {e: int(v) for e, v in c.items() if v and e <= K}

    @classmethod
    def one(cls, K):
        return cls({0: 1}, K)

    @classmethod
    def q(cls, e, K, coeff=1):
        return cls({e: coeff}, K)

    def __add__(self, o):
        K = min(self.K, o.K)
        c = dict(self.c)
        for e, v in o.c.items():
            c[e] = c.get(e, 0) + v
        return S(c, K)

    def __neg__(self):
        return S({e: -v for e, v in self.c.items()}, self.K)

    def __sub__(self, o):
        return self + (-o)

    def __mul__(self, o):
        if isinstance(o, int):
            return S({e: v * o for e, v in self.c.items()}, self.K)
        K = min(self.K, o.K)
        c: dict = {}
        for e1, v1 in self.c.items():
            for e2, v2 in o.c.items():
                if e1 + e2 <= K:
                    c[e1 + e2] = c.get(e1 + e2, 0) + v1 * v2
        return S(c, K)

    def shift(self, k):
        return S({e + k: v for e, v in self.c.items()}, self.K + k)

    def trunc(self, K):
        return S({e: v for e, v in self.c.items() if e <= K}, K)

    def val(self):
        return min(self.c) if self.c else None

    def eq_to(self, o, K=None):
        K = min(self.K, o.K) if K is None else K
        return {e: v for e, v in self.c.items() if e <= K} == {e: v for e, v in o.c.items() if e <= K}

    def __repr__(self):
        body = " ".join(f"{v:+d}q^{e}" for e, v in sorted(self.c.items())) or "0"
        return body if self.K >= 10 ** 6 else f"{body} + O(q^{self.K + 1})"


def inv_one_minus(k: int, K: int) -> S:
    """1/(1 - q^k) = Σ_{m>=0} q^{km}, k > 0."""
    return S({k * m: 1 for m in range(K // k + 1)}, K)


def qpoch_n(k: int, n: int, K: int) -> S:
    """(q^k; q^k)_n = Π_{j=1}^{n} (1 - q^{kj})."""
    out = S.one(K)
    for j in range(1, n + 1):
        out = out * S({0: 1, k * j: -1}, K)
    return out


def inv_qpoch_n(k: int, n: int, K: int) -> S:
    out = S.one(K)
    for j in range(1, n + 1):
        out = out * inv_one_minus(k * j, K)
    return out


def qpoch_inf(k: int, K: int) -> S:
    """(q^k; q^k)_∞ truncated at q^K."""
    out = S.one(K)
    j = 1
    while k * j <= K:
        out = out * S({0: 1, k * j: -1}, K)
        j += 1
    return out


def nahm(exp_fn, K: int, k: int = 2) -> S:
    """Σ_n q^{exp_fn(n)} / (q^k; q^k)_n, summed while exp_fn(n) <= K."""
    out = S({}, K)
    n = 0
    while exp_fn(n) <= K:
        out = out + S.q(exp_fn(n), K) * inv_qpoch_n(k, n, K)
        n += 1
    return out


def product_over(K: int, factors) -> S:
    """Π 1/(1 - q^e) over the exponents e yielded by `factors(K)`."""
    out = S.one(K)
    for e in factors(K):
        out = out * inv_one_minus(e, K)
    return out


def chi0(K):  # Yang-Lee χ_0(q) = Π 1/((1-q^{5n-2})(1-q^{5n-3}))
    return product_over(K, lambda K: [e for n in range(1, K // 5 + 2) for e in (5 * n - 2, 5 * n - 3) if e <= K])


def chi1(K):  # χ_1(q) = Π 1/((1-q^{5n-1})(1-q^{5n-4}))
    return product_over(K, lambda K: [e for n in range(1, K // 5 + 2) for e in (5 * n - 1, 5 * n - 4) if e <= K])


def sub_q2(s: S) -> S:
    """q -> q^2."""
    return S({2 * e: v for e, v in s.c.items()}, 2 * s.K)


def series_div(num: S, den: S, K: int) -> S:
    """num/den as a Laurent series to order K (den must have a unit leading coefficient ±1)."""
    v = den.val()
    lead = den.c[v]
    assert lead in (1, -1), "series_div: leading coefficient must be ±1 for an integral quotient"
    d = S({e - v: c for e, c in den.c.items()}, den.K - v)  # normalised: constant term ±1
    n = S({e - v: c for e, c in num.c.items()}, num.K - v)
    out: dict = {}
    rem = dict(n.c)
    lo = min(rem) if rem else 0
    for e in range(lo, K + 1):
        coeff = rem.get(e, 0) * lead
        if coeff:
            out[e] = coeff
            for de, dc in d.c.items():
                if e + de <= K + 1:
                    rem[e + de] = rem.get(e + de, 0) - coeff * dc
    return S(out, K)


def rps_to_S(t, K=None) -> S:
    """A repo RPowerSeries over the trivial ring -> S (integer coefficients)."""
    K = t.K if K is None else K
    c = {}
    for e, v in t.coeffs.items():
        if hasattr(v, "terms"):  # RElement over a ZPlusRing
            terms = {k: x for k, x in v.terms.items() if x}
            if not terms:
                continue
            assert set(terms) <= {v.ring.one_basis()}, "rps_to_S: non-trivial flavour coefficient"
            v = terms.get(v.ring.one_basis(), 0)
        if v:
            c[e] = v
    return S(c, K)


def lp_to_S(p: LaurentPoly, K: int = 10**6) -> S:
    return S(dict(getattr(p, "_coeffs", {})), K)


def elem(terms: dict) -> Element:
    """{label: {exp: coeff}} -> Element."""
    return Element({lab: LaurentPoly(dict(c)) for lab, c in terms.items()})


def check(name, ok, note=""):
    return {"name": name, "ok": bool(ok), "note": note}


# --------------------------------------------------------------------------
# The K1..K5 verifier battery (the instruments), run on a label window
# --------------------------------------------------------------------------
VERIFIERS = [
    # axiom, method, arity, takes K
    ("K1", "verify_bar_involution", 2, False),
    ("K2", "verify_identity_in_basis", 0, False),
    ("K2", "verify_unit_law", 1, False),
    ("K2", "verify_associativity", 3, False),
    ("K3", "verify_rho_fixes_identity", 0, False),
    ("K3", "verify_rho_inverse", 1, False),
    ("K3", "verify_rho_is_automorphism", 2, False),
    ("K4", "verify_orthonormality", 2, True),
    ("K4", "verify_trace_pairing_faces", 2, True),
    ("K4", "verify_rho_twisted_trace", 2, True),
    ("K5", "verify_trace_intertwines_rho", 1, True),
    ("K5", "verify_pairing_rho_star_symmetric", 2, True),
]


def run_verifiers(A, labels, K, triple_labels=None):
    """{method: (n_pass, n_fail, first failing arguments)} over the window."""
    out = {}
    triple_labels = labels if triple_labels is None else triple_labels
    for _ax, name, arity, takesK in VERIFIERS:
        fn = getattr(A, name)
        n_pass = n_fail = 0
        first = None
        if arity == 0:
            args_iter = [()]
        elif arity == 1:
            args_iter = [(a,) for a in labels]
        elif arity == 2:
            args_iter = itertools.product(labels, labels)
        else:
            args_iter = itertools.product(triple_labels, triple_labels, triple_labels)
        for args in args_iter:
            ok = fn(*args, K=K) if takesK else fn(*args)
            if ok:
                n_pass += 1
            else:
                n_fail += 1
                if first is None:
                    first = args
        out[name] = (n_pass, n_fail, first)
    return out


def axiom_of(name):
    return next(ax for ax, n, _a, _k in VERIFIERS if n == name)


# --------------------------------------------------------------------------
# (a) def:kq/contract — transcription + discrimination
# --------------------------------------------------------------------------
TRANSCRIPTION = [
    ("K1", "ax:bar", "bar involution: antimultiplicative, q -> q^{-1}", "verify_bar_involution",
     "C^c_{ab}(q^{-1}) = C^c_{ba}(q) on every pair of the window (structure constants of the canonical basis)"),
    ("K2", "ax:basis", "canonical basis of bar-invariant elements including 1", "verify_identity_in_basis / verify_unit_law / verify_associativity",
     "1 is a basis label and a two-sided unit; products of basis elements are Z[q^{±1}]-combinations of basis elements (closure: every product label is a label); bar-invariance of the basis is K1's statement on the structure constants"),
    ("K3", "ax:rho", "algebra automorphism rho permuting the basis", "verify_rho_is_automorphism / verify_rho_inverse / verify_rho_fixes_identity",
     "rho(L_a L_b) = rho(L_a) rho(L_b); rho^{-1} rho = id; rho(1) = 1"),
    ("K4", "ax:trace, eq:orthonormality", "rho^2-twisted trace with I_{a,b} = Tr L_{rho(a)} L_b = Tr L_b L_{rho^{-1}(a)} = delta_{a,b} + O(q)", "verify_rho_twisted_trace / verify_trace_pairing_faces / verify_orthonormality",
     "Tr(L_a L_b) = Tr(rho^2(L_b) L_a); the two faces agree; I_{a,b} has no negative q-powers and I_{a,b}[q^0] = delta_{a,b}"),
    ("K5", "ax:rhotr", "Tr L_{rho(a)} = Tr L_a (hence I_{b,a} = I_{a,b})", "verify_trace_intertwines_rho / verify_pairing_rho_star_symmetric",
     "Tr(L_{rho(a)}) = *(Tr L_a) and I_{b,a} = *(I_{a,b}), * the identity on Z[q^{±1}] (the rep-ring duality with flavour)"),
]

B2 = [[0, 1], [-1, 0]]


class QT_K1(QuantumTorusKAlg):
    """Countermodel for K1: basis L'_a = q^{a_1^2} X_a — NOT bar-invariant.  Same
    algebra (the phase is a coboundary, so associativity and the unit survive),
    structure constants q^{<a,b> + 2 a_1 b_1}."""

    def multiply(self, a, b):
        a, b = tuple(a), tuple(b)
        target = tuple(x + y for x, y in zip(a, b))
        return Element({target: LaurentPoly({self._bracket(a, b) + 2 * a[0] * b[0]: 1})})


class QT_K2(QuantumTorusKAlg):
    """Countermodel for K2: the label 0 is in the basis but acts as q·(identity)."""

    def multiply(self, a, b):
        a, b = tuple(a), tuple(b)
        za, zb = all(x == 0 for x in a), all(x == 0 for x in b)
        target = tuple(x + y for x, y in zip(a, b))
        e = self._bracket(a, b) + (1 if (za != zb) else 0)
        return Element({target: LaurentPoly({e: 1})})


class QT_K3(QuantumTorusKAlg):
    """Countermodel for K3: rho(a_1, a_2) = (a_1, -a_2) is an ANTI-automorphism of the torus."""

    def rho(self, a):
        a = tuple(a)
        return (a[0], -a[1])

    def rho_inverse(self, a):
        return self.rho(a)


class QT_K4(QuantumTorusKAlg):
    """Countermodel for K4: the trace doubled, so I_{a,a}[q^0] = 2."""

    def trace(self, a, K=20):
        t = super().trace(a, K)
        return RPowerSeries(t.ring, {e: v * 2 for e, v in t.coeffs.items()}, K)


class QT_K5(QuantumTorusKAlg):
    """Countermodel for K5 on the rank-1 DEGENERATE torus (Γ = Z, <,> = 0, R = Z[μ^{±1}]):
    Tr X_a = μ^{|a|}, an even function of a, so Tr rho(X_a) = μ^{|a|} ≠ *(Tr X_a) = μ^{-|a|}
    while orthonormality (the unit-character component of I_{a,b}) still holds."""

    def trace(self, a, K=20):
        a = tuple(a)
        mu = self._R.basis_element((abs(a[0]),))
        return RPowerSeries(self._R, {0: mu}, K)


def check_contract(environment):
    K = 6
    labels2 = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    labels1 = [(i,) for i in (-2, -1, 0, 1, 2)]
    checks = []
    from kalgebra import KAlgebra
    for ax, lab, text, verifiers, formula in TRANSCRIPTION:
        names = [v.strip() for v in verifiers.split("/")]
        checks.append(check(f"transcription {ax} ({lab}): {text} <-> {verifiers}",
                            all(hasattr(KAlgebra, n) for n in names), formula))
    # positive control: the genuine quantum torus passes every verifier on the window
    genuine = run_verifiers(QuantumTorusKAlg(B2), labels2, K, triple_labels=[(0, 0), (1, 0), (0, 1), (-1, 1)])
    all_pass = all(f == 0 for _p, f, _x in genuine.values())
    checks.append(check("positive control: the Z^2 quantum torus passes all 12 verifiers on |γ_i| <= 1", all_pass,
                        "; ".join(f"{n} {p}/{p + f}" for n, (p, f, _x) in genuine.items())))
    # discrimination: one countermodel per axiom
    models = [
        ("K1", QT_K1(B2), labels2, "verify_bar_involution"),
        ("K2", QT_K2(B2), labels2, "verify_unit_law"),
        ("K3", QT_K3(B2), labels2, "verify_rho_is_automorphism"),
        ("K4", QT_K4(B2), labels2, "verify_orthonormality"),
        ("K5", QT_K5([[0]]), labels1, "verify_trace_intertwines_rho"),
    ]
    matrix = {}
    for ax, A, labels, target in models:
        res = run_verifiers(A, labels, K, triple_labels=labels[:4])
        failing = sorted({axiom_of(n) for n, (p, f, _x) in res.items() if f})
        matrix[ax] = {n: f for n, (p, f, _x) in res.items()}
        tgt_fail = res[target][1] > 0
        checks.append(check(f"discrimination {ax}: the {ax}-countermodel is rejected by {target}", tgt_fail,
                            f"verifier groups that fire: {failing}; failures per verifier: "
                            + ", ".join(f"{n} {f}" for n, f in matrix[ax].items() if f)))
    only = {ax: sorted({axiom_of(n) for n, f in m.items() if f}) for ax, m in matrix.items()}
    notes = ("Discrimination matrix (countermodel -> axiom groups whose verifiers fire): "
             + "; ".join(f"{ax} -> {v}" for ax, v in only.items())
             + ".  Where more than the targeted group fires the axioms are entangled in the construction, not the verifier: "
               "a non-bar-invariant basis also breaks orthonormality (I_{a,a} = q^{2a_1^2}), an anti-automorphism rho breaks "
               "I_{a,a} for a_1 != 0, a q-scaled unit breaks the bar symmetry of the pairs (1, b).  K4 and K5 are isolated: "
               "the doubled trace fails orthonormality alone, and on the degenerate torus the even flavour trace fails "
               "rho-equivariance alone, so K5 is independent of K1-K4 as a statement about flavoured algebras.")
    return {"checks": checks, "population": {"toy realisations": 6, "window": "rank 2: |γ_i| <= 1 (9 labels); rank 1: |a| <= 2", "K": K},
            "controls": {"positive": "genuine torus passes all verifiers", "negative": "each countermodel rejected by its axiom's verifier"},
            "notes": notes, "inputs": []}


# --------------------------------------------------------------------------
# (b) ex:qt — QuantumTorusKAlg implements the paper's quantum torus verbatim
# --------------------------------------------------------------------------
def bracket(B, a, b):
    return sum(a[i] * B[i][j] * b[j] for i in range(len(a)) for j in range(len(a)))


def check_quantum_torus(environment):
    K = 10
    checks = []
    # positive control on the independent series: Euler's pentagonal-number theorem for (q^2;q^2)_∞
    euler = qpoch_inf(2, 40)
    pent = S({}, 40)
    for k in range(-5, 6):
        e = 2 * (k * (3 * k - 1) // 2)
        if 0 <= e <= 40:
            pent = pent + S.q(e, 40, (-1) ** k)
    checks.append(check("positive control: independent (q^2;q^2)_∞ equals the pentagonal-number series to q^40", euler.eq_to(pent)))
    pairings = {
        "rank 2, <e1,e2> = 1 (the pentagon's IR torus)": [[0, 1], [-1, 0]],
        "rank 2, <e1,e2> = 2": [[0, 2], [-2, 0]],
        "rank 4, block (1, 3)": [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 3], [0, 0, -3, 0]],
    }
    n_rel = n_rho = n_tr = n_ip = 0
    for name, B in pairings.items():
        n = len(B)
        Q = QuantumTorusKAlg(B)
        rng = (-1, 0, 1)
        labels = list(itertools.product(rng, repeat=n))
        zero = tuple([0] * n)
        # relations X_a X_b = q^{<a,b>} X_{a+b}, exactly
        ok_rel = all(Q.multiply(a, b) == elem({tuple(x + y for x, y in zip(a, b)): {bracket(B, a, b): 1}})
                     for a in labels for b in labels)
        n_rel += len(labels) ** 2
        # rho(γ) = -γ, rho^2 = id
        ok_rho = all(Q.rho(a) == tuple(-x for x in a) and Q.rho_inverse(a) == tuple(-x for x in a) for a in labels)
        n_rho += len(labels)
        # trace: (q^2;q^2)_∞^{rk} δ_{γ,0}
        pref = S.one(K)
        for _ in range(n):
            pref = pref * qpoch_inf(2, K)
        ok_tr = all(rps_to_S(Q.trace(a, K)).eq_to(pref if a == zero else S({}, K), K) for a in labels)
        n_tr += len(labels)
        # I_{γ,γ'} = (q^2;q^2)_∞^{rk} δ_{γ,γ'}
        ok_ip = all(rps_to_S(Q.inner_product(a, b, K)).eq_to(pref if a == b else S({}, K), K) for a in labels for b in labels)
        n_ip += len(labels) ** 2
        checks.append(check(f"{name}: X_γ X_γ' = q^<γ,γ'> X_(γ+γ') on all {len(labels)**2} pairs", ok_rel))
        checks.append(check(f"{name}: rho(γ) = -γ and rho^-1(γ) = -γ on {len(labels)} labels", ok_rho))
        checks.append(check(f"{name}: Tr X_γ = (q^2;q^2)_∞^{n} δ_(γ,0) to q^{K} (independent series)", ok_tr))
        checks.append(check(f"{name}: I_(γ,γ') = (q^2;q^2)_∞^{n} δ_(γ,γ') to q^{K} on all pairs", ok_ip))
        if n == 2:
            res = run_verifiers(Q, labels, K=6, triple_labels=[(0, 0), (1, 0), (0, 1), (-1, 1)])
            checks.append(check(f"{name}: K1-K5 verifiers all pass (consistency witness of def:kq)",
                                all(f == 0 for _p, f, _x in res.values()),
                                "; ".join(f"{k} {p}/{p + f}" for k, (p, f, _x) in res.items())))
    # negative control: a wrong prefactor exponent is detected
    Q = QuantumTorusKAlg(B2)
    wrong = qpoch_inf(2, K)  # exponent 1 instead of 2
    checks.append(check("negative control: the rank-2 trace differs from (q^2;q^2)_∞^1", not rps_to_S(Q.trace((0, 0), K)).eq_to(wrong, K)))
    return {"checks": checks, "population": {"pairings": list(pairings), "labels": "|γ_i| <= 1", "pairs checked": n_rel, "K": K},
            "controls": {"positive": "pentagonal-number theorem on the independent Euler product", "negative": "wrong prefactor exponent detected"},
            "notes": "The bar involution fixing X_γ is K1 on structure constants that are pure q-powers: q^<a,b> -> q^-<a,b> = q^<b,a>.",
            "inputs": []}


# --------------------------------------------------------------------------
# (c) the pentagon
# --------------------------------------------------------------------------
def L(i, a=1, b=0):
    return _pent_canon_key(i, a, b)


def pent_window(max_ab):
    labs = {(0, 0, 0)}
    for i in range(5):
        for a in range(0, max_ab + 1):
            for b in range(0, max_ab + 1 - a):
                labs.add(_pent_canon_key(i, a, b))
    return sorted(labs)


class PentagonPerturbedSeed(PentagonKAlg):
    """Negative control: Tr L_i perturbed by +q^3 (the ratio Tr L_1 / Tr 1 wrong at order q^3)."""

    def _trace_residual(self, seed_label, K):
        t = super()._trace_residual(seed_label, K)
        if seed_label == (0, 1, 0):
            c = dict(t.coeffs)
            c[3] = c.get(3, 0) + 1
            return RPowerSeries(t.ring, c, K)
        return t


def tr(P, label, K):
    return rps_to_S(P.trace(label, K), K)


def tr_elem(P, x: Element, K, widen=40):
    return rps_to_S(P.trace_element(x, K + widen), K + widen).trunc(K)


def check_pentagon_implementation(environment):
    P = PentagonKAlg()
    K = 24
    checks = []
    one = elem({(0, 0, 0): {0: 1}})
    # relations, all i in Z/5
    ok_swap = all(P.multiply(L(i + 1), L(i)) == elem({lab: {e + 2: c for e, c in lp_to_S(cf).c.items()}
                                                    for lab, cf in P.multiply(L(i), L(i + 1)).terms.items()}) for i in range(5))
    ok_pl1 = all(P.multiply(L(i + 1), L(i - 1)) == elem({(0, 0, 0): {0: 1}, L(i): {1: 1}}) for i in range(5))
    ok_pl2 = all(P.multiply(L(i - 1), L(i + 1)) == elem({(0, 0, 0): {0: 1}, L(i): {-1: 1}}) for i in range(5))
    checks.append(check("relation L_{i+1} L_i = q^2 L_i L_{i+1}, i in Z/5", ok_swap))
    checks.append(check("relation L_{i+1} L_{i-1} = 1 + q L_i, i in Z/5", ok_pl1))
    checks.append(check("relation L_{i-1} L_{i+1} = 1 + q^{-1} L_i, i in Z/5", ok_pl2))
    checks.append(check("cyclic notation L_{i+5} = L_i (labels reduce mod 5)", all(L(i + 5) == L(i) for i in range(5))))
    # canonical basis L_{i;a,b} = q^{ab} L_i^a L_{i+1}^b  <=>  L_i^a L_{i+1}^b = q^{-ab} L_{i;a,b}
    ok_basis = all(P.multiply(L(i, a), L(i + 1, b)) == elem({(i, a, b): {-a * b: 1}}) for i in range(5) for a in range(1, 5) for b in range(1, 5))
    ok_pow = all(P.multiply(L(i, a - 1) if a > 1 else (0, 0, 0), L(i)) == elem({L(i, a): {0: 1}}) for i in range(5) for a in range(1, 6))
    checks.append(check("canonical basis: L_i^a L_{i+1}^b = q^{-ab} L_{i;a,b}, 1 <= a,b <= 4, all i", ok_basis))
    checks.append(check("pure powers: L_i^{a-1} L_i = L_i^a is a basis element, a <= 5", ok_pow))
    checks.append(check("the identity is the basis element (0,0,0) and 1·L_i = L_i·1 = L_i", all(P.multiply((0, 0, 0), L(i)) == elem({L(i): {0: 1}}) and P.multiply(L(i), (0, 0, 0)) == elem({L(i): {0: 1}}) for i in range(5))))
    # rho(L_i) = L_{i+2}, order exactly 5, on the whole window
    W = pent_window(3)
    ok_rho_gen = all(P.rho(L(i)) == L(i + 2) and P.rho_inverse(L(i)) == L(i - 2) for i in range(5))

    def rho_k(lab, k):
        for _ in range(k):
            lab = P.rho(lab)
        return lab
    ok_rho_ab = all(P.rho(_pent_canon_key(i, a, b)) == _pent_canon_key(i + 2, a, b) for (i, a, b) in W)
    ok_order = all(rho_k(lab, 5) == lab for lab in W) and all(any(rho_k(lab, k) != lab for lab in W) for k in range(1, 5))
    checks.append(check("rho(L_i) = L_{i+2} and rho^{-1}(L_i) = L_{i-2}", ok_rho_gen))
    checks.append(check("rho(L_{i;a,b}) = L_{i+2;a,b} on the window a+b <= 3", ok_rho_ab))
    checks.append(check("rho has order exactly 5 on the window (rho^5 = id, rho^k != id for k = 1..4)", ok_order))
    # negative control on the comparison: an altered relation is detected
    checks.append(check("negative control: L_{i+1} L_{i-1} != 1 + q^2 L_i is detected", not any(P.multiply(L(i + 1), L(i - 1)) == elem({(0, 0, 0): {0: 1}, L(i): {2: 1}}) for i in range(5))))
    return {"checks": checks, "population": {"generators": 5, "basis window": "a, b <= 4 for the basis form; a+b <= 3 for rho", "K": K},
            "controls": {"positive": "the paper's relations reproduced on all i", "negative": "altered relation detected"},
            "notes": "The stand-alone class is kalgebra_samples.PentagonKAlg (a ConeKAlgebra: cone data pentagon_cone_data.PENTAGON_CONE_DATA; trace seeds inline).  These checks are the specification a Lean formalisation of the pentagon example (a local session, user 2026-09-21) has to meet.",
            "inputs": []}


def check_pentagon_trace_seeds(environment):
    P = PentagonKAlg()
    K = 30
    checks = []
    Tr1 = tr(P, (0, 0, 0), K)
    TrL = [tr(P, L(i), K) for i in range(5)]
    # printed leading terms (positive controls on the reading of the paper)
    checks.append(check("printed: Tr 1 = 1 + q^4 + q^6 + q^8 + q^10 + ...", Tr1.eq_to(S({0: 1, 4: 1, 6: 1, 8: 1, 10: 1}, 11), 11)))
    checks.append(check("printed: Tr L_i = -q - q^7 - q^9 - q^11 + ...", TrL[0].eq_to(S({1: -1, 7: -1, 9: -1, 11: -1}, 12), 12)))
    # Nahm sums, computed independently
    nahm1 = nahm(lambda n: 2 * n * (n + 1), K)
    nahmL = -nahm(lambda n: 2 * (n + 1) ** 2 - 1, K)
    checks.append(check(f"Tr 1 = Σ q^(2n(n+1))/(q^2;q^2)_n to q^{K}", Tr1.eq_to(nahm1, K)))
    checks.append(check(f"Tr L_i = -Σ q^(2(n+1)^2-1)/(q^2;q^2)_n to q^{K}, all five i", all(t.eq_to(nahmL, K) for t in TrL)))
    # Yang-Lee characters: printed expansions, product = sum (Rogers-Ramanujan), and the trace relations
    c0, c1 = chi0(K), chi1(K)
    checks.append(check("printed: χ_0 = 1 + q^2 + q^3 + q^4 + q^5 + 2q^6 + 2q^7 + 3q^8 + 3q^9 + 4q^10", c0.eq_to(S({0: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3, 9: 3, 10: 4}, 10), 10)))
    checks.append(check("printed: χ_1 = 1 + q + q^2 + q^3 + 2q^4 + 2q^5 + 3q^6 + 3q^7 + 4q^8 + 5q^9", c1.eq_to(S({0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 5}, 9), 9)))
    rr0 = nahm(lambda n: n * (n + 1), K, k=1)
    rr1 = nahm(lambda n: n * n, K, k=1)
    checks.append(check(f"Rogers-Ramanujan: χ_0 product = Σ q^(n(n+1))/(q;q)_n and χ_1 product = Σ q^(n^2)/(q;q)_n to q^{K}", c0.eq_to(rr0, K) and c1.eq_to(rr1, K)))
    checks.append(check(f"Tr 1 = χ_0(q^2) to q^{K}", Tr1.eq_to(sub_q2(c0).trunc(K), K)))
    checks.append(check(f"Tr L_i = q^-1 (χ_0(q^2) - χ_1(q^2)) to q^{K}", all(t.eq_to((sub_q2(c0) - sub_q2(c1)).shift(-1).trunc(K), K) for t in TrL)))
    checks.append(check("negative control: the perturbed-seed class differs from the Nahm sum", not tr(PentagonPerturbedSeed(), L(0), K).eq_to(nahmL, K)))
    return {"checks": checks, "population": {"K": K, "seeds": 2, "orbit": 5},
            "controls": {"positive": "the paper's printed expansions", "negative": "perturbed seed detected"},
            "notes": "Both closed forms of each seed (Nahm sum, Yang-Lee product) recomputed independently of the repository's series types.", "inputs": []}


def check_pentagon_is_kalgebra(environment):
    P = PentagonKAlg()
    K = 8
    W = pent_window(3)
    checks = []
    res = run_verifiers(P, W, K, triple_labels=[L(i) for i in range(5)] + [(0, 0, 0)])
    for name, (p, f, first) in res.items():
        checks.append(check(f"{axiom_of(name)} {name} on the window a+b <= 3 ({p + f} instances)", f == 0, "" if f == 0 else f"first failure at {first}"))
    # closure: every product of window labels decomposes on canonical labels with Laurent coefficients
    ok_closure = True
    n_prod = 0
    for a in W:
        for b in W:
            x = P.multiply(a, b)
            n_prod += 1
            for lab, cf in x.terms.items():
                i, u, v = lab
                if _pent_canon_key(i, u, v) != lab or u < 0 or v < 0 or not isinstance(cf, LaurentPoly):
                    ok_closure = False
    checks.append(check(f"K2 closure: all {n_prod} products of window labels lie in the Z[q^±1]-span of canonical labels", ok_closure))
    # the trace is pinned by the axioms: a seed perturbed at order q^3 is caught by orthonormality on the paper's
    # own pair, Tr rho(L_{i-2}) L_i = Tr L_i^2 = O(q) (the display after the five cyclicity families), and by
    # off-diagonal pairs of the window; on the DIAGONAL pure-power pairs (L^n, L^n) it is invisible, because the
    # product rho(L^n) L^n carries positive q-powers that push the perturbation above order zero.
    Pp = PentagonPerturbedSeed()
    paper_pair = all(not Pp.verify_orthonormality(L(i - 2), L(i), K=12) for i in range(5))
    Wsmall = [(0, 0, 0)] + [L(i) for i in range(5)] + [L(i, 2) for i in range(5)]
    caught = [(a, b) for a in Wsmall for b in Wsmall if not Pp.verify_orthonormality(a, b, K=12)]
    diag = [n for n in range(1, 6) if not Pp.verify_orthonormality(L(0, n), L(0, n), K=12)]
    checks.append(check("negative control: Tr L_i + q^3 fails orthonormality on the paper's pair (L_{i-2}, L_i), i.e. Tr L_i^2 = O(q) fails, all i", paper_pair,
                        f"{len(caught)} failing pairs among identity, generators and squares, e.g. {caught[:3]}; diagonal pure powers catching it: {diag} (none expected)"))
    # the presentations are one algebra (finite zoo object): matched sample labels per realisation, by transport
    try:
        from finite_kalgebra_objects import kalgebra_object
        O = kalgebra_object("pentagon")
        keys = list(O.keys())
        src = "closed-form"
        std = [(0, 0, 0)] + [L(i) for i in range(5)] + [L(0, 2), L(1, 1, 1)]
        # warm-up: the frozen-cone presentation builds its normalisation table incrementally and asks for the
        # generator pairs to be multiplied before any other label is transported into it
        gens = {k: [next(iter(O.transport(L(i), src, k).terms)) if k != src else L(i) for i in range(5)] for k in keys}
        for k in keys:
            R_ = O.realization(k)
            for g in gens[k]:
                for h in gens[k]:
                    try:
                        R_.multiply(g, h)
                    except Exception:  # noqa: BLE001 — warm-up only
                        pass
        samples = {k: [] for k in keys}
        kept = []
        for lab in std:
            images = {}
            for k in keys:
                if k == src:
                    images[k] = lab
                    continue
                el = O.transport(lab, src, k)
                if len(el.terms) != 1:
                    images = None
                    break
                images[k] = next(iter(el.terms))
            if images is not None:
                kept.append(lab)
                for k in keys:
                    samples[k].append(images[k])
        info = f"realizations {keys}; {len(kept)} matched sample labels"
        pw = O.verify_pairwise(samples, pairs=True, trace_K=6)
        bad = [(e, [c for c, v in r.items() if not v]) for e, r in pw.items() if not all(r.values())]
        checks.append(check(f"the presentations are one algebra: KAlgebraIso witnesses on {len(pw)} edges (multiply, trace, rho transported)", bool(pw) and not bad, info + (f"; failing: {bad}" if bad else "")))
        coh = O.verify_coherence(samples, max_path_len=4)
        checks.append(check("the presentations are one algebra: path-independence of the transports (coherence)", bool(coh), info))
    except Exception as ex:  # noqa: BLE001 — record the failure, do not hide it
        checks.append(check("the presentations are one algebra (finite_kalgebra_objects pentagon)", False, f"{type(ex).__name__}: {ex}"))
    return {"checks": checks, "population": {"window": f"{len(W)} labels, a+b <= 3, all i", "K": K},
            "controls": {"positive": "all verifiers pass on the window", "negative": "perturbed seed caught by orthonormality"},
            "notes": "All five axioms are EMERGENT on this hand-built presentation: the product is the cone data of the relations, the trace is the closed-form seeds plus the twisted-cyclicity reducer, so orthonormality with these seeds is a genuine check that the paper's traces satisfy δ + O(q).", "inputs": []}


def check_pentagon_recursion(environment):
    P = PentagonKAlg()
    K = 16
    checks = []
    Tr1 = tr(P, (0, 0, 0), K + 40)
    TrL = tr(P, L(0), K + 40)

    def trpow(i, n):  # Tr L_i^n
        return tr(P, L(i, n) if n > 0 else (0, 0, 0), K + 40)

    # eq:izero  Tr L_i^a L_{i+1}^b = Tr L_i^{a+1} L_{i+1}^{b-1}   (products as elements)
    ok_izero = True
    for i in range(5):
        for a in range(0, 4):
            for b in range(1, 4):
                lhs = tr_elem(P, P.multiply(L(i, a) if a else (0, 0, 0), L(i + 1, b)), K)
                rhs = tr_elem(P, P.multiply(L(i, a + 1), L(i + 1, b - 1) if b > 1 else (0, 0, 0)), K)
                ok_izero &= lhs.eq_to(rhs, K)
    checks.append(check("eq:izero  Tr L_i^a L_{i+1}^b = Tr L_i^{a+1} L_{i+1}^{b-1}, a <= 3, 1 <= b <= 3, all i", ok_izero))
    # consequence: Tr L_{i;a,b} = q^{ab} Tr L_i^{a+b}
    ok_ab = all(tr(P, (i, a, b), K + 40).eq_to(trpow(i, a + b).shift(a * b), K) for i in range(5) for a in range(1, 4) for b in range(1, 4))
    checks.append(check("Tr L_{i;a,b} = q^{ab} Tr L_i^{a+b}, 1 <= a,b <= 3, all i", ok_ab))
    # eq:pentarec  Tr L_i^n = q^{1-2n} Tr L_i^{n-1} + q^{2-2n} Tr L_i^{n-2}
    ok_rec = all(trpow(i, n).eq_to(trpow(i, n - 1).shift(1 - 2 * n) + trpow(i, n - 2).shift(2 - 2 * n), K) for i in range(5) for n in range(2, 9))
    checks.append(check("eq:pentarec  Tr L_i^n = q^{1-2n} Tr L_i^{n-1} + q^{2-2n} Tr L_i^{n-2}, 2 <= n <= 8, all i", ok_rec))
    checks.append(check("Tr L_i^2 = q^-2 Tr 1 + q^-3 Tr L_i", all(trpow(i, 2).eq_to(Tr1.shift(-2) + TrL.shift(-3), K) for i in range(5))))
    # the five families of twisted-cyclicity checks  Tr L_0 L_{i;a,b} = Tr L_{i;a,b} L_1
    ok_cyc = True
    n_cyc = 0
    for i in range(5):
        for a in range(0, 4):
            for b in range(0, 4):
                lab = _pent_canon_key(i, a, b)
                lhs = tr_elem(P, P.multiply(L(0), lab), K)
                rhs = tr_elem(P, P.multiply(lab, L(1)), K)
                ok_cyc &= lhs.eq_to(rhs, K)
                n_cyc += 1
    checks.append(check(f"twisted cyclicity on generators: Tr L_0 L_{{i;a,b}} = Tr L_{{i;a,b}} L_1, {n_cyc} instances (a,b <= 3, all i)", ok_cyc))
    # Tr L_1 = (-q + O(q^4)) Tr 1
    ratio = series_div(TrL, Tr1, 12)
    checks.append(check("Tr L_1 / Tr 1 = -q + O(q^4)", (ratio + S.q(1, 12)).val() is not None and (ratio + S.q(1, 12)).val() >= 4, f"ratio = {ratio}"))
    return {"checks": checks, "population": {"K": K, "powers": "n <= 8", "cyclicity instances": n_cyc},
            "controls": {"positive": "the recursion reproduces Tr L_i^2 from the seeds", "negative": "n.a. (identities between traces of one algebra)"},
            "notes": "Traces of products are computed at K+40 and compared to q^K, absorbing the negative q-powers q^{-ab} of the basis normalisation.", "inputs": []}


def AB_coefficients(n_max):
    """A_n, B_n with q^{n^2-1} Tr L^n = A_n Tr 1 + B_n Tr L, from eq:pentarec:
    T_n := q^{n^2-1} Tr L^n satisfies T_n = T_{n-1} + q^{2n-2} T_{n-2}, T_0 = q^{-1} Tr 1, T_1 = Tr L."""
    KK = 10 ** 6
    A = {0: S({-1: 1}, KK), 1: S({}, KK)}
    B = {0: S({}, KK), 1: S({0: 1}, KK)}
    for n in range(2, n_max + 1):
        A[n] = A[n - 1] + A[n - 2].shift(2 * n - 2)
        B[n] = B[n - 1] + B[n - 2].shift(2 * n - 2)
    return A, B


TABLE = {  # tab:pentagon-stabilization, verbatim
    3: ({1: 1}, {0: 1, 4: 1}),
    4: ({1: 1, 7: 1}, {0: 1, 4: 1, 6: 1}),
    5: ({1: 1, 7: 1, 9: 1}, {0: 1, 4: 1, 6: 1, 8: 1, 12: 1}),
    6: ({1: 1, 7: 1, 9: 1, 11: 1, 17: 1}, {0: 1, 4: 1, 6: 1, 8: 1, 10: 1, 12: 1, 14: 1, 16: 1}),
}


def check_pentagon_AB(environment):
    P = PentagonKAlg()
    K = 40
    checks = []
    A, B = AB_coefficients(14)
    Tr1 = tr(P, (0, 0, 0), K + 60)
    TrL = tr(P, L(0), K + 60)
    checks.append(check("tab:pentagon-stabilization rows n = 3..6 verbatim (A_n, B_n from the recursion)",
                        all(A[n].c == TABLE[n][0] and B[n].c == TABLE[n][1] for n in TABLE),
                        "; ".join(f"A_{n} = {A[n]}, B_{n} = {B[n]}" for n in (3, 4))))
    ok_def = all((tr(P, L(0, n), K + 60).shift(n * n - 1)).eq_to(A[n] * Tr1 + B[n] * TrL, K) for n in range(1, 11))
    checks.append(check(f"eq:ABdef  q^(n^2-1) Tr L^n = A_n Tr 1 + B_n Tr L_1 against the class trace, n <= 10, to q^{K}", ok_def))
    # the ratio is determined to O(q^{n^2}): r = -A_n/B_n + O(q^{n^2})
    r = series_div(TrL, Tr1, 60)
    orders = {}
    for n in range(2, 9):
        approx = series_div(-A[n], B[n], 60)
        d = (r - approx).val()
        orders[n] = d
    checks.append(check("the ratio Tr L_1/Tr 1 agrees with -A_n/B_n to O(q^{n^2}), n = 2..8", all(orders[n] is None or orders[n] >= n * n for n in orders),
                        "agreement order (first differing power) per n: " + ", ".join(f"n={n}: {orders[n]}" for n in orders)))
    vals = {n: tr(P, L(0, n), 60).val() for n in range(1, 9)}
    checks.append(check("Tr L^n has no negative powers of q and starts at order >= 1, n <= 8", all(v is not None and v >= 1 for v in vals.values()), "valuations: " + ", ".join(f"n={n}: q^{v}" for n, v in vals.items())))
    # continued fraction 1 - q Tr L / Tr 1 = 1 + q^2/(1 + q^4/(1 + ...))
    target = S.one(60) - r.shift(1)
    cf_orders = {}
    for depth in range(1, 9):
        # x_k = 1 + q^{2k}/x_{k+1}, innermost x = 1; with x = n/d: 1 + q^{2k} d/n = (n + q^{2k} d)/n
        n_, d_ = S.one(60), S.one(60)
        for k in range(depth, 0, -1):
            n_, d_ = n_ + S.q(2 * k, 60) * d_, n_
        conv = series_div(n_, d_, 60)
        cf_orders[depth] = (target - conv).val()
    checks.append(check("continued fraction 1 - q Tr L_i/Tr 1 = 1 + q^2/(1 + q^4/(1 + ...)): convergents approach to increasing order", all(cf_orders[d] is None or cf_orders[d] > cf_orders[d - 1] for d in range(2, 9) if cf_orders[d - 1] is not None),
                        "first differing power per depth: " + ", ".join(f"{d}: {v}" for d, v in cf_orders.items())))
    return {"checks": checks, "population": {"n": "<= 10 for eq:ABdef, <= 8 for the ratio, table rows 3..6", "K": K},
            "controls": {"positive": "the table rows reproduced from the recursion", "negative": "n.a."},
            "notes": "A_n, B_n are generated from eq:pentarec alone (T_n = T_{n-1} + q^{2n-2} T_{n-2}), independently of pentagon_trace.py, then compared both with the printed table and with the class's traces.", "inputs": []}


def check_pentagon_miracle(environment):
    P = PentagonKAlg()
    N = 16
    K = 80
    A, B = AB_coefficients(N)
    Tr1 = tr(P, (0, 0, 0), K)
    TrL = tr(P, L(0), K)
    sA = {n: (A[n] + TrL).val() for n in range(2, N + 1)}      # A_n -> -Tr L_1 : first power where A_n + Tr L differs
    sB = {n: (B[n] - Tr1).val() for n in range(2, N + 1)}      # B_n -> Tr 1

    def nondecreasing_unbounded(seq):
        vals = [seq[n] for n in sorted(seq)]
        if any(v is None for v in vals):
            return True
        return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)) and vals[-1] > vals[0] + (len(vals) - 1)
    lawA = all(sA[n] == 2 * n + 1 for n in range(3, N + 1))
    lawB = all(sB[n] == 2 * n + 2 for n in range(2, N + 1))
    checks = [
        check("A_n -> -Tr L_1: the agreement order is non-decreasing and unbounded over n = 2..16", nondecreasing_unbounded(sA),
              "first differing power: " + ", ".join(f"n={n}: {sA[n]}" for n in sA) + (";  measured law: 2n+1 for n >= 3" if lawA else "")),
        check("B_n -> Tr 1: the agreement order is non-decreasing and unbounded over n = 2..16", nondecreasing_unbounded(sB),
              "first differing power: " + ", ".join(f"n={n}: {sB[n]}" for n in sB) + (";  measured law: 2n+2 for n >= 2" if lawB else "")),
        check("each row extends the one above, termwise: A_n - A_{n-1} and B_n - B_{n-1} have non-negative coefficients, n = 3..16",
              all(all(v >= 0 for v in (A[n] - A[n - 1]).c.values()) and all(v >= 0 for v in (B[n] - B[n - 1]).c.values()) for n in range(3, N + 1)),
              "a consequence of the recursion A_n - A_{n-1} = q^{2n-2} A_{n-2}; note the new terms need not lie above the previous top term (A_7 adds q^13 below A_6's q^17)"),
        check("A_16 agrees with -Tr L_1 through q^30 and B_16 with Tr 1 through q^30", (sA[N] is None or sA[N] > 30) and (sB[N] is None or sB[N] > 30)),
    ]
    return {"checks": checks, "population": {"n": f"<= {N}", "K": K},
            "controls": {"positive": "the table's limit row", "negative": "n.a. (a measured convergence)"},
            "notes": "Measured, not derived: the orders grow without a visible bound over the window; the statement A_∞ = -Tr L_1, B_∞ = Tr 1 remains the paper's remark.", "inputs": []}


ADAPTERS = {
    "def:kq/contract": check_contract,
    "ex:qt": check_quantum_torus,
    "def:penta/implementation": check_pentagon_implementation,
    "def:penta/trace-seeds": check_pentagon_trace_seeds,
    "def:penta/existence": check_pentagon_is_kalgebra,
    "def:penta/recursion": check_pentagon_recursion,
    "def:penta/AB-table": check_pentagon_AB,
    "rem:miracle": check_pentagon_miracle,
}
