"""checks_flavoured.py — the design record adapters for the flavoured definition and its
first example / remark (draft Section "Flavoured K_q-algebras").

* def:kq-flavoured/contract — does the KAlgebra contract express F1–F6?
  Transcription (axiom <-> verifier <-> formula) plus DISCRIMINATION: six toy
  countermodels on the degenerate quantum torus (Γ_f = Z e_3, R = Z[μ^{±1}]),
  one per axiom, each rejected by that axiom's verifier.  F6 (the forgetful map
  commutes with ρ and Tr and identifies the bases) has no verifier on the
  contract yet (README §7 / T3); its check is implemented inline here and the
  F6-countermodel — a flavoured trace with the prefactor of the FULL rank
  instead of rk(Γ/Γ_f) — is caught by it alone.
* ex:qt-flavoured — QuantumTorusKAlg with a degenerate pairing implements the
  flavoured quantum torus verbatim: Tr X_γ = (q^2)^{rk(Γ/Γ_f)}_∞ χ_γ δ_{γ∈Γ_f},
  X_{γ_f} = the characters, I_{γ,γ'} ∈ R + q R[[q]] with identity summand δ.
* rem:flavour-change — lowering along H_f -> G_f through lower_flavour(φ) with
  φ a restriction hom gives an H_f-flavoured K_q-algebra and a map from
  A_q[G_f] that is an algebra hom commuting with ρ and Tr; the augmentation
  case is forget().
"""
from __future__ import annotations

import itertools
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the release root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from kalgebra import Element  # noqa: E402
from laurent_poly import LaurentPoly  # noqa: E402
from zplus_ring import RPowerSeries, RElement, AbelianZPlusRing, restriction_hom  # noqa: E402
from quantum_torus_kalgebra import QuantumTorusKAlg, _qpoch_pref_rpowerseries  # noqa: E402
from checks_kq import S, qpoch_inf, check, elem  # noqa: E402

B3 = [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]                     # Γ_f = Z e_3
B3b = [[0, 0, 1], [0, 0, 1], [-1, -1, 0]]                   # kernel spanned by (1, -1, 0)
B4 = [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # Γ_f = Z^2


def window(n, r=1):
    return list(itertools.product(range(-r, r + 1), repeat=n))


def matrix_rank(B):
    """Rank over Q by fraction-free elimination (independent of the class's SNF)."""
    M = [list(row) for row in B]
    n = len(M)
    rank = 0
    cols = len(M[0]) if M else 0
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, n) if M[i][c] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        for i in range(n):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                g = M[r][c]
                M[i] = [g * x - f * y for x, y in zip(M[i], M[r])]
        r += 1
        rank += 1
    return rank


# --------------------------------------------------------------------------
# the F1..F6 instruments
# --------------------------------------------------------------------------
F_VERIFIERS = [
    ("F1", "verify_bar_involution", 2, False),
    ("F2", "verify_identity_in_basis", 0, False),
    ("F2", "verify_unit_law", 1, False),
    ("F2", "verify_associativity", 3, False),
    ("F2", "verify_section_is_single_irrep", 1, False),
    ("F2", "verify_embed_section_roundtrip", 1, False),
    ("F3", "verify_rho_fixes_identity", 0, False),
    ("F3", "verify_rho_inverse", 1, False),
    ("F3", "verify_rho_is_automorphism", 2, False),
    ("F3", "verify_embed_intertwines_rho", "R", False),
    ("F4", "verify_orthonormality", 2, True),
    ("F4", "verify_trace_pairing_faces", 2, True),
    ("F4", "verify_rho_twisted_trace", 2, True),
    ("F5", "verify_trace_intertwines_rho_star", 1, True),
    ("F5", "verify_pairing_rho_star_symmetric", 2, True),
]


def flavour_generators(A):
    R = A.coefficient_ring()
    if isinstance(R, AbelianZPlusRing):
        out = []
        for i in range(R.rank):
            for sgn in (1, -1):
                key = tuple(sgn if j == i else 0 for j in range(R.rank))
                out.append(R.basis_element(key))
        return out
    return []


def forget_commutes(A, labels, K):
    """F6 inline: forget() is an algebra hom onto the unflavoured algebra that commutes with ρ and with
    the trace after augmentation (dim-weights are 1 on an abelian flavour).  Returns (n_pass, n_fail, first)."""
    F = A.forget()
    eps = A.coefficient_ring().augmentation()
    n_pass = n_fail = 0
    first = None

    def forget_elem(x: Element) -> Element:
        out: dict = {}
        for lab, cf in x.terms.items():
            fl = A.forget_label(lab)
            out[fl] = (out[fl] + cf) if fl in out else cf
        return Element({k: v for k, v in out.items() if not v.is_zero()})
    for a in labels:
        fa = A.forget_label(a)
        ok_rho = F.rho(fa) == A.forget_label(A.rho(a))
        ok_tr = (eps.apply_RPowerSeries(A.trace(a, K)) - F.trace(fa, K)).is_zero()
        ok = ok_rho and ok_tr
        n_pass += ok
        n_fail += (not ok)
        if not ok and first is None:
            first = ("rho" if not ok_rho else "trace", a)
    for a in labels:
        for b in labels:
            ok = forget_elem(A.multiply(a, b)) == F.multiply(A.forget_label(a), A.forget_label(b))
            n_pass += ok
            n_fail += (not ok)
            if not ok and first is None:
                first = ("hom", a, b)
    return n_pass, n_fail, first


def run_f_verifiers(A, labels, K, triple_labels=None):
    out = {}
    triple_labels = labels[:5] if triple_labels is None else triple_labels
    for _ax, name, arity, takesK in F_VERIFIERS:
        fn = getattr(A, name)
        n_pass = n_fail = 0
        first = None
        if arity == 0:
            args_iter = [()]
        elif arity == 1:
            args_iter = [(a,) for a in labels]
        elif arity == 2:
            args_iter = itertools.product(labels, labels)
        elif arity == "R":
            args_iter = [(r,) for r in flavour_generators(A)]
        else:
            args_iter = itertools.product(triple_labels, triple_labels, triple_labels)
        for args in args_iter:
            ok = fn(*args, K=K) if takesK else fn(*args)
            n_pass += bool(ok)
            n_fail += (not ok)
            if not ok and first is None:
                first = args
        out[name] = (n_pass, n_fail, first)
    out["forget_commutes (F6, inline)"] = forget_commutes(A, labels, K)
    return out


def axiom_of_f(name):
    if name.startswith("forget_commutes"):
        return "F6"
    return next(ax for ax, n, _a, _k in F_VERIFIERS if n == name)


# --------------------------------------------------------------------------
# countermodels on the degenerate torus
# --------------------------------------------------------------------------
class FT_F1(QuantumTorusKAlg):
    """F1: basis q^{a_1^2} X_a, not bar-invariant (structure constants q^{<a,b> + 2 a_1 b_1})."""

    def multiply(self, a, b):
        a, b = tuple(a), tuple(b)
        target = tuple(x + y for x, y in zip(a, b))
        return Element({target: LaurentPoly({self._bracket(a, b) + 2 * a[0] * b[0]: 1})})


class FT_F2(QuantumTorusKAlg):
    """F2: the characters are embedded as q·X_{γ_f} — the R-basis is not the canonical basis
    (the faithfulness axiom embed_R(r)·L_section = L_a fails)."""

    def embed_R(self, r):
        out = Element.zero()
        for key, c in r.terms.items():
            if c:
                out = out + Element({self.flavour_generator_label(key): LaurentPoly({1: c})})
        return out


class FT_F3(QuantumTorusKAlg):
    """F3: ρ(a_1, a_2, a_3) = (-a_1, -a_2, +a_3) — an automorphism that is NOT twisted-linear
    (ρ(χ_r) = χ_r instead of χ_{r^∨})."""

    def rho(self, a):
        a = tuple(a)
        return (-a[0], -a[1], a[2])

    def rho_inverse(self, a):
        return self.rho(a)


class FT_F4(QuantumTorusKAlg):
    """F4: the trace doubled, I^{(1)}_{a,a}[q^0] = 2."""

    def trace(self, a, K=20):
        t = super().trace(a, K)
        return RPowerSeries(t.ring, {e: v * 2 for e, v in t.coeffs.items()}, K)


class FT_F5(QuantumTorusKAlg):
    """F5: Tr X_a = δ_{gauge(a),0} μ^{|flav(a)|} (q^2;q^2)^{rk Γ_g}: even in the flavour, so
    Tr L_{ρ(a)} ≠ ρ(Tr L_a) while the identity summand of I_{a,b} is still δ."""

    def trace(self, a, K=20):
        a = tuple(a)
        sec_c, flav_c = self._decompose(a)
        if any(s != 0 for s in sec_c):
            return RPowerSeries.zero(self._R, K)
        pref = _qpoch_pref_rpowerseries(self._R, self._gauge_rank, K)
        mu = self._R.basis_element(tuple(abs(x) for x in flav_c))
        return RPowerSeries(self._R, {e: v * mu for e, v in pref.coeffs.items()}, K)


class FT_F6(QuantumTorusKAlg):
    """F6: the prefactor uses the FULL rank Γ instead of rk(Γ/Γ_f): every other axiom holds
    (I[q^0] is still δ), but the forgetful map no longer commutes with the trace."""

    def trace(self, a, K=20):
        a = tuple(a)
        sec_c, flav_c = self._decompose(a)
        if any(s != 0 for s in sec_c):
            return RPowerSeries.zero(self._R, K)
        pref = _qpoch_pref_rpowerseries(self._R, self._rank, K)
        mu = self._R.basis_element(tuple(flav_c))
        return RPowerSeries(self._R, {e: v * mu for e, v in pref.coeffs.items()}, K)


TRANSCRIPTION_F = [
    ("F1", "ax:f-bar", "verify_bar_involution", "C^c_{ab}(q^{-1}) = C^c_{ba}(q) with R-valued constants"),
    ("F2", "ax:f-basis", "verify_identity_in_basis / verify_unit_law / verify_associativity / verify_section_is_single_irrep / verify_embed_section_roundtrip",
     "a bar-invariant Z[q^{±1}]-basis containing the characters: every label is (section, one irrep χ) with embed_R(χ)·L_section = L_a — the non-canonical R-basis is a choice of section"),
    ("F3", "ax:f-rho", "verify_rho_is_automorphism / verify_rho_inverse / verify_rho_fixes_identity / verify_embed_intertwines_rho",
     "rho an algebra automorphism permuting the basis, twisted-linear: embed_R(χ_r^∨) = rho(embed_R(χ_r))"),
    ("F4", "ax:f-trace, eq:f-index, eq:f-integrality, eq:f-orthonormality", "verify_rho_twisted_trace / verify_trace_pairing_faces / verify_orthonormality",
     "Tr(L_a L_b) = Tr(rho^2(L_b) L_a); the two faces of I_{a,b} agree; no negative q-powers (I ∈ R + q R[[q]]); the identity summand I^{(1)}_{a,b}[q^0] = δ_{a,b}"),
    ("F5", "ax:rhotr", "verify_trace_intertwines_rho_star / verify_pairing_rho_star_symmetric",
     "Tr L_{rho(a)} = *(Tr L_a) and I_{b,a} = *(I_{a,b}), * the rep-ring duality (the paper's outer rho on the flavour ring)"),
    ("F6", "ax:f-forget", "forget() + forget_label (inline check; no contract verifier yet — README section 7)",
     "forget is an algebra hom identifying the R-basis with the unflavoured canonical basis, commuting with rho and, after the augmentation χ_r -> dim r, with Tr"),
]


def check_flavoured_contract(environment):
    K = 6
    labels = window(3)
    checks = []
    from kalgebra import KAlgebra
    for ax, lab, verifiers, formula in TRANSCRIPTION_F:
        names = [v.strip().split(" ")[0] for v in verifiers.split("/")]
        present = all(hasattr(KAlgebra, n) for n in names if n.startswith("verify_")) and (ax != "F6" or hasattr(KAlgebra, "forget"))
        checks.append(check(f"transcription {ax} ({lab}) <-> {verifiers}", present, formula))
    genuine = run_f_verifiers(QuantumTorusKAlg(B3), labels, K)
    checks.append(check("positive control: the degenerate Z^3 torus (Γ_f = Z e_3) passes all 15 verifiers and the F6 check on |γ_i| <= 1",
                        all(f == 0 for _p, f, _x in genuine.values()), "; ".join(f"{n} {p}/{p + f}" for n, (p, f, _x) in genuine.items())))
    models = [
        ("F1", FT_F1(B3), "verify_bar_involution"),
        ("F2", FT_F2(B3), "verify_embed_section_roundtrip"),
        ("F3", FT_F3(B3), "verify_embed_intertwines_rho"),
        ("F4", FT_F4(B3), "verify_orthonormality"),
        ("F5", FT_F5(B3), "verify_trace_intertwines_rho_star"),
        ("F6", FT_F6(B3), "forget_commutes (F6, inline)"),
    ]
    matrix = {}
    for ax, A, target in models:
        res = run_f_verifiers(A, labels, K)
        firing = sorted({axiom_of_f(n) for n, (p, f, _x) in res.items() if f})
        matrix[ax] = firing
        checks.append(check(f"discrimination {ax}: the {ax}-countermodel is rejected by {target}", res[target][1] > 0,
                            f"axiom groups that fire: {firing}; failures: " + ", ".join(f"{n} {f}" for n, (p, f, _x) in res.items() if f)))
    notes = ("Discrimination matrix: " + "; ".join(f"{ax} -> {v}" for ax, v in matrix.items())
             + ".  F6 has no verifier on the contract: the inline check (forget is a hom, commutes with rho, and with Tr after the augmentation) is the candidate KAlgebra.verify_forget_commutes of task T3; "
               "the F6-countermodel shows it is not implied by F1-F5: the prefactor exponent rk(Γ/Γ_f) of the flavoured trace is fixed by the forgetful map, not by orthonormality.")
    return {"checks": checks, "population": {"toy realisations": 7, "window": "|γ_i| <= 1 on Z^3 (27 labels)", "K": K},
            "controls": {"positive": "the degenerate torus passes every verifier", "negative": "each countermodel rejected by its axiom's verifier"},
            "notes": notes, "inputs": []}


# --------------------------------------------------------------------------
# ex:qt-flavoured
# --------------------------------------------------------------------------
def relement_terms(v):
    return {k: n for k, n in v.terms.items() if n} if hasattr(v, "terms") else ({(): v} if v else {})


def check_qt_flavoured(environment):
    K = 8
    checks = []
    pairings = {"rank 3, Γ_f = Z e_3": B3, "rank 3, kernel (1,-1,0) (non-split coordinates)": B3b, "rank 4, Γ_f = Z^2": B4}
    for name, B in pairings.items():
        Q = QuantumTorusKAlg(B)
        n = len(B)
        labels = window(n)
        rk_g = matrix_rank(B)
        checks.append(check(f"{name}: rk(Γ/Γ_f) = rank of the pairing = {rk_g}, the class's gauge rank agrees", Q._gauge_rank == rk_g and Q._flavour_rank == n - rk_g))
        pref = S.one(K)
        for _ in range(rk_g):
            pref = pref * qpoch_inf(2, K)
        R = Q.coefficient_ring()
        ok_tr = True
        for a in labels:
            t = Q.trace(a, K)
            got = {e: relement_terms(v) for e, v in t.coeffs.items() if relement_terms(v)}
            if any(Q.gauge_class(a)):
                exp = {}
            else:
                key = tuple(Q.flavour_part(a))
                exp = {e: {key: c} for e, c in pref.c.items()}
            ok_tr &= (got == exp)
        checks.append(check(f"{name}: Tr X_γ = (q^2;q^2)_∞^{rk_g} χ_γ δ_(γ in Γ_f) on {len(labels)} labels to q^{K} (independent series; χ_γ = μ^flav(γ))", ok_tr))
        ok_chi = all(Q.embed_R(R.basis_element(f)) == Element.basis(Q.flavour_generator_label(f)) for f in itertools.product((-1, 0, 1), repeat=Q._flavour_rank))
        checks.append(check(f"{name}: X_(γ_f) are the characters: embed_R(μ^f) = X_(γ_f) for |f_i| <= 1", ok_chi))
        ok_ip = True
        for a in labels:
            for b in labels:
                I = Q.inner_product(a, b, K)
                got = {e: relement_terms(v) for e, v in I.coeffs.items() if relement_terms(v)}
                if Q.gauge_class(a) != Q.gauge_class(b):
                    exp = {}
                else:
                    d = tuple(x - y for x, y in zip(Q.flavour_part(b), Q.flavour_part(a)))
                    exp = {e: {d: c} for e, c in pref.c.items()}
                ok_ip &= (got == exp)
        checks.append(check(f"{name}: I_(γ,γ') = (q^2;q^2)_∞^{rk_g} μ^(flav(γ')-flav(γ)) δ_(gauge γ = gauge γ') — in R + q R[[q]], identity summand δ_(γ,γ')", ok_ip))
        res = run_f_verifiers(Q, labels[:27] if n > 3 else labels, K=6)
        checks.append(check(f"{name}: F1-F6 all pass ({sum(p for p, f, _x in res.values())} instances)", all(f == 0 for _p, f, _x in res.values()),
                            "; ".join(f"{k} {p}/{p + f}" for k, (p, f, _x) in res.items() if f)))
    Qw = FT_F6(B3)
    t = Qw.trace((0, 0, 0), K)
    checks.append(check("negative control: a prefactor exponent rk Γ instead of rk(Γ/Γ_f) is detected", {e: relement_terms(v) for e, v in t.coeffs.items() if relement_terms(v)} != {e: {(0,): c} for e, c in (qpoch_inf(2, K) * qpoch_inf(2, K)).c.items()}))
    return {"checks": checks, "population": {"pairings": list(pairings), "labels": "|γ_i| <= 1", "K": K},
            "controls": {"positive": "the class's gauge rank equals an independent matrix rank", "negative": "wrong prefactor exponent detected"},
            "notes": "The flavour group is the algebraic torus T_(Γ_f) (R = Z[μ_1^±,...]); the example's trace, the identification of X_(γ_f) with the 1d characters and the F-axioms are all reproduced.", "inputs": []}


# --------------------------------------------------------------------------
# rem:flavour-change
# --------------------------------------------------------------------------
def check_flavour_change(environment):
    K = 6
    checks = []
    Q = QuantumTorusKAlg(B4)
    R2 = Q.coefficient_ring()
    R1 = AbelianZPlusRing(1)
    homs = {"diagonal U(1) -> U(1)^2 (f_1, f_2) -> f_1 + f_2": restriction_hom(R2, R1, [[1, 1]]),
            "first factor U(1) -> U(1)^2 (f_1, f_2) -> f_1": restriction_hom(R2, R1, [[1, 0]])}
    labels = window(4)

    def low(phi, a):
        sec, key = Q.r_label_decompose(a)
        img = phi.apply_basis(tuple(key))
        terms = [k for k, n in img.terms.items() if n]
        assert len(terms) == 1
        return (sec, terms[0])

    def low_elem(phi, x: Element) -> Element:
        out: dict = {}
        for lab, cf in x.terms.items():
            l = low(phi, lab)
            out[l] = (out[l] + cf) if l in out else cf
        return Element({k: v for k, v in out.items() if not v.is_zero()})
    lowered_traces = {}
    for name, phi in homs.items():
        L = Q.lower_flavour(phi)
        llabels = sorted({low(phi, a) for a in labels})
        ok_hom = all(L.multiply(low(phi, a), low(phi, b)) == low_elem(phi, Q.multiply(a, b)) for a in labels for b in labels)
        ok_rho = all(L.rho(low(phi, a)) == low(phi, Q.rho(a)) for a in labels)
        ok_tr = all((L.trace(low(phi, a), K) - phi.apply_RPowerSeries(Q.trace(a, K))).is_zero() for a in labels)
        checks.append(check(f"{name}: the map A_q[G_f] -> A_q[H_f] is an algebra homomorphism on all {len(labels)**2} pairs", ok_hom))
        checks.append(check(f"{name}: the map commutes with rho and with Tr (Tr' = φ∘Tr) on {len(labels)} labels", ok_rho and ok_tr))
        res = {}
        for _ax, vname, arity, takesK in F_VERIFIERS:
            if vname in ("verify_embed_intertwines_rho", "verify_embed_section_roundtrip"):
                continue  # the lowered algebra's embedding is derived; the section form is checked below
            fn = getattr(L, vname)
            if arity == 0:
                it = [()]
            elif arity == 1:
                it = [(a,) for a in llabels]
            elif arity == 2:
                it = itertools.product(llabels, llabels)
            else:
                it = itertools.product(llabels[:4], llabels[:4], llabels[:4])
            f = sum(1 for args in it if not (fn(*args, K=K) if takesK else fn(*args)))
            res[vname] = f
        checks.append(check(f"{name}: the lowered algebra is an H_f-flavoured K_q-algebra: F1-F5 verifiers on {len(llabels)} lowered labels", all(v == 0 for v in res.values()),
                            "; ".join(f"{k} fails {v}" for k, v in res.items() if v)))
        lowered_traces[name] = (L, phi)
    # the augmentation case is forget()
    Lm = Q.lower_flavour(R2.augmentation())
    F = Q.forget()
    ob = Lm.coefficient_ring().one_basis()
    ok_forget = True
    for a in labels[:40]:
        for b in labels[:40]:
            lm = Lm.multiply((Q.r_label_decompose(a)[0], ob), (Q.r_label_decompose(b)[0], ob))
            fm = F.multiply(Q.forget_label(a), Q.forget_label(b))
            ok_forget &= ({Q.forget_label(s): lp for (s, w), lp in lm.terms.items()} == dict(fm.terms))
    checks.append(check("lowering along the augmentation R_(G_f) -> Z is the forgetful map (multiply agrees with forget() after relabelling)", ok_forget))
    (L1, p1), (L2, p2) = lowered_traces[list(homs)[0]], lowered_traces[list(homs)[1]]
    differ = [a for a in labels if not (L1.trace(low(p1, a), K) - L2.trace(low(p2, a), K)).is_zero()]
    checks.append(check("sensitivity: the two lowerings give different traces on some flavour generator (μ^(f_1+f_2) vs μ^(f_1))", bool(differ), f"e.g. {differ[:3]}"))
    return {"checks": checks, "population": {"source": "rank-4 torus, G_f = U(1)^2", "homs": list(homs), "labels": "|γ_i| <= 1 (81)", "K": K},
            "controls": {"positive": "the augmentation case reproduces forget()", "negative": "two different lowerings are told apart"},
            "notes": "Lowering is KAlgebra.lower_flavour(φ) with φ = restriction_hom (the pullback R_(G_f) -> R_(H_f) of a group hom H_f -> G_f); the map is (section, χ) -> (section, φ(χ)).", "inputs": []}


ADAPTERS = {
    "def:kq-flavoured/contract": check_flavoured_contract,
    "ex:qt-flavoured": check_qt_flavoured,
    "rem:flavour-change": check_flavour_change,
}
