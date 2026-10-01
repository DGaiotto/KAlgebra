"""The real case of the draft's flavour enhancement (the design record, sec:coulomb/flavour-enhancement; 2026-09-26): N=2* —
one adjoint hypermultiplet of SU(2), a real representation — carries a manifest U(1) flavour that enhances to
USp(2) = SU(2)_F.

A U(1) character lifts to USp(2) iff its weight diagram is integral and invariant under w ↦ −w.  The twist is fixed by
ρ: it shifts the flavour charge by κ_f(m) = μ^{-2m} (su_2 units), so the section at magnetic charge m is μ^m times a
USp(2) section, and a trace is twisted by μ^{-m}.

1. the flavour currents: Tr(1) at q² is the USp(2) adjoint μ^-2 + 1 + μ², at both global forms;
2. the twisted traces of the SU(2) form's lines with m ≤ 2 are USp(2) characters at every order through q⁴, and the
   untwisted traces with m ≠ 0 are not (the twist is needed, not decorative);
3. at the SO(3) form, whose lattice admits m = 1/2, the pairings of the m = 1/2 sections are USp(2) characters.

Run from the repo root:  `python3 run_tests.py`
"""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "implementations"))

from root_datum import su_2

K = 4


def _n2star(form):
    from gn_abe_kalgebra import GNAbeKAlgebra
    from g_matter_roster import highest_root
    from global_form import adjoint_lines
    kw = {"lines": adjoint_lines(su_2())} if form == "SO(3)" else {}
    return GNAbeKAlgebra(su_2(), highest_root(su_2()), nf=1, **kw)


def _twisted(R, c, shift):
    return {(Fraction(w[0]) + shift,): v for w, v in R.to_abelian(c).terms.items() if v}


def _usp2(poly):
    return all(w[0].denominator == 1 for w in poly) and {(-w[0],): v for w, v in poly.items()} == poly


def test_flavour_currents_are_the_usp2_adjoint():
    adj = {(Fraction(-2),): 1, (Fraction(0),): 1, (Fraction(2),): 1}
    for form in ("SU(2)", "SO(3)"):
        A = _n2star(form)
        tr = A.trace(A.identity(), K=2)
        assert _twisted(A.coefficient_ring(), tr.coeffs[2], 0) == adj, (form, tr.coeffs[2])


def test_twisted_traces_are_usp2_and_the_twist_is_needed():
    A = _n2star("SU(2)")
    R = A.coefficient_ring()
    bare_fail = 0
    for m in (0, 1, 2):
        for e in (0, 1):
            s = (A.fold((m,), (e,))[0], (0,))
            tr = A.trace(s, K=K)
            for k, c in tr.coeffs.items():
                if not c.terms:
                    continue
                assert _usp2(_twisted(R, c, -m)), (m, e, k)
                if m:
                    bare_fail += not _usp2(_twisted(R, c, 0))
    assert bare_fail > 0


def test_so3_form_half_integral_pairings():
    A = _n2star("SO(3)")
    R = A.coefficient_ring()
    H = Fraction(1, 2)
    secs = [(A.fold((H,), (e,))[0], (0,)) for e in (0, 2)]
    orders = 0
    for a in secs:
        for b in secs:
            I = A.inner_product(a, b, K=K)
            for k, c in I.coeffs.items():
                if c.terms:
                    orders += 1
                    assert _usp2(_twisted(R, c, 0)), (a, b, k)
    assert orders > 0


if __name__ == "__main__":
    test_flavour_currents_are_the_usp2_adjoint(); print("flavour currents = USp(2) adjoint: OK")
    test_twisted_traces_are_usp2_and_the_twist_is_needed(); print("twisted traces are USp(2) characters: OK")
    test_so3_form_half_integral_pairings(); print("SO(3) form, m = 1/2 pairings: OK")
    print("\nAll N=2* USp(2) tests passed.")
