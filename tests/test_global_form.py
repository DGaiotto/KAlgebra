"""The type-A global-form charge infrastructure (U(N)/SU(N)/PSU(N)) —
labeling + charge lattices + S-duality + the U(N) bridge.

Pins the *labeling* layer (rigorous: lattices, center charges, minimal
lines, S-duality, integer representatives).  Does not assert that a U(N)
chart equals the genuine PSU(N) canonical — that is a build relationship,
verified per label elsewhere.

Run:  `python3 run_tests.py`
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from global_form import (
    u_n_form, su_n_form, psu_n_form, s_dual)


def test_charge_lattices():
    for N in (2, 3, 4, 5):
        U, S, P = u_n_form(N), su_n_form(N), psu_n_form(N)
        # SU: magnetic traceless (coroot); electric full
        assert S.mag_valid((1,) + (0,) * (N - 2) + (-1,))       # adjoint ok
        assert not S.mag_valid((1,) + (0,) * (N - 1))           # minuscule NOT a coroot
        assert S.elec_valid((1,) + (0,) * (N - 1))              # fundamental Wilson ok
        # PSU: magnetic full mod diag; electric traceless (root)
        assert P.mag_valid((1,) + (0,) * (N - 1))               # minuscule monopole ok
        assert P.elec_valid((1,) + (0,) * (N - 2) + (-1,))      # adjoint Wilson ok
        assert not P.elec_valid((1,) + (0,) * (N - 1))          # fundamental Wilson NOT a root
        # U: both full
        assert U.mag_valid((1,) + (0,) * (N - 1))
        assert U.elec_valid((1,) + (0,) * (N - 1))
    print("  PASS: test_charge_lattices")


def test_center_charges_and_minimal_lines():
    for N in (2, 3, 4, 5):
        S, P = su_n_form(N), psu_n_form(N)
        # SU minimal 't Hooft = adjoint (center-0 magnetic); PSU = minuscule (center-1)
        assert S.minimal_thooft() == (1,) + (0,) * (N - 2) + (-1,)
        assert P.minimal_thooft() == (1,) + (0,) * (N - 1)
        assert S.mag_center(S.minimal_thooft()) == 0            # SU: z_m = 0
        assert P.mag_center(P.minimal_thooft()) == 1            # PSU: z_m = 1 (minuscule)
        # minimal Wilson: SU fundamental (center-1), PSU adjoint (center-0)
        assert S.minimal_wilson() == (1,) + (0,) * (N - 1)
        assert P.minimal_wilson() == (1,) + (0,) * (N - 2) + (-1,)
        assert P.elec_center(P.minimal_wilson()) == 0           # PSU: z_e = 0
    print("  PASS: test_center_charges_and_minimal_lines")


def test_s_duality():
    for N in (2, 3, 4, 5):
        S, P = su_n_form(N), psu_n_form(N)
        assert s_dual(S).name == P.name and s_dual(P).name == S.name
        # S exchanges SU minimal Wilson (fundamental) with PSU minimal 't Hooft (minuscule)
        assert S.minimal_wilson() == P.minimal_thooft()
        # and SU minimal 't Hooft (adjoint) with PSU minimal Wilson (adjoint)
        assert S.minimal_thooft() == P.minimal_wilson()
    print("  PASS: test_s_duality")


def test_un_bridge():
    for N in (2, 3, 4, 5):
        P = psu_n_form(N)
        # the PSU minuscule monopole's U(N) representative is the minuscule (1,0,…,0)
        m0, e0 = P.un_representative(P.minimal_thooft(), (0,) * N)
        assert m0 == (1,) + (0,) * (N - 1), f"{m0}"
        assert e0 == (0,) * N
        # mod-diag reps land the trace in [0, N)
        for shift in (0, 1, -2, 3):
            m = tuple(x + shift for x in P.minimal_thooft())
            r = P.mag_rep(m)
            assert 0 <= sum(r) < N, f"{m} -> {r}"
            assert P.mag_center(r) == P.mag_center(m)           # center preserved
    print("  PASS: test_un_bridge")



def test_psu_no_fractional_q():
    """No fractional powers of 𝖖: PSU's fractional coweight pairs
    integrally against the root-lattice electric (Σe=0), and the built chart
    is integer-𝖖 throughout."""
    from pure_g_abe_kalgebra import PureGAbeKAlgebra   # (the N=2* route stood here until 2026-09-19)
    from root_datum import u_n
    for N in (3, 4, 5):
        P = psu_n_form(N)
        m = P.minimal_thooft()                            # (1,0,…,0)
        roots = [tuple((1 if i == a else 0) - (1 if i == b else 0)
                       for i in range(N))
                 for a in range(N) for b in range(N) if a != b]
        for e in roots:
            assert P.elec_valid(e)                        # root lattice
            pairing = sum(mi * ei for mi, ei in zip(m, e))
            assert isinstance(pairing, int)               # integer Dirac pairing
    # the built chart has only integer q-powers (it is over Z[q^±])
    A3 = PureGAbeKAlgebra(u_n(3))
    ch = A3.chart(A3.fold((1, 0, 0), (0, 0, 0)))
    for tr in ch.residuals().values():
        for lp in tr.simplify()._num._t.values():
            assert all(isinstance(e, int) for e in lp._coeffs)
    print("  PASS: test_psu_no_fractional_q")


if __name__ == "__main__":
    test_charge_lattices()
    test_center_charges_and_minimal_lines()
    test_s_duality()
    test_un_bridge()
    test_psu_no_fractional_q()
    print("All global-form tests passed.")
