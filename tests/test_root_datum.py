"""Tests for `root_datum.RootDatum` — the weight-lattice + root-system datum.

The `sun_characters` cross-check is a **𝖖-free type-A skeleton** agreement
(weight lattice + Weyl group + Weyl denominator), NOT a 𝖖→1 limit of any ring:
`sun_characters` has no 𝖖 to begin with.  See the module docstring discussion.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

from root_datum import RootDatum, u_n, su_n, su_2, _lpoly_mul, _det
import sun_characters as sc


def _clean(poly: dict) -> dict:
    return {tuple(e): c for e, c in poly.items() if c != 0}


def test_u_n_shapes():
    for N in (2, 3, 4):
        d = u_n(N)
        assert d.dim == N
        assert len(d.weyl) == math.factorial(N), f"U({N}) |W|"
        assert len(d.positive_roots()) == N * (N - 1) // 2, f"U({N}) |Φ⁺|"
        # roots are closed under the Weyl action
        R = set(d.roots())
        for a in R:
            for w in d.weyl_elements():
                assert d.act(w, a) in R, f"U({N}) root closure"
        # sign == det, identity present
        for w in d.weyl_elements():
            assert d.sign(w) == _det(d.weyl[w][0]) in (1, -1)
    print("  PASS: U(N) shapes / |W|=N! / Φ⁺ / root closure / sign=det")


def test_su_n_shapes():
    for N in (2, 3, 4):
        d = su_n(N)
        assert d.dim == N - 1
        assert len(d.weyl) == math.factorial(N), f"SU({N}) |W|"
        assert len(d.positive_roots()) == N * (N - 1) // 2, f"SU({N}) |Φ⁺|"
        R = set(d.roots())
        for a in R:
            for w in d.weyl_elements():
                assert d.act(w, a) in R, f"SU({N}) root closure"
    print("  PASS: SU(N) shapes / |W|=N! / Φ⁺ / root closure")


def test_su2_root_is_two():
    """The headline: SU(2)'s single positive root is (2) ⇒ v^α = v², which is
    exactly the old `_sq` (1 − 𝖖^k v²) factor, now uniform `1 − 𝖖^k v^α`."""
    d = su_2()
    assert d.dim == 1
    assert d.positive_roots() == [(2,)]
    assert set(d.roots()) == {(2,), (-2,)}
    # Weyl group {±1}; the reflection sends the weight λ → −λ
    assert d.weyl_orbit((1,)) == {(1,), (-1,)}
    assert d.is_dominant((1,)) and not d.is_dominant((-1,))
    assert d.dominant_rep((-1,)) == (1,)
    print("  PASS: SU(2) root = (2)  ⇒  v^α = v²  (subsumes _sq)")


def test_orient():
    d = u_n(3)
    a_pos = (1, -1, 0)
    assert d.orient(a_pos) == (a_pos, False)
    a_neg = (-1, 1, 0)
    assert d.orient(a_neg) == (a_pos, True)
    try:
        d.orient((1, 0, 0))  # not a root
        assert False, "expected ValueError on non-root"
    except ValueError:
        pass
    print("  PASS: orient (positive / flipped / non-root raises)")


def test_shift_pairing():
    d = u_n(3)
    # ⟨c, e_i - e_j⟩ = c_i - c_j
    assert d.shift_pairing((5, 2, 1), (1, -1, 0)) == 3
    assert d.shift_pairing((5, 2, 1), (0, 1, -1)) == 1
    # SU(2): ⟨c, (2)⟩ = 2c
    assert su_2().shift_pairing((3,), (2,)) == 6
    print("  PASS: shift_pairing = ⟨c, α⟩ (dot product)")


def test_weyl_denominator_identity():
    """The Weyl denominator identity `a_ρ = v^ρ·∏_{α>0}(1 − v^{−α})`, with the
    integer Weyl vector ρ per realization (e-basis staircase / ω-basis all-ones),
    plus anti-invariance of the bialternant `a_ρ`.  Note: ∏(1−v^{−α}) *alone* is
    NOT anti-invariant — only after the `v^ρ` shift."""
    cases = [(u_n(3), (2, 1, 0)), (u_n(4), (3, 2, 1, 0)),
             (su_n(3), (1, 1)), (su_2(), (1,))]
    for d, rho in cases:
        # bialternant a_ρ = Σ_w sgn(w)·v^{w·ρ}
        a: dict = {}
        for w in d.weyl_elements():
            e = d.act(w, rho)
            a[e] = a.get(e, 0) + d.sign(w)
        a = _clean(a)
        rhs = _clean(_lpoly_mul({tuple(rho): 1}, d.weyl_denominator()))
        assert a == rhs, f"{d.name}: a_ρ != v^ρ·∏(1−v^{{−α}})"
        # a_ρ is W-anti-invariant: w·a_ρ = sgn(w)·a_ρ
        for w in d.weyl_elements():
            wa = _clean({d.act(w, e): c for e, c in a.items()})
            assert wa == _clean({e: d.sign(w) * c for e, c in a.items()}), \
                f"{d.name}: a_ρ not anti-invariant"
    print("  PASS: Weyl denominator identity a_ρ = v^ρ·∏(1−v^{−α}) + anti-invariance")


def test_cocharacters():
    """The cocharacter side: action preserves the pairing ⟨w·m, w·λ⟩ = ⟨m,λ⟩;
    U(N) is self-dual (act_cochar == act); SU(N) genuinely differs."""
    # U(N): self-dual — cochar action equals weight action
    for N in (2, 3):
        d = u_n(N)
        for w in d.weyl_elements():
            assert d.weyl_cochar[w][0] == d.weyl[w][0], f"U({N}) not self-dual"
        # cochar dominance == weight dominance here
        for m in [(2, 1, 0)[:N], (0, 1)[:N] + (0,) * (N - 2)]:
            assert d.is_dominant_cochar(m) == d.is_dominant(m)
    # pairing preserved by (weight action, cochar action) for ALL datums
    for d in (u_n(3), su_n(3), su_2()):
        lam = (1,) * d.dim
        m = tuple(range(1, d.dim + 1))
        base = sum(m[i] * lam[i] for i in range(d.dim))
        for w in d.weyl_elements():
            wl, wm = d.act(w, lam), d.act_cochar(w, m)
            assert sum(wm[i] * wl[i] for i in range(d.dim)) == base, \
                f"{d.name}: pairing not preserved"
    # SU(3): cochar dominance is genuinely different from weight dominance
    d = su_n(3)
    # weight (1,0) is dominant (Dynkin labels ≥0); as a cocharacter, dominance is
    # ⟨α_i,m⟩≥0 = (Cartan·m)≥0, a different condition
    assert d.is_dominant((1, 0)) is True
    assert d.is_dominant_cochar((1, 0)) == all(
        sum(a[i] * (1, 0)[i] for i in range(2)) >= 0 for a in d.simple_roots)
    print("  PASS: cocharacter action (contragredient), pairing-preserving, U(N) self-dual")


def test_against_sun_characters():
    """𝖖-free type-A skeleton agreement with the established `sun_characters`:
    the Weyl orbits coincide, and δ_sun = v^ρ · ∏_{α>0}(1 − v^{−α})."""
    for N in (2, 3, 4):
        d = u_n(N)
        # Weyl orbits coincide (both = coordinate permutations)
        for lam in [(N - 1,) + (0,) * (N - 1), tuple(range(N))]:
            assert d.weyl_orbit(lam) == set(sc.weyl_orbit(N, lam)), \
                f"U({N}) orbit vs sun_characters"
        # δ_sun (bialternant, ρ=(N-1,…,0)) = v^ρ · ∏_{α>0}(1 − v^{−α})
        rho = tuple(range(N - 1, -1, -1))
        lifted = _clean(_lpoly_mul({rho: 1}, d.weyl_denominator()))
        assert lifted == _clean(sc.weyl_denominator(N)), \
            f"U({N}) Weyl denominator vs sun_characters"
    print("  PASS: 𝖖-free skeleton matches sun_characters (orbits + Weyl denom)")


def test_sun_rho_sign_consistent_with_un():
    """ρ-twist sign consistency between SU(N) and U(N) (the fix for the
    rank-≥2 bug): a pure-SU(N) monopole is a trace-zero U(N) monopole, so the
    ρ sign `(−1)^{rho_sign_exp}` must agree.  In the SU(N) ω-basis a magnetic
    coroot-cochar `k` lifts to the trace-zero u_n cochar `mₜ = c_{t+1}−cₜ`; the
    default coordinate staircase (`Σ(dim−1−2t)k_t`) is coordinate-dependent and
    gave the WRONG parity for rank ≥ 2 — `su_n` now overrides `rho_sign` to the
    U(N)-restricted (always-even) value.  Regression: parity matches U(N) on
    every trace-zero cochar."""
    import itertools

    def _coroot(m, N):
        return tuple(sum(m[:i + 1]) for i in range(N - 1))

    for N in (2, 3, 4, 5):
        un, sun = u_n(N), su_n(N)
        n = 0
        for m in itertools.product(range(-2, 3), repeat=N):
            if sum(m) != 0:
                continue
            n += 1
            assert sun.rho_sign_exp(_coroot(m, N)) % 2 == un.rho_sign_exp(m) % 2, \
                f"SU({N}) ρ-sign parity != U(N) at trace-zero cochar {m}"
        assert n > 0
    print("  PASS: SU(N) ρ-twist sign consistent with U(N) (rank-≥2 fix)")


def main():
    print("RootDatum tests")
    test_u_n_shapes()
    test_su_n_shapes()
    test_su2_root_is_two()
    test_orient()
    test_shift_pairing()
    test_weyl_denominator_identity()
    test_cocharacters()
    test_against_sun_characters()
    test_sun_rho_sign_consistent_with_un()
    print("All RootDatum tests passed.")


if __name__ == "__main__":
    main()
