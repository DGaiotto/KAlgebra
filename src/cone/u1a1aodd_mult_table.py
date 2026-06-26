"""
u1a1aodd_mult_table.py
======================

Ray-generator multiplication structure for the gauged `[A_1, A_{2k+1}]`
family, read off the validated oracle `U1A1AoddGaugedRG` (see
`U1A1AODD_GAUGED_PROGRESS.md`).

Verified structure: with the gauged algebra `≅ BPS` over the `A_{2k+2}`
linear chain,

  * **q-commuting ray pairs (most of the table): cocycle = the chain
    pairing of charges** — `c_fwd − c_bwd = 2·⟨γ_a,γ_b⟩`, `⟨·,·⟩` the
    `A_{2k+2}` chain pairing (= the geometric chord-crossing form).
    Holds on *every* q-commuting pair at any k.
  * **crossing pairs: 2-term Plücker** (cluster exchange), ρ-reduced to
    a few orbit reps.

ρ_UV-equivariance lifts cocycle + the Plücker reps to the full table.

`chain_pairing` + `bijection_charge` are closed-form (general k).
`k1_true_rays` are the hand-verified 9 rays at k=1 (3 long @ mag 0 +
6 short @ mag ±1), giving a clean ≤2-term table.

TODO (remaining for the closed-form `ConeKAlgebra`): the general-k
*true-ray* generator. Rays = single-term-`RG` seeds swept by ρ_UV, with
a **judicious `X_{(±1,0)}`** (gauge shift) on the even-length chords to
balance them; a naive scan over `c0` over-generates (picks up dressed /
cluster-monomial elements → spurious ≥3-term products). The clean
per-parity gauge shift is the open piece.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# NOTE: the RG-flow spine (`u1a1aodd_gauged_rg`) and `a1a2k_bps_iso` are imported
# LAZILY inside `select_chords` / `bijection_charge` (the construction-only
# extractors), so `chain_pairing` (the per-cocycle arithmetic) imports spine-free.


def bijection_charge(k):
    """Return `phi`: oracle-label → `B_GAUGED_k` charge (the verified iso
    map; `tests/test_u1a1aodd_gauged_rg.py`)."""
    from a1a2k_bps_iso import _compute_chord_charges   # lazy: spine-adjacent
    n = 2 * k + 2
    CC = {(a, i): tuple(v)
          for (a, i), v in _compute_chord_charges(k, {a: 0 for a in range(1, k + 1)}).items()}
    mu = tuple(1 if (j % 2 == 0 and j <= 2 * k) else 0 for j in range(n))
    vg = tuple(0 if j % 2 == 0 else -1 for j in range(n))

    def phi(lbl):
        chord, (c0, c1) = lbl
        cc = [0] * (2 * k)
        for (a, i, e) in chord:
            pc = CC[(a, i)]
            for t in range(2 * k):
                cc[t] += e * pc[t]
        out = [0] * n
        for t in range(2 * k):
            out[1 + t] += cc[t]
        for j in range(n):
            out[j] += c1 * mu[j] + c0 * vg[j]
        return tuple(out)

    return phi


def chain_pairing(g, h):
    """`A_n` linear-chain antisymmetric pairing `⟨g,h⟩ = Σ_i(g_i h_{i+1}
    − g_{i+1} h_i)` — the geometric chord-crossing / cocycle form."""
    return sum(g[i] * h[i + 1] - g[i + 1] * h[i] for i in range(len(g) - 1))


def _rho_orbit(T, seed, m):
    o = [seed]
    cur = seed
    for _ in range(m - 1):
        cur = T.rho(cur)
        o.append(cur)
    return o


def k1_true_rays(T):
    """The 9 hand-verified true rays at k=1: 6 short (ρ-orbit of seed
    `(((1,0,1),),(1,-1))`) + 3 long (ρ-orbit of `(((1,3,1),),(0,-1))`).
    They give a clean ≤2-term cluster/cone structure."""
    return _rho_orbit(T, (((1, 0, 1),), (1, -1)), 6) + \
           _rho_orbit(T, (((1, 3, 1),), (0, -1)), 3)


def _qpow(elt):
    (l, c), = elt.terms.items()
    ks = list(c._coeffs.keys())
    return ks[0] if len(ks) == 1 else None


def select_chords(k):
    """General-k chord (cluster-variable) selection from the oracle.

    The chords are the irreducible ρ_UV-orbits of the monomial-RG seeds.
    An orbit is a *cluster monomial* (not a chord) iff it factors as
    `chord · (already-marked monomial)` with matching charge (mod E=μ)
    and a single-term (q-commuting) product — checked greedily by
    charge-norm with charge-guided factoring (fast: the charge lookup
    bounds the candidate factorisations, so no semigroup-closure blowup).

    Returns `(T, phi, chords)` where `chords[(a, i)] = oracle element`:
    `a` = chord type (1-based ρ_UV-orbit index), `i` = ρ_UV position.
    Verified counts `(k+2)(2k+1)` at k=1,2,3 (k=1 = U1Hex's L1/L2 mod E).
    """
    from u1a1aodd_gauged_rg import U1A1AoddGaugedRG   # lazy: the RG-flow spine
    T = U1A1AoddGaugedRG(k)
    phi = bijection_charge(k)
    H = T._H
    n = 2 * k + 2
    mu = tuple(1 if (j % 2 == 0 and j <= 2 * k) else 0 for j in range(n))

    def modE(g):
        s = g[0]
        return tuple(g[j] - s * mu[j] for j in range(n))

    def norm(g):
        return sum(abs(v) for v in g)

    def orbit(x, cap=2 * (2 * k + 4) + 4):
        seen = {modE(phi(x))}
        out = [x]
        cur = x
        for _ in range(cap):
            cur = T.rho(cur)
            m = modE(phi(cur))
            if m in seen:
                break
            seen.add(m)
            out.append(cur)
        return out

    # monomial-RG seeds, grouped into ρ_UV-orbits (mod E), ordered by norm
    seeds = [(((a, i, 1),), (c0, c1))
             for a in range(1, k + 1) for i in range(H)
             for c0 in range(-2, 3) for c1 in range(-2, 3)
             if len(T.RG((((a, i, 1),), (c0, c1))).terms) == 1]
    orbits = []
    seen = set()
    for s in sorted(seeds, key=lambda x: norm(phi(x))):
        if modE(phi(s)) in seen:
            continue
        el = orbit(s)
        oset = frozenset(modE(phi(e)) for e in el)
        if oset & seen:
            continue
        seen |= oset
        orbits.append(el)
    orbits.sort(key=lambda el: norm(phi(el[0])))

    marked = {}                  # modE charge -> [elements] (chords + products)

    def add_marked(e):
        m = modE(phi(e))
        marked.setdefault(m, [])
        if e not in marked[m]:
            marked[m].append(e)

    chord_orbits = []
    chord_elts = []              # (full charge, element)
    for el in orbits:
        gO = phi(el[0])
        isprod = False
        for (gc, ce) in chord_elts:
            need = modE(tuple(gO[j] - gc[j] for j in range(n)))
            for me in marked.get(need, []):
                pr = T.multiply(ce, me)
                if len(pr.terms) == 1:
                    (l, _), = pr.terms.items()
                    if modE(phi(l)) == modE(gO):
                        isprod = True
                        break
            if isprod:
                break
        if isprod:
            for e in el:
                add_marked(e)
        else:
            chord_orbits.append(el)
            for e in el:
                chord_elts.append((phi(e), e))
                add_marked(e)

    chords = {(a, i): e
              for a, el in enumerate(chord_orbits, 1)
              for i, e in enumerate(el)}
    return T, phi, chords


def verify_cocycle_is_chain_pairing(T, phi, rays):
    """`c_fwd − c_bwd == 2·⟨γ_a,γ_b⟩` on every q-commuting ray pair.
    Returns `(n_ok, n_qcommute)`."""
    n_ok = n_qc = 0
    for a in rays:
        for b in rays:
            pab, pba = T.multiply(a, b), T.multiply(b, a)
            if len(pab.terms) == 1 and len(pba.terms) == 1:
                cf, cb = _qpow(pab), _qpow(pba)
                if cf is None or cb is None:
                    continue
                n_qc += 1
                if cf - cb == 2 * chain_pairing(phi(a), phi(b)):
                    n_ok += 1
    return n_ok, n_qc


def max_terms(T, rays):
    """Maximum #terms over all ordered ray products (≤2 ⟺ clean cone)."""
    return max(len(T.multiply(a, b).terms) for a in rays for b in rays)


if __name__ == "__main__":
    T1 = U1A1AoddGaugedRG(1)
    phi1 = bijection_charge(1)
    rays = k1_true_rays(T1)
    ok, qc = verify_cocycle_is_chain_pairing(T1, phi1, rays)
    print(f"k=1 true rays: {len(rays)}; cocycle==chain pairing {ok}/{qc}; "
          f"max product terms = {max_terms(T1, rays)}")
