"""SL(2,ℤ) / mapping-class action on the SU(2)+N_f=4 canonical basis
(a, b, c form the canonical basis, and SL(2,ℤ) permutes it).

On the tetrahedral S²₀,₄ the six edges are the six puncture pairs
`01,02,03,12,13,23` (`triangulation.edges`), and the three canonical generators
are the three puncture **pairings** (separating curves):

    a = {0,1}|{2,3} = (0,1,1,1,1,0)        (0 on the two internal edges 01,23)
    b = {0,2}|{1,3} = (1,0,1,1,0,1)
    c = {0,3}|{1,2} = (1,1,0,0,1,1)

The mapping-class group acts; its finite part is the **puncture permutation
group S₄**, which permutes the three pairings exactly as **S₃ on {a,b,c}** — the
channel/triality action (= triality on Spin(8): the 3-cycle of channels is the
8v→8s→8c 3-cycle, `spin8_characters.triality`).  A puncture permutation σ maps
the edge `{p,q}` to `{σp,σq}`, inducing a coordinate permutation
(`act_on_curve`), so it transports any canonical curve to another — the basis
permutation.  The structure constants are S₃-covariant: every generator
cross-product follows from `a·b` by a channel permutation (the SL(2,ℤ) bootstrap).

The **infinite** part (Dehn twists / the Farey/Markov tree of slopes) generates
the higher rays — the twisted partners `ã,b̃,c̃` that appear in `a·b` are the
first Dehn-twist images; the SL(2,ℤ) orbit of the three pairings is all slopes.
"""
from __future__ import annotations

from itertools import permutations as _perms

# edge id -> puncture pair (tetrahedron_S2_4)
EDGE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_PAIR_TO_EDGE = {frozenset(p): i for i, p in enumerate(EDGE_PAIRS)}

# the three canonical generators (channel = puncture pairing)
GEN = {
    "a": (0, 1, 1, 1, 1, 0),   # {0,1}|{2,3}
    "b": (1, 0, 1, 1, 0, 1),   # {0,2}|{1,3}
    "c": (1, 1, 0, 0, 1, 1),   # {0,3}|{1,2}
}
# channel -> the pairing it separates
CHANNEL_PAIRING = {
    "a": ((0, 1), (2, 3)),
    "b": ((0, 2), (1, 3)),
    "c": ((0, 3), (1, 2)),
}


def edge_perm(sigma) -> tuple:
    """Induced permutation of the 6 edge-coordinates from a puncture
    permutation σ (a 4-tuple, σ[p] = image of puncture p).

    Returns a 6-tuple `p` with `p[i]` = the edge that edge i maps TO, i.e. the
    new coordinate j reads the old coordinate `inv` such that …; here we return
    the *pullback* index: `act_on_curve` uses `new[i] = old[src[i]]`."""
    src = [0] * 6
    for i, (u, v) in enumerate(EDGE_PAIRS):
        # edge i = {u,v}; the curve coordinate at the NEW edge {σu,σv} should be
        # the old coordinate at {u,v}.  So new[ edge{σu,σv} ] = old[ i ].
        j = _PAIR_TO_EDGE[frozenset((sigma[u], sigma[v]))]
        src[j] = i
    return tuple(src)


def act_on_curve(sigma, coords) -> tuple:
    """Transport a multicurve's normal coordinates by the puncture permutation
    σ (∈ S₄).  `new[i] = old[src[i]]`."""
    src = edge_perm(sigma)
    return tuple(coords[src[i]] for i in range(6))


def channel_perm(sigma) -> dict:
    """How σ ∈ S₄ permutes the three channels {a,b,c} (the S₃ quotient)."""
    out = {}
    for nm, g in GEN.items():
        img = act_on_curve(sigma, g)
        match = next((m for m, gg in GEN.items() if gg == img), None)
        out[nm] = match
    return out


# the six S₄ elements that realise the full S₃ on channels (one lift each)
def s3_lifts() -> dict:
    """{channel-permutation (a,b,c image) : a representative σ ∈ S₄}."""
    out = {}
    for sigma in _perms(range(4)):
        cp = channel_perm(sigma)
        key = (cp["a"], cp["b"], cp["c"])
        if None in key:
            continue
        out.setdefault(key, sigma)
    return out


__all__ = [
    "EDGE_PAIRS", "GEN", "CHANNEL_PAIRING",
    "edge_perm", "act_on_curve", "channel_perm", "s3_lifts",
]


if __name__ == "__main__":
    print("=== puncture permutations permute the canonical generators a,b,c ===")
    tests = {
        (0, 2, 1, 3): "swap punctures 1,2",
        (1, 0, 2, 3): "swap punctures 0,1",
        (0, 1, 3, 2): "swap punctures 2,3",
        (1, 2, 0, 3): "3-cycle 0->1->2",
    }
    for sigma, desc in tests.items():
        cp = channel_perm(sigma)
        print(f"  σ={sigma} ({desc}): a→{cp['a']}  b→{cp['b']}  c→{cp['c']}")

    print("\n=== the S₃ channel group is fully realised (6 distinct perms) ===")
    lifts = s3_lifts()
    for key in sorted(lifts):
        print(f"  (a,b,c)→{key}  by σ={lifts[key]}")
    print(f"  #channel perms = {len(lifts)} (expect 6 = |S₃|)")

    # cross-check: a·b transported by the (a→b,b→a) swap should give b·a
    print("\n=== structure-constant covariance (transport a·b) ===")
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from su2_nf4_skein_chars import chi_mul, GA, GB
    ab = chi_mul(GA, 1, GB, 1)
    ba = chi_mul(GB, 1, GA, 1)
    # under the swap a<->b (σ swapping punctures to exchange channels a,b),
    # the curve labels in a·b should map to those in b·a
    swap = next(s for k, s in s3_lifts().items() if k == ("b", "a", "c"))

    def parity(sigma):
        seen, par = [False] * len(sigma), 0
        for i in range(len(sigma)):
            if seen[i]:
                continue
            j, ln = i, 0
            while not seen[j]:
                seen[j] = True
                j = sigma[j]
                ln += 1
            par += ln - 1
        return par % 2

    odd = parity(swap)        # transpositions are orientation-reversing → bar A

    def transport_label(lbl):
        if isinstance(lbl[0], str):              # ('P', periph) flavour·𝟙
            _tag, periph = lbl
            return ("P", tuple(periph[swap[p]] for p in range(4)))
        slope, k = lbl[0], lbl[1]
        return (act_on_curve(swap, slope), k)

    transported = {transport_label(lbl): (co.bar() if odd else co)
                   for lbl, co in ab.items()}
    match = (transported == ba)
    print(f"  a·b: {len(ab)} terms;  b·a: {len(ba)} terms;  swap parity={'odd' if odd else 'even'}")
    print(f"  swap-transport(a·b) == b·a : {match}")
    print("self-tests done")
