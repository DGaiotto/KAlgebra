"""a1dodd_fork_quiver — the D_{2k+3} BPS fork-quiver data.

Vendored from the Cluster scaffold
(`experiments/a1dodd_general_k_scaffold.py`, verbatim functions): the
`SkeinAtlas.a1dodd(k)` factory needs only the quiver constructor and
the elementary seed charges.
"""


def build_dodd_bps_quiver(k):
    """Return (pairing, node_charges) for the D_{2k+3} BPS fork quiver."""
    N = 2 * k + 3
    node_charges = []
    for j in range(2 * k + 1):                  # e_1 .. e_{2k+1}
        c = [0] * N
        c[j] = 1
        node_charges.append(tuple(c))
    g, s = 2 * k + 1, 2 * k + 2
    cp = [0] * N
    cp[g] = 1
    cp[s] = 1
    node_charges.append(tuple(cp))
    cm = [0] * N
    cm[g] = 1
    cm[s] = -1
    node_charges.append(tuple(cm))
    P = [[0] * N for _ in range(N)]
    for i in range(2 * k + 1):                   # A_{2k+2} chain
        P[i][i + 1] = 1
        P[i + 1][i] = -1
    return P, node_charges


def seed_charge(k, a, p):
    """Charge representative for the elementary seed (level a, parity p)."""
    N = 2 * k + 3
    d = [0] * N
    if p == 0:
        for j in range(1, a + 1):
            d[2 * j - 2] = 1
    else:
        for j in range(a):
            d[2 * k + 1 - 2 * j] = 1
    return tuple(d)
