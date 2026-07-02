"""Per-factor Dynkin data for simply-laced ADE types.

Self-contained (rank, Coxeter number, Dynkin edges) lookup for
lattice/algebra computations over ADE Dynkin types.

Bourbaki numbering, 0-indexed.
"""

# Dynkin edges for E_n (Bourbaki, 0-indexed).
_E_EDGES: dict[int, list[tuple[int, int]]] = {
    6: [(0, 2), (2, 3), (3, 4), (4, 5), (1, 3)],
    7: [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 3)],
    8: [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 3)],
}
_E_COXETER: dict[int, int] = {6: 12, 7: 18, 8: 30}


def _factor_data(tp: str, n: int) -> tuple[int, int, list[tuple[int, int]]]:
    """Rank r, Coxeter number h, and 0-indexed Dynkin edges for (tp, n)."""
    if tp == "A":
        if n < 1:
            raise ValueError(f"A_n requires n >= 1, got n={n}")
        return n, n + 1, [(i, i + 1) for i in range(n - 1)]
    if tp == "D":
        if n < 4:
            raise ValueError(f"D_n requires n >= 4, got n={n}")
        return (
            n,
            2 * n - 2,
            [(i, i + 1) for i in range(n - 2)] + [(n - 3, n - 1)],
        )
    if tp == "E":
        if n not in _E_EDGES:
            raise ValueError(f"E_n requires n in {{6,7,8}}, got n={n}")
        return n, _E_COXETER[n], _E_EDGES[n]
    raise ValueError(f"unknown simple-factor type {tp!r}; expected A/D/E")
