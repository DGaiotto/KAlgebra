"""Generate ClusterApplet share-URLs for BPS quivers / `BPSKAlgebra`s.

ClusterApplet (https://berserkdvd.github.io/ClusterApplet/) loads a quiver
from a URL fragment: ``<base>#<URL-encoded JSON>`` where the JSON has the
schema (source of truth: ``coordination/to_clusterapplet/presets.jsx``)::

    {name, n, positions, frozen, B, charges?, spec?}

* ``name``      — display label.
* ``n``         — number of nodes.
* ``positions`` — per-node ``[x, y]`` pixel coords (drawing only).
* ``frozen``    — per-node bool (frozen = non-mutable / flavour node).
* ``B``         — antisymmetric exchange matrix (= the Dirac pairing /
                  BPS-quiver adjacency; ``B[i][j]`` = #arrows ``i -> j``).
* ``charges``   — (optional) per-node charge vectors in ``Gamma``.
* ``spec``      — (optional) ``{seq, charges, method}`` BPS spectrum data:
                  ``charges`` = the ordered BPS states feeding
                  ``S = prod E_q(X_gamma)``; ``seq`` = 0-based node order.

This module turns any ``BPSKAlgebra`` (or a bare pairing) into that URL.
The encoding matches the working preset URLs exactly: ``json.dumps`` with
the default ``", "`` / ``": "`` separators, then percent-encoded with
``urllib.parse.quote(safe="")``.

Quick use::

    from clusterapplet_url import bpskalgebra_applet_url
    from bps_kalgebra import BPSKAlgebra
    A = BPSKAlgebra(pairing=[[0,1],[-1,0]], node_charges=[(1,0),(0,1)])
    print(bpskalgebra_applet_url(A, name="A_2 pentagon"))
"""
from __future__ import annotations

import json
import math
import urllib.parse
from typing import Sequence

APPLET_BASE = "https://berserkdvd.github.io/ClusterApplet/"


def _auto_positions(n: int, *, cx: int = 450, cy: int = 375,
                    radius: int = 220) -> list[list[int]]:
    """A circular layout of `n` nodes (node 0 at top, clockwise).  Generic
    fallback when the caller does not pass an explicit `positions`."""
    if n == 1:
        return [[cx, cy]]
    out = []
    for i in range(n):
        theta = 2.0 * math.pi * i / n - math.pi / 2.0
        out.append([round(cx + radius * math.cos(theta)),
                    round(cy + radius * math.sin(theta))])
    return out


def _as_int_matrix(B) -> list[list[int]]:
    return [[int(x) for x in row] for row in B]


def applet_url(
    B,
    *,
    name: str = "quiver",
    positions: Sequence[Sequence[int]] | None = None,
    frozen: Sequence[bool] | None = None,
    charges: Sequence[Sequence[int]] | None = None,
    spec: dict | None = None,
    base: str = APPLET_BASE,
) -> str:
    """Build a ClusterApplet share-URL from an exchange matrix `B`.

    `B` is the antisymmetric quiver matrix (the only required field).
    `positions` defaults to a circular layout; `frozen` defaults to all
    `False`.  `charges` and `spec` are included only if given.  Returns the
    full ``<base>#<encoded JSON>`` string.
    """
    B = _as_int_matrix(B)
    n = len(B)
    if any(len(row) != n for row in B):
        raise ValueError(f"B must be square; got rows of lengths "
                         f"{[len(r) for r in B]}")
    if positions is None:
        positions = _auto_positions(n)
    positions = [[int(x), int(y)] for x, y in positions]
    if len(positions) != n:
        raise ValueError(f"positions has {len(positions)} entries, expected {n}")
    if frozen is None:
        frozen = [False] * n
    frozen = [bool(f) for f in frozen]
    if len(frozen) != n:
        raise ValueError(f"frozen has {len(frozen)} entries, expected {n}")

    obj: dict = {"name": name, "n": n, "positions": positions,
                 "frozen": frozen, "B": B}
    if charges is not None:
        charges = [[int(x) for x in c] for c in charges]
        # ClusterApplet requires an n x n charges block (else it throws on load).
        if len(charges) != n or any(len(c) != n for c in charges):
            raise ValueError(
                f"charges must be n x n (n={n}); got shape "
                f"{len(charges)}x{[len(c) for c in charges]}")
        obj["charges"] = charges
    if spec is not None:
        # ClusterApplet REQUIRES spec.seq (array of ints) and THROWS without it
        # (the whole preset then silently fails to load).  Validate up front.
        seq = spec.get("seq")
        if not isinstance(seq, (list, tuple)) or not all(
                isinstance(v, int) for v in seq):
            raise ValueError(
                "spec must be {seq:[ints], charges?:[[...]], method?:str}; "
                "a spec without an integer `seq` makes ClusterApplet fail to load")
        obj["spec"] = {
            "seq": [int(v) for v in seq],
            "charges": [[int(x) for x in c] for c in spec.get("charges", [])],
            "method": str(spec.get("method", "imported")),
        }

    payload = json.dumps(obj)               # default ", " / ": " separators
    return base + "#" + urllib.parse.quote(payload, safe="")


def bpskalgebra_applet_url(
    A,
    *,
    name: str | None = None,
    positions: Sequence[Sequence[int]] | None = None,
    frozen: Sequence[bool] | None = None,
    include_charges: bool = True,
    spec_seq: Sequence[int] | None = None,
    base: str = APPLET_BASE,
) -> str:
    """ClusterApplet share-URL for a `BPSKAlgebra` (or anything exposing
    ``lattice.pairing`` + ``node_charges`` [+ ``spec``]).

    Uses the Dirac pairing as ``B`` and the node charges as ``charges``
    (unless ``include_charges=False`` or they are the identity basis).

    **The BPS spectrum** is attached only if you pass ``spec_seq`` — the
    0-based node-mutation order (e.g. ``[0, 2, 1, 0, 3]`` for SU2A1D3).
    ClusterApplet *requires* this integer ``seq`` and silently fails to load a
    ``spec`` that lacks it, so without ``spec_seq`` the URL is the bare quiver
    (which always renders, and from which the applet can re-run its own
    spectrum-generator search).  When given, ``A.spec`` supplies ``spec.charges``.

    `positions`/`frozen`/`name` may be overridden; otherwise a circular layout,
    all-unfrozen, and the class name are used.
    """
    B = [list(row) for row in A.lattice.pairing]
    n = len(B)
    if name is None:
        name = type(A).__name__

    charges = None
    if include_charges:
        nc = [list(c) for c in getattr(A, "node_charges", []) or []]
        identity = [[1 if i == j else 0 for j in range(len(nc[0]))]
                    for i in range(len(nc))] if nc else []
        # include only if non-trivial (a non-identity embedding carries info)
        if nc and nc != identity:
            charges = nc

    spec = None
    if spec_seq is not None:
        spec = {"seq": [int(v) for v in spec_seq],
                "charges": [list(g) for g in getattr(A, "spec", []) or []],
                "method": "negating_sequence"}

    return applet_url(B, name=name, positions=positions, frozen=frozen,
                      charges=charges, spec=spec, base=base)


if __name__ == "__main__":
    import os
    import sys
    _SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _r, _ds, _ in os.walk(_SRC):
        _ds[:] = [d for d in _ds if d != "__pycache__"]
        if _r not in sys.path:
            sys.path.insert(0, _r)
    from bps_kalgebra import BPSKAlgebra

    # Demo: the pentagon K_q([A_1, A_2]) from its BPS quiver.
    A = BPSKAlgebra(pairing=[[0, 1], [-1, 0]],
                    node_charges=[(1, 0), (0, 1)],
                    spec=[(1, 0), (0, 1)])
    pos = [[300, 300], [500, 300]]
    print("# bare quiver (always renders):")
    print(bpskalgebra_applet_url(A, name="pentagon [A1,A2]", positions=pos))
    print("\n# with the BPS spectrum:")
    print(bpskalgebra_applet_url(A, name="pentagon [A1,A2]", positions=pos,
                                 spec_seq=[0, 1]))
