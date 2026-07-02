"""Shared spine-freeness helper for the Step-3 (RG-flow) self-tests.

"Spine" = the BPS realisation-engine layer (`src/bps/`): F-solve, Schur
accumulators, spectrum generators, chart graphs, atlases.  The Step-3
tests certify that the RG engine and its auxiliaries run **without** any
of it imported, so a green run is a proof of spine-freeness.

The module list is **derived from the filesystem** (`src/bps/*.py`), so
it cannot go stale when modules are added, renamed, or retired — the
failure mode of the previous per-file hand-copied tuples, which named
five modules that no longer existed and omitted several that did.

Three `src/bps/` modules are excluded as **shared engine-free
primitives**: `spec_sigma` (the σ / half-monodromy lattice maps),
`mutation` and `lattice_mutation` (mutation arithmetic).  They are pure
lattice combinatorics with no F-solve / Schur / spectrum machinery, and
the cone layer imports them lazily (e.g. the A1A2k trace path uses
`spec_sigma`), so their presence does not indicate an engine leak.
"""
import os
import sys

_BPS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "bps",
)

#: `src/bps/` modules that are engine-free lattice combinatorics shared
#: with other layers (see module docstring).
SHARED_PRIMITIVES = frozenset({"spec_sigma", "mutation", "lattice_mutation"})

#: The realisation-spine module names (filesystem-derived; never stale).
SPINE = tuple(sorted(
    f[:-3] for f in os.listdir(_BPS_DIR)
    if f.endswith(".py") and f[:-3] not in SHARED_PRIMITIVES
))


def spine_hits():
    """The spine modules currently imported (empty list = spine-free)."""
    return sorted(
        m for m in sys.modules
        if any(m == s or m.startswith(s + ".") for s in SPINE)
    )


def assert_spine_free(context: str = ""):
    """Assert no spine module has been imported in this process."""
    hit = spine_hits()
    assert not hit, (
        f"spine modules leaked{' into ' + context if context else ''}", hit,
    )
