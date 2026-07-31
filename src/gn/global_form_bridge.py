"""The **one** global-form choice, seen from both sides.

A `BPSKAlgebra` is not determined by a BPS quiver alone.  It is determined by a
quiver **plus a choice of how its node charges embed in a charge lattice** `Γ`,
and that choice *is* the 4d gauge group.  For a pure ADE theory the node charges
live in `Q^∨ ⊕ Q`, and the admissible `Γ` are the unimodular lattices

    Λ₀ = P^∨ ⊕ Q  ⊆  Γ  ⊆  Q^∨ ⊕ P,

classified by the Lagrangian subgroups of `(P/Q)²` — the simply connected form
`Γ = P^∨ ⊕ P`, the adjoint form `Γ = Q^∨ ⊕ Q`, and the intermediate forms in
between.  `pure_ade_lattice.pure_ade_lattice_data(factors, global_form=…)` owns
that classification.

The same choice is made a second time, in a different vocabulary, on the
abelianized side: `global_form.LineLattice(datum, H)` is the 4d gauge group as a
maximal set of mutually local Kapustin `(m, e)` labels, with `H` the subgroup of
the centre quotiented by.  Nothing in the repo related the two, so "the BPS chart
and the `AbeKAlgebra` describe the same theory" was a convention maintained by
hand.  This module is the missing map, and the invariants that make it checkable:

    bps_global_form(lines)                    LineLattice  →  "sc" / "adj"
    bps_lattice_for(lines, factors)           LineLattice  →  the Γ data
    verify_same_quiver_different_embedding()  the lesson, as an assertion
    verify_choice_matches(lines, …)           the two sides agree

`verify_same_quiver_different_embedding` is the one worth reading: it asserts
that `"sc"` and `"adj"` return the **same exchange matrix** (same quiver) and
**different node charges** (different embedding).  That is the whole content of
"a quiver plus a choice", written so a future change cannot quietly break it.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pure_ade_lattice import pure_ade_lattice_data

__all__ = [
    "bps_global_form",
    "bps_lattice_for",
    "exchange_matrix",
    "verify_same_quiver_different_embedding",
    "verify_choice_matches",
]


class IntermediateFormUnsupported(NotImplementedError):
    """Raised for a `LineLattice` whose `H` is a proper nontrivial subgroup.

    Such a form is a genuine Γ — it is one of the intermediate Lagrangians — but
    it is not `"sc"` or `"adj"`, so it must be passed to
    `pure_ade_lattice_data` as an explicit list of `(m_vec, e_vec)` pairs rather
    than a string.  Reported rather than guessed at: silently rounding an
    intermediate form to one of the two extremes is exactly the conflation this
    module exists to prevent."""


def bps_global_form(lines) -> str:
    """The `pure_ade_lattice_data` `global_form=` argument matching `lines`.

    `H = ()` is the simply connected form (`Γ = P^∨ ⊕ P`); `H` the whole centre
    is the adjoint form (`Γ = Q^∨ ⊕ Q`).  Anything between raises
    `IntermediateFormUnsupported` — see that class."""
    H = set(lines.H)
    full = set(range(lines.invariants[0])) if lines.invariants else set()
    if not H:
        return "sc"
    if H == full:
        return "adj"
    raise IntermediateFormUnsupported(
        f"{lines.name}: H={tuple(sorted(H))} is a proper nontrivial subgroup of "
        f"the centre Z_{lines.invariants[0]}; pass an explicit Lagrangian "
        f"(list of (m_vec, e_vec) pairs) to pure_ade_lattice_data instead.")


def bps_lattice_for(lines, factors) -> dict:
    """The charge-lattice data for `factors` at the global form `lines` names.

    `factors` is the ADE spec (`[("A", 1)]`, `[("D", 4)]`, …) — the *quiver*;
    `lines` supplies only the *embedding*.  That split is the point."""
    return pure_ade_lattice_data(factors, global_form=bps_global_form(lines))


def exchange_matrix(data) -> tuple:
    """`⟨n_i, n_j⟩` over the node charges — the quiver, stripped of its
    embedding.  Two global forms of one theory must agree here."""
    B, nodes = data["B"], data["nodes"]
    def pair(u, v):
        return sum(u[i] * B[i][j] * v[j] for i in range(len(u)) for j in range(len(v)))
    return tuple(tuple(pair(a, b) for b in nodes) for a in nodes)


def verify_same_quiver_different_embedding(factors) -> bool:
    """The lesson, as a checkable assertion.

    Across the simply connected and adjoint forms of `factors`: the exchange
    matrix is **identical** (it is the same quiver) while the node charges
    **differ** (they are embedded in different Γ).  Returns `True` when both
    hold; a `False` means the repo has started to conflate the two again."""
    sc = pure_ade_lattice_data(factors, global_form="sc")
    adj = pure_ade_lattice_data(factors, global_form="adj")
    same_quiver = exchange_matrix(sc) == exchange_matrix(adj)
    different_embedding = [tuple(n) for n in sc["nodes"]] != [tuple(n) for n in adj["nodes"]]
    return bool(same_quiver and different_embedding)


def verify_choice_matches(lines, factors, node_charges) -> bool:
    """Whether a BPS chart built on `node_charges` really is the form `lines`
    names — i.e. whether the two sides made the *same* choice.

    This is what ties `PureSO3KAlgebra` to `adjoint_lines(su_2())`: without it,
    the oracle and the abelianized tier agree numerically with nothing in the
    code saying why they should."""
    data = bps_lattice_for(lines, factors)
    return [tuple(n) for n in data["nodes"]] == [tuple(n) for n in node_charges]


if __name__ == "__main__":
    import root_datum as rd
    from global_form import simply_connected_lines, adjoint_lines

    print("one quiver, two embeddings — pure A_1:")
    for label, lines in (("SU(2)", simply_connected_lines(rd.su_2())),
                         ("SO(3)", adjoint_lines(rd.su_2()))):
        data = bps_lattice_for(lines, [("A", 1)])
        print(f"  {label:6s} global_form={bps_global_form(lines)!r:6s} "
              f"nodes={[tuple(n) for n in data['nodes']]}  "
              f"exchange={exchange_matrix(data)}")
    for factors in ([("A", 1)], [("A", 2)], [("D", 4)]):
        ok = verify_same_quiver_different_embedding(factors)
        print(f"  same quiver / different embedding at {factors}: {ok}")
