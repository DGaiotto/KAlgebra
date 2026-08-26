# Verification scope

`python3 run_tests.py` is the validation gate. This note records what it
does **not** certify, so that "the gate is green" is read at its true
strength. Two companion notes cover adjacent ground:
[`frozen-data-provenance.md`](frozen-data-provenance.md) (what validates
the shipped `.pkl` tables and the generated finite-type zoo, given that
their builders are not included) and the `conjectures-step*.md` files
(the verification-scope table for each layer's conjectural content).

Everything below is documented at its point of use as well; this page
exists so that a reader asking "what is actually proven here?" does not
have to find two docstrings deep in the source to get the answer.

## Axiom checkers that are best-effort by construction

`KAlgebra` derives a family of `verify_*` checkers, and the gate runs
them across the samples, the cone realisations, the RG flows, the BPS
realisation, the abelianized presentations, the skein algebras, and the
general-`G` charts — the bar involution, the unit law, associativity,
the `ρ`-automorphism property, `ρ²`-twisted trace cyclicity, and
orthonormality.

Two `RGKAlgebra` checkers are **not** wired into the gate, and that is
deliberate rather than an oversight:

| checker | what it would assert | why it is not gate-wired |
|---|---|---|
| `verify_rg_twist` (`rgkalgebra.py`) | `RG_a · S_RG = S_RG · ρ_aux⁻¹(RG_{ρ_self(a)})` | The truncated form fails strict equality at the boundary: the filtration cutoff and the `𝖖`-expansion each introduce residuals, so a `False` at the window edge carries no information. Certifying it needs a boundary-aware rewrite. |
| `verify_rg_inner_product` | `I^self_{a,b}(q) = ⟨RG_a · S_RG, RG_b · S_RG⟩_aux(q)` | Same boundary residuals, same consequence. |

Both are usable interactively away from the window edge; neither is
evidence the gate collects. The RG-side properties the gate *does*
certify are `verify_rg_discovery` and `verify_F_S_leading`, together
with the full axiom battery on every flow.

## A search bound that is proven only in the pointed case

`cone_contains` (`src/cone/lattice.py`, with a second copy in
`src/bps/bps_quiver_tools.py`) decides whether a vector lies in the cone
spanned by a generator set, by enumerating the free variables of a
lattice system up to `max_t`. That bound has two regimes:

* **Proven.** `_strict_witness_box` searches a small box for a strict
  witness `f` with `⟨f, g_i⟩ ≥ 1` on every generator. When one is found,
  any non-negative representation `v = Σ λ_i g_i` satisfies
  `Σ λ_i ≤ ⟨f, v⟩`, so `max_t = ⟨f, v⟩` makes the enumeration
  **complete**. This is the path taken for a pointed cone.
* **Heuristic.** When no witness is found inside the box — a non-pointed
  cone, or a pointed one whose witness lies outside the box, whose
  half-width is `max(3, 2·max|g_i|)` — the fallback is the norm estimate
  `max(‖v‖₁ // min_i ‖g_i‖₁ + 2, 10)`. For a strongly sheared
  overcomplete generator set this can in principle under-bound, and a
  false negative silently truncates F-solver support rather than raising.

Closing the second regime means an exact LP or a proven bound covering
the non-pointed case. Nothing in the shipped realisations is known to
land there — every cone the gate exercises takes the witness path — but
the guarantee is conditional, not absolute.

## What the abelianized tier's axiom checkers do and do not certify

`star_bubbling.verify_axioms` runs the five conditions of the `L_{m,e}`
construction against a **finished** element. Read at its true strength:

| condition | what a pass means |
|---|---|
| `bar (W1)` | **enforced** for a solved element — the solver imposes palindromicity — and **emergent** for one built any other way, which is why the retired constructive routes are kept runnable |
| `seed (W2)` | **emergent as a label check**: it is what identifies *which* canonical the element is, so it catches an element filed under the wrong `e` |
| `support` | **subsumed**, not independent — measured to follow from (★) + W2 (30/30 out-of-support grafts rejected). Kept because it is cheap and localises a failure |
| `O(𝖖)` | **enforced** for a solved element; the check itself is exact, with the truncation bound discharged rather than assumed |
| `(★)` | **emergent, and the load-bearing one** — a membership condition on `Λ = R(G)`, so it is the only one of the five that can reject a bubbling-stripped candidate (23/23) |

Three things it does **not** certify:

* **uniqueness.** That is a property of the linear *system*, not of a finished
  element, and it stays the solver's own first guard. Passing this checklist does
  not say the label is pinned.
* **closure.** That the canonicals span the algebra of (★)-satisfying operators
  is open — see [`conjectures-step5-abe.md`](conjectures-step5-abe.md) §2b. The
  gate measures it on products by exact reconstruction, which is a measurement.
* **that the axiom list is complete.** The negative controls show each condition
  catches something the others do not; they do not show that nothing else is
  needed.

A related and easily-misread point: the derivation in `conjectures-step5-abe.md`
shows that orthonormality at `𝖖⁰` is **blind to the bubbling**. So an
orthonormality pass is *not* evidence that a canonical is complete — it is
evidence about the leading orbit only. Where the gate reports both, (★) is the
one carrying the weight.

## What "spine-free" asserts

Seven of the eight Step-3 RG suites assert that no realisation-spine
module (`src/bps/`) is imported, against a shared module list derived
from the filesystem (`tests/_spine.py`) rather than hand-maintained.
This certifies that those layers *compute* without the BPS engine. It is
a statement about the import graph at run time, not a proof that the
mathematics is independent of it.
