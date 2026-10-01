# The pentagon trace-miracle, reduced to a single scalar

*Goal 2.11 (the trace miracle). Code: `experiments/pentagon_miracle_reduction.py`,
`tests/test_pentagon_miracle_reduction.py`. Companion to `a1a2k_all_orders.md`
(Goal 2.12) and `trace_bootstrap.md`.*

## The miracle

The Layer-1 cyclicity reduction expresses every power trace *structurally* —
from `multiply` + `ρ` alone, **no trace values used** — as

> `Tr(L₀^a) = c₁^{(a)}(q)·T₀ + c_L^{(a)}(q)·T₁`,   `T₀=Tr(1)`, `T₁=Tr(L)`.

The miracle (`tests/test_cone_kalgebra.py::test_miracle_*`) is that the
normalised reduction coefficients reproduce the elementary traces themselves:

> `c₁^{(a)}/q^{emin} → −T₁`,   `c_L^{(a)}/q^{emin} → T₀`   (as `a → ∞`).

A *structural* quantity (a reduction coefficient) converging to an *analytic*
one (the trace) is the 20-year puzzle; it is what pins `Tr(1)` (removing the
`1+O(q)` rescale freedom that orthonormality alone leaves — Goal 2.12).

## The reduction

`Q((q))²` has basis `{(−T₁,T₀), (T₀,T₁)}` — a basis because the Gram unit
`T₀²+T₁² = 1+O(q)` is invertible. Decompose the reduction vector:

> `(c₁^{(a)}, c_L^{(a)}) = λ_a·(−T₁,T₀) + μ_a·(T₀,T₁)`.

Pairing with `(T₀,T₁)` gives `c₁T₀ + c_LT₁ = μ_a·(T₀²+T₁²)`, and the left side
**is** `Tr(L₀^a)`, which orthonormality forces to be `O(q^a)` (since
`L₀^a ≠ 1`, `Tr(L₀^a) = ⟨L₀^a,1⟩ = O(q^a)`). Hence

> **(★)  `μ_a = Tr(L₀^a)/(T₀²+T₁²) = O(q^a)`**  — the transverse component
> vanishes to order `a`, *forced by orthonormality* with no analytic input.

The other basis vector `(−T₁,T₀)` is exactly the orthonormality null-direction
(`(−T₁)·T₀ + T₀·T₁ ≡ 0`). Since `emin(a) → −∞` (the coefficients carry deep
negative powers, `emin = −(a²+…)`-scale), normalising by `q^{emin}` annihilates
the `O(q^a)` transverse part:

> `(c₁,c_L)/q^{emin} = (λ_a/q^{emin})·(−T₁,T₀) + O(q^{a−emin})`.

Therefore:

> **The pentagon miracle ⟺ the single scalar convergence `λ_a/q^{emin(a)} → 1`,**

with `λ_a = (c_L·T₀ − c₁·T₁)/(T₀²+T₁²)` the residual scalar. Verified exactly
(`experiments/pentagon_miracle_reduction.py`, seeds to `q²⁶⁰`):

| `a` | `emin` | `val Tr(L^a)` | `val μ_a` (★) | `λ_a/q^{emin}` `=1` through |
|--:|--:|--:|--:|--:|
| 4 | −15 | 4 | 4 | `q⁸` |
| 6 | −35 | 6 | 6 | `q¹²` |
| 8 | −63 | 8 | 8 | `q¹⁶` |
| 10 | −99 | 10 | 10 | `q²⁰` |
| 12 | −143 | 12 | 12 | `q²⁴` |
| 14 | −195 | 14 | 14 | `q²⁸` |

i.e. `λ_a/q^{emin} = 1 + O(q^{2a})`.

## What this buys, and what remains

* **Provable / proven here:** the *direction* `(−T₁,T₀)` and the *transverse
  vanishing* `μ_a=O(q^a)` are pure consequences of orthonormality + 2D linear
  algebra (the Gram unit). They carry **none** of the miracle's mystery.
* **The residual open core:** the scalar `λ_a/q^{emin} → 1`. This is the genuine
  q-difference / resummation statement — the overall normalisation of the
  cyclicity recursion's deep-negative-power coefficients converging to the
  canonical scale (the user's "feel of `q ↦ 1/q` structure", the resummed /
  Habiro layer). Proving it is the residual; the empirical rate is `O(q^{2a})`.

## Scope — why this is special to the pentagon

The 2D collapse uses that there are exactly **two** seeds, so the orthonormality
null-space of `(T₀,T₁)` is **1-dimensional** — the direction is pinned. For
`A1A2k(k)` with `k ≥ 2` (heptagon, nonagon, …) there are `k+1` seeds, the
null-space of `(T₀,…,T_k)` is `k`-dimensional, and orthonormality no longer pins
the direction: the miracle there genuinely needs more than the linear-algebra
collapse (consistent with its being open). The reduction here is therefore a
sharp result for the two-seed corner, isolating exactly where the analytic input
must enter.

## Sharper: the recursion-level reduction (convergence proven)

`experiments/pentagon_miracle_recursion.py` analyses the residual scalar `λ_a`
via the cyclicity recursion and reduces the miracle further — to a single
leading-coefficient identity, with the convergence now *proven*.

The structural reduction coefficients satisfy the pentagon cyclicity recursion
(verified directly, both components):
`c^{(a)} = q^{1-2a}c^{(a-1)} + q^{2-2a}c^{(a-2)}`, `c^{(0)}=(1,0)`, `c^{(1)}=(0,1)`.
Hence `λ_a` does too, and:

* **`emin(a) = 1 − a²`** — closed form (`emin(a)=emin(a-1)+1-2a`, `emin(1)=0`).
* Substituting `λ_a = q^{1-a²}·ψ_a` gives the **normalised recursion**
  `ψ_a = ψ_{a-1} + q^{2a-2}·ψ_{a-2}`. The correction has valuation `2a-2 → ∞`,
  so `ψ_a = ψ_∞ + O(q^{2a})` — **convergence and rate are manifest (proven)**,
  not empirical. The miracle ⟺ `ψ_∞ = 1`.
* **Casoratian.** With the recessive solution `μ_a = Tr(L₀^a)/(T₀²+T₁²) = O(q^a)`,
  `C_a = λ_a μ_{a-1} − λ_{a-1} μ_a = (−1)^{a-1} q^{a(1-a)}/(T₀²+T₁²)` (exact).
  Feeding `λ_a = q^{1-a²}ψ_a` and `a→∞` gives `Tr(L₀^a) ~ (−1)^a q^a / ψ_∞`, so

> **`ψ_∞ = 1`  ⟺  `Tr(L₀^a)` has leading coefficient `(−1)^a` at `q^a`**

(verified `a=1..12`). The right side is "the trace's leading term is
multiplicative on powers of `L₀`" (`Tr(L₀)[q¹] = −1`). So the **entire pentagon
miracle reduces to this one leading-coefficient identity** — everything else
(direction, transverse vanishing, `emin` closed form, convergence + rate,
Casoratian) is proven/verified.

### The residual, characterised: the `Tr(L₀^n)` gap structure

The power-trace has an explicit shape (verified `n=1..7`):

> `Tr(L₀^n)·(−1)^n = q^n + q^{3n+4}·(1 + q² + …)`  —  leading `(−1)^n` at `q^n`,
> a **zero gap on `[n+1, 3n+3]`**, second nonzero term `(−1)^n` at `q^{3n+4}`,
> then an RR/partition-like tail.

This makes `Tr(L₀^a)[q^a]=(−1)^a` provable **by induction on the gap structure**:
the cyclicity recursion gives, at order `q^a`,

> `t_{a,a} = t_{a-1,3a-1} + t_{a-2,3a-2}`,

and `t_{a-1,3a-1}=0` (since `3a-1 ∈ [a,3a]`, the gap of `Tr(L^{a-1})`) while
`t_{a-2,3a-2}=(−1)^{a-2}` (since `3a-2 = 3(a-2)+4`, the *second* nonzero term of
`Tr(L^{a-2})`), so `t_{a,a}=(−1)^a` (verified `a=3..8`). The induction therefore
closes the leading coefficient **given the full gap structure** — and that gap
structure is the genuine Rogers–Ramanujan content (the M(2,5) fine structure),
the irreducible residual of the miracle.

### Creative reduction: the residual as an elementary recursion (no RR input)

Write `Tr(L₀^m)·(−1)^m = q^m + q^{3m+4}·U_m`, so `U_m` is a power series with
`U_m = 1 + O(q²)` (this *is* the gap structure). Feeding the form into the
cyclicity recursion `Tr(L₀^a)=q^{1-2a}Tr(L₀^{a-1})+q^{2-2a}Tr(L₀^{a-2})` and
simplifying gives an **elementary, self-contained recursion** for the tails
(verified `a=2..7`):

> **`U_{a-2} − q²·U_{a-1} = 1 + q^{2a+4}·U_a`**   (a ≥ 2).

Crucially this carries **no Rogers–Ramanujan / character input** — it follows
from the cyclicity recursion alone. It re-expresses the whole residual:

> **MIRACLE ⟺ this recursion preserves the gap** — i.e. each `U_a`, defined by
> `q^{2a+4}U_a = U_{a-2} − q²U_{a-1} − 1`, is again a power series with
> `U_a[q⁰]=1` (equivalently, `U_{a-2} − q²U_{a-1} − 1` starts exactly at
> `q^{2a+4}` with coefficient 1).

The stationary limit `U_∞ = 1/(1−q²)` is consistent (`U_{a-2}−q²U_{a-1} →
1/(1−q²) − q²/(1−q²) = 1`). So the open core is now a **pure q-series problem
about an explicit elementary recursion**, fully decoupled from the analytic RR
character theory — the cleanest statement of what remains. Code:
`verify_U_recursion`, `verify_U_starts_at_one`.

**Sharpest form (`verify_U_congruent_geometric`).** The correction
`W_m = U_m − 1/(1−q²)` starts *exactly* at `q^{2m+8}`, i.e.

> **`U_m ≡ 1/(1−q²)  (mod q^{2m+8})`**   (verified `m=0..7`),

so each `U_m` agrees with the plain geometric series `1+q²+q⁴+…` through order
`q^{2m+7}` (giving `U_m[q⁰]=1` and the gap at once), and `U_m → 1/(1−q²)`.
Substituting `U_m = 1/(1−q²) + W_m` into the elementary recursion gives the
correction recursion

> `W_{a-2} − q²W_{a-1} = q^{2a+4}·(1/(1−q²) + W_a)`,

which propagates the onset. Proving the onset is *exactly* `2m+8` for all `m`
(equivalently `U_m ≡ 1/(1−q²)` to that order) is the **irreducible
Rogers–Ramanujan cancellation** — the floor of this reduction: every further
structural step restates the same q-series cancellation, so closing it requires
genuine RR / q-difference machinery, not more algebraic peeling.

## Pointers

* `experiments/pentagon_miracle_reduction.py` — `reduction_decomposition(a)`.
* `experiments/pentagon_miracle_recursion.py` — the recursion-level chain
  (`verify_reduction_recursion`, `…_emin_closed_form`,
  `…_normalized_recursion_and_convergence`, `…_casoratian`,
  `…_leading_coeff`); `tests/test_pentagon_miracle_recursion.py`.
* `tests/test_cone_kalgebra.py::test_miracle_*` — the miracle as originally pinned.
* `a1a2k_all_orders.md` — the Goal-2.12 (uniqueness-up-to-rescale) side, where
  the same family's all-orders closure was reduced to the Andrews–Gordon
  (non-C-finite) structure.
* `RESEARCH_GOALS.md` §2.11; `repo_audit.md` Level D.
