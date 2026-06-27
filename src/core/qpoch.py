"""PowerSeries (truncated Z[[q]] / Z((q))) and q-Pochhammer utilities.

Canonical-surface migration of `schur_index.PowerSeries` and the
q-Pochhammer helpers (Plan 07 Stage A5).  Self-contained: only
depends on stdlib.

This module is the *boundary layer* for Nahm / Schur-index
computations: the `(q^2;q^2)_inf^r` prefactor and the final Schur
output are held as PowerSeries, but intermediate state and overlap
arithmetic lives in exact `HabiroElement` form.

Originally:
Truncated power series in q with integer coefficients, plus q-Pochhammer
utilities.

This module is the *boundary layer* for Nahm / Schur-index computations:
the `(q^2;q^2)_inf^r` prefactor and the final Schur-index output are held
as `PowerSeries`, but the intermediate state and overlap arithmetic lives
in exact `HabiroElement` form (see `habiro.py` and `nahm_data.py`).

What lives where
----------------

* `PowerSeries`, `qpoch_finite`, `qpoch_infty`, `inv_qpoch_finite` — here
  (used by both the Habiro pipeline and downstream consumers).
* `schur_index_nahm`, `schur_gram`, `s_gamma_habiro`, `c_gamma_habiro` —
  `nahm_data.py`.  The modern (Habiro-backed, no K_internal) Schur-index
  pipeline.  Prefer these for all new work.
* `compute_S_ket`, `apply_F_to_state`, `eq_coefficients`, the old
  `schur_index()` function, and `nahm_inner_product` — retired.  The old
  PowerSeries-throughout pipeline is known to fail for
  deep-support F_a (pentagon `(-3,2)` etc.).

All arithmetic here is exact integer; results are truncated to a
prescribed order K in q.
"""

from __future__ import annotations



# ---------------------------------------------------------------------------
# Truncated power series in q with integer coefficients
# ---------------------------------------------------------------------------


class PowerSeries:
    """A formal power series in q truncated to degree <= K.

    Internally stored as a dict  {exponent: int_coefficient}  with only
    nonzero entries.
    """

    __slots__ = ("_c", "K")

    def __init__(self, coeffs: dict[int, int] | None = None, K: int = 40):
        self.K = K
        self._c: dict[int, int] = {}
        if coeffs:
            for e, v in coeffs.items():
                if v != 0 and e <= K:
                    self._c[e] = v

    # --- constructors ---

    @classmethod
    def zero(cls, K: int = 40) -> "PowerSeries":
        return cls(K=K)

    @classmethod
    def one(cls, K: int = 40) -> "PowerSeries":
        return cls({0: 1}, K=K)

    @classmethod
    def qpow(cls, n: int, K: int = 40) -> "PowerSeries":
        """Return q^n."""
        if n > K:
            return cls.zero(K)
        return cls({n: 1}, K=K)

    # --- queries ---

    def is_zero(self) -> bool:
        return not self._c

    def __getitem__(self, e: int) -> int:
        return self._c.get(e, 0)

    # --- arithmetic ---

    def __neg__(self) -> "PowerSeries":
        return PowerSeries({e: -v for e, v in self._c.items()}, self.K)

    def __add__(self, other: "PowerSeries") -> "PowerSeries":
        K = min(self.K, other.K)
        out: dict[int, int] = dict(self._c)
        for e, v in other._c.items():
            if e > K:
                continue
            s = out.get(e, 0) + v
            if s == 0:
                out.pop(e, None)
            else:
                out[e] = s
        return PowerSeries(out, K)

    def __sub__(self, other: "PowerSeries") -> "PowerSeries":
        return self + (-other)

    def __mul__(self, other: "PowerSeries") -> "PowerSeries":
        K = min(self.K, other.K)
        out: dict[int, int] = {}
        for e1, v1 in self._c.items():
            if e1 > K:
                continue
            for e2, v2 in other._c.items():
                e = e1 + e2
                if e > K:
                    continue
                out[e] = out.get(e, 0) + v1 * v2
        return PowerSeries({e: v for e, v in out.items() if v != 0}, K)

    def __rmul__(self, other: int) -> "PowerSeries":
        if other == 0:
            return PowerSeries.zero(self.K)
        return PowerSeries({e: other * v for e, v in self._c.items()}, self.K)

    def shift(self, n: int) -> "PowerSeries":
        """Multiply by q^n."""
        return PowerSeries(
            {e + n: v for e, v in self._c.items() if e + n <= self.K},
            self.K,
        )

    def __repr__(self) -> str:
        if not self._c:
            return "0"
        parts = []
        for e in sorted(self._c):
            v = self._c[e]
            if e == 0:
                parts.append(str(v))
            elif v == 1:
                parts.append(f"q^{e}")
            elif v == -1:
                parts.append(f"-q^{e}")
            else:
                parts.append(f"{v}*q^{e}")
        return " + ".join(parts).replace("+ -", "- ")


# ---------------------------------------------------------------------------
# q-Pochhammer (q^2; q^2)_k and (q^2; q^2)_infty as power series
# ---------------------------------------------------------------------------


def qpoch_finite(k: int, K: int) -> PowerSeries:
    """Compute (q^2; q^2)_k = prod_{j=1}^k (1 - q^{2j}) as a truncated series."""
    result = PowerSeries.one(K)
    for j in range(1, k + 1):
        factor = PowerSeries({0: 1, 2 * j: -1}, K)
        result = result * factor
    return result


def qpoch_infty(K: int) -> PowerSeries:
    """Compute (q^2; q^2)_infty = prod_{j>=1} (1 - q^{2j}) truncated to q^K."""
    result = PowerSeries.one(K)
    for j in range(1, K // 2 + 1):
        if 2 * j > K:
            break
        factor = PowerSeries({0: 1, 2 * j: -1}, K)
        result = result * factor
    return result


def inv_qpoch_finite(k: int, K: int) -> PowerSeries:
    """Compute 1/(q^2; q^2)_k as a truncated power series.

    Since (q^2;q^2)_k starts with constant term 1, inversion is done by
    iterative "long division" in the power series ring.
    """
    denom = qpoch_finite(k, K)
    return _invert_series(denom, K)


def _invert_series(f: PowerSeries, K: int) -> PowerSeries:
    """Invert a power series f with f[0] != 0, truncated to order K."""
    assert f[0] != 0, "constant term must be nonzero to invert"
    # For our use: f[0] = 1 always, so inv[0] = 1.
    inv: dict[int, int] = {0: 1}
    f0 = f[0]
    for n in range(1, K + 1):
        s = 0
        for m in range(1, n + 1):
            fm = f[m] if m in f._c else 0
            if fm != 0 and (n - m) in inv:
                s += fm * inv[n - m]
        if s != 0:
            # inv[n] = -s / f0.  f0 = 1 so inv[n] = -s.
            inv[n] = -s
    return PowerSeries(inv, K)

