"""
u1e7_cone_derivation.py
=======================

Chord (cluster-variable) selection for `U1E7GaugedRG` (u(1)-gauged E7), per the
**spine-free QTCone-from-RG recipe**.  This replaces
the earlier ray-heuristic (which mis-modelled the magnetic leg) — see that note
for why the explicit-`(c0,c1)`-atom approach failed.

The recipe (mirrors `u1a1aodd_mult_table.select_chords`):

  * **rays** are labels with monomial RG; many A6 chords are only monomial after
    dressing by a power of the magnetic leg `X_{(1,0)}` (`c0`), so we sweep
    `c0 ∈ {-3..3}` (and the bare monopole `((), (c0, 0))`);
  * group the rays into **ρ-orbits mod E** (`modE` zeroes the gauge leg `c1`, so
    orbits are finite — raw ρ drifts `c1` unboundedly on magnetic chords);
  * an orbit is a **cluster monomial** (drop) iff some element factors as a
    single-term (q-commuting) product of two *lower-norm already-accepted*
    elements (chords AND monomials) with matching charge mod E.

The surviving orbits are the **dyonic chords**; magnetic charge is carried in the
chord identity (charge `phi`), never as a free torus coordinate (`X_{(1,0)}` is
NOT a torus direction — not all its powers are monomial).

On the corrected flow (dressing chord `(3, 0)`): 315 monomial-RG seeds (306
dressed chords, 9 bare monopoles); 200 chords, forming 20 ρ-orbits of length 10
mod E (`c0` distribution −3:5, −2:25, −1:35, 0:70, 1:35, 2:25, 3:5); 4160
cones; and the cone `multiply` closes on every ordered atom pair.  (On the earlier `(2, 2)` dressing — gauged `[A₁,D₇]`, see
`u1a1e7_rgkalgebra.py` — it gave 182 chords and 2508 cones.)

Cost control (same output).  ρ on the flow reflects the gauge leg:
`ρ(E) = E⁻¹`, and `E` q-commutes by one power with every term of `RG(L_x)` (all
terms carry `x`'s `c0`), so `ρ((w, (c0, c1))) = (π, (c0', δ − c1))` with
`(π, (c0', δ)) = ρ((w, (c0, 0)))`; the orbit walk evaluates ρ at `c1 = 0` only.
The product filter's single-term test and mod-E charge are unchanged by
`E`-shifts, so it multiplies `c1 = 0` representatives.  The flow's memo caches
are dropped when the process grows past `_TRIM_MB`.
"""
from __future__ import annotations

import gc
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from u1e7_gauged_rg import U1E7GaugedRG
from a1a2k_bps_iso import _compute_chord_charges

# additive A6 chord charges (6-vector per (a, i)); the U1E7 charge appends (c0,c1)
A6_CHG = _compute_chord_charges(3, {})


def _nz(elt):
    return {l: c for l, c in elt.terms.items() if not c.is_zero()}


def phi(label):
    """Additive 8-charge of a U1E7 label `(chord, (c0, c1))`:
    `Σ exp·A6_charge(a,i)` (6 components) followed by `(c0, c1)`."""
    chord, (c0, c1) = label
    g = [0] * 6
    for (a, i, e) in chord:
        c = A6_CHG[(a, i)]
        for j in range(6):
            g[j] += e * c[j]
    return tuple(g) + (c0, c1)


def modE(g):
    """Charge modulo the gauge leg `E = X_{(0,1)}` (zero the last component)."""
    return g[:7] + (0,)


def _norm(g):
    return sum(abs(v) for v in g)


def is_monomial_rg(T, label) -> bool:
    """True iff `RG(label)` is a single auxiliary monomial (a clean ray)."""
    return len(_nz(T.RG(label))) == 1


# resident size (MB) above which the build drops the flow's memo caches (with
# room below a 3.5 GB address-space cap for one large product)
_TRIM_MB = 2200


def _resident_mb():
    """This process's resident size in MB (`/proc/self/status`), or None."""
    try:
        with open("/proc/self/status") as f:
            return next(int(ln.split()[1]) for ln in f
                        if ln.startswith("VmRSS:")) // 1024
    except (OSError, StopIteration, ValueError):
        return None


def _trim_oracle_caches(T, limit_mb: int = _TRIM_MB, keep=()) -> bool:
    """Drop the flow's memo caches, and its survivor's cone-reducer memo, when
    this process's resident size exceeds `limit_mb` (they are pure memos, so
    only time is traded).  The `RG` images of the labels in `keep` (the atoms,
    whose images are reused by every product) are kept unless memory is still
    above the limit afterwards.  A no-op where the size cannot be measured.
    To avoid trimming on every call when the kept data alone is near the
    limit, it trims again only after the size has grown 256 MB past the level
    left by the previous trim.  Returns whether it trimmed."""
    rss = _resident_mb()
    floor = T.__dict__.get("_trim_floor_mb", 0)
    if rss is None or rss <= max(limit_mb, floor + 256):
        return False
    for k in ("_rho_inv_cache", "_trg_cache", "_rho_cache", "_multiply_cache"):
        T.__dict__.pop(k, None)
    surv = getattr(T, "_surv", None)
    if surv is not None and getattr(surv.cone_data(), "_reduce_word_cache", None):
        surv.cone_data()._reduce_word_cache = {}
    rg = T.__dict__.get("_rg_cache")
    if rg:
        for lab in [lab for lab in rg if lab not in keep]:
            del rg[lab]
    gc.collect()
    if keep and rg and (_resident_mb() or 0) > limit_mb:
        rg.clear()
        gc.collect()
    T.__dict__["_trim_floor_mb"] = _resident_mb() or 0
    return True


def _at_zero_gauge_charge(x):
    """`x` with its gauge charge `c1` set to 0 (`x = E^{c1}·that`, up to a
    power of q)."""
    chord, (c0, _c1) = x
    return (chord, (c0, 0))


def _rho_by_reflection(T, x):
    """`T.rho(x)`, evaluated at `c1 = 0`: `ρ((w, (c0, c1))) = (π, (c0', δ − c1))`
    with `(π, (c0', δ)) = ρ((w, (c0, 0)))` (see the module docstring)."""
    chord, (c0, c1) = x
    pi, (pc0, delta) = T.rho((chord, (c0, 0)))
    return (pi, (pc0, delta - c1))


def _orbit_modE(T, x, cap=40):
    """The ρ-orbit of `x` (closed mod E), as a list of labels."""
    seen = {modE(phi(x))}
    out = [x]
    cur = x
    for _ in range(cap):
        _trim_oracle_caches(T)
        cur = _rho_by_reflection(T, cur)
        m = modE(phi(cur))
        if m in seen:
            break
        seen.add(m)
        out.append(cur)
    return out


def select_chords(T, c0_range=range(-3, 4), c1_range=range(-1, 2)):
    """The dyonic chords of `U1E7GaugedRG`, as a list of native labels.

    Sweeps monomial-RG rays (A6 chords dressed by `X_{(1,0)}^{c0}`, plus the bare
    monopole), groups them into ρ-orbits mod E ordered by charge-norm, and drops
    cluster-monomial orbits (single-term products of two lower-norm accepted
    elements).  Returns every chord at every ρ-position (`mod E`).
    """
    seeds = [(((a, i, 1),), (c0, c1))
             for (a, i) in A6_CHG for c0 in c0_range for c1 in c1_range
             if is_monomial_rg(T, (((a, i, 1),), (c0, c1)))]
    seeds += [((), (c0, c1)) for c0 in c0_range for c1 in c1_range
              if c0 and is_monomial_rg(T, ((), (c0, c1)))]

    orbits, seen = [], set()
    for s in sorted(seeds, key=lambda x: _norm(phi(x))):
        if modE(phi(s)) in seen:
            continue
        el = _orbit_modE(T, s)
        oset = frozenset(modE(phi(e)) for e in el)
        if oset & seen:
            continue
        seen |= oset
        orbits.append(el)
    orbits.sort(key=lambda el: _norm(phi(el[0])))

    accepted: dict = {}     # modE charge -> [elements] (chords AND monomials)

    def mark(e):
        m = modE(phi(e))
        accepted.setdefault(m, [])
        if e not in accepted[m]:
            accepted[m].append(e)

    chords, acc = [], []    # acc: (full charge, element) of accepted chords
    for el in orbits:
        gO = phi(el[0])
        n0 = _norm(gO)
        isprod = False
        for (gc, ce) in acc:
            if _norm(gc) >= n0:
                continue
            need = modE(tuple(gO[j] - gc[j] for j in range(8)))
            for me in accepted.get(need, []):
                if _norm(phi(me)) >= n0:
                    continue
                _trim_oracle_caches(T)
                pr = _nz(T.multiply(_at_zero_gauge_charge(ce),
                                    _at_zero_gauge_charge(me)))
                if len(pr) == 1 and modE(phi(next(iter(pr)))) == modE(gO):
                    isprod = True
                    break
            if isprod:
                break
        for e in el:
            mark(e)
        if not isprod:
            # Keep every ρ-orbit member as a mult-gen — including the
            # multi-letter ones: a multi-letter ρ-image like (1,1)·(1,4) whose
            # constituent single chords do NOT q-commute is a genuine canonical
            # that is *not* a single-letter cone monomial, so it must be its own
            # mult-gen.  (Single-letter `acc` entries drive the product filter.)
            for e in el:
                chords.append(e)
                if len(e[0]) <= 1:
                    acc.append((phi(e), e))
    return chords


if __name__ == "__main__":
    import time
    T = U1E7GaugedRG()
    t0 = time.time()
    chords = select_chords(T)
    bykey = {}
    for ch in chords:
        k = (ch[0], ch[1][0])
        if k not in bykey or abs(ch[1][1]) < abs(bykey[k][1][1]):
            bykey[k] = ch
    from collections import Counter
    print(f"chords: {len(chords)} -> {len(bykey)} (chord_part,c0) atoms "
          f"({time.time()-t0:.1f}s)")
    print("c0 distribution:",
          dict(sorted(Counter(c[1][0] for c in bykey.values()).items())))
