"""Condition (b): are a class's traces FULLY defined?

The companion's Section 2 extended examples admit
only self-contained classes with an analytic canonical basis AND fully defined
traces.  This sweep traces canonical labels and classifies every call:

  su3ad   `SU3ADKAlg` ([A_1,D_4]): every canonical label with a + b <= 3 and
          flavour character 1, 3 or 3bar, at K = 12, plus four deep labels at
          K = 20.  Positive control: the identity reproduces the printed vacuum
          head 1 + chi_(1,1) q^2 + ...
  u1aodd  `U1A1AoddKAlg(k)`, k = 1, 2, 3 (the u(1)-gauged [A_1, A_{2k+1}]):
          every E^n (|n| <= 8), every chord times E^n (|n| <= 6) and every chord
          power up to 6, at K = 16.  Each call is CLEAN (the canonical
          representative passes the class's own well-formedness guard),
          RESCUED (another representative of its rho^2-orbit passes), or
          REFUSED (none does: the class honest-fails).

Run from the repo root:
    PYTHONPATH=. python3 battery/trace_definedness.py su3ad
    PYTHONPATH=. python3 battery/trace_definedness.py u1aodd
"""
import os, sys, time, signal
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the release root
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "implementations"))
which = sys.argv[1]

def timed(f, *a, limit=120):
    signal.signal(signal.SIGALRM, lambda *x: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(limit)
    t = time.time()
    try:
        r = f(*a); return "ok", r, time.time() - t
    except NotImplementedError as e:
        return "refused", str(e)[:160], time.time() - t
    except TimeoutError:
        return "timeout", None, limit
    except Exception as e:
        return "error", "%s: %s" % (type(e).__name__, str(e)[:160]), time.time() - t
    finally:
        signal.alarm(0)

if which == "su3ad":
    from su3_ad_kalg import SU3ADKAlg
    A = SU3ADKAlg()
    labels = set()
    for tile in range(8):
        for a in range(4):
            for b in range(4 - a):
                for (p, q) in ((0, 0), (1, 0), (0, 1)):
                    labels.add(A.canonicalise((tile, a, b, p, q)))
    labels = sorted(labels)
    # positive control: the identity must give the printed vacuum head 1 + chi_(1,1) q^2
    st, r, dt = timed(lambda: A.trace(A.identity(), K=4))
    print("CONTROL Tr1 K=4:", st, str(r)[:80], flush=True)
    counts = {}
    worst = 0.0
    for lab in labels:
        st, r, dt = timed(lambda l=lab: A.trace(l, K=12))
        counts[st] = counts.get(st, 0) + 1
        worst = max(worst, dt)
        if st != "ok":
            print("  ", lab, st, r, flush=True)
    print("SU3AD K=12: %d canonical labels (a+b<=3, chi in {1,3,3bar}): %s; slowest %.1f s" % (len(labels), counts, worst), flush=True)
    for lab in [(0, 3, 2, 1, 1), (1, 2, 3, 0, 2), (4, 4, 0, 2, 0), (5, 0, 5, 0, 0)]:
        lab = A.canonicalise(lab)
        st, r, dt = timed(lambda l=lab: A.trace(l, K=20), limit=300)
        print("  deep", lab, "K=20:", st, "%.1f s" % dt, str(r)[:70] if st == "ok" else r, flush=True)
    print("DONE su3ad", flush=True)

elif which == "u1aodd":
    from u1a1aodd_kalg import U1A1AoddKAlg
    for k in (1, 2, 3):
        A = U1A1AoddKAlg(k)
        H = 2 * k + 4
        # instrument the guard: count first-try passes vs orbit-retry rescues
        stats = {"clean": 0, "rescued": 0, "refused": 0, "error": 0, "timeout": 0}
        refused_examples = []
        base_trace = type(A).__mro__[1].trace
        guard = A._guard_trace_wellformed
        def classify(lab, K):
            a = A._canonical_rho2_orbit_rep(lab)
            first = base_trace(A, a, K)
            try:
                guard(a, first); return "clean"
            except NotImplementedError:
                pass
            st, r, dt = timed(lambda: A.trace(lab, K))
            return "rescued" if st == "ok" else st
        labels = [((), n) for n in range(-8, 9)]
        for t in range(1, k + 2):
            for i in range(H):
                for n in range(-6, 7):
                    labels.append((((t, i, 1),), n))
                for m in range(2, 7):
                    labels.append((((t, i, m),), 0))
        seen = set()
        for lab in labels:
            try:
                c = A.canonicalise(lab) if hasattr(A, "canonicalise") else lab
            except Exception:
                c = lab
            if c in seen: continue
            seen.add(c)
            st = timed(lambda l=c: classify(l, 16))
            verdict = st[1] if st[0] == "ok" else st[0]
            stats[verdict] = stats.get(verdict, 0) + 1
            if verdict in ("refused", "error", "timeout") and len(refused_examples) < 6:
                refused_examples.append((c, verdict))
        print("U1A1AoddKAlg(%d) K=16: %d labels -> %s" % (k, len(seen), stats), flush=True)
        for ex in refused_examples:
            print("   e.g.", ex, flush=True)
    print("DONE u1aodd", flush=True)
