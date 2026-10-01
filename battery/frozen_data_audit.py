"""Which presentation of each finite-type row rests on analytic statements plus
bootstrap, and which reads frozen data?  (the design record: the companion's
examples rest on analytic statements + bootstrap strategies, never on frozen
tables.)

Each candidate is built, traced (Tr 1 and one seed, K = 4) and multiplied in a
FRESH process with three instruments on:

  * the zoo's frozen trace-table module, `elem_trace_data`,
    made to raise on import (it is empty, and its reader in `elem_traces` —
    `_frozen`, `supported_ids` — was retired on 2026-09-23, the design record; a
    reader brought back would have to import it);
  * an `open()` spy recording every non-Python file the run opens (pickles,
    json, gz);
  * `sys.setprofile` from BEFORE the candidate's import, recording the files
    whose FUNCTIONS execute (a mere import is not use; an import-time
    computation is).

Afterwards every repo module the run loaded is scanned for numeric-literal
density.  A frozen data module is read by attribute lookup, so it never runs a
function; density is what exposes it.  Measured separation (2026-09-22): the
frozen modules sit at 1.4, 3.7 and 12.3 literals per line; the densest CODE any
candidate loads is 0.58 (`u1_pgon_layer2.py`, closed-form sign/shift rules).
The threshold 1.0 sits in that gap; the maximum code density is printed so a
drift toward it is visible.

Positive controls, asserted (the script aborts rather than report if one fails):
  FROZEN  the zoo `FiniteE8KAlgebra` — through the density of its exported cone
          table (`finite_e8_kalg.py`, about 207 numeric literals per line).  It
          was also the frozen-trace-table control until 2026-09-23, when the
          last table (e8's) left the serving path;
          earlier still it was `FiniteE6KAlgebra`, whose table went
  FROZEN  a direct import of `elem_trace_data` (the import
          guard's own control: no zoo trace reads a frozen table any more)
  FROZEN  the retired stand-alone `U1OctagonKAlg`, loaded from the source repository's archive (its
          frozen chord charges + the chord-pair product table).  Until
          2026-09-23 the control was the named `OctagonKAlg`, which wrapped it;
          since the named and stand-alone gauged/ungauged [A1,A_2k+1] classes
          were put on `U1A1AoddKAlg(k)` `OctagonKAlg` reads no
          frozen data and is an ordinary candidate below
  CHART   a `BPSKAlgebra` pentagon (its trace runs `bps_kalgebra.py`)

Its instruments (`INSTRUMENTS`, `CHART_FILES`, `literal_density`) are shared with
the suite in the source repository, the per-row regression guard of the classes that
serve the ADE rows, which loads this module by path.

Run from the repo root:
    PYTHONPATH=.:implementations python the design record
"""
import ast
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the release root
DENSITY_FROZEN = 1.0      # numeric literals per line
MIN_LITERALS = 100
# The BPS engine: a candidate whose run executes a function of one of these runs
# the BPS chart.
CHART_FILES = ("bps_kalgebra.py", "bps_factor_spectrum.py")

# (name, row, import, constructor, seed, second factor, expected verdict or None)
CANDIDATES = [
    ("zoo FiniteE8KAlgebra", "control", "from finite_e8_kalg import FiniteE8KAlgebra as K", "K()",
     "((0,1),)", "((1,1),)", "FROZEN"),
    ("elem_trace_data import", "control", "import finite_kalgebras.elem_trace_data as K", "K",
     None, None, "FROZEN"),
    ("legacy U1OctagonKAlg", "control", "from legacy.u1_octagon_kalg import U1OctagonKAlg as K", "K()",
     "(((2,0,1),),0)", "(((2,1,1),),0)", "FROZEN"),
    ("BPSKAlgebra pentagon", "control", "from bps_kalgebra import BPSKAlgebra as K",
     "K(pairing=[[0,1],[-1,0]], node_charges=[(1,0),(0,1)])", "(1,0)", "(0,1)", "CHART"),
    ("A1A2kKAlg(2)", "[A1,A_2k], k=2", "from a1a2k_kalg import A1A2kKAlg as K", "K(2)",
     "((1,0,1),)", "((1,1,1),)", None),
    ("ungauge_u1a1aodd(2)", "[A1,A_2k+1], k=2", "from ungauge_kalgebra import ungauge_u1a1aodd as K", "K(2)",
     "(((2,0,1),),0)", "(((2,1,1),),0)", None),
    ("OctagonKAlg", "[A1,A_2k+1], k=2 (named)", "from octagon_kalg import OctagonKAlg as K", "K()",
     "(((2,0,1),),0)", "(((2,1,1),),0)", None),
    ("A1D3KAlg", "[A1,D3]=[A1,A3]", "from a1d3_kalg import A1D3KAlg as K", "K()", "A.T(0)", "A.T(1)", None),
    ("SU3ADKAlg", "[A1,D4]", "from su3_ad_kalg import SU3ADKAlg as K", "K()", "A.T(0)", "A.T(1)", None),
    ("A1DoddRGKAlgebra(1)", "[A1,D_2k+3], k=1: D5", "from a1dodd_rgkalgebra import A1DoddRGKAlgebra as K", "K(1)",
     None, None, None),
    ("A1DevenRGKAlgebra(3)", "[A1,D_2k+2], k=3: D8", "from a1deven_rgkalgebra import A1DevenRGKAlgebra as K", "K(3)",
     None, None, None),
    ("E6RGKAlgebra", "[A1,E6]", "from e6_rgkalgebra import E6RGKAlgebra as K", "K()", None, None, None),
    ("E7RGKAlgebra", "[A1,E7]", "from e7_rgkalgebra import E7RGKAlgebra as K", "K()",
     "(((3,0,1),),(1,))", None, None),
    ("E8RGKAlgebra", "[A1,E8]", "from e8_rgkalgebra import E8RGKAlgebra as K", "K()", None, None, None),
    ("U1A1AoddKAlg(2)", "gauged [A1,A5]", "from u1a1aodd_kalg import U1A1AoddKAlg as K", "K(2)",
     "(((1,0,1),),0)", "(((1,1,1),),0)", None),
]

# The three instruments, installed at the top of a fresh child process before
# anything of the candidate is imported.  Shared with
# the suite in the source repository, which loads this module by path and runs its
# rows under the same code; `%(repo)r` is the only substitution.
INSTRUMENTS = r'''
import sys, os, json, signal, builtins, io, gzip
REPO = %(repo)r
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "implementations"))
class FrozenRead(RuntimeError): pass
# The guard is an import hook, so that installing it does not itself import the
# zoo package (whose __init__ preloads every zoo module): importing the frozen
# trace-table module raises, and is recorded in `guard_hits` in case the caller
# swallows the exception.
import importlib.abc
guard_hits = []
class _Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "finite_kalgebras.elem_trace_data":
            guard_hits.append(name)
            raise FrozenRead("import finite_kalgebras.elem_trace_data")
        return None
sys.meta_path.insert(0, _Guard())
opened = set()
_open, _gz = builtins.open, gzip.open
def spy(f, *a, **k):
    try: opened.add(os.path.abspath(os.fspath(f)))
    except Exception: pass
    return _open(f, *a, **k)
def spy_gz(f, *a, **k):
    try: opened.add(os.path.abspath(os.fspath(f)))
    except Exception: pass
    return _gz(f, *a, **k)
builtins.open = spy; io.open = spy; gzip.open = spy_gz
# `ran`: the files whose functions execute; `calls`: the functions themselves,
# as (file, qualified name) -- the bare name before Python 3.11.
ran = set()
calls = set()
def prof(frame, event, arg):
    if event == "call" and (frame.f_code.co_flags & 3) == 3:
        code = frame.f_code
        ran.add(code.co_filename)
        calls.add((code.co_filename, getattr(code, "co_qualname", code.co_name)))
# Importing the guard's package preloads the whole zoo (`finite_kalgebras/__init__`),
# so a module counts as the candidate's only if the candidate loads it or runs it.
preloaded = set(sys.modules)
'''

CHILD = INSTRUMENTS + r'''
res = {}
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError()))
signal.alarm(280)
sys.setprofile(prof)
try:
    %(imp)s
    A = %(ctor)s
    res["tr1"] = str(A.trace(A.identity(), K=4))[:100]
    s1 = %(s1)s
    if s1 is not None:
        res["tr_seed"] = str(A.trace(s1, K=4))[:100]
        s2 = %(s2)s
        if s2 is not None:
            res["product"] = str(A.multiply(s1, s2))[:100]
    res["frozen_trace_read"] = None
except FrozenRead as ex:
    res["frozen_trace_read"] = str(ex)
except TimeoutError:
    res["error"] = "TIMEOUT"
except Exception as ex:
    res["error"] = "%%s: %%s" %% (type(ex).__name__, str(ex)[:100])
finally:
    sys.setprofile(None)
signal.alarm(0)
inrepo = lambda p: p.startswith(REPO + os.sep)
rel = lambda p: os.path.relpath(p, REPO)
res["executed"] = sorted({rel(p) for p in ran if inrepo(p)})
res["loaded"] = sorted({rel(m.__file__) for k, m in list(sys.modules.items())
                        if k not in preloaded and getattr(m, "__file__", None) and inrepo(m.__file__)})
res["opened"] = sorted({rel(p) for p in opened if inrepo(p) and not p.endswith((".py", ".pyc"))})
print("RESULT " + json.dumps(res), flush=True)
'''

_stats = {}


def literal_density(relpath):
    """(numeric literals, lines) of a repo .py file."""
    if relpath not in _stats:
        try:
            src = open(os.path.join(REPO, relpath)).read()
            tree = ast.parse(src)
            n = sum(1 for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
                    and not isinstance(node.value, bool))
            _stats[relpath] = (n, src.count("\n") + 1)
        except (OSError, SyntaxError, ValueError):
            _stats[relpath] = (0, 1)
    return _stats[relpath]


def run(name, row, imp, ctor, s1, s2):
    code = CHILD % {"repo": REPO, "imp": imp, "ctor": ctor, "s1": s1 or "None", "s2": s2 or "None"}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=320, cwd=REPO)
    lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not lines:
        return {"name": name, "row": row, "verdict": "CHILD FAILED", "stderr": p.stderr[-400:]}
    r = json.loads(lines[-1][7:])
    dense, code_max = [], (0.0, "")
    for f in sorted(set(r["loaded"]) | set(r["executed"])):
        n, L = literal_density(f)
        d = n / max(L, 1)
        if n >= MIN_LITERALS and d >= DENSITY_FROZEN:
            dense.append("%s (%.2f/line, %d literals)" % (f, d, n))
        elif d > code_max[0]:
            code_max = (d, f)
    chart = [f for f in r["executed"] if os.path.basename(f) in CHART_FILES]
    if r.get("error"):
        verdict = "ERROR"
    elif r.get("frozen_trace_read") or dense or r["opened"]:
        verdict = "FROZEN"
    elif chart:
        verdict = "CHART"
    else:
        verdict = "ANALYTIC"
    return {"name": name, "row": row, "verdict": verdict, "tr1": r.get("tr1"),
            "frozen_trace_read": r.get("frozen_trace_read"), "dense_modules": dense,
            "data_files_opened": r["opened"], "bps_chart_executed": chart,
            "max_code_density": "%.2f (%s)" % code_max, "error": r.get("error")}


def main():
    for name, row, imp, ctor, s1, s2, expected in CANDIDATES:
        out = run(name, row, imp, ctor, s1, s2)
        print(json.dumps(out), flush=True)
        if expected is not None and out["verdict"] != expected:
            print("ABORT: positive control %r returned %s, expected %s -- the detector "
                  "cannot be trusted" % (name, out["verdict"], expected), flush=True)
            sys.exit(1)
    print("FROZEN_DATA_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
