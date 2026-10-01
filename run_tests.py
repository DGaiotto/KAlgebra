"""Validation gate — pure Python 3, no third-party dependencies.

Puts every ``src/<layer>/`` directory on ``sys.path`` (the project's bare-name
import convention), then runs the contract self-tests for every layer:

    python3 run_tests.py

Step 1 / Step 2 (core, samples, cones, isomorphism witnesses):

  * ``tests/test_samples.py``          — Step-1 sample algebras
  * ``tests/test_cones.py``            — Step-2 ConeKAlgebra realizations + zoo
  * ``tests/test_sample_cone_iso.py``  — Step-1 ↔ Step-2 KAlgebraIso witnesses

Step 3 (the live RG-flow engine; every suite asserts spine-freeness via the
shared, filesystem-derived list in ``tests/_spine.py``):

  * ``tests/test_rg_flows.py``         — the three reference flows
  * ``tests/test_a1an_chain.py``       — the A-type Argyres-Douglas chain
  * ``tests/test_dn_chain.py``         — the D-type gauged chain
  * ``tests/test_e_type.py``           — the E6 / E8 / gauged-E7 flows
  * ``tests/test_flavoured_fork.py``   — the ungauged add_flavour fork
  * ``tests/test_over_pure.py``        — the over-pure SU(2) gauge-theory corner
  * ``tests/test_su2_gauged_chain.py`` — the nested SU(2)-gauged chain
  * ``tests/test_wild.py``             — the "wild" formal flows

Step 4 (the BPS-quiver realisation engine — *this* is the spine):

  * ``tests/test_bps_flows.py``        — the BPS realisation + atlas + node-drop RG

Step 5 (the abelianized-presentation tier + the object-layer capstone; relies on
Steps 1-4 by design, so it runs after the spine-free suites):

  * ``tests/test_abe_flows.py``        — the AbeKAlgebra tier (pure U(N)/SU(N),
    matter as Abe+RG, N=2* machinery, the KAlgebraObject layer, SU(2)/SU(3)
    families, and the flow-typed RGKAlgebraObjects)

Step 6 (the skein tier — SU(2) skein algebras of marked surfaces realised as
``A_𝖖[T]``; relies on Steps 1-5, so it also runs after the spine-free suites):

  * ``tests/test_skein_flows.py``      — the SkeinKAlgebra tier (the intrinsic
    Kauffman-bracket layer, the two-parent ``SkeinKAlgebra`` class, the named
    roster, the KAlgebraIso / KAlgebraObject legs, the contract axioms, the
    ``SkeinAtlas`` flip charts, and the independent BPS-quiver vacuum anchor)

Step 7 (gauge theory at an arbitrary 4d gauge group with arbitrary matter;
relies on Steps 1-6, so it also runs after the spine-free suites):

  * ``tests/test_gn_flows.py``         — general-`G` pure gauge over an arbitrary
    root datum with the guarded route ladder and the licensed (star) solve, `(G, N)`
    matter at any datum and any matter representation, the 4d gauge group data and
    its Langlands duals, and the vacuum-state Schur pairing

Notes on the order and the entry point:

  * ``pytest`` is **not** a supported entry point and is refused loudly by
    ``conftest.py``.  It would skip ``test_cones.py`` and
    ``test_sample_cone_iso.py`` (neither exposes ``test_``-prefixed
    collectables), and importing the BPS suite at collection time would defeat
    the spine-freeness assertions the Step-3 suites make.  Use this script.
  * ``test_cones.py`` is the slowest suite by a wide margin — it walks the whole
    finite-type cone zoo.
  * the spine-importing tiers (Steps 4-7) run **last**, after every suite that
    asserts no spine module is loaded, so that ordering is what makes those
    assertions meaningful rather than accidental.

"""
import pathlib
import runpy
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
_SRC = _ROOT / "src"
sys.path[:0] = [
    str(p) for p in _SRC.rglob("*") if p.is_dir() and p.name != "__pycache__"
]


def _bare_name_collisions():
    """Modules import one another by bare name, so one name in two `src/`
    layers would be resolved by path order alone — which is how a stale copy
    left behind by an incomplete update (a module that moved layer) would
    shadow the current one without any error."""
    seen = {}
    for p in sorted(_SRC.rglob("*.py")):
        if "__pycache__" not in p.parts:
            seen.setdefault(p.name, []).append(str(p.relative_to(_ROOT)))
    return {n: ps for n, ps in seen.items() if len(ps) > 1}


_dupes = _bare_name_collisions()
if _dupes:
    for _n, _ps in sorted(_dupes.items()):
        print(f"DUPLICATE MODULE {_n}: {', '.join(_ps)}", file=sys.stderr)
    sys.exit("run_tests.py: the same module name lives in two src/ layers; "
             "this tree is not a clean copy of a release — remove the stale file(s).")
# tests/ itself: the shared spine-freeness helper (tests/_spine.py) is
# imported by the Step-3 suites.
sys.path.insert(0, str(_ROOT / "tests"))

_STEP_SUITES = ("tests/test_samples.py", "tests/test_cones.py",
                "tests/test_sample_cone_iso.py",
                "tests/test_rg_flows.py", "tests/test_a1an_chain.py",
                "tests/test_dn_chain.py", "tests/test_e_type.py",
                "tests/test_flavoured_fork.py", "tests/test_over_pure.py",
                "tests/test_su2_gauged_chain.py", "tests/test_wild.py",
                "tests/test_bps_flows.py", "tests/test_abe_flows.py",
                "tests/test_skein_flows.py", "tests/test_gn_flows.py")


def _run_cited(jobs: int) -> int:
    """The second tier: every test the paper companion cites
    (`tests/cited.txt`), each in its own process — they were written as
    standalone scripts, not to share one — with every `src/<layer>/` and
    `tests/` on `PYTHONPATH`.  About half an hour serially (measured
    2026-09-29); `--jobs N` runs N at once."""
    import os
    import subprocess
    import time
    from concurrent.futures import ThreadPoolExecutor

    names = [ln.strip() for ln in (_ROOT / "tests" / "cited.txt").read_text().split("\n")
             if ln.strip() and not ln.startswith("#")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(p) for p in sorted(_SRC.iterdir()) if p.is_dir() and p.name != "__pycache__"]
        + [str(_ROOT / "tests"), str(_ROOT)]))    # the root: `import dictionaries.<builder>`

    def one(name):
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, "-S", str(_ROOT / "tests" / name)], cwd=_ROOT,
                               env=env, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:       # a failure of this test, not of the run
            return name, 124, time.time() - t0, "timed out after 1800 s"
        return name, r.returncode, time.time() - t0, (r.stdout + r.stderr)[-2000:]

    if not (_ROOT / "dictionaries" / "n_001.json").exists():
        # test_dictionary_loader reads the seed-closure dictionary, which is
        # built on demand rather than shipped (about a minute).
        print("  building the seed-closure dictionary (dictionaries/build.py) ...", flush=True)
        subprocess.run([sys.executable, "-S", str(_ROOT / "dictionaries" / "build.py")],
                       cwd=_ROOT, env=env, check=True, capture_output=True)

    failed = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for name, rc, dt, tail in pool.map(one, names):
            print(f"  {'PASS' if rc == 0 else 'FAIL'}  {name}  ({dt:.0f} s)", flush=True)
            if rc:
                failed.append(name)
                print("    " + tail.strip().replace("\n", "\n    ")[-1200:])
    print(f"\ncited tests: {len(names) - len(failed)}/{len(names)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    _args = sys.argv[1:]
    _jobs = int(_args[_args.index("--jobs") + 1]) if "--jobs" in _args else 1
    if "--cited-only" not in _args:
        for _test in _STEP_SUITES:
            print(f"\n=== {_test} ===")
            runpy.run_path(str(_ROOT / _test), run_name="__main__")
    if "--cited" in _args or "--cited-only" in _args:
        print("\n=== the tests the paper companion cites ===")
        sys.exit(_run_cited(_jobs))
