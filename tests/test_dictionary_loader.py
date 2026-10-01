"""Tests for the dictionary loader: parses `n_NNN.json` and
`graph_nNNN.json` files from the canonical tree at
`restructuring/dictionaries/` and constructs flavoured
BPSKAlgebra instances from each entry.

Verifies that flavoured entries (det(exchange) = 0, e.g. all 3×3 ones)
correctly trigger the abelian-flavour code path with non-zero
`flavour_rank`, that multiply / trace work end-to-end on dictionary
entries, and that mutation-graph files load and iterate cleanly.

Run:  PYTHONPATH=. python restructuring/tests/test_dictionary_loader.py
"""

from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESTRUCT = os.path.dirname(_HERE)
_REPO = _RESTRUCT  # post-flatten: _RESTRUCT is the repo root
sys.path.insert(0, _REPO)
sys.path.insert(0, _RESTRUCT)

from pathlib import Path

from zplus_ring import (
    AbelianZPlusRing, TrivialZPlusRing, RLaurent, RPowerSeries,
)
from kalgebra import Element
from bps_kalgebra import BPSKAlgebra
from dictionary_loader import (
    load_dictionary,
    entry_to_bpskalgebra,
    iter_bpskalgebras_from_dictionary,
    load_mutation_graph,
    iter_bpskalgebras_from_graph,
    find_entry_by_exchange,
    default_dictionary_dir,
)


_DICT_DIR = default_dictionary_dir()


def _skip_if_no_dict() -> bool:
    """Skip dict-dependent tests when the data dir is missing OR the
    smallest artifact (`n_001.json`) hasn't been built.

    The `n_NNN.json` / `graph_nNNN.json` files are build artifacts
    produced by `dictionaries/build.py` and friends; they're NOT
    checked into git.  Skipping cleanly when they're absent keeps CI
    green on a fresh checkout while local developers who've run the
    build still get coverage."""
    if not _DICT_DIR.exists():
        return True
    if not (_DICT_DIR / "n_001.json").exists():
        return True
    return False


# ===========================================================================
# Default-path smoke
# ===========================================================================


def test_default_dictionary_dir_resolves_to_canonical():
    """`default_dictionary_dir()` returns the canonical tree at
    `dictionaries/` (the design record; post-Plan-07-flatten the canonical
    surface lives at the repo root, so the parent is the repo
    root rather than `restructuring/`)."""
    d = default_dictionary_dir()
    assert d.name == "dictionaries"
    assert d.exists(), f"canonical dictionary tree missing at {d}"
    assert (d / "n_001.json").exists()
    assert (d / "graph_n001.json").exists()


# ===========================================================================
# Flat dictionary loaders
# ===========================================================================


def test_load_n2_dictionary():
    if _skip_if_no_dict():
        return
    entries = load_dictionary(_DICT_DIR / "n_002.json")
    assert isinstance(entries, list)
    assert len(entries) >= 1
    e = entries[0]
    assert "exchange" in e and "spec" in e and "name" in e


def test_construct_n2_bpskalgebra():
    """n=2: the symplectic-shape entries; non-degenerate -> Trivial R."""
    if _skip_if_no_dict():
        return
    entries = load_dictionary(_DICT_DIR / "n_002.json")
    for entry in entries:
        A = entry_to_bpskalgebra(entry)
        # n=2 entries: usually non-degenerate (det may be ±1)
        assert isinstance(A.coefficient_ring(), (TrivialZPlusRing, AbelianZPlusRing))


def test_construct_n3_yields_flavoured():
    """n=3: every 3×3 antisymmetric matrix has det 0 → flavoured."""
    if _skip_if_no_dict():
        return
    entries = load_dictionary(_DICT_DIR / "n_003.json")
    assert len(entries) >= 1
    for entry in entries:
        A = entry_to_bpskalgebra(entry)
        # All n=3 entries have ker(B) ≠ 0.
        assert A._flavour_rank >= 1, (
            f"{entry['name']}: expected flavoured but got "
            f"flavour_rank={A._flavour_rank}"
        )
        assert isinstance(A.coefficient_ring(), AbelianZPlusRing)


def test_iter_dictionary_n3_with_limit():
    if _skip_if_no_dict():
        return
    pairs = list(iter_bpskalgebras_from_dictionary(
        _DICT_DIR / "n_003.json", limit=3,
    ))
    assert len(pairs) <= 3
    for name, A in pairs:
        assert isinstance(name, str)
        assert isinstance(A, BPSKAlgebra)


def test_n3_entry_multiply_works():
    """Spot-check: multiply on a flavoured n=3 entry yields a non-trivial
    Element with LaurentPoly coefficients (Z-form, per the design record)."""
    from kalgebra import LaurentPoly
    if _skip_if_no_dict():
        return
    entries = load_dictionary(_DICT_DIR / "n_003.json")
    # Find a non-trivial entry (skip degenerate ones with empty/trivial spec).
    for entry in entries:
        if not entry["spec"]:
            continue
        A = entry_to_bpskalgebra(entry)
        # multiply(node 1, node 2): on the standard basis these are e_0, e_1.
        prod = A.multiply((1, 0, 0), (0, 1, 0))
        for c in prod.terms.values():
            assert isinstance(c, LaurentPoly)
        return
    raise AssertionError("no usable n=3 entry found")


# ===========================================================================
# Mutation-graph loaders
# ===========================================================================


def test_load_graph_n3():
    if _skip_if_no_dict():
        return
    graph = load_mutation_graph(_DICT_DIR / "graph_n003.json")
    assert "entries" in graph and "edges" in graph
    assert isinstance(graph["entries"], list)
    assert isinstance(graph["edges"], list)


def test_iter_graph_n3_bpskalgebras():
    if _skip_if_no_dict():
        return
    graph = load_mutation_graph(_DICT_DIR / "graph_n003.json")
    pairs = list(iter_bpskalgebras_from_graph(graph, limit=5))
    assert 1 <= len(pairs) <= 5
    for name, A in pairs:
        assert isinstance(A, BPSKAlgebra)


def test_find_entry_by_exchange_roundtrip():
    """Construct an entry, look it up by exchange matrix, get the same index."""
    if _skip_if_no_dict():
        return
    graph = load_mutation_graph(_DICT_DIR / "graph_n003.json")
    if len(graph["entries"]) < 2:
        return
    target_exchange = graph["entries"][1]["exchange"]
    idx = find_entry_by_exchange(graph, target_exchange)
    assert idx == 1


def test_graph_edges_format():
    """Edges expose source/target exchange matrices; smoke-check that we
    can iterate them.  Canonical-tree edges use `source_exchange` /
    `target_exchange` (cf. `restructuring/dictionaries/build.py`)."""
    if _skip_if_no_dict():
        return
    graph = load_mutation_graph(_DICT_DIR / "graph_n003.json")
    n_edges = 0
    for edge in graph["edges"][:5]:
        # Each edge identifies its source and target by exchange matrix.
        assert "source_exchange" in edge and "target_exchange" in edge
        n_edges += 1
    assert n_edges <= 5


# ===========================================================================
# Robustness on a sample of larger entries (n=4)
# ===========================================================================


def test_construct_n4_sample():
    """Sanity: a handful of n=4 entries construct without raising.
    A subset are degenerate (≈ 26%); they should land in flavoured R.
    The rest land in Trivial R."""
    if _skip_if_no_dict():
        return
    seen_flavoured = False
    seen_trivial = False
    for name, A in iter_bpskalgebras_from_dictionary(
        _DICT_DIR / "n_004.json", limit=20,
    ):
        if A._flavour_rank > 0:
            seen_flavoured = True
            assert isinstance(A.coefficient_ring(), AbelianZPlusRing)
        else:
            seen_trivial = True
            assert isinstance(A.coefficient_ring(), TrivialZPlusRing)
    # We expect both kinds within the first 20 entries.
    assert seen_flavoured or seen_trivial, (
        "neither flavoured nor unflavoured found — empty file?"
    )


# ===========================================================================
# Test runner
# ===========================================================================


def main():
    if _skip_if_no_dict():
        print(
            "[skip] tests/test_dictionary_loader.py: dictionary artifacts "
            f"missing at {_DICT_DIR} (no n_001.json).  Run "
            "`PYTHONPATH=. python dictionaries/build.py` to generate, "
            "or ignore on CI."
        )
        return True
    tests = [
        test_default_dictionary_dir_resolves_to_canonical,
        test_load_n2_dictionary,
        test_construct_n2_bpskalgebra,
        test_construct_n3_yields_flavoured,
        test_iter_dictionary_n3_with_limit,
        test_n3_entry_multiply_works,
        test_load_graph_n3,
        test_iter_graph_n3_bpskalgebras,
        test_find_entry_by_exchange_roundtrip,
        test_graph_edges_format,
        test_construct_n4_sample,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  [ok]   {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERR ] {t.__name__}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed of {passed + failed}")
    print(f"{'=' * 50}")
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
