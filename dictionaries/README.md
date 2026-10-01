# BPS-quiver dictionaries

Each shipped tier is a directory of gzipped, self-describing shards (quiver,
spectrum-generator sequence, certification) with a `manifest.json` and a
`README.md` of its own.  They are read through `dictionary_loader`
(`src/bps/`).

| directory | contents | reader |
|---|---|---|
| `enumerated/` | every strongly connected BPS quiver of total arrow weight at most 12, up to node permutation, with one accepted spectrum-generator sequence or the crystalline certification; every other quiver is answered by composing its strongly connected components | `lookup_enumerated(exchange)`, `iter_enumerated_entries()` |
| `flavoured/`, `flavoured_su3/`, … `flavoured_su2su2su2/` | the refinements by a manifest flavour symmetry: the quivers carrying a node orbit of size `N` (`N` identical, interchangeable nodes), or a unique family of such orbits for the product tiers, each with a covariant spectrum-generator sequence or the covariant crystalline certification — ten tiers, from `SU(2)` to `SU(2)³` | `lookup_flavoured(exchange, orbit_size=N)` (`orbit_size` a tuple for a product tier) |

The builders:

- `build_enumerated.py`, `build_flavoured.py` — the builders of the two shipped
  families;
- `build.py` — the seed-closure dictionary (`n_NNN.json`, about 75 s), which is
  **not shipped**: build it with `PYTHONPATH=. python3 dictionaries/build.py`
  (`python3 run_tests.py --cited` does so when it is missing);
- `build_fingerprint.py` — the `I_{id,id}(𝖖)` fingerprints of the seed-closure
  entries.
