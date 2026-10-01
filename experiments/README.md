# Research probes

The five modules here are research probes of the source repository.  They ship
because the battery's adapters (`battery/checks_rg.py`) import them, so that the
battery rows built on them can be re-run from this repository.  They are
development records rather than part of the library: nothing under `src/`
imports them, and their docstrings keep the source repository's working
vocabulary.

| module | in its own opening words |
|---|---|
| `pbw_e_group.py` | Exact arithmetic in the pro-nilpotent group `G` and its subgroup `E`. |
| `pbw_controls.py` | Controls for `experiments/pbw_e_group.py` — run these before any sweep. |
| `sw_flow_conjectures.py` | The remaining Seiberg-Witten RG flow conjectures, on the BPS-quiver dictionary. |
| `sw_factored_flows.py` | The factored-RG-flow conjecture of the Seiberg-Witten RG flows section. |
| `upper_cluster_box_certificate.py` | A finite test of the second half of conj:upper on a box of charges. |

They are imported as `experiments.<name>` and run from the repository root:

    PYTHONPATH=. python3 -m experiments.pbw_controls
