# Scripts

Scripts the paper companion cites or its cited tests call.  Run them from the
repository root with `PYTHONPATH=.`.

| script | what it does |
|---|---|
| `certify_directional_on_dictionary.py` | certifies the directional node-drop `RGKAlgebra` flows in bulk over the BPS-quiver dictionary |
| `atlas_catalogue_a1d.py` | the `BPSAtlas` cone test over the `[A₁,Dₙ]` theories, through a flavour rebase onto the abelian Cartan a `BPSKAlgebra` chart sees (called by `tests/test_bps_atlas_a1d.py`) |
| `atlas_catalogue_gauged_ad.py` | the `BPSAtlas` cone test over the `U(1)`-gauged Argyres–Douglas family, through flow composition (called by `tests/test_bps_atlas_gauged_ad.py`) |
