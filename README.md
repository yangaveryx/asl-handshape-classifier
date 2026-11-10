# asl-handshape-classifier

Testing handshape phonological model on SEMLEX dataset.

Experiment scripts for evaluating a pretrained handshape classifier on SEM-LEX,
selecting a reproducible c/f/g/w pilot, and inspecting hand landmarks. This is
inference/evaluation tooling, not a training implementation or CL0 integration.

## Contents

- `handshape.py`: one-video inference and optional landmark overlays.
- `batch_handshape.py`: validation-manifest evaluation; model loaded once.
- `test_handshape.py`: existing preprocessing/checkpoint tests.
- `data_processing/select_handshape_videos.sql`: DuckDB metadata selection.
- `data_processing/create_handshape_pilot.py`: cleaning and seeded pilot selection.
- [HANDSHAPE.md](HANDSHAPE.md): setup and run instructions.
- [Upstream dependencies](docs/upstream_dependencies.md): sources, pinned versions,
  required local assets, and optional signing-region integration.
- [Experiment notes](docs/experiment_notes.md): methodology and limitations.

## Layout

Keep the upstream checkout separate:

```text
research/
  asl-handshape-classifier/  # this repository
  Modeling-ASL-Phonology/    # upstream dependency, not included here
  asl_summer26/              # optional: automatic signing-region helper
```

The handshape script defaults to the sibling `Modeling-ASL-Phonology` folder.
Override its location with `HANDSHAPE_UPSTREAM_DIR`. No code is downloaded
implicitly. You must obtain the upstream assets through their authorized source.

Videos, dataset metadata, pilot manifests, model weights, environments and
credentials are intentionally excluded. Git ignores CSVs and common binary/data
formats. Store those locally or on approved cluster storage. This repository
alone cannot run inference until the documented upstream assets are available.
