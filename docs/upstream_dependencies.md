# Upstream dependencies and attribution

## Modeling-ASL-Phonology

- Source: https://github.com/leekezar/Modeling-ASL-Phonology
- Local base commit: `26a7549c811e66f70cf3b018494fc73fc6c78d3d`.
- Paper: Kezar, Thomason and Sehyr, *Improving Sign Recognition with Phonology*
  (2023), https://arxiv.org/abs/2302.05759 .
- OpenHands is the underlying framework; see the upstream README for attribution.

The handshape scripts reuse the upstream GCN encoder and graph utilities, pose
transforms, top2 graph configuration, checkpoint and category metadata. This
repository does not contain those upstream implementations or assets.

Clone/check out the source in a sibling directory, or set
`HANDSHAPE_UPSTREAM_DIR` to an existing authorized checkout. Pin the commit above
for the baseline. Required paths within it include:

```text
openhands/models/encoder/graph/decoupled_gcn.py
openhands/models/encoder/graph/graph_utils.py
openhands/datasets/pose_transforms.py
gcn_top2_all_test.yaml
pretrained_models/slgcn_top2phonemes.ckpt
training_data/wlasl_new.json
```

The checkpoint is a two-channel, 27-node, eight-group encoder plus a 50-category
handshape head. Category order is reconstructed from sorted string Handshape
values in `wlasl_new.json`; checkpoint metadata does not embed the label names.
The size and supplied mapping rule agree, but original training provenance is
not independently established. Preserve `-1` at its original index.

## Application helper (optional)

- Source: https://github.com/ashlynwinship/asl_summer26
- Local base commit: `3aab4347b950a065232e347796aafc83f913da3d`.

Full-video inference and batch evaluation do not need this application.
`--trim-signing-region` additionally expects the sibling `asl_summer26/server`
folder with `keyframe.py` and `keyframe_fps.py`. Our local helper modification
makes `DetectedHand` a type-only import so it does not load the API/CL0:

```python
from __future__ import annotations  # at the start of keyframe.py
from typing import TYPE_CHECKING
# Replace the unconditional `from server.main import DetectedHand` with:
if TYPE_CHECKING:
    from server.main import DetectedHand
```

This helper adjustment is documented rather than copying application code here.
The existing signing-region unit test requires that helper setup.

## Local-only assets

SEM-LEX recordings are accessed on Unicorn under `/share/compling/data/semlex`.
The metadata CSV, selections and restricted recordings must be obtained using
approved dataset access. Availability in cluster storage does not imply public
redistribution permission. Check upstream/dataset terms before sharing assets.

No `bert.py`, hardcoded credentials, CL0 weights or third-party Git histories
were copied into this repository.
