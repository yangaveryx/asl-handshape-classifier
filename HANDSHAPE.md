# Running the handshape experiments

## Setup

First obtain the separate upstream checkout and required assets described in
[upstream_dependencies.md](docs/upstream_dependencies.md). Expected layout:

```text
research/asl-handshape-classifier
research/Modeling-ASL-Phonology
```

For a different upstream location:

```sh
export HANDSHAPE_UPSTREAM_DIR="/path/to/Modeling-ASL-Phonology"
```

Python 3.12 was used for initial verification; it is a tested version, not an
inherent requirement of the script. On Unicorn, create/use a Conda environment
inside a compute allocation. For an existing Miniforge installation:

```sh
source ~/miniforge3/etc/profile.d/conda.sh
conda create -n handshape python=3.12 pip -y
conda activate handshape
cd ~/asl-handshape-classifier
python -m pip install -r requirements-handshape.txt
```

Do not copy a Mac Python environment to the Linux cluster. Do not run model
processing or environment installation on the login node. Request a compute
allocation using the cluster's current instructions. Inference currently uses
CPU; requesting a GPU does not automatically move the model to it.

## One video

```sh
python handshape.py --video "/full/path/to/video.webm" --output clip-result.json
```

MediaPipe may download its heavy pose model on first use. Top-k and all category
scores are returned. Input uses all frames unless `--trim-signing-region` is
requested; that option requires the separately documented application helper.
`trimmed: true` means the heuristic was enabled, not that it removed frames.

## Landmark overlays

```sh
python handshape.py --video "/full/path/to/video.webm"   --output clip-result.json --overlay-dir clip-overlays --overlay-frames 8
```

Use a new or empty overlay directory. Images show stored inference landmarks,
not a second MediaPipe detection. All 21 joints are drawn per hand; yellow dots
highlight the ten selected for the model. Left is cyan, right orange. Missing
hands are not drawn. The manifest records source frame indices and detection
status; increase the count to inspect more frames.

## Pilot preparation

DuckDB (separate optional tool) can run the SQL from this repository root after
placing an authorized metadata copy at `data_processing/semlex.csv`:

```sh
duckdb :memory: < data_processing/select_handshape_videos.sql
```

The SQL displays counts and matching records; it does not export CSVs. Save its
selection with columns `video_id,word,handshape,signer_id,split` as
`data_processing/handshape_candidates.csv`, or supply your own candidate path.
The cleaning script uses only the Python standard library:

```sh
python data_processing/create_handshape_pilot.py --seed 42
```

Outputs go into a new `data_processing/handshape_pilot` directory. Keep those
lists fixed between model comparisons. Source metadata and generated lists are
ignored by Git. Existence and annotation review are separate steps.

## Batch validation

With an existing validation manifest and upstream assets, run in a compute job:

```sh
python batch_handshape.py   --manifest /path/to/pilot_val.csv   --video-root /share/compling/data/semlex   --output-dir runs/baseline-val-run1
```

The model loads once. Paths are `VIDEO_ROOT/val/val-videos/VIDEO_ID.webm`.
No training, trimming, overlays or four-category renormalization is performed.
Validation-only manifests are checked for duplicate IDs. All errors remain
recorded. Outputs are `predictions.csv`, `predictions.jsonl`, `summary.json`, and
`run_info.json`. Check coverage before comparing accuracy; accuracies are
fractions, not percentages. Use a new output directory each run.

## Tests and status

```sh
python -m unittest test_handshape -v
```

These existing tests need the upstream assets; the signing-region test also
needs the application helper. Initial checkpoint/preprocessing tests passed in
the original workspace. Single-video runs were subsequently performed on
Unicorn by the researcher. Batch/overlay additions have not been comprehensively
execution-tested, and this extracted repository layout has not been run yet.
