"""Evaluate a fixed validation manifest without updating the pretrained model."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

import handshape

CATEGORIES = ('c', 'f', 'g', 'w')
CSV_FIELDS = ['video_id', 'word_labels', 'signer_id', 'split', 'expected_handshape',
              'video_path', 'status', 'predicted_handshape', 'correct', 'top_probability',
              'expected_probability', 'expected_rank', 'frames_decoded',
              'left_hand_frame_fraction', 'right_hand_frame_fraction', 'elapsed_seconds', 'error']


def read_manifest(path):
    """Reject accidental training/test input, duplicate IDs and unsupported targets."""
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {'video_id', 'word_labels', 'handshape', 'signer_id', 'split'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Manifest needs columns: {sorted(required)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Manifest contains no videos')
    seen = set()
    for row in rows:
        video_id = row['video_id']
        if not video_id or '/' in video_id or '\\' in video_id or video_id in {'.', '..'}:
            raise ValueError(f'Invalid video ID: {video_id!r}')
        if video_id in seen:
            raise ValueError(f'Duplicate video ID: {video_id}')
        seen.add(video_id)
        if row['split'] != 'val':
            raise ValueError(f'{video_id}: this baseline script accepts val rows only')
        if row['handshape'] not in CATEGORIES:
            raise ValueError(f'{video_id}: expected handshape must be one of {CATEGORIES}')
    return rows


def summarize(records, manifest):
    """Separate classification accuracy from processing coverage and failures."""
    groups = {}
    for category in CATEGORIES:
        requested = sum(r['handshape'] == category for r in manifest)
        attempted = [r for r in records if r['expected_handshape'] == category]
        successful = [r for r in attempted if r['status'] == 'ok']
        correct = sum(r['correct'] for r in successful)
        groups[category] = {
            'requested': requested, 'succeeded': len(successful),
            'failed': sum(r['status'] == 'error' for r in attempted),
            'not_processed': requested - len(attempted), 'correct': correct,
            'accuracy_on_successful': correct / len(successful) if successful else None,
            'predicted_label_counts': dict(Counter(r['predicted_handshape'] for r in successful)),
        }
    successful = [r for r in records if r['status'] == 'ok']
    correct = sum(r['correct'] for r in successful)
    return {
        'requested': len(manifest), 'succeeded': len(successful),
        'failed': sum(r['status'] == 'error' for r in records),
        'not_processed': len(manifest) - len(records), 'correct': correct,
        'coverage': len(successful) / len(manifest),
        'accuracy_on_successful': correct / len(successful) if successful else None,
        'correct_fraction_of_all_requested': correct / len(manifest),
        'per_handshape': groups,
        'failure_ids': [r['video_id'] for r in records if r['status'] == 'error'],
        'not_processed_ids': [r['video_id'] for r in manifest
                              if r['video_id'] not in {x['video_id'] for x in records}],
    }


def run_batch(args):
    rows = read_manifest(args.manifest)
    if args.output_dir.exists():
        raise ValueError(f'Use a new output directory; refusing to overwrite: {args.output_dir}')
    if not args.video_root.is_dir():
        raise ValueError(f'Video root directory not found: {args.video_root}')
    print('Loading pretrained encoder and handshape head once...', file=sys.stderr, flush=True)
    bundle = handshape.load_model(args.checkpoint, args.label_metadata)
    if not set(r['handshape'] for r in rows).issubset(bundle[1]):
        raise ValueError('Manifest handshape labels do not exist in checkpoint label mapping')
    provenance = {
        'manifest': {'path': str(args.manifest.resolve()), 'sha256': handshape.sha256(args.manifest)},
        'checkpoint': {'path': str(args.checkpoint.resolve()), 'sha256': handshape.sha256(args.checkpoint)},
        'label_metadata': {'path': str(args.label_metadata.resolve()), 'sha256': handshape.sha256(args.label_metadata)},
        'script_sha256': handshape.sha256(Path(__file__)),
        'handshape_script_sha256': handshape.sha256(Path(handshape.__file__)),
        'python': platform.python_version(),
        'packages': {name: version(name) for name in ['torch', 'numpy', 'mediapipe', 'omegaconf']},
        'video_root': str(args.video_root.resolve()), 'extension': args.extension,
        'input_policy': 'Full-video, single-video inference; no trimming, overlays or cross-video padding.',
        'output_policy': 'All checkpoint categories compete, including -1; no four-category renormalization.',
        'label_review': 'Expected labels are supplied metadata, not asserted to be visually reviewed.',
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'run_info.json').write_text(json.dumps(provenance, indent=2) + '\n')
    records, interrupted = [], False
    started = time.monotonic()
    with (args.output_dir / 'predictions.csv').open('x', newline='') as csv_handle, \
            (args.output_dir / 'predictions.jsonl').open('x') as json_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_handle.flush()
        try:
            for number, row in enumerate(rows, 1):
                video = args.video_root / row['split'] / f'{row["split"]}-videos' / f'{row["video_id"]}{args.extension}'
                record = {key: row[key] for key in ['video_id', 'word_labels', 'signer_id', 'split']}
                record.update(expected_handshape=row['handshape'], video_path=str(video.resolve()))
                item_start = time.monotonic()
                detail = {}
                try:
                    if not video.is_file():
                        raise FileNotFoundError(f'Video does not exist at expected path: {video}')
                    detail = handshape.analyze_video(
                        video, args.checkpoint, args.label_metadata, model_bundle=bundle,
                    )
                    ranked = sorted(detail['distribution'], key=lambda x: x['probability'], reverse=True)
                    expected_rank, expected = next((i + 1, entry) for i, entry in enumerate(ranked)
                                                   if entry['label'] == row['handshape'])
                    record.update(
                        status='ok', predicted_handshape=ranked[0]['label'],
                        correct=ranked[0]['label'] == row['handshape'],
                        top_probability=ranked[0]['probability'],
                        expected_probability=expected['probability'], expected_rank=expected_rank,
                        frames_decoded=detail['frames_decoded'],
                        **detail['detection'], error='',
                    )
                except Exception as error:
                    record.update(status='error', error=f'{type(error).__name__}: {error}')
                record['elapsed_seconds'] = round(time.monotonic() - item_start, 3)
                writer.writerow(record)
                json_handle.write(json.dumps({'record': record, 'inference': detail or None}, allow_nan=False) + '\n')
                csv_handle.flush()
                json_handle.flush()
                records.append(record)
                print(f'[{number}/{len(rows)}] {row["video_id"]}: {record["status"]} '
                      f'expected={row["handshape"]} predicted={record.get("predicted_handshape", "--")}',
                      file=sys.stderr, flush=True)
        except KeyboardInterrupt:
            interrupted = True
    summary = summarize(records, rows)
    summary.update(interrupted=interrupted, elapsed_seconds=round(time.monotonic() - started, 3),
                   metrics_note='Accuracies are fractions (0-1). accuracy_on_successful excludes processing failures; coverage and correct_fraction_of_all_requested expose incomplete runs.')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))
    return 2 if interrupted or summary['failed'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='Clean pilot_val.csv')
    parser.add_argument('--video-root', type=Path, default=Path('/share/compling/data/semlex'))
    parser.add_argument('--output-dir', type=Path, required=True, help='New output directory')
    parser.add_argument('--checkpoint', type=Path, default=handshape.DEFAULT_CHECKPOINT)
    parser.add_argument('--label-metadata', type=Path, default=handshape.DEFAULT_LABELS)
    parser.add_argument('--extension', choices=['.webm', '.mp4'], default='.webm')
    args = parser.parse_args()
    try:
        return run_batch(args)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        parser.exit(1, f'Batch setup failed: {error}\n')


if __name__ == '__main__':
    sys.exit(main())
