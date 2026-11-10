"""Clean handshape candidates and select a deterministic, diversity-first pilot.

Run from any directory: python3 data_processing/create_handshape_pilot.py
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATEGORIES = ('c', 'f', 'g', 'w')
MISSING = {'', '-1', 'NA', 'nan', 'None'}
FIELDS = ['video_id', 'word_labels', 'handshape', 'signer_id', 'split']


def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def fingerprint(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write_csv(path, fields, rows):
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_group(records, requested, rng):
    remaining = sorted(records, key=lambda row: row['video_id'])
    rng.shuffle(remaining)
    signer_uses, word_uses = Counter(), Counter()
    chosen = []
    while remaining and len(chosen) < requested:
        # prioritizes least-used signers, then least-used word labels
        index = min(range(len(remaining)), key=lambda i: (
            signer_uses[remaining[i]['signer_id']],
            sum(word_uses[w] for w in json.loads(remaining[i]['word_labels']))
            / len(json.loads(remaining[i]['word_labels'])),
        ))
        row = remaining.pop(index)
        chosen.append(row)
        signer_uses[row['signer_id']] += 1
        word_uses.update(json.loads(row['word_labels']))
    return chosen


def create_pilot(candidates_path, metadata_path, output, seed=42, train_count=20, val_count=10):
    if train_count < 1 or val_count < 1:
        raise ValueError('Requested per-category counts must be positive')
    if output.exists():
        raise ValueError(f'Refusing to overwrite existing output directory: {output}')
    candidates = read_rows(candidates_path)
    metadata = read_rows(metadata_path)
    groups, full = defaultdict(list), defaultdict(list)
    for row in candidates:
        groups[row['video_id']].append(row)
    for row in metadata:
        full[row['video_id']].append(row)
    clean, exclusions, duplicate_audit = [], [], []
    for video_id, rows in sorted(groups.items()):
        source = full.get(video_id, [])
        reasons = []
        splits = sorted({r['split'].strip() for r in source})
        shapes = sorted({r['Handshape'].strip() for r in source if r['Handshape'].strip() not in MISSING})
        signers = sorted({r['signer_id'].strip() for r in source if r['signer_id'].strip() not in MISSING})
        if not source:
            reasons.append('missing_from_full_metadata')
        if len(splits) > 1:
            reasons.append('multiple_splits')
        if len(shapes) > 1:
            reasons.append('conflicting_handshape')
        if len(signers) > 1:
            reasons.append('conflicting_signer')
        if source and (not splits or splits[0] not in {'train', 'val', 'test'}):
            reasons.append('invalid_split')
        if source and not signers:
            reasons.append('missing_signer')
        if source and (len(shapes) == 0 or any(s not in CATEGORIES for s in shapes)):
            reasons.append('missing_or_out_of_scope_handshape')
        # Verify the candidate rows are genuine projections of labeled ASL-LEX rows.
        tuples = {(r['label'], r['Handshape'], r['signer_id'], r['split'])
                  for r in source if r['label_type'] == 'asllex'}
        if any((r['word'], r['handshape'], r['signer_id'], r['split']) not in tuples for r in rows):
            reasons.append('candidate_metadata_mismatch')
        words = sorted({r['word'].strip() for r in rows if r['word'].strip()})
        if not words:
            reasons.append('missing_word_label')
        if reasons:
            exclusions.append({
                'video_id': video_id, 'reasons': ';'.join(reasons),
                'splits': json.dumps(splits), 'handshapes': json.dumps(shapes),
                'signer_ids': json.dumps(signers), 'candidate_row_count': len(rows),
                'full_metadata_row_count': len(source),
            })
            continue
        clean.append({'video_id': video_id, 'word_labels': json.dumps(words),
                      'handshape': shapes[0], 'signer_id': signers[0], 'split': splits[0]})
        if len(rows) > 1:
            duplicate_audit.append({'video_id': video_id, 'original_candidate_rows': len(rows),
                                    'rows_collapsed': len(rows) - 1, 'word_labels': json.dumps(words)})
    rng = random.Random(seed)
    selected = {'train': [], 'val': []}
    group_counts = []
    for split, requested in [('train', train_count), ('val', val_count)]:
        for category in CATEGORIES:
            pool = [r for r in clean if r['split'] == split and r['handshape'] == category]
            picks = select_group(pool, requested, rng)
            selected[split].extend(picks)
            group_counts.append({'split': split, 'handshape': category, 'eligible': len(pool),
                                 'requested': requested, 'selected': len(picks),
                                 'shortfall': max(0, requested - len(picks)),
                                 'distinct_signers': len({r['signer_id'] for r in picks}),
                                 'distinct_word_labels': len({w for r in picks for w in json.loads(r['word_labels'])})})
    train_ids = {r['video_id'] for r in selected['train']}
    val_ids = {r['video_id'] for r in selected['val']}
    if train_ids & val_ids:
        raise ValueError('Selection contains train/validation video overlap')
    summary = {
        'seed': seed, 'algorithm_version': 1,
        'selection_method': 'Groups in train/val then c/f/g/w order; sort IDs, seeded shuffle; greedily prefer least-used signer, then least-used mean word-label frequency; seeded order breaks ties.',
        'sources': {'candidates': fingerprint(candidates_path), 'full_metadata': fingerprint(metadata_path)},
        'candidate_rows': len(candidates), 'candidate_unique_video_ids': len(groups),
        'excluded_unique_video_ids': len(exclusions),
        'exclusion_reason_counts': dict(Counter(reason for r in exclusions for reason in r['reasons'].split(';'))),
        'clean_unique_video_ids': len(clean),
        'duplicate_rows_collapsed_among_retained_ids': sum(r['rows_collapsed'] for r in duplicate_audit),
        'groups': group_counts,
        'train_val_video_overlap': sorted(train_ids & val_ids),
        'train_val_signer_overlap': sorted({r['signer_id'] for r in selected['train']} & {r['signer_id'] for r in selected['val']}),
        'video_files_checked': False, 'visually_reviewed': False,
        'test_policy': 'Test records retained only in clean inventory; none selected for pilot.',
        'limitations': ['Metadata audit does not detect identical recordings with different IDs.',
                        'Word labels are alternative annotations, not necessarily separate signs.',
                        'Preserved source splits do not guarantee unseen-signer evaluation.',
                        'Availability and annotation review must precede training.'],
    }
    output.mkdir(parents=True)
    write_csv(output / 'handshape_candidates_clean.csv', FIELDS, clean)
    for split in selected:
        write_csv(output / f'pilot_{split}.csv', FIELDS, selected[split])
    write_csv(output / 'pilot_exclusions.csv',
              ['video_id', 'reasons', 'splits', 'handshapes', 'signer_ids', 'candidate_row_count', 'full_metadata_row_count'], exclusions)
    write_csv(output / 'pilot_duplicate_collapses.csv',
              ['video_id', 'original_candidate_rows', 'rows_collapsed', 'word_labels'], duplicate_audit)
    (output / 'pilot_selection_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, default=HERE / 'handshape_candidates.csv')
    parser.add_argument('--metadata', type=Path, default=HERE / 'semlex.csv')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'handshape_pilot')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-per-category', type=int, default=20)
    parser.add_argument('--val-per-category', type=int, default=10)
    args = parser.parse_args()
    create_pilot(args.candidates, args.metadata, args.output_dir, args.seed,
                 args.train_per_category, args.val_per_category)
