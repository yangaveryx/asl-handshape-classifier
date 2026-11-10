from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import torch
from omegaconf import OmegaConf

# Upstream code/data stay in a separate checkout; they are not vendored here.
ROOT = Path(os.environ.get(
    "HANDSHAPE_UPSTREAM_DIR",
    str(Path(__file__).resolve().parent.parent / "Modeling-ASL-Phonology"),
)).expanduser().resolve()
DEFAULT_CHECKPOINT = ROOT / 'pretrained_models/slgcn_top2phonemes.ckpt'
DEFAULT_LABELS = ROOT / 'training_data/wlasl_new.json'


def source_module(name, path, package=False):
    """Load existing modules without running OpenHands' training-heavy __init__s."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def encoder_class():
    # A private namespace preserves graph_utils' relative import without importing
    # openhands.models.encoder (which imports unrelated CNN/training dependencies).
    name = '_handshape_graph'
    graph_dir = ROOT / 'openhands/models/encoder/graph'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(graph_dir)]
        sys.modules[name] = package
    return source_module(name + '.decoupled_gcn', graph_dir / 'decoupled_gcn.py').DecoupledGCN


def label_mapping(path):
    """Match WLASLDataset.read_params: sorted string labels across all instances."""
    content = json.loads(Path(path).read_text())
    labels = sorted({str(instance['Handshape']) for sign in content for instance in sign['instances']})
    if not labels:
        raise ValueError('No handshape labels in metadata')
    return labels


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class _LegacyCheckpointMetadata:
    """Inert placeholder for the old Lightning callback metadata key."""


def read_checkpoint(path):
    # Old Lightning stores a callback class in metadata. Permit only an inert
    # stand-in for that exact global, without importing Lightning or enabling
    # unrestricted pickle execution.
    with torch.serialization.safe_globals([
        (_LegacyCheckpointMetadata, 'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint'),
        (_LegacyCheckpointMetadata, 'pytorch_lightning.callbacks.early_stopping.EarlyStopping'),
    ]):
        return torch.load(path, map_location='cpu', weights_only=True)


def load_model(checkpoint=DEFAULT_CHECKPOINT, metadata=DEFAULT_LABELS):
    labels = label_mapping(metadata)
    state = read_checkpoint(checkpoint)['state_dict']
    encoder_state = {k.removeprefix('model.encoder.'): v for k, v in state.items()
                     if k.startswith('model.encoder.')}
    head_prefix = 'model.decoder.param_clfs.Handshape.'
    head_state = {k.removeprefix(head_prefix): v for k, v in state.items() if k.startswith(head_prefix)}
    if head_state.get('weight', torch.empty(0)).shape != (len(labels), 256):
        raise ValueError('Handshape head and metadata disagree; use the matching training label mapping')
    if any('adapter_module' in k for k in encoder_state):
        raise ValueError('Adapter checkpoints are not supported by this baseline')
    cfg = OmegaConf.load(ROOT / 'gcn_top2_all_test.yaml')
    encoder = encoder_class()(
        in_channels=2, adapters=False, learn_adapter=False,
        graph_args=cfg.model.encoder.params.graph_args, groups=8,
    )
    encoder.load_state_dict(encoder_state, strict=True)
    head = torch.nn.Linear(256, len(labels))
    head.load_state_dict(head_state, strict=True)
    return torch.nn.Sequential(encoder, head).eval(), labels


def extract_video(path):
    """Stream RGB frames through legacy Holistic; retain aligned image XYZ data."""
    import cv2
    import mediapipe as mp

    if not hasattr(mp, 'solutions'):
        raise RuntimeError('Legacy Holistic required: install requirements-handshape.txt with Python 3.12')
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    frames, confidence = [], []
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    try:
        with redirect_stdout(sys.stderr):
            holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=2)
        with holistic:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                result = holistic.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                points = np.zeros((75, 3), dtype=np.float32)
                scores = np.zeros(75, dtype=np.float32)
                for component, start, count in [
                    (result.pose_landmarks, 0, 33),
                    (result.left_hand_landmarks, 33, 21),
                    (result.right_hand_landmarks, 54, 21),
                ]:
                    if component is not None:
                        points[start:start + count] = [[p.x, p.y, p.z] for p in component.landmark]
                        scores[start:start + count] = (
                            [p.visibility for p in component.landmark] if start == 0 else 1
                        )
                frames.append(points)
                confidence.append(scores)
    finally:
        capture.release()
    if not frames:
        raise ValueError('Video contains no decodable frames')
    return np.stack(frames), np.stack(confidence), fps


def signing_region(points):
    """Reuse app velocity/onset logic, keeping every frame inside inclusive bounds."""
    app_dir = ROOT.parent / 'asl_summer26'
    sys.path.insert(0, str(app_dir))
    try:
        from server.keyframe import compute_velocities, find_signing_region
    finally:
        sys.path.pop(0)
    from types import SimpleNamespace
    hands = []
    for frame in points:
        hands.append([
            SimpleNamespace(label=label, landmarks=frame[start:start + 21].reshape(-1).tolist())
            for label, start in [('Left', 33), ('Right', 54)]
            if np.any(frame[start:start + 21])
        ])
    velocity = compute_velocities(points[:, :33].reshape(-1, 99).tolist(), hands)
    return find_signing_region(velocity)


def prepare_input(points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (75, 3) or len(points) == 0:
        raise ValueError('Expected a nonempty (T,75,3) image-coordinate landmark array')
    if not np.isfinite(points).all():
        raise ValueError('Landmarks contain nonfinite coordinates')
    shoulders = points[:, [11, 12], :2]
    if not np.any(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=-1) > 1e-6):
        raise ValueError('No usable shoulder separation; cannot normalize')
    if not np.any(points[:, 33:]):
        raise ValueError('No hand landmarks detected in the selected interval')
    transforms = source_module('_handshape_transforms', ROOT / 'openhands/datasets/pose_transforms.py')
    data = {'frames': torch.from_numpy(points[:, :, :2].copy()).permute(2, 0, 1)}
    data = transforms.PoseSelect(preset='mediapipe_holistic_minimal_27')(data)
    data = transforms.CenterAndScaleNormalize(
        reference_points_preset='shoulder_mediapipe_holistic_minimal_27', scale_factor=1,
    )(data)
    tensor = data['frames'].unsqueeze(0).contiguous()
    if not torch.isfinite(tensor).all():
        raise ValueError('Normalization produced nonfinite coordinates')
    return tensor


def save_landmark_overlays(video, points, confidence, fps, start, end, directory, count=8):
    """Redecode selected images, drawing stored inference landmarks (no new detection)."""
    import cv2

    if count < 1:
        raise ValueError('Overlay frame count must be positive')
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError(f'Overlay directory must be new or empty: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    indices = np.unique(np.linspace(start, end, min(count, end - start + 1)).astype(int))
    wanted = set(indices.tolist())
    transforms = source_module('_handshape_transforms', ROOT / 'openhands/datasets/pose_transforms.py')
    selected = set(transforms.PoseSelect.KEYPOINT_PRESETS['mediapipe_holistic_minimal_27'])
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
             (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
             (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f'Cannot reopen video for overlays: {video}')
    records = []
    try:
        for index in range(end + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Video ended before requested overlay frame {end}')
            if index not in wanted:
                continue
            height, width = frame.shape[:2]
            detected = {}
            for label, base, color in [('Left', 33, (255, 220, 0)), ('Right', 54, (0, 140, 255))]:
                present = bool(confidence[index, base] > 0)
                detected[label.lower()] = present
                if not present:
                    continue  # Never draw zero-filled missing hands at the image origin.
                xy = points[index, base:base + 21, :2]
                # Do not clamp: off-image coordinates should not masquerade as valid border joints.
                pixels = np.rint(xy * [width, height]).astype(int)
                visible = np.isfinite(xy).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                visible &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
                for first, second in edges:
                    if visible[first] and visible[second]:
                        cv2.line(frame, tuple(pixels[first]), tuple(pixels[second]), color, 2, cv2.LINE_AA)
                for joint, pixel in enumerate(pixels):
                    if not visible[joint]:
                        continue
                    position = tuple(pixel)
                    cv2.circle(frame, position, 5 if base + joint in selected else 3,
                               (0, 255, 255) if base + joint in selected else color, -1, cv2.LINE_AA)
                    cv2.putText(frame, str(joint), (int(pixel[0]) + 5, int(pixel[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, str(joint), (int(pixel[0]) + 5, int(pixel[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            timestamp = index / fps if np.isfinite(fps) and fps > 0 else None
            canvas = cv2.copyMakeBorder(frame, 80, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
            time_text = f'{timestamp:.3f}s' if timestamp is not None else 'time unavailable'
            messages = [f'Frame {index} | {time_text} | model interval {start}-{end}',
                        f'Left/cyan: {detected["left"]} | Right/orange: {detected["right"]}',
                        'Yellow dots: hand nodes selected for model; numbers: joints 0-20']
            for line, message in enumerate(messages):
                cv2.putText(canvas, message, (8, 20 + line * 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 1, cv2.LINE_AA)
            filename = f'frame-{index:06d}.png'
            if not cv2.imwrite(str(directory / filename), canvas):
                raise OSError(f'Could not write overlay: {directory / filename}')
            records.append({'frame_index': index, 'timestamp_seconds': timestamp,
                            'file': filename, 'hands_detected': detected})
    finally:
        capture.release()
    manifest = {
        'video': str(Path(video).resolve()),
        'interval': {'start_frame': start, 'end_frame_inclusive': end},
        'landmark_source': 'Stored raw image-coordinate landmarks from this inference run; not redetected.',
        'selected_hand_offsets': sorted(i - 33 for i in selected if 33 <= i < 54),
        'frames': records,
    }
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return {'directory': str(directory.resolve()), 'manifest': 'manifest.json', 'frames': records}


def analyze_video(video, checkpoint=DEFAULT_CHECKPOINT, metadata=DEFAULT_LABELS, trim=False, top_k=5,
                  overlay_dir=None, overlay_frames=8, model_bundle=None):
    # Batch callers reuse a loaded (model, labels) pair; single-video behavior is unchanged.
    model, labels = load_model(checkpoint, metadata) if model_bundle is None else model_bundle
    points, confidence, fps = extract_video(video)
    start, end = signing_region(points) if trim else (0, len(points) - 1)
    selected = points[start:end + 1]
    tensor = prepare_input(selected)
    with torch.inference_mode():
        logits = model(tensor)[0]
        if not torch.isfinite(logits).all():
            raise ValueError('Model produced nonfinite logits')
        probabilities = logits.softmax(dim=-1).tolist()
    ranked = sorted(range(len(labels)), key=lambda i: probabilities[i], reverse=True)
    quality = confidence[start:end + 1]
    warnings = ['Scores are uncalibrated model probabilities, not sign-correctness percentages.',
                'Label ordering is reconstructed from supplied metadata; checkpoint contains no label names.']
    if labels[ranked[0]] == '-1':
        warnings.append('Top category is -1 (unknown/unannotated), not a named handshape.')
    if np.any(np.all(quality[:, [11, 12]] == 0, axis=1)):
        warnings.append('Some frames have missing pose; zero padding is retained to match baseline preprocessing.')
    result = {
        'video': str(Path(video).resolve()),
        'checkpoint': {'path': str(Path(checkpoint).resolve()), 'sha256': sha256(checkpoint)},
        'label_metadata': {'path': str(Path(metadata).resolve()), 'sha256': sha256(metadata)},
        'frames_decoded': len(points), 'fps': fps if np.isfinite(fps) and fps > 0 else None,
        'signing_interval': {'start_frame': start, 'end_frame_inclusive': end, 'trimmed': trim},
        'input_shape': list(tensor.shape),
        'detection': {'left_hand_frame_fraction': float(np.mean(quality[:, 33] > 0)),
                      'right_hand_frame_fraction': float(np.mean(quality[:, 54] > 0))},
        'top_k': [{'index': i, 'label': labels[i], 'probability': probabilities[i]} for i in ranked[:top_k]],
        'distribution': [{'index': i, 'label': label, 'probability': probabilities[i]}
                         for i, label in enumerate(labels)],
        'warnings': warnings,
    }
    if overlay_dir is not None:
        result['overlays'] = save_landmark_overlays(
            video, points, confidence, fps, start, end, overlay_dir, overlay_frames,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--label-metadata', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--trim-signing-region', action='store_true', help='Opt-in heuristic; full clip is the baseline')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--output', type=Path, help='Optional JSON result; otherwise print to stdout')
    parser.add_argument('--overlay-dir', type=Path, help='Save annotated PNGs to a new or empty directory')
    parser.add_argument('--overlay-frames', type=int, default=8,
                        help='Number of evenly spaced overlay frames within the model interval (default: 8)')
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    if args.overlay_frames < 1:
        parser.error('--overlay-frames must be positive')
    try:
        result = analyze_video(args.video, args.checkpoint, args.label_metadata,
                               args.trim_signing_region, args.top_k,
                               overlay_dir=args.overlay_dir, overlay_frames=args.overlay_frames)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        parser.exit(1, f'Handshape inference failed: {error}\n')
    output = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
