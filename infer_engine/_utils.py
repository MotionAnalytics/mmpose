import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

def flatten_bbox(bbox_val: Any) -> List[float]:
    """Handle possible tuple-wrapped bbox coming from split_instances.

    Returns a flat [x1, y1, x2, y2] list when possible.
    """
    if isinstance(bbox_val, (list, tuple)) and len(bbox_val) == 1 and isinstance(bbox_val[0], (list, tuple)):
        return list(bbox_val[0])
    if isinstance(bbox_val, (list, tuple)):
        return list(bbox_val)
    return bbox_val


def normalize_predictions(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize result['predictions'] into a flat list of instance dicts.

    Each instance dict contains keys: keypoints, keypoint_scores, and
    optionally bbox, bbox_score, track_id.
    """
    preds = res.get('predictions', [])
    if not preds:
        return []
    # For top-down, preds is typically a list with one element: list[instances]
    first = preds[0]
    if isinstance(first, (list, tuple)):
        instances = list(first)
    elif isinstance(first, dict):
        # single instance case
        instances = list(preds)  # type: ignore[assignment]
    else:
        instances = []
    # Coerce numeric types for bbox_score if present
    norm_instances = []
    for inst in instances:
        inst_out: Dict[str, Any] = {
            'keypoints': inst.get('keypoints', []),
            'keypoint_scores': inst.get('keypoint_scores', []),
        }

        # <--- MODIFIED: Capture the tracking ID ---
        # When processing a video, the inferencer will assign a consistent
        # 'track_id' (or 'instance_id') to each person.
        if 'track_id' in inst:
            inst_out['track_id'] = inst['track_id']
        elif 'instance_id' in inst:
            inst_out['track_id'] = inst['instance_id']
        # ------------------------------------------

        if 'bbox' in inst:
            inst_out['bbox'] = flatten_bbox(inst['bbox'])
        if 'bbox_score' in inst:
            try:
                inst_out['bbox_score'] = float(inst['bbox_score'])
            except Exception:
                inst_out['bbox_score'] = inst['bbox_score']
        norm_instances.append(inst_out)
    return norm_instances


def load_saved_video_predictions(predictions_dir: Path, video_path: Path) -> List[Dict[str, Any]]:
    """Load predictions JSON produced by pred_out_dir for a given video.

    Returns a list of dicts: [{frame_id: int, instances: [..]}, ...]
    where each instance matches the normalized format.
    """
    fname = video_path.stem + '.json'
    json_path = predictions_dir / fname
    if not json_path.exists():
        raise FileNotFoundError(f'Predictions JSON not found: {json_path}')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Normalize bbox wrapping if needed
    for frame in data:
        for inst in frame.get('instances', []):
            if 'bbox' in inst:
                inst['bbox'] = flatten_bbox(inst['bbox'])
            if 'bbox_score' in inst:
                try:
                    inst['bbox_score'] = float(inst['bbox_score'])
                except Exception:
                    pass
    return data


def frames_to_arrays(frames: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert frames list to per-frame numpy arrays (keypoints, scores).

    Returns two lists of arrays:
    - keypoints_per_frame: list of (N, K, 2)
    - scores_per_frame: list of (N, K)
    If a frame has no instances, returns empty arrays with shape (0, 0, 2) and (0, 0).
    """
    kps_shape = None
    kps_list = []
    scores_list = []
    for frame in frames:
        instances = frame.get('instances', [])
        if not instances:
            kps_list.append(None)
            scores_list.append(np.zeros((0, 0), dtype=np.float32))
            continue
        # Assume all instances in a frame share the same K
        kps = np.array([inst['keypoints'] for inst in instances], dtype=np.float32)  # (N, K, 2)
        scs = np.array([inst.get('keypoint_scores', []) for inst in instances], dtype=np.float32)  # (N, K)
        kps_list.append(kps)
        scores_list.append(scs)
    # check whether all kps have the same shape (N,K,2 \ 3)
    for kps in kps_list:
        if kps is not None:
            if kps_shape is None:
                kps_shape = kps.shape
            else:
                if kps.shape != kps_shape:
                    kps_shape = None
                    break
    # if all kps have the same shape, convert None to zeros
    if kps_shape is not None:
        for i in range(len(kps_list)):
            if kps_list[i] is None:
                kps_list[i] = np.zeros((0, kps_shape[1], kps_shape[2]), dtype=np.float32)
    return kps_list, scores_list