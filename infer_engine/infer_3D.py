from infer_engine._utils import *
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import numpy as np
import os  # Added os
import torch  # Added torch for device check
from collections import defaultdict

# --- MODIFIED: Set Model Weight Download Directory ---
# We must set the cache directory *before* importing MMPoseInferencer.

SCRIPT_DIR = Path(__file__).resolve().parent
# Assume 'model_weights' dir is at the project root (one level above script dir)
MODEL_WEIGHTS_DIR = SCRIPT_DIR.parent / "model_weights"
MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Set MMEngine cache (for mmpose/mmdet models)
os.environ['MMENGINE_CACHE_DIR'] = str(MODEL_WEIGHTS_DIR)

# 2. Set TORCH_HOME (for PyTorch's hub downloads)
#    This is the key fix for the issue you observed.
TORCH_CACHE_DIR = MODEL_WEIGHTS_DIR / "torch_cache"
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ['TORCH_HOME'] = str(TORCH_CACHE_DIR)
# ----------------------------------------------------

from mmpose.apis.inferencers import MMPoseInferencer
from mmpose.apis.inference_3d import collate_pose_sequence



def _flatten_bbox(bbox_val: Any) -> List[float]:
    """Handle possible tuple-wrapped bbox coming from split_instances.

    Returns a flat [x1, y1, x2, y2] list when possible.
    """
    if isinstance(bbox_val, (list, tuple)) and len(bbox_val) == 1 and isinstance(bbox_val[0], (list, tuple)):
        return list(bbox_val[0])
    if isinstance(bbox_val, (list, tuple)):
        return list(bbox_val)
    return bbox_val


def _normalize_predictions(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize result['predictions'] into a flat list of instance dicts.

    Each instance dict contains keys: keypoints, keypoint_scores, and
    optionally bbox, bbox_score.
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
        if 'bbox' in inst:
            inst_out['bbox'] = _flatten_bbox(inst['bbox'])
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
                inst['bbox'] = _flatten_bbox(inst['bbox'])
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
    kps_list= []
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


def pred_vid(video,vis_dir,pred_dir) -> None:
    # Initialize inferencer with VitPose Small model
    # Option 1: Use model config path (recommended)
    inferencer = MMPoseInferencer(
        pose3d="human3d",
        # Or use full config path:
        # pose2d='configs/body_2d_keypoint/topdown_heatmap/coco/vitpose_small_coco_256x192.py',
        # device='cuda:0',  # specify device if needed
        show_progress=True,
    )

    all_frames: List[Dict[str, Any]] = []

    # Run inference; also save visualizations and predictions to disk
    # pred_out_dir will produce a single JSON for the video upon finalize
    for frame_idx, res in enumerate(
        inferencer(
            str(video),
            vis_out_dir=str(vis_dir),
            pred_out_dir=str(pred_dir),
            #draw_bbox=True,
            #kpt_thr=0.3,
        )
    ):
        instances = _normalize_predictions(res)
        all_frames.append({'frame_id': frame_idx, 'instances': instances})

    # Read back the saved JSON to demonstrate disk I/O
    saved_frames = load_saved_video_predictions(pred_dir, video)

    # Transform to numpy for downstream processing
    kps_per_frame, scores_per_frame = frames_to_arrays(all_frames)

    # Small human-readable summary
    total_frames = len(all_frames)
    total_people = sum(len(f['instances']) for f in all_frames)
    print(f'Processed {total_frames} frames, detected {total_people} person-instances.')
    print(f'Saved visualizations to: {vis_dir}')
    print(f'Saved predictions JSON to: {pred_dir / (video.stem + ".json")}')

    if total_frames:
        first = all_frames[0]
        print(f"Frame 0 instances: {len(first['instances'])}")
        if first['instances']:
            kp = first['instances'][0]['keypoints']
            print(f'First instance has {len(kp)} keypoints. Example: {kp[:3]} ...')
            # Example: numpy shapes for first frame
            print('First frame arrays:', kps_per_frame[0].shape, scores_per_frame[0].shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=""
                    , help="Container path for input video")
    ap.add_argument("--outdir", default="", help="Container path for outputs")

    args = ap.parse_args()
    if not args.video:
        args.video = str(Path(__file__).resolve().parent.parent/"tests/data/posetrack18/videos/000001_mpiinew_test/000001_mpiinew_test.mp4")
    if not args.outdir:
        args.outdir = str(Path(__file__).resolve().parent.parent/"outputs")
     # Prepare output directories
    out_dir = Path(args.outdir)
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Input video not found: {video_path}")
        sys.exit(1)

    vis_dir = out_dir / "visualizations"
    pred_dir = out_dir / "predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_vid(video_path,vis_dir,pred_dir)



if __name__ == "__main__":
    main()

