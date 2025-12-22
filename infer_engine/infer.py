from infer_engine._utils import *
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import numpy as np
import os
import torch
import math

# --- Setup Cache Directories ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_WEIGHTS_DIR = SCRIPT_DIR.parent / "model_weights"
MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ['MMENGINE_CACHE_DIR'] = str(MODEL_WEIGHTS_DIR)

TORCH_CACHE_DIR = MODEL_WEIGHTS_DIR / "torch_cache"
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ['TORCH_HOME'] = str(TORCH_CACHE_DIR)
# --------------------------------

from mmpose.apis.inferencers import MMPoseInferencer

models_3d = ["human3d"]
models_2d = ["vitpose", "vitpose_swimming"]

model_map = {
    "vitpose_swimming": (
        MODEL_WEIGHTS_DIR / "swimming_vitpose" / "td-hm_ViTPose-base_8xb64-210e_coco-256x192" / "td-hm_ViTPose-base_8xb64-210e_coco-256x192.py",
        MODEL_WEIGHTS_DIR / "swimming_vitpose" / "td-hm_ViTPose-base_8xb64-210e_coco-256x192" / "best_coco_AP_epoch_18.pth"
    ),
}


def load_custom_bboxes(bbox_path: str) -> List[Any]:
    """
    Loads bounding boxes from a JSON file.
    """
    if not bbox_path:
        return None

    path = Path(bbox_path)
    if not path.exists():
        raise FileNotFoundError(f"BBox file not found: {bbox_path}")

    print(f"Loading custom bounding boxes from: {bbox_path}")
    with open(path, 'r') as f:
        data = json.load(f)

    return data


def clean_and_interpolate_bboxes(bboxes: List[List[List[float]]]) -> List[List[List[float]]]:
    """
    Replaces NaN values in bounding boxes with valid coordinates via linear interpolation.
    Also ensures boxes have a 5th element (score=1.0) for MMPose compatibility.
    """
    print("Cleaning, interpolating, and scoring bounding boxes...")

    flat_bboxes = []
    # 1. Flatten and Handle NaNs
    for frame_boxes in bboxes:
        if not frame_boxes or len(frame_boxes) == 0:
            flat_bboxes.append(np.array([np.nan, np.nan, np.nan, np.nan]))
        else:
            # Take only the first 4 coordinates (x1, y1, x2, y2) ignoring any existing score
            box = frame_boxes[0][:4]
            flat_bboxes.append(np.array(box))

    flat_bboxes = np.array(flat_bboxes)  # Shape (N, 4)

    # 2. Interpolate
    n_frames = len(flat_bboxes)
    valid_mask = ~np.isnan(flat_bboxes).any(axis=1)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        print("Warning: No valid bounding boxes found in file!")
        return bboxes

    cleaned_bboxes = np.copy(flat_bboxes)
    for col in range(4):
        cleaned_bboxes[:, col] = np.interp(
            np.arange(n_frames),
            valid_indices,
            flat_bboxes[valid_indices, col]
        )

    # 3. Reconstruct with Score Appended [x1, y1, x2, y2, 1.0]
    final_output = []
    for row in cleaned_bboxes:
        # Append 1.0 as the confidence score
        box_with_score = row.tolist() + [1.0]
        final_output.append([box_with_score])

    print(f"Interpolated {n_frames} frames and appended confidence scores.")
    return final_output


def pred_vid(video, vis_dir, pred_dir, model, bboxes_path=None) -> None:
    print(f"Initializing inferencer. Weights will be downloaded to: {os.environ['MMENGINE_CACHE_DIR']}")

    # Check if we are using custom bboxes
    custom_bboxes = load_custom_bboxes(bboxes_path)

    if custom_bboxes is not None:
        custom_bboxes = clean_and_interpolate_bboxes(custom_bboxes)

    if model in models_3d:
        inferencer = MMPoseInferencer(
            pose3d=f'{model}',
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            show_progress=True,
        )
    elif model in models_2d:
        inferencer = MMPoseInferencer(
            pose2d=str(model_map.get(model)[0]) if model in model_map else f'{model}',
            pose2d_weights=str(model_map.get(model)[1]) if model in model_map else None,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            show_progress=True,
        )
    else:
        raise ValueError(f"Model {model} not recognized. Choose from {models_2d + models_3d}.")

    all_frames: List[Dict[str, Any]] = []

    # Prepare arguments for the inferencer
    inference_kwargs = {
        'vis_out_dir': str(vis_dir),
        'pred_out_dir': str(pred_dir),
        'draw_bbox': True,
        'kpt_thr': 0.3,
    }

    # Inject Custom Bboxes
    if custom_bboxes is not None:
        print(f"Using {len(custom_bboxes)} frames of custom bounding boxes.")
        inference_kwargs['bboxes'] = custom_bboxes

    # Run inference
    # Note: If custom_bboxes length < video length, MMPose usually detects on remaining frames.
    for frame_idx, res in enumerate(inferencer(str(video), **inference_kwargs)):
        instances = normalize_predictions(res)
        all_frames.append({'frame_id': frame_idx, 'instances': instances})

    try:
        saved_frames = load_saved_video_predictions(pred_dir, video)
    except Exception as e:
        print(f"Notice: Could not load saved JSON automatically ({e}). Using in-memory results.")
        saved_frames = all_frames

    total_frames = len(all_frames)
    total_people = sum(len(f['instances']) for f in all_frames)
    print(f'Processed {total_frames} frames, detected {total_people} person-instances.')
    print(f'Saved visualizations to: {vis_dir}')
    print(f'Saved predictions JSON to: {pred_dir / (video.stem + model + ".json")}')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="", help="Container path for input video")
    ap.add_argument("--outdir", default="", help="Container path for outputs")
    ap.add_argument("--model", default="vitpose", help="Pose model to use")
    ap.add_argument("--bboxes", default=None, help="Path to JSON file containing custom bounding boxes")

    args = ap.parse_args()
    if not args.video:
        args.video = str(SCRIPT_DIR / 'demos' / "gait" / "IMG_9711-000" / 'IMG_9711-000.mp4')
        args.bboxes = str(SCRIPT_DIR / 'demos' / "gait" / "IMG_9711-000" / 'SplitterPipeline_splitter-IMG_9711-000-detections.json')
    if not args.outdir:
        args.outdir = str(SCRIPT_DIR / "demos" / "outputs")
    return args


def main():
    args = parse_args()
    out_dir = Path(args.outdir)
    video_path = Path(args.video)

    if not video_path.exists():
        print(f"Input video not found: {video_path}")
        sys.exit(1)

    vis_dir = out_dir / "visualizations"
    pred_dir = video_path.parent / "predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    pred_vid(video=video_path, vis_dir=vis_dir, pred_dir=pred_dir, model=args.model, bboxes_path=args.bboxes)


if __name__ == "__main__":
    main()