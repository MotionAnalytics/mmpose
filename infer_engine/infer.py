from infer_engine._utils import *
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import numpy as np
import os  # Added os
import torch  # Added torch for device check

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





def pred_vid(video, vis_dir, pred_dir) -> None:
    # --- MODIFIED: Initialize inferencer for 2D Pose Tracking ---
    print(f"Initializing inferencer. Weights will be downloaded to: {os.environ['MMENGINE_CACHE_DIR']}")
    inferencer = MMPoseInferencer(
        pose2d='vitpose',  # Use ViTPose model for 2D pose estimation
        #det_model='rtmdet-m',  # Use RTMDet-m for person detection (required for top-down)
        # When processing a video, the inferencer automatically applies
        # tracking (e.g., OCSORT) to the detected instances.
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        show_progress=True,
    )
    # ----------------------------------------------------------

    all_frames: List[Dict[str, Any]] = []
    raw_mmpose_results: List[List[Any]] = []

    # Run inference; also save visualizations and predictions to disk
    # pred_out_dir will produce a single JSON for the video upon finalize
    for frame_idx, res in enumerate(
            inferencer(
                str(video),
                vis_out_dir=str(vis_dir),
                pred_out_dir=str(pred_dir),
                draw_bbox=True,
                kpt_thr=0.3,  # Enabled kpt_thr and draw_bbox for better visualization
            )
    ):
        raw_mmpose_results.append(res['predictions'][0])
        instances = normalize_predictions(res)
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



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=""
                    , help="Container path for input video")
    ap.add_argument("--outdir", default="", help="Container path for outputs")

    args = ap.parse_args()
    if not args.video:
        args.video = str(Path(
            __file__).resolve().parent.parent / "tests/data/posetrack18/videos/000001_mpiinew_test/000001_mpiinew_test.mp4")
    if not args.outdir:
        args.outdir = str(Path(__file__).resolve().parent.parent / "outputs")
    # Prepare output directories
    out_dir = Path(args.outdir)
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Input video not found: {video_path}")
        sys.exit(1)

    vis_dir = out_dir / "visualizations"
    pred_dir = video_path.parent / "predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # <--- MODIFIED: Report weight dir in main
    print(f"Model weights will be saved to: {os.environ.get('MMENGINE_CACHE_DIR', 'Default cache')}")
    # -------------------------------------------

    pred_vid(video_path, vis_dir, pred_dir)


if __name__ == "__main__":
    main()