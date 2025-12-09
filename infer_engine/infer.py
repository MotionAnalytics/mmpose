from infer_engine._utils import *
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import numpy as np
import os
import torch

# --- Setup Cache Directories (Kept from your original script) ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_WEIGHTS_DIR = SCRIPT_DIR.parent / "model_weights"
MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ['MMENGINE_CACHE_DIR'] = str(MODEL_WEIGHTS_DIR)

TORCH_CACHE_DIR = MODEL_WEIGHTS_DIR / "torch_cache"
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ['TORCH_HOME'] = str(TORCH_CACHE_DIR)

# ----------------------------------------------------------------

from mmpose.apis.inferencers import MMPoseInferencer

models_3d = ["human3d"]
models_2d = ["vitpose","vitpose_swimming"]

model_map = {
    #"vitpose":("vitpose",None),
    "vitpose_swimming":(
        MODEL_WEIGHTS_DIR/"swimming_vitpose"/"td-hm_ViTPose-base_8xb64-210e_coco-256x192"/"td-hm_ViTPose-base_8xb64-210e_coco-256x192.py",
        MODEL_WEIGHTS_DIR/"swimming_vitpose"/"td-hm_ViTPose-base_8xb64-210e_coco-256x192"/"best_coco_AP_epoch_18.pth"
                     ),
             }




def load_custom_bboxes(bbox_path: str) -> List[Any]:
    """
    Loads bounding boxes from a JSON file.
    Expected format: A list of lists, where index i corresponds to frame i.
    Example: [ [[x1, y1, x2, y2], [x1, y1, x2, y2]], ... ]
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


def pred_vid(video, vis_dir, pred_dir, model, bboxes_path=None) -> None:
    print(f"Initializing inferencer. Weights will be downloaded to: {os.environ['MMENGINE_CACHE_DIR']}")

    # Check if we are using custom bboxes
    custom_bboxes = load_custom_bboxes(bboxes_path)

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
    raw_mmpose_results: List[List[Any]] = []

    # Prepare arguments for the inferencer
    inference_kwargs = {
        'vis_out_dir': str(vis_dir),
        'pred_out_dir': str(pred_dir),
        'draw_bbox': True,
        'kpt_thr': 0.3,
    }

    # --- MODIFICATION: Inject Custom Bboxes ---
    if custom_bboxes is not None:
        print(f"Using {len(custom_bboxes)} frames of custom bounding boxes.")
        inference_kwargs['bboxes'] = custom_bboxes
    # ------------------------------------------

    # Run inference
    for frame_idx, res in enumerate(inferencer(str(video), **inference_kwargs)):
        raw_mmpose_results.append(res['predictions'][0])
        instances = normalize_predictions(res)
        all_frames.append({'frame_id': frame_idx, 'instances': instances})

    # Read back the saved JSON to demonstrate disk I/O
    # Note: If MMPose saves files incrementally, this logic holds.
    # If not, you might rely on 'all_frames' directly.
    try:
        saved_frames = load_saved_video_predictions(pred_dir, video)
    except Exception as e:
        print(f"Notice: Could not load saved JSON automatically ({e}). Using in-memory results.")
        saved_frames = all_frames

    # Transform to numpy for downstream processing
    kps_per_frame, scores_per_frame = frames_to_arrays(all_frames)

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
        #args.video = str(Path(__file__).parent / 'demos' /"basketball"/"basketball_vid_1"/'basketball_vid_1.mp4')
        #args.video = str(SCRIPT_DIR / 'demos' /"swimming"/"sagittal_test"/'sagittal_test.mp4')
        args.video = str(SCRIPT_DIR / 'demos' /"gait"/"Koren-Adj-F-L2R-Med-11"/'Koren-Adj-F-L2R-Med-11.mp4')
        args.bboxes = str(SCRIPT_DIR / 'demos' /"gait"/"Koren-Adj-F-L2R-Med-11"/'Koren-Adj-F-L2R-Med-11_T0.json')
    if not args.outdir:
        args.outdir = str(SCRIPT_DIR / "demos"/"outputs")
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

    print(f"Model weights will be saved to: {os.environ.get('MMENGINE_CACHE_DIR', 'Default cache')}")

    # Pass the bbox argument
    pred_vid(video=video_path,vis_dir= vis_dir,pred_dir=pred_dir,model = args.model,bboxes_path=args.bboxes)

if __name__ == "__main__":
    main()