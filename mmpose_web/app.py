"""
MMPose Web Server
A lightweight web server for running pose estimation on videos using MMPose.
Supports long-running tasks with background processing.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import sys
import logging
from datetime import datetime
from typing import Dict, Optional
import uuid
import traceback

# Add parent directory to path to import infer_engine
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infer_engine.infer import pred_vid
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION
)

# In-memory task storage (for production, use Redis or a database)
tasks: Dict[str, Dict] = {}


class VideoProcessRequest(BaseModel):
    """
    Request model for video processing

    Fields:
        video_filename: Path to video file relative to /inputs/ directory
                       Examples: "my_video.mp4" or "/inputs/my_video.mp4"
        model: Model name to use for inference (default: vitpose)
        bboxes_path: Optional path to bboxes JSON file relative to /inputs/ directory

    Note:
        All file paths should be relative to the /inputs/ directory which is
        mounted from the host system via Docker volume mapping.
    """
    video_filename: str
    model: str = config.DEFAULT_MODEL
    bboxes_path: Optional[str] = None


class TaskResponse(BaseModel):
    """Response model for task submission"""
    task_id: str
    status: str
    message: str


def process_video_task(task_id: str, video_path: Path, vis_dir: Path, pred_dir: Path, model: str, bboxes_path: Optional[str]):
    """
    Background task to process video using pred_vid function.
    Updates task status in the tasks dictionary.
    """
    try:
        logger.info(f"Task {task_id}: Starting video processing for {video_path}")
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["started_at"] = datetime.now().isoformat()

        # Verify paths exist before calling pred_vid
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found at: {video_path}")
        if bboxes_path and not Path(bboxes_path).exists():
            raise FileNotFoundError(f"Bboxes file not found at: {bboxes_path}")

        # Call the pred_vid function from infer_engine
        pred_vid(
            video=video_path,
            vis_dir=vis_dir,
            pred_dir=pred_dir,
            model=model,
            bboxes_path=bboxes_path
        )

        # Update task status to completed
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["completed_at"] = datetime.now().isoformat()
        tasks[task_id]["result"] = {
            "visualization_dir": str(vis_dir),
            "predictions_dir": str(pred_dir),
            "predictions_file": str(pred_dir / f"{video_path.stem}{model}.json")
        }
        logger.info(f"Task {task_id}: Completed successfully")

    except FileNotFoundError as e:
        error_msg = f"{str(e)} - Check that /inputs volume is correctly mounted"
        logger.error(f"Task {task_id}: {error_msg}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = error_msg
        tasks[task_id]["traceback"] = traceback.format_exc()
        tasks[task_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        logger.error(f"Task {task_id}: Failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["traceback"] = traceback.format_exc()
        tasks[task_id]["completed_at"] = datetime.now().isoformat()


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "MMPose Video Processing API",
        "version": "1.0.0",
        "endpoints": {
            "/process": "POST - Submit a video for processing",
            "/status/{task_id}": "GET - Check task status",
            "/tasks": "GET - List all tasks"
        }
    }


@app.post("/process", response_model=TaskResponse)
async def process_video(request: VideoProcessRequest, background_tasks: BackgroundTasks):
    """
    Submit a video for pose estimation processing.

    Args:
        request: VideoProcessRequest containing video_filename, model, and optional bboxes_path
        background_tasks: FastAPI background tasks handler

    Returns:
        TaskResponse with task_id and status

    Note:
        Expects video_filename to be a path relative to /inputs/ (e.g., "my_video.mp4" or "/inputs/my_video.mp4")
        Outputs will be saved to /outputs/visualizations/ and /outputs/predictions/
    """
    try:
        # Handle video path - support both "/inputs/video.mp4" and "video.mp4"
        # Also normalize Windows backslashes to forward slashes
        video_filename = request.video_filename.replace("\\", "/")
        if video_filename.startswith("/inputs/"):
            video_path = Path(video_filename)
        else:
            # Assume it's relative to /inputs
            video_path = Path("/inputs") / video_filename

        # Validate video file exists
        if not video_path.exists():
            error_msg = (
                f"Video file not found at: {video_path}\n"
                f"Requested filename: {request.video_filename}\n"
                f"Expected location: Files should be in /inputs/ directory (mapped from host)\n"
                f"Make sure the file exists in the mounted /inputs volume"
            )
            raise HTTPException(status_code=404, detail=error_msg)

        # Create output directories in /outputs
        vis_dir = Path("/outputs") / "visualizations"
        pred_dir = Path("/outputs") / "predictions"
        vis_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        # Handle bboxes path if provided
        bboxes_path = None
        if request.bboxes_path:
            # Normalize Windows backslashes to forward slashes
            bboxes_filename = request.bboxes_path.replace("\\", "/")
            if bboxes_filename.startswith("/inputs/"):
                bboxes_path = Path(bboxes_filename)
            else:
                bboxes_path = Path("/inputs") / bboxes_filename

            if not bboxes_path.exists():
                error_msg = (
                    f"Bboxes file not found at: {bboxes_path}\n"
                    f"Requested filename: {request.bboxes_path}\n"
                    f"Expected location: Files should be in /inputs/ directory (mapped from host)\n"
                    f"Make sure the file exists in the mounted /inputs volume"
                )
                raise HTTPException(status_code=404, detail=error_msg)

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Initialize task record
        tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "video_path": str(video_path),
            "model": request.model,
            "bboxes_path": str(bboxes_path) if bboxes_path else None,
            "created_at": datetime.now().isoformat(),
            "vis_dir": str(vis_dir),
            "pred_dir": str(pred_dir)
        }

        # Add background task
        background_tasks.add_task(
            process_video_task,
            task_id,
            video_path,
            vis_dir,
            pred_dir,
            request.model,
            str(bboxes_path) if bboxes_path else None
        )
        
        logger.info(f"Task {task_id}: Queued for video {video_path}")
        
        return TaskResponse(
            task_id=task_id,
            status="queued",
            message=f"Video processing task queued successfully. Use /status/{task_id} to check progress."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status of a processing task.

    Args:
        task_id: The unique task identifier

    Returns:
        Task status information including progress and results
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return JSONResponse(content=tasks[task_id])


@app.get("/tasks")
async def list_tasks():
    """
    List all tasks with their current status.

    Returns:
        Dictionary of all tasks
    """
    return JSONResponse(content={
        "total_tasks": len(tasks),
        "tasks": list(tasks.values())
    })


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """
    Delete a task record (does not stop running tasks).

    Args:
        task_id: The unique task identifier

    Returns:
        Confirmation message
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    del tasks[task_id]
    return {"message": f"Task {task_id} deleted successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint with volume mount verification"""
    inputs_exists = Path("/inputs").exists()
    outputs_exists = Path("/outputs").exists()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len([t for t in tasks.values() if t["status"] == "processing"]),
        "queued_tasks": len([t for t in tasks.values() if t["status"] == "queued"]),
        "total_tasks": len(tasks),
        "volumes": {
            "/inputs": "mounted" if inputs_exists else "NOT MOUNTED - Check Docker volume mapping",
            "/outputs": "mounted" if outputs_exists else "NOT MOUNTED - Check Docker volume mapping"
        }
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting MMPose Web Server on {config.SERVER_HOST}:{config.SERVER_PORT}")
    logger.info(f"Timeout: {config.TIMEOUT_KEEP_ALIVE} seconds")
    logger.info(f"Available models: {', '.join(config.AVAILABLE_MODELS)}")

    # Run the server
    uvicorn.run(
        "app:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=config.SERVER_RELOAD,
        timeout_keep_alive=config.TIMEOUT_KEEP_ALIVE
    )

