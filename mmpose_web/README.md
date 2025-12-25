# MMPose Web Server

A lightweight web server for running pose estimation on videos using MMPose. This API supports long-running video processing tasks (15-30 minutes) with background task processing to handle multiple concurrent requests.

## Features

- **Asynchronous Processing**: Submit videos and get immediate task ID, check status later
- **Long Timeout Support**: Handles video processing tasks up to 30 minutes
- **Concurrent Requests**: Process multiple videos simultaneously using background tasks
- **RESTful API**: Simple POST/GET endpoints for easy integration
- **Task Management**: Track, list, and manage processing tasks

## Installation

1. Install the required dependencies:

```bash
cd mmpose_web
pip install -r requirements.txt
```

Note: Make sure you have already installed the main MMPose dependencies in the parent project.

## Running the Server

Start the server using:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 1800
```

The server will start on `http://localhost:8000`

## API Endpoints

### 1. Submit Video for Processing

**POST** `/process`

Submit a video file for pose estimation processing.

**Request Body:**
```json
{
  "video_filename": "/path/to/video.mp4",
  "model": "vitpose",
  "bboxes_path": "/path/to/bboxes.json"  // optional
}
```

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Video processing task queued successfully. Use /status/{task_id} to check progress."
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{
    "video_filename": "C:/path/to/video.mp4",
    "model": "vitpose"
  }'
```

**Example using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/process",
    json={
        "video_filename": "C:/path/to/video.mp4",
        "model": "vitpose"
    }
)
task_id = response.json()["task_id"]
print(f"Task ID: {task_id}")
```

### 2. Check Task Status

**GET** `/status/{task_id}`

Get the current status and results of a processing task.

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "video_path": "/path/to/video.mp4",
  "model": "vitpose",
  "created_at": "2025-12-24T18:30:00",
  "started_at": "2025-12-24T18:30:05",
  "completed_at": "2025-12-24T18:45:00",
  "result": {
    "visualization_dir": "/path/to/visualizations",
    "predictions_dir": "/path/to/predictions",
    "predictions_file": "/path/to/predictions/video_vitpose.json"
  }
}
```

**Status values:**
- `queued`: Task is waiting to be processed
- `processing`: Task is currently being processed
- `completed`: Task completed successfully
- `failed`: Task failed with an error

**Example:**
```bash
curl "http://localhost:8000/status/550e8400-e29b-41d4-a716-446655440000"
```

### 3. List All Tasks

**GET** `/tasks`

Get a list of all tasks and their statuses.

**Example:**
```bash
curl "http://localhost:8000/tasks"
```

### 4. Delete Task

**DELETE** `/tasks/{task_id}`

Delete a task record (does not stop running tasks).

**Example:**
```bash
curl -X DELETE "http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000"
```

### 5. Health Check

**GET** `/health`

Check server health and get statistics.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-24T18:30:00",
  "active_tasks": 2,
  "queued_tasks": 1,
  "total_tasks": 10
}
```

## Configuration

### Timeout Settings

The server is configured with a 30-minute keep-alive timeout to support long-running video processing tasks. You can adjust this in `app.py`:

```python
uvicorn.run(
    "app:app",
    host="0.0.0.0",
    port=8000,
    timeout_keep_alive=1800  # 30 minutes in seconds
)
```

### Concurrent Processing

The server uses FastAPI's BackgroundTasks to handle multiple requests concurrently. Each video processing task runs in the background, allowing the API to accept new requests immediately.

## Notes

- Video files and output directories are created relative to the video file location
- Predictions are saved to `{video_parent_dir}/predictions/`
- Visualizations are saved to `{video_parent_dir}/visualizations/`
- Task records are stored in memory and will be lost when the server restarts
- For production use, consider using Redis or a database for task storage

## Troubleshooting

### Video file not found
Ensure the full absolute path to the video file is provided in the request.

### Task stuck in "processing"
Check the server logs for errors. The task may have failed but the status wasn't updated.

### Server timeout
For very long videos (>30 minutes), increase the `timeout_keep_alive` parameter.

## License

This web server is part of the MMPose project.

