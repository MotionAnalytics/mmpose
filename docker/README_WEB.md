# MMPose Web Server - Docker

Docker image with REST API server for pose estimation on videos.

## Purpose

The MMPose web Docker container provides a REST API server for pose estimation processing. It enables users to submit videos for pose detection and tracking through HTTP endpoints.

## Quick Start

### Build the Image

```bash
docker build -f docker/DockerfileWeb -t mmpose-web:latest .
```

### Run the Container

**Using PowerShell script (Windows):**
```powershell
cd docker
.\run_web.ps1
```

**With GPU:**
```bash
docker run -d --gpus all -p 8000:8000 --name mmpose-web mmpose-web:latest
```

**Without GPU:**
```bash
docker run -d -p 8000:8000 --name mmpose-web mmpose-web:latest
```

**With volumes (recommended):**
```bash
docker run -d --gpus all -p 8000:8000 \
  -v $(pwd)/videos:/videos \
  -v $(pwd)/outputs:/outputs \
  --name mmpose-web \
  mmpose-web:latest
```

## API Access

Once running, access the API at:
- **Swagger UI**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

## Basic Usage

**Submit a video:**
```bash
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{"video_filename": "/videos/video.mp4", "model": "vitpose"}'
```

**Check status:**
```bash
curl "http://localhost:8000/status/{task_id}"
```

## Common Commands

```bash
# View logs
docker logs -f mmpose-web

# Stop container
docker stop mmpose-web

# Remove container
docker rm mmpose-web
```

## System Requirements

- Docker 20.10+
- NVIDIA Docker Runtime (for GPU support)
- 8GB+ RAM
- ~10GB disk space
