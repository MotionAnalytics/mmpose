# MMPose Web Server - PowerShell Run Script

# Configuration parameters
$IMAGE_NAME = "mmpose-web"
$CONTAINER_NAME = "mmpose-web"
$PORT = 8080
$INPUT_DIR = "$PSScriptRoot\..\test_dataset\inputs"
$OUTPUT_DIR = "$PSScriptRoot\..\test_dataset\outputs"
$HEALTH_URL = "http://localhost:$PORT/health"

Write-Host "========================================" -ForegroundColor Green
Write-Host "MMPose Web Server - Startup Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Create input/output directories if they don't exist
if (-not (Test-Path $INPUT_DIR)) {
    New-Item -ItemType Directory -Path $INPUT_DIR -Force | Out-Null
    Write-Host "Created: $INPUT_DIR" -ForegroundColor Yellow
}
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR -Force | Out-Null
    Write-Host "Created: $OUTPUT_DIR" -ForegroundColor Yellow
}

# Check if server is already running
Write-Host "Checking if server is already running..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri $HEALTH_URL -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "Server is already running and healthy!" -ForegroundColor Green
        Write-Host "URL: $HEALTH_URL" -ForegroundColor Green
        exit 0
    }
} catch {
    Write-Host "Server is not running. Starting..." -ForegroundColor Yellow
}

# Stop and remove existing container if it exists (but not running)
$existing = docker ps -a -q -f name=$CONTAINER_NAME 2>$null
if ($existing) {
    Write-Host "Removing existing container..." -ForegroundColor Yellow
    docker rm -f $CONTAINER_NAME 2>$null | Out-Null
}

# Run the container
Write-Host "Starting Docker container..." -ForegroundColor Cyan
Write-Host "Image: $IMAGE_NAME" -ForegroundColor Gray
Write-Host "Port: $PORT" -ForegroundColor Gray
Write-Host ""

docker run `
    --rm `
    --gpus all `
    --name $CONTAINER_NAME `
    -p "${PORT}:8000" `
    -v "${INPUT_DIR}:/inputs" `
    -v "${OUTPUT_DIR}:/outputs" `
    -w /mmpose/mmpose_web `
    -e PYTHONPATH=/opt/project `
    -d `
    $IMAGE_NAME

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start container!" -ForegroundColor Red
    exit 1
}

# Wait for server to start
Write-Host ""
Write-Host "Waiting for server to start..." -ForegroundColor Cyan
$maxAttempts = 30
$attempt = 0
$success = $false

while ($attempt -lt $maxAttempts) {
    Start-Sleep -Seconds 2
    $attempt++
    
    try {
        $response = Invoke-WebRequest -Uri $HEALTH_URL -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $success = $true
            break
        }
    } catch {
        Write-Host "." -NoNewline -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host ""

if ($success) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Server started successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "API URL:      http://localhost:$PORT" -ForegroundColor Cyan
    Write-Host "Swagger UI:   http://localhost:$PORT/docs" -ForegroundColor Cyan
    Write-Host "Health Check: $HEALTH_URL" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Container: $CONTAINER_NAME" -ForegroundColor Gray
    Write-Host ""
    Write-Host "To stop: docker stop $CONTAINER_NAME" -ForegroundColor Yellow
    Write-Host "To view logs: docker logs -f $CONTAINER_NAME" -ForegroundColor Yellow
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Failed to start server!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check logs with: docker logs $CONTAINER_NAME" -ForegroundColor Yellow
    exit 1
}

