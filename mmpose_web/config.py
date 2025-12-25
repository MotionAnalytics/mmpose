"""
Configuration file for MMPose Web Server
"""

# Server Configuration
SERVER_HOST = "0.0.0.0"  # Listen on all interfaces
SERVER_PORT = 8000
SERVER_RELOAD = False  # Set to True for development

# Timeout Configuration (in seconds)
TIMEOUT_KEEP_ALIVE = 1800  # 30 minutes
REQUEST_TIMEOUT = 1800  # 30 minutes

# Logging Configuration
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Task Configuration
MAX_CONCURRENT_TASKS = 5  # Maximum number of concurrent video processing tasks
TASK_CLEANUP_INTERVAL = 3600  # Clean up completed tasks older than 1 hour (in seconds)

# Default Model Configuration
DEFAULT_MODEL = "vitpose"
AVAILABLE_MODELS = ["vitpose", "vitpose_swimming", "human3d"]

# API Configuration
API_TITLE = "MMPose Video Processing API"
API_DESCRIPTION = "API for running pose estimation on videos"
API_VERSION = "1.0.0"

