#!/bin/bash

# This script automates building and running the DELTA Docker image with train/eval/shell modes.

set -e
set -o pipefail

# Default values
MODE="shell"
CONFIG_PATH=""
GPU_FLAG="all"
REBUILD_IMAGE=0

# Argument parsing
while (( "$#" )); do
  case "$1" in
    --train)
      MODE="train"
      shift
      ;;
    --eval)
      MODE="eval"
      shift
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --gpu)
      GPU_FLAG="$2"
      shift 2
      ;;
    --rebuild)
      REBUILD_IMAGE=1
      shift
      ;;
    --help)
      echo "Usage: bash run_docker.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --train                Run training mode"
      echo "  --eval                 Run evaluation mode"
      echo "  --config PATH          Path to configuration file"
      echo "  --gpu DEVICES          GPU device(s) to use (default: all, e.g. 0,1)"
      echo "  --rebuild              Force rebuild the Docker image before running"
      echo "  --help                 Show this help message"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown parameter: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
IMAGE_NAME="delta"
CONTAINER_NAME="delta-run"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Build Docker image if not present or explicitly requested
if [[ "$REBUILD_IMAGE" == "1" || "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "[INFO] Building Docker image ($IMAGE_NAME)..."
    docker build -t $IMAGE_NAME "$REPO_DIR"
fi

# Remove existing container if present
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "[INFO] Removing existing container ($CONTAINER_NAME)..."
    docker rm -f $CONTAINER_NAME
fi

DOCKER_ENV_ARGS=(
    -e HF_HOME="/app/dataset"
    -e HF_DATASETS_CACHE="/app/dataset"
    -e DATASET_BASE_DIR="/app/dataset"
    -e HF_HUB_DISABLE_XET="1"
    -e HF_HUB_ENABLE_HF_TRANSFER="0"
    -e TOKENIZERS_PARALLELISM="false"
    -e LOG_DIR="/app/logs"
)

if [[ -n "$CONFIG_PATH" ]]; then
    case "$CONFIG_PATH" in
        /*)
            CONFIG_IN_CONTAINER="$CONFIG_PATH"
            ;;
        *)
            CONFIG_IN_CONTAINER="/app/$CONFIG_PATH"
            ;;
    esac
    DOCKER_ENV_ARGS+=(-e CONFIG_PATH="$CONFIG_IN_CONTAINER")
fi

GPU_ARGS=(--gpus all)
if [[ "$GPU_FLAG" != "all" ]]; then
    GPU_ARGS=(--gpus "device=$GPU_FLAG")
fi

# Determine which command to run in container
if [[ "$MODE" == "train" ]]; then
    DOCKER_CMD="source /venv/bin/activate && chmod +x scripts/train.sh && bash scripts/train.sh"
elif [[ "$MODE" == "eval" ]]; then
    DOCKER_CMD="source /venv/bin/activate && chmod +x scripts/eval.sh && bash scripts/eval.sh"
else
    DOCKER_CMD="bash"
fi

echo "[INFO] Running DELTA Docker ($MODE mode)..."
docker run "${GPU_ARGS[@]}" -it \
    --name $CONTAINER_NAME \
    --shm-size="16g" \
    -v "$REPO_DIR:/app" \
    -v "$REPO_DIR/data:/app/data" \
    -v "$REPO_DIR/dataset:/app/dataset" \
    -v "$REPO_DIR/output:/app/output" \
    -v "$REPO_DIR/logs:/app/logs" \
    "${DOCKER_ENV_ARGS[@]}" \
    $IMAGE_NAME \
    bash -c "$DOCKER_CMD"

echo "[INFO] Done!"
