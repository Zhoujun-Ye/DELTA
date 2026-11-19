#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")

echo "Starting training..."
bash "$SCRIPT_DIR/train.sh"
echo "Training finished. Starting evaluation..."
bash "$SCRIPT_DIR/eval.sh"
