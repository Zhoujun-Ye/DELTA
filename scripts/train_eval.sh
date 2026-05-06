#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")

echo "Starting training..."
if bash "$SCRIPT_DIR/train.sh"; then
    echo "Training finished. Starting evaluation..."
    bash "$SCRIPT_DIR/eval.sh"
fi
