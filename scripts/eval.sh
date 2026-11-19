DEVICES=${CUDA_VISIBLE_DEVICES:-${DEVICES:-0}}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")

export HF_DATASETS_CACHE="$PARENT_DIR/dataset"
export HF_HOME="$PARENT_DIR/dataset"
export DATASET_BASE_DIR="$PARENT_DIR/dataset"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false

CFG_PATH="${CONFIG_PATH:-$PARENT_DIR/config/eval_delta.yaml}"

echo "Running with config: $CFG_PATH"
CUDA_VISIBLE_DEVICES=$DEVICES python \
    "$PARENT_DIR/evaluate.py" \
    --cfg-path "$CFG_PATH"
