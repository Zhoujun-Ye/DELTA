DEVICES=${CUDA_VISIBLE_DEVICES:-${DEVICES:-0}}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")

export HF_DATASETS_CACHE="$PARENT_DIR/dataset"
export HF_HOME="$PARENT_DIR/dataset"
export DATASET_BASE_DIR="$PARENT_DIR/dataset"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export LOG_DIR="${LOG_DIR:-$PARENT_DIR/logs}"

CFG_PATH="${CONFIG_PATH:-$PARENT_DIR/config/train_delta.yaml}"

echo "Currently Running with Config: $CFG_PATH"

CUDA_VISIBLE_DEVICES=$DEVICES python \
    "$PARENT_DIR/train.py" \
        --cfg-path "$CFG_PATH"
