# DELTA

Implementation for **Beyond Global Alignment: Structured Compositional Reasoning in Vision-Language Models**.

## Abstract

Despite remarkable progress in image-text understanding, vision-language models (VLMs) still struggle with compositional reasoning. In particular, they often fail to distinguish relation direction and attribute-object binding, leading to similar representations for semantically different image-text pairs. This problem mainly stems from the reliance on global image-text alignment, which captures coarse correspondence but overlooks fine-grained compositional structures. Towards this end, we propose an evidence-aware framework, termed Relation Bucketing with Binding Localization (DELTA), to improve compositional understanding in VLMs. The key idea of our DELTA is to model compositionality from two complementary perspectives: direction-aware relation bucketing and object-conditioned binding localization. More specifically, we first leverage a learnable gate to model the cumulative contextual distance between anchor terms, thereby assigning relation concepts to direction-aware discrete buckets. To enrich textual representations with fine-grained relational semantics, we calibrate the global semantic representations by incorporating intermediate-layer hidden states. Furthermore, DELTA leverages textual cues to ground visual evidence for object attributes, pulling image patches closer to their matched attribute descriptions while pushing them away from incorrect augmented ones. Extensive experiments on four compositional datasets demonstrate the effectiveness of our method.

## Repository Structure

```text
config/
  train_delta.yaml       # default training configuration
  eval_delta.yaml        # default multi-benchmark evaluation configuration
scripts/
  download_datasets.py  # pre-download datasets into ./dataset
  train.sh              # launch training with config/train_delta.yaml
  eval.sh               # launch evaluation with config/eval_delta.yaml
  train_eval.sh         # run training, then evaluation
  run_docker.sh         # Docker launcher for shell/train/eval modes
src/
  builders/             # dataset builders
  collators/            # image/text batch preparation
  models/               # DELTA model and configuration
  runners/              # trainer and evaluators
  tasks/                # task construction from YAML configs
train.py
evaluate.py
```

## Environment

### Option 1: Docker

The Docker image pins CUDA 12.1, PyTorch 2.3.1, and the Python packages in `requirements.txt`.

```bash
bash scripts/run_docker.sh --rebuild
```

To run training or evaluation directly in Docker:

```bash
bash scripts/run_docker.sh --train --gpu 0
bash scripts/run_docker.sh --eval --gpu 0
```

Use `--gpu all` to expose all GPUs, or `--config path/to/config.yaml` to override the default config used by the launcher.

### Option 2: Python Environment

Create a Python environment on a GPU server, install PyTorch, then install the project dependencies.

```bash
conda create -n delta python=3.10 -y
conda activate delta

pip install --upgrade pip
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

If your server uses a different CUDA version, install the matching PyTorch build from the official PyTorch index before installing `requirements.txt`.

## Data Preparation

All dataset caches are expected under the repository-local `dataset/` directory. The launcher scripts set these variables automatically, but it is useful to export them before manual runs:

```bash
export DATASET_BASE_DIR="$PWD/dataset"
export HF_HOME="$PWD/dataset"
export HF_DATASETS_CACHE="$PWD/dataset"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
```

Pre-download the datasets used by the default training and evaluation configs:

```bash
python scripts/download_datasets.py
```

This prepares the COCO Karpathy training data with images and the default evaluation datasets loaded by the builders, including CREPE, VALSE, SugarCrepe, SugarCrepe++, ARO-COCO-Order, and ARO-Flickr-Order.

The default evaluation config also includes ColorSwap. If the dataset is not already available locally, provide Hugging Face access:

```bash
export HF_TOKEN=<your_huggingface_token>
```

Alternatively, place downloaded ColorSwap parquet files under:

```text
dataset/colorswap/data/test-*.parquet
```

## Training

Run the default DELTA training configuration:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh
```

The script uses `config/train_delta.yaml` by default and writes checkpoints to:

```text
output/delta
```

To use a custom config:

```bash
CONFIG_PATH=config/train_delta.yaml CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh
```

For distributed training, launch `train.py` with `torchrun` and the desired config:

```bash
torchrun --nproc_per_node=4 train.py --cfg-path config/train_delta.yaml
```

Common settings to adjust are in `config/train_delta.yaml`, including `trainer.per_device_train_batch_size`, `trainer.num_train_epochs`, `trainer.learning_rate`, and `trainer.output_dir`.

## Evaluation

Evaluate the trained model:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh
```

The default evaluation config loads the model from:

```text
output/delta
```

and writes results to:

```text
result/delta
```

Each benchmark result is saved as a JSON file, and the aggregate metrics are saved to:

```text
result/delta/summary.json
```

To evaluate another checkpoint, edit `model.config.pretrained_model_name_or_path` in `config/eval_delta.yaml`, or pass a custom config:

```bash
CONFIG_PATH=config/eval_delta.yaml CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh
```

To run training followed by evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_eval.sh
```