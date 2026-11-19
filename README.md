# DELTA

Implementation of **Beyond Global Alignment: Structured Compositional Reasoning for Vision-Language Models**.

## Installation

```bash
conda create -n delta python=3.10 -y
conda activate delta
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data

Training uses COCO; evaluation uses CREPE, VALSE, SugarCrepe, and ColorSwap.

```bash
python scripts/download_datasets.py
```

Data is stored in `dataset/`. ColorSwap requires authorized Hugging Face access
for the initial download. VALSE's VisualDialog image preparation is not yet complete.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh
```

Configuration: `config/train_delta.yaml`. Checkpoints: `output/delta/`.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh
```

Configuration: `config/eval_delta.yaml`. Results: `result/delta/summary.json`.
Use a new result directory when changing the checkpoint or dataset to avoid
reusing previous results.

Run training and evaluation together with `bash scripts/train_eval.sh`.
