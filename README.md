<h1 align="center">DELTA</h1>

<p align="center">
  <strong>Beyond Global Alignment: Structured Compositional Reasoning for Vision-Language Models</strong><br>
  Zhoujun Ye, Yiwei Fu, Qiyun Huang, Jie Yang, Dongjie Wang, Xiao Luo<br>
  <em>NeurIPS 2026</em>
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=4HnBl3JKLK"><img alt="Paper on OpenReview" src="https://img.shields.io/badge/Paper-OpenReview-8c1d40"></a>
  <a href="LICENSE"><img alt="License: BSD 3-Clause" src="https://img.shields.io/badge/License-BSD_3--Clause-2d6a4f"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#citation">Citation</a> &middot;
  <a href="#license">License</a>
</p>

This is the official implementation of DELTA. It helps vision-language models
distinguish relation direction and bind attributes to the correct objects.

## Quick Start

### 1. Install

```bash
conda create -n delta python=3.10 -y
conda activate delta
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python scripts/download_datasets.py
```

The script stores data in `dataset/`.

| Stage | Datasets |
| --- | --- |
| Training | COCO |
| Evaluation | CREPE, VALSE, SugarCrepe, ColorSwap |

### 3. Train

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh
```

### 4. Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh
```

Run training and evaluation together with `bash scripts/train_eval.sh`.

## Citation

If you use DELTA in your research, please cite:

```bibtex
@inproceedings{ye2026beyond,
  title     = {Beyond Global Alignment: Structured Compositional Reasoning for Vision-Language Models},
  author    = {Ye, Zhoujun and Fu, Yiwei and Huang, Qiyun and Yang, Jie and Wang, Dongjie and Luo, Xiao},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=4HnBl3JKLK}
}
```

## License

The DELTA code is released under the [BSD 3-Clause License](LICENSE).
The Salesforce utility files in `src/utils/` retain their original copyright
notices and BSD 3-Clause license. External datasets and pretrained models
remain subject to their respective terms.
