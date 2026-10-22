# The Impostor AI — Deepfake Image Pair Classification

A computer vision project inspired by **Task 1 — Kẻ Mạo Danh (The Impostor)** from the [OLP AI PTIT 2026 competition](https://aichallenge.ptit.edu.vn/competitions/126/#pages-tab). Given two face images, the goal is to identify which one is AI-generated or manipulated.

This is an **independent educational implementation**, not an official competition solution. All reported scores are from **internal validation**, not the competition leaderboard.

## Overview

Each pair contains one real image and one fake image. The model predicts `fake_position`: `0` if the first image is fake, or `1` if the second image is fake.

The project compares three approaches:

1. Handcrafted image features + Logistic Regression.
2. Frozen DenseNet121 embeddings + Logistic Regression.
3. Fine-tuned DenseNet121 with 3-fold cross-validation.

The final system ensembles **three fine-tuned DenseNet121 models**. For each image, the models estimate its fake probability; the image with the higher ensemble-averaged probability is selected.

## Results

| Method | Evaluation | Macro-F1 |
|---|---|---:|
| Handcrafted features + Logistic Regression | 3-fold OOF | 78.62% |
| Frozen DenseNet121 + Logistic Regression | 3-fold OOF | 85.00% |
| Fine-tuned DenseNet121 | 3-fold OOF | 96.50% |
| **3-model DenseNet121 ensemble** | **Internal holdout (200 pairs)** | **98.00%** |

The final ensemble correctly classified **196/200 holdout pairs** (98.00% accuracy). The original 1,000 labeled training pairs were divided into **800 development pairs** and **200 holdout pairs**; the development set was used for 3-fold cross-validation.

> **Evaluation note:** The frozen-feature and fine-tuned experiments use different preprocessing and prediction heads, so their score difference cannot be attributed solely to fine-tuning. No official public/private leaderboard score is claimed.

## Model

- **Backbone:** DenseNet121 initialized with ImageNet pretrained weights.
- **Classifier:** Two-class real/fake prediction head.
- **Preprocessing:** Center crop to 60% of image dimensions, resize to `224 × 224`, ImageNet normalization.
- **Training augmentation:** Random horizontal flip.
- **Inference:** Average fake probabilities across three fold-specific models and compare the two images.

## Quick Start

**Requirements:** Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/chienvoxn/The-Impostor-AI.git
cd The-Impostor-AI
uv sync
```

### Download pretrained checkpoints

Download all three `.pt` files from **[GitHub Releases — v1.0.0](https://github.com/chienvoxn/The-Impostor-AI/releases/tag/v1.0.0)** and put them in the following directory:

```text
outputs/checkpoints/
├── densenet121_fold0_best.pt
├── densenet121_fold1_best.pt
└── densenet121_fold2_best.pt
```

The weights are distributed as **release assets**, not tracked in Git. You **do not need to retrain** to run inference.

### Run inference

On **Windows PowerShell**:

```powershell
$env:PYTHONPATH = "src"
uv run python inference.py --image0 "path/to/image_0.jpg" --image1 "path/to/image_1.jpg"
```

Example output:

```text
Image 0 - P(fake): 0.020141
Image 1 - P(fake): 0.987265
Predicted fake_position: 1
```

The example comes from labeled training pair `00001`; the script can also run on any two image files. It uses CUDA if available, otherwise CPU.

## Dataset

The official task provides **1,000 labeled training pairs**, **100 public-test pairs**, and **100 private-test pairs**. Dataset files are **not included** in this repository.

- [Competition description](https://aichallenge.ptit.edu.vn/competitions/126/#pages-tab)
- [Dataset archive shared for this project](https://drive.google.com/file/d/1yPp8LtjUcqiYz2iKcKxxAbDa7curttpY/view) (not an official download endpoint; check access and redistribution permissions)

For local notebook experiments, extract the dataset so that the following paths exist:

```text
data/data/train/pairs.csv
data/data/train/images/
```

The fixed development/holdout split is saved at `configs/development_split.csv`. **The dataset is not required to run inference on your own two images** once the checkpoints are downloaded.

## Project Structure

```text
The-Impostor-AI/
├── configs/                 # Development / holdout split
├── notebooks/               # EDA, baselines, training and evaluation (00–06)
├── src/kemaodanh_oai/       # Features, datasets, transforms, models, training
├── tests/                   # Project tests
├── outputs/                 # Local checkpoints and results (gitignored)
├── inference.py             # Inference on a pair of images
├── pyproject.toml
└── README.md
```

The main experiments are in `notebooks/04-finetune-densenet121.ipynb` (training), `notebooks/05-holdout-evaluation.ipynb` (holdout evaluation), and `notebooks/06-public-private-test-inference.ipynb` (unlabeled test inference). GPU experiments were run on Kaggle using an NVIDIA Tesla T4.

## Acknowledgment

Inspired by the **Kẻ Mạo Danh** task from OLP AI PTIT 2026. This repository is an independent project and does not represent the organizers or an official competition submission.
