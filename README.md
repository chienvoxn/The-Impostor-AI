# The Impostor AI — Deepfake Image Pair Classification

An independent computer vision / deep learning project based on **Task 1 — Kẻ Mạo Danh (The Impostor)** from **[REOPEN] Vòng sơ loại OLP AI PTIT 2026 – Ca 1**, organized by the **Faculty of Artificial Intelligence, Posts and Telecommunications Institute of Technology (PTIT)**.

- **Official competition:** [OLP AI PTIT 2026 — Competition 126](https://aichallenge.ptit.edu.vn/competitions/126/#pages-tab)
- **Task:** Identify which image in a pair of face images is AI-generated or AI-manipulated.
- **Project scope:** Independent educational implementation and evaluation of a DenseNet121-based solution. This repository is **not** an official organizer solution and does not claim competition participation, placement, or leaderboard performance.

Given two images, the model predicts `fake_position`:

- `0`: Image 0 is predicted to be fake.
- `1`: Image 1 is predicted to be fake.

## 1. Overview

This project explores three approaches to deepfake image pair classification:

1. Handcrafted image features with Logistic Regression.
2. Frozen DenseNet121 embeddings with Logistic Regression.
3. End-to-end fine-tuning of DenseNet121 with 3-fold cross-validation.

The final solution uses an ensemble of three fine-tuned DenseNet121 models.

## 2. Official Task and Dataset

According to the published OLP AI PTIT 2026 task description, each pair contains **exactly one real face image and one AI-generated or manipulated face image**. The target `fake_position` indicates the position of the fake image (`0` or `1`). The official task uses **Macro-F1** as its primary raw evaluation metric; the competition leaderboard additionally applies a normalization formula.

| Official split | Image pairs | Images | Labels |
|---|---:|---:|---|
| `train` | 1,000 | 2,000 | Available (`fake_position`) |
| `public_test` | 100 | 200 | Hidden |
| `private_test` | 100 | 200 | Hidden |

The official training labels are stored in `train/pairs.csv`, with columns `pair_id`, `image_0`, `image_1`, and `fake_position`. Test CSV files omit `fake_position`. The official task description states that each image appears in only one pair within its split.

**Important distinction:** This repository takes the **official 1,000 labeled training pairs** and creates its own **800-pair development / 200-pair holdout split**. Its reported OOF and holdout metrics are **internal project results**, **not official public/private test scores or normalized leaderboard scores**.

## 3. Experimental Results

| Method | Evaluation | Macro-F1 |
|---|---|---|
| Handcrafted Features + Logistic Regression | 3-fold OOF | 78.62% |
| Frozen DenseNet121 + Logistic Regression | 3-fold OOF | 85.00% |
| Fine-tuned DenseNet121 | 3-fold OOF | 96.50% |
| Fine-tuned DenseNet121 (3-model ensemble) | Holdout | **98.00%** |

### Preprocessing differences and interpretation

The frozen-feature and fine-tuned DenseNet121 experiments **do not use identical image preprocessing**:

| Experiment | Input preprocessing | Training approach |
|---|---|---|
| Notebook 03 — Frozen DenseNet121 + Logistic Regression | `DenseNet121_Weights.IMAGENET1K_V1.transforms()` (ImageNet preset resize/crop and normalization) | Fixed pretrained feature extractor; Logistic Regression fitted on pair features |
| Notebook 04 — Fine-tuned DenseNet121 | Center crop to 60% of image width and height, resize to `224 × 224`, ImageNet normalization; random horizontal flip during training | End-to-end fine-tuning with a two-class classification head |

Therefore, the improvement from **85.00%** (frozen) to **96.50%** (fine-tuned) OOF Macro-F1 **cannot be attributed solely to fine-tuning**: input preprocessing, prediction head and training setup also differ. These are results of two *pipelines*, not a controlled single-variable ablation. A controlled experiment would use the same center60 transform for frozen feature extraction and keep the validation protocol fixed.

The holdout evaluation achieved:

- **Macro-F1:** 98.00%
- **Accuracy:** 98.00%
- **Correct predictions:** 196/200 image pairs

The public and private test sets were also processed to generate prediction files. Their ground-truth evaluation scores are not reported here.

## 4. Download and Prepare the Dataset

The **official training split** contains 1,000 labeled image pairs. This project uses only those labeled pairs for its internal development/holdout experiments.

The data was divided into:

- **Development set:** 800 pairs for training and cross-validation.
- **Holdout set:** 200 pairs reserved for final evaluation.

The development set was further divided into three folds.

The holdout set was excluded from model training and was reserved for final evaluation.

### Obtaining and organizing the dataset

The dataset is **not stored in this Git repository**. Download the dataset archive from the following link shared for this project:

- **[Official competition page (PTIT)](https://aichallenge.ptit.edu.vn/competitions/126/#pages-tab)** — original task description and competition context.
- **[Dataset archive (Google Drive)](https://drive.google.com/file/d/1yPp8LtjUcqiYz2iKcKxxAbDa7curttpY/view)** — copy of the archive shared for this project (not presented as an official PTIT download endpoint).

The archive used in this project is named `the_imposter.zip`. The Google Drive link was supplied by the project owner; its public access permissions and the right to redistribute the competition data have not been independently verified. Check the competition rules and the file-sharing permissions before using or sharing it. `uv sync` does **not** download the dataset.

Extract the archive. Inside it, locate `the_imposter/data/train/pairs.csv` and `the_imposter/data/train/images/`. Copy the archive’s inner `data/` folder into the repository’s `data/` folder so the expected local structure is:

```text
data/                              # gitignored
└── data/
    ├── train/
    │   ├── pairs.csv
    │   └── images/
    │       ├── 00001.jpg
    │       └── ...
    ├── public_test/               # optional; for notebook 06
    └── private_test/              # optional; for notebook 06
```

The outer `data/` is the repository's local data folder; the inner `data/` is copied from `the_imposter/data/` in the downloaded archive. **Do not copy the top-level `the_imposter/` folder directly into the repository root.** Confirm that both `data/data/train/pairs.csv` and `data/data/train/images/` exist. The repository includes the fixed split file `configs/development_split.csv` (800 development pairs, 200 holdout pairs). **Keep this split unchanged** to compare with the reported results.

Notebook 00 also searches for `data/train/` and `data/data/data/train/` if your extraction layout differs. For Kaggle notebooks 03–06, attach the dataset as a Kaggle Input dataset containing the `train/` directory and `pairs.csv` (and the split CSV when required); those notebooks search under `/kaggle/input`.

The images and labels are not redistributed in this repository. Verify that you have permission to access and use the dataset, and respect its original license and redistribution conditions.

## 5. Model Architecture

The final model uses DenseNet121 initialized with ImageNet pretrained weights.

The original classifier is replaced with a two-class classification head, and the network is fine-tuned for real/fake image classification.

Image preprocessing includes:

- Center cropping to 60% of the original width and height.
- Resizing to 224 × 224 pixels.
- Conversion to tensors.
- ImageNet normalization.

Random horizontal flipping is used during training only.

For pair-level prediction, the model calculates the probability that each image is fake. The image with the higher fake probability is selected.

The final ensemble averages the fake probabilities produced by the three fold-specific models.

## 6. Project Structure

```text
The-Impostor-AI/
├── configs/
│   └── development_split.csv
├── data/                         # Downloaded dataset (gitignored)
├── notebooks/
│   ├── 00_problem_and_data_documented.ipynb
│   ├── 01_eda_and_baseline.ipynb
│   ├── 02_features_and_lr.ipynb
│   ├── 03-frozen-cnn-features.ipynb
│   ├── 04-finetune-densenet121.ipynb
│   ├── 05-holdout-evaluation.ipynb
│   └── 06-public-private-test-inference.ipynb
├── outputs/                      # Local results and checkpoints
│   ├── checkpoints/
│   └── results/
├── src/
│   └── kemaodanh_oai/
│       ├── dataset.py
│       ├── features.py
│       ├── models.py
│       ├── training.py
│       └── transforms.py
├── inference.py
├── pyproject.toml
└── README.md
```

The notebooks are numbered in the intended experiment order, from data exploration to final inference.

## 7. Environment Setup

Requirements:

- Python 3.12
- `uv`
- PyTorch and torchvision
- Other dependencies declared in `pyproject.toml`

Clone the repository:

```bash
git clone https://github.com/chienvoxn/The-Impostor-AI.git
cd The-Impostor-AI
```

Install the project dependencies:

```bash
uv sync
```

## 8. Model Checkpoints

The final inference script expects three trained checkpoint files:

```text
outputs/checkpoints/
├── densenet121_fold0_best.pt
├── densenet121_fold1_best.pt
└── densenet121_fold2_best.pt
```

These checkpoint files are not included in the Git repository.

Each checkpoint stores the fine-tuned model weights under the `model_state_dict` key.

### How to obtain the trained checkpoints

The `.pt` files are generated by **Notebook 04**; they are not included in Git because model weights are excluded by `.gitignore`. To obtain them:

1. Open `notebooks/04-finetune-densenet121.ipynb` in Kaggle with a GPU enabled and the original training dataset attached as Kaggle Input. The notebook clones this repository and locates `train/pairs.csv` in `/kaggle/input`.
2. Run all cells to train the three folds. The notebook writes best checkpoints to `/kaggle/working/outputs/checkpoints/` and packages the best checkpoints and OOF results into `/kaggle/working/densenet121_3fold_results.zip`.
3. Download `densenet121_3fold_results.zip` from Kaggle Output, extract it, and copy the three `checkpoints/densenet121_fold{0,1,2}_best.pt` files to the repository's `outputs/checkpoints/` directory.
4. Confirm that all three files exist before running `inference.py`. The inference script **does not download checkpoints** and will raise `FileNotFoundError` if any is missing.

If you already trained the model, reuse your existing three best checkpoints instead of retraining. For full reproducibility, use the same code revision and environment as the original experiment; the Kaggle notebooks currently clone/pull `master`, so the revision may change over time.

**Important:** `git clone` and `uv sync` install only code and dependencies. They do **not** download the dataset or the trained checkpoints. You need both for evaluation on the original data, and the checkpoints plus any two image files for local inference.

## 9. Run Local Inference

The inference script accepts two image paths and predicts which image is fake.

On Windows PowerShell, set the Python source path:

```powershell
$env:PYTHONPATH = "src"
```

Run inference:

```powershell
uv run python inference.py --image0 "path/to/image_0.jpg" --image1 "path/to/image_1.jpg"
```

Example output:

```text
Device: cpu
Loaded fold 0
Loaded fold 1
Loaded fold 2

========== RESULT ==========
Image 0 - P(fake): 0.020141
Image 1 - P(fake): 0.987265
Predicted fake_position: 1
Ảnh được dự đoán là giả: image_1
```

The example above was obtained from training pair `00001`, whose ground-truth `fake_position` is `1`.

The inference script automatically uses CUDA when available and falls back to CPU otherwise.

## 10. Training and Evaluation

The main experimental workflow is organized into notebooks:

- **Notebook 04:** Fine-tune DenseNet121 with 3-fold cross-validation.
- **Notebook 05:** Evaluate the ensemble on the holdout set.
- **Notebook 06:** Generate public and private test predictions.

GPU-based experiments were performed on Kaggle using an NVIDIA Tesla T4.
