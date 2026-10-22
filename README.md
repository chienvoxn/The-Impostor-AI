# The Impostor AI — Deepfake Image Pair Classification

A machine learning project for identifying the manipulated image in a pair of images using computer vision and deep learning.

Given two images, the model predicts `fake_position`:

- `0`: Image 0 is predicted to be fake.
- `1`: Image 1 is predicted to be fake.

## 1. Overview

This project explores three approaches to deepfake image pair classification:

1. Handcrafted image features with Logistic Regression.
2. Frozen DenseNet121 embeddings with Logistic Regression.
3. End-to-end fine-tuning of DenseNet121 with 3-fold cross-validation.

The final solution uses an ensemble of three fine-tuned DenseNet121 models.

## 2. Experimental Results

| Method | Evaluation | Macro-F1 |
|---|---|---|
| Handcrafted Features + Logistic Regression | 3-fold OOF | 78.62% |
| Frozen DenseNet121 + Logistic Regression | 3-fold OOF | 85.00% |
| Fine-tuned DenseNet121 | 3-fold OOF | 96.50% |
| Fine-tuned DenseNet121 (3-model ensemble) | Holdout | **98.00%** |

The holdout evaluation achieved:

- **Macro-F1:** 98.00%
- **Accuracy:** 98.00%
- **Correct predictions:** 196/200 image pairs

The public and private test sets were also processed to generate prediction files. Their ground-truth evaluation scores are not reported here.

## 3. Dataset and Validation Strategy

The training dataset contains 1,000 labeled image pairs.

The data was divided into:

- **Development set:** 800 pairs for training and cross-validation.
- **Holdout set:** 200 pairs reserved for final evaluation.

The development set was further divided into three folds.

The holdout set was excluded from model training and hyperparameter selection.

## 4. Model Architecture

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

## 5. Project Structure

```text
The-Impostor-AI/
├── configs/
│   └── development_split.csv
├── data/                         # Local dataset (not tracked)
├── notebooks/
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

The repository also contains earlier experimentation notebooks.

## 6. Environment Setup

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

## 7. Model Checkpoints

The final inference script expects three trained checkpoint files:

```text
outputs/checkpoints/
├── densenet121_fold0_best.pt
├── densenet121_fold1_best.pt
└── densenet121_fold2_best.pt
```

These checkpoint files are not included in the Git repository.

Each checkpoint stores the fine-tuned model weights under the `model_state_dict` key.

## 8. Run Local Inference

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

## 9. Training and Evaluation

The main experimental workflow is organized into notebooks:

- **Notebook 04:** Fine-tune DenseNet121 with 3-fold cross-validation.
- **Notebook 05:** Evaluate the ensemble on the holdout set.
- **Notebook 06:** Generate public and private test predictions.

GPU-based experiments were performed on Kaggle using an NVIDIA Tesla T4.

## 10. Notes

- Dataset files, model checkpoints, and generated outputs are excluded from Git tracking.
- The reported holdout performance is measured on the reserved 200-pair split.
- Public/private test predictions should not be interpreted as independently verified accuracy scores.
- The final inference script requires the three fine-tuned checkpoints to be available locally.
