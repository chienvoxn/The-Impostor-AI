from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, laplace
import cv2

FEATURE_NAMES = [
    "log_bytes",
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "q10_r", "q10_g", "q10_b",
    "q90_r", "q90_g", "q90_b",
    "sat_mean", "sat_std",
    "val_mean", "val_std",
    "gray_mean", "gray_std",
    "entropy",
    "grad_mean", "grad_std",
    "lap_std", "lap_abs",
    "residual_std", "residual_abs",
    "block_x", "block_y",
    "center_mean", "center_std",
    "border_mean", "border_std",
]


# ============================================================
# 1. IMAGE LOADING
# ============================================================

def load_rgb_image(image_path):
    """Load an RGB image normalized to [0, 1]."""
    image_path = Path(image_path)

    with Image.open(image_path) as image:
        rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.float32,
        ) / 255.0

    return rgb


# ============================================================
# 2. RGB STATISTICS (12 FEATURES)
# ============================================================

def extract_rgb_statistics(rgb):
    pixels = rgb.reshape(-1, 3)

    return [
        *pixels.mean(axis=0),
        *pixels.std(axis=0),
        *np.quantile(pixels, 0.10, axis=0),
        *np.quantile(pixels, 0.90, axis=0),
    ]


# ============================================================
# 3. HSV STATISTICS (4 FEATURES)
# ============================================================

def extract_hsv_statistics(rgb):
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)

    saturation = (
        (maximum - minimum)
        / np.maximum(maximum, 1e-8)
    )

    value = maximum

    return [
        saturation.mean(),
        saturation.std(),
        value.mean(),
        value.std(),
    ]


# ============================================================
# 4. GRAYSCALE
# ============================================================

def rgb_to_grayscale(rgb):
    return (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )


# ============================================================
# 5. HISTOGRAM ENTROPY (1 FEATURE)
# ============================================================

def extract_entropy(gray):
    hist, _ = np.histogram(
        gray,
        bins=64,
        range=(0, 1),
    )

    probabilities = hist / max(hist.sum(), 1)
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return entropy


# ============================================================
# 6. FREQUENCY STATISTICS (6 FEATURES)
# ============================================================

def extract_frequency_statistics(gray):
    # Sobel gradient
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)

    gradient = np.hypot(gx, gy)

    # Laplacian
    lap = laplace(gray)

    # Gaussian residual
    blurred = gaussian_filter(gray, sigma=1.2)
    residual = gray - blurred

    return [
        gradient.mean(),
        gradient.std(),
        lap.std(),
        np.abs(lap).mean(),
        residual.std(),
        np.abs(residual).mean(),
    ]


# ============================================================
# 7. JPEG BLOCK BOUNDARY (2 FEATURES)
# ============================================================

def extract_block_statistics(gray):
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))

    block_x = dx[:, 7::8]
    block_y = dy[7::8, :]

    return [
        block_x.mean() if block_x.size else 0.0,
        block_y.mean() if block_y.size else 0.0,
    ]


# ============================================================
# 8. SPATIAL STATISTICS (4 FEATURES)
# ============================================================

def extract_spatial_statistics(gray):
    height, width = gray.shape

    y0, y1 = int(height * 0.2), int(height * 0.8)
    x0, x1 = int(width * 0.2), int(width * 0.8)

    center = gray[y0:y1, x0:x1]

    mask = np.ones(gray.shape, dtype=bool)
    mask[y0:y1, x0:x1] = False

    border = gray[mask]

    if center.size == 0 or border.size == 0:
        raise ValueError(
            "Image is too small for spatial features"
        )

    return [
        center.mean(),
        center.std(),
        border.mean(),
        border.std(),
    ]


# ============================================================
# 9. BASELINE RGB FEATURES (6 FEATURES)
# ============================================================

def extract_rgb_features(image_path):
    """Extract RGB mean and std on the original [0, 255] scale."""
    with Image.open(image_path) as img:
        image = np.asarray(
            img.convert("RGB"),
            dtype=np.float32,
        )

    rgb_mean = image.mean(axis=(0, 1))
    rgb_std = image.std(axis=(0, 1))

    return np.concatenate([
        rgb_mean,
        rgb_std,
    ])


# ============================================================
# 10. FULL IMAGE FEATURES (32 FEATURES)
# ============================================================


def extract_image_features(image_path):
    image_path = Path(image_path)

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(
            f"Cannot load image: {image_path}"
        )

    rgb = cv2.cvtColor(
        bgr, cv2.COLOR_BGR2RGB
    ).astype(np.float32) / 255.0

    gray = cv2.cvtColor(
        bgr, cv2.COLOR_BGR2GRAY
    ).astype(np.float32) / 255.0

    hsv = cv2.cvtColor(
        bgr, cv2.COLOR_BGR2HSV
    ).astype(np.float32) / 255.0

    features = [
        float(np.log1p(image_path.stat().st_size))
    ]

    # RGB: 12
    features.extend(extract_rgb_statistics(rgb))

    # HSV: 4
    sat = hsv[..., 1]
    val = hsv[..., 2]
    features.extend([
        sat.mean(), sat.std(),
        val.mean(), val.std(),
    ])

    # Grayscale + entropy: 3
    features.extend([
        gray.mean(),
        gray.std(),
        extract_entropy(gray),
    ])

    # Sobel + Laplacian: 4
    gx = cv2.Sobel(
        gray, cv2.CV_32F, 1, 0, ksize=3
    )
    gy = cv2.Sobel(
        gray, cv2.CV_32F, 0, 1, ksize=3
    )

    gradient = np.sqrt(gx * gx + gy * gy)

    lap = cv2.Laplacian(gray, cv2.CV_32F)

    features.extend([
        gradient.mean(),
        gradient.std(),
        lap.std(),
        np.abs(lap).mean(),
    ])

    # Gaussian residual: 2
    blurred = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=1.2, sigmaY=1.2
    )
    residual = gray - blurred

    features.extend([
        residual.std(),
        np.abs(residual).mean(),
    ])

    # JPEG block: 2
    features.extend(extract_block_statistics(gray))

    # Center/border: 4
    features.extend(extract_spatial_statistics(gray))

    result = np.asarray(features, dtype=np.float64)

    if result.shape != (32,):
        raise ValueError(
            f"Expected 32 features, got {result.shape}"
        )

    if not np.isfinite(result).all():
        raise ValueError(
            "Non-finite image features detected"
        )

    return result



# ============================================================
# 11. PAIR FEATURES (32 FEATURES)
# ============================================================

def extract_pair_features(row, image_path):
    image_path = Path(image_path)

    features_0 = extract_image_features(
        image_path / row["image_0"]
    )

    features_1 = extract_image_features(
        image_path / row["image_1"]
    )

    return features_1 - features_0


# ============================================================
# 12. JPEG FILE SIZE DIFFERENCE (1 FEATURE)
# ============================================================

def extract_jpeg_feature(row, image_path):
    image_path = Path(image_path)

    image_0_path = image_path / row["image_0"]
    image_1_path = image_path / row["image_1"]

    log_size_0 = np.log1p(
        image_0_path.stat().st_size
    )

    log_size_1 = np.log1p(
        image_1_path.stat().st_size
    )

    return log_size_1 - log_size_0


# ============================================================
# 13. BASELINE RGB + JPEG (7 FEATURES)
# ============================================================

def extract_combine_feature(row, image_path):
    image_path = Path(image_path)

    rgb_0 = extract_rgb_features(
        image_path / row["image_0"]
    )

    rgb_1 = extract_rgb_features(
        image_path / row["image_1"]
    )

    rgb_difference = rgb_1 - rgb_0

    jpeg_difference = extract_jpeg_feature(
        row, image_path
    )

    combined_features = np.concatenate([
        rgb_difference,
        np.array([jpeg_difference]),
    ])

    return combined_features
