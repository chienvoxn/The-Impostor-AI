# RGB Statistics

from pathlib import Path

import numpy as np
from PIL import Image

def extract_rgb_features(image_path):
    with Image.open(image_path) as img:
        image = np.asarray(img.convert("RGB"), dtype=np.float32)

    rgb_mean = np.mean(image, axis=(0, 1))
    rgb_std = np.std(image, axis=(0, 1))

    features = np.concatenate([rgb_mean, rgb_std])

    return features

def extract_pair_features(row, image_path):
    image_path = Path(image_path)

    img_0_path = image_path / row["image_0"]
    img_1_path = image_path / row["image_1"]

    features_0 = extract_rgb_features(img_0_path)
    features_1 = extract_rgb_features(img_1_path)

    pair_features = features_0 - features_1

    return pair_features

def extract_jpeg_feature(row, image_path):
    image_path = Path(image_path)

    image_0_path = image_path / row['image_0']
    image_1_path = image_path / row['image_1']

    log_size_0 = image_0_path.stat().st_size
    log_size_1 = image_1_path.stat().st_size

    return log_size_0 - log_size_1

def extract_combine_feature(row, image_path):
    rgb_features = extract_pair_features(row, image_path)
    jpeg_feature = extract_jpeg_feature(row, image_path)

    combined_features = np.concatenate([rgb_features, np.array([jpeg_feature])])

    return combined_features
