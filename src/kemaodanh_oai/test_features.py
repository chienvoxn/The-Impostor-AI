import numpy as np
from PIL import Image


def test_full_feature_dimension(tmp_path):
    from kemaodanh_oai.features import (
        FEATURE_NAMES,
        extract_image_features,
    )

    rng = np.random.default_rng(42)

    image = rng.integers(
        0, 256,
        size=(512, 512, 3),
        dtype=np.uint8,
    )

    path = tmp_path / "test.jpg"
    Image.fromarray(image).save(path)

    features = extract_image_features(path)

    assert len(FEATURE_NAMES) == 32
    assert features.shape == (32,)
    assert np.isfinite(features).all()


def test_jpeg_feature_matches_full_features(tmp_path):
    from kemaodanh_oai.features import (
        extract_pair_features,
        extract_jpeg_feature,
    )

    rng = np.random.default_rng(42)

    for name in ["a.jpg", "b.jpg"]:
        image = rng.integers(
            0, 256,
            size=(64, 64, 3),
            dtype=np.uint8,
        )
        Image.fromarray(image).save(tmp_path / name)

    pair = {
        "image_0": "a.jpg",
        "image_1": "b.jpg",
    }

    full_features = extract_pair_features(
        pair, tmp_path
    )

    jpeg_feature = extract_jpeg_feature(
        pair, tmp_path
    )

    np.testing.assert_allclose(
        full_features[0],
        jpeg_feature,
        rtol=1e-10,
        atol=1e-10,
    )


def test_pair_features_antisymmetric(tmp_path):
    from kemaodanh_oai.features import extract_pair_features

    rng = np.random.default_rng(42)

    for name in ["a.jpg", "b.jpg"]:
        image = rng.integers(
            0, 256,
            size=(64, 64, 3),
            dtype=np.uint8,
        )
        Image.fromarray(image).save(tmp_path / name)

    pair_ab = {"image_0": "a.jpg", "image_1": "b.jpg"}
    pair_ba = {"image_0": "b.jpg", "image_1": "a.jpg"}

    features_ab = extract_pair_features(pair_ab, tmp_path)
    features_ba = extract_pair_features(pair_ba, tmp_path)

    np.testing.assert_allclose(
        features_ab,
        -features_ba,
        rtol=1e-10,
        atol=1e-10,
    )
