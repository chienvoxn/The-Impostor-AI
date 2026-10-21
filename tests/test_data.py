import pandas as pd
import pytest

from kemaodanh_oai.data import load_pairs


def test_duplicate_pair_id(tmp_path):
    df = pd.DataFrame({
        "pair_id": ["001", "001"],
        "image_0": ["a.jpg", "b.jpg"],
        "image_1": ["c.jpg", "d.jpg"],
        "fake_position": [0, 1]
    })

    csv_path = tmp_path / "pairs.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_pairs(csv_path)