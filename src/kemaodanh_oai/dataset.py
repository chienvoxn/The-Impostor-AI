from pathlib import Path

import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset


def load_development_data(
    train_dir: str | Path,
    split_csv: str | Path,
) -> pd.DataFrame:
    """Đọc và ghép dữ liệu development theo pair_id."""

    train_dir = Path(train_dir)
    split_csv = Path(split_csv)

    pairs_df = pd.read_csv(
        train_dir / "pairs.csv",
        dtype={"pair_id": str},
    )

    split_df = pd.read_csv(
        split_csv,
        dtype={"pair_id": str},
    )

    merged_df = pairs_df.merge(
        split_df,
        on="pair_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged_df) != len(pairs_df):
        raise ValueError(
            "Some pairs are missing from development_split.csv"
        )

    dev_df = (
        merged_df[merged_df["split"] == "development"]
        .sort_values("pair_id")
        .reset_index(drop=True)
    )

    return dev_df


def make_image_dataframe(
    dev_df: pd.DataFrame,
    train_dir: str | Path,
) -> pd.DataFrame:
    """Chuyển pair-level data thành image-level data."""

    train_dir = Path(train_dir)
    records = []

    for row in dev_df.itertuples(index=False):
        records.append({
            "pair_id": row.pair_id,
            "image_path": str(train_dir / row.image_0),
            "label": int(row.fake_position == 0),
            "fold": int(row.fold),
            "position": 0,
        })

        records.append({
            "pair_id": row.pair_id,
            "image_path": str(train_dir / row.image_1),
            "label": int(row.fake_position == 1),
            "fold": int(row.fold),
            "position": 1,
        })

    return pd.DataFrame(records)


class FaceClassificationDataset(Dataset):
    """Dataset phân loại từng ảnh thật/giả."""

    def __init__(self, dataframe: pd.DataFrame, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        with Image.open(row["image_path"]) as img:
            image = self.transform(img.convert("RGB"))

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long,
        )

        return image, label