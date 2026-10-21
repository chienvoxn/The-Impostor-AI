from pathlib import Path
import pandas as pd

def load_pairs(csv_path: str) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, dtype={"pair_id":str})

    required_columns = {"pair_id", "image_0", "image_1", "fake_position"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing values: {missing_columns}")
    
    if df["pair_id"].isna().any():
        raise ValueError("Pair id contains missing values")

    if df["pair_id"].duplicated().any():
        raise ValueError("Pair id contains duplicated values")

    valid_labels = {0, 1}

    labels = pd.to_numeric(df["fake_position"], errors="coerce")

    invalid = labels.notna() & ~labels.isin(valid_labels)

    if invalid.any():
        raise ValueError("Invalid fake position labels")

    image_dir = csv_path.parent

    for column in ["image_0", "image_1"]:
        for image_name in df[column]:
            if pd.isna(image_name):
                raise ValueError(f"Missing image name in {column}")

            image_path = image_dir / str(image_name)

            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")

    return df
