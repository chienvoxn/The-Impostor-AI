
import argparse
from pathlib import Path

import torch
from PIL import Image

from kemaodanh_oai.models import build_densenet121
from kemaodanh_oai.transforms import build_valid_transform


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"


def load_models(device):
    models = []

    for fold in range(3):
        checkpoint_path = (
            CHECKPOINT_DIR / f"densenet121_fold{fold}_best.pt"
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint: {checkpoint_path}"
            )

        # pretrained=False vì đã có trọng số fine-tuned.
        model = build_densenet121(
            pretrained=False,
            num_classes=2,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        models.append(model)
        print(f"Loaded fold {fold}")

    return models


@torch.inference_mode()
def predict_fake_probability(models, image_path, transform, device):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    probabilities = []

    for model in models:
        logits = model(tensor)
        prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
        probabilities.append(prob_fake)

    return sum(probabilities) / len(probabilities)


def main():
    parser = argparse.ArgumentParser(
        description="Dự đoán ảnh giả trong một cặp ảnh."
    )

    parser.add_argument("--image0", required=True)
    parser.add_argument("--image1", required=True)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    transform = build_valid_transform(
        image_size=224,
        crop_fraction=0.60,
    )

    models = load_models(device)

    prob_fake_0 = predict_fake_probability(
        models, args.image0, transform, device
    )

    prob_fake_1 = predict_fake_probability(
        models, args.image1, transform, device
    )

    fake_position = int(prob_fake_1 > prob_fake_0)

    print("\n========== RESULT ==========")
    print(f"Image 0 - P(fake): {prob_fake_0:.6f}")
    print(f"Image 1 - P(fake): {prob_fake_1:.6f}")
    print(f"Predicted fake_position: {fake_position}")
    print(f"Ảnh được dự đoán là giả: image_{fake_position}")


if __name__ == "__main__":
    main()
