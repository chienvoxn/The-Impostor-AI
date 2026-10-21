from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CenterCropFraction:
    """
    Crop vùng trung tâm theo tỷ lệ chiều rộng và chiều cao.

    Ví dụ:
        fraction=0.60 -> giữ 60% chiều rộng
                         và 60% chiều cao.
    """

    def __init__(self, fraction: float = 0.60):
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")

        self.fraction = fraction

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size

        crop_width = max(1, round(width * self.fraction))
        crop_height = max(1, round(height * self.fraction))

        left = (width - crop_width) // 2
        top = (height - crop_height) // 2

        return image.crop(
            (
                left,
                top,
                left + crop_width,
                top + crop_height,
            )
        )


def build_train_transform(
    image_size: int = 224,
    crop_fraction: float = 0.60,
):
    return transforms.Compose(
        [
            CenterCropFraction(crop_fraction),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def build_valid_transform(
    image_size: int = 224,
    crop_fraction: float = 0.60,
):
    return transforms.Compose(
        [
            CenterCropFraction(crop_fraction),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )