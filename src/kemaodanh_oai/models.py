import torch.nn as nn

from torchvision.models import (
    densenet121,
    DenseNet121_Weights,
)


def build_densenet121(
    pretrained: bool = True,
    num_classes: int = 2,
) -> nn.Module:
    """
    Xây dựng DenseNet121 để phân loại ảnh thật/giả.

    Nhãn:
        0 = ảnh thật
        1 = ảnh giả

    pretrained=True:
        Khởi tạo backbone bằng trọng số ImageNet.

    Toàn bộ tham số được phép fine-tune.
    """

    weights = (
        DenseNet121_Weights.IMAGENET1K_V1
        if pretrained
        else None
    )

    model = densenet121(weights=weights)

    # DenseNet121 có classifier đầu ra 1000 lớp.
    # Thay bằng classifier 2 lớp.
    in_features = model.classifier.in_features

    model.classifier = nn.Linear(
        in_features,
        num_classes,
    )

    # Fine-tune toàn bộ mạng
    for param in model.parameters():
        param.requires_grad = True

    return model