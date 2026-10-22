import torch
import torch.nn.functional as F

from tqdm.auto import tqdm

import pandas as pd
from sklearn.metrics import f1_score


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    accumulation_steps=3,
):
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_samples = 0

    use_amp = device.type == "cuda"

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    # Số mẫu trong nhóm accumulation hiện tại
    group_samples = 0

    for step, (images, labels) in enumerate(progress):
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        batch_size = images.size(0)

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Cộng gradient theo tổng loss của từng mẫu.
        # Cuối nhóm sẽ chia cho tổng số mẫu thực tế.
        scaled_loss = loss * batch_size

        scaler.scale(scaled_loss).backward()

        group_samples += batch_size

        is_update_step = (
            (step + 1) % accumulation_steps == 0
            or (step + 1) == len(loader)
        )

        if is_update_step:
            # Chuyển gradient về trung bình theo số mẫu.
            # Chia trước scaler.step để hỗ trợ AMP.
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(group_samples)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            group_samples = 0

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / total_samples


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_probs = []
    all_labels = []

    use_amp = device.type == "cuda"

    for images, labels in tqdm(
        loader,
        desc="Validation",
        leave=False,
    ):
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        probs = torch.softmax(
            logits.float(),
            dim=1,
        )[:, 1]

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    probabilities = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    return (
        total_loss / total_samples,
        probabilities,
        labels,
    )

def evaluate_pair_predictions(valid_df, probabilities, dev_df):
    predictions = valid_df[
        ["pair_id", "position"]
    ].copy()

    if len(predictions) != len(probabilities):
        raise ValueError("Prediction count mismatch")

    predictions["fake_probability"] = probabilities

    pair_probs = predictions.pivot(
        index="pair_id",
        columns="position",
        values="fake_probability",
    )

    targets = (
        dev_df.set_index("pair_id")["fake_position"]
        .astype(int)
        .rename("true_fake_position")
    )

    results = pair_probs.rename(
        columns={
            0: "prob_fake_0",
            1: "prob_fake_1",
        }
    ).join(targets, how="inner", validate="one_to_one")

    if results.isna().any().any():
        raise ValueError("Missing pair predictions or labels")

    results["pred_fake_position"] = (
        results["prob_fake_1"] > results["prob_fake_0"]
    ).astype(int)

    score = f1_score(
        results["true_fake_position"],
        results["pred_fake_position"],
        average="macro",
    )

    return score, results.reset_index()