"""Training engine: optimizer setup and the manually written epoch loops."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import POSITIVE_CLASS


@dataclass
class EvalResult:
    """Everything one pass over an evaluation set produced, in loader order."""

    loss: float           # Mean loss per sample.
    y_true: np.ndarray    # [N] true class indices.
    y_pred: np.ndarray    # [N] predicted class indices (argmax).
    p_positive: np.ndarray  # [N] softmax probability of Black_Spot.


def build_optimizer(
    model: nn.Module,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    weight_decay: float = 1e-2,
) -> torch.optim.Optimizer:
    """Create AdamW with one parameter group per learning rate.

    The new head starts from random weights and needs large steps. Unfrozen
    backbone stages are already pretrained, so large steps would destroy
    their features; they get a smaller learning rate. Frozen parameters
    (requires_grad=False) are left out entirely.
    """
    head_params = [p for p in model.fc.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_ids]

    param_groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:  # Empty when the whole backbone is frozen.
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def get_device() -> torch.device:
    """Pick CUDA (Kaggle), then Apple MPS (local Mac), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one pass over the training data and update the weights.

    Returns:
        (mean loss per sample, accuracy) over the epoch. Accuracy is only a
        rough progress signal here; model quality is judged on val metrics.
    """
    model.train()  # Training-mode behavior for BatchNorm/Dropout.
    total_loss, total_correct, total_seen = 0.0, 0, 0

    for images, labels in loader:
        # non_blocking lets the copy overlap with compute when pin_memory=True.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()             # 1. Clear gradients from last step.
        logits = model(images)            # 2. Forward pass: [B, num_classes].
        loss = criterion(logits, labels)  # 3. Mean loss over the batch.
        loss.backward()                   # 4. Autograd adds dLoss/dW to .grad.
        optimizer.step()                  # 5. W <- W - update(W.grad).

        # .item() returns a Python float, so the graph is not kept alive.
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()  # No graph, no saved activations: less memory, faster.
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvalResult:
    """Run the model over a loader and keep every prediction.

    Unlike accuracy, F1 cannot be averaged over batches (a batch without
    Black_Spot images has no Black_Spot recall at all), so we collect all
    predictions and compute metrics once over the whole set.
    """
    model.eval()  # BatchNorm uses running stats and stops updating them.
    total_loss, total_seen = 0.0, 0
    y_true, y_pred, p_positive = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)            # Forward only.
        loss = criterion(logits, labels)  # No backward(), no step().

        total_loss += loss.item() * labels.size(0)
        total_seen += labels.size(0)
        y_true.append(labels.cpu())
        y_pred.append(logits.argmax(dim=1).cpu())
        p_positive.append(logits.softmax(dim=1)[:, POSITIVE_CLASS].cpu())

    return EvalResult(
        loss=total_loss / total_seen,
        y_true=torch.cat(y_true).numpy(),
        y_pred=torch.cat(y_pred).numpy(),
        p_positive=torch.cat(p_positive).numpy(),
    )
