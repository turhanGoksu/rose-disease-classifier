"""Save and load model checkpoints as plain state_dicts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.dataset import CLASS_NAMES
from src.model import build_model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    trainable_layers: tuple[str, ...],
    epoch: int,
    val_loss: float,
    val_macro_f1: float,
    config: dict[str, Any],
) -> None:
    """Save everything needed to rebuild the model and read its output."""
    torch.save({
        # All parameters AND buffers (BatchNorm running_mean/var).
        "model_state": model.state_dict(),
        # Output index -> class name. The order is fixed (Black_Spot = 1),
        # so a checkpoint must never be read with a different order.
        "class_names": list(CLASS_NAMES),
        # Needed to rebuild the same architecture before loading weights.
        "trainable_layers": list(trainable_layers),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_macro_f1": val_macro_f1,
        "config": config,  # The run's arguments, for reproducibility.
    }, path)


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model from a checkpoint, ready for inference.

    Returns:
        (model in eval mode on `device`, the raw checkpoint dict).
    """
    # map_location: a checkpoint saved on a CUDA GPU can be opened on a Mac.
    # weights_only: refuse arbitrary pickled objects, which could run code.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if tuple(checkpoint["class_names"]) != CLASS_NAMES:
        raise ValueError(f"Checkpoint class order {checkpoint['class_names']}"
                         f" does not match {CLASS_NAMES}")
    model = build_model(
        num_classes=len(CLASS_NAMES),
        trainable_layers=tuple(checkpoint["trainable_layers"]),
        pretrained=False,  # Weights come from the checkpoint, skip download.
    )
    model.load_state_dict(checkpoint["model_state"])  # strict=True by default.
    model.to(device)
    model.eval()
    return model, checkpoint
