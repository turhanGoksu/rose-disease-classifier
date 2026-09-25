"""Model setup: pretrained ResNet18 with a new two-class head."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.dataset import CLASS_NAMES

UNFREEZABLE_LAYERS = ("layer1", "layer2", "layer3", "layer4")


def build_model(
    num_classes: int = len(CLASS_NAMES),
    trainable_layers: tuple[str, ...] = ("layer4",),
    pretrained: bool = True,
) -> nn.Module:
    """Load ImageNet-pretrained ResNet18 and replace its final layer.

    Two outputs (one logit per class) with CrossEntropyLoss, not one output
    with BCEWithLogitsLoss: softmax over two logits depends only on their
    difference, so both can learn the same thing. Two logits keep the 2A
    code, give CrossEntropyLoss a per-class weight in step 7, and give
    Grad-CAM one score per class to explain in step 8.

    Args:
        num_classes: Number of output classes for the new head.
        trainable_layers: Backbone stages to keep trainable. The default,
            ("layer4",), was the best setup in Project 2A.
        pretrained: Download ImageNet weights. Use False when the weights
            will be overwritten by a checkpoint anyway.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    # 1) Freeze every parameter that exists right now (the whole backbone).
    for param in model.parameters():
        param.requires_grad = False

    # 2) Optionally switch selected stages back on for partial fine-tuning.
    for name in trainable_layers:
        if name not in UNFREEZABLE_LAYERS:
            raise ValueError(f"Unknown layer {name!r}; "
                             f"choose from {UNFREEZABLE_LAYERS}")
        for param in getattr(model, name).parameters():
            param.requires_grad = True

    # 3) Replace the head AFTER freezing: new parameters are created with
    #    requires_grad=True, so the head is always trainable.
    in_features = model.fc.in_features  # 512 for ResNet18
    model.fc = nn.Linear(in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def main() -> None:
    """Smoke test: forward pass, and softmax depends only on differences."""
    model = build_model()
    trainable, total = count_parameters(model)
    print(f"trainable params: {trainable:,} / {total:,}")

    model.eval()  # BatchNorm uses running stats: deterministic outputs.
    with torch.no_grad():
        logits = model(torch.randn(4, 3, 224, 224))  # Fake batch [B, C, H, W].
    print(f"logits shape: {tuple(logits.shape)}  (batch, {CLASS_NAMES})")

    # Shifting both logits by the same amount leaves the probabilities alone.
    pairs = torch.tensor([[2.0, 4.0], [10.0, 12.0], [7.0, 5.0]])
    for pair, probs in zip(pairs, pairs.softmax(dim=1)):
        print(f"logits {pair.tolist()} -> p(Black_Spot) = {probs[1]:.2f}")


if __name__ == "__main__":
    main()
