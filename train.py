"""Entry point: train the rose disease classifier with a plain PyTorch loop.

The best epoch is chosen on val macro-F1 only. The test split is evaluated
once, with the best checkpoint, after training.

Usage:
    python train.py --data-dir "~/Desktop/final dataset/Dataset/Rose"
    python train.py --data-dir /kaggle/input/<dataset> \
        --output-dir /kaggle/working/outputs --balance sampler
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from src.checkpoint import load_checkpoint, save_checkpoint
from src.dataset import (CLASS_NAMES, DEFAULT_SPLIT_FILE, POSITIVE_CLASS,
                         Sample, balanced_class_weights, balanced_sampler,
                         build_dataloaders, class_counts)
from src.engine import (EvalResult, build_optimizer, evaluate, get_device,
                        train_one_epoch)
from src.metrics import format_report, subset_metrics
from src.model import UNFREEZABLE_LAYERS, build_model
from src.plotting import History, plot_confusion_matrices, plot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="The Rose folder with one sub-folder per class.")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--balance", choices=("none", "sampler", "loss"),
                        default="none",
                        help="Class imbalance handling: none, balanced "
                             "batches (WeightedRandomSampler), or class "
                             "weights in the loss.")
    parser.add_argument("--run-name", default=None,
                        help="Sub-folder of --output-dir. Default: balance.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate of the new classification head.")
    parser.add_argument("--backbone-lr", type=float, default=1e-4,
                        help="Learning rate of unfrozen backbone stages.")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trainable-layers", nargs="*", default=["layer4"],
                        choices=UNFREEZABLE_LAYERS,
                        help="Backbone stages to fine-tune.")
    return parser.parse_args()


def save_predictions(
    path: Path, result: EvalResult, samples: list[Sample], split: str
) -> None:
    """One row per image: lets us pick right/wrong examples for Grad-CAM."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "path", "source", "true", "pred",
                         "p_black_spot"])
        for sample, true, pred, prob in zip(samples, result.y_true,
                                            result.y_pred, result.p_positive):
            writer.writerow([split, sample.path.as_posix(), sample.source,
                             CLASS_NAMES[true], CLASS_NAMES[pred],
                             f"{prob:.4f}"])


def preview_sampler_epoch(samples: list[Sample], seed: int) -> str:
    """Describe what one epoch of balanced sampling draws, per class.

    Uses its own sampler (seed + 1), so the real training order is not
    affected by the preview.
    """
    draws: Counter[int] = Counter()
    distinct: dict[int, set[int]] = {c: set() for c in range(len(CLASS_NAMES))}
    for index in balanced_sampler(samples, seed + 1):
        label = samples[index].label
        draws[label] += 1
        distinct[label].add(index)
    totals = class_counts(samples)
    return ", ".join(
        f"{name}: {draws[c]} draws of {len(distinct[c])}/{totals[c]} "
        f"distinct images" for c, name in enumerate(CLASS_NAMES))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)  # Same head init and shuffle order each run.
    run_name = args.run_name or args.balance
    output_dir = args.output_dir.expanduser() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}

    device = get_device()
    loaders = build_dataloaders(
        args.data_dir.expanduser(), args.split_file,
        batch_size=args.batch_size, num_workers=args.num_workers,
        balanced_sampling=args.balance == "sampler", seed=args.seed,
    )
    train_samples = loaders["train"].dataset.samples
    val_sources = [s.source for s in loaders["val"].dataset.samples]
    model = build_model(trainable_layers=tuple(args.trainable_layers))
    model = model.to(device)  # Move weights before creating the optimizer.
    # Class weights only change the TRAINING loss. Val/test loss stays
    # unweighted, so it means the same thing in every run.
    class_weights = (balanced_class_weights(train_samples).to(device)
                     if args.balance == "loss" else None)
    train_criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, head_lr=args.lr,
                                backbone_lr=args.backbone_lr,
                                weight_decay=args.weight_decay)

    print(f"run={run_name} device={device} balance={args.balance} "
          f"train={len(train_samples)} val={len(loaders['val'].dataset)} "
          f"test={len(loaders['test'].dataset)}")
    print(f"train images per class: "
          f"{dict(zip(CLASS_NAMES, class_counts(train_samples)))}")
    if args.balance == "sampler":
        print(f"one sampler epoch (preview): "
              f"{preview_sampler_epoch(train_samples, args.seed)}")
    if class_weights is not None:
        print(f"loss class weights: "
              f"{dict(zip(CLASS_NAMES, class_weights.tolist()))}")

    history: History = {key: [] for key in (
        "train_loss", "train_acc", "val_loss", "val_macro_f1",
        "val_same_source_macro_f1", "val_black_spot_recall")}
    checkpoint_path = output_dir / "best.pt"
    best_f1, best_loss, best_epoch = -1.0, float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], train_criterion, optimizer, device)
        val = evaluate(model, loaders["val"], criterion, device)
        metrics = subset_metrics(val.y_true, val.y_pred, val_sources)
        val_f1 = metrics["overall"]["macro_f1"]
        bs_recall = (metrics["overall"]["per_class"]
                     [CLASS_NAMES[POSITIVE_CLASS]]["recall"])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val.loss)
        history["val_macro_f1"].append(val_f1)
        history["val_same_source_macro_f1"].append(
            metrics["same_source"]["macro_f1"])
        history["val_black_spot_recall"].append(bs_recall)

        # Macro-F1 is what we care about. With 80 Black_Spot images in val it
        # moves in coarse steps, so ties go to the lower val loss.
        is_best = val_f1 > best_f1 or (val_f1 == best_f1
                                       and val.loss < best_loss)
        if is_best:
            best_f1, best_loss, best_epoch = val_f1, val.loss, epoch
            save_checkpoint(checkpoint_path, model,
                            tuple(args.trainable_layers), epoch, val.loss,
                            val_f1, config)
        print(f"epoch {epoch:>3}/{args.epochs} | "
              f"train loss {train_loss:.3f} acc {train_acc:.1%} | "
              f"val loss {val.loss:.3f} macro-F1 {val_f1:.3f} "
              f"same-source {metrics['same_source']['macro_f1']:.3f} "
              f"BS recall {bs_recall:.1%}"
              f"{'  <- saved best' if is_best else ''}")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, output_dir / "curves.png", best_epoch=best_epoch)

    # Rebuild the best model from disk: checks the checkpoint round-trip
    # and gives the numbers we report. Test is touched only here, once.
    best_model, checkpoint = load_checkpoint(checkpoint_path, device)
    report = {"run": run_name, "best_epoch": checkpoint["epoch"],
              "config": config}
    for split in ("val", "test"):
        result = evaluate(best_model, loaders[split], criterion, device)
        samples = loaders[split].dataset.samples
        metrics = subset_metrics(result.y_true, result.y_pred,
                                 [s.source for s in samples])
        report[split] = {"loss": result.loss, "metrics": metrics}
        save_predictions(output_dir / f"predictions_{split}.csv", result,
                         samples, split)
        plot_confusion_matrices(
            metrics, CLASS_NAMES, ("overall", "same_source"),
            f"{run_name}: {split} set, best epoch {checkpoint['epoch']}",
            output_dir / f"confusion_{split}.png")
        print(f"\n{split} (best checkpoint, epoch {checkpoint['epoch']}):")
        print(format_report(metrics))

    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved checkpoint, history, curves, confusion matrices, "
          f"predictions and metrics.json to {output_dir}")


if __name__ == "__main__":  # Required for DataLoader workers on macOS.
    main()
