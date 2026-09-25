"""Classification metrics over a whole evaluation set, overall and per source.

Why not accuracy alone: with ~74% Healthy images, a model that answers
"Healthy" every time already scores ~74% accuracy while catching no disease.
Macro-F1 gives each class an equal vote, so a weak Black_Spot class cannot
hide behind a strong Healthy class.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from src.dataset import CLASS_NAMES

# Evaluation subsets by image source (see docs/data_audit.md).
#   same_source: both classes come from the same 1024px "resized_" sources,
#                so background and resolution cannot give the answer away.
#   studio: only white-background Healthy images, the shortcut-prone part.
SUBSETS: dict[str, tuple[str, ...]] = {
    "overall": ("studio", "fresh_leaf", "black_spot"),
    "same_source": ("fresh_leaf", "black_spot"),
    "studio": ("studio",),
}


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class precision/recall/F1, confusion matrix.

    Macro-F1 is None when a class has no images in `y_true` (e.g. the studio
    subset): the F1 of an absent class is undefined, not zero.
    """
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    has_all_classes = bool((support > 0).all())
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()) if has_all_classes else None,
        "per_class": {
            name: {"precision": float(precision[i]),
                   "recall": float(recall[i]),
                   "f1": float(f1[i]),
                   "support": int(support[i])}
            for i, name in enumerate(CLASS_NAMES)
        },
        # Rows = true class, columns = predicted class, in CLASS_NAMES order.
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels).tolist(),
    }


def subset_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, sources: list[str]
) -> dict[str, dict[str, Any]]:
    """classification_metrics() for every subset in SUBSETS."""
    source_array = np.array(sources)
    results = {}
    for name, allowed in SUBSETS.items():
        mask = np.isin(source_array, allowed)
        results[name] = classification_metrics(y_true[mask], y_pred[mask])
    return results


def format_report(metrics: dict[str, dict[str, Any]]) -> str:
    """Human-readable summary of subset_metrics() output."""
    lines = []
    for subset, m in metrics.items():
        f1 = "  n/a" if m["macro_f1"] is None else f"{m['macro_f1']:.3f}"
        lines.append(f"  {subset:<12} n={m['n']:>4}  acc={m['accuracy']:.3f}"
                     f"  macro-F1={f1}")
        for name, c in m["per_class"].items():
            if c["support"]:
                lines.append(f"      {name:<16} precision={c['precision']:.3f}"
                             f"  recall={c['recall']:.3f}  f1={c['f1']:.3f}"
                             f"  (n={c['support']})")
        lines.append(f"      confusion matrix (rows=true, cols=pred): "
                     f"{m['confusion_matrix']}")
    return "\n".join(lines)
