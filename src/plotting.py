"""Training curves and confusion matrices as PNG files.

Uses the object-oriented Figure API instead of pyplot, so it works without a
display (Kaggle, SSH) and keeps no global plotting state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

# Categorical slots 1-5 in fixed order (validated colorblind-safe for
# adjacent pairs). Slots 3-5 are below 3:1 contrast on the surface, so every
# line is also direct-labeled and has its own line style.
SERIES_STYLE = {
    "train_loss": {"color": "#2a78d6", "linestyle": "-", "marker": "o",
                   "label": "train"},
    "val_loss": {"color": "#eb6834", "linestyle": "--", "marker": "s",
                 "label": "val"},
    "val_macro_f1": {"color": "#1baf7a", "linestyle": "-", "marker": "o",
                     "label": "macro-F1"},
    "val_same_source_macro_f1": {"color": "#eda100", "linestyle": "-.",
                                 "marker": "s",
                                 "label": "same-source macro-F1"},
    "val_black_spot_recall": {"color": "#e87ba4", "linestyle": "--",
                              "marker": "^", "label": "Black_Spot recall"},
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
# Sequential single-hue ramp (light -> dark blue) for confusion matrices.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

History = dict[str, list[float]]


def plot_history(
    history: History, output_path: Path, best_epoch: int | None = None
) -> None:
    """Save loss curves (train vs val) and val metric curves side by side.

    If best_epoch is given, a vertical marker shows the checkpointed epoch.
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = Figure(figsize=(11, 4), facecolor=SURFACE)
    ax_loss, ax_metric = fig.subplots(1, 2)

    panels = (
        (ax_loss, ("train_loss", "val_loss"), "Cross-entropy loss"),
        (ax_metric, ("val_macro_f1", "val_same_source_macro_f1",
                     "val_black_spot_recall"), "Validation metrics"),
    )
    for ax, keys, title in panels:
        ends = {}
        for key in keys:
            style = dict(SERIES_STYLE[key])
            label = style.pop("label")
            ax.plot(epochs, history[key], linewidth=2, markersize=6,
                    label=label, **style)
            ends[label] = history[key][-1]
        _style_axes(ax, title)
        if ax is ax_metric:
            ax.set_ylim(top=1.005)  # Scores end at 1; keep markers whole.
        _direct_labels(ax, epochs[-1], ends)  # After the y-limits are set.
        ax.legend(frameon=False, labelcolor=INK, fontsize=9)
        if best_epoch is not None:
            ax.axvline(best_epoch, color=MUTED, linewidth=1, linestyle=":")

    if best_epoch is not None:
        # Caption below the plots, so it never collides with the curves.
        fig.text(0.01, 0.01, f"Dotted line: best checkpoint "
                 f"(highest val macro-F1, epoch {best_epoch})",
                 color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)


def plot_confusion_matrices(
    metrics: dict[str, dict[str, Any]],
    class_names: tuple[str, ...],
    subsets: tuple[str, ...],
    title: str,
    output_path: Path,
) -> None:
    """Save one confusion matrix per subset, side by side.

    Cell color is the share of the TRUE class (row-normalized), so panels
    with different sizes share one color scale; the text gives the count
    and that share.
    """
    fig = Figure(figsize=(4.2 * len(subsets), 4.2), facecolor=SURFACE)
    axes = np.atleast_1d(fig.subplots(1, len(subsets)))
    short = [name.replace("Rose_", "") for name in class_names]

    for ax, subset in zip(axes, subsets):
        counts = np.array(metrics[subset]["confusion_matrix"])
        row_totals = counts.sum(axis=1, keepdims=True)
        shares = np.divide(counts, row_totals, where=row_totals > 0,
                           out=np.zeros(counts.shape, dtype=float))
        n = len(class_names)
        for row in range(n):
            for col in range(n):
                share = shares[row, col]
                # 2px surface gap between cells via the edge color.
                ax.add_patch(Rectangle((col, row), 1, 1,
                                       facecolor=SEQUENTIAL(share),
                                       edgecolor=SURFACE, linewidth=2))
                ink = "white" if share > 0.5 else INK
                ax.text(col + 0.5, row + 0.5,
                        f"{counts[row, col]}\n{share:.0%}",
                        ha="center", va="center", color=ink, fontsize=11)
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)  # Row 0 at the top, like a table.
        ax.set_xticks(np.arange(n) + 0.5, short)
        ax.set_yticks(np.arange(n) + 0.5, short)
        ax.set_xlabel("predicted", color=MUTED)
        ax.set_ylabel("true", color=MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        f1 = metrics[subset]["macro_f1"]
        f1_text = "" if f1 is None else f", macro-F1 {f1:.3f}"
        ax.set_title(f"{subset} (n={metrics[subset]['n']}{f1_text})",
                     color=INK, loc="left", fontsize=10)
        ax.set_aspect("equal")

    fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.01, "Color and % = share of the true class (row).",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)


def _direct_labels(ax, x: float, ends: dict[str, float]) -> None:
    """Label each line at its last point. Labels closer than a minimum gap
    are pushed apart downward, then the stack is kept inside the axes."""
    bottom, top = ax.get_ylim()
    gap = 0.06 * (top - bottom)
    items = sorted(ends.items(), key=lambda item: -item[1])  # Top first.
    positions: list[float] = []
    for _, y in items:
        y = min(y, top - gap / 2)
        if positions and positions[-1] - y < gap:
            y = positions[-1] - gap
        positions.append(y)
    overflow = bottom + gap / 2 - positions[-1]
    if overflow > 0:  # Stack fell below the axes: shift it all up.
        positions = [y + overflow for y in positions]
    for (label, _), y in zip(items, positions):
        ax.annotate(label, xy=(x, y), xytext=(8, 0),
                    textcoords="offset points", va="center",
                    color=INK, fontsize=8, annotation_clip=False)


def _style_axes(ax, title: str) -> None:
    """Recessive axes: hairline grid, muted ticks, no top/right frame."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.set_xlabel("epoch", color=MUTED)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.margins(x=0.08)
    ax.set_xlim(left=0.5, right=len(ax.lines[0].get_xdata()) + 2.5)
