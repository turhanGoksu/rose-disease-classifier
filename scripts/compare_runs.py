"""Compare training runs side by side from their metrics.json files.

Every run must use the same split, seed and hyperparameters; only the
imbalance handling (--balance) should differ, so any difference in the
table comes from that one change.

Usage:
    python scripts/compare_runs.py --runs-dir outputs --runs none sampler loss
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

POSITIVE = "Rose_Black_Spot"
NEGATIVE = "Rose_Healthy"
# Settings that must match for the comparison to be fair.
SHARED_SETTINGS = ("split_file", "epochs", "batch_size", "lr", "backbone_lr",
                   "weight_decay", "seed", "trainable_layers")


def row(run: str, report: dict[str, Any], split: str) -> str:
    """One markdown table row for a run on one split."""
    m = report[split]["metrics"]
    overall, same = m["overall"], m["same_source"]
    bs = overall["per_class"][POSITIVE]
    (tn, fp), (fn, tp) = overall["confusion_matrix"]  # Rows = true class.
    return (f"| {run} | {report['best_epoch']} "
            f"| {overall['accuracy']:.3f} | {overall['macro_f1']:.3f} "
            f"| {same['macro_f1']:.3f} "
            f"| {bs['precision']:.3f} | {bs['recall']:.3f} "
            f"| {overall['per_class'][NEGATIVE]['recall']:.3f} "
            f"| {fn} | {fp} | {m['studio']['accuracy']:.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--runs", nargs="+",
                        default=["none", "sampler", "loss"])
    parser.add_argument("--output", type=Path, default=None,
                        help="Also write the tables to this markdown file.")
    args = parser.parse_args()

    reports = {run: json.loads(
        (args.runs_dir.expanduser() / run / "metrics.json").read_text())
        for run in args.runs}

    # Refuse to compare runs that differ in more than the one variable.
    for key in SHARED_SETTINGS:
        values = {run: r["config"].get(key) for run, r in reports.items()}
        if len({json.dumps(v) for v in values.values()}) > 1:
            raise ValueError(f"Runs differ in {key!r}: {values}")

    header = ("| run | best epoch | accuracy | macro-F1 | same-source "
              "macro-F1 | Black_Spot precision | Black_Spot recall "
              "| Healthy recall | missed Black_Spot (FN) "
              "| false alarms (FP) | studio accuracy |\n"
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    sections = []
    for split in ("val", "test"):
        note = (" (used to choose the epoch and the method)" if split == "val"
                else " (evaluated once, not used for any choice)")
        lines = [f"### {split}{note}", "", header]
        lines += [row(run, report, split) for run, report in reports.items()]
        sections.append("\n".join(lines))
    text = "\n\n".join(sections) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
