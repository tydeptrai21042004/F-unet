#!/usr/bin/env python3
"""Print and export the manuscript table for the full component ablation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from .component_ablation_spec import (
        COMPONENT_COMPARISONS,
        DISPLAY_NAMES,
        FULL_COMPONENT_ABLATION_MODELS,
    )
except ImportError:  # Direct script execution
    from component_ablation_spec import (
        COMPONENT_COMPARISONS,
        DISPLAY_NAMES,
        FULL_COMPONENT_ABLATION_MODELS,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--latex-path", required=True)
    parser.add_argument("--delta-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    latex_path = Path(args.latex_path)
    delta_path = Path(args.delta_path) if args.delta_path else latex_path.with_suffix(".deltas.csv")

    with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = {row["model"]: row for row in csv.DictReader(file)}
    missing = set(FULL_COMPONENT_ABLATION_MODELS) - set(rows)
    if missing:
        raise SystemExit(f"ERROR: summary lacks models: {sorted(missing)}")

    columns = (
        "method",
        "dice_mean_pm_std",
        "iou_mean_pm_std",
        "precision_mean_pm_std",
        "recall_mean_pm_std",
        "mae_mean_pm_std",
        "loss_mean_pm_std",
    )
    pretty = []
    for model in FULL_COMPONENT_ABLATION_MODELS:
        row = dict(rows[model])
        row["method"] = DISPLAY_NAMES[model]
        pretty.append(row)

    widths = {
        column: max(len(column), *(len(str(row[column])) for row in pretty))
        for column in columns
    }
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in pretty:
        print(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))

    ranking = sorted(
        FULL_COMPONENT_ABLATION_MODELS,
        key=lambda model: (
            -float(rows[model]["dice_mean"]),
            -float(rows[model]["iou_mean"]),
            float(rows[model]["mae_mean"]),
        ),
    )
    print("\nRanking by Dice, IoU, then MAE:")
    for index, model in enumerate(ranking, start=1):
        print(
            f"{index:2d}. {DISPLAY_NAMES[model]}: "
            f"Dice={rows[model]['dice_mean_pm_std']}, "
            f"IoU={rows[model]['iou_mean_pm_std']}, "
            f"MAE={rows[model]['mae_mean_pm_std']}"
        )

    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full component ablation of Plain Fourier U-Net and APDR-Fourier U-Net on ETIS-LaribPolypDB over three random seeds.}",
        r"\label{tab:full-component-ablation-etis}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Dice $\uparrow$ & IoU $\uparrow$ & Precision $\uparrow$ & Recall $\uparrow$ & MAE $\downarrow$ & Loss $\downarrow$ \\",
        r"\midrule",
    ]
    for model in FULL_COMPONENT_ABLATION_MODELS:
        row = rows[model]
        method = DISPLAY_NAMES[model].replace("_", r"\_")
        latex_lines.append(
            f"{method} & {row['dice_mean_pm_std']} & {row['iou_mean_pm_std']} & "
            f"{row['precision_mean_pm_std']} & {row['recall_mean_pm_std']} & "
            f"{row['mae_mean_pm_std']} & {row['loss_mean_pm_std']} \\\\"
        )
        if model in {"unet", "plain_fourier_unet"}:
            latex_lines.append(r"\midrule")
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    deltas = []
    print("\nComponent contribution deltas:")
    for ablated, complete in COMPONENT_COMPARISONS:
        dice_delta = float(rows[complete]["dice_mean"]) - float(rows[ablated]["dice_mean"])
        iou_delta = float(rows[complete]["iou_mean"]) - float(rows[ablated]["iou_mean"])
        mae_reduction = float(rows[ablated]["mae_mean"]) - float(rows[complete]["mae_mean"])
        item = {
            "complete_model": complete,
            "ablated_model": ablated,
            "dice_gain": dice_delta,
            "iou_gain": iou_delta,
            "mae_reduction": mae_reduction,
        }
        deltas.append(item)
        print(
            f"{complete:28s} versus {ablated:32s} | "
            f"ΔDice={dice_delta:+.6f} | ΔIoU={iou_delta:+.6f} | "
            f"MAE reduction={mae_reduction:+.6f}"
        )

    delta_path.parent.mkdir(parents=True, exist_ok=True)
    with delta_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)

    print(f"\nLaTeX table: {latex_path}")
    print(f"Contribution deltas: {delta_path}")


if __name__ == "__main__":
    main()
