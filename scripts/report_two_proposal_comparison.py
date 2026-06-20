#!/usr/bin/env python3
"""Print and export the two-proposal comparison table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

MODELS = ("plain_fourier_unet", "apdr_fourier_unet")
DISPLAY = {
    "plain_fourier_unet": "Proposal I: Plain Fourier U-Net",
    "apdr_fourier_unet": "Proposal II: APDR-Fourier U-Net",
}


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
    if set(rows) != set(MODELS):
        raise SystemExit(f"ERROR: expected {MODELS}, found {sorted(rows)}")

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
    for model in MODELS:
        row = dict(rows[model])
        row["method"] = DISPLAY[model]
        pretty.append(row)
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in pretty))
        for column in columns
    }
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in pretty:
        print(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))

    plain = rows[MODELS[0]]
    apdr = rows[MODELS[1]]
    delta_rows = []
    for metric, higher_better in (
        ("dice", True),
        ("iou", True),
        ("precision", True),
        ("recall", True),
        ("mae", False),
        ("loss", False),
    ):
        plain_value = float(plain[f"{metric}_mean"])
        apdr_value = float(apdr[f"{metric}_mean"])
        improvement = apdr_value - plain_value if higher_better else plain_value - apdr_value
        delta_rows.append(
            {
                "metric": metric,
                "plain_fourier_value": plain_value,
                "apdr_value": apdr_value,
                "apdr_improvement": improvement,
            }
        )
        print(f"{metric:9s}: APDR improvement={improvement:+.6f}")

    dataset = plain.get("dataset", "dataset")
    seeds = plain.get("num_seeds", "")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{Comparison of the two proposed methods on {dataset} over {seeds} random seeds.}}",
        r"\label{tab:two-proposal-comparison}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Dice $\uparrow$ & IoU $\uparrow$ & Precision $\uparrow$ & Recall $\uparrow$ & MAE $\downarrow$ & Loss $\downarrow$ \\",
        r"\midrule",
    ]
    for model in MODELS:
        row = rows[model]
        lines.append(
            f"{DISPLAY[model]} & {row['dice_mean_pm_std']} & {row['iou_mean_pm_std']} & "
            f"{row['precision_mean_pm_std']} & {row['recall_mean_pm_std']} & "
            f"{row['mae_mean_pm_std']} & {row['loss_mean_pm_std']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    delta_path.parent.mkdir(parents=True, exist_ok=True)
    with delta_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(delta_rows[0]))
        writer.writeheader()
        writer.writerows(delta_rows)

    print(f"LaTeX table: {latex_path}")
    print(f"Delta table: {delta_path}")


if __name__ == "__main__":
    main()
