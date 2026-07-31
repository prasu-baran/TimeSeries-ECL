import argparse
import csv
from pathlib import Path

from train_timebase_pmformer import plot_epoch_curves, safe_file_name, save_svg_line_plot, write_csv


def load_epoch_rows(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "model": row["model"],
                    "horizon": int(row["horizon"]),
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_pred_mse": float(row["train_pred_mse"]),
                    "val_loss": float(row["val_loss"]),
                    "val_pred_mse": float(row["val_pred_mse"]),
                    "learning_rate": float(row["learning_rate"]),
                    "gate": row["gate"],
                }
            )
    return rows


def ema_smooth(values, alpha=0.10):
    smoothed = []
    current = values[0]
    for value in values:
        current = alpha * value + (1.0 - alpha) * current
        smoothed.append(current)
    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Generate train/validation loss curves from results/epoch_log.csv.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    epoch_rows = load_epoch_rows(results_dir / "epoch_log.csv")
    models = sorted({row["model"] for row in epoch_rows})
    horizons = sorted({row["horizon"] for row in epoch_rows})

    for model in models:
        for horizon in horizons:
            plot_epoch_curves(results_dir, epoch_rows, model, horizon)

    summary_rows = []
    faculty_curve_rows = []
    for model in models:
        for horizon in horizons:
            rows = [
                row for row in epoch_rows
                if row["model"] == model and row["horizon"] == horizon
            ]
            rows = sorted(rows, key=lambda row: row["epoch"])
            best_so_far = []
            running_best = float("inf")
            for row in rows:
                running_best = min(running_best, row["val_loss"])
                best_so_far.append((row["epoch"], running_best))

            validation = [(row["epoch"], row["val_loss"]) for row in rows]
            train_loss = [(row["epoch"], row["train_loss"]) for row in rows]
            validation_smooth_values = ema_smooth([row["val_loss"] for row in rows], alpha=0.10)
            validation_smooth = [
                (row["epoch"], value)
                for row, value in zip(rows, validation_smooth_values)
            ]

            save_svg_line_plot(
                results_dir / f"faculty_loss_curve_{safe_file_name(model)}_{horizon}h.svg",
                f"{model} Training Loss and Validation Trend | {horizon}h",
                "Epoch",
                "Loss",
                [
                    ("Training Loss", train_loss, "#2563eb"),
                    ("Validation Loss Trend", validation_smooth, "#dc2626"),
                ],
            )

            save_svg_line_plot(
                results_dir / f"checkpoint_loss_curve_{safe_file_name(model)}_{horizon}h.svg",
                f"{model} Training Loss and Best Validation Loss | {horizon}h",
                "Epoch",
                "Loss",
                [
                    ("Training Loss", train_loss, "#2563eb"),
                    ("Best Validation Loss So Far", best_so_far, "#059669"),
                ],
            )

            for row, smoothed_val in zip(rows, validation_smooth_values):
                faculty_curve_rows.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "epoch": row["epoch"],
                        "training_loss": row["train_loss"],
                        "raw_validation_loss": row["val_loss"],
                        "validation_loss_trend_ema_alpha_0_10": smoothed_val,
                    }
                )

            save_svg_line_plot(
                results_dir / f"validation_best_so_far_{safe_file_name(model)}_{horizon}h.svg",
                f"{model} Validation Trend | {horizon}h",
                "Epoch",
                "Validation Loss",
                [
                    ("Validation Loss", validation, "#dc2626"),
                    ("Best Validation Loss So Far", best_so_far, "#059669"),
                ],
            )

            first = rows[0]["val_loss"]
            final = rows[-1]["val_loss"]
            best = min(row["val_loss"] for row in rows)
            best_epoch = next(row["epoch"] for row in rows if row["val_loss"] == best)
            summary_rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "epoch_1_val_loss": first,
                    "best_epoch": best_epoch,
                    "best_val_loss": best,
                    "final_epoch": rows[-1]["epoch"],
                    "final_val_loss": final,
                    "best_drop_from_epoch_1_pct": 100.0 * (first - best) / first,
                    "final_drop_from_epoch_1_pct": 100.0 * (first - final) / first,
                }
            )

    write_csv(results_dir / "validation_trend_summary.csv", summary_rows)
    write_csv(results_dir / "faculty_loss_curve_values.csv", faculty_curve_rows)

    print(f"Saved train/validation loss graphs and validation trend graphs to {results_dir.resolve()}")


if __name__ == "__main__":
    main()
