import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "train": "#2563eb",
    "val": "#059669",
    "axis": "#111827",
    "grid": "#e5e7eb",
    "text": "#1f2937",
}


def load_rows(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "model": row["model"],
                    "horizon": int(row["horizon"]),
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                }
            )
    return rows


def best_so_far(points):
    out = []
    best = float("inf")
    for epoch, value in points:
        best = min(best, value)
        out.append((epoch, best))
    return out


def get_font(size=14, bold=False):
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def normalize_to_start(points):
    start = points[0][1]
    if not start:
        return [(epoch, 0.0) for epoch, _ in points]
    return [(epoch, 100.0 * value / start) for epoch, value in points]


def draw_combined_loss_image(path, model, horizon, train_points, val_points):
    width, height = 1200, 620
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    train_plot_points = normalize_to_start(train_points)
    val_plot_points = normalize_to_start(val_points)

    left = 104
    right = width - 104
    top = 118
    bottom = height - 112
    plot_w = right - left
    plot_h = bottom - top

    all_epochs = [p[0] for p in train_plot_points + val_plot_points]
    x_min, x_max = min(all_epochs), max(all_epochs)

    all_values = [p[1] for p in train_plot_points + val_plot_points]
    y_min = min(all_values)
    y_max = max(all_values)
    pad = (y_max - y_min) * 0.12 if y_max > y_min else 0.01
    y_min = max(0.0, y_min - pad)
    y_max = y_max + pad

    def sx(x):
        return left + plot_w / 2 if x_max == x_min else left + (x - x_min) * plot_w / (x_max - x_min)

    def sy(y):
        return top + plot_h / 2 if y_max == y_min else top + (y_max - y) * plot_h / (y_max - y_min)

    title_font = get_font(24, bold=True)
    subtitle_font = get_font(14)
    label_font = get_font(13)
    tick_font = get_font(11)

    title = f"{model} Normalized Best-So-Far Loss | {horizon}h Forecast"
    subtitle = f"Checkpoint loss from {len(train_points)} logged epochs, normalized to Epoch 1 = 100%"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    draw.text(((width - title_box[2]) / 2, 22), title, fill=COLORS["text"], font=title_font)
    draw.text(((width - subtitle_box[2]) / 2, 58), subtitle, fill="#475569", font=subtitle_font)

    draw.rounded_rectangle((34, 88, width - 34, height - 30), radius=8, outline="#d1d5db", fill="#ffffff")

    for i in range(6):
        loss_y = y_min + (y_max - y_min) * i / 5
        y = sy(loss_y)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)

        loss_label = f"{loss_y:.2f}%"
        loss_box = draw.textbbox((0, 0), loss_label, font=tick_font)
        draw.text((left - 12 - loss_box[2], y - 7), loss_label, fill="#4b5563", font=tick_font)

    for x_val in range(int(x_min), int(x_max) + 1):
        x = sx(x_val)
        draw.line((x, top, x, bottom), fill="#f3f4f6", width=1)
        label = str(x_val)
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - box[2] / 2, bottom + 18), label, fill="#4b5563", font=tick_font)

    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=2)
    draw.line((left, top, left, bottom), fill=COLORS["axis"], width=2)
    draw.line((right, top, right, bottom), fill="#d1d5db", width=1)

    def draw_series(points, color):
        xy = [(sx(x), sy(y)) for x, y in points]
        if len(xy) > 1:
            draw.line(xy, fill=color, width=4)
        for x, y in xy:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="white", outline=color, width=2)

    draw_series(train_plot_points, COLORS["train"])
    draw_series(val_plot_points, COLORS["val"])

    legend_x = right - 250
    legend_y = top + 18
    draw.rounded_rectangle((legend_x - 16, legend_y - 12, legend_x + 226, legend_y + 58), radius=6, fill="#ffffff", outline="#d1d5db")
    draw.line((legend_x, legend_y + 8, legend_x + 30, legend_y + 8), fill=COLORS["train"], width=4)
    draw.text((legend_x + 40, legend_y), "Best Training Loss So Far", fill=COLORS["axis"], font=label_font)
    draw.line((legend_x, legend_y + 36, legend_x + 30, legend_y + 36), fill=COLORS["val"], width=4)
    draw.text((legend_x + 40, legend_y + 28), "Best Validation Loss So Far", fill=COLORS["axis"], font=label_font)

    x_box = draw.textbbox((0, 0), "Epoch", font=label_font)
    draw.text((left + plot_w / 2 - x_box[2] / 2, height - 74), "Epoch", fill=COLORS["axis"], font=label_font)
    draw.text((44, top - 26), "Loss (% of Epoch 1)", fill=COLORS["axis"], font=label_font)

    def note(label, points):
        start = points[0][1]
        end = points[-1][1]
        drop = 100.0 * (start - end) / start if start else 0.0
        return f"Best {label}: {start:.4f} -> {end:.4f} | Drop {drop:.2f}%"

    train_note = note("Train", train_points)
    val_note = note("Validation", val_points)
    draw.text((left, height - 48), train_note, fill=COLORS["train"], font=label_font)
    val_note_box = draw.textbbox((0, 0), val_note, font=label_font)
    draw.text((right - val_note_box[2], height - 48), val_note, fill=COLORS["val"], font=label_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def ema_smooth(points, alpha=0.10):
    current = points[0][1]
    out = []
    for epoch, value in points:
        current = alpha * value + (1.0 - alpha) * current
        out.append((epoch, current))
    return out


def value_range(points):
    values = [value for _, value in points]
    y_min = min(values)
    y_max = max(values)
    pad = (y_max - y_min) * 0.22 if y_max > y_min else max(abs(y_max) * 0.02, 0.001)
    return max(0.0, y_min - pad), y_max + pad


def draw_publication_loss_image(path, model, horizon, train_raw, val_raw):
    train_points = ema_smooth(train_raw)
    val_points = ema_smooth(val_raw)

    width, height = 1200, 700
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)

    left = 112
    right = width - 104
    upper_top, upper_bottom = 132, 330
    lower_top, lower_bottom = 374, 572
    plot_w = right - left
    x_min = min(epoch for epoch, _ in train_points + val_points)
    x_max = max(epoch for epoch, _ in train_points + val_points)
    train_min, train_max = value_range(train_points)
    val_min, val_max = value_range(val_points)

    def sx(x):
        return left + plot_w / 2 if x_max == x_min else left + (x - x_min) * plot_w / (x_max - x_min)

    def sy(y, y_min, y_max, top, bottom):
        return top + (y_max - y) * (bottom - top) / (y_max - y_min)

    title_font = get_font(24, bold=True)
    subtitle_font = get_font(14)
    label_font = get_font(13)
    tick_font = get_font(11)

    title = f"{model} 5-Epoch Training and Validation Loss | {horizon}h Forecast"
    subtitle = "EMA-smoothed from logged losses (alpha=0.10); broken y-axis highlights both loss ranges"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    draw.text(((width - title_box[2]) / 2, 24), title, fill=COLORS["text"], font=title_font)
    draw.text(((width - subtitle_box[2]) / 2, 62), subtitle, fill="#475569", font=subtitle_font)
    draw.rounded_rectangle((34, 96, width - 34, height - 34), radius=8, outline="#d1d5db", fill="#ffffff")

    def draw_band(top, bottom, y_min, y_max, label):
        for i in range(5):
            y_val = y_min + (y_max - y_min) * i / 4
            y = sy(y_val, y_min, y_max, top, bottom)
            draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
            loss_label = f"{y_val:.4f}"
            loss_box = draw.textbbox((0, 0), loss_label, font=tick_font)
            draw.text((left - 12 - loss_box[2], y - 7), loss_label, fill="#4b5563", font=tick_font)
        for x_val in range(int(x_min), int(x_max) + 1):
            x = sx(x_val)
            draw.line((x, top, x, bottom), fill="#f3f4f6", width=1)
        draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=2)
        draw.line((left, top, left, bottom), fill=COLORS["axis"], width=2)
        draw.line((right, top, right, bottom), fill="#d1d5db", width=1)
        draw.text((right - 126, top + 10), label, fill="#64748b", font=label_font)

    draw_band(upper_top, upper_bottom, train_min, train_max, "training range")
    draw_band(lower_top, lower_bottom, val_min, val_max, "validation range")

    for x_val in range(int(x_min), int(x_max) + 1):
        x = sx(x_val)
        label = str(x_val)
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - box[2] / 2, lower_bottom + 18), label, fill="#4b5563", font=tick_font)

    # Broken-axis marks.
    for x in (left, right):
        draw.line((x - 8, upper_bottom + 12, x + 8, upper_bottom - 4), fill=COLORS["axis"], width=2)
        draw.line((x - 8, lower_top + 4, x + 8, lower_top - 12), fill=COLORS["axis"], width=2)

    def draw_series(points, color, y_min, y_max, top, bottom, marker):
        xy = [(sx(x), sy(y, y_min, y_max, top, bottom)) for x, y in points]
        if len(xy) > 1:
            draw.line(xy, fill=color, width=4)
        for x, y in xy:
            if marker == "square":
                draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill="white", outline=color, width=2)
            else:
                draw.polygon([(x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y)], fill="white", outline=color)
                draw.line([(x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y), (x, y - 6)], fill=color, width=2)

    draw_series(train_points, COLORS["train"], train_min, train_max, upper_top, upper_bottom, "square")
    draw_series(val_points, COLORS["val"], val_min, val_max, lower_top, lower_bottom, "diamond")

    legend_x = right - 260
    legend_y = upper_top + 18
    draw.rounded_rectangle((legend_x - 16, legend_y - 12, legend_x + 238, legend_y + 58), radius=6, fill="#ffffff", outline="#d1d5db")
    draw.line((legend_x, legend_y + 8, legend_x + 30, legend_y + 8), fill=COLORS["train"], width=4)
    draw.text((legend_x + 40, legend_y), "training loss", fill=COLORS["axis"], font=label_font)
    draw.line((legend_x, legend_y + 36, legend_x + 30, legend_y + 36), fill=COLORS["val"], width=4)
    draw.text((legend_x + 40, legend_y + 28), "validation loss", fill=COLORS["axis"], font=label_font)

    x_box = draw.textbbox((0, 0), "Epoch", font=label_font)
    draw.text((left + plot_w / 2 - x_box[2] / 2, height - 76), "Epoch", fill=COLORS["axis"], font=label_font)
    draw.text((42, 338), "Loss", fill=COLORS["axis"], font=label_font)

    def note(label, points):
        start = points[0][1]
        end = points[-1][1]
        drop = 100.0 * (start - end) / start if start else 0.0
        return f"{label}: {start:.4f} -> {end:.4f} | Drop {drop:.2f}%"

    train_note = note("Train EMA", train_points)
    val_note = note("Validation EMA", val_points)
    draw.text((left, height - 48), train_note, fill=COLORS["train"], font=label_font)
    val_note_box = draw.textbbox((0, 0), val_note, font=label_font)
    draw.text((right - val_note_box[2], height - 48), val_note, fill=COLORS["val"], font=label_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def draw_single_panel_loss_image(path, model, horizon, train_raw, val_raw):
    train_points = ema_smooth(train_raw)
    val_points = ema_smooth(val_raw)

    width, height = 1200, 620
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)

    left = 104
    right = width - 104
    top = 118
    bottom = height - 112
    plot_w = right - left
    plot_h = bottom - top

    all_epochs = [epoch for epoch, _ in train_points + val_points]
    x_min, x_max = min(all_epochs), max(all_epochs)
    all_values = [value for _, value in train_points + val_points]
    y_min = min(all_values)
    y_max = max(all_values)
    pad = (y_max - y_min) * 0.12 if y_max > y_min else max(abs(y_max) * 0.02, 0.001)
    y_min = max(0.0, y_min - pad)
    y_max = y_max + pad

    def sx(x):
        return left + plot_w / 2 if x_max == x_min else left + (x - x_min) * plot_w / (x_max - x_min)

    def sy(y):
        return top + plot_h / 2 if y_max == y_min else top + (y_max - y) * plot_h / (y_max - y_min)

    title_font = get_font(24, bold=True)
    subtitle_font = get_font(14)
    label_font = get_font(13)
    tick_font = get_font(11)

    title = f"{model} 5-Epoch Training and Validation Loss | {horizon}h Forecast"
    subtitle = "EMA-smoothed from logged losses (alpha=0.10); single shared y-axis"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    draw.text(((width - title_box[2]) / 2, 22), title, fill=COLORS["text"], font=title_font)
    draw.text(((width - subtitle_box[2]) / 2, 58), subtitle, fill="#475569", font=subtitle_font)
    draw.rounded_rectangle((34, 88, width - 34, height - 30), radius=8, outline="#d1d5db", fill="#ffffff")

    for i in range(6):
        loss_y = y_min + (y_max - y_min) * i / 5
        y = sy(loss_y)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        loss_label = f"{loss_y:.4f}"
        loss_box = draw.textbbox((0, 0), loss_label, font=tick_font)
        draw.text((left - 12 - loss_box[2], y - 7), loss_label, fill="#4b5563", font=tick_font)

    for x_val in range(int(x_min), int(x_max) + 1):
        x = sx(x_val)
        draw.line((x, top, x, bottom), fill="#f3f4f6", width=1)
        label = str(x_val)
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - box[2] / 2, bottom + 18), label, fill="#4b5563", font=tick_font)

    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=2)
    draw.line((left, top, left, bottom), fill=COLORS["axis"], width=2)
    draw.line((right, top, right, bottom), fill="#d1d5db", width=1)

    def draw_series(points, color, marker):
        xy = [(sx(x), sy(y)) for x, y in points]
        if len(xy) > 1:
            draw.line(xy, fill=color, width=4)
        for x, y in xy:
            if marker == "square":
                draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill="white", outline=color, width=2)
            else:
                draw.polygon([(x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y)], fill="white", outline=color)
                draw.line([(x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y), (x, y - 6)], fill=color, width=2)

    draw_series(train_points, COLORS["train"], "square")
    draw_series(val_points, COLORS["val"], "diamond")

    legend_x = right - 260
    legend_y = top + 18
    draw.rounded_rectangle((legend_x - 16, legend_y - 12, legend_x + 238, legend_y + 58), radius=6, fill="#ffffff", outline="#d1d5db")
    draw.line((legend_x, legend_y + 8, legend_x + 30, legend_y + 8), fill=COLORS["train"], width=4)
    draw.text((legend_x + 40, legend_y), "training loss", fill=COLORS["axis"], font=label_font)
    draw.line((legend_x, legend_y + 36, legend_x + 30, legend_y + 36), fill=COLORS["val"], width=4)
    draw.text((legend_x + 40, legend_y + 28), "validation loss", fill=COLORS["axis"], font=label_font)

    x_box = draw.textbbox((0, 0), "Epoch", font=label_font)
    draw.text((left + plot_w / 2 - x_box[2] / 2, height - 74), "Epoch", fill=COLORS["axis"], font=label_font)
    draw.text((44, top - 26), "Loss", fill=COLORS["axis"], font=label_font)

    def note(label, points):
        start = points[0][1]
        end = points[-1][1]
        drop = 100.0 * (start - end) / start if start else 0.0
        return f"{label}: {start:.4f} -> {end:.4f} | Drop {drop:.2f}%"

    train_note = note("Train EMA", train_points)
    val_note = note("Validation EMA", val_points)
    draw.text((left, height - 48), train_note, fill=COLORS["train"], font=label_font)
    val_note_box = draw.textbbox((0, 0), val_note, font=label_font)
    draw.text((right - val_note_box[2], height - 48), val_note, fill=COLORS["val"], font=label_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def draw_panel(draw, rect, title, x_label, y_label, points, color):
    x0, y0, x1, y1 = rect
    left = x0 + 72
    right = x1 - 22
    top = y0 + 48
    bottom = y1 - 58
    plot_w = right - left
    plot_h = bottom - top

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad = (y_max - y_min) * 0.18 if y_max > y_min else 0.01
    y_min = max(0.0, y_min - pad)
    y_max = y_max + pad

    def sx(x):
        return left + plot_w / 2 if x_max == x_min else left + (x - x_min) * plot_w / (x_max - x_min)

    def sy(y):
        return top + plot_h / 2 if y_max == y_min else top + (y_max - y) * plot_h / (y_max - y_min)

    title_font = get_font(18, bold=True)
    label_font = get_font(13)
    tick_font = get_font(11)

    draw.rounded_rectangle((x0, y0, x1, y1), radius=8, outline="#d1d5db", fill="#ffffff")
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text((x0 + (x1 - x0 - title_box[2]) / 2, y0 + 14), title, fill=COLORS["text"], font=title_font)

    for i in range(6):
        y_val = y_min + (y_max - y_min) * i / 5
        y = sy(y_val)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        label = f"{y_val:.4f}"
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - 10 - box[2], y - 7), label, fill="#4b5563", font=tick_font)

    for x_val in range(int(x_min), int(x_max) + 1):
        x = sx(x_val)
        draw.line((x, top, x, bottom), fill="#f3f4f6", width=1)
        label = str(x_val)
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - box[2] / 2, bottom + 18), label, fill="#4b5563", font=tick_font)

    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=2)
    draw.line((left, top, left, bottom), fill=COLORS["axis"], width=2)

    xy = [(sx(x), sy(y)) for x, y in points]
    if len(xy) > 1:
        draw.line(xy, fill=color, width=4)
    for x, y in xy:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="white", outline=color, width=2)

    x_box = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text((left + plot_w / 2 - x_box[2] / 2, y1 - 32), x_label, fill=COLORS["axis"], font=label_font)
    draw.text((x0 + 14, top + plot_h / 2 - 8), y_label, fill=COLORS["axis"], font=label_font)

    start = points[0][1]
    end = points[-1][1]
    drop = 100.0 * (start - end) / start if start else 0.0
    note = f"Start {start:.4f} -> End {end:.4f} | Drop {drop:.2f}%"
    note_box = draw.textbbox((0, 0), note, font=label_font)
    draw.text((x0 + (x1 - x0 - note_box[2]) / 2, y1 - 54), note, fill=color, font=label_font)


def drop_pct(points):
    start = points[0][1]
    end = points[-1][1]
    return 100.0 * (start - end) / start if start else 0.0


def write_figure_summary(path, rows):
    if not rows:
        return
    fieldnames = [
        "model",
        "horizon",
        "epochs",
        "start_train_loss",
        "final_train_loss_ema_alpha_0_10",
        "train_ema_drop_pct",
        "start_val_loss",
        "final_val_loss_ema_alpha_0_10",
        "val_ema_drop_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cleanup_extra_images(results_dir, keep_names):
    image_suffixes = {".png", ".svg", ".jpg", ".jpeg", ".webp"}
    for path in results_dir.iterdir():
        if path.is_file() and path.suffix.lower() in image_suffixes and path.name not in keep_names:
            path.unlink()


def generate(results_dir):
    rows = load_rows(results_dir / "epoch_log.csv")
    summary_rows = []
    keep_names = set()

    for model in ["TimeBase", "TimeBase+Informer"]:
        for horizon in [1, 8, 16, 24]:
            subset = [
                row for row in rows
                if row["model"] == model and row["horizon"] == horizon
            ]
            if not subset:
                continue
            subset.sort(key=lambda row: row["epoch"])
            train = [(row["epoch"], row["train_loss"]) for row in subset]
            validation = [(row["epoch"], row["val_loss"]) for row in subset]
            train_trend = ema_smooth(train)
            validation_trend = ema_smooth(validation)

            prefix = "timebase_informer" if model == "TimeBase+Informer" else "timebase"
            output = results_dir / f"{prefix}_{horizon}h_loss_curve.png"
            draw_single_panel_loss_image(output, model, horizon, train, validation)
            keep_names.add(output.name)
            summary_rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "epochs": len(subset),
                    "start_train_loss": train[0][1],
                    "final_train_loss_ema_alpha_0_10": train_trend[-1][1],
                    "train_ema_drop_pct": drop_pct(train_trend),
                    "start_val_loss": validation[0][1],
                    "final_val_loss_ema_alpha_0_10": validation_trend[-1][1],
                    "val_ema_drop_pct": drop_pct(validation_trend),
                }
            )

    write_figure_summary(results_dir / "figure_loss_summary.csv", summary_rows)
    cleanup_extra_images(results_dir, keep_names)


def main():
    generate(Path("results"))
    print("Saved combined loss-curve PNGs and figure_loss_summary.csv.")


if __name__ == "__main__":
    main()
