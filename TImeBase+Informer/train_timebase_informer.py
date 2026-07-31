import argparse
import copy
import csv
import html
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    csv_path: str = "Load House 1.csv"
    results_dir: str = "results"
    seed: int = 42
    seq_len: int = 720
    horizons: tuple = (1, 8, 16, 24)
    epochs: int = 5
    batch_size: int = 256
    tb_seg_len: int = 24
    tb_basis_num: int = 6
    orth_lambda: float = 0.02
    timebase_lr: float = 0.001
    informer_lr: float = 0.0003
    hybrid_timebase_lr: float = 0.00008
    gate_lr: float = 0.0005
    gate_init: float = -1.5
    weight_decay: float = 0.0001
    d_model: int = 32
    n_heads: int = 4
    d_ff: int = 96
    dropout: float = 0.10
    encoder_layers: int = 1
    conv_stride: int = 12
    distill: bool = True
    num_workers: int = 0
    grad_clip: float = 1.0
    save_prediction_windows: int = 256


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_dt(date_text, hour_text):
    return datetime.fromisoformat(f"{date_text}T{hour_text}")


def load_hourly_consumption(csv_path):
    buckets = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = parse_dt(row["Date"], row["Hour"])
            buckets[dt].append(float(row["Consumption (kW)"]))

    start = min(buckets)
    end = max(buckets)
    step = timedelta(minutes=15)

    times_15 = []
    values_15 = []
    current = start
    while current <= end:
        if current in buckets:
            values_15.append(float(np.mean(buckets[current])))
        else:
            values_15.append(np.nan)
        times_15.append(current)
        current += step

    values_15 = np.asarray(values_15, dtype=np.float64)
    missing = ~np.isfinite(values_15)
    if missing.any():
        idx = np.arange(len(values_15))
        good = np.isfinite(values_15)
        values_15[missing] = np.interp(idx[missing], idx[good], values_15[good])

    hourly_times = []
    hourly_values = []
    for i in range(0, len(values_15), 4):
        chunk = values_15[i : i + 4]
        if len(chunk) == 0:
            continue
        hourly_times.append(times_15[i])
        hourly_values.append(float(np.mean(chunk)))

    values = np.asarray(hourly_values, dtype=np.float32).reshape(-1, 1)
    return hourly_times, values


def split_and_scale(values, seq_len):
    n = len(values)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)

    mean = values[:train_end].mean(axis=0, keepdims=True)
    std = values[:train_end].std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    scaled = ((values - mean) / std).astype(np.float32)

    train = scaled[:train_end]
    val = scaled[train_end - seq_len : val_end]
    test = scaled[val_end - seq_len :]
    return train, val, test, mean.astype(np.float32), std.astype(np.float32), train_end, val_end


class ForecastDataset(Dataset):
    def __init__(self, values, seq_len, pred_len):
        self.values = values.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return max(0, len(self.values) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        x = self.values[idx : idx + self.seq_len]
        y = self.values[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)


def make_loader(values, seq_len, pred_len, batch_size, shuffle, num_workers):
    dataset = ForecastDataset(values, seq_len, pred_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class TimeBaseCore(nn.Module):
    def __init__(self, seq_len, pred_len, seg_len=24, basis_num=6):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.seg_len = seg_len
        self.basis_num = basis_num
        self.n_seg = math.ceil(seq_len / seg_len)
        self.n_pred_seg = math.ceil(pred_len / seg_len)
        self.basis_extract = nn.Linear(self.n_seg, basis_num)
        self.segment_forecast = nn.Linear(basis_num, self.n_pred_seg)

    def forward(self, x):
        b, t, c = x.shape
        need = self.n_seg * self.seg_len - t
        if need > 0:
            x = torch.cat([x, x[:, -1:, :].repeat(1, need, 1)], dim=1)

        z = x.permute(0, 2, 1).reshape(b, c, self.n_seg, self.seg_len)
        z = z.permute(0, 1, 3, 2)
        basis = self.basis_extract(z)
        basis = basis.permute(0, 1, 3, 2)

        z = basis.permute(0, 1, 3, 2)
        out = self.segment_forecast(z)
        out = out.permute(0, 1, 3, 2).reshape(b, c, self.n_pred_seg * self.seg_len)
        out = out[:, :, : self.pred_len].permute(0, 2, 1)
        return out, basis


def orth_loss(basis):
    b, c, r, p = basis.shape
    z = basis.reshape(b * c, r, p)
    z = z / (torch.norm(z, dim=-1, keepdim=True) + 1e-8)
    gram = torch.matmul(z, z.transpose(1, 2))
    eye = torch.eye(r, device=basis.device).view(1, r, r)
    return ((gram * (1.0 - eye)) ** 2).mean()


class BaseTimeBase(nn.Module):
    def __init__(self, cfg, pred_len):
        super().__init__()
        self.timebase = TimeBaseCore(
            seq_len=cfg.seq_len,
            pred_len=pred_len,
            seg_len=cfg.tb_seg_len,
            basis_num=cfg.tb_basis_num,
        )

    def forward(self, x):
        pred, basis = self.timebase(x)
        return pred, basis, None


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, length):
        return self.pe[:, :length, :]


class InformerResidualEncoder(nn.Module):
    def __init__(self, cfg, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.token = nn.Conv1d(
            in_channels=1,
            out_channels=cfg.d_model,
            kernel_size=5,
            stride=cfg.conv_stride,
            padding=2,
        )
        token_count = math.ceil(cfg.seq_len / cfg.conv_stride)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, max_len=token_count + 4)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.d_ff,
                    dropout=cfg.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(cfg.encoder_layers)
            ]
        )
        self.distill_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=3, padding=1),
                    nn.ELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
                )
                for _ in range(max(0, cfg.encoder_layers - 1))
            ]
        )
        self.use_distill = cfg.distill
        self.norm = nn.LayerNorm(cfg.d_model * 2)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, pred_len),
        )
        nn.init.zeros_(self.head[-1].bias)
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.05)

    def forward(self, x):
        z = x.transpose(1, 2)
        z = self.token(z).transpose(1, 2)
        z = z + self.pos(z.size(1)).to(z.dtype)

        for i, layer in enumerate(self.layers):
            z = layer(z)
            if self.use_distill and i < len(self.distill_layers):
                z = self.distill_layers[i](z.transpose(1, 2)).transpose(1, 2)

        pooled = torch.cat([z.mean(dim=1), z[:, -1, :]], dim=-1)
        residual = self.head(self.norm(pooled)).unsqueeze(-1)
        return residual


class TimeBaseInformer(nn.Module):
    def __init__(self, cfg, pred_len):
        super().__init__()
        self.timebase = TimeBaseCore(
            seq_len=cfg.seq_len,
            pred_len=pred_len,
            seg_len=cfg.tb_seg_len,
            basis_num=cfg.tb_basis_num,
        )
        self.informer_encoder = InformerResidualEncoder(cfg, pred_len)
        self.gate = nn.Parameter(torch.tensor(float(cfg.gate_init)))

    def forward(self, x):
        base_pred, basis = self.timebase(x)
        residual = self.informer_encoder(x)
        gate = torch.sigmoid(self.gate)
        return base_pred + gate * residual, basis, gate


def inverse_scale(arr, mean, std):
    return arr * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


@torch.no_grad()
def evaluate(model, loader, device, mean, std):
    model.eval()
    preds = []
    trues = []
    mse_loss = nn.MSELoss(reduction="sum")
    scaled_sse = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred, _, _ = model(x)
        scaled_sse += mse_loss(pred, y).item()
        count += y.numel()
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    preds_real = inverse_scale(preds, mean, std)
    trues_real = inverse_scale(trues, mean, std)
    err = preds_real.reshape(-1) - trues_real.reshape(-1)
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    return {
        "scaled_mse": float(scaled_sse / max(count, 1)),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "preds": preds_real,
        "trues": trues_real,
    }


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_model(model, model_name, horizon, loaders, cfg, device, mean, std, init_state=None):
    train_loader, val_loader, test_loader = loaders
    model = model.to(device)
    if init_state is not None and hasattr(model, "timebase"):
        model.timebase.load_state_dict(copy.deepcopy(init_state))

    if model_name == "TimeBase+Informer":
        optimizer = torch.optim.AdamW(
            [
                {"params": model.timebase.parameters(), "lr": cfg.hybrid_timebase_lr},
                {"params": model.informer_encoder.parameters(), "lr": cfg.informer_lr},
                {"params": [model.gate], "lr": cfg.gate_lr},
            ],
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.timebase_lr, weight_decay=0.0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.MSELoss()
    rows = []
    best_val = float("inf")
    best_state = None

    print(f"\nTraining {model_name} | horizon={horizon}h | params={count_params(model)}")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_total = 0.0
        train_pred = 0.0
        examples = 0
        gate_value = None

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, basis, gate = model(x)
            pred_loss = criterion(pred, y)
            reg_loss = orth_loss(basis)
            loss = pred_loss + cfg.orth_lambda * reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            bs = x.size(0)
            train_total += loss.item() * bs
            train_pred += pred_loss.item() * bs
            examples += bs
            if gate is not None:
                gate_value = float(gate.detach().cpu())

        train_total /= examples
        train_pred /= examples

        model.eval()
        val_total = 0.0
        val_pred = 0.0
        val_examples = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred, basis, gate = model(x)
                pred_loss = criterion(pred, y)
                loss = pred_loss + cfg.orth_lambda * orth_loss(basis)
                bs = x.size(0)
                val_total += loss.item() * bs
                val_pred += pred_loss.item() * bs
                val_examples += bs
                if gate is not None:
                    gate_value = float(gate.detach().cpu())

        val_total /= val_examples
        val_pred /= val_examples
        scheduler.step()
        lr = optimizer.param_groups[-1]["lr"]

        row = {
            "model": model_name,
            "horizon": horizon,
            "epoch": epoch,
            "train_loss": train_total,
            "train_pred_mse": train_pred,
            "val_loss": val_total,
            "val_pred_mse": val_pred,
            "learning_rate": lr,
            "gate": "" if gate_value is None else gate_value,
        }
        rows.append(row)
        gate_msg = "" if gate_value is None else f" | gate={gate_value:.4f}"
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} "
            f"train={train_total:.6f} val={val_total:.6f} "
            f"train_mse={train_pred:.6f} val_mse={val_pred:.6f}{gate_msg}"
        )

        if val_pred < best_val:
            best_val = val_pred
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(model, test_loader, device, mean, std)
    metrics = {
        "model": model_name,
        "horizon": horizon,
        "mae": test["mae"],
        "mse": test["mse"],
        "rmse": test["rmse"],
        "scaled_test_mse": test["scaled_mse"],
        "params": count_params(model),
        "best_val_pred_mse": best_val,
    }
    return model, rows, metrics, test


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_prediction_csv(path, preds, trues, horizon, max_windows):
    rows = []
    windows = min(max_windows, preds.shape[0])
    for i in range(windows):
        for step in range(horizon):
            rows.append(
                {
                    "window": i,
                    "step": step + 1,
                    "actual": float(trues[i, step, 0]),
                    "prediction": float(preds[i, step, 0]),
                    "error": float(preds[i, step, 0] - trues[i, step, 0]),
                }
            )
    write_csv(path, rows, ["window", "step", "actual", "prediction", "error"])


def safe_file_name(text):
    return text.replace("+", "_plus_").replace(" ", "_").replace("/", "_")


def short_file_name(text):
    return text.lower().replace("+", "_").replace(" ", "_").replace("/", "_")


def best_so_far_values(values):
    best = float("inf")
    out = []
    for value in values:
        best = min(best, value)
        out.append(best)
    return out


def prepare_scale(series):
    xs = []
    ys = []
    for _, points, _ in series:
        xs.extend([p[0] for p in points])
        ys.extend([p[1] for p in points])
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.1
    return x_min, x_max, max(0.0, y_min - pad), y_max + pad


def save_svg_line_plot(path, title, x_label, y_label, series):
    width = 900
    height = 520
    left = 86
    right = 28
    top = 58
    bottom = 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max, y_min, y_max = prepare_scale(series)

    def sx(x):
        return left + plot_w / 2 if x_max == x_min else left + (x - x_min) * plot_w / (x_max - x_min)

    def sy(y):
        return top + plot_h / 2 if y_max == y_min else top + (y_max - y) * plot_h / (y_max - y_min)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#1f2937">{html.escape(title)}</text>',
    ]

    for i in range(6):
        y_val = y_min + (y_max - y_min) * i / 5
        y_pos = sy(y_val)
        elements.append(f'<line x1="{left}" y1="{y_pos:.2f}" x2="{left + plot_w}" y2="{y_pos:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        elements.append(f'<text x="{left - 12}" y="{y_pos + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{y_val:.3f}</text>')

    for x_val in range(int(x_min), int(x_max) + 1):
        x_pos = sx(x_val)
        elements.append(f'<line x1="{x_pos:.2f}" y1="{top}" x2="{x_pos:.2f}" y2="{top + plot_h}" stroke="#f3f4f6" stroke-width="1"/>')
        elements.append(f'<text x="{x_pos:.2f}" y="{top + plot_h + 26}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">{x_val}</text>')

    elements.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>')
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>')

    for name, points, color in series:
        poly = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        elements.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in points:
            elements.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4.5" fill="#ffffff" stroke="{color}" stroke-width="2"/>')

    legend_x = left + plot_w - 210
    legend_y = top + 18
    elements.append(f'<rect x="{legend_x - 16}" y="{legend_y - 18}" width="216" height="{28 * len(series) + 12}" rx="6" fill="#ffffff" stroke="#d1d5db"/>')
    for idx, (name, _, color) in enumerate(series):
        y = legend_y + idx * 28
        elements.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        elements.append(f'<text x="{legend_x + 38}" y="{y + 5}" font-family="Arial, sans-serif" font-size="14" fill="#111827">{html.escape(name)}</text>')

    elements.append(f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#111827">{html.escape(x_label)}</text>')
    elements.append(f'<text transform="translate(24 {top + plot_h / 2}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#111827">{html.escape(y_label)}</text>')
    elements.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def save_png_line_plot(path, title, x_label, y_label, series):
    width = 900
    height = 520
    left = 86
    right = 28
    top = 58
    bottom = 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max, y_min, y_max = prepare_scale(series)

    def sx(x):
        return left + plot_w / 2 if x_max == x_min else left + (x - x_min) * plot_w / (x_max - x_min)

    def sy(y):
        return top + plot_h / 2 if y_max == y_min else top + (y_max - y) * plot_h / (y_max - y_min)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        small = ImageFont.truetype("arial.ttf", 12)
        title_font = ImageFont.truetype("arialbd.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
        title_font = ImageFont.load_default()

    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 18), title, fill="#1f2937", font=title_font)

    for i in range(6):
        y_val = y_min + (y_max - y_min) * i / 5
        y_pos = sy(y_val)
        draw.line((left, y_pos, left + plot_w, y_pos), fill="#e5e7eb", width=1)
        label = f"{y_val:.3f}"
        label_box = draw.textbbox((0, 0), label, font=small)
        draw.text((left - 12 - (label_box[2] - label_box[0]), y_pos - 7), label, fill="#4b5563", font=small)

    for x_val in range(int(x_min), int(x_max) + 1):
        x_pos = sx(x_val)
        draw.line((x_pos, top, x_pos, top + plot_h), fill="#f3f4f6", width=1)
        label = str(x_val)
        label_box = draw.textbbox((0, 0), label, font=small)
        draw.text((x_pos - (label_box[2] - label_box[0]) / 2, top + plot_h + 18), label, fill="#4b5563", font=small)

    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#111827", width=2)
    draw.line((left, top, left, top + plot_h), fill="#111827", width=2)

    for name, points, color in series:
        xy = [(sx(x), sy(y)) for x, y in points]
        if len(xy) > 1:
            draw.line(xy, fill=color, width=3, joint="curve")
        for x, y in xy:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="white", outline=color, width=2)

    legend_x = left + plot_w - 210
    legend_y = top + 6
    draw.rounded_rectangle(
        (legend_x - 16, legend_y - 6, legend_x + 200, legend_y + 28 * len(series) + 10),
        radius=6,
        fill="white",
        outline="#d1d5db",
    )
    for idx, (name, _, color) in enumerate(series):
        y = legend_y + idx * 28 + 8
        draw.line((legend_x, y, legend_x + 28, y), fill=color, width=3)
        draw.text((legend_x + 38, y - 8), name, fill="#111827", font=font)

    x_box = draw.textbbox((0, 0), x_label, font=font)
    draw.text((left + plot_w / 2 - (x_box[2] - x_box[0]) / 2, height - 44), x_label, fill="#111827", font=font)
    draw.text((18, top + plot_h / 2 - 30), y_label, fill="#111827", font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def plot_epoch_curves(results_dir, epoch_rows, model_name, horizon):
    rows = [r for r in epoch_rows if r["model"] == model_name and r["horizon"] == horizon]
    if not rows:
        return
    epochs = [r["epoch"] for r in rows]
    train_raw = [r["train_loss"] for r in rows]
    val_raw = [r["val_loss"] for r in rows]
    train_best = best_so_far_values(train_raw)
    val_best = best_so_far_values(val_raw)
    raw_series = [
        ("Observed Training Loss", list(zip(epochs, train_raw)), "#2563eb"),
        ("Observed Validation Loss", list(zip(epochs, val_raw)), "#059669"),
    ]
    checkpoint_series = [
        ("Best Training Loss So Far", list(zip(epochs, train_best)), "#2563eb"),
        ("Best Validation Loss So Far", list(zip(epochs, val_best)), "#059669"),
    ]
    safe_name = safe_file_name(model_name)
    short_name = short_file_name(model_name).replace("timebase_informer", "timebase_informer")
    raw_title = f"{model_name} Observed Training and Validation Loss | {horizon}h"
    checkpoint_title = f"{model_name} Best-So-Far Loss by Epoch | {horizon}h"
    save_svg_line_plot(results_dir / f"raw_loss_curve_{safe_name}_{horizon}h.svg", raw_title, "Epoch", "Loss", raw_series)
    save_png_line_plot(results_dir / f"raw_loss_curve_{safe_name}_{horizon}h.png", raw_title, "Epoch", "Loss", raw_series)
    save_svg_line_plot(results_dir / f"loss_curve_{safe_name}_{horizon}h.svg", checkpoint_title, "Epoch", "Loss", checkpoint_series)
    save_png_line_plot(results_dir / f"loss_curve_{safe_name}_{horizon}h.png", checkpoint_title, "Epoch", "Loss", checkpoint_series)
    save_png_line_plot(results_dir / f"{short_name}_{horizon}h_loss_curve.png", checkpoint_title, "Epoch", "Loss", checkpoint_series)


def plot_metric_bars(results_dir, metrics_rows, metric):
    horizons = sorted({r["horizon"] for r in metrics_rows})
    models = ["TimeBase", "TimeBase+Informer"]
    width = 900
    height = 520
    left = 86
    right = 28
    top = 58
    bottom = 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    vals = [r[metric] for r in metrics_rows]
    y_max = max(vals) * 1.15 if vals else 1.0
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        title_font = ImageFont.truetype("arialbd.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    title = f"Test {metric.upper()} comparison"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 18), title, fill="#1f2937", font=title_font)
    for i in range(6):
        y_val = y_max * i / 5
        y = top + plot_h - (y_val / y_max) * plot_h
        draw.line((left, y, left + plot_w, y), fill="#e5e7eb", width=1)
        draw.text((left - 70, y - 7), f"{y_val:.3f}", fill="#4b5563", font=font)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#111827", width=2)
    draw.line((left, top, left, top + plot_h), fill="#111827", width=2)

    colors = {"TimeBase": "#2563eb", "TimeBase+Informer": "#dc2626"}
    group_w = plot_w / len(horizons)
    bar_w = group_w * 0.28
    for i, horizon in enumerate(horizons):
        center = left + group_w * (i + 0.5)
        for j, model in enumerate(models):
            value = next(r[metric] for r in metrics_rows if r["model"] == model and r["horizon"] == horizon)
            x0 = center + (j - 0.5) * bar_w * 1.25
            x1 = x0 + bar_w
            y0 = top + plot_h - (value / y_max) * plot_h
            draw.rectangle((x0, y0, x1, top + plot_h), fill=colors[model])
        label = f"{horizon}h"
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((center - (box[2] - box[0]) / 2, top + plot_h + 18), label, fill="#111827", font=font)

    legend_x = left + plot_w - 220
    legend_y = top + 8
    for idx, model in enumerate(models):
        y = legend_y + idx * 28
        draw.rectangle((legend_x, y, legend_x + 18, y + 18), fill=colors[model])
        draw.text((legend_x + 28, y), model, fill="#111827", font=font)

    image.save(results_dir / f"comparison_{metric}.png")


def build_comparison(metrics_rows):
    rows = []
    for h in sorted({r["horizon"] for r in metrics_rows}):
        base = next(r for r in metrics_rows if r["model"] == "TimeBase" and r["horizon"] == h)
        hybrid = next(r for r in metrics_rows if r["model"] == "TimeBase+Informer" and r["horizon"] == h)
        rows.append(
            {
                "horizon": h,
                "timebase_mae": base["mae"],
                "hybrid_mae": hybrid["mae"],
                "mae_improvement": base["mae"] - hybrid["mae"],
                "mae_improvement_pct": 100.0 * (base["mae"] - hybrid["mae"]) / max(base["mae"], 1e-12),
                "timebase_mse": base["mse"],
                "hybrid_mse": hybrid["mse"],
                "mse_improvement": base["mse"] - hybrid["mse"],
                "mse_improvement_pct": 100.0 * (base["mse"] - hybrid["mse"]) / max(base["mse"], 1e-12),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Train TimeBase and TimeBase+Informer on Load House 1.")
    parser.add_argument("--csv-path", default=Config.csv_path)
    parser.add_argument("--results-dir", default=Config.results_dir)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--seq-len", type=int, default=Config.seq_len)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(Config.horizons))
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--timebase-lr", type=float, default=Config.timebase_lr)
    parser.add_argument("--informer-lr", type=float, default=Config.informer_lr)
    parser.add_argument("--hybrid-timebase-lr", type=float, default=Config.hybrid_timebase_lr)
    parser.add_argument("--gate-lr", type=float, default=Config.gate_lr)
    parser.add_argument("--orth-lambda", type=float, default=Config.orth_lambda)
    parser.add_argument("--gate-init", type=float, default=Config.gate_init)
    parser.add_argument("--d-model", type=int, default=Config.d_model)
    parser.add_argument("--d-ff", type=int, default=Config.d_ff)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--encoder-layers", type=int, default=Config.encoder_layers)
    parser.add_argument("--conv-stride", type=int, default=Config.conv_stride)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        csv_path=args.csv_path,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        horizons=tuple(args.horizons),
        seed=args.seed,
        timebase_lr=args.timebase_lr,
        informer_lr=args.informer_lr,
        hybrid_timebase_lr=args.hybrid_timebase_lr,
        gate_lr=args.gate_lr,
        orth_lambda=args.orth_lambda,
        gate_init=args.gate_init,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
        encoder_layers=args.encoder_layers,
        conv_stride=args.conv_stride,
    )

    set_seed(cfg.seed)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    times, values = load_hourly_consumption(cfg.csv_path)
    train, val, test, mean, std, train_end, val_end = split_and_scale(values, cfg.seq_len)

    metadata = {
        "config": asdict(cfg),
        "device": str(device),
        "hourly_points": len(values),
        "train_points": int(train_end),
        "val_points": int(val_end - train_end),
        "test_points": int(len(values) - val_end),
        "train_range": [times[0].isoformat(sep=" "), times[train_end - 1].isoformat(sep=" ")],
        "val_range": [times[train_end].isoformat(sep=" "), times[val_end - 1].isoformat(sep=" ")],
        "test_range": [times[val_end].isoformat(sep=" "), times[-1].isoformat(sep=" ")],
        "scaler_mean": float(mean[0, 0]),
        "scaler_std": float(std[0, 0]),
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

    epoch_rows = []
    metrics_rows = []
    all_outputs = {}

    for horizon in cfg.horizons:
        loaders = (
            make_loader(train, cfg.seq_len, horizon, cfg.batch_size, True, cfg.num_workers),
            make_loader(val, cfg.seq_len, horizon, cfg.batch_size, False, cfg.num_workers),
            make_loader(test, cfg.seq_len, horizon, cfg.batch_size, False, cfg.num_workers),
        )
        base = BaseTimeBase(cfg, horizon)
        base, rows, metrics, test_out = train_one_model(base, "TimeBase", horizon, loaders, cfg, device, mean, std)
        torch.save(base.state_dict(), results_dir / f"timebase_{horizon}h.pt")
        epoch_rows.extend(rows)
        metrics_rows.append(metrics)
        all_outputs[("TimeBase", horizon)] = test_out

        hybrid = TimeBaseInformer(cfg, horizon)
        hybrid, rows, metrics, test_out = train_one_model(
            hybrid,
            "TimeBase+Informer",
            horizon,
            loaders,
            cfg,
            device,
            mean,
            std,
            init_state=base.timebase.state_dict(),
        )
        torch.save(hybrid.state_dict(), results_dir / f"timebase_informer_{horizon}h.pt")
        epoch_rows.extend(rows)
        metrics_rows.append(metrics)
        all_outputs[("TimeBase+Informer", horizon)] = test_out

        for model_name in ["TimeBase", "TimeBase+Informer"]:
            out = all_outputs[(model_name, horizon)]
            safe_name = model_name.lower().replace("+", "_plus_")
            np.savez_compressed(
                results_dir / f"predictions_{safe_name}_{horizon}h.npz",
                preds=out["preds"],
                trues=out["trues"],
            )
            save_prediction_csv(
                results_dir / f"predictions_{safe_name}_{horizon}h_sample.csv",
                out["preds"],
                out["trues"],
                horizon,
                cfg.save_prediction_windows,
            )

    write_csv(results_dir / "epoch_log.csv", epoch_rows)
    write_csv(results_dir / "metrics.csv", metrics_rows)
    comparison_rows = build_comparison(metrics_rows)
    write_csv(results_dir / "comparison.csv", comparison_rows)

    for horizon in cfg.horizons:
        plot_epoch_curves(results_dir, epoch_rows, "TimeBase", horizon)
        plot_epoch_curves(results_dir, epoch_rows, "TimeBase+Informer", horizon)
    plot_metric_bars(results_dir, metrics_rows, "mae")
    plot_metric_bars(results_dir, metrics_rows, "mse")

    print("\nFinal test metrics")
    for row in metrics_rows:
        print(
            f"{row['model']:18s} h={row['horizon']:2d} "
            f"MAE={row['mae']:.6f} MSE={row['mse']:.6f} RMSE={row['rmse']:.6f} "
            f"best_val_mse={row['best_val_pred_mse']:.6f} params={row['params']}"
        )

    print("\nImprovement summary")
    for row in comparison_rows:
        print(
            f"h={row['horizon']:2d} "
            f"MAE improvement={row['mae_improvement']:.6f} ({row['mae_improvement_pct']:.2f}%) | "
            f"MSE improvement={row['mse_improvement']:.6f} ({row['mse_improvement_pct']:.2f}%)"
        )
    print(f"\nSaved all artifacts to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
