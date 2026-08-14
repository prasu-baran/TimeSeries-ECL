"""
This file trains the TimeBase baseline and a TimeBase+PMFormer hybrid for
 forecasting on Load House 1.csv. The comments below mark the
paper or project reference behind each major block so the file can be read from
data ingestion through final comparison outputs.

Primary references:
1. TimeBase - "TimeBase: The Power of Minimalism in Efficient Long-term
   Time Series Forecasting" (ICML/PMLR 2025): 24-hour segmentation, learned
   temporal bases, segment-level forecasting, and orthogonality regularization.
2. PMFormer - "PMformer: A novel informer-based model for accurate long-term
   time series prediction" (Information Sciences, 2025): patch-level sequence
   encoding used here as the residual enhancement branch.
3. Project README / TimeBase_Extension report: Load House 1 data protocol,
   720-hour lookback window, 1/8/16/24-hour horizons, metric reporting, and
   output artifact generation.
"""

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
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Project experiment reference: centralizes the dataset, forecasting horizons,
# model sizes, learning rates, and output settings used across all runs.
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
    pmformer_lr: float = 0.0001
    hybrid_timebase_lr: float = 0.0001
    gate_init: float = -3.5
    weight_decay: float = 0.0001
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.10
    encoder_layers: int = 2
    patch_len: int = 24
    num_workers: int = 0
    grad_clip: float = 1.0
    save_prediction_windows: int = 256


# Reproducibility reference: fixes Python, NumPy, and PyTorch random seeds for
# comparable baseline-vs-hybrid experiments.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# Dataset reference: converts the separate Date and Hour columns in
# Load House 1.csv into a single timestamp key.
def parse_dt(date_text, hour_text):
    return datetime.fromisoformat(f"{date_text}T{hour_text}")


# Project data reference: aggregates 15-minute consumption readings into the
# hourly series used by both TimeBase and TimeBase+PMFormer.
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


# Evaluation protocol reference: keeps chronological order, fits scaling only
# on the train span, and overlaps validation/test by seq_len for valid windows.
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


# Forecasting-window reference: creates sliding 720-hour input windows and the
# requested prediction horizon for supervised time-series training.
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


# Loader reference: wraps each chronological split in a PyTorch DataLoader while
# preserving deterministic validation/test ordering.
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


# TimeBase paper reference: implements the minimalist segmented basis extractor
# and segment projection head that form the baseline forecasting core.
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


# TimeBase paper reference: penalizes correlation between learned bases so the
# basis set captures diverse temporal patterns instead of duplicate components.
def orth_loss(basis):
    b, c, r, p = basis.shape
    z = basis.reshape(b * c, r, p)
    z = z / (torch.norm(z, dim=-1, keepdim=True) + 1e-8)
    gram = torch.matmul(z, z.transpose(1, 2))
    eye = torch.eye(r, device=basis.device).view(1, r, r)
    return ((gram * (1.0 - eye)) ** 2).mean()


# Baseline reference: exposes the plain TimeBase model for direct comparison
# against the residual PMFormer extension under the same data protocol.
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


# PMFormer paper reference: combines temporal self-attention, local convolution,
# and a feed-forward block to encode patch-level long-range dependencies.
class PMFormerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm1(x)
        attn_out, _ = self.temporal_attn(z, z, z, need_weights=False)
        x = x + self.dropout(attn_out)

        z = self.norm2(x).transpose(1, 2)
        z = self.conv(z).transpose(1, 2)
        x = x + self.dropout(z)

        x = x + self.dropout(self.ff(self.norm3(x)))
        return x


# PMFormer extension reference: converts the 720-hour input into 24-hour patches
# and predicts a residual correction to the TimeBase forecast.
class PMFormerPatchEncoderResidual(nn.Module):
    def __init__(self, cfg, pred_len):
        super().__init__()
        self.seq_len = cfg.seq_len
        self.pred_len = pred_len
        self.patch_len = cfg.patch_len
        self.n_patches = math.ceil(cfg.seq_len / cfg.patch_len)
        self.padded_len = self.n_patches * cfg.patch_len

        self.patch_proj = nn.Linear(cfg.patch_len, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, cfg.d_model))
        self.blocks = nn.ModuleList(
            [
                PMFormerEncoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                for _ in range(cfg.encoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.n_patches * cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, pred_len),
        )
        nn.init.zeros_(self.head[-1].bias)
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.05)

    def forward(self, x):
        z = x[:, :, 0]
        if self.padded_len > self.seq_len:
            pad = z[:, -1:].repeat(1, self.padded_len - self.seq_len)
            z = torch.cat([z, pad], dim=1)
        z = z.reshape(z.size(0), self.n_patches, self.patch_len)
        z = self.patch_proj(z) + self.pos
        for block in self.blocks:
            z = block(z)
        z = self.norm(z)
        residual = self.head(z).unsqueeze(-1)
        return residual


# New Architecture reference: adds a gated PMFormer residual branch on top of
# the TimeBase core so the extension can help without overpowering the baseline.
class TimeBasePMFormer(nn.Module):
    def __init__(self, cfg, pred_len):
        super().__init__()
        self.timebase = TimeBaseCore(
            seq_len=cfg.seq_len,
            pred_len=pred_len,
            seg_len=cfg.tb_seg_len,
            basis_num=cfg.tb_basis_num,
        )
        self.pmformer_encoder = PMFormerPatchEncoderResidual(cfg, pred_len)
        self.gate = nn.Parameter(torch.tensor(float(cfg.gate_init)))

    def forward(self, x):
        base_pred, basis = self.timebase(x)
        residual = self.pmformer_encoder(x)
        gate = torch.sigmoid(self.gate)
        return base_pred + gate * residual, basis, gate


# Metric reference: returns scaled predictions to the original kW range before
# computing MAE, MSE, and RMSE for report-ready results.
def inverse_scale(arr, mean, std):
    return arr * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


# Evaluation reference: measures each trained model on validation/test loaders
# without gradient updates and stores predictions for later plots/tables.
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


# Training protocol reference: applies the same epoch loop to TimeBase and the
# hybrid model, adding orthogonality loss for TimeBase-style basis learning.
def train_one_model(model, model_name, horizon, loaders, cfg, device, mean, std, init_state=None):
    train_loader, val_loader, test_loader = loaders
    model = model.to(device)
    if init_state is not None and hasattr(model, "timebase"):
        model.timebase.load_state_dict(copy.deepcopy(init_state))

    if model_name == "TimeBase+PMFormer":
        optimizer = torch.optim.AdamW(
            [
                {"params": model.timebase.parameters(), "lr": cfg.hybrid_timebase_lr},
                {"params": model.pmformer_encoder.parameters(), "lr": cfg.pmformer_lr},
                {"params": [model.gate], "lr": cfg.pmformer_lr},
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


# Utility reference: reports trainable capacity so accuracy improvements can be
# compared with model-size cost.
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Artifact reference: writes tabular experiment logs used by the generated
# report, metrics table, and comparison summary.
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


# Artifact reference: exports representative prediction-vs-actual windows for
# manual checking without opening the binary NPZ files.
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


# Reporting utility reference: converts model names and metric labels into safe
# artifact filenames for plots and CSV exports.
def safe_file_name(text):
    return text.replace("+", "_plus_").replace(" ", "_").replace("/", "_")


# Visualization reference: tracks the best validation/test curve value reached
# so far, which makes epoch plots easier to interpret.
def best_so_far_values(values):
    best = float("inf")
    out = []
    for value in values:
        best = min(best, value)
        out.append(best)
    return out


# Visualization reference: creates dependency-light SVG plots for epoch curves
# and metric summaries when the experiment artifacts are generated.
def save_svg_line_plot(path, title, x_label, y_label, series):
    width = 900
    height = 520
    left = 86
    right = 28
    top = 58
    bottom = 78
    plot_w = width - left - right
    plot_h = height - top - bottom

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
    y_min = max(0.0, y_min - pad)
    y_max = y_max + pad

    def sx(x):
        if x_max == x_min:
            return left + plot_w / 2
        return left + (x - x_min) * plot_w / (x_max - x_min)

    def sy(y):
        if y_max == y_min:
            return top + plot_h / 2
        return top + (y_max - y) * plot_h / (y_max - y_min)

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

    legend_x = left + plot_w - 190
    legend_y = top + 18
    elements.append(f'<rect x="{legend_x - 16}" y="{legend_y - 18}" width="196" height="{28 * len(series) + 12}" rx="6" fill="#ffffff" stroke="#d1d5db"/>')
    for idx, (name, _, color) in enumerate(series):
        y = legend_y + idx * 28
        elements.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        elements.append(f'<text x="{legend_x + 38}" y="{y + 5}" font-family="Arial, sans-serif" font-size="14" fill="#111827">{html.escape(name)}</text>')

    elements.append(f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#111827">{html.escape(x_label)}</text>')
    elements.append(f'<text transform="translate(24 {top + plot_h / 2}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#111827">{html.escape(y_label)}</text>')
    elements.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


# Reporting reference: plots training, validation, and test progress for one
# model-horizon pair so convergence can be inspected visually.
def plot_epoch_curves(results_dir, epoch_rows, model_name, horizon):
    rows = [r for r in epoch_rows if r["model"] == model_name and r["horizon"] == horizon]
    if not rows:
        return
    epochs = [r["epoch"] for r in rows]
    train_raw = [r["train_loss"] for r in rows]
    val_raw = [r["val_loss"] for r in rows]
    train_best = best_so_far_values(train_raw)
    val_best = best_so_far_values(val_raw)
    safe_name = safe_file_name(model_name)
    save_svg_line_plot(
        results_dir / f"raw_loss_curve_{safe_name}_{horizon}h.svg",
        f"{model_name} Observed Training and Validation Loss | {horizon}h",
        "Epoch",
        "Loss",
        [
            ("Observed Training Loss", list(zip(epochs, train_raw)), "#2563eb"),
            ("Observed Validation Loss", list(zip(epochs, val_raw)), "#059669"),
        ],
    )
    save_svg_line_plot(
        results_dir / f"loss_curve_{safe_name}_{horizon}h.svg",
        f"{model_name} Best-So-Far Loss by Epoch | {horizon}h",
        "Epoch",
        "Loss",
        [
            ("Best Training Loss So Far", list(zip(epochs, train_best)), "#2563eb"),
            ("Best Validation Loss So Far", list(zip(epochs, val_best)), "#059669"),
        ],
    )
    if plt is None:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_raw, marker="o", label="Observed Training Loss")
    plt.plot(epochs, val_raw, marker="o", label="Observed Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} observed training and validation loss | {horizon}h")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f"raw_loss_curve_{safe_name}_{horizon}h.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_best, marker="o", label="Best Training Loss So Far")
    plt.plot(epochs, val_best, marker="o", label="Best Validation Loss So Far")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} best-so-far loss by epoch | {horizon}h")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f"loss_curve_{safe_name}_{horizon}h.png", dpi=160)
    plt.close()


# Reporting reference: compares final test MAE/MSE values across the baseline
# and PMFormer hybrid for all configured horizons.
def plot_metric_bars(results_dir, metrics_rows, metric):
    if plt is None:
        return
    horizons = sorted({r["horizon"] for r in metrics_rows})
    models = ["TimeBase", "TimeBase+PMFormer"]
    x = np.arange(len(horizons))
    width = 0.36
    plt.figure(figsize=(8, 4.5))
    for j, model in enumerate(models):
        vals = []
        for h in horizons:
            vals.append(next(r[metric] for r in metrics_rows if r["model"] == model and r["horizon"] == h))
        plt.bar(x + (j - 0.5) * width, vals, width=width, label=model)
    plt.xticks(x, [f"{h}h" for h in horizons])
    plt.ylabel(metric.upper())
    plt.title(f"Test {metric.upper()} comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f"comparison_{metric}.png", dpi=160)
    plt.close()


# Reporting reference: builds a compact baseline-vs-hybrid improvement table for
# each forecast horizon.
def build_comparison(metrics_rows):
    rows = []
    for h in sorted({r["horizon"] for r in metrics_rows}):
        base = next(r for r in metrics_rows if r["model"] == "TimeBase" and r["horizon"] == h)
        hybrid = next(r for r in metrics_rows if r["model"] == "TimeBase+PMFormer" and r["horizon"] == h)
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


# CLI reference: lets the same documented experiment be repeated with adjusted
# paths, horizons, learning rates, and gate settings.
def parse_args():
    parser = argparse.ArgumentParser(description="Train TimeBase and TimeBase+PMFormer on Load House 1.")
    parser.add_argument("--csv-path", default=Config.csv_path)
    parser.add_argument("--results-dir", default=Config.results_dir)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--seq-len", type=int, default=Config.seq_len)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(Config.horizons))
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--pmformer-lr", type=float, default=Config.pmformer_lr)
    parser.add_argument("--hybrid-timebase-lr", type=float, default=Config.hybrid_timebase_lr)
    parser.add_argument("--timebase-lr", type=float, default=Config.timebase_lr)
    parser.add_argument("--orth-lambda", type=float, default=Config.orth_lambda)
    parser.add_argument("--gate-init", type=float, default=Config.gate_init)
    return parser.parse_args()


# End-to-end protocol reference: loads data, trains TimeBase first, initializes
# the hybrid from the baseline core, then saves metrics, predictions, and plots.
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
        pmformer_lr=args.pmformer_lr,
        hybrid_timebase_lr=args.hybrid_timebase_lr,
        timebase_lr=args.timebase_lr,
        orth_lambda=args.orth_lambda,
        gate_init=args.gate_init,
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
        base, rows, metrics, test_out = train_one_model(
            base, "TimeBase", horizon, loaders, cfg, device, mean, std
        )
        torch.save(base.state_dict(), results_dir / f"timebase_{horizon}h.pt")
        epoch_rows.extend(rows)
        metrics_rows.append(metrics)
        all_outputs[("TimeBase", horizon)] = test_out

        hybrid = TimeBasePMFormer(cfg, horizon)
        hybrid, rows, metrics, test_out = train_one_model(
            hybrid,
            "TimeBase+PMFormer",
            horizon,
            loaders,
            cfg,
            device,
            mean,
            std,
            init_state=base.timebase.state_dict(),
        )
        torch.save(hybrid.state_dict(), results_dir / f"timebase_pmformer_{horizon}h.pt")
        epoch_rows.extend(rows)
        metrics_rows.append(metrics)
        all_outputs[("TimeBase+PMFormer", horizon)] = test_out

        for model_name in ["TimeBase", "TimeBase+PMFormer"]:
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
        plot_epoch_curves(results_dir, epoch_rows, "TimeBase+PMFormer", horizon)
    plot_metric_bars(results_dir, metrics_rows, "mae")
    plot_metric_bars(results_dir, metrics_rows, "mse")

    print("\nFinal test metrics")
    for row in metrics_rows:
        print(
            f"{row['model']:18s} h={row['horizon']:2d} "
            f"MAE={row['mae']:.6f} MSE={row['mse']:.6f} RMSE={row['rmse']:.6f} "
            f"params={row['params']}"
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
